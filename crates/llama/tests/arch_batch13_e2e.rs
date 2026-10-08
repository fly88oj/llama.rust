//! arch_batch13_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! 2026-09: **the P0 standard-attention queue** — llama4 / qwen3vl /
//! qwen3vlmoe / qwen2vl / glm4 / glm-dsa / chatglm / mistral3 / cohere2 /
//! minicpm3 / exaone4 / bitnet / dbrx / ernie4-5 (dense) + the
//! NEMOTRON_H_MOE loader arm (llama.cpp bd4f514db1, the P0 rows of
//! parity/AUDIT_models.md — real public GGUFs of these archs hard-refused at
//! load before this batch).
//!
//! Same protocol as batches 1-12 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of most of these archs exists, so each is
//! verified on a *synthetic* file built with the port's byte-exact GGUF
//! writer — `tokenizer.*` KV copied verbatim from the llama SPM vocab
//! fixture, the arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32. The cells drive llama-cli itself
//! (batch-6+ protocol); these in-port tests pin the loaders + builders and
//! smoke both FA modes.
//!
//! Vision-LM trunks (qwen2vl / qwen3vl / qwen3vlmoe) are text-side graphs
//! here — the vision side lives in mtmd/clip. The synthetic files carry the
//! mrope/IMRoPE `rope.dimension_sections` the loaders require, and the text
//! batches feed the same position id in all 4 blocks (what the reference
//! does for text-only input); qwen3vl's deepstack views read the zero-padded
//! `n_deepstack_layers` blocks of the token-embedding rows (build_inp_embd's
//! `ggml_pad`, llama-graph.cpp:2408-2411) exactly like the reference's
//! text-only batches.
//!
//! glm-dsa is the batch's DSA member (deepseek32's lightning indexer + the
//! shared-indexer layers of GLM-5.2) plus the `graph_mtp` draft head — the
//! MTP graph is verified in-port here (the parity script covers the trunk).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaSwaType;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch13";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    /// plain RMS decoders with the per-arch FFN/attention quirks
    Cohere2,
    Chatglm,
    Bitnet,
    Dbrx,
    Ernie45,
    Mistral3,
    Minicpm3,
    Glm4,
    Exaone4,
    Llama4,
    Qwen2Vl,
    Qwen3Vl,
    Qwen3VlMoe,
    GlmDsa,
    NemotronHMoe,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    key_length: i64,
    value_length: i64,
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    // ---- attention quirks ----
    /// mrope/IMRoPE sections (None → no rope.dimension_sections key)
    rope_sections: Option<[u32; 4]>,
    /// hparams.n_deepstack_layers (qwen3vl family)
    n_deepstack: u32,
    /// hparams.f_attn_temp_scale (mistral3/llama4; 0 → no key)
    f_attn_temp_scale: f32,
    /// sliding_window + pattern keys (None → no keys)
    swa: Option<(u32, Vec<u32>)>,
    /// mistral3's yarn block (sets n_ctx_orig_yarn for the temp floor)
    mistral_yarn: bool,
    /// bitnet's weight-scale tensors
    bitnet_scales: bool,
    // ---- MoE ----
    n_ff_exp: i64,
    n_ff_shexp: i64,
    n_expert_shared: i64,
    /// llama4: hparams.n_moe_layer_step (0 → no interleave key)
    n_moe_layer_step: u32,
    n_layer_dense_lead: u32,
    // ---- MLA (minicpm3 / glm-dsa) ----
    q_lora_rank: i64,
    kv_lora_rank: i64,
    key_length_mla: i64,
    value_length_mla: i64,
    // ---- glm-dsa indexer ----
    indexer_n_head: i64,
    indexer_head_size: i64,
    indexer_top_k: i64,
    /// None → no attention.indexer_types key (the pre-5.2 all-full BC when
    /// n_ctx_train < 1M)
    indexer_types: Option<Vec<u32>>,
    /// glm-dsa: write the n_ctx_train above 1M so the default pattern loads
    ctx_train_5_2: bool,
    // ---- nemotron-h-moe ----
    n_ff_arr: Option<Vec<i64>>,
    head_kv_arr: Option<Vec<i64>>,
    d_conv: i64,
    d_inner: i64,
    d_state: i64,
    dt_rank: i64,
    n_group: i64,
    moe_latent: i64,
}

impl SynthSpec {
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut s = self.clone();
        f(&mut s);
        s
    }
}

fn base(arch: &'static str, family: Family) -> SynthSpec {
    SynthSpec {
        arch,
        suffix: "",
        family,
        n_layer: 4,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        key_length: 16,
        value_length: 16,
        rope_dim: 16,
        n_ff: 48,
        n_ctx: 256,
        write_output: false,
        rope_sections: None,
        n_deepstack: 0,
        f_attn_temp_scale: 0.0,
        swa: None,
        mistral_yarn: false,
        bitnet_scales: false,
        n_ff_exp: 24,
        n_ff_shexp: 24,
        n_expert_shared: 0,
        n_moe_layer_step: 0,
        n_layer_dense_lead: 0,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        key_length_mla: 0,
        value_length_mla: 0,
        indexer_n_head: 0,
        indexer_head_size: 0,
        indexer_top_k: 0,
        indexer_types: None,
        ctx_train_5_2: false,
        n_ff_arr: None,
        head_kv_arr: None,
        d_conv: 0,
        d_inner: 0,
        d_state: 0,
        dt_rank: 0,
        n_group: 0,
        moe_latent: 0,
    }
}

/// cohere2 — LLM_NORM, the every-4th pattern with rope ONLY on the SWA
/// layers, logit_scale on the tied head
fn spec_cohere2() -> SynthSpec {
    base("cohere2", Family::Cohere2).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
    })
}

fn spec_chatglm() -> SynthSpec {
    base("chatglm", Family::Chatglm)
}

/// bitnet — with the per-tensor `.scale` weights, sub-norms, tok_embd head
fn spec_bitnet() -> SynthSpec {
    base("bitnet", Family::Bitnet).with(|s| {
        s.n_head_kv = 4; // wk {n_embd, n_embd}
        s.bitnet_scales = true;
    })
}

fn spec_dbrx() -> SynthSpec {
    base("dbrx", Family::Dbrx).with(|s| {
        s.write_output = true;
    })
}

fn spec_ernie45() -> SynthSpec {
    base("ernie4_5", Family::Ernie45)
}

fn spec_mistral3() -> SynthSpec {
    base("mistral3", Family::Mistral3)
}

/// mistral3 `-temp` — attention.temperature_scale + the yarn block that
/// supplies n_ctx_orig_yarn (mistral3.cpp:14-19)
fn spec_mistral3_temp() -> SynthSpec {
    spec_mistral3().with(|s| {
        s.suffix = "-temp";
        s.f_attn_temp_scale = 0.5;
        s.mistral_yarn = true;
    })
}

fn spec_minicpm3() -> SynthSpec {
    base("minicpm3", Family::Minicpm3).with(|s| {
        // key_length = qk_nope + qk_rope = 16 + 16
        s.key_length = 32;
        s.q_lora_rank = 16;
        s.kv_lora_rank = 24;
        // the repeated-k_pe K rows are n_head wide (minicpm3.cpp:184-186) —
        // the cache must be MHA-width, i.e. head_count_kv == head_count
        // (exactly what the real MiniCPM3 files declare)
        s.n_head_kv = 4;
    })
}

fn spec_glm4() -> SynthSpec {
    base("glm4", Family::Glm4)
}

/// glm4 `-mrope` — rope.dimension_sections present → ggml_rope_multi + 4
/// position ids per token
fn spec_glm4_mrope() -> SynthSpec {
    spec_glm4().with(|s| {
        s.suffix = "-mrope";
        s.rope_sections = Some([4, 4, 4, 0]);
    })
}

/// exaone4 (the 4-layer plain file: no sliding_window key → graph<false>)
fn spec_exaone4() -> SynthSpec {
    base("exaone4", Family::Exaone4)
}

/// exaone4 `-swa` — the 64-layer 32B-shaped file (the hard swa block of
/// exaone4.cpp:4-12), window 32 so the masks bind
fn spec_exaone4_swa() -> SynthSpec {
    spec_exaone4().with(|s| {
        s.suffix = "-swa";
        s.n_layer = 64;
        // the pattern array covers EVERY layer (get_key_or_arr's array fill —
        // entries beyond the array stay 0 = full attention)
        let mut pattern = Vec::new();
        for i in 0..64 {
            pattern.push(u32::from(i % 4 != 0));
        }
        s.swa = Some((32, pattern));
    })
}

/// llama4 — the CHUNKED default (no sliding_window key at all): temp scale
/// 0.1/8192, every-4th no-rope layers with the post-rope Q/K rms norm, the
/// every-4th sigmoid MoE + shared expert
fn spec_llama4() -> SynthSpec {
    base("llama4", Family::Llama4).with(|s| {
        s.n_moe_layer_step = 4;
        // NO sliding_window key at all — found_swa == None takes the CHUNKED
        // arm (llama4.cpp:8-11; only an explicit 0 flips it to NONE). The
        // pattern is load_swa_pattern(ml, 4)'s scalar default: the
        // full-attention layer sits at il%4==3 (the MoE layer, like the real
        // Scout/Maverick interleave)
    })
}

/// llama4 `-noswa` — sliding_window 0 → swa NONE, always rope, no temp scale
/// (llama4.cpp:8-11)
fn spec_llama4_noswa() -> SynthSpec {
    spec_llama4().with(|s| {
        s.suffix = "-noswa";
        s.swa = Some((0, Vec::new()));
    })
}

fn spec_qwen2vl() -> SynthSpec {
    base("qwen2vl", Family::Qwen2Vl).with(|s| {
        s.rope_sections = Some([4, 4, 4, 0]);
    })
}

fn spec_qwen3vl() -> SynthSpec {
    base("qwen3vl", Family::Qwen3Vl).with(|s| {
        s.rope_sections = Some([4, 4, 4, 0]);
        s.n_deepstack = 2;
    })
}

fn spec_qwen3vlmoe() -> SynthSpec {
    base("qwen3vlmoe", Family::Qwen3VlMoe).with(|s| {
        s.rope_sections = Some([4, 4, 4, 0]);
        s.n_deepstack = 2;
    })
}

/// glm-dsa — the pre-5.2 shape (n_ctx_train < 1M → every layer a full
/// indexer)
fn spec_glm_dsa() -> SynthSpec {
    base("glm-dsa", Family::GlmDsa).with(|s| {
        s.n_embd = 128;
        s.n_head = 4;
        s.n_head_kv = 1;
        s.key_length = 48; // kv_lora 32 + qk_rope 16
        s.value_length = 32; // kv_lora
        s.q_lora_rank = 32;
        s.kv_lora_rank = 32;
        s.key_length_mla = 40; // qk_nope 24 + qk_rope 16
        s.value_length_mla = 20;
        s.n_ff = 48;
        s.indexer_n_head = 2;
        s.indexer_head_size = 64; // a power of two — the Hadamard k_rot
        s.indexer_top_k = 8;
        s.n_layer_dense_lead = 1;
        s.n_expert_shared = 1; // shexp width n_ff_exp * 1
    })
}

/// glm-dsa `-shared` — the GLM-5.2 shape: explicit attention.indexer_types
/// [1,0,0,0] so layers 1-3 reuse layer 0's top-k (glm-dsa.cpp:362-367)
fn spec_glm_dsa_shared() -> SynthSpec {
    spec_glm_dsa().with(|s| {
        s.suffix = "-shared";
        s.ctx_train_5_2 = true;
        s.indexer_types = Some(vec![1, 0, 0, 0]);
    })
}

/// nemotron_h_moe — batch-5's nemotron-h MoE recipe under the real arch
/// name (models.h:1539-1543 — the loader + graph are nemotron-h's; only the
/// arch string differs): mamba2 layers 0/2/5, attention 1/3, MoE FFN 4
fn spec_nemotron_h_moe() -> SynthSpec {
    base("nemotron_h_moe", Family::NemotronHMoe).with(|s| {
        s.n_layer = 6;
        s.n_ff = 96;
        s.n_ff_arr = Some(vec![0, 0, 0, 0, 96, 0]);
        s.head_kv_arr = Some(vec![0, 2, 0, 2, 2, 0]);
        s.d_conv = 4;
        s.d_inner = 128;
        s.d_state = 16;
        s.dt_rank = 8;
        s.n_group = 2;
        s.n_ff_shexp = 24;
        s.moe_latent = 32;
    })
}

/// every file the ignored writer test emits
fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_cohere2(),
        spec_chatglm(),
        spec_bitnet(),
        spec_dbrx(),
        spec_ernie45(),
        spec_mistral3(),
        spec_mistral3_temp(),
        spec_minicpm3(),
        spec_glm4(),
        spec_glm4_mrope(),
        spec_exaone4(),
        spec_exaone4_swa(),
        spec_llama4(),
        spec_llama4_noswa(),
        spec_qwen2vl(),
        spec_qwen3vl(),
        spec_qwen3vlmoe(),
        spec_glm_dsa(),
        spec_glm_dsa_shared(),
        spec_nemotron_h_moe(),
    ]
}

/// the parity cells (batch-13 default set): every file creates a memory in
/// the pinned reference. The "-long" cells ride the same files. EXCEPTION:
/// llama4-noswa's file is written and pinned here (the in-port
/// variants-differ / temp-scale tests drive it) but has NO reference cell —
/// the pinned reference aborts on it (llm_graph_input_attn_temp::set_input's
/// f_attn_temp_scale != 0 assert, llama-graph.cpp:161 — the noswa variant
/// carries no attention.temperature_scale), see arch_batch_parity.sh's
/// batch-13 comment.
fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_cohere2(),
        spec_chatglm(),
        spec_bitnet(),
        spec_dbrx(),
        spec_ernie45(),
        spec_mistral3(),
        spec_mistral3_temp(),
        spec_minicpm3(),
        spec_glm4(),
        spec_glm4_mrope(),
        spec_exaone4(),
        spec_exaone4_swa(),
        spec_llama4(),
        spec_llama4_noswa(),
        spec_qwen2vl(),
        spec_qwen3vl(),
        spec_qwen3vlmoe(),
        spec_glm_dsa(),
        spec_glm_dsa_shared(),
        spec_nemotron_h_moe(),
    ]
}

#[test]
#[ignore = "writes the /tmp/arch-batch13 parity files (ARCH_BATCH13 cells)"]
fn arch_batch13_write_synth() {
    for spec in all_specs() {
        let (n, ck) = build_file(&spec);
        println!("{}: {n} tensors, {ck} bytes", spec.path());
    }
}

// ---------------------------------------------------------------------------
// per-arch tensor tables — the create_tensor calls of each
// src/models/<arch>.cpp load_arch_tensors, in file order
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Norm,
    Bias,
    Proj,
    Router,
    /// nemotron-h's A/dt decay tensors
    Decay,
}

type TensorSpec = (String, Vec<i64>);

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let hd = spec.key_length;
    let n_expert = N_EXPERT;
    let mut t: Vec<(TensorSpec, Role)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, role: Role| t.push(((name, ne), role));

    push(
        "token_embd.weight".into(),
        vec![n_embd, N_VOCAB],
        Role::Proj,
    );

    match spec.family {
        Family::Cohere2 => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            // cohere2.cpp:29 — output ties token_embd (TENSOR_DUPLICATED)
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = n_embd; // q is n_embd wide (cohere2.cpp:37)
                let kv_w = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
            }
        }
        Family::Chatglm => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head;
                let kv_w = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}ffn_up.weight"),
                    vec![n_embd, n_ff * 2],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
            }
        }
        Family::Bitnet => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let kv_w = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_sub_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}ffn_sub_norm.weight"), vec![n_ff], Role::Norm);
                push(
                    format!("{b}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                if spec.bitnet_scales {
                    push(format!("{b}attn_q.scale"), vec![1], Role::Norm);
                    push(format!("{b}attn_k.scale"), vec![1], Role::Norm);
                    push(format!("{b}attn_v.scale"), vec![1], Role::Norm);
                    push(format!("{b}attn_output.scale"), vec![1], Role::Norm);
                    push(format!("{b}ffn_gate.scale"), vec![1], Role::Norm);
                    push(format!("{b}ffn_down.scale"), vec![1], Role::Norm);
                    push(format!("{b}ffn_up.scale"), vec![1], Role::Norm);
                }
            }
        }
        Family::Dbrx => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let gqa = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}attn_qkv.weight"),
                    vec![n_embd, n_embd + 2 * gqa],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("{b}ffn_gate_inp.weight"),
                    vec![n_embd, n_expert],
                    Role::Router,
                );
                push(
                    format!("{b}ffn_gate_exps.weight"),
                    vec![n_embd, n_ff, n_expert],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down_exps.weight"),
                    vec![n_ff, n_embd, n_expert],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_up_exps.weight"),
                    vec![n_embd, n_ff, n_expert],
                    Role::Proj,
                );
            }
        }
        Family::Ernie45 | Family::Mistral3 => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            let mistral = spec.family == Family::Mistral3;
            if mistral {
                // ROPE_FREQS's template has no blk slot ("rope_freqs") — ONE
                // top-level tensor serves every layer's duplicated request
                push(
                    "rope_freqs.weight".into(),
                    vec![spec.rope_dim / 2],
                    Role::Norm,
                );
            }
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head;
                let k_w = if mistral {
                    spec.key_length * spec.n_head_kv
                } else {
                    spec.value_length * spec.n_head_kv
                };
                let v_w = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, k_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, v_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                if mistral {
                    push(format!("{b}attn_output.bias"), vec![n_embd], Role::Bias);
                }
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
            }
        }
        Family::Minicpm3 => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            let qk_rope = spec.rope_dim;
            let qk_nope = spec.key_length - qk_rope;
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}attn_q_a_norm.weight"),
                    vec![spec.q_lora_rank],
                    Role::Norm,
                );
                push(
                    format!("{b}attn_kv_a_norm.weight"),
                    vec![spec.kv_lora_rank],
                    Role::Norm,
                );
                push(
                    format!("{b}attn_q_a.weight"),
                    vec![n_embd, spec.q_lora_rank],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_q_b.weight"),
                    vec![spec.q_lora_rank, spec.key_length * spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_kv_a_mqa.weight"),
                    vec![n_embd, spec.kv_lora_rank + qk_rope],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_kv_b.weight"),
                    vec![
                        spec.kv_lora_rank,
                        spec.n_head * (qk_nope + spec.value_length),
                    ],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output.weight"),
                    vec![spec.n_head * spec.value_length, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                // rope_long/rope_short stay absent (NOT_REQUIRED)
                let _ = i;
            }
        }
        Family::Glm4 | Family::Exaone4 => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            let exaone = spec.family == Family::Exaone4;
            if exaone {
                push(
                    "rope_freqs.weight".into(),
                    vec![spec.rope_dim / 2],
                    Role::Norm,
                );
            }
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head;
                let kv_w = if exaone {
                    spec.key_length * spec.n_head_kv
                } else {
                    spec.value_length * spec.n_head_kv
                };
                if !exaone {
                    push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                }
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                if exaone {
                    push(format!("{b}attn_q_norm.weight"), vec![hd], Role::Norm);
                    push(format!("{b}attn_k_norm.weight"), vec![hd], Role::Norm);
                }
                push(
                    format!("{b}post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                if !exaone {
                    push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                }
                if exaone {
                    push(
                        format!("{b}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                } else {
                    // glm4 — the fused gate|up tensor
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up.weight"),
                        vec![n_embd, n_ff * 2],
                        Role::Proj,
                    );
                }
                push(format!("{b}post_ffw_norm.weight"), vec![n_embd], Role::Norm);
            }
        }
        Family::Llama4 => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            push(
                "rope_freqs.weight".into(),
                vec![spec.rope_dim / 2],
                Role::Norm,
            );
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head;
                let kv_w = spec.key_length * spec.n_head_kv;
                let is_moe = (i as u32 + 1) % spec.n_moe_layer_step == 0;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                if is_moe {
                    let e = spec.n_ff_exp;
                    push(
                        format!("{b}ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("{b}ffn_gate_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_exps.weight"),
                        vec![e, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                    // the shared expert — n_ff_shexp == n_ff_exp (llama4.cpp:84)
                    push(
                        format!("{b}ffn_gate_shexp.weight"),
                        vec![n_embd, e],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_shexp.weight"),
                        vec![e, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_shexp.weight"),
                        vec![n_embd, e],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("{b}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                }
            }
        }
        Family::Qwen2Vl | Family::Qwen3Vl | Family::Qwen3VlMoe => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            let qwen2 = spec.family == Family::Qwen2Vl;
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = if qwen2 { n_embd } else { hd * spec.n_head };
                let kv_w = spec.value_length * spec.n_head_kv;
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                if !qwen2 {
                    push(format!("{b}attn_k_norm.weight"), vec![hd], Role::Norm);
                    push(format!("{b}attn_q_norm.weight"), vec![hd], Role::Norm);
                }
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                if spec.family == Family::Qwen3VlMoe {
                    let e = if spec.n_ff_exp != 0 {
                        spec.n_ff_exp
                    } else {
                        n_ff / N_EXPERT_USED
                    };
                    push(
                        format!("{b}ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("{b}ffn_gate_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_exps.weight"),
                        vec![e, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("{b}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                }
            }
        }
        Family::GlmDsa => {
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            let k_mla = spec.key_length_mla;
            let v_mla = spec.value_length_mla;
            let qk_rope = spec.rope_dim;
            let qk_nope = k_mla - qk_rope;
            let inh = spec.indexer_n_head;
            let ih = spec.indexer_head_size;
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}attn_q_a_norm.weight"),
                    vec![spec.q_lora_rank],
                    Role::Norm,
                );
                push(
                    format!("{b}attn_kv_a_norm.weight"),
                    vec![spec.kv_lora_rank],
                    Role::Norm,
                );
                push(
                    format!("{b}attn_q_a.weight"),
                    vec![n_embd, spec.q_lora_rank],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_q_b.weight"),
                    vec![spec.q_lora_rank, spec.n_head * k_mla],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_kv_a_mqa.weight"),
                    vec![n_embd, spec.kv_lora_rank + qk_rope],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_k_b.weight"),
                    vec![qk_nope, spec.kv_lora_rank, spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_v_b.weight"),
                    vec![spec.kv_lora_rank, v_mla, spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output.weight"),
                    vec![spec.n_head * v_mla, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                // the DSA indexer five (glm-dsa.cpp:144-148)
                push(format!("{b}indexer.k_norm.weight"), vec![ih], Role::Norm);
                push(format!("{b}indexer.k_norm.bias"), vec![ih], Role::Bias);
                push(
                    format!("{b}indexer.proj.weight"),
                    vec![n_embd, inh],
                    Role::Proj,
                );
                push(
                    format!("{b}indexer.attn_k.weight"),
                    vec![n_embd, ih],
                    Role::Proj,
                );
                push(
                    format!("{b}indexer.attn_q_b.weight"),
                    vec![spec.q_lora_rank, inh * ih],
                    Role::Proj,
                );
                if (i as u32) < spec.n_layer_dense_lead {
                    push(
                        format!("{b}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                } else {
                    let e = spec.n_ff_exp;
                    let sh = e * spec.n_expert_shared;
                    push(
                        format!("{b}ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("{b}ffn_gate_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_exps.weight"),
                        vec![e, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_exps.weight"),
                        vec![n_embd, e, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_gate_shexp.weight"),
                        vec![n_embd, sh],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_shexp.weight"),
                        vec![sh, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_shexp.weight"),
                        vec![n_embd, sh],
                        Role::Proj,
                    );
                }
            }
        }
        Family::NemotronHMoe => {
            // batch-5's nemotron-h MoE table, arch nemotron_h_moe
            let d_conv = spec.d_conv;
            let d_inner = spec.d_inner;
            let d_state = spec.d_state;
            let dt_rank = spec.dt_rank;
            let ng = spec.n_group;
            let conv_dim = d_inner + 2 * ng * d_state;
            let d_in_proj = 2 * d_inner + 2 * ng * d_state + dt_rank;
            let moe_n_embd = if spec.moe_latent > 0 {
                spec.moe_latent
            } else {
                n_embd
            };
            let n_ff_exp = spec.n_ff_exp;
            let n_ff_shexp = spec.n_ff_shexp;
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                let head_kv_i = spec
                    .head_kv_arr
                    .as_ref()
                    .map(|a| a[i])
                    .unwrap_or(spec.n_head_kv);
                let n_ff_i = spec.n_ff_arr.as_ref().map(|a| a[i]).unwrap_or(n_ff);
                let is_recr = head_kv_i == 0 && n_ff_i == 0;
                let is_ffn = !is_recr && n_ff_i != 0;
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                if is_recr {
                    push(
                        format!("blk.{i}.ssm_in.weight"),
                        vec![n_embd, d_in_proj],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![d_conv, conv_dim],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ssm_conv1d.bias"),
                        vec![conv_dim],
                        Role::Bias,
                    );
                    push(format!("blk.{i}.ssm_dt.bias"), vec![dt_rank], Role::Bias);
                    push(format!("blk.{i}.ssm_a"), vec![1, dt_rank], Role::Decay);
                    push(format!("blk.{i}.ssm_d"), vec![1, dt_rank], Role::Bias);
                    push(
                        format!("blk.{i}.ssm_norm.weight"),
                        vec![d_inner / ng, ng],
                        Role::Norm,
                    );
                    push(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![d_inner, n_embd],
                        Role::Proj,
                    );
                } else if is_ffn {
                    if n_ff_exp > 0 {
                        push(
                            format!("blk.{i}.ffn_gate_inp.weight"),
                            vec![n_embd, n_expert],
                            Role::Router,
                        );
                        push(
                            format!("blk.{i}.exp_probs_b.bias"),
                            vec![n_expert],
                            Role::Bias,
                        );
                        push(
                            format!("blk.{i}.ffn_latent_down.weight"),
                            vec![n_embd, moe_n_embd],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_latent_up.weight"),
                            vec![moe_n_embd, n_embd],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_down_exps.weight"),
                            vec![n_ff_exp, moe_n_embd, n_expert],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_up_exps.weight"),
                            vec![moe_n_embd, n_ff_exp, n_expert],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_down_shexp.weight"),
                            vec![n_ff_shexp, n_embd],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_up_shexp.weight"),
                            vec![n_embd, n_ff_shexp],
                            Role::Proj,
                        );
                    } else {
                        push(
                            format!("blk.{i}.ffn_down.weight"),
                            vec![n_ff_i, n_embd],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.ffn_up.weight"),
                            vec![n_embd, n_ff_i],
                            Role::Proj,
                        );
                    }
                } else {
                    let n_kv = spec.key_length * head_kv_i;
                    push(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, hd * spec.n_head],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, n_kv],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, n_kv],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.attn_output.weight"),
                        vec![hd * spec.n_head, n_embd],
                        Role::Proj,
                    );
                }
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch9-12_e2e.rs)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn scale_of(role: Role, n_embd: i64) -> f32 {
    match role {
        Role::Norm => 1.0,
        Role::Bias => 0.02,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        Role::Router => 1.0 / (n_embd as f32).sqrt(),
        Role::Decay => 0.02,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn build_file(spec: &SynthSpec) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch13");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = spec.arch;
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!(
        "general.name",
        Value::String(format!("llama-rust-synth-{a}"))
    );
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(spec.n_ctx));
    kv!(
        format!("{a}.embedding_length"),
        Value::U32(spec.n_embd as u32)
    );
    kv!(format!("{a}.block_count"), Value::U32(spec.n_layer as u32));
    if let Some(arr) = &spec.n_ff_arr {
        kv!(
            format!("{a}.feed_forward_length"),
            Value::Array(
                GgufType::Uint32,
                arr.iter().map(|&f| Value::U32(f as u32)).collect()
            )
        );
    } else {
        kv!(
            format!("{a}.feed_forward_length"),
            Value::U32(spec.n_ff as u32)
        );
    }
    if let Some(arr) = &spec.head_kv_arr {
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::Array(
                GgufType::Uint32,
                arr.iter().map(|&f| Value::U32(f as u32)).collect()
            )
        );
    } else {
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::U32(spec.n_head_kv as u32)
        );
    }
    kv!(
        format!("{a}.attention.head_count"),
        Value::U32(spec.n_head as u32)
    );
    kv!(
        format!("{a}.attention.key_length"),
        Value::U32(spec.key_length as u32)
    );
    kv!(
        format!("{a}.attention.value_length"),
        Value::U32(spec.value_length as u32)
    );
    if spec.key_length_mla > 0 {
        kv!(
            format!("{a}.attention.key_length_mla"),
            Value::U32(spec.key_length_mla as u32)
        );
        kv!(
            format!("{a}.attention.value_length_mla"),
            Value::U32(spec.value_length_mla as u32)
        );
    }
    if spec.q_lora_rank > 0 {
        kv!(
            format!("{a}.attention.q_lora_rank"),
            Value::U32(spec.q_lora_rank as u32)
        );
    }
    if spec.kv_lora_rank > 0 {
        kv!(
            format!("{a}.attention.kv_lora_rank"),
            Value::U32(spec.kv_lora_rank as u32)
        );
    }
    // the norm eps: cohere2/dbrx are LLM_NORM archs (the LN eps key),
    // qwen2vl reads no eps at all, everything else the RMS key
    match spec.family {
        Family::Cohere2 | Family::Dbrx => {
            kv!(
                format!("{a}.attention.layer_norm_epsilon"),
                Value::F32(1e-5)
            );
        }
        Family::Qwen2Vl => {}
        Family::NemotronHMoe => {
            // nemotron-h.cpp:18-21 — the LN eps, RMS falls back to it
            kv!(
                format!("{a}.attention.layer_norm_epsilon"),
                Value::F32(1e-5)
            );
            kv!(
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-6)
            );
        }
        _ => {
            kv!(
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-5)
            );
        }
    }
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.rope_dim as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    if let Some(sections) = spec.rope_sections {
        kv!(
            format!("{a}.rope.dimension_sections"),
            Value::Array(
                GgufType::Int32,
                sections.iter().map(|&v| Value::I32(v as i32)).collect()
            )
        );
    }
    if let Some((swa, pattern)) = &spec.swa {
        // swa == 0 with an empty pattern -> the window key only (llama4's
        // swa==0 discriminator; the pattern array would be an unconsumed KV)
        kv!(format!("{a}.attention.sliding_window"), Value::U32(*swa));
        if !pattern.is_empty() {
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(
                    GgufType::Uint32,
                    pattern.iter().map(|&v| Value::U32(v)).collect()
                )
            );
        }
    }
    if spec.f_attn_temp_scale != 0.0 {
        kv!(
            format!("{a}.attention.temperature_scale"),
            Value::F32(spec.f_attn_temp_scale)
        );
    }
    if spec.family == Family::Cohere2 {
        // cohere2.cpp:12 — REQUIRED
        kv!(format!("{a}.logit_scale"), Value::F32(8.0));
    }
    if spec.family == Family::Dbrx {
        // dbrx.cpp:5 — REQUIRED
        kv!(format!("{a}.attention.clamp_kqv"), Value::F32(8.0));
    }
    if spec.mistral_yarn {
        // mistral3.cpp:14-19 — the temp floor borrows n_ctx_orig_yarn; the
        // yarn block is what sets it
        kv!(
            format!("{a}.rope.scaling.type"),
            Value::String("yarn".into())
        );
        kv!(
            format!("{a}.rope.scaling.original_context_length"),
            Value::U32(1024)
        );
        kv!(format!("{a}.rope.scaling.factor"), Value::F32(2.0));
    }
    if spec.n_deepstack > 0 {
        kv!(
            format!("{a}.n_deepstack_layers"),
            Value::U32(spec.n_deepstack)
        );
    }
    // MoE keys
    let needs_experts = matches!(
        spec.family,
        Family::Dbrx | Family::Llama4 | Family::Qwen3VlMoe | Family::GlmDsa | Family::NemotronHMoe
    );
    if needs_experts {
        kv!(
            format!("{a}.expert_count"),
            Value::U32(n_expert(spec) as u32)
        );
        kv!(
            format!("{a}.expert_used_count"),
            Value::U32(N_EXPERT_USED as u32)
        );
    }
    match spec.family {
        Family::Llama4 => {
            // REQUIRED (llama4.cpp:5-6)
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::Array(
                    GgufType::Uint32,
                    (0..spec.n_layer)
                        .map(|_| Value::U32(spec.n_ff_exp as u32))
                        .collect()
                )
            );
            kv!(
                format!("{a}.interleave_moe_layer_step"),
                Value::U32(spec.n_moe_layer_step)
            );
        }
        Family::Qwen3VlMoe => {
            // optional — the fallback is n_ff / n_expert_used
        }
        Family::GlmDsa => {
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(
                format!("{a}.expert_shared_count"),
                Value::U32(spec.n_expert_shared as u32)
            );
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
            // expert_weights_norm absent → false; gating absent → SIGMOID
            kv!(
                format!("{a}.attention.indexer.head_count"),
                Value::U32(spec.indexer_n_head as u32)
            );
            kv!(
                format!("{a}.attention.indexer.key_length"),
                Value::U32(spec.indexer_head_size as u32)
            );
            kv!(
                format!("{a}.attention.indexer.top_k"),
                Value::U32(spec.indexer_top_k as u32)
            );
            if let Some(types) = &spec.indexer_types {
                kv!(
                    format!("{a}.attention.indexer.types"),
                    Value::Array(
                        GgufType::Uint32,
                        types.iter().map(|&v| Value::U32(v)).collect()
                    )
                );
            }
        }
        Family::NemotronHMoe => {
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(
                format!("{a}.expert_shared_feed_forward_length"),
                Value::U32(spec.n_ff_shexp as u32)
            );
            kv!(
                format!("{a}.moe_latent_size"),
                Value::U32(spec.moe_latent as u32)
            );
            // the ssm keys (nemotron-h.cpp:6-10)
            kv!(
                format!("{a}.ssm.conv_kernel"),
                Value::U32(spec.d_conv as u32)
            );
            kv!(
                format!("{a}.ssm.inner_size"),
                Value::U32(spec.d_inner as u32)
            );
            kv!(
                format!("{a}.ssm.state_size"),
                Value::U32(spec.d_state as u32)
            );
            kv!(
                format!("{a}.ssm.time_step_rank"),
                Value::U32(spec.dt_rank as u32)
            );
            kv!(
                format!("{a}.ssm.group_count"),
                Value::U32(spec.n_group as u32)
            );
        }
        _ => {}
    }
    if spec.ctx_train_5_2 {
        // glm-dsa's is_pre_5_2 discriminator (glm-dsa.cpp:60)
        kv!(format!("{a}.context_length"), Value::U32(1048576));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
        let n: i64 = ne.iter().product();
        let scale = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = (0..n).map(|_| rng.next() * scale).collect();
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }
    // write via a temp + rename so concurrent test threads never observe a
    // half-written file
    let path = spec.path();
    let tmp = format!("{path}.tmp{}", std::process::id());
    let f = std::fs::File::create(&tmp).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    std::fs::rename(&tmp, &path).expect("publish synth gguf");
    (table.len(), std::fs::metadata(&path).unwrap().len())
}

fn n_expert(spec: &SynthSpec) -> i64 {
    if spec.family == Family::NemotronHMoe {
        N_EXPERT
    } else {
        N_EXPERT
    }
}

/// the build-once lock — cargo runs the tests on parallel threads and
/// `load_synth` must not race the writer
fn build_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// loading + the pinning checks
// ---------------------------------------------------------------------------

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth gguf");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn load_synth(spec: &SynthSpec) -> LlamaModel {
    let path = spec.path();
    if !std::path::Path::new(&path).exists() {
        let _g = build_lock();
        if !std::path::Path::new(&path).exists() {
            build_file(spec);
        }
    }
    open_model(&path)
}

/// the tensor set pins: every declared tensor consumed (no extras, none
/// missing)
fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let table = tensors_for(spec);
    assert_eq!(
        m.tensors.len(),
        table.len(),
        "{}: consumed tensor count (loaded {} vs table {})",
        spec.arch,
        m.tensors.len(),
        table.len()
    );
    for ((name, ne), _) in &table {
        let id = *m
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("{}: tensor {name} not loaded", spec.arch));
        let got = m.ctx.ne(id);
        let mut want = [1i64; 4];
        for (i, &d) in ne.iter().take(4).enumerate() {
            want[i] = d;
        }
        assert_eq!(got[..], want[..], "{}: {name} shape", spec.arch);
    }
}

/// the hparams pins: the geometry each graph derives its views from
fn pin_hparams(m: &LlamaModel, spec: &SynthSpec) {
    let hp = &m.hparams;
    assert_eq!(
        hp.n_layer() as usize,
        spec.n_layer,
        "{}: n_layer",
        spec.arch
    );
    assert_eq!(hp.n_embd as i64, spec.n_embd, "{}: n_embd", spec.arch);
    match spec.family {
        Family::Cohere2 => {
            assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            assert_eq!(hp.n_swa, 32);
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_swa(il), il % 4 != 0, "cohere2 is_swa({il})");
            }
            assert_eq!(hp.f_logit_scale, 8.0); // written below
        }
        Family::Dbrx => {
            assert_eq!(hp.f_clamp_kqv, 8.0);
            assert_eq!(hp.n_expert as i64, N_EXPERT);
            assert_eq!(hp.swa_type, LlamaSwaType::NONE);
        }
        Family::Chatglm | Family::Bitnet | Family::Ernie45 | Family::Glm4 | Family::Qwen2Vl => {
            assert_eq!(hp.swa_type, LlamaSwaType::NONE);
        }
        Family::Mistral3 => {
            if spec.f_attn_temp_scale != 0.0 {
                assert_eq!(hp.f_attn_temp_scale, 0.5);
                assert_eq!(hp.n_attn_temp_floor_scale, 1024); // n_ctx_orig_yarn
            } else {
                assert_eq!(hp.f_attn_temp_scale, 0.0);
            }
        }
        Family::Minicpm3 => {
            assert_eq!(hp.n_lora_q as i64, spec.q_lora_rank);
            assert_eq!(hp.n_lora_kv as i64, spec.kv_lora_rank);
        }
        Family::Exaone4 => {
            if spec.suffix == "-swa" {
                assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
                assert_eq!(hp.n_swa, 32);
                for il in 0..spec.n_layer {
                    assert_eq!(hp.is_swa(il), il % 4 != 0, "exaone4-swa is_swa({il})");
                }
            } else {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
            }
        }
        Family::Llama4 => {
            if spec.suffix == "-noswa" {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
                assert_eq!(hp.n_no_rope_layer_step, hp.n_layer()); // always rope
                assert_eq!(hp.f_attn_temp_scale, 0.0);
            } else {
                assert_eq!(hp.swa_type, LlamaSwaType::CHUNKED);
                assert_eq!(hp.n_swa, 8192);
                assert_eq!(hp.f_attn_temp_scale, 0.1);
                assert_eq!(hp.f_attn_temp_offset, 1.0);
                assert_eq!(hp.n_attn_temp_floor_scale, 8192);
                // load_swa_pattern(ml, 4)'s scalar default (dense_first =
                // false): the full-attention layer sits at il%4==3
                for il in 0..spec.n_layer {
                    assert_eq!(hp.is_swa(il), il % 4 != 3, "llama4 is_swa({il})");
                }
            }
            assert_eq!(hp.n_ff_exp(3) as i64, spec.n_ff_exp);
            assert_eq!(hp.n_moe_layer_step, 4);
            assert!(hp.use_kq_norm); // n_expert 4 != 128
        }
        Family::Qwen2Vl | Family::Qwen3Vl | Family::Qwen3VlMoe => {
            assert_eq!(hp.rope_sections, [4i32, 4, 4, 0]);
            if spec.family != Family::Qwen2Vl {
                assert_eq!(hp.n_deepstack_layers, 2);
            }
        }
        Family::GlmDsa => {
            assert!(hp.is_mla());
            assert_eq!(hp.n_lora_kv as i64, spec.kv_lora_rank);
            assert_eq!(hp.indexer_n_head as i64, spec.indexer_n_head);
            assert_eq!(hp.indexer_head_size as i64, spec.indexer_head_size);
            assert_eq!(hp.indexer_top_k as i64, spec.indexer_top_k);
            assert_eq!(
                hp.expert_gating_func,
                llama::hparams::LlamaExpertGatingFuncType::SIGMOID as u32
            );
            match &spec.indexer_types {
                Some(types) => {
                    for (il, &v) in types.iter().enumerate() {
                        assert_eq!(
                            hp.is_indexer_full(il),
                            v != 0,
                            "glm-dsa indexer_types[{il}]"
                        );
                    }
                }
                None => {
                    // the pre-5.2 BC: every layer a full indexer
                    for il in 0..spec.n_layer {
                        assert!(hp.is_indexer_full(il), "glm-dsa pre-5.2 full indexer {il}");
                    }
                }
            }
        }
        Family::NemotronHMoe => {
            // the nemotron-h hparams arm serves the MOE arch unchanged
            assert_eq!(hp.ssm_d_conv as i64, spec.d_conv);
            assert_eq!(hp.n_ff_exp(4) as i64, spec.n_ff_exp);
            assert_eq!(hp.n_ff_shexp as i64, spec.n_ff_shexp);
            assert_eq!(hp.moe_latent_size as i64, spec.moe_latent);
            assert_eq!(hp.n_layer() as usize, 6);
            assert!(hp.is_recr(0) && hp.is_recr(2) && hp.is_recr(5));
            assert!(!hp.is_recr(1) && !hp.is_recr(3));
        }
    }
}

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn: fa,
    }
}

// ---------------------------------------------------------------------------
// the per-arch weight bundles (the CLI/server wiring, mirrored)
// ---------------------------------------------------------------------------

fn cohere2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2ModelWeights {
    graph_arch::Cohere2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Cohere2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn chatglm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ChatglmModelWeights {
    graph_arch::ChatglmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::ChatglmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
            })
            .collect(),
    }
}

fn bitnet_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BitnetModelWeights {
    graph_arch::BitnetModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::BitnetLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_sub_norm: l.attn_sub_norm.unwrap(),
                wq: l.wq.unwrap(),
                wq_s: l.wq_s,
                wk: l.wk.unwrap(),
                wk_s: l.wk_s,
                wv: l.wv.unwrap(),
                wv_s: l.wv_s,
                wo: l.wo.unwrap(),
                wo_s: l.wo_s,
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_sub_norm: l.ffn_sub_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_gate_s: l.ffn_gate_s,
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_s: l.ffn_down_s,
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_s: l.ffn_up_s,
            })
            .collect(),
    }
}

fn dbrx_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DbrxModelWeights {
    graph_arch::DbrxModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::DbrxLayerWeights {
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                attn_norm: l.attn_norm.unwrap(),
                attn_out_norm: l.attn_out_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn mistral3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Mistral3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Mistral3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Mistral3LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if 256 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                },
                ffn_gate: l.ffn_gate,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn minicpm3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Minicpm3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Minicpm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Minicpm3LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wq_a: l.wq_a.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                wkv_b: l.wkv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if 256 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                },
            })
            .collect(),
    }
}

fn glm4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4ModelWeights {
    graph_arch::Glm4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Glm4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

fn exaone4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Exaone4ModelWeights {
    graph_arch::Exaone4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Exaone4LayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                rope_freqs: l.rope_freqs,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

fn llama4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Llama4ModelWeights {
    graph_arch::Llama4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Llama4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn qwen2vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen2VlModelWeights {
    graph_arch::Qwen2VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen2VlLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn qwen3vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3VlModelWeights {
    graph_arch::Qwen3VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen3VlLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
            })
            .collect(),
    }
}

fn glm_dsa_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GlmDsaModelWeights {
    graph_arch::GlmDsaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::GlmDsaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wq_a: l.wq_a.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                wk_b: l.wk_b.unwrap(),
                wv_b: l.wv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// batch-5's mamba2 mixer bundle (the MOE arch reuses it wholesale)
fn mamba2_mixer_of(l: &llama::model::LayerTensors) -> graph_arch::Mamba2Mixer {
    graph_arch::Mamba2Mixer {
        ssm_in: l.ssm_in.unwrap(),
        ssm_conv1d: l.ssm_conv1d.unwrap(),
        ssm_conv1d_b: l.ssm_conv1d_b,
        ssm_dt_b: l.ssm_dt_b.unwrap(),
        ssm_a: l.ssm_a.unwrap(),
        ssm_d: l.ssm_d.unwrap(),
        ssm_norm: l.ssm_norm,
        ssm_out: l.ssm_out.unwrap(),
    }
}

fn nemotron_h_weights_of(m: &LlamaModel, n_trunk: usize) -> graph_arch::NemotronHModelWeights {
    graph_arch::NemotronHModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::NemotronHLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                mamba: l.ssm_in.map(|_| mamba2_mixer_of(l)),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo,
                wo_b: l.wo_b,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_latent_down: l.ffn_latent_down,
                ffn_latent_up: l.ffn_latent_up,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// the driver
// ---------------------------------------------------------------------------

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let mut attn = synth_attn(m, fa);
    let w = match m.arch {
        llama::arch::LlmArch::COHERE2 => ForwardWeights::Cohere2(
            cohere2_weights(m, n_trunk),
            graph_arch::Cohere2Params {
                attn,
                norm_ln_eps: hp.f_norm_eps,
                is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                logit_scale: hp.f_logit_scale,
                freq_base_swa: hp.rope_freq_base_train_swa,
                freq_scale_swa: hp.rope_freq_scale_train_swa,
            },
        ),
        llama::arch::LlmArch::CHATGLM => ForwardWeights::Chatglm(
            chatglm_weights(m, n_trunk),
            graph_arch::ChatglmParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::BITNET => ForwardWeights::Bitnet(
            bitnet_weights(m, n_trunk),
            graph_arch::BitnetParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::DBRX => ForwardWeights::Dbrx(
            dbrx_weights(m, n_trunk),
            graph_arch::DbrxParams {
                attn,
                norm_ln_eps: hp.f_norm_eps,
                clamp_kqv: hp.f_clamp_kqv,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::ERNIE4_5 => ForwardWeights::Ernie45Moe(
            ernie45moe_weights(m, n_trunk),
            graph_arch::Ernie45MoeParams {
                attn,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                n_moe_layer_step: 1,
                n_layer_dense_lead: hp.n_layer(),
                n_ff_shexp: hp.n_ff_shexp as i64,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::MISTRAL3 => ForwardWeights::Mistral3(
            mistral3_weights(m, n_trunk),
            graph_arch::Mistral3Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                f_attention_scale: hp.f_attention_scale,
                f_attn_temp_scale: hp.f_attn_temp_scale,
                f_attn_temp_offset: hp.f_attn_temp_offset,
                n_attn_temp_floor_scale: hp.n_attn_temp_floor_scale,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::MINICPM3 => {
            let mut a = attn;
            a.n_head_kv = a.n_head; // the repeated-k_pe MHA rows
            attn = a;
            ForwardWeights::Minicpm3(
                minicpm3_weights(m, n_trunk),
                graph_arch::Minicpm3Params {
                    attn: a,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    q_lora_rank: hp.n_lora_q as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    scale_embd: 12.0,
                    scale_depth: 1.4,
                    n_embd_base: 256,
                },
            )
        }
        llama::arch::LlmArch::GLM4 => ForwardWeights::Glm4(
            glm4_weights(m, n_trunk),
            graph_arch::Glm4Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                rope_sections: hp.rope_sections,
            },
        ),
        llama::arch::LlmArch::EXAONE4 => ForwardWeights::Exaone4(
            exaone4_weights(m, n_trunk),
            graph_arch::Exaone4Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                swa_none: hp.swa_type == LlamaSwaType::NONE,
                freq_base_swa: hp.rope_freq_base_train_swa,
                freq_scale_swa: hp.rope_freq_scale_train_swa,
            },
        ),
        llama::arch::LlmArch::LLAMA4 => {
            let freq_base: Vec<f32> = (0..n_trunk)
                .map(|il| {
                    if hp.is_swa(il) {
                        hp.rope_freq_base_train_swa
                    } else {
                        hp.rope_freq_base_train
                    }
                })
                .collect();
            let freq_scale: Vec<f32> = (0..n_trunk)
                .map(|il| {
                    if hp.is_swa(il) {
                        hp.rope_freq_scale_train_swa
                    } else {
                        hp.rope_freq_scale_train
                    }
                })
                .collect();
            ForwardWeights::Llama4(
                llama4_weights(m, n_trunk),
                graph_arch::Llama4Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    use_kq_norm: hp.use_kq_norm,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    f_attn_temp_offset: hp.f_attn_temp_offset,
                    n_attn_temp_floor_scale: hp.n_attn_temp_floor_scale,
                    freq_base,
                    freq_scale,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::QWEN2VL => ForwardWeights::Qwen2Vl(
            qwen2vl_weights(m, n_trunk),
            graph_arch::Qwen2VlParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                rope_sections: hp.rope_sections,
            },
        ),
        llama::arch::LlmArch::QWEN3VL | llama::arch::LlmArch::QWEN3VLMOE => {
            ForwardWeights::Qwen3Vl(
                qwen3vl_weights(m, n_trunk),
                graph_arch::Qwen3VlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                    n_deepstack_layers: hp.n_deepstack_layers as usize,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::GLM_DSA => {
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64; // kv_lora + qk_rope
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            attn = a;
            ForwardWeights::GlmDsa(
                glm_dsa_weights(m, n_trunk),
                graph_arch::GlmDsaParams {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    rope_yarn_log_mul: hp.rope_yarn_log_mul,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k as i64,
                    f_norm_eps: hp.f_norm_eps,
                    is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
                },
            )
        }
        llama::arch::LlmArch::NEMOTRON_H | llama::arch::LlmArch::NEMOTRON_H_MOE => {
            let il0 = (0..n_trunk)
                .find(|&il| !hp.is_recr(il) && hp.n_ff(il) == 0)
                .unwrap_or(0);
            let a = {
                let mut a = synth_attn(m, fa);
                a.n_head = hp.n_head(il0) as i64;
                a.n_head_kv = hp.n_head_kv(il0) as i64;
                a
            };
            attn = a;
            ForwardWeights::NemotronH(
                nemotron_h_weights_of(m, n_trunk),
                graph_arch::NemotronHParams {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    n_ff: (0..n_trunk).map(|il| hp.n_ff(il) as i64).collect(),
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    n_ssm_head: hp.ssm_dt_rank as i64,
                    n_group: hp.ssm_n_group as i64,
                    norm_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        other => panic!("no batch-13 driver for {}", other.name()),
    };
    let _ = &mut attn;
    (w, attn)
}

fn ernie45moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Ernie45MoeModelWeights {
    graph_arch::Ernie45MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Ernie45MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn driver_for(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let hp = &m.hparams;
    if hp.swa_type != LlamaSwaType::NONE && hp.is_swa_any() {
        DecodeContext::new_with_swa(
            gctx,
            weights,
            attn,
            256,
            8,
            512,
            llama::kv_cache::SwaCacheSpec::from_hparams(hp),
        )
    } else {
        DecodeContext::new_with(gctx, weights, attn, 256, 8, 512)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// llama4-noswa vs plain at SMALL positions: the temperature scale is
/// log(floor((pos+1)/8192)+1)*0.1+1 = 1.0 exactly for pos < 8191, and the
/// swa==0 arm's `n_no_rope_layer_step = n_layer` keeps the same rope skip
/// (the last layer), so the two files agree bit-for-bit below the first
/// CHUNKED window boundary — the C formula's property, pinned
#[test]
fn arch_batch13_llama4_temp_scale_below_first_chunk() {
    let lg1 = {
        let mut m = load_synth(&spec_llama4());
        let mut d = driver_for(&mut m, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let lg2 = {
        let mut m = load_synth(&spec_llama4_noswa());
        let mut d = driver_for(&mut m, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let diff: f32 = lg1
        .iter()
        .zip(&lg2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert_eq!(diff, 0.0, "llama4 temp scale is exactly 1.0 below pos 8192");
    println!("llama4 CHUNKED vs -noswa below the first chunk: max |Δlogit| = {diff}");
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}

/// the tensor/hparams pins + the double-FA smoke of the parity cells
#[test]
fn arch_batch13_pin_and_smoke() {
    for spec in parity_specs() {
        {
            let m = load_synth(&spec);
            pin_hparams(&m, &spec);
            pin_tensors(&m, &spec);
        }
        for fa in [false, true] {
            // the model's Context moves into the driver — reload per FA mode
            let mut m = load_synth(&spec);
            let mut dctx = driver_for(&mut m, fa);
            let prompt: Vec<i32> = (1..=12).collect();
            let pos: Vec<i32> = (0..12).collect();
            let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{} fa={fa}: non-finite logits",
                spec.arch
            );
            let tk = logits_of_argmax(logits.chunks(32000).last().unwrap());
            let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
            assert!(next.iter().all(|v| v.is_finite()));
            assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
        }
        println!(
            "{}{}: pinned + smoke ok (both FA modes)",
            spec.arch, spec.suffix
        );
    }
}

/// the >64-token prompt cell of the parity protocol, in one ubatch — the
/// iswa members (cohere2 / exaone4-swa with window 32) cross their windows,
/// llama4's chunked pattern + no-rope layers run past 64, glm-dsa's lid rows
/// + top-k selection evolve across ~100 tokens
#[test]
fn arch_batch13_long_prompt_cells() {
    for spec in parity_specs() {
        let mut m = load_synth(&spec);
        let mut dctx = driver_for(&mut m, false);
        let n = 100usize;
        let prompt: Vec<i32> = (1..=n as i32).collect();
        let pos: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let lg = dctx.decode(&prompt, &pos).expect("long prefill").to_vec();
        assert!(lg.iter().all(|v| v.is_finite()), "{}: long cell", spec.arch);
        let tk = logits_of_argmax(&lg);
        let next = dctx.decode(&[tk], &[n as i32]).expect("decode").to_vec();
        assert!(next.iter().all(|v| v.is_finite()));
        println!("{}{}: 100-token cell ok", spec.arch, spec.suffix);
    }
}

/// the variants must genuinely diverge: mistral3-temp vs plain (the post-rope
/// temperature scaling), llama4-noswa vs plain (CHUNKED+no-rope+temp vs
/// always-rope), glm-dsa shared vs full indexers
#[test]
fn arch_batch13_variants_differ() {
    let pairs = [
        (spec_mistral3(), spec_mistral3_temp()),
        (spec_glm4(), spec_glm4_mrope()),
        (spec_glm_dsa(), spec_glm_dsa_shared()),
    ];
    // 12 tokens: the glm-dsa pair needs n_kv > top_k (8) so the shared
    // layers' reused top-k actually diverges from a fresh per-layer one
    let ids: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    for (plain, var) in pairs {
        let lg1 = {
            let mut m = load_synth(&plain);
            let mut d = driver_for(&mut m, false);
            d.decode(&ids, &pos).unwrap().to_vec()
        };
        let lg2 = {
            let mut m = load_synth(&var);
            let mut d = driver_for(&mut m, false);
            d.decode(&ids, &pos).unwrap().to_vec()
        };
        let diff: f32 = lg1
            .iter()
            .zip(&lg2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            diff > 0.0,
            "{}{} must differ from the plain file (max |d| = {diff})",
            var.arch,
            var.suffix
        );
        println!(
            "{}{} vs plain: max |Δlogit| = {diff:.4}",
            var.arch, var.suffix
        );
    }
}

// ---------------------------------------------------------------------------
// DECDMP1 node-dump mirror (parity/ref_decode_dump.c protocol) — the
// whole-graph bisect stream for `parity/decode_dump_cmp.py` (the batch-11a
// precedent). Run:
//   B13_DUMP_MODEL=... B13_DUMP_OUT=... B13_FA_OFF=1 cargo test --release
//     -p llama --test arch_batch13_e2e -- --ignored --nocapture
//     arch_batch13_prefill_node_dump
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// nodes at/above this element count carry no payload (2^19, the C probe rule)
const DUMP_ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static DUMP_ACTIVE: AtomicU32 = AtomicU32::new(0);

fn dump_op_desc(op: ggml::GgmlOp, op_params: &[i32]) -> &'static str {
    use ggml::GgmlOp::*;
    // the port folds every unary into GgmlOp::Silu with the real op in
    // params[0] — expand it like the C probe's ggml_op_desc
    // (GGML_UNARY_OP_NAME, ggml.c:1224-1246) so the streams align by name
    if op == Silu {
        const NAMES: [&str; 22] = [
            "ABS",
            "SGN",
            "NEG",
            "STEP",
            "TANH",
            "ELU",
            "RELU",
            "SIGMOID",
            "GELU",
            "GELU_QUICK",
            "SILU",
            "HARDSWISH",
            "HARDSIGMOID",
            "EXP",
            "EXPM1",
            "SOFTPLUS",
            "GELU_ERF",
            "XIELU",
            "FLOOR",
            "CEIL",
            "ROUND",
            "TRUNC",
        ];
        let u = op_params[0] as usize;
        if u < NAMES.len() {
            return NAMES[u];
        }
        return "UNARY?";
    }
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        Mul => "MUL",
        Div => "DIV",
        Sub => "SUB",
        Norm => "NORM",
        SquaredMulMat => "MUL_MAT_SQ",
        MulMat => "MUL_MAT",
        Scale => "SCALE",
        Cpy => "CPY",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        DiagMaskInf => "DIAG_MASK_INF",
        SoftMax => "SOFT_MAX",
        RoPE => "ROPE",
        RoPEBack => "ROPE_BACK",
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => "SILU",
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => "GLU",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        Clamp => "CLAMP",
        Gdn => "GATED_DELTA_NET",
        Im2col => "IM2COL",
        Upscale => "UPSCALE",
        Fill => "FILL",
        LightningIndexer => "LIGHTNING_INDEXER",
        TopK => "TOP_K",
        Sqrt => "SQRT",
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        Pad => "PAD",
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn dump_type_desc(ty: ggml::types::GgmlType) -> &'static str {
    use ggml::types::GgmlType::*;
    match ty {
        F32 => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        I64 => "i64",
        I32 => "i32",
        I16 => "i16",
        I8 => "i8",
        _ => "other",
    }
}

fn dump_put_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len().min(255);
    buf.push(len as u8);
    buf.extend_from_slice(&s.as_bytes()[..len]);
}

fn dump_cb(node: &ggml::compute::EvalNode<'_>, ask: bool) -> bool {
    use ggml::types::GgmlType;
    if ask {
        return true;
    }
    let mut guard = DUMP.get().unwrap().lock().unwrap();
    let Some(st) = guard.as_mut() else {
        return true;
    };
    let n: i64 = node.ne.iter().product();
    st.nodes += 1;
    dump_put_str(&mut st.out, dump_op_desc(node.op, &node.op_params));
    dump_put_str(&mut st.out, node.name);
    dump_put_str(&mut st.out, dump_type_desc(node.ty));
    st.out.extend(&node.ne.map(|v| v.to_le_bytes()).concat());
    st.out.extend(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= DUMP_ELEM_CAP {
        return true;
    }
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
        return true;
    }
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v: f32 = if node.ty == GgmlType::F32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            let h = half::f16::from_le_bytes([data[off], data[off + 1]]);
            h.to_f32()
        };
        st.out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

/// stream every node of one batch-13 `decode_embed` prefill to B13_DUMP_OUT —
/// the DECDMP1 mirror of `parity/ref_decode_dump.c` for the batch-13 files
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for a batch-13 parity file"]
fn arch_batch13_prefill_node_dump() {
    let model_path = std::env::var("B13_DUMP_MODEL")
        .unwrap_or_else(|_| "/tmp/arch-batch13/llama4-synth.gguf".to_string());
    let out_path =
        std::env::var("B13_DUMP_OUT").unwrap_or_else(|_| "/tmp/b13-port.bin".to_string());
    let prompt =
        std::env::var("B13_DUMP_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
    let fa_off = std::env::var("B13_FA_OFF").is_ok();

    let mut m = open_model(&model_path);
    let vocab =
        llama::vocab::Vocab::load(&Gguf::open(&model_path).expect("open gguf")).expect("vocab");
    let ids = vocab.tokenize(&prompt, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!(
        "tokens: {ids:?} ({} tokens, fa={}, {})",
        ids.len(),
        if fa_off { "off" } else { "on" },
        model_path
    );

    // decode_embed protocol: every token an output row (the C probe's
    // --embeddings --pooling none context)
    let mut dctx =
        driver_for(&mut m, !fa_off).with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);

    DUMP.get_or_init(|| {
        Mutex::new(Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        }))
    });
    {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        *guard = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    ggml::compute::set_eval_callback(Some(dump_cb));
    DUMP_ACTIVE.store(1, Ordering::SeqCst);
    let embd = dctx.decode_embed(&ids, &pos).expect("decode_embed");
    DUMP_ACTIVE.store(0, Ordering::SeqCst);
    ggml::compute::set_eval_callback(None);

    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    use std::io::Write as _;
    let mut f = std::fs::File::create(&out_path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
    for &id in &ids {
        f.write_all(&id.to_le_bytes()).unwrap();
    }
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    drop(f);

    println!(
        "arch_batch13_prefill_node_dump: {nodes} nodes, embd {}x{}, -> {out_path}",
        embd.n_rows, embd.n_embd_out
    );
}
