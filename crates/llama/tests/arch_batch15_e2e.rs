//! arch_batch15_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! 2026-10: **the P1+P2 queue of parity/AUDIT_models.md** — qwen(v1) /
//! maincoder / pangu-embed / plm / cogvlm / spark2-5 / muse-glimmer / llada /
//! hunyuan-vl(+dense) / granite-swa / afmoe / mellum / paddleocr /
//! gemma-embedding / llama-embed / mistral4 / hy-v3 / hy-v4 / mimo2 / step35
//! (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-14 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): each arch is verified on a *synthetic* file built with the
//! port's byte-exact GGUF writer — `tokenizer.*` KV copied verbatim from the
//! llama SPM vocab fixture, the arch's own KV, and exactly the tensor names +
//! shapes its `load_arch_tensors` asks for, all F32. The parity cells drive
//! llama-cli itself (batch-6+ protocol); these in-port tests pin the loaders
//! + builders and smoke both FA modes.
//!
//! Exceptions (the established precedents):
//!   * **llada** — the diffusion family (llama-model.cpp:2289-2295 creates no
//!     memory; every reference driver refuses to generate): verified in-port
//!     like llada-moe/dream/rnd1 (PARITY.md 批次 10/11b).
//!   * **gemma-embedding / llama-embed** — encoder-side graphs (no logits):
//!     verified in-port here + against the reference `llama_encode` dumps via
//!     `parity/ref_encode_dump` (the ignored reference-parity tests below,
//!     the bert/t5/eurobert protocol).
//!   * **gemma4-assistant** — documented-skip (PARITY.md 批次 15): the graph
//!     GGML_ASSERTs `cparams.ctx_other != nullptr` (gemma4-assistant.cpp:108)
//!     — it is a draft module of the gemma4 trunk, unloadable standalone.

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{DecodeContext, EncoderContext, EncoderWeights, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaSwaType;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch15";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Qwen1,
    Maincoder,
    PanguEmbed,
    Cogvlm,
    Spark25,
    MuseGlimmer,
    Llada,
    Plm,
    HunyuanVl,
    GraniteSwa,
    GraniteSwaMoe,
    Afmoe,
    Mellum,
    PaddleOcr,
    HyV3,
    Mimo2,
    Step35,
    HyV4,
    HyV4Dsa,
    /// the deepseek2-MLA recipe under the mistral4 arch name (models.h:1393)
    Mistral4,
    GemmaEmbedding,
    LlamaEmbed,
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
    write_output: bool,
    rope_sections: Option<[u32; 4]>,
    /// sliding_window + pattern keys (None → no keys)
    swa: Option<(u32, Vec<u32>)>,
    /// muse-glimmer's final_logit_softcapping (0 → no key)
    final_logit_softcapping: f32,
    /// logit scale keys (granite-swa / muse-glimmer — REQUIRED)
    logit_scale: Option<f32>,
    /// granite scale keys (0 → no key)
    embedding_scale: f32,
    residual_scale: f32,
    attention_scale: f32,
    /// attention sinks (granite-swa REQUIRED, mimo2 optional)
    sinks: bool,
    /// attention.value_scale (mimo2; 0 → no key)
    value_scale: f32,
    // ---- MoE ----
    n_ff_exp: i64,
    n_ff_shexp: i64,
    n_expert_shared: i64,
    n_layer_dense_lead: u32,
    /// per-layer ffn_exp array (afmoe/mellum/hy-v3/mimo2/step35/hy-v4 read
    /// n_ff_exp; write the scalar key)
    // ---- MLA (plm / hy-v4 / mistral4) ----
    q_lora_rank: i64,
    kv_lora_rank: i64,
    key_length_mla: i64,
    value_length_mla: i64,
    // ---- hy-v4 iHC + DSA ----
    hc_mult: i64,
    indexer_n_head: i64,
    indexer_head_size: i64,
    indexer_top_k: i64,
    /// None → no attention.indexer.types key
    indexer_types: Option<Vec<u32>>,
    // ---- gemma-embedding / encoders ----
    /// the symmetric swa window (gemma-embedding — REQUIRED)
    sym_swa: u32,
    /// rope.scaling.alpha (hunyuan-vl's XDRoPE; 0 → no key)
    xdr_alpha: f32,
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
        swa: None,
        final_logit_softcapping: 0.0,
        logit_scale: None,
        embedding_scale: 0.0,
        residual_scale: 0.0,
        attention_scale: 0.0,
        sinks: false,
        value_scale: 0.0,
        n_ff_exp: 24,
        n_ff_shexp: 24,
        n_expert_shared: 0,
        n_layer_dense_lead: 0,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        key_length_mla: 0,
        value_length_mla: 0,
        hc_mult: 0,
        indexer_n_head: 0,
        indexer_head_size: 0,
        indexer_top_k: 0,
        indexer_types: None,
        sym_swa: 0,
        xdr_alpha: 0.0,
    }
}

fn spec_qwen1() -> SynthSpec {
    base("qwen", Family::Qwen1).with(|s| {
        s.n_head_kv = 4; // MHA — the fused QKV thirds
        s.write_output = true;
    })
}

fn spec_maincoder() -> SynthSpec {
    base("maincoder", Family::Maincoder)
}

fn spec_pangu_embed() -> SynthSpec {
    base("pangu-embedded", Family::PanguEmbed)
}

fn spec_cogvlm() -> SynthSpec {
    base("cogvlm", Family::Cogvlm).with(|s| {
        s.n_head_kv = 4; // MHA — the n_embd-sized view offsets
    })
}

/// spark2_5 — iswa + the per-head gate; the GELU FFN
fn spec_spark25() -> SynthSpec {
    base("spark2_5", Family::Spark25).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
    })
}

/// muse-glimmer — iswa (rope ONLY on the SWA layers), the wide gate, the
/// dual 1e-8 post-norms, logit_scale + the tanh softcap
fn spec_muse_glimmer() -> SynthSpec {
    base("muse-glimmer", Family::MuseGlimmer).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
        s.logit_scale = Some(8.0);
        s.write_output = true;
    })
}

/// muse-glimmer `-softcap` — the final_logit_softcapping variant
fn spec_muse_glimmer_softcap() -> SynthSpec {
    spec_muse_glimmer().with(|s| {
        s.suffix = "-softcap";
        // an extreme cap so the [x/c → tanh → ×c] round trip saturates on
        // these small synthetic logits (the gemma3-sized 30.0 rounds to x
        // bit-exactly at this magnitude)
        s.final_logit_softcapping = 0.001;
    })
}

/// llada — the dense diffusion arch (in-port only; no reference memory)
fn spec_llada() -> SynthSpec {
    base("llada", Family::Llada)
}

/// plm — the MLA-lite (shared k_pe, relu² FFN, tied head)
fn spec_plm() -> SynthSpec {
    base("plm", Family::Plm).with(|s| {
        s.key_length = 32; // qk_nope 16 + qk_rope 16
        s.kv_lora_rank = 24;
        s.n_head_kv = 4; // the repeated-k_pe rows make the cache MHA-wide
    })
}

/// hunyuan-vl — the plain (no-sections) file → rope_ext
fn spec_hunyuan_vl() -> SynthSpec {
    base("hunyuan_vl", Family::HunyuanVl)
}

/// hunyuan-vl `-mrope` — rope.dimension_sections → ggml_rope_multi + the
/// XDRoPE alpha rebase (hunyuan-vl.cpp:8-12 — the sections alone partition
/// the SAME text pos id across the blocks, which equals the plain rope; the
/// alpha rebase of rope_freq_base_train is the arch's differentiator)
fn spec_hunyuan_vl_mrope() -> SynthSpec {
    spec_hunyuan_vl().with(|s| {
        s.suffix = "-mrope";
        s.rope_sections = Some([4, 4, 4, 0]);
        s.xdr_alpha = 1.3;
    })
}

/// hunyuan-dense — the pure typedef (hunyuan_vl's loader+graph, arch name
/// only, models.h:2101-2103); sections like the VL files
fn spec_hunyuan_dense() -> SynthSpec {
    base("hunyuan-dense", Family::HunyuanVl).with(|s| {
        s.rope_sections = Some([4, 4, 4, 0]);
    })
}

/// granite-swa — the dense text-only file (sinks + the granite scales)
fn spec_granite_swa() -> SynthSpec {
    base("granite_swa", Family::GraniteSwa).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
        s.logit_scale = Some(8.0);
        s.embedding_scale = 12.0;
        s.residual_scale = 0.1;
        s.sinks = true;
    })
}

/// granite-swa `-moe` — the softmax MoE + the SWIGLU fused shexp
fn spec_granite_swa_moe() -> SynthSpec {
    spec_granite_swa().with(|s| {
        s.suffix = "-moe";
        s.family = Family::GraniteSwaMoe;
        s.n_ff_shexp = 24;
    })
}

/// afmoe — the iswa file (n_swa > 0 — REQUIRED for the reference: the graph
/// calls build_attn_inp_kv_iswa unconditionally, afmoe.cpp:123, and a
/// swa-less file segfaults the pinned reference's iswa input build): MuP
/// scale + dual norms + the gated attention + the sigmoid MoE + the wide
/// shared expert
fn spec_afmoe() -> SynthSpec {
    base("afmoe", Family::Afmoe).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
        s.n_expert_shared = 1;
        s.n_layer_dense_lead = 1;
    })
}

/// afmoe `-noswa` — the n_swa == 0 variant (in-port only: swa_type NONE has
/// no iswa pair for the reference's unconditional build_attn_inp_kv_iswa)
fn spec_afmoe_noswa() -> SynthSpec {
    spec_afmoe().with(|s| {
        s.suffix = "-noswa";
        s.swa = None;
    })
}

fn spec_mellum() -> SynthSpec {
    // mellum.cpp:34 — the separate head is REQUIRED
    base("mellum", Family::Mellum).with(|s| {
        s.write_output = true;
    })
}

/// mellum `-swa` — the plain-rope SWA layers
fn spec_mellum_swa() -> SynthSpec {
    spec_mellum().with(|s| {
        s.suffix = "-swa";
        s.swa = Some((32, vec![0, 1, 1, 1]));
    })
}

fn spec_paddleocr() -> SynthSpec {
    base("paddleocr", Family::PaddleOcr).with(|s| {
        s.rope_sections = Some([4, 4, 4, 0]);
    })
}

fn spec_hy_v3() -> SynthSpec {
    base("hy_v3", Family::HyV3).with(|s| {
        s.n_ff_shexp = 24;
        s.n_layer_dense_lead = 1;
    })
}

fn spec_mimo2() -> SynthSpec {
    base("mimo2", Family::Mimo2).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
        s.sinks = true;
        s.write_output = true;
    })
}

/// mimo2 `-vscale` — attention.value_scale 0.5
fn spec_mimo2_vscale() -> SynthSpec {
    spec_mimo2().with(|s| {
        s.suffix = "-vscale";
        s.value_scale = 0.5;
    })
}

/// step35 — iswa MoE with the HALVED full-attention rope dims (n_rot 16 →
/// the full layers rope 8), the shared rope_freqs, the optional Q/K norms +
/// the per-head gate
fn spec_step35() -> SynthSpec {
    base("step35", Family::Step35).with(|s| {
        s.swa = Some((32, vec![0, 1, 1, 1]));
        s.n_ff_shexp = 24;
        s.n_layer_dense_lead = 1;
        s.write_output = true;
        s.sinks = false;
    })
}

/// hy-v4 — the all-full-attention file (no DSA indexer): iHC + gated MLA +
/// the sigmoid MoE + the wide shared expert
fn spec_hy_v4() -> SynthSpec {
    base("hy_v4", Family::HyV4).with(|s| {
        s.n_embd = 128;
        s.n_head = 4;
        s.n_head_kv = 1;
        s.key_length = 48; // kv_lora 32 + qk_rope 16
        s.value_length = 32; // kv_lora
        s.q_lora_rank = 32;
        s.kv_lora_rank = 32;
        s.key_length_mla = 40; // qk_nope 24 + qk_rope 16
        s.value_length_mla = 20;
        s.hc_mult = 2;
        s.n_expert_shared = 1;
        s.n_layer_dense_lead = 1;
        s.write_output = true;
    })
}

/// hy-v4 `-dsa` — the lightning indexer + the shared-indexer layers
/// (attention.indexer_types [1,0,0,0])
fn spec_hy_v4_dsa() -> SynthSpec {
    spec_hy_v4().with(|s| {
        s.suffix = "-dsa";
        s.family = Family::HyV4Dsa;
        s.indexer_n_head = 2;
        s.indexer_head_size = 64;
        s.indexer_top_k = 8;
        s.indexer_types = Some(vec![1, 0, 0, 0]);
    })
}

/// mistral4 — the deepseek2-MLA recipe under the mistral4 arch name
fn spec_mistral4() -> SynthSpec {
    base("mistral4", Family::Mistral4).with(|s| {
        s.n_embd = 128;
        s.n_head = 4;
        s.n_head_kv = 1;
        s.key_length = 48;
        s.value_length = 32;
        s.q_lora_rank = 32;
        s.kv_lora_rank = 32;
        s.key_length_mla = 40;
        s.value_length_mla = 20;
        s.n_expert_shared = 1;
        s.n_layer_dense_lead = 1;
    })
}

/// gemma-embedding — the symmetric-SWA encoder (window 8 so the 12-token
/// prompts bind it; the every-6th-full pattern)
fn spec_gemma_embedding() -> SynthSpec {
    base("gemma-embedding", Family::GemmaEmbedding).with(|s| {
        s.n_head_kv = 4;
        s.swa = Some((8, Vec::new()));
        s.sym_swa = 8;
    })
}

/// llama-embed — the LLAMA tensor set under the embed arch name
fn spec_llama_embed() -> SynthSpec {
    base("llama-embed", Family::LlamaEmbed)
}

/// every file the ignored writer test emits
fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_qwen1(),
        spec_maincoder(),
        spec_pangu_embed(),
        spec_cogvlm(),
        spec_spark25(),
        spec_muse_glimmer(),
        spec_muse_glimmer_softcap(),
        spec_llada(),
        spec_plm(),
        spec_hunyuan_vl(),
        spec_hunyuan_vl_mrope(),
        spec_hunyuan_dense(),
        spec_granite_swa(),
        spec_granite_swa_moe(),
        spec_afmoe(),
        spec_afmoe_noswa(),
        spec_mellum(),
        spec_mellum_swa(),
        spec_paddleocr(),
        spec_hy_v3(),
        spec_mimo2(),
        spec_mimo2_vscale(),
        spec_step35(),
        spec_hy_v4(),
        spec_hy_v4_dsa(),
        spec_mistral4(),
        spec_gemma_embedding(),
        spec_llama_embed(),
    ]
}

/// the parity cells (batch-15 default set): every file that creates a memory
/// in the pinned reference (llada is the diffusion exception; the two
/// encoders ride llama_encode)
fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_qwen1(),
        spec_maincoder(),
        spec_pangu_embed(),
        spec_cogvlm(),
        spec_spark25(),
        spec_muse_glimmer(),
        spec_plm(),
        spec_hunyuan_vl(),
        spec_hunyuan_vl_mrope(),
        spec_hunyuan_dense(),
        spec_granite_swa(),
        spec_granite_swa_moe(),
        spec_afmoe(),
        spec_mellum(),
        spec_paddleocr(),
        spec_hy_v3(),
        spec_mimo2(),
        spec_step35(),
        spec_hy_v4(),
        spec_hy_v4_dsa(),
        spec_mistral4(),
    ]
}

#[test]
#[ignore = "writes the /tmp/arch-batch15 parity files (ARCH_BATCH15 cells)"]
fn arch_batch15_write_synth() {
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
    Decay,
}

fn tensors_for(spec: &SynthSpec) -> Vec<((String, Vec<i64>), Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let hd = spec.key_length;
    let n_expert = N_EXPERT;
    let mut t: Vec<((String, Vec<i64>), Role)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, role: Role| t.push(((name, ne), role));

    push(
        "token_embd.weight".into(),
        vec![n_embd, N_VOCAB],
        Role::Proj,
    );
    push("output_norm.weight".into(), vec![n_embd], Role::Norm);
    if spec.write_output {
        push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
    }

    for i in 0..spec.n_layer as i32 {
        let qkv = |push: &mut dyn FnMut(String, Vec<i64>, Role), nq: i64, nk: i64, nv: i64| {
            // the separate triple (the port's create_tensor_qkv picks the
            // separate branch when no fused attn_qkv.weight exists)
            push(
                format!("blk.{i}.attn_q.weight"),
                vec![n_embd, nq],
                Role::Proj,
            );
            push(
                format!("blk.{i}.attn_k.weight"),
                vec![n_embd, nk],
                Role::Proj,
            );
            push(
                format!("blk.{i}.attn_v.weight"),
                vec![n_embd, nv],
                Role::Proj,
            );
        };
        match spec.family {
            Family::Qwen1 => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, n_embd * 3],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_qkv.bias"),
                    vec![n_embd * 3],
                    Role::Bias,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff / 2],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff / 2, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff / 2],
                    Role::Proj,
                );
            }
            Family::Maincoder => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::PanguEmbed | Family::HunyuanVl => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                if spec.family == Family::PanguEmbed {
                    push(
                        format!("blk.{i}.attn_output.bias"),
                        vec![n_embd],
                        Role::Bias,
                    );
                } else {
                    push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                    push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                }
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::Cogvlm => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, hd * spec.n_head * 3],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.vis_attn_qkv.weight"),
                    vec![n_embd, hd * spec.n_head * 3],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.vis_attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.vis_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.vis_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.vis_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::Spark25 => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_gate.weight"),
                    vec![n_embd, spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
            }
            Family::MuseGlimmer => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(
                    format!("blk.{i}.attn_gate.weight"),
                    vec![n_embd, hd * spec.n_head],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.post_ffw_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::Llada => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::Plm => {
                let kv_lora = spec.kv_lora_rank;
                let qk_rope = spec.rope_dim;
                let qk_nope = hd - qk_rope;
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, hd * spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, kv_lora + qk_rope],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_kv_a_norm.weight"),
                    vec![kv_lora],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_kv_b.weight"),
                    vec![kv_lora, spec.n_head * (qk_nope + spec.value_length)],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.value_length, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::GraniteSwa | Family::GraniteSwaMoe => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_sinks.weight"),
                    vec![spec.n_head],
                    Role::Bias,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if spec.family == Family::GraniteSwa {
                    push(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![n_ff, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, n_ff, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, n_ff, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, 2 * spec.n_ff_shexp],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_shexp, n_embd],
                        Role::Proj,
                    );
                }
            }
            Family::Afmoe => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(
                    format!("blk.{i}.attn_gate.weight"),
                    vec![n_embd, hd * spec.n_head],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.post_ffw_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                if (i as u32) >= spec.n_layer_dense_lead {
                    let nffs = spec.n_ff_exp * spec.n_expert_shared;
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
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, nffs],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![nffs, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, nffs],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                }
            }
            Family::Mellum => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_expert],
                    Role::Router,
                );
                push(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_expert],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_expert],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_expert],
                    Role::Proj,
                );
            }
            Family::PaddleOcr => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
            Family::HyV3 => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if (i as u32) < spec.n_layer_dense_lead {
                    push(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    // hy-v3.cpp:68 — the suffix-LESS "exp_probs_b"
                    push(format!("blk.{i}.exp_probs_b"), vec![n_expert], Role::Bias);
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, spec.n_ff_shexp],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, spec.n_ff_shexp],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_shexp, n_embd],
                        Role::Proj,
                    );
                }
            }
            Family::Mimo2 => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.value_length * spec.n_head, n_embd],
                    Role::Proj,
                );
                if spec.sinks {
                    push(
                        format!("blk.{i}.attn_sinks.weight"),
                        vec![spec.n_head],
                        Role::Bias,
                    );
                }
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_expert],
                    Role::Router,
                );
                push(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_expert],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_expert],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_expert],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.exp_probs_b.bias"),
                    vec![n_expert],
                    Role::Bias,
                );
            }
            Family::Step35 => {
                let n_rot_max = spec.rope_dim; // the widest layer's rope
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.value_length * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_gate.weight"),
                    vec![n_embd, spec.n_head],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if (i as u32) < spec.n_layer_dense_lead {
                    push(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.exp_probs_b.bias"),
                        vec![n_expert],
                        Role::Bias,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, spec.n_ff_shexp],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, spec.n_ff_shexp],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_shexp, n_embd],
                        Role::Proj,
                    );
                }
            }
            Family::HyV4 | Family::HyV4Dsa | Family::Mistral4 => {
                let kv_lora = spec.kv_lora_rank;
                let qk_rope = spec.rope_dim;
                let k_mla = spec.key_length_mla;
                let v_mla = spec.value_length_mla;
                let qk_nope = k_mla - qk_rope;
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_q_a.weight"),
                    vec![n_embd, spec.q_lora_rank],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_q_a_norm.weight"),
                    vec![spec.q_lora_rank],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_q_b.weight"),
                    vec![spec.q_lora_rank, spec.n_head * k_mla],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, kv_lora + qk_rope],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_kv_a_norm.weight"),
                    vec![kv_lora],
                    Role::Norm,
                );
                push(
                    format!("blk.{i}.attn_k_b.weight"),
                    vec![qk_nope, kv_lora, spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_v_b.weight"),
                    vec![kv_lora, v_mla, spec.n_head],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * v_mla, n_embd],
                    Role::Proj,
                );
                if spec.family == Family::HyV4 || spec.family == Family::HyV4Dsa {
                    let hc = spec.hc_mult;
                    push(
                        format!("blk.{i}.attn_gate.weight"),
                        vec![n_embd, spec.n_head * v_mla],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.attn_sinks.weight"),
                        vec![spec.n_head],
                        Role::Bias,
                    );
                    if spec.family == Family::HyV4Dsa && i == 0 {
                        // only the "full" indexer layer ships the five
                        let ih = spec.indexer_head_size;
                        let inh = spec.indexer_n_head;
                        push(
                            format!("blk.{i}.indexer.attn_q_b.weight"),
                            vec![spec.q_lora_rank, inh * ih],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.indexer.attn_k.weight"),
                            vec![n_embd, ih],
                            Role::Proj,
                        );
                        push(
                            format!("blk.{i}.indexer.k_norm.weight"),
                            vec![ih],
                            Role::Norm,
                        );
                        push(format!("blk.{i}.indexer.k_norm.bias"), vec![ih], Role::Bias);
                        push(
                            format!("blk.{i}.indexer.proj.weight"),
                            vec![n_embd, inh],
                            Role::Proj,
                        );
                    }
                    push(
                        format!("blk.{i}.hc_attn_fn.weight"),
                        vec![hc * n_embd, 2 * hc],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.hc_attn_base.weight"),
                        vec![2 * hc],
                        Role::Norm,
                    );
                    push(format!("blk.{i}.hc_attn_scale.weight"), vec![2], Role::Norm);
                    push(
                        format!("blk.{i}.hc_ffn_fn.weight"),
                        vec![hc * n_embd, 2 * hc],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.hc_ffn_base.weight"),
                        vec![2 * hc],
                        Role::Norm,
                    );
                    push(format!("blk.{i}.hc_ffn_scale.weight"), vec![2], Role::Norm);
                }
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if (i as u32) < spec.n_layer_dense_lead {
                    push(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                } else {
                    let nffs = spec.n_ff_exp * spec.n_expert_shared;
                    push(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, n_expert],
                        Role::Router,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_expert],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, nffs],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![nffs, n_embd],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, nffs],
                        Role::Proj,
                    );
                }
            }
            Family::GemmaEmbedding => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                push(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.post_ffw_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
            }
            Family::LlamaEmbed => {
                push(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm,
                );
                qkv(
                    &mut push,
                    hd * spec.n_head,
                    hd * spec.n_head_kv,
                    hd * spec.n_head_kv,
                );
                push(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * spec.n_head, n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj,
                );
            }
        }
    }

    // step35's shared rope-factors tensor — ROPE_FREQS's template has no
    // blk slot ("rope_freqs"), so ONE tensor serves every layer (the
    // layer-0 NOT_REQUIRED read + the DUPLICATED re-requests)
    if spec.family == Family::Step35 {
        let n_rot_max = spec.rope_dim;
        push("rope_freqs.weight".into(), vec![n_rot_max / 2], Role::Norm);
    }

    // the model-level iHC head (hy-v4)
    if matches!(spec.family, Family::HyV4 | Family::HyV4Dsa) {
        let hc = spec.hc_mult;
        push(
            "output_hc_fn.weight".into(),
            vec![hc * spec.n_embd, hc],
            Role::Proj,
        );
        push("output_hc_base.weight".into(), vec![hc], Role::Norm);
        push("output_hc_scale.weight".into(), vec![1], Role::Norm);
    }
    t
}

// ---------------------------------------------------------------------------
// writer (same recipe as the earlier arch batches)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch15");

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
    kv!(
        format!("{a}.feed_forward_length"),
        Value::U32(spec.n_ff as u32)
    );
    kv!(
        format!("{a}.attention.head_count"),
        Value::U32(spec.n_head as u32)
    );
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::U32(spec.n_head_kv as u32)
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
    // every batch-15 arch reads the RMS eps
    kv!(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5)
    );
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
    if spec.final_logit_softcapping != 0.0 {
        kv!(
            format!("{a}.final_logit_softcapping"),
            Value::F32(spec.final_logit_softcapping)
        );
    }
    if let Some(ls) = spec.logit_scale {
        kv!(format!("{a}.logit_scale"), Value::F32(ls));
    }
    if spec.embedding_scale != 0.0 {
        kv!(
            format!("{a}.embedding_scale"),
            Value::F32(spec.embedding_scale)
        );
    }
    if spec.residual_scale != 0.0 {
        kv!(
            format!("{a}.residual_scale"),
            Value::F32(spec.residual_scale)
        );
    }
    if spec.attention_scale != 0.0 {
        kv!(
            format!("{a}.attention.scale"),
            Value::F32(spec.attention_scale)
        );
    }
    if spec.value_scale != 0.0 {
        kv!(
            format!("{a}.attention.value_scale"),
            Value::F32(spec.value_scale)
        );
    }
    if spec.xdr_alpha != 0.0 {
        // XDRoPE (hunyuan-vl.cpp:8-12) — the alpha KV is read unconditionally
        // (llama-model.cpp:1364, no scaling.type gate)
        kv!(
            format!("{a}.rope.scaling.alpha"),
            Value::F32(spec.xdr_alpha)
        );
    }
    // MoE keys
    let needs_experts = matches!(
        spec.family,
        Family::Afmoe
            | Family::Mellum
            | Family::HyV3
            | Family::Mimo2
            | Family::Step35
            | Family::HyV4
            | Family::HyV4Dsa
            | Family::Mistral4
            | Family::GraniteSwaMoe
    );
    if needs_experts {
        kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
        kv!(
            format!("{a}.expert_used_count"),
            Value::U32(N_EXPERT_USED as u32)
        );
        kv!(
            format!("{a}.expert_feed_forward_length"),
            Value::U32(spec.n_ff_exp as u32)
        );
    }
    match spec.family {
        Family::Afmoe => {
            // REQUIRED (afmoe.cpp:5-10)
            kv!(
                format!("{a}.expert_shared_count"),
                Value::U32(spec.n_expert_shared as u32)
            );
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
            // gating absent → SIGMOID default; weights_norm absent → false
        }
        Family::HyV3 | Family::Step35 => {
            kv!(
                format!("{a}.expert_shared_feed_forward_length"),
                Value::U32(spec.n_ff_shexp as u32)
            );
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
        }
        Family::HyV4 | Family::HyV4Dsa => {
            kv!(
                format!("{a}.expert_shared_count"),
                Value::U32(spec.n_expert_shared as u32)
            );
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
            // the sigmoid router (no NONE default in hy-v4.cpp — the real
            // files carry the key)
            kv!(format!("{a}.expert_gating_func"), Value::U32(2));
            // iHC (hy-v4.cpp:41-43 — all REQUIRED)
            kv!(
                format!("{a}.hyper_connection.count"),
                Value::U32(spec.hc_mult as u32)
            );
            kv!(format!("{a}.hyper_connection.epsilon"), Value::F32(1e-6));
            kv!(format!("{a}.hyper_connection.magnitude"), Value::F32(2.0));
            if spec.family == Family::HyV4Dsa {
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
        }
        Family::Mistral4 => {
            // deepseek2's hparams (models.h:1393-1395): V3-style routing
            kv!(
                format!("{a}.expert_shared_count"),
                Value::U32(spec.n_expert_shared as u32)
            );
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
            kv!(format!("{a}.expert_gating_func"), Value::U32(2)); // SIGMOID
            kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
        }
        Family::GraniteSwaMoe => {
            // granite-swa.cpp:13-14/53 — expert_used_count array + shexp
            kv!(
                format!("{a}.expert_used_count"),
                Value::Array(
                    GgufType::Uint32,
                    (0..spec.n_layer)
                        .map(|_| Value::U32(N_EXPERT_USED as u32))
                        .collect()
                )
            );
            kv!(
                format!("{a}.expert_shared_feed_forward_length"),
                Value::U32(spec.n_ff_shexp as u32)
            );
        }
        _ => {}
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
        let n: i64 = ne.iter().product();
        let scale = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = match role {
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            _ => (0..n).map(|_| rng.next() * scale).collect(),
        };
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
        Family::Spark25
        | Family::MuseGlimmer
        | Family::GraniteSwa
        | Family::GraniteSwaMoe
        | Family::Mimo2
        | Family::Step35 => {
            assert_eq!(hp.swa_type, LlamaSwaType::STANDARD, "{}: swa", spec.arch);
            assert_eq!(hp.n_swa, 32);
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_swa(il), il % 4 != 0, "{} is_swa({il})", spec.arch);
            }
        }
        Family::Afmoe => {
            if spec.suffix == "-noswa" {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
            } else {
                assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            }
        }
        Family::Mellum => {
            if spec.suffix == "-swa" {
                assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            } else {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
            }
            assert_eq!(hp.n_expert as i64, N_EXPERT);
        }
        Family::Llada => {
            assert!(!hp.causal_attn, "llada: causal_attn false");
        }
        Family::MuseGlimmer => {
            if spec.suffix == "-softcap" {
                assert_eq!(hp.f_final_logit_softcapping, 0.001, "softcap key");
            } else {
                assert_eq!(hp.f_final_logit_softcapping, 0.0);
            }
        }
        Family::Step35 => {}
        Family::Plm => {
            assert_eq!(hp.n_lora_kv as i64, spec.kv_lora_rank);
        }
        Family::HunyuanVl => {
            if spec.rope_sections.is_some() {
                assert_eq!(hp.rope_sections, [4i32, 4, 4, 0]);
            }
        }
        Family::PaddleOcr => {
            assert_eq!(hp.rope_sections, [4i32, 4, 4, 0]);
        }
        Family::GraniteSwa | Family::GraniteSwaMoe => {
            assert_eq!(hp.f_logit_scale, 8.0);
            assert_eq!(hp.f_embedding_scale, 12.0);
            assert_eq!(hp.f_residual_scale, 0.1);
        }
        Family::Afmoe => {
            assert_eq!(
                hp.expert_gating_func,
                llama::hparams::LlamaExpertGatingFuncType::SIGMOID as u32
            );
            assert_eq!(hp.n_expert_shared as i64, spec.n_expert_shared);
        }
        Family::HyV3 => {
            assert_eq!(
                hp.expert_gating_func,
                llama::hparams::LlamaExpertGatingFuncType::SIGMOID as u32
            );
            assert_eq!(hp.n_ff_shexp as i64, spec.n_ff_shexp);
        }
        Family::Mimo2 => {
            if spec.suffix == "-vscale" {
                assert_eq!(hp.f_attn_value_scale, 0.5);
            } else {
                assert_eq!(hp.f_attn_value_scale, 0.0);
            }
        }
        Family::Step35 => {}
        Family::HyV4 | Family::HyV4Dsa => {
            assert!(hp.is_mla());
            assert_eq!(hp.n_lora_kv as i64, spec.kv_lora_rank);
            assert_eq!(hp.n_lora_q as i64, spec.q_lora_rank);
            assert_eq!(hp.dsv4_hc_mult as i64, spec.hc_mult);
            assert_eq!(hp.hc_magnitude, 2.0);
            if spec.family == Family::HyV4Dsa {
                assert_eq!(hp.indexer_top_k as i64, spec.indexer_top_k);
                assert!(hp.is_indexer_full(0));
                assert!(!hp.is_indexer_full(1));
            } else {
                assert_eq!(hp.indexer_top_k, 0);
            }
        }
        Family::Mistral4 => {
            assert!(hp.is_mla());
            assert_eq!(hp.n_lora_kv as i64, spec.kv_lora_rank);
        }
        Family::GemmaEmbedding => {
            assert_eq!(hp.swa_type, LlamaSwaType::SYMMETRIC);
            assert_eq!(hp.n_swa, spec.sym_swa);
            assert!(!hp.causal_attn);
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_swa(il), il % 6 != 5, "gemma-embedding is_swa({il})");
            }
            let want = 1.0 / (spec.key_length as f32).sqrt();
            assert!((hp.f_attention_scale - want).abs() < 1e-6);
        }
        _ => {}
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
// the driver
// ---------------------------------------------------------------------------

/// map a loaded model to its batch-15 weights bundle (the CLI's derivation)
fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let mut attn = synth_attn(m, fa);
    let per_layer_rope = |hp: &llama::hparams::LlamaHparams| {
        let fb = |il: usize| {
            if hp.is_swa(il) {
                hp.rope_freq_base_train_swa
            } else {
                hp.rope_freq_base_train
            }
        };
        let fs = |il: usize| {
            if hp.is_swa(il) {
                hp.rope_freq_scale_train_swa
            } else {
                hp.rope_freq_scale_train
            }
        };
        (
            (0..hp.n_layer() as usize).map(fb).collect::<Vec<_>>(),
            (0..hp.n_layer() as usize).map(fs).collect::<Vec<_>>(),
        )
    };
    let w = match m.arch {
        llama::arch::LlmArch::QWEN => ForwardWeights::Qwen1(
            batch15_qwen1(m, n_trunk),
            graph_arch::Qwen1Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::MAINCODER => ForwardWeights::Maincoder(
            batch15_maincoder(m, n_trunk),
            graph_arch::MaincoderParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::PANGU_EMBED => ForwardWeights::PanguEmbed(
            batch15_pangu_embed(m, n_trunk),
            graph_arch::PanguEmbedParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::COGVLM => ForwardWeights::Cogvlm(
            batch15_cogvlm(m, n_trunk),
            graph_arch::CogvlmParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::SPARK2_5 => {
            let (fb, fs) = per_layer_rope(hp);
            ForwardWeights::Spark25(
                batch15_spark25(m, n_trunk),
                graph_arch::Spark25Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: fb,
                    freq_scale: fs,
                },
            )
        }
        llama::arch::LlmArch::MUSE_GLIMMER => {
            let (fb, fs) = per_layer_rope(hp);
            ForwardWeights::MuseGlimmer(
                batch15_muse_glimmer(m, n_trunk),
                graph_arch::MuseGlimmerParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: fb,
                    freq_scale: fs,
                    logit_scale: hp.f_logit_scale,
                    final_logit_softcapping: hp.f_final_logit_softcapping,
                },
            )
        }
        llama::arch::LlmArch::LLADA => ForwardWeights::Llada(
            batch15_llada(m, n_trunk),
            graph_arch::LladaParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
            },
        ),
        llama::arch::LlmArch::PLM => {
            let mut a = attn;
            a.n_head_kv = a.n_head; // the repeated-k_pe MHA rows
            attn = a;
            ForwardWeights::Plm(
                batch15_plm(m, n_trunk),
                graph_arch::PlmParams {
                    attn: a,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    kv_lora_rank: hp.n_lora_kv as i64,
                },
            )
        }
        llama::arch::LlmArch::HUNYUAN_VL | llama::arch::LlmArch::HUNYUAN_DENSE => {
            ForwardWeights::HunyuanVl(
                batch15_hunyuan_vl(m, n_trunk),
                graph_arch::HunyuanVlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    use_mrope: hp.use_mrope(),
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::GRANITE_SWA => ForwardWeights::GraniteSwa(
            batch15_granite_swa(m, n_trunk),
            graph_arch::GraniteSwaParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                has_rope: (0..n_trunk).map(|il| hp.has_rope(il)).collect(),
                deepstack_mapping: hp.deepstack_mapping_arr.clone(),
                inp_embd_rows: None,
                f_logit_scale: hp.f_logit_scale,
                f_residual_scale: hp.f_residual_scale,
                f_embedding_scale: hp.f_embedding_scale,
                f_attention_scale: hp.f_attention_scale,
                n_expert: hp.n_expert as i64,
                n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il) as i64).collect(),
                n_ff_shexp: hp.n_ff_shexp as i64,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::AFMOE => {
            let (fb, fs) = per_layer_rope(hp);
            ForwardWeights::Afmoe(
                batch15_afmoe(m, n_trunk),
                graph_arch::AfmoeParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: fb,
                    freq_scale: fs,
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_embd: hp.n_embd as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::MELLUM => ForwardWeights::Mellum(
            batch15_mellum(m, n_trunk),
            graph_arch::MellumParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                is_swa: if hp.swa_type == LlamaSwaType::NONE {
                    Vec::new()
                } else {
                    (0..n_trunk).map(|il| hp.is_swa(il)).collect()
                },
                freq_base_swa: hp.rope_freq_base_train_swa,
                freq_scale_swa: hp.rope_freq_scale_train_swa,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::PADDLEOCR => ForwardWeights::PaddleOcr(
            batch15_paddleocr(m, n_trunk),
            graph_arch::PaddleOcrParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                rope_sections: hp.rope_sections,
            },
        ),
        llama::arch::LlmArch::HY_V3 => ForwardWeights::HyV3(
            batch15_hy_v3(m, n_trunk),
            graph_arch::HyV3Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
                expert_gating_func: hp.expert_gating_func as i32,
            },
        ),
        llama::arch::LlmArch::MIMO2 => {
            let (fb, fs) = per_layer_rope(hp);
            ForwardWeights::Mimo2(
                batch15_mimo2(m, n_trunk),
                graph_arch::Mimo2Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    freq_base: fb,
                    freq_scale: fs,
                    v_scale: hp.f_attn_value_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::STEP35 => {
            let (fb, fs) = per_layer_rope(hp);
            ForwardWeights::Step35(
                batch15_step35(m, n_trunk),
                graph_arch::Step35Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    n_rot: (0..n_trunk).map(|il| hp.n_rot(il) as i64).collect(),
                    freq_base: fb,
                    freq_scale: fs,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::HY_V4 => {
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64; // kv_lora + qk_rope
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            attn = a;
            ForwardWeights::HyV4(
                batch15_hy_v4(m, n_trunk),
                graph_arch::HyV4Params {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    hc_mult: hp.dsv4_hc_mult as i64,
                    hc_eps: hp.dsv4_hc_eps,
                    hc_magnitude: hp.hc_magnitude,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    indexer_top_k: hp.indexer_top_k as i64,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    f_norm_eps: hp.f_norm_eps,
                    is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
                },
            )
        }
        llama::arch::LlmArch::MISTRAL4 => {
            // the deepseek2 weights + params verbatim (models.h:1393-1395)
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64;
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            attn = a;
            ForwardWeights::Deepseek2(
                batch15_mistral4_ds2(m, n_trunk),
                graph_arch::Deepseek2Params {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    rope_yarn_log_mul: hp.rope_yarn_log_mul,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    is_ocr: false,
                },
            )
        }
        other => panic!("no batch-15 driver for {}", other.name()),
    };
    let _ = &mut attn;
    (w, attn)
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

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}

/// the tensor/hparams pins + the double-FA smoke of the parity cells
#[test]
fn arch_batch15_pin_and_smoke() {
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

/// the in-port-only cells: llada (no reference memory), the variants
/// (muse-softcap / granite-moe already in parity; afmoe-swa / mellum-swa /
/// mimo2-vscale), and the encoders' loaders
#[test]
fn arch_batch15_inport_only_cells() {
    for spec in [
        spec_llada(),
        spec_muse_glimmer_softcap(),
        spec_afmoe_noswa(),
        spec_mellum_swa(),
        spec_mimo2_vscale(),
    ] {
        {
            let m = load_synth(&spec);
            pin_hparams(&m, &spec);
            pin_tensors(&m, &spec);
        }
        for fa in [false, true] {
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
        }
        // the no-cache llada has no kv cells to check
        println!(
            "{}{}: in-port cell ok (both FA modes)",
            spec.arch, spec.suffix
        );
    }
}

/// the >64-token prompt cell of the parity protocol, in one ubatch — the
/// iswa members (window 32) cross their windows, step35's halved rope runs
/// past 64, hy-v4's lid rows + top-k evolve
#[test]
fn arch_batch15_long_prompt_cells() {
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

/// the variants must genuinely diverge from their plain files
#[test]
fn arch_batch15_variants_differ() {
    let pairs = [
        (spec_hunyuan_vl(), spec_hunyuan_vl_mrope()),
        (spec_muse_glimmer(), spec_muse_glimmer_softcap()),
        (spec_mimo2(), spec_mimo2_vscale()),
        (spec_hy_v4(), spec_hy_v4_dsa()),
    ];
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
// the encoders (gemma-embedding / llama-embed) — in-port encode smoke; the
// reference comparison is the ignored reference-parity tests below
// ---------------------------------------------------------------------------

/// qwen.cpp:13-37
fn batch15_qwen1(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen1ModelWeights {
    graph_arch::Qwen1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen1LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// maincoder.cpp:12-41
fn batch15_maincoder(m: &LlamaModel, n_trunk: usize) -> graph_arch::MaincoderModelWeights {
    graph_arch::MaincoderModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MaincoderLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// pangu-embed.cpp:13-52
fn batch15_pangu_embed(m: &LlamaModel, n_trunk: usize) -> graph_arch::PanguEmbedModelWeights {
    graph_arch::PanguEmbedModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PanguEmbedLayerWeights {
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
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// cogvlm.cpp:12-47
fn batch15_cogvlm(m: &LlamaModel, n_trunk: usize) -> graph_arch::CogvlmModelWeights {
    graph_arch::CogvlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::CogvlmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                visexp_attn_wqkv: l.visexp_attn_wqkv.unwrap(),
                visexp_attn_wo: l.visexp_attn_wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                visexp_ffn_gate: l.visexp_ffn_gate.unwrap(),
                visexp_ffn_down: l.visexp_ffn_down.unwrap(),
                visexp_ffn_up: l.visexp_ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// spark2-5.cpp:20-50
fn batch15_spark25(m: &LlamaModel, n_trunk: usize) -> graph_arch::Spark25ModelWeights {
    graph_arch::Spark25ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Spark25LayerWeights {
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
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// muse-glimmer.cpp:21-55
fn batch15_muse_glimmer(m: &LlamaModel, n_trunk: usize) -> graph_arch::MuseGlimmerModelWeights {
    graph_arch::MuseGlimmerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MuseGlimmerLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// llada.cpp:19-60
fn batch15_llada(m: &LlamaModel, n_trunk: usize) -> graph_arch::LladaModelWeights {
    graph_arch::LladaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: None,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LladaLayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plm.cpp:13-42
fn batch15_plm(m: &LlamaModel, n_trunk: usize) -> graph_arch::PlmModelWeights {
    graph_arch::PlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PlmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wkv_b: l.wkv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// hunyuan-vl.cpp:23-54 (+ the hunyuan-dense typedef)
fn batch15_hunyuan_vl(m: &LlamaModel, n_trunk: usize) -> graph_arch::HunyuanVlModelWeights {
    graph_arch::HunyuanVlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HunyuanVlLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// granite-swa.cpp:56-115
fn batch15_granite_swa(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteSwaModelWeights {
    graph_arch::GraniteSwaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::GraniteSwaLayerWeights {
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
                attn_sinks: l.attn_sinks.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// afmoe.cpp:38-101
fn batch15_afmoe(m: &LlamaModel, n_trunk: usize) -> graph_arch::AfmoeModelWeights {
    graph_arch::AfmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::AfmoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
            })
            .collect(),
    }
}

/// mellum.cpp:27-64
fn batch15_mellum(m: &LlamaModel, n_trunk: usize) -> graph_arch::MellumModelWeights {
    graph_arch::MellumModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MellumLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
            })
            .collect(),
    }
}

/// paddleocr.cpp (the ernie4_5 tensor set)
fn batch15_paddleocr(m: &LlamaModel, n_trunk: usize) -> graph_arch::PaddleOcrModelWeights {
    graph_arch::PaddleOcrModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PaddleOcrLayerWeights {
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
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// hy-v3.cpp:22-97 (trunk tensors; the MTP block's nextn set is loaded but
/// its graph_mtp is the documented batch-15 skip)
fn batch15_hy_v3(m: &LlamaModel, n_trunk: usize) -> graph_arch::HyV3ModelWeights {
    graph_arch::HyV3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HyV3LayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// mimo2.cpp:25-82 (trunk tensors; graph_mtp is the documented batch-15 skip)
fn batch15_mimo2(m: &LlamaModel, n_trunk: usize) -> graph_arch::Mimo2ModelWeights {
    graph_arch::Mimo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Mimo2LayerWeights {
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
                attn_sinks: l.attn_sinks,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
            })
            .collect(),
    }
}

/// step35.cpp:37-182 (trunk tensors; graph_mtp is the documented batch-15 skip)
fn batch15_step35(m: &LlamaModel, n_trunk: usize) -> graph_arch::Step35ModelWeights {
    graph_arch::Step35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Step35LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                rope_freqs: l.rope_freqs,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wqkv_gate: l.wqkv_gate,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// hy-v4.cpp:70-155
fn batch15_hy_v4(m: &LlamaModel, n_trunk: usize) -> graph_arch::HyV4ModelWeights {
    graph_arch::HyV4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.unwrap(),
        hc_head_base: m.hc_head_base.unwrap(),
        hc_head_scale: m.hc_head_scale.unwrap(),
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HyV4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_sinks: l.attn_sinks.unwrap(),
                wq_a: l.wq_a.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wk_b: l.wk_b.unwrap(),
                wv_b: l.wv_b.unwrap(),
                wo: l.wo.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                indexer_attn_q_b: l.indexer_attn_q_b,
                indexer_attn_k: l.indexer_attn_k,
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                hc_attn_fn: l.hc_attn_fn.unwrap(),
                hc_attn_base: l.hc_attn_base.unwrap(),
                hc_attn_scale: l.hc_attn_scale.unwrap(),
                hc_ffn_fn: l.hc_ffn_fn.unwrap(),
                hc_ffn_base: l.hc_ffn_base.unwrap(),
                hc_ffn_scale: l.hc_ffn_scale.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// the deepseek2 weights bundle for the mistral4 co-arm (models.h:1393-1395)
fn batch15_mistral4_ds2(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek2ModelWeights {
    fn ds2_layer(l: &llama::model::LayerTensors) -> graph_arch::Deepseek2LayerWeights {
        graph_arch::Deepseek2LayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq,
            wqkv: l.wqkv,
            wqkv_b: l.wqkv_b,
            wk: l.wk,
            wv: l.wv,
            wq_a: l.wq_a,
            attn_q_a_norm: l.attn_q_a_norm,
            wq_b: l.wq_b,
            wkv_a_mqa: l.wkv_a_mqa,
            attn_kv_a_norm: l.attn_kv_a_norm,
            indexer_k_norm: l.indexer_k_norm,
            indexer_k_norm_b: l.indexer_k_norm_b,
            indexer_proj: l.indexer_proj,
            indexer_attn_k: l.indexer_attn_k,
            indexer_attn_q_b: l.indexer_attn_q_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wkv_b: l.wkv_b,
            wo: l.wo.unwrap(),
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_gate_up_exps: l.ffn_gate_up_exps,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
        }
    }
    graph_arch::Deepseek2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk].iter().map(ds2_layer).collect(),
    }
}

fn llama_embed_w(m: &LlamaModel) -> graph_arch::LlamaModelWeights {
    let layers = m
        .layers
        .iter()
        .map(|l| graph_arch::LlamaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq.unwrap(),
            wk: l.wk.unwrap(),
            wv: l.wv.unwrap(),
            wo: l.wo.unwrap(),
            wq_b: l.wq_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wo_b: l.wo_b,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate.unwrap(),
            ffn_down: l.ffn_down.unwrap(),
            ffn_up: l.ffn_up.unwrap(),
            ffn_gate_b: l.ffn_gate_b,
            ffn_down_b: l.ffn_down_b,
            ffn_up_b: l.ffn_up_b,
        })
        .collect();
    graph_arch::LlamaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers,
    }
}

#[test]
fn arch_batch15_encoder_smoke() {
    for spec in [spec_gemma_embedding(), spec_llama_embed()] {
        let mut m = load_synth(&spec);
        pin_hparams(&m, &spec);
        pin_tensors(&m, &spec);
        let hp = m.hparams.clone();
        let rope = hp.rope_runtime();
        let gr = graph_arch::EurobertRope {
            n_rot: hp.n_rot(0) as i32,
            rope_mode: hp.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        };
        let gemma_swa = (m.arch == llama::arch::LlmArch::GEMMA_EMBEDDING).then(|| {
            (
                graph_arch::GemmaEmbeddingSwa {
                    is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                    n_swa: hp.n_swa,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
                gr,
                hp.f_attention_scale,
            )
        });
        // llama_encode forces causal_attn = false (llama-context.cpp:1526-1529)
        let causal = false;
        let params = graph_arch::EncoderParams {
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head: hp.n_embd_head_k(0) as i64,
            n_rel_attn_bkts: hp.n_rel_attn_bkts,
            f_norm_eps: hp.f_norm_eps,
            f_norm_rms_eps: hp.f_norm_rms_eps,
            pool: llama::hparams::LlamaPoolingType::NONE,
            euro_rope: Some(gr),
            gemma_swa,
            causal,
        };
        let enc_w = match m.arch {
            llama::arch::LlmArch::GEMMA_EMBEDDING => {
                EncoderWeights::GemmaEmbedding(m.gemma_embedding_weights())
            }
            _ => EncoderWeights::LlamaEmbed(llama_embed_w(&m)),
        };
        let gctx = std::mem::replace(&mut m.ctx, Context::new());
        let mut enc = EncoderContext::new(gctx, enc_w, params, 8);
        let tokens: Vec<i32> = (1..=12).collect();
        let emb = enc.encode(&tokens).expect("encode");
        assert_eq!(emb.n_rows, 12, "{}: per-token rows", spec.arch);
        assert_eq!(emb.n_embd_out, spec.n_embd as usize, "{}: width", spec.arch);
        assert!(
            emb.values.iter().all(|v| v.is_finite()),
            "{}: non-finite embeddings",
            spec.arch
        );
        println!("{}{}: encoder smoke ok", spec.arch, spec.suffix);
    }
}

/// the reference `llama_encode` dumps (the bert/t5/eurobert protocol):
/// `parity/ref_encode_dump <file> <out> --ids … --pool none --fa off`
#[test]
#[ignore = "needs parity/ref_encode_dump (the reference libllama dumper)"]
fn arch_batch15_encoder_reference_parity() {
    for spec in [spec_gemma_embedding(), spec_llama_embed()] {
        let model_path = spec.path();
        if !std::path::Path::new(&model_path).exists() {
            let _g = build_lock();
            if !std::path::Path::new(&model_path).exists() {
                build_file(&spec);
            }
        }
        let dump = format!("{OUT_DIR}/{}-enc-ref.bin", spec.arch);
        let dumper = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/ref_encode_dump");
        assert!(
            std::path::Path::new(dumper).exists(),
            "{dumper} missing — build it with parity/gen_encode_ref.sh"
        );
        let ids = "1,2,3,4,5,6,7,8,9,10,11,12";
        let st = std::process::Command::new(dumper)
            .arg(&model_path)
            .arg(&dump)
            .arg("--ids")
            .arg(ids)
            .arg("--pool")
            .arg("none")
            .arg("--fa")
            .arg("off")
            .status()
            .expect("run ref_encode_dump");
        if !st.success() {
            // gemma-embedding: the pinned reference's own llama_encode
            // crashes (GGML_ASSERT(buffer), ggml-backend.cpp:205 — the
            // symmetric-SWA no-cache mask twin never gets a backend buffer
            // on the encoder path); there is no reference output to
            // compare against — the port stays in-port verified
            // (arch_batch15_encoder_smoke)
            assert_eq!(
                spec.arch,
                "gemma-embedding",
                "ref_encode_dump failed (only the gemma-embedding dumper crash is the documented skip)"
            );
            eprintln!(
                "gemma-embedding: reference llama_encode crashes (GGML_ASSERT(buffer)) \
                 — documented skip, see PARITY.md 批次 15"
            );
            continue;
        }

        let bytes = std::fs::read(&dump).expect("read dump");
        assert_eq!(&bytes[..8], b"LENCE1\0\0", "dump magic");
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let n_tokens = rd_u32(8) as usize;
        let n_embd_out = rd_u32(12) as usize;
        let n_rows = rd_u32(16) as usize;
        assert_eq!(n_tokens, 12);
        assert_eq!(n_rows, 12, "pooling NONE keeps every row");
        let hdr = 8 + 4 * 4;
        let ref_vals_off = hdr + 4 * n_tokens;
        let ref_vals: Vec<f32> = bytes[ref_vals_off..ref_vals_off + 4 * n_embd_out * n_rows]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // the port side (same construction as the smoke test)
        let mut m = load_synth(&spec);
        let hp = m.hparams.clone();
        let rope = hp.rope_runtime();
        let gr = graph_arch::EurobertRope {
            n_rot: hp.n_rot(0) as i32,
            rope_mode: hp.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        };
        let gemma_swa = (m.arch == llama::arch::LlmArch::GEMMA_EMBEDDING).then(|| {
            (
                graph_arch::GemmaEmbeddingSwa {
                    is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                    n_swa: hp.n_swa,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
                gr,
                hp.f_attention_scale,
            )
        });
        // llama_encode forces causal_attn = false (llama-context.cpp:1526-1529)
        let causal = false;
        let params = graph_arch::EncoderParams {
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head: hp.n_embd_head_k(0) as i64,
            n_rel_attn_bkts: hp.n_rel_attn_bkts,
            f_norm_eps: hp.f_norm_eps,
            f_norm_rms_eps: hp.f_norm_rms_eps,
            pool: llama::hparams::LlamaPoolingType::NONE,
            euro_rope: Some(gr),
            gemma_swa,
            causal,
        };
        let enc_w = match m.arch {
            llama::arch::LlmArch::GEMMA_EMBEDDING => {
                EncoderWeights::GemmaEmbedding(m.gemma_embedding_weights())
            }
            _ => EncoderWeights::LlamaEmbed(llama_embed_w(&m)),
        };
        let gctx = std::mem::replace(&mut m.ctx, Context::new());
        let mut enc = EncoderContext::new(gctx, enc_w, params, 8);
        let tokens: Vec<i32> = (1..=12).collect();
        let emb = enc.encode(&tokens).expect("encode");

        assert_eq!(emb.values.len(), ref_vals.len(), "{}: row count", spec.arch);
        let mut worst = 0f32;
        for (a, b) in emb.values.iter().zip(&ref_vals) {
            worst = worst.max((a - b).abs());
        }
        assert_eq!(worst, 0.0, "{}: max |Δembd| vs the reference", spec.arch);
        println!(
            "{}{}: reference encode parity exact",
            spec.arch, spec.suffix
        );
    }
}

// ---------------------------------------------------------------------------
// the node-dump probe (the qwen3_prefill_dump protocol): dump one prefill's
// every graph node so parity/decode_dump_cmp.py can localize a divergence
// against parity/ref_decode_dump — env B15_DUMP_ARCH selects the file
// ---------------------------------------------------------------------------

use ggml::compute::{set_eval_callback, EvalNode};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

static DUMP: OnceLock<Mutex<Option<Vec<u8>>>> = OnceLock::new();
static NODES: AtomicU32 = AtomicU32::new(0);

fn op_desc(op: ggml::GgmlOp) -> &'static str {
    use ggml::GgmlOp::*;
    // base names only: the port encodes RMS_NORM as Norm+flag and the unary
    // variants (SILU/GELU/...) as Silu+params — the comparator aligns by
    // index/shape/name, the op string is for display
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
        // arch batch 7 (deepseek4) ops — the deepseek4 dump lives in
        // arch_batch7_e2e.rs with the full UNARY/GLU-aware op_desc
        Sqrt => "SQRT",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        // arch batch 10 (graniteswitch) — the router lane's right-pad
        Pad => "PAD",
        // display-only arms for enum variants added after this file (neither
        // op appears in a qwen3/gpt-oss graph — exhaustiveness only)
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
        // audio/mean rounds' later variants — display only, exhaustiveness
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn type_desc(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Bf16 => "bf16",
        GgmlType::I64 => "i64",
        GgmlType::I32 => "i32",
        GgmlType::I16 => "i16",
        GgmlType::I8 => "i8",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q5_1 => "q5_1",
        GgmlType::Q8_1 => "q8_1",
        _ => "other",
    }
}

fn b15_dump_cb(node: &EvalNode<'_>, ask: bool) -> bool {
    use std::fmt::Write as _;
    if ask {
        return true;
    }
    let mut guard = DUMP.get().unwrap().lock().unwrap();
    let Some(out) = guard.as_mut() else { return true };
    let n: i64 = node.ne.iter().product();
    NODES.fetch_add(1, Ordering::Relaxed);
    let mut put = |s: &str| {
        let l = s.len().min(255);
        out.push(l as u8);
        out.extend_from_slice(&s.as_bytes()[..l]);
    };
    put(op_desc(node.op));
    put(node.name);
    put(type_desc(node.ty));
    out.extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= (1 << 19) {
        return true; // no payload (the same shape rule as the C probe)
    }
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        // the C probe's zero payload for non-float nodes
        out.extend(std::iter::repeat(0u8).take(4 * n as usize));
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
            half::f16::from_le_bytes([data[off], data[off + 1]]).to_f32()
        };
        let _ = std::write!(String::new(), "");
        out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

#[test]
#[ignore = "debug probe: B15_DUMP_ARCH=<arch> cargo test -- --ignored arch_batch15_prefill_node_dump"]
fn arch_batch15_prefill_node_dump() {
    let arch = std::env::var("B15_DUMP_ARCH").expect("B15_DUMP_ARCH");
    let spec = all_specs().into_iter().find(|s| s.arch == arch).expect("spec");
    let out_path =
        std::env::var("B15_DUMP_OUT").unwrap_or_else(|_| format!("/tmp/b15-{arch}-port.bin"));
    let fa_off = std::env::var("B15_FA_OFF").is_ok();

    set_eval_callback(Some(b15_dump_cb));
    DUMP.get_or_init(|| Mutex::new(Some(Vec::new())));
    // "a b c d e f" through the SPM tokenizer (add_special=true) — the same
    // token stream parity/ref_decode_dump produces
    let ids: Vec<i32> = if std::env::var("B15_DUMP_IDS").is_ok() {
        std::env::var("B15_DUMP_IDS").unwrap().split(',').map(|v| v.parse().unwrap()).collect()
    } else {
        vec![1, 263, 289, 274, 270, 321, 285]
    };
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    let mut m = load_synth(&spec);
    let mut dctx = driver_for(&mut m, !fa_off);
    // decode_all: no out_ids gather — the dump compares the reference's
    // all-rows embeddings shape (ref_decode_dump runs --embeddings)
    let _ = dctx.decode_all(&ids, &pos).expect("prefill");
    // B15_DECODE_IDS: append single-token decode steps to the same node
    // stream — the mirror of ref_decode_dump's --decode-tail/--decode-ids
    // (the mimo2/step35 decode-step bisect)
    if let Ok(tail) = std::env::var("B15_DECODE_IDS") {
        let feed: Vec<i32> =
            tail.split(',').map(|v| v.parse().unwrap()).collect();
        for (s, &t) in feed.iter().enumerate() {
            let p = (ids.len() + s) as i32;
            let _ = dctx.decode(&[t], &[p]).expect("decode step");
        }
    }
    set_eval_callback(None);
    let st = DUMP.get().unwrap().lock().unwrap().take().unwrap();
    let n_nodes = NODES.load(Ordering::Relaxed);
    let mut out = b"DECDMP1\0".to_vec();
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &t in &ids {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&n_nodes.to_le_bytes());
    out.extend_from_slice(&st);
    std::fs::write(&out_path, &out).unwrap();
    println!("b15 dump: {arch} nodes={n_nodes} -> {out_path}");
}
