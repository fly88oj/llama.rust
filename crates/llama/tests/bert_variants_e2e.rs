//! bert_variants_e2e.rs — the BERT-variant encoder family (arch batch 16):
//! **jina-bert-v2 / jina-bert-v3 / nomic-bert / nomic-bert-moe / neo-bert /
//! modern-bert** (src/models/*.cpp, llama.cpp bd4f514db1).
//!
//! The four `jina/nomic` archs instantiate the *same* `llama_model_bert::graph`
//! body with their arch-keyed branches live (models.h `using graph =`);
//! neo-bert and modern-bert carry their own graphs, and modern-bert adds the
//! GTE reranker head read by RANK pooling (llama-graph.cpp:3722-3766).
//!
//! Protocol (the batch-15 encoder precedent — gemma-embedding / llama-embed):
//! a *synthetic* GGUF per variant, built with the port's byte-exact GGUF
//! writer (`tokenizer.*` KV copied verbatim from the llama SPM vocab fixture,
//! the arch's own KV, exactly the tensor names + shapes its load_arch_tensors
//! asks for, all F32), then
//!
//!   * default-run pinning tests — loader tensor map + hparams + the built
//!     (not computed) graph shape + a full finite forward;
//!   * `#[ignore]`d reference parity — `parity/ref_encode_dump` (the reference
//!     libllama's own `llama_encode`) over the same synthetic file, embedding
//!     rows compared **bit-exact** (max |Δ| == 0.0), the bert/t5/eurobert
//!     protocol. The dumps anchor FA *off*: the port's encoder baseline is the
//!     non-FA branch of build_attn_mha (no `ggml_flash_attn_ext` / F16 cast in
//!     the port, the bert_e2e.rs precedent), and this family has no kq_b to
//!     disable FA on the reference side either.
//!
//! modern-bert is verified twice: the plain embedder file (`--pool none` /
//! `--pool mean`) and the reranker file (`--pool rank` — the classification
//! head: mean → cls → GELU → cls_norm → cls_out, [n_cls_out] result).
//! modern-bert `-silu` swaps `hidden_activation = silu` in (the granite
//! derivative path: SwiGLU instead of GeGLU).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{EncoderContext, EncoderWeights};
use llama::graph_arch;
use llama::hparams::LlamaPoolingType as P;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/bert-variants";
const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    JinaV2,
    JinaV2Gated,
    JinaV3,
    Nomic,
    NomicMoe,
    NeoBert,
    ModernBert,
    /// modern-bert + the GTE reranker head (cls / cls_norm / cls_out) — the
    /// RANK-pooling file
    ModernBertRank,
    /// modern-bert with `hidden_activation = silu` (the granite derivative)
    ModernBertSilu,
}

struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    /// jina-v2 loads full-width q/k norms; nomic needs its gate; the moe file
    /// interleaves expert layers (moe_every_n_layers = 2 → layers 1, 3)
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    /// jina-bert-v2's folded-in gate (ffn_up 2*n_ff wide, no ffn_gate tensor)
    jina_folded_gate: bool,
}

impl SynthSpec {
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut s = Self {
            arch: self.arch,
            suffix: self.suffix,
            family: self.family,
            n_layer: self.n_layer,
            n_embd: self.n_embd,
            n_head: self.n_head,
            rope_dim: self.rope_dim,
            n_ff: self.n_ff,
            n_ctx: self.n_ctx,
            jina_folded_gate: self.jina_folded_gate,
        };
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
        rope_dim: 16,
        n_ff: 48,
        n_ctx: 256,
        jina_folded_gate: false,
    }
}

/// jina-bert-v2 — ALiBi (8.0), full-width q/k norms, attn_norm_2, the
/// folded-in GEGLU gate, required wo_b / ffn_down_b
fn spec_jina_v2() -> SynthSpec {
    base("jina-bert-v2", Family::JinaV2).with(|s| {
        s.jina_folded_gate = true;
    })
}

/// jina-bert-v2 `-gated` — the separate ffn_gate tensor (GEGLU split form)
fn spec_jina_v2_gated() -> SynthSpec {
    spec_jina_v2().with(|s| {
        s.suffix = "-gated";
        s.family = Family::JinaV2Gated;
        s.jina_folded_gate = false;
    })
}

fn spec_jina_v3() -> SynthSpec {
    base("jina-bert-v3", Family::JinaV3)
}

fn spec_nomic() -> SynthSpec {
    base("nomic-bert", Family::Nomic)
}

/// nomic-bert-moe — `moe_every_n_layers = 2`: layers 1 and 3 are MoE
/// (il % 2 == 1), layers 0 and 2 dense GELU-SEQ
fn spec_nomic_moe() -> SynthSpec {
    base("nomic-bert-moe", Family::NomicMoe)
}

fn spec_neo_bert() -> SynthSpec {
    base("neo-bert", Family::NeoBert)
}

/// modern-bert — the plain embedder (sliding_window 8 + the every-3rd dense
/// pattern, layer-0 identity attn_norm, GEGLU FFN)
fn spec_modern_bert() -> SynthSpec {
    base("modern-bert", Family::ModernBert)
}

/// modern-bert `-rank` — the GTE reranker head (+ classifier labels so
/// n_cls_out = 2)
fn spec_modern_bert_rank() -> SynthSpec {
    spec_modern_bert().with(|s| {
        s.suffix = "-rank";
        s.family = Family::ModernBertRank;
    })
}

/// modern-bert `-silu` — `hidden_activation = silu` → SwiGLU
fn spec_modern_bert_silu() -> SynthSpec {
    spec_modern_bert().with(|s| {
        s.suffix = "-silu";
        s.family = Family::ModernBertSilu;
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_jina_v2(),
        spec_jina_v2_gated(),
        spec_jina_v3(),
        spec_nomic(),
        spec_nomic_moe(),
        spec_neo_bert(),
        spec_modern_bert(),
        spec_modern_bert_rank(),
        spec_modern_bert_silu(),
    ]
}

#[test]
#[ignore = "writes the /tmp/bert-variants parity files (ARCH_BATCH16 cells)"]
fn bert_variants_write_synth() {
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
    NormBias,
    Bias,
    Proj,
    Router,
}

fn tensors_for(spec: &SynthSpec) -> Vec<((String, Vec<i64>), Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let mut t: Vec<((String, Vec<i64>), Role)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, role: Role| t.push(((name, ne), role));

    push("token_embd.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);

    match spec.family {
        Family::JinaV2 | Family::JinaV2Gated => {
            // jina-bert-v2.cpp:17-24 — type_embd REQUIRED, cls {n_embd, 1}
            push(
                "token_types.weight".into(),
                vec![n_embd, 1],
                Role::Proj,
            );
            push("cls.weight".into(), vec![n_embd, 1], Role::Proj);
            push("cls.bias".into(), vec![1], Role::Bias);
            push("token_embd_norm.weight".into(), vec![n_embd], Role::Norm);
            push("token_embd_norm.bias".into(), vec![n_embd], Role::NormBias);
        }
        Family::JinaV3 | Family::Nomic | Family::NomicMoe => {
            push("token_types.weight".into(), vec![n_embd, 1], Role::Proj);
            push("token_embd_norm.weight".into(), vec![n_embd], Role::Norm);
            push("token_embd_norm.bias".into(), vec![n_embd], Role::NormBias);
        }
        Family::NeoBert => {
            // neo-bert.cpp:22 — enc.output_norm, no token types / embedding LN
            push("enc.output_norm.weight".into(), vec![n_embd], Role::Norm);
        }
        Family::ModernBert | Family::ModernBertRank | Family::ModernBertSilu => {
            // modern-bert.cpp:38/40 — the weight-only embedding LN + the
            // final output norm
            push("token_embd_norm.weight".into(), vec![n_embd], Role::Norm);
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
        }
    }

    for i in 0..spec.n_layer as i32 {
        // create_tensor_qkv picks the separate triple when no fused
        // attn_qkv.weight exists (jina/nomic); neo/modern require the fused one
        let qkv_separate = |push: &mut dyn FnMut(String, Vec<i64>, Role)| {
            push(format!("blk.{i}.attn_q.weight"), vec![n_embd, n_embd], Role::Proj);
            push(format!("blk.{i}.attn_k.weight"), vec![n_embd, n_embd], Role::Proj);
            push(format!("blk.{i}.attn_v.weight"), vec![n_embd, n_embd], Role::Proj);
        };
        match spec.family {
            Family::JinaV2 | Family::JinaV2Gated => {
                qkv_separate(&mut push);
                // jina-bert-v2.cpp:30-34 — the FULL-width q/k norms + biases
                push(format!("blk.{i}.attn_q_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_q_norm.bias"), vec![n_embd], Role::NormBias);
                push(format!("blk.{i}.attn_k_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_k_norm.bias"), vec![n_embd], Role::NormBias);
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.attn_output.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.attn_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_output_norm.bias"), vec![n_embd], Role::NormBias);
                // attn_norm_2 (jina-bert-v2.cpp:42-43) — layer 1 only, so the
                // Some/None branch of the graph is exercised
                if i == 1 {
                    push(format!("blk.{i}.attn_norm_2.weight"), vec![n_embd], Role::Norm);
                    push(format!("blk.{i}.attn_norm_2.bias"), vec![n_embd], Role::NormBias);
                }
                if !spec.jina_folded_gate {
                    push(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], Role::Proj);
                }
                // jina-bert-v2.cpp:47-53 — up is 2*n_ff when the gate is folded in
                let n_up = if spec.jina_folded_gate { 2 * n_ff } else { n_ff };
                push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_up], Role::Proj);
                push(format!("blk.{i}.ffn_up.bias"), vec![n_up], Role::Bias);
                push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
                push(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.layer_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.layer_output_norm.bias"), vec![n_embd], Role::NormBias);
            }
            Family::JinaV3 => {
                qkv_separate(&mut push);
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.attn_output.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.attn_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_output_norm.bias"), vec![n_embd], Role::NormBias);
                push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                push(format!("blk.{i}.ffn_up.bias"), vec![n_ff], Role::Bias);
                push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
                push(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.layer_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.layer_output_norm.bias"), vec![n_embd], Role::NormBias);
            }
            Family::Nomic => {
                qkv_separate(&mut push);
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.attn_output.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.attn_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_output_norm.bias"), vec![n_embd], Role::NormBias);
                push(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], Role::Proj);
                push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
                push(format!("blk.{i}.layer_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.layer_output_norm.bias"), vec![n_embd], Role::NormBias);
            }
            Family::NomicMoe => {
                qkv_separate(&mut push);
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.attn_output.bias"), vec![n_embd], Role::Bias);
                push(format!("blk.{i}.attn_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.attn_output_norm.bias"), vec![n_embd], Role::NormBias);
                if i % 2 == 1 {
                    // the MoE layers (nomic-bert-moe.cpp:38-40)
                    push(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, n_ff, N_EXPERT],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![n_ff, n_embd, N_EXPERT],
                        Role::Proj,
                    );
                    push(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, N_EXPERT],
                        Role::Router,
                    );
                } else {
                    push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                    push(format!("blk.{i}.ffn_up.bias"), vec![n_ff], Role::Bias);
                    push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
                    push(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                }
                push(format!("blk.{i}.layer_output_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.layer_output_norm.bias"), vec![n_embd], Role::NormBias);
            }
            Family::NeoBert => {
                // neo-bert.cpp:27-35 — fused QKV (n_embd + 2*n_embd_gqa),
                // RMS norms, the 2*n_ff-wide up
                push(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, 3 * n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, 2 * n_ff], Role::Proj);
                push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
            }
            Family::ModernBert | Family::ModernBertRank | Family::ModernBertSilu => {
                // modern-bert.cpp:42-59 — layer 0's attn_norm optional; skip it
                // there so the identity branch is exercised
                if i != 0 {
                    push(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                }
                push(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, 3 * n_embd],
                    Role::Proj,
                );
                push(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], Role::Proj);
                push(format!("blk.{i}.ffn_up.weight"), vec![n_embd, 2 * n_ff], Role::Proj);
                push(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
                push(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
            }
        }
    }

    // the GTE reranker head (modern-bert.cpp:61-64). CLS_OUT's arch-wide file
    // name is "cls.output" (llama-arch.cpp:483), NOT cls_out — the jina-v2
    // reranker's cls {n_embd, 1} is LLM_TENSOR_CLS ("cls") with cls_b {1}
    if spec.family == Family::ModernBertRank {
        push("cls.output.weight".into(), vec![n_embd, 2], Role::Proj);
        push("cls.output.bias".into(), vec![2], Role::Bias);
        push("cls.weight".into(), vec![n_embd, n_embd], Role::Proj);
        push("cls.norm.weight".into(), vec![n_embd], Role::Norm);
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
        Role::Norm | Role::NormBias => 1.0,
        Role::Bias => 0.02,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        Role::Router => 1.0 / (n_embd as f32).sqrt(),
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/bert-variants");

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
    // bert.cpp:26 reads the model-agnostic tokenizer key (not {arch}-prefixed)
    kv!("tokenizer.ggml.token_type_count", Value::U32(1));
    kv!(
        format!("{a}.feed_forward_length"),
        Value::U32(spec.n_ff as u32)
    );
    kv!(
        format!("{a}.attention.head_count"),
        Value::U32(spec.n_head as u32)
    );
    // MHA — no head_count_kv key, n_head_kv falls back to n_head
    match spec.family {
        Family::JinaV2 | Family::JinaV2Gated | Family::JinaV3 | Family::Nomic | Family::NomicMoe => {
            kv!(
                format!("{a}.attention.layer_norm_epsilon"),
                Value::F32(1e-5)
            );
        }
        Family::NeoBert => {
            kv!(
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-6)
            );
        }
        Family::ModernBert | Family::ModernBertRank | Family::ModernBertSilu => {
            // modern-bert.cpp:4-11 — sliding_window + SYMMETRIC swa; keep the
            // window small so the 12-token prompts bind it
            kv!(format!("{a}.attention.sliding_window"), Value::U32(8));
            kv!(
                format!("{a}.attention.layer_norm_epsilon"),
                Value::F32(1e-5)
            );
            // modern-bert.cpp:15-21 — GEGLU default, hidden_activation override
            if spec.family == Family::ModernBertSilu {
                kv!(format!("{a}.hidden_activation"), Value::String("silu".into()));
            }
            // the reranker's classifier labels (n_cls_out = 2)
            if spec.family == Family::ModernBertRank {
                kv!(
                    format!("{a}.classifier.output_labels"),
                    Value::Array(
                        GgufType::String,
                        vec![
                            Value::String("yes".into()),
                            Value::String("no".into())
                        ]
                    )
                );
            }
        }
    }
    if !matches!(spec.family, Family::JinaV2) {
        // the rope'd variants (nomic / jina-v3 / neo / modern); jina-bert-v2
        // has no rope at all (rope_type NONE)
        kv!(
            format!("{a}.rope.dimension_count"),
            Value::U32(spec.rope_dim as u32)
        );
        kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    }
    if spec.family == Family::NomicMoe {
        kv!(format!("{a}.moe_every_n_layers"), Value::U32(2));
        kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
        kv!(
            format!("{a}.expert_used_count"),
            Value::U32(N_EXPERT_USED as u32)
        );
        kv!(
            format!("{a}.expert_weights_scale"),
            Value::F32(1.0)
        );
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
        "{}{}: consumed tensor count (loaded {} vs table {})",
        spec.arch,
        spec.suffix,
        m.tensors.len(),
        table.len()
    );
    for ((name, ne), _) in &table {
        let id = *m
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("{}{}: tensor {name} not loaded", spec.arch, spec.suffix));
        let got = m.ctx.ne(id);
        let mut want = [1i64; 4];
        for (i, &d) in ne.iter().take(4).enumerate() {
            want[i] = d;
        }
        assert_eq!(got[..], want[..], "{}{}: {name} shape", spec.arch, spec.suffix);
    }
}

/// the hparams pins: the geometry + the arch-specific KV each graph derives
/// its branches from
fn pin_hparams(m: &LlamaModel, spec: &SynthSpec) {
    let hp = &m.hparams;
    assert_eq!(
        hp.n_layer() as usize,
        spec.n_layer,
        "{}{}: n_layer",
        spec.arch,
        spec.suffix
    );
    assert_eq!(hp.n_embd as i64, spec.n_embd, "{}{}: n_embd", spec.arch, spec.suffix);
    assert_eq!(hp.n_head_kv(0), hp.n_head(0), "{}{}: MHA", spec.arch, spec.suffix);
    match spec.family {
        Family::JinaV2 | Family::JinaV2Gated => {
            // jina-bert-v2.cpp:5 — the ALiBi bias flips use_alibi
            // (llama-model.cpp:1419-1421)
            assert_eq!(hp.f_max_alibi_bias, 8.0);
            assert!(hp.use_alibi, "jina-bert-v2: use_alibi");
            assert_eq!(hp.rope_type, llama::hparams::LlamaRopeType::NONE);
        }
        Family::JinaV3 | Family::Nomic | Family::NomicMoe => {
            assert_eq!(
                hp.rope_type,
                llama::hparams::LlamaRopeType::NEOX,
                "{}{}: rope type",
                spec.arch,
                spec.suffix
            );
            // (NomicMoe's own geometry pins — a match arm can't re-match an
            // earlier or-pattern's member, so they ride this arm's body)
            if spec.family == Family::NomicMoe {
                assert_eq!(hp.moe_every_n_layers, 2);
                assert_eq!(hp.n_expert as i64, N_EXPERT);
                assert_eq!(hp.n_expert_used(0) as i64, N_EXPERT_USED);
            }
        }
        Family::NeoBert => {
            assert_eq!(
                hp.rope_type,
                llama::hparams::LlamaRopeType::NORM,
                "neo-bert: rope type"
            );
        }
        Family::ModernBert | Family::ModernBertRank | Family::ModernBertSilu => {
            // modern-bert.cpp:4-11 — SYMMETRIC swa, pattern 3 dense-first
            assert_eq!(hp.swa_type, llama::hparams::LlamaSwaType::SYMMETRIC);
            assert_eq!(hp.n_swa, 8);
            for il in 0..spec.n_layer {
                assert_eq!(
                    hp.is_swa(il),
                    il % 3 != 0,
                    "modern-bert is_swa({il}) — every 3rd layer dense, first"
                );
            }
            if spec.family == Family::ModernBertSilu {
                assert_eq!(
                    hp.llm_ffn_op,
                    llama::hparams::LlmFfnOpType::SWIGLU,
                    "hidden_activation = silu"
                );
            } else {
                assert_eq!(
                    hp.llm_ffn_op,
                    llama::hparams::LlmFfnOpType::GEGLU,
                    "the GEGLU default"
                );
            }
            if spec.family == Family::ModernBertRank {
                assert_eq!(hp.n_cls_out, 2, "classifier.output_labels");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the encoder driver (the same construction the reference's llama_encode
// reaches these graphs with)
// ---------------------------------------------------------------------------

fn encoder(mut m: LlamaModel, pool: P) -> EncoderContext {
    let hp = m.hparams.clone();
    let rope = hp.rope_runtime();
    // the rope facts of the rope'd variants (ggml_rope_ext at bert.cpp:126-133
    // / neo-bert.cpp:73-83 / modern-bert.cpp:112-122)
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
    // llama_encode forces causal_attn = false (llama-context.cpp:1526-1529)
    let params = graph_arch::EncoderParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: llama::context::resolve_pooling(pool, hp.pooling_type),
        euro_rope: Some(gr),
        gemma_swa: None,
        causal: false,
    };
    let weights = match m.arch {
        llama::arch::LlmArch::JINA_BERT_V2 => {
            let vp = graph_arch::BertVariantParams {
                max_alibi_bias: hp.f_max_alibi_bias,
                moe_every_n_layers: 0,
                n_expert: 0,
                n_expert_used: 0,
                expert_weights_scale: 0.0,
                n_ff: hp.n_ff(0) as i64,
            };
            EncoderWeights::BertVariant(m.bert_variant_weights(graph_arch::BertVariant::JinaV2), vp)
        }
        llama::arch::LlmArch::JINA_BERT_V3 => {
            let vp = graph_arch::BertVariantParams {
                max_alibi_bias: hp.f_max_alibi_bias,
                moe_every_n_layers: 0,
                n_expert: 0,
                n_expert_used: 0,
                expert_weights_scale: 0.0,
                n_ff: hp.n_ff(0) as i64,
            };
            EncoderWeights::BertVariant(m.bert_variant_weights(graph_arch::BertVariant::JinaV3), vp)
        }
        llama::arch::LlmArch::NOMIC_BERT => {
            let vp = graph_arch::BertVariantParams {
                max_alibi_bias: hp.f_max_alibi_bias,
                moe_every_n_layers: 0,
                n_expert: 0,
                n_expert_used: 0,
                expert_weights_scale: 0.0,
                n_ff: hp.n_ff(0) as i64,
            };
            EncoderWeights::BertVariant(m.bert_variant_weights(graph_arch::BertVariant::Nomic), vp)
        }
        llama::arch::LlmArch::NOMIC_BERT_MOE => {
            let vp = graph_arch::BertVariantParams {
                max_alibi_bias: hp.f_max_alibi_bias,
                moe_every_n_layers: hp.moe_every_n_layers,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_weights_scale: hp.expert_weights_scale,
                n_ff: hp.n_ff(0) as i64,
            };
            EncoderWeights::BertVariant(
                m.bert_variant_weights(graph_arch::BertVariant::NomicMoe),
                vp,
            )
        }
        llama::arch::LlmArch::NEO_BERT => EncoderWeights::NeoBert(m.neo_bert_weights()),
        _ => {
            let mp = graph_arch::ModernBertParams {
                swa: graph_arch::ModernBertSwa {
                    is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                    n_swa: hp.n_swa,
                    freq_base: hp.rope_freq_base_train,
                    freq_scale: hp.rope_freq_scale_train,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
                ffn_op: hp.llm_ffn_op,
            };
            EncoderWeights::ModernBert(m.modern_bert_weights(), mp)
        }
    };
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    EncoderContext::new(gctx, weights, params, 8)
}

// ---------------------------------------------------------------------------
// default-run tests
// ---------------------------------------------------------------------------

/// Loader + hparams + graph-shape + finite-forward pins for every file.
#[test]
fn bert_variants_loader_and_forward() {
    for spec in all_specs() {
        let m = load_synth(&spec);
        pin_hparams(&m, &spec);
        pin_tensors(&m, &spec);
        let tag = format!("{}{}", spec.arch, spec.suffix);

        // the built graph (no compute): per-token rows for NONE
        let pool = if spec.family == Family::ModernBertRank { P::RANK } else { P::NONE };
        let mut ctx = encoder(m, pool);
        let tokens: Vec<i32> = (1..=12).collect();
        let g = ctx.build(&tokens).expect("build");
        match pool {
            P::NONE => {
                assert_eq!(
                    *ctx.gctx.ne(g.embd),
                    [spec.n_embd, 12, 1, 1],
                    "{tag}: last hidden state shape"
                );
            }
            P::RANK => {
                // the pooled row is the [n_cls_out] rerank score
                assert_eq!(*ctx.gctx.ne(g.pooled), [2, 1, 1, 1], "{tag}: rank row");
            }
            _ => unreachable!(),
        }
        assert_eq!(g.layer_outs.len(), spec.n_layer, "{tag}: per-layer outputs");

        // the full forward
        let emb = ctx.encode(&tokens).expect("encode");
        assert!(emb.values.iter().all(|v| v.is_finite()), "{tag}: non-finite");
        match pool {
            P::NONE => {
                assert_eq!(emb.n_rows, 12, "{tag}: per-token rows");
                assert_eq!(emb.n_embd_out, spec.n_embd as usize, "{tag}: width");
            }
            P::RANK => {
                assert_eq!(emb.n_rows, 1, "{tag}: one rank row");
                assert_eq!(emb.n_embd_out, 2, "{tag}: n_cls_out-wide rank row");
            }
            _ => unreachable!(),
        }
        println!("{tag}: encoder ok ({} values)", emb.values.len());
    }
}

/// Structural signature of the variant branches — the op mix that separates
/// each family from the plain bert graph (rope pairs, the ALiBi mask fill,
/// MoE ops, the fused GLUs, the swa twin, the rank head tail).
#[test]
fn bert_variants_graph_structure() {
    use ggml::tensor::GgmlOp;
    let count = |ctx: &llama::context::EncoderContext, g: &llama::context::EncoderGraph, op: GgmlOp| {
        g.graph
            .nodes
            .iter()
            .filter(|&&n| ctx.gctx.op(n) == op)
            .count()
    };
    let gelu = |ctx: &llama::context::EncoderContext, g: &llama::context::EncoderGraph| {
        g.graph
            .nodes
            .iter()
            .filter(|&&n| {
                ctx.gctx.op(n) == GgmlOp::Silu
                    && ctx.gctx.op_params(n)[0] == ggml::ops::GGML_UNARY_OP_GELU
            })
            .count()
    };
    let tanh = |ctx: &llama::context::EncoderContext, g: &llama::context::EncoderGraph| {
        g.graph
            .nodes
            .iter()
            .filter(|&&n| {
                ctx.gctx.op(n) == GgmlOp::Silu
                    && ctx.gctx.op_params(n)[0] == ggml::ops::GGML_UNARY_OP_TANH
            })
            .count()
    };
    let tokens: Vec<i32> = (1..=12).collect();
    // every file of this batch has 4 layers
    let n: usize = spec_nomic().n_layer;

    // jina-bert-v2: NO rope, NO pos add; the q/k norms add 2 extra
    // Norm+Mul+Add pairs on layer 1..; the ALiBi mask is filled with
    // -|p0-p1| (checked via the forward values below, not the graph)
    {
        let spec = spec_jina_v2();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        assert_eq!(count(&ctx, &g, GgmlOp::RoPE), 0, "jina-v2: no rope");
        // 3 LNs per layer (attn_out, attn_norm_2-only-layer-1... see the
        // tensor table) + tok_norm
        let norms = count(&ctx, &g, GgmlOp::Norm);
        assert_eq!(norms, 2 * n + 1 /* base */ + n /* q norms */ + n /* k norms */ + 1 /* attn_norm_2 */, "jina-v2: norm count");
        assert_eq!(count(&ctx, &g, GgmlOp::GetRows), 3, "jina-v2: tok_embd + type row0 pair (no pos)");
    }

    // nomic: rope pair per layer + the swiglu_split GLU
    {
        let spec = spec_nomic();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        assert_eq!(count(&ctx, &g, GgmlOp::RoPE), 2 * n, "nomic: q/k rope per layer");
        assert_eq!(count(&ctx, &g, GgmlOp::Glu), n, "nomic: one swiglu_split per layer");
    }

    // nomic-bert-moe: 2 MoE layers → mul_mat_id trio + argsort + softmax of
    // the router; 2 dense layers → gelu
    {
        let spec = spec_nomic_moe();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        // 2 MoE layers → the up/down mul_mat_id pair each (the router itself
        // is a plain mul_mat, build_moe_ffn's build_lora_mm(gate_inp) —
        // llama-graph.cpp:2028) + argsort_top_k + softmax of the router probs
        assert_eq!(count(&ctx, &g, GgmlOp::MulMatId), 2 * 2, "nomic-moe: up/down mm_id per MoE layer");
        assert_eq!(count(&ctx, &g, GgmlOp::Argsort), 2, "nomic-moe: one argsort per MoE layer");
        // the MoE experts are GELU too (build_moe_ffn(..., LLM_FFN_GELU) —
        // bert.cpp:174), so every layer carries one gelu
        assert_eq!(gelu(&ctx, &g), 4, "nomic-moe: gelu on the dense AND MoE layers");
        assert_eq!(count(&ctx, &g, GgmlOp::RoPE), 2 * n, "nomic-moe: rope per layer");
    }

    // neo-bert: fused qkv (one mm) + swiglu GLU + RMS norms
    {
        let spec = spec_neo_bert();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        // per layer: wqkv, kq, kqv, wo, ffn_up, ffn_down = 6
        assert_eq!(count(&ctx, &g, GgmlOp::MulMat), 6 * n, "neo-bert: 6 mm per layer");
        assert_eq!(count(&ctx, &g, GgmlOp::Glu), n, "neo-bert: one swiglu per layer");
        assert_eq!(count(&ctx, &g, GgmlOp::RoPE), 2 * n, "neo-bert: rope per layer");
    }

    // modern-bert: geglu GLU + gelu on the rank head only; the silu file's
    // GLU is swiglu instead
    {
        let spec = spec_modern_bert();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        assert_eq!(count(&ctx, &g, GgmlOp::Glu), n, "modern-bert: one geglu per layer");
        assert_eq!(gelu(&ctx, &g), 0, "modern-bert: no plain gelu (embedder file)");
    }
    {
        let spec = spec_modern_bert_silu();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::NONE);
        let g = ctx.build(&tokens).unwrap();
        assert_eq!(count(&ctx, &g, GgmlOp::Glu), n, "modern-bert-silu: one swiglu per layer");
    }
    // the rank head: gelu + the cls/cls_norm/cls_out tail on top of the mean
    {
        let spec = spec_modern_bert_rank();
        let m = load_synth(&spec);
        let mut ctx = encoder(m, P::RANK);
        let g = ctx.build(&tokens).unwrap();
        // the head's own gelu — the per-layer GEGLU is the fused Glu op,
        // asserted via count(Glu) above. 37ac63456 maps the classifier gelu
        // to the exact erf variant (GGML_UNARY_OP_GELU_ERF, the modern-bert
        // loader default when the file names no activation)
        let gelu_erf = |ctx: &llama::context::EncoderContext,
                        g: &llama::context::EncoderGraph| {
            g.graph.nodes.iter().filter(|&&n| {
                ctx.gctx.op(n) == ggml::GgmlOp::Silu
                    && ctx.gctx.op_params(n)[0] == ggml::ops::GGML_UNARY_OP_GELU_ERF
            })
            .count()
        };
        assert_eq!(
            gelu_erf(&ctx, &g),
            1,
            "modern-bert-rank: the head's gelu_erf (llama-graph.cpp:3729, 37ac63456)"
        );
        // no tanh and no tanh-approx gelu on the modern GTE flavor
        assert_eq!(tanh(&ctx, &g), 0, "modern-bert-rank: GTE head uses gelu, not tanh");
        assert_eq!(gelu(&ctx, &g), 0, "modern-bert-rank: exact erf gelu, not the tanh approx");
    }
}

/// jina-bert-v2's ALiBi actually biases the attention: with the mask filled
/// `-|p0 - p1|` and soft_max_ext(max_bias = 8), a run with the alibi facts
/// must differ from one without (the same weights, alibi zeroed) — pinning
/// that the mask fill + max_bias both ride the graph.
#[test]
fn bert_variants_jina_v2_alibi_changes_output() {
    let spec = spec_jina_v2();
    let tokens: Vec<i32> = (1..=12).collect();

    let m = load_synth(&spec);
    let mut ctx = encoder(m, P::NONE);
    let with_alibi = ctx.encode(&tokens).unwrap();

    // the same weights, alibi facts zeroed (max_alibi_bias = 0 → the plain
    // 0.0 mask + no per-head slopes)
    let mut m2 = load_synth(&spec);
    let hp2 = m2.hparams.clone();
    let rope = hp2.rope_runtime();
    let params = graph_arch::EncoderParams {
        n_head: hp2.n_head(0) as i64,
        n_head_kv: hp2.n_head_kv(0) as i64,
        n_embd_head: hp2.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp2.n_rel_attn_bkts,
        f_norm_eps: hp2.f_norm_eps,
        f_norm_rms_eps: hp2.f_norm_rms_eps,
        pool: P::NONE,
        euro_rope: Some(graph_arch::EurobertRope {
            n_rot: hp2.n_rot(0) as i32,
            rope_mode: hp2.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp2.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        }),
        gemma_swa: None,
        causal: false,
    };
    let gctx = std::mem::replace(&mut m2.ctx, Context::new());
    let mut ctx0 = EncoderContext::new(
        gctx,
        EncoderWeights::BertVariant(
            m2.bert_variant_weights(graph_arch::BertVariant::JinaV2),
            graph_arch::BertVariantParams {
                max_alibi_bias: 0.0,
                moe_every_n_layers: 0,
                n_expert: 0,
                n_expert_used: 0,
                expert_weights_scale: 0.0,
                n_ff: hp2.n_ff(0) as i64,
            },
        ),
        params,
        8,
    );
    let without = ctx0.encode(&tokens).unwrap();

    let n_diff = with_alibi
        .values
        .iter()
        .zip(&without.values)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(n_diff > 0, "alibi must change the embeddings");
    // and both runs stay finite (the -|p0-p1| fill must not -inf the diagonal)
    assert!(
        with_alibi.values.iter().all(|v| v.is_finite()),
        "alibi output finite"
    );
}

// ---------------------------------------------------------------------------
// the reference `llama_encode` dumps (the bert/t5/eurobert/gemma-embed
// protocol): parity/ref_encode_dump <file> <out> --ids … --fa off
// ---------------------------------------------------------------------------

/// one dump cell: (spec, pooling arg, dump suffix)
fn dump_cells() -> Vec<(SynthSpec, &'static str, &'static str)> {
    vec![
        (spec_jina_v2(), "none", "jina-bert-v2"),
        (spec_jina_v2_gated(), "none", "jina-bert-v2-gated"),
        (spec_jina_v3(), "none", "jina-bert-v3"),
        (spec_nomic(), "none", "nomic-bert"),
        (spec_nomic_moe(), "none", "nomic-bert-moe"),
        (spec_neo_bert(), "none", "neo-bert"),
        (spec_modern_bert(), "none", "modern-bert"),
        // the swa file also rides MEAN so the mean path of a symmetric-SWA
        // encoder is covered
        (spec_modern_bert(), "mean", "modern-bert-mean"),
        (spec_modern_bert_rank(), "rank", "modern-bert-rank"),
        (spec_modern_bert_silu(), "none", "modern-bert-silu"),
    ]
}

/// Read one `parity/encode_*` artifact (the ref_encode_dump format).
struct RefDump {
    pooling: i32,
    values: Vec<f32>,
}

fn read_ref(path: &str) -> Option<RefDump> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot read {path}: {e} (run parity/gen_encode_bert_variants_ref.sh)");
            return None;
        }
    };
    assert_eq!(&bytes[..8], b"LENCE1\0\0", "{path}: magic");
    let u32at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (n_tokens, n_embd_out, n_rows, pooling) =
        (u32at(8) as usize, u32at(12) as usize, u32at(16) as usize, u32at(20) as i32);
    let mut off = 24 + 4 * n_tokens;
    let values: Vec<f32> = bytes[off..off + 4 * n_embd_out * n_rows]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    off += 4 * n_embd_out * n_rows;
    assert_eq!(off, bytes.len(), "{path}: payload size");
    assert_eq!(n_rows, if pooling == 0 { n_tokens } else { 1 });
    Some(RefDump { pooling, values })
}

#[test]
#[ignore = "needs parity/ref_encode_dump + parity/encode_bert_variants_*.bin (gen_encode_bert_variants_ref.sh)"]
fn bert_variants_reference_parity() {
    let dumper = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/ref_encode_dump");
    assert!(
        std::path::Path::new(dumper).exists(),
        "{dumper} missing — build it with parity/gen_encode_ref.sh"
    );
    let ids = "1,2,3,4,5,6,7,8,9,10,11,12";
    let tokens: Vec<i32> = ids.split(',').map(|v| v.parse().unwrap()).collect();

    for (spec, pool, cell) in dump_cells() {
        // regenerate the dump if missing (self-contained parity cell)
        let dump = format!(
            "{}/../../parity/encode_bert_variants_{}.bin",
            env!("CARGO_MANIFEST_DIR"),
            cell
        );
        if !std::path::Path::new(&dump).exists() {
            let _g = build_lock();
            if !std::path::Path::new(&spec.path()).exists() {
                build_file(&spec);
            }
            let st = std::process::Command::new(dumper)
                .arg(spec.path())
                .arg(&dump)
                .arg("--ids")
                .arg(ids)
                .arg("--pool")
                .arg(pool)
                .arg("--fa")
                .arg("off")
                .status()
                .expect("run ref_encode_dump");
            assert!(st.success(), "{cell}: ref_encode_dump failed");
        }
        let Some(r) = read_ref(&dump) else { return };

        let pool_p = match pool {
            "none" => P::NONE,
            "mean" => P::MEAN,
            "rank" => P::RANK,
            _ => unreachable!(),
        };
        assert_eq!(
            r.pooling, pool_p as i32,
            "{cell}: pooling header ({} vs {})",
            r.pooling, pool_p as i32
        );

        let m = load_synth(&spec);
        let mut ctx = encoder(m, pool_p);
        let emb = ctx.encode(&tokens).expect("encode");

        assert_eq!(emb.values.len(), r.values.len(), "{cell}: row count");
        let mut worst = 0f32;
        let mut bits = 0usize;
        for (a, b) in emb.values.iter().zip(&r.values) {
            worst = worst.max((a - b).abs());
            bits += usize::from(a.to_bits() == b.to_bits());
        }
        println!(
            "{cell}: n={} bit-exact={}/{} max|Δ|={worst:.3e}",
            emb.values.len(),
            bits,
            r.values.len()
        );
        assert_eq!(worst, 0.0, "{cell}: max |Δembd| vs the reference");
    }
}
