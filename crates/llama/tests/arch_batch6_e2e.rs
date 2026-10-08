//! arch_batch6_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-24: **the DeepSeek MLA family** — deepseek2 (classic MLA with
//! the absorbed MQA form + the wv_b decompression) with its lite / legacy
//! non-MLA / OCR variants, plus the non-MLA v2 base deepseek
//! (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-5 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32. Every MoE file carries
//! `n_expert = 4`, `n_expert_used = 2` so the routing is actually exercised.
//!
//! The five files pin the four attention paths of deepseek2.cpp:
//!   * `deepseek2` — the DEFAULT MLA absorbed path (:540-592): q_lora
//!     projection chain, wk_b absorption, the compressed K-only cache
//!     (concat(kv_cmpr, k_pe)) and the wv_b decompression, V3-style MoE
//!     (sigmoid gating + e-score bias + norm_w);
//!   * `deepseek2-v2` — same graph, V2-style MoE (softmax, no bias, no norm);
//!   * `deepseek2-lite` — `q_lora_rank = 0` → plain wq (:109/:498-511);
//!   * `deepseek2-legacy` — no `attention.key_length_mla` → `is_mla() ==
//!     false`, the unsplit wkv_b decompressed-MHA path (:593-634) incl. the
//!     trailing-segment rope (`ggml_rope_set_offset`, :617);
//!   * `deepseek2-ocr` — the is_ocr MHA + NEOX-rope branch (:473-494);
//!   * `deepseek` — the non-MLA v2 base (deepseek.cpp:74-194).
//!
//! Unlike batches 3-5, this batch's ForwardWeights arms landed in the same
//! change (context.rs), so the default-run tests drive
//! `DecodeContext::new_with` itself; the parity runs drive the release CLI
//! (`ARCH_BATCH6=1 ./parity/arch_batch_parity.sh …`, batch-1 protocol).

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::{AttnParams, ForwardResult};
use llama::graph_arch::{self};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch6";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec — real MLA proportions, small: n_embd 128, 4 heads,
// kv_lora_rank 32, qk_rope 16, qk_nope 24 (k_mla 40), v_mla 20
// ---------------------------------------------------------------------------

/// which of deepseek2.cpp's four attention branches the file drives
#[derive(Clone, Copy, PartialEq)]
enum DsVariant {
    /// default MLA absorbed path, V3-style MoE
    Mla,
    /// default MLA absorbed path, V2-style MoE
    MlaV2,
    /// q_lora_rank == 0 → plain wq
    Lite,
    /// no mla keys → legacy wkv_b decompressed MHA
    Legacy,
    /// deepseek2-ocr (own arch name)
    Ocr,
    /// the non-MLA v2 base deepseek (own arch name, GQA q/k/v)
    Base,
    /// deepseek32 (V3.2) — MLA + the DSA lightning indexer (own arch name)
    Dsa32,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    variant: DsVariant,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    /// attention.key_length (kv_lora_rank + qk_rope for MLA files)
    key_length: i64,
    /// attention.value_length (kv_lora_rank for MLA files)
    value_length: i64,
    /// attention.key_length_mla (absent for Legacy/Ocr/deepseek)
    key_length_mla: Option<i64>,
    value_length_mla: Option<i64>,
    kv_lora_rank: Option<i64>,
    /// attention.q_lora_rank (0 → lite wq; absent only via is_lite layers)
    q_lora_rank: Option<i64>,
    n_rot: i64,
    n_ff: i64,
    n_ff_exp: i64,
    dense_lead: u32,
    n_ctx: u32,
    /// V3-style MoE: sigmoid gating + exp_probs_b + expert_weights_norm
    v3_moe: bool,
}

impl SynthSpec {
    fn qk_rope(&self) -> i64 {
        self.n_rot
    }
    fn qk_nope(&self) -> i64 {
        self.key_length_mla.unwrap_or(self.key_length) - self.qk_rope()
    }
    fn k_mla(&self) -> i64 {
        self.key_length_mla.unwrap_or(self.key_length)
    }
    fn v_mla(&self) -> i64 {
        self.value_length_mla.unwrap_or(self.value_length)
    }
    fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }
    fn n_ff_shexp(&self) -> i64 {
        self.n_ff_exp // n_expert_shared = 1
    }
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut s = self.clone();
        f(&mut s);
        s
    }
}

fn base_ds2(variant: DsVariant) -> SynthSpec {
    SynthSpec {
        arch: "deepseek2",
        suffix: "",
        variant,
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 1,
        key_length: 48,
        value_length: 32,
        key_length_mla: Some(40),
        value_length_mla: Some(20),
        kv_lora_rank: Some(32),
        q_lora_rank: Some(32),
        n_rot: 16,
        n_ff: 48,
        n_ff_exp: 24,
        dense_lead: 1,
        n_ctx: 256,
        v3_moe: true,
    }
}

/// deepseek2 — the DEFAULT MLA absorbed path, V3-style MoE (sigmoid gating +
/// e-score bias + norm_w, the DeepSeek-V3 configuration)
fn spec_deepseek2() -> SynthSpec {
    base_ds2(DsVariant::Mla)
}

/// deepseek2-v2 — same graph, V2-style MoE (softmax, no bias, no norm)
fn spec_deepseek2_v2() -> SynthSpec {
    base_ds2(DsVariant::MlaV2).with(|s| {
        s.suffix = "-v2";
        s.v3_moe = false;
    })
}

/// deepseek2-lite — q_lora_rank = 0 → the plain wq of deepseek2.cpp:109
fn spec_deepseek2_lite() -> SynthSpec {
    base_ds2(DsVariant::Lite).with(|s| {
        s.suffix = "-lite";
        s.q_lora_rank = Some(0);
    })
}

/// deepseek2-legacy — no attention.key_length_mla → is_mla() == false: the
/// unsplit wkv_b + decompressed MHA path (deepseek2.cpp:593-634), GQA=MHA
/// head counts, key_length = qk_nope + qk_rope
fn spec_deepseek2_legacy() -> SynthSpec {
    base_ds2(DsVariant::Legacy).with(|s| {
        s.suffix = "-legacy";
        s.key_length = 40; // 24 nope + 16 rope
        s.value_length = 20;
        s.key_length_mla = None;
        s.value_length_mla = None;
        s.kv_lora_rank = Some(32);
        s.q_lora_rank = Some(32);
        s.n_head_kv = 4; // decompressed MHA — full head groups
        s.v3_moe = false;
    })
}

/// deepseek2-ocr — the is_ocr branch: MHA q/k/v + NEOX rope at 10000
/// (deepseek2.cpp:473-494; arch name deepseek2-ocr)
fn spec_deepseek2_ocr() -> SynthSpec {
    SynthSpec {
        arch: "deepseek2-ocr",
        suffix: "",
        variant: DsVariant::Ocr,
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 4,
        key_length: 32,
        value_length: 32,
        key_length_mla: None,
        value_length_mla: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        n_rot: 32,
        n_ff: 48,
        n_ff_exp: 24,
        dense_lead: 1,
        n_ctx: 256,
        v3_moe: false,
    }
}

/// deepseek — the non-MLA v2 base (deepseek.cpp): GQA MHA + dense-lead MoE
fn spec_deepseek() -> SynthSpec {
    SynthSpec {
        arch: "deepseek",
        suffix: "",
        variant: DsVariant::Base,
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 2,
        key_length: 32,
        value_length: 32,
        key_length_mla: None,
        value_length_mla: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        n_rot: 32,
        n_ff: 48,
        n_ff_exp: 24,
        dense_lead: 1,
        n_ctx: 256,
        v3_moe: false,
    }
}

/// deepseek32 — the MLA graph + the DSA lightning indexer (deepseek32.cpp):
/// indexer head 64 (a power of two — the Hadamard `k_rot` is 64x64),
/// 2 indexer heads, top_k 8
fn spec_deepseek32() -> SynthSpec {
    SynthSpec {
        arch: "deepseek32",
        suffix: "",
        variant: DsVariant::Dsa32,
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 1,
        key_length: 48,
        value_length: 32,
        key_length_mla: Some(40),
        value_length_mla: Some(20),
        kv_lora_rank: Some(32),
        q_lora_rank: Some(32),
        n_rot: 16,
        n_ff: 48,
        n_ff_exp: 24,
        dense_lead: 1,
        n_ctx: 256,
        v3_moe: true,
    }
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_deepseek2(),
        spec_deepseek2_v2(),
        spec_deepseek2_lite(),
        spec_deepseek2_legacy(),
        spec_deepseek2_ocr(),
        spec_deepseek(),
        spec_deepseek32(),
    ]
}

// ---------------------------------------------------------------------------
// the tensor tables (exactly what each arch's load_arch_tensors asks for)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Bias,
    Proj,
    Router,
}

/// (name, ne) pairs — the loader's create_tensor requests, verbatim
fn tensors_for(spec: &SynthSpec) -> Vec<(String, Vec<i64>, Role)> {
    let mut v: Vec<(String, Vec<i64>, Role)> = Vec::new();
    let n_embd = spec.n_embd;
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $role))
        };
    }
    // model level — output.weight always absent (tied head fallback)
    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    push!("output_norm.weight", vec![n_embd], Role::Norm);

    let n_layer = spec.n_layer;
    let is_ds2 = spec.arch == "deepseek2";

    for i in 0..n_layer as i32 {
        push!(
            format!("blk.{i}.attn_norm.weight"),
            vec![n_embd],
            Role::Norm
        );

        match spec.variant {
            DsVariant::Mla | DsVariant::MlaV2 | DsVariant::Lite | DsVariant::Legacy => {
                let kv_lora_rank = spec.kv_lora_rank.unwrap();
                if spec.q_lora_rank.unwrap() > 0 {
                    push!(
                        format!("blk.{i}.attn_q_a_norm.weight"),
                        vec![spec.q_lora_rank.unwrap()],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_q_a.weight"),
                        vec![n_embd, spec.q_lora_rank.unwrap()],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_q_b.weight"),
                        vec![spec.q_lora_rank.unwrap(), spec.n_head * spec.k_mla()],
                        Role::Proj
                    );
                } else {
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, spec.n_head * spec.k_mla()],
                        Role::Proj
                    );
                }
                push!(
                    format!("blk.{i}.attn_kv_a_norm.weight"),
                    vec![kv_lora_rank],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, kv_lora_rank + spec.qk_rope()],
                    Role::Proj
                );
                if spec.key_length_mla.is_some() {
                    push!(
                        format!("blk.{i}.attn_k_b.weight"),
                        vec![spec.qk_nope(), kv_lora_rank, spec.n_head],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v_b.weight"),
                        vec![kv_lora_rank, spec.v_mla(), spec.n_head],
                        Role::Proj
                    );
                } else {
                    push!(
                        format!("blk.{i}.attn_kv_b.weight"),
                        vec![kv_lora_rank, spec.n_head * (spec.qk_nope() + spec.v_mla())],
                        Role::Proj
                    );
                }
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.v_mla(), n_embd],
                    Role::Proj
                );
            }
            DsVariant::Ocr | DsVariant::Base => {
                // deepseek2ocr.cpp:43 / deepseek.cpp:39 — create_tensor_qkv:
                // ocr is MHA (q/k/v all [n_embd, n_embd]), the v2 base keeps
                // GQA (k/v at n_embd_head * n_head_kv)
                let kv_dim = spec.n_embd / spec.n_head * spec.n_head_kv;
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k.weight"),
                    vec![n_embd, kv_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, kv_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
            }
            DsVariant::Dsa32 => {
                // deepseek32.cpp:90-114 — the MLA set (always wq_a/wq_b, no
                // lite branch) + the five indexer tensors
                let kv_lora_rank = spec.kv_lora_rank.unwrap();
                push!(
                    format!("blk.{i}.attn_q_a_norm.weight"),
                    vec![spec.q_lora_rank.unwrap()],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_a.weight"),
                    vec![n_embd, spec.q_lora_rank.unwrap()],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_q_b.weight"),
                    vec![spec.q_lora_rank.unwrap(), spec.n_head * spec.k_mla()],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_kv_a_norm.weight"),
                    vec![kv_lora_rank],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, kv_lora_rank + spec.qk_rope()],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k_b.weight"),
                    vec![spec.qk_nope(), kv_lora_rank, spec.n_head],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v_b.weight"),
                    vec![kv_lora_rank, spec.v_mla(), spec.n_head],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.v_mla(), n_embd],
                    Role::Proj
                );
                // indexer (deepseek32.cpp:109-114) — head size 64, 2 heads
                let ih = 64i64;
                let inh = 2i64;
                push!(
                    format!("blk.{i}.indexer.k_norm.weight"),
                    vec![ih],
                    Role::Norm
                );
                push!(format!("blk.{i}.indexer.k_norm.bias"), vec![ih], Role::Bias);
                push!(
                    format!("blk.{i}.indexer.proj.weight"),
                    vec![n_embd, inh],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer.attn_k.weight"),
                    vec![n_embd, ih],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.indexer.attn_q_b.weight"),
                    vec![spec.q_lora_rank.unwrap(), inh * ih],
                    Role::Proj
                );
            }
        }

        push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);

        if (i as u32) < spec.dense_lead {
            push!(
                format!("blk.{i}.ffn_gate.weight"),
                vec![n_embd, spec.n_ff],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_down.weight"),
                vec![spec.n_ff, n_embd],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_up.weight"),
                vec![n_embd, spec.n_ff],
                Role::Proj
            );
        } else {
            push!(
                format!("blk.{i}.ffn_gate_inp.weight"),
                vec![n_embd, N_EXPERT],
                Role::Router
            );
            if spec.v3_moe && is_ds2 {
                push!(
                    format!("blk.{i}.exp_probs_b.bias"),
                    vec![N_EXPERT],
                    Role::Bias
                );
            }
            push!(
                format!("blk.{i}.ffn_down_exps.weight"),
                vec![spec.n_ff_exp, n_embd, N_EXPERT],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_gate_exps.weight"),
                vec![n_embd, spec.n_ff_exp, N_EXPERT],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_up_exps.weight"),
                vec![n_embd, spec.n_ff_exp, N_EXPERT],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_gate_shexp.weight"),
                vec![n_embd, spec.n_ff_shexp()],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_down_shexp.weight"),
                vec![spec.n_ff_shexp(), n_embd],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_up_shexp.weight"),
                vec![n_embd, spec.n_ff_shexp()],
                Role::Proj
            );
        }
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch4_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch6");

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
    kv!(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5)
    );
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.n_rot as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // the MoE keys (deepseek2.cpp:18-22 / deepseek2ocr.cpp:6-11)
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(
        format!("{a}.expert_used_count"),
        Value::U32(N_EXPERT_USED as u32)
    );
    kv!(
        format!("{a}.expert_feed_forward_length"),
        Value::U32(spec.n_ff_exp as u32)
    );
    kv!(format!("{a}.expert_shared_count"), Value::U32(1));
    kv!(
        format!("{a}.leading_dense_block_count"),
        Value::U32(spec.dense_lead)
    );
    if let Some(r) = spec.kv_lora_rank {
        kv!(format!("{a}.attention.kv_lora_rank"), Value::U32(r as u32));
    }
    if let Some(r) = spec.q_lora_rank {
        kv!(format!("{a}.attention.q_lora_rank"), Value::U32(r as u32));
    }
    if let Some(k) = spec.key_length_mla {
        kv!(
            format!("{a}.attention.key_length_mla"),
            Value::U32(k as u32)
        );
    }
    if let Some(v) = spec.value_length_mla {
        kv!(
            format!("{a}.attention.value_length_mla"),
            Value::U32(v as u32)
        );
    }
    if spec.v3_moe {
        // DeepSeek-V3 routing: sigmoid gating + bias + normalized weights
        kv!(format!("{a}.expert_gating_func"), Value::U32(2)); // SIGMOID
        kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
    }
    if let DsVariant::Dsa32 = spec.variant {
        // the DSA indexer keys (deepseek32.cpp:30-33 — all required)
        kv!(format!("{a}.attention.indexer.head_count"), Value::U32(2));
        kv!(format!("{a}.attention.indexer.key_length"), Value::U32(64));
        kv!(format!("{a}.attention.indexer.top_k"), Value::U32(8));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for (name, ne, role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = match role {
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            _ => (0..n).map(|_| s * rng.next()).collect(),
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

    let path = spec.path();
    let f = std::fs::File::create(&path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    (table.len(), std::fs::metadata(&path).unwrap().len())
}

fn load_synth(spec: &SynthSpec) -> LlamaModel {
    build_file(spec);
    open_model(&spec.path())
}

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let mut want: Vec<String> = tensors_for(spec).into_iter().map(|(n, _, _)| n).collect();
    want.sort();
    want.dedup();
    let got = {
        let mut v: Vec<String> = m.tensors.keys().cloned().collect();
        v.sort();
        v
    };
    assert_eq!(got, want, "{}: created tensor set mismatch", spec.arch);
}

// ---------------------------------------------------------------------------
// weights + params assembly (the same derivations llama-cli's forward_weights
// performs — kept in sync with crates/tools/llama-cli/src/main.rs)
// ---------------------------------------------------------------------------

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

fn deepseek2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek2ModelWeights {
    graph_arch::Deepseek2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk].iter().map(ds2_layer).collect(),
    }
}

fn deepseek_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DeepseekModelWeights {
    graph_arch::DeepseekModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::DeepseekLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
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

/// (ForwardWeights, AttnParams) of one loaded model — the same assembly the
/// CLI performs; this batch landed its context.rs arms together, so the
/// default tests drive `DecodeContext` itself.
fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa);
    let n_trunk = hp.n_layer() as usize;
    match m.arch {
        llama::arch::LlmArch::DEEPSEEK2 | llama::arch::LlmArch::DEEPSEEK2OCR => {
            let w = (
                ForwardWeights::Deepseek2(
                    deepseek2_weights(m, n_trunk),
                    graph_arch::Deepseek2Params {
                        attn,
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
                        is_ocr: m.arch == llama::arch::LlmArch::DEEPSEEK2OCR,
                    },
                ),
                attn,
            );
            w
        }
        llama::arch::LlmArch::DEEPSEEK => (
            ForwardWeights::Deepseek(
                deepseek_weights(m, n_trunk),
                graph_arch::DeepseekParams {
                    attn,
                    f_attention_scale: hp.f_attention_scale,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::DEEPSEEK32 => {
            let ds2 = llama::graph_arch::Deepseek2Params {
                attn,
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
            };
            (
                ForwardWeights::Deepseek32(
                    deepseek2_weights(m, n_trunk),
                    llama::graph_arch::Deepseek32Params {
                        ds2,
                        indexer_n_head: hp.indexer_n_head as i64,
                        indexer_head_size: hp.indexer_head_size as i64,
                        indexer_top_k: hp.indexer_top_k as i64,
                        f_norm_eps: hp.f_norm_eps,
                    },
                ),
                attn,
            )
        }
        other => panic!("arch {other:?} not in batch 6"),
    }
}

// ---------------------------------------------------------------------------
// the decode harness — DecodeContext itself (the batch's context.rs arms)
// ---------------------------------------------------------------------------

fn driver_for(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    // n_ctx 512 like the parity runs; n_batch covers the long prompts
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as i32
}

/// 3-token prefill + one more step: finite logits with a real spread, and a
/// bit-identical repeat on a cleared cache — through both FA modes so wiring
/// issues surface here rather than in the parity runs.
fn smoke_forward(spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut m = load_synth(spec);
    let mut d = driver_for(&mut m, fa);

    let toks = [3i32, 17, 42];
    let pos: Vec<i32> = (0..3).collect();
    let a = d.decode(&toks, &pos).expect("decode").to_vec();
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{}{}: non-finite logits (fa={fa})",
        spec.arch,
        spec.suffix
    );
    let spread = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - a.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}{}: logits degenerate (spread {spread}, fa={fa})",
        spec.arch,
        spec.suffix
    );

    let next = argmax(&a);
    let b = d.decode(&[next], &[3]).expect("decode2").to_vec();
    assert!(b.iter().all(|v| v.is_finite()));
    assert_eq!(d.kv.used_cells(), 4);

    // fresh cache → bit-identical first decode
    d.reset_sequence();
    let a2 = d.decode(&toks, &pos).expect("decode repeat").to_vec();
    assert_eq!(
        a, a2,
        "{}{}: repeat mismatch (fa={fa})",
        spec.arch, spec.suffix
    );
    a
}

// ---------------------------------------------------------------------------
// default-run tests (no reference needed)
// ---------------------------------------------------------------------------

#[test]
fn synth_deepseek2_loader_and_forward() {
    let spec = spec_deepseek2();
    let (n, bytes) = build_file(&spec);
    // 2 + 4 layers x (9 MLA attn tensors + ffn_norm) + dense layer's 3 FFN +
    // 3 MoE layers x 8 (router + e-score bias + 3 experts + 3 shexp)
    assert_eq!(n, 2 + 4 * 10 + 3 + 3 * 8, "deepseek2 tensor count");
    println!(
        "deepseek2 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_lora_q, 32);
    assert_eq!(hp.n_lora_kv, 32);
    assert!(hp.is_mla());
    assert_eq!(hp.n_embd_head_k_mla(), 40);
    assert_eq!(hp.n_embd_head_v_mla(), 20);
    assert_eq!(
        hp.n_embd_head_k(0),
        48,
        "cache K width = kv_lora_rank + qk_rope"
    );
    assert_eq!(hp.n_head_kv(0), 1, "MLA converts into MQA");
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert_eq!(hp.expert_gating_func, 2, "sigmoid (V3-style file)");
    assert!(hp.expert_weights_norm);
    assert_eq!(m.output, m.tok_embd, "tied head");
    // wk_b {qk_nope, kv_lora_rank, n_head} / wv_b {kv_lora_rank, v_mla, n_head}
    let l1 = &m.layers[1];
    assert_eq!(*m.ctx.ne(l1.wk_b.unwrap()), [24, 32, 4, 1]);
    assert_eq!(*m.ctx.ne(l1.wv_b.unwrap()), [32, 20, 4, 1]);
    assert_eq!(*m.ctx.ne(l1.wo.unwrap()), [80, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].ffn_gate.unwrap()), [128, 48, 1, 1]);
    assert!(m.layers[0].ffn_gate_inp.is_none(), "layer 0 is dense-lead");

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek2 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deepseek2_v2_moe_style() {
    let spec = spec_deepseek2_v2();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert_eq!(
        m.hparams.expert_gating_func, 1,
        "softmax fallback (V2 file)"
    );
    assert!(!m.hparams.expert_weights_norm);
    assert!(m.layers[1].ffn_exp_probs_b.is_none());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek2-v2 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deepseek2_lite_loader_and_forward() {
    let spec = spec_deepseek2_lite();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    // q_lora_rank == 0 → plain wq (deepseek2.cpp:109), no q_a norm
    assert!(m.layers[0].wq.is_some());
    assert!(m.layers[0].wq_a.is_none());
    assert!(m.layers[0].attn_q_a_norm.is_none());
    assert_eq!(*m.ctx.ne(m.layers[0].wq.unwrap()), [128, 160, 1, 1]);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek2-lite synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deepseek2_legacy_loader_and_forward() {
    let spec = spec_deepseek2_legacy();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    // is_mla() false → unsplit wkv_b, no wk_b/wv_b (deepseek2.cpp:114-120)
    assert!(!m.hparams.is_mla());
    assert!(m.layers[0].wkv_b.is_some());
    assert!(m.layers[0].wk_b.is_none());
    assert!(m.layers[0].wv_b.is_none());
    assert_eq!(*m.ctx.ne(m.layers[0].wkv_b.unwrap()), [32, 176, 1, 1]);
    // decompressed MHA: full head groups
    assert_eq!(m.hparams.n_head_kv(0), 4);
    assert_eq!(m.hparams.n_embd_head_k(0), 40);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek2-legacy synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deepseek2_ocr_loader_and_forward() {
    let spec = spec_deepseek2_ocr();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert!(!m.hparams.is_mla());
    assert_eq!(m.hparams.n_head_kv(0), 4, "ocr is plain MHA");
    assert_eq!(m.hparams.n_embd_head_k(0), 32);
    assert_eq!(m.hparams.n_rot(0), 32);
    assert_eq!(
        m.hparams.expert_gating_func, 1,
        "softmax fallback (ocr file)"
    );
    // build_qkv weights: q/k/v all [n_embd, n_embd]
    assert_eq!(*m.ctx.ne(m.layers[0].wq.unwrap()), [128, 128, 1, 1]);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek2-ocr synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deepseek_loader_and_forward() {
    let spec = spec_deepseek();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert_eq!(m.hparams.n_head_kv(0), 2);
    assert_eq!(m.hparams.n_embd_head_k(0), 32);
    assert_eq!(m.hparams.n_rot(0), 32);
    assert_eq!(m.hparams.n_layer_dense_lead, 1);
    // deepseek.cpp:39 — q full-width, k/v the GQA width
    assert_eq!(*m.ctx.ne(m.layers[0].wq.unwrap()), [128, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].wk.unwrap()), [128, 64, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].wv.unwrap()), [128, 64, 1, 1]);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// deepseek32 — the DSA lightning indexer over the `new_dsa` cache pair
#[test]
fn synth_deepseek32_loader_and_forward() {
    let spec = spec_deepseek32();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert!(hp.is_mla());
    assert_eq!(hp.indexer_n_head, 2);
    assert_eq!(hp.indexer_head_size, 64);
    assert_eq!(hp.indexer_top_k, 8);
    assert_eq!(hp.f_norm_eps, 1e-6, "deepseek32 hard-codes the LN eps (:9)");
    let l1 = &m.layers[1];
    assert_eq!(*m.ctx.ne(l1.indexer_attn_q_b.unwrap()), [32, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(l1.indexer_proj.unwrap()), [128, 2, 1, 1]);
    assert_eq!(*m.ctx.ne(l1.indexer_k_norm_b.unwrap()), [64, 1, 1, 1]);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deepseek32 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_deepseek2().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("deepseek2"));
    assert_eq!(g.get_u32("deepseek2.attention.kv_lora_rank"), Some(32));
    assert_eq!(g.get_u32("deepseek2.attention.q_lora_rank"), Some(32));
    assert_eq!(g.get_u32("deepseek2.attention.key_length"), Some(48));
    assert_eq!(g.get_u32("deepseek2.attention.value_length"), Some(32));
    assert_eq!(g.get_u32("deepseek2.attention.key_length_mla"), Some(40));
    assert_eq!(g.get_u32("deepseek2.attention.value_length_mla"), Some(20));
    assert_eq!(g.get_u32("deepseek2.expert_shared_count"), Some(1));
    assert_eq!(g.get_u32("deepseek2.expert_gating_func"), Some(2));
    let spec = spec_deepseek2_ocr().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("deepseek2-ocr"));
}

/// MLA K-only-cache structure pin: the kv cache allocates the 48-wide
/// compressed K rows and the builder never touches the v tensors. Computed
/// numerically: decode the same token twice at the same position on two
/// caches — the second run must equal the first (cache cells hold
/// concat(kv_cmpr, k_pe), not decompressed k/v).
#[test]
fn mla_k_only_cache_geometry() {
    let spec = spec_deepseek2();
    build_file(&spec);
    let mut m = open_model(&spec.path());
    let mut d = driver_for(&mut m, false);
    // the K row width is hparams-derived: n_embd_head_k(48) * n_head_kv(1)
    assert_eq!(
        d.kv.n_embd_k_gqa, 48,
        "MLA cache K rows are the compressed concat(kv_cmpr, k_pe)"
    );
    let toks = [7i32, 9];
    let a = d.decode(&toks, &[0, 1]).expect("decode").to_vec();
    // a third token attends both cached latents through the wv_b path
    let b = d.decode(&[argmax(&a)], &[2]).expect("decode2").to_vec();
    assert!(b.iter().all(|v| v.is_finite()));
    assert_eq!(d.kv.used_cells(), 3);
}

// ---------------------------------------------------------------------------
// generator (#[ignore]) — the parity runs drive the release llama-cli itself
// (batch-1 protocol; this batch's CLI arms landed with the graph)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~90 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch6_write_synth() {
    for spec in all_specs() {
        let (n, bytes) = build_file(&spec);
        println!(
            "{:>14}: {:4} tensors, {:>10} bytes -> {}",
            format!("{}{}", spec.arch, spec.suffix),
            n,
            bytes,
            spec.path()
        );
    }
    let gguf = Gguf::open(spec_deepseek2().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH6=1 ./parity/arch_batch_parity.sh deepseek2 deepseek2-lite \
         deepseek2-legacy deepseek2-ocr deepseek deepseek2-long"
    );
}
