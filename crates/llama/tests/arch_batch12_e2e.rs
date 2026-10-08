//! arch_batch12_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-10: **the final long-tail queue** — hrm-text / laguna / maple
//! (llama.cpp bd4f514db1, the last three src/models/*.cpp files without a
//! ported graph).
//!
//! Same protocol as batches 1-11 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//!   * hrm-text — alternating low/high transformer stacks over the same token
//!     stream (hrm-text.cpp): 2 physical stacks × lps 2 with h_cycles 2 /
//!     l_cycles 1 → 8 cache slots aliasing 4 physical blocks; the learned
//!     [n_embd] low-cycle state `hrm.z_l_init`, the weightless per-layer and
//!     per-stack RMS norms, the sigmoid attention gate before o_proj, and
//!     `embedding_scale` inside build_inp_embd;
//!   * laguna — sigmoid-routed MoE with the score-correction bias
//!     (`ffn_exp_probs.bias`), the always-on shared expert, the softplus
//!     attention output gate, QK-norm before rope, per-layer head counts and
//!     per-layer-type RoPE. The default file is the XS.2-like hybrid
//!     (sliding_window 64, period-4 dense-first pattern, full layers at
//!     θ=500000 with the YaRN-carrying ext_factor, SWA layers at plain
//!     θ=10000 over rope_swa.dimension_count dims, the per-element gate width
//!     on the full layer and the per-head one on the SWA layers); the
//!     `-full` variant is the M.1-like all-full file (no sliding window key
//!     → plain KV input, per-element gates everywhere, dense lead 2);
//!   * maple — the softmax MoE over the iswa pair (maple.cpp): the pattern
//!     array `[0,1,1,1]` (layer 0 full, 1-3 SWA), rope ONLY on the SWA layers
//!     (n_rot(il) = n_rot_swa there, get_rope_freq_base reads the *_swa
//!     copies), and the REQUIRED-but-unused `swiglu_clamp_exp` key.
//!
//! The batch's ForwardWeights/CLI/server arms landed with the graph
//! (context.rs), so the default-run tests drive `DecodeContext::new_with`
//! itself (batch-6+ protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH12=1 ./parity/arch_batch_parity.sh …`).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaSwaType;
use llama::kv_cache::SwaCacheSpec;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch12";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    HrmText,
    Laguna,
    Maple,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    /// per-layer head counts (laguna varies them; a single value broadcasts)
    n_head: Vec<i64>,
    head_kv: Vec<i64>,
    key_length: i64,
    value_length: i64,
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    // ---- hrm-text ----
    hrm_lps: u32,
    hrm_h_cycles: u32,
    hrm_l_cycles: u32,
    hrm_embedding_scale: f32,
    // ---- laguna ----
    /// None → no sliding_window key (the M.1-like all-full file)
    laguna_swa: Option<u32>,
    n_layer_dense_lead: u32,
    n_ff_exp: i64,
    n_ff_shexp: i64,
    n_rot_swa: i64,
    freq_base_full: f32,
    freq_base_swa: f32,
    /// per-layer gate widths (true = per-head [1, n_head_il], false =
    /// per-element [head_dim * n_head_il])
    gate_per_head: Vec<bool>,
    // ---- maple ----
    /// the sliding_window_pattern array (maple writes it always)
    maple_pattern: Vec<u32>,
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
        n_embd: 128,
        n_head: vec![4; 4],
        head_kv: vec![2; 4],
        key_length: 32,
        value_length: 32,
        rope_dim: 32,
        n_ff: 96,
        n_ctx: 512,
        write_output: true,
        hrm_lps: 2,
        hrm_h_cycles: 2,
        hrm_l_cycles: 1,
        hrm_embedding_scale: 0.0,
        laguna_swa: None,
        n_layer_dense_lead: 0,
        n_ff_exp: 32,
        n_ff_shexp: 40,
        n_rot_swa: 32,
        freq_base_full: 10_000.0,
        freq_base_swa: 10_000.0,
        gate_per_head: vec![false; 4],
        maple_pattern: Vec::new(),
    }
}

/// hrm-text — lps 2 × h_cycles 2 × (l_cycles 1 + 1) = 8 cache slots over the
/// 4 physical blocks; the embedding_scale keys the shared build_inp_embd.
/// NB: the GGUF arch string is "hrm_text" (underscore, llama-arch.cpp:138).
fn spec_hrm_text() -> SynthSpec {
    // block_count = n_slot = lps * h * (l + 1) = 2*2*2 = 8; only the 4
    // physical blocks carry tensors (the alias passes copy the structs)
    base("hrm_text", Family::HrmText).with(|s| {
        s.n_layer = 8;
        s.hrm_embedding_scale = 1.7;
        s.n_head = vec![4; 8];
        s.head_kv = vec![2; 8];
    })
}

/// laguna — the XS.2-like hybrid: sliding_window 64 (period-4 dense-first
/// pattern, layer 0 full), per-layer heads [4, 6, 4, 6], the full layer
/// per-element gated, the SWA layers per-head gated, YaRN-ish θ 500000 on the
/// full layer vs plain θ 10000 over 24 dims on the SWA layers
fn spec_laguna() -> SynthSpec {
    base("laguna", Family::Laguna).with(|s| {
        s.laguna_swa = Some(64);
        s.n_head = vec![4, 6, 4, 6];
        s.n_layer_dense_lead = 1;
        s.n_rot_swa = 24;
        s.freq_base_full = 500_000.0;
        s.freq_base_swa = 10_000.0;
        // layer 0 is the full layer — the per-element (M.1) gate width; the
        // SWA layers 1-3 carry the per-head (XS.2) width
        s.gate_per_head = vec![false, true, true, true];
    })
}

/// laguna `-full` — the M.1-like file: no sliding_window key (plain KV input,
/// no iswa pair), uniform heads, per-element gates everywhere, dense lead 2
fn spec_laguna_full() -> SynthSpec {
    spec_laguna().with(|s| {
        s.suffix = "-full";
        s.laguna_swa = None;
        s.n_head = vec![4; 4];
        s.n_layer_dense_lead = 2;
        s.gate_per_head = vec![false; 4];
    })
}

/// maple — pattern [0,1,1,1]: layer 0 full attention, 1-3 SWA; rope only on
/// the SWA layers (rope_swa.dimension_count 24, rope_swa.freq_base 10000)
fn spec_maple() -> SynthSpec {
    base("maple", Family::Maple).with(|s| {
        s.maple_pattern = vec![0, 1, 1, 1];
        s.n_rot_swa = 24;
        s.freq_base_swa = 10_000.0;
    })
}

/// every file the ignored writer test emits
fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_hrm_text(),
        spec_laguna(),
        spec_laguna_full(),
        spec_maple(),
    ]
}

/// the parity cells (batch-12 default set): every file has a reference-side
/// generation path (the three archs all create a memory in the pinned
/// reference — none is on the diffusion list of llama-model.cpp:2289-2295)
fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_hrm_text(),
        spec_laguna(),
        spec_laguna_full(),
        spec_maple(),
    ]
}

#[test]
#[ignore = "writes the /tmp/arch-batch12 parity files (ARCH_BATCH12 cells)"]
fn arch_batch12_write_synth() {
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
    /// routers and experts: 1/sqrt(n_embd)-scaled random weights so the
    /// softmax over the router logits stays spread out
    Router,
}

type TensorSpec = (String, Vec<i64>);

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let hd = spec.key_length;
    let mut t: Vec<(TensorSpec, Role)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, role: Role| t.push(((name, ne), role));

    push(
        "token_embd.weight".into(),
        vec![n_embd, N_VOCAB],
        Role::Proj,
    );

    match spec.family {
        Family::HrmText => {
            if spec.write_output {
                push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            }
            // the learned low-cycle state (hrm-text.cpp:46); `tn(LLM_TENSOR_
            // HRM_Z_L_INIT)` has NO ".weight" suffix — the bare name; the
            // per-layer and per-stack norms are parameterless — no norm
            // tensors at all
            push("hrm.z_l_init".into(), vec![n_embd], Role::Norm);
            let lps = spec.hrm_lps as usize;
            let n_phys = 2 * lps; // blocks [0, lps) low + [lps, 2*lps) high
            for i in 0..n_phys {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head[0];
                let kv_w = hd * spec.head_kv[0];
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_gate.weight"),
                    vec![n_embd, q_w],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
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
        Family::Laguna => {
            if spec.write_output {
                push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            }
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let n_head_il = spec.n_head[i];
                let q_w = hd * n_head_il;
                let kv_w = hd * spec.head_kv[0];
                let gate_w = if spec.gate_per_head[i] {
                    n_head_il
                } else {
                    hd * n_head_il
                };
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_gate.weight"),
                    vec![n_embd, gate_w],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("{b}attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                if (i as u32) >= spec.n_layer_dense_lead {
                    let e = spec.n_ff_exp;
                    push(
                        format!("{b}ffn_gate_inp.weight"),
                        vec![n_embd, N_EXPERT],
                        Role::Router,
                    );
                    // the tensor template is blk.%d.exp_probs_b (llama-arch.cpp)
                    push(format!("{b}exp_probs_b.bias"), vec![N_EXPERT], Role::Bias);
                    push(
                        format!("{b}ffn_gate_exps.weight"),
                        vec![n_embd, e, N_EXPERT],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_exps.weight"),
                        vec![n_embd, e, N_EXPERT],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_exps.weight"),
                        vec![e, n_embd, N_EXPERT],
                        Role::Proj,
                    );
                    let sh = spec.n_ff_shexp;
                    push(
                        format!("{b}ffn_gate_shexp.weight"),
                        vec![n_embd, sh],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_up_shexp.weight"),
                        vec![n_embd, sh],
                        Role::Proj,
                    );
                    push(
                        format!("{b}ffn_down_shexp.weight"),
                        vec![sh, n_embd],
                        Role::Proj,
                    );
                } else {
                    push(
                        format!("{b}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj,
                    );
                    push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                    push(
                        format!("{b}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj,
                    );
                }
            }
        }
        Family::Maple => {
            // maple.cpp:32-33 — output_norm + output REQUIRED (no tie fallback)
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            for i in 0..spec.n_layer {
                let b = format!("blk.{i}.");
                let q_w = hd * spec.n_head[0];
                let kv_w = hd * spec.head_kv[0];
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_q_norm.weight"), vec![hd], Role::Norm);
                push(format!("{b}attn_k_norm.weight"), vec![hd], Role::Norm);
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                let e = spec.n_ff_exp;
                push(
                    format!("{b}ffn_gate_inp.weight"),
                    vec![n_embd, N_EXPERT],
                    Role::Router,
                );
                push(
                    format!("{b}ffn_gate_exps.weight"),
                    vec![n_embd, e, N_EXPERT],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down_exps.weight"),
                    vec![e, n_embd, N_EXPERT],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_up_exps.weight"),
                    vec![n_embd, e, N_EXPERT],
                    Role::Proj,
                );
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch9/10/11_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch12");

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
    // the expanded cache-slot count for hrm-text (lps * h * (l + 1))
    kv!(format!("{a}.block_count"), Value::U32(spec.n_layer as u32));
    kv!(
        format!("{a}.feed_forward_length"),
        Value::U32(spec.n_ff as u32)
    );
    if spec.n_head.iter().all(|&v| v == spec.n_head[0]) {
        kv!(
            format!("{a}.attention.head_count"),
            Value::U32(spec.n_head[0] as u32)
        );
    } else {
        kv!(
            format!("{a}.attention.head_count"),
            Value::Array(
                GgufType::Uint32,
                spec.n_head.iter().map(|&v| Value::U32(v as u32)).collect()
            )
        );
    }
    if spec.head_kv.iter().all(|&v| v == spec.head_kv[0]) {
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::U32(spec.head_kv[0] as u32)
        );
    } else {
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::Array(
                GgufType::Uint32,
                spec.head_kv.iter().map(|&v| Value::U32(v as u32)).collect()
            )
        );
    }
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
        Value::U32(spec.rope_dim as u32)
    );
    kv!(
        format!("{a}.rope.freq_base"),
        Value::F32(spec.freq_base_full)
    );

    match spec.family {
        Family::HrmText => {
            // hrm-text.cpp:7-15
            if spec.hrm_embedding_scale != 0.0 {
                kv!(
                    format!("{a}.embedding_scale"),
                    Value::F32(spec.hrm_embedding_scale)
                );
            }
            kv!(
                format!("{a}.hrm.layers_per_stack"),
                Value::U32(spec.hrm_lps)
            );
            kv!(format!("{a}.hrm.h_cycles"), Value::U32(spec.hrm_h_cycles));
            kv!(format!("{a}.hrm.l_cycles"), Value::U32(spec.hrm_l_cycles));
            // hrm.prefix_lm stays absent (causal attention only)
        }
        Family::Laguna => {
            // laguna.cpp:10-49
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.n_layer_dense_lead)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_shared_feed_forward_length"),
                Value::U32(spec.n_ff_shexp as u32)
            );
            // expert_gating_func absent → the SIGMOID default (laguna.cpp:51-55)
            if let Some(swa) = spec.laguna_swa {
                kv!(format!("{a}.attention.sliding_window"), Value::U32(swa));
                // the dotted KV templates of llama-arch.cpp (%s.rope.freq_base_swa …)
                kv!(
                    format!("{a}.rope.freq_base_swa"),
                    Value::F32(spec.freq_base_swa)
                );
                kv!(
                    format!("{a}.rope.dimension_count_swa"),
                    Value::U32(spec.n_rot_swa as u32)
                );
            }
        }
        Family::Maple => {
            // maple.cpp:6-16
            kv!(format!("{a}.attention.sliding_window"), Value::U32(64));
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(
                    GgufType::Uint32,
                    spec.maple_pattern.iter().map(|&v| Value::U32(v)).collect()
                )
            );
            kv!(
                format!("{a}.rope.freq_base_swa"),
                Value::F32(spec.freq_base_swa)
            );
            kv!(
                format!("{a}.rope.dimension_count_swa"),
                Value::U32(spec.n_rot_swa as u32)
            );
            // REQUIRED but never consumed by the graph (maple.cpp:16)
            kv!(format!("{a}.swiglu_clamp_exp"), Value::F32(80.0));
        }
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
/// missing) — the count check plus a spot lookup per name
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
    assert_eq!(hp.n_embd as i64, spec.n_embd);
    match spec.family {
        Family::HrmText => {
            // the slot algebra of hrm-text.cpp:21-23
            let n_slot = spec.hrm_lps * spec.hrm_h_cycles * (spec.hrm_l_cycles + 1);
            assert_eq!(hp.n_layer(), n_slot);
            assert_eq!(hp.n_hrm_layers_per_stack, spec.hrm_lps);
            assert_eq!(hp.n_hrm_h_cycles, spec.hrm_h_cycles);
            assert_eq!(hp.n_hrm_l_cycles, spec.hrm_l_cycles);
            assert!(!hp.hrm_prefix_lm); // key absent
            assert_eq!(hp.f_embedding_scale, spec.hrm_embedding_scale);
            // the alias passes share the physical stacks' tensors: slots
            // [4,6) alias [0,2), slots [6,8) alias [2,4)
            assert_eq!(m.layers[4].wq, m.layers[0].wq, "hrm alias slot 4");
            assert_eq!(m.layers[7].wo, m.layers[3].wo, "hrm alias slot 7");
        }
        Family::Laguna => match spec.laguna_swa {
            Some(swa) => {
                assert_eq!(hp.n_swa, swa);
                assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
                // the period-4 dense-first template: FULL at il%4==0
                for il in 0..spec.n_layer {
                    assert_eq!(hp.is_swa(il), il % 4 != 0, "laguna is_swa({il})");
                }
                // the per-layer-type rope facts (laguna.cpp:46-48)
                assert_eq!(hp.rope_freq_base_train, spec.freq_base_full);
                assert_eq!(hp.rope_freq_base_train_swa, spec.freq_base_swa);
                assert_eq!(hp.rope_freq_scale_train_swa, 1.0);
                assert_eq!(hp.n_rot_swa, spec.n_rot_swa as u32);
                // per-layer head counts survive the load
                for il in 0..spec.n_layer {
                    assert_eq!(hp.n_head(il) as i64, spec.n_head[il], "laguna n_head({il})");
                }
            }
            None => {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
                assert!(!hp.is_swa_any());
                // n_ff_shexp falls back to n_ff_exp * n_expert_shared only
                // when the key is absent — the file carries it
                assert_eq!(hp.n_ff_shexp as i64, spec.n_ff_shexp);
            }
        },
        Family::Maple => {
            assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            assert_eq!(hp.n_swa, 64);
            for (il, &p) in spec.maple_pattern.iter().enumerate() {
                assert_eq!(hp.is_swa(il), p != 0, "maple pattern[{il}]");
            }
            assert_eq!(hp.rope_freq_base_train_swa, spec.freq_base_swa);
            assert_eq!(hp.n_rot_swa, spec.n_rot_swa as u32);
            // read-but-unused: the clamp key lands in hparams only
            assert_eq!(hp.swiglu_clamp_exp[0], 80.0);
        }
    }
    // the SIGMOID default of both MoE archs (laguna.cpp:51-55)
    if spec.family == Family::Laguna {
        assert_eq!(
            hp.expert_gating_func,
            llama::hparams::LlamaExpertGatingFuncType::SIGMOID as u32
        );
        assert_eq!(hp.n_layer_dense_lead, spec.n_layer_dense_lead);
        assert_eq!(hp.n_expert as i64, N_EXPERT);
        assert_eq!(hp.n_expert_used(0) as i64, N_EXPERT_USED);
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

fn hrm_text_weights(m: &LlamaModel, n_slot: usize) -> graph_arch::HrmTextModelWeights {
    graph_arch::HrmTextModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        hrm_z_l_init: m.hrm_z_l_init.unwrap(),
        layers: m
            .layers
            .iter()
            .take(n_slot)
            .map(|l| graph_arch::HrmTextLayerWeights {
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
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn laguna_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::LagunaModelWeights {
    graph_arch::LagunaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LagunaLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_up: l.ffn_up,
                ffn_down: l.ffn_down,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

fn maple_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MapleModelWeights {
    graph_arch::MapleModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MapleLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = m.hparams.clone();
    let n_trunk = hp.n_layer() as usize;
    let attn = synth_attn(m, fa);
    let fw = match m.arch {
        llama::arch::LlmArch::HRM_TEXT => ForwardWeights::HrmText(
            hrm_text_weights(m, n_trunk),
            graph_arch::HrmTextParams {
                attn,
                n_layers_per_stack: hp.n_hrm_layers_per_stack as usize,
                n_h_cycles: hp.n_hrm_h_cycles as usize,
                n_l_cycles: hp.n_hrm_l_cycles as usize,
                f_embedding_scale: hp.f_embedding_scale,
            },
        ),
        llama::arch::LlmArch::LAGUNA => ForwardWeights::Laguna(
            laguna_weights(m, n_trunk),
            graph_arch::LagunaParams {
                attn,
                n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                has_swa: hp.swa_type != LlamaSwaType::NONE && hp.is_swa_any(),
                n_rot_swa: hp.n_rot_swa as i32,
                freq_base_swa: hp.rope_freq_base_train_swa,
                freq_scale_swa: hp.rope_freq_scale_train_swa,
                n_ctx_train: hp.n_ctx_train as i32,
                n_layer_dense_lead: hp.n_layer_dense_lead,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                expert_gating_func: hp.expert_gating_func as i32,
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
            },
        ),
        llama::arch::LlmArch::MAPLE => ForwardWeights::Maple(
            maple_weights(m, n_trunk),
            graph_arch::MapleParams {
                attn,
                is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                freq_base_swa: hp.rope_freq_base_train_swa,
                freq_scale_swa: hp.rope_freq_scale_train_swa,
                n_rot_swa: hp.n_rot_swa as i32,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
            },
        ),
        other => panic!("arch {other:?} not in batch 12"),
    };
    (fw, attn)
}

// ---------------------------------------------------------------------------
// the decode harness — DecodeContext itself (the batch's context.rs arms);
// SWA models get the iswa pair exactly like llama-model.cpp:2687-2690
// ---------------------------------------------------------------------------

fn driver_for(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let hp = &m.hparams;
    if hp.swa_type != LlamaSwaType::NONE && hp.is_swa_any() {
        DecodeContext::new_with_swa(
            gctx,
            weights,
            attn,
            512,
            8,
            512,
            SwaCacheSpec::from_hparams(hp),
        )
    } else {
        DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
    }
}

/// one prefill (a 12-token prompt) + one decode step; asserts the caches
/// landed where they should
fn smoke_forward(m: &mut LlamaModel, spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut dctx = driver_for(m, fa);
    let prompt: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    let tk = logits_of_argmax(logits.chunks(32000).last().unwrap());
    let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
    // 13 cells per layer's cache (used_cells is one layer's position count)
    assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the tensor/hparams pins + the double-FA smoke of the parity cells
#[test]
fn arch_batch12_pin_and_smoke() {
    for spec in parity_specs() {
        let mut m = load_synth(&spec);
        pin_hparams(&m, &spec);
        pin_tensors(&m, &spec);
        for fa in [false, true] {
            // the model's Context moves into the driver — reload per FA mode
            let mut m = load_synth(&spec);
            let logits = smoke_forward(&mut m, &spec, fa);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{} fa={fa}: non-finite logits",
                spec.arch
            );
        }
        println!("{}: pinned + smoke ok (both FA modes)", spec.arch);
    }
}

/// the >64-token prompt cell of the parity protocol, in one ubatch:
///   * laguna (XS.2-like) / maple — positions past the n_swa 64 window: the
///     SWA layers' window masks bind on both sides of the iswa pair, and the
///     SWA layers' rope runs at positions > 64 with the plain-rope params
///   * hrm-text — the zH + zL state carried across ~100 tokens
#[test]
fn arch_batch12_long_prompt_cells() {
    for spec in parity_specs() {
        let mut m = load_synth(&spec);
        let mut dctx = driver_for(&mut m, false);
        let n = 100usize;
        let prompt: Vec<i32> = (1..=n as i32).collect();
        let pos: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let lg = dctx.decode(&prompt, &pos).expect("long prefill").to_vec();
        assert!(lg.iter().all(|v| v.is_finite()));
        assert_eq!(dctx.kv.used_cells(), n as u32, "{}: kv cells", spec.arch);
        let tk = logits_of_argmax(&lg);
        let next = dctx.decode(&[tk], &[n as i32]).expect("decode").to_vec();
        assert!(next.iter().all(|v| v.is_finite()));
        println!("{}: 100-token cell ok", spec.arch);
    }
}

/// maple vs a same-geometry file where every layer ropes: the SWA-only rope
/// skip must change the logits (maple.cpp:90-99 — the full-attention layers
/// never rotate, so their K rows are raw)
#[test]
fn arch_batch12_maple_swa_rope_changes_logits() {
    // the all-SWA variant: pattern [1,1,1,1] ropes every layer
    let spec_all = spec_maple().with(|s| {
        s.suffix = "-allswa";
        s.maple_pattern = vec![1, 1, 1, 1];
    });
    let mut m1 = load_synth(&spec_maple());
    let mut m2 = load_synth(&spec_all);
    let lg1 = {
        let mut d = driver_for(&mut m1, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let lg2 = {
        let mut d = driver_for(&mut m2, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let diff: f32 = lg1
        .iter()
        .zip(&lg2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        diff > 0.0,
        "the SWA-only rope must change the logits (max |d| = {diff})"
    );
    println!("maple swa-vs-allswa rope max |Δlogit| = {diff:.4}");
}

/// laguna-swa vs laguna-full (same weights per-layer modulo the head counts):
/// the per-layer-type RoPE + iswa windows vs the plain all-full path must
/// produce different logits — a wiring sanity check that the two cells
/// exercise genuinely different code paths
#[test]
fn arch_batch12_laguna_swa_vs_full_differ() {
    let mut m1 = load_synth(&spec_laguna());
    let mut m2 = load_synth(&spec_laguna_full());
    let lg1 = {
        let mut d = driver_for(&mut m1, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let lg2 = {
        let mut d = driver_for(&mut m2, false);
        d.decode(&[1, 2, 3], &[0, 1, 2]).unwrap().to_vec()
    };
    let diff: f32 = lg1
        .iter()
        .zip(&lg2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff > 0.0, "laguna swa/full must differ (max |d| = {diff})");
    println!("laguna swa-vs-full max |Δlogit| = {diff:.4}");
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}
