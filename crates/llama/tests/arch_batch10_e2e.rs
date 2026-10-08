//! arch_batch10_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-10: **the small-arch + EXP-op batch** — smallthinker / llada-moe /
//! minimax-01 / graniteswitch (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-9 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//!   * smallthinker — the no-SWA shape (every layer ropes,
//!     `n_no_rope_layer_step == n_layer`), the probs_in router (the MoE eats
//!     logits computed from the RAW inpL before attn_norm) + the ReGLU expert
//!     activation (`ggml_reglu_split`) + the norm_w weight normalization, the
//!     fused `attn_qkv`; a `-swa` variant carries the iswa graph template
//!     (window key present → n_swa re-pinned 4096, pattern 4 dense-first,
//!     layer 0 ropes NOT — the `il % step == 0` skip — freq_base_swa 30000);
//!   * llada-moe — the non-causal diffusion attention over
//!     `build_attn_inp_no_cache` (an all-visible [n_tokens, n_tokens] mask —
//!     a decode step attends ONLY the tokens of its own ubatch), per-head
//!     q/k norms before rope, the dense-SiLU softmax MoE with the
//!     `n_ff/n_expert_used` fallback;
//!   * minimax-01 — lightning attention: explicit
//!     `attention.recurrent_layers` [1,1,0,1] (layer 2 is the softmax GQA
//!     layer, partial rope 8 < head 16), the `llm_graph_input_la` decay
//!     inputs (slopes/q_decay/k_decay/diag_decay from the positions), the
//!     per-layer `slope_scale`, `ggml_exp` on all three decays + the
//!     block decay, the n_embd_head_la²·n_head recurrent state, the
//!     attn_norm_2 RMS over the flattened heads, the sigmoid output gate,
//!     the residual_scale pair and the norm_w softmax MoE with scale 2.0;
//!   * graniteswitch — the adapter router: `adapters.count` 2 / `lora_rank`
//!     8 / `router_gain` 10.0 (large enough that the causal softmax readback
//!     recovers the slot exactly), activate tokens [101, 202] substituting
//!     [111, 222], the in-graph router head (pad → single-head causal
//!     attention → clamp+round+cast), the per-token switched LoRA deltas on
//!     every projection (two mul_mat_id each), the router layer at index
//!     n_layer (1 head, no rope, skipped by the K-shift), the granite
//!     scalars (embedding/residual/logit scale) and NORM rope.
//!
//! The batch's ForwardWeights/CLI arms landed with the graph (context.rs),
//! so the default-run tests drive `DecodeContext::new_with{,_swa}` itself
//! (batch-6 protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH10=1 ./parity/arch_batch_parity.sh …`).

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

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch10";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Smallthinker,
    LladaMoe,
    Minimax01,
    GraniteSwitch,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    head_kv: Vec<i64>,
    key_length: i64,
    value_length: i64,
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    n_ff_exp: i64,
    gating: Option<u32>,
    weights_scale: Option<f32>,
    /// smallthinker: write the sliding_window key (the iswa template)
    swa: bool,
    freq_base_swa: Option<f32>,
    /// minimax-01: explicit attention.recurrent_layers
    recr_layers: Option<Vec<u32>>,
    residual_scale: f32,
    /// graniteswitch: the adapter tables
    n_adapters: u32,
    lora_rank: u32,
    router_gain: f32,
    activate_tokens: Vec<i32>,
    substitute_tokens: Vec<i32>,
    embedding_scale: Option<f32>,
    attention_scale: Option<f32>,
    logit_scale: Option<f32>,
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
    fn is_recr(&self, il: usize) -> bool {
        match self.family {
            Family::Smallthinker | Family::LladaMoe | Family::GraniteSwitch => false,
            Family::Minimax01 => match &self.recr_layers {
                Some(v) => v[il] != 0,
                // the interval-8 default — every one of our 4 layers would be
                // lightning, so the spec always pins the array for minimax
                None => (il as u32 + 1) % 8 != 0,
            },
        }
    }
    fn n_slots(&self) -> i64 {
        self.n_adapters as i64 + 1
    }
}

fn base(arch: &'static str, family: Family) -> SynthSpec {
    SynthSpec {
        arch,
        suffix: "",
        family,
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        head_kv: vec![2; 4],
        key_length: 32,
        value_length: 32,
        rope_dim: 32,
        n_ff: 96,
        n_ctx: 512,
        write_output: true,
        n_ff_exp: 32,
        gating: None,
        weights_scale: None,
        swa: false,
        freq_base_swa: None,
        recr_layers: None,
        residual_scale: 1.0,
        n_adapters: 0,
        lora_rank: 0,
        router_gain: 0.0,
        activate_tokens: Vec::new(),
        substitute_tokens: Vec::new(),
        embedding_scale: None,
        attention_scale: None,
        logit_scale: None,
    }
}

/// smallthinker — the no-SWA shape (every layer ropes; n_no_rope_layer_step
/// collapses to n_layer), probs_in + ReGLU MoE, tied output
fn spec_smallthinker() -> SynthSpec {
    base("smallthinker", Family::Smallthinker).with(|s| {
        s.gating = Some(1); // SOFTMAX — NONE aborts the reference's build_moe_ffn
        s.write_output = false;
        // NB: smallthinker.cpp never reads expert_weights_scale (it stays 0 —
        // the same never-read trap as qwen3next)
    })
}

/// smallthinker `-swa` — the iswa graph template: the window key flips the
/// hparams (n_swa 4096, pattern 4 dense-first, freq_base_swa 30000) and layer
/// 0 stops roping (`il % n_no_rope_layer_step == 0`)
fn spec_smallthinker_swa() -> SynthSpec {
    spec_smallthinker().with(|s| {
        s.suffix = "-swa";
        s.swa = true;
        s.freq_base_swa = Some(30000.0);
    })
}

/// llada-moe — the non-causal no-cache diffusion attention, per-head q/k
/// norms, dense-SiLU softmax MoE
fn spec_llada_moe() -> SynthSpec {
    base("llada-moe", Family::LladaMoe).with(|s| {
        s.n_embd = 128;
        s.head_kv = vec![2; 4];
        s.key_length = 32; // head_dim * n_head == n_embd
        s.value_length = 32;
        s.rope_dim = 32;
    })
}

/// minimax-01 — lightning attention on layers 0/1/3, softmax GQA on layer 2
/// (partial rope 8 < head 16), residual_scale 0.1; the experts sit at the
/// dense n_ff (n_ff_exp is never read)
fn spec_minimax01() -> SynthSpec {
    base("minimax-01", Family::Minimax01).with(|s| {
        s.n_embd = 64;
        s.n_head = 4;
        s.head_kv = vec![2; 4];
        s.key_length = 16; // n_embd_head_la — n_embd_s = 16² * 4
        s.value_length = 16;
        s.rope_dim = 8; // partial rope (the real model: 64 < 128)
        s.n_ff = 64;
        s.n_ff_exp = 64; // the loader asks for the dense n_ff
        s.recr_layers = Some(vec![1, 1, 0, 1]);
        s.residual_scale = 0.1;
        s.write_output = false;
    })
}

/// graniteswitch — 3 trunk layers + the router layer, 2 adapters of rank 8,
/// router_gain 10.0, activate [101, 202] → substitute [111, 222]
fn spec_graniteswitch() -> SynthSpec {
    base("graniteswitch", Family::GraniteSwitch).with(|s| {
        s.n_layer = 3; // block_count 3 → n_layer_all 4 (the router layer)
        s.head_kv = vec![2; 3];
        s.key_length = 16;
        s.value_length = 16;
        s.rope_dim = 16; // full-head NORM rope
        s.n_ff = 32;
        s.n_adapters = 2;
        s.lora_rank = 8;
        s.router_gain = 10.0;
        s.activate_tokens = vec![101, 202];
        s.substitute_tokens = vec![111, 222];
        s.embedding_scale = Some(12.0); // granite-4.0's embedding scale
        s.residual_scale = 0.022; // granite-4.0's residual scale
        s.logit_scale = Some(8.0);
        s.write_output = false;
    })
}

/// every file the ignored writer test emits (incl. the in-port-only ones —
/// llada-moe has no reference-side parity cell, see `parity_specs`)
fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_smallthinker(),
        spec_smallthinker_swa(),
        spec_llada_moe(),
        spec_minimax01(),
        spec_graniteswitch(),
    ]
}

/// the parity cells (batch-10 default set): smallthinker / minimax-01 /
/// graniteswitch — llada-moe has NO reference-side cell: the pinned reference
/// creates no memory for the diffusion archs (llama-model.cpp:2289-2295,
/// `res = nullptr` for LLADA/LLADA_MOE/DREAM/RND1) and both llama-server
/// ("the current context does not logits computation",
/// server-context.cpp:3177-3180) and llama-cli refuse to generate on a
/// memory-less context. The port's no-cache graph is verified in-port
/// (arch_batch10_pin_and_smoke / arch_batch10_long_prompt_cells).
fn parity_specs() -> Vec<SynthSpec> {
    vec![spec_smallthinker(), spec_minimax01(), spec_graniteswitch()]
}

#[test]
#[ignore = "writes the /tmp/arch-batch10 parity files (ARCH_BATCH10 cells)"]
fn arch_batch10_write_synth() {
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
    /// softmax/sigmoid over the router logits stays spread out
    Router,
    /// decay/gate vectors — small values keep exp() bounded
    Decay,
}

type TensorSpec = (String, Vec<i64>);

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let n_exp = N_EXPERT;
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    push!("output_norm.weight", vec![n_embd], Role::Norm);
    if spec.write_output {
        push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);
    }

    match spec.family {
        Family::Smallthinker => {
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length;
                let kv = spec.head_kv[i] * spec.key_length;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.key_length, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
            }
        }
        Family::LladaMoe => {
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length; // == n_embd
                let kv = spec.head_kv[i] * spec.key_length;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![q, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
            }
        }
        Family::Minimax01 => {
            let q = spec.n_head * spec.key_length;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if !spec.is_recr(i) {
                    let kkv = spec.head_kv[i] * spec.key_length;
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, q + 2 * kkv],
                        Role::Proj
                    );
                } else {
                    push!(format!("blk.{i}.attn_norm_2.weight"), vec![q], Role::Norm);
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, 3 * q],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_gate.weight"),
                        vec![n_embd, q],
                        Role::Proj
                    );
                }
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![q, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
            }
        }
        Family::GraniteSwitch => {
            let n_slots = spec.n_slots();
            let n_rank = spec.lora_rank as i64;
            let q = spec.n_head * spec.key_length;
            let kv = spec.head_kv[0] * spec.key_length;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![q, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                // the stacked switch-LoRA slots (granite-switch.cpp:115-130)
                push!(
                    format!("blk.{i}.attn_q.lora_a"),
                    vec![n_embd, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_q.lora_b"),
                    vec![n_rank, q, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_k.lora_a"),
                    vec![n_embd, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_k.lora_b"),
                    vec![n_rank, kv, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_v.lora_a"),
                    vec![n_embd, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_v.lora_b"),
                    vec![n_rank, kv, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_output.lora_a"),
                    vec![q, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.attn_output.lora_b"),
                    vec![n_rank, n_embd, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_gate.lora_a"),
                    vec![n_embd, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_gate.lora_b"),
                    vec![n_rank, n_ff, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_up.lora_a"),
                    vec![n_embd, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_up.lora_b"),
                    vec![n_rank, n_ff, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_down.lora_a"),
                    vec![n_ff, n_rank, n_slots],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.ffn_down.lora_b"),
                    vec![n_rank, n_embd, n_slots],
                    Role::Decay
                );
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch9_e2e.rs)
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
        // the switch-LoRA slots: small so the deltas perturb rather than
        // dominate the base projections (and slot 0 stays a valid zero-ish
        // base — the router clamps to it for plain tokens)
        Role::Decay => 0.01,
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch10");

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
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    match spec.family {
        Family::Smallthinker => {
            if spec.swa {
                kv!(format!("{a}.attention.sliding_window"), Value::U32(512));
            }
            if let Some(v) = spec.freq_base_swa {
                kv!(format!("{a}.rope.freq_base_swa"), Value::F32(v));
            }
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            if let Some(v) = spec.gating {
                kv!(format!("{a}.expert_gating_func"), Value::U32(v));
            }
        }
        Family::LladaMoe => {
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
        Family::Minimax01 => {
            kv!(
                format!("{a}.residual_scale"),
                Value::F32(spec.residual_scale)
            );
            if let Some(v) = &spec.recr_layers {
                kv!(
                    format!("{a}.attention.recurrent_layers"),
                    Value::Array(GgufType::Uint32, v.iter().map(|&x| Value::U32(x)).collect())
                );
            }
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            // NB: no expert_feed_forward_length — minimax-01 never reads it
            // (the experts sit at the dense n_ff)
        }
        Family::GraniteSwitch => {
            kv!(
                format!("{a}.logit_scale"),
                Value::F32(spec.logit_scale.unwrap())
            );
            kv!(
                format!("{a}.residual_scale"),
                Value::F32(spec.residual_scale)
            );
            kv!(
                format!("{a}.embedding_scale"),
                Value::F32(spec.embedding_scale.unwrap())
            );
            if let Some(v) = spec.attention_scale {
                kv!(format!("{a}.attention_scale"), Value::F32(v));
            }
            kv!(format!("{a}.rope.scaling.finetuned"), Value::Bool(true));
            kv!(format!("{a}.adapters.count"), Value::U32(spec.n_adapters));
            kv!(
                format!("{a}.adapters.lora_rank"),
                Value::U32(spec.lora_rank)
            );
            kv!(
                format!("{a}.adapters.router_gain"),
                Value::F32(spec.router_gain)
            );
            kv!(
                format!("{a}.adapters.token_ids_activate"),
                Value::Array(
                    GgufType::Int32,
                    spec.activate_tokens
                        .iter()
                        .map(|&x| Value::I32(x))
                        .collect()
                )
            );
            kv!(
                format!("{a}.adapters.token_ids_substitute"),
                Value::Array(
                    GgufType::Int32,
                    spec.substitute_tokens
                        .iter()
                        .map(|&x| Value::I32(x))
                        .collect()
                )
            );
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
    assert_eq!(hp.n_layer() as usize, spec.n_layer);
    assert_eq!(hp.n_embd as i64, spec.n_embd);
    for il in 0..spec.n_layer {
        assert_eq!(
            hp.is_recr(il),
            spec.is_recr(il),
            "{}: is_recr[{il}]",
            spec.arch
        );
    }
    match spec.family {
        Family::Smallthinker => {
            if spec.swa {
                assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
                assert_eq!(hp.n_swa, 4096); // re-pinned by the loader
                assert_eq!(hp.rope_freq_base_train_swa, 30000.0);
                // load_swa_pattern(ml, 4, dense_first=true): il%4==0 is full
                assert_eq!(hp.is_swa(0), false);
                assert_eq!(hp.is_swa(1), true);
            } else {
                assert_eq!(hp.swa_type, LlamaSwaType::NONE);
                assert_eq!(hp.n_no_rope_layer_step as usize, spec.n_layer);
            }
            assert_eq!(hp.expert_gating_func, 1); // SOFTMAX
            assert_eq!(hp.expert_weights_scale, 0.0); // never read (smallthinker.cpp)
        }
        Family::LladaMoe => {
            assert!(!hp.causal_attn, "llada-moe.cpp:8 — non-causal");
        }
        Family::Minimax01 => {
            assert_eq!(hp.n_embd_head_la as i64, spec.key_length);
            assert_eq!(hp.f_residual_scale, 0.1);
            // n_embd_s = head_la² * n_head (llama-hparams.cpp:249-253)
            assert_eq!(
                hp.n_embd_s(),
                (spec.key_length * spec.key_length * spec.n_head) as u32
            );
            assert_eq!(hp.n_embd_r(), 0); // no ssm keys — the zero conv cell
        }
        Family::GraniteSwitch => {
            assert_eq!(hp.router_layer as usize, spec.n_layer);
            assert_eq!(hp.n_layer_all as usize, spec.n_layer + 1);
            assert_eq!(hp.n_layer_nextn, 1);
            assert_eq!(hp.graniteswitch_n_adapters, spec.n_adapters);
            assert_eq!(hp.graniteswitch_max_lora_rank, spec.lora_rank);
            assert_eq!(hp.graniteswitch_router_gain, spec.router_gain);
            // the extra single-head router layer (granite-switch.cpp:69-71)
            assert_eq!(hp.n_head(spec.n_layer), 1);
            assert_eq!(hp.n_head_kv(spec.n_layer), 1);
            assert_eq!(hp.n_ff(spec.n_layer), 0);
            assert!(!hp.has_rope(spec.n_layer));
            assert!(hp.has_rope(0));
        }
    }
}

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
    let hp = &m.hparams;
    let first_attn = (0..hp.n_layer() as usize)
        .find(|&il| match m.arch {
            llama::arch::LlmArch::MINIMAX_01 => !hp.is_recr(il),
            _ => true,
        })
        .unwrap_or(0);
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(first_attn) as i64,
        n_head_kv: hp.n_head_kv(first_attn) as i64,
        n_embd_head_k: hp.n_embd_head_k(first_attn) as i64,
        n_embd_head_v: hp.n_embd_head_v(first_attn) as i64,
        n_rot: hp.n_rot(first_attn) as i64,
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
// the per-arch weight bundles (the CLI wiring of main.rs, mirrored)
// ---------------------------------------------------------------------------

fn smallthinker_weights(m: &LlamaModel) -> graph_arch::SmallthinkerModelWeights {
    graph_arch::SmallthinkerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::SmallthinkerLayerWeights {
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
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn llada_moe_weights(m: &LlamaModel) -> graph_arch::LladaMoeModelWeights {
    graph_arch::LladaMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::LladaMoeLayerWeights {
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

fn minimax01_weights(m: &LlamaModel) -> graph_arch::Minimax01ModelWeights {
    graph_arch::Minimax01ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Minimax01LayerWeights {
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
                attn_norm_2: l.attn_norm_2,
                wqkv_la: l.wqkv,
                wg: l.wqkv_gate,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_exp_probs_b: l.ffn_exp_probs_b,
            })
            .collect(),
    }
}

fn graniteswitch_weights(m: &LlamaModel) -> graph_arch::GraniteSwitchModelWeights {
    graph_arch::GraniteSwitchModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(m.hparams.n_layer() as usize)
            .map(|l| graph_arch::GraniteSwitchLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                sl: l.switch_lora.unwrap(),
            })
            .collect(),
        token_to_slot: m.graniteswitch_token_to_slot.clone(),
        token_to_substitute: m.graniteswitch_token_to_substitute.clone(),
    }
}

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = synth_attn(m, fa);
    let first_attn = (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap_or(0);
    match m.arch {
        llama::arch::LlmArch::SMALLTHINKER => (
            ForwardWeights::Smallthinker(
                smallthinker_weights(m),
                graph_arch::SmallthinkerParams {
                    attn,
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    n_layer: n_trunk,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::LLADA_MOE => (
            ForwardWeights::LladaMoe(
                llada_moe_weights(m),
                graph_arch::LladaMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::MINIMAX_01 => (
            ForwardWeights::Minimax01(
                minimax01_weights(m),
                graph_arch::Minimax01Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_head: hp.n_head(first_attn) as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    f_residual_scale: hp.f_residual_scale,
                    n_embd_head_la: hp.n_embd_head_la as i64,
                    n_embd_s: hp.n_embd_s(),
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::GRANITE_SWITCH => (
            ForwardWeights::GraniteSwitch(
                graniteswitch_weights(m),
                graph_arch::GraniteSwitchParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    router_layer: hp.router_layer as usize,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                    has_rope: (0..=n_trunk).map(|il| hp.has_rope(il)).collect(),
                    n_ff: hp.n_ff(0) as i64,
                    f_logit_scale: hp.f_logit_scale,
                    f_residual_scale: hp.f_residual_scale,
                    f_embedding_scale: hp.f_embedding_scale,
                    f_attention_scale: hp.f_attention_scale,
                    n_adapters: hp.graniteswitch_n_adapters as i64,
                    router_gain: hp.graniteswitch_router_gain,
                },
            ),
            attn,
        ),
        other => panic!("arch {other:?} not in batch 10"),
    }
}

// ---------------------------------------------------------------------------
// the decode harness — DecodeContext itself (the batch's context.rs arms)
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

/// one prefill (a 12-token prompt) + one decode step; asserts the caches and
/// recurrent cells landed where they should
fn smoke_forward(m: &mut LlamaModel, spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut dctx = driver_for(m, fa);
    let prompt: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    let tk = logits_of_argmax(logits.chunks(32000).last().unwrap());
    let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
    match spec.family {
        // the diffusion arch's graph never touches the cache, but the driver
        // still owns one (the reference's llama_kv_cache) and the cells fill
        Family::LladaMoe | Family::GraniteSwitch | Family::Smallthinker => {
            assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
        }
        // minimax-01: the softmax layers cache (2 cells each step over the
        // single such layer), the lightning layers carry the la state instead
        Family::Minimax01 => {
            let n_attn = (0..spec.n_layer).filter(|&il| !spec.is_recr(il)).count();
            assert_eq!(
                dctx.kv.used_cells(),
                (13 * n_attn) as u32,
                "{} fa={fa}: kv cells",
                spec.arch
            );
        }
    }
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the tensor/hparams pins of the in-port-only cells too (llada-moe + the
/// smallthinker iswa variant): the loaders must consume them exactly
#[test]
fn arch_batch10_inport_only_pins() {
    for spec in [spec_llada_moe(), spec_smallthinker_swa()] {
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
        println!("{}: in-port pin + smoke ok (both FA modes)", spec.path());
    }
}

#[test]
fn arch_batch10_pin_and_smoke() {
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

#[test]
fn arch_batch10_variant_swa() {
    // the iswa graph template: layer 0 carries no rope, the SWA layers
    // (1..3) read freq_base_swa; the driver builds the iswa cache pair
    let spec = spec_smallthinker_swa();
    let mut m = load_synth(&spec);
    pin_hparams(&m, &spec);
    pin_tensors(&m, &spec);
    for fa in [false, true] {
        let mut m = load_synth(&spec);
        let logits = smoke_forward(&mut m, &spec, fa);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "smallthinker-swa fa={fa}: non-finite logits"
        );
    }
    println!("{}: iswa variant ok", spec.path());
}

#[test]
fn arch_batch10_long_prompt_cells() {
    // the >64-token prompt cell of the parity protocol, in one ubatch:
    //   * llada-moe — the [T, T] non-causal mask over ~100 tokens
    //   * minimax-01 — the diag_decay matrix, the decays at pos_rel up to
    //     ~100 and the block decay exp(-slope·T)
    let spec = spec_llada_moe();
    let mut m = load_synth(&spec);
    let mut dctx = driver_for(&mut m, false);
    let n = 100usize;
    let prompt: Vec<i32> = (1..=n as i32).collect();
    let pos: Vec<i32> = (0..n).map(|i| i as i32).collect();
    let lg = dctx.decode(&prompt, &pos).expect("long prefill").to_vec();
    assert!(lg.iter().all(|v| v.is_finite()));
    assert_eq!(dctx.kv.used_cells(), n as u32);
    let tk = logits_of_argmax(&lg);
    let next = dctx.decode(&[tk], &[n as i32]).expect("decode").to_vec();
    assert!(next.iter().all(|v| v.is_finite()));

    let spec = spec_minimax01();
    let mut m = load_synth(&spec);
    let mut dctx = driver_for(&mut m, false);
    let mut tk = 1i32;
    let mut logits = Vec::new();
    // 80 single-token steps: the la state evolves far past the prompt, the
    // softmax layer's kv cache grows to 80
    for p in 0..80 {
        let lg = dctx.decode(&[tk], &[p]).expect("decode").to_vec();
        tk = logits_of_argmax(&lg);
        logits = lg;
    }
    assert!(logits.iter().all(|v| v.is_finite()));
    // one attention layer × 80 cells
    assert_eq!(dctx.kv.used_cells(), 80);
}

#[test]
fn arch_batch10_adapter_routing() {
    // graniteswitch's router: a decode step whose token IS an activate token
    // must (a) embed the substitute id and (b) select the adapter's slot.
    // With router_gain 10 the softmax readback is exact, so the logits of
    // [.., 101, ..] (adapter) and [.., 111, ..] (raw substitute) must differ
    // — the switch-LoRA deltas only fire through the adapter path.
    let spec = spec_graniteswitch();
    let mut m = load_synth(&spec);
    let mut dctx_plain = {
        let mut m2 = load_synth(&spec);
        driver_for(&mut m2, false)
    };
    let mut dctx_adapt = driver_for(&mut m, false);

    // prompt [1, 2, 111] — plain tokens + the raw substitute id
    let lg_plain = dctx_plain
        .decode(&[1, 2, 111], &[0, 1, 2])
        .expect("plain")
        .to_vec();
    // prompt [1, 2, 101] — 101 is the activate token: the graph embeds 111
    // (the substitute) but routes slot 1 through the switch deltas
    let lg_adapt = dctx_adapt
        .decode(&[1, 2, 101], &[0, 1, 2])
        .expect("adapted")
        .to_vec();
    assert!(lg_plain.iter().all(|v| v.is_finite()));
    assert!(lg_adapt.iter().all(|v| v.is_finite()));
    // the same embedding enters both, only the LoRA slot differs
    let diff: f32 = lg_plain
        .iter()
        .zip(&lg_adapt)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        diff > 0.0,
        "the adapter slot must change the logits (max |d| = {diff})"
    );
    println!("graniteswitch: adapter-vs-plain max |Δlogit| = {diff:.4}");

    // and a mixed decode continues cleanly across the boundary
    let next = dctx_adapt
        .decode(&[logits_of_argmax(&lg_adapt)], &[3])
        .expect("decode after adapter")
        .to_vec();
    assert!(next.iter().all(|v| v.is_finite()));
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}
