//! arch_batch8_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-30: **the MoE long-tail family** — hunyuan-moe / dots1 /
//! bailingmoe / bailingmoe2 / glm4-moe / minimax-m2 / cohere2moe / exaone-moe
//! (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-7 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32. Every file carries `n_expert = 4`,
//! `n_expert_used = 2` so `ggml_argsort_top_k` / `ggml_mul_mat_id` routing is
//! actually exercised, and each arch's routing knobs are live:
//!
//!   * hunyuan-moe — MoE `norm_w = true` + **both** the shared-expert MLP and
//!     the routed experts (hunyuan-moe.cpp:137-161);
//!   * dots1 — norm_w from the GGUF (true) + softmax gating + the optional
//!     `exp_probs_b` bias + the fat n_ff_exp*n_expert_shared shared expert;
//!   * bailingmoe — norm_w true + softmax, kq_scale = 1/sqrt(n_rot);
//!   * bailingmoe2 — fused `attn_qkv`, norm-before-rope q/k norms, dense
//!     lead, REQUIRED gating key, `exp_probs_b` + `n_ff_shexp` override;
//!   * glm4moe — gating key *absent* (defaults to SIGMOID, glm4-moe.cpp:
//!     16-18), REQUIRED `exp_probs_b`, attn_post_norm as the FFN norm, 2
//!     shared experts, q/k norms on half the layers (the 355B variant flag);
//!   * minimax-m2 — full-width q/k norms, partial rope (n_rot 64 < head 128),
//!     SIGMOID gating + REQUIRED bias, experts at the dense n_ff;
//!   * cohere2moe — the iswa every-4th dense-first SWA pattern with
//!     n_swa = 64 (the `-long` cell crosses the window), the fused
//!     `ffn_gate_up_exps` tensor + sigmoid gating + the (moe+shexp)*0.5
//!     re-scale + logit_scale; a `-sep-ln` variant with separate gate/up and
//!     the LLM_NORM (layer_norm_epsilon) norm type;
//!   * exaone-moe — iswa pattern 4 (NOT dense-first) with n_swa = 64, rope
//!     only on the SWA layers (layer 3 skips rope), REQUIRED gating.
//!
//! The batch's ForwardWeights/CLI arms landed with the graph (context.rs),
//! so the default-run tests drive `DecodeContext::new_with{,_swa}` itself
//! (batch-6 protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH8=1 ./parity/arch_batch_parity.sh …`).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaSwaType;
use llama::kv_cache::SwaCacheSpec;
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch8";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec — small dims, real head geometry
// ---------------------------------------------------------------------------

/// where the arch's q/k norm weights sit (and whether they exist at all)
#[derive(Clone, Copy, PartialEq)]
enum QkNorms {
    /// none at all
    None,
    /// per-head [n_embd_head_k] (dots1 / bailingmoe2 / exaone-moe)
    PerHead,
    /// per-head, optional in the file — written on layers >= n_layer/2 only
    /// (glm4-moe's 355B variant tensors)
    PerHeadPartial,
    /// full-width [n_embd_head_k * n_head] / [n_embd_k_gqa] (minimax-m2)
    FullWidth,
}

#[derive(Clone)]
struct SynthSpec {
    /// the GGUF `general.architecture` name (llama-arch.cpp)
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    /// attention.key_length / value_length (defaults to n_embd/n_head)
    key_length: i64,
    /// rope.dimension_count (defaults to key_length)
    rope_dim: i64,
    n_ff: i64,
    n_ff_exp: i64,
    /// expert_shared_feed_forward_length (hunyuan-moe / bailingmoe2)
    n_ff_shexp: Option<i64>,
    /// expert_shared_count
    n_expert_shared: u32,
    dense_lead: u32,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    /// write `ffn_exp_probs_b.bias` on the MoE layers (glm4moe / minimax-m2
    /// require it, dots1 / bailingmoe2 / exaone-moe carry it optionally)
    exp_probs_b: bool,
    /// write the fused `attn_qkv.weight` instead of separate q/k/v
    /// (bailingmoe2's loader prefers it)
    fused_qkv: bool,
    /// write the fused `ffn_gate_up_exps.weight` (cohere2moe's
    /// create_tensor_gate_up_exps prefers it)
    fused_gate_up: bool,
    /// write `attention.layer_norm_epsilon` instead of the RMS eps
    /// (cohere2moe's LLM_NORM fallback)
    norm_ln: bool,
    qk_norms: QkNorms,
    gating: Option<u32>,
    weights_norm: bool,
    weights_scale: Option<f32>,
    logit_scale: Option<f32>,
    /// attention.sliding_window (cohere2moe / exaone-moe REQUIRE it)
    n_swa: Option<u32>,
}

impl SynthSpec {
    fn n_embd_head(&self) -> i64 {
        self.key_length
    }
    fn n_embd_kv(&self) -> i64 {
        self.n_head_kv * self.n_embd_head()
    }
    /// the shared-expert width each arch's loader derives
    fn n_ff_shexp_of(&self) -> i64 {
        if let Some(v) = self.n_ff_shexp {
            v
        } else {
            self.n_ff_exp * self.n_expert_shared as i64
        }
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

fn base(arch: &'static str) -> SynthSpec {
    SynthSpec {
        arch,
        suffix: "",
        n_layer: 4,
        n_embd: 128,
        n_head: 4,
        n_head_kv: 2,
        key_length: 32,
        rope_dim: 32,
        n_ff: 96,
        n_ff_exp: 32,
        n_ff_shexp: None,
        n_expert_shared: 1,
        dense_lead: 0,
        n_ctx: 512,
        write_output: true,
        exp_probs_b: false,
        fused_qkv: false,
        fused_gate_up: false,
        norm_ln: false,
        qk_norms: QkNorms::None,
        gating: None,
        weights_norm: false,
        weights_scale: None,
        logit_scale: None,
        n_swa: None,
    }
}

/// hunyuan-moe — every layer MoE with BOTH the shared-expert MLP and the
/// routed experts; q/k norms AFTER rope (hunyuan-moe.cpp:110-118)
fn spec_hunyuan_moe() -> SynthSpec {
    base("hunyuan-moe").with(|s| {
        s.write_output = false; // the A13B ties the head (hunyuan-moe.cpp:21-25)
        s.qk_norms = QkNorms::PerHead;
        s.n_ff_shexp = Some(24); // expert_shared_feed_forward_length
    })
}

/// dots1 — one dense lead layer, MHA (n_head_kv == n_head), softmax gating
/// from the GGUF, norm_w true, the DeepSeek-V3 style router bias and the fat
/// shared expert (dots1.cpp:3-16)
fn spec_dots1() -> SynthSpec {
    base("dots1").with(|s| {
        s.n_layer = 4;
        s.n_head_kv = s.n_head; // dots1.cpp:34 — MHA widths
        s.qk_norms = QkNorms::PerHead;
        s.dense_lead = 1;
        s.exp_probs_b = true;
        s.gating = Some(1); // SOFTMAX (the key must exist — NONE would abort)
        s.weights_norm = true;
    })
}

/// bailingmoe — MoE everywhere, norm_w true, kq_scale = 1/sqrt(n_rot)
fn spec_bailingmoe() -> SynthSpec {
    base("bailingmoe").with(|s| {
        s.weights_norm = true;
    })
}

/// bailingmoe2 — fused attn_qkv, per-head q/k norms BEFORE rope, one dense
/// lead layer, REQUIRED gating key, exp_probs_b + the n_ff_shexp override
/// (bailingmoe2.cpp:3-18)
fn spec_bailingmoe2() -> SynthSpec {
    base("bailingmoe2").with(|s| {
        s.n_layer = 4;
        s.fused_qkv = true;
        s.qk_norms = QkNorms::PerHead;
        s.dense_lead = 1;
        s.exp_probs_b = true;
        s.gating = Some(1); // REQUIRED key
        s.weights_norm = true;
        s.n_ff_shexp = Some(24);
    })
}

/// glm4moe — gating key absent (defaults SIGMOID), REQUIRED router bias,
/// attn_post_norm as the FFN norm, 2 shared experts, q/k norms on the upper
/// half of the layers only (the 355B variant tensors, glm4-moe.cpp:66-70)
fn spec_glm4_moe() -> SynthSpec {
    base("glm4moe").with(|s| {
        s.n_layer = 4;
        s.write_output = false; // tied head
        s.qk_norms = QkNorms::PerHeadPartial;
        s.dense_lead = 1;
        s.exp_probs_b = true;
        s.n_expert_shared = 2;
        s.weights_norm = true;
    })
}

/// minimax-m2 — full-width q/k norms, partial rope (64 of 128), SIGMOID
/// gating with the REQUIRED bias, experts at the dense n_ff
fn spec_minimax_m2() -> SynthSpec {
    base("minimax-m2").with(|s| {
        s.n_embd = 256;
        s.n_head = 2;
        s.n_head_kv = 2; // MHA
        s.key_length = 128;
        s.rope_dim = 64; // minimax-m2.cpp:51 — head 128, n_rot 64
        s.qk_norms = QkNorms::FullWidth;
        s.exp_probs_b = true;
        s.gating = Some(2); // SIGMOID (the Minimax-M2 routing)
    })
}

/// cohere2moe — iswa pattern 4 dense-first with n_swa 64, fused gate_up
/// experts, sigmoid gating, (moe+shexp)*0.5, logit_scale; 6 layers so layer 4
/// (il % 4 == 0, past the dense lead) is the NO-rope full-attention layer
fn spec_cohere2moe() -> SynthSpec {
    base("cohere2moe").with(|s| {
        s.n_layer = 6;
        s.write_output = false;
        s.dense_lead = 1;
        s.fused_gate_up = true;
        s.n_swa = Some(64);
        s.logit_scale = Some(1.2);
    })
}

/// cohere2moe `-sep-ln` — separate gate/up experts + the LLM_NORM norm type
/// (`attention.layer_norm_epsilon`, cohere2moe.cpp:4-11)
fn spec_cohere2moe_sep_ln() -> SynthSpec {
    spec_cohere2moe().with(|s| {
        s.suffix = "-sep-ln";
        s.fused_gate_up = false;
        s.norm_ln = true;
    })
}

/// exaone-moe — iswa pattern 4 (NOT dense-first) with n_swa 64: rope only on
/// the SWA layers (layer 3 skips rope entirely), REQUIRED gating
fn spec_exaone_moe() -> SynthSpec {
    base("exaone-moe").with(|s| {
        s.n_layer = 6;
        s.write_output = true; // exaone-moe.cpp:41 — REQUIRED (the tie branch is dead)
        s.qk_norms = QkNorms::PerHead;
        s.dense_lead = 1;
        s.exp_probs_b = true;
        s.gating = Some(1); // REQUIRED key
        s.weights_norm = true;
        s.n_swa = Some(64);
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_hunyuan_moe(),
        spec_dots1(),
        spec_bailingmoe(),
        spec_bailingmoe2(),
        spec_glm4_moe(),
        spec_minimax_m2(),
        spec_cohere2moe(),
        spec_cohere2moe_sep_ln(),
        spec_exaone_moe(),
    ]
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
}

type TensorSpec = (String, Vec<i64>);

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let head = spec.n_embd_head();
    let n_kv = spec.n_embd_kv();
    let n_exp = N_EXPERT;
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    // every arch of the batch shares the model-level trio
    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    push!("output_norm.weight", vec![n_embd], Role::Norm);
    if spec.write_output {
        push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);
    }

    for i in 0..spec.n_layer {
        let q_partial = match spec.qk_norms {
            QkNorms::PerHeadPartial => i >= spec.n_layer / 2,
            _ => true,
        };
        push!(
            format!("blk.{i}.attn_norm.weight"),
            vec![n_embd],
            Role::Norm
        );

        // the attention projections (create_tensor_qkv: fused or separate)
        if spec.fused_qkv {
            let n_q = head * spec.n_head;
            push!(
                format!("blk.{i}.attn_qkv.weight"),
                vec![n_embd, n_q + 2 * n_kv],
                Role::Proj
            );
        } else {
            // bailingmoe's widths come from n_rot (bailingmoe.cpp:35) — equal
            // to key_length in every file that lines up with the graph
            push!(
                format!("blk.{i}.attn_q.weight"),
                vec![n_embd, head * spec.n_head],
                Role::Proj
            );
            push!(
                format!("blk.{i}.attn_k.weight"),
                vec![n_embd, n_kv],
                Role::Proj
            );
            push!(
                format!("blk.{i}.attn_v.weight"),
                vec![n_embd, n_kv],
                Role::Proj
            );
        }
        push!(
            format!("blk.{i}.attn_output.weight"),
            vec![head * spec.n_head, n_embd],
            Role::Proj
        );

        match spec.qk_norms {
            QkNorms::None => {}
            QkNorms::PerHead | QkNorms::PerHeadPartial => {
                if q_partial {
                    push!(
                        format!("blk.{i}.attn_q_norm.weight"),
                        vec![head],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_k_norm.weight"),
                        vec![head],
                        Role::Norm
                    );
                }
            }
            QkNorms::FullWidth => {
                // minimax-m2.cpp:30-31 — [n_embd_head_k * n_head] / [n_embd_k_gqa]
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![head * spec.n_head],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![n_kv],
                    Role::Norm
                );
            }
        }

        // glm4moe's attn_post_norm doubles as the FFN norm (tensor name
        // `post_attention_norm`, llama-arch.cpp:458); cohere2moe has NO
        // separate FFN norm at all (the graph feeds attn_norm(inpL) to the
        // FFN); the rest carry a plain ffn_norm
        if spec.arch == "glm4moe" {
            push!(
                format!("blk.{i}.post_attention_norm.weight"),
                vec![n_embd],
                Role::Norm
            );
        } else if spec.arch != "cohere2moe" {
            push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
        }

        let is_moe = (i as u32) >= spec.dense_lead;
        if !is_moe {
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
            continue;
        }

        // the MoE branch
        push!(
            format!("blk.{i}.ffn_gate_inp.weight"),
            vec![n_embd, n_exp],
            Role::Router
        );
        if spec.exp_probs_b {
            push!(format!("blk.{i}.exp_probs_b.bias"), vec![n_exp], Role::Bias);
        }
        let n_ff_exp = if spec.arch == "hunyuan-moe" {
            n_ff // hunyuan-moe.cpp:42-44 — experts at the dense n_ff
        } else if spec.arch == "minimax-m2" {
            n_ff // minimax-m2.cpp:36-38 — experts at the dense n_ff
        } else {
            spec.n_ff_exp
        };
        if spec.fused_gate_up {
            // cohere2moe's create_tensor_gate_up_exps
            push!(
                format!("blk.{i}.ffn_gate_up_exps.weight"),
                vec![n_embd, 2 * n_ff_exp, n_exp],
                Role::Router
            );
        } else {
            push!(
                format!("blk.{i}.ffn_gate_exps.weight"),
                vec![n_embd, n_ff_exp, n_exp],
                Role::Router
            );
            push!(
                format!("blk.{i}.ffn_up_exps.weight"),
                vec![n_embd, n_ff_exp, n_exp],
                Role::Router
            );
        }
        push!(
            format!("blk.{i}.ffn_down_exps.weight"),
            vec![n_ff_exp, n_embd, n_exp],
            Role::Router
        );

        // the shared expert (absent on minimax-m2; conditional on
        // n_expert_shared > 0 for glm4moe — always >= 1 in the specs here)
        if spec.arch != "minimax-m2" {
            let n_sh = spec.n_ff_shexp_of();
            push!(
                format!("blk.{i}.ffn_gate_shexp.weight"),
                vec![n_embd, n_sh],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_up_shexp.weight"),
                vec![n_embd, n_sh],
                Role::Proj
            );
            push!(
                format!("blk.{i}.ffn_down_shexp.weight"),
                vec![n_sh, n_embd],
                Role::Proj
            );
        }
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch4/6_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch8");

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
        Value::U32(spec.key_length as u32)
    );
    // cohere2moe's LLM_NORM variant carries the plain eps instead of the RMS one
    if spec.norm_ln {
        kv!(
            format!("{a}.attention.layer_norm_epsilon"),
            Value::F32(1e-5)
        );
    } else {
        kv!(
            format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-5)
        );
    }
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.rope_dim as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // the MoE keys every arch of the batch reads (all REQUIRED unless noted)
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(
        format!("{a}.expert_used_count"),
        Value::U32(N_EXPERT_USED as u32)
    );
    kv!(
        format!("{a}.expert_feed_forward_length"),
        Value::U32(spec.n_ff_exp as u32)
    );
    if spec.n_expert_shared > 0 {
        kv!(
            format!("{a}.expert_shared_count"),
            Value::U32(spec.n_expert_shared)
        );
    }
    if let Some(v) = spec.n_ff_shexp {
        kv!(
            format!("{a}.expert_shared_feed_forward_length"),
            Value::U32(v as u32)
        );
    }
    if spec.dense_lead > 0 {
        kv!(
            format!("{a}.leading_dense_block_count"),
            Value::U32(spec.dense_lead)
        );
    }
    if let Some(g) = spec.gating {
        kv!(format!("{a}.expert_gating_func"), Value::U32(g));
    }
    if spec.weights_norm {
        kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
    }
    if let Some(v) = spec.weights_scale {
        kv!(format!("{a}.expert_weights_scale"), Value::F32(v));
    }
    if let Some(v) = spec.logit_scale {
        kv!(format!("{a}.logit_scale"), Value::F32(v));
    }
    if let Some(v) = spec.n_swa {
        kv!(format!("{a}.attention.sliding_window"), Value::U32(v));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
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

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn load_synth(spec: &SynthSpec) -> LlamaModel {
    build_file(spec);
    open_model(&spec.path())
}

fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let mut want: Vec<String> = tensors_for(spec).into_iter().map(|((n, _), _)| n).collect();
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
// weights + params assembly — the same derivations llama-cli's forward_weights
// performs (kept in sync with crates/tools/llama-cli/src/main.rs)
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

fn hunyuan_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::HunyuanMoeModelWeights {
    graph_arch::HunyuanMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::HunyuanMoeLayerWeights {
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
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_gate_shexp: l.ffn_gate_shexp.unwrap(),
                ffn_down_shexp: l.ffn_down_shexp.unwrap(),
                ffn_up_shexp: l.ffn_up_shexp.unwrap(),
            })
            .collect(),
    }
}

fn dots1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Dots1ModelWeights {
    graph_arch::Dots1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Dots1LayerWeights {
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
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
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

fn bailingmoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BailingmoeModelWeights {
    graph_arch::BailingmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::BailingmoeLayerWeights {
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
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_gate_shexp: l.ffn_gate_shexp.unwrap(),
                ffn_down_shexp: l.ffn_down_shexp.unwrap(),
                ffn_up_shexp: l.ffn_up_shexp.unwrap(),
            })
            .collect(),
    }
}

fn bailingmoe2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Bailingmoe2ModelWeights {
    graph_arch::Bailingmoe2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Bailingmoe2LayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
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

fn glm4_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4MoeModelWeights {
    graph_arch::Glm4MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Glm4MoeLayerWeights {
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
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                attn_post_norm: l.attn_post_norm.unwrap(),
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

fn minimax_m2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MinimaxM2ModelWeights {
    graph_arch::MinimaxM2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::MinimaxM2LayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_exp_probs_b: l.ffn_exp_probs_b.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn cohere2moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2MoeModelWeights {
    graph_arch::Cohere2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Cohere2MoeLayerWeights {
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
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
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

fn exaone_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ExaoneMoeModelWeights {
    graph_arch::ExaoneMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::ExaoneMoeLayerWeights {
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
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
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

/// (ForwardWeights, AttnParams) of one loaded model — the same assembly the
/// CLI performs (batch-6 protocol: the context.rs arms landed with the graph)
fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa);
    let n_trunk = hp.n_layer() as usize;
    let gating = hp.expert_gating_func as i32;
    match m.arch {
        llama::arch::LlmArch::HUNYUAN_MOE => (
            ForwardWeights::HunyuanMoe(
                hunyuan_moe_weights(m, n_trunk),
                graph_arch::HunyuanMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::DOTS1 => (
            ForwardWeights::Dots1(
                dots1_weights(m, n_trunk),
                graph_arch::Dots1Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::BAILINGMOE => (
            ForwardWeights::Bailingmoe(
                bailingmoe_weights(m, n_trunk),
                graph_arch::BailingmoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::BAILINGMOE2 => (
            ForwardWeights::Bailingmoe2(
                bailingmoe2_weights(m, n_trunk),
                graph_arch::Bailingmoe2Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::GLM4_MOE => (
            ForwardWeights::Glm4Moe(
                glm4_moe_weights(m, n_trunk),
                graph_arch::Glm4MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::MINIMAX_M2 => (
            ForwardWeights::MinimaxM2(
                minimax_m2_weights(m, n_trunk),
                graph_arch::MinimaxM2Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::COHERE2MOE => (
            ForwardWeights::Cohere2Moe(
                cohere2moe_weights(m, n_trunk),
                graph_arch::Cohere2MoeParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    norm_ln_eps: hp.f_norm_eps,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                    logit_scale: hp.f_logit_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::EXAONE_MOE => (
            ForwardWeights::ExaoneMoe(
                exaone_moe_weights(m, n_trunk),
                graph_arch::ExaoneMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: gating,
                },
            ),
            attn,
        ),
        other => panic!("arch {other:?} not in batch 8"),
    }
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

fn smoke_both(spec: &SynthSpec) -> (i32, i32) {
    let a = smoke_forward(spec, false);
    let b = smoke_forward(spec, true);
    (argmax(&a), argmax(&b))
}

// ---------------------------------------------------------------------------
// default-run tests (no reference needed)
// ---------------------------------------------------------------------------

#[test]
fn synth_hunyuan_moe_loader_and_forward() {
    let spec = spec_hunyuan_moe();
    let (n, bytes) = build_file(&spec);
    // 2 model (tied head) + 4 layers x (attn_norm + 3 qkv + wo + 2 qk norms +
    // ffn_norm + router + 3 experts + 3 shexp) = 2 + 4*15
    assert_eq!(n, 2 + 4 * 15, "hunyuan-moe tensor count");
    println!(
        "hunyuan-moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_ff_exp(0), 32);
    assert_eq!(hp.n_ff_shexp, 24);
    // the expert tensors sit at the dense n_ff (hunyuan-moe.cpp:42-44)
    assert_eq!(
        *m.ctx.ne(m.layers[0].ffn_gate_exps.unwrap()),
        [128, 96, 4, 1]
    );
    assert_eq!(
        *m.ctx.ne(m.layers[0].ffn_up_shexp.unwrap()),
        [128, 24, 1, 1]
    );
    // hunyuan-moe.cpp:3-12 reads no expert_weights_scale — the graph passes
    // the fresh-struct 0.0 (build_moe_ffn skips the multiply)
    assert_eq!(hp.expert_weights_scale, 0.0);
    assert_eq!(m.output, m.tok_embd, "tied head");

    let (o, i) = smoke_both(&spec);
    println!("hunyuan-moe synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_dots1_loader_and_forward() {
    let spec = spec_dots1();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert_eq!(hp.n_expert_shared, 1);
    assert!(hp.expert_weights_norm);
    assert_eq!(hp.expert_gating_func, 1);
    assert_eq!(hp.n_head_kv(0), hp.n_head(0), "dots1 is MHA");
    assert!(m.layers[0].ffn_gate.is_some(), "layer 0 dense lead");
    assert!(m.layers[0].ffn_gate_inp.is_none());
    assert!(m.layers[1].ffn_gate_inp.is_some());
    assert!(m.layers[1].ffn_exp_probs_b.is_some());
    // the fat shared expert: n_ff_exp * n_expert_shared = 32
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_up_shexp.unwrap()),
        [128, 32, 1, 1]
    );
    // dots1.cpp:27 — output is a plain required tensor (no tied fallback)
    assert!(m.output != m.tok_embd);

    let (o, i) = smoke_both(&spec);
    println!("dots1 synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_bailingmoe_loader_and_forward() {
    let spec = spec_bailingmoe();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert!(hp.expert_weights_norm);
    assert_eq!(
        hp.expert_gating_func, 0,
        "bailingmoe hard-codes softmax at the graph"
    );
    // the qkv widths come from n_rot (bailingmoe.cpp:35)
    assert_eq!(*m.ctx.ne(m.layers[0].wq.unwrap()), [128, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].wk.unwrap()), [128, 64, 1, 1]);
    assert_eq!(hp.n_rot(0), 32);
    assert!(m.layers[0].ffn_gate.is_none(), "every layer is MoE");
    assert!(m.layers[0].ffn_exp_probs_b.is_none(), "no router bias");

    let (o, i) = smoke_both(&spec);
    println!("bailingmoe synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_bailingmoe2_loader_and_forward() {
    let spec = spec_bailingmoe2();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert_eq!(hp.n_ff_shexp, 24);
    assert_eq!(hp.expert_gating_func, 1, "REQUIRED key read");
    assert!(hp.expert_weights_norm);
    // the fused attn_qkv of width n_embd + 2*n_embd_gqa (bailingmoe2.cpp:46)
    assert_eq!(*m.ctx.ne(m.layers[0].wqkv.unwrap()), [128, 256, 1, 1]);
    assert!(m.layers[0].wq.is_none());
    assert!(m.layers[0].ffn_gate.is_some(), "layer 0 dense lead");
    assert!(m.layers[1].ffn_exp_probs_b.is_some());
    // n_ff_shexp overrides the n_ff_exp*n_expert_shared default
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_up_shexp.unwrap()),
        [128, 24, 1, 1]
    );

    let (o, i) = smoke_both(&spec);
    println!("bailingmoe2 synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_glm4_moe_loader_and_forward() {
    let spec = spec_glm4_moe();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    // glm4-moe.cpp:16-18 — the absent gating key defaults to SIGMOID
    assert_eq!(hp.expert_gating_func, 2);
    assert_eq!(hp.n_expert_shared, 2);
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert!(hp.expert_weights_norm);
    assert_eq!(m.output, m.tok_embd, "tied head");
    assert!(m.layers[0].ffn_gate.is_some(), "layer 0 dense lead");
    assert!(m.layers[0].ffn_gate_inp.is_none());
    assert!(
        m.layers[0].attn_q_norm.is_none(),
        "355B-variant norms on half the layers"
    );
    assert!(m.layers[2].attn_q_norm.is_some());
    // required router bias (glm4-moe.cpp:82)
    assert!(m.layers[1].ffn_exp_probs_b.is_some());
    // 2 shared experts → shexp width n_ff_exp * 2
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_up_shexp.unwrap()),
        [128, 64, 1, 1]
    );
    assert_eq!(
        *m.ctx.ne(m.layers[1].attn_post_norm.unwrap()),
        [128, 1, 1, 1]
    );

    let (o, i) = smoke_both(&spec);
    println!("glm4moe synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_minimax_m2_loader_and_forward() {
    let spec = spec_minimax_m2();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.expert_gating_func, 2, "SIGMOID (the M2 routing)");
    assert_eq!(hp.n_embd_head_k(0), 128);
    assert_eq!(hp.n_rot(0), 64, "partial rope — head 128, n_rot 64");
    // the full-width q/k norms (minimax-m2.cpp:30-31)
    assert_eq!(*m.ctx.ne(m.layers[0].attn_q_norm.unwrap()), [256, 1, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[0].attn_k_norm.unwrap()), [256, 1, 1, 1]);
    // the experts sit at the dense n_ff (minimax-m2.cpp:36-38) and the router
    // bias is required
    assert_eq!(
        *m.ctx.ne(m.layers[0].ffn_gate_exps.unwrap()),
        [256, 96, 4, 1]
    );
    assert!(m.layers[0].ffn_exp_probs_b.is_some());
    assert!(m.layers[0].ffn_up_shexp.is_none(), "no shared expert");

    let (o, i) = smoke_both(&spec);
    println!("minimax-m2 synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_cohere2moe_loader_and_forward() {
    let spec = spec_cohere2moe();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.f_logit_scale, 1.2);
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert_eq!(hp.n_swa, 64);
    assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
    assert_eq!(hp.expert_gating_func, 2, "absent key → SIGMOID");
    assert_eq!(m.output, m.tok_embd, "tied head");
    // set_swa_pattern(4, dense_first): il % 4 != 0 → layers 1,2,3,5 SWA;
    // layer 4 is the full-attention NO-rope layer (il >= dense_lead)
    let swa: Vec<bool> = (0..6).map(|il| hp.is_swa(il)).collect();
    assert_eq!(swa, vec![false, true, true, true, false, true]);
    // the fused gate_up tensor wins (create_tensor_gate_up_exps)
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_gate_up_exps.unwrap()),
        [128, 64, 4, 1]
    );
    assert!(m.layers[1].ffn_gate_exps.is_none());
    assert!(m.layers[1].ffn_up_exps.is_none());
    assert!(m.layers[0].ffn_gate.is_some(), "layer 0 dense lead");
    // shexp width falls back to n_ff_exp * n_expert_shared
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_up_shexp.unwrap()),
        [128, 32, 1, 1]
    );

    let (o, i) = smoke_both(&spec);
    println!("cohere2moe synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_cohere2moe_sep_ln_variant() {
    let spec = spec_cohere2moe_sep_ln();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    // cohere2moe.cpp:9-11 — no RMS key → f_norm_rms_eps = 0 → LLM_NORM
    assert_eq!(m.hparams.f_norm_rms_eps, 0.0);
    assert_eq!(m.hparams.f_norm_eps, 1e-5);
    // the separate gate/up fallback of create_tensor_gate_up_exps
    assert!(m.layers[1].ffn_gate_up_exps.is_none());
    assert!(m.layers[1].ffn_gate_exps.is_some());
    assert!(m.layers[1].ffn_up_exps.is_some());

    let (o, i) = smoke_both(&spec);
    println!("cohere2moe-sep-ln synth: greedy first token {o} / {i} (fa off/on)");
}

#[test]
fn synth_exaone_moe_loader_and_forward() {
    let spec = spec_exaone_moe();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.n_swa, 64);
    assert_eq!(hp.expert_gating_func, 1, "REQUIRED key");
    assert!(hp.expert_weights_norm);
    assert_eq!(hp.n_layer_dense_lead, 1);
    assert!(
        m.output != m.tok_embd,
        "exaone-moe output.weight is required"
    );
    // set_swa_pattern(4, dense_first = false): il % 4 < 3 → layers 0,1,2,4,5
    // SWA (roped); layer 3 is full-attention and carries NO rope
    let swa: Vec<bool> = (0..6).map(|il| hp.is_swa(il)).collect();
    assert_eq!(swa, vec![true, true, true, false, true, true]);
    assert!(m.layers[0].ffn_gate.is_some(), "layer 0 dense lead");
    // shexp falls back to n_ff_exp (exaone-moe.cpp:32)
    assert_eq!(
        *m.ctx.ne(m.layers[1].ffn_up_shexp.unwrap()),
        [128, 32, 1, 1]
    );
    assert!(
        m.layers[1].rope_freqs.is_none(),
        "no rope_freqs in the file"
    );

    let (o, i) = smoke_both(&spec);
    println!("exaone-moe synth: greedy first token {o} / {i} (fa off/on)");
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_glm4_moe().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("glm4moe"));
    assert_eq!(g.get_u32("glm4moe.expert_shared_count"), Some(2));
    assert_eq!(g.get_u32("glm4moe.leading_dense_block_count"), Some(1));
    assert!(g.find_key("glm4moe.expert_gating_func").is_none());

    let spec = spec_cohere2moe().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("cohere2moe"));
    assert_eq!(g.get_u32("cohere2moe.attention.sliding_window"), Some(64));
    assert_eq!(g.get_f32("cohere2moe.logit_scale"), Some(1.2));
    assert_eq!(g.get_u32("cohere2moe.expert_feed_forward_length"), Some(32));

    let spec = spec_minimax_m2().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_u32("minimax-m2.rope.dimension_count"), Some(64));
    assert_eq!(g.get_u32("minimax-m2.expert_gating_func"), Some(2));
}

/// The windowed-SWA smoke of the batch: n_swa = 64, positions beyond the
/// window must stay finite and the SWA cache must purge (used cells stay
/// below the walked distance) while the base cache keeps every cell — the
/// in-port twin of the `-long` parity cells.
#[test]
fn iswa_window_binding() {
    // private suffix: the plain cohere2moe file belongs to the loader test
    // (concurrent build_file on one path = mmap/SIGBUS race, batch-5 lesson)
    let spec = spec_cohere2moe().with(|s| s.suffix = "-win");
    build_file(&spec);
    let mut m = open_model(&spec.path());
    let mut d = driver_for(&mut m, false);
    assert!(d.kv.has_swa(), "cohere2moe must create the iswa pair");

    // walk 100 tokens past the 64-token window in 25-token steps
    let mut pos = 0i32;
    let mut logits = None;
    while pos < 100 {
        let end = (pos + 25).min(100);
        let ids: Vec<i32> = ((pos as i32 + 7)..(end as i32 + 7)).collect();
        let poss: Vec<i32> = (pos..end).collect();
        logits = Some(d.decode(&ids, &poss).expect("decode").to_vec());
        pos = end;
    }
    let l = logits.unwrap();
    assert!(l.iter().all(|v| v.is_finite()), "logits past the window");
    assert_eq!(d.kv.used_cells(), 100, "base cache keeps every cell");
    // the swa pair exists with the 64-token window; eviction past the window
    // is lazy (only under cell pressure, llama-kv-cache.cpp find_slot), the
    // *mask* enforces the window — see the `-long` parity cells for the
    // numeric proof against the reference
    let swa = d.kv.swa.as_ref().expect("swa cache");
    assert_eq!(swa.n_swa, 64);
    assert!(d.kv.n_kv_swa() >= 64, "swa rows readable past the window");
}

// ---------------------------------------------------------------------------
// generator (#[ignore]) — the parity runs drive the release llama-cli itself
// (batch-1 protocol; this batch's CLI arms landed with the graph)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~110 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch8_write_synth() {
    for spec in all_specs() {
        let (n, bytes) = build_file(&spec);
        println!(
            "{:>16}: {:4} tensors, {:>10} bytes -> {}",
            format!("{}{}", spec.arch, spec.suffix),
            n,
            bytes,
            spec.path()
        );
    }
    let gguf = Gguf::open(spec_hunyuan_moe().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH8=1 ./parity/arch_batch_parity.sh hunyuan-moe dots1 bailingmoe \
         bailingmoe2 glm4moe minimax-m2 cohere2moe exaone-moe"
    );
    println!(
        "long:  ARCH_BATCH8=1 ./parity/arch_batch_parity.sh hunyuan-moe-long dots1-long \
         bailingmoe-long bailingmoe2-long glm4moe-long minimax-m2-long cohere2moe-long \
         exaone-moe-long"
    );
}
