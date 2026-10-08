//! arch_batch9_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-10: **the linear-attention family** — plamo3 / qwen3next /
//! kimi-linear / bailingmoe3 (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-8 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//!   * plamo3 — the every-8th-full SWA pattern (n_swa = 64, layers 0-6 SWA,
//!     layer 7 full), the SWA rope frequency pair (freq_base_swa 30000), the
//!     fused wqkv with **head_dim_q 32 != head_dim_v 48**, q/k norms before
//!     rope, the post-norms, and the fused swiglu FFN (up emits 2*n_ff);
//!   * qwen3next — the GDN geometry with H_k (ssm.group_count 2) != H_v
//!     (ssm.time_step_rank 4) so the head-repeat path runs, the packed
//!     `ssm_ba` tensor, the fused `ffn_gate_up_exps` + the sigmoid-gated
//!     shared expert (`ffn_gate_inp_shexp`), routed_scaling_factor 2.0, the
//!     QG-wide attention q projection with its sigmoid gate, explicit
//!     `attention.recurrent_layers` [1,1,1,0,1,1]; a `-legacy` variant
//!     carries the single fused `ssm_in` QKVZ tensor + separate gate/up
//!     experts instead (verified in-port);
//!   * kimi-linear — KDA layers 0/2 (head_count_kv 0) + no-rope MLA layers
//!     1/3 (head_count_kv 1) with the split wk_b/wv_b absorbed cache (the
//!     deepseek2 trick: attention.key_length = kv_lora+rope = 48, so the
//!     generic cache machinery sees the compressed row), the fused attn_qkv
//!     KDA projection, f_a/f_b two-stage decay, g_a/g_b output gate,
//!     probs_b + softmax gating + routed_scaling_factor 2.446, one dense lead
//!     layer; a `-legacy` variant carries the unsplit wkv_b (MHA cache,
//!     head_count_kv = n_head) — verified in-port ONLY: the pinned reference
//!     segfaults on such a file (kimi-linear.cpp:482 dereferences the null
//!     inp_attn_kv because is_mla() is forced true by the required
//!     attention.key_length_mla read, :245-248; PARITY.md batch 9 §6.5);
//!   * bailingmoe3 — KDA layers 0/2 with the safe gate
//!     (kda.gate_lower_bound -0.1), separate wq/wk/wv KDA projections (a
//!     fused attn_qkv file would hand the reference a nullptr in
//!     bailingmoe3_causal_conv1d), MLA with the q_lora_rank 16 compression +
//!     the sigmoid attention output gate (attn_gate), q/k rope **NORM** on
//!     the pe tails, weights_norm + probs_b + sigmoidless softmax gating, and
//!     the swiglu_clamp_exp/shexp 7.0 limits on the MoE + shared experts and
//!     the dense lead.
//!
//! The batch's ForwardWeights/CLI arms landed with the graph (context.rs),
//! so the default-run tests drive `DecodeContext::new_with{,_swa}` itself
//! (batch-6 protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH9=1 ./parity/arch_batch_parity.sh …`).

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
const OUT_DIR: &str = "/tmp/arch-batch9";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

/// which linear-attention family the file belongs to
#[derive(Clone, Copy, PartialEq)]
enum Family {
    /// plain SWA attention (plamo3)
    Plamo3,
    /// GDN + gated attention (qwen3next)
    Qwen3Next,
    /// KDA + no-rope MLA (kimi-linear)
    KimiLinear,
    /// KDA + gated MLA (bailingmoe3)
    BailingMoe3,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    /// attention.head_count_kv per layer (KDA archs: 0 on the KDA layers)
    head_kv: Vec<i64>,
    /// attention.key_length (the MLA archs' cache row width; qwen3next: head dim)
    key_length: i64,
    /// attention.value_length
    value_length: i64,
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    // ---- the linear-attention geometry ----
    ssm_d_conv: i64,
    ssm_d_inner: i64,
    ssm_d_state: i64,
    ssm_dt_rank: i64,
    ssm_n_group: i64,
    /// explicit attention.recurrent_layers (qwen3next) — None → interval 4
    recr_layers: Option<Vec<u32>>,
    // ---- the KDA/MLA geometry (kimi-linear / bailingmoe3) ----
    kda_head_dim: i64,
    kv_lora_rank: i64,
    q_lora_rank: i64,
    key_length_mla: i64,
    value_length_mla: i64,
    kda_gate_lower_bound: f32,
    /// MLA cache split (wk_b/wv_b) vs the legacy unsplit wkv_b
    mla_split: bool,
    /// KDA q/k/v: the fused attn_qkv (kimi) or separate (bailingmoe3)
    kda_fused_qkv: bool,
    // ---- MoE knobs ----
    dense_lead: u32,
    n_ff_exp: i64,
    n_ff_shexp: Option<i64>,
    n_expert_shared: u32,
    exp_probs_b: bool,
    gating: Option<u32>,
    weights_norm: bool,
    weights_scale: Option<f32>,
    /// fused ffn_gate_up_exps (qwen3next primary); false → separate gate/up
    fused_gate_up: bool,
    /// the single fused ssm_in QKVZ tensor (qwen3next legacy files)
    legacy_ssm_in: bool,
    /// plamo3's SWA pair
    n_swa: Option<u32>,
    freq_base_swa: Option<f32>,
    /// bailingmoe3's swiglu_clamp limits (scalar → broadcast)
    swiglu_clamp: Option<f32>,
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
            Family::Plamo3 => false,
            Family::Qwen3Next => match &self.recr_layers {
                Some(v) => v[il] != 0,
                None => (il as u32 + 1) % 4 != 0,
            },
            Family::KimiLinear | Family::BailingMoe3 => self.head_kv[il] == 0,
        }
    }
    /// the GDN geometry (qwen3next)
    fn key_dim(&self) -> i64 {
        self.ssm_d_state * self.ssm_n_group
    }
    fn value_dim(&self) -> i64 {
        self.ssm_d_state * self.ssm_dt_rank
    }
    fn conv_dim(&self) -> i64 {
        self.key_dim() * 2 + self.value_dim()
    }
    /// the KDA dims (kimi-linear / bailingmoe3)
    fn kda_d_inner(&self) -> i64 {
        self.kda_head_dim * self.n_head
    }
    fn qk_rope(&self) -> i64 {
        self.rope_dim
    }
    fn qk_nope(&self) -> i64 {
        self.key_length_mla - self.qk_rope()
    }
    fn n_ff_shexp_of(&self) -> i64 {
        if let Some(v) = self.n_ff_shexp {
            v
        } else if self.family == Family::KimiLinear {
            self.n_ff_exp * self.n_expert_shared as i64
        } else {
            self.n_ff_exp
        }
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
        ssm_d_conv: 4,
        ssm_d_inner: 32,
        ssm_d_state: 8,
        ssm_dt_rank: 4,
        ssm_n_group: 2,
        recr_layers: None,
        kda_head_dim: 16,
        kv_lora_rank: 32,
        q_lora_rank: 0,
        key_length_mla: 40,
        value_length_mla: 20,
        kda_gate_lower_bound: -0.1,
        mla_split: true,
        kda_fused_qkv: true,
        dense_lead: 0,
        n_ff_exp: 32,
        n_ff_shexp: None,
        n_expert_shared: 1,
        exp_probs_b: false,
        gating: None,
        weights_norm: false,
        weights_scale: None,
        fused_gate_up: true,
        legacy_ssm_in: false,
        n_swa: None,
        freq_base_swa: None,
        swiglu_clamp: None,
    }
}

/// plamo3 — 8 layers of the every-8th-full SWA pattern, head_dim_q 32 !=
/// head_dim_v 48, the SWA rope frequency pair, tied output
fn spec_plamo3() -> SynthSpec {
    base("plamo3", Family::Plamo3).with(|s| {
        s.n_layer = 8;
        s.n_head = 4;
        s.head_kv = vec![2; 8];
        s.key_length = 32;
        s.value_length = 48;
        s.rope_dim = 32;
        s.n_ff = 96;
        s.write_output = false;
        s.n_swa = Some(64);
        s.freq_base_swa = Some(30000.0);
    })
}

/// qwen3next — explicit recurrent_layers [1,1,1,0,1,1] (layer 3 is the
/// gated-attention layer), H_k 2 != H_v 4 (the head-repeat path), fused
/// gate_up experts + the gated shared expert, scaling 2.0
fn spec_qwen3next() -> SynthSpec {
    base("qwen3next", Family::Qwen3Next).with(|s| {
        s.n_layer = 6;
        s.head_kv = vec![2; 6];
        s.recr_layers = Some(vec![1, 1, 1, 0, 1, 1]);
        // GDN: d_state 8, group 2, dt_rank 4, d_inner 32 → head_v_dim 8,
        // repeat factor 2; conv_dim = 32 + 2*2*8 = 64
        s.ssm_d_conv = 4;
        s.ssm_d_inner = 32;
        s.ssm_d_state = 8;
        s.ssm_dt_rank = 4;
        s.ssm_n_group = 2;
        s.write_output = false;
        s.n_ff_exp = 32;
        s.n_ff_shexp = Some(24);
        // NB: qwen3next.cpp never reads expert_weights_scale (it stays 0 —
        // build_moe_ffn's optional scale is never applied for this arch)
    })
}

/// qwen3next `-legacy` — the single fused ssm_in QKVZ tensor + separate
/// gate/up experts (qwen3next.cpp:336-392, verified in-port)
fn spec_qwen3next_legacy() -> SynthSpec {
    spec_qwen3next().with(|s| {
        s.suffix = "-legacy";
        s.legacy_ssm_in = true;
        s.fused_gate_up = false;
    })
}

/// kimi-linear — KDA layers 0/2, no-rope MLA layers 1/3 with the split
/// wk_b/wv_b cache (attention.key_length = kv_lora+rope = 48), fused
/// attn_qkv on the KDA layers, probs_b + softmax + scale 2.446, one dense
/// lead layer
fn spec_kimi_linear() -> SynthSpec {
    base("kimi-linear", Family::KimiLinear).with(|s| {
        s.head_kv = vec![0, 1, 0, 1];
        s.key_length = 48; // kv_lora 32 + qk_rope 16 — the compressed row
        s.value_length = 32; // kv_lora
        s.rope_dim = 16; // qk_rope_head_dim (rope_type NONE — never applied)
        s.n_ff = 64;
        s.dense_lead = 1;
        s.exp_probs_b = true;
        s.gating = Some(1); // SOFTMAX (REQUIRED key)
        s.weights_scale = Some(2.446);
        s.kda_fused_qkv = true;
    })
}

/// kimi-linear `-legacy` — the unsplit wkv_b with the MHA cache
/// (head_count_kv = n_head on the MLA layers, attention.key_length =
/// qk_head_dim), verified in-port
fn spec_kimi_linear_legacy() -> SynthSpec {
    spec_kimi_linear().with(|s| {
        s.suffix = "-legacy";
        s.mla_split = false;
        s.head_kv = vec![0, 4, 0, 4];
        s.key_length = 40; // qk_head_dim (qk_nope 24 + rope 16)
        s.value_length = 20; // v_mla
    })
}

/// bailingmoe3 — KDA layers 0/2 with the safe gate, separate KDA q/k/v, MLA
/// with the q_lora compression + the attention output gate, NORM rope on the
/// pe tails, weights_norm, swiglu_clamp 7.0
fn spec_bailingmoe3() -> SynthSpec {
    base("bailingmoe3", Family::BailingMoe3).with(|s| {
        s.head_kv = vec![0, 1, 0, 1];
        s.key_length = 48;
        s.value_length = 32;
        s.rope_dim = 16;
        s.n_ff = 64;
        s.dense_lead = 1;
        s.q_lora_rank = 16;
        s.kda_fused_qkv = false;
        s.exp_probs_b = true;
        s.gating = Some(1); // SOFTMAX (REQUIRED key)
        s.weights_norm = true;
        s.n_ff_shexp = Some(24);
        s.swiglu_clamp = Some(7.0);
        s.write_output = false;
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_plamo3(),
        spec_qwen3next(),
        spec_qwen3next_legacy(),
        spec_kimi_linear(),
        spec_kimi_linear_legacy(),
        spec_bailingmoe3(),
    ]
}

/// the parity cells (batch-9 default set): the primary files only
fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_plamo3(),
        spec_qwen3next(),
        spec_kimi_linear(),
        spec_bailingmoe3(),
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
    /// decay/gate vectors — small negatives keep exp(g) bounded
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
        Family::Plamo3 => {
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length;
                let k = spec.head_kv[i] * spec.key_length;
                let kv = spec.head_kv[i] * spec.value_length;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + k + kv],
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
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.value_length, n_embd],
                    Role::Proj
                );
                // the suffix-less plamo-family post norms (plamo3.cpp:50)
                push!(
                    format!("blk.{i}.post_attention_norm"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.post_ffw_norm"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff * 2],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
            }
        }
        Family::Qwen3Next => {
            let key_dim = spec.key_dim();
            let value_dim = spec.value_dim();
            let ba_dim = spec.ssm_dt_rank * 2;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if spec.is_recr(i) {
                    if spec.legacy_ssm_in {
                        push!(
                            format!("blk.{i}.ssm_in.weight"),
                            vec![n_embd, key_dim * 2 + value_dim * 2],
                            Role::Proj
                        );
                    } else {
                        push!(
                            format!("blk.{i}.attn_qkv.weight"),
                            vec![n_embd, key_dim * 2 + value_dim],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.attn_gate.weight"),
                            vec![n_embd, value_dim],
                            Role::Proj
                        );
                    }
                    push!(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![spec.ssm_d_conv, spec.conv_dim()],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_dt.bias"),
                        vec![spec.ssm_dt_rank],
                        Role::Bias
                    );
                    push!(
                        format!("blk.{i}.ssm_a"),
                        vec![spec.ssm_dt_rank],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_ba.weight"),
                        vec![n_embd, ba_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_norm.weight"),
                        vec![spec.ssm_d_state],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![value_dim, n_embd],
                        Role::Proj
                    );
                } else {
                    let q = spec.n_head * spec.key_length * 2;
                    let k = spec.head_kv[i] * spec.key_length;
                    let kv = spec.head_kv[i] * spec.value_length;
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, q],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, k],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, kv],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![spec.key_length * spec.n_head, n_embd],
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
                }
                // the MoE tail + the gated shared expert (every layer)
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                if spec.fused_gate_up {
                    push!(
                        format!("blk.{i}.ffn_gate_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp * 2, n_exp],
                        Role::Router
                    );
                } else {
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                }
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                    vec![n_embd],
                    Role::Router
                );
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
        Family::KimiLinear | Family::BailingMoe3 => {
            let kda_d_inner = spec.kda_d_inner();
            let kv_lora = spec.kv_lora_rank;
            let qk_rope = spec.qk_rope();
            let qk_nope = spec.qk_nope();
            let v_mla = spec.value_length_mla;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if spec.is_recr(i) {
                    // the per-stream conv kernels (4D [d_conv, 1, d_inner, 1])
                    push!(
                        format!("blk.{i}.ssm_conv1d_q.weight"),
                        vec![spec.ssm_d_conv, 1, kda_d_inner, 1],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d_k.weight"),
                        vec![spec.ssm_d_conv, 1, kda_d_inner, 1],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d_v.weight"),
                        vec![spec.ssm_d_conv, 1, kda_d_inner, 1],
                        Role::Proj
                    );
                    if spec.kda_fused_qkv {
                        push!(
                            format!("blk.{i}.attn_qkv.weight"),
                            vec![n_embd, 3 * kda_d_inner],
                            Role::Proj
                        );
                    } else {
                        push!(
                            format!("blk.{i}.attn_q.weight"),
                            vec![n_embd, kda_d_inner],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.attn_k.weight"),
                            vec![n_embd, kda_d_inner],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.attn_v.weight"),
                            vec![n_embd, kda_d_inner],
                            Role::Proj
                        );
                    }
                    match spec.family {
                        Family::KimiLinear => {
                            // the two-stage decay f_a/f_b + the g_a/g_b gate
                            push!(
                                format!("blk.{i}.ssm_f_a.weight"),
                                vec![n_embd, spec.kda_head_dim],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ssm_f_b.weight"),
                                vec![spec.kda_head_dim, kda_d_inner],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ssm_g_a.weight"),
                                vec![n_embd, spec.kda_head_dim],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ssm_g_b.weight"),
                                vec![spec.kda_head_dim, kda_d_inner],
                                Role::Proj
                            );
                        }
                        Family::BailingMoe3 => {
                            // the single-stage f_a / g_a
                            push!(
                                format!("blk.{i}.ssm_f_a.weight"),
                                vec![n_embd, kda_d_inner],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ssm_g_a.weight"),
                                vec![n_embd, kda_d_inner],
                                Role::Proj
                            );
                        }
                        _ => unreachable!(),
                    }
                    push!(
                        format!("blk.{i}.ssm_beta.weight"),
                        vec![n_embd, spec.n_head],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_a"),
                        vec![1, spec.n_head, 1, 1],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_dt.bias"),
                        vec![kda_d_inner],
                        Role::Bias
                    );
                    push!(
                        format!("blk.{i}.ssm_norm.weight"),
                        vec![spec.kda_head_dim],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![kda_d_inner, n_embd],
                        Role::Proj
                    );
                } else {
                    // the MLA layers
                    if spec.q_lora_rank > 0 {
                        push!(
                            format!("blk.{i}.attn_q_a.weight"),
                            vec![n_embd, spec.q_lora_rank],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.attn_q_a_norm.weight"),
                            vec![spec.q_lora_rank],
                            Role::Norm
                        );
                        push!(
                            format!("blk.{i}.attn_q_b.weight"),
                            vec![spec.q_lora_rank, spec.n_head * spec.key_length_mla],
                            Role::Proj
                        );
                    } else {
                        push!(
                            format!("blk.{i}.attn_q.weight"),
                            vec![n_embd, spec.n_head * spec.key_length_mla],
                            Role::Proj
                        );
                    }
                    push!(
                        format!("blk.{i}.attn_kv_a_mqa.weight"),
                        vec![n_embd, kv_lora + qk_rope],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_kv_a_norm.weight"),
                        vec![kv_lora],
                        Role::Norm
                    );
                    if spec.mla_split {
                        push!(
                            format!("blk.{i}.attn_k_b.weight"),
                            vec![qk_nope, kv_lora, spec.n_head],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.attn_v_b.weight"),
                            vec![kv_lora, v_mla, spec.n_head],
                            Role::Proj
                        );
                    } else {
                        push!(
                            format!("blk.{i}.attn_kv_b.weight"),
                            vec![kv_lora, spec.n_head * (qk_nope + v_mla)],
                            Role::Proj
                        );
                    }
                    if spec.family == Family::BailingMoe3 {
                        push!(
                            format!("blk.{i}.attn_gate.weight"),
                            vec![n_embd, spec.n_head],
                            Role::Proj
                        );
                    }
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![spec.n_head * v_mla, n_embd],
                        Role::Proj
                    );
                }
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if (i as u32) < spec.dense_lead {
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
                } else {
                    push!(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, n_exp],
                        Role::Router
                    );
                    if spec.exp_probs_b {
                        push!(format!("blk.{i}.exp_probs_b.bias"), vec![n_exp], Role::Bias);
                    }
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_exp],
                        Role::Router
                    );
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
        }
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch8_e2e.rs)
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
        // the decay vector: -exp(A_log) — small negatives, then scaled by
        // the C's conversion-side negation (we bake -exp(a) directly)
        Role::Decay => -0.05,
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch9");

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
        if spec.head_kv[0] != 0 {
            kv!(
                format!("{a}.attention.head_count_kv"),
                Value::U32(spec.head_kv[0] as u32)
            );
        }
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
        Family::Plamo3 => {
            if let Some(v) = spec.n_swa {
                kv!(format!("{a}.attention.sliding_window"), Value::U32(v));
            }
            if let Some(v) = spec.freq_base_swa {
                kv!(format!("{a}.rope.freq_base_swa"), Value::F32(v));
            }
        }
        Family::Qwen3Next => {
            kv!(
                format!("{a}.ssm.conv_kernel"),
                Value::U32(spec.ssm_d_conv as u32)
            );
            kv!(
                format!("{a}.ssm.inner_size"),
                Value::U32(spec.ssm_d_inner as u32)
            );
            kv!(
                format!("{a}.ssm.state_size"),
                Value::U32(spec.ssm_d_state as u32)
            );
            kv!(
                format!("{a}.ssm.time_step_rank"),
                Value::U32(spec.ssm_dt_rank as u32)
            );
            kv!(
                format!("{a}.ssm.group_count"),
                Value::U32(spec.ssm_n_group as u32)
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
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            if let Some(v) = spec.n_ff_shexp {
                kv!(
                    format!("{a}.expert_shared_feed_forward_length"),
                    Value::U32(v as u32)
                );
            }
            if let Some(v) = spec.weights_scale {
                kv!(format!("{a}.expert_weights_scale"), Value::F32(v));
            }
        }
        Family::KimiLinear | Family::BailingMoe3 => {
            kv!(
                format!("{a}.attention.key_length_mla"),
                Value::U32(spec.key_length_mla as u32)
            );
            kv!(
                format!("{a}.attention.value_length_mla"),
                Value::U32(spec.value_length_mla as u32)
            );
            kv!(
                format!("{a}.attention.kv_lora_rank"),
                Value::U32(spec.kv_lora_rank as u32)
            );
            if spec.q_lora_rank > 0 {
                kv!(
                    format!("{a}.attention.q_lora_rank"),
                    Value::U32(spec.q_lora_rank as u32)
                );
            }
            kv!(
                format!("{a}.ssm.conv_kernel"),
                Value::U32(spec.ssm_d_conv as u32)
            );
            kv!(
                format!("{a}.kda.head_dim"),
                Value::U32(spec.kda_head_dim as u32)
            );
            if spec.family == Family::BailingMoe3 {
                kv!(
                    format!("{a}.kda.gate_lower_bound"),
                    Value::F32(spec.kda_gate_lower_bound)
                );
                if let Some(v) = spec.swiglu_clamp {
                    kv!(format!("{a}.swiglu_clamp_exp"), Value::F32(v));
                    kv!(format!("{a}.swiglu_clamp_shexp"), Value::F32(v));
                }
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
            kv!(
                format!("{a}.expert_shared_count"),
                Value::U32(spec.n_expert_shared)
            );
            if spec.family == Family::KimiLinear {
                if let Some(v) = spec.n_ff_shexp {
                    // kimi derives the shexp width from n_ff_exp * shared —
                    // do NOT write the override for the primary file
                    let _ = v;
                }
            } else if let Some(v) = spec.n_ff_shexp {
                kv!(
                    format!("{a}.expert_shared_feed_forward_length"),
                    Value::U32(v as u32)
                );
            }
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.dense_lead)
            );
            if let Some(v) = spec.weights_scale {
                kv!(format!("{a}.expert_weights_scale"), Value::F32(v));
            }
            if spec.weights_norm {
                kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
            }
            if let Some(v) = spec.gating {
                kv!(format!("{a}.expert_gating_func"), Value::U32(v));
            }
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
        Family::Plamo3 => {
            assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            assert_eq!(hp.n_swa, 64);
            assert_eq!(hp.rope_freq_base_train_swa, 30000.0);
            // load_swa_pattern(ml, 8): il % 8 < 7 → layer 7 is full attention
            for il in 0..8 {
                assert_eq!(hp.is_swa(il), il % 8 < 7, "plamo3 is_swa[{il}]");
            }
        }
        Family::Qwen3Next => {
            assert_eq!(hp.ssm_d_conv, 4);
            assert_eq!(hp.ssm_d_inner, 32);
            assert_eq!(hp.ssm_d_state, 8);
            assert_eq!(hp.ssm_dt_rank, 4);
            assert_eq!(hp.ssm_n_group, 2);
            assert_eq!(hp.expert_weights_scale, 0.0); // never read (qwen3next.cpp)
            assert_eq!(hp.n_ff_shexp, 24);
        }
        Family::KimiLinear => {
            assert_eq!(hp.n_embd_head_kda, 16);
            assert_eq!(hp.n_embd_head_k_mla(), 40);
            assert_eq!(hp.n_embd_head_v_mla(), 20);
            assert_eq!(hp.n_lora_kv, 32);
            assert_eq!(hp.n_rot(1), 16);
            assert_eq!(hp.expert_gating_func, 1); // SOFTMAX
            assert_eq!(hp.expert_weights_scale, 2.446);
            assert_eq!(hp.n_layer_dense_lead, 1);
            // the KDA state geometry (llama-hparams.cpp:216-223 / :242-247)
            let d_inner = hp.n_head(0) * hp.n_embd_head_kda;
            assert_eq!(hp.n_embd_r(), 3 * (hp.ssm_d_conv - 1) * d_inner);
            assert_eq!(
                hp.n_embd_s(),
                hp.n_embd_head_kda * hp.n_embd_head_kda * hp.n_head(0)
            );
        }
        Family::BailingMoe3 => {
            assert_eq!(hp.n_embd_head_kda, 16);
            assert_eq!(hp.kda_gate_lower_bound, -0.1);
            assert_eq!(hp.n_lora_q, 16);
            assert_eq!(hp.n_embd_head_k_mla(), 40);
            assert!(hp.expert_weights_norm);
            assert_eq!(hp.swiglu_clamp_exp[0], 7.0);
            assert_eq!(hp.swiglu_clamp_shexp[0], 7.0);
            assert_eq!(hp.n_ff_shexp, 24); // the explicit key wins
            assert_eq!(hp.n_layer_dense_lead, 1);
        }
    }
}

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
    let hp = &m.hparams;
    let first_attn = (0..hp.n_layer() as usize)
        .find(|&il| match m.arch {
            llama::arch::LlmArch::PLAMO3 => true,
            _ => !hp.is_recr(il),
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

fn plamo3_weights(m: &LlamaModel) -> graph_arch::Plamo3ModelWeights {
    graph_arch::Plamo3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(_il, l)| graph_arch::Plamo3LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wo: l.wo.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
            })
            .collect(),
    }
}

fn qwen3next_weights(m: &LlamaModel) -> graph_arch::Qwen3NextModelWeights {
    graph_arch::Qwen3NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(_il, l)| graph_arch::Qwen3NextLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta_alpha: l.ssm_beta_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

fn kimi_linear_weights(m: &LlamaModel) -> graph_arch::KimiLinearModelWeights {
    graph_arch::KimiLinearModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(_il, l)| graph_arch::KimiLinearLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wkv_b: l.wkv_b,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn bailingmoe3_weights(m: &LlamaModel) -> graph_arch::BailingMoe3ModelWeights {
    graph_arch::BailingMoe3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(_il, l)| graph_arch::BailingMoe3LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv_gate: l.wqkv_gate,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = synth_attn(m, fa);
    let first_attn = (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap_or(0);
    match m.arch {
        llama::arch::LlmArch::PLAMO3 => (
            ForwardWeights::Plamo3(
                plamo3_weights(m),
                graph_arch::Plamo3Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::QWEN3NEXT => (
            ForwardWeights::Qwen3Next(
                qwen3next_weights(m),
                graph_arch::Qwen3NextParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il)).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il)).collect(),
                    n_embd_head_k: (0..n_trunk).map(|il| hp.n_embd_head_k(il)).collect(),
                    n_embd_head_v: (0..n_trunk).map(|il| hp.n_embd_head_v(il)).collect(),
                    n_rot: (0..n_trunk).map(|il| hp.n_rot(il)).collect(),
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    f_attention_scale: hp.f_attention_scale,
                    ssm_d_conv: hp.ssm_d_conv as i64,
                    ssm_d_inner: hp.ssm_d_inner as i64,
                    ssm_d_state: hp.ssm_d_state as i64,
                    ssm_dt_rank: hp.ssm_dt_rank as i64,
                    ssm_n_group: hp.ssm_n_group as i64,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::KIMI_LINEAR => (
            ForwardWeights::KimiLinear(
                kimi_linear_weights(m),
                graph_arch::KimiLinearParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_head: hp.n_head(first_attn) as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    n_embd_head_kda: hp.n_embd_head_kda as i64,
                    ssm_d_conv: hp.ssm_d_conv as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    n_lora_kv: hp.n_lora_kv as i64,
                    n_embd_head_qk_rope: hp.n_rot(first_attn) as i64,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::BAILINGMOE3 => (
            ForwardWeights::BailingMoe3(
                bailingmoe3_weights(m),
                graph_arch::BailingMoe3Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_head: hp.n_head(first_attn) as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    n_embd_head_kda: hp.n_embd_head_kda as i64,
                    ssm_d_conv: hp.ssm_d_conv as i64,
                    kda_gate_lower_bound: hp.kda_gate_lower_bound,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    n_lora_kv: hp.n_lora_kv as i64,
                    n_lora_q: hp.n_lora_q as i64,
                    n_embd_head_qk_rope: hp.n_rot(first_attn) as i64,
                    rope_sections: hp.rope_sections,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    swiglu_clamp_exp: hp.swiglu_clamp_exp[..n_trunk].to_vec(),
                    swiglu_clamp_shexp: hp.swiglu_clamp_shexp[..n_trunk].to_vec(),
                },
            ),
            attn,
        ),
        other => panic!("arch {other:?} not in batch 9"),
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

/// one prefill (a 12-token prompt, crossing nothing yet) + one decode step;
/// asserts the recurrent cells and kv rows landed where they should
fn smoke_forward(m: &mut LlamaModel, spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut dctx = driver_for(m, fa);
    let prompt: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    // a decode step: the recurrence + the kv cache must continue cleanly
    let tk = logits
        .chunks(32000)
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32;
    let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
    // kv rows: the cell list is shared across the layers (llama_kv_cache's
    // single llama_kv_cells array) — 12 prompt rows + the decode row
    let used = dctx.kv.used_cells();
    assert_eq!(used, 13, "{} fa={fa}: kv cells", spec.arch);
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "writes the /tmp/arch-batch9 parity files (ARCH_BATCH9 cells)"]
fn arch_batch9_write_synth() {
    for spec in all_specs() {
        let (n, ck) = build_file(&spec);
        println!("{}: {n} tensors, {ck} bytes", spec.path());
    }
}

#[test]
fn arch_batch9_pin_and_smoke() {
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
fn arch_batch9_variant_legacy() {
    // the two in-port-only variants: qwen3next's fused ssm_in + separate
    // experts, kimi-linear's unsplit wkv_b MHA cache
    for spec in [spec_qwen3next_legacy(), spec_kimi_linear_legacy()] {
        let mut m = load_synth(&spec);
        pin_tensors(&m, &spec);
        for fa in [false, true] {
            let mut m = load_synth(&spec);
            let logits = smoke_forward(&mut m, &spec, fa);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{} fa={fa}: non-finite logits",
                spec.arch
            );
        }
        println!("{}: variant ok", spec.path());
    }
}

#[test]
fn arch_batch9_long_prompt_recurrence() {
    // a >64-token prompt in ONE ubatch is impossible (n_batch 512 but the
    // driver's ubatch is 8? see below) — instead decode 80 single tokens so
    // the recurrence runs far past the prompt and (for plamo3) past the SWA
    // window edges; kimi exercises the KDA state evolution over 80 steps
    let mut m = load_synth(&spec_kimi_linear());
    let mut dctx = driver_for(&mut m, false);
    let mut tk = 1i32;
    let mut logits = Vec::new();
    for p in 0..80 {
        let lg = dctx.decode(&[tk], &[p]).expect("decode").to_vec();
        tk = logits_of_argmax(&lg);
        logits = lg;
    }
    assert!(logits.iter().all(|v| v.is_finite()));
    assert_eq!(dctx.kv.used_cells(), 80);

    // plamo3: 80 positions cross the n_swa 64 window on the SWA layers
    let mut m = load_synth(&spec_plamo3());
    let mut dctx = driver_for(&mut m, false);
    let mut tk = 1i32;
    for p in 0..80 {
        let lg = dctx.decode(&[tk], &[p]).expect("decode").to_vec();
        tk = logits_of_argmax(&lg);
    }
    assert_eq!(dctx.kv.used_cells(), 80);
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}

/// the recurrent rollback planes (`cparams.n_rs_seq > 0`,
/// llama-memory-recurrent.cpp:180-210 + delta-net-base.cpp:497-522/:546-606 —
/// the port's host-side snapshot ring behind
/// `DecodeContext::with_rs_rollback`): decode → `seq_rm` back to a snapshot
/// → decode the same tokens again must reproduce the first pass's
/// post-snapshot outputs bit-identically (the GDN/KDA conv+ssm cells
/// restored from the ring, the attention cells rewound).
#[test]
fn qwen3next_rs_rollback_reproduces_post_snapshot_tokens() {
    let spec = spec_qwen3next().with(|s| s.suffix = "-rsrb");
    build_file(&spec);

    let mut m = load_synth(&spec);
    let (weights, attn) = forward_of(&mut m, false);
    let gctx = std::mem::replace(&mut m.ctx, ggml::Context::new());
    let mut d = DecodeContext::new_with(gctx, weights, attn, 512, 8, 512).with_rs_rollback(3);

    // first pass: greedy single-token decodes through 10 positions
    let n_tokens = 10usize;
    let mut logits_at: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    let mut toks = vec![1i32];
    let mut cur = 1i32;
    for p in 0..n_tokens as i32 {
        let o = d.decode(&[cur], &[p]).expect("first-pass step");
        let row = o.to_vec();
        logits_at.push(row.clone());
        cur = logits_of_argmax(&row);
        toks.push(cur);
    }

    // rollback 3: keep positions [0, 7) (p0 = 7 drops the last three tokens;
    // rollback = 9 - 6 = 3 == n_rs_seq)
    let p0 = 7i32;
    d.seq_rm(0, p0, -1);
    assert_eq!(
        d.seq_pos_max(0),
        6,
        "the attention cells rewound to position 6"
    );

    // second pass: re-decode the recorded tokens from the snapshot — the
    // outputs must reproduce the first pass's bit-identically
    for (i, &t) in toks[p0 as usize..n_tokens].iter().enumerate() {
        let p = p0 + i as i32;
        let o = d.decode(&[t], &[p]).expect("second-pass step");
        assert_eq!(
            o,
            &logits_at[p as usize][..],
            "position {p}: the rolled-back decode must reproduce the first pass"
        );
    }
}
