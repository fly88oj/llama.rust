//! arch_batch5_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-24: **the mamba family** — `mamba` (mamba1) / `mamba2` /
//! `jamba` (hybrid mamba1+attention, dense|MoE FFN) / `nemotron_h` (hybrid
//! mamba2 / attention / relu²-FFN(dense|MoE)) (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-4 (`crates/llama/tests/arch_batch{,2,3,4}_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV (incl. the `ssm.*` quartet/quintet and the per-layer
//! `head_count_kv` / `feed_forward_length` arrays of the hybrids), and exactly
//! the tensor names + shapes its `load_arch_tensors` asks for, all F32.
//!
//! The recurrent-state driver is the granite-hybrid one generalised
//! (`RecurrentState` with the arch's real `is_recr` pattern + n_embd_r /
//! n_embd_s): one conv cell + one SSM cell per recurrent layer, zeroed at a
//! fresh sequence (`llama_memory_recurrent`'s rs_zero), written back by the
//! graph's cpy nodes each step.
//!
//! Default-run tests (no reference needed): one per arch — write the file,
//! load it, pin the created tensor set / hparams / the recurrent pattern, and
//! run a 3-token decode through the arch builder with a bit-identical repeat
//! **after a full recurrent-state reset** (both -fa off and -fa on).
//!
//! `#[ignore]`d:
//!   * `arch_batch5_write_synth` — writes the files into /tmp/arch-batch5/
//!   * `arch5_cli_driver` — the port side of the parity runs: env-driven
//!     greedy decode printing llama-cli's `step N: top5 [...] greedy=` /
//!     `gen tokens: [...]` lines (`ARCH5_MODEL/ARCH5_FA/ARCH5_PROMPT/ARCH5_N`),
//!     compared by `parity/arch_batch_cmp.py` against the fresh-reference
//!     capture (`ARCH_BATCH5=1 ./parity/arch_batch_parity.sh`).

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::graph::{AttnParams, DecodeInputs, ForwardResult};
use llama::graph_arch::{self, RecurrentState};
use llama::kv_cache::{KvCache, SlotInfo};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch5";
const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

/// One batch-5 arch's synthetic geometry. `layer_kind[i]` drives both the
/// tensor table and the per-layer KV arrays: the hybrids discriminate their
/// layers by `head_count_kv == 0` (jamba) / `head_count_kv == 0 && n_ff == 0`
/// (nemotron-h) exactly like the C loaders.
#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    /// per-layer head_count_kv (written as the array the hybrids need; mamba
    /// writes nothing — pure-recurrent has no attention at all)
    head_kv: Vec<i64>,
    n_ff: i64,
    /// per-layer feed_forward_length (nemotron-h; None → scalar n_ff)
    n_ff_arr: Option<Vec<i64>>,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    // the ssm quintet (mamba1 archs never write group_count)
    d_conv: i64,
    d_inner: i64,
    d_state: i64,
    dt_rank: i64,
    n_group: i64,
    /// jamba: which layers carry ffn_gate_inp (the per-layer dense|MoE split,
    /// jamba.cpp:107-128)
    jamba_moe_layers: Vec<bool>,
    /// nemotron-h MoE variant keys
    n_ff_exp: Option<i64>,
    n_ff_shexp: Option<i64>,
    moe_latent: Option<i64>,
    /// nemotron-h writes attention.layer_norm_epsilon (required) instead of
    /// the RMS key
    eps_ln: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum LayerKind {
    /// mamba / mamba2 / jamba ssm / nemotron-h ssm
    Ssm,
    Attn,
    /// nemotron-h FFN-only block (relu² MLP or MoE)
    Ffn,
}

impl SynthSpec {
    fn head_dim(&self) -> i64 {
        self.n_embd / self.n_head
    }
    fn n_embd_kv(&self) -> i64 {
        self.n_head * self.head_dim()
    }
    /// the layer split each loader derives (jamba.cpp:12-14 /
    /// nemotron-h.cpp:14-16; mamba/mamba2 are all-ssm)
    fn layer_kind(&self, i: usize) -> LayerKind {
        match self.arch {
            "mamba" | "mamba2" => LayerKind::Ssm,
            "jamba" => {
                if self.head_kv[i] == 0 {
                    LayerKind::Ssm
                } else {
                    LayerKind::Attn
                }
            }
            "nemotron_h" => {
                let n_ff_i = self.n_ff_arr.as_ref().map(|a| a[i]).unwrap_or(self.n_ff);
                if self.head_kv[i] == 0 && n_ff_i == 0 {
                    LayerKind::Ssm
                } else if n_ff_i == 0 {
                    LayerKind::Attn
                } else {
                    LayerKind::Ffn
                }
            }
            other => panic!("no layer split for arch {other}"),
        }
    }
    fn is_recr(&self) -> Vec<bool> {
        (0..self.n_layer)
            .map(|i| self.layer_kind(i) == LayerKind::Ssm)
            .collect()
    }
    /// hparams.n_embd_r / n_embd_s of the recurrent cells
    fn n_embd_r(&self) -> i64 {
        match self.arch {
            "mamba" | "jamba" => (self.d_conv - 1) * self.d_inner,
            // mamba2-style: (d_conv-1) * (d_inner + 2*n_group*d_state)
            _ => (self.d_conv - 1) * (self.d_inner + 2 * self.n_group * self.d_state),
        }
    }
    fn n_embd_s(&self) -> i64 {
        self.d_state * self.d_inner
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

/// mamba (mamba1) — pure recurrent, 4 layers, d_inner = 2*n_embd
fn spec_mamba() -> SynthSpec {
    SynthSpec {
        arch: "mamba",
        suffix: "",
        n_layer: 4,
        n_embd: 64,
        n_head: 0,
        head_kv: vec![0; 4],
        n_ff: 0,
        n_ff_arr: None,
        n_ctx: 512,
        write_output: true,
        d_conv: 4,
        d_inner: 128,
        d_state: 16,
        dt_rank: 8,
        n_group: 0,
        jamba_moe_layers: vec![],
        n_ff_exp: None,
        n_ff_shexp: None,
        moe_latent: None,
        eps_ln: false,
    }
}

/// mamba2 — the scalar-decay form (A/D at {1, n_head}, grouped B/C, ssm_norm)
fn spec_mamba2() -> SynthSpec {
    spec_mamba().with(|s| {
        s.arch = "mamba2";
        s.n_group = 2;
        s.write_output = false;
    })
}

/// jamba — hybrid mamba1 + rope-less attention: mamba at 0/2/4, attention at
/// 1/3/5, dense FFN on every layer
fn spec_jamba() -> SynthSpec {
    SynthSpec {
        arch: "jamba",
        suffix: "",
        n_layer: 6,
        n_embd: 64,
        n_head: 4,
        head_kv: vec![0, 2, 0, 2, 0, 2],
        n_ff: 96,
        n_ff_arr: None,
        n_ctx: 512,
        write_output: true,
        d_conv: 4,
        d_inner: 128,
        d_state: 16,
        dt_rank: 8,
        n_group: 0,
        jamba_moe_layers: vec![false; 6],
        n_ff_exp: None,
        n_ff_shexp: None,
        moe_latent: None,
        eps_ln: false,
    }
}

/// jamba-moe — the per-layer dense|MoE FFN split of jamba.cpp:107-128: layers
/// 0/1/3/5 carry ffn_gate_inp (MoE), 2/4 stay dense
fn spec_jamba_moe() -> SynthSpec {
    spec_jamba().with(|s| {
        s.suffix = "-moe";
        s.jamba_moe_layers = vec![true, true, false, true, false, true];
        // jamba's MoE has no expert_feed_forward_length key — the experts run
        // at the dense n_ff (jamba.cpp:118-123)
    })
}

/// nemotron_h — hybrid mamba2 (0/2/5) / attention (1/3) / dense relu² MLP (4)
fn spec_nemotron_h() -> SynthSpec {
    SynthSpec {
        arch: "nemotron_h",
        suffix: "",
        n_layer: 6,
        n_embd: 64,
        n_head: 4,
        head_kv: vec![0, 2, 0, 2, 2, 0],
        n_ff: 96,
        n_ff_arr: Some(vec![0, 0, 0, 0, 96, 0]),
        n_ctx: 512,
        write_output: true,
        d_conv: 4,
        d_inner: 128,
        d_state: 16,
        dt_rank: 8,
        n_group: 2,
        jamba_moe_layers: vec![],
        n_ff_exp: None,
        n_ff_shexp: None,
        moe_latent: None,
        eps_ln: true,
    }
}

/// nemotron_h-moe — layer 4 becomes the MoE branch (sigmoid gating, router
/// bias, relu² experts, the latent projections and the relu² shared expert,
/// nemotron-h.cpp:111-130/291-333)
fn spec_nemotron_h_moe() -> SynthSpec {
    spec_nemotron_h().with(|s| {
        s.suffix = "-moe";
        s.n_ff_exp = Some(32);
        s.n_ff_shexp = Some(24);
        s.moe_latent = Some(32);
        s.write_output = false;
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_mamba(),
        spec_mamba2(),
        spec_jamba(),
        spec_jamba_moe(),
        spec_nemotron_h(),
        spec_nemotron_h_moe(),
    ]
}

// ---------------------------------------------------------------------------
// per-arch tensor tables — the create_tensor calls of each
// src/models/<arch>.cpp load_arch_tensors, in file order
// ---------------------------------------------------------------------------

type TensorSpec = (String, Vec<i64>);

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Norm,
    Bias,
    Embd,
    Head,
    Proj,
    Router,
    /// A / dt_proj-weight: small-scale signed random (keeps exp(dt*A) bounded
    /// over 48 recurrent steps; any value is parity-safe — the reference
    /// reads the same bytes)
    Decay,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let d_conv = spec.d_conv;
    let d_inner = spec.d_inner;
    let d_state = spec.d_state;
    let dt_rank = spec.dt_rank;
    let ng = spec.n_group;
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    match spec.arch {
        // ---- mamba.cpp:62-114 (mamba1) ----
        "mamba" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ssm_in.weight"),
                    vec![n_embd, 2 * d_inner],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_conv1d.weight"),
                    vec![d_conv, d_inner],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_conv1d.bias"),
                    vec![d_inner],
                    Role::Bias
                );
                push!(
                    format!("blk.{i}.ssm_x.weight"),
                    vec![d_inner, dt_rank + 2 * d_state],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_dt.weight"),
                    vec![dt_rank, d_inner],
                    Role::Decay
                );
                push!(format!("blk.{i}.ssm_dt.bias"), vec![d_inner], Role::Bias);
                // no .weight suffix for ssm_a / ssm_d
                push!(
                    format!("blk.{i}.ssm_a"),
                    vec![d_state, d_inner],
                    Role::Decay
                );
                push!(format!("blk.{i}.ssm_d"), vec![d_inner], Role::Bias);
                push!(
                    format!("blk.{i}.ssm_out.weight"),
                    vec![d_inner, n_embd],
                    Role::Proj
                );
            }
        }

        // ---- mamba2.cpp:54-90 ----
        "mamba2" => {
            let conv_dim = d_inner + 2 * ng * d_state;
            let d_in_proj = d_inner + conv_dim + dt_rank;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ssm_in.weight"),
                    vec![n_embd, d_in_proj],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_conv1d.weight"),
                    vec![d_conv, conv_dim],
                    Role::Proj
                );
                // mamba2.cpp:70 — the conv bias is REQUIRED
                push!(
                    format!("blk.{i}.ssm_conv1d.bias"),
                    vec![conv_dim],
                    Role::Bias
                );
                push!(format!("blk.{i}.ssm_dt.bias"), vec![dt_rank], Role::Bias);
                push!(format!("blk.{i}.ssm_a"), vec![1, dt_rank], Role::Decay);
                push!(format!("blk.{i}.ssm_d"), vec![1, dt_rank], Role::Bias);
                push!(
                    format!("blk.{i}.ssm_norm.weight"),
                    vec![d_inner / ng, ng],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ssm_out.weight"),
                    vec![d_inner, n_embd],
                    Role::Proj
                );
            }
        }

        // ---- jamba.cpp:47-128 ----
        "jamba" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if spec.layer_kind(i) == LayerKind::Ssm {
                    push!(
                        format!("blk.{i}.ssm_in.weight"),
                        vec![n_embd, 2 * d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![d_conv, d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d.bias"),
                        vec![d_inner],
                        Role::Bias
                    );
                    push!(
                        format!("blk.{i}.ssm_x.weight"),
                        vec![d_inner, dt_rank + 2 * d_state],
                        Role::Proj
                    );
                    // the dt/B/C RMS trio is required on jamba's mamba layers
                    push!(
                        format!("blk.{i}.ssm_dt_norm.weight"),
                        vec![dt_rank],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.ssm_dt.weight"),
                        vec![dt_rank, d_inner],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_dt.bias"), vec![d_inner], Role::Bias);
                    push!(
                        format!("blk.{i}.ssm_b_norm.weight"),
                        vec![d_state],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.ssm_c_norm.weight"),
                        vec![d_state],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.ssm_a"),
                        vec![d_state, d_inner],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_d"), vec![d_inner], Role::Bias);
                    push!(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![d_inner, n_embd],
                        Role::Proj
                    );
                } else {
                    // attention layers (jamba.cpp:104-108): separate q/k/v at
                    // MHA width (q = n_embd, k/v = n_embd_gqa), no biases
                    let n_kv = spec.head_kv[i] * spec.head_dim();
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, n_embd],
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
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![n_embd, n_embd],
                        Role::Proj
                    );
                }
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if spec.jamba_moe_layers[i] {
                    push!(
                        format!("blk.{i}.ffn_gate_inp.weight"),
                        vec![n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, n_ff, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![n_ff, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, n_ff, N_EXPERT],
                        Role::Router
                    );
                } else {
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
                }
            }
        }

        // ---- nemotron-h.cpp:66-140 (the no-nextn slice: block_count 6,
        // nextn_predict_layers absent) ----
        "nemotron_h" => {
            let conv_dim = d_inner + 2 * ng * d_state;
            let d_in_proj = 2 * d_inner + 2 * ng * d_state + dt_rank;
            let moe_n_embd = spec.moe_latent.unwrap_or(0).max(0);
            let moe_n_embd = if moe_n_embd > 0 { moe_n_embd } else { n_embd };
            let n_ff_exp = spec.n_ff_exp.unwrap_or(0);
            let n_ff_shexp = spec.n_ff_shexp.unwrap_or(0);
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            if spec.write_output {
                push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            }
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                match spec.layer_kind(i) {
                    LayerKind::Ssm => {
                        push!(
                            format!("blk.{i}.ssm_in.weight"),
                            vec![n_embd, d_in_proj],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.ssm_conv1d.weight"),
                            vec![d_conv, conv_dim],
                            Role::Proj
                        );
                        // nemotron-h.cpp:90 — the conv bias is optional
                        push!(
                            format!("blk.{i}.ssm_conv1d.bias"),
                            vec![conv_dim],
                            Role::Bias
                        );
                        push!(format!("blk.{i}.ssm_dt.bias"), vec![dt_rank], Role::Bias);
                        push!(format!("blk.{i}.ssm_a"), vec![1, dt_rank], Role::Decay);
                        push!(format!("blk.{i}.ssm_d"), vec![1, dt_rank], Role::Bias);
                        push!(
                            format!("blk.{i}.ssm_norm.weight"),
                            vec![d_inner / ng, ng],
                            Role::Norm
                        );
                        push!(
                            format!("blk.{i}.ssm_out.weight"),
                            vec![d_inner, n_embd],
                            Role::Proj
                        );
                    }
                    LayerKind::Attn => {
                        let n_head_i = spec.n_head;
                        let n_kv = spec.head_kv[i] * spec.head_dim();
                        push!(
                            format!("blk.{i}.attn_q.weight"),
                            vec![n_embd, spec.head_dim() * n_head_i],
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
                        push!(
                            format!("blk.{i}.attn_output.weight"),
                            vec![spec.head_dim() * n_head_i, n_embd],
                            Role::Proj
                        );
                    }
                    LayerKind::Ffn => {
                        let n_ff_i = spec.n_ff_arr.as_ref().map(|a| a[i]).unwrap_or(n_ff);
                        if n_ff_exp > 0 {
                            // MoE branch (nemotron-h.cpp:118-130)
                            push!(
                                format!("blk.{i}.ffn_gate_inp.weight"),
                                vec![n_embd, N_EXPERT],
                                Role::Router
                            );
                            push!(
                                format!("blk.{i}.exp_probs_b.bias"),
                                vec![N_EXPERT],
                                Role::Bias
                            );
                            push!(
                                format!("blk.{i}.ffn_latent_down.weight"),
                                vec![n_embd, moe_n_embd],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ffn_latent_up.weight"),
                                vec![moe_n_embd, n_embd],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ffn_down_exps.weight"),
                                vec![n_ff_exp, moe_n_embd, N_EXPERT],
                                Role::Router
                            );
                            push!(
                                format!("blk.{i}.ffn_up_exps.weight"),
                                vec![moe_n_embd, n_ff_exp, N_EXPERT],
                                Role::Router
                            );
                            push!(
                                format!("blk.{i}.ffn_down_shexp.weight"),
                                vec![n_ff_shexp, n_embd],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ffn_up_shexp.weight"),
                                vec![n_embd, n_ff_shexp],
                                Role::Proj
                            );
                        } else {
                            // dense mlp (nemotron-h.cpp:133-138)
                            push!(
                                format!("blk.{i}.ffn_down.weight"),
                                vec![n_ff_i, n_embd],
                                Role::Proj
                            );
                            push!(
                                format!("blk.{i}.ffn_up.weight"),
                                vec![n_embd, n_ff_i],
                                Role::Proj
                            );
                        }
                    }
                }
            }
        }

        other => panic!("no synthetic tensor table for arch {other}"),
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch3/4_e2e.rs)
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
        Role::Embd | Role::Head => 1.0,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        Role::Router => 1.0 / (n_embd as f32).sqrt(),
        // keep exp(dt*A) bounded across 48+ recurrent steps
        Role::Decay => 0.05,
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch5");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, v) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, v.clone());
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
    // the ssm keys (mamba.cpp:4-8 / mamba2.cpp:4-9 / jamba.cpp:4-8 /
    // nemotron-h.cpp:6-10)
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
    if spec.n_group > 0 {
        // mamba1/jamba never write group_count (that is what keeps their
        // n_embd_r at (d_conv-1)*d_inner)
        kv!(
            format!("{a}.ssm.group_count"),
            Value::U32(spec.n_group as u32)
        );
    }
    // the norm eps: nemotron-h requires attention.layer_norm_epsilon and falls
    // back to it for the RMS eps (nemotron-h.cpp:18-21)
    if spec.eps_ln {
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
    // attention geometry (the attention layers of the hybrids; head_dim via
    // the scalar key/value-length keys — jamba's graph reads n_embd_head_v()
    // of layer 0)
    if spec.n_head > 0 {
        kv!(
            format!("{a}.attention.head_count"),
            Value::U32(spec.n_head as u32)
        );
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::Array(
                GgufType::Uint32,
                spec.head_kv.iter().map(|&h| Value::U32(h as u32)).collect()
            )
        );
        kv!(
            format!("{a}.attention.key_length"),
            Value::U32(spec.head_dim() as u32)
        );
        kv!(
            format!("{a}.attention.value_length"),
            Value::U32(spec.head_dim() as u32)
        );
        kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    }
    // FFN widths: the hybrids need per-layer arrays (nemotron-h's n_ff is the
    // attention-vs-FFN discriminator)
    if let Some(arr) = &spec.n_ff_arr {
        kv!(
            format!("{a}.feed_forward_length"),
            Value::Array(
                GgufType::Uint32,
                arr.iter().map(|&f| Value::U32(f as u32)).collect()
            )
        );
    } else if spec.n_ff > 0 {
        kv!(
            format!("{a}.feed_forward_length"),
            Value::U32(spec.n_ff as u32)
        );
    }
    // MoE keys (nemotron-h.cpp:24-29; jamba's MoE uses only expert_count /
    // expert_used_count)
    if spec.arch == "nemotron_h" || !spec.jamba_moe_layers.is_empty() {
        if spec.n_ff_exp.is_some() || !spec.jamba_moe_layers.iter().all(|&m| !m) {
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
        }
    }
    if let Some(v) = spec.n_ff_exp {
        kv!(
            format!("{a}.expert_feed_forward_length"),
            Value::U32(v as u32)
        );
    }
    if let Some(v) = spec.n_ff_shexp {
        kv!(
            format!("{a}.expert_shared_feed_forward_length"),
            Value::U32(v as u32)
        );
    }
    if let Some(v) = spec.moe_latent {
        kv!(format!("{a}.moe_latent_size"), Value::U32(v as u32));
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

fn pin_hparams(m: &LlamaModel, spec: &SynthSpec) {
    assert_eq!(m.arch.name(), spec.arch);
    let hp = &m.hparams;
    assert_eq!(hp.n_embd as i64, spec.n_embd);
    assert_eq!(hp.n_layer() as usize, spec.n_layer);
    assert_eq!(hp.n_ctx_train, spec.n_ctx);
    assert_eq!(hp.ssm_d_conv as i64, spec.d_conv);
    assert_eq!(hp.ssm_d_inner as i64, spec.d_inner);
    assert_eq!(hp.ssm_d_state as i64, spec.d_state);
    assert_eq!(hp.ssm_dt_rank as i64, spec.dt_rank);
    assert_eq!(hp.ssm_n_group as i64, spec.n_group);
    assert_eq!(hp.f_norm_rms_eps, 1e-5);
    // the recurrent pattern (jamba.cpp:12-14 / nemotron-h.cpp:14-16; mamba /
    // mamba2 all-1 via llm_arch_is_recurrent)
    assert_eq!(
        (0..spec.n_layer)
            .map(|il| hp.is_recr(il))
            .collect::<Vec<_>>(),
        spec.is_recr(),
        "{}: is_recr pattern",
        spec.arch
    );
    // the recurrent cell geometry (llama-hparams.cpp n_embd_r/n_embd_s)
    assert_eq!(hp.n_embd_r() as i64, spec.n_embd_r());
    assert_eq!(hp.n_embd_s() as i64, spec.n_embd_s());
    // rope type NONE for the whole family (llama-model.cpp rope_type switch)
    assert_eq!(hp.rope_type, llama::hparams::LlamaRopeType::NONE);
}

// ---------------------------------------------------------------------------
// the arch dispatch: weights + params assembly (one enum arm per arch)
// ---------------------------------------------------------------------------

fn synth_attn(m: &LlamaModel, fa: bool, il: usize) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head_k: hp.n_embd_head_k(il) as i64,
        n_embd_head_v: hp.n_embd_head_v(il) as i64,
        n_rot: hp.n_rot(il) as i64,
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

enum Batch5Model {
    Mamba(graph_arch::MambaModelWeights, graph_arch::MambaParams, bool),
    Jamba(graph_arch::JambaModelWeights, graph_arch::JambaParams),
    NemotronH(
        graph_arch::NemotronHModelWeights,
        graph_arch::NemotronHParams,
    ),
}

fn mamba1_mixer_of(l: &llama::model::LayerTensors) -> graph_arch::Mamba1Mixer {
    graph_arch::Mamba1Mixer {
        ssm_in: l.ssm_in.unwrap(),
        ssm_conv1d: l.ssm_conv1d.unwrap(),
        ssm_conv1d_b: l.ssm_conv1d_b.unwrap(),
        ssm_x: l.ssm_x.unwrap(),
        ssm_dt: l.ssm_dt.unwrap(),
        ssm_dt_b: l.ssm_dt_b.unwrap(),
        ssm_dt_norm: l.ssm_dt_norm,
        ssm_b_norm: l.ssm_b_norm,
        ssm_c_norm: l.ssm_c_norm,
        ssm_a: l.ssm_a.unwrap(),
        ssm_d: l.ssm_d.unwrap(),
        ssm_out: l.ssm_out.unwrap(),
    }
}

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

impl Batch5Model {
    fn assemble(m: &LlamaModel, fa: bool) -> Self {
        let hp = &m.hparams;
        let n_trunk = hp.n_layer() as usize;
        match m.arch {
            llama::arch::LlmArch::MAMBA | llama::arch::LlmArch::MAMBA2 => {
                let mamba2 = m.arch == llama::arch::LlmArch::MAMBA2;
                let layers = m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::MambaLayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        mixer: if mamba2 {
                            graph_arch::MambaLayerMixer::Mamba2(mamba2_mixer_of(l))
                        } else {
                            graph_arch::MambaLayerMixer::Mamba1(mamba1_mixer_of(l))
                        },
                    })
                    .collect();
                Batch5Model::Mamba(
                    graph_arch::MambaModelWeights {
                        tok_embd: m.tok_embd,
                        output_norm: m.output_norm,
                        output: m.output,
                        layers,
                    },
                    graph_arch::MambaParams {
                        d_conv: hp.ssm_d_conv as i64,
                        d_inner: hp.ssm_d_inner as i64,
                        d_state: hp.ssm_d_state as i64,
                        dt_rank: hp.ssm_dt_rank as i64,
                        n_group: hp.ssm_n_group as i64,
                        ssm_dt_b_c_rms: hp.ssm_dt_b_c_rms,
                        norm_eps: hp.f_norm_rms_eps,
                        n_embd_r: hp.n_embd_r(),
                        n_embd_s: hp.n_embd_s(),
                    },
                    mamba2,
                )
            }
            llama::arch::LlmArch::JAMBA => {
                let attn = synth_attn(m, fa, (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap());
                let layers = m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::JambaLayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        mamba: l.ssm_in.map(|_| mamba1_mixer_of(l)),
                        wq: l.wq,
                        wk: l.wk,
                        wv: l.wv,
                        wq_b: l.wq_b,
                        wk_b: l.wk_b,
                        wv_b: l.wv_b,
                        wo: l.wo,
                        ffn_norm: l.ffn_norm.unwrap(),
                        ffn_gate: l.ffn_gate,
                        ffn_down: l.ffn_down,
                        ffn_up: l.ffn_up,
                        ffn_gate_inp: l.ffn_gate_inp,
                        ffn_gate_exps: l.ffn_gate_exps,
                        ffn_down_exps: l.ffn_down_exps,
                        ffn_up_exps: l.ffn_up_exps,
                    })
                    .collect();
                Batch5Model::Jamba(
                    graph_arch::JambaModelWeights {
                        tok_embd: m.tok_embd,
                        output_norm: m.output_norm,
                        output: m.output,
                        layers,
                    },
                    graph_arch::JambaParams {
                        attn,
                        n_embd: hp.n_embd as i64,
                        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                        d_conv: hp.ssm_d_conv as i64,
                        d_inner: hp.ssm_d_inner as i64,
                        d_state: hp.ssm_d_state as i64,
                        dt_rank: hp.ssm_dt_rank as i64,
                        norm_eps: hp.f_norm_rms_eps,
                        n_expert: hp.n_expert as i64,
                        n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il)).collect(),
                        expert_weights_scale: hp.expert_weights_scale,
                        n_embd_r: hp.n_embd_r(),
                        n_embd_s: hp.n_embd_s(),
                    },
                )
            }
            llama::arch::LlmArch::NEMOTRON_H => {
                let attn = synth_attn(
                    m,
                    fa,
                    (0..n_trunk)
                        .find(|&il| !hp.is_recr(il) && hp.n_ff(il) == 0)
                        .unwrap(),
                );
                let layers = m.layers[..n_trunk]
                    .iter()
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
                    .collect();
                Batch5Model::NemotronH(
                    graph_arch::NemotronHModelWeights {
                        tok_embd: m.tok_embd,
                        output_norm: m.output_norm,
                        output: m.output,
                        layers,
                    },
                    graph_arch::NemotronHParams {
                        attn,
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
            other => panic!("batch5: arch {:?} not wired", other),
        }
    }

    /// the `RecurrentState` geometry (llama_memory_recurrent's per-layer cells)
    fn recurrent_geometry(&self) -> (Vec<bool>, u32, u32) {
        match self {
            Batch5Model::Mamba(w, p, _) => (vec![true; w.layers.len()], p.n_embd_r, p.n_embd_s),
            Batch5Model::Jamba(_, p) => (p.is_recr.clone(), p.n_embd_r, p.n_embd_s),
            Batch5Model::NemotronH(_, p) => (p.is_recr.clone(), p.n_embd_r, p.n_embd_s),
        }
    }

    fn n_layer(&self) -> usize {
        match self {
            Batch5Model::Mamba(w, _, _) => w.layers.len(),
            Batch5Model::Jamba(w, _) => w.layers.len(),
            Batch5Model::NemotronH(w, _) => w.layers.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        ctx: &mut Context,
        st: &RecurrentState,
        kv: &KvCache,
        inp: &DecodeInputs,
        sinfo: SlotInfo,
        n_kv: u32,
        n_tokens: usize,
    ) -> ForwardResult {
        match self {
            Batch5Model::Mamba(w, p, _) => graph_arch::build_mamba_forward(ctx, w, p, st, inp),
            Batch5Model::Jamba(w, p) => {
                graph_arch::build_jamba_forward(ctx, w, p, st, kv, inp, n_kv, n_tokens)
            }
            Batch5Model::NemotronH(w, p) => {
                graph_arch::build_nemotron_h_forward(ctx, w, p, st, kv, inp, sinfo, n_kv, n_tokens)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the decode harness — batch 4's Driver with a *real* RecurrentState (the
// granite-hybrid pattern: cells allocated below the watermark, zeroed on a
// fresh sequence = llama_memory_recurrent's rs_zero)
// ---------------------------------------------------------------------------

struct Driver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    fa: bool,
    rstate: RecurrentState,
}

fn driver_for(m: &mut LlamaModel, fa: bool) -> (Batch5Model, Driver) {
    let model = Batch5Model::assemble(m, fa);
    let n_layer = model.n_layer();
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    // llama-kv-cache.cpp:210-211 allocates n_embd_k_gqa(il) rows per layer; the
    // pure-recurrent archs have n_head_kv == 0 everywhere, so their rows are
    // 0-wide (llama-model.cpp:2547 creates no KV cache at all there — the
    // port's 0-row cache is the same no-op)
    let k_row: Vec<i64> = (0..n_layer).map(|il| m.n_embd_k_gqa(il) as i64).collect();
    let v_row: Vec<i64> = (0..n_layer).map(|il| m.n_embd_v_gqa(il) as i64).collect();
    let kv = if k_row.iter().all(|&k| k == k_row[0]) && v_row.iter().all(|&v| v == v_row[0]) {
        KvCache::new(&mut gctx, n_layer, k_row[0], v_row[0], 512)
    } else {
        KvCache::new_with_dims(&mut gctx, &k_row, &v_row, 512)
    };
    // the recurrent cells live below the watermark so reset_graph_to keeps
    // them across steps (llama.cpp allocates them in the memory module's own
    // context, outside the per-ubatch graph)
    let (is_recr, n_embd_r, n_embd_s) = model.recurrent_geometry();
    let rstate = RecurrentState::new(&mut gctx, &is_recr, n_embd_r, n_embd_s);
    let watermark = gctx.mark();
    (
        model,
        Driver {
            gctx,
            kv,
            watermark,
            fa,
            rstate,
        },
    )
}

impl Driver {
    /// Decode `tokens` at `pos`; returns the *last* token's logits [n_vocab].
    fn decode(&mut self, model: &Batch5Model, tokens: &[i32], pos: &[i32]) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        // assign first, then the (256-padded) n_kv — step_inputs order
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        // FA requires an F16 mask (llama-graph.cpp:38-39)
        let mask_ty = if self.fa {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        let kq_mask = self.gctx.new_tensor_2d(mask_ty, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| p.copy_from_slice(pos))
            .unwrap();
        {
            let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
            self.gctx
                .data_bytes_mut(row_idx)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&idxs));
        }
        {
            // set_input_kq_mask (llama-kv-cache.cpp:1557-1705), non-alibi
            // causal fill (no batch-5 arch has max_alibi_bias). Padded
            // (empty) cells keep pos = -1 → masked by the fills.
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            if mask_ty == GgmlType::F16 {
                let mask: &mut [half::f16] = bytemuck::cast_slice_mut(mask_bytes);
                llama::graph::fill_causal_mask_f16(mask, &kv_pos, pos);
            } else {
                let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
                llama::graph::fill_causal_mask(mask, &kv_pos, pos);
            }
        }
        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };

        let result = model.forward(
            &mut self.gctx,
            &self.rstate,
            &self.kv,
            &inputs,
            sinfo,
            n_kv,
            n,
        );
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);
        // (assign already happened before the graph build, step_inputs order)

        let n_vocab = self.gctx.ne(logits)[0] as usize;
        let all: Vec<f32> = self
            .gctx
            .data_bytes(logits)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        all[n_vocab * (n - 1)..n_vocab * n].to_vec()
    }

    /// fresh sequence: clear the KV cells **and zero the recurrent cells**
    /// (`llama_memory_recurrent`'s rs_zero — a sequence with no history starts
    /// from the all-zero state)
    fn reset_sequence(&mut self) {
        self.kv.clear();
        self.rstate.zero(&mut self.gctx);
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
/// bit-identical repeat **after a full recurrent-state reset** — through both
/// FA modes so wiring issues surface here rather than in the parity runs.
fn smoke_forward(spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut m = load_synth(spec);
    let (model, mut d) = driver_for(&mut m, fa);

    let toks = [3i32, 17, 42];
    let pos: Vec<i32> = (0..3).collect();
    let a = d.decode(&model, &toks, &pos);
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{}: non-finite logits (fa={fa})",
        spec.arch
    );
    let spread = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - a.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}: logits degenerate (spread {spread}, fa={fa})",
        spec.arch
    );

    let next = argmax(&a);
    let b = d.decode(&model, &[next], &[3]);
    assert!(b.iter().all(|v| v.is_finite()));

    // the recurrent state must reset bit-identically (rs_zero)
    d.reset_sequence();
    let c = d.decode(&model, &toks, &pos);
    assert_eq!(
        a, c,
        "{}: prefill not deterministic after state reset (fa={fa})",
        spec.arch
    );
    a
}

// ---------------------------------------------------------------------------
// default-run tests, one per arch
// ---------------------------------------------------------------------------

#[test]
fn synth_mamba_loader_and_forward() {
    let spec = spec_mamba();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 4 * 10, "mamba tensor count");
    println!("mamba synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // mamba1's own mixer tensors: x_proj {d_inner, dt_rank+2*d_state},
    // dt_proj {dt_rank, d_inner}, per-channel A {d_state, d_inner}
    let l0 = &m.layers[0];
    assert_eq!(*m.ctx.ne(l0.ssm_x.unwrap()), [128, 40, 1, 1]);
    assert_eq!(*m.ctx.ne(l0.ssm_dt.unwrap()), [8, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(l0.ssm_a.unwrap()), [16, 128, 1, 1]);
    assert_eq!(*m.ctx.ne(l0.ssm_d.unwrap()), [128, 1, 1, 1]);
    assert!(l0.ssm_dt_norm.is_none() && l0.ssm_b_norm.is_none() && l0.ssm_c_norm.is_none());
    assert!(!m.hparams.ssm_dt_b_c_rms);
    // all layers recurrent (llm_arch_is_recurrent)
    assert!(m.hparams.is_recr(0) && m.hparams.is_recr(3));

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "mamba synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_mamba2_loader_and_forward() {
    let spec = spec_mamba2();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 4 * 9, "mamba2 tensor count (tied head)");
    println!(
        "mamba2 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    let l0 = &m.layers[0];
    // mamba2's scalar decay: A/D at {1, n_head}, grouped norm {d_inner/ng, ng}
    assert_eq!(*m.ctx.ne(l0.ssm_a.unwrap()), [1, 8, 1, 1]);
    assert_eq!(*m.ctx.ne(l0.ssm_d.unwrap()), [1, 8, 1, 1]);
    assert_eq!(*m.ctx.ne(l0.ssm_norm.unwrap()), [64, 2, 1, 1]);
    // d_in_proj = d_inner + conv_dim + dt_rank = 128 + (128+2*2*16) + 8
    assert_eq!(*m.ctx.ne(l0.ssm_in.unwrap()), [64, 328, 1, 1]);
    assert_eq!(m.output, m.tok_embd, "mamba2 ties the head");
    assert!(m.layers.iter().all(|l| l.ssm_conv1d_b.is_some()));

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "mamba2 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_jamba_loader_and_forward() {
    let spec = spec_jamba();
    let (n, bytes) = build_file(&spec);
    // 3 model + mamba layers (attn_norm + 12 ssm tensors + ffn_norm + 3 ffn =
    // 17 each) + attention layers (attn_norm + q/k/v/out + ffn_norm + 3 ffn =
    // 9 each)
    assert_eq!(n, 3 + 3 * 17 + 3 * 9, "jamba tensor count");
    println!("jamba synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // the hybrid split: mamba layers carry the dt/B/C RMS trio
    for il in [0usize, 2, 4] {
        let l = &m.layers[il];
        assert!(l.ssm_in.is_some() && l.ssm_x.is_some() && l.wq.is_none());
        assert!(l.ssm_dt_norm.is_some() && l.ssm_b_norm.is_some() && l.ssm_c_norm.is_some());
    }
    for il in [1usize, 3, 5] {
        let l = &m.layers[il];
        assert!(l.wq.is_some() && l.wk.is_some() && l.wv.is_some() && l.ssm_in.is_none());
        assert!(l.ffn_gate.is_some() && l.ffn_gate_inp.is_none());
    }

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "jamba synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_jamba_moe_loader_and_forward() {
    let spec = spec_jamba_moe();
    let (n, bytes) = build_file(&spec);
    // mamba+MoE (layer 0) = 18, attn+MoE (1/3/5) = 10 each, mamba dense (2/4)
    // = 17 each
    assert_eq!(n, 3 + 18 + 3 * 10 + 2 * 17, "jamba-moe tensor count");
    println!(
        "jamba-moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    assert_eq!(m.hparams.n_expert as i64, N_EXPERT);
    assert_eq!(m.hparams.n_expert_used(0) as i64, N_EXPERT_USED);
    for (il, l) in m.layers.iter().enumerate() {
        if spec.jamba_moe_layers[il] {
            assert!(l.ffn_gate_inp.is_some() && l.ffn_gate_exps.is_some());
            assert!(l.ffn_gate.is_none());
        } else {
            assert!(l.ffn_gate_inp.is_none() && l.ffn_gate.is_some());
        }
    }

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "jamba-moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_nemotron_h_loader_and_forward() {
    let spec = spec_nemotron_h();
    let (n, bytes) = build_file(&spec);
    // 3 model + ssm layers (0/2/5: attn_norm + 8 ssm tensors each) + attn
    // layers (1/3: attn_norm + q/k/v/out each) + the ffn layer (4: attn_norm +
    // down + up)
    assert_eq!(n, 3 + 3 * 9 + 2 * 5 + 3, "nemotron-h tensor count");
    println!(
        "nemotron-h synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // the triple split (nemotron-h.cpp:14-16)
    for il in [0usize, 2, 5] {
        assert!(m.hparams.is_recr(il) && m.layers[il].ssm_in.is_some());
        assert!(m.layers[il].wq.is_none() && m.layers[il].ffn_up.is_none());
    }
    for il in [1usize, 3] {
        assert!(!m.hparams.is_recr(il) && m.layers[il].wq.is_some());
        assert!(m.layers[il].ssm_in.is_none() && m.layers[il].ffn_up.is_none());
    }
    let l4 = &m.layers[4];
    assert!(l4.wq.is_none() && l4.ssm_in.is_none());
    assert!(l4.ffn_up.is_some() && l4.ffn_down.is_some());
    // the LayerNorm-eps fallback (nemotron-h.cpp:18-21)
    assert_eq!(m.hparams.f_norm_eps, 1e-5);
    assert_eq!(m.hparams.f_norm_rms_eps, 1e-5);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "nemotron-h synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_nemotron_h_moe_loader_and_forward() {
    let spec = spec_nemotron_h_moe();
    let (n, bytes) = build_file(&spec);
    // tied head (2 model tensors); layer 4 carries the 8 MoE tensors +
    // attn_norm instead of the 2 dense ffn ones
    assert_eq!(n, 2 + 3 * 9 + 2 * 5 + 9, "nemotron-h-moe tensor count");
    println!(
        "nemotron-h-moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    let l4 = &m.layers[4];
    assert!(l4.ffn_gate_inp.is_some() && l4.ffn_exp_probs_b.is_some());
    assert!(l4.ffn_up_exps.is_some() && l4.ffn_down_exps.is_some());
    assert!(l4.ffn_latent_down.is_some() && l4.ffn_latent_up.is_some());
    assert!(l4.ffn_up_shexp.is_some() && l4.ffn_down_shexp.is_some());
    // the latent dims (moe_latent_size = 32)
    assert_eq!(*m.ctx.ne(l4.ffn_latent_down.unwrap()), [64, 32, 1, 1]);
    assert_eq!(*m.ctx.ne(l4.ffn_down_exps.unwrap()), [32, 32, N_EXPERT, 1]);
    assert_eq!(m.output, m.tok_embd);
    assert_eq!(m.hparams.n_ff_exp(4) as i64, 32);
    assert_eq!(m.hparams.n_ff_shexp, 24);
    assert_eq!(m.hparams.moe_latent_size, 32);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "nemotron-h-moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_mamba2().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("mamba2"));
    assert_eq!(g.get_u32("mamba2.ssm.conv_kernel"), Some(4));
    assert_eq!(g.get_u32("mamba2.ssm.group_count"), Some(2));
    let spec = spec_jamba().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    // the per-layer head_count_kv array survives the round-trip
    assert!(matches!(
        g.find_key("jamba.attention.head_count_kv")
            .map(|v| v.type_()),
        Some(GgufType::Array)
    ));
    let spec = spec_nemotron_h_moe().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert!(matches!(
        g.find_key("nemotron_h.feed_forward_length")
            .map(|v| v.type_()),
        Some(GgufType::Array)
    ));
    assert_eq!(g.get_u32("nemotron_h.moe_latent_size"), Some(32));
}

/// Batch-vs-step consistency of the recurrent state (hybrid_e2e.rs's
/// `granite_state_vs_full_recompute`, on the synthetic files): one prefill
/// batch over N tokens must agree with N single-token decodes — the conv/ssm
/// state cells carry exactly the same information (this is what catches a
/// conv-state write-back that mishandles T > 1).
///
/// Unlike the granite Q4_K probe, the synthetic F32 files cannot be
/// bit-identical end-to-end: the T == 1 decodes route the projections through
/// the GEMV kernel while the batch uses the GEMM one, and the last bits move
/// (the documented band of `context.rs`'s multi-seq test — the same
/// asymmetry the reference's own single-vs-batched decode has). The MoE
/// variants amplify it further (the router's sigmoid/softmax of logits whose
/// last bits moved), so the bound matches the existing hybrid family's
/// `lfm2_state_vs_full_recompute` band (5e-3) — still orders below any
/// semantic error.
#[test]
fn mamba_state_vs_full_recompute() {
    // leak a 'static suffix (the spec field is &'static str; the test runs
    // once so the leak is bounded to six short strings)
    let leak_str = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    for spec in all_specs() {
        // private file suffix: the parallel loader tests rewrite the shared
        // synth paths, and `load_synth` re-creates the file on every call —
        // without this the open can race a concurrent truncating write
        // ("Truncated(magic)" flake)
        let spec = spec.with(|s| s.suffix = leak_str(format!("{}-state", s.suffix)));
        let toks = [3i32, 17, 42, 5, 9, 21];
        // (a) one batch over the whole prefix
        let mut m = load_synth(&spec);
        let (model, mut hb) = driver_for(&mut m, false);
        let full = hb
            .decode(&model, &toks, &(0..toks.len() as i32).collect::<Vec<_>>())
            .to_vec();

        // (b) token by token (the recurrent state must carry the same info)
        let mut m2 = load_synth(&spec);
        let (model2, mut hs) = driver_for(&mut m2, false);
        let mut step = Vec::new();
        for (i, &t) in toks.iter().enumerate() {
            step = hs.decode(&model2, &[t], &[i as i32]).to_vec();
        }

        let d = full
            .iter()
            .zip(&step)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{}: state-vs-batch max abs logit diff {d:.3e}", spec.arch);
        assert_eq!(
            argmax(&full),
            argmax(&step),
            "{}: argmax must agree",
            spec.arch
        );
        assert!(d < 5e-3, "{}: state carry diverged: {d}", spec.arch);
    }
}

// ---------------------------------------------------------------------------
// generator + the parity CLI driver (both #[ignore])
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~250 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch5_write_synth() {
    for spec in all_specs() {
        let (n, bytes) = build_file(&spec);
        println!(
            "{:>13}: {:4} tensors, {:>10} bytes -> {}",
            spec.arch,
            n,
            bytes,
            spec.path()
        );
    }
    let gguf = Gguf::open(spec_mamba().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH5=1 ./parity/arch_batch_parity.sh mamba mamba2 jamba \
         jamba-moe nemotron-h nemotron-h-moe"
    );
}

/// The port side of the parity runs — env-driven greedy decode printing
/// llama-cli's `LLAMA_RUST_DEBUG` format (`step N: top5 [...] greedy=` /
/// `gen tokens: [...]`), so `parity/arch_batch_cmp.py` works unchanged.
///
/// Env: ARCH5_MODEL (gguf path), ARCH5_FA (on|off), ARCH5_PROMPT,
/// ARCH5_N (default 48 — the recurrence must be exercised well past the
/// prompt). Driven by `parity/arch_batch_parity.sh` with ARCH_BATCH5=1.
#[test]
#[ignore = "manual: the parity driver — env ARCH5_MODEL/ARCH5_FA/ARCH5_PROMPT/ARCH5_N"]
fn arch5_cli_driver() {
    let path = std::env::var("ARCH5_MODEL").expect("ARCH5_MODEL");
    let fa = match std::env::var("ARCH5_FA").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        other => panic!("ARCH5_FA must be on|off, got {other:?}"),
    };
    let prompt =
        std::env::var("ARCH5_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let n_predict: usize = std::env::var("ARCH5_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);

    let gguf = Gguf::open(&path).expect("open model");
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize(&prompt, true, true);
    println!("prompt tokens ({}) {:?}", ids.len(), ids);

    let mut m = open_model(&path);
    let arch_name = m.arch.name().to_string();
    let n_tensors = m.tensors.len();
    let (model, mut d) = driver_for(&mut m, fa);
    println!("arch = {arch_name}, {n_tensors} tensors, fa = {fa}");

    let dump_step = |step: usize, lg: &[f32]| {
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]));
        let mut gm = 0usize;
        for (i, &v) in lg.iter().enumerate() {
            if v > lg[gm] {
                gm = i;
            }
        }
        let max = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse: f64 = lg
            .iter()
            .map(|&v| (v - max) as f64)
            .map(f64::exp)
            .sum::<f64>()
            .ln()
            + max as f64;
        println!(
            "step {step}: top5 {:?} greedy={}",
            idx[..5]
                .iter()
                .map(|&i| (i as i32, lg[i], lg[i] as f64 - lse))
                .collect::<Vec<_>>(),
            gm,
        );
    };

    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    let mut logits = d.decode(&model, &ids, &pos);
    let mut cur_pos = ids.len() as i32;
    let mut out_tokens: Vec<i32> = Vec::with_capacity(n_predict);
    for step in 0..n_predict {
        dump_step(step, &logits);
        let tok = argmax(&logits);
        out_tokens.push(tok);
        if vocab.is_eog(tok) {
            eprintln!(" [end of text]");
            break;
        }
        logits = d.decode(&model, &[tok], &[cur_pos]);
        cur_pos += 1;
    }
    println!("gen tokens: {:?}", out_tokens);
}
