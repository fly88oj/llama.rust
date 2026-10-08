//! arch_batch4_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-28: **the MoE family** — qwen2moe / qwen3moe / granite-moe /
//! phimoe / arctic / olmoe / ernie4-5-moe — plus the dense archs that fell
//! out of it, **smollm3 / seed-oss / openelm** (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-3 (`crates/llama/tests/arch_batch{,2,3}_e2e.rs`,
//! PARITY.md): no local GGUF of most of these archs exists, so each is
//! verified on a *synthetic* file built with the port's byte-exact GGUF
//! writer — `tokenizer.*` KV copied verbatim from the llama SPM vocab
//! fixture, the arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32. Every MoE file carries
//! `n_expert = 4`, `n_expert_used = 2` so `ggml_argsort_top_k` /
//! `ggml_mul_mat_id` routing (and the arch's top-k normalization: qwen2moe
//! and olmoe `norm_w = false`, the rest `true`) is actually exercised.
//!
//! Default-run tests (no reference needed): one per arch — write the file,
//! load it, pin the created tensor set / hparams / the arch's special values
//! (router shapes, ernie's dense lead, smollm3's nope pattern, openelm's
//! per-layer heads), and run a 3-token decode through the arch builder with
//! a bit-identical repeat (both -fa off and -fa on).
//!
//! `#[ignore]`d:
//!   * `arch_batch4_write_synth` — writes the files into /tmp/arch-batch4/
//!   * `arch4_cli_driver` — the port side of the parity runs: env-driven
//!     greedy decode that prints llama-cli's `step N: top5 [...] greedy=` /
//!     `gen tokens: [...]` lines, so `parity/arch_batch_cmp.py` compares it
//!     against the fresh-reference capture unchanged (batch 3's
//!     `arch3_cli_driver` pattern — the ForwardWeights arms live in
//!     context.rs, owned by the integrator).

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::graph::{AttnParams, ForwardResult};
use llama::graph_arch::{self, RecurrentState};
use llama::kv_cache::{KvCache, SlotInfo};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch4";

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

/// every arch of the batch is RMS-norm; the shared MoE sizing keeps files
/// small while the routing stays live
const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    n_ff: i64,
    n_ctx: u32,
    /// false → the loader ties output.weight to token_embd.weight
    write_output: bool,
    /// write `%s.expert_feed_forward_length` (scalar; broadcast per layer)
    n_ff_exp: Option<i64>,
    /// write `%s.expert_shared_feed_forward_length`
    n_ff_shexp: Option<i64>,
    /// ernie: `exp_probs_b` per MoE layer (the DeepSeek-V3 router bias)
    exp_probs_b: bool,
    /// ernie: `interleave_moe_layer_step` (required) / `leading_dense_block_count`
    ernie_step: Option<u32>,
    ernie_dense_lead: Option<u32>,
    /// granite-moe's scale quartet (logit_scale required)
    logit_scale: Option<f32>,
    residual_scale: Option<f32>,
    embedding_scale: Option<f32>,
    attention_scale: Option<f32>,
    /// openelm: per-layer (head, head_kv, n_ff) arrays
    openelm_geom: Option<Vec<(i64, i64, i64)>>,
}

impl SynthSpec {
    fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }
    fn n_embd_kv(&self) -> i64 {
        self.n_head_kv * self.n_embd_head()
    }
    /// the expert intermediate dim each arch's loader derives
    fn n_ff_exp_of(&self) -> i64 {
        if let Some(v) = self.n_ff_exp {
            v
        } else {
            self.n_ff / N_EXPERT_USED
        }
    }
    fn n_ff_shexp_of(&self) -> i64 {
        self.n_ff_shexp.unwrap_or(self.n_ff)
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
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 96,
        n_ctx: 256,
        write_output: true,
        n_ff_exp: Some(32),
        n_ff_shexp: None,
        exp_probs_b: false,
        ernie_step: None,
        ernie_dense_lead: None,
        logit_scale: None,
        residual_scale: None,
        embedding_scale: None,
        attention_scale: None,
        openelm_geom: None,
    }
}

/// qwen2moe — every layer MoE + the sigmoid-gated shared expert
/// (qwen2moe.cpp:51-57)
fn spec_qwen2moe() -> SynthSpec {
    base("qwen2moe").with(|s| s.n_ff_shexp = Some(24))
}

/// qwen3moe — per-head q/k norms, no shared expert, tied head
/// (qwen3moe.cpp:21-25)
fn spec_qwen3moe() -> SynthSpec {
    base("qwen3moe").with(|s| s.write_output = false)
}

/// granite-moe — logit_scale required, MoE experts at the *dense* n_ff
/// (granite-moe.cpp:66-69), shared expert via n_ff_shexp
fn spec_granite_moe() -> SynthSpec {
    base("granitemoe").with(|s| {
        s.write_output = false;
        s.n_ff_exp = None; // no expert_feed_forward_length key
        s.n_ff_shexp = Some(24);
        s.logit_scale = Some(6.0);
        s.residual_scale = Some(0.22);
        s.embedding_scale = Some(4.0);
        s.attention_scale = Some(0.125);
    })
}

/// phimoe — biased RMS norms + separate q/k/v + the phi3 attention scale
/// convention (Q pre-scaled, softmax scale 1.0)
fn spec_phimoe() -> SynthSpec {
    base("phimoe").with(|s| s.n_ff_exp = None) // experts at the dense n_ff (phimoe.cpp:39-41)
}

/// arctic — the double FFN: dense square SwiGLU + MoE over the raw input
fn spec_arctic() -> SynthSpec {
    base("arctic").with(|s| {
        s.write_output = false;
        s.n_ff_exp = None; // experts at the dense n_ff (arctic.cpp:46-48)
    })
}

/// olmoe — full-width q/k norms, no weight normalization. The [n_embd]
/// k-norm weight only lines up with the K projection when n_embd_k_gqa ==
/// n_embd (the real OlmoE-1B is MHA, n_head_kv == n_head) — a GQA file would
/// trip ggml_mul's shape check in the reference too, so the synthetic file
/// keeps n_head_kv == n_head.
fn spec_olmoe() -> SynthSpec {
    base("olmoe").with(|s| {
        s.n_head_kv = s.n_head;
        s.n_ff_exp = None; // experts at the dense n_ff (olmoe.cpp:43-45)
    })
}

/// ernie4-5-moe — one dense lead layer, one MoE layer without the shared
/// expert, one MoE layer with it plus the router bias: every branch of
/// ernie4-5-moe.cpp:63-109 in one file
fn spec_ernie45moe() -> SynthSpec {
    base("ernie4_5-moe").with(|s| {
        s.n_layer = 3;
        s.write_output = false;
        s.n_ff_shexp = Some(24);
        s.exp_probs_b = true;
        s.ernie_step = Some(1);
        s.ernie_dense_lead = Some(1);
    })
}

/// smollm3 — 4 layers so the nope pattern (layer 3 skips rope) is live
/// ernie4-5-moe without the shared expert — the `else` branch of
/// ernie4-5-moe.cpp:105-107 (cur = moe_out), still with the router bias
fn spec_ernie45moe_nosh() -> SynthSpec {
    spec_ernie45moe().with(|s| {
        s.suffix = "-nosh";
        s.n_ff_shexp = None;
    })
}

fn spec_smollm3() -> SynthSpec {
    base("smollm3").with(|s| {
        s.n_layer = 4;
        s.write_output = false;
    })
}

/// seed-oss — attn_post_norm doubles as the FFN norm, tied head
fn spec_seed_oss() -> SynthSpec {
    base("seed_oss").with(|s| s.write_output = false)
}

/// openelm — per-layer head counts / FFN widths from GGUF arrays
/// (openelm.cpp:26-28)
fn spec_openelm() -> SynthSpec {
    base("openelm").with(|s| {
        s.n_layer = 3;
        s.write_output = false;
        // spec.n_head mirrors layer 0 so pin_hparams' generic checks hold
        // (the scalar head_count KV is overwritten by the array below)
        s.n_head = 8;
        s.n_head_kv = 4;
        s.openelm_geom = Some(vec![(8, 4, 96), (4, 4, 64), (4, 2, 96)]);
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_qwen2moe(),
        spec_qwen3moe(),
        spec_granite_moe(),
        spec_phimoe(),
        spec_arctic(),
        spec_olmoe(),
        spec_ernie45moe(),
        spec_ernie45moe_nosh(),
        spec_smollm3(),
        spec_seed_oss(),
        spec_openelm(),
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
    /// routers and experts: 1/sqrt(n_embd)-scaled random weights so the
    /// softmax over the router logits stays spread out
    Router,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let n_kv = spec.n_embd_kv();
    let n_exp = N_EXPERT;
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    // every arch shares the RMS eps key; qwen2moe/olmoe write the head,
    // qwen3moe/granite-moe/arctic/ernie/smollm3/seed-oss/openelm tie it
    match spec.arch {
        // ---- qwen2moe.cpp:19-57 ----
        "qwen2moe" => {
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
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                let n_ff_exp = spec.n_ff_exp_of();
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff_exp, n_exp],
                    Role::Router
                );
                let n_sh = spec.n_ff_shexp_of();
                push!(
                    format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                    vec![n_embd],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_shexp.weight"),
                    vec![n_embd, n_sh],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down_shexp.weight"),
                    vec![n_sh, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up_shexp.weight"),
                    vec![n_embd, n_sh],
                    Role::Proj
                );
            }
        }

        // ---- qwen3moe.cpp:17-55 ----
        "qwen3moe" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
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
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![spec.n_embd_head()],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![spec.n_embd_head()],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                let n_ff_exp = spec.n_ff_exp_of();
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff_exp, n_exp],
                    Role::Router
                );
            }
        }

        // ---- granite-moe.cpp:24-77 (arch name "granitemoe", llama-arch.cpp:104) (arch name "granitemoe", llama-arch.cpp:104) (same set as granite.cpp's MoE branch;
        // experts at the dense n_ff, shared expert at n_ff_shexp) ----
        "granitemoe" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
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
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
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

        // ---- phimoe.cpp:17-45 ----
        "phimoe" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            push!("output.bias", vec![N_VOCAB], Role::Bias);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Bias);
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
                push!(
                    format!("blk.{i}.attn_output.bias"),
                    vec![n_embd],
                    Role::Bias
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Bias);
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
            }
        }

        // ---- arctic.cpp:19-49 ----
        "arctic" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
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
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                // the dense FFN is square (arctic.cpp:40-42)
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_norm_exps.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
            }
        }

        // ---- olmoe.cpp:15-46 ----
        "olmoe" => {
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
                // full-width norms (olmoe.cpp:28-29)
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![n_embd],
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
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff, n_exp],
                    Role::Router
                );
            }
        }

        // ---- ernie4-5.cpp:26-69 (the ERNIE4_5_MOE branch; arch name
        // "ernie4_5-moe", llama-arch.cpp:120) ----
        "ernie4_5-moe" => {
            let dense_lead = spec.ernie_dense_lead.unwrap_or(0) as usize;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
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
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if i >= dense_lead {
                    let n_ff_exp = spec.n_ff_exp_of();
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
                        vec![n_embd, n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![n_ff_exp, n_embd, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, n_ff_exp, n_exp],
                        Role::Router
                    );
                    if let Some(n_sh) = spec.n_ff_shexp {
                        push!(
                            format!("blk.{i}.ffn_gate_shexp.weight"),
                            vec![n_embd, n_sh],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.ffn_down_shexp.weight"),
                            vec![n_sh, n_embd],
                            Role::Proj
                        );
                        push!(
                            format!("blk.{i}.ffn_up_shexp.weight"),
                            vec![n_embd, n_sh],
                            Role::Proj
                        );
                    }
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

        // ---- smollm3.cpp:16-39 ----
        "smollm3" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
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
            }
        }

        // ---- seed-oss.cpp:19-42 (arch name "seed_oss", llama-arch.cpp:134) ----
        "seed_oss" => {
            let head_dim = spec.n_embd_head();
            let n_qo = spec.n_head * head_dim;
            let n_kv_dim = spec.n_head_kv * head_dim;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, n_qo],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k.weight"),
                    vec![n_embd, n_kv_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, n_kv_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_qo, n_embd],
                    Role::Proj
                );
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
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
            }
        }

        // ---- openelm.cpp:18-43 ----
        "openelm" => {
            let geom = spec
                .openelm_geom
                .clone()
                .expect("openelm needs per-layer geometry");
            // head_dim is layer-0 derived: n_embd / n_head(0)
            // (llama-model.cpp:1371)
            let hd = spec.n_embd / geom[0].0;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for (i, (n_head_i, n_head_kv_i, n_ff_i)) in geom.iter().enumerate() {
                let n_head_qkv = 2 * n_head_kv_i + n_head_i;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, n_head_qkv * hd],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_head_i * hd, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, *n_ff_i],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![*n_ff_i, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, *n_ff_i],
                    Role::Proj
                );
            }
        }

        other => panic!("no synthetic tensor table for arch {other}"),
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch3_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch4");

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
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // the MoE keys (granite-moe/phimoe/arctic/olmoe have no expert_ffl key —
    // their loaders use the dense n_ff for the experts)
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(
        format!("{a}.expert_used_count"),
        Value::U32(N_EXPERT_USED as u32)
    );
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
    if let Some(v) = spec.ernie_step {
        kv!(format!("{a}.interleave_moe_layer_step"), Value::U32(v));
    }
    if let Some(v) = spec.ernie_dense_lead {
        kv!(format!("{a}.leading_dense_block_count"), Value::U32(v));
    }
    if let Some(v) = spec.logit_scale {
        kv!(format!("{a}.logit_scale"), Value::F32(v));
    }
    if let Some(v) = spec.residual_scale {
        kv!(format!("{a}.residual_scale"), Value::F32(v));
    }
    if let Some(v) = spec.embedding_scale {
        kv!(format!("{a}.embedding_scale"), Value::F32(v));
    }
    if let Some(v) = spec.attention_scale {
        kv!(format!("{a}.attention.scale"), Value::F32(v));
    }
    if let Some(geom) = &spec.openelm_geom {
        // per-layer geometry arrays (openelm.cpp:26-28)
        kv!(
            format!("{a}.attention.head_count"),
            Value::Array(
                GgufType::Uint32,
                geom.iter().map(|g| Value::U32(g.0 as u32)).collect()
            )
        );
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::Array(
                GgufType::Uint32,
                geom.iter().map(|g| Value::U32(g.1 as u32)).collect()
            )
        );
        kv!(
            format!("{a}.feed_forward_length"),
            Value::Array(
                GgufType::Uint32,
                geom.iter().map(|g| Value::U32(g.2 as u32)).collect()
            )
        );
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
    assert_eq!(hp.n_head(0) as i64, spec.n_head);
    assert_eq!(hp.n_head_kv(0) as i64, spec.n_head_kv);
    assert_eq!(hp.n_ff(0) as i64, spec.n_ff);
    assert_eq!(hp.n_ctx_train, spec.n_ctx);
    assert_eq!(hp.n_embd_head_k(0) as i64, spec.n_embd_head());
    assert_eq!(hp.f_norm_rms_eps, 1e-5);
    assert_eq!(hp.n_expert as i64, N_EXPERT);
    assert_eq!(hp.n_expert_used(0) as i64, N_EXPERT_USED);
}

// ---------------------------------------------------------------------------
// the arch dispatch: weights + params assembly (one enum arm per arch)
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

/// Weights + params of one loaded model, dispatching the arch to its builder
/// (the port's `ForwardWeights` enum lives in context.rs, which this batch
/// does not own — the integrator wires the CLI arms afterwards).
enum Batch4Model {
    Qwen2Moe(graph_arch::Qwen2MoeModelWeights, graph_arch::Qwen2MoeParams),
    Qwen3Moe(graph_arch::Qwen3MoeModelWeights, graph_arch::Qwen3MoeParams),
    GraniteMoe(graph_arch::GraniteModelWeights, graph_arch::GraniteParams),
    Phimoe(graph_arch::PhimoeModelWeights, graph_arch::PhimoeParams),
    Arctic(graph_arch::ArcticModelWeights, graph_arch::ArcticParams),
    Olmoe(graph_arch::OlmoeModelWeights, graph_arch::OlmoeParams),
    Ernie45Moe(
        graph_arch::Ernie45MoeModelWeights,
        graph_arch::Ernie45MoeParams,
    ),
    Smollm3(graph_arch::Smollm3ModelWeights, graph_arch::Smollm3Params),
    SeedOss(graph_arch::SeedOssModelWeights, graph_arch::SeedOssParams),
    Openelm(graph_arch::OpenelmModelWeights, graph_arch::OpenelmParams),
}

fn qwen2moe_weights(m: &LlamaModel) -> graph_arch::Qwen2MoeModelWeights {
    graph_arch::Qwen2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Qwen2MoeLayerWeights {
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
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp.unwrap(),
                ffn_gate_shexp: l.ffn_gate_shexp.unwrap(),
                ffn_down_shexp: l.ffn_down_shexp.unwrap(),
                ffn_up_shexp: l.ffn_up_shexp.unwrap(),
            })
            .collect(),
    }
}

fn qwen3moe_weights(m: &LlamaModel) -> graph_arch::Qwen3MoeModelWeights {
    graph_arch::Qwen3MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Qwen3MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

/// granite-moe reuses the granite builder's weight struct — the mamba2 slots
/// stay None (no recurrent layers)
fn granite_moe_weights(m: &LlamaModel) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|x| graph_arch::GraniteLayerWeights {
                attn_norm: x.attn_norm.unwrap(),
                ssm_in: None,
                ssm_conv1d: None,
                ssm_conv1d_b: None,
                ssm_dt_b: None,
                ssm_a: None,
                ssm_d: None,
                ssm_norm: None,
                ssm_out: None,
                wq: x.wq,
                wk: x.wk,
                wv: x.wv,
                wo: x.wo,
                wo_b: x.wo_b,
                rope_freqs: x.rope_freqs,
                ffn_norm: x.ffn_norm.unwrap(),
                ffn_gate: x.ffn_gate,
                ffn_down: x.ffn_down,
                ffn_up: x.ffn_up,
                ffn_gate_b: x.ffn_gate_b,
                ffn_down_b: x.ffn_down_b,
                ffn_up_b: x.ffn_up_b,
                ffn_gate_inp: x.ffn_gate_inp,
                ffn_gate_exps: x.ffn_gate_exps,
                ffn_down_exps: x.ffn_down_exps,
                ffn_up_exps: x.ffn_up_exps,
                ffn_gate_shexp: x.ffn_gate_shexp,
                ffn_down_shexp: x.ffn_down_shexp,
                ffn_up_shexp: x.ffn_up_shexp,
            })
            .collect(),
    }
}

fn phimoe_weights(m: &LlamaModel) -> graph_arch::PhimoeModelWeights {
    graph_arch::PhimoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::PhimoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                // the synthetic files carry no rope factor tensors —
                // `get_rope_factors` returns NULL on both sides
                rope_factors: None,
            })
            .collect(),
    }
}

fn arctic_weights(m: &LlamaModel) -> graph_arch::ArcticModelWeights {
    graph_arch::ArcticModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::ArcticLayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_norm_exps: l.ffn_norm_exps.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn olmoe_weights(m: &LlamaModel) -> graph_arch::OlmoeModelWeights {
    graph_arch::OlmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::OlmoeLayerWeights {
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
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn ernie45moe_weights(m: &LlamaModel) -> graph_arch::Ernie45MoeModelWeights {
    graph_arch::Ernie45MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Ernie45MoeLayerWeights {
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

fn smollm3_weights(m: &LlamaModel) -> graph_arch::Smollm3ModelWeights {
    graph_arch::Smollm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Smollm3LayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn seed_oss_weights(m: &LlamaModel) -> graph_arch::SeedOssModelWeights {
    graph_arch::SeedOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::SeedOssLayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn openelm_weights(m: &LlamaModel) -> graph_arch::OpenelmModelWeights {
    graph_arch::OpenelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OpenelmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                n_head: m.hparams.n_head(il) as i64,
                n_head_kv: m.hparams.n_head_kv(il) as i64,
                n_ff: m.hparams.n_ff(il) as i64,
            })
            .collect(),
    }
}

impl Batch4Model {
    /// Assemble the arch's weights + params off a loaded model.
    fn assemble(m: &LlamaModel, fa: bool) -> Self {
        let hp = &m.hparams;
        use llama::arch::LlmArch::*;
        match m.arch {
            QWEN2MOE => Batch4Model::Qwen2Moe(
                qwen2moe_weights(m),
                graph_arch::Qwen2MoeParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            QWEN3MOE => Batch4Model::Qwen3Moe(
                qwen3moe_weights(m),
                graph_arch::Qwen3MoeParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            GRANITE_MOE => Batch4Model::GraniteMoe(
                granite_moe_weights(m),
                graph_arch::GraniteParams::dense(
                    synth_attn(m, fa),
                    hp,
                    hp.f_logit_scale,
                    hp.f_residual_scale,
                    hp.f_embedding_scale,
                    hp.f_attention_scale,
                ),
            ),
            PHIMOE => Batch4Model::Phimoe(
                phimoe_weights(m),
                graph_arch::PhimoeParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            ARCTIC => Batch4Model::Arctic(
                arctic_weights(m),
                graph_arch::ArcticParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            OLMOE => Batch4Model::Olmoe(
                olmoe_weights(m),
                graph_arch::OlmoeParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            ERNIE4_5_MOE => Batch4Model::Ernie45Moe(
                ernie45moe_weights(m),
                graph_arch::Ernie45MoeParams {
                    attn: synth_attn(m, fa),
                    n_expert: N_EXPERT,
                    n_expert_used: N_EXPERT_USED,
                    n_moe_layer_step: hp.n_moe_layer_step,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_ff_shexp: hp.n_ff_shexp as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            SMOLLM3 => Batch4Model::Smollm3(
                smollm3_weights(m),
                graph_arch::Smollm3Params {
                    attn: synth_attn(m, fa),
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    f_attention_scale: hp.f_attention_scale,
                },
            ),
            SEED_OSS => Batch4Model::SeedOss(
                seed_oss_weights(m),
                graph_arch::SeedOssParams {
                    attn: synth_attn(m, fa),
                    f_attention_scale: hp.f_attention_scale,
                },
            ),
            OPENELM => Batch4Model::Openelm(
                openelm_weights(m),
                graph_arch::OpenelmParams {
                    attn: synth_attn(m, fa),
                },
            ),
            other => panic!("batch4: arch {:?} not wired", other),
        }
    }

    fn n_layer(&self) -> usize {
        match self {
            Batch4Model::Qwen2Moe(w, _) => w.layers.len(),
            Batch4Model::Qwen3Moe(w, _) => w.layers.len(),
            Batch4Model::GraniteMoe(w, _) => w.layers.len(),
            Batch4Model::Phimoe(w, _) => w.layers.len(),
            Batch4Model::Arctic(w, _) => w.layers.len(),
            Batch4Model::Olmoe(w, _) => w.layers.len(),
            Batch4Model::Ernie45Moe(w, _) => w.layers.len(),
            Batch4Model::Smollm3(w, _) => w.layers.len(),
            Batch4Model::SeedOss(w, _) => w.layers.len(),
            Batch4Model::Openelm(w, _) => w.layers.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        ctx: &mut Context,
        st: &RecurrentState,
        kv: &KvCache,
        inp: &llama::graph::DecodeInputs,
        sinfo: SlotInfo,
        n_kv: u32,
        n_tokens: usize,
    ) -> ForwardResult {
        match self {
            Batch4Model::Qwen2Moe(w, p) => {
                graph_arch::build_qwen2moe_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Qwen3Moe(w, p) => {
                graph_arch::build_qwen3moe_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::GraniteMoe(w, p) => {
                graph_arch::build_granite_forward(ctx, w, p, st, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Phimoe(w, p) => {
                graph_arch::build_phimoe_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Arctic(w, p) => {
                graph_arch::build_arctic_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Olmoe(w, p) => {
                graph_arch::build_olmoe_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Ernie45Moe(w, p) => {
                graph_arch::build_ernie45_moe_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Smollm3(w, p) => {
                graph_arch::build_smollm3_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::SeedOss(w, p) => {
                graph_arch::build_seed_oss_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch4Model::Openelm(w, p) => {
                graph_arch::build_openelm_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the decode harness (arch_batch3_e2e.rs's Driver, without the alibi switch —
// every batch-4 arch is non-alibi — but with the granite recurrent state)
// ---------------------------------------------------------------------------

struct Driver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    fa: bool,
    /// granite-moe's (empty) recurrent state, allocated below the watermark
    rstate: RecurrentState,
    /// MUL_MAT_ID nodes of the last decode's graph — the structural pin that
    /// the MoE routing (up/gate/down experts, one mul_mat_id each) is live
    last_n_mul_mat_id: usize,
}

fn driver_for(m: &mut LlamaModel, fa: bool) -> (Batch4Model, Driver) {
    let model = Batch4Model::assemble(m, fa);
    let n_layer = model.n_layer();
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    // llama-kv-cache.cpp:210-211 allocates n_embd_k_gqa(il) rows per layer —
    // uniform for every batch-4 arch but openelm, whose head counts vary
    let k_row: Vec<i64> = (0..n_layer).map(|il| m.n_embd_k_gqa(il) as i64).collect();
    let v_row: Vec<i64> = (0..n_layer).map(|il| m.n_embd_v_gqa(il) as i64).collect();
    let kv = if k_row.iter().all(|&k| k == k_row[0]) && v_row.iter().all(|&v| v == v_row[0]) {
        KvCache::new(&mut gctx, n_layer, k_row[0], v_row[0], 512)
    } else {
        KvCache::new_with_dims(&mut gctx, &k_row, &v_row, 512)
    };
    // no recurrent layers anywhere in the batch — the all-false state is
    // exactly the C `n_rs_seq == 0` configuration
    let is_recr = vec![false; n_layer];
    let rstate = RecurrentState::new(&mut gctx, &is_recr, 0, 0);
    let watermark = gctx.mark();
    (
        model,
        Driver {
            gctx,
            kv,
            watermark,
            fa,
            rstate,
            last_n_mul_mat_id: 0,
        },
    )
}

impl Driver {
    /// Decode `tokens` at `pos`; returns the *last* token's logits [n_vocab].
    fn decode(&mut self, model: &Batch4Model, tokens: &[i32], pos: &[i32]) -> Vec<f32> {
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
            // causal fill (batch-4 archs have no max_alibi_bias). Padded
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
        let inputs = llama::graph::DecodeInputs {
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
        self.last_n_mul_mat_id = gf
            .nodes
            .iter()
            .filter(|&&n| self.gctx.op(n) == ggml::tensor::GgmlOp::MulMatId)
            .count();
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

    d.kv.clear();
    let c = d.decode(&model, &toks, &pos);
    assert_eq!(a, c, "{}: prefill not deterministic (fa={fa})", spec.arch);
    a
}

/// The MoE routing must be structurally live: a prefill through the arch's
/// builder must contain exactly 3 MUL_MAT_ID nodes per MoE layer (up / gate /
/// down experts — llama-graph.cpp:2168-2235) plus the ARGSORT_TOP_K selection.
fn moe_routing_is_live(spec: &SynthSpec, n_moe_layers: usize) {
    let mut m = load_synth(spec);
    let (model, mut d) = driver_for(&mut m, false);
    d.decode(&model, &[3, 17, 42], &[0, 1, 2]);
    assert_eq!(
        d.last_n_mul_mat_id,
        3 * n_moe_layers,
        "{}: expected 3 mul_mat_id per MoE layer",
        spec.arch
    );
}

// ---------------------------------------------------------------------------
// default-run tests, one per arch
// ---------------------------------------------------------------------------

#[test]
fn synth_qwen2moe_loader_and_forward() {
    let spec = spec_qwen2moe();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 2 * 14, "qwen2moe tensor count");
    println!(
        "qwen2moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // qwen2moe.cpp:45 — n_ff_exp comes from the expert_feed_forward_length key
    assert_eq!(m.hparams.n_ff_exp(0) as i64, 32);
    // qwen2moe.cpp:52 — n_ff_shexp from expert_shared_feed_forward_length
    assert_eq!(m.hparams.n_ff_shexp, 24);
    assert_ne!(m.output, m.tok_embd, "qwen2moe head is required");
    assert!(m.layers[0].ffn_gate_inp_shexp.is_some());
    let l0 = &m.layers[0];
    assert!(
        l0.ffn_gate_shexp.is_some() && l0.ffn_up_shexp.is_some() && l0.ffn_down_shexp.is_some()
    );
    // router + experts: [n_embd, n_expert] / 3D [., ., n_expert]
    assert_eq!(m.ctx.ne(l0.ffn_gate_inp.unwrap())[1] as i64, N_EXPERT);
    assert_eq!(m.ctx.ne(l0.ffn_gate_exps.unwrap())[2] as i64, N_EXPERT);
    assert_eq!(*m.ctx.ne(l0.ffn_down_exps.unwrap()), [32, 64, N_EXPERT, 1]);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "qwen2moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_qwen3moe_loader_and_forward() {
    let spec = spec_qwen3moe();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 12, "qwen3moe tensor count (tied head)");
    println!(
        "qwen3moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    assert_eq!(m.output, m.tok_embd, "qwen3moe ties the head");
    assert!(m.layers[0].attn_q_norm.is_some() && m.layers[0].attn_k_norm.is_some());
    assert_eq!(
        m.ctx.ne(m.layers[0].attn_q_norm.unwrap())[0] as i64,
        spec.n_embd_head()
    );
    assert!(m.layers[0].ffn_gate_exps.is_some() && m.layers[0].ffn_up_exps.is_some());
    // no shared expert anywhere
    assert!(m.layers.iter().all(|l| l.ffn_gate_shexp.is_none()));
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "qwen3moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_granite_moe_loader_and_forward() {
    let spec = spec_granite_moe();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 13, "granite-moe tensor count (tied head)");
    println!(
        "granite-moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // granite-moe.cpp:5-8 — logit_scale required, the rest optional
    assert_eq!(m.hparams.f_logit_scale, 6.0);
    assert_eq!(m.hparams.f_residual_scale, 0.22);
    assert_eq!(m.hparams.f_embedding_scale, 4.0);
    assert_eq!(m.hparams.f_attention_scale, 0.125);
    assert_eq!(m.output, m.tok_embd);
    // granite-moe.cpp:67-69 — experts at the DENSE n_ff (no expert_ffl key)
    assert_eq!(
        *m.ctx.ne(m.layers[0].ffn_down_exps.unwrap()),
        [96, 64, N_EXPERT, 1]
    );
    assert_eq!(m.hparams.n_ff_shexp, 24);
    // granite-moe does not read rope_finetuned — rope_pattern stays all-1
    assert!(m.hparams.has_rope(0) && m.hparams.has_rope(1));

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "granite-moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_phimoe_loader_and_forward() {
    let spec = spec_phimoe();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 5 + 2 * 13, "phimoe tensor count (biases everywhere)");
    println!(
        "phimoe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // phimoe.cpp:20-23 — the bias quartet is required
    assert!(m.output_norm_b.is_some() && m.output_b.is_some());
    let l0 = &m.layers[0];
    assert!(l0.attn_norm_b.is_some() && l0.ffn_norm_b.is_some() && l0.wo_b.is_some());
    assert_ne!(m.output, m.tok_embd);
    // rope factor tensors stay optional-and-absent
    assert!(l0.rope_long.is_none() && l0.rope_short.is_none());
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "phimoe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_arctic_loader_and_forward() {
    let spec = spec_arctic();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 14, "arctic tensor count (tied head)");
    println!(
        "arctic synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    let l0 = &m.layers[0];
    // arctic.cpp:40-42 — the dense FFN is square
    assert_eq!(*m.ctx.ne(l0.ffn_gate.unwrap()), [64, 64, 1, 1]);
    assert!(
        l0.ffn_norm_exps.is_some(),
        "arctic needs the second expert norm"
    );
    assert_eq!(m.output, m.tok_embd);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NORM);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "arctic synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_olmoe_loader_and_forward() {
    let spec = spec_olmoe();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 2 * 12, "olmoe tensor count");
    println!("olmoe synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    let l0 = &m.layers[0];
    // olmoe.cpp:28-29 — full-width norms
    assert_eq!(m.ctx.ne(l0.attn_q_norm.unwrap())[0] as i64, spec.n_embd);
    assert_eq!(m.ctx.ne(l0.attn_k_norm.unwrap())[0] as i64, spec.n_embd);
    assert_ne!(m.output, m.tok_embd);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "olmoe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);
}

#[test]
fn synth_ernie45moe_loader_and_forward() {
    let spec = spec_ernie45moe();
    let (n, bytes) = build_file(&spec);
    // layer 0 dense (9 tensors), layers 1-2 MoE (14 each), head tied
    assert_eq!(n, 2 + 9 + 2 * 14, "ernie4-5-moe tensor count");
    println!(
        "ernie4-5-moe synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // meta.rs's ERNIE4_5_MOE arm read the four MoE keys
    assert_eq!(m.hparams.n_moe_layer_step, 1);
    assert_eq!(m.hparams.n_layer_dense_lead, 1);
    assert_eq!(m.hparams.n_ff_shexp, 24);
    assert_eq!(m.hparams.n_ff_exp(1) as i64, 32);
    // the layer split: dense lead, then MoE
    assert!(m.layers[0].ffn_gate.is_some() && m.layers[0].ffn_gate_inp.is_none());
    assert!(m.layers[1].ffn_gate.is_none() && m.layers[1].ffn_gate_inp.is_some());
    assert!(
        m.layers[1].ffn_exp_probs_b.is_some(),
        "the router bias is on every MoE layer"
    );
    assert!(m.layers[2].ffn_exp_probs_b.is_some());
    assert!(m.layers[2].ffn_gate_shexp.is_some());
    assert_eq!(m.output, m.tok_embd);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NORM);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "ernie4-5-moe synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
    moe_routing_is_live(&spec, 2);

    // the no-shared-expert variant (ernie4-5-moe.cpp:105-107): same file
    // minus the shexp trio — the graph's else branch
    let nosh = spec_ernie45moe_nosh();
    let (nn, _) = build_file(&nosh);
    assert_eq!(nn, 2 + 9 + 2 * 11, "ernie4-5-moe-nosh tensor count");
    let mn = load_synth(&nosh);
    pin_tensors(&mn, &nosh);
    pin_hparams(&mn, &nosh);
    assert_eq!(mn.hparams.n_ff_shexp, 0);
    assert!(mn.layers.iter().all(|l| l.ffn_gate_shexp.is_none()));
    assert!(
        mn.layers[1].ffn_exp_probs_b.is_some(),
        "the router bias stays"
    );
    let a = smoke_forward(&nosh, false);
    let b = smoke_forward(&nosh, true);
    println!(
        "ernie4-5-moe-nosh synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_smollm3_loader_and_forward() {
    let spec = spec_smollm3();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 4 * 9, "smollm3 tensor count (tied head)");
    println!(
        "smollm3 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // smollm3.cpp:5 — the nope step is hard-coded 4
    assert_eq!(m.hparams.n_no_rope_layer_step, 4);
    assert_eq!(m.output, m.tok_embd);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NORM);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "smollm3 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_seed_oss_loader_and_forward() {
    let spec = spec_seed_oss();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 9, "seed-oss tensor count (tied head)");
    println!(
        "seed-oss synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // seed-oss has no ffn_norm — post_attention_norm feeds the FFN
    assert!(m.layers.iter().all(|l| l.ffn_norm.is_none()));
    assert!(m.layers[0].attn_post_norm.is_some());
    assert_eq!(m.output, m.tok_embd);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "seed-oss synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_openelm_loader_and_forward() {
    let spec = spec_openelm();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 3 * 9, "openelm tensor count (tied head)");
    println!(
        "openelm synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // per-layer geometry straight off the GGUF arrays (openelm.cpp:26-28)
    assert_eq!(m.hparams.n_head(0), 8);
    assert_eq!(m.hparams.n_head(1), 4);
    assert_eq!(m.hparams.n_head_kv(1), 4);
    assert_eq!(m.hparams.n_ff(1) as i64, 64);
    // head_dim is layer-0 derived: n_embd / n_head(0)
    assert_eq!(m.hparams.n_embd_head_k(0) as i64, spec.n_embd_head());
    // the fused qkv packs [q | k | v] heads: (4 + 4 + 4) * 8 = 96 at layer 1
    assert_eq!(*m.ctx.ne(m.layers[1].wqkv.unwrap()), [64, 96, 1, 1]);
    assert_eq!(*m.ctx.ne(m.layers[1].wo.unwrap()), [32, 64, 1, 1]);
    assert_eq!(m.output, m.tok_embd, "openelm always ties the head");

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "openelm synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_qwen2moe().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("qwen2moe"));
    assert_eq!(g.get_u32("qwen2moe.expert_count"), Some(N_EXPERT as u32));
    assert_eq!(
        g.get_u32("qwen2moe.expert_used_count"),
        Some(N_EXPERT_USED as u32)
    );
    assert_eq!(g.get_u32("qwen2moe.expert_feed_forward_length"), Some(32));
    assert_eq!(
        g.get_u32("qwen2moe.expert_shared_feed_forward_length"),
        Some(24)
    );
    let spec = spec_ernie45moe().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_u32("ernie4_5-moe.interleave_moe_layer_step"), Some(1));
    assert_eq!(g.get_u32("ernie4_5-moe.leading_dense_block_count"), Some(1));
    let spec = spec_openelm().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    // per-layer arrays survive the round-trip
    assert!(matches!(
        g.find_key("openelm.attention.head_count")
            .map(|v| v.type_()),
        Some(GgufType::Array)
    ));
    assert!(matches!(
        g.find_key("tokenizer.ggml.tokens").map(|v| v.type_()),
        Some(GgufType::Array)
    ));
}

// ---------------------------------------------------------------------------
// generator + the parity CLI driver (both #[ignore])
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~180 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch4_write_synth() {
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
    let gguf = Gguf::open(spec_olmoe().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH4=1 ./parity/arch_batch_parity.sh qwen2moe qwen3moe granite-moe \
         phimoe arctic olmoe ernie4-5-moe smollm3 seed-oss openelm"
    );
}

/// The port side of the parity runs — env-driven greedy decode printing
/// llama-cli's `LLAMA_RUST_DEBUG` format (`step N: top5 [...] greedy=` /
/// `gen tokens: [...]`), so `parity/arch_batch_cmp.py` works unchanged.
///
/// Env: ARCH4_MODEL (gguf path), ARCH4_FA (on|off), ARCH4_PROMPT,
/// ARCH4_N (default 16). Driven by `parity/arch_batch_parity.sh` with
/// ARCH_BATCH4=1.
#[test]
#[ignore = "manual: the parity driver — env ARCH4_MODEL/ARCH4_FA/ARCH4_PROMPT/ARCH4_N"]
fn arch4_cli_driver() {
    let path = std::env::var("ARCH4_MODEL").expect("ARCH4_MODEL");
    let fa = match std::env::var("ARCH4_FA").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        other => panic!("ARCH4_FA must be on|off, got {other:?}"),
    };
    let prompt =
        std::env::var("ARCH4_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let n_predict: usize = std::env::var("ARCH4_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

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
