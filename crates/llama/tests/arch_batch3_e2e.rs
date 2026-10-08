//! arch_batch3_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-09-27: **baichuan / bloom / mpt / starcoder / refact / plamo /
//! stablelm / granite(dense) / minicpm** (llama.cpp bd4f514db1), plus the
//! batch's shared mechanism, **ALiBi**.
//!
//! Same protocol as batches 1/2 (`crates/llama/tests/arch_batch{,2}_e2e.rs`,
//! PARITY.md): no local GGUF of any of these archs exists, so each is verified
//! on a *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//! Default-run tests (no reference needed): one per arch — write the file,
//! load it, pin the created tensor set / hparams / the arch's special values
//! (alibi bias, granite scales, stablelm variants), and run a 3-token decode
//! through the arch builder with a bit-identical repeat. The ALiBi mechanism
//! itself is pinned in `graph.rs` (mask values) and `flash_attn.rs` (slopes);
//! `alibi_softmax_matches_naive_slopes` below pins the non-FA softmax against
//! hand-computed slopes end-to-end.
//!
//! `#[ignore]`d:
//!   * `arch_batch3_write_synth` — writes the files into /tmp/arch-batch3/
//!   * `arch3_cli_driver` — the port side of the parity runs: env-driven
//!     greedy decode that prints llama-cli's `step N: top5 [...] greedy=` /
//!     `gen tokens: [...]` lines, so `parity/arch_batch_cmp.py` compares it
//!     against the fresh-reference capture unchanged.

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

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch3";

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

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
    rms_eps: bool,
    write_output: bool,
    /// learned absolute positions (starcoder required / mpt optional)
    pos_embd: bool,
    /// bloom's token_embd_norm pair
    tok_norm: bool,
    /// mpt's optional extras: biases, full-width Q/K norms, act scales
    mpt_biases: bool,
    mpt_qk_norms: bool,
    mpt_act_scales: bool,
    clamp_kqv: Option<f32>,
    /// mpt reads its alibi bias from the GGUF
    kv_alibi: Option<f32>,
    /// stablelm: attn_q_norm/attn_k_norm (12B) and/or ffn_norm (parallel-res
    /// files have neither norm pair)
    stablelm_qk_norms: bool,
    stablelm_ffn_norm: bool,
    /// partial rope (stablelm's rotary_pct): `rope.dimension_count` < head_dim
    rope_dim_count: Option<u32>,
    /// granite family scalars (logit_scale required for granite; the rest
    /// optional); None = key not written
    logit_scale: Option<f32>,
    residual_scale: Option<f32>,
    embedding_scale: Option<f32>,
    attention_scale: Option<f32>,
}

impl SynthSpec {
    fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }
    fn n_embd_kv(&self) -> i64 {
        self.n_head_kv * self.n_embd_head()
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

fn base(arch: &'static str, rms_eps: bool) -> SynthSpec {
    SynthSpec {
        arch,
        suffix: "",
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps,
        write_output: true,
        pos_embd: false,
        tok_norm: false,
        mpt_biases: false,
        mpt_qk_norms: false,
        mpt_act_scales: false,
        clamp_kqv: None,
        kv_alibi: None,
        stablelm_qk_norms: false,
        stablelm_ffn_norm: true,
        rope_dim_count: None,
        logit_scale: None,
        residual_scale: None,
        embedding_scale: None,
        attention_scale: None,
    }
}

/// the 13B: 40 layers ⇒ alibi 8.0, no rope (baichuan.cpp:6-14)
fn spec_baichuan13() -> SynthSpec {
    base("baichuan", true).with(|s| s.n_layer = 40)
}

/// the 7B: 32 layers ⇒ rope, no alibi
fn spec_baichuan7() -> SynthSpec {
    base("baichuan", true).with(|s| {
        s.n_layer = 32;
        s.suffix = "-7b";
    })
}

fn spec_bloom() -> SynthSpec {
    base("bloom", false).with(|s| {
        s.tok_norm = true;
        s.write_output = false; // the tie fallback (bloom.cpp:28-31)
    })
}

fn spec_mpt() -> SynthSpec {
    base("mpt", false).with(|s| {
        s.kv_alibi = Some(8.0); // mpt.cpp:6 — the GGUF key
        s.write_output = false;
    })
}

/// every reachable mpt optional at once: pos_embd, biases, act scales,
/// clamp, the untied head. The full-width Q/K norms are deliberately absent:
/// mpt.cpp:27 requires a *fused* qkv and the Q/K-norm branch then reshapes
/// the non-contiguous fused view (mpt.cpp:88-97), which GGML_ASSERTs in the
/// reference (ggml.c:3729) — the C FIXME at mpt.cpp:43 says as much, so no
/// loadable mpt file can carry them and the port keeps the same failure.
fn spec_mpt_full() -> SynthSpec {
    spec_mpt().with(|s| {
        s.suffix = "-full";
        s.pos_embd = true;
        s.mpt_biases = true;
        s.mpt_act_scales = true;
        // small enough that the clamp actually bites the ±0.125-scale scores
        s.clamp_kqv = Some(0.02);
        s.write_output = true;
    })
}

fn spec_starcoder() -> SynthSpec {
    base("starcoder", false).with(|s| {
        s.pos_embd = true; // starcoder.cpp:20 — required
        s.write_output = false;
    })
}

fn spec_refact() -> SynthSpec {
    base("refact", true).with(|s| s.write_output = false)
}

fn spec_plamo() -> SynthSpec {
    base("plamo", true)
}

/// StableLM 2 12B shape: per-head Q/K norms + parallel residual (no ffn_norm)
fn spec_stablelm_12b() -> SynthSpec {
    base("stablelm", false).with(|s| {
        s.stablelm_qk_norms = true;
        s.stablelm_ffn_norm = false;
    })
}

/// the 3B shape: sequential ffn_norm, no per-head norms, partial rope
fn spec_stablelm_3b() -> SynthSpec {
    base("stablelm", false).with(|s| {
        s.suffix = "-3b";
        s.rope_dim_count = Some(12); // < head_dim 16 (rotary_pct)
    })
}

fn spec_granite() -> SynthSpec {
    base("granite", true).with(|s| {
        s.logit_scale = Some(6.0); // required (granite.cpp:21)
        s.residual_scale = Some(0.5);
        s.embedding_scale = Some(4.0);
        s.attention_scale = Some(0.125);
        s.write_output = false;
    })
}

fn spec_minicpm() -> SynthSpec {
    base("minicpm", true).with(|s| s.write_output = false)
    // no scale keys: the minicpm defaults kick in (minicpm.cpp:5-7)
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_baichuan13(),
        spec_baichuan7(),
        spec_bloom(),
        spec_mpt(),
        spec_mpt_full(),
        spec_starcoder(),
        spec_refact(),
        spec_plamo(),
        spec_stablelm_12b(),
        spec_stablelm_3b(),
        spec_granite(),
        spec_minicpm(),
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
    Pos,
    /// positive, well away from 1 (the act scales divide the gelu output)
    Scalar,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let n_kv = spec.n_embd_kv();
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    match spec.arch {
        // ---- baichuan.cpp:17-45 ----
        "baichuan" => {
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

        // ---- bloom.cpp:21-67 ----
        "bloom" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("token_embd_norm.weight", vec![n_embd], Role::Norm);
            push!("token_embd_norm.bias", vec![n_embd], Role::Bias);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            if spec.write_output {
                push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            }
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Bias);
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, n_embd + 2 * n_kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_qkv.bias"),
                    vec![n_embd + 2 * n_kv],
                    Role::Bias
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
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_up.bias"), vec![n_ff], Role::Bias);
            }
        }

        // ---- mpt.cpp:17-53 ----
        "mpt" => {
            let b = spec.mpt_biases;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            if spec.pos_embd {
                push!(
                    "position_embd.weight",
                    vec![n_embd, spec.n_ctx as i64],
                    Role::Pos
                );
            }
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            if b {
                push!("output_norm.bias", vec![n_embd], Role::Bias);
            }
            if spec.write_output {
                push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            }
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if b {
                    push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Bias);
                }
                if spec.mpt_qk_norms {
                    unreachable!("mpt Q/K norms are not loadable (see spec_mpt_full)")
                } else {
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, n_embd + 2 * n_kv],
                        Role::Proj
                    );
                    if b {
                        push!(
                            format!("blk.{i}.attn_qkv.bias"),
                            vec![n_embd + 2 * n_kv],
                            Role::Bias
                        );
                    }
                }
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                if b {
                    push!(
                        format!("blk.{i}.attn_output.bias"),
                        vec![n_embd],
                        Role::Bias
                    );
                }
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if b {
                    push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Bias);
                }
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                if b {
                    push!(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                }
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                if b {
                    push!(format!("blk.{i}.ffn_up.bias"), vec![n_ff], Role::Bias);
                }
                if spec.mpt_act_scales {
                    // LLM_TENSOR_FFN_ACT's template is "blk.%d.ffn.act"
                    // (llama-arch.cpp) → `blk.N.ffn.act.scales`
                    push!(format!("blk.{i}.ffn.act.scales"), vec![n_ff], Role::Scalar);
                }
            }
        }

        // ---- starcoder.cpp:17-60 ----
        "starcoder" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!(
                "position_embd.weight",
                vec![n_embd, spec.n_ctx as i64],
                Role::Pos
            );
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            if spec.write_output {
                push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            }
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Bias);
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, n_embd + 2 * n_kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_qkv.bias"),
                    vec![n_embd + 2 * n_kv],
                    Role::Bias
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
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_down.bias"), vec![n_embd], Role::Bias);
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_up.bias"), vec![n_ff], Role::Bias);
            }
        }

        // ---- refact.cpp:16-76 (dense) ----
        "refact" => {
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
                if i == 0 {
                    // REPEATING class: one model-level tensor for every layer
                    push!(
                        "rope_freqs.weight",
                        vec![spec.n_embd_head() / 2],
                        Role::Norm
                    );
                }
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

        // ---- plamo.cpp:12-36 ----
        "plamo" => {
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

        // ---- stablelm.cpp:13-58 ----
        "stablelm" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
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
                if spec.stablelm_qk_norms {
                    push!(
                        format!("blk.{i}.attn_q_norm.weight"),
                        vec![spec.n_embd_head(), spec.n_head],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_k_norm.weight"),
                        vec![spec.n_embd_head(), spec.n_head_kv],
                        Role::Norm
                    );
                }
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                if spec.stablelm_ffn_norm {
                    push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                    push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Bias);
                }
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

        // ---- granite.cpp:79-150 + minicpm.cpp:29-96 (same tensor set) ----
        "granite" | "minicpm" => {
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

        other => panic!("no synthetic tensor table for arch {other}"),
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch2_e2e.rs)
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
        Role::Pos => 1.0 / (n_embd as f32).sqrt(),
        Role::Scalar => 2.0,
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch3");

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
    if spec.rms_eps {
        kv!(
            format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-5)
        );
    } else {
        kv!(
            format!("{a}.attention.layer_norm_epsilon"),
            Value::F32(1e-5)
        );
    }
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    if let Some(v) = spec.clamp_kqv {
        kv!(format!("{a}.attention.clamp_kqv"), Value::F32(v));
    }
    if let Some(v) = spec.kv_alibi {
        kv!(format!("{a}.attention.max_alibi_bias"), Value::F32(v));
    }
    if let Some(v) = spec.rope_dim_count {
        kv!(format!("{a}.rope.dimension_count"), Value::U32(v));
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

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = match role {
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            Role::Scalar => (0..n).map(|_| 1.5 + 0.5 * rng.next()).collect(),
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
    if spec.rms_eps {
        assert_eq!(hp.f_norm_rms_eps, 1e-5);
    } else {
        assert_eq!(hp.f_norm_eps, 1e-5);
    }
    assert_eq!(hp.f_clamp_kqv, spec.clamp_kqv.unwrap_or(0.0));
}

// ---------------------------------------------------------------------------
// the arch dispatch: weights + params assembly (one enum arm per arch)
// ---------------------------------------------------------------------------

fn is_rms_arch(m: &LlamaModel) -> bool {
    use llama::arch::LlmArch::*;
    matches!(m.arch, BAICHUAN | REFACT | PLAMO | GRANITE | MINICPM)
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
        norm_eps: if is_rms_arch(m) {
            hp.f_norm_rms_eps
        } else {
            hp.f_norm_eps
        },
        use_flash_attn: fa,
    }
}

/// Weights + params of one loaded model, dispatching the arch to its builder
/// (the port's `ForwardWeights` enum lives in context.rs, which this batch
/// does not own — the integrator wires the CLI arms afterwards).
enum Batch3Model {
    Baichuan(graph_arch::BaichuanModelWeights, graph_arch::BaichuanParams),
    Bloom(graph_arch::BloomModelWeights, graph_arch::BloomParams),
    Mpt(graph_arch::MptModelWeights, graph_arch::MptParams),
    Starcoder(
        graph_arch::StarcoderModelWeights,
        graph_arch::StarcoderParams,
    ),
    Refact(graph_arch::RefactModelWeights, graph_arch::RefactParams),
    Plamo(graph_arch::PlamoModelWeights, graph_arch::PlamoParams),
    Stablelm(graph_arch::StablelmModelWeights, graph_arch::StablelmParams),
    Granite(graph_arch::GraniteModelWeights, graph_arch::GraniteParams),
}

fn baichuan_weights(m: &LlamaModel) -> graph_arch::BaichuanModelWeights {
    graph_arch::BaichuanModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::BaichuanLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn bloom_weights(m: &LlamaModel) -> graph_arch::BloomModelWeights {
    graph_arch::BloomModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: m.token_embd_norm.unwrap(),
        tok_norm_b: m.token_embd_norm_b.unwrap(),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::BloomLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b.unwrap(),
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
            })
            .collect(),
    }
}

fn mpt_weights(m: &LlamaModel) -> graph_arch::MptModelWeights {
    graph_arch::MptModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::MptLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b,
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b,
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b,
                attn_q_norm: l.attn_q_norm,
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm,
                attn_k_norm_b: l.attn_k_norm_b,
                ffn_act: l.ffn_act,
            })
            .collect(),
    }
}

fn starcoder_weights(m: &LlamaModel) -> graph_arch::StarcoderModelWeights {
    graph_arch::StarcoderModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap(),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::StarcoderLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b.unwrap(),
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
            })
            .collect(),
    }
}

fn refact_weights(m: &LlamaModel) -> graph_arch::RefactModelWeights {
    graph_arch::RefactModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::RefactLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                rope_freqs: l.rope_freqs,
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

fn plamo_weights(m: &LlamaModel) -> graph_arch::PlamoModelWeights {
    graph_arch::PlamoModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::PlamoLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn stablelm_weights(m: &LlamaModel) -> graph_arch::StablelmModelWeights {
    graph_arch::StablelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::StablelmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                ffn_norm: l.ffn_norm,
                ffn_norm_b: l.ffn_norm_b,
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn granite_dense_weights(m: &LlamaModel) -> graph_arch::GraniteModelWeights {
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

impl Batch3Model {
    /// Assemble the arch's weights + params off a loaded model. The weights'
    /// TensorIds live in the model's Context, which `driver_for` subsequently
    /// lends to the Driver (the batch-2 `smoke_forward` pattern).
    fn assemble(m: &LlamaModel, fa: bool) -> Self {
        let hp = &m.hparams;
        use llama::arch::LlmArch::*;
        match m.arch {
            BAICHUAN => {
                // baichuan.cpp:4-15: 13B (40 layers) = alibi, 7B (32) = rope
                let use_rope = hp.n_layer() == 32;
                let f_max_alibi_bias = if hp.n_layer() == 40 { 8.0 } else { 0.0 };
                Batch3Model::Baichuan(
                    baichuan_weights(m),
                    graph_arch::BaichuanParams {
                        attn: synth_attn(m, fa),
                        f_max_alibi_bias,
                        use_rope,
                    },
                )
            }
            BLOOM => Batch3Model::Bloom(
                bloom_weights(m),
                graph_arch::BloomParams {
                    attn: synth_attn(m, fa),
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            ),
            MPT => Batch3Model::Mpt(
                mpt_weights(m),
                graph_arch::MptParams {
                    attn: synth_attn(m, fa),
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                    f_clamp_kqv: hp.f_clamp_kqv,
                },
            ),
            STARCODER => Batch3Model::Starcoder(
                starcoder_weights(m),
                graph_arch::StarcoderParams {
                    attn: synth_attn(m, fa),
                },
            ),
            REFACT => Batch3Model::Refact(
                refact_weights(m),
                graph_arch::RefactParams {
                    attn: synth_attn(m, fa),
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            ),
            PLAMO => Batch3Model::Plamo(
                plamo_weights(m),
                graph_arch::PlamoParams {
                    attn: synth_attn(m, fa),
                },
            ),
            STABLELM => Batch3Model::Stablelm(
                stablelm_weights(m),
                graph_arch::StablelmParams {
                    attn: synth_attn(m, fa),
                },
            ),
            GRANITE | MINICPM => Batch3Model::Granite(
                granite_dense_weights(m),
                graph_arch::GraniteParams::dense(
                    synth_attn(m, fa),
                    hp,
                    hp.f_logit_scale,
                    hp.f_residual_scale,
                    hp.f_embedding_scale,
                    hp.f_attention_scale,
                ),
            ),
            other => panic!("batch3: arch {:?} not wired", other),
        }
    }

    fn n_layer(&self) -> usize {
        match self {
            Batch3Model::Baichuan(w, _) => w.layers.len(),
            Batch3Model::Bloom(w, _) => w.layers.len(),
            Batch3Model::Mpt(w, _) => w.layers.len(),
            Batch3Model::Starcoder(w, _) => w.layers.len(),
            Batch3Model::Refact(w, _) => w.layers.len(),
            Batch3Model::Plamo(w, _) => w.layers.len(),
            Batch3Model::Stablelm(w, _) => w.layers.len(),
            Batch3Model::Granite(w, _) => w.layers.len(),
        }
    }

    /// `hparams.f_max_alibi_bias > 0` — the driver picks the mask fill on it
    /// (llama-model.cpp:1419 `use_alibi`)
    fn max_alibi_bias(&self) -> f32 {
        match self {
            Batch3Model::Baichuan(_, p) => p.f_max_alibi_bias,
            Batch3Model::Bloom(_, p) => p.f_max_alibi_bias,
            Batch3Model::Mpt(_, p) => p.f_max_alibi_bias,
            Batch3Model::Refact(_, p) => p.f_max_alibi_bias,
            _ => 0.0,
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
            Batch3Model::Baichuan(w, p) => {
                graph_arch::build_baichuan_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Bloom(w, p) => {
                graph_arch::build_bloom_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Mpt(w, p) => {
                graph_arch::build_mpt_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Starcoder(w, p) => {
                graph_arch::build_starcoder_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Refact(w, p) => {
                graph_arch::build_refact_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Plamo(w, p) => {
                graph_arch::build_plamo_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Stablelm(w, p) => {
                graph_arch::build_stablelm_forward(ctx, w, p, kv, inp, sinfo, n_kv, n_tokens)
            }
            Batch3Model::Granite(w, p) => {
                graph_arch::build_granite_forward(ctx, w, p, st, kv, inp, sinfo, n_kv, n_tokens)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the decode harness (gpt_oss_e2e pattern) with the ALiBi mask switch
// ---------------------------------------------------------------------------

struct Driver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    /// hparams.use_alibi — selects `fill_kq_mask_alibi*` over the causal fill
    alibi: bool,
    fa: bool,
    /// the granite family's (empty on dense) recurrent state, allocated below
    /// the watermark like every other persistent tensor
    rstate: RecurrentState,
    /// debug: the kq_mask of the last decode
    last_mask: Option<ggml::TensorId>,
}

/// Take the model's ggml Context, park the KV cache (and the granite family's
/// empty recurrent state) in it, and pair the assembled weights with a driver.
fn driver_for(m: &mut LlamaModel, fa: bool) -> (Batch3Model, Driver) {
    let model = Batch3Model::assemble(m, fa);
    let driver = driver_with(m, &model, fa);
    (model, driver)
}

/// [`driver_for`] for a pre-assembled model (the alibi A/B test mutates the
/// assembled params before the driver snapshot reads them).
fn driver_with(m: &mut LlamaModel, model: &Batch3Model, fa: bool) -> Driver {
    let n_layer = model.n_layer();
    let n_k = m.n_embd_k_gqa_max() as i64;
    let n_v = m.n_embd_v_gqa_max() as i64;
    let alibi = model.max_alibi_bias() > 0.0;
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
    // dense granite/minicpm has no recurrent layers — an all-false state is
    // exactly the C `n_rs_seq == 0` configuration
    let is_recr = vec![false; n_layer];
    let rstate = RecurrentState::new(&mut gctx, &is_recr, 0, 0);
    let watermark = gctx.mark();
    Driver {
        gctx,
        kv,
        watermark,
        alibi,
        fa,
        rstate,
        last_mask: None,
    }
}

impl Driver {
    /// Decode `tokens` at `pos`; returns the *last* token's logits [n_vocab].
    fn decode(&mut self, model: &Batch3Model, tokens: &[i32], pos: &[i32]) -> Vec<f32> {
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
            // set_input_kq_mask (llama-kv-cache.cpp:1557-1705): the kept value
            // is -|p0-p1| exactly when hparams.use_alibi (:1693-1697); the
            // append-only slot invariant keeps p0 = slot position. Padded
            // (empty) cells keep pos = -1 → masked by the fills.
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            if mask_ty == GgmlType::F16 {
                let mask: &mut [half::f16] = bytemuck::cast_slice_mut(mask_bytes);
                if self.alibi {
                    llama::graph::fill_kq_mask_alibi_f16(
                        mask,
                        &kv_pos,
                        pos,
                        0,
                        llama::hparams::LlamaSwaType::NONE,
                    );
                } else {
                    llama::graph::fill_causal_mask_f16(mask, &kv_pos, pos);
                }
            } else {
                let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
                if self.alibi {
                    llama::graph::fill_kq_mask_alibi(
                        mask,
                        &kv_pos,
                        pos,
                        0,
                        llama::hparams::LlamaSwaType::NONE,
                    );
                } else {
                    llama::graph::fill_causal_mask(mask, &kv_pos, pos);
                }
            }
        }
        let inputs = llama::graph::DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        self.last_mask = Some(kq_mask);

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
/// bit-identical repeat on a cleared cache (the batch-2 `smoke_forward`
/// contract, driven through the batch's own harness + arch dispatch).
fn smoke_forward(spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut m = load_synth(spec);
    let (model, mut d) = driver_for(&mut m, fa);

    let toks = [3i32, 17, 42];
    let pos: Vec<i32> = (0..3).collect();
    let a = d.decode(&model, &toks, &pos);
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{}: non-finite logits",
        spec.arch
    );
    let spread = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - a.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}: logits degenerate (spread {spread})",
        spec.arch
    );

    let next = argmax(&a);
    let b = d.decode(&model, &[next], &[3]);
    assert!(b.iter().all(|v| v.is_finite()));

    d.kv.clear();
    let c = d.decode(&model, &toks, &pos);
    assert_eq!(a, c, "{}: prefill not deterministic", spec.arch);
    a
}

// ---------------------------------------------------------------------------
// default-run tests, one per arch
// ---------------------------------------------------------------------------

#[test]
fn synth_baichuan_loader_and_forward() {
    let spec = spec_baichuan13();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 40 * 9, "baichuan-13B tensor count");
    println!(
        "baichuan-13B synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // baichuan.cpp:12-14 — 13B ⇒ alibi 8.0 and use_alibi; rope type NORM
    // (the llama-model.cpp:2933 "normal RoPE" group — baichuan 7B's rope is
    // GPT-J-style, not NEOX)
    assert_eq!(m.hparams.f_max_alibi_bias, 8.0);
    assert!(m.hparams.use_alibi, "13B must set use_alibi");
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NORM);
    assert_ne!(m.output, m.tok_embd, "baichuan head is required");

    // the 7B: 32 layers ⇒ rope, NO alibi
    let spec7 = spec_baichuan7();
    let (n7, _) = build_file(&spec7);
    assert_eq!(n7, 3 + 32 * 9);
    let m7 = load_synth(&spec7);
    pin_tensors(&m7, &spec7);
    assert_eq!(m7.hparams.f_max_alibi_bias, 0.0);
    assert!(!m7.hparams.use_alibi, "7B must not set use_alibi");

    // the alibi mechanism is live: the same 13B file with the bias zeroed
    // (and the plain causal mask) must produce different logits
    let mut m13 = open_model(&spec.path());
    let (model_a, mut d) = driver_for(&mut m13, false);
    let lg_alibi = d.decode(&model_a, &[3, 17, 42], &[0, 1, 2]);
    let mut m13b = open_model(&spec.path());
    // zero the *assembled* params — the baichuan arm derives the bias from
    // n_layer, so mutating hparams alone would be a no-op here
    let model_b = {
        let mb = Batch3Model::assemble(&m13b, false);
        match mb {
            Batch3Model::Baichuan(w, mut p) => {
                p.f_max_alibi_bias = 0.0;
                Batch3Model::Baichuan(w, p)
            }
            other => other,
        }
    };
    let mut d2 = driver_with(&mut m13b, &model_b, false);
    let lg_plain = d2.decode(&model_b, &[3, 17, 42], &[0, 1, 2]);
    assert_ne!(
        lg_alibi[..64],
        lg_plain[..64],
        "max_bias=8 + alibi mask must change the logits"
    );

    let a = smoke_forward(&spec, false);
    println!("baichuan-13B synth: greedy first token {}", argmax(&a));
    let b = smoke_forward(&spec7, false);
    println!("baichuan-7B synth: greedy first token {}", argmax(&b));
}

#[test]
fn synth_bloom_loader_and_forward() {
    let spec = spec_bloom();
    let (n, bytes) = build_file(&spec);
    assert_eq!(
        n,
        5 + 2 * 12,
        "bloom tensor count (no output.weight — tied)"
    );
    println!("bloom synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // bloom.cpp:18 — unconditional alibi
    assert_eq!(m.hparams.f_max_alibi_bias, 8.0);
    assert!(m.hparams.use_alibi);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NONE);
    assert_eq!(m.output, m.tok_embd, "bloom ties the head to tok_embd");
    assert!(m.token_embd_norm.is_some() && m.token_embd_norm_b.is_some());
    assert!(m.layers[0].wqkv.is_some() && m.layers[0].wqkv_b.is_some());
    assert!(m.layers[0].wo_b.is_some() && m.layers[0].ffn_up_b.is_some());

    let a = smoke_forward(&spec, false);
    println!("bloom synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_mpt_loader_and_forward() {
    let spec = spec_mpt();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 6, "mpt tensor count (the minimal file)");
    println!("mpt synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // mpt.cpp:6 — the bias comes from the GGUF key
    assert_eq!(m.hparams.f_max_alibi_bias, 8.0);
    assert!(m.hparams.use_alibi);
    assert_eq!(m.hparams.f_clamp_kqv, 0.0);
    assert!(m.position_embd.is_none());
    assert!(m.layers[0].attn_q_norm.is_none() && m.layers[0].ffn_act.is_none());
    assert_eq!(m.output, m.tok_embd);

    // the -full variant: every optional at once
    let full = spec_mpt_full();
    let (nf, _) = build_file(&full);
    let mf = load_synth(&full);
    pin_tensors(&mf, &full);
    pin_hparams(&mf, &full);
    assert_eq!(mf.hparams.f_clamp_kqv, 0.02);
    assert!(mf.position_embd.is_some());
    assert!(mf.layers[0].ffn_act.is_some());
    assert!(mf.layers[0].wqkv_b.is_some() && mf.output_norm_b.is_some());
    assert_ne!(mf.output, mf.tok_embd);
    assert_eq!(nf, tensors_for(&full).len());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&full, false);
    println!(
        "mpt synth: minimal first token {}, -full {}",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(a, b, "the mpt optionals must change the logits");
}

#[test]
fn synth_starcoder_loader_and_forward() {
    let spec = spec_starcoder();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 12, "starcoder tensor count");
    println!(
        "starcoder synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // starcoder has NO alibi at this revision (its loader never sets one)
    assert_eq!(m.hparams.f_max_alibi_bias, 0.0);
    assert!(!m.hparams.use_alibi);
    assert!(m.position_embd.is_some(), "starcoder pos_embd is required");
    assert_eq!(m.output, m.tok_embd, "head ties to tok_embd");
    assert!(m.layers[0].wqkv_b.is_some() && m.layers[0].wo_b.is_some());

    let a = smoke_forward(&spec, false);
    println!("starcoder synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_refact_loader_and_forward() {
    let spec = spec_refact();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 9 + 1, "refact tensor count (+ rope_freqs)");
    println!(
        "refact synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // refact.cpp:12 — unconditional alibi; rope type NONE
    assert_eq!(m.hparams.f_max_alibi_bias, 8.0);
    assert!(m.hparams.use_alibi);
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NONE);
    assert_eq!(m.output, m.tok_embd);
    // the rope factors are consumed but never read by the graph
    assert!(m.layers[0].rope_freqs.is_some() && m.layers[1].rope_freqs.is_some());
    assert_eq!(
        m.layers[0].rope_freqs, m.layers[1].rope_freqs,
        "dup request reuses it"
    );

    let a = smoke_forward(&spec, false);
    println!("refact synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_plamo_loader_and_forward() {
    let spec = spec_plamo();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 2 * 8, "plamo tensor count (no ffn_norm)");
    println!("plamo synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    assert!(
        m.layers.iter().all(|l| l.ffn_norm.is_none()),
        "plamo has no ffn_norm"
    );
    assert_ne!(m.output, m.tok_embd, "plamo head is required");
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);

    let a = smoke_forward(&spec, false);
    println!("plamo synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_stablelm_loader_and_forward() {
    // 12B shape: per-head q/k norms + parallel residual
    let spec = spec_stablelm_12b();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 11, "stablelm-12B tensor count (no ffn_norm)");
    println!(
        "stablelm synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    assert!(m.layers[0].attn_q_norm.is_some() && m.layers[0].attn_k_norm.is_some());
    assert!(
        m.layers.iter().all(|l| l.ffn_norm.is_none()),
        "12B runs parallel residual"
    );
    assert!(m.output_norm_b.is_some());

    // 3B shape: sequential ffn_norm, partial rotary
    let spec3 = spec_stablelm_3b();
    let (n3, _) = build_file(&spec3);
    let m3 = load_synth(&spec3);
    pin_tensors(&m3, &spec3);
    pin_hparams(&m3, &spec3);
    assert!(m3.layers[0].attn_q_norm.is_none());
    assert!(m3.layers[0].ffn_norm.is_some() && m3.layers[0].ffn_norm_b.is_some());
    assert_eq!(
        m3.hparams.n_rot(0),
        12,
        "partial rope via rope.dimension_count"
    );
    assert_eq!(m3.hparams.n_embd_head_k(0), 16);
    assert_eq!(n3, tensors_for(&spec3).len());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec3, false);
    println!(
        "stablelm synth: 12B first token {}, 3B {}",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(a, b);
}

#[test]
fn synth_granite_dense_loader_and_forward() {
    let spec = spec_granite();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 9, "granite(dense) tensor count");
    println!(
        "granite synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // granite.cpp:20-24 — logit_scale required, the rest optional overrides
    assert_eq!(m.hparams.f_logit_scale, 6.0);
    assert_eq!(m.hparams.f_residual_scale, 0.5);
    assert_eq!(m.hparams.f_embedding_scale, 4.0);
    assert_eq!(m.hparams.f_attention_scale, 0.125);
    // rope_finetuned defaults true → every layer ropes
    assert!(m.hparams.has_rope(0) && m.hparams.has_rope(1));
    assert_eq!(m.output, m.tok_embd);

    let a = smoke_forward(&spec, false);
    println!("granite synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_minicpm_loader_and_forward() {
    let spec = spec_minicpm();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 9, "minicpm tensor count (granite's set)");
    println!(
        "minicpm synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    // minicpm.cpp:5-7 — the backward-compatible defaults, no scale keys
    assert_eq!(m.hparams.f_embedding_scale, 12.0);
    assert!((m.hparams.f_residual_scale - 1.4 / 2f32.sqrt()).abs() < 1e-6);
    assert!((m.hparams.f_logit_scale - 256.0 / 64.0).abs() < 1e-6);
    assert!(m.hparams.has_rope(0), "minicpm ropes by default");

    let a = smoke_forward(&spec, false);
    println!("minicpm synth: greedy first token {}", argmax(&a));
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_mpt().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("mpt"));
    assert_eq!(g.get_f32("mpt.attention.max_alibi_bias"), Some(8.0));
    let spec = spec_granite().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_f32("granite.logit_scale"), Some(6.0));
    assert_eq!(g.get_f32("granite.attention.scale"), Some(0.125));
    let spec = spec_stablelm_3b().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_u32("stablelm.rope.dimension_count"), Some(12));
    assert!(matches!(
        g.find_key("tokenizer.ggml.tokens").map(|v| v.type_()),
        Some(GgufType::Array)
    ));
}

/// The non-FA ALiBi softmax end-to-end against hand-computed slopes: the
/// kernel's `wp[i] += slope * mask[i]` (ops.cpp:5636) with the alibi mask's
/// `-|p0-p1|` must equal a naive softmax over `kq*scale + slope*(-dist)`.
/// Slopes for n_head=4, max_bias=8: n_head_log2=4, m0=2^(-8/4) →
/// [1/4, 1/16, 1/64, 1/256] (hand-computed, ops.cpp:5618-5637).
#[test]
fn alibi_softmax_matches_naive_slopes() {
    use ggml::compute::graph_compute;
    use ggml::graph::Graph;

    let (h, n_kv, t) = (4i64, 6i64, 2i64);
    let mut ctx = Context::new();
    // kq [n_kv, T, H] — deterministic values
    let kq = ctx.new_tensor_3d(GgmlType::F32, n_kv, t, h);
    ctx.arena_resize_tensor(kq);
    ctx.with_f32_mut(kq, |p| {
        for (i, v) in p.iter_mut().enumerate() {
            *v = ((i as f32) * 0.37).sin();
        }
    })
    .unwrap();
    // alibi F32 mask [n_kv, T]: kept = -|p0-p1| over kv positions 0..6,
    // queries 4 and 5 (kv 5 is in the future of query 4 → -inf) — the exact
    // hand-computed values of graph::alibi_mask_kept_value_is_neg_distance
    let mask = ctx.new_tensor_2d(GgmlType::F32, n_kv, t);
    ctx.arena_resize_tensor(mask);
    {
        let m: &mut [f32] = bytemuck::cast_slice_mut(ctx.data_bytes_mut(mask).unwrap());
        llama::graph::fill_kq_mask_alibi(
            m,
            &[0, 1, 2, 3, 4, 5],
            &[4, 5],
            0,
            llama::hparams::LlamaSwaType::NONE,
        );
    }

    let scale = 0.5f32;
    let max_bias = 8.0f32;
    let sm = ctx.soft_max_ext(kq, Some(mask), scale, max_bias);
    let mut g = Graph::new(8);
    g.build_forward(&ctx, sm);
    graph_compute(&mut ctx, &mut g, 1);

    let slopes: [f32; 4] = [
        0.25f32.powi(1),
        0.25f32.powi(2),
        0.25f32.powi(3),
        0.25f32.powi(4),
    ];
    let mvals: [[f32; 6]; 2] = [
        [-4.0, -3.0, -2.0, -1.0, 0.0, f32::NEG_INFINITY],
        [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
    ];
    let got: Vec<f32> = ctx
        .data_bytes(sm)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let src: Vec<f32> = ctx
        .data_bytes(kq)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for hh in 0..h as usize {
        for tt in 0..t as usize {
            let w: Vec<f32> = (0..n_kv as usize)
                .map(|s| {
                    (src[s + tt * n_kv as usize + hh * (n_kv * t) as usize] * scale
                        + slopes[hh] * mvals[tt][s])
                        .exp()
                })
                .collect();
            let sum: f32 = w.iter().sum();
            for s in 0..n_kv as usize {
                let idx = s + tt * n_kv as usize + hh * (n_kv * t) as usize;
                let want = w[s] / sum;
                assert!(
                    (got[idx] - want).abs() < 1e-6,
                    "h={hh} t={tt} s={s}: {} vs {want}",
                    got[idx]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// generator + the parity CLI driver (both #[ignore])
// ---------------------------------------------------------------------------

#[test]
#[ignore = "manual: writes ~600 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch3_write_synth() {
    for spec in all_specs() {
        let (n, bytes) = build_file(&spec);
        println!(
            "{:>11}: {:4} tensors, {:>10} bytes -> {}",
            spec.arch,
            n,
            bytes,
            spec.path()
        );
    }
    let gguf = Gguf::open(spec_refact().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH3=1 ./parity/arch_batch_parity.sh baichuan13 baichuan7 bloom mpt \
         starcoder refact plamo stablelm granite minicpm"
    );
}

/// The port side of the parity runs — env-driven greedy decode printing
/// llama-cli's `LLAMA_RUST_DEBUG` format (`step N: top5 [...] greedy=` /
/// `gen tokens: [...]`), so `parity/arch_batch_cmp.py` works unchanged.
///
/// Env: ARCH3_MODEL (gguf path), ARCH3_FA (on|off), ARCH3_PROMPT,
/// ARCH3_N (default 16). Driven by `parity/arch_batch_parity.sh` with
/// ARCH_BATCH3=1.
#[test]
#[ignore = "manual: the parity driver — env ARCH3_MODEL/ARCH3_FA/ARCH3_PROMPT/ARCH3_N"]
fn arch3_cli_driver() {
    let path = std::env::var("ARCH3_MODEL").expect("ARCH3_MODEL");
    let fa = match std::env::var("ARCH3_FA").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        other => panic!("ARCH3_FA must be on|off, got {other:?}"),
    };
    let prompt =
        std::env::var("ARCH3_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let n_predict: usize = std::env::var("ARCH3_N")
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
    println!(
        "arch = {arch_name}, {n_tensors} tensors, use_alibi = {}, fa = {fa}",
        d.alibi
    );

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
