//! arch_batch6b_e2e.rs — synthetic-GGUF verification of architecture batch
//! 6b (llama.cpp bd4f514db1): **nemotron (dense) / grok / chameleon / deci /
//! jais / falcon-h1 / plamo2** — the dense post-norm + ALiBi + hybrid-mamba
//! entries of the batch-7 queue.
//!
//! Same protocol as batches 1-6 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32. Every MoE file carries
//! `n_expert = 4`, `n_expert_used = 2` so the routing is actually exercised.
//!
//! The files pin one variant per interesting branch:
//!   * `nemotron` — the LayerNorm+bias + relu² dense graph with every
//!     optional bias present (nemotron.cpp:50-150);
//!   * `grok` — post-norm attention + GELU MoE (softmax, norm_w) under the
//!     `post_ffw_norm` name; `grok-dense` adds the dense FFN branch and the
//!     `layer_output_norm` name (grok.cpp:75-84/:171-184);
//!   * `chameleon` — the full-width q/k LayerNorms + SwiGLU + the
//!     image-token logit suppression; `chameleon-swin` flips swin_norm;
//!   * `deci` — uniform attention layers; `deci-mixed` drives every layer
//!     kind (attention / linear-attention / attention-free / FFN-free) via
//!     the per-layer head/ffn arrays + per-layer rope_freqs factors;
//!   * `jais` — LayerNorm+bias everywhere, fused qkv+bias, no rope, ALiBi
//!     from the GGUF KV (`attention.max_alibi_bias = 8`, jais.cpp:5);
//!   * `falcon-h1` — the mamba2 mixer AND attention in *every* layer with
//!     the double-residual aggregation (falcon-h1.cpp:112-209);
//!   * `plamo2` — the hybrid with its own mamba mixer (per-head z/x split,
//!     bcdt x-projection, RMS-normed dt/B/C) and attention layers whose V
//!     head dim (48) differs from Q/K (32).
//!
//! Like batch 6, this batch's ForwardWeights/CLI arms landed with the graph,
//! so the default-run tests drive `DecodeContext::new_with` itself and the
//! parity runs drive the release CLI (`ARCH_BATCH6B=1 ./parity/
//! arch_batch_parity.sh …`, batch-1 protocol; "-long" appends the >64-token
//! prompt cell).

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{self};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch6b";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model specs
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: Vec<i64>,
    n_head_kv: Vec<i64>,
    n_embd_head_k: i64,
    n_embd_head_v: i64,
    n_rot: i64,
    n_ff: Vec<i64>,
    n_ff_exp: i64,
    n_ctx: u32,
    // arch-specific bits
    /// grok only: carry the dense FFN branch + the layer_output_norm name
    grok_dense: bool,
    /// chameleon only: swin_norm
    swin_norm: bool,
    /// deci only: carry the model-level `rope_freqs.weight` factor tensor
    /// (the C template is "%d"-less, so every layer resolves the same one)
    deci_rope_freqs: bool,
    /// jais only: attention.max_alibi_bias
    max_alibi_bias: Option<f32>,
    /// falcon-h1 / plamo2 SSM geometry
    ssm: Option<SsmSpec>,
}

#[derive(Clone, Copy)]
struct SsmSpec {
    d_conv: i64,
    d_inner: i64,
    d_state: i64,
    /// mamba head count: ssm_dt_rank
    n_head: i64,
    n_group: i64,
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
    fn head_count_kv(&self) -> i64 {
        self.n_head_kv[0]
    }
}

fn base_spec(arch: &'static str, n_layer: usize, n_head: i64, n_head_kv: i64) -> SynthSpec {
    SynthSpec {
        arch,
        suffix: "",
        n_layer,
        n_embd: 128,
        n_head: vec![n_head; n_layer],
        n_head_kv: vec![n_head_kv; n_layer],
        n_embd_head_k: 32,
        n_embd_head_v: 32,
        n_rot: 32,
        n_ff: vec![64; n_layer],
        n_ff_exp: 32,
        n_ctx: 256,
        grok_dense: false,
        swin_norm: false,
        deci_rope_freqs: false,
        max_alibi_bias: None,
        ssm: None,
    }
}

/// nemotron (dense) — LayerNorm+bias + relu², GQA 4/2, every optional bias
fn spec_nemotron() -> SynthSpec {
    base_spec("nemotron", 4, 4, 2)
}

/// grok — post-norm attention + GELU MoE, post_ffw_norm name, no dense branch
fn spec_grok() -> SynthSpec {
    base_spec("grok", 4, 8, 2).with(|s| {
        s.n_embd_head_k = 16;
        s.n_embd_head_v = 16;
        s.n_rot = 16;
    })
}

/// grok-dense — the dense FFN branch + the layer_output_norm tensor name
fn spec_grok_dense() -> SynthSpec {
    spec_grok().with(|s| {
        s.suffix = "-dense";
        s.grok_dense = true;
    })
}

fn spec_chameleon() -> SynthSpec {
    base_spec("chameleon", 4, 4, 2)
}

fn spec_chameleon_swin() -> SynthSpec {
    spec_chameleon().with(|s| {
        s.suffix = "-swin";
        s.swin_norm = true;
    })
}

/// deci base — uniform attention layers, no rope factor tensors
fn spec_deci() -> SynthSpec {
    base_spec("deci", 6, 4, 2)
}

/// deci-mixed — one layer of each kind (attention / linear / attention-free
/// / FFN-free) via the per-layer arrays, with rope_freqs factors everywhere
fn spec_deci_mixed() -> SynthSpec {
    let mut s = base_spec("deci", 8, 4, 2);
    s.suffix = "-mixed";
    s.n_head = vec![4, 2, 0, 4, 4, 4, 4, 4];
    s.n_head_kv = vec![2, 0, 0, 2, 2, 2, 2, 2];
    s.n_ff = vec![64, 64, 64, 0, 64, 64, 64, 64];
    s.deci_rope_freqs = true;
    s
}

/// jais — MHA, fused qkv+bias, no rope, ALiBi 8.0 from the GGUF KV
fn spec_jais() -> SynthSpec {
    base_spec("jais", 4, 4, 4).with(|s| {
        s.max_alibi_bias = Some(8.0);
    })
}

/// falcon-h1 — mamba2 mixer + attention in every layer (is_recr all true)
fn spec_falcon_h1() -> SynthSpec {
    base_spec("falcon-h1", 4, 4, 2).with(|s| {
        s.ssm = Some(SsmSpec {
            d_conv: 4,
            d_inner: 64,
            d_state: 16,
            n_head: 8,
            n_group: 1,
        });
    })
}

/// plamo2 — 6 layers, is_recr = n_head_kv == 0 (layers 0/1/3/4 mamba), the
/// attention layers carry a distinct V head dim (48 vs Q/K 32)
fn spec_plamo2() -> SynthSpec {
    base_spec("plamo2", 6, 4, 2).with(|s| {
        s.n_head_kv = vec![0, 0, 2, 0, 0, 2];
        s.n_embd_head_v = 48;
        s.ssm = Some(SsmSpec {
            d_conv: 4,
            d_inner: 64,
            d_state: 16,
            n_head: 8,
            n_group: 0,
        });
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_nemotron(),
        spec_grok(),
        spec_grok_dense(),
        spec_chameleon(),
        spec_chameleon_swin(),
        spec_deci(),
        spec_deci_mixed(),
        spec_jais(),
        spec_falcon_h1(),
        spec_plamo2(),
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

fn tensors_for(spec: &SynthSpec) -> Vec<(String, Vec<i64>, Role)> {
    let mut v: Vec<(String, Vec<i64>, Role)> = Vec::new();
    let n_embd = spec.n_embd;
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $role))
        };
    }
    // model level — the tied-head archs omit output.weight (the loader's
    // TENSOR_DUPLICATED fallback), nemotron/jais require it
    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    match spec.arch {
        "nemotron" | "jais" => {
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);
        }
        _ => {
            push!("output_norm.weight", vec![n_embd], Role::Norm);
        }
    }
    // deci's rope factors: the C's `tn(LLM_TENSOR_ROPE_FREQS, "weight", i)`
    // formats the "%d"-less "rope_freqs" template — ONE model-level tensor
    // every layer resolves to (the TENSOR_DUPLICATED flag dedups the name)
    if spec.deci_rope_freqs {
        push!("rope_freqs.weight", vec![spec.n_rot / 2], Role::Norm);
    }

    let n_ff_exp = spec.n_ff_exp;
    for i in 0..spec.n_layer as i32 {
        let il = i as usize;
        let n_head = spec.n_head[il];
        let n_head_kv = spec.n_head_kv[il];
        let n_ff = spec.n_ff[il];
        let hd = spec.n_embd_head_k;
        let n_embd_gqa_k = hd * n_head_kv;
        let n_embd_gqa_v = spec.n_embd_head_v * n_head_kv;

        match spec.arch {
            "nemotron" => {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k.weight"),
                    vec![n_embd, n_embd_gqa_k],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, n_embd_gqa_v],
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
                push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Norm);
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
            "grok" => {
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
                    vec![n_embd, n_embd_gqa_k],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, n_embd_gqa_v],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if spec.grok_dense {
                    // the dense branch (grok.cpp:66-68)
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
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, N_EXPERT],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, n_ff_exp, N_EXPERT],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![n_ff_exp, n_embd, N_EXPERT],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, n_ff_exp, N_EXPERT],
                    Role::Proj
                );
                // ffn_post_norm: layer_output_norm (dense variant) or
                // post_ffw_norm (grok.cpp:75-78)
                if spec.grok_dense {
                    push!(
                        format!("blk.{i}.layer_output_norm.weight"),
                        vec![n_embd],
                        Role::Norm
                    );
                } else {
                    push!(
                        format!("blk.{i}.post_ffw_norm.weight"),
                        vec![n_embd],
                        Role::Norm
                    );
                }
            }
            "chameleon" => {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                // full-width q/k norms (+ optional biases, chameleon.cpp:33-36)
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![hd, n_head],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_norm.bias"),
                    vec![hd, n_head],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![hd, n_head_kv],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.bias"),
                    vec![hd, n_head_kv],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k.weight"),
                    vec![n_embd, n_embd_gqa_k],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, n_embd_gqa_v],
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
            "deci" => {
                if n_head_kv == 0 && n_head > 0 {
                    // linear attention (deci.cpp:36-40)
                    push!(
                        format!("blk.{i}.attn_norm.weight"),
                        vec![n_embd],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![n_embd, n_embd],
                        Role::Proj
                    );
                } else if n_head_kv > 0 {
                    push!(
                        format!("blk.{i}.attn_norm.weight"),
                        vec![n_embd],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, hd * n_head],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, n_embd_gqa_k],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, n_embd_gqa_v],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![hd * n_head, n_embd],
                        Role::Proj
                    );
                }
                // n_head == 0: attention-free — no attention tensors at all
                push!(
                    format!("blk.{i}.attn_output.bias"),
                    vec![n_embd],
                    Role::Bias
                );
                if n_ff > 0 {
                    push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                }
                // rope factors: the model-level rope_freqs tensor (see the
                // model-level push above) — nothing per-layer
                if n_ff > 0 {
                    push!(
                        format!("blk.{i}.ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj
                    );
                    push!(format!("blk.{i}.ffn_gate.bias"), vec![n_ff], Role::Bias);
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
            "jais" => {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_norm.bias"), vec![n_embd], Role::Norm);
                let qkv = n_embd + 2 * n_embd_gqa_k;
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, qkv],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_qkv.bias"), vec![qkv], Role::Bias);
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
                push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_gate.bias"), vec![n_ff], Role::Bias);
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
            "falcon-h1" => {
                let s = spec.ssm.unwrap();
                let conv_dim = s.d_inner + 2 * s.n_group * s.d_state;
                let d_in_proj = s.d_inner + conv_dim + s.n_head;
                // SSM LAYERS (falcon-h1.cpp:71-85)
                push!(
                    format!("blk.{i}.ssm_in.weight"),
                    vec![n_embd, d_in_proj],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_conv1d.weight"),
                    vec![s.d_conv, conv_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ssm_conv1d.bias"),
                    vec![conv_dim],
                    Role::Bias
                );
                push!(format!("blk.{i}.ssm_dt.bias"), vec![s.n_head], Role::Bias);
                push!(format!("blk.{i}.ssm_a"), vec![1, s.n_head], Role::Bias);
                push!(format!("blk.{i}.ssm_d"), vec![1, s.n_head], Role::Bias);
                push!(
                    format!("blk.{i}.ssm_norm.weight"),
                    vec![s.d_inner / s.n_group, s.n_group],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ssm_out.weight"),
                    vec![s.d_inner, n_embd],
                    Role::Proj
                );
                // ATTENTION (falcon-h1.cpp:87-92)
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, hd * n_head],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k.weight"),
                    vec![n_embd, n_embd_gqa_k],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v.weight"),
                    vec![n_embd, n_embd_gqa_v],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![hd * n_head, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.bias"),
                    vec![n_embd],
                    Role::Bias
                );
                // FFN (falcon-h1.cpp:95-104) — ffn_norm has no ".weight"
                push!(format!("blk.{i}.ffn_norm"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_gate.bias"), vec![n_ff], Role::Bias);
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
            "plamo2" => {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if n_head_kv == 0 {
                    // mamba (plamo2.cpp:66-80)
                    let s = spec.ssm.unwrap();
                    let dt_dim = 64.max(spec.n_embd / 16);
                    push!(
                        format!("blk.{i}.ssm_in.weight"),
                        vec![n_embd, 2 * s.d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![s.d_conv, s.d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_x.weight"),
                        vec![s.d_inner, dt_dim + 2 * s.d_state],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_dt.weight"),
                        vec![dt_dim, s.n_head],
                        Role::Proj
                    );
                    push!(format!("blk.{i}.ssm_dt.bias"), vec![s.n_head], Role::Bias);
                    push!(format!("blk.{i}.ssm_a"), vec![s.n_head], Role::Bias);
                    push!(format!("blk.{i}.ssm_d"), vec![s.n_head], Role::Bias);
                    push!(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![s.d_inner, n_embd],
                        Role::Proj
                    );
                    push!(format!("blk.{i}.ssm_dt_norm"), vec![dt_dim], Role::Norm);
                    push!(format!("blk.{i}.ssm_b_norm"), vec![s.d_state], Role::Norm);
                    push!(format!("blk.{i}.ssm_c_norm"), vec![s.d_state], Role::Norm);
                } else {
                    // attention (plamo2.cpp:81-94) — V carries its own dim
                    let qk_dim = spec.n_embd_head_k;
                    let v_dim = spec.n_embd_head_v;
                    let q_proj = qk_dim * n_head;
                    let k_proj = qk_dim * n_head_kv;
                    let v_proj = v_dim * n_head_kv;
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, q_proj + k_proj + v_proj],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_q_norm.weight"),
                        vec![qk_dim, n_head],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_k_norm.weight"),
                        vec![qk_dim, n_head_kv],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![n_head * v_dim, n_embd],
                        Role::Proj
                    );
                }
                // shared tail (plamo2.cpp:97-102) — attn_post_norm /
                // ffn_post_norm carry no ".weight"; ffn_up is 2*n_ff wide
                push!(
                    format!("blk.{i}.post_attention_norm"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up.weight"),
                    vec![n_embd, n_ff * 2],
                    Role::Proj
                );
                push!(format!("blk.{i}.post_ffw_norm"), vec![n_embd], Role::Norm);
            }
            other => panic!("unknown synth arch {other}"),
        }
    }
    v
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch6_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch6b");

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
    kv!(format!("{a}.attention.head_count"), {
        // deci-mixed / plamo2 write the per-layer array; everything else a
        // scalar (the generic get_key_or_arr fills either way)
        if spec.n_head.iter().all(|&h| h == spec.n_head[0]) {
            Value::U32(spec.n_head[0] as u32)
        } else {
            Value::Array(
                GgufType::Uint32,
                spec.n_head.iter().map(|&h| Value::U32(h as u32)).collect(),
            )
        }
    });
    kv!(format!("{a}.attention.head_count_kv"), {
        if spec.n_head_kv.iter().all(|&h| h == spec.head_count_kv()) {
            Value::U32(spec.head_count_kv() as u32)
        } else {
            Value::Array(
                GgufType::Uint32,
                spec.n_head_kv
                    .iter()
                    .map(|&h| Value::U32(h as u32))
                    .collect(),
            )
        }
    });
    if spec.n_ff.iter().all(|&f| f == spec.n_ff[0]) {
        kv!(
            format!("{a}.feed_forward_length"),
            Value::U32(spec.n_ff[0] as u32)
        );
    } else {
        let ffn_arr = Value::Array(
            GgufType::Uint32,
            spec.n_ff.iter().map(|&f| Value::U32(f as u32)).collect(),
        );
        kv!(format!("{a}.feed_forward_length"), ffn_arr);
    }
    kv!(
        format!("{a}.attention.key_length"),
        Value::U32(spec.n_embd_head_k as u32)
    );
    if spec.n_embd_head_v != spec.n_embd_head_k {
        // plamo2's distinct V head dim (attention.value_length)
        kv!(
            format!("{a}.attention.value_length"),
            Value::U32(spec.n_embd_head_v as u32)
        );
    }
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.n_rot as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    match a {
        // LayerNorm eps (nemotron.cpp:4 / jais.cpp:4)
        "nemotron" | "jais" => {
            kv!(
                format!("{a}.attention.layer_norm_epsilon"),
                Value::F32(1e-5)
            );
        }
        _ => {
            kv!(
                format!("{a}.attention.layer_norm_rms_epsilon"),
                Value::F32(1e-5)
            );
        }
    }
    if a == "grok" {
        // grok.cpp:16-21 — the optional scale/softcap keys; omitting them all
        // pins the loader defaults (logit 1/sqrt(3), attn softcap 30, ...)
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
    if a == "chameleon" && spec.swin_norm {
        kv!(format!("{a}.swin_norm"), Value::Bool(true));
    }
    if let Some(bias) = spec.max_alibi_bias {
        // the ALiBi bias as a GGUF key (jais.cpp:5)
        kv!(format!("{a}.attention.max_alibi_bias"), Value::F32(bias));
    }
    if let Some(s) = spec.ssm {
        // the SSM keys (falcon-h1.cpp:7-12 / plamo2.cpp:7-12)
        kv!(format!("{a}.ssm.conv_kernel"), Value::U32(s.d_conv as u32));
        kv!(format!("{a}.ssm.inner_size"), Value::U32(s.d_inner as u32));
        kv!(format!("{a}.ssm.state_size"), Value::U32(s.d_state as u32));
        kv!(
            format!("{a}.ssm.time_step_rank"),
            Value::U32(s.n_head as u32)
        );
        kv!(format!("{a}.ssm.group_count"), Value::U32(s.n_group as u32));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f00d ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
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
// weights + params assembly (mirrors crates/tools/llama-cli/src/main.rs)
// ---------------------------------------------------------------------------

fn attn_of(m: &LlamaModel, il: usize, fa: bool) -> AttnParams {
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

/// (ForwardWeights, AttnParams) of one loaded model — the same assembly the
/// CLI performs (this batch landed its context.rs arms together).
fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let first_attn = (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap_or(0);
    match m.arch {
        llama::arch::LlmArch::NEMOTRON => {
            let mut attn = attn_of(m, 0, fa);
            attn.norm_eps = hp.f_norm_eps;
            let w = graph_arch::NemotronModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output_norm_b: m.output_norm_b.unwrap(),
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::NemotronLayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        attn_norm_b: l.attn_norm_b.unwrap(),
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
                        ffn_norm_b: l.ffn_norm_b.unwrap(),
                        ffn_up: l.ffn_up.unwrap(),
                        ffn_up_b: l.ffn_up_b,
                        ffn_down: l.ffn_down.unwrap(),
                        ffn_down_b: l.ffn_down_b,
                    })
                    .collect(),
            };
            (
                ForwardWeights::Nemotron(w, graph_arch::NemotronParams { attn }),
                attn,
            )
        }
        llama::arch::LlmArch::GROK => {
            let attn = attn_of(m, 0, fa);
            let w = graph_arch::GrokModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::GrokLayerWeights {
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
                        attn_out_norm: l.attn_out_norm.unwrap(),
                        ffn_norm: l.ffn_norm.unwrap(),
                        ffn_gate: l.ffn_gate,
                        ffn_down: l.ffn_down,
                        ffn_up: l.ffn_up,
                        ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                        ffn_gate_exps: l.ffn_gate_exps,
                        ffn_down_exps: l.ffn_down_exps.unwrap(),
                        ffn_up_exps: l.ffn_up_exps.unwrap(),
                        ffn_post_norm: l.ffn_post_norm.unwrap(),
                    })
                    .collect(),
            };
            let p = graph_arch::GrokParams {
                attn,
                n_expert: hp.n_expert as i64,
                n_expert_used: hp.n_expert_used(0) as i64,
                f_logit_scale: hp.f_logit_scale,
                f_embedding_scale: hp.f_embedding_scale,
                f_final_logit_softcapping: hp.f_final_logit_softcapping,
                f_attn_out_scale: hp.f_attn_out_scale,
                f_attn_logit_softcapping: hp.f_attn_logit_softcapping,
            };
            (ForwardWeights::Grok(w, p), attn)
        }
        llama::arch::LlmArch::CHAMELEON => {
            let attn = attn_of(m, 0, fa);
            let w = graph_arch::ChameleonModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::ChameleonLayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        attn_q_norm: l.attn_q_norm.unwrap(),
                        attn_q_norm_b: l.attn_q_norm_b,
                        attn_k_norm: l.attn_k_norm.unwrap(),
                        attn_k_norm_b: l.attn_k_norm_b,
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
            };
            let p = graph_arch::ChameleonParams {
                attn,
                f_norm_eps_qk: hp.f_norm_eps,
                swin_norm: hp.swin_norm,
            };
            (ForwardWeights::Chameleon(w, p), attn)
        }
        llama::arch::LlmArch::DECI => {
            let il0 = (0..n_trunk).find(|&il| hp.n_head_kv(il) > 0).unwrap_or(0);
            let mut attn = attn_of(m, il0, fa);
            attn.norm_eps = hp.f_norm_rms_eps;
            // get_rope_factors (llama-model.cpp:2259-2272) with n_ctx 512:
            // the rope_freqs tensors win when present
            let n_ctx_seq = 512i64;
            let w = graph_arch::DeciModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::DeciLayerWeights {
                        attn_norm: l.attn_norm,
                        wqkv: l.wqkv,
                        wqkv_b: l.wqkv_b,
                        wq: l.wq,
                        wq_b: l.wq_b,
                        wk: l.wk,
                        wk_b: l.wk_b,
                        wv: l.wv,
                        wv_b: l.wv_b,
                        wo: l.wo,
                        wo_b: l.wo_b,
                        rope_factors: if l.rope_freqs.is_some() {
                            l.rope_freqs
                        } else if n_ctx_seq > attn.n_ctx_orig as i64 {
                            l.rope_long
                        } else {
                            l.rope_short
                        },
                        ffn_norm: l.ffn_norm,
                        ffn_gate: l.ffn_gate,
                        ffn_gate_b: l.ffn_gate_b,
                        ffn_down: l.ffn_down,
                        ffn_down_b: l.ffn_down_b,
                        ffn_up: l.ffn_up,
                        ffn_up_b: l.ffn_up_b,
                    })
                    .collect(),
            };
            let p = graph_arch::DeciParams {
                attn,
                n_head: (0..n_trunk).map(|il| hp.n_head(il)).collect(),
                n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il)).collect(),
                n_ff: (0..n_trunk).map(|il| hp.n_ff(il) as i64).collect(),
                f_attention_scale: hp.f_attention_scale,
            };
            (ForwardWeights::Deci(w, p), attn)
        }
        llama::arch::LlmArch::JAIS => {
            let mut attn = attn_of(m, 0, fa);
            attn.norm_eps = hp.f_norm_eps;
            let w = graph_arch::JaisModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output_norm_b: m.output_norm_b.unwrap(),
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::JaisLayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        attn_norm_b: l.attn_norm_b.unwrap(),
                        wqkv: l.wqkv.unwrap(),
                        wqkv_b: l.wqkv_b.unwrap(),
                        wo: l.wo.unwrap(),
                        wo_b: l.wo_b.unwrap(),
                        ffn_norm: l.ffn_norm.unwrap(),
                        ffn_norm_b: l.ffn_norm_b.unwrap(),
                        ffn_gate: l.ffn_gate.unwrap(),
                        ffn_gate_b: l.ffn_gate_b.unwrap(),
                        ffn_down: l.ffn_down.unwrap(),
                        ffn_down_b: l.ffn_down_b.unwrap(),
                        ffn_up: l.ffn_up.unwrap(),
                        ffn_up_b: l.ffn_up_b.unwrap(),
                    })
                    .collect(),
            };
            let p = graph_arch::JaisParams {
                attn,
                f_max_alibi_bias: hp.f_max_alibi_bias,
            };
            (ForwardWeights::Jais(w, p), attn)
        }
        llama::arch::LlmArch::FALCON_H1 => {
            let attn = attn_of(m, 0, fa);
            let w = graph_arch::FalconH1ModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::FalconH1LayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        ssm_in: l.ssm_in.unwrap(),
                        ssm_conv1d: l.ssm_conv1d.unwrap(),
                        ssm_conv1d_b: l.ssm_conv1d_b,
                        ssm_dt_b: l.ssm_dt_b.unwrap(),
                        ssm_a: l.ssm_a.unwrap(),
                        ssm_d: l.ssm_d.unwrap(),
                        ssm_norm: l.ssm_norm,
                        ssm_out: l.ssm_out.unwrap(),
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
                        ffn_gate_b: l.ffn_gate_b,
                        ffn_down: l.ffn_down.unwrap(),
                        ffn_down_b: l.ffn_down_b,
                        ffn_up: l.ffn_up.unwrap(),
                        ffn_up_b: l.ffn_up_b,
                    })
                    .collect(),
            };
            let p = graph_arch::FalconH1Params {
                attn,
                n_embd: hp.n_embd as i64,
                d_conv: hp.ssm_d_conv as i64,
                d_inner: hp.ssm_d_inner as i64,
                d_state: hp.ssm_d_state as i64,
                n_ssm_head: hp.ssm_dt_rank as i64,
                n_group: hp.ssm_n_group as i64,
                norm_eps: hp.f_norm_rms_eps,
                f_attention_scale: hp.f_attention_scale,
                n_embd_r: hp.n_embd_r(),
                n_embd_s: hp.n_embd_s(),
            };
            (ForwardWeights::FalconH1(w, p), attn)
        }
        llama::arch::LlmArch::PLAMO2 => {
            let attn = attn_of(m, first_attn, fa);
            let w = graph_arch::Plamo2ModelWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                layers: m.layers[..n_trunk]
                    .iter()
                    .map(|l| graph_arch::Plamo2LayerWeights {
                        attn_norm: l.attn_norm.unwrap(),
                        ssm_in: l.ssm_in,
                        ssm_conv1d: l.ssm_conv1d,
                        ssm_x: l.ssm_x,
                        ssm_dt: l.ssm_dt,
                        ssm_dt_b: l.ssm_dt_b,
                        ssm_a: l.ssm_a,
                        ssm_d: l.ssm_d,
                        ssm_out: l.ssm_out,
                        ssm_dt_norm: l.ssm_dt_norm,
                        ssm_b_norm: l.ssm_b_norm,
                        ssm_c_norm: l.ssm_c_norm,
                        wqkv: l.wqkv,
                        attn_q_norm: l.attn_q_norm,
                        attn_k_norm: l.attn_k_norm,
                        wo: l.wo,
                        attn_post_norm: l.attn_post_norm.unwrap(),
                        ffn_norm: l.ffn_norm.unwrap(),
                        ffn_down: l.ffn_down.unwrap(),
                        ffn_up: l.ffn_up.unwrap(),
                        ffn_post_norm: l.ffn_post_norm.unwrap(),
                    })
                    .collect(),
            };
            let p = graph_arch::Plamo2Params {
                attn,
                n_embd: hp.n_embd as i64,
                is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                d_conv: hp.ssm_d_conv as i64,
                d_inner: hp.ssm_d_inner as i64,
                d_state: hp.ssm_d_state as i64,
                n_heads: hp.ssm_dt_rank as i64,
                n_group: hp.ssm_n_group as i64,
                norm_eps: hp.f_norm_rms_eps,
                dt_dim: 64.max((hp.n_embd / 16) as i64),
                n_embd_r: hp.n_embd_r(),
                n_embd_s: hp.n_embd_s(),
            };
            (ForwardWeights::Plamo2(w, p), attn)
        }
        other => panic!("arch {other:?} not in batch 6b"),
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
/// bit-identical repeat on a cleared cache/state — through both FA modes so
/// wiring issues surface here rather than in the parity runs.
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

    // fresh cache + zeroed recurrent state → bit-identical first decode
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
fn synth_nemotron_loader_and_forward() {
    let spec = spec_nemotron();
    let (n, bytes) = build_file(&spec);
    // 4 model-level + 4 layers x 13 (norms 4, qkv 3, wo 2, ffn 4)
    assert_eq!(n, 4 + 4 * 13, "nemotron tensor count");
    println!(
        "nemotron synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.f_norm_eps, 1e-5);
    assert_eq!(hp.n_head_kv(0), 2, "GQA 4/2");
    assert_ne!(m.output, m.tok_embd, "untied head (nemotron.cpp:20)");
    assert!(
        m.layers[0].wo_b.is_some(),
        "optional wo_b present in the file"
    );
    assert!(m.layers[0].ffn_up_b.is_some() && m.layers[0].ffn_down_b.is_some());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "nemotron synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_grok_loader_and_forward() {
    let spec = spec_grok();
    let (n, bytes) = build_file(&spec);
    // 2 model-level tensors (tied head) + 4 layers x 12 (norm, qkv 3, wo,
    // out_norm, ffn_norm, router, gate/down/up exps, post_norm)
    assert_eq!(n, 2 + 4 * 12, "grok tensor count");
    println!("grok synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    // grok.cpp:4-12 defaults (no scale keys in the file)
    assert_eq!(hp.f_logit_scale, 0.5773502691896257);
    assert_eq!(hp.f_attn_out_scale, 0.08838834764831845);
    assert_eq!(hp.f_attn_logit_softcapping, 30.0);
    assert_eq!(hp.f_final_logit_softcapping, 0.0);
    assert_eq!(hp.yarn_beta_fast, 8.0);
    assert_eq!(hp.n_ff_exp(0), 32);
    assert_eq!(m.output, m.tok_embd, "tied head");
    assert!(
        m.layers[0].ffn_gate.is_none(),
        "no dense branch in the base file"
    );

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "grok synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_grok_dense_branch() {
    let spec = spec_grok_dense();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    // the dense branch rides alongside the MoE (grok.cpp:171-184)
    assert!(m.layers[0].ffn_gate.is_some());
    assert!(m.layers[0].ffn_down.is_some());
    assert!(m.layers[0].ffn_up.is_some());
    // layer_output_norm won over the post_ffw_norm name (grok.cpp:75-78)
    assert!(m.tensor("blk.0.layer_output_norm.weight").is_some());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "grok-dense synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_chameleon_loader_and_forward() {
    let spec = spec_chameleon();
    let (n, bytes) = build_file(&spec);
    // 2 model-level + 4 layers x 13
    assert_eq!(n, 2 + 4 * 13, "chameleon tensor count");
    println!(
        "chameleon synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert!(!m.hparams.swin_norm);
    assert_eq!(m.hparams.f_norm_eps, 1e-5, "qk-norm eps (chameleon.cpp:6)");
    assert_eq!(m.output, m.tok_embd, "tied head");

    // the img-logit suppression: a single-token decode's row 0 must carry
    // -FLT_MAX over [4, 8196) and untouched text logits elsewhere
    let mut mm = load_synth(&spec);
    let mut d = driver_for(&mut mm, false);
    let lg = d.decode(&[7], &[0]).expect("decode").to_vec();
    assert_eq!(lg[10], f32::MIN, "img token suppressed");
    assert_eq!(lg[8195], f32::MIN);
    assert!(
        lg[3].is_finite() && lg[3] != f32::MIN,
        "leading text logits kept"
    );
    assert!(
        lg[9000].is_finite() && lg[9000] != f32::MIN,
        "trailing text logits kept"
    );

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "chameleon synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_chameleon_swin_norm() {
    let spec = spec_chameleon_swin();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert!(m.hparams.swin_norm);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "chameleon-swin synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deci_loader_and_forward() {
    let spec = spec_deci();
    let (n, bytes) = build_file(&spec);
    // 2 model-level + 6 layers x 10 (norm, qkv 3, wo+bias, ffn_norm,
    // gate/down/up + 2 biases... base = norm + 3 qkv + 2 wo + 1 ffn_norm +
    // 3 ffn + 3 ffn biases = 13)
    assert_eq!(n, 2 + 6 * 13, "deci tensor count");
    println!("deci synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    assert!(
        m.layers[0].rope_freqs.is_none(),
        "no rope tensors in the base file"
    );

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deci synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_deci_mixed_layer_kinds() {
    let spec = spec_deci_mixed();
    build_file(&spec);
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    // the per-layer arrays came through the generic loader
    assert_eq!(hp.n_head(1), 2, "linear-attention layer");
    assert_eq!(hp.n_head_kv(1), 0);
    assert_eq!(hp.n_head(2), 0, "attention-free layer");
    assert_eq!(hp.n_ff(3), 0, "FFN-free layer");
    // the loader created exactly the right tensors per kind
    assert!(
        m.layers[1].wo.is_some() && m.layers[1].wq.is_none(),
        "layer 1 linear"
    );
    assert!(m.layers[2].wq.is_none() && m.layers[2].wo.is_none() && m.layers[2].ffn_gate.is_some());
    assert!(
        m.layers[3].ffn_gate.is_none() && m.layers[3].wq.is_some(),
        "layer 3 FFN-free"
    );
    assert!(m.layers[0].rope_freqs.is_some(), "rope factors present");

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "deci-mixed synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_jais_loader_and_forward() {
    let spec = spec_jais();
    let (n, bytes) = build_file(&spec);
    // 4 model-level + 4 layers x 14
    assert_eq!(n, 4 + 4 * 14, "jais tensor count");
    println!("jais synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    assert_eq!(hp.f_norm_eps, 1e-5);
    assert_eq!(
        hp.f_max_alibi_bias, 8.0,
        "ALiBi from the GGUF KV (jais.cpp:5)"
    );
    assert!(hp.use_alibi, "use_alibi flip (llama-model.cpp:1419)");
    assert_eq!(hp.rope_type, llama::hparams::LlamaRopeType::NONE);

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "jais synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_falcon_h1_loader_and_forward() {
    let spec = spec_falcon_h1();
    let (n, bytes) = build_file(&spec);
    // 2 model-level + 4 layers x 21 (ssm 8, attn 5, ffn_norm + ffn 6, biases)
    println!(
        "falcon-h1 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    // every layer is recurrent AND attentive (falcon-h1.cpp:14)
    assert!((0..4).all(|il| hp.is_recr(il)));
    assert_eq!(
        hp.n_embd_r(),
        3 * (64 + 2 * 1 * 16),
        "conv cell (d_conv-1)*conv_dim"
    );
    assert_eq!(hp.n_embd_s(), 16 * 64, "ssm cell d_state*d_inner");
    assert!(m.layers[0].ssm_norm.is_some());
    assert!(m.layers[0].ffn_norm.is_some());
    // ffn_norm has no ".weight" in the file — the tensor map pins the name
    assert!(m.tensor("blk.0.ffn_norm").is_some());

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "falcon-h1 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

#[test]
fn synth_plamo2_loader_and_forward() {
    let spec = spec_plamo2();
    let (n, bytes) = build_file(&spec);
    println!(
        "plamo2 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let m = load_synth(&spec);
    pin_tensors(&m, &spec);
    let hp = &m.hparams;
    // is_recr = n_head_kv == 0 (plamo2.cpp:18-20)
    assert_eq!(
        (0..6).map(|il| hp.is_recr(il)).collect::<Vec<_>>(),
        vec![true, true, false, true, true, false]
    );
    assert_eq!(hp.n_embd_head_k(0), 32);
    assert_eq!(
        hp.n_embd_head_v(0),
        48,
        "attention.value_length (plamo2.cpp:16)"
    );
    assert_eq!(hp.n_embd_r(), 3 * 64, "conv cell (n_group == 0)");
    assert_eq!(hp.n_embd_s(), 16 * 64);
    // the mamba/attention per-layer split
    assert!(m.layers[0].ssm_x.is_some() && m.layers[0].wqkv.is_none());
    assert!(m.layers[2].wqkv.is_some() && m.layers[2].ssm_x.is_none());
    assert_eq!(
        *m.ctx.ne(m.layers[2].wo.unwrap()),
        [192, 128, 1, 1],
        "wo = q_heads*v_dim"
    );
    assert_eq!(
        *m.ctx.ne(m.layers[0].ffn_up.unwrap()),
        [128, 128, 1, 1],
        "ffn_up 2*n_ff"
    );

    let a = smoke_forward(&spec, false);
    let b = smoke_forward(&spec, true);
    println!(
        "plamo2 synth: greedy first token {} / {} (fa off/on)",
        argmax(&a),
        argmax(&b)
    );
}

/// LlmKv round-trip of the batch's extra keys through the writer.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_grok().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("grok"));
    assert_eq!(g.get_u32("grok.expert_count"), Some(4));
    assert_eq!(g.get_u32("grok.expert_feed_forward_length"), Some(32));
    let spec = spec_jais().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_f32("jais.attention.max_alibi_bias"), Some(8.0));
    let spec = spec_deci_mixed().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    // the per-layer arrays ride as GGUF array values
    let arr = g.find_key("deci.attention.head_count_kv").unwrap();
    let Value::Array(_, items) = arr else {
        panic!("head_count_kv array")
    };
    assert_eq!(items.len(), 8);
    let spec = spec_falcon_h1().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_u32("falcon-h1.ssm.group_count"), Some(1));
    let spec = spec_plamo2().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_u32("plamo2.attention.value_length"), Some(48));
}

/// generator (#[ignore]) — the parity runs drive the release llama-cli
/// itself (batch-1 protocol; this batch's CLI arms landed with the graph)
#[test]
#[ignore = "manual: writes ~200 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch6b_write_synth() {
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
    let gguf = Gguf::open(spec_jais().path()).unwrap();
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH6B=1 ./parity/arch_batch_parity.sh nemotron grok grok-dense \
         chameleon chameleon-swin deci deci-mixed jais falcon-h1 plamo2 jais-long grok-long \
         deci-mixed-long falcon-h1-long plamo2-long"
    );
}
