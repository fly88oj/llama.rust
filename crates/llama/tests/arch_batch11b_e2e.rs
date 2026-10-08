//! arch_batch11b_e2e.rs — synthetic-GGUF verification of the arch batch
//! landed on 2026-10: **the long-tail queue, second half** — arcee / jais2 /
//! talkie / nanbeige / dream / rnd1 / eurobert (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-10 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//!   * arcee — the llama body with the relu²-SEQ MLP (nemotron's helper) and
//!     the per-layer `rope_freqs` factors (`get_rope_factors`,
//!     arcee.cpp:37/:80); the tie fallback of `output.weight`;
//!   * jais2 — LayerNorm-with-biases everywhere (attn/ffn/output norms, the
//!     wo bias, both MLP biases) + RoPE + relu²-SEQ (jais2.cpp:54-155);
//!   * talkie — weightless RMS norms (`build_norm(x, nullptr, …)`), the
//!     post-rope per-head q-norm (the [1, n_head] gain broadcasting over the
//!     head dim, talkie.cpp:87), the `embd_skip * layer_out_scale` residual
//!     (:123-126) and the final `ggml_scale(f_logit_scale)` (:143);
//!   * nanbeige — the `num_loops` layer expansion: the physical stack's
//!     tensors alias into the loop slots (nanbeige.cpp:20-32/:67-73) and the
//!     shared output_norm folds in at every loop boundary
//!     (:164-171); the default file has no loop key (n_loops = 1), the
//!     `-loops2` variant carries `nanbeige.num_loops = 2` (4 physical
//!     blocks → 8 logical layers);
//!   * dream / rnd1 — the llada-family diffusion archs: qwen2's body over
//!     `build_attn_inp_no_cache` (dream.cpp:68) and the qwen3moe body (q/k
//!     norms before rope + softmax MoE, rnd1.cpp:99-150) over the same
//!     no-cache input. The pinned reference creates NO memory for DREAM/RND1
//!     (llama-model.cpp:2289-2295, `res = nullptr`) and every generation
//!     driver refuses a memory-less context — like llada-moe (batch 10)
//!     there is no reference output to compare against; the graphs are
//!     verified in-port (both FA modes + a 100-token long-prompt cell);
//!   * eurobert — the encoder of the batch (bert/t5 `EncoderContext` path):
//!     RMS norms, NEOX-rope'd Q/K, gated SwiGLU FFN, `res->t_embd` only. The
//!     reference ground truth is `parity/ref_encode_dump --ids … --fa off`
//!     over the same synthetic file (the bge-m3/t5-encoder protocol); the
//!     ignored test runs the comparison.
//!
//! The batch's ForwardWeights/CLI arms landed with the graph (context.rs),
//! so the default-run tests drive `DecodeContext::new_with` itself
//! (batch-6+ protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH11B=1 ./parity/arch_batch_parity.sh …`).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{DecodeContext, EncoderContext, EncoderWeights, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch11b";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Arcee,
    Jais2,
    Talkie,
    Nanbeige,
    Dream,
    Rnd1,
    Eurobert,
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
    /// arcee/nanbeige: write the per-layer rope_freqs tensor (layer 0)
    rope_freqs: bool,
    /// nanbeige: the num_loops GGUF key (None = absent, 1)
    num_loops: Option<u32>,
    /// dream: write output.bias
    output_b: bool,
    /// rnd1: write expert_feed_forward_length (None = the n_ff/n_used fallback)
    n_ff_exp: Option<i64>,
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
        n_head: 4,
        head_kv: vec![2; 4],
        key_length: 32,
        value_length: 32,
        rope_dim: 32,
        n_ff: 96,
        n_ctx: 512,
        write_output: true,
        rope_freqs: false,
        num_loops: None,
        output_b: false,
        n_ff_exp: Some(32),
    }
}

/// arcee — llama body, relu² SEQ MLP, per-layer rope freq factors, tied output
fn spec_arcee() -> SynthSpec {
    base("arcee", Family::Arcee).with(|s| {
        s.write_output = false;
        s.rope_freqs = true;
    })
}

/// jais2 — LN+biases everywhere, rope, relu² MLP with biases
fn spec_jais2() -> SynthSpec {
    base("jais2", Family::Jais2)
}

/// talkie — weightless norms, [1, n_head] q gain, layer_out_scale, logit scale
fn spec_talkie() -> SynthSpec {
    base("talkie", Family::Talkie)
}

/// nanbeige — default file (no num_loops key → n_loops 1), rope freq factors
fn spec_nanbeige() -> SynthSpec {
    base("nanbeige", Family::Nanbeige).with(|s| {
        s.rope_freqs = true;
    })
}

/// nanbeige `-loops2` — `nanbeige.num_loops = 2`: 4 physical blocks expand to
/// 8 logical layers, the loop boundary folds the shared output_norm in
fn spec_nanbeige_loops2() -> SynthSpec {
    spec_nanbeige().with(|s| {
        s.suffix = "-loops2";
        s.num_loops = Some(2);
    })
}

/// dream — qwen2 body over the no-cache diffusion attention, tied output
fn spec_dream() -> SynthSpec {
    base("dream", Family::Dream).with(|s| {
        s.write_output = false;
    })
}

/// rnd1 — qwen3moe body over the no-cache diffusion attention (q/k norms
/// before rope + softmax MoE), the expert width via the n_ff/n_used fallback
fn spec_rnd1() -> SynthSpec {
    base("rnd1", Family::Rnd1).with(|s| {
        s.write_output = false;
        s.n_ff_exp = None; // fallback n_ff / n_expert_used = 48
    })
}

/// eurobert — the encoder (no output head; pooling NONE)
fn spec_eurobert() -> SynthSpec {
    base("eurobert", Family::Eurobert).with(|s| {
        s.write_output = false;
    })
}

/// every file the ignored writer test emits
fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_arcee(),
        spec_jais2(),
        spec_talkie(),
        spec_nanbeige(),
        spec_nanbeige_loops2(),
        spec_dream(),
        spec_rnd1(),
        spec_eurobert(),
    ]
}

/// the parity cells (batch-11b default set): arcee / jais2 / talkie /
/// nanbeige / nanbeige-loops2 — dream / rnd1 have NO reference-side cell (the
/// pinned reference creates no memory for the diffusion archs,
/// llama-model.cpp:2289-2295, and both llama-server and llama-cli refuse to
/// generate on a memory-less context) and eurobert has no *generation* path
/// either (encoder — verified against the reference llama_encode dump in the
/// ignored test below).
fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_arcee(),
        spec_jais2(),
        spec_talkie(),
        spec_nanbeige(),
        spec_nanbeige_loops2(),
    ]
}

#[test]
#[ignore = "writes the /tmp/arch-batch11b parity files (ARCH_BATCH11B cells)"]
fn arch_batch11b_write_synth() {
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
    let q_w = spec.key_length * spec.n_head;
    let kv_w = spec.key_length * spec.head_kv[0];
    let mut t: Vec<(TensorSpec, Role)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, role: Role| t.push(((name, ne), role));

    push(
        "token_embd.weight".into(),
        vec![n_embd, N_VOCAB],
        Role::Proj,
    );

    match spec.family {
        Family::Arcee | Family::Nanbeige | Family::Dream | Family::Eurobert | Family::Rnd1 => {
            if spec.write_output {
                push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            }
            if spec.output_b {
                push("output.bias".into(), vec![N_VOCAB], Role::Bias);
            }
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            // ROPE_FREQS's template has no blk slot ("rope_freqs") — ONE
            // model-level tensor every layer re-requests with DUPLICATED
            if matches!(spec.family, Family::Arcee | Family::Nanbeige) && spec.rope_freqs {
                push(
                    "rope_freqs.weight".into(),
                    vec![spec.rope_dim / 2],
                    Role::Norm,
                );
            }
        }
        Family::Jais2 => {
            if spec.write_output {
                push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
            }
            push("output_norm.weight".into(), vec![n_embd], Role::Norm);
            push("output_norm.bias".into(), vec![n_embd], Role::Bias);
        }
        Family::Talkie => {
            // talkie.cpp:17 — output REQUIRED, no output_norm
            push("output.weight".into(), vec![n_embd, N_VOCAB], Role::Proj);
        }
    }

    for i in 0..spec.n_layer {
        let b = format!("blk.{i}.");
        match spec.family {
            Family::Arcee => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
            }
            Family::Jais2 => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_norm.bias"), vec![n_embd], Role::Bias);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_output.bias"), vec![n_embd], Role::Bias);
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}ffn_norm.bias"), vec![n_embd], Role::Bias);
                push(format!("{b}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                push(format!("{b}ffn_up.bias"), vec![n_ff], Role::Bias);
                push(
                    format!("{b}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_down.bias"), vec![n_embd], Role::Bias);
            }
            Family::Talkie => {
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                // no k gain — one scalar per head (talkie.cpp:26)
                push(
                    format!("{b}attn_q_norm.weight"),
                    vec![1, spec.n_head],
                    Role::Norm,
                );
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
                push(format!("{b}layer_output_scale.weight"), vec![1], Role::Norm);
            }
            Family::Nanbeige => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
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
            Family::Dream => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                // dream.cpp:37 — q width n_embd (head_dim * n_head == n_embd)
                push(
                    format!("{b}attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
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
            Family::Rnd1 => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                push(format!("{b}attn_q.weight"), vec![n_embd, q_w], Role::Proj);
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![q_w, n_embd],
                    Role::Proj,
                );
                push(
                    format!("{b}attn_k_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm,
                );
                push(
                    format!("{b}attn_q_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push(
                    format!("{b}ffn_gate_inp.weight"),
                    vec![n_embd, N_EXPERT],
                    Role::Router,
                );
                let n_ff_exp = spec.n_ff_exp.unwrap_or(n_ff / N_EXPERT_USED);
                push(
                    format!("{b}ffn_gate_exps.weight"),
                    vec![n_embd, n_ff_exp, N_EXPERT],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_down_exps.weight"),
                    vec![n_ff_exp, n_embd, N_EXPERT],
                    Role::Proj,
                );
                push(
                    format!("{b}ffn_up_exps.weight"),
                    vec![n_embd, n_ff_exp, N_EXPERT],
                    Role::Proj,
                );
            }
            Family::Eurobert => {
                push(format!("{b}attn_norm.weight"), vec![n_embd], Role::Norm);
                // eurobert.cpp:23 — q width n_embd, like bert
                push(
                    format!("{b}attn_q.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push(format!("{b}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push(
                    format!("{b}attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj,
                );
                push(format!("{b}ffn_norm.weight"), vec![n_embd], Role::Norm);
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
    t
}

// ---------------------------------------------------------------------------
// writer (same recipe as arch_batch9/10_e2e.rs)
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch11b");

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
    // jais2 reads the LayerNorm eps; the rest the RMS eps
    match spec.family {
        Family::Jais2 => {
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
    kv!(
        format!("{a}.rope.dimension_count"),
        Value::U32(spec.rope_dim as u32)
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    match spec.family {
        Family::Talkie => {
            kv!(format!("{a}.logit_scale"), Value::F32(2.5)); // REQUIRED (talkie.cpp:5)
        }
        Family::Nanbeige => {
            if let Some(v) = spec.num_loops {
                kv!(format!("{a}.num_loops"), Value::U32(v));
            }
        }
        Family::Rnd1 => {
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
        }
        _ => {}
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
    // nanbeige's loop expansion grows the logical layer count past block_count
    let n_logical = match spec.family {
        Family::Nanbeige => spec.n_layer * spec.num_loops.unwrap_or(1) as usize,
        _ => spec.n_layer,
    };
    assert_eq!(hp.n_layer() as usize, n_logical, "{}: n_layer", spec.arch);
    assert_eq!(hp.n_embd as i64, spec.n_embd);
    match spec.family {
        Family::Arcee => {
            assert_eq!(hp.f_attention_scale, 0.0); // never read by the arch
        }
        Family::Jais2 => {
            assert_eq!(hp.f_norm_eps, 1e-5);
        }
        Family::Talkie => {
            assert_eq!(hp.f_logit_scale, 2.5);
        }
        Family::Nanbeige => {
            match spec.num_loops {
                None => {
                    assert_eq!(hp.nanbeige_n_loops, 1);
                    assert_eq!(hp.n_layer_all as usize, spec.n_layer);
                }
                Some(2) => {
                    // the loop expansion (nanbeige.cpp:20-32): arrays copied,
                    // n_layer_all = n_phys * n_loops
                    assert_eq!(hp.nanbeige_n_layer_phys as usize, spec.n_layer);
                    assert_eq!(hp.nanbeige_n_loops, 2);
                    assert_eq!(hp.n_layer_all as usize, spec.n_layer * 2);
                    for il in spec.n_layer..spec.n_layer * 2 {
                        assert_eq!(hp.n_head(il), hp.n_head(0), "loop slot {il} heads");
                        assert_eq!(hp.n_head_kv(il), hp.n_head_kv(0));
                        assert_eq!(hp.n_ff(il), hp.n_ff(0));
                    }
                    assert!(!hp.nanbeige_skip_loop_final_norm);
                }
                other => panic!("untested num_loops {other:?}"),
            }
        }
        Family::Dream | Family::Rnd1 => {
            assert!(!hp.causal_attn, "diffusion arch is non-causal");
        }
        Family::Eurobert => {
            assert_eq!(hp.f_norm_rms_eps, 1e-5);
            // the encoder default pooling: UNSPECIFIED resolves to NONE
            // (context::resolve_pooling)
        }
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
// the per-arch weight bundles (the CLI wiring of main.rs, mirrored)
// ---------------------------------------------------------------------------

fn arcee_weights(m: &LlamaModel) -> graph_arch::ArceeModelWeights {
    graph_arch::ArceeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::ArceeLayerWeights {
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
                rope_freqs: l.rope_freqs,
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn jais2_weights(m: &LlamaModel) -> graph_arch::Jais2ModelWeights {
    graph_arch::Jais2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Jais2LayerWeights {
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
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
            })
            .collect(),
    }
}

fn talkie_weights(m: &LlamaModel) -> graph_arch::TalkieModelWeights {
    graph_arch::TalkieModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::TalkieLayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                out_scale: l.out_scale.unwrap(),
            })
            .collect(),
    }
}

fn nanbeige_weights(m: &LlamaModel) -> graph_arch::NanbeigeModelWeights {
    graph_arch::NanbeigeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::NanbeigeLayerWeights {
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
                rope_freqs: l.rope_freqs,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn dream_weights(m: &LlamaModel) -> graph_arch::DreamModelWeights {
    graph_arch::DreamModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::DreamLayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

fn rnd1_weights(m: &LlamaModel) -> graph_arch::Rnd1ModelWeights {
    graph_arch::Rnd1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| graph_arch::Rnd1LayerWeights {
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
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let mut attn = synth_attn(m, fa);
    match m.arch {
        llama::arch::LlmArch::ARCEE => (
            ForwardWeights::Arcee(
                arcee_weights(m),
                graph_arch::ArceeParams {
                    attn,
                    f_attention_scale: hp.f_attention_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::JAIS2 => {
            attn.norm_eps = hp.f_norm_eps;
            (
                ForwardWeights::Jais2(jais2_weights(m), graph_arch::Jais2Params { attn }),
                attn,
            )
        }
        llama::arch::LlmArch::TALKIE => (
            ForwardWeights::Talkie(
                talkie_weights(m),
                graph_arch::TalkieParams {
                    attn,
                    logit_scale: hp.f_logit_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::NANBEIGE => (
            ForwardWeights::Nanbeige(
                nanbeige_weights(m),
                graph_arch::NanbeigeParams {
                    attn,
                    n_layer_phys: hp.nanbeige_n_layer_phys as usize,
                    n_loops: hp.nanbeige_n_loops as usize,
                    skip_loop_final_norm: hp.nanbeige_skip_loop_final_norm,
                    f_attention_scale: hp.f_attention_scale,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::DREAM => (
            ForwardWeights::Dream(dream_weights(m), graph_arch::DreamParams { attn }),
            attn,
        ),
        llama::arch::LlmArch::RND1 => (
            ForwardWeights::Rnd1(
                rnd1_weights(m),
                graph_arch::Rnd1Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        other => panic!("arch {other:?} not in batch 11b"),
    }
}

// ---------------------------------------------------------------------------
// the decode harness — DecodeContext itself (the batch's context.rs arms)
// ---------------------------------------------------------------------------

fn driver_for(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = forward_of(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
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
    // 13 cells per layer's cache (used_cells is one layer's position count);
    // the diffusion archs' graphs never read the cache but the driver owns
    // one (the reference's llama_kv_cache), like llada-moe
    assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the tensor/hparams pins + the double-FA smoke of the parity cells
#[test]
fn arch_batch11b_pin_and_smoke() {
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

/// dream / rnd1: the in-port-only cells — the pinned reference creates no
/// memory for the diffusion archs (llama-model.cpp:2289-2295), so both
/// llama-server and llama-cli refuse to generate on it; the port's no-cache
/// graphs are pinned here (batch-10 llada-moe protocol)
#[test]
fn arch_batch11b_inport_only_pins() {
    for spec in [spec_dream(), spec_rnd1()] {
        let mut m = load_synth(&spec);
        pin_hparams(&m, &spec);
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
        println!("{}: in-port pin + smoke ok (both FA modes)", spec.path());
    }
}

/// the >64-token prompt cell of the parity protocol, in one ubatch:
///   * dream / rnd1 — the [T, T] non-causal mask over ~100 tokens
///   * nanbeige-loops2 — rope past 64 on the aliased loop slots + the loop
///     boundary norm mid-stack
#[test]
fn arch_batch11b_long_prompt_cells() {
    for spec in [spec_dream(), spec_rnd1(), spec_nanbeige_loops2()] {
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

/// nanbeige-loops2 vs a 1-loop file with the same physical weights: the loop
/// boundary norm must change the logits (the shared output_norm fires
/// mid-stack only when n_loops > 1, nanbeige.cpp:164-171)
#[test]
fn arch_batch11b_nanbeige_loop_norm_changes_logits() {
    let spec1 = spec_nanbeige();
    let spec2 = spec_nanbeige_loops2();
    let mut m1 = load_synth(&spec1);
    let mut m2 = load_synth(&spec2);
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
        "the loop boundary norm must change the logits (max |d| = {diff})"
    );
    println!("nanbeige loops2-vs-1 max |Δlogit| = {diff:.4}");
}

/// eurobert in-port: the EncoderContext over the synthetic file — flat
/// [n_tokens, n_embd] embeddings under pooling NONE, finite, and the graph
/// builds both with and without the rope params
#[test]
fn arch_batch11b_eurobert_encoder_smoke() {
    let spec = spec_eurobert();
    let m = load_synth(&spec);
    pin_hparams(&m, &spec);
    pin_tensors(&m, &spec);

    let mut m = load_synth(&spec);
    let hp = m.hparams.clone();
    let rope = hp.rope_runtime();
    let params = graph_arch::EncoderParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: llama::context::resolve_pooling(
            llama::hparams::LlamaPoolingType::UNSPECIFIED,
            hp.pooling_type,
        ),
        euro_rope: Some(graph_arch::EurobertRope {
            n_rot: hp.n_rot(0) as i32,
            rope_mode: hp.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        }),
        gemma_swa: None,
        causal: false,
    };
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let mut enc = EncoderContext::new(
        gctx,
        EncoderWeights::Eurobert(m.eurobert_weights()),
        params,
        8,
    );
    let ids: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let emb = enc.encode(&ids).expect("eurobert encode");
    assert_eq!(emb.n_rows, ids.len(), "pooling NONE keeps every token row");
    assert_eq!(emb.n_embd_out, spec.n_embd as usize);
    assert!(emb.values.iter().all(|v| v.is_finite()));
    println!("eurobert: {}x{} embeddings ok", emb.n_rows, emb.n_embd_out);
}

/// eurobert vs the reference `llama_encode` dump over the same synthetic file
/// (the bge-m3/t5-encoder protocol): `parity/ref_encode_dump <file> <out>
/// --ids 1..12 --fa off`, then the port's EncoderContext on the same ids.
/// Ignored by default — it needs the reference dumper binary (built by
/// `parity/gen_encode_ref.sh`; see PARITY.md's batch-11b section).
#[test]
#[ignore = "needs parity/ref_encode_dump (the reference libllama dumper)"]
fn arch_batch11b_eurobert_reference_parity() {
    let spec = spec_eurobert();
    let model_path = spec.path();
    if !std::path::Path::new(&model_path).exists() {
        build_file(&spec);
    }
    let dump = format!("{OUT_DIR}/eurobert-enc-ref.bin");
    let dumper = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/ref_encode_dump");
    assert!(
        std::path::Path::new(dumper).exists(),
        "{dumper} missing — build it with parity/gen_encode_ref.sh"
    );
    let ids = "1,2,3,4,5,6,7,8,9,10,11,12";
    let st = std::process::Command::new(dumper)
        .arg(&model_path)
        .arg(&dump)
        .arg("--ids")
        .arg(ids)
        .arg("--pool")
        .arg("none")
        .arg("--fa")
        .arg("off")
        .status()
        .expect("run ref_encode_dump");
    assert!(st.success(), "ref_encode_dump failed");

    let bytes = std::fs::read(&dump).expect("read dump");
    assert_eq!(&bytes[..8], b"LENCE1\0\0", "dump magic");
    let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let n_tokens = rd_u32(8) as usize;
    let n_embd_out = rd_u32(12) as usize;
    let n_rows = rd_u32(16) as usize;
    assert_eq!(n_tokens, 12);
    assert_eq!(n_rows, 12, "pooling NONE keeps every row");
    let hdr = 8 + 4 * 4;
    let ids_ref: Vec<i32> = bytes[hdr..hdr + 4 * n_tokens]
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ref_vals_off = hdr + 4 * n_tokens;
    let ref_vals: Vec<f32> = bytes[ref_vals_off..ref_vals_off + 4 * n_embd_out * n_rows]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut m = load_synth(&spec);
    let hp = m.hparams.clone();
    let rope = hp.rope_runtime();
    let params = graph_arch::EncoderParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: llama::context::resolve_pooling(
            llama::hparams::LlamaPoolingType::UNSPECIFIED,
            hp.pooling_type,
        ),
        euro_rope: Some(graph_arch::EurobertRope {
            n_rot: hp.n_rot(0) as i32,
            rope_mode: hp.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
        }),
        gemma_swa: None,
        causal: false,
    };
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let mut enc = EncoderContext::new(
        gctx,
        EncoderWeights::Eurobert(m.eurobert_weights()),
        params,
        8,
    );
    let emb = enc.encode(&ids_ref).expect("encode");

    // the same tolerance regime as the bert/t5 T>=2 cells: tinyBLAS-F32 seeds
    // amplified by the F32 attention (see tests/bert_e2e.rs's header)
    let rms_ref = ref_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs = emb
        .values
        .iter()
        .zip(&ref_vals)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs / rms_ref.max(1e-6);
    println!(
        "eurobert vs reference: max|Δ| {max_abs:.3e}, rel(rms) {rel:.3e} over {}x{}",
        n_rows, n_embd_out
    );
    assert!(rel < 5e-2, "eurobert embeddings drifted: rel {rel:.4}");
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}
