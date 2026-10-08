//! arch_batch2_e2e.rs — synthetic-GGUF verification of the arch batch landed on
//! 2026-09-25: **codeshell / orion / olmo / xverse / internlm2 / exaone /
//! gemma(v1) / falcon** (llama.cpp bd4f514db1).
//!
//! Same protocol as the first batch (`crates/llama/tests/arch_batch_e2e.rs`,
//! documented in PARITY.md): no local GGUF of any of these archs exists on this
//! machine, so each is verified on a *synthetic* file built here with the port's
//! byte-exact GGUF writer (`ggml::gguf_write`), which the reference
//! `llama-server` accepts. Layout of a file:
//!   * every `tokenizer.*` KV copied verbatim from a llama.cpp vocab fixture
//!     (`models/ggml-vocab-llama-spm.gguf` — 32000 SPM tokens),
//!   * the arch's own KV (dims + norm eps + rope base, plus `olmo`'s optional
//!     `attention.clamp_kqv`),
//!   * exactly the tensor names + shapes that arch's `load_arch_tensors` asks
//!     for, all F32 (the reference's CPU repack buffer has no F16 entry, so F32
//!     keeps both sides on the plain path).
//!
//! Default-run tests (no reference needed): one per arch. Each writes the file,
//! loads it, pins the *exact* created tensor set (names + shapes + count), pins
//! the hparams derivation, and runs a 3-token decode through the arch builder
//! (finite, non-degenerate logits + bit-identical repeat).
//!
//! `#[ignore]`d: `arch_batch2_write_synth` writes the eight files into
//! /tmp/arch-batch2/ for the reference parity runs (`parity/arch_batch_parity.sh`
//! with an `ARCH_BATCH2=1` env, see PARITY.md).
//!
//! The parity protocol per (arch, -fa on|off) cell: a *fresh* reference
//! `llama-server`, the FIRST request on it, `-c 512 -t 8 -fa on|off`,
//! temperature 0, logprobs 20, cache_prompt false, 16 tokens, prompt
//! "The capital of France is" — compared against `./target/release/llama-cli`
//! on the same file.

use std::sync::Arc;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::kv_cache::KvCache;
use llama::model::{load_model, LlamaModel};

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

/// llama.cpp's own SPM vocab fixture (tokenizer KV only, no tensors). Every arch
/// of this batch reads it fine: the reference's pre-tokenizer selection has no
/// arch-dependent default for any of the eight (`llama-vocab.cpp` switches on
/// the arch only for the bpe/spm *model* key, which the fixture carries).
const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;

/// Where the synthetic files live (also what the parity runs point at).
const OUT_DIR: &str = "/tmp/arch-batch2";

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SynthSpec {
    arch: &'static str,
    /// file-name suffix, so variants of a spec can coexist on disk
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    n_ff: i64,
    n_ctx: u32,
    /// RMS eps key instead of the LayerNorm one
    rms_eps: bool,
    /// write `output.weight` (the arch's head; required where the C requires it)
    write_output: bool,
    /// write `token_embd.weight` (codeshell's reverse-tie variant: false)
    write_tok_embd: bool,
    /// write falcon's `attn_norm_2` pair (the 40B shape)
    attn_norm_2: bool,
    /// write exaone's `blk.0.rope_freqs.weight`
    rope_freqs: bool,
    /// write `attention.clamp_kqv` (olmo.cpp:5)
    clamp_kqv: Option<f32>,
}

impl SynthSpec {
    fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }
    fn n_embd_kv(&self) -> i64 {
        self.n_head_kv * self.n_embd_head()
    }
    fn n_rot(&self) -> i64 {
        self.n_embd_head()
    }
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut s = *self;
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
        write_tok_embd: true,
        attn_norm_2: false,
        rope_freqs: false,
        clamp_kqv: None,
    }
}

fn spec_codeshell() -> SynthSpec {
    base("codeshell", false) // codeshell.cpp:4 layer_norm_epsilon
}

fn spec_orion() -> SynthSpec {
    base("orion", false) // orion.cpp:4 layer_norm_epsilon
}

fn spec_olmo() -> SynthSpec {
    // olmo.cpp:21-25 — the head is optional and ties to tok_embd
    base("olmo", false).with(|s| s.write_output = false)
}

fn spec_xverse() -> SynthSpec {
    base("xverse", true) // xverse.cpp:4 layer_norm_rms_epsilon
}

fn spec_internlm2() -> SynthSpec {
    base("internlm2", true)
}

fn spec_exaone() -> SynthSpec {
    base("exaone", true).with(|s| s.rope_freqs = true)
}

fn spec_gemma1() -> SynthSpec {
    // gemma.cpp:20 — the head is always the duplicated token embedding
    base("gemma", true).with(|s| s.write_output = false)
}

fn spec_falcon() -> SynthSpec {
    base("falcon", false).with(|s| s.write_output = true)
}

fn all_specs() -> [SynthSpec; 8] {
    [
        spec_codeshell(),
        spec_orion(),
        spec_olmo(),
        spec_xverse(),
        spec_internlm2(),
        spec_exaone(),
        spec_gemma1(),
        spec_falcon(),
    ]
}

// ---------------------------------------------------------------------------
// per-arch tensor tables — exactly the create_tensor calls of
// src/models/<arch>.cpp load_arch_tensors, in file order
// ---------------------------------------------------------------------------

type TensorSpec = (String, Vec<i64>);

#[derive(Clone, Copy, PartialEq)]
enum Role {
    /// norm weight — near 1.0 so the synthetic model is well conditioned
    Norm,
    /// bias — small
    Bias,
    /// token embedding (shared with the head on tied archs)
    Embd,
    /// lm head
    Head,
    /// attention / FFN projection
    Proj,
    /// rope freq factor (≈1.0, the realistic scale)
    Freq,
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
        // ---- codeshell.cpp:15-45 ----
        "codeshell" => {
            if spec.write_tok_embd {
                push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            }
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

        // ---- orion.cpp:15-36 ----
        "orion" => {
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
                // no attn_output.bias in an orion file
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.ffn_norm.bias"), vec![n_embd], Role::Bias);
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

        // ---- olmo.cpp:18-36 ----
        "olmo" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            if spec.write_output {
                push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            }
            for i in 0..spec.n_layer {
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

        // ---- xverse.cpp:17-34 / internlm2.cpp:16-37 (same tensor set) ----
        "xverse" | "internlm2" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            if spec.write_output {
                // both archs *require* the head (xverse.cpp:20, internlm2.cpp:21);
                // the flag lets a variant pin that the loader refuses a headless file
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

        // ---- exaone.cpp:15-39 ----
        "exaone" => {
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
                if spec.rope_freqs && i == 0 {
                    // exaone.cpp:35 — `{n_rot/2}` freq factors. NOTE the name:
                    // llama-arch.cpp:438's template for LLM_TENSOR_ROPE_FREQS is
                    // "rope_freqs" (no `blk.%d.`), so the file carries a single
                    // model-level `rope_freqs.weight` regardless of the per-layer
                    // request (llama-arch.cpp:1021 `::format` only substitutes
                    // `%d` when the template has one) — the i != 0 requests are
                    // TENSOR_DUPLICATED and reuse it
                    push!(
                        "rope_freqs.weight".to_string(),
                        vec![spec.n_rot() / 2],
                        Role::Freq
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

        // ---- gemma.cpp:16-34 ----
        "gemma" => {
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

        // ---- falcon.cpp:16-43 ----
        "falcon" => {
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
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
                if spec.attn_norm_2 {
                    push!(
                        format!("blk.{i}.attn_norm_2.weight"),
                        vec![n_embd],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_norm_2.bias"),
                        vec![n_embd],
                        Role::Bias
                    );
                }
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, n_embd + 2 * n_kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![n_embd, n_embd],
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
// writer
// ---------------------------------------------------------------------------

/// splitmix64 → f32 in [-1, 1): deterministic across runs and machines.
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

/// Value scale per role. `Proj` is 1/sqrt(n_embd) (a real init scale); `Embd` /
/// `Head` are ~8x that so the logits keep a top-1 margin of ~1 logit (a random
/// model whose top-2 gap is ~0.1 logits would flip tokens on any last-bit
/// kernel difference, making token parity uninformative).
fn scale_of(role: Role, n_embd: i64) -> f32 {
    match role {
        Role::Norm => 1.0,
        Role::Bias => 0.02,
        Role::Embd | Role::Head => 1.0,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        // rope freq factors multiply the inverse rope frequencies; keep them in
        // a realistic band that still differs from 1.0 so a builder that drops
        // the tensor (src[2]) diverges visibly
        Role::Freq => 1.0,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Build one synthetic GGUF at `spec.path()`. Returns (n_tensors, file bytes).
fn build_file(spec: &SynthSpec) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch2");

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

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ (a.len() as u64) ^ ((spec.n_layer as u64) << 8));
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = match role {
            // norm weights ride near 1.0 (an all-zero LayerNorm weight collapses
            // the model), biases near 0, and rope freq factors near but not
            // exactly 1
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            Role::Freq => (0..n).map(|_| 0.9 + 0.2 * rng.next()).collect(),
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

/// (Re)build the file and load it. Always rebuilding keeps the on-disk variant
/// in sync with the spec (a variant changes the KV or the tensor count, so a
/// file left from an earlier run must not be reused).
fn load_synth(spec: &SynthSpec) -> LlamaModel {
    build_file(spec);
    let gguf = Gguf::open(spec.path()).expect("open synth");
    let f = std::fs::File::open(spec.path()).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

/// The created tensor names, sorted — the pinned contract.
fn tensor_names(m: &LlamaModel) -> Vec<String> {
    let mut v: Vec<String> = m.tensors.keys().cloned().collect();
    v.sort();
    v
}

fn shape_of(m: &LlamaModel, name: &str) -> [i64; 4] {
    let t = m
        .tensor(name)
        .unwrap_or_else(|| panic!("tensor {name} missing"));
    *m.ctx.ne(t)
}

/// `AttnParams` of a synthetic model, derived the way llama-cli derives them
/// (`attn_params` + the per-arch `f_norm_eps` override), so the default tests
/// reach the builders through the same numbers the parity runs use.
fn synth_attn(m: &LlamaModel) -> AttnParams {
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
        norm_eps: if spec_is_rms(m) {
            hp.f_norm_rms_eps
        } else {
            hp.f_norm_eps
        },
        use_flash_attn: false,
    }
}

/// Which of the batch-2 archs take `f_norm_rms_eps` (the CLI arms mirror this).
fn spec_is_rms(m: &LlamaModel) -> bool {
    use llama::arch::LlmArch::*;
    matches!(m.arch, XVERSE | INTERNLM2 | EXAONE | GEMMA)
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

/// 3-token prefill + one more step through the arch builder: finite logits with
/// a real spread, and a bit-identical repeat on a cleared cache.
fn smoke_forward(m: &mut LlamaModel, weights: ForwardWeights) -> Vec<f32> {
    let attn = synth_attn(m);
    let n_layer = m.layers.len();
    let n_k = m.n_embd_k_gqa_max() as i64;
    let n_v = m.n_embd_v_gqa_max() as i64;
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    let mut kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 64);
    let mut dctx = DecodeContext::new_with(gctx, weights, attn, 64, 4, 8);

    let toks = [3i32, 17, 42];
    let pos: Vec<i32> = (0..3).collect();
    let a = dctx.decode(&toks, &pos).expect("prefill").to_vec();
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{}: non-finite logits",
        m.arch.name()
    );
    let spread = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - a.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}: logits degenerate (spread {spread})",
        m.arch.name()
    );

    let next = argmax(&a);
    let b = dctx.decode(&[next], &[3]).expect("step").to_vec();
    assert!(b.iter().all(|v| v.is_finite()));

    dctx.kv.clear();
    let c = dctx.decode(&toks, &pos).expect("prefill 2").to_vec();
    assert_eq!(a, c, "{}: prefill not deterministic", m.arch.name());
    kv.clear();
    a
}

// ---------------------------------------------------------------------------
// shared pins
// ---------------------------------------------------------------------------

/// The exact tensor set the loader must create = the file's tensors (every
/// declared tensor is consumed by `done_getting_tensors`); the `TENSOR_DUPLICATED`
/// re-requests create no extra *names*.
fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let mut want: Vec<String> = tensors_for(spec).into_iter().map(|((n, _), _)| n).collect();
    want.sort();
    want.dedup();
    let got = tensor_names(m);
    assert_eq!(got, want, "{}: created tensor set mismatch", spec.arch);
}

fn pin_hparams(m: &LlamaModel, spec: &SynthSpec) {
    assert_eq!(m.arch.name(), spec.arch);
    assert_eq!(m.hparams.n_embd as i64, spec.n_embd);
    assert_eq!(m.hparams.n_layer() as usize, spec.n_layer);
    assert_eq!(m.hparams.n_head(0) as i64, spec.n_head);
    assert_eq!(m.hparams.n_head_kv(0) as i64, spec.n_head_kv);
    assert_eq!(m.hparams.n_ff(0) as i64, spec.n_ff);
    assert_eq!(m.hparams.n_ctx_train, spec.n_ctx);
    assert_eq!(m.hparams.n_embd_head_k(0) as i64, spec.n_embd_head());
    assert_eq!(m.hparams.n_rot(0) as i64, spec.n_rot());
    if spec.rms_eps {
        assert_eq!(m.hparams.f_norm_rms_eps, 1e-5);
    } else {
        assert_eq!(m.hparams.f_norm_eps, 1e-5);
    }
    assert_eq!(m.hparams.rope_freq_base_train, 10_000.0);
    assert_eq!(m.hparams.f_clamp_kqv, spec.clamp_kqv.unwrap_or(0.0));
}

/// The rope-type switch of llama-model.cpp:2920-3060 for the batch's archs.
fn pin_rope_type(m: &LlamaModel, neox: bool) {
    use llama::hparams::LlamaRopeType;
    let want = if neox {
        LlamaRopeType::NEOX
    } else {
        LlamaRopeType::NORM
    };
    assert_eq!(m.hparams.rope_type, want, "{}: rope type", m.arch.name());
}

// ---------------------------------------------------------------------------
// default-run tests, one per arch
// ---------------------------------------------------------------------------

#[test]
fn synth_codeshell_loader_and_forward() {
    let spec = spec_codeshell();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 13, "codeshell tensor count");
    println!(
        "codeshell synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, true); // llama-model.cpp:2988

    assert_ne!(m.tok_embd, m.output);
    assert!(m.tensor("blk.0.attn_q.weight").is_some(), "separate qkv");
    assert!(m.tensor("blk.0.attn_qkv.weight").is_none(), "no fused qkv");
    assert!(m.tensor("blk.0.attn_q.bias").is_none(), "no qkv biases");
    assert!(m.output_norm_b.is_some() && m.layers[0].wo_b.is_some());
    assert_eq!(shape_of(&m, "blk.0.attn_q.weight"), [64, 64, 1, 1]);
    assert_eq!(shape_of(&m, "blk.0.attn_k.weight"), [64, 32, 1, 1]);

    // reverse tie variant (codeshell.cpp:15-20): no token_embd.weight, so the
    // embedding comes off output.weight
    let tie = spec.with(|s| {
        s.suffix = "-tie";
        s.write_tok_embd = false;
    });
    let (n_tie, _) = build_file(&tie);
    let m_tie = load_synth(&tie);
    pin_tensors(&m_tie, &tie);
    assert_eq!(m_tie.ctx.name(m_tie.tok_embd), "output.weight");
    assert_eq!(n_tie, tensors_for(&tie).len());

    let w = llama::graph_arch::CodeshellModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::CodeshellLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
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
    };
    let p = llama::graph_arch::CodeshellParams {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Codeshell(w, p));
    println!("codeshell synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_orion_loader_and_forward() {
    let spec = spec_orion();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 11, "orion tensor count");
    println!("orion synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, true); // llama-model.cpp:3005 (NEOX list)

    assert!(m.layers[0].wo_b.is_none(), "orion has no attn_output.bias");
    assert!(m.output_b.is_none(), "orion head has no bias");
    assert!(m.layers[0].attn_norm_b.is_some() && m.layers[0].ffn_norm_b.is_some());
    assert_eq!(shape_of(&m, "blk.0.ffn_gate.weight"), [64, 128, 1, 1]);

    let w = llama::graph_arch::OrionModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::OrionLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::OrionParams {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Orion(w, p));
    println!("orion synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_olmo_loader_and_forward() {
    let spec = spec_olmo();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 1 + 2 * 7, "olmo tensor count (no norm tensors at all)");
    println!("olmo synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, false); // llama-model.cpp:2941 (NORM list)

    // no norm tensors exist, tied head
    assert!(
        m.tensor("output_norm.weight").is_none(),
        "olmo has no output_norm"
    );
    assert!(m
        .layers
        .iter()
        .all(|l| l.attn_norm.is_none() && l.ffn_norm.is_none()));
    assert_eq!(
        m.output, m.tok_embd,
        "absent output.weight ties to tok_embd"
    );
    assert_eq!(m.hparams.f_clamp_kqv, 0.0);

    // the head variant: `output.weight` present + `attention.clamp_kqv`
    let with_head = spec.with(|s| {
        s.suffix = "-head";
        s.write_output = true;
        s.clamp_kqv = Some(8.0);
    });
    let (n_head, _) = build_file(&with_head);
    let mut m_head = load_synth(&with_head);
    pin_tensors(&m_head, &with_head);
    pin_hparams(&m_head, &with_head);
    assert_ne!(m_head.output, m_head.tok_embd);
    assert_eq!(m_head.hparams.f_clamp_kqv, 8.0);
    assert_eq!(n_head, tensors_for(&with_head).len());

    let build_w = |mm: &LlamaModel| llama::graph_arch::OlmoModelWeights {
        tok_embd: mm.tok_embd,
        output: mm.output,
        layers: mm
            .layers
            .iter()
            .map(|l| llama::graph_arch::OlmoLayerWeights {
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::OlmoParams {
        attn: synth_attn(&m),
        f_clamp_kqv: 0.0,
    };
    let w = build_w(&m);
    let a = smoke_forward(&mut m, ForwardWeights::Olmo(w, p));

    // the clamped variant must produce different logits (the clamp is live)
    let p_c = llama::graph_arch::OlmoParams {
        attn: synth_attn(&m_head),
        f_clamp_kqv: 8.0,
    };
    let w_c = build_w(&m_head);
    let b = smoke_forward(&mut m_head, ForwardWeights::Olmo(w_c, p_c));
    println!(
        "olmo synth ({n} + {n_head} tensors): plain first token {}, clamp_kqv=8 first token {}",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(a, b, "the f_clamp_kqv branch must change the logits");
}

#[test]
fn synth_xverse_loader_and_forward() {
    let spec = spec_xverse();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 2 * 9, "xverse tensor count");
    println!(
        "xverse synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, false); // llama-model.cpp:2937 (NORM list)

    assert!(m.layers[0].attn_norm_b.is_none(), "RMS norms have no bias");
    assert!(m.layers[0].ffn_norm.is_some());
    assert!(m.output_norm_b.is_none(), "xverse output_norm has no bias");
    assert_eq!(shape_of(&m, "blk.0.attn_output.weight"), [64, 64, 1, 1]);

    let w = llama::graph_arch::XverseModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::XverseLayerWeights {
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
    };
    let p = llama::graph_arch::XverseParams {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Xverse(w, p));
    println!("xverse synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_internlm2_loader_and_forward() {
    let spec = spec_internlm2();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 3 + 2 * 9, "internlm2 tensor count");
    println!(
        "internlm2 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, false);

    // internlm2.cpp:21 — the head is required, never tied
    assert_ne!(m.output, m.tok_embd);
    assert!(m.tensor("output.weight").is_some());

    // the same file minus output.weight must FAIL (no tie fallback in C)
    let no_head = spec.with(|s| {
        s.suffix = "-nohead";
        s.write_output = false;
    });
    build_file(&no_head);
    let gguf = Gguf::open(no_head.path()).unwrap();
    let f = std::fs::File::open(no_head.path()).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    let err = load_model(&gguf, mmap).unwrap_err();
    assert!(err.contains("output.weight"), "unexpected error: {err}");

    let w = llama::graph_arch::Internlm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::Internlm2LayerWeights {
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
    };
    let p = llama::graph_arch::Internlm2Params {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Internlm2(w, p));
    println!("internlm2 synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_exaone_loader_and_forward() {
    let spec = spec_exaone();
    let (n, bytes) = build_file(&spec);
    assert_eq!(
        n,
        3 + 9 + 9 + 1,
        "exaone tensor count (+ the single rope_freqs)"
    );
    println!(
        "exaone synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, true); // llama-model.cpp:3015 (NEOX list)

    // the model-level `rope_freqs.weight` is read once and duplicated into
    // every layer's slot (llama-arch.cpp:438 template has no `blk.%d.`)
    assert_eq!(shape_of(&m, "rope_freqs.weight"), [8, 1, 1, 1]);
    assert_eq!(
        m.layers[0].rope_freqs, m.layers[1].rope_freqs,
        "dup request reuses it"
    );
    assert!(m.layers[0].rope_freqs.is_some());

    let build_w = |mm: &LlamaModel| llama::graph_arch::ExaoneModelWeights {
        tok_embd: mm.tok_embd,
        output_norm: mm.output_norm,
        output: mm.output,
        layers: mm
            .layers
            .iter()
            .map(|l| llama::graph_arch::ExaoneLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::ExaoneParams {
        attn: synth_attn(&m),
    };
    let wa = build_w(&m);
    let a = smoke_forward(&mut m, ForwardWeights::Exaone(wa, p));

    // the same spec with the freq factors dropped: the graph must move
    let plain = spec.with(|s| {
        s.suffix = "-norope";
        s.rope_freqs = false;
    });
    let mut m2 = load_synth(&plain);
    pin_tensors(&m2, &plain);
    assert!(m2.layers.iter().all(|l| l.rope_freqs.is_none()));
    let p2 = llama::graph_arch::ExaoneParams {
        attn: synth_attn(&m2),
    };
    let wb = build_w(&m2);
    let b = smoke_forward(&mut m2, ForwardWeights::Exaone(wb, p2));
    println!(
        "exaone synth: rope_freqs first token {}, without {} (must differ)",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(
        a, b,
        "the rope_freqs tensor must reach ggml_rope_ext src[2]"
    );
}

#[test]
fn synth_gemma1_loader_and_forward() {
    let spec = spec_gemma1();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 2 + 2 * 9, "gemma(v1) tensor count");
    println!("gemma synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, true); // llama-model.cpp:3018 (NEOX list)

    // gemma.cpp:20 — the head is the token embedding, never a separate tensor
    assert_eq!(m.output, m.tok_embd);
    assert!(m.tensor("output.weight").is_none());
    assert!(m.output_norm_b.is_none());
    assert!(m
        .layers
        .iter()
        .all(|l| l.attn_post_norm.is_none() && l.ffn_post_norm.is_none()));
    assert_eq!(shape_of(&m, "blk.0.attn_output.weight"), [64, 64, 1, 1]);

    let w = llama::graph_arch::Gemma1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::Gemma1LayerWeights {
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
    };
    let attn = synth_attn(&m);
    let attention_scale = 1.0 / (attn.n_embd_head_v as f32).sqrt();
    let p = llama::graph_arch::Gemma1Params {
        attn,
        attention_scale,
    };
    let a = smoke_forward(&mut m, ForwardWeights::Gemma1(w, p));
    println!("gemma synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_falcon_loader_and_forward() {
    let spec = spec_falcon();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 6, "falcon tensor count");
    println!(
        "falcon synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);
    pin_rope_type(&m, true); // llama-model.cpp:2979 (NEOX list)

    // no ffn_norm / ffn_gate in the file; the FFN runs off attn_norm
    assert!(m
        .layers
        .iter()
        .all(|l| l.ffn_norm.is_none() && l.ffn_gate.is_none()));
    assert!(
        m.layers[0].attn_norm_2.is_none(),
        "7B shape: no attn_norm_2"
    );
    assert_eq!(shape_of(&m, "blk.0.attn_qkv.weight"), [64, 128, 1, 1]);
    assert!(
        m.tensor("blk.0.attn_qkv.bias").is_none(),
        "fused qkv has no bias"
    );

    // the 40B shape: attn_norm_2 present (falcon.cpp:35-36, 79-88)
    let forty = spec.with(|s| {
        s.suffix = "-40b";
        s.attn_norm_2 = true;
    });
    let (n40, _) = build_file(&forty);
    let mut m40 = load_synth(&forty);
    pin_tensors(&m40, &forty);
    assert!(m40.layers[0].attn_norm_2.is_some());
    assert!(m40.layers[0].attn_norm_2_b.is_some());
    assert_eq!(n40, tensors_for(&forty).len());

    // the tie variant: no output.weight → the head is the token embedding
    let tie = spec.with(|s| {
        s.suffix = "-tie";
        s.write_output = false;
    });
    let m_tie = load_synth(&tie);
    pin_tensors(&m_tie, &tie);
    assert_eq!(m_tie.output, m_tie.tok_embd);

    let p = llama::graph_arch::FalconParams {
        attn: synth_attn(&m),
    };
    let wa = falcon_weights(&m);
    let a = smoke_forward(&mut m, ForwardWeights::Falcon(wa, p));

    let p40 = llama::graph_arch::FalconParams {
        attn: synth_attn(&m40),
    };
    let wb = falcon_weights(&m40);
    let b = smoke_forward(&mut m40, ForwardWeights::Falcon(wb, p40));
    println!(
        "falcon synth ({n} + {n40} tensors): 7B first token {}, 40B first token {}",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(a, b, "the attn_norm_2 path must change the logits");
}

// ---------------------------------------------------------------------------
// falcon context probe (#[ignore]) — attribution tooling, see PARITY.md
// ---------------------------------------------------------------------------

/// falcon.cpp:13-44 weight assembly (shared by the loader test and the probe).
fn falcon_weights(m: &LlamaModel) -> llama::graph_arch::FalconModelWeights {
    llama::graph_arch::FalconModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| llama::graph_arch::FalconLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                attn_norm_2: l.attn_norm_2,
                attn_norm_2_b: l.attn_norm_2_b,
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// Teacher-force the port on the *reference's* falcon token sequence and print
/// the port's per-step argmax + the logprob of the reference's token at each
/// step. This separates "the port's logits are wrong at step k" from "the
/// sampling loop is wrong": a distribution shift shows up as a growing
/// |Δlogprob| on the reference's own tokens, while a loop/bookkeeping bug shows
/// up as the port's argmax following the reference's sequence shifted.
///
/// The reference side is `parity/arch_batch2_falcon_ref.txt` (produced by
/// `parity/falcon_probe.sh`); the comparison is
/// `python3 parity/falcon_probe_cmp.py`.
/// Teacher-force the port on the *reference's* falcon token sequence and on an
/// arbitrary list of contexts, comparing against the reference's own
/// distributions. This is how the batch-2 falcon "divergence" was attributed:
///
/// The pinned server omits the `completion_probabilities` entry of a token whose
/// piece is not valid UTF-8 (a byte token such as `<0xDE>`): `tokens_predicted`
/// counts it, the reported list does not, and its bytes are merged into the next
/// entry's `bytes` field (server-task.cpp:285-290 `validate_utf8` truncates the
/// piece). The port's own sequence does contain it, so a naive id-by-id
/// comparison misaligns and shows a "flipped token" (here: token 225 = `<0xDE>`
/// between `▁Carter` and `▁Moz`) — `parity/arch_batch_cmp.py` now aligns the
/// ids as a subsequence, and the parity cells report 16/16 with the one
/// reference-omitted entry printed.
///
/// Pass A below is the exact parity prompt/token flow; `A step k` prints the
/// port's argmax and the reference's token's logprob at every step (the port
/// reproduces the reference's whole sequence, byte token included). Pass B
/// replays arbitrary contexts from /tmp/arch-batch2/falcon-probe-contexts.txt
/// (one context per line) through a *fresh* model and a single full prefill;
/// `parity/falcon_probe.sh` asks the reference for the same contexts as
/// `prompt = [token ids]` requests, and `parity/falcon_probe_cmp.py` pairs the
/// `CTX`/`REF` lines with the byte-token alignment applied.
#[test]
#[ignore = "manual: needs the reference distribution probe (parity/falcon_probe.sh)"]
fn falcon_teacher_force_probe() {
    let spec = spec_falcon();
    // pieces of the ids the attribution keeps hitting (225 is the omitted byte
    // token; the `▁Carter` family is the pre-divergence context)
    {
        let g = Gguf::open(spec.path()).unwrap_or_else(|_| {
            build_file(&spec);
            Gguf::open(spec.path()).unwrap()
        });
        let v = llama::vocab::Vocab::load(&g).expect("vocab");
        for id in [22264usize, 225, 18129, 18763, 18592, 28729, 1] {
            println!(
                "  piece id {id:5} -> {:?}",
                v.id_to_token.get(id).map(|t| t.text.as_str())
            );
        }
    }
    let prompt = [1i32, 450, 7483, 310, 3444, 338];
    let r#ref: Vec<i32> = std::fs::read_to_string("/tmp/arch-batch2/falcon-ref-tokens.txt")
        .expect("run parity/falcon_probe.sh first")
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();

    // ---- pass A: incremental (prefill the prompt, then one token per step) ----
    {
        let mut m = load_synth(&spec);
        let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
        let (n_k, n_v, n_layer) = (
            m.n_embd_k_gqa_max() as i64,
            m.n_embd_v_gqa_max() as i64,
            m.layers.len(),
        );
        let _kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
        let attn = synth_attn(&m);
        let w = falcon_weights(&m);
        let p = llama::graph_arch::FalconParams { attn };
        let mut dctx =
            DecodeContext::new_with(gctx, ForwardWeights::Falcon(w, p), attn, 512, 8, 512);
        let mut lg = dctx
            .decode(&prompt, &(0..prompt.len() as i32).collect::<Vec<_>>())
            .unwrap()
            .to_vec();
        let mut pos = prompt.len() as i32;
        for (i, &want) in r#ref.iter().enumerate() {
            println!(
                "A step {i}: argmax {} (lp {:.4})  ref token {want} (port lp {:.4}, logit {:.3})",
                argmax(&lg),
                softmax_logprob(&lg, argmax(&lg) as usize),
                softmax_logprob(&lg, want as usize),
                lg[want as usize],
            );
            lg = dctx.decode(&[want], &[pos]).unwrap().to_vec();
            pos += 1;
        }
    }

    // ---- pass B: arbitrary contexts, each a fresh model + one full prefill.
    // File format: one context per line, space-separated token ids. The same
    // file is fed to the reference by parity/falcon_probe.sh, which prints
    // `REF <i> ...` lines the comparator (parity/falcon_probe_cmp.py) pairs with
    // the `CTX <i> ...` lines below.
    let path = "/tmp/arch-batch2/falcon-probe-contexts.txt";
    let Ok(text) = std::fs::read_to_string(path) else {
        println!("(no {path} — skipping pass B)");
        return;
    };
    for (i, line) in text.lines().enumerate() {
        let toks: Vec<i32> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let mut m = load_synth(&spec);
        let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
        let (n_k, n_v, n_layer) = (
            m.n_embd_k_gqa_max() as i64,
            m.n_embd_v_gqa_max() as i64,
            m.layers.len(),
        );
        let _kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
        let attn = synth_attn(&m);
        let w = falcon_weights(&m);
        let p = llama::graph_arch::FalconParams { attn };
        let mut dctx =
            DecodeContext::new_with(gctx, ForwardWeights::Falcon(w, p), attn, 512, 8, 512);
        let posv: Vec<i32> = (0..toks.len() as i32).collect();
        let lg = dctx.decode(&toks, &posv).unwrap().to_vec();
        let top = top_k(&lg, 5);
        println!(
            "CTX {i} n={} argmax {} top5 {}",
            toks.len(),
            argmax(&lg),
            top.iter()
                .map(|&(id, lp)| format!("({id},{lp:.4})"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // the port's own next step: when the port's first token is a byte token
        // the reference omits it from completion_probabilities, so this line is
        // what the reference's first *reported* entry must match
        let lg2 = dctx
            .decode(&[argmax(&lg)], &[toks.len() as i32])
            .unwrap()
            .to_vec();
        let top2 = top_k(&lg2, 5);
        println!(
            "CTXN {i} n={} argmax {} top5 {}",
            toks.len() + 1,
            argmax(&lg2),
            top2.iter()
                .map(|&(id, lp)| format!("({id},{lp:.4})"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

/// The `k` highest-logit entries as (id, logprob), descending.
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<usize> = (0..logits.len()).collect();
    v.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    v.truncate(k);
    v.into_iter()
        .map(|i| (i, softmax_logprob(logits, i)))
        .collect()
}

fn softmax_logprob(logits: &[f32], token: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse: f64 = logits
        .iter()
        .map(|&v| (v - max) as f64)
        .map(f64::exp)
        .sum::<f64>()
        .ln();
    (logits[token] as f64 - max as f64) - lse
}

// ---------------------------------------------------------------------------
// generator for the reference parity runs (#[ignore])
// ---------------------------------------------------------------------------

/// Write every synthetic file into /tmp/arch-batch2/ and print the prompt token
/// ids, then the exact parity commands (module docs). Ignored because those runs
/// need the files to stay in place.
#[test]
#[ignore = "manual: writes ~100 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch2_write_synth() {
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
    // the token ids both sides must agree on (the reference prints the same ids
    // from /tokenize)
    let gguf = Gguf::open(spec_olmo().path()).unwrap();
    let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("The capital of France is", true, true);
    println!("spm: prompt ids {ids:?}");
    println!(
        "\nparity: ARCH_BATCH2=1 ./parity/arch_batch_parity.sh codeshell orion olmo xverse \
         internlm2 exaone gemma falcon"
    );
}

/// The metadata of a freshly written file round-trips through the writer and
/// carries no fixture KV beyond `tokenizer.*`. Uses its own `-meta` suffixes:
/// tests run in parallel and share /tmp/arch-batch2 (a second writer on the same
/// path would let the reader see a half-written file).
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_falcon().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("falcon"));
    assert_eq!(g.get_u32("falcon.block_count"), Some(2));
    assert_eq!(g.get_f32("falcon.attention.layer_norm_epsilon"), Some(1e-5));
    assert_eq!(g.get_str("tokenizer.ggml.model"), Some("llama"));
    assert_eq!(
        g.find_tensor("blk.0.attn_qkv.weight").map(|t| t.ne),
        Some([64, 128, 1, 1])
    );
    assert_eq!(g.get_u32("falcon.attention.head_count"), Some(4));
    // the olmo clamp_kqv key travels when asked for
    let olmo = spec_olmo().with(|s| {
        s.suffix = "-meta";
        s.clamp_kqv = Some(8.0);
    });
    build_file(&olmo);
    let g = Gguf::open(olmo.path()).unwrap();
    assert_eq!(g.get_f32("olmo.attention.clamp_kqv"), Some(8.0));
}

/// GgufType is used by the fixture copy path (kept import-explicit).
#[test]
fn synth_writer_type_check() {
    let spec = spec_gemma1().with(|s| s.suffix = "-meta");
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    let ty = g.find_key("tokenizer.ggml.tokens").map(|v| v.type_());
    assert!(
        matches!(ty, Some(GgufType::Array)),
        "tokens array survived: {ty:?}"
    );
}
