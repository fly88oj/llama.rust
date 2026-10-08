//! arch_batch_e2e.rs — synthetic-GGUF verification of the arch batch landed on
//! 2026-09-24: **gpt2 / phi2 / starcoder2 / command-r / gptneox / olmo2**
//! (llama.cpp bd4f514db1).
//!
//! No local GGUF of any of these archs exists on this machine, so the whole
//! batch is verified on *synthetic* files built here with the port's byte-exact
//! GGUF writer (`ggml::gguf_write` — the same writer the LoRA / synthetic-gemma
//! tests use; the reference accepts its output, see PARITY.md).
//!
//! Layout of a synthetic file (all weights F32: the reference's CPU repack
//! buffer covers only a fixed quantized set — ggml-cpu/repack.cpp has no F16
//! entry at all — so F32 keeps both sides on the same plain path):
//!   * every `tokenizer.*` KV copied verbatim from a llama.cpp vocab fixture of
//!     the pinned tree (`models/ggml-vocab-*.gguf`). Those fixtures are the
//!     reference's own test vocabularies and the port's tokenizer is verified
//!     id-for-id against them (crates/llama/tests/tokenizer_fixtures/).
//!   * the arch's own KV: block_count / embedding_length / feed_forward_length /
//!     head counts / the norm eps / rope base, plus the arch-specific keys
//!     (`gptneox.use_parallel_residual`, `command-r.logit_scale`, olmo2's SWA
//!     pair).
//!   * exactly the tensor names + shapes `load_arch_tensors` asks for.
//!
//! Default-run tests (no reference needed): one per arch. Each writes the file,
//! loads it through `llama::model::load_model`, pins the *exact* created tensor
//! set (names + shapes + count), pins the hparams derivation, and runs a 3-token
//! decode through `DecodeContext` (finite, non-degenerate logits + bit-identical
//! repeat).
//!
//! `#[ignore]`d: `arch_batch_write_synth` writes all six files into
//! /tmp/arch-batch/ for the reference parity runs. Protocol (PARITY.md): a
//! *fresh* reference server, the FIRST request on it, temperature 0, logprobs
//! 20, cache_prompt false, `-c 512 -t 8 -fa on|off`:
//!
//! Results captured 2026-09-24 (reference build bd4f514db1, prompt "The capital
//! of France is", 16 tokens, `parity/arch_batch_parity.sh`):
//!
//! | arch     | -fa off          | -fa on           | worst \|dlogprob\| |
//! |----------|------------------|------------------|--------------------|
//! | gpt2     | 16/16 | 16/16 | 0.0001 |
//! | phi2     | 16/16 | 16/16 | 0.0001 |
//! | starcoder2 | 16/16 | 16/16 | 0.0001 |
//! | command-r  | 16/16 | 16/16 | 0.0006 / 0.0013 |
//! | gptneox    | 16/16 | 16/16 | 0.0001 |
//! | olmo2      | 16/16 | 16/16 | 0.0001 |
//!
//! plus the variants: `gptneox-synth-seq.gguf` (use_par_res = false) 16/16 and
//! `olmo2-synth-swa.gguf` 16/16 (n_swa 4096 > the 21-token context, so the
//! window never binds; the truly windowed SWA path needs `new_with_swa` and is
//! NOT verified — see PARITY.md). `command-r-synth-63.gguf` (the layer count
//! below the q/k-norm threshold) loads in both implementations.
//!
//! ```text
//! cargo test -p llama --test arch_batch_e2e -- --ignored --nocapture arch_batch_write_synth
//! for a in gpt2 phi2 starcoder2 command-r gptneox olmo2; do
//!   for fa in off on; do
//!     ./parity/run_cli_arch_parity.sh tokens /tmp/arch-batch/$a-synth.gguf "$fa" 8790 \
//!         "$a-$fa" "The capital of France is" 16
//!   done
//! done
//! ```
//!
//! The port side of that script is `./target/release/llama-cli` (the arch
//! dispatch this batch extended); the reference side is the pinned
//! `llama-server` at /home/jeffrey/llm/llama.cpp/build-rust-ref/bin.

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

/// llama.cpp's own vocab fixtures (tokenizer KV, no tensors).
const VOCAB_GPT2: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-gpt-2.gguf");
const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const VOCAB_PHI3: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-phi-3.gguf");

/// Where the synthetic files live (also what the parity runs point at).
const OUT_DIR: &str = "/tmp/arch-batch";

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

/// One synthetic architecture. Every dim is layer-0-uniform (none of the batch
/// archs has per-layer geometry) and small enough that a file is 10-30 MiB.
#[derive(Clone, Copy)]
struct SynthSpec {
    arch: &'static str,
    /// file-name suffix, so variants of a spec can coexist on disk
    suffix: &'static str,
    /// tokenizer fixture the `tokenizer.*` KV is copied from
    vocab_src: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    n_ff: i64,
    n_ctx: u32,
    /// LayerNorm archs write `attention.layer_norm_epsilon`, olmo2 the RMS one
    rms_eps: bool,
    /// `command-r.logit_scale` (command-r.cpp:4)
    logit_scale: Option<f32>,
    /// `gptneox.use_parallel_residual` (gptneox.cpp:5, required key)
    use_par_res: Option<bool>,
    /// write `output.weight` even where C falls back to the token embedding
    write_output: bool,
    /// write olmo2's `attention.sliding_window` + pattern (SWA variant)
    swa: Option<u32>,
}

impl SynthSpec {
    fn n_vocab(&self) -> i64 {
        // both fixtures' token counts (gpt-2 50257, llama-spm 32000, phi-3 32064)
        match self.vocab_src {
            VOCAB_GPT2 => 50257,
            VOCAB_PHI3 => 32064,
            _ => 32000,
        }
    }
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
        let mut s = *self;
        f(&mut s);
        s
    }
}

/// gpt2 (gpt2.cpp) — 2 layers, the real gpt-2 tokenizer (BPE, 50257), tied head.
fn spec_gpt2() -> SynthSpec {
    SynthSpec {
        arch: "gpt2",
        suffix: "",
        vocab_src: VOCAB_GPT2,
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: false,
        logit_scale: None,
        use_par_res: None,
        write_output: false, // exercises the TENSOR_DUPLICATED tie fallback
        swa: None,
    }
}

/// phi2 (phi2.cpp) — `output.weight` AND `output.bias` are required (no tie).
fn spec_phi2() -> SynthSpec {
    SynthSpec {
        arch: "phi2",
        suffix: "",
        vocab_src: VOCAB_PHI3,
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: false,
        logit_scale: None,
        use_par_res: None,
        write_output: true,
        swa: None,
    }
}

/// starcoder2 (starcoder2.cpp) — untied head present in the file.
fn spec_starcoder2() -> SynthSpec {
    SynthSpec {
        arch: "starcoder2",
        suffix: "",
        vocab_src: VOCAB_SPM,
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: false,
        logit_scale: None,
        use_par_res: None,
        write_output: true,
        swa: None,
    }
}

/// command-r (command-r.cpp) — 64 layers is the threshold that switches the
/// per-head `attn_q_norm`/`attn_k_norm` on (command-r.cpp:28-31); the layers are
/// narrow so the file stays small.
fn spec_command_r() -> SynthSpec {
    SynthSpec {
        arch: "command-r",
        suffix: "",
        vocab_src: VOCAB_SPM,
        n_layer: 64,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: false,
        logit_scale: Some(0.5),
        use_par_res: None,
        write_output: false, // C always ties: output == token_embd
        swa: None,
    }
}

/// gptneox (gptneox.cpp) — fused qkv + fused bias, `use_parallel_residual` true
/// (what the Pythia / 20B files carry).
fn spec_gptneox() -> SynthSpec {
    SynthSpec {
        arch: "gptneox",
        suffix: "",
        vocab_src: VOCAB_SPM,
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: false,
        logit_scale: None,
        use_par_res: Some(true),
        write_output: true,
        swa: None,
    }
}

/// olmo2 (olmo2.cpp) — non-SWA file (no `<arch>.attention.sliding_window` key
/// → `graph<false>`).
fn spec_olmo2() -> SynthSpec {
    SynthSpec {
        arch: "olmo2",
        suffix: "",
        vocab_src: VOCAB_SPM,
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_ff: 128,
        n_ctx: 256,
        rms_eps: true,
        logit_scale: None,
        use_par_res: None,
        write_output: true,
        swa: None,
    }
}

fn all_specs() -> [SynthSpec; 6] {
    [
        spec_gpt2(),
        spec_phi2(),
        spec_starcoder2(),
        spec_command_r(),
        spec_gptneox(),
        spec_olmo2(),
    ]
}

// ---------------------------------------------------------------------------
// per-arch tensor tables — exactly the create_tensor calls of
// src/models/<arch>.cpp load_arch_tensors, in file order
// ---------------------------------------------------------------------------

type TensorSpec = (String, Vec<i64>);

/// Weight role drives the value generator (see [`scale_of`]).
#[derive(Clone, Copy, PartialEq)]
enum Role {
    /// norm weight — near 1.0 so the synthetic model is well conditioned
    Norm,
    /// bias — small
    Bias,
    /// token embedding (shared with the head on tied archs, hence wide)
    Embd,
    /// lm head — wide, so the logits keep a top-1 margin
    Head,
    /// attention / FFN projection
    Proj,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff, n_vocab) = (spec.n_embd, spec.n_ff, spec.n_vocab());
    let n_kv = spec.n_embd_kv();
    let hd = spec.n_embd_head();
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    match spec.arch {
        // ---- gpt2.cpp:18-50 ----
        "gpt2" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!(
                "position_embd.weight",
                vec![n_embd, spec.n_ctx as i64],
                Role::Embd
            );
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            if spec.write_output {
                push!("output.weight", vec![n_embd, n_vocab], Role::Head);
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

        // ---- phi2.cpp:16-40 ----
        "phi2" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            push!("output.weight", vec![n_embd, n_vocab], Role::Head);
            push!("output.bias", vec![n_vocab], Role::Bias);
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

        // ---- starcoder2.cpp:19-52 ----
        "starcoder2" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            if spec.write_output {
                push!("output.weight", vec![n_embd, n_vocab], Role::Head);
            }
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

        // ---- command-r.cpp:16-39 ----
        "command-r" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if spec.n_layer >= 64 {
                    // per-head q/k LayerNorm, shapes {n_embd_head_k, n_head*}
                    push!(
                        format!("blk.{i}.attn_q_norm.weight"),
                        vec![hd, spec.n_head],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_k_norm.weight"),
                        vec![hd, spec.n_head_kv],
                        Role::Norm
                    );
                }
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

        // ---- gptneox.cpp:57-86 ----
        "gptneox" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Bias);
            push!("output.weight", vec![n_embd, n_vocab], Role::Head);
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

        // ---- olmo2.cpp:31-48 ----
        "olmo2" => {
            push!("token_embd.weight", vec![n_embd, n_vocab], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, n_vocab], Role::Head);
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
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![n_kv],
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
                push!(
                    format!("blk.{i}.post_ffw_norm.weight"),
                    vec![n_embd],
                    Role::Norm
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
/// `Head` are ~8x that so the logits keep a top-1 margin of ~1 logit. A random
/// model whose top-2 gap is ~0.1 logits would flip tokens on any last-bit
/// kernel difference, which would make token parity uninformative.
fn scale_of(role: Role, n_embd: i64) -> f32 {
    match role {
        Role::Norm => 1.0,
        Role::Bias => 0.02,
        Role::Embd | Role::Head => 1.0,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch");

    let src = Gguf::open(spec.vocab_src)
        .unwrap_or_else(|e| panic!("open vocab fixture {}: {e}", spec.vocab_src));
    let mut w = GgufWriter::new(32);
    for (k, v) in &src.kv {
        // the fixture's own arch keys (gpt2.* / tokenizer.* of the fixture's
        // arch) are dropped: only the tokenizer travels, the arch KV below
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, v.clone());
        }
    }

    let a = spec.arch;
    // `set_kv` takes &str; the keys are built per arch
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
    if a != "gpt2" {
        // gpt2 takes no rope at all (llama-model.cpp:2905)
        kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    }
    if let Some(v) = spec.logit_scale {
        kv!(format!("{a}.logit_scale"), Value::F32(v));
    }
    if let Some(v) = spec.use_par_res {
        kv!(format!("{a}.use_parallel_residual"), Value::Bool(v));
    }
    if let Some(n) = spec.swa {
        // olmo2.cpp:6-14 — the SWA variant; the pattern array is u32 so the
        // reference's `get_arr<uint32_t>` type check passes (meta.rs's
        // `arr_elem_u32` also accepts I32, the reference does not).
        kv!(format!("{a}.attention.sliding_window"), Value::U32(n));
        kv!(
            format!("{a}.attention.sliding_window_pattern"),
            Value::Array(
                GgufType::Uint32,
                (0..spec.n_layer)
                    .map(|i| Value::U32(u32::from(i % 2 == 0)))
                    .collect(),
            )
        );
        kv!(format!("{a}.rope.freq_base_swa"), Value::F32(500_000.0));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0x5eed_0000 ^ (a.len() as u64) ^ (spec.n_layer as u64) << 8);
    let table = tensors_for(spec);
    for ((name, ne), role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role, spec.n_embd);
        let vals: Vec<f32> = match role {
            // norm weights ride near 1.0 (an all-zero LayerNorm weight collapses
            // the model), biases near 0
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

fn spec_is_rms(m: &LlamaModel) -> bool {
    matches!(m.arch, llama::arch::LlmArch::OLMO2)
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
// shared pin
// ---------------------------------------------------------------------------

/// The exact tensor set the loader must create: every table entry, plus
/// `output.weight` *only* when the file carries one (the tied fallback resolves
/// to the existing `token_embd.weight` id, so no second name appears).
fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    let mut want: Vec<String> = tensors_for(spec).into_iter().map(|((n, _), _)| n).collect();
    if !spec.write_output {
        want.retain(|n| n != "output.weight");
    }
    want.sort();
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
    if spec.rms_eps {
        assert_eq!(m.hparams.f_norm_rms_eps, 1e-5);
    } else {
        assert_eq!(m.hparams.f_norm_eps, 1e-5);
    }
    if spec.arch != "gpt2" {
        assert_eq!(m.hparams.rope_freq_base_train, 10_000.0);
    }
    assert_eq!(m.hparams.f_logit_scale, spec.logit_scale.unwrap_or(0.0));
    assert_eq!(m.hparams.use_par_res, spec.use_par_res.unwrap_or(false));
}

// ---------------------------------------------------------------------------
// default-run tests, one per arch
// ---------------------------------------------------------------------------

#[test]
fn synth_gpt2_loader_and_forward() {
    let spec = spec_gpt2();
    let (n, bytes) = build_file(&spec);
    assert_eq!(n, 4 + 2 * 12, "gpt2 tensor count");
    println!("gpt2 synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    // non-rope arch: rope_type NONE, n_rot still derived from the head dim
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NONE);
    assert_eq!(m.hparams.n_rot(0), 16);
    // learned absolute position embedding + tied head
    assert_eq!(shape_of(&m, "position_embd.weight"), [64, 256, 1, 1]);
    assert_eq!(
        m.output, m.tok_embd,
        "absent output.weight must alias tok_embd"
    );
    assert!(m.output_norm_b.is_some(), "gpt2 carries output_norm.bias");
    assert!(m.output_b.is_none(), "gpt2 has no lm-head bias");
    assert_eq!(shape_of(&m, "blk.0.attn_qkv.weight"), [64, 128, 1, 1]);

    let w = llama::graph_arch::Gpt2ModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap(),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::Gpt2LayerWeights {
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
    };
    let p = llama::graph_arch::Gpt2Params {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Gpt2(w, p));
    println!("gpt2 synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_phi2_loader_and_forward() {
    let spec = spec_phi2();
    let (n, bytes) = build_file(&spec);
    println!("phi2 synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);
    assert_ne!(m.output, m.tok_embd, "phi2 loads a real output.weight");
    assert!(m.output_b.is_some(), "phi2 has a required lm-head bias");
    assert_eq!(shape_of(&m, "blk.0.attn_q.weight"), [64, 64, 1, 1]);
    assert_eq!(shape_of(&m, "blk.0.attn_k.weight"), [64, 32, 1, 1]);
    assert_eq!(shape_of(&m, "output.bias"), [32064, 1, 1, 1]);

    let w = llama::graph_arch::Phi2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        output_b: m.output_b.unwrap(),
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::Phi2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wq: l.wq.unwrap(),
                wq_b: l.wq_b,
                wk: l.wk.unwrap(),
                wk_b: l.wk_b,
                wv: l.wv.unwrap(),
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::Phi2Params {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Phi2(w, p));
    println!("phi2 synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_starcoder2_loader_and_forward() {
    let spec = spec_starcoder2();
    let (n, bytes) = build_file(&spec);
    println!(
        "starcoder2 synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    assert_ne!(m.output, m.tok_embd);
    let l0 = &m.layers[0];
    assert!(
        l0.attn_norm_b.is_some() && l0.ffn_norm_b.is_some(),
        "LN biases required"
    );
    assert!(l0.wo_b.is_some(), "starcoder2 requires attn_out.bias");
    assert!(
        l0.ffn_up_b.is_some() && l0.ffn_down_b.is_some(),
        "FFN biases required"
    );

    let w = llama::graph_arch::StarCoder2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::StarCoder2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wq: l.wq.unwrap(),
                wq_b: l.wq_b,
                wk: l.wk.unwrap(),
                wk_b: l.wk_b,
                wv: l.wv.unwrap(),
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::StarCoder2Params {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::StarCoder2(w, p));
    println!("starcoder2 synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_command_r_loader_and_forward() {
    let spec = spec_command_r();
    let (n, bytes) = build_file(&spec);
    println!(
        "command-r synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    // 64 layers ≥ the q/k-norm threshold (command-r.cpp:28)
    assert_eq!(m.layers.len(), 64);
    assert_eq!(shape_of(&m, "blk.0.attn_q_norm.weight"), [16, 4, 1, 1]);
    assert_eq!(shape_of(&m, "blk.0.attn_k_norm.weight"), [16, 2, 1, 1]);
    assert_eq!(m.hparams.f_logit_scale, 0.5);
    assert_eq!(
        m.output, m.tok_embd,
        "command-r always ties output to token_embd"
    );

    // the same spec one layer below the threshold: no q/k norms created
    let spec63 = spec.with(|s| {
        s.n_layer = 63;
        s.suffix = "-63";
    });
    let (n63, bytes63) = build_file(&spec63);
    println!(
        "command-r 63-layer synth: {n63} tensors, {bytes63} bytes -> {}",
        spec63.path()
    );
    let m63 = load_synth(&spec63);
    pin_tensors(&m63, &spec63);
    assert!(
        m63.layers
            .iter()
            .all(|l| l.attn_q_norm.is_none() && l.attn_k_norm.is_none()),
        "n_layer < 64 must not create q/k norms"
    );

    let w = llama::graph_arch::CommandRModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::CommandRLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wq: l.wq.unwrap(),
                wq_b: l.wq_b,
                wk: l.wk.unwrap(),
                wk_b: l.wk_b,
                wv: l.wv.unwrap(),
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::CommandRParams {
        attn: synth_attn(&m),
        logit_scale: m.hparams.f_logit_scale,
    };
    let a = smoke_forward(&mut m, ForwardWeights::CommandR(w, p));
    println!("command-r synth: greedy first token {}", argmax(&a));
}

#[test]
fn synth_gptneox_loader_and_forward() {
    let spec = spec_gptneox();
    let (n, bytes) = build_file(&spec);
    println!(
        "gptneox synth: {n} tensors, {bytes} bytes -> {}",
        spec.path()
    );
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    assert!(
        m.hparams.use_par_res,
        "use_parallel_residual read from the file"
    );
    assert_eq!(m.hparams.rope_type, llama::hparams::LlamaRopeType::NEOX);
    assert_eq!(shape_of(&m, "blk.0.attn_qkv.weight"), [64, 128, 1, 1]);
    assert_eq!(shape_of(&m, "blk.0.attn_qkv.bias"), [128, 1, 1, 1]);

    // the sequential branch (gptneox.cpp:176-191) is the same spec minus the key
    let spec_seq = spec.with(|s| {
        s.use_par_res = Some(false);
        s.suffix = "-seq";
    });
    let (n_seq, _) = build_file(&spec_seq);
    let mut m_seq = load_synth(&spec_seq);
    pin_tensors(&m_seq, &spec_seq);
    assert!(!m_seq.hparams.use_par_res);

    let build_w = |mm: &LlamaModel| llama::graph_arch::GptNeoxModelWeights {
        tok_embd: mm.tok_embd,
        output_norm: mm.output_norm,
        output_norm_b: mm.output_norm_b.unwrap(),
        output: mm.output,
        layers: mm
            .layers
            .iter()
            .map(|l| llama::graph_arch::GptNeoxLayerWeights {
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
    };
    let p = llama::graph_arch::GptNeoxParams {
        attn: synth_attn(&m),
        use_par_res: true,
    };
    let w_par = build_w(&m);
    let a = smoke_forward(&mut m, ForwardWeights::GptNeox(w_par, p));

    let p_seq = llama::graph_arch::GptNeoxParams {
        attn: synth_attn(&m_seq),
        use_par_res: false,
    };
    let w_seq = build_w(&m_seq);
    let b = smoke_forward(&mut m_seq, ForwardWeights::GptNeox(w_seq, p_seq));
    println!(
        "gptneox synth ({n} + {n_seq} tensors): par_res first token {}, seq first token {}",
        argmax(&a),
        argmax(&b)
    );
    assert_ne!(
        a, b,
        "the two use_par_res branches must not produce identical logits"
    );
}

#[test]
fn synth_olmo2_loader_and_forward() {
    let spec = spec_olmo2();
    let (n, bytes) = build_file(&spec);
    println!("olmo2 synth: {n} tensors, {bytes} bytes -> {}", spec.path());
    let mut m = load_synth(&spec);
    pin_tensors(&m, &spec);
    pin_hparams(&m, &spec);

    // no attn_norm at all (olmo2.cpp:36-40); per-head q/k norms instead
    assert!(
        m.layers.iter().all(|l| l.attn_norm.is_none()),
        "olmo2 has no attn_norm"
    );
    assert_eq!(shape_of(&m, "blk.0.attn_q_norm.weight"), [64, 1, 1, 1]);
    assert_eq!(shape_of(&m, "blk.0.attn_k_norm.weight"), [32, 1, 1, 1]);
    assert_eq!(
        shape_of(&m, "blk.0.post_attention_norm.weight"),
        [64, 1, 1, 1]
    );
    assert_eq!(shape_of(&m, "blk.0.post_ffw_norm.weight"), [64, 1, 1, 1]);
    // no sliding_window key → swa_type NONE (olmo2.cpp:16) → graph<false>
    assert_eq!(m.hparams.swa_type, llama::hparams::LlamaSwaType::NONE);
    assert_eq!(m.hparams.n_swa, 0);

    let w = llama::graph_arch::Olmo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|l| llama::graph_arch::Olmo2LayerWeights {
                wq: l.wq.unwrap(),
                wk: l.wk.unwrap(),
                wv: l.wv.unwrap(),
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    };
    let p = llama::graph_arch::Olmo2Params {
        attn: synth_attn(&m),
    };
    let a = smoke_forward(&mut m, ForwardWeights::Olmo2(w, p));
    println!("olmo2 synth: greedy first token {}", argmax(&a));
}

// ---------------------------------------------------------------------------
// olmo2 SWA hparams from metadata (load only — the SWA graph needs an iswa
// DecodeContext, see PARITY.md)
// ---------------------------------------------------------------------------

/// olmo2.cpp:6-17: `attention.sliding_window > 0` switches swa_type STANDARD,
/// runs `load_swa_pattern(ml, 4)`, copies the rope base and forces
/// `rope_freq_scale_train_swa = 1.0`; absent/smaller leaves NONE.
#[test]
fn olmo2_swa_hparams_from_metadata() {
    let base = spec_olmo2();
    build_file(&base);
    let swa = base.with(|s| {
        s.suffix = "-swa";
        s.swa = Some(4096);
    });
    let (n, _) = build_file(&swa);
    let m = load_synth(&swa);
    pin_tensors(&m, &swa);
    assert_eq!(n, tensors_for(&swa).len());

    assert_eq!(m.hparams.swa_type, llama::hparams::LlamaSwaType::STANDARD);
    assert_eq!(m.hparams.n_swa, 4096);
    assert_eq!(m.hparams.rope_freq_base_train_swa, 500_000.0);
    assert_eq!(m.hparams.rope_freq_scale_train_swa, 1.0);
    assert_eq!(m.hparams.rope_freq_base_train, 10_000.0);
    // the written pattern array (even layers SWA) wins over the default
    assert_eq!(m.hparams.is_swa(0), true);
    assert_eq!(m.hparams.is_swa(1), false);
    // ... while the plain file stays NONE
    let m0 = load_synth(&base);
    assert_eq!(m0.hparams.swa_type, llama::hparams::LlamaSwaType::NONE);
}

// ---------------------------------------------------------------------------
// generator for the reference parity runs (#[ignore])
// ---------------------------------------------------------------------------

/// Write every synthetic file into /tmp/arch-batch/ and print the prompt token
/// ids of the two tokenizers used, then the exact parity commands (module docs).
/// Ignored because those runs need the files to stay in place.
#[test]
#[ignore = "manual: writes ~100 MiB of synthetic GGUF into /tmp for the reference parity runs"]
fn arch_batch_write_synth() {
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
    for (label, path) in [("gpt2", spec_gpt2().path()), ("spm", spec_olmo2().path())] {
        let gguf = Gguf::open(&path).unwrap();
        let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
        let ids = vocab.tokenize("The capital of France is", true, true);
        println!("{label}: prompt ids {ids:?}");
    }
    println!("\nparity: ./parity/run_cli_arch_parity.sh tokens <file> off|on 8790 <tag> \"The capital of France is\" 16");
}

/// The metadata of a freshly written file (the writer's own round-trip): the
/// tokenizer KV survives the copy and the arch KV is what was asked for.
#[test]
fn synth_metadata_roundtrip() {
    let spec = spec_gpt2();
    build_file(&spec);
    let g = Gguf::open(spec.path()).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("gpt2"));
    assert_eq!(g.get_u32("gpt2.block_count"), Some(2));
    assert_eq!(g.get_u32("gpt2.embedding_length"), Some(64));
    assert_eq!(g.get_f32("gpt2.attention.layer_norm_epsilon"), Some(1e-5));
    assert_eq!(g.get_str("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(
        g.find_tensor("position_embd.weight").map(|t| t.ne),
        Some([64, 256, 1, 1])
    );
    // the fixture's own arch keys must NOT travel into an arch file
    assert_eq!(g.get_u32("gpt2.attention.head_count"), Some(4));
}
