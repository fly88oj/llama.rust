//! arch_batch14_e2e.rs — synthetic-GGUF verification of the arch batch landed
//! on 2026-10: **the P0 new-mechanism queue** — the RWKV family (`rwkv6` /
//! `rwkv6qwen2` over the rwkv6-base helpers, `rwkv7` / `arwkv7` over the
//! rwkv7-base helpers) and `gemma3n` (per-layer embeddings + altup/laurel)
//! (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-13: no local GGUF of these archs exists, so
//! each is verified on a *synthetic* file built with the port's byte-exact
//! GGUF writer — `tokenizer.*` KV copied verbatim from the llama SPM vocab
//! fixture, the arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//! The RWKV archs are pure-recurrent (`llm_arch_is_recurrent`): the memory is
//! `llama_memory_recurrent` alone — the port's `RecurrentState` conv cell
//! carries the token-shift state (hparams.n_embd_r = token_shift_count *
//! n_embd) and the SSM cell the WKV state (n_embd_s = n_embd * wkv_head_size)
//! per layer. The three fused reference kernels (GGML_OP_RWKV_WKV6 /
//! GGML_OP_GATED_LINEAR_ATTN / GGML_OP_RWKV_WKV7) are not in the port's ggml
//! yet — the graphs run the composed per-token scans (integrator items; see
//! graph_arch.rs's batch-14 header), so the parity criterion is the
//! generation-level token/logprob match like every batch.
//!
//! gemma3n carries the KV-reuse layers (n_layer_kv_from_start = 20 hardcoded,
//! gemma3n.cpp:7): layers >= 20 attend the earlier layer's cache rows
//! (llama-model.cpp:2658-2666), so the synthetic file has 22 layers.
//!
//! Default-run tests (no reference needed): write the file, load it, pin the
//! created tensor set / hparams, and run a prefill + decode step through
//! `DecodeContext` (both FA modes — RWKV is attention-free so the modes
//! degenerate, gemma3n exercises both branches of its attention).
//!
//! `#[ignore]`d:
//!   * `arch_batch14_write_synth` — writes the files into /tmp/arch-batch14/
//!   * the reference half is `ARCH_BATCH14=1 ./parity/arch_batch_parity.sh`
//!     (batch-12 protocol: llama-cli itself against the fresh reference
//!     server's first /completion answer).

use ggml::gguf_write::GgufWriter;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::hparams::LlamaSwaType;
use llama::kv_cache::SwaCacheSpec;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch14";

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SynthSpec {
    /// the GGUF arch name
    arch: &'static str,
    suffix: &'static str,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    n_embd_head: i64,
    n_ff: i64,
    n_ctx: u32,
    write_output: bool,
    // RWKV keys (rwkv6.cpp:3-11 / rwkv7.cpp:3-11)
    wkv_head_size: i64,
    time_mix_extra_dim: i64,
    time_decay_extra_dim: i64,
    token_shift_count: i64,
    rescale_every_n_layers: i64,
    n_lora_decay: i64,
    n_lora_iclr: i64,
    n_lora_value_res_mix: i64,
    n_lora_gate: i64,
    /// rwkv6: write the fused lerp (true) or the separate w/k/v/r/g lerps
    /// (the "backward compatibility" branch, rwkv6-base.cpp:78-91)
    rwkv6_fused_lerp: bool,
    /// rwkv7: write the required g1/g2 + 6-plane lerp_fused (true) or the
    /// arwkv7 optional-gate 5-plane layout
    rwkv7_gating: bool,
    /// gemma3n attention keys
    gemma3n: bool,
    /// RNG seed offset for the synthetic weights (fixture-stability knob)
    seed: u64,
    /// damp the rwkv7 a/v lora pair to Small scale (fixture-stability knob —
    /// the sa·b rank-1 injection amplifies the reference's own thread noise
    /// on random weights)
    damp_lora: bool,
}

impl SynthSpec {
    fn path(&self) -> String {
        format!("{OUT_DIR}/{}-synth{}.gguf", self.arch, self.suffix)
    }
    fn n_embd_r(&self) -> i64 {
        self.token_shift_count * self.n_embd
    }
    fn n_embd_s(&self) -> i64 {
        self.n_embd * self.wkv_head_size
    }
}

/// rwkv6 — the World-v6 layout: LN0, LLM_NORM pairs, channel mix,
/// token_shift_count 2, fused lerps
fn spec_rwkv6() -> SynthSpec {
    SynthSpec {
        arch: "rwkv6",
        suffix: "",
        n_layer: 2,
        n_embd: 64,
        n_head: 0,
        n_head_kv: 0,
        n_embd_head: 0,
        n_ff: 96,
        n_ctx: 512,
        write_output: true,
        wkv_head_size: 16,
        time_mix_extra_dim: 32,
        time_decay_extra_dim: 32,
        token_shift_count: 2,
        rescale_every_n_layers: 0,
        n_lora_decay: 0,
        n_lora_iclr: 0,
        n_lora_value_res_mix: 0,
        n_lora_gate: 0,
        rwkv6_fused_lerp: true,
        rwkv7_gating: false,
        gemma3n: false,
        seed: 0,
        damp_lora: false,
    }
}

/// rwkv6-legacy — the separate w/k/v/r/g lerps (the rwkv6-base.cpp:78-91
/// branch) + rescale_every_n_layers = 1 (the 0.5 scale path, rwkv6.cpp:164-166)
fn spec_rwkv6_legacy() -> SynthSpec {
    let mut s = spec_rwkv6();
    s.suffix = "-legacy";
    s.rwkv6_fused_lerp = false;
    s.rescale_every_n_layers = 1;
    s
}

/// rwkv6qwen2 — the "qwen2-style" variant: RMS norms, GLA time mix
/// (time_mix_first absent), SwiGLU FFN, token_shift_count 1, head_count_kv 2
/// exercising the k/v repeat of rwkv6-base.cpp:110-117
fn spec_rwkv6qwen2() -> SynthSpec {
    SynthSpec {
        arch: "rwkv6qwen2",
        suffix: "",
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_embd_head: 0,
        n_ff: 96,
        n_ctx: 512,
        write_output: false,
        wkv_head_size: 16,
        time_mix_extra_dim: 32,
        time_decay_extra_dim: 32,
        token_shift_count: 1,
        rescale_every_n_layers: 0,
        n_lora_decay: 0,
        n_lora_iclr: 0,
        n_lora_value_res_mix: 0,
        n_lora_gate: 0,
        rwkv6_fused_lerp: true,
        rwkv7_gating: false,
        gemma3n: false,
        seed: 0,
        damp_lora: false,
    }
}

/// rwkv7 — the w/a/v lora triples, required gating pair, 6-plane lerp_fused,
/// channel mix, token_shift_count 2
fn spec_rwkv7() -> SynthSpec {
    SynthSpec {
        arch: "rwkv7",
        suffix: "",
        n_layer: 2,
        n_embd: 64,
        n_head: 0,
        n_head_kv: 0,
        n_embd_head: 0,
        n_ff: 96,
        n_ctx: 512,
        write_output: true,
        wkv_head_size: 16,
        time_mix_extra_dim: 0,
        time_decay_extra_dim: 0,
        token_shift_count: 2,
        rescale_every_n_layers: 0,
        n_lora_decay: 16,
        n_lora_iclr: 16,
        n_lora_value_res_mix: 16,
        n_lora_gate: 16,
        rwkv6_fused_lerp: true,
        rwkv7_gating: true,
        gemma3n: false,
        seed: 0,
        // fixture stability (see spec_arwkv7): damped a/v loras keep the
        // 72-token -long recurrence from amplifying rounding noise
        damp_lora: true,
    }
}

/// arwkv7 — the "a" variant: RMS norms, SwiGLU FFN, token_shift_count 1,
/// NO gating pair (the 5-plane lerp_fused of arwkv7.cpp:88-93's fallback) and
/// no ln/ln_b
fn spec_arwkv7() -> SynthSpec {
    let mut s = spec_rwkv7();
    s.arch = "arwkv7";
    s.n_head = 0;
    s.n_head_kv = 0;
    s.token_shift_count = 1;
    s.write_output = true;
    s.rwkv7_gating = false;
    // fixture stability: the sa·b rank-1 state injection amplifies the
    // reference's own run-to-run thread noise on random weights (its top
    // token flips across fresh servers) — damped a/v loras keep the
    // recurrence contractive
    s.damp_lora = true;
    s.seed = 0xB14D;
    s
}

/// gemma3n — 22 layers (the first 20 own KV, 20/21 reuse), the 4:1 SWA
/// pattern of load_swa_pattern(ml, 5), per-layer embeddings + altup/laurel
fn spec_gemma3n() -> SynthSpec {
    SynthSpec {
        arch: "gemma3n",
        suffix: "",
        n_layer: 22,
        n_embd: 64,
        n_head: 4,
        n_head_kv: 2,
        n_embd_head: 16,
        n_ff: 96,
        n_ctx: 512,
        write_output: false,
        wkv_head_size: 0,
        time_mix_extra_dim: 0,
        time_decay_extra_dim: 0,
        token_shift_count: 0,
        rescale_every_n_layers: 0,
        n_lora_decay: 0,
        n_lora_iclr: 0,
        n_lora_value_res_mix: 0,
        n_lora_gate: 0,
        rwkv6_fused_lerp: false,
        rwkv7_gating: false,
        gemma3n: true,
        seed: 0,
        damp_lora: false,
    }
}

fn parity_specs() -> Vec<SynthSpec> {
    vec![
        spec_rwkv6(),
        spec_rwkv6qwen2(),
        spec_rwkv7(),
        spec_arwkv7(),
        spec_gemma3n(),
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
    /// ~1-mean small values (lerps/decays/routers/coefs — keep the exp()s
    /// and sigmoid()s in their linear zones)
    Small,
    Embd,
    Head,
    Proj,
}

fn tensors_for(spec: &SynthSpec) -> Vec<(TensorSpec, Role)> {
    let (n_embd, n_ff) = (spec.n_embd, spec.n_ff);
    let mut v: Vec<(TensorSpec, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push((($name.to_string(), $ne), $role))
        };
    }

    match spec.arch {
        // ---- rwkv6.cpp:29-86 (+ the -legacy lerp variant) ----
        "rwkv6" => {
            let hs = spec.wkv_head_size;
            let tmed = spec.time_mix_extra_dim;
            let tded = spec.time_decay_extra_dim;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("token_embd_norm.weight", vec![n_embd], Role::Norm);
            push!("token_embd_norm.bias", vec![n_embd], Role::Norm);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output_norm.bias", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            for i in 0..spec.n_layer {
                let p = format!("blk.{i}.");
                push!(format!("{p}attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("{p}attn_norm.bias"), vec![n_embd], Role::Norm);
                push!(format!("{p}attn_norm_2.weight"), vec![n_embd], Role::Norm);
                push!(format!("{p}attn_norm_2.bias"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}time_mix_w1.weight"),
                    vec![n_embd, tmed * 5],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_w2.weight"),
                    vec![tmed, n_embd, 5],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_lerp_x.weight"),
                    vec![n_embd, 1, 1],
                    Role::Small
                );
                if spec.rwkv6_fused_lerp {
                    push!(
                        format!("{p}time_mix_lerp_fused.weight"),
                        vec![n_embd, 1, 1, 5],
                        Role::Small
                    );
                } else {
                    push!(
                        format!("{p}time_mix_lerp_w.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                    push!(
                        format!("{p}time_mix_lerp_k.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                    push!(
                        format!("{p}time_mix_lerp_v.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                    push!(
                        format!("{p}time_mix_lerp_r.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                    push!(
                        format!("{p}time_mix_lerp_g.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                }
                push!(
                    format!("{p}time_mix_first.weight"),
                    vec![hs, n_embd / hs],
                    Role::Small
                );
                push!(
                    format!("{p}time_mix_decay.weight"),
                    vec![n_embd],
                    Role::Small
                );
                push!(
                    format!("{p}time_mix_decay_w1.weight"),
                    vec![n_embd, tded],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_decay_w2.weight"),
                    vec![tded, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_key.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_value.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_receptance.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_gate.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(format!("{p}time_mix_ln.weight"), vec![n_embd], Role::Norm);
                push!(format!("{p}time_mix_ln.bias"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}time_mix_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}channel_mix_lerp_k.weight"),
                    vec![n_embd, 1, 1],
                    Role::Small
                );
                push!(
                    format!("{p}channel_mix_lerp_r.weight"),
                    vec![n_embd, 1, 1],
                    Role::Small
                );
                push!(
                    format!("{p}channel_mix_key.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(
                    format!("{p}channel_mix_value.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}channel_mix_receptance.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
            }
        }

        // ---- rwkv6qwen2.cpp:29-77 ----
        "rwkv6qwen2" => {
            let hs = spec.wkv_head_size;
            let tmed = spec.time_mix_extra_dim;
            let tded = spec.time_decay_extra_dim;
            // rwkv6qwen2.cpp:39-44 — head_count_kv 2 < n_embd/head_size 4 →
            // the GQA key/value width
            let akvs = spec.n_head_kv * hs;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            for i in 0..spec.n_layer {
                let p = format!("blk.{i}.");
                push!(format!("{p}attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}time_mix_w1.weight"),
                    vec![n_embd, tmed * 5],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_w2.weight"),
                    vec![tmed, n_embd, 5],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_lerp_x.weight"),
                    vec![n_embd, 1, 1],
                    Role::Small
                );
                push!(
                    format!("{p}time_mix_lerp_fused.weight"),
                    vec![n_embd, 1, 1, 5],
                    Role::Small
                );
                // time_mix_first intentionally absent — that is what makes the
                // graph take the is_qrwkv GLA path (rwkv6-base.cpp:50)
                push!(
                    format!("{p}time_mix_decay.weight"),
                    vec![n_embd],
                    Role::Small
                );
                push!(
                    format!("{p}time_mix_decay_w1.weight"),
                    vec![n_embd, tded],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_decay_w2.weight"),
                    vec![tded, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_key.weight"),
                    vec![n_embd, akvs],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_value.weight"),
                    vec![n_embd, akvs],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_receptance.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_gate.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                // optional biases (rwkv6qwen2.cpp:66-68) — present in the file
                push!(format!("{p}time_mix_key.bias"), vec![akvs], Role::Bias);
                push!(format!("{p}time_mix_value.bias"), vec![akvs], Role::Bias);
                push!(
                    format!("{p}time_mix_receptance.bias"),
                    vec![n_embd],
                    Role::Bias
                );
                push!(
                    format!("{p}time_mix_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(format!("{p}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(
                    format!("{p}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(format!("{p}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
            }
        }

        // ---- rwkv7.cpp:49-116 / arwkv7.cpp:49-112 ----
        "rwkv7" | "arwkv7" => {
            let arwkv = spec.arch == "arwkv7";
            let (nld, nli, nlv, nlg) = (
                spec.n_lora_decay,
                spec.n_lora_iclr,
                spec.n_lora_value_res_mix,
                spec.n_lora_gate,
            );
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            if !arwkv {
                push!("token_embd_norm.weight", vec![n_embd], Role::Norm);
                push!("token_embd_norm.bias", vec![n_embd], Role::Norm);
                push!("output_norm.bias", vec![n_embd], Role::Norm);
            }
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            push!("output.weight", vec![n_embd, N_VOCAB], Role::Head);
            for i in 0..spec.n_layer {
                let p = format!("blk.{i}.");
                push!(format!("{p}attn_norm.weight"), vec![n_embd], Role::Norm);
                if !arwkv {
                    push!(format!("{p}attn_norm.bias"), vec![n_embd], Role::Norm);
                    push!(format!("{p}attn_norm_2.weight"), vec![n_embd], Role::Norm);
                    push!(format!("{p}attn_norm_2.bias"), vec![n_embd], Role::Norm);
                }
                push!(format!("{p}time_mix_w0.weight"), vec![n_embd], Role::Small);
                push!(
                    format!("{p}time_mix_w1.weight"),
                    vec![n_embd, nld],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_w2.weight"),
                    vec![nld, n_embd],
                    Role::Proj
                );
                push!(format!("{p}time_mix_a0.weight"), vec![n_embd], Role::Small);
                push!(
                    format!("{p}time_mix_a1.weight"),
                    vec![n_embd, nli],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_a2.weight"),
                    vec![nli, n_embd],
                    Role::Proj
                );
                // layer 0's v-triple loads at the iclr width (rwkv7.cpp:84-93)
                let vmid = if i == 0 { nli } else { nlv };
                push!(format!("{p}time_mix_v0.weight"), vec![n_embd], Role::Small);
                push!(
                    format!("{p}time_mix_v1.weight"),
                    vec![n_embd, vmid],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_v2.weight"),
                    vec![vmid, n_embd],
                    Role::Proj
                );
                if spec.rwkv7_gating {
                    push!(
                        format!("{p}time_mix_g1.weight"),
                        vec![n_embd, nlg],
                        Role::Proj
                    );
                    push!(
                        format!("{p}time_mix_g2.weight"),
                        vec![nlg, n_embd],
                        Role::Proj
                    );
                    push!(
                        format!("{p}time_mix_lerp_fused.weight"),
                        vec![n_embd, 1, 1, 6],
                        Role::Small
                    );
                } else {
                    // ARWKV models may not have gate tensors — the 5-plane
                    // fallback (arwkv7.cpp:88-93)
                    push!(
                        format!("{p}time_mix_lerp_fused.weight"),
                        vec![n_embd, 1, 1, 5],
                        Role::Small
                    );
                }
                push!(format!("{p}time_mix_k_k.weight"), vec![n_embd], Role::Small);
                push!(format!("{p}time_mix_k_a.weight"), vec![n_embd], Role::Small);
                push!(format!("{p}time_mix_r_k.weight"), vec![n_embd], Role::Small);
                push!(
                    format!("{p}time_mix_key.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_value.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}time_mix_receptance.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                // arwkv7's ln pair is NOT_REQUIRED — present in the fixture so
                // the group norm runs (its absence leaves the WKV output
                // unnormalized, which the random-weight recurrence amplifies
                // into the reference's own thread-noise cliff)
                push!(format!("{p}time_mix_ln.weight"), vec![n_embd], Role::Norm);
                push!(format!("{p}time_mix_ln.bias"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}time_mix_output.weight"),
                    vec![n_embd, n_embd],
                    Role::Proj
                );
                if !arwkv {
                    push!(
                        format!("{p}channel_mix_lerp_k.weight"),
                        vec![n_embd, 1, 1],
                        Role::Small
                    );
                    push!(
                        format!("{p}channel_mix_key.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj
                    );
                    push!(
                        format!("{p}channel_mix_value.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj
                    );
                } else {
                    push!(format!("{p}ffn_norm.weight"), vec![n_embd], Role::Norm);
                    push!(
                        format!("{p}ffn_gate.weight"),
                        vec![n_embd, n_ff],
                        Role::Proj
                    );
                    push!(
                        format!("{p}ffn_down.weight"),
                        vec![n_ff, n_embd],
                        Role::Proj
                    );
                    push!(format!("{p}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                }
            }
        }

        // ---- gemma3n.cpp:21-76 (n_embd_altup 256 / n_altup 4 / laurel_rank
        // 64 — the hardcoded llama-hparams.h defaults) ----
        "gemma3n" => {
            let (n_altup, laurel_rank, n_embd_altup) = (4i64, 64i64, 256i64);
            let n_head = spec.n_head;
            let hd = spec.n_embd_head;
            let kv_w = spec.n_head_kv * hd;
            push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Embd);
            // no output.weight — the loader falls back to tok_embd
            // (TENSOR_DUPLICATED, gemma3n.cpp:28-32)
            push!(
                "altup_proj.weight",
                vec![n_embd, n_embd, n_altup - 1],
                Role::Proj
            );
            push!(
                "altup_unembd_proj.weight",
                vec![n_embd, n_embd, n_altup - 1],
                Role::Proj
            );
            push!(
                "per_layer_token_embd.weight",
                vec![n_embd_altup * spec.n_layer as i64, N_VOCAB],
                Role::Embd
            );
            push!(
                "per_layer_model_proj.weight",
                vec![n_embd, n_embd_altup * spec.n_layer as i64],
                Role::Proj
            );
            push!("per_layer_proj_norm.weight", vec![n_embd_altup], Role::Norm);
            push!("output_norm.weight", vec![n_embd], Role::Norm);
            for i in 0..spec.n_layer {
                let p = format!("blk.{i}.");
                push!(format!("{p}attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}attn_q.weight"),
                    vec![n_embd, hd * n_head],
                    Role::Proj
                );
                push!(format!("{p}attn_k.weight"), vec![n_embd, kv_w], Role::Proj);
                push!(format!("{p}attn_v.weight"), vec![n_embd, kv_w], Role::Proj);
                push!(
                    format!("{p}attn_output.weight"),
                    vec![hd * n_head, n_embd],
                    Role::Proj
                );
                push!(format!("{p}attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("{p}attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(
                    format!("{p}post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("{p}ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}ffn_gate.weight"),
                    vec![n_embd, n_ff],
                    Role::Proj
                );
                push!(format!("{p}ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
                push!(
                    format!("{p}ffn_down.weight"),
                    vec![n_ff, n_embd],
                    Role::Proj
                );
                push!(format!("{p}post_ffw_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}inp_gate.weight"),
                    vec![n_embd, n_embd_altup],
                    Role::Proj
                );
                push!(
                    format!("{p}proj.weight"),
                    vec![n_embd_altup, n_embd],
                    Role::Proj
                );
                push!(format!("{p}post_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("{p}altup_correct_coef.weight"),
                    vec![n_altup, n_altup],
                    Role::Small
                );
                push!(
                    format!("{p}altup_correct_scale.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("{p}altup_predict_coef.weight"),
                    vec![n_altup, n_altup * n_altup],
                    Role::Small
                );
                push!(
                    format!("{p}altup_router.weight"),
                    vec![n_embd, n_altup],
                    Role::Small
                );
                push!(
                    format!("{p}altup_router_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("{p}laurel_l.weight"),
                    vec![n_embd, laurel_rank],
                    Role::Proj
                );
                push!(
                    format!("{p}laurel_r.weight"),
                    vec![laurel_rank, n_embd],
                    Role::Proj
                );
                push!(
                    format!("{p}laurel_post_norm.weight"),
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
// writer (same recipe as arch_batch5_e2e.rs)
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
        // ~[-0.02, 0.02]: lerps stay around 0.5 after the +x lerp, the decays
        // keep exp(-exp(·)) near 1, the routers near 0
        Role::Small => 0.02,
        Role::Embd | Role::Head => 1.0,
        Role::Proj => 1.0 / (n_embd as f32).sqrt(),
    }
}

/// fixture-stability damping of the rwkv7 a/v lora pair (spec.damp_lora)
fn damped(name: &str, role: Role, spec: &SynthSpec) -> Role {
    if spec.damp_lora
        && (name.contains("time_mix_a1")
            || name.contains("time_mix_a2")
            || name.contains("time_mix_v1")
            || name.contains("time_mix_v2"))
    {
        return Role::Small;
    }
    role
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn build_file(spec: &SynthSpec) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch14");

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
    if spec.gemma3n {
        // gemma3n.cpp:3-18 — SWA pattern 5, the required sliding window, rms eps
        kv!(format!("{a}.attention.sliding_window"), Value::U32(512));
        kv!(
            format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-5)
        );
        // head geometry (the generic loader's keys — key/value lengths scalar)
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
            Value::U32(spec.n_embd_head as u32)
        );
        kv!(
            format!("{a}.attention.value_length"),
            Value::U32(spec.n_embd_head as u32)
        );
        kv!(
            format!("{a}.feed_forward_length"),
            Value::U32(spec.n_ff as u32)
        );
        kv!(format!("{a}.rope.freq_base"), Value::F32(10000.0));
    } else {
        // the RWKV quartet (rwkv6.cpp:6-10 / rwkv7.cpp:6-11)
        kv!(
            format!("{a}.wkv.head_size"),
            Value::U32(spec.wkv_head_size as u32)
        );
        if spec.arch == "rwkv6" || spec.arch == "rwkv6qwen2" {
            kv!(
                format!("{a}.time_mix_extra_dim"),
                Value::U32(spec.time_mix_extra_dim as u32)
            );
            kv!(
                format!("{a}.time_decay_extra_dim"),
                Value::U32(spec.time_decay_extra_dim as u32)
            );
            if spec.rescale_every_n_layers != 0 {
                kv!(
                    format!("{a}.rescale_every_n_layers"),
                    Value::U32(spec.rescale_every_n_layers as u32)
                );
            }
        } else {
            kv!(
                format!("{a}.attention.decay_lora_rank"),
                Value::U32(spec.n_lora_decay as u32)
            );
            kv!(
                format!("{a}.attention.iclr_lora_rank"),
                Value::U32(spec.n_lora_iclr as u32)
            );
            kv!(
                format!("{a}.attention.value_residual_mix_lora_rank"),
                Value::U32(spec.n_lora_value_res_mix as u32)
            );
            if spec.rwkv7_gating {
                kv!(
                    format!("{a}.attention.gate_lora_rank"),
                    Value::U32(spec.n_lora_gate as u32)
                );
            }
        }
        kv!(
            format!("{a}.token_shift_count"),
            Value::U32(spec.token_shift_count as u32)
        );
        // rwkv6 reads both eps keys optionally; the LN path uses the plain one
        kv!(
            format!("{a}.attention.layer_norm_epsilon"),
            Value::F32(1e-5)
        );
        kv!(
            format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-5)
        );
        if spec.n_head > 0 {
            // rwkv6qwen2's GQA geometry (the loader's n_head_kv, rwkv6qwen2.cpp:40-44)
            kv!(
                format!("{a}.attention.head_count"),
                Value::U32(spec.n_head as u32)
            );
            kv!(
                format!("{a}.attention.head_count_kv"),
                Value::U32(spec.n_head_kv as u32)
            );
        }
        kv!(
            format!("{a}.feed_forward_length"),
            Value::U32(spec.n_ff as u32)
        );
    }

    let tensors = tensors_for(spec);
    let mut rng = Rng(0x1234_5678_9abc_def0 ^ spec.arch.len() as u64 ^ spec.seed.rotate_left(17));
    let mut data: Vec<Vec<u8>> = Vec::new();
    for ((name, ne), role) in &tensors {
        let role = damped(name, *role, spec);
        let scale = scale_of(role, spec.n_embd);
        let n: usize = ne.iter().map(|&d| d as usize).product();
        let vals: Vec<f32> = (0..n).map(|_| rng.next() * scale).collect();
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, ggml::types::GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }

    let path = spec.path();
    let f = std::fs::File::create(&path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    (tensors.len(), std::fs::metadata(&path).unwrap().len())
}

fn load_synth(spec: &SynthSpec) -> LlamaModel {
    build_file(spec);
    open_model(&spec.path())
}

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

// ---------------------------------------------------------------------------
// hparams / tensor pins
// ---------------------------------------------------------------------------

fn pin_hparams(m: &LlamaModel, spec: &SynthSpec) {
    let hp = &m.hparams;
    assert_eq!(m.arch.name(), spec.arch);
    assert_eq!(hp.n_embd as i64, spec.n_embd);
    assert_eq!(hp.n_layer() as usize, spec.n_layer);
    assert_eq!(hp.n_ctx_train, spec.n_ctx);
    if spec.gemma3n {
        assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
        // the 4:1 pattern of load_swa_pattern(ml, 5): il%5 < 4
        for il in 0..spec.n_layer {
            assert_eq!(hp.is_swa(il), (il as u32 % 5) < 4, "gemma3n is_swa({il})");
        }
        assert_eq!(
            hp.n_layer_kv_from_start, 20,
            "gemma3n n_layer_kv_from_start"
        );
        assert_eq!(hp.f_attention_scale, 1.0);
        assert_eq!(hp.f_norm_rms_eps, 1e-5);
        assert_eq!(hp.n_head(0) as i64, spec.n_head);
        assert_eq!(hp.n_head_kv(0) as i64, spec.n_head_kv);
        assert_eq!(hp.n_embd_head_k(0) as i64, spec.n_embd_head);
    } else {
        assert_eq!(hp.wkv_head_size as i64, spec.wkv_head_size);
        assert_eq!(hp.token_shift_count as i64, spec.token_shift_count);
        assert_eq!(
            hp.rescale_every_n_layers as i64,
            spec.rescale_every_n_layers
        );
        // the recurrent cell geometry (llama-hparams.cpp n_embd_r/n_embd_s)
        assert_eq!(
            hp.n_embd_r() as i64,
            spec.n_embd_r(),
            "{} n_embd_r",
            spec.arch
        );
        assert_eq!(
            hp.n_embd_s() as i64,
            spec.n_embd_s(),
            "{} n_embd_s",
            spec.arch
        );
        // every layer recurrent + no rope (llm_arch_is_recurrent / rope NONE)
        assert!(hp.is_recr(0) && hp.is_recr(spec.n_layer - 1));
        assert_eq!(hp.rope_type, llama::hparams::LlamaRopeType::NONE);
    }
}

fn pin_tensors(m: &LlamaModel, spec: &SynthSpec) {
    for ((name, ne), _) in tensors_for(spec) {
        let id = m
            .tensors
            .get(&name)
            .unwrap_or_else(|| panic!("{}: tensor {name} not created", spec.arch));
        let want = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        assert_eq!(*m.ctx.ne(*id), want, "{}: {name} shape", spec.arch);
    }
}

// ---------------------------------------------------------------------------
// the decode harness — DecodeContext (batch 12's protocol; the recurrent
// cells auto-allocate from ForwardWeights::recurrent_dims)
// ---------------------------------------------------------------------------

fn forward_of(m: &LlamaModel, fa: bool) -> (ForwardWeights, llama::graph::AttnParams) {
    // the same derivations llama-cli's forward_weights applies (main.rs's
    // batch-14 arms), inlined here so the test runs without the binary
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let rope = hp.rope_runtime();
    let attn = llama::graph::AttnParams {
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
        norm_eps: if hp.f_norm_rms_eps > 0.0 {
            hp.f_norm_rms_eps
        } else {
            hp.f_norm_eps
        },
        use_flash_attn: fa,
    };
    match m.arch {
        llama::arch::LlmArch::RWKV6 => (
            ForwardWeights::Rwkv6(rwkv6_w(m, n_trunk, false), rwkv6_p(hp)),
            attn,
        ),
        llama::arch::LlmArch::RWKV6QWEN2 => (
            ForwardWeights::Rwkv6Qwen2(rwkv6_w(m, n_trunk, true), rwkv6_p(hp)),
            attn,
        ),
        llama::arch::LlmArch::RWKV7 => (
            ForwardWeights::Rwkv7(rwkv7_w(m, n_trunk, false), rwkv7_p(hp)),
            attn,
        ),
        llama::arch::LlmArch::ARWKV7 => (
            ForwardWeights::Arwkv7(rwkv7_w(m, n_trunk, true), rwkv7_p(hp)),
            attn,
        ),
        llama::arch::LlmArch::GEMMA3N => (
            ForwardWeights::Gemma3n(
                gemma3n_w(m, n_trunk),
                llama::graph_arch::Gemma3nParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_altup: hp.n_altup as i64,
                    i_altup_act: hp.i_altup_act as i64,
                    n_embd_altup: hp.n_embd_altup as i64,
                    laurel_rank: hp.laurel_rank as i64,
                    n_layer_sparsity: 10,
                    f_sparsity_std_mul: 1.644_853_4,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    n_layer_kv_from_start: hp.n_layer_kv_from_start as i64,
                    f_attention_scale: hp.f_attention_scale,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    norm_eps: hp.f_norm_rms_eps,
                    f_final_logit_softcapping: hp.f_final_logit_softcapping,
                },
            ),
            attn,
        ),
        other => panic!("batch-14 forward for {other:?}"),
    }
}

fn rwkv6_w(m: &LlamaModel, n_trunk: usize, qwen2: bool) -> llama::graph_arch::Rwkv6ModelWeights {
    llama::graph_arch::Rwkv6ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if qwen2 { None } else { m.token_embd_norm },
        tok_norm_b: if qwen2 { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| llama::graph_arch::Rwkv6LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: if qwen2 { None } else { l.attn_norm_b },
                attn_norm_2: if qwen2 { None } else { l.attn_norm_2 },
                attn_norm_2_b: if qwen2 { None } else { l.attn_norm_2_b },
                time_mix_w1: l.time_mix_w1.unwrap(),
                time_mix_w2: l.time_mix_w2.unwrap(),
                time_mix_lerp_x: l.time_mix_lerp_x.unwrap(),
                time_mix_lerp_w: l.time_mix_lerp_w,
                time_mix_lerp_k: l.time_mix_lerp_k,
                time_mix_lerp_v: l.time_mix_lerp_v,
                time_mix_lerp_r: l.time_mix_lerp_r,
                time_mix_lerp_g: l.time_mix_lerp_g,
                time_mix_lerp_fused: l.time_mix_lerp_fused,
                time_mix_first: l.time_mix_first,
                time_mix_decay: l.time_mix_decay.unwrap(),
                time_mix_decay_w1: l.time_mix_decay_w1.unwrap(),
                time_mix_decay_w2: l.time_mix_decay_w2.unwrap(),
                time_mix_key: l.time_mix_key.unwrap(),
                time_mix_value: l.time_mix_value.unwrap(),
                time_mix_receptance: l.time_mix_receptance.unwrap(),
                time_mix_gate: l.time_mix_gate.unwrap(),
                time_mix_key_b: l.time_mix_key_b,
                time_mix_value_b: l.time_mix_value_b,
                time_mix_receptance_b: l.time_mix_receptance_b,
                time_mix_ln: if qwen2 { None } else { l.time_mix_ln },
                time_mix_ln_b: if qwen2 { None } else { l.time_mix_ln_b },
                time_mix_output: l.time_mix_output.unwrap(),
                channel_mix_lerp_k: if qwen2 { None } else { l.channel_mix_lerp_k },
                channel_mix_lerp_r: if qwen2 { None } else { l.channel_mix_lerp_r },
                channel_mix_key: if qwen2 { None } else { l.channel_mix_key },
                channel_mix_value: if qwen2 { None } else { l.channel_mix_value },
                channel_mix_receptance: if qwen2 {
                    None
                } else {
                    l.channel_mix_receptance
                },
                ffn_norm: if qwen2 { l.ffn_norm } else { None },
                ffn_gate: if qwen2 { l.ffn_gate } else { None },
                ffn_down: if qwen2 { l.ffn_down } else { None },
                ffn_up: if qwen2 { l.ffn_up } else { None },
            })
            .collect(),
    }
}

fn rwkv6_p(hp: &llama::hparams::LlamaHparams) -> llama::graph_arch::Rwkv6Params {
    llama::graph_arch::Rwkv6Params {
        n_embd: hp.n_embd as i64,
        wkv_head_size: hp.wkv_head_size as i64,
        time_mix_extra_dim: hp.time_mix_extra_dim as i64,
        token_shift_count: hp.token_shift_count as i64,
        rescale_every_n_layers: hp.rescale_every_n_layers as i64,
        norm_eps: hp.f_norm_eps,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

fn rwkv7_w(m: &LlamaModel, n_trunk: usize, arwkv: bool) -> llama::graph_arch::Rwkv7ModelWeights {
    llama::graph_arch::Rwkv7ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if arwkv { None } else { m.token_embd_norm },
        tok_norm_b: if arwkv { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: if arwkv { None } else { m.output_norm_b },
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| llama::graph_arch::Rwkv7LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: if arwkv { None } else { l.attn_norm_b },
                attn_norm_2: if arwkv { None } else { l.attn_norm_2 },
                attn_norm_2_b: if arwkv { None } else { l.attn_norm_2_b },
                time_mix_w0: l.time_mix_w0.unwrap(),
                time_mix_w1: l.time_mix_w1.unwrap(),
                time_mix_w2: l.time_mix_w2.unwrap(),
                time_mix_a0: l.time_mix_a0.unwrap(),
                time_mix_a1: l.time_mix_a1.unwrap(),
                time_mix_a2: l.time_mix_a2.unwrap(),
                time_mix_v0: l.time_mix_v0.unwrap(),
                time_mix_v1: l.time_mix_v1.unwrap(),
                time_mix_v2: l.time_mix_v2.unwrap(),
                time_mix_g1: l.time_mix_g1,
                time_mix_g2: l.time_mix_g2,
                time_mix_lerp_fused: l.time_mix_lerp_fused.unwrap(),
                time_mix_k_k: l.time_mix_k_k.unwrap(),
                time_mix_k_a: l.time_mix_k_a.unwrap(),
                time_mix_r_k: l.time_mix_r_k.unwrap(),
                time_mix_key: l.time_mix_key.unwrap(),
                time_mix_value: l.time_mix_value.unwrap(),
                time_mix_receptance: l.time_mix_receptance.unwrap(),
                time_mix_ln: l.time_mix_ln,
                time_mix_ln_b: l.time_mix_ln_b,
                time_mix_output: l.time_mix_output.unwrap(),
                channel_mix_lerp_k: if arwkv { None } else { l.channel_mix_lerp_k },
                channel_mix_key: if arwkv { None } else { l.channel_mix_key },
                channel_mix_value: if arwkv { None } else { l.channel_mix_value },
                ffn_norm: if arwkv { l.ffn_norm } else { None },
                ffn_gate: if arwkv { l.ffn_gate } else { None },
                ffn_down: if arwkv { l.ffn_down } else { None },
                ffn_up: if arwkv { l.ffn_up } else { None },
            })
            .collect(),
    }
}

fn rwkv7_p(hp: &llama::hparams::LlamaHparams) -> llama::graph_arch::Rwkv7Params {
    llama::graph_arch::Rwkv7Params {
        n_embd: hp.n_embd as i64,
        wkv_head_size: hp.wkv_head_size as i64,
        token_shift_count: hp.token_shift_count as i64,
        norm_eps: hp.f_norm_eps,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

fn gemma3n_w(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Gemma3nModelWeights {
    llama::graph_arch::Gemma3nModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        output_norm: m.output_norm,
        altup_proj: m.altup_proj.unwrap(),
        altup_unembd_proj: m.altup_unembd_proj.unwrap(),
        per_layer_tok_embd: m.per_layer_tok_embd.unwrap(),
        per_layer_model_proj: m.per_layer_model_proj.unwrap(),
        per_layer_proj_norm: m.per_layer_proj_norm.unwrap(),
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| llama::graph_arch::Gemma3nLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                per_layer_inp_gate: l.per_layer_inp_gate.unwrap(),
                per_layer_proj: l.per_layer_proj.unwrap(),
                per_layer_post_norm: l.per_layer_post_norm.unwrap(),
                altup_correct_coef: l.altup_correct_coef.unwrap(),
                altup_correct_scale: l.altup_correct_scale.unwrap(),
                altup_predict_coef: l.altup_predict_coef.unwrap(),
                altup_router: l.altup_router.unwrap(),
                altup_router_norm: l.altup_router_norm.unwrap(),
                laurel_l: l.laurel_l.unwrap(),
                laurel_r: l.laurel_r.unwrap(),
                laurel_post_norm: l.laurel_post_norm.unwrap(),
            })
            .collect(),
    }
}

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

fn logits_of_argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as i32
}

/// one prefill (a 12-token prompt) + one decode step; asserts finite logits
/// with a real spread, and a bit-identical prefill repeat after a full state
/// reset (the RWKV rs_zero / the KV clear)
fn smoke_forward(m: &mut LlamaModel, spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut dctx = driver_for(m, fa);
    let prompt: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "{}: non-finite logits (fa={fa})",
        spec.arch
    );
    let spread = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1.0,
        "{}: logits degenerate (spread {spread}, fa={fa})",
        spec.arch
    );
    let tk = logits_of_argmax(&logits);
    let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
    assert!(next.iter().all(|v| v.is_finite()));

    // fresh sequence: clear() drops the KV cells and zeroes the recurrent
    // cells (the rs_zero read rule) — the prefill must repeat bit-identically
    dctx.reset_sequence();
    let again = dctx.decode(&prompt, &pos).expect("prefill again").to_vec();
    assert_eq!(
        logits, again,
        "{}: prefill not deterministic after state reset (fa={fa})",
        spec.arch
    );
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the tensor/hparams pins + the double-FA smoke of the parity cells
#[test]
fn arch_batch14_pin_and_smoke() {
    for spec in parity_specs() {
        let mut m = load_synth(&spec);
        pin_hparams(&m, &spec);
        pin_tensors(&m, &spec);
        // the model's Context moves into the driver — reload per FA mode
        for fa in [false, true] {
            // RWKV is attention-free: the FA modes degenerate (the flag only
            // rides the unused mask type); gemma3n runs both attention paths
            let mut m = load_synth(&spec);
            let logits = smoke_forward(&mut m, &spec, fa);
            println!(
                "{} fa={fa}: decode-1 greedy first id {}",
                spec.arch,
                logits_of_argmax(&logits)
            );
        }
        println!("{}: pins + smoke ok (both FA modes)", spec.arch);
    }
}

/// the two in-port variants: rwkv6-legacy (the separate w/k/v/r/g lerps +
/// rescale_every_n_layers) — loadable by the reference too, but kept out of
/// the default parity set (the fused file is the rwkv6 cell)
#[test]
fn arch_batch14_rwkv6_legacy_variant() {
    let spec = spec_rwkv6_legacy();
    let mut m = load_synth(&spec);
    pin_hparams(&m, &spec);
    pin_tensors(&m, &spec);
    let l0 = &m.layers[0];
    assert!(l0.time_mix_lerp_fused.is_none(), "legacy: no fused lerp");
    assert!(l0.time_mix_lerp_w.is_some() && l0.time_mix_lerp_g.is_some());
    assert_eq!(m.hparams.rescale_every_n_layers, 1);
    let logits = smoke_forward(&mut m, &spec, false);
    println!(
        "rwkv6-legacy: decode-1 greedy first id {}",
        logits_of_argmax(&logits)
    );
}

/// write all the synthetic files (the parity runs' generator)
#[test]
#[ignore]
fn arch_batch14_write_synth() {
    for spec in parity_specs() {
        let (n, bytes) = build_file(&spec);
        println!(
            "{}: {n} tensors, {bytes} data bytes -> {}",
            spec.arch,
            spec.path()
        );
    }
}

// ---------------------------------------------------------------------------
// DECDMP1 node-dump mirror (parity/ref_decode_dump.c protocol) — the
// whole-graph bisect stream for `parity/decode_dump_cmp.py`
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// nodes at/above this element count carry no payload (2^19, the C probe rule)
const DUMP_ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static DUMP_ACTIVE: AtomicU32 = AtomicU32::new(0);

fn dump_op_desc(op: ggml::GgmlOp, op_params: &[i32]) -> &'static str {
    use ggml::GgmlOp::*;
    // the port folds every unary into GgmlOp::Silu with the real op in
    // params[0] — expand it like the C probe's ggml_op_desc
    // (GGML_UNARY_OP_NAME, ggml.c:1224-1246) so the streams align by name
    if op == Silu {
        const NAMES: [&str; 22] = [
            "ABS",
            "SGN",
            "NEG",
            "STEP",
            "TANH",
            "ELU",
            "RELU",
            "SIGMOID",
            "GELU",
            "GELU_QUICK",
            "SILU",
            "HARDSWISH",
            "HARDSIGMOID",
            "EXP",
            "EXPM1",
            "SOFTPLUS",
            "GELU_ERF",
            "XIELU",
            "FLOOR",
            "CEIL",
            "ROUND",
            "TRUNC",
        ];
        let u = op_params[0] as usize;
        if u < NAMES.len() {
            return NAMES[u];
        }
        return "UNARY?";
    }
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        Mul => "MUL",
        Div => "DIV",
        Sub => "SUB",
        Norm => "NORM",
        SquaredMulMat => "MUL_MAT_SQ",
        MulMat => "MUL_MAT",
        Scale => "SCALE",
        Cpy => "CPY",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        DiagMaskInf => "DIAG_MASK_INF",
        SoftMax => "SOFT_MAX",
        RoPE => "ROPE",
        RoPEBack => "ROPE_BACK",
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => "SILU",
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => "GLU",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        Clamp => "CLAMP",
        Gdn => "GATED_DELTA_NET",
        Im2col => "IM2COL",
        Upscale => "UPSCALE",
        Fill => "FILL",
        LightningIndexer => "LIGHTNING_INDEXER",
        TopK => "TOP_K",
        Sqrt => "SQRT",
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        Pad => "PAD",
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn dump_type_desc(ty: ggml::types::GgmlType) -> &'static str {
    use ggml::types::GgmlType::*;
    match ty {
        F32 => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        I64 => "i64",
        I32 => "i32",
        I16 => "i16",
        I8 => "i8",
        _ => "other",
    }
}

fn dump_put_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len().min(255);
    buf.push(len as u8);
    buf.extend_from_slice(&s.as_bytes()[..len]);
}

fn dump_cb(node: &ggml::compute::EvalNode<'_>, ask: bool) -> bool {
    use ggml::types::GgmlType;
    if ask {
        return true;
    }
    let mut guard = DUMP.get().unwrap().lock().unwrap();
    let Some(st) = guard.as_mut() else {
        return true;
    };
    let n: i64 = node.ne.iter().product();
    st.nodes += 1;
    dump_put_str(&mut st.out, dump_op_desc(node.op, &node.op_params));
    dump_put_str(&mut st.out, node.name);
    dump_put_str(&mut st.out, dump_type_desc(node.ty));
    st.out.extend(&node.ne.map(|v| v.to_le_bytes()).concat());
    st.out.extend(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= DUMP_ELEM_CAP {
        return true;
    }
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
        return true;
    }
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v: f32 = if node.ty == GgmlType::F32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            let h = half::f16::from_le_bytes([data[off], data[off + 1]]);
            h.to_f32()
        };
        st.out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

/// stream every node of one gemma3n `decode_embed` prefill to B14_DUMP_OUT —
/// the DECDMP1 mirror of `parity/ref_decode_dump.c` for the batch-14 files
/// (run: B14_DUMP_MODEL=... B14_DUMP_OUT=... B14_FA_OFF=1 cargo test
///  --release -p llama --test arch_batch14_e2e -- --ignored --nocapture
///  arch_batch14_prefill_node_dump)
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the gemma3n parity file"]
fn arch_batch14_prefill_node_dump() {
    let model_path = std::env::var("B14_DUMP_MODEL")
        .unwrap_or_else(|_| "/tmp/arch-batch14/gemma3n-synth.gguf".to_string());
    let out_path =
        std::env::var("B14_DUMP_OUT").unwrap_or_else(|_| "/tmp/b14-port.bin".to_string());
    let prompt =
        std::env::var("B14_DUMP_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
    let fa_off = std::env::var("B14_FA_OFF").is_ok();

    let mut m = open_model(&model_path);
    let vocab =
        llama::vocab::Vocab::load(&Gguf::open(&model_path).expect("open gguf")).expect("vocab");
    let ids = vocab.tokenize(&prompt, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!(
        "tokens: {ids:?} ({} tokens, fa={}, {})",
        ids.len(),
        if fa_off { "off" } else { "on" },
        model_path
    );

    // decode_embed protocol: every token an output row (the C probe's
    // --embeddings --pooling none context)
    let mut dctx =
        driver_for(&mut m, !fa_off).with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);

    DUMP.get_or_init(|| {
        Mutex::new(Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        }))
    });
    {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        *guard = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    ggml::compute::set_eval_callback(Some(dump_cb));
    DUMP_ACTIVE.store(1, Ordering::SeqCst);
    let embd = dctx.decode_embed(&ids, &pos).expect("decode_embed");
    DUMP_ACTIVE.store(0, Ordering::SeqCst);
    ggml::compute::set_eval_callback(None);

    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    use std::io::Write as _;
    let mut f = std::fs::File::create(&out_path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
    for &id in &ids {
        f.write_all(&id.to_le_bytes()).unwrap();
    }
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    drop(f);

    println!(
        "arch_batch14_prefill_node_dump: {nodes} nodes, embd {}x{}, -> {out_path}",
        embd.n_rows, embd.n_embd_out
    );
}

/// debug: mul_mat GQA-broadcast unit check — a {K, M, 2} contiguous, b {K, N, 4}
#[test]
#[ignore]
fn arch_batch14_debug_mm_gqa() {
    use ggml::GgmlType;
    let (k, m, n) = (8i64, 4i64, 3i64);
    let mut ctx = Context::new();
    let av: Vec<f32> = (0..k * m * 2).map(|i| (i as f32) * 0.5 - 10.0).collect();
    let bv: Vec<f32> = (0..k * n * 4).map(|i| (i as f32) * 0.25 - 8.0).collect();
    let a = ctx.new_tensor_3d(GgmlType::F32, k, m, 2);
    ctx.arena_resize_tensor(a);
    ctx.data_bytes_mut(a)
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&av));
    let b = ctx.new_tensor_3d(GgmlType::F32, k, n, 4);
    ctx.arena_resize_tensor(b);
    ctx.data_bytes_mut(b)
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&bv));
    let c = ctx.mul_mat(a, b);
    let mut graph = ggml::Graph::new(16);
    graph.build_forward(&ctx, c);
    ggml::compute::graph_compute(&mut ctx, &mut graph, 1);
    let out = ctx.data_bytes(c).unwrap()[..(m * n * 4) as usize * 4].to_vec();
    std::fs::write("/tmp/b14_mmgqa_a.bin", bytemuck::cast_slice(&av)).unwrap();
    std::fs::write("/tmp/b14_mmgqa_b.bin", bytemuck::cast_slice(&bv)).unwrap();
    std::fs::write("/tmp/b14_mmgqa_c.bin", out).unwrap();
    println!("mm gqa unit done");
}
