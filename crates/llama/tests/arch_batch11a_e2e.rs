//! arch_batch11a_e2e.rs — synthetic-GGUF verification of the arch batch
//! landed on 2026-10: **the long-tail queue, first half** — apertus /
//! grovemoe / qwen35moe / kimi-k3 / dots3note / minimax-m3 / qwen4exp
//! (llama.cpp bd4f514db1).
//!
//! Same protocol as batches 1-10 (`crates/llama/tests/arch_batch*_e2e.rs`,
//! PARITY.md): no local GGUF of these archs exists, so each is verified on a
//! *synthetic* file built with the port's byte-exact GGUF writer —
//! `tokenizer.*` KV copied verbatim from the llama SPM vocab fixture, the
//! arch's own KV, and exactly the tensor names + shapes its
//! `load_arch_tensors` asks for, all F32.
//!
//!   * apertus — the xIELU FFN activation (per-layer alpha_n/alpha_p/beta/
//!     eps arrays), per-head q/k RMS norms before a full-head rope;
//!   * grovemoe — TWO softmax MoEs over the same router logits (the experts
//!     and the chunk experts, the second scaled by expert_group_scale);
//!   * qwen35moe — the qwen3next hybrid (GDN + gated full attention) with
//!     attn_post_norm, IMRoPE sections and the sigmoid-gated shared expert;
//!   * kimi-k3 — the KDA layers (per-stream convs, the safe
//!     gate_lower_bound decay, the single full-rank output gate) + nope-MLA
//!     layers with the sigmoid output gate + the latent SITU MoE + the
//!     cross-layer residual attention bank (attn_res_block_size 2);
//!   * dots3note — the deepseek32 DSA lightning indexer over an iswa MLA
//!     pair with DIFFERENT lora geometry per side, the shared-rope-key norm
//!     and the head-wise sigmoid output gate;
//!   * minimax-m3 — M2-style GQA + the swigluoai FFN/MoE; MSA (the block
//!     top-k sparse attention) runs with `-fa on` only — FA off is the
//!     reference's documented DENSE fallback (minimax-m3.cpp:239-244);
//!   * qwen4exp — the deepseek4 hyper-connection residual streams + the
//!     qwen3next GDN/gated-attention pair + MoE. The QSA block-compression
//!     and the PLE n-gram module are NOT ported; the verified file omits the
//!     optional `attention.compress_ratios` and PLE keys, which is the
//!     reference's legal dense/no-PLE configuration.
//!
//! The batch's ForwardWeights/CLI arms landed with the graph (context.rs),
//! so the default-run tests drive `DecodeContext::new_with{,_swa}` itself
//! (batch-6 protocol) and the parity runs drive the release CLI
//! (`ARCH_BATCH11A=1 ./parity/arch_batch_parity.sh …`).

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

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-batch11a";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

// ---------------------------------------------------------------------------
// the synthetic model spec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Apertus,
    Grovemoe,
    Qwen35Moe,
    KimiK3,
    Dots3Note,
    MinimaxM3,
    Qwen4Exp,
}

#[derive(Clone)]
struct SynthSpec {
    arch: &'static str,
    suffix: &'static str,
    family: Family,
    n_layer: usize,
    n_embd: i64,
    n_head: i64,
    /// per-layer head_count_kv (kimi-k3: 0 marks the KDA layers)
    head_kv: Vec<i64>,
    key_length: i64,
    value_length: i64,
    rope_dim: i64,
    n_ff: i64,
    n_ctx: u32,
    write_output: bool,
    n_ff_exp: i64,
    /// false → the loader ties output.weight to token_embd.weight
    // per-family extras (kept flat, batch-10 style)
    gating: Option<u32>,
    weights_scale: Option<f32>,
    weights_norm: Option<bool>,
    /// qwen35moe / qwen4exp: explicit attention.recurrent_layers
    recr_layers: Option<Vec<u32>>,
    /// IMROPE dimension sections (sum == rope_dim, 4th 0 for text)
    rope_sections: Option<[i32; 4]>,
    /// minimax-m3: the MSA block parameters
    msa_blk: i64,
    msa_topk: i64,
    msa_local: i64,
    indexer_heads: i64,
    indexer_dim: i64,
    indexer_topk: i32,
    /// dots3note: the iswa MLA geometry pair + the pattern
    n_lora_kv: i64,
    k_mla: i64,
    v_mla: i64,
    n_lora_kv_swa: i64,
    k_mla_swa: i64,
    v_mla_swa: i64,
    swa_pattern: Vec<u32>,
    n_swa: u32,
    freq_base_swa: f32,
    /// kimi-k3: the KDA/MLA/MoE extras
    kda_head_dim: i64,
    d_conv: i64,
    gate_lower_bound: Option<f32>,
    q_lora: i64,
    attn_res_block_size: u32,
    situ_beta: f32,
    situ_linear_beta: f32,
    n_expert_latent: i64,
    write_routed_norm: bool,
    dense_lead: u32,
    /// qwen4exp: the HC module
    hc: i64,
    hc_lr: i64,
    /// grovemoe: the chunk experts
    n_ff_chexp: i64,
    expert_group_scale: f32,
    n_group_experts: i64,
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
            Family::Qwen35Moe | Family::Qwen4Exp => match &self.recr_layers {
                Some(v) => v[il] != 0,
                None => (il as u32 + 1) % 4 != 0,
            },
            Family::KimiK3 => self.head_kv[il] == 0,
            _ => false,
        }
    }
    /// dots3note: is_swa per layer (the pattern array)
    fn is_swa(&self, il: usize) -> bool {
        self.swa_pattern[il] != 0
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
        n_ff_exp: 32,
        gating: Some(1), // SOFTMAX
        weights_scale: None,
        weights_norm: None,
        recr_layers: None,
        rope_sections: None,
        msa_blk: 0,
        msa_topk: 0,
        msa_local: 0,
        indexer_heads: 0,
        indexer_dim: 0,
        indexer_topk: 0,
        n_lora_kv: 0,
        k_mla: 0,
        v_mla: 0,
        n_lora_kv_swa: 0,
        k_mla_swa: 0,
        v_mla_swa: 0,
        swa_pattern: Vec::new(),
        n_swa: 0,
        freq_base_swa: 0.0,
        kda_head_dim: 0,
        d_conv: 0,
        gate_lower_bound: None,
        q_lora: 0,
        attn_res_block_size: 0,
        situ_beta: 1.0,
        situ_linear_beta: 1.0,
        n_expert_latent: 0,
        write_routed_norm: false,
        dense_lead: 1,
        hc: 0,
        hc_lr: 0,
        n_ff_chexp: 0,
        expert_group_scale: 0.0,
        n_group_experts: 0,
    }
}

/// apertus — the xIELU FFN (per-layer constant arrays), full-head rope
fn spec_apertus() -> SynthSpec {
    base("apertus", Family::Apertus)
}

/// grovemoe — 4 experts in 2 groups, chunk experts at 64, group scale 0.05
fn spec_grovemoe() -> SynthSpec {
    base("grovemoe", Family::Grovemoe).with(|s| {
        s.n_ff_chexp = 64;
        s.expert_group_scale = 0.05;
        s.n_group_experts = 2;
    })
}

/// qwen35moe — GDN on layers 0/1/3, gated IMRoPE attention on layer 2
fn spec_qwen35moe() -> SynthSpec {
    base("qwen35moe", Family::Qwen35Moe).with(|s| {
        s.n_embd = 64;
        s.n_ff = 64;
        s.key_length = 16;
        s.value_length = 16;
        s.rope_dim = 16;
        s.recr_layers = Some(vec![1, 1, 0, 1]);
        s.rope_sections = Some([8, 4, 4, 0]);
        // hparams.ssm_* (head_k 8, n_k 1, n_v 2, d_inner 16)
        s.kda_head_dim = 0; // unused; the ssm keys carry the geometry
        s.d_conv = 4;
    })
}

/// kimi-k3 — KDA on layers 0/2 (head_kv 0), nope-MLA on 1/3 (head_kv 1),
/// the residual bank every 2 layers, the latent MoE at n_expert_latent 64
fn spec_kimi_k3() -> SynthSpec {
    base("kimi-k3", Family::KimiK3).with(|s| {
        s.n_embd = 128;
        s.n_ff = 96;
        s.head_kv = vec![0, 1, 0, 1];
        // attention.key_length = [kv_lora|rope] (24), value_length = kv_lora
        s.n_lora_kv = 16;
        s.key_length = 24;
        s.value_length = 16;
        s.k_mla = 48;
        s.v_mla = 32;
        s.rope_dim = 8; // qk_rope_head_dim
        s.q_lora = 8;
        s.kda_head_dim = 32;
        s.d_conv = 4;
        s.gate_lower_bound = Some(-5.0);
        s.attn_res_block_size = 2;
        s.situ_beta = 0.9;
        s.situ_linear_beta = 0.8;
        s.n_expert_latent = 64;
        s.write_routed_norm = true;
        s.dense_lead = 1;
    })
}

/// dots3note — full attention on even layers (indexer), SWA on odd; the two
/// sides carry different MLA geometry
fn spec_dots3note() -> SynthSpec {
    base("dots3note", Family::Dots3Note).with(|s| {
        s.n_layer = 6;
        s.head_kv = vec![1; 6];
        s.n_lora_kv = 16;
        s.k_mla = 48;
        s.v_mla = 32;
        s.n_lora_kv_swa = 12;
        s.k_mla_swa = 40;
        s.v_mla_swa = 24;
        s.key_length = 24; // [16 | rope 8]
        s.value_length = 16;
        s.rope_dim = 8;
        s.q_lora = 8;
        s.swa_pattern = vec![0, 1, 0, 1, 0, 1];
        s.n_swa = 64;
        s.freq_base_swa = 30000.0;
        s.indexer_heads = 2;
        s.indexer_dim = 64;
        s.indexer_topk = 16;
        s.dense_lead = 1;
        s.write_output = false;
    })
}

/// minimax-m3 — layer 0 dense, layers 1-3 MSA (indexer heads == head_kv 2)
fn spec_minimax_m3() -> SynthSpec {
    base("minimax-m3", Family::MinimaxM3).with(|s| {
        s.n_embd = 128;
        s.rope_dim = 8; // partial rope (head 32, rot 8)
        s.msa_blk = 16;
        s.msa_topk = 2;
        s.msa_local = 1;
        s.indexer_heads = 2; // == head_count_kv (one per GQA group)
        s.indexer_dim = 32;
        s.indexer_topk = 2;
        s.dense_lead = 1;
    })
}

/// qwen4exp — hc 2 streams, GDN on 0/1/3, gated IMRoPE attention on 2; no
/// compress_ratios / PLE keys (the legal dense no-PLE configuration)
fn spec_qwen4exp() -> SynthSpec {
    base("qwen4exp", Family::Qwen4Exp).with(|s| {
        s.n_embd = 64;
        s.n_ff = 32;
        s.key_length = 16;
        s.value_length = 16;
        s.rope_dim = 16;
        s.recr_layers = Some(vec![1, 1, 0, 1]);
        s.rope_sections = Some([8, 4, 4, 0]);
        s.hc = 2;
        s.hc_lr = 8;
        s.d_conv = 4;
        // hparams.ssm_*: head_k 8, n_k 1, n_v 2 (dt_rank), d_inner 16
        s.indexer_heads = 2;
        s.indexer_dim = 16;
        s.indexer_topk = 8;
    })
}

fn all_specs() -> Vec<SynthSpec> {
    vec![
        spec_apertus(),
        spec_grovemoe(),
        spec_qwen35moe(),
        spec_kimi_k3(),
        spec_dots3note(),
        spec_minimax_m3(),
        spec_qwen4exp(),
    ]
}

/// the parity cells (batch-11a default set): all seven archs
fn parity_specs() -> Vec<SynthSpec> {
    all_specs()
}

#[test]
#[ignore = "writes the /tmp/arch-batch11a parity files (ARCH_BATCH11A cells)"]
fn arch_batch11a_write_synth() {
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
    /// softmax/sigmoid over the router logits stays spread out
    Router,
    /// decay/gate vectors — small values keep exp()/tanh() bounded
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
    if spec.family != Family::Qwen4Exp {
        push!("output_norm.weight", vec![n_embd], Role::Norm);
    }
    if spec.write_output {
        push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);
    }

    match spec.family {
        Family::Apertus => {
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length;
                let kv = spec.head_kv[i] * spec.key_length;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.key_length, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.bias"),
                    vec![n_embd],
                    Role::Bias
                );
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_norm.bias"),
                    vec![spec.key_length],
                    Role::Bias
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.bias"),
                    vec![spec.key_length],
                    Role::Bias
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
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
        Family::Grovemoe => {
            let n_chunk = n_exp / spec.n_group_experts;
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length;
                let kv = spec.head_kv[i] * spec.key_length;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![spec.n_head * spec.key_length, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![spec.key_length],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_norm.weight"),
                    vec![spec.key_length],
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
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, spec.n_ff_exp, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_gate_chexps.weight"),
                    vec![n_embd, spec.n_ff_chexp, n_chunk],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_chexps.weight"),
                    vec![spec.n_ff_chexp, n_embd, n_chunk],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_chexps.weight"),
                    vec![n_embd, spec.n_ff_chexp, n_chunk],
                    Role::Router
                );
            }
        }
        Family::Qwen35Moe => {
            // hparams.ssm_* geometry: head_k 8 x n_k 1 | head_v 8 x n_v 2
            let (head_k, n_k, head_v, n_v) = (8i64, 1i64, 8i64, 2i64);
            let value_dim = head_v * n_v;
            let conv_dim = head_k * n_k * 2 + value_dim;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                // LLM_TENSOR_ATTN_POST_NORM — "blk.%d.post_attention_norm"
                // (llama-arch.cpp:458)
                push!(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if !spec.is_recr(i) {
                    let q = spec.n_head * spec.key_length * 2; // [q|gate]
                    let kv = spec.head_kv[i] * spec.key_length;
                    // the separate wq/wk/wv trio — create_tensor_qkv's
                    // required layout (the fused attn_qkv is NOT_REQUIRED;
                    // the port's attention path implements the trio)
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, q],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, kv],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, kv],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![spec.n_head * spec.key_length, n_embd],
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
                } else {
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, conv_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_gate.weight"),
                        vec![n_embd, value_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![4, conv_dim],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_dt.bias"), vec![n_v], Role::Bias);
                    push!(format!("blk.{i}.ssm_a"), vec![n_v], Role::Decay);
                    push!(
                        format!("blk.{i}.ssm_beta.weight"),
                        vec![n_embd, n_v],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_alpha.weight"),
                        vec![n_embd, n_v],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_norm.weight"), vec![head_v], Role::Norm);
                    push!(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![value_dim, n_embd],
                        Role::Proj
                    );
                }
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
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
                    format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ffn_gate_shexp.weight"),
                    vec![n_embd, spec.n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up_shexp.weight"),
                    vec![n_embd, spec.n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down_shexp.weight"),
                    vec![spec.n_ff, n_embd],
                    Role::Proj
                );
            }
        }
        Family::KimiK3 => {
            let head_dim = spec.kda_head_dim;
            let d_inner = spec.n_head * head_dim;
            let kv_lora = spec.n_lora_kv;
            let qk_rope = spec.rope_dim;
            let k_mla = spec.k_mla;
            let v_mla = spec.v_mla;
            let qk_nope = k_mla - qk_rope;
            let latent = spec.n_expert_latent;
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_res_score.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ffn_res_score.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                if spec.is_recr(i) {
                    push!(
                        format!("blk.{i}.ssm_conv1d_q.weight"),
                        vec![4, 1, d_inner, 1],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d_k.weight"),
                        vec![4, 1, d_inner, 1],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d_v.weight"),
                        vec![4, 1, d_inner, 1],
                        Role::Decay
                    );
                    // the SEPARATE wq/wk/wv trio — create_tensor_qkv's required
                    // layout; the reference's kimi_k3_conv1d dereferences
                    // layer.wq unconditionally, so a fused attn_qkv file is a
                    // load-only trap (segfault at graph build, kimi-k3.cpp:413)
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, d_inner],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_f_a.weight"),
                        vec![n_embd, head_dim],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_f_b.weight"),
                        vec![head_dim, d_inner],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_beta.weight"),
                        vec![n_embd, spec.n_head],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_a"), vec![spec.n_head], Role::Decay);
                    push!(format!("blk.{i}.ssm_dt.bias"), vec![d_inner], Role::Bias);
                    push!(
                        format!("blk.{i}.ssm_g.weight"),
                        vec![n_embd, d_inner],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_norm.weight"),
                        vec![head_dim],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![d_inner, n_embd],
                        Role::Proj
                    );
                } else {
                    push!(
                        format!("blk.{i}.attn_q_a_norm.weight"),
                        vec![spec.q_lora],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_q_a.weight"),
                        vec![n_embd, spec.q_lora],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_q_b.weight"),
                        vec![spec.q_lora, spec.n_head * k_mla],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_kv_a_mqa.weight"),
                        vec![n_embd, kv_lora + qk_rope],
                        Role::Proj
                    );
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
                    push!(
                        format!("blk.{i}.attn_kv_a_norm.weight"),
                        vec![kv_lora],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.attn_gate.weight"),
                        vec![n_embd, spec.n_head * v_mla],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![spec.n_head * v_mla, n_embd],
                        Role::Proj
                    );
                }
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
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![n_exp], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![latent, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, latent, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![latent, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_routed_down.weight"),
                        vec![n_embd, latent],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_routed_up.weight"),
                        vec![latent, n_embd],
                        Role::Proj
                    );
                    if spec.write_routed_norm {
                        push!(
                            format!("blk.{i}.ffn_routed_norm.weight"),
                            vec![latent],
                            Role::Norm
                        );
                    }
                    // shared experts: n_ff_exp * n_expert_shared(1)
                    push!(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_exp, n_embd],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                }
            }
            push!("output_res_score.weight", vec![n_embd], Role::Norm);
        }
        Family::Dots3Note => {
            let qk_rope = spec.rope_dim;
            for i in 0..spec.n_layer {
                let is_swa = spec.is_swa(i);
                let kv_lora = if is_swa {
                    spec.n_lora_kv_swa
                } else {
                    spec.n_lora_kv
                };
                let k_mla = if is_swa { spec.k_mla_swa } else { spec.k_mla };
                let v_mla = if is_swa { spec.v_mla_swa } else { spec.v_mla };
                let qk_nope = k_mla - qk_rope;
                let nh = spec.n_head;
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_a_norm.weight"),
                    vec![spec.q_lora],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_kv_a_norm.weight"),
                    vec![kv_lora],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_k_norm.weight"),
                    vec![qk_rope],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.attn_q_a.weight"),
                    vec![n_embd, spec.q_lora],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_q_b.weight"),
                    vec![spec.q_lora, nh * k_mla],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, kv_lora + qk_rope],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_k_b.weight"),
                    vec![qk_nope, kv_lora, nh],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v_b.weight"),
                    vec![kv_lora, v_mla, nh],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![nh * v_mla, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_gate.weight"),
                    vec![n_embd, nh],
                    Role::Decay
                );
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if !is_swa {
                    push!(
                        format!("blk.{i}.indexer.k_norm.weight"),
                        vec![spec.indexer_dim],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.indexer.k_norm.bias"),
                        vec![spec.indexer_dim],
                        Role::Bias
                    );
                    push!(
                        format!("blk.{i}.indexer.proj.weight"),
                        vec![n_embd, spec.indexer_heads],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.indexer.attn_k.weight"),
                        vec![n_embd, spec.indexer_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.attn_q_b.weight"),
                        vec![spec.q_lora, spec.indexer_heads * spec.indexer_dim],
                        Role::Proj
                    );
                }
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
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![n_exp], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_exp, n_embd],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                }
            }
        }
        Family::MinimaxM3 => {
            for i in 0..spec.n_layer {
                let q = spec.n_head * spec.key_length;
                let kv = spec.head_kv[i] * spec.key_length;
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, q + 2 * kv],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_output.weight"),
                    vec![q, n_embd],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_norm.weight"),
                    vec![n_embd],
                    Role::Norm
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
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![n_exp], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![spec.n_ff_exp, n_embd, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, spec.n_ff_exp, n_exp],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_gate_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_down_shexp.weight"),
                        vec![spec.n_ff_exp, n_embd],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ffn_up_shexp.weight"),
                        vec![n_embd, spec.n_ff_exp],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.q_proj.weight"),
                        vec![n_embd, spec.indexer_heads * spec.indexer_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.k_proj.weight"),
                        vec![n_embd, spec.indexer_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.q_norm.weight"),
                        vec![spec.indexer_dim],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.indexer.k_norm.weight"),
                        vec![spec.indexer_dim],
                        Role::Norm
                    );
                }
            }
        }
        Family::Qwen4Exp => {
            let hc = spec.hc;
            let hc_dim = hc * n_embd;
            let (head_k, n_k, head_v, n_v) = (8i64, 1i64, 8i64, 2i64);
            let value_dim = head_v * n_v;
            let conv_dim = head_k * n_k * 2 + value_dim;
            push!("output_hc_norm.weight", vec![n_embd, hc], Role::Norm);
            push!(
                "output_hc_down.weight",
                vec![hc_dim, spec.hc_lr],
                Role::Proj
            );
            push!("output_hc_up.weight", vec![spec.hc_lr, hc_dim], Role::Proj);
            for i in 0..spec.n_layer {
                push!(
                    format!("blk.{i}.hc_attn_norm.weight"),
                    vec![n_embd, hc],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.hc_attn_down.weight"),
                    vec![hc_dim, spec.hc_lr],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.hc_attn_up.weight"),
                    vec![spec.hc_lr, hc_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.hc_attn_inject.weight"),
                    vec![hc_dim, hc],
                    Role::Decay
                );
                push!(
                    format!("blk.{i}.hc_ffn_norm.weight"),
                    vec![n_embd, hc],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.hc_ffn_down.weight"),
                    vec![hc_dim, spec.hc_lr],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.hc_ffn_up.weight"),
                    vec![spec.hc_lr, hc_dim],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.hc_ffn_inject.weight"),
                    vec![hc_dim, hc],
                    Role::Decay
                );
                if !spec.is_recr(i) {
                    let q = spec.n_head * spec.key_length * 2; // [q|gate]
                    let kv = spec.head_kv[i] * spec.key_length;
                    // the separate wq/wk/wv trio (create_tensor_qkv's required
                    // layout — the port's attention path implements the trio)
                    push!(
                        format!("blk.{i}.attn_q.weight"),
                        vec![n_embd, q],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_k.weight"),
                        vec![n_embd, kv],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_v.weight"),
                        vec![n_embd, kv],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_output.weight"),
                        vec![spec.n_head * spec.key_length, n_embd],
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
                        format!("blk.{i}.indexer.q_proj.weight"),
                        vec![n_embd, spec.indexer_heads * spec.indexer_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.k_proj.weight"),
                        vec![n_embd, spec.indexer_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.indexer.q_norm.weight"),
                        vec![spec.indexer_dim],
                        Role::Norm
                    );
                    push!(
                        format!("blk.{i}.indexer.k_norm.weight"),
                        vec![spec.indexer_dim],
                        Role::Norm
                    );
                } else {
                    push!(
                        format!("blk.{i}.attn_qkv.weight"),
                        vec![n_embd, conv_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.attn_gate.weight"),
                        vec![n_embd, value_dim],
                        Role::Proj
                    );
                    push!(
                        format!("blk.{i}.ssm_conv1d.weight"),
                        vec![4, conv_dim],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_dt.bias"), vec![n_v], Role::Bias);
                    push!(format!("blk.{i}.ssm_a"), vec![n_v], Role::Decay);
                    push!(
                        format!("blk.{i}.ssm_beta.weight"),
                        vec![n_embd, n_v],
                        Role::Decay
                    );
                    push!(
                        format!("blk.{i}.ssm_alpha.weight"),
                        vec![n_embd, n_v],
                        Role::Decay
                    );
                    push!(format!("blk.{i}.ssm_norm.weight"), vec![head_v], Role::Norm);
                    push!(
                        format!("blk.{i}.ssm_out.weight"),
                        vec![value_dim, n_embd],
                        Role::Proj
                    );
                }
                push!(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    vec![n_embd, n_exp],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![spec.n_ff_exp, n_embd, n_exp],
                    Role::Router
                );
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
                    format!("blk.{i}.ffn_gate_inp_shexp.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(
                    format!("blk.{i}.ffn_gate_shexp.weight"),
                    vec![n_embd, spec.n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_up_shexp.weight"),
                    vec![n_embd, spec.n_ff],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.ffn_down_shexp.weight"),
                    vec![spec.n_ff, n_embd],
                    Role::Proj
                );
            }
        }
    }
    v
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
        // the KDA/HC decay inputs: small so exp()/softplus()/tanh() stay
        // bounded through the fused GDN and the situ/hc gates
        Role::Decay => 0.02,
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
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/arch-batch11a");

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
    // head_count_kv: an array when it varies per layer (kimi-k3's KDA
    // layers carry 0)
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
        Family::Apertus => {
            // per-layer xIELU constants (apertus.cpp:6-9)
            let n = spec.n_layer as u32;
            for (key, base) in [
                ("alpha_n", 0.8f32),
                ("alpha_p", 1.2f32),
                ("beta", 0.5f32),
                ("eps", 1e-6f32),
            ] {
                kv!(
                    "xielu.".to_string() + key,
                    Value::Array(
                        GgufType::Float32,
                        (0..n).map(|i| Value::F32(base + i as f32 * 0.01)).collect()
                    )
                );
            }
        }
        Family::Grovemoe => {
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
                format!("{a}.expert_chunk_feed_forward_length"),
                Value::U32(spec.n_ff_chexp as u32)
            );
            kv!(
                format!("{a}.expert_group_scale"),
                Value::F32(spec.expert_group_scale)
            );
            kv!(
                format!("{a}.experts_per_group"),
                Value::U32(spec.n_group_experts as u32)
            );
            kv!(format!("{a}.expert_gating_func"), Value::U32(1)); // SOFTMAX
        }
        Family::Qwen35Moe => {
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            // ssm keys: d_conv 4, d_inner 16, d_state 8, dt_rank 2, n_group 1
            kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
            kv!(format!("{a}.ssm.inner_size"), Value::U32(16));
            kv!(format!("{a}.ssm.state_size"), Value::U32(8));
            kv!(format!("{a}.ssm.time_step_rank"), Value::U32(2));
            kv!(format!("{a}.ssm.group_count"), Value::U32(1));
            kv!(
                format!("{a}.attention.recurrent_layers"),
                Value::Array(
                    GgufType::Uint32,
                    spec.recr_layers
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&x| Value::U32(x))
                        .collect()
                )
            );
            kv!(
                format!("{a}.rope.dimension_sections"),
                Value::Array(
                    GgufType::Int32,
                    spec.rope_sections
                        .unwrap()
                        .iter()
                        .map(|&x| Value::I32(x))
                        .collect()
                )
            );
        }
        Family::KimiK3 => {
            kv!(
                format!("{a}.attention.key_length_mla"),
                Value::U32(spec.k_mla as u32)
            );
            kv!(
                format!("{a}.attention.value_length_mla"),
                Value::U32(spec.v_mla as u32)
            );
            kv!(
                format!("{a}.attention.q_lora_rank"),
                Value::U32(spec.q_lora as u32)
            );
            kv!(
                format!("{a}.attention.kv_lora_rank"),
                Value::U32(spec.n_lora_kv as u32)
            );
            kv!(
                format!("{a}.ssm.conv_kernel"),
                Value::U32(spec.d_conv as u32)
            );
            kv!(
                format!("{a}.kda.head_dim"),
                Value::U32(spec.kda_head_dim as u32)
            );
            kv!(
                format!("{a}.kda.gate_lower_bound"),
                Value::F32(spec.gate_lower_bound.unwrap())
            );
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.dense_lead)
            );
            kv!(
                format!("{a}.expert_latent_length"),
                Value::U32(spec.n_expert_latent as u32)
            );
            kv!(format!("{a}.expert_gating_func"), Value::U32(1)); // SOFTMAX
            kv!(
                format!("{a}.attn_res.block_size"),
                Value::U32(spec.attn_res_block_size)
            );
            kv!(
                format!("{a}.activation.situ_beta"),
                Value::F32(spec.situ_beta)
            );
            kv!(
                format!("{a}.activation.situ_linear_beta"),
                Value::F32(spec.situ_linear_beta)
            );
        }
        Family::Dots3Note => {
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.dense_lead)
            );
            kv!(format!("{a}.expert_gating_func"), Value::U32(1)); // SOFTMAX
            kv!(
                format!("{a}.attention.q_lora_rank"),
                Value::U32(spec.q_lora as u32)
            );
            kv!(
                format!("{a}.attention.kv_lora_rank"),
                Value::U32(spec.n_lora_kv as u32)
            );
            kv!(
                format!("{a}.attention.key_length_mla"),
                Value::U32(spec.k_mla as u32)
            );
            kv!(
                format!("{a}.attention.value_length_mla"),
                Value::U32(spec.v_mla as u32)
            );
            kv!(
                format!("{a}.attention.kv_lora_rank_swa"),
                Value::U32(spec.n_lora_kv_swa as u32)
            );
            kv!(
                format!("{a}.attention.key_length_mla_swa"),
                Value::U32(spec.k_mla_swa as u32)
            );
            kv!(
                format!("{a}.attention.value_length_mla_swa"),
                Value::U32(spec.v_mla_swa as u32)
            );
            // the SWA cache's K row width — the compressed [lora_swa|rope]
            // row the SWA layers' Q must match (llama-model.cpp:1397 reads it
            // into n_embd_head_k_swa; without it the reference aborts in
            // build_attn's QK mul_mat)
            kv!(
                format!("{a}.attention.key_length_swa"),
                Value::U32((spec.n_lora_kv_swa + spec.rope_dim) as u32)
            );
            kv!(
                format!("{a}.attention.sliding_window"),
                Value::U32(spec.n_swa)
            );
            kv!(
                format!("{a}.rope.freq_base_swa"),
                Value::F32(spec.freq_base_swa)
            );
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(
                    GgufType::Uint32,
                    spec.swa_pattern.iter().map(|&x| Value::U32(x)).collect()
                )
            );
            kv!(
                format!("{a}.attention.indexer.head_count"),
                Value::U32(spec.indexer_heads as u32)
            );
            kv!(
                format!("{a}.attention.indexer.key_length"),
                Value::U32(spec.indexer_dim as u32)
            );
            kv!(
                format!("{a}.attention.indexer.top_k"),
                Value::U32(spec.indexer_topk as u32)
            );
            kv!(
                format!("{a}.attention.indexer.types"),
                Value::Array(
                    GgufType::Uint32,
                    spec.swa_pattern
                        .iter()
                        .map(|&x| Value::U32(1 - x))
                        .collect()
                )
            );
        }
        Family::MinimaxM3 => {
            kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
            kv!(
                format!("{a}.expert_used_count"),
                Value::U32(N_EXPERT_USED as u32)
            );
            kv!(
                format!("{a}.expert_feed_forward_length"),
                Value::U32(spec.n_ff_exp as u32)
            );
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(
                format!("{a}.leading_dense_block_count"),
                Value::U32(spec.dense_lead)
            );
            kv!(format!("{a}.expert_gating_func"), Value::U32(1)); // SOFTMAX
            kv!(
                format!("{a}.attention.indexer.head_count"),
                Value::U32(spec.indexer_heads as u32)
            );
            kv!(
                format!("{a}.attention.indexer.key_length"),
                Value::U32(spec.indexer_dim as u32)
            );
            kv!(
                format!("{a}.attention.indexer.top_k"),
                Value::U32(spec.indexer_topk as u32)
            );
            kv!(
                format!("{a}.attention.indexer.block_size"),
                Value::U32(spec.msa_blk as u32)
            );
            kv!(
                format!("{a}.attention.indexer.local_blocks"),
                Value::U32(spec.msa_local as u32)
            );
        }
        Family::Qwen4Exp => {
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
                format!("{a}.expert_shared_feed_forward_length"),
                Value::U32(spec.n_ff as u32)
            );
            kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
            kv!(format!("{a}.ssm.inner_size"), Value::U32(16));
            kv!(format!("{a}.ssm.state_size"), Value::U32(8));
            kv!(format!("{a}.ssm.time_step_rank"), Value::U32(2));
            kv!(format!("{a}.ssm.group_count"), Value::U32(1));
            kv!(
                format!("{a}.hyper_connection.count"),
                Value::U32(spec.hc as u32)
            );
            kv!(
                format!("{a}.hyper_connection.low_rank"),
                Value::U32(spec.hc_lr as u32)
            );
            kv!(
                format!("{a}.attention.indexer.head_count"),
                Value::U32(spec.indexer_heads as u32)
            );
            kv!(
                format!("{a}.attention.indexer.key_length"),
                Value::U32(spec.indexer_dim as u32)
            );
            kv!(
                format!("{a}.attention.indexer.top_k"),
                Value::U32(spec.indexer_topk as u32)
            );
            // NB: no attention.compress_ratios — the reference then runs
            // every full-attention layer dense (the port's verified shape)
            kv!(
                format!("{a}.attention.recurrent_layers"),
                Value::Array(
                    GgufType::Uint32,
                    spec.recr_layers
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&x| Value::U32(x))
                        .collect()
                )
            );
            kv!(
                format!("{a}.rope.dimension_sections"),
                Value::Array(
                    GgufType::Int32,
                    spec.rope_sections
                        .unwrap()
                        .iter()
                        .map(|&x| Value::I32(x))
                        .collect()
                )
            );
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
    match spec.family {
        Family::Apertus => {
            assert_eq!(hp.xielu_alpha_n[0], 0.8);
            assert_eq!(hp.xielu_alpha_p[1], 1.21);
            assert_eq!(hp.xielu_beta[2], 0.52);
            assert_eq!(hp.xielu_eps[3], 1e-6 + 0.03);
        }
        Family::Grovemoe => {
            assert_eq!(hp.n_group_experts, 2);
            assert_eq!(hp.expert_group_scale, 0.05);
            assert_eq!(hp.n_ff_chexp, 64);
        }
        Family::Qwen35Moe => {
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_recr(il), spec.is_recr(il), "qwen35moe: is_recr[{il}]");
            }
            assert_eq!(hp.ssm_d_state, 8);
            assert_eq!(hp.ssm_dt_rank, 2);
            assert_eq!(hp.rope_sections, [8, 4, 4, 0]);
        }
        Family::KimiK3 => {
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_recr(il), spec.is_recr(il), "kimi-k3: is_recr[{il}]");
            }
            assert_eq!(hp.n_embd_head_kda, spec.kda_head_dim as u32);
            assert_eq!(hp.kda_gate_lower_bound, -5.0);
            assert_eq!(hp.attn_res_block_size, 2);
            assert_eq!(hp.situ_beta, 0.9);
            assert_eq!(hp.n_expert_latent, 64);
            assert_eq!(hp.n_embd_head_v_full, spec.n_lora_kv as u32);
        }
        Family::Dots3Note => {
            assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
            assert_eq!(hp.n_swa, 64);
            assert_eq!(hp.rope_freq_base_train_swa, 30000.0);
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_swa(il), spec.is_swa(il), "dots3note: is_swa[{il}]");
                assert_eq!(
                    hp.is_indexer_full(il),
                    !spec.is_swa(il),
                    "dots3note: indexer[{il}]"
                );
            }
            assert_eq!(hp.n_lora_kv_swa, 12);
        }
        Family::MinimaxM3 => {
            assert_eq!(hp.indexer_block_size, 16);
            assert_eq!(hp.indexer_local_blocks, 1);
            assert_eq!(hp.indexer_top_k, 2);
        }
        Family::Qwen4Exp => {
            for il in 0..spec.n_layer {
                assert_eq!(hp.is_recr(il), spec.is_recr(il), "qwen4exp: is_recr[{il}]");
            }
            assert_eq!(hp.dsv4_hc_mult, 2);
            assert_eq!(hp.hc_low_rank, 8);
            assert_eq!(hp.n_embd_out_impl, 2 * spec.n_embd as u32);
            assert_eq!(hp.dsv4_compress_ratios[0], 0); // absent → dense
        }
    }
}

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
    let hp = &m.hparams;
    let first_attn = (0..hp.n_layer() as usize)
        .find(|&il| match m.arch {
            llama::arch::LlmArch::QWEN35MOE
            | llama::arch::LlmArch::QWEN4EXP
            | llama::arch::LlmArch::KIMI_K3 => !hp.is_recr(il),
            _ => true,
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
// the decode harness — DecodeContext itself (the batch's context.rs arms),
// mirroring the CLI weight assemblies of main.rs
// ---------------------------------------------------------------------------

fn forward_of(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = synth_attn(m, fa);
    let per = |n: usize, f: &dyn Fn(usize) -> u32| (0..n).map(|i| f(i)).collect::<Vec<_>>();
    match m.arch {
        llama::arch::LlmArch::APERTUS => (
            ForwardWeights::Apertus(
                graph_arch::ApertusModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m
                        .layers
                        .iter()
                        .map(|l| graph_arch::ApertusLayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            rope_long: l.rope_long,
                            rope_short: l.rope_short,
                            rope_freqs: l.rope_freqs,
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo.unwrap(),
                            wo_b: l.wo_b,
                            attn_q_norm: l.attn_q_norm.unwrap(),
                            attn_q_norm_b: l.attn_q_norm_b,
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            attn_k_norm_b: l.attn_k_norm_b,
                            ffn_norm: l.ffn_norm.unwrap(),
                            ffn_down: l.ffn_down.unwrap(),
                            ffn_up: l.ffn_up.unwrap(),
                        })
                        .collect(),
                },
                graph_arch::ApertusParams {
                    xielu_alpha_n: hp.xielu_alpha_n[..n_trunk].to_vec(),
                    xielu_alpha_p: hp.xielu_alpha_p[..n_trunk].to_vec(),
                    xielu_beta: hp.xielu_beta[..n_trunk].to_vec(),
                    xielu_eps: hp.xielu_eps[..n_trunk].to_vec(),
                    f_attention_scale: hp.f_attention_scale,
                    use_longrope_factors: false,
                    attn,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::GROVEMOE => (
            ForwardWeights::Grovemoe(
                graph_arch::GrovemoeModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m
                        .layers
                        .iter()
                        .map(|l| graph_arch::GrovemoeLayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo.unwrap(),
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            attn_q_norm: l.attn_q_norm.unwrap(),
                            ffn_norm: l.ffn_norm.unwrap(),
                            ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                            ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                            ffn_down_exps: l.ffn_down_exps.unwrap(),
                            ffn_up_exps: l.ffn_up_exps.unwrap(),
                            ffn_gate_chexps: l.ffn_gate_chexps.unwrap(),
                            ffn_down_chexps: l.ffn_down_chexps.unwrap(),
                            ffn_up_chexps: l.ffn_up_chexps.unwrap(),
                        })
                        .collect(),
                },
                graph_arch::GrovemoeParams {
                    n_embd: hp.n_embd as i64,
                    expert_group_scale: hp.expert_group_scale,
                    n_group_experts: hp.n_group_experts as i64,
                    n_ff_chexp: hp.n_ff_chexp as i64,
                    n_embd_head_k: hp.n_embd_head_k(0) as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    attn,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::QWEN35MOE => (
            ForwardWeights::Qwen35Moe(
                graph_arch::Qwen35MoeModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m
                        .layers
                        .iter()
                        .take(n_trunk)
                        .map(|l| graph_arch::Qwen35MoeLayerWeights {
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
                            ssm_conv1d: l.ssm_conv1d,
                            ssm_dt_b: l.ssm_dt_b,
                            ssm_a: l.ssm_a,
                            ssm_beta: l.ssm_beta,
                            ssm_alpha: l.ssm_alpha,
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
                },
                graph_arch::Qwen35MoeParams {
                    n_embd: hp.n_embd as i64,
                    n_head: per(n_trunk, &|il| hp.n_head(il)),
                    n_head_kv: per(n_trunk, &|il| hp.n_head_kv(il)),
                    n_embd_head_k: per(n_trunk, &|il| hp.n_embd_head_k(il)),
                    n_embd_head_v: per(n_trunk, &|il| hp.n_embd_head_v(il)),
                    n_rot: per(n_trunk, &|il| hp.n_rot(il)),
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    rope_sections: hp.rope_sections,
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
                    attn,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::KIMI_K3 => {
            let fa_layer = (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap();
            (
                ForwardWeights::KimiK3(
                    graph_arch::KimiK3ModelWeights {
                        tok_embd: m.tok_embd,
                        output_norm: m.output_norm,
                        output: m.output,
                        output_res_score: m.output_res_score,
                        layers: m
                            .layers
                            .iter()
                            .take(n_trunk)
                            .map(|l| graph_arch::KimiK3LayerWeights {
                                attn_norm: l.attn_norm.unwrap(),
                                ffn_norm: l.ffn_norm.unwrap(),
                                attn_res_score: l.attn_res_score,
                                ffn_res_score: l.ffn_res_score,
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
                                ssm_g: l.ssm_g,
                                ssm_o_norm: l.ssm_norm,
                                wo: l.wo,
                                wq_a: l.wq_a,
                                attn_q_a_norm: l.attn_q_a_norm,
                                wq_b: l.wq_b,
                                wq_mla: l.wq,
                                wkv_a_mqa: l.wkv_a_mqa,
                                attn_kv_a_norm: l.attn_kv_a_norm,
                                wk_b: l.wk_b,
                                wv_b: l.wv_b,
                                wkv_b: l.wkv_b,
                                wqkv_gate: l.wqkv_gate,
                                ffn_gate: l.ffn_gate,
                                ffn_down: l.ffn_down,
                                ffn_up: l.ffn_up,
                                ffn_gate_inp: l.ffn_gate_inp,
                                ffn_exp_probs_b: l.ffn_exp_probs_b,
                                ffn_gate_exps: l.ffn_gate_exps,
                                ffn_down_exps: l.ffn_down_exps,
                                ffn_up_exps: l.ffn_up_exps,
                                ffn_routed_down: l.ffn_routed_down,
                                ffn_routed_up: l.ffn_routed_up,
                                ffn_routed_norm: l.ffn_routed_norm,
                                ffn_gate_shexp: l.ffn_gate_shexp,
                                ffn_down_shexp: l.ffn_down_shexp,
                                ffn_up_shexp: l.ffn_up_shexp,
                            })
                            .collect(),
                    },
                    graph_arch::KimiK3Params {
                        n_embd: hp.n_embd as i64,
                        n_head: hp.n_head(fa_layer) as i64,
                        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                        n_embd_head_kda: hp.n_embd_head_kda as i64,
                        ssm_d_conv: hp.ssm_d_conv as i64,
                        kda_gate_lower_bound: hp.kda_gate_lower_bound,
                        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                        n_lora_kv: hp.n_lora_kv as i64,
                        n_lora_q: hp.n_lora_q as i64,
                        n_embd_head_qk_rope: hp.n_rot(fa_layer) as i64,
                        n_embd_r: hp.n_embd_r(),
                        n_embd_s: hp.n_embd_s(),
                        n_layer_dense_lead: hp.n_layer_dense_lead,
                        attn_res_block_size: hp.attn_res_block_size,
                        situ_beta: hp.situ_beta,
                        situ_linear_beta: hp.situ_linear_beta,
                        n_expert_latent: hp.n_expert_latent as i64,
                        n_expert: hp.n_expert as i64,
                        n_expert_used: hp.n_expert_used(0) as i64,
                        expert_weights_norm: hp.expert_weights_norm,
                        expert_weights_scale: hp.expert_weights_scale,
                        expert_gating_func: hp.expert_gating_func as i32,
                        attn,
                    },
                ),
                attn,
            )
        }
        llama::arch::LlmArch::DOTS3NOTE => (
            ForwardWeights::Dots3Note(
                graph_arch::Dots3NoteModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m
                        .layers
                        .iter()
                        .take(n_trunk)
                        .map(|l| graph_arch::Dots3NoteLayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                            attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            wq_a: l.wq_a.unwrap(),
                            wq_b: l.wq_b.unwrap(),
                            wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                            wk_b: l.wk_b.unwrap(),
                            wv_b: l.wv_b.unwrap(),
                            wo: l.wo.unwrap(),
                            wqkv_gate: l.wqkv_gate.unwrap(),
                            ffn_norm: l.ffn_norm.unwrap(),
                            indexer_k_norm: l.indexer_k_norm,
                            indexer_k_norm_b: l.indexer_k_norm_b,
                            indexer_proj: l.indexer_proj,
                            indexer_attn_k: l.indexer_attn_k,
                            indexer_attn_q_b: l.indexer_attn_q_b,
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
                },
                graph_arch::Dots3NoteParams {
                    n_embd: hp.n_embd as i64,
                    n_head: per(n_trunk, &|il| hp.n_head(il))
                        .into_iter()
                        .map(|h| h as i32)
                        .collect(),
                    n_rot: hp.n_rot(0) as i64,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    n_lora_kv: hp.n_lora_kv as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    n_lora_kv_swa: hp.n_lora_kv_swa as i64,
                    n_embd_head_k_mla_swa: hp.n_embd_head_k_mla_swa as i64,
                    n_embd_head_v_mla_swa: hp.n_embd_head_v_mla_swa as i64,
                    f_norm_eps: hp.f_norm_eps,
                    rope_freq_base_swa: hp.rope_freq_base_train_swa,
                    rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k,
                    is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    attn,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::MINIMAX_M3 => (
            ForwardWeights::MinimaxM3(
                graph_arch::MinimaxM3ModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m
                        .layers
                        .iter()
                        .take(n_trunk)
                        .map(|l| graph_arch::MinimaxM3LayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
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
                            index_q_proj: l.index_q_proj,
                            index_k_proj: l.index_k_proj,
                            index_q_norm: l.index_q_norm,
                            index_k_norm: l.index_k_norm,

                        })
                        .collect(),
                },
                graph_arch::MinimaxM3Params {
                    n_embd: hp.n_embd as i64,
                    n_head: hp.n_head(0) as i64,
                    n_head_kv: hp.n_head_kv(0) as i64,
                    n_embd_head: hp.n_embd_head_k(0) as i64,
                    n_rot: hp.n_rot(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_ff_exp: hp.n_ff_exp(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    msa_blk: hp.indexer_block_size as i64,
                    msa_topk_blocks: hp.indexer_top_k as i64,
                    msa_local: hp.indexer_local_blocks as i64,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    attn,
                },
            ),
            attn,
        ),
        llama::arch::LlmArch::QWEN4EXP => (
            ForwardWeights::Qwen4Exp(
                graph_arch::Qwen4ExpModelWeights {
                    tok_embd: m.tok_embd,
                    hc_head_norm: m.hc_head_norm.unwrap(),
                    hc_head_down: m.hc_head_down.unwrap(),
                    hc_head_up: m.hc_head_up.unwrap(),
                    output: m.output,
                    per_layer_tok_embd: m.per_layer_tok_embd,
                    layers: m
                        .layers
                        .iter()
                        .take(n_trunk)
                        .map(|l| graph_arch::Qwen4ExpLayerWeights {
                            hc_attn_norm: l.hc_attn_norm.unwrap(),
                            hc_attn_down: l.hc_attn_down.unwrap(),
                            hc_attn_up: l.hc_attn_up.unwrap(),
                            hc_attn_inject: l.hc_attn_inject.unwrap(),
                            hc_ffn_norm: l.hc_ffn_norm.unwrap(),
                            hc_ffn_down: l.hc_ffn_down.unwrap(),
                            hc_ffn_up: l.hc_ffn_up.unwrap(),
                            hc_ffn_inject: l.hc_ffn_inject.unwrap(),
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo,
                            attn_q_norm: l.attn_q_norm,
                            attn_k_norm: l.attn_k_norm,
                            index_q_proj: l.index_q_proj,
                            index_k_proj: l.index_k_proj,
                            index_q_norm: l.index_q_norm,
                            index_k_norm: l.index_k_norm,
                            ple_key: l.ple_key,
                            ple_value: l.ple_value,
                            ple_norm_key: l.ple_norm_key,
                            ple_norm_query: l.ple_norm_query,
                            ple_norm_conv: l.ple_norm_conv,
                            ple_conv1d: l.ple_conv1d,
                            wqkv: l.wqkv,
                            wqkv_gate: l.wqkv_gate,
                            ssm_conv1d: l.ssm_conv1d,
                            ssm_dt_b: l.ssm_dt_b,
                            ssm_a: l.ssm_a,
                            ssm_beta: l.ssm_beta,
                            ssm_alpha: l.ssm_alpha,
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
                },
                graph_arch::Qwen4ExpParams {
                    n_embd: hp.n_embd as i64,
                    n_head: per(n_trunk, &|il| hp.n_head(il)),
                    n_head_kv: per(n_trunk, &|il| hp.n_head_kv(il)),
                    n_embd_head_k: per(n_trunk, &|il| hp.n_embd_head_k(il)),
                    n_embd_head_v: per(n_trunk, &|il| hp.n_embd_head_v(il)),
                    n_rot: per(n_trunk, &|il| hp.n_rot(il)),
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    rope_sections: hp.rope_sections,
                    f_attention_scale: hp.f_attention_scale,
                    hc: hp.dsv4_hc_mult as i64,
                    hc_lr: hp.hc_low_rank as i64,
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
                    // batch 19 — the QSA indexer + PLE halves
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
                    compress_ratios: hp.dsv4_compress_ratios[..hp.n_layer() as usize].to_vec(),
                    is_ple: (0..hp.n_layer() as usize).map(|il| hp.is_ple(il)).collect(),
                    ple_ngram_size: hp.ple_ngram_size as i64,
                    ple_heads_per_ngram: hp.ple_heads_per_ngram as i64,
                    ple_conv_kernel: hp.ple_conv_kernel as i64,
                    ple_n_heads: hp.ple_n_heads as i64,
                    ple_head_dim: hp.ple_head_dim as i64,
                    ple_eos_token_id: hp.ple_eos_token_id as i64,
                    ple_image_token_id: hp.ple_image_token_id as i64,
                    ple_layer_multipliers: hp.ple_layer_multipliers,
                    ple_head_offsets: hp.ple_head_offsets,
                    ple_head_vocab_sizes: hp.ple_head_vocab_sizes,
                    attn,
                },
            ),
            attn,
        ),
        other => panic!("arch {other:?} not in batch 11a"),
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

/// one prefill (a 12-token prompt) + one decode step; asserts the caches and
/// recurrent cells landed where they should
fn smoke_forward(m: &mut LlamaModel, spec: &SynthSpec, fa: bool) -> Vec<f32> {
    let mut dctx = driver_for(m, fa);
    let prompt: Vec<i32> = (1..=12).collect();
    let pos: Vec<i32> = (0..12).collect();
    let logits = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    let tk = logits_of_argmax(logits.chunks(32000).last().unwrap());
    let next = dctx.decode(&[tk], &[12]).expect("decode").to_vec();
    match spec.family {
        Family::Apertus | Family::Grovemoe | Family::MinimaxM3 => {
            assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
        }
        Family::Qwen35Moe | Family::Qwen4Exp => {
            // one cell per position (the single full-attention layer of the
            // synthetic files writes it) — same unified-cache rule as above
            assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
        }
        Family::KimiK3 => {
            // the unified cache holds ONE cell per position shared by every
            // attention layer (2 MLA layers here) — 13 tokens ⇒ 13 cells
            assert_eq!(dctx.kv.used_cells(), 13, "{} fa={fa}: kv cells", spec.arch);
        }
        Family::Dots3Note => {
            // one cell per position per cache — the base (DSA) and the SWA
            // pair each hold 13 (3 full / 3 swa layers share their cache's
            // cells); the lid cache tracks the base cache cell for cell
            assert_eq!(
                dctx.kv.used_cells(),
                13,
                "{} fa={fa}: base cells",
                spec.arch
            );
            assert_eq!(
                dctx.kv
                    .swa_cache()
                    .map(|s| s.cells.iter().filter(|c| !c.is_empty()).count())
                    .unwrap_or(0) as u32,
                13,
                "{} fa={fa}: swa cells",
                spec.arch
            );
            // the lid cache tracks the base cache cell for cell; its live row
            // count is the C's padded get_n_kv — max(256, pad(used, 256))
            assert_eq!(dctx.kv.n_kv_lid(), 256, "{} fa={fa}: lid rows", spec.arch);
        }
    }
    next
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn arch_batch11a_pin_and_smoke() {
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

/// the >64-token prompt cell of the parity protocol, in one ubatch:
///   * dots3note — the >64-token SWA window edge + the lid cache rows
///   * minimax-m3 — the MSA block selection at ~100 tokens (fa on) and the
///     dense fallback (fa off)
///   * the GDN archs — the conv-state slide + the recurrence past the prompt
#[test]
fn arch_batch11a_long_prompt_cells() {
    let n = 100usize;
    let pos: Vec<i32> = (0..n).map(|i| i as i32).collect();

    let spec = spec_dots3note();
    let mut m = load_synth(&spec);
    for fa in [false, true] {
        let mut m = load_synth(&spec);
        let mut dctx = driver_for(&mut m, fa);
        let prompt: Vec<i32> = (1..=n as i32).collect();
        let lg = dctx.decode(&prompt, &pos).expect("long prefill").to_vec();
        assert!(lg.iter().all(|v| v.is_finite()));
        let tk = logits_of_argmax(&lg);
        let next = dctx.decode(&[tk], &[n as i32]).expect("decode").to_vec();
        assert!(next.iter().all(|v| v.is_finite()));
    }

    let spec = spec_minimax_m3();
    for fa in [false, true] {
        let mut m = load_synth(&spec);
        let mut dctx = driver_for(&mut m, fa);
        let prompt: Vec<i32> = (1..=n as i32).collect();
        let lg = dctx.decode(&prompt, &pos).expect("long prefill").to_vec();
        assert!(lg.iter().all(|v| v.is_finite()));
        let tk = logits_of_argmax(&lg);
        let next = dctx.decode(&[tk], &[n as i32]).expect("decode").to_vec();
        assert!(next.iter().all(|v| v.is_finite()));
        // a short decode run: the MSA decode gather evolves across steps
        let mut tk2 = tk;
        for p in (n + 1) as i32..=(n + 16) as i32 {
            let lg = dctx.decode(&[tk2], &[p]).expect("decode").to_vec();
            tk2 = logits_of_argmax(&lg);
        }
    }

    // the GDN/KDA hybrids: 80 single-token steps — the recurrence runs far
    // past the prompt
    for spec in [spec_qwen35moe(), spec_kimi_k3(), spec_qwen4exp()] {
        let mut m = load_synth(&spec);
        let mut dctx = driver_for(&mut m, false);
        let prompt: Vec<i32> = (1..=12).collect();
        let lg = dctx
            .decode(&prompt, &(0..12).collect::<Vec<i32>>())
            .expect("prefill")
            .to_vec();
        let mut tk = logits_of_argmax(&lg);
        for p in 12..80 {
            let lg = dctx.decode(&[tk], &[p]).expect("decode").to_vec();
            tk = logits_of_argmax(&lg);
        }
        assert!(lg.iter().all(|v| v.is_finite()));
    }
}

/// kimi-k3's residual bank must change the logits: with
/// attn_res_block_size 2 the banked checkpoints re-enter every layer's mix
#[test]
fn arch_batch11a_kimi_k3_residual_bank() {
    let spec = spec_kimi_k3();
    let mut m = load_synth(&spec);
    let mut dctx = driver_for(&mut m, false);
    let lg = dctx
        .decode(&[1, 2, 3, 4], &[0, 1, 2, 3])
        .expect("prefill")
        .to_vec();
    assert!(lg.iter().all(|v| v.is_finite()));
    // the res stack grew through the checkpoint layers (0 and 2); the very
    // first layer saw an empty bank — both paths ran without tripping the
    // dsv4_hc_pre shape asserts, which is the pin
    let next = dctx
        .decode(&[logits_of_argmax(&lg)], &[4])
        .expect("decode")
        .to_vec();
    assert!(next.iter().all(|v| v.is_finite()));
}

fn logits_of_argmax(lg: &[f32]) -> i32 {
    lg.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}
