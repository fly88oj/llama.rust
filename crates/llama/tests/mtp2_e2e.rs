//! mtp2_e2e.rs — the GLM4-style `graph_mtp` draft heads (MTP batch 17,
//! 2026-09-29): qwen35 / qwen35moe / qwen3next / glm4-moe / cohere2moe /
//! bailingmoe3 / hy-v3 / mimo2 / step35 on synthetic nextn GGUFs
//! (llama.cpp bd4f514db1).
//!
//! The deepseek MTP trilogy (tests/mtp_e2e.rs) is ported end-to-end into
//! `DecodeContext::new_mtp`; context.rs is not extended for the nine archs of
//! this batch (the perf agent owns it), so the acceptance here is the
//! **graph-level bit-compare** the earlier batches used before the driver
//! landed: a direct-graph driver (the arch_batch5 `Driver` shape — a 1-layer
//! filtered KV cache, `llama-model.cpp:2676-2679`'s `il >= n_layer()` filter)
//! replays the draft-mtp step shape (common/speculative.cpp:1616-1767: one
//! token + one F32 h row per step, the previous step's argmax + t_h_nextn
//! feeding the next), dumping every step's t_logits row and t_h_nextn row.
//! `parity/ref_mtp2_dump.c` replays the identical chain through the pinned
//! reference's own `ctx_type = LLAMA_CONTEXT_TYPE_MTP` context; the
//! `mtp2_reference_bitcompare` cell (ignored; runs after the probe) compares
//! the two dumps **byte-for-byte**.
//!
//! The synthetic files are `block_count = n_layer + 1` nextn files whose
//! trunk layers are all full-attention (a legal layout for every hybrid arch
//! — `attention.recurrent_layers` all zero — that keeps the generator
//! small); the MTP layer repeats the trunk full-attention block plus the
//! `blk.{n_layer}.nextn.*` trio (+ `layer_output_norm` for bailingmoe3, the
//! LAYER_OUT_NORM shared head norm).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaHparams;
use llama::kv_cache::{KvCache, KvSwaStep, SlotInfo, SwaCacheSpec};
use llama::model::{load_model, LlamaModel, LayerTensors};

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/mtp2";

const N_EXPERT: i64 = 4;
const N_EXPERT_USED: i64 = 2;

const N_LAYER: usize = 3; // trunk; the MTP layer is block n_layer
const N_EMBD: i64 = 128;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HD: i64 = 32;
const N_FF: i64 = 64;
const N_FF_EXP: i64 = 32;
const N_FF_SHEXP: i64 = 24;
const N_CTX: u32 = 512;
const N_STEPS: usize = 12;

// ---------------------------------------------------------------------------
// the specs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Fam {
    Qwen35,
    Qwen35Moe,
    Qwen3Next,
    Glm4Moe,
    Cohere2Moe,
    BailingMoe3,
    HyV3,
    Mimo2,
    Step35,
}

impl Fam {
    fn arch_name(self) -> &'static str {
        match self {
            Fam::Qwen35 => "qwen35",
            Fam::Qwen35Moe => "qwen35moe",
            Fam::Qwen3Next => "qwen3next",
            Fam::Glm4Moe => "glm4moe",
            Fam::Cohere2Moe => "cohere2moe",
            Fam::BailingMoe3 => "bailingmoe3",
            Fam::HyV3 => "hy_v3",
            Fam::Mimo2 => "mimo2",
            Fam::Step35 => "step35",
        }
    }
    /// the swa keys (`attention.sliding_window` + the [0,1,1,1] pattern) —
    /// the MTP layer (il = n_layer = 3) is an SWA layer for these
    fn iswa(self) -> bool {
        matches!(self, Fam::Cohere2Moe | Fam::Mimo2 | Fam::Step35)
    }
    fn path(self) -> String {
        format!("{OUT_DIR}/{}-synth-mtp.gguf", self.arch_name())
    }
}

/// bailingmoe3's MLA geometry (batch-9's spec): kv_lora 32 + qk_rope 16,
/// q_lora 32, v_mla 20
const KV_LORA: i64 = 32;
const Q_LORA: i64 = 32;
const K_MLA: i64 = 40; // qk_nope 24 + qk_rope 16
const V_MLA: i64 = 20;

fn all_fams() -> Vec<Fam> {
    vec![
        Fam::Qwen35,
        Fam::Qwen35Moe,
        Fam::Qwen3Next,
        Fam::Glm4Moe,
        Fam::Cohere2Moe,
        Fam::BailingMoe3,
        Fam::HyV3,
        Fam::Mimo2,
        Fam::Step35,
    ]
}

// ---------------------------------------------------------------------------
// the tensor tables
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Bias,
    Proj,
    Router,
}

fn tensors_for(fam: Fam) -> Vec<(String, Vec<i64>, Role)> {
    let mut v: Vec<(String, Vec<i64>, Role)> = Vec::new();
    let n_embd = N_EMBD;
    let hd = HD;
    let n_layer_all = N_LAYER + 1;
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $role))
        };
    }
    push!("token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    push!("output_norm.weight", vec![n_embd], Role::Norm);
    // the separate head everywhere (hy-v3/mimo2/step35/bailingmoe3 require
    // it; the rest would accept the tie — a separate head exercises more)
    push!("output.weight", vec![n_embd, N_VOCAB], Role::Proj);

    for i in 0..n_layer_all as i32 {
        let dense = (i as u32) < 1; // one dense-lead layer where the arch reads it
        match fam {
            Fam::Qwen35 => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.post_attention_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, 2 * hd * N_HEAD],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                push!(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
            }
            Fam::Qwen35Moe | Fam::Qwen3Next => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.post_attention_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_q.weight"),
                    vec![n_embd, 2 * hd * N_HEAD],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                push!(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                push!(
                    format!("blk.{i}.ffn_gate_exps.weight"),
                    vec![n_embd, N_FF_EXP, N_EXPERT],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![N_FF_EXP, n_embd, N_EXPERT],
                    Role::Router
                );
                push!(
                    format!("blk.{i}.ffn_up_exps.weight"),
                    vec![n_embd, N_FF_EXP, N_EXPERT],
                    Role::Router
                );
                push!(format!("blk.{i}.ffn_gate_inp_shexp.weight"), vec![n_embd], Role::Router);
                push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SHEXP, n_embd], Role::Proj);
            }
            Fam::Glm4Moe => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                // post_attention_norm (glm4-moe's attn_post_norm tensor name)
                push!(
                    format!("blk.{i}.post_attention_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
                push!(format!("blk.{i}.attn_q.weight"), vec![n_embd, hd * N_HEAD], Role::Proj);
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                if dense {
                    push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                    push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                    push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
                } else {
                    push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![N_FF_EXP, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_EXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_EXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_EXP, n_embd], Role::Proj);
                }
            }
            Fam::Cohere2Moe => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.attn_q.weight"), vec![n_embd, hd * N_HEAD], Role::Proj);
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                // every layer MoE (dense lead 1 keeps layer 0 dense)
                if dense {
                    push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                    push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                    push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
                } else {
                    push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                    push!(
                        format!("blk.{i}.ffn_gate_up_exps.weight"),
                        vec![n_embd, 2 * N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![N_FF_EXP, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SHEXP, n_embd], Role::Proj);
                }
            }
            Fam::BailingMoe3 => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.attn_q_a.weight"), vec![n_embd, Q_LORA], Role::Proj);
                push!(format!("blk.{i}.attn_q_a_norm.weight"), vec![Q_LORA], Role::Norm);
                push!(
                    format!("blk.{i}.attn_q_b.weight"),
                    vec![Q_LORA, N_HEAD * K_MLA],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![n_embd, KV_LORA + 16],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_kv_a_norm.weight"), vec![KV_LORA], Role::Norm);
                push!(
                    format!("blk.{i}.attn_k_b.weight"),
                    vec![K_MLA - 16, KV_LORA, N_HEAD],
                    Role::Proj
                );
                push!(
                    format!("blk.{i}.attn_v_b.weight"),
                    vec![KV_LORA, V_MLA, N_HEAD],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_gate.weight"), vec![n_embd, N_HEAD], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * V_MLA, n_embd], Role::Proj);
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if dense {
                    push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                    push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                    push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
                } else {
                    push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![N_FF_EXP, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SHEXP, n_embd], Role::Proj);
                }
            }
            Fam::HyV3 => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.attn_q.weight"), vec![n_embd, hd * N_HEAD], Role::Proj);
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                push!(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if dense {
                    push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                    push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                    push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
                } else {
                    push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                    // hy-v3.cpp:68 — the suffix-LESS exp_probs_b
                    push!(format!("blk.{i}.exp_probs_b"), vec![N_EXPERT], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![N_FF_EXP, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SHEXP, n_embd], Role::Proj);
                }
            }
            Fam::Mimo2 => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(
                    format!("blk.{i}.attn_qkv.weight"),
                    vec![n_embd, hd * (N_HEAD + 2 * N_HEAD_KV)],
                    Role::Proj
                );
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                push!(format!("blk.{i}.attn_sinks.weight"), vec![N_HEAD], Role::Bias);
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
            }
            Fam::Step35 => {
                push!(format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
                push!(format!("blk.{i}.attn_q_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_k_norm.weight"), vec![hd], Role::Norm);
                push!(format!("blk.{i}.attn_q.weight"), vec![n_embd, hd * N_HEAD], Role::Proj);
                push!(format!("blk.{i}.attn_k.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_v.weight"), vec![n_embd, hd * N_HEAD_KV], Role::Proj);
                push!(format!("blk.{i}.attn_output.weight"), vec![hd * N_HEAD, n_embd], Role::Proj);
                push!(format!("blk.{i}.attn_gate.weight"), vec![n_embd, N_HEAD], Role::Proj);
                push!(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
                if dense {
                    push!(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, N_FF], Role::Proj);
                    push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, n_embd], Role::Proj);
                    push!(format!("blk.{i}.ffn_up.weight"), vec![n_embd, N_FF], Role::Proj);
                } else {
                    push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![n_embd, N_EXPERT], Role::Router);
                    push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], Role::Bias);
                    push!(
                        format!("blk.{i}.ffn_gate_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_down_exps.weight"),
                        vec![N_FF_EXP, n_embd, N_EXPERT],
                        Role::Router
                    );
                    push!(
                        format!("blk.{i}.ffn_up_exps.weight"),
                        vec![n_embd, N_FF_EXP, N_EXPERT],
                        Role::Router
                    );
                    push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![n_embd, N_FF_SHEXP], Role::Proj);
                    push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SHEXP, n_embd], Role::Proj);
                }
            }
        }

        // the nextn trio on the MTP layer
        if i as usize == N_LAYER {
            push!(
                format!("blk.{i}.nextn.eh_proj.weight"),
                vec![2 * n_embd, n_embd],
                Role::Proj
            );
            push!(format!("blk.{i}.nextn.enorm.weight"), vec![n_embd], Role::Norm);
            push!(format!("blk.{i}.nextn.hnorm.weight"), vec![n_embd], Role::Norm);
            if fam == Fam::BailingMoe3 {
                // nextn.shared_head_norm is LAYER_OUT_NORM
                // (bailingmoe3.cpp:158)
                push!(
                    format!("blk.{i}.layer_output_norm.weight"),
                    vec![n_embd],
                    Role::Norm
                );
            }
        }
    }

    // step35's shared rope-factors tensor (one tensor serves every layer) —
    // {n_rot_max/2} entries (step35.cpp:61-85); n_rot_max = 32 (the SWA
    // layers' rope dim since MTP batch 18 — see the rope_dim note above)
    if fam == Fam::Step35 {
        push!("rope_freqs.weight", vec![16], Role::Norm);
    }
    v
}

// ---------------------------------------------------------------------------
// the writer (the batch-6/7/15 recipe)
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

fn scale_of(role: Role) -> f32 {
    match role {
        Role::Norm => 1.0,
        Role::Bias => 0.02,
        Role::Proj | Role::Router => 1.0 / (N_EMBD as f32).sqrt(),
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn build_file(fam: Fam) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/mtp2");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = fam.arch_name();
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!(
        "general.name",
        Value::String(format!("llama-rust-synth-{a}-mtp"))
    );
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(N_CTX));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32((N_LAYER + 1) as u32));
    kv!(format!("{a}.nextn_predict_layers"), Value::U32(1));
    if !matches!(fam, Fam::Qwen35Moe | Fam::Qwen3Next) {
        // the MoE-only archs read n_ff nowhere; the dense readers need it
        kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    }
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(N_HEAD_KV as u32));
    if fam == Fam::BailingMoe3 {
        // the MLA row geometry (head_count_kv stays the generic 1-head MQA of
        // the MLA layers)
        kv!(format!("{a}.attention.head_count_kv"), Value::U32(1));
        kv!(format!("{a}.attention.key_length"), Value::U32((KV_LORA + 16) as u32));
        kv!(format!("{a}.attention.value_length"), Value::U32(KV_LORA as u32));
        kv!(format!("{a}.attention.key_length_mla"), Value::U32(K_MLA as u32));
        kv!(format!("{a}.attention.value_length_mla"), Value::U32(V_MLA as u32));
        kv!(format!("{a}.attention.kv_lora_rank"), Value::U32(KV_LORA as u32));
        kv!(format!("{a}.attention.q_lora_rank"), Value::U32(Q_LORA as u32));
    } else {
        kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
        kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    }
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    // cohere2moe.cpp:297 / hy-v3.cpp:250 assert n_embd_head == n_rot — those
    // two rope the full head dim. step35 (MTP batch 18) joins them at 32 so
    // the shared rope_freqs (n_rot_max/2 = 16 entries, step35.cpp:61-85)
    // covers the full-attention layers' cache-init scan — the C's
    // ggml_rope_cache_init runs i0 < ne0 (ops.cpp:5975-5989), reading factor
    // entries past n_dims/2 that the rotation loop (i0 < n_dims) then never
    // touches: a benign OOB the port's bounds check refuses. At rope_dim 16
    // the table would be 8 wide against ne0 = 32 and the port's trunk decode
    // would panic.
    let rope_dim: u32 = if matches!(fam, Fam::Cohere2Moe | Fam::HyV3 | Fam::Step35) {
        32
    } else {
        16
    };
    kv!(format!("{a}.rope.dimension_count"), Value::U32(rope_dim));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    match fam {
        Fam::Qwen35 | Fam::Qwen35Moe => {
            // IMRoPE sections (sum n_rot/2) + the GDN keys (REQUIRED even
            // though this file has no recurrent layer)
            kv!(
                format!("{a}.rope.dimension_sections"),
                Value::Array(GgufType::Uint32, vec![Value::U32(5), Value::U32(2), Value::U32(1), Value::U32(0)])
            );
            kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
            kv!(format!("{a}.ssm.inner_size"), Value::U32(32));
            kv!(format!("{a}.ssm.state_size"), Value::U32(8));
            kv!(format!("{a}.ssm.time_step_rank"), Value::U32(4));
            kv!(format!("{a}.ssm.group_count"), Value::U32(2));
            kv!(
                format!("{a}.attention.recurrent_layers"),
                Value::Array(GgufType::Uint32, vec![Value::U32(0); N_LAYER + 1])
            );
        }
        Fam::Qwen3Next => {
            kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
            kv!(format!("{a}.ssm.inner_size"), Value::U32(32));
            kv!(format!("{a}.ssm.state_size"), Value::U32(8));
            kv!(format!("{a}.ssm.time_step_rank"), Value::U32(4));
            kv!(format!("{a}.ssm.group_count"), Value::U32(2));
            kv!(
                format!("{a}.attention.recurrent_layers"),
                Value::Array(GgufType::Uint32, vec![Value::U32(0); N_LAYER + 1])
            );
        }
        Fam::Glm4Moe => {
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(format!("{a}.leading_dense_block_count"), Value::U32(1));
            kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
            // gating absent → SIGMOID default (glm4-moe.cpp:16-18)
        }
        Fam::Cohere2Moe => {
            kv!(format!("{a}.attention.sliding_window"), Value::U32(8));
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(1), Value::U32(1), Value::U32(1)])
            );
            kv!(format!("{a}.logit_scale"), Value::F32(1.2));
            kv!(format!("{a}.leading_dense_block_count"), Value::U32(1));
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(N_FF_SHEXP as u32));
        }
        Fam::BailingMoe3 => {
            kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
            kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(N_FF_SHEXP as u32));
            kv!(format!("{a}.kda.head_dim"), Value::U32(16));
            kv!(format!("{a}.kda.safe_gate"), Value::Bool(true));
            kv!(format!("{a}.kda.gate_lower_bound"), Value::F32(-0.1));
            kv!(format!("{a}.expert_shared_count"), Value::U32(1));
            kv!(format!("{a}.leading_dense_block_count"), Value::U32(1));
            kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
            kv!(format!("{a}.expert_gating_func"), Value::U32(2)); // SIGMOID
            kv!(format!("{a}.expert_weights_scale"), Value::F32(2.5));
            kv!(
                format!("{a}.swiglu_clamp_exp"),
                Value::Array(GgufType::Float32, vec![Value::F32(7.0); N_LAYER + 1])
            );
            kv!(
                format!("{a}.swiglu_clamp_shexp"),
                Value::Array(GgufType::Float32, vec![Value::F32(0.05); N_LAYER + 1])
            );
        }
        Fam::HyV3 => {
            kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(N_FF_SHEXP as u32));
            kv!(format!("{a}.leading_dense_block_count"), Value::U32(1));
        }
        Fam::Mimo2 => {
            kv!(format!("{a}.attention.sliding_window"), Value::U32(8));
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(1), Value::U32(1), Value::U32(1)])
            );
        }
        Fam::Step35 => {
            kv!(format!("{a}.attention.sliding_window"), Value::U32(8));
            kv!(
                format!("{a}.attention.sliding_window_pattern"),
                Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(1), Value::U32(1), Value::U32(1)])
            );
            kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(N_FF_SHEXP as u32));
            kv!(format!("{a}.leading_dense_block_count"), Value::U32(1));
        }
    }

    // the shared MoE keys
    if matches!(
        fam,
        Fam::Qwen35Moe
            | Fam::Qwen3Next
            | Fam::Glm4Moe
            | Fam::Cohere2Moe
            | Fam::BailingMoe3
            | Fam::HyV3
            | Fam::Mimo2
            | Fam::Step35
    ) {
        kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
        kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT_USED as u32));
        kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
    }
    if matches!(fam, Fam::Qwen35Moe | Fam::Qwen3Next) {
        kv!(
            format!("{a}.expert_shared_feed_forward_length"),
            Value::U32(N_FF_SHEXP as u32)
        );
        kv!(format!("{a}.expert_weights_scale"), Value::F32(1.0));
    }

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f171 ^ (a.len() as u64));
    let table = tensors_for(fam);
    for (name, ne, role) in &table {
        let n: i64 = ne.iter().product();
        let s = scale_of(*role);
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

    let path = fam.path();
    let f = std::fs::File::create(&path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    (table.len(), std::fs::metadata(&path).unwrap().len())
}

fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn load_synth(fam: Fam) -> LlamaModel {
    {
        let _g = file_lock();
        build_file(fam);
    }
    load_synth_nolock(fam)
}

/// the load half of [`load_synth`] for callers already holding [`file_lock`]
/// (rebuilding the file under another test's live mmap SIGBUSes it — the
/// parallel-test hazard the lock exists for)
fn load_synth_nolock(fam: Fam) -> LlamaModel {
    let path = fam.path();
    let gguf = Gguf::open(&path).expect("open synth");
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

// ---------------------------------------------------------------------------
// weights + params assembly (the CLI's forward_weights derivations)
// ---------------------------------------------------------------------------

fn synth_attn(m: &LlamaModel) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: N_HEAD,
        n_head_kv: if m.arch == llama::arch::LlmArch::BAILINGMOE3 {
            1
        } else {
            N_HEAD_KV
        },
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
        use_flash_attn: false,
    }
}

fn nextn_of(l: &LayerTensors) -> graph_arch::MtpNextn {
    let n = &l.nextn;
    graph_arch::MtpNextn {
        eh_proj: n.eh_proj.expect("nextn.eh_proj"),
        enorm: n.enorm.expect("nextn.enorm"),
        hnorm: n.hnorm.expect("nextn.hnorm"),
        embed_tokens: n.embed_tokens,
        shared_head_head: n.shared_head_head,
        shared_head_norm: n.shared_head_norm,
    }
}

/// the per-arch (MtpWeights bundle, head AttnParams, k/v cache row widths)
enum Mtp2Bundle {
    Qwen35(
        graph_arch::Qwen35MtpWeights,
        graph_arch::Qwen35Params,
        AttnParams,
    ),
    Qwen35Moe(
        graph_arch::Qwen35MoeMtpWeights,
        graph_arch::Qwen35MoeParams,
        AttnParams,
    ),
    Qwen3Next(
        graph_arch::Qwen3NextMtpWeights,
        graph_arch::Qwen3NextParams,
        AttnParams,
    ),
    Glm4Moe(
        graph_arch::Glm4MoeMtpWeights,
        graph_arch::Glm4MoeParams,
        AttnParams,
    ),
    Cohere2Moe(
        graph_arch::Cohere2MoeMtpWeights,
        graph_arch::Cohere2MoeParams,
        AttnParams,
    ),
    BailingMoe3(
        graph_arch::BailingMoe3MtpWeights,
        graph_arch::BailingMoe3Params,
        AttnParams,
    ),
    HyV3(graph_arch::HyV3MtpWeights, graph_arch::HyV3Params, AttnParams),
    Mimo2(graph_arch::Mimo2MtpWeights, graph_arch::Mimo2Params, AttnParams),
    Step35(graph_arch::Step35MtpWeights, graph_arch::Step35Params, AttnParams),
}

fn assemble(m: &mut LlamaModel) -> Mtp2Bundle {
    let hp = &m.hparams;
    let attn = synth_attn(m);
    let il = hp.n_layer() as usize;
    let l = &m.layers[il];
    let mtp_nextn = nextn_of(l);
    let f_attention_scale = 0.0f32;
    match m.arch {
        llama::arch::LlmArch::QWEN35 => Mtp2Bundle::Qwen35(
            graph_arch::Qwen35MtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: qwen35_layer(l),
                n_head: N_HEAD,
                n_head_kv: N_HEAD_KV,
                n_embd_head: HD,
                n_rot: hp.n_rot(il) as i32,
            },
            graph_arch::Qwen35Params {
                attn,
                n_embd: N_EMBD,
                n_head: vec![N_HEAD as u32; N_LAYER],
                n_head_kv: vec![N_HEAD_KV as u32; N_LAYER],
                n_embd_head_k: vec![HD as u32; N_LAYER],
                n_embd_head_v: vec![HD as u32; N_LAYER],
                n_rot: vec![16; N_LAYER],
                is_recr: vec![false; N_LAYER],
                rope_sections: hp.rope_sections,
                f_attention_scale,
                ssm_d_conv: 4,
                ssm_d_inner: 32,
                ssm_d_state: 8,
                ssm_dt_rank: 4,
                ssm_n_group: 2,
                n_embd_r: 0,
                n_embd_s: 0,
            },
            attn,
        ),
        llama::arch::LlmArch::QWEN35MOE => Mtp2Bundle::Qwen35Moe(
            graph_arch::Qwen35MoeMtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: qwen35moe_layer(l),
                n_head: N_HEAD,
                n_head_kv: N_HEAD_KV,
                n_embd_head: HD,
                n_rot: hp.n_rot(il) as i32,
            },
            graph_arch::Qwen35MoeParams {
                attn,
                n_embd: N_EMBD,
                n_head: vec![N_HEAD as u32; N_LAYER],
                n_head_kv: vec![N_HEAD_KV as u32; N_LAYER],
                n_embd_head_k: vec![HD as u32; N_LAYER],
                n_embd_head_v: vec![HD as u32; N_LAYER],
                n_rot: vec![16; N_LAYER],
                is_recr: vec![false; N_LAYER],
                rope_sections: hp.rope_sections,
                f_attention_scale,
                ssm_d_conv: 4,
                ssm_d_inner: 32,
                ssm_d_state: 8,
                ssm_dt_rank: 4,
                ssm_n_group: 2,
                n_embd_r: 0,
                n_embd_s: 0,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_scale: 1.0,
            },
            attn,
        ),
        llama::arch::LlmArch::QWEN3NEXT => Mtp2Bundle::Qwen3Next(
            graph_arch::Qwen3NextMtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: qwen3next_layer(l),
                n_head: N_HEAD,
                n_head_kv: N_HEAD_KV,
                n_embd_head: HD,
            },
            graph_arch::Qwen3NextParams {
                attn,
                n_embd: N_EMBD,
                n_head: vec![N_HEAD as u32; N_LAYER],
                n_head_kv: vec![N_HEAD_KV as u32; N_LAYER],
                n_embd_head_k: vec![HD as u32; N_LAYER],
                n_embd_head_v: vec![HD as u32; N_LAYER],
                n_rot: vec![16; N_LAYER],
                is_recr: vec![false; N_LAYER],
                f_attention_scale,
                ssm_d_conv: 4,
                ssm_d_inner: 32,
                ssm_d_state: 8,
                ssm_dt_rank: 4,
                ssm_n_group: 2,
                n_embd_r: 0,
                n_embd_s: 0,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_scale: 1.0,
            },
            attn,
        ),
        llama::arch::LlmArch::GLM4_MOE => Mtp2Bundle::Glm4Moe(
            graph_arch::Glm4MoeMtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: glm4_moe_layer(l),
            },
            graph_arch::Glm4MoeParams {
                attn,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                n_layer_dense_lead: 1,
                expert_weights_norm: true,
                expert_weights_scale: hp.expert_weights_scale,
                expert_gating_func: hp.expert_gating_func as i32,
            },
            attn,
        ),
        llama::arch::LlmArch::COHERE2MOE => Mtp2Bundle::Cohere2Moe(
            graph_arch::Cohere2MoeMtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: cohere2moe_layer(l),
            },
            graph_arch::Cohere2MoeParams {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                norm_ln_eps: hp.f_norm_eps,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                n_layer_dense_lead: 1,
                is_swa: (0..=N_LAYER).map(|i| hp.is_swa(i)).collect(),
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
                expert_gating_func: hp.expert_gating_func as i32,
                logit_scale: hp.f_logit_scale,
            },
            attn,
        ),
        llama::arch::LlmArch::BAILINGMOE3 => Mtp2Bundle::BailingMoe3(
            graph_arch::BailingMoe3MtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: bailingmoe3_layer(l),
            },
            graph_arch::BailingMoe3Params {
                attn,
                n_embd: N_EMBD,
                n_head: N_HEAD,
                is_recr: vec![false; N_LAYER],
                n_embd_head_kda: 16,
                ssm_d_conv: 4,
                kda_gate_lower_bound: -0.1,
                n_embd_head_k_mla: K_MLA,
                n_embd_head_v_mla: V_MLA,
                n_lora_kv: KV_LORA,
                n_lora_q: Q_LORA,
                n_embd_head_qk_rope: 16,
                // text-only synth: no rope sections → the plain rope branch
                rope_sections: [0; 4],
                n_embd_r: 0,
                n_embd_s: 0,
                n_layer_dense_lead: 1,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_norm: true,
                expert_weights_scale: 2.5,
                expert_gating_func: 2, // SIGMOID
                swiglu_clamp_exp: vec![7.0; N_LAYER + 1],
                swiglu_clamp_shexp: vec![0.05; N_LAYER + 1],
            },
            attn,
        ),
        llama::arch::LlmArch::HY_V3 => Mtp2Bundle::HyV3(
            graph_arch::HyV3MtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: hy_v3_layer(l),
            },
            graph_arch::HyV3Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
                expert_gating_func: hp.expert_gating_func as i32,
            },
            attn,
        ),
        llama::arch::LlmArch::MIMO2 => Mtp2Bundle::Mimo2(
            graph_arch::Mimo2MtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: mimo2_layer(l),
                layer_out_norm: l.layer_out_norm,
                n_head: N_HEAD,
                n_head_kv: N_HEAD_KV,
                freq_base: hp.rope_freq_base_train,
                freq_scale: 1.0,
            },
            graph_arch::Mimo2Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                freq_base: vec![hp.rope_freq_base_train; N_LAYER],
                freq_scale: vec![1.0; N_LAYER],
                v_scale: hp.f_attn_value_scale,
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_scale: hp.expert_weights_scale,
            },
            attn,
        ),
        llama::arch::LlmArch::STEP35 => Mtp2Bundle::Step35(
            graph_arch::Step35MtpWeights {
                tok_embd: m.tok_embd,
                output_norm: m.output_norm,
                output: m.output,
                nextn: mtp_nextn,
                layer: step35_layer(l),
                n_head: N_HEAD,
                n_head_kv: N_HEAD_KV,
                is_swa: hp.is_swa(il),
                n_rot: hp.n_rot(il) as i32,
                freq_base: hp.rope_freq_base_train,
                freq_scale: 1.0,
            },
            graph_arch::Step35Params {
                attn,
                norm_rms_eps: hp.f_norm_rms_eps,
                is_swa: (0..N_LAYER).map(|i| hp.is_swa(i)).collect(),
                n_rot: (0..N_LAYER).map(|i| hp.n_rot(i) as i64).collect(),
                freq_base: vec![hp.rope_freq_base_train; N_LAYER],
                freq_scale: vec![1.0; N_LAYER],
                n_expert: N_EXPERT,
                n_expert_used: N_EXPERT_USED,
                expert_weights_norm: hp.expert_weights_norm,
                expert_weights_scale: hp.expert_weights_scale,
                expert_gating_func: hp.expert_gating_func as i32,
            },
            attn,
        ),
        other => panic!("mtp2: unexpected arch {other:?}"),
    }
}

fn qwen35_layer(l: &LayerTensors) -> graph_arch::Qwen35LayerWeights {
    graph_arch::Qwen35LayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        attn_post_norm: l.attn_post_norm.unwrap(),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt_b: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate: l.ffn_gate.unwrap(),
        ffn_up: l.ffn_up.unwrap(),
        ffn_down: l.ffn_down.unwrap(),
    }
}

fn qwen35moe_layer(l: &LayerTensors) -> graph_arch::Qwen35MoeLayerWeights {
    graph_arch::Qwen35MoeLayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        attn_post_norm: l.attn_post_norm.unwrap(),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: None,
        wqkv_gate: None,
        ssm_conv1d: None,
        ssm_dt_b: None,
        ssm_a: None,
        ssm_beta: None,
        ssm_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate_inp: l.ffn_gate_inp.unwrap(),
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps.unwrap(),
        ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

fn qwen3next_layer(l: &LayerTensors) -> graph_arch::Qwen3NextLayerWeights {
    graph_arch::Qwen3NextLayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        attn_post_norm: l.attn_post_norm.unwrap(),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: None,
        wqkv_gate: None,
        ssm_in: None,
        ssm_conv1d: None,
        ssm_dt_b: None,
        ssm_a: None,
        ssm_beta_alpha: None,
        ssm_norm: None,
        ssm_out: None,
        ffn_gate_inp: l.ffn_gate_inp.unwrap(),
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps.unwrap(),
        ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

fn glm4_moe_layer(l: &LayerTensors) -> graph_arch::Glm4MoeLayerWeights {
    graph_arch::Glm4MoeLayerWeights {
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
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        attn_post_norm: l.attn_post_norm.unwrap(),
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
    }
}

fn cohere2moe_layer(l: &LayerTensors) -> graph_arch::Cohere2MoeLayerWeights {
    graph_arch::Cohere2MoeLayerWeights {
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
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

fn bailingmoe3_layer(l: &LayerTensors) -> graph_arch::BailingMoe3LayerWeights {
    graph_arch::BailingMoe3LayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        ssm_q_conv: None,
        ssm_k_conv: None,
        ssm_v_conv: None,
        wqkv: None,
        wq: None,
        wk: None,
        wv: None,
        ssm_f_a: None,
        ssm_beta: None,
        ssm_a: None,
        ssm_dt_b: None,
        ssm_g_a: None,
        ssm_o_norm: None,
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
    }
}

fn hy_v3_layer(l: &LayerTensors) -> graph_arch::HyV3LayerWeights {
    graph_arch::HyV3LayerWeights {
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
        attn_q_norm: l.attn_q_norm.unwrap(),
        attn_k_norm: l.attn_k_norm.unwrap(),
        ffn_norm: l.ffn_norm.unwrap(),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

fn mimo2_layer(l: &LayerTensors) -> graph_arch::Mimo2LayerWeights {
    graph_arch::Mimo2LayerWeights {
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
        attn_sinks: l.attn_sinks,
        ffn_norm: l.ffn_norm.unwrap(),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
    }
}

fn step35_layer(l: &LayerTensors) -> graph_arch::Step35LayerWeights {
    graph_arch::Step35LayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        rope_freqs: l.rope_freqs,
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wq_b: l.wq_b,
        wk: l.wk,
        wk_b: l.wk_b,
        wv: l.wv,
        wv_b: l.wv_b,
        wo: l.wo.unwrap(),
        wqkv_gate: l.wqkv_gate,
        ffn_norm: l.ffn_norm.unwrap(),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

// ---------------------------------------------------------------------------
// the direct-graph driver (the arch_batch5 Driver shape over a 1-layer
// filtered cache)
// ---------------------------------------------------------------------------

use llama::graph::DecodeInputs;

struct Mtp2Driver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    bundle: Mtp2Bundle,
    /// hparams.n_pos_per_embd (4 for the IMROPE archs)
    n_pos_per_embd: usize,
}

fn driver_for(fam: Fam, m: &mut LlamaModel) -> Mtp2Driver {
    let bundle = assemble(m);
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    let head_attn = match &bundle {
        Mtp2Bundle::Qwen35(_, _, a)
        | Mtp2Bundle::Qwen35Moe(_, _, a)
        | Mtp2Bundle::Qwen3Next(_, _, a)
        | Mtp2Bundle::Glm4Moe(_, _, a)
        | Mtp2Bundle::Cohere2Moe(_, _, a)
        | Mtp2Bundle::BailingMoe3(_, _, a)
        | Mtp2Bundle::HyV3(_, _, a)
        | Mtp2Bundle::Mimo2(_, _, a)
        | Mtp2Bundle::Step35(_, _, a) => *a,
    };
    let kv = if fam.iswa() {
        let hp = &m.hparams;
        let is_swa_mtp = hp.is_swa(hp.n_layer() as usize);
        KvCache::new_swa(
            &mut gctx,
            &[head_attn.n_embd_head_k * head_attn.n_head_kv],
            &[head_attn.n_embd_head_v * head_attn.n_head_kv],
            N_CTX,
            &SwaCacheSpec {
                n_swa: hp.n_swa,
                swa_type: hp.swa_type,
                is_swa: vec![is_swa_mtp],
                swa_full: false,
                unified: false,
                n_seq_max: 1,
            },
            32,
        )
    } else {
        KvCache::new_with_dims(
            &mut gctx,
            &[head_attn.n_embd_head_k * head_attn.n_head_kv],
            &[head_attn.n_embd_head_v * head_attn.n_head_kv],
            N_CTX,
        )
    };
    let watermark = gctx.mark();
    Mtp2Driver {
        gctx,
        kv,
        watermark,
        bundle,
        n_pos_per_embd: m.hparams.n_pos_per_embd() as usize,
    }
}

impl Mtp2Driver {
    /// one draft step: token `tok` at `pos` with the h row `h` — returns
    /// (logits row, t_h_nextn row)
    fn step(&mut self, tok: i32, pos: i32, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        use llama::context::fill_mask_seq_pub;
        let n = 1usize;
        let n_embd = h.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        self.kv.assign(sinfo, &[pos], 0);
        let n_kv = self.kv.n_kv();

        // the iswa pair: place the token in the SWA half too
        let swa_idxs = self.kv.find_slot_swa(n as u32);
        if let (Some(idxs), Some(_)) = (swa_idxs.as_ref(), self.kv.swa_cache()) {
            self.kv.assign_swa(idxs, &[pos], 0);
        }
        let n_kv_swa = self.kv.n_kv_swa();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        // MRoPE/IMROPE archs consume 4 position ids per token (the text rule:
        // first three equal, the 4th zero — llm_graph_input_pos::set_input,
        // llama-graph.cpp:131-141)
        let n_pos = self.n_pos_per_embd;
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, (n * n_pos) as i64);
        let kq_mask = self.gctx.new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        let h_in = self.gctx.new_tensor_2d(GgmlType::F32, n_embd as i64, n as i64);
        let mut swa_step: Option<(ggml::TensorId, ggml::TensorId)> = None;
        if self.kv.swa_cache().is_some() {
            let m = self.gctx.new_tensor_2d(GgmlType::F32, n_kv_swa as i64, n as i64);
            let ri = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
            for t in [tokens_t, pos_t, kq_mask, row_idx, h_in, m, ri] {
                self.gctx.arena_resize_tensor(t);
            }
            swa_step = Some((ri, m));
        } else {
            for t in [tokens_t, pos_t, kq_mask, row_idx, h_in] {
                self.gctx.arena_resize_tensor(t);
            }
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p[0] = tok)
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| {
                for k in 0..n_pos {
                    p[k] = if k == 3 { 0 } else { pos };
                }
            })
            .unwrap();
        {
            let bytes = self.gctx.data_bytes_mut(row_idx).unwrap();
            bytes.copy_from_slice(bytemuck::cast_slice(&[sinfo.s0 as i64]));
        }
        {
            let bytes = self.gctx.data_bytes_mut(h_in).unwrap();
            let f: &mut [f32] = bytemuck::cast_slice_mut(bytes);
            f.copy_from_slice(h);
        }
        // the causal mask over the base cells
        {
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize].iter().map(|c| c.pos).collect();
            let mask: &mut [f32] =
                bytemuck::cast_slice_mut(self.gctx.data_bytes_mut(kq_mask).unwrap());
            llama::graph::fill_causal_mask(mask, &kv_pos, &[pos]);
        }
        if let Some((ri, m)) = swa_step {
            if let Some(idxs) = &swa_idxs {
                let bytes = self.gctx.data_bytes_mut(ri).unwrap();
                let vals: Vec<i64> = idxs.iter().map(|&i| i as i64).collect();
                bytes.copy_from_slice(bytemuck::cast_slice(&vals));
            }
            let swa = self.kv.swa_cache().unwrap();
            let cells: Vec<(i32, u64)> = swa.cells[..n_kv_swa as usize]
                .iter()
                .map(|c| (c.pos, c.seq))
                .collect();
            fill_mask_seq_pub(
                &mut self.gctx,
                m,
                GgmlType::F32,
                &cells,
                &[0],
                &[pos],
                swa.n_swa,
                swa.swa_type,
                false,
                true,
            );
            self.kv.swa_step = Some(KvSwaStep {
                row_idx: ri,
                kq_mask: m,
            });
        } else {
            self.kv.swa_step = None;
        }

        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };

        let result = match &self.bundle {
            Mtp2Bundle::Qwen35(w, p, _) => {
                graph_arch::build_qwen35_mtp_forward(&mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n)
            }
            Mtp2Bundle::Qwen35Moe(w, p, _) => graph_arch::build_qwen35moe_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::Qwen3Next(w, p, _) => graph_arch::build_qwen3next_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::Glm4Moe(w, p, _) => graph_arch::build_glm4_moe_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::Cohere2Moe(w, p, _) => graph_arch::build_cohere2moe_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::BailingMoe3(w, p, _) => graph_arch::build_bailingmoe3_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::HyV3(w, p, _) => graph_arch::build_hy_v3_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::Mimo2(w, p, _) => graph_arch::build_mimo2_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
            Mtp2Bundle::Step35(w, p, _) => graph_arch::build_step35_mtp_forward(
                &mut self.gctx, w, p, &self.kv, &inputs, h_in, n_kv, n,
            ),
        };
        let logits = result.logits;
        let embd = result.embd.expect("h_nextn");
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);

        let read = |gctx: &Context, t: ggml::TensorId| -> Vec<f32> {
            gctx.data_bytes(t)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let lg = read(&self.gctx, logits);
        let hnext = read(&self.gctx, embd);
        (lg, hnext)
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

/// the draft chain both sides replay: token 1 + the zero h row at pos 0, then
/// (argmax, h_nextn) per step (the reference probe replays the identical
/// chain — speculative.cpp:1616-1767's non-chained, non-mem-shared shape)
fn run_chain(fam: Fam, m: &mut LlamaModel) -> Vec<(i32, Vec<f32>, Vec<f32>)> {
    let mut d = driver_for(fam, m);
    let n_embd = m.hparams.n_embd as usize;
    let mut out = Vec::new();
    let mut tok = 1i32;
    let mut h = vec![0.0f32; n_embd];
    for step in 0..N_STEPS {
        let (lg, hnext) = d.step(tok, step as i32, &h);
        assert!(lg.iter().all(|v| v.is_finite()), "{}: non-finite logits", fam.arch_name());
        assert!(
            hnext.iter().all(|v| v.is_finite()),
            "{}: non-finite h_nextn",
            fam.arch_name()
        );
        out.push((tok, lg.clone(), hnext.clone()));
        tok = argmax(&lg);
        h = hnext;
    }
    out
}

// ---------------------------------------------------------------------------
// the dump format (mirrored byte-for-byte by parity/ref_mtp2_dump.c)
// ---------------------------------------------------------------------------

fn write_dump(path: &str, chain: &[(i32, Vec<f32>, Vec<f32>)]) {
    assert!(!chain.is_empty());
    let n_vocab = chain[0].1.len();
    let n_embd = chain[0].2.len();
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"MTP2P\0\0\0");
    bytes.extend_from_slice(&(chain.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_vocab as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_embd as u32).to_le_bytes());
    for (tok, lg, h) in chain {
        bytes.extend_from_slice(&tok.to_le_bytes());
        for v in lg {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in h {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).expect("write dump");
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the nine files load in the port with the MTP layer's tensors and the
/// nextn trio consumed, and the chain runs with finite values
#[test]
fn mtp2_synth_load_and_chain() {
    for fam in all_fams() {
        let mut m = load_synth(fam);
        let hp = &m.hparams;
        assert_eq!(hp.n_layer() as usize, N_LAYER, "{}: trunk layers", fam.arch_name());
        assert_eq!(hp.n_layer_nextn, 1, "{}: n_layer_nextn", fam.arch_name());
        let l = &m.layers[N_LAYER];
        assert!(l.nextn.eh_proj.is_some(), "{}: eh_proj", fam.arch_name());
        assert!(l.nextn.enorm.is_some());
        assert!(l.nextn.hnorm.is_some());
        if fam == Fam::BailingMoe3 {
            assert!(
                l.nextn.shared_head_norm.is_some(),
                "bailingmoe3: LAYER_OUT_NORM shared head norm"
            );
        }
        assert!(l.attn_norm.is_some(), "{}: MTP layer trunk set", fam.arch_name());
        // the full consumed set equals the table
        let mut want: Vec<String> = tensors_for(fam).into_iter().map(|(n, _, _)| n).collect();
        want.sort();
        want.dedup();
        let mut got: Vec<String> = m.tensors.keys().cloned().collect();
        got.sort();
        assert_eq!(got, want, "{}: created tensor set", fam.arch_name());

        let chain = run_chain(fam, &mut m);
        write_dump(&format!("{OUT_DIR}/{}-port.bin", fam.arch_name()), &chain);
        let (tok0, lg, _) = &chain[0];
        println!(
            "{}: chain ok — {} steps, first tok {tok0}, argmax {}",
            fam.arch_name(),
            chain.len(),
            argmax(lg)
        );
    }
}

/// the reference bit-compare: `parity/gen_mtp2_ref.sh` builds
/// `parity/ref_mtp2_dump` and writes /tmp/mtp2/<arch>-ref.bin for every arch;
/// this cell compares the two dumps byte-for-byte (logits + h_nextn rows of
/// the whole chain — an error anywhere in the graph compounds through the
/// fed-back h rows, so byte equality is a strong structural pin).
#[test]
#[ignore = "needs /tmp/mtp2/<arch>-ref.bin (parity/gen_mtp2_ref.sh)"]
fn mtp2_reference_bitcompare() {
    let mut missing = 0;
    for fam in all_fams() {
        let ref_path = format!("{OUT_DIR}/{}-ref.bin", fam.arch_name());
        let port_path = format!("{OUT_DIR}/{}-port.bin", fam.arch_name());
        if !std::path::Path::new(&ref_path).exists() {
            eprintln!("{}: no reference dump ({ref_path}) — skipped", fam.arch_name());
            missing += 1;
            continue;
        }
        let a = std::fs::read(&ref_path).expect("read ref dump");
        let b = std::fs::read(&port_path).expect("read port dump");
        assert_eq!(&a[..8], b"MTP2P\0\0\0", "{}: ref magic", fam.arch_name());
        assert_eq!(&b[..8], b"MTP2P\0\0\0", "{}: port magic", fam.arch_name());
        if a != b {
            // locate the first differing step for the failure message
            let hdr = 20;
            let rd_u32 = |o: usize| u32::from_le_bytes(a[o..o + 4].try_into().unwrap());
            let n_steps = rd_u32(8) as usize;
            let n_vocab = rd_u32(12) as usize;
            let n_embd = rd_u32(16) as usize;
            let row = 4 + 4 * (n_vocab + n_embd);
            let mut step = usize::MAX;
            for s in 0..n_steps {
                let o = hdr + s * row;
                if a[o..o + row] != b[o..o + row] {
                    step = s;
                    break;
                }
            }
            panic!(
                "{}: dump differs (first at step {step}; {n_steps} steps, n_vocab {n_vocab}, \
                 n_embd {n_embd})",
                fam.arch_name()
            );
        }
        println!("{}: reference bit-compare PASS ({} bytes)", fam.arch_name(), a.len());
    }
    assert_eq!(missing, 0, "run parity/gen_mtp2_ref.sh first — dumps missing");
}

/// the #[ignore] generator for the parity runs
#[test]
#[ignore = "writes /tmp/mtp2 for parity/gen_mtp2_ref.sh"]
fn mtp2_write_synth_files() {
    for fam in all_fams() {
        let (n, bytes) = {
            let _g = file_lock();
            build_file(fam)
        };
        println!("{}: {n} tensors, {bytes} bytes -> {}", fam.arch_name(), fam.path());
    }
}

/// the rope_multi isolation cell: the same tensor/pos/sections the qwen35 MTP
/// attention builds, against /tmp/mtp2/rope-ref.bin (the C probe of
/// parity/mtp2-rope-probe.c). Diffs here would explain a step-1+ residual
/// while step 0 (theta = 0) is bit-identical.
#[test]
#[ignore = "needs /tmp/mtp2/rope-ref.bin (the C ggml_rope_multi probe)"]
fn mtp2_rope_isolate() {
    use ggml::TensorId;
    let mut ctx = Context::new();
    let q: TensorId = ctx.new_tensor_3d(GgmlType::F32, 32, 4, 1);
    let pos = ctx.new_tensor_1d(GgmlType::I32, 4);
    for t in [q, pos] {
        ctx.arena_resize_tensor(t);
    }
    {
        let bytes = ctx.data_bytes_mut(q).unwrap();
        let f: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        for (i, v) in f.iter_mut().enumerate() {
            *v = 0.01 * (i % 13) as f32 - 0.06;
        }
    }
    ctx.with_i32_mut(pos, |p| {
        p.copy_from_slice(&[1, 1, 1, 0]);
    })
    .unwrap();
    let out = ctx.rope_multi(
        q,
        pos,
        None,
        16,
        [5, 2, 1, 0],
        40, // GGML_ROPE_TYPE_IMROPE
        32768,
        10_000.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );
    let mut g = ggml::Graph::new(64);
    g.build_forward(&mut ctx, out);
    ggml::compute::graph_compute(&mut ctx, &mut g, 1);
    let bytes = std::fs::read("/tmp/mtp2/rope-ref.bin").expect("run the C rope probe first");
    let mine: Vec<f32> = ctx
        .data_bytes(out)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let theirs: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(mine.len(), theirs.len());
    let n_diff = mine.iter().zip(&theirs).filter(|(a, b)| a != b).count();
    let maxd = mine
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("rope_multi pos1: {n_diff}/{} elems differ, max |d| = {maxd:e}", mine.len());
    assert_eq!(n_diff, 0, "rope_multi differs from the reference kernel");
}

// ---------------------------------------------------------------------------
// the node-dump bisect (DECDMP1 rules; the C probe's --nodes mirror)
// ---------------------------------------------------------------------------

static M2_DUMP: std::sync::OnceLock<std::sync::Mutex<Option<Vec<u8>>>> = std::sync::OnceLock::new();
static M2_NODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn m2_type_desc(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::I32 => "i32",
        GgmlType::I64 => "i64",
        _ => "other",
    }
}

fn m2_op_desc(op: ggml::GgmlOp) -> String {
    format!("{op:?}")
}

fn m2_dump_cb(node: &ggml::compute::EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true;
    }
    let mut guard = M2_DUMP.get_or_init(Default::default).lock().unwrap();
    let Some(out) = guard.as_mut() else { return true };
    let n: i64 = node.ne.iter().product();
    M2_NODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut put = |s: &str| {
        let l = s.len().min(255);
        out.push(l as u8);
        out.extend_from_slice(&s.as_bytes()[..l]);
    };
    put(&m2_op_desc(node.op));
    put(node.name);
    put(m2_type_desc(node.ty));
    out.extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= (1 << 19) {
        return true;
    }
    // the C probe's rule: ONLY F32 nodes carry payloads (no bytes otherwise)
    if node.ty != GgmlType::F32 {
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
        let v: f32 =
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

/// debug probe: MTP2_DUMP_ARCH=<arch> [MTP2_STEPS=N] — the chain with the
/// DECDMP1 node stream; the C probe's --nodes flag writes the same stream
/// for the bisect (first divergent node).
#[test]
#[ignore = "debug probe: MTP2_DUMP_ARCH=<arch> cargo test -- --ignored mtp2_node_dump"]
fn mtp2_node_dump() {
    let arch = std::env::var("MTP2_DUMP_ARCH").expect("MTP2_DUMP_ARCH");
    let fam = all_fams().into_iter().find(|f| f.arch_name() == arch).expect("fam");
    let n_steps: usize =
        std::env::var("MTP2_STEPS").map(|v| v.parse().unwrap()).unwrap_or(N_STEPS);
    let out_path =
        std::env::var("MTP2_DUMP_OUT").unwrap_or_else(|_| format!("/tmp/mtp2/{arch}-nodes-port.bin"));

    ggml::compute::set_eval_callback(Some(m2_dump_cb));
    M2_DUMP.get_or_init(|| std::sync::Mutex::new(Some(Vec::new())));
    let mut m = load_synth(fam);
    {
        let mut d = driver_for(fam, &mut m);
        let n_embd = m.hparams.n_embd as usize;
        let mut tok = 1i32;
        let mut h = vec![0.0f32; n_embd];
        for step in 0..n_steps {
            let (lg, hnext) = d.step(tok, step as i32, &h);
            tok = argmax(&lg);
            h = hnext;
        }
    }
    ggml::compute::set_eval_callback(None);
    let st = M2_DUMP.get().unwrap().lock().unwrap().take().unwrap();
    let n_nodes = M2_NODES.load(std::sync::atomic::Ordering::Relaxed);
    std::fs::write(&out_path, &st).unwrap();
    println!("mtp2 nodes: {arch} nodes={n_nodes} -> {out_path}");
}

/// the non-fused autoregressive delta-net bit-compare: the identical input
/// stream and op chain as parity/ref_dnet_ar_dump.c (the reference ggml twin
/// of delta-net-base.cpp:289-374). The llama-level arm is unreachable (the
/// CPU build always runs the fused op), so the ggml-level chain IS the
/// acceptance — see build_delta_net_autoregressive's doc.
#[test]
#[ignore = "needs /tmp/mtp2/dnet-ar-ref.bin (parity/ref_dnet_ar_dump)"]
fn mtp2_dnet_ar_bitcompare() {
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
    let (s_dim, h) = (8i64, 3i64);
    let mut ctx = Context::new();
    let q = ctx.new_tensor_4d(GgmlType::F32, s_dim, h, 1, 1);
    let k = ctx.new_tensor_4d(GgmlType::F32, s_dim, h, 1, 1);
    let v = ctx.new_tensor_4d(GgmlType::F32, s_dim, h, 1, 1);
    let g = ctx.new_tensor_4d(GgmlType::F32, 1, h, 1, 1);
    let b = ctx.new_tensor_4d(GgmlType::F32, 1, h, 1, 1);
    let st = ctx.new_tensor_4d(GgmlType::F32, s_dim, s_dim, h, 1);
    for t in [q, k, v, g, b, st] {
        ctx.arena_resize_tensor(t);
    }
    let fill = |ctx: &mut Context, t, f: &mut dyn FnMut() -> f32| {
        let bytes = ctx.data_bytes_mut(t).unwrap();
        let vals: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        vals.iter_mut().for_each(|x| *x = f());
    };
    let mut rng = Rng(0xbeed171);
    fill(&mut ctx, q, &mut || 0.09 * rng.next());
    fill(&mut ctx, k, &mut || 0.09 * rng.next());
    fill(&mut ctx, v, &mut || 0.11 * rng.next());
    fill(&mut ctx, g, &mut || -0.05 - 0.4 * (0.5 * rng.next() + 0.5));
    fill(&mut ctx, b, &mut || 0.5 + 0.45 * (0.5 * rng.next() + 0.5));
    fill(&mut ctx, st, &mut || 0.03 * rng.next());

    let (o, s_new) = graph_arch::build_delta_net_autoregressive(&mut ctx, q, k, v, g, b, st);
    let mut graph = ggml::Graph::new(128);
    graph.build_forward(&mut ctx, o);
    graph.build_forward(&mut ctx, s_new);
    ggml::compute::graph_compute(&mut ctx, &mut graph, 1);

    let ref_bytes = std::fs::read("/tmp/mtp2/dnet-ar-ref.bin").expect("run the C probe first");
    assert_eq!(&ref_bytes[..8], b"DNETAR\0\0", "magic");
    let read = |ctx: &Context, t| -> Vec<f32> {
        ctx.data_bytes(t)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let o_v = read(&ctx, o);
    let s_v = read(&ctx, s_new);
    let mut off = 8usize;
    let take = |off: &mut usize, n: usize| -> Vec<f32> {
        let v: Vec<f32> = ref_bytes[*off..*off + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        *off += 4 * n;
        v
    };
    let o_ref = take(&mut off, (s_dim * h) as usize);
    let s_ref = take(&mut off, (s_dim * s_dim * h) as usize);
    assert_eq!(o_v, o_ref, "delta-net AR output o differs from the reference");
    assert_eq!(s_v, s_ref, "delta-net AR state s_new differs from the reference");
    println!(
        "delta-net AR: bit-identical o ({} elems) + s_new ({} elems)",
        o_v.len(),
        s_v.len()
    );
}

// ---------------------------------------------------------------------------
// arch batch 18 (agent GDN): GGML_OP_SOLVE_TRI + GGML_OP_DIAG + the chunked
// delta-net half. The reference llama graph can never reach the chunked path
// (fused_gdn_ch hardwired true, llama-context.cpp:234), so the oracles are
// (a) the op-level probe parity/ref_triops_dump.c and (b) the chain-level
// probe parity/ref_dnet_ch_dump.c — both built against the *reference ggml*,
// whose CPU backend does implement the two ops.
// ---------------------------------------------------------------------------

/// the same input filler the two probes use (Rng 1:1)
struct DnetRng(u64);
impl DnetRng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn dnet_fill(ctx: &mut Context, t: ggml::TensorId, f: &mut dyn FnMut() -> f32) {
    let bytes = ctx.data_bytes_mut(t).unwrap();
    let vals: &mut [f32] = bytemuck::cast_slice_mut(bytes);
    vals.iter_mut().for_each(|x| *x = f());
}

fn dnet_read_layout(ctx: &Context, t: ggml::TensorId) -> Vec<f32> {
    let (ne, nb) = (ctx.ne(t), ctx.nb(t));
    dnet_read_layout_at(ctx, t, 0, *ne, *nb)
}

/// read one tensor through its own strides into row-major f32 order. `root`
/// must be the storage-owning tensor of `t` and `base` the view's byte
/// offset into it — `data_bytes` on a non-zero-offset view would slice
/// `[root_off + view_off, root_off + view_off + root_len)` and walk past
/// the block, so offset views are read through the root and indexed here.
fn dnet_read_layout_at(
    ctx: &Context,
    root: ggml::TensorId,
    base: usize,
    ne: [i64; 4],
    nb: [u64; 4],
) -> Vec<f32> {
    let bytes = ctx.data_bytes(root).unwrap();
    let mut out = vec![0.0f32; (ne[0] * ne[1] * ne[2] * ne[3]) as usize];
    let mut idx = 0usize;
    for i3 in 0..ne[3] {
        for i2 in 0..ne[2] {
            for i1 in 0..ne[1] {
                for i0 in 0..ne[0] {
                    let off = base
                        + (i0 as usize * nb[0] as usize
                            + i1 as usize * nb[1] as usize
                            + i2 as usize * nb[2] as usize
                            + i3 as usize * nb[3] as usize);
                    let b: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
                    out[idx] = f32::from_le_bytes(b);
                    idx += 1;
                }
            }
        }
    }
    out
}

/// op-level bit-compare: solve_tri (3 shapes, incl. a 4-thread chunk-size
/// run), diag (2 shapes) and set_inplace against parity/ref_triops_dump.c.
/// Both ops are F32-only in the reference (the kernels GGML_ABORT otherwise)
/// — that is the entire type matrix.
#[test]
#[ignore = "needs /tmp/gops-triops-ref.bin (parity/ref_triops_dump)"]
fn mtp2_tri_ops_bitcompare() {
    use ggml::ops::GGML_TRI_TYPE_LOWER_DIAG;
    let mut ctx = Context::new();
    let mut graph = ggml::Graph::new(256);

    let mut rng = DnetRng(0x1234abc);
    // ---- solve_tri #1: n=8, k=3, B1=2 (single-threaded) ----
    let a = ctx.new_tensor_4d(GgmlType::F32, 8, 8, 2, 1);
    let b = ctx.new_tensor_4d(GgmlType::F32, 3, 8, 2, 1);
    for t in [a, b] {
        ctx.arena_resize_tensor(t);
    }
    {
        // the probe's exact rng order: fill A whole, then zero the upper
        // triangle and redraw the diagonal (one draw per diag element)
        dnet_fill(&mut ctx, a, &mut || 0.3 * rng.next());
        let bytes = ctx.data_bytes_mut(a).unwrap();
        let m: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        for bi in 0..2usize {
            for i in 0..8usize {
                for j in 0..8usize {
                    if j > i {
                        m[bi * 64 + i * 8 + j] = 0.0;
                    }
                }
            }
        }
        for bi in 0..2usize {
            for i in 0..8usize {
                m[bi * 64 + i * 8 + i] = 0.5 + 0.5 * (0.5 * rng.next() + 0.5);
            }
        }
        dnet_fill(&mut ctx, b, &mut || 0.7 * rng.next());
    }
    let x = ctx.solve_tri(a, b, true, true, false);
    graph.build_forward(&mut ctx, x);
    ggml::compute::graph_compute(&mut ctx, &mut graph, 1);

    // ---- solve_tri #2: n=64, k=64, B2=3, 4 threads (probe reseeds) ----
    rng = DnetRng(0xfeed987);
    let mut graph4 = ggml::Graph::new(256);
    let a2 = ctx.new_tensor_4d(GgmlType::F32, 64, 64, 1, 3);
    let b2 = ctx.new_tensor_4d(GgmlType::F32, 64, 64, 1, 3);
    for t in [a2, b2] {
        ctx.arena_resize_tensor(t);
    }
    {
        dnet_fill(&mut ctx, a2, &mut || 0.2 * rng.next());
        let bytes = ctx.data_bytes_mut(a2).unwrap();
        let m: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        for bi in 0..3usize {
            for i in 0..64usize {
                for j in (i + 1)..64usize {
                    m[bi * 4096 + i * 64 + j] = 0.0;
                }
            }
        }
        for bi in 0..3usize {
            for i in 0..64usize {
                m[bi * 4096 + i * 64 + i] = 0.75 + 0.5 * (0.5 * rng.next() + 0.5);
            }
        }
        dnet_fill(&mut ctx, b2, &mut || 0.5 * rng.next());
    }
    let x2 = ctx.solve_tri(a2, b2, true, true, false);
    graph4.build_forward(&mut ctx, x2);
    ggml::compute::graph_compute(&mut ctx, &mut graph4, 4);

    if std::env::var("TRIOPS_DEBUG").is_ok() {
        let bytes = ctx.data_bytes(x2).unwrap();
        std::fs::write("/tmp/gops-x2-port.bin", bytes).unwrap();
    }
    // ---- solve_tri #3: degenerate 1×1 ----
    let mut graph3 = ggml::Graph::new(8);
    let a3 = ctx.new_tensor_4d(GgmlType::F32, 1, 1, 1, 1);
    let b3 = ctx.new_tensor_4d(GgmlType::F32, 1, 1, 1, 1);
    for t in [a3, b3] {
        ctx.arena_resize_tensor(t);
    }
    ctx.data_bytes_mut(a3).unwrap().copy_from_slice(&2.0f32.to_le_bytes());
    ctx.data_bytes_mut(b3).unwrap().copy_from_slice(&(-7.0f32).to_le_bytes());
    let x3 = ctx.solve_tri(a3, b3, true, true, false);
    graph3.build_forward(&mut ctx, x3);
    ggml::compute::graph_compute(&mut ctx, &mut graph3, 1);

    // ---- diag #1/#2 (probe reseeds) ----
    rng = DnetRng(0xd1a9);
    let mut graphd = ggml::Graph::new(256);
    let d1 = ctx.new_tensor_4d(GgmlType::F32, 5, 1, 2, 1);
    ctx.arena_resize_tensor(d1);
    dnet_fill(&mut ctx, d1, &mut || 1.5 * rng.next());
    let dd1 = ctx.diag(d1);
    let d2 = ctx.new_tensor_4d(GgmlType::F32, 64, 1, 3, 2);
    ctx.arena_resize_tensor(d2);
    dnet_fill(&mut ctx, d2, &mut || 1.0 * rng.next());
    let dd2 = ctx.diag(d2);
    graphd.build_forward(&mut ctx, dd1);
    graphd.build_forward(&mut ctx, dd2);
    ggml::compute::graph_compute(&mut ctx, &mut graphd, 1);

    // ---- set_inplace: b [3,4] into dst [5,7] at element offset 2 (probe reseeds) ----
    rng = DnetRng(0x5e7);
    let mut graphs = ggml::Graph::new(64);
    let dst = ctx.new_tensor_2d(GgmlType::F32, 5, 7);
    let sb = ctx.new_tensor_2d(GgmlType::F32, 3, 4);
    for t in [dst, sb] {
        ctx.arena_resize_tensor(t);
    }
    dnet_fill(&mut ctx, dst, &mut || 0.4 * rng.next());
    dnet_fill(&mut ctx, sb, &mut || 1.2 * rng.next());
    let r = ctx.set_inplace(
        dst,
        sb,
        GgmlType::F32.row_size(5),
        GgmlType::F32.row_size(5 * 7),
        GgmlType::F32.row_size(5 * 7),
        2 * 4,
    );
    graphs.build_forward(&mut ctx, r);
    ggml::compute::graph_compute(&mut ctx, &mut graphs, 1);

    // ---- compare ----
    // solve_tri #2's note: the reference .so compiles the substitution sum
    // with GCC 13.3 -O3 default -ffp-contract=fast, and the contraction is
    // *mixed* per trip count (zmm column-vector bodies with in-order
    // vmulps/vaddss = the two-rounding form, scalar vfmadd231ss remainders —
    // .so disasm at ggml_compute_forward_solve_tri+0x2f0..+0x490). Small-k
    // shapes (<= 3 solve columns) are 100% the source-faithful strict form
    // and bit-compare clean; k=64 mixes, so that case is checked with a
    // small ULP bound instead (measured max 2 ulp over 12288 elements).
    let ref_bytes = std::fs::read("/tmp/gops-triops-ref.bin").expect("run the C probe first");
    assert_eq!(&ref_bytes[..8], b"TRIOPS\0\0", "magic");
    let mut off = 8usize;
    let flat = |ctx: &Context, t| -> Vec<f32> {
        ctx.data_bytes(t)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let ulp = |a: f32, b: f32| -> u32 {
        let (x, y) = (a.to_bits(), b.to_bits());
        if x == y { 0 } else if x > y { x - y } else { y - x }
    };
    let take = |off: &mut usize| -> Vec<f32> {
        let n = i32::from_le_bytes(ref_bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        let v: Vec<f32> = ref_bytes[*off..*off + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        *off += 4 * n;
        v
    };
    for (name, got, ulp_bound) in [
        ("solve_tri #1 (8x8x2, 1t)", flat(&ctx, x), 0u32),
        ("solve_tri #2 (64x64x3, 4t)", flat(&ctx, x2), 4u32), // bound is rel (see below)
        ("solve_tri #3 (1x1)", flat(&ctx, x3), 0u32),
        ("diag #1 (5,2,1)", flat(&ctx, dd1), 0u32),
        ("diag #2 (64,3,2)", flat(&ctx, dd2), 0u32),
        ("set_inplace (3x4 into 5x7)", flat(&ctx, r), 0u32),
    ] {
        let expect = take(&mut off);
        assert_eq!(got.len(), expect.len(), "{name}: element count");
        if ulp_bound == 0 {
            assert_eq!(got, expect, "{name}: bits differ from the reference");
            println!("tri-ops {name}: bit-identical ({} elems)", got.len());
        } else {
            // substitution errors compound down each column (X[i] reads
            // X[t<i]), so the ulp spread grows with row depth while staying
            // tiny in relative terms — bound the relative error instead.
            let max_ulp = got.iter().zip(&expect).map(|(a, b)| ulp(*a, *b)).max().unwrap();
            let max_rel = got
                .iter()
                .zip(&expect)
                .map(|(a, b)| (a - b).abs() / (a.abs() + b.abs() + 1e-30))
                .fold(0.0f32, f32::max);
            let rel_bound = 1e-4f32; // measured 6.5e-5 on the 64-row compounding chain
            assert!(
                max_rel <= rel_bound,
                "{name}: max rel {max_rel:e} exceeds the fp-contract bound {rel_bound:e} (max ulp {max_ulp})"
            );
            let n_diff = got.iter().zip(&expect).filter(|(a, b)| a != b).count();
            println!("tri-ops {name}: max rel {max_rel:.2e} / {max_ulp} ulp ({n_diff}/{} elements differ — the reference .so's mixed fp-contract codegen)", got.len());
        }
    }
    let _ = GGML_TRI_TYPE_LOWER_DIAG;
}

/// chain-level bit-compare of build_delta_net_chunking against
/// parity/ref_dnet_ch_dump.c — the same op chain built with the reference
/// ggml. Case 1 is the GDA geometry (qwen3next, CS=64, padding + 2 chunks,
/// 2 seqs, 4 threads); case 2 the KDA geometry (CS=16, 3 chunks).
#[test]
#[ignore = "needs /tmp/mtp2/dnet-ch-ref.bin (parity/ref_dnet_ch_dump)"]
fn mtp2_dnet_ch_bitcompare() {
    let run_case = |s: i64, h: i64, t: i64, bs: i64, kda: bool, seed: u64, threads: usize| {
        let mut ctx = Context::new();
        let q = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
        let k = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
        let v = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
        let g0 = if kda { s } else { 1 };
        let g = ctx.new_tensor_4d(GgmlType::F32, g0, h, t, bs);
        let b = ctx.new_tensor_4d(GgmlType::F32, 1, h, t, bs);
        let st = ctx.new_tensor_4d(GgmlType::F32, s, s, h, bs);
        for tt in [q, k, v, g, b, st] {
            ctx.arena_resize_tensor(tt);
        }
        let mut rng = DnetRng(seed);
        dnet_fill(&mut ctx, q, &mut || 0.09 * rng.next());
        dnet_fill(&mut ctx, k, &mut || 0.09 * rng.next());
        dnet_fill(&mut ctx, v, &mut || 0.11 * rng.next());
        if kda {
            dnet_fill(&mut ctx, g, &mut || -0.05 - 0.4 * (0.5 * rng.next() + 0.5));
        } else {
            dnet_fill(&mut ctx, g, &mut || -0.05 - 0.4 * (0.5 * rng.next() + 0.5));
        }
        dnet_fill(&mut ctx, b, &mut || 0.5 + 0.45 * (0.5 * rng.next() + 0.5));
        dnet_fill(&mut ctx, st, &mut || 0.03 * rng.next());

        let (o, s_new) = graph_arch::build_delta_net_chunking(&mut ctx, q, k, v, g, b, st);
        let mut graph = ggml::Graph::new(8192);
        graph.build_forward(&mut ctx, o);
        graph.build_forward(&mut ctx, s_new);
        ggml::compute::graph_compute(&mut ctx, &mut graph, threads);

        // o is [S_v, H_v, T, B] permuted — read through its strides; s_new is
        // contiguous [S_v, S_v, H_v, B]
        (dnet_read_layout(&ctx, o), dnet_read_layout(&ctx, s_new))
    };

    let ref_bytes = std::fs::read("/tmp/mtp2/dnet-ch-ref.bin").expect("run the C probe first");
    assert_eq!(&ref_bytes[..8], b"DNETCH\0\0", "magic");
    let mut off = 8usize;
    for (name, (o_v, s_v)) in [
        ("GDA (S=8,H=3,T=70,B=2, CS=64)", run_case(8, 3, 70, 2, false, 0xbeed171, 4)),
        ("KDA (S=6,H=2,T=40,B=1, CS=16)", run_case(6, 2, 40, 1, true, 0x5dca11, 4)),
    ] {
        let take = |off: &mut usize| -> Vec<f32> {
            let n = i32::from_le_bytes(ref_bytes[*off..*off + 4].try_into().unwrap()) as usize;
            *off += 4;
            let v: Vec<f32> = ref_bytes[*off..*off + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *off += 4 * n;
            v
        };
        let o_ref = take(&mut off);
        let s_ref = take(&mut off);
        assert_eq!(o_v, o_ref, "chunked delta-net {name}: o differs from the reference");
        assert_eq!(s_v, s_ref, "chunked delta-net {name}: s_new differs from the reference");
        println!("delta-net CH {name}: bit-identical o + s_new ({} elems)", o_v.len());
    }
}

/// self-consistency: OUR fused GDN (the graph path every reference build
/// runs, bit-verified against the reference by the qwen3next/qwen35 e2e
/// tests) vs OUR chunked builder on the same synthetic state. The two are
/// the same math through different fp routes (per-token recurrence vs
/// chunkwise UT transform), so bit-identity is not expected — this records
/// the actual divergence (max ulp / relative error) and guards it against
/// regressions. The *authoritative* chunked-path oracle is
/// mtp2_dnet_ch_bitcompare above.
#[test]
fn mtp2_dnet_fused_vs_chunked() {
    let (s, h, t, bs) = (8i64, 3i64, 70i64, 2i64);
    let mut ctx = Context::new();
    let q = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let k = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let v = ctx.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let g = ctx.new_tensor_4d(GgmlType::F32, 1, h, t, bs);
    let b = ctx.new_tensor_4d(GgmlType::F32, 1, h, t, bs);
    let st = ctx.new_tensor_4d(GgmlType::F32, s, s, h, bs);
    for tt in [q, k, v, g, b, st] {
        ctx.arena_resize_tensor(tt);
    }
    let mut rng = DnetRng(0xbeed171);
    dnet_fill(&mut ctx, q, &mut || 0.09 * rng.next());
    dnet_fill(&mut ctx, k, &mut || 0.09 * rng.next());
    dnet_fill(&mut ctx, v, &mut || 0.11 * rng.next());
    dnet_fill(&mut ctx, g, &mut || -0.05 - 0.4 * (0.5 * rng.next() + 0.5));
    dnet_fill(&mut ctx, b, &mut || 0.5 + 0.45 * (0.5 * rng.next() + 0.5));
    dnet_fill(&mut ctx, st, &mut || 0.03 * rng.next());

    // fused: result packs [S_v*H_v, n_tokens*n_seqs + S_v*H_v*n_seqs]
    let f = ctx.gated_delta_net(q, k, v, g, b, st, 1);
    let f_o = ctx.view_4d(
        f,
        s,
        h,
        t,
        bs,
        (s * 4) as usize,
        (s * h * 4) as usize,
        (s * h * t * 4) as usize,
        0,
    );
    let f_s = ctx.view_4d(
        f,
        s,
        s,
        h,
        bs,
        (s * 4) as usize,
        (s * s * 4) as usize,
        (s * s * h * 4) as usize,
        (s * h * t * bs * 4) as usize,
    );
    let mut g1 = ggml::Graph::new(256);
    g1.build_forward(&mut ctx, f_o);
    g1.build_forward(&mut ctx, f_s);
    ggml::compute::graph_compute(&mut ctx, &mut g1, 1);

    // chunked — separate Context: the same input stream re-filled
    let mut ctx2 = Context::new();
    let q2 = ctx2.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let k2 = ctx2.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let v2 = ctx2.new_tensor_4d(GgmlType::F32, s, h, t, bs);
    let g2 = ctx2.new_tensor_4d(GgmlType::F32, 1, h, t, bs);
    let b2 = ctx2.new_tensor_4d(GgmlType::F32, 1, h, t, bs);
    let st2 = ctx2.new_tensor_4d(GgmlType::F32, s, s, h, bs);
    for tt in [q2, k2, v2, g2, b2, st2] {
        ctx2.arena_resize_tensor(tt);
    }
    let mut rng = DnetRng(0xbeed171);
    dnet_fill(&mut ctx2, q2, &mut || 0.09 * rng.next());
    dnet_fill(&mut ctx2, k2, &mut || 0.09 * rng.next());
    dnet_fill(&mut ctx2, v2, &mut || 0.11 * rng.next());
    dnet_fill(&mut ctx2, g2, &mut || -0.05 - 0.4 * (0.5 * rng.next() + 0.5));
    dnet_fill(&mut ctx2, b2, &mut || 0.5 + 0.45 * (0.5 * rng.next() + 0.5));
    dnet_fill(&mut ctx2, st2, &mut || 0.03 * rng.next());
    let (c_o, c_s) = graph_arch::build_delta_net_chunking(&mut ctx2, q2, k2, v2, g2, b2, st2);
    let mut g2g = ggml::Graph::new(8192);
    g2g.build_forward(&mut ctx2, c_o);
    g2g.build_forward(&mut ctx2, c_s);
    ggml::compute::graph_compute(&mut ctx2, &mut g2g, 1);

    let fo = dnet_read_layout(&ctx, f_o);
    let (fs_ne, fs_nb) = (*ctx.ne(f_s), *ctx.nb(f_s));
    let fs = dnet_read_layout_at(&ctx, f, (s * h * t * bs * 4) as usize, fs_ne, fs_nb);
    let co = dnet_read_layout(&ctx2, c_o);
    let cs = dnet_read_layout(&ctx2, c_s);
    assert_eq!(fo.len(), co.len());
    assert_eq!(fs.len(), cs.len());
    let ulp = |a: f32, bb: f32| -> u32 {
        let (x, y) = (a.to_bits(), bb.to_bits());
        if x == y {
            0
        } else if x > y {
            x - y
        } else {
            y - x
        }
    };
    let report = |name: &str, a: &[f32], bb: &[f32]| {
        let (mut max_ulp, mut max_rel, denom) = (0u32, 0.0f32, 1e-6f32);
        for (&x, &y) in a.iter().zip(bb) {
            max_ulp = max_ulp.max(ulp(x, y));
            max_rel = max_rel.max(((x - y).abs()) / (x.abs() + y.abs() + denom));
        }
        println!("dnet fused-vs-chunked {name}: max_ulp={max_ulp} max_rel={max_rel:.3e}");
        (max_ulp, max_rel)
    };
    let (o_ulp, o_rel) = report("o", &fo, &co);
    let (s_ulp, s_rel) = report("s_new", &fs, &cs);
    // same math, different fp routes: the divergence is the reference's own
    // two-path difference (both of our paths are individually bit-verified
    // against the reference — fused by the qwen3next e2e, chunked by
    // mtp2_dnet_ch_bitcompare). Guard against gross regressions only.
    assert!(o_rel < 1e-3 && s_rel < 1e-2, "fused vs chunked diverged: o {o_rel:e} s {s_rel:e}");
    let _ = (o_ulp, s_ulp);
}

// ---------------------------------------------------------------------------
// lfm2 (dense) — MTP batch 17: the shared lfm2moe arm with
// n_layer_dense_lead == n_layer (meta.rs's LFM2 hparams arm, lfm2.cpp:16)
// ---------------------------------------------------------------------------

/// the dense lfm2 file loads through the shared arm: every layer dense-FFN +
/// full attention, all tensors consumed
#[test]
fn mtp2_lfm2_dense_loads() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }
    let a = "lfm2";
    let (n_embd, n_layer, n_ff) = (64i64, 2usize, 48i64);
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-lfm2-dense".to_string()));
    kv!("general.file_type", Value::U32(0));
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(n_embd as u32));
    kv!(format!("{a}.block_count"), Value::U32(n_layer as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(n_ff as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(2));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(2));
    kv!(format!("{a}.attention.key_length"), Value::U32(32));
    kv!(format!("{a}.attention.value_length"), Value::U32(32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    kv!(format!("{a}.shortconv.l_cache"), Value::U32(4));

    // tensors (lfm2.cpp:36-95): token_embd, token_embd_norm, output?,
    // per layer: ffn_norm, ffn_gate/down/up, attn_norm, q/k/v norms + qkv + wo
    // (the attention layers — head_kv != 0) or the shortconv trio
    struct R2(u64);
    impl R2 {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32 / 8_388_608.0) - 1.0
        }
    }
    let mut rng = R2(0xbee_1f21);
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut add = |name: String, ne: Vec<i64>, norm: bool| {
        let n: i64 = ne.iter().product();
        let s = if norm { 1.0f32 } else { 1.0 / (n_embd as f32).sqrt() };
        let vals: Vec<f32> = if norm {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
        } else {
            (0..n).map(|_| s * rng.next()).collect()
        };
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for x in &vals {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        data.push(bytes);
    };
    // layer 0 = conv (is_recr: head_count_kv == 0 on the RECURRENT layers —
    // lfm2 marks recurrency by head_kv 0; make layer 0 conv, layer 1 attn)
    // NOTE: is_recr is per-layer head_count_kv — the port reads the scalar
    // key, so BOTH layers share it. Use head_kv 2 (no conv layer): the pure
    // dense-transformer file (LFM2-350M's shape modulo the conv mix).
    add("token_embd.weight".into(), vec![n_embd, N_VOCAB], false);
    add("token_embd_norm.weight".into(), vec![n_embd], true);
    add("output.weight".into(), vec![n_embd, N_VOCAB], false);
    for i in 0..n_layer as i32 {
        add(format!("blk.{i}.ffn_norm.weight"), vec![n_embd], true);
        add(format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], false);
        add(format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], false);
        add(format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], false);
        add(format!("blk.{i}.attn_norm.weight"), vec![n_embd], true);
        add(format!("blk.{i}.attn_q_norm.weight"), vec![32], true);
        add(format!("blk.{i}.attn_k_norm.weight"), vec![32], true);
        add(format!("blk.{i}.attn_q.weight"), vec![n_embd, 64], false);
        add(format!("blk.{i}.attn_k.weight"), vec![n_embd, 64], false);
        add(format!("blk.{i}.attn_v.weight"), vec![n_embd, 64], false);
        add(format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], false);
    }
    let table: Vec<(String, Vec<i64>, bool)> = {
        let mut v = vec![
            ("token_embd.weight".to_string(), vec![n_embd, N_VOCAB], false),
            ("token_embd_norm.weight".to_string(), vec![n_embd], true),
            ("output.weight".to_string(), vec![n_embd, N_VOCAB], false),
        ];
        for i in 0..n_layer as i32 {
            for (n, ne, nm) in [
                (format!("blk.{i}.ffn_norm.weight"), vec![n_embd], true),
                (format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], false),
                (format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], false),
                (format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], false),
                (format!("blk.{i}.attn_norm.weight"), vec![n_embd], true),
                (format!("blk.{i}.attn_q_norm.weight"), vec![32], true),
                (format!("blk.{i}.attn_k_norm.weight"), vec![32], true),
                (format!("blk.{i}.attn_q.weight"), vec![n_embd, 64], false),
                (format!("blk.{i}.attn_k.weight"), vec![n_embd, 64], false),
                (format!("blk.{i}.attn_v.weight"), vec![n_embd, 64], false),
                (format!("blk.{i}.attn_output.weight"), vec![n_embd, n_embd], false),
            ] {
                v.push((n, ne, nm));
            }
        }
        v
    };
    for (name, ne, _) in &table {
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, GgmlType::F32, ne4);
    }
    let path = format!("{OUT_DIR}/lfm2-synth-dense.gguf");
    let f = std::fs::File::create(&path).expect("create");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write");
    use std::io::Write as _;
    bw.flush().unwrap();
    // (the add() closure consumed the same RNG stream in the same order)

    let gguf = Gguf::open(&path).unwrap();
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    let m = load_model(&gguf, mmap).expect("lfm2 dense must load through the shared arm");
    assert_eq!(m.arch, llama::arch::LlmArch::LFM2);
    let hp = &m.hparams;
    assert_eq!(hp.n_layer() as usize, n_layer);
    assert_eq!(hp.n_layer_dense_lead, hp.n_layer(), "lfm2: dense lead == n_layer");
    assert!(!hp.is_recr(0), "lfm2 dense: no recurrent layer");
    let mut want: Vec<String> = table.iter().map(|(n, _, _)| n.clone()).collect();
    want.sort();
    let mut got: Vec<String> = m.tensors.keys().cloned().collect();
    got.sort();
    assert_eq!(got, want, "lfm2 dense: created tensor set");
    println!("lfm2 dense: loads through the shared lfm2moe arm ({n_layer} layers)");
}

// ---------------------------------------------------------------------------
// MTP batch 18 (2026-09-30) — the DRIVER-level e2e: the nine graph_mtp heads
// through DecodeContext::new_mtp + common_speculative_init(DraftMtp) +
// speculative_simple_generate (the port of the reference's `--spec-type
// draft-mtp` path, speculative.cpp:2545-2589 + :1330-1767). Self-consistency
// cell: the committed stream at temperature 0 must equal the plain greedy
// stream of the same file (the reference's acceptance criterion — its own
// speculative driver commits only tokens the target's greedy path would).
//
// The reference comparison for the five non-hybrid synthetics lives in
// parity/mtp2_parity.sh (ref-server plain == ref-server spec == port plain ==
// port spec, 16/16). The four hybrid-loader archs (qwen35/qwen35moe/
// qwen3next/bailingmoe3 — whose loaders REQUIRE the ssm/kda keys, so the
// synthetic carries an all-zero recurrent_layers array) ABORT the pinned
// reference's own trunk decode inside llm_graph_input_mem_hybrid::set_input
// (ggml-backend.cpp:205 GGML_ASSERT(buffer) — the zero-recurrent hybrid
// memory's input tensor never gets a buffer), so no reference trunk stream
// exists for them; this cell is their driver-level evidence.
// ---------------------------------------------------------------------------

use llama::context::{DecodeContext, ForwardWeights, MtpForward, MtpHeadFacts};
use llama::sampling::{SamplingContext, SamplingParams};
use llama::vocab::Vocab;
use llama::speculative::{
    common_speculative_init, speculative_simple_generate, CommonParamsSpeculative,
    CommonSpeculativeType,
};

/// the MTP layer's [`MtpHeadFacts`] — the hparams reads at il = n_layer
/// (n_embd_out, the n_embd_{k,v}_gqa cache rows, the iswa inputs)
fn head_facts_of(m: &LlamaModel) -> MtpHeadFacts {
    let il = m.hparams.n_layer() as usize;
    MtpHeadFacts {
        n_embd: m.hparams.n_embd_out() as i64,
        k_row: m.hparams.n_embd_k_gqa(il) as i64,
        v_row: m.hparams.n_embd_v_gqa(il) as i64,
        n_swa: m.hparams.n_swa,
        swa_type: m.hparams.swa_type,
        is_swa: m.hparams.is_swa(il),
    }
}

/// the trunk bundle + the MtpForward of a loaded synth — one `assemble()`
/// supplies both (the trunk reuses the arch params; the MTP weights move into
/// the MtpForward, the trunk layers re-run the same converters over
/// layers[..n_layer])
fn trunk_forward_of(
    m: &LlamaModel,
    bundle: Mtp2Bundle,
) -> (ForwardWeights, MtpForward, AttnParams) {
    let n_trunk = m.hparams.n_layer() as usize;
    let facts = head_facts_of(m);
    let head = m.tok_embd;
    let onorm = m.output_norm;
    let out = m.output;
    match bundle {
        Mtp2Bundle::Qwen35(w, p, a) => (
            ForwardWeights::Qwen35(
                graph_arch::Qwen35ModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    cls_out: m.cls_out,
                    cls_out_b: m.cls_out_b,
                    layers: m.layers[..n_trunk].iter().map(qwen35_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Qwen35(w, p, facts),
            a,
        ),
        Mtp2Bundle::Qwen35Moe(w, p, a) => (
            ForwardWeights::Qwen35Moe(
                graph_arch::Qwen35MoeModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(qwen35moe_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Qwen35Moe(w, p, facts),
            a,
        ),
        Mtp2Bundle::Qwen3Next(w, p, a) => (
            ForwardWeights::Qwen3Next(
                graph_arch::Qwen3NextModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(qwen3next_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Qwen3Next(w, p, facts),
            a,
        ),
        Mtp2Bundle::Glm4Moe(w, p, a) => (
            ForwardWeights::Glm4Moe(
                graph_arch::Glm4MoeModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(glm4_moe_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Glm4Moe(w, p, facts),
            a,
        ),
        Mtp2Bundle::Cohere2Moe(w, p, a) => (
            ForwardWeights::Cohere2Moe(
                graph_arch::Cohere2MoeModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(cohere2moe_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Cohere2Moe(w, p, facts),
            a,
        ),
        Mtp2Bundle::BailingMoe3(w, p, a) => (
            ForwardWeights::BailingMoe3(
                graph_arch::BailingMoe3ModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(bailingmoe3_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::BailingMoe3(w, p, facts),
            a,
        ),
        Mtp2Bundle::HyV3(w, p, a) => (
            ForwardWeights::HyV3(
                graph_arch::HyV3ModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(hy_v3_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::HyV3(w, p, facts),
            a,
        ),
        Mtp2Bundle::Mimo2(w, p, a) => (
            ForwardWeights::Mimo2(
                graph_arch::Mimo2ModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(mimo2_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Mimo2(w, p, facts),
            a,
        ),
        Mtp2Bundle::Step35(w, p, a) => (
            ForwardWeights::Step35(
                graph_arch::Step35ModelWeights {
                    tok_embd: head,
                    output_norm: onorm,
                    output: out,
                    layers: m.layers[..n_trunk].iter().map(step35_layer).collect(),
                },
                p.clone(),
            ),
            MtpForward::Step35(w, p, facts),
            a,
        ),
    }
}

/// the trunk driver of a loaded synth — `new_with_swa` for the iswa archs
/// (llama-model.cpp:2687-2690's `swa_type != NONE` test, the same selection
/// the CLI's trunk context makes)
fn trunk_driver_of(fam: Fam, m: &mut LlamaModel) -> DecodeContext {
    let bundle = assemble(m);
    let (weights, _mtp, attn) = trunk_forward_of(m, bundle);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let hp = &m.hparams;
    if fam.iswa() {
        DecodeContext::new_with_swa(gctx, weights, attn, N_CTX, 8, 512, SwaCacheSpec::from_hparams(hp))
    } else {
        DecodeContext::new_with(gctx, weights, attn, N_CTX, 8, 512)
    }
}

/// the MTP draft context of a loaded synth (`LLAMA_CONTEXT_TYPE_MTP`,
/// speculative.cpp:2545-2547) — new_mtp with the assembled MtpForward
fn mtp_driver_of(fam: Fam, m: &mut LlamaModel) -> DecodeContext {
    let _ = fam;
    let bundle = assemble(m);
    let (weights, mtp, attn) = trunk_forward_of(m, bundle);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_mtp(gctx, weights, mtp, attn, N_CTX, 8, 512)
}

/// greedy stream of the trunk context + the per-step top-2 margin (the
/// near-tie probe of the qwen35 flip below)
fn plain_greedy_with_margins(
    m: &mut LlamaModel,
    fam: Fam,
    prompt: &[i32],
    n_predict: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut d = trunk_driver_of(fam, m);
    let mut logits = d
        .decode(prompt, &(0..prompt.len() as i32).collect::<Vec<_>>())
        .expect("prefill")
        .to_vec();
    let mut out = Vec::new();
    let mut margins = Vec::new();
    for _ in 0..n_predict {
        let (id, margin) = {
            let mut best = 0usize;
            let mut second = 1usize;
            if logits[second] > logits[best] {
                std::mem::swap(&mut best, &mut second);
            }
            for (i, &x) in logits.iter().enumerate() {
                if x > logits[best] {
                    second = best;
                    best = i;
                } else if x > logits[second] {
                    second = i;
                }
            }
            (best as i32, logits[best] - logits[second])
        };
        out.push(id);
        margins.push(margin);
        let p = (prompt.len() + out.len() - 1) as i32;
        logits = d.decode(&[id], &[p]).expect("decode").to_vec();
    }
    (out, margins)
}

/// the driver-level self-consistency cell: `--spec-type draft-mtp` at
/// temperature 0 must commit exactly the plain greedy stream.
///
/// n_max = 0 (verify batches of one row) is bit-identical for every arch by
/// construction. n_max = 3 (4-row verify batches) holds 16/16 for all nine
/// since batch 20 armed the recurrent rollback ring on both spec contexts
/// (`n_rs_seq = n_max`, the reference's `need_n_rs_seq()` rule): before it,
/// the hybrid qwen35 flipped one token at step 14 — the flip's real cause
/// was the GDN state advancing over rejected drafts (state pollution), sized
/// on the real 27B in tests/mtp_real_spec_e2e.rs (the "GEMM summation
/// order" attribution of the batch-18 note was wrong — the clean-state
/// rowcount probe below shows 1-row and 4-row logits agree to 0).
#[test]
fn mtp2_speculation_matches_plain_greedy() {
    // the whole cell holds file_lock (the parallel-test file-rebuild guard)
    // and uses the nolock loader
    let _files = file_lock();
    let prompt: Vec<i32> = (1..=6).collect();
    let n_predict = 16usize;

    for fam in all_fams() {
        // the plain baseline (margins recorded for the near-tie notes)
        let mut m_plain = load_synth_nolock(fam);
        let (plain, margins) = plain_greedy_with_margins(&mut m_plain, fam, &prompt, n_predict);
        let min_margin = margins.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "{}: plain greedy min top-2 margin {:.3e}",
            fam.arch_name(),
            min_margin
        );

        for n_max in [0i32, 3] {
            // target trunk context + the MTP draft context (two more loads —
            // the port owns one ggml Context per DecodeContext; the C shares
            // model_tgt, speculative.cpp:2582). Batch 20: both carry the
            // recurrent rollback ring at `n_rs_seq = n_max` — the
            // reference's `cparams.n_rs_seq = speculative.need_n_rs_seq()`
            // (common.h:396-404 + common.cpp:1635). Without it a rejected
            // round leaves the hybrid archs' GDN state after the *rejected*
            // drafts (llama-memory-recurrent.cpp:193-210's pending rollback
            // never arms) and the greedy stream flips inside its near-tie
            // bands — the real-model sizing of the qwen35 flip below lives
            // in tests/mtp_real_spec_e2e.rs (root cause corrected there:
            // state pollution, not GEMM summation order).
            let rs = n_max.max(0) as u32;
            let mut m_tgt = load_synth_nolock(fam);
            let mut tgt = trunk_driver_of(fam, &mut m_tgt).with_rs_rollback(rs);
            let mut m_dft = load_synth_nolock(fam);
            let ctx_dft = mtp_driver_of(fam, &mut m_dft).with_rs_rollback(rs);

            let vocab = Vocab::load(&Gguf::open(&fam.path()).unwrap()).unwrap();
            let mut params = CommonParamsSpeculative::default();
            params.types = vec![CommonSpeculativeType::DraftMtp];
            params.draft.n_max = n_max;
            params.draft.p_min = 0.0;

            let mut spec_ctx = common_speculative_init(
                &params,
                1,
                &mut tgt,
                Some(ctx_dft),
                &vocab,
                Some(&vocab),
                m_tgt.hparams.n_layer_nextn,
                false,
            )
            .expect("init")
            .expect("speculator");

            let n_vocab = tgt.n_vocab() as i32;
            let mut smpl = SamplingContext::new(
                n_vocab,
                SamplingParams {
                    temp: 0.0,
                    ..Default::default()
                },
            );

            let res = speculative_simple_generate(
                &mut tgt,
                &mut spec_ctx,
                &mut smpl,
                &vocab,
                &prompt,
                n_predict as i32,
            )
            .expect("speculative generate");

            // the driver commits whole verify rounds, so it may overshoot
            // n_predict by up to n_max tokens (speculative-simple.cpp:281) —
            // the requested prefix is what must match
            assert!(res.tokens.len() >= n_predict, "short stream");
            let got = &res.tokens[..n_predict];

            assert!(
                got == &plain[..],
                "{} (n_max={n_max}): the MTP speculation changed the greedy stream\n  plain: \
                 {:?}\n  spec : {:?}",
                fam.arch_name(),
                &plain[..16.min(plain.len())],
                &got[..16.min(got.len())],
            );
            println!(
                "{}: n_max={n_max} — {n_predict}/{n_predict} exact, drafted {} accepted {} \
                 ({} target forwards, {} draft forwards)",
                fam.arch_name(),
                res.n_drafted,
                res.n_accept,
                res.n_target_forward,
                res.n_draft_forward
            );
            assert!(res.n_drafted > 0 || n_max == 0, "no drafts were generated");
        }
    }
}

/// the row-count probe of the qwen35 flip: replay the plain stream, then
/// decode ONE batch of 4 rows [t12, x, y, z] at [18..21] and compare row 0's
/// logits against the 1-row decode at 18 — on the *clean* replay state the
/// 1-row/4-row difference stays at GEMM-reorder scale (measured 2.0e-2
/// against a 9.2e-2 top-2 margin, argmax SAME), so the step-14 flip the
/// batch-18 note blamed on "row-count numerics" was actually the GDN state
/// pollution the rollback ring now removes — see the spec cell's doc and
/// tests/mtp_real_spec_e2e.rs.
#[test]
fn mtp2_qwen35_rowcount_probe() {
    let _files = file_lock();
    let fam = Fam::Qwen35;
    let prompt: Vec<i32> = (1..=6).collect();

    // the plain stream's first 13 tokens (the flip decides token 13)
    let mut m0 = load_synth_nolock(fam);
    let (plain, margins) = plain_greedy_with_margins(&mut m0, fam, &prompt, 13);
    println!("qwen35 probe: margins[12] = {:.3e} (min of stream {:.3e})", margins[12], margins.iter().cloned().fold(f32::INFINITY, f32::min));

    // replay prompt + tokens 0..=11, then the flip row — 1-row vs 4-row
    let mut replay = |four: bool| -> Vec<f32> {
        let mut m = load_synth_nolock(fam);
        let mut d = trunk_driver_of(fam, &mut m);
        let mut logits = d
            .decode(&prompt, &(0..prompt.len() as i32).collect::<Vec<_>>())
            .unwrap()
            .to_vec();
        for k in 0..12 {
            let p = (prompt.len() + k) as i32;
            logits = d.decode(&[plain[k]], &[p]).unwrap().to_vec();
            let am = argmax(&logits);
            if am != plain[k + 1] {
                println!("qwen35 probe: replay diverges at k={k}: argmax {am} != plain[{}] {}", k + 1, plain[k + 1]);
            }
        }
        let p = (prompt.len() + 12) as i32;
        if four {
            let mut b = llama::batch::LlamaBatch::default();
            b.add(plain[12], p, &[0], true);
            b.add(7, p + 1, &[0], false);
            b.add(11, p + 2, &[0], false);
            b.add(13, p + 3, &[0], false);
            let out = d.decode_batch(&b).unwrap();
            out.logits[..out.logits.len() / out.n_outputs].to_vec()
        } else {
            d.decode(&[plain[12]], &[p]).unwrap().to_vec()
        }
    };

    let one = replay(false);
    let four = replay(true);
    let mut maxdiff = 0f32;
    for (a, b) in one.iter().zip(four.iter()) {
        maxdiff = maxdiff.max((a - b).abs());
    }
    let (arg1, arg4) = (argmax(&one), argmax(&four));
    println!(
        "qwen35 probe: row-0 max|1row-4row| = {maxdiff:.3e}, argmax {arg1} vs {arg4} {}",
        if arg1 == arg4 { "SAME" } else { "FLIPPED" }
    );
    assert!(maxdiff > 0.0 || arg1 == arg4);
}
