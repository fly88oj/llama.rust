//! Architecture dispatch: `LlamaModel` -> the `ForwardWeights` bundle the
//! matching `graph_arch` builder consumes.
//!
//! The wiring is the same one `crates/tools/llama-cli` and the per-arch e2e
//! tests use (llama-cli's `main.rs`, section "arch wiring"), so llama-bench
//! reaches exactly the builders those tests verified. One arm per builder in
//! `graph_arch.rs`; `Err` names the arch, like llama-cli's fallback.

use llama::arch::LlmArch;
use llama::context::ForwardWeights;
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::graph_arch;
use llama::hparams::LlamaHparams;
use llama::model::LlamaModel;

/// `AttnParams` of layer `il` (llama-model.cpp's generic head geometry +
/// `rope_runtime()` = the full llama-context.cpp:106-215 rope derivation).
pub fn attn_params(hp: &LlamaHparams, il: usize, use_flash_attn: bool) -> AttnParams {
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
        use_flash_attn,
    }
}

/// First non-recurrent layer — the hybrid archs keep their attention geometry
/// there (`granite-hybrid` / `lfm2moe` params derive it exactly like this).
fn first_attn_layer(hp: &LlamaHparams, n_layer: usize) -> usize {
    (0..n_layer).find(|&il| !hp.is_recr(il)).unwrap_or(0)
}

/// The layer-0 `AttnParams` the (uniform-geometry) builders take; the hybrid
/// archs get the first *attention* layer's instead.
pub fn build_attn(hp: &LlamaHparams, arch: LlmArch, flash_attn: bool) -> AttnParams {
    let n_trunk = hp.n_layer() as usize;
    match arch {
        LlmArch::GRANITE_HYBRID | LlmArch::LFM2MOE => {
            attn_params(hp, first_attn_layer(hp, n_trunk), flash_attn)
        }
        _ => attn_params(hp, 0, flash_attn),
    }
}

/// The dispatch of llama-cli's `main.rs` (same arms, same builders).
pub fn build_weights(model: &LlamaModel, attn: AttnParams) -> Result<ForwardWeights, String> {
    let hp = &model.hparams;
    let n_trunk = hp.n_layer() as usize;
    Ok(match model.arch {
        LlmArch::QWEN2 => {
            let layers: Vec<LayerWeights> = model
                .layers
                .iter()
                .map(|l| LayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    wq_b: l.wq_b,
                    wk_b: l.wk_b,
                    wv_b: l.wv_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                })
                .collect();
            ForwardWeights::Qwen2(ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers,
            })
        }
        LlmArch::LLAMA => {
            let layers = model
                .layers
                .iter()
                .map(|l| graph_arch::LlamaLayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    wq_b: l.wq_b,
                    wk_b: l.wk_b,
                    wv_b: l.wv_b,
                    wo_b: l.wo_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                    ffn_gate_b: l.ffn_gate_b,
                    ffn_down_b: l.ffn_down_b,
                    ffn_up_b: l.ffn_up_b,
                })
                .collect();
            ForwardWeights::Llama(graph_arch::LlamaModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        LlmArch::PHI3 => {
            let layers = model
                .layers
                .iter()
                .map(|l| graph_arch::Phi3LayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wqkv: l.wqkv.expect("wqkv"),
                    wqkv_b: l.wqkv_b,
                    wo: l.wo.expect("wo"),
                    wo_b: l.wo_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                })
                .collect();
            ForwardWeights::Phi3(graph_arch::Phi3ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        LlmArch::GEMMA2 | LlmArch::GEMMA3 => {
            let gemma2 = model.arch == LlmArch::GEMMA2;
            let layers = model
                .layers
                .iter()
                .map(|l| graph_arch::GemmaLayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    attn_post_norm: l.attn_post_norm.expect("attn_post_norm"),
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                    ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
                    attn_q_norm: l.attn_q_norm,
                    attn_k_norm: l.attn_k_norm,
                })
                .collect();
            let gp = graph_arch::GemmaParams {
                attn,
                attention_scale: 1.0 / (attn.n_embd_head_k as f32).sqrt(),
                attn_logit_softcapping: hp.f_attn_logit_softcapping,
                final_logit_softcapping: hp.f_final_logit_softcapping,
                attn_soft_cap: hp.attn_soft_cap,
                final_softcap_unguarded: gemma2, // gemma2.cpp:167 unguarded
            };
            let w = graph_arch::GemmaModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers,
            };
            if gemma2 {
                ForwardWeights::Gemma2(w, gp)
            } else {
                ForwardWeights::Gemma3(w, gp)
            }
        }
        LlmArch::QWEN3 => ForwardWeights::Qwen3(qwen3_weights(model, n_trunk)),
        LlmArch::OPENAI_MOE => {
            // gpt-oss: the builder needs the head/rope geometry and the MoE
            // width / per-layer SWA pattern (openai-moe.cpp:3-18 +
            // llama-model.cpp:2251)
            ForwardWeights::GptOss(gpt_oss_weights(model, n_trunk), gpt_oss_params(hp, n_trunk))
        }
        LlmArch::GEMMA4 => ForwardWeights::Gemma4(gemma4_weights(model, n_trunk), gemma4_params(hp, n_trunk, attn)),
        LlmArch::GRANITE_HYBRID => {
            ForwardWeights::Granite(granite_weights(model, n_trunk), granite_params(hp, n_trunk, attn))
        }
        LlmArch::LFM2MOE => ForwardWeights::Lfm2(lfm2_weights(model, n_trunk), lfm2_params(hp, n_trunk, attn)),
        LlmArch::QWEN35 => ForwardWeights::Qwen35(qwen35_weights(model, n_trunk), qwen35_params(hp, n_trunk, attn)),
        other => {
            return Err(format!(
                "arch '{}' ({other:?}) has a loader but no forward builder in this port \
                 (see FILE_MAP.md's architecture matrix)",
                other.name()
            ));
        }
    })
}

/// qwen3.cpp:31-46 — separate Q/K/V + per-head attn_q_norm/attn_k_norm.
fn qwen3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3ModelWeights {
    graph_arch::Qwen3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// openai-moe.cpp:30-66 — 22 tensors per layer.
fn gpt_oss_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GptOssModelWeights {
    graph_arch::GptOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GptOssLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l.attn_post_norm.unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                // openai-moe.cpp:58 makes attn_out.bias required for gpt-oss
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                attn_sinks: l.attn_sinks.unwrap_or_else(|| panic!("layer {il}: attn_sinks")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_inp_b: l.ffn_gate_inp_b.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_b")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_up_exps_b: l.ffn_up_exps_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps_b")),
                ffn_gate_exps: l.ffn_gate_exps.unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_gate_exps_b: l.ffn_gate_exps_b.unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps_b")),
                ffn_down_exps: l.ffn_down_exps.unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_down_exps_b: l.ffn_down_exps_b.unwrap_or_else(|| panic!("layer {il}: ffn_down_exps_b")),
            })
            .collect(),
    }
}

/// openai-moe.cpp:3-18 + llama-graph.cpp:2288-2290: the MoE/SWA parameters.
fn gpt_oss_params(hp: &LlamaHparams, n_layer: usize) -> graph_arch::GptOssParams {
    graph_arch::GptOssParams {
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        swiglu_oai_alpha: 1.702,
        swiglu_oai_limit: 7.0,
        expert_weights_scale: hp.expert_weights_scale,
        // openai-moe.cpp:12 `load_swa_pattern(ml, 2)`
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
    }
}

/// gemma4.cpp:12-147 — fused or separate qkv, per-head q/k norms, MoE branch.
fn gemma4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma4ModelWeights {
    graph_arch::Gemma4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        per_layer_model_proj: m.per_layer_model_proj,
        per_layer_proj_norm: m.per_layer_proj_norm,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gemma4LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l.attn_post_norm.unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                rope_freqs: l.rope_freqs,
                out_scale: l.out_scale,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_post_norm: l.ffn_post_norm.unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_inp_s: l.ffn_gate_inp_s,
                ffn_pre_norm_2: l.ffn_pre_norm_2,
                ffn_post_norm_1: l.ffn_post_norm_1,
                ffn_post_norm_2: l.ffn_post_norm_2,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_down_exps_s: l.ffn_down_exps_s,
                per_layer_inp_gate: l.per_layer_inp_gate,
                per_layer_proj: l.per_layer_proj,
                per_layer_post_norm: l.per_layer_post_norm,
            })
            .collect(),
    }
}

/// gemma4.cpp:3-25 + the per-layer vectors (SWA layers 256x8, full 512x1).
fn gemma4_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Gemma4Params {
    graph_arch::Gemma4Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
        f_attention_scale: hp.f_attention_scale,
        f_final_logit_softcapping: hp.f_final_logit_softcapping,
        n_embd_per_layer: hp.n_embd_per_layer as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: (0..n_layer).map(|il| hp.n_ff_exp(il)).collect(),
    }
}

/// granite-hybrid.cpp:12-142 — mamba2 mixer + attention tensors, MoE experts.
fn granite_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GraniteLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_conv1d_b: l.ssm_conv1d_b,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_d: l.ssm_d,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                wo_b: l.wo_b,
                // granite-hybrid.cpp:222 passes `layer.rope_freqs` (NULL on
                // layer 0 — the C loader skips blk.0.rope_freqs)
                rope_freqs: if il == 0 { None } else { l.rope_freqs },
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
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

/// granite-hybrid.cpp:4-9/124-139 + hparams — `attn` must be an *attention*
/// layer's geometry (see `first_attn_layer`).
fn granite_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::GraniteParams {
    graph_arch::GraniteParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        has_rope: (0..n_layer).map(|il| hp.has_rope(il)).collect(),
        d_conv: hp.ssm_d_conv as i64,
        d_inner: hp.ssm_d_inner as i64,
        d_state: hp.ssm_d_state as i64,
        n_ssm_head: hp.ssm_dt_rank as i64,
        n_group: hp.ssm_n_group as i64,
        f_logit_scale: hp.f_logit_scale,
        f_residual_scale: hp.f_residual_scale,
        f_embedding_scale: hp.f_embedding_scale,
        f_attention_scale: hp.f_attention_scale,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_shexp: hp.n_ff_shexp as i64,
        expert_weights_scale: hp.expert_weights_scale,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

/// lfm2moe.cpp — shortconv block + gated attention tensors, dense|MoE FFN.
fn lfm2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Lfm2ModelWeights {
    graph_arch::Lfm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Lfm2LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                shortconv_conv: l.shortconv_conv,
                shortconv_in_proj: l.shortconv_in_proj,
                shortconv_out_proj: l.shortconv_out_proj,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
            })
            .collect(),
    }
}

/// lfm2moe.cpp:3-20 + hparams — `attn` must be an attention layer's geometry.
fn lfm2_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Lfm2Params {
    graph_arch::Lfm2Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_shortconv_l_cache: hp.n_shortconv_l_cache as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: hp.n_ff_exp(0),
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        causal_attn: hp.causal_attn,
        n_embd_r: hp.n_embd_r(),
    }
}

/// qwen35.cpp:31-153 — attention layers carry separate q/k/v + q/k norms,
/// recurrent (gated delta net) layers the fused wqkv + ssm tensors.
fn qwen35_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35ModelWeights {
    graph_arch::Qwen35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        cls_out: m.cls_out,
        cls_out_b: m.cls_out_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen35LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l.attn_post_norm.unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
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
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

/// qwen35.cpp:3-30 + hparams — IMROPE sections, per-layer head geometry, the
/// gated delta net geometry and the recurrent state cell sizes.
fn qwen35_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35Params {
    graph_arch::Qwen35Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}