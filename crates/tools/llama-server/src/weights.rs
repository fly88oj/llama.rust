//! weights.rs — the LlamaModel → graph_arch weight-bundle helpers of the
//! arch batches 1-12, verbatim copies of llama-cli's wiring (the same
//! derivations the per-arch e2e tests use, crates/llama/tests/
//! arch_batch*_e2e.rs). The server's `forward_weights` dispatch (main.rs)
//! reaches these so every wired arch serves the exact graph llama-cli drives —
//! keeping the two tools byte-comparable is the point of the server parity
//! protocol (parity/run_server_parity.sh).
//!
//! eurobert (batch 11b) is encoder-only and never passes through
//! `forward_weights` — the server reaches it through `EncoderContext` /
//! `EncoderWeights::Eurobert` like BERT (main.rs's encoder branch).

use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::LlamaHparams;
use llama::model::LlamaModel;

/// First non-recurrent layer — the hybrid archs keep their attention geometry
/// there (granite-hybrid / lfm2moe params derive it exactly like this).
pub fn first_attn_layer(hp: &LlamaHparams, n_layer: usize) -> usize {
    (0..n_layer).find(|&il| !hp.is_recr(il)).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// arch batch 1 (2026-09): the first wave — gpt-oss / gemma4 / granite-hybrid /
// lfm2moe / qwen35 / gpt2 / phi2 / starcoder2 / command-r / gptneox / olmo2.
// Verbatim copies of llama-cli's wiring (the `weights.rs` pattern: the server
// and the CLI must drive the identical graph).
// ---------------------------------------------------------------------------

/// openai-moe.cpp:30-66 — 22 tensors per layer (q/k/v + wo biases, attn_sinks,
/// router bias, per-expert biases; all required there except the q/k/v biases).
pub fn gpt_oss_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GptOssModelWeights {
    graph_arch::GptOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GptOssLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                attn_sinks: l
                    .attn_sinks
                    .unwrap_or_else(|| panic!("layer {il}: attn_sinks")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_inp_b: l
                    .ffn_gate_inp_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_b")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_up_exps_b: l
                    .ffn_up_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps_b")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_gate_exps_b: l
                    .ffn_gate_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps_b")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_down_exps_b: l
                    .ffn_down_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps_b")),
            })
            .collect(),
    }
}

/// openai-moe.cpp:3-18 + llama-graph.cpp:2288-2290: the MoE/SWA parameters.
pub fn gpt_oss_params(hp: &LlamaHparams, n_layer: usize) -> graph_arch::GptOssParams {
    graph_arch::GptOssParams {
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        swiglu_oai_alpha: 1.702,
        swiglu_oai_limit: 7.0,
        expert_weights_scale: hp.expert_weights_scale,
        // openai-moe.cpp:12 `load_swa_pattern(ml, 2)` -> the is_swa_impl vector
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        // llama-model.cpp:2251 get_rope_freq_base/scale
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
    }
}

/// gemma4.cpp:12-147 — fused or separate qkv, per-head q/k norms, optional
/// rope freq factors / out_scale, dense|MoE FFN with the second branch.
pub fn gemma4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma4ModelWeights {
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
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
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
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
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

/// gemma4.cpp:3-25 + the per-layer vectors (gemma4 is the arch whose head
/// dims/rot/n_ff vary per layer: SWA layers 256x8, full layers 512x1).
pub fn gemma4_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Gemma4Params {
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

/// granite-hybrid.cpp:12-142 — mamba2 mixer + attention tensors, optional
/// biases, MoE experts + shared expert.
pub fn granite_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GraniteLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
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
pub fn granite_params(
    hp: &LlamaHparams,
    n_layer: usize,
    attn: AttnParams,
) -> graph_arch::GraniteParams {
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

/// mamba.cpp:75-110 / mamba2.cpp:60-88 — the pure-recurrent stack. `mamba2`
/// picks the mixer enum variant the shared mamba graph branches on
/// (mamba.cpp:87 `model.arch == LLM_ARCH_MAMBA2`).
pub fn mamba_weights(m: &LlamaModel, n_trunk: usize, mamba2: bool) -> graph_arch::MambaModelWeights {
    graph_arch::MambaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::MambaLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mixer: if mamba2 {
                    graph_arch::MambaLayerMixer::Mamba2(graph_arch::Mamba2Mixer {
                        ssm_in: l.ssm_in.expect("ssm_in"),
                        ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                        ssm_conv1d_b: l.ssm_conv1d_b,
                        ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                        ssm_a: l.ssm_a.expect("ssm_a"),
                        ssm_d: l.ssm_d.expect("ssm_d"),
                        ssm_norm: l.ssm_norm,
                        ssm_out: l.ssm_out.expect("ssm_out"),
                    })
                } else {
                    graph_arch::MambaLayerMixer::Mamba1(graph_arch::Mamba1Mixer {
                        ssm_in: l.ssm_in.expect("ssm_in"),
                        ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                        ssm_conv1d_b: l.ssm_conv1d_b.expect("ssm_conv1d_b"),
                        ssm_x: l.ssm_x.expect("ssm_x"),
                        ssm_dt: l.ssm_dt.expect("ssm_dt"),
                        ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                        ssm_dt_norm: l.ssm_dt_norm,
                        ssm_b_norm: l.ssm_b_norm,
                        ssm_c_norm: l.ssm_c_norm,
                        ssm_a: l.ssm_a.expect("ssm_a"),
                        ssm_d: l.ssm_d.expect("ssm_d"),
                        ssm_out: l.ssm_out.expect("ssm_out"),
                    })
                },
            })
            .collect(),
    }
}

/// jamba.cpp:67-128 — mamba1 layers (with the dt/B/C RMS trio) or rope-less
/// attention, then dense|MoE FFN tensors.
pub fn jamba_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::JambaModelWeights {
    graph_arch::JambaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::JambaLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mamba: l.ssm_in.map(|ssm_in| graph_arch::Mamba1Mixer {
                    ssm_in,
                    ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                    ssm_conv1d_b: l.ssm_conv1d_b.expect("ssm_conv1d_b"),
                    ssm_x: l.ssm_x.expect("ssm_x"),
                    ssm_dt: l.ssm_dt.expect("ssm_dt"),
                    ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                    ssm_dt_norm: l.ssm_dt_norm,
                    ssm_b_norm: l.ssm_b_norm,
                    ssm_c_norm: l.ssm_c_norm,
                    ssm_a: l.ssm_a.expect("ssm_a"),
                    ssm_d: l.ssm_d.expect("ssm_d"),
                    ssm_out: l.ssm_out.expect("ssm_out"),
                }),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo,
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
            })
            .collect(),
    }
}

/// nemotron-h.cpp:79-181 — per layer exactly one of {mamba2 mixer, attention,
/// relu² FFN(dense|MoE)} keyed on the loader's is_recr / n_ff split.
pub fn nemotron_h_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::NemotronHModelWeights {
    graph_arch::NemotronHModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::NemotronHLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mamba: l.ssm_in.map(|ssm_in| graph_arch::Mamba2Mixer {
                    ssm_in,
                    ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                    ssm_conv1d_b: l.ssm_conv1d_b,
                    ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                    ssm_a: l.ssm_a.expect("ssm_a"),
                    ssm_d: l.ssm_d.expect("ssm_d"),
                    ssm_norm: l.ssm_norm,
                    ssm_out: l.ssm_out.expect("ssm_out"),
                }),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo,
                wo_b: l.wo_b,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_latent_down: l.ffn_latent_down,
                ffn_latent_up: l.ffn_latent_up,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// deepseek2.cpp:98-158 (+deepseek2ocr.cpp:40-73 — the OCR layers populate
/// only the wqkv/wq/wk/wv/wo + FFN fields, the MLA ones stay None).
pub fn deepseek2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek2ModelWeights {
    graph_arch::Deepseek2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk].iter().map(deepseek2_layer_of).collect(),
    }
}

/// the `model.layers[i]` → `Deepseek2LayerWeights` mapping (deepseek2.cpp /
/// deepseek32.cpp load_arch_tensors field-for-field)
fn deepseek2_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Deepseek2LayerWeights {
    graph_arch::Deepseek2LayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        wq: l.wq,
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wk: l.wk,
        wv: l.wv,
        wq_a: l.wq_a,
        attn_q_a_norm: l.attn_q_a_norm,
        wq_b: l.wq_b,
        wkv_a_mqa: l.wkv_a_mqa,
        attn_kv_a_norm: l.attn_kv_a_norm,
        indexer_k_norm: l.indexer_k_norm,
        indexer_k_norm_b: l.indexer_k_norm_b,
        indexer_proj: l.indexer_proj,
        indexer_attn_k: l.indexer_attn_k,
        indexer_attn_q_b: l.indexer_attn_q_b,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        wkv_b: l.wkv_b,
        wo: l.wo.expect("wo"),
        ffn_norm: l.ffn_norm.expect("ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

/// deepseek4.cpp:82-183 — the trunk weights (hyper-connection mixers, the
/// o_group/o_lora output lora, the per-ratio compressors, hash layers).
pub fn deepseek4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek4ModelWeights {
    graph_arch::Deepseek4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.expect("output_hc_fn"),
        hc_head_base: m.hc_head_base.expect("output_hc_base"),
        hc_head_scale: m.hc_head_scale.expect("output_hc_scale"),
        layers: m.layers[..n_trunk].iter().map(deepseek4_layer_of).collect(),
    }
}

/// the `model.layers[i]` → `Deepseek4LayerWeights` mapping
fn deepseek4_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Deepseek4LayerWeights {
    graph_arch::Deepseek4LayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        attn_sinks: l.attn_sinks.expect("attn_sinks"),
        wq_a: l.wq_a.expect("attn_q_a"),
        attn_q_a_norm: l.attn_q_a_norm.expect("attn_q_a_norm"),
        wq_b: l.wq_b.expect("attn_q_b"),
        wkv: l.wkv_a_mqa.expect("attn_kv"),
        attn_kv_norm: l.attn_kv_a_norm.expect("attn_kv_norm"),
        wo_a: l.wo_a.expect("attn_output_a"),
        wo_b: l.wo_b_dsv4.expect("attn_output_b"),
        hc_attn_fn: l.hc_attn_fn.expect("hc_attn_fn"),
        hc_attn_base: l.hc_attn_base.expect("hc_attn_base"),
        hc_attn_scale: l.hc_attn_scale.expect("hc_attn_scale"),
        hc_ffn_fn: l.hc_ffn_fn.expect("hc_ffn_fn"),
        hc_ffn_base: l.hc_ffn_base.expect("hc_ffn_base"),
        hc_ffn_scale: l.hc_ffn_scale.expect("hc_ffn_scale"),
        attn_comp_wkv: l.attn_comp_wkv,
        attn_comp_wgate: l.attn_comp_wgate,
        attn_comp_ape: l.attn_comp_ape,
        attn_comp_norm: l.attn_comp_norm,
        indexer_proj: l.indexer_proj,
        indexer_attn_q_b: l.indexer_attn_q_b,
        indexer_comp_wkv: l.indexer_comp_wkv,
        indexer_comp_wgate: l.indexer_comp_wgate,
        indexer_comp_ape: l.indexer_comp_ape,
        indexer_comp_norm: l.indexer_comp_norm,
        ffn_norm: l.ffn_norm.expect("ffn_norm"),
        ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
        ffn_gate_tid2eid: l.ffn_gate_tid2eid,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_exp_probs_b_vl: l.ffn_exp_probs_b_vl,
        ffn_gate_exps: l.ffn_gate_exps.expect("ffn_gate_exps"),
        ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
        ffn_up_exps: l.ffn_up_exps.expect("ffn_up_exps"),
        ffn_gate_shexp: l.ffn_gate_shexp.expect("ffn_gate_shexp"),
        ffn_down_shexp: l.ffn_down_shexp.expect("ffn_down_shexp"),
        ffn_up_shexp: l.ffn_up_shexp.expect("ffn_up_shexp"),
    }
}

/// deepseek.cpp:34-67 — plain MHA (build_qkv weights) + dense-lead/MoE FFN.
pub fn deepseek_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DeepseekModelWeights {
    graph_arch::DeepseekModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::DeepseekLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
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

/// lfm2moe.cpp — shortconv block + gated attention tensors, dense|MoE FFN.
pub fn lfm2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Lfm2ModelWeights {
    graph_arch::Lfm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Lfm2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
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
pub fn lfm2_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Lfm2Params {
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
pub fn qwen35_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35ModelWeights {
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
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
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
pub fn qwen35_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35Params {
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

// ---------------------------------------------------------------------------
// arch batch 1 cont. (2026-09-24): gpt2 / phi2 / starcoder2 / command-r /
// gptneox / olmo2 — weight bundles mirroring load_arch_tensors (model.rs) 1:1
// ---------------------------------------------------------------------------

/// gpt2.cpp:15-51 — fused wqkv + both LayerNorm biases, learned pos embedding.
pub fn gpt2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gpt2ModelWeights {
    graph_arch::Gpt2ModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap_or_else(|| panic!("pos_embd")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gpt2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// phi2.cpp:13-41 — separate qkv (fused allowed by `create_tensor_qkv`), both
/// LayerNorm biases and a required lm-head bias.
pub fn phi2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Phi2ModelWeights {
    graph_arch::Phi2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        output_b: m.output_b.unwrap_or_else(|| panic!("output_b")),
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Phi2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// starcoder2.cpp:16-53 — LayerNorm norms with biases, split qkv, GELU FFN.
pub fn starcoder2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StarCoder2ModelWeights {
    graph_arch::StarCoder2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StarCoder2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// command-r.cpp:13-40 — biasless LayerNorm, split qkv, optional per-head Q/K
/// norms (n_layer >= 64), SwiGLU FFN.
pub fn command_r_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::CommandRModelWeights {
    graph_arch::CommandRModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::CommandRLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// gptneox.cpp:54-87 — fused wqkv + fused bias, LayerNorm biases, GELU FFN.
pub fn gptneox_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GptNeoxModelWeights {
    graph_arch::GptNeoxModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GptNeoxLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// olmo2.cpp:26-49 — no attn_norm; per-head q/k norms on the flat projections
/// plus post-attention / post-FFN norms.
pub fn olmo2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Olmo2ModelWeights {
    graph_arch::Olmo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Olmo2LayerWeights {
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
            })
            .collect(),
    }
}

// ---- arch batch 2 (2026-09-25) weight assembly ----

/// codeshell.cpp:12-46 — LN biases everywhere, biased qkv/ffn, required
/// `output.weight` (the token embedding falls back to it, not the other way).
pub fn codeshell_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::CodeshellModelWeights {
    graph_arch::CodeshellModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::CodeshellLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// orion.cpp:12-37 — LN biases, no `attn_output.bias`, unbiased SwiGLU FFN.
pub fn orion_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OrionModelWeights {
    graph_arch::OrionModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OrionLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// olmo.cpp:15-37 — no norm tensors at all (weightless graph norms).
pub fn olmo_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OlmoModelWeights {
    graph_arch::OlmoModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OlmoLayerWeights {
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// xverse.cpp:14-35 — RMS norms, required head.
pub fn xverse_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::XverseModelWeights {
    graph_arch::XverseModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::XverseLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// internlm2.cpp:13-38 — as xverse, but NORM-mode rope (llama-model.cpp:2935).
pub fn internlm2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Internlm2ModelWeights {
    graph_arch::Internlm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Internlm2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// exaone.cpp:12-40 — optional per-layer rope freq factors.
pub fn exaone_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ExaoneModelWeights {
    graph_arch::ExaoneModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ExaoneLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// gemma.cpp:13-35 — v1 (no post-norms, head is always the tied embedding).
pub fn gemma1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma1ModelWeights {
    graph_arch::Gemma1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gemma1LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// falcon.cpp:13-44 — fused qkv, optional 40B second attention norm.
pub fn falcon_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::FalconModelWeights {
    graph_arch::FalconModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::FalconLayerWeights {
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

// ---- arch batch 3 (2026-09-27) weight assembly (mirror of
// arch_batch3_e2e.rs's per-arch helpers) ----

/// baichuan.cpp:28-121 — RMS norms, separate Q/K/V, SwiGLU FFN.
pub fn baichuan_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BaichuanModelWeights {
    graph_arch::BaichuanModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BaichuanLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// bloom.cpp:46-151 — LN-with-bias everywhere, fused biased qkv, token-embedding
/// norm, ALiBi.
pub fn bloom_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BloomModelWeights {
    graph_arch::BloomModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: m
            .token_embd_norm
            .unwrap_or_else(|| panic!("token_embd_norm")),
        tok_norm_b: m
            .token_embd_norm_b
            .unwrap_or_else(|| panic!("token_embd_norm_b")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BloomLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// mpt.cpp:55-171 — fused qkv, optional learned positions, optional q/k norms,
/// GELU-seq FFN with act scales, ALiBi from GGUF KV.
pub fn mpt_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MptModelWeights {
    graph_arch::MptModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MptLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l.attn_norm_b,
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l.ffn_norm_b,
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
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

/// starcoder.cpp:46-154 — LN-with-bias, fused biased qkv, required learned
/// positions, GELU FFN, no ALiBi in this revision.
pub fn starcoder_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StarcoderModelWeights {
    graph_arch::StarcoderModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap_or_else(|| panic!("pos_embd")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StarcoderLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// refact.cpp:41-160 — RMS norms, separate Q/K/V, ALiBi 8.0, SwiGLU with
/// optional biases (rope_freqs loaded but never read).
pub fn refact_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::RefactModelWeights {
    graph_arch::RefactModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::RefactLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plamo.cpp:39-136 — RMS norm, separate Q/K/V, NEOX rope, double residual.
pub fn plamo_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::PlamoModelWeights {
    graph_arch::PlamoModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::PlamoLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// stablelm.cpp:43-172 — LN(-bias), separate Q/K/V, optional per-head q/k LNs,
/// NEOX (partial) rope, sequential-or-parallel residual.
pub fn stablelm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StablelmModelWeights {
    graph_arch::StablelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StablelmLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                ffn_norm: l.ffn_norm,
                ffn_norm_b: l.ffn_norm_b,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// granite.cpp:123-320 (dense) — the granite tensor set minus the mamba2 mixer
/// (all `ssm_*` None); unlike the hybrid arm every layer may carry rope_freqs
/// (layer 0 is an attention layer here).
pub fn granite_dense_weights(m: &LlamaModel) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GraniteLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_in: None,
                ssm_conv1d: None,
                ssm_conv1d_b: None,
                ssm_dt_b: None,
                ssm_a: None,
                ssm_d: None,
                ssm_norm: None,
                ssm_out: None,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                wo_b: l.wo_b,
                rope_freqs: l.rope_freqs,
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

// ---- arch batch 4 (2026-09-28) weight assembly (mirror of
// arch_batch4_e2e.rs's per-arch helpers; granite-moe goes through
// granite_dense_weights above) ----

/// qwen2moe.cpp:19-57 — every layer MoE: the shared-expert router
/// `ffn_gate_inp_shexp` (:54) + the shexp gate/down/up trio.
pub fn qwen2moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen2MoeModelWeights {
    graph_arch::Qwen2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen2MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_inp_shexp: l
                    .ffn_gate_inp_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_shexp")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

// ---- arch batch 8 (2026-09-30): the MoE long-tail family's weight bundles
// (same derivations as arch_batch8_e2e.rs) ----

/// hunyuan-moe.cpp:19-49 — experts at the dense n_ff, the shared expert MLP,
/// per-head q/k norms applied after rope.
pub fn hunyuan_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::HunyuanMoeModelWeights {
    graph_arch::HunyuanMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::HunyuanMoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

/// dots1.cpp:19-67 — dense lead layers + MoE layers with the fat shared
/// expert (n_ff_exp * n_expert_shared wide) and the optional router bias.
pub fn dots1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Dots1ModelWeights {
    graph_arch::Dots1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Dots1LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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

/// bailingmoe.cpp:19-55 — MoE in every layer, shared expert, no q/k norms.
pub fn bailingmoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BailingmoeModelWeights {
    graph_arch::BailingmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BailingmoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

/// bailingmoe2.cpp:21-83 — fused qkv + per-head q/k norms + dense lead.
pub fn bailingmoe2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Bailingmoe2ModelWeights {
    graph_arch::Bailingmoe2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Bailingmoe2LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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

/// glm4-moe.cpp:29-122 — optional q/k norms, attn_post_norm as the FFN norm,
/// optional shared expert, required router bias.
pub fn glm4_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4MoeModelWeights {
    graph_arch::Glm4MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Glm4MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
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

/// minimax-m2.cpp:15-40 — full-width q/k norms, experts at n_ff, required
/// router bias.
pub fn minimax_m2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MinimaxM2ModelWeights {
    graph_arch::MinimaxM2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MinimaxM2LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_exp_probs_b: l
                    .ffn_exp_probs_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_exp_probs_b")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// cohere2moe.cpp:40-143 — the dense/MoE split with the optional fused
/// `ffn_gate_up_exps` tensor and the optional shared expert.
pub fn cohere2moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2MoeModelWeights {
    graph_arch::Cohere2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Cohere2MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
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
            })
            .collect(),
    }
}

/// exaone-moe.cpp:28-101 — per-head q/k norms, dense lead, unconditional
/// shared expert on the MoE layers.
pub fn exaone_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ExaoneMoeModelWeights {
    graph_arch::ExaoneMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ExaoneMoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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

/// qwen3moe.cpp:17-55 — separate q/k/v + per-head q/k norms, MoE experts only
/// (no shared expert anywhere).
pub fn qwen3moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3MoeModelWeights {
    graph_arch::Qwen3MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

// ---- arch batch 6b (2026-09-24) weight assembly: nemotron / grok /
// chameleon / deci / jais / falcon-h1 / plamo2 ----

/// nemotron.cpp:22-43 — LayerNorm+bias norms, create_tensor_qkv attention,
/// relu² up/down MLP with the optional biases.
pub fn nemotron_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::NemotronModelWeights {
    graph_arch::NemotronModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.expect("nemotron output_norm_b"),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::NemotronLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_norm_b: l.attn_norm_b.expect("attn_norm_b"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_norm_b: l.ffn_norm_b.expect("ffn_norm_b"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b,
            })
            .collect(),
    }
}

/// grok.cpp:54-79 — the post-norm attention + GELU MoE (+ optional dense
/// branch) tensor set; `ffn_post_norm` is whichever name the loader filled.
pub fn grok_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GrokModelWeights {
    graph_arch::GrokModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::GrokLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                attn_out_norm: l.attn_out_norm.expect("attn_out_norm"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
                ffn_up_exps: l.ffn_up_exps.expect("ffn_up_exps"),
                ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
            })
            .collect(),
    }
}

/// chameleon.cpp:29-46 — the full-width q/k LayerNorms + SwiGLU tensors.
pub fn chameleon_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ChameleonModelWeights {
    graph_arch::ChameleonModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::ChameleonLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_q_norm: l.attn_q_norm.expect("attn_q_norm"),
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm.expect("attn_k_norm"),
                attn_k_norm_b: l.attn_k_norm_b,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_up: l.ffn_up.expect("ffn_up"),
            })
            .collect(),
    }
}

/// deci.cpp:28-73 — the per-layer attention kinds; the rope factors resolved
/// like the reference's `get_rope_factors` (llama-model.cpp:2259-2272), see
/// `phimoe_weights`.
pub fn deci_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
    n_ctx_orig_yarn: i32,
) -> graph_arch::DeciModelWeights {
    graph_arch::DeciModelWeights {
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
                // get_rope_factors (llama-model.cpp:2259-2272)
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > n_ctx_orig_yarn as i64 {
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
    }
}

/// jais.cpp:25-48 — everything required, including the fused qkv bias.
pub fn jais_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::JaisModelWeights {
    graph_arch::JaisModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.expect("jais output_norm_b"),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::JaisLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_norm_b: l.attn_norm_b.expect("attn_norm_b"),
                wqkv: l.wqkv.expect("wqkv"),
                wqkv_b: l.wqkv_b.expect("wqkv_b"),
                wo: l.wo.expect("wo"),
                wo_b: l.wo_b.expect("wo_b"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_norm_b: l.ffn_norm_b.expect("ffn_norm_b"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_gate_b: l.ffn_gate_b.expect("ffn_gate_b"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b.expect("ffn_down_b"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b.expect("ffn_up_b"),
            })
            .collect(),
    }
}

/// falcon-h1.cpp:68-105 — every layer carries the mamba2 mixer AND the
/// attention tensors.
pub fn falcon_h1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::FalconH1ModelWeights {
    graph_arch::FalconH1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::FalconH1LayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                ssm_in: l.ssm_in.expect("ssm_in"),
                ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                ssm_conv1d_b: l.ssm_conv1d_b,
                ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                ssm_a: l.ssm_a.expect("ssm_a"),
                ssm_d: l.ssm_d.expect("ssm_d"),
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out.expect("ssm_out"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plamo2.cpp:59-103 — the mamba/attention per-layer split; every layer
/// carries the post-mixer/post-FFN norms and the SWIGLU FFN.
pub fn plamo2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Plamo2ModelWeights {
    graph_arch::Plamo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Plamo2LayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
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
                attn_post_norm: l.attn_post_norm.expect("attn_post_norm"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
            })
            .collect(),
    }
}

/// phimoe.cpp:17-45 — the phi3 tensor set (biased RMS norms, fused-or-
/// separate qkv, required wo bias) + the MoE experts. `rope_factors` resolved
/// like the reference's `get_rope_factors` (llama-model.cpp:2259-2272): the
/// layer's own rope_freqs, else long vs short by n_ctx_seq (the server passes
/// the slot-context budget) against n_ctx_orig_yarn.
pub fn phimoe_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
    n_ctx_orig_yarn: i32,
) -> graph_arch::PhimoeModelWeights {
    graph_arch::PhimoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        output_b: m.output_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::PhimoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                // get_rope_factors (llama-model.cpp:2259-2272)
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                },
            })
            .collect(),
    }
}

/// arctic.cpp:19-49 — the double FFN per layer: dense square gate/down/up +
/// the MoE quartet behind `ffn_norm_exps`.
pub fn arctic_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ArcticModelWeights {
    graph_arch::ArcticModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ArcticLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_norm_exps: l
                    .ffn_norm_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_exps")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// olmoe.cpp:15-46 — the full-width [n_embd] q/k norms are required (the MHA
/// precondition is checked by the caller).
pub fn olmoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OlmoeModelWeights {
    graph_arch::OlmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OlmoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// ernie4-5.cpp:26-69 (the ERNIE4_5_MOE branch) — dense lead layers carry the
/// gate/down/up trio, MoE layers the router (+ optional exp_probs_b) and the
/// experts (+ optional shexp trio); both halves stay Option and the builder
/// picks per layer from the dense-lead/MoE-step split.
pub fn ernie45moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Ernie45MoeModelWeights {
    graph_arch::Ernie45MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Ernie45MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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

/// smollm3.cpp:16-39 — the plain qwen2 tensor set (the nope rope pattern is
/// params-only).
pub fn smollm3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Smollm3ModelWeights {
    graph_arch::Smollm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Smollm3LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// seed-oss.cpp:19-42 — q width n_head*head_dim (may differ from n_embd),
/// attn_post_norm doubling as the FFN norm.
pub fn seed_oss_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::SeedOssModelWeights {
    graph_arch::SeedOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::SeedOssLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// openelm.cpp:18-43 — the fused per-layer-width wqkv + per-head q/k norms;
/// the per-layer head counts / FFN widths ride on the layer weights
/// (openelm.cpp:67-69).
pub fn openelm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OpenelmModelWeights {
    graph_arch::OpenelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OpenelmLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                n_head: m.hparams.n_head(il) as i64,
                n_head_kv: m.hparams.n_head_kv(il) as i64,
                n_ff: m.hparams.n_ff(il) as i64,
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// arch batch 9 (2026-10): the linear-attention family — plamo3 / qwen3next /
// kimi-linear / bailingmoe3 weight bundles mirroring load_arch_tensors 1:1
// ---------------------------------------------------------------------------

/// plamo3.cpp:34-57 — the fused-qkv + post-norm + swiglu dense stack.
pub fn plamo3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Plamo3ModelWeights {
    graph_arch::Plamo3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Plamo3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

/// qwen3next.cpp:68-106 — attention layers carry the QG-wide q projection +
/// q/k norms; the GDN layers the wqkv/wqkv_gate (or legacy ssm_in) +
/// conv1d/dt/a/beta_alpha/norm/out set; every layer the MoE tail + the gated
/// shared expert.
pub fn qwen3next_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3NextModelWeights {
    graph_arch::Qwen3NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3NextLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta_alpha: l.ssm_beta_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// qwen3next.cpp:4-29 + hparams — per-layer head geometry, the GDN geometry
/// and the recurrent state cell sizes (the qwen35_params shape).
pub fn qwen3next_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen3NextParams {
    graph_arch::Qwen3NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
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
    }
}

/// kimi-linear.cpp:43-165 — the KDA layers carry the per-stream conv kernels
/// + f_a/f_b/beta/a/dt/g_a/g_b/o_norm set; the MLA layers the (optional q
/// compression +) wkv_a_mqa + the split wk_b/wv_b or legacy wkv_b.
pub fn kimi_linear_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::KimiLinearModelWeights {
    graph_arch::KimiLinearModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::KimiLinearLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
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
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wkv_b: l.wkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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
            })
            .collect(),
    }
}

/// kimi-linear.cpp:4-32 + hparams — the MLA dims, the KDA geometry and the
/// recurrent state cell sizes. `attn` carries whatever cache geometry the
/// file's attention layers describe ([kv_lora|rope] x 1 for the split files,
/// [qk_head_dim x n_head] for the legacy wkv_b ones).
pub fn kimi_linear_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::KimiLinearParams {
    graph_arch::KimiLinearParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_qk_rope: hp.n_rot(first_attn_layer(hp, n_layer)) as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
    }
}

/// bailingmoe3.cpp:76-158 — the KDA set (single-stage f_a, the safe gate) +
/// the MLA layers (optional q compression + the output gate) + the dense/MoE
/// FFNs with the swiglu_clamp limits.
pub fn bailingmoe3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BailingMoe3ModelWeights {
    graph_arch::BailingMoe3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BailingMoe3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv_gate: l.wqkv_gate,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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
            })
            .collect(),
    }
}

/// bailingmoe3.cpp:6-44 + hparams — the safe-gate bound, the MLA dims, the
/// KDA geometry, the swiglu_clamp arrays and the MoE knobs.
pub fn bailingmoe3_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::BailingMoe3Params {
    graph_arch::BailingMoe3Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_lora_q: hp.n_lora_q as i64,
        n_embd_head_qk_rope: hp.n_rot(first_attn_layer(hp, n_layer)) as i64,
        rope_sections: hp.rope_sections,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        swiglu_clamp_exp: hp.swiglu_clamp_exp[..n_layer].to_vec(),
        swiglu_clamp_shexp: hp.swiglu_clamp_shexp[..n_layer].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// arch batch 10 (2026-10): the small-arch + EXP-op family
// ---------------------------------------------------------------------------

pub fn smallthinker_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::SmallthinkerModelWeights {
    graph_arch::SmallthinkerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::SmallthinkerLayerWeights {
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
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

pub fn llada_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::LladaMoeModelWeights {
    graph_arch::LladaMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LladaMoeLayerWeights {
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

pub fn minimax01_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Minimax01ModelWeights {
    graph_arch::Minimax01ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Minimax01LayerWeights {
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
                attn_norm_2: l.attn_norm_2,
                wqkv_la: l.wqkv,
                wg: l.wqkv_gate,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_exp_probs_b: l.ffn_exp_probs_b,
            })
            .collect(),
    }
}

pub fn minimax01_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Minimax01Params {
    graph_arch::Minimax01Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        f_residual_scale: hp.f_residual_scale,
        n_embd_head_la: hp.n_embd_head_la as i64,
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

pub fn graniteswitch_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteSwitchModelWeights {
    graph_arch::GraniteSwitchModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let sl = l.switch_lora.expect("graniteswitch switch_lora");
                graph_arch::GraniteSwitchLayerWeights {
                    attn_norm: l.attn_norm.unwrap(),
                    wqkv: l.wqkv.unwrap(),
                    wo: l.wo.unwrap(),
                    ffn_norm: l.ffn_norm.unwrap(),
                    ffn_gate: l.ffn_gate.unwrap(),
                    ffn_down: l.ffn_down.unwrap(),
                    ffn_up: l.ffn_up.unwrap(),
                    sl,
                }
            })
            .collect(),
        token_to_slot: m.graniteswitch_token_to_slot.clone(),
        token_to_substitute: m.graniteswitch_token_to_substitute.clone(),
    }
}

pub fn graniteswitch_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::GraniteSwitchParams {
    graph_arch::GraniteSwitchParams {
        attn,
        n_embd: hp.n_embd as i64,
        router_layer: hp.router_layer as usize,
        n_head: (0..n_layer).map(|il| hp.n_head(il) as i64).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il) as i64).collect(),
        has_rope: (0..=n_layer).map(|il| hp.has_rope(il)).collect(),
        n_ff: hp.n_ff(0) as i64,
        f_logit_scale: hp.f_logit_scale,
        f_residual_scale: hp.f_residual_scale,
        f_embedding_scale: hp.f_embedding_scale,
        f_attention_scale: hp.f_attention_scale,
        n_adapters: hp.graniteswitch_n_adapters as i64,
        router_gain: hp.graniteswitch_router_gain,
    }
}

// ---------------------------------------------------------------------------
// arch batch 11a (2026-10): apertus / grovemoe / qwen35moe / kimi-k3 /
// dots3note / minimax-m3 / qwen4exp — the same derivations the per-arch e2e
// tests use (crates/llama/tests/arch_batch11a_e2e.rs)
// ---------------------------------------------------------------------------

/// apertus.cpp:17-54 — per-head q/k RMS norms + the xIELU FFN.
pub fn apertus_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ApertusModelWeights {
    graph_arch::ApertusModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ApertusLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                rope_long: l.rope_long,
                rope_short: l.rope_short,
                rope_freqs: l.rope_freqs,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_k_norm_b: l.attn_k_norm_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

pub fn apertus_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::ApertusParams {
    graph_arch::ApertusParams {
        attn,
        xielu_alpha_n: hp.xielu_alpha_n[..n_layer].to_vec(),
        xielu_alpha_p: hp.xielu_alpha_p[..n_layer].to_vec(),
        xielu_beta: hp.xielu_beta[..n_layer].to_vec(),
        xielu_eps: hp.xielu_eps[..n_layer].to_vec(),
        f_attention_scale: hp.f_attention_scale,
        use_longrope_factors: hp.rope_scaling_type_train
            == llama::hparams::LlamaRopeScalingType::LONGROPE,
    }
}

/// grovemoe.cpp:16-61 — GQA + the dual (expert, chunk-expert) MoE.
pub fn grovemoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GrovemoeModelWeights {
    graph_arch::GrovemoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GrovemoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_chexps: l
                    .ffn_gate_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_chexps")),
                ffn_down_chexps: l
                    .ffn_down_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_chexps")),
                ffn_up_chexps: l
                    .ffn_up_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_chexps")),
            })
            .collect(),
    }
}

pub fn grovemoe_params(hp: &LlamaHparams, _n_layer: usize, attn: AttnParams) -> graph_arch::GrovemoeParams {
    graph_arch::GrovemoeParams {
        attn,
        n_embd: hp.n_embd as i64,
        expert_group_scale: hp.expert_group_scale,
        n_group_experts: hp.n_group_experts as i64,
        n_ff_chexp: hp.n_ff_chexp as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// qwen35moe.cpp:36-147 — the trunk GDN/attention layers + the MoE everywhere.
pub fn qwen35moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35MoeModelWeights {
    graph_arch::Qwen35MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen35MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
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
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

pub fn qwen35moe_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35MoeParams {
    graph_arch::Qwen35MoeParams {
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
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// kimi-k3.cpp:53-167 — KDA layers (per-stream convs + the full-rank gate) +
/// nope-MLA layers + the latent MoE + the residual-bank scores.
pub fn kimi_k3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::KimiK3ModelWeights {
    graph_arch::KimiK3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_res_score: m.output_res_score,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::KimiK3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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
    }
}

pub fn kimi_k3_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::KimiK3Params {
    let fa = first_attn_layer(hp, n_layer);
    graph_arch::KimiK3Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(fa) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_lora_q: hp.n_lora_q as i64,
        n_embd_head_qk_rope: hp.n_rot(fa) as i64,
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
    }
}

/// dots3note.cpp:48-146 — the trunk MLA/indexer/MoE tensors.
pub fn dots3note_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Dots3NoteModelWeights {
    graph_arch::Dots3NoteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Dots3NoteLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_q_a_norm: l
                    .attn_q_a_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_a_norm")),
                attn_kv_a_norm: l
                    .attn_kv_a_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_kv_a_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wq_a: l.wq_a.unwrap_or_else(|| panic!("layer {il}: wq_a")),
                wq_b: l.wq_b.unwrap_or_else(|| panic!("layer {il}: wq_b")),
                wkv_a_mqa: l.wkv_a_mqa.unwrap_or_else(|| panic!("layer {il}: wkv_a_mqa")),
                wk_b: l.wk_b.unwrap_or_else(|| panic!("layer {il}: wk_b")),
                wv_b: l.wv_b.unwrap_or_else(|| panic!("layer {il}: wv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wqkv_gate: l.wqkv_gate.unwrap_or_else(|| panic!("layer {il}: wqkv_gate")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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
    }
}

pub fn dots3note_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Dots3NoteParams {
    graph_arch::Dots3NoteParams {
        n_embd: hp.n_embd as i64,
        attn,
        n_head: (0..n_layer).map(|il| hp.n_head(il) as i32).collect(),
        n_rot: hp.n_rot(0) as i64,
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
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
        is_indexer_full: (0..n_layer).map(|il| hp.is_indexer_full(il)).collect(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
    }
}

/// minimax-m3.cpp:41-90 — GQA + per-head q/k norms + the dense/MoE split.
pub fn minimax_m3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MinimaxM3ModelWeights {
    graph_arch::MinimaxM3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MinimaxM3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
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
    }
}

pub fn minimax_m3_params(hp: &LlamaHparams, _n_layer: usize, attn: AttnParams) -> graph_arch::MinimaxM3Params {
    graph_arch::MinimaxM3Params {
        attn,
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
    }
}

/// qwen4exp.cpp:150-259 — the HC mixers + the GDN/gated-attention layers.
pub fn qwen4exp_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen4ExpModelWeights {
    graph_arch::Qwen4ExpModelWeights {
        tok_embd: m.tok_embd,
        hc_head_norm: m.hc_head_norm.unwrap_or_else(|| panic!("hc_head_norm")),
        hc_head_down: m.hc_head_down.unwrap_or_else(|| panic!("hc_head_down")),
        hc_head_up: m.hc_head_up.unwrap_or_else(|| panic!("hc_head_up")),
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen4ExpLayerWeights {
                hc_attn_norm: l
                    .hc_attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_norm")),
                hc_attn_down: l
                    .hc_attn_down
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_down")),
                hc_attn_up: l.hc_attn_up.unwrap_or_else(|| panic!("layer {il}: hc_attn_up")),
                hc_attn_inject: l
                    .hc_attn_inject
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_inject")),
                hc_ffn_norm: l
                    .hc_ffn_norm
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_norm")),
                hc_ffn_down: l
                    .hc_ffn_down
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_down")),
                hc_ffn_up: l.hc_ffn_up.unwrap_or_else(|| panic!("layer {il}: hc_ffn_up")),
                hc_ffn_inject: l
                    .hc_ffn_inject
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_inject")),
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
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

pub fn qwen4exp_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen4ExpParams {
    graph_arch::Qwen4ExpParams {
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
        compress_ratios: hp.dsv4_compress_ratios[..n_layer].to_vec(),
        is_ple: (0..n_layer).map(|il| hp.is_ple(il)).collect(),
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
    }
}

// ---------------------------------------------------------------------------
// batch 19 — glm5-next (the hybrid_idx family): the same bundle llama-cli's
// glm5_weights/glm5_params build
// ---------------------------------------------------------------------------

pub fn glm5_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm5NextModelWeights {
    graph_arch::Glm5NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Glm5NextLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                hc_attn_fn: l.hc_attn_fn,
                hc_attn_base: l.hc_attn_base,
                hc_attn_scale: l.hc_attn_scale,
                hc_ffn_fn: l.hc_ffn_fn,
                hc_ffn_base: l.hc_ffn_base,
                hc_ffn_scale: l.hc_ffn_scale,
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wqkv: l.wqkv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wo: l.wo,
                attn_q_a_norm: l.attn_q_a_norm,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wq_a: l.wq_a,
                wq_b: l.wq_b,
                wkv_a_mqa: l.wkv_a_mqa,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                indexer_kpool_gate: l.indexer_kpool_gate,
                indexer_kpool_ape: l.indexer_kpool_ape,
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

pub fn glm5_params(
    hp: &LlamaHparams,
    n_trunk: usize,
    attn: AttnParams,
) -> graph_arch::Glm5NextParams {
    graph_arch::Glm5NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(0) as i64,
        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_lora_q: hp.n_lora_q as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_rot: hp.n_rot(0) as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_expert_shared: hp.n_expert_shared as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        indexer_kpool_select_tail: hp.indexer_kpool_select_tail,
        is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
        hc: hp.dsv4_hc_mult as i64,
        hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters,
        hc_eps: hp.dsv4_hc_eps,
        f_norm_eps: hp.f_norm_eps,
    }
}

// ---------------------------------------------------------------------------
// arch batch 11b (2026-10) — the long-tail queue, second half: arcee / jais2
// / talkie / nanbeige / dream / rnd1 (eurobert is encoder-only: the server
// reaches it through EncoderContext, see the module docs above)
// ---------------------------------------------------------------------------

/// arcee.cpp:27-41 — the llama tensor set + the per-layer rope_freqs factors
pub fn arcee_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ArceeModelWeights {
    graph_arch::ArceeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
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

/// jais2.cpp:26-45 — LN pairs + biases on every norm and both MLP matrices
pub fn jais2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Jais2ModelWeights {
    graph_arch::Jais2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
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

/// talkie.cpp:19-33 — no norms except the [1, n_head] q gain, the layer_out_scale
pub fn talkie_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::TalkieModelWeights {
    graph_arch::TalkieModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
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

/// nanbeige.cpp:49-73 — the physical stack plus the aliased loop slots (the
/// loader already mirrored layers[i + j*n_phys] = layers[i])
pub fn nanbeige_weights(m: &LlamaModel, n_all: usize) -> graph_arch::NanbeigeModelWeights {
    graph_arch::NanbeigeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_all)
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

/// dream.cpp:32-45 — qwen2's tensor set (fused-or-separate qkv, SwiGLU gate)
pub fn dream_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DreamModelWeights {
    graph_arch::DreamModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
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

/// rnd1.cpp:29-57 — qwen3moe's tensor set (per-head q/k norms + the softmax
/// MoE experts at the dense n_ff fallback width)
pub fn rnd1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Rnd1ModelWeights {
    graph_arch::Rnd1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
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

// ---------------------------------------------------------------------------
// arch batch 12 (2026-10): the final long-tail queue — hrm-text / laguna /
// maple (crates/llama/tests/arch_batch12_e2e.rs)
// ---------------------------------------------------------------------------

/// hrm-text.cpp:34-86 — the two physical stacks' tensor set (qkv + gate +
/// wo + SwiGLU FFN), aliased onto the n_slot cache slots by the loader
pub fn hrm_text_weights(m: &LlamaModel, n_slot: usize) -> graph_arch::HrmTextModelWeights {
    graph_arch::HrmTextModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        hrm_z_l_init: m.hrm_z_l_init.expect("hrm.z_l_init"),
        layers: m
            .layers
            .iter()
            .take(n_slot)
            .map(|l| graph_arch::HrmTextLayerWeights {
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
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// laguna.cpp:65-146 — per-layer head counts, the two gate widths, qk norms,
/// the dense-lead / MoE split and the always-on shared expert
pub fn laguna_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::LagunaModelWeights {
    graph_arch::LagunaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LagunaLayerWeights {
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
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_up: l.ffn_up,
                ffn_down: l.ffn_down,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// maple.cpp:24-59 — qkv + per-head qk norms + the softmax-MoE expert stack
pub fn maple_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MapleModelWeights {
    graph_arch::MapleModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MapleLayerWeights {
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

// ---------------------------------------------------------------------------
// arch batch 14 (2026-10): the RWKV family + gemma3n
// ---------------------------------------------------------------------------

/// rwkv6 / rwkv6qwen2 tensor bundle (rwkv6.cpp:29-86 / rwkv6qwen2.cpp:29-77).
pub fn rwkv6_weights(m: &LlamaModel, n_trunk: usize, qwen2: bool) -> graph_arch::Rwkv6ModelWeights {
    graph_arch::Rwkv6ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if qwen2 { None } else { m.token_embd_norm },
        tok_norm_b: if qwen2 { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Rwkv6LayerWeights {
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
                channel_mix_receptance: if qwen2 { None } else { l.channel_mix_receptance },
                ffn_norm: if qwen2 { l.ffn_norm } else { None },
                ffn_gate: if qwen2 { l.ffn_gate } else { None },
                ffn_down: if qwen2 { l.ffn_down } else { None },
                ffn_up: if qwen2 { l.ffn_up } else { None },
            })
            .collect(),
    }
}

pub fn rwkv6_params(hp: &LlamaHparams) -> graph_arch::Rwkv6Params {
    graph_arch::Rwkv6Params {
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

/// rwkv7 / arwkv7 tensor bundle (rwkv7.cpp:49-116 / arwkv7.cpp:49-112).
pub fn rwkv7_weights(m: &LlamaModel, n_trunk: usize, arwkv: bool) -> graph_arch::Rwkv7ModelWeights {
    graph_arch::Rwkv7ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if arwkv { None } else { m.token_embd_norm },
        tok_norm_b: if arwkv { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: if arwkv { None } else { m.output_norm_b },
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Rwkv7LayerWeights {
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

pub fn rwkv7_params(hp: &LlamaHparams) -> graph_arch::Rwkv7Params {
    graph_arch::Rwkv7Params {
        n_embd: hp.n_embd as i64,
        wkv_head_size: hp.wkv_head_size as i64,
        token_shift_count: hp.token_shift_count as i64,
        norm_eps: hp.f_norm_eps,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

/// gemma3n tensor bundle (gemma3n.cpp:21-76).
pub fn gemma3n_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma3nModelWeights {
    graph_arch::Gemma3nModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        output_norm: m.output_norm,
        altup_proj: m.altup_proj.unwrap(),
        altup_unembd_proj: m.altup_unembd_proj.unwrap(),
        per_layer_tok_embd: m.per_layer_tok_embd.unwrap(),
        per_layer_model_proj: m.per_layer_model_proj.unwrap(),
        per_layer_proj_norm: m.per_layer_proj_norm.unwrap(),
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Gemma3nLayerWeights {
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

// ---------------------------------------------------------------------------
// arch batch 13 (2026-09): the P0 standard-attention queue — verbatim copies
// of llama-cli's wiring (the server and the CLI must drive the identical
// graph).
// ---------------------------------------------------------------------------

/// cohere2.cpp:21-44
pub fn cohere2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2ModelWeights {
    graph_arch::Cohere2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Cohere2LayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// chatglm.cpp:25-52
pub fn chatglm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ChatglmModelWeights {
    graph_arch::ChatglmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::ChatglmLayerWeights {
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
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
            })
            .collect(),
    }
}

/// bitnet.cpp:12-45
pub fn bitnet_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BitnetModelWeights {
    graph_arch::BitnetModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::BitnetLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_sub_norm: l.attn_sub_norm.unwrap(),
                wq: l.wq.unwrap(),
                wq_s: l.wq_s,
                wk: l.wk.unwrap(),
                wk_s: l.wk_s,
                wv: l.wv.unwrap(),
                wv_s: l.wv_s,
                wo: l.wo.unwrap(),
                wo_s: l.wo_s,
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_sub_norm: l.ffn_sub_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_gate_s: l.ffn_gate_s,
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_s: l.ffn_down_s,
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_s: l.ffn_up_s,
            })
            .collect(),
    }
}

/// dbrx.cpp:13-41
pub fn dbrx_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DbrxModelWeights {
    graph_arch::DbrxModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::DbrxLayerWeights {
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                attn_norm: l.attn_norm.unwrap(),
                attn_out_norm: l.attn_out_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

/// mistral3.cpp:29-87 — the rope factors resolved like the reference's
/// `get_rope_factors` (llama-model.cpp:2259-2272, the deci/phimoe pattern)
pub fn mistral3_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
) -> graph_arch::Mistral3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Mistral3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let rope_factors = if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                };
                graph_arch::Mistral3LayerWeights {
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
                    rope_factors,
                    ffn_gate: l.ffn_gate,
                    ffn_gate_b: l.ffn_gate_b,
                    ffn_down: l.ffn_down,
                    ffn_down_b: l.ffn_down_b,
                    ffn_up: l.ffn_up,
                    ffn_up_b: l.ffn_up_b,
                    ffn_gate_inp: l.ffn_gate_inp,
                    ffn_gate_exps: l.ffn_gate_exps,
                    ffn_down_exps: l.ffn_down_exps,
                    ffn_up_exps: l.ffn_up_exps,
                    ffn_gate_shexp: l.ffn_gate_shexp,
                    ffn_down_shexp: l.ffn_down_shexp,
                    ffn_up_shexp: l.ffn_up_shexp,
                }
            })
            .collect(),
    }
}

/// minicpm3.cpp:14-57 — same get_rope_factors resolution
pub fn minicpm3_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
) -> graph_arch::Minicpm3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Minicpm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let rope_factors = if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                };
                graph_arch::Minicpm3LayerWeights {
                    attn_norm: l.attn_norm.unwrap(),
                    attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                    attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                    wq_a: l.wq_a.unwrap(),
                    wq_b: l.wq_b.unwrap(),
                    wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                    wkv_b: l.wkv_b.unwrap(),
                    wo: l.wo.unwrap(),
                    ffn_norm: l.ffn_norm.unwrap(),
                    ffn_gate: l.ffn_gate.unwrap(),
                    ffn_down: l.ffn_down.unwrap(),
                    ffn_up: l.ffn_up.unwrap(),
                    rope_factors,
                }
            })
            .collect(),
    }
}

/// glm4.cpp:15-62 (trunk layers only — the port loads trunk-only files)
pub fn glm4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4ModelWeights {
    graph_arch::Glm4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Glm4LayerWeights {
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
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

/// exaone4.cpp:24-71 (trunk layers only)
pub fn exaone4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Exaone4ModelWeights {
    graph_arch::Exaone4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Exaone4LayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                rope_freqs: l.rope_freqs,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

/// llama4.cpp:44-94
pub fn llama4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Llama4ModelWeights {
    graph_arch::Llama4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Llama4LayerWeights {
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
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
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

/// qwen2vl.cpp:8-36
pub fn qwen2vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen2VlModelWeights {
    graph_arch::Qwen2VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen2VlLayerWeights {
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
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// qwen3vl.cpp:16-54 / qwen3vlmoe.cpp:16-58 (one bundle for both)
pub fn qwen3vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3VlModelWeights {
    graph_arch::Qwen3VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen3VlLayerWeights {
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
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
            })
            .collect(),
    }
}

/// glm-dsa.cpp:74-187 (trunk layers only)
pub fn glm_dsa_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GlmDsaModelWeights {
    graph_arch::GlmDsaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::GlmDsaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wq_a: l.wq_a.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                wk_b: l.wk_b.unwrap(),
                wv_b: l.wv_b.unwrap(),
                wo: l.wo.unwrap(),
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
    }
}

/// llama-embed's weight bundle (llama.cpp:36-61 — the LLAMA tensor set)
pub fn llama_embed_weights(m: &LlamaModel) -> graph_arch::LlamaModelWeights {
    let layers = m
        .layers
        .iter()
        .map(|l| graph_arch::LlamaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq.unwrap(),
            wk: l.wk.unwrap(),
            wv: l.wv.unwrap(),
            wo: l.wo.unwrap(),
            wq_b: l.wq_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wo_b: l.wo_b,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate.unwrap(),
            ffn_down: l.ffn_down.unwrap(),
            ffn_up: l.ffn_up.unwrap(),
            ffn_gate_b: l.ffn_gate_b,
            ffn_down_b: l.ffn_down_b,
            ffn_up_b: l.ffn_up_b,
        })
        .collect();
    graph_arch::LlamaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers,
    }
}
