//! model.rs — weight loading (port of `llama-model-loader.cpp` tensors path +
//! `src/models/*.cpp` load_arch_tensors + `llama-model.cpp` validation).
//!
//! Reference: llama.cpp bd4f514db1.
//! Mapping:
//!   llama_model_loader::create_tensor / check_tensor_dims /
//!   done_getting_tensors                -> `ModelLoader::{create_tensor,check_tensor_dims,done_getting_tensors}`
//!   llama_model_base::create_tensor_qkv -> `create_tensor_qkv`
//!   models/{qwen2,llama,qwen3,gemma2,gemma3,phi3}.cpp load_arch_tensors
//!                                        -> `load_arch_tensors` (match below)
//!   models/{openai-moe,gemma4,lfm2moe,granite-hybrid,qwen35}.cpp load_arch_tensors
//!                                        -> `load_arch_tensors` (agent P addition)
//!   llama_layer / model-level tensor members -> `LayerTensors` / `LlamaModel`
//!
//! Deviations (documented):
//!   * ggml tensor *type* always comes from the GGUF tensor table (exactly like
//!     the meta tensors `gguf_init_from_file` produces in C++) — C++ has no
//!     "expected type", so no type conversion ever happens here. Norm weights
//!     keep their file type (F32 in every model checked); if a model ships F16
//!     norm weights the compute layer must cope (rms_norm needs F32 src0).
//!   * mmap: every created tensor gets `set_external_storage` pointing into the
//!     same `Arc<Mmap>` at file offset `gguf.data_offset + ti.offset` — the
//!     `use_mmap` path of `llama_model_loader::load_all_data`.
//!   * TENSOR_SKIP / TENSOR_SKIP_IF_VIRTUAL / TENSOR_READ_LAZY /
//!     TENSOR_ALLOW_RESHAPE and the unused-tensor (GGML_OP_NONE) skip path are
//!     not ported: no virtual/remote files, no lazy reads, and none of the
//!     supported archs requests an op==NONE tensor (only MASKED_EMBD_* and
//!     DEC_CROSS_ATTN_REL_B carry GGML_OP_NONE in the C table).
//!     Consequences for qwen35: the port has no `load_mtp` flag, so MTP blocks
//!     are always loaded — the file's nextn tensors are consumed instead of
//!     being skipped (equivalent to C++ `load_mtp = true`); the default C++
//!     `load_mtp = false` would leave them uncreated.
//!   * `done_getting_tensors` never runs in `partial` mode here.
//!   * The create_tensor input/output-vs-repeating layer sanity check keys off
//!     whether the tensor-name template contains a `%d` block slot: C++ reads
//!     `tn.bid`, and for ROPE_FACTORS_LONG/SHORT (REPEATING but nameless in
//!     `blk.` space) reference llama.cpp passes bid >= 0 without tripping the
//!     GGML_ABORT — the effective condition is the template, not the enum.
//!   * `n_created` counts only non-duplicated creations, exactly like C++
//!     (`create_tensor` bumps `size_data` for TENSOR_DUPLICATED instead); the
//!     gemma4 output→token_embd tie is the case where a dup request really
//!     creates a second tensor before `token_embd` itself exists.
//!   * The generic optional `.scale` pass of `llama_model::load_tensors` is
//!     ported only for the tensor kinds carried by the archs here
//!     (ffn_gate_inp / ffn_gate_exps / ffn_down_exps / ffn_up_exps) — see
//!     `create_optional_scale_tensors`.

use std::collections::HashMap;
use std::sync::Arc;

use ggml::tensor::{Context, TensorId};
use ggml::{Gguf, GgufType, TensorInfo, Value};
use memmap2::Mmap;

use crate::arch::{kv_name, tensor_name_suffix, LlmArch, LlmKv, LlmTensor, LlmTensorLayer};
use crate::hparams::{LlamaExpertGatingFuncType, LlamaHparams, LlamaRopeScalingType, LlamaSwaType};
use crate::meta::load_hparams;

// ---------------------------------------------------------------------------
// flags — llama-model-loader.h `llama_model_tensor_flags`
// ---------------------------------------------------------------------------

pub const TENSOR_NOT_REQUIRED: u32 = 1 << 0;
pub const TENSOR_DUPLICATED: u32 = 1 << 1;
/// arch batch 7 (deepseek4): TENSOR_ALLOW_RESHAPE — check total elements only
/// and create the context tensor at the *requested* shape (the C's t_meta
/// re-ne, llama-model-loader.cpp:896-903/:1348-1360). deepseek4's
/// `attn_output_a` is stored {n_head*n_embd_head/o_groups, o_lora_rank*o_groups}
/// and reshaped to 3-D (deepseek4.cpp:120-122).
pub const TENSOR_ALLOW_RESHAPE: u32 = 1 << 2;

// ---------------------------------------------------------------------------
// LayerTensors — the llama_layer tensor members relevant to the supported archs
// ---------------------------------------------------------------------------

/// `llama_layer::nextn` (speculative/next-token-prediction block members).
#[derive(Debug, Clone, Copy, Default)]
pub struct NextnTensors {
    pub eh_proj: Option<TensorId>,
    pub enorm: Option<TensorId>,
    pub hnorm: Option<TensorId>,
    pub embed_tokens: Option<TensorId>,
    pub shared_head_head: Option<TensorId>,
    pub shared_head_norm: Option<TensorId>,
    /// qwen4exp (a7b94df2c llama-model.h): the MTP block collapses its
    /// hyper-connection streams with its own mixer
    pub hc_head_norm: Option<TensorId>,
    pub hc_head_down: Option<TensorId>,
    pub hc_head_up: Option<TensorId>,
}

/// Per-layer tensor handles (1:1 with the `llama_layer` fields used by
/// qwen2/llama/qwen3/gemma2/gemma3/phi3/gemma4/gpt-oss/lfm2moe/
/// granitehybrid/qwen35; each field is optional exactly where the C++ member
/// can stay nullptr).
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerTensors {
    // normalization
    pub attn_norm: Option<TensorId>,
    pub attn_norm_b: Option<TensorId>,
    pub attn_q_norm: Option<TensorId>,
    pub attn_q_norm_b: Option<TensorId>,
    pub attn_k_norm: Option<TensorId>,
    pub attn_k_norm_b: Option<TensorId>,
    pub attn_post_norm: Option<TensorId>,
    /// bitnet's post-attention sub-norm (`attn_sub_norm`, bitnet.cpp:24)
    pub attn_sub_norm: Option<TensorId>,
    /// falcon-40B's second attention norm (`attn_norm_2`, falcon.cpp:35-36)
    pub attn_norm_2: Option<TensorId>,
    pub attn_norm_2_b: Option<TensorId>,
    pub ffn_norm: Option<TensorId>,
    pub ffn_norm_b: Option<TensorId>,
    pub ffn_post_norm: Option<TensorId>,
    /// bitnet's mid-FFN sub-norm (`ffn_sub_norm`, bitnet.cpp:36)
    pub ffn_sub_norm: Option<TensorId>,

    // attention
    pub wq: Option<TensorId>,
    pub wk: Option<TensorId>,
    pub wv: Option<TensorId>,
    pub wo: Option<TensorId>,
    pub wqkv: Option<TensorId>,
    pub wq_b: Option<TensorId>,
    pub wk_b: Option<TensorId>,
    pub wv_b: Option<TensorId>,
    pub wqkv_b: Option<TensorId>,
    pub wo_b: Option<TensorId>,

    // bitnet b1.58 weight-scale tensors (bitnet.cpp:27-43 — the `.scale`
    // companions of the plain q/k/v/o/gate/down/up weights)
    pub wq_s: Option<TensorId>,
    pub wk_s: Option<TensorId>,
    pub wv_s: Option<TensorId>,
    pub wo_s: Option<TensorId>,
    pub ffn_gate_s: Option<TensorId>,
    pub ffn_down_s: Option<TensorId>,
    pub ffn_up_s: Option<TensorId>,

    /// `attn_sinks` (gpt-oss) — attention sink logits, one per head
    pub attn_sinks: Option<TensorId>,
    /// `wqkv_gate` (qwen35 gated delta net: `attn_gate`, the z gate)
    pub wqkv_gate: Option<TensorId>,

    // deepseek2 MLA projections (deepseek2.cpp:100-120). NOTE: like the C
    // `llama_layer`, `wq_b`/`wk_b`/`wv_b` above carry the deepseek meanings
    // ATTN_Q_B / ATTN_K_B / ATTN_V_B (the split MLA B projections) when the
    // arch uses them — the qkv-bias meaning belongs to the create_tensor_qkv
    // archs only.
    /// `attn_q_a` — q_lora down projection {n_embd, q_lora_rank}
    pub wq_a: Option<TensorId>,
    /// `attn_q_a_norm` — {q_lora_rank}
    pub attn_q_a_norm: Option<TensorId>,
    /// `attn_kv_a_norm` — {kv_lora_rank}
    pub attn_kv_a_norm: Option<TensorId>,
    /// `attn_kv_a_mqa` — {n_embd, kv_lora_rank + qk_rope_head_dim}
    pub wkv_a_mqa: Option<TensorId>,
    /// legacy unsplit `attn_kv_b` — {kv_lora_rank, n_head*(qk_nope+v_mla)}
    /// (only old non-MLA deepseek2 GGUFs)
    pub wkv_b: Option<TensorId>,
    // deepseek32's DSA lightning-indexer tensors (deepseek32.cpp:109-114)
    pub indexer_k_norm: Option<TensorId>,
    pub indexer_k_norm_b: Option<TensorId>,
    pub indexer_proj: Option<TensorId>,
    pub indexer_attn_k: Option<TensorId>,
    pub indexer_attn_q_b: Option<TensorId>,
    // glm5-next's k-pool indexer (glm5-next.cpp:148-154) — the gate/ape read
    // the SAME file names as dflash's compressor tensors
    // (LLM_TENSOR_INDEXER_KPOOL_GATE/APE alias
    // "blk.%d.indexer_compressor_gate"/"_ape", llama-arch.cpp:698-699)
    pub indexer_kpool_gate: Option<TensorId>,
    pub indexer_kpool_ape: Option<TensorId>,

    // arch batch 7 (deepseek4): hyper-connection mixers (deepseek4.cpp:125-130)
    // and the per-ratio compressors (:136-150). All Option: only the layers
    // whose dsv4_compress_ratios[il] matches carry the compressor tensors.
    pub hc_attn_fn: Option<TensorId>,
    pub hc_attn_base: Option<TensorId>,
    pub hc_attn_scale: Option<TensorId>,
    pub hc_ffn_fn: Option<TensorId>,
    pub hc_ffn_base: Option<TensorId>,
    pub hc_ffn_scale: Option<TensorId>,
    /// `attn_output_a` — reshaped at load to {n_head*n_embd_head/o_groups,
    /// o_lora_rank, o_groups} (TENSOR_ALLOW_RESHAPE, deepseek4.cpp:122)
    pub wo_a: Option<TensorId>,
    /// `attn_output_b` (LLM_TENSOR_ATTN_OUT_B "weight") — the o_lora output
    /// projection {o_groups*o_lora_rank, n_embd} (deepseek4.cpp:123). Distinct
    /// field from the attn-output *bias* `wo_b` above (the C reuses the
    /// `wo_b` member name for both).
    pub wo_b_dsv4: Option<TensorId>,
    // (`attn_sinks` above carries deepseek4's sinks too — same
    // LLM_TENSOR_ATTN_SINKS name, deepseek4.cpp:114)
    /// `attn_compressor_kv` — {n_embd, coff*n_embd_head} (:136/:147)
    pub attn_comp_wkv: Option<TensorId>,
    /// `attn_compressor_gate` (:137/:148)
    pub attn_comp_wgate: Option<TensorId>,
    /// `attn_compressor_ape` — absolute-position embedding {coff*n_embd_head,
    /// ratio} (:138/:149)
    pub attn_comp_ape: Option<TensorId>,
    /// `attn_compressor_norm` — {n_embd_head} (:139)
    pub attn_comp_norm: Option<TensorId>,
    /// the indexer twin of the compressor four (:147-150, ratio-4 layers only)
    pub indexer_comp_wkv: Option<TensorId>,
    pub indexer_comp_wgate: Option<TensorId>,
    pub indexer_comp_ape: Option<TensorId>,
    pub indexer_comp_norm: Option<TensorId>,
    /// `ffn_gate_tid2eid` — hash-layer routing table {n_expert_used, n_vocab}
    /// (:158), only the first dsv4_hash_layer_count layers
    pub ffn_gate_tid2eid: Option<TensorId>,
    /// `exp_probs_b.bias_vl` — vision-only routing bias (:163, NOT_REQUIRED;
    /// no vision file in the port)
    pub ffn_exp_probs_b_vl: Option<TensorId>,

    // rope factors
    pub rope_freqs: Option<TensorId>,
    pub rope_long: Option<TensorId>,
    pub rope_short: Option<TensorId>,

    /// cogvlm's vision-expert twins (cogvlm.cpp:33-34/:43-45 — the text
    /// tensors above have vis_* counterparts the embd batches run)
    pub visexp_attn_wqkv: Option<TensorId>,
    pub visexp_attn_wo: Option<TensorId>,
    pub visexp_ffn_gate: Option<TensorId>,
    pub visexp_ffn_down: Option<TensorId>,
    pub visexp_ffn_up: Option<TensorId>,

    // feed-forward (dense)
    pub ffn_gate: Option<TensorId>,
    pub ffn_down: Option<TensorId>,
    pub ffn_up: Option<TensorId>,
    pub ffn_gate_b: Option<TensorId>,
    pub ffn_down_b: Option<TensorId>,
    pub ffn_up_b: Option<TensorId>,
    /// `ffn.act.scales` — mpt's AWQ ScaleActivation undo (mpt.cpp:52)
    pub ffn_act: Option<TensorId>,

    // feed-forward (MoE)
    pub ffn_gate_inp: Option<TensorId>,
    pub ffn_gate_exps: Option<TensorId>,
    pub ffn_down_exps: Option<TensorId>,
    pub ffn_up_exps: Option<TensorId>,
    pub ffn_gate_shexp: Option<TensorId>,
    pub ffn_down_shexp: Option<TensorId>,
    pub ffn_up_shexp: Option<TensorId>,
    /// qwen2moe's shared-expert router weight (`ffn_gate_inp_shexp`,
    /// qwen2moe.cpp:54 — a 1-D [n_embd] whose output feeds silu(x)/x)
    pub ffn_gate_inp_shexp: Option<TensorId>,
    /// nemotron-h's optional latent projections (`ffn_latent_{down,up}`,
    /// nemotron-h.cpp:122-123)
    pub ffn_latent_down: Option<TensorId>,
    pub ffn_latent_up: Option<TensorId>,
    /// arctic's second FFN norm for the expert branch (`ffn_norm_exps`,
    /// arctic.cpp:45 — norm of the *pre-attention* residual)
    pub ffn_norm_exps: Option<TensorId>,

    /// combined gate+up expert tensor (gemma4 `ffn_gate_up_exps`)
    pub ffn_gate_up_exps: Option<TensorId>,
    /// router probability bias (lfm2moe `exp_probs_b`)
    pub ffn_exp_probs_b: Option<TensorId>,

    // MoE expert biases (gpt-oss)
    pub ffn_gate_inp_b: Option<TensorId>,
    pub ffn_gate_exps_b: Option<TensorId>,
    pub ffn_down_exps_b: Option<TensorId>,
    pub ffn_up_exps_b: Option<TensorId>,

    // optional per-tensor / per-expert `.scale` tensors (the generic
    // `llama_model::load_tensors` pass; only the kinds the ported archs carry
    // are modeled — see `create_optional_scale_tensors`)
    pub ffn_gate_inp_s: Option<TensorId>,
    pub ffn_gate_exps_s: Option<TensorId>,
    pub ffn_down_exps_s: Option<TensorId>,
    pub ffn_up_exps_s: Option<TensorId>,

    // second MoE branch norms (gemma4: pre/post norms + router scale)
    pub ffn_pre_norm_2: Option<TensorId>,
    pub ffn_post_norm_1: Option<TensorId>,
    pub ffn_post_norm_2: Option<TensorId>,
    /// per-layer output scale (gemma4 `layer_output_scale`)
    pub out_scale: Option<TensorId>,

    // encoder-only archs (bert / t5encoder)
    /// bert post-attention layer norm (`attn_output_norm`, bert.cpp:51-52)
    pub attn_out_norm: Option<TensorId>,
    pub attn_out_norm_b: Option<TensorId>,
    /// bert final per-layer norm (`layer_output_norm`, bert.cpp:59-60)
    pub layer_out_norm: Option<TensorId>,
    pub layer_out_norm_b: Option<TensorId>,
    /// t5 encoder members (t5encoder.cpp:27-38 / t5.cpp:66-77)
    pub enc_attn_norm: Option<TensorId>,
    pub enc_attn_rel_b: Option<TensorId>,
    pub enc_wq: Option<TensorId>,
    pub enc_wk: Option<TensorId>,
    pub enc_wv: Option<TensorId>,
    pub enc_wo: Option<TensorId>,
    pub enc_ffn_norm: Option<TensorId>,
    pub enc_ffn_gate: Option<TensorId>,
    pub enc_ffn_down: Option<TensorId>,
    pub enc_ffn_up: Option<TensorId>,
    // t5.cpp:60-107 — the decoder half (dec.blk.%d.*)
    pub dec_attn_norm: Option<TensorId>,
    pub dec_attn_rel_b: Option<TensorId>,
    pub dec_wq: Option<TensorId>,
    pub dec_wk: Option<TensorId>,
    pub dec_wv: Option<TensorId>,
    pub dec_wo: Option<TensorId>,
    pub dec_attn_norm_cross: Option<TensorId>,
    pub dec_wq_cross: Option<TensorId>,
    pub dec_wk_cross: Option<TensorId>,
    pub dec_wv_cross: Option<TensorId>,
    pub dec_wo_cross: Option<TensorId>,
    pub dec_ffn_norm: Option<TensorId>,
    pub dec_ffn_gate: Option<TensorId>,
    pub dec_ffn_down: Option<TensorId>,
    pub dec_ffn_up: Option<TensorId>,

    // shortconv mixer (lfm2 / lfm2moe)
    pub shortconv_conv: Option<TensorId>,
    pub shortconv_in_proj: Option<TensorId>,
    pub shortconv_out_proj: Option<TensorId>,

    // SSM / linear attention (granitehybrid mamba2, qwen35 gated delta net)
    pub ssm_in: Option<TensorId>,
    pub ssm_conv1d: Option<TensorId>,
    pub ssm_conv1d_b: Option<TensorId>,
    pub ssm_dt_b: Option<TensorId>,
    pub ssm_a: Option<TensorId>,
    pub ssm_d: Option<TensorId>,
    pub ssm_norm: Option<TensorId>,
    pub ssm_out: Option<TensorId>,
    pub ssm_alpha: Option<TensorId>,
    pub ssm_beta: Option<TensorId>,
    // arch batch 9 (the KDA family, kimi-linear / bailingmoe3): the
    // per-stream causal-conv kernels, the KDA decay / output-gate projection
    // stages, and qwen3next's packed beta|alpha router-side tensor
    pub ssm_q_conv: Option<TensorId>,
    pub ssm_k_conv: Option<TensorId>,
    pub ssm_v_conv: Option<TensorId>,
    pub ssm_f_a: Option<TensorId>,
    pub ssm_f_b: Option<TensorId>,
    pub ssm_g_a: Option<TensorId>,
    pub ssm_g_b: Option<TensorId>,
    pub ssm_beta_alpha: Option<TensorId>,
    // mamba1 mixer (mamba / jamba): the x_proj, dt_proj weight and the
    // FalconMamba/Jamba dt/B/C RMS-norm trio (mamba.cpp:83-97 / jamba.cpp:82-91)
    pub ssm_x: Option<TensorId>,
    pub ssm_dt: Option<TensorId>,
    pub ssm_dt_norm: Option<TensorId>,
    pub ssm_b_norm: Option<TensorId>,
    pub ssm_c_norm: Option<TensorId>,

    // per-layer embeddings (gemma4)
    pub per_layer_inp_gate: Option<TensorId>,
    pub per_layer_proj: Option<TensorId>,
    pub per_layer_post_norm: Option<TensorId>,

    // nextn / MTP block (qwen35)
    pub nextn: NextnTensors,

    /// arch batch 10 (graniteswitch): the stacked per-slot switch-LoRA deltas
    /// of the layer (granite-switch.cpp:113-130 — `switch_lora` in C). Slot 0
    /// is the base/zero delta; `mul_mat_id` selects the adapter per token.
    pub switch_lora: Option<SwitchLoraTensors>,

    // ------------------------------------------------------------------------
    // arch batch 11a (2026-10) — the long-tail queue, first half
    // ------------------------------------------------------------------------
    /// kimi-k3 (kimi-k3.cpp:73-76): the cross-layer residual attention's
    /// per-layer score vectors (attn_res_block_size > 0)
    pub attn_res_score: Option<TensorId>,
    pub ffn_res_score: Option<TensorId>,

    /// kimi-k3 (:103): the single full-rank output gate [n_embd, d_inner]
    pub ssm_g: Option<TensorId>,

    /// kimi-k3 (:154-158): the latent-MoE bridge (n_expert_latent > 0)
    pub ffn_routed_down: Option<TensorId>,
    pub ffn_routed_up: Option<TensorId>,
    pub ffn_routed_norm: Option<TensorId>,

    /// grovemoe (grovemoe.cpp:57-59): the chunk-expert MoE tensors
    pub ffn_gate_chexps: Option<TensorId>,
    pub ffn_down_chexps: Option<TensorId>,
    pub ffn_up_chexps: Option<TensorId>,

    /// qwen4exp (qwen4exp.cpp:207-214): the two HC modules per layer
    pub hc_attn_norm: Option<TensorId>,
    pub hc_attn_down: Option<TensorId>,
    pub hc_attn_up: Option<TensorId>,
    pub hc_attn_inject: Option<TensorId>,
    pub hc_ffn_norm: Option<TensorId>,
    pub hc_ffn_down: Option<TensorId>,
    pub hc_ffn_up: Option<TensorId>,
    pub hc_ffn_inject: Option<TensorId>,

    /// minimax-m3 / qwen4exp (minimax-m3.cpp:84-87, qwen4exp.cpp:224-228):
    /// the per-token indexer projections/norms (INDEXER_Q_PROJ family)
    pub index_q_proj: Option<TensorId>,
    pub index_k_proj: Option<TensorId>,
    pub index_q_norm: Option<TensorId>,
    /// batch 19 (qwen4exp PLE, qwen4exp.cpp:244-250) — the per-layer n-gram
    /// module of the PLE layer
    pub ple_key: Option<TensorId>,
    pub ple_value: Option<TensorId>,
    pub ple_norm_key: Option<TensorId>,
    pub ple_norm_query: Option<TensorId>,
    pub ple_norm_conv: Option<TensorId>,
    pub ple_conv1d: Option<TensorId>,
    pub index_k_norm: Option<TensorId>,

    // ------------------------------------------------------------------------
    // arch batch 14 (2026-10) — the RWKV family + gemma3n
    // ------------------------------------------------------------------------
    /// RWKV time mix — one member set shared by rwkv6/rwkv6qwen2/rwkv7/
    /// arwkv7 exactly like the C's single `llama_layer` (rwkv6.cpp:55-78 /
    /// rwkv7.cpp:76-110). `w1`/`w2` carry the rwkv6 5-chunk meaning, `w0`/
    /// `a*`/`v*`/`g*`/`k_k`/`k_a`/`r_k` the rwkv7 lora meaning.
    pub time_mix_w0: Option<TensorId>,
    pub time_mix_w1: Option<TensorId>,
    pub time_mix_w2: Option<TensorId>,
    pub time_mix_a0: Option<TensorId>,
    pub time_mix_a1: Option<TensorId>,
    pub time_mix_a2: Option<TensorId>,
    pub time_mix_v0: Option<TensorId>,
    pub time_mix_v1: Option<TensorId>,
    pub time_mix_v2: Option<TensorId>,
    pub time_mix_g1: Option<TensorId>,
    pub time_mix_g2: Option<TensorId>,
    pub time_mix_k_k: Option<TensorId>,
    pub time_mix_k_a: Option<TensorId>,
    pub time_mix_r_k: Option<TensorId>,
    pub time_mix_lerp_x: Option<TensorId>,
    pub time_mix_lerp_w: Option<TensorId>,
    pub time_mix_lerp_k: Option<TensorId>,
    pub time_mix_lerp_v: Option<TensorId>,
    pub time_mix_lerp_r: Option<TensorId>,
    pub time_mix_lerp_g: Option<TensorId>,
    pub time_mix_lerp_fused: Option<TensorId>,
    /// rwkv6's time_first (required there, NOT_REQUIRED on rwkv6qwen2 — a
    /// missing one is what selects the GLA path, rwkv6-base.cpp:50)
    pub time_mix_first: Option<TensorId>,
    pub time_mix_decay: Option<TensorId>,
    pub time_mix_decay_w1: Option<TensorId>,
    pub time_mix_decay_w2: Option<TensorId>,
    pub time_mix_key: Option<TensorId>,
    pub time_mix_value: Option<TensorId>,
    pub time_mix_receptance: Option<TensorId>,
    /// rwkv6qwen2's optional biases (rwkv6qwen2.cpp:66-68)
    pub time_mix_key_b: Option<TensorId>,
    pub time_mix_value_b: Option<TensorId>,
    pub time_mix_receptance_b: Option<TensorId>,
    pub time_mix_gate: Option<TensorId>,
    /// the group-norm affine pair (required rwkv6, NOT_REQUIRED rwkv7/arwkv7)
    pub time_mix_ln: Option<TensorId>,
    pub time_mix_ln_b: Option<TensorId>,
    pub time_mix_output: Option<TensorId>,

    /// RWKV channel mix (rwkv6.cpp:80-85 / rwkv7.cpp:112-115)
    pub channel_mix_lerp_k: Option<TensorId>,
    pub channel_mix_lerp_r: Option<TensorId>,
    pub channel_mix_key: Option<TensorId>,
    pub channel_mix_value: Option<TensorId>,
    pub channel_mix_receptance: Option<TensorId>,

    /// gemma3n (gemma3n.cpp:63-74): the altup/laurel tensors
    pub altup_correct_coef: Option<TensorId>,
    pub altup_correct_scale: Option<TensorId>,
    pub altup_predict_coef: Option<TensorId>,
    pub altup_router: Option<TensorId>,
    pub altup_router_norm: Option<TensorId>,
    pub laurel_l: Option<TensorId>,
    pub laurel_r: Option<TensorId>,
    pub laurel_post_norm: Option<TensorId>,

    // audio round 5 (TTS): wavtokenizer-dec's per-stack members
    // (llama-model.h:166-215 `llama_layer_posnet` / `llama_layer_convnext`)
    pub posnet: Option<PosnetTensors>,
    pub convnext: Option<ConvnextTensors>,
}

/// wavtokenizer-dec's `llama_layer_posnet` (llama-model.h:166-199): the
/// resnet conv pairs (blocks 0/1/3/4), the single attention block (2) and
/// the trailing group-norm (5) — the trailing norm reuses the ATTN_NORM
/// tensor slot like the C loader does (wavtokenizer-dec.cpp:70-71).
#[derive(Debug, Clone, Copy, Default)]
pub struct PosnetTensors {
    pub norm1: Option<TensorId>,
    pub norm1_b: Option<TensorId>,
    pub conv1: Option<TensorId>,
    pub conv1_b: Option<TensorId>,
    pub norm2: Option<TensorId>,
    pub norm2_b: Option<TensorId>,
    pub conv2: Option<TensorId>,
    pub conv2_b: Option<TensorId>,
    pub attn_norm: Option<TensorId>,
    pub attn_norm_b: Option<TensorId>,
    pub attn_q: Option<TensorId>,
    pub attn_q_b: Option<TensorId>,
    pub attn_k: Option<TensorId>,
    pub attn_k_b: Option<TensorId>,
    pub attn_v: Option<TensorId>,
    pub attn_v_b: Option<TensorId>,
    pub attn_o: Option<TensorId>,
    pub attn_o_b: Option<TensorId>,
    pub norm: Option<TensorId>,
    pub norm_b: Option<TensorId>,
}

/// wavtokenizer-dec's `llama_layer_convnext` (llama-model.h:201-215).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvnextTensors {
    pub dw: Option<TensorId>,
    pub dw_b: Option<TensorId>,
    pub norm: Option<TensorId>,
    pub norm_b: Option<TensorId>,
    pub pw1: Option<TensorId>,
    pub pw1_b: Option<TensorId>,
    pub pw2: Option<TensorId>,
    pub pw2_b: Option<TensorId>,
    pub gamma: Option<TensorId>,
}

/// granite-switch's per-layer `switch_lora` member (granite-switch.cpp:113-130):
/// `a_*` {n_in, n_rank, n_slots} + `b_*` {n_rank, n_out, n_slots} pairs for
/// the q/k/v/o projections and the gate/up/down FFN matrices.
#[derive(Debug, Clone, Copy)]
pub struct SwitchLoraTensors {
    pub a_q: TensorId,
    pub b_q: TensorId,
    pub a_k: TensorId,
    pub b_k: TensorId,
    pub a_v: TensorId,
    pub b_v: TensorId,
    pub a_o: TensorId,
    pub b_o: TensorId,
    pub a_gate: TensorId,
    pub b_gate: TensorId,
    pub a_up: TensorId,
    pub b_up: TensorId,
    pub a_down: TensorId,
    pub b_down: TensorId,
}

/// Model-level tensors only some archs create (gemma4 per-layer embeddings,
/// lfm2 ColBERT `dense_2` head).
/// `Clone` only (not `Copy`) — arch batch 10's graniteswitch routing tables
/// are `Vec`s.
#[derive(Debug, Clone, Default)]
pub struct ModelTensors {
    pub per_layer_tok_embd: Option<TensorId>,
    pub per_layer_model_proj: Option<TensorId>,
    pub per_layer_proj_norm: Option<TensorId>,
    pub dense_2_out_layers: Option<TensorId>,
    pub dense_2_out_layers_b: Option<TensorId>,
    /// final-norm bias (`output_norm.bias`, llama-model.h:632 `output_norm_b`)
    /// — gpt2 / phi2 / starcoder2 / gptneox carry one, the RMS-norm archs do not
    pub output_norm_b: Option<TensorId>,
    /// bert token-type embedding (`token_types.weight`, bert.cpp:30)
    pub token_types: Option<TensorId>,
    /// bert absolute position embedding (`position_embd.weight`, bert.cpp:32)
    pub position_embd: Option<TensorId>,
    /// bert embedding layer norm (bert.cpp:40-41)
    pub token_embd_norm: Option<TensorId>,
    pub token_embd_norm_b: Option<TensorId>,
    /// bert classification head (bert.cpp:34-38) — RANK pooling only
    pub cls: Option<TensorId>,
    pub cls_b: Option<TensorId>,
    pub cls_out_b: Option<TensorId>,
    /// modern-bert's `cls.norm` head norm (modern-bert.cpp:64) — RANK pooling
    /// only, between the head activation and cls_out
    pub cls_norm: Option<TensorId>,
    /// `cls.norm.bias` (a7b94df2c llama-model.h) — the decision head's
    /// LayerNorm bias
    pub cls_norm_b: Option<TensorId>,
    /// clef's decision head (a7b94df2c clef.cpp:30-123) — present iff the
    /// arch is CLEF
    pub clef_head: Option<crate::clef::ClefHeadTensors>,
    // arch batch 7 (deepseek4): the model-level hyper-connection head
    // (deepseek4.cpp:105-107)
    pub hc_head_fn: Option<TensorId>,
    pub hc_head_base: Option<TensorId>,
    pub hc_head_scale: Option<TensorId>,
    // arch batch 10 (graniteswitch): the adapter routing tables read by
    // `llm_graph_input_switch::set_input` (granite-switch.cpp:151-185) —
    // (activate token, stacked slot) / (activate token, substitute) pairs;
    // slot 0 is the base/zero delta so slots are 1-based
    pub graniteswitch_token_to_slot: Vec<(i32, i32)>,
    pub graniteswitch_token_to_substitute: Vec<(i32, i32)>,

    // ------------------------------------------------------------------------
    // arch batch 11a (2026-10)
    // ------------------------------------------------------------------------
    /// kimi-k3 (kimi-k3.cpp:63-65): the final residual-bank score vector
    /// (OUTPUT_RES_SCORE, attn_res_block_size > 0)
    pub output_res_score: Option<TensorId>,

    /// qwen4exp (qwen4exp.cpp:161-163): the model-level HC head mixer — the
    /// arch's output norm (there is no separate output_norm)
    pub hc_head_norm: Option<TensorId>,
    pub hc_head_down: Option<TensorId>,
    pub hc_head_up: Option<TensorId>,

    /// t5 (t5.cpp:46): the encoder half's final norm `enc.output_norm`
    /// (ENC_OUTPUT_NORM) — the generic slot carries dec.output_norm
    pub enc_output_norm: Option<TensorId>,

    /// hrm-text (hrm-text.cpp:46): the learned [n_embd] low-cycle state
    /// `hrm.z_l_init` the graph threads through the low stacks
    pub hrm_z_l_init: Option<TensorId>,

    // ------------------------------------------------------------------------
    // arch batch 14 (2026-10) — the RWKV family + gemma3n
    // ------------------------------------------------------------------------
    /// gemma3n (gemma3n.cpp:36-37): the model-level altup projections
    pub altup_proj: Option<TensorId>,
    pub altup_unembd_proj: Option<TensorId>,

    // audio round 5 (TTS): wavtokenizer-dec's model-level members
    // (wavtokenizer-dec.cpp:12-15 / :80-81 / :106-111)
    pub conv1d: Option<TensorId>,
    pub conv1d_b: Option<TensorId>,
    pub tok_norm: Option<TensorId>,
    pub tok_norm_b: Option<TensorId>,
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// `GGML_PREC_Q8` (ggml.h's `enum ggml_prec`, e9f824d8c) — the one precision
/// value the policy files carry today (`general.tensor_extra.prec_a4`).
pub const GGML_PREC_Q8: u8 = 30;

/// `struct llama_prec_policy` (llama-model.h:618-633, e9f824d8c) — the
/// per-tensor activation precision policy. `prec_src1` recommends the
/// activation precision of the mul_mat each weight feeds (the model-driven
/// W4A4 path; the C stores `ggml_prec_set_src` op-params the GPU kernels
/// read — a CPU build has no such kernel, so the port keeps the policy
/// name-keyed for the saver round-trip and `build_lora_mm` consults it as a
/// no-op hint).
#[derive(Clone, Debug, Default)]
pub struct PrecPolicy {
    /// tensor name -> recommended activation precision. The C keys by the
    /// weight `ggml_tensor *` (`res->src[0]`); the port's weights are
    /// name-addressed, and a `BTreeMap` fixes the iteration order the C's
    /// `unordered_map` leaves unspecified.
    pub prec_src1: std::collections::BTreeMap<String, u8>,
}

impl PrecPolicy {
    /// `llama_prec_policy::apply` (llama-model.cpp:1227-1239) — the C sets
    /// the mul_mat's src1 precision op-param (`ggml_prec_set_src`,
    /// ggml.c — a GPU-only hint). The CPU port has no op-param slot for it;
    /// the consult is documented at the call sites (`build_lora_mm`).
    pub fn apply(&self, _weight: &str) -> bool {
        false
    }

    /// `llama_prec_policy::load` (llama-model.cpp:1244-1281): the
    /// `general.tensor_extra.name` string array plus the
    /// `general.tensor_extra.prec_a4` bool array — tensors that cannot use
    /// 4-bit activations keep their src1 at higher precision (value 0).
    pub fn load(
        gguf: &Gguf,
        arch: LlmArch,
        tensors: &HashMap<String, TensorId>,
    ) -> Result<Self, String> {
        // the general.* keys carry no arch %s, but resolve through the same
        // `ml.llm_kv` templating (llama-model.cpp:1252)
        let key_names = kv_name(arch, LlmKv::GENERAL_TENSOR_EXTRA_NAME);
        let key_prec = kv_name(arch, LlmKv::GENERAL_TENSOR_EXTRA_PREC_A4);

        // `ml.get_arr(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names, false)`
        // — absent key: no policy
        let Some(Value::Array(ggml::GgufType::String, name_vals)) = gguf.find_key(&key_names) else {
            return Ok(PrecPolicy::default());
        };
        let tensor_names: Vec<String> = name_vals
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err(format!("{key_names} must be a string array")),
            })
            .collect::<Result<_, _>>()?;

        // the bool-array check (llama-model.cpp:1254-1258)
        let Some(Value::Array(arr_ty, values)) = gguf.find_key(&key_prec) else {
            return Err(format!("{key_prec} must be a bool array"));
        };
        if *arr_ty != GgufType::Bool {
            return Err(format!("{key_prec} must be a bool array"));
        }

        if values.len() != tensor_names.len() {
            return Err(format!(
                "{key_prec} tensor/value length mismatch ({} vs {})",
                tensor_names.len(),
                values.len()
            ));
        }

        // tensors that can not use 4-bit activations, keep src1 at higher
        // precision — resolve the names against the model's tensors
        // (llama-model.cpp:1271-1279, `for ([name, w] : model.tensors_by_name)`)
        let mut prec_src1 = std::collections::BTreeMap::new();
        for (name, v) in tensor_names.iter().zip(values.iter()) {
            let zero = !matches!(v, Value::Bool(b) if *b);
            if zero && tensors.contains_key(name) {
                prec_src1.insert(name.clone(), GGML_PREC_Q8);
            }
        }

        Ok(PrecPolicy { prec_src1 })
    }
}

/// Model-level structure: the weight tensors actually loaded into a
/// [`Context`], mirroring `llama_model`'s tensor members for the supported
/// archs.
pub struct LlamaModel {
    pub arch: LlmArch,
    pub hparams: LlamaHparams,

    /// `llama_model::name` (llama-model.cpp:1246 `ml.get_key(LLM_KV_GENERAL_NAME,
    /// name, false)`) — "" when the file carries no `general.name`.
    pub name: String,
    /// `llama_model::classifier_labels` (llama-model.cpp:1405) — the
    /// `%s.classifier.output_labels` array (empty for non-classifier models);
    /// `hparams.n_cls_out` is its length (meta.rs reads the same key).
    pub classifier_labels: Vec<String>,
    /// `llama_model::ftype` (llama-model.cpp:1417 `pimpl->ftype = ml.ftype`) —
    /// the `general.file_type` KV as the raw i32 (the C keeps the enum value).
    pub ftype: i32,
    /// `llama_model::prec_policy` (llama-model.h:640-642, e9f824d8c) —
    /// loaded right after the tensors (`prec_policy.load(ml, *this)`,
    /// llama-model.cpp:1809-1811)
    pub prec_policy: PrecPolicy,
    /// `llama_model_base::load_stats` (llama-model.cpp:1226-1229): the
    /// loader's element/byte totals over the file's tensor table.
    pub n_elements: u64,
    pub n_bytes: u64,

    /// ggml context holding every weight tensor (external mmap storage)
    pub ctx: Context,

    /// full tensor-name → id map of all created tensors (duplicated requests
    /// resolve to the same id)
    pub tensors: HashMap<String, TensorId>,

    // input
    pub tok_embd: TensorId,

    // output
    pub output_norm: TensorId,
    /// `output_norm.bias` — the LayerNorm archs of the batch (llama-model.h:632)
    pub output_norm_b: Option<TensorId>,
    pub output_b: Option<TensorId>,
    /// `output.weight`; when absent in the file this aliases `tok_embd`
    /// (the C TENSOR_DUPLICATED fallback)
    pub output: TensorId,

    // output rerank head (qwen3)
    pub cls_out: Option<TensorId>,

    // arch batch 7 (deepseek4): the model-level hyper-connection head
    // (deepseek4.cpp:105-107 — output_hc_fn / output_hc_base / output_hc_scale)
    pub hc_head_fn: Option<TensorId>,
    pub hc_head_base: Option<TensorId>,
    pub hc_head_scale: Option<TensorId>,

    // arch batch 10 (graniteswitch): the adapter routing tables
    // (granite-switch.cpp:48-54 — `adapter_token_to_slot` / `_to_substitute`)
    pub graniteswitch_token_to_slot: Vec<(i32, i32)>,
    pub graniteswitch_token_to_substitute: Vec<(i32, i32)>,

    // arch batch 11a: kimi-k3's final residual-bank score (OUTPUT_RES_SCORE)
    pub output_res_score: Option<TensorId>,
    // arch batch 11a: qwen4exp's model-level HC head mixer (HC_HEAD_NORM /
    // HC_HEAD_DOWN / HC_HEAD_UP — the arch's output norm)
    pub hc_head_norm: Option<TensorId>,
    pub hc_head_down: Option<TensorId>,


    pub hc_head_up: Option<TensorId>,

    // MTP batch 17 (t5): the ENCODER half's final norm (`enc.output_norm`,
    // t5.cpp:46) — the generic output_norm slot carries the DECODER norm
    // (dec.output_norm) on the full T5 arch
    pub enc_output_norm: Option<TensorId>,

    // arch batch 12: hrm-text's learned low-cycle state (hrm-text.cpp:46,
    // llama-model.h:647 `hrm_z_l_init`)
    pub hrm_z_l_init: Option<TensorId>,

    // arch batch 14: gemma3n's model-level altup projections
    // (gemma3n.cpp:36-37)
    pub altup_proj: Option<TensorId>,
    pub altup_unembd_proj: Option<TensorId>,

    // audio round 5 (TTS): wavtokenizer-dec's model-level members
    // (wavtokenizer-dec.cpp:12-15 / :80-81 / :106-111) — the stem conv pair,
    // the between-stacks LayerNorm and the head bias
    pub conv1d: Option<TensorId>,
    pub conv1d_b: Option<TensorId>,
    pub tok_norm: Option<TensorId>,
    pub tok_norm_b: Option<TensorId>,

    // model-level tensors of specific archs (gemma4 per-layer embeddings,
    // lfm2 ColBERT `dense_2` head)
    pub per_layer_tok_embd: Option<TensorId>,
    pub per_layer_model_proj: Option<TensorId>,
    pub per_layer_proj_norm: Option<TensorId>,
    pub dense_2_out_layers: Option<TensorId>,
    pub dense_2_out_layers_b: Option<TensorId>,

    // bert members (models/bert.cpp:29-41)
    /// `type_embd` = `token_types.weight`
    pub token_types: Option<TensorId>,
    /// `pos_embd` = `position_embd.weight`
    pub position_embd: Option<TensorId>,
    /// `tok_norm` / `tok_norm_b` = `token_embd_norm.{weight,bias}`
    pub token_embd_norm: Option<TensorId>,
    pub token_embd_norm_b: Option<TensorId>,
    /// `cls` / `cls_b` / `cls_out` / `cls_out_b` (bert.cpp:34-38) — the
    /// reranker head, only read by RANK pooling
    pub cls: Option<TensorId>,
    pub cls_b: Option<TensorId>,
    pub cls_out_b: Option<TensorId>,
    /// `cls.norm` (modern-bert.cpp:64) — the head norm of the GTE reranker
    pub cls_norm: Option<TensorId>,
    /// `cls.norm.bias` (a7b94df2c llama-model.h) — decision head LayerNorm
    pub cls_norm_b: Option<TensorId>,
    /// clef's decision head (a7b94df2c clef.cpp:30-123)
    pub clef_head: Option<crate::clef::ClefHeadTensors>,

    pub layers: Vec<LayerTensors>,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("arch", &self.arch)
            .field("n_layer", &self.layers.len())
            .field("n_tensors", &self.tensors.len())
            .finish()
    }
}

impl LlamaModel {
    /// `llama_model::tensor_by_name` equivalent.
    pub fn tensor(&self, name: &str) -> Option<TensorId> {
        self.tensors.get(name).copied()
    }

    // ---- model-level calculators: thin delegates to the hparams calculator
    //      (llama-hparams.cpp) — never recompute here ----

    /// `llama_model::memory_breakdown` (llama-model.cpp:1933-1952) — the C
    /// sums backend-buffer sizes (the mmap mapping's length for a mapped
    /// model); the port reports the loaded tensors' byte total
    /// (`load_stats`'s n_bytes over the file's tensor table).
    pub fn memory_breakdown(&self) -> u64 {
        self.n_bytes
    }

    pub fn n_layer(&self) -> u32 {
        self.hparams.n_layer()
    }
    pub fn n_embd(&self) -> u32 {
        self.hparams.n_embd
    }
    pub fn n_head(&self, il: usize) -> u32 {
        self.hparams.n_head(il)
    }
    pub fn n_head_kv(&self, il: usize) -> u32 {
        self.hparams.n_head_kv(il)
    }
    /// `n_gqa` = n_head / n_head_kv (0 when n_head_kv == 0)
    pub fn n_gqa(&self, il: usize) -> u32 {
        self.hparams.n_gqa(il)
    }
    pub fn n_ff(&self, il: usize) -> u32 {
        self.hparams.n_ff(il)
    }
    pub fn n_embd_head_k(&self, il: usize) -> u32 {
        self.hparams.n_embd_head_k(il)
    }
    pub fn n_embd_head_v(&self, il: usize) -> u32 {
        self.hparams.n_embd_head_v(il)
    }
    /// `n_embd_k_gqa` — n_embd_head_k * n_head_kv
    pub fn n_embd_k_gqa(&self, il: usize) -> u32 {
        self.hparams.n_embd_k_gqa(il)
    }
    /// `n_embd_v_gqa` — n_embd_head_v * n_head_kv
    pub fn n_embd_v_gqa(&self, il: usize) -> u32 {
        self.hparams.n_embd_v_gqa(il)
    }
    pub fn n_embd_k_gqa_max(&self) -> u32 {
        self.hparams.n_embd_k_gqa_max()
    }
    pub fn n_embd_v_gqa_max(&self) -> u32 {
        self.hparams.n_embd_v_gqa_max()
    }
    pub fn n_rot(&self, il: usize) -> u32 {
        self.hparams.n_rot(il)
    }
}

// ---------------------------------------------------------------------------
// port-status marker (mirrors meta::ArchHparamsSupport)
// ---------------------------------------------------------------------------

/// Port-status of an arch's `load_arch_tensors` (not present in C++ — a
/// migration marker used by the port's callers/tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchTensorsSupport {
    /// `load_arch_tensors` ported 1:1 from src/models/<arch>.cpp.
    Full,
    /// tensor set ported; arch kept flagged pending e2e graph verification.
    Partial,
    /// `load_model` refuses this arch for now.
    Unsupported,
}

pub fn arch_tensors_support(arch: LlmArch) -> ArchTensorsSupport {
    use ArchTensorsSupport::*;
    match arch {
        LlmArch::QWEN2 | LlmArch::LLAMA => Full,
        LlmArch::QWEN3
        | LlmArch::GEMMA2
        | LlmArch::GEMMA3
        | LlmArch::PHI3
        | LlmArch::OPENAI_MOE
        | LlmArch::GEMMA4
        | LlmArch::LFM2MOE
        // MTP batch 17: lfm2 dense — the shared lfm2moe arm (synthetic-file
        // e2e in tests/mtp2_e2e.rs)
        | LlmArch::LFM2
        | LlmArch::GRANITE_HYBRID
        | LlmArch::QWEN35
        // arch batch (2026-09-24): tensor set + graph + synthetic-GGUF parity
        // vs the reference server (tests/arch_batch_e2e.rs)
        | LlmArch::GPT2
        | LlmArch::PHI2
        | LlmArch::STARCODER2
        | LlmArch::COMMAND_R
        | LlmArch::GPTNEOX
        | LlmArch::OLMO2
        // arch batch 2 (2026-09-25): codeshell / orion / olmo / xverse /
        // internlm2 / exaone / gemma(v1) / falcon — tensor set + graph +
        // synthetic-GGUF parity vs the reference server
        // (tests/arch_batch2_e2e.rs)
        | LlmArch::CODESHELL
        | LlmArch::ORION
        | LlmArch::OLMO
        | LlmArch::XVERSE
        | LlmArch::INTERNLM2
        | LlmArch::EXAONE
        | LlmArch::GEMMA
        | LlmArch::FALCON
        // arch batch 3 (2026-09-27): the ALiBi family + the cheap no-rope
        // archs — tensor set + graph + synthetic-GGUF parity vs the reference
        // server (tests/arch_batch3_e2e.rs)
        | LlmArch::BAICHUAN
        | LlmArch::BLOOM
        | LlmArch::MPT
        | LlmArch::STARCODER
        | LlmArch::REFACT
        | LlmArch::PLAMO
        | LlmArch::STABLELM
        | LlmArch::GRANITE
        | LlmArch::MINICPM
        // arch batch 4 (2026-09-28): the MoE family (granite-moe / phimoe /
        // arctic / olmoe / qwen2moe / qwen3moe / ernie4-5-moe) + the dense
        // archs that fell out of it (smollm3 / seed-oss / openelm) — tensor
        // set + graph + synthetic-GGUF parity vs the reference server
        // (tests/arch_batch4_e2e.rs)
        | LlmArch::QWEN2MOE
        | LlmArch::QWEN3MOE
        | LlmArch::GRANITE_MOE
        | LlmArch::PHIMOE
        | LlmArch::ARCTIC
        | LlmArch::OLMOE
        | LlmArch::ERNIE4_5_MOE
        | LlmArch::SMOLLM3
        | LlmArch::SEED_OSS
        | LlmArch::OPENELM
        // arch batch 5 (2026-09-24): the mamba family — the recurrent-state
        // archs (mamba / mamba2 / jamba / nemotron-h), tensor set + graph +
        // synthetic-GGUF parity vs the reference server
        // (tests/arch_batch5_e2e.rs)
        | LlmArch::MAMBA
        | LlmArch::MAMBA2
        | LlmArch::JAMBA
        | LlmArch::NEMOTRON_H
        // arch batch 6 (2026-09-24): the DeepSeek MLA family — deepseek2
        // (classic MLA absorption + MoE) with the lite/OCR/non-MLA variants
        // and the non-MLA v2 base, tensor set + graph + synthetic-GGUF parity
        // vs the reference server (tests/arch_batch6_e2e.rs). deepseek32
        // (V3.2) landed with the DSA lightning indexer.
        | LlmArch::DEEPSEEK
        | LlmArch::DEEPSEEK2
        | LlmArch::DEEPSEEK2OCR
        | LlmArch::DEEPSEEK32
        // arch batch 7 (deepseek4): hyper-connections + the compressed DSV4
        // KV cache — tensor set + graph + synthetic-GGUF parity vs the
        // reference server (tests/arch_batch7_e2e.rs)
        | LlmArch::DEEPSEEK4
        // arch batch 6b (2026-09-24): nemotron (dense) / grok / chameleon /
        // deci / jais / falcon-h1 / plamo2 — tensor set + graph +
        // synthetic-GGUF parity vs the reference server
        // (tests/arch_batch6b_e2e.rs). falcon-h1 / plamo2 are the hybrid
        // mamba archs (RecurrentState + KvCache in one graph).
        | LlmArch::NEMOTRON
        | LlmArch::GROK
        | LlmArch::CHAMELEON
        | LlmArch::DECI
        | LlmArch::JAIS
        | LlmArch::FALCON_H1
        | LlmArch::PLAMO2
        // arch batch 8 (2026-09-30): the MoE long-tail family — hunyuan-moe /
        // dots1 / bailingmoe / bailingmoe2 / glm4-moe / minimax-m2 /
        // cohere2moe / exaone-moe — tensor set + graph + synthetic-GGUF parity
        // vs the reference server (tests/arch_batch8_e2e.rs). cohere2moe /
        // exaone-moe are the iswa archs (the every-4th SWA pattern); their
        // ForwardWeights/CLI arms landed with the graph.
        | LlmArch::HUNYUAN_MOE
        | LlmArch::DOTS1
        | LlmArch::BAILINGMOE
        | LlmArch::BAILINGMOE2
        | LlmArch::GLM4_MOE
        | LlmArch::MINIMAX_M2
        | LlmArch::COHERE2MOE
        | LlmArch::EXAONE_MOE
        // arch batch 9 (2026-10): the linear-attention family — plamo3 (the
        // SWA + post-norm + swiglu dense arch), qwen3next (GDN + gated
        // attention + MoE), kimi-linear / bailingmoe3 (KDA delta net + MLA)
        // — tensor set + graph + synthetic-GGUF parity vs the reference
        // server (tests/arch_batch9_e2e.rs). ForwardWeights/CLI arms landed
        // with the graph.
        | LlmArch::PLAMO3
        | LlmArch::QWEN3NEXT
        | LlmArch::KIMI_LINEAR
        | LlmArch::BAILINGMOE3
        // arch batch 10 (2026-10): small-arch + EXP-op batch — smallthinker
        // (the probs_in + ReGLU MoE, iswa graph template), llada-moe (the
        // non-causal no-cache diffusion attention), minimax-01 (lightning
        // attention, GGML_UNARY_OP_EXP) and graniteswitch (the stacked
        // switch-LoRA + in-graph adapter router) — tensor set + graph +
        // synthetic-GGUF parity vs the reference server
        // (tests/arch_batch10_e2e.rs). ForwardWeights/CLI arms landed with
        // the graph.
        | LlmArch::SMALLTHINKER
        | LlmArch::LLADA_MOE
        | LlmArch::MINIMAX_01
        | LlmArch::GRANITE_SWITCH
        // arch batch 11a (2026-10) — the long-tail queue, first half:
        // apertus (the xIELU FFN), grovemoe (the dual softmax MoE with the
        // chunk experts), qwen35moe (the qwen3next hybrid with attn_post_norm
        // + IMRoPE), kimi-k3 (KDA + nope-MLA + the latent SITU MoE + the
        // residual bank), dots3note (DSA indexer over the iswa MLA pair),
        // minimax-m3 (MSA sparse attention, dense with FA off) and qwen4exp
        // (HC residual streams + GDN; QSA/PLE not ported — the no-ratio/no-PLE
        // file is the verified configuration). Tensor set + graph +
        // synthetic-GGUF parity vs the reference server
        // (tests/arch_batch11a_e2e.rs). ForwardWeights/CLI arms landed with
        // the graph.
        | LlmArch::APERTUS
        | LlmArch::GROVEMOE
        | LlmArch::QWEN35MOE
        | LlmArch::KIMI_K3
        | LlmArch::DOTS3NOTE
        | LlmArch::MINIMAX_M3
        | LlmArch::QWEN4EXP
        // arch batch 11b (2026-10) — the long-tail queue, second half:
        // arcee (the xverse body + relu² + per-layer rope factors), jais2
        // (LN+bias + rope + relu² MLP with biases), talkie (weightless RMS +
        // the post-rope [1, n_head] q-norm + the embd-skip residual),
        // nanbeige (the n_loops layer expansion), dream / rnd1 (the
        // llada-family diffusion archs — reference memory = nullptr,
        // verified in-port) and eurobert (encoder, the reference
        // llama_encode dumps). Tensor set + graph + synthetic-GGUF parity
        // vs the reference server (tests/arch_batch11b_e2e.rs).
        // ForwardWeights/CLI arms landed with the graph.
        | LlmArch::ARCEE
        | LlmArch::JAIS2
        | LlmArch::TALKIE
        | LlmArch::NANBEIGE
        | LlmArch::DREAM
        | LlmArch::RND1
        | LlmArch::EUROBERT
        // arch batch 12 (2026-10) — the final long-tail queue: hrm-text (the
        // alternating low/high stacks with the [n_embd] low-cycle state and
        // the sigmoid attention gate; cache slots alias the two physical
        // stacks), laguna (sigmoid-routed MoE + score-correction bias + the
        // softplus attention output gate, optional hybrid full/SWA with
        // per-layer-type RoPE) and maple (softmax MoE over the iswa pair;
        // rope on the SWA layers only). Tensor set + graph + synthetic-GGUF
        // parity vs the reference server (tests/arch_batch12_e2e.rs).
        // ForwardWeights/CLI/server arms landed with the graph.
        | LlmArch::HRM_TEXT
        | LlmArch::LAGUNA
        | LlmArch::MAPLE
        // arch batch 13 (2026-09) — the P0 standard-attention queue: llama4 /
        // qwen3vl / qwen3vlmoe / qwen2vl / glm4 / glm-dsa / chatglm /
        // mistral3 / cohere2 / minicpm3 / exaone4 / bitnet / dbrx /
        // ernie4-5 (dense) + the NEMOTRON_H_MOE loader arm (its graph reuses
        // nemotron-h's; graph_mtp stays documented-skip, PARITY batch 5 §5).
        // Tensor set + graph + synthetic-GGUF parity vs the reference server
        // (tests/arch_batch13_e2e.rs). ForwardWeights/CLI/server arms landed
        // with the graph.
        | LlmArch::LLAMA4
        | LlmArch::QWEN3VL
        | LlmArch::QWEN3VLMOE
        | LlmArch::QWEN2VL
        | LlmArch::GLM4
        | LlmArch::GLM_DSA
        | LlmArch::CHATGLM
        | LlmArch::MISTRAL3
        | LlmArch::COHERE2
        | LlmArch::MINICPM3
        | LlmArch::EXAONE4
        | LlmArch::BITNET
        | LlmArch::DBRX
        | LlmArch::ERNIE4_5
        | LlmArch::NEMOTRON_H_MOE
        // arch batch 14 (2026-10) — the P0 new-mechanism queue: the RWKV
        // family (rwkv6 / rwkv6qwen2 — the rwkv6-base time/channel mix + the
        // WKV6/GLA scans; rwkv7 / arwkv7 — the rwkv7-base mix + the WKV7
        // scan; pure-recurrent, the RecurrentState token-shift + WKV cells)
        // and gemma3n (per-layer embeddings + the altup/laurel machinery +
        // the KV-reuse layers). Tensor set + graph + synthetic-GGUF parity
        // vs the reference server (tests/arch_batch14_e2e.rs).
        // ForwardWeights/CLI/server arms landed with the graph.
        | LlmArch::RWKV6
        | LlmArch::RWKV6QWEN2
        | LlmArch::RWKV7
        | LlmArch::ARWKV7
        | LlmArch::GEMMA3N
        // arch batch 15 (2026-10) — the P1+P2 queue of AUDIT_models.md:
        // qwen(v1) / maincoder / pangu-embed / plm / cogvlm / spark2-5 /
        // muse-glimmer / llada / hunyuan-vl / granite-swa / afmoe / mellum /
        // gemma-embedding (encoder, the llama_encode dumps) / hy-v3 /
        // mimo2 / step35 / hy-v4 (iHC + gated MLA + the optional DSA
        // indexer) + the co-arms mistral4 (deepseek2's loader+graph),
        // paddleocr (ernie4_5's loader) and hunyuan-dense (hunyuan_vl's).
        // MTP tensors of hy-v3/mimo2/step35 load; their graph_mtp heads are
        // the documented batch-15 skip (PARITY.md batch 15) — the bailingmoe3/
        // cohere2moe/glm4-moe precedent. llama-embed reuses the LLAMA arm
        // (graph<true> = the port's build_llama_forward embed tap).
        | LlmArch::QWEN
        | LlmArch::MAINCODER
        | LlmArch::PANGU_EMBED
        | LlmArch::PLM
        | LlmArch::COGVLM
        | LlmArch::SPARK2_5
        | LlmArch::MUSE_GLIMMER
        | LlmArch::LLADA
        | LlmArch::HUNYUAN_VL
        | LlmArch::HUNYUAN_DENSE
        | LlmArch::GRANITE_SWA
        | LlmArch::AFMOE
        | LlmArch::MELLUM
        | LlmArch::GEMMA_EMBEDDING
        | LlmArch::HY_V3
        | LlmArch::MIMO2
        | LlmArch::STEP35
        | LlmArch::HY_V4
        | LlmArch::MISTRAL4
        | LlmArch::PADDLEOCR
        | LlmArch::LLAMA_EMBED
        // encoder-only archs: tensor set + graph ported, verified against the
        // reference `llama_encode` dumps (tests/{bert,t5}_e2e.rs)
        | LlmArch::BERT
        | LlmArch::T5ENCODER
        // MTP batch 17: the full T5 (encoder + the decoder half's tensor set
        // + graph, tests/mtp2_e2e.rs)
        | LlmArch::T5
        // the bert-variant encoder family (arch batch 16): jina-bert-v2 /
        // jina-bert-v3 / nomic-bert / nomic-bert-moe (the arch-keyed branches
        // of the shared bert.cpp graph body), neo-bert and modern-bert (own
        // graphs + modern-bert's RANK pooling head) — verified against the
        // reference llama_encode dumps (tests/bert_variants_e2e.rs)
        | LlmArch::JINA_BERT_V2
        | LlmArch::JINA_BERT_V3
        | LlmArch::NOMIC_BERT
        | LlmArch::NOMIC_BERT_MOE
        | LlmArch::NEO_BERT
        | LlmArch::MODERN_BERT => Partial,
        // audio round 5 (TTS): the audio-family LM archs — qwen3tts (a pure
        // typedef of qwen3vl, models.h:625-627), pockettts (the CALM flow
        // backbone) and wavtokenizer-dec (the code→PCM decoder). Loader arms
        // + graphs in graph_arch.rs; the graph-side probes are
        // tests/tts_archs_e2e.rs (the ForwardWeights/CLI routing is an
        // integrator item — context.rs is outside this batch's ownership)
        | LlmArch::QWEN3TTS
        | LlmArch::POCKETTTS
        | LlmArch::WAVTOKENIZER_DEC => Partial,
        // sync batch A (def4d406a): glm5-next — loader 1:1 (tests/glm5_e2e.rs +
        // parity/glm5_parity.sh vs the NEW reference); the graph waits on the
        // kv-cache lane's kpool port
        // sync batch A2 (2026-10-05, a7b94df2c) — clef: the qwen35 trunk
        // (no memory: the causal no-cache mask + zero-state GDN) + the
        // joint decision head (clef.cpp, upstream 99b95488c). Loader arm
        // shares QWEN35's (clef.cpp:27-28); graph + driver live in
        // crates/llama/src/clef.rs; synthetic-file parity vs the NEW
        // reference (tests/clef_e2e.rs).
        LlmArch::CLEF => Full,
        LlmArch::GLM5_NEXT => Partial,
        _ => Unsupported,
    }
}

// ---------------------------------------------------------------------------
// ModelLoader — llama_model_loader::create_tensor / check_tensor_dims /
//               done_getting_tensors (mmap path)
// ---------------------------------------------------------------------------

struct ModelLoader<'a> {
    gguf: &'a Gguf,
    mmap: Arc<Mmap>,
    ctx: Context,
    by_name: HashMap<String, TensorId>,
    /// C `n_created`: gguf tensors consumed by create_tensor (duplicated
    /// re-requests and absent optionals never count)
    n_created: usize,
}

impl<'a> ModelLoader<'a> {
    fn new(gguf: &'a Gguf, mmap: Arc<Mmap>) -> Self {
        Self {
            gguf,
            mmap,
            ctx: Context::new(),
            by_name: HashMap::new(),
            n_created: 0,
        }
    }

    /// `llama_format_tensor_shape`
    fn format_shape(ne: &[i64]) -> String {
        let dims: Vec<String> = ne.iter().map(|d| d.to_string()).collect();
        format!("[{}]", dims.join(", "))
    }

    /// `llama_model_loader::check_tensor_dims` with `allow_reshape == false`
    /// (every supported arch uses the default): expected dims must equal
    /// `ne[i]`, expected dims beyond the list must be 1.
    fn check_tensor_dims(
        &self,
        name: &str,
        ne: &[i64],
        required: bool,
        allow_reshape: bool,
    ) -> Result<Option<&'a TensorInfo>, String> {
        let Some(ti) = self.gguf.find_tensor(name) else {
            if !required {
                return Ok(None);
            }
            return Err(format!("tensor '{name}' not found"));
        };

        if allow_reshape {
            // llama-model-loader.cpp:896-903 — check total number of elements
            // only; create_tensor re-shapes the context tensor to `ne`
            let ncur: i64 = ti.ne.iter().product();
            let nexp: i64 = ne.iter().product();
            if ncur != nexp {
                return Err(format!(
                    "tensor '{name}' has wrong shape; expected {}, got {}",
                    Self::format_shape(&ne[..ne.len().min(4)]),
                    Self::format_shape(&ti.ne),
                ));
            }
            return Ok(Some(ti));
        }

        // (i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)
        for i in 0..4 {
            let want = if i < ne.len() { ne[i] } else { 1 };
            if want != ti.ne[i] {
                return Err(format!(
                    "tensor '{name}' has wrong shape; expected {}, got {}",
                    Self::format_shape(&ne[..ne.len().min(4)]),
                    Self::format_shape(&ti.ne),
                ));
            }
        }
        Ok(Some(ti))
    }

    /// `llama_model_loader::create_tensor` (real-file path).
    ///
    /// `suffix` mirrors the C++ `tn(tensor, suffix, bid)`: an empty string is
    /// the suffix-less form `tn(tensor, bid)` (used by ssm_a/ssm_d, whose file
    /// names carry no `.weight`).
    ///
    /// Returns the tensor id, `None` for an absent TENSOR_NOT_REQUIRED tensor
    /// (the C NULL), or an error mirroring the C++ runtime_error / GGML_ABORT
    /// texts.
    fn create_tensor(
        &mut self,
        tensor: LlmTensor,
        suffix: &str,
        bid: i32,
        ne: &[i64],
        flags: u32,
    ) -> Result<Option<TensorId>, String> {
        let name = if suffix.is_empty() {
            crate::arch::tensor_name(tensor, bid)
        } else {
            tensor_name_suffix(tensor, suffix, bid, -1)
        };

        // sanity checks (create_tensor): input/output tensors carry no block
        // id, repeating tensors must have one. The condition is "does the
        // tensor name contain a block slot" (see module docs — ROPE_FACTORS_*
        // are REPEATING but live outside blk. space and pass bid >= 0 in C++).
        let has_blk_slot = tensor.template().map(|t| t.contains("%d")).unwrap_or(false);
        match tensor.layer() {
            LlmTensorLayer::INPUT | LlmTensorLayer::OUTPUT => {
                if bid != -1 {
                    return Err(format!(
                        "input/output layer tensor {name} used with a layer number"
                    ));
                }
            }
            LlmTensorLayer::REPEATING => {
                if bid == -1 && has_blk_slot {
                    return Err(format!(
                        "repeating layer tensor {name} used without a layer number"
                    ));
                }
            }
        }

        let required = flags & TENSOR_NOT_REQUIRED == 0;
        let allow_reshape = flags & TENSOR_ALLOW_RESHAPE != 0;
        let Some(ti) = self.check_tensor_dims(&name, ne, required, allow_reshape)? else {
            return Ok(None);
        };

        // duplicated: reuse the already-created tensor (same name), no recount
        if flags & TENSOR_DUPLICATED != 0 {
            if let Some(&id) = self.by_name.get(&name) {
                return Ok(Some(id));
            }
        }

        // create the context tensor with the file's type/shape (the meta
        // tensor gguf_init_from_file produced in C) and point it at the mmap;
        // TENSOR_ALLOW_RESHAPE creates it at the *requested* shape instead
        // (llama-model-loader.cpp:1348-1360 re-ne's t_meta)
        let mut ne4 = [1i64; 4];
        if allow_reshape {
            for (i, &d) in ne.iter().enumerate() {
                ne4[i] = d;
            }
        } else {
            for (i, &d) in ti.ne.iter().enumerate() {
                ne4[i] = d;
            }
        }
        let id = self.ctx.new_tensor(ti.ty, ne4);
        self.ctx.set_name(id, &name);
        // absolute file offset of this tensor's storage inside its mmap — for a
        // multi-part (`split.*`) GGUF each tensor points into the part that
        // declares it, the port-side `weights_map`/`files[idx]` of
        // llama-model-loader.cpp:583-651 (Gguf::open assembled the parts);
        // single-file GGUFs are the `part == 0` special case
        let (mmap, part_base) = self.gguf.part_storage(ti.part);
        let data_off = (part_base + ti.offset) as usize;
        self.ctx.set_external_storage(id, mmap, data_off);

        self.by_name.insert(name, id);
        // C++ `llama_model_loader::create_tensor` counts only non-duplicated
        // creations (a TENSOR_DUPLICATED request bumps `size_data` instead);
        // `done_getting_tensors` therefore compares n_created against the
        // number of distinct file tensors. gemma4's output→token_embd tie
        // fallback happens *before* token_embd itself is created, so the dup
        // request really does duplicate there.
        if flags & TENSOR_DUPLICATED == 0 {
            self.n_created += 1;
        }
        Ok(Some(id))
    }

    /// `vocab.n_token_types()` — `tokenizer.ggml.token_type_count`
    /// (llama-vocab.cpp:1947), needed by the BERT loader (bert.cpp:26-30). The
    /// C reads it off the *vocab*; model.rs has no vocab, so it comes from the
    /// same gguf key.
    fn token_type_count(&self) -> u32 {
        let key = kv_name(LlmArch::BERT, LlmKv::TOKENIZER_TOKEN_TYPE_COUNT);
        self.gguf.get_u32(&key).unwrap_or(0)
    }

    /// `done_getting_tensors(partial=false)` — every declared gguf tensor must
    /// have been consumed by the arch's create_tensor calls.
    fn done_getting_tensors(&self) -> Result<(), String> {
        let n_tensors = self.gguf.tensors.len();
        if self.n_created > n_tensors {
            return Err(format!(
                "too many tensors created; expected {n_tensors}, got {}",
                self.n_created
            ));
        }
        if self.n_created < n_tensors {
            if std::env::var("LLAMA_DEBUG_UNCONSUMED").is_ok() {
                let unconsumed: Vec<&str> = self
                    .gguf
                    .tensors
                    .iter()
                    .map(|t| t.name.as_str())
                    .filter(|n| !self.by_name.contains_key(*n))
                    .collect();
                eprintln!("unconsumed tensors: {unconsumed:?}");
            }
            return Err(format!(
                "wrong number of tensors; expected {n_tensors}, got {}",
                self.n_created
            ));
        }
        Ok(())
    }
}

/// `llama_model_base::create_tensor_qkv` (the flags path; the TENSOR_SKIP
/// branch is dropped — see module docs).
fn create_tensor_qkv(
    l: &mut LayerTensors,
    ld: &mut ModelLoader,
    bid: i32,
    n_embd_: i64,
    n_embd_q_: i64,
    n_embd_k_: i64,
    n_embd_v_: i64,
    flags: u32,
) -> Result<(), String> {
    let n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    l.wqkv = ld.create_tensor(
        LlmTensor::ATTN_QKV,
        "weight",
        bid,
        &[n_embd_, n_embd_qkv],
        TENSOR_NOT_REQUIRED,
    )?;
    if l.wqkv.is_some() {
        l.wqkv_b = ld.create_tensor(
            LlmTensor::ATTN_QKV,
            "bias",
            bid,
            &[n_embd_qkv],
            TENSOR_NOT_REQUIRED,
        )?;
        // Fused weights may coexist with separate Q/K/V biases in legacy or
        // custom GGUFs.
        if l.wqkv_b.is_none() {
            l.wq_b = ld.create_tensor(
                LlmTensor::ATTN_Q,
                "bias",
                bid,
                &[n_embd_q_],
                TENSOR_NOT_REQUIRED,
            )?;
            l.wk_b = ld.create_tensor(
                LlmTensor::ATTN_K,
                "bias",
                bid,
                &[n_embd_k_],
                TENSOR_NOT_REQUIRED,
            )?;
            l.wv_b = ld.create_tensor(
                LlmTensor::ATTN_V,
                "bias",
                bid,
                &[n_embd_v_],
                TENSOR_NOT_REQUIRED,
            )?;
        }
    } else {
        l.wq = ld.create_tensor(
            LlmTensor::ATTN_Q,
            "weight",
            bid,
            &[n_embd_, n_embd_q_],
            flags,
        )?;
        l.wk = ld.create_tensor(
            LlmTensor::ATTN_K,
            "weight",
            bid,
            &[n_embd_, n_embd_k_],
            flags,
        )?;
        l.wv = ld.create_tensor(
            LlmTensor::ATTN_V,
            "weight",
            bid,
            &[n_embd_, n_embd_v_],
            flags,
        )?;
        l.wq_b = ld.create_tensor(
            LlmTensor::ATTN_Q,
            "bias",
            bid,
            &[n_embd_q_],
            TENSOR_NOT_REQUIRED,
        )?;
        l.wk_b = ld.create_tensor(
            LlmTensor::ATTN_K,
            "bias",
            bid,
            &[n_embd_k_],
            TENSOR_NOT_REQUIRED,
        )?;
        l.wv_b = ld.create_tensor(
            LlmTensor::ATTN_V,
            "bias",
            bid,
            &[n_embd_v_],
            TENSOR_NOT_REQUIRED,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// n_vocab — LLAMA_LOAD_LOCALS's vocab.n_tokens()
// ---------------------------------------------------------------------------

/// The C++ tensor shapes use `vocab.n_tokens()`; the vocab module counts
/// `tokenizer.ggml.tokens`. Resolution chain identical to the C++
/// load_arch_hparams heuristic: `%s.vocab_size` KV, else the token-list length
/// (the two agree in every real file).
fn n_vocab_from_gguf(gguf: &Gguf, arch: LlmArch) -> Result<i64, String> {
    if let Some(v) = gguf.get_u32(&kv_name(arch, LlmKv::VOCAB_SIZE)) {
        return Ok(v as i64);
    }
    if let Some(ggml::Value::Array(_, items)) = gguf.find_key(&kv_name(arch, LlmKv::TOKENIZER_LIST))
    {
        return Ok(items.len() as i64);
    }
    Err(format!(
        "cannot determine vocab size for arch '{}' (no {}.vocab_size, no tokenizer.ggml.tokens)",
        arch.name(),
        arch.name()
    ))
}

// ---------------------------------------------------------------------------
// load_arch_tensors — src/models/<arch>.cpp
// ---------------------------------------------------------------------------

/// `LLAMA_LOAD_LOCALS` — every dim is an int64 layer-0 value, as in C++.
/// (Some macro locals are unused by the archs ported so far; they stay for a
/// 1:1 mapping with the C macro.)
#[allow(dead_code)]
struct LoadLocals {
    n_layer: usize,
    n_layer_all: usize,
    n_layer_nextn: usize,
    n_head: i64,
    n_head_kv: i64,
    n_embd: i64,
    n_embd_k_gqa: i64,
    n_embd_v_gqa: i64,
    n_embd_head_k: i64,
    n_embd_head_v: i64,
    n_ff: i64,
    n_vocab: i64,
    n_rot: i64,
    n_expert: i64,
    n_expert_used: i64,
    n_ctx_train: i64,
}

impl LoadLocals {
    fn new(h: &LlamaHparams, n_vocab: i64) -> Self {
        // n_head()/n_head_kv()/... are layer-0 getters inside LLAMA_LOAD_LOCALS
        let il0 = 0usize;
        Self {
            n_layer: h.n_layer() as usize,
            n_layer_all: h.n_layer_all as usize,
            n_layer_nextn: h.n_layer_nextn as usize,
            n_head: h.n_head(il0) as i64,
            n_head_kv: h.n_head_kv(il0) as i64,
            n_embd: h.n_embd as i64,
            n_embd_k_gqa: h.n_embd_k_gqa(il0) as i64,
            n_embd_v_gqa: h.n_embd_v_gqa(il0) as i64,
            n_embd_head_k: h.n_embd_head_k(il0) as i64,
            n_embd_head_v: h.n_embd_head_v(il0) as i64,
            n_ff: h.n_ff(il0) as i64,
            n_vocab,
            n_rot: h.n_rot(il0) as i64,
            n_expert: h.n_expert as i64,
            n_expert_used: h.n_expert_used(il0) as i64,
            n_ctx_train: h.n_ctx_train as i64,
        }
    }
}

/// Return value of [`load_arch_tensors`] — the model-level members every
/// arch assigns plus the ones only some archs create.
struct ArchTensors {
    tok_embd: TensorId,
    output_norm: TensorId,
    output_b: Option<TensorId>,
    output: TensorId,
    cls_out: Option<TensorId>,
    extra: ModelTensors,
    layers: Vec<LayerTensors>,
}

fn load_arch_tensors(
    arch: LlmArch,
    hparams: &LlamaHparams,
    n_vocab: i64,
    ld: &mut ModelLoader,
) -> Result<ArchTensors, String> {
    let lc = LoadLocals::new(hparams, n_vocab);
    // C++ `layers.resize(n_layer_all)`; nextn/MTP blocks included
    let mut layers: Vec<LayerTensors> = (0..lc.n_layer_all)
        .map(|_| LayerTensors::default())
        .collect();
    let mut extra = ModelTensors::default();

    macro_rules! req {
        ($t:expr, $suf:expr, $bid:expr, $ne:expr) => {
            ld.create_tensor($t, $suf, $bid, $ne, 0)?.unwrap()
        };
    }
    macro_rules! opt {
        ($t:expr, $suf:expr, $bid:expr, $ne:expr) => {
            ld.create_tensor($t, $suf, $bid, $ne, TENSOR_NOT_REQUIRED)?
        };
    }
    macro_rules! dup_fallback {
        ($ne:expr) => {
            ld.create_tensor(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[$ne.0, $ne.1],
                TENSOR_DUPLICATED,
            )?
            .unwrap()
        };
    }
    // arch batch 8: the C's `flags`-parameterized create_tensor calls — the
    // NextN/MTP layers pass TENSOR_SKIP on top (the port loads trunk-only
    // files, so those reads degrade to NOT_REQUIRED, deepseek4's convention)
    macro_rules! opt_or_req {
        ($t:expr, $suf:expr, $bid:expr, $ne:expr, $skip:expr) => {
            if $skip {
                ld.create_tensor($t, $suf, $bid, $ne, TENSOR_NOT_REQUIRED)?
            } else {
                Some(ld.create_tensor($t, $suf, $bid, $ne, 0)?.unwrap())
            }
        };
    }

    /// `llama_model_clef::load_arch_tensors`'s head half (a7b94df2c
    /// clef.cpp:30-123) — runs after `llama_model_qwen35::load_arch_tensors`
    /// (the trunk loader above).
    fn load_clef_head(
        hparams: &LlamaHparams,
        ld: &mut ModelLoader,
        lc: &LoadLocals,
    ) -> Result<crate::clef::ClefHeadTensors, String> {
        use crate::clef::{ClefAttnTensors, ClefHeadLayerTensors, ClefHeadTensors, ClefNormTensors};

        let n_head_decision = hparams.clef_n_head_decision as i64;
        let n_layer_routing = hparams.clef_n_layer_routing as usize;

        // the head geometry comes from the file's tensors (clef.cpp:34-40)
        let w_memory = ld.gguf.find_tensor("decision.proj_memory.weight");
        let w_ffn = ld.gguf.find_tensor("dec.blk.0.ffn_up.weight");
        if w_memory.is_none() || w_ffn.is_none() {
            return Err("the decision head is missing".into());
        }
        let n_embd_h = w_memory.unwrap().ne[1];
        let n_ff_h = w_ffn.unwrap().ne[1];
        if n_embd_h % n_head_decision != 0 {
            return Err("invalid width of the decision head".into());
        }

        // load_norm (clef.cpp:48-51)
        let norm = |ld: &mut ModelLoader,
                    ty: LlmTensor,
                    bid: i32,
                    size: i64|
         -> Result<ClefNormTensors, String> {
            Ok(ClefNormTensors {
                w: ld.create_tensor(ty, "weight", bid, &[size], 0)?.unwrap(),
                b: ld.create_tensor(ty, "bias", bid, &[size], 0)?.unwrap(),
            })
        };
        // load_attn (clef.cpp:53-60)
        let attn = |ld: &mut ModelLoader,
                    q: LlmTensor,
                    k: LlmTensor,
                    v: LlmTensor,
                    o: LlmTensor,
                    bid: i32|
         -> Result<ClefAttnTensors, String> {
            Ok(ClefAttnTensors {
                wq: ld.create_tensor(q, "weight", bid, &[n_embd_h, n_embd_h], 0)?.unwrap(),
                bq: ld.create_tensor(q, "bias", bid, &[n_embd_h], 0)?.unwrap(),
                wk: ld.create_tensor(k, "weight", bid, &[n_embd_h, n_embd_h], 0)?.unwrap(),
                bk: ld.create_tensor(k, "bias", bid, &[n_embd_h], 0)?.unwrap(),
                wv: ld.create_tensor(v, "weight", bid, &[n_embd_h, n_embd_h], 0)?.unwrap(),
                bv: ld.create_tensor(v, "bias", bid, &[n_embd_h], 0)?.unwrap(),
                wo: ld.create_tensor(o, "weight", bid, &[n_embd_h, n_embd_h], 0)?.unwrap(),
                bo: ld.create_tensor(o, "bias", bid, &[n_embd_h], 0)?.unwrap(),
            })
        };

        // routing blocks first (no self attention), then joint blocks
        // (clef.cpp:66-87)
        let n_head_layers = n_layer_routing + hparams.n_layer_decision as usize;
        let mut layers = Vec::with_capacity(n_head_layers);
        for il in 0..n_head_layers {
            let bid = il as i32;

            let (self_norm, self_attn, cross_norm_kv) = if il < n_layer_routing {
                (
                    None,
                    None,
                    Some(norm(ld, LlmTensor::DEC_CROSS_ATTN_NORM_KV, bid, n_embd_h)?),
                )
            } else {
                (
                    Some(norm(ld, LlmTensor::DEC_ATTN_NORM, bid, n_embd_h)?),
                    Some(attn(
                        ld,
                        LlmTensor::DEC_ATTN_Q,
                        LlmTensor::DEC_ATTN_K,
                        LlmTensor::DEC_ATTN_V,
                        LlmTensor::DEC_ATTN_OUT,
                        bid,
                    )?),
                    None,
                )
            };

            let cross_norm = norm(ld, LlmTensor::DEC_CROSS_ATTN_NORM, bid, n_embd_h)?;
            let cross_attn = attn(
                ld,
                LlmTensor::DEC_CROSS_ATTN_Q,
                LlmTensor::DEC_CROSS_ATTN_K,
                LlmTensor::DEC_CROSS_ATTN_V,
                LlmTensor::DEC_CROSS_ATTN_OUT,
                bid,
            )?;

            let ffn_norm = norm(ld, LlmTensor::DEC_FFN_NORM, bid, n_embd_h)?;
            let ffn_up = ld
                .create_tensor(LlmTensor::DEC_FFN_UP, "weight", bid, &[n_embd_h, n_ff_h], 0)?
                .unwrap();
            let ffn_up_b =
                ld.create_tensor(LlmTensor::DEC_FFN_UP, "bias", bid, &[n_ff_h], 0)?.unwrap();
            let ffn_down = ld
                .create_tensor(LlmTensor::DEC_FFN_DOWN, "weight", bid, &[n_ff_h, n_embd_h], 0)?
                .unwrap();
            let ffn_down_b = ld
                .create_tensor(LlmTensor::DEC_FFN_DOWN, "bias", bid, &[n_embd_h], 0)?
                .unwrap();

            layers.push(ClefHeadLayerTensors {
                self_norm,
                self_attn,
                cross_norm,
                cross_norm_kv,
                cross_attn,
                ffn_norm,
                ffn_up,
                ffn_up_b,
                ffn_down,
                ffn_down_b,
            });
        }

        let hidden_norm = norm(ld, LlmTensor::DECISION_HIDDEN_NORM, -1, lc.n_embd)?;
        let option_summary_norm =
            norm(ld, LlmTensor::DECISION_OPTION_SUMMARY_NORM, -1, n_embd_h)?;
        let field_norm = norm(ld, LlmTensor::DECISION_FIELD_NORM, -1, n_embd_h)?;
        let option_norm = norm(ld, LlmTensor::DECISION_OPTION_NORM, -1, n_embd_h)?;

        let proj = |ld: &mut ModelLoader, ty: LlmTensor| -> Result<TensorId, String> {
            Ok(ld
                .create_tensor(ty, "weight", -1, &[lc.n_embd, n_embd_h], 0)?
                .unwrap())
        };
        let proj_memory = proj(ld, LlmTensor::DECISION_PROJ_MEMORY)?;
        let proj_question = proj(ld, LlmTensor::DECISION_PROJ_QUESTION)?;
        let proj_option_question = proj(ld, LlmTensor::DECISION_PROJ_OPTION_QUESTION)?;
        let proj_global = proj(ld, LlmTensor::DECISION_PROJ_GLOBAL)?;
        let proj_option_context = proj(ld, LlmTensor::DECISION_PROJ_OPTION_CONTEXT)?;
        let proj_option_lexical = proj(ld, LlmTensor::DECISION_PROJ_OPTION_LEXICAL)?;

        // DECISION_SCALES carries no suffix (tn(LLM_TENSOR_DECISION_SCALES),
        // clef.cpp:91)
        let scales = ld
            .create_tensor(LlmTensor::DECISION_SCALES, "", -1, &[3], 0)?
            .unwrap();
        let type_embd = ld
            .create_tensor(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[n_embd_h, 3],
                0,
            )?
            .unwrap();

        let scorer = ld
            .create_tensor(
                LlmTensor::DECISION_SCORER,
                "weight",
                -1,
                &[4 * n_embd_h, n_embd_h],
                0,
            )?
            .unwrap();
        let scorer_b = ld
            .create_tensor(LlmTensor::DECISION_SCORER, "bias", -1, &[n_embd_h], 0)?
            .unwrap();
        let scorer_out = ld
            .create_tensor(LlmTensor::DECISION_SCORER_OUT, "weight", -1, &[n_embd_h, 1], 0)?
            .unwrap();
        let scorer_out_b = ld
            .create_tensor(LlmTensor::DECISION_SCORER_OUT, "bias", -1, &[1], 0)?
            .unwrap();

        Ok(ClefHeadTensors {
            layers,
            hidden_norm,
            option_summary_norm,
            field_norm,
            option_norm,
            proj_memory,
            proj_question,
            proj_option_question,
            proj_global,
            proj_option_context,
            proj_option_lexical,
            scales,
            type_embd,
            scorer,
            scorer_b,
            scorer_out,
            scorer_out_b,
        })
    }

    let (tok_embd, output_norm, output_b, output, cls_out): (
        TensorId,
        TensorId,
        Option<TensorId>,
        TensorId,
        Option<TensorId>,
    ) = match arch {
        // ---- models/qwen2.cpp ----
        LlmArch::QWEN2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_b = opt!(LlmTensor::OUTPUT, "bias", -1, &[lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, output_b, output.unwrap(), None)
        }

        // ---- models/llama.cpp ----
        LlmArch::LLAMA | LlmArch::LLAMA_EMBED => {
            // LLAMA_EMBED is the graph<true> co-arm (models.h:175-183) — the
            // port's build_llama_forward embed tap serves its EncoderContext
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // optional bias tensors
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                if lc.n_expert == 0 {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));

                    // optional MLP bias
                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));

                    // For Granite MoE Shared
                    if hparams.n_ff_shexp > 0 {
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, hparams.n_ff_shexp as i64]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, hparams.n_ff_shexp as i64]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[hparams.n_ff_shexp as i64, lc.n_embd]
                        ));
                    }
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/qwen3.cpp ----
        LlmArch::QWEN3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // output rerank head
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_cls_out as i64]
            );

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), cls_out)
        }

        // ---- models/gemma2.cpp ----
        LlmArch::GEMMA2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            // same as tok_embd, duplicated to allow offloading
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/gemma3.cpp ----
        LlmArch::GEMMA3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // Dense linear weights (sentence-transformers exports)
            let _dense_2 = opt!(
                LlmTensor::DENSE_2_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.dense_2_feat_out as i64]
            );
            let _dense_3 = opt!(
                LlmTensor::DENSE_3_OUT,
                "weight",
                -1,
                &[hparams.dense_3_feat_in as i64, lc.n_embd]
            );

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/phi3.cpp ----
        LlmArch::PHI3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    TENSOR_NOT_REQUIRED,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * lc.n_ff]
                ));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_long = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_LONG,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
                l.rope_short = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_SHORT,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/openai-moe.cpp (gpt-oss) ----
        LlmArch::OPENAI_MOE => {
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (required, not tied in every file but no NOT_REQUIRED flag)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                // n_head * n_rot q dims (n_rot == n_embd_head_k here)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_head * lc.n_rot,
                    lc.n_head_kv * lc.n_rot,
                    lc.n_head_kv * lc.n_rot,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * lc.n_rot, lc.n_embd]
                ));

                l.attn_sinks = Some(req!(LlmTensor::ATTN_SINKS, "weight", bid, &[lc.n_head]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_gate_inp_b = Some(req!(LlmTensor::FFN_GATE_INP, "bias", bid, &[lc.n_expert]));
                l.ffn_gate_exps_b = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "bias",
                    bid,
                    &[n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps_b = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "bias",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps_b = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "bias",
                    bid,
                    &[n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/gemma4.cpp ----
        LlmArch::GEMMA4 => {
            let n_embd_per_layer = hparams.n_embd_per_layer as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            if lc.n_embd_head_k != lc.n_embd_head_v {
                return Err("Gemma 4 requires n_embd_head_k == n_embd_head_v".to_string());
            }
            if hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa {
                return Err("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa".to_string());
            }

            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            if n_embd_per_layer > 0 {
                // the C++ TENSOR_READ_LAZY flag is a load-mode hint only (the
                // port has no lazy/virtual tensors — see module docs)
                extra.per_layer_tok_embd = Some(req!(
                    LlmTensor::PER_LAYER_TOKEN_EMBD,
                    "weight",
                    -1,
                    &[n_embd_per_layer * lc.n_layer as i64, lc.n_vocab]
                ));
                extra.per_layer_model_proj = Some(req!(
                    LlmTensor::PER_LAYER_MODEL_PROJ,
                    "weight",
                    0,
                    &[lc.n_embd, n_embd_per_layer * lc.n_layer as i64]
                ));
                extra.per_layer_proj_norm = Some(req!(
                    LlmTensor::PER_LAYER_PROJ_NORM,
                    "weight",
                    0,
                    &[n_embd_per_layer]
                ));
            }

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);

            let mut rope_freqs_flag = 0u32;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let il = i;
                let n_head = hparams.n_head(il) as i64;
                let n_embd_head = hparams.n_embd_head_k(il) as i64;
                let n_embd_k = hparams.n_embd_k_gqa(il) as i64;
                let n_embd_v = hparams.n_embd_v_gqa(il) as i64;
                let kv_flags = if hparams.has_kv(il) {
                    0
                } else {
                    TENSOR_NOT_REQUIRED
                };

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // note: use_alternative_attention (v_proj is optional, if it's
                // not present, use k_proj); the C++ TENSOR_SKIP_IF_VIRTUAL is a
                // no-op here (no virtual files)
                l.wqkv = opt!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, n_embd_head * n_head + n_embd_k + n_embd_v]
                );
                if l.wqkv.is_none() {
                    l.wq = Some(req!(
                        LlmTensor::ATTN_Q,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_head * n_head]
                    ));
                    l.wk = ld.create_tensor(
                        LlmTensor::ATTN_K,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_k],
                        kv_flags,
                    )?;
                    l.wv = opt!(LlmTensor::ATTN_V, "weight", bid, &[lc.n_embd, n_embd_v]);
                }
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_embd_head * n_head, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[n_embd_head]));
                l.attn_k_norm = ld.create_tensor(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[n_embd_head],
                    kv_flags,
                )?;
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.out_scale = opt!(LlmTensor::LAYER_OUT_SCALE, "weight", bid, &[1]);

                if !hparams.is_swa(il) {
                    // full_attention layers use rope_freqs for proportional rope
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[n_embd_head / 2],
                        rope_freqs_flag,
                    )?;
                    rope_freqs_flag = TENSOR_DUPLICATED;
                }

                // handle use_double_wide_mlp
                let n_ff_cur = hparams.n_ff(il) as i64;

                // for expert layers, we use normal FFN as shared expert (same
                // as python code)
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_cur]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_cur]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[n_ff_cur, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));

                // MoE router
                l.ffn_gate_inp = opt!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                );
                let has_expert = l.ffn_gate_inp.is_some();

                if has_expert {
                    l.ffn_gate_inp_s =
                        Some(req!(LlmTensor::FFN_GATE_INP, "scale", bid, &[lc.n_embd]));

                    l.ffn_pre_norm_2 =
                        Some(req!(LlmTensor::FFN_PRE_NORM_2, "weight", bid, &[lc.n_embd]));
                    l.ffn_post_norm_1 = Some(req!(
                        LlmTensor::FFN_POST_NORM_1,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    ));
                    l.ffn_post_norm_2 = Some(req!(
                        LlmTensor::FFN_POST_NORM_2,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    ));

                    // MoE FFN
                    l.ffn_gate_up_exps = opt!(
                        LlmTensor::FFN_GATE_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * 2, lc.n_expert]
                    );
                    if l.ffn_gate_up_exps.is_none() {
                        l.ffn_gate_exps = Some(req!(
                            LlmTensor::FFN_GATE_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                        l.ffn_up_exps = Some(req!(
                            LlmTensor::FFN_UP_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                    }

                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));

                    // per-expert scale is created by the generic `.scale` pass
                    // of llama_model::load_tensors (create_optional_scale_tensors)
                }

                // per-layer embeddings
                if n_embd_per_layer > 0 {
                    l.per_layer_inp_gate = Some(req!(
                        LlmTensor::PER_LAYER_INP_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_per_layer]
                    ));
                    l.per_layer_proj = Some(req!(
                        LlmTensor::PER_LAYER_PROJ,
                        "weight",
                        bid,
                        &[n_embd_per_layer, lc.n_embd]
                    ));
                    l.per_layer_post_norm = Some(req!(
                        LlmTensor::PER_LAYER_POST_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/lfm2moe.cpp ----
        // ---- models/lfm2.cpp:36-95 + models/lfm2moe.cpp (the two files'
        // load_arch_tensors bodies are the same loop; lfm2's
        // n_layer_dense_lead == n_layer makes every layer dense) ----
        LlmArch::LFM2 | LlmArch::LFM2MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM_LFM2, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let il = i;
                let is_moe_layer = i >= hparams.n_layer_dense_lead as usize;

                // ffn/moe is same for transformer and conv layers
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                if is_moe_layer {
                    if lc.n_expert == 0 || lc.n_expert_used == 0 {
                        return Err("lfm2moe: n_expert and n_expert_used must be > 0".to_string());
                    }
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));
                } else {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                }

                // for operator_norm
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if !hparams.is_recr(il) {
                    l.attn_q_norm = Some(req!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k]
                    ));
                    l.attn_k_norm = Some(req!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k]
                    ));
                    // GGML_ASSERT(n_embd_v_gqa == n_embd_k_gqa)
                    if hparams.n_embd_v_gqa(il) != hparams.n_embd_k_gqa(il) {
                        return Err(format!(
                            "lfm2moe: n_embd_v_gqa ({}) != n_embd_k_gqa ({})",
                            hparams.n_embd_v_gqa(il),
                            hparams.n_embd_k_gqa(il)
                        ));
                    }

                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd,
                        hparams.n_embd_k_gqa(il) as i64,
                        hparams.n_embd_v_gqa(il) as i64,
                        0,
                    )?;

                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                } else {
                    l.shortconv_conv = Some(req!(
                        LlmTensor::SHORTCONV_CONV,
                        "weight",
                        bid,
                        &[hparams.n_shortconv_l_cache as i64, lc.n_embd]
                    ));
                    l.shortconv_in_proj = Some(req!(
                        LlmTensor::SHORTCONV_INPROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, 3 * lc.n_embd]
                    ));
                    l.shortconv_out_proj = Some(req!(
                        LlmTensor::SHORTCONV_OUTPROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                }
            }

            // for LFM2-ColBert-350M
            extra.dense_2_out_layers = opt!(
                LlmTensor::DENSE_2_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_embd_out() as i64]
            );
            extra.dense_2_out_layers_b = opt!(
                LlmTensor::DENSE_2_OUT,
                "bias",
                -1,
                &[hparams.n_embd_out() as i64]
            );

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/granite-hybrid.cpp (granitehybrid) ----
        LlmArch::GRANITE_HYBRID => {
            // mamba2 Mixer SSM params (int64_t for tensor dimensions)
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = hparams.ssm_d_inner as i64;
            let d_state = hparams.ssm_d_state as i64;
            let n_ssm_head = hparams.ssm_dt_rank as i64;
            let n_group = hparams.ssm_n_group as i64;
            let d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;

            // only an expansion factor of 2 is supported for now
            if 2 * lc.n_embd != d_inner {
                return Err(format!(
                    "granitehybrid: 2 * n_embd ({}) != ssm.inner_size ({d_inner})",
                    2 * lc.n_embd
                ));
            }

            // embeddings
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed, duplicated to
            // allow offloading
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let il = i;

                // norm
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.is_recr(il) {
                    // ssm layers
                    l.ssm_in = Some(req!(
                        LlmTensor::SSM_IN,
                        "weight",
                        bid,
                        &[lc.n_embd, d_in_proj]
                    ));

                    l.ssm_conv1d = Some(req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[d_conv, d_inner + 2 * n_group * d_state]
                    ));
                    l.ssm_conv1d_b = opt!(
                        LlmTensor::SSM_CONV1D,
                        "bias",
                        bid,
                        &[d_inner + 2 * n_group * d_state]
                    );

                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[n_ssm_head]));

                    // no "weight" suffix for these
                    l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[1, n_ssm_head]));
                    l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[1, n_ssm_head]));

                    l.ssm_norm = Some(req!(
                        LlmTensor::SSM_NORM,
                        "weight",
                        bid,
                        &[d_inner / n_group, n_group]
                    ));

                    // out_proj
                    l.ssm_out = Some(req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else {
                    // attention layers (with optional bias)
                    let n_head_i = hparams.n_head(il) as i64;
                    let n_embd_k_gqa_i = hparams.n_embd_k_gqa(il) as i64;
                    let n_embd_v_gqa_i = hparams.n_embd_v_gqa(il) as i64;
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * n_head_i,
                        n_embd_k_gqa_i,
                        n_embd_v_gqa_i,
                        0,
                    )?;
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * n_head_i, lc.n_embd]
                    ));
                    l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                }

                // feed forward (w/ optional biases)
                if lc.n_expert > 0 {
                    // MoE FFN
                    l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                    let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));

                    // For Granite MoE Shared
                    if hparams.n_ff_shexp > 0 {
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, hparams.n_ff_shexp as i64]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, hparams.n_ff_shexp as i64]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[hparams.n_ff_shexp as i64, lc.n_embd]
                        ));
                    }
                } else {
                    l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                    let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/qwen35.cpp (gated delta net + optional MTP blocks) +
        // models/clef.cpp (the clef arch inherits the qwen35 loader
        // wholesale, clef.cpp:27-28) ----
        LlmArch::QWEN35 | LlmArch::CLEF => {
            // C++ `mtp_only`: the file has nextn layers but no trunk block 0
            let mtp_only =
                lc.n_layer_nextn > 0 && ld.gguf.find_tensor("blk.0.attn_norm.weight").is_none();
            let trunk_flags = if mtp_only { TENSOR_NOT_REQUIRED } else { 0 };
            // C++ `mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0`; the port has no
            // load_mtp switch and no TENSOR_SKIP (see module docs), so MTP
            // blocks are always loaded — equivalent to load_mtp = true.
            let mtp_flags = 0u32;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }
            // optional projection of the embeddings output (a7b94df2c
            // qwen35.cpp:46-49, commit a4cb4c61f)
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_embd_out() as i64]
            );
            extra.cls_out_b = opt!(
                LlmTensor::CLS_OUT,
                "bias",
                -1,
                &[hparams.n_embd_out() as i64]
            );

            // Calculate dimensions from hyperparameters (gated delta net)
            let head_k_dim = hparams.ssm_d_state as i64;
            let head_v_dim = hparams.ssm_d_state as i64;
            let n_k_heads = hparams.ssm_n_group as i64;
            let n_v_heads = hparams.ssm_dt_rank as i64;
            let key_dim = head_k_dim * n_k_heads;
            let value_dim = head_v_dim * n_v_heads;
            let conv_dim = key_dim * 2 + value_dim;

            let load_block_trunk = |l: &mut LayerTensors,
                                    ld: &mut ModelLoader,
                                    il: usize,
                                    flags: u32|
             -> Result<(), String> {
                let bid = il as i32;

                l.attn_norm =
                    ld.create_tensor(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], flags)?;
                l.attn_post_norm = ld.create_tensor(
                    LlmTensor::ATTN_POST_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd],
                    flags,
                )?;

                if !hparams.is_recr(il) {
                    // Attention layers
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * lc.n_head * 2,
                        lc.n_embd_k_gqa,
                        lc.n_embd_v_gqa,
                        flags,
                    )?;
                    l.wo = ld.create_tensor(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                        flags,
                    )?;

                    // Q/K normalization for attention layers
                    l.attn_q_norm = ld.create_tensor(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        flags,
                    )?;
                    l.attn_k_norm = ld.create_tensor(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        flags,
                    )?;
                } else {
                    // Linear attention (gated delta net) specific tensors
                    l.wqkv = ld.create_tensor(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, key_dim * 2 + value_dim],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.wqkv_gate = ld.create_tensor(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, value_dim],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.ssm_conv1d = ld.create_tensor(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[hparams.ssm_d_conv as i64, conv_dim],
                        flags,
                    )?;
                    l.ssm_dt_b = ld.create_tensor(
                        LlmTensor::SSM_DT,
                        "bias",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        flags,
                    )?;
                    // SSM_A_NOSCAN carries no suffix
                    l.ssm_a = ld.create_tensor(
                        LlmTensor::SSM_A_NOSCAN,
                        "",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        flags,
                    )?;
                    l.ssm_beta = ld.create_tensor(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads],
                        flags,
                    )?;
                    l.ssm_alpha = ld.create_tensor(
                        LlmTensor::SSM_ALPHA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads],
                        flags,
                    )?;
                    l.ssm_norm =
                        ld.create_tensor(LlmTensor::SSM_NORM, "weight", bid, &[head_v_dim], flags)?;
                    l.ssm_out = ld.create_tensor(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[value_dim, lc.n_embd],
                        flags,
                    )?;
                }

                l.ffn_gate = ld.create_tensor(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff],
                    flags,
                )?;
                l.ffn_down = ld.create_tensor(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd],
                    flags,
                )?;
                l.ffn_up = ld.create_tensor(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff],
                    flags,
                )?;
                Ok(())
            };

            for (i, l) in layers.iter_mut().enumerate().take(lc.n_layer) {
                load_block_trunk(l, ld, i, trunk_flags)?;
            }

            // MTP blocks look like a full-attention Qwen3.5 decoder block
            for (i, l) in layers.iter_mut().enumerate().skip(lc.n_layer) {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head * 2,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    mtp_flags,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                // NextN-specific tensors that define the MTP block
                l.nextn.eh_proj = Some(req!(
                    LlmTensor::NEXTN_EH_PROJ,
                    "weight",
                    bid,
                    &[2 * lc.n_embd, lc.n_embd]
                ));
                l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                l.nextn.embed_tokens = ld.create_tensor(
                    LlmTensor::NEXTN_EMBED_TOKENS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_vocab],
                    mtp_flags | TENSOR_NOT_REQUIRED,
                )?;
                l.nextn.shared_head_head = ld.create_tensor(
                    LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_vocab],
                    mtp_flags | TENSOR_NOT_REQUIRED,
                )?;
                l.nextn.shared_head_norm = ld.create_tensor(
                    LlmTensor::NEXTN_SHARED_HEAD_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd],
                    mtp_flags | TENSOR_NOT_REQUIRED,
                )?;
            }

            // the clef decision head (a7b94df2c clef.cpp:30-123) — loads
            // after the trunk it scores
            if arch == LlmArch::CLEF {
                extra.clef_head = Some(load_clef_head(hparams, ld, &lc)?);
            }

            (tok_embd, output_norm, None, output.unwrap(), cls_out)
        }

        // ---- models/gpt2.cpp:15-52 ----
        LlmArch::GPT2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // learned absolute position embedding (gpt2.cpp:19); the only
            // non-rope arch of the batch — its graph gathers rows from it
            let pos_embd = req!(
                LlmTensor::POS_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_ctx_train]
            );

            // output (gpt2.cpp:22-24) — output.weight is TENSOR_NOT_REQUIRED
            // and falls back to the token embedding (:27-29)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // fused qkv + fused bias (gpt2.cpp:37-38)
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wqkv_b = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.position_embd = Some(pos_embd);
            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/phi2.cpp:13-41 ----
        LlmArch::PHI2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (phi2.cpp:19-22) — unlike every other arch of this batch,
            // `output.weight` AND `output.bias` are both required (no tied-head
            // fallback in C)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_b = req!(LlmTensor::OUTPUT, "bias", -1, &[lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — phi2.cpp:30
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, Some(output_b), output, None)
        }

        // ---- models/starcoder2.cpp:16-53 ----
        LlmArch::STARCODER2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (starcoder2.cpp:22-29) — the tied-head fallback is the
            // usual TENSOR_DUPLICATED request on token_embd (gpt2/exaone style)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — starcoder2.cpp:37
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // the C comment says "optional" but the flags argument is 0 —
                // required, like wo (starcoder2.cpp:41)
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/command-r.cpp:13-40 ----
        LlmArch::COMMAND_R => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (command-r.cpp:19-21): no output.weight in the file — the
            // token embedding is requested as TENSOR_DUPLICATED unconditionally
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // command-r.cpp:28-31 — per-head q/k LayerNorm, only for the
                // 35B-class files (n_layer >= 64). Shapes are the *model*
                // geometry: q {n_embd_head_k, n_head}, k {n_embd_head_k,
                // n_head_kv} — the graph norms the unreshaped [n_embd, T] Q/K
                // with this broadcast weight (command-r.cpp:79-89).
                if lc.n_layer >= 64 {
                    l.attn_q_norm = Some(req!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k, lc.n_head]
                    ));
                    l.attn_k_norm = Some(req!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k, lc.n_head_kv]
                    ));
                }

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — command-r.cpp:33
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/gptneox.cpp:54-87 ----
        LlmArch::GPTNEOX => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (gptneox.cpp:58-60) — all three required
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // fused qkv + fused bias (gptneox.cpp:66-67)
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wqkv_b = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/olmo2.cpp:26-49 ----
        LlmArch::OLMO2 => {
            // olmo2.cpp:29 — `const int64_t n_embd_head = n_embd / n_head`,
            // i.e. the raw quotient (NOT the n_embd_head_k override)
            let n_embd_head = lc.n_embd / lc.n_head;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (olmo2.cpp:32-33) — both required, no tied fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // note: no attn_norm — olmo2 normalizes the attention *output*
                // (attn_post_norm) instead (olmo2.cpp:36-40)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // q norm is over the full unreshaped n_embd; k norm over the
                // flat n_embd_k_gqa = n_head_kv * n_embd_head (olmo2.cpp:38-39)
                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_head_kv * n_embd_head]
                ));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/codeshell.cpp:12-46 ----
        LlmArch::CODESHELL => {
            // codeshell.cpp:15-20 — the only arch of the batch whose tie points
            // the *other* way: `token_embd.weight` is optional and, when absent,
            // the token embedding becomes `output.weight` (TENSOR_DUPLICATED);
            // `output.weight` itself is required right after (:25)
            let mut tok_embd = opt!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            if tok_embd.is_none() {
                tok_embd = ld.create_tensor(
                    LlmTensor::OUTPUT,
                    "weight",
                    -1,
                    &[lc.n_embd, lc.n_vocab],
                    TENSOR_DUPLICATED,
                )?;
            }

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — codeshell.cpp:33
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd.unwrap(), output_norm, None, output, None)
        }

        // ---- models/orion.cpp:12-37 ----
        LlmArch::ORION => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — orion.cpp:27
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                // no attn_output.bias in an orion file (orion.cpp:28)
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/olmo.cpp:15-37 ----
        LlmArch::OLMO => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // olmo.cpp:21-25 — optional head with the tok_embd tie fallback
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // note: olmo creates no norm tensors at all — its graph calls
            // build_norm with NULL weight and NULL bias at every site
            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — olmo.cpp:30
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            // olmo creates no output_norm tensor either (olmo.cpp:15-37 stops
            // after the head): `LlamaModel::output_norm` is a decode-path
            // member the olmo builder never reads — the graph calls
            // `build_norm(cur, NULL, NULL, LLM_NORM)` — so it is set to the lm
            // head to keep the struct total, the same convention the BERT arm
            // documents below.
            let output = output.unwrap();
            (tok_embd, output, None, output, None)
        }

        // ---- models/xverse.cpp:14-35 ----
        LlmArch::XVERSE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // xverse.cpp:19-20 — both required, no tied fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — xverse.cpp:27
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/internlm2.cpp:13-38 ----
        LlmArch::INTERNLM2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // internlm2.cpp:21 — `output.weight` is *required* here (the C
            // comments out the qkv fuse and drops the tied fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) — internlm2.cpp:29
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/exaone.cpp:12-40 ----
        LlmArch::EXAONE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // exaone.cpp:19-24 — optional head with the tok_embd tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head,
                // n_embd_k_gqa, n_embd_v_gqa, 0) — exaone.cpp:31
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // exaone.cpp:35 — rope freq factors, {n_rot/2}, created on layer
                // 0 and duplicated everywhere else
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_freqs = ld.create_tensor(
                    LlmTensor::ROPE_FREQS,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/gemma.cpp:13-35 ----
        LlmArch::GEMMA => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // gemma.cpp:19-20 — no output_norm bias, and the head is always the
            // duplicated token embedding
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head,
                // n_embd_k_gqa, n_embd_v_gqa, 0) — gemma.cpp:27
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/falcon.cpp:13-44 ----
        LlmArch::FALCON => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // falcon.cpp:19-27 — biased output norm, optional head with the
            // tok_embd tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // falcon.cpp:35-36 — the 40B second norm (both optional)
                l.attn_norm_2 = opt!(LlmTensor::ATTN_NORM_2, "weight", bid, &[lc.n_embd]);
                l.attn_norm_2_b = opt!(LlmTensor::ATTN_NORM_2, "bias", bid, &[lc.n_embd]);

                // fused qkv, no bias (falcon.cpp:38)
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                // falcon has no ffn_norm and no ffn_gate: the FFN runs off the
                // attention norm (falcon.cpp:41-42)
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/baichuan.cpp:17-45 ----
        LlmArch::BAICHUAN => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // baichuan.cpp:18-21 — both required, no tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/bloom.cpp:21-67 ----
        LlmArch::BLOOM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // bloom.cpp:23-24 — REPEATING class, hence bid 0 (like bert)
            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                // bloom.cpp:35 — the fused bias is required
                l.wqkv_b = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);
            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/mpt.cpp:17-53 ----
        LlmArch::MPT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let pos_embd = opt!(
                LlmTensor::POS_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_ctx_train]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = opt!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = opt!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]);
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wqkv_b = opt!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                );
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = opt!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]);
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                // full-width Q/K norms (:44-49; the C's TENSOR_SKIP_IF_VIRTUAL
                // half of the q_norm request is dropped with the rest of the
                // SKIP support)
                l.attn_q_norm = opt!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd]);
                l.attn_q_norm_b = opt!(LlmTensor::ATTN_Q_NORM, "bias", bid, &[lc.n_embd]);
                l.attn_k_norm = opt!(LlmTensor::ATTN_K_NORM, "weight", bid, &[lc.n_embd]);
                l.attn_k_norm_b = opt!(LlmTensor::ATTN_K_NORM, "bias", bid, &[lc.n_embd]);
                // AWQ ScaleActivation (:52)
                l.ffn_act = opt!(LlmTensor::FFN_ACT, "scales", bid, &[lc.n_ff]);
            }

            extra.position_embd = pos_embd;
            extra.output_norm_b = output_norm_b;
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/starcoder.cpp:17-60 ----
        LlmArch::STARCODER => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // :20 — required
            let pos_embd = req!(
                LlmTensor::POS_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_ctx_train]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wqkv_b = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.position_embd = Some(pos_embd);
            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/refact.cpp:16-76 (dense branch; MoE files rejected) ----
        LlmArch::REFACT => {
            if lc.n_expert != 0 {
                return Err(
                    "refact MoE files (n_expert > 0) are not ported — dense refact only \
                     (refact.cpp:53-70)"
                        .into(),
                );
            }
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                // refact.cpp:40-51 — the rope-factor pair only differs by the
                // LONGROPE scaling type; neither is read by the graph (rope
                // type NONE), but the loader consumes the tensors either way
                // (REPEATING class: one file tensor, DUPLICATED re-requests)
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == crate::hparams::LlamaRopeScalingType::LONGROPE
                {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/plamo.cpp:12-36 ----
        LlmArch::PLAMO => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // :16-17 — both required, no tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // no ffn_norm tensor at all (plamo.cpp:34-36)
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/stablelm.cpp:13-58 ----
        LlmArch::STABLELM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // :21-22 — the bias is created first in C; both required
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // StableLM 2 12B per-head norms (:36-37): [n_embd_head_k, n_head]
                l.attn_q_norm = opt!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head]
                );
                l.attn_k_norm = opt!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head_kv]
                );
                // optional FFN norm — absent in the parallel-residual files
                l.ffn_norm = opt!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]);
                l.ffn_norm_b = opt!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]);
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/granite.cpp:79-150 (dense; minicpm's tensor set below
        //      is identical — models/minicpm.cpp:29-96) ----
        LlmArch::GRANITE | LlmArch::MINICPM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                if l.wqkv.is_some() {
                    // create_tensor_qkv prefers a fused tensor when present
                    // (llama-model.cpp:3287) and the dense granite builder only
                    // spells the separate branch — no real granite/minicpm file
                    // carries a fused qkv, refuse it explicitly
                    return Err(
                        "fused 'attn_qkv.weight' granite/minicpm files are not supported by \
                         the port's dense granite builder (granite.cpp:97-101 expects separate \
                         q/k/v)"
                            .into(),
                    );
                }
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == crate::hparams::LlamaRopeScalingType::LONGROPE
                {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                if lc.n_expert == 0 {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                } else {
                    // the MoE branch granite.cpp's loader also accepts (dense
                    // granite files with experts; the builder reuses the
                    // hybrid's `build_moe_ffn_silu`)
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));
                    if hparams.n_ff_shexp > 0 {
                        let n_ff_shexp = hparams.n_ff_shexp as i64;
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd]
                        ));
                    }
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- arch batch 4 (2026-09-28): the MoE family + the cheap dense
        // archs that fall out of it (smollm3 / seed-oss / openelm) ----

        // ---- models/qwen2moe.cpp:16-59 ----
        LlmArch::QWEN2MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (required — qwen2moe has no tie fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0 for QWEN2MOE".into());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0 for QWEN2MOE".into());
                }

                // MoE branch (qwen2moe.cpp:45)
                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                // Shared expert branch (qwen2moe.cpp:52)
                let n_ff_shexp = if hparams.n_ff_shexp != 0 {
                    hparams.n_ff_shexp as i64
                } else {
                    lc.n_ff
                };

                l.ffn_gate_inp_shexp = Some(req!(
                    LlmTensor::FFN_GATE_INP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                l.ffn_gate_shexp = Some(req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
                l.ffn_down_shexp = Some(req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                ));
                l.ffn_up_shexp = Some(req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/qwen3moe.cpp:14-56 ----
        LlmArch::QWEN3MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0 for QWEN3MOE".into());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0 for QWEN3MOE".into());
                }

                // MoE branch (qwen3moe.cpp:50)
                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/granite-moe.cpp:21-79 (same tensor set as the granite
        // dense arm's MoE branch — granite.cpp's loader, which that arm
        // already ports 1:1) ----
        LlmArch::GRANITE_MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                if l.wqkv.is_some() {
                    // the builder only spells the separate branch (same
                    // reasoning as the granite dense arm above)
                    return Err(
                        "fused 'attn_qkv.weight' granite-moe files are not supported by \
                         the port's granite builder (granite-moe.cpp:40 expects separate \
                         q/k/v)"
                            .into(),
                    );
                }
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // optional bias tensors (granite-moe.cpp:44)
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == crate::hparams::LlamaRopeScalingType::LONGROPE
                {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                if lc.n_expert == 0 {
                    // granite-moe.cpp:56-64 — the dense branch of the shared
                    // loader; a real granite-moe file always has experts, but
                    // the loader accepts dense files too
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));

                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                } else {
                    // MoE branch (granite-moe.cpp:66-76 — note n_ff, not
                    // n_ff_exp: the experts' intermediate dim is the *dense*
                    // feed_forward_length)
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));

                    // For Granite MoE Shared (granite-moe.cpp:72)
                    if hparams.n_ff_shexp > 0 {
                        let n_ff_shexp = hparams.n_ff_shexp as i64;
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd]
                        ));
                    }
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/phimoe.cpp:12-46 ----
        LlmArch::PHIMOE => {
            let n_embd_head = lc.n_embd / lc.n_head;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (all four tensors required, including the biases)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_b = req!(LlmTensor::OUTPUT, "bias", -1, &[lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_long = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_LONG,
                    "weight",
                    bid,
                    &[n_embd_head / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
                l.rope_short = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_SHORT,
                    "weight",
                    bid,
                    &[n_embd_head / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
            }
            // output_norm_b travels through `extra` (ModelTensors) — see the
            // phi2/stablelm arms for the same pattern
            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, Some(output_b), output, None)
        }

        // ---- models/arctic.cpp:16-50 ----
        LlmArch::ARCTIC => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // the dense FFN is square [n_embd, n_embd] (arctic.cpp:40-42)
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_norm_exps = Some(req!(LlmTensor::FFN_NORM_EXPS, "weight", bid, &[lc.n_embd]));
                // `false` here is C++'s int 0 — required (arctic.cpp:46)
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/olmoe.cpp:12-47 ----
        LlmArch::OLMOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (required)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // full-width Q/K norms (olmoe.cpp:28-29 — [n_embd], not
                // [n_embd_head]; the graph norms before the 3D reshape)
                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_k_norm = Some(req!(LlmTensor::ATTN_K_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0".into());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0".into());
                }

                // MoE branch (olmoe.cpp:42)
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/ernie4-5.cpp:23-70 (the LLM_ARCH_ERNIE4_5_MOE branch of
        // the shared ernie4-5/ernie4-5-moe loader) ----
        LlmArch::ERNIE4_5_MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // optional bias tensors (ernie4-5.cpp:45)
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) >= hparams.n_layer_dense_lead {
                    // MoE layers (ernie4-5.cpp:49-63)
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;

                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // Shared expert (if present) (ernie4-5.cpp:59)
                    if hparams.n_ff_shexp > 0 {
                        let n_ff_shexp = hparams.n_ff_shexp as i64;
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                    }
                } else {
                    // Dense layers (ernie4-5.cpp:64-68)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/smollm3.cpp:13-40 ----
        LlmArch::SMOLLM3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/seed-oss.cpp:12-43 ----
        LlmArch::SEED_OSS => {
            let head_dim = hparams.n_embd_head_k(0) as i64;
            let n_qo_dim = lc.n_head * head_dim;
            let n_kv_dim = lc.n_head_kv * head_dim;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                create_tensor_qkv(l, ld, bid, lc.n_embd, n_qo_dim, n_kv_dim, n_kv_dim, 0)?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_qo_dim, lc.n_embd]
                ));

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/openelm.cpp:15-44 ----
        LlmArch::OPENELM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            // init output from the input tok embed (unconditional,
            // openelm.cpp:22-23)
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // per-layer geometry (openelm.cpp:26-28): head counts and FFN
                // width come from the GGUF arrays
                let n_head_i = hparams.n_head(i) as i64;
                let n_head_kv_i = hparams.n_head_kv(i) as i64;
                let n_head_qkv = 2 * n_head_kv_i + n_head_i;
                let n_ff_i = hparams.n_ff(i) as i64;
                let hd_k = hparams.n_embd_head_k(i) as i64;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // the fused qkv packs [n_head Q | n_head_kv K | n_head_kv V]
                // heads (openelm.cpp:34) — created directly, not via
                // create_tensor_qkv (which would expect the [q|k|v] row split
                // of the other fused archs)
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, n_head_qkv * hd_k]
                ));
                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[hd_k]));
                l.attn_k_norm = Some(req!(LlmTensor::ATTN_K_NORM, "weight", bid, &[hd_k]));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_head_i * hd_k, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_i]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[n_ff_i, lc.n_embd]
                ));
                l.ffn_up = Some(req!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, n_ff_i]));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- arch batch 5 (2026-09-24): the mamba family ----

        // ---- models/mamba.cpp:35-114 (mamba1; every layer recurrent) ----
        LlmArch::MAMBA => {
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = hparams.ssm_d_inner as i64;
            let d_state = hparams.ssm_d_state as i64;
            let dt_rank = hparams.ssm_dt_rank as i64;

            // only an expansion factor of 2 is supported for now (mamba.cpp:45)
            if 2 * lc.n_embd != d_inner {
                return Err("only an expansion factor of 2 is supported for now".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed, duplicated to
            // allow offloading (mamba.cpp:56-59)
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // norm
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.ssm_in = Some(req!(
                    LlmTensor::SSM_IN,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * d_inner]
                ));

                l.ssm_conv1d = Some(req!(
                    LlmTensor::SSM_CONV1D,
                    "weight",
                    bid,
                    &[d_conv, d_inner]
                ));
                l.ssm_conv1d_b = Some(req!(LlmTensor::SSM_CONV1D, "bias", bid, &[d_inner]));

                l.ssm_x = Some(req!(
                    LlmTensor::SSM_X,
                    "weight",
                    bid,
                    &[d_inner, dt_rank + 2 * d_state]
                ));

                l.ssm_dt = Some(req!(LlmTensor::SSM_DT, "weight", bid, &[dt_rank, d_inner]));
                l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[d_inner]));

                // no "weight" suffix for these (mamba.cpp:100-101)
                l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[d_state, d_inner]));
                l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[d_inner]));

                // out_proj
                l.ssm_out = Some(req!(
                    LlmTensor::SSM_OUT,
                    "weight",
                    bid,
                    &[d_inner, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/mamba2.cpp:35-90 (every layer recurrent) ----
        LlmArch::MAMBA2 => {
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = hparams.ssm_d_inner as i64;
            let d_state = hparams.ssm_d_state as i64;
            let n_group = hparams.ssm_n_group as i64;
            let dt_rank = hparams.ssm_dt_rank as i64;

            let conv_dim = d_inner + 2 * n_group * d_state;
            let d_in_proj = d_inner + conv_dim + dt_rank;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // norm
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.ssm_in = Some(req!(
                    LlmTensor::SSM_IN,
                    "weight",
                    bid,
                    &[lc.n_embd, d_in_proj]
                ));

                l.ssm_conv1d = Some(req!(
                    LlmTensor::SSM_CONV1D,
                    "weight",
                    bid,
                    &[d_conv, d_inner + 2 * n_group * d_state]
                ));
                // mamba2.cpp:70 — the conv bias is REQUIRED here (granite's is
                // optional)
                l.ssm_conv1d_b = Some(req!(
                    LlmTensor::SSM_CONV1D,
                    "bias",
                    bid,
                    &[d_inner + 2 * n_group * d_state]
                ));

                l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[dt_rank]));

                // no "weight" suffix for these (mamba2.cpp:80-81)
                l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[1, dt_rank]));
                l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[1, dt_rank]));

                l.ssm_norm = Some(req!(
                    LlmTensor::SSM_NORM,
                    "weight",
                    bid,
                    &[d_inner / n_group, n_group]
                ));

                // out_proj
                l.ssm_out = Some(req!(
                    LlmTensor::SSM_OUT,
                    "weight",
                    bid,
                    &[d_inner, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/jamba.cpp:32-133 (hybrid: mamba1 + rope-less attention,
        // dense FFN | MoE) ----
        LlmArch::JAMBA => {
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = hparams.ssm_d_inner as i64;
            let d_state = hparams.ssm_d_state as i64;
            let dt_rank = hparams.ssm_dt_rank as i64;

            // only an expansion factor of 2 is supported for now
            // (jamba.cpp:45, GGML_ASSERT)
            if 2 * lc.n_embd != d_inner {
                return Err("jamba: 2 * n_embd != ssm.inner_size".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let n_head_kv = hparams.n_head_kv(i);
                let n_embd_gqa = hparams.n_embd_v_gqa(i) as i64;

                // norm
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if n_head_kv == 0 {
                    // Mamba layer (jamba.cpp:67-108)
                    l.ssm_in = Some(req!(
                        LlmTensor::SSM_IN,
                        "weight",
                        bid,
                        &[lc.n_embd, 2 * d_inner]
                    ));

                    l.ssm_conv1d = Some(req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[d_conv, d_inner]
                    ));
                    l.ssm_conv1d_b = Some(req!(LlmTensor::SSM_CONV1D, "bias", bid, &[d_inner]));

                    l.ssm_x = Some(req!(
                        LlmTensor::SSM_X,
                        "weight",
                        bid,
                        &[d_inner, dt_rank + 2 * d_state]
                    ));

                    l.ssm_dt_norm = Some(req!(LlmTensor::SSM_DT_NORM, "weight", bid, &[dt_rank]));

                    l.ssm_dt = Some(req!(LlmTensor::SSM_DT, "weight", bid, &[dt_rank, d_inner]));
                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[d_inner]));

                    l.ssm_b_norm = Some(req!(LlmTensor::SSM_B_NORM, "weight", bid, &[d_state]));
                    l.ssm_c_norm = Some(req!(LlmTensor::SSM_C_NORM, "weight", bid, &[d_state]));

                    // no "weight" suffix for these (jamba.cpp:100-101)
                    l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[d_state, d_inner]));
                    l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[d_inner]));

                    // out_proj
                    l.ssm_out = Some(req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else {
                    // Attention layers (jamba.cpp:104-108)
                    create_tensor_qkv(l, ld, bid, lc.n_embd, lc.n_embd, n_embd_gqa, n_embd_gqa, 0)?;
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                }

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = opt!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                );

                if l.ffn_gate_inp.is_some() {
                    // MoE (jamba.cpp:113-122)
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));
                } else {
                    // FFN (no MoE) (jamba.cpp:123-128)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/nemotron-h.cpp:49-181 (hybrid mamba2 / attention / FFN;
        // the MTP head below is the no-nextn slice). NEMOTRON_H_MOE reuses
        // this loader wholesale (models.h:1539-1543 — `llama_model_nemotron_
        // h_moe` inherits load_arch_hparams/load_arch_tensors; only its
        // graph_mtp, documented-skip PARITY batch 5 §5, differs) ----
        LlmArch::NEMOTRON_H | LlmArch::NEMOTRON_H_MOE => {
            // mamba2 Mixer SSM params (int64_t for tensor dimensions)
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = hparams.ssm_d_inner as i64;
            let d_state = hparams.ssm_d_state as i64;
            let n_ssm_head = hparams.ssm_dt_rank as i64;
            let n_group = hparams.ssm_n_group as i64;
            let d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head;
            let moe_n_embd = if hparams.moe_latent_size > 0 {
                hparams.moe_latent_size as i64
            } else {
                lc.n_embd
            };

            // embeddings
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // the port has no load_mtp switch (no TENSOR_SKIP, see module
            // docs): the C trunk_flags/mtp_flags are 0 for a normally-loaded
            // file, so every req!/opt! below already carries the right flag
            // and the mtp-only / skip-mtp shapes cannot occur
            for (i, l) in layers.iter_mut().enumerate().take(lc.n_layer) {
                let bid = i as i32;

                // all blocks use the attn norm (nemotron-h.cpp:83)
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.is_recr(i) {
                    // ssm layers (nemotron-h.cpp:86-101)
                    l.ssm_in = Some(req!(
                        LlmTensor::SSM_IN,
                        "weight",
                        bid,
                        &[lc.n_embd, d_in_proj]
                    ));

                    l.ssm_conv1d = Some(req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[d_conv, d_inner + 2 * n_group * d_state]
                    ));
                    l.ssm_conv1d_b = opt!(
                        LlmTensor::SSM_CONV1D,
                        "bias",
                        bid,
                        &[d_inner + 2 * n_group * d_state]
                    );

                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[n_ssm_head]));

                    // no "weight" suffix for these (nemotron-h.cpp:95-96)
                    l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[1, n_ssm_head]));
                    l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[1, n_ssm_head]));

                    l.ssm_norm = Some(req!(
                        LlmTensor::SSM_NORM,
                        "weight",
                        bid,
                        &[d_inner / n_group, n_group]
                    ));

                    // out_proj
                    l.ssm_out = Some(req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else if hparams.n_ff(i) == 0 {
                    // attention layers (with optional bias) (nemotron-h.cpp:103-109)
                    let n_head_i = hparams.n_head(i) as i64;
                    let n_embd_k_gqa_i = hparams.n_embd_k_gqa(i) as i64;
                    let n_embd_v_gqa_i = hparams.n_embd_v_gqa(i) as i64;
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * n_head_i,
                        n_embd_k_gqa_i,
                        n_embd_v_gqa_i,
                        0,
                    )?;
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * n_head_i, lc.n_embd]
                    ));
                    l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                } else if lc.n_expert != 0 {
                    // MoE layers (nemotron-h.cpp:111-130). Per-layer n_ff_exp
                    // with the n_ff/n_expert_used fallback (:112-115)
                    let n_ff_exp_i = if hparams.n_ff_exp(i) > 0 {
                        hparams.n_ff_exp(i) as i64
                    } else {
                        hparams.n_ff(i) as i64 / hparams.n_expert_used(i) as i64
                    };
                    let n_ff_shexp = hparams.n_ff_shexp as i64;

                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));

                    // optional latent projections (:122-123)
                    l.ffn_latent_down = opt!(
                        LlmTensor::FFN_LATENT_DOWN,
                        "weight",
                        bid,
                        &[lc.n_embd, moe_n_embd]
                    );
                    l.ffn_latent_up = opt!(
                        LlmTensor::FFN_LATENT_UP,
                        "weight",
                        bid,
                        &[moe_n_embd, lc.n_embd]
                    );

                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp_i, moe_n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[moe_n_embd, n_ff_exp_i, lc.n_expert]
                    ));

                    // Shared expert branch (:129-130) — n_ff_shexp comes from
                    // `expert_shared_feed_forward_length` (required nonzero for
                    // a MoE file to satisfy the {n_ff_shexp, n_embd} shape)
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    ));
                } else {
                    // mlp layers (nemotron-h.cpp:133-138)
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[hparams.n_ff(i) as i64, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, hparams.n_ff(i) as i64]
                    ));
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[hparams.n_ff(i) as i64]);
                }
            }

            // NextN/MTP draft head (nemotron-h.cpp:144-181): each predict
            // layer folds an attention sub-layer and a MoE sub-layer into a
            // single trailing block. Ported 1:1 (the port always "loads MTP",
            // like qwen35's nextn handling; mtp_flags == 0, and the C
            // `mtp_flags | TENSOR_NOT_REQUIRED` reads are plain opt!s here).
            for i in lc.n_layer..lc.n_layer_all {
                let l = &mut layers[i];
                let bid = i as i32;

                let n_head_i = hparams.n_head(i) as i64;
                let n_embd_k_gqa_i = hparams.n_embd_k_gqa(i) as i64;
                let n_embd_v_gqa_i = hparams.n_embd_v_gqa(i) as i64;
                let n_expert_used_i = hparams.n_expert_used(i) as i64;
                let n_ff_exp_i = hparams.n_ff_exp(i) as i64;
                if n_ff_exp_i == 0 && n_expert_used_i == 0 {
                    return Err(format!(
                        "load_arch_tensors: layer {i} declares neither expert_feed_forward_length nor expert_used_count, cannot determine the expert FFN size"
                    ));
                }
                let n_ff_exp = if n_ff_exp_i > 0 {
                    n_ff_exp_i
                } else {
                    lc.n_ff / n_expert_used_i
                };
                let n_ff_shexp = hparams.n_ff_shexp as i64;

                // NextN input-fusion tensors (:160-163)
                l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                l.nextn.eh_proj = Some(req!(
                    LlmTensor::NEXTN_EH_PROJ,
                    "weight",
                    bid,
                    &[2 * lc.n_embd, lc.n_embd]
                ));
                l.nextn.shared_head_norm = Some(req!(
                    LlmTensor::NEXTN_SHARED_HEAD_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));

                // attention sub-layer (:166-169)
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * n_head_i,
                    n_embd_k_gqa_i,
                    n_embd_v_gqa_i,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * n_head_i, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                // MoE sub-layer (:172-180)
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_exp_probs_b = Some(req!(
                    LlmTensor::FFN_EXP_PROBS_B,
                    "bias",
                    bid,
                    &[lc.n_expert]
                ));
                l.ffn_latent_down = opt!(
                    LlmTensor::FFN_LATENT_DOWN,
                    "weight",
                    bid,
                    &[lc.n_embd, moe_n_embd]
                );
                l.ffn_latent_up = opt!(
                    LlmTensor::FFN_LATENT_UP,
                    "weight",
                    bid,
                    &[moe_n_embd, lc.n_embd]
                );
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, moe_n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[moe_n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_shexp = Some(req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                ));
                l.ffn_up_shexp = Some(req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/bert.cpp:23-62 (LLM_ARCH_BERT branch) ----
        LlmArch::BERT => {
            // bert.cpp:26-28: "model needs to define token type count"
            let n_token_types = ld.token_type_count();
            if n_token_types == 0 {
                return Err(format!(
                    "{} model needs to define token type count",
                    arch.name()
                ));
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let type_embd = opt!(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[lc.n_embd, n_token_types as i64]
            );

            let pos_embd = req!(
                LlmTensor::POS_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_ctx_train]
            );

            // classification head (RANK pooling only; absent in bge-m3)
            let cls = opt!(LlmTensor::CLS, "weight", -1, &[lc.n_embd, lc.n_embd]);
            let cls_b = opt!(LlmTensor::CLS, "bias", -1, &[lc.n_embd]);
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_cls_out as i64]
            );
            let cls_out_b = opt!(LlmTensor::CLS_OUT, "bias", -1, &[hparams.n_cls_out as i64]);

            // bert.cpp:40-41 — REPEATING layer class, hence bid 0, like C++
            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0)
                // — bert.cpp:46 (fused first, then the separate fallback)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_out_norm_b = Some(req!(LlmTensor::ATTN_OUT_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);

                l.layer_out_norm =
                    Some(req!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.layer_out_norm_b =
                    Some(req!(LlmTensor::LAYER_OUT_NORM, "bias", bid, &[lc.n_embd]));
            }

            extra.token_types = type_embd;
            extra.position_embd = Some(pos_embd);
            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);
            extra.cls = cls;
            extra.cls_b = cls_b;
            extra.cls_out_b = cls_out_b;

            // `LlamaModel::output_norm` / `output` are members of the *decode*
            // path only; a BERT file has neither a `output_norm.weight` nor an
            // `output.weight` (bert.cpp creates `token_embd_norm` and stops at
            // `result_embd`, bert.cpp:40/218). They are set to the only
            // model-level norm / the duplicated token embedding so the struct
            // stays total; the encoder builder reads `token_embd_norm` and the
            // per-layer `layer_out_norm` instead.
            (
                tok_embd,
                tok_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                cls_out,
            )
        }

        // ---- models/jina-bert-v2.cpp:14-61 (LLM_ARCH_JINA_BERT_V2) ----
        LlmArch::JINA_BERT_V2 => {
            // no token-type-count guard here (unlike bert.cpp:26-28): the
            // required type_embd below fails on its own when the count is 0
            let n_token_types = ld.token_type_count();

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let type_embd = req!(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[lc.n_embd, n_token_types as i64]
            );

            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            // jina-bert-v2.cpp:23-24 — cls {n_embd, 1} / cls_b {1}, the
            // reranker head of the RANK-pooled files
            let cls = opt!(LlmTensor::CLS, "weight", -1, &[lc.n_embd, 1]);
            let cls_b = opt!(LlmTensor::CLS, "bias", -1, &[1]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                // jina-bert-v2.cpp:30-34 — the FULL-width q/k norms (the graph
                // reshapes Q/K to 2d before norming, bert.cpp:109-123)
                l.attn_q_norm = opt!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd]);
                l.attn_q_norm_b = opt!(LlmTensor::ATTN_Q_NORM, "bias", bid, &[lc.n_embd]);
                l.attn_k_norm = opt!(LlmTensor::ATTN_K_NORM, "weight", bid, &[lc.n_embd]);
                l.attn_k_norm_b = opt!(LlmTensor::ATTN_K_NORM, "bias", bid, &[lc.n_embd]);

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                // wo_b REQUIRED (jina-bert-v2.cpp:37)
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_out_norm_b = Some(req!(LlmTensor::ATTN_OUT_NORM, "bias", bid, &[lc.n_embd]));

                l.attn_norm_2 = opt!(LlmTensor::ATTN_NORM_2, "weight", bid, &[lc.n_embd]);
                l.attn_norm_2_b = opt!(LlmTensor::ATTN_NORM_2, "bias", bid, &[lc.n_embd]);

                l.ffn_gate = opt!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                );

                // jina-bert-v2.cpp:47-53 — probe ffn_up's width first: the
                // gate may be folded in (2*n_ff), asserted to be one of the two
                let tn_ffn_up_weight = tensor_name_suffix(LlmTensor::FFN_UP, "weight", bid, -1);
                let n_ffn_up = match ld.gguf.find_tensor(&tn_ffn_up_weight) {
                    Some(ti) => ti.ne[1],
                    None => lc.n_ff,
                };
                if n_ffn_up != lc.n_ff && n_ffn_up != lc.n_ff * 2 {
                    return Err(format!(
                        "jina-bert-v2: ffn_up width {n_ffn_up} at layer {i} is neither n_ff \
                         ({}) nor 2*n_ff ({})",
                        lc.n_ff,
                        lc.n_ff * 2
                    ));
                }
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ffn_up]
                ));
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[n_ffn_up]);

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                // ffn_down_b REQUIRED (jina-bert-v2.cpp:56)
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.layer_out_norm = Some(req!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.layer_out_norm_b = Some(req!(LlmTensor::LAYER_OUT_NORM, "bias", bid, &[lc.n_embd]));
            }

            extra.token_types = Some(type_embd);
            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);
            extra.cls = cls;
            extra.cls_b = cls_b;

            (
                tok_embd,
                tok_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                None,
            )
        }

        // ---- models/jina-bert-v3.cpp:13-44 (LLM_ARCH_JINA_BERT_V3) ----
        LlmArch::JINA_BERT_V3 => {
            let n_token_types = ld.token_type_count();
            if n_token_types == 0 {
                return Err(format!(
                    "{} model needs to define token type count",
                    arch.name()
                ));
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let type_embd = opt!(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[lc.n_embd, n_token_types as i64]
            );

            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_out_norm_b = Some(req!(LlmTensor::ATTN_OUT_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);

                l.layer_out_norm = Some(req!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.layer_out_norm_b = Some(req!(LlmTensor::LAYER_OUT_NORM, "bias", bid, &[lc.n_embd]));
            }

            extra.token_types = type_embd;
            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);

            (
                tok_embd,
                tok_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                None,
            )
        }

        // ---- models/nomic-bert.cpp:13-46 (LLM_ARCH_NOMIC_BERT) ----
        LlmArch::NOMIC_BERT => {
            let n_token_types = ld.token_type_count();
            if n_token_types == 0 {
                return Err(format!(
                    "{} model needs to define token type count",
                    arch.name()
                ));
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let type_embd = opt!(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[lc.n_embd, n_token_types as i64]
            );

            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_out_norm_b = Some(req!(LlmTensor::ATTN_OUT_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);

                // nomic-bert.cpp:41 — REQUIRED (the gated SwiGLU FFN's gate)
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                l.layer_out_norm = Some(req!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.layer_out_norm_b = Some(req!(LlmTensor::LAYER_OUT_NORM, "bias", bid, &[lc.n_embd]));
            }

            extra.token_types = type_embd;
            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);

            (
                tok_embd,
                tok_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                None,
            )
        }

        // ---- models/nomic-bert-moe.cpp:14-51 (LLM_ARCH_NOMIC_BERT_MOE) ----
        LlmArch::NOMIC_BERT_MOE => {
            let n_token_types = ld.token_type_count();
            if n_token_types == 0 {
                return Err(format!(
                    "{} model needs to define token type count",
                    arch.name()
                ));
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let type_embd = opt!(
                LlmTensor::TOKEN_TYPES,
                "weight",
                -1,
                &[lc.n_embd, n_token_types as i64]
            );

            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);
            let tok_norm_b = req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_out_norm_b = Some(req!(LlmTensor::ATTN_OUT_NORM, "bias", bid, &[lc.n_embd]));

                if hparams.moe_every_n_layers > 0 && i % hparams.moe_every_n_layers as usize == 1 {
                    // MoE layer (nomic-bert-moe.cpp:37-40)
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                } else {
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                }

                l.layer_out_norm = Some(req!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]));
                l.layer_out_norm_b = Some(req!(LlmTensor::LAYER_OUT_NORM, "bias", bid, &[lc.n_embd]));
            }

            extra.token_types = type_embd;
            extra.token_embd_norm = Some(tok_norm);
            extra.token_embd_norm_b = Some(tok_norm_b);

            (
                tok_embd,
                tok_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                None,
            )
        }

        // ---- models/neo-bert.cpp:11-37 (LLM_ARCH_NEO_BERT) ----
        LlmArch::NEO_BERT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let cls = opt!(LlmTensor::CLS, "weight", -1, &[lc.n_embd, lc.n_embd]);
            let cls_b = opt!(LlmTensor::CLS, "bias", -1, &[lc.n_embd]);
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_cls_out as i64]
            );
            let cls_out_b = opt!(
                LlmTensor::CLS_OUT,
                "bias",
                -1,
                &[hparams.n_cls_out as i64]
            );

            // neo-bert.cpp:22 — enc.output_norm, the encoder's final RMS norm;
            // mapped onto the generic output_norm slot (the t5encoder
            // precedent)
            let output_norm_enc = req!(LlmTensor::ENC_OUTPUT_NORM, "weight", -1, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    // n_embd_gqa == n_embd_v_gqa (llama-model.h:847)
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_v_gqa]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff * 2]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                let _ = i;
            }

            extra.cls = cls;
            extra.cls_b = cls_b;
            extra.cls_out_b = cls_out_b;

            (
                tok_embd,
                output_norm_enc,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                cls_out,
            )
        }

        // ---- models/modern-bert.cpp:34-66 (LLM_ARCH_MODERN_BERT) ----
        LlmArch::MODERN_BERT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // no tok_norm bias (modern-bert.cpp:38 creates the weight only)
            let tok_norm = req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]);

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);

            // the trailing n_layer_decision blocks form the decision head and
            // load below (a7b94df2c modern-bert.cpp:63)
            let n_layer_enc = lc.n_layer - hparams.n_layer_decision as usize;

            for (i, l) in layers.iter_mut().enumerate().take(n_layer_enc) {
                let bid = i as i32;

                if i != 0 {
                    l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                } else {
                    // layer 0 uses identity (modern-bert.cpp:45-50)
                    l.attn_norm = opt!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]);
                }

                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, 3 * lc.n_embd]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
            }

            // the GTE reranker head (modern-bert.cpp:61-64) — optional
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_cls_out as i64]
            );
            let cls_out_b = opt!(
                LlmTensor::CLS_OUT,
                "bias",
                -1,
                &[hparams.n_cls_out as i64]
            );
            let cls = opt!(LlmTensor::CLS, "weight", -1, &[lc.n_embd, lc.n_embd]);
            let cls_norm = opt!(LlmTensor::CLS_NORM, "weight", -1, &[lc.n_embd]);

            let mut cls_b = None;
            let mut cls_norm_b = None;

            if hparams.n_layer_decision > 0 {
                // decision head: plain pre-norm blocks with biases (a7b94df2c
                // modern-bert.cpp:89-108)
                for (i, l) in layers.iter_mut().enumerate().skip(n_layer_enc) {
                    let bid = i as i32;
                    let n_ff_head = hparams.n_ff(i) as i64;

                    l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                    l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                    l.wqkv = Some(req!(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, 3 * lc.n_embd]
                    ));
                    l.wqkv_b = Some(req!(LlmTensor::ATTN_QKV, "bias", bid, &[3 * lc.n_embd]));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                    l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                    l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                    l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_head]
                    ));
                    l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[n_ff_head]));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[n_ff_head, lc.n_embd]
                    ));
                    l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));
                }

                // one token type per question type (choice, score, noul)
                let n_token_types = ld.token_type_count();
                if n_token_types != 3 {
                    return Err(
                        "decision model must have one token type per question type".into()
                    );
                }
                extra.token_types = Some(req!(
                    LlmTensor::TOKEN_TYPES,
                    "weight",
                    -1,
                    &[lc.n_embd, n_token_types as i64]
                ));

                cls_b = Some(req!(LlmTensor::CLS, "bias", -1, &[lc.n_embd]));
                cls_norm_b = Some(req!(LlmTensor::CLS_NORM, "bias", -1, &[lc.n_embd]));

                if cls.is_none() || cls_norm.is_none() || cls_out.is_none() || cls_out_b.is_none()
                {
                    return Err("decision model is missing the scorer tensors".into());
                }
            }

            extra.token_embd_norm = Some(tok_norm);
            extra.cls = cls;
            extra.cls_b = cls_b;
            extra.cls_out_b = cls_out_b;
            extra.cls_norm = cls_norm;
            extra.cls_norm_b = cls_norm_b;

            (
                tok_embd,
                output_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                cls_out,
            )
        }

        // ---- models/t5.cpp:44-107 (the full encoder-decoder file: the enc
        // half of t5encoder.cpp + the dec.blk.* set of :60-107) ----
        LlmArch::T5 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // t5.cpp:46-53 — enc.output_norm (the extra slot) + dec.output_norm
            // (the generic one) + the optional output with the tie fallback
            extra.enc_output_norm = Some(req!(LlmTensor::ENC_OUTPUT_NORM, "weight", -1, &[lc.n_embd]));
            let output_norm = req!(LlmTensor::DEC_OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_rel_attn_bkts = hparams.n_rel_attn_bkts as i64;

            // load encoder layers (t5.cpp:58-78)
            for (i, l) in layers.iter_mut().enumerate().take(lc.n_layer) {
                let bid = i as i32;
                l.enc_attn_norm = Some(req!(LlmTensor::ENC_ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.enc_attn_rel_b = opt!(
                    LlmTensor::ENC_ATTN_REL_B,
                    "weight",
                    bid,
                    &[lc.n_head, n_rel_attn_bkts]
                );
                l.enc_wq = Some(req!(
                    LlmTensor::ENC_ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.enc_wk = Some(req!(
                    LlmTensor::ENC_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.enc_wv = Some(req!(
                    LlmTensor::ENC_ATTN_V,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.enc_wo = Some(req!(
                    LlmTensor::ENC_ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_v_gqa, lc.n_embd]
                ));
                l.enc_ffn_norm = Some(req!(LlmTensor::ENC_FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.enc_ffn_gate = opt!(
                    LlmTensor::ENC_FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                );
                l.enc_ffn_down = Some(req!(
                    LlmTensor::ENC_FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.enc_ffn_up =
                    Some(req!(LlmTensor::ENC_FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]));
            }

            // t5.cpp:55-57 — `if (dec_n_layer > n_layer) layers.resize(...)`
            let dec_n_layer = hparams.dec_n_layer as usize;
            if dec_n_layer > layers.len() {
                layers.resize(dec_n_layer, Default::default());
            }

            // load decoder layers (t5.cpp:60-107)
            for (i, l) in layers.iter_mut().enumerate().take(dec_n_layer) {
                let bid = i as i32;
                l.dec_attn_norm = Some(req!(LlmTensor::DEC_ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.dec_attn_rel_b = opt!(
                    LlmTensor::DEC_ATTN_REL_B,
                    "weight",
                    bid,
                    &[lc.n_head, n_rel_attn_bkts]
                );
                l.dec_wq = Some(req!(
                    LlmTensor::DEC_ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.dec_wk = Some(req!(
                    LlmTensor::DEC_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.dec_wv = Some(req!(
                    LlmTensor::DEC_ATTN_V,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.dec_wo = Some(req!(
                    LlmTensor::DEC_ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_v_gqa, lc.n_embd]
                ));
                l.dec_attn_norm_cross = Some(req!(
                    LlmTensor::DEC_CROSS_ATTN_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                // attn_rel_b_cross is TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL
                // ("seems to be unused in HF transformers", t5.cpp:93-95) —
                // the port has no SKIP_IF_VIRTUAL; NOT_REQUIRED only
                l.dec_wo_cross = Some(req!(
                    LlmTensor::DEC_CROSS_ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_v_gqa, lc.n_embd]
                ));
                l.dec_wq_cross = Some(req!(
                    LlmTensor::DEC_CROSS_ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.dec_wk_cross = Some(req!(
                    LlmTensor::DEC_CROSS_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.dec_wv_cross = Some(req!(
                    LlmTensor::DEC_CROSS_ATTN_V,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.dec_ffn_norm = Some(req!(LlmTensor::DEC_FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.dec_ffn_gate = opt!(
                    LlmTensor::DEC_FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                );
                l.dec_ffn_down = Some(req!(
                    LlmTensor::DEC_FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.dec_ffn_up =
                    Some(req!(LlmTensor::DEC_FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]));
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/t5encoder.cpp:9-40 ----
        LlmArch::T5ENCODER => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output — `output_norm_enc` is t5encoder.cpp:17; the C member is
            // `output_norm_enc`, mapped here onto the generic `output_norm`
            // slot (same tensor, same role: the encoder's final norm)
            let output_norm = req!(LlmTensor::ENC_OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_rel_attn_bkts = hparams.n_rel_attn_bkts as i64;
            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.enc_attn_norm = Some(req!(LlmTensor::ENC_ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.enc_attn_rel_b = opt!(
                    LlmTensor::ENC_ATTN_REL_B,
                    "weight",
                    bid,
                    &[lc.n_head, n_rel_attn_bkts]
                );

                l.enc_wq = Some(req!(
                    LlmTensor::ENC_ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.enc_wk = Some(req!(
                    LlmTensor::ENC_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_k_gqa]
                ));
                l.enc_wv = Some(req!(
                    LlmTensor::ENC_ATTN_V,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.enc_wo = Some(req!(
                    LlmTensor::ENC_ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_v_gqa, lc.n_embd]
                ));

                l.enc_ffn_norm = Some(req!(LlmTensor::ENC_FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.enc_ffn_gate = opt!(
                    LlmTensor::ENC_FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                );
                l.enc_ffn_down = Some(req!(
                    LlmTensor::ENC_FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.enc_ffn_up = Some(req!(
                    LlmTensor::ENC_FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ====================================================================
        // arch batch 6 (2026-09-24): the DeepSeek MLA family — deepseek /
        // deepseek2 (+deepseek2-ocr). deepseek32/deepseek4 (the DSA/dsv4
        // sparse variants) are a later batch: they need the lightning indexer
        // cache (llama-kv-cache-dsa.cpp) / the dsv4 compressed kv_b cache.
        // ====================================================================

        // ---- models/deepseek2.cpp:55-161 ----
        LlmArch::DEEPSEEK2 | LlmArch::MISTRAL4 => {
            // MISTRAL4 is the pure co-arm (models.h:1393-1395 reuses
            // deepseek2's loader + graph); the arch-specific guards below
            // see arch == MISTRAL4 and take the plain path, like the C
            let n_expert_shared = hparams.n_expert_shared as i64;

            // mtp_only / trunk_only probing (deepseek2.cpp:59-67) decides the
            // TENSOR_NOT_REQUIRED pattern; the port always loads the trunk
            // (create_tensor's TENSOR_SKIP is dropped, see module docs), so a
            // nextn layer's tensors are created NOT_REQUIRED — a file that
            // carries them consumes them, one that does not skips them.
            let is_mla = hparams.is_mla();

            // note: these are the actual head sizes you get when treating as
            // MHA or after "decompression" using wv_b for MLA (deepseek2.cpp:71)
            let n_embd_head_k_mla = hparams.n_embd_head_k_mla() as i64;
            let n_embd_head_v_mla = hparams.n_embd_head_v_mla() as i64;
            let n_embd_head_qk_rope = lc.n_rot;
            let n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
            assert!(n_embd_head_qk_nope >= 1);

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output — tied fallback (deepseek2.cpp:87-92)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                if q_lora_rank > 0 {
                    l.attn_q_a_norm = Some(req!(
                        LlmTensor::ATTN_Q_A_NORM,
                        "weight",
                        bid,
                        &[q_lora_rank]
                    ));
                }

                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));

                if q_lora_rank > 0 {
                    l.wq_a = Some(req!(
                        LlmTensor::ATTN_Q_A,
                        "weight",
                        bid,
                        &[lc.n_embd, q_lora_rank]
                    ));
                    l.wq_b = Some(req!(
                        LlmTensor::ATTN_Q_B,
                        "weight",
                        bid,
                        &[q_lora_rank, lc.n_head * n_embd_head_k_mla]
                    ));
                } else {
                    l.wq = Some(req!(
                        LlmTensor::ATTN_Q,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head * n_embd_head_k_mla]
                    ));
                }

                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));

                // note: only old legacy GGUF files will have the unsplit wkv_b
                // tensor in (deepseek2.cpp:114-120)
                if is_mla {
                    l.wk_b = Some(req!(
                        LlmTensor::ATTN_K_B,
                        "weight",
                        bid,
                        &[n_embd_head_qk_nope, kv_lora_rank, lc.n_head]
                    ));
                    l.wv_b = Some(req!(
                        LlmTensor::ATTN_V_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, n_embd_head_v_mla, lc.n_head]
                    ));
                } else {
                    l.wkv_b = Some(req!(
                        LlmTensor::ATTN_KV_B,
                        "weight",
                        bid,
                        &[
                            kv_lora_rank,
                            lc.n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)
                        ]
                    ));
                }

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * n_embd_head_v_mla, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".into());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".into());
                    }

                    // MoE branch (deepseek2.cpp:142-148)
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    // create_tensor_gate_up_exps (llama-model.cpp:3253-3268):
                    // fused gate_up optional, split gate/up otherwise
                    l.ffn_gate_up_exps = opt!(
                        LlmTensor::FFN_GATE_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * 2, lc.n_expert]
                    );
                    if l.ffn_gate_up_exps.is_none() {
                        l.ffn_gate_exps = Some(req!(
                            LlmTensor::FFN_GATE_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                        l.ffn_up_exps = Some(req!(
                            LlmTensor::FFN_UP_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                    }

                    // Shared expert branch (deepseek2.cpp:146-148)
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }

                // NextN/MTP tensors (deepseek2.cpp:152-159) — see the batch-6
                // note above: NOT_REQUIRED in the port (the C requires them
                // on a full load of a nextn file; the port's speculative path
                // does not drive the MTP graph yet)
                if i >= lc.n_layer {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/deepseek2ocr.cpp:23-75 (deepseek2's graph, no MLA) ----
        LlmArch::DEEPSEEK2OCR => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // similar to deepseek2, but without MLA (deepseek2ocr.cpp:43)
                create_tensor_qkv(l, ld, bid, lc.n_embd, lc.n_embd, lc.n_embd, lc.n_embd, 0)?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".into());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".into());
                    }

                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_up_exps = opt!(
                        LlmTensor::FFN_GATE_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * 2, lc.n_expert]
                    );
                    if l.ffn_gate_up_exps.is_none() {
                        l.ffn_gate_exps = Some(req!(
                            LlmTensor::FFN_GATE_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                        l.ffn_up_exps = Some(req!(
                            LlmTensor::FFN_UP_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert]
                        ));
                    }

                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/deepseek.cpp:17-68 (DeepSeek v2 base, non-MLA MHA) ----
        LlmArch::DEEPSEEK => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // deepseek.cpp:39 — q/k full-width, v the GQA width
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".into());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".into());
                    }

                    // MoE branch (deepseek.cpp:58-60) — always split gate/up
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // Shared expert branch (deepseek.cpp:63-65)
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/deepseek32.cpp:46-154 (DeepSeek V3.2, MLA + DSA) ----
        LlmArch::DEEPSEEK32 => {
            let n_expert_shared = hparams.n_expert_shared as i64;

            if !hparams.is_mla() {
                return Err("DEEPSEEK32 architecture requires MLA".into());
            }

            // deepseek32.cpp:71-74
            let n_embd_head_k_mla = hparams.n_embd_head_k_mla() as i64;
            let n_embd_head_v_mla = hparams.n_embd_head_v_mla() as i64;
            let n_embd_head_qk_rope = lc.n_rot;
            let n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;
            let indexer_head_size = hparams.indexer_head_size as i64;
            let indexer_n_head = hparams.indexer_n_head as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_q_a_norm = Some(req!(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank]
                ));
                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));

                l.wq_a = Some(req!(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank]
                ));
                l.wq_b = Some(req!(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, lc.n_head * n_embd_head_k_mla]
                ));

                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));

                l.wk_b = Some(req!(
                    LlmTensor::ATTN_K_B,
                    "weight",
                    bid,
                    &[n_embd_head_qk_nope, kv_lora_rank, lc.n_head]
                ));
                l.wv_b = Some(req!(
                    LlmTensor::ATTN_V_B,
                    "weight",
                    bid,
                    &[kv_lora_rank, n_embd_head_v_mla, lc.n_head]
                ));

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * n_embd_head_v_mla, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // DSA indexer (deepseek32.cpp:109-114)
                l.indexer_k_norm = Some(req!(
                    LlmTensor::INDEXER_K_NORM,
                    "weight",
                    bid,
                    &[indexer_head_size]
                ));
                l.indexer_k_norm_b = Some(req!(
                    LlmTensor::INDEXER_K_NORM,
                    "bias",
                    bid,
                    &[indexer_head_size]
                ));
                l.indexer_proj = Some(req!(
                    LlmTensor::INDEXER_PROJ,
                    "weight",
                    bid,
                    &[lc.n_embd, indexer_n_head]
                ));
                l.indexer_attn_k = Some(req!(
                    LlmTensor::INDEXER_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, indexer_head_size]
                ));
                l.indexer_attn_q_b = Some(req!(
                    LlmTensor::INDEXER_ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, indexer_n_head * indexer_head_size]
                ));

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".into());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".into());
                    }

                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    // deepseek32.cpp:131-134 — split gate/up only
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }

                // NextN/MTP tensors (deepseek32.cpp:141-150) — see the
                // deepseek2 arm's note (NOT_REQUIRED in the port)
                if i >= lc.n_layer {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- arch batch 7: models/deepseek4.cpp:82-183 load_arch_tensors
        // (hyper-connections + o_group/o_lora output lora + per-ratio
        // compressors + hash layers + MTP block) ----
        LlmArch::DEEPSEEK4 => {
            let n_expert_shared = hparams.n_expert_shared as i64;

            // deepseek4.cpp:85-94
            let q_lora_rank = hparams.n_lora_q as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64; // get_key_or_arr covers all layers
            let n_embd_head = lc.n_embd_head_k;
            let o_groups = hparams.dsv4_o_group_count as i64;
            let o_lora_rank = hparams.dsv4_o_lora_rank as i64;
            let hc_mult = hparams.dsv4_hc_mult as i64;
            let hc_dim = hc_mult * lc.n_embd;
            let hc_mix_dim = (2 + hc_mult) * hc_mult;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // the model-level hyper-connection head (deepseek4.cpp:105-107)
            extra.hc_head_fn = Some(req!(
                LlmTensor::HC_HEAD_FN,
                "weight",
                -1,
                &[hc_dim, hc_mult]
            ));
            extra.hc_head_base = Some(req!(LlmTensor::HC_HEAD_BASE, "weight", -1, &[hc_mult]));
            extra.hc_head_scale = Some(req!(LlmTensor::HC_HEAD_SCALE, "weight", -1, &[1]));

            // trunk flags (deepseek4.cpp:96-98): mtp_only when the file only
            // carries the MTP block; the port loads trunk-only files, so the
            // MTP block's tensors are NOT_REQUIRED (TENSOR_SKIP in C)
            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_sinks = Some(req!(LlmTensor::ATTN_SINKS, "weight", bid, &[lc.n_head]));
                l.wq_a = Some(req!(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank]
                ));
                l.attn_q_a_norm = Some(req!(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank]
                ));
                l.wq_b = Some(req!(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, lc.n_head * n_embd_head]
                ));
                // `attn_kv` — {n_embd, n_embd_head} MQA key/value projection
                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV,
                    "weight",
                    bid,
                    &[lc.n_embd, n_embd_head]
                ));
                l.attn_kv_a_norm =
                    Some(req!(LlmTensor::ATTN_KV_NORM, "weight", bid, &[n_embd_head]));
                // the file stores {n_head*n_embd_head/o_groups, o_lora_rank*o_groups};
                // reshaped here to avoid reshaping in the graph
                // (deepseek4.cpp:120-122, TENSOR_ALLOW_RESHAPE)
                l.wo_a = Some(
                    ld.create_tensor(
                        LlmTensor::ATTN_OUT_A,
                        "weight",
                        bid,
                        &[lc.n_head * n_embd_head / o_groups, o_lora_rank, o_groups],
                        TENSOR_ALLOW_RESHAPE,
                    )?
                    .unwrap(),
                );
                l.wo_b_dsv4 = Some(req!(
                    LlmTensor::ATTN_OUT_B,
                    "weight",
                    bid,
                    &[o_groups * o_lora_rank, lc.n_embd]
                ));

                l.hc_attn_fn = Some(req!(
                    LlmTensor::HC_ATTN_FN,
                    "weight",
                    bid,
                    &[hc_dim, hc_mix_dim]
                ));
                l.hc_attn_base = Some(req!(LlmTensor::HC_ATTN_BASE, "weight", bid, &[hc_mix_dim]));
                l.hc_attn_scale = Some(req!(LlmTensor::HC_ATTN_SCALE, "weight", bid, &[3]));
                l.hc_ffn_fn = Some(req!(
                    LlmTensor::HC_FFN_FN,
                    "weight",
                    bid,
                    &[hc_dim, hc_mix_dim]
                ));
                l.hc_ffn_base = Some(req!(LlmTensor::HC_FFN_BASE, "weight", bid, &[hc_mix_dim]));
                l.hc_ffn_scale = Some(req!(LlmTensor::HC_FFN_SCALE, "weight", bid, &[3]));

                // the per-ratio compressors (deepseek4.cpp:132-154)
                let ratio = hparams.dsv4_compress_ratios[i];
                if ratio != 0 {
                    let coff: i64 = if ratio == 4 { 2 } else { 1 };

                    l.attn_comp_wkv = Some(req!(
                        LlmTensor::ATTN_COMPRESSOR_WKV,
                        "weight",
                        bid,
                        &[lc.n_embd, coff * n_embd_head]
                    ));
                    l.attn_comp_wgate = Some(req!(
                        LlmTensor::ATTN_COMPRESSOR_WGATE,
                        "weight",
                        bid,
                        &[lc.n_embd, coff * n_embd_head]
                    ));
                    l.attn_comp_ape = Some(req!(
                        LlmTensor::ATTN_COMPRESSOR_APE,
                        "weight",
                        bid,
                        &[coff * n_embd_head, ratio as i64]
                    ));
                    l.attn_comp_norm = Some(req!(
                        LlmTensor::ATTN_COMPRESSOR_NORM,
                        "weight",
                        bid,
                        &[n_embd_head]
                    ));

                    if ratio == 4 {
                        let n_embd_indexer = hparams.indexer_head_size as i64;
                        let n_indexer_head = hparams.indexer_n_head as i64;

                        l.indexer_proj = Some(req!(
                            LlmTensor::INDEXER_PROJ,
                            "weight",
                            bid,
                            &[lc.n_embd, n_indexer_head]
                        ));
                        l.indexer_attn_q_b = Some(req!(
                            LlmTensor::INDEXER_ATTN_Q_B,
                            "weight",
                            bid,
                            &[q_lora_rank, n_indexer_head * n_embd_indexer]
                        ));

                        l.indexer_comp_wkv = Some(req!(
                            LlmTensor::INDEXER_COMPRESSOR_WKV,
                            "weight",
                            bid,
                            &[lc.n_embd, 2 * n_embd_indexer]
                        ));
                        l.indexer_comp_wgate = Some(req!(
                            LlmTensor::INDEXER_COMPRESSOR_WGATE,
                            "weight",
                            bid,
                            &[lc.n_embd, 2 * n_embd_indexer]
                        ));
                        l.indexer_comp_ape = Some(req!(
                            LlmTensor::INDEXER_COMPRESSOR_APE,
                            "weight",
                            bid,
                            &[2 * n_embd_indexer, ratio as i64]
                        ));
                        l.indexer_comp_norm = Some(req!(
                            LlmTensor::INDEXER_COMPRESSOR_NORM,
                            "weight",
                            bid,
                            &[n_embd_indexer]
                        ));
                    } else if ratio != 128 {
                        return Err(
                            "DeepSeek-V4 loader only supports compression ratios 0, 4, and 128"
                                .into(),
                        );
                    }
                }

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                if (i as u32) < hparams.dsv4_hash_layer_count {
                    // hash layer: token-id -> expert table replaces the router
                    // bias (deepseek4.cpp:157-159)
                    l.ffn_gate_tid2eid = Some(req!(
                        LlmTensor::FFN_GATE_TID2EID,
                        "weight",
                        bid,
                        &[lc.n_expert_used, lc.n_vocab]
                    ));
                } else {
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));
                }
                // vision variant only (deepseek4.cpp:163): absent in text files
                l.ffn_exp_probs_b_vl =
                    opt!(LlmTensor::FFN_EXP_PROBS_B_VL, "bias", bid, &[lc.n_expert]);
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                l.ffn_gate_shexp = Some(req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp * n_expert_shared]
                ));
                l.ffn_down_shexp = Some(req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_exp * n_expert_shared, lc.n_embd]
                ));
                l.ffn_up_shexp = Some(req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp * n_expert_shared]
                ));

                // the MTP/nextn block (deepseek4.cpp:174-181) — consumed as
                // NOT_REQUIRED like nemotron-h's (the trunk graph never reads
                // it; the MTP graph is not ported)
                if i >= lc.n_layer {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- arch batch 6b (2026-09-24): nemotron (dense) / grok /
        // chameleon / deci / jais / falcon-h1 / plamo2 ----

        // ---- models/nemotron.cpp:12-44 (Nemotron-4 dense: LayerNorm+bias
        // everywhere, relu² MLP, no tied head) ----
        LlmArch::NEMOTRON => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (nemotron.cpp:17-20) — output.weight AND output_norm.bias
            // are required (no TENSOR_DUPLICATED fallback in C)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa,
                // n_embd_gqa, 0) (nemotron.cpp:28)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                // optional bias tensors (nemotron.cpp:31-32)
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                // optional MLP bias (nemotron.cpp:40-42)
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/grok.cpp:35-80 (grok-1: post-norm attention, GELU MoE +
        // optional dense "FFN-free-on-grok-1-is-actually-dense" branch) ----
        LlmArch::GROK => {
            if lc.n_expert == 0 {
                // grok.cpp:38-40
                return Err("grok model cannot have zero experts".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (grok.cpp:44-51) — the usual tied-head fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // grok-1 n_ff_exp == n_ff (grok.cpp:53)
            let n_ff_exp = if hparams.n_ff_exp(0) != 0 {
                hparams.n_ff_exp(0) as i64
            } else {
                lc.n_ff
            };

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // optional dense branch (grok.cpp:66-68)
                l.ffn_gate = opt!(LlmTensor::FFN_GATE, "weight", bid, &[lc.n_embd, lc.n_ff]);
                l.ffn_down = opt!(LlmTensor::FFN_DOWN, "weight", bid, &[lc.n_ff, lc.n_embd]);
                l.ffn_up = opt!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]);

                // MoE (grok.cpp:70-73)
                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = opt!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                );
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                // ffn_post_norm: layer_output_norm first, else the
                // post_ffw_norm name (grok.cpp:75-78)
                l.ffn_post_norm = opt!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]);
                if l.ffn_post_norm.is_none() {
                    l.ffn_post_norm =
                        Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/chameleon.cpp:16-47 (full-width q/k LayerNorms, SwiGLU,
        // the image-token logit suppression of the graph) ----
        LlmArch::CHAMELEON => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (chameleon.cpp:21-27) — tied-head fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // the q/k norms span the whole head axis (chameleon.cpp:33-36)
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head_kv]
                ));
                l.attn_q_norm_b = opt!(
                    LlmTensor::ATTN_Q_NORM,
                    "bias",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head]
                );
                l.attn_k_norm_b = opt!(
                    LlmTensor::ATTN_K_NORM,
                    "bias",
                    bid,
                    &[lc.n_embd_head_k, lc.n_head_kv]
                );

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/deci.cpp:14-73 (per-layer head/ffn arrays: attention /
        // linear-attention / attention-free / FFN-free layer kinds) ----
        LlmArch::DECI => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (deci.cpp:20-26) — tied-head fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // per-layer geometry (deci.cpp:30-34)
                let n_embd_k_gqa = hparams.n_embd_k_gqa(i) as i64;
                let n_embd_v_gqa = hparams.n_embd_v_gqa(i) as i64;
                let n_ff = hparams.n_ff(i) as i64;
                let n_head = hparams.n_head(i) as i64;
                let n_head_kv = hparams.n_head_kv(i) as i64;
                let n_embd_head_k = hparams.n_embd_head_k(i) as i64;

                if n_head_kv == 0 && n_head > 0 {
                    // linear attention for DeciLMCausalModel (deci.cpp:36-40)
                    l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                } else if n_head_kv > 0 {
                    l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                    // q at head_dim*n_head, k/v the GQA width (deci.cpp:44-45)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        n_embd_head_k * n_head,
                        n_embd_k_gqa,
                        n_embd_v_gqa,
                        0,
                    )?;
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[n_embd_head_k * n_head, lc.n_embd]
                    ));
                }
                // n_head == 0: attention-free layer — no attention tensors at all

                // optional bias tensors (deci.cpp:48-49)
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                if n_ff > 0 {
                    l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                }

                // rope factors: long/short for LONGROPE files, else the
                // layer-duplicated rope_freqs (deci.cpp:55-61)
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                if n_ff > 0 {
                    l.ffn_gate = Some(req!(LlmTensor::FFN_GATE, "weight", bid, &[lc.n_embd, n_ff]));
                    l.ffn_down = Some(req!(LlmTensor::FFN_DOWN, "weight", bid, &[n_ff, lc.n_embd]));
                    l.ffn_up = Some(req!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, n_ff]));
                }

                // optional MLP bias (deci.cpp:69-72)
                l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[n_ff]);
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[n_ff]);
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/jais.cpp:15-49 (LayerNorm+bias, fused qkv+bias, ALiBi
        // from the GGUF KV) ----
        LlmArch::JAIS => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (jais.cpp:20-23) — required head + required norm bias
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                // fused qkv + REQUIRED fused bias (jais.cpp:31-32)
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));
                l.wqkv_b = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_k_gqa]
                ));

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_gate_b = Some(req!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]));

                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/falcon-h1.cpp:34-106 (hybrid: mamba2 mixer AND attention
        // in every layer, "is_recr" all-true) ----
        LlmArch::FALCON_H1 => {
            // mamba2 Mixer SSM params (falcon-h1.cpp:41-47)
            let ssm_conv_kernel_size = hparams.ssm_d_conv as i64;
            let ssm_n_groups = hparams.ssm_n_group as i64;
            let ssm_state_size = hparams.ssm_d_state as i64;
            let ssm_intermediate_size = hparams.ssm_d_inner as i64;
            let ssm_num_heads = hparams.ssm_dt_rank as i64;
            let ssm_conv_dim = ssm_intermediate_size + 2 * ssm_n_groups * ssm_state_size;
            let ssm_projection_size = ssm_intermediate_size + ssm_conv_dim + ssm_num_heads;

            // attn params (falcon-h1.cpp:50-51)
            let attn_num_attention_head = hparams.n_head(0) as i64;
            let attn_num_key_value_head = hparams.n_head_kv(0) as i64;

            // ffn params (falcon-h1.cpp:54)
            let ffn_intermediate_size = hparams.n_ff(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (falcon-h1.cpp:59-66) — tied-head fallback
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // SSM LAYERS (falcon-h1.cpp:71-85)
                l.ssm_in = Some(req!(
                    LlmTensor::SSM_IN,
                    "weight",
                    bid,
                    &[lc.n_embd, ssm_projection_size]
                ));
                l.ssm_conv1d = Some(req!(
                    LlmTensor::SSM_CONV1D,
                    "weight",
                    bid,
                    &[ssm_conv_kernel_size, ssm_conv_dim]
                ));
                l.ssm_conv1d_b = opt!(LlmTensor::SSM_CONV1D, "bias", bid, &[ssm_conv_dim]);
                l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[ssm_num_heads]));
                // no "weight" suffix for these (falcon-h1.cpp:79-81)
                l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[1, ssm_num_heads]));
                l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[1, ssm_num_heads]));
                l.ssm_norm = opt!(
                    LlmTensor::SSM_NORM,
                    "weight",
                    bid,
                    &[ssm_intermediate_size / ssm_n_groups, ssm_n_groups]
                );
                l.ssm_out = Some(req!(
                    LlmTensor::SSM_OUT,
                    "weight",
                    bid,
                    &[ssm_intermediate_size, lc.n_embd]
                ));

                // ATTENTION LAYERS (falcon-h1.cpp:87-92)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * attn_num_attention_head,
                    attn_num_key_value_head * lc.n_embd_head_k,
                    attn_num_key_value_head * lc.n_embd_head_v,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * attn_num_attention_head, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // feed forward (falcon-h1.cpp:95-104) — NOTE the ffn_norm has
                // no ".weight" suffix in falcon-h1 files (falcon-h1.cpp:96)
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "", bid, &[lc.n_embd]));
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_freqs = ld.create_tensor(
                    LlmTensor::ROPE_FREQS,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, ffn_intermediate_size]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[ffn_intermediate_size, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, ffn_intermediate_size]
                ));

                l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[ffn_intermediate_size]);
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[ffn_intermediate_size]);
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/plamo2.cpp:35-104 (hybrid: per-layer mamba | attention,
        // is_recr = n_head_kv == 0; post-mixer and post-FFN norms everywhere) ----
        LlmArch::PLAMO2 => {
            // mamba parameters (plamo2.cpp:39-43)
            let d_conv = hparams.ssm_d_conv as i64;
            let d_state = hparams.ssm_d_state as i64;
            let num_heads = hparams.ssm_dt_rank as i64;
            let intermediate_size = hparams.ssm_d_inner as i64;
            // std::max(64, int(n_embd / 16)) (plamo2.cpp:43)
            let dt_dim = 64.max((hparams.n_embd / 16) as i64);

            // attention parameters (plamo2.cpp:46-47) — qk/v head dims can
            // differ (attention.value_length)
            let qk_dim = hparams.n_embd_head_k(0) as i64;
            let v_dim = hparams.n_embd_head_v(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (plamo2.cpp:52-57) — tied-head fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let is_mamba_layer = hparams.is_recr(i);

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if is_mamba_layer {
                    // PLaMo-2 Mamba tensors (plamo2.cpp:66-80) — note the
                    // dt/b/c norm trio carries no ".weight" suffix
                    l.ssm_in = Some(req!(
                        LlmTensor::SSM_IN,
                        "weight",
                        bid,
                        &[lc.n_embd, 2 * intermediate_size]
                    ));
                    l.ssm_conv1d = Some(req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[d_conv, intermediate_size]
                    ));

                    l.ssm_x = Some(req!(
                        LlmTensor::SSM_X,
                        "weight",
                        bid,
                        &[intermediate_size, dt_dim + 2 * d_state]
                    ));
                    l.ssm_dt = Some(req!(LlmTensor::SSM_DT, "weight", bid, &[dt_dim, num_heads]));
                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[num_heads]));

                    l.ssm_a = Some(req!(LlmTensor::SSM_A, "", bid, &[num_heads]));
                    l.ssm_d = Some(req!(LlmTensor::SSM_D, "", bid, &[num_heads]));

                    l.ssm_out = Some(req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[intermediate_size, lc.n_embd]
                    ));

                    l.ssm_dt_norm = Some(req!(LlmTensor::SSM_DT_NORM, "", bid, &[dt_dim]));
                    l.ssm_b_norm = Some(req!(LlmTensor::SSM_B_NORM, "", bid, &[d_state]));
                    l.ssm_c_norm = Some(req!(LlmTensor::SSM_C_NORM, "", bid, &[d_state]));
                } else {
                    // PLaMo-2 attention tensors (plamo2.cpp:81-94)
                    let num_attention_heads = hparams.n_head(i) as i64;
                    let num_key_value_heads = hparams.n_head_kv(i) as i64;
                    let q_proj_dim = num_attention_heads * qk_dim;
                    let k_proj_dim = num_key_value_heads * qk_dim;
                    let v_proj_dim = num_key_value_heads * v_dim;

                    l.wqkv = Some(req!(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, q_proj_dim + k_proj_dim + v_proj_dim]
                    ));
                    l.attn_q_norm = Some(req!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[qk_dim, num_attention_heads]
                    ));
                    l.attn_k_norm = Some(req!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[qk_dim, num_key_value_heads]
                    ));
                    // wo is {q_num_heads * v_dim, n_embd} (plamo2.cpp:94)
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[num_attention_heads * v_dim, lc.n_embd]
                    ));
                }

                // All layers have post-attention norm, FFN norm, and FFN
                // tensors (plamo2.cpp:97-102) — ffn_up is 2*n_ff wide (SWIGLU)
                // and attn_post_norm / ffn_post_norm carry no ".weight" suffix
                l.attn_post_norm = Some(req!(LlmTensor::ATTN_POST_NORM, "", bid, &[lc.n_embd]));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff * 2]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "", bid, &[lc.n_embd]));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ==================================================================
        // arch batch 8 (2026-09-30): the MoE long-tail family —
        // hunyuan-moe / dots1 / bailingmoe / bailingmoe2 / glm4-moe /
        // minimax-m2 / cohere2moe / exaone-moe
        // ==================================================================

        // ---- models/hunyuan-moe.cpp:14-50 ----
        LlmArch::HUNYUAN_MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // hunyuan-moe.cpp:29 — shexp width falls back to the dense n_ff
                let n_ff_shexp = if hparams.n_ff_shexp > 0 {
                    hparams.n_ff_shexp as i64
                } else {
                    hparams.n_ff(i) as i64
                };

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // hunyuan-moe.cpp:41-44 — the expert tensors sit at the *dense*
                // n_ff width (A13B ships both equal), not n_ff_exp
                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));

                l.ffn_gate_shexp = Some(req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
                l.ffn_up_shexp = Some(req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
                l.ffn_down_shexp = Some(req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/dots1.cpp:18-68 ----
        LlmArch::DOTS1 => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // dots1.cpp:27 — output is required (no tied-head fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // dots1.cpp:34 — MHA: k/v widths are n_embd_head_k * n_head
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_head_k * lc.n_head,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".to_string());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".to_string());
                    }

                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    // MoE branch (dots1.cpp:58-60)
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // Shared expert branch (dots1.cpp:63-65) — n_expert_shared
                    // side-by-side experts in one fat tensor
                    let n_sh = n_ff_exp * n_expert_shared;
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_sh]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_sh, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_sh]
                    ));
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/bailingmoe.cpp:18-56 ----
        LlmArch::BAILINGMOE => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // bailingmoe.cpp:28 — output required
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // bailingmoe.cpp:35 — the qkv widths come from n_rot, and the
                // graph reshapes with n_embd_head_k (equal in every real file)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_head * lc.n_rot,
                    lc.n_head_kv * lc.n_rot,
                    lc.n_head_kv * lc.n_rot,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * lc.n_rot, lc.n_embd]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0".to_string());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0".to_string());
                }

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                let n_sh = n_ff_exp * n_expert_shared;
                l.ffn_gate_shexp = Some(req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_sh]
                ));
                l.ffn_down_shexp = Some(req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_sh, lc.n_embd]
                ));
                l.ffn_up_shexp = Some(req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_sh]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/bailingmoe2.cpp:20-84 ----
        LlmArch::BAILINGMOE2 => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            if lc.n_expert <= 0 {
                return Err("n_expert must be > 0 for bailingmoe2".to_string());
            }
            if lc.n_expert_used <= 0 {
                return Err("n_expert_used must be > 0 for bailingmoe2".to_string());
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // the C marks the NextN layers TENSOR_SKIP (bailingmoe2.cpp:
                // 36-39); the port loads trunk-only files, so those reads are
                // NOT_REQUIRED here (deepseek4's convention)
                let skip = i >= lc.n_layer;

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);

                // bailingmoe2.cpp:46 — the fused qkv is created directly
                // (required), width n_embd + 2*n_embd_gqa; the graph's
                // build_qkv reads the fused tensor
                l.wqkv = opt_or_req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_v_gqa],
                    skip
                );
                l.wqkv_b = opt!(
                    LlmTensor::ATTN_QKV,
                    "bias",
                    bid,
                    &[lc.n_embd + 2 * lc.n_embd_v_gqa]
                );
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                    skip
                );

                l.attn_q_norm = opt_or_req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k],
                    skip
                );
                l.attn_k_norm = opt_or_req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k],
                    skip
                );

                l.ffn_norm = opt_or_req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], skip);

                if (i as u32) >= hparams.n_layer_dense_lead {
                    // MoE layers (bailingmoe2.cpp:54-66)
                    let n_ff_shexp = if hparams.n_ff_shexp != 0 {
                        hparams.n_ff_shexp as i64
                    } else {
                        n_ff_exp
                    } * n_expert_shared;

                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );

                    l.ffn_gate_shexp = opt_or_req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                    l.ffn_down_shexp = opt_or_req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd],
                        skip
                    );
                    l.ffn_up_shexp = opt_or_req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                } else {
                    // Dense layers (bailingmoe2.cpp:67-71)
                    l.ffn_gate = opt_or_req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_down = opt_or_req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        skip
                    );
                    l.ffn_up = opt_or_req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                }

                // NextN/MTP tensors (bailingmoe2.cpp:74-82, preserved unused)
                if skip {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                    l.layer_out_norm = opt!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]);
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/glm4-moe.cpp:28-123 (trunk; the NextN/MTP layers of
        // :111-121 stay NOT_REQUIRED like bailingmoe2's) ----
        LlmArch::GLM4_MOE => {
            let n_expert_shared = hparams.n_expert_shared as i64;

            if lc.n_expert <= 0 {
                return Err("n_expert must be > 0 for GLM4_MOE MoE layers".to_string());
            }
            if lc.n_expert_used <= 0 {
                return Err("n_expert_used must be > 0 for GLM4_MOE MoE layers".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer;

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);

                // GLM-style attention (glm4-moe.cpp:62)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    if skip { TENSOR_NOT_REQUIRED } else { 0 },
                )?;
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                    skip
                );

                // K/Q norm tensors (optional — GLM-4.5 355B variant, glm4-moe.cpp:66-70)
                l.attn_q_norm = opt!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd_head_k]);
                l.attn_k_norm = opt!(LlmTensor::ATTN_K_NORM, "weight", bid, &[lc.n_embd_head_k]);

                l.attn_post_norm =
                    opt_or_req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                // glm4-moe.cpp:76 — layer 0 dense, layers 1+ MoE
                let use_moe = (i as u32) >= hparams.n_layer_dense_lead;
                if use_moe {
                    let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                        hparams.n_ff_exp(i) as i64
                    } else {
                        lc.n_ff / lc.n_expert_used
                    };

                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    // glm4-moe.cpp:82 — the router bias is required (unlike dots1)
                    l.ffn_exp_probs_b = opt_or_req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert],
                        skip
                    );

                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );

                    // Shared expert (glm4-moe.cpp:95-103) — only when
                    // n_expert_shared > 0
                    if n_expert_shared > 0 {
                        let n_ff_shexp = n_ff_exp * n_expert_shared;
                        l.ffn_gate_shexp = opt_or_req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp],
                            skip
                        );
                        l.ffn_down_shexp = opt_or_req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd],
                            skip
                        );
                        l.ffn_up_shexp = opt_or_req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp],
                            skip
                        );
                    }
                } else {
                    // Dense layers (glm4-moe.cpp:106-108)
                    l.ffn_gate = opt_or_req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_down = opt_or_req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        skip
                    );
                    l.ffn_up = opt_or_req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                }

                // NextN/MTP tensors (glm4-moe.cpp:112-121)
                if skip {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/minimax-m2.cpp:14-41 ----
        LlmArch::MINIMAX_M2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (minimax-m2.cpp:21 — required)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // minimax-m2.cpp:30-31 — FULL-WIDTH q/k norms
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_k_gqa]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // minimax-m2.cpp:35-39 — the experts sit at the dense n_ff and
                // the router bias is required
                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_exp_probs_b = Some(req!(
                    LlmTensor::FFN_EXP_PROBS_B,
                    "bias",
                    bid,
                    &[lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/cohere2moe.cpp:40-143 (trunk; the MTP block of
        // :102-134 stays NOT_REQUIRED) ----
        LlmArch::COHERE2MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            if lc.n_expert == 0 {
                return Err("n_expert must be > 0 for Cohere2Moe".to_string());
            }
            if lc.n_expert_used == 0 {
                return Err("n_expert_used must be > 0 for Cohere2Moe".to_string());
            }

            // load_block_trunk (cohere2moe.cpp:74-100)
            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer;

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    if skip { TENSOR_NOT_REQUIRED } else { 0 },
                )?;
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                    skip
                );

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = opt_or_req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_down = opt_or_req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        skip
                    );
                    l.ffn_up = opt_or_req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                } else {
                    let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                        hparams.n_ff_exp(i) as i64
                    } else {
                        lc.n_ff
                    };

                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    // create_tensor_gate_up_exps (llama-model.cpp:3253-3273):
                    // the fused {n_embd, 2*n_ff_exp, n_expert} tensor wins,
                    // else the separate gate/up pair
                    l.ffn_gate_up_exps = opt!(
                        LlmTensor::FFN_GATE_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, 2 * n_ff_exp, lc.n_expert]
                    );
                    if l.ffn_gate_up_exps.is_none() {
                        l.ffn_gate_exps = opt_or_req!(
                            LlmTensor::FFN_GATE_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert],
                            skip
                        );
                        l.ffn_up_exps = opt_or_req!(
                            LlmTensor::FFN_UP_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_exp, lc.n_expert],
                            skip
                        );
                    }

                    if hparams.n_expert_shared > 0 {
                        let n_ff_shexp = if hparams.n_ff_shexp != 0 {
                            hparams.n_ff_shexp as i64
                        } else {
                            n_ff_exp * hparams.n_expert_shared as i64
                        };
                        l.ffn_gate_shexp = opt_or_req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp],
                            skip
                        );
                        l.ffn_down_shexp = opt_or_req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd],
                            skip
                        );
                        l.ffn_up_shexp = opt_or_req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp],
                            skip
                        );
                    }
                }

                // load_block_mtp (cohere2moe.cpp:102-134) — the MTP block
                // looks like a full-attention Cohere2 MoE decoder block: the
                // qkv/wo/norm/MoE requests above already cover it (the C's
                // separate loop creates the same names); only the nextn five
                // are additive. opt (SKIP in C for trunk-only files)
                if skip {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/exaone-moe.cpp:28-102 (trunk; the NextN layers of
        // :91-100 stay NOT_REQUIRED) ----
        LlmArch::EXAONE_MOE => {
            let n_ff_exp = hparams.n_ff_exp(0) as i64;
            // exaone-moe.cpp:32 — shexp falls back to n_ff_exp
            let n_ff_shexp = if hparams.n_ff_shexp > 0 {
                hparams.n_ff_shexp as i64
            } else {
                n_ff_exp
            };
            let head_dim = lc.n_embd_head_k;
            let n_qo_dim = lc.n_head * head_dim;
            let n_kv_dim = lc.n_head_kv * head_dim;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (exaone-moe.cpp:40-45) — created with flags 0 (REQUIRED;
            // the C's tie fallback below it is dead code, and the reference
            // indeed rejects a file without output.weight)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    n_qo_dim,
                    n_kv_dim,
                    n_kv_dim,
                    if skip { TENSOR_NOT_REQUIRED } else { 0 },
                )?;
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_qo_dim, lc.n_embd],
                    skip
                );

                // exaone-moe.cpp:58 — rope_freqs (NOT_REQUIRED; DUPLICATED for
                // i != 0 so only layer 0's copy is consumed)
                l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);
                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);
                l.attn_q_norm =
                    opt_or_req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[head_dim], skip);
                l.attn_k_norm =
                    opt_or_req!(LlmTensor::ATTN_K_NORM, "weight", bid, &[head_dim], skip);

                l.ffn_norm = opt_or_req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], skip);

                // dense for the lead layers AND the trailing nextn layers
                // (exaone-moe.cpp:67)
                if (i as u32) < hparams.n_layer_dense_lead || skip {
                    l.ffn_gate = opt_or_req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_down = opt_or_req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        skip
                    );
                    l.ffn_up = opt_or_req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                } else {
                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".to_string());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".to_string());
                    }

                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );

                    // the shared expert is unconditional on MoE layers
                    // (exaone-moe.cpp:86-88)
                    l.ffn_gate_shexp = opt_or_req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                    l.ffn_down_shexp = opt_or_req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd],
                        skip
                    );
                    l.ffn_up_shexp = opt_or_req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                }

                // NextN/MTP tensors (exaone-moe.cpp:92-99)
                if skip {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/plamo3.cpp:20-58 (arch batch 9) ----
        LlmArch::PLAMO3 => {
            let head_dim_q = lc.n_embd_head_k;
            let head_dim_v = lc.n_embd_head_v;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let n_head = hparams.n_head(i) as i64;
                let n_head_kv = hparams.n_head_kv(i) as i64;
                let q_proj_dim = n_head * head_dim_q;
                let k_proj_dim = n_head_kv * head_dim_q;
                let v_proj_dim = n_head_kv * head_dim_v;
                let n_ff_cur = hparams.n_ff(i) as i64;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // plamo3.cpp:45-46 — the fused qkv
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, q_proj_dim + k_proj_dim + v_proj_dim]
                ));
                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[head_dim_q]));
                l.attn_k_norm = Some(req!(LlmTensor::ATTN_K_NORM, "weight", bid, &[head_dim_q]));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_head * head_dim_v, lc.n_embd]
                ));
                // the post norms carry NO suffix (plamo3.cpp:50/:53 — the
                // plamo family's `tn(tensor, i)` form)
                l.attn_post_norm = Some(req!(LlmTensor::ATTN_POST_NORM, "", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "", bid, &[lc.n_embd]));

                // plamo3.cpp:55 — the fused up emits 2*n_ff
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_cur * 2]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[n_ff_cur, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/qwen3next.cpp:31-129 (arch batch 9; trunk — the NextN
        // layers of :108-121 stay NOT_REQUIRED like qwen35's) ----
        LlmArch::QWEN3NEXT => {
            if lc.n_expert == 0 {
                return Err("qwen3next model cannot have zero experts".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // the GDN geometry (qwen3next.cpp:53-66)
            let head_k_dim = hparams.ssm_d_state as i64;
            let head_v_dim = hparams.ssm_d_state as i64;
            let n_k_heads = hparams.ssm_n_group as i64;
            let n_v_heads = hparams.ssm_dt_rank as i64;
            let key_dim = head_k_dim * n_k_heads;
            let value_dim = head_v_dim * n_v_heads;
            let conv_dim = key_dim * 2 + value_dim;
            let qkvz_dim = key_dim * 2 + value_dim * 2;
            let ba_dim = n_v_heads * 2;

            let n_ff_exp = if hparams.n_ff_exp(0) != 0 {
                hparams.n_ff_exp(0) as i64
            } else {
                lc.n_ff / lc.n_expert_used
            };

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer;
                // qwen3next.cpp:70 — shexp falls back to n_ff(il)
                let n_ff_shexp = if hparams.n_ff_shexp > 0 {
                    hparams.n_ff_shexp as i64
                } else {
                    hparams.n_ff(i) as i64
                };

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);
                l.attn_post_norm =
                    opt_or_req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                if !hparams.is_recr(i) {
                    // Attention layers (qwen3next.cpp:75-81) — the Q projection
                    // is twice as wide (it carries the per-head gate)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * lc.n_head * 2,
                        lc.n_embd_k_gqa,
                        lc.n_embd_v_gqa,
                        if skip { TENSOR_NOT_REQUIRED } else { 0 },
                    )?;
                    l.wo = opt_or_req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                        skip
                    );
                    l.attn_q_norm = opt_or_req!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        skip
                    );
                    l.attn_k_norm = opt_or_req!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        skip
                    );
                } else {
                    // Linear attention (gated delta net) tensors
                    // (qwen3next.cpp:83-94) — ssm_in serves the legacy GGUFs
                    l.ssm_in = opt!(LlmTensor::SSM_IN, "weight", bid, &[lc.n_embd, qkvz_dim]);
                    l.wqkv = opt!(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, key_dim * 2 + value_dim]
                    );
                    l.wqkv_gate =
                        opt!(LlmTensor::ATTN_GATE, "weight", bid, &[lc.n_embd, value_dim]);
                    l.ssm_conv1d = opt_or_req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[hparams.ssm_d_conv as i64, conv_dim],
                        skip
                    );
                    l.ssm_dt_b = opt_or_req!(
                        LlmTensor::SSM_DT,
                        "bias",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        skip
                    );
                    // SSM_A_NOSCAN carries no suffix
                    l.ssm_a = ld.create_tensor(
                        LlmTensor::SSM_A_NOSCAN,
                        "",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        if skip { TENSOR_NOT_REQUIRED } else { 0 },
                    )?;
                    l.ssm_beta_alpha = opt_or_req!(
                        LlmTensor::SSM_BETA_ALPHA,
                        "weight",
                        bid,
                        &[lc.n_embd, ba_dim],
                        skip
                    );
                    l.ssm_norm =
                        opt_or_req!(LlmTensor::SSM_NORM, "weight", bid, &[head_v_dim], skip);
                    l.ssm_out = opt_or_req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[value_dim, lc.n_embd],
                        skip
                    );
                }

                // the shared MoE tail (qwen3next.cpp:97-105)
                l.ffn_gate_inp = opt_or_req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert],
                    skip
                );
                l.ffn_down_exps = opt_or_req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert],
                    skip
                );
                // create_tensor_gate_up_exps (llama-model.cpp:3253-3273): the
                // fused {n_embd, 2*n_ff_exp, n_expert} tensor wins, else the
                // separate gate/up pair
                l.ffn_gate_up_exps = opt!(
                    LlmTensor::FFN_GATE_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * n_ff_exp, lc.n_expert]
                );
                if l.ffn_gate_up_exps.is_none() {
                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                }

                // the shared expert with its own gate (qwen3next.cpp:102-105)
                l.ffn_gate_inp_shexp = opt_or_req!(
                    LlmTensor::FFN_GATE_INP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd],
                    skip
                );
                l.ffn_gate_shexp = opt_or_req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp],
                    skip
                );
                l.ffn_up_shexp = opt_or_req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp],
                    skip
                );
                l.ffn_down_shexp = opt_or_req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd],
                    skip
                );

                // the NextN trio + the optional shared-head/embedding
                // fallbacks (load_block_mtp's additive tail,
                // qwen3next.cpp:106-121) — mtp_flags == 0 in the port
                // (load_mtp always on), so they are REQUIRED on a nextn file
                // and absent from a trunk-only one (the loop never reaches
                // i >= n_layer there)
                if skip {
                    l.nextn.eh_proj = Some(req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    ));
                    l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.embed_tokens = ld.create_tensor(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.nextn.shared_head_head = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.nextn.shared_head_norm = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        TENSOR_NOT_REQUIRED,
                    )?;
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/kimi-linear.cpp:34-167 (arch batch 9) ----
        LlmArch::KIMI_LINEAR => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // KDA head_dim = 128 (linear_attn_config.head_dim)
            let n_embd_head_k_kda = hparams.n_embd_head_kda as i64;
            let n_embd_head_v_kda = hparams.n_embd_head_kda as i64;
            let ssm_d_conv = hparams.ssm_d_conv as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.is_recr(i) {
                    // KDA layer (kimi-linear.cpp:56-103)
                    // Conv1d weights: 4D first, then 3D (quantization may drop
                    // the trailing 1) — both map to the same [d_conv*d_inner]
                    // element count, the single create below accepts either
                    // (the port's create_tensor compares nelements + dims of
                    // the leading axes)
                    let d_inner_k = n_embd_head_k_kda * lc.n_head;
                    let d_inner_v = n_embd_head_v_kda * lc.n_head;
                    // Conv1d weights: the 4D [d_conv, 1, d_inner, 1] request
                    // also accepts the 3D files (quantization may drop the
                    // trailing 1) — check_tensor_dims treats missing dims as 1
                    l.ssm_q_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_Q,
                        "weight",
                        bid,
                        &[ssm_d_conv, 1, d_inner_k, 1],
                        0,
                    )?;
                    l.ssm_k_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_K,
                        "weight",
                        bid,
                        &[ssm_d_conv, 1, d_inner_k, 1],
                        0,
                    )?;
                    l.ssm_v_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_V,
                        "weight",
                        bid,
                        &[ssm_d_conv, 1, d_inner_v, 1],
                        0,
                    )?;

                    // q, k, v projections (kimi-linear.cpp:76)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        n_embd_head_k_kda * lc.n_head,
                        n_embd_head_k_kda * lc.n_head,
                        n_embd_head_v_kda * lc.n_head,
                        0,
                    )?;

                    // the KDA projections (kimi-linear.cpp:80-100)
                    l.ssm_f_a = Some(req!(
                        LlmTensor::SSM_F_A,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_head_k_kda]
                    ));
                    l.ssm_f_b = Some(req!(
                        LlmTensor::SSM_F_B,
                        "weight",
                        bid,
                        &[n_embd_head_k_kda, n_embd_head_k_kda * lc.n_head]
                    ));
                    l.ssm_beta = Some(req!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head]
                    ));
                    // SSM_A_NOSCAN — [1, n_head, 1, 1] (4D) or [1, n_head]
                    // (2D after quantization); the 4D request accepts both
                    // (no suffix, kimi-linear.cpp:87-90)
                    l.ssm_a = ld.create_tensor(
                        LlmTensor::SSM_A_NOSCAN,
                        "",
                        bid,
                        &[1, lc.n_head, 1, 1],
                        0,
                    )?;
                    l.ssm_dt_b = Some(req!(
                        LlmTensor::SSM_DT,
                        "bias",
                        bid,
                        &[n_embd_head_k_kda * lc.n_head]
                    ));
                    l.ssm_g_a = Some(req!(
                        LlmTensor::SSM_G_A,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_head_k_kda]
                    ));
                    l.ssm_g_b = Some(req!(
                        LlmTensor::SSM_G_B,
                        "weight",
                        bid,
                        &[n_embd_head_k_kda, n_embd_head_k_kda * lc.n_head]
                    ));
                    // o_norm (reusing SSM_NORM)
                    l.ssm_norm = Some(req!(
                        LlmTensor::SSM_NORM,
                        "weight",
                        bid,
                        &[n_embd_head_k_kda]
                    ));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[n_embd_head_v_kda * lc.n_head, lc.n_embd]
                    ));
                } else {
                    // MLA layer (kimi-linear.cpp:105-135)
                    let q_lora_rank = hparams.n_lora_q as i64;
                    let kv_lora_rank = hparams.n_lora_kv as i64;
                    let n_embd_head_k_mla = hparams.n_embd_head_k_mla() as i64;
                    let n_embd_head_v_mla = hparams.n_embd_head_v_mla() as i64;
                    let qk_rope_head_dim = lc.n_rot; // config.qk_rope_head_dim

                    l.attn_q_a_norm = opt!(LlmTensor::ATTN_Q_A_NORM, "weight", bid, &[q_lora_rank]);
                    if l.attn_q_a_norm.is_some() {
                        l.wq_a = Some(req!(
                            LlmTensor::ATTN_Q_A,
                            "weight",
                            bid,
                            &[lc.n_embd, q_lora_rank]
                        ));
                        l.wq_b = Some(req!(
                            LlmTensor::ATTN_Q_B,
                            "weight",
                            bid,
                            &[q_lora_rank, lc.n_head * n_embd_head_k_mla]
                        ));
                    } else {
                        // Kimi MLA without Q compression
                        l.wq = Some(req!(
                            LlmTensor::ATTN_Q,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_head * n_embd_head_k_mla]
                        ));
                    }

                    // wkv_a_mqa [n_embd, kv_lora_rank + qk_rope_head_dim]
                    l.wkv_a_mqa = Some(req!(
                        LlmTensor::ATTN_KV_A_MQA,
                        "weight",
                        bid,
                        &[lc.n_embd, kv_lora_rank + qk_rope_head_dim]
                    ));
                    // attn_kv_a_norm {kv_lora_rank} (kimi-linear.cpp:113)
                    l.attn_kv_a_norm = Some(req!(
                        LlmTensor::ATTN_KV_A_NORM,
                        "weight",
                        bid,
                        &[kv_lora_rank]
                    ));
                    // the split wk_b/wv_b pair wins; legacy files carry the
                    // unsplit wkv_b (MLA KV cache disabled, kimi-linear.cpp:128-133)
                    l.wk_b = opt!(
                        LlmTensor::ATTN_K_B,
                        "weight",
                        bid,
                        &[
                            n_embd_head_k_mla - qk_rope_head_dim,
                            kv_lora_rank,
                            lc.n_head
                        ]
                    );
                    l.wv_b = opt!(
                        LlmTensor::ATTN_V_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, n_embd_head_v_mla, lc.n_head]
                    );
                    if l.wk_b.is_none() || l.wv_b.is_none() {
                        l.wk_b = None;
                        l.wv_b = None;
                        l.wkv_b = Some(req!(
                            LlmTensor::ATTN_KV_B,
                            "weight",
                            bid,
                            &[
                                kv_lora_rank,
                                lc.n_head
                                    * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)
                            ]
                        ));
                    }
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_head * n_embd_head_v_mla, lc.n_embd]
                    ));
                }

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if (i as u32) < hparams.n_layer_dense_lead {
                    // Dense FFN layer (kimi-linear.cpp:144-148)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    // MoE layer (kimi-linear.cpp:150-164)
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // the shared experts are optional, width n_ff_exp * shared
                    // (kimi-linear.cpp:156-162)
                    let n_ff_shexp_actual = n_ff_exp * i64::from(hparams.n_expert_shared.max(1));
                    l.ffn_gate_shexp = opt!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp_actual]
                    );
                    l.ffn_down_shexp = opt!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp_actual, lc.n_embd]
                    );
                    l.ffn_up_shexp = opt!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp_actual]
                    );

                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/bailingmoe3.cpp:46-158 (arch batch 9; trunk — the
        // NextN layers of :127-158 stay NOT_REQUIRED) ----
        LlmArch::BAILINGMOE3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let head_dim = hparams.n_embd_head_kda as i64;
            let d_inner = head_dim * lc.n_head;
            let d_conv = hparams.ssm_d_conv as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;
            let q_lora_rank = hparams.n_lora_q as i64;
            let qk_rope_head_dim = lc.n_rot;
            let qk_head_dim = hparams.n_embd_head_k_mla() as i64;
            let v_head_dim = hparams.n_embd_head_v_mla() as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer;

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);

                if hparams.is_recr(i) && !skip {
                    // KDA layer (bailingmoe3.cpp:81-93) — the conv kernels are
                    // the 4D [d_conv, 1, d_inner, 1] layout
                    l.ssm_q_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_Q,
                        "weight",
                        bid,
                        &[d_conv, 1, d_inner, 1],
                        0,
                    )?;
                    l.ssm_k_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_K,
                        "weight",
                        bid,
                        &[d_conv, 1, d_inner, 1],
                        0,
                    )?;
                    l.ssm_v_conv = ld.create_tensor(
                        LlmTensor::SSM_CONV1D_V,
                        "weight",
                        bid,
                        &[d_conv, 1, d_inner, 1],
                        0,
                    )?;

                    create_tensor_qkv(l, ld, bid, lc.n_embd, d_inner, d_inner, d_inner, 0)?;
                    l.ssm_f_a = Some(req!(
                        LlmTensor::SSM_F_A,
                        "weight",
                        bid,
                        &[lc.n_embd, d_inner]
                    ));
                    l.ssm_beta = Some(req!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head]
                    ));
                    // SSM_A_NOSCAN — no suffix (bailingmoe3.cpp:89)
                    l.ssm_a =
                        ld.create_tensor(LlmTensor::SSM_A_NOSCAN, "", bid, &[1, lc.n_head], 0)?;
                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[d_inner]));
                    l.ssm_g_a = Some(req!(
                        LlmTensor::SSM_G_A,
                        "weight",
                        bid,
                        &[lc.n_embd, d_inner]
                    ));
                    l.ssm_norm = Some(req!(LlmTensor::SSM_NORM, "weight", bid, &[head_dim]));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else if !skip {
                    // MLA layer (bailingmoe3.cpp:94-108)
                    if q_lora_rank > 0 {                        l.wq_a = Some(req!(
                            LlmTensor::ATTN_Q_A,
                            "weight",
                            bid,
                            &[lc.n_embd, q_lora_rank]
                        ));
                        l.attn_q_a_norm = Some(req!(
                            LlmTensor::ATTN_Q_A_NORM,
                            "weight",
                            bid,
                            &[q_lora_rank]
                        ));
                        l.wq_b = Some(req!(
                            LlmTensor::ATTN_Q_B,
                            "weight",
                            bid,
                            &[q_lora_rank, lc.n_head * qk_head_dim]
                        ));
                    } else {
                        l.wq = Some(req!(
                            LlmTensor::ATTN_Q,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_head * qk_head_dim]
                        ));
                    }
                    l.wkv_a_mqa = Some(req!(
                        LlmTensor::ATTN_KV_A_MQA,
                        "weight",
                        bid,
                        &[lc.n_embd, kv_lora_rank + qk_rope_head_dim]
                    ));
                    l.attn_kv_a_norm = Some(req!(
                        LlmTensor::ATTN_KV_A_NORM,
                        "weight",
                        bid,
                        &[kv_lora_rank]
                    ));
                    l.wk_b = Some(req!(
                        LlmTensor::ATTN_K_B,
                        "weight",
                        bid,
                        &[qk_head_dim - qk_rope_head_dim, kv_lora_rank, lc.n_head]
                    ));
                    l.wv_b = Some(req!(
                        LlmTensor::ATTN_V_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, v_head_dim, lc.n_head]
                    ));
                    l.wqkv_gate = Some(req!(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head]
                    ));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_head * v_head_dim, lc.n_embd]
                    ));
                }

                // MTP/NextN block (bailingmoe3.cpp:126-159) — the extra
                // blocks at i >= n_layer carry the gated-MLA attention set +
                // the MoE FFN (the ffn_* requests below already cover them)
                // + the nextn trio and the LAYER_OUT_NORM shared head norm.
                // mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0
                // (:46-49 — `blk.{n_layer}.nextn.eh_proj.weight` absent).
                if skip {
                    let mtp_probe = format!("blk.{}.nextn.eh_proj.weight", lc.n_layer);
                    let trunk_only =
                        lc.n_layer_nextn > 0 && ld.gguf.find_tensor(&mtp_probe).is_none();
                    let mskip = trunk_only;
                    if q_lora_rank > 0 {
                        l.wq_a = opt_or_req!(
                            LlmTensor::ATTN_Q_A,
                            "weight",
                            bid,
                            &[lc.n_embd, q_lora_rank],
                            mskip
                        );
                        l.attn_q_a_norm = opt_or_req!(
                            LlmTensor::ATTN_Q_A_NORM,
                            "weight",
                            bid,
                            &[q_lora_rank],
                            mskip
                        );
                        l.wq_b = opt_or_req!(
                            LlmTensor::ATTN_Q_B,
                            "weight",
                            bid,
                            &[q_lora_rank, lc.n_head * qk_head_dim],
                            mskip
                        );
                    } else {
                        l.wq = opt_or_req!(
                            LlmTensor::ATTN_Q,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_head * qk_head_dim],
                            mskip
                        );
                    }
                    l.wkv_a_mqa = opt_or_req!(
                        LlmTensor::ATTN_KV_A_MQA,
                        "weight",
                        bid,
                        &[lc.n_embd, kv_lora_rank + qk_rope_head_dim],
                        mskip
                    );
                    l.attn_kv_a_norm = opt_or_req!(
                        LlmTensor::ATTN_KV_A_NORM,
                        "weight",
                        bid,
                        &[kv_lora_rank],
                        mskip
                    );
                    l.wk_b = opt_or_req!(
                        LlmTensor::ATTN_K_B,
                        "weight",
                        bid,
                        &[qk_head_dim - qk_rope_head_dim, kv_lora_rank, lc.n_head],
                        mskip
                    );
                    l.wv_b = opt_or_req!(
                        LlmTensor::ATTN_V_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, v_head_dim, lc.n_head],
                        mskip
                    );
                    l.wqkv_gate = opt_or_req!(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head],
                        mskip
                    );
                    l.wo = opt_or_req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_head * v_head_dim, lc.n_embd],
                        mskip
                    );
                    l.nextn.eh_proj = opt_or_req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd],
                        mskip
                    );
                    l.nextn.enorm = opt_or_req!(
                        LlmTensor::NEXTN_ENORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        mskip
                    );
                    l.nextn.hnorm = opt_or_req!(
                        LlmTensor::NEXTN_HNORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        mskip
                    );
                    // nextn.shared_head_norm is the LAYER_OUT_NORM tensor
                    // (bailingmoe3.cpp:158)
                    l.nextn.shared_head_norm = opt_or_req!(
                        LlmTensor::LAYER_OUT_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        mskip
                    );
                }

                l.ffn_norm = opt_or_req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], skip);
                if (i as u32) < hparams.n_layer_dense_lead && !skip {
                    l.ffn_gate = opt_or_req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_up = opt_or_req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        skip
                    );
                    l.ffn_down = opt_or_req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        skip
                    );
                } else {
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_exp_probs_b = opt_or_req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert],
                        skip
                    );
                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    let n_ff_shexp = hparams.n_ff_shexp as i64;
                    l.ffn_gate_shexp = opt_or_req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                    l.ffn_up_shexp = opt_or_req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                    l.ffn_down_shexp = opt_or_req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd],
                        skip
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/glm5-next.cpp:61-186 (NEW arch, def4d406a) ----
        // GLM5-Next: hybrid KDA + nope-MLA DSA layers with the k-pool
        // indexer, mHC residual streams, DeepSeek-style MoE + NextN block.
        // The port has no TENSOR_SKIP — the NextN block always loads
        // (≡ load_mtp=true, PARITY.md batch 18 model.rs:29-32).
        LlmArch::GLM5_NEXT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(
                LlmTensor::OUTPUT,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // glm5-next.cpp:64-65 — the mHC mix geometry
            let hc = hparams.dsv4_hc_mult as i64;
            let hc_mix_dim = (2 + hc) * hc;

            let head_dim = hparams.n_embd_head_kda as i64;
            let d_conv = hparams.ssm_d_conv as i64;
            let d_inner = head_dim * lc.n_head;

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;
            let qk_head_dim = hparams.n_embd_head_k_mla() as i64;
            let v_head_dim = hparams.n_embd_head_v_mla() as i64;
            let qk_rope_head_dim = hparams.n_rot(0);
            let qk_nope_head_dim = qk_head_dim - qk_rope_head_dim as i64;

            let n_indexer_head = hparams.indexer_n_head as i64;
            let n_embd_indexer = hparams.indexer_head_size as i64;
            let kpool = hparams.indexer_kpool as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let skip = i >= lc.n_layer; // the mtp_flags half in C

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);
                l.ffn_norm = opt_or_req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], skip);

                // the mHC mixers exist on the trunk layers only
                // (glm5-next.cpp:87-94)
                if !skip {
                    l.hc_attn_fn = Some(req!(
                        LlmTensor::HC_ATTN_FN,
                        "weight",
                        bid,
                        &[hc * lc.n_embd, hc_mix_dim]
                    ));
                    l.hc_attn_base = Some(req!(
                        LlmTensor::HC_ATTN_BASE,
                        "weight",
                        bid,
                        &[hc_mix_dim]
                    ));
                    l.hc_attn_scale = Some(req!(
                        LlmTensor::HC_ATTN_SCALE,
                        "weight",
                        bid,
                        &[3]
                    ));
                    l.hc_ffn_fn = Some(req!(
                        LlmTensor::HC_FFN_FN,
                        "weight",
                        bid,
                        &[hc * lc.n_embd, hc_mix_dim]
                    ));
                    l.hc_ffn_base = Some(req!(
                        LlmTensor::HC_FFN_BASE,
                        "weight",
                        bid,
                        &[hc_mix_dim]
                    ));
                    l.hc_ffn_scale = Some(req!(
                        LlmTensor::HC_FFN_SCALE,
                        "weight",
                        bid,
                        &[3]
                    ));
                }

                if hparams.is_recr(i) {
                    // KDA layer (glm5-next.cpp:100-121) — the conv kernels
                    // prefer the 4D [d_conv, 1, d_inner, 1] layout with the
                    // 3D fallback (:101-104)
                    let conv = |l: &mut LayerTensors,
                                ld: &mut ModelLoader,
                                bid: i32,
                                t: LlmTensor|
                     -> Result<TensorId, String> {
                        if let Some(id) = ld.create_tensor(
                            t,
                            "weight",
                            bid,
                            &[d_conv, 1, d_inner, 1],
                            TENSOR_NOT_REQUIRED,
                        )? {
                            return Ok(id);
                        }
                        Ok(ld
                            .create_tensor(t, "weight", bid, &[d_conv, 1, d_inner], 0)?
                            .unwrap())
                    };
                    l.ssm_q_conv = Some(conv(l, ld, bid, LlmTensor::SSM_CONV1D_Q)?);
                    l.ssm_k_conv = Some(conv(l, ld, bid, LlmTensor::SSM_CONV1D_K)?);
                    l.ssm_v_conv = Some(conv(l, ld, bid, LlmTensor::SSM_CONV1D_V)?);

                    create_tensor_qkv(l, ld, bid, lc.n_embd, d_inner, d_inner, d_inner, 0)?;

                    l.ssm_f_a = Some(req!(
                        LlmTensor::SSM_F_A,
                        "weight",
                        bid,
                        &[lc.n_embd, head_dim]
                    ));
                    l.ssm_f_b = Some(req!(
                        LlmTensor::SSM_F_B,
                        "weight",
                        bid,
                        &[head_dim, d_inner]
                    ));
                    l.ssm_beta = Some(req!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head]
                    ));
                    // SSM_A_NOSCAN — no suffix, {n_head} (glm5-next.cpp:115)
                    l.ssm_a =
                        ld.create_tensor(LlmTensor::SSM_A_NOSCAN, "", bid, &[lc.n_head], 0)?;
                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[d_inner]));
                    l.ssm_g_a = Some(req!(
                        LlmTensor::SSM_G_A,
                        "weight",
                        bid,
                        &[lc.n_embd, head_dim]
                    ));
                    l.ssm_g_b = Some(req!(
                        LlmTensor::SSM_G_B,
                        "weight",
                        bid,
                        &[head_dim, d_inner]
                    ));
                    l.ssm_norm = Some(req!(LlmTensor::SSM_NORM, "weight", bid, &[head_dim]));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else {
                    // nope-MLA layer (glm5-next.cpp:122-155)
                    l.attn_q_a_norm = opt_or_req!(
                        LlmTensor::ATTN_Q_A_NORM,
                        "weight",
                        bid,
                        &[q_lora_rank],
                        skip
                    );
                    l.attn_kv_a_norm = opt_or_req!(
                        LlmTensor::ATTN_KV_A_NORM,
                        "weight",
                        bid,
                        &[kv_lora_rank],
                        skip
                    );
                    l.wq_a = opt_or_req!(
                        LlmTensor::ATTN_Q_A,
                        "weight",
                        bid,
                        &[lc.n_embd, q_lora_rank],
                        skip
                    );
                    l.wq_b = opt_or_req!(
                        LlmTensor::ATTN_Q_B,
                        "weight",
                        bid,
                        &[q_lora_rank, lc.n_head * qk_head_dim],
                        skip
                    );
                    l.wkv_a_mqa = opt_or_req!(
                        LlmTensor::ATTN_KV_A_MQA,
                        "weight",
                        bid,
                        &[lc.n_embd, kv_lora_rank + qk_rope_head_dim as i64],
                        skip
                    );
                    l.wk_b = opt_or_req!(
                        LlmTensor::ATTN_K_B,
                        "weight",
                        bid,
                        &[qk_nope_head_dim, kv_lora_rank, lc.n_head],
                        skip
                    );
                    l.wv_b = opt_or_req!(
                        LlmTensor::ATTN_V_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, v_head_dim, lc.n_head],
                        skip
                    );
                    l.wo = opt_or_req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_head * v_head_dim, lc.n_embd],
                        skip
                    );

                    // the k-pool indexer (glm5-next.cpp:141-154) — the NextN
                    // block always has a full indexer; shared trunk layers
                    // carry none (TENSOR_NOT_REQUIRED in C)
                    let full = skip || hparams.is_indexer_full(i);
                    l.indexer_k_norm = opt_or_req!(
                        LlmTensor::INDEXER_K_NORM,
                        "weight",
                        bid,
                        &[n_embd_indexer],
                        !full
                    );
                    l.indexer_k_norm_b = opt_or_req!(
                        LlmTensor::INDEXER_K_NORM,
                        "bias",
                        bid,
                        &[n_embd_indexer],
                        !full
                    );
                    l.indexer_proj = opt_or_req!(
                        LlmTensor::INDEXER_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, n_indexer_head],
                        !full
                    );
                    l.indexer_attn_k = opt_or_req!(
                        LlmTensor::INDEXER_ATTN_K,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_indexer],
                        !full
                    );
                    l.indexer_attn_q_b = opt_or_req!(
                        LlmTensor::INDEXER_ATTN_Q_B,
                        "weight",
                        bid,
                        &[q_lora_rank, n_indexer_head * n_embd_indexer],
                        !full
                    );
                    l.indexer_kpool_gate = opt_or_req!(
                        LlmTensor::INDEXER_KPOOL_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_indexer],
                        !full
                    );
                    l.indexer_kpool_ape = opt_or_req!(
                        LlmTensor::INDEXER_KPOOL_APE,
                        "weight",
                        bid,
                        &[n_embd_indexer, kpool],
                        !full
                    );
                }

                // dense lead FFN vs MoE (glm5-next.cpp:157-175) — `flags` on
                // the MoE half, plain required on the dense half
                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    let n_ff_shexp = n_ff_exp * hparams.n_expert_shared as i64;
                    l.ffn_gate_inp = opt_or_req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_exp_probs_b = opt_or_req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert],
                        skip
                    );
                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_down_exps = opt_or_req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_gate_shexp = opt_or_req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                    l.ffn_down_shexp = opt_or_req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd],
                        skip
                    );
                    l.ffn_up_shexp = opt_or_req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp],
                        skip
                    );
                }

                if skip {
                    // the NextN block (glm5-next.cpp:177-184)
                    l.nextn.eh_proj = Some(req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    ));
                    l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/smallthinker.cpp:30-64 (arch batch 10) ----
        LlmArch::SMALLTHINKER => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (smallthinker.cpp:36-42 — NOT_REQUIRED + the embd fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_k_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0 for SMALLTHINKER".to_string());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0 for SMALLTHINKER".to_string());
                }

                // MoE branch (smallthinker.cpp:57-63 — gate_exps REQUIRED)
                let n_ff_exp = hparams.n_ff_exp(i) as i64;
                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/llada-moe.cpp:16-48 (arch batch 10) ----
        LlmArch::LLADA_MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (llada-moe.cpp:22-23 — REQUIRED, no fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            if lc.n_expert == 0 {
                return Err("n_expert must be > 0 for llada-moe".to_string());
            }
            if lc.n_expert_used == 0 {
                return Err("n_expert_used must be > 0 for llada-moe".to_string());
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // llada-moe.cpp:33 — q width n_embd (head dim * n_head == n_embd)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/minimax-01.cpp:26-61 (arch batch 10) ----
        LlmArch::MINIMAX_01 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output (minimax-01.cpp:32-38 — NOT_REQUIRED + the embd fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let n_embd_q = lc.n_embd_head_k * lc.n_head;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                if !hparams.is_recr(i) {
                    // softmax attention layer (create_tensor_qkv, minimax-01.cpp:46)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        n_embd_q,
                        lc.n_embd_k_gqa,
                        lc.n_embd_v_gqa,
                        0,
                    )?;
                } else {
                    // lightning attention layer (minimax-01.cpp:48-50)
                    l.attn_norm_2 = Some(req!(LlmTensor::ATTN_NORM_2, "weight", bid, &[n_embd_q]));
                    l.wqkv = Some(req!(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, 3 * n_embd_q]
                    ));
                    l.wqkv_gate = Some(req!(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_q]
                    ));
                }
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_embd_q, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                // the experts sit at the DENSE n_ff (minimax-01.cpp:57-59 —
                // n_ff_exp is never read for this arch); gate_exps NOT_REQUIRED
                l.ffn_gate_exps = opt!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                );
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/granite-switch.cpp:74-131 (arch batch 10) ----
        LlmArch::GRANITE_SWITCH => {
            let n_slots = hparams.graniteswitch_n_adapters as i64 + 1; // slot 0 = base/zero delta
            let n_rank = hparams.graniteswitch_max_lora_rank as i64;
            let n_embd_q = lc.n_embd_head_k * lc.n_head;
            let n_embd_kv = lc.n_embd_k_gqa;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // the adapter routing tables (granite-switch.cpp:37-54; read by
            // load_arch_hparams in C — the port re-reads here where the
            // substitute range check against n_vocab lives, :85-91)
            let read_i32_arr = |key: &str| -> Result<Vec<i32>, String> {
                match ld.gguf.find_key(key) {
                    Some(ggml::Value::Array(_, items)) => items
                        .iter()
                        .map(|v| match v {
                            ggml::Value::I32(x) => Ok(*x),
                            ggml::Value::U32(x) => Ok(*x as i32),
                            _ => Err(format!("key {key} has wrong array element type")),
                        })
                        .collect(),
                    _ => Err(format!("key {key} not found")),
                }
            };
            let token_ids = read_i32_arr(&kv_name(arch, LlmKv::ADAPTER_TOKEN_IDS_ACTIVATE))?;
            let substitute_ids = read_i32_arr(&kv_name(arch, LlmKv::ADAPTER_TOKEN_IDS_SUBSTITUTE))?;
            // substitute ids index tok_embd rows directly; range-check against
            // n_vocab (granite-switch.cpp:85-91)
            for &sub in &substitute_ids {
                if sub < 0 || sub as i64 >= lc.n_vocab {
                    return Err(format!(
                        "graniteswitch: substitute token id {sub} out of range [0, {})",
                        lc.n_vocab
                    ));
                }
            }
            let mut token_to_slot = Vec::with_capacity(token_ids.len());
            let mut token_to_substitute = Vec::with_capacity(token_ids.len());
            for i in 0..hparams.graniteswitch_n_adapters as usize {
                // adapter i -> stacked slot i+1 (slot 0 is the base/zero delta)
                token_to_slot.push((token_ids[i], (i + 1) as i32));
                token_to_substitute.push((token_ids[i], substitute_ids[i]));
            }
            extra.graniteswitch_token_to_slot = token_to_slot;
            extra.graniteswitch_token_to_substitute = token_to_substitute;

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // trunk layers only — the router layer (index n_layer) holds no
            // weights (granite-switch.cpp:99: `for (int i = 0; i < n_layer; ++i)`)
            for (i, l) in layers.iter_mut().enumerate() {
                if i >= lc.n_layer {
                    continue;
                }
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, n_embd_q + 2 * n_embd_kv]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_embd_q, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                let a_q = req!(
                    LlmTensor::ATTN_Q,
                    "lora_a",
                    bid,
                    &[lc.n_embd, n_rank, n_slots]
                );
                let b_q = req!(
                    LlmTensor::ATTN_Q,
                    "lora_b",
                    bid,
                    &[n_rank, n_embd_q, n_slots]
                );
                let a_k = req!(
                    LlmTensor::ATTN_K,
                    "lora_a",
                    bid,
                    &[lc.n_embd, n_rank, n_slots]
                );
                let b_k = req!(
                    LlmTensor::ATTN_K,
                    "lora_b",
                    bid,
                    &[n_rank, n_embd_kv, n_slots]
                );
                let a_v = req!(
                    LlmTensor::ATTN_V,
                    "lora_a",
                    bid,
                    &[lc.n_embd, n_rank, n_slots]
                );
                let b_v = req!(
                    LlmTensor::ATTN_V,
                    "lora_b",
                    bid,
                    &[n_rank, n_embd_kv, n_slots]
                );

                let a_o = req!(
                    LlmTensor::ATTN_OUT,
                    "lora_a",
                    bid,
                    &[n_embd_q, n_rank, n_slots]
                );
                let b_o = req!(
                    LlmTensor::ATTN_OUT,
                    "lora_b",
                    bid,
                    &[n_rank, lc.n_embd, n_slots]
                );

                let a_gate = req!(
                    LlmTensor::FFN_GATE,
                    "lora_a",
                    bid,
                    &[lc.n_embd, n_rank, n_slots]
                );
                let b_gate = req!(
                    LlmTensor::FFN_GATE,
                    "lora_b",
                    bid,
                    &[n_rank, lc.n_ff, n_slots]
                );
                let a_up = req!(
                    LlmTensor::FFN_UP,
                    "lora_a",
                    bid,
                    &[lc.n_embd, n_rank, n_slots]
                );
                let b_up = req!(
                    LlmTensor::FFN_UP,
                    "lora_b",
                    bid,
                    &[n_rank, lc.n_ff, n_slots]
                );
                let a_down = req!(
                    LlmTensor::FFN_DOWN,
                    "lora_a",
                    bid,
                    &[lc.n_ff, n_rank, n_slots]
                );
                let b_down = req!(
                    LlmTensor::FFN_DOWN,
                    "lora_b",
                    bid,
                    &[n_rank, lc.n_embd, n_slots]
                );
                l.switch_lora = Some(SwitchLoraTensors {
                    a_q,
                    b_q,
                    a_k,
                    b_k,
                    a_v,
                    b_v,
                    a_o,
                    b_o,
                    a_gate,
                    b_gate,
                    a_up,
                    b_up,
                    a_down,
                    b_down,
                });
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ====================================================================
        // arch batch 11a (2026-10) — the long-tail queue, first half
        // ====================================================================

        // ---- models/apertus.cpp:17-54 (arch batch 11a) ----
        LlmArch::APERTUS => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // longrope files keep the long/short factor pair, plain files
                // the model-level rope_freqs name (apertus.cpp:31-36)
                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long =
                        opt!(LlmTensor::ROPE_FACTORS_LONG, "weight", bid, &[lc.n_rot / 2]);
                    l.rope_short = opt!(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2]
                    );
                } else {
                    l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);
                }

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                // the per-head q/k layer norms (weight required, bias optional,
                // apertus.cpp:49-52)
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm_b = opt!(LlmTensor::ATTN_Q_NORM, "bias", bid, &[lc.n_embd_head_k]);
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm_b = opt!(LlmTensor::ATTN_K_NORM, "bias", bid, &[lc.n_embd_head_k]);
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/grovemoe.cpp:16-61 (arch batch 11a) ----
        LlmArch::GROVEMOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // GGML_ASSERTs of grovemoe.cpp:29-31
            if lc.n_expert <= 0 {
                return Err("grovemoe: n_expert must be > 0".to_string());
            }
            if lc.n_expert_used <= 0 {
                return Err("grovemoe: n_expert_used must be > 0".to_string());
            }
            if hparams.n_group_experts == 0 {
                return Err("grovemoe: experts_per_group must be > 0".to_string());
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                // the MoE branch (grovemoe.cpp:48-59)
                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };
                let n_ff_chexp = if hparams.n_ff_chexp != 0 {
                    hparams.n_ff_chexp as i64
                } else {
                    lc.n_embd_head_k
                };
                let n_chunk_expert = lc.n_expert / hparams.n_group_experts as i64;

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));

                l.ffn_gate_chexps = Some(req!(
                    LlmTensor::FFN_GATE_CHEXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_chexp, n_chunk_expert]
                ));
                l.ffn_down_chexps = Some(req!(
                    LlmTensor::FFN_DOWN_CHEXPS,
                    "weight",
                    bid,
                    &[n_ff_chexp, lc.n_embd, n_chunk_expert]
                ));
                l.ffn_up_chexps = Some(req!(
                    LlmTensor::FFN_UP_CHEXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_chexp, n_chunk_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/qwen35moe.cpp:36-147 (arch batch 11a; trunk — the
        // NextN/MTP blocks of :106-139 stay NOT_REQUIRED) ----
        LlmArch::QWEN35MOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let mtp = i >= lc.n_layer as usize;
                let skip = mtp; // opt_or_req! bool; create_tensor_qkv gets the flag below

                let head_k_dim = hparams.ssm_d_state as i64;
                let head_v_dim = hparams.ssm_d_state as i64;
                let n_k_heads = hparams.ssm_n_group as i64;
                let n_v_heads = hparams.ssm_dt_rank as i64;
                let key_dim = head_k_dim * n_k_heads;
                let value_dim = head_v_dim * n_v_heads;
                let conv_dim = key_dim * 2 + value_dim;

                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };
                let n_ff_shexp = if hparams.n_ff_shexp != 0 {
                    hparams.n_ff_shexp as i64
                } else {
                    lc.n_ff
                };

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);
                l.attn_post_norm =
                    opt_or_req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                if !hparams.is_recr(i) {
                    // attention layers: wq holds [q|gate] (qwen35moe.cpp:74)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * lc.n_head * 2,
                        lc.n_embd_k_gqa,
                        lc.n_embd_v_gqa,
                        if skip { TENSOR_NOT_REQUIRED } else { 0 },
                    )?;
                    l.wo = opt_or_req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                        skip
                    );
                    l.attn_q_norm = opt_or_req!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        skip
                    );
                    l.attn_k_norm = opt_or_req!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k],
                        skip
                    );
                } else {
                    // GDN layers (qwen35moe.cpp:81-91)
                    l.wqkv = ld.create_tensor(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, key_dim * 2 + value_dim],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.wqkv_gate = ld.create_tensor(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, value_dim],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.ssm_conv1d = opt_or_req!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[hparams.ssm_d_conv as i64, conv_dim],
                        skip
                    );
                    l.ssm_dt_b = opt_or_req!(
                        LlmTensor::SSM_DT,
                        "bias",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        skip
                    );
                    l.ssm_a = ld.create_tensor(
                        LlmTensor::SSM_A_NOSCAN,
                        "",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        if skip { TENSOR_NOT_REQUIRED } else { 0 },
                    )?;
                    l.ssm_beta = opt_or_req!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads],
                        skip
                    );
                    l.ssm_alpha = opt_or_req!(
                        LlmTensor::SSM_ALPHA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads],
                        skip
                    );
                    l.ssm_norm =
                        opt_or_req!(LlmTensor::SSM_NORM, "weight", bid, &[head_v_dim], skip);
                    l.ssm_out = opt_or_req!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[value_dim, lc.n_embd],
                        skip
                    );
                }

                // routed experts + the shared expert (qwen35moe.cpp:94-103)
                l.ffn_gate_inp = opt_or_req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert],
                    skip
                );
                l.ffn_down_exps = opt_or_req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert],
                    skip
                );
                l.ffn_gate_up_exps = opt!(
                    LlmTensor::FFN_GATE_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * n_ff_exp, lc.n_expert]
                );
                if l.ffn_gate_up_exps.is_none() {
                    l.ffn_gate_exps = opt_or_req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                    l.ffn_up_exps = opt_or_req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        skip
                    );
                }
                l.ffn_gate_inp_shexp = opt_or_req!(
                    LlmTensor::FFN_GATE_INP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd],
                    skip
                );
                l.ffn_gate_shexp = opt_or_req!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp],
                    skip
                );
                l.ffn_up_shexp = opt_or_req!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp],
                    skip
                );
                l.ffn_down_shexp = opt_or_req!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd],
                    skip
                );

                // NextN-specific tensors that define the MTP block
                // (qwen35moe.cpp:131-137). C++ `mtp_flags = TENSOR_SKIP` on a
                // trunk-only load; the port's skip→NOT_REQUIRED degradation
                // consumes them when the file ships the MTP layer (Ornith's
                // blk.40.nextn.*) and tolerates trunk-only files.
                if mtp {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/kimi-k3.cpp:53-167 (arch batch 11a) ----
        LlmArch::KIMI_K3 => {
            let n_embd_latent = if hparams.n_expert_latent > 0 {
                hparams.n_expert_latent as i64
            } else {
                lc.n_embd
            };

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // the final residual-bank score (kimi-k3.cpp:63-65)
            if hparams.attn_res_block_size > 0 {
                extra.output_res_score = Some(req!(
                    LlmTensor::OUTPUT_RES_SCORE,
                    "weight",
                    -1,
                    &[lc.n_embd]
                ));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.attn_res_block_size > 0 {
                    l.attn_res_score =
                        Some(req!(LlmTensor::ATTN_RES_SCORE, "weight", bid, &[lc.n_embd]));
                    l.ffn_res_score =
                        Some(req!(LlmTensor::FFN_RES_SCORE, "weight", bid, &[lc.n_embd]));
                }

                let head_dim = hparams.n_embd_head_kda as i64;
                let d_conv = hparams.ssm_d_conv as i64;
                let d_inner = head_dim * lc.n_head;

                if hparams.is_recr(i) {
                    // the KDA layer (kimi-k3.cpp:82-105)
                    // 4D [d_conv, 1, d_inner, 1] or 3D after quantization —
                    // the 4D request accepts both (kimi-k3.cpp:84-87)
                    let mut conv = |tid: LlmTensor| -> Result<TensorId, String> {
                        ld.create_tensor(tid, "weight", bid, &[d_conv, 1, d_inner, 1], 0)
                            .and_then(|t| {
                                t.ok_or_else(|| "kimi-k3 conv tensor missing".to_string())
                            })
                    };
                    l.ssm_q_conv = Some(conv(LlmTensor::SSM_CONV1D_Q)?);
                    l.ssm_k_conv = Some(conv(LlmTensor::SSM_CONV1D_K)?);
                    l.ssm_v_conv = Some(conv(LlmTensor::SSM_CONV1D_V)?);

                    create_tensor_qkv(l, ld, bid, lc.n_embd, d_inner, d_inner, d_inner, 0)?;

                    l.ssm_f_a = Some(req!(
                        LlmTensor::SSM_F_A,
                        "weight",
                        bid,
                        &[lc.n_embd, head_dim]
                    ));
                    l.ssm_f_b = Some(req!(
                        LlmTensor::SSM_F_B,
                        "weight",
                        bid,
                        &[head_dim, d_inner]
                    ));
                    l.ssm_beta = Some(req!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head]
                    ));
                    // K3's A_log is a plain 1-D [n_head] tensor (kimi-k3.cpp:99)
                    l.ssm_a =
                        ld.create_tensor(LlmTensor::SSM_A_NOSCAN, "", bid, &[lc.n_head], 0)?;
                    l.ssm_dt_b = Some(req!(LlmTensor::SSM_DT, "bias", bid, &[d_inner]));

                    // the single full-rank gate (:103)
                    l.ssm_g = Some(req!(LlmTensor::SSM_G, "weight", bid, &[lc.n_embd, d_inner]));
                    l.ssm_norm = Some(req!(LlmTensor::SSM_NORM, "weight", bid, &[head_dim]));
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[d_inner, lc.n_embd]
                    ));
                } else {
                    // the MLA layer (kimi-k3.cpp:106-137)
                    let q_lora_rank = hparams.n_lora_q as i64;
                    let kv_lora_rank = hparams.n_lora_kv as i64;
                    let n_embd_head_k = hparams.n_embd_head_k_mla() as i64;
                    let n_embd_head_v = hparams.n_embd_head_v_mla() as i64;
                    let qk_rope_head_dim = lc.n_rot;
                    let qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

                    l.attn_q_a_norm = opt!(LlmTensor::ATTN_Q_A_NORM, "weight", bid, &[q_lora_rank]);
                    if l.attn_q_a_norm.is_some() {
                        l.wq_a = Some(req!(
                            LlmTensor::ATTN_Q_A,
                            "weight",
                            bid,
                            &[lc.n_embd, q_lora_rank]
                        ));
                        l.wq_b = Some(req!(
                            LlmTensor::ATTN_Q_B,
                            "weight",
                            bid,
                            &[q_lora_rank, lc.n_head * n_embd_head_k]
                        ));
                    } else {
                        l.wq = Some(req!(
                            LlmTensor::ATTN_Q,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_head * n_embd_head_k]
                        ));
                    }

                    l.wkv_a_mqa = Some(req!(
                        LlmTensor::ATTN_KV_A_MQA,
                        "weight",
                        bid,
                        &[lc.n_embd, kv_lora_rank + qk_rope_head_dim]
                    ));
                    // the split pair wins; the unsplit wkv_b is the legacy file
                    // (kimi-k3.cpp:125-131)
                    l.wkv_b = ld.create_tensor(
                        LlmTensor::ATTN_KV_B,
                        "weight",
                        bid,
                        &[kv_lora_rank, lc.n_head * (qk_nope_head_dim + n_embd_head_v)],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    if l.wkv_b.is_none() {
                        l.wk_b = Some(req!(
                            LlmTensor::ATTN_K_B,
                            "weight",
                            bid,
                            &[qk_nope_head_dim, kv_lora_rank, lc.n_head]
                        ));
                        l.wv_b = Some(req!(
                            LlmTensor::ATTN_V_B,
                            "weight",
                            bid,
                            &[kv_lora_rank, n_embd_head_v, lc.n_head]
                        ));
                    }
                    l.attn_kv_a_norm = Some(req!(
                        LlmTensor::ATTN_KV_A_NORM,
                        "weight",
                        bid,
                        &[kv_lora_rank]
                    ));

                    // the sigmoid output gate (:134)
                    l.wqkv_gate = ld.create_tensor(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_head * n_embd_head_v],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.wo = Some(req!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_head * n_embd_head_v, lc.n_embd]
                    ));
                }

                if i < hparams.n_layer_dense_lead as usize {
                    // dense lead FFN (kimi-k3.cpp:139-142)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    // the latent MoE (kimi-k3.cpp:143-165)
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));
                    // routed experts live in the latent space (:150-152)
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[n_embd_latent, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, n_embd_latent, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[n_embd_latent, n_ff_exp, lc.n_expert]
                    ));
                    if hparams.n_expert_latent > 0 {
                        l.ffn_routed_down = Some(req!(
                            LlmTensor::FFN_ROUTED_DOWN,
                            "weight",
                            bid,
                            &[lc.n_embd, n_embd_latent]
                        ));
                        l.ffn_routed_up = Some(req!(
                            LlmTensor::FFN_ROUTED_UP,
                            "weight",
                            bid,
                            &[n_embd_latent, lc.n_embd]
                        ));
                        l.ffn_routed_norm =
                            opt!(LlmTensor::FFN_ROUTED_NORM, "weight", bid, &[n_embd_latent]);
                    }
                    // shared experts stay at n_embd (:160-164)
                    let n_ff_shexp = n_ff_exp * i64::from(hparams.n_expert_shared.max(1));
                    l.ffn_gate_shexp = opt!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    );
                    l.ffn_down_shexp = opt!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd]
                    );
                    l.ffn_up_shexp = opt!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    );
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/dots3note.cpp:48-146 (arch batch 11a; trunk — the
        // NextN/MTP blocks of :138-144 stay NOT_REQUIRED) ----
        LlmArch::DOTS3NOTE => {
            if !hparams.is_mla() {
                return Err("dots3note: the architecture requires MLA".to_string());
            }

            let n_embd_head_qk_rope = lc.n_rot;
            let q_lora_rank = hparams.n_lora_q as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;
            let n_expert_shared = hparams.n_expert_shared as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let is_mtp = i >= lc.n_layer as usize;
                // the NextN/MTP block uses the sliding-attention geometry
                let is_swa = is_mtp || hparams.is_swa(i);
                let flags = if is_mtp { TENSOR_NOT_REQUIRED } else { 0 };

                let kv_lora_rank = if is_swa {
                    hparams.n_lora_kv_swa as i64
                } else {
                    hparams.n_lora_kv as i64
                };
                let n_embd_head_k_mla = if is_swa {
                    hparams.n_embd_head_k_mla_swa as i64
                } else {
                    hparams.n_embd_head_k_mla() as i64
                };
                let n_embd_head_v_mla = if is_swa {
                    hparams.n_embd_head_v_mla_swa as i64
                } else {
                    hparams.n_embd_head_v_mla() as i64
                };
                let n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

                l.attn_norm =
                    ld.create_tensor(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], flags)?;
                l.attn_q_a_norm = ld.create_tensor(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank],
                    flags,
                )?;
                l.attn_kv_a_norm = ld.create_tensor(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank],
                    flags,
                )?;
                // the norm on the shared rope key, before rope (:91)
                l.attn_k_norm = ld.create_tensor(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[n_embd_head_qk_rope],
                    flags,
                )?;

                l.wq_a = ld.create_tensor(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank],
                    flags,
                )?;
                l.wq_b = ld.create_tensor(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, hparams.n_head(i) as i64 * n_embd_head_k_mla],
                    flags,
                )?;

                l.wkv_a_mqa = ld.create_tensor(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope],
                    flags,
                )?;
                l.wk_b = ld.create_tensor(
                    LlmTensor::ATTN_K_B,
                    "weight",
                    bid,
                    &[n_embd_head_qk_nope, kv_lora_rank, hparams.n_head(i) as i64],
                    flags,
                )?;
                l.wv_b = ld.create_tensor(
                    LlmTensor::ATTN_V_B,
                    "weight",
                    bid,
                    &[kv_lora_rank, n_embd_head_v_mla, hparams.n_head(i) as i64],
                    flags,
                )?;
                l.wo = ld.create_tensor(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[hparams.n_head(i) as i64 * n_embd_head_v_mla, lc.n_embd],
                    flags,
                )?;

                // the head-wise sigmoid output gate (:104)
                l.wqkv_gate = ld.create_tensor(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, hparams.n_head(i) as i64],
                    flags,
                )?;

                l.ffn_norm =
                    ld.create_tensor(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], flags)?;

                // the DSA indexer — full-attention indexer layers only (:109-115)
                if !is_mtp && hparams.is_indexer_full(i) {
                    l.indexer_k_norm = ld.create_tensor(
                        LlmTensor::INDEXER_K_NORM,
                        "weight",
                        bid,
                        &[hparams.indexer_head_size as i64],
                        flags,
                    )?;
                    l.indexer_k_norm_b = ld.create_tensor(
                        LlmTensor::INDEXER_K_NORM,
                        "bias",
                        bid,
                        &[hparams.indexer_head_size as i64],
                        flags,
                    )?;
                    l.indexer_proj = ld.create_tensor(
                        LlmTensor::INDEXER_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, hparams.indexer_n_head as i64],
                        flags,
                    )?;
                    l.indexer_attn_k = ld.create_tensor(
                        LlmTensor::INDEXER_ATTN_K,
                        "weight",
                        bid,
                        &[lc.n_embd, hparams.indexer_head_size as i64],
                        flags,
                    )?;
                    l.indexer_attn_q_b = ld.create_tensor(
                        LlmTensor::INDEXER_ATTN_Q_B,
                        "weight",
                        bid,
                        &[
                            q_lora_rank,
                            hparams.indexer_n_head as i64 * hparams.indexer_head_size as i64,
                        ],
                        flags,
                    )?;
                }

                if is_mtp || i < hparams.n_layer_dense_lead as usize {
                    l.ffn_gate = ld.create_tensor(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        flags,
                    )?;
                    l.ffn_down = ld.create_tensor(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd],
                        flags,
                    )?;
                    l.ffn_up = ld.create_tensor(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff],
                        flags,
                    )?;
                } else {
                    if lc.n_expert == 0 || lc.n_expert_used == 0 {
                        return Err("dots3note: n_expert and n_expert_used must be > 0".to_string());
                    }
                    l.ffn_gate_inp = ld.create_tensor(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert],
                        flags,
                    )?;
                    l.ffn_exp_probs_b = ld.create_tensor(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert],
                        flags,
                    )?;
                    l.ffn_gate_exps = ld.create_tensor(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        flags,
                    )?;
                    l.ffn_down_exps = ld.create_tensor(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert],
                        flags,
                    )?;
                    l.ffn_up_exps = ld.create_tensor(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert],
                        flags,
                    )?;
                    l.ffn_gate_shexp = ld.create_tensor(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared],
                        flags,
                    )?;
                    l.ffn_down_shexp = ld.create_tensor(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd],
                        flags,
                    )?;
                    l.ffn_up_shexp = ld.create_tensor(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared],
                        flags,
                    )?;
                }

                if is_mtp {
                    // the NextN block tensors load 1:1 but no MTP graph exists
                    // (dots3note.cpp:138-144)
                    l.nextn.eh_proj = ld.create_tensor(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd],
                        flags,
                    )?;
                    l.nextn.enorm = ld.create_tensor(
                        LlmTensor::NEXTN_ENORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        flags,
                    )?;
                    l.nextn.hnorm = ld.create_tensor(
                        LlmTensor::NEXTN_HNORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        flags,
                    )?;
                    l.nextn.embed_tokens = ld.create_tensor(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab],
                        flags,
                    )?;
                    l.nextn.shared_head_norm = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        flags,
                    )?;
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/minimax-m3.cpp:41-90 (arch batch 11a) ----
        LlmArch::MINIMAX_M3 => {
            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // a single head_dim vector applied to every head (:59-61)
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if i < hparams.n_layer_dense_lead as usize {
                    // leading dense layers (:65-69)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    // routed + shared experts (:70-81)
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));

                    // the indexer (:83-87)
                    l.index_q_proj = Some(req!(
                        LlmTensor::INDEXER_Q_PROJ,
                        "weight",
                        bid,
                        &[
                            lc.n_embd,
                            hparams.indexer_n_head as i64 * hparams.indexer_head_size as i64
                        ]
                    ));
                    l.index_k_proj = Some(req!(
                        LlmTensor::INDEXER_K_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, hparams.indexer_head_size as i64]
                    ));
                    l.index_q_norm = Some(req!(
                        LlmTensor::INDEXER_Q_NORM,
                        "weight",
                        bid,
                        &[hparams.indexer_head_size as i64]
                    ));
                    l.index_k_norm = Some(req!(
                        LlmTensor::INDEXER_K_NORM,
                        "weight",
                        bid,
                        &[hparams.indexer_head_size as i64]
                    ));
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/qwen4exp.cpp:150-259 (arch batch 11a; trunk — no
        // NextN blocks in this file, PLE tensors only when the keys exist) ----
        LlmArch::QWEN4EXP => {
            let hc = hparams.dsv4_hc_mult as i64;
            let hc_dim = hc * lc.n_embd;
            let hc_lr = hparams.hc_low_rank as i64;

            // an MTP-only file carries the MTP block, the embeddings and the
            // LM head, but no trunk (a7b94df2c qwen4exp.cpp:178-181)
            let mtp_only = lc.n_layer_nextn > 0
                && ld.gguf.find_tensor("blk.0.hc_attn_norm.weight").is_none();
            let trunk_flags = if mtp_only { TENSOR_NOT_REQUIRED } else { 0 };
            // C++ `mtp_flags = ml.load_mtp ? 0 : TENSOR_SKIP`; the port has
            // no load_mtp switch — MTP blocks always load
            let mtp_flags = 0u32;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // there is no output_norm: the final hyper-connection mixer
            // carries it (qwen4exp.cpp:159-163)
            extra.hc_head_norm = ld.create_tensor(
                LlmTensor::HC_HEAD_NORM,
                "weight",
                -1,
                &[lc.n_embd, hc],
                trunk_flags | TENSOR_ALLOW_RESHAPE,
            )?;
            let output_norm = extra.hc_head_norm.unwrap();
            extra.hc_head_down = Some(ld.create_tensor(
                LlmTensor::HC_HEAD_DOWN,
                "weight",
                -1,
                &[hc_dim, hc_lr],
                trunk_flags,
            )?.unwrap());
            extra.hc_head_up = Some(ld.create_tensor(
                LlmTensor::HC_HEAD_UP,
                "weight",
                -1,
                &[hc_lr, hc_dim],
                trunk_flags,
            )?.unwrap());

            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // load_block(il, flags): the trunk pass and the MTP block
                // share the body (a7b94df2c qwen4exp.cpp:188-272/233-236); the
                // macro lives inside the loop so `flags` is at its definition
                // site (macro hygiene)
                let flags = if i < lc.n_layer { trunk_flags } else { mtp_flags };
                macro_rules! reqf {
                    ($t:expr, $suf:expr, $bid:expr, $ne:expr) => {
                        ld.create_tensor($t, $suf, $bid, $ne, flags)?.unwrap()
                    };
                }

                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };
                let n_ff_shexp = if hparams.n_ff_shexp != 0 {
                    hparams.n_ff_shexp as i64
                } else {
                    lc.n_ff
                };

                let head_k_dim = hparams.ssm_d_state as i64;
                let head_v_dim = hparams.ssm_d_state as i64;
                let n_k_heads = hparams.ssm_n_group as i64;
                let n_v_heads = hparams.ssm_dt_rank as i64;
                let key_dim = head_k_dim * n_k_heads;
                let value_dim = head_v_dim * n_v_heads;
                let conv_dim = key_dim * 2 + value_dim;

                // two HC modules per layer (qwen4exp.cpp:207-214)
                l.hc_attn_norm = ld.create_tensor(
                    LlmTensor::HC_ATTN_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd, hc],
                    flags | TENSOR_ALLOW_RESHAPE,
                )?;
                l.hc_attn_down = Some(reqf!(
                    LlmTensor::HC_ATTN_DOWN,
                    "weight",
                    bid,
                    &[hc_dim, hc_lr]
                ));
                l.hc_attn_up = Some(reqf!(LlmTensor::HC_ATTN_UP, "weight", bid, &[hc_lr, hc_dim]));
                l.hc_attn_inject = Some(reqf!(
                    LlmTensor::HC_ATTN_INJECT,
                    "weight",
                    bid,
                    &[hc_dim, hc]
                ));
                l.hc_ffn_norm = ld.create_tensor(
                    LlmTensor::HC_FFN_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd, hc],
                    flags | TENSOR_ALLOW_RESHAPE,
                )?;
                l.hc_ffn_down = Some(reqf!(
                    LlmTensor::HC_FFN_DOWN,
                    "weight",
                    bid,
                    &[hc_dim, hc_lr]
                ));
                l.hc_ffn_up = Some(reqf!(LlmTensor::HC_FFN_UP, "weight", bid, &[hc_lr, hc_dim]));
                l.hc_ffn_inject =
                    Some(reqf!(LlmTensor::HC_FFN_INJECT, "weight", bid, &[hc_dim, hc]));

                if !hparams.is_recr(i) {
                    // full attention: wq holds [q|gate] interleaved per head
                    // (qwen4exp.cpp:216-228)
                    create_tensor_qkv(
                        l,
                        ld,
                        bid,
                        lc.n_embd,
                        lc.n_embd_head_k * lc.n_head * 2,
                        lc.n_embd_k_gqa,
                        lc.n_embd_v_gqa,
                        flags,
                    )?;
                    l.wo = Some(reqf!(
                        LlmTensor::ATTN_OUT,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                    ));
                    l.attn_q_norm = Some(reqf!(
                        LlmTensor::ATTN_Q_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k]
                    ));
                    l.attn_k_norm = Some(reqf!(
                        LlmTensor::ATTN_K_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd_head_k]
                    ));

                    // the QSA indexer tensors (loaded 1:1; the graph reads
                    // them only under compress_ratios > 0 — not ported)
                    let idx_dim = hparams.indexer_head_size as i64;
                    l.index_q_proj = Some(reqf!(
                        LlmTensor::INDEXER_Q_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, hparams.indexer_n_head as i64 * idx_dim]
                    ));
                    l.index_k_proj = Some(reqf!(
                        LlmTensor::INDEXER_K_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, idx_dim]
                    ));
                    l.index_q_norm =
                        Some(reqf!(LlmTensor::INDEXER_Q_NORM, "weight", bid, &[idx_dim]));
                    l.index_k_norm =
                        Some(reqf!(LlmTensor::INDEXER_K_NORM, "weight", bid, &[idx_dim]));
                } else {
                    // GDN layers (qwen4exp.cpp:229-238)
                    l.wqkv = Some(reqf!(
                        LlmTensor::ATTN_QKV,
                        "weight",
                        bid,
                        &[lc.n_embd, key_dim * 2 + value_dim]
                    ));
                    l.wqkv_gate = Some(reqf!(
                        LlmTensor::ATTN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, value_dim]
                    ));
                    l.ssm_conv1d = Some(reqf!(
                        LlmTensor::SSM_CONV1D,
                        "weight",
                        bid,
                        &[hparams.ssm_d_conv as i64, conv_dim]
                    ));
                    l.ssm_dt_b = Some(reqf!(
                        LlmTensor::SSM_DT,
                        "bias",
                        bid,
                        &[hparams.ssm_dt_rank as i64]
                    ));
                    l.ssm_a = ld.create_tensor(
                        LlmTensor::SSM_A_NOSCAN,
                        "",
                        bid,
                        &[hparams.ssm_dt_rank as i64],
                        0,
                    )?;
                    l.ssm_beta = Some(reqf!(
                        LlmTensor::SSM_BETA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads]
                    ));
                    l.ssm_alpha = Some(reqf!(
                        LlmTensor::SSM_ALPHA,
                        "weight",
                        bid,
                        &[lc.n_embd, n_v_heads]
                    ));
                    l.ssm_norm = Some(reqf!(LlmTensor::SSM_NORM, "weight", bid, &[head_v_dim]));
                    l.ssm_out = Some(reqf!(
                        LlmTensor::SSM_OUT,
                        "weight",
                        bid,
                        &[value_dim, lc.n_embd]
                    ));
                }

                // the MoE + gated shared expert (qwen4exp.cpp:250-257)
                l.ffn_gate_inp = Some(reqf!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_down_exps = Some(reqf!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_up_exps = opt!(
                    LlmTensor::FFN_GATE_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * n_ff_exp, lc.n_expert]
                );
                if l.ffn_gate_up_exps.is_none() {
                    l.ffn_gate_exps = Some(reqf!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(reqf!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                }
                l.ffn_gate_inp_shexp = Some(reqf!(
                    LlmTensor::FFN_GATE_INP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                l.ffn_gate_shexp = Some(reqf!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
                l.ffn_up_shexp = Some(reqf!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                ));
                l.ffn_down_shexp = Some(reqf!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                ));

                // the PLE module of the layer (qwen4exp.cpp:244-251)
                if hparams.is_ple(i) {
                    l.ple_key = Some(reqf!(
                        LlmTensor::PLE_KEY,
                        "weight",
                        bid,
                        &[lc.n_embd, hc_dim]
                    ));
                    l.ple_value = Some(reqf!(
                        LlmTensor::PLE_VALUE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_embd]
                    ));
                    l.ple_norm_key = ld.create_tensor(
                        LlmTensor::PLE_NORM_KEY,
                        "weight",
                        bid,
                        &[lc.n_embd, hc],
                        flags | TENSOR_ALLOW_RESHAPE,
                    )?;
                    l.ple_norm_query = ld.create_tensor(
                        LlmTensor::PLE_NORM_QUERY,
                        "weight",
                        bid,
                        &[lc.n_embd, hc],
                        flags | TENSOR_ALLOW_RESHAPE,
                    )?;
                    l.ple_norm_conv = ld.create_tensor(
                        LlmTensor::PLE_NORM_CONV,
                        "weight",
                        bid,
                        &[lc.n_embd, hc],
                        flags | TENSOR_ALLOW_RESHAPE,
                    )?;
                    l.ple_conv1d = Some(reqf!(
                        LlmTensor::PLE_CONV1D,
                        "weight",
                        bid,
                        &[hparams.ple_conv_kernel as i64, hc_dim]
                    ));
                }
            }

            // the MTP block's own tensors (a7b94df2c qwen4exp.cpp:274-285):
            // one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] ->
            // eh_proj per hc stream
            for (i, l) in layers.iter_mut().enumerate().skip(lc.n_layer) {
                let bid = i as i32;
                let flags = mtp_flags;
                macro_rules! reqf2 {
                    ($t:expr, $suf:expr, $bid2:expr, $ne:expr) => {
                        ld.create_tensor($t, $suf, $bid2, $ne, flags)?.unwrap()
                    };
                }
                l.nextn.eh_proj = Some(reqf2!(
                    LlmTensor::NEXTN_EH_PROJ,
                    "weight",
                    bid,
                    &[2 * lc.n_embd, lc.n_embd]
                ));
                l.nextn.enorm = Some(reqf2!(
                    LlmTensor::NEXTN_ENORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                // RMS per hc stream of the trunk residual, so the gammas load
                // as [n_embd, hc] like the mixer norms
                l.nextn.hnorm = ld.create_tensor(
                    LlmTensor::NEXTN_HNORM,
                    "weight",
                    bid,
                    &[lc.n_embd, hc],
                    mtp_flags | TENSOR_ALLOW_RESHAPE,
                )?;
                l.nextn.hc_head_norm = ld.create_tensor(
                    LlmTensor::NEXTN_HC_HEAD_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd, hc],
                    mtp_flags | TENSOR_ALLOW_RESHAPE,
                )?;
                l.nextn.hc_head_down = Some(reqf2!(
                    LlmTensor::NEXTN_HC_HEAD_DOWN,
                    "weight",
                    bid,
                    &[hc_dim, hc_lr]
                ));
                l.nextn.hc_head_up = Some(reqf2!(
                    LlmTensor::NEXTN_HC_HEAD_UP,
                    "weight",
                    bid,
                    &[hc_lr, hc_dim]
                ));
            }

            // the flat [ple_head_dim, n_rows] gather table
            // (qwen4exp.cpp:173-193) — the head ranges are what the gather
            // indexes, so they set the minimum row count; the converter pads
            // the table, a synthesised file just carries the minimum (the
            // port's loader has no pre-table to consult, :180-189)
            if hparams.ple_n_heads > 0 {
                let mut ple_rows = 0i64;
                for h in 0..hparams.ple_n_heads as usize {
                    ple_rows = ple_rows.max(
                        hparams.ple_head_offsets[h] as i64
                            + hparams.ple_head_vocab_sizes[h] as i64,
                    );
                }
                extra.per_layer_tok_embd = opt!(LlmTensor::PER_LAYER_TOKEN_EMBD, "weight", -1, &[hparams.ple_head_dim as i64, ple_rows]);
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- arch batch 11b (2026-10): the long-tail queue, second half ----

        // ---- models/arcee.cpp:13-42 ----
        LlmArch::ARCEE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // arcee.cpp:19-25 — optional head with the tok_embd tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head,
                // n_embd_k_gqa, n_embd_v_gqa, 0) — arcee.cpp:32
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // arcee.cpp:37 — rope freq factors {n_rot/2}, created on layer
                // 0 and duplicated everywhere else (the exaone pattern)
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_freqs = ld.create_tensor(
                    LlmTensor::ROPE_FREQS,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/jais2.cpp:13-46 ----
        LlmArch::JAIS2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // jais2.cpp:19-24 — LN pair + optional head with the tie fallback
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                // Jais-2 uses simple MLP (no gate) with biases (jais2.cpp:41-44)
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_b = Some(req!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_b = Some(req!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]));
            }

            extra.output_norm_b = Some(output_norm_b);
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/talkie.cpp:13-34 ----
        LlmArch::TALKIE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );
            // talkie.cpp:17 — output REQUIRED (no tie fallback); there is no
            // `output_norm` tensor either (talkie.cpp's graph runs a weightless
            // final RMS, :137). The decode path never reads `output_norm` for
            // this arch (ForwardWeights::Talkie carries no norm), so the struct
            // slot aliases tok_embd the way the BERT arm does.
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_norm = tok_embd;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // no k gain — the q norm is one scalar per head (talkie.cpp:26)
                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[1, lc.n_head]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));

                l.out_scale = Some(req!(LlmTensor::LAYER_OUT_SCALE, "weight", bid, &[1]));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/nanbeige.cpp:37-74 ----
        LlmArch::NANBEIGE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // the physical stack the file carries; the loop slots alias it
            let n_phys = if hparams.nanbeige_n_layer_phys > 0 {
                hparams.nanbeige_n_layer_phys as usize
            } else {
                lc.n_layer_all
            };
            for (i, l) in layers.iter_mut().enumerate().take(n_phys) {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // nanbeige.cpp:57-58 — optional rope freq factors (duplicated
                // off layer 0), the exaone/arcee pattern
                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_freqs = ld.create_tensor(
                    LlmTensor::ROPE_FREQS,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            // share physical weights across loops; each slot still has its own
            // KV index (nanbeige.cpp:66-73)
            let n_loops = if hparams.nanbeige_n_loops > 0 {
                hparams.nanbeige_n_loops as usize
            } else {
                1
            };
            if n_loops > 1 {
                for j in 1..n_loops {
                    for i in 0..n_phys {
                        layers[i + j * n_phys] = layers[i];
                    }
                }
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/dream.cpp:18-46 (llada-family diffusion) ----
        LlmArch::DREAM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_b = opt!(LlmTensor::OUTPUT, "bias", -1, &[lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // dream.cpp:37 — q width n_embd (head_dim * n_head == n_embd)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }

            (tok_embd, output_norm, None, output.unwrap(), output_b)
        }

        // ---- models/rnd1.cpp:16-58 (qwen3moe converted to diffusion) ----
        LlmArch::RND1 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            if lc.n_expert == 0 {
                return Err("n_expert must be > 0 for QWEN3MOE".to_string());
            }
            if lc.n_expert_used == 0 {
                return Err("n_expert_used must be > 0 for QWEN3MOE".to_string());
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                // MoE branch — n_ff_exp falls back to n_ff / n_expert_used
                // (rnd1.cpp:52)
                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/eurobert.cpp:11-32 (encoder) ----
        LlmArch::EUROBERT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // eurobert.cpp:23 — q width n_embd, like bert
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
            }

            // the encoder has no output head (res->t_embd only, eurobert.cpp:
            // 121) — the decode-path `output` slot aliases tok_embd like BERT
            (
                tok_embd,
                output_norm,
                None,
                dup_fallback!((lc.n_embd, lc.n_vocab)),
                None,
            )
        }

        // ---- arch batch 12 (2026-10): the final long-tail queue ----

        // ---- models/hrm-text.cpp:34-86 ----
        LlmArch::HRM_TEXT => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // hrm-text.cpp:39-44 — optional head with the tok_embd tie fallback
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // the learned low-cycle state (hrm-text.cpp:46, def4d406a) —
            // `tn(LLM_TENSOR_HRM_Z_L_INIT, 0)` with NO suffix: the bare
            // "hrm.z_l_init" name (the layer class became REPEATING in
            // llama-arch.cpp:731, so the bid is 0 now; the template carries
            // no %d, the name is unchanged)
            extra.hrm_z_l_init = Some(
                ld.create_tensor(LlmTensor::HRM_Z_L_INIT, "", 0, &[lc.n_embd], 0)?
                    .unwrap(),
            );

            // blocks [0, lps) hold the low stack, blocks [lps, 2*lps) the high
            // stack; the first low and high passes create the layers, later
            // passes alias them (hrm-text.cpp:48-85)
            let lps = hparams.n_hrm_layers_per_stack as usize;
            let n_l_cycles = hparams.n_hrm_l_cycles as usize;
            let n_h_cycles = hparams.n_hrm_h_cycles as usize;
            let l_first = 0usize;
            let h_first = n_l_cycles * lps;

            for h in 0..n_h_cycles {
                for l in 0..(n_l_cycles + 1) {
                    let slot_base = (h * (n_l_cycles + 1) + l) * lps;
                    let blk_base = if l == n_l_cycles { lps } else { 0 };

                    if h > 0 || (l > 0 && l < n_l_cycles) {
                        // alias pass: these cache slots hold the same layers
                        // as the first passes (hrm-text.cpp:60-67)
                        let src_base = if l == n_l_cycles { h_first } else { l_first };
                        for il in 0..lps {
                            layers[slot_base + il] = layers[src_base + il];
                        }
                        continue;
                    }

                    for il in 0..lps {
                        let l = &mut layers[slot_base + il];
                        let bid = (blk_base + il) as i32;

                        create_tensor_qkv(
                            l,
                            ld,
                            bid,
                            lc.n_embd,
                            lc.n_embd_head_k * lc.n_head,
                            lc.n_embd_k_gqa,
                            lc.n_embd_v_gqa,
                            0,
                        )?;

                        // sigmoid attention gate, applied to the attention
                        // output before o_proj (hrm-text.cpp:75-78)
                        l.wqkv_gate = Some(req!(
                            LlmTensor::ATTN_GATE,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_embd_head_k * lc.n_head]
                        ));
                        l.wo = Some(req!(
                            LlmTensor::ATTN_OUT,
                            "weight",
                            bid,
                            &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                        ));

                        l.ffn_gate = Some(req!(
                            LlmTensor::FFN_GATE,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_ff]
                        ));
                        l.ffn_down = Some(req!(
                            LlmTensor::FFN_DOWN,
                            "weight",
                            bid,
                            &[lc.n_ff, lc.n_embd]
                        ));
                        l.ffn_up = Some(req!(
                            LlmTensor::FFN_UP,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_ff]
                        ));
                    }
                }
            }

            // the hrm-text stack norms are parameterless — no output_norm
            // tensor exists; the decode-path slot reuses tok_embd (unused)
            (tok_embd, tok_embd, None, output.unwrap(), None)
        }

        // ---- models/laguna.cpp:65-146 ----
        LlmArch::LAGUNA => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                // tied embeddings fallback (laguna.cpp:72-75)
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_ff_shexp = hparams.n_ff_shexp as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // per-layer head count — Laguna varies n_head between the full
                // and the SWA layers; the KV head count is uniform
                // (laguna.cpp:83-89)
                let n_head_il = hparams.n_head(i) as i64;
                let n_head_kv_il = hparams.n_head_kv(i) as i64;
                let n_embd_q_il = lc.n_embd_head_k * n_head_il;
                let n_embd_k_il = lc.n_embd_head_k * n_head_kv_il;
                let n_embd_v_il = lc.n_embd_head_v * n_head_kv_il;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    n_embd_q_il,
                    n_embd_k_il,
                    n_embd_v_il,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_embd_q_il, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                // attention output gate (laguna.cpp:99-122): the stored tensor
                // width picks the per-head (XS.2) vs per-element (M.1) layout;
                // require EXACTLY one of the two valid widths, else the
                // per-head fallback of the weightless fixtures
                let n_gate_per_head = n_head_il;
                let n_gate_per_elem = lc.n_embd_head_k * n_head_il;
                let gate_name = tensor_name_suffix(LlmTensor::ATTN_GATE, "weight", bid, -1);
                let n_gate_out = match ld.gguf.find_tensor(&gate_name) {
                    Some(ti) => {
                        let w = ti.ne[1];
                        if w != n_gate_per_head && w != n_gate_per_elem {
                            return Err(format!(
                                "Laguna: unexpected attention gate width {w} at layer {i} \
                                 (expected {n_gate_per_head} per-head or {n_gate_per_elem} per-element)"
                            ));
                        }
                        w
                    }
                    None => n_gate_per_head,
                };
                l.wqkv_gate = Some(req!(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, n_gate_out]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                let n_ff_exp = hparams.n_ff_exp(i) as i64;
                if i >= hparams.n_layer_dense_lead as usize {
                    // MoE layer (laguna.cpp:126-138)
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));

                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));

                    // always-on shared expert (laguna.cpp:135-138)
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd]
                    ));
                } else {
                    // dense layer — the leading n_layer_dense_lead layers
                    // (laguna.cpp:139-144)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                }
            }

            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/maple.cpp:24-59 ----
        LlmArch::MAPLE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // maple.cpp:32-33 — output_norm + output REQUIRED (no fallback)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            if lc.n_expert == 0 {
                return Err("n_expert must be > 0 for Maple".to_string());
            }
            if lc.n_expert_used == 0 {
                return Err("n_expert_used must be > 0 for Maple".to_string());
            }

            let n_ff_exp = hparams.n_ff_exp(0) as i64;
            let head_dim = lc.n_embd_head_k;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // maple.cpp:47 — v width n_head_kv * head_dim (same as k)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_head * head_dim,
                    lc.n_head_kv * head_dim,
                    lc.n_head_kv * head_dim,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * head_dim, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[head_dim]));
                l.attn_k_norm = Some(req!(LlmTensor::ATTN_K_NORM, "weight", bid, &[head_dim]));
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }

            (tok_embd, output_norm, None, output, None)
        }

        // ---- arch batch 13 (2026-09): the P0 standard-attention queue ----

        // ---- models/cohere2.cpp:21-44 ----
        LlmArch::COHERE2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // output — init from the input tok embed (cohere2.cpp:27-30)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/chatglm.cpp:25-52 ----
        LlmArch::CHATGLM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            // if output is NULL, init from the input tok embed
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                // chatglm.cpp:42 — q is n_embd_head_k * n_head wide
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // {n_embd, n_ff * 2} — the fused gate|up tensor (chatglm.cpp:48)
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff * 2]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/bitnet.cpp:12-45 ----
        LlmArch::BITNET => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // bitnet has no output tensor at all — the graph multiplies
            // tok_embd itself (bitnet.cpp:163-164)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = tok_embd;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_sub_norm = Some(req!(LlmTensor::ATTN_SUB_NORM, "weight", bid, &[lc.n_embd]));

                l.wq = Some(req!(
                    LlmTensor::ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wq_s = opt!(LlmTensor::ATTN_Q, "scale", bid, &[1]);
                l.wk = Some(req!(
                    LlmTensor::ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.wk_s = opt!(LlmTensor::ATTN_K, "scale", bid, &[1]);
                l.wv = Some(req!(
                    LlmTensor::ATTN_V,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_v_gqa]
                ));
                l.wv_s = opt!(LlmTensor::ATTN_V, "scale", bid, &[1]);
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
                l.wo_s = opt!(LlmTensor::ATTN_OUT, "scale", bid, &[1]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_sub_norm = Some(req!(LlmTensor::FFN_SUB_NORM, "weight", bid, &[lc.n_ff]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_gate_s = opt!(LlmTensor::FFN_GATE, "scale", bid, &[1]);
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_down_s = opt!(LlmTensor::FFN_DOWN, "scale", bid, &[1]);
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up_s = opt!(LlmTensor::FFN_UP, "scale", bid, &[1]);
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/dbrx.cpp:13-41 ----
        LlmArch::DBRX => {
            if lc.n_expert == 0 {
                return Err("DBRX model cannot have zero experts".to_string());
            }

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd + 2 * lc.n_embd_v_gqa]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.attn_out_norm = Some(req!(LlmTensor::ATTN_OUT_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));
                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/ernie4-5.cpp:23-70 (the dense ERNIE4_5 files; the MoE
        // arm above is the ERNIE4_5_MOE twin — the C class pair shares this
        // loader, `arch == LLM_ARCH_ERNIE4_5_MOE` selects the MoE branch) ----
        LlmArch::ERNIE4_5 | LlmArch::PADDLEOCR => {
            // PADDLEOCR reuses ernie4_5's loader (models.h:1984-1986); its
            // own graph is build_paddleocr_forward (the mrope sections ride
            // the rope input)
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // optional bias tensors (ernie4-5.cpp:45)
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // Dense layers — every layer on the dense files (the C's
                // `arch == LLM_ARCH_ERNIE4_5_MOE && i >= n_layer_dense_lead`
                // guard is false throughout)
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/mistral3.cpp:29-87 ----
        LlmArch::MISTRAL3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // optional bias tensors
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                    l.rope_short = ld.create_tensor(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                } else {
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                if lc.n_expert == 0 {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));

                    // optional MLP bias
                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    );
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff, lc.n_expert]
                    ));

                    // For Granite MoE Shared
                    if hparams.n_ff_shexp > 0 {
                        let n_ff_shexp = hparams.n_ff_shexp as i64;
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd]
                        ));
                    }
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/minicpm3.cpp:14-57 ----
        LlmArch::MINICPM3 => {
            let n_embd_head_qk_rope = hparams.n_rot(0) as i64;
            let n_embd_head_qk_nope = (hparams.n_embd_head_k(0) - hparams.n_rot(0)) as i64;

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_q_a_norm = Some(req!(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank]
                ));
                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));

                l.wq_a = Some(req!(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank]
                ));
                l.wq_b = Some(req!(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, lc.n_head * lc.n_embd_head_k]
                ));

                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));
                l.wkv_b = Some(req!(
                    LlmTensor::ATTN_KV_B,
                    "weight",
                    bid,
                    &[
                        kv_lora_rank,
                        lc.n_head * (n_embd_head_qk_nope + lc.n_embd_head_v)
                    ]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * lc.n_embd_head_v, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_long = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_LONG,
                    "weight",
                    bid,
                    &[n_embd_head_qk_rope / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
                l.rope_short = ld.create_tensor(
                    LlmTensor::ROPE_FACTORS_SHORT,
                    "weight",
                    bid,
                    &[n_embd_head_qk_rope / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/glm4.cpp:15-62 ----
        LlmArch::GLM4 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // NextN/MTP layers load with TENSOR_SKIP in C (glm4.cpp:28-33);
                // the port loads trunk-only files (the skip degrades to
                // NOT_REQUIRED, deepseek4's convention)
                let skip = i >= lc.n_layer;

                l.attn_norm = opt_or_req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd], skip);
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    if skip { TENSOR_NOT_REQUIRED } else { 0 },
                )?;
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd],
                    skip
                );

                l.attn_post_norm =
                    opt_or_req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                l.ffn_norm = opt_or_req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd], skip);
                l.ffn_down = opt_or_req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd],
                    skip
                );
                // {n_embd, n_ff * 2} — the fused gate|up tensor (glm4.cpp:46)
                l.ffn_up = opt_or_req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff * 2],
                    skip
                );

                l.ffn_post_norm =
                    opt_or_req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                // NextN/MTP tensors (preserved but unused) — conditionally
                // loaded for the last nextn_predict_layers (glm4.cpp:50-60)
                if i >= lc.n_layer {
                    l.nextn.eh_proj = opt_or_req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd],
                        skip
                    );
                    l.nextn.enorm =
                        opt_or_req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd], skip);
                    l.nextn.hnorm =
                        opt_or_req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd], skip);

                    // Optional tensors
                    l.nextn.embed_tokens = ld.create_tensor(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.nextn.shared_head_head = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab],
                        TENSOR_NOT_REQUIRED,
                    )?;
                    l.nextn.shared_head_norm = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        TENSOR_NOT_REQUIRED,
                    )?;
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/exaone4.cpp:24-71 ----
        LlmArch::EXAONE4 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // NextN/MTP layers preserved in GGUF but not executed
                // (exaone4.cpp:38-44); the port loads trunk-only files
                let is_nextn = i >= lc.n_layer;
                let skip = is_nextn;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    if skip { TENSOR_NOT_REQUIRED } else { 0 },
                )?;
                l.wo = opt_or_req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd],
                    skip
                );

                if !is_nextn {
                    let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[lc.n_rot / 2],
                        TENSOR_NOT_REQUIRED | dup_if_not_first,
                    )?;
                }

                l.attn_post_norm =
                    opt_or_req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd], skip);
                l.attn_q_norm = opt_or_req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k],
                    skip
                );
                l.attn_k_norm = opt_or_req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k],
                    skip
                );

                l.ffn_gate = opt_or_req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff],
                    skip
                );
                l.ffn_down = opt_or_req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd],
                    skip
                );
                l.ffn_up = opt_or_req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff],
                    skip
                );
                l.ffn_post_norm =
                    opt_or_req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd], skip);

                if is_nextn {
                    l.nextn.eh_proj = opt_or_req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd],
                        skip
                    );
                    l.nextn.enorm =
                        opt_or_req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd], skip);
                    l.nextn.hnorm =
                        opt_or_req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd], skip);
                    l.nextn.shared_head_norm = ld.create_tensor(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd],
                        TENSOR_NOT_REQUIRED,
                    )?;
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/llama4.cpp:44-94 ----
        LlmArch::LLAMA4 => {
            if lc.n_expert == 0 {
                return Err(format!("{} model cannot have zero experts", arch.name()));
            }
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // llama4.cpp:62 — the every-step MoE interleave
                let is_moe_layer =
                    hparams.n_moe_layer_step > 0 && (i as u32 + 1) % hparams.n_moe_layer_step == 0;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                let dup_if_not_first = if i != 0 { TENSOR_DUPLICATED } else { 0 };
                l.rope_freqs = ld.create_tensor(
                    LlmTensor::ROPE_FREQS,
                    "weight",
                    bid,
                    &[lc.n_rot / 2],
                    TENSOR_NOT_REQUIRED | dup_if_not_first,
                )?;

                if is_moe_layer {
                    let n_ff_exp = hparams.n_ff_exp(i) as i64;

                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // Shared expert — n_ff_shexp == n_ff_exp (llama4.cpp:84)
                    let n_ff_shexp = n_ff_exp;
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_shexp, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_shexp]
                    ));
                } else {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/qwen2vl.cpp:8-36 ----
        LlmArch::QWEN2VL => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            let output_b = opt!(LlmTensor::OUTPUT, "bias", -1, &[lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // qwen2vl.cpp:27 — q is n_embd wide (not head*n_head)
                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, output_b, output.unwrap(), None)
        }

        // ---- models/qwen3vl.cpp:16-54 ----
        // (QWEN3TTS is a pure typedef — models.h:625-627 reuses qwen3vl's
        // hparams/tensors/graph wholesale, qwen3tts.cpp:3)
        LlmArch::QWEN3VL | LlmArch::QWEN3TTS => {
            // [TAG_LLAMA_N_VOCAB_OUT] qwen3vl.cpp:19-23 — the qwen3tts head
            // covers only the 3072 text tokens (the codec tokens never
            // surface as logits)
            let n_vocab_out = if arch == LlmArch::QWEN3TTS {
                3072
            } else {
                lc.n_vocab
            };
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, n_vocab_out]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            // output rerank head (qwen3vl.cpp:36 — loaded, unused by the
            // graph)
            let cls_out = opt!(
                LlmTensor::CLS_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.n_cls_out as i64]
            );

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), cls_out)
        }

        // ---- models/pockettts.cpp:16-42 ----
        LlmArch::POCKETTTS => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output_norm_b = opt!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            // no output head, the logits are unused; reuse the embedding
            // table so a sampler can still run (pockettts.cpp:23-24,
            // TENSOR_DUPLICATED)
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    TENSOR_NOT_REQUIRED,
                );

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_norm_b = Some(req!(LlmTensor::FFN_NORM, "bias", bid, &[lc.n_embd]));

                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            extra.output_norm_b = output_norm_b;
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/wavtokenizer-dec.cpp:9-112 ----
        LlmArch::WAVTOKENIZER_DEC => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            extra.conv1d = Some(req!(
                LlmTensor::CONV1D,
                "weight",
                0,
                &[7, lc.n_embd, hparams.posnet.n_embd as i64]
            ));
            extra.conv1d_b = Some(req!(
                LlmTensor::CONV1D,
                "bias",
                0,
                &[1, hparams.posnet.n_embd as i64]
            ));

            // posnet (wavtokenizer-dec.cpp:17-76): blocks 0/1/3/4 resnet,
            // 2 attention, 5 trailing group-norm
            {
                let n_embd = hparams.posnet.n_embd as i64;
                for (i, l) in
                    layers.iter_mut().enumerate().take(hparams.posnet.n_layer as usize)
                {
                    let bid = i as i32;
                    let mut p = PosnetTensors::default();
                    match i {
                        0 | 1 | 3 | 4 => {
                            p.norm1 = Some(req!(
                                LlmTensor::POS_NET_NORM1,
                                "weight",
                                bid,
                                &[1, n_embd]
                            ));
                            p.norm1_b = Some(req!(
                                LlmTensor::POS_NET_NORM1,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.conv1 = Some(req!(
                                LlmTensor::POS_NET_CONV1,
                                "weight",
                                bid,
                                &[3, n_embd, n_embd]
                            ));
                            p.conv1_b = Some(req!(
                                LlmTensor::POS_NET_CONV1,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.norm2 = Some(req!(
                                LlmTensor::POS_NET_NORM2,
                                "weight",
                                bid,
                                &[1, n_embd]
                            ));
                            p.norm2_b = Some(req!(
                                LlmTensor::POS_NET_NORM2,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.conv2 = Some(req!(
                                LlmTensor::POS_NET_CONV2,
                                "weight",
                                bid,
                                &[3, n_embd, n_embd]
                            ));
                            p.conv2_b = Some(req!(
                                LlmTensor::POS_NET_CONV2,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));
                        }
                        2 => {
                            p.attn_norm = Some(req!(
                                LlmTensor::POS_NET_ATTN_NORM,
                                "weight",
                                bid,
                                &[1, n_embd]
                            ));
                            p.attn_norm_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_NORM,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.attn_q = Some(req!(
                                LlmTensor::POS_NET_ATTN_Q,
                                "weight",
                                bid,
                                &[1, n_embd, n_embd]
                            ));
                            p.attn_q_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_Q,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.attn_k = Some(req!(
                                LlmTensor::POS_NET_ATTN_K,
                                "weight",
                                bid,
                                &[1, n_embd, n_embd]
                            ));
                            p.attn_k_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_K,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.attn_v = Some(req!(
                                LlmTensor::POS_NET_ATTN_V,
                                "weight",
                                bid,
                                &[1, n_embd, n_embd]
                            ));
                            p.attn_v_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_V,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));

                            p.attn_o = Some(req!(
                                LlmTensor::POS_NET_ATTN_OUT,
                                "weight",
                                bid,
                                &[1, n_embd, n_embd]
                            ));
                            p.attn_o_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_OUT,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));
                        }
                        5 => {
                            // the trailing norm reuses the ATTN_NORM slot
                            // (wavtokenizer-dec.cpp:70-71)
                            p.norm = Some(req!(
                                LlmTensor::POS_NET_ATTN_NORM,
                                "weight",
                                bid,
                                &[1, n_embd]
                            ));
                            p.norm_b = Some(req!(
                                LlmTensor::POS_NET_ATTN_NORM,
                                "bias",
                                bid,
                                &[1, n_embd]
                            ));
                        }
                        _ => return Err("wavtokenizer: unknown posnet layer".to_string()),
                    }
                    l.posnet = Some(p);
                }
            }

            // GGML_ASSERT(posnet.n_embd == convnext.n_embd)
            if hparams.posnet.n_embd != hparams.convnext.n_embd {
                return Err(
                    "wavtokenizer: posnet.embedding_length != convnext.embedding_length"
                        .to_string(),
                );
            }

            extra.tok_norm = Some(req!(
                LlmTensor::TOKEN_EMBD_NORM,
                "weight",
                0,
                &[hparams.posnet.n_embd as i64]
            ));
            extra.tok_norm_b = Some(req!(
                LlmTensor::TOKEN_EMBD_NORM,
                "bias",
                0,
                &[hparams.posnet.n_embd as i64]
            ));

            // convnext (wavtokenizer-dec.cpp:83-108)
            {
                let n_embd = hparams.convnext.n_embd as i64;
                for (i, l) in
                    layers.iter_mut().enumerate().take(hparams.convnext.n_layer as usize)
                {
                    let bid = i as i32;
                    let mut c = ConvnextTensors::default();

                    c.dw = Some(req!(LlmTensor::CONVNEXT_DW, "weight", bid, &[7, 1, n_embd]));
                    c.dw_b = Some(req!(LlmTensor::CONVNEXT_DW, "bias", bid, &[1, n_embd]));

                    c.norm = Some(req!(LlmTensor::CONVNEXT_NORM, "weight", bid, &[n_embd]));
                    c.norm_b = Some(req!(LlmTensor::CONVNEXT_NORM, "bias", bid, &[n_embd]));

                    c.pw1 = Some(req!(
                        LlmTensor::CONVNEXT_PW1,
                        "weight",
                        bid,
                        &[n_embd, lc.n_ff]
                    ));
                    c.pw1_b = Some(req!(LlmTensor::CONVNEXT_PW1, "bias", bid, &[lc.n_ff]));

                    c.pw2 = Some(req!(
                        LlmTensor::CONVNEXT_PW2,
                        "weight",
                        bid,
                        &[lc.n_ff, n_embd]
                    ));
                    c.pw2_b = Some(req!(LlmTensor::CONVNEXT_PW2, "bias", bid, &[n_embd]));

                    c.gamma = Some(req!(LlmTensor::CONVNEXT_GAMMA, "weight", bid, &[n_embd]));

                    l.convnext = Some(c);
                }

                // output norm + bias live in the convnext stack's tail
                // (wavtokenizer-dec.cpp:106-107)
                extra.output_norm_b = Some(req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[n_embd]));
            }

            let output_norm = req!(
                LlmTensor::OUTPUT_NORM,
                "weight",
                -1,
                &[hparams.convnext.n_embd as i64]
            );
            let output_b = Some(req!(
                LlmTensor::OUTPUT,
                "bias",
                -1,
                &[hparams.n_embd_out_impl as i64]
            ));
            let output = req!(
                LlmTensor::OUTPUT,
                "weight",
                -1,
                &[hparams.convnext.n_embd as i64, hparams.n_embd_out_impl as i64]
            );
            (tok_embd, output_norm, output_b, output, None)
        }

        // ---- models/qwen3vlmoe.cpp:16-58 ----
        LlmArch::QWEN3VLMOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            if lc.n_expert == 0 {
                return Err("n_expert must be > 0 for QWEN3MOE".to_string());
            }
            if lc.n_expert_used == 0 {
                return Err("n_expert_used must be > 0 for QWEN3MOE".to_string());
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_v_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                // MoE branch (qwen3vlmoe.cpp:52)
                let n_ff_exp = if hparams.n_ff_exp(i) != 0 {
                    hparams.n_ff_exp(i) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/glm-dsa.cpp:74-187 ----
        LlmArch::GLM_DSA => {
            let n_expert_shared = hparams.n_expert_shared as i64;

            if !hparams.is_mla() {
                return Err("GLM_DSA architecture requires MLA".to_string());
            }

            // note: these are the actual head sizes you get when treating as
            // MHA or after "decompression" using wv_b for MLA (glm-dsa.cpp:98-99)
            let n_embd_head_k_mla = hparams.n_embd_head_k_mla() as i64;
            let n_embd_head_v_mla = hparams.n_embd_head_v_mla() as i64;

            let n_embd_head_qk_rope = hparams.n_rot(0) as i64;
            let n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;

            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // try to load output.weight, if not found, use token_embd (tied)
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_q_a_norm = Some(req!(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank]
                ));
                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));

                l.wq_a = Some(req!(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank]
                ));
                l.wq_b = Some(req!(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, lc.n_head * n_embd_head_k_mla]
                ));

                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));

                // note: only old legacy GGUF files will have the unsplit wkv_b
                // tensor — glm-dsa always loads the split pair (glm-dsa.cpp:135)
                l.wk_b = Some(req!(
                    LlmTensor::ATTN_K_B,
                    "weight",
                    bid,
                    &[n_embd_head_qk_nope, kv_lora_rank, lc.n_head]
                ));
                l.wv_b = Some(req!(
                    LlmTensor::ATTN_V_B,
                    "weight",
                    bid,
                    &[kv_lora_rank, n_embd_head_v_mla, lc.n_head]
                ));

                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * n_embd_head_v_mla, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                // DSA indexer (glm-dsa.cpp:144-148 — all NOT_REQUIRED)
                l.indexer_k_norm = opt!(
                    LlmTensor::INDEXER_K_NORM,
                    "weight",
                    bid,
                    &[hparams.indexer_head_size as i64]
                );
                l.indexer_k_norm_b = opt!(
                    LlmTensor::INDEXER_K_NORM,
                    "bias",
                    bid,
                    &[hparams.indexer_head_size as i64]
                );
                l.indexer_proj = opt!(
                    LlmTensor::INDEXER_PROJ,
                    "weight",
                    bid,
                    &[lc.n_embd, hparams.indexer_n_head as i64]
                );
                l.indexer_attn_k = opt!(
                    LlmTensor::INDEXER_ATTN_K,
                    "weight",
                    bid,
                    &[lc.n_embd, hparams.indexer_head_size as i64]
                );
                l.indexer_attn_q_b = opt!(
                    LlmTensor::INDEXER_ATTN_Q_B,
                    "weight",
                    bid,
                    &[
                        q_lora_rank,
                        (hparams.indexer_n_head * hparams.indexer_head_size) as i64
                    ]
                );

                if (i as u32) < hparams.n_layer_dense_lead {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".to_string());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".to_string());
                    }

                    // MoE branch
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    // Shared expert branch — n_ff_exp * n_expert_shared wide
                    // (glm-dsa.cpp:170-172)
                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }

                // NextN/MTP tensors — the port loads trunk-only files (the C's
                // TENSOR_SKIP arm degrades to NOT_REQUIRED)
                if i >= lc.n_layer {
                    l.nextn.eh_proj = opt!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    );
                    l.nextn.enorm = opt!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.hnorm = opt!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]);
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ------------------------------------------------------------------------
        // arch batch 14 (2026-10) — the RWKV family + gemma3n
        // ------------------------------------------------------------------------

        // ---- models/rwkv6.cpp:26-88 ----
        LlmArch::RWKV6 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // Block 0, LN0 (rwkv6.cpp:31-33)
            extra.token_embd_norm =
                Some(req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]));
            extra.token_embd_norm_b =
                Some(req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]));

            // output (rwkv6.cpp:35-38) — the LLM_NORM pair with bias
            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            extra.output_norm_b = Some(req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]));
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let time_mix_extra_dim = hparams.time_mix_extra_dim as i64;
            let time_decay_extra_dim = hparams.time_decay_extra_dim as i64;
            let head_size = hparams.wkv_head_size as i64;
            let attn_hidden_size = lc.n_embd;
            let ffn_size = hparams.n_ff(0) as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                l.attn_norm_2 = Some(req!(LlmTensor::ATTN_NORM_2, "weight", bid, &[lc.n_embd]));
                l.attn_norm_2_b = Some(req!(LlmTensor::ATTN_NORM_2, "bias", bid, &[lc.n_embd]));

                l.time_mix_w1 = Some(req!(
                    LlmTensor::TIME_MIX_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, time_mix_extra_dim * 5]
                ));
                l.time_mix_w2 = Some(req!(
                    LlmTensor::TIME_MIX_W2,
                    "weight",
                    bid,
                    &[time_mix_extra_dim, lc.n_embd, 5]
                ));

                l.time_mix_lerp_x = Some(req!(
                    LlmTensor::TIME_MIX_LERP_X,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                ));
                l.time_mix_lerp_w = opt!(
                    LlmTensor::TIME_MIX_LERP_W,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                );
                l.time_mix_lerp_k = opt!(
                    LlmTensor::TIME_MIX_LERP_K,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                );
                l.time_mix_lerp_v = opt!(
                    LlmTensor::TIME_MIX_LERP_V,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                );
                l.time_mix_lerp_r = opt!(
                    LlmTensor::TIME_MIX_LERP_R,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                );
                l.time_mix_lerp_g = opt!(
                    LlmTensor::TIME_MIX_LERP_G,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                );
                l.time_mix_lerp_fused = opt!(
                    LlmTensor::TIME_MIX_LERP_FUSED,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1, 5]
                );
                // rwkv6.cpp:65 — at least one of the fused / plain lerps
                if l.time_mix_lerp_fused.is_none() && l.time_mix_lerp_w.is_none() {
                    return Err("rwkv6: needs time_mix_lerp_fused or time_mix_lerp_w".to_string());
                }

                l.time_mix_first = Some(req!(
                    LlmTensor::TIME_MIX_FIRST,
                    "weight",
                    bid,
                    &[head_size, lc.n_embd / head_size]
                ));
                l.time_mix_decay =
                    Some(req!(LlmTensor::TIME_MIX_DECAY, "weight", bid, &[lc.n_embd]));
                l.time_mix_decay_w1 = Some(req!(
                    LlmTensor::TIME_MIX_DECAY_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, time_decay_extra_dim]
                ));
                l.time_mix_decay_w2 = Some(req!(
                    LlmTensor::TIME_MIX_DECAY_W2,
                    "weight",
                    bid,
                    &[time_decay_extra_dim, attn_hidden_size]
                ));
                l.time_mix_key = Some(req!(
                    LlmTensor::TIME_MIX_KEY,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_value = Some(req!(
                    LlmTensor::TIME_MIX_VALUE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_receptance = Some(req!(
                    LlmTensor::TIME_MIX_RECEPTANCE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_gate = Some(req!(
                    LlmTensor::TIME_MIX_GATE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));

                l.time_mix_ln = Some(req!(LlmTensor::TIME_MIX_LN, "weight", bid, &[lc.n_embd]));
                l.time_mix_ln_b = Some(req!(LlmTensor::TIME_MIX_LN, "bias", bid, &[lc.n_embd]));
                l.time_mix_output = Some(req!(
                    LlmTensor::TIME_MIX_OUTPUT,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_hidden_size]
                ));

                l.channel_mix_lerp_k = Some(req!(
                    LlmTensor::CHANNEL_MIX_LERP_K,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                ));
                l.channel_mix_lerp_r = Some(req!(
                    LlmTensor::CHANNEL_MIX_LERP_R,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                ));

                l.channel_mix_key = Some(req!(
                    LlmTensor::CHANNEL_MIX_KEY,
                    "weight",
                    bid,
                    &[lc.n_embd, ffn_size]
                ));
                l.channel_mix_value = Some(req!(
                    LlmTensor::CHANNEL_MIX_VALUE,
                    "weight",
                    bid,
                    &[ffn_size, lc.n_embd]
                ));
                l.channel_mix_receptance = Some(req!(
                    LlmTensor::CHANNEL_MIX_RECEPTANCE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/rwkv6qwen2.cpp:26-78 ----
        LlmArch::RWKV6QWEN2 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            extra.output_norm_b = opt!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let time_mix_extra_dim = hparams.time_mix_extra_dim as i64;
            let time_decay_extra_dim = hparams.time_decay_extra_dim as i64;
            let head_size = hparams.wkv_head_size as i64;
            let attn_hidden_size = lc.n_embd;
            // rwkv6qwen2.cpp:39-44 — the GQA-style key/value width
            let attn_key_value_size =
                if lc.n_head_kv == 0 || attn_hidden_size / head_size == lc.n_head_kv {
                    attn_hidden_size
                } else {
                    lc.n_head_kv * head_size
                };

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.time_mix_w1 = Some(req!(
                    LlmTensor::TIME_MIX_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, time_mix_extra_dim * 5]
                ));
                l.time_mix_w2 = Some(req!(
                    LlmTensor::TIME_MIX_W2,
                    "weight",
                    bid,
                    &[time_mix_extra_dim, lc.n_embd, 5]
                ));

                l.time_mix_lerp_x = Some(req!(
                    LlmTensor::TIME_MIX_LERP_X,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                ));
                l.time_mix_lerp_fused = Some(req!(
                    LlmTensor::TIME_MIX_LERP_FUSED,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1, 5]
                ));

                l.time_mix_first = opt!(
                    LlmTensor::TIME_MIX_FIRST,
                    "weight",
                    bid,
                    &[head_size, lc.n_embd / head_size]
                );
                l.time_mix_decay =
                    Some(req!(LlmTensor::TIME_MIX_DECAY, "weight", bid, &[lc.n_embd]));
                l.time_mix_decay_w1 = Some(req!(
                    LlmTensor::TIME_MIX_DECAY_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, time_decay_extra_dim]
                ));
                l.time_mix_decay_w2 = Some(req!(
                    LlmTensor::TIME_MIX_DECAY_W2,
                    "weight",
                    bid,
                    &[time_decay_extra_dim, attn_hidden_size]
                ));
                l.time_mix_key = Some(req!(
                    LlmTensor::TIME_MIX_KEY,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_key_value_size]
                ));
                l.time_mix_value = Some(req!(
                    LlmTensor::TIME_MIX_VALUE,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_key_value_size]
                ));
                l.time_mix_receptance = Some(req!(
                    LlmTensor::TIME_MIX_RECEPTANCE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_gate = Some(req!(
                    LlmTensor::TIME_MIX_GATE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                // optional bias tensors (rwkv6qwen2.cpp:66-68)
                l.time_mix_key_b =
                    opt!(LlmTensor::TIME_MIX_KEY, "bias", bid, &[attn_key_value_size]);
                l.time_mix_value_b = opt!(
                    LlmTensor::TIME_MIX_VALUE,
                    "bias",
                    bid,
                    &[attn_key_value_size]
                );
                l.time_mix_receptance_b = opt!(
                    LlmTensor::TIME_MIX_RECEPTANCE,
                    "bias",
                    bid,
                    &[attn_hidden_size]
                );

                l.time_mix_output = Some(req!(
                    LlmTensor::TIME_MIX_OUTPUT,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_hidden_size]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/rwkv7.cpp:46-118 ----
        LlmArch::RWKV7 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            // Block 0, LN0 (rwkv7.cpp:51-53)
            extra.token_embd_norm =
                Some(req!(LlmTensor::TOKEN_EMBD_NORM, "weight", 0, &[lc.n_embd]));
            extra.token_embd_norm_b =
                Some(req!(LlmTensor::TOKEN_EMBD_NORM, "bias", 0, &[lc.n_embd]));

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            extra.output_norm_b = Some(req!(LlmTensor::OUTPUT_NORM, "bias", -1, &[lc.n_embd]));
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let n_lora_decay = hparams.n_lora_decay as i64;
            let n_lora_iclr = hparams.n_lora_iclr as i64;
            let n_lora_value_res_mix = hparams.n_lora_value_res_mix as i64;
            let n_lora_gate = hparams.n_lora_gate as i64;
            let attn_hidden_size = lc.n_embd;
            let ffn_size = hparams.n_ff(0) as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_norm_b = Some(req!(LlmTensor::ATTN_NORM, "bias", bid, &[lc.n_embd]));

                l.attn_norm_2 = Some(req!(LlmTensor::ATTN_NORM_2, "weight", bid, &[lc.n_embd]));
                l.attn_norm_2_b = Some(req!(LlmTensor::ATTN_NORM_2, "bias", bid, &[lc.n_embd]));

                l.time_mix_w0 = Some(req!(LlmTensor::TIME_MIX_W0, "weight", bid, &[lc.n_embd]));
                l.time_mix_w1 = Some(req!(
                    LlmTensor::TIME_MIX_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_decay]
                ));
                l.time_mix_w2 = Some(req!(
                    LlmTensor::TIME_MIX_W2,
                    "weight",
                    bid,
                    &[n_lora_decay, lc.n_embd]
                ));

                l.time_mix_a0 = Some(req!(LlmTensor::TIME_MIX_A0, "weight", bid, &[lc.n_embd]));
                l.time_mix_a1 = Some(req!(
                    LlmTensor::TIME_MIX_A1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_iclr]
                ));
                l.time_mix_a2 = Some(req!(
                    LlmTensor::TIME_MIX_A2,
                    "weight",
                    bid,
                    &[n_lora_iclr, lc.n_embd]
                ));

                // layer 0's v-triple is "actually not used" but still loaded
                // at the iclr width (rwkv7.cpp:84-93)
                let v_mid = if i == 0 {
                    n_lora_iclr
                } else {
                    n_lora_value_res_mix
                };
                l.time_mix_v0 = Some(req!(LlmTensor::TIME_MIX_V0, "weight", bid, &[lc.n_embd]));
                l.time_mix_v1 = Some(req!(
                    LlmTensor::TIME_MIX_V1,
                    "weight",
                    bid,
                    &[lc.n_embd, v_mid]
                ));
                l.time_mix_v2 = Some(req!(
                    LlmTensor::TIME_MIX_V2,
                    "weight",
                    bid,
                    &[v_mid, lc.n_embd]
                ));

                l.time_mix_g1 = Some(req!(
                    LlmTensor::TIME_MIX_G1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_gate]
                ));
                l.time_mix_g2 = Some(req!(
                    LlmTensor::TIME_MIX_G2,
                    "weight",
                    bid,
                    &[n_lora_gate, lc.n_embd]
                ));

                l.time_mix_lerp_fused = Some(req!(
                    LlmTensor::TIME_MIX_LERP_FUSED,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1, 6]
                ));

                l.time_mix_k_k = Some(req!(
                    LlmTensor::TIME_MIX_K_K,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));
                l.time_mix_k_a = Some(req!(
                    LlmTensor::TIME_MIX_K_A,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));
                l.time_mix_r_k = Some(req!(
                    LlmTensor::TIME_MIX_R_K,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));

                l.time_mix_key = Some(req!(
                    LlmTensor::TIME_MIX_KEY,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_value = Some(req!(
                    LlmTensor::TIME_MIX_VALUE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_receptance = Some(req!(
                    LlmTensor::TIME_MIX_RECEPTANCE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));

                l.time_mix_ln = Some(req!(LlmTensor::TIME_MIX_LN, "weight", bid, &[lc.n_embd]));
                l.time_mix_ln_b = Some(req!(LlmTensor::TIME_MIX_LN, "bias", bid, &[lc.n_embd]));
                l.time_mix_output = Some(req!(
                    LlmTensor::TIME_MIX_OUTPUT,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_hidden_size]
                ));

                l.channel_mix_lerp_k = Some(req!(
                    LlmTensor::CHANNEL_MIX_LERP_K,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1]
                ));

                l.channel_mix_key = Some(req!(
                    LlmTensor::CHANNEL_MIX_KEY,
                    "weight",
                    bid,
                    &[lc.n_embd, ffn_size]
                ));
                l.channel_mix_value = Some(req!(
                    LlmTensor::CHANNEL_MIX_VALUE,
                    "weight",
                    bid,
                    &[ffn_size, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/arwkv7.cpp:46-114 ----
        LlmArch::ARWKV7 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let n_lora_decay = hparams.n_lora_decay as i64;
            let n_lora_iclr = hparams.n_lora_iclr as i64;
            let n_lora_value_res_mix = hparams.n_lora_value_res_mix as i64;
            let n_lora_gate = hparams.n_lora_gate as i64;
            let attn_hidden_size = lc.n_embd;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.time_mix_w0 = Some(req!(LlmTensor::TIME_MIX_W0, "weight", bid, &[lc.n_embd]));
                l.time_mix_w1 = Some(req!(
                    LlmTensor::TIME_MIX_W1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_decay]
                ));
                l.time_mix_w2 = Some(req!(
                    LlmTensor::TIME_MIX_W2,
                    "weight",
                    bid,
                    &[n_lora_decay, lc.n_embd]
                ));

                l.time_mix_a0 = Some(req!(LlmTensor::TIME_MIX_A0, "weight", bid, &[lc.n_embd]));
                l.time_mix_a1 = Some(req!(
                    LlmTensor::TIME_MIX_A1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_iclr]
                ));
                l.time_mix_a2 = Some(req!(
                    LlmTensor::TIME_MIX_A2,
                    "weight",
                    bid,
                    &[n_lora_iclr, lc.n_embd]
                ));

                let v_mid = if i == 0 {
                    n_lora_iclr
                } else {
                    n_lora_value_res_mix
                };
                l.time_mix_v0 = Some(req!(LlmTensor::TIME_MIX_V0, "weight", bid, &[lc.n_embd]));
                l.time_mix_v1 = Some(req!(
                    LlmTensor::TIME_MIX_V1,
                    "weight",
                    bid,
                    &[lc.n_embd, v_mid]
                ));
                l.time_mix_v2 = Some(req!(
                    LlmTensor::TIME_MIX_V2,
                    "weight",
                    bid,
                    &[v_mid, lc.n_embd]
                ));

                l.time_mix_g1 = opt!(
                    LlmTensor::TIME_MIX_G1,
                    "weight",
                    bid,
                    &[lc.n_embd, n_lora_gate]
                );
                l.time_mix_g2 = opt!(
                    LlmTensor::TIME_MIX_G2,
                    "weight",
                    bid,
                    &[n_lora_gate, lc.n_embd]
                );

                // arwkv7.cpp:88-93 — try the 6-plane lerp_fused (gated), fall
                // back to the 5-plane one (ARWKV models may not have gates)
                l.time_mix_lerp_fused = match ld.create_tensor(
                    LlmTensor::TIME_MIX_LERP_FUSED,
                    "weight",
                    bid,
                    &[lc.n_embd, 1, 1, 6],
                    0,
                ) {
                    Ok(t) => Some(t.unwrap()),
                    Err(_) => Some(
                        ld.create_tensor(
                            LlmTensor::TIME_MIX_LERP_FUSED,
                            "weight",
                            bid,
                            &[lc.n_embd, 1, 1, 5],
                            0,
                        )?
                        .unwrap(),
                    ),
                };

                l.time_mix_k_k = Some(req!(
                    LlmTensor::TIME_MIX_K_K,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));
                l.time_mix_k_a = Some(req!(
                    LlmTensor::TIME_MIX_K_A,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));
                l.time_mix_r_k = Some(req!(
                    LlmTensor::TIME_MIX_R_K,
                    "weight",
                    bid,
                    &[attn_hidden_size]
                ));

                l.time_mix_key = Some(req!(
                    LlmTensor::TIME_MIX_KEY,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_value = Some(req!(
                    LlmTensor::TIME_MIX_VALUE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));
                l.time_mix_receptance = Some(req!(
                    LlmTensor::TIME_MIX_RECEPTANCE,
                    "weight",
                    bid,
                    &[attn_hidden_size, lc.n_embd]
                ));

                l.time_mix_ln = opt!(LlmTensor::TIME_MIX_LN, "weight", bid, &[lc.n_embd]);
                l.time_mix_ln_b = opt!(LlmTensor::TIME_MIX_LN, "bias", bid, &[lc.n_embd]);
                l.time_mix_output = Some(req!(
                    LlmTensor::TIME_MIX_OUTPUT,
                    "weight",
                    bid,
                    &[lc.n_embd, attn_hidden_size]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/gemma3n.cpp:21-76 ----
        LlmArch::GEMMA3N => {
            let n_altup = hparams.n_altup as i64;
            let laurel_rank = hparams.laurel_rank as i64;
            let n_embd_altup = hparams.n_embd_altup as i64;

            // output — fall back to the input tok embed (gemma3n.cpp:28-34)
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            extra.altup_proj = Some(req!(
                LlmTensor::ALTUP_PROJ,
                "weight",
                -1,
                &[lc.n_embd, lc.n_embd, n_altup - 1]
            ));
            extra.altup_unembd_proj = Some(req!(
                LlmTensor::ALTUP_UNEMBD_PROJ,
                "weight",
                -1,
                &[lc.n_embd, lc.n_embd, n_altup - 1]
            ));

            extra.per_layer_tok_embd = Some(req!(
                LlmTensor::PER_LAYER_TOKEN_EMBD,
                "weight",
                -1,
                &[n_embd_altup * lc.n_layer as i64, lc.n_vocab]
            ));
            extra.per_layer_model_proj = Some(req!(
                LlmTensor::PER_LAYER_MODEL_PROJ,
                "weight",
                0,
                &[lc.n_embd, n_embd_altup * lc.n_layer as i64]
            ));
            extra.per_layer_proj_norm = Some(req!(
                LlmTensor::PER_LAYER_PROJ_NORM,
                "weight",
                0,
                &[n_embd_altup]
            ));

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));

                // altup & laurel (gemma3n.cpp:63-74)
                l.per_layer_inp_gate = Some(req!(
                    LlmTensor::PER_LAYER_INP_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, n_embd_altup]
                ));
                l.per_layer_proj = Some(req!(
                    LlmTensor::PER_LAYER_PROJ,
                    "weight",
                    bid,
                    &[n_embd_altup, lc.n_embd]
                ));
                l.per_layer_post_norm = Some(req!(
                    LlmTensor::PER_LAYER_POST_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                l.altup_correct_coef = Some(req!(
                    LlmTensor::ALTUP_CORRECT_COEF,
                    "weight",
                    bid,
                    &[n_altup, n_altup]
                ));
                l.altup_correct_scale = Some(req!(
                    LlmTensor::ALTUP_CORRECT_SCALE,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                l.altup_predict_coef = Some(req!(
                    LlmTensor::ALTUP_PREDICT_COEF,
                    "weight",
                    bid,
                    &[n_altup, n_altup * n_altup]
                ));
                l.altup_router = Some(req!(
                    LlmTensor::ALTUP_ROUTER,
                    "weight",
                    bid,
                    &[lc.n_embd, n_altup]
                ));
                l.altup_router_norm = Some(req!(
                    LlmTensor::ALTUP_ROUTER_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
                l.laurel_l = Some(req!(
                    LlmTensor::LAUREL_L,
                    "weight",
                    bid,
                    &[lc.n_embd, laurel_rank]
                ));
                l.laurel_r = Some(req!(
                    LlmTensor::LAUREL_R,
                    "weight",
                    bid,
                    &[laurel_rank, lc.n_embd]
                ));
                l.laurel_post_norm = Some(req!(
                    LlmTensor::LAUREL_POST_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ==================================================================
        // arch batch 15 (2026-10) — the P1+P2 queue of parity/AUDIT_models.md
        // ==================================================================

        // ---- models/qwen.cpp:13-37 ----
        LlmArch::QWEN => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                // qwen.cpp:27-28 — the fused QKV + bias at the MHA width
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd * 3]
                ));
                l.wqkv_b = Some(req!(LlmTensor::ATTN_QKV, "bias", bid, &[lc.n_embd * 3]));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                // the half-width FFN (qwen.cpp:33-35)
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff / 2]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff / 2, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff / 2]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/maincoder.cpp:12-41 ----
        LlmArch::MAINCODER => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    // mellum.cpp:41 — n_embd_gqa for both K and V
                    lc.n_embd_k_gqa,
                    lc.n_embd_k_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/pangu-embed.cpp:13-52 ----
        LlmArch::PANGU_EMBED => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // pangu-embed.cpp:37 — the REQUIRED output bias
                l.wo_b = Some(req!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long =
                        opt!(LlmTensor::ROPE_FACTORS_LONG, "weight", bid, &[lc.n_rot / 2]);
                    l.rope_short = opt!(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2]
                    );
                } else {
                    l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);
                }

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/cogvlm.cpp:12-47 ----
        LlmArch::COGVLM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.wqkv = Some(req!(
                    LlmTensor::ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_head_k * lc.n_head * 3]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                // the vision-expert twins (cogvlm.cpp:33-34/:43-45)
                l.visexp_attn_wqkv = Some(req!(
                    LlmTensor::VISEXP_ATTN_QKV,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_head_k * lc.n_head * 3]
                ));
                l.visexp_attn_wo = Some(req!(
                    LlmTensor::VISEXP_ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                l.visexp_ffn_gate = Some(req!(
                    LlmTensor::VISEXP_FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.visexp_ffn_down = Some(req!(
                    LlmTensor::VISEXP_FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.visexp_ffn_up = Some(req!(
                    LlmTensor::VISEXP_FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/spark2-5.cpp:20-50 ----
        LlmArch::SPARK2_5 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                let n_head_i = hparams.n_head(i) as i64;
                let n_head_kv_i = hparams.n_head_kv(i) as i64;
                let n_embd_q = hparams.n_embd_head_k(i) as i64 * n_head_i;
                let n_embd_k = hparams.n_embd_head_k(i) as i64 * n_head_kv_i;
                let n_embd_v = hparams.n_embd_head_v(i) as i64 * n_head_kv_i;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                create_tensor_qkv(l, ld, bid, lc.n_embd, n_embd_q, n_embd_k, n_embd_v, 0)?;
                l.wqkv_gate = Some(req!(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, n_head_i]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[n_embd_q, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/muse-glimmer.cpp:21-55 ----
        LlmArch::MUSE_GLIMMER => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                // muse-glimmer.cpp:31-33 — the weight+1 norms folded at
                // conversion time
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                // the wide attention gate (muse-glimmer.cpp:44)
                l.wqkv_gate = Some(req!(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_head_k * lc.n_head]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/llada.cpp:19-60 ----
        LlmArch::LLADA => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));
                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));

                // the optional MLP biases the graph never reads (llada.cpp:53-58)
                l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/plm.cpp:13-42 ----
        LlmArch::PLM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            // plm.cpp:24-25 — tied to tok_embd outright
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            let n_embd_head_qk_rope = hparams.n_rot(0) as i64;
            let n_embd_head_qk_nope = lc.n_embd_head_k - n_embd_head_qk_rope;
            let kv_lora_rank = hparams.n_lora_kv as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                l.wq = Some(req!(
                    LlmTensor::ATTN_Q,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_head_k * lc.n_head]
                ));
                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));
                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));
                l.wkv_b = Some(req!(
                    LlmTensor::ATTN_KV_B,
                    "weight",
                    bid,
                    &[
                        kv_lora_rank,
                        lc.n_head * (n_embd_head_qk_nope + lc.n_embd_head_v)
                    ]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * lc.n_embd_head_v, lc.n_embd]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/hunyuan-vl.cpp:23-54 (+ hunyuan-dense typedef) ----
        LlmArch::HUNYUAN_VL | LlmArch::HUNYUAN_DENSE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/granite-swa.cpp:56-115 ----
        LlmArch::GRANITE_SWA => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.wo_b = opt!(LlmTensor::ATTN_OUT, "bias", bid, &[lc.n_embd]);

                // the per-layer attention sinks (granite-swa.cpp:82, REQUIRED)
                l.attn_sinks = Some(req!(LlmTensor::ATTN_SINKS, "weight", bid, &[lc.n_head]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long =
                        opt!(LlmTensor::ROPE_FACTORS_LONG, "weight", bid, &[lc.n_rot / 2]);
                    l.rope_short = opt!(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[lc.n_rot / 2]
                    );
                } else {
                    l.rope_freqs = opt!(LlmTensor::ROPE_FREQS, "weight", bid, &[lc.n_rot / 2]);
                }

                if lc.n_expert == 0 {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));

                    l.ffn_gate_b = opt!(LlmTensor::FFN_GATE, "bias", bid, &[lc.n_ff]);
                    l.ffn_down_b = opt!(LlmTensor::FFN_DOWN, "bias", bid, &[lc.n_embd]);
                    l.ffn_up_b = opt!(LlmTensor::FFN_UP, "bias", bid, &[lc.n_ff]);
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd, lc.n_expert]
                    ));
                    // create_tensor_gate_up_exps — the fused tensor wins
                    l.ffn_gate_up_exps = opt!(
                        LlmTensor::FFN_GATE_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, 2 * lc.n_ff, lc.n_expert]
                    );
                    if l.ffn_gate_up_exps.is_none() {
                        l.ffn_gate_exps = Some(req!(
                            LlmTensor::FFN_GATE_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_ff, lc.n_expert]
                        ));
                        l.ffn_up_exps = Some(req!(
                            LlmTensor::FFN_UP_EXPS,
                            "weight",
                            bid,
                            &[lc.n_embd, lc.n_ff, lc.n_expert]
                        ));
                    }

                    // the SWIGLU-fused shared expert (granite-swa.cpp:109-112)
                    if hparams.n_ff_shexp > 0 {
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, 2 * hparams.n_ff_shexp as i64]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[hparams.n_ff_shexp as i64, lc.n_embd]
                        ));
                    }
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/afmoe.cpp:38-101 ----
        LlmArch::AFMOE => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_expert_shared = hparams.n_expert_shared as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;

                // the dual attention normalization (afmoe.cpp:59-60)
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.wqkv_gate = Some(req!(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_embd_head_k * lc.n_head]
                ));

                // the dual ffn normalization (:74-75)
                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));

                if i >= hparams.n_layer_dense_lead as usize {
                    // MoE layers (:79-93)
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b = Some(req!(
                        LlmTensor::FFN_EXP_PROBS_B,
                        "bias",
                        bid,
                        &[lc.n_expert]
                    ));

                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));

                    if n_expert_shared > 0 {
                        let n_ff_shexp = n_ff_exp * n_expert_shared;
                        l.ffn_gate_shexp = Some(req!(
                            LlmTensor::FFN_GATE_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                        l.ffn_down_shexp = Some(req!(
                            LlmTensor::FFN_DOWN_SHEXP,
                            "weight",
                            bid,
                            &[n_ff_shexp, lc.n_embd]
                        ));
                        l.ffn_up_shexp = Some(req!(
                            LlmTensor::FFN_UP_SHEXP,
                            "weight",
                            bid,
                            &[lc.n_embd, n_ff_shexp]
                        ));
                    }
                } else {
                    // dense layers (:96-98)
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                }
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/mellum.cpp:27-64 ----
        LlmArch::MELLUM => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    // mellum.cpp:41 — n_embd_gqa for both K and V
                    lc.n_embd_k_gqa,
                    lc.n_embd_k_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate_inp = Some(req!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                ));

                if lc.n_expert == 0 {
                    return Err("n_expert must be > 0 for Mellum".to_string());
                }
                if lc.n_expert_used == 0 {
                    return Err("n_expert_used must be > 0 for Mellum".to_string());
                }

                let n_ff_exp = if hparams.n_ff_exp(0) > 0 {
                    hparams.n_ff_exp(0) as i64
                } else {
                    lc.n_ff / lc.n_expert_used
                };

                l.ffn_gate_exps = Some(req!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
                l.ffn_down_exps = Some(req!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                ));
                l.ffn_up_exps = Some(req!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                ));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/gemma-embedding.cpp:30-67 ----
        LlmArch::GEMMA_EMBEDDING => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            // gemma-embedding.cpp:37-42 — never read by the encoder graph
            let output = dup_fallback!((lc.n_embd, lc.n_vocab));

            // the sentence-transformers dense modules (:45-46 — NOT_REQUIRED;
            // the graph of this revision never consumes them)
            let _ = opt!(
                LlmTensor::DENSE_2_OUT,
                "weight",
                -1,
                &[lc.n_embd, hparams.dense_2_feat_out as i64]
            );
            let _ = opt!(
                LlmTensor::DENSE_3_OUT,
                "weight",
                -1,
                &[hparams.dense_3_feat_in as i64, lc.n_embd]
            );

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_post_norm =
                    Some(req!(LlmTensor::ATTN_POST_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));
                l.ffn_gate = Some(req!(
                    LlmTensor::FFN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_up = Some(req!(
                    LlmTensor::FFN_UP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_ff]
                ));
                l.ffn_down = Some(req!(
                    LlmTensor::FFN_DOWN,
                    "weight",
                    bid,
                    &[lc.n_ff, lc.n_embd]
                ));
                l.ffn_post_norm = Some(req!(LlmTensor::FFN_POST_NORM, "weight", bid, &[lc.n_embd]));
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/hy-v3.cpp:22-97 (trunk + MTP tensors; graph_mtp is the
        // documented batch-15 skip) ----
        LlmArch::HY_V3 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let mut output = opt!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);
            if output.is_none() {
                output = Some(dup_fallback!((lc.n_embd, lc.n_vocab)));
            }

            let n_ff_exp = if hparams.n_ff_exp(0) > 0 {
                hparams.n_ff_exp(0) as i64
            } else {
                lc.n_ff
                    / if lc.n_expert_used > 0 {
                        lc.n_expert_used
                    } else {
                        1
                    }
            };
            let n_ff_shexp = if hparams.n_ff_shexp > 0 {
                hparams.n_ff_shexp as i64
            } else {
                n_ff_exp
            };

            let mut load_block = |l: &mut LayerTensors,
                                  bid: i32,
                                  is_nextn: bool|
             -> Result<(), String> {
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * lc.n_head,
                    lc.n_embd_k_gqa,
                    lc.n_embd_v_gqa,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k * lc.n_head, lc.n_embd]
                ));

                l.attn_q_norm = Some(req!(
                    LlmTensor::ATTN_Q_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));
                l.attn_k_norm = Some(req!(
                    LlmTensor::ATTN_K_NORM,
                    "weight",
                    bid,
                    &[lc.n_embd_head_k]
                ));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = opt!(LlmTensor::FFN_GATE, "weight", bid, &[lc.n_embd, lc.n_ff]);
                l.ffn_down = opt!(LlmTensor::FFN_DOWN, "weight", bid, &[lc.n_ff, lc.n_embd]);
                l.ffn_up = opt!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]);

                l.ffn_gate_inp = opt!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                );
                // hy-v3.cpp:68 — the suffix-LESS tn(LLM_TENSOR_FFN_EXP_PROBS_B, i)
                // form ("blk.N.exp_probs_b", no ".bias")
                l.ffn_exp_probs_b = ld.create_tensor(
                    LlmTensor::FFN_EXP_PROBS_B,
                    "",
                    bid,
                    &[lc.n_expert],
                    TENSOR_NOT_REQUIRED,
                )?;
                l.ffn_down_exps = opt!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                );
                // create_tensor_gate_up_exps — the fused tensor wins
                l.ffn_gate_up_exps = opt!(
                    LlmTensor::FFN_GATE_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, 2 * n_ff_exp, lc.n_expert]
                );
                if l.ffn_gate_up_exps.is_none() {
                    l.ffn_gate_exps = opt!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    );
                    l.ffn_up_exps = opt!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    );
                }

                l.ffn_gate_shexp = opt!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                );
                l.ffn_up_shexp = opt!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                );
                l.ffn_down_shexp = opt!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                );

                if is_nextn {
                    // the NextN/MTP head tensors (hy-v3.cpp:83-96)
                    l.nextn.eh_proj = Some(req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    ));
                    l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
                Ok(())
            };

            for (i, l) in layers.iter_mut().enumerate() {
                load_block(l, i as i32, i >= lc.n_layer)?;
            }
            (tok_embd, output_norm, None, output.unwrap(), None)
        }

        // ---- models/mimo2.cpp:25-82 (trunk + MTP tensors; graph_mtp is the
        // documented batch-15 skip) ----
        LlmArch::MIMO2 => {
            // def4d406a: upstream's mimo2.cpp:28-32 probes
            // "blk.0.attn_norm.weight" for mtp-only files (trunk tensors
            // TENSOR_NOT_REQUIRED) next to the old trunk-only probe — the
            // port has no partial-file modes (full files only, PARITY.md
            // batch 18 model.rs:29-32), both probes stay N/A here.
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let n_ff_exp = hparams.n_ff_exp(0) as i64;

            for i in 0..lc.n_layer_all {
                let l = &mut layers[i];
                let bid = i as i32;
                let n_head = hparams.n_head(i) as i64;

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * n_head,
                    hparams.n_embd_k_gqa(i) as i64,
                    hparams.n_embd_v_gqa(i) as i64,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[hparams.n_embd_head_v(i) as i64 * n_head, lc.n_embd]
                ));

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_sinks = opt!(LlmTensor::ATTN_SINKS, "weight", bid, &[n_head]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = opt!(LlmTensor::FFN_GATE, "weight", bid, &[lc.n_embd, lc.n_ff]);
                l.ffn_down = opt!(LlmTensor::FFN_DOWN, "weight", bid, &[lc.n_ff, lc.n_embd]);
                l.ffn_up = opt!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]);

                l.ffn_gate_inp = opt!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                );
                l.ffn_gate_exps = opt!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                );
                l.ffn_down_exps = opt!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                );
                l.ffn_up_exps = opt!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                );
                l.ffn_exp_probs_b = opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                if i >= lc.n_layer {
                    // the NextN block (mimo2.cpp:72-80)
                    l.nextn.eh_proj = Some(req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    ));
                    l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                    l.layer_out_norm = opt!(LlmTensor::LAYER_OUT_NORM, "weight", bid, &[lc.n_embd]);
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/step35.cpp:37-182 (trunk + MTP tensors; graph_mtp is
        // the documented batch-15 skip) ----
        LlmArch::STEP35 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            // the shared rope-factors tensor covers the WIDEST layer
            // (step35.cpp:60-67)
            let mut n_rot_max = 0u32;
            for i in 0..lc.n_layer {
                n_rot_max = n_rot_max.max(hparams.n_rot(i));
            }
            let n_rot_max = if n_rot_max == 0 {
                hparams.n_rot(0)
            } else {
                n_rot_max
            };

            let n_ff_exp = hparams.n_ff_exp(0) as i64;
            let n_ff_shexp = hparams.n_ff_shexp as i64;

            let mut load_block_trunk = |l: &mut LayerTensors,
                                        bid: i32,
                                        is_nextn: bool|
             -> Result<(), String> {
                let n_head_l = hparams.n_head(bid as usize) as i64;

                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_q_norm = opt!(LlmTensor::ATTN_Q_NORM, "weight", bid, &[lc.n_embd_head_k]);
                l.attn_k_norm = opt!(LlmTensor::ATTN_K_NORM, "weight", bid, &[lc.n_embd_head_k]);

                if hparams.rope_scaling_type_train == LlamaRopeScalingType::LONGROPE {
                    l.rope_long = opt!(
                        LlmTensor::ROPE_FACTORS_LONG,
                        "weight",
                        bid,
                        &[(n_rot_max / 2) as i64]
                    );
                    l.rope_short = opt!(
                        LlmTensor::ROPE_FACTORS_SHORT,
                        "weight",
                        bid,
                        &[(n_rot_max / 2) as i64]
                    );
                } else {
                    // NOT_REQUIRED | DUPLICATED beyond layer 0 (step35.cpp:85)
                    let dup = if bid != 0 { TENSOR_DUPLICATED } else { 0 };
                    l.rope_freqs = ld.create_tensor(
                        LlmTensor::ROPE_FREQS,
                        "weight",
                        bid,
                        &[(n_rot_max / 2) as i64],
                        TENSOR_NOT_REQUIRED | dup,
                    )?;
                }

                create_tensor_qkv(
                    l,
                    ld,
                    bid,
                    lc.n_embd,
                    lc.n_embd_head_k * n_head_l,
                    hparams.n_embd_k_gqa(bid as usize) as i64,
                    hparams.n_embd_v_gqa(bid as usize) as i64,
                    0,
                )?;
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[
                        hparams.n_embd_head_v(bid as usize) as i64 * n_head_l,
                        lc.n_embd
                    ]
                ));

                l.wqkv_gate = opt!(LlmTensor::ATTN_GATE, "weight", bid, &[lc.n_embd, n_head_l]);

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                l.ffn_gate = opt!(LlmTensor::FFN_GATE, "weight", bid, &[lc.n_embd, lc.n_ff]);
                l.ffn_down = opt!(LlmTensor::FFN_DOWN, "weight", bid, &[lc.n_ff, lc.n_embd]);
                l.ffn_up = opt!(LlmTensor::FFN_UP, "weight", bid, &[lc.n_embd, lc.n_ff]);

                l.ffn_gate_inp = opt!(
                    LlmTensor::FFN_GATE_INP,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_expert]
                );
                l.ffn_gate_exps = opt!(
                    LlmTensor::FFN_GATE_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                );
                l.ffn_down_exps = opt!(
                    LlmTensor::FFN_DOWN_EXPS,
                    "weight",
                    bid,
                    &[n_ff_exp, lc.n_embd, lc.n_expert]
                );
                l.ffn_up_exps = opt!(
                    LlmTensor::FFN_UP_EXPS,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_exp, lc.n_expert]
                );
                l.ffn_exp_probs_b = opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                l.ffn_gate_shexp = opt!(
                    LlmTensor::FFN_GATE_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                );
                l.ffn_up_shexp = opt!(
                    LlmTensor::FFN_UP_SHEXP,
                    "weight",
                    bid,
                    &[lc.n_embd, n_ff_shexp]
                );
                l.ffn_down_shexp = opt!(
                    LlmTensor::FFN_DOWN_SHEXP,
                    "weight",
                    bid,
                    &[n_ff_shexp, lc.n_embd]
                );

                if is_nextn {
                    // the NextN/MTP head tensors (step35.cpp:164-170)
                    l.nextn.eh_proj = Some(req!(
                        LlmTensor::NEXTN_EH_PROJ,
                        "weight",
                        bid,
                        &[2 * lc.n_embd, lc.n_embd]
                    ));
                    l.nextn.enorm = Some(req!(LlmTensor::NEXTN_ENORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.hnorm = Some(req!(LlmTensor::NEXTN_HNORM, "weight", bid, &[lc.n_embd]));
                    l.nextn.embed_tokens = opt!(
                        LlmTensor::NEXTN_EMBED_TOKENS,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_head = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_HEAD,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_vocab]
                    );
                    l.nextn.shared_head_norm = opt!(
                        LlmTensor::NEXTN_SHARED_HEAD_NORM,
                        "weight",
                        bid,
                        &[lc.n_embd]
                    );
                }
                Ok(())
            };

            for (i, l) in layers.iter_mut().enumerate() {
                load_block_trunk(l, i as i32, i >= lc.n_layer)?;
            }
            (tok_embd, output_norm, None, output, None)
        }

        // ---- models/hy-v4.cpp:70-155 ----
        LlmArch::HY_V4 => {
            let tok_embd = req!(
                LlmTensor::TOKEN_EMBD,
                "weight",
                -1,
                &[lc.n_embd, lc.n_vocab]
            );

            let output_norm = req!(LlmTensor::OUTPUT_NORM, "weight", -1, &[lc.n_embd]);
            let output = req!(LlmTensor::OUTPUT, "weight", -1, &[lc.n_embd, lc.n_vocab]);

            let n_embd_head_k_mla = hparams.n_embd_head_k_mla() as i64;
            let n_embd_head_v_mla = hparams.n_embd_head_v_mla() as i64;
            let n_embd_head_qk_rope = hparams.n_rot(0) as i64;
            let n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
            assert!(n_embd_head_qk_nope >= 1);

            let q_lora_rank = hparams.n_lora_q as i64;
            let kv_lora_rank = hparams.n_lora_kv as i64;
            let n_ff_exp = hparams.n_ff_exp(0) as i64;
            let n_expert_shared = hparams.n_expert_shared as i64;
            let hc = hparams.dsv4_hc_mult as i64;

            // the global iHC head (hy-v4.cpp:90-92)
            extra.hc_head_fn = Some(req!(
                LlmTensor::HC_HEAD_FN,
                "weight",
                -1,
                &[hc * lc.n_embd, hc]
            ));
            extra.hc_head_base = Some(req!(LlmTensor::HC_HEAD_BASE, "weight", -1, &[hc]));
            extra.hc_head_scale = Some(req!(LlmTensor::HC_HEAD_SCALE, "weight", -1, &[1]));

            for (i, l) in layers.iter_mut().enumerate() {
                let bid = i as i32;
                l.attn_norm = Some(req!(LlmTensor::ATTN_NORM, "weight", bid, &[lc.n_embd]));
                l.attn_sinks = Some(req!(LlmTensor::ATTN_SINKS, "weight", bid, &[lc.n_head]));

                l.wq_a = Some(req!(
                    LlmTensor::ATTN_Q_A,
                    "weight",
                    bid,
                    &[lc.n_embd, q_lora_rank]
                ));
                l.attn_q_a_norm = Some(req!(
                    LlmTensor::ATTN_Q_A_NORM,
                    "weight",
                    bid,
                    &[q_lora_rank]
                ));
                l.wq_b = Some(req!(
                    LlmTensor::ATTN_Q_B,
                    "weight",
                    bid,
                    &[q_lora_rank, lc.n_head * n_embd_head_k_mla]
                ));
                l.wkv_a_mqa = Some(req!(
                    LlmTensor::ATTN_KV_A_MQA,
                    "weight",
                    bid,
                    &[lc.n_embd, kv_lora_rank + n_embd_head_qk_rope]
                ));
                l.attn_kv_a_norm = Some(req!(
                    LlmTensor::ATTN_KV_A_NORM,
                    "weight",
                    bid,
                    &[kv_lora_rank]
                ));
                l.wk_b = Some(req!(
                    LlmTensor::ATTN_K_B,
                    "weight",
                    bid,
                    &[n_embd_head_qk_nope, kv_lora_rank, lc.n_head]
                ));
                l.wv_b = Some(req!(
                    LlmTensor::ATTN_V_B,
                    "weight",
                    bid,
                    &[kv_lora_rank, n_embd_head_v_mla, lc.n_head]
                ));
                l.wo = Some(req!(
                    LlmTensor::ATTN_OUT,
                    "weight",
                    bid,
                    &[lc.n_head * n_embd_head_v_mla, lc.n_embd]
                ));
                l.wqkv_gate = Some(req!(
                    LlmTensor::ATTN_GATE,
                    "weight",
                    bid,
                    &[lc.n_embd, lc.n_head * n_embd_head_v_mla]
                ));

                // only the "full" indexer layers ship the indexer five
                // (hy-v4.cpp:111-120)
                if hparams.indexer_top_k > 0 && hparams.is_indexer_full(i) {
                    let n_indexer_head = hparams.indexer_n_head as i64;
                    let n_embd_indexer = hparams.indexer_head_size as i64;
                    l.indexer_attn_q_b = Some(req!(
                        LlmTensor::INDEXER_ATTN_Q_B,
                        "weight",
                        bid,
                        &[q_lora_rank, n_indexer_head * n_embd_indexer]
                    ));
                    l.indexer_attn_k = Some(req!(
                        LlmTensor::INDEXER_ATTN_K,
                        "weight",
                        bid,
                        &[lc.n_embd, n_embd_indexer]
                    ));
                    l.indexer_k_norm = Some(req!(
                        LlmTensor::INDEXER_K_NORM,
                        "weight",
                        bid,
                        &[n_embd_indexer]
                    ));
                    l.indexer_k_norm_b = Some(req!(
                        LlmTensor::INDEXER_K_NORM,
                        "bias",
                        bid,
                        &[n_embd_indexer]
                    ));
                    l.indexer_proj = Some(req!(
                        LlmTensor::INDEXER_PROJ,
                        "weight",
                        bid,
                        &[lc.n_embd, n_indexer_head]
                    ));
                }

                l.hc_attn_fn = Some(req!(
                    LlmTensor::HC_ATTN_FN,
                    "weight",
                    bid,
                    &[hc * lc.n_embd, 2 * hc]
                ));
                l.hc_attn_base = Some(req!(LlmTensor::HC_ATTN_BASE, "weight", bid, &[2 * hc]));
                l.hc_attn_scale = Some(req!(LlmTensor::HC_ATTN_SCALE, "weight", bid, &[2]));
                l.hc_ffn_fn = Some(req!(
                    LlmTensor::HC_FFN_FN,
                    "weight",
                    bid,
                    &[hc * lc.n_embd, 2 * hc]
                ));
                l.hc_ffn_base = Some(req!(LlmTensor::HC_FFN_BASE, "weight", bid, &[2 * hc]));
                l.hc_ffn_scale = Some(req!(LlmTensor::HC_FFN_SCALE, "weight", bid, &[2]));

                l.ffn_norm = Some(req!(LlmTensor::FFN_NORM, "weight", bid, &[lc.n_embd]));

                if i < hparams.n_layer_dense_lead as usize {
                    l.ffn_gate = Some(req!(
                        LlmTensor::FFN_GATE,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                    l.ffn_down = Some(req!(
                        LlmTensor::FFN_DOWN,
                        "weight",
                        bid,
                        &[lc.n_ff, lc.n_embd]
                    ));
                    l.ffn_up = Some(req!(
                        LlmTensor::FFN_UP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_ff]
                    ));
                } else {
                    l.ffn_gate_inp = Some(req!(
                        LlmTensor::FFN_GATE_INP,
                        "weight",
                        bid,
                        &[lc.n_embd, lc.n_expert]
                    ));
                    l.ffn_exp_probs_b =
                        opt!(LlmTensor::FFN_EXP_PROBS_B, "bias", bid, &[lc.n_expert]);

                    if lc.n_expert == 0 {
                        return Err("n_expert must be > 0".to_string());
                    }
                    if lc.n_expert_used == 0 {
                        return Err("n_expert_used must be > 0".to_string());
                    }

                    l.ffn_gate_exps = Some(req!(
                        LlmTensor::FFN_GATE_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_up_exps = Some(req!(
                        LlmTensor::FFN_UP_EXPS,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp, lc.n_expert]
                    ));
                    l.ffn_down_exps = Some(req!(
                        LlmTensor::FFN_DOWN_EXPS,
                        "weight",
                        bid,
                        &[n_ff_exp, lc.n_embd, lc.n_expert]
                    ));

                    l.ffn_gate_shexp = Some(req!(
                        LlmTensor::FFN_GATE_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                    l.ffn_down_shexp = Some(req!(
                        LlmTensor::FFN_DOWN_SHEXP,
                        "weight",
                        bid,
                        &[n_ff_exp * n_expert_shared, lc.n_embd]
                    ));
                    l.ffn_up_shexp = Some(req!(
                        LlmTensor::FFN_UP_SHEXP,
                        "weight",
                        bid,
                        &[lc.n_embd, n_ff_exp * n_expert_shared]
                    ));
                }
            }
            (tok_embd, output_norm, None, output, None)
        }

        _ => {
            return Err(format!(
                "arch '{}' tensor loading not ported yet",
                arch.name()
            ));
        }
    };

    // generic optional `.scale` pass of `llama_model::load_tensors`
    // (llama-model.cpp) — see `create_optional_scale_tensors`
    create_optional_scale_tensors(ld, &lc, &mut layers)?;

    Ok(ArchTensors {
        tok_embd,
        output_norm,
        output_b,
        output,
        cls_out,
        extra,
        layers,
    })
}

/// `llama_model::load_tensors` generic pass: optional per-tensor / per-expert
/// `*.scale` tensors, created only when the corresponding weight exists.
///
/// Deviation (documented): only the scale kinds carried by the ported archs
/// are modeled (`ffn_gate_inp`, `ffn_gate_exps`, `ffn_down_exps`,
/// `ffn_up_exps`); the dense-attn/ffn/ssm/nextn scales of the full C++ pass
/// are not created (no local file of a ported arch has them).
fn create_optional_scale_tensors(
    ld: &mut ModelLoader,
    lc: &LoadLocals,
    layers: &mut [LayerTensors],
) -> Result<(), String> {
    for (i, l) in layers.iter_mut().enumerate() {
        let bid = i as i32;
        // router scale (gemma4 loads this one explicitly in its arch function;
        // creating it here first is idempotent — the same name dedups)
        if l.ffn_gate_inp_s.is_none() && l.ffn_gate_inp.is_some() {
            l.ffn_gate_inp_s = ld.create_tensor(
                LlmTensor::FFN_GATE_INP,
                "scale",
                bid,
                &[lc.n_embd],
                TENSOR_NOT_REQUIRED,
            )?;
        }
        // MoE expert weight scales (per-expert, shape {n_expert})
        if l.ffn_gate_exps_s.is_none() && l.ffn_gate_exps.is_some() {
            l.ffn_gate_exps_s = ld.create_tensor(
                LlmTensor::FFN_GATE_EXPS,
                "scale",
                bid,
                &[lc.n_expert],
                TENSOR_NOT_REQUIRED,
            )?;
        }
        if l.ffn_down_exps_s.is_none() && l.ffn_down_exps.is_some() {
            l.ffn_down_exps_s = ld.create_tensor(
                LlmTensor::FFN_DOWN_EXPS,
                "scale",
                bid,
                &[lc.n_expert],
                TENSOR_NOT_REQUIRED,
            )?;
        }
        if l.ffn_up_exps_s.is_none() && l.ffn_up_exps.is_some() {
            l.ffn_up_exps_s = ld.create_tensor(
                LlmTensor::FFN_UP_EXPS,
                "scale",
                bid,
                &[lc.n_expert],
                TENSOR_NOT_REQUIRED,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// load_model — llm_load_tensors equivalent
// ---------------------------------------------------------------------------

/// Load model weights from a GGUF into a fresh [`LlamaModel`].
///
/// `mmap` must be the memory map `gguf` was parsed from (or a map of the same
/// bytes) — tensor storage is a zero-copy view into it, exactly like the
/// `use_mmap` path of `llama_model_loader::load_all_data`.
///
/// Validation (mirroring the C++ loader):
///   * every required tensor exists with the exact expected shape
///     (`check_tensor_dims` semantics, incl. the `wrong shape` error text)
///   * `output.weight` missing → aliased to `token_embd` (TENSOR_DUPLICATED)
///   * every declared gguf tensor is consumed (`done_getting_tensors`)
pub fn load_model(gguf: &Gguf, mmap: Arc<Mmap>) -> Result<LlamaModel, String> {
    let (arch, mut hparams) = load_hparams(gguf)?;
    load_arch_hparams_t5(gguf, arch, &mut hparams)?;
    load_arch_hparams_batch(gguf, arch, &mut hparams)?;

    if arch_tensors_support(arch) == ArchTensorsSupport::Unsupported {
        return Err(format!(
            "arch '{}' tensor loading not ported yet",
            arch.name()
        ));
    }

    let n_vocab = n_vocab_from_gguf(gguf, arch)?;

    let mut ld = ModelLoader::new(gguf, mmap);
    let t = load_arch_tensors(arch, &hparams, n_vocab, &mut ld)?;

    ld.done_getting_tensors()?;

    let tensors = std::mem::take(&mut ld.by_name);
    let mut ctx = std::mem::replace(&mut ld.ctx, Context::new());

    // CPU_REPACK warm-up (ggml::compute::warm_repack): the reference converts
    // every qualifying weight into its repack buffer while the model loads
    // (multithreaded, before any timed forward); without this the port's
    // lazily materialized cache pays the same work inside the first forward's
    // measured time.
    let _repacked_bytes = ggml::compute::warm_repack(&mut ctx);
    // prefault the mmap'd weights (see `prefault_weights`): the reference's
    // load leaves resident pages behind, while a pure-mmap load would pay the
    // minor-fault storm inside the first timed forward
    let _prefaulted_bytes = ggml::compute::prefault_weights(&ctx);

    // load_stats (llama-model.cpp:1226-1229) — the loader's totals over the
    // file's tensor table (llama_model_loader's n_elements/n_bytes); the port
    // recomputes them from the same table (every file tensor, file type).
    let mut n_elements: u64 = 0;
    let mut n_bytes: u64 = 0;
    for t in &gguf.tensors {
        n_elements += t.ne.iter().product::<i64>() as u64;
        n_bytes += t.size_bytes();
    }
    // `ml.ftype` (llama-model-loader.cpp:754-790): the `general.file_type`
    // KV when present, else guessed from the majority tensor type with the
    // LLAMA_FTYPE_GUESSED bit set (display::guess_ftype — the same mapping
    // the verified llama-bench model_info uses).
    let ftype = crate::display::guess_ftype(gguf);
    let classifier_labels = crate::meta::read_classifier_labels(gguf, arch).unwrap_or_default();
    // per-tensor activation precision policy (llama-model.cpp:1809-1811,
    // e9f824d8c) — a malformed pair is a load error, same throw as the C
    let prec_policy = PrecPolicy::load(gguf, arch, &tensors)?;

    Ok(LlamaModel {
        arch,
        hparams,
        name: gguf.get_str("general.name").unwrap_or("").to_string(),
        prec_policy,
        classifier_labels,
        ftype,
        n_elements,
        n_bytes,
        ctx,
        tensors,
        tok_embd: t.tok_embd,
        output_norm: t.output_norm,
        output_norm_b: t.extra.output_norm_b,
        output_b: t.output_b,
        output: t.output,
        cls_out: t.cls_out,
        per_layer_tok_embd: t.extra.per_layer_tok_embd,
        per_layer_model_proj: t.extra.per_layer_model_proj,
        per_layer_proj_norm: t.extra.per_layer_proj_norm,
        dense_2_out_layers: t.extra.dense_2_out_layers,
        dense_2_out_layers_b: t.extra.dense_2_out_layers_b,
        token_types: t.extra.token_types,
        position_embd: t.extra.position_embd,
        token_embd_norm: t.extra.token_embd_norm,
        token_embd_norm_b: t.extra.token_embd_norm_b,
        cls: t.extra.cls,
        cls_b: t.extra.cls_b,
        cls_out_b: t.extra.cls_out_b,
        cls_norm: t.extra.cls_norm,
        cls_norm_b: t.extra.cls_norm_b,
        clef_head: t.extra.clef_head,
        hc_head_fn: t.extra.hc_head_fn,
        hc_head_base: t.extra.hc_head_base,
        hc_head_scale: t.extra.hc_head_scale,
        graniteswitch_token_to_slot: t.extra.graniteswitch_token_to_slot,
        graniteswitch_token_to_substitute: t.extra.graniteswitch_token_to_substitute,
        output_res_score: t.extra.output_res_score,
        hc_head_norm: t.extra.hc_head_norm,
        hc_head_down: t.extra.hc_head_down,
        hc_head_up: t.extra.hc_head_up,
        enc_output_norm: t.extra.enc_output_norm,
        hrm_z_l_init: t.extra.hrm_z_l_init,
        altup_proj: t.extra.altup_proj,
        altup_unembd_proj: t.extra.altup_unembd_proj,
        conv1d: t.extra.conv1d,
        conv1d_b: t.extra.conv1d_b,
        tok_norm: t.extra.tok_norm,
        tok_norm_b: t.extra.tok_norm_b,
        layers: t.layers,
    })
}

/// `models/t5.cpp:3-13` / `models/t5encoder.cpp:3-7` `load_arch_hparams`.
///
/// The arch-specific hparams arms live in meta.rs (`load_arch_hparams`), which
/// is outside this task's file ownership; the reads are identical and happen
/// before any tensor is created, so the effect matches the C. Two keys are
/// *required* there (`get_key` with no default): `attention.layer_norm_rms_epsilon`
/// and `attention.relative_buckets_count` — both present in the local
/// `t5-v1_1-xxl-encoder` file (1e-6 / 32).
fn load_arch_hparams_t5(gguf: &Gguf, arch: LlmArch, h: &mut LlamaHparams) -> Result<(), String> {
    if arch != LlmArch::T5 && arch != LlmArch::T5ENCODER {
        return Ok(());
    }
    let k = |kv: LlmKv| kv_name(arch, kv);

    let eps = gguf.get_f32(&k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS));
    h.f_norm_rms_eps =
        eps.ok_or_else(|| format!("key {} not found", k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS)))?;
    let nbkts = gguf.get_u32(&k(LlmKv::ATTENTION_RELATIVE_BUCKETS_COUNT));
    h.n_rel_attn_bkts = nbkts.ok_or_else(|| {
        format!(
            "key {} not found",
            k(LlmKv::ATTENTION_RELATIVE_BUCKETS_COUNT)
        )
    })?;

    if arch == LlmArch::T5 {
        // t5.cpp:7-13 — decoder bookkeeping (the T5 decoder is not ported; the
        // fields are read so hparams match the reference for the encoder too).
        // `dec_start_token_id` is a llama_token (i32); the C stores the u32 read
        // from the file as-is (llama-model.h `uint32_t` → hparams int32).
        if let Some(v) = gguf.get_u32(&k(LlmKv::DECODER_START_TOKEN_ID)) {
            h.dec_start_token_id = v as i32;
        }
        h.dec_n_layer = h.n_layer();
        if let Some(v) = gguf.get_u32(&k(LlmKv::DECODER_BLOCK_COUNT)) {
            h.dec_n_layer = v;
        }
    }
    Ok(())
}

/// `get_key_or_arr<uint32_t>` (llama-model-loader.cpp) — the array-or-scalar
/// read the batch-4 MoE archs use for `expert_feed_forward_length`. meta.rs's
/// private helper is replicated here 1:1 (that file is outside this batch's
/// ownership).
/// The element count of an array KV (granite-switch's adapter token-id arrays,
/// `get_arr(LLM_KV_ADAPTER_TOKEN_IDS_*, std::vector<llama_token>&)` — the
/// length check of granite-switch.cpp:42-46). `None` when the key is absent or
/// not an array.
fn arr_len(gguf: &Gguf, key: &str) -> Option<u32> {
    match gguf.find_key(key) {
        Some(ggml::Value::Array(_, items)) => Some(items.len() as u32),
        _ => None,
    }
}

/// `ml.get_arr` for an I32 array (granite-swa's deepstack_mapping).
fn get_arr_i32_local(gguf: &Gguf, key: &str) -> Result<Option<Vec<i32>>, String> {
    match gguf.find_key(key) {
        None => Ok(None),
        Some(Value::Array(_, items)) => {
            let vals = items
                .iter()
                .map(|v| match v {
                    Value::I32(x) => Ok(*x),
                    Value::U32(x) => Ok(*x as i32),
                    _ => Err(format!("key {key} has wrong array element type")),
                })
                .collect::<Result<Vec<i32>, String>>()?;
            Ok(Some(vals))
        }
        Some(_) => Err(format!("key {key} has wrong type")),
    }
}

/// `ml.get_arr` for a U32 array (granite-swa's attention.rope_pattern).
fn get_arr_u32_local(gguf: &Gguf, key: &str) -> Result<Option<Vec<u32>>, String> {
    match gguf.find_key(key) {
        None => Ok(None),
        Some(Value::Array(_, items)) => {
            let vals = items
                .iter()
                .map(|v| match v {
                    Value::U32(x) => Ok(*x),
                    Value::I32(x) => u32::try_from(*x).map_err(|_| key.to_string()),
                    Value::Bool(b) => Ok(u32::from(*b)),
                    _ => Err(format!("key {key} has wrong array element type")),
                })
                .collect::<Result<Vec<u32>, String>>()?;
            Ok(Some(vals))
        }
        Some(_) => Err(format!("key {key} has wrong type")),
    }
}

fn get_key_or_arr_u32_local(gguf: &Gguf, key: &str, n: usize) -> Result<Option<Vec<u32>>, String> {
    match gguf.find_key(key) {
        None => Ok(None),
        Some(Value::Array(_, items)) => {
            if items.len() != n {
                return Err(format!(
                    "key {key} has wrong array length; expected {n}, got {}",
                    items.len()
                ));
            }
            let vals = items
                .iter()
                .map(|v| match v {
                    Value::Bool(b) => Ok(u32::from(*b)),
                    Value::U32(x) => Ok(*x),
                    Value::I32(x) => u32::try_from(*x).map_err(|_| key.to_string()),
                    _ => Err(key.to_string()),
                })
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|key| {
                    format!("{key} is not a string/float32/uint32/int32/uint64 array")
                })?;
            Ok(Some(vals))
        }
        Some(Value::U32(x)) => Ok(Some(vec![*x; n])),
        Some(v) => Err(format!(
            "key {key} is not a uint32 scalar or array; got {:?}",
            v.type_()
        )),
    }
}

/// The float32 flavour of [`get_key_or_arr_u32_local`] — the
/// `get_key_or_arr(key, std::array<float, N>, n)` overload maple's
/// `swiglu_clamp_exp` read uses (maple.cpp:16, llama-model-loader.h:212).
fn get_key_or_arr_f32_local(gguf: &Gguf, key: &str, n: usize) -> Result<Option<Vec<f32>>, String> {
    match gguf.find_key(key) {
        None => Ok(None),
        Some(Value::Array(_, items)) => {
            if items.len() != n {
                return Err(format!(
                    "key {key} has wrong array length; expected {n}, got {}",
                    items.len()
                ));
            }
            let vals = items
                .iter()
                .map(|v| match v {
                    Value::F32(x) => Ok(*x),
                    Value::F64(x) => Ok(*x as f32),
                    _ => Err(()),
                })
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|_| format!("key {key} is not a float32 array"))?;
            Ok(Some(vals))
        }
        Some(Value::F32(x)) => Ok(Some(vec![*x; n])),
        Some(v) => Err(format!(
            "key {key} is not a float32 scalar or array; got {:?}",
            v.type_()
        )),
    }
}

/// `load_arch_hparams` of the arch batch landed in this file:
/// models/{gpt2,phi2,starcoder2,command-r,olmo2,gptneox}.cpp (llama.cpp
/// bd4f514db1). Same placement rationale as [`load_arch_hparams_t5`]: the
/// generic reads happen in meta.rs's `load_hparams`, and meta.rs is owned by
/// another task, so the arch-specific tail runs here — before any tensor is
/// created, exactly like the C `load_hparams` order. The LLM_TYPE_* switch
/// each of those functions ends with only fills `model.type`, which is
/// description-string cosmetics in C and not modeled by the port (meta.rs:17).
fn load_arch_hparams_batch(gguf: &Gguf, arch: LlmArch, h: &mut LlamaHparams) -> Result<(), String> {
    let k = |kv: LlmKv| kv_name(arch, kv);
    // `llama_model_loader::get_key` with `required = true` — the C signature
    // default. The message mirrors the port's other required reads.
    let req_f32 = |key: String| -> Result<f32, String> {
        gguf.get_f32(&key)
            .ok_or_else(|| format!("key {key} not found"))
    };
    let req_u32 = |key: String| -> Result<u32, String> {
        gguf.get_u32(&key)
            .ok_or_else(|| format!("key {key} not found"))
    };

    match arch {
        // ---- models/gpt2.cpp:3-13 ----
        LlmArch::GPT2 => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/phi2.cpp:3-11 ----
        LlmArch::PHI2 => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/starcoder2.cpp:3-14 ----
        LlmArch::STARCODER2 => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/command-r.cpp:3-11 ----
        LlmArch::COMMAND_R => {
            if let Some(v) = gguf.get_f32(&k(LlmKv::LOGIT_SCALE)) {
                h.f_logit_scale = v;
            }
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/gptneox.cpp:3-52 ----
        LlmArch::GPTNEOX => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            // `use_parallel_residual` is a required key in C and a bool read
            // (`gguf_get_val_bool`); the reference GGUF of every gpt-neox model
            // carries it.
            h.use_par_res = gguf
                .get_bool(&k(LlmKv::USE_PARALLEL_RESIDUAL))
                .ok_or_else(|| format!("key {} not found", k(LlmKv::USE_PARALLEL_RESIDUAL)))?;
        }

        // ---- models/olmo2.cpp:3-24 ----
        LlmArch::OLMO2 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            let found_swa = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW));
            match found_swa {
                Some(v) if v > 0 => {
                    h.n_swa = v;
                    h.swa_type = LlamaSwaType::STANDARD;
                    // `load_swa_pattern(ml, 4)` (olmo2.cpp:8 → llama-model.cpp:
                    // 3308-3315); meta.rs's helper is private, so the two reads
                    // live in `inline_load_swa_pattern` below.
                    inline_load_swa_pattern(gguf, arch, h)?;

                    h.rope_freq_base_train_swa = h.rope_freq_base_train;
                    h.rope_freq_scale_train_swa = 1.0; // olmo2.cpp:11
                    if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                        h.rope_freq_base_train_swa = v;
                    }
                }
                // olmo2.cpp:16 — no `attention.sliding_window` (or 0) is a plain
                // non-SWA model; the fresh struct already has NONE, set it
                // explicitly like the C else-branch.
                _ => h.swa_type = LlamaSwaType::NONE,
            }
        }

        // ---- arch batch 2 (2026-09-25) ----

        // ---- models/codeshell.cpp:3-11 ----
        LlmArch::CODESHELL => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/orion.cpp:3-11 ----
        LlmArch::ORION => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/olmo.cpp:3-13 ----
        LlmArch::OLMO => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            // `attention.clamp_kqv` is optional (`required = false`) and feeds
            // build_qkv's clamp branch
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_CLAMP_KQV)) {
                h.f_clamp_kqv = v;
            }
        }

        // ---- models/xverse.cpp:3-12 ----
        LlmArch::XVERSE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/internlm2.cpp:3-12 ----
        LlmArch::INTERNLM2 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/exaone.cpp:3-10 ----
        LlmArch::EXAONE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/falcon.cpp:3-11 ----
        LlmArch::FALCON => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // gemma v1 (models/gemma.cpp:3-11) is already an
        // `ArchHparamsSupport::Full` arm in meta.rs (its single RMS-eps read
        // landed there with gemma2/3), so nothing is duplicated here.

        // ---- arch batch 3 (2026-09-27) ----

        // ---- models/baichuan.cpp:3-15 ----
        LlmArch::BAICHUAN => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // the type switch (:6-11): 32 layers = 7B, 40 = 13B; only the 13B
            // gets ALiBi (:12-14, "TODO: become GGUF KV parameter")
            if h.n_layer() == 40 {
                h.f_max_alibi_bias = 8.0;
            }
        }

        // ---- models/bloom.cpp:3-19 ----
        LlmArch::BLOOM => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            // :18 — unconditional ("TODO: become GGUF KV parameter")
            h.f_max_alibi_bias = 8.0;
        }

        // ---- models/mpt.cpp:3-15 ----
        LlmArch::MPT => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_CLAMP_KQV)) {
                h.f_clamp_kqv = v;
            }
            // the only arch whose ALiBi bias is a GGUF key (:6)
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_MAX_ALIBI_BIAS)) {
                h.f_max_alibi_bias = v;
            }
        }

        // ---- models/starcoder.cpp:3-14 ----
        // NOTE: no ALiBi in this pinned revision — starcoder's
        // load_arch_hparams reads only the LayerNorm eps and never assigns
        // f_max_alibi_bias (checked against the pre-refactor llama-model.cpp
        // too), so `max_bias` stays 0.0 for this arch.
        LlmArch::STARCODER => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/refact.cpp:3-14 ----
        LlmArch::REFACT => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // :12 — unconditional
            h.f_max_alibi_bias = 8.0;
        }

        // ---- models/plamo.cpp:3-10 ----
        LlmArch::PLAMO => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/stablelm.cpp:3-13 ----
        LlmArch::STABLELM => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/granite.cpp:3-77 (dense) ----
        LlmArch::GRANITE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::RESIDUAL_SCALE)) {
                h.f_residual_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_SCALE)) {
                h.f_attention_scale = v;
            }
            // deepstack_mapping_arr (granite vision) is not ported — a dense
            // text-only file carries no `granite4.deepstack_mapping`, so the
            // graph's deepstack injection stays dead (n_deepstack_layers 0)
            // rope_finetuned defaults *true* for granite and fills the rope
            // pattern (granite.cpp:37-39)
            let mut rope_finetuned = true;
            if let Some(v) = gguf.get_bool(&k(LlmKv::ROPE_SCALING_FINETUNED)) {
                rope_finetuned = v;
            }
            h.rope_finetuned = rope_finetuned;
            h.rope_pattern
                .iter_mut()
                .for_each(|p| *p = rope_finetuned as u32);
            // granite-moe shared expert (:75-77)
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
        }

        // ---- models/minicpm.cpp:3-27 ----
        LlmArch::MINICPM => {
            // backward-compatible defaults (:5-7) overridable by newer GGUFs
            // (:11-14)
            h.f_embedding_scale = 12.0;
            h.f_residual_scale = 1.4 / (h.n_layer() as f32).sqrt();
            h.f_logit_scale = if h.n_embd != 0 {
                256.0 / h.n_embd as f32
            } else {
                1.0
            };
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::RESIDUAL_SCALE)) {
                h.f_residual_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::LOGIT_SCALE)) {
                h.f_logit_scale = v;
            }
            // minicpm ropes by default, unlike granite (:17)
            h.rope_finetuned = true;
            h.rope_pattern.iter_mut().for_each(|p| *p = 1);
        }

        // ---- arch batch 4 (2026-09-28): the MoE family ----

        // ---- models/qwen2moe.cpp:3-14 ----
        LlmArch::QWEN2MOE => {
            // get_key_or_arr(EXPERT_FEED_FORWARD_LENGTH, n_layer_all, false)
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/qwen3moe.cpp:3-12 ----
        LlmArch::QWEN3MOE => {
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/granite-moe.cpp:3-19 ----
        LlmArch::GRANITE_MOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::RESIDUAL_SCALE)) {
                h.f_residual_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_SCALE)) {
                h.f_attention_scale = v;
            }
            // granite-moe does NOT read rope_finetuned (granite.cpp:33-37 is
            // granite-dense only) — the generic default (rope_pattern all 1,
            // rope runs everywhere) stands, and no deepstack keys are read
            // either
            // For Granite MoE Shared (granite-moe.cpp:18)
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
        }

        // ---- models/phimoe.cpp:3-10 ----
        LlmArch::PHIMOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/arctic.cpp:3-14 ----
        LlmArch::ARCTIC => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/olmoe.cpp:3-10 ----
        LlmArch::OLMOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ernie4-5-moe's hparams (ernie4-5.cpp:3-21) are already an
        // `ArchHparamsSupport::Full` arm in meta.rs (the ERNIE4_5_MOE branch
        // reading n_ff_exp_arr / n_ff_shexp / n_moe_layer_step /
        // n_layer_dense_lead) — nothing duplicated here.

        // ---- the dense free-riders of the batch ----

        // ---- models/smollm3.cpp:3-11 ----
        LlmArch::SMOLLM3 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // smollm3.cpp:5 — nope layers: rope skips every 4th layer
            h.n_no_rope_layer_step = 4;
        }

        // ---- models/seed-oss.cpp:3-10 ----
        LlmArch::SEED_OSS => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/openelm.cpp:3-13 ----
        LlmArch::OPENELM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- arch batch 5 (2026-09-24): the mamba family ----

        // ---- models/mamba.cpp:3-33 ----
        // (is_recr was already set to all-1 by the generic path —
        // `llm_arch_is_recurrent(MAMBA)` = true, llama-model.cpp:1316)
        LlmArch::MAMBA => {
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            if let Some(v) = gguf.get_bool(&k(LlmKv::SSM_DT_B_C_RMS)) {
                h.ssm_dt_b_c_rms = v;
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/mamba2.cpp:3-33 ----
        // (is_recr all-1 via `llm_arch_is_recurrent(MAMBA2)` = true)
        LlmArch::MAMBA2 => {
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            h.ssm_n_group = req_u32(k(LlmKv::SSM_GROUP_COUNT))?;
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/jamba.cpp:3-30 ----
        LlmArch::JAMBA => {
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // jamba.cpp:12-14 — a layer is recurrent iff n_head_kv == 0
            for il in 0..h.n_layer() as usize {
                h.is_recr_impl[il] = u32::from(h.n_head_kv(il) == 0);
            }
        }

        // ---- models/nemotron-h.cpp:5-47 (NEMOTRON_H_MOE inherits this
        // hparams arm — models.h:1539-1543) ----
        LlmArch::NEMOTRON_H | LlmArch::NEMOTRON_H_MOE => {
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            h.ssm_n_group = req_u32(k(LlmKv::SSM_GROUP_COUNT))?;

            // A layer is recurrent IFF n_head_kv == 0 AND n_ff == 0; appended
            // MTP blocks are dense (nemotron-h.cpp:12-16). The port always
            // loads MTP tensors (no TENSOR_SKIP), so n_layer_nextn stays 0 in
            // every synthetic file and the loop covers n_layer_all anyway.
            for il in 0..h.n_layer_all as usize {
                h.is_recr_impl[il] =
                    u32::from((il as u32) < h.n_layer() && h.n_head_kv(il) == 0 && h.n_ff(il) == 0);
            }

            // nemotron-h.cpp:18-21 — LayerNorm eps required (MTP head), RMS
            // eps falls back to it when the file carries no RMS key
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            match gguf.get_f32(&k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS)) {
                Some(v) => h.f_norm_rms_eps = v,
                None => h.f_norm_rms_eps = h.f_norm_eps,
            }

            // Puzzle models set a different expert FFN size per layer
            // (nemotron-h.cpp:24-29)
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_COUNT)) {
                h.n_expert_shared = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::MOE_LATENT_SIZE)) {
                h.moe_latent_size = v;
            }
        }

        // ---- arch batch 6b (2026-09-24): nemotron / grok / chameleon / deci
        // / jais / falcon-h1 / plamo2 ----

        // ---- models/nemotron.cpp:3-10 ----
        LlmArch::NEMOTRON => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/grok.cpp:3-33 ----
        LlmArch::GROK => {
            // defaults for old GGUFs (grok.cpp:4-12)
            h.yarn_beta_fast = 8.0;
            h.f_logit_scale = 0.5773502691896257;
            h.f_embedding_scale = 78.38367176906169;
            h.f_attn_out_scale = 0.08838834764831845;
            h.f_attn_logit_softcapping = 30.0;
            h.f_router_logit_softcapping = 30.0;
            // no final_logit_softcapping in grok-1 (grok.cpp:12)
            h.f_final_logit_softcapping = 0.0;

            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // get_key_or_arr(EXPERT_FEED_FORWARD_LENGTH, n_layer_all, false)
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::LOGIT_SCALE)) {
                h.f_logit_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_OUTPUT_SCALE)) {
                h.f_attn_out_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTN_LOGIT_SOFTCAPPING)) {
                h.f_attn_logit_softcapping = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROUTER_LOGIT_SOFTCAPPING)) {
                h.f_router_logit_softcapping = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::FINAL_LOGIT_SOFTCAPPING)) {
                h.f_final_logit_softcapping = v;
            }

            // (grok.cpp:23-27) — read but unused by the grok graph
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_TEMPERATURE_LENGTH)) {
                h.attn_temp_length = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_EXT_FACTOR)) {
                h.yarn_ext_factor = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_ATTN_FACTOR)) {
                h.yarn_attn_factor = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_BETA_FAST)) {
                h.yarn_beta_fast = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_BETA_SLOW)) {
                h.yarn_beta_slow = v;
            }
        }

        // ---- models/chameleon.cpp:3-14 ----
        LlmArch::CHAMELEON => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_norm_eps = 1e-5; // eps for qk-norm, torch default (chameleon.cpp:6)
            if let Some(v) = gguf.get_bool(&k(LlmKv::SWIN_NORM)) {
                h.swin_norm = v;
            }
        }

        // ---- models/deci.cpp:3-12 ----
        LlmArch::DECI => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/jais.cpp:3-13 ----
        LlmArch::JAIS => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            // the ALiBi max bias is an (optional) GGUF key for jais
            // (jais.cpp:5) — f_max_alibi_bias > 0 flips use_alibi in
            // load_hparams (llama-model.cpp:1419-1421)
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_MAX_ALIBI_BIAS)) {
                h.f_max_alibi_bias = v;
            }
        }

        // ---- models/falcon-h1.cpp:3-32 ----
        LlmArch::FALCON_H1 => {
            // Common parameters (falcon-h1.cpp:5)
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // SSM parameters (falcon-h1.cpp:7-12)
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            h.ssm_n_group = req_u32(k(LlmKv::SSM_GROUP_COUNT))?;

            // every layer is recurrent AND attentive (falcon-h1.cpp:14)
            h.is_recr_impl.iter_mut().for_each(|v| *v = 1);
        }

        // ---- models/plamo2.cpp:4-33 ----
        LlmArch::PLAMO2 => {
            // Common parameters (plamo2.cpp:5)
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // Load Mamba SSM parameters (plamo2.cpp:7-12)
            h.ssm_d_conv = req_u32(k(LlmKv::SSM_CONV_KERNEL))?;
            h.ssm_d_inner = req_u32(k(LlmKv::SSM_INNER_SIZE))?;
            h.ssm_d_state = req_u32(k(LlmKv::SSM_STATE_SIZE))?;
            h.ssm_dt_rank = req_u32(k(LlmKv::SSM_TIME_STEP_RANK))?;
            h.ssm_n_group = req_u32(k(LlmKv::SSM_GROUP_COUNT))?;

            // attention.key_length / attention.value_length ride the generic
            // reads into n_embd_head_k_full / n_embd_head_v_full (the C's own
            // reads at plamo2.cpp:15-16 land there too)

            // a layer is recurrent iff n_head_kv == 0 (plamo2.cpp:18-20)
            for il in 0..h.n_layer_all as usize {
                h.is_recr_impl[il] = u32::from(h.n_head_kv(il) == 0);
            }
        }

        // ==============================================================
        // arch batch 8 (2026-09-30): the MoE long-tail family
        // ==============================================================

        // ---- models/hunyuan-moe.cpp:3-12 ----
        LlmArch::HUNYUAN_MOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // get_key_or_arr(..., required = true)
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
        }

        // ---- models/dots1.cpp:3-16 ----
        LlmArch::DOTS1 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
        }

        // ---- models/bailingmoe.cpp:3-16 ----
        LlmArch::BAILINGMOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
        }

        // ---- models/bailingmoe2.cpp:3-18 ----
        LlmArch::BAILINGMOE2 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            // REQUIRED (bailingmoe2.cpp:11)
            h.expert_gating_func = req_u32(k(LlmKv::EXPERT_GATING_FUNC))?;
        }

        // ---- models/glm4-moe.cpp:3-26 ----
        LlmArch::GLM4_MOE => {
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // optional mrope sections (glm4-moe.cpp:6) — only the vision files
            // carry them; the graph's rope_multi branch is not ported
            if let Some(ggml::Value::Array(_, items)) =
                gguf.find_key(&k(LlmKv::ROPE_DIMENSION_SECTIONS))
            {
                if items.len() == 4 {
                    for (sec, it) in h.rope_sections.iter_mut().zip(items.iter()) {
                        *sec = match it {
                            ggml::Value::I32(x) => *x,
                            ggml::Value::U32(x) => *x as i32,
                            _ => 0,
                        };
                    }
                }
            }

            // MoE parameters (glm4-moe.cpp:9-12)
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }

            // Expert gating function (glm4-moe.cpp:14-18 — GLM-4.5 uses sigmoid)
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }
        }

        // ---- models/minimax-m2.cpp:3-12 ----
        LlmArch::MINIMAX_M2 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
        }

        // ---- models/cohere2moe.cpp:3-38 ----
        LlmArch::COHERE2MOE => {
            // cohere2moe.cpp:4-11 — either epsilon, RMS wins when both exist
            let found_norm = gguf.find_key(&k(LlmKv::ATTENTION_LAYERNORM_EPS)).is_some();
            if found_norm {
                h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            }
            let found_norm_rms = gguf
                .find_key(&k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))
                .is_some();
            if found_norm_rms {
                h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            }
            if !found_norm && !found_norm_rms {
                return Err("missing Cohere2 MoE norm epsilon".to_string());
            }
            if !found_norm_rms {
                h.f_norm_rms_eps = 0.0;
            }

            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            h.n_layer_dense_lead = req_u32(k(LlmKv::LEADING_DENSE_BLOCK_COUNT))?;
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_COUNT)) {
                h.n_expert_shared = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }

            // cohere2moe.cpp:27-32 — the standard SWA pattern, dense-first
            h.swa_type = LlamaSwaType::STANDARD;
            b8_load_swa_pattern(gguf, arch, h, 4, true)?;

            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }
        }

        // ---- models/exaone-moe.cpp:3-26 ----
        LlmArch::EXAONE_MOE => {
            // exaone-moe.cpp:4-8 — hard defaults, then the pattern (NOT
            // dense-first), then the required window key overrides n_swa
            h.swa_type = LlamaSwaType::STANDARD;
            h.n_swa = 128;
            b8_load_swa_pattern(gguf, arch, h, 4, false)?;
            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_COUNT)) {
                h.n_expert_shared = v;
            }
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            // REQUIRED (exaone-moe.cpp:16)
            h.expert_gating_func = req_u32(k(LlmKv::EXPERT_GATING_FUNC))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
        }

        // ---- models/smallthinker.cpp:3-28 (arch batch 10) ----
        LlmArch::SMALLTHINKER => {
            // smallthinker.cpp:4-17 — an optional SWA key flips the file into
            // the iswa graph template (window re-pinned to 4096, pattern 4
            // dense-first, the *_swa rope pair); without it every layer ropes
            // and n_no_rope_layer_step collapses to n_layer (no layer skips)
            let found_swa = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW));
            if matches!(found_swa, Some(v) if v > 0) {
                h.swa_type = LlamaSwaType::STANDARD;
                h.n_swa = 4096;
                b8_load_swa_pattern(gguf, arch, h, 4, true)?;

                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
            } else {
                h.swa_type = LlamaSwaType::NONE;
                h.n_no_rope_layer_step = h.n_layer();
            }

            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?;
            if let Some(vals) = vals {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
        }

        // ---- models/llada-moe.cpp:3-14 (arch batch 10) ----
        LlmArch::LLADA_MOE => {
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?;
            if let Some(vals) = vals {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // diffusion language model uses non-causal attention
            // (llada-moe.cpp:7-8)
            h.causal_attn = false;
        }

        // ---- models/minimax-01.cpp:4-24 (arch batch 10) ----
        LlmArch::MINIMAX_01 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_residual_scale = req_f32(k(LlmKv::RESIDUAL_SCALE))?;

            // we use n_embd_head_la to set recurrent memory n_embd_s
            // (minimax-01.cpp:9)
            h.n_embd_head_la = h.n_embd_head_k_full;

            // Mark recurrent layers (lightning attention layers) — the
            // explicit array wins, else the every-8th-full interval
            // (minimax-01.cpp:12-18; the qwen3next/qwen35 default is 4).
            // NB: minimax-01 never reads expert_feed_forward_length nor
            // expert_weights_scale — the experts sit at the dense n_ff and
            // the scale stays 0 (the same "never read" trap as qwen3next)
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::ATTENTION_RECURRENT_LAYERS),
                h.n_layer_all as usize,
            )?;
            if let Some(vals) = vals {
                h.is_recr_impl[..vals.len()].copy_from_slice(&vals);
            } else {
                let full_attn_interval = gguf
                    .get_u32(&k(LlmKv::FULL_ATTENTION_INTERVAL))
                    .unwrap_or(8);
                for i in 0..h.n_layer_all as usize {
                    h.is_recr_impl[i] = u32::from(
                        (i < h.n_layer() as usize) && ((i as u32 + 1) % full_attn_interval != 0),
                    );
                }
            }
        }

        // ---- models/granite-switch.cpp:5-72 (arch batch 10) ----
        LlmArch::GRANITE_SWITCH => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::RESIDUAL_SCALE)) {
                h.f_residual_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_SCALE)) {
                h.f_attention_scale = v;
            }

            // granite-switch.cpp:12-15 — rope_finetuned defaults true and
            // paints the whole rope_pattern
            let rope_finetuned = gguf
                .get_bool(&k(LlmKv::ROPE_SCALING_FINETUNED))
                .unwrap_or(true);
            h.rope_finetuned = rope_finetuned; // needed for round trip save
            h.rope_pattern
                .iter_mut()
                .for_each(|p| *p = u32::from(rope_finetuned));

            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }

            // adapter bookkeeping (granite-switch.cpp:25-26) — the maps are
            // re-read by the tensor loader, which owns the substitute range
            // check against n_vocab (granite-switch.cpp:85-91)
            let n_adapters = req_u32(k(LlmKv::ADAPTER_COUNT))?;
            let max_lora_rank = req_u32(k(LlmKv::ADAPTER_LORA_RANK))?;
            h.graniteswitch_n_adapters = n_adapters;
            h.graniteswitch_max_lora_rank = max_lora_rank;
            h.graniteswitch_router_gain =
                gguf.get_f32(&k(LlmKv::ADAPTER_ROUTER_GAIN)).unwrap_or(0.0);

            // bound counts that size tensors (granite-switch.cpp:30-35)
            if n_adapters > 4096 {
                return Err(format!("graniteswitch: invalid adapter count {n_adapters}"));
            }
            if max_lora_rank > 4096 {
                return Err(format!("graniteswitch: invalid lora rank {max_lora_rank}"));
            }

            // the two token-id arrays must match the adapter count
            // (granite-switch.cpp:42-46)
            let n_tok = arr_len(gguf, &k(LlmKv::ADAPTER_TOKEN_IDS_ACTIVATE));
            let n_sub = arr_len(gguf, &k(LlmKv::ADAPTER_TOKEN_IDS_SUBSTITUTE));
            if n_tok != Some(n_adapters) || n_sub != Some(n_adapters) {
                return Err(format!(
                    "graniteswitch: adapter token id arrays ({}, {}) do not match adapter count {}",
                    n_tok.map(|v| v.to_string()).unwrap_or_else(|| "0".into()),
                    n_sub.map(|v| v.to_string()).unwrap_or_else(|| "0".into()),
                    n_adapters
                ));
            }

            // extra single-head attention layer at the END (index n_real)
            // holds the router K/V. reusing n_layer_nextn keeps n_layer() ==
            // n_real, so the regular layers keep their indices and the KV
            // cache shift/defrag skips the router layer (granite-switch.cpp:58-71)
            let n_real = h.n_layer();
            if n_real as usize >= crate::arch::LLAMA_MAX_LAYERS {
                return Err(format!(
                    "graniteswitch: block count {n_real} exceeds LLAMA_MAX_LAYERS"
                ));
            }
            h.router_layer = n_real as i32;
            h.n_layer_all = n_real + 1;
            h.n_layer_nextn = 1;

            h.n_head_arr[n_real as usize] = 1;
            h.n_head_kv_arr[n_real as usize] = 1;
            h.n_ff_arr[n_real as usize] = 0;
        }

        // ---- arch batch 11b (2026-10): the long-tail queue, second half ----

        // ---- models/arcee.cpp:3-11 ----
        LlmArch::ARCEE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/jais2.cpp:3-11 ----
        LlmArch::JAIS2 => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/talkie.cpp:3-11 ----
        LlmArch::TALKIE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
        }

        // ---- models/nanbeige.cpp:3-35 ----
        LlmArch::NANBEIGE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // num_loops (default 1) + skip_loop_final_norm (default false)
            // (nanbeige.cpp:6-11)
            let n_loops = gguf.get_u32(&k(LlmKv::NUM_LOOPS)).unwrap_or(1);
            if n_loops < 1 {
                return Err(format!("nanbeige: num_loops {n_loops} < 1"));
            }
            h.nanbeige_skip_loop_final_norm = gguf
                .get_bool(&k(LlmKv::SKIP_LOOP_FINAL_NORM))
                .unwrap_or(false);

            let n_layer_phys = h.n_layer();
            h.nanbeige_n_layer_phys = n_layer_phys;
            h.nanbeige_n_loops = n_loops;

            // expand the logical layer count before load_tensors allocates
            // the layers / KV (nanbeige.cpp:15-32)
            if (n_layer_phys as usize) * (n_loops as usize) > crate::arch::LLAMA_MAX_LAYERS {
                return Err(format!(
                    "nanbeige: n_layer_phys {n_layer_phys} * n_loops {n_loops} exceeds LLAMA_MAX_LAYERS"
                ));
            }
            if n_loops > 1 {
                for j in 1..n_loops as usize {
                    for i in 0..n_layer_phys as usize {
                        let dst = i + j * n_layer_phys as usize;
                        h.n_head_arr[dst] = h.n_head_arr[i];
                        h.n_head_kv_arr[dst] = h.n_head_kv_arr[i];
                        h.n_ff_arr[dst] = h.n_ff_arr[i];
                        h.is_swa_impl[dst] = h.is_swa_impl[i];
                        h.is_recr_impl[dst] = h.is_recr_impl[i];
                    }
                }
                h.n_layer_all = n_layer_phys * n_loops;
            }
        }

        // ---- models/dream.cpp:3-16 (llada-family diffusion) ----
        LlmArch::DREAM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // diffusion language model uses non-causal attention
            // (dream.cpp:14-15)
            h.causal_attn = false;
        }

        // ---- models/rnd1.cpp:3-14 (qwen3moe converted to diffusion) ----
        LlmArch::RND1 => {
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?;
            if let Some(vals) = vals {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // non-causal attention for diffusion (rnd1.cpp:12-13)
            h.causal_attn = false;
        }

        // ---- models/eurobert.cpp:3-9 (encoder) ----
        LlmArch::EUROBERT => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- arch batch 12 (2026-10): the final long-tail queue ----

        // ---- models/hrm-text.cpp:6-32 ----
        LlmArch::HRM_TEXT => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }

            h.n_hrm_layers_per_stack = req_u32(k(LlmKv::HRM_LAYERS_PER_STACK))?;
            h.n_hrm_h_cycles = req_u32(k(LlmKv::HRM_H_CYCLES))?;
            h.n_hrm_l_cycles = req_u32(k(LlmKv::HRM_L_CYCLES))?;

            // prefix-LM prefill is not implemented (causal attention only);
            // kept for round-trip (hrm-text.cpp:14-15)
            if let Some(v) = gguf.get_bool(&k(LlmKv::HRM_PREFIX_LM)) {
                h.hrm_prefix_lm = v;
            }

            // hrm-text.cpp:17-19
            if h.n_hrm_layers_per_stack == 0 || h.n_hrm_h_cycles == 0 || h.n_hrm_l_cycles == 0 {
                return Err("hrm-text: layers_per_stack / h_cycles / l_cycles must be > 0".into());
            }

            // the GGUF block count is the expanded cache-slot count
            // (hrm-text.cpp:21-23)
            let n_slot = h.n_hrm_layers_per_stack * h.n_hrm_h_cycles * (h.n_hrm_l_cycles + 1);
            if h.n_layer() != n_slot {
                return Err(format!(
                    "hrm-text: block count {} != lps*h_cycles*(l_cycles+1) = {n_slot}",
                    h.n_layer()
                ));
            }
        }

        // ---- models/laguna.cpp:9-63 ----
        LlmArch::LAGUNA => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.n_layer_dense_lead = req_u32(k(LlmKv::LEADING_DENSE_BLOCK_COUNT))?;
            // get_key_or_arr(EXPERT_FEED_FORWARD_LENGTH, n_layer_all) — the
            // C default `required = true`
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }

            // one shared expert whose size is stored directly (routed and
            // shared may differ); the count defaults to 1 (laguna.cpp:17-22)
            h.n_expert_shared = 1;
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_COUNT)) {
                h.n_expert_shared = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            if h.n_ff_shexp == 0 {
                // weightless fixtures omit the key; derive a nonzero size so
                // the shared expert is still built (laguna.cpp:23-28)
                h.n_ff_shexp = h.n_ff_exp(0) * h.n_expert_shared;
            }

            // sliding-window attention is OPTIONAL — XS.2 is hybrid (period 4
            // starting with full), M.1 has no sliding window at all
            // (laguna.cpp:30-49)
            h.n_swa = 0;
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW)) {
                h.n_swa = v;
            }
            if h.n_swa > 0 {
                h.swa_type = LlamaSwaType::STANDARD;
                b8_load_swa_pattern(gguf, arch, h, 4, true)?; // XS.2: FULL at il%4==0

                // per-layer-type RoPE: full layers keep the generic
                // rope_freq_base_train the base load read; SWA layers run
                // plain rope (no YaRN scaling — do NOT inherit 1/factor)
                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = 1.0;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
                if let Some(v) = gguf.get_u32(&k(LlmKv::ROPE_DIMENSION_COUNT_SWA)) {
                    h.n_rot_swa = v;
                }
            }

            // default the expert gating function to SIGMOID when the key is
            // absent (laguna.cpp:51-55, matches the HF reference)
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }
        }

        // ---- models/maple.cpp:3-22 ----
        LlmArch::MAPLE => {
            h.swa_type = LlamaSwaType::STANDARD;
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // get_key_or_arr — REQUIRED (the C default)
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);

            // ml.get_arr(ATTENTION_SLIDING_WINDOW_PATTERN, is_swa_impl) —
            // REQUIRED; the array wins outright (inline_load_swa_pattern's
            // Array branch, no scalar fallback)
            let key = kv_name(arch, LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN);
            match gguf.find_key(&key) {
                Some(ggml::Value::Array(_, items)) => {
                    h.is_swa_impl = vec![0u32; crate::arch::LLAMA_MAX_LAYERS];
                    for (il, it) in items.iter().enumerate() {
                        // std::array<uint32_t, N> get_arr: BOOL accepted with
                        // `x != 0` widening (llama-model-loader.cpp:371,396)
                        h.is_swa_impl[il] = match it {
                            ggml::Value::Bool(b) => u32::from(*b),
                            ggml::Value::U32(x) => *x,
                            ggml::Value::I32(x) => *x as u32,
                            _ => return Err(format!("key {key} has wrong array element type")),
                        };
                    }
                }
                _ => return Err(format!("key {key} not found")),
            }

            // maple.cpp:12-14 — the SWA rope pair (the scale copy inherits
            // the full layer's rope_freq_scale_train)
            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            // get_key_or_arr(SWIGLU_CLAMP_EXP, swiglu_clamp_exp, n_layer_all)
            // — REQUIRED, but the graph never consumes it (maple.cpp:16; no
            // clamp argument on the build_moe_ffn call at :121-131)
            if let Some(vals) =
                get_key_or_arr_f32_local(gguf, &k(LlmKv::SWIGLU_CLAMP_EXP), h.n_layer_all as usize)?
            {
                h.swiglu_clamp_exp[..vals.len()].copy_from_slice(&vals);
            } else {
                return Err(format!("key {} not found", k(LlmKv::SWIGLU_CLAMP_EXP)));
            }
        }

        // ---- arch batch 13 (2026-09): the P0 standard-attention queue ----

        // ---- models/cohere2.cpp:3-19 ----
        LlmArch::COHERE2 => {
            h.swa_type = LlamaSwaType::STANDARD;
            inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)

            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/chatglm.cpp:3-23 ----
        LlmArch::CHATGLM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/bitnet.cpp:3-10 ----
        LlmArch::BITNET => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/dbrx.cpp:3-11 ----
        LlmArch::DBRX => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            h.f_clamp_kqv = req_f32(k(LlmKv::ATTENTION_CLAMP_KQV))?;
        }

        // ---- models/mistral3.cpp:3-27 ----
        LlmArch::MISTRAL3 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_attn_temp_scale = gguf
                .get_f32(&k(LlmKv::ATTENTION_TEMPERATURE_SCALE))
                .unwrap_or(0.0);

            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_BETA_FAST)) {
                h.yarn_beta_fast = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_BETA_SLOW)) {
                h.yarn_beta_slow = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_SCALING_YARN_LOG_MUL)) {
                h.rope_yarn_log_mul = v;
            }

            h.f_attn_temp_offset = 0.0;

            // mistral3.cpp:14-19 — the floor scale borrows n_ctx_orig_yarn
            if h.f_attn_temp_scale != 0.0 {
                h.n_attn_temp_floor_scale = h.n_ctx_orig_yarn;
                if h.n_attn_temp_floor_scale == 0 {
                    return Err(
                        "invalid n_ctx_orig_yarn for attention temperature scaling".to_string()
                    );
                }
            }
        }

        // ---- models/minicpm3.cpp:3-12 ----
        LlmArch::MINICPM3 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.n_lora_q = req_u32(k(LlmKv::ATTENTION_Q_LORA_RANK))?;
            h.n_lora_kv = req_u32(k(LlmKv::ATTENTION_KV_LORA_RANK))?;
        }

        // ---- models/glm4.cpp:3-13 ----
        LlmArch::GLM4 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(sections) = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                false,
            )? {
                h.rope_sections = sections;
            }
        }

        // ---- models/exaone4.cpp:3-22 ----
        LlmArch::EXAONE4 => {
            if h.n_layer() == 64 {
                // 32B
                h.swa_type = LlamaSwaType::STANDARD;
                h.n_swa = 4096;
                inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)

                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
            }

            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW)) {
                h.n_swa = v;
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/llama4.cpp:3-42 ----
        LlmArch::LLAMA4 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_moe_layer_step = req_u32(k(LlmKv::INTERLEAVE_MOE_LAYER_STEP))?;

            let found_swa = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW));
            if found_swa == Some(0) {
                h.swa_type = LlamaSwaType::NONE;
                h.n_no_rope_layer_step = h.n_layer(); // always use rope
            } else {
                h.swa_type = LlamaSwaType::CHUNKED;
                h.n_swa = 8192;
                h.n_attn_temp_floor_scale = 8192;
                h.f_attn_temp_scale = 0.1;
                h.f_attn_temp_offset = 1.0;

                inline_load_swa_pattern(gguf, arch, h)?; // pattern: 3 chunked - 1 full

                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
            }

            // the LLM_TYPE switch + use_kq_norm (llama4.cpp:26-41): 17B_128E
            // (n_expert 128) is the only type with use_kq_norm false; the
            // n_expert == 0 MobileLLM types set it true too (the loader
            // refuses those files right after, at llama4.cpp:47-49)
            h.use_kq_norm = h.n_expert != 128;
        }

        // ---- models/qwen2vl.cpp:3-5 ----
        LlmArch::QWEN2VL => {
            h.rope_sections = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                true,
            )?
            .unwrap();
        }

        // ---- models/qwen3vl.cpp:3-14 ----
        // (QWEN3TTS shares it — the pure typedef of models.h:625-627)
        LlmArch::QWEN3VL | LlmArch::QWEN3TTS => {
            h.n_deepstack_layers = gguf.get_u32(&k(LlmKv::NUM_DEEPSTACK_LAYERS)).unwrap_or(0);
            h.rope_sections = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                true,
            )?
            .unwrap();
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/pockettts.cpp:6-13 ----
        LlmArch::POCKETTTS => {
            // LLM_KV_ATTENTION_LAYERNORM_EPS (the graph's LLM_NORM eps);
            // the LLM_TYPE_109M/335M n_layer mapping is description-string
            // cosmetics the port does not keep (meta.rs module doc)
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
        }

        // ---- models/wavtokenizer-dec.cpp:3-7 ----
        LlmArch::WAVTOKENIZER_DEC => {
            h.f_norm_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_EPS))?;
            h.f_norm_group_eps = req_f32(k(LlmKv::ATTENTION_GROUPNORM_EPS))?;
            h.n_norm_groups = req_u32(k(LlmKv::ATTENTION_GROUPNORM_GROUPS))?;
        }

        // ---- models/qwen3vlmoe.cpp:3-14 ----
        LlmArch::QWEN3VLMOE => {
            h.n_deepstack_layers = gguf.get_u32(&k(LlmKv::NUM_DEEPSTACK_LAYERS)).unwrap_or(0);
            h.rope_sections = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                true,
            )?
            .unwrap();
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/glm-dsa.cpp:29-72 ----
        LlmArch::GLM_DSA => {
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(sections) = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                false,
            )? {
                h.rope_sections = sections;
            }

            // MoE parameters
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }

            // deepseek MLA parameters (glm-dsa.cpp:40-46 — the duplicated
            // get_key_or_arr/EXPERT_SHARED_COUNT reads are a no-op the second
            // time)
            h.n_lora_q = req_u32(k(LlmKv::ATTENTION_Q_LORA_RANK))?;
            h.n_lora_kv = req_u32(k(LlmKv::ATTENTION_KV_LORA_RANK))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_KEY_LENGTH_MLA)) {
                h.n_embd_head_k_mla_impl = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_VALUE_LENGTH_MLA)) {
                h.n_embd_head_v_mla_impl = v;
            }

            // DSA parameters
            h.indexer_n_head = req_u32(k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT))?;
            h.indexer_head_size = req_u32(k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH))?;
            h.indexer_top_k = req_u32(k(LlmKv::ATTENTION_INDEXER_TOP_K))?;

            // Expert gating function (GLM-4.5 uses sigmoid) — NONE (absent)
            // defaults to SIGMOID (glm-dsa.cpp:53-57)
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == crate::hparams::LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = crate::hparams::LlamaExpertGatingFuncType::SIGMOID as u32;
            }

            // BC for GLM 5, 5.1 (full indexers) without indexer_types
            // metadata (glm-dsa.cpp:59-66)
            let is_pre_5_2 = h.n_ctx_train < 1048576;
            if is_pre_5_2 {
                h.is_indexer_full_impl = vec![1u32; crate::arch::LLAMA_MAX_LAYERS];
            } else {
                // GLM_5_2_DEFAULT_INDEXER_TYPES (glm-dsa.cpp:6-27): a 1 then
                // [1, 0, 0, 0] × 21 — two leading full layers, then the
                // every-4th-full pattern
                let mut d = vec![0u32; crate::arch::LLAMA_MAX_LAYERS];
                d[0] = 1;
                d[1] = 1;
                for i in 0..21u32 {
                    d[2 + (i * 4) as usize] = 1;
                }
                h.is_indexer_full_impl = d;
            }
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::ATTENTION_INDEXER_TYPES),
                h.n_layer() as usize,
            )? {
                h.is_indexer_full_impl[..vals.len()].copy_from_slice(&vals);
            }
        }

        // ---- arch batch 15 (2026-10): the P1+P2 queue ----

        // ---- models/qwen.cpp:3-11 ----
        LlmArch::QWEN => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/maincoder.cpp:3-10 ----
        LlmArch::MAINCODER => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/pangu-embed.cpp:3-11 ----
        LlmArch::PANGU_EMBED => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/cogvlm.cpp:3-10 ----
        LlmArch::COGVLM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
        }

        // ---- models/spark2-5.cpp:3-18 ----
        LlmArch::SPARK2_5 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;

            h.swa_type = LlamaSwaType::STANDARD;
            inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)

            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }
        }

        // ---- models/muse-glimmer.cpp:3-19 ----
        LlmArch::MUSE_GLIMMER => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::FINAL_LOGIT_SOFTCAPPING)) {
                h.f_final_logit_softcapping = v;
            }
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;

            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            h.swa_type = LlamaSwaType::STANDARD;
            inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)
        }

        // ---- models/llada.cpp:3-17 ----
        LlmArch::LLADA => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // non-causal attention for the diffusion model (llada.cpp:16)
            h.causal_attn = false;
        }

        // ---- models/plm.cpp:3-11 ----
        LlmArch::PLM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.n_lora_kv = req_u32(k(LlmKv::ATTENTION_KV_LORA_RANK))?;
        }

        // ---- models/hunyuan-vl.cpp:3-21 (+ hunyuan-dense typedef) ----
        LlmArch::HUNYUAN_VL | LlmArch::HUNYUAN_DENSE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(sections) = crate::meta::get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                false,
            )? {
                h.rope_sections = sections;
            }

            // XDRoPE / NTK-aware scaling: base = theta * alpha^(dim/(dim-2))
            // (hunyuan-vl.cpp:8-12)
            if h.rope_scaling_alpha > 0.0 {
                let dim = h.n_embd_head_k(0) as f32;
                h.rope_freq_base_train =
                    h.rope_freq_base_train * h.rope_scaling_alpha.powf(dim / (dim - 2.0));
            }
        }

        // ---- models/granite-swa.cpp:5-54 ----
        LlmArch::GRANITE_SWA => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            h.f_logit_scale = req_f32(k(LlmKv::LOGIT_SCALE))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::RESIDUAL_SCALE)) {
                h.f_residual_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EMBEDDING_SCALE)) {
                h.f_embedding_scale = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_SCALE)) {
                h.f_attention_scale = v;
            }

            // MoE expert configuration
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_COUNT)) {
                h.n_expert = v;
            }
            if let Some(vals) = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_USED_COUNT),
                h.n_layer_all as usize,
            )? {
                h.n_expert_used_arr[..vals.len()].copy_from_slice(&vals);
            }

            // iSWA configuration
            inline_load_swa_pattern(gguf, arch, h)?;
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.swa_type = LlamaSwaType::STANDARD;

            // Granite-4 Vision deepstack mapping (granite-swa.cpp:22-31) —
            // text-only files carry none (the hparams default is all -1)
            if let Some(vals) = get_arr_i32_local(gguf, &k(LlmKv::DEEPSTACK_MAPPING))? {
                h.deepstack_mapping_arr = vals;
                let unique: std::collections::BTreeSet<i32> = h
                    .deepstack_mapping_arr
                    .iter()
                    .copied()
                    .filter(|&v| v >= 0)
                    .collect();
                h.n_deepstack_layers = unique.len() as u32;
                for &v in &unique {
                    if v as u32 > h.n_deepstack_layers {
                        return Err(format!(
                            "Invalid deepstack index: {v} > {}",
                            h.n_deepstack_layers
                        ));
                    }
                }
            }

            // per-layer RoPE pattern (optional)
            if let Some(vals) = get_arr_u32_local(gguf, &k(LlmKv::ATTENTION_ROPE_PATTERN))? {
                h.rope_pattern = vals;
            }

            // For Granite MoE Shared (granite-swa.cpp:53)
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
        }

        // ---- models/afmoe.cpp:3-36 ----
        LlmArch::AFMOE => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            // n_ff_exp is REQUIRED for afmoe (:6)
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW)) {
                h.n_swa = v;
            }

            // iSWA when a window is declared (afmoe.cpp:15-24)
            if h.n_swa > 0 {
                h.swa_type = LlamaSwaType::STANDARD;
                inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)

                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
            } else {
                h.swa_type = LlamaSwaType::NONE;
            }

            // the sigmoid default (:27-29)
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }
        }

        // ---- models/mellum.cpp:3-25 ----
        LlmArch::MELLUM => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            // n_ff_exp is REQUIRED for mellum (:5)
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_SLIDING_WINDOW)) {
                h.n_swa = v;
            }

            if h.n_swa > 0 {
                h.swa_type = LlamaSwaType::STANDARD;
                inline_load_swa_pattern(gguf, arch, h)?; // load_swa_pattern(ml, 4)

                h.rope_freq_base_train_swa = h.rope_freq_base_train;
                h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
                if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                    h.rope_freq_base_train_swa = v;
                }
            } else {
                h.swa_type = LlamaSwaType::NONE;
            }
        }

        // ---- models/gemma-embedding.cpp:3-28 ----
        LlmArch::GEMMA_EMBEDDING => {
            h.swa_type = LlamaSwaType::SYMMETRIC;
            b8_load_swa_pattern(gguf, arch, h, 6, false)?; // load_swa_pattern(ml, 6)

            // embeddings do not use causal attention (gemma-embedding.cpp:7)
            h.causal_attn = false;

            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            // the sentence-transformers dense modules (:14-17)
            if let Some(v) = gguf.get_u32(&k(LlmKv::DENSE_2_FEAT_IN)) {
                h.dense_2_feat_in = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::DENSE_2_FEAT_OUT)) {
                h.dense_2_feat_out = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::DENSE_3_FEAT_IN)) {
                h.dense_3_feat_in = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::DENSE_3_FEAT_OUT)) {
                h.dense_3_feat_out = v;
            }

            // the GGML_ASSERTs of :19-20
            if h.dense_2_feat_in != 0 && h.dense_2_feat_in != h.n_embd {
                return Err("dense_2_feat_in must be equal to n_embd".to_string());
            }
            if h.dense_3_feat_out != 0 && h.dense_3_feat_out != h.n_embd {
                return Err("dense_3_feat_out must be equal to n_embd".to_string());
            }

            h.f_attention_scale = 1.0 / (h.n_embd_head_k(0) as f32).sqrt();
        }

        // ---- models/hy-v3.cpp:3-20 ----
        LlmArch::HY_V3 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }

            // the sigmoid router default (hy-v3.cpp:12-14)
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }
        }

        // ---- models/mimo2.cpp:3-23 ----
        LlmArch::MIMO2 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            h.swa_type = LlamaSwaType::STANDARD;

            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            inline_load_swa_pattern(gguf, arch, h)?;

            // the value scale (:14-17 — only a non-1.0 value counts)
            if let Some(v) = gguf.get_f32(&k(LlmKv::ATTENTION_VALUE_SCALE)) {
                if v != 1.0 {
                    h.f_attn_value_scale = v;
                }
            }
        }

        // ---- models/step35.cpp:3-35 ----
        LlmArch::STEP35 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;

            h.swa_type = LlamaSwaType::STANDARD;

            // full-attention layers use only half of the RoPE dimensions
            // (step35.cpp:9)
            h.n_rot_full = h.n_rot_full / 2;

            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH)) {
                h.n_ff_shexp = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }

            // the sigmoid default (step35.cpp:19-21)
            if h.expert_gating_func == LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = LlamaExpertGatingFuncType::SIGMOID as u32;
            }

            h.n_swa = req_u32(k(LlmKv::ATTENTION_SLIDING_WINDOW))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::ROPE_FREQ_BASE_SWA)) {
                h.rope_freq_base_train_swa = v;
            }

            inline_load_swa_pattern(gguf, arch, h)?;

            if let Some(vals) =
                get_key_or_arr_f32_local(gguf, &k(LlmKv::SWIGLU_CLAMP_EXP), h.n_layer_all as usize)?
            {
                h.swiglu_clamp_exp = vals;
            }
            if let Some(vals) = get_key_or_arr_f32_local(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_SHEXP),
                h.n_layer_all as usize,
            )? {
                h.swiglu_clamp_shexp = vals;
            }
        }

        // ---- models/hy-v4.cpp:24-68 ----
        LlmArch::HY_V4 => {
            h.f_norm_rms_eps = req_f32(k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS))?;
            if let Some(v) = gguf.get_u32(&k(LlmKv::LEADING_DENSE_BLOCK_COUNT)) {
                h.n_layer_dense_lead = v;
            }
            h.n_lora_q = req_u32(k(LlmKv::ATTENTION_Q_LORA_RANK))?;
            h.n_lora_kv = req_u32(k(LlmKv::ATTENTION_KV_LORA_RANK))?;
            h.n_embd_head_k_mla_impl = req_u32(k(LlmKv::ATTENTION_KEY_LENGTH_MLA))?;
            h.n_embd_head_v_mla_impl = req_u32(k(LlmKv::ATTENTION_VALUE_LENGTH_MLA))?;
            let vals = get_key_or_arr_u32_local(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
            )?
            .ok_or_else(|| format!("key {} not found", k(LlmKv::EXPERT_FEED_FORWARD_LENGTH)))?;
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = req_u32(k(LlmKv::EXPERT_SHARED_COUNT))?;
            if let Some(v) = gguf.get_f32(&k(LlmKv::EXPERT_WEIGHTS_SCALE)) {
                h.expert_weights_scale = v;
            }
            if let Some(v) = gguf.get_bool(&k(LlmKv::EXPERT_WEIGHTS_NORM)) {
                h.expert_weights_norm = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::EXPERT_GATING_FUNC)) {
                h.expert_gating_func = v;
            }

            // the routed-expert SwiGLU clamp (:39 — read for the KV
            // contract; this revision's graph never consumes it)
            if let Some(vals) =
                get_key_or_arr_f32_local(gguf, &k(LlmKv::SWIGLU_CLAMP_EXP), h.n_layer_all as usize)?
            {
                h.swiglu_clamp_exp = vals;
            }

            h.dsv4_hc_mult = req_u32(k(LlmKv::HYPER_CONNECTION_COUNT))?;
            h.dsv4_hc_eps = req_f32(k(LlmKv::HYPER_CONNECTION_EPSILON))?;
            h.hc_magnitude = req_f32(k(LlmKv::HYPER_CONNECTION_MAGNITUDE))?;

            // DSA is absent on the all-full-attention checkpoints (:46-48)
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT)) {
                h.indexer_n_head = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH)) {
                h.indexer_head_size = v;
            }
            if let Some(v) = gguf.get_u32(&k(LlmKv::ATTENTION_INDEXER_TOP_K)) {
                h.indexer_top_k = v;
            }

            if h.indexer_top_k > 0 {
                // the indexer k_norm LayerNorm reads f_norm_eps (:52-53)
                h.f_norm_eps = h.f_norm_rms_eps;

                if h.indexer_n_head == 0 || h.indexer_head_size <= h.n_rot(0) {
                    return Err("hy_v4: bad indexer head count / key length".to_string());
                }

                if let Some(vals) = get_key_or_arr_u32_local(
                    gguf,
                    &k(LlmKv::ATTENTION_INDEXER_TYPES),
                    h.n_layer() as usize,
                )? {
                    h.is_indexer_full_impl = vec![0u32; crate::arch::LLAMA_MAX_LAYERS];
                    h.is_indexer_full_impl[..vals.len()].copy_from_slice(&vals);
                }
                if !h.is_indexer_full(0) {
                    return Err(
                        "hy_v4: layer 0 must own an indexer, nothing precedes it to share"
                            .to_string(),
                    );
                }
            }

            if !h.is_mla() {
                return Err("hy_v4: is_mla() must hold".to_string());
            }
        }

        _ => {}
    }

    // llama-model.cpp:1419-1421 — re-checked *after* the arch switch, so the
    // biases assigned inside `load_arch_hparams` (baichuan 13B / bloom /
    // refact hard-code 8.0) also flip use_alibi. meta.rs's generic path runs
    // before this function and only sees the KV-loaded biases.
    if h.f_max_alibi_bias > 0.0 {
        h.use_alibi = true;
    }
    Ok(())
}

/// `llama_model_base::load_swa_pattern(ml, n_pattern = 4, dense_first = false)`
/// for the arch batch's call site (olmo2.cpp:8): the pattern array wins
/// outright, else the scalar pattern key overrides the default before
/// `set_swa_pattern`. `meta.rs::load_swa_pattern` is the same code but private
/// to that module.
fn inline_load_swa_pattern(gguf: &Gguf, arch: LlmArch, h: &mut LlamaHparams) -> Result<(), String> {
    let key = kv_name(arch, LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN);

    let mut n_pattern = 4u32;
    match gguf.find_key(&key) {
        Some(ggml::Value::Array(_, items)) => {
            h.is_swa_impl = vec![0u32; crate::arch::LLAMA_MAX_LAYERS];
            for (il, it) in items.iter().enumerate() {
                // the std::array<uint32_t, N> get_arr overload accepts BOOL
                // too (llama-model-loader.cpp:371) with `x != 0` widening
                // (:396-398) — Olmo-3 ships the pattern as array(bool)
                h.is_swa_impl[il] = match it {
                    ggml::Value::Bool(b) => u32::from(*b),
                    ggml::Value::U32(x) => *x,
                    ggml::Value::I32(x) => *x as u32,
                    _ => return Err(format!("key {key} has wrong array element type")),
                };
            }
            return Ok(());
        }
        Some(ggml::Value::U32(v)) => n_pattern = *v,
        Some(ggml::Value::I32(v)) => n_pattern = *v as u32,
        _ => {}
    }
    h.set_swa_pattern(n_pattern, false);
    Ok(())
}

/// arch batch 8: `llama_model_base::load_swa_pattern(ml, n_pattern,
/// dense_first)` with the `dense_first` parameter spelled out — cohere2moe
/// passes true (:28), exaone-moe false (:6). Same semantics as
/// [`inline_load_swa_pattern`]: the pattern array wins outright, else the
/// scalar key overrides the default.
fn b8_load_swa_pattern(
    gguf: &Gguf,
    arch: LlmArch,
    h: &mut LlamaHparams,
    n_pattern_default: u32,
    dense_first: bool,
) -> Result<(), String> {
    let key = kv_name(arch, LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN);

    let mut n_pattern = n_pattern_default;
    match gguf.find_key(&key) {
        Some(ggml::Value::Array(_, items)) => {
            h.is_swa_impl = vec![0u32; crate::arch::LLAMA_MAX_LAYERS];
            for (il, it) in items.iter().enumerate() {
                // the std::array<uint32_t, N> get_arr overload accepts BOOL
                // too (llama-model-loader.cpp:371) with `x != 0` widening
                // (:396-398)
                h.is_swa_impl[il] = match it {
                    ggml::Value::Bool(b) => u32::from(*b),
                    ggml::Value::U32(x) => *x,
                    ggml::Value::I32(x) => *x as u32,
                    _ => return Err(format!("key {key} has wrong array element type")),
                };
            }
            return Ok(());
        }
        Some(ggml::Value::U32(v)) => n_pattern = *v,
        Some(ggml::Value::I32(v)) => n_pattern = *v as u32,
        _ => {}
    }
    h.set_swa_pattern(n_pattern, dense_first);
    Ok(())
}

/// Assemble the encoder weights for the encoder-only archs — the port's
/// analogue of the `model.layers[il].*` members src/models/{bert,t5}.cpp read
/// (C keeps them on `llama_model`, which is why they are built here and not at
/// the graph). `load_model` has already validated every tensor, so a missing
/// member is a loader bug, not a file problem.
impl LlamaModel {
    /// `models/bert.cpp:68-221` members.
    pub fn bert_weights(&self) -> crate::graph_arch::BertModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(self.arch, LlmArch::BERT, "bert_weights on {:?}", self.arch);
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| ga::BertLayerWeights {
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                attn_out_norm: l
                    .attn_out_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_out_norm")),
                attn_out_norm_b: l
                    .attn_out_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_out_norm_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l.ffn_down_b,
                layer_out_norm: l
                    .layer_out_norm
                    .unwrap_or_else(|| panic!("layer {il}: layer_out_norm")),
                layer_out_norm_b: l
                    .layer_out_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: layer_out_norm_b")),
            })
            .collect();
        ga::BertModelWeights {
            tok_embd: self.tok_embd,
            type_embd: self.token_types,
            pos_embd: self.position_embd.expect("position_embd"),
            tok_norm: self.token_embd_norm.expect("token_embd_norm"),
            tok_norm_b: self.token_embd_norm_b.expect("token_embd_norm_b"),
            layers,
            n_embd: self.n_embd() as i64,
        }
    }

    /// The wavtokenizer-dec weights (audio round 5) — the code→PCM decoder
    /// arch driven graph-side (`build_wavtokenizer_dec_forward`).
    pub fn wavtokenizer_dec_weights(&self) -> crate::graph_arch::WavtokenizerDecModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch,
            LlmArch::WAVTOKENIZER_DEC,
            "wavtokenizer_dec_weights on {:?}",
            self.arch
        );
        let posnet_layers = self
            .layers
            .iter()
            .take(self.hparams.posnet.n_layer as usize)
            .map(|l| {
                let p = l.posnet.as_ref().expect("posnet layer");
                ga::WavtokenizerPosnetLayerWeights {
                    norm1: p.norm1,
                    norm1_b: p.norm1_b,
                    conv1: p.conv1,
                    conv1_b: p.conv1_b,
                    norm2: p.norm2,
                    norm2_b: p.norm2_b,
                    conv2: p.conv2,
                    conv2_b: p.conv2_b,
                    attn_norm: p.attn_norm,
                    attn_norm_b: p.attn_norm_b,
                    attn_q: p.attn_q,
                    attn_q_b: p.attn_q_b,
                    attn_k: p.attn_k,
                    attn_k_b: p.attn_k_b,
                    attn_v: p.attn_v,
                    attn_v_b: p.attn_v_b,
                    attn_o: p.attn_o,
                    attn_o_b: p.attn_o_b,
                    norm: p.norm,
                    norm_b: p.norm_b,
                }
            })
            .collect();
        let convnext_layers = self
            .layers
            .iter()
            .take(self.hparams.convnext.n_layer as usize)
            .map(|l| {
                let c = l.convnext.as_ref().expect("convnext layer");
                ga::WavtokenizerConvnextLayerWeights {
                    dw: c.dw.expect("convnext dw"),
                    dw_b: c.dw_b.expect("convnext dw_b"),
                    norm: c.norm.expect("convnext norm"),
                    norm_b: c.norm_b.expect("convnext norm_b"),
                    pw1: c.pw1.expect("convnext pw1"),
                    pw1_b: c.pw1_b.expect("convnext pw1_b"),
                    pw2: c.pw2.expect("convnext pw2"),
                    pw2_b: c.pw2_b.expect("convnext pw2_b"),
                    gamma: c.gamma.expect("convnext gamma"),
                }
            })
            .collect();
        ga::WavtokenizerDecModelWeights {
            tok_embd: self.tok_embd,
            conv1d: self.conv1d.expect("conv1d"),
            conv1d_b: self.conv1d_b.expect("conv1d_b"),
            tok_norm: self.tok_norm.expect("tok_norm"),
            tok_norm_b: self.tok_norm_b.expect("tok_norm_b"),
            output_norm: self.output_norm,
            output_norm_b: self.output_norm_b.expect("output_norm_b"),
            output: self.output,
            output_b: self.output_b.expect("output_b"),
            posnet_layers,
            convnext_layers,
        }
    }

    /// The pockettts weights (audio round 5) — the CALM flow backbone driven
    /// graph-side (`build_pockettts_forward`).
    pub fn pockettts_weights(&self) -> crate::graph_arch::PocketttsModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch,
            LlmArch::POCKETTTS,
            "pockettts_weights on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .take(self.n_layer() as usize)
            .enumerate()
            .map(|(il, l)| ga::PocketttsLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l.attn_norm_b.unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l.ffn_norm_b.unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect();
        ga::PocketttsModelWeights {
            tok_embd: self.tok_embd,
            output_norm: self.output_norm,
            output_norm_b: self.output_norm_b.expect("output_norm_b"),
            output: self.output,
            layers,
        }
    }

    /// The bert-variant family members — jina-bert-v2 / jina-bert-v3 /
    /// nomic-bert / nomic-bert-moe (the arch-keyed branches of the shared
    /// bert.cpp graph body).
    pub fn bert_variant_weights(
        &self,
        variant: crate::graph_arch::BertVariant,
    ) -> crate::graph_arch::BertVariantModelWeights {
        use crate::graph_arch as ga;
        let want = match variant {
            ga::BertVariant::JinaV2 => LlmArch::JINA_BERT_V2,
            ga::BertVariant::JinaV3 => LlmArch::JINA_BERT_V3,
            ga::BertVariant::Nomic => LlmArch::NOMIC_BERT,
            ga::BertVariant::NomicMoe => LlmArch::NOMIC_BERT_MOE,
        };
        assert_eq!(
            self.arch, want,
            "bert_variant_weights({variant:?}) on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .take(self.hparams.n_layer() as usize)
            .enumerate()
            .map(|(il, l)| ga::BertVariantLayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                attn_out_norm: l
                    .attn_out_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_out_norm")),
                attn_out_norm_b: l
                    .attn_out_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_out_norm_b")),
                attn_q_norm: l.attn_q_norm,
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm,
                attn_k_norm_b: l.attn_k_norm_b,
                attn_norm_2: l.attn_norm_2,
                attn_norm_2_b: l.attn_norm_2_b,
                ffn_gate: l.ffn_gate,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_inp: l.ffn_gate_inp,
                layer_out_norm: l
                    .layer_out_norm
                    .unwrap_or_else(|| panic!("layer {il}: layer_out_norm")),
                layer_out_norm_b: l
                    .layer_out_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: layer_out_norm_b")),
            })
            .collect();
        // the RANK head of the jina reranker files (cls/cls_b only, the tanh
        // flavor — jina-bert-v2.cpp:23-24); absent on plain embedders
        let rank_head = match (self.cls, self.cls_b) {
            (Some(cls), cls_b) => Some(ga::RankHead {
                cls: Some(cls),
                cls_b,
                cls_out: None,
                cls_out_b: None,
                cls_norm: None,
                // llama-graph.cpp:3755 (def4d406a): mean-first rides
                // `%s.classifier.pooling_type`, not the arch
                mean_first: self.hparams.pooling_type_cls
                    == crate::hparams::LlamaPoolingType::MEAN,
                modern_bert: false,
            }),
            (None, _) => None,
        };
        ga::BertVariantModelWeights {
            variant,
            tok_embd: self.tok_embd,
            type_embd: self.token_types,
            tok_norm: self.token_embd_norm.expect("token_embd_norm"),
            tok_norm_b: self.token_embd_norm_b.expect("token_embd_norm_b"),
            layers,
            n_embd: self.n_embd() as i64,
            rank_head,
        }
    }

    /// `models/neo-bert.cpp:43-134` members.
    pub fn neo_bert_weights(&self) -> crate::graph_arch::NeoBertModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(self.arch, LlmArch::NEO_BERT, "neo_bert_weights on {:?}", self.arch);
        let layers = self
            .layers
            .iter()
            .take(self.hparams.n_layer() as usize)
            .enumerate()
            .map(|(il, l)| ga::NeoBertLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l
                    .ffn_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect();
        ga::NeoBertModelWeights {
            tok_embd: self.tok_embd,
            // the loader mapped enc.output_norm onto the generic slot
            output_norm_enc: self.output_norm,
            layers,
        }
    }

    /// `models/modern-bert.cpp:72-172` members + the GTE reranker head
    /// (modern-bert.cpp:61-64, RANK pooling).
    pub fn modern_bert_weights(&self) -> crate::graph_arch::ModernBertModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch, LlmArch::MODERN_BERT,
            "modern_bert_weights on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .take(self.hparams.n_layer() as usize)
            .enumerate()
            .map(|(il, l)| ga::ModernBertLayerWeights {
                attn_norm: l.attn_norm,
                attn_norm_b: l.attn_norm_b,
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l
                    .ffn_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l.ffn_norm_b,
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l.ffn_down_b,
            })
            .collect();
        // the decision head (a7b94df2c modern-bert.cpp:110-122) — every
        // tensor is required when n_layer_decision > 0
        let decision = if self.hparams.n_layer_decision > 0 {
            Some(ga::ModernBertDecisionHead {
                type_embd: self.token_types.expect("decision: token_types"),
                cls: self.cls.expect("decision: cls"),
                cls_b: self.cls_b.expect("decision: cls_b"),
                cls_norm: self.cls_norm.expect("decision: cls_norm"),
                cls_norm_b: self.cls_norm_b.expect("decision: cls_norm_b"),
                cls_out: self.cls_out.expect("decision: cls_out"),
                cls_out_b: self.cls_out_b.expect("decision: cls_out_b"),
            })
        } else {
            None
        };
        // the head exists iff any of its tensors loaded (headless embedders
        // carry none of them)
        let rank_head = if self.cls.is_some()
            || self.cls_out.is_some()
            || self.cls_norm.is_some()
        {
            Some(ga::RankHead {
                cls: self.cls,
                cls_b: self.cls_b,
                cls_out: self.cls_out,
                cls_out_b: self.cls_out_b,
                cls_norm: self.cls_norm,
                // mean-first unless the file says otherwise
                // (`%s.classifier.pooling_type`; the loader defaults
                // UNSPECIFIED → MEAN, modern-bert.cpp:23-26)
                mean_first: self.hparams.pooling_type_cls
                    == crate::hparams::LlamaPoolingType::MEAN,
                modern_bert: true,
            })
        } else {
            None
        };
        ga::ModernBertModelWeights {
            tok_embd: self.tok_embd,
            tok_norm: self.token_embd_norm.expect("token_embd_norm"),
            output_norm: self.output_norm,
            layers,
            n_layer_decision: self.hparams.n_layer_decision as usize,
            decision,
            rank_head,
        }
    }

    /// `models/clef.cpp` members (a7b94df2c) — the qwen35 trunk (no
    /// nextn/MTP blocks run) + the decision head. Drives
    /// [`crate::clef::ClefState`].
    pub fn clef_weights(&self) -> crate::clef::ClefModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch, LlmArch::CLEF,
            "clef_weights on {:?}",
            self.arch
        );
        let n_trunk = self.hparams.n_layer() as usize;
        let layers = self.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| ga::Qwen35LayerWeights {
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
            .collect();
        crate::clef::ClefModelWeights {
            trunk: ga::Qwen35ModelWeights {
                tok_embd: self.tok_embd,
                output_norm: self.output_norm,
                output: self.output,
                cls_out: self.cls_out,
                cls_out_b: self.cls_out_b,
                layers,
            },
            head: self
                .clef_head
                .clone()
                .expect("clef_weights: the head tensors are missing"),
        }
    }

    /// clef's graph geometry — the qwen35 trunk facts + the head's LayerNorm
    /// eps and block counts (a7b94df2c clef.cpp:10-25).
    pub fn clef_params(
        &self,
        attn: crate::graph::AttnParams,
    ) -> crate::clef::ClefParams {
        let hp = &self.hparams;
        let n_layer = hp.n_layer() as usize;
        crate::clef::ClefParams {
            qwen35: crate::graph_arch::Qwen35Params {
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
            },
            f_norm_eps: hp.f_norm_eps,
            head: crate::clef::ClefHeadParams {
                n_layer_routing: hp.clef_n_layer_routing as usize,
                n_head_decision: hp.clef_n_head_decision as i64,
            },
        }
    }

    /// `models/t5.cpp:264-358` `graph<true>` members (the encoder half of
    /// t5encoder.cpp:24-39).
    pub fn t5_encoder_weights(&self) -> crate::graph_arch::T5EncoderModelWeights {
        use crate::graph_arch as ga;
        assert!(
            matches!(self.arch, LlmArch::T5 | LlmArch::T5ENCODER),
            "t5_encoder_weights on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| ga::T5EncoderLayerWeights {
                attn_norm_enc: l
                    .enc_attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_enc")),
                attn_rel_b_enc: l.enc_attn_rel_b,
                wq_enc: l.enc_wq.unwrap_or_else(|| panic!("layer {il}: wq_enc")),
                wk_enc: l.enc_wk.unwrap_or_else(|| panic!("layer {il}: wk_enc")),
                wv_enc: l.enc_wv.unwrap_or_else(|| panic!("layer {il}: wv_enc")),
                wo_enc: l.enc_wo.unwrap_or_else(|| panic!("layer {il}: wo_enc")),
                ffn_norm_enc: l
                    .enc_ffn_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_enc")),
                ffn_gate_enc: l.enc_ffn_gate,
                ffn_down_enc: l
                    .enc_ffn_down
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_enc")),
                ffn_up_enc: l
                    .enc_ffn_up
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_enc")),
            })
            .collect();
        ga::T5EncoderModelWeights {
            tok_embd: self.tok_embd,
            // t5encoder's arm maps enc.output_norm onto the generic slot; the
            // full T5 arch keeps it in enc_output_norm (MTP batch 17)
            output_norm_enc: match self.arch {
                LlmArch::T5 => self.enc_output_norm.expect("t5: enc.output_norm"),
                _ => self.output_norm,
            },
            layers,
        }
    }

    /// the decoder half of the full T5 arch (t5.cpp:60-107 — the dec.blk.*
    /// set; MTP batch 17)
    pub fn t5_decoder_weights(&self) -> crate::graph_arch::T5DecoderModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(self.arch, LlmArch::T5, "t5_decoder_weights on {:?}", self.arch);
        let layers = self
            .layers
            .iter()
            .take(self.hparams.dec_n_layer as usize)
            .enumerate()
            .map(|(il, l)| ga::T5DecoderLayerWeights {
                attn_norm: l.dec_attn_norm.unwrap_or_else(|| panic!("layer {il}: dec_attn_norm")),
                attn_rel_b: l.dec_attn_rel_b,
                wq: l.dec_wq.unwrap_or_else(|| panic!("layer {il}: dec_wq")),
                wk: l.dec_wk.unwrap_or_else(|| panic!("layer {il}: dec_wk")),
                wv: l.dec_wv.unwrap_or_else(|| panic!("layer {il}: dec_wv")),
                wo: l.dec_wo.unwrap_or_else(|| panic!("layer {il}: dec_wo")),
                attn_norm_cross: l
                    .dec_attn_norm_cross
                    .unwrap_or_else(|| panic!("layer {il}: dec_attn_norm_cross")),
                wq_cross: l.dec_wq_cross.unwrap_or_else(|| panic!("layer {il}: dec_wq_cross")),
                wk_cross: l.dec_wk_cross.unwrap_or_else(|| panic!("layer {il}: dec_wk_cross")),
                wv_cross: l.dec_wv_cross.unwrap_or_else(|| panic!("layer {il}: dec_wv_cross")),
                wo_cross: l
                    .dec_wo_cross
                    .unwrap_or_else(|| panic!("layer {il}: dec_wo_cross")),
                ffn_norm: l.dec_ffn_norm.unwrap_or_else(|| panic!("layer {il}: dec_ffn_norm")),
                ffn_gate: l.dec_ffn_gate,
                ffn_down: l.dec_ffn_down.unwrap_or_else(|| panic!("layer {il}: dec_ffn_down")),
                ffn_up: l.dec_ffn_up.unwrap_or_else(|| panic!("layer {il}: dec_ffn_up")),
            })
            .collect();
        ga::T5DecoderModelWeights {
            tok_embd: self.tok_embd,
            output_norm: self.output_norm,
            output: self.output,
            layers,
        }
    }

    /// `models/eurobert.cpp:11-32` members (arch batch 11b).
    /// arch batch 15 — gemma-embedding.cpp:30-67 (the encoder bundle)
    pub fn gemma_embedding_weights(&self) -> crate::graph_arch::GemmaEmbeddingModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch,
            LlmArch::GEMMA_EMBEDDING,
            "gemma_embedding_weights on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| ga::GemmaEmbeddingLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
            })
            .collect();
        ga::GemmaEmbeddingModelWeights {
            tok_embd: self.tok_embd,
            output_norm: self.output_norm,
            layers,
        }
    }

    pub fn eurobert_weights(&self) -> crate::graph_arch::EurobertModelWeights {
        use crate::graph_arch as ga;
        assert_eq!(
            self.arch,
            LlmArch::EUROBERT,
            "eurobert_weights on {:?}",
            self.arch
        );
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| ga::EurobertLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect();
        ga::EurobertModelWeights {
            tok_embd: self.tok_embd,
            output_norm: self.output_norm,
            layers,
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::tensor_name_suffix;
    use ggml::types::GgmlType;
    use ggml::{GgufType, Value};
    use std::fs::File;
    use std::path::Path;

    // local test models (present on this machine; tests skip if absent)
    const QWEN25: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const QWEN3_EMB: &str = "/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf";
    const PHI4_MINI: &str =
        "/home/jeffrey/.lmstudio/models/unsloth/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q6_K.gguf";
    // agent P arch expansion (all local; full-load tests are mmap-cheap because
    // the gguf header is the only thing read — weights stay in the mapping)
    const GPT_OSS_MXFP4: &str =
        "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";
    const GEMMA4_12B_QAT: &str =
        "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
    const GEMMA4_26B_A4B_QAT: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf";
    const LFM2_8B_A1B: &str =
        "/home/jeffrey/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf";
    const GRANITE_H_TINY: &str =
        "/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf";
    const GRANITE_H_MICRO: &str =
        "/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-micro-GGUF/granite-4.0-h-micro-Q4_K_M.gguf";
    const GPT_OSS_Q4KM: &str =
        "/home/jeffrey/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";
    const QWEN36_27B: &str =
        "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf";
    const QWEN38_27B_MTP: &str =
        "/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf";

    fn open_model(path: &str) -> Option<(Gguf, Arc<Mmap>)> {
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return None;
        }
        let file = File::open(path).unwrap();
        // SAFETY: model files are read-only for us (same policy as Gguf::open)
        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
        Some((gguf, mmap))
    }

    /// Per-type histogram over all created tensors (deduped by tensor id, so
    /// TENSOR_DUPLICATED aliases count once) — the "type histogram" reported
    /// for each arch.
    fn type_histogram(m: &LlamaModel) -> std::collections::BTreeMap<String, usize> {
        let mut seen = std::collections::HashSet::new();
        let mut hist = std::collections::BTreeMap::new();
        for id in m.tensors.values() {
            if !seen.insert(*id) {
                continue;
            }
            *hist.entry(format!("{:?}", m.ctx.ty(*id))).or_insert(0usize) += 1;
        }
        hist
    }

    /// Strongest cheap consistency check for a loaded model: the model's
    /// tensor-name map is exactly the file's tensor set, and every loaded
    /// tensor carries the file's shape/type (the loader copies them from the
    /// gguf table, so this catches mapping mistakes like wrong dedup).
    fn assert_matches_file(m: &LlamaModel, gguf: &Gguf) {
        assert_eq!(m.tensors.len(), gguf.tensors.len(), "tensor count");
        for ti in &gguf.tensors {
            let id = *m
                .tensors
                .get(&ti.name)
                .unwrap_or_else(|| panic!("gguf tensor '{}' not loaded", ti.name));
            assert_eq!(m.ctx.ne(id), &ti.ne, "{} shape", ti.name);
            assert_eq!(m.ctx.ty(id), ti.ty, "{} type", ti.name);
        }
    }

    // ---- qwen2: full real-model load -------------------------------------

    #[test]
    fn qwen25_real_model_tensor_map() {
        // tensor mapping / shape / type table for qwen2.5-0.5b-instruct
        // Q4_K_M, cross-checked against the pinned gguf-py reader for this
        // exact file (291 tensors: 24 layers x 12 + 3).
        let Some((gguf, mmap)) = open_model(QWEN25) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("qwen2.5 load_model");

        assert_eq!(m.arch, LlmArch::QWEN2);
        assert_eq!(m.layers.len(), 24);
        assert_eq!(m.hparams.n_layer(), 24);

        // input/output tensors
        assert_eq!(m.ctx.ty(m.tok_embd), GgmlType::Q5_0);
        assert_eq!(m.ctx.ne(m.tok_embd), &[896, 151936, 1, 1]);
        assert_eq!(m.ctx.ty(m.output_norm), GgmlType::F32);
        assert_eq!(m.ctx.ne(m.output_norm), &[896, 1, 1, 1]);
        // this file HAS output.weight (Q8_0): not the tok_embd alias
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q8_0);
        assert_ne!(m.output, m.tok_embd);
        assert!(m.output_b.is_none());
        assert!(m.cls_out.is_none());

        // every layer: full 9-tensor set + F32 q/k/v biases (this file has them)
        for (i, l) in m.layers.iter().enumerate() {
            let get = |id: Option<TensorId>, what: &str| {
                id.unwrap_or_else(|| panic!("layer {i} missing {what}"))
            };
            let attn_norm = get(l.attn_norm, "attn_norm");
            // norm weights stay F32 exactly as stored (no conversion — see docs)
            assert_eq!(
                m.ctx.ty(attn_norm),
                GgmlType::F32,
                "layer {i} attn_norm type"
            );
            assert_eq!(m.ctx.ne(attn_norm), &[896, 1, 1, 1]);

            let wq = get(l.wq, "wq");
            assert_eq!(m.ctx.ne(wq), &[896, 896, 1, 1]);
            let wk = get(l.wk, "wk");
            assert_eq!(m.ctx.ne(wk), &[896, 128, 1, 1], "layer {i} wk (GQA 7:1)");
            let wv = get(l.wv, "wv");
            assert_eq!(m.ctx.ne(wv), &[896, 128, 1, 1]);
            let wo = get(l.wo, "wo");
            assert_eq!(m.ctx.ne(wo), &[896, 896, 1, 1]);

            // qwen2 GQA: n_head_kv=2, n_embd_k_gqa = n_embd_head_k*n_head_kv = 128
            assert_eq!(m.n_embd_k_gqa(i), 128);
            assert_eq!(m.n_embd_v_gqa(i), 128);
            assert_eq!(m.n_head_kv(i), 2);
            assert_eq!(m.n_gqa(i), 7);

            // biases present in this file (F32)
            assert_eq!(m.ctx.ty(get(l.wq_b, "wq_b")), GgmlType::F32);
            assert_eq!(m.ctx.ne(get(l.wq_b, "wq_b")), &[896, 1, 1, 1]);
            assert_eq!(m.ctx.ne(get(l.wk_b, "wk_b")), &[128, 1, 1, 1]);
            assert_eq!(m.ctx.ne(get(l.wv_b, "wv_b")), &[128, 1, 1, 1]);
            assert!(l.wqkv.is_none() && l.wqkv_b.is_none());
            assert!(l.wo_b.is_none());

            let ffn_norm = get(l.ffn_norm, "ffn_norm");
            assert_eq!(m.ctx.ty(ffn_norm), GgmlType::F32);
            assert_eq!(m.ctx.ne(get(l.ffn_gate, "ffn_gate")), &[896, 4864, 1, 1]);
            assert_eq!(m.ctx.ne(get(l.ffn_down, "ffn_down")), &[4864, 896, 1, 1]);
            assert_eq!(m.ctx.ne(get(l.ffn_up, "ffn_up")), &[896, 4864, 1, 1]);
        }

        // quant mix spot checks (types read from the file with gguf-py):
        // attn_v is Q8_0 in layers {0,1,3,6,7,8,9,10,13,16,19,21}, else Q5_0
        assert_eq!(m.ctx.ty(m.layers[0].wv.unwrap()), GgmlType::Q8_0);
        assert_eq!(m.ctx.ty(m.layers[1].wv.unwrap()), GgmlType::Q8_0);
        assert_eq!(m.ctx.ty(m.layers[2].wv.unwrap()), GgmlType::Q5_0);
        assert_eq!(m.ctx.ty(m.layers[23].wv.unwrap()), GgmlType::Q5_0);
        // blk.0.ffn_down Q6_K / blk.23.ffn_down Q4_K
        assert_eq!(m.ctx.ty(m.layers[0].ffn_down.unwrap()), GgmlType::Q6K);
        assert_eq!(m.ctx.ty(m.layers[23].ffn_down.unwrap()), GgmlType::Q4K);

        // all 291 declared tensors consumed and addressable by name
        assert_eq!(m.tensors.len(), 291);
        assert_eq!(gguf.tensors.len(), 291);
        assert_eq!(
            m.tensor("blk.23.attn_v.weight"),
            Some(m.layers[23].wv.unwrap())
        );
        assert_eq!(m.tensor("token_embd.weight"), Some(m.tok_embd));
        assert_eq!(m.tensor("output.weight"), Some(m.output));
        assert!(m.tensor("blk.24.attn_v.weight").is_none());
    }

    #[test]
    fn qwen25_mmap_storage_pointer() {
        // ctx.data_bytes must alias the exact mmap bytes: first 18 bytes of
        // token_embd equal both gguf.tensor_data and the reference bytes read
        // with the pinned gguf-py reader for this exact file.
        let Some((gguf, mmap)) = open_model(QWEN25) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("load_model");

        let via_ctx = m
            .ctx
            .data_bytes(m.tok_embd)
            .expect("external storage readable");
        let via_gguf = gguf.tensor_data("token_embd.weight").unwrap();
        assert_eq!(via_ctx.len(), via_gguf.len());
        assert_eq!(&via_ctx[..18], &via_gguf[..18]);

        // reference: pinned gguf-py reader, first 18 bytes of this tensor
        const REF18: [u8; 18] = [
            0x38, 0x99, 0xf9, 0xee, 0x11, 0x8c, 0x84, 0x90, 0xfc, 0xf0, 0x5b, 0xe1, 0xc0, 0x68,
            0x75, 0xa3, 0x93, 0x56,
        ];
        assert_eq!(&via_ctx[..18], &REF18[..]);

        // spot-check one other tensor's mmap alias too (Q6_K ffn_down block 0)
        let down = m.layers[0].ffn_down.unwrap();
        assert_eq!(
            &m.ctx.data_bytes(down).unwrap()[..18],
            &gguf.tensor_data("blk.0.ffn_down.weight").unwrap()[..18]
        );
    }

    #[test]
    fn qwen25_q5_0_values_bit_exact() {
        // Dequantize token_embd block 0 (Q5_0) through our quants and compare
        // bit-exactly against the python reference decode (pinned gguf-py read
        // + struct unpack of d/qh/qs + the ggml Q5_0 algorithm) of this file.
        let Some((gguf, mmap)) = open_model(QWEN25) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("load_model");

        let bytes = m.ctx.data_bytes(m.tok_embd).unwrap();
        let mut y = [0f32; 32]; // one Q5_0 block = 32 values
        ggml::quants::dequantize_row(GgmlType::Q5_0, &bytes[..22], &mut y);

        // python reference output, printed as float bit patterns (0x...);
        // layout per ggml-quants.c: y[j] = x0*d for the first 16 values
        // (low nibbles), y[j+16] = x1*d for the last 16 (high nibbles)
        let ref_bits: [u32; 32] = [
            0xbc270000, 0x3d270000, 0x3c270000, 0x80000000, 0xbce5a000, 0xbb270000, 0x80000000,
            0xbca70000, 0x3ce5a000, 0xbbfa8000, 0xbbfa8000, 0xbc7a8000, 0x3c7a8000, 0x80000000,
            0xbce5a000, 0xbb270000, // ---- x1 half ----
            0xbca70000, 0x3c922000, 0x3b270000, 0x3b270000, 0xbc50c000, 0x3ba70000, 0x3c270000,
            0x3cd0c000, 0x3cbbe000, 0x3c7a8000, 0xbcbbe000, 0xbc50c000, 0x3c922000, 0x3cd0c000,
            0x3bfa8000, 0xbb270000,
        ];
        for (i, (&got, &want)) in y.iter().zip(ref_bits.iter()).enumerate() {
            assert_eq!(got.to_bits(), want, "block0 value {i}");
        }
    }

    // ---- qwen3: tied output + q/k norms ----------------------------------

    #[test]
    fn qwen3_embedding_real_model() {
        // Qwen3-Embedding-0.6B Q8_0: qwen3 arch, no output.weight → output
        // aliases tok_embd (TENSOR_DUPLICATED); attn_q_norm/attn_k_norm.
        let Some((gguf, mmap)) = open_model(QWEN3_EMB) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("qwen3 load_model");

        assert_eq!(m.arch, LlmArch::QWEN3);
        assert_eq!(m.layers.len(), 28);
        // tied embeddings
        assert_eq!(m.output, m.tok_embd);
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q8_0);
        assert_eq!(m.ctx.ne(m.tok_embd), &[1024, 151669, 1, 1]);
        assert!(m.cls_out.is_none());

        for (i, l) in m.layers.iter().enumerate() {
            assert_eq!(m.ctx.ne(l.wq.unwrap()), &[1024, 2048, 1, 1], "layer {i} wq");
            assert_eq!(m.ctx.ne(l.wk.unwrap()), &[1024, 1024, 1, 1], "layer {i} wk");
            assert_eq!(m.ctx.ne(l.wv.unwrap()), &[1024, 1024, 1, 1], "layer {i} wv");
            assert_eq!(m.ctx.ne(l.wo.unwrap()), &[2048, 1024, 1, 1], "layer {i} wo");
            assert_eq!(m.ctx.ne(l.attn_q_norm.unwrap()), &[128, 1, 1, 1]);
            assert_eq!(m.ctx.ne(l.attn_k_norm.unwrap()), &[128, 1, 1, 1]);
        }
        assert_eq!(m.n_embd_k_gqa(0), 1024); // 8 kv heads * 128
                                             // 28 layers * 11 tensors + tok_embd + output_norm = 310, all consumed
        assert_eq!(m.tensors.len(), 310);
        assert_eq!(gguf.tensors.len(), 310);
    }

    // ---- phi3: fused qkv + duplicated rope factors -----------------------

    #[test]
    fn phi4_mini_real_model() {
        // Phi-4-mini (phi3 arch): fused attn_qkv, ffn_up packs 2*n_ff, and
        // rope_factors_long/short stored ONCE (no blk prefix) — layers > 0
        // reuse layer 0's tensor via TENSOR_DUPLICATED.
        let Some((gguf, mmap)) = open_model(PHI4_MINI) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("phi3 load_model");

        assert_eq!(m.arch, LlmArch::PHI3);
        assert_eq!(m.layers.len(), 32);

        // tied output (no output.weight in the file)
        assert_eq!(m.output, m.tok_embd);

        let l0 = &m.layers[0];
        let wqkv = l0.wqkv.expect("fused wqkv");
        // n_embd_qkv = n_embd + n_embd_k_gqa + n_embd_v_gqa = 3072+1024+1024
        assert_eq!(m.ctx.ne(wqkv), &[3072, 5120, 1, 1]);
        assert!(l0.wq.is_none() && l0.wk.is_none() && l0.wv.is_none());
        // n_rot = 96 (partial rotary); ffn_up packs 2*n_ff
        assert_eq!(m.n_rot(0), 96);
        assert_eq!(m.ctx.ne(l0.ffn_up.unwrap()), &[3072, 2 * 8192, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[8192, 3072, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wo.unwrap()), &[3072, 3072, 1, 1]);

        // rope factors: same tensor id for every layer, counted once
        let long0 = m.layers[0].rope_long.expect("rope_long");
        let short0 = m.layers[0].rope_short.expect("rope_short");
        assert_eq!(m.ctx.ne(long0), &[48, 1, 1, 1]); // n_rot/2 = 48
        assert_eq!(m.ctx.ne(short0), &[48, 1, 1, 1]);
        for l in &m.layers {
            assert_eq!(l.rope_long, Some(long0), "rope_long dedup");
            assert_eq!(l.rope_short, Some(short0), "rope_short dedup");
        }
        // 32*(attn_norm+qkv+attn_out+ffn_norm+ffn_down+ffn_up) + token_embd +
        // output_norm + rope_long + rope_short = 196, all consumed exactly once
        assert_eq!(m.tensors.len(), 196);
        assert_eq!(gguf.tensors.len(), 196);
    }

    // ---- llama: synthetic GGUF (no local llama/mistral model file) --------

    /// Minimal GGUF writer: kv pairs + F32 tensors with a real data section
    /// (32-byte aligned offsets, like real writers). `label` keeps the temp
    /// filename unique per test (tests run in parallel).
    fn write_test_gguf(
        label: &str,
        kvs: Vec<(&str, Value)>,
        tensors: &[(String, [i64; 2])], // (name, [ne0, ne1]) — all F32
    ) -> std::path::PathBuf {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        for (k, v) in &kvs {
            put_str(&mut b, k);
            let (ty, bytes): (u32, Vec<u8>) = match v {
                Value::String(s) => (8, {
                    let mut x = (s.len() as u64).to_le_bytes().to_vec();
                    x.extend_from_slice(s.as_bytes());
                    x
                }),
                Value::U32(x) => (4, x.to_le_bytes().to_vec()),
                Value::F32(x) => (6, x.to_le_bytes().to_vec()),
                Value::Array(t, items) => {
                    let et = match t {
                        GgufType::String => 8u32,
                        GgufType::Uint32 => 4,
                        GgufType::Float32 => 6,
                        // gemma4 sliding_window_pattern is a bool array
                        GgufType::Bool => 7,
                        other => panic!("unsupported test elem type {other:?}"),
                    };
                    let mut x = et.to_le_bytes().to_vec();
                    x.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for it in items {
                        match it {
                            Value::String(s) => {
                                x.extend_from_slice(&(s.len() as u64).to_le_bytes());
                                x.extend_from_slice(s.as_bytes());
                            }
                            Value::U32(v) => x.extend_from_slice(&v.to_le_bytes()),
                            Value::F32(v) => x.extend_from_slice(&v.to_le_bytes()),
                            Value::Bool(v) => x.push(u8::from(*v)),
                            other => panic!("unsupported test elem {other:?}"),
                        }
                    }
                    (9, x)
                }
                other => panic!("unsupported test kv {other:?}"),
            };
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&bytes);
        }
        // tensor infos
        let mut offs = Vec::with_capacity(tensors.len());
        let mut cursor = 0u64;
        for (_, ne) in tensors {
            let len = ne[0] * ne[1] * 4;
            let aligned = (cursor + 31) / 32 * 32;
            offs.push(aligned);
            cursor = aligned + len as u64;
        }
        for ((name, ne), &off) in tensors.iter().zip(&offs) {
            put_str(&mut b, name);
            b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
            b.extend_from_slice(&ne[0].to_le_bytes());
            b.extend_from_slice(&ne[1].to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
            b.extend_from_slice(&off.to_le_bytes());
        }
        // pad header to alignment, then the data section
        let data_start = ((b.len() + 31) / 32 * 32) as u64;
        b.resize(data_start as usize, 0);
        for ((name, ne), &off) in tensors.iter().zip(&offs) {
            let start = data_start as usize + off as usize;
            let len = (ne[0] * ne[1] * 4) as usize;
            b.resize(start + len, 0);
            if name == "token_embd.weight" {
                // one recognizable f32 pattern in the first row
                b[start..start + 4].copy_from_slice(&0.25f32.to_le_bytes());
            }
        }
        let dir = std::env::temp_dir().join("llama_rust_model_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("llama-synth-{label}-{}.gguf", tensors.len()));
        std::fs::write(&path, b).unwrap();
        path
    }

    /// Tiny 2-layer llama: n_embd=64, n_head=4, n_head_kv=4 (→ n_embd_head_k
    /// = 16, n_rot = 16), n_ff=128, vocab=32 (tokenizer.ggml.tokens length).
    fn llama_synth_kvs(scaling: &str) -> Vec<(&'static str, Value)> {
        vec![
            ("general.architecture", Value::String("llama".into())),
            ("llama.context_length", Value::U32(512)),
            ("llama.embedding_length", Value::U32(64)),
            ("llama.block_count", Value::U32(2)),
            ("llama.attention.head_count", Value::U32(4)),
            ("llama.attention.head_count_kv", Value::U32(4)),
            ("llama.feed_forward_length", Value::U32(128)),
            ("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("llama.rope.freq_base", Value::F32(10000.0)),
            ("llama.rope.scaling.type", Value::String(scaling.into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(
                    GgufType::String,
                    (0..32).map(|i| Value::String(format!("t{i}"))).collect(),
                ),
            ),
        ]
    }

    fn llama_synth_tensors(with_rope: bool) -> Vec<(String, [i64; 2])> {
        let mut t: Vec<(String, [i64; 2])> = vec![
            ("token_embd.weight".into(), [64, 32]),
            ("output_norm.weight".into(), [64, 1]),
        ];
        for i in 0..2 {
            let p = format!("blk.{i}.");
            t.push((format!("{p}attn_norm.weight"), [64, 1]));
            t.push((format!("{p}attn_q.weight"), [64, 64]));
            t.push((format!("{p}attn_k.weight"), [64, 64]));
            t.push((format!("{p}attn_v.weight"), [64, 64]));
            t.push((format!("{p}attn_output.weight"), [64, 64]));
            t.push((format!("{p}ffn_norm.weight"), [64, 1]));
            t.push((format!("{p}ffn_gate.weight"), [64, 128]));
            t.push((format!("{p}ffn_down.weight"), [128, 64]));
            t.push((format!("{p}ffn_up.weight"), [64, 128]));
        }
        if with_rope {
            // llama.cpp stores these once, no blk prefix (see phi3 note)
            t.push(("rope_factors_long.weight".into(), [8, 1])); // n_rot/2 = 8
            t.push(("rope_factors_short.weight".into(), [8, 1]));
        }
        t
    }

    fn gguf_mmap(path: &Path) -> Arc<Mmap> {
        let file = File::open(path).unwrap();
        Arc::new(unsafe { Mmap::map(&file).unwrap() })
    }

    #[test]
    fn llama_synthetic_longrope_tied_output() {
        let kvs = llama_synth_kvs("longrope");
        let tensors = llama_synth_tensors(true);
        let path = write_test_gguf("longrope", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let m = load_model(&gguf, gguf_mmap(&path)).expect("llama synth load");

        assert_eq!(m.arch, LlmArch::LLAMA);
        assert_eq!(
            m.hparams.rope_scaling_type_train,
            LlamaRopeScalingType::LONGROPE
        );
        assert_eq!(m.layers.len(), 2);

        // tied output: no output.weight in the file → alias of token_embd
        assert_eq!(m.output, m.tok_embd);
        assert_eq!(m.ctx.ne(m.tok_embd), &[64, 32, 1, 1]);

        // mmap alias: token_embd row 0 starts with our 0.25 pattern
        assert_eq!(m.ctx.f32s(m.tok_embd).unwrap()[0], 0.25);

        for (i, l) in m.layers.iter().enumerate() {
            // llama shapes: wq {n_embd, n_embd_head_k*n_head}
            assert_eq!(m.ctx.ne(l.wq.unwrap()), &[64, 64, 1, 1]);
            assert_eq!(m.ctx.ne(l.wk.unwrap()), &[64, 64, 1, 1]);
            assert_eq!(m.ctx.ne(l.wv.unwrap()), &[64, 64, 1, 1]);
            assert_eq!(m.ctx.ne(l.wo.unwrap()), &[64, 64, 1, 1]);
            assert_eq!(m.ctx.ne(l.ffn_gate.unwrap()), &[64, 128, 1, 1]);
            // longrope: rope_long/short dedup to layer 0's tensors
            assert_eq!(l.rope_long, m.layers[0].rope_long, "layer {i}");
            assert_eq!(l.rope_short, m.layers[0].rope_short, "layer {i}");
            assert!(l.rope_freqs.is_none());
            // absent optionals stay None
            assert!(l.wo_b.is_none());
            assert!(l.ffn_gate_b.is_none() && l.ffn_down_b.is_none() && l.ffn_up_b.is_none());
            assert!(l.ffn_gate_inp.is_none());
        }
        assert_eq!(m.ctx.ne(m.layers[0].rope_long.unwrap()), &[8, 1, 1, 1]); // n_rot/2

        // 2*(9 tensors) + token_embd + output_norm + rope_long + rope_short = 22
        assert_eq!(m.tensors.len(), 22);
        assert_eq!(gguf.tensors.len(), 22);
    }

    #[test]
    fn llama_synthetic_default_rope_freqs() {
        // rope.scaling.type = linear → ROPE_FREQS branch (absent: all None)
        let kvs = llama_synth_kvs("linear");
        let tensors = llama_synth_tensors(false);
        let path = write_test_gguf("rope-freqs", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let m = load_model(&gguf, gguf_mmap(&path)).expect("llama synth load");
        for l in &m.layers {
            assert!(l.rope_freqs.is_none());
            assert!(l.rope_long.is_none() && l.rope_short.is_none());
        }
        assert_eq!(m.tensors.len(), 20);
    }

    #[test]
    fn error_paths_shape_missing_unconsumed() {
        // 1) missing required tensor
        let kvs = llama_synth_kvs("linear");
        let mut tensors = llama_synth_tensors(false);
        tensors.retain(|(n, _)| n != "blk.0.ffn_up.weight");
        let path = write_test_gguf("missing", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains("tensor 'blk.0.ffn_up.weight' not found"),
            "{err}"
        );

        // 2) wrong shape
        let kvs = llama_synth_kvs("linear");
        let mut tensors = llama_synth_tensors(false);
        for (n, ne) in tensors.iter_mut() {
            if n == "blk.1.ffn_gate.weight" {
                *ne = [64, 127]; // expected {64, 128}
            }
        }
        let path = write_test_gguf("shape", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains(
                "tensor 'blk.1.ffn_gate.weight' has wrong shape; expected [64, 128], got [64, 127, 1, 1]"
            ),
            "{err}"
        );

        // 3) extra unconsumed tensor → wrong number of tensors
        let kvs = llama_synth_kvs("linear");
        let mut tensors = llama_synth_tensors(false);
        tensors.push(("blk.0.attn_norm_b.weight".into(), [64, 1])); // never requested by llama
        let path = write_test_gguf("unconsumed", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains("wrong number of tensors; expected 21, got 20"),
            "{err}"
        );

        // 4) BERT is a ported arch now: the vocab resolution runs first (as in
        //    C, `LLAMA_LOAD_LOCALS` needs `vocab.n_tokens()`), so a file with
        //    neither a token list nor `bert.vocab_size` fails there
        let bert_kvs = |vocab_size: bool, token_types: bool| -> Vec<(&'static str, Value)> {
            let mut kvs: Vec<(&'static str, Value)> = vec![
                ("general.architecture", Value::String("bert".into())),
                ("bert.context_length", Value::U32(512)),
                ("bert.embedding_length", Value::U32(64)),
                ("bert.block_count", Value::U32(2)),
                ("bert.attention.head_count", Value::U32(4)),
                ("bert.attention.layer_norm_epsilon", Value::F32(1e-5)),
            ];
            if vocab_size {
                kvs.push(("bert.vocab_size", Value::U32(32)));
            }
            if token_types {
                kvs.push(("tokenizer.ggml.token_type_count", Value::U32(1)));
            }
            kvs
        };
        let path = write_test_gguf("bert", bert_kvs(false, false), &[]);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains("cannot determine vocab size for arch 'bert'"),
            "{err}"
        );

        // 4b) vocab present, token-type count missing → the BERT loader's own
        //     guard (models/bert.cpp:26-28)
        let path = write_test_gguf("bert_tt", bert_kvs(true, false), &[]);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains("bert model needs to define token type count"),
            "{err}"
        );

        // 4c) both present → the first missing required tensor is reported
        let path = write_test_gguf("bert_ok", bert_kvs(true, true), &[]);
        let gguf = Gguf::open(&path).unwrap();
        let err = load_model(&gguf, gguf_mmap(&path)).unwrap_err();
        assert!(
            err.contains("tensor 'token_embd.weight' not found"),
            "{err}"
        );
    }

    #[test]
    fn llama_name_mapping_unit() {
        // no local llama/mistral GGUF: verify the consumed name set of the
        // llama arch maps 1:1 to LLM_TENSOR names (spot table, weight + bias)
        for (t, suffix, blk, expect) in [
            (LlmTensor::TOKEN_EMBD, "weight", -1, "token_embd.weight"),
            (LlmTensor::OUTPUT, "weight", -1, "output.weight"),
            (LlmTensor::OUTPUT_NORM, "weight", -1, "output_norm.weight"),
            (LlmTensor::ATTN_NORM, "weight", 7, "blk.7.attn_norm.weight"),
            (LlmTensor::ATTN_Q, "weight", 7, "blk.7.attn_q.weight"),
            (LlmTensor::ATTN_K, "bias", 7, "blk.7.attn_k.bias"),
            (LlmTensor::ATTN_V, "weight", 7, "blk.7.attn_v.weight"),
            (LlmTensor::ATTN_OUT, "weight", 7, "blk.7.attn_output.weight"),
            (LlmTensor::ATTN_OUT, "bias", 7, "blk.7.attn_output.bias"),
            (LlmTensor::FFN_NORM, "weight", 7, "blk.7.ffn_norm.weight"),
            (LlmTensor::FFN_GATE, "weight", 7, "blk.7.ffn_gate.weight"),
            (LlmTensor::FFN_GATE, "bias", 7, "blk.7.ffn_gate.bias"),
            (LlmTensor::FFN_DOWN, "weight", 7, "blk.7.ffn_down.weight"),
            (LlmTensor::FFN_UP, "weight", 7, "blk.7.ffn_up.weight"),
            (
                LlmTensor::ROPE_FACTORS_LONG,
                "weight",
                0,
                "rope_factors_long.weight",
            ),
            (
                LlmTensor::ROPE_FACTORS_SHORT,
                "weight",
                0,
                "rope_factors_short.weight",
            ),
            (LlmTensor::ROPE_FREQS, "weight", 0, "rope_freqs.weight"),
            (
                LlmTensor::FFN_GATE_EXPS,
                "weight",
                7,
                "blk.7.ffn_gate_exps.weight",
            ),
            (
                LlmTensor::FFN_GATE_SHEXP,
                "weight",
                7,
                "blk.7.ffn_gate_shexp.weight",
            ),
        ] {
            assert_eq!(tensor_name_suffix(t, suffix, blk, -1), expect);
        }

        // tensor() lookup resolves names produced this way
        let kvs = llama_synth_kvs("longrope");
        let path = write_test_gguf("namemap", kvs, &llama_synth_tensors(true));
        let gguf = Gguf::open(&path).unwrap();
        let m = load_model(&gguf, gguf_mmap(&path)).unwrap();
        assert_eq!(
            m.tensor(&tensor_name_suffix(LlmTensor::FFN_DOWN, "weight", 1, -1)),
            m.layers[1].ffn_down
        );
        assert!(m.tensor("blk.9.attn_v.weight").is_none());
    }

    // ---- gpt-oss (openai-moe): MXFP4 MoE, attn sinks, expert biases ------

    #[test]
    fn gpt_oss_20b_real_model_tensor_map() {
        // tensor mapping / shape / type table for gpt-oss-20b MXFP4,
        // cross-checked against the pinned gguf-py reader for this exact file:
        // 459 tensors = 24 layers x 19 + 3 (token_embd, output_norm, output),
        // types: Q8_0 x98, F32 x289, MXFP4 x72 (one MXFP4 tensor per layer
        // and expert block: gate/up/down exps).
        let Some((gguf, mmap)) = open_model(GPT_OSS_MXFP4) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gpt-oss load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::OPENAI_MOE);
        assert_eq!(m.layers.len(), 24);
        assert_eq!(m.hparams.n_layer(), 24);

        // input/output
        assert_eq!(m.ctx.ty(m.tok_embd), GgmlType::Q8_0);
        assert_eq!(m.ctx.ne(m.tok_embd), &[2880, 201088, 1, 1]);
        assert_eq!(m.ctx.ty(m.output_norm), GgmlType::F32);
        // this file has an untied Q8_0 output
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q8_0);
        assert_ne!(m.output, m.tok_embd);
        assert!(m.output_b.is_none() && m.cls_out.is_none());

        let l0 = &m.layers[0];
        // q/k/v dims are n_head*n_rot / n_head_kv*n_rot (rot == key_length = 64)
        assert_eq!(m.hparams.n_rot(0), 64);
        assert_eq!(m.ctx.ne(l0.wq.unwrap()), &[2880, 4096, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wk.unwrap()), &[2880, 512, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wv.unwrap()), &[2880, 512, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wo.unwrap()), &[4096, 2880, 1, 1]);
        assert!(l0.wqkv.is_none()); // no fused qkv in this file
                                    // attention sinks: one logit per head
        assert_eq!(m.ctx.ty(l0.attn_sinks.unwrap()), GgmlType::F32);
        assert_eq!(m.ctx.ne(l0.attn_sinks.unwrap()), &[64, 1, 1, 1]);
        // both norms of the layer
        assert_eq!(m.ctx.ne(l0.attn_norm.unwrap()), &[2880, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_post_norm.unwrap()), &[2880, 1, 1, 1]);

        // MoE: 32 experts, 4 used, 2880-wide expert FFN (MXFP4 3D exps)
        assert_eq!(m.hparams.n_expert, 32);
        assert_eq!(m.hparams.n_expert_used_max(), 4);
        assert_eq!(m.hparams.n_ff_exp(0), 2880);
        assert_eq!(m.ctx.ty(l0.ffn_gate_exps.unwrap()), GgmlType::Mxfp4);
        assert_eq!(m.ctx.ne(l0.ffn_gate_exps.unwrap()), &[2880, 2880, 32, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down_exps.unwrap()), &[2880, 2880, 32, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up_exps.unwrap()), &[2880, 2880, 32, 1]);
        assert_eq!(m.ctx.ty(l0.ffn_gate_inp.unwrap()), GgmlType::F32);
        assert_eq!(m.ctx.ne(l0.ffn_gate_inp.unwrap()), &[2880, 32, 1, 1]);
        // router bias + expert biases (F32) — gpt-oss specific
        assert_eq!(m.ctx.ne(l0.ffn_gate_inp_b.unwrap()), &[32, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_gate_exps_b.unwrap()), &[2880, 32, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down_exps_b.unwrap()), &[2880, 32, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up_exps_b.unwrap()), &[2880, 32, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wo_b.unwrap()), &[2880, 1, 1, 1]);

        // every layer carries the same set
        for (i, l) in m.layers.iter().enumerate() {
            assert!(l.attn_sinks.is_some(), "layer {i} attn_sinks");
            assert!(l.ffn_gate_exps.is_some(), "layer {i} ffn_gate_exps");
            assert!(l.ffn_gate_exps_b.is_some(), "layer {i} ffn_gate_exps_b");
            assert!(l.ffn_gate_shexp.is_none(), "layer {i} has no shexp");
        }
        assert_eq!(
            m.ctx.ne(m.layers[23].ffn_down_exps.unwrap()),
            &[2880, 2880, 32, 1]
        );

        // full inventory cross-check + type histogram (MXFP4 x72 etc.)
        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&289), "histogram {hist:?}");
        assert_eq!(hist.get("Q8_0"), Some(&98), "histogram {hist:?}");
        assert_eq!(hist.get("Mxfp4"), Some(&72), "histogram {hist:?}");
        eprintln!(
            "gpt-oss-20b MXFP4: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );

        // the MXFP4 storage is reachable through the mmap (dequant ready)
        let exps = m
            .ctx
            .data_bytes(m.layers[0].ffn_gate_exps.unwrap())
            .unwrap();
        assert_eq!(
            exps.len(),
            gguf.tensor_data("blk.0.ffn_gate_exps.weight")
                .unwrap()
                .len()
        );
        assert_eq!(
            &exps[..8],
            &gguf.tensor_data("blk.0.ffn_gate_exps.weight").unwrap()[..8]
        );
    }

    #[test]
    #[ignore = "manual: 11 GB file, full load (mmap) — run with --ignored"]
    fn gpt_oss_20b_full_load() {
        let Some((gguf, mmap)) = open_model(GPT_OSS_MXFP4) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gpt-oss load_model");
        eprintln!(
            "gpt-oss-20b: {} tensors in {:?}",
            m.tensors.len(),
            t0.elapsed()
        );
        assert_eq!(m.tensors.len(), 459);
    }

    // ---- gemma4: per-layer head dims, global rope_freqs, out_scale ---------

    #[test]
    fn gemma4_12b_real_model_tensor_map() {
        // tensor mapping for gemma-4-12B-it-QAT-Q4_0, cross-checked against the
        // pinned gguf-py reader for this exact file: 667 tensors =
        // 48 layers x 13 + 3 (token_embd, output_norm, rope_freqs),
        // types: F32 x338, Q4_0 x328, Q6_K x1.
        // SWA layers (pattern: 5 swa + 1 dense) use head dim 256 with 8 kv
        // heads; dense layers use head dim 512 with 1 kv head.
        let Some((gguf, mmap)) = open_model(GEMMA4_12B_QAT) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gemma4 load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::GEMMA4);
        assert_eq!(m.layers.len(), 48);
        assert_eq!(m.hparams.n_layer(), 48);

        assert_eq!(m.ctx.ty(m.tok_embd), GgmlType::Q6K);
        assert_eq!(m.ctx.ne(m.tok_embd), &[3840, 262144, 1, 1]);
        assert_eq!(m.ctx.ty(m.output_norm), GgmlType::F32);
        // no output.weight in the file -> the tie fallback duplicates
        // token_embd (C++ creates the dup *before* tok_embd itself, so the two
        // are distinct ggml tensors over the same file bytes — exactly like
        // `llama_model::output` vs `llama_model::tok_embd`)
        assert_ne!(m.output, m.tok_embd);
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q6K);
        assert_eq!(
            m.ctx.data_bytes(m.output).unwrap(),
            m.ctx.data_bytes(m.tok_embd).unwrap()
        );
        assert_eq!(m.tensor("token_embd.weight"), Some(m.tok_embd));
        // no per-layer embeddings in this model (embedding_length_per_layer_input = 0)
        assert_eq!(m.hparams.n_embd_per_layer, 0);
        assert!(m.per_layer_tok_embd.is_none());
        assert!(m.per_layer_model_proj.is_none() && m.per_layer_proj_norm.is_none());

        // SWA layer 0
        let l0 = &m.layers[0];
        assert!(m.hparams.is_swa(0));
        assert_eq!(m.ctx.ne(l0.wq.unwrap()), &[3840, 4096, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wk.unwrap()), &[3840, 2048, 1, 1]); // 8 kv heads * 256
        assert_eq!(m.ctx.ne(l0.wv.unwrap()), &[3840, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wo.unwrap()), &[4096, 3840, 1, 1]);
        assert_eq!(m.ctx.ty(l0.attn_q_norm.unwrap()), GgmlType::F32);
        assert_eq!(m.ctx.ne(l0.attn_q_norm.unwrap()), &[256, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_k_norm.unwrap()), &[256, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_norm.unwrap()), &[3840, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_post_norm.unwrap()), &[3840, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_post_norm.unwrap()), &[3840, 1, 1, 1]);
        // per-layer output scale (1 element)
        assert_eq!(m.ctx.ty(l0.out_scale.unwrap()), GgmlType::F32);
        assert_eq!(m.ctx.ne(l0.out_scale.unwrap()), &[1, 1, 1, 1]);
        // SWA layers have no rope_freqs (only full_attention layers do)
        assert!(l0.rope_freqs.is_none());
        // dense FFN (double-wide handled through n_ff_arr)
        assert_eq!(m.ctx.ne(l0.ffn_gate.unwrap()), &[3840, 15360, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[15360, 3840, 1, 1]);
        assert!(l0.ffn_gate_inp.is_none(), "12B has no MoE");
        assert!(l0.wqkv.is_none(), "this file uses separate q/k/v");

        // dense (full attention) layer 5: head dim 512, 1 kv head
        let l5 = &m.layers[5];
        assert!(!m.hparams.is_swa(5));
        assert_eq!(m.ctx.ne(l5.wq.unwrap()), &[3840, 8192, 1, 1]); // 16 * 512
        assert_eq!(m.ctx.ne(l5.wk.unwrap()), &[3840, 512, 1, 1]); // 1 * 512
        assert_eq!(m.ctx.ne(l5.wo.unwrap()), &[8192, 3840, 1, 1]);
        assert_eq!(m.ctx.ne(l5.attn_q_norm.unwrap()), &[512, 1, 1, 1]);
        // rope_freqs is stored once, no blk prefix -> every dense layer reuses it
        let rf = l5.rope_freqs.expect("dense layer rope_freqs");
        assert_eq!(m.ctx.ne(rf), &[256, 1, 1, 1]); // n_embd_head(512) / 2
        for (i, l) in m.layers.iter().enumerate() {
            if m.hparams.is_swa(i) {
                assert!(l.rope_freqs.is_none(), "layer {i}");
            } else {
                assert_eq!(l.rope_freqs, Some(rf), "layer {i} rope_freqs dedup");
            }
        }

        // full inventory cross-check + histogram
        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&338), "histogram {hist:?}");
        assert_eq!(hist.get("Q4_0"), Some(&328), "histogram {hist:?}");
        assert_eq!(hist.get("Q6K"), Some(&1), "histogram {hist:?}");
        eprintln!(
            "gemma4-12B-QAT: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
    }

    #[test]
    fn gemma4_26b_a4b_moe_tensor_map() {
        // MoE gemma4 variant: ffn_gate_up_exps (combined gate+up) instead of
        // separate gate/up, router scale (ffn_gate_inp.scale, explicit) and
        // per-expert scale (ffn_down_exps.scale, generic pass) — plus the MoE
        // branch norms pre_ffw_norm_2 / post_ffw_norm_1 / post_ffw_norm_2.
        // Cross-checked with the pinned gguf-py reader: 658 tensors.
        let Some((gguf, mmap)) = open_model(GEMMA4_26B_A4B_QAT) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gemma4 26B-A4B load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::GEMMA4);
        assert_eq!(m.layers.len(), 30);
        assert_eq!(m.hparams.n_layer(), 30);
        assert_eq!(m.hparams.n_expert, 128);
        assert_eq!(m.hparams.n_ff_exp(0), 704); // per-layer expert ff
        assert_eq!(m.ctx.ne(m.tok_embd), &[2816, 262144, 1, 1]);
        // tied output (no output.weight in this file either): distinct dup
        // tensor over the same bytes, see the 12B test note
        assert_ne!(m.output, m.tok_embd);
        assert_eq!(
            m.ctx.data_bytes(m.output).unwrap(),
            m.ctx.data_bytes(m.tok_embd).unwrap()
        );

        let l0 = &m.layers[0];
        // combined gate+up: {n_embd, 2*n_ff_exp, n_expert}
        assert_eq!(
            m.ctx.ne(l0.ffn_gate_up_exps.unwrap()),
            &[2816, 1408, 128, 1]
        );
        assert!(l0.ffn_gate_exps.is_none(), "combined gate+up file");
        assert!(l0.ffn_up_exps.is_none());
        assert_eq!(m.ctx.ne(l0.ffn_down_exps.unwrap()), &[704, 2816, 128, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_gate_inp.unwrap()), &[2816, 128, 1, 1]);
        // router scale {n_embd} + per-expert down scale {n_expert}
        assert_eq!(m.ctx.ne(l0.ffn_gate_inp_s.unwrap()), &[2816, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down_exps_s.unwrap()), &[128, 1, 1, 1]);
        assert!(l0.ffn_gate_exps_s.is_none() && l0.ffn_up_exps_s.is_none());
        // MoE branch norms
        assert_eq!(m.ctx.ne(l0.ffn_pre_norm_2.unwrap()), &[2816, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_post_norm_1.unwrap()), &[2816, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_post_norm_2.unwrap()), &[2816, 1, 1, 1]);
        // dense shared-expert FFN still present
        assert_eq!(m.ctx.ne(l0.ffn_gate.unwrap()), &[2816, 2112, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[2112, 2816, 1, 1]);

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&392), "histogram {hist:?}");
        assert_eq!(hist.get("Q4_0"), Some(&265), "histogram {hist:?}");
        eprintln!(
            "gemma4-26B-A4B-QAT: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
    }

    #[test]
    #[ignore = "manual: 6.5 GB gemma4, full load + all tensors"]
    fn gemma4_12b_full_load() {
        let Some((gguf, mmap)) = open_model(GEMMA4_12B_QAT) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gemma4 load_model");
        eprintln!(
            "gemma4-12B: {} tensors in {:?}",
            m.tensors.len(),
            t0.elapsed()
        );
        assert_eq!(m.tensors.len(), 667);
    }

    // ---- lfm2moe: hybrid shortconv/MoE -----------------------------------

    #[test]
    fn lfm2moe_8b_a1b_real_model_tensor_map() {
        // tensor mapping for LFM2-8B-A1B Q4_K_M, cross-checked against the
        // pinned gguf-py reader for this exact file: 256 tensors,
        // types: Q4_K x118, F32 x123, Q6_K x15.
        // 24 layers: 2 dense-lead (dense FFN + shortconv), then hybrid
        // shortconv blocks and 8-head attention blocks (head_count_kv per
        // layer: 0 => recurrent, 8 => attention), with MoE on every
        // non-dense-lead layer.
        let Some((gguf, mmap)) = open_model(LFM2_8B_A1B) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("lfm2moe load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::LFM2MOE);
        assert_eq!(m.layers.len(), 24);
        assert_eq!(m.hparams.n_layer(), 24);
        assert_eq!(m.hparams.n_layer_dense_lead, 2);
        assert_eq!(m.hparams.n_shortconv_l_cache, 3);

        assert_eq!(m.ctx.ne(m.tok_embd), &[2048, 65536, 1, 1]);
        // LFM2 names the output norm `token_embd_norm`
        assert_eq!(m.ctx.ty(m.output_norm), GgmlType::F32);
        assert_eq!(m.ctx.ne(m.output_norm), &[2048, 1, 1, 1]);
        // no output.weight -> dup fallback over token_embd
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q6K);

        // layer 0: dense lead + recurrent (shortconv) mixer
        let l0 = &m.layers[0];
        assert!(m.hparams.is_recr(0));
        assert_eq!(m.ctx.ne(l0.ffn_gate.unwrap()), &[2048, 7168, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[7168, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up.unwrap()), &[2048, 7168, 1, 1]);
        assert!(l0.ffn_gate_inp.is_none(), "dense lead layer has no MoE");
        assert_eq!(m.ctx.ne(l0.shortconv_conv.unwrap()), &[3, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l0.shortconv_in_proj.unwrap()), &[2048, 6144, 1, 1]);
        assert_eq!(
            m.ctx.ne(l0.shortconv_out_proj.unwrap()),
            &[2048, 2048, 1, 1]
        );
        assert!(l0.wq.is_none() && l0.wk.is_none() && l0.wv.is_none());

        // layer 2: attention layer (8 kv heads) + MoE
        let l2 = &m.layers[2];
        assert!(!m.hparams.is_recr(2));
        assert_eq!(m.ctx.ne(l2.wq.unwrap()), &[2048, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l2.wk.unwrap()), &[2048, 512, 1, 1]); // 8 * 64
        assert_eq!(m.ctx.ne(l2.wv.unwrap()), &[2048, 512, 1, 1]);
        assert_eq!(m.ctx.ne(l2.wo.unwrap()), &[2048, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l2.attn_q_norm.unwrap()), &[64, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l2.attn_k_norm.unwrap()), &[64, 1, 1, 1]);
        assert!(l2.shortconv_conv.is_none());
        // MoE: 32 experts, 4 used, 1792-wide expert FFN
        assert_eq!(m.hparams.n_expert, 32);
        assert_eq!(m.hparams.n_expert_used_max(), 4);
        assert_eq!(m.ctx.ne(l2.ffn_gate_inp.unwrap()), &[2048, 32, 1, 1]);
        assert_eq!(m.ctx.ne(l2.ffn_gate_exps.unwrap()), &[2048, 1792, 32, 1]);
        assert_eq!(m.ctx.ne(l2.ffn_down_exps.unwrap()), &[1792, 2048, 32, 1]);
        assert_eq!(m.ctx.ne(l2.ffn_up_exps.unwrap()), &[2048, 1792, 32, 1]);
        assert_eq!(m.ctx.ne(l2.ffn_exp_probs_b.unwrap()), &[32, 1, 1, 1]);
        assert!(l2.ffn_gate.is_none(), "MoE layer has no dense ffn gate");

        // ColBERT head absent in this file
        assert!(m.dense_2_out_layers.is_none() && m.dense_2_out_layers_b.is_none());

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&123), "histogram {hist:?}");
        assert_eq!(hist.get("Q4K"), Some(&118), "histogram {hist:?}");
        assert_eq!(hist.get("Q6K"), Some(&15), "histogram {hist:?}");
        eprintln!(
            "LFM2-8B-A1B: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );

        // n_tensors == file count (all consumed, incl. the dup output)
        assert_eq!(m.tensors.len(), 256);
        assert_eq!(gguf.tensors.len(), 256);
    }

    // ---- granitehybrid: mamba2 mixer + MoE + shared expert ----------------

    #[test]
    fn granitehybrid_tiny_real_model_tensor_map() {
        // tensor mapping for granite-4.0-h-tiny Q4_K_M, cross-checked against
        // the pinned gguf-py reader for this exact file: 666 tensors,
        // types: F32 x337, Q4_K x186, Q5_K x80, Q6_K x63.
        // 40 layers: is_recr = (head_count_kv == 0) -> only layers
        // 5/15/25/35 have attention (4 kv heads); the rest are mamba2 with
        // d_inner = 2*n_embd = 3072 (expansion factor 2 assertion).
        let Some((gguf, mmap)) = open_model(GRANITE_H_TINY) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("granitehybrid load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::GRANITE_HYBRID);
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.hparams.n_layer(), 40);
        assert_eq!(m.hparams.ssm_d_conv, 4);
        assert_eq!(m.hparams.ssm_d_inner, 3072);
        assert_eq!(m.hparams.ssm_d_state, 128);
        assert_eq!(m.hparams.ssm_dt_rank, 48);
        assert_eq!(m.hparams.ssm_n_group, 1);
        // granite-4.0-h-tiny: rope.scaling.finetuned = false -> no RoPE
        assert!(!m.hparams.rope_finetuned);
        assert!(!m.hparams.has_rope(0));
        // granite scalars
        assert_eq!(m.hparams.f_logit_scale, 6.0);
        assert_eq!(m.hparams.f_residual_scale, 0.22);
        assert_eq!(m.hparams.f_embedding_scale, 12.0);
        assert_eq!(m.hparams.f_attention_scale, 0.0078125);
        assert_eq!(m.hparams.n_ff_shexp, 1024);

        assert_eq!(m.ctx.ne(m.tok_embd), &[1536, 100352, 1, 1]);
        assert_eq!(m.ctx.ty(m.output_norm), GgmlType::F32);
        // no output.weight -> dup fallback
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q6K);

        // layer 0: mamba2 mixer
        let l0 = &m.layers[0];
        assert!(m.hparams.is_recr(0));
        // d_in_proj = 2*3072 + 2*1*128 + 48 = 6448
        assert_eq!(m.ctx.ne(l0.ssm_in.unwrap()), &[1536, 6448, 1, 1]);
        // conv dim = d_inner + 2*n_group*d_state = 3072 + 256 = 3328
        assert_eq!(m.ctx.ne(l0.ssm_conv1d.unwrap()), &[4, 3328, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_conv1d_b.unwrap()), &[3328, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_dt_b.unwrap()), &[48, 1, 1, 1]);
        // ssm_a / ssm_d carry no `.weight` suffix in the file
        assert_eq!(m.ctx.ne(l0.ssm_a.unwrap()), &[1, 48, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_d.unwrap()), &[1, 48, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_norm.unwrap()), &[3072, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_out.unwrap()), &[3072, 1536, 1, 1]);
        assert!(m.tensor("blk.0.ssm_a").is_some());
        assert!(m.tensor("blk.0.ssm_a.weight").is_none());
        assert!(l0.wq.is_none() && l0.wo.is_none());

        // MoE everywhere + shared expert
        assert_eq!(m.hparams.n_expert, 64);
        assert_eq!(m.ctx.ne(l0.ffn_gate_inp.unwrap()), &[1536, 64, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_gate_exps.unwrap()), &[1536, 512, 64, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down_exps.unwrap()), &[512, 1536, 64, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up_exps.unwrap()), &[1536, 512, 64, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_gate_shexp.unwrap()), &[1536, 1024, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down_shexp.unwrap()), &[1024, 1536, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up_shexp.unwrap()), &[1536, 1024, 1, 1]);
        // rope_freqs absent (rope.scaling.finetuned = false)
        assert!(l0.rope_freqs.is_none());

        // layer 5: attention layer
        let l5 = &m.layers[5];
        assert!(!m.hparams.is_recr(5));
        assert_eq!(m.ctx.ne(l5.wq.unwrap()), &[1536, 1536, 1, 1]); // 12 * 128
        assert_eq!(m.ctx.ne(l5.wk.unwrap()), &[1536, 512, 1, 1]); // 4 * 128
        assert_eq!(m.ctx.ne(l5.wv.unwrap()), &[1536, 512, 1, 1]);
        assert_eq!(m.ctx.ne(l5.wo.unwrap()), &[1536, 1536, 1, 1]);
        assert!(l5.wo_b.is_none());
        assert!(l5.ssm_in.is_none());

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&337), "histogram {hist:?}");
        assert_eq!(hist.get("Q4K"), Some(&186), "histogram {hist:?}");
        assert_eq!(hist.get("Q5K"), Some(&80), "histogram {hist:?}");
        assert_eq!(hist.get("Q6K"), Some(&63), "histogram {hist:?}");
        eprintln!(
            "granite-4.0-h-tiny: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
        assert_eq!(m.tensors.len(), 666);
    }

    // ---- qwen35: gated delta net + full-attention interleave ---------------

    #[test]
    fn qwen35_27b_real_model_tensor_map() {
        // tensor mapping for Qwen3.6-27B Q4_K_M, cross-checked against the
        // pinned gguf-py reader for this exact file: 851 tensors,
        // types: Q4_K x433, F32 x353, Q6_K x65.
        // 64 layers, full_attention_interval = 4 -> layers 3,7,...,63 are
        // full attention, the rest are gated delta net linear attention
        // (ssm_d_state 128, 16 key heads, 48 value heads); no MTP/nextn
        // blocks in this file (block_count == n_layer).
        let Some((gguf, mmap)) = open_model(QWEN36_27B) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("qwen35 load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::QWEN35);
        assert_eq!(m.layers.len(), 64);
        assert_eq!(m.hparams.n_layer(), 64);
        assert_eq!(m.hparams.n_layer_nextn, 0);
        assert_eq!(m.hparams.n_embd, 5120);
        assert_eq!(m.hparams.n_head(0), 24);
        assert_eq!(m.hparams.n_head_kv(0), 4);
        assert_eq!(m.hparams.n_embd_head_k(0), 256);
        assert_eq!(m.hparams.n_rot(0), 64);
        assert_eq!(m.hparams.rope_sections, [11, 11, 10, 0]);
        assert_eq!(m.hparams.ssm_d_conv, 4);
        assert_eq!(m.hparams.ssm_d_state, 128);
        assert_eq!(m.hparams.ssm_n_group, 16);
        assert_eq!(m.hparams.ssm_dt_rank, 48);
        // recurrent pattern derived from full_attention_interval = 4
        assert!(m.hparams.is_recr(0));
        assert!(m.hparams.is_recr(2));
        assert!(!m.hparams.is_recr(3));
        assert!(!m.hparams.is_recr(63));
        assert!(m.hparams.is_recr(62));

        assert_eq!(m.ctx.ne(m.tok_embd), &[5120, 248320, 1, 1]);
        // this file has a separate Q6_K output head
        assert_eq!(m.ctx.ty(m.output), GgmlType::Q6K);
        assert_ne!(m.output, m.tok_embd);
        assert_eq!(m.ctx.ne(m.output), &[5120, 248320, 1, 1]);

        // layer 0: linear attention (gated delta net)
        let l0 = &m.layers[0];
        // key_dim = 128*16 = 2048, value_dim = 128*48 = 6144, conv_dim = 10240
        assert_eq!(m.ctx.ne(l0.wqkv.unwrap()), &[5120, 10240, 1, 1]);
        assert_eq!(m.ctx.ne(l0.wqkv_gate.unwrap()), &[5120, 6144, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_conv1d.unwrap()), &[4, 10240, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_dt_b.unwrap()), &[48, 1, 1, 1]);
        // ssm_a carries no `.weight` suffix, shape {ssm_dt_rank}
        assert_eq!(m.ctx.ne(l0.ssm_a.unwrap()), &[48, 1, 1, 1]);
        assert!(m.tensor("blk.0.ssm_a").is_some());
        assert_eq!(m.ctx.ne(l0.ssm_beta.unwrap()), &[5120, 48, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_alpha.unwrap()), &[5120, 48, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_norm.unwrap()), &[128, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_out.unwrap()), &[6144, 5120, 1, 1]);
        assert!(l0.wq.is_none() && l0.attn_q_norm.is_none());
        assert_eq!(m.ctx.ne(l0.ffn_gate.unwrap()), &[5120, 17408, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[17408, 5120, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up.unwrap()), &[5120, 17408, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_norm.unwrap()), &[5120, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_post_norm.unwrap()), &[5120, 1, 1, 1]);

        // layer 3: full attention (n_embd_head_k * n_head * 2 q dims)
        let l3 = &m.layers[3];
        assert_eq!(m.ctx.ne(l3.wq.unwrap()), &[5120, 12288, 1, 1]);
        assert_eq!(m.ctx.ne(l3.wk.unwrap()), &[5120, 1024, 1, 1]); // 4 * 256
        assert_eq!(m.ctx.ne(l3.wv.unwrap()), &[5120, 1024, 1, 1]);
        assert_eq!(m.ctx.ne(l3.wo.unwrap()), &[6144, 5120, 1, 1]); // 24 * 256
        assert_eq!(m.ctx.ne(l3.attn_q_norm.unwrap()), &[256, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l3.attn_k_norm.unwrap()), &[256, 1, 1, 1]);
        assert!(l3.wqkv.is_none() && l3.wqkv_gate.is_none());
        assert!(l3.ssm_a.is_none() && l3.ssm_out.is_none());

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&353), "histogram {hist:?}");
        assert_eq!(hist.get("Q4K"), Some(&433), "histogram {hist:?}");
        assert_eq!(hist.get("Q6K"), Some(&65), "histogram {hist:?}");
        eprintln!(
            "Qwen3.6-27B: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
        assert_eq!(m.tensors.len(), 851);

        // no nextn tensors were requested on any layer
        assert!(m.layers.iter().all(|l| l.nextn.eh_proj.is_none()));
    }

    #[test]
    fn qwen35_mtp_block_tensor_map() {
        // qwen35 with nextn_predict_layers = 1 (block_count 65): the extra
        // block 64 is an MTP block — a full-attention decoder block plus
        // nextn.eh_proj / enorm / hnorm / shared_head_norm (embed_tokens and
        // shared_head_head are absent -> TENSOR_NOT_REQUIRED).
        // Cross-checked with the pinned gguf-py reader: 866 tensors.
        let Some((gguf, mmap)) = open_model(QWEN38_27B_MTP) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("qwen35 MTP load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::QWEN35);
        assert_eq!(m.hparams.n_layer_all, 65);
        assert_eq!(m.hparams.n_layer_nextn, 1);
        assert_eq!(m.hparams.n_layer(), 64);
        assert_eq!(m.layers.len(), 65); // C++ resizes to n_layer_all

        let mtp = &m.layers[64];
        assert!(!m.hparams.is_recr(64), "MTP layers are attention-only");
        assert_eq!(m.ctx.ne(mtp.wq.unwrap()), &[5120, 12288, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.wk.unwrap()), &[5120, 1024, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.wo.unwrap()), &[6144, 5120, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.attn_q_norm.unwrap()), &[256, 1, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.ffn_gate.unwrap()), &[5120, 17408, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.ffn_down.unwrap()), &[17408, 5120, 1, 1]);
        // NextN-specific tensors
        assert_eq!(m.ctx.ne(mtp.nextn.eh_proj.unwrap()), &[10240, 5120, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.nextn.enorm.unwrap()), &[5120, 1, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.nextn.hnorm.unwrap()), &[5120, 1, 1, 1]);
        assert_eq!(
            m.ctx.ne(mtp.nextn.shared_head_norm.unwrap()),
            &[5120, 1, 1, 1]
        );
        assert!(mtp.nextn.embed_tokens.is_none());
        assert!(mtp.nextn.shared_head_head.is_none());
        // no ssm tensors on the MTP block
        assert!(mtp.ssm_a.is_none() && mtp.wqkv.is_none() && mtp.ssm_out.is_none());
        // trunk block 63 is full attention too (interval pattern)
        assert!(!m.hparams.is_recr(63));

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        eprintln!(
            "Qwen3.8-27B-MTP: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
        assert_eq!(m.tensors.len(), 866);
    }

    // ---- LM Studio sweep regressions (2026-09): bool-array swa pattern ------
    // Olmo-3-32B-Think ships `olmo2.attention.sliding_window_pattern` as
    // array(bool) — the std::array<uint32_t, N> get_arr overload accepts BOOL
    // with `x != 0` widening (llama-model-loader.cpp:371,396-398), which
    // inline_load_swa_pattern must mirror. Before the fix the load died with
    // "key olmo2.attention.sliding_window_pattern has wrong array element type".
    const OLMO3_32B_THINK: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-GGUF/Olmo-3-32B-Think-Q4_K_M.gguf";

    #[test]
    fn olmo3_bool_array_swa_pattern() {
        let Some((gguf, mmap)) = open_model(OLMO3_32B_THINK) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("olmo3 load_model");
        assert_eq!(m.arch, LlmArch::OLMO2);
        assert_eq!(m.hparams.n_layer(), 64);
        assert_eq!(m.hparams.n_swa, 4096);
        // pattern [true,true,true,false]*16: every 4th layer dense
        assert!(m.hparams.is_swa(0));
        assert!(m.hparams.is_swa(2));
        assert!(!m.hparams.is_swa(3), "bool false must widen to dense");
        assert!(m.hparams.is_swa(4));
        assert!(!m.hparams.is_swa(63));
        assert!(m.hparams.is_swa_any());
        assert_matches_file(&m, &gguf);
    }

    // Muse-Glimmer-30B: same bool-array pattern under the muse-glimmer arch
    // key (`muse-glimmer.attention.sliding_window_pattern`, load_swa_pattern
    // (ml, 4) at muse-glimmer.cpp:13 → inline_load_swa_pattern here).
    const MUSE_GLIMMER_30B: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf";

    #[test]
    fn muse_glimmer_bool_array_swa_pattern() {
        let Some((gguf, mmap)) = open_model(MUSE_GLIMMER_30B) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("muse-glimmer load_model");
        assert_eq!(m.arch, LlmArch::MUSE_GLIMMER);
        assert_eq!(m.hparams.n_layer(), 52);
        assert_eq!(m.hparams.n_swa, 2048);
        assert!(m.hparams.is_swa(0));
        assert!(!m.hparams.is_swa(3), "bool false must widen to dense");
        assert!(m.hparams.is_swa(4));
        assert!(!m.hparams.is_swa(51));
        assert_matches_file(&m, &gguf);
    }

    // ---- LM Studio sweep regression (2026-09): ornith qwen35moe nextn ------
    // Ornith-1.5-35B is a qwen35moe file with nextn_predict_layers = 1: the
    // MTP block 40 carries blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}
    // that load_block_mtp must consume (qwen35moe.cpp:131-137) — before the
    // fix done_getting_tensors rejected the file ("wrong number of tensors;
    // expected 753, got 749").
    const ORNITH_15_35B: &str =
        "/home/jeffrey/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";

    #[test]
    fn ornith_qwen35moe_nextn_tensor_map() {
        let Some((gguf, mmap)) = open_model(ORNITH_15_35B) else {
            return;
        };
        let m = load_model(&gguf, mmap).expect("ornith load_model");
        assert_eq!(m.arch, LlmArch::QWEN35MOE);
        assert_eq!(m.hparams.n_layer_all, 41);
        assert_eq!(m.hparams.n_layer_nextn, 1);
        assert_eq!(m.hparams.n_layer(), 40);
        assert_eq!(m.layers.len(), 41); // C++ resizes to n_layer_all

        // the MTP block's NextN tensors are consumed (this is the regression)
        let mtp = &m.layers[40];
        assert_eq!(m.ctx.ne(mtp.nextn.eh_proj.unwrap()), &[4096, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.nextn.enorm.unwrap()), &[2048, 1, 1, 1]);
        assert_eq!(m.ctx.ne(mtp.nextn.hnorm.unwrap()), &[2048, 1, 1, 1]);
        assert_eq!(
            m.ctx.ne(mtp.nextn.shared_head_norm.unwrap()),
            &[2048, 1, 1, 1]
        );
        assert!(mtp.nextn.embed_tokens.is_none());
        assert!(mtp.nextn.shared_head_head.is_none());
        // the trunk blocks never request nextn tensors
        assert!(m.layers[..40].iter().all(|l| l.nextn.eh_proj.is_none()));

        // every declared gguf tensor is consumed exactly once
        assert_matches_file(&m, &gguf);
        assert_eq!(m.tensors.len(), 753);
    }

    #[test]
    #[ignore = "manual: 16 GB qwen35, full load (mmap)"]
    fn qwen35_27b_full_load() {
        let Some((gguf, mmap)) = open_model(QWEN36_27B) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("qwen35 load_model");
        eprintln!(
            "Qwen3.6-27B: {} tensors in {:?}",
            m.tensors.len(),
            t0.elapsed()
        );
        assert_eq!(m.tensors.len(), 851);
    }

    #[test]
    fn granitehybrid_micro_dense_ffn_tensor_map() {
        // granite-4.0-h-micro has expert_count = 0 -> the dense FFN branch of
        // granite-hybrid.cpp (ffn_gate/down/up, no MoE/shexp tensors), a
        // different mamba2 layout (n_embd 2048 => d_inner 4096, dt_rank 64)
        // and no per-layer ffn biases in this file.
        // Cross-checked with the pinned gguf-py reader: 506 tensors.
        let Some((gguf, mmap)) = open_model(GRANITE_H_MICRO) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("granitehybrid micro load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::GRANITE_HYBRID);
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.hparams.n_embd, 2048);
        assert_eq!(m.hparams.n_expert, 0);
        assert_eq!(m.hparams.n_ff(0), 8192);
        assert_eq!(m.hparams.ssm_d_inner, 4096);
        assert_eq!(m.hparams.ssm_dt_rank, 64);

        let l0 = &m.layers[0];
        // d_in_proj = 2*4096 + 2*1*128 + 64 = 8512
        assert_eq!(m.ctx.ne(l0.ssm_in.unwrap()), &[2048, 8512, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_conv1d.unwrap()), &[4, 4352, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_dt_b.unwrap()), &[64, 1, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_a.unwrap()), &[1, 64, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ssm_out.unwrap()), &[4096, 2048, 1, 1]);
        // dense FFN branch
        assert_eq!(m.ctx.ne(l0.ffn_gate.unwrap()), &[2048, 8192, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_down.unwrap()), &[8192, 2048, 1, 1]);
        assert_eq!(m.ctx.ne(l0.ffn_up.unwrap()), &[2048, 8192, 1, 1]);
        assert!(l0.ffn_gate_inp.is_none() && l0.ffn_gate_exps.is_none());
        assert!(l0.ffn_gate_shexp.is_none() && l0.ffn_down_shexp.is_none());
        assert!(l0.ffn_gate_b.is_none() && l0.ffn_down_b.is_none() && l0.ffn_up_b.is_none());
        assert!(l0.rope_freqs.is_none());

        // attention layer 5 (8 kv heads, head dim 64)
        let l5 = &m.layers[5];
        assert_eq!(m.ctx.ne(l5.wq.unwrap()), &[2048, 2048, 1, 1]); // 32 * 64
        assert_eq!(m.ctx.ne(l5.wk.unwrap()), &[2048, 512, 1, 1]); // 8 * 64
        assert_eq!(m.ctx.ne(l5.wo.unwrap()), &[2048, 2048, 1, 1]);

        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&297), "histogram {hist:?}");
        assert_eq!(hist.get("Q4K"), Some(&186), "histogram {hist:?}");
        assert_eq!(hist.get("Q6K"), Some(&23), "histogram {hist:?}");
        eprintln!(
            "granite-4.0-h-micro: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
        assert_eq!(m.tensors.len(), 506);
    }

    #[test]
    fn gpt_oss_q4_k_m_same_mapping() {
        // The same gpt-oss arch in a K-quant conversion: the mapping/shapes are
        // identical, only the types differ from the MXFP4 file (and the expert
        // tensors are still MXFP4 here).
        let Some((gguf, mmap)) = open_model(GPT_OSS_Q4KM) else {
            return;
        };
        let t0 = std::time::Instant::now();
        let m = load_model(&gguf, mmap).expect("gpt-oss Q4_K_M load_model");
        let load_ms = t0.elapsed();

        assert_eq!(m.arch, LlmArch::OPENAI_MOE);
        assert_eq!(m.layers.len(), 24);
        assert_eq!(
            m.ctx.ne(m.layers[0].ffn_gate_exps.unwrap()),
            &[2880, 2880, 32, 1]
        );
        assert_eq!(
            m.ctx.ty(m.layers[0].ffn_gate_exps.unwrap()),
            GgmlType::Mxfp4
        );
        assert_matches_file(&m, &gguf);
        let hist = type_histogram(&m);
        assert_eq!(hist.get("F32"), Some(&289), "histogram {hist:?}");
        assert_eq!(hist.get("Mxfp4"), Some(&72), "histogram {hist:?}");
        assert_eq!(hist.get("Q5_0"), Some(&61), "histogram {hist:?}");
        assert_eq!(hist.get("Q4K"), Some(&24), "histogram {hist:?}");
        assert_eq!(hist.get("Q8_0"), Some(&13), "histogram {hist:?}");
        eprintln!(
            "gpt-oss-20b-Q4_K_M: {} tensors loaded in {load_ms:?}; {hist:?}",
            m.tensors.len()
        );
        assert_eq!(m.tensors.len(), 459);
    }

    #[test]
    fn gemma4_synthetic_per_layer_and_shared_kv() {
        // No local gemma4 file has per-layer embeddings (n_embd_per_layer > 0)
        // or shared KV layers, and none uses the fused attn_qkv path — this
        // tiny synthetic file covers exactly those branches of gemma4.cpp:
        //   * per_layer_token_embd / per_layer_model_proj / per_layer_proj_norm
        //     (model level) + inp_gate / proj / post_norm (per layer)
        //   * shared_kv_layers = 1 -> layer 1 has has_kv() == false, so wk and
        //     attn_k_norm are optional (absent from the file -> None)
        //   * layer 1 uses the fused attn_qkv variant (use_alternative_attention)
        //   * rope_freqs is created once, on the first full-attention layer
        let n_embd = 64i64;
        let n_head = 4i64;
        let n_head_kv = 2i64;
        let head_dim = 16i64;
        let n_embd_per_layer = 8i64;
        let n_ff = 128i64;
        let n_vocab = 32i64;

        let kvs = vec![
            ("general.architecture", Value::String("gemma4".into())),
            ("gemma4.context_length", Value::U32(512)),
            ("gemma4.embedding_length", Value::U32(n_embd as u32)),
            ("gemma4.block_count", Value::U32(2)),
            ("gemma4.attention.head_count", Value::U32(n_head as u32)),
            (
                "gemma4.attention.head_count_kv",
                Value::U32(n_head_kv as u32),
            ),
            ("gemma4.attention.key_length", Value::U32(head_dim as u32)),
            ("gemma4.attention.value_length", Value::U32(head_dim as u32)),
            (
                "gemma4.attention.key_length_swa",
                Value::U32(head_dim as u32),
            ),
            (
                "gemma4.attention.value_length_swa",
                Value::U32(head_dim as u32),
            ),
            ("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("gemma4.attention.sliding_window", Value::U32(1024)),
            (
                "gemma4.attention.sliding_window_pattern",
                Value::Array(
                    GgufType::Bool,
                    vec![Value::Bool(true), Value::Bool(false)], // layer 1 dense
                ),
            ),
            ("gemma4.attention.shared_kv_layers", Value::U32(1)),
            (
                "gemma4.embedding_length_per_layer_input",
                Value::U32(n_embd_per_layer as u32),
            ),
            ("gemma4.final_logit_softcapping", Value::F32(30.0)),
            ("gemma4.rope.dimension_count", Value::U32(head_dim as u32)),
            ("gemma4.rope.freq_base", Value::F32(1_000_000.0)),
            ("gemma4.feed_forward_length", Value::U32(n_ff as u32)),
            (
                "tokenizer.ggml.tokens",
                Value::Array(
                    GgufType::String,
                    (0..n_vocab)
                        .map(|i| Value::String(format!("t{i}")))
                        .collect(),
                ),
            ),
        ];

        let n_q = head_dim * n_head; // 64
        let n_kv = head_dim * n_head_kv; // 32
        let per_layer_total = n_embd_per_layer * 2; // n_layer * per-layer dim

        let mut tensors: Vec<(String, [i64; 2])> = vec![
            ("token_embd.weight".into(), [n_embd, n_vocab]),
            ("output.weight".into(), [n_embd, n_vocab]),
            ("output_norm.weight".into(), [n_embd, 1]),
            (
                "per_layer_token_embd.weight".into(),
                [per_layer_total, n_vocab],
            ),
            (
                "per_layer_model_proj.weight".into(),
                [n_embd, per_layer_total],
            ),
            ("per_layer_proj_norm.weight".into(), [n_embd_per_layer, 1]),
        ];
        // layer 0: SWA + kv -> separate q/k/v
        let p = "blk.0.";
        tensors.extend([
            (format!("{p}attn_norm.weight"), [n_embd, 1]),
            (format!("{p}attn_q.weight"), [n_embd, n_q]),
            (format!("{p}attn_k.weight"), [n_embd, n_kv]),
            (format!("{p}attn_v.weight"), [n_embd, n_kv]),
            (format!("{p}attn_output.weight"), [n_q, n_embd]),
            (format!("{p}attn_q_norm.weight"), [head_dim, 1]),
            (format!("{p}attn_k_norm.weight"), [head_dim, 1]),
            (format!("{p}post_attention_norm.weight"), [n_embd, 1]),
            (format!("{p}layer_output_scale.weight"), [1, 1]),
            (format!("{p}ffn_norm.weight"), [n_embd, 1]),
            (format!("{p}ffn_gate.weight"), [n_embd, n_ff]),
            (format!("{p}ffn_up.weight"), [n_embd, n_ff]),
            (format!("{p}ffn_down.weight"), [n_ff, n_embd]),
            (format!("{p}post_ffw_norm.weight"), [n_embd, 1]),
            (format!("{p}inp_gate.weight"), [n_embd, n_embd_per_layer]),
            (format!("{p}proj.weight"), [n_embd_per_layer, n_embd]),
            (format!("{p}post_norm.weight"), [n_embd, 1]),
        ]);
        // layer 1: full attention + shared KV (no wk/attn_k_norm) + fused qkv
        let p = "blk.1.";
        tensors.extend([
            (format!("{p}attn_norm.weight"), [n_embd, 1]),
            (format!("{p}attn_qkv.weight"), [n_embd, n_q + n_kv + n_kv]),
            (format!("{p}attn_output.weight"), [n_q, n_embd]),
            (format!("{p}attn_q_norm.weight"), [head_dim, 1]),
            (format!("{p}post_attention_norm.weight"), [n_embd, 1]),
            (format!("{p}layer_output_scale.weight"), [1, 1]),
            (format!("{p}ffn_norm.weight"), [n_embd, 1]),
            (format!("{p}ffn_gate.weight"), [n_embd, n_ff]),
            (format!("{p}ffn_up.weight"), [n_embd, n_ff]),
            (format!("{p}ffn_down.weight"), [n_ff, n_embd]),
            (format!("{p}post_ffw_norm.weight"), [n_embd, 1]),
            (format!("{p}inp_gate.weight"), [n_embd, n_embd_per_layer]),
            (format!("{p}proj.weight"), [n_embd_per_layer, n_embd]),
            (format!("{p}post_norm.weight"), [n_embd, 1]),
        ]);
        // first full-attention layer (blk.1) creates rope_freqs, no blk prefix
        tensors.push(("rope_freqs.weight".into(), [head_dim / 2, 1]));

        let path = write_test_gguf("gemma4-perlayer", kvs, &tensors);
        let gguf = Gguf::open(&path).unwrap();
        let m = load_model(&gguf, gguf_mmap(&path)).expect("gemma4 synth load");

        assert_eq!(m.arch, LlmArch::GEMMA4);
        assert_eq!(m.layers.len(), 2);
        assert_eq!(m.hparams.n_embd_per_layer, n_embd_per_layer as u32);
        // shared_kv_layers = 1 -> only layer 0 carries KV
        assert_eq!(m.hparams.n_layer_kv_from_start, 1);
        assert!(m.hparams.has_kv(0));
        assert!(!m.hparams.has_kv(1));

        // model-level per-layer embedding tensors
        assert_eq!(
            m.ctx.ne(m.per_layer_tok_embd.unwrap()),
            &[per_layer_total, n_vocab, 1, 1]
        );
        assert_eq!(
            m.ctx.ne(m.per_layer_model_proj.unwrap()),
            &[n_embd, per_layer_total, 1, 1]
        );
        assert_eq!(
            m.ctx.ne(m.per_layer_proj_norm.unwrap()),
            &[n_embd_per_layer, 1, 1, 1]
        );

        let l0 = &m.layers[0];
        assert!(m.hparams.is_swa(0));
        assert_eq!(m.ctx.ne(l0.wk.unwrap()), &[n_embd, n_kv, 1, 1]);
        assert_eq!(m.ctx.ne(l0.attn_k_norm.unwrap()), &[head_dim, 1, 1, 1]);
        assert!(l0.wqkv.is_none());
        assert!(l0.rope_freqs.is_none(), "SWA layer has no rope_freqs");
        assert_eq!(
            m.ctx.ne(l0.per_layer_inp_gate.unwrap()),
            &[n_embd, n_embd_per_layer, 1, 1]
        );
        assert_eq!(
            m.ctx.ne(l0.per_layer_proj.unwrap()),
            &[n_embd_per_layer, n_embd, 1, 1]
        );
        assert_eq!(
            m.ctx.ne(l0.per_layer_post_norm.unwrap()),
            &[n_embd, 1, 1, 1]
        );

        let l1 = &m.layers[1];
        assert!(!m.hparams.is_swa(1));
        // fused qkv (v_proj present -> no separate q/k/v)
        assert_eq!(m.ctx.ne(l1.wqkv.unwrap()), &[n_embd, n_q + 2 * n_kv, 1, 1]);
        assert!(l1.wq.is_none() && l1.wk.is_none() && l1.wv.is_none());
        // shared KV: wk / attn_k_norm are optional and absent
        assert!(l1.wk.is_none() && l1.attn_k_norm.is_none());
        // rope_freqs created on the first full-attention layer only
        assert_eq!(m.ctx.ne(l1.rope_freqs.unwrap()), &[head_dim / 2, 1, 1, 1]);

        assert_matches_file(&m, &gguf);
        // 6 top-level + 17 (layer 0) + 14 (layer 1) + rope_freqs = 38
        assert_eq!(m.tensors.len(), 38);
    }

    #[test]
    fn support_matrix() {
        use ArchTensorsSupport::*;
        assert_eq!(arch_tensors_support(LlmArch::QWEN2), Full);
        assert_eq!(arch_tensors_support(LlmArch::LLAMA), Full);
        assert_eq!(arch_tensors_support(LlmArch::QWEN3), Partial);
        assert_eq!(arch_tensors_support(LlmArch::GEMMA2), Partial);
        assert_eq!(arch_tensors_support(LlmArch::GEMMA3), Partial);
        assert_eq!(arch_tensors_support(LlmArch::PHI3), Partial);
        // agent P additions
        assert_eq!(arch_tensors_support(LlmArch::OPENAI_MOE), Partial);
        assert_eq!(arch_tensors_support(LlmArch::GEMMA4), Partial);
        assert_eq!(arch_tensors_support(LlmArch::LFM2MOE), Partial);
        assert_eq!(arch_tensors_support(LlmArch::GRANITE_HYBRID), Partial);
        assert_eq!(arch_tensors_support(LlmArch::QWEN35), Partial);
        // dense granite landed with arch batch 3 (this file, GRANITE arm)
        assert_eq!(arch_tensors_support(LlmArch::GRANITE), Partial);
        assert_eq!(arch_tensors_support(LlmArch::MINICPM), Partial);
        // arch batch 3: the ALiBi family + plamo / stablelm
        for a in [
            LlmArch::BAICHUAN,
            LlmArch::BLOOM,
            LlmArch::MPT,
            LlmArch::STARCODER,
            LlmArch::REFACT,
            LlmArch::PLAMO,
            LlmArch::STABLELM,
        ] {
            assert_eq!(arch_tensors_support(a), Partial);
        }
        // MTP batch 17 (2026-09-29): lfm2 (dense) rides the lfm2moe arm —
        // lfm2.cpp's load_arch_tensors is the same body with
        // n_layer_dense_lead == n_layer (meta.rs's LFM2 hparams arm), so the
        // is_moe_layer split degenerates to the dense FFN everywhere
        assert_eq!(arch_tensors_support(LlmArch::LFM2), Partial);
        // qwen35moe landed with arch batch 11a (the long-tail queue's first
        // half) — Partial since then
        assert_eq!(arch_tensors_support(LlmArch::QWEN35MOE), Partial);
        assert_eq!(arch_tensors_support(LlmArch::UNKNOWN), Unsupported);
    }
}

#[cfg(test)]
mod batch12_count_check {
    /// sanity for the PARITY/FILE_MAP/COVERAGE count claims: the ported-arch
    /// count (Full + Partial — the "arch_tensors_support convention") is 95 after
    /// batch 12 (appended by the batch-12 integrator; remove freely)
    #[test]
    fn arch_ported_count_is_95() {
        let n = crate::arch::LlmArch::ALL
            .iter()
            .filter(|&&a| super::arch_tensors_support(a) != super::ArchTensorsSupport::Unsupported)
            .count();
        // 95 after batch 12; 115 after batch 13/14; 136 after batch 15; 142
        // with the bert-variant family; 145 with the audio-round-5 TTS trio
        // (QWEN3TTS/POCKETTTS/WAVTOKENIZER_DEC, all Partial — graphs in
        // graph_arch.rs, ForwardWeights routing is the integrator item)
        // 147 with MTP batch 17 (LFM2 sharing the lfm2moe arm + the full T5
        // encoder-decoder arch); 148 with sync batch A's GLM5_NEXT (Partial —
        // loader/table face, the graph lands with the kpool memory port,
        // PARITY.md sync batch A §2); 149 with sync batch A2's CLEF (Full —
        // clef.rs + the decision tables, PARITY.md sync batch A2)
        assert_eq!(
            n, 149,
            "arch_tensors_support ported count"
        );
    }
}
