//! hparams.rs — 1:1 port of llama.cpp `src/llama-hparams.h/.cpp` (def4d406a).
//!
//! Per-layer `std::array<T, LLAMA_MAX_LAYERS>` members become `Vec<T>` kept at
//! length [`LLAMA_MAX_LAYERS`] (the loader sizes them, exactly like the C++
//! `std::fill(...begin, ...end)` prologue). Out-of-range layer indices panic,
//! mirroring the C++ `GGML_ABORT` paths.

use crate::arch::LLAMA_MAX_LAYERS;

/// `#define LLAMA_MAX_EXPERTS 1024` — Kimi K3
pub const LLAMA_MAX_EXPERTS: u32 = 1024;
/// `#define LLAMA_MAX_PLE_NGRAM 8` — qwen4exp
pub const LLAMA_MAX_PLE_NGRAM: usize = 8;
/// `#define LLAMA_MAX_PLE_HEADS 64` — qwen4exp
pub const LLAMA_MAX_PLE_HEADS: usize = 64;

/// `enum llama_expert_gating_func_type`
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LlamaExpertGatingFuncType {
    NONE = 0,
    SOFTMAX = 1,
    SIGMOID = 2,
    /// applied to the router weights instead of the logits
    SOFTMAX_WEIGHT = 3,
    SQRT_SOFTPLUS = 4,
}

/// `enum llama_swa_type`
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum LlamaSwaType {
    #[default]
    NONE = 0,
    STANDARD = 1,
    CHUNKED = 2,
    SYMMETRIC = 3,
}

/// `enum llama_non_causal_type` — how the non-causal mask is constructed with
/// `llama_set_causal_attn(ctx, false)` (e.g. mtmd decoding image tokens).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum LlamaNonCausalType {
    /// all layers non-causal, SWA still applied (gemma 3, qwen-vl, ...)
    #[default]
    ALL = 0,
    /// SWA layers non-causal, dense layers stay causal (gemma 4)
    SWA_ONLY = 1,
    /// all layers non-causal, SWA not applied between tokens of the current
    /// ubatch (deepseek 4)
    SWA_FULL = 2,
}

/// `enum llama_pooling_type` (llama.h)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum LlamaPoolingType {
    /// LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    UNSPECIFIED = -1,
    #[default]
    NONE = 0,
    MEAN = 1,
    CLS = 2,
    LAST = 3,
    /// used by reranking models to attach the classification head to the graph
    RANK = 4,
}

/// `enum llama_rope_type` (llama.h; values are ggml rope mode bits)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum LlamaRopeType {
    NONE = -1,
    #[default]
    NORM = 0,
    /// GGML_ROPE_TYPE_NEOX = 2
    NEOX = 2,
    /// GGML_ROPE_TYPE_MROPE = 8
    MROPE = 8,
    /// GGML_ROPE_TYPE_IMROPE = 40
    IMROPE = 40,
    /// GGML_ROPE_TYPE_VISION = 24
    VISION = 24,
}

/// `enum llama_rope_scaling_type` (llama.h)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum LlamaRopeScalingType {
    UNSPECIFIED = -1,
    #[default]
    NONE = 0,
    LINEAR = 1,
    YARN = 2,
    LONGROPE = 3,
}

impl LlamaRopeScalingType {
    /// `LLAMA_ROPE_SCALING_TYPES` inverse — `llama_rope_scaling_type_from_string`
    pub fn from_name(name: &str) -> Self {
        match name {
            "none" => Self::NONE,
            "linear" => Self::LINEAR,
            "yarn" => Self::YARN,
            "longrope" => Self::LONGROPE,
            _ => Self::UNSPECIFIED,
        }
    }
}

/// `enum llm_ffn_op_type : int` (llama-graph.h) — resolved FFN gated activation.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum LlmFfnOpType {
    /// sentinel: unset; archs must assign before use
    #[default]
    NONE = 0,
    SILU = 1,
    GELU = 2,
    RELU = 3,
    RELU_SQR = 4,
    SWIGLU = 5,
    GEGLU = 6,
    REGLU = 7,
    SWIGLU_OAI_MOE = 8,
    /// kimi-k3
    SITU = 9,
}

/// `LLM_CLS_ACT_TYPES_FROM_STRING` (37ac63456, llama-model.cpp:1081-1086) —
/// transformers names, "gelu" is the *exact* (erf) variant. Returns `None`
/// for an unknown name — the C's `GGML_ASSERT(it != end && "unsupported
/// classifier activation")` (llama-model.cpp:1348-1352) fires at the loader.
pub fn cls_act_type_from_string(name: &str) -> Option<i32> {
    match name {
        // transformers names, "gelu" is the exact (erf) variant
        "gelu" => Some(ggml::ops::GGML_UNARY_OP_GELU_ERF),
        "silu" => Some(ggml::ops::GGML_UNARY_OP_SILU),
        "tanh" => Some(ggml::ops::GGML_UNARY_OP_TANH),
        _ => None,
    }
}

/// `struct llama_hparams_posnet` (WavTokenizer)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlamaHparamsPosnet {
    pub n_embd: u32,
    pub n_layer: u32,
}

/// `struct llama_hparams_convnext` (WavTokenizer)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LlamaHparamsConvnext {
    pub n_embd: u32,
    pub n_layer: u32,
}

/// `struct llama_hparams` (llama-hparams.h).
///
/// Field order and names follow the C struct 1:1 (snake_case as in C).
/// `std::array<T, LLAMA_MAX_LAYERS>` members are `Vec<T>` sized 512 by
/// [`LlamaHparams::new`]; `_impl` suffixes on members that also have getters
/// are kept.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaHparams {
    pub vocab_only: bool,
    pub no_alloc: bool,
    pub rope_finetuned: bool,
    pub use_par_res: bool,
    pub swin_norm: bool,
    pub norm_before_residual: bool,
    pub norm_before_fc: bool,

    /// context size the model was trained on
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer_all: u32,
    pub n_layer_nextn: u32,
    /// trailing blocks that form the decision head (llama-hparams.h:68,
    /// modern-bert / clef)
    pub n_layer_decision: u32,
    /// clef (a7b94df2c clef.cpp:13): the head's routing block count — an
    /// upstream *model* member (`llama_model_clef::n_layer_routing`); kept
    /// here like `router_layer` (the granite-switch precedent)
    pub clef_n_layer_routing: u32,
    /// clef (clef.cpp:14): the head's attention head count
    /// (`llama_model_clef::n_head_decision`)
    pub clef_n_head_decision: u32,

    /// granite-switch: index of the single-head "router" KV layer that encodes
    /// per-token adapter selection. -1 when the model has no such layer.
    pub router_layer: i32,
    /// granite-switch: `n_adapters` / `max_lora_rank` / `router_gain` of the
    /// stacked switch-LoRA slots (granite-switch.cpp:25-28; C keeps them on
    /// the model subclass — the port carries them on hparams so the tensor
    /// loader and the graph builder see one source of truth). The
    /// activate/substitute token maps are re-read by the loader (model.rs).
    pub graniteswitch_n_adapters: u32,
    pub graniteswitch_max_lora_rank: u32,
    pub graniteswitch_router_gain: f32,
    /// nanbeige: the GGUF `block_count` before the loop expansion
    /// (`n_layer_phys`, nanbeige.cpp:13) and the loop count read from
    /// `num_loops` (:6-17; C keeps them on the model subclass — the port
    /// carries them on hparams so the tensor loader's layer aliasing and the
    /// graph's loop-boundary norm see one source of truth)
    pub nanbeige_n_layer_phys: u32,
    pub nanbeige_n_loops: u32,
    pub nanbeige_skip_loop_final_norm: bool,
    pub n_expert: u32,
    pub n_rel_attn_bkts: u32,
    /// MoVA value experts (K2 Horizon) — llama-hparams.h:75-76 (462524043)
    pub n_value_expert: u32,
    pub n_value_expert_used: u32,

    /// if non-negative, the first n_layer_kv_from_start layers have KV cache
    /// TODO: this needs to be reworked (as in C)
    pub n_layer_kv_from_start: i32,

    /// different head size for full_attention and SWA layers:
    /// dimension of keys (d_k). d_q is assumed to be the same, but there are
    /// n_head q heads, and only n_head_kv k-v heads
    pub n_embd_head_k_full: u32,
    /// dimension of values (d_v) aka n_embd_head
    pub n_embd_head_v_full: u32,
    pub n_embd_head_k_swa: u32,
    pub n_embd_head_v_swa: u32,

    /// different RoPE dimensions for full_attention and SWA layers
    pub n_rot_full: u32,
    pub n_rot_swa: u32,

    /// note: deepseek2 using MLA converts into MQA with larger heads, then
    /// decompresses to MHA
    pub n_embd_head_k_mla_impl: u32,
    pub n_embd_head_v_mla_impl: u32,

    // for WavTokenizer
    pub posnet: LlamaHparamsPosnet,
    pub convnext: LlamaHparamsConvnext,

    pub n_shortconv_l_cache: u32,

    pub n_head_arr: Vec<u32>,
    pub n_head_kv_arr: Vec<u32>,
    pub n_ff_arr: Vec<u32>,

    /// per-layer expert feed-forward size
    pub n_ff_exp_arr: Vec<u32>,
    /// per-layer top-k expert routing count
    pub n_expert_used_arr: Vec<u32>,

    pub n_layer_dense_lead: u32,
    pub n_lora_q: u32,
    pub n_lora_kv: u32,
    pub n_ff_shexp: u32,
    pub n_ff_chexp: u32,
    pub n_expert_shared: u32,
    pub n_norm_groups: u32,
    pub n_expert_groups: u32,
    pub n_group_used: u32,
    pub n_group_experts: u32,

    // MLA + SWA (i.e. dots3note)
    pub n_lora_kv_swa: u32,
    pub n_embd_head_k_mla_swa: u32,
    pub n_embd_head_v_mla_swa: u32,

    pub expert_group_scale: f32,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32, // LlamaExpertGatingFuncType as raw value (C keeps uint32_t)
    pub moe_every_n_layers: u32,
    pub moe_latent_size: u32,

    pub f_norm_eps: f32,
    pub f_norm_rms_eps: f32,
    pub f_norm_group_eps: f32,

    pub f_attn_logit_softcapping: f32,
    pub f_router_logit_softcapping: f32,
    pub f_final_logit_softcapping: f32,

    // for RWKV
    pub rescale_every_n_layers: u32,
    pub time_mix_extra_dim: u32,
    pub time_decay_extra_dim: u32,
    pub wkv_head_size: u32,
    pub token_shift_count: u32,
    pub n_lora_decay: u32,
    pub n_lora_iclr: u32,
    pub n_lora_value_res_mix: u32,
    pub n_lora_gate: u32,

    pub rope_attn_factor: f32,
    pub rope_freq_base_train: f32,
    pub rope_freq_base_train_swa: f32,
    pub rope_freq_scale_train: f32,
    pub rope_freq_scale_train_swa: f32,
    /// NTK-aware alpha for XDRoPE
    pub rope_scaling_alpha: f32,

    pub n_ctx_orig_yarn: u32,
    pub rope_yarn_log_mul: f32,

    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,

    pub rope_sections: [i32; 4],

    /// Per-layer RoPE enable flags (1 = use RoPE, 0 = NoPE);
    /// by default, all layers use RoPE (controlled by rope_finetuned)
    pub rope_pattern: Vec<u32>,

    /// Sliding Window Attention (SWA)
    pub swa_type: LlamaSwaType,
    /// the size of the sliding window (0 - no SWA)
    pub n_swa: u32,

    /// see LlamaNonCausalType
    /// note: for SWA_FULL, older tokens (outside the current ubatch) are still
    /// window-clipped
    pub non_causal_type: LlamaNonCausalType,

    /// if is_swa_impl[il] == 1, then layer il is SWA
    /// if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    /// by default, all layers are dense
    /// note: using uint32_t type for compatibility reason (as in C)
    pub is_swa_impl: Vec<u32>,

    /// for hybrid state space models
    pub is_recr_impl: Vec<u32>,

    // for State Space Models
    pub ssm_d_conv: u32,
    pub ssm_d_inner: u32,
    pub ssm_d_state: u32,
    pub ssm_dt_rank: u32,
    pub ssm_n_group: u32,

    // for MiniMax-Text-01 linear attention
    pub n_embd_head_la: u32,

    // for Kimi Linear KDA
    pub n_embd_head_kda: u32,
    pub kda_safe_gate: bool,

    // kimi-k3
    /// routed_expert_hidden_size (0 = experts run at n_embd)
    pub n_expert_latent: u32,
    /// 0 = no cross-layer attention residuals
    pub attn_res_block_size: u32,
    pub kda_gate_lower_bound: f32,
    pub situ_beta: f32,
    /// 0 = no linear-beta transform on the up branch
    pub situ_linear_beta: f32,

    // hrm-text (looped H/L stacks)
    pub n_hrm_layers_per_stack: u32,
    pub n_hrm_h_cycles: u32,
    pub n_hrm_l_cycles: u32,
    pub hrm_prefix_lm: bool,

    pub ssm_dt_b_c_rms: bool,

    pub f_clamp_kqv: f32,
    pub f_max_alibi_bias: f32,
    pub f_logit_scale: f32,

    // Additional scale factors (Granite/Granite MoE)
    pub f_residual_scale: f32,
    pub f_embedding_scale: f32,
    pub f_attention_scale: f32,

    // grok-2
    pub f_attn_out_scale: f32,
    pub attn_temp_length: u32,

    pub f_attn_value_scale: f32,

    pub causal_attn: bool,
    pub use_alibi: bool,
    pub attn_soft_cap: bool,
    pub use_kq_norm: bool,

    // for Classifiers
    pub n_cls_out: u32,

    /// input embedding dimension (0 = use n_embd)
    pub n_embd_inp_impl: u32,

    /// encoder input embedding dimension (0 = use n_embd_inp())
    /// e.g. the eagle3 encoder fuses target_layers * target_hidden features
    pub n_embd_inp_enc_impl: u32,

    /// output embedding dimension (0 = use n_embd)
    pub n_embd_out_impl: u32,

    pub dflash_block_size: u32,
    pub dflash_conv_kernel_size: u32,
    pub dflash_conv_group_size: u32,
    pub dflash_selector_rank: u32,
    pub dflash_selector_top_k: u32,

    // llama4 smallthinker
    pub n_moe_layer_step: u32,
    pub n_no_rope_layer_step: u32,
    pub n_attn_temp_floor_scale: u32,
    pub f_attn_temp_scale: f32,
    /// offset position index
    pub f_attn_temp_offset: f32,

    // gemma3n altup
    /// altup_num_inputs
    pub n_altup: u32,
    /// altup_active_idx
    pub i_altup_act: u32,
    pub laurel_rank: u32,
    pub n_embd_altup: u32,

    // needed for sentence-transformers dense layers
    /// in_features of the 2_Dense
    pub dense_2_feat_in: u32,
    /// out_features of the 2_Dense
    pub dense_2_feat_out: u32,
    /// in_features of the 3_Dense
    pub dense_3_feat_in: u32,
    /// out_features of the 3_Dense
    pub dense_3_feat_out: u32,

    // xIELU
    pub xielu_alpha_n: Vec<f32>,
    pub xielu_alpha_p: Vec<f32>,
    pub xielu_beta: Vec<f32>,
    pub xielu_eps: Vec<f32>,

    // DSA (deepseek sparse attention)
    pub indexer_n_head: u32,
    pub indexer_head_size: u32,
    pub indexer_top_k: u32,
    /// k-pool size (glm5-next, llama-hparams.h:285)
    pub indexer_kpool: u32,
    /// k-pool tail selection (llama-hparams.h:286; the file default is true)
    pub indexer_kpool_select_tail: bool,
    /// head-size slots per cached indexer row, the last one holds the pooled
    /// key (llama-hparams.h:288-289; glm5-next = 3 key|gate|pooled, qwen4exp
    /// = 2 raw|pooled)
    pub indexer_kpool_row: u32,
    /// pools are consecutive cells in sequence order, not runs of consecutive
    /// positions (llama-hparams.h:290-291; qwen4exp sets it)
    pub indexer_kpool_by_order: bool,
    // MSA
    pub indexer_block_size: u32,
    pub indexer_local_blocks: u32,

    /// Indexer is "full" (1) or "shared" (0)
    /// Shared indexers reuse top-k from previous full layer
    pub is_indexer_full_impl: Vec<u32>,

    // DeepSeek-V4
    pub dsv4_o_group_count: u32,
    pub dsv4_o_lora_rank: u32,
    pub dsv4_hc_mult: u32,
    pub dsv4_hc_sinkhorn_iters: u32,
    pub dsv4_hash_layer_count: u32,
    pub dsv4_compress_rope_base: f32,
    pub dsv4_hc_eps: f32,
    pub dsv4_compress_ratios: Vec<u32>,

    /// 0 = full rank (DeepSeek-V4)
    pub hc_low_rank: u32,

    /// scale of the hyper-connection post gate (DeepSeek-V4 hardcodes 2.0)
    pub hc_magnitude: f32,

    pub ple_ngram_size: u32,
    pub ple_heads_per_ngram: u32,
    pub ple_conv_kernel: u32,
    /// (ngram_size - 1) * heads_per_ngram
    pub ple_n_heads: u32,
    pub ple_head_dim: u32,
    pub ple_eos_token_id: u32,
    /// the id the PLE hash stands in at image positions; 0 makes the loader
    /// fall back to EOS
    pub ple_image_token_id: u32,
    /// the file lists PLE layer indices, so this is never a per-layer gguf
    /// array and can hold one bit per layer
    pub is_ple_impl: Vec<bool>,
    /// the hash multipliers reach ~2e13 and have to stay 64-bit
    pub ple_layer_multipliers: [u64; LLAMA_MAX_PLE_NGRAM],
    /// head offsets and vocab sizes are token-space indices; the gather
    /// truncates them to int32 anyway
    pub ple_head_offsets: [u32; LLAMA_MAX_PLE_HEADS],
    pub ple_head_vocab_sizes: [u32; LLAMA_MAX_PLE_HEADS],

    // qwen3vl deepstack
    // When parsed from GGUF, this implies the first N layers consume the first
    // N deepstack embeddings. Use deepstack_mapping_arr if you need a more
    // complex mapping. If using deepstack_mapping_arr, also make sure to set
    // n_deepstack_layers to the number of unique deepstack layers so that
    // n_embd_imp is accurate (see granite.cpp).
    // TODO: can be expressed via the `new n_embd_inp_impl` and remove this param
    pub n_deepstack_layers: u32,

    /// deepstack layer array (Granite4 Vision)
    /// -1  => no deepstack
    /// >=0 => input embedding index for deepstack injection
    pub deepstack_mapping_arr: Vec<i32>,

    /// gemma4 per-layer embedding
    pub n_embd_per_layer: u32,

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggml-org/llama.cpp/pull/8141
    /// LLAMA_TOKEN_NULL == -1
    pub dec_start_token_id: i32,
    pub dec_n_layer: u32,

    pub pooling_type: LlamaPoolingType,
    /// pooling before the classifier head (RANK) — llama-hparams.h:355,
    /// `%s.classifier.pooling_type`; UNSPECIFIED lets the arch default
    pub pooling_type_cls: LlamaPoolingType,
    pub rope_type: LlamaRopeType,
    pub rope_scaling_type_train: LlamaRopeScalingType,

    /// Resolved FFN gated activation flavor for archs that read
    /// `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
    /// Defaults to LLM_FFN_NONE (sentinel = 0); the mapping from the GGUF
    /// string to a real op is done at hparam-load time in meta.rs, mirroring
    /// how rope_scaling_type_train is handled.
    pub llm_ffn_op: LlmFfnOpType,

    /// `hparams.act_cls` (37ac63456, llama-hparams.h:374) — activation of
    /// the classifier head (RANK pooling), a `ggml_unary_op` code
    /// (`ggml::ops::GGML_UNARY_OP_*`). Default TANH; `%s.classifier.
    /// activation` overrides it ("gelu" = the *exact* erf variant / "silu" /
    /// "tanh", LLM_CLS_ACT_TYPES_FROM_STRING, llama-model.cpp:1081-1086 —
    /// `cls_act_type_from_string`). `build_pooling` applies it where the
    /// modern_bert gelu special case used to be (llama-graph.cpp:3906).
    pub act_cls: i32,

    // Step35: optional per-layer clamps for (Swi)GLU
    /// clamping for expert FFN
    pub swiglu_clamp_exp: Vec<f32>,
    /// shared expert
    pub swiglu_clamp_shexp: Vec<f32>,
}

impl Default for LlamaHparams {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime rope parameters derived exactly like llama-context.cpp:106-215
/// (`llama_context` cparams derivation). The CLI/tests previously passed raw
/// hparams fields, which dropped `yarn_attn_factor *= rope_attn_factor`
/// (agent N's P4 root cause) and the negative-ext-factor mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeRuntime {
    pub n_ctx_orig_yarn: i32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl LlamaHparams {
    /// `params.*` overrides we do not expose are at their llama.cpp defaults:
    /// `yarn_orig_ctx = 0`, `yarn_ext_factor = -1.0f` (llama-context.cpp:3713),
    /// `rope_freq_scale = 0.0f`.
    pub fn rope_runtime(&self) -> RopeRuntime {
        use crate::hparams::LlamaRopeScalingType as S;
        // cparams.rope_freq_scale = params(0.0) == 0 ? hparams : params
        let mut freq_scale = if self.rope_freq_scale_train == 0.0 {
            1.0 // llama-hparams.h default (reference has 1.0f; file key absent -> 1.0)
        } else {
            self.rope_freq_scale_train
        };
        let n_ctx_orig_yarn = if self.n_ctx_orig_yarn != 0 {
            self.n_ctx_orig_yarn as i32
        } else {
            self.n_ctx_train as i32
        };
        let scaling = self.rope_scaling_type_train;
        if scaling == S::NONE {
            freq_scale = 1.0; // llama-context.cpp:170-172
        }
        // params.yarn_ext_factor == -1 (unset) -> hparams value
        let mut ext_factor = self.yarn_ext_factor;
        if ext_factor < 0.0 {
            ext_factor = if scaling == S::YARN { 1.0 } else { 0.0 };
        }
        let mut attn_factor = 1.0f32; // params.yarn_attn_factor default (llama-context.cpp:3712)
        if ext_factor != 0.0 {
            let factor = 1.0 / freq_scale;
            let get_mscale = |scale: f32, mscale: f32| -> f32 {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * scale.ln() + 1.0
                }
            };
            if self.rope_yarn_log_mul != 0.0 {
                let mscale = 1.0f32;
                let mscale_all_dims = self.rope_yarn_log_mul;
                attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);
            } else {
                attn_factor = get_mscale(factor, 1.0);
            }
            // cancel the ops.cpp rope_yarn mscale (llama-context.cpp:206-209)
            attn_factor *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        attn_factor *= self.rope_attn_factor; // llama-context.cpp:214 (P4)
        RopeRuntime {
            n_ctx_orig_yarn,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast: self.yarn_beta_fast,
            beta_slow: self.yarn_beta_slow,
        }
    }
}

impl LlamaHparams {
    /// C++ value-initialization equivalent: zeroed members + the explicit
    /// default member initializers from llama-hparams.h. Per-layer arrays are
    /// allocated at LLAMA_MAX_LAYERS like the C++ `std::array` members and
    /// zeroed (the loader's `std::fill` prologue in meta.rs redoes this).
    pub fn new() -> Self {
        let mut h = Self {
            vocab_only: false,
            no_alloc: false,
            rope_finetuned: false,
            use_par_res: false,
            swin_norm: false,
            norm_before_residual: false,
            norm_before_fc: false,
            n_ctx_train: 0,
            n_embd: 0,
            n_layer_all: 0,
            n_layer_nextn: 0,
            n_layer_decision: 0,
            clef_n_layer_routing: 0,
            clef_n_head_decision: 0,
            router_layer: -1,
            graniteswitch_n_adapters: 0,
            graniteswitch_max_lora_rank: 0,
            graniteswitch_router_gain: 0.0,
            nanbeige_n_layer_phys: 0,
            nanbeige_n_loops: 1,
            nanbeige_skip_loop_final_norm: false,
            n_expert: 0,
            n_rel_attn_bkts: 0,
            n_value_expert: 0,
            n_value_expert_used: 0,
            n_layer_kv_from_start: -1,
            n_embd_head_k_full: 0,
            n_embd_head_v_full: 0,
            n_embd_head_k_swa: 0,
            n_embd_head_v_swa: 0,
            n_rot_full: 0,
            n_rot_swa: 0,
            n_embd_head_k_mla_impl: 0,
            n_embd_head_v_mla_impl: 0,
            posnet: LlamaHparamsPosnet::default(),
            convnext: LlamaHparamsConvnext::default(),
            n_shortconv_l_cache: 0,
            n_head_arr: Vec::new(),
            n_head_kv_arr: Vec::new(),
            n_ff_arr: Vec::new(),
            n_ff_exp_arr: Vec::new(),
            n_expert_used_arr: Vec::new(),
            n_layer_dense_lead: 0,
            n_lora_q: 0,
            n_lora_kv: 0,
            n_ff_shexp: 0,
            n_ff_chexp: 0,
            n_expert_shared: 0,
            n_norm_groups: 0,
            n_expert_groups: 0,
            n_group_used: 0,
            n_group_experts: 0,
            n_lora_kv_swa: 0,
            n_embd_head_k_mla_swa: 0,
            n_embd_head_v_mla_swa: 0,
            expert_group_scale: 0.05,
            expert_weights_scale: 0.0,
            expert_weights_norm: false,
            expert_gating_func: LlamaExpertGatingFuncType::NONE as u32,
            moe_every_n_layers: 0,
            moe_latent_size: 0,
            f_norm_eps: 0.0,
            f_norm_rms_eps: 0.0,
            f_norm_group_eps: 0.0,
            f_attn_logit_softcapping: 50.0,
            f_router_logit_softcapping: 30.0,
            f_final_logit_softcapping: 30.0,
            rescale_every_n_layers: 0,
            time_mix_extra_dim: 0,
            time_decay_extra_dim: 0,
            wkv_head_size: 0,
            token_shift_count: 2,
            n_lora_decay: 0,
            n_lora_iclr: 0,
            n_lora_value_res_mix: 0,
            n_lora_gate: 0,
            rope_attn_factor: 1.0, // llama-hparams.h:149 default is 1.0f (agent N; 0.0 zeroed rope on models lacking the key)
            rope_freq_base_train: 0.0,
            rope_freq_base_train_swa: 10000.0,
            rope_freq_scale_train: 0.0,
            rope_freq_scale_train_swa: 1.0,
            rope_scaling_alpha: 0.0,
            n_ctx_orig_yarn: 0,
            rope_yarn_log_mul: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            rope_sections: [0; 4],
            rope_pattern: Vec::new(),
            swa_type: LlamaSwaType::NONE,
            n_swa: 0,
            non_causal_type: LlamaNonCausalType::ALL,
            is_swa_impl: Vec::new(),
            is_recr_impl: Vec::new(),
            ssm_d_conv: 0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_dt_rank: 0,
            ssm_n_group: 0,
            n_embd_head_la: 0,
            n_embd_head_kda: 0,
            kda_safe_gate: false,
            n_expert_latent: 0,
            attn_res_block_size: 0,
            kda_gate_lower_bound: f32::NEG_INFINITY,
            situ_beta: 1.0,
            situ_linear_beta: 0.0,
            n_hrm_layers_per_stack: 0,
            n_hrm_h_cycles: 0,
            n_hrm_l_cycles: 0,
            hrm_prefix_lm: false,
            ssm_dt_b_c_rms: false,
            f_clamp_kqv: 0.0,
            f_max_alibi_bias: 0.0,
            f_logit_scale: 0.0,
            f_residual_scale: 0.0,
            f_embedding_scale: 0.0,
            f_attention_scale: 0.0,
            f_attn_out_scale: 0.0,
            attn_temp_length: 0,
            f_attn_value_scale: 0.0,
            causal_attn: true,
            use_alibi: false,
            attn_soft_cap: false,
            use_kq_norm: false,
            n_cls_out: 1,
            n_embd_inp_impl: 0,
            n_embd_inp_enc_impl: 0,
            n_embd_out_impl: 0,
            dflash_block_size: 0,
            dflash_conv_kernel_size: 0,
            dflash_conv_group_size: 0,
            dflash_selector_rank: 0,
            dflash_selector_top_k: 0,
            n_moe_layer_step: 0,
            n_no_rope_layer_step: 4,
            n_attn_temp_floor_scale: 0,
            f_attn_temp_scale: 0.0,
            f_attn_temp_offset: 0.0,
            n_altup: 4,
            i_altup_act: 0,
            laurel_rank: 64,
            n_embd_altup: 256,
            dense_2_feat_in: 0,
            dense_2_feat_out: 0,
            dense_3_feat_in: 0,
            dense_3_feat_out: 0,
            xielu_alpha_n: Vec::new(),
            xielu_alpha_p: Vec::new(),
            xielu_beta: Vec::new(),
            xielu_eps: Vec::new(),
            indexer_n_head: 0,
            indexer_head_size: 0,
            indexer_top_k: 0,
            indexer_kpool: 0,
            indexer_kpool_select_tail: true,
            indexer_kpool_row: 3,
            indexer_kpool_by_order: false,
            indexer_block_size: 0,
            indexer_local_blocks: 0,
            is_indexer_full_impl: Vec::new(),
            dsv4_o_group_count: 0,
            dsv4_o_lora_rank: 0,
            dsv4_hc_mult: 0,
            dsv4_hc_sinkhorn_iters: 0,
            dsv4_hash_layer_count: 0,
            dsv4_compress_rope_base: 0.0,
            dsv4_hc_eps: 0.0,
            dsv4_compress_ratios: Vec::new(),
            hc_low_rank: 0,
            hc_magnitude: 0.0,
            ple_ngram_size: 0,
            ple_heads_per_ngram: 0,
            ple_conv_kernel: 0,
            ple_n_heads: 0,
            ple_head_dim: 0,
            ple_eos_token_id: 0,
            ple_image_token_id: 0,
            is_ple_impl: Vec::new(),
            ple_layer_multipliers: [0; LLAMA_MAX_PLE_NGRAM],
            ple_head_offsets: [0; LLAMA_MAX_PLE_HEADS],
            ple_head_vocab_sizes: [0; LLAMA_MAX_PLE_HEADS],
            n_deepstack_layers: 0,
            deepstack_mapping_arr: Vec::new(),
            n_embd_per_layer: 0,
            dec_start_token_id: -1, // LLAMA_TOKEN_NULL
            dec_n_layer: 0,
            pooling_type: LlamaPoolingType::NONE,
            pooling_type_cls: LlamaPoolingType::UNSPECIFIED,
            rope_type: LlamaRopeType::NONE,
            rope_scaling_type_train: LlamaRopeScalingType::NONE,
            llm_ffn_op: LlmFfnOpType::NONE,
            // GGML_UNARY_OP_TANH (the C member initialiser, llama-hparams.h:374)
            act_cls: ggml::ops::GGML_UNARY_OP_TANH,
            swiglu_clamp_exp: Vec::new(),
            swiglu_clamp_shexp: Vec::new(),
        };
        let z = || vec![0u32; LLAMA_MAX_LAYERS];
        h.n_head_arr = z();
        h.n_head_kv_arr = z();
        h.n_ff_arr = z();
        h.n_ff_exp_arr = z();
        h.n_expert_used_arr = z();
        h.rope_pattern = vec![1u32; LLAMA_MAX_LAYERS];
        h.is_swa_impl = z();
        h.is_recr_impl = z();
        h.is_indexer_full_impl = z();
        h.xielu_alpha_n = vec![0.0; LLAMA_MAX_LAYERS];
        h.xielu_alpha_p = vec![0.0; LLAMA_MAX_LAYERS];
        h.xielu_beta = vec![0.0; LLAMA_MAX_LAYERS];
        h.xielu_eps = vec![0.0; LLAMA_MAX_LAYERS];
        h.deepstack_mapping_arr = vec![-1i32; LLAMA_MAX_LAYERS];
        h.swiglu_clamp_exp = vec![0.0; LLAMA_MAX_LAYERS];
        h.swiglu_clamp_shexp = vec![0.0; LLAMA_MAX_LAYERS];
        h.dsv4_compress_ratios = z();
        h.is_ple_impl = vec![false; LLAMA_MAX_LAYERS];
        h
    }

    /// `llama_hparams::set_swa_pattern`
    ///
    /// this value n_pattern means that every nth layer is dense (i.e. non-SWA)
    /// dense_first means whether the pattern is start with a dense layer
    /// note that if n_pattern == 0, all layers are SWA
    ///           if n_pattern == 1, all layers are dense
    /// example 1: n_pattern = 3, dense_first = false
    ///   il == 0: swa
    ///   il == 1: swa
    ///   il == 2: dense
    ///   il == 3: swa
    ///   il == 4: swa
    ///   il == 5: dense
    ///   il == 6: swa
    ///   etc ...
    /// example 2: n_pattern = 2, dense_first = true
    ///   il == 0: dense
    ///   il == 1: swa
    ///   il == 2: dense
    ///   il == 3: swa
    ///   etc ...
    pub fn set_swa_pattern(&mut self, n_pattern: u32, dense_first: bool) {
        if dense_first {
            for il in 0..self.n_layer() as usize {
                self.is_swa_impl[il] = u32::from(n_pattern == 0 || (il as u32 % n_pattern != 0));
            }
        } else {
            // n_pattern == 0 short-circuits before the modulo / sub (as in C++)
            for il in 0..self.n_layer() as usize {
                self.is_swa_impl[il] = u32::from(
                    n_pattern == 0 || (il as u32 % n_pattern) < n_pattern.wrapping_sub(1),
                );
            }
        }

        for il in self.n_layer() as usize..self.n_layer_all as usize {
            self.is_swa_impl[il] = 0;
        }
    }

    /// `llama_hparams::set_recr_pattern`
    pub fn set_recr_pattern(&mut self, n_pattern: u32, dense_first: bool) {
        if dense_first {
            for il in 0..self.n_layer() as usize {
                self.is_recr_impl[il] = u32::from(n_pattern == 0 || (il as u32 % n_pattern != 0));
            }
        } else {
            for il in 0..self.n_layer() as usize {
                self.is_recr_impl[il] = u32::from(
                    n_pattern == 0 || (il as u32 % n_pattern) < n_pattern.wrapping_sub(1),
                );
            }
        }

        for il in self.n_layer() as usize..self.n_layer_all as usize {
            self.is_recr_impl[il] = 0;
        }
    }

    /// `llama_hparams::is_swa_any` — return true if one of the layers is SWA
    pub fn is_swa_any(&self) -> bool {
        (0..self.n_layer_all as usize).any(|il| self.is_swa_impl[il] != 0)
    }

    /// `llama_hparams::n_head`
    pub fn n_head(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return self.n_head_arr[il];
        }
        panic!(
            "n_head: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_head_kv`
    pub fn n_head_kv(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return self.n_head_kv_arr[il];
        }
        panic!(
            "n_head_kv: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_ff`
    pub fn n_ff(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return self.n_ff_arr[il];
        }
        panic!(
            "n_ff: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_ff_exp`
    pub fn n_ff_exp(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return self.n_ff_exp_arr[il];
        }
        panic!(
            "n_ff_exp: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_expert_used`
    pub fn n_expert_used(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return self.n_expert_used_arr[il];
        }
        panic!(
            "n_expert_used: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_expert_used_max` — max n_expert_used across all layers
    pub fn n_expert_used_max(&self) -> u32 {
        (0..self.n_layer_all as usize)
            .map(|il| self.n_expert_used(il))
            .max()
            .unwrap_or(0)
    }

    /// `llama_hparams::n_gqa`
    pub fn n_gqa(&self, il: usize) -> u32 {
        let n_head = self.n_head(il);
        let n_head_kv = self.n_head_kv(il);

        if n_head_kv == 0 {
            return 0;
        }

        n_head / n_head_kv
    }

    /// `llama_hparams::n_rot`
    pub fn n_rot(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return if self.is_swa(il) {
                self.n_rot_swa
            } else {
                self.n_rot_full
            };
        }
        panic!(
            "n_rot: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_embd_inp` — dimension of main + auxiliary input embeddings
    pub fn n_embd_inp(&self) -> u32 {
        if self.n_embd_inp_impl > 0 {
            return self.n_embd_inp_impl;
        }

        let mut n_embd_inp = self.n_embd;

        if self.n_deepstack_layers > 0 {
            n_embd_inp += self.n_embd * self.n_deepstack_layers;
        }

        n_embd_inp
    }

    /// `llama_hparams::n_embd_inp_enc` — dimension of the encoder input embeddings
    pub fn n_embd_inp_enc(&self) -> u32 {
        if self.n_embd_inp_enc_impl > 0 {
            self.n_embd_inp_enc_impl
        } else {
            self.n_embd_inp()
        }
    }

    /// `llama_hparams::n_embd_out` — dimension of output embeddings
    pub fn n_embd_out(&self) -> u32 {
        if self.n_embd_out_impl > 0 {
            self.n_embd_out_impl
        } else {
            self.n_embd
        }
    }

    /// `llama_hparams::n_embd_head_k` — dimension of key embeddings for each
    /// head (per layer)
    pub fn n_embd_head_k(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return if self.is_swa(il) {
                self.n_embd_head_k_swa
            } else {
                self.n_embd_head_k_full
            };
        }
        panic!(
            "n_embd_head_k: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_embd_head_v` — dimension of value embeddings for each
    /// head (per layer)
    pub fn n_embd_head_v(&self, il: usize) -> u32 {
        if il < self.n_layer_all as usize {
            return if self.is_swa(il) {
                self.n_embd_head_v_swa
            } else {
                self.n_embd_head_v_full
            };
        }
        panic!(
            "n_embd_head_v: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_embd_k_gqa` — dimension of key embeddings across all
    /// k-v heads
    pub fn n_embd_k_gqa(&self, il: usize) -> u32 {
        let n_head_kv = self.n_head_kv(il);
        self.n_embd_head_k(il) * n_head_kv
    }

    /// `llama_hparams::n_embd_v_gqa` — dimension of value embeddings across
    /// all k-v heads
    pub fn n_embd_v_gqa(&self, il: usize) -> u32 {
        let n_head_kv = self.n_head_kv(il);
        self.n_embd_head_v(il) * n_head_kv
    }

    /// `llama_hparams::is_n_embd_k_gqa_variable` — true if any layer has a
    /// different n_embd_k_gqa
    pub fn is_n_embd_k_gqa_variable(&self) -> bool {
        let val = self.n_embd_k_gqa(0);
        (0..self.n_layer_all as usize).any(|il| val != self.n_embd_k_gqa(il))
    }

    /// `llama_hparams::is_n_embd_v_gqa_variable`
    pub fn is_n_embd_v_gqa_variable(&self) -> bool {
        let val = self.n_embd_v_gqa(0);
        (0..self.n_layer_all as usize).any(|il| val != self.n_embd_v_gqa(il))
    }

    /// `llama_hparams::n_embd_k_gqa_max`
    pub fn n_embd_k_gqa_max(&self) -> u32 {
        let mut val = self.n_embd_k_gqa(0);
        for il in 0..self.n_layer_all as usize {
            val = val.max(self.n_embd_k_gqa(il));
        }
        val
    }

    /// `llama_hparams::n_embd_v_gqa_max`
    pub fn n_embd_v_gqa_max(&self) -> u32 {
        let mut val = self.n_embd_v_gqa(0);
        for il in 0..self.n_layer_all as usize {
            val = val.max(self.n_embd_v_gqa(il));
        }
        val
    }

    /// `llama_hparams::n_embd_r` — dimension of the rolling state embeddings
    /// corresponds to Mamba's conv_states size or RWKV's token_shift states size
    pub fn n_embd_r(&self) -> u32 {
        if self.wkv_head_size != 0 {
            // for RWKV models
            return self.token_shift_count * self.n_embd;
        }

        if self.n_shortconv_l_cache != 0 {
            // for LFM2 models
            return self.n_embd * (self.n_shortconv_l_cache - 1);
        }

        if self.n_embd_head_kda != 0 {
            // for Kimi KDA layers
            // Conv state for Q, K, V: 3 * (d_conv - 1) * n_head * head_dim
            let d_inner = self.n_head(0) * self.n_embd_head_kda; // 32 * 128 = 4096
            return 3
                * (if self.ssm_d_conv > 0 {
                    self.ssm_d_conv - 1
                } else {
                    3
                })
                * d_inner;
        }

        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        // Corresponds to Mamba's conv_states size
        //
        // PLE conv history needs its own row: Meta splits cache_r_l by head, so
        // a history packed behind the first is unaddressable
        // it lives in cache_ple_r_l instead, mirrored like the rest of the PLE module
        (if self.ssm_d_conv > 0 {
            self.ssm_d_conv - 1
        } else {
            0
        }) * (self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state)
    }

    /// `llama_hparams::n_embd_s` — dimension of the recurrent state embeddings
    pub fn n_embd_s(&self) -> u32 {
        if self.wkv_head_size != 0 {
            // corresponds to RWKV's wkv_states size
            return self.n_embd * self.wkv_head_size;
        }

        if self.n_embd_head_kda != 0 {
            // for Kimi KDA layers
            // Full recurrent state: head_dim * head_dim * n_head
            // h tensor shape for delta attention: [head_dim, head_dim, n_head]
            return self.n_embd_head_kda * self.n_embd_head_kda * self.n_head(0);
            // 128 * 128 * 32 = 524288
        }

        if self.n_embd_head_la != 0 {
            // for MiniMax-Text-01 linear attention layers
            // Full recurrent state: head_dim * head_dim * n_head
            // tensor shape for linear attention: [head_dim, head_dim, n_head]
            return self.n_embd_head_la * self.n_embd_head_la * self.n_head(0); // 128 * 128 * 64 = 1048576
        }

        // corresponds to Mamba's ssm_states size
        self.ssm_d_state * self.ssm_d_inner
    }

    /// `llama_hparams::is_recr` — whether or not the given layer is recurrent
    /// (for hybrid models)
    pub fn is_recr(&self, il: usize) -> bool {
        if il < self.n_layer_all as usize {
            return self.is_recr_impl[il] != 0;
        }
        panic!(
            "is_recr: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::ple_conv_state` — PLE conv history rows:
    /// (kernel - 1) * ngram_size; 0 without a PLE module
    pub fn ple_conv_state(&self) -> u32 {
        if self.ple_n_heads == 0 || self.ple_conv_kernel == 0 {
            return 0;
        }

        // dilation equals the n-gram size, matching the reference module
        (self.ple_conv_kernel - 1) * self.ple_ngram_size * self.dsv4_hc_mult * self.n_embd
    }

    /// `llama_hparams::is_ple`
    pub fn is_ple(&self, il: usize) -> bool {
        if il < self.n_layer_all as usize {
            return self.is_ple_impl[il];
        }
        panic!(
            "is_ple: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_pos_per_embd`
    pub fn n_pos_per_embd(&self) -> u32 {
        // GGML_MROPE_SECTIONS (ggml.h:256) — the C spells the literal 4 as the
        // constant since def4d406a; same value
        if matches!(self.rope_type, LlamaRopeType::MROPE | LlamaRopeType::IMROPE) {
            ggml::ops::GGML_MROPE_SECTIONS as u32
        } else {
            1
        }
    }

    /// `llama_hparams::is_swa`
    pub fn is_swa(&self, il: usize) -> bool {
        if il < self.n_layer_all as usize {
            return self.is_swa_impl[il] != 0;
        }
        panic!(
            "is_swa: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::is_mla` — note: currently only support if either all or
    /// none of the layers are MLA
    pub fn is_mla(&self) -> bool {
        debug_assert!(
            (self.n_embd_head_k_mla_impl == 0 && self.n_embd_head_v_mla_impl == 0)
                || (self.n_embd_head_k_mla_impl != 0 && self.n_embd_head_v_mla_impl != 0)
        );

        self.n_embd_head_k_mla_impl != 0 && self.n_embd_head_v_mla_impl != 0
    }

    /// `llama_hparams::is_indexer_full`
    pub fn is_indexer_full(&self, il: usize) -> bool {
        if il < self.n_layer() as usize {
            return self.is_indexer_full_impl[il] != 0;
        }
        panic!(
            "is_indexer_full: il ({il}) out of bounds (n_layer: {})",
            self.n_layer()
        );
    }

    /// `llama_hparams::n_embd_head_k_mla`
    pub fn n_embd_head_k_mla(&self) -> u32 {
        if self.is_mla() {
            self.n_embd_head_k_mla_impl
        } else {
            self.n_embd_head_k(0)
        }
    }

    /// `llama_hparams::n_embd_head_v_mla`
    pub fn n_embd_head_v_mla(&self) -> u32 {
        if self.is_mla() {
            self.n_embd_head_v_mla_impl
        } else {
            self.n_embd_head_v(0)
        }
    }

    /// `llama_hparams::has_kv`
    pub fn has_kv(&self, il: usize) -> bool {
        if self.n_layer_kv_from_start >= 0 {
            if il < self.n_layer_kv_from_start as usize {
                return true;
            }

            return false;
        }

        // by default, all layers have kv
        true
    }

    /// `llama_hparams::has_rope`
    pub fn has_rope(&self, il: usize) -> bool {
        // the router layer stores adapter routing signal, not positional info,
        // so it must not be RoPE-shifted
        if self.router_layer >= 0 && il as i32 == self.router_layer {
            return false;
        }

        if il < self.n_layer_all as usize {
            return self.rope_pattern[il] != 0;
        }
        panic!(
            "has_rope: il ({il}) out of bounds (n_layer_all: {})",
            self.n_layer_all
        );
    }

    /// `llama_hparams::n_layer` — number of effective layers (excludes nextn
    /// layers)
    pub fn n_layer(&self) -> u32 {
        self.n_layer_all - self.n_layer_nextn
    }

    /// `llama_hparams::is_masked_swa` (static)
    ///
    /// note that this function uses different SWA parameters from those in the
    /// hparams
    // TODO: think of a better place for this function
    // TODO: pack the SWA params in a struct?
    pub fn is_masked_swa(n_swa: u32, swa_type: LlamaSwaType, p0: i32, p1: i32) -> bool {
        debug_assert!(p0 >= 0 && p1 >= 0);

        match swa_type {
            LlamaSwaType::NONE => {}
            LlamaSwaType::STANDARD => {
                if p1 - p0 >= n_swa as i32 {
                    return true;
                }
            }
            LlamaSwaType::CHUNKED => {
                let pos_chunk_start = (p1 / n_swa as i32) * n_swa as i32;

                if p0 < pos_chunk_start {
                    return true;
                }
            }
            LlamaSwaType::SYMMETRIC => {
                let half_n_swa = n_swa as i32 / 2;
                let pos_diff = p1 - p0;

                // Mask if outside the symmetric window
                if pos_diff < -half_n_swa || pos_diff > half_n_swa {
                    return true;
                }
            }
        }

        false
    }

    /// `llama_hparams::use_mrope`
    pub fn use_mrope(&self) -> bool {
        self.rope_sections[0] > 0 && self.rope_sections[1] > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen25_hparams() -> LlamaHparams {
        // values confirmed against the real qwen2.5-0.5b-instruct GGUF (see meta.rs e2e)
        let mut h = LlamaHparams::new();
        h.n_layer_all = 24;
        h.n_embd = 896;
        h.n_ctx_train = 32768;
        h.n_head_arr.iter_mut().take(24).for_each(|v| *v = 14);
        h.n_head_kv_arr.iter_mut().take(24).for_each(|v| *v = 2);
        h.n_ff_arr.iter_mut().take(24).for_each(|v| *v = 4864);
        h.n_embd_head_k_full = 64;
        h.n_embd_head_v_full = 64;
        h.n_rot_full = 64;
        h.rope_freq_base_train = 1000000.0;
        h
    }

    /// P4 regression (agent N): the rope runtime derivation must reproduce
    /// llama-context.cpp:106-215 — Phi-4-mini values from its GGUF.
    #[test]
    fn rope_runtime_matches_reference_wiring() {
        let mut h = LlamaHparams::default();
        h.n_ctx_train = 131072;
        h.n_ctx_orig_yarn = 4096;
        h.rope_attn_factor = 1.190238118171692;
        h.yarn_ext_factor = -1.0; // "not set" (llama-context.cpp:173)
        h.rope_freq_scale_train = 1.0;
        h.rope_scaling_type_train = LlamaRopeScalingType::LONGROPE;
        let rt = h.rope_runtime();
        assert_eq!(rt.n_ctx_orig_yarn, 4096); // NOT n_ctx_train
        assert_eq!(rt.ext_factor, 0.0); // -1 -> non-YARN -> 0 (llama-context.cpp:173-175)
        assert!(
            (rt.attn_factor - 1.1902381).abs() < 1e-6,
            "attn_factor = {}",
            rt.attn_factor
        );
        assert_eq!(rt.freq_scale, 1.0);
        assert_eq!((rt.beta_fast, rt.beta_slow), (32.0, 1.0));
        // YARN scaling with ext_factor unset -> ext 1.0 and the mscale path
        let mut y = LlamaHparams::default();
        y.rope_scaling_type_train = LlamaRopeScalingType::YARN;
        y.yarn_ext_factor = -1.0;
        y.rope_freq_scale_train = 0.25; // factor = 4
        let rty = y.rope_runtime();
        assert_eq!(rty.ext_factor, 1.0);
        // get_mscale(4,1)/ (1 + 0.1*ln(4)) = (0.1*ln4+1)/(1+0.1*ln4) = 1.0
        assert!((rty.attn_factor - 1.0).abs() < 1e-6, "{}", rty.attn_factor);
    }

    #[test]
    fn default_initializers_match_c() {
        let h = LlamaHparams::new();
        // explicit default member initializers from llama-hparams.h
        assert_eq!(h.n_layer_kv_from_start, -1);
        assert_eq!(h.expert_group_scale, 0.05);
        assert_eq!(h.f_attn_logit_softcapping, 50.0);
        assert_eq!(h.f_router_logit_softcapping, 30.0);
        assert_eq!(h.f_final_logit_softcapping, 30.0);
        assert_eq!(h.token_shift_count, 2);
        assert_eq!(h.rope_freq_base_train_swa, 10000.0);
        assert_eq!(h.rope_freq_scale_train_swa, 1.0);
        assert_eq!(h.yarn_ext_factor, -1.0);
        assert_eq!(h.yarn_attn_factor, 1.0);
        assert_eq!(h.yarn_beta_fast, 32.0);
        assert_eq!(h.yarn_beta_slow, 1.0);
        assert_eq!(h.swa_type, LlamaSwaType::NONE);
        assert_eq!(h.non_causal_type, LlamaNonCausalType::ALL);
        assert!(h.causal_attn);
        assert_eq!(h.n_cls_out, 1);
        assert_eq!(h.n_no_rope_layer_step, 4);
        assert_eq!(h.n_altup, 4);
        assert_eq!(h.laurel_rank, 64);
        assert_eq!(h.n_embd_altup, 256);
        assert_eq!(h.kda_gate_lower_bound, f32::NEG_INFINITY);
        assert_eq!(h.situ_beta, 1.0);
        assert_eq!(h.situ_linear_beta, 0.0);
        // zero/one-filled arrays (loader prologue semantics)
        assert_eq!(h.n_head_arr.len(), LLAMA_MAX_LAYERS);
        assert_eq!(h.rope_pattern, vec![1u32; LLAMA_MAX_LAYERS]);
        assert_eq!(h.is_swa_impl, vec![0u32; LLAMA_MAX_LAYERS]);
        assert_eq!(h.deepstack_mapping_arr, vec![-1i32; LLAMA_MAX_LAYERS]);
    }

    #[test]
    fn qwen25_calculators() {
        let h = qwen25_hparams();
        assert_eq!(h.n_layer(), 24);
        assert_eq!(h.n_head(0), 14);
        assert_eq!(h.n_head_kv(23), 2);
        assert_eq!(h.n_gqa(0), 7);
        assert_eq!(h.n_ff(5), 4864);
        assert_eq!(h.n_embd_head_k(0), 64);
        assert_eq!(h.n_embd_k_gqa(0), 128);
        assert_eq!(h.n_embd_v_gqa(3), 128);
        assert_eq!(h.n_rot(11), 64);
        assert!(!h.is_swa_any());
        assert!(h.has_kv(0));
        assert!(h.has_rope(0));
        assert!(!h.is_mla());
        assert_eq!(h.n_embd_out(), h.n_embd);
        assert_eq!(h.n_embd_inp(), h.n_embd);
        assert!(!h.is_n_embd_k_gqa_variable());
    }

    #[test]
    fn swa_pattern_examples() {
        // example 1: n_pattern = 3, dense_first = false
        let mut h = LlamaHparams::new();
        h.n_layer_all = 7;
        h.set_swa_pattern(3, false);
        let expect = [1, 1, 0, 1, 1, 0, 1]; // swa swa dense swa swa dense swa
        assert_eq!(&h.is_swa_impl[..7], &expect);
        assert!(h.is_swa_any());

        // example 2: n_pattern = 2, dense_first = true
        h.set_swa_pattern(2, true);
        let expect = [0, 1, 0, 1, 0, 1, 0]; // dense swa dense swa ...
        assert_eq!(&h.is_swa_impl[..7], &expect);

        // n_pattern == 0 -> all SWA; n_pattern == 1 -> all dense
        h.set_swa_pattern(0, false);
        assert!(h.is_swa_impl.iter().take(7).all(|&v| v == 1));
        h.set_swa_pattern(1, false);
        assert!(h.is_swa_impl.iter().take(7).all(|&v| v == 0));

        // nextn layers stay dense
        h.n_layer_nextn = 1;
        h.set_swa_pattern(0, false);
        assert_eq!(h.is_swa_impl[6], 0);
        assert_eq!(h.n_layer(), 6);
    }

    #[test]
    fn is_masked_swa_cases() {
        use LlamaSwaType::*;
        // NONE never masks
        assert!(!LlamaHparams::is_masked_swa(4, NONE, 0, 100));
        // STANDARD: masked iff p1 - p0 >= n_swa
        assert!(!LlamaHparams::is_masked_swa(4, STANDARD, 0, 3));
        assert!(LlamaHparams::is_masked_swa(4, STANDARD, 0, 4));
        // CHUNKED: p0 outside p1's chunk
        assert!(!LlamaHparams::is_masked_swa(4, CHUNKED, 4, 7));
        assert!(LlamaHparams::is_masked_swa(4, CHUNKED, 3, 4));
        // SYMMETRIC: |p1 - p0| <= n_swa/2
        assert!(!LlamaHparams::is_masked_swa(4, SYMMETRIC, 0, 2));
        assert!(LlamaHparams::is_masked_swa(4, SYMMETRIC, 0, 3));
        assert!(LlamaHparams::is_masked_swa(4, SYMMETRIC, 5, 2));
    }

    #[test]
    fn kv_from_start_and_router_layer() {
        let mut h = LlamaHparams::new();
        h.n_layer_all = 4;
        h.rope_pattern.iter_mut().for_each(|v| *v = 1);
        h.n_layer_kv_from_start = 2;
        assert!(h.has_kv(1));
        assert!(!h.has_kv(2));
        h.router_layer = 1;
        assert!(!h.has_rope(1));
        assert!(h.has_rope(0));
    }

    #[test]
    fn mrope_and_pos_per_embd() {
        let mut h = LlamaHparams::new();
        assert_eq!(h.n_pos_per_embd(), 1);
        h.rope_type = LlamaRopeType::MROPE;
        assert_eq!(h.n_pos_per_embd(), 4);
        h.rope_sections = [16, 16, 0, 0];
        assert!(h.use_mrope());
        h.rope_sections = [0, 16, 16, 0];
        assert!(!h.use_mrope());
    }

    #[test]
    fn recurrent_state_dims() {
        // RWKV rolling state: token_shift_count * n_embd
        let mut h = LlamaHparams::new();
        h.n_layer_all = 1;
        h.n_embd = 64;
        h.wkv_head_size = 32;
        h.token_shift_count = 2;
        assert_eq!(h.n_embd_r(), 128);
        assert_eq!(h.n_embd_s(), 64 * 32);

        // LFM2 rolling state: n_embd * (n_shortconv_l_cache - 1)
        let mut h = LlamaHparams::new();
        h.n_layer_all = 1;
        h.n_embd = 64;
        h.n_shortconv_l_cache = 4;
        assert_eq!(h.n_embd_r(), 64 * 3);

        // Mamba rolling/recurrent state
        let mut h = LlamaHparams::new();
        h.n_layer_all = 1;
        h.ssm_d_conv = 4;
        h.ssm_d_inner = 100;
        h.ssm_n_group = 2;
        h.ssm_d_state = 16;
        assert_eq!(h.n_embd_r(), 3 * (100 + 2 * 2 * 16));
        assert_eq!(h.n_embd_s(), 16 * 100);
    }
}
