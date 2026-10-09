//! saver.rs — port of `src/llama-model-saver.cpp` (llama-model-saver.cpp:16-505,
//! pinned bd4f514db1) + the public entry `llama_model_save_to_file`
//! (llama.cpp:498-503).
//!
//! Mapping:
//!   llama_model_saver_supports_arch     -> [`supports_arch`]
//!   llama_model_saver ctor/dtor         -> [`LlamaModelSaver::new`] (the
//!                                           gguf_ctx ctor at :34 is the
//!                                           `with_writer` half; the port has
//!                                           no gguf_context to borrow)
//!   add_kv overloads (:43-72)           -> [`LlamaModelSaver::add_kv_*`]
//!   add_kv template + vector<string>    -> [`LlamaModelSaver::add_kv_arr_u32`]
//!                                           / `_i32` / `_f32` / `_u64` /
//!                                           `_strings` (+ the `per_layer`
//!                                           collapse at :84-96)
//!   add_tensor (:131-143)               -> [`LlamaModelSaver::add_tensor`]
//!   add_kv_from_model (:145-449)        -> [`LlamaModelSaver::add_kv_from_model`]
//!   add_tensors_from_model (:451-497)   -> [`LlamaModelSaver::add_tensors_from_model`]
//!   save (:499-505)                     -> [`LlamaModelSaver::save`]
//!   llama_model_save_to_file            -> [`save_model_to_file`]
//!
//! Byte-exactness contract: the KV order mirrors add_kv_from_model's call
//! order line for line, the tensor order mirrors add_tensors_from_model's
//! member order (the C iterates the raw `llama_layer` memory as a
//! `ggml_tensor*` array — declaration order of llama-model.h:254-343), and the
//! file bytes come from the port's byte-exact [`ggml::gguf_write::GgufWriter`]
//! (verified against the reference writer in gguf_write.rs's
//! `gguf_write_bit_exact_vs_reference`).

use std::collections::HashSet;
use std::io::Write;

use ggml::gguf::GgufType;
use ggml::gguf_write::GgufWriter;
use ggml::tensor::TensorId;
use ggml::Value;

use crate::arch::{kv_name, LlmArch, LlmKv};
use crate::display::rope_scaling_type_name;
use crate::hparams::LlamaSwaType;
use crate::model::LlamaModel;
use crate::vocab::{
    TokenData, Vocab, VocabType, ATTR_BYTE, ATTR_CONTROL, ATTR_NORMAL, ATTR_UNKNOWN, ATTR_UNUSED,
    ATTR_USER_DEFINED, TOKEN_NULL,
};

/// `llama_model_saver_supports_arch` (llama-model-saver.cpp:16-27 +
/// a7b94df2c's CLEF case — the head tensors are not saved).
pub fn supports_arch(arch: LlmArch) -> bool {
    !matches!(
        arch,
        LlmArch::GEMMA3N
            | LlmArch::BITNET
            | LlmArch::T5
            | LlmArch::APERTUS
            | LlmArch::STEP35
            // the clef decision head's tensors are not saved
            // (llama-model-saver.cpp:23, a7b94df2c)
            | LlmArch::CLEF
    )
}

/// `llama_token_type` values written for the token-attribute table
/// (include/llama.h:92-100).
const TOKEN_TYPE_UNDEFINED: i32 = 0;
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_UNUSED: i32 = 5;
const TOKEN_TYPE_BYTE: i32 = 6;

pub struct LlamaModelSaver<'a> {
    model: &'a LlamaModel,
    pub w: GgufWriter,
}

impl<'a> LlamaModelSaver<'a> {
    /// `llama_model_saver::llama_model_saver(const llama_model *)`
    /// (llama-model-saver.cpp:29-32) — `gguf_init_empty()` starts a writer at
    /// the default alignment.
    pub fn new(model: &'a LlamaModel) -> Self {
        assert!(supports_arch(model.arch), "llama_model_saver: arch not supported");
        LlamaModelSaver {
            model,
            w: GgufWriter::new(ggml::gguf::GGUF_DEFAULT_ALIGNMENT),
        }
    }

    fn key(&self, kv: LlmKv) -> String {
        kv_name(self.model.arch, kv)
    }

    // ---- add_kv overloads (llama-model-saver.cpp:43-72) ----

    pub fn add_kv_u32(&mut self, kv: LlmKv, v: u32) {
        self.w.set_kv(&self.key(kv), Value::U32(v));
    }
    pub fn add_kv_i32(&mut self, kv: LlmKv, v: i32) {
        self.w.set_kv(&self.key(kv), Value::I32(v));
    }
    pub fn add_kv_u64(&mut self, kv: LlmKv, v: u64) {
        self.w.set_kv(&self.key(kv), Value::U64(v));
    }
    pub fn add_kv_f32(&mut self, kv: LlmKv, v: f32) {
        self.w.set_kv(&self.key(kv), Value::F32(v));
    }
    pub fn add_kv_bool(&mut self, kv: LlmKv, v: bool) {
        self.w.set_kv(&self.key(kv), Value::Bool(v));
    }
    pub fn add_kv_str(&mut self, kv: LlmKv, v: &str) {
        self.w.set_kv(&self.key(kv), Value::String(v.to_string()));
    }

    // ---- the typed-array template (llama-model-saver.cpp:74-117) ----
    //
    // `n_values == 0` returns early (:80-82); the `per_layer` collapse
    // (:84-96) writes a scalar when every one of the first n_layer values is
    // identical.

    fn add_kv_arr<T: Clone + PartialEq + 'static>(
        &mut self,
        kv: LlmKv,
        mk: impl Fn(T) -> Value,
        value: &[T],
        per_layer: bool,
    ) {
        let n_values = if per_layer {
            self.model.hparams.n_layer_all as usize
        } else {
            value.len()
        };
        assert!(n_values <= value.len());
        if n_values == 0 {
            return;
        }
        if per_layer {
            let all_same = value[1..n_values].iter().all(|v| *v == value[0]);
            if all_same {
                self.w.set_kv(&self.key(kv), mk(value[0].clone()));
                return;
            }
        }
        self.w.set_kv(
            &self.key(kv),
            Value::Array(
                arr_type::<T>(),
                value[..n_values].iter().map(|v| mk(v.clone())).collect(),
            ),
        );
    }

    pub fn add_kv_arr_u32(&mut self, kv: LlmKv, v: &[u32], per_layer: bool) {
        self.add_kv_arr(kv, Value::U32, v, per_layer);
    }
    pub fn add_kv_arr_i32(&mut self, kv: LlmKv, v: &[i32], per_layer: bool) {
        self.add_kv_arr(kv, Value::I32, v, per_layer);
    }
    pub fn add_kv_arr_f32(&mut self, kv: LlmKv, v: &[f32], per_layer: bool) {
        self.add_kv_arr(kv, Value::F32, v, per_layer);
    }
    pub fn add_kv_arr_u64(&mut self, kv: LlmKv, v: &[u64], per_layer: bool) {
        self.add_kv_arr(kv, Value::U64, v, per_layer);
    }
    pub fn add_kv_arr_bool(&mut self, kv: LlmKv, v: &[bool], per_layer: bool) {
        self.add_kv_arr(kv, Value::Bool, v, per_layer);
    }

    /// `add_kv(key, const std::vector<std::string> &)` (llama-model-saver.cpp:123-129)
    /// — note the C overload has *no* empty-array early return (an empty
    /// `tokenizer.ggml.merges` array is written for non-BPE vocabularies).
    pub fn add_kv_arr_strings(&mut self, kv: LlmKv, v: &[String]) {
        self.w.set_kv(
            &self.key(kv),
            Value::Array(GgufType::String, v.iter().map(|s| Value::String(s.clone())).collect()),
        );
    }

    // ---- add_tensor (llama-model-saver.cpp:131-143) ----

    /// Add a model tensor to the output table. A `None` is the C nullptr
    /// no-op; an already-present name must be one of the rope-factor FIXME
    /// tensors or the C's assert fires.
    pub fn add_tensor(&mut self, tensor: Option<TensorId>) {
        let Some(id) = tensor else { return };
        let name = self.model.ctx.name(id).to_string();
        if self.w.tensors.iter().any(|t| t.name == name) {
            assert!(
                name == "rope_freqs.weight"
                    || name == "rope_factors_long.weight"
                    || name == "rope_factors_short.weight", // FIXME (llama-model-saver.cpp:138-139)
            );
            return;
        }
        let ne = *self.model.ctx.ne(id);
        let ty = self.model.ctx.ty(id);
        self.w.add_tensor(&name, ty, ne);
    }

    // ---- add_kv_from_model (llama-model-saver.cpp:145-449) ----

    pub fn add_kv_from_model(&mut self, vocab: &Vocab) {
        let hparams = &self.model.hparams;

        // the tokenizer tables (:149-176)
        let n_vocab = vocab.id_to_token.len() as i32;
        let mut tokens: Vec<String> = vec![String::new(); n_vocab as usize];
        let mut scores: Vec<f32> = vec![0.0; n_vocab as usize];
        let mut token_types: Vec<i32> = vec![TOKEN_TYPE_UNDEFINED; n_vocab as usize];
        if vocab.ty != VocabType::None {
            for (id, TokenData { text, score, attr }) in vocab.id_to_token.iter().enumerate() {
                tokens[id] = text.clone();
                scores[id] = *score;
                // FIXME should this be treated as flags? (:161-174 — exact
                // switch on the whole attr word, everything else UNDEFINED)
                token_types[id] = match *attr {
                    ATTR_UNKNOWN => TOKEN_TYPE_UNKNOWN,
                    ATTR_UNUSED => TOKEN_TYPE_UNUSED,
                    ATTR_NORMAL => TOKEN_TYPE_NORMAL,
                    ATTR_CONTROL => TOKEN_TYPE_CONTROL,
                    ATTR_USER_DEFINED => TOKEN_TYPE_USER_DEFINED,
                    ATTR_BYTE => TOKEN_TYPE_BYTE,
                    _ => TOKEN_TYPE_UNDEFINED,
                };
            }
        }

        self.add_kv_str(LlmKv::GENERAL_ARCHITECTURE, self.model.arch.name()); // :179
        self.add_kv_str(LlmKv::GENERAL_NAME, &self.model.name); // :195

        // the per-tensor activation precision policy (llama-model-saver.cpp:
        // 196-209, e9f824d8c) — round-trips `general.tensor_extra.{name,prec_a4}`.
        // The C iterates its `unordered_map` (unspecified order); the port's
        // BTreeMap writes name-sorted, and the loader accepts any order.
        if !self.model.prec_policy.prec_src1.is_empty() {
            let tensor_names: Vec<String> =
                self.model.prec_policy.prec_src1.keys().cloned().collect();
            let values: Vec<bool> = self
                .model
                .prec_policy
                .prec_src1
                .values()
                .map(|&prec| prec != crate::model::GGML_PREC_Q8)
                .collect();
            self.add_kv_arr_strings(LlmKv::GENERAL_TENSOR_EXTRA_NAME, &tensor_names);
            self.add_kv_arr_bool(LlmKv::GENERAL_TENSOR_EXTRA_PREC_A4, &values, false);
        }
        self.add_kv_u32(LlmKv::VOCAB_SIZE, vocab.id_to_token.len() as u32); // :204
        self.add_kv_u32(LlmKv::CONTEXT_LENGTH, hparams.n_ctx_train); // :205
        self.add_kv_u32(LlmKv::EMBEDDING_LENGTH, hparams.n_embd); // :206
        if hparams.n_embd_out_impl > 0 {
            self.add_kv_u32(LlmKv::EMBEDDING_LENGTH_OUT, hparams.n_embd_out_impl); // :207-209
        }
        self.add_kv_u32(LlmKv::BLOCK_COUNT, hparams.n_layer_all); // :210
        self.add_kv_u32(LlmKv::LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead); // :211
        self.add_kv_arr_u32(LlmKv::FEED_FORWARD_LENGTH, &hparams.n_ff_arr, true); // :212
        self.add_kv_u32(LlmKv::EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp(0)); // :213
        self.add_kv_u32(LlmKv::EXPERT_LATENT_LENGTH, hparams.n_expert_latent); // :214
        self.add_kv_u32(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp); // :215
        self.add_kv_u32(LlmKv::EXPERT_CHUNK_FEED_FORWARD_LENGTH, hparams.n_ff_chexp); // :216
        let n_layer_all = hparams.n_layer_all as usize;
        self.add_kv_arr_f32(
            // :217-218
            LlmKv::SWIGLU_CLAMP_EXP,
            &hparams.swiglu_clamp_exp[..n_layer_all.min(hparams.swiglu_clamp_exp.len())],
            false,
        );
        self.add_kv_arr_f32(
            // :219-220
            LlmKv::SWIGLU_CLAMP_SHEXP,
            &hparams.swiglu_clamp_shexp[..n_layer_all.min(hparams.swiglu_clamp_shexp.len())],
            false,
        );
        self.add_kv_bool(LlmKv::USE_PARALLEL_RESIDUAL, hparams.use_par_res); // :221
        self.add_kv_u32(LlmKv::EXPERT_COUNT, hparams.n_expert); // :223
        self.add_kv_u32(LlmKv::EXPERT_USED_COUNT, hparams.n_expert_used(0)); // :224
        self.add_kv_u32(LlmKv::EXPERT_SHARED_COUNT, hparams.n_expert_shared); // :225
        self.add_kv_u32(LlmKv::EXPERT_GROUP_COUNT, hparams.n_expert_groups); // :226
        self.add_kv_u32(LlmKv::EXPERT_GROUP_USED_COUNT, hparams.n_group_used); // :227
        self.add_kv_f32(LlmKv::EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale); // :228
        self.add_kv_bool(LlmKv::EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm); // :229
        self.add_kv_u32(LlmKv::EXPERT_GATING_FUNC, hparams.expert_gating_func); // :230
        self.add_kv_f32(LlmKv::EXPERT_GROUP_SCALE, hparams.expert_group_scale); // :231
        self.add_kv_u32(LlmKv::EXPERTS_PER_GROUP, hparams.n_group_experts); // :232
        self.add_kv_u32(LlmKv::MOE_EVERY_N_LAYERS, hparams.moe_every_n_layers); // :233
        self.add_kv_u32(LlmKv::NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn); // :234
        self.add_kv_u32(LlmKv::NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers); // :235
        self.add_kv_arr_i32(LlmKv::DEEPSTACK_MAPPING, &hparams.deepstack_mapping_arr, false); // :236
        self.add_kv_u32(LlmKv::POOLING_TYPE, hparams.pooling_type as u32); // :237
        self.add_kv_f32(LlmKv::LOGIT_SCALE, hparams.f_logit_scale); // :238
        self.add_kv_i32(LlmKv::DECODER_START_TOKEN_ID, hparams.dec_start_token_id); // :239
        self.add_kv_u32(LlmKv::DECODER_BLOCK_COUNT, hparams.dec_n_layer); // :240
        self.add_kv_f32(LlmKv::ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping); // :241
        self.add_kv_f32(LlmKv::ROUTER_LOGIT_SOFTCAPPING, hparams.f_router_logit_softcapping); // :242
        self.add_kv_f32(LlmKv::FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping); // :243
        self.add_kv_bool(LlmKv::SWIN_NORM, hparams.swin_norm); // :244
        self.add_kv_u32(LlmKv::RESCALE_EVERY_N_LAYERS, hparams.rescale_every_n_layers); // :245
        self.add_kv_u32(LlmKv::TIME_MIX_EXTRA_DIM, hparams.time_mix_extra_dim); // :246
        self.add_kv_u32(LlmKv::TIME_DECAY_EXTRA_DIM, hparams.time_decay_extra_dim); // :247
        self.add_kv_f32(LlmKv::RESIDUAL_SCALE, hparams.f_residual_scale); // :248
        self.add_kv_f32(LlmKv::EMBEDDING_SCALE, hparams.f_embedding_scale); // :249
        self.add_kv_u32(LlmKv::HRM_LAYERS_PER_STACK, hparams.n_hrm_layers_per_stack); // :250
        self.add_kv_u32(LlmKv::HRM_H_CYCLES, hparams.n_hrm_h_cycles); // :251
        self.add_kv_u32(LlmKv::HRM_L_CYCLES, hparams.n_hrm_l_cycles); // :252
        self.add_kv_bool(LlmKv::HRM_PREFIX_LM, hparams.hrm_prefix_lm); // :253
        self.add_kv_u32(LlmKv::TOKEN_SHIFT_COUNT, hparams.token_shift_count); // :254
        self.add_kv_u32(LlmKv::INTERLEAVE_MOE_LAYER_STEP, hparams.n_moe_layer_step); // :255

        self.add_kv_arr_u32(LlmKv::ATTENTION_HEAD_COUNT, &hparams.n_head_arr, true); // :258
        self.add_kv_arr_u32(LlmKv::ATTENTION_HEAD_COUNT_KV, &hparams.n_head_kv_arr, true); // :259
        self.add_kv_f32(LlmKv::ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias); // :260
        self.add_kv_f32(LlmKv::ATTENTION_CLAMP_KQV, hparams.f_clamp_kqv); // :261
        self.add_kv_u32(LlmKv::ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full); // :262
        self.add_kv_u32(LlmKv::ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full); // :263
        self.add_kv_f32(LlmKv::ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps); // :264
        self.add_kv_f32(LlmKv::ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps); // :265
        self.add_kv_f32(LlmKv::ATTENTION_GROUPNORM_EPS, hparams.f_norm_group_eps); // :266
        self.add_kv_u32(LlmKv::ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups); // :267
        // MoVA value experts (462524043, llama-model-saver.cpp:282-283)
        self.add_kv_u32(LlmKv::ATTENTION_VALUE_EXPERT_COUNT, hparams.n_value_expert);
        self.add_kv_u32(
            LlmKv::ATTENTION_VALUE_EXPERT_USED_COUNT,
            hparams.n_value_expert_used,
        );
        self.add_kv_bool(LlmKv::ATTENTION_CAUSAL, hparams.causal_attn); // :268
        self.add_kv_u32(LlmKv::ATTENTION_Q_LORA_RANK, hparams.n_lora_q); // :269
        self.add_kv_u32(LlmKv::ATTENTION_KV_LORA_RANK, hparams.n_lora_kv); // :270
        self.add_kv_u32(LlmKv::ATTENTION_DECAY_LORA_RANK, hparams.n_lora_decay); // :271
        self.add_kv_u32(LlmKv::ATTENTION_ICLR_LORA_RANK, hparams.n_lora_iclr); // :272
        self.add_kv_u32(
            // :273
            LlmKv::ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK,
            hparams.n_lora_value_res_mix,
        );
        self.add_kv_u32(LlmKv::ATTENTION_GATE_LORA_RANK, hparams.n_lora_gate); // :274
        self.add_kv_u32(LlmKv::ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts); // :275
        self.add_kv_arr_u32(LlmKv::ATTENTION_ROPE_PATTERN, &hparams.rope_pattern, true); // :276
        self.add_kv_u32(LlmKv::ATTENTION_SLIDING_WINDOW, hparams.n_swa); // :277
        if hparams.swa_type != LlamaSwaType::NONE {
            // never collapsed to a scalar: the loaders read a scalar as a period
            // (:278-282)
            self.add_kv_arr_u32(
                LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN,
                &hparams.is_swa_impl[..n_layer_all.min(hparams.is_swa_impl.len())],
                false,
            );
        }
        self.add_kv_f32(LlmKv::ATTENTION_SCALE, hparams.f_attention_scale); // :283
        self.add_kv_f32(LlmKv::ATTENTION_OUTPUT_SCALE, hparams.f_attn_out_scale); // :284
        self.add_kv_f32(LlmKv::ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale); // :285
        self.add_kv_u32(LlmKv::ATTENTION_TEMPERATURE_LENGTH, hparams.attn_temp_length); // :286
        self.add_kv_f32(LlmKv::ATTENTION_TEMPERATURE_SCALE, hparams.f_attn_temp_scale); // :287
        self.add_kv_u32(LlmKv::ATTENTION_KEY_LENGTH_MLA, hparams.n_embd_head_k_mla_impl); // :288
        self.add_kv_u32(LlmKv::ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl); // :289
        self.add_kv_u32(LlmKv::ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa); // :290
        self.add_kv_u32(LlmKv::ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa); // :291
        self.add_kv_u32(LlmKv::ATTENTION_KEY_LENGTH_MLA_SWA, hparams.n_embd_head_k_mla_swa); // :292
        self.add_kv_u32(LlmKv::ATTENTION_VALUE_LENGTH_MLA_SWA, hparams.n_embd_head_v_mla_swa); // :293
        self.add_kv_u32(LlmKv::ATTENTION_KV_LORA_RANK_SWA, hparams.n_lora_kv_swa); // :294
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head); // :295
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size); // :296
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k); // :297
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_BLOCK_SIZE, hparams.indexer_block_size); // :298
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_KPOOL, hparams.indexer_kpool); // :299
        self.add_kv_bool(
            LlmKv::ATTENTION_INDEXER_KPOOL_SELECT_TAIL,
            hparams.indexer_kpool_select_tail,
        ); // :300
        self.add_kv_u32(LlmKv::ATTENTION_INDEXER_LOCAL_BLOCKS, hparams.indexer_local_blocks); // :301
        self.add_kv_arr_u32(LlmKv::ATTENTION_INDEXER_TYPES, &hparams.is_indexer_full_impl, true); // :300
        self.add_kv_arr_u32(LlmKv::ATTENTION_RECURRENT_LAYERS, &hparams.is_recr_impl, true); // :301
        self.add_kv_u32(LlmKv::ATTENTION_OUTPUT_GROUP_COUNT, hparams.dsv4_o_group_count); // :302
        self.add_kv_u32(LlmKv::ATTENTION_OUTPUT_LORA_RANK, hparams.dsv4_o_lora_rank); // :303
        self.add_kv_f32(LlmKv::ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base); // :304
        if self.model.arch == LlmArch::DEEPSEEK4 || hparams.dsv4_hc_mult > 0 {
            // the loader requires one compress ratio per layer, including nextn
            // layers (:305-309)
            self.add_kv_arr_u32(
                LlmKv::ATTENTION_COMPRESS_RATIOS,
                &hparams.dsv4_compress_ratios[..n_layer_all.min(hparams.dsv4_compress_ratios.len())],
                false,
            );
        } else {
            self.add_kv_arr_u32(LlmKv::ATTENTION_COMPRESS_RATIOS, &hparams.dsv4_compress_ratios, true); // :310-312
        }
        self.add_kv_u32(LlmKv::HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult); // :313
        self.add_kv_u32(LlmKv::HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters); // :314
        self.add_kv_f32(LlmKv::HYPER_CONNECTION_EPSILON, hparams.dsv4_hc_eps); // :315
        self.add_kv_f32(LlmKv::HYPER_CONNECTION_MAGNITUDE, hparams.hc_magnitude); // :316
        self.add_kv_u32(LlmKv::HASH_LAYER_COUNT, hparams.dsv4_hash_layer_count); // :317
        self.add_kv_u32(LlmKv::HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank); // :318

        // the PLE group only means anything whole: write all of it or none
        // (:320-343)
        if hparams.ple_n_heads > 0 {
            let ple_layers: Vec<u32> = (0..n_layer_all as u32)
                .filter(|&il| hparams.is_ple_impl[il as usize])
                .collect();
            self.add_kv_arr_u32(LlmKv::PLE_LAYERS, &ple_layers, false);
            self.add_kv_u32(LlmKv::PLE_NGRAM_SIZE, hparams.ple_ngram_size);
            self.add_kv_u32(LlmKv::PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
            self.add_kv_u32(LlmKv::PLE_CONV_KERNEL, hparams.ple_conv_kernel);
            self.add_kv_u32(LlmKv::PLE_EOS_TOKEN_ID, hparams.ple_eos_token_id);
            self.add_kv_u32(LlmKv::EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
            let n_ngram = hparams.ple_ngram_size as usize;
            let n_heads = hparams.ple_n_heads as usize;
            self.add_kv_arr_u64(
                LlmKv::PLE_LAYER_MULTIPLIERS,
                &hparams.ple_layer_multipliers[..n_ngram.min(hparams.ple_layer_multipliers.len())],
                false,
            );
            // the C widens the 32-bit port fields to the u64 GGUF array
            // (ple_head_offsets / ple_head_vocab_sizes are vector<uint64_t> in
            // the C++ template instantiation, :337-342)
            self.add_kv_arr_u64(
                LlmKv::PLE_HEAD_OFFSETS,
                &hparams.ple_head_offsets[..n_heads.min(hparams.ple_head_offsets.len())]
                    .iter()
                    .map(|&v| v as u64)
                    .collect::<Vec<_>>(),
                false,
            );
            self.add_kv_arr_u64(
                LlmKv::PLE_HEAD_VOCAB_SIZES,
                &hparams.ple_head_vocab_sizes[..n_heads.min(hparams.ple_head_vocab_sizes.len())]
                    .iter()
                    .map(|&v| v as u64)
                    .collect::<Vec<_>>(),
                false,
            );
        }

        let rope_scaling_factor =
            if hparams.rope_freq_scale_train == 1.0 { 0.0 } else { 1.0 / hparams.rope_freq_scale_train }; // :345

        self.add_kv_u32(LlmKv::ROPE_DIMENSION_COUNT, hparams.n_rot_full); // :347
        self.add_kv_u32(LlmKv::ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa); // :348
        self.add_kv_arr_i32(LlmKv::ROPE_DIMENSION_SECTIONS, &hparams.rope_sections, false); // :349
        self.add_kv_f32(LlmKv::ROPE_FREQ_BASE, hparams.rope_freq_base_train); // :350
        self.add_kv_f32(LlmKv::ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa); // :351
        self.add_kv_str(
            LlmKv::ROPE_SCALING_TYPE,
            &rope_scaling_type_name(hparams.rope_scaling_type_train),
        ); // :353
        self.add_kv_f32(LlmKv::ROPE_SCALING_FACTOR, rope_scaling_factor); // :354
        self.add_kv_f32(LlmKv::ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor); // :355
        self.add_kv_u32(LlmKv::ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn); // :356
        self.add_kv_bool(LlmKv::ROPE_SCALING_FINETUNED, hparams.rope_finetuned); // :357
        self.add_kv_f32(LlmKv::ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul); // :358
        self.add_kv_f32(LlmKv::ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarn_ext_factor); // :359
        self.add_kv_f32(LlmKv::ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor); // :360
        self.add_kv_f32(LlmKv::ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast); // :361
        self.add_kv_f32(LlmKv::ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow); // :362

        self.add_kv_u32(LlmKv::SSM_INNER_SIZE, hparams.ssm_d_inner); // :369
        self.add_kv_u32(LlmKv::SSM_CONV_KERNEL, hparams.ssm_d_conv); // :370
        self.add_kv_u32(LlmKv::SSM_STATE_SIZE, hparams.ssm_d_state); // :371
        self.add_kv_u32(LlmKv::SSM_TIME_STEP_RANK, hparams.ssm_dt_rank); // :372
        self.add_kv_u32(LlmKv::SSM_GROUP_COUNT, hparams.ssm_n_group); // :373
        self.add_kv_bool(LlmKv::SSM_DT_B_C_RMS, hparams.ssm_dt_b_c_rms); // :374

        self.add_kv_u32(LlmKv::KDA_HEAD_DIM, hparams.n_embd_head_kda); // :376
        self.add_kv_bool(LlmKv::KDA_SAFE_GATE, hparams.kda_safe_gate); // :377
        self.add_kv_f32(LlmKv::KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound); // :378

        self.add_kv_u32(LlmKv::WKV_HEAD_SIZE, hparams.wkv_head_size); // :380

        self.add_kv_str(LlmKv::TOKENIZER_MODEL, &vocab.tokenizer_model); // :382
        self.add_kv_str(LlmKv::TOKENIZER_PRE, &vocab.tokenizer_pre); // :383
        self.add_kv_arr_strings(LlmKv::TOKENIZER_LIST, &tokens); // :384
        self.add_kv_arr_i32(LlmKv::TOKENIZER_TOKEN_TYPE, &token_types, false); // :385
        self.add_kv_u32(LlmKv::TOKENIZER_TOKEN_TYPE_COUNT, vocab.n_token_types); // :386
        self.add_kv_arr_f32(LlmKv::TOKENIZER_SCORES, &scores, false); // :387
        self.add_kv_arr_strings(LlmKv::TOKENIZER_MERGES, &vocab.get_bpe_merges()); // :388
        // FIXME llama_token is type i32 but u32 is expected when reading
        // (:389-396)
        if vocab.special_bos_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_BOS_ID, vocab.special_bos_id as u32);
        }
        if vocab.special_eos_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_EOS_ID, vocab.special_eos_id as u32);
        }
        if vocab.special_eot_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_EOT_ID, vocab.special_eot_id as u32);
        }
        if vocab.special_eom_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_EOM_ID, vocab.special_eom_id as u32);
        }
        if vocab.special_unk_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_UNK_ID, vocab.special_unk_id as u32);
        }
        if vocab.special_sep_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_SEP_ID, vocab.special_sep_id as u32);
        }
        if vocab.special_pad_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_PAD_ID, vocab.special_pad_id as u32);
        }
        self.add_kv_bool(LlmKv::TOKENIZER_ADD_BOS, vocab.add_bos); // :399
        self.add_kv_bool(LlmKv::TOKENIZER_ADD_EOS, vocab.add_eos); // :400
        self.add_kv_bool(LlmKv::TOKENIZER_ADD_SEP, vocab.add_sep); // :401
        self.add_kv_bool(LlmKv::TOKENIZER_ADD_PREFIX, vocab.add_space_prefix); // :402
        self.add_kv_bool(LlmKv::TOKENIZER_REMOVE_EXTRA_WS, vocab.remove_extra_whitespaces); // :403
        // TOKENIZER_PRECOMPILED_CHARSMAP (:404) — the C passes a
        // std::vector<char>, which matches none of the template's value_type
        // branches (char is a distinct type from int8_t) and hits the
        // GGML_ABORT("fatal error") fallthrough (llama-model-saver.cpp:114-115)
        // — saving a model with a precompiled charsmap aborts in the reference
        // too; mirrored as the same fatal error.
        if !vocab.precompiled_charsmap.is_empty() {
            panic!("fatal error"); // llama-model-saver.cpp:114-115 (vector<char> branch)
        }
        if vocab.special_fim_pre_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_PRE_ID, vocab.special_fim_pre_id as u32); // :407
        }
        if vocab.special_fim_suf_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_SUF_ID, vocab.special_fim_suf_id as u32); // :408
        }
        if vocab.special_fim_mid_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_MID_ID, vocab.special_fim_mid_id as u32); // :409
        }
        if vocab.special_fim_pad_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_PAD_ID, vocab.special_fim_pad_id as u32); // :410
        }
        if vocab.special_fim_rep_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_REP_ID, vocab.special_fim_rep_id as u32); // :411
        }
        if vocab.special_fim_sep_id != TOKEN_NULL {
            self.add_kv_u32(LlmKv::TOKENIZER_FIM_SEP_ID, vocab.special_fim_sep_id as u32); // :412
        }

        self.add_kv_u32(LlmKv::POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd); // :421
        self.add_kv_u32(LlmKv::POSNET_BLOCK_COUNT, hparams.posnet.n_layer); // :422

        self.add_kv_u32(LlmKv::CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd); // :424
        self.add_kv_u32(LlmKv::CONVNEXT_BLOCK_COUNT, hparams.convnext.n_layer); // :425

        self.add_kv_arr_strings(LlmKv::CLASSIFIER_OUTPUT_LABELS, &self.model.classifier_labels); // :427

        self.add_kv_u32(LlmKv::SHORTCONV_L_CACHE, hparams.n_shortconv_l_cache); // :429

        self.add_kv_arr_f32(LlmKv::XIELU_ALPHA_N, &hparams.xielu_alpha_n, false); // :431
        self.add_kv_arr_f32(LlmKv::XIELU_ALPHA_P, &hparams.xielu_alpha_p, false); // :432
        self.add_kv_arr_f32(LlmKv::XIELU_BETA, &hparams.xielu_beta, false); // :433
        self.add_kv_arr_f32(LlmKv::XIELU_EPS, &hparams.xielu_eps, false); // :434

        self.add_kv_u32(LlmKv::ATTN_RES_BLOCK_SIZE, hparams.attn_res_block_size); // :436
        self.add_kv_f32(LlmKv::ACTIVATION_SITU_BETA, hparams.situ_beta); // :437
        self.add_kv_f32(LlmKv::ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta); // :438

        self.add_kv_u32(LlmKv::DENSE_2_FEAT_IN, hparams.dense_2_feat_in); // :445
        self.add_kv_u32(LlmKv::DENSE_2_FEAT_OUT, hparams.dense_2_feat_out); // :446
        self.add_kv_u32(LlmKv::DENSE_3_FEAT_IN, hparams.dense_3_feat_in); // :447
        self.add_kv_u32(LlmKv::DENSE_3_FEAT_OUT, hparams.dense_3_feat_out); // :448
    }

    // ---- add_tensors_from_model (llama-model-saver.cpp:451-497) ----

    pub fn add_tensors_from_model(&mut self) {
        let m = self.model;
        // some models use the same tensor for tok_embd and output (:452-455)
        if m.ctx.name(m.output) != m.ctx.name(m.tok_embd) {
            self.add_tensor(Some(m.tok_embd));
        }
        // model-level members in the C's call order (:456-482). Members the
        // port never creates for a loadable arch (always nullptr in C for
        // them) are skipped: output_norm_enc/output_s/output_in_s (T5/dflash/
        // NVFP4 sidecars), nextn_proj_pre/post (gemma4-assistant), cls_norm
        // (modern-bert) — see FILE_MAP.
        self.add_tensor(m.token_types); // type_embd (:456)
        self.add_tensor(m.position_embd); // pos_embd (:457)
        self.add_tensor(m.token_embd_norm); // tok_norm (:458)
        self.add_tensor(m.token_embd_norm_b); // tok_norm_b (:459)
        self.add_tensor(Some(m.output_norm)); // :460
        self.add_tensor(m.output_norm_b); // :461
        self.add_tensor(Some(m.output)); // :462
        self.add_tensor(m.output_b); // :463
        self.add_tensor(m.output_res_score); // :467
        self.add_tensor(m.cls); // :470
        self.add_tensor(m.cls_b); // :471
        self.add_tensor(m.cls_out); // :472
        self.add_tensor(m.cls_out_b); // :473
        self.add_tensor(m.hrm_z_l_init); // :475
        self.add_tensor(m.hc_head_fn); // :476
        self.add_tensor(m.hc_head_base); // :477
        self.add_tensor(m.hc_head_scale); // :478
        self.add_tensor(m.per_layer_tok_embd); // :479
        self.add_tensor(m.hc_head_norm); // :480
        self.add_tensor(m.hc_head_down); // :481
        self.add_tensor(m.hc_head_up); // :482

        // the layer loop (:486-496) — the C walks llama_layer's raw memory as
        // a ggml_tensor* array, i.e. the declaration order of llama-model.h
        // (norms, projections, FFN, SSM, RWKV time/channel mix, rope factors,
        // the granite scales, altup/laurel, then the arch tail), then the
        // posnet/convnext/shortconv/nextn/switch_lora sub-structs. Each tensor
        // is saved once (pointer identity == the port's TensorId identity for
        // duplicated requests).
        let mut seen: HashSet<TensorId> = HashSet::new();
        for layer in &m.layers {
            for t in layer_member_order(layer) {
                if let Some(id) = t {
                    if !seen.insert(id) {
                        continue;
                    }
                    self.add_tensor(Some(id));
                }
            }
        }
    }

    // ---- save (llama-model-saver.cpp:499-505) ----

    /// `gguf_write_to_file(gguf_ctx, path, /*only_meta=*/false)` — the meta
    /// section then every tensor's payload, padded (the port's GgufWriter).
    pub fn save(&self, w: &mut impl Write) -> std::io::Result<()> {
        let data: Vec<&[u8]> = self
            .w
            .tensors
            .iter()
            .map(|t| {
                let id = self
                    .model
                    .tensors
                    .get(&t.name)
                    .copied()
                    .unwrap_or_else(|| panic!("saver: tensor {} not in model", t.name));
                self.model.ctx.data_bytes(id).unwrap_or_else(|| {
                    panic!("saver: tensor {} has no storage", t.name)
                })
            })
            .collect();
        self.w.write(w, &data)
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut bw = std::io::BufWriter::with_capacity(1 << 20, f);
        self.save(&mut bw)?;
        bw.flush()
    }
}

/// The element GGUF type of the C template's `std::is_same<value_type, T>`
/// branches (llama-model-saver.cpp:98-111).
fn arr_type<T: 'static>() -> GgufType {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        GgufType::Uint32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        GgufType::Int32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        GgufType::Float32
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        GgufType::Uint64
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        GgufType::Bool
    } else {
        GgufType::Uint8 // unreachable for the instantiations used
    }
}

/// `llama_model_save_to_file` (llama.cpp:498-503).
pub fn save_model_to_file(model: &LlamaModel, vocab: &Vocab, path_model: &str) -> std::io::Result<()> {
    let mut ms = LlamaModelSaver::new(model);
    ms.add_kv_from_model(vocab);
    ms.add_tensors_from_model();
    ms.save_to_file(path_model)
}

// ---------------------------------------------------------------------------
// the C's raw llama_layer walk (llama-model-saver.cpp:488-496), as field order
// ---------------------------------------------------------------------------

/// One layer's tensor members in the C `llama_layer` declaration order
/// (llama-model.h:254-343 flat members, then the posnet/convnext/shortconv/
/// nextn/switch_lora sub-structs at :335-343 — the C's
/// `reinterpret_cast<ggml_tensor**>(&layer)[i]` walk covers exactly these).
///
/// Members marked `// ✗` are the C members no port-loadable arch ever
/// populates (their archs are not ported or the tensors are NVFP4 sidecars
/// the port's optional-scale pass does not create) — the C skips nullptr the
/// same way. The port keeps a few archs' members under aliased fields (e.g.
/// deepseek4's `attn_kv_norm` rides `attn_kv_a_norm`); those save at the
/// port field's position instead of the aliased C member's — same tensor
/// set, different file order, for those archs only.
fn layer_member_order(l: &crate::model::LayerTensors) -> Vec<Option<TensorId>> {
    let mut v: Vec<Option<TensorId>> = Vec::with_capacity(160);
    // -- normalization (llama-model.h:256-276) --
    v.extend([
        l.attn_norm,          // 1
        l.attn_norm_b,        // 2
        l.attn_norm_2,        // 3
        l.attn_norm_2_b,      // 4
        l.attn_q_norm,        // 5
        l.attn_q_norm_b,      // 6
        l.attn_k_norm,        // 7
        l.attn_k_norm_b,      // 8
        l.attn_out_norm,      // 9
        l.attn_out_norm_b,    // 10
        l.attn_q_a_norm,      // 11
        l.attn_kv_a_norm,     // 12 (deepseek4's attn_kv_norm aliases here)
        l.attn_sub_norm,      // 13
        l.attn_post_norm,     // 14
        l.ffn_sub_norm,       // 15
        // 16 attn_norm_cross ✗ (T5 decoder — arch rejected)
        l.enc_attn_norm,      // 17 attn_norm_enc
        l.ssm_norm,           // 18
        l.ssm_dt_norm,        // 19
        l.ssm_b_norm,         // 20
        l.ssm_c_norm,         // 21
    ]);
    // -- attention projections (:279-303) --
    v.extend([
        l.wq,              // 22
        l.wk,              // 23
        l.wv,              // 24
        l.wo,              // 25
        l.wqkv,            // 26
        // 27 wg ✗ (no arch creates it)
        l.wq_a,            // 28
        l.wq_b,            // 29
        l.wkv_a_mqa,       // 30
        l.wkv_b,           // 31
        // 32 wkv ✗ (deepseek4's attn_kv rides wkv_a_mqa at 30)
        l.wk_b,            // 33
        l.wv_b,            // 34
        l.wqkv_b,          // 35
        l.wo_a,            // 36
        l.wo_b,            // 37
        // 38-41 wq/wk/wv/wo_cross ✗ (T5 decoder)
        l.enc_wq,          // 42 wq_enc
        l.enc_wk,          // 43
        l.enc_wv,          // 44
        l.enc_wo,          // 45
        l.wqkv_gate,       // 46
    ]);
    // -- relative bias (:306-308) --
    v.extend([
        l.enc_attn_rel_b, // 47 attn_rel_b_enc (attn_rel_b itself is :53; only
                          // the encoder variant exists among port archs —
                          // arcee/t5encoder keep it here like every other
                          // attn_rel_b would slot)
        // 48 attn_rel_b / 49 attn_rel_b_cross ✗
    ]);
    // -- FFN norms (:311-320) --
    v.extend([
        l.ffn_norm,         // 50
        l.ffn_norm_b,       // 51
        l.ffn_post_norm,    // 52
        l.ffn_post_norm_1,  // 53
        l.ffn_post_norm_2,  // 54
        l.ffn_pre_norm_2,   // 55
        l.layer_out_norm,   // 56
        l.layer_out_norm_b, // 57
        l.ffn_norm_exps,    // 58
        l.enc_ffn_norm,     // 59 ffn_norm_enc
    ]);
    // -- FFN projections (:323-328) --
    v.extend([
        l.ffn_gate,     // 60
        l.ffn_down,     // 61
        l.ffn_up,       // 62
        l.enc_ffn_gate, // 63
        l.enc_ffn_down, // 64
        l.enc_ffn_up,   // 65
    ]);
    // -- MoE (:331-341) --
    v.extend([
        l.ffn_gate_inp,        // 66
        l.ffn_gate_inp_s,      // 67
        l.ffn_gate_exps,       // 68
        l.ffn_down_exps,       // 69
        l.ffn_up_exps,         // 70
        l.ffn_gate_up_exps,    // 71
        l.ffn_gate_inp_b,      // 72
        l.ffn_gate_exps_b,     // 73
        l.ffn_down_exps_b,     // 74
        l.ffn_up_exps_b,       // 75
        // 76 ffn_gate_up_exps_b ✗ (no port arch creates it)
        l.ffn_gate_exps_s,     // 77
        l.ffn_down_exps_s,     // 78
        l.ffn_up_exps_s,       // 79
    ]);
    v.extend([
        l.ffn_latent_down,     // 80
        l.ffn_latent_up,       // 81
        l.ffn_gate_inp_shexp,  // 82
        l.ffn_gate_shexp,      // 83
        l.ffn_down_shexp,      // 84
        l.ffn_up_shexp,        // 85
        l.ffn_gate_chexps,     // 86
        l.ffn_down_chexps,     // 87
        l.ffn_up_chexps,       // 88
        l.ffn_gate_b,          // 89
        l.ffn_down_b,          // 90
        l.ffn_up_b,            // 91
        l.ffn_act,             // 92
        l.ffn_exp_probs_b,     // 93
        l.ffn_exp_probs_b_vl,  // 94
        l.ffn_gate_tid2eid,    // 95
        // 96-99 dflash_{attn,ffn}_{conv_base,conv_proj} ✗ (dflash trunk)
    ]);
    // -- SSM (:372-384) --
    v.extend([
        l.ssm_in,      // 100
        l.ssm_x,       // 101
        l.ssm_dt,      // 102
        l.ssm_out,     // 103
        l.ssm_conv1d,  // 104
        l.ssm_a,       // 105
        l.ssm_d,       // 106
        l.ssm_conv1d_b, // 107
        l.ssm_dt_b,    // 108
        l.ssm_beta_alpha, // 109
        l.ssm_alpha,   // 110
    ]);
    // -- RWKV time mix (:387-422) --
    v.extend([
        l.time_mix_w1,
        l.time_mix_w2,
        l.time_mix_lerp_x,
        l.time_mix_lerp_w,
        l.time_mix_lerp_k,
        l.time_mix_lerp_v,
        l.time_mix_lerp_r,
        l.time_mix_lerp_g,
        l.time_mix_lerp_fused,
        l.time_mix_first,
        l.time_mix_decay,
        l.time_mix_decay_w1,
        l.time_mix_decay_w2,
        l.time_mix_key,
        l.time_mix_key_b,
        l.time_mix_value,
        l.time_mix_value_b,
        l.time_mix_receptance,
        l.time_mix_receptance_b,
        l.time_mix_gate,
        l.time_mix_w0,
        l.time_mix_a0,
        l.time_mix_a1,
        l.time_mix_a2,
        l.time_mix_v0,
        l.time_mix_v1,
        l.time_mix_v2,
        l.time_mix_g1,
        l.time_mix_g2,
        l.time_mix_k_k,
        l.time_mix_k_a,
        l.time_mix_r_k,
        l.time_mix_ln,
        l.time_mix_ln_b,
        l.time_mix_output,
    ]);
    // -- RWKV channel mix (:424-429) --
    v.extend([
        l.channel_mix_lerp_k,
        l.channel_mix_lerp_r,
        l.channel_mix_key,
        l.channel_mix_receptance,
        l.channel_mix_value,
    ]);
    // -- rope factors (:432-434) --
    v.extend([
        l.rope_long,  // 151
        l.rope_short, // 152
        l.rope_freqs, // 153
    ]);
    // -- granite scales (:437-452; only the unsuffixed members are ported —
    //    the port models granite-switch with switch_lora, the *_s sidecars are
    //    NVFP4-only) --
    v.extend([
        l.wq_s,        // 154
        l.wk_s,        // 155
        l.wv_s,        // 156
        l.wo_s,        // 157
        // 158 wqkv_s ✗ (no port arch creates it)
        // 159 wqkv_gate_s ✗
        l.ffn_gate_s,  // 160
        l.ffn_up_s,    // 161
        l.ffn_down_s,  // 162
        // 163-169 *_shexp_s / ssm_*_s ✗ ; 170-188 *_in_s ✗
    ]);
    // -- per-layer inputs / altup / laurel (:456-466) --
    v.extend([
        l.per_layer_inp_gate,  // 189
        l.per_layer_proj,      // 190
        l.per_layer_post_norm, // 191
        l.altup_correct_coef,  // 192
        l.altup_correct_scale, // 193
        l.altup_predict_coef,  // 194
        l.altup_router,        // 195
        l.altup_router_norm,   // 196
        l.laurel_l,            // 197
        l.laurel_r,            // 198
        l.laurel_post_norm,    // 199
    ]);
    // -- the arch tail (:469-542) --
    v.extend([
        l.attn_sinks,          // 200
        // 201 attn_kv_norm ✗ (rides attn_kv_a_norm for deepseek4)
        l.hc_attn_fn,          // 202
        l.hc_attn_base,        // 203
        l.hc_attn_scale,       // 204
        l.hc_ffn_fn,           // 205
        l.hc_ffn_base,         // 206
        l.hc_ffn_scale,        // 207
        l.attn_comp_wkv,       // 208
        l.attn_comp_wgate,     // 209
        l.attn_comp_ape,       // 210
        l.attn_comp_norm,      // 211
        l.indexer_comp_wkv,    // 212
        l.indexer_comp_wgate,  // 213
        l.indexer_comp_ape,    // 214
        l.indexer_comp_norm,   // 215
        l.visexp_attn_wqkv,    // 216
        l.visexp_attn_wo,      // 217
        l.visexp_ffn_gate,     // 218
        l.visexp_ffn_down,     // 219
        l.visexp_ffn_up,       // 220
        // 221-224 ffn_act_alpha_n/_p/beta/eps ✗ (no arch creates them)
        l.ssm_q_conv,          // 225
        l.ssm_k_conv,          // 226
        l.ssm_v_conv,          // 227
        l.ssm_f_a,             // 228
        l.ssm_f_b,             // 229
        l.ssm_beta,            // 230
        l.ssm_g_a,             // 231
        l.ssm_g_b,             // 232
        // 233 ssm_o_norm ✗ (kimi-linear/bailingmoe3 alias — not a LayerTensors
        //     field; their graphs read it via their own weights structs)
        l.ssm_g,               // 234
        l.attn_res_score,      // 235
        l.ffn_res_score,       // 236
        l.ffn_routed_down,     // 237
        l.ffn_routed_up,       // 238
        l.ffn_routed_norm,     // 239
        l.indexer_k_norm,      // 240
        l.indexer_k_norm_b,    // 241
        l.indexer_proj,        // 242
        l.indexer_attn_k,      // 243
        l.indexer_attn_q_b,    // 244
        l.index_q_proj,        // 245
        l.index_k_proj,        // 246
        l.index_q_norm,        // 247
        l.index_k_norm,        // 248
        l.hc_attn_norm,        // 249
        l.hc_attn_down,        // 250
        l.hc_attn_up,          // 251
        l.hc_attn_inject,      // 252
        l.hc_ffn_norm,         // 253
        l.hc_ffn_down,         // 254
        l.hc_ffn_up,           // 255
        l.hc_ffn_inject,       // 256
        // 257-262 ple_* ✗ (qwen4exp PLE, documented skip)
        l.out_scale,           // 263
    ]);
    // -- sub-structs (:335-343) --
    // posnet ✗ (gemma3n — arch rejected); convnext ✗ (wavtokenizer)
    v.extend([
        l.shortconv_in_proj,  // shortconv.in_proj (:226)
        l.shortconv_conv,     // shortconv.conv
        l.shortconv_out_proj, // shortconv.out_proj
    ]);
    v.extend([
        l.nextn.eh_proj,          // nextn.eh_proj (:341 llama_layer_nextn)
        // eh_proj_s / eh_proj_in_s ✗ (NVFP4 sidecars)
        l.nextn.embed_tokens,     // nextn.embed_tokens
        l.nextn.enorm,            // nextn.enorm
        l.nextn.hnorm,            // nextn.hnorm
        l.nextn.shared_head_head, // nextn.shared_head_head
        // shared_head_head_s / _in_s ✗
        l.nextn.shared_head_norm, // nextn.shared_head_norm
    ]);
    if let Some(s) = l.switch_lora.as_ref() {
        v.extend([
            Some(s.a_q),
            Some(s.b_q),
            Some(s.a_k),
            Some(s.b_k),
            Some(s.a_v),
            Some(s.b_v),
            Some(s.a_o),
            Some(s.b_o),
            Some(s.a_gate),
            Some(s.b_gate),
            Some(s.a_up),
            Some(s.b_up),
            Some(s.a_down),
            Some(s.b_down),
        ]);
    }
    v
}
