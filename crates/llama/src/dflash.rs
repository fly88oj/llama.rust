//! dflash — the DFlash/DSpark draft-model loader + graphs, the port of the
//! reference's dflash arch support outside `common/speculative.cpp` (the draft
//! driver itself lives in [`crate::speculative`]'s draft-dflash impl):
//!
//!   * `llama_model_dflash::load_arch_hparams` (src/models/dflash.cpp:7-105) —
//!     the dflash.* GGUF keys (`target_layers` required, the block/conv/
//!     selector knobs optional) plus the meta strings the draft driver and the
//!     markov head read back through `llama_model_meta_val_str`
//!     (`dflash.block_size` / `dflash.sample_from_anchor` /
//!     `dflash.attention.causal` / `dflash.has_confidence_head`,
//!     speculative.cpp:969-991 / dflash.cpp:310-318);
//!   * `llama_model_dflash::load_arch_tensors` (dflash.cpp:107-259) — the
//!     plain-transformer tensor set: fc / enc.output_norm / output_norm /
//!     output (optional) / token_embd (optional) / d2t (optional) + the DSpark
//!     markov head (markov_w1 / markov_w2[.scale] / conf_proj[.bias]);
//!   * `llama_model_dflash::graph<false>` (dflash.cpp:572-853) — the dual-mode
//!     decoder: an **embd batch** injects the fused target features' K/V into
//!     the draft cache (`build_dflash_inject_forward`), a **token batch** runs
//!     the noise-block diffusion (`build_dflash_noise_forward`, incl. the
//!     DSpark markov/confidence heads of dflash.cpp:295-406);
//!   * the DFlash2 conv/selector graphs — `build_dflash2_conv` /
//!     `build_dflash2_selector` (dflash.cpp:408-567) with their tensors
//!     (:138-159, :249-257): the per-layer dynamic-bias depthwise convolutions
//!     around attention and FFN, and the top-k candidate lattice packed into
//!     the nextn output slot for the host-side walk (already literal in
//!     [`crate::speculative`], speculative.cpp:1236-1263);
//!   * the DSV4 DSpark backbone — `llama_model_dflash::graph_dsv4`
//!     (dflash.cpp:52-93 / :855-1028, a full deepseek4 stage stack —
//!     hyper-connection + MLA + MoE — over `dsv4_hc_mult > 0` drafts; the
//!     hc/MLA/MoE stage bodies are local re-instantiations of the
//!     graph_arch.rs dsv4 builders, which this file cannot reach);
//!   * `cparams.ctx_other` (llama-context.cpp:144-161 feeds dflash) — the
//!     decoder's optional token_embd / output come from the *target* model;
//!     like eagle.rs the port materializes them as tensors of the draft's own
//!     ggml Context backed by the target file's mmap (`set_external_storage`).
//!
//! Not ported (refused with a load error, C file:line):
//!   * M-RoPE drafts (`llama_model_rope_type == LLAMA_ROPE_TYPE_MROPE`,
//!     speculative.cpp:1014-1018 / dflash.cpp:598-606): the port's draft
//!     contexts carry one position per token; the degenerate-sections case
//!     is refused.

use std::sync::Arc;

use ggml::{Context, Gguf, Graph, TensorId};

use crate::graph::{self, AttnParams, DecodeInputs, ForwardResult};
use crate::kv_cache::{KvCache, SlotInfo};

/// `build_norm(cur, weight, NULL, LLM_NORM_RMS, -1)` (llama-graph.cpp:918-936)
/// — RMSNorm with an optional weight (NULL weight = the bare `ggml_rms_norm`
/// of the shared-KV V path, dflash.cpp:637/:730).
fn build_norm_rms_opt(ctx: &mut Context, x: TensorId, w: Option<TensorId>, eps: f32) -> TensorId {
    let n = ctx.rms_norm(x, eps);
    match w {
        Some(w) => ctx.mul(n, w),
        None => n,
    }
}

/// `build_ffn(..., hparams.llm_ffn_op, LLM_FFN_PAR, il)` (dflash.cpp:767-772)
/// — SILU (default) or GELU (gelu / gelu_pytorch_tanh, dflash.cpp:13-19) in
/// the parallel gate/up order.
fn build_ffn_dflash(
    ctx: &mut Context,
    cur: TensorId,
    gate: TensorId,
    up: TensorId,
    down: TensorId,
    gelu: bool,
) -> TensorId {
    if gelu {
        // build_ffn LLM_FFN_GELU + LLM_FFN_PAR: ggml_geglu_split(gate, up)
        let g = crate::adapter::lora_mm(ctx, gate, cur);
        let u = crate::adapter::lora_mm(ctx, up, cur);
        let act = ctx.geglu_split(g, u);
        crate::adapter::lora_mm(ctx, down, act)
    } else {
        // build_ffn LLM_FFN_SILU + LLM_FFN_PAR (graph_arch::build_ffn_silu_par):
        // silu(gate) * up, then down
        let g = crate::adapter::lora_mm(ctx, gate, cur);
        let u = crate::adapter::lora_mm(ctx, up, cur);
        let g = ctx.silu(g);
        let prod = ctx.mul(g, u);
        crate::adapter::lora_mm(ctx, down, prod)
    }
}

// ---------------------------------------------------------------------------
// the weights (llama-model.h:365-375 / :675-692)
// ---------------------------------------------------------------------------

/// one decoder layer of the plain (non-DSV4) dflash backbone
/// (dflash.cpp:223-258)
pub struct DflashLayerWeights {
    pub attn_norm: TensorId,
    pub wq: TensorId,
    pub wk: TensorId,
    /// `TENSOR_NOT_REQUIRED` (:230) — None = the shared-KV mode where V is
    /// `rms_norm(Kcur)` (:629-637/:720-731)
    pub wv: Option<TensorId>,
    pub wo: TensorId,
    pub attn_q_norm: TensorId,
    pub attn_k_norm: TensorId,
    /// optional post-norms (:236-237)
    pub attn_post_norm: Option<TensorId>,
    pub ffn_post_norm: Option<TensorId>,
    /// optional per-layer output scale (:238)
    pub out_scale: Option<TensorId>,
    /// rope freq factors (:239, layer 0 shared)
    pub rope_freqs: Option<TensorId>,
    /// optional per-head attention sinks (:242)
    pub attn_sinks: Option<TensorId>,
    pub ffn_norm: TensorId,
    pub ffn_gate: TensorId,
    pub ffn_down: TensorId,
    pub ffn_up: TensorId,
    /// DFlash2: `{n_embd, kernel, 2}` — the static conv base (:253/:255);
    /// None on non-DFlash2 layers
    pub dflash_attn_conv_base: Option<TensorId>,
    /// DFlash2: `{n_embd, 2*kernel*groups}` (:254)
    pub dflash_attn_conv_proj: Option<TensorId>,
    /// DFlash2 FFN-side pair (:255-256)
    pub dflash_ffn_conv_base: Option<TensorId>,
    pub dflash_ffn_conv_proj: Option<TensorId>,
}

/// the loaded dflash draft model (dflash.cpp:107-259). `tok_embd` / `output`
/// are the tensors to use — own or the target's (the loader materializes the
/// ctx_other ones from the target file's mmap).
pub struct DflashWeights {
    /// None = the target model's (`cparams.ctx_other`, dflash.cpp:679-687);
    /// the loader resolves it before returning, so it is never None in a
    /// loaded draft
    pub tok_embd: Option<TensorId>,
    /// None = the target model's (:799-808); resolved like `tok_embd`
    pub output: Option<TensorId>,
    /// `{n_embd}` — the decoder final norm (:164)
    pub output_norm: TensorId,
    /// `{n_embd}` — the encoder hidden_norm after fc (:163)
    pub output_norm_enc: TensorId,
    /// `{n_embd_inp_enc, n_embd}` — the feature fusion layer (:161)
    pub fc: TensorId,
    pub fc_s: Option<TensorId>,
    /// `{n_vocab_draft}` I64 — the draft→target vocab map (:115-121); when
    /// present the decoder scatters its draft-vocab logits into a
    /// target-vocab row (:826-839)
    pub d2t: Option<TensorId>,
    /// DSpark: `{markov_rank, n_vocab}` (:128)
    pub dspark_markov_w1: Option<TensorId>,
    /// DSpark: `{markov_rank, n_vocab_draft}` (:129)
    pub dspark_markov_w2: Option<TensorId>,
    pub dspark_markov_w2_s: Option<TensorId>,
    /// DSpark confidence head `{n_embd + rank, 1}` (:132-133, optional)
    pub dspark_conf_proj: Option<TensorId>,
    pub dspark_conf_proj_b: Option<TensorId>,
    /// DFlash2: `{rank, n_vocab}` — `selector_predecessor` (:152)
    pub dflash_selector_prev: Option<TensorId>,
    /// DFlash2: `{rank, n_vocab}` — `selector_successor` (:153)
    pub dflash_selector_next: Option<TensorId>,
    /// DFlash2: `{n_embd, rank}` — `selector_hidden` (:154)
    pub dflash_selector_hidden: Option<TensorId>,
    /// the DSV4 DSpark backbone (`hparams.dsv4_hc_mult > 0`, dflash.cpp:
    /// 173-221): the deepseek4 stage stack — hc tensors + MLA attention +
    /// MoE per layer. None on the plain backbone (whose layers live above).
    pub dsv4: Option<DsparkDsv4Staged>,
    pub layers: Vec<DflashLayerWeights>,
}

/// one DSV4 DSpark stage (dflash.cpp:188-219 — the deepseek4 layer table of
/// deepseek4.cpp:110-172 minus the compressor/indexer tensors: the loader
/// requires `compress_ratios` all zero)
pub struct DsparkDsv4LayerWeights {
    pub attn_norm: TensorId,
    pub attn_sinks: TensorId,
    pub wq_a: TensorId,
    pub attn_q_a_norm: TensorId,
    pub wq_b: TensorId,
    pub wkv: TensorId,
    pub attn_kv_norm: TensorId,
    /// `{n_head*n_embd_head/o_groups, o_lora_rank, o_groups}` — reshaped from
    /// the file's 2-D `{..., o_lora_rank*o_groups}` (TENSOR_ALLOW_RESHAPE,
    /// dflash.cpp:198)
    pub wo_a: TensorId,
    pub wo_b: TensorId,

    pub hc_attn_fn: TensorId,
    pub hc_attn_base: TensorId,
    pub hc_attn_scale: TensorId,
    pub hc_ffn_fn: TensorId,
    pub hc_ffn_base: TensorId,
    pub hc_ffn_scale: TensorId,

    pub ffn_gate_inp: TensorId,
    pub ffn_exp_probs_b: TensorId,
    pub ffn_norm: TensorId,
    pub ffn_gate_exps: TensorId,
    pub ffn_down_exps: TensorId,
    pub ffn_up_exps: TensorId,
    pub ffn_gate_shexp: TensorId,
    pub ffn_down_shexp: TensorId,
    pub ffn_up_shexp: TensorId,
}

/// the DSV4 DSpark stage params (load_arch_hparams' dsv4 arm, dflash.cpp:
/// 52-93 — the deepseek4 knobs the stage graphs need)
#[derive(Clone)]
pub struct DsparkDsv4Params {
    /// hparams.dsv4_hc_mult — asserted == 4 (deepseek4.cpp:367)
    pub hc_mult: i64,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: i32,
    pub o_group_count: i64,
    pub o_lora_rank: i64,
    /// hparams.n_swa — the draft KV ring's sliding window
    pub n_swa: u32,
    pub n_expert: i64,
    pub n_expert_used: i64,
    pub n_ff_exp: i64,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    /// per layer, `swiglu_clamp_exp` / `shexp` (shexp defaults to exp)
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
}

/// the DSV4 DSpark backbone's own tensors (dflash.cpp:173-221): the model
/// head's hc mixer + one stage per layer. The shared pieces (tok_embd /
/// output / output_norm / fc / the markov head) ride the enclosing
/// [`DflashWeights`].
pub struct DsparkDsv4Staged {
    /// `{hc_dim, hc_mult}` (:184)
    pub hc_head_fn: TensorId,
    /// `{hc_mult}` (:185)
    pub hc_head_base: TensorId,
    /// `{1}` (:186)
    pub hc_head_scale: TensorId,
    pub layers: Vec<DsparkDsv4LayerWeights>,
    pub params: DsparkDsv4Params,
}

/// the draft model's hyperparameters (load_arch_hparams, dflash.cpp:7-105) +
/// the meta strings the driver re-reads (speculative.cpp:969-991)
#[derive(Clone)]
pub struct DflashParams {
    pub attn: AttnParams,
    /// hparams.n_embd — the draft hidden size (== n_embd_dec)
    pub n_embd: i64,
    /// hparams.n_embd_inp_enc() == target_layer_ids_n * n_embd_tgt (:40)
    pub n_embd_inp_enc: i64,
    /// model.vocab.n_tokens() — the target row count the d2t scatter needs
    pub n_vocab: i64,
    /// the draft's own vocab size (== n_vocab without d2t)
    pub n_vocab_draft: i64,
    /// `target_layers` (required, :36-38) — what the driver reads through
    /// `llama_model_target_layer_ids` (speculative.cpp:957-958)
    pub target_layer_ids: Vec<i32>,
    /// `llama_model_n_embd(model_tgt)` (speculative.cpp:961) — read off the
    /// target file
    pub n_embd_tgt: i64,
    /// `{arch}.embedding_scale` (:9, default 0 = off)
    pub f_embedding_scale: f32,
    /// `{arch}.attention.scale` (:10, default 0 → 1/sqrt(n_embd_head))
    pub f_attention_scale: f32,
    /// `{arch}.attention.value_scale` (dflash.cpp:11, def4d406a — default 0
    /// = no post-attention scale)
    pub f_attn_value_scale: f32,
    /// `{arch}.logit_scale` (:23, default 0)
    pub f_logit_scale: f32,
    /// `{arch}.final_logit_softcapping` (:24-25, default 0)
    pub f_final_logit_softcapping: f32,
    /// hidden_act == gelu / gelu_pytorch_tanh (:13-19)
    pub ffn_gelu: bool,
    /// `dflash.block_size` meta string (speculative.cpp:966-971, default 16)
    pub block_size: i32,
    /// `dflash.sample_from_anchor` meta string (:972-974, default true)
    pub sample_from_anchor: bool,
    /// `dflash.attention.causal` meta string (:975-977, default false)
    pub attention_causal: bool,
    /// `dflash.has_confidence_head` meta string (:986-988, default true)
    pub has_confidence_head: bool,
    /// `llama_model_dflash_selector_top_k` (:980) — > 0 marks DFlash2
    pub selector_top_k: u32,
    /// `hparams.dflash_block_size` (the U32 hparams read of the same GGUF
    /// key the meta-string `block_size` above renders, :30) — the trained
    /// block length the selector clamps to (dflash.cpp:496)
    pub dflash_block_size: u32,
    /// `hparams.dflash_conv_kernel_size` (:31)
    pub conv_kernel_size: u32,
    /// `hparams.dflash_conv_group_size` (:32)
    pub conv_group_size: u32,
    /// `hparams.dflash_selector_rank` (:33)
    pub selector_rank: u32,
}

/// `gguf_kv_to_str` (llama-impl.cpp) restricted to the scalar types the dflash
/// meta keys use — the renderer behind `llama_model_meta_val_str`
/// (llama-model.cpp:3085-3099): every non-array KV lands in `gguf_kv` as its
/// string rendering (u32 → "4", bool → "true"/"false").
fn meta_val_str(gguf: &Gguf, key: &str) -> Option<String> {
    let v = gguf.find_key(key)?;
    Some(match v {
        ggml::Value::String(s) => s.clone(),
        ggml::Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
        ggml::Value::U8(v) => v.to_string(),
        ggml::Value::I8(v) => v.to_string(),
        ggml::Value::U16(v) => v.to_string(),
        ggml::Value::I16(v) => v.to_string(),
        ggml::Value::U32(v) => v.to_string(),
        ggml::Value::I32(v) => v.to_string(),
        ggml::Value::U64(v) => v.to_string(),
        ggml::Value::I64(v) => v.to_string(),
        ggml::Value::F32(v) => v.to_string(),
        ggml::Value::F64(v) => v.to_string(),
        _ => return None,
    })
}

/// one created tensor + its gguf name (the create_tensor mirror of eagle.rs:
/// exact dims check, external mmap storage)
struct DflashLoader<'a> {
    gguf: &'a Gguf,
    mmap: Arc<memmap2::Mmap>,
    ctx: Context,
    by_name: std::collections::HashMap<String, TensorId>,
    n_created: usize,
}

impl<'a> DflashLoader<'a> {
    fn new(gguf: &'a Gguf, mmap: Arc<memmap2::Mmap>) -> Self {
        Self {
            gguf,
            mmap,
            ctx: Context::new(),
            by_name: std::collections::HashMap::new(),
            n_created: 0,
        }
    }

    /// `llama_model_loader::create_tensor` restricted to what the dflash
    /// draft needs: exact dims, required-or-optional.
    fn create_tensor(
        &mut self,
        name: &str,
        ne: &[i64],
        required: bool,
    ) -> Result<Option<TensorId>, String> {
        let Some(ti) = self.gguf.find_tensor(name) else {
            if !required {
                return Ok(None);
            }
            return Err(format!("tensor '{name}' not found"));
        };
        for i in 0..4 {
            let want = if i < ne.len() { ne[i] } else { 1 };
            if want != ti.ne[i] {
                return Err(format!(
                    "tensor '{name}' has wrong shape; expected [{:?}], got [{:?}]",
                    &ne[..ne.len().min(4)],
                    ti.ne
                ));
            }
        }
        let id = self.ctx.new_tensor(ti.ty, ti.ne);
        self.ctx.set_name(id, name);
        let data_off = self.gguf.data_offset as usize + ti.offset as usize;
        self.ctx
            .set_external_storage(id, self.mmap.clone(), data_off);
        self.by_name.insert(name.to_string(), id);
        self.n_created += 1;
        Ok(Some(id))
    }

    /// `create_tensor` with `TENSOR_ALLOW_RESHAPE`: the file's (flattened)
    /// element count must match `ne`; the tensor is created with the given
    /// (possibly higher-D) shape over the same bytes.
    fn create_tensor_reshaped(&mut self, name: &str, ne: &[i64]) -> Result<TensorId, String> {
        let Some(ti) = self.gguf.find_tensor(name) else {
            return Err(format!("tensor '{name}' not found"));
        };
        let want_flat: i64 = ne.iter().product();
        let got_flat: i64 = ti.ne.iter().product();
        if want_flat != got_flat {
            return Err(format!(
                "tensor '{name}' has wrong shape; expected [{:?}] (or a flat form, {want_flat} \
                 elements), got [{:?}] ({got_flat} elements)",
                &ne[..ne.len().min(4)],
                ti.ne
            ));
        }
        let ne4 = [
            *ne.first().unwrap_or(&1),
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        let id = self.ctx.new_tensor(ti.ty, ne4);
        self.ctx.set_name(id, name);
        let data_off = self.gguf.data_offset as usize + ti.offset as usize;
        self.ctx
            .set_external_storage(id, self.mmap.clone(), data_off);
        self.by_name.insert(name.to_string(), id);
        self.n_created += 1;
        Ok(id)
    }

    /// the ctx_other tensor (the target model's token_embd / output), same
    /// bytes through the target file's mmap — eagle.rs's mechanism
    fn create_tensor_in(
        &mut self,
        other_gguf: &Gguf,
        other_mmap: &Arc<memmap2::Mmap>,
        name: &str,
        ne: &[i64],
    ) -> Result<TensorId, String> {
        let Some(ti) = other_gguf.find_tensor(name) else {
            return Err(format!(
                "DFlash decoder requires '{name}' (own or from target model) — not in either file"
            ));
        };
        for i in 0..4 {
            let want = if i < ne.len() { ne[i] } else { 1 };
            if want != ti.ne[i] {
                return Err(format!(
                    "tensor '{name}' has wrong shape; expected [{:?}], got [{:?}]",
                    &ne[..ne.len().min(4)],
                    ti.ne
                ));
            }
        }
        let id = self.ctx.new_tensor(ti.ty, ti.ne);
        self.ctx.set_name(id, &format!("other.{name}"));
        let data_off = other_gguf.data_offset as usize + ti.offset as usize;
        self.ctx
            .set_external_storage(id, other_mmap.clone(), data_off);
        Ok(id)
    }
}

/// the loaded draft: its own ggml Context (every TensorId belongs to it) +
/// weights + params.
pub struct DflashDraft {
    pub ctx: Context,
    pub weights: DflashWeights,
    pub params: DflashParams,
}

/// `load_dflash_draft` — the dflash hparams + tensors (dflash.cpp:7-259) for
/// the plain DFlash1 / DSpark backbones. `target_gguf` / `target_mmap` are the
/// *target* model's file (the ctx_other tensors + `n_embd_tgt`); `n_vocab` is
/// the shared vocabulary size.
pub fn load_dflash_draft(
    gguf: &Gguf,
    mmap: Arc<memmap2::Mmap>,
    target_gguf: &Gguf,
    target_mmap: Arc<memmap2::Mmap>,
    n_vocab: i64,
    fa: bool,
) -> Result<DflashDraft, String> {
    // ---- load_arch_hparams (dflash.cpp:7-105) ----
    let f_embedding_scale = gguf.get_f32("dflash.embedding_scale").unwrap_or(0.0);
    let f_attention_scale = gguf.get_f32("dflash.attention.scale").unwrap_or(0.0);
    // dflash.cpp:11 (def4d406a) — LLM_KV_ATTENTION_VALUE_SCALE, optional
    let f_attn_value_scale = gguf.get_f32("dflash.attention.value_scale").unwrap_or(0.0);
    let ffn_gelu = match gguf.get_str("dflash.hidden_activation") {
        Some("gelu") | Some("gelu_pytorch_tanh") => true,
        Some("silu") => false,
        Some(other) => {
            return Err(format!("unsupported DFlash hidden activation: {other}"));
        }
        None => false,
    };
    let eps = gguf
        .get_f32("dflash.attention.layer_norm_rms_epsilon")
        .ok_or("key dflash.attention.layer_norm_rms_epsilon not found in model file")?;
    let f_logit_scale = gguf.get_f32("dflash.logit_scale").unwrap_or(0.0);
    let f_final_logit_softcapping = gguf
        .get_f32("dflash.final_logit_softcapping")
        .unwrap_or(0.0);

    // the meta strings the driver + the markov head re-read
    // (speculative.cpp:965-991 / dflash.cpp:310-318); `std::atoi` / the
    // "true" compares happen on the gguf_kv rendering
    let block_size: i32 = meta_val_str(gguf, "dflash.block_size")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(16);
    let sample_from_anchor = meta_val_str(gguf, "dflash.sample_from_anchor")
        .map(|s| s == "true")
        .unwrap_or(true);
    let attention_causal = meta_val_str(gguf, "dflash.attention.causal")
        .map(|s| s == "true")
        .unwrap_or(false);
    let has_confidence_head = meta_val_str(gguf, "dflash.has_confidence_head")
        .map(|s| s == "true")
        .unwrap_or(true);

    // LLM_KV_TARGET_LAYERS (required, :36-38)
    let target_layer_ids: Vec<i32> = {
        let arr = gguf
            .find_key("dflash.target_layers")
            .and_then(|v| v.as_array())
            .ok_or("DFlash model requires 'target_layers' in GGUF metadata")?;
        let ids: Result<Vec<i32>, ()> = arr
            .1
            .iter()
            .map(|v| match v {
                ggml::Value::U32(u) => Ok(*u as i32),
                ggml::Value::I32(i) => Ok(*i),
                ggml::Value::U64(u) => Ok(*u as i32),
                ggml::Value::I64(i) => Ok(*i as i32),
                ggml::Value::F32(f) => Ok(*f as i32),
                _ => Err(()),
            })
            .collect();
        ids.map_err(|()| "DFlash: malformed dflash.target_layers array".to_string())?
    };
    let selector_top_k = gguf.get_u32("dflash.selector_top_k").unwrap_or(0);
    let dflash_block_size = gguf.get_u32("dflash.block_size").unwrap_or(0);
    let conv_kernel_size = gguf.get_u32("dflash.conv_kernel_size").unwrap_or(0);
    let conv_group_size = gguf.get_u32("dflash.conv_group_size").unwrap_or(0);
    let selector_rank = gguf.get_u32("dflash.selector_rank").unwrap_or(0);

    // ---- the DSV4 DSpark backbone (dflash.cpp:52-93): stages are full DSV4
    // blocks over a uniform sliding window (the draft KV ring) ----
    let dsv4_hc_mult = gguf.get_u32("dflash.hyper_connection.count").unwrap_or(0);
    let dsv4: Option<DsparkDsv4Params> = if dsv4_hc_mult > 0 {
        let q_lora_rank = gguf
            .get_u32("dflash.attention.q_lora_rank")
            .ok_or("key dflash.attention.q_lora_rank not found in model file")?
            as i64; // :54
        let n_swa = gguf
            .get_u32("dflash.attention.sliding_window")
            .ok_or("key dflash.attention.sliding_window not found in model file")?; // :55
        let n_layer_all = gguf
            .get_u32("dflash.block_count")
            .ok_or("key dflash.block_count not found")? as usize;

        // the per-layer (or scalar) arrays (:56-64)
        let arr_or_scalar = |key: &str| -> Result<Vec<f32>, String> {
            match gguf.find_key(key) {
                Some(ggml::Value::Array(_, v)) => v
                    .iter()
                    .map(|x| match x {
                        ggml::Value::F32(f) => Ok(*f),
                        ggml::Value::F64(f) => Ok(*f as f32),
                        _ => Err(format!("DFlash: malformed {key} array")),
                    })
                    .collect(),
                Some(ggml::Value::F32(f)) => Ok(vec![*f; n_layer_all]),
                Some(ggml::Value::U32(u)) => Ok(vec![*u as f32; n_layer_all]),
                _ => Ok(Vec::new()),
            }
        };
        let n_ff_exp_arr = arr_or_scalar("dflash.expert_feed_forward_length")?;
        if n_ff_exp_arr.is_empty() {
            return Err("key dflash.expert_feed_forward_length not found in model file".into());
        }
        let n_ff_exp = n_ff_exp_arr[0] as i64;

        let n_expert =
            gguf.get_u32("dflash.expert_count")
                .ok_or("key dflash.expert_count not found in model file")? as i64;
        let n_expert_used = gguf
            .get_u32("dflash.expert_used_count")
            .unwrap_or(n_expert as u32) as i64;
        let expert_weights_scale = gguf.get_f32("dflash.expert_weights_scale").unwrap_or(0.0);
        let expert_weights_norm = match gguf.find_key("dflash.expert_weights_norm") {
            Some(ggml::Value::Bool(b)) => *b,
            Some(ggml::Value::String(s)) => s == "true",
            Some(ggml::Value::U32(u)) => u != &0,
            _ => false,
        }; // ml.get_key(bool) reads gguf_kv's "true"/"false" rendering
        let expert_gating_func = gguf
            .get_u32("dflash.expert_gating_func")
            .ok_or("key dflash.expert_gating_func not found in model file")?;
        if expert_gating_func != 4 {
            // LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS (:73-75)
            return Err("DSpark DSV4 draft expects sqrtsoftplus MoE scoring".into());
        }

        let mut swiglu_clamp_exp = arr_or_scalar("dflash.swiglu_clamp_exp")?;
        if swiglu_clamp_exp.is_empty() {
            return Err("key dflash.swiglu_clamp_exp not found in model file".into());
        }
        let mut swiglu_clamp_shexp = arr_or_scalar("dflash.swiglu_clamp_shexp")?;
        if swiglu_clamp_shexp.is_empty() {
            swiglu_clamp_shexp = swiglu_clamp_exp.clone(); // :62-64
        }
        swiglu_clamp_exp.resize(n_layer_all, swiglu_clamp_exp[0]);
        swiglu_clamp_shexp.resize(n_layer_all, swiglu_clamp_shexp[0]);

        let o_group_count = gguf
            .get_u32("dflash.attention.output_group_count")
            .ok_or("key dflash.attention.output_group_count not found in model file")?
            as i64; // :65
        let o_lora_rank = gguf
            .get_u32("dflash.attention.output_lora_rank")
            .ok_or("key dflash.attention.output_lora_rank not found in model file")?
            as i64; // :66
        if o_group_count == 0 {
            return Err("DSpark DSV4 draft: attention.output_group_count must be > 0".into());
            // :71
        }
        let hc_sinkhorn_iters = gguf
            .get_u32("dflash.hyper_connection.sinkhorn_iterations")
            .unwrap_or(4) as i32;
        let hc_eps = gguf
            .get_f32("dflash.hyper_connection.epsilon")
            .unwrap_or(1e-3);

        // compress_ratios must be absent-or-all-zero (:69/:76-80)
        if let Some(ggml::Value::Array(_, v)) = gguf.find_key("dflash.attention.compress_ratios") {
            if v.iter().any(|x| match x {
                ggml::Value::U32(u) => *u != 0,
                ggml::Value::I32(i) => *i != 0,
                ggml::Value::F32(f) => *f != 0.0,
                _ => false,
            }) {
                return Err(
                    "DSpark DSV4 draft expects uncompressed attention on all stages".into(),
                );
            }
        }

        if n_swa == 0 {
            return Err("DSpark DSV4 draft: attention.sliding_window must be > 0".into());
            // :82
        }

        let _ = q_lora_rank; // consumed by the tensor shapes below
        Some(DsparkDsv4Params {
            hc_mult: dsv4_hc_mult as i64,
            hc_eps,
            hc_sinkhorn_iters,
            o_group_count,
            o_lora_rank,
            n_swa,
            n_expert,
            n_expert_used,
            n_ff_exp,
            expert_weights_norm,
            expert_weights_scale,
            swiglu_clamp_exp,
            swiglu_clamp_shexp,
        })
    } else {
        None
    };
    // M-RoPE drafts (rope_sections / speculative.cpp:1014-1018) — not ported
    if gguf
        .find_key("dflash.rope.dimension_sections")
        .and_then(|v| v.as_array())
        .is_some()
    {
        return Err(
            "M-RoPE DFlash drafts (dflash.rope.dimension_sections, speculative.cpp:1014-1018) \
             are not ported — the port's draft contexts carry one position per token"
                .into(),
        );
    }

    // the generic dims the graphs need (llama_model_base::load_hparams)
    let n_embd = gguf
        .get_u32("dflash.embedding_length")
        .ok_or("key dflash.embedding_length not found in model file")? as i64;
    let n_layer = gguf
        .get_u32("dflash.block_count")
        .ok_or("key dflash.block_count not found in model file")? as usize;
    let n_head = gguf
        .get_u32("dflash.attention.head_count")
        .ok_or("key dflash.attention.head_count not found in model file")? as i64;
    let n_head_kv = gguf
        .get_u32("dflash.attention.head_count_kv")
        .unwrap_or(n_head as u32) as i64;
    let n_embd_head_k = gguf
        .get_u32("dflash.attention.key_length")
        .unwrap_or((n_embd / n_head.max(1)) as u32) as i64;
    let n_embd_head_v = gguf
        .get_u32("dflash.attention.value_length")
        .unwrap_or((n_embd / n_head.max(1)) as u32) as i64;
    let n_rot = gguf
        .get_u32("dflash.rope.dimension_count")
        .ok_or("key dflash.rope.dimension_count not found in model file")? as i64;
    let freq_base = gguf.get_f32("dflash.rope.freq_base").unwrap_or(10000.0);
    let freq_scale = gguf.get_f32("dflash.rope.freq_scale").unwrap_or(1.0);
    let n_ctx_train = gguf.get_u32("dflash.context_length").unwrap_or(0) as i32;
    let n_ff = gguf.get_u32("dflash.feed_forward_length").unwrap_or(0) as i64;

    if n_head <= 0 || n_embd_head_k <= 0 || n_embd_head_v <= 0 {
        return Err("dflash: attention.head_count missing or zero".into());
    }
    // GGML_ASSERT(n_embd_head_v == n_embd_head_k()) of the decoder graph
    // (dflash.cpp:575-577)
    if n_embd_head_v != n_embd_head_k {
        return Err("dflash: n_embd_head_v must equal n_embd_head_k".into());
    }

    // the target's hidden size (speculative.cpp:961 reads it off the target
    // model); n_embd_inp_enc = target_layer_ids_n * n_embd_tgt (dflash.cpp:40)
    let tgt_arch = target_gguf
        .get_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let n_embd_tgt = target_gguf
        .get_u32(&format!("{tgt_arch}.embedding_length"))
        .ok_or("the target model's embedding_length is required for a dflash draft")?
        as i64;
    let n_embd_inp_enc = target_layer_ids.len() as i64 * n_embd_tgt;

    // ---- load_arch_tensors (dflash.cpp:107-259) ----
    let mut ld = DflashLoader::new(gguf, mmap);

    let tok_embd = ld.create_tensor("token_embd.weight", &[n_embd, n_vocab], false)?;

    // reduced draft vocab (optional): d2t maps draft rows to target ids
    // (:115-121)
    let n_vocab_draft = gguf
        .find_tensor("d2t")
        .map(|ti| ti.ne[0])
        .unwrap_or(n_vocab);
    let d2t = ld.create_tensor("d2t", &[n_vocab_draft], false)?;
    if d2t.is_some() {
        eprintln!("dflash: DFlash using d2t mapping (draft_vocab_size = {n_vocab_draft})");
    }

    // DSpark = DFlash + the Markov head and (optional) Confidence head
    // (:123-136)
    let (
        dspark_markov_w1,
        dspark_markov_w2,
        dspark_markov_w2_s,
        dspark_conf_proj,
        dspark_conf_proj_b,
    ) = match gguf.find_tensor("markov_w1.weight") {
        Some(markov_meta) => {
            let dspark_markov_rank = markov_meta.ne[0];
            let w1 = ld
                .create_tensor("markov_w1.weight", &[dspark_markov_rank, n_vocab], true)?
                .expect("markov_w1.weight");
            let w2 = ld
                .create_tensor(
                    "markov_w2.weight",
                    &[dspark_markov_rank, n_vocab_draft],
                    true,
                )?
                .expect("markov_w2.weight");
            let w2_s = ld.create_tensor("markov_w2.scale", &[1], false)?;
            let conf_proj =
                ld.create_tensor("conf_proj.weight", &[n_embd + dspark_markov_rank, 1], false)?;
            let conf_proj_b = ld.create_tensor("conf_proj.bias", &[1], false)?;
            eprintln!("dflash: DFlash with DSpark markov head (rank = {dspark_markov_rank})");
            (Some(w1), Some(w2), w2_s, conf_proj, conf_proj_b)
        }
        None => (None, None, None, None, None),
    };

    let fc = ld
        .create_tensor("fc.weight", &[n_embd_inp_enc, n_embd], true)?
        .expect("fc.weight");
    let fc_s = ld.create_tensor("fc.scale", &[1], false)?;
    let output_norm_enc = ld
        .create_tensor("enc.output_norm.weight", &[n_embd], true)?
        .expect("enc.output_norm.weight");
    let output_norm = ld
        .create_tensor("output_norm.weight", &[n_embd], true)?
        .expect("output_norm.weight");

    // optional: reduced-vocab drafts ship their own lm head, full-vocab
    // drafts can share the target's via ctx_other (:166-171)
    let mut output = ld.create_tensor("output.weight", &[n_embd, n_vocab_draft], false)?;
    if output.is_none() && tok_embd.is_some() {
        // TENSOR_DUPLICATED: tie to the draft's own embeddings
        output = tok_embd;
    }

    // the DFlash2 selector head (:138-159): present iff selector_hidden is in
    // the file; the conv/selector hparams must then all be set
    let selector_meta = gguf.find_tensor("selector_hidden.weight").is_some();
    let (dflash_selector_prev, dflash_selector_next, dflash_selector_hidden) = if selector_meta {
        if dsv4.is_some() {
            return Err(
                "DFlash2 conv/selector tensors are a plain-backbone feature (the DSV4 DSpark \
                 backbone has no selector)"
                    .into(),
            );
        }
        let rank = selector_rank as i64;
        if rank <= 0
            || dflash_block_size == 0
            || selector_top_k == 0
            || conv_kernel_size == 0
            || conv_group_size == 0
        {
            return Err("DFlash2 model is missing conv/selector metadata".into());
            // :141-143
        }
        if n_embd % conv_group_size as i64 != 0 {
            return Err("DFlash2 hidden size must be divisible by conv_group_size".into());
            // :145-147
        }
        let top_k = selector_top_k as i64;
        if n_embd < top_k * (top_k + 1) {
            return Err("DFlash2 hidden size is too small for the selector lattice".into());
            // :148-150
        }

        let prev = ld
            .create_tensor("selector_predecessor.weight", &[rank, n_vocab], true)?
            .expect("selector_predecessor.weight"); // :152
        let next = ld
            .create_tensor("selector_successor.weight", &[rank, n_vocab], true)?
            .expect("selector_successor.weight"); // :153
        let hidden = ld
            .create_tensor("selector_hidden.weight", &[n_embd, rank], true)?
            .expect("selector_hidden.weight"); // :154
        eprintln!(
            "dflash: DFlash2 conv kernel = {conv_kernel_size}, group = {conv_group_size}, \
             selector rank = {selector_rank}, top-k = {selector_top_k}"
        ); // :156-158
        (Some(prev), Some(next), Some(hidden))
    } else {
        (None, None, None)
    };

    // ---- the per-layer tensor table ----
    let n_embd_k_gqa = n_embd_head_k * n_head_kv;
    let (dsv4_staged, layers) = match &dsv4 {
        Some(dp) => {
            // the DSV4 stage table (dflash.cpp:173-221)
            let q_lora_rank = gguf
                .get_u32("dflash.attention.q_lora_rank")
                .expect("q_lora_rank (read above)") as i64;
            let n_ff_exp = dp.n_ff_exp;
            let n_expert_shared =
                gguf.get_u32("dflash.expert_shared_count")
                    .expect("expert_shared_count (read above)") as i64;
            let n_embd_head = n_embd_head_k;
            let o_groups = dp.o_group_count;
            let o_lora_rank = dp.o_lora_rank;
            let hc_mult = dp.hc_mult;
            let hc_dim = hc_mult * n_embd;
            let hc_mix_dim = (2 + hc_mult) * hc_mult;

            let hc_head_fn = ld
                .create_tensor("output_hc_fn.weight", &[hc_dim, hc_mult], true)?
                .expect("output_hc_fn.weight"); // :184
            let hc_head_base = ld
                .create_tensor("output_hc_base.weight", &[hc_mult], true)?
                .expect("output_hc_base.weight"); // :185
            let hc_head_scale = ld
                .create_tensor("output_hc_scale.weight", &[1], true)?
                .expect("hc_head_scale"); // :186

            let mut st_layers = Vec::with_capacity(n_layer);
            for i in 0..n_layer as i32 {
                let attn_norm = ld
                    .create_tensor(&format!("blk.{i}.attn_norm.weight"), &[n_embd], true)?
                    .expect("attn_norm"); // :191
                let attn_sinks = ld
                    .create_tensor(&format!("blk.{i}.attn_sinks.weight"), &[n_head], true)?
                    .expect("attn_sinks"); // :192
                let wq_a = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_q_a.weight"),
                        &[n_embd, q_lora_rank],
                        true,
                    )?
                    .expect("wq_a"); // :193
                let attn_q_a_norm = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_q_a_norm.weight"),
                        &[q_lora_rank],
                        true,
                    )?
                    .expect("attn_q_a_norm"); // :194
                let wq_b = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_q_b.weight"),
                        &[q_lora_rank, n_head * n_embd_head],
                        true,
                    )?
                    .expect("wq_b"); // :195
                let wkv = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_kv.weight"),
                        &[n_embd, n_embd_head],
                        true,
                    )?
                    .expect("wkv"); // :196
                let attn_kv_norm = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_kv_a_norm.weight"),
                        &[n_embd_head],
                        true,
                    )?
                    .expect("attn_kv_norm"); // :197
                                             // wo_a: the file carries the 2-D `{n_head*n_embd_head/o_groups,
                                             // o_lora_rank*o_groups}` — reshaped to 3-D here
                                             // (TENSOR_ALLOW_RESHAPE, :198)
                let wo_a = ld.create_tensor_reshaped(
                    &format!("blk.{i}.attn_output_a.weight"),
                    &[n_head * n_embd_head / o_groups, o_lora_rank, o_groups],
                )?;
                let wo_b = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_output_b.weight"),
                        &[o_groups * o_lora_rank, n_embd],
                        true,
                    )?
                    .expect("wo_b"); // :199

                let hc_attn_fn = ld
                    .create_tensor(
                        &format!("blk.{i}.hc_attn_fn.weight"),
                        &[hc_dim, hc_mix_dim],
                        true,
                    )?
                    .expect("hc_attn_fn"); // :201
                let hc_attn_base = ld
                    .create_tensor(&format!("blk.{i}.hc_attn_base.weight"), &[hc_mix_dim], true)?
                    .expect("hc_attn_base"); // :202
                let hc_attn_scale = ld
                    .create_tensor(&format!("blk.{i}.hc_attn_scale.weight"), &[3], true)?
                    .expect("hc_attn_scale"); // :203
                let hc_ffn_fn = ld
                    .create_tensor(
                        &format!("blk.{i}.hc_ffn_fn.weight"),
                        &[hc_dim, hc_mix_dim],
                        true,
                    )?
                    .expect("hc_ffn_fn"); // :204
                let hc_ffn_base = ld
                    .create_tensor(&format!("blk.{i}.hc_ffn_base.weight"), &[hc_mix_dim], true)?
                    .expect("hc_ffn_base"); // :205
                let hc_ffn_scale = ld
                    .create_tensor(&format!("blk.{i}.hc_ffn_scale.weight"), &[3], true)?
                    .expect("hc_ffn_scale"); // :206

                let ffn_gate_inp = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_gate_inp.weight"),
                        &[n_embd, dp.n_expert],
                        true,
                    )?
                    .expect("ffn_gate_inp"); // :208
                let ffn_exp_probs_b = ld
                    .create_tensor(&format!("blk.{i}.exp_probs_b.bias"), &[dp.n_expert], true)?
                    .expect("ffn_exp_probs_b"); // :209
                let ffn_norm = ld
                    .create_tensor(&format!("blk.{i}.ffn_norm.weight"), &[n_embd], true)?
                    .expect("ffn_norm"); // :210

                let ffn_gate_exps = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_gate_exps.weight"),
                        &[n_embd, n_ff_exp, dp.n_expert],
                        true,
                    )?
                    .expect("ffn_gate_exps"); // :212
                let ffn_down_exps = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_down_exps.weight"),
                        &[n_ff_exp, n_embd, dp.n_expert],
                        true,
                    )?
                    .expect("ffn_down_exps"); // :213
                let ffn_up_exps = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_up_exps.weight"),
                        &[n_embd, n_ff_exp, dp.n_expert],
                        true,
                    )?
                    .expect("ffn_up_exps"); // :214

                let ffn_gate_shexp = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_gate_shexp.weight"),
                        &[n_embd, n_ff_exp * n_expert_shared],
                        true,
                    )?
                    .expect("ffn_gate_shexp"); // :216
                let ffn_down_shexp = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_down_shexp.weight"),
                        &[n_ff_exp * n_expert_shared, n_embd],
                        true,
                    )?
                    .expect("ffn_down_shexp"); // :217
                let ffn_up_shexp = ld
                    .create_tensor(
                        &format!("blk.{i}.ffn_up_shexp.weight"),
                        &[n_embd, n_ff_exp * n_expert_shared],
                        true,
                    )?
                    .expect("ffn_up_shexp"); // :218

                st_layers.push(DsparkDsv4LayerWeights {
                    attn_norm,
                    attn_sinks,
                    wq_a,
                    attn_q_a_norm,
                    wq_b,
                    wkv,
                    attn_kv_norm,
                    wo_a,
                    wo_b,
                    hc_attn_fn,
                    hc_attn_base,
                    hc_attn_scale,
                    hc_ffn_fn,
                    hc_ffn_base,
                    hc_ffn_scale,
                    ffn_gate_inp,
                    ffn_exp_probs_b,
                    ffn_norm,
                    ffn_gate_exps,
                    ffn_down_exps,
                    ffn_up_exps,
                    ffn_gate_shexp,
                    ffn_down_shexp,
                    ffn_up_shexp,
                });
            }

            (
                Some(DsparkDsv4Staged {
                    hc_head_fn,
                    hc_head_base,
                    hc_head_scale,
                    layers: st_layers,
                    params: dp.clone(),
                }),
                Vec::new(),
            )
        }
        None => {
            // the plain backbone's layer table (:223-258, the DFlash2 conv
            // pair at :249-257)
            let mut layers = Vec::with_capacity(n_layer);
            for i in 0..n_layer as i32 {
                let attn_norm = ld
                    .create_tensor(&format!("blk.{i}.attn_norm.weight"), &[n_embd], true)?
                    .expect("attn_norm");
                let wq = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_q.weight"),
                        &[n_embd, n_embd_head_k * n_head],
                        true,
                    )?
                    .expect("wq");
                let wk = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_k.weight"),
                        &[n_embd, n_embd_k_gqa],
                        true,
                    )?
                    .expect("wk");
                let wv = ld.create_tensor(
                    &format!("blk.{i}.attn_v.weight"),
                    &[n_embd, n_embd_k_gqa],
                    false,
                )?;
                let wo = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_output.weight"),
                        &[n_embd_head_k * n_head, n_embd],
                        true,
                    )?
                    .expect("wo");
                let attn_q_norm = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_q_norm.weight"),
                        &[n_embd_head_k],
                        true,
                    )?
                    .expect("attn_q_norm");
                let attn_k_norm = ld
                    .create_tensor(
                        &format!("blk.{i}.attn_k_norm.weight"),
                        &[n_embd_head_k],
                        true,
                    )?
                    .expect("attn_k_norm");
                let attn_post_norm = ld.create_tensor(
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &[n_embd],
                    false,
                )?;
                let ffn_post_norm =
                    ld.create_tensor(&format!("blk.{i}.post_ffw_norm.weight"), &[n_embd], false)?;
                let out_scale =
                    ld.create_tensor(&format!("blk.{i}.layer_output_scale.weight"), &[1], false)?;
                let rope_freqs = ld
                    .create_tensor(&format!("blk.{i}.rope_freqs.weight"), &[n_rot / 2], false)?
                    .filter(|_| i == 0); // TENSOR_DUPLICATED for i > 0 (:239)
                let attn_sinks =
                    ld.create_tensor(&format!("blk.{i}.attn_sinks.weight"), &[n_head], false)?;
                let ffn_norm = ld
                    .create_tensor(&format!("blk.{i}.ffn_norm.weight"), &[n_embd], true)?
                    .expect("ffn_norm");
                let ffn_gate = ld
                    .create_tensor(&format!("blk.{i}.ffn_gate.weight"), &[n_embd, n_ff], true)?
                    .expect("ffn_gate");
                let ffn_down = ld
                    .create_tensor(&format!("blk.{i}.ffn_down.weight"), &[n_ff, n_embd], true)?
                    .expect("ffn_down");
                let ffn_up = ld
                    .create_tensor(&format!("blk.{i}.ffn_up.weight"), &[n_embd, n_ff], true)?
                    .expect("ffn_up");

                // DFlash2's per-layer conv pair (:249-257)
                let (attn_conv_base, attn_conv_proj, ffn_conv_base, ffn_conv_proj) =
                    if selector_meta {
                        let kernel = conv_kernel_size as i64;
                        let groups = n_embd / conv_group_size as i64;
                        let projected = 2 * kernel * groups;
                        let ab = ld
                            .create_tensor(
                                &format!("blk.{i}.attn_conv_base"),
                                &[n_embd, kernel, 2],
                                true,
                            )?
                            .expect("attn_conv_base"); // :253 (no suffix — LLM_TENSOR_DFLASH_ATTN_CONV_BASE)
                        let ap = ld
                            .create_tensor(
                                &format!("blk.{i}.attn_conv_proj.weight"),
                                &[n_embd, projected],
                                true,
                            )?
                            .expect("attn_conv_proj"); // :254
                        let fb = ld
                            .create_tensor(
                                &format!("blk.{i}.ffn_conv_base"),
                                &[n_embd, kernel, 2],
                                true,
                            )?
                            .expect("ffn_conv_base"); // :255
                        let fp = ld
                            .create_tensor(
                                &format!("blk.{i}.ffn_conv_proj.weight"),
                                &[n_embd, projected],
                                true,
                            )?
                            .expect("ffn_conv_proj"); // :256
                        (Some(ab), Some(ap), Some(fb), Some(fp))
                    } else {
                        (None, None, None, None)
                    };

                layers.push(DflashLayerWeights {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    attn_q_norm,
                    attn_k_norm,
                    attn_post_norm,
                    ffn_post_norm,
                    out_scale,
                    rope_freqs,
                    attn_sinks,
                    ffn_norm,
                    ffn_gate,
                    ffn_down,
                    ffn_up,
                    dflash_attn_conv_base: attn_conv_base,
                    dflash_attn_conv_proj: attn_conv_proj,
                    dflash_ffn_conv_base: ffn_conv_base,
                    dflash_ffn_conv_proj: ffn_conv_proj,
                });
            }
            (None, layers)
        }
    };

    // done_getting_tensors (model.rs:1089-1096): every file tensor consumed
    let n_file_tensors = gguf.tensors.len();
    if ld.n_created != n_file_tensors {
        let mut missing: Vec<&str> = gguf
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| !ld.by_name.contains_key(*n))
            .collect();
        missing.sort_unstable();
        return Err(format!(
            "dflash draft has {n_file_tensors} tensors but the loader created {} (unknown: \
             {missing:?})",
            ld.n_created
        ));
    }

    // ---- the ctx_other tensors (dflash.cpp:679-687 / :799-808): own or the
    // target model's ----
    let tok_embd = match tok_embd {
        Some(t) => t,
        None => ld.create_tensor_in(
            target_gguf,
            &target_mmap,
            "token_embd.weight",
            &[n_embd_tgt, n_vocab],
        )?,
    };
    let output = match output {
        Some(t) => t,
        None => ld.create_tensor_in(
            target_gguf,
            &target_mmap,
            "output.weight",
            &[n_embd_tgt, n_vocab],
        )?,
    };

    let attn = AttnParams {
        n_head,
        n_head_kv,
        n_embd_head_k,
        n_embd_head_v,
        n_rot,
        // llama_model_rope_type's DFLASH arm (llama-model.cpp:3053-3059):
        // "DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy DFlash
        // backbones are NeoX" — dsv4_hc_mult > 0 ? NORM : NEOX
        rope_mode: if dsv4.is_some() {
            crate::hparams::LlamaRopeType::NORM as i32
        } else {
            crate::hparams::LlamaRopeType::NEOX as i32
        },
        n_ctx_orig: n_ctx_train,
        freq_base,
        freq_scale,
        ext_factor: -1.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        norm_eps: eps,
        use_flash_attn: fa,
    };

    let weights = DflashWeights {
        tok_embd: Some(tok_embd),
        output: Some(output),
        output_norm,
        output_norm_enc,
        fc,
        fc_s,
        d2t,
        dspark_markov_w1,
        dspark_markov_w2,
        dspark_markov_w2_s,
        dspark_conf_proj,
        dspark_conf_proj_b,
        dflash_selector_prev,
        dflash_selector_next,
        dflash_selector_hidden,
        dsv4: dsv4_staged,
        layers,
    };
    let params = DflashParams {
        attn,
        n_embd,
        n_embd_inp_enc,
        n_vocab,
        n_vocab_draft,
        target_layer_ids,
        n_embd_tgt,
        f_embedding_scale,
        f_attention_scale,
        f_attn_value_scale,
        f_logit_scale,
        f_final_logit_softcapping,
        ffn_gelu,
        block_size,
        sample_from_anchor,
        attention_causal,
        has_confidence_head,
        selector_top_k,
        dflash_block_size,
        conv_kernel_size,
        conv_group_size,
        selector_rank,
    };

    Ok(DflashDraft {
        ctx: ld.ctx,
        weights,
        params,
    })
}

// ---------------------------------------------------------------------------
// graph<false>, the KV-injection branch (dflash.cpp:608-677)
// ---------------------------------------------------------------------------

/// the embd-batch branch of `llama_model_dflash::graph<false>`
/// (dflash.cpp:608-677): fuse the target features through the encoder
/// (fc + enc.output_norm), then project + norm + rope K (and V, or
/// `rms_norm(K)` in the shared-KV mode) and scatter both into the draft cache
/// at the batch's rows. The fused features (`res->t_embd`, :673) ride the
/// returned `logits` slot — the batch has no output rows, so nothing is read.
pub fn build_dflash_inject_forward(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    kv: &KvCache,
    inp: &DecodeInputs,
    // the F32 `[n_embd_inp_enc, n_tokens]` target-feature input
    // (`llm_graph_input_embd::embd`, dflash.cpp:612)
    features: TensorId,
    n_tokens: usize,
) -> ForwardResult {
    let a = &p.attn;
    let t = n_tokens as i64;

    let mut graph = Graph::new(256);

    // no iswa for the plain drafts (hparams.swa_type == NONE at :582-586 —
    // the loader refuses the sectioned/iSWA variants)
    assert!(
        !kv.has_swa(),
        "dflash inject: the iswa layout is not ported"
    );

    // fuse the target features through the encoder (:621-623)
    let inp_target = features;
    ctx.set_name(inp_target, "inp_target_features"); // cb(..., :616)
    let inp_g = crate::adapter::lora_mm_s(ctx, w.fc, inp_target, w.fc_s);
    let inp_g = build_norm_rms_opt(ctx, inp_g, Some(w.output_norm_enc), a.norm_eps);
    ctx.set_name(inp_g, "inp_g_embeddings"); // cb(..., :623)

    for (il, lw) in w.layers.iter().enumerate() {
        let kcur = crate::adapter::lora_mm(ctx, lw.wk, inp_g); // :628
        let shared_kv = lw.wv.is_none(); // :629
        let vcur = match lw.wv {
            Some(wv) => crate::adapter::lora_mm(ctx, wv, inp_g), // :630
            None => kcur,
        };

        let kcur = ctx.reshape_3d(kcur, a.n_embd_head_k, a.n_head_kv, t); // :632
        let vcur = ctx.reshape_3d(vcur, a.n_embd_head_v, a.n_head_kv, t); // :633

        let kcur = build_norm_rms_opt(ctx, kcur, Some(lw.attn_k_norm), a.norm_eps); // :635
        let vcur = if shared_kv {
            ctx.rms_norm(vcur, a.norm_eps) // :636-638
        } else {
            vcur
        };
        let kcur = ctx.rope_ext(
            kcur,
            inp.pos,
            w.layers[0].rope_freqs,
            a.n_rot as i32,
            a.rope_mode,
            a.n_ctx_orig,
            a.freq_base,
            a.freq_scale,
            a.ext_factor,
            a.attn_factor,
            a.beta_fast,
            a.beta_slow,
        ); // :639 (build_rope, the non-MRoPE arm)
        ctx.set_name(kcur, &format!("Kcur_injected-{il}")); // cb(..., :640)
        ctx.set_name(vcur, &format!("Vcur_injected-{il}")); // cb(..., :641)

        // cpy_k / cpy_v (llama-kv-cache.cpp:1318/1353): merge head dims 0/1
        // into one row dim, one set_rows at the batch's row indices
        let k_rows = {
            let nb2 = ctx.nb(kcur)[2] as usize;
            ctx.view_2d(kcur, a.n_embd_head_k * a.n_head_kv, t, nb2, 0)
        };
        let v_rows = {
            let nb2 = ctx.nb(vcur)[2] as usize;
            ctx.view_2d(vcur, a.n_embd_head_v * a.n_head_kv, t, nb2, 0)
        };
        let k_dst = ctx.set_rows(kv.layers[il].k, k_rows, inp.row_idx); // :668
        let v_dst = ctx.set_rows(kv.layers[il].v, v_rows, inp.row_idx); // :669
        graph.build_forward(ctx, k_dst);
        graph.build_forward(ctx, v_dst);
    }

    // res->t_embd = inp_g (:673) — rides the logits slot; no t_h_nextn
    graph.build_forward(ctx, inp_g); // :675
    ForwardResult {
        logits: inp_g,
        graph,
        embd: None,
    }
}

// ---------------------------------------------------------------------------
// graph<false>, the noise-block branch (dflash.cpp:679-853)
// ---------------------------------------------------------------------------

/// the DSpark markov + confidence heads (`build_dspark_markov_head`,
/// dflash.cpp:295-406): bias the draft logits with the Markov head, chained
/// per block position by the greedy argmax; the confidence head's per-position
/// sigmoid lands in `t_h_nextn` (broadcast to n_embd rows, :394-401).
///
/// `n_seqs_unq` is the batch's block count (`g.ubatch.n_seqs_unq`, :320) and
/// `block_size` the trained block size from the metadata (:310-318).
fn build_dspark_markov_head(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    graph: &mut Graph,
    tokens: TensorId,
    // res->t_logits — the raw draft logits [n_vocab, n_tokens]
    base: TensorId,
    // res->t_embd — the result_norm rows [n_embd, n_tokens]
    t_embd: TensorId,
    n_seqs_unq: u32,
) -> (TensorId, Option<TensorId>) {
    let w1 = w.dspark_markov_w1.expect("dspark markov_w1");
    let w2 = w.dspark_markov_w2.expect("dspark markov_w2");

    // confidence head is optional (:304)
    let has_conf = w.dspark_conf_proj.is_some();

    let n_vocab = ctx.ne(base)[0];
    let n_tok = ctx.ne(base)[1];

    let block_size = p.block_size as i64;
    assert!(
        block_size > 0,
        "DSpark draft requires 'dflash.block_size' in GGUF metadata"
    );

    // bonus anchor (SpecForge exports): slot 0 is a bonus token (:315-318)
    let i_draft_beg: i64 = if p.sample_from_anchor { 0 } else { 1 };

    let n_blocks = n_seqs_unq as i64;
    assert!(
        n_blocks > 0 && n_tok % n_blocks == 0,
        "DSpark markov head requires equal-size blocks"
    );
    // runtime tokens per block in this ubatch (anchor + drafted positions),
    // bounded by training block_size (:322-326)
    let block_drafts = n_tok / n_blocks;
    if block_drafts > block_size {
        return (base, None);
    }

    // anchor (committed last) token of every block: token 0 of each block,
    // i.e. a strided view (:328-333); `ggml_cont_1d` makes it a dense 1-D index
    let token_stride = (block_drafts * (ctx.nb(tokens)[0] as i64)) as usize;
    let prev = ctx.view_2d(tokens, 1, n_blocks, token_stride, 0);
    let prev_cont = ctx.cont(prev);
    let mut prev = ctx.reshape_1d(prev_cont, n_blocks);

    let mut cat: Option<TensorId> = None;
    let mut cat_conf: Option<TensorId> = None;

    if !p.sample_from_anchor {
        // bonus anchor slot: pass the logits through unbiased, pad the
        // (unread) confidence column (:338-344)
        let base_stride = (block_drafts * ctx.nb(base)[1] as i64) as usize;
        let col = ctx.view_2d(base, n_vocab, n_blocks, base_stride, 0);
        cat = Some(ctx.cont(col));
        if has_conf {
            let c = ctx.view_2d(base, 1, n_blocks, base_stride, 0);
            let c = ctx.cont(c);
            cat_conf = Some(ctx.sigmoid(c));
        }
    }

    // the in-graph chain is greedy (argmax); sampling params affect only the
    // final token pick (:346-347)
    let base_stride = (block_drafts * ctx.nb(base)[1] as i64) as usize;
    for i in i_draft_beg..block_drafts {
        let w1_prev = ctx.get_rows(w1, prev); // [R, n_blocks] (:349)
        let bias = crate::adapter::lora_mm_s(ctx, w2, w1_prev, w.dspark_markov_w2_s); // :350
        if w.d2t.is_some() {
            unimplemented!(
                "dspark markov d2t scatter (dflash.cpp:351-359) — reduced-vocab DSpark drafts"
            )
        }

        // position i of every block: strided view [n_vocab, n_blocks] (:362)
        let base_i = ctx.view_2d(
            base,
            n_vocab,
            n_blocks,
            base_stride,
            i as usize * ctx.nb(base)[1] as usize,
        );
        let col = ctx.add(base_i, bias); // :363

        cat = Some(match cat {
            Some(c) => ctx.concat(c, col, 1), // :365
            None => col,
        });

        if has_conf {
            // confidence head input: predicts per-position acceptance (:368-380)
            let conf_inp = t_embd; // [n_embd, n_tok]
            let conf_inp_i = ctx.view_2d(
                conf_inp,
                ctx.ne(conf_inp)[0],
                n_blocks,
                (block_drafts * ctx.nb(conf_inp)[1] as i64) as usize,
                i as usize * ctx.nb(conf_inp)[1] as usize,
            );
            let conf_cont = ctx.cont(conf_inp_i);
            let feat = ctx.concat(conf_cont, w1_prev, 0); // :373
            let mut conf = ctx.mul_mat(w.dspark_conf_proj.expect("conf_proj"), feat); // :374
            if let Some(b) = w.dspark_conf_proj_b {
                conf = ctx.add(conf, b); // :375-377
            }
            let conf = ctx.sigmoid(conf); // :378
            cat_conf = Some(match cat_conf {
                Some(c) => ctx.concat(c, conf, 1), // :380
                None => conf,
            });
        }

        if i + 1 < block_drafts {
            prev = ctx.argmax(col); // :383-385
        }
    }

    // cat is position-major; restore ubatch block-major order (:388-391)
    let cat = cat.expect("dspark markov head produced no columns");
    let out = ctx.reshape_3d(cat, n_vocab, n_blocks, block_drafts);
    let out = ctx.permute(out, 0, 2, 1, 3); // [n_vocab, block_drafts, n_blocks]
    let out = ctx.reshape_2d(out, n_vocab, n_tok);

    let mut t_h_nextn = None;
    if has_conf {
        let conf = cat_conf.expect("dspark conf head produced no columns");
        let conf = ctx.reshape_3d(conf, 1, n_blocks, block_drafts);
        let conf = ctx.permute(conf, 0, 2, 1, 3);
        let conf = ctx.reshape_2d(conf, 1, n_tok);

        // broadcast the [1, n_tok] confidences to n_embd-wide rows to reuse
        // `llama_get_embeddings_nextn` (:398-400)
        let conf = ctx.repeat(conf, t_embd);
        t_h_nextn = Some(conf);
        graph.build_forward(ctx, conf); // :401
    }

    graph.build_forward(ctx, out); // :405
    (out, t_h_nextn)
}

// ---------------------------------------------------------------------------
// the DFlash2 conv + selector graphs (dflash.cpp:408-567)
// ---------------------------------------------------------------------------

/// `build_dflash2_conv` (dflash.cpp:408-474): the DFlash2 dynamic-bias
/// depthwise convolution — per block position `t`, the weight of tap `k` on
/// group `g` is `dynamic[k, g, side, t] + base[:, k, side]`, broadcast over
/// the group's channels; the output is the sum over taps of weight·shifted
/// values (a causal within-block depthwise conv whose kernel mixes the two
/// `side`s of the `base` tensor: side 0 pre-attention/FFN, side 1 post).
///
/// `side` is 0 (input) or 1 (output); `dynamic` is the lora_mm of the layer's
/// `conv_proj` weight — `[2*kernel*n_groups, n_tokens]`.
fn build_dflash2_conv(
    ctx: &mut Context,
    p: &DflashParams,
    hidden: TensorId,
    dynamic: TensorId,
    base: TensorId,
    side: i64,
    n_seqs_unq: u32,
) -> TensorId {
    let hidden_size = ctx.ne(hidden)[0];
    let n_tokens = ctx.ne(hidden)[1];
    let n_blocks = n_seqs_unq as i64;
    let kernel_size = p.conv_kernel_size as i64;
    let group_size = p.conv_group_size as i64;
    let n_groups = hidden_size / group_size;

    assert!(
        n_blocks > 0 && n_tokens % n_blocks == 0,
        "dflash2 conv: n_tokens % n_blocks"
    );
    assert!(side >= 0 && side < 2);

    let block_size = n_tokens / n_blocks;

    // ggml_cont copies even when the tensor is already contiguous (:427-433)
    let mut hidden = hidden;
    if !ggml::ops::is_contiguous_ctx(ctx, hidden) || ctx.ne(hidden)[1] != n_tokens {
        hidden = ctx.cont_2d(hidden, hidden_size, n_tokens);
    }
    let mut dynamic = dynamic;
    if !ggml::ops::is_contiguous_ctx(ctx, dynamic) || ctx.ne(dynamic)[1] != n_tokens {
        let d0 = ctx.ne(dynamic)[0];
        dynamic = ctx.cont_2d(dynamic, d0, n_tokens);
    }

    let blocks = ctx.reshape_3d(hidden, hidden_size, block_size, n_blocks); // :434
    let coeffs = ctx.reshape_4d(dynamic, n_groups, kernel_size, 2, n_tokens); // :435
    let coeffs_side = {
        let nb = ctx.nb(coeffs);
        ctx.view_3d(
            coeffs,
            n_groups,
            kernel_size,
            n_tokens,
            nb[1] as usize,
            nb[3] as usize,
            (side * nb[2] as i64) as usize,
        )
    }; // :436-437

    let mut coeff_all = ctx.cont(coeffs_side); // :439
    coeff_all = ctx.reshape_4d(coeff_all, 1, n_groups, kernel_size, n_tokens); // :440
    coeff_all = ctx.repeat_4d(coeff_all, group_size, n_groups, kernel_size, n_tokens); // :441

    let base_side = {
        let base_view = {
            let nb2 = ctx.nb(base)[2] as usize;
            ctx.view_1d(base, hidden_size * kernel_size, (side as usize) * nb2)
        }; // :444
        ctx.reshape_4d(base_view, group_size, n_groups, kernel_size, 1) // :443-445
    };

    let weight_all = ctx.add(coeff_all, base_side); // :447

    // taps at or past block_size only read the left padding and add nothing
    // (dflash.cpp:450-451, def4d406a)
    let n_taps = kernel_size.min(block_size);
    let mut result: Option<TensorId> = None;
    for tap in 0..n_taps {
        // the shifted values: `tap` zero rows prepended within every block —
        // ggml_pad_ext(previous, lp1=tap) (dflash.cpp:452-455, def4d406a; the
        // old concat-with-zeros form, same values)
        let mut values = blocks;
        if tap > 0 {
            let nb = ctx.nb(blocks);
            let previous = ctx.view_3d(
                blocks,
                hidden_size,
                block_size - tap,
                n_blocks,
                nb[1] as usize,
                nb[2] as usize,
                0,
            ); // :453-454
            values = ctx.pad_ext(previous, 0, 0, tap as i32, 0, 0, 0, 0, 0); // :455
        }
        let values = ctx.reshape_2d(values, hidden_size, n_tokens); // :463

        let weight = {
            let nb = ctx.nb(weight_all);
            let tap_view = ctx.view_4d(
                weight_all,
                group_size,
                n_groups,
                1,
                n_tokens,
                nb[1] as usize,
                nb[2] as usize,
                nb[3] as usize,
                (tap * nb[2] as i64) as usize,
            ); // :466-467
            let tap_cont = ctx.cont(tap_view);
            ctx.reshape_2d(tap_cont, hidden_size, n_tokens) // :465-468
        };

        let term = ctx.mul(weight, values); // :470
        result = Some(match result {
            Some(r) => ctx.add(r, term), // :471
            None => term,
        });
    }
    result.expect("dflash2 conv: kernel_size > 0 (validated at load)")
}

/// `build_dflash2_selector` (dflash.cpp:478-567): the top-k candidate lattice.
/// For every block position the row packs `[top_k candidate ids as F32 | top_k
/// × top_k pairwise transition scores]` (padded to n_embd); a position's
/// scores read only the candidate sets at pos-1 and pos, so a run of
/// positions scores in one batched matmul (`score_run`). The packed rows land
/// in `res->t_h_nextn` for the CPU-side walk (speculative.cpp:1236-1263).
///
/// Returns `(packed, logit_transformed)`-style plumbing through the ForwardResult
/// embd slot; `t_logits` (raw or markov-biased) is consumed in place.
fn build_dflash2_selector(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    graph: &mut Graph,
    tokens: TensorId,
    // res->t_logits — [n_vocab, n_tokens]
    t_logits: TensorId,
    // res->t_embd — the result_norm rows [n_embd, n_tokens]
    t_embd: TensorId,
    n_tokens: i64,
    n_seqs_unq: u32,
) -> TensorId {
    let n_embd = p.n_embd;
    let top_k = p.selector_top_k as i64;
    let rank = p.selector_rank as i64;
    let n_blocks = n_seqs_unq as i64;
    assert!(n_blocks > 0 && n_tokens % n_blocks == 0);
    assert_eq!(ctx.ne(t_logits)[1], n_tokens);

    let tokens_per_block = n_tokens / n_blocks;
    let block_size = tokens_per_block.min(p.dflash_block_size as i64); // :496
    let row_used = top_k + top_k * top_k; // :497

    let candidates = ctx.top_k(t_logits, top_k as i32); // :499
    let logits_rows = {
        let ne = ctx.ne(t_logits);
        ctx.reshape_3d(t_logits, 1, ne[0], n_tokens) // :500
    };
    let unary = {
        let gr = ctx.get_rows(logits_rows, candidates); // :502
        ctx.reshape_2d(gr, top_k, n_tokens)
    };
    let gate = crate::adapter::lora_mm(
        ctx,
        w.dflash_selector_hidden.expect("selector_hidden"),
        t_embd,
    ); // :503

    // Everything below indexes [.., tokens_per_block, n_blocks]: the block
    // position varies fastest, sequences are the outer dimension (:505-509)
    let cand_blk = ctx.reshape_3d(candidates, top_k, tokens_per_block, n_blocks);
    let unary_blk = ctx.reshape_3d(unary, top_k, tokens_per_block, n_blocks);
    let gate_blk = ctx.reshape_3d(gate, rank, tokens_per_block, n_blocks);

    // a position's score reads only the candidate sets at pos-1 and pos, so a
    // run of positions has no internal dependency and scores in one batched
    // matmul (:511-543)
    let score_run = |ctx: &mut Context, beg_pos: i64, n_pos: i64, pred_ids: TensorId| -> TensorId {
        let (cand_run, unary_run, gate_run) = {
            let (cn1, cn2) = (ctx.nb(cand_blk)[1] as i64, ctx.nb(cand_blk)[2] as i64);
            let cv = ctx.view_3d(
                cand_blk,
                top_k,
                n_pos,
                n_blocks,
                cn1 as usize,
                cn2 as usize,
                (beg_pos * cn1) as usize,
            ); // :514-515
            let cand_run = ctx.cont(cv);
            let (un1, un2) = (ctx.nb(unary_blk)[1] as i64, ctx.nb(unary_blk)[2] as i64);
            let uv = ctx.view_3d(
                unary_blk,
                top_k,
                n_pos,
                n_blocks,
                un1 as usize,
                un2 as usize,
                (beg_pos * un1) as usize,
            ); // :516-517
            let unary_run = ctx.cont(uv);
            let (gn1, gn2) = (ctx.nb(gate_blk)[1] as i64, ctx.nb(gate_blk)[2] as i64);
            let gv = ctx.view_3d(
                gate_blk,
                rank,
                n_pos,
                n_blocks,
                gn1 as usize,
                gn2 as usize,
                (beg_pos * gn1) as usize,
            ); // :518-519
            let gate_run = ctx.cont(gv);
            (cand_run, unary_run, gate_run)
        };

        let n_pred = ctx.ne(pred_ids)[0] / (n_pos * n_blocks); // :521

        let successor = {
            let flat = ctx.reshape_1d(cand_run, top_k * n_pos * n_blocks); // :524
            let gr = ctx.get_rows(w.dflash_selector_next.expect("selector_next"), flat);
            ctx.reshape_4d(gr, rank, top_k, n_pos, n_blocks) // :523-525
        };
        let predecessor = {
            let gr = ctx.get_rows(w.dflash_selector_prev.expect("selector_prev"), pred_ids);
            ctx.reshape_4d(gr, rank, n_pred, n_pos, n_blocks) // :526-528
        };

        let gate_bcast = ctx.reshape_4d(gate_run, rank, 1, n_pos, n_blocks); // :530
        let rep = ctx.repeat(gate_bcast, predecessor);
        let cond = ctx.mul(predecessor, rep); // :531
        let mut score = ctx.mul_mat(successor, cond); // :532
        if n_pred == 1 {
            score = ctx.repeat_4d(score, top_k, top_k, n_pos, n_blocks); // :533-535
        }
        let unary_bcast = ctx.reshape_4d(unary_run, top_k, 1, n_pos, n_blocks); // :536
        let rep_u = ctx.repeat(unary_bcast, score);
        score = ctx.add(score, rep_u); // :537

        let cast_cand = ctx.cast(cand_run, ggml::types::GgmlType::F32); // :540
        let score_3d = ctx.reshape_3d(score, top_k * top_k, n_pos, n_blocks); // :541
        let row = ctx.concat(cast_cand, score_3d, 0); // :539-541
        ctx.pad(row, (n_embd - row_used) as i32, 0, 0, 0) // :542
    };

    let packed_t = ctx.new_tensor_3d(ggml::types::GgmlType::F32, n_embd, 1, n_blocks);
    let mut packed = ctx.fill(packed_t, 0.0); // :545-546

    if block_size > 1 {
        // Position 1 alone: its predecessor is the anchor token, one id per
        // sequence rather than a candidate set (:548-554)
        let anchor_view = {
            let nb0 = ctx.nb(tokens)[0] as usize;
            ctx.view_2d(tokens, 1, n_blocks, tokens_per_block as usize * nb0, 0)
            // :551-552
        };
        let anchor_cont = ctx.cont(anchor_view);
        let anchor_ids = ctx.reshape_1d(anchor_cont, n_blocks);
        let col = score_run(ctx, 1, 1, anchor_ids);
        packed = ctx.concat(packed, col, 1);
    }
    if block_size > 2 {
        let prev_ids = {
            let nb1 = ctx.nb(cand_blk)[1] as usize;
            let nb2 = ctx.nb(cand_blk)[2] as usize;
            let view = ctx.view_3d(cand_blk, top_k, block_size - 2, n_blocks, nb1, nb2, nb1); // :557-558 (offset = cand_blk->nb[1], i.e. skip position 0)
            let cont = ctx.cont(view);
            ctx.reshape_1d(cont, top_k * (block_size - 2) * n_blocks) // :556-559
        };
        let col = score_run(ctx, 2, block_size - 2, prev_ids);
        packed = ctx.concat(packed, col, 1); // :560
    }

    let packed = ctx.reshape_2d(packed, n_embd, block_size * n_blocks); // :563
    ctx.set_name(packed, "dflash2_lattice"); // cb(..., :564)
    graph.build_forward(ctx, packed); // :566
    packed // res->t_h_nextn (:565)
}

// ---------------------------------------------------------------------------
// the DSV4 DSpark backbone (dflash.cpp:855-1028) — the deepseek4 stage stack
// (hyper-connection + MLA + MoE) over the iswa ring cache whose SWA half
// carries every layer (set_swa_pattern(0), all is_swa_impl true, :82-87).
// The hc / MLA / MoE stage bodies are local re-instantiations of the
// graph_arch.rs deepseek4 builders (private there — byte-compatible copies
// with the C file:line carried over).
// ---------------------------------------------------------------------------

/// `dsv4_elem_offset` (deepseek4.cpp:192-195)
fn dsv4_elem_offset(ctx: &Context, t: TensorId, i0: i64) -> usize {
    ctx.ty(t).row_size(i0 as usize)
}

/// `dsv4_hc_affine` (deepseek4.cpp:280-288)
fn dsv4_hc_affine(ctx: &mut Context, x: TensorId, scale: TensorId, base: TensorId) -> TensorId {
    let x = ctx.mul(x, scale);
    ctx.add(x, base)
}

/// `build_hc_pre(x, weights, il)` — the fused stream mix
/// (deepseek4.cpp:290-315; cparams.fused_dsv4_hc_pre is the default)
fn dsv4_hc_pre_stream(ctx: &mut Context, x: TensorId, weights: TensorId) -> TensorId {
    ctx.dsv4_hc_pre(x, weights)
}

/// `llama_model_deepseek4::graph::build_hc_pre(x, hc_fn, hc_scale, hc_base,
/// &post, &comb, il)` (deepseek4.cpp:354-410) — returns (cur, post, comb)
fn build_hc_pre_dspark_dsv4(
    ctx: &mut Context,
    x: TensorId,
    hc_fn: TensorId,
    hc_scale: TensorId,
    hc_base: TensorId,
    p: &DsparkDsv4Params,
    norm_eps: f32,
    il: i32,
) -> (TensorId, TensorId, TensorId) {
    let hc = p.hc_mult;
    let n_embd = ctx.ne(x)[0];
    let nt = ctx.ne(x)[2];

    assert_eq!(hc, 4, "dspark dsv4: hc == 4 (deepseek4.cpp:367)");
    assert_eq!(ctx.ne(hc_fn)[1], (2 + hc) * hc);

    let flat = ctx.reshape_2d(x, n_embd * hc, nt);
    let flat_norm = ctx.rms_norm(flat, norm_eps);
    let mixes = ctx.mul_mat(hc_fn, flat_norm);
    ctx.set_name(mixes, &format!("hc_mixes-{il}"));

    let scale_pre = ctx.view_1d(hc_scale, 1, dsv4_elem_offset(ctx, hc_scale, 0));
    let scale_post = ctx.view_1d(hc_scale, 1, dsv4_elem_offset(ctx, hc_scale, 1));
    let base_pre = ctx.view_1d(hc_base, hc, dsv4_elem_offset(ctx, hc_base, 0));
    let base_post = ctx.view_1d(hc_base, hc, dsv4_elem_offset(ctx, hc_base, hc));

    let nb1 = ctx.nb(mixes)[1] as usize;
    let pre_v = ctx.view_2d(mixes, hc, nt, nb1, 0);
    let pre = dsv4_hc_affine(ctx, pre_v, scale_pre, base_pre);
    let pre = ctx.sigmoid(pre);
    let pre = ctx.scale_bias(pre, 1.0, p.hc_eps);
    ctx.set_name(pre, &format!("hc_pre-{il}"));

    let off_post = dsv4_elem_offset(ctx, mixes, hc);
    let post_v = ctx.view_2d(mixes, hc, nt, nb1, off_post);
    let post = dsv4_hc_affine(ctx, post_v, scale_post, base_post);
    let post = ctx.sigmoid(post);
    let post = ctx.scale(post, 2.0);
    ctx.set_name(post, &format!("hc_post-{il}"));

    // the fused comb (cparams.fused_dsv4_hc_comb, deepseek4.cpp:393-396)
    let comb = ctx.dsv4_hc_comb(mixes, hc_scale, hc_base, p.hc_eps, p.hc_sinkhorn_iters);
    ctx.set_name(comb, &format!("hc_comb-{il}"));

    let result = dsv4_hc_pre_stream(ctx, x, pre);
    (result, post, comb)
}

/// `build_hc_post` (deepseek4.cpp:412-447; fused branch :421-424)
fn build_hc_post_dspark_dsv4(
    ctx: &mut Context,
    x: TensorId,
    residual: TensorId,
    post: TensorId,
    comb: TensorId,
) -> TensorId {
    ctx.dsv4_hc_post(x, residual, post, Some(comb))
}

/// `build_hc_head` (deepseek4.cpp:449-469)
fn build_hc_head_dspark_dsv4(
    ctx: &mut Context,
    x: TensorId,
    hc_fn: TensorId,
    hc_scale: TensorId,
    hc_base: TensorId,
    p: &DsparkDsv4Params,
    norm_eps: f32,
) -> TensorId {
    let hc = p.hc_mult;
    let n_embd = ctx.ne(x)[0];
    let nt = ctx.ne(x)[2];

    let flat = ctx.reshape_2d(x, n_embd * hc, nt);
    let flat_norm = ctx.rms_norm(flat, norm_eps);
    let mixes = ctx.mul_mat(hc_fn, flat_norm);
    ctx.set_name(mixes, "hc_head_mixes");

    let pre = dsv4_hc_affine(ctx, mixes, hc_scale, hc_base);
    let pre = ctx.sigmoid(pre);
    let pre = ctx.scale_bias(pre, 1.0, p.hc_eps);
    ctx.set_name(pre, "hc_head_pre");

    dsv4_hc_pre_stream(ctx, x, pre)
}

/// `build_attn_mha` for the DSV4 MLA attention (llama-graph.cpp:2602-2738
/// with v == k, sinks, no v_mla, no softcap) — the local twin of graph_arch's
/// `attn_dsv4`. q [H, n_head, T] F32, k_all [H, 1, n_kv, 1] F16,
/// kq_mask [n_kv, T], sinks [n_head].
fn attn_dspark_dsv4(
    ctx: &mut Context,
    q: TensorId,
    k_all: TensorId,
    kq_mask: TensorId,
    sinks: TensorId,
    kq_scale: f32,
    use_fa: bool,
) -> TensorId {
    let t = ctx.ne(q)[2];
    let n_embd_head_v = ctx.ne(k_all)[0];
    let n_head = ctx.ne(q)[1];

    if use_fa {
        // llama-graph.cpp:2626-2669 + the sinks of :2646-2647
        return graph::flash_attn_core_sinks(
            ctx,
            q,
            k_all,
            k_all,
            kq_mask,
            Some(sinks),
            kq_scale,
            0.0,
            0.0,
        );
    }

    // non-FA (llama-graph.cpp:2670-2733)
    let q = ctx.permute(q, 0, 2, 1, 3);
    let k_view = ctx.permute(k_all, 0, 2, 1, 3);
    let mut kq = ctx.mul_mat(k_view, q);
    kq = ctx.soft_max_ext(kq, Some(kq_mask), kq_scale, 0.0);
    ctx.soft_max_add_sinks(kq, Some(sinks));

    let v_view = ctx.permute(k_all, 0, 2, 1, 3);
    let v_t = ctx.transpose(v_view);
    let v_c = ctx.cont(v_t);
    let kqv = ctx.mul_mat(v_c, kq); // [head_dim, T, heads]

    let kqv = ctx.permute(kqv, 0, 2, 1, 3);
    ctx.cont_2d(kqv, n_embd_head_v * n_head, t)
}

/// `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)` over the shared experts with
/// the shexp swiglu clamp (deepseek4.cpp's `build_ffn_shexp` — the trunk
/// twin of graph_arch's `build_ffn_shexp_dsv4`, llama-graph.cpp:1829-1842)
fn build_ffn_shexp_dspark_dsv4(
    ctx: &mut Context,
    cur: TensorId,
    lw: &DsparkDsv4LayerWeights,
    p: &DsparkDsv4Params,
    il: usize,
) -> TensorId {
    let gate = crate::adapter::lora_mm(ctx, lw.ffn_gate_shexp, cur);
    let up = crate::adapter::lora_mm(ctx, lw.ffn_up_shexp, cur);
    let limit = p.swiglu_clamp_shexp[il];
    let prod = if limit > 1e-6 {
        ctx.swiglu_clamp(gate, up, limit)
    } else {
        ctx.swiglu_split(gate, up)
    };
    crate::adapter::lora_mm(ctx, lw.ffn_down_shexp, prod)
}

/// `build_moe_ffn` with the SQRT_SOFTPLUS gating and the swiglu_clamp expert
/// FFN (llama-graph.cpp:1993-2368 — the local twin of graph_arch's
/// `build_moe_ffn_dsv4`; dflash.cpp:972-982's call has no hash-layer input)
#[allow(clippy::too_many_arguments)]
fn build_moe_ffn_dspark_dsv4(
    ctx: &mut Context,
    graph: &mut Graph,
    cur: TensorId,
    lw: &DsparkDsv4LayerWeights,
    p: &DsparkDsv4Params,
    il: usize,
) -> TensorId {
    let n_embd = ctx.ne(cur)[0];
    let n_tokens = ctx.ne(cur)[1];
    let n_expert = p.n_expert;
    let n_expert_used = p.n_expert_used;

    // routing (llama-graph.cpp:2021-2057): probs = sqrt(softplus(logits))
    let logits = crate::adapter::lora_mm(ctx, lw.ffn_gate_inp, cur);
    let sp = ctx.softplus(logits);
    let probs = ctx.sqrt(sp);

    // e-score bias steers only the top-k (:2062-2067)
    let selection_probs = ctx.add(probs, lw.ffn_exp_probs_b);
    let selected = ctx.argsort_top_k(selection_probs, n_expert_used as i32);

    let probs3 = ctx.reshape_3d(probs, 1, n_expert, n_tokens);
    let mut weights = ctx.get_rows(probs3, selected);
    if p.expert_weights_norm {
        let w2 = ctx.reshape_2d(weights, n_expert_used, n_tokens);
        let sum = ctx.sum_rows(w2);
        let sum = ctx.clamp(sum, 6.103515625e-5, f32::INFINITY);
        let w2 = ctx.div(w2, sum);
        weights = ctx.reshape_3d(w2, 1, n_expert_used, n_tokens);
    }
    if p.expert_weights_scale != 0.0 && p.expert_weights_scale != 1.0 {
        weights = ctx.scale(weights, p.expert_weights_scale);
    }
    graph.build_forward(ctx, weights);

    // experts (llama-graph.cpp:2190-2234 — the separate gate/up path)
    let cur3 = ctx.reshape_3d(cur, n_embd, 1, n_tokens);
    let up = crate::adapter::lora_mm_id(ctx, lw.ffn_up_exps, cur3, selected);
    let gate = crate::adapter::lora_mm_id(ctx, lw.ffn_gate_exps, cur3, selected);

    // the clamp'd swiglu (:2225-2233, the DEEPSEEK4 branch)
    let limit = p.swiglu_clamp_exp[il];
    let act = if limit > 1e-6 {
        ctx.swiglu_clamp(gate, up, limit)
    } else {
        ctx.swiglu_split(gate, up)
    };

    let mut experts = crate::adapter::lora_mm_id(ctx, lw.ffn_down_exps, act, selected);
    experts = ctx.mul(experts, weights);
    graph.build_forward(ctx, experts);

    // aggregate the k expert views (:2329-2358)
    let nb1 = ctx.nb(experts)[1] as usize;
    let nb2 = ctx.nb(experts)[2] as usize;
    let mut moe_out = ctx.view_2d(experts, n_embd, n_tokens, nb2, 0);
    graph.build_forward(ctx, moe_out);
    for i in 1..n_expert_used as usize {
        let vi = ctx.view_2d(experts, n_embd, n_tokens, nb2, i * nb1);
        graph.build_forward(ctx, vi);
        moe_out = ctx.add(moe_out, vi);
        graph.build_forward(ctx, moe_out);
    }
    if n_expert_used == 1 {
        moe_out = ctx.cont(moe_out);
    }
    moe_out
}

/// the embd-batch branch of `llama_model_dflash::graph_dsv4`
/// (dflash.cpp:870-910): fuse the target features through the encoder
/// (fc + enc.output_norm), then per stage `kv_norm(wkv(main_x))` with rope on
/// the trailing dims (the uncompressed-attention rope of deepseek4.cpp:
/// 903-910) injected into the ring cache's SWA half. The fused features ride
/// the returned `logits` slot (`res->t_embd`, :906).
pub fn build_dspark_dsv4_inject_forward(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    staged: &DsparkDsv4Staged,
    kv: &KvCache,
    inp: &DecodeInputs,
    features: TensorId,
    n_tokens: usize,
) -> ForwardResult {
    let a = &p.attn;
    let t = n_tokens as i64;
    let n_embd_head = a.n_embd_head_k;
    let n_embd_head_rope = a.n_rot;
    let n_embd_head_nope = n_embd_head - n_embd_head_rope;

    assert!(kv.has_swa(), "dspark dsv4 inject: the iswa ring pair");
    let step = kv
        .swa_step
        .as_ref()
        .expect("dspark dsv4 inject: swa step inputs");

    let mut graph = Graph::new(256);

    // fuse the target features through the encoder (:881-884)
    let inp_target = features;
    ctx.set_name(inp_target, "inp_target_features"); // cb(..., :877)
    let inp_g = crate::adapter::lora_mm_s(ctx, w.fc, inp_target, w.fc_s);
    let inp_g = build_norm_rms_opt(ctx, inp_g, Some(w.output_norm_enc), a.norm_eps);
    ctx.set_name(inp_g, "inp_g_embeddings"); // cb(..., :884)

    for (il, lw) in staged.layers.iter().enumerate() {
        // main-track KV: kv_norm(wkv(main_x)) (:889-893)
        let kvv = crate::adapter::lora_mm(ctx, lw.wkv, inp_g); // :891
        let kvv = build_norm_rms_opt(ctx, kvv, Some(lw.attn_kv_norm), a.norm_eps); // :892
        let kvv = ctx.reshape_3d(kvv, n_embd_head, 1, t); // :893

        // the rope of the uncompressed layers (:895-896 — freq_base, scale 1,
        // ext 0, attn 1, beta 0) + the MLA nope offset (:897)
        let kvv = ctx.rope_ext(
            kvv,
            inp.pos,
            None,
            n_embd_head_rope as i32,
            a.rope_mode,
            0,
            a.freq_base,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        let kvv = ctx.rope_set_offset(kvv, n_embd_head_nope as i32);
        ctx.set_name(kvv, &format!("kv_injected-{il}")); // cb(..., :898)

        // (:900-903) — no k_rot for the DFLASH arch (attn_rot_k's arch list,
        // llama-kv-cache.cpp:327-332); cpy_k through the swa side's rows
        let nb2 = ctx.nb(kvv)[2] as usize;
        let kv_rows = ctx.view_2d(kvv, n_embd_head, t, nb2, 0);
        let dst = ctx.set_rows(kv.layers[il].k, kv_rows, step.row_idx);
        graph.build_forward(ctx, dst);
    }

    // res->t_embd = inp_g (:906) — rides the logits slot
    graph.build_forward(ctx, inp_g); // :908
    ForwardResult {
        logits: inp_g,
        graph,
        embd: None,
    }
}

/// the token-batch branch of `llama_model_dflash::graph_dsv4`
/// (dflash.cpp:912-1028): the noise block through the full DSV4 stage stack —
/// hc_init, per stage hc_pre → attn_norm → MLA attention over the ring →
/// hc_post, hc_pre → ffn_norm → MoE + shexp → hc_post — then hc_head,
/// output_norm, the lm head and (with DSpark weights) the markov/conf heads.
/// The returned `embd` slot carries `res->t_h_nextn` (the markov head's
/// confidence rows; the C sets `res->t_embd` to the pre-norm collapsed hidden
/// state at :1003 — the confidence head input, not extracted by the context).
pub fn build_dspark_dsv4_noise_forward(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    staged: &DsparkDsv4Staged,
    kv: &KvCache,
    inp: &DecodeInputs,
    n_tokens: usize,
    n_seqs_unq: u32,
) -> ForwardResult {
    let a = &p.attn;
    let t = n_tokens as i64;
    let n_embd_head = a.n_embd_head_k;
    let n_embd_head_rope = a.n_rot;
    let n_embd_head_nope = n_embd_head - n_embd_head_rope;
    let n_groups = staged.params.o_group_count;
    let n_heads_group = a.n_head / n_groups;
    let o_lora_rank = staged.params.o_lora_rank;
    let o_group_dim = n_heads_group * n_embd_head;

    assert!(kv.has_swa(), "dspark dsv4 noise: the iswa ring pair");
    let step = kv
        .swa_step
        .as_ref()
        .expect("dspark dsv4 noise: swa step inputs");
    let n_kv = kv.n_kv_swa();

    assert_eq!(n_embd_head, a.n_embd_head_v);
    assert_eq!(a.n_head % n_groups, 0);

    // the stage rope — the raw/uncompressed parameters
    // (deepseek4.cpp:903-910, use_compress_rope == false)
    let freq_base_l = a.freq_base;
    let freq_scale_l = 1.0;
    let ext_factor_l = 0.0;
    let attn_factor_l = 1.0f32; // dsv4_rope_attn_factor(1.0, 0.0)
    let beta_fast_l = 0.0;
    let beta_slow_l = 0.0;
    let n_ctx_orig_l = 0;
    let rope_l = |ctx: &mut Context, x: TensorId| -> TensorId {
        let x = ctx.rope_ext(
            x,
            inp.pos,
            None,
            n_embd_head_rope as i32,
            a.rope_mode,
            n_ctx_orig_l,
            freq_base_l,
            freq_scale_l,
            ext_factor_l,
            attn_factor_l,
            beta_fast_l,
            beta_slow_l,
        );
        ctx.rope_set_offset(x, n_embd_head_nope as i32)
    };

    let mut graph = Graph::new(1024);

    // tok_embd — own or the target model's (:912-920)
    let tok_embd = w.tok_embd.expect("dspark dsv4 tok_embd (loader-resolved)");
    let inp_l = ctx.get_rows(tok_embd, inp.tokens); // :929
    ctx.set_name(inp_l, "inp_noise_embd"); // cb(..., :930)

    // hc_init (:934-937)
    let hc = staged.params.hc_mult;
    let inp3 = ctx.reshape_3d(inp_l, p.n_embd, 1, t);
    let mut inp_l = ctx.repeat_4d(inp3, p.n_embd, hc, t, 1);
    ctx.set_name(inp_l, "hc_init"); // cb(..., :937)

    for (il, lw) in staged.layers.iter().enumerate() {
        // the attention half (:942-959)
        let residual = inp_l;
        let (cur, post, comb) = build_hc_pre_dspark_dsv4(
            ctx,
            inp_l,
            lw.hc_attn_fn,
            lw.hc_attn_scale,
            lw.hc_attn_base,
            &staged.params,
            a.norm_eps,
            il as i32,
        );
        ctx.set_name(cur, &format!("hc_attn_pre-{il}")); // cb(..., :951)

        let cur = build_norm_rms_opt(ctx, cur, Some(lw.attn_norm), a.norm_eps); // :953
        ctx.set_name(cur, &format!("attn_norm-{il}")); // cb(..., :954)

        // build_attention (deepseek4.cpp:879-1234's inp_mtp arm, :1191-1199
        // + the q/kv projections of :912-936)
        let mut qr = crate::adapter::lora_mm(ctx, lw.wq_a, cur); // :912
        qr = build_norm_rms_opt(ctx, qr, Some(lw.attn_q_a_norm), a.norm_eps); // :913-914
        let mut q = crate::adapter::lora_mm(ctx, lw.wq_b, qr); // :916
        q = ctx.reshape_3d(q, n_embd_head, a.n_head, t); // :917
        q = ctx.rms_norm(q, a.norm_eps); // :918 — per-head, unweighted
        q = rope_l(ctx, q); // :919-922

        let mut kvt = crate::adapter::lora_mm(ctx, lw.wkv, cur); // :925
        kvt = build_norm_rms_opt(ctx, kvt, Some(lw.attn_kv_norm), a.norm_eps); // :926
        kvt = ctx.reshape_3d(kvt, n_embd_head, 1, t); // :927
        kvt = rope_l(ctx, kvt); // :928-931

        let kq_scale = 1.0 / (n_embd_head as f32).sqrt(); // :1197
        graph.build_forward(ctx, q); // llama-graph.cpp:3186
        let nb2 = ctx.nb(kvt)[2] as usize;
        let kv_rows = ctx.view_2d(kvt, n_embd_head, t, nb2, 0);
        let dst = ctx.set_rows(kv.layers[il].k, kv_rows, step.row_idx); // :3200
        graph.build_forward(ctx, dst);

        // MLA-style: the cached K is used as V (llama-graph.cpp:3209-3217)
        let k_all = kv.get_k(ctx, il, n_embd_head, 1, n_kv);
        let out = attn_dspark_dsv4(
            ctx,
            q,
            k_all,
            step.kq_mask,
            lw.attn_sinks,
            kq_scale,
            a.use_flash_attn,
        );
        ctx.set_name(out, &format!("attn_raw-{il}")); // cb(..., :1199)

        // attn_derope (:1204-1208)
        let out = ctx.reshape_3d(out, n_embd_head, a.n_head, t);
        let out = {
            let x = ctx.rope_ext_back(
                out,
                inp.pos,
                None,
                n_embd_head_rope as i32,
                a.rope_mode,
                n_ctx_orig_l,
                freq_base_l,
                freq_scale_l,
                ext_factor_l,
                attn_factor_l,
                beta_fast_l,
                beta_slow_l,
            );
            ctx.rope_set_offset(x, n_embd_head_nope as i32)
        };
        ctx.set_name(out, &format!("attn_derope-{il}"));

        // the o_group/o_lora output projection (:1210-1218)
        let out = ctx.reshape_3d(out, o_group_dim, n_groups, t);
        let out = ctx.permute(out, 0, 2, 1, 3);
        let mut oa = ctx.mul_mat(lw.wo_a, out);
        ctx.set_name(oa, &format!("attn_wo_a-{il}")); // cb(..., :1215)
        oa = ctx.permute(oa, 0, 2, 1, 3);
        let oa = ctx.cont_2d(oa, o_lora_rank * n_groups, t);
        let out = crate::adapter::lora_mm(ctx, lw.wo_b, oa); // :1217
        ctx.set_name(out, &format!("attn_out-{il}")); // cb(..., :1218)

        inp_l = build_hc_post_dspark_dsv4(ctx, out, residual, post, comb); // :958
        ctx.set_name(inp_l, &format!("hc_attn_post-{il}")); // cb(..., :959)

        // the FFN half (:961-996)
        let residual = inp_l;
        let (cur, post, comb) = build_hc_pre_dspark_dsv4(
            ctx,
            inp_l,
            lw.hc_ffn_fn,
            lw.hc_ffn_scale,
            lw.hc_ffn_base,
            &staged.params,
            a.norm_eps,
            il as i32,
        );
        ctx.set_name(cur, &format!("hc_ffn_pre-{il}")); // cb(..., :967)

        let cur = build_norm_rms_opt(ctx, cur, Some(lw.ffn_norm), a.norm_eps); // :969
        ctx.set_name(cur, &format!("ffn_norm-{il}")); // cb(..., :970)

        let moe_out = build_moe_ffn_dspark_dsv4(ctx, &mut graph, cur, lw, &staged.params, il); // :972-982
        ctx.set_name(moe_out, &format!("ffn_moe_out-{il}")); // cb(..., :983)
        let ffn_shexp = build_ffn_shexp_dspark_dsv4(ctx, cur, lw, &staged.params, il); // :985-989
        ctx.set_name(ffn_shexp, &format!("ffn_shexp-{il}")); // cb(..., :990)
        let cur = ctx.add(moe_out, ffn_shexp); // :992
        ctx.set_name(cur, &format!("ffn_out-{il}")); // cb(..., :993)

        inp_l = build_hc_post_dspark_dsv4(ctx, cur, residual, post, comb); // :995
        ctx.set_name(inp_l, &format!("l_out-{il}")); // cb(..., :996)
    }

    // hc_head (:999-1000); res->t_embd = the pre-norm collapsed hidden state
    // (:1002-1003 — the confidence head input)
    let cur = build_hc_head_dspark_dsv4(
        ctx,
        inp_l,
        staged.hc_head_fn,
        staged.hc_head_scale,
        staged.hc_head_base,
        &staged.params,
        a.norm_eps,
    );
    ctx.set_name(cur, "hc_head"); // cb(..., :1000)
    let t_embd = cur;

    let cur = build_norm_rms_opt(ctx, cur, Some(w.output_norm), a.norm_eps); // :1005
    ctx.set_name(cur, "result_norm"); // cb(..., :1006)

    // lm_head — own or the target model's (:1008-1019)
    let output = w.output.expect("dspark dsv4 output (loader-resolved)");
    let cur = crate::adapter::lora_mm(ctx, output, cur);
    ctx.set_name(cur, "result_output"); // cb(..., :1020)

    graph.build_forward(ctx, cur); // :1023

    // DSpark: the Markov head (:1025-1027) — shared with the plain backbone
    let (logits, t_h_nextn) = if w.dspark_markov_w1.is_some() {
        build_dspark_markov_head(ctx, w, p, &mut graph, inp.tokens, cur, t_embd, n_seqs_unq)
    } else {
        (cur, None)
    };

    ForwardResult {
        logits,
        graph,
        embd: t_h_nextn,
    }
}

/// the token-batch branch of `llama_model_dflash::graph<false>`
/// (dflash.cpp:679-853): embed the noise block, run the non-causal
/// cache-aware attention + FFN stack, then output_norm → lm_head (+ the d2t
/// scatter). With DSpark weights the markov head biases the logits and the
/// confidence rows become the nextn output.
///
/// The returned `embd` slot carries `res->t_h_nextn` — `Some` only for the
/// DSpark confidence head (the plain drafts set no t_h_nextn; the port's
/// nextn extraction skips the pass, exactly like the C's `t_h_nextn &&`
/// check of llama-context.cpp:2017).
#[allow(clippy::too_many_arguments)]
pub fn build_dflash_noise_forward(
    ctx: &mut Context,
    w: &DflashWeights,
    p: &DflashParams,
    kv: &KvCache,
    inp: &DecodeInputs,
    sinfo: SlotInfo,
    n_kv: u32,
    n_tokens: usize,
    n_seqs_unq: u32,
) -> ForwardResult {
    let a = &p.attn;
    let t = n_tokens as i64;

    assert!(!kv.has_swa(), "dflash noise: the iswa layout is not ported");

    let mut graph = Graph::new(1024);

    // hparams.f_attention_scale or 1/sqrt(n_embd_head) (:592)
    let kq_scale = if p.f_attention_scale != 0.0 {
        p.f_attention_scale
    } else {
        1.0 / (a.n_embd_head_v as f32).sqrt()
    };

    // token embeddings — own or the target model's (:679-687)
    let tok_embd = w.tok_embd.expect("dflash tok_embd (loader-resolved)");
    let mut inpL = ctx.get_rows(tok_embd, inp.tokens); // :697
    if p.f_embedding_scale != 0.0 {
        inpL = ctx.scale(inpL, p.f_embedding_scale); // :698-700
    }
    ctx.set_name(inpL, "inp_noise_embd"); // cb(..., :701)

    for (il, lw) in w.layers.iter().enumerate() {
        // noise_norm (:708)
        let mut noise_norm = build_norm_rms_opt(ctx, inpL, Some(lw.attn_norm), a.norm_eps);
        ctx.set_name(noise_norm, &format!("noise_norm-{il}")); // cb(..., :709)

        // DFlash2's dynamic conv on the attention input (:711-716)
        let mut attn_dynamic: Option<TensorId> = None;
        if let (Some(proj), Some(base)) = (lw.dflash_attn_conv_proj, lw.dflash_attn_conv_base) {
            let dynamic = crate::adapter::lora_mm(ctx, proj, noise_norm); // :713
            attn_dynamic = Some(dynamic);
            noise_norm = build_dflash2_conv(ctx, p, noise_norm, dynamic, base, 0, n_seqs_unq); // :714
            ctx.set_name(noise_norm, &format!("attn_conv_in-{il}")); // cb(..., :715)
        }

        let qcur = crate::adapter::lora_mm(ctx, lw.wq, noise_norm); // :718
        let kcur = crate::adapter::lora_mm(ctx, lw.wk, noise_norm); // :719
        let shared_kv = lw.wv.is_none(); // :720
        let vcur = match lw.wv {
            Some(wv) => crate::adapter::lora_mm(ctx, wv, noise_norm), // :721
            None => kcur,
        };

        let qcur = ctx.reshape_3d(qcur, a.n_embd_head_k, a.n_head, t); // :723
        let kcur = ctx.reshape_3d(kcur, a.n_embd_head_k, a.n_head_kv, t); // :724
        let vcur = ctx.reshape_3d(vcur, a.n_embd_head_v, a.n_head_kv, t); // :725

        let qcur = build_norm_rms_opt(ctx, qcur, Some(lw.attn_q_norm), a.norm_eps); // :727
        let kcur = build_norm_rms_opt(ctx, kcur, Some(lw.attn_k_norm), a.norm_eps); // :728
        let vcur = if shared_kv {
            ctx.rms_norm(vcur, a.norm_eps) // :729-731
        } else {
            vcur
        };

        let qcur = ctx.rope_ext(
            qcur,
            inp.pos,
            w.layers[0].rope_freqs,
            a.n_rot as i32,
            a.rope_mode,
            a.n_ctx_orig,
            a.freq_base,
            a.freq_scale,
            a.ext_factor,
            a.attn_factor,
            a.beta_fast,
            a.beta_slow,
        ); // :733 (build_rope)
        let kcur = ctx.rope_ext(
            kcur,
            inp.pos,
            w.layers[0].rope_freqs,
            a.n_rot as i32,
            a.rope_mode,
            a.n_ctx_orig,
            a.freq_base,
            a.freq_scale,
            a.ext_factor,
            a.attn_factor,
            a.beta_fast,
            a.beta_slow,
        ); // :734
        ctx.set_name(qcur, &format!("Qcur-{il}")); // cb(..., :735)
        ctx.set_name(kcur, &format!("Kcur-{il}")); // cb(..., :736)
        ctx.set_name(vcur, &format!("Vcur-{il}")); // cb(..., :737)

        // cache-aware, non-causal attention (:739-742) — build_attn over the
        // unified cache: scatter K/V, kq matmul, soft_max_ext with the batch
        // mask (the non-causal fill lives in the driver), v matmul, wo
        let k_rows = {
            let nb2 = ctx.nb(kcur)[2] as usize;
            ctx.view_2d(kcur, a.n_embd_head_k * a.n_head_kv, t, nb2, 0)
        };
        let v_rows = {
            let nb2 = ctx.nb(vcur)[2] as usize;
            ctx.view_2d(vcur, a.n_embd_head_v * a.n_head_kv, t, nb2, 0)
        };
        let _ = sinfo;
        let k_dst = ctx.set_rows(kv.layers[il].k, k_rows, inp.row_idx);
        let v_dst = ctx.set_rows(kv.layers[il].v, v_rows, inp.row_idx);
        graph.build_forward(ctx, k_dst);
        graph.build_forward(ctx, v_dst);

        let k_view = kv.get_k(ctx, il, a.n_embd_head_k, a.n_head_kv, n_kv);
        let v_view = kv.get_v(ctx, il, a.n_embd_head_v, a.n_head_kv, n_kv);

        let cur = if a.use_flash_attn {
            // build_attn_mha's FA branch (llama-graph.cpp:2626-2669)
            graph::flash_attn_core_sinks(
                ctx,
                qcur,
                k_view,
                v_view,
                inp.kq_mask,
                lw.attn_sinks,
                kq_scale,
                0.0,
                0.0,
            )
        } else {
            // build_attn_mha's non-FA branch (llama-graph.cpp:2670-2733)
            let q = ctx.permute(qcur, 0, 2, 1, 3);
            let k_view = ctx.permute(k_view, 0, 2, 1, 3);
            let mut kq = ctx.mul_mat(k_view, q);
            ctx.set_name(kq, &format!("kq-{il}")); // cb(kq, "kq", il) :2672
            kq = ctx.soft_max_ext(kq, Some(inp.kq_mask), kq_scale, 0.0);
            ctx.set_name(kq, &format!("kq_soft_max-{il}")); // :2707
            let v_view = ctx.permute(v_view, 0, 2, 1, 3);
            let v_t = ctx.transpose(v_view);
            let v_c = ctx.cont(v_t);
            let kqv = ctx.mul_mat(v_c, kq);
            ctx.set_name(kqv, &format!("kqv-{il}")); // :2716
            let kqv = ctx.permute(kqv, 0, 2, 1, 3);
            let kqv = ctx.cont(kqv);
            ctx.reshape_2d(kqv, a.n_embd_head_v * a.n_head, t) // :2733
        };
        ctx.set_name(cur, &format!("kqv_out-{il}")); // cb(cur, "kqv_out", il) :2801

        // wo projection (:740-742)
        let mut cur = crate::adapter::lora_mm(ctx, lw.wo, cur);

        // the post-attention value scale (dflash.cpp:742-745, def4d406a)
        if p.f_attn_value_scale != 0.0 {
            cur = ctx.scale(cur, p.f_attn_value_scale);
            ctx.set_name(cur, &format!("attn_out_scaled-{il}")); // cb(..., :744)
        }

        // DFlash2's dynamic conv on the attention output (:744-747)
        if let (Some(dynamic), Some(base)) = (attn_dynamic, lw.dflash_attn_conv_base) {
            cur = build_dflash2_conv(ctx, p, cur, dynamic, base, 1, n_seqs_unq);
            ctx.set_name(cur, &format!("attn_conv_out-{il}")); // cb(..., :746)
        }

        if let Some(post) = lw.attn_post_norm {
            cur = build_norm_rms_opt(ctx, cur, Some(post), a.norm_eps); // :749-752
            ctx.set_name(cur, &format!("attn_post_norm-{il}"));
        }

        let ffn_inp = ctx.add(cur, inpL); // :754
        ctx.set_name(ffn_inp, &format!("ffn_inp-{il}")); // cb(..., :755)

        let cur = build_norm_rms_opt(ctx, ffn_inp, Some(lw.ffn_norm), a.norm_eps); // :757
        ctx.set_name(cur, &format!("ffn_norm-{il}")); // cb(..., :758)

        // DFlash2's dynamic conv on the FFN input (:760-765) — the dynamic
        // coefficients are computed once on the pre-FFN input and reused for
        // the output conv below (:775-778)
        let mut ffn_dynamic: Option<TensorId> = None;
        let cur =
            if let (Some(proj), Some(base)) = (lw.dflash_ffn_conv_proj, lw.dflash_ffn_conv_base) {
                let dynamic = crate::adapter::lora_mm(ctx, proj, cur); // :762
                ffn_dynamic = Some(dynamic);
                let c = build_dflash2_conv(ctx, p, cur, dynamic, base, 0, n_seqs_unq); // :763
                ctx.set_name(c, &format!("ffn_conv_in-{il}")); // cb(..., :764)
                c
            } else {
                cur
            };

        let mut cur = build_ffn_dflash(ctx, cur, lw.ffn_gate, lw.ffn_up, lw.ffn_down, p.ffn_gelu);
        ctx.set_name(cur, &format!("ffn_out-{il}")); // cb(..., :773)

        // DFlash2's dynamic conv on the FFN output (:775-778)
        if let (Some(dynamic), Some(base)) = (ffn_dynamic, lw.dflash_ffn_conv_base) {
            cur = build_dflash2_conv(ctx, p, cur, dynamic, base, 1, n_seqs_unq);
            ctx.set_name(cur, &format!("ffn_conv_out-{il}")); // cb(..., :777)
        }

        if let Some(post) = lw.ffn_post_norm {
            cur = build_norm_rms_opt(ctx, cur, Some(post), a.norm_eps); // :781-783
            ctx.set_name(cur, &format!("ffn_post_norm-{il}"));
        }

        let mut cur = ctx.add(cur, ffn_inp); // :785
        if let Some(s) = lw.out_scale {
            cur = ctx.mul(cur, s); // :786-788
        }
        ctx.set_name(cur, &format!("l_out-{il}")); // cb(..., :789)

        inpL = cur;
    }

    // result_norm (:794)
    let cur = build_norm_rms_opt(ctx, inpL, Some(w.output_norm), a.norm_eps);
    ctx.set_name(cur, "result_norm"); // cb(..., :795)
    let t_embd = cur; // res->t_embd = cur (:797)

    // lm_head — own or the target model's (:799-810)
    let output = w.output.expect("dflash output (loader-resolved)");
    let mut cur = crate::adapter::lora_mm(ctx, output, cur);

    // DFlash2 feeds these logits to the selector, so they need the target's
    // output transforms; DFlash1 and DSpark read them through the sampler
    // instead (:812-823)
    if w.dflash_selector_hidden.is_some() {
        if p.f_logit_scale != 0.0 {
            cur = ctx.scale(cur, p.f_logit_scale); // :815-817
        }
        if p.f_final_logit_softcapping > 0.0 {
            cur = ctx.scale(cur, 1.0 / p.f_final_logit_softcapping); // :818-820
            cur = ctx.tanh(cur);
            cur = ctx.scale(cur, p.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target
    // vocabulary via d2t (:825-839)
    if let Some(d2t) = w.d2t {
        let n_draft_vocab = ctx.ne(cur)[0];
        let n_outputs = ctx.ne(cur)[1];

        assert_eq!(
            ctx.ty(d2t),
            ggml::types::GgmlType::I64,
            "dflash: model.d2t->type == GGML_TYPE_I64"
        );
        assert_eq!(ctx.ne(d2t)[0], n_draft_vocab);

        let logits = ctx.new_tensor_3d(ggml::types::GgmlType::F32, 1, p.n_vocab, n_outputs);
        let logits = ctx.fill(logits, f32::NEG_INFINITY); // :834
        let rows = ctx.reshape_3d(cur, 1, n_draft_vocab, n_outputs);
        let idx = ctx.reshape_3d(d2t, n_draft_vocab, 1, 1);
        cur = ctx.set_rows(logits, rows, idx);
        cur = ctx.reshape_2d(cur, p.n_vocab, n_outputs); // :838
    }
    ctx.set_name(cur, "result_output"); // cb(..., :840)

    graph.build_forward(ctx, cur); // :843

    // DSpark: bias the draft logits with the Markov head (:845-848)
    let (logits, mut t_h_nextn) = if w.dspark_markov_w1.is_some() {
        build_dspark_markov_head(ctx, w, p, &mut graph, inp.tokens, cur, t_embd, n_seqs_unq)
    } else {
        (cur, None)
    };

    // DFlash2: the selector lattice (:850-852) — consumes the (possibly
    // markov-biased) logits and replaces t_h_nextn with the packed rows
    if w.dflash_selector_hidden.is_some() {
        let packed = build_dflash2_selector(
            ctx, w, p, &mut graph, inp.tokens, logits, t_embd, t, n_seqs_unq,
        );
        t_h_nextn = Some(packed);
    }

    ForwardResult {
        logits,
        graph,
        embd: t_h_nextn,
    }
}

/// the 1-layer "trunk" bundle a DecodeContext needs for its generic sizing
/// (n_layer / output / n_pos_per_embd) — the dflash branch of `forward`
/// returns before this is ever matched; the tensor ids are the draft's own.
/// The DSV4 backbone fabricates the stub from its stage tensors (also never
/// dispatched — only the sizing reads it).
pub fn dflash_trunk_stub(w: &DflashWeights) -> graph::ModelWeights {
    if let Some(staged) = &w.dsv4 {
        let l = &staged.layers[0];
        return graph::ModelWeights {
            tok_embd: w.tok_embd.expect("dflash tok_embd"),
            output_norm: w.output_norm,
            output: w.output.expect("dflash output"),
            layers: vec![graph::LayerWeights {
                attn_norm: l.attn_norm,
                wq: l.wq_a,
                wk: l.wkv,
                wv: l.wkv,
                wo: l.wo_b,
                wq_b: None,
                wk_b: None,
                wv_b: None,
                ffn_norm: l.ffn_norm,
                ffn_gate: l.ffn_gate_exps,
                ffn_down: l.ffn_down_shexp,
                ffn_up: l.ffn_up_exps,
            }],
        };
    }
    let l = &w.layers[0];
    graph::ModelWeights {
        tok_embd: w.tok_embd.expect("dflash tok_embd"),
        output_norm: w.output_norm,
        output: w.output.expect("dflash output"),
        layers: vec![graph::LayerWeights {
            attn_norm: l.attn_norm,
            wq: l.wq,
            wk: l.wk,
            // the stub never dispatches; the shared-KV None resolves to the
            // k tensor (a valid id of the draft's context)
            wv: l.wv.unwrap_or(l.wk),
            wo: l.wo,
            wq_b: None,
            wk_b: None,
            wv_b: None,
            ffn_norm: l.ffn_norm,
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
        }],
    }
}
