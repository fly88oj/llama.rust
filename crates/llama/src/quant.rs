//! quant.rs — 1:1 port of llama.cpp `src/llama-quant.cpp` (bd4f514db1):
//! the tensor-type selection strategy used by `llama-quantize`.
//!
//! Every item carries the C line number(s) it was translated from. The
//! quantization *pipeline* (file I/O, row loops) lives in
//! `crates/tools/quantize`; this module holds the pure decision logic:
//! `llama_tensor_get_type` and everything it calls.
//!
//! Not ported here (see the tool for the rest):
//! * `llama_tensor_dequantize_impl` / `llama_tensor_quantize_impl` — row loops
//!   over ggml (tool-side, uses `ggml::quants`),
//! * imatrix loading (`--imatrix` is rejected by the tool),
//! * `llama_quant_*` C ABI helpers (llama-quant.cpp:1401-1486).

use ggml::types::GgmlType;

use crate::arch::{LlmArch, LlmTensor};
use crate::hparams::LlamaHparams;

// ---------------------------------------------------------------------------
// tensor categorization - llama-quant.cpp:24-39
// ---------------------------------------------------------------------------

/// `enum class tensor_category` (llama-quant.cpp:26-39).
///
/// "this is different from LLM_TN - we want broad categories, not specific
/// tensor names per arch."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    TokenEmbd,
    AttentionQ,
    AttentionV,
    AttentionK,
    AttentionQkv,
    AttentionKvB,
    AttentionOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Output,
    Other,
}

/// `tensor_name_match_token_embd` (llama-quant.cpp:104-107).
pub fn tensor_name_match_token_embd(tensor_name: &str) -> bool {
    tensor_name == "token_embd.weight" || tensor_name == "per_layer_token_embd.weight"
}

/// `tensor_name_match_output_weight` (llama-quant.cpp:109-111).
pub fn tensor_name_match_output_weight(tensor_name: &str) -> bool {
    tensor_name == "output.weight"
}

/// `tensor_get_category` (llama-quant.cpp:119-154).
///
/// Order of the checks is load-bearing (e.g. `attn_qkv` before `attn_v`,
/// `output` / `token_embd` before the `find`-based rules).
pub fn tensor_get_category(tensor_name: &str) -> TensorCategory {
    if tensor_name_match_output_weight(tensor_name) {
        return TensorCategory::Output;
    }
    if tensor_name_match_token_embd(tensor_name) {
        return TensorCategory::TokenEmbd;
    }
    if tensor_name.contains("attn_qkv.weight") {
        return TensorCategory::AttentionQkv;
    }
    if tensor_name.contains("attn_kv_b.weight") {
        return TensorCategory::AttentionKvB;
    }
    if tensor_name.contains("attn_v.weight") {
        return TensorCategory::AttentionV;
    }
    if tensor_name.contains("attn_k.weight") {
        return TensorCategory::AttentionK;
    }
    if tensor_name.contains("attn_q.weight") {
        return TensorCategory::AttentionQ;
    }
    if tensor_name.contains("attn_output.weight") {
        return TensorCategory::AttentionOutput;
    }
    if tensor_name.contains("ffn_up") {
        return TensorCategory::FfnUp;
    }
    if tensor_name.contains("ffn_gate") {
        return TensorCategory::FfnGate;
    }
    if tensor_name.contains("ffn_down") {
        return TensorCategory::FfnDown;
    }
    TensorCategory::Other
}

/// `category_is_attn_v` (llama-quant.cpp:157-161).
pub fn category_is_attn_v(cat: TensorCategory) -> bool {
    cat == TensorCategory::AttentionV
        || cat == TensorCategory::AttentionQkv
        || cat == TensorCategory::AttentionKvB
}

// ---------------------------------------------------------------------------
// ftype - include/llama.h:117-162, llama-quant.cpp:847-890
// ---------------------------------------------------------------------------

/// `enum llama_ftype` (include/llama.h:117-162). Discriminants are the values
/// written to `general.file_type`; do not reorder.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Ftype {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2_K = 10,
    MostlyQ3_K_S = 11,
    MostlyQ3_K_M = 12,
    MostlyQ3_K_L = 13,
    MostlyQ4_K_S = 14,
    MostlyQ4_K_M = 15,
    MostlyQ5_K_S = 16,
    MostlyQ5_K_M = 17,
    MostlyQ6_K = 18,
    MostlyIQ2_XXS = 19,
    MostlyIQ2_XS = 20,
    MostlyQ2_K_S = 21,
    MostlyIQ3_XS = 22,
    MostlyIQ3_XXS = 23,
    MostlyIQ1_S = 24,
    MostlyIQ4_NL = 25,
    MostlyIQ3_S = 26,
    MostlyIQ3_M = 27,
    MostlyIQ2_S = 28,
    MostlyIQ2_M = 29,
    MostlyIQ4_XS = 30,
    MostlyIQ1_M = 31,
    MostlyBF16 = 32,
    MostlyTQ1_0 = 36,
    MostlyTQ2_0 = 37,
    MostlyMXFP4_MOE = 38,
    MostlyNVFP4 = 39,
    MostlyQ1_0 = 40,
    MostlyQ2_0 = 41,
    /// `LLAMA_FTYPE_GUESSED` (llama.h:161)
    Guessed = 1024,
}

impl Default for Ftype {
    fn default() -> Self {
        Ftype::MostlyQ8_0 // llama-quant.cpp:1368
    }
}

impl Ftype {
    /// `llama_ftype_get_default_type` (llama-quant.cpp:847-890).
    /// `None` == C's `GGML_TYPE_COUNT` (invalid / not handled).
    pub fn default_type(self) -> Option<GgmlType> {
        use Ftype::*;
        Some(match self {
            MostlyQ4_0 => GgmlType::Q4_0,
            MostlyQ4_1 => GgmlType::Q4_1,
            MostlyQ5_0 => GgmlType::Q5_0,
            MostlyQ5_1 => GgmlType::Q5_1,
            MostlyQ8_0 => GgmlType::Q8_0,
            MostlyF16 => GgmlType::F16,
            MostlyBF16 => GgmlType::Bf16,
            AllF32 => GgmlType::F32,
            MostlyQ1_0 => GgmlType::Q1_0,
            MostlyQ2_0 => GgmlType::Q2_0,

            MostlyMXFP4_MOE => GgmlType::Mxfp4,
            MostlyNVFP4 => GgmlType::Nvfp4,

            // K-quants
            MostlyQ2_K_S | MostlyQ2_K => GgmlType::Q2K,
            MostlyIQ3_XS => GgmlType::Iq3S,
            MostlyQ3_K_S | MostlyQ3_K_M | MostlyQ3_K_L => GgmlType::Q3K,
            MostlyQ4_K_S | MostlyQ4_K_M => GgmlType::Q4K,
            MostlyQ5_K_S | MostlyQ5_K_M => GgmlType::Q5K,
            MostlyQ6_K => GgmlType::Q6K,
            MostlyTQ1_0 => GgmlType::Tq1_0,
            MostlyTQ2_0 => GgmlType::Tq2_0,
            MostlyIQ2_XXS => GgmlType::Iq2Xxs,
            MostlyIQ2_XS => GgmlType::Iq2Xs,
            MostlyIQ2_S => GgmlType::Iq2Xs,
            MostlyIQ2_M => GgmlType::Iq2S,
            MostlyIQ3_XXS => GgmlType::Iq3Xxs,
            MostlyIQ1_S => GgmlType::Iq1S,
            MostlyIQ1_M => GgmlType::Iq1M,
            MostlyIQ4_NL => GgmlType::Iq4Nl,
            MostlyIQ4_XS => GgmlType::Iq4Xs,
            MostlyIQ3_S | MostlyIQ3_M => GgmlType::Iq3S,

            // llama-quant.cpp:888 `default: return GGML_TYPE_COUNT;`
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// "max amount of tensor data kept in memory" - llama-quant.cpp:42
// ---------------------------------------------------------------------------

/// `LLAMA_QUANT_MAX_BUF_SIZE` (llama-quant.cpp:42).
pub const LLAMA_QUANT_MAX_BUF_SIZE: usize = 8 * 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// params - llama-quant.cpp:1365-1385 (llama_model_quantize_default_params)
// ---------------------------------------------------------------------------

/// One `--tensor-type` entry. The C side stores a compiled `std::regex` and
/// uses `std::regex_search` (llama-quant.cpp:188, 196, 693, 716); patterns are
/// searched, not anchored.
pub trait NameMatcher: Send + Sync {
    fn is_match(&self, tensor_name: &str) -> bool;
}

/// result of parsing --tensor-type (llama-quant.cpp:19-22 / quantize.cpp:23-26)
pub struct TensorTypeOverride {
    /// `std::regex pattern` (llama-quant.cpp:188)
    pub pattern: Box<dyn NameMatcher>,
    pub ty: GgmlType,
}

/// `llama_model_quantize_params` — only the fields the pinned reference uses
/// (imatrix / kv_overrides / prune_layers / keep_split are not supported by
/// this port, see the tool).
pub struct QuantizeParams {
    /// `nthread` (0 → hardware concurrency)
    pub nthread: usize,
    pub ftype: Ftype,
    /// `output_tensor_type` — `None` == `GGML_TYPE_COUNT`
    pub output_tensor_type: Option<GgmlType>,
    /// `token_embedding_type` — `None` == `GGML_TYPE_COUNT`
    pub token_embedding_type: Option<GgmlType>,
    pub allow_requantize: bool,
    pub quantize_output_tensor: bool,
    pub only_copy: bool,
    pub pure: bool,
    pub dry_run: bool,
    /// `max_buf_size` (llama-quant.cpp:1381 default = LLAMA_QUANT_MAX_BUF_SIZE)
    pub max_buf_size: usize,
    /// `tt_overrides` (llama-quant.cpp:1379)
    pub tt_overrides: Vec<TensorTypeOverride>,
}

impl QuantizeParams {
    /// `llama_model_quantize_default_params` (llama-quant.cpp:1365-1385).
    pub fn new() -> Self {
        QuantizeParams {
            nthread: 0,
            ftype: Ftype::default(),
            output_tensor_type: None,
            token_embedding_type: None,
            allow_requantize: false,
            quantize_output_tensor: true,
            only_copy: false,
            pure: false,
            dry_run: false,
            max_buf_size: LLAMA_QUANT_MAX_BUF_SIZE,
            tt_overrides: Vec::new(),
        }
    }
}

impl Default for QuantizeParams {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// model facts needed by the strategy
// ---------------------------------------------------------------------------

/// The subset of `llama_model` / `llama_hparams` that `llama_tensor_get_type`
/// reads. In the reference these come from a fully loaded model
/// (`llama_model_create` + `load_hparams`, llama-quant.cpp:941-950); here they
/// are read straight from the input GGUF metadata.
#[derive(Debug, Clone, Copy)]
pub struct QuantModelInfo {
    pub arch: LlmArch,
    /// `hparams.n_expert`
    pub n_expert: i32,
    /// `hparams.n_layer_all`
    pub n_layer_all: i32,
    /// `hparams.n_layer()` == n_layer_all - n_layer_nextn (llama-hparams.cpp:347)
    pub n_layer: i32,
    /// `hparams.n_head(0)`
    pub n_head: i32,
    /// `hparams.n_head_kv(0)`
    pub n_head_kv: i32,
    /// `hparams.n_embd`
    pub n_embd: i32,
    /// `hparams.n_vocab` (only used by the LLAMA arch type detection)
    pub n_vocab: i32,
    /// `model.type == LLM_TYPE_70B` (llama-quant.cpp:559). Only this one
    /// llm_type influences quantization. It is assigned per arch in the
    /// model loaders: models/llama.cpp:28 (80 layers, n_head != n_head_kv),
    /// models/qwen2.cpp:14, models/olmo.cpp:10, models/deci.cpp:8,
    /// models/jais2.cpp:8 (68 layers) — all `case 80`/`68` → LLM_TYPE_70B.
    pub is_type_70b: bool,
}

impl QuantModelInfo {
    /// Build from loaded hparams (`hparams.n_vocab` is not stored in the Rust
    /// hparams, so it is passed separately).
    pub fn from_hparams(arch: LlmArch, h: &LlamaHparams, n_vocab: i32) -> Self {
        let n_layer = h.n_layer_all as i32 - h.n_layer_nextn as i32;
        let n_head = h.n_head(0) as i32;
        let n_head_kv = h.n_head_kv(0) as i32;
        let n_expert = h.n_expert as i32;
        let is_type_70b = match arch {
            // src/models/llama.cpp:19-31 (n_expert == 8 branch has priority)
            LlmArch::LLAMA => n_expert != 8 && n_layer == 80 && n_head != n_head_kv,
            // src/models/qwen2.cpp:8-16
            LlmArch::QWEN2 => n_layer == 80,
            // src/models/olmo.cpp:8-12
            LlmArch::OLMO => n_layer == 80,
            // src/models/deci.cpp:6-11
            LlmArch::DECI => n_layer == 80,
            // src/models/jais2.cpp:6-11
            LlmArch::JAIS2 => n_layer == 68,
            _ => false,
        };
        QuantModelInfo {
            arch,
            n_expert,
            n_layer_all: h.n_layer_all as i32,
            n_layer,
            n_head,
            n_head_kv,
            n_embd: h.n_embd as i32,
            n_vocab,
            is_type_70b,
        }
    }

    /// `llama_hparams::n_gqa(0)` (llama-hparams.cpp:99-105). Note the C call
    /// in the quantizer uses the default `il = 0`.
    pub fn n_gqa(&self) -> i32 {
        if self.n_head_kv == 0 {
            return 0;
        }
        self.n_head / self.n_head_kv
    }
}

// ---------------------------------------------------------------------------
// quantization state - llama-quant.cpp:167-200
// ---------------------------------------------------------------------------

/// `struct quantize_state_impl` (llama-quant.cpp:167-200).
pub struct QuantizeState<'a> {
    pub model: &'a QuantModelInfo,
    /// C keeps the compiled patterns inside the state (llama-quant.cpp:188);
    /// the params borrow is split out here.
    pub tensor_type_patterns: &'a [TensorTypeOverride],

    pub n_attention_wv: i32,
    pub n_ffn_down: i32,
    pub n_ffn_gate: i32,
    pub n_ffn_up: i32,
    pub i_attention_wv: i32,
    pub i_ffn_down: i32,
    pub i_ffn_gate: i32,
    pub i_ffn_up: i32,

    pub n_fallback: i32,

    pub has_imatrix: bool,

    /// "assume tied until we see output.weight" (llama-quant.cpp:185)
    pub has_tied_embeddings: bool,
}

impl<'a> QuantizeState<'a> {
    /// `quantize_state_impl(model, params)` (llama-quant.cpp:190-199).
    pub fn new(model: &'a QuantModelInfo, patterns: &'a [TensorTypeOverride]) -> Self {
        QuantizeState {
            model,
            tensor_type_patterns: patterns,
            n_attention_wv: 0,
            n_ffn_down: 0,
            n_ffn_gate: 0,
            n_ffn_up: 0,
            i_attention_wv: 0,
            i_ffn_down: 0,
            i_ffn_gate: 0,
            i_ffn_up: 0,
            n_fallback: 0,
            has_imatrix: false,
            has_tied_embeddings: true,
        }
    }

    /// The reset block of `llama_quant_compute_types` (llama-quant.cpp:1455-1465).
    pub fn reset_counters(&mut self) {
        self.n_attention_wv = 0;
        self.n_ffn_down = 0;
        self.n_ffn_gate = 0;
        self.n_ffn_up = 0;
        self.i_attention_wv = 0;
        self.i_ffn_down = 0;
        self.i_ffn_gate = 0;
        self.i_ffn_up = 0;
        self.n_fallback = 0;
        self.has_imatrix = false;
        self.has_tied_embeddings = true;
    }
}

// ---------------------------------------------------------------------------
// per-tensor metadata - llama-quant.cpp:202-210
// ---------------------------------------------------------------------------

/// `struct tensor_metadata` (llama-quant.cpp:202-210).
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub target_type: GgmlType,
    pub category: TensorCategory,
    pub remapped_imatrix_name: String,
    pub allows_quantization: bool,
    pub requires_imatrix: bool,
}

impl TensorMetadata {
    pub fn new(name: String) -> Self {
        TensorMetadata {
            name,
            target_type: GgmlType::F32, // set in the preliminary loop
            category: TensorCategory::Other,
            remapped_imatrix_name: String::new(),
            allows_quantization: false,
            requires_imatrix: false,
        }
    }
}

// ---------------------------------------------------------------------------
// dequantizability check - llama-quant.cpp:220-228
// ---------------------------------------------------------------------------

/// The type gate of `llama_tensor_dequantize_impl` (llama-quant.cpp:220-228);
/// the row loops themselves live in the tool.
pub fn check_dequantizable(ty: GgmlType) -> Result<(), String> {
    if ty.is_quantized() {
        if !has_dequantize(ty) {
            return Err(format!(
                "type {} unsupported for integer quantization: no dequantization available",
                ty.name()
            ));
        }
    } else if !matches!(ty, GgmlType::F16 | GgmlType::Bf16) {
        // llama-quant.cpp:225-228 — F32 never reaches this in the main loop
        // because it uses the source buffer directly (llama-quant.cpp:1302).
        return Err(format!(
            "cannot dequantize/convert tensor type {}",
            ty.name()
        ));
    }
    Ok(())
}

/// Does this port have a dequantizer for `ty`? (C: `qtype->to_float != NULL`,
/// llama-quant.cpp:222). Mirrors `ggml::quants::dequantize_row` coverage.
pub fn has_dequantize(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(
        ty,
        F32 | F16
            | Bf16
            | Q1_0
            | Q2_0
            | Q4_0
            | Q4_1
            | Q5_0
            | Q5_1
            | Q8_0
            | Q2K
            | Q3K
            | Q4K
            | Q5K
            | Q6K
            | Mxfp4
            | Nvfp4
            | Iq2Xxs
            | Iq2Xs
            | Iq2S
            | Iq3Xxs
            | Iq3S
            | Iq1S
            | Iq1M
            | Iq4Nl
            | Iq4Xs
    )
}

// ---------------------------------------------------------------------------
// tensor_allows_quantization - llama-quant.cpp:287-365
// ---------------------------------------------------------------------------

/// `ggml_n_dims` (ggml.c): the number of leading dims that are > 1.
pub fn ggml_n_dims(ne: &[i64; 4]) -> usize {
    if ne[3] > 1 {
        4
    } else if ne[2] > 1 {
        3
    } else if ne[1] > 1 {
        2
    } else {
        1
    }
}

/// `tensor_allows_quantization` (llama-quant.cpp:287-365).
///
/// `arch` drives the glm5-next exclusion block (:331-348, def4d406a); the
/// two `LLM_TN(arch)(...)` comparisons still resolve through the global
/// `LLM_TENSOR_NAMES` table and ignore it (llama-arch.cpp:1016-1028).
pub fn tensor_allows_quantization(
    params: &QuantizeParams,
    arch: LlmArch,
    name: &str,
    ne: &[i64; 4],
) -> bool {
    // trivial checks first -- no string ops needed
    if params.only_copy {
        return false;
    }

    // quantize only 2D and 3D tensors (experts)
    if ggml_n_dims(ne) < 2 {
        return false;
    }

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    let mut quantize = name.ends_with("weight"); // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= !name.contains("_norm.weight");

    quantize &= params.quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= !name.contains("ffn_gate_inp.weight");

    // do not quantize the i32 token-id -> expert-id routing table (DeepSeek-V4)
    quantize &= !name.contains("ffn_gate_tid2eid.weight");

    // these are very small (e.g. 4x4)
    quantize &= !name.contains("altup");
    quantize &= !name.contains("laurel");

    // these are not too big so keep them as it is
    quantize &= !name.contains("per_layer_model_proj");

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != crate::arch::tensor_name_suffix(LlmTensor::POS_EMBD, "weight", -1, -1);
    quantize &= name != crate::arch::tensor_name_suffix(LlmTensor::TOKEN_TYPES, "weight", -1, -1);

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= !name.contains("ssm_conv1d");
    quantize &= !name.contains("shortconv.conv.weight");

    // do not quantize MiniMax's indexer projection weights, they are tiny
    quantize &= !name.contains("indexer.k_proj.weight");
    quantize &= !name.contains("indexer.q_proj.weight");

    // glm5-next (llama-quant.cpp:331-348, 649dcb103)
    if arch == LlmArch::GLM5_NEXT {
        quantize &= !name.contains("hc_");
        quantize &= !name.contains("indexer.attn_q_b");
        quantize &= !name.contains("indexer.attn_k");
        quantize &= !name.contains("indexer.proj");
        quantize &= !name.contains("indexer_compressor_gate");
        quantize &= !name.contains("indexer_compressor_ape");
        quantize &= !name.contains("ssm_f_a.weight");
        quantize &= !name.contains("ssm_f_b.weight");
        quantize &= !name.contains("ssm_g_a.weight");
        quantize &= !name.contains("ssm_g_b.weight");
        quantize &= !name.contains("ssm_beta.weight");
        quantize &= !name.contains("attn_kv_a_mqa.weight");
        quantize &= !name.contains("attn_k_b.weight");
        quantize &= !name.contains("attn_v_b.weight");
    }

    // do not quantize RWKV's small yet 2D weights
    quantize &= !name.contains("time_mix_first.weight");
    quantize &= !name.contains("time_mix_w0.weight");
    quantize &= !name.contains("time_mix_w1.weight");
    quantize &= !name.contains("time_mix_w2.weight");
    quantize &= !name.contains("time_mix_v0.weight");
    quantize &= !name.contains("time_mix_v1.weight");
    quantize &= !name.contains("time_mix_v2.weight");
    quantize &= !name.contains("time_mix_a0.weight");
    quantize &= !name.contains("time_mix_a1.weight");
    quantize &= !name.contains("time_mix_a2.weight");
    quantize &= !name.contains("time_mix_g1.weight");
    quantize &= !name.contains("time_mix_g2.weight");
    quantize &= !name.contains("time_mix_decay_w1.weight");
    quantize &= !name.contains("time_mix_decay_w2.weight");
    quantize &= !name.contains("time_mix_lerp_fused.weight");

    // do not quantize relative position bias (T5)
    quantize &= !name.contains("attn_rel_b.weight");

    // do not quantize specific multimodal tensors
    quantize &= !name.contains(".position_embd");
    quantize &= !name.contains("sam.pos_embd");
    quantize &= !name.contains("sam.neck.");
    quantize &= !name.contains("sam.net_");
    quantize &= !name.contains(".rel_pos");
    quantize &= !name.contains(".patch_embd");
    quantize &= !name.contains(".patch_merger");

    // audio codebook
    quantize &= !name.contains("a.rvq.codebook");
    quantize &= !name.contains("mm.a.code_embd");

    quantize
}

// ---------------------------------------------------------------------------
// tensor_type_fallback - llama-quant.cpp:372-425
// ---------------------------------------------------------------------------

/// `tensor_type_fallback` (llama-quant.cpp:372-425).
///
/// Returns the (possibly demoted) type; increments `qs.n_fallback` and logs
/// exactly where C does. `Err` mirrors the C `throw` (llama-quant.cpp:408).
pub fn tensor_type_fallback(
    qs: &mut QuantizeState,
    name: &str,
    ncols: i64,
    target_type: GgmlType,
) -> Result<GgmlType, String> {
    let mut return_type = target_type;

    let qk_k = target_type.blck_size() as i64;

    if ncols % qk_k != 0 {
        // this tensor's shape is incompatible with this quant
        log_warn(&format!(
            "warning: {:<36} - ncols {:6} not divisible by {:3} (required for type {:7}) ",
            name,
            ncols,
            qk_k,
            target_type.name()
        ));
        qs.n_fallback += 1;

        match target_type {
            // types on the left: block size 256
            GgmlType::Iq1S
            | GgmlType::Iq1M
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S // types on the right: block size 32
            | GgmlType::Iq4Xs => return_type = GgmlType::Iq4Nl,
            GgmlType::Q2_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Tq1_0
            | GgmlType::Tq2_0 => return_type = GgmlType::Q4_0,
            GgmlType::Q4K => return_type = GgmlType::Q5_0,
            GgmlType::Q5K => return_type = GgmlType::Q5_1,
            GgmlType::Q6K => return_type = GgmlType::Q8_0,
            _ => {
                if qk_k <= 32 {
                    // the target is already a 32-block type, so there is no
                    // smaller block to demote to
                    // the check below turns it into F16, as a 256-block type
                    // does when its fallback does not fit
                    return_type = target_type;
                } else {
                    return Err(format!(
                        "no tensor type fallback is defined for type {}",
                        target_type.name()
                    ));
                }
            }
        }
        if ncols % return_type.blck_size() as i64 != 0 {
            // the fallback return type is still not compatible for this tensor!
            log_warn("(WARNING: must use F16 due to unusual shape) ");
            return_type = GgmlType::F16;
        }
        log_warn(&format!("-> falling back to {:7}\n", return_type.name()));
    }
    Ok(return_type)
}

// ---------------------------------------------------------------------------
// llama_tensor_get_type_impl - llama-quant.cpp:428-680
// ---------------------------------------------------------------------------

/// `use_more_bits` lambda (llama-quant.cpp:434-436). C integer division and
/// `%` (truncating) are Rust's defaults for i32.
pub fn use_more_bits(i_layer: i32, n_layers: i32) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// `sscanf(name, "blk.%d.", &i_layer)` — the layer number, or `None` when no
/// conversion takes place (C: `!= 1` → throw, llama-quant.cpp:444-446).
pub fn parse_blk_layer(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("blk.")?;
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    let mut negative = false;
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        negative = bytes[i] == b'-';
        i += 1;
    }
    let start = i;
    let mut value: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = value * 10 + (bytes[i] - b'0') as i64;
        i += 1;
    }
    if i == start {
        return None; // sscanf assigned nothing
    }
    Some((if negative { -value } else { value }) as i32)
}

/// `layer_info` lambda (llama-quant.cpp:438-452): for MoE models the layer is
/// parsed from the tensor name, otherwise the running counter is used.
pub fn layer_info(
    i_layer: i32,
    n_layer: i32,
    name: &str,
    n_expert: i32,
) -> Result<(i32, i32), String> {
    let mut i_layer = i_layer;
    if n_expert > 1 {
        // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not
        // consecutive, but occasionally randomly sprinkled in the model.
        // Hence, simply dividing i_ffn_down by n_expert does not work ...
        match parse_blk_layer(name) {
            Some(v) => i_layer = v,
            None => return Err(format!("Failed to determine layer for tensor {name}")),
        }
        if i_layer < 0 || i_layer >= n_layer {
            return Err(format!(
                "Bad layer {i_layer} for tensor {name}. Must be in [0, {n_layer})"
            ));
        }
    }
    Ok((i_layer, n_layer))
}

/// `llama_tensor_get_type_impl` (llama-quant.cpp:428-680).
///
/// The `output_tensor_type` / `token_embedding_type` overrides that C reads
/// through `qs.params` (llama-quant.cpp:457, 492) are applied by the outer
/// `llama_tensor_get_type` before this function is reached
/// (llama-quant.cpp:687-706), so they are not repeated here.
#[allow(clippy::too_many_lines)]
pub fn llama_tensor_get_type_impl(
    qs: &mut QuantizeState,
    mut new_type: GgmlType,
    name: &str,
    ne: &[i64; 4],
    ftype: Ftype,
    category: TensorCategory,
) -> Result<GgmlType, String> {
    // TODO: avoid hardcoded tensor names - use the TN_* constants
    let arch = qs.model.arch;
    let n_expert = std::cmp::max(1, qs.model.n_expert);

    // by default, for glm5-next, don't let these tensors be quantized below
    // Q8_0 (llama-quant.cpp:472-486, 649dcb103)
    if arch == LlmArch::GLM5_NEXT
        && (name.contains("attn_q_a")
            || name.contains("attn_q_b")
            || name.contains("nextn.eh_proj"))
    {
        match new_type {
            GgmlType::F32 | GgmlType::Bf16 | GgmlType::F16 => {}
            _ => return Ok(GgmlType::Q8_0),
        }
    }

    // for arches that share the same tensor between the token embeddings and
    // the output, we quantize the token embeddings with the quantization of
    // the output tensor
    if category == TensorCategory::Output
        || (qs.has_tied_embeddings && category == TensorCategory::TokenEmbd)
    {
        if ftype == Ftype::MostlyMXFP4_MOE {
            new_type = GgmlType::Q8_0;
        } else if arch == LlmArch::FALCON || ne[0] % new_type.blck_size() as i64 != 0 {
            new_type = GgmlType::Q8_0;
        } else if matches!(
            ftype,
            Ftype::MostlyIQ2_XXS
                | Ftype::MostlyIQ2_XS
                | Ftype::MostlyIQ3_XXS
                | Ftype::MostlyIQ1_S
                | Ftype::MostlyIQ2_S
                | Ftype::MostlyIQ2_M
                | Ftype::MostlyIQ1_M
        ) {
            new_type = GgmlType::Q5K;
        } else if new_type != GgmlType::Q8_0 {
            new_type = GgmlType::Q6K;
        }
    } else if ftype == Ftype::MostlyMXFP4_MOE {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        // MLA projection tensors are also 3D, so match expert tensor roles explicitly.
        let is_bailingmoe3_expert = arch == LlmArch::BAILINGMOE3
            && (category == TensorCategory::FfnUp
                || category == TensorCategory::FfnGate
                || category == TensorCategory::FfnDown);
        if ne[2] > 1 && (arch != LlmArch::BAILINGMOE3 || is_bailingmoe3_expert) {
            new_type = GgmlType::Mxfp4;
        } else {
            new_type = GgmlType::Q8_0;
        }
    } else if category == TensorCategory::TokenEmbd {
        if matches!(
            ftype,
            Ftype::MostlyIQ2_XXS | Ftype::MostlyIQ2_XS | Ftype::MostlyIQ1_S | Ftype::MostlyIQ1_M
        ) {
            new_type = GgmlType::Q2K;
        } else if matches!(ftype, Ftype::MostlyIQ2_S | Ftype::MostlyIQ2_M) {
            new_type = GgmlType::Iq3S;
        } else if ftype == Ftype::MostlyIQ3_XXS {
            new_type = GgmlType::Iq3S;
        } else if matches!(
            ftype,
            Ftype::MostlyTQ1_0 | Ftype::MostlyTQ2_0 | Ftype::MostlyQ2_0
        ) {
            new_type = GgmlType::Q4K;
        }
    } else if matches!(
        ftype,
        Ftype::MostlyIQ2_XXS
            | Ftype::MostlyIQ2_XS
            | Ftype::MostlyIQ1_S
            | Ftype::MostlyIQ2_S
            | Ftype::MostlyIQ2_M
            | Ftype::MostlyIQ1_M
    ) {
        if category_is_attn_v(category) {
            if qs.model.n_gqa() >= 4 || qs.model.n_expert >= 4 {
                new_type = GgmlType::Q4K;
            } else {
                new_type = if matches!(ftype, Ftype::MostlyIQ2_S | Ftype::MostlyIQ2_M) {
                    GgmlType::Iq3S
                } else {
                    GgmlType::Q2K
                };
            }
            qs.i_attention_wv += 1;
        } else if qs.model.n_expert == 8 && category == TensorCategory::AttentionK {
            new_type = GgmlType::Q4K;
        } else if category == TensorCategory::FfnDown {
            if qs.i_ffn_down < qs.n_ffn_down / 8 {
                new_type = if matches!(ftype, Ftype::MostlyIQ2_S | Ftype::MostlyIQ2_M) {
                    GgmlType::Iq3S
                } else {
                    GgmlType::Q2K
                };
            }
            qs.i_ffn_down += 1;
        } else if category == TensorCategory::AttentionOutput {
            if qs.model.n_expert == 8 {
                new_type = GgmlType::Q5K;
            } else if matches!(ftype, Ftype::MostlyIQ1_S | Ftype::MostlyIQ1_M) {
                new_type = GgmlType::Iq2Xxs;
            } else if matches!(ftype, Ftype::MostlyIQ2_S | Ftype::MostlyIQ2_M) {
                new_type = GgmlType::Iq3S;
            }
        }
    } else if category_is_attn_v(category) {
        if ftype == Ftype::MostlyQ2_K {
            new_type = if qs.model.n_gqa() >= 4 {
                GgmlType::Q4K
            } else {
                GgmlType::Q3K
            };
        } else if ftype == Ftype::MostlyQ2_K_S && qs.model.n_gqa() >= 4 {
            new_type = GgmlType::Q4K;
        } else if ftype == Ftype::MostlyIQ3_XXS {
            new_type = if qs.model.n_gqa() >= 4 {
                GgmlType::Q4K
            } else if !qs.has_imatrix {
                GgmlType::Iq3S
            } else {
                GgmlType::Iq3Xxs
            };
        } else if matches!(ftype, Ftype::MostlyIQ3_XS | Ftype::MostlyIQ3_S) && qs.model.n_gqa() >= 4
        {
            new_type = GgmlType::Q4K;
        } else if ftype == Ftype::MostlyIQ3_M {
            new_type = GgmlType::Q4K;
        } else if ftype == Ftype::MostlyQ3_K_M {
            new_type = if qs.i_attention_wv < 2 {
                GgmlType::Q5K
            } else {
                GgmlType::Q4K
            };
        } else if ftype == Ftype::MostlyQ3_K_L {
            new_type = GgmlType::Q5K;
        } else if matches!(ftype, Ftype::MostlyIQ4_NL | Ftype::MostlyIQ4_XS)
            && qs.model.n_gqa() >= 4
        {
            new_type = GgmlType::Q5K;
        } else if matches!(ftype, Ftype::MostlyQ4_K_M | Ftype::MostlyQ5_K_M)
            && use_more_bits(qs.i_attention_wv, qs.n_attention_wv)
        {
            new_type = GgmlType::Q6K;
        } else if ftype == Ftype::MostlyQ4_K_S && qs.i_attention_wv < 4 {
            new_type = GgmlType::Q5K;
        }
        if qs.model.is_type_70b {
            // In the 70B model we have 8 heads sharing the same attn_v
            // weights. As a result, the attn_v.weight tensor is 8x smaller ...
            if new_type == GgmlType::Q3K || new_type == GgmlType::Q4K {
                new_type = GgmlType::Q5K;
            }
        }
        if qs.model.n_expert == 8 {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            new_type = GgmlType::Q8_0;
        }
        qs.i_attention_wv += 1;
    } else if category == TensorCategory::AttentionK {
        if qs.model.n_expert == 8 {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            new_type = GgmlType::Q8_0;
        } else if ftype == Ftype::MostlyIQ3_XS {
            new_type = GgmlType::Iq3Xxs;
        } else if ftype == Ftype::MostlyIQ3_XXS {
            new_type = GgmlType::Iq2S;
        }
    } else if category == TensorCategory::AttentionQ {
        if ftype == Ftype::MostlyIQ3_XS {
            new_type = GgmlType::Iq3Xxs;
        } else if ftype == Ftype::MostlyIQ3_XXS {
            new_type = GgmlType::Iq2S;
        }
    } else if category == TensorCategory::FfnDown {
        let (i_layer, n_layer) = layer_info(qs.i_ffn_down, qs.n_ffn_down, name, n_expert)?;
        if ftype == Ftype::MostlyQ2_K {
            new_type = GgmlType::Q3K;
        } else if ftype == Ftype::MostlyQ2_K_S {
            if i_layer < n_layer / 8 {
                new_type = GgmlType::Q4K;
            }
        } else if ftype == Ftype::MostlyIQ3_XXS && !qs.has_imatrix {
            new_type = if i_layer < n_layer / 8 {
                GgmlType::Q4K
            } else {
                GgmlType::Q3K
            };
        } else if ftype == Ftype::MostlyQ3_K_M {
            new_type = if i_layer < n_layer / 16 {
                GgmlType::Q5K
            } else if arch != LlmArch::FALCON || use_more_bits(i_layer, n_layer) {
                GgmlType::Q4K
            } else {
                GgmlType::Q3K
            };
        } else if ftype == Ftype::MostlyIQ3_M
            && (i_layer < n_layer / 8
                || (qs.model.n_expert == 8 && use_more_bits(i_layer, n_layer)))
        {
            new_type = GgmlType::Q4K;
        } else if ftype == Ftype::MostlyQ3_K_L {
            new_type = if arch == LlmArch::FALCON {
                GgmlType::Q4K
            } else {
                GgmlType::Q5K
            };
        } else if ftype == Ftype::MostlyQ4_K_M {
            if arch == LlmArch::FALCON {
                new_type = if i_layer < n_layer / 16 {
                    GgmlType::Q6K
                } else if use_more_bits(i_layer, n_layer) {
                    GgmlType::Q5K
                } else {
                    GgmlType::Q4K
                };
            } else if use_more_bits(i_layer, n_layer) {
                new_type = GgmlType::Q6K;
            }
        } else if i_layer < n_layer / 8
            && matches!(ftype, Ftype::MostlyIQ4_NL | Ftype::MostlyIQ4_XS)
            && !qs.has_imatrix
        {
            new_type = GgmlType::Q5K;
        } else if ftype == Ftype::MostlyQ5_K_M && use_more_bits(i_layer, n_layer) {
            new_type = GgmlType::Q6K;
        } else if ftype == Ftype::MostlyQ4_K_S && arch != LlmArch::FALCON && i_layer < n_layer / 8 {
            new_type = GgmlType::Q5K;
        } else if matches!(ftype, Ftype::MostlyQ4_0 | Ftype::MostlyQ5_0)
            && qs.has_imatrix
            && i_layer < n_layer / 8
        {
            // Guard against craziness in the first few ffn_down layers that
            // can happen even with imatrix for Q4_0/Q5_0. ...
            new_type = if ftype == Ftype::MostlyQ4_0 {
                GgmlType::Q4_1
            } else {
                GgmlType::Q5_1
            };
        }
        qs.i_ffn_down += 1;
    } else if category == TensorCategory::AttentionOutput {
        if arch != LlmArch::FALCON {
            if qs.model.n_expert == 8 {
                if matches!(
                    ftype,
                    Ftype::MostlyQ2_K
                        | Ftype::MostlyIQ3_XS
                        | Ftype::MostlyIQ3_XXS
                        | Ftype::MostlyQ3_K_S
                        | Ftype::MostlyQ3_K_M
                        | Ftype::MostlyIQ4_NL
                        | Ftype::MostlyQ4_K_S
                        | Ftype::MostlyQ4_K_M
                        | Ftype::MostlyIQ3_S
                        | Ftype::MostlyIQ3_M
                        | Ftype::MostlyIQ4_XS
                ) {
                    new_type = GgmlType::Q5K;
                }
            } else if ftype == Ftype::MostlyQ2_K {
                new_type = GgmlType::Q3K;
            } else if ftype == Ftype::MostlyIQ3_XXS {
                new_type = GgmlType::Iq3S;
            } else if ftype == Ftype::MostlyQ3_K_M {
                new_type = GgmlType::Q4K;
            } else if ftype == Ftype::MostlyQ3_K_L {
                new_type = GgmlType::Q5K;
            } else if ftype == Ftype::MostlyIQ3_M {
                new_type = GgmlType::Q4K;
            }
        } else if ftype == Ftype::MostlyQ3_K_L {
            new_type = GgmlType::Q4K;
        }
    } else if category == TensorCategory::AttentionQkv {
        if matches!(
            ftype,
            Ftype::MostlyQ3_K_M | Ftype::MostlyQ3_K_L | Ftype::MostlyIQ3_M
        ) {
            new_type = GgmlType::Q4K;
        } else if ftype == Ftype::MostlyQ4_K_M {
            new_type = GgmlType::Q5K;
        } else if ftype == Ftype::MostlyQ5_K_M {
            new_type = GgmlType::Q6K;
        }
    } else if category == TensorCategory::FfnGate {
        let (i_layer, n_layer) = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name, n_expert)?;
        if ftype == Ftype::MostlyIQ3_XS && (i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8) {
            new_type = GgmlType::Iq3Xxs;
        }
        qs.i_ffn_gate += 1;
    } else if category == TensorCategory::FfnUp {
        let (i_layer, n_layer) = layer_info(qs.i_ffn_up, qs.n_ffn_up, name, n_expert)?;
        if ftype == Ftype::MostlyIQ3_XS && (i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8) {
            new_type = GgmlType::Iq3Xxs;
        }
        qs.i_ffn_up += 1;
    }

    Ok(new_type)
}

// ---------------------------------------------------------------------------
// llama_tensor_get_type - llama-quant.cpp:683-739
// ---------------------------------------------------------------------------

/// `llama_tensor_get_type` (llama-quant.cpp:683-739).
///
/// `tm` carries the per-tensor metadata built in the preliminary loop
/// (`name`, `category`); `cur_type` is `tensor->type`.
pub fn llama_tensor_get_type(
    qs: &mut QuantizeState,
    params: &QuantizeParams,
    name: &str,
    ne: &[i64; 4],
    cur_type: GgmlType,
    default_type: GgmlType,
    tm: &TensorMetadata,
) -> Result<GgmlType, String> {
    if !tensor_allows_quantization(params, qs.model.arch, name, ne) {
        return Ok(cur_type);
    }
    if params.token_embedding_type.is_some() && tm.category == TensorCategory::TokenEmbd {
        // per_layer_token_embd follows --token-embedding-type by default, but
        // it is a large separate table, so let an explicit --tensor-type name it
        let mut named = false;
        if name == "per_layer_token_embd.weight" {
            for o in qs.tensor_type_patterns.iter() {
                if o.pattern.is_match(name) {
                    named = true;
                    break;
                }
            }
        }
        if !named {
            return Ok(params.token_embedding_type.unwrap());
        }
    }
    if params.output_tensor_type.is_some() && tm.category == TensorCategory::Output {
        return Ok(params.output_tensor_type.unwrap());
    }

    let mut new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if default_type.is_quantized() {
        // if the user provided tensor types - use those
        let mut manual = false;
        if !qs.tensor_type_patterns.is_empty() {
            for o in qs.tensor_type_patterns.iter() {
                if o.pattern.is_match(name) {
                    if o.ty != new_type {
                        log_warn(&format!(
                            "llama_tensor_get_type: {:<36} - applying manual override: {} -> {}\n",
                            name,
                            new_type.name(),
                            o.ty.name()
                        ));
                        new_type = o.ty;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization
        // type based on the selected mixture
        if !manual && !params.pure {
            new_type =
                llama_tensor_get_type_impl(qs, new_type, name, ne, params.ftype, tm.category)?;
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, name, ne[0], new_type)?;
    }

    Ok(new_type)
}

// ---------------------------------------------------------------------------
// tensor_requires_imatrix - llama-quant.cpp:822-841
// ---------------------------------------------------------------------------

/// `tensor_requires_imatrix` (llama-quant.cpp:822-841).
pub fn tensor_requires_imatrix(tensor_name: &str, dst_type: GgmlType, ftype: Ftype) -> bool {
    if tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name) {
        return false;
    }
    match dst_type {
        GgmlType::Iq3Xxs
        | GgmlType::Iq2Xxs
        | GgmlType::Iq2Xs
        | GgmlType::Iq2S
        | GgmlType::Iq1M
        | GgmlType::Iq1S => true,
        // as a general rule, the k-type quantizations don't require imatrix data.
        // the only exception is Q2_K tensors that are part of a Q2_K_S file.
        GgmlType::Q2K => ftype == Ftype::MostlyQ2_K_S,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// init_quantize_state_counters - llama-quant.cpp:893-907
// ---------------------------------------------------------------------------

/// `init_quantize_state_counters` (llama-quant.cpp:893-907): fills in each
/// metadata's category and the state's n_* counters.
pub fn init_quantize_state_counters(qs: &mut QuantizeState, metadata: &mut [TensorMetadata]) {
    for tm in metadata.iter_mut() {
        let cat = tensor_get_category(&tm.name);
        tm.category = cat;

        if category_is_attn_v(cat) {
            qs.n_attention_wv += 1;
        }

        if cat == TensorCategory::Output {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.model.n_layer_all;
    qs.n_ffn_gate = qs.model.n_layer_all;
    qs.n_ffn_up = qs.model.n_layer_all;
}

// ---------------------------------------------------------------------------
// tensor ordering - llama-model-loader.h:53-65 (weight_name_comparer)
// ---------------------------------------------------------------------------

/// `weight_name_comparer::operator()` (llama-model-loader.h:54-65): sort by
/// `sscanf("blk.%d.")` layer (-1 for non-blk tensors), then by name
/// (`std::string::operator<` == byte-wise).
pub fn weight_name_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let a_layer = parse_blk_layer(a).unwrap_or(-1);
    let b_layer = parse_blk_layer(b).unwrap_or(-1);
    if a_layer != b_layer {
        return a_layer.cmp(&b_layer);
    }
    a.cmp(b)
}

/// Sort a tensor list the way `ml.weights_map` (a `std::map` with
/// `weight_name_comparer`) iterates it — the order tensors are emitted in.
pub fn sort_weights_by_name<T>(items: &mut [T], name_of: impl Fn(&T) -> &str) {
    items.sort_by(|a, b| weight_name_cmp(name_of(a), name_of(b)));
}

// ---------------------------------------------------------------------------
// logging (the reference goes through LLAMA_LOG_WARN / LLAMA_LOG_ERROR)
// ---------------------------------------------------------------------------

/// `LLAMA_LOG_WARN` — routed through the ported log-routing surface
/// (`crate::impl_log`, llama-impl.cpp:32-36) so a registered callback sees
/// these lines; the default behavior is the same stderr output as before.
pub fn log_warn(msg: &str) {
    crate::llama_log_warn!("{}", msg);
}

/// `LLAMA_LOG_ERROR` — same routing as [`log_warn`].
pub fn log_error(msg: &str) {
    crate::llama_log_error!("{}", msg);
}
