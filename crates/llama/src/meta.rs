//! meta.rs — GGUF metadata → (arch, hparams) (port of the metadata-reading
//! path of llama.cpp bd4f514db1).
//!
//! Covers:
//!  - `llama_model_loader`'s typed KV accessors (`get_key` / `get_arr` /
//!    `get_key_or_arr` / `get_arr_n`, incl. the strict GGUFMeta wire-type
//!    checks) — `src/llama-model-loader.cpp`
//!  - `llama_model_base::load_hparams` — the arch-independent KV → hparams
//!    path plus its GGML_ASSERT / runtime_error validation — `src/llama-model.cpp`
//!  - per-arch `load_arch_hparams` — `src/models/<arch>.cpp` for the archs
//!    listed as [`ArchHparamsSupport::Full`]; everything else follows the
//!    generic path + defaults and is marked partial
//!  - `llama_model_rope_type` — the complete arch → rope-type switch
//!
//! Deviations (documented): GGML_ASSERT aborts become `Err(String)`; the
//! vocab-only / CLIP early-return keeps the same observable field defaults;
//! model "type" detection (LLM_TYPE_*) is description-string cosmetics and is
//! not ported here (only where an hparams field depends on it, e.g. gemma2/3
//! `f_attention_scale`, the equivalent layer-count condition is used).

use ggml::gguf::{Gguf, GgufType, Value};

use crate::arch::{kv_name, llm_arch_is_recurrent, LlmArch, LlmKv, LLAMA_MAX_LAYERS};
use crate::hparams::{
    LlamaHparams, LlamaNonCausalType, LlamaPoolingType, LlamaRopeScalingType, LlamaRopeType,
    LlamaSwaType, LlmFfnOpType, LLAMA_MAX_EXPERTS,
};

// ---------------------------------------------------------------------------
// typed KV access — port of llama_model_loader::get_key/get_arr/get_key_or_arr
// ---------------------------------------------------------------------------

/// GGUFMeta::GKV<T>::get_kv type check — "key %s has wrong type %s but
/// expected type %s".
fn wrong_type(key: &str, want: &str, got: GgufType) -> String {
    format!(
        "key {key} has wrong type {} but expected type {want}",
        got.name()
    )
}

fn not_found(key: &str) -> String {
    format!("key not found in model: {key}")
}

/// `get_key<T = uint32_t>`: strict GGUF_TYPE_UINT32.
fn get_key_u32(gguf: &Gguf, key: &str, required: bool) -> Result<Option<u32>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::U32(x)) => Ok(Some(*x)),
        Some(v) => Err(wrong_type(key, "uint32", v.type_())),
    }
}

/// `get_key<T = float>`: strict GGUF_TYPE_FLOAT32.
fn get_key_f32(gguf: &Gguf, key: &str, required: bool) -> Result<Option<f32>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::F32(x)) => Ok(Some(*x)),
        Some(v) => Err(wrong_type(key, "float32", v.type_())),
    }
}

/// `get_key<T = bool>`: strict GGUF_TYPE_BOOL.
fn get_key_bool(gguf: &Gguf, key: &str, required: bool) -> Result<Option<bool>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::Bool(x)) => Ok(Some(*x)),
        Some(v) => Err(wrong_type(key, "bool", v.type_())),
    }
}

/// `get_key<T = std::string>`: strict GGUF_TYPE_STRING.
fn get_key_string(gguf: &Gguf, key: &str, required: bool) -> Result<Option<String>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(wrong_type(key, "string", v.type_())),
    }
}

/// array element → u32 with the `get_arr<std::array<uint32_t, N>>` type rules
/// (BOOL/UINT32/INT32 accepted, widened).
fn arr_elem_u32(key: &str, v: &Value) -> Result<u32, String> {
    match v {
        Value::Bool(b) => Ok(u32::from(*b)),
        Value::U32(x) => Ok(*x),
        Value::I32(x) => u32::try_from(*x)
            .map_err(|_| format!("{key} is not a string/float32/uint32/int32/uint64 array")),
        _ => Err(format!(
            "{key} is not a string/float32/uint32/int32/uint64 array"
        )),
    }
}

/// array element → i32 (`get_arr<std::array<int32_t, N>>`).
fn arr_elem_i32(key: &str, v: &Value) -> Result<i32, String> {
    match v {
        Value::Bool(b) => Ok(i32::from(*b)),
        Value::U32(x) => i32::try_from(*x)
            .map_err(|_| format!("{key} is not a string/float32/uint32/int32/uint64 array")),
        Value::I32(x) => Ok(*x),
        _ => Err(format!(
            "{key} is not a string/float32/uint32/int32/uint64 array"
        )),
    }
}

fn wrong_array_elem(key: &str, got: GgufType) -> String {
    format!("{key} has wrong array element type {}", got.name())
}

/// `get_arr(key, std::array<uint32_t, LLAMA_MAX_LAYERS>&, required)`:
/// array of bool/uint32/int32, length <= LLAMA_MAX_LAYERS.
fn get_arr_u32(gguf: &Gguf, key: &str, required: bool) -> Result<Option<Vec<u32>>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(format!("array key not found in model: {key}"));
            }
            Ok(None)
        }
        Some(Value::Array(GgufType::Bool, items)) => {
            let vals = items
                .iter()
                .map(|v| arr_elem_u32(key, v))
                .collect::<Result<Vec<_>, _>>()?;
            finish_arr_u32(key, vals)
        }
        Some(Value::Array(GgufType::Uint32, items)) => {
            let vals = items
                .iter()
                .map(|v| arr_elem_u32(key, v))
                .collect::<Result<Vec<_>, _>>()?;
            finish_arr_u32(key, vals)
        }
        Some(Value::Array(GgufType::Int32, items)) => {
            let vals = items
                .iter()
                .map(|v| arr_elem_u32(key, v))
                .collect::<Result<Vec<_>, _>>()?;
            finish_arr_u32(key, vals)
        }
        Some(Value::Array(t, _)) => Err(format!(
            "{key} is not a string/float32/uint32/int32/uint64 array (element type {})",
            t.name()
        )),
        Some(v) => {
            if required || v.type_() == GgufType::Array {
                // C++ get_arr also fails the required check when the key is
                // present but not an array
                return Err(format!("array key not found in model: {key}"));
            }
            Ok(None)
        }
    }
}

fn finish_arr_u32(key: &str, vals: Vec<u32>) -> Result<Option<Vec<u32>>, String> {
    if vals.len() > LLAMA_MAX_LAYERS {
        return Err(format!(
            "array length {} for key {key} exceeds max {LLAMA_MAX_LAYERS}",
            vals.len()
        ));
    }
    Ok(Some(vals))
}

/// `get_arr(key, std::vector<std::string>&, required)`.
fn get_arr_str(gguf: &Gguf, key: &str, required: bool) -> Result<Option<Vec<String>>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(format!("array key not found in model: {key}"));
            }
            Ok(None)
        }
        Some(Value::Array(GgufType::String, items)) => {
            let vals = items
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err(wrong_array_elem(key, v.type_())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(vals))
        }
        Some(Value::Array(t, _)) => Err(wrong_array_elem(key, *t)),
        Some(_) => {
            if required {
                return Err(format!("array key not found in model: {key}"));
            }
            Ok(None)
        }
    }
}

/// `get_arr_n<T>(key, result, required)` — store the ARRAY LENGTH into result.
fn get_arr_n(gguf: &Gguf, key: &str, required: bool) -> Result<Option<u32>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::Array(_, items)) => Ok(Some(items.len() as u32)),
        Some(v) => Err(wrong_type(key, "array", v.type_())),
    }
}

/// `get_key_or_arr<T = uint32_t>(key, std::array<T, N>&, n, required)`:
/// a scalar broadcast to n entries, or an array of exactly n (n <= N_MAX).
fn get_key_or_arr_u32(
    gguf: &Gguf,
    key: &str,
    n: usize,
    required: bool,
) -> Result<Option<Vec<u32>>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::Array(_, items)) => {
            if items.len() != n {
                return Err(format!(
                    "key {key} has wrong array length; expected {n}, got {}",
                    items.len()
                ));
            }
            let vals = items
                .iter()
                .map(|v| arr_elem_u32(key, v))
                .collect::<Result<Vec<_>, _>>()?;
            finish_arr_u32(key, vals)
        }
        Some(Value::U32(x)) => Ok(Some(vec![*x; n])),
        Some(v) => Err(wrong_type(key, "uint32", v.type_())),
    }
}

/// `get_key_or_arr<T = float>(key, std::array<float, N>&, n, required)` —
/// the f32 twin of [`get_key_or_arr_u32`] (deepseek4's swiglu_clamp_exp /
/// swiglu_clamp_shexp, deepseek4.cpp:36-39).
fn get_key_or_arr_f32(
    gguf: &Gguf,
    key: &str,
    n: usize,
    required: bool,
) -> Result<Option<Vec<f32>>, String> {
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::Array(GgufType::Float32, items)) => {
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
                    v => Err(wrong_type(key, "float32", v.type_())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            if vals.len() > LLAMA_MAX_LAYERS {
                return Err(format!("key {key} has too many elements"));
            }
            Ok(Some(vals))
        }
        Some(Value::F32(x)) => Ok(Some(vec![*x; n])),
        Some(v) => Err(wrong_type(key, "float32", v.type_())),
    }
}

/// `get_key_or_arr<T = int32_t>(key, rope_sections, 4, required)` — the
/// rope-sections specialisation (`std::array<int, 4>`).
pub(crate) fn get_key_or_arr_rope_sections(
    gguf: &Gguf,
    key: &str,
    required: bool,
) -> Result<Option<[i32; 4]>, String> {
    const N: usize = 4;
    match gguf.find_key(key) {
        None => {
            if required {
                return Err(not_found(key));
            }
            Ok(None)
        }
        Some(Value::Array(_, items)) => {
            if items.len() != N {
                return Err(format!(
                    "key {key} has wrong array length; expected {N}, got {}",
                    items.len()
                ));
            }
            let mut out = [0i32; N];
            for (o, v) in out.iter_mut().zip(items.iter()) {
                *o = arr_elem_i32(key, v)?;
            }
            Ok(Some(out))
        }
        Some(Value::I32(x)) => Ok(Some([*x; N])),
        Some(v) => Err(wrong_type(key, "int32", v.type_())),
    }
}

// ---------------------------------------------------------------------------
// load_hparams — llama_model_base::load_hparams
// ---------------------------------------------------------------------------

/// Port-status of an arch's `load_arch_hparams` (not present in C++ — this is
/// a migration marker used by the port's callers/tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchHparamsSupport {
    /// `load_arch_hparams` ported 1:1 from src/models/<arch>.cpp.
    Full,
    /// only the arch-independent path + defaults run; the arch-specific KV
    /// reads (norm eps, MoE dims, ...) are not applied yet.
    Partial,
}

/// Which archs have a 1:1 `load_arch_hparams` port in this file.
pub fn arch_hparams_support(arch: LlmArch) -> ArchHparamsSupport {
    use LlmArch::*;
    match arch {
        LLAMA | QWEN2 | QWEN3 | GEMMA | GEMMA2 | GEMMA3 | GEMMA4 | PHI3 | DEEPSEEK | DEEPSEEK2
        | DEEPSEEK2OCR | DEEPSEEK32 | OPENAI_MOE | ERNIE4_5 | ERNIE4_5_MOE | LFM2 | LFM2MOE
        | BERT | GRANITE_HYBRID | QWEN35
        // the bert-variant family (jina-bert-v2 / jina-bert-v3 / nomic-bert /
        // nomic-bert-moe / neo-bert / modern-bert)
        | JINA_BERT_V2 | JINA_BERT_V3 | NOMIC_BERT | NOMIC_BERT_MOE | NEO_BERT | MODERN_BERT => {
            ArchHparamsSupport::Full
        }
        CLIP => ArchHparamsSupport::Full, // hparams path is a trivial early return in C++
        _ => ArchHparamsSupport::Partial,
    }
}

/// Read `general.architecture` and resolve the arch (llama_model_loader ctor).
/// An absent / unrecognized value yields UNKNOWN, exactly like
/// `llm_arch_from_string`; the subsequent required arch-prefixed KV read then
/// fails with the C++ error text ("key not found in model: (unknown).context_length").
fn detect_arch(gguf: &Gguf) -> LlmArch {
    let name = get_key_string(
        gguf,
        kv_name(LlmArch::UNKNOWN, LlmKv::GENERAL_ARCHITECTURE).as_str(),
        false,
    )
    .ok()
    .flatten()
    .unwrap_or_default();
    LlmArch::from_name(&name).unwrap_or(LlmArch::UNKNOWN)
}

/// Port of `llama_model_base::load_hparams` (generic path) + per-arch
/// `load_arch_hparams`.
///
/// Note: the C++ `hparams.vocab_only` early return cannot trigger here (the
/// vocab module is loaded separately in the port), so the full path always
/// runs — matching a normal `llama_model_load` invocation.
/// `LLM_CLS_ACT_TYPES_FROM_STRING` (37ac63456, llama-model.cpp): the
/// transformers names — "gelu" is the exact (erf) variant.
fn cls_act_from_string(name: &str) -> Option<i32> {
    match name {
        "gelu" => Some(ggml::ops::GGML_UNARY_OP_GELU_ERF),
        "silu" => Some(ggml::ops::GGML_UNARY_OP_SILU),
        "tanh" => Some(ggml::ops::GGML_UNARY_OP_TANH),
        _ => None,
    }
}

pub fn load_hparams(gguf: &Gguf) -> Result<(LlmArch, LlamaHparams), String> {
    let arch = detect_arch(gguf);

    // for CLIP models, we only need to load tensors, no hparams
    if arch == LlmArch::CLIP {
        return Ok((arch, LlamaHparams::new()));
    }

    let mut h = LlamaHparams::new();
    load_hparams_generic(gguf, arch, &mut h)?;
    load_arch_hparams(gguf, arch, &mut h)?;

    if h.f_max_alibi_bias > 0.0 {
        h.use_alibi = true;
    }

    h.rope_type = llama_model_rope_type(arch, &h);

    Ok((arch, h))
}

/// The arch-independent prologue of `llama_model_base::load_hparams`.
fn load_hparams_generic(gguf: &Gguf, arch: LlmArch, h: &mut LlamaHparams) -> Result<(), String> {
    let k = |kv: LlmKv| kv_name(arch, kv);

    // required general model keys
    h.n_ctx_train = get_key_u32(gguf, &k(LlmKv::CONTEXT_LENGTH), true)?.unwrap();
    h.n_embd = get_key_u32(gguf, &k(LlmKv::EMBEDDING_LENGTH), true)?.unwrap();
    // the classifier head activation (37ac63456, llama-model.cpp:1382-1388)
    if let Some(act) = get_key_string(gguf, &k(LlmKv::CLASSIFIER_ACTIVATION), false)? {
        h.act_cls = cls_act_from_string(&act)
            .ok_or_else(|| format!("unsupported classifier activation: {act}"))?;
    }

    if let Some(v) = get_key_u32(gguf, &k(LlmKv::EMBEDDING_LENGTH_OUT), false)? {
        h.n_embd_out_impl = v;
    }
    if let Some(v) = get_key_bool(gguf, &k(LlmKv::ATTENTION_CAUSAL), false)? {
        h.causal_attn = v;
    }
    // enum specialization: not found => UNSPECIFIED (not the field default)
    h.pooling_type = match get_key_u32(gguf, &k(LlmKv::POOLING_TYPE), false)? {
        Some(v) => pooling_type_from_u32(v),
        None => LlamaPoolingType::UNSPECIFIED,
    };
    // the classifier-head pooling (RANK) — llama-model.cpp:1326 (def4d406a):
    // `%s.classifier.pooling_type`, optional; UNSPECIFIED lets the arch
    // default (modern-bert: MEAN)
    h.pooling_type_cls = match get_key_u32(gguf, &k(LlmKv::CLASSIFIER_POOLING_TYPE), false)? {
        Some(v) => pooling_type_from_u32(v),
        None => LlamaPoolingType::UNSPECIFIED,
    };
    h.n_layer_all = get_key_u32(gguf, &k(LlmKv::BLOCK_COUNT), true)?.unwrap();
    // GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS)
    if h.n_layer_all == 0 || h.n_layer_all as usize > LLAMA_MAX_LAYERS {
        return Err(format!(
            "invalid block count {} (must be in 1..={LLAMA_MAX_LAYERS})",
            h.n_layer_all
        ));
    }
    if let Some(v) = get_key_u32(gguf, &k(LlmKv::NEXTN_PREDICT_LAYERS), false)? {
        h.n_layer_nextn = v;
    }
    // GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all)
    if h.n_layer_nextn > h.n_layer_all {
        return Err(format!(
            "nextn_predict_layers {} > block_count {}",
            h.n_layer_nextn, h.n_layer_all
        ));
    }
    if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_COUNT), false)? {
        h.n_expert = v;
    }
    // std::fill(n_expert_used_arr, 0) is already the fresh-struct state
    if let Some(vals) = get_key_or_arr_u32(
        gguf,
        &k(LlmKv::EXPERT_USED_COUNT),
        h.n_layer_all as usize,
        false,
    )? {
        h.n_expert_used_arr[..vals.len()].copy_from_slice(&vals);
    }
    if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_GROUP_COUNT), false)? {
        h.n_expert_groups = v;
    }
    if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_GROUP_USED_COUNT), false)? {
        h.n_group_used = v;
    }

    if arch == LlmArch::HUNYUAN_VL || arch == LlmArch::HUNYUAN_DENSE {
        if h.n_expert <= 1 {
            h.n_expert = 0;
            h.n_expert_used_arr.iter_mut().for_each(|v| *v = 0);
        }
    }

    if arch == LlmArch::WAVTOKENIZER_DEC {
        h.n_embd = get_key_u32(gguf, &k(LlmKv::FEATURES_LENGTH), true)?.unwrap();
        h.n_embd_out_impl = get_key_u32(gguf, &k(LlmKv::EMBEDDING_LENGTH), true)?.unwrap();

        h.posnet.n_embd = get_key_u32(gguf, &k(LlmKv::POSNET_EMBEDDING_LENGTH), true)?.unwrap();
        h.posnet.n_layer = get_key_u32(gguf, &k(LlmKv::POSNET_BLOCK_COUNT), true)?.unwrap();

        h.convnext.n_embd = get_key_u32(gguf, &k(LlmKv::CONVNEXT_EMBEDDING_LENGTH), true)?.unwrap();
        h.convnext.n_layer = get_key_u32(gguf, &k(LlmKv::CONVNEXT_BLOCK_COUNT), true)?.unwrap();

        // GGML_ASSERT(posnet.n_layer <= n_layer_all && convnext.n_layer <= n_layer_all)
        if h.posnet.n_layer > h.n_layer_all || h.convnext.n_layer > h.n_layer_all {
            return Err(
                "wavtokenizer: posnet/convnext block_count exceeds block_count".to_string(),
            );
        }
    }
    // models may route a different number of experts per layer, so validate the maximum
    let n_expert_used_max = h.n_expert_used_max();

    // GGML_ASSERT(n_expert <= LLAMA_MAX_EXPERTS); GGML_ASSERT(n_expert_used_max <= n_expert)
    if h.n_expert > LLAMA_MAX_EXPERTS {
        return Err(format!(
            "expert_count {} > LLAMA_MAX_EXPERTS {LLAMA_MAX_EXPERTS}",
            h.n_expert
        ));
    }
    if n_expert_used_max > h.n_expert {
        return Err(format!(
            "expert_used_count {n_expert_used_max} > expert_count {}",
            h.n_expert
        ));
    }
    if h.n_expert > 0 {
        if n_expert_used_max == 0 {
            return Err("model has expert layers but no expert layers are used".to_string());
        }
        // GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert)
        if h.n_expert_groups >= h.n_expert {
            return Err(format!(
                "expert_group_count {} >= expert_count {}",
                h.n_expert_groups, h.n_expert
            ));
        }
        if h.n_expert_groups > 1 {
            if h.n_expert % h.n_expert_groups != 0 {
                return Err(format!(
                    "expert_count {} not divisible by expert_group_count {}",
                    h.n_expert, h.n_expert_groups
                ));
            }
            if h.n_group_used == 0 {
                return Err("expert_group_used_count == 0".to_string());
            }
            if h.n_group_used >= h.n_expert_groups {
                return Err(format!(
                    "expert_group_used_count {} >= expert_group_count {}",
                    h.n_group_used, h.n_expert_groups
                ));
            }
        }
    } else {
        if n_expert_used_max != 0 {
            return Err("expert_used_count set without expert_count".to_string());
        }
        if h.n_expert_groups != 0 {
            return Err("expert_group_count set without expert_count".to_string());
        }
    }

    // the fresh struct already mirrors the std::fill prologue (zeros /
    // rope_pattern=1 / is_swa_impl=0 / is_recr per arch / deepstack=-1 / ...),
    // except the arch-conditional is_recr default:
    if llm_arch_is_recurrent(arch) {
        h.is_recr_impl.iter_mut().for_each(|v| *v = 1);
    }

    if let Some(vals) = get_key_or_arr_u32(
        gguf,
        &k(LlmKv::FEED_FORWARD_LENGTH),
        h.n_layer_all as usize,
        false,
    )? {
        h.n_ff_arr[..vals.len()].copy_from_slice(&vals);
    }
    if let Some(vals) = get_key_or_arr_u32(
        gguf,
        &k(LlmKv::ATTENTION_HEAD_COUNT),
        h.n_layer_all as usize,
        false,
    )? {
        h.n_head_arr[..vals.len()].copy_from_slice(&vals);
    }

    // n_head_kv is optional, default to n_head
    h.n_head_kv_arr.copy_from_slice(&h.n_head_arr);
    if let Some(vals) = get_key_or_arr_u32(
        gguf,
        &k(LlmKv::ATTENTION_HEAD_COUNT_KV),
        h.n_layer_all as usize,
        false,
    )? {
        h.n_head_kv_arr[..vals.len()].copy_from_slice(&vals);
    }

    h.rope_finetuned =
        get_key_bool(gguf, &k(LlmKv::ROPE_SCALING_FINETUNED), false)?.unwrap_or(false);

    h.n_ctx_orig_yarn = h.n_ctx_train;
    if let Some(v) = get_key_u32(gguf, &k(LlmKv::ROPE_SCALING_ORIG_CTX_LEN), false)? {
        h.n_ctx_orig_yarn = v;
    }

    // rope_freq_base (optional)
    h.rope_freq_base_train = 10000.0;
    if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE), false)? {
        h.rope_freq_base_train = v;
    }

    let rope_scaling = get_key_string(gguf, &k(LlmKv::ROPE_SCALING_TYPE), false)?
        .unwrap_or_else(|| "linear".to_string());
    h.rope_scaling_type_train = LlamaRopeScalingType::from_name(&rope_scaling);
    // GGML_ASSERT(rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED)
    if h.rope_scaling_type_train == LlamaRopeScalingType::UNSPECIFIED {
        return Err(format!("invalid rope scaling type: {rope_scaling}"));
    }

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    let mut ropescale = 0.0f32;
    let got = get_key_f32(gguf, &k(LlmKv::ROPE_SCALING_FACTOR), false)?;
    let got = match got {
        Some(v) => Some(v),
        None => {
            // try the old key name
            get_key_f32(gguf, &k(LlmKv::ROPE_SCALE_LINEAR), false)?
        }
    };
    if let Some(v) = got {
        ropescale = v;
    }
    h.rope_freq_scale_train = if ropescale == 0.0 {
        1.0
    } else {
        1.0 / ropescale
    };

    if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_SCALING_ATTN_FACTOR), false)? {
        h.rope_attn_factor = v;
    }
    if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_SCALING_ALPHA), false)? {
        h.rope_scaling_alpha = v;
    }

    // non-transformer models do not have attention heads
    if h.n_head(0) > 0 {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        h.n_embd_head_k_full = h.n_embd / h.n_head(0);
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH), false)? {
            h.n_embd_head_k_full = v;
        }

        h.n_embd_head_v_full = h.n_embd / h.n_head(0);
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH), false)? {
            h.n_embd_head_v_full = v;
        }

        // sanity check for n_rot (optional)
        h.n_rot_full = h.n_embd_head_k_full;
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ROPE_DIMENSION_COUNT), false)? {
            h.n_rot_full = v;
        }

        if matches!(
            arch,
            LlmArch::LLAMA | LlmArch::DECI | LlmArch::FALCON | LlmArch::LLAMA_EMBED
        ) && h.n_rot_full != h.n_embd_head_k_full
        {
            return Err(format!(
                "invalid n_rot: {}, expected {}",
                h.n_rot_full, h.n_embd_head_k_full
            ));
        }
    } else {
        h.n_rot_full = 0;
        h.n_embd_head_k_full = 0;
        h.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        h.n_embd_head_k_swa = h.n_embd_head_k_full;
        h.n_embd_head_v_swa = h.n_embd_head_v_full;
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_SWA), false)? {
            h.n_embd_head_k_swa = v;
        }
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_SWA), false)? {
            h.n_embd_head_v_swa = v;
        }

        h.n_rot_swa = h.n_rot_full;
        if let Some(v) = get_key_u32(gguf, &k(LlmKv::ROPE_DIMENSION_COUNT_SWA), false)? {
            h.n_rot_swa = v;
        }
    }

    // for classifier models
    if let Some(labels) = get_arr_str(gguf, &k(LlmKv::CLASSIFIER_OUTPUT_LABELS), false)? {
        if !labels.is_empty() {
            h.n_cls_out = labels.len() as u32;
        }
    }

    Ok(())
}

/// `ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false)`
/// (llama-model.cpp:1405-1408) — the label list the model keeps beside
/// `n_cls_out` (consumed by print_info and the model saver).
pub fn read_classifier_labels(gguf: &Gguf, arch: LlmArch) -> Result<Vec<String>, String> {
    let k = |kv: LlmKv| kv_name(arch, kv);
    Ok(get_arr_str(gguf, &k(LlmKv::CLASSIFIER_OUTPUT_LABELS), false)?.unwrap_or_default())
}

fn pooling_type_from_u32(v: u32) -> LlamaPoolingType {
    match v {
        0 => LlamaPoolingType::NONE,
        1 => LlamaPoolingType::MEAN,
        2 => LlamaPoolingType::CLS,
        3 => LlamaPoolingType::LAST,
        4 => LlamaPoolingType::RANK,
        _ => LlamaPoolingType::UNSPECIFIED,
    }
}

/// `llama_model_base::load_swa_pattern`
fn load_swa_pattern(
    gguf: &Gguf,
    arch: LlmArch,
    h: &mut LlamaHparams,
    n_pattern: u32,
    dense_first: bool,
) -> Result<(), String> {
    if let Some(vals) = get_arr_u32(
        gguf,
        &kv_name(arch, LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN),
        false,
    )? {
        h.is_swa_impl = vec![0u32; LLAMA_MAX_LAYERS];
        h.is_swa_impl[..vals.len()].copy_from_slice(&vals);
        return Ok(());
    }

    let n_pattern = get_key_u32(
        gguf,
        &kv_name(arch, LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN),
        false,
    )?
    .unwrap_or(n_pattern);
    h.set_swa_pattern(n_pattern, dense_first);
    Ok(())
}

// ---------------------------------------------------------------------------
// per-arch load_arch_hparams — src/models/<arch>.cpp (1:1 for Full archs)
// ---------------------------------------------------------------------------

fn load_arch_hparams(gguf: &Gguf, arch: LlmArch, h: &mut LlamaHparams) -> Result<(), String> {
    let k = |kv: LlmKv| kv_name(arch, kv);

    match arch {
        // ---- models/llama.cpp (LLAMA / LLAMA4 family base); llama-embed
        // reuses the arm (models.h:175-177) ----
        LlmArch::LLAMA | LlmArch::LLAMA_EMBED => {
            // n_vocab is only used for the type heuristic in C++
            let _n_vocab = get_key_u32(gguf, &k(LlmKv::VOCAB_SIZE), false)?
                .map(Ok)
                .unwrap_or_else(|| {
                    get_arr_n(gguf, &kv_name(arch, LlmKv::TOKENIZER_LIST), false)
                        .map(|x| x.unwrap_or(0))
                })?;

            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/qwen2.cpp ----
        LlmArch::QWEN2 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/qwen3.cpp ----
        LlmArch::QWEN3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/gemma.cpp ----
        LlmArch::GEMMA => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/gemma2.cpp ----
        LlmArch::GEMMA2 => {
            h.swa_type = LlamaSwaType::STANDARD;
            h.n_swa = 4096; // default value of gemma 2
            load_swa_pattern(gguf, arch, h, 2, false)?;
            h.attn_soft_cap = true;
            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;

            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                h.rope_freq_base_train_swa = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)? {
                h.n_swa = v;
            }
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTN_LOGIT_SOFTCAPPING), false)? {
                h.f_attn_logit_softcapping = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::FINAL_LOGIT_SOFTCAPPING), false)? {
                h.f_final_logit_softcapping = v;
            }

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173
            // (type == LLM_TYPE_27B iff n_layer() == 46)
            h.f_attention_scale = if h.n_layer() == 46 {
                1.0 / ((h.n_embd / h.n_head(0)) as f32).sqrt()
            } else {
                1.0 / (h.n_embd_head_k_full as f32).sqrt()
            };
        }

        // ---- models/gemma3.cpp ----
        LlmArch::GEMMA3 => {
            let found_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)?;
            let found_swa_and_positive = matches!(found_swa, Some(v) if v > 0);
            if let Some(v) = found_swa {
                h.n_swa = v;
            }
            if found_swa_and_positive {
                h.swa_type = LlamaSwaType::STANDARD;
                load_swa_pattern(gguf, arch, h, 6, false)?;

                if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                    h.rope_freq_base_train_swa = v;
                }
            } else {
                h.swa_type = LlamaSwaType::NONE;
            }

            h.f_final_logit_softcapping = 0.0;
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::FINAL_LOGIT_SOFTCAPPING), false)? {
                h.f_final_logit_softcapping = v;
            }
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L289
            // (type == LLM_TYPE_27B iff n_layer() == 62)
            h.f_attention_scale = if h.n_layer() == 62 {
                1.0 / ((h.n_embd / h.n_head(0)) as f32).sqrt()
            } else {
                1.0 / (h.n_embd_head_k_full as f32).sqrt()
            };
        }

        // ---- models/gemma4.cpp ----
        LlmArch::GEMMA4 => {
            h.swa_type = LlamaSwaType::STANDARD;
            let vals =
                get_arr_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN), true)?.unwrap();
            h.is_swa_impl = vec![0u32; LLAMA_MAX_LAYERS];
            h.is_swa_impl[..vals.len()].copy_from_slice(&vals);

            let n_kv_shared_layers =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_SHARED_KV_LAYERS), false)?.unwrap_or(0);

            h.n_layer_kv_from_start = h.n_layer_all as i32 - n_kv_shared_layers as i32;
            // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
            h.f_attention_scale = 1.0;

            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                h.rope_freq_base_train_swa = v;
            }
            if let Some(vals) = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                false,
            )? {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            h.n_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), true)?.unwrap();
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.n_embd_per_layer =
                get_key_u32(gguf, &k(LlmKv::EMBEDDING_LENGTH_PER_LAYER), true)?.unwrap();
            h.n_embd_head_k_swa =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_SWA), true)?.unwrap();
            h.n_embd_head_v_swa =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_SWA), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::FINAL_LOGIT_SOFTCAPPING), false)? {
                h.f_final_logit_softcapping = v;
            }

            // when non_causal is set, the model will use bidirectional attention
            // on SWA layers only, while dense layers will remain causal
            // ref: use_bidirectional_attention == "vision" in HF config
            // note: E2B/E4B are always causal, bypassing this logic
            h.non_causal_type = LlamaNonCausalType::SWA_ONLY;
        }

        // ---- models/phi3.cpp ----
        LlmArch::PHI3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            let found_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)?;
            if let Some(v) = found_swa {
                h.n_swa = v;
            }
            if matches!(found_swa, Some(v) if v > 0) {
                // "Phi SWA is currently disabled - results might be suboptimal
                // for some models" (llama.cpp#13676)
                //
                // TODO: fix conversion scripts to correctly populate `n_swa`
                // and `n_swa_pattern`
                h.swa_type = LlamaSwaType::NONE;

                h.n_swa = 0;
                h.set_swa_pattern(1, false);
            }
        }

        // ---- models/deepseek2.cpp ----
        LlmArch::DEEPSEEK2 | LlmArch::MISTRAL4 => {
            let n_vocab = get_key_u32(gguf, &k(LlmKv::VOCAB_SIZE), false)?
                .map(Ok)
                .unwrap_or_else(|| {
                    get_arr_n(gguf, &kv_name(arch, LlmKv::TOKENIZER_LIST), false)
                        .map(|x| x.unwrap_or(0))
                })?;

            // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B, Kanana-2-30B-A3B
            let is_lite =
                h.n_layer() == 27 || h.n_layer() == 26 || (h.n_layer() == 48 && n_vocab == 128256);

            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            if !is_lite {
                h.n_lora_q = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), true)?.unwrap();
            }
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), false)? {
                h.n_embd_head_k_mla_impl = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), false)? {
                h.n_embd_head_v_mla_impl = v;
            }
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), false)? {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == crate::hparams::LlamaExpertGatingFuncType::NONE as u32 {
                // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
                // that have no expert_gating_func model parameter set
                if (h.n_layer() == 47 || h.n_layer() == 48) && n_vocab == 154880 {
                    // GLM 4.7 Lite
                    h.expert_gating_func =
                        crate::hparams::LlamaExpertGatingFuncType::SIGMOID as u32;
                } else {
                    h.expert_gating_func =
                        crate::hparams::LlamaExpertGatingFuncType::SOFTMAX as u32;
                }
            }

            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_SCALING_YARN_LOG_MUL), false)? {
                h.rope_yarn_log_mul = v;
                // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
                // cancel the factor from the convert script
                h.rope_yarn_log_mul /= 0.1;
            }

            // (optional) temperature tuning - used by mistral-large
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_TEMPERATURE_SCALE), false)? {
                h.f_attn_temp_scale = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_TEMPERATURE_LENGTH), false)? {
                h.n_attn_temp_floor_scale = v; // FIXME why not use temperature_length?
            }

            h.f_attn_temp_offset = 0.0;
        }

        // ---- models/deepseek32.cpp (DeepSeek V3.2) ----
        LlmArch::DEEPSEEK32 => {
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.f_norm_eps = 1e-6; // eps for layer norm
            if let Some(sections) =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), false)?
            {
                h.rope_sections = sections;
            }

            // MoE parameters
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }

            // deepseek MLA parameters
            h.n_lora_q = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), true)?.unwrap();
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), false)? {
                h.n_embd_head_k_mla_impl = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), false)? {
                h.n_embd_head_v_mla_impl = v;
            }

            // DSA parameters
            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();

            // Expert gating function
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();

            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_SCALING_YARN_LOG_MUL), false)? {
                h.rope_yarn_log_mul = v;
                // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
                // cancel the factor from the convert script
                h.rope_yarn_log_mul /= 0.1;
            }
        }

        // ---- arch batch 7: models/deepseek4.cpp:19-80 load_arch_hparams
        // (hyper-connection head counts, per-layer swiglu clamps,
        // compress_ratios, SQRT_SOFTPLUS gating, SWA_FULL non-causal type) ----
        LlmArch::DEEPSEEK4 => {
            // deepseek4.cpp:20-26 — drop n_layer_nextn when the MTP block's
            // tensors are absent (ml.get_weight probe)
            if h.n_layer_nextn > 0 {
                let n_layer_main = h.n_layer_all - h.n_layer_nextn;
                let mtp_probe = format!("blk.{n_layer_main}.nextn.eh_proj.weight");
                if gguf.find_tensor(&mtp_probe).is_none() {
                    h.n_layer_nextn = 0;
                }
            }

            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.n_lora_q = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), true)?.unwrap();
            h.n_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), true)?.unwrap();

            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            // deepseek4.cpp:34-35 — both required (unlike deepseek32's
            // optional reads)
            h.expert_weights_scale =
                get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), true)?.unwrap();
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            // swiglu_clamp_exp / shexp (deepseek4.cpp:36-39) — f32 arrays
            // broadcastable to n_layer_all; shexp falls back to exp when absent
            let clamp_exp = get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_EXP),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.swiglu_clamp_exp[..clamp_exp.len()].copy_from_slice(&clamp_exp);
            match get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_SHEXP),
                h.n_layer_all as usize,
                false,
            )? {
                Some(v) => h.swiglu_clamp_shexp[..v.len()].copy_from_slice(&v),
                None => h.swiglu_clamp_shexp = h.swiglu_clamp_exp.clone(),
            }

            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();

            h.dsv4_o_group_count =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_OUTPUT_GROUP_COUNT), true)?.unwrap();
            h.dsv4_o_lora_rank =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_OUTPUT_LORA_RANK), true)?.unwrap();
            h.dsv4_compress_rope_base =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_COMPRESS_ROPE_FREQ_BASE), true)?.unwrap();
            h.dsv4_hc_mult = get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_COUNT), true)?.unwrap();
            h.dsv4_hc_sinkhorn_iters =
                get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_SINKHORN_ITERATIONS), true)?.unwrap();
            h.dsv4_hc_eps = get_key_f32(gguf, &k(LlmKv::HYPER_CONNECTION_EPSILON), true)?.unwrap();
            h.dsv4_hash_layer_count =
                get_key_u32(gguf, &k(LlmKv::HASH_LAYER_COUNT), true)?.unwrap();

            h.n_embd_out_impl = h.dsv4_hc_mult * h.n_embd;

            // deepseek4.cpp:55-61 — compress_ratios must cover block_count
            let n_compress_ratios =
                get_arr_n(gguf, &k(LlmKv::ATTENTION_COMPRESS_RATIOS), true)?.unwrap();
            if (n_compress_ratios as usize) < h.n_layer_all as usize {
                return Err("DeepSeek-V4 compress_ratios is shorter than block_count".into());
            }
            let ratios = get_arr_u32(gguf, &k(LlmKv::ATTENTION_COMPRESS_RATIOS), true)?.unwrap();
            h.dsv4_compress_ratios = ratios;

            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();
            if h.expert_gating_func
                != crate::hparams::LlamaExpertGatingFuncType::SQRT_SOFTPLUS as u32
            {
                return Err("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring".into());
            }
            h.swa_type = LlamaSwaType::STANDARD;
            h.set_swa_pattern(0, false);
            // tokens of an image span attend bidirectionally to the whole
            // span, the window only applies to older tokens (deepseek4.cpp:69-71)
            h.non_causal_type = LlamaNonCausalType::SWA_FULL;
            for il in h.n_layer() as usize..h.n_layer_all as usize {
                h.is_swa_impl[il] = 1;
            }
        }

        // ---- models/deepseek.cpp (DeepSeek v2 base, non-MLA) ----
        LlmArch::DEEPSEEK => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
        }

        // ---- models/deepseek2ocr.cpp (deepseek2's graph, always non-MLA) ----
        LlmArch::DEEPSEEK2OCR => {
            // similar to deepseek2, but without MLA
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), false)? {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == crate::hparams::LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = crate::hparams::LlamaExpertGatingFuncType::SOFTMAX as u32;
            }
        }

        // ---- models/openai-moe.cpp (gpt-oss) ----
        LlmArch::OPENAI_MOE => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.n_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), true)?.unwrap();

            h.swa_type = LlamaSwaType::STANDARD;
            load_swa_pattern(gguf, arch, h, 2, false)?;

            h.rope_freq_base_train_swa = h.rope_freq_base_train;
            h.rope_freq_scale_train_swa = h.rope_freq_scale_train;
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                h.rope_freq_base_train_swa = v;
            }
        }

        // ---- models/glm4.cpp:3-13 (the rope.sections read specifically) ----
        // The full GLM4 hparams arm lives in model.rs's load_arch_hparams_
        // batch (the batch-6+ convention), but the rope.sections key must be
        // read HERE: llama_model_rope_type's GLM4 arm is dynamic
        // (use_mrope() ? MROPE : NORM, llama-model.cpp:3072-3073) and this
        // function's caller computes h.rope_type right after — the batch arm
        // runs later (inside load_model) and would leave rope_type = NORM on
        // section-bearing files. The batch arm re-reads the same key
        // (idempotent, like the C's single read in glm4.cpp:5).
        LlmArch::GLM4 => {
            if let Some(sections) =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), false)?
            {
                h.rope_sections = sections;
            }
        }

        // ---- models/hunyuan-vl.cpp:5 (the rope.sections read specifically) —
        // like GLM4 above, llama_model_rope_type's HUNYUAN_VL arm is dynamic
        // (use_mrope() ? MROPE : NEOX, llama-model.cpp:3125-3126) and runs
        // right after this function; the batch arm re-reads the same key
        // (idempotent, like the C's single read) ----
        LlmArch::HUNYUAN_VL | LlmArch::HUNYUAN_DENSE => {
            if let Some(sections) =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), false)?
            {
                h.rope_sections = sections;
            }
        }

        // ---- models/ernie4-5.cpp (dense + MoE share one class) ----
        LlmArch::ERNIE4_5 | LlmArch::ERNIE4_5_MOE | LlmArch::PADDLEOCR => {
            // paddleocr need mrope_section
            if let Some(sections) =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), false)?
            {
                h.rope_sections = sections;
            }

            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if arch == LlmArch::ERNIE4_5_MOE {
                let vals = get_key_or_arr_u32(
                    gguf,
                    &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                    h.n_layer_all as usize,
                    true,
                )?
                .unwrap();
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
                if let Some(v) =
                    get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
                {
                    h.n_ff_shexp = v;
                }
                h.n_moe_layer_step =
                    get_key_u32(gguf, &k(LlmKv::INTERLEAVE_MOE_LAYER_STEP), true)?.unwrap();
                if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                    h.n_layer_dense_lead = v;
                }
            }
        }

        // ---- models/lfm2.cpp ----
        LlmArch::LFM2 => {
            h.n_shortconv_l_cache = get_key_u32(gguf, &k(LlmKv::SHORTCONV_L_CACHE), true)?.unwrap();
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            for il in 0..h.n_layer() as usize {
                h.is_recr_impl[il] = u32::from(h.n_head_kv(il) == 0);
            }

            h.n_layer_dense_lead = h.n_layer();

            // the decision-model head (88dcc460d, lfm2.cpp:29-35): trailing
            // blocks scored per question type — one token type each
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::DECISION_BLOCK_COUNT), false)? {
                h.n_layer_decision = v;
            }
            if h.n_layer_decision > 0 {
                if h.n_layer_decision >= h.n_layer() || h.causal_attn {
                    return Err("invalid decision head".into());
                }
                h.f_norm_eps =
                    get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
                // N_DECISION_TYPES = 3: choice, score, noul
                h.n_embd_out_impl = 3;
            }

            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)? {
                h.n_swa = v;
            }
            if h.n_swa > 0 {
                h.swa_type = LlamaSwaType::STANDARD;
                for il in 0..h.n_layer() as usize {
                    h.is_swa_impl[il] = u32::from(h.is_recr_impl[il] == 0);
                }
            }
        }

        // ---- models/bert.cpp ----
        LlmArch::BERT => {
            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
        }

        // ---- models/lfm2moe.cpp ----
        LlmArch::LFM2MOE => {
            h.n_shortconv_l_cache = get_key_u32(gguf, &k(LlmKv::SHORTCONV_L_CACHE), true)?.unwrap();
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            let vals = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();

            for il in 0..h.n_layer() as usize {
                h.is_recr_impl[il] = u32::from(h.n_head_kv(il) == 0);
            }
        }

        // ---- models/granite-hybrid.cpp ----
        LlmArch::GRANITE_HYBRID => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::LOGIT_SCALE), false)? {
                h.f_logit_scale = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::RESIDUAL_SCALE), false)? {
                h.f_residual_scale = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EMBEDDING_SCALE), false)? {
                h.f_embedding_scale = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_SCALE), false)? {
                h.f_attention_scale = v;
            }

            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.ssm_d_inner = get_key_u32(gguf, &k(LlmKv::SSM_INNER_SIZE), true)?.unwrap();
            h.ssm_d_state = get_key_u32(gguf, &k(LlmKv::SSM_STATE_SIZE), true)?.unwrap();
            h.ssm_dt_rank = get_key_u32(gguf, &k(LlmKv::SSM_TIME_STEP_RANK), true)?.unwrap();
            h.ssm_n_group = get_key_u32(gguf, &k(LlmKv::SSM_GROUP_COUNT), true)?.unwrap();

            // Granite uses rope_finetuned as a switch for rope, so default to true
            let rope_finetuned =
                get_key_bool(gguf, &k(LlmKv::ROPE_SCALING_FINETUNED), false)?.unwrap_or(true);
            h.rope_finetuned = rope_finetuned; // needed for round trip save
            h.rope_pattern
                .iter_mut()
                .for_each(|v| *v = u32::from(rope_finetuned));

            // A layer is recurrent IFF the n_head_kv value is set to 0
            for i in 0..h.n_layer() as usize {
                h.is_recr_impl[i] = u32::from(h.n_head_kv(i) == 0);
            }

            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            // For Granite MoE Shared
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_shexp = v;
            }
        }

        // ---- models/qwen35.cpp ----
        // qwen35 + clef share the backbone hparams (clef.cpp:8-9 calls
        // llama_model_qwen35::load_arch_hparams first)
        LlmArch::QWEN35 | LlmArch::CLEF => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.rope_sections =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), true)?
                    .unwrap();

            // Load linear attention (gated delta net) parameters
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.ssm_d_inner = get_key_u32(gguf, &k(LlmKv::SSM_INNER_SIZE), true)?.unwrap();
            h.ssm_d_state = get_key_u32(gguf, &k(LlmKv::SSM_STATE_SIZE), true)?.unwrap();
            h.ssm_dt_rank = get_key_u32(gguf, &k(LlmKv::SSM_TIME_STEP_RANK), true)?.unwrap();
            h.ssm_n_group = get_key_u32(gguf, &k(LlmKv::SSM_GROUP_COUNT), true)?.unwrap();

            // Mark recurrent layers (linear attention layers). MTP layers are
            // dense attention-only and must be flagged non-recurrent.
            let recr = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_RECURRENT_LAYERS),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = recr {
                h.is_recr_impl[..vals.len()].copy_from_slice(&vals);
            } else {
                let full_attn_interval =
                    get_key_u32(gguf, &k(LlmKv::FULL_ATTENTION_INTERVAL), false)?.unwrap_or(4);
                for i in 0..h.n_layer_all as usize {
                    // a 0 interval is UB in the C++ modulo; treat as "all dense"
                    h.is_recr_impl[i] = u32::from(
                        i < h.n_layer() as usize
                            && full_attn_interval != 0
                            && (i as u32 + 1) % full_attn_interval != 0,
                    );
                }
            }

            // the PLE conv history is a row of the recurrent cache, which
            // linear layers alone have (:140-145 — after the recurrent read)
            for i in 0..h.n_layer_all as usize {
                if h.is_ple(i) && !h.is_recr(i) {
                    return Err(format!("PLE layer {i} is not a linear attention layer"));
                }
            }

            // clef's own reads (a7b94df2c clef.cpp:10-25): the decision head
            // geometry — the joint count lands in n_layer_decision (the C
            // keeps it as the model member n_layer_joint), the routing count
            // and head count ride ModelTensors (the graniteswitch scalar
            // precedent)
            if arch == LlmArch::CLEF {
                let n_layer_routing =
                    get_key_u32(gguf, &k(LlmKv::DECISION_ROUTING_BLOCK_COUNT), true)?.unwrap();
                let n_layer_joint =
                    get_key_u32(gguf, &k(LlmKv::DECISION_BLOCK_COUNT), true)?.unwrap();
                let n_head_decision =
                    get_key_u32(gguf, &k(LlmKv::DECISION_HEAD_COUNT), true)?.unwrap();

                const MAX_LAYERS: u32 = LLAMA_MAX_LAYERS as u32;
                if n_head_decision == 0
                    || n_layer_routing > MAX_LAYERS
                    || n_layer_joint > MAX_LAYERS
                {
                    return Err("invalid size of the decision head".into());
                }

                // used by the head — LayerNorm eps (f_norm_eps)
                h.f_norm_eps =
                    get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();

                // the output is one score per token, see
                // llama_batch_ext_set_decision_order()
                h.n_embd_out_impl = 1;

                h.n_layer_decision = n_layer_joint;
                h.clef_n_layer_routing = n_layer_routing;
                h.clef_n_head_decision = n_head_decision;
            }
        }

        // ---- models/plamo3.cpp (arch batch 9) ----
        LlmArch::PLAMO3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            let found_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)?;
            if matches!(found_swa, Some(v) if v > 0) {
                h.n_swa = found_swa.unwrap();
                h.swa_type = LlamaSwaType::STANDARD;
                if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                    h.rope_freq_base_train_swa = v;
                }
                load_swa_pattern(gguf, arch, h, 8, false)?;
            } else {
                h.swa_type = LlamaSwaType::NONE;
            }
        }

        // ---- models/qwen3next.cpp (arch batch 9) ----
        LlmArch::QWEN3NEXT => {
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = ff {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_shexp = v;
            }
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            // Load linear attention (gated delta net) parameters
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.ssm_d_inner = get_key_u32(gguf, &k(LlmKv::SSM_INNER_SIZE), true)?.unwrap();
            h.ssm_d_state = get_key_u32(gguf, &k(LlmKv::SSM_STATE_SIZE), true)?.unwrap();
            h.ssm_dt_rank = get_key_u32(gguf, &k(LlmKv::SSM_TIME_STEP_RANK), true)?.unwrap();
            h.ssm_n_group = get_key_u32(gguf, &k(LlmKv::SSM_GROUP_COUNT), true)?.unwrap();

            // Mark recurrent layers (linear attention layers) — the same
            // interval-4 pattern default as qwen35 (qwen3next.cpp:17-23)
            let recr = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_RECURRENT_LAYERS),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = recr {
                h.is_recr_impl[..vals.len()].copy_from_slice(&vals);
            } else {
                let full_attn_interval =
                    get_key_u32(gguf, &k(LlmKv::FULL_ATTENTION_INTERVAL), false)?.unwrap_or(4);
                for i in 0..h.n_layer_all as usize {
                    h.is_recr_impl[i] = u32::from(
                        i < h.n_layer() as usize
                            && full_attn_interval != 0
                            && (i as u32 + 1) % full_attn_interval != 0,
                    );
                }
            }
        }

        // ---- models/kimi-linear.cpp (arch batch 9) ----
        LlmArch::KIMI_LINEAR => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.n_embd_head_k_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), true)?.unwrap();
            h.n_embd_head_v_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), true)?.unwrap();
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.n_embd_head_kda = get_key_u32(gguf, &k(LlmKv::KDA_HEAD_DIM), true)?.unwrap();

            // Mark KDA layers as recurrent using the n_head_kv pattern (like
            // Jamba): n_head_kv == 0 marks the KDA layers (kimi-linear.cpp:17-19)
            for i in 0..h.n_layer() as usize {
                h.is_recr_impl[i] = u32::from(h.n_head_kv(i) == 0);
            }

            // MoE parameters (kimi uses moe_intermediate_size = 1024)
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();
        }

        // ---- models/bailingmoe3.cpp (arch batch 9) ----
        LlmArch::BAILINGMOE3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.n_embd_head_k_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), true)?.unwrap();
            h.n_embd_head_v_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), true)?.unwrap();
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), false)? {
                h.n_lora_q = v;
            }
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.n_embd_head_kda = get_key_u32(gguf, &k(LlmKv::KDA_HEAD_DIM), true)?.unwrap();
            // GGML_ASSERT(hparams.kda_safe_gate) at bailingmoe3.cpp:32 — a
            // false key aborts the reference loader; mirror as a load error
            if !get_key_bool(gguf, &k(LlmKv::KDA_SAFE_GATE), false)?.unwrap_or(true) {
                return Err("bailingmoe3: kda.safe_gate must be true".to_string());
            }
            h.kda_gate_lower_bound =
                get_key_f32(gguf, &k(LlmKv::KDA_GATE_LOWER_BOUND), true)?.unwrap();
            // bailingmoe3.cpp:18 (def4d406a): VL files carry mrope sections;
            // `get_key_or_arr(..., false)` — text-only files keep [0;4]
            h.rope_sections = get_key_or_arr_rope_sections(
                gguf,
                &k(LlmKv::ROPE_DIMENSION_SECTIONS),
                false,
            )?
            .unwrap_or([0; 4]);
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_shexp = v;
            }
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            h.n_layer_dense_lead =
                get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();
            if let Some(vals) = get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_EXP),
                h.n_layer_all as usize,
                false,
            )? {
                h.swiglu_clamp_exp[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(vals) = get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_SHEXP),
                h.n_layer_all as usize,
                false,
            )? {
                h.swiglu_clamp_shexp[..vals.len()].copy_from_slice(&vals);
            }

            if h.n_ff_shexp == 0 {
                h.n_ff_shexp = h.n_ff_exp(0) * std::cmp::max(1, h.n_expert_shared);
            }

            if h.kda_gate_lower_bound >= 0.0 {
                return Err("bailingmoe3: kda.gate_lower_bound must be < 0".to_string());
            }

            for i in 0..h.n_layer() as usize {
                h.is_recr_impl[i] = u32::from(h.n_head_kv(i) == 0);
            }
        }

        // ---- models/glm5-next.cpp:7-59 (NEW arch, def4d406a) ----
        // GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a
        // k-pool DSA indexer, mHC residual streams, DeepSeek-style MoE.
        LlmArch::GLM5_NEXT => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), false)? {
                h.f_norm_eps = v;
            }
            h.n_embd_head_k_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), true)?.unwrap();
            h.n_embd_head_v_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), true)?.unwrap();
            h.n_lora_q = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), true)?.unwrap();
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.n_embd_head_kda = get_key_u32(gguf, &k(LlmKv::KDA_HEAD_DIM), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::KDA_GATE_LOWER_BOUND), false)? {
                h.kda_gate_lower_bound = v;
            }

            // the MLA cache holds the compressed latent (glm5-next.cpp:19)
            h.n_embd_head_v_full = h.n_lora_kv;

            // glm5-next.cpp:21-23 — the loop covers n_layer_all (the NextN
            // block is a DSA layer)
            for i in 0..h.n_layer_all as usize {
                h.is_recr_impl[i] = u32::from(h.n_head_kv(i) == 0);
            }

            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), false)? {
                h.expert_gating_func = v;
            }
            if h.expert_gating_func == crate::hparams::LlamaExpertGatingFuncType::NONE as u32 {
                h.expert_gating_func = crate::hparams::LlamaExpertGatingFuncType::SIGMOID as u32;
            }
            if let Some(vals) = get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_EXP),
                h.n_layer_all as usize,
                false,
            )? {
                h.swiglu_clamp_exp = vals.clone();
                h.swiglu_clamp_shexp = vals;
            }
            // glm5-next.cpp:35-37 — clamp_shexp falls back to clamp_exp
            if let Some(vals) = get_key_or_arr_f32(
                gguf,
                &k(LlmKv::SWIGLU_CLAMP_SHEXP),
                h.n_layer_all as usize,
                false,
            )? {
                h.swiglu_clamp_shexp = vals;
            }

            // DSA indexer with k-pool compression (glm5-next.cpp:40-47)
            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();
            h.indexer_kpool = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KPOOL), true)?.unwrap();
            if let Some(v) = get_key_bool(
                gguf,
                &k(LlmKv::ATTENTION_INDEXER_KPOOL_SELECT_TAIL),
                false,
            )? {
                h.indexer_kpool_select_tail = v;
            }
            // GGML_ASSERT(hparams.indexer_kpool > 1 && top_k % kpool == 0)
            if h.indexer_kpool <= 1 || h.indexer_top_k % h.indexer_kpool != 0 {
                return Err(format!(
                    "glm5-next: invalid indexer kpool {} / top_k {}",
                    h.indexer_kpool, h.indexer_top_k
                ));
            }
            // std::fill(is_indexer_full_impl, 1) then the optional per-layer
            // array over n_layer()
            h.is_indexer_full_impl = vec![1u32; LLAMA_MAX_LAYERS];
            if let Some(vals) = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_INDEXER_TYPES),
                h.n_layer() as usize,
                false,
            )? {
                h.is_indexer_full_impl[..vals.len()].copy_from_slice(&vals);
            }

            // mHC (glm5-next.cpp:50-53)
            h.dsv4_hc_mult = get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_COUNT), true)?.unwrap();
            h.dsv4_hc_sinkhorn_iters =
                get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_SINKHORN_ITERATIONS), true)?.unwrap();
            h.dsv4_hc_eps = get_key_f32(gguf, &k(LlmKv::HYPER_CONNECTION_EPSILON), true)?.unwrap();
            // GGML_ASSERT(dsv4_hc_mult == 4 && "mHC with hc_mult != 4 is not
            // supported") — mirrored as a load error (the bailingmoe3
            // safe_gate precedent)
            if h.dsv4_hc_mult != 4 {
                return Err("glm5-next: mHC with hc_mult != 4 is not supported".to_string());
            }
        }

        // ====================================================================
        // arch batch 11a (long-tail queue, first half)
        // ====================================================================

        // ---- models/apertus.cpp:3-15 (arch batch 11a) ----
        LlmArch::APERTUS => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();

            // per-layer xIELU constants (apertus.cpp:6-9)
            let n = h.n_layer() as usize;
            let an = get_key_or_arr_f32(gguf, &k(LlmKv::XIELU_ALPHA_N), n, true)?.unwrap();
            let ap = get_key_or_arr_f32(gguf, &k(LlmKv::XIELU_ALPHA_P), n, true)?.unwrap();
            let be = get_key_or_arr_f32(gguf, &k(LlmKv::XIELU_BETA), n, true)?.unwrap();
            let ep = get_key_or_arr_f32(gguf, &k(LlmKv::XIELU_EPS), n, true)?.unwrap();
            h.xielu_alpha_n[..an.len()].copy_from_slice(&an);
            h.xielu_alpha_p[..ap.len()].copy_from_slice(&ap);
            h.xielu_beta[..be.len()].copy_from_slice(&be);
            h.xielu_eps[..ep.len()].copy_from_slice(&ep);
        }

        // ---- models/grovemoe.cpp:3-14 (arch batch 11a) ----
        LlmArch::GROVEMOE => {
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_CHUNK_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_chexp = v;
            }
            h.expert_group_scale = get_key_f32(gguf, &k(LlmKv::EXPERT_GROUP_SCALE), true)?.unwrap();
            h.n_group_experts = get_key_u32(gguf, &k(LlmKv::EXPERTS_PER_GROUP), true)?.unwrap();
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/minimax-m3.cpp:13-39 (arch batch 11a) ----
        LlmArch::MINIMAX_M3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();
            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();
            h.indexer_block_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_BLOCK_SIZE), true)?.unwrap();
            h.indexer_local_blocks =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_LOCAL_BLOCKS), true)?.unwrap();

            // GGML_ASSERT(hparams.indexer_block_size > 0) (minimax-m3.cpp:33)
            if h.indexer_block_size == 0 {
                return Err("minimax-m3: indexer.block_size must be > 0".to_string());
            }
        }

        // ---- models/qwen35moe.cpp:4-34 (arch batch 11a; the qwen3next
        // hparams shape with the MoE FFN + rope sections) ----
        LlmArch::QWEN35MOE => {
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = ff {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_shexp = v;
            }
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.rope_sections =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), true)?
                    .unwrap();

            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.ssm_d_inner = get_key_u32(gguf, &k(LlmKv::SSM_INNER_SIZE), true)?.unwrap();
            h.ssm_d_state = get_key_u32(gguf, &k(LlmKv::SSM_STATE_SIZE), true)?.unwrap();
            h.ssm_dt_rank = get_key_u32(gguf, &k(LlmKv::SSM_TIME_STEP_RANK), true)?.unwrap();
            h.ssm_n_group = get_key_u32(gguf, &k(LlmKv::SSM_GROUP_COUNT), true)?.unwrap();

            // Mark recurrent layers; MTP layers are dense attention-only
            // (qwen35moe.cpp:20-26)
            let recr = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_RECURRENT_LAYERS),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = recr {
                h.is_recr_impl[..vals.len()].copy_from_slice(&vals);
            } else {
                let full_attn_interval =
                    get_key_u32(gguf, &k(LlmKv::FULL_ATTENTION_INTERVAL), false)?.unwrap_or(4);
                for i in 0..h.n_layer_all as usize {
                    h.is_recr_impl[i] = u32::from(
                        i < h.n_layer() as usize
                            && full_attn_interval != 0
                            && (i as u32 + 1) % full_attn_interval != 0,
                    );
                }
            }
        }

        // ---- models/kimi-k3.cpp:16-51 (arch batch 11a) ----
        LlmArch::KIMI_K3 => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.n_embd_head_k_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), true)?.unwrap();
            h.n_embd_head_v_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), false)? {
                h.n_lora_q = v;
            }
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.n_embd_head_kda = get_key_u32(gguf, &k(LlmKv::KDA_HEAD_DIM), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::KDA_GATE_LOWER_BOUND), false)? {
                h.kda_gate_lower_bound = v;
            }

            // the MLA cache holds the compressed latent (kimi-k3.cpp:27-28)
            h.n_embd_head_v_full = h.n_lora_kv;

            // n_head_kv == 0 marks a KDA layer, as in kimi-linear (:31-33)
            for i in 0..h.n_layer() as usize {
                h.is_recr_impl[i] = u32::from(h.n_head_kv(i) == 0);
            }

            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), false)? {
                h.n_layer_dense_lead = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_LATENT_LENGTH), false)? {
                h.n_expert_latent = v;
            }

            h.attn_res_block_size =
                get_key_u32(gguf, &k(LlmKv::ATTN_RES_BLOCK_SIZE), true)?.unwrap();
            h.situ_beta = get_key_f32(gguf, &k(LlmKv::ACTIVATION_SITU_BETA), true)?.unwrap();
            h.situ_linear_beta =
                get_key_f32(gguf, &k(LlmKv::ACTIVATION_SITU_LINEAR_BETA), true)?.unwrap();
        }

        // ---- models/dots3note.cpp:8-46 (arch batch 11a) ----
        LlmArch::DOTS3NOTE => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            // eps of the indexer k_norm layer norm (dots3note.cpp:10)
            h.f_norm_eps = 1e-6;

            h.n_expert_shared = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_COUNT), true)?.unwrap();
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                true,
            )?
            .unwrap();
            h.n_ff_exp_arr[..ff.len()].copy_from_slice(&ff);
            h.n_layer_dense_lead =
                get_key_u32(gguf, &k(LlmKv::LEADING_DENSE_BLOCK_COUNT), true)?.unwrap();
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::EXPERT_WEIGHTS_SCALE), false)? {
                h.expert_weights_scale = v;
            }
            if let Some(v) = get_key_bool(gguf, &k(LlmKv::EXPERT_WEIGHTS_NORM), false)? {
                h.expert_weights_norm = v;
            }
            h.expert_gating_func = get_key_u32(gguf, &k(LlmKv::EXPERT_GATING_FUNC), true)?.unwrap();

            // MLA geometry of the full-attention layers (:21-24)
            h.n_lora_q = get_key_u32(gguf, &k(LlmKv::ATTENTION_Q_LORA_RANK), true)?.unwrap();
            h.n_lora_kv = get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK), true)?.unwrap();
            h.n_embd_head_k_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA), true)?.unwrap();
            h.n_embd_head_v_mla_impl =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA), true)?.unwrap();

            // ... and of the sliding-window layers (:27-29)
            h.n_lora_kv_swa =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KV_LORA_RANK_SWA), true)?.unwrap();
            h.n_embd_head_k_mla_swa =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_KEY_LENGTH_MLA_SWA), true)?.unwrap();
            h.n_embd_head_v_mla_swa =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_VALUE_LENGTH_MLA_SWA), true)?.unwrap();

            // the iswa pair (:31-34)
            h.swa_type = LlamaSwaType::STANDARD;
            h.n_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), true)?.unwrap();
            h.rope_freq_base_train_swa =
                get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), true)?.unwrap();
            if let Some(vals) =
                get_arr_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW_PATTERN), true)?
            {
                h.is_swa_impl = vec![0u32; LLAMA_MAX_LAYERS];
                h.is_swa_impl[..vals.len()].copy_from_slice(&vals);
            }

            // DSA parameters (:37-40)
            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();
            if let Some(vals) = get_arr_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TYPES), true)? {
                h.is_indexer_full_impl = vec![0u32; LLAMA_MAX_LAYERS];
                h.is_indexer_full_impl[..vals.len()].copy_from_slice(&vals);
            }
        }

        // ---- models/qwen4exp.cpp:26-148 (arch batch 11a) ----
        LlmArch::QWEN4EXP => {
            let ff = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::EXPERT_FEED_FORWARD_LENGTH),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = ff {
                h.n_ff_exp_arr[..vals.len()].copy_from_slice(&vals);
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::EXPERT_SHARED_FEED_FORWARD_LENGTH), false)?
            {
                h.n_ff_shexp = v;
            }
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            h.rope_sections =
                get_key_or_arr_rope_sections(gguf, &k(LlmKv::ROPE_DIMENSION_SECTIONS), true)?
                    .unwrap();

            h.ssm_d_conv = get_key_u32(gguf, &k(LlmKv::SSM_CONV_KERNEL), true)?.unwrap();
            h.ssm_d_inner = get_key_u32(gguf, &k(LlmKv::SSM_INNER_SIZE), true)?.unwrap();
            h.ssm_d_state = get_key_u32(gguf, &k(LlmKv::SSM_STATE_SIZE), true)?.unwrap();
            h.ssm_dt_rank = get_key_u32(gguf, &k(LlmKv::SSM_TIME_STEP_RANK), true)?.unwrap();
            h.ssm_n_group = get_key_u32(gguf, &k(LlmKv::SSM_GROUP_COUNT), true)?.unwrap();
            for (kv, v) in [
                (LlmKv::SSM_CONV_KERNEL, h.ssm_d_conv),
                (LlmKv::SSM_INNER_SIZE, h.ssm_d_inner),
                (LlmKv::SSM_STATE_SIZE, h.ssm_d_state),
                (LlmKv::SSM_TIME_STEP_RANK, h.ssm_dt_rank),
                (LlmKv::SSM_GROUP_COUNT, h.ssm_n_group),
            ] {
                if v == 0 {
                    return Err(format!("qwen4exp: {} must be > 0", k(kv)));
                }
            }

            // HC; low_rank is qwen4exp-specific (:44-53)
            h.dsv4_hc_mult = get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_COUNT), true)?.unwrap();
            h.hc_low_rank = get_key_u32(gguf, &k(LlmKv::HYPER_CONNECTION_LOW_RANK), true)?.unwrap();
            if h.dsv4_hc_mult <= 1 {
                return Err(format!(
                    "qwen4exp: {} must be > 1",
                    k(LlmKv::HYPER_CONNECTION_COUNT)
                ));
            }
            h.n_embd_out_impl = h.dsv4_hc_mult * h.n_embd;

            h.indexer_n_head =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_HEAD_COUNT), true)?.unwrap();
            h.indexer_head_size =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_KEY_LENGTH), true)?.unwrap();
            h.indexer_top_k = get_key_u32(gguf, &k(LlmKv::ATTENTION_INDEXER_TOP_K), true)?.unwrap();
            for (kv, v) in [
                (LlmKv::ATTENTION_INDEXER_HEAD_COUNT, h.indexer_n_head),
                (LlmKv::ATTENTION_INDEXER_KEY_LENGTH, h.indexer_head_size),
                (LlmKv::ATTENTION_INDEXER_TOP_K, h.indexer_top_k),
            ] {
                if v == 0 {
                    return Err(format!("qwen4exp: {} must be > 0", k(kv)));
                }
                let _ = kv;
            }
            if let Some(vals) = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_COMPRESS_RATIOS),
                h.n_layer_all as usize,
                false,
            )? {
                h.dsv4_compress_ratios[..vals.len()].copy_from_slice(&vals);
            }

            // QSA pools the indexer keys of blocks of compress_ratio cells,
            // one block size for the whole model (a7b94df2c
            // qwen4exp.cpp:64-83, commits 66e0c17ee/c061df198)
            h.indexer_kpool = 0;
            for il in 0..h.n_layer_all as usize {
                let r = h.dsv4_compress_ratios[il];
                if r == 0 {
                    continue;
                }
                if h.indexer_kpool != 0 && r != h.indexer_kpool {
                    return Err(format!(
                        "QSA layers must share one compress ratio, got {} and {}",
                        h.indexer_kpool, r
                    ));
                }
                h.indexer_kpool = r;
            }
            if h.indexer_kpool == 1
                || (h.indexer_kpool > 0 && h.indexer_top_k % h.indexer_kpool != 0)
            {
                return Err(format!(
                    "QSA needs a compress ratio above 1 that divides the budget, got {} and {}",
                    h.indexer_kpool, h.indexer_top_k
                ));
            }
            // the reference groups the visible tokens in cache order and
            // always keeps the tail
            h.indexer_kpool_row = 2; // raw key | pooled key
            h.indexer_kpool_by_order = true;
            h.indexer_kpool_select_tail = true;

            // PLE n-gram hash embeddings (qwen4exp.cpp:67-128); if the key
            // group is absent every field stays zero
            h.is_ple_impl = vec![false; h.n_layer_all as usize];
            h.ple_n_heads = 0;
            if let Some(n_ple) = get_arr_n(gguf, &k(LlmKv::PLE_LAYERS), false)? {
                if n_ple > 0 {
                    // hparams holds one set of hash constants, so several
                    // PLE modules cannot be represented (:76-80)
                    if n_ple != 1 {
                        return Err(format!(
                            "qwen4exp: {} lists {} layers, but only one PLE layer is supported",
                            k(LlmKv::PLE_LAYERS),
                            n_ple
                        ));
                    }
                    let ple_layers = gguf
                        .find_key(&k(LlmKv::PLE_LAYERS))
                        .and_then(|v| v.as_array().map(|a| a.1.clone()))
                        .unwrap_or_default();
                    for v in &ple_layers {
                        let il = match v {
                            Value::U32(x) => *x as usize,
                            Value::I32(x) => *x as usize,
                            Value::I64(x) => *x as usize,
                            Value::U64(x) => *x as usize,
                            _ => {
                                return Err(format!(
                                    "{} entries must be integers",
                                    k(LlmKv::PLE_LAYERS)
                                ))
                            }
                        };
                        if il >= h.n_layer_all as usize {
                            return Err(format!("PLE layer {il} is out of range"));
                        }
                        h.is_ple_impl[il] = true;
                    }

                    h.ple_ngram_size =
                        get_key_u32(gguf, &k(LlmKv::PLE_NGRAM_SIZE), true)?.unwrap();
                    h.ple_heads_per_ngram =
                        get_key_u32(gguf, &k(LlmKv::PLE_HEADS_PER_NGRAM), true)?.unwrap();
                    h.ple_conv_kernel =
                        get_key_u32(gguf, &k(LlmKv::PLE_CONV_KERNEL), true)?.unwrap();
                    h.ple_eos_token_id =
                        get_key_u32(gguf, &k(LlmKv::PLE_EOS_TOKEN_ID), true)?.unwrap();
                    // optional: files written before this key fall back to the
                    // EOS token (:92-93)
                    h.ple_image_token_id =
                        get_key_u32(gguf, &k(LlmKv::PLE_IMAGE_TOKEN_ID), false)?
                            .unwrap_or(h.ple_eos_token_id);
                    h.n_embd_per_layer = get_key_u32(
                        gguf,
                        &k(LlmKv::EMBEDDING_LENGTH_PER_LAYER),
                        true,
                    )?
                    .unwrap();
                    for (kv, v) in [
                        (LlmKv::PLE_CONV_KERNEL, h.ple_conv_kernel),
                        (LlmKv::EMBEDDING_LENGTH_PER_LAYER, h.n_embd_per_layer),
                    ] {
                        if v == 0 {
                            return Err(format!(
                                "qwen4exp: {} must be greater than zero, got {v}",
                                k(kv)
                            ));
                        }
                    }

                    h.ple_n_heads = (h.ple_ngram_size - 1) * h.ple_heads_per_ngram;
                    h.ple_head_dim = h.n_embd_per_layer;
                    if !(2..=crate::hparams::LLAMA_MAX_PLE_NGRAM as u32)
                        .contains(&h.ple_ngram_size)
                    {
                        return Err(format!(
                            "PLE n-gram size {} is out of range",
                            h.ple_ngram_size
                        ));
                    }
                    if h.ple_n_heads == 0
                        || h.ple_n_heads > crate::hparams::LLAMA_MAX_PLE_HEADS as u32
                    {
                        return Err(format!(
                            "PLE head count {} is out of range",
                            h.ple_n_heads
                        ));
                    }

                    // qwen4exp_require_arr_len (:20-27)
                    let require_arr_len = |kv: LlmKv, n_min: u32| -> Result<(), String> {
                        let n_arr =
                            get_arr_n(gguf, &k(kv), true)?.ok_or_else(|| not_found(&k(kv)))?;
                        if n_arr < n_min {
                            return Err(format!(
                                "{} has {n_arr} entries, but at least {n_min} are required",
                                k(kv)
                            ));
                        }
                        Ok(())
                    };
                    require_arr_len(LlmKv::PLE_LAYER_MULTIPLIERS, h.ple_ngram_size)?;
                    require_arr_len(LlmKv::PLE_HEAD_OFFSETS, h.ple_n_heads)?;
                    require_arr_len(LlmKv::PLE_HEAD_VOCAB_SIZES, h.ple_n_heads)?;

                    let arr_u64 = |kv: LlmKv| -> Vec<u64> {
                        gguf.find_key(&k(kv))
                            .and_then(|v| v.as_array().map(|a| a.1.clone()))
                            .unwrap_or_default()
                            .into_iter()
                            .map(|v| match v {
                                Value::U64(x) => x,
                                Value::I64(x) => x as u64,
                                Value::U32(x) => x as u64,
                                Value::I32(x) => x as u64,
                                _ => 0,
                            })
                            .collect()
                    };
                    let mults = arr_u64(LlmKv::PLE_LAYER_MULTIPLIERS);
                    let n_m = mults.len().min(h.ple_layer_multipliers.len());
                    h.ple_layer_multipliers[..n_m].copy_from_slice(&mults[..n_m]);

                    // the file stores the head ranges as uint64, so read at
                    // that width and narrow to the int32 the gather uses
                    // (:113-127)
                    let head_offsets = arr_u64(LlmKv::PLE_HEAD_OFFSETS);
                    let head_vocab_sizes = arr_u64(LlmKv::PLE_HEAD_VOCAB_SIZES);
                    for hh in 0..h.ple_n_heads as usize {
                        let (off, voc) = (head_offsets[hh], head_vocab_sizes[hh]);
                        if voc == 0
                            || off > i32::MAX as u64
                            || voc > i32::MAX as u64
                            || off + voc > i32::MAX as u64
                        {
                            return Err(format!(
                                "PLE head {hh} range does not fit the int32 row index"
                            ));
                        }
                        h.ple_head_offsets[hh] = off as u32;
                        h.ple_head_vocab_sizes[hh] = voc as u32;
                    }
                }
            }



            // linear attention everywhere except every
            // full_attention_interval-th layer (:127-135)
            let recr = get_key_or_arr_u32(
                gguf,
                &k(LlmKv::ATTENTION_RECURRENT_LAYERS),
                h.n_layer_all as usize,
                false,
            )?;
            if let Some(vals) = recr {
                h.is_recr_impl[..vals.len()].copy_from_slice(&vals);
            } else {
                let full_attn_interval =
                    get_key_u32(gguf, &k(LlmKv::FULL_ATTENTION_INTERVAL), false)?.unwrap_or(4);
                if full_attn_interval == 0 {
                    return Err(format!(
                        "qwen4exp: {} must be > 0",
                        k(LlmKv::FULL_ATTENTION_INTERVAL)
                    ));
                }
                for i in 0..h.n_layer_all as usize {
                    h.is_recr_impl[i] = u32::from(
                        i < h.n_layer() as usize && (i as u32 + 1) % full_attn_interval != 0,
                    );
                }
            }
        }

        // --------------------------------------------------------------------
        // arch batch 14 (2026-10) — the RWKV family + gemma3n
        // --------------------------------------------------------------------

        // ---- models/rwkv6.cpp:3-24 / rwkv6qwen2.cpp:3-24 ----
        LlmArch::RWKV6 | LlmArch::RWKV6QWEN2 => {
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), false)? {
                h.f_norm_eps = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), false)? {
                h.f_norm_rms_eps = v;
            }
            h.wkv_head_size = get_key_u32(gguf, &k(LlmKv::WKV_HEAD_SIZE), true)?.unwrap();
            h.time_mix_extra_dim = get_key_u32(gguf, &k(LlmKv::TIME_MIX_EXTRA_DIM), true)?.unwrap();
            h.time_decay_extra_dim =
                get_key_u32(gguf, &k(LlmKv::TIME_DECAY_EXTRA_DIM), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::RESCALE_EVERY_N_LAYERS), false)? {
                h.rescale_every_n_layers = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::TOKEN_SHIFT_COUNT), false)? {
                h.token_shift_count = v;
            }
            // the LLM_TYPE heuristics of rwkv6.cpp:12-23 only feed the model
            // name — not observable through this port's hparams
        }

        // ---- models/rwkv7.cpp:3-44 / arwkv7.cpp:3-44 ----
        LlmArch::RWKV7 | LlmArch::ARWKV7 => {
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), false)? {
                h.f_norm_eps = v;
            }
            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), false)? {
                h.f_norm_rms_eps = v;
            }
            h.wkv_head_size = get_key_u32(gguf, &k(LlmKv::WKV_HEAD_SIZE), true)?.unwrap();
            h.n_lora_decay =
                get_key_u32(gguf, &k(LlmKv::ATTENTION_DECAY_LORA_RANK), true)?.unwrap();
            h.n_lora_iclr = get_key_u32(gguf, &k(LlmKv::ATTENTION_ICLR_LORA_RANK), true)?.unwrap();
            h.n_lora_value_res_mix = get_key_u32(
                gguf,
                &k(LlmKv::ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK),
                true,
            )?
            .unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_GATE_LORA_RANK), false)? {
                h.n_lora_gate = v;
            }
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::TOKEN_SHIFT_COUNT), false)? {
                h.token_shift_count = v;
            }
        }

        // ---- models/gemma3n.cpp:3-19 ----
        LlmArch::GEMMA3N => {
            h.swa_type = LlamaSwaType::STANDARD;
            load_swa_pattern(gguf, arch, h, 5, false)?;

            h.n_layer_kv_from_start = 20;
            h.f_attention_scale = 1.0;

            if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                h.rope_freq_base_train_swa = v;
            }
            h.n_swa = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), true)?.unwrap();
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
            // the altup/laurel constants are the llama-hparams.h defaults
            // (n_altup 4 / i_altup_act 0 / laurel_rank 64 / n_embd_altup 256
            // — never read from the GGUF in this revision)
        }

        // ---- models/jina-bert-v2.cpp:3-12 ----
        LlmArch::JINA_BERT_V2 => {
            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
            // jina-bert-v2.cpp:5 — the ALiBi bias; the generic post-step of
            // load_hparams flips use_alibi on it (llama-model.cpp:1419-1421)
            h.f_max_alibi_bias = 8.0;
        }

        // ---- models/jina-bert-v3.cpp:3-11 ----
        LlmArch::JINA_BERT_V3 => {
            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
        }

        // ---- models/nomic-bert.cpp:3-11 ----
        LlmArch::NOMIC_BERT => {
            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
        }

        // ---- models/nomic-bert-moe.cpp:3-12 ----
        LlmArch::NOMIC_BERT_MOE => {
            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::MOE_EVERY_N_LAYERS), false)? {
                h.moe_every_n_layers = v;
            }
        }

        // ---- models/neo-bert.cpp:3-9 ----
        LlmArch::NEO_BERT => {
            h.f_norm_rms_eps =
                get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_RMS_EPS), true)?.unwrap();
        }

        // ---- models/modern-bert.cpp:3-32 ----
        LlmArch::MODERN_BERT => {
            // the sliding window is optional; present + > 0 → SYMMETRIC swa
            // with the pattern of 3 (dense first) + the optional swa rope base
            // (modern-bert.cpp:4-11)
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::ATTENTION_SLIDING_WINDOW), false)? {
                h.n_swa = v;
            }
            if h.n_swa > 0 {
                h.swa_type = LlamaSwaType::SYMMETRIC;
                if let Some(v) = get_key_f32(gguf, &k(LlmKv::ROPE_FREQ_BASE_SWA), false)? {
                    h.rope_freq_base_train_swa = v;
                }
                load_swa_pattern(gguf, arch, h, 3, true)?;
            } else {
                h.swa_type = LlamaSwaType::NONE;
            }

            h.f_norm_eps = get_key_f32(gguf, &k(LlmKv::ATTENTION_LAYERNORM_EPS), true)?.unwrap();

            // Some ModernBert derivatives (e.g. IBM Granite Embedding 97m R2)
            // use SiLU/SwiGLU in the FFN instead of the default GELU/GeGLU
            // (modern-bert.cpp:15-21): llm_ffn_op_type_from_string over the
            // LLM_FFN_OP_TYPES_FROM_STRING table (llama-model.cpp:1042-1050)
            // with GEGLU as the fallback
            h.llm_ffn_op = LlmFfnOpType::GEGLU;
            if let Some(act) = get_key_string(gguf, &k(LlmKv::HIDDEN_ACT), false)? {
                h.llm_ffn_op = llm_ffn_op_type_from_string(&act, LlmFfnOpType::GEGLU);
            }

            // GGUFs without a classifier pooling type use mean
            // (gte-reranker-modernbert-base) — modern-bert.cpp:23-26
            // (def4d406a)
            if h.pooling_type_cls == LlamaPoolingType::UNSPECIFIED {
                h.pooling_type_cls = LlamaPoolingType::MEAN;
            }

            // GGUFs without a classifier activation use gelu, the
            // transformers default (37ac63456, modern-bert.cpp:31-35)
            if get_key_string(gguf, &k(LlmKv::CLASSIFIER_ACTIVATION), false)?.is_none() {
                h.act_cls = ggml::ops::GGML_UNARY_OP_GELU_ERF;
            }

            // the decision-model head (a7b94df2c modern-bert.cpp:31-41,
            // commit a4cb4c61f): trailing blocks scored per question type
            if let Some(v) = get_key_u32(gguf, &k(LlmKv::DECISION_BLOCK_COUNT), false)? {
                h.n_layer_decision = v;
            }
            if h.n_layer_decision > 0 {
                if h.n_layer_decision >= h.n_layer() {
                    return Err("invalid number of decision blocks".into());
                }
                // the head blocks always use full attention
                let n_layer = h.n_layer();
                for il in n_layer - h.n_layer_decision..n_layer {
                    h.is_swa_impl[il as usize] = 0;
                }
                // the output is one score per question type
                // (N_DECISION_TYPES = 3: choice, score, noul)
                h.n_embd_out_impl = 3;
            }
        }

        // everything else: generic path + defaults (see arch_hparams_support)
        _ => {}
    }

    Ok(())
}

/// `llm_ffn_op_type_from_string` (llama-model.cpp:1052-1058) over the
/// `LLM_FFN_OP_TYPES_FROM_STRING` table (:1042-1050) — unrecognized values
/// fall back to the caller's fallback (GeGLU for the ModernBert-style archs).
fn llm_ffn_op_type_from_string(name: &str, fallback: LlmFfnOpType) -> LlmFfnOpType {
    use LlmFfnOpType as Op;
    match name {
        "gelu" => Op::GEGLU,
        "geglu" => Op::GEGLU,
        "silu" => Op::SWIGLU,
        "swish" => Op::SWIGLU,
        "swiglu" => Op::SWIGLU,
        "relu" => Op::RELU,
        "reglu" => Op::REGLU,
        _ => fallback,
    }
}

// ---------------------------------------------------------------------------
// llama_model_rope_type — complete arch switch (llama-model.cpp)
// ---------------------------------------------------------------------------

/// `llama_model_rope_type(model)` — note the dynamic cases (dflash / glm4 /
// hunyuan-vl) consult hparams. Exhaustive over LlmArch like the C++ switch.
pub fn llama_model_rope_type(arch: LlmArch, hparams: &LlamaHparams) -> LlamaRopeType {
    use LlamaRopeType::*;
    match arch {
        // these models do not use RoPE
        LlmArch::CLIP
        | LlmArch::GPT2
        | LlmArch::GPTJ
        | LlmArch::MPT
        | LlmArch::REFACT
        | LlmArch::BLOOM
        | LlmArch::MAMBA
        | LlmArch::MAMBA2
        | LlmArch::JAMBA
        | LlmArch::JINA_BERT_V2
        | LlmArch::T5
        | LlmArch::T5ENCODER
        | LlmArch::JAIS
        | LlmArch::RWKV6
        | LlmArch::RWKV6QWEN2
        | LlmArch::RWKV7
        | LlmArch::ARWKV7
        | LlmArch::WAVTOKENIZER_DEC
        | LlmArch::NEMOTRON_H
        | LlmArch::NEMOTRON_H_MOE
        | LlmArch::KIMI_LINEAR
        | LlmArch::KIMI_K3
        | LlmArch::GLM5_NEXT => NONE,

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        LlmArch::LLAMA
        | LlmArch::LLADA
        | LlmArch::LLAMA4
        | LlmArch::DECI
        | LlmArch::BAICHUAN
        | LlmArch::STARCODER
        | LlmArch::INTERNLM2
        | LlmArch::MINICPM
        | LlmArch::XVERSE
        | LlmArch::COMMAND_R
        | LlmArch::COHERE2
        | LlmArch::COHERE2MOE
        | LlmArch::OLMO
        | LlmArch::ARCTIC
        | LlmArch::DEEPSEEK
        | LlmArch::DEEPSEEK2
        | LlmArch::DEEPSEEK2OCR
        | LlmArch::DEEPSEEK32
        | LlmArch::DEEPSEEK4
        | LlmArch::MUSE_GLIMMER
        | LlmArch::PLM
        | LlmArch::CHATGLM
        | LlmArch::GRANITE
        | LlmArch::GRANITE_MOE
        | LlmArch::GRANITE_HYBRID
        | LlmArch::GRANITE_SWITCH
        | LlmArch::GRANITE_SWA
        | LlmArch::CHAMELEON
        | LlmArch::BAILINGMOE
        | LlmArch::NEO_BERT
        | LlmArch::SMOLLM3
        | LlmArch::ARCEE
        | LlmArch::ERNIE4_5
        | LlmArch::ERNIE4_5_MOE
        | LlmArch::MISTRAL3
        | LlmArch::EAGLE3
        | LlmArch::MISTRAL4
        | LlmArch::LLAMA_EMBED
        | LlmArch::MAINCODER
        | LlmArch::GLM_DSA
        | LlmArch::DOTS3NOTE
        | LlmArch::NANBEIGE
        | LlmArch::POCKETTTS
        // HY_V4 rotates consecutive pairs, matching the reference implementation
        | LlmArch::HY_V4 => NORM,

        // the pairs of head values are offset by n_rot/2
        LlmArch::FALCON
        | LlmArch::FALCON_H1
        | LlmArch::GROK
        | LlmArch::DBRX
        | LlmArch::BERT
        | LlmArch::JINA_BERT_V3
        | LlmArch::MODERN_BERT
        | LlmArch::NOMIC_BERT
        | LlmArch::NOMIC_BERT_MOE
        | LlmArch::EUROBERT
        | LlmArch::STABLELM
        | LlmArch::BITNET
        | LlmArch::QWEN
        | LlmArch::QWEN2
        | LlmArch::DREAM
        | LlmArch::QWEN2MOE
        | LlmArch::QWEN3
        | LlmArch::QWEN3MOE
        | LlmArch::LLADA_MOE
        | LlmArch::RND1
        | LlmArch::OLMO2
        | LlmArch::OLMOE
        | LlmArch::PHI2
        | LlmArch::PHI3
        | LlmArch::PHIMOE
        | LlmArch::PLAMO
        | LlmArch::PLAMO2
        | LlmArch::PLAMO3
        | LlmArch::GEMMA
        | LlmArch::GEMMA2
        | LlmArch::GEMMA3
        | LlmArch::GEMMA3N
        | LlmArch::GEMMA4
        | LlmArch::GEMMA4_ASSISTANT
        | LlmArch::GEMMA_EMBEDDING
        | LlmArch::GEMMA_EMBEDDING2
        | LlmArch::STARCODER2
        | LlmArch::OPENELM
        | LlmArch::GPTNEOX
        | LlmArch::CODESHELL
        | LlmArch::ORION
        | LlmArch::NEMOTRON
        | LlmArch::EXAONE
        | LlmArch::EXAONE4
        | LlmArch::EXAONE_MOE
        | LlmArch::MINICPM3
        | LlmArch::BAILINGMOE2
        | LlmArch::DOTS1
        | LlmArch::HUNYUAN_MOE
        | LlmArch::JAIS2
        | LlmArch::OPENAI_MOE
        | LlmArch::HUNYUAN_DENSE
        | LlmArch::HY_V3
        | LlmArch::LFM2
        | LlmArch::LFM2MOE
        | LlmArch::SMALLTHINKER
        | LlmArch::SEED_OSS
        | LlmArch::GROVEMOE
        | LlmArch::APERTUS
        | LlmArch::MINIMAX_01
        | LlmArch::MINIMAX_M2
        | LlmArch::MINIMAX_M3
        | LlmArch::COGVLM
        | LlmArch::PANGU_EMBED
        | LlmArch::AFMOE
        | LlmArch::LAGUNA
        | LlmArch::QWEN3NEXT
        | LlmArch::MIMO2
        | LlmArch::STEP35
        | LlmArch::SPARK2_5
        | LlmArch::TALKIE
        | LlmArch::K2_HORIZON
        | LlmArch::MELLUM
        | LlmArch::MAPLE
        | LlmArch::HRM_TEXT => NEOX,

        LlmArch::DFLASH => {
            // drafts for M-RoPE targets carry rope sections and follow the
            // target's temporal dim
            let s = hparams.rope_sections;
            if s[0] != 0 || s[1] != 0 || s[2] != 0 || s[3] != 0 {
                return MROPE;
            }
            // DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy
            // DFlash backbones are NeoX
            if hparams.dsv4_hc_mult > 0 {
                NORM
            } else {
                NEOX
            }
        }

        LlmArch::QWEN2VL | LlmArch::PADDLEOCR => MROPE,
        LlmArch::QWEN3VL
        | LlmArch::QWEN3VLMOE
        | LlmArch::QWEN35
        | LlmArch::QWEN35MOE
        | LlmArch::CLEF
        | LlmArch::QWEN4EXP
        | LlmArch::QWEN3TTS => IMROPE,

        LlmArch::GLM4 => {
            if hparams.use_mrope() {
                MROPE
            } else {
                NORM
            }
        }
        // VL files carry mrope sections; text-only files keep NORM rope
        // (llama-model.cpp:3092-3095, def4d406a)
        LlmArch::BAILINGMOE3 => {
            if hparams.use_mrope() {
                MROPE
            } else {
                NORM
            }
        }
        LlmArch::GLM4_MOE => {
            if hparams.use_mrope() {
                MROPE
            } else {
                NEOX
            }
        }
        LlmArch::HUNYUAN_VL => {
            if hparams.use_mrope() {
                MROPE
            } else {
                NEOX
            }
        }

        // all model arches should be listed explicitly here
        LlmArch::UNKNOWN => panic!("unknown architecture"), // GGML_ABORT
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::tensor_name_suffix;
    use crate::arch::LlmTensor;
    use std::path::Path;

    // local test models (present on this machine; tests skip if absent)
    const QWEN25: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const GEMMA4_12B: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
    const PHI4_MINI: &str = "/home/jeffrey/.lmstudio/models/unsloth/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q6_K.gguf";
    const GPT_OSS_20B: &str =
        "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";
    const ERNIE45: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/ERNIE-4.5-21B-A3B-PT-GGUF/ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf";
    const LFM2: &str =
        "/home/jeffrey/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf";
    const BGE_M3: &str = "/home/jeffrey/localai/models/bge-m3-Q8_0.gguf";
    const QWEN3_EMB: &str = "/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf";
    const SEED_OSS: &str = "/home/jeffrey/.lmstudio/models/unsloth/Seed-OSS-36B-Instruct-GGUF/Seed-OSS-36B-Instruct-Q4_K_M.gguf";
    const GRANITE_H_TINY: &str = "/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf";
    const QWEN36_27B: &str = "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf";
    const QWEN38_27B_MTP: &str =
        "/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf";

    fn open(path: &str) -> Option<Gguf> {
        if !Path::new(path).exists() {
            return None; // skip silently when the model is not installed
        }
        Some(Gguf::open(path).expect("gguf open"))
    }

    fn load(path: &str) -> (LlmArch, LlamaHparams) {
        let gguf = open(path).expect("model present");
        load_hparams(&gguf).expect("load_hparams")
    }

    /// Real-model hparams tests: skip (not fail) when the local model file
    /// is not installed — the suite must stay green on clean checkouts.
    macro_rules! skip_unless_present {
        ($p:expr) => {
            if Path::new($p).exists() == false {
                eprintln!("skipping: model {} not present", $p);
                return;
            }
        };
    }

    /// minimal hand-rolled GGUF writer for error-path tests
    fn write_minimal_gguf(kvs: &[(&str, Value)]) -> std::path::PathBuf {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        for (k, v) in kvs {
            put_str(&mut b, k);
            let ty: u32 = match v {
                Value::String(_) => 8,
                Value::U32(_) => 4,
                Value::F32(_) => 6,
                Value::Bool(_) => 7,
                _ => unreachable!(),
            };
            b.extend_from_slice(&ty.to_le_bytes());
            match v {
                Value::String(s) => put_str(&mut b, s),
                Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
                Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
                Value::Bool(x) => b.extend_from_slice(&(*x as u8).to_le_bytes()),
                _ => unreachable!(),
            };
        }
        let dir = std::env::temp_dir().join("llama_rust_meta_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("minimal-{}.gguf", kvs.len()));
        std::fs::write(&path, b).unwrap();
        path
    }

    #[test]
    fn qwen25_exact_values() {
        // Reference: qwen2.5-0.5b-instruct Q4_K_M GGUF + `llama-cli -m ... -p hi -n 1`
        // load log (bd4f514db1): arch = qwen2, n_ctx_train = 32768,
        // n_embd = 896, n_head = 14, n_head_kv = 2, n_ff = 4864,
        // n_embd_head_k = 64, rope.freq_base = 1000000.0
        let Some(gguf) = open(QWEN25) else { return };
        let (arch, h) = load_hparams(&gguf).expect("load_hparams");
        assert_eq!(arch, LlmArch::QWEN2);
        assert_eq!(h.n_layer(), 24);
        assert_eq!(h.n_layer_all, 24);
        assert_eq!(h.n_embd, 896);
        assert_eq!(h.n_head(0), 14);
        assert_eq!(h.n_head_kv(0), 2);
        assert_eq!(h.n_gqa(0), 7);
        assert_eq!(h.n_ff(0), 4864);
        assert_eq!(h.n_embd_head_k(0), 64);
        assert_eq!(h.n_embd_head_v(0), 64);
        assert_eq!(h.n_rot(0), 64);
        assert_eq!(h.n_embd_k_gqa(0), 128);
        assert_eq!(h.n_ctx_train, 32768);
        // f32 1e-6 round-trips to 9.999999974752427e-07
        assert!((h.f_norm_rms_eps - 1e-6).abs() < 1e-12);
        // qwen2.rope.freq_base = 1000000.0 (exact f32)
        assert_eq!(h.rope_freq_base_train, 1000000.0);
        // no rope.scaling.* keys: default "linear" + scale 1.0
        assert_eq!(h.rope_scaling_type_train, LlamaRopeScalingType::LINEAR);
        assert_eq!(h.rope_freq_scale_train, 1.0);
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
        // no SWA, no experts
        assert_eq!(h.swa_type, LlamaSwaType::NONE);
        assert!(!h.is_swa_any());
        assert_eq!(h.n_expert, 0);
        assert_eq!(h.n_embd_out(), h.n_embd);
        assert!(h.causal_attn);
        assert!(!h.rope_finetuned);
    }

    #[test]
    fn qwen25_tensor_names_against_gguf() {
        let Some(gguf) = open(QWEN25) else { return };
        // spot-check tensor_name() against the real tensor table
        for (t, suffix, blk, expect) in [
            (LlmTensor::TOKEN_EMBD, "weight", -1, "token_embd.weight"),
            (LlmTensor::ATTN_NORM, "weight", 0, "blk.0.attn_norm.weight"),
            (
                LlmTensor::ATTN_NORM,
                "weight",
                23,
                "blk.23.attn_norm.weight",
            ),
            (LlmTensor::OUTPUT_NORM, "weight", -1, "output_norm.weight"),
            (LlmTensor::OUTPUT, "weight", -1, "output.weight"),
            (LlmTensor::FFN_GATE, "weight", 0, "blk.0.ffn_gate.weight"),
            (LlmTensor::FFN_DOWN, "weight", 0, "blk.0.ffn_down.weight"),
            (LlmTensor::FFN_UP, "weight", 0, "blk.0.ffn_up.weight"),
            (LlmTensor::ATTN_Q, "weight", 0, "blk.0.attn_q.weight"),
            (LlmTensor::ATTN_K, "weight", 0, "blk.0.attn_k.weight"),
            (LlmTensor::ATTN_V, "weight", 0, "blk.0.attn_v.weight"),
            (LlmTensor::ATTN_OUT, "weight", 0, "blk.0.attn_output.weight"),
        ] {
            let name = tensor_name_suffix(t, suffix, blk, -1);
            assert_eq!(name, expect);
            assert!(
                gguf.find_tensor(&name).is_some(),
                "tensor {name} missing in qwen2.5 gguf"
            );
        }
    }

    #[test]
    fn gemma4_12b_values() {
        // Reference: llama-cli load log for gemma-4-12B-it-QAT-Q4_0 (see
        // verify_against_reference_llama_cli in this module for the capture):
        // arch = gemma4, n_embd = 3840, n_layer = 48, n_head = 16,
        // per-layer head_count_kv = [8,8,8,8,8,1,...] (i32 array!),
        // key_length = 512, key_length_swa = 256, sliding_window = 1024,
        // sliding_window_pattern = bool[48] with every 6th layer dense.
        let Some(_gguf) = open(GEMMA4_12B) else {
            return;
        };
        let (arch, h) = load(GEMMA4_12B);
        assert_eq!(arch, LlmArch::GEMMA4);
        assert_eq!(h.n_embd, 3840);
        assert_eq!(h.n_layer(), 48);
        assert_eq!(h.n_head(0), 16);
        assert_eq!(h.n_head_kv(0), 8);
        // per-layer kv heads: layers 5, 11, ... drop to 1 (per-layer i32 array)
        assert_eq!(h.n_head_kv(5), 1);
        assert_eq!(h.n_head_kv(6), 8);
        assert_eq!(h.n_embd_head_k_full, 512); // attention.key_length override
        assert_eq!(h.n_embd_head_v_full, 512);
        assert_eq!(h.n_embd_head_k_swa, 256);
        assert_eq!(h.n_embd_head_v_swa, 256);
        assert_eq!(h.n_rot_full, 512); // rope.dimension_count
        assert_eq!(h.n_rot_swa, 256); // rope.dimension_count_swa
        assert_eq!(h.n_swa, 1024);
        assert_eq!(h.swa_type, LlamaSwaType::STANDARD);
        assert!(h.is_swa_any());
        // bool-array sliding_window_pattern: first 5 layers SWA, 6th dense
        assert!(h.is_swa(0));
        assert!(!h.is_swa(5));
        assert!(h.is_swa(6));
        assert_eq!(h.non_causal_type, LlamaNonCausalType::SWA_ONLY);
        // shared_kv_layers = 0 -> all 48 layers carry KV
        assert_eq!(h.n_layer_kv_from_start, 48);
        assert!(h.has_kv(0));
        assert!(h.has_kv(47));
        assert_eq!(h.f_attention_scale, 1.0);
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
        assert_eq!(h.n_embd_per_layer, 0); // embedding_length_per_layer_input = 0
        assert_eq!(h.rope_freq_base_train, 1000000.0);
        assert_eq!(h.rope_freq_base_train_swa, 10000.0);
        assert_eq!(h.f_final_logit_softcapping, 30.0);
        // full head is 512-wide; layer 5 is dense with a single kv head
        assert_eq!(h.n_embd_k_gqa(5), 512 * 1);
        // layer 0 is SWA (256-wide head, 8 kv heads)
        assert_eq!(h.n_embd_k_gqa(0), 256 * 8);
    }

    #[test]
    fn phi4_mini_values() {
        skip_unless_present!(PHI4_MINI);
        // Reference: llama-cli load log for Phi-4-mini-instruct Q6_K:
        // arch = phi3, n_embd = 3072, n_layer = 32, n_head = 24, n_head_kv = 8,
        // n_ff = 8192 (ffn_up packs 2*n_ff), n_rot = 96 (partial rotary)
        let (arch, h) = load(PHI4_MINI);
        assert_eq!(arch, LlmArch::PHI3);
        assert_eq!(h.n_embd, 3072);
        assert_eq!(h.n_layer(), 32);
        assert_eq!(h.n_head(0), 24);
        assert_eq!(h.n_head_kv(0), 8);
        assert_eq!(h.n_ff(0), 8192);
        assert_eq!(h.n_embd_head_k(0), 128); // 3072/24
        assert_eq!(h.n_rot(0), 96); // rope.dimension_count
        assert_eq!(h.n_ctx_train, 131072);
        assert_eq!(h.n_ctx_orig_yarn, 4096); // rope.scaling.original_context_length
                                             // rope.scaling.attn_factor present
        assert!((h.rope_attn_factor - 1.190238118171692).abs() < 1e-6);
        // attention.sliding_window = 262144 exists, but Phi SWA is
        // force-disabled in the reference implementation
        assert_eq!(h.swa_type, LlamaSwaType::NONE);
        assert_eq!(h.n_swa, 0);
        assert!(h.is_swa_impl.iter().all(|&v| v == 0));
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
    }

    #[test]
    fn gpt_oss_20b_values() {
        skip_unless_present!(GPT_OSS_20B);
        // Reference: llama-cli load log for gpt-oss-20b MXFP4:
        // arch = gpt-oss, n_layer = 24, n_head = 64, n_head_kv = 8,
        // n_embd = 2880, key_length = 64, n_expert = 32, n_expert_used = 4,
        // expert_feed_forward_length = 2880, swa 128 (pattern 2), yarn x32
        let (arch, h) = load(GPT_OSS_20B);
        assert_eq!(arch, LlmArch::OPENAI_MOE);
        assert_eq!(h.n_layer(), 24);
        assert_eq!(h.n_embd, 2880);
        assert_eq!(h.n_head(0), 64);
        assert_eq!(h.n_head_kv(0), 8);
        assert_eq!(h.n_embd_head_k(0), 64); // attention.key_length override
        assert_eq!(h.n_expert, 32);
        assert_eq!(h.n_expert_used_max(), 4);
        assert_eq!(h.n_ff_exp(0), 2880);
        assert_eq!(h.n_swa, 128);
        assert_eq!(h.swa_type, LlamaSwaType::STANDARD);
        // load_swa_pattern(ml, 2): even layers are SWA, odd layers dense
        // (set_swa_pattern(2, dense_first=false): il % 2 < 1)
        assert!(h.is_swa(0));
        assert!(!h.is_swa(1));
        // rope.scaling.type = yarn, factor = 32 -> freq_scale = 1/32
        assert_eq!(h.rope_scaling_type_train, LlamaRopeScalingType::YARN);
        assert!((h.rope_freq_scale_train - 1.0 / 32.0).abs() < 1e-9);
        assert_eq!(h.n_ctx_orig_yarn, 4096);
        assert_eq!(h.rope_freq_base_train, 150000.0);
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
    }

    #[test]
    fn ernie45_moe_values() {
        skip_unless_present!(ERNIE45);
        // Reference: llama-cli load log for ERNIE-4.5-21B-A3B-PT Q4_K_M:
        // arch = ernie4_5-moe (same class as ernie4_5), n_embd = 2560,
        // n_layer = 28, head_count = 20, head_count_kv = 4,
        // expert_count = 64, expert_used_count = 6 (+1 shared, ffn_shexp 3072),
        // interleave_moe_layer_step = 1, leading_dense_block_count = 1
        let (arch, h) = load(ERNIE45);
        assert!(
            matches!(arch, LlmArch::ERNIE4_5 | LlmArch::ERNIE4_5_MOE),
            "unexpected arch {arch:?}"
        );
        assert_eq!(h.n_layer(), 28);
        assert_eq!(h.n_embd, 2560);
        assert_eq!(h.n_head(0), 20);
        assert_eq!(h.n_head_kv(0), 4);
        assert_eq!(h.n_embd_head_k(0), 128);
        assert_eq!(h.n_ff(0), 12288);
        assert_eq!(h.n_expert, 64);
        assert_eq!(h.n_expert_used_max(), 6);
        if arch == LlmArch::ERNIE4_5_MOE {
            assert_eq!(h.n_moe_layer_step, 1);
            assert_eq!(h.n_layer_dense_lead, 1);
            assert_eq!(h.n_ff_exp(0), 1536); // dense lead layer repeats the value
            assert_eq!(h.n_ff_shexp, 3072);
        }
        assert_eq!(h.rope_freq_base_train, 500000.0);
        assert_eq!(h.rope_type, LlamaRopeType::NORM);
    }

    #[test]
    fn lfm2_8b_a1b_values() {
        skip_unless_present!(LFM2);
        // Reference: llama-cli load log for LFM2-8B-A1B Q4_K_M:
        // arch = lfm2moe (the -A1B models are MoE), n_layer = 24,
        // n_embd = 2048, hybrid conv/gated-attention with per-layer
        // head_count_kv = [0,0,8,0,0,0,8,...] (i32 array), 32 experts / 4 used,
        // shortconv.l_cache = 3, leading_dense_block_count = 2
        let (arch, h) = load(LFM2);
        assert_eq!(arch, LlmArch::LFM2MOE);
        assert_eq!(h.n_layer(), 24);
        assert_eq!(h.n_embd, 2048);
        assert_eq!(h.n_head(0), 32);
        assert_eq!(h.n_shortconv_l_cache, 3);
        assert_eq!(h.n_layer_dense_lead, 2);
        assert_eq!(h.n_expert, 32);
        assert_eq!(h.n_expert_used_max(), 4);
        assert_eq!(h.n_ff_exp(0), 1792);
        assert_eq!(h.expert_gating_func, 2); // sigmoid
                                             // per-layer kv heads: recurrent (0) then attention (2)
        assert_eq!(h.n_head_kv(0), 0);
        assert_eq!(h.n_head_kv(2), 8);
        // some layers recurrent (n_head_kv == 0), some not
        let n_recr = (0..h.n_layer() as usize)
            .filter(|&il| h.is_recr(il))
            .count();
        assert!(
            n_recr > 0 && n_recr < h.n_layer() as usize,
            "LFM2 must be hybrid"
        );
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
        // rolling state size for conv layers: n_embd * (l_cache - 1)
        assert_eq!(h.n_embd_r(), h.n_embd * (h.n_shortconv_l_cache - 1));
    }

    #[test]
    fn granitehybrid_tiny_values() {
        skip_unless_present!(GRANITE_H_TINY);
        // Reference: llama-cli load log for granite-4.0-h-tiny Q4_K_M:
        // arch = granitehybrid, n_layer = 40, n_embd = 1536, n_head = 12,
        // per-layer head_count_kv = [0 x5, 4, 0 x9, 4, ...] (i32 array),
        // n_ff = 512, 64 experts / 6 used + 1024-wide shared expert,
        // mamba2 ssm: conv_kernel 4, inner 3072, state 128, dt_rank 48, group 1,
        // rope.scaling.finetuned = false, granite scalars logit 6.0,
        // residual 0.22, embedding 12.0, attention 0.0078125
        let (arch, h) = load(GRANITE_H_TINY);
        assert_eq!(arch, LlmArch::GRANITE_HYBRID);
        assert_eq!(h.n_layer(), 40);
        assert_eq!(h.n_embd, 1536);
        assert_eq!(h.n_head(0), 12);
        assert_eq!(h.n_embd_head_k(0), 128);
        assert_eq!(h.n_rot(0), 128);
        assert_eq!(h.n_ff(0), 512);
        assert_eq!(h.n_expert, 64);
        assert_eq!(h.n_expert_used_max(), 6);
        assert_eq!(h.n_ff_shexp, 1024);

        // mamba2 params
        assert_eq!(h.ssm_d_conv, 4);
        assert_eq!(h.ssm_d_inner, 3072);
        assert_eq!(h.ssm_d_state, 128);
        assert_eq!(h.ssm_dt_rank, 48);
        assert_eq!(h.ssm_n_group, 1);

        // granite scalars
        assert_eq!(h.f_logit_scale, 6.0);
        assert_eq!(h.f_residual_scale, 0.22);
        assert_eq!(h.f_embedding_scale, 12.0);
        assert_eq!(h.f_attention_scale, 0.0078125);

        // rope_finetuned = false -> rope_pattern all 0 -> no RoPE anywhere
        assert!(!h.rope_finetuned);
        assert!((0..h.n_layer() as usize).all(|il| !h.has_rope(il)));

        // is_recr = (n_head_kv == 0): only layers 5/15/25/35 attend
        assert_eq!(h.n_head_kv(5), 4);
        assert_eq!(h.n_head_kv(6), 0);
        assert!(h.is_recr(0));
        assert!(!h.is_recr(5));
        assert!(h.is_recr(39));
        let n_recr = (0..h.n_layer() as usize)
            .filter(|&il| h.is_recr(il))
            .count();
        assert_eq!(n_recr, 36);

        assert_eq!(h.rope_type, LlamaRopeType::NORM);
        assert_eq!(
            h.n_embd_r(),
            (h.ssm_d_conv - 1) * (h.ssm_d_inner + 2 * h.ssm_n_group * h.ssm_d_state)
        );
        assert_eq!(h.n_embd_s(), h.ssm_d_state * h.ssm_d_inner);
    }

    #[test]
    fn qwen35_27b_values() {
        skip_unless_present!(QWEN36_27B);
        // Reference: llama-cli load log for Qwen3.6-27B Q4_K_M:
        // arch = qwen35, n_layer = 64, n_embd = 5120, n_head = 24,
        // head_count_kv = 4, key_length = 256, n_ff = 17408,
        // full_attention_interval = 4 (linear attention on 3 of 4 layers),
        // gated delta net: state 128, group 16, dt_rank 48,
        // rope.dimension_sections = [11, 11, 10, 0] (imrope)
        let (arch, h) = load(QWEN36_27B);
        assert_eq!(arch, LlmArch::QWEN35);
        assert_eq!(h.n_layer(), 64);
        assert_eq!(h.n_embd, 5120);
        assert_eq!(h.n_head(0), 24);
        assert_eq!(h.n_head_kv(0), 4);
        assert_eq!(h.n_embd_head_k(0), 256); // attention.key_length
        assert_eq!(h.n_rot(0), 64); // rope.dimension_count
        assert_eq!(h.n_ff(0), 17408);

        // gated delta net params
        assert_eq!(h.ssm_d_conv, 4);
        assert_eq!(h.ssm_d_inner, 6144);
        assert_eq!(h.ssm_d_state, 128);
        assert_eq!(h.ssm_dt_rank, 48);
        assert_eq!(h.ssm_n_group, 16);

        // rope sections + derived rope type
        assert_eq!(h.rope_sections, [11, 11, 10, 0]);
        assert!(h.use_mrope());
        assert_eq!(h.rope_type, LlamaRopeType::IMROPE);
        assert_eq!(h.n_pos_per_embd(), 4);

        // recurrent pattern derived from full_attention_interval = 4
        assert!(h.is_recr(0));
        assert!(h.is_recr(1));
        assert!(h.is_recr(2));
        assert!(!h.is_recr(3));
        assert!(!h.is_recr(7));
        assert!(!h.is_recr(63));
        let n_recr = (0..h.n_layer() as usize)
            .filter(|&il| h.is_recr(il))
            .count();
        assert_eq!(n_recr, 48);

        // no MTP layers in this file
        assert_eq!(h.n_layer_nextn, 0);
    }

    #[test]
    fn qwen35_mtp_values() {
        skip_unless_present!(QWEN38_27B_MTP);
        // Same arch with nextn_predict_layers = 1: n_layer() excludes the MTP
        // block, and the MTP block itself is a dense attention layer.
        let (arch, h) = load(QWEN38_27B_MTP);
        assert_eq!(arch, LlmArch::QWEN35);
        assert_eq!(h.n_layer_all, 65);
        assert_eq!(h.n_layer_nextn, 1);
        assert_eq!(h.n_layer(), 64);
        assert!(!h.is_recr(64), "MTP layer must be non-recurrent");
        assert!(h.is_recr(62));
        // the recurrence pattern is still the full-attention interval
        assert!(!h.is_recr(63));
    }

    #[test]
    fn bge_m3_bert_values() {
        skip_unless_present!(BGE_M3);
        // Reference: llama-cli load log for bge-m3 Q8_0:
        // arch = bert, n_embd = 1024, n_layer = 24, n_head = 16,
        // layer_norm_epsilon = 1e-5, pooling_type = CLS, causal = false
        let (arch, h) = load(BGE_M3);
        assert_eq!(arch, LlmArch::BERT);
        assert_eq!(h.n_embd, 1024);
        assert_eq!(h.n_layer(), 24);
        assert_eq!(h.n_head(0), 16);
        assert_eq!(h.n_head_kv(0), 16);
        assert_eq!(h.n_embd_head_k(0), 64);
        // bert uses layer_norm_epsilon (f_norm_eps), not rms
        assert!((h.f_norm_eps - 1e-5).abs() < 1e-12);
        assert_eq!(h.pooling_type, LlamaPoolingType::CLS);
        assert!(!h.causal_attn); // attention.causal = false
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
    }

    #[test]
    fn qwen3_embedding_values() {
        skip_unless_present!(QWEN3_EMB);
        // Reference: Qwen3-Embedding-0.6B: arch = qwen3, n_embd = 1024,
        // n_layer = 28, n_head = 16, n_head_kv = 8, key_length = 128,
        // pooling_type = LAST (3)
        let (arch, h) = load(QWEN3_EMB);
        assert_eq!(arch, LlmArch::QWEN3);
        assert_eq!(h.n_embd, 1024);
        assert_eq!(h.n_layer(), 28);
        assert_eq!(h.n_head(0), 16);
        assert_eq!(h.n_head_kv(0), 8);
        assert_eq!(h.n_embd_head_k(0), 128); // attention.key_length override
        assert_eq!(h.pooling_type, LlamaPoolingType::LAST);
        assert_eq!(h.rope_freq_base_train, 1000000.0);
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
    }

    #[test]
    fn seed_oss_partial_arch() {
        // seed_oss follows the generic path (arch-specific loader not ported)
        let Some(gguf) = open(SEED_OSS) else { return };
        let (arch, h) = load_hparams(&gguf).expect("generic path must succeed");
        assert_eq!(arch, LlmArch::SEED_OSS);
        assert_eq!(arch_hparams_support(arch), ArchHparamsSupport::Partial);
        // generic path still fills the basics
        assert_eq!(h.n_layer(), 64);
        assert_eq!(h.n_head(0), 80);
        assert_eq!(h.n_head_kv(0), 8);
        assert_eq!(h.n_embd_head_k(0), 128);
        assert_eq!(h.rope_type, LlamaRopeType::NEOX);
    }

    #[test]
    fn error_paths() {
        // no architecture at all -> the first required arch-prefixed read fails
        // with the C++ error text
        let p = write_minimal_gguf(&[]);
        let gguf = Gguf::open(&p).unwrap();
        let err = load_hparams(&gguf).unwrap_err();
        assert!(
            err.contains("key not found in model: (unknown).context_length"),
            "{err}"
        );

        // arch present but required keys missing
        let p = write_minimal_gguf(&[("general.architecture", Value::String("qwen2".into()))]);
        let gguf = Gguf::open(&p).unwrap();
        let err = load_hparams(&gguf).unwrap_err();
        assert!(
            err.contains("key not found in model: qwen2.context_length"),
            "{err}"
        );

        // full generic set, no head_count_kv -> n_head_kv defaults to n_head;
        // missing attention.layer_norm_rms_epsilon -> arch-specific error
        let p = write_minimal_gguf(&[
            ("general.architecture", Value::String("qwen2".into())),
            ("qwen2.context_length", Value::U32(512)),
            ("qwen2.embedding_length", Value::U32(64)),
            ("qwen2.block_count", Value::U32(2)),
            ("qwen2.attention.head_count", Value::U32(4)),
            ("qwen2.feed_forward_length", Value::U32(128)),
        ]);
        let gguf = Gguf::open(&p).unwrap();
        let err = load_hparams(&gguf).unwrap_err();
        assert!(
            err.contains("key not found in model: qwen2.attention.layer_norm_rms_epsilon"),
            "{err}"
        );

        // ... and with the eps present it loads; head_count_kv defaults to head_count
        let p = write_minimal_gguf(&[
            ("general.architecture", Value::String("qwen2".into())),
            ("qwen2.context_length", Value::U32(512)),
            ("qwen2.embedding_length", Value::U32(64)),
            ("qwen2.block_count", Value::U32(2)),
            ("qwen2.attention.head_count", Value::U32(4)),
            ("qwen2.feed_forward_length", Value::U32(128)),
            ("qwen2.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ]);
        let gguf = Gguf::open(&p).unwrap();
        let (arch, h) = load_hparams(&gguf).unwrap();
        assert_eq!(arch, LlmArch::QWEN2);
        assert_eq!(h.n_head_kv(0), 4); // defaulted from n_head
        assert_eq!(h.n_embd_head_k(0), 16);

        // wrong wire type for a required key
        let p = write_minimal_gguf(&[
            ("general.architecture", Value::String("qwen2".into())),
            ("qwen2.context_length", Value::F32(512.0)),
        ]);
        let gguf = Gguf::open(&p).unwrap();
        let err = load_hparams(&gguf).unwrap_err();
        assert!(
            err.contains("qwen2.context_length has wrong type float32 but expected type uint32"),
            "{err}"
        );

        // block_count == 0 trips the GGML_ASSERT port
        let p = write_minimal_gguf(&[
            ("general.architecture", Value::String("qwen2".into())),
            ("qwen2.context_length", Value::U32(512)),
            ("qwen2.embedding_length", Value::U32(64)),
            ("qwen2.block_count", Value::U32(0)),
        ]);
        let gguf = Gguf::open(&p).unwrap();
        assert!(load_hparams(&gguf).is_err());
    }

    #[test]
    fn support_matrix_sanity() {
        use ArchHparamsSupport::*;
        for &a in LlmArch::ALL {
            match arch_hparams_support(a) {
                Full => assert_ne!(a, LlmArch::UNKNOWN),
                Partial => {}
            }
        }
        assert_eq!(arch_hparams_support(LlmArch::QWEN2), Full);
        // agent P: granitehybrid + qwen35 hparams now ported 1:1
        assert_eq!(arch_hparams_support(LlmArch::GRANITE_HYBRID), Full);
        assert_eq!(arch_hparams_support(LlmArch::QWEN35), Full);
        // still on the generic path only
        assert_eq!(arch_hparams_support(LlmArch::SEED_OSS), Partial);
        assert_eq!(arch_hparams_support(LlmArch::QWEN3NEXT), Partial);
    }
}
