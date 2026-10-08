//! speculative — port of `common/speculative.cpp` / `common/speculative.h`
//! (llama.cpp bd4f514db1), plus the two pieces of the reference that drive it:
//!
//!   * `common/sampling.cpp:678-715` — `common_sampler_sample_and_accept_n`,
//!     the per-token accept/reject rule of the verification step,
//!   * `examples/speculative-simple/speculative-simple.cpp:126-342` — the
//!     draft → verify → accept loop ([`speculative_simple_generate`]), including
//!     the KV-cache bookkeeping of both models (`llama_memory_seq_rm` on the
//!     target and on the draft context) and the sampler-state rule (advanced
//!     once per accepted token — see [`common_sampler_sample_and_accept_n`]).
//!
//! Ported speculative implementations: **draft-simple**, **draft-eagle3**
//! (speculative.cpp:426-907 over the eagle-head graphs in graph_arch.rs and
//! the eagle3 head context of context.rs/eagle.rs), **draft-mtp**
//! (the deepseek2/deepseek32/deepseek4 MTP/nextn family — the impl of
//! speculative.cpp:1330-1767 driving the `graph_mtp` builders in
//! graph_arch.rs and the `LLAMA_CONTEXT_TYPE_MTP` draft context in
//! context.rs), **draft-dflash / draft-dspark** (speculative.cpp:910-1328 —
//! the same impl for both types, over the dflash draft context of
//! context.rs/dflash.rs: `process()` injects the target's extract-layer
//! features into the draft KV, `draft()` decodes the noise block; the DSpark
//! markov-biased logits + confidence rows included, the DFlash2 selector
//! lattice and the DSV4 backbone refused at load — see dflash.rs) and the
//! **ngram family** — `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` /
//! `ngram-mod` / `ngram-cache` (speculative.cpp:1769-2181
//! over the ported `crate::ngram_map` / `crate::ngram_mod` /
//! `crate::ngram_cache`, the C's own `common/ngram-*.cpp` files). The pinned
//! CLI/server reach them with `--spec-type ngram-*` / `--spec-type
//! draft-eagle3` / `--spec-type draft-dflash` / `--spec-type draft-dspark`
//! (`common/arg.cpp` 4136-4250); self-drafting (draft ==
//! target, no `-md`) is the ngram use case.
//!
//! Ownership mapping of the C's raw pointers (each documented at the use site):
//!   * `common_params_speculative_draft::ctx_dft` — the port's
//!     [`CommonSpeculative`] **owns** the draft [`DecodeContext`], because the
//!     impls decode on it (`common_speculative_impl_draft_simple::process` /
//!     `::draft`); the C keeps a pointer into `common_speculative_init_result`
//!     and the driver keeps a second one. The driver reaches the context
//!     through [`CommonSpeculative::ctx_dft`].
//!   * `common_speculative_draft_params::result` / `::prompt` — owned vectors
//!     instead of pointers into the driver's `draft` / `prompt_tgt`
//!     (speculative.h:68-71).
//!   * `common_speculative::impl_last` — an index into `impls` instead of a
//!     pointer (speculative.cpp:2190).
//!
//! Verification (see PARITY.md's speculative section and
//! `crates/llama/tests/speculative_e2e.rs`): on a margin-stable 36-token run
//! (0.5B draft + the local 7B target) the port's speculative stream equals its
//! plain greedy stream and the reference's own plain *and* `-md` runs, with
//! 100% draft acceptance (mean accepted length 4.00) and 73% fewer target
//! forwards. At `temperature 0.8` with a fixed seed the streams still match,
//! which is the direct check of the "sampler chain advances once per committed
//! token" rule. **Speed**: the port is currently *slower* (≈0.5x) because a
//! 4-row verify forward costs 2.8-3.0x a single-row one — the q4_K/Q6_K GEMM
//! path is missing (`tinyblas::resolve` covers F32/F16/BF16/Q4_0/Q5_0/Q8_0
//! only) and `mul_mat` falls back to one `vec_dot` per activation row
//! (`crates/ggml/src/compute.rs:976-980`).

use std::time::Instant;

use crate::batch::LlamaBatch;
use crate::context::{BatchOutput, DecodeContext};
use crate::sampling::{
    init_dist, init_logit_bias, init_top_k, LlamaToken, SamplerChain, SamplingContext, TokenData,
    TokenDataArray, LLAMA_DEFAULT_SEED,
};
use crate::vocab::Vocab;

// ---------------------------------------------------------------------------
// constants (speculative.cpp:30-31)
// ---------------------------------------------------------------------------

/// `SPEC_VOCAB_MAX_SIZE_DIFFERENCE` (speculative.cpp:30)
pub const SPEC_VOCAB_MAX_SIZE_DIFFERENCE: i32 = 128;
/// `SPEC_VOCAB_CHECK_START_TOKEN_ID` (speculative.cpp:31)
pub const SPEC_VOCAB_CHECK_START_TOKEN_ID: i32 = 5;

/// The reference's `SPC_DBG` / `SPC_TRC` lines are gated by the global log
/// level (`common/log.h`, default INFO) and therefore silent by default; the
/// port models that level with this flag. `SPC_WRN` / `SPC_ERR` equivalent
/// messages always print.
static SPEC_VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// enable the `SPC_DBG` / `SPC_TRC` equivalent traces (`--verbose` / `-v` in
/// the reference)
pub fn spec_set_verbose(verbose: bool) {
    SPEC_VERBOSE.store(verbose, std::sync::atomic::Ordering::Relaxed);
}

fn spec_verbose() -> bool {
    SPEC_VERBOSE.load(std::sync::atomic::Ordering::Relaxed)
}

/// `SPC_DBG` (speculative.cpp:23) — silent unless [`spec_set_verbose`]
macro_rules! spec_dbg {
    ($($arg:tt)*) => {
        if spec_verbose() {
            eprintln!($($arg)*);
        }
    };
}

// ---------------------------------------------------------------------------
// common_speculative_type (common.h:170-183) + the name maps
// (speculative.cpp:33-45, 2222-2325)
// ---------------------------------------------------------------------------

/// `enum common_speculative_type` (common.h:170-183); the C's `_COUNT` member
/// is folded into [`CommonSpeculativeType::COUNT`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommonSpeculativeType {
    None,
    DraftSimple,
    DraftEagle3,
    DraftMtp,
    DraftDflash,
    DraftDspark,
    NgramSimple,
    NgramMapK,
    NgramMapK4v,
    NgramMod,
    NgramCache,
}

impl CommonSpeculativeType {
    /// `COMMON_SPECULATIVE_TYPE_COUNT` (common.h:183)
    pub const COUNT: usize = 11;

    /// every value in enum order (`common_speculative_all_types_str` walks
    /// `0..COMMON_SPECULATIVE_TYPE_COUNT`, speculative.cpp:2234-2244)
    pub const ALL: [CommonSpeculativeType; Self::COUNT] = [
        CommonSpeculativeType::None,
        CommonSpeculativeType::DraftSimple,
        CommonSpeculativeType::DraftEagle3,
        CommonSpeculativeType::DraftMtp,
        CommonSpeculativeType::DraftDflash,
        CommonSpeculativeType::DraftDspark,
        CommonSpeculativeType::NgramSimple,
        CommonSpeculativeType::NgramMapK,
        CommonSpeculativeType::NgramMapK4v,
        CommonSpeculativeType::NgramMod,
        CommonSpeculativeType::NgramCache,
    ];

    /// `common_speculative_type_to_str` (speculative.cpp:2246-2261)
    pub fn to_str(self) -> &'static str {
        match self {
            CommonSpeculativeType::None => "none",
            CommonSpeculativeType::DraftSimple => "draft-simple",
            CommonSpeculativeType::DraftEagle3 => "draft-eagle3",
            CommonSpeculativeType::DraftMtp => "draft-mtp",
            CommonSpeculativeType::DraftDflash => "draft-dflash",
            CommonSpeculativeType::DraftDspark => "draft-dspark",
            CommonSpeculativeType::NgramSimple => "ngram-simple",
            CommonSpeculativeType::NgramMapK => "ngram-map-k",
            CommonSpeculativeType::NgramMapK4v => "ngram-map-k4v",
            CommonSpeculativeType::NgramMod => "ngram-mod",
            CommonSpeculativeType::NgramCache => "ngram-cache",
        }
    }

    /// `common_speculative_type_from_name_map` (speculative.cpp:33-45)
    pub fn from_name(name: &str) -> Option<CommonSpeculativeType> {
        Some(match name {
            "none" => CommonSpeculativeType::None,
            "draft-simple" => CommonSpeculativeType::DraftSimple,
            "draft-eagle3" => CommonSpeculativeType::DraftEagle3,
            "draft-mtp" => CommonSpeculativeType::DraftMtp,
            "draft-dflash" => CommonSpeculativeType::DraftDflash,
            "draft-dspark" => CommonSpeculativeType::DraftDspark,
            "ngram-simple" => CommonSpeculativeType::NgramSimple,
            "ngram-map-k" => CommonSpeculativeType::NgramMapK,
            "ngram-map-k4v" => CommonSpeculativeType::NgramMapK4v,
            "ngram-mod" => CommonSpeculativeType::NgramMod,
            "ngram-cache" => CommonSpeculativeType::NgramCache,
            _ => return None,
        })
    }
}

/// `common_speculative_type_name_str` (speculative.cpp:2222-2232)
pub fn common_speculative_type_name_str(types: &[CommonSpeculativeType]) -> String {
    let mut result = String::new();

    for (i, ty) in types.iter().enumerate() {
        if i > 0 {
            result.push(',');
        }
        result.push_str(ty.to_str());
    }
    result
}

/// `common_speculative_all_types_str` (speculative.cpp:2234-2244)
pub fn common_speculative_all_types_str() -> String {
    common_speculative_type_name_str(&CommonSpeculativeType::ALL)
}

/// `common_speculative_types_from_names` (speculative.cpp:2263-2280): a `none`
/// entry short-circuits the whole list; an unknown name is an error (the C
/// throws `std::invalid_argument`).
pub fn common_speculative_types_from_names(
    names: &[String],
) -> Result<Vec<CommonSpeculativeType>, String> {
    let mut types = Vec::with_capacity(names.len());

    for name in names {
        match CommonSpeculativeType::from_name(name) {
            Some(CommonSpeculativeType::None) => return Ok(vec![CommonSpeculativeType::None]),
            Some(ty) => types.push(ty),
            None => return Err(format!("unknown speculative type: {name}")),
        }
    }

    Ok(types)
}

/// `common_speculative_types_from_gguf` (speculative.cpp:2290-2325) — infers
/// the speculative type from the draft GGUF's metadata; empty if unknown.
///
/// `gguf_find_tensor` needs the tensor *names* only, which the port's reader
/// has without touching tensor data (`Gguf::tensors`).
pub fn common_speculative_types_from_gguf(path: &str) -> Vec<CommonSpeculativeType> {
    // gguf_init_params{ .no_alloc = true } — the port's reader never allocates
    // tensor data anyway
    let Ok(gguf) = ggml::Gguf::open(path) else {
        return Vec::new();
    };

    let Some(arch) = gguf.get_str("general.architecture") else {
        return Vec::new();
    };
    let arch = arch.to_string();

    if arch != "dflash" {
        // `gguf_get_val_u32(..., (arch + ".block_count").c_str())`
        // (speculative.cpp:2308)
        let Some(block_count) = gguf.get_u32(&format!("{arch}.block_count")) else {
            return Vec::new();
        };

        if gguf
            .find_tensor(&format!("blk.{}.nextn.eh_proj.weight", block_count - 1))
            .is_some()
        {
            return vec![CommonSpeculativeType::DraftMtp];
        }

        return Vec::new();
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    let ty = if gguf.find_tensor("markov_w1.weight").is_some() {
        CommonSpeculativeType::DraftDspark
    } else {
        CommonSpeculativeType::DraftDflash
    };

    spec_dbg!(
        "spec common_speculative_types_from_gguf: auto-detected speculative type '{}' from the draft model metadata",
        ty.to_str()
    );

    vec![ty]
}

/// `common_speculative_are_compatible` (speculative.cpp:67-130) — the draft and
/// target vocabularies must be interchangeable for speculation to be valid.
pub fn common_speculative_are_compatible(model_tgt: &Vocab, model_dft: &Vocab) -> bool {
    let vocab_type_tgt = model_tgt.get_type();
    spec_dbg!("spec common_speculative_are_compatible: vocab_type tgt: {vocab_type_tgt:?}");

    let vocab_type_dft = model_dft.get_type();
    spec_dbg!("spec common_speculative_are_compatible: vocab_type dft: {vocab_type_dft:?}");

    if vocab_type_tgt != vocab_type_dft {
        eprintln!(
            "spec common_speculative_are_compatible: draft model vocab type must match target \
             model to use speculation but vocab_type_dft = {vocab_type_dft:?} while \
             vocab_type_tgt = {vocab_type_tgt:?}"
        );
        return false;
    }

    if model_tgt.get_add_bos() != model_dft.get_add_bos()
        || (model_tgt.get_add_bos() && model_tgt.token_bos() != model_dft.token_bos())
    {
        eprintln!(
            "spec common_speculative_are_compatible: draft model bos tokens must match target \
             model to use speculation. add: {} - {}, id: {} - {}",
            model_tgt.get_add_bos(),
            model_dft.get_add_bos(),
            model_tgt.token_bos(),
            model_dft.token_bos(),
        );
        return false;
    }

    if model_tgt.get_add_eos() != model_dft.get_add_eos()
        || (model_tgt.get_add_eos() && model_tgt.token_eos() != model_dft.token_eos())
    {
        eprintln!(
            "spec common_speculative_are_compatible: draft model eos tokens must match target \
             model to use speculation. add: {} - {}, id: {} - {}",
            model_tgt.get_add_eos(),
            model_dft.get_add_eos(),
            model_tgt.token_eos(),
            model_dft.token_eos(),
        );
        return false;
    }

    {
        let n_vocab_tgt = model_tgt.n_tokens() as i32;
        let n_vocab_dft = model_dft.n_tokens() as i32;
        let vocab_diff = (n_vocab_tgt - n_vocab_dft).abs();

        if vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE {
            spec_dbg!(
                "spec common_speculative_are_compatible: draft model vocab must closely match \
                 target model to use speculation but target vocab size {n_vocab_tgt} does not \
                 match draft vocab size {n_vocab_dft} - difference {vocab_diff}, max allowed \
                 {SPEC_VOCAB_MAX_SIZE_DIFFERENCE}"
            );
            return false;
        }

        for i in SPEC_VOCAB_CHECK_START_TOKEN_ID..n_vocab_tgt.min(n_vocab_dft) {
            let token_text_tgt = model_tgt.token_get_text(i);
            let token_text_dft = model_dft.token_get_text(i);

            if token_text_tgt != token_text_dft {
                spec_dbg!(
                    "spec common_speculative_are_compatible: draft model vocab must match target \
                     model to use speculation but token {i} content differs - target '{}', \
                     draft '{}'",
                    model_tgt.token_to_piece(i),
                    model_dft.token_to_piece(i),
                );
                return false;
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
// common_params_speculative (common.h:325-401 subset — what the CLI reaches
// through -md / --spec-type / --spec-draft-* / --spec-synth-*)
// ---------------------------------------------------------------------------

/// `struct common_params_speculative_draft` (common.h:325-350)
#[derive(Clone, Debug)]
pub struct CommonParamsSpeculativeDraft {
    /// maximum number of tokens to draft during speculative decoding
    pub n_max: i32,
    /// minimum number of draft tokens to use for speculative decoding
    pub n_min: i32,

    /// speculative decoding split probability (draft-mtp / dflash only)
    pub p_split: f32,
    /// minimum speculative decoding probability (greedy) — the *draft* token
    /// confidence gate of `common_speculative_impl_draft_simple::draft`
    /// (speculative.cpp:336-342)
    pub p_min: f32,

    /// offload draft sampling to the backend (default: on) — only the
    /// eagle3/dflash impls use it (speculative.cpp:497-511); carried for
    /// argument parity
    pub backend_sampling: bool,

    /// sample the draft and verify by rejection, instead of argmax and match
    /// (common.h:337, upstream a7b94df2c) — `--spec-draft-sampling
    /// {greedy,probabilistic}`
    pub probabilistic: bool,

    /// `common_params_model::path` of the draft model (common.h:325-334
    /// `mparams`) — `--spec-draft-model` / `-md`
    pub model_path: String,
}

impl Default for CommonParamsSpeculativeDraft {
    fn default() -> Self {
        CommonParamsSpeculativeDraft {
            n_max: 3,
            n_min: 0,
            p_split: 0.1,
            p_min: 0.0,
            backend_sampling: true,
            probabilistic: false,
            model_path: String::new(),
        }
    }
}

/// `struct common_params_speculative_ngram_mod` (common.h:352-357)
#[derive(Clone, Debug)]
pub struct CommonParamsSpeculativeNgramMod {
    pub n_match: i32,
    pub n_max: i32,
    pub n_min: i32,
}

impl Default for CommonParamsSpeculativeNgramMod {
    fn default() -> Self {
        CommonParamsSpeculativeNgramMod {
            n_match: 24,
            n_max: 64,
            n_min: 48,
        }
    }
}

/// `struct common_params_speculative_ngram_map` (common.h:359-363)
#[derive(Clone, Debug)]
pub struct CommonParamsSpeculativeNgramMap {
    /// ngram size for lookup
    pub size_n: u16,
    /// mgram size for speculative tokens
    pub size_m: u16,
    /// minimum hits at ngram/mgram lookup for mgram to be proposed
    pub min_hits: u16,
}

impl Default for CommonParamsSpeculativeNgramMap {
    fn default() -> Self {
        CommonParamsSpeculativeNgramMap {
            size_n: 12,
            size_m: 48,
            min_hits: 1,
        }
    }
}

/// `struct common_params_speculative_ngram_cache` (common.h:365-368)
#[derive(Clone, Debug, Default)]
pub struct CommonParamsSpeculativeNgramCache {
    /// path of static ngram cache file for lookup decoding
    pub lookup_cache_static: String,
    /// path of dynamic ngram cache file for lookup decoding
    pub lookup_cache_dynamic: String,
}

/// `struct common_params_speculative` (common.h:370-401)
#[derive(Clone, Debug)]
pub struct CommonParamsSpeculative {
    /// `types` — `{ COMMON_SPECULATIVE_TYPE_NONE }` by default (common.h:371)
    pub types: Vec<CommonSpeculativeType>,

    /// `synth_len` / `synth_rates` (speculative.cpp:2385-2459)
    pub synth_len: f64,
    pub synth_rates: Vec<f64>,

    pub draft: CommonParamsSpeculativeDraft,

    /// the ngram family sub-params (common.h:379-384)
    pub ngram_mod: CommonParamsSpeculativeNgramMod,
    pub ngram_simple: CommonParamsSpeculativeNgramMap,
    pub ngram_map_k: CommonParamsSpeculativeNgramMap,
    pub ngram_map_k4v: CommonParamsSpeculativeNgramMap,
    pub ngram_cache: CommonParamsSpeculativeNgramCache,
}

impl Default for CommonParamsSpeculative {
    fn default() -> Self {
        CommonParamsSpeculative {
            types: vec![CommonSpeculativeType::None],
            synth_len: -1.0,
            synth_rates: Vec::new(),
            draft: CommonParamsSpeculativeDraft::default(),
            ngram_mod: CommonParamsSpeculativeNgramMod::default(),
            ngram_simple: CommonParamsSpeculativeNgramMap::default(),
            ngram_map_k: CommonParamsSpeculativeNgramMap::default(),
            ngram_map_k4v: CommonParamsSpeculativeNgramMap::default(),
            ngram_cache: CommonParamsSpeculativeNgramCache::default(),
        }
    }
}

impl CommonParamsSpeculative {
    /// `common_params_speculative::has_dft` (common.h:386-388)
    pub fn has_dft(&self) -> bool {
        !self.draft.model_path.is_empty()
    }

    /// `common_params_speculative::has_synth` (common.h:390-392)
    pub fn has_synth(&self) -> bool {
        self.synth_len != -1.0 || !self.synth_rates.is_empty()
    }
}

/// `common_speculative_n_max(const common_params_speculative *)`
/// (speculative.cpp:2335-2369) — the ngram branches read their own
/// sub-params now that the family is ported (the defaults are common.h:
/// 352-368).
pub fn common_speculative_n_max_params(spec: &CommonParamsSpeculative) -> i32 {
    let mut n_max = 0;

    for &ty in &spec.types {
        match ty {
            CommonSpeculativeType::DraftSimple
            | CommonSpeculativeType::DraftEagle3
            | CommonSpeculativeType::DraftMtp
            | CommonSpeculativeType::DraftDflash
            | CommonSpeculativeType::DraftDspark => {
                n_max = n_max.max(spec.draft.n_max.max(0));
            }
            CommonSpeculativeType::NgramSimple => {
                n_max = n_max.max(spec.ngram_simple.size_m as i32); // :2348
            }
            CommonSpeculativeType::NgramMapK => {
                n_max = n_max.max(spec.ngram_map_k.size_m as i32); // :2351
            }
            CommonSpeculativeType::NgramMapK4v => {
                n_max = n_max.max(spec.ngram_map_k4v.size_m as i32); // :2354
            }
            CommonSpeculativeType::NgramMod => {
                n_max = n_max.max(spec.ngram_mod.n_max.max(0)); // :2357
            }
            CommonSpeculativeType::NgramCache => {
                n_max = n_max.max(8); // :2360 (TODO get from config?)
            }
            CommonSpeculativeType::None => {}
        }
    }

    n_max
}

/// `common_speculative_synth_rates_resolve` (speculative.cpp:2385-2459). The C
/// throws `std::invalid_argument`; the port returns `Err`.
pub fn common_speculative_synth_rates_resolve(
    spec: &CommonParamsSpeculative,
    n_max: i32,
) -> Result<Vec<f64>, String> {
    let has_length = spec.synth_len != -1.0;
    let has_rates = !spec.synth_rates.is_empty();

    if !has_length && !has_rates {
        return Ok(Vec::new());
    }
    if has_length && has_rates {
        return Err("synthetic acceptance length and rates are mutually exclusive".into());
    }

    if n_max <= 0 {
        return Err("synthetic acceptance requires at least one speculative token".into());
    }

    if has_rates {
        let rates = spec.synth_rates.clone();
        if rates.len() != n_max as usize {
            return Err(format!(
                "synthetic acceptance rates must contain {n_max} values, got {}",
                rates.len()
            ));
        }

        for (i, &rate) in rates.iter().enumerate() {
            if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                return Err("synthetic acceptance rates must be finite and within [0, 1]".into());
            }
            if i > 0 && rate > rates[i - 1] {
                return Err(
                    "synthetic acceptance rates must be monotonically non-increasing".into(),
                );
            }
        }

        return Ok(rates);
    }

    // `synth_len` → unconditional rates by bisection on p
    // (`sum_{j<n_max} p^(j+1) == length - 1`, speculative.cpp:2426-2448)
    let length = spec.synth_len;
    let length_max = n_max as f64 + 1.0;
    if !length.is_finite() || length < 1.0 || length > length_max {
        return Err(format!(
            "synthetic acceptance length must be finite and within [1, {length_max:.0}]"
        ));
    }

    let mut p = 0.0;
    if length == length_max {
        p = 1.0;
    } else if length > 1.0 {
        let mut p_min = 0.0;
        let mut p_max = 1.0;
        for _ in 0..32 {
            let p_mid = 0.5 * (p_min + p_max);
            let mut sum = 0.0;
            let mut term = p_mid;
            for _ in 0..n_max {
                sum += term;
                term *= p_mid;
            }

            if sum < length - 1.0 {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    let mut rates = Vec::with_capacity(n_max as usize);
    let mut rate = p;
    for _ in 0..n_max {
        rates.push(rate);
        rate *= p;
    }

    Ok(rates)
}

/// `common_speculative_output_limits` (speculative.h:40-47)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommonSpeculativeOutputLimits {
    pub total: i32,
    pub per_seq: i32,
}

/// `common_speculative_get_output_limits` (speculative.cpp:2606-2615)
pub fn common_speculative_get_output_limits(
    n_batch: i32,
    n_parallel: i32,
    n_draft: i32,
) -> CommonSpeculativeOutputLimits {
    let per_seq = 1 + n_draft.max(0) as i64;
    let total = n_parallel as i64 * per_seq;

    CommonSpeculativeOutputLimits {
        total: n_batch.min(total as i32),
        per_seq: n_batch.min(per_seq as i32),
    }
}

// ---------------------------------------------------------------------------
// common_speculative_draft_params (speculative.h:53-72)
// ---------------------------------------------------------------------------

/// `struct common_speculative_draft_params` (speculative.h:53-72).
///
/// `prompt` is an owned copy of the driver's `prompt_tgt` and `result` is the
/// draft output buffer the driver reads back; the C stores pointers into the
/// driver's own vectors (speculative.h:68-71).
#[derive(Clone, Debug)]
pub struct CommonSpeculativeDraftParams {
    /// chain the drafts through all the available implementations: after the
    /// first successful draft from an implementation it is cleared so the
    /// remaining impls skip that sequence; `common_speculative_draft` resets
    /// every flag at the end of the call (speculative.h:54-58)
    pub drafting: bool,

    /// overrides individual configurations (-1 disabled), e.g. constrained to
    /// the remaining context size (speculative.h:60-62)
    pub n_max: i32,

    pub pos0: i32,
    pub id_last: i32,

    /// TODO: remove in the future by keeping track of the prompt from the
    /// _begin() call and the consecutive accept calls (speculative.h:67-68)
    pub prompt: Vec<i32>,

    /// the generated draft from the last _draft() call (speculative.h:70-71)
    pub result: Vec<i32>,

    /// candidate distribution per drafted token; set it to make draft-simple
    /// and draft-mtp sample (speculative.h:75-76, upstream a7b94df2c). The C
    /// stores a pointer to the driver's `slot.spec_draft_q`; the port owns
    /// the buffer and the driver takes it after the draft — the same
    /// take-back pattern `result` uses
    pub result_q: Option<Vec<Vec<TokenData>>>,

    /// the target's temp and seed, read only when the drafter samples
    /// probabilistically (speculative.h:78-79)
    pub temp: f32,
    pub seed: u32,
}

impl Default for CommonSpeculativeDraftParams {
    fn default() -> Self {
        CommonSpeculativeDraftParams {
            drafting: false,
            n_max: -1,
            pos0: 0,
            id_last: 0,
            prompt: Vec::new(),
            result: Vec::new(),
            result_q: None,
            temp: 1.0,
            seed: LLAMA_DEFAULT_SEED,
        }
    }
}

// ---------------------------------------------------------------------------
// per-implementation statistics (speculative.cpp:138-177)
// ---------------------------------------------------------------------------

/// The counters of `common_speculative_impl` (speculative.cpp:141-160).
/// `gen_perf` is `const bool ... = true` in the C (:156), so the port always
/// measures; the `t_*_us` accumulators are `common_time_meas` deltas.
#[derive(Clone, Debug, Default)]
pub struct CommonSpeculativeImplStats {
    /// number of times this implementation was called for refresh
    pub n_call_begin: u64,
    /// number of times this implementation was called for generation
    pub n_call_draft: u64,
    /// number of times this implementation was called for accumulation
    pub n_call_accept: u64,

    /// number of times a draft or part was generated by this implementation
    pub n_gen_drafts: u64,
    /// number of times a draft or part was accepted by the target model
    pub n_acc_drafts: u64,
    /// number of tokens generated by this implementation
    pub n_gen_tokens: u64,
    /// number of tokens accepted by the target model
    pub n_acc_tokens: u64,

    /// number of tokens accepted per draft position
    pub n_acc_tokens_per_pos: Vec<u64>,

    pub t_begin_us: i64,
    pub t_draft_us: i64,
    pub t_accept_us: i64,

    /// port-only: `llama_decode` calls this implementation made on its own
    /// context (the draft model's forward passes — the reference reports the
    /// same work through `common_perf_print`/`llama_perf_context`);
    /// `n_call_draft` counts `draft()` calls, not forward passes
    pub n_forward: u64,
}

impl CommonSpeculativeImplStats {
    /// mean accepted length, including the target token
    /// (`common_speculative_print_stats`, speculative.cpp:2971-2973)
    pub fn mean_acc_len(&self) -> f64 {
        if self.n_call_accept == 0 {
            return 0.0;
        }
        1.0 + self.n_acc_tokens as f64 / self.n_call_accept as f64
    }

    /// `100 * n_acc_tokens / n_gen_tokens` — the acceptance rate
    /// `examples/speculative-simple` prints as `accept = %.3f%%`
    /// (speculative-simple.cpp:358).
    pub fn accept_rate(&self) -> f64 {
        if self.n_gen_tokens == 0 {
            return 0.0;
        }
        100.0 * self.n_acc_tokens as f64 / self.n_gen_tokens as f64
    }
}

/// `ggml_time_us()` — monotonic microseconds (`common_time_meas`'s clock).
fn now_us() -> i64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_micros() as i64
}

// ---------------------------------------------------------------------------
// the implementation interface (speculative.cpp:138-177)
// ---------------------------------------------------------------------------

/// `struct common_speculative_impl` (speculative.cpp:138-177). The
/// implementation-specific state lives in the implementor (the C uses a
/// subclass); the common counters/timers live in
/// [`CommonSpeculativeImplStats`] so `common_speculative_accept` and
/// `print_stats` stay impl-agnostic.
pub trait CommonSpeculativeImpl {
    /// `common_speculative_impl::type` (speculative.cpp:139)
    fn ty(&self) -> CommonSpeculativeType;

    /// `common_speculative_impl::n_max` — maximum draft length after
    /// implementation-specific limits (speculative.cpp:142)
    fn n_max(&self) -> i32;

    fn stats(&self) -> &CommonSpeculativeImplStats;
    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats;

    /// `virtual void begin(llama_seq_id, const llama_tokens &)`
    /// (speculative.cpp:166)
    fn begin(&mut self, seq_id: i32, prompt: &[i32]);

    /// `virtual bool process(const llama_batch &)` (speculative.cpp:168).
    /// The port hands the target context in — draft-mtp reads its
    /// `embeddings_nextn` rows (speculative.cpp:1531-1534/:1591-1592), which
    /// the C reaches through the `params.ctx_tgt` pointer its impls hold.
    fn process(&mut self, batch: &LlamaBatch, ctx_tgt: &DecodeContext) -> bool;

    /// `virtual void draft(common_speculative_draft_params_vec &)`
    /// (speculative.cpp:170)
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        ctx_tgt: &mut DecodeContext,
    );

    /// `virtual void accept(llama_seq_id, uint16_t, bool is_other)`
    /// (speculative.cpp:172)
    fn accept(&mut self, seq_id: i32, n_accepted: u16, is_other: bool);

    /// port-only accessor for `params.ctx_dft`: the C's driver holds that
    /// pointer itself, but the port's impls own their context, so the driver
    /// reaches it here (only draft-simple has one — `get_state`/`set_state`
    /// dispatch the same way, speculative.cpp:2929-2951).
    fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        None
    }
}

// ---------------------------------------------------------------------------
// the draft-side sampler — a `common_sampler` built the way
// `common_speculative_impl_draft_simple` builds it (speculative.cpp:226-236):
// `common_params_sampling{ .no_perf = false, .top_k = 10,
// .samplers = { COMMON_SAMPLER_TYPE_TOP_K } }`, which `common_sampler_init`
// turns into the chain [logit_bias?, top_k(10), dist]
// (common/sampling.cpp:262-282).
// ---------------------------------------------------------------------------

/// `struct common_sampler` (common/sampling.h:20-45) restricted to what the
/// draft impl uses: the chain plus the candidate cache `cur_p` that
/// `common_sampler_get_candidates` sorts (common/sampling.cpp:731-754).
pub struct DraftSampler {
    chain: SamplerChain,
    /// `common_sampler::cur` — candidates of the last sample() call
    cur: Vec<TokenData>,
    /// `common_sampler::cur_p::selected`
    selected: i64,
    /// `common_sampler::cur_p::sorted`
    sorted: bool,

    /// the model's suppress tokens + vocab size — what `spec_retune`
    /// (speculative.cpp:33-71) needs to rebuild the chain through
    /// `common_sampler_init`'s logit-bias merge (common/sampling.cpp:329-344)
    n_vocab: i32,
    suppress: Vec<i32>,
}

impl DraftSampler {
    /// `common_sampler_init(llama_get_model(ctx_dft), params)`
    /// (speculative.cpp:227-235), with the model's suppress tokens merged into
    /// the leading logit_bias sampler like `common_sampler_init` does
    /// (common/sampling.cpp:262-275).
    pub fn new(vocab_dft: &Vocab) -> Self {
        let n_vocab = vocab_dft.n_tokens() as i32;
        let suppress = vocab_dft.get_suppress_tokens().to_vec();

        let mut chain = SamplerChain::new();

        // logit bias: user biases (none here) + model suppress tokens (-INFINITY)
        if !suppress.is_empty() {
            let biases: Vec<crate::sampling::LogitBias> = suppress
                .iter()
                .map(|&token| crate::sampling::LogitBias {
                    token,
                    bias: f32::NEG_INFINITY,
                })
                .collect();
            chain.add(init_logit_bias(n_vocab, &biases));
        }

        // `case COMMON_SAMPLER_TYPE_TOP_K: llama_sampler_init_top_k(params.top_k)`
        chain.add(init_top_k(10));
        // "default: sample from distribution" (common/sampling.cpp:282) — the
        // sampled token is discarded by the draft impl (which drafts the
        // argmax), but the dist sampler computes the probabilities the `p_min`
        // gate reads and advances the RNG exactly like the reference
        chain.add(init_dist(LLAMA_DEFAULT_SEED));

        DraftSampler {
            chain,
            cur: Vec::new(),
            selected: -1,
            sorted: false,
            n_vocab,
            suppress,
        }
    }

    /// `spec_retune` (speculative.cpp:33-71, upstream a7b94df2c) — rebuild the
    /// sequence's draft sampler at the target's temperature: rejection weighs
    /// q against p, so both have to sample alike. Only temp and seed carry
    /// over; the draft keeps its own top_k. The chain is the one
    /// `common_sampler_init` builds for
    /// `samplers = {TOP_K, TEMPERATURE}`: [logit_bias(suppress), top_k(10),
    /// temperature(temp), dist(seed)].
    pub fn retune(&mut self, temp: f32, seed: u32) {
        // "must be explicit, the default reseeds at random; mixed so it
        // differs from the target's" (speculative.cpp:54-55)
        let seed = if seed == LLAMA_DEFAULT_SEED {
            seed
        } else {
            seed ^ 0x85eb_ca6b
        };

        let mut chain = SamplerChain::new();

        if !self.suppress.is_empty() {
            let biases: Vec<crate::sampling::LogitBias> = self
                .suppress
                .iter()
                .map(|&token| crate::sampling::LogitBias {
                    token,
                    bias: f32::NEG_INFINITY,
                })
                .collect();
            chain.add(init_logit_bias(self.n_vocab, &biases));
        }
        // `sparams.top_k = 10` / `sparams.samplers = {TOP_K, TEMPERATURE}`
        chain.add(init_top_k(10));
        chain.add(crate::sampling::init_temp(temp));
        chain.add(init_dist(seed));

        self.chain = chain;
        self.cur.clear();
        self.selected = -1;
        self.sorted = false;
    }

    /// `common_sampler_sample(gsmpl, ctx_dft, i_batch, true)`
    /// (common/sampling.cpp:594-676) — the CPU path over one logits row:
    /// `set_logits` → `llama_sampler_apply(chain)` → the selected token, kept
    /// as the new candidate cache.
    pub fn sample(&mut self, logits: &[f32]) -> LlamaToken {
        let mut cur_p = TokenDataArray::from_logits(logits);

        self.chain.apply(&mut cur_p);

        assert!(
            cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
            "DraftSampler::sample: chain did not select a token"
        );

        let id = cur_p.data[cur_p.selected as usize].id;
        self.selected = cur_p.selected;
        self.sorted = cur_p.sorted;
        self.cur = cur_p.data[..cur_p.size].to_vec();

        id
    }

    /// `common_sampler_get_candidates(gsmpl, do_sort)`
    /// (common/sampling.cpp:731-754): sort by `p` descending, restoring the
    /// selected token's index. The C's `std::sort` is unstable; sorting by `p`
    /// preserves the argmax and only `data[0]` is read.
    pub fn get_candidates(&mut self, do_sort: bool) -> &[TokenData] {
        if do_sort && !self.sorted {
            let id = if self.selected >= 0 && (self.selected as usize) < self.cur.len() {
                Some(self.cur[self.selected as usize].id)
            } else {
                None
            };

            self.cur
                .sort_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(id) = id {
                for (i, d) in self.cur.iter().enumerate() {
                    if d.id == id {
                        self.selected = i as i64;
                        break;
                    }
                }
            }

            self.sorted = true;
        }

        &self.cur
    }

    /// `common_sampler_accept(gsmpl, id, true)`
    pub fn accept(&mut self, id: LlamaToken) {
        self.chain.accept(id);
    }

    /// `common_sampler_reset(gsmpl)` — `llama_sampler_chain_reset` + clear the
    /// candidate cache
    pub fn reset(&mut self) {
        self.chain.reset();
        self.cur.clear();
        self.selected = -1;
        self.sorted = false;
    }
}

// ---------------------------------------------------------------------------
// common_speculative_impl_draft_simple (speculative.cpp:179-389)
// ---------------------------------------------------------------------------

/// the temp/seed cache `std::vector<common_params_sampling> smpls_cfg` keeps
/// per sequence (speculative.cpp:229, upstream a7b94df2c) — the initial
/// sampler has no temperature (NAN), so no request may match the cache and
/// skip a rebuild (speculative.cpp:41-44)
#[derive(Clone, Copy, Debug)]
struct DraftSamplerCfg {
    temp: f32,
    seed: u32,
}

/// `spec_retune` (speculative.cpp:33-71, upstream a7b94df2c) — rebuild
/// seq_id's draft sampler at the target's temperature: rejection weighs q
/// against p, so both have to sample alike. Only temp and seed carry over;
/// the draft keeps its own top_k.
fn spec_retune(
    smpls: &mut [DraftSampler],
    cfg: &mut Vec<DraftSamplerCfg>,
    seq_id: usize,
    temp: f32,
    seed: u32,
) {
    if cfg.len() != smpls.len() {
        let n_old = cfg.len();
        cfg.resize(
            smpls.len(),
            DraftSamplerCfg {
                temp: f32::NAN,
                seed: 0,
            },
        );

        // the initial sampler has no temperature, so no request may match the
        // cache and skip a rebuild (speculative.cpp:41-44)
        for c in &mut cfg[n_old..] {
            c.temp = f32::NAN;
        }
    }

    let cur = &mut cfg[seq_id];

    if cur.temp == temp && cur.seed == seed {
        return;
    }

    cur.temp = temp;
    cur.seed = seed;

    smpls[seq_id].retune(cur.temp, cur.seed);
}

/// `struct common_speculative_impl_draft_simple` (speculative.cpp:179-389):
/// owns the draft context (`params.ctx_dft`) and one draft sampler per
/// sequence.
pub struct CommonSpeculativeImplDraftSimple {
    params: CommonParamsSpeculativeDraft,

    /// `params.ctx_dft` — a pointer in the C, owned here
    ctx_dft: DecodeContext,

    /// `llama_batch batch` reused by draft() (speculative.cpp:182, 207)
    batch: LlamaBatch,

    /// `std::vector<common_sampler_ptr> smpls` (speculative.cpp:184)
    smpls: Vec<DraftSampler>,

    /// `std::vector<common_params_sampling> smpls_cfg`
    /// (speculative.cpp:229, upstream a7b94df2c) — the retune cache; only
    /// temp/seed are read
    smpls_cfg: Vec<DraftSamplerCfg>,

    stats: CommonSpeculativeImplStats,
}

impl CommonSpeculativeImplDraftSimple {
    /// (speculative.cpp:186-252)
    pub fn new(
        params: &CommonParamsSpeculative,
        n_seq: u32,
        ctx_dft: DecodeContext,
        vocab_tgt: &Vocab,
        vocab_dft: &Vocab,
    ) -> Result<Self, String> {
        // `if (!ctx_dft) throw std::runtime_error("draft-simple requires a draft
        // context")` — the port takes the context by value, so the C's null
        // check lives in the caller (`Option<DecodeContext>`)

        // SPC_TRC of speculative.cpp:197-198
        spec_dbg!(
            "spec draft-simple: adding speculative implementation 'draft-simple' \
             (n_max={}, n_min={}, p_min={})",
            params.draft.n_max,
            params.draft.n_min,
            params.draft.p_min
        );

        // `batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1)`
        // (speculative.cpp:207) — the port's batch grows on demand
        let batch = LlamaBatch::default();

        // `smpls.resize(n_seq)` + one sampler per sequence
        // (speculative.cpp:226-236)
        let mut smpls = Vec::with_capacity(n_seq as usize);
        for _ in 0..n_seq {
            smpls.push(DraftSampler::new(vocab_dft));
        }

        // `common_speculative_are_compatible(llama_get_model(ctx_tgt),
        // llama_get_model(ctx_dft))` (speculative.cpp:238-245)
        let vocab_cmpt = common_speculative_are_compatible(vocab_tgt, vocab_dft);
        spec_dbg!("spec draft-simple: vocab_cmpt = {vocab_cmpt}");

        if !vocab_cmpt {
            eprintln!("spec draft-simple: the target and draft vocabs are not compatible");
            return Err("draft model vocab type must match target model to use speculation".into());
        }

        // the C also asserts `n_seq == llama_n_seq_max(ctx_dft)`
        // (speculative.cpp:247-251); the port's DecodeContext has no
        // per-context sequence cap (its KV is `LLAMA_MAX_SEQ` wide), so the
        // check is dropped.

        Ok(CommonSpeculativeImplDraftSimple {
            params: params.draft.clone(),
            ctx_dft,
            batch,
            smpls,
            smpls_cfg: Vec::new(),
            stats: CommonSpeculativeImplStats::default(),
        })
    }

    /// the C's `params.ctx_dft` pointer, for the driver's KV bookkeeping
    pub fn ctx_dft(&mut self) -> &mut DecodeContext {
        &mut self.ctx_dft
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplDraftSimple {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::DraftSimple
    }

    fn n_max(&self) -> i32 {
        self.params.n_max
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/)
    /// override` (speculative.cpp:299-302, upstream a7b94df2c): reset here
    /// rather than per round, or two identical requests differ.
    fn begin(&mut self, seq_id: i32, _prompt: &[i32]) {
        self.smpls[seq_id as usize].reset();
    }

    /// `bool process(const llama_batch & batch) override`
    /// (speculative.cpp:262-277): decode the batch on the draft context with
    /// the logits disabled (`batch_dft.logits = nullptr`). The target context
    /// is unused by draft-simple.
    fn process(&mut self, batch: &LlamaBatch, _ctx_tgt: &DecodeContext) -> bool {
        let mut batch_dft = batch.clone();
        batch_dft.logits = None;

        if let Err(e) = self.ctx_dft.decode_batch(&batch_dft) {
            eprintln!("spec draft-simple: failed to decode draft batch, ret = {e}");

            return false;
        }
        // port-only accounting (see CommonSpeculativeImplStats::n_forward)
        self.stats.n_forward += 1;

        true
    }

    fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        Some(&mut self.ctx_dft)
    }

    /// `void draft(common_speculative_draft_params_vec & dparams) override`
    /// (speculative.cpp:279-445, probabilistic-drafting additions of
    /// a7b94df2c at :365-381/:403-440).
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        let n_seq = dparams.len();

        self.batch.clear();

        // keep track of which sequences are still drafting
        let mut n_drafting = 0;
        let mut drafting = vec![false; n_seq];

        for seq_id in 0..n_seq {
            let dp = &dparams[seq_id];

            if !dp.drafting {
                continue;
            }

            n_drafting += 1;
            drafting[seq_id] = true;

            // greedy drafting leaves no candidates behind, so the verifier
            // falls back to sample-and-match (speculative.cpp:368-370)
            if !self.params.probabilistic {
                dparams[seq_id].result_q = None;
            }

            // result_q is only set when the caller wants rejection, so it
            // also gates the retune (:373-375)
            if dparams[seq_id].result_q.is_some() {
                let (temp, seed) = (dparams[seq_id].temp, dparams[seq_id].seed);
                spec_retune(
                    &mut self.smpls,
                    &mut self.smpls_cfg,
                    seq_id,
                    temp,
                    seed,
                );
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            // (:378-380)
            if dparams[seq_id].result_q.is_none() {
                self.smpls[seq_id].reset();
            }

            let dp = &dparams[seq_id];
            self.batch.add(dp.id_last, dp.pos0, &[seq_id as i32], true);
        }

        // `llama_decode(ctx_dft, batch)` — the seed step, logits of dp.id_last
        // (speculative.cpp:302-306)
        let mut seed_out = match self.ctx_dft.decode_batch(&self.batch) {
            Ok(out) => out,
            Err(_) => {
                eprintln!("spec draft-simple: llama_decode returned error");
                return;
            }
        };
        // port-only accounting (see CommonSpeculativeImplStats::n_forward)
        self.stats.n_forward += 1;

        let mut i = 0;

        while n_drafting > 0 {
            let mut i_batch = 0;

            self.batch.clear();

            for seq_id in 0..n_seq {
                if !drafting[seq_id] {
                    continue;
                }

                // `const llama_token id_sampled =
                // common_sampler_sample(smpl, ctx_dft, i_batch, true)`
                // (speculative.cpp:406) — kept: probabilistic drafting drafts
                // the sampled token, greedy drafting drafts the argmax
                let logits = seed_out
                    .logits_ith(i_batch)
                    .expect("draft seed logits")
                    .to_vec();
                let id_sampled = self.smpls[seq_id].sample(&logits);
                i_batch += 1;

                // `cur_p = common_sampler_get_candidates(smpl, true)`
                // (speculative.cpp:409-415): the drafted token is the argmax
                // (`cur_p->data[0].id`) unless the drafter samples
                // (`dparams.at(seq_id).result_q ? id_sampled : ...`, :418)
                let (id, p_top, candidates) = {
                    let cur_p = self.smpls[seq_id].get_candidates(true).to_vec();

                    for (k, cand) in cur_p.iter().take(3).enumerate() {
                        spec_dbg!(
                            "spec draft-simple: - seq_id {seq_id}, draft candidate {k:3}, \
                             pos {i:3}: {:6} ({:8.3})",
                            cand.id,
                            cand.p
                        );
                    }

                    let id = if dparams[seq_id].result_q.is_some() {
                        id_sampled
                    } else {
                        cur_p[0].id
                    };

                    (id, cur_p[0].p, cur_p)
                };

                // only collect very high-confidence draft tokens
                // (speculative.cpp:421-427)
                if p_top < self.params.p_min {
                    drafting[seq_id] = false;
                    n_drafting -= 1;

                    continue;
                }

                self.smpls[seq_id].accept(id);

                let dp = &mut dparams[seq_id];
                let pos0 = dp.pos0;
                let dp_n_max = dp.n_max;

                dp.result.push(id);

                // `dp.result_q->emplace_back(cur_p->data, cur_p->data +
                // cur_p->size)` (speculative.cpp:435-437) — the candidate
                // distribution of this drafted token
                if let Some(rq) = dp.result_q.as_mut() {
                    rq.push(candidates);
                }

                if (self.params.n_max <= dp.result.len() as i32)
                    || (dp_n_max > 0 && dp_n_max <= dp.result.len() as i32)
                {
                    drafting[seq_id] = false;
                    n_drafting -= 1;
                    continue;
                }

                self.batch.add(id, pos0 + i + 1, &[seq_id as i32], true);
            }

            if self.batch.token.is_empty() {
                break;
            }

            // evaluate the drafted tokens on the draft model
            // (speculative.cpp:365-370)
            match self.ctx_dft.decode_batch(&self.batch) {
                Ok(out) => seed_out = out,
                Err(_) => {
                    eprintln!("spec draft-simple: llama_decode[{i}] returned error");
                    break;
                }
            }
            // port-only accounting (see CommonSpeculativeImplStats::n_forward)
            self.stats.n_forward += 1;

            i += 1;
        }

        // drop drafts shorter than n_min (speculative.cpp:375-383)
        for dp in dparams.iter_mut() {
            if !dp.drafting {
                continue;
            }

            if dp.result.len() < self.params.n_min as usize {
                dp.result.clear();
            }
        }
    }

    /// `void accept(...) override { // noop }` (speculative.cpp:386-388)
    fn accept(&mut self, _seq_id: i32, _n_accepted: u16, _is_other: bool) {}
}

// ---------------------------------------------------------------------------
// common_speculative_impl_draft_mtp (speculative.cpp:1330-1767)
// ---------------------------------------------------------------------------

/// `struct common_speculative_impl_draft_mtp` (speculative.cpp:1330-1767):
/// the MTP/nextn draft driver over the *target model's own* MTP block — the
/// draft context is the `LLAMA_CONTEXT_TYPE_MTP` second context
/// `common_speculative_init_from_params` creates (speculative.cpp:2545-2589),
/// and the drafts pair every token with the target's `h_nextn` row of the
/// *previous* position (the `batch.embd` rows of :1521-1547).
///
/// Ported modes:
///   * the "neither" mode of :1345-1348 — one trained MTP head with its own
///     KV (deepseek2/deepseek32/deepseek4, all `n_layer_nextn == 1`);
///   * `is_mem_shared` (:1344/:1423/:1717-1721, the gemma4 assistants) — the
///     head is attached to the *target* context
///     ([`crate::context::DecodeContext::attach_gemma4_assistant`], the
///     port's `ctx_other == ctx_tgt`) and every draft step decodes over the
///     shared target KV at the *same* position `pos0`.
/// Not ported:
///   * `chain_heads` (step35's `n_mtp_layers > 1`, one head per draft step
///     through `llama_set_nextn_layer_offset`, :1348/:1553-1561/:1643-1651/
///     :1706-1716) — unreachable for the deepseek family (the graph_mtp
///     classes assert `n_layer_nextn == 1`, deepseek2.cpp:173 /
///     deepseek32.cpp:498 / deepseek4.cpp:1366);
///   * `backend_sampling` (the sampler offloaded onto the draft context's
///     backend, :1404-1418) — the port is CPU-only, the CPU chain of :1396-
///     1402 is what runs.
pub struct CommonSpeculativeImplDraftMtp {
    params: CommonParamsSpeculativeDraft,

    /// the draft decode backend — the ported MTP context, or the gemma4
    /// mem-shared mode (`is_mem_shared`, :1423) whose head rides the target
    dft: MtpDraftCtx,

    /// `llama_batch batch` (speculative.cpp:1333) — token ids **and** the
    /// previous position's h rows (`batch.embd`, :1390-1394)
    batch: LlamaBatch,

    /// `std::vector<common_sampler_ptr> smpls` (:1335)
    smpls: Vec<DraftSampler>,

    /// `std::vector<common_params_sampling> smpls_cfg`
    /// (speculative.cpp:1397, upstream a7b94df2c) — the retune cache
    smpls_cfg: Vec<DraftSamplerCfg>,

    /// `int32_t n_embd = llama_model_n_embd_out(...)` (:1340)
    n_embd: usize,
    /// `int32_t n_mtp_layers = max(1, llama_model_n_layer_nextn(...))`
    /// (:1346) — 1 for every ported arch
    n_mtp_layers: i32,

    /// `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` (:1423) —
    /// the gemma4 assistants: no catch-up decode in process() (:1519), every
    /// draft step at `pos0` (:1718-1722)
    is_mem_shared: bool,

    /// per-sequence cross-batch carryover: the (h_p, x_{p+1}) pair's h half
    /// (:1350-1353) — `[n_seq][n_embd]`
    pending_h: Vec<Vec<f32>>,

    /// first/last batch index of each sequence in the last process() batch
    /// (:1355-1356)
    i_batch_beg: Vec<i32>,
    i_batch_end: Vec<i32>,

    /// the target h rows of the last verification batch, per seq (:1358-1361)
    verify_h: Vec<Vec<f32>>,
    verify_h_rows: Vec<i32>,

    /// `std::vector<int> i_last` (:1363) — the row of the newest token
    i_last: Vec<i32>,

    stats: CommonSpeculativeImplStats,
}

/// the draft decode backend of [`CommonSpeculativeImplDraftMtp`]
enum MtpDraftCtx {
    /// the "neither" mode — a ported MTP context with its own KV
    Own(DecodeContext),
    /// `is_mem_shared` (:1423) — the gemma4-assistant head attached to the
    /// target context; each draft step runs
    /// [`DecodeContext::decode_gemma4_assistant`] over the shared KV
    Gemma4Shared,
}

impl MtpDraftCtx {
    fn as_own(&mut self) -> Option<&mut DecodeContext> {
        match self {
            MtpDraftCtx::Own(ctx) => Some(ctx),
            MtpDraftCtx::Gemma4Shared => None,
        }
    }
}

/// one draft decode's outputs — the own-KV context's `BatchOutput` or the
/// gemma4 shared step's logits/h_next rows
enum StepOut {
    Own(crate::context::BatchOutput),
    Shared(crate::gemma4_assistant::Gemma4AssistantStep),
}

impl CommonSpeculativeImplDraftMtp {
    /// (speculative.cpp:1366-1444)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: &CommonParamsSpeculative,
        n_seq: u32,
        ctx_dft: DecodeContext,
        ctx_tgt: &DecodeContext,
        n_layer_nextn: u32,
        vocab_dft: &Vocab,
    ) -> Result<Self, String> {
        // `GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and
        // ctx_dft to be set")` (:1372) — both are owned/reachable here
        //
        // `n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft))` +
        // the width-match assert (:1374-1376)
        let n_embd = ctx_dft.n_embd_out();
        assert!(
            n_embd > 0,
            "draft-mtp: the draft context is not an MTP context"
        );
        let n_embd_tgt = ctx_tgt.n_embd_out();
        if n_embd != n_embd_tgt {
            return Err(format!(
                "draft-mtp: MTP input row width must match the target h_nextn width ({n_embd} != \
                 {n_embd_tgt})"
            ));
        }

        let n_mtp_layers = 1i32.max(n_layer_nextn as i32);

        spec_dbg!(
            "spec draft-mtp: adding speculative implementation 'draft-mtp' (n_max={}, n_min={}, \
             p_min={:.2}, n_embd={n_embd}, backend_sampling={})",
            params.draft.n_max,
            params.draft.n_min,
            params.draft.p_min,
            params.draft.backend_sampling,
        );

        // `llama_batch_init(n_b, embd = n_embd, 1)` + the token malloc
        // (:1389-1394) — the port's batch grows on demand
        let batch = LlamaBatch::default();

        // one draft sampler per sequence (:1396-1402) — the CPU chain even
        // with backend_sampling on (see the struct docs)
        let mut smpls = Vec::with_capacity(n_seq as usize);
        for _ in 0..n_seq {
            smpls.push(DraftSampler::new(vocab_dft));
        }

        // `llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false)` +
        // `(ctx_dft, true, /*masked*/ true)` (:1420-1421) — the port applies
        // the taps where the contexts are assembled
        // (`common_speculative_init`), since the impl cannot reach the
        // target mutably from here.

        // `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` (:1423)
        // is false for the deepseek family (see the struct docs);
        // `chain_heads = n_mtp_layers > 1 && !is_mem_shared` (:1424) would
        // then clamp n_max to the head count (:1426-1433) — unreachable
        if n_mtp_layers > 1 {
            return Err(
                "draft-mtp: chained MTP heads (n_layer_nextn > 1, the step35 mode of \
                 speculative.cpp:1643-1651) are not ported"
                    .into(),
            );
        }

        Ok(CommonSpeculativeImplDraftMtp {
            params: params.draft.clone(),
            dft: MtpDraftCtx::Own(ctx_dft),
            batch,
            smpls,
            smpls_cfg: Vec::new(),
            n_embd,
            n_mtp_layers,
            is_mem_shared: false,
            pending_h: vec![vec![0.0; n_embd]; n_seq as usize],
            i_last: vec![-1; n_seq as usize],
            i_batch_beg: vec![-1; n_seq as usize],
            i_batch_end: vec![-1; n_seq as usize],
            verify_h: (0..n_seq as usize).map(|_| Vec::new()).collect(),
            verify_h_rows: vec![0; n_seq as usize],
            stats: CommonSpeculativeImplStats::default(),
        })
    }

    /// the gemma4-assistant construction — `is_mem_shared` mode (:1423): the
    /// head has no context of its own (the draft graph runs on the target's,
    /// where the CLI attached it), the samplers still draft from the head's
    /// tied vocabulary. `n_embd` is the shared backbone width (the target's
    /// `h_nextn` row width == the head's `embedding_length_out`).
    pub fn new_gemma4_shared(
        params: &CommonParamsSpeculative,
        n_seq: u32,
        n_embd: usize,
        vocab_dft: &Vocab,
    ) -> Self {
        spec_dbg!(
            "spec draft-mtp: adding speculative implementation 'draft-mtp' (gemma4-assistant, \
             shared target KV; n_max={}, n_min={}, p_min={:.2}, n_embd={n_embd})",
            params.draft.n_max,
            params.draft.n_min,
            params.draft.p_min,
        );
        let mut smpls = Vec::with_capacity(n_seq as usize);
        for _ in 0..n_seq {
            smpls.push(DraftSampler::new(vocab_dft));
        }
        CommonSpeculativeImplDraftMtp {
            params: params.draft.clone(),
            dft: MtpDraftCtx::Gemma4Shared,
            batch: LlamaBatch::default(),
            smpls,
            smpls_cfg: Vec::new(),
            n_embd,
            n_mtp_layers: 1,
            is_mem_shared: true,
            pending_h: vec![vec![0.0; n_embd]; n_seq as usize],
            i_last: vec![-1; n_seq as usize],
            i_batch_beg: vec![-1; n_seq as usize],
            i_batch_end: vec![-1; n_seq as usize],
            verify_h: (0..n_seq as usize).map(|_| Vec::new()).collect(),
            verify_h_rows: vec![0; n_seq as usize],
            stats: CommonSpeculativeImplStats::default(),
        }
    }

    /// the C's `params.ctx_dft` pointer, for the driver's KV bookkeeping
    pub fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        match &mut self.dft {
            MtpDraftCtx::Own(ctx) => Some(ctx),
            // the gemma4 mem-shared mode has no draft context — its KV IS
            // the target's
            MtpDraftCtx::Gemma4Shared => None,
        }
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplDraftMtp {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::DraftMtp
    }

    fn n_max(&self) -> i32 {
        self.params.n_max
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(...)` (speculative.cpp:1466-1482): the pos_max sanity
    /// warning — if the draft context never saw the prompt, process() hooks
    /// may have missed ubatches and drafts degrade (the SPC_WRN prints
    /// unconditionally in the C, `&& !is_mem_shared` at :1480).
    fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        // reset here rather than per round, or two identical requests differ
        // (speculative.cpp:1519-1521, upstream a7b94df2c)
        self.smpls[seq_id as usize].reset();

        let n = prompt.len() as i32;
        if n <= 0 {
            return;
        }

        // `pos_max < N - 1 && !is_mem_shared` (:1480) — the gemma4 shared
        // mode has no draft memory of its own, nothing to check
        if self.is_mem_shared {
            return;
        }
        let pos_max = match &mut self.dft {
            MtpDraftCtx::Own(ctx) => ctx.seq_pos_max(seq_id),
            MtpDraftCtx::Gemma4Shared => return,
        };
        if pos_max < n - 1 {
            eprintln!(
                "spec draft-mtp: ctx_dft pos_max={pos_max} < N-1={} - process() hook may not have \
                 run on every prefill ubatch (need_embd / logits=1 on every prompt position?). \
                 Drafts may degrade.",
                n - 1
            );
        }
    }

    /// `bool process(const llama_batch & batch_in)` (speculative.cpp:1484-
    /// 1600): decode the batch on the draft context — every token paired
    /// with the target's h row of the *previous* position — then stash the
    /// target's h rows of the batch (verify_h) and the last one (pending_h).
    /// The gemma4 shared mode skips the catch-up decode entirely (:1519
    /// `if (!is_mem_shared)` wraps the whole decode block — the KV lives in
    /// the target) and only harvests the h rows.
    fn process(&mut self, batch_in: &LlamaBatch, ctx_tgt: &DecodeContext) -> bool {
        let n_tokens = batch_in.token.len();
        if n_tokens == 0 {
            return true;
        }

        // TODO of the C kept: vision batches (embd input) are skipped
        // (:1489-1492)
        if batch_in.embd.is_some() {
            return true;
        }

        // remember the first and last batch index for each sequence
        // (:1496-1511)
        self.i_batch_beg.fill(-1);
        self.i_batch_end.fill(-1);
        let seq_ids = batch_in.seq_id.as_ref().expect("MTP process: batch seq_id");
        let pos = batch_in.pos.as_ref().expect("MTP process: batch pos");
        for (k, ids) in seq_ids.iter().enumerate() {
            assert_eq!(
                ids.len(),
                1,
                "batch_in.n_seq_id[k] == 1 (speculative.cpp:1502)"
            );
            let seq_id = ids[0] as usize;
            if seq_id < self.i_batch_end.len() {
                self.i_batch_end[seq_id] = k as i32;
                if self.i_batch_beg[seq_id] < 0 {
                    self.i_batch_beg[seq_id] = k as i32;
                }
            }
        }

        let n_embd = self.n_embd;

        // the catch-up decode — skipped when the KV is shared with the
        // target (gemma4, :1519 `if (!is_mem_shared)`); the own-KV archs run it
        if !self.is_mem_shared {
            self.batch.clear();

            for k in 0..n_tokens {
                self.batch
                    .add(batch_in.token[k], pos[k], &[seq_ids[k][0]], false);
            }

            // shift the tgt embeddings to the right by one position
            // (:1526-1534): embd rows 1..n take the target's h rows 0..n-1
            // (the tokens are assumed sequential per sequence — the C's
            // comment at :1527-1530)
            let h_tgt = ctx_tgt.get_embeddings_nextn();
            let mut embd = vec![0f32; n_tokens * n_embd];
            embd[n_embd..].copy_from_slice(&h_tgt[..(n_tokens - 1) * n_embd]);

            // the pending h of a previous run pairs with the first token of
            // each sequence (:1536-1547)
            for seq_id in 0..self.i_batch_beg.len() {
                let beg = self.i_batch_beg[seq_id];
                if beg < 0 {
                    continue;
                }
                embd[beg as usize * n_embd..(beg as usize + 1) * n_embd]
                    .copy_from_slice(&self.pending_h[seq_id]);
            }
            self.batch.embd = Some(embd);

            // the per-head loop (:1552-1571) runs once — chain_heads false —
            // and the layer-offset restore (:1573-1575) is chain-heads-only
            let MtpDraftCtx::Own(ctx_dft) = &mut self.dft else {
                unreachable!("the catch-up decode runs only in the own-KV mode");
            };
            if let Err(e) = ctx_dft.decode_batch(&self.batch) {
                eprintln!(
                    "spec draft-mtp: llama_decode(ctx_dft) failed ({e}) (pos={})",
                    pos[0]
                );
                return false;
            }
            // port-only accounting (see CommonSpeculativeImplStats::n_forward)
            self.stats.n_forward += 1;
        }

        // harvest the target's h rows of this batch (:1581-1597): verify_h
        // rows 0..n_rows-1 in batch order, pending_h = the last row
        for seq_id in 0..self.i_batch_end.len() {
            let (beg, end) = (self.i_batch_beg[seq_id], self.i_batch_end[seq_id]);
            if end < 0 {
                continue;
            }

            let n_rows = (end - beg + 1) as usize;
            self.verify_h_rows[seq_id] = n_rows as i32;
            self.verify_h[seq_id].resize(n_rows * n_embd, 0.0);
            for i in 0..n_rows {
                self.verify_h[seq_id][i * n_embd..(i + 1) * n_embd]
                    .copy_from_slice(ctx_tgt.get_embeddings_nextn_ith(beg + i as i32));
            }
            let last = (n_rows - 1) * n_embd;
            self.pending_h[seq_id].copy_from_slice(&self.verify_h[seq_id][last..last + n_embd]);
        }

        let _ = self.n_mtp_layers; // 1 — asserted at construction

        true
    }

    fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        Self::ctx_dft(self)
    }

    /// `void draft(common_speculative_draft_params_vec &)` (speculative.cpp:
    /// 1602-1751) — the single-head growing-KV loop: seed with
    /// (id_last, pending_h), decode, take the argmax candidate while its p
    /// clears `p_min`, and feed (id, h_row) forward. The gemma4 shared mode
    /// decodes every step at `pos0` over the target's KV (:1717-1722).
    fn draft(&mut self, dparams: &mut [CommonSpeculativeDraftParams], ctx_tgt: &mut DecodeContext) {
        let n_seq = dparams.len();

        self.batch.clear();

        // keep track of which sequences are still drafting
        let mut n_drafting = 0usize;
        let mut drafting = vec![false; n_seq];

        let n_embd = self.n_embd;

        for seq_id in 0..n_seq {
            let dp = &dparams[seq_id];
            if !dp.drafting {
                continue;
            }

            n_drafting += 1;
            drafting[seq_id] = true;

            // greedy drafting leaves no candidates behind, so the verifier
            // falls back to sample-and-match (speculative.cpp:1666-1668)
            if !self.params.probabilistic {
                dparams[seq_id].result_q = None;
            }

            // result_q is only set when the caller wants rejection, so it
            // also gates the retune (:1671-1673)
            if dparams[seq_id].result_q.is_some() {
                let (temp, seed) = (dparams[seq_id].temp, dparams[seq_id].seed);
                spec_retune(
                    &mut self.smpls,
                    &mut self.smpls_cfg,
                    seq_id,
                    temp,
                    seed,
                );
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            // (:1676-1678)
            if dparams[seq_id].result_q.is_none() {
                self.smpls[seq_id].reset();
            }

            let dp = &dparams[seq_id];
            self.batch.add(dp.id_last, dp.pos0, &[seq_id as i32], true);
            let at = self.batch.token.len() - 1;
            self.batch
                .embd
                .get_or_insert_with(|| vec![0.0; (at + 1) * n_embd])
                .resize((at + 1) * n_embd, 0.0);
            self.batch.embd.as_mut().unwrap()[at * n_embd..(at + 1) * n_embd]
                .copy_from_slice(&self.pending_h[seq_id]);
            self.i_last[seq_id] = at as i32;
        }

        let mut i = 0i32;

        while n_drafting > 0 {
            // chain_heads' per-step seq_rm + layer switch (:1643-1651) is
            // not ported — the single head keeps its KV across steps
            //
            // the two backends: the own-KV decode (:1675-1677) or the gemma4
            // mem-shared step over the target's KV
            let step_out: StepOut = if self.is_mem_shared {
                match ctx_tgt.decode_gemma4_assistant(&self.batch) {
                    Ok(s) => StepOut::Shared(s),
                    Err(e) => {
                        eprintln!("spec draft-mtp: llama_decode[{i}] returned {e}");
                        break;
                    }
                }
            } else {
                let MtpDraftCtx::Own(ctx_dft) = &mut self.dft else {
                    unreachable!();
                };
                match ctx_dft.decode_batch(&self.batch) {
                    Ok(o) => StepOut::Own(o),
                    Err(e) => {
                        eprintln!("spec draft-mtp: llama_decode[{i}] returned {e}");
                        break;
                    }
                }
            };
            // port-only accounting (see CommonSpeculativeImplStats::n_forward)
            self.stats.n_forward += 1;

            // rebuild the batch for the next step: only the new token is
            // re-added (the KV already holds the prefix)
            self.batch.clear();

            for seq_id in 0..n_seq {
                if !drafting[seq_id] {
                    continue;
                }

                // `const llama_token id_sampled =
                // common_sampler_sample(smpl, ctx_dft, i_last, true)`
                // (:1728) — kept: probabilistic drafting drafts the sampled
                // token, greedy drafting drafts the argmax (:1740)
                let i_last = self.i_last[seq_id] as usize;
                let logits = match &step_out {
                    StepOut::Own(o) => o
                        .logits_ith(self.i_last[seq_id])
                        .expect("draft-mtp: draft logits row")
                        .to_vec(),
                    StepOut::Shared(s) => s.logits[i_last * s.n_vocab()..(i_last + 1) * s.n_vocab()]
                        .to_vec(),
                };
                let id_sampled = self.smpls[seq_id].sample(&logits);
                // `llama_get_embeddings_nextn_ith(ctx_dft, i_last)` (:1672)
                let h_row = match &step_out {
                    StepOut::Own(_) => self
                        .dft
                        .as_own()
                        .expect("own-KV step")
                        .get_embeddings_nextn_ith(self.i_last[seq_id])
                        .to_vec(),
                    StepOut::Shared(s) => s.h_row(i_last),
                };

                // `cur_p = common_sampler_get_candidates(smpl, true)`
                // (:1731) + the drafted token: the argmax (`cur_p->data[0].id`)
                // unless the drafter samples (`dparams.at(seq_id).result_q ?
                // id_sampled : ...`, :1740)
                let (id, p_top, candidates) = {
                    let cur_p = self.smpls[seq_id].get_candidates(true).to_vec();
                    for (k, cand) in cur_p.iter().take(3).enumerate() {
                        spec_dbg!(
                            "spec draft-mtp: - seq_id {seq_id}, draft candidate {k:3}, pos {i:3}: \
                             {:6} ({:8.3})",
                            cand.id,
                            cand.p
                        );
                    }
                    let id = if dparams[seq_id].result_q.is_some() {
                        id_sampled
                    } else {
                        cur_p[0].id
                    };
                    (id, cur_p[0].p, cur_p)
                };

                // only collect very high-confidence draft tokens (:1685-1691)
                if p_top < self.params.p_min {
                    drafting[seq_id] = false;
                    n_drafting -= 1;
                    continue;
                }

                self.smpls[seq_id].accept(id);

                let dp = &mut dparams[seq_id];
                let pos0 = dp.pos0;

                dp.result.push(id);

                // `dp.result_q->emplace_back(cur_p->data, cur_p->data +
                // cur_p->size)` (speculative.cpp:1757-1759)
                if let Some(rq) = dp.result_q.as_mut() {
                    rq.push(candidates);
                }

                if self.params.n_max <= dp.result.len() as i32 {
                    drafting[seq_id] = false;
                    n_drafting -= 1;
                    continue;
                }

                // the step feed: the growing-KV arm re-adds only the new
                // token at pos0 + i + 1 paired with the h row just read
                // (:1722-1724); the gemma4 shared arm re-adds it at the SAME
                // pos0 (:1718-1722 — "we use the same position for all draft
                // tokens", the head attends the target's KV, not its own)
                let feed_pos = if self.is_mem_shared {
                    pos0
                } else {
                    pos0 + i + 1
                };
                self.batch.add(id, feed_pos, &[seq_id as i32], true);
                let at = self.batch.token.len() - 1;
                self.batch
                    .embd
                    .get_or_insert_with(|| vec![0.0; (at + 1) * n_embd])
                    .resize((at + 1) * n_embd, 0.0);
                self.batch.embd.as_mut().unwrap()[at * n_embd..(at + 1) * n_embd]
                    .copy_from_slice(&h_row);

                self.i_last[seq_id] = at as i32;
            }

            if self.batch.token.is_empty() {
                break;
            }

            i += 1;
        }

        // drop drafts shorter than n_min (:1741-1750)
        for dp in dparams.iter_mut() {
            if !dp.drafting {
                continue;
            }
            if dp.result.len() < self.params.n_min as usize {
                dp.result.clear();
            }
        }
    }

    /// `void accept(...)` (speculative.cpp:1753-1766): re-anchor pending_h
    /// on the h row of the last accepted position.
    fn accept(&mut self, seq_id: i32, n_accepted: u16, _is_other: bool) {
        if seq_id < 0 || seq_id as usize >= self.verify_h_rows.len() {
            return;
        }

        let n_rows = self.verify_h_rows[seq_id as usize];
        if n_rows <= 0 {
            return;
        }

        let i_h = (n_accepted as i32).min(n_rows - 1) as usize;
        let n_embd = self.n_embd;
        self.pending_h[seq_id as usize]
            .copy_from_slice(&self.verify_h[seq_id as usize][i_h * n_embd..(i_h + 1) * n_embd]);
    }
}

// ---------------------------------------------------------------------------
// common_speculative_impl_draft_eagle3 (speculative.cpp:426-907)
// ---------------------------------------------------------------------------

/// `struct common_speculative_impl_draft_eagle3` (speculative.cpp:426-907):
/// the EAGLE3 driver over a separate eagle-head model — the draft context is
/// the eagle3 head context (`common_speculative_init_from_params`'s has_draft
/// arm, speculative.cpp:2553-2576, with `cparams.ctx_other = ctx_tgt`),
/// carrying BOTH the encoder (`llama_encode`, the head's `graph<true>`) and
/// the one-layer decoder (`llama_decode`, `graph<false>`).
///
/// The EAGLE3 input convention (the C's block comment, :392-425): at draft
/// memory pos P the decoder input pair is `(token[P+1], g_embd[P])` — every
/// g row is one position *behind* its token. Each `process()` call runs the
/// encoder over the target's fused extract-layer inputs to make the g rows,
/// then trains the decoder KV on all positions except the last, which is
/// *deferred* (the pair's token lives outside the batch) and completed by the
/// next `process()` (the cross-ubatch bridge) or `draft()` (dp.id_last, the
/// target's freshest sample). `accept()` rebases the deferred boundary onto
/// the last accepted row of the stashed verify snapshot.
///
/// Not ported (see the struct docs of [`CommonSpeculativeImplDraftMtp`] for
/// the shared pieces): `backend_sampling` (:497-511 — the port is CPU-only,
/// the CPU sampler chain of :489-495 runs) and the `get_state`/`set_state`
/// boundary stash (:865-906 — only needed for recurrent/hybrid targets whose
/// checkpoints drop the g row; the ported tap targets are full-attention).
pub struct CommonSpeculativeImplDraftEagle3 {
    params: CommonParamsSpeculativeDraft,

    /// `params.ctx_dft` — the eagle3 head context (owned, like draft-simple)
    ctx_dft: DecodeContext,

    /// `std::vector<common_sampler_ptr> smpls` (:430) — the top_k(10) chains
    /// of :488-495
    smpls: Vec<DraftSampler>,

    /// `int32_t n_embd_dec` (:435) — the head's hidden size
    n_embd_dec: usize,
    /// `int32_t n_embd_enc` (:436) — target_layer_ids_n * target_hidden_size
    n_embd_enc: usize,
    /// `int32_t n_embd_tgt` (:437) — the target model's hidden size (from the
    /// head's `eagle3.target_hidden_size`; equals llama_model_n_embd(model_tgt)
    /// of :477 in any working pair)
    n_embd_tgt: usize,
    /// `int32_t n_layer_tgt` (:438) — the target model's layer count
    n_layer_tgt: usize,

    /// `target_layer_ids` (:440) — exactly 3 (asserted by the constructor,
    /// :472-475)
    target_layer_ids: Vec<i32>,

    /// [per-seq] deferred boundary state (:443-445): the (g, pos) of the last
    /// training position of the newest process() call
    pending_g_last: Vec<Vec<f32>>,
    pending_pos_last: Vec<i32>,

    /// [per-seq] snapshot of the most recent process()'s encoder output
    /// (:447-450) — `verify_g[seq][row * n_embd_dec ..]`
    verify_g: Vec<Vec<f32>>,
    /// pos of verify_g[seq][0] (:449)
    verify_pos_first: Vec<i32>,
    /// number of rows (:450)
    verify_g_rows: Vec<i32>,

    /// scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    /// (:453) and the encoder output [n_tokens, n_embd_dec] (:454)
    features_buf: Vec<f32>,
    g_embd_buf: Vec<f32>,

    /// `llama_batch batch` (:428) — token ids **and** g rows (`batch.embd`)
    batch: LlamaBatch,

    stats: CommonSpeculativeImplStats,
}

impl CommonSpeculativeImplDraftEagle3 {
    /// (speculative.cpp:456-534). The context-level taps
    /// (`llama_set_embeddings_layer_inp(ctx_tgt, ...)` of :513-522 and the
    /// two `llama_set_embeddings_nextn` of :524-526) are applied by
    /// [`common_speculative_init`] — it holds the target context mutably.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: &CommonParamsSpeculative,
        n_seq: u32,
        ctx_dft: DecodeContext,
        ctx_tgt: &DecodeContext,
        target_layer_ids: Vec<i32>,
        n_embd_tgt: usize,
        vocab_dft: &Vocab,
    ) -> Result<Self, String> {
        // `target_layer_ids_n != 3` → runtime_error (:472-475); the head
        // loader already enforces exactly 3 (eagle3.cpp:9-11)
        if target_layer_ids.len() != 3 {
            return Err(format!(
                "draft model is not eagle3 (expected 3 extract layers, got {})",
                target_layer_ids.len()
            ));
        }

        // the dims of :477-480. n_embd_dec/n_embd_enc come from the head
        // context (hparams.n_embd / n_embd_inp_enc); n_layer_tgt from the
        // target context (llama_model_n_layer(model_tgt)).
        let eagle = ctx_dft
            .eagle
            .as_ref()
            .expect("draft-eagle3: the draft context is not an eagle3 head context");
        let n_embd_dec = eagle.1.n_embd as usize;
        let n_embd_enc = eagle.1.n_embd_inp_enc as usize;
        let n_layer_tgt = ctx_tgt.n_layer();

        // EAGLE3: target layer id %d exceeds target n_layer %d (:519-521)
        for &lid in &target_layer_ids {
            if lid < 0 || lid as usize > n_layer_tgt {
                return Err(format!(
                    "EAGLE3: target layer id {lid} exceeds target n_layer {n_layer_tgt}"
                ));
            }
        }

        spec_dbg!(
            "spec draft-eagle3: adding speculative implementation 'draft-eagle3' (n_max={}, \
             n_min={}, p_min={:.6}, n_embd_dec={n_embd_dec}, n_embd_enc={n_embd_enc}, \
             backend_sampling={})",
            params.draft.n_max,
            params.draft.n_min,
            params.draft.p_min,
            params.draft.backend_sampling,
        );

        // `llama_batch_init(n_b, embd = n_embd_dec, 1)` + the token malloc
        // (:482-486) — the port's batch grows on demand
        let batch = LlamaBatch::default();

        // one draft sampler per sequence (:488-495)
        let mut smpls = Vec::with_capacity(n_seq as usize);
        for _ in 0..n_seq {
            smpls.push(DraftSampler::new(vocab_dft));
        }

        // backend_chains (:497-511) — not ported, see the struct docs

        Ok(CommonSpeculativeImplDraftEagle3 {
            params: params.draft.clone(),
            ctx_dft,
            smpls,
            n_embd_dec,
            n_embd_enc,
            n_embd_tgt,
            n_layer_tgt,
            target_layer_ids,
            // pending_g_last.assign(n_seq, vector<float>(n_embd_dec, 0)),
            // pending_pos_last.assign(n_seq, -1) (:528-529)
            pending_g_last: vec![vec![0.0; n_embd_dec]; n_seq as usize],
            pending_pos_last: vec![-1; n_seq as usize],
            verify_g: (0..n_seq as usize).map(|_| Vec::new()).collect(),
            verify_pos_first: vec![-1; n_seq as usize],
            verify_g_rows: vec![0; n_seq as usize],
            features_buf: Vec::new(),
            g_embd_buf: Vec::new(),
            batch,
            stats: CommonSpeculativeImplStats::default(),
        })
    }

    /// the C's `params.ctx_dft` pointer, for the driver's KV bookkeeping
    pub fn ctx_dft(&mut self) -> &mut DecodeContext {
        &mut self.ctx_dft
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplDraftEagle3 {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::DraftEagle3
    }

    fn n_max(&self) -> i32 {
        self.params.n_max
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(...)` (speculative.cpp:556-570): the pos_max sanity warning
    /// — after prefill the draft context holds pos 0..N-2 (the last position
    /// is deferred), so warn only when more than one position is missing.
    fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        let n = prompt.len() as i32;
        if n <= 0 {
            return;
        }
        let pos_max = self.ctx_dft.seq_pos_max(seq_id);
        if pos_max < n - 2 {
            eprintln!(
                "spec draft-eagle3: ctx_dft pos_max={pos_max} < N-2={} - process() did not run on \
                 every prefill ubatch. Drafts may degrade.",
                n - 2
            );
        }
    }

    /// `bool process(const llama_batch & batch_in)` (speculative.cpp:572-719):
    /// run the encoder over the target's extract-layer inputs of this batch,
    /// then train the decoder KV on (token[k+1], g_embd[k]) at pos[k] per
    /// sequence — everything except each sequence's last position, which
    /// becomes the new deferred boundary.
    fn process(&mut self, batch_in: &LlamaBatch, ctx_tgt: &DecodeContext) -> bool {
        let n_tokens = batch_in.token.len();
        if n_tokens == 0 {
            return true;
        }

        // `batch_in.token == nullptr || batch_in.embd != nullptr` — vision
        // batches are skipped (:577-579)
        if batch_in.embd.is_some() {
            return true;
        }

        // i_batch_beg / i_batch_end per seq (:583-598)
        let n_seq = self.pending_pos_last.len();
        let mut i_batch_beg = vec![-1i32; n_seq];
        let mut i_batch_end = vec![-1i32; n_seq];
        let seq_ids = batch_in
            .seq_id
            .as_ref()
            .expect("eagle3 process: batch seq_id");
        let pos = batch_in.pos.as_ref().expect("eagle3 process: batch pos");
        for (k, ids) in seq_ids.iter().enumerate() {
            assert_eq!(
                ids.len(),
                1,
                "batch_in.n_seq_id[k] == 1 (speculative.cpp:589)"
            );
            let seq_id = ids[0];
            if seq_id < 0 || seq_id as usize >= n_seq as i32 as usize {
                continue;
            }
            i_batch_end[seq_id as usize] = k as i32;
            if i_batch_beg[seq_id as usize] < 0 {
                i_batch_beg[seq_id as usize] = k as i32;
            }
        }

        // Interleave each extract_layer's hidden state into features_buf
        // [n_tokens, 3 * n_embd_tgt] (:603-620)
        self.features_buf.clear();
        self.features_buf.resize(n_tokens * self.n_embd_enc, 0.0);
        let n_e = self.n_embd_tgt;
        for (k, &lid) in self.target_layer_ids.iter().enumerate() {
            // layer < n_layer → the layer-input tap; == n_layer → the
            // unmasked nextn tap (:608-614)
            let layer: &[f32] = if (lid as usize) < self.n_layer_tgt {
                ctx_tgt.get_embeddings_layer_inp(lid as u32)
            } else {
                ctx_tgt.get_embeddings_nextn()
            };
            assert!(
                layer.len() >= n_tokens * n_e,
                "EAGLE3: target layer {lid} input not extracted ({} rows < {n_tokens})",
                layer.len() / n_e.max(1)
            );
            for i in 0..n_tokens {
                let dst = i * self.n_embd_enc + k * n_e;
                let src = i * n_e;
                self.features_buf[dst..dst + n_e].copy_from_slice(&layer[src..src + n_e]);
            }
        }

        // the chunked encoder (:622-652): llama_encode requires the whole
        // chunk to fit one ubatch; the per-token encoder is chunkable
        self.g_embd_buf.clear();
        self.g_embd_buf.resize(n_tokens * self.n_embd_dec, 0.0);
        let n_ubatch_dft = self.ctx_dft.n_batch;
        let mut i = 0usize;
        while i < n_tokens {
            let n_chunk = n_ubatch_dft.min(n_tokens - i);
            if let Err(e) = self.ctx_dft.encode_eagle3(
                &self.features_buf[i * self.n_embd_enc..(i + n_chunk) * self.n_embd_enc],
            ) {
                eprintln!(
                    "spec draft-eagle3: llama_encode(ctx_dft) failed ({e}) (n_tokens={}, \
                     offset={i})",
                    n_chunk
                );
                return false;
            }
            // port-only accounting (see CommonSpeculativeImplStats::n_forward)
            self.stats.n_forward += 1;

            // g_embd chunk from the pre-norm embeddings buffer (:646-651)
            let g = self.ctx_dft.get_embeddings_nextn();
            self.g_embd_buf[i * self.n_embd_dec..(i + n_chunk) * self.n_embd_dec]
                .copy_from_slice(&g[..n_chunk * self.n_embd_dec]);
            i += n_chunk;
        }

        let row_len = self.n_embd_dec;

        // the decoder training batch (:658-716): per seq (a) the cross-ubatch
        // bridge, (b) the main write loop (k in [beg, end-1]), (c) refresh the
        // deferred state (verify snapshot + pending boundary)
        self.batch.clear();

        for seq_id in 0..n_seq {
            let beg = i_batch_beg[seq_id];
            let end = i_batch_end[seq_id];
            if beg < 0 || end < 0 {
                continue;
            }

            // (a) cross-ubatch bridge (:678-691): complete the prior ubatch's
            // deferred boundary with this ubatch's first token. Fires iff
            // pending_pos >= 0, pending_pos + 1 == pos[beg] and
            // pending_pos > dft_pos_max (the C's TODO'd third check)
            let pending_pos = self.pending_pos_last[seq_id];
            if pending_pos >= 0 && pending_pos + 1 == pos[beg as usize] {
                let dft_pos_max = self.ctx_dft.seq_pos_max(seq_id as i32);
                if pending_pos > dft_pos_max {
                    self.batch.add(
                        batch_in.token[beg as usize],
                        pending_pos,
                        &[seq_id as i32],
                        false,
                    );
                    let at = self.batch.token.len() - 1;
                    let embd = self.batch.embd.get_or_insert_with(Vec::new);
                    embd.resize((at + 1) * row_len, 0.0);
                    embd[at * row_len..(at + 1) * row_len]
                        .copy_from_slice(&self.pending_g_last[seq_id]);
                }
            }

            // (b) main write loop — (token[k+1], g_embd[k]) at pos[k]
            // (:693-697)
            for k in beg..end {
                self.batch.add(
                    batch_in.token[(k + 1) as usize],
                    pos[k as usize],
                    &[seq_id as i32],
                    false,
                );
                let at = self.batch.token.len() - 1;
                let embd = self.batch.embd.get_or_insert_with(Vec::new);
                embd.resize((at + 1) * row_len, 0.0);
                embd[at * row_len..(at + 1) * row_len].copy_from_slice(
                    &self.g_embd_buf[k as usize * row_len..(k as usize + 1) * row_len],
                );
            }

            // (c) refresh deferred state (:699-706)
            let n_rows = (end - beg + 1) as usize;
            self.verify_pos_first[seq_id] = pos[beg as usize];
            self.pending_pos_last[seq_id] = pos[end as usize];
            self.verify_g_rows[seq_id] = n_rows as i32;
            self.verify_g[seq_id].resize(n_rows * row_len, 0.0);
            self.verify_g[seq_id].copy_from_slice(
                &self.g_embd_buf[beg as usize * row_len..(end as usize + 1) * row_len],
            );
            self.pending_g_last[seq_id].copy_from_slice(
                &self.g_embd_buf[end as usize * row_len..(end as usize + 1) * row_len],
            );
        }

        if !self.batch.token.is_empty() {
            if let Err(e) = self.ctx_dft.decode_batch(&self.batch) {
                eprintln!(
                    "spec draft-eagle3: llama_decode(ctx_dft) failed ({e}) (n_tokens={}, \
                     ubatch_pos[0]={})",
                    self.batch.token.len(),
                    pos[0]
                );
                return false;
            }
            // port-only accounting (see CommonSpeculativeImplStats::n_forward)
            self.stats.n_forward += 1;
        }

        true
    }

    fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        Some(&mut self.ctx_dft)
    }

    /// `void draft(...)` (speculative.cpp:721-846): complete the deferred
    /// boundary with dp.id_last at pending_pos_last (after dropping the draft
    /// tail), decode, and walk the argmax chain while p >= p_min — each step
    /// feeding (id, prenorm row) at pending_pos_last + i + 1.
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        let n_seq = self.pending_pos_last.len();

        self.batch.clear();

        // keep track of which sequences are still drafting (:726-728)
        let mut n_drafting = 0usize;
        let mut drafting = vec![false; n_seq];

        let row_len = self.n_embd_dec;

        // the deferred boundary pair (:732-756)
        for seq_id in 0..n_seq {
            if !dparams[seq_id].drafting {
                continue;
            }
            if self.pending_pos_last[seq_id] < 0 {
                continue;
            }

            n_drafting += 1;
            drafting[seq_id] = true;
            self.smpls[seq_id].reset();

            // `llama_memory_seq_rm(ctx_dft, seq_id, pending_pos_last, -1)`
            // (:750)
            self.ctx_dft
                .seq_rm(seq_id as i32, self.pending_pos_last[seq_id], -1);

            self.batch.add(
                dparams[seq_id].id_last,
                self.pending_pos_last[seq_id],
                &[seq_id as i32],
                true,
            );
            let at = self.batch.token.len() - 1;
            let embd = self.batch.embd.get_or_insert_with(Vec::new);
            embd.resize((at + 1) * row_len, 0.0);
            embd[at * row_len..(at + 1) * row_len].copy_from_slice(&self.pending_g_last[seq_id]);
        }

        if self.batch.token.is_empty() {
            return;
        }

        let mut ret = self.ctx_dft.decode_batch(&self.batch);
        if let Err(e) = ret {
            eprintln!("spec draft-eagle3: llama_decode returned {e}");
            return;
        }
        // port-only accounting (see CommonSpeculativeImplStats::n_forward)
        self.stats.n_forward += 1;

        // the batch row of each drafting sequence — the seed batch put one
        // row per drafting seq, in order (:771)
        let mut i_batch_of = vec![-1i32; n_seq];
        {
            let mut ib = 0i32;
            for seq_id in 0..n_seq {
                if drafting[seq_id] {
                    i_batch_of[seq_id] = ib;
                    ib += 1;
                }
            }
        }

        let mut i = 0i32;

        while n_drafting > 0 {
            self.batch.clear();

            for seq_id in 0..n_seq {
                if !drafting[seq_id] {
                    continue;
                }

                let _i_batch = i_batch_of[seq_id];

                // `common_sampler_sample(smpl, ctx_dft, i_batch, true)`
                // (:782) — the sampled token is discarded (the drafted one is
                // the argmax, :795); i_batch is the running row counter the
                // C bumps once per drafting sequence (:785), the port's
                // i_batch_of snapshot below
                let logits = ret
                    .as_ref()
                    .expect("draft loop decode result")
                    .logits_ith(i_batch_of[seq_id])
                    .expect("draft-eagle3: draft logits row")
                    .to_vec();
                self.smpls[seq_id].sample(&logits);
                // the pre-norm hidden state of this position becomes the next
                // step's g_embd (:783-784)
                let prenorm = self
                    .ctx_dft
                    .get_embeddings_nextn_ith(i_batch_of[seq_id])
                    .to_vec();

                // `cur_p = common_sampler_get_candidates(smpl, true)` (:787)
                let (id, p_top) = {
                    let cur_p = self.smpls[seq_id].get_candidates(true);
                    for (k, cand) in cur_p.iter().take(3).enumerate() {
                        spec_dbg!(
                            "spec draft-eagle3: - seq_id {seq_id}, draft candidate {k:3}, pos \
                             {i:3}: {:6} ({:8.3})",
                            cand.id,
                            cand.p
                        );
                    }
                    (cur_p[0].id, cur_p[0].p)
                };

                // only collect very high-confidence draft tokens (:797-804)
                if p_top < self.params.p_min {
                    drafting[seq_id] = false;
                    n_drafting -= 1;
                    continue;
                }

                self.smpls[seq_id].accept(id);

                let pos_base = self.pending_pos_last[seq_id];
                dparams[seq_id].result.push(id);

                if self.params.n_max <= dparams[seq_id].result.len() as i32 {
                    drafting[seq_id] = false;
                    n_drafting -= 1;
                    continue;
                }

                // the next step's pair at pos pending_pos_last + i + 1
                // (:819-820)
                self.batch.add(id, pos_base + i + 1, &[seq_id as i32], true);
                let at = self.batch.token.len() - 1;
                let embd = self.batch.embd.get_or_insert_with(Vec::new);
                embd.resize((at + 1) * row_len, 0.0);
                embd[at * row_len..(at + 1) * row_len].copy_from_slice(&prenorm);
            }

            if self.batch.token.is_empty() {
                break;
            }

            ret = self.ctx_dft.decode_batch(&self.batch);
            if let Err(e) = ret {
                eprintln!("spec draft-eagle3: llama_decode[{i}] returned {e}");
                break;
            }
            // port-only accounting
            self.stats.n_forward += 1;

            // re-derive the batch rows: the new batch holds one row per still
            // drafting seq, in seq order (:775-786's i_batch counter)
            let mut ib = 0i32;
            for s in 0..n_seq {
                if drafting[s] {
                    i_batch_of[s] = ib;
                    ib += 1;
                }
            }

            i += 1;
        }

        // drop drafts shorter than n_min (:836-845)
        for dp in dparams.iter_mut() {
            if !dp.drafting {
                continue;
            }
            if dp.result.len() < self.params.n_min as usize {
                dp.result.clear();
            }
        }
    }

    /// `void accept(...)` (speculative.cpp:848-863): rebase the deferred
    /// boundary to the verify snapshot's row of the last accepted position.
    fn accept(&mut self, seq_id: i32, n_accepted: u16, _is_other: bool) {
        if seq_id < 0 || seq_id as usize >= self.verify_g_rows.len() {
            return;
        }

        let n_rows = self.verify_g_rows[seq_id as usize];
        if n_rows <= 0 {
            return;
        }

        // `const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1)`
        // (:858)
        let i_g = (n_accepted as i32).min(n_rows - 1) as usize;
        self.pending_pos_last[seq_id as usize] =
            self.verify_pos_first[seq_id as usize] + i_g as i32;
        let n_e = self.n_embd_dec;
        self.pending_g_last[seq_id as usize]
            .copy_from_slice(&self.verify_g[seq_id as usize][i_g * n_e..(i_g + 1) * n_e]);
    }
}

// ---------------------------------------------------------------------------
// common_speculative_impl_draft_dflash (speculative.cpp:909-1328)
// ---------------------------------------------------------------------------

/// `struct common_speculative_impl_draft_dflash` (speculative.cpp:910-1328):
/// the block-diffusion drafter — the DFlash/DSpark driver over the dflash
/// draft context ([`crate::dflash`]'s dual-mode decoder). `process()` replays
/// every target prefill/verify batch as a KV *injection* (the target's
/// extract-layer inputs fused into the draft cache), `draft()` decodes one
/// noise block `[id_last, <mask> * (block_size-1)]` per sequence in a single
/// batch and reads the block's predictions off the logits (DSpark: the
/// markov-biased logits + the confidence rows of the nextn tap).
///
/// `draft_dspark` is the same impl constructed with
/// `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` (speculative.cpp:2671-2674): the
/// draft carries the Markov head (`is_dspark` below) and, when
/// `sample_from_anchor`, drafts the full block incl. the anchor slot.
///
/// Not ported (the dflash loader refuses those drafts, see dflash.rs):
/// `backend_sampling` (:1031-1043 — the port is CPU-only, the CPU smpls of
/// :1020-1027 are what runs), the M-RoPE injection rows (:1013-1018/:1161-
/// 1165) and the DFlash2/DSV4 backbones.
pub struct CommonSpeculativeImplDraftDflash {
    /// `common_params_speculative_draft params` — a copy of `params.draft`
    /// with `n_max`/`n_min` clamped to the trained block size (:1002-1008)
    params: CommonParamsSpeculativeDraft,

    /// `params.ctx_dft` — a pointer in the C, owned here
    ctx_dft: DecodeContext,

    /// `llama_batch batch` — the noise-token batch (:913), reused by draft()
    batch: LlamaBatch,

    /// `llama_batch batch_inject` — the target-feature rows for KV injection
    /// (:914). The C's batch is embd-only (`llama_batch_init(..., n_embd_enc,
    /// n_seq)`); the port's LlamaBatch always carries token ids, so the
    /// injection batches hold **dummy ids** (never read by the injection
    /// graph) plus the F32 feature rows.
    batch_inject: LlamaBatch,

    /// `std::vector<common_sampler_ptr> smpls` (:916)
    smpls: Vec<DraftSampler>,

    n_embd_dec: usize,
    n_embd_enc: usize,
    n_embd_tgt: usize,

    block_size: i32,
    mask_token_id: i32,

    is_dflash2: bool,
    is_dspark: bool,
    sample_from_anchor: bool,
    causal_attn: bool,
    selector_top_k: i32,

    /// `target_layer_ids` (:941) — the draft model's extract layer indices
    target_layer_ids: Vec<i32>,

    stats: CommonSpeculativeImplStats,
}

impl CommonSpeculativeImplDraftDflash {
    /// (speculative.cpp:944-1053). The hparams the C reads off the draft
    /// model come from [`crate::dflash::DflashParams`] (carried by the draft
    /// context); `vocab_dft` supplies `llama_vocab_mask` (:982).
    pub fn new(
        params: &CommonParamsSpeculative,
        n_seq: u32,
        mut ctx_dft: DecodeContext,
        ctx_tgt: &DecodeContext,
        vocab_dft: &Vocab,
    ) -> Result<Self, String> {
        // GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and
        // ctx_dft to be set") (:952) — both are owned/reachable here
        let dflash = ctx_dft
            .dflash
            .as_ref()
            .expect("draft-dflash: the draft context is not a dflash draft context");

        let target_layer_ids = dflash.1.target_layer_ids.clone();
        assert!(
            !target_layer_ids.is_empty(),
            "DFlash model has no target_layer_ids" // :959
        );

        let n_embd_tgt = dflash.1.n_embd_tgt as usize; // llama_model_n_embd(model_tgt), :961
        assert!(
            n_embd_tgt > 0,
            "DFlash: the target model's hidden size is required (the loader reads it off the \
             target file)"
        );
        let n_embd_dec = dflash.1.n_embd as usize;
        let n_embd_enc = target_layer_ids.len() * n_embd_tgt; // :963

        // the trained block size + the layout flags read back from the GGUF
        // meta strings (:965-978) — parsed by the dflash loader
        let block_size = dflash.1.block_size;
        let mut sample_from_anchor = dflash.1.sample_from_anchor;
        let causal_attn = dflash.1.attention_causal;
        let has_conf = dflash.1.has_confidence_head;

        let is_dspark = params.types.contains(&CommonSpeculativeType::DraftDspark);
        let selector_top_k = dflash.1.selector_top_k as i32;
        let is_dflash2 = selector_top_k > 0; // :980-981
        let mask_token_id = vocab_dft.token_mask();

        // DSpark's confidence gate needs the confidence head (:984-992)
        if is_dspark && params.draft.p_min > 0.0 && !has_conf {
            return Err(
                "DSpark draft has no confidence head: please set --spec-draft-p-min 0".into(),
            );
        }

        spec_dbg!(
            "spec draft-dflash: adding speculative implementation '{}' (n_max={}, n_min={}, \
             p_min={:.2}, block_size={block_size}, mask_token_id={mask_token_id}, n_extract={}, \
             sample_from_anchor={sample_from_anchor})",
            if is_dspark {
                "draft-dspark"
            } else {
                "draft-dflash"
            },
            params.draft.n_max,
            params.draft.n_min,
            params.draft.p_min,
            target_layer_ids.len(),
        );

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place
        // denoising yields at most block_size-1 draft tokens, anchor-first
        // DSpark a full block_size (:999-1001)
        let n_draft_max = if is_dspark && sample_from_anchor {
            block_size
        } else {
            block_size - 1
        };
        let mut params_d = params.draft.clone();
        if params_d.n_max > n_draft_max || params_d.n_min > n_draft_max {
            eprintln!(
                "spec draft-dflash: requested draft size (n_max={}, n_min={}) exceeds the \
                 trained block size {block_size} -- clamping to {n_draft_max}",
                params_d.n_max, params_d.n_min
            );
            params_d.n_max = params_d.n_max.min(n_draft_max); // :1005
            params_d.n_min = params_d.n_min.min(n_draft_max); // :1006
        }

        // `batch = llama_batch_init(llama_n_batch(ctx_dft), 0, n_seq)` +
        // `batch_inject = llama_batch_init(llama_n_ubatch(ctx_dft),
        // n_embd_enc, n_seq)` (:1010-1011) — the port's batches grow on demand
        let batch = LlamaBatch::default();
        let batch_inject = LlamaBatch::default();

        // one draft sampler per sequence (:1020-1027)
        let mut smpls = Vec::with_capacity(n_seq as usize);
        for _ in 0..n_seq {
            smpls.push(DraftSampler::new(vocab_dft));
        }

        // backend sampling offload (:1031-1043) — not ported, see the struct
        // docs
        let _ = sample_from_anchor;

        // the draft context's nextn tap + non-causal attention (:1051-1052) —
        // DFlash2 reads its selector lattice from h_nextn and never consumes
        // raw logits, hence the unmasked tap
        ctx_dft.set_embeddings_nextn(true, !is_dflash2);
        ctx_dft.set_causal_attn(causal_attn);

        Ok(CommonSpeculativeImplDraftDflash {
            params: params_d,
            ctx_dft,
            batch,
            batch_inject,
            smpls,
            n_embd_dec,
            n_embd_enc,
            n_embd_tgt,
            block_size,
            mask_token_id,
            is_dflash2,
            is_dspark,
            sample_from_anchor,
            causal_attn,
            selector_top_k,
            target_layer_ids,
            stats: CommonSpeculativeImplStats::default(),
        })
    }

    /// the C's `params.ctx_dft` pointer, for the driver's KV bookkeeping
    pub fn ctx_dft(&mut self) -> &mut DecodeContext {
        &mut self.ctx_dft
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplDraftDflash {
    fn ty(&self) -> CommonSpeculativeType {
        if self.is_dspark {
            CommonSpeculativeType::DraftDspark
        } else {
            CommonSpeculativeType::DraftDflash
        }
    }

    fn n_max(&self) -> i32 {
        self.params.n_max // this->n_max = params.n_max (:1008)
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(...)` (speculative.cpp:1072-1088): the pos_max sanity
    /// warning — the draft cache is seeded by the injection passes, so a
    /// pos_max short of N-1 means process() missed prefill ubatches.
    fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        let n = prompt.len() as i32;
        if seq_id < 0 || seq_id as usize >= self.smpls.len() || n <= 0 {
            return;
        }

        let pos_max = self.ctx_dft.seq_pos_max(seq_id);
        if pos_max < n - 1 {
            eprintln!(
                "spec draft-dflash: ctx_dft pos_max={pos_max} < N-1={} - process() did not run \
                 on every prefill ubatch. Drafts may degrade.",
                n - 1
            );
        }
    }

    /// `bool process(const llama_batch & batch_in)` (speculative.cpp:1090-
    /// 1180): gather the target's extract-layer inputs of this batch and
    /// inject their fused K/V into the draft cache, chunked by the draft
    /// context's n_ubatch.
    fn process(&mut self, batch_in: &LlamaBatch, ctx_tgt: &DecodeContext) -> bool {
        if batch_in.token.is_empty() {
            return true;
        }

        // `const bool has_tokens = batch_in.token != nullptr; const bool
        // has_embeddings = batch_in.embd != nullptr; if (has_tokens ==
        // has_embeddings) return true;` (:1099-1103) — the port's batch always
        // carries tokens, so only a token-only target batch proceeds (the
        // embd batches are the vision inputs the C also skips here)
        if batch_in.embd.is_some() {
            return true;
        }

        let n_tokens = batch_in.token.len();
        let n_seq = self.smpls.len();

        // per-seq inclusive batch range (assumes each seq's tokens are
        // contiguous in the batch, :1107-1120)
        let mut i_batch_beg = vec![-1i32; n_seq];
        let mut i_batch_end = vec![-1i32; n_seq];
        for (k, ids) in batch_in
            .seq_id
            .as_ref()
            .expect("dflash process: seq_id")
            .iter()
            .enumerate()
        {
            assert_eq!(
                ids.len(),
                1,
                "batch_in.n_seq_id[k] == 1 (speculative.cpp:1111)"
            );
            let seq_id = ids[0];
            if seq_id < 0 || seq_id as usize >= n_seq as i32 as usize {
                continue;
            }
            i_batch_end[seq_id as usize] = k as i32;
            if i_batch_beg[seq_id as usize] < 0 {
                i_batch_beg[seq_id as usize] = k as i32;
            }
        }

        let n_embd_enc = self.n_embd_enc;
        let n_embd_tgt = self.n_embd_tgt;
        let n_ubatch = self.ctx_dft.n_batch; // llama_n_ubatch(ctx_dft) (:1125)

        for seq_id in 0..n_seq as i32 {
            if i_batch_beg[seq_id as usize] < 0 {
                continue;
            }
            let beg = i_batch_beg[seq_id as usize];
            let n_rows = (i_batch_end[seq_id as usize] - beg + 1) as usize;

            // an M-RoPE image pins all its rows to one position (:1133-1138) —
            // not ported (the loader refuses M-RoPE drafts)

            let mut offset = 0usize;
            while offset < n_rows {
                let n_chunk = n_ubatch.min(n_rows - offset);

                // gather target features per extract layer; the fused decode
                // encodes and injects them into the K/V cache at the target
                // positions (:1145-1156)
                let mut embd = vec![0f32; n_chunk * n_embd_enc];
                for (k, &lid) in self.target_layer_ids.iter().enumerate() {
                    let layer = ctx_tgt.get_embeddings_layer_inp(lid as u32);
                    assert!(
                        layer.len() >= (beg as usize + offset + n_chunk) * n_embd_tgt,
                        "DFlash: target layer {lid} input not extracted."
                    );
                    for i in 0..n_chunk {
                        let dst = i * n_embd_enc + k * n_embd_tgt;
                        let src = (beg as usize + offset + i) * n_embd_tgt;
                        embd[dst..dst + n_embd_tgt].copy_from_slice(&layer[src..src + n_embd_tgt]);
                    }
                }

                // batch_inject rows (:1158-1169) — token ids are dummy (never
                // read by the injection graph), logits are all false
                let pos = batch_in.pos.as_ref().expect("dflash process: pos");
                self.batch_inject.clear();
                for i in 0..n_chunk {
                    let p = pos[beg as usize + offset + i];
                    self.batch_inject.add(0, p, &[seq_id], false);
                }
                self.batch_inject.embd = Some(embd);

                if let Err(e) = self.ctx_dft.decode_batch(&self.batch_inject) {
                    eprintln!(
                        "spec draft-dflash: llama_decode(ctx_dft) failed {e} (n_tokens={}, \
                         offset={offset})",
                        n_chunk
                    );
                    return false;
                }
                // port-only accounting (see CommonSpeculativeImplStats::n_forward)
                self.stats.n_forward += 1;

                offset += n_chunk;
            }
        }

        true
    }

    fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        Some(&mut self.ctx_dft)
    }

    /// `void draft(common_speculative_draft_params_vec &)` (speculative.cpp:
    /// 1182-1323): decode every drafting sequence's noise block in one batch,
    /// then read the block's predictions — the argmax chain off the (for
    /// DSpark markov-biased) logits, gated by p_min / the confidence rows.
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        let n_seq = dparams.len();

        self.batch.clear();

        // build one batch holding every drafting sequence's noise block into a
        // single decode; record where each block starts and its size
        // (:1187-1190)
        let mut i_block_beg = vec![-1i32; n_seq];
        let mut n_block = vec![0i32; n_seq];

        for seq_id in 0..n_seq {
            let dp = &dparams[seq_id];
            if !dp.drafting {
                continue;
            }

            self.smpls[seq_id].reset();

            let n = dp.pos0;
            let n_draft = self.params.n_max;

            let n_block_tokens = n_draft
                + if self.is_dspark && self.sample_from_anchor {
                    0
                } else {
                    1
                };
            i_block_beg[seq_id] = self.batch.token.len() as i32;
            n_block[seq_id] = n_block_tokens;
            for i in 0..n_block_tokens {
                let tok = if i == 0 {
                    dp.id_last
                } else {
                    self.mask_token_id
                };
                self.batch
                    .add(tok, n + i, &[seq_id as i32], !self.is_dflash2);
            }
        }

        if self.batch.token.is_empty() {
            return;
        }

        // decode all sequences' noise block in a single batch (:1216-1221)
        let out = match self.ctx_dft.decode_batch(&self.batch) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("spec draft-dflash: llama_decode returned {e}");
                return;
            }
        };
        // port-only accounting (see CommonSpeculativeImplStats::n_forward)
        self.stats.n_forward += 1;

        let n_embd_dec = self.n_embd_dec;

        for seq_id in 0..n_seq {
            let beg = i_block_beg[seq_id];
            if beg < 0 {
                continue;
            }
            let n_block_tokens = n_block[seq_id];

            // `common_sampler_sample(smpl, ctx_dft, idx, true)` +
            // `common_sampler_get_candidates(smpl, true)` per block position —
            // the drafted token is the argmax, never the sampled one. `pos` is
            // the block position the C's SPC_DBG prints (i-1 for DFlash,
            // speculative.cpp:1304; i for DSpark, :1284)
            let sample_argmax = |idx: i32, pos: i32, smpl: &mut DraftSampler| -> (i32, f32) {
                let logits = out
                    .logits_ith(idx)
                    .expect("draft-dflash: draft logits row")
                    .to_vec();
                smpl.sample(&logits);
                let cur_p = smpl.get_candidates(true);
                for (k, cand) in cur_p.iter().take(3).enumerate() {
                    spec_dbg!(
                        "spec draft-dflash: - seq_id {seq_id}, draft candidate {k:3}, pos {pos:3}: \
                         {:6} ({:8.3})",
                        cand.id,
                        cand.p
                    );
                }
                (cur_p[0].id, cur_p[0].p)
            };

            if self.is_dflash2 {
                // the DFlash2 lattice walk (:1236-1263) — reads the selector
                // lattice from the (unmasked) nextn tap. The C prints no
                // candidate trace on this path; the port mirrors the chain
                // probe's per-round draft print (parity/ref_dflash_chain.cpp)
                // under LLAMA_SPEC_VERBOSE for the parity cell.
                let lattice = self.ctx_dft.get_embeddings_nextn().to_vec();
                assert!(!lattice.is_empty(), "DFlash2 selector produced no lattice");

                let mut predecessor = 0i32;
                let top_k = self.selector_top_k;
                for i in 1..n_block_tokens {
                    let row = beg as usize * n_embd_dec + i as usize * n_embd_dec;
                    let scores = top_k as usize + predecessor as usize * top_k as usize;
                    let scores = &lattice[row + scores..row + scores + top_k as usize];

                    predecessor = scores
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0);
                    if self.params.p_min > 0.0 {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k -
                        // s_max)) (:1247-1256)
                        let mut sum = 0.0f32;
                        for &s in scores.iter() {
                            sum += (s - scores[predecessor as usize]).exp();
                        }
                        if 1.0 / sum < self.params.p_min {
                            break;
                        }
                    }
                    // `result.push_back((llama_token) row[predecessor])` — the
                    // candidate ids live in the lattice row's head
                    let cand = lattice[row + predecessor as usize] as i32;
                    dparams[seq_id].result.push(cand);
                }

                if dparams[seq_id].result.len() < self.params.n_min as usize {
                    dparams[seq_id].result.clear();
                }
                continue;
            }

            if self.is_dspark {
                // DSpark: read from the first draft slot, truncate below the
                // confidence threshold (:1266-1293)
                let conf: Option<Vec<f32>> = if self.params.p_min > 0.0 {
                    Some(self.ctx_dft.get_embeddings_nextn().to_vec())
                } else {
                    None
                };
                // bonus-anchor drafts read the mask positions only, like
                // DFlash (:1269-1270)
                let i_draft_beg = if self.sample_from_anchor { 0 } else { 1 };
                for i in i_draft_beg..n_block_tokens {
                    let idx = beg + i;

                    if let Some(conf) = conf.as_ref() {
                        if conf[idx as usize * n_embd_dec] < self.params.p_min {
                            break;
                        }
                    }

                    let (id, _p) = sample_argmax(idx, i, &mut self.smpls[seq_id]);

                    self.smpls[seq_id].accept(id);

                    dparams[seq_id].result.push(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise
                // positions 1..n_block_tokens-1 (:1294-1316)
                for i in 1..n_block_tokens {
                    let idx = beg + i;

                    let (id, p_top) = sample_argmax(idx, i - 1, &mut self.smpls[seq_id]);

                    // only collect very high-confidence draft tokens
                    // (:1309-1311)
                    if p_top < self.params.p_min {
                        break;
                    }

                    self.smpls[seq_id].accept(id);

                    dparams[seq_id].result.push(id);
                }
            }

            // drop drafts shorter than n_min (:1319-1321)
            if dparams[seq_id].result.len() < self.params.n_min as usize {
                dparams[seq_id].result.clear();
            }
        }
    }

    /// `void accept(...) override { // noop }` (speculative.cpp:1325-1327)
    fn accept(&mut self, _seq_id: i32, _n_accepted: u16, _is_other: bool) {}
}

// ---------------------------------------------------------------------------
// the ngram family (speculative.cpp:1769-2181 over common/ngram-*.cpp)
// ---------------------------------------------------------------------------

/// `struct common_speculative_impl_ngram_simple` (speculative.cpp:1769-1813):
/// the backward pattern search of
/// [`crate::ngram_map::common_ngram_simple_draft`], shared across sequences.
pub struct CommonSpeculativeImplNgramSimple {
    /// `common_params_speculative_ngram_map params` (the `params.ngram_simple`
    /// arm — the min_hits field is unused by the simple algorithm)
    size_n: u16,
    size_m: u16,
    _min_hits: u16,

    stats: CommonSpeculativeImplStats,
}

impl CommonSpeculativeImpl for CommonSpeculativeImplNgramSimple {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::NgramSimple
    }

    fn n_max(&self) -> i32 {
        self.size_m as i32
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(...) override { // noop }` (speculative.cpp:1788-1790)
    fn begin(&mut self, _seq_id: i32, _prompt: &[i32]) {}

    /// `bool process(...) override { return true; }` (:1792-1795, TODO in
    /// the reference — no draft model to feed)
    fn process(&mut self, _batch: &LlamaBatch, _ctx_tgt: &DecodeContext) -> bool {
        true
    }

    /// `void draft(...) override` (:1797-1808)
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        let config = crate::ngram_map::CommonNgramSimpleConfig {
            size_ngram: self.size_n,
            size_mgram: self.size_m,
        };

        for dp in dparams.iter_mut() {
            if !dp.drafting {
                continue;
            }

            dp.result =
                crate::ngram_map::common_ngram_simple_draft(&config, &dp.prompt, dp.id_last);
        }
    }

    /// `void accept(...) override { // noop }` (:1810-1812)
    fn accept(&mut self, _seq_id: i32, _n_accepted: u16, _is_other: bool) {}
}

/// `struct common_speculative_impl_ngram_map_k` (speculative.cpp:1815-1867):
/// the n-gram→m-gram map of [`crate::ngram_map::CommonNgramMap`], one map per
/// sequence — the `ngram-map-k` (`key_only`) and `ngram-map-k4v` modes of the
/// same impl (the type flag only changes the map's `key_only`).
pub struct CommonSpeculativeImplNgramMapK {
    /// the impl's reported type — NGramMapK (key_only) or NGramMapK4V
    ty: CommonSpeculativeType,
    /// n_seq configs (:1817)
    config: Vec<crate::ngram_map::CommonNgramMap>,

    stats: CommonSpeculativeImplStats,
}

impl CommonSpeculativeImpl for CommonSpeculativeImplNgramMapK {
    fn ty(&self) -> CommonSpeculativeType {
        self.ty
    }

    fn n_max(&self) -> i32 {
        // `config.size_value` (:1823)
        self.config[0].size_value as i32
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(seq_id, prompt) override` (:1834-1838)
    fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        assert!((seq_id as usize) < self.config.len());
        self.config[seq_id as usize].begin(prompt);
    }

    /// `bool process(...) override { return true; }` (:1840-1843)
    fn process(&mut self, _batch: &LlamaBatch, _ctx_tgt: &DecodeContext) -> bool {
        true
    }

    /// `void draft(...) override` (:1845-1856)
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        for (seq_id, dp) in dparams.iter_mut().enumerate() {
            if !dp.drafting {
                continue;
            }

            self.config[seq_id].draft(&dp.prompt, dp.id_last, &mut dp.result);
        }
    }

    /// `void accept(seq_id, n_accepted, is_other) override` (:1858-1866)
    fn accept(&mut self, seq_id: i32, n_accepted: u16, is_other: bool) {
        assert!((seq_id as usize) < self.config.len());

        if is_other {
            return;
        }

        self.config[seq_id as usize].accept(n_accepted);
    }
}

/// `struct common_speculative_impl_ngram_mod` (speculative.cpp:1869-2042):
/// the modular-hash n-gram container of [`crate::ngram_mod::CommonNgramMod`]
/// (ref https://github.com/ggml-org/llama.cpp/pull/19164).
pub struct CommonSpeculativeImplNgramMod {
    /// `common_params_speculative_ngram_mod params` (the `params.ngram_mod`
    /// arm, :1895)
    n_match: i32,
    n_max: i32,
    n_min: i32,

    /// shared across all sequences (:1873) — 4M slots (:1896)
    mod_: crate::ngram_mod::CommonNgramMod,

    /// `struct seq_info` (:1878-1887), one per sequence
    sinfos: Vec<NgramModSeqInfo>,

    stats: CommonSpeculativeImplStats,
}

/// the per-sequence bookkeeping of `common_speculative_impl_ngram_mod`
/// (speculative.cpp:1878-1887)
struct NgramModSeqInfo {
    /// the last position in the prompt that was added to the ngram container
    i_last: usize,
    /// length of the last drafted n-gram (number of tokens returned by draft)
    n_draft_last: usize,
    /// consecutive accept rounds with low acceptance fraction (< 0.5)
    n_low: i32,
}

impl CommonSpeculativeImplNgramMod {
    /// the constructor (:1891-1912) minus the `LLAMA_TRACE` flag (the port
    /// folds the trace prints into `spec_dbg!`)
    pub fn new(params: &CommonParamsSpeculative, n_seq: u32) -> Self {
        let p = &params.ngram_mod;

        spec_dbg!(
            "spec ngram-mod: adding speculative implementation 'ngram-mod' \
             (n_match={}, n_max={}, n_min={})",
            p.n_match,
            p.n_max,
            p.n_min
        );

        if p.n_match < 16 {
            eprintln!(
                "spec ngram-mod: ngram_mod n_match={} is too small - poor quality is possible, \
                 see: https://github.com/ggml-org/llama.cpp/pull/19164",
                p.n_match
            );
        }

        CommonSpeculativeImplNgramMod {
            n_match: p.n_match,
            n_max: p.n_max,
            n_min: p.n_min,
            mod_: crate::ngram_mod::CommonNgramMod::new(p.n_match as u16, 4 * 1024 * 1024),
            sinfos: (0..n_seq)
                .map(|_| NgramModSeqInfo {
                    i_last: 0,
                    n_draft_last: 0,
                    n_low: 0,
                })
                .collect(),
            stats: CommonSpeculativeImplStats::default(),
        }
    }

    /// `draft_one` (speculative.cpp:1942-1996): extend the draft from the
    /// mod table — the key is the last n_match tokens (prompt tail +
    /// id_last), each next key chains the drafted tokens.
    fn draft_one(&mut self, seq_id: usize, dp: &mut CommonSpeculativeDraftParams) {
        let sinfo = &mut self.sinfos[seq_id];
        let prompt = &dp.prompt;
        let n = self.mod_.get_n();

        sinfo.n_draft_last = 0;

        let cur_len = prompt.len();
        if cur_len < n {
            return; // :1953-1955
        }

        // add new ngrams in chunks (:1959-1966)
        if sinfo.i_last + 32 < cur_len {
            for i in sinfo.i_last..cur_len - n {
                self.mod_.add(&prompt[i..]);
            }

            sinfo.i_last = cur_len - n;
        }

        // `result.resize(n + n_max)` (:1968) — the first n entries hold the
        // key, then n_max drafted followers
        let mut result: Vec<i32> = vec![0; n + self.n_max as usize];
        for i in 0..n - 1 {
            result[i] = prompt[cur_len - n + 1 + i];
        }
        result[n - 1] = dp.id_last;

        for i in 0..self.n_max as usize {
            let token = self.mod_.get(&result[i..]);
            if token == crate::ngram_mod::EMPTY {
                if i < self.n_min as usize {
                    result.clear(); // :1977-1980
                    dp.result = result;
                    return;
                }

                result.truncate(n + i); // :1982-1984
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted (:1988-1992)
        let m = result.len() - n;
        result.drain(..n);
        debug_assert_eq!(result.len(), m);

        // store length of drafted n-gram for later acceptance analysis (:1994)
        sinfo.n_draft_last = result.len();
        dp.result = result;
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplNgramMod {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::NgramMod
    }

    fn n_max(&self) -> i32 {
        self.n_max
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(seq_id, prompt) override` (:1914-1940)
    fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        let sinfo = &mut self.sinfos[seq_id as usize];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        let n = self.mod_.get_n();
        if prompt.len() < n {
            return;
        }

        for i in 0..prompt.len() - n {
            self.mod_.add(&prompt[i..]);
        }

        sinfo.i_last = prompt.len() - n;

        // the occupancy warning + reset (:1931-1939)
        let f = self.mod_.get_used() as f64 / self.mod_.size() as f64;
        spec_dbg!("spec ngram-mod: ngram_mod occupancy = {f:.2}");
        const F_THOLD: f64 = 0.25;
        if f > F_THOLD {
            eprintln!(
                "spec ngram-mod: ngram_mod occupancy {f:.2} exceeds threshold ({F_THOLD}) - resetting"
            );

            self.mod_.reset();
        }
    }

    /// `bool process(...) override { return true; }` (:1998-2001)
    fn process(&mut self, _batch: &LlamaBatch, _ctx_tgt: &DecodeContext) -> bool {
        true
    }

    /// `void draft(...) override` (:2003-2014)
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        for (seq_id, dp) in dparams.iter_mut().enumerate() {
            if !dp.drafting {
                continue;
            }

            self.draft_one(seq_id, dp);
        }
    }

    /// `void accept(seq_id, n_accepted, is_other) override` (:2016-2041): the
    /// low-acceptance streak resets the table
    fn accept(&mut self, seq_id: i32, n_accepted: u16, is_other: bool) {
        if is_other {
            return;
        }

        let sinfo = &mut self.sinfos[seq_id as usize];

        // compute acceptance fraction if we have a recorded draft length
        if sinfo.n_draft_last > 0 {
            let f_acc = n_accepted as f64 / sinfo.n_draft_last as f64;
            if f_acc < 0.25 {
                sinfo.n_low += 1;
                if sinfo.n_low >= 5 {
                    spec_dbg!(
                        "spec ngram-mod: low acceptance streak ({}) - resetting ngram_mod",
                        sinfo.n_low
                    );

                    self.mod_.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
}

/// `struct common_speculative_impl_ngram_cache` (speculative.cpp:2044-2181):
/// the lookup-decoding caches of [`crate::ngram_cache`] (context / dynamic /
/// static), one set per sequence.
pub struct CommonSpeculativeImplNgramCache {
    /// `uint16_t n_draft` — the TODO-configured draft length (:2211)
    n_draft: u16,

    /// `struct seq_info` (:2052-2058), one per sequence
    sinfos: Vec<NgramCacheSeqInfo>,

    stats: CommonSpeculativeImplStats,
}

struct NgramCacheSeqInfo {
    /// number of tokens in n-gram cache
    cache_size: usize,
    ngram_cache_context: crate::ngram_cache::CommonNgramCache,
    ngram_cache_dynamic: crate::ngram_cache::CommonNgramCache,
    ngram_cache_static: crate::ngram_cache::CommonNgramCache,
}

impl CommonSpeculativeImplNgramCache {
    /// the constructor (:2062-2109) — `create_state_ngram_cache`
    /// (speculative.cpp:2206-2220) resolves `n_draft = 8` and the two
    /// `save_*` flags (TODOs in the reference, both false) here
    pub fn new(params: &CommonParamsSpeculative, n_seq: u32, n_draft: u16) -> Result<Self, String> {
        let p = &params.ngram_cache;

        spec_dbg!(
            "spec ngram-cache: adding speculative implementation 'ngram-cache' (n_draft={n_draft}, \
             cache_static={}, cache_dynamic={})",
            if p.lookup_cache_static.is_empty() { "none" } else { &p.lookup_cache_static },
            if p.lookup_cache_dynamic.is_empty() { "none" } else { &p.lookup_cache_dynamic },
        );

        let mut sinfos: Vec<NgramCacheSeqInfo> = (0..n_seq)
            .map(|_| NgramCacheSeqInfo {
                cache_size: 0,
                ngram_cache_context: crate::ngram_cache::CommonNgramCache::new(),
                ngram_cache_dynamic: crate::ngram_cache::CommonNgramCache::new(),
                ngram_cache_static: crate::ngram_cache::CommonNgramCache::new(),
            })
            .collect();

        if !p.lookup_cache_static.is_empty() {
            let cache = crate::ngram_cache::common_ngram_cache_load(&p.lookup_cache_static)
                .map_err(|e| {
                    // SPC_ERR + GGML_ABORT ("Couldn't read static lookup
                    // cache", :2092-2094)
                    format!("Couldn't read static lookup cache: {e}")
                })?;
            for sinfo in sinfos.iter_mut() {
                sinfo.ngram_cache_static = cache.clone();
            }
        }

        if !p.lookup_cache_dynamic.is_empty() {
            let cache = crate::ngram_cache::common_ngram_cache_load(&p.lookup_cache_dynamic)
                .map_err(|e| {
                    format!("Couldn't read dynamic lookup cache: {e}") // :2104-2106
                })?;
            for sinfo in sinfos.iter_mut() {
                sinfo.ngram_cache_dynamic = cache.clone();
            }
        }

        Ok(CommonSpeculativeImplNgramCache {
            n_draft,
            sinfos,
            stats: CommonSpeculativeImplStats::default(),
        })
    }

    /// `draft_one` (speculative.cpp:2115-2158)
    fn draft_one(&mut self, seq_id: usize, dp: &mut CommonSpeculativeDraftParams) {
        let sinfo = &mut self.sinfos[seq_id];
        let prompt = &dp.prompt;

        // :2123-2137 — fold the new tokens (prompt tail + id_last) into the
        // context cache
        if sinfo.cache_size < prompt.len() + 1 {
            let mut tokens_new: Vec<i32> = Vec::with_capacity(prompt.len() + 1 - sinfo.cache_size);
            for &t in &prompt[sinfo.cache_size.min(prompt.len())..] {
                tokens_new.push(t);
            }
            tokens_new.push(dp.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            crate::ngram_cache::common_ngram_cache_update(
                &mut sinfo.ngram_cache_context,
                crate::ngram_cache::LLAMA_NGRAM_MIN,
                crate::ngram_cache::LLAMA_NGRAM_MAX,
                &tokens_new,
                tokens_new.len(),
                false,
            );
            sinfo.cache_size = prompt.len() + 1;
        }

        // :2139-2144 — inp = prompt + id_last
        let mut inp: Vec<i32> = Vec::with_capacity(prompt.len() + 1);
        inp.extend_from_slice(prompt);
        inp.push(dp.id_last);

        // :2146-2157
        let mut result = vec![dp.id_last];
        crate::ngram_cache::common_ngram_cache_draft(
            &inp,
            &mut result,
            self.n_draft as i32,
            crate::ngram_cache::LLAMA_NGRAM_MIN,
            crate::ngram_cache::LLAMA_NGRAM_MAX,
            &mut sinfo.ngram_cache_context,
            &mut sinfo.ngram_cache_dynamic,
            &mut sinfo.ngram_cache_static,
        );

        if !result.is_empty() {
            // delete first token in result (which is the id_last token)
            result.remove(0);
        }
        dp.result = result;
    }
}

impl CommonSpeculativeImpl for CommonSpeculativeImplNgramCache {
    fn ty(&self) -> CommonSpeculativeType {
        CommonSpeculativeType::NgramCache
    }

    fn n_max(&self) -> i32 {
        self.n_draft as i32
    }

    fn stats(&self) -> &CommonSpeculativeImplStats {
        &self.stats
    }

    fn stats_mut(&mut self) -> &mut CommonSpeculativeImplStats {
        &mut self.stats
    }

    /// `void begin(...) override { // noop }` (:2111-2113)
    fn begin(&mut self, _seq_id: i32, _prompt: &[i32]) {}

    /// `bool process(...) override { return true; }` (:2160-2163)
    fn process(&mut self, _batch: &LlamaBatch, _ctx_tgt: &DecodeContext) -> bool {
        true
    }

    /// `void draft(...) override` (:2165-2176)
    fn draft(
        &mut self,
        dparams: &mut [CommonSpeculativeDraftParams],
        _ctx_tgt: &mut DecodeContext,
    ) {
        for (seq_id, dp) in dparams.iter_mut().enumerate() {
            if !dp.drafting {
                continue;
            }

            self.draft_one(seq_id, dp);
        }
    }

    /// `void accept(...) override { // noop }` (:2178-2180)
    fn accept(&mut self, _seq_id: i32, _n_accepted: u16, _is_other: bool) {}
}

// ---------------------------------------------------------------------------
// common_speculative — the container + state machine
// (speculative.cpp:2183-2997)
// ---------------------------------------------------------------------------

/// `struct common_speculative` (speculative.cpp:2183-2193)
pub struct CommonSpeculative {
    /// `dparams` — one entry per sequence
    pub dparams: Vec<CommonSpeculativeDraftParams>,

    /// list of implementations to use and their states
    pub impls: Vec<Box<dyn CommonSpeculativeImpl>>,

    /// which implementation was used for a given seq_id (the C stores a
    /// pointer, speculative.cpp:2190; the port stores an index into `impls`)
    pub impl_last: Vec<Option<usize>>,

    /// `synth_probs` — conditional synthetic acceptance probabilities
    pub synth_probs: Vec<f64>,
}

/// `common_speculative_init` returns `nullptr` when no implementation was
/// selected (speculative.cpp:2724-2727) — the port's `Ok(None)`.
pub type CommonSpeculativeInitResult = Result<Option<CommonSpeculative>, String>;

/// `common_speculative_init(common_params_speculative & params, uint32_t n_seq)`
/// (speculative.cpp:2619-2762).
///
/// Rust ownership deviation: the C reads `params.draft.ctx_dft` /
/// `params.draft.ctx_tgt` (raw pointers built by
/// `common_speculative_init_from_params`, speculative.cpp:2523-2604) and gets
/// the vocabs through `llama_get_model(ctx)`. The port takes the target
/// context mutably (draft-mtp switches its `embeddings_nextn` tap on,
/// speculative.cpp:1420), the draft context by value (it is what the impls
/// decode on), both vocabs explicitly and the draft model's
/// `n_layer_nextn` (the MTP head count of :1377).
#[allow(clippy::too_many_arguments)]
pub fn common_speculative_init(
    params: &CommonParamsSpeculative,
    n_seq: u32,
    ctx_tgt: &mut DecodeContext,
    ctx_dft: Option<DecodeContext>,
    vocab_tgt: &Vocab,
    vocab_dft: Option<&Vocab>,
    n_layer_nextn_dft: u32,
    // the gemma4-assistant head is attached to the target context
    // (`is_mem_shared`, speculative.cpp:1423) — the mem-shared draft-mtp mode
    gemma4_shared: bool,
) -> CommonSpeculativeInitResult {
    // Compute the implementations to use based on the config and their order of
    // preference (speculative.cpp:2620-2647)
    let mut configs: Vec<CommonSpeculativeType> = Vec::new();
    {
        let enabled_configs: u32 = params
            .types
            .iter()
            .fold(0u32, |acc, &t| acc | (1u32 << (t as u32)));

        let mut add_config_if_enabled = |ty: CommonSpeculativeType, available: bool| {
            if available && (enabled_configs & (1u32 << (ty as u32))) != 0 {
                configs.push(ty);
            }
        };

        // this list here defines the priority of the speculators; the ones with
        // the highest priority are listed first (speculative.cpp:2634-2646)
        add_config_if_enabled(CommonSpeculativeType::NgramSimple, true);
        add_config_if_enabled(CommonSpeculativeType::NgramMapK, true);
        add_config_if_enabled(CommonSpeculativeType::NgramMapK4v, true);
        add_config_if_enabled(CommonSpeculativeType::NgramMod, true);
        add_config_if_enabled(CommonSpeculativeType::NgramCache, true);

        add_config_if_enabled(CommonSpeculativeType::DraftSimple, true);
        // the C gates eagle3/mtp/dflash/dspark on `params.draft.ctx_dft != nullptr`
        // (speculative.cpp:2643-2646)
        let has_dft_ctx = ctx_dft.is_some();
        add_config_if_enabled(CommonSpeculativeType::DraftEagle3, has_dft_ctx);
        add_config_if_enabled(CommonSpeculativeType::DraftMtp, has_dft_ctx || gemma4_shared);
        add_config_if_enabled(CommonSpeculativeType::DraftDflash, has_dft_ctx);
        add_config_if_enabled(CommonSpeculativeType::DraftDspark, has_dft_ctx);
    }

    let mut impls: Vec<Box<dyn CommonSpeculativeImpl>> = Vec::new();
    let mut ctx_dft = ctx_dft;

    for &ty in &configs {
        match ty {
            CommonSpeculativeType::None => {}
            CommonSpeculativeType::DraftSimple => {
                // `throw std::runtime_error("draft-simple requires a draft context")`
                // (speculative.cpp:193-195)
                let Some(ctx) = ctx_dft.take() else {
                    return Err(
                        "draft-simple requires a draft context (pass -md / --spec-draft-model)"
                            .into(),
                    );
                };
                let Some(vocab_dft) = vocab_dft else {
                    return Err("draft-simple requires the draft model's vocabulary".into());
                };
                impls.push(Box::new(CommonSpeculativeImplDraftSimple::new(
                    params, n_seq, ctx, vocab_tgt, vocab_dft,
                )?));
            }
            CommonSpeculativeType::DraftMtp => {
                // the gemma4 mem-shared mode — the head rides the target
                // context (speculative.cpp:1423 `is_mem_shared`); no draft
                // context, the taps and the drafts run over the shared KV
                if gemma4_shared {
                    let Some(vocab_dft) = vocab_dft else {
                        return Err("draft-mtp requires the draft model's vocabulary".into());
                    };
                    let n_embd = {
                        let head = ctx_tgt
                            .gemma4_assistant
                            .as_ref()
                            .ok_or("draft-mtp: the gemma4-assistant head is not attached")?;
                        head.params.n_embd_backbone as usize
                    };
                    // llama_set_embeddings_nextn(ctx_tgt, true, false)
                    // (speculative.cpp:1420)
                    ctx_tgt.set_embeddings_nextn(true, false);
                    impls.push(Box::new(CommonSpeculativeImplDraftMtp::new_gemma4_shared(
                        params, n_seq, n_embd, vocab_dft,
                    )));
                    continue;
                }
                // the MTP draft context is the target model's own — created
                // by the caller (`common_speculative_init_from_params`,
                // speculative.cpp:2545-2589)
                let Some(mut ctx) = ctx_dft.take() else {
                    return Err(
                        "draft-mtp requires an MTP draft context over the target model".into(),
                    );
                };
                let Some(vocab_dft) = vocab_dft else {
                    return Err("draft-mtp requires the draft model's vocabulary".into());
                };
                // llama_set_embeddings_nextn(ctx_tgt, true, false) +
                // (ctx_dft, true, true) (speculative.cpp:1420-1421)
                ctx_tgt.set_embeddings_nextn(true, false);
                ctx.set_embeddings_nextn(true, true);
                impls.push(Box::new(CommonSpeculativeImplDraftMtp::new(
                    params,
                    n_seq,
                    ctx,
                    ctx_tgt,
                    n_layer_nextn_dft,
                    vocab_dft,
                )?));
            }
            CommonSpeculativeType::DraftEagle3 => {
                // the eagle3 head context — created by the caller over the
                // head model (`common_speculative_init_from_params`'s
                // has_draft arm, speculative.cpp:2553-2576, with
                // `cparams.ctx_other = ctx_tgt`)
                let Some(ctx) = ctx_dft.take() else {
                    return Err(
                        "draft-eagle3 requires an eagle3 head context (pass -md <head.gguf> / \
                         --spec-draft-model)"
                            .into(),
                    );
                };
                let Some(vocab_dft) = vocab_dft else {
                    return Err("draft-eagle3 requires the draft model's vocabulary".into());
                };
                // the context-level taps of the constructor
                // (speculative.cpp:513-526): turn on extraction of the target
                // layers' hidden states — layer < n_layer reads the
                // layer-input tap, == n_layer the unmasked nextn tap (which
                // archs without a trunk nextn tap refuse, like the C's
                // GGML_ABORT at :612-613) — and the draft model's pre-norm
                // hidden state
                let (target_layer_ids, n_embd_tgt) = {
                    let e = ctx
                        .eagle
                        .as_ref()
                        .expect("draft-eagle3: the draft context is not an eagle3 head context");
                    (e.1.target_layer_ids.to_vec(), e.1.n_embd_tgt as usize)
                };
                let n_layer_tgt = ctx_tgt.n_layer();
                for &lid in &target_layer_ids {
                    if (lid as usize) < n_layer_tgt {
                        ctx_tgt.set_embeddings_layer_inp(lid as u32, true);
                    } else if lid as usize == n_layer_tgt {
                        ctx_tgt.set_embeddings_nextn(true, false);
                    } else {
                        panic!(
                            "EAGLE3: target layer id {lid} exceeds target n_layer {n_layer_tgt}"
                        );
                    }
                }
                // `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true)`
                // (:526)
                let mut ctx = ctx;
                ctx.set_embeddings_nextn(true, true);
                impls.push(Box::new(CommonSpeculativeImplDraftEagle3::new(
                    params,
                    n_seq,
                    ctx,
                    ctx_tgt,
                    target_layer_ids,
                    n_embd_tgt,
                    vocab_dft,
                )?));
            }
            CommonSpeculativeType::DraftDflash | CommonSpeculativeType::DraftDspark => {
                // (speculative.cpp:2667-2675) — one impl for both types; the
                // DSPARK construction passes its type through, which the
                // impl reads back as `is_dspark` (:944-948)
                let Some(mut ctx) = ctx_dft.take() else {
                    return Err(format!(
                        "'{}' requires a dflash draft context (pass -md <dflash.gguf> / \
                         --spec-draft-model)",
                        ty.to_str()
                    ));
                };
                let Some(vocab_dft) = vocab_dft else {
                    return Err(format!(
                        "'{}' requires the draft model's vocabulary",
                        ty.to_str()
                    ));
                };
                // the target-side taps of the constructor
                // (speculative.cpp:1046-1048): turn on extraction of the
                // target layers' input embeddings. layer < n_layer reads the
                // layer-input tap; == n_layer the unmasked nextn tap (the
                // eagle3 arm's rule) — a lid beyond n_layer aborts like the
                // C's "layer input tensor not found"
                let (target_layer_ids, n_embd_tgt) = {
                    let d = ctx
                        .dflash
                        .as_ref()
                        .expect("draft-dflash: the draft context is not a dflash draft context");
                    (d.1.target_layer_ids.to_vec(), d.1.n_embd_tgt as usize)
                };
                let n_layer_tgt = ctx_tgt.n_layer();
                for &lid in &target_layer_ids {
                    if (lid as usize) < n_layer_tgt {
                        ctx_tgt.set_embeddings_layer_inp(lid as u32, true);
                    } else if lid as usize == n_layer_tgt {
                        ctx_tgt.set_embeddings_nextn(true, false);
                    } else {
                        panic!(
                            "DFlash: target layer id {lid} exceeds target n_layer {n_layer_tgt}"
                        );
                    }
                }
                let _ = n_embd_tgt;
                impls.push(Box::new(CommonSpeculativeImplDraftDflash::new(
                    params, n_seq, ctx, ctx_tgt, vocab_dft,
                )?));
            }
            // the ngram family (speculative.cpp:2676-2718)
            CommonSpeculativeType::NgramSimple => {
                // get_common_ngram_map (:2195-2204) over params.ngram_simple,
                // then the simple config from its sizes (:2677-2685)
                let map = params.ngram_simple.clone();
                let state = CommonSpeculativeImplNgramSimple {
                    size_n: map.size_n,
                    size_m: map.size_m,
                    _min_hits: map.min_hits,
                    stats: CommonSpeculativeImplStats::default(),
                };
                impls.push(Box::new(state));
            }
            CommonSpeculativeType::NgramMapK => {
                // get_common_ngram_map (speculative.cpp:2195-2204)
                let m = params.ngram_map_k.clone();
                let mut config = Vec::with_capacity(n_seq as usize);
                for _ in 0..n_seq {
                    config.push(crate::ngram_map::CommonNgramMap::new(
                        m.size_n, m.size_m, true, m.min_hits,
                    ));
                }
                impls.push(Box::new(CommonSpeculativeImplNgramMapK {
                    ty: CommonSpeculativeType::NgramMapK,
                    config,
                    stats: CommonSpeculativeImplStats::default(),
                }));
            }
            CommonSpeculativeType::NgramMapK4v => {
                let m = params.ngram_map_k4v.clone();
                let mut config = Vec::with_capacity(n_seq as usize);
                for _ in 0..n_seq {
                    config.push(crate::ngram_map::CommonNgramMap::new(
                        m.size_n, m.size_m, false, m.min_hits,
                    ));
                }
                impls.push(Box::new(CommonSpeculativeImplNgramMapK {
                    ty: CommonSpeculativeType::NgramMapK4v,
                    config,
                    stats: CommonSpeculativeImplStats::default(),
                }));
            }
            CommonSpeculativeType::NgramMod => {
                impls.push(Box::new(CommonSpeculativeImplNgramMod::new(params, n_seq)));
            }
            CommonSpeculativeType::NgramCache => {
                // create_state_ngram_cache (speculative.cpp:2206-2220):
                // n_draft 8 (TODO get from config?), both save flags false
                impls.push(Box::new(CommonSpeculativeImplNgramCache::new(
                    params, n_seq, 8,
                )?));
            }
        }
    }

    if impls.is_empty() {
        // SPC_TRC("no implementations specified for speculative decoding")
        // (speculative.cpp:2724-2727) → nullptr
        return Ok(None);
    }

    let mut result = CommonSpeculative {
        dparams: (0..n_seq as usize)
            .map(|_| CommonSpeculativeDraftParams::default())
            .collect(),
        impls,
        impl_last: vec![None; n_seq as usize],
        synth_probs: Vec::new(),
    };

    // synthetic acceptance probabilities (speculative.cpp:2736-2759)
    let n_max_configured = common_speculative_n_max_params(params);
    let n_max_effective = common_speculative_n_max(&result);
    let rates = common_speculative_synth_rates_resolve(params, n_max_effective)?;

    let mut rates_str: Vec<String> = Vec::with_capacity(rates.len());
    result.synth_probs.reserve(rates.len());
    let mut rate_prev = 1.0;
    let mut acceptance_length = 1.0;
    for &rate in &rates {
        result.synth_probs.push(if rate_prev > 0.0 {
            rate / rate_prev
        } else {
            0.0
        });
        rates_str.push(format!("{rate:.6}"));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if !result.synth_probs.is_empty() {
        eprintln!(
            "spec: synthetic speculative acceptance is enabled for benchmarking; generated \
             output is not valid"
        );
        if n_max_effective != n_max_configured {
            eprintln!(
                "spec: synthetic acceptance draft limit was reduced from {n_max_configured} to \
                 {n_max_effective} by the initialized speculative implementations"
            );
        }
        eprintln!(
            "spec: synthetic acceptance: n_max = {}, mean length = {acceptance_length:.6}, \
             rates = [{}]",
            rates.len(),
            rates_str.join(", ")
        );
    }

    Ok(Some(result))
}

/// `int32_t common_speculative_n_max(const common_speculative *)`
/// (speculative.cpp:2371-2383)
pub fn common_speculative_n_max(spec: &CommonSpeculative) -> i32 {
    let mut n_max = 0;

    for impl_ in &spec.impls {
        n_max = n_max.max(impl_.n_max().max(0));
    }

    n_max
}

/// `const std::vector<double> & common_speculative_get_synth_probs`
/// (speculative.cpp:2461-2464)
pub fn common_speculative_get_synth_probs(spec: &CommonSpeculative) -> &[f64] {
    &spec.synth_probs
}

impl CommonSpeculative {
    /// `common_speculative_get_draft_params` (speculative.cpp:2772-2779)
    pub fn get_draft_params(&mut self, seq_id: i32) -> &mut CommonSpeculativeDraftParams {
        assert!(seq_id >= 0 && (seq_id as usize) < self.dparams.len());
        &mut self.dparams[seq_id as usize]
    }

    /// `common_speculative_begin` (speculative.cpp:2781-2791)
    pub fn begin(&mut self, seq_id: i32, prompt: &[i32]) {
        for impl_ in self.impls.iter_mut() {
            // `common_time_meas tm(impl->t_begin_us, !impl->gen_perf)`
            let t0 = now_us();
            impl_.begin(seq_id, prompt);
            let dt = now_us() - t0;

            let stats = impl_.stats_mut();
            stats.t_begin_us += dt;
            stats.n_call_begin += 1;
        }
    }

    /// `common_speculative_process` (speculative.cpp:2793-2805). The target
    /// context rides along for draft-mtp's `embeddings_nextn` reads.
    pub fn process(&mut self, batch: &LlamaBatch, ctx_tgt: &DecodeContext) -> bool {
        let mut result = true;

        for impl_ in self.impls.iter_mut() {
            result = impl_.process(batch, ctx_tgt) && result;
        }

        result
    }

    /// `common_speculative_draft` (speculative.cpp:2807-2890). The target
    /// context rides along for the gemma4 mem-shared draft-mtp mode (the C's
    /// impls reach it through their `params.ctx_tgt` member).
    pub fn draft(&mut self, ctx_tgt: &mut DecodeContext) {
        let CommonSpeculative {
            dparams,
            impls,
            impl_last,
            ..
        } = self;

        {
            let mut n_drafting = 0;

            for dp in dparams.iter() {
                assert!(!dp.drafting || dp.result.is_empty());

                if dp.drafting {
                    n_drafting += 1;
                }
            }

            if n_drafting == 0 {
                return;
            }
        }

        for (impl_idx, impl_) in impls.iter_mut().enumerate() {
            {
                // `common_time_meas tm(impl->t_draft_us, !impl->gen_perf)`
                let t0 = now_us();
                impl_.draft(dparams, ctx_tgt);
                let dt = now_us() - t0;

                let stats = impl_.stats_mut();
                stats.t_draft_us += dt;
                stats.n_call_draft += 1;
            }

            let mut n_drafting = 0;

            for seq_id in 0..dparams.len() {
                let dp = &mut dparams[seq_id];

                if !dp.drafting {
                    continue;
                }

                // a new draft has been sampled
                if !dp.result.is_empty() {
                    dp.drafting = false;

                    if dp.n_max > 0 && dp.result.len() > dp.n_max as usize {
                        spec_dbg!("spec: truncating draft to {} tokens", dp.n_max);
                        dp.result.resize(dp.n_max as usize, 0);

                        // trim the candidates only if the drafter produced
                        // them (n-gram drafters do not)
                        // (speculative.cpp:2917-2921, upstream a7b94df2c)
                        if let Some(rq) = dp.result_q.as_mut() {
                            if !rq.is_empty() {
                                rq.truncate(dp.n_max as usize);
                            }
                        }
                    }
                    // the chain probe's per-round draft print
                    // (parity/ref_dflash_chain.cpp's round_draft:) — the
                    // driver-observed, post-truncation draft vector
                    spec_dbg!("spec: - seq_id {seq_id}, round draft: {:?}", dp.result);

                    if !dp.result.is_empty() {
                        spec_dbg!(
                            "spec: called impl {}, hist size = {}, call_count = {}, gen = {}",
                            impl_.ty().to_str(),
                            dp.prompt.len(),
                            impl_.stats().n_call_draft,
                            dp.result.len()
                        );

                        // remember which implementation was used
                        impl_last[seq_id] = Some(impl_idx);

                        let n = dp.result.len() as u64;
                        let stats = impl_.stats_mut();
                        stats.n_gen_drafts += 1;
                        stats.n_gen_tokens += n;
                    }
                }

                if dp.drafting {
                    n_drafting += 1;
                }
            }

            if n_drafting == 0 {
                break;
            }
        }

        // these sequences failed to generate a draft
        for dp in dparams.iter_mut() {
            if dp.drafting {
                dp.drafting = false;
            }
        }
    }

    /// `common_speculative_accept` (speculative.cpp:2892-2926)
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) {
        let Some(impl_idx) = self.impl_last[seq_id as usize] else {
            assert_eq!(n_accepted, 0);
            return;
        };

        {
            let impl_ = &mut self.impls[impl_idx];

            // `common_time_meas tm(impl->t_accept_us, !impl->gen_perf)`
            let t0 = now_us();

            {
                let stats = impl_.stats_mut();
                if stats.n_acc_tokens_per_pos.len() < n_accepted as usize {
                    stats.n_acc_tokens_per_pos.resize(n_accepted as usize, 0);
                }

                for i in 0..n_accepted as usize {
                    stats.n_acc_tokens_per_pos[i] += 1;
                }

                if n_accepted > 0 {
                    stats.n_acc_drafts += 1;
                    stats.n_acc_tokens += n_accepted as u64;
                }
            }

            impl_.accept(seq_id, n_accepted, false);

            let dt = now_us() - t0;
            let stats = impl_.stats_mut();
            stats.t_accept_us += dt;
            stats.n_call_accept += 1;
        }

        // accept with the rest of the implementations, using is_other == true
        for (i, impl_other) in self.impls.iter_mut().enumerate() {
            if i != impl_idx {
                impl_other.accept(seq_id, n_accepted, true);
            }
        }
    }

    /// the draft context the driver must keep in sync with the target's KV
    /// (`params.draft.ctx_dft` in the C). The C's driver keeps its own pointer;
    /// here the context is owned by the impl, so hand it out mutably.
    pub fn ctx_dft(&mut self) -> Option<&mut DecodeContext> {
        for impl_ in self.impls.iter_mut() {
            if let Some(ctx) = impl_.ctx_dft() {
                return Some(ctx);
            }
        }
        None
    }

    /// the stats of the implementation used for a sequence, if any
    /// (`spec->impl_last[seq_id]`)
    pub fn impl_stats(&self, seq_id: i32) -> Option<&CommonSpeculativeImplStats> {
        self.impl_last[seq_id as usize].map(|i| self.impls[i].stats())
    }

    /// `common_speculative_print_stats` (speculative.cpp:2953-2997). The C
    /// LOG_TRC's one block per implementation; the port returns the same text
    /// (a line per impl) and lets the caller print it.
    pub fn print_stats(&self) -> String {
        let mut out = String::new();

        for impl_ in &self.impls {
            let stats = impl_.stats();

            let str_perf = format!(
                ", dur(b,g,a) = {:.3}, {:.3}, {:.3} ms",
                stats.t_begin_us as f64 / 1000.0,
                stats.t_draft_us as f64 / 1000.0,
                stats.t_accept_us as f64 / 1000.0
            );

            let mut str_stats = String::new();
            if stats.n_call_accept > 0 {
                let mean = stats.mean_acc_len();
                let per_pos: Vec<String> = stats
                    .n_acc_tokens_per_pos
                    .iter()
                    .map(|&c| format!("{:.3}", c as f64 / stats.n_call_accept as f64))
                    .collect();
                str_stats = format!(
                    ", #mean acc len = {:.2}, #acc rate/pos = ({})",
                    mean,
                    per_pos.join(", ")
                );
            }

            out.push_str(&format!(
                "statistics {:>16}: #calls(b,g,a) = {:4} {:6} {:6}, #gen drafts = {:6}, \
                 #acc drafts = {:5}, #gen tokens = {:6}, #acc tokens = {:5}{}{}\n",
                impl_.ty().to_str(),
                stats.n_call_begin,
                stats.n_call_draft,
                stats.n_call_accept,
                stats.n_gen_drafts,
                stats.n_acc_drafts,
                stats.n_gen_tokens,
                stats.n_acc_tokens,
                str_stats,
                str_perf,
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// the verification rule: `common_sampler_sample_and_accept_n`
// (common/sampling.cpp:678-715)
// ---------------------------------------------------------------------------

/// `common_sampler_sample_and_accept_n(gsmpl, ctx, draft)`
/// (common/sampling.cpp:708-715 with the idxs overload :678-706).
///
/// For each draft token: sample the target's token at that batch row and
/// `common_sampler_accept` it — whether it matches or not, because the
/// mismatching token *is* the next output token — then keep it while it equals
/// the draft entry. When the whole draft matched, one more token is sampled
/// from the bonus row (the driver asserts `ids.size() > 0`,
/// speculative-simple.cpp:262).
///
/// Sampling-state rule (the C's answer to "advance once per accepted token"):
/// `common_sampler_accept` runs exactly once per token pushed into the result,
/// which is exactly the sequence of tokens the target emits — the chain stays
/// consistent with the KV cache and this path needs no rollback. The checkpoint
/// path of `examples/speculative-simple` (speculative-simple.cpp:242-246 +
/// :285) restores a *clone* of the sampler instead, because it re-samples the
/// same positions in the next iteration; that path is only taken for contexts
/// without partial `seq_rm`, which the port's KV cache does not have.
pub fn common_sampler_sample_and_accept_n(
    smpl: &mut SamplingContext,
    vocab: &crate::vocab::Vocab,
    out: &BatchOutput,
    draft: &[i32],
) -> Vec<i32> {
    let mut result: Vec<i32> = Vec::with_capacity(draft.len() + 1);

    let mut i = 0usize;
    while i < draft.len() {
        let logits = out.logits_ith(i as i32).expect("verify logits row");
        let id = smpl.sample(logits); // common_sampler_sample + common_sampler_accept

        result.push(id);

        // do not accept draft tokens after an EOG - they are not output but
        // would stay in the context; on replay the last token is from the
        // target and can be EOG, so a trailing EOG is still accepted
        // (common/sampling.cpp:694-695)
        if draft[i] != id || (vocab.is_eog(id) && i + 1 < draft.len()) {
            break;
        }

        i += 1;
    }

    if i == draft.len() {
        let logits = out.logits_ith(i as i32).expect("bonus logits row");
        let id = smpl.sample(logits);

        result.push(id);
    }

    result
}

// ---------------------------------------------------------------------------
// the driver loop (examples/speculative-simple/speculative-simple.cpp:126-342)
// ---------------------------------------------------------------------------

/// counters + output of [`speculative_simple_generate`]
#[derive(Clone, Debug, Default)]
pub struct SpeculativeSimpleResult {
    /// the generated tokens (the C's `ids` stream, printed through
    /// `common_token_to_piece`, speculative-simple.cpp:305-322)
    pub tokens: Vec<i32>,
    /// `n_predict` — tokens committed (draft tokens + target tokens)
    pub n_predict: i32,
    /// `n_drafted` (speculative-simple.cpp:296)
    pub n_drafted: i32,
    /// `n_accept` (speculative-simple.cpp:297)
    pub n_accept: i32,
    /// number of `llama_decode(ctx_tgt)` calls (target forward passes)
    pub n_target_forward: usize,
    /// number of `llama_decode(ctx_dft)` calls (draft forward passes)
    pub n_draft_forward: usize,
    /// wall time of the generation loop in µs
    pub t_us: i64,
    /// true if an end-of-generation token was produced
    pub has_eos: bool,
}

/// The speculative decode loop of `examples/speculative-simple`
/// (speculative-simple.cpp:126-342), restricted to the path the pinned
/// reference takes for contexts with partial `seq_rm` (which is everything the
/// port's KV cache supports: `common_context_can_seq_rm` returns
/// `COMMON_CONTEXT_SEQ_RM_TYPE_PART` for the full-attention archs,
/// common/common.cpp:1583-1620):
///
///   * the prompt is evaluated on the target and fed to the speculator
///     (`common_speculative_process`, :135),
///   * each iteration drafts (`common_speculative_draft`, :196), rolls the
///     draft context back to the prompt prefix (:206-213), evaluates
///     `[id_last, draft...]` on the target in one batch (:222-234), feeds the
///     same batch to the speculator (:237), samples/accepts
///     (`common_sampler_sample_and_accept_n`, :258), reports the acceptance
///     (:292) and drops the tail of both KV caches beyond `n_past` (:329-337).
///
/// The checkpoint-based partial-acceptance fallback (:242-290, taken when the
/// context cannot remove partial sequences) is not ported — see
/// [`common_sampler_sample_and_accept_n`].
pub fn speculative_simple_generate(
    tgt: &mut DecodeContext,
    spec: &mut CommonSpeculative,
    smpl: &mut SamplingContext,
    vocab_tgt: &Vocab,
    inp: &[i32],
    n_predict_limit: i32,
) -> Result<SpeculativeSimpleResult, String> {
    // `if (params.n_predict < -1)` (speculative-simple.cpp:28-31)
    if n_predict_limit < -1 {
        return Err("--n-predict must be >= -1".into());
    }

    let mut result = SpeculativeSimpleResult::default();

    let n_ctx_tgt = tgt.kv.size as i32;

    // "the prompt exceeds the context size / batch size" (:80-90)
    if n_ctx_tgt < inp.len() as i32 {
        return Err(format!(
            "the prompt exceeds the context size ({} tokens, ctx {n_ctx_tgt})",
            inp.len()
        ));
    }
    if (tgt.n_batch as i32) < inp.len() as i32 {
        return Err(format!(
            "the prompt exceeds the batch size ({} tokens, batch {})",
            inp.len(),
            tgt.n_batch
        ));
    }
    if inp.is_empty() {
        return Err("empty prompt".into());
    }

    let t_dec_start = now_us();

    let seq_id = 0;

    // eval the prompt on the target and feed it to the speculative
    // implementation(s) (:126-139)
    {
        let mut batch_prompt = LlamaBatch::default();
        for i in 0..inp.len() - 1 {
            batch_prompt.add(inp[i], i as i32, &[seq_id], false);
        }

        tgt.decode_batch(&batch_prompt)?;
        result.n_target_forward += 1;

        if !spec.process(&batch_prompt, tgt) {
            return Err("failed to process speculative prompt".into());
        }
    }

    // note: keep the last token separate!
    let mut id_last = inp[inp.len() - 1];

    // all tokens currently in the target context (:145)
    let mut prompt_tgt: Vec<i32> = inp[..inp.len() - 1].to_vec();

    let mut n_past = inp.len() as i32 - 1;

    spec.begin(seq_id, &prompt_tgt);

    let mut batch_tgt = LlamaBatch::default();

    let mut draft: Vec<i32> = Vec::new();

    let mut n_predict = 0;
    let mut n_drafted = 0;
    let mut n_accept = 0;
    let mut has_eos = false;

    // the relevant stuff for speculative decoding starts here
    // (speculative-simple.cpp:107-111)

    loop {
        // generate or reuse draft tokens (:162-219) — this is the most
        // important part of the speculation: the more probable tokens are
        // provided here, the better the performance
        if draft.is_empty() {
            // determine the max draft that fits the remaining context and
            // generation budget (:180-185)
            let mut n_draft_max = n_ctx_tgt - n_past - 2;
            if n_predict_limit >= 0 {
                n_draft_max = n_draft_max.min(n_predict_limit - n_predict - 1);
            }
            n_draft_max = n_draft_max.max(0);

            // generate a new draft (:187-196)
            {
                let dp = spec.get_draft_params(seq_id);
                dp.drafting = true;
                dp.n_max = n_draft_max;
                dp.pos0 = n_past;
                dp.id_last = id_last;
                dp.prompt = prompt_tgt.clone();
                dp.result = std::mem::take(&mut draft);
            }
            spec.draft(tgt);
            draft = std::mem::take(&mut spec.get_draft_params(seq_id).result);

            // save a checkpoint of the target context before evaluating the
            // draft (:198-204, only used by the non-ported checkpoint path);
            // then reset the draft context to the checkpoint before
            // verification (:206-213): `llama_memory_seq_rm(ctx_dft, seq_id,
            // ckpt.pos_max + 1, -1)` with `ckpt.pos_max` = the target's
            // `seq_pos_max` == n_past - 1, i.e. drop every draft cell past the
            // prompt prefix. This is what removes the cells draft() just wrote
            // for the seed + draft tokens, so the verify batch below starts from
            // the same state on both models.
            if let Some(dft) = spec.ctx_dft() {
                dft.seq_rm(seq_id, n_past, -1);
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past, &[seq_id], true);
        n_past += 1;

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        // (:222-234) — `llama_decode(ctx_tgt, batch_tgt)` + the context's
        // logits buffer (the C reads it back per row via
        // `llama_get_logits_ith`, the port gets it as `BatchOutput`)
        let out = {
            for (i, &d) in draft.iter().enumerate() {
                batch_tgt.add(d, n_past + i as i32, &[seq_id], true);
            }

            let out = tgt.decode_batch(&batch_tgt)?;
            result.n_target_forward += 1;
            out
        };

        // feed the batch to the speculative implementation(s) — this drives the
        // draft model (:236-240)
        if !spec.process(&batch_tgt, tgt) {
            return Err("failed to process speculative batch".into());
        }

        // save the size of the draft being verified (:248-249)
        let n_draft = draft.len();

        // sample from the full target batch and return the accepted tokens
        // based on the target sampler (:251-258)
        let ids = common_sampler_sample_and_accept_n(smpl, vocab_tgt, &out, &draft);

        assert!(!ids.is_empty()); // there will always be at least one accepted token

        // full acceptance: consume the draft and commit accepted tokens
        // (:292-298). The partial-acceptance checkpoint branch (:264-290) is
        // not ported: the port's KV cache removes partial sequences, so the
        // direct path below is the one the reference takes for it.
        spec.accept(seq_id, ids.len() as u16 - 1);

        n_past += ids.len() as i32 - 1;
        n_drafted += n_draft as i32; // note: we ignore the discarded small drafts
        n_accept += ids.len() as i32 - 1;
        n_predict += ids.len() as i32;

        // process the accepted tokens and update contexts (:300-322)
        for &id in ids.iter() {
            prompt_tgt.push(id_last);

            id_last = id;

            if vocab_tgt.is_eog(id_last) {
                has_eos = true;
                break;
            }

            result.tokens.push(id_last);
        }

        // clear the draft since it has been consumed (:326)
        draft.clear();

        {
            // clear kv cache from any extra tokens, n_past (:329-337)
            tgt.seq_rm(seq_id, n_past, -1);

            if let Some(dft) = spec.ctx_dft() {
                dft.seq_rm(seq_id, n_past, -1);
            }
        }

        if (n_predict_limit >= 0 && n_predict > n_predict_limit) || has_eos {
            break;
        }
    }

    let t_dec_end = now_us();

    result.n_predict = n_predict;
    result.n_drafted = n_drafted;
    result.n_accept = n_accept;
    result.has_eos = has_eos;
    result.t_us = t_dec_end - t_dec_start;
    // the draft model's own forward passes (port-only accounting; the C gets
    // them from `llama_perf_context(ctx_dft)`)
    result.n_draft_forward = spec
        .impls
        .iter()
        .map(|i| i.stats().n_forward as usize)
        .sum();

    Ok(result)
}

// ---------------------------------------------------------------------------
// unit tests for the pure parts (speculative.cpp:2222-2280, 2335-2464,
// 2606-2615)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_roundtrip() {
        // common_speculative_type_to_str / _from_name (speculative.cpp:2246-2288)
        for ty in CommonSpeculativeType::ALL {
            assert_eq!(CommonSpeculativeType::from_name(ty.to_str()), Some(ty));
        }
        assert_eq!(CommonSpeculativeType::from_name("bogus"), None);

        // common_speculative_all_types_str (speculative.cpp:2234-2244)
        assert_eq!(
            common_speculative_all_types_str(),
            "none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,\
             ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache"
        );
        assert_eq!(
            common_speculative_type_name_str(&[
                CommonSpeculativeType::NgramSimple,
                CommonSpeculativeType::DraftSimple
            ]),
            "ngram-simple,draft-simple"
        );
    }

    #[test]
    fn types_from_names_none_short_circuits() {
        // speculative.cpp:2270-2273 — a "none" entry returns just { NONE }
        let names = vec!["draft-simple".to_string(), "none".to_string()];
        assert_eq!(
            common_speculative_types_from_names(&names).unwrap(),
            vec![CommonSpeculativeType::None]
        );
        assert!(common_speculative_types_from_names(&["nope".to_string()]).is_err());
        assert_eq!(
            common_speculative_types_from_names(&["draft-simple".to_string()]).unwrap(),
            vec![CommonSpeculativeType::DraftSimple]
        );
    }

    #[test]
    fn n_max_from_params() {
        // speculative.cpp:2335-2369
        let mut params = CommonParamsSpeculative::default();
        assert_eq!(
            common_speculative_n_max_params(&params),
            0,
            "types = {{NONE}}"
        );

        params.types = vec![CommonSpeculativeType::DraftSimple];
        params.draft.n_max = 5;
        assert_eq!(common_speculative_n_max_params(&params), 5);

        params.types = vec![
            CommonSpeculativeType::DraftSimple,
            CommonSpeculativeType::NgramMod,
        ];
        assert_eq!(common_speculative_n_max_params(&params), 64); // ngram-mod wins

        params.draft.n_max = -3;
        params.types = vec![CommonSpeculativeType::DraftSimple];
        assert_eq!(common_speculative_n_max_params(&params), 0); // max(0, -3)
    }

    #[test]
    fn synth_rates() {
        // speculative.cpp:2385-2459
        let params = CommonParamsSpeculative::default();
        assert!(common_speculative_synth_rates_resolve(&params, 3)
            .unwrap()
            .is_empty());

        let mut params = CommonParamsSpeculative::default();
        params.synth_len = 2.0;
        params.synth_rates = vec![0.5];
        assert!(common_speculative_synth_rates_resolve(&params, 1).is_err());

        let mut params = CommonParamsSpeculative::default();
        params.synth_len = 4.0; // == n_max + 1 → p = 1 → every rate is 1
        assert_eq!(
            common_speculative_synth_rates_resolve(&params, 3).unwrap(),
            vec![1.0, 1.0, 1.0]
        );

        // mean length = 1 + sum(rates) == the requested length
        let mut params = CommonParamsSpeculative::default();
        params.synth_len = 2.5;
        let rates = common_speculative_synth_rates_resolve(&params, 4).unwrap();
        let mean: f64 = 1.0 + rates.iter().sum::<f64>();
        assert!((mean - 2.5).abs() < 1e-6, "mean = {mean}");

        // rates are used verbatim, monotonicity enforced
        let mut params = CommonParamsSpeculative::default();
        params.synth_rates = vec![0.8, 0.4, 0.2];
        assert_eq!(
            common_speculative_synth_rates_resolve(&params, 3).unwrap(),
            vec![0.8, 0.4, 0.2]
        );
        params.synth_rates = vec![0.2, 0.4];
        assert!(common_speculative_synth_rates_resolve(&params, 2).is_err());
        params.synth_rates = vec![1.5];
        assert!(common_speculative_synth_rates_resolve(&params, 1).is_err());
    }

    #[test]
    fn output_limits() {
        // speculative.cpp:2606-2615
        let l = common_speculative_get_output_limits(512, 4, 3);
        assert_eq!((l.total, l.per_seq), (16, 4));
        let l = common_speculative_get_output_limits(4, 4, 3);
        assert_eq!((l.total, l.per_seq), (4, 4));
        let l = common_speculative_get_output_limits(512, 1, 0);
        assert_eq!((l.total, l.per_seq), (1, 1));
    }
}
