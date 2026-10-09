//! vocab.rs — tokenizer (port of llama.cpp `src/llama-vocab.cpp`). Owner: agent B.
//!
//! Covers all tokenizer types (SPM/BPE/WPM/UGM/RWKV/PLaMo-2/TEST), GGUF
//! loading via the `ggml` crate's GGUF reader (keys WITHOUT arch prefix:
//! `tokenizer.ggml.*`), special-token detection, the special-token fragment
//! partitioner (`tokenizer_st_partition`), tokenization and detokenization
//! (including the `clean_spaces` post-processing passes).
//!
//! Regex-based BPE pre-tokenizers use `crate::unicode::regex_split`, which
//! ports llama.cpp's hand-written splitters plus the collapsed-byte
//! `std::regex` fallback exactly.

use ggml::gguf::{Gguf, GgufType, Value};
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::unicode;

/// `llama_token` (int32); `LLAMA_TOKEN_NULL` == -1.
pub type Token = i32;
pub const TOKEN_NULL: Token = -1;

// `enum llama_token_attr` bits
pub const ATTR_UNKNOWN: u32 = 1 << 0;
pub const ATTR_UNUSED: u32 = 1 << 1;
pub const ATTR_NORMAL: u32 = 1 << 2;
pub const ATTR_CONTROL: u32 = 1 << 3; // SPECIAL?
pub const ATTR_USER_DEFINED: u32 = 1 << 4;
pub const ATTR_BYTE: u32 = 1 << 5;
pub const ATTR_NORMALIZED: u32 = 1 << 6;
pub const ATTR_LSTRIP: u32 = 1 << 7;
pub const ATTR_RSTRIP: u32 = 1 << 8;
pub const ATTR_SINGLE_WORD: u32 = 1 << 9;

/// `enum llama_vocab_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    None,
    Spm,
    Bpe,
    Wpm,
    Ugm,
    Rwkv,
    Plamo2,
    /// `LLAMA_VOCAB_TYPE_PLAMO3` (abeada335, llama-vocab.h) — the PLaMo-2
    /// table tokenizer with the PLaMo-3 pre-segmentation
    Plamo3,
    Test,
}

/// `enum llama_vocab_pre_type` (60 values; see llama-vocab.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    Default,
    Llama3,
    DeepseekLlm,
    DeepseekCoder,
    Falcon,
    Mpt,
    Starcoder,
    Gpt2,
    Refact,
    CommandR,
    Stablelm2,
    Qwen2,
    Olmo,
    Dbrx,
    Smaug,
    Poro,
    Chatglm3,
    Chatglm4,
    Viking,
    Jais,
    Tekken,
    Smollm,
    Codeshell,
    Bloom,
    Gpt3Finnish,
    Exaone,
    Chameleon,
    Minerva,
    Deepseek3Llm,
    Gpt4o,
    Superbpe,
    Trillion,
    Bailingmoe,
    Llama4,
    Pixtral,
    SeedCoder,
    Hunyuan,
    KimiK2,
    HunyuanDense,
    Grok2,
    GraniteDocling,
    MinimaxM2,
    Afmoe,
    SolarOpen,
    Youtu,
    ExaoneMoe,
    Qwen35,
    TinyAya,
    JoyaiLlm,
    Jais2,
    Gemma4,
    SarvamMoe,
    Minicpm5,
    Whitespace,
    GraniteEmbMulti,
    Mellum2,
    Laguna,
    HyV4,
    Spark25,
    Ufakzeka,
    /// `LLAMA_VOCAB_PRE_TYPE_MMBERT` = 60 (a7b94df2c, llama-vocab.h:71)
    Mmbert,
    /// `LLAMA_VOCAB_PRE_TYPE_K2_HORIZON` (462524043, llama-vocab.h)
    K2Horizon,
}

/// `llama_vocab::token_data`.
#[derive(Debug, Clone, Default)]
pub struct TokenData {
    pub text: String,
    pub score: f32,
    pub attr: u32,
}

/// `llama_vocab::normalizer_options` (BertNormalizer).
#[derive(Debug, Clone, Copy)]
pub struct NormalizerOptions {
    pub lowercase: bool,
    pub strip_accents: bool,
}

impl Default for NormalizerOptions {
    fn default() -> Self {
        NormalizerOptions {
            lowercase: true,
            strip_accents: true,
        }
    }
}

// ---------------------------------------------------------------------------
// naive_trie
// ---------------------------------------------------------------------------

/// `struct naive_trie` — byte trie over token texts.
#[derive(Default)]
struct Trie {
    children: HashMap<u8, Box<Trie>>,
    has_value: bool,
    value: Token,
}

impl Trie {
    fn insert(&mut self, key: &[u8], value: Token) {
        if key.is_empty() {
            self.has_value = true;
            self.value = value;
            return;
        }
        self.children
            .entry(key[0])
            .or_default()
            .insert(&key[1..], value);
    }

    /// `get_longest_prefix` — walks while children exist; returns the reached
    /// depth (== length of the longest trie path matching the key prefix).
    fn get_longest_prefix(&self, key: &[u8], offset: usize) -> usize {
        if offset >= key.len() {
            return offset;
        }
        match self.children.get(&key[offset]) {
            Some(child) => child.get_longest_prefix(key, offset + 1),
            None => offset,
        }
    }

    fn traverse(&self, c: u8) -> Option<&Trie> {
        self.children.get(&c).map(|b| b.as_ref())
    }
}

// ---------------------------------------------------------------------------
// tokenizer state (constructed at load time)
// ---------------------------------------------------------------------------

/// `llm_tokenizer_bpe` — pre-tokenizer regexes + GPT-2 byte-encoding flag.
struct BpeTokenizer {
    regex_exprs: Vec<String>,
    byte_encode: bool,
}

/// `llm_tokenizer_ugm` — Viterbi unigram tokenizer state.
struct UgmTokenizer {
    /// escaped space symbol U+2581
    escaped_space: &'static [u8],
    prefix_replacements: Vec<u8>,
    xcda_array: Vec<u32>,
    user_defined_token_matcher: Trie,
    min_score: f32,
    #[allow(dead_code)]
    max_score: f32,
    unknown_token_score: f32,
    token_matcher: Trie,
}

/// `llm_tokenizer_rwkv` — trie over unescaped token texts.
struct RwkvTokenizer {
    token_matcher: Trie,
}

/// `llm_tokenizer_plamo2` — Aho-Corasick-ish table tokenizer.
struct Plamo2Tokenizer {
    tokens: Vec<String>,
    bytes: [Token; 256],
    to_suffix_id: HashMap<i64, i32>,
    /// rows: [piece_length, token_id, score, piece_id]
    table: Vec<[i32; 4]>,
    /// `pre_segment_` (abeada335, llama-vocab.cpp:1745) — PLaMo-3's
    /// pre-segmentation pass
    pre_segment: bool,
}

// table column constants (llm_tokenizer_plamo2)
const TABLE_PIECE_LENGTH: usize = 0;
const TABLE_TOKEN_ID: usize = 1;
const TABLE_SCORE: usize = 2;
const TABLE_PIECE_ID: usize = 3;
const PATH_TOKEN_LENGTH: usize = 0;
const PATH_TOKEN_ID: usize = 1;
const PATH_NUM_TOKENS: usize = 2;
const PLAMO2_INVALID_SCORE: i32 = -20000000;
const PLAMO2_UNKNOWN_SCORE: i32 = -10000000;

enum Tokenizer {
    None,
    Spm,
    Bpe(Box<BpeTokenizer>),
    Wpm,
    Ugm(Box<UgmTokenizer>),
    Rwkv(Box<RwkvTokenizer>),
    Plamo2(Box<Plamo2Tokenizer>),
    Test,
}

// ---------------------------------------------------------------------------
// llama_vocab
// ---------------------------------------------------------------------------

/// `struct llama_vocab` (+ its impl).
pub struct Vocab {
    pub n_token_types: u32,

    pub tokenizer_model: String,
    pub tokenizer_pre: String,

    pub ty: VocabType,
    pub pre_type: PreType,

    /// used for optimizing longest token search
    pub max_token_len: usize,

    // default LLaMA special tokens
    pub special_bos_id: Token,
    pub special_eos_id: Token,
    pub special_eot_id: Token,
    pub special_eom_id: Token,
    pub special_unk_id: Token,
    pub special_sep_id: Token,
    pub special_pad_id: Token,
    pub special_mask_id: Token,

    pub linefeed_id: Token,

    // fim tokens
    pub special_fim_pre_id: Token,
    pub special_fim_suf_id: Token,
    pub special_fim_mid_id: Token,
    pub special_fim_pad_id: Token,
    pub special_fim_rep_id: Token, // repo
    pub special_fim_sep_id: Token, // file separator

    // tokenizer flags
    pub add_space_prefix: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub ignore_merges: bool,
    pub clean_spaces: bool, // clean_up_tokenization_spaces
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,

    // BertNormalizer options
    pub normalizer_opts: NormalizerOptions,

    /// keyed by raw bytes (token texts need not be valid UTF-8 subsets)
    pub token_to_id: HashMap<Vec<u8>, Token>,
    pub id_to_token: Vec<TokenData>,

    cache_special_tokens: Vec<Token>,
    /// llama_token_to_piece(special = true) cache
    cache_token_to_piece: Vec<Vec<u8>>,

    bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), i32>,

    /// set of all tokens that cause "end of generation"
    pub special_eog_ids: BTreeSet<Token>,

    pub suppress_tokens: Vec<Token>,

    tokenizer: Tokenizer,

    pub precompiled_charsmap: Vec<u8>,
}

/// reserved suffix (U+E000) for hybriddna k-mers; erased from id_to_token at load
const DNA_KMER_MARKER: &[u8] = b"\xee\x80\x80";

// ---------------------------------------------------------------------------
// loading (impl::load)
// ---------------------------------------------------------------------------

// KV keys (no arch prefix)
const KV_TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
const KV_TOKENIZER_PRE: &str = "tokenizer.ggml.pre";
const KV_TOKENIZER_LIST: &str = "tokenizer.ggml.tokens";
const KV_TOKENIZER_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
const KV_TOKENIZER_TOKEN_TYPE_COUNT: &str = "tokenizer.ggml.token_type_count";
const KV_TOKENIZER_SCORES: &str = "tokenizer.ggml.scores";
const KV_TOKENIZER_MERGES: &str = "tokenizer.ggml.merges";
const KV_TOKENIZER_BOS_ID: &str = "tokenizer.ggml.bos_token_id";
const KV_TOKENIZER_EOS_ID: &str = "tokenizer.ggml.eos_token_id";
const KV_TOKENIZER_EOT_ID: &str = "tokenizer.ggml.eot_token_id";
const KV_TOKENIZER_EOM_ID: &str = "tokenizer.ggml.eom_token_id";
const KV_TOKENIZER_UNK_ID: &str = "tokenizer.ggml.unknown_token_id";
const KV_TOKENIZER_SEP_ID: &str = "tokenizer.ggml.seperator_token_id";
const KV_TOKENIZER_PAD_ID: &str = "tokenizer.ggml.padding_token_id";
const KV_TOKENIZER_MASK_ID: &str = "tokenizer.ggml.mask_token_id";
const KV_TOKENIZER_ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
const KV_TOKENIZER_ADD_EOS: &str = "tokenizer.ggml.add_eos_token";
const KV_TOKENIZER_ADD_SEP: &str = "tokenizer.ggml.add_sep_token";
const KV_TOKENIZER_ADD_PREFIX: &str = "tokenizer.ggml.add_space_prefix";
const KV_TOKENIZER_REMOVE_EXTRA_WS: &str = "tokenizer.ggml.remove_extra_whitespaces";
const KV_TOKENIZER_PRECOMPILED_CHARSMAP: &str = "tokenizer.ggml.precompiled_charsmap";
const KV_TOKENIZER_NORMALIZER_LOWERCASE: &str = "tokenizer.ggml.normalizer.lowercase";
const KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS: &str = "tokenizer.ggml.normalizer.strip_accents";
const KV_TOKENIZER_FIM_PRE_ID: &str = "tokenizer.ggml.fim_pre_token_id";
const KV_TOKENIZER_FIM_SUF_ID: &str = "tokenizer.ggml.fim_suffix_token_id";
const KV_TOKENIZER_FIM_MID_ID: &str = "tokenizer.ggml.fim_mid_token_id";
const KV_TOKENIZER_FIM_PAD_ID: &str = "tokenizer.ggml.fim_pad_token_id";
const KV_TOKENIZER_FIM_REP_ID: &str = "tokenizer.ggml.fim_rep_token_id";
const KV_TOKENIZER_FIM_SEP_ID: &str = "tokenizer.ggml.fim_sep_token_id";
const KV_TOKENIZER_PREFIX_ID: &str = "tokenizer.ggml.prefix_token_id";
const KV_TOKENIZER_SUFFIX_ID: &str = "tokenizer.ggml.suffix_token_id";
const KV_TOKENIZER_MIDDLE_ID: &str = "tokenizer.ggml.middle_token_id";
const KV_TOKENIZER_SUPPRESS_TOKENS: &str = "tokenizer.ggml.suppress_tokens";
const KV_GENERAL_NAME: &str = "general.name";
const KV_GENERAL_ARCHITECTURE: &str = "general.architecture";

fn get_arr_strings(gguf: &Gguf, key: &str) -> Result<Vec<String>, String> {
    match gguf.find_key(key) {
        Some(Value::Array(GgufType::String, items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("invalid string in {key}"))
            })
            .collect(),
        Some(v) => Err(format!(
            "invalid gguf type for {key}: expected array of string, got {}",
            v.type_().name()
        )),
        None => Err(format!("key not found: {key}")),
    }
}

impl Vocab {
    /// `llama_vocab::impl::load` — load from a GGUF metadata map.
    pub fn load(gguf: &Gguf) -> Result<Vocab, String> {
        let mut v = Vocab::default_partial();

        // determine vocab type
        {
            v.tokenizer_model = gguf
                .get_str(KV_TOKENIZER_MODEL)
                .ok_or_else(|| format!("cannot find {KV_TOKENIZER_MODEL} in model file"))?
                .to_string();
            v.tokenizer_pre = gguf.get_str(KV_TOKENIZER_PRE).unwrap_or("").to_string();
            v.n_token_types = gguf.get_u32(KV_TOKENIZER_TOKEN_TYPE_COUNT).unwrap_or(0);

            if v.tokenizer_model == "no_vocab" || v.tokenizer_model == "none" {
                v.ty = VocabType::None;
                v.special_bos_id = TOKEN_NULL;
                v.special_eos_id = TOKEN_NULL;
                v.special_unk_id = TOKEN_NULL;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
                v.special_mask_id = TOKEN_NULL;
                v.linefeed_id = TOKEN_NULL;

                // read vocab size from metadata ({arch}.vocab_size)
                let arch = gguf.get_str(KV_GENERAL_ARCHITECTURE).unwrap_or("");
                if let Some(n_tokens) = gguf.get_u32(&format!("{arch}.vocab_size")) {
                    v.id_to_token
                        .resize(n_tokens as usize, TokenData::default());
                }
                v.tokenizer = Tokenizer::None;
                return Ok(v);
            }

            if v.tokenizer_model == "llama" {
                v.ty = VocabType::Spm;
                v.special_bos_id = 1;
                v.special_eos_id = 2;
                v.special_unk_id = 0;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
                v.special_mask_id = TOKEN_NULL;
            } else if v.tokenizer_model == "bert" {
                v.ty = VocabType::Wpm;
                v.special_bos_id = 101;
                v.special_eos_id = TOKEN_NULL;
                v.special_unk_id = 100;
                v.special_sep_id = 102;
                v.special_pad_id = 0;
                v.special_mask_id = 103;
                v.add_sep = true;
            } else if v.tokenizer_model == "gpt2"
                || v.tokenizer_model == "hybriddna"
                || v.tokenizer_model == "whitespace"
            {
                v.ty = VocabType::Bpe;

                // read bpe merges and populate bpe ranks
                let is_kimi_k2 = v.tokenizer_pre == "kimi-k2";
                match gguf.find_key(KV_TOKENIZER_MERGES) {
                    None => {
                        if !is_kimi_k2 {
                            return Err("cannot find tokenizer merges in model file".to_string());
                        }
                        // Kimi-K2 doesn't need merges, skip
                    }
                    Some(Value::Array(GgufType::String, items)) => {
                        for (i, item) in items.iter().enumerate() {
                            let word = item.as_str().ok_or("invalid merge entry")?;
                            // find first ' ' at position >= 1
                            let pos = word.as_bytes()[1.min(word.len())..]
                                .iter()
                                .position(|&b| b == b' ')
                                .map(|p| p + 1.min(word.len()));
                            let (first, second) = match pos {
                                Some(p) => (&word.as_bytes()[..p], &word.as_bytes()[p + 1..]),
                                None => (&word.as_bytes()[..0], &word.as_bytes()[..0]),
                            };
                            v.bpe_ranks
                                .entry((first.to_vec(), second.to_vec()))
                                .or_insert(i as i32);
                        }
                    }
                    Some(val) => {
                        return Err(format!(
                            "invalid gguf type for {KV_TOKENIZER_MERGES}: {}",
                            val.type_().name()
                        ));
                    }
                }

                v.special_bos_id = 11;
                v.special_eos_id = 11;
                v.special_unk_id = TOKEN_NULL;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
                v.special_mask_id = TOKEN_NULL;
            } else if v.tokenizer_model == "t5" {
                v.ty = VocabType::Ugm;
                v.special_bos_id = TOKEN_NULL;
                v.special_eos_id = 1;
                v.special_unk_id = 2;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = 0;
                v.special_mask_id = TOKEN_NULL;

                match gguf.find_key(KV_TOKENIZER_PRECOMPILED_CHARSMAP) {
                    Some(Value::Array(t @ (GgufType::Int8 | GgufType::Uint8), items)) => {
                        let mut charsmap = Vec::with_capacity(items.len());
                        for item in items {
                            let b = match item {
                                Value::I8(x) => *x as u8,
                                Value::U8(x) => *x,
                                _ => return Err("invalid charsmap element".to_string()),
                            };
                            charsmap.push(b);
                        }
                        let _ = t;
                        if charsmap.len() < 4 {
                            return Err(
                                "precompiled_charsmap too small for xcda_blob_size header!"
                                    .to_string(),
                            );
                        }
                        let xcda_blob_size = u32::from_le_bytes([
                            charsmap[0],
                            charsmap[1],
                            charsmap[2],
                            charsmap[3],
                        ]) as usize;
                        if xcda_blob_size + 4 >= charsmap.len() {
                            return Err(
                                "Index out of array bounds in precompiled charsmap!".to_string()
                            );
                        }
                        v.precompiled_charsmap = charsmap;
                    }
                    Some(val) => {
                        return Err(format!(
                            "invalid gguf type for {KV_TOKENIZER_PRECOMPILED_CHARSMAP}: {}",
                            val.type_().name()
                        ));
                    }
                    None => {}
                }
            } else if v.tokenizer_model == "rwkv" {
                v.ty = VocabType::Rwkv;
                v.special_bos_id = TOKEN_NULL;
                v.special_eos_id = TOKEN_NULL;
                v.special_unk_id = TOKEN_NULL;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
            } else if v.tokenizer_model == "test" {
                v.ty = VocabType::Test;
                v.special_bos_id = TOKEN_NULL;
                v.special_eos_id = TOKEN_NULL;
                v.special_unk_id = TOKEN_NULL;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
                v.special_mask_id = TOKEN_NULL;
            } else if v.tokenizer_model == "plamo2" || v.tokenizer_model == "plamo3" {
                // abeada335 (llama-vocab.cpp:2213-2214): "plamo3" selects the
                // pre-segmenting twin of the PLaMo-2 tokenizer
                v.ty = if v.tokenizer_model == "plamo2" {
                    VocabType::Plamo2
                } else {
                    VocabType::Plamo3
                };
                // PLaMo default special tokens (overridden by model config)
                v.special_bos_id = 1; // <|plamo:bos|>
                v.special_eos_id = 2; // <|plamo:eos|>
                v.special_unk_id = 0; // <|plamo:unk|>
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = 3; // <|plamo:pad|>
                v.special_mask_id = TOKEN_NULL;
            } else if v.tokenizer_model == "gemma4" {
                v.ty = VocabType::Bpe;

                let items = get_arr_strings(gguf, KV_TOKENIZER_MERGES)?;
                for (i, word) in items.iter().enumerate() {
                    let bytes = word.as_bytes();
                    let pos = bytes[1.min(bytes.len())..]
                        .iter()
                        .position(|&b| b == b' ')
                        .map(|p| p + 1.min(bytes.len()));
                    let (first, second) = match pos {
                        Some(p) => (&bytes[..p], &bytes[p + 1..]),
                        None => (&bytes[..0], &bytes[..0]),
                    };
                    v.bpe_ranks
                        .entry((first.to_vec(), second.to_vec()))
                        .or_insert(i as i32);
                }

                v.special_bos_id = TOKEN_NULL;
                v.special_eos_id = TOKEN_NULL;
                v.special_unk_id = TOKEN_NULL;
                v.special_sep_id = TOKEN_NULL;
                v.special_pad_id = TOKEN_NULL;
                v.special_mask_id = TOKEN_NULL;

                v.tokenizer_pre = "gemma4".to_string();
            } else {
                return Err(format!("unknown tokenizer: '{}'", v.tokenizer_model));
            }

            // for now, only BPE models have pre-tokenizers
            if v.ty == VocabType::Bpe {
                v.add_space_prefix = false;
                v.escape_whitespaces = false;
                v.clean_spaces = true;
                map_pre_tokenizer(&mut v)?;
            } else if v.ty == VocabType::Spm {
                v.pre_type = PreType::Default;
                v.add_space_prefix = true;
                v.clean_spaces = false;
                v.add_bos = true;
                v.add_eos = false;
            } else if v.ty == VocabType::Wpm {
                v.pre_type = PreType::Default;
                v.add_space_prefix = false;
                v.clean_spaces = true;
                v.add_bos = true;
                v.add_eos = false;
                v.add_sep = true;
            } else if v.ty == VocabType::Ugm {
                v.pre_type = PreType::Default;
                v.add_bos = false;
                v.add_eos = true;
            } else if v.ty == VocabType::Rwkv {
                v.pre_type = PreType::Default;
                v.add_space_prefix = false;
                v.clean_spaces = false;
                v.add_bos = false;
                v.add_eos = false;
            } else {
                v.pre_type = PreType::Default;
            }

            if let Some(b) = gguf.get_bool(KV_TOKENIZER_ADD_PREFIX) {
                v.add_space_prefix = b;
            }
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_REMOVE_EXTRA_WS) {
                v.remove_extra_whitespaces = b;
            }
        }

        let tokens = get_arr_strings(gguf, KV_TOKENIZER_LIST)
            .map_err(|_| "cannot find tokenizer vocab in model file".to_string())?;

        let n_tokens = tokens.len();

        // scores: f32 or i32 array
        let mut scores_f32: Option<Vec<f32>> = None;
        let mut scores_i32: Option<Vec<i32>> = None;
        match gguf.find_key(KV_TOKENIZER_SCORES) {
            Some(Value::Array(GgufType::Float32, items)) => {
                let s: Option<Vec<f32>> = items.iter().map(|x| x.as_f32()).collect();
                scores_f32 = s;
            }
            Some(Value::Array(GgufType::Int32, items)) => {
                let s: Option<Vec<i32>> = items
                    .iter()
                    .map(|x| match x {
                        Value::I32(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                scores_i32 = s;
            }
            Some(val) => {
                return Err(format!(
                    "invalid gguf type for {KV_TOKENIZER_SCORES}: {}",
                    val.type_().name()
                ));
            }
            None => {}
        }
        if let Some(s) = &scores_f32 {
            if s.len() < n_tokens {
                return Err(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                ));
            }
        }
        if let Some(s) = &scores_i32 {
            if s.len() < n_tokens {
                return Err(format!(
                    "Index out of array bounds for scores ({} < {})",
                    s.len(),
                    n_tokens
                ));
            }
        }

        // token types: i32 array
        let mut toktypes: Option<Vec<i32>> = None;
        match gguf.find_key(KV_TOKENIZER_TOKEN_TYPE) {
            Some(Value::Array(GgufType::Int32, items)) => {
                let t: Option<Vec<i32>> = items
                    .iter()
                    .map(|x| match x {
                        Value::I32(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                toktypes = t;
            }
            Some(val) => {
                return Err(format!(
                    "invalid gguf type for {KV_TOKENIZER_TOKEN_TYPE}: {}",
                    val.type_().name()
                ));
            }
            None => {}
        }
        if let Some(t) = &toktypes {
            if t.len() < n_tokens {
                return Err(format!(
                    "Index out of array bounds for toktypes ({} < {})",
                    t.len(),
                    n_tokens
                ));
            }
        }

        v.id_to_token = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let mut word = tokens[i].clone();
            if word.is_empty() {
                word = format!("[EMPTY_{i}]");
            }

            v.token_to_id.insert(word.as_bytes().to_vec(), i as Token);
            v.max_token_len = v.max_token_len.max(word.len());

            let score = if let Some(s) = &scores_f32 {
                s[i]
            } else if let Some(s) = &scores_i32 {
                s[i] as f32
            } else {
                0.0
            };

            let mut attr = ATTR_NORMAL;
            if let Some(t) = &toktypes {
                attr = match t[i] {
                    2 => ATTR_UNKNOWN,      // LLAMA_TOKEN_TYPE_UNKNOWN
                    5 => ATTR_UNUSED,       // LLAMA_TOKEN_TYPE_UNUSED
                    1 => ATTR_NORMAL,       // LLAMA_TOKEN_TYPE_NORMAL
                    3 => ATTR_CONTROL,      // LLAMA_TOKEN_TYPE_CONTROL
                    4 => ATTR_USER_DEFINED, // LLAMA_TOKEN_TYPE_USER_DEFINED
                    6 => ATTR_BYTE,         // LLAMA_TOKEN_TYPE_BYTE
                    _ => 0,                 // LLAMA_TOKEN_TYPE_UNDEFINED
                };
            }

            v.id_to_token.push(TokenData {
                text: word,
                score,
                attr,
            });
        }
        debug_assert_eq!(v.id_to_token.len(), v.token_to_id.len());

        // hybriddna: erase the k-mer marker suffix from id_to_token texts
        if v.tokenizer_model == "hybriddna" {
            if let Some(&idx) = v.token_to_id.get(b"<oov>".as_slice()) {
                for td in v.id_to_token.iter_mut().skip(idx as usize + 1) {
                    if td.text.len() > DNA_KMER_MARKER.len()
                        && &td.text.as_bytes()[td.text.len() - DNA_KMER_MARKER.len()..]
                            == DNA_KMER_MARKER
                    {
                        let new_len = td.text.len() - DNA_KMER_MARKER.len();
                        td.text.truncate(new_len);
                    }
                }
            }
        }

        v.init_tokenizer();

        // determine the newline token
        if v.ty == VocabType::Spm {
            match v.byte_to_token(b'\n') {
                Ok(t) => v.linefeed_id = t,
                Err(_) => v.linefeed_id = v.special_pad_id,
            }
        } else if v.ty == VocabType::Wpm {
            v.linefeed_id = v.special_pad_id;
        } else if v.ty == VocabType::Rwkv {
            let ids = v.tokenize_internal(b"\n", false, false);
            v.linefeed_id = *ids.first().expect("model vocab missing newline token");
        } else {
            let ids = v.tokenize_internal(b"\n", false, false);
            v.linefeed_id = match ids.first() {
                Some(&id) => id,
                None => v.special_pad_id,
            };
        }

        // special tokens
        {
            let n = v.id_to_token.len();
            // (key, target, deprecated?)
            let special_token_keys: &[(&str, fn(&mut Vocab) -> &mut Token)] = &[
                (KV_TOKENIZER_BOS_ID, |v: &mut Vocab| &mut v.special_bos_id),
                (KV_TOKENIZER_EOS_ID, |v: &mut Vocab| &mut v.special_eos_id),
                (KV_TOKENIZER_EOT_ID, |v: &mut Vocab| &mut v.special_eot_id),
                (KV_TOKENIZER_EOM_ID, |v: &mut Vocab| &mut v.special_eom_id),
                (KV_TOKENIZER_UNK_ID, |v: &mut Vocab| &mut v.special_unk_id),
                (KV_TOKENIZER_SEP_ID, |v: &mut Vocab| &mut v.special_sep_id),
                (KV_TOKENIZER_PAD_ID, |v: &mut Vocab| &mut v.special_pad_id),
                (KV_TOKENIZER_MASK_ID, |v: &mut Vocab| &mut v.special_mask_id),
                (KV_TOKENIZER_FIM_PRE_ID, |v: &mut Vocab| {
                    &mut v.special_fim_pre_id
                }),
                (KV_TOKENIZER_FIM_SUF_ID, |v: &mut Vocab| {
                    &mut v.special_fim_suf_id
                }),
                (KV_TOKENIZER_FIM_MID_ID, |v: &mut Vocab| {
                    &mut v.special_fim_mid_id
                }),
                (KV_TOKENIZER_FIM_PAD_ID, |v: &mut Vocab| {
                    &mut v.special_fim_pad_id
                }),
                (KV_TOKENIZER_FIM_REP_ID, |v: &mut Vocab| {
                    &mut v.special_fim_rep_id
                }),
                (KV_TOKENIZER_FIM_SEP_ID, |v: &mut Vocab| {
                    &mut v.special_fim_sep_id
                }),
                // deprecated
                (KV_TOKENIZER_PREFIX_ID, |v: &mut Vocab| {
                    &mut v.special_fim_pre_id
                }),
                (KV_TOKENIZER_SUFFIX_ID, |v: &mut Vocab| {
                    &mut v.special_fim_suf_id
                }),
                (KV_TOKENIZER_MIDDLE_ID, |v: &mut Vocab| {
                    &mut v.special_fim_mid_id
                }),
            ];

            for (key, getter) in special_token_keys {
                if *getter(&mut v) >= 0 && (*getter(&mut v) as usize) >= n {
                    *getter(&mut v) = TOKEN_NULL; // out of vocab range, disable
                }
                let Some(new_id) = gguf.get_u32(key) else {
                    continue;
                };
                if new_id as usize >= n {
                    // bad special token: keep default
                } else {
                    *getter(&mut v) = new_id as Token;
                }
            }

            // Handle add_bos, add_eos and add_sep
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_ADD_BOS) {
                v.add_bos = b;
            }
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_ADD_EOS) {
                v.add_eos = b;
            }
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_ADD_SEP) {
                v.add_sep = b;
            }

            // workaround for Gemma 4
            if v.pre_type == PreType::Gemma4 && !v.add_bos {
                v.add_bos = true;
            }

            // BertNormalizer options
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_NORMALIZER_LOWERCASE) {
                v.normalizer_opts.lowercase = b;
            }
            v.normalizer_opts.strip_accents = v.normalizer_opts.lowercase;
            if let Some(b) = gguf.get_bool(KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS) {
                v.normalizer_opts.strip_accents = b;
            }

            // suppress tokens
            if let Some(Value::Array(GgufType::Int32, items)) =
                gguf.find_key(KV_TOKENIZER_SUPPRESS_TOKENS)
            {
                for item in items {
                    if let Value::I32(id) = item {
                        if *id >= 0 && (*id as usize) < v.id_to_token.len() {
                            v.suppress_tokens.push(*id);
                        }
                    }
                }
            }

            // auto-detect special tokens by text (iterating ids ascending —
            // the C++ iterates an unordered_map, so "first match" order is a
            // benign divergence affecting only eot/eom/fim id selection)
            for id in 0..v.id_to_token.len() as Token {
                let text = v.id_to_token[id as usize].text.clone();
                let text = text.as_str();

                // find EOT token
                if v.special_eot_id == TOKEN_NULL
                    && (text == "<|eot_id|>"
                        || text == "<|im_end|>"
                        || text == "<|end|>"
                        || text == "<end_of_turn>"
                        || text == "<|endoftext|>"
                        || text == "<|end_of_text|>"
                        || text == "<EOT>"
                        || text == "_<EOT>"
                        || text == "[EOT]"
                        || text == "<｜end▁of▁sentence｜>"
                        || text == "<end_of_utterance>")
                {
                    v.special_eot_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find EOM token
                if v.special_eom_id == TOKEN_NULL && text == "<|eom_id|>" {
                    v.special_eom_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_PRE token
                if v.special_fim_pre_id == TOKEN_NULL
                    && (text == "<|fim_prefix|>"
                        || text == "<fim-prefix>"
                        || text == "<fim_prefix>"
                        || text == "<｜fim▁begin｜>"
                        || text == "<PRE>"
                        || text == "▁<PRE>"
                        || text == "<|code_prefix|>"
                        || text == "<|prefix|>"
                        // PLaMo-3 (42b021b4d, llama-vocab.cpp:2863)
                        || text == "<|plamo:fim_prefix|>")
                {
                    v.special_fim_pre_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_SUF token
                if v.special_fim_suf_id == TOKEN_NULL
                    && (text == "<|fim_suffix|>"
                        || text == "<fim-suffix>"
                        || text == "<fim_suffix>"
                        || text == "<｜fim▁hole｜>"
                        || text == "<SUF>"
                        || text == "▁<SUF>"
                        || text == "<|code_suffix|>"
                        || text == "<|suffix|>"
                        // PLaMo-3 (42b021b4d, llama-vocab.cpp:2885)
                        || text == "<|plamo:fim_suffix|>")
                {
                    v.special_fim_suf_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_MID token
                if v.special_fim_mid_id == TOKEN_NULL
                    && (text == "<|fim_middle|>"
                        || text == "<fim-middle>"
                        || text == "<fim_middle>"
                        || text == "<｜fim▁end｜>"
                        || text == "<MID>"
                        || text == "▁<MID>"
                        || text == "<|code_middle|>"
                        || text == "<|middle|>"
                        // PLaMo-3 (42b021b4d, llama-vocab.cpp:2907)
                        || text == "<|plamo:fim_middle|>")
                {
                    v.special_fim_mid_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_PAD token
                if v.special_fim_pad_id == TOKEN_NULL
                    && (text == "<|fim_pad|>"
                        || text == "<fim-pad>"
                        || text == "<fim_pad>"
                        || text == "<PAD>"
                        || text == "[PAD]")
                {
                    v.special_fim_pad_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_REP token
                if v.special_fim_rep_id == TOKEN_NULL
                    && (text == "<|fim_repo|>"
                        || text == "<|repo_name|>"
                        || text == "<fim-repo>"
                        || text == "<REPO>"
                        || text == "<reponame>")
                {
                    v.special_fim_rep_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }

                // find FIM_SEP token
                // PLaMo-2/3 (42b021b4d, llama-vocab.cpp:2958)
                if v.special_fim_sep_id == TOKEN_NULL
                    && (text == "<|file_sep|>" || text == "<|plamo:file_separator|>")
                {
                    v.special_fim_sep_id = id;
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }
            }

            // auto-detect unused tokens
            for id in 0..v.id_to_token.len() {
                let td = &v.id_to_token[id];
                if td.attr & ATTR_CONTROL == 0 {
                    continue;
                }
                if td.attr & ATTR_UNUSED == 0 && td.text.contains("unused") {
                    v.id_to_token[id].attr |= ATTR_UNUSED;
                }
            }

            // maintain the list of tokens that cause end-of-generation
            v.special_eog_ids.clear();

            if v.special_fim_pad_id != TOKEN_NULL {
                v.special_eog_ids.insert(v.special_fim_pad_id);
            }
            if v.special_fim_rep_id != TOKEN_NULL {
                v.special_eog_ids.insert(v.special_fim_rep_id);
            }
            if v.special_fim_sep_id != TOKEN_NULL {
                v.special_eog_ids.insert(v.special_fim_sep_id);
            }

            for id in 0..v.id_to_token.len() as Token {
                let text = v.id_to_token[id as usize].text.as_str();
                if text == "<|eot_id|>"
                    || text == "<|im_end|>"
                    || text == "<|end|>"
                    || text == "<|return|>"
                    || text == "<|call|>"
                    || text == "<|flush|>"
                    || text == "<|calls|>"
                    || text == "<end_of_turn>"
                    || text == "<|endoftext|>"
                    || text == "</s>"
                    || text == "<|eom_id|>"
                    || text == "<EOT>"
                    || text == "_<EOT>"
                    || text == "[EOT]"
                    || text == "[EOS]"
                    || text == "<|end_of_text|>"
                    || text == "<end_of_utterance>"
                    || text == "<eos>"
                    || text == "<turn|>"
                    || text == "<|tool_response>"
                    || text == "<｜end▁of▁sentence｜>"
                    || text == "[e~["
                    // k2-horizon (462524043, llama-vocab.cpp:3039)
                    || text == "<|ifm|im_end|>"
                {
                    v.special_eog_ids.insert(id);
                    let attr = &mut v.id_to_token[id as usize].attr;
                    if *attr & ATTR_CONTROL == 0 {
                        *attr |= ATTR_CONTROL;
                    }
                }
            }

            // quick hack for gpt-oss: always render these tokens
            for id in 0..v.id_to_token.len() {
                let text = v.id_to_token[id].text.as_str();
                if text == "<|channel|>"
                    || text == "<|message|>"
                    || text == "<|start|>"
                    || text == "<|constrain|>"
                {
                    v.id_to_token[id].attr = ATTR_USER_DEFINED;
                }
            }

            // sanity checks
            if v.special_eos_id != TOKEN_NULL && !v.special_eog_ids.contains(&v.special_eos_id) {
                v.special_eog_ids.insert(v.special_eos_id);
            }
            if v.special_eot_id != TOKEN_NULL && !v.special_eog_ids.contains(&v.special_eot_id) {
                v.special_eog_ids.insert(v.special_eot_id);
            }
            if v.special_eom_id != TOKEN_NULL && !v.special_eog_ids.contains(&v.special_eom_id) {
                v.special_eog_ids.insert(v.special_eom_id);
            }

            // workaround for o200k_harmony / solar-open: remove "<|end|>" from EOG
            {
                let mut has_return = false;
                let mut has_call = false;
                let mut has_end = false;
                let mut has_flush = false;
                let mut end_id = TOKEN_NULL;

                for &tid in &v.special_eog_ids {
                    if tid < 0 || tid as usize >= v.id_to_token.len() {
                        continue;
                    }
                    let text = v.id_to_token[tid as usize].text.as_str();
                    if text == "<|return|>" {
                        has_return = true;
                    } else if text == "<|call|>" || text == "<|calls|>" {
                        has_call = true;
                    } else if text == "<|flush|>" {
                        has_flush = true;
                    } else if text == "<|end|>" {
                        has_end = true;
                        end_id = tid;
                    }
                }

                if (has_return && has_call && has_end) || (has_call && has_flush && has_end) {
                    v.special_eog_ids.remove(&end_id);
                    v.id_to_token[end_id as usize].attr = ATTR_USER_DEFINED;
                }
            }

            // gemma4 and plamo have a normal </s> token, unlike paddleocr
            // (llama-vocab.cpp:3004-3032, a3f84faf4 + db33d3cb8)
            {
                let mut has_normal_s_marker = false;
                let mut has_s = false;
                let mut s_id = TOKEN_NULL;

                for &tid in &v.special_eog_ids {
                    if tid < 0 || tid as usize >= v.id_to_token.len() {
                        continue;
                    }
                    let text = v.id_to_token[tid as usize].text.as_str();
                    if text == "<|tool_response>" || text == "<|plamo:eos|>" {
                        has_normal_s_marker = true;
                    } else if text == "</s>" {
                        has_s = true;
                        s_id = tid;
                    }
                }

                if has_normal_s_marker && has_s {
                    v.special_eog_ids.remove(&s_id);
                    v.id_to_token[s_id as usize].attr = ATTR_NORMAL;
                }
            }
        }

        // build special tokens cache (sorted by text length desc)
        {
            let mut cache: Vec<Token> = (0..n_tokens as Token)
                .filter(|&id| {
                    v.id_to_token[id as usize].attr
                        & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN)
                        != 0
                })
                .collect();
            // NOTE: std::sort is unstable; ties keep id order here (stable)
            cache.sort_by(|&a, &b| {
                v.id_to_token[b as usize]
                    .text
                    .len()
                    .cmp(&v.id_to_token[a as usize].text.len())
            });
            v.cache_special_tokens = cache;
        }

        // build token to piece cache (special = true)
        {
            v.cache_token_to_piece = (0..n_tokens as Token)
                .map(|id| v.token_to_piece_uncached(id, true))
                .collect();
        }

        // per-token attribute adjustments by model/tokenizer/arch name
        {
            let model_name_raw = gguf.get_str(KV_GENERAL_NAME).unwrap_or("").to_string();
            let tokenizer_pre = gguf.get_str(KV_TOKENIZER_PRE).unwrap_or("");
            let general_arch = gguf.get_str(KV_GENERAL_ARCHITECTURE).unwrap_or("");
            let model_name = model_name_raw.to_lowercase();

            let contains_any = |s: &str, subs: &[&str]| subs.iter().any(|sub| s.contains(sub));

            let set_tokenid_attr = |v: &mut Vocab, id: Token, attr: u32, value: bool| {
                let cur = v.id_to_token[id as usize].attr;
                v.id_to_token[id as usize].attr = if value { cur | attr } else { cur & !attr };
            };
            let set_token_attr = |v: &mut Vocab, token: &str, attr: u32, value: bool| {
                let id = *v
                    .token_to_id
                    .get(token.as_bytes())
                    .expect("token missing in vocab");
                set_tokenid_attr(v, id, attr, value);
            };

            if contains_any(
                tokenizer_pre,
                &["jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert"],
            ) || contains_any(general_arch, &["nomic-bert-moe", "jina-bert-v3"])
            {
                if !v.token_to_id.contains_key(b"<mask>".as_slice()) {
                    eprintln!("warn: Mask token is missing in vocab, please reconvert model!");
                } else {
                    set_token_attr(&mut v, "<mask>", ATTR_LSTRIP, true);
                }
            } else if contains_any(&model_name, &["phi-3", "phi3"]) {
                let specials = v.cache_special_tokens.clone();
                for id in specials {
                    set_tokenid_attr(&mut v, id, ATTR_RSTRIP, true);
                }
                set_token_attr(&mut v, "</s>", ATTR_RSTRIP, true);
                for token in ["<unk>", "<s>", "<|endoftext|>"] {
                    set_token_attr(&mut v, token, ATTR_RSTRIP, false);
                }
            } else if contains_any(&model_name, &["modern-bert"]) {
                if !v.token_to_id.contains_key(b"[MASK]".as_slice()) {
                    eprintln!("warn: Mask token missing in vocab!");
                } else {
                    set_token_attr(&mut v, "[MASK]", ATTR_LSTRIP, true);
                }
            }
        }

        Ok(v)
    }

    /// an empty vocab for tests: every id maps to "not EOG" (there is no
    /// C-side counterpart — `common_sampler_sample_and_accept_n` takes the
    /// vocab from the context, and tests drive it without one)
    pub fn empty() -> Vocab {
        Vocab::default_partial()
    }

    /// defaults matching `struct llama_vocab::impl` member initializers
    fn default_partial() -> Vocab {
        Vocab {
            n_token_types: 0,
            tokenizer_model: String::new(),
            tokenizer_pre: String::new(),
            ty: VocabType::Spm,
            pre_type: PreType::Default,
            max_token_len: 0,
            special_bos_id: 1,
            special_eos_id: 2,
            special_eot_id: TOKEN_NULL,
            special_eom_id: TOKEN_NULL,
            special_unk_id: 0,
            special_sep_id: TOKEN_NULL,
            special_pad_id: TOKEN_NULL,
            special_mask_id: TOKEN_NULL,
            linefeed_id: 13,
            special_fim_pre_id: TOKEN_NULL,
            special_fim_suf_id: TOKEN_NULL,
            special_fim_mid_id: TOKEN_NULL,
            special_fim_pad_id: TOKEN_NULL,
            special_fim_rep_id: TOKEN_NULL,
            special_fim_sep_id: TOKEN_NULL,
            add_space_prefix: false,
            add_bos: false,
            add_eos: false,
            add_sep: false,
            ignore_merges: false,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
            normalizer_opts: NormalizerOptions::default(),
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            cache_special_tokens: Vec::new(),
            cache_token_to_piece: Vec::new(),
            bpe_ranks: HashMap::new(),
            special_eog_ids: BTreeSet::new(),
            suppress_tokens: Vec::new(),
            tokenizer: Tokenizer::None,
            precompiled_charsmap: Vec::new(),
        }
    }

    /// `impl::init_tokenizer`
    fn init_tokenizer(&mut self) {
        self.tokenizer = match self.ty {
            VocabType::None => Tokenizer::None,
            VocabType::Spm => Tokenizer::Spm,
            VocabType::Bpe => {
                let (regex_exprs, byte_encode) = bpe_pre_regexes(self.pre_type);
                Tokenizer::Bpe(Box::new(BpeTokenizer {
                    regex_exprs,
                    byte_encode,
                }))
            }
            VocabType::Wpm => Tokenizer::Wpm,
            VocabType::Ugm => Tokenizer::Ugm(Box::new(UgmTokenizer::build(self))),
            VocabType::Rwkv => Tokenizer::Rwkv(Box::new(RwkvTokenizer::build(self))),
            // abeada335 (llama-vocab.cpp:3366-3370): pre_segment = false for
            // PLaMo-2 (the leading U+FEFF is dropped), true for PLaMo-3
            VocabType::Plamo2 => {
                Tokenizer::Plamo2(Box::new(Plamo2Tokenizer::build(self, false)))
            }
            VocabType::Plamo3 => {
                Tokenizer::Plamo2(Box::new(Plamo2Tokenizer::build(self, true)))
            }
            VocabType::Test => Tokenizer::Test,
        };
    }

    // -----------------------------------------------------------------------
    // accessors (llama_vocab::* getters)
    // -----------------------------------------------------------------------

    pub fn n_tokens(&self) -> u32 {
        self.id_to_token.len() as u32
    }

    pub fn get_type(&self) -> VocabType {
        self.ty
    }

    pub fn type_name(&self) -> &'static str {
        match self.ty {
            VocabType::None => "no vocab",
            VocabType::Spm => "SPM",
            VocabType::Bpe => "BPE",
            VocabType::Wpm => "WPM",
            VocabType::Ugm => "UGM",
            VocabType::Rwkv => "RWKV",
            VocabType::Plamo2 => "PLaMo2",
            VocabType::Plamo3 => "PLaMo3",
            VocabType::Test => "TEST",
        }
    }

    pub fn is_normal(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_NORMAL != 0
    }
    pub fn is_unknown(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_UNKNOWN != 0
    }
    pub fn is_control(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_CONTROL != 0
    }
    pub fn is_byte(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_BYTE != 0
    }
    pub fn is_user_defined(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_USER_DEFINED != 0
    }
    pub fn is_unused(&self, id: Token) -> bool {
        self.id_to_token[id as usize].attr & ATTR_UNUSED != 0
    }
    pub fn is_eog(&self, id: Token) -> bool {
        id != TOKEN_NULL && self.special_eog_ids.contains(&id)
    }

    pub fn token_get_attr(&self, id: Token) -> u32 {
        self.id_to_token[id as usize].attr
    }
    pub fn token_get_text(&self, id: Token) -> &str {
        &self.id_to_token[id as usize].text
    }
    pub fn token_get_score(&self, id: Token) -> f32 {
        self.id_to_token[id as usize].score
    }
    pub fn get_token_data(&self, id: Token) -> &TokenData {
        &self.id_to_token[id as usize]
    }

    pub fn token_bos(&self) -> Token {
        self.special_bos_id
    }
    pub fn token_eos(&self) -> Token {
        self.special_eos_id
    }
    pub fn token_eot(&self) -> Token {
        self.special_eot_id
    }
    pub fn token_eom(&self) -> Token {
        self.special_eom_id
    }
    pub fn token_unk(&self) -> Token {
        self.special_unk_id
    }
    pub fn token_sep(&self) -> Token {
        self.special_sep_id
    }
    pub fn token_nl(&self) -> Token {
        self.linefeed_id
    }
    pub fn token_pad(&self) -> Token {
        self.special_pad_id
    }
    pub fn token_mask(&self) -> Token {
        self.special_mask_id
    }

    // flag getters (llama_vocab::get_*)
    pub fn get_add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }
    pub fn get_add_bos(&self) -> bool {
        self.add_bos
    }
    pub fn get_add_eos(&self) -> bool {
        self.add_eos
    }
    pub fn get_add_sep(&self) -> bool {
        self.add_sep
    }
    pub fn get_ignore_merges(&self) -> bool {
        self.ignore_merges
    }
    pub fn get_clean_spaces(&self) -> bool {
        self.clean_spaces
    }
    pub fn get_remove_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }
    pub fn get_escape_whitespaces(&self) -> bool {
        self.escape_whitespaces
    }
    pub fn get_treat_whitespace_as_suffix(&self) -> bool {
        self.treat_whitespace_as_suffix
    }
    pub fn get_normalizer_opts(&self) -> NormalizerOptions {
        self.normalizer_opts
    }
    pub fn get_suppress_tokens(&self) -> &[Token] {
        &self.suppress_tokens
    }

    /// `impl::token_to_byte` — hex byte encoded in the token text `<0xXX>`.
    pub fn token_to_byte(&self, id: Token) -> u8 {
        debug_assert!(self.is_byte(id));
        let text = &self.id_to_token[id as usize].text;
        let buf = &text[3.min(text.len())..5.min(text.len())];
        u8::from_str_radix(buf, 16).unwrap_or(0)
    }

    /// `llama_vocab::byte_to_token` — `Err` mirrors the C++ `map.at` throw.
    pub fn byte_to_token(&self, ch: u8) -> Result<Token, String> {
        assert!(self.ty != VocabType::None);
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        match self.ty {
            VocabType::Spm | VocabType::Ugm => {
                let buf = [
                    b'<',
                    b'0',
                    b'x',
                    HEX[(ch >> 4) as usize],
                    HEX[(ch & 15) as usize],
                    b'>',
                ];
                if let Some(&t) = self.token_to_id.get(&buf[..]) {
                    return Ok(t);
                }
                // fall back to just the byte as a string
                self.token_to_id
                    .get(&[ch][..])
                    .copied()
                    .ok_or_else(|| format!("byte token for 0x{ch:02x} not found"))
            }
            VocabType::Wpm | VocabType::Bpe => self
                .token_to_id
                .get(unicode::byte_to_utf8(ch).as_bytes())
                .copied()
                .ok_or_else(|| format!("byte token for 0x{ch:02x} not found")),
            // PLaMo uses byte tokens in format <0xXX> (llama-vocab.cpp:
            // 4168-4172 covers PLAMO2 and PLAMO3 with one arm)
            VocabType::Plamo2 | VocabType::Plamo3 => {
                let buf = format!("<0x{ch:02X}>");
                self.token_to_id
                    .get(buf.as_bytes())
                    .copied()
                    .ok_or_else(|| format!("byte token for 0x{ch:02x} not found"))
            }
            VocabType::Rwkv => panic!("fatal error: byte_to_token on RWKV vocab"),
            VocabType::Test => Ok(TOKEN_NULL),
            VocabType::None => Err("no vocab".to_string()),
        }
    }

    /// `llama_vocab::text_to_token` (returns LLAMA_TOKEN_NULL when missing).
    pub fn text_to_token(&self, text: &[u8]) -> Token {
        match self.token_to_id.get(text) {
            Some(&t) => t,
            None => TOKEN_NULL,
        }
    }

    /// `llama_vocab::find_bpe_rank`.
    pub fn find_bpe_rank(&self, token_left: &[u8], token_right: &[u8]) -> i32 {
        debug_assert!(!token_left.contains(&b' '));
        debug_assert!(!token_right.contains(&b' '));
        match self
            .bpe_ranks
            .get(&(token_left.to_vec(), token_right.to_vec()))
        {
            Some(&r) => r,
            None => -1,
        }
    }

    /// `llama_vocab::get_bpe_merges`.
    pub fn get_bpe_merges(&self) -> Vec<String> {
        let max_rank = self.bpe_ranks.values().copied().max().unwrap_or(-1);
        let mut result = vec![String::new(); (max_rank + 1) as usize];
        for ((first, second), &rank) in &self.bpe_ranks {
            let mut s = String::from_utf8_lossy(first).into_owned();
            s.push(' ');
            s.push_str(&String::from_utf8_lossy(second));
            result[rank as usize] = s;
        }
        result
    }

    // -----------------------------------------------------------------------
    // tokenize
    // -----------------------------------------------------------------------

    /// `llama_vocab::tokenize(raw_text, add_special, parse_special)`.
    pub fn tokenize(&self, raw_text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        self.tokenize_bytes(raw_text.as_bytes(), add_special, parse_special)
    }

    /// Byte-level entry point (C `llama_tokenize` takes a char* + length and
    /// handles arbitrary bytes through the byte-fallback vocabulary). The
    /// internal path was always byte-based (`bpe_tokenize(&[u8], ..)` etc.);
    /// this just exposes it, matching the reference for non-UTF-8 input.
    pub fn tokenize_bytes(
        &self,
        raw_text: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Vec<Token> {
        self.tokenize_internal(raw_text, add_special, parse_special)
    }

    fn tokenize_internal(
        &self,
        raw_text: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Vec<Token> {
        let mut output: Vec<Token> = Vec::new();
        let mut fragment_buffer: Vec<Fragment> = Vec::new();

        if !raw_text.is_empty() {
            fragment_buffer.push(Fragment::RawText {
                offset: 0,
                length: raw_text.len(),
            });
            self.tokenizer_st_partition(raw_text, &mut fragment_buffer, parse_special);
        }

        match self.ty {
            VocabType::Spm => {
                // OG tokenizer behavior: encode('', add_special_tokens=True) -> [bos]
                let mut is_prev_special = true; // prefix with space if first token

                if add_special && self.add_bos {
                    assert!(self.special_bos_id != TOKEN_NULL);
                    output.push(self.special_bos_id);
                    is_prev_special = true;
                }

                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let mut text: Vec<u8> = Vec::with_capacity(length + 3);
                            // prefix with space if previous is special
                            if self.add_space_prefix && is_prev_special {
                                text.push(b' ');
                            }
                            text.extend_from_slice(&raw_text[*offset..*offset + *length]);
                            escape_whitespace(&mut text);
                            self.spm_tokenize(&text, &mut output);
                            is_prev_special = false;
                        }
                        Fragment::Token(t) => {
                            output.push(*t);
                            is_prev_special = true;
                        }
                    }
                }

                if add_special && self.add_eos {
                    assert!(self.special_eos_id != TOKEN_NULL);
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Bpe => {
                let tok_bpe = match &self.tokenizer {
                    Tokenizer::Bpe(t) => t,
                    _ => unreachable!("BPE tokenizer not initialized"),
                };

                let use_hybriddna = self.tokenizer_model == "hybriddna";
                let use_whitespace = self.tokenizer_model == "whitespace";

                if add_special && self.add_bos {
                    assert!(self.special_bos_id != TOKEN_NULL);
                    output.push(self.special_bos_id);
                }
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let mut text = raw_text[*offset..*offset + *length].to_vec();
                            if self.escape_whitespaces {
                                escape_whitespace(&mut text);
                            }
                            if use_hybriddna {
                                self.bpe_tokenize_hybriddna(tok_bpe, &text, &mut output);
                            } else if use_whitespace {
                                self.bpe_tokenize_whitespace(tok_bpe, &text, &mut output);
                            } else {
                                self.bpe_tokenize(tok_bpe, &text, &mut output);
                            }
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
                if add_special {
                    if self.add_eos {
                        assert!(self.special_eos_id != TOKEN_NULL);
                        output.push(self.special_eos_id);
                    }
                    self.check_double_bos_eos(&output);
                }
            }
            VocabType::Wpm => {
                if add_special {
                    assert!(self.special_bos_id != TOKEN_NULL);
                    output.push(self.special_bos_id);
                }
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let text = &raw_text[*offset..*offset + *length];
                            self.wpm_tokenize(text, &mut output);
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
                if add_special {
                    assert!(self.special_sep_id != TOKEN_NULL);
                    output.push(self.special_sep_id);
                }
            }
            VocabType::Ugm => {
                if add_special && self.add_bos {
                    assert!(self.special_bos_id != TOKEN_NULL);
                    output.push(self.special_bos_id);
                }
                let tok_ugm = match &self.tokenizer {
                    Tokenizer::Ugm(t) => t,
                    _ => unreachable!("UGM tokenizer not initialized"),
                };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let text = &raw_text[*offset..*offset + *length];
                            self.ugm_tokenize(tok_ugm, text, &mut output);
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
                if add_special && self.add_eos {
                    assert!(self.special_eos_id != TOKEN_NULL);
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Rwkv => {
                let tok_rwkv = match &self.tokenizer {
                    Tokenizer::Rwkv(t) => t,
                    _ => unreachable!("RWKV tokenizer not initialized"),
                };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let text = &raw_text[*offset..*offset + *length];
                            self.rwkv_tokenize(tok_rwkv, text, &mut output);
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
            }
            VocabType::Plamo2 | VocabType::Plamo3 => {
                // honor the BOS/EOS settings (llama-vocab.cpp:3602-3605 +
                // 3620-3628, db33d3cb8; abeada335 gave PLAMO3 the same arm)
                if add_special && self.add_bos {
                    assert!(self.special_bos_id != TOKEN_NULL);
                    output.push(self.special_bos_id);
                }
                let tok = match &self.tokenizer {
                    Tokenizer::Plamo2(t) => t,
                    _ => unreachable!("PLaMo2 tokenizer not initialized"),
                };
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let text = &raw_text[*offset..*offset + *length];
                            let tokens = tok.encode(text);
                            output.extend_from_slice(&tokens);
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
                if add_special && self.add_bos && output.len() >= 2 && output[1] == self.special_bos_id {
                    eprintln!(
                        "warn: tokenize: Added a BOS token to the prompt as specified by the model \
                         but the prompt also starts with a BOS token. So now the final prompt starts \
                         with 2 BOS tokens. Are you sure this is what you want?"
                    );
                }

                if add_special && self.add_eos {
                    assert!(self.special_eos_id != TOKEN_NULL);
                    output.push(self.special_eos_id);
                }
            }
            VocabType::Test => {
                const CHUNK_SIZE: usize = 5;
                let n_vocab = self.n_tokens() as u64;
                for fragment in &fragment_buffer {
                    match fragment {
                        Fragment::RawText { offset, length } => {
                            let text = &raw_text[*offset..*offset + *length];
                            let mut pos = 0usize;
                            while pos < text.len() {
                                let n = CHUNK_SIZE.min(text.len() - pos);
                                let mut hash: u64 = 0;
                                for i in 0..n {
                                    hash = hash.wrapping_mul(31).wrapping_add(text[pos + i] as u64);
                                }
                                output.push((hash % n_vocab) as Token);
                                pos += n;
                            }
                        }
                        Fragment::Token(t) => output.push(*t),
                    }
                }
            }
            VocabType::None => panic!("fatal error: tokenize on vocab type none"),
        }

        output
    }

    fn check_double_bos_eos(&self, output: &[Token]) {
        if self.add_bos && output.len() >= 2 && output[1] == self.special_bos_id {
            eprintln!(
                "warn: tokenize: Added a BOS token to the prompt as specified by the model but the prompt \
                 also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. \
                 Are you sure this is what you want?"
            );
        }
        if self.add_eos && output.len() >= 2 && output[output.len() - 2] == self.special_eos_id {
            eprintln!(
                "warn: tokenize: Added an EOS token to the prompt as specified by the model but the prompt \
                 also ends with an EOS token. So now the final prompt ends with 2 EOS tokens. \
                 Are you sure this is what you want?"
            );
        }
    }

    // -----------------------------------------------------------------------
    // special token partitioning (tokenizer_st_partition)
    // -----------------------------------------------------------------------

    fn tokenizer_st_partition(
        &self,
        raw_text: &[u8],
        buffer: &mut Vec<Fragment>,
        parse_special: bool,
    ) {
        // for each special token
        for i in 0..self.cache_special_tokens.len() {
            let special_id = self.cache_special_tokens[i];
            let (text, attr) = {
                let data = &self.id_to_token[special_id as usize];
                (data.text.clone(), data.attr)
            };
            let text = text.as_bytes();

            if !parse_special && (attr & (ATTR_CONTROL | ATTR_UNKNOWN)) != 0 {
                // ignore control/unknown tokens when parse_special == false;
                // user-defined tokens are still pre-tokenized before everything else
                continue;
            }

            let mut idx = 0usize;
            while idx < buffer.len() {
                let (base_offset, base_length) = match buffer[idx] {
                    Fragment::RawText { offset, length } => (offset, length),
                    Fragment::Token(_) => {
                        idx += 1;
                        continue;
                    }
                };

                // collect the replacement pieces for this fragment
                let mut pieces: Vec<Fragment> = Vec::new();
                let mut offset = base_offset;
                let mut length = base_length;

                loop {
                    // find first occurrence of `text` within raw_text[offset..offset+length)
                    let search_area = &raw_text[offset..offset + length];
                    let rel = find_sub(search_area, text);
                    let Some(rel) = rel else {
                        // no further occurrence: this remainder is raw text and
                        // is pushed exactly once, here
                        if length > 0 {
                            pieces.push(Fragment::RawText { offset, length });
                        }
                        break;
                    };
                    let m = offset + rel;

                    // text to the left of the match
                    if m > offset {
                        let left_offset = offset;
                        let mut left_length = m - offset;
                        if attr & ATTR_LSTRIP != 0 {
                            while left_length > 0
                                && is_space(raw_text[left_offset + left_length - 1])
                            {
                                left_length -= 1;
                            }
                        }
                        if left_length > 0 {
                            pieces.push(Fragment::RawText {
                                offset: left_offset,
                                length: left_length,
                            });
                        }
                    }

                    // the special token itself
                    pieces.push(Fragment::Token(special_id));

                    // text to the right of the match: it becomes the next
                    // "source" (the C erases the source fragment on every
                    // match and re-scans the right piece), so it is NOT pushed
                    // here — only the final remainder survives the loop
                    // (llama-vocab.cpp:3332-3347)
                    if m + text.len() < offset + length {
                        let mut right_offset = m + text.len();
                        let mut right_length = length - ((m - offset) + text.len());

                        if attr & ATTR_RSTRIP != 0 {
                            while right_length > 0 && is_space(raw_text[right_offset]) {
                                right_offset += 1;
                                right_length -= 1;
                            }
                        }

                        // continue with the right remainder
                        offset = right_offset;
                        length = right_length;
                        if length == 0 {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if pieces.is_empty() {
                    idx += 1;
                } else {
                    let n_pieces = pieces.len();
                    buffer.splice(idx..idx + 1, pieces);
                    idx += n_pieces;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // detokenize
    // -----------------------------------------------------------------------

    /// `llama_vocab::detokenize(tokens, special)` wrapper (remove_special=false).
    pub fn detokenize(&self, tokens: &[Token], special: bool) -> String {
        let bytes = self.detokenize_impl(tokens, false, special);
        String::from_utf8(bytes).unwrap_or_default()
    }

    /// `impl::detokenize` — the C-buffer version, writing into an unbounded
    /// Vec<u8> (equivalent to calling the C version with a sufficient buffer,
    /// as the std::string wrapper does).
    pub fn detokenize_impl(
        &self,
        tokens: &[Token],
        remove_special: bool,
        unparse_special: bool,
    ) -> Vec<u8> {
        if self.ty == VocabType::None {
            return Vec::new();
        }

        let mut text: Vec<u8> = Vec::new();

        // remove the leading space
        let mut remove_space = self.add_space_prefix;
        let mut tokens = tokens;

        if remove_special && self.add_bos && !tokens.is_empty() && tokens[0] == self.special_bos_id
        {
            remove_space = false;
            tokens = &tokens[1..];
        }

        if remove_special
            && self.add_eos
            && !tokens.is_empty()
            && tokens[tokens.len() - 1] == self.special_eos_id
        {
            tokens = &tokens[..tokens.len() - 1];
        }

        for &token in tokens {
            // cache-based token_to_piece with lstrip = remove_space
            const ATTR_SPECIAL: u32 = ATTR_UNKNOWN | ATTR_CONTROL;
            let attr = self.token_get_attr(token);
            if !unparse_special && (attr & ATTR_SPECIAL) != 0 {
                continue;
            }
            let mut piece: &[u8] = &self.cache_token_to_piece[token as usize];
            if remove_space {
                while !piece.is_empty() && piece[0] == b' ' {
                    piece = &piece[1..];
                }
            }
            text.extend_from_slice(piece);
            remove_space = false;
        }

        let total = text.len();

        if self.clean_spaces {
            // first pass: characters ?!.,
            let total1 = total;
            let mut w = if total1 > 0 { 1usize } else { 0 };
            for i in 1..total1 {
                let x = text[i];
                if text[i - 1] == b' ' && (x == b'?' || x == b'!' || x == b'.' || x == b',') {
                    // remove space (do not advance w)
                } else {
                    text[w] = x;
                    w += 1;
                }
            }

            // second pass: strip single apostrophe between spaces
            let total2 = w;
            let mut w = if total2 > 0 { 1usize } else { 0 };
            let mut i = 1usize;
            while i < total2 {
                let x = text[i];
                if x == b'\'' && i + 1 < total2 && text[i - 1] == b' ' && text[i + 1] == b' ' {
                    // remove prev space; skip next space
                    if w > 0 {
                        w -= 1;
                    }
                    i += 1;
                    text[i] = 0;
                }
                text[w] = x;
                w += 1;
                i += 1;
            }

            // third pass: apostrophe contractions
            let total3 = w;
            let mut w = if total3 > 0 { 1usize } else { 0 };
            for i in 1..total3 {
                let x = text[i];
                if text[i - 1] == b' ' && x == b'\'' && i + 1 < total3 {
                    let x1 = text[i + 1];
                    if x1 == b't' || x1 == b'd' {
                        // keep space
                    } else if x1 == b's' || x1 == b'm' {
                        w -= 1; // remove space
                    } else if i + 2 < total3 {
                        let x2 = text[i + 2];
                        if x1 == b'l' && x2 == b'l' {
                            // keep space
                        } else if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                            w -= 1; // remove space
                        }
                    }
                }
                text[w] = x;
                w += 1;
            }
            text.truncate(w);
        } else {
            text.truncate(total);
        }

        text
    }

    /// `impl::token_to_piece` (uncached path; used to build the cache).
    fn token_to_piece_uncached(&self, token: Token, special: bool) -> Vec<u8> {
        const ATTR_SPECIAL: u32 = ATTR_UNKNOWN | ATTR_CONTROL;
        let attr = self.token_get_attr(token);
        if !special && (attr & ATTR_SPECIAL) != 0 {
            return Vec::new();
        }

        if 0 <= token && (token as usize) < self.id_to_token.len() {
            let token_text = &self.id_to_token[token as usize].text;
            match self.ty {
                VocabType::Wpm | VocabType::Spm | VocabType::Ugm => {
                    // NOTE: we accept all unsupported token types,
                    // suppressing them like CONTROL tokens.
                    if attr & (ATTR_SPECIAL | ATTR_USER_DEFINED) != 0 {
                        return token_text.as_bytes().to_vec();
                    }
                    if attr & ATTR_NORMAL != 0 {
                        return unescape_whitespace(token_text);
                    }
                    if attr & ATTR_BYTE != 0 {
                        return vec![self.token_to_byte(token)];
                    }
                }
                VocabType::Bpe => {
                    if attr & (ATTR_SPECIAL | ATTR_USER_DEFINED) != 0 {
                        return token_text.as_bytes().to_vec();
                    }
                    if attr & ATTR_NORMAL != 0 {
                        if self.escape_whitespaces {
                            // SPM-style BPE: tokens contain ▁ for spaces
                            return unescape_whitespace(token_text);
                        }
                        return decode_text(token_text);
                    }
                    if attr & ATTR_BYTE != 0 {
                        return vec![self.token_to_byte(token)];
                    }
                }
                VocabType::Rwkv => {
                    return unescape_rwkv_token(token_text);
                }
                VocabType::Test => {
                    return format!("{token:x}").into_bytes();
                }
                VocabType::Plamo2 | VocabType::Plamo3 => {
                    if self.is_byte(token) {
                        // handle byte tokens like <0xXX>
                        let bytes = token_text.as_bytes();
                        if bytes.len() == 6 && &bytes[..3] == b"<0x" && bytes[5] == b'>' {
                            let hex = &token_text[3..5];
                            if hex.chars().all(|c| c.is_ascii_hexdigit()) {
                                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                                    return vec![byte];
                                }
                            }
                        }
                    }
                    return token_text.as_bytes().to_vec();
                }
                VocabType::None => {}
            }
        }

        Vec::new()
    }

    /// cached `token_to_piece(token, special = true)` as raw bytes.
    pub fn token_to_piece_bytes(&self, token: Token) -> &[u8] {
        &self.cache_token_to_piece[token as usize]
    }

    /// uncached `llama_vocab::token_to_piece(token, buf, len, lstrip = 0, special)`
    /// as owned bytes — the cache is built with `special = true`
    /// (llama-vocab.cpp:3054), so callers needing the `special = false` form
    /// (e.g. the infill sampler, llama-sampler.cpp:4155) go through this.
    /// (Added by the DRY/adaptive-p sampling task.)
    pub fn token_to_piece_special(&self, token: Token, special: bool) -> Vec<u8> {
        self.token_to_piece_uncached(token, special)
    }

    /// cached `token_to_piece(token)` as &str (lossy for non-UTF-8 RWKV pieces).
    pub fn token_to_piece(&self, token: Token) -> &str {
        std::str::from_utf8(&self.cache_token_to_piece[token as usize]).unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// fragments + helpers
// ---------------------------------------------------------------------------

/// `struct fragment_buffer_variant`.
#[derive(Debug, Clone, Copy)]
enum Fragment {
    Token(Token),
    /// (offset, length) into the raw text bytes
    RawText {
        offset: usize,
        length: usize,
    },
}

/// C `isspace` (includes \v, unlike Rust's `is_ascii_whitespace`).
#[inline]
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// `llama_escape_whitespace` — ' ' → U+2581 (▁)
fn escape_whitespace(text: &mut Vec<u8>) {
    let mut out = Vec::with_capacity(text.len());
    for &b in text.iter() {
        if b == b' ' {
            out.extend_from_slice(b"\xe2\x96\x81");
        } else {
            out.push(b);
        }
    }
    *text = out;
}

/// `llama_unescape_whitespace` — U+2581 (▁) → ' '
fn unescape_whitespace(text: &str) -> Vec<u8> {
    text.replace("\u{2581}", " ").into_bytes()
}

/// `llama_decode_text` — byte-decode a BPE token piece (GPT-2 mapping).
fn decode_text(text: &str) -> Vec<u8> {
    let mut decoded_text = Vec::new();
    for cpt in unicode::cpts_from_utf8(text.as_bytes()) {
        let utf8 = unicode::cpt_to_utf8(cpt);
        match unicode::utf8_to_byte(&utf8) {
            Some(b) => decoded_text.push(b),
            None => {
                decoded_text.extend_from_slice(b"[UNK_BYTE_0x");
                for c in utf8.as_bytes() {
                    decoded_text.extend_from_slice(format!("{:02x}", c).as_bytes());
                }
                // NOTE: mirrors the C++ exactly — the whole input text is appended
                decoded_text.extend_from_slice(text.as_bytes());
                decoded_text.push(b']');
            }
        }
    }
    decoded_text
}

/// `llama_unescape_rwkv_token` — decode \xNN / \t / \n / \r escapes.
fn unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut output = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining = 0u8;
    let mut hex_acc = 0u8;

    for &c in escaped.as_bytes() {
        if hex_remaining != 0 {
            let value: u8 = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = hex_acc.wrapping_shl(4).wrapping_add(value);
            hex_remaining -= 1;
            if hex_remaining == 0 {
                output.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        if escaping {
            match c {
                b't' => output.push(b'\t'),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b'x' => hex_remaining = 2,
                _ => output.push(c),
            }
            escaping = false;
            continue;
        }

        if c == b'\\' {
            escaping = true;
            continue;
        }

        output.push(c);
    }

    output
}

// ---------------------------------------------------------------------------
// SPM tokenizer (llm_tokenizer_spm_session)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Symbol {
    prev: i64,
    next: i64,
    /// byte range into the owning word/text
    start: usize,
    n: usize,
}

/// `llm_bigram_spm` — priority: higher score; tie → smaller left index.
struct BigramSpm {
    left: usize,
    right: usize,
    score: f32,
    size: usize,
}

impl PartialEq for BigramSpm {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for BigramSpm {}
impl PartialOrd for BigramSpm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigramSpm {
    fn cmp(&self, other: &Self) -> Ordering {
        // C++ comparator: (l.score < r.score) || (l.score == r.score && l.left > r.left)
        // → max-heap pops highest score first, ties → smallest `left`
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(other.left.cmp(&self.left))
    }
}

/// `llm_bigram_bpe` — priority: lower rank; tie → smaller left index.
#[derive(Clone)]
struct BigramBpe {
    left: usize,
    right: usize,
    text: Vec<u8>,
    rank: i32,
    #[allow(dead_code)]
    size: usize,
}

impl PartialEq for BigramBpe {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for BigramBpe {}
impl PartialOrd for BigramBpe {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigramBpe {
    fn cmp(&self, other: &Self) -> Ordering {
        // C++ comparator: l.rank > r.rank || (l.rank == r.rank && l.left > r.left)
        // → max-heap pops lowest rank first, ties → smallest `left`
        other.rank.cmp(&self.rank).then(other.left.cmp(&self.left))
    }
}

impl Vocab {
    fn spm_tokenize(&self, text: &[u8], output: &mut Vec<Token>) {
        // split string into utf8 chars
        let mut symbols: Vec<Symbol> = Vec::new();
        {
            let mut offs = 0usize;
            let mut index: i64 = 0;
            while offs < text.len() {
                let len = unicode::len_utf8(text[offs]);
                let n = len.min(text.len() - offs);
                symbols.push(Symbol {
                    prev: index - 1,
                    next: if offs + n == text.len() {
                        -1
                    } else {
                        index + 1
                    },
                    start: offs,
                    n,
                });
                offs += n;
                index += 1;
            }
        }

        let mut work_queue: BinaryHeap<BigramSpm> = BinaryHeap::new();
        let mut rev_merge: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();

        // seed the work queue with all possible 2-character tokens
        for i in 1..symbols.len() {
            self.spm_try_add_bigram(
                text,
                &symbols,
                i as i64 - 1,
                i as i64,
                &mut work_queue,
                &mut rev_merge,
            );
        }

        // keep substituting the highest frequency pairs for as long as we can
        while let Some(bigram) = work_queue.pop() {
            let (left_n, right_n, right_next) = {
                let left_sym = &symbols[bigram.left];
                let right_sym = &symbols[bigram.right];
                if left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size {
                    continue;
                }
                (left_sym.n, right_sym.n, right_sym.next)
            };
            let _ = (left_n, right_n);

            // merge the right sym into the left one
            symbols[bigram.left].n += symbols[bigram.right].n;
            symbols[bigram.right].n = 0;

            // remove the right sym from the chain
            symbols[bigram.left].next = right_next;
            if right_next >= 0 {
                symbols[right_next as usize].prev = bigram.left as i64;
            }

            // find more substitutions
            let prev = symbols[bigram.left].prev;
            let next = symbols[bigram.left].next;
            self.spm_try_add_bigram(
                text,
                &symbols,
                prev,
                bigram.left as i64,
                &mut work_queue,
                &mut rev_merge,
            );
            self.spm_try_add_bigram(
                text,
                &symbols,
                bigram.left as i64,
                next,
                &mut work_queue,
                &mut rev_merge,
            );
        }

        let mut i: i64 = 0;
        while i != -1 {
            self.spm_resegment(text, &symbols, &rev_merge, i as usize, output);
            i = symbols[i as usize].next;
        }
    }

    fn spm_try_add_bigram(
        &self,
        text: &[u8],
        symbols: &[Symbol],
        left: i64,
        right: i64,
        work_queue: &mut BinaryHeap<BigramSpm>,
        rev_merge: &mut HashMap<Vec<u8>, (usize, usize)>,
    ) {
        if left == -1 || right == -1 {
            return;
        }
        let (left, right) = (left as usize, right as usize);
        let l = &symbols[left];
        let r = &symbols[right];
        let word = &text[l.start..l.start + l.n + r.n];
        let token = self.text_to_token(word);
        if token == TOKEN_NULL {
            return;
        }
        if token as u32 >= self.n_tokens() {
            return;
        }
        let score = self.get_token_data(token).score;
        work_queue.push(BigramSpm {
            left,
            right,
            score,
            size: word.len(),
        });
        rev_merge.insert(word.to_vec(), (left, right));
    }

    fn spm_resegment(
        &self,
        text: &[u8],
        symbols: &[Symbol],
        rev_merge: &HashMap<Vec<u8>, (usize, usize)>,
        idx: usize,
        output: &mut Vec<Token>,
    ) {
        let sym = symbols[idx];
        let word = &text[sym.start..sym.start + sym.n];
        let token = self.text_to_token(word);

        if token != TOKEN_NULL {
            output.push(token);
            return;
        }

        match rev_merge.get(word) {
            None => {
                // output any symbols that did not form tokens as bytes
                output.reserve(output.len() + sym.n);
                for j in 0..sym.n {
                    let id = self.byte_to_token(text[sym.start + j]).unwrap_or_else(|e| {
                        panic!("SPM vocabulary, but byte token not found: {e}")
                    });
                    output.push(id);
                }
            }
            Some(&(l, r)) => {
                self.spm_resegment(text, symbols, rev_merge, l, output);
                self.spm_resegment(text, symbols, rev_merge, r, output);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BPE tokenizer (llm_tokenizer_bpe_session)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct WordSymbol {
    prev: i64,
    next: i64,
    /// index into word_collection
    word: usize,
    start: usize,
    n: usize,
}

impl Vocab {
    fn bpe_tokenize(&self, tok: &BpeTokenizer, text: &[u8], output: &mut Vec<Token>) {
        let mut final_prev_index: i64 = -1;
        // Invalid UTF-8 bytes decode to U+FFFD like the C pre-splitter's UTF-8
        // decoder (verified: raw 0xFF -> token of U+FFFD, reference
        // llama-tokenize gives [64, 5691, 65] for b"a\xffb"). Previously an
        // unwrap_or("") silently dropped the whole fragment.
        let lossy = std::string::String::from_utf8_lossy(text);
        let word_collection = unicode::regex_split(&lossy, &tok.regex_exprs, tok.byte_encode);

        // Metaspace pre-tokenizer (a7b94df2c, llama-vocab.cpp:632-650): a
        // text starts with an escaped space, and each escaped space starts a
        // new word
        let word_collection: Vec<std::string::String> = if self.pre_type == PreType::Mmbert {
            const SPACE: &str = "\u{2581}"; // "\xe2\x96\x81"
            let mut words: Vec<std::string::String> = Vec::new();
            for word in word_collection {
                if word.bytes().all(|b| b == b'\n') {
                    words.push(word);
                    continue;
                }
                let word = if !word.starts_with(SPACE) {
                    format!("{SPACE}{word}")
                } else {
                    word
                };
                let bytes = word.as_bytes();
                let space = SPACE.as_bytes();
                let mut start = 0usize;
                while start < bytes.len() {
                    // word.find(space, start + space.len()) — the C searches
                    // past the word's own leading marker
                    let from = (start + space.len()).min(bytes.len());
                    let end = bytes[from..]
                        .windows(space.len())
                        .position(|w| w == space)
                        .map(|i| from + i)
                        .unwrap_or(bytes.len());
                    words.push(std::string::String::from_utf8_lossy(&bytes[start..end]).into_owned());
                    start = end;
                }
            }
            words
        } else {
            word_collection
        };

        let mut symbols_final: Vec<WordSymbol> = Vec::new();

        for (word_idx, word) in word_collection.iter().enumerate() {
            let word_bytes = word.as_bytes();
            let mut work_queue: BinaryHeap<BigramBpe> = BinaryHeap::new();
            let mut symbols: Vec<WordSymbol> = Vec::new();

            let mut index: i64 = 0;
            let mut offset = 0usize;

            if self.ignore_merges && self.text_to_token(word_bytes) != TOKEN_NULL {
                symbols.push(WordSymbol {
                    prev: -1,
                    next: -1,
                    word: word_idx,
                    start: 0,
                    n: word_bytes.len(),
                });
                offset = word_bytes.len();
            } else if matches!(self.pre_type, PreType::Gemma4 | PreType::Mmbert)
                && word_bytes.iter().all(|&b| b == b'\n')
            {
                // fix for gemma 4 (and mmbert, a7b94df2c): whole-newline
                // words tokenize directly
                if self.text_to_token(word_bytes) != TOKEN_NULL {
                    symbols.push(WordSymbol {
                        prev: -1,
                        next: -1,
                        word: word_idx,
                        start: 0,
                        n: word_bytes.len(),
                    });
                    offset = word_bytes.len();
                }
            }

            while offset < word_bytes.len() {
                let char_len =
                    (word_bytes.len() - offset).min(unicode::len_utf8(word_bytes[offset]));
                let start = offset;
                offset += char_len;
                symbols.push(WordSymbol {
                    prev: index - 1,
                    next: if offset == word_bytes.len() {
                        -1
                    } else {
                        index + 1
                    },
                    word: word_idx,
                    start,
                    n: char_len,
                });
                index += 1;
            }

            for i in 1..symbols.len() {
                self.bpe_add_new_bigram(
                    &word_collection,
                    &symbols,
                    i as i64 - 1,
                    i as i64,
                    &mut work_queue,
                );
            }

            // build token(s)
            while let Some(bigram) = work_queue.pop() {
                let (left_n, right_n) = (symbols[bigram.left].n, symbols[bigram.right].n);
                if left_n == 0 || right_n == 0 {
                    continue;
                }
                let l = &symbols[bigram.left];
                let r = &symbols[bigram.right];
                let left_token = &word_collection[l.word].as_bytes()[l.start..l.start + l.n];
                let right_token = &word_collection[r.word].as_bytes()[r.start..r.start + r.n];
                let mut combined = Vec::with_capacity(left_token.len() + right_token.len());
                combined.extend_from_slice(left_token);
                combined.extend_from_slice(right_token);
                if combined != bigram.text {
                    continue; // outdated bigram
                }

                // merge the right sym into the left one
                symbols[bigram.left].n += symbols[bigram.right].n;
                symbols[bigram.right].n = 0;

                // remove the right sym from the chain
                let right_next = symbols[bigram.right].next;
                symbols[bigram.left].next = right_next;
                if right_next >= 0 {
                    symbols[right_next as usize].prev = bigram.left as i64;
                }

                let prev = symbols[bigram.left].prev;
                let next = symbols[bigram.left].next;
                self.bpe_add_new_bigram(
                    &word_collection,
                    &symbols,
                    prev,
                    bigram.left as i64,
                    &mut work_queue,
                );
                self.bpe_add_new_bigram(
                    &word_collection,
                    &symbols,
                    bigram.left as i64,
                    next,
                    &mut work_queue,
                );
            }

            // add the finished tokens to the final list keeping order
            for i in 0..symbols.len() {
                if symbols[i].n > 0 {
                    symbols[i].prev = final_prev_index;
                    symbols[i].next = -1;
                    if final_prev_index != -1 {
                        let idx = final_prev_index as usize;
                        symbols_final[idx].next = symbols_final.len() as i64;
                    }
                    final_prev_index = symbols_final.len() as i64;
                    symbols_final.push(symbols[i]);
                }
            }
        }

        if !symbols_final.is_empty() {
            let mut i: i64 = 0;
            while i != -1 {
                let sym = symbols_final[i as usize];
                if sym.n == 0 {
                    i = symbols_final[i as usize].next;
                    continue;
                }

                let word_bytes = &word_collection[sym.word].as_bytes();
                let str_ = &word_bytes[sym.start..sym.start + sym.n];
                let token = self.text_to_token(str_);

                if token == TOKEN_NULL {
                    for &b in str_ {
                        let token_multibyte = if tok.byte_encode {
                            self.text_to_token(&[b])
                        } else {
                            // non-byte-encoded BPE (e.g. gemma-4): <0xXX> byte tokens
                            const HEX: &[u8; 16] = b"0123456789ABCDEF";
                            let buf = [
                                b'<',
                                b'0',
                                b'x',
                                HEX[(b >> 4) as usize],
                                HEX[(b & 15) as usize],
                                b'>',
                            ];
                            self.text_to_token(&buf)
                        };
                        if token_multibyte != TOKEN_NULL {
                            output.push(token_multibyte);
                        }
                    }
                } else {
                    output.push(token);
                }

                i = symbols_final[i as usize].next;
            }
        }
    }

    fn bpe_add_new_bigram(
        &self,
        word_collection: &[String],
        symbols: &[WordSymbol],
        left: i64,
        right: i64,
        work_queue: &mut BinaryHeap<BigramBpe>,
    ) {
        if left == -1 || right == -1 {
            return;
        }
        let (left, right) = (left as usize, right as usize);
        let l = &symbols[left];
        let r = &symbols[right];
        let lw = &word_collection[l.word].as_bytes()[l.start..l.start + l.n];
        let rw = &word_collection[r.word].as_bytes()[r.start..r.start + r.n];

        let rank = self.find_bpe_rank(lw, rw);
        if rank < 0 {
            return;
        }

        let mut text = Vec::with_capacity(lw.len() + rw.len());
        text.extend_from_slice(lw);
        text.extend_from_slice(rw);

        work_queue.push(BigramBpe {
            left,
            right,
            size: text.len(),
            text,
            rank,
        });
    }

    /// `llm_tokenizer_hybriddna_session::tokenize`
    fn bpe_tokenize_hybriddna(&self, tok: &BpeTokenizer, text: &[u8], output: &mut Vec<Token>) {
        const OPEN_TAG: &[u8] = b"<dna>";
        const CLOSE_TAG: &[u8] = b"</dna>";

        let dna_begin_id = self.text_to_token(OPEN_TAG);
        let dna_end_id = self.text_to_token(CLOSE_TAG);
        let dna_oov_id = self.text_to_token(b"<oov>");

        if dna_begin_id == TOKEN_NULL || dna_end_id == TOKEN_NULL || dna_oov_id == TOKEN_NULL {
            self.bpe_tokenize(tok, text, output);
            return;
        }

        const K: usize = 6;
        let mut pos = 0usize;

        while pos < text.len() {
            let start = match find_sub(&text[pos..], OPEN_TAG) {
                Some(rel) => pos + rel,
                None => {
                    if pos < text.len() {
                        self.bpe_tokenize(tok, &text[pos..], output);
                    }
                    break;
                }
            };
            if start > pos {
                self.bpe_tokenize(tok, &text[pos..start], output);
            }
            output.push(dna_begin_id);

            let content_start = start + OPEN_TAG.len();
            let end = find_sub(&text[content_start..], CLOSE_TAG).map(|rel| content_start + rel);
            let content_end = end.unwrap_or(text.len());

            self.emit_dna_kmers(&text[content_start..content_end], K, dna_oov_id, output);

            let Some(end) = end else { break };
            output.push(dna_end_id);
            pos = end + CLOSE_TAG.len();
        }
    }

    fn emit_dna_kmers(&self, raw: &[u8], k: usize, oov_id: Token, output: &mut Vec<Token>) {
        let mut seq = raw.to_vec();
        for c in seq.iter_mut() {
            if c.is_ascii_lowercase() {
                *c = c.to_ascii_uppercase();
            }
        }

        // k-mers carry the reserved marker suffix; a non-ACGT k-mer simply
        // isn't in the vocab and falls back to <oov>
        let kmer_token = |kmer: &[u8]| -> Token {
            let mut key = kmer.to_vec();
            key.extend_from_slice(DNA_KMER_MARKER);
            let tok = self.text_to_token(&key);
            if tok != TOKEN_NULL {
                tok
            } else {
                oov_id
            }
        };

        let mut i = 0usize;
        while i + k <= seq.len() {
            output.push(kmer_token(&seq[i..i + k]));
            i += k;
        }
        if i < seq.len() {
            let mut kmer = seq[i..].to_vec();
            kmer.resize(k, b'A');
            output.push(kmer_token(&kmer));
        }
    }

    /// `llm_tokenizer_whitespace_session::tokenize`
    fn bpe_tokenize_whitespace(&self, tok: &BpeTokenizer, text: &[u8], output: &mut Vec<Token>) {
        let lowercase = self.normalizer_opts.lowercase;

        let mut segment: Vec<u8> = Vec::new();
        for cpt in unicode::cpts_from_utf8(text) {
            if unicode::cpt_flags_from_cpt(cpt).is_whitespace() {
                if !segment.is_empty() {
                    self.bpe_tokenize(tok, &segment, output);
                    segment.clear();
                }
            } else {
                let c = if lowercase {
                    unicode::tolower(cpt)
                } else {
                    cpt
                };
                segment.extend_from_slice(unicode::cpt_to_utf8(c).as_bytes());
            }
        }
        if !segment.is_empty() {
            self.bpe_tokenize(tok, &segment, output);
        }
    }
}

// ---------------------------------------------------------------------------
// WPM tokenizer (llm_tokenizer_wpm_session)
// ---------------------------------------------------------------------------

impl Vocab {
    fn wpm_tokenize(&self, text: &[u8], output: &mut Vec<Token>) {
        // normalize and split by whitespace
        let words = wpm_preprocess(text, &self.normalizer_opts);

        for word in words {
            if word.is_empty() {
                continue;
            }

            // prepend phantom space (U+2581)
            let mut word1: Vec<u8> = Vec::with_capacity(word.len() + 3);
            word1.extend_from_slice(b"\xe2\x96\x81");
            word1.extend_from_slice(&word);
            let n = word1.len();

            let current_tokens = output.len();

            let mut i = 0usize;
            while i < n {
                let mut matched = false;
                let mut j = n.min(i + self.max_token_len + 1);
                while j > i {
                    if let Some(&id) = self.token_to_id.get(&word1[i..j]) {
                        output.push(id);
                        matched = true;
                        i = j;
                        break;
                    }
                    j -= 1;
                }

                if !matched {
                    output.truncate(current_tokens); // discard all
                    break; // and discard next tokens
                }
            }

            if current_tokens == output.len() {
                output.push(self.token_unk());
            }
        }
    }
}

/// `llm_tokenizer_wpm_session::preprocess`
fn wpm_preprocess(text: &[u8], normalizer_opts: &NormalizerOptions) -> Vec<Vec<u8>> {
    let mut cpts = unicode::cpts_from_utf8(text);
    if normalizer_opts.strip_accents {
        cpts = unicode::cpts_normalize_nfd(&cpts);
    }

    let mut words: Vec<Vec<u8>> = vec![Vec::new()];

    for cpt in cpts {
        let flags = unicode::cpt_flags_from_cpt(cpt);

        if flags.is_whitespace() {
            if !words.last().unwrap().is_empty() {
                words.push(Vec::new());
            }
            continue;
        }

        if cpt == 0 || cpt == 0xFFFD || flags.is_control() {
            continue;
        }

        if normalizer_opts.strip_accents && flags.is_accent_mark() {
            continue;
        }

        let s = unicode::cpt_to_utf8(if normalizer_opts.lowercase {
            unicode::tolower(cpt)
        } else {
            cpt
        });
        if flags.is_punctuation() || (cpt < 0x7F && flags.is_symbol()) || wpm_is_chinese_char(cpt) {
            if !words.last().unwrap().is_empty() {
                words.push(Vec::new());
            }
            *words.last_mut().unwrap() = s.into_bytes(); // single char word
            words.push(Vec::new()); // start a new word
        } else {
            words.last_mut().unwrap().extend_from_slice(s.as_bytes());
        }
    }

    if words.last().unwrap().is_empty() {
        words.pop();
    }

    words
}

/// `llm_tokenizer_wpm_session::is_chinese_char`
fn wpm_is_chinese_char(cpt: u32) -> bool {
    (0x04E00..=0x09FFF).contains(&cpt)
        || (0x03400..=0x04DBF).contains(&cpt)
        || (0x20000..=0x2A6DF).contains(&cpt)
        || (0x2A700..=0x2B73F).contains(&cpt)
        || (0x2B740..=0x2B81F).contains(&cpt)
        || (0x2B920..=0x2CEAF).contains(&cpt) // this should be 0x2B820 but in hf rust code it is 0x2B920
        || (0x0F900..=0x0FAFF).contains(&cpt)
        || (0x2F800..=0x2FA1F).contains(&cpt)
}

// ---------------------------------------------------------------------------
// UGM tokenizer (llm_tokenizer_ugm + session)
// ---------------------------------------------------------------------------

impl UgmTokenizer {
    fn build(v: &Vocab) -> UgmTokenizer {
        let charsmap = &v.precompiled_charsmap;
        let mut prefix_replacements: Vec<u8> = Vec::new();
        let mut xcda_array: Vec<u32> = Vec::new();

        if !charsmap.is_empty() {
            let xcda_blob_size =
                u32::from_le_bytes([charsmap[0], charsmap[1], charsmap[2], charsmap[3]]) as usize;
            let mut charsmap_offset = 4usize;
            let n = xcda_blob_size / 4;
            xcda_array = Vec::with_capacity(n);
            for i in 0..n {
                let o = charsmap_offset + i * 4;
                xcda_array.push(u32::from_le_bytes([
                    charsmap[o],
                    charsmap[o + 1],
                    charsmap[o + 2],
                    charsmap[o + 3],
                ]));
            }
            charsmap_offset += xcda_blob_size;
            prefix_replacements = charsmap[charsmap_offset..].to_vec();
        }

        let mut ugm = UgmTokenizer {
            escaped_space: b"\xe2\x96\x81",
            prefix_replacements,
            xcda_array,
            user_defined_token_matcher: Trie::default(),
            min_score: f32::MAX,
            max_score: -f32::MAX,
            unknown_token_score: 0.0,
            token_matcher: Trie::default(),
        };

        for id in 0..v.n_tokens() as Token {
            let token_data = &v.id_to_token[id as usize];

            if v.is_normal(id) {
                ugm.min_score = ugm.min_score.min(token_data.score);
                ugm.max_score = ugm.max_score.max(token_data.score);
            }

            if v.is_normal(id) || v.is_user_defined(id) || v.is_unused(id) {
                ugm.token_matcher.insert(token_data.text.as_bytes(), id);
            }

            if v.is_user_defined(id) {
                ugm.user_defined_token_matcher
                    .insert(token_data.text.as_bytes(), 0);
            }
        }

        ugm.unknown_token_score = ugm.min_score - 10.0;

        ugm
    }

    // XCDA view helpers (xcda_array_view)
    fn xcda_node(&self, index: usize) -> u32 {
        *self
            .xcda_array
            .get(index)
            .expect("Index out of array bounds in XCDA array!")
    }
    fn xcda_get_base(&self, index: usize) -> u32 {
        let packed_node = self.xcda_node(index);
        (packed_node >> 10) << ((packed_node & (1 << 9)) >> 6)
    }
    fn xcda_get_lcheck(&self, index: usize) -> u32 {
        self.xcda_node(index) & ((1 << 31) | 0xff)
    }
    fn xcda_get_leaf(&self, index: usize) -> bool {
        (self.xcda_node(index) >> 8) & 1 == 1
    }
    fn xcda_get_value(&self, index: usize) -> u32 {
        self.xcda_node(index) & ((1 << 31) - 1)
    }
}

#[derive(Clone, Copy)]
struct BestTokenization {
    token_id: Token,
    input_offset: usize,
    score_sum: f64,
}

/// Where the bytes that replace the consumed input prefix come from. The
/// reference (`normalization_result`) carries a `const char *` that points at
/// either the input, the charsmap table or a string literal.
#[derive(Clone, Copy)]
enum NormalizedBytes {
    /// charsmap replacement table entry, `[start, start + len)`
    Charsmap { start: usize, len: usize },
    /// slice of the original input, `[start, start + len)`
    Input { start: usize, len: usize },
    /// a literal — the U+FFFD fallback is written as `"\xEF\xBF\xBD"` in C
    Literal(&'static [u8]),
}

struct NormalizationResult {
    repl: NormalizedBytes,
    consumed_input: usize,
}

impl Vocab {
    fn ugm_tokenize(&self, tok: &UgmTokenizer, text: &[u8], output: &mut Vec<Token>) {
        let output_size = output.len();

        // normalize the input first
        let normalized = ugm_normalize(self, tok, text);
        let input_len = normalized.len();
        if input_len == 0 {
            return;
        }

        // Viterbi over tokenization prefixes
        let mut tokenization_results = vec![
            BestTokenization {
                token_id: self.token_unk(),
                input_offset: 0,
                score_sum: f64::MIN
            };
            input_len + 1
        ];
        tokenization_results[0] = BestTokenization {
            token_id: self.token_unk(),
            input_offset: 0,
            score_sum: 0.0,
        };

        let mut input_offset = 0usize;
        while input_offset < input_len {
            let mut prefix_offset = input_offset;
            // how many code units are in the currently processed UTF code point
            let n_utf8_code_units =
                unicode::len_utf8(normalized[input_offset]).min(input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            let mut single_codepoint_token_found = false;
            let current_best_score = tokenization_results[input_offset].score_sum;
            let mut node = tok.token_matcher.traverse(normalized[prefix_offset]);
            prefix_offset += 1;

            while prefix_offset <= input_len {
                let Some(n) = node else { break };
                if n.has_value {
                    // does it correspond to the whole UTF code point?
                    if prefix_offset - input_offset == n_utf8_code_units {
                        single_codepoint_token_found = true;
                    }
                    let token_id = n.value;
                    // user-defined token scores are 0 to make them more likely
                    let token_score: f64 = if self.is_user_defined(token_id) {
                        0.0
                    } else {
                        self.get_token_data(token_id).score as f64
                    };
                    let challenger_score = current_best_score + token_score;
                    let champ = &mut tokenization_results[prefix_offset];
                    if challenger_score > champ.score_sum {
                        *champ = BestTokenization {
                            token_id,
                            input_offset,
                            score_sum: challenger_score,
                        };
                    }
                }
                let c = if prefix_offset < input_len {
                    normalized[prefix_offset]
                } else {
                    0
                };
                node = n.traverse(c);
                prefix_offset += 1;
            }

            // no valid token for the whole code point → unknown token
            if !single_codepoint_token_found {
                let challenger_score = current_best_score + tok.unknown_token_score as f64;
                prefix_offset = input_offset + n_utf8_code_units;
                let champ = &mut tokenization_results[prefix_offset];
                if challenger_score > champ.score_sum {
                    *champ = BestTokenization {
                        token_id: self.token_unk(),
                        input_offset,
                        score_sum: challenger_score,
                    };
                }
            }

            input_offset += n_utf8_code_units;
        }

        // backtrack from the end; merge consecutive unknown tokens
        let mut is_prev_unknown = false;
        let mut idx = input_len;
        loop {
            let tokenization = tokenization_results[idx];
            let is_unknown = tokenization.token_id == self.token_unk();
            if !(is_prev_unknown && is_unknown) {
                output.push(tokenization.token_id);
            }
            if tokenization.input_offset == 0 {
                break;
            }
            is_prev_unknown = is_unknown;
            idx = tokenization.input_offset;
        }

        output[output_size..].reverse();
    }
}

/// `llm_tokenizer_ugm_session::normalize`
fn ugm_normalize(v: &Vocab, tok: &UgmTokenizer, input: &[u8]) -> Vec<u8> {
    let mut normalized: Vec<u8> = Vec::with_capacity(input.len() * 3);

    let space: &[u8] = if v.escape_whitespaces {
        tok.escaped_space
    } else {
        b" "
    };

    let shall_prepend_space = !v.treat_whitespace_as_suffix && v.add_space_prefix;
    let shall_append_space = v.treat_whitespace_as_suffix && v.add_space_prefix;
    let shall_merge_spaces = v.remove_extra_whitespaces;

    let mut is_space_prepended = false;
    let mut processing_non_ws = false;

    let mut input_offset = 0usize;
    while input_offset < input.len() {
        let norm_res = ugm_normalize_prefix(tok, input, input_offset);
        let (repl_bytes, consumed): (&[u8], usize) = match norm_res.repl {
            NormalizedBytes::Charsmap { start, len } => (
                &tok.prefix_replacements[start..start + len],
                norm_res.consumed_input,
            ),
            NormalizedBytes::Input { start, len } => {
                (&input[start..start + len], norm_res.consumed_input)
            }
            NormalizedBytes::Literal(bytes) => (bytes, norm_res.consumed_input),
        };

        for &c in repl_bytes {
            if c != b' ' {
                if !processing_non_ws {
                    processing_non_ws = true;
                    if (shall_prepend_space && !is_space_prepended) || shall_merge_spaces {
                        normalized.extend_from_slice(space);
                        is_space_prepended = true;
                    }
                }
                normalized.push(c);
            } else {
                if processing_non_ws {
                    processing_non_ws = false;
                }
                if !shall_merge_spaces {
                    normalized.extend_from_slice(space);
                }
            }
        }

        input_offset += consumed;
    }

    if shall_append_space {
        normalized.extend_from_slice(space);
    }

    normalized
}

/// `llm_tokenizer_ugm_session::normalize_prefix`
fn ugm_normalize_prefix(
    tok: &UgmTokenizer,
    input: &[u8],
    input_offset: usize,
) -> NormalizationResult {
    if input_offset == input.len() {
        return NormalizationResult {
            repl: NormalizedBytes::Input {
                start: input_offset,
                len: 0,
            },
            consumed_input: 0,
        };
    }

    // user-defined token as normalization result
    let user_match = tok
        .user_defined_token_matcher
        .get_longest_prefix(&input[input_offset..], 0);
    if user_match > 0 {
        return NormalizationResult {
            repl: NormalizedBytes::Input {
                start: input_offset,
                len: user_match,
            },
            consumed_input: user_match,
        };
    }

    let mut longest_prefix_length = 0usize;
    let mut longest_prefix_offset = 0usize;

    if !tok.xcda_array.is_empty() {
        // walk the XOR-compressed compact double array (XCDA) from the root
        let mut node_index: u32 = 0;
        node_index = tok.xcda_get_base(node_index as usize);
        for prefix_offset in input_offset..input.len() {
            let c = input[prefix_offset];
            if c == 0 {
                break;
            }
            node_index ^= c as u32;
            if tok.xcda_get_lcheck(node_index as usize) != c as u32 {
                break;
            }
            let is_leaf = tok.xcda_get_leaf(node_index as usize);
            node_index ^= tok.xcda_get_base(node_index as usize);
            if is_leaf {
                longest_prefix_length = prefix_offset - input_offset + 1;
                longest_prefix_offset = tok.xcda_get_value(node_index as usize) as usize;
            }
        }
    }

    if longest_prefix_length > 0 {
        // return the replacement sequence from the charsmap
        assert!(
            longest_prefix_offset < tok.prefix_replacements.len(),
            "Index out of array bounds in precompiled charsmap!"
        );
        let max_len = tok.prefix_replacements.len() - longest_prefix_offset;
        let mut repl_len = 0usize;
        while repl_len < max_len && tok.prefix_replacements[longest_prefix_offset + repl_len] != 0 {
            repl_len += 1;
        }
        assert!(
            repl_len != max_len,
            "Unterminated string in precompiled charsmap!"
        );
        return NormalizationResult {
            repl: NormalizedBytes::Charsmap {
                start: longest_prefix_offset,
                len: repl_len,
            },
            consumed_input: longest_prefix_length,
        };
    }

    // try to consume one valid UTF-8 code point
    let mut prefix_offset = input_offset;
    match unicode::cpt_from_utf8(input, &mut prefix_offset) {
        Ok(_) => NormalizationResult {
            repl: NormalizedBytes::Input {
                start: input_offset,
                len: prefix_offset - input_offset,
            },
            consumed_input: prefix_offset - input_offset,
        },
        Err(()) => NormalizationResult {
            // "if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER"
            repl: NormalizedBytes::Literal("\u{FFFD}".as_bytes()),
            consumed_input: 1,
        },
    }
}

// ---------------------------------------------------------------------------
// RWKV tokenizer (llm_tokenizer_rwkv + session)
// ---------------------------------------------------------------------------

impl RwkvTokenizer {
    fn build(v: &Vocab) -> RwkvTokenizer {
        let mut matcher = Trie::default();
        for id in 0..v.n_tokens() as Token {
            let data = &v.id_to_token[id as usize];
            let text = unescape_rwkv_token(&data.text);
            matcher.insert(&text, id);
        }
        RwkvTokenizer {
            token_matcher: matcher,
        }
    }
}

impl Vocab {
    fn rwkv_tokenize(&self, tok: &RwkvTokenizer, text: &[u8], output: &mut Vec<Token>) {
        let mut position = 0usize;
        while position < text.len() {
            let Some(mut node) = tok.token_matcher.traverse(text[position]) else {
                // no matching token found, add unknown token
                output.push(self.token_unk());
                position += 1;
                continue;
            };

            // traverse the trie to find the longest matching token
            let mut token_id: Token = 0;
            let mut token_length = 0usize;
            loop {
                if node.has_value {
                    token_id = node.value;
                    token_length = position + 1;
                }
                if position + 1 >= text.len() {
                    break;
                }
                position += 1;
                match node.traverse(text[position]) {
                    Some(n) => node = n,
                    None => break,
                }
            }

            // add the longest matching token
            output.push(token_id);
            position = token_length;
        }
    }
}

// ---------------------------------------------------------------------------
// PLaMo-2 tokenizer (llm_tokenizer_plamo2 + session)
// ---------------------------------------------------------------------------

/// the PLaMo-3 pre-segmentation of abeada335 (llama-vocab.cpp:1550-1617):
/// the cut points of pass 1 (`<|plamo:...|>` fencing) and pass 2 (runs),
/// then the U+EE00 boundaries — returns the emitted segments' `[start, end)`
/// ranges (the boundary marker itself is skipped). Public for the parity
/// probe test (parity/ref_k2_plamo_split.c + tests/k2_plamo_split_parity.rs).
pub fn plamo3_segments(unicode_data: &[u32]) -> Vec<(usize, usize)> {
    let n = unicode_data.len();
    let mut cut = vec![false; n + 1];

    // pass 1: <|plamo:...|> (:1554-1577)
    {
        const PREFIX: [u32; 8] = ['<' as u32, '|' as u32, 'p' as u32, 'l' as u32,
                                  'a' as u32, 'm' as u32, 'o' as u32, ':' as u32];
        let prefix_len = PREFIX.len();
        let mut i = 0usize;
        while i + prefix_len <= n {
            if unicode_data[i..i + prefix_len] != PREFIX {
                i += 1;
                continue;
            }
            // An empty body is valid.
            let mut j = i + prefix_len;
            // Treat U+001C..U+001F as whitespace (equivalent to Python \s).
            while j < n
                && j - (i + prefix_len) < 64
                && unicode_data[j] != '|' as u32
                && !(0x1C..=0x1F).contains(&unicode_data[j])
                && !unicode::cpt_flags_from_cpt(unicode_data[j]).is_whitespace()
            {
                j += 1;
            }
            if j + 1 < n && unicode_data[j] == '|' as u32 && unicode_data[j + 1] == '>' as u32
            {
                cut[i] = true;
                cut[j + 2] = true;
                i = j + 2;
            } else {
                i += 1;
            }
        }
    }

    // pass 2: runs of repeated characters / spaces (a run never crosses
    // a boundary from pass 1) (:1580-1599)
    {
        let mut i = 0usize;
        while i < n {
            let c = unicode_data[i];
            let mut run = 1usize;
            while i + run < n && unicode_data[i + run] == c && !cut[i + run] {
                run += 1;
            }

            let is_repeated_chars = c != '\n' as u32 && run >= 4;
            let is_spaces = c == ' ' as u32 && run >= 2;
            if is_repeated_chars || is_spaces {
                cut[i] = true;
                cut[i + run] = true;
            }

            // a run that does not match cannot match at any later
            // position either (it only gets shorter)
            i += run;
        }
    }

    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0usize;
    for seg_end in 0..=n {
        // U+EE00 is the tokenizer's private-use boundary marker; literal
        // occurrences split segments and are not emitted (:1604-1617)
        let is_boundary = seg_end < n && unicode_data[seg_end] == 0xEE00;
        if seg_end == n || cut[seg_end] || is_boundary {
            if seg_start < seg_end {
                segments.push((seg_start, seg_end));
            }
            seg_start = seg_end + usize::from(is_boundary);
        }
    }

    segments
}

impl Plamo2Tokenizer {
    /// `llm_tokenizer_plamo2(vocab, pre_segment)` (abeada335,
    /// llama-vocab.cpp:1403) — pre_segment is true only for PLaMo-3
    fn build(v: &Vocab, pre_segment: bool) -> Plamo2Tokenizer {
        let mut tokens: Vec<String> = Vec::new();
        let mut bytes = [0 as Token; 256];
        let mut to_suffix_id: HashMap<i64, i32> = HashMap::new();

        let mut suffix_to_score: HashMap<String, f32> = HashMap::new();
        let mut token_to_id: HashMap<String, Token> = HashMap::new();

        for token_id in 0..v.n_tokens() as usize {
            let entry = &v.id_to_token[token_id];
            tokens.push(entry.text.clone());
            token_to_id.insert(entry.text.clone(), token_id as Token);

            // handle byte tokens
            if v.is_byte(token_id as Token) {
                let t = entry.text.as_bytes();
                if t.len() == 6 && &t[..3] == b"<0x" && t[5] == b'>' {
                    let hex = &entry.text[3..5];
                    if hex.chars().all(|c| c.is_ascii_hexdigit()) {
                        if let Ok(byte_val) = u8::from_str_radix(hex, 16) {
                            bytes[byte_val as usize] = token_id as Token;
                        }
                    }
                }
                continue;
            }

            // add token and all its suffixes to suffix_to_score
            suffix_to_score.insert(entry.text.clone(), entry.score);

            let cpts = unicode::cpts_from_utf8(entry.text.as_bytes());
            for i in 1..cpts.len() {
                let mut suffix = String::new();
                for &cpt in &cpts[i..] {
                    suffix.push_str(&unicode::cpt_to_utf8(cpt));
                }
                suffix_to_score.entry(suffix).or_insert(f32::NAN);
            }
        }

        // check that all byte tokens are set
        for (i, &b) in bytes.iter().enumerate() {
            if b == 0 {
                panic!("Byte token for <0x{i}> is not set");
            }
        }

        // suffix list in lexicographical order of reversed strings
        let mut suffixes: Vec<String> = suffix_to_score.keys().cloned().collect();
        suffixes.push(String::new()); // empty suffix
        suffixes.sort_by(|a, b| {
            let rev_a: Vec<u8> = a.bytes().rev().collect();
            let rev_b: Vec<u8> = b.bytes().rev().collect();
            rev_a.cmp(&rev_b)
        });

        let mut suffix_to_id: HashMap<String, i32> = HashMap::new();
        let mut num_pieces: i32 = 0;

        for suffix in &suffixes {
            suffix_to_id.insert(suffix.clone(), num_pieces);
            if !suffix.is_empty() {
                let cpts = unicode::cpts_from_utf8(suffix.as_bytes());

                let mut remaining = String::new();
                for &cpt in &cpts[1..] {
                    remaining.push_str(&unicode::cpt_to_utf8(cpt));
                }

                let piece_code = ((cpts[0] as i64) << 32) | suffix_to_id[&remaining] as i64;
                to_suffix_id.insert(piece_code, num_pieces);

                // count number of pieces for this suffix
                let mut pieces_for_suffix: i32 = 1; // sentinel row
                for piece_length in (1..=cpts.len() as i32).rev() {
                    let mut piece = String::new();
                    for &cpt in &cpts[..piece_length as usize] {
                        piece.push_str(&unicode::cpt_to_utf8(cpt));
                    }
                    if suffix_to_score.contains_key(&piece) {
                        pieces_for_suffix += 1;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces += 1; // empty suffix contributes one piece (sentinel row)
            }
        }

        // build flattened table
        let mut table: Vec<[i32; 4]> = vec![[0; 4]; num_pieces as usize];
        let mut table_idx = 0usize;

        for suffix in &suffixes {
            let cpts = unicode::cpts_from_utf8(suffix.as_bytes());
            // prefixes of the suffix in decreasing order of length
            for piece_length in (1..=cpts.len()).rev() {
                let mut piece = String::new();
                for &cpt in &cpts[..piece_length] {
                    piece.push_str(&unicode::cpt_to_utf8(cpt));
                }

                let Some(&score) = suffix_to_score.get(&piece) else {
                    continue;
                };

                table[table_idx][TABLE_PIECE_LENGTH] = piece_length as i32;
                table[table_idx][TABLE_TOKEN_ID] = token_to_id.get(&piece).copied().unwrap_or(-1);
                table[table_idx][TABLE_SCORE] = if score.is_finite() {
                    (score as f64 * 1e4).round() as i32
                } else {
                    PLAMO2_INVALID_SCORE
                };
                table[table_idx][TABLE_PIECE_ID] = suffix_to_id[&piece];

                table_idx += 1;
            }

            // sentinel row
            table[table_idx][TABLE_PIECE_LENGTH] = 1;
            table[table_idx][TABLE_TOKEN_ID] = -1;
            table[table_idx][TABLE_SCORE] = PLAMO2_UNKNOWN_SCORE;
            table_idx += 1;
        }

        Plamo2Tokenizer {
            tokens,
            bytes,
            to_suffix_id,
            table,
            pre_segment,
        }
    }

    /// `llm_tokenizer_plamo2::encode` (abeada335, llama-vocab.cpp:1545-
    /// 1620): PLaMo-3 pre-segments the input — pass 1 fences `<|plamo:...|>`
    /// special-token bodies (≤ 64 code points, U+001C..U+001F treated as
    /// whitespace, terminated by `|>`), pass 2 cuts runs of ≥ 4 repeated
    /// characters or ≥ 2 spaces, and literal U+EE00 (the tokenizer's
    /// private-use boundary marker) splits segments without being emitted.
    /// PLaMo-2 (pre_segment = false) only drops a leading BOM.
    fn encode(&self, text: &[u8]) -> Vec<Token> {
        let unicode_data = unicode::cpts_from_utf8(text);
        if !self.pre_segment {
            // PLaMo-2: skip the first code point if it is a BOM (the
            // PLaMo-3 tokenizer keeps a leading U+FEFF in the input)
            let mut data = unicode_data;
            if !data.is_empty() && data[0] == 0xFEFF {
                data.remove(0);
            }
            return self.encode_cpts(&data);
        }

        let mut output: Vec<Token> = Vec::new();
        for (a, b) in Self::plamo3_segments(&unicode_data) {
            let tokens = self.encode_cpts(&unicode_data[a..b]);
            output.extend_from_slice(&tokens);
        }
        output
    }

    fn plamo3_segments(unicode_data: &[u32]) -> Vec<(usize, usize)> {
        plamo3_segments(unicode_data)
    }


    /// the pre-abelo335 encode body — the dynamic-programming table search
    fn encode_cpts(&self, unicode_data: &[u32]) -> Vec<Token> {
        if unicode_data.is_empty() {
            return Vec::new();
        }

        let data_len = unicode_data.len();

        // scores array (dynamic programming)
        let mut scores: Vec<i64> = vec![1i64 << 60; data_len + 1];
        scores[data_len] = 0;

        // path array to track best tokenization
        let mut path: Vec<[i32; 3]> = vec![[0; 3]; data_len + 1];

        let mut suffix_id: usize = 0;

        // process from end to beginning
        for i in (0..data_len).rev() {
            let c = unicode_data[i];

            // find next suffix ID
            for p in suffix_id..self.table.len() {
                let piece_code = ((c as i64) << 32) | self.table[p][TABLE_PIECE_ID] as i64;
                suffix_id = self.to_suffix_id.get(&piece_code).copied().unwrap_or(0) as usize;

                if suffix_id > 0 || self.table[p][TABLE_SCORE] == PLAMO2_UNKNOWN_SCORE {
                    break;
                }
            }

            // update best path
            let mut p = suffix_id;
            while p < self.table.len() {
                let score = self.table[p][TABLE_SCORE];
                if score > PLAMO2_INVALID_SCORE {
                    let piece_length = self.table[p][TABLE_PIECE_LENGTH] as usize;
                    let s = scores[(i + piece_length).min(data_len)].wrapping_sub(score as i64);

                    if s < scores[i] {
                        scores[i] = s;
                        path[i][PATH_TOKEN_LENGTH] = piece_length as i32;
                        path[i][PATH_TOKEN_ID] = self.table[p][TABLE_TOKEN_ID];
                        path[i][PATH_NUM_TOKENS] = path[i + piece_length][PATH_NUM_TOKENS] + 1;

                        if score == PLAMO2_UNKNOWN_SCORE {
                            // add UTF-8 byte count
                            path[i][PATH_NUM_TOKENS] +=
                                (c >= 0x80) as i32 + (c >= 0x800) as i32 + (c >= 0x10000) as i32;
                        }
                    }
                }

                if score == PLAMO2_UNKNOWN_SCORE {
                    break;
                }
                p += 1;
            }
        }

        // decode the best path
        let mut token_ids: Vec<Token> =
            Vec::with_capacity(path[0][PATH_NUM_TOKENS].max(0) as usize);

        let mut pos = 0usize;
        while pos < data_len {
            if path[pos][PATH_TOKEN_ID] >= 0 {
                token_ids.push(path[pos][PATH_TOKEN_ID]);
            } else {
                // fall back to byte tokens
                let c = unicode_data[pos];
                let s: usize =
                    1 + (c >= 0x80) as usize + (c >= 0x800) as usize + (c >= 0x10000) as usize;

                for i in 0..s {
                    let b: u8 = if s == 1 {
                        c as u8
                    } else if i == 0 {
                        ((0xF00 >> s) & 0xFF) as u8
                    } else {
                        0x80
                    };
                    token_ids
                        .push(self.bytes[(b | ((c >> ((s - i - 1) * 6)) & 0x3F) as u8) as usize]);
                }
            }

            debug_assert!(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH] as usize;
        }

        token_ids
    }
}

// ---------------------------------------------------------------------------
// pre-tokenizer string mapping (load()'s else-if chain)
// ---------------------------------------------------------------------------

fn map_pre_tokenizer(v: &mut Vocab) -> Result<(), String> {
    let pre = v.tokenizer_pre.clone();
    if pre.is_empty() {
        eprintln!("warn: missing pre-tokenizer type, using: 'default'");
        eprintln!("warn: ************************************");
        eprintln!("warn: GENERATION QUALITY WILL BE DEGRADED!");
        eprintln!("warn: CONSIDER REGENERATING THE MODEL");
        eprintln!("warn: ************************************");
        v.pre_type = PreType::Default;
    } else if pre == "default" {
        v.pre_type = PreType::Default;
    } else if pre == "minicpm5" {
        v.pre_type = PreType::Minicpm5;
        v.ignore_merges = true;
    } else if matches!(
        pre.as_str(),
        "llama3"
            | "llama-v3"
            | "llama-bpe"
            | "falcon3"
            | "falcon-h1"
            | "pixtral"
            | "midm-2.0"
            | "lfm2"
            | "jina-v5-nano"
    ) {
        v.pre_type = PreType::Llama3;
        v.ignore_merges = true;
        v.add_bos = true;
    } else if pre == "deepseek-llm" {
        v.pre_type = PreType::DeepseekLlm;
        v.clean_spaces = false;
    } else if pre == "deepseek-coder" {
        v.pre_type = PreType::DeepseekCoder;
        v.clean_spaces = false;
    } else if pre == "deepseek-v3" {
        v.pre_type = PreType::Deepseek3Llm;
        v.clean_spaces = false;
    } else if pre == "spark2_5" {
        v.pre_type = PreType::Spark25;
        v.clean_spaces = false;
    } else if pre == "youtu" {
        v.pre_type = PreType::Youtu;
        v.clean_spaces = false;
        v.ignore_merges = true;
    } else if pre == "falcon" {
        v.pre_type = PreType::Falcon;
    } else if pre == "mpt" {
        v.pre_type = PreType::Mpt;
    } else if pre == "starcoder" {
        v.pre_type = PreType::Starcoder;
    } else if matches!(
        pre.as_str(),
        "gpt-2"
            | "phi-2"
            | "jina-es"
            | "jina-de"
            | "gigachat"
            | "jina-v2-es"
            | "jina-v2-de"
            | "a.x-4.0"
            | "mellum"
            | "modern-bert"
    ) {
        v.pre_type = PreType::Gpt2;
    } else if pre == "jais-2" {
        v.pre_type = PreType::Jais2;
    } else if matches!(pre.as_str(), "gemma4" | "granite-embed-multi-311m") {
        v.pre_type = PreType::Gemma4;
        v.escape_whitespaces = true;
    } else if pre == "mmbert" {
        // (llama-vocab.cpp:2266-2269, a7b94df2c)
        v.pre_type = PreType::Mmbert;
        v.escape_whitespaces = true;
    } else if pre == "sarvam-moe" {
        v.pre_type = PreType::SarvamMoe;
        v.escape_whitespaces = true;
        v.clean_spaces = false;
    } else if matches!(pre.as_str(), "jina-v1-en" | "jina-v2-code" | "roberta-bpe") {
        v.pre_type = PreType::Gpt2;
        v.add_sep = true;
    } else if pre == "whitespace" {
        v.pre_type = PreType::Whitespace;
        v.normalizer_opts.lowercase = false;
    } else if pre == "refact" {
        v.pre_type = PreType::Refact;
    } else if pre == "command-r" {
        v.pre_type = PreType::CommandR;
        v.clean_spaces = false;
    } else if matches!(
        pre.as_str(),
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2"
    ) {
        v.pre_type = PreType::Qwen2;
        v.clean_spaces = false;
    } else if pre == "qwen35" {
        v.pre_type = PreType::Qwen35;
        v.clean_spaces = false;
    } else if pre == "stablelm2" {
        v.pre_type = PreType::Stablelm2;
    } else if pre == "olmo" {
        v.pre_type = PreType::Olmo;
    } else if pre == "dbrx" {
        v.pre_type = PreType::Dbrx;
    } else if pre == "smaug-bpe" {
        v.pre_type = PreType::Smaug;
    } else if pre == "poro-chat" {
        v.pre_type = PreType::Poro;
        v.clean_spaces = false;
    } else if pre == "chatglm-bpe" {
        v.pre_type = PreType::Chatglm4;
        v.special_bos_id = TOKEN_NULL;
    } else if matches!(pre.as_str(), "glm4" | "glm5") {
        // glm4/glm5 tokenize with ignore_merges (llama-vocab.cpp:2289-2297 —
        // the "glm5" arm landed with GLM-5.3-Flash, 649dcb103)
        v.pre_type = PreType::Chatglm4;
        v.special_bos_id = TOKEN_NULL;
        v.ignore_merges = true;
    } else if pre == "viking" {
        v.pre_type = PreType::Viking;
        v.clean_spaces = false;
    } else if pre == "jais" {
        v.pre_type = PreType::Jais;
    } else if pre == "tekken" {
        v.pre_type = PreType::Tekken;
        v.clean_spaces = false;
        v.ignore_merges = true;
        v.add_bos = true;
    } else if pre == "smollm" {
        v.pre_type = PreType::Smollm;
        v.clean_spaces = false;
    } else if pre == "codeshell" {
        v.pre_type = PreType::Codeshell;
    } else if pre == "bloom" {
        v.pre_type = PreType::Bloom;
    } else if pre == "gpt3-finnish" {
        v.pre_type = PreType::Gpt3Finnish;
    } else if pre == "exaone" {
        v.pre_type = PreType::Exaone;
    } else if pre == "exaone4" {
        v.pre_type = PreType::Gpt2;
    } else if pre == "exaone-moe" {
        v.pre_type = PreType::ExaoneMoe;
    } else if pre == "chameleon" {
        v.pre_type = PreType::Chameleon;
        v.add_bos = true;
        v.clean_spaces = false;
    } else if pre == "minerva-7b" {
        v.pre_type = PreType::Minerva;
    } else if pre == "megrez" {
        v.pre_type = PreType::Qwen2;
    } else if matches!(pre.as_str(), "gpt-4o" | "llama4" | "kanana2" | "talkie") {
        v.pre_type = PreType::Gpt4o;
        v.clean_spaces = false;
    } else if pre == "granite-embed-multi-97m" {
        v.pre_type = PreType::GraniteEmbMulti;
        v.clean_spaces = false;
        v.ignore_merges = true;
    } else if matches!(pre.as_str(), "tiny_aya" | "cohere2moe") {
        v.pre_type = PreType::TinyAya;
        v.clean_spaces = false;
    } else if pre == "superbpe" {
        v.pre_type = PreType::Superbpe;
        v.clean_spaces = false;
    } else if pre == "trillion" {
        v.pre_type = PreType::Trillion;
        v.clean_spaces = false;
    } else if pre == "granite-docling" {
        v.pre_type = PreType::GraniteDocling;
        v.clean_spaces = false;
    } else if matches!(pre.as_str(), "bailingmoe" | "bailingmoe2" | "llada-moe") {
        v.pre_type = PreType::Bailingmoe;
        v.clean_spaces = false;
    } else if pre == "seed-coder" {
        v.pre_type = PreType::SeedCoder;
        v.clean_spaces = false;
    } else if pre == "hunyuan" {
        v.pre_type = PreType::Hunyuan;
        v.clean_spaces = false;
    } else if pre == "hunyuan-dense" {
        v.pre_type = PreType::HunyuanDense;
        v.clean_spaces = false;
    } else if pre == "hy_v4" {
        v.pre_type = PreType::HyV4;
        v.clean_spaces = false;
    } else if pre == "joyai-llm" {
        v.pre_type = PreType::JoyaiLlm;
        v.clean_spaces = false;
    } else if pre == "kimi-k2" {
        v.pre_type = PreType::KimiK2;
        v.clean_spaces = false;
    } else if pre == "ufakzeka" {
        v.pre_type = PreType::Ufakzeka;
        v.clean_spaces = false;
    } else if pre == "grok-2" {
        v.pre_type = PreType::Grok2;
        v.clean_spaces = false;
    } else if pre == "afmoe" {
        v.pre_type = PreType::Afmoe;
        v.clean_spaces = false;
    } else if pre == "laguna" {
        v.pre_type = PreType::Laguna;
        v.clean_spaces = false;
    } else if pre == "minimax-m2" {
        v.pre_type = PreType::MinimaxM2;
        v.clean_spaces = false;
    } else if pre == "solar-open" {
        v.pre_type = PreType::SolarOpen;
        v.clean_spaces = false;
    } else if pre == "mellum2" {
        v.pre_type = PreType::Mellum2;
    } else if pre == "k2-horizon" {
        // 462524043 (llama-vocab.cpp:2544-2547)
        v.pre_type = PreType::K2Horizon;
        v.clean_spaces = false;
    } else {
        return Err(format!("unknown pre-tokenizer type: '{pre}'"));
    }
    Ok(())
}

/// Pre-tokenizer regex table — port of the `llm_tokenizer_bpe` constructor
/// switch in llama-vocab.cpp (generated by vocab_pre_gen.py; do not hand-edit).
fn bpe_pre_regexes(pre: PreType) -> (Vec<String>, bool) {
    let exprs: &[&str] = match pre {
        PreType::Llama3 => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Jais2 => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
        ],
        PreType::Dbrx | PreType::Smaug => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::DeepseekLlm => &[
            "[\r\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            "\\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}+",
        ],
        PreType::Deepseek3Llm | PreType::HunyuanDense | PreType::JoyaiLlm | PreType::HyV4 => &[
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Spark25 => &[
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
            "\\p{N}",
        ],
        PreType::Youtu => &[
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::DeepseekCoder => &[
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}",
        ],
        PreType::Falcon => &[
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "[0-9][0-9][0-9]",
        ],
        PreType::Starcoder | PreType::Refact | PreType::CommandR | PreType::Smollm | PreType::Codeshell | PreType::Exaone | PreType::Minerva | PreType::Mellum2 => &[
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        ],
        PreType::Gpt2 | PreType::Mpt | PreType::Olmo | PreType::Jais | PreType::Trillion | PreType::GraniteDocling => &[
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        ],
        PreType::Stablelm2 | PreType::Qwen2 | PreType::Hunyuan | PreType::SolarOpen => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Qwen35 => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Poro | PreType::Bloom | PreType::Gpt3Finnish => &[
            " ?[^(\\s|.,!?…。，、।۔،)]+",
        ],
        PreType::Chatglm4 => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Viking => &[
            " ?[^(\\s|.,!?…。，、।۔،)]+",
            "\\p{N}",
        ],
        PreType::Tekken => &[
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Chameleon => &[
            "<sentinel:[0-9]+>",
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
            "([\\t\\n]|    |  )",
            "\\p{N}",
            "[\\p{P}!-/:-@\\[-`{-~]",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        ],
        PreType::Gpt4o | PreType::MinimaxM2 => &[
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::GraniteEmbMulti => &[
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::TinyAya => &[
            "\\d{1,3}(?=(?:\\d{3})*\\b)",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::KimiK2 => &[
            "\\p{Han}+",
        ],
        // 462524043 (llama-vocab.cpp:561-564) — served by the hand-written
        // unicode splitter (regex_split_custom_k2_horizon)
        PreType::K2Horizon => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Superbpe => &[
            "\\p{N}+",
            "(?=(\\d{3})+(?!\\d))",
        ],
        PreType::Bailingmoe => &[
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        ],
        PreType::SeedCoder => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Ufakzeka => &[
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Grok2 => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Afmoe => &[
            "\\p{AFMoE_digits}",
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Laguna => &[
            "[^\\n]+|[\\n]+",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::ExaoneMoe => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        ],
        PreType::Gemma4 => &[
            "[^\\n]+|[\\n]+",
        ],
        // same as Gemma4, the words are split in tokenize()
        // (llama-vocab.cpp:546-551, a7b94df2c)
        PreType::Mmbert => &[
            "[^\\n]+|[\\n]+",
        ],
        PreType::SarvamMoe => &[
            "[^\\n]+|[\\n]+",
        ],
        PreType::Minicpm5 => &[
            "\\p{N}{1,3}",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Whitespace => &[
            "\\S+",
        ],
        // default regex for BPE tokenization pre-processing
        _ => &[
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]",
        ],
    };
    // byte_encode = false (SPM-style BPE on raw UTF-8)
    let byte_encode = !matches!(
        pre,
        PreType::Gemma4 | PreType::SarvamMoe | PreType::Whitespace | PreType::Mmbert
    );
    (exprs.iter().map(|s| s.to_string()).collect(), byte_encode)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab GGUFs mirrored from the llama.cpp test-model corpus (MIT) so the
    /// suite runs on clean checkouts; the original tree is used as fallback.
    const MODELS_DIR: &str = "/home/jeffrey/llm/llama.cpp-pinned/models";
    const FIXTURES_DIR: &str = "tests/tokenizer_fixtures";

    /// Resolve a vocab GGUF by name: vendored copy first, original tree second.
    fn vocab_model_path(name: &str) -> String {
        let vendored = format!("{FIXTURES_DIR}/vocab/{name}");
        if std::fs::metadata(&vendored).is_ok() {
            return vendored;
        }
        format!("{MODELS_DIR}/{name}")
    }

    /// Resolve a fixture "# model:" entry: bare names go through the vendored
    /// vocab dir; absolute paths are kept as-is (machine-local models).
    fn resolve_fixture_model(model: &str) -> String {
        if model.starts_with('/') {
            model.to_string()
        } else {
            vocab_model_path(model)
        }
    }

    fn load_vocab(path: &str) -> Vocab {
        let g = ggml::Gguf::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        Vocab::load(&g).unwrap_or_else(|e| panic!("load vocab {path}: {e}"))
    }

    struct FixtureCase {
        text: String,
        expected: Vec<Token>,
        round_trip: bool,
    }

    /// fixture format (see tests/tokenizer_fixtures/gen_fixtures.py):
    ///   # comment lines (model path on the first one)
    ///   <hex of utf-8 sentence>
    ///   <space-separated reference token ids>
    ///   RT|NORT
    fn load_fixture(name: &str) -> (String, Vec<FixtureCase>) {
        let content = std::fs::read_to_string(format!("{FIXTURES_DIR}/{name}"))
            .unwrap_or_else(|e| panic!("read fixture {name}: {e}"));
        let mut model = String::new();
        let mut cases = Vec::new();
        let mut lines = content.lines().peekable();
        while let Some(line) = lines.next() {
            if line.starts_with('#') {
                if let Some(m) = line.strip_prefix("# model: ") {
                    model = m.to_string();
                }
                continue;
            }
            let hex = line.trim();
            let ids_line = lines.next().expect("ids line").trim();
            let rt_line = lines.next().expect("rt line").trim();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
                .collect::<Vec<u8>>();
            cases.push(FixtureCase {
                text: String::from_utf8(bytes).expect("utf8 fixture"),
                expected: ids_line
                    .split_whitespace()
                    .map(|t| t.parse().expect("token id"))
                    .collect(),
                round_trip: rt_line == "RT",
            });
        }
        (model, cases)
    }

    /// token ids must match the reference llama-tokenize exactly; reference
    /// defaults: add_special == model add_bos, parse_special == true
    fn check_fixture(fixture: &str) {
        let (model, cases) = load_fixture(fixture);
        let model = resolve_fixture_model(&model);
        if std::fs::metadata(&model).is_err() {
            eprintln!("skipping {fixture}: model {model} not present");
            return;
        }
        let v = load_vocab(&model);
        let add_special = v.get_add_bos();
        let mut n = 0;
        for case in &cases {
            let got = v.tokenize(&case.text, add_special, true);
            assert_eq!(
                got, case.expected,
                "{fixture}: token mismatch for {:#?}:\n  got      {got:?}\n  expected {:?}",
                case.text, case.expected
            );
            if case.round_trip {
                let detok = v.detokenize(&got, false);
                assert_eq!(
                    detok, case.text,
                    "{fixture}: round-trip mismatch for {:#?}: got {:#?}",
                    case.text, detok
                );
            }
            n += 1;
        }
        eprintln!("{fixture}: {n}/{n} sentences match the reference exactly");
    }

    #[test]
    fn test_fixture_qwen2() {
        check_fixture("qwen2.txt");
    }

    #[test]
    fn test_fixture_llama_bpe() {
        check_fixture("llama-bpe.txt");
    }

    #[test]
    fn test_fixture_llama_spm() {
        check_fixture("llama-spm.txt");
    }

    #[test]
    fn test_fixture_qwen25_real_model() {
        check_fixture("qwen25-real.txt");
    }

    // extended coverage: other tokenizer types / pre-tokenizer regex paths
    #[test]
    fn test_fixture_bert_bge_wpm() {
        check_fixture("bert-bge.txt");
    }

    #[test]
    fn test_fixture_gpt2() {
        check_fixture("gpt-2.txt");
    }

    #[test]
    fn test_fixture_deepseek_llm() {
        check_fixture("deepseek-llm.txt");
    }

    #[test]
    fn test_fixture_qwen35() {
        check_fixture("qwen35.txt");
    }

    #[test]
    fn test_fixture_falcon() {
        check_fixture("falcon.txt");
    }

    #[test]
    fn test_fixture_command_r() {
        check_fixture("command-r.txt");
    }

    #[test]
    fn test_fixture_phi3_spm() {
        check_fixture("phi-3.txt");
    }

    #[test]
    fn test_fixture_gemma4() {
        check_fixture("gemma-4.txt");
    }

    #[test]
    fn test_load_qwen2_vocab_basic() {
        let v = load_vocab(&vocab_model_path("ggml-vocab-qwen2.gguf"));
        assert_eq!(v.get_type(), VocabType::Bpe);
        assert_eq!(v.tokenizer_pre, "qwen2");
        assert_eq!(v.pre_type, PreType::Qwen2);
        assert_eq!(v.n_tokens(), 151936);
        assert_eq!(v.token_bos(), 151643);
        assert_eq!(v.token_eos(), 151643);
        assert!(!v.get_add_bos());
        assert!(!v.add_space_prefix);
        assert!(!v.escape_whitespaces);
        assert!(!v.clean_spaces);
        // newline token = token for "\n" = "Ċ" = id 198
        assert_eq!(v.token_nl(), 198);
        assert_eq!(v.token_to_piece(9707), "Hello");
        // eog: eos (151643) + <end_of_turn>? llama-bpe? qwen2 has <|im_end|>
        assert!(v.is_eog(151643));
    }

    #[test]
    fn test_load_llama_spm_vocab_basic() {
        let v = load_vocab(&vocab_model_path("ggml-vocab-llama-spm.gguf"));
        assert_eq!(v.get_type(), VocabType::Spm);
        assert_eq!(v.n_tokens(), 32000);
        assert_eq!(v.token_bos(), 1);
        assert_eq!(v.token_eos(), 2);
        assert_eq!(v.token_unk(), 0);
        assert!(v.add_bos);
        assert!(!v.add_eos);
        assert!(v.add_space_prefix);
        assert_eq!(v.byte_to_token(b'\n').unwrap(), 13);
        assert_eq!(v.token_nl(), 13);
        // SPM linefeed via byte token <0x0A>
        let toks = v.tokenize("a\nb", false, false);
        assert_eq!(toks[1], 13);
    }

    #[test]
    fn test_load_llama_bpe_vocab_basic() {
        let v = load_vocab(&vocab_model_path("ggml-vocab-llama-bpe.gguf"));
        assert_eq!(v.get_type(), VocabType::Bpe);
        assert_eq!(v.pre_type, PreType::Llama3);
        assert!(v.ignore_merges);
        assert!(v.add_bos);
        assert_eq!(v.token_bos(), 128000);
        assert_eq!(v.token_eos(), 128001);
        // llama3: digit runs are split into groups of 3
        let toks = v.tokenize("1234567", false, false);
        let pieces: Vec<&str> = toks.iter().map(|&t| v.token_to_piece(t)).collect();
        assert_eq!(pieces, vec!["123", "456", "7"]);
    }

    /// Byte-level entry: identical to `tokenize` for UTF-8, and handles
    /// invalid bytes through byte-fallback tiles like the reference
    /// (`a\xffb` -> [64, 5691, 65] for the qwen2 vocab, verified against the
    /// reference llama-tokenize by agent Y).
    #[test]
    fn test_tokenize_bytes_entry() {
        let path = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
        let Ok(gguf) = Gguf::open(path) else {
            eprintln!("skipping: model not present");
            return;
        };
        let v = Vocab::load(&gguf).unwrap();
        let s = "The capital of France is";
        assert_eq!(
            v.tokenize_bytes(s.as_bytes(), true, true),
            v.tokenize(s, true, true)
        );
        let invalid = b"a\xffb";
        let ids = v.tokenize_bytes(invalid, false, false);
        assert_eq!(
            ids,
            vec![64, 5691, 65],
            "invalid byte -> U+FFFD token (reference-verified)"
        );
    }

    /// `tokenizer_st_partition` with a *repeated* special token: the right
    /// piece of each match becomes the next "source" and only the final
    /// remainder is kept (llama-vocab.cpp:3332-3347 erases the source fragment
    /// on every match). The port used to push every right piece and keep
    /// scanning it, duplicating the tail: "<|endoftext|>A<|endoftext|>B" gave
    /// 7 tokens instead of the reference's 4. Reference answer verified against
    /// `llama-tokenize -m ggml-vocab-qwen2.gguf`.
    #[test]
    fn test_st_partition_repeated_special() {
        let v = load_vocab(&vocab_model_path("ggml-vocab-qwen2.gguf"));
        assert_eq!(
            v.tokenize("<|endoftext|>A<|endoftext|>B", false, true),
            vec![151643, 32, 151643, 33]
        );
    }

    /// The unigram (SentencePiece) normalizer path for input that is not valid
    /// UTF-8: the reference consumes one byte and substitutes the literal
    /// U+FFFD bytes (llama-vocab.cpp:1187-1192). Our port kept a `usize::MAX`
    /// sentinel for the replacement offset and then sliced `input` with it,
    /// which panicked (`range start index 18446744073709551615 out of range`).
    /// Expected ids are from the reference llama-tokenize on the same vocab.
    #[test]
    fn test_invalid_utf8_ugm_normalization() {
        let path = vocab_model_path("ggml-vocab-llama-spm.gguf");
        if std::fs::metadata(&path).is_err() {
            eprintln!("skipping: {path} not present");
            return;
        }
        let v = load_vocab(&path);
        assert_eq!(v.get_type(), VocabType::Spm);
        for (input, expected) in [
            (&b"a\xffb"[..], &[263, 258, 101][..]),
            (&b"\xff"[..], &[29871, 258][..]),
            (&b"a\xc3(b"[..], &[263, 198, 43, 29890][..]),
            (&b"\xe4\xbd\xa0\xff"[..], &[29871, 30919, 258][..]),
            // truncated 4-byte sequence: F0 9F then end of input
            (&b"a\xf0\x9f"[..], &[263, 243, 162][..]),
        ] {
            assert_eq!(
                v.tokenize_bytes(input, false, false),
                expected,
                "ugm invalid-utf8 {input:02x?}"
            );
        }
    }

    #[test]
    fn test_special_token_partition() {
        let v = load_vocab(&vocab_model_path("ggml-vocab-qwen2.gguf"));
        // parse_special=true splits the special token out
        let toks = v.tokenize("a<|endoftext|>b", false, true);
        let pieces: Vec<String> = toks
            .iter()
            .map(|&t| v.token_to_piece(t).to_string())
            .collect();
        assert_eq!(pieces, vec!["a", "<|endoftext|>", "b"]);
        // parse_special=false treats it as plain text
        let toks2 = v.tokenize("a<|endoftext|>b", false, false);
        assert_ne!(toks, toks2);
    }
}
