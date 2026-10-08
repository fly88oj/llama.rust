//! Batch sanitising + splitting — port of src/llama-batch.h / llama-batch.cpp
//! (`llama_ubatch`, `llama_batch_ext`, `llama_batch_compat`,
//! `llama_batch_allocr`) plus the `llama_batch` container of llama.h
//! (`llama_batch_get_one` / `llama_batch_init`, llama-batch.cpp:931-987).
//!
//! Since fc343a84b (`llama: add llama_batch_ext`) the internal batch is
//! `llama_batch_ext` — tokens carrying an optional id, optional F32 embedding
//! rows, per-section positions and a sequence-id *set* — and
//! `llama_batch_allocr::init` consumes it (llama-batch.cpp:26-391). The old
//! caller-facing `llama_batch` is translated by `llama_batch_compat`
//! (llama-batch.cpp:1189-1233), which also inherits the auto-generated
//! positions (each sequence continues at `memory->seq_pos_max + 1`).
//! `llama_ubatch` is what the graph and the KV cache consume: tokens with one
//! or more sequence ids each, positions, the output flags, and — since
//! 4453b535f — `batch_idxs`, each token's index in the logical batch.

/// `LLAMA_MAX_SEQ` (llama.h) — the port's KV cells carry a 64-bit sequence
/// bitmask (kv_cache.rs `KvCell::seq`), which is this constant.
pub const LLAMA_MAX_SEQ: usize = 64;

/// `GGML_MROPE_SECTIONS` (ggml.h:256) — the width of a batch-ext token's
/// position array.
pub const GGML_MROPE_SECTIONS: usize = 4;

/// `LLAMA_TOKEN_NULL` (llama.h) — "no token id", the batch-ext token's
/// `id = LLAMA_TOKEN_NULL` initialiser.
pub const LLAMA_TOKEN_NULL: i32 = -1;

/// `enum llama_decision_order` (llama-ext.h:107-116, a7b94df2c) — marks the
/// entries that a joint decision head (clef) reads, the default is 0. A run
/// of entries with the same value is one span, spans must be separated by
/// entries with value 0; an option belongs to the last question before it.
/// The embeddings output has one value per entry: row i is the score of
/// option i. See <https://github.com/ggml-org/llama.cpp/pull/29831>.
pub const DECISION_ORDER_NONE: i32 = 0; // not read by the head
pub const DECISION_ORDER_QUESTION_NOUL: i32 = 1; // text of a question
pub const DECISION_ORDER_QUESTION_CHOICE: i32 = 2;
pub const DECISION_ORDER_QUESTION_SCORE: i32 = 3;
pub const DECISION_ORDER_OPTION: i32 = 4; // text of an option

/// `enum llama_context_type` (llama.h:224-227).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LlamaContextType {
    #[default]
    Default,
    /// `LLAMA_CONTEXT_TYPE_MTP` — the draft head re-feeds the target's
    /// hidden state, so its batches carry `n_embd_out`-wide rows.
    Mtp,
}

/// `struct llama_batch` (llama.h) — the caller-facing batch. The C allows the
/// token/pos/seq_id/logits pointers to be null ("auto-generate"); the port
/// models that with `Option`.
#[derive(Clone, Debug, Default)]
pub struct LlamaBatch {
    /// `batch.token` — always present in the port (no embedding inputs yet).
    pub token: Vec<i32>,
    /// `batch.pos` — `None` = auto (llama-batch.cpp:90-118)
    pub pos: Option<Vec<i32>>,
    /// `batch.n_seq_id` + `batch.seq_id` — `None` = every token in seq 0
    pub seq_id: Option<Vec<Vec<i32>>>,
    /// `batch.logits` — `None` = output only the last token (:120-146)
    pub logits: Option<Vec<bool>>,
    /// `batch.embd` — F32 rows `[n_embd * n_tokens]`, carried by the MTP
    /// draft hook batches (the token ids ride `token` *and* the previous
    /// position's `h_nextn` rows ride this; speculative.cpp:1390-1394 mallocs
    /// both). Row-major `[n_tokens][n_embd]`.
    pub embd: Option<Vec<f32>>,
}

impl LlamaBatch {
    /// `llama_batch_get_one` (llama-batch.cpp:931-943): tokens only, every
    /// other field auto-generated.
    pub fn get_one(token: Vec<i32>) -> Self {
        LlamaBatch {
            token,
            ..Default::default()
        }
    }

    /// `common_batch_add` (common/common.cpp) — append one token.
    pub fn add(&mut self, token: i32, pos: i32, seq_ids: &[i32], logits: bool) {
        self.token.push(token);
        self.pos.get_or_insert_with(Vec::new).push(pos);
        self.seq_id
            .get_or_insert_with(Vec::new)
            .push(seq_ids.to_vec());
        self.logits.get_or_insert_with(Vec::new).push(logits);
    }

    /// `common_batch_clear`
    pub fn clear(&mut self) {
        self.token.clear();
        self.pos = None;
        self.seq_id = None;
        self.logits = None;
        self.embd = None;
    }
}

/// `struct llama_ubatch` (llama-batch.h:15-69) with owned storage.
#[derive(Clone, Debug, Default)]
pub struct LlamaUbatch {
    /// `b_equal_seqs` — every sequence set has `n_seq_tokens` tokens
    pub equal_seqs: bool,
    /// total tokens (`n_seq_tokens * n_seqs`)
    pub n_tokens: u32,
    /// tokens per sequence set
    pub n_seq_tokens: u32,
    /// sequence sets in the ubatch
    pub n_seqs: u32,
    /// unique sequence ids in the ubatch
    pub n_seqs_unq: u32,
    /// `n_pos` — position inputs per token (`hparams.n_pos_per_embd()`)
    pub n_pos: u32,

    pub token: Vec<i32>,
    /// [n_tokens * n_pos]
    pub pos: Vec<i32>,
    /// the batch's F32 embedding rows, split per token by `ubatch_add`
    /// (`batch.embd` of llama-batch.h:44 — only the MTP draft hook batches
    /// carry them)
    pub embd: Vec<f32>,
    /// [n_tokens]
    pub n_seq_id: Vec<i32>,
    /// [n_tokens] — one sequence-id list per token
    pub seq_id: Vec<Vec<i32>>,
    /// [n_seqs_unq]
    pub seq_id_unq: Vec<i32>,
    /// [LLAMA_MAX_SEQ], -1 = absent
    pub seq_idx: Vec<i32>,
    /// [n_tokens]
    pub output: Vec<bool>,
    /// `decision_order` (llama-batch.h:55, a7b94df2c) — [n_tokens], empty
    /// (the C's NULL) if no entry has one. Marks the entries a joint
    /// decision head (clef) reads; a run of entries with the same value is
    /// one span, spans must be separated by entries with value 0. See
    /// [`DECISION_ORDER_*`] and `llama_batch_ext_set_decision_order`.
    pub decision_order: Vec<i32>,
    /// `data_t::batch_idxs` (llama-batch.h:65, 4453b535f) — the original
    /// batch index of each token. The logits follow the output order, but
    /// the layer-input/nextn extractions index the *whole* batch, so their
    /// rows are permuted back through this map at the end of the decode
    /// (`output_reorder`, llama-context.cpp:2367-2386).
    pub batch_idxs: Vec<i32>,
}

impl LlamaUbatch {
    pub fn empty() -> Self {
        LlamaUbatch::default()
    }

    pub fn is_empty(&self) -> bool {
        self.n_tokens == 0
    }

    /// `llama_ubatch::equal_seqs()` (llama-batch.h:16)
    pub fn equal_seqs(&self) -> bool {
        self.equal_seqs
    }

    /// `llama_ubatch::is_pos_2d()` (llama-batch.h:25-28): M-RoPE with 3+
    /// position inputs per token.
    pub fn is_pos_2d(&self) -> bool {
        self.n_pos >= 3
    }

    /// number of output-flagged tokens in the ubatch (llama-context.cpp:
    /// 1867-1880 counts them into `n_outputs` before the graph is built)
    pub fn n_outputs(&self) -> usize {
        self.output.iter().filter(|&&o| o).count()
    }
}

/// Position lookup of the memory module the batch is validated against
/// (`llama_memory_i::seq_pos_min/max`, llama-memory.h:50-59). The port's KV
/// cache implements it from its cells; `None` (no memory — e.g. a plain
/// `split_*` geometry test) skips every check that reads it
/// (llama-batch.cpp:96-101).
pub trait BatchMemory {
    fn seq_pos_min(&self, seq_id: i32) -> i32;
    fn seq_pos_max(&self, seq_id: i32) -> i32;
}

/// `llama_batch_ext_select_n_embd_inp` (llama-batch.cpp:1024-1034): the embd
/// row width a context's batches carry — the target hidden size for the MTP
/// hook batches, the fused target features at the encoder input width for
/// DFlash, the token embeddings otherwise.
pub fn batch_ext_select_n_embd_inp(
    ctx_type: LlamaContextType,
    arch: crate::arch::LlmArch,
    hparams: &crate::hparams::LlamaHparams,
) -> usize {
    if ctx_type == LlamaContextType::Mtp {
        return hparams.n_embd_out() as usize;
    }
    if arch == crate::arch::LlmArch::DFLASH {
        return hparams.n_embd_inp_enc() as usize;
    }
    hparams.n_embd_inp() as usize
}

/// `llama_batch_ext::token` (llama-batch.h:95-102). The C's `seq_ids` is an
/// `std::unordered_set`; the port keeps insertion order (a superset of the C's
/// contract — the set semantics are the dedup, the order only fixes which id
/// `sids[0]`-style reads see, and the port's callers feed id order).
#[derive(Clone, Debug)]
pub struct BatchExtToken {
    /// `LLAMA_TOKEN_NULL` = no id (an embedding-only entry)
    pub id: i32,
    /// whether `embd_off` is set
    pub has_embd: bool,
    /// index offset in the ext's `embd` array
    pub embd_off: usize,
    pub output: bool,
    /// `decision_order` (llama-batch.h:101, a7b94df2c) — 0 by default; see
    /// [`DECISION_ORDER_*`] and [`LlamaBatchExt::set_decision_order`].
    pub decision_order: i32,
    /// the sequence-id set of the entry
    pub seq_ids: Vec<i32>,
    /// `std::array<llama_pos, GGML_MROPE_SECTIONS> = {0, 0, 0, 0}`
    pub pos: [i32; GGML_MROPE_SECTIONS],
}

impl Default for BatchExtToken {
    /// the C's member initialisers (llama-batch.h:95-102): `id =
    /// LLAMA_TOKEN_NULL`, everything else zero/false/empty
    fn default() -> Self {
        BatchExtToken {
            id: LLAMA_TOKEN_NULL,
            has_embd: false,
            embd_off: 0,
            output: false,
            decision_order: 0,
            seq_ids: Vec::new(),
            pos: [0; GGML_MROPE_SECTIONS],
        }
    }
}

/// `struct llama_embd` (llama.h) — a borrowed 2-D block of F32 rows.
#[derive(Clone, Copy)]
pub struct LlamaEmbd<'a> {
    pub data: &'a [f32],
    pub n_rows: usize,
    pub n_embd: usize,
}

/// `struct llama_batch_ext` (llama-batch.h:82-128, fc343a84b) — the internal
/// batch: one entry per token carrying an optional id and/or F32 embedding
/// rows, per-section positions and a sequence-id set.
pub struct LlamaBatchExt<'a> {
    /// max number of tokens that can be stored in the batch
    pub n_tokens_max: usize,
    /// decoder embd row width
    pub n_embd_inp: usize,
    /// encoder embd row width (eagle3/dflash extracted features)
    pub n_embd_inp_enc: usize,
    pub n_seq_max: i32,
    /// memory for position inference (`llama_memory_i * mem`)
    pub mem: Option<&'a dyn BatchMemory>,
    /// max token ID that we accept
    pub n_vocab: i32,
    pub n_pos_per_embd: usize,

    /// actual embd row width of this batch, set by the first
    /// `set_token_embd()`; must be either `n_embd_inp` or `n_embd_inp_enc` —
    /// encode/decode verify it against the graph input
    pub n_embd: usize,

    pub tokens: Vec<BatchExtToken>,
    pub embd: Vec<f32>,
}

impl<'a> LlamaBatchExt<'a> {
    /// the standalone constructor (llama-batch.cpp:1064-1081) — "build without
    /// a llama_context, used by tests".
    pub fn new(
        n_tokens_max: usize,
        n_embd_inp: usize,
        n_embd_inp_enc: usize,
        n_seq_max: i32,
        mem: Option<&'a dyn BatchMemory>,
        n_vocab: i32,
        n_pos_per_embd: usize,
    ) -> Self {
        LlamaBatchExt {
            n_tokens_max,
            n_embd_inp,
            n_embd_inp_enc,
            n_seq_max,
            mem,
            n_vocab,
            n_pos_per_embd,
            n_embd: 0,
            tokens: Vec::new(),
            embd: Vec::new(),
        }
    }

    /// `clear` (llama-batch.cpp:1083-1087)
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.embd.clear();
        self.n_embd = 0;
    }

    /// `add_token` (llama-batch.cpp:1089-1106) — add an entry with an
    /// undefined position; the caller must set it via `set_token_pos`.
    pub fn add_token(&mut self, seq_id: i32) -> i32 {
        if self.tokens.len() >= self.n_tokens_max {
            return -1; // size limit reached
        }
        if seq_id < 0 || seq_id >= self.n_seq_max {
            return -3; // invalid sequence id
        }

        let mut t = BatchExtToken::default();
        t.seq_ids.push(seq_id);
        self.tokens.push(t);

        (self.tokens.len() - 1) as i32
    }

    /// `add_seq` (llama-batch.cpp:1108-1122)
    pub fn add_seq(&mut self, idx: i32, seq_id: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if seq_id < 0 || seq_id >= self.n_seq_max {
            return false;
        }
        let t = &mut self.tokens[idx as usize];
        if !t.seq_ids.contains(&seq_id) {
            t.seq_ids.push(seq_id);
        }
        true
    }

    /// `set_token_id` (llama-batch.cpp:1124-1135)
    pub fn set_token_id(&mut self, idx: i32, id: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if id < 0 || id >= self.n_vocab {
            return false;
        }
        self.tokens[idx as usize].id = id;
        true
    }

    /// `set_token_embd` (llama-batch.cpp:1137-1178)
    pub fn set_token_embd(&mut self, idx: i32, embd_in: LlamaEmbd<'_>) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if embd_in.data.is_empty() {
            return false;
        }

        let n_total = embd_in.n_rows * embd_in.n_embd;
        if n_total != embd_in.data.len() {
            return false;
        }
        if self.n_embd == 0 {
            if n_total != self.n_embd_inp && n_total != self.n_embd_inp_enc {
                crate::impl_log::log_internal(
                    crate::impl_log::LogLevel::Error,
                    format_args!(
                        "set_token_embd: embedding size mismatch, got {} rows x {} = {}, expected \
                         {} or {}",
                        embd_in.n_rows, embd_in.n_embd, n_total, self.n_embd_inp, self.n_embd_inp_enc
                    ),
                );
                return false;
            }
            self.n_embd = n_total;
        } else if n_total != self.n_embd {
            crate::impl_log::log_internal(
                crate::impl_log::LogLevel::Error,
                format_args!(
                    "set_token_embd: embedding size mismatch, got {} rows x {} = {}, expected {}",
                    embd_in.n_rows, embd_in.n_embd, n_total, self.n_embd
                ),
            );
            return false;
        }

        let t = &mut self.tokens[idx as usize];
        if t.has_embd {
            crate::impl_log::log_internal(
                crate::impl_log::LogLevel::Error,
                format_args!("set_token_embd: embedding for token {idx} is already set"),
            );
            return false;
        }

        t.has_embd = true;
        t.embd_off = self.embd.len();
        self.embd.extend_from_slice(embd_in.data);

        true
    }

    /// `set_token_pos` (llama-batch.cpp:1180-1195): a token entry reads one
    /// position, an embedding entry `n_pos_per_embd` (one per RoPE section).
    pub fn set_token_pos(&mut self, idx: i32, pos_in: &[i32]) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        if pos_in.is_empty() {
            return false;
        }

        let t = &mut self.tokens[idx as usize];

        let n_pos = if t.id != LLAMA_TOKEN_NULL {
            1
        } else {
            self.n_pos_per_embd
        };
        for i in 0..n_pos {
            t.pos[i] = pos_in[i];
        }

        true
    }

    /// `set_output` (llama-batch.cpp:1197-1203)
    pub fn set_output(&mut self, idx: i32, output_last: bool) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        self.tokens[idx as usize].output = output_last;
        true
    }

    /// `llama_batch_ext::set_decision_order` (llama-batch.cpp:1194-1200,
    /// a7b94df2c) — behind `llama_batch_ext_set_decision_order`
    /// (llama-ext.h:118).
    pub fn set_decision_order(&mut self, idx: i32, order: i32) -> bool {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return false;
        }
        self.tokens[idx as usize].decision_order = order;
        true
    }
}

/// `llama_batch_compat::init` (llama-batch.cpp:1216-1233 + the static
/// `init`, :1236-1305, fc343a84b) — fill an existing `llama_batch_ext` from a
/// `llama_batch` (the old API). Auto-generates the positions locally when
/// `batch_inp.pos` is null, continuing each sequence from its
/// `mem->seq_pos_max + 1`.
pub fn batch_compat_init(dst: &mut LlamaBatchExt<'_>, batch_inp: &LlamaBatch, n_embd_row: usize) {
    let n_embd_row = if n_embd_row == 0 {
        dst.n_embd_inp
    } else {
        n_embd_row
    };

    // a batch can carry both, for example the MTP hook batches. The port's
    // container is token-indexed (placeholder ids on the embd-only batches),
    // so `has_token` is always true and the embd flag only guards the row
    // copy below — the C's `has_token = batch_inp.token != nullptr` arm is
    // folded into it.
    let has_token = !batch_inp.token.is_empty();
    #[allow(unused_variables)]
    let has_embd = batch_inp.embd.is_some();

    // auto-generates positions locally when batch_inp.pos is null, continuing
    // from memory (:1252-1256)
    let mut pos_next: Vec<i32> = vec![0; dst.n_seq_max.max(0) as usize];
    for s in 0..pos_next.len() as i32 {
        pos_next[s as usize] = dst
            .mem
            .map(|m| m.seq_pos_max(s) + 1) // assume next pos
            .unwrap_or(0);
    }

    for i in 0..batch_inp.n_tokens() {
        let n_sid = batch_inp
            .n_seq_id()
            .map(|n| n[i])
            .unwrap_or(1); // default_n_seq_id
        let default_seq_id: Vec<i32> = vec![0];
        let sids: &[i32] = batch_inp
            .seq_id
            .as_ref()
            .map(|s| s[i].as_slice())
            .unwrap_or(default_seq_id.as_slice());

        let mut t = BatchExtToken::default();

        // seq_ids
        for &s in sids.iter().take(n_sid.max(0) as usize) {
            if !t.seq_ids.contains(&s) {
                t.seq_ids.push(s);
            }
        }

        // position(s)
        match &batch_inp.pos {
            Some(pos) => {
                if has_token {
                    // token batch: one position per token
                    t.pos[0] = pos[i];
                } else {
                    // embedding batch (M-RoPE): section-major layout
                    // pos[j*n_tokens + i]
                    for j in 0..dst.n_pos_per_embd {
                        t.pos[j] = pos[j * batch_inp.n_tokens() + i];
                    }
                }
            }
            None => {
                // auto-generate position from the first seq_id
                t.pos[0] = pos_next[sids[0] as usize];
                pos_next[sids[0] as usize] += 1;
            }
        }

        // token id and/or embeddings
        if has_token {
            t.id = batch_inp.token[i];
        }

        if let Some(e) = batch_inp.embd.as_ref() {
            t.has_embd = true;
            t.embd_off = dst.embd.len();
            let src = &e[i * n_embd_row..(i + 1) * n_embd_row];
            dst.embd.extend_from_slice(src);
            dst.n_embd = n_embd_row;
        }

        // output flag — if no logits array is given, default to only the last
        // token being an output
        t.output = match &batch_inp.logits {
            Some(l) => l[i],
            None => i == batch_inp.n_tokens() - 1,
        };

        dst.tokens.push(t);
    }
}

impl LlamaBatch {
    /// `batch.n_tokens` (llama.h). The port's container is always
    /// token-indexed — the C's embd-only batches (`token == nullptr`) ride
    /// placeholder ids here (e.g. the dflash injection batches, one `0` per
    /// row), so `token.len()` *is* n_tokens.
    fn n_tokens(&self) -> usize {
        self.token.len()
    }

    /// `batch.n_seq_id[i]` when provided
    fn n_seq_id(&self) -> Option<Vec<i32>> {
        self.seq_id.as_ref().map(|s| {
            s.iter()
                .map(|ids| ids.len() as i32)
                .collect::<Vec<i32>>()
        })
    }
}

/// `llama_batch_allocr` (llama-batch.h:72-174).
pub struct BatchAllocr {
    batch: LlamaBatch,

    n_pos_per_embd: u32,
    n_embd: u32,
    n_seq_max: u32,
    n_outputs: u32,

    pos: Vec<i32>,
    n_seq_id: Vec<i32>,
    seq_id: Vec<Vec<i32>>,
    seq_id_unq: Vec<i32>,
    seq_idx: Vec<i32>,
    output: Vec<bool>,
    /// `decision_order` (llama-batch.h:209, a7b94df2c) — empty if no entry
    /// has one
    decision_order: Vec<i32>,

    /// `seq_pos[s]`: the set of positions of sequence s in this batch (:155).
    /// The C uses `std::set<llama_pos>` and reads only min/max + size, so a
    /// sorted Vec is equivalent.
    seq_pos: Vec<Vec<i32>>,
    /// `seq_cpl[s0][s1]`: s0 coupled to s1 (:156)
    seq_cpl: Vec<Vec<bool>>,
    has_cpl: bool,

    /// `seq_set[i]`: the sequence set of token i, as the C's
    /// `std::bitset<LLAMA_MAX_SEQ>` (:159-163)
    seq_set: Vec<u64>,
    /// `seq_set_map`: the token indices at which each sequence set appears
    seq_set_map: Vec<(u64, Vec<i32>)>,

    /// batch indices of the outputs, in the order they were encountered while
    /// splitting (:165-166)
    out_ids: Vec<i32>,

    n_used: u32,
    used: Vec<bool>,

    pub debug: i32,
}

impl BatchAllocr {
    /// `llama_batch_allocr::llama_batch_allocr` (llama-batch.cpp:12-23)
    pub fn new(n_pos_per_embd: u32) -> Self {
        let debug = std::env::var("LLAMA_BATCH_DEBUG")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        BatchAllocr {
            batch: LlamaBatch::default(),
            n_pos_per_embd,
            n_embd: 0,
            n_seq_max: 0,
            n_outputs: 0,
            pos: Vec::new(),
            n_seq_id: Vec::new(),
            seq_id: Vec::new(),
            seq_id_unq: Vec::new(),
            seq_idx: vec![-1; LLAMA_MAX_SEQ],
            output: Vec::new(),
            decision_order: Vec::new(),
            seq_pos: vec![Vec::new(); LLAMA_MAX_SEQ],
            seq_cpl: vec![vec![false; LLAMA_MAX_SEQ]; LLAMA_MAX_SEQ],
            has_cpl: false,
            seq_set: Vec::new(),
            seq_set_map: Vec::new(),
            out_ids: Vec::new(),
            n_used: 0,
            used: Vec::new(),
            debug,
        }
    }

    /// `llama_batch_allocr::init` (llama-batch.cpp:26-391, fc343a84b) —
    /// convert a [`LlamaBatchExt`] to the internal batch and sanitize it.
    ///
    /// `vocab_n_tokens` mirrors the C's kept `vocab` member reference — token
    /// ids are validated against the ext's own `n_vocab` (the width the ext
    /// was built with, llama-batch.cpp:60-66).
    pub fn init(
        &mut self,
        batch_inp: &LlamaBatchExt<'_>,
        vocab_n_tokens: u32,
        output_all: bool,
    ) -> Result<(), String> {
        let _ = vocab_n_tokens;
        self.clear();

        self.n_embd = if batch_inp.n_embd > 0 {
            batch_inp.n_embd
        } else {
            batch_inp.n_embd_inp
        } as u32;
        self.n_seq_max = batch_inp.n_seq_max as u32;

        let n_tok = batch_inp.tokens.len() as i32;
        if n_tok <= 0 {
            return Err("batch.n_tokens must be > 0".into());
        }

        if self.n_seq_max as usize > LLAMA_MAX_SEQ {
            return Err(format!(
                "n_seq_max = {} > {LLAMA_MAX_SEQ}",
                self.n_seq_max
            ));
        }

        let mem = batch_inp.mem;

        //
        // determine the content types of the batch (:43-62)
        // an entry can carry a token id, a token embedding, or both (e.g. MTP
        // hook batches); all entries must carry the same combination
        //

        let has_token = batch_inp.tokens[0].id != LLAMA_TOKEN_NULL;
        let has_embd = batch_inp.tokens[0].has_embd;

        for i in 1..n_tok as usize {
            if (batch_inp.tokens[i].id != LLAMA_TOKEN_NULL) != has_token
                || batch_inp.tokens[i].has_embd != has_embd
            {
                return Err(
                    "all entries in the batch must have the same content types".into()
                );
            }
        }

        if !has_token && !has_embd {
            return Err("batch has neither token ids nor embeddings".into());
        }

        //
        // build flat token/embd array (:64-84)
        //

        let mut token_vec: Vec<i32> = Vec::new();
        if has_token {
            token_vec = vec![0; n_tok as usize];
            for (i, tok) in batch_inp.tokens.iter().enumerate() {
                let id = tok.id;
                if id < 0 || id >= batch_inp.n_vocab {
                    return Err(format!("invalid token[{i}] = {id}"));
                }
                token_vec[i] = id;
            }
        }

        let embd_vec: Vec<f32> = if has_embd {
            batch_inp.embd.clone()
        } else {
            Vec::new()
        };

        //
        // build flat pos array (:86-104)
        // token batch:     pos[i]            = tokens[i].pos[0]
        // embedding batch: pos[j*n_tok + i]  = tokens[i].pos[j]  (section-major)
        //

        let n_pos_total = if has_token {
            n_tok as usize
        } else {
            n_tok as usize * batch_inp.n_pos_per_embd
        };
        self.pos.resize(n_pos_total, 0);
        if has_token {
            for i in 0..n_tok as usize {
                self.pos[i] = batch_inp.tokens[i].pos[0];
            }
        } else {
            for i in 0..n_tok as usize {
                for j in 0..batch_inp.n_pos_per_embd {
                    self.pos[j * n_tok as usize + i] = batch_inp.tokens[i].pos[j];
                }
            }
        }

        //
        // build n_seq_id / seq_id arrays (:106-133)
        //

        self.n_seq_id = vec![0; n_tok as usize];
        self.seq_id = vec![Vec::new(); n_tok as usize + 1];

        {
            let mut off = 0usize;
            for i in 0..n_tok as usize {
                let sids = &batch_inp.tokens[i].seq_ids;
                self.n_seq_id[i] = sids.len() as i32;
                self.seq_id[i] = sids.clone();
                off += sids.len();

                for (s, &id) in sids.iter().enumerate() {
                    if id < 0 || id >= self.n_seq_max as i32 {
                        return Err(format!(
                            "invalid seq_id[{i}][{s}] = {id} >= {}",
                            self.n_seq_max
                        ));
                    }
                }
            }
            let _ = off;
        }

        //
        // build output/logits array (:135-155)
        //

        {
            self.output = vec![false; n_tok as usize];
            for i in 0..n_tok as usize {
                self.output[i] = batch_inp.tokens[i].output;
            }

            if output_all {
                let warn = self.output.iter().any(|&o| !o);
                if warn {
                    crate::impl_log::log_internal(
                        crate::impl_log::LogLevel::Warn,
                        format_args!(
                            "embeddings required but some input tokens were not marked as outputs \
                             -> overriding"
                        ),
                    );
                    self.output.fill(true);
                }
            }
        }

        // kept empty if no entry has one (llama-batch.cpp:171-177, a7b94df2c)
        // — the resize lands on the first non-zero entry, so a decision batch
        // carries the full [n_tok] row and every other batch stays empty (the
        // C's NULL pointer)
        for i in 0..n_tok as usize {
            if batch_inp.tokens[i].decision_order != 0 {
                self.decision_order.resize(n_tok as usize, 0);
                self.decision_order[i] = batch_inp.tokens[i].decision_order;
            }
        }

        //
        // set up the internal batch to point to our owned arrays (:157-167)
        //

        self.batch = LlamaBatch {
            token: token_vec,
            pos: Some(self.pos.clone()),
            // the C's `seq_id` array carries n_tok + 1 entries (the null
            // terminator, llama-batch.cpp:109-110) — the port's container
            // takes the n_tok real ones
            seq_id: Some(self.seq_id[..n_tok as usize].to_vec()),
            logits: Some(self.output.clone()),
            embd: if has_embd { Some(embd_vec) } else { None },
        };

        //
        // compute stats (:170-207) — count the outputs in this batch
        //

        {
            let logits = self.batch.logits.as_ref().unwrap();
            for i in 0..n_tok as usize {
                self.n_outputs += logits[i] as u32;
            }
        }

        self.has_cpl = false;
        self.seq_set.clear();
        self.seq_set_map.clear();

        let seq_id = self.batch.seq_id.as_ref().unwrap().clone();
        for i in 0..n_tok as usize {
            let s0 = seq_id[i][0];
            for &s1 in seq_id[i].iter() {
                self.seq_pos[s1 as usize].push(self.batch.pos.as_ref().unwrap()[i]);
                if s1 != s0 {
                    // mark that sequence s1 is coupled to s0 (:172-179)
                    self.seq_cpl[s1 as usize][s0 as usize] = true;
                    self.has_cpl = true;
                }
            }
        }
        // the C's std::set dedups and sorts; only min/max/size are read
        for v in self.seq_pos.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }

        // the sequence sets of each token + the unique seq ids (:184-207)
        {
            let mut seq_set_unq = 0u64;
            for i in 0..n_tok as usize {
                let mut cur = 0u64;
                for &s in seq_id[i].iter() {
                    cur |= 1u64 << s;
                    seq_set_unq |= 1u64 << s;
                }
                self.seq_set.push(cur);
                match self.seq_set_map.iter_mut().find(|(k, _)| *k == cur) {
                    Some((_, idxs)) => idxs.push(i as i32),
                    None => self.seq_set_map.push((cur, vec![i as i32])),
                }
            }
            for s in 0..self.n_seq_max as usize {
                if seq_set_unq & (1u64 << s) != 0 {
                    self.seq_idx[s] = self.seq_id_unq.len() as i32;
                    self.seq_id_unq.push(s as i32);
                }
            }
        }

        // consistency checks (:251-386)
        let pos = self.batch.pos.as_ref().unwrap();
        // same discriminator as `ubatch_add`: a flat pos array with one entry
        // per RoPE section marks an embedding batch, whose positions MAY
        // overlap the memory (:283 "embedding inputs can have overlapping
        // positions"); token batches must move strictly forward (:266).
        let embed_batch = self.n_pos_per_embd > 1
            && pos.len() >= n_tok as usize * self.n_pos_per_embd as usize;
        if self.n_pos_per_embd > 1 {
            // M-RoPE: positions may jump forward (:255-288)
            for s in 0..self.n_seq_max as usize {
                let (smin, smax) = match Self::pos_min_max(&self.seq_pos[s]) {
                    Some(v) => v,
                    None => continue,
                };
                let _ = smax;
                let p0 = mem.map(|m| m.seq_pos_max(s as i32)).unwrap_or(-1);
                if p0 >= 0 && (if embed_batch { p0 > smin } else { p0 >= smin }) {
                    return Err(format!(
                        "the tokens of sequence {s} in the input batch have inconsistent sequence \
                         positions: memory max X = {p0}, batch start Y = {smin}; for M-RoPE it is \
                         required that X < Y"
                    ));
                }
            }
        } else {
            for s in 0..self.n_seq_max as usize {
                let (smin, smax) = match Self::pos_min_max(&self.seq_pos[s]) {
                    Some(v) => v,
                    None => continue,
                };
                let p0 = mem.map(|m| m.seq_pos_max(s as i32)).unwrap_or(-1);
                if p0 >= 0 && smin != p0 + 1 {
                    return Err(format!(
                        "the tokens of sequence {s} in the input batch have inconsistent sequence \
                         positions: memory max X = {p0}, batch start Y = {smin}; Y = X + 1 is required"
                    ));
                }
                if smax - smin + 1 > self.seq_pos[s].len() as i32 {
                    return Err(format!("sequence {s} positions are not continuous"));
                }
            }
        }

        if let Some(m) = mem {
            for s0 in 0..self.n_seq_max as usize {
                for s1 in 0..self.n_seq_max as usize {
                    if self.seq_cpl[s0][s1]
                        && (m.seq_pos_min(s0 as i32) != m.seq_pos_min(s1 as i32)
                            || m.seq_pos_max(s0 as i32) != m.seq_pos_max(s1 as i32))
                    {
                        return Err(format!(
                            "sequence {s0} is coupled to {s1} in the input batch, but have diverged"
                        ));
                    }
                }
            }
        }

        // disallow partial sequence sub-sets and decreasing positions (:354-386)
        {
            let mut cur_seq_set = vec![u64::MAX; LLAMA_MAX_SEQ];
            let mut cur_seq_pos = vec![-1i32; LLAMA_MAX_SEQ];
            for i in 0..n_tok as usize {
                let p = pos[i];
                for &s in seq_id[i].iter() {
                    let s = s as usize;
                    cur_seq_set[s] &= self.seq_set[i];
                    if cur_seq_set[s] == 0 {
                        return Err(format!(
                            "sequence {s} belongs to incompatible sequence sets (not allowed)"
                        ));
                    }
                    if p < cur_seq_pos[s] {
                        return Err(format!(
                            "sequence {s} positions are decreasing (not allowed)"
                        ));
                    }
                    cur_seq_pos[s] = p;
                }
            }
        }

        self.split_reset();
        Ok(())
    }

    fn pos_min_max(p: &[i32]) -> Option<(i32, i32)> {
        if p.is_empty() {
            None
        } else {
            Some((p[0], p[p.len() - 1]))
        }
    }

    pub fn get_batch(&self) -> &LlamaBatch {
        &self.batch
    }

    pub fn get_n_tokens(&self) -> u32 {
        self.batch.token.len() as u32
    }

    pub fn get_n_outputs(&self) -> u32 {
        self.n_outputs
    }

    pub fn get_n_used(&self) -> u32 {
        self.n_used
    }

    /// `get_out_ids` (:455)
    pub fn get_out_ids(&self) -> &[i32] {
        &self.out_ids
    }

    /// `seq_pos_min` (:459)
    pub fn seq_pos_min(&self, seq_id: i32) -> i32 {
        Self::pos_min_max(&self.seq_pos[seq_id as usize])
            .map(|(m, _)| m)
            .unwrap_or(-1)
    }

    /// `seq_pos_max` (:463)
    pub fn seq_pos_max(&self, seq_id: i32) -> i32 {
        Self::pos_min_max(&self.seq_pos[seq_id as usize])
            .map(|(_, m)| m)
            .unwrap_or(-1)
    }

    /// `split_reset` (:467-474)
    pub fn split_reset(&mut self) {
        self.out_ids.clear();
        self.n_used = 0;
        self.used = vec![false; self.batch.token.len()];
    }

    /// `split_simple` (:476-508) — the only split the single-stream KV cache
    /// uses (llama-kv-cache.cpp:668 `n_stream == 1 ? split_simple(n_ubatch) : …`).
    pub fn split_simple(&mut self, n_ubatch: u32) -> LlamaUbatch {
        let mut cur_idx = 0usize;
        while cur_idx < self.used.len() && self.used[cur_idx] {
            cur_idx += 1;
        }
        if cur_idx >= self.used.len() {
            return LlamaUbatch::empty();
        }

        let mut idxs = Vec::new();
        loop {
            idxs.push(cur_idx as i32);
            self.used[cur_idx] = true;
            self.n_used += 1;
            cur_idx += 1;
            if cur_idx >= self.used.len() {
                break;
            }
            if idxs.len() >= n_ubatch as usize {
                break;
            }
        }

        self.ubatch_add(&idxs, idxs.len() as u32, false)
    }

    /// `split_equal` (:510-679) — equal-length sequence sets. Not reachable
    /// with the port's single-stream KV cache (`n_stream == 1`); ported so the
    /// geometry is testable and a pipeline/multi-stream cache can use it.
    pub fn split_equal(
        &mut self,
        n_ubatch: u32,
        sequential: bool,
        n_keep_tail: u32,
    ) -> LlamaUbatch {
        if sequential && self.has_cpl {
            return LlamaUbatch::empty();
        }

        let mut cur_seq_set: Vec<u64> = Vec::new();
        let mut last_seq_id: i32 = -1;

        // the non-overlapping sequence sets participating in this ubatch
        for i in 0..self.batch.token.len() {
            if self.used[i] {
                continue;
            }
            let mut add = true;
            for s in cur_seq_set.iter() {
                if s & self.seq_set[i] != 0 {
                    add = false;
                    break;
                }
            }
            if sequential {
                let seq0 = self.batch.seq_id.as_ref().unwrap()[i][0];
                add = add && (cur_seq_set.is_empty() || seq0 == last_seq_id + 1);
            }
            if add {
                cur_seq_set.push(self.seq_set[i]);
                last_seq_id = self.batch.seq_id.as_ref().unwrap()[i][0];
                if cur_seq_set.len() > n_ubatch as usize {
                    break;
                }
            }
        }

        let mut n_seqs = cur_seq_set.len();
        if n_seqs == 0 {
            return LlamaUbatch::empty();
        }

        // the token indices of each sequence set (:570-571)
        let seq_set_map: Vec<Vec<i32>> = cur_seq_set
            .iter()
            .map(|&s| self.seq_set_map_of(s))
            .collect();

        // the current batch index of each sequence set (:560-567)
        let mut cur_idx = vec![0usize; n_seqs];
        for s in 0..n_seqs {
            let map = &seq_set_map[s];
            while self.used[map[cur_idx[s]] as usize] {
                cur_idx[s] += 1;
            }
        }

        let mut idxs_per_seq: Vec<Vec<i32>> = vec![Vec::new(); n_seqs];
        loop {
            // (:573-603) expand only while every set has another unused token
            let mut can_expand = true;
            for s in 0..n_seqs {
                if cur_idx[s] >= seq_set_map[s].len() {
                    can_expand = false;
                    break;
                }
            }
            if !can_expand {
                break;
            }

            for s in 0..n_seqs {
                let idx = seq_set_map[s][cur_idx[s]];
                idxs_per_seq[s].push(idx);
                self.used[idx as usize] = true;
                self.n_used += 1;
                cur_idx[s] += 1;
            }

            if (idxs_per_seq[0].len() as u32 + 1) * n_seqs as u32 > n_ubatch {
                break;
            }
        }

        if n_keep_tail > 0 {
            // (:608-669) never split the trailing n_keep_tail tokens of a seq
            assert!(n_ubatch > n_keep_tail);
            let n_remaining = |s: usize, cur_idx: &Vec<usize>| -> u32 {
                (seq_set_map[s].len() - cur_idx[s]) as u32
            };

            let mut n_keep = 0usize;
            while n_keep < n_seqs {
                let rem = n_remaining(n_keep, &cur_idx);
                if rem != 0 && rem < n_keep_tail {
                    break;
                }
                n_keep += 1;
            }

            if n_keep == 0 {
                let map = &seq_set_map[0];
                if idxs_per_seq[0].len() as u32 + n_remaining(0, &cur_idx) <= n_ubatch {
                    while n_remaining(0, &cur_idx) > 0 {
                        let idx = map[cur_idx[0]];
                        idxs_per_seq[0].push(idx);
                        self.used[idx as usize] = true;
                        self.n_used += 1;
                        cur_idx[0] += 1;
                    }
                } else {
                    while n_remaining(0, &cur_idx) < n_keep_tail {
                        let idx = idxs_per_seq[0].pop().unwrap();
                        self.used[idx as usize] = false;
                        self.n_used -= 1;
                        cur_idx[0] -= 1;
                    }
                }
                n_keep = 1;
            }

            for s in n_keep..n_seqs {
                for &idx in idxs_per_seq[s].iter() {
                    self.used[idx as usize] = false;
                    self.n_used -= 1;
                }
            }
            n_seqs = n_keep;
        }

        let mut idxs = Vec::new();
        for s in 0..n_seqs {
            idxs.extend_from_slice(&idxs_per_seq[s]);
        }
        self.ubatch_add(&idxs, n_seqs as u32, true)
    }

    /// `split_seq` (:681-721) — one sequence set per ubatch.
    pub fn split_seq(&mut self, n_ubatch: u32) -> LlamaUbatch {
        let mut cur_idx = 0usize;
        while cur_idx < self.used.len() && self.used[cur_idx] {
            cur_idx += 1;
        }
        if cur_idx >= self.used.len() {
            return LlamaUbatch::empty();
        }

        let mut cur_seq_set = self.seq_set[cur_idx];
        let mut idxs = Vec::new();
        loop {
            idxs.push(cur_idx as i32);
            self.used[cur_idx] = true;
            self.n_used += 1;

            if idxs.len() >= n_ubatch as usize {
                break;
            }

            loop {
                cur_idx += 1;
                if cur_idx >= self.batch.token.len() {
                    break;
                }
                if !self.used[cur_idx]
                    && (cur_seq_set & self.seq_set[cur_idx]) == self.seq_set[cur_idx]
                {
                    break;
                }
            }
            if cur_idx == self.batch.token.len() {
                break;
            }
            cur_seq_set = self.seq_set[cur_idx];
        }

        self.ubatch_add(&idxs, 1, true)
    }

    /// `ubatch_reserve` (:393-437) — a well-defined ubatch without a batch.
    pub fn ubatch_reserve(&mut self, n_seq_tokens: u32, n_seqs: u32) -> LlamaUbatch {
        self.clear();
        self.split_reset();

        let n_tokens = n_seq_tokens * n_seqs;
        let mut ub = LlamaUbatch {
            equal_seqs: true,
            n_tokens,
            n_seq_tokens,
            n_seqs,
            n_seqs_unq: n_seqs,
            n_pos: self.n_pos_per_embd,
            token: vec![0; n_tokens as usize],
            pos: vec![0; (n_tokens * self.n_pos_per_embd) as usize],
            embd: Vec::new(),
            n_seq_id: vec![1; n_tokens as usize],
            seq_id: vec![vec![0]; n_tokens as usize],
            seq_id_unq: Vec::new(),
            seq_idx: vec![-1; LLAMA_MAX_SEQ],
            output: vec![false; n_tokens as usize],
            // `ubatch_reserve` leaves decision_order empty (the C's nullptr,
            // llama-batch.cpp:474, a7b94df2c)
            decision_order: Vec::new(),
            // `ubatch_reserve` leaves batch_idxs empty (llama-batch.h:65 —
            // only `ubatch_add` carries the logical-batch indices)
            batch_idxs: Vec::new(),
        };
        for s in 0..n_seqs {
            ub.seq_idx[s as usize] = s as i32;
            ub.seq_id_unq.push(s as i32);
        }
        ub
    }

    /// `clear` (:757-781) — the C clears `token_vec` / `embd_vec` /
    /// `seq_id_data` (the owned arrays behind the internal batch's pointers,
    /// fc343a84b); the port folds all three into `self.batch`.
    fn clear(&mut self) {
        self.n_outputs = 0;
        self.batch = LlamaBatch::default();
        self.pos.clear();
        self.n_seq_id.clear();
        self.seq_id.clear();
        self.seq_id_unq.clear();
        self.output.clear();
        self.decision_order.clear();
        for v in self.seq_pos.iter_mut() {
            v.clear();
        }
        for v in self.seq_cpl.iter_mut() {
            v.fill(false);
        }
        self.seq_set.clear();
        self.seq_set_map.clear();
        self.seq_idx.fill(-1);
    }

    fn seq_set_map_of(&self, set: u64) -> Vec<i32> {
        self.seq_set_map
            .iter()
            .find(|(k, _)| *k == set)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }

    /// `ubatch_add` (:749-844)
    fn ubatch_add(&mut self, idxs: &[i32], n_seqs: u32, equal_seqs: bool) -> LlamaUbatch {
        let n_tokens = idxs.len() as u32;
        assert!(n_seqs > 0 && n_tokens % n_seqs == 0);

        let batch_pos = self.batch.pos.clone().unwrap();
        let batch_logits = self.batch.logits.clone().unwrap();
        let batch_seq_id = self.batch.seq_id.clone().unwrap();
        // the MTP draft hook batches carry F32 embd rows (speculative.cpp:
        // 1521-1547); the width is the row count / token count
        let batch_embd = self.batch.embd.as_ref();
        let n_embd_row = batch_embd
            .map(|e| e.len() / self.batch.token.len().max(1))
            .unwrap_or(0);
        // llama-batch.cpp:781-788: with an *embedding* batch (`batch.embd !=
        // NULL` in C) the caller supplies one position per RoPE section
        // (`batch.pos` laid out [n_pos][n_tokens], the image-embedding
        // convention of mtmd-helper-common.h:120); a token batch carries one
        // position per token and it is broadcast across the sections. This port
        // has no `embd` field, so a flat pos array long enough to hold
        // n_pos_per_embd positions per token selects the per-section copy — the
        // existing (shorter) arrays keep the broadcast behaviour unchanged.
        let n_tokens_all = self.batch.token.len() as usize;
        let per_section = self.n_pos_per_embd > 1
            && batch_pos.len() >= n_tokens_all * self.n_pos_per_embd as usize;

        let mut ub = LlamaUbatch {
            equal_seqs,
            n_tokens,
            n_seq_tokens: n_tokens / n_seqs,
            n_seqs,
            n_seqs_unq: 0,
            n_pos: self.n_pos_per_embd,
            token: Vec::with_capacity(n_tokens as usize),
            pos: vec![0; (n_tokens * self.n_pos_per_embd) as usize],
            embd: Vec::with_capacity(
                batch_embd
                    .map(|_| (n_tokens as usize) * n_embd_row)
                    .unwrap_or(0),
            ),
            n_seq_id: vec![0; n_tokens as usize],
            seq_id: Vec::with_capacity(n_tokens as usize),
            seq_id_unq: Vec::new(),
            seq_idx: vec![-1; LLAMA_MAX_SEQ],
            output: vec![false; n_tokens as usize],
            // `udata->decision_order.resize(decision_order.empty() ? 0 :
            // n_tokens)` (llama-batch.cpp:813, a7b94df2c)
            decision_order: vec![0; if self.decision_order.is_empty() {
                0
            } else {
                n_tokens as usize
            }],
            batch_idxs: idxs.to_vec(),
        };

        let mut seq_set_unq = 0u64;
        for (i, &idx) in idxs.iter().enumerate() {
            let idx = idx as usize;
            ub.token.push(self.batch.token[idx]);
            if let Some(e) = batch_embd {
                // one F32 row per token, [n_tokens][n_embd] row-major
                ub.embd
                    .extend_from_slice(&e[idx * n_embd_row..(idx + 1) * n_embd_row]);
            }
            for j in 0..self.n_pos_per_embd as usize {
                // `src_off = batch.token ? 0 : j*batch.n_tokens` (llama-batch.cpp:784)
                let src_off = if per_section { j * n_tokens_all } else { 0 };
                ub.pos[j * n_tokens as usize + i] = batch_pos[src_off + idx];
            }
            let ids = &batch_seq_id[idx];
            ub.n_seq_id[i] = ids.len() as i32;
            ub.output[i] = batch_logits[idx];
            // (llama-batch.cpp:841-843, a7b94df2c)
            if !self.decision_order.is_empty() {
                ub.decision_order[i] = self.decision_order[idx];
            }
            for &s in ids.iter() {
                seq_set_unq |= 1u64 << s;
            }
            ub.seq_id.push(ids.clone());
            if ub.output[i] {
                self.out_ids.push(idx as i32);
            }
        }

        for s in 0..self.n_seq_max as usize {
            if seq_set_unq & (1u64 << s) != 0 {
                ub.seq_idx[s] = ub.seq_id_unq.len() as i32;
                ub.seq_id_unq.push(s as i32);
            }
        }
        ub.n_seqs_unq = ub.seq_id_unq.len() as u32;

        if self.debug > 0 {
            self.ubatch_print(&ub);
        }

        ub
    }

    /// `ubatch_print` (:846-925), the LLAMA_BATCH_DEBUG > 0 diagnostics.
    fn ubatch_print(&self, ub: &LlamaUbatch) {
        eprintln!("ubatch_add:   equal_seqs   = {}", ub.equal_seqs as i32);
        eprintln!("ubatch_add:   n_tokens     = {}", ub.n_tokens);
        eprintln!("ubatch_add:   n_seq_tokens = {}", ub.n_seq_tokens);
        eprintln!("ubatch_add:   n_seqs       = {}", ub.n_seqs);
        eprintln!("ubatch_add:   n_seqs_unq   = {}", ub.n_seqs_unq);
        for i in 0..ub.n_tokens as usize {
            eprintln!(
                "ubatch_add:  {:4}: id = {:6}, pos = {:4}, n_seq_id = {:2}, seq_id = {:?}, output = {}",
                i, ub.token[i], ub.pos[i], ub.n_seq_id[i], ub.seq_id[i], ub.output[i] as i32
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocr() -> BatchAllocr {
        BatchAllocr::new(1)
    }

    /// `batch_builder` (tests/test-batch-alloc.cpp:52-90, fc343a84b) — build a
    /// `llama_batch_ext` from the caller-facing batch via the compat layer,
    /// the way `llama_context::decode(const llama_batch &)` does
    /// (llama-context.cpp:4376-4385).
    fn ext_of<'m>(
        b: &LlamaBatch,
        mem: Option<&'m dyn BatchMemory>,
        n_seq_max: u32,
    ) -> LlamaBatchExt<'m> {
        let mut ext = LlamaBatchExt::new(
            b.token.len().max(1) * 2,
            8,
            8,
            n_seq_max as i32,
            mem,
            100,
            1,
        );
        batch_compat_init(&mut ext, b, 0);
        ext
    }

    /// The auto-generated fields of `init` (llama-batch.cpp:69-146): a bare
    /// token-only batch gets pos = 0..n, one seq, and only the last token as
    /// an output.
    #[test]
    fn init_autogenerates_pos_seq_and_logits() {
        let mut a = allocr();
        let b = LlamaBatch::get_one(vec![1, 2, 3]);
        a.init(&ext_of(&b, None, 1), 100, false).unwrap();
        let got = a.get_batch();
        assert_eq!(got.pos.as_ref().unwrap(), &[0, 1, 2]);
        assert_eq!(got.seq_id.as_ref().unwrap(), &vec![vec![0]; 3]);
        assert_eq!(got.logits.as_ref().unwrap(), &[false, false, true]);
        assert_eq!(a.get_n_outputs(), 1);
        assert_eq!(a.seq_pos_min(0), 0);
        assert_eq!(a.seq_pos_max(0), 2);
    }

    /// With a memory module the auto positions continue each sequence from its
    /// own `seq_pos_max + 1` (:93-115) — the per-sequence bookkeeping the
    /// server's prompt reuse relies on.
    #[test]
    fn init_autogenerates_pos_from_memory_per_seq() {
        struct Mem(Vec<i32>);
        impl BatchMemory for Mem {
            fn seq_pos_min(&self, s: i32) -> i32 {
                if s == 0 {
                    0
                } else {
                    -1
                }
            }
            fn seq_pos_max(&self, s: i32) -> i32 {
                if s == 0 {
                    9
                } else {
                    -1
                }
            }
        }
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3, 4],
            pos: None,
            seq_id: Some(vec![vec![0], vec![1], vec![0], vec![1]]),
            logits: None,
            embd: None,
        };
        let mem = Mem(vec![0, 9]);
        a.init(&ext_of(&b, Some(&mem), 2), 100, false).unwrap();
        // seq 0 resumes at 10, seq 1 starts at 0
        assert_eq!(a.get_batch().pos.as_ref().unwrap(), &[10, 0, 11, 1]);
        assert_eq!(a.seq_pos_max(0), 11);
        assert_eq!(a.seq_pos_min(1), 0);
    }

    /// Non-consecutive positions of one sequence are rejected unless
    /// `n_pos_per_embd > 1` (llama-batch.cpp:289-321).
    #[test]
    fn init_rejects_non_consecutive_positions() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2],
            pos: Some(vec![0, 2]),
            seq_id: Some(vec![vec![0], vec![0]]),
            logits: None,
            embd: None,
        };
        let err = a.init(&ext_of(&b, None, 1), 100, false).unwrap_err();
        assert!(err.contains("not continuous"), "{err}");
    }

    /// Two sequences in one batch: the sequence sets and the unique seq ids
    /// (:184-207), plus the ubatch split keeping all of them in one ubatch.
    #[test]
    fn split_simple_multi_seq_geometry() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3, 4],
            pos: Some(vec![0, 1, 0, 1]),
            seq_id: Some(vec![vec![0], vec![0], vec![1], vec![1]]),
            logits: Some(vec![false, true, false, true]),
            embd: None,
        };
        a.init(&ext_of(&b, None, 4), 100, false).unwrap();
        assert_eq!(a.get_n_outputs(), 2);

        let ub = a.split_simple(8);
        assert_eq!(ub.n_tokens, 4);
        assert_eq!(ub.n_seqs, 4);
        assert_eq!(ub.n_seqs_unq, 2);
        assert!(!ub.equal_seqs());
        assert_eq!(ub.seq_id, vec![vec![0], vec![0], vec![1], vec![1]]);
        assert_eq!(ub.seq_id_unq, vec![0, 1]);
        assert_eq!(ub.seq_idx[0], 0);
        assert_eq!(ub.seq_idx[1], 1);
        assert_eq!(ub.seq_idx[2], -1);
        assert_eq!(ub.output, vec![false, true, false, true]);
        assert_eq!(a.get_out_ids(), &[1, 3]);
        assert_eq!(a.get_n_used(), 4);

        // the whole batch is consumed
        let ub2 = a.split_simple(8);
        assert!(ub2.is_empty());
    }

    /// A batch larger than n_ubatch is split, and `out_ids` collects the
    /// outputs in the order they were encountered (:800-802).
    #[test]
    fn split_simple_chunks() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3, 4, 5],
            pos: None,
            seq_id: None,
            logits: Some(vec![false, true, false, true, false]),
            embd: None,
        };
        a.init(&ext_of(&b, None, 1), 100, false).unwrap();
        let u0 = a.split_simple(2);
        assert_eq!(u0.token, vec![1, 2]);
        assert_eq!(u0.output, vec![false, true]);
        let u1 = a.split_simple(2);
        assert_eq!(u1.token, vec![3, 4]);
        let u2 = a.split_simple(2);
        assert_eq!(u2.token, vec![5]);
        assert!(a.split_simple(2).is_empty());
        assert_eq!(a.get_out_ids(), &[1, 3]);
    }

    /// `decision_order` (a7b94df2c, llama-batch.cpp:171-177/:813/:841-843):
    /// the ext carries one row per marked entry, the ubatch splits follow it
    /// through `batch_idxs`, and a batch with no marked entry keeps the
    /// ubatch's vector empty (the C's NULL pointer).
    #[test]
    fn decision_order_flows_to_ubatches() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3, 4],
            pos: None,
            seq_id: None,
            logits: None,
            embd: None,
        };
        let mut ext = ext_of(&b, None, 1);
        // one question span (2 entries) then one option span
        assert!(ext.set_decision_order(0, DECISION_ORDER_QUESTION_NOUL));
        assert!(ext.set_decision_order(1, DECISION_ORDER_QUESTION_NOUL));
        assert!(ext.set_decision_order(3, DECISION_ORDER_OPTION));
        assert!(!ext.set_decision_order(-1, DECISION_ORDER_OPTION));
        assert!(!ext.set_decision_order(4, DECISION_ORDER_OPTION));
        a.init(&ext, 100, false).unwrap();

        let u0 = a.split_simple(2);
        assert_eq!(u0.token, vec![1, 2]);
        assert_eq!(u0.decision_order, vec![DECISION_ORDER_QUESTION_NOUL, DECISION_ORDER_QUESTION_NOUL]);
        let u1 = a.split_simple(2);
        assert_eq!(u1.token, vec![3, 4]);
        assert_eq!(u1.decision_order, vec![0, DECISION_ORDER_OPTION]);

        // no marked entry → the ubatch's decision_order stays empty
        let mut a2 = allocr();
        a2.init(&ext_of(&b, None, 1), 100, false).unwrap();
        let u = a2.split_simple(4);
        assert_eq!(u.token, vec![1, 2, 3, 4]);
        assert!(u.decision_order.is_empty());

        // `ubatch_reserve` leaves it empty too (llama-batch.cpp:474)
        let mut a3 = allocr();
        let u = a3.ubatch_reserve(2, 2);
        assert_eq!(u.n_tokens, 4);
        assert!(u.decision_order.is_empty());
    }

    /// A token assigned to two sequences couples them (:162-182) and the
    /// sequence sets then overlap.
    #[test]
    fn coupled_sequences_share_a_token() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3],
            pos: Some(vec![0, 0, 1]),
            seq_id: Some(vec![vec![0, 1], vec![0, 1], vec![1]]),
            logits: None,
            embd: None,
        };
        a.init(&ext_of(&b, None, 4), 100, false).unwrap();
        assert_eq!(a.get_batch().pos.as_ref().unwrap(), &[0, 0, 1]);
        // the shared prefix puts both sequences at pos 0; seq 1 then continues
        // at pos 1, which is what seq_pos_max reports
        assert_eq!(a.seq_pos_max(1), 1);
        assert_eq!(a.seq_pos_min(0), 0);

        let ub = a.split_simple(8);
        assert_eq!(ub.n_seq_id, vec![2, 2, 1]);
        assert_eq!(ub.seq_id_unq, vec![0, 1]);
    }

    /// `split_equal` (:510-679): two sequences of the same length land in one
    /// ubatch of equal sequence sets.
    #[test]
    fn split_equal_geometry() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3, 4, 5, 6],
            pos: Some(vec![0, 1, 2, 0, 1, 2]),
            seq_id: Some(vec![vec![0], vec![0], vec![0], vec![1], vec![1], vec![1]]),
            logits: None,
            embd: None,
        };
        a.init(&ext_of(&b, None, 4), 100, false).unwrap();
        let ub = a.split_equal(4, false, 0);
        assert!(ub.equal_seqs());
        assert_eq!(ub.n_seqs, 2);
        assert_eq!(ub.n_seq_tokens, 2);
        assert_eq!(ub.token, vec![1, 2, 4, 5]);
        // the second call still takes one token of each set before the
        // "can_expand" check ends it (:573-603 — the expansion loop always runs
        // at least once)
        let ub2 = a.split_equal(4, false, 0);
        assert_eq!(ub2.token, vec![3, 6]);
        assert_eq!(ub2.n_seq_tokens, 1);
        assert!(a.split_equal(4, false, 0).is_empty());
    }

    /// `split_seq` (:681-721): only tokens whose sequence set is a subset of
    /// the current one are added.
    #[test]
    fn split_seq_keeps_one_sequence_set() {
        let mut a = allocr();
        let b = LlamaBatch {
            token: vec![1, 2, 3],
            pos: Some(vec![0, 0, 1]),
            seq_id: Some(vec![vec![0, 1], vec![1], vec![0]]),
            logits: None,
            embd: None,
        };
        a.init(&ext_of(&b, None, 4), 100, false).unwrap();
        let ub0 = a.split_seq(8);
        assert_eq!(ub0.token, vec![1, 2]); // {0,1} then {1}
        let ub1 = a.split_seq(8);
        assert_eq!(ub1.token, vec![3]);
    }

    /// `ubatch_reserve` (:393-437).
    #[test]
    fn ubatch_reserve_geometry() {
        let mut a = allocr();
        let ub = a.ubatch_reserve(3, 2);
        assert_eq!(ub.n_tokens, 6);
        assert_eq!(ub.n_seq_tokens, 3);
        assert_eq!(ub.n_seqs, 2);
        assert_eq!(ub.n_seqs_unq, 2);
        assert!(ub.equal_seqs());
        assert_eq!(ub.seq_id_unq, vec![0, 1]);
        assert_eq!(ub.n_seq_id, vec![1; 6]);
    }

    /// The `llama_batch_ext` surface itself (llama-batch.cpp:1089-1203,
    /// fc343a84b): the test constructor, the add/set methods, and the
    /// content-type uniformity rule of `init` (:43-62).
    #[test]
    fn batch_ext_add_set_and_content_types() {
        // "build without a llama_context, used by tests" (:1064-1081)
        let mut ext = LlamaBatchExt::new(4, 8, 8, 2, None, 100, 1);

        // add_token leaves the position undefined; set_token_id validates
        // against n_vocab
        let i0 = ext.add_token(0);
        assert_eq!(i0, 0);
        assert!(ext.set_token_id(i0, 5));
        assert!(!ext.set_token_id(i0, 100)); // == n_vocab is out of range
        assert!(ext.set_token_pos(i0, &[7, 0, 0, 0]));
        assert!(ext.set_output(i0, false));

        // a second sequence on the same entry couples them
        assert!(ext.add_seq(i0, 1));

        // an invalid sequence id is rejected before anything is stored
        assert_eq!(ext.add_token(9), -3); // invalid sequence id

        // the size limit (n_tokens_max = 4) — positions continue sequence 0
        for (k, pos) in [8i32, 9, 10].iter().enumerate() {
            let i = ext.add_token(0);
            assert_eq!(i, 1 + k as i32);
            assert!(ext.set_token_id(i, 6 + k as i32));
            assert!(ext.set_token_pos(i, &[*pos, 0, 0, 0]));
        }
        assert_eq!(ext.add_token(0), -1); // size limit reached

        // a token-only batch initializes
        let mut a = allocr();
        a.init(&ext, 100, false).unwrap();
        assert_eq!(a.get_n_tokens(), 4);
        let ub = a.split_simple(8);
        // batch_idxs preserve the logical order (4453b535f)
        assert_eq!(ub.batch_idxs, vec![0, 1, 2, 3]);

        // the content-type uniformity rule: one entry without an id among
        // id-carrying entries is rejected (:48-58)
        let mut mixed = LlamaBatchExt::new(4, 8, 8, 1, None, 100, 1);
        let m0 = mixed.add_token(0);
        assert!(mixed.set_token_id(m0, 1));
        // the second entry keeps id = LLAMA_TOKEN_NULL
        let _ = mixed.add_token(0);
        let mut a2 = allocr();
        assert!(a2
            .init(&mixed, 100, false)
            .unwrap_err()
            .contains("same content types"));

        // and a batch with neither ids nor embeddings is rejected (:60-62)
        let mut none = LlamaBatchExt::new(2, 8, 8, 1, None, 100, 1);
        none.add_token(0);
        none.add_token(0);
        let mut a3 = allocr();
        assert!(a3
            .init(&none, 100, false)
            .unwrap_err()
            .contains("neither token ids nor embeddings"));
    }

    /// `set_token_embd` fixes the ext's row width on first use and rejects
    /// mismatches (llama-batch.cpp:1137-1178).
    #[test]
    fn batch_ext_embd_width_is_fixed_by_first_row() {
        let mut ext = LlamaBatchExt::new(2, 8, 8, 1, None, 100, 1);
        let i = ext.add_token(0);
        let row = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(ext.set_token_embd(i, LlamaEmbd { data: &row, n_rows: 1, n_embd: 8 }));
        assert_eq!(ext.n_embd, 8);
        // a second row of a different width is rejected
        assert!(!ext.set_token_embd(i, LlamaEmbd { data: &row[..4], n_rows: 1, n_embd: 4 }));
        // an embedding-only batch initializes and carries every row
        let mut a = allocr();
        a.init(&ext, 100, true).unwrap();
        assert_eq!(a.get_batch().embd.as_ref().unwrap().len(), 8);
    }

    /// M-RoPE batches (`n_pos_per_embd = 4`) broadcast one position across the
    /// sections and *allow* position jumps (:255-288).
    #[test]
    fn mrope_positions_broadcast_and_jump() {
        let mut a = BatchAllocr::new(4);
        let b = LlamaBatch {
            token: vec![1, 2],
            pos: Some(vec![5, 9]),
            seq_id: Some(vec![vec![0], vec![0]]),
            logits: None,
            embd: None,
        };
        a.init(&ext_of(&b, None, 1), 100, false).unwrap();
        let ub = a.split_simple(8);
        assert_eq!(ub.n_pos, 4);
        assert!(ub.is_pos_2d());
        // pos[j*n + i] = pos[i] for text batches
        assert_eq!(ub.pos, vec![5, 9, 5, 9, 5, 9, 5, 9]);
    }
}
