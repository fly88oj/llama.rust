//! Slot lifecycle + the decode loop — port of tools/server/server-context.cpp's
//! `server_slot`, `server_batch`, `pre_decode`/`update_slots`/`post_decode`
//! (:239-560, :2793-4120) and of server-queue.cpp's task queue.
//!
//! One engine thread owns the `DecodeContext`; connection threads submit tasks
//! and consume a channel of response frames. Each slot holds one sequence id
//! (= its slot index) in the shared KV cache, which is what makes the batched
//! multi-sequence decode of `DecodeContext::decode_batch` the reference's
//! `llama_decode` over several slots.
//!
//! The chat surface runs the reference's parse loop: every OAI-chat task
//! carries the autoparser's PEG arena (`ChatStreamState`, the port of
//! `task_result_state`, server-task.h:104-133), each generated token
//! re-parses the accumulated text (`update_chat_msg`, server-task.cpp:162-230)
//! and the response deltas are the `common_chat_msg_diff`s. The tool-call
//! grammar attaches lazily — sampling is unconstrained until a
//! WORD/TOKEN/PATTERN trigger fires (`common_grammar_trigger`, the port of
//! llama-grammar.cpp:1401-1443 lives in `grammar_accept_lazy` below; the
//! PATTERN regexes go through `llama::regex_lite`, the port of
//! `llama_grammar_trigger_pattern::find`, llama-grammar.cpp:378-409).
//!
//! Speculative decoding (:254-257 slot state, :2995-3075 drafting, :3742-3757
//! `common_speculative_process`, :3897-4017 sample-and-accept) follows the
//! reference's per-slot state machine; the checkpoint restore branch
//! (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`) is not ported — the port's KV cache
//! always supports partial sequence removal, which is the branch the reference
//! takes for its full-attention contexts.
//!
//! Not ported (see PARITY.md): multimodal chunks, the prompt-cache checkpoints,
//! KV shifting on context overflow, `n_cmpl > 1` child slots, and the `idle`
//! slot sleep/purge logic.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use llama::batch::LlamaBatch;
use llama::chat_tools::{chat_parse, ChatMsg, ChatMsgDiff, ChatParserParams};
use llama::context::{BatchOutput, DecodeContext, EncoderContext};
use llama::sampling::{GrammarSampler, SamplingContext, TokenDataArray};
use llama::speculative::CommonSpeculative;
use llama::vocab::Vocab;

use llama::json_schema::Json;

use crate::api::{
    common_embd_normalize, probs_from_candidates, probs_vector_to_json, string_find_partial_stop,
    task_params_to_json, usage_json_oaicompat, validate_utf8_len, GenStats, ResponseType,
    StopType, TaskParams, TokenProbs,
};
use crate::http::StreamEvent;

/// `server_task_type` (server-task.h:14-25) — the members the port runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TaskKind {
    /// `SERVER_TASK_TYPE_COMPLETION` — sample tokens
    Completion,
    /// `SERVER_TASK_TYPE_EMBEDDING` — one encode, no sampling
    Embedding,
    /// `SERVER_TASK_TYPE_SLOT_SAVE` — `POST /slots/{id}?action=save`
    SlotSave,
    /// `SERVER_TASK_TYPE_SLOT_RESTORE` — `POST /slots/{id}?action=restore`
    SlotRestore,
    /// `SERVER_TASK_TYPE_SLOT_ERASE` — `POST /slots/{id}?action=erase`
    SlotErase,
    /// `SERVER_TASK_TYPE_DECISION` — `POST /v1/systemone`: one forward pass
    /// per question, no token generated (upstream a7b94df2c)
    Decision,
}

/// `server_task::slot_action` (server-task.h) — the /slots task payload
#[derive(Clone)]
pub struct SlotAction {
    pub id_slot: i32,
    pub filename: String,
    pub filepath: String,
}

/// `enum slot_state` (server-context.cpp:100-108). The port folds the C's
/// STARTED → PROCESSING_PROMPT → DONE_PROMPT → GENERATING transitions into one
/// `update_slots` pass, so `DonePrompt` is reached by the slot that just
/// consumed the last prompt token (server-context.cpp:3593) and read by the
/// same pass's `post_decode`.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SlotState {
    Idle,
    /// assigned a task, prompt not started
    Started,
    ProcessingPrompt,
    DonePrompt,
    Generating,
}

/// `server_task` (server-task.h) — one request to decode.
#[derive(Clone)]
pub struct Task {
    pub id: i64,
    pub index: i32,
    pub kind: TaskKind,
    pub tokens: Vec<i32>,
    pub params: TaskParams,
    pub prompt_text: String,
    /// `task.slot_action` — the payload of the /slots save/restore/erase tasks
    pub slot_action: Option<SlotAction>,
    /// `SERVER_TASK_TYPE_DECISION` — the raw `/v1/systemone` body (the
    /// decision context that parses it lives on the engine)
    pub decision_body: Json,
    pub tx: mpsc::Sender<StreamEvent>,
}

impl Task {
    /// `task.n_tokens()`
    pub fn n_tokens(&self) -> i32 {
        self.tokens.len() as i32
    }
}

/// `server_slot` (server-context.cpp:239-560)
pub struct Slot {
    pub id: i32,
    pub state: SlotState,
    /// `slot.n_ctx` = `params_base.n_ctx / params_base.n_parallel`
    pub n_ctx: u32,
    /// `slot.prompt.tokens` — the tokens of this slot's sequence in the cache
    pub prompt_tokens: Vec<i32>,
    /// `slot.prompt.checkpoints` — the saved rollback points of the prompt
    /// processing (always empty until the prompt-cache checkpoint machinery
    /// lands; restored slot files may carry them, 033df86b6)
    pub checkpoints: std::collections::VecDeque<SlotCheckpoint>,
    pub task: Option<Task>,
    pub sampler: Option<SamplingContext>,
    pub grammar: Option<GrammarSampler>,

    /// `slot.stats`
    pub stats: GenStats,
    /// the token the next batch will decode (`handle_last_sampled_token`)
    pub sampled: i32,
    /// `slot.i_batch` — the batch index of the token whose logits this slot needs
    pub i_batch: i32,
    /// `slot.generated` — the generated text *and* its tokens, accumulated as
    /// the parse input (`common_chat_input`, server-context.cpp:288; 18b5f8b18)
    pub generated: llama::chat_tools::ChatInput,
    pub generated_tokens: Vec<i32>,
    /// `n_sent_text` — how much of `generated.text` the client already has
    pub n_sent_text: usize,
    pub has_next_token: bool,
    pub has_new_line: bool,
    pub stop: StopType,
    pub stopping_word: String,
    pub truncated: bool,
    /// the per-token probabilities to report (`generated_token_probs`)
    pub probs_output: Vec<TokenProbs>,
    pub n_predict_max: i32,
    /// `t_last_used` for the LRU slot search (0 = never used)
    pub t_last_used: i64,
    /// stream mode: the client was already told the request started
    pub sent_begin: bool,
    /// `task_result_state` (server-task.h:104-133) — the streaming parse state
    /// of an OAI-chat task (`slot.smpl`'s sibling `slot.task`'s
    /// `chat_parser_params`); `None` on non-chat tasks
    pub chat: Option<ChatStreamState>,
    /// the lazy-grammar trigger state (`llama_grammar`'s trigger fields,
    /// llama-grammar.h:148-153) of a `grammar_lazy` task
    pub lazy: Option<LazyTriggers>,
    // ---- speculative decoding (server-context.cpp:254-257) ----
    /// `slot.spec_draft` — the draft tokens of the current round
    pub spec_draft: Vec<i32>,
    /// `slot.spec_draft_q` — draft candidates per token in spec_draft; only
    /// draft-simple and draft-mtp fill it (upstream a7b94df2c)
    pub spec_draft_q: Vec<Vec<llama::sampling::TokenData>>,
    /// `slot.spec_i_batch` — batch indices of [sampled, draft0, ..., draftN-1]
    pub spec_i_batch: Vec<i32>,
    /// `task ? task : task_prev` (server_slot::to_json) — the finished task a
    /// released slot still reports in `GET /slots`
    pub prev: Option<PrevTask>,
}

/// the `task_prev` remnants of `server_slot::to_json` (server-context.cpp:688)
#[derive(Clone)]
pub struct PrevTask {
    pub id: i64,
    pub params: TaskParams,
    pub n_prompt_tokens: usize,
    pub n_prompt_processed: u64,
    pub n_prompt_cached: u64,
    pub n_gen: u64,
    pub has_next_token: bool,
    pub has_new_line: bool,
    pub n_remaining: i64,
    pub prompt: String,
    pub generated: String,
}

impl Slot {
    pub fn is_processing(&self) -> bool {
        self.state != SlotState::Idle
    }

    /// `slot.has_budget()`
    pub fn has_budget(&self) -> bool {
        self.n_predict_max < 0 || (self.stats.n_gen as i32) < self.n_predict_max
    }

    fn release(&mut self) {
        if self.is_processing() {
            // snapshot what `GET /slots` still reports after the release
            // (`task ? task : task_prev`, server-context.cpp:697)
            if let Some(t) = self.task.as_ref() {
                self.prev = Some(PrevTask {
                    id: t.id,
                    params: t.params.clone(),
                    n_prompt_tokens: self.prompt_tokens.len(),
                    n_prompt_processed: self.stats.n_prompt_processed,
                    n_prompt_cached: self.stats.n_prompt_cached,
                    n_gen: self.stats.n_gen,
                    has_next_token: self.has_next_token,
                    has_new_line: self.has_new_line,
                    n_remaining: if self.n_predict_max < 0 {
                        -1
                    } else {
                        self.n_predict_max as i64 - self.stats.n_gen as i64
                    },
                    prompt: t.prompt_text.clone(),
                    generated: self.generated.text.clone(),
                });
            }
            self.state = SlotState::Idle;
            self.t_last_used = now_us();
            // the cache stays (slot.prompt.tokens is kept) — server-context.cpp:538-556
            self.task = None;
            self.sampler = None;
            self.grammar = None;
            self.chat = None;
            self.lazy = None;
            self.i_batch = -1;
            // `slot_reset()`'s speculative clear (server-context.cpp:382-385)
            self.spec_draft.clear();
            self.spec_draft_q.clear();
            self.spec_i_batch.clear();
        }
    }

    /// `slot.prompt.tokens.pos_next()`
    fn pos_next(&self) -> i32 {
        self.prompt_tokens.len() as i32
    }
}

fn now_us() -> i64 {
    // `ggml_time_us` — a monotonic clock in microseconds
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let t0 = START.get_or_init(std::time::Instant::now);
    t0.elapsed().as_micros() as i64
}

// ---------------------------------------------------------------------------
// `POST /slots/{id}` support — the byte formats of the reference's slot files
// ---------------------------------------------------------------------------

/// `LLAMA_STATE_SEQ_MAGIC` = `LLAMA_FILE_MAGIC_GGSQ` (include/llama.h:43/48)
pub const STATE_SEQ_MAGIC: u32 = 0x6767_7371; // 'ggsq'
/// `LLAMA_STATE_SEQ_VERSION` (include/llama.h:49) — 4 since 210791069 (the
/// KV state blob gained the exact `n_rot_k`/`n_rot_v` rotation metadata)
pub const STATE_SEQ_VERSION: u32 = 4;
/// `LLAMA_TOKEN_NULL` (include/llama.h:39) — the marker word of the packed
/// `server_tokens` format
const TOKEN_NULL: i32 = -1;
/// `SERVER_TOKENS_STATE_VERSION` (server-common.cpp:286)
const SERVER_TOKENS_STATE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// the context-checkpoint appendix of slot save files (server-context.cpp:2586-2714,
// 033df86b6): checkpoints are appended after the llama state payload because
// they cannot be recreated from the final state alone (a recurrent state
// cannot be rewound)
// ---------------------------------------------------------------------------

/// `SLOT_CKPT_MAGIC` = "SCKP" (server-context.cpp:2589)
const SLOT_CKPT_MAGIC: u32 = 0x504b_4353;
/// `SLOT_CKPT_VERSION` (server-context.cpp:2590)
const SLOT_CKPT_VERSION: u32 = 1;
/// `params_base.n_ctx_checkpoints` (common.h:637) — the port does not parse
/// the arg, so the default cap applies
const N_CTX_CHECKPOINTS: usize = 32;

/// `common_prompt_checkpoint` (common.h:1272-1306) — one saved rollback point
/// of a slot's prompt processing. The port does not yet create checkpoints
/// while processing prompts, but slot save files written by the reference
/// carry them, and the appendix round-trips through here.
#[derive(Clone, Debug, Default)]
pub struct SlotCheckpoint {
    /// `id_task` — `-1` marks a checkpoint restored from a slot file (not
    /// created by a task; server-context.cpp:2661)
    pub id_task: i64,
    pub n_tokens: usize,
    pub pos_min: i32,
    pub pos_max: i32,
    pub data_tgt: Vec<u8>,
    pub data_dft: Vec<u8>,
    pub data_spec: Vec<u8>,
}

impl SlotCheckpoint {
    /// `common_prompt_checkpoint::size()` (common.h:1301-1303)
    #[allow(dead_code)] // used by the checkpoint machinery (not yet ported)
    pub fn size(&self) -> usize {
        self.data_tgt.len() + self.data_dft.len() + self.data_spec.len()
    }
}

/// `ckpt_read` (server-context.cpp:2592-2597) over an in-memory cursor —
/// every read advances `n_read` like the C's byte counter
fn ckpt_read<'a>(rd: &mut &'a [u8], n: usize, n_read: &mut usize) -> Option<&'a [u8]> {
    if rd.len() < n {
        return None;
    }
    let (head, rest) = rd.split_at(n);
    *rd = rest;
    *n_read += n;
    Some(head)
}

/// `ckpt_read_buf` (server-context.cpp:2599-2607): read the u64 length then
/// the blob — the size is checked against the bytes left in the appendix
/// before allocating (the size field may be corrupted)
fn ckpt_read_buf(rd: &mut &[u8], n_avail: usize, n_read: &mut usize) -> Option<Vec<u8>> {
    let n = u64::from_le_bytes(ckpt_read(rd, 8, n_read)?.try_into().ok()?) as usize;
    if n > n_avail - *n_read {
        return None;
    }
    let buf = ckpt_read(rd, n, n_read)?.to_vec();
    Some(buf)
}

/// the `decision.label_groups` reduction of `send_decision`
/// (server-context.cpp:2347-2355, 88dcc460d): one output per group, the max
/// of the group's label scores
fn reduce_label_groups(scores: &mut Vec<f32>, spec: &crate::server_decision::DecisionSpec) {
    if spec.label_groups.is_empty() {
        return;
    }
    let mut reduced: Vec<f32> = Vec::new();
    let mut i = 0usize;
    for &n in &spec.label_groups {
        assert!(n > 0 && i + n as usize <= scores.len());
        let end = i + n as usize;
        reduced.push(scores[i..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        i = end;
    }
    *scores = reduced;
}

/// `ckpt_write_buf` (server-context.cpp:2616-2621): the u64 length then the blob
fn ckpt_write_buf(out: &mut Vec<u8>, buf: &[u8], n_written: &mut usize) {
    let n = buf.len() as u64;
    out.extend_from_slice(&n.to_le_bytes());
    *n_written += 8;
    if n > 0 {
        out.extend_from_slice(buf);
        *n_written += buf.len();
    }
}

/// `save_slot_checkpoints`'s serialization (server-context.cpp:2623-2653):
/// `[u32 magic][u32 version][u32 count]` then per checkpoint
/// `[u64 n_tokens][i32 pos_min][i32 pos_max]` and three length-prefixed blobs
/// (target, draft, speculative). Writes nothing for an empty list.
fn slot_checkpoints_appendix(
    checkpoints: &std::collections::VecDeque<SlotCheckpoint>,
    out: &mut Vec<u8>,
) -> usize {
    if checkpoints.is_empty() {
        return 0;
    }
    let mut n_written = 0usize;
    out.extend_from_slice(&SLOT_CKPT_MAGIC.to_le_bytes());
    out.extend_from_slice(&SLOT_CKPT_VERSION.to_le_bytes());
    out.extend_from_slice(&(checkpoints.len() as u32).to_le_bytes());
    n_written += 12;
    for cur in checkpoints {
        out.extend_from_slice(&(cur.n_tokens as u64).to_le_bytes());
        out.extend_from_slice(&cur.pos_min.to_le_bytes());
        out.extend_from_slice(&cur.pos_max.to_le_bytes());
        n_written += 16;
        ckpt_write_buf(out, &cur.data_tgt, &mut n_written);
        ckpt_write_buf(out, &cur.data_dft, &mut n_written);
        ckpt_write_buf(out, &cur.data_spec, &mut n_written);
    }
    n_written
}

/// `load_slot_checkpoints`'s parse (server-context.cpp:2655-2714): returns the
/// parsed checkpoints (marked `id_task = -1`, capped at the last
/// `n_ctx_checkpoints`) and the bytes consumed; 0 consumed means there is no
/// usable appendix (absent, wrong magic, damaged, or an empty target state —
/// a damaged appendix is ignored, never fatal).
fn parse_slot_checkpoints_appendix(
    bytes: &[u8],
    offset: usize,
    slot_id: i32,
) -> (std::collections::VecDeque<SlotCheckpoint>, usize) {
    if offset > bytes.len() {
        return (Default::default(), 0);
    }
    let n_avail = bytes.len() - offset; // bytes after the llama state payload
    let mut rd: &[u8] = &bytes[offset..];
    let mut n_read = 0usize;
    let take = |rd: &mut &[u8], n: usize, n_read: &mut usize| -> Option<Vec<u8>> {
        ckpt_read(rd, n, n_read).map(|s| s.to_vec())
    };
    let read_u32 = |rd: &mut &[u8], n_read: &mut usize| -> Option<u32> {
        take(rd, 4, n_read).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(u32::from_le_bytes)
    };
    let read_i32 = |rd: &mut &[u8], n_read: &mut usize| -> Option<i32> {
        take(rd, 4, n_read).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(i32::from_le_bytes)
    };
    let read_u64 = |rd: &mut &[u8], n_read: &mut usize| -> Option<u64> {
        take(rd, 8, n_read).and_then(|b| <[u8; 8]>::try_from(b).ok()).map(u64::from_le_bytes)
    };
    let invalid = |what: &str| {
        eprintln!("slot {slot_id}: {what} context checkpoint appendix - ignored");
    };
    match read_u32(&mut rd, &mut n_read) {
        Some(m) if m == SLOT_CKPT_MAGIC => {}
        // no appendix at all — not an error, nothing consumed
        _ => return (Default::default(), 0),
    }
    if !matches!(read_u32(&mut rd, &mut n_read), Some(v) if v == SLOT_CKPT_VERSION) {
        invalid("invalid");
        return (Default::default(), 0);
    }
    let Some(count) = read_u32(&mut rd, &mut n_read) else {
        invalid("invalid");
        return (Default::default(), 0);
    };
    let mut checkpoints: std::collections::VecDeque<SlotCheckpoint> = Default::default();
    for _ in 0..count {
        let mut cur = SlotCheckpoint::default();
        cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
        let ok = (|| -> Option<()> {
            cur.n_tokens = read_u64(&mut rd, &mut n_read)? as usize;
            cur.pos_min = read_i32(&mut rd, &mut n_read)?;
            cur.pos_max = read_i32(&mut rd, &mut n_read)?;
            cur.data_tgt = ckpt_read_buf(&mut rd, n_avail, &mut n_read)?;
            cur.data_dft = ckpt_read_buf(&mut rd, n_avail, &mut n_read)?;
            cur.data_spec = ckpt_read_buf(&mut rd, n_avail, &mut n_read)?;
            Some(())
        })();
        if ok.is_none() {
            invalid("truncated");
            return (Default::default(), 0);
        }
        // a saved checkpoint always holds a target state - an empty blob
        // would roll back without restoring anything
        if cur.data_tgt.is_empty() {
            invalid("invalid");
            return (Default::default(), 0);
        }
        checkpoints.push_back(cur);
        if checkpoints.len() > N_CTX_CHECKPOINTS {
            checkpoints.pop_front();
        }
    }
    (checkpoints, n_read)
}

/// `server_tokens_state_writer::take`'s trailing pad — the byte stream grows
/// to a multiple of `sizeof(llama_token)`
fn pad_to_token(mut v: Vec<u8>) -> Vec<u8> {
    while v.len() % 4 != 0 {
        v.push(0);
    }
    v
}

/// `server_tokens::serialize` (server-common.cpp:570-588) — the port has no
/// mtmd, so the media maps are always empty:
/// `[i32 LLAMA_TOKEN_NULL][u32 VERSION][u32 n][i32 tokens][u32 0 media keys]`,
/// padded to a 4-byte multiple.
fn server_tokens_serialize(tokens: &[i32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(12 + tokens.len() * 4);
    v.extend_from_slice(&TOKEN_NULL.to_le_bytes());
    v.extend_from_slice(&SERVER_TOKENS_STATE_VERSION.to_le_bytes());
    v.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        v.extend_from_slice(&t.to_le_bytes());
    }
    // media_keys: empty (no mmproj → no media chunks)
    v.extend_from_slice(&0u32.to_le_bytes());
    pad_to_token(v)
}

/// `server_tokens::deserialize` (server-common.cpp:592-633) — a payload whose
/// first word is not `LLAMA_TOKEN_NULL` is "a plain token list, as written by
/// older versions". Returns the token vector; the errors are the C's texts.
fn server_tokens_deserialize(packed: &[i32]) -> Result<Vec<i32>, String> {
    if packed.is_empty() || packed[0] != TOKEN_NULL {
        return Ok(packed.to_vec());
    }
    let mut rd = &packed[1..];
    let take = |rd: &mut &[i32]| -> Option<i32> {
        let (h, rest) = rd.split_first()?;
        *rd = rest;
        Some(*h)
    };
    let Some(version) = take(&mut rd) else {
        return Err("Unexpected end of server tokens state".into());
    };
    if version as u32 != SERVER_TOKENS_STATE_VERSION {
        return Err("Unsupported server tokens state version".into());
    }
    let Some(n) = take(&mut rd).map(|v| v as u32) else {
        return Err("Unexpected end of server tokens state".into());
    };
    if n as usize > rd.len() {
        return Err("Unexpected end of server tokens state".into());
    }
    let tokens = rd[..n as usize].to_vec();
    rd = &rd[n as usize..];
    let Some(n_keys) = take(&mut rd).map(|v| v as u32) else {
        return Err("Unexpected end of server tokens state".into());
    };
    // `!media_keys.empty() && !has_mtmd` — the port never has an mmproj
    if n_keys > 0 {
        return Err("Cannot restore media tokens without an mmproj".into());
    }
    // "Trailing data in server tokens state" (the reader's final check)
    if !rd.is_empty() {
        return Err("Trailing data in server tokens state".into());
    }
    Ok(tokens)
}

/// `fs_validate_filename(filename, allow_subdirs = false)`
/// (common/common.cpp:825-897): UTF-8 sanity, the forbidden codepoints, the
/// Windows-trimming cases and the path traversal guards.
pub fn fs_validate_filename(filename: &str) -> bool {
    if filename.is_empty() || filename.len() > 255 {
        return false;
    }
    for c in filename.chars() {
        let c = c as u32;
        if c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (0x80..=0x9F).contains(&c) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (0xD800..=0xDFFF).contains(&c) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || matches!(c, 0x3A | 0x2A | 0x3F | 0x22 | 0x3C | 0x3E | 0x7C) // : * ? " < > |
            // allow_subdirs == false: reject path separators
            || c == '/' as u32
            || c == '\\' as u32
        {
            return false;
        }
    }
    // leading/trailing ' ' and trailing '.' are stripped on Windows
    if filename.starts_with(' ') || filename.ends_with(' ') || filename.ends_with('.') {
        return false;
    }
    if filename.contains("..") || filename == "." {
        return false;
    }
    true
}

/// the `timings` sub-object of `server_task_result_slot_save_load::to_json`
fn save_ms_json(key: &str, t_ms: f64) -> Json {
    Json::Object(vec![(key.into(), Json::Double(t_ms))])
}

// ---------------------------------------------------------------------------
// `task_result_state` (server-task.h:104-133) — the streaming chat-parse state
// the reference keeps in its response reader (`server_response_reader::states`,
// server-queue.h:213); the port keeps it on the slot because its engine thread
// renders the final SSE frames.
// ---------------------------------------------------------------------------

/// the parse state of one OAI-chat task
pub struct ChatStreamState {
    /// `chat_parser_params` — the autoparser's arena + the generation prompt
    pub parser: ChatParserParams,
    /// `chat_msg` — the previous parsed message (the diff base)
    pub msg: ChatMsg,
    /// `generated_input` — the text+tokens accumulated through
    /// `update_chat_msg` (`common_chat_input`, server-task.h:111; 18b5f8b18)
    /// (only what was *sent* — the C adds each partial's content chunk)
    pub generated_input: llama::chat_tools::ChatInput,
    /// `generated_tool_call_ids` — the ids already handed out
    pub generated_tool_call_ids: Vec<String>,
}

impl ChatStreamState {
    /// `task_result_state`'s constructor (server-task.cpp:141-159) from the
    /// task's chat metadata; `is_continuation` is always false (the port does
    /// not splice continuation prompts, see PARITY.md). The generation prompt
    /// is token-aligned via `common_chat_input_tokenize` — the schema's
    /// `generation_prompt` handler does this at request-parse time when a
    /// vocab is available (server-schema.cpp:314-321, 18b5f8b18).
    fn new(params: &TaskParams, vocab: &Vocab) -> Result<ChatStreamState, String> {
        let mut parser = ChatParserParams::default();
        parser.format = chat_format_from_name(&params.chat_format)?;
        parser.generation_prompt = llama::chat_tools::chat_input_tokenize(vocab, &params.generation_prompt);
        if !params.chat_parser.is_empty() {
            // `chat_parser_params.parser.load(data.at("chat_parser"))`
            // (server-schema.cpp:318-327)
            parser.parser = llama::peg::PegArena::default();
            parser
                .parser
                .load(&params.chat_parser)
                .map_err(|e| format!("Failed to initialize chat parser: {e}"))?;
        }
        Ok(ChatStreamState {
            parser,
            msg: ChatMsg::default(),
            generated_input: llama::chat_tools::ChatInput::default(),
            generated_tool_call_ids: Vec::new(),
        })
    }

    /// `task_result_state::update_chat_msg` (server-task.cpp:162-230) with
    /// `filter_tool_calls = false` — re-parse the accumulated text, hand out
    /// tool-call ids, and diff against the previous message. Since 18b5f8b18
    /// the accumulated parse input carries the generating tokens.
    fn update_chat_msg(
        &mut self,
        added: &llama::chat_tools::ChatInput,
        is_partial: bool,
    ) -> Result<Vec<ChatMsgDiff>, String> {
        self.generated_input.append_chunk(added);
        let msg_prv = self.msg.clone();
        //SRV_DBG("Parsing chat message: %s\n", …)
        let new_msg = chat_parse(&self.generated_input, is_partial, &self.parser)?;
        let mut diffs = Vec::new();
        if !new_msg.empty() {
            let mut new_msg = new_msg;
            // `new_msg.set_tool_call_ids(generated_tool_call_ids,
            // gen_tool_call_id)` — `gen_tool_call_id()` is a bare
            // `random_string()` (server-common.cpp:130-132)
            let mut ids = std::mem::take(&mut self.generated_tool_call_ids);
            new_msg.set_tool_call_ids(&mut ids, || crate::api::random_string(32));
            self.generated_tool_call_ids = ids;
            self.msg = new_msg;
            diffs = ChatMsgDiff::compute_diffs(&msg_prv, &self.msg)?;
        }
        Ok(diffs)
    }
}

/// `static_cast<common_chat_format>(data.at("chat_format"))` — the task params
/// carry the format *name* (`generation_settings.chat_format`); map it back.
fn chat_format_from_name(name: &str) -> Result<llama::chat_tools::ChatFormat, String> {
    use llama::chat_tools::ChatFormat;
    Ok(match name {
        "Content-only" => ChatFormat::ContentOnly,
        "peg-simple" => ChatFormat::PegSimple,
        "peg-native" => ChatFormat::PegNative,
        "peg-gemma4" => ChatFormat::PegGemma4,
        "peg-minimax-m3" => ChatFormat::PegMinimaxM3,
        other => return Err(format!("unknown chat format: {other}")),
    })
}

/// `server_chat_msg_diff_to_json_oaicompat` (server-chat.cpp:621-649) — the
/// SSE `delta` object of one `common_chat_msg_diff`.
fn chat_msg_diff_to_json(diff: &ChatMsgDiff) -> Json {
    let mut delta: Vec<(String, Json)> = Vec::new();
    if !diff.reasoning_content_delta.is_empty() {
        delta.push(("reasoning_content".into(), Json::String(diff.reasoning_content_delta.clone())));
    }
    if !diff.content_delta.is_empty() {
        delta.push(("content".into(), Json::String(diff.content_delta.clone())));
    }
    if diff.tool_call_index != ChatMsgDiff::NPOS {
        let mut tool_call: Vec<(String, Json)> =
            vec![("index".into(), Json::Int(diff.tool_call_index as i64))];
        if !diff.tool_call_delta.id.is_empty() {
            tool_call.push(("id".into(), Json::String(diff.tool_call_delta.id.clone())));
            tool_call.push(("type".into(), Json::String("function".into())));
        }
        if !diff.tool_call_delta.name.is_empty() || !diff.tool_call_delta.arguments.is_empty() {
            let mut function: Vec<(String, Json)> = Vec::new();
            if !diff.tool_call_delta.name.is_empty() {
                function.push(("name".into(), Json::String(diff.tool_call_delta.name.clone())));
            }
            if !diff.tool_call_delta.arguments.is_empty() {
                function.push((
                    "arguments".into(),
                    Json::String(diff.tool_call_delta.arguments.clone()),
                ));
            }
            tool_call.push(("function".into(), Json::Object(function)));
        }
        delta.push(("tool_calls".into(), Json::Array(vec![Json::Object(tool_call)])));
    }
    Json::Object(delta)
}

// ---------------------------------------------------------------------------
// the lazy tool-call grammar — `llama_grammar_accept_impl`'s awaiting-trigger
// branch (llama-grammar.cpp:1401-1443). The port's `Grammar` carries the
// `awaiting_trigger` flag (`apply` is a no-op while it is set); the trigger
// match itself runs on a byte buffer the engine owns — the C buffers raw
// `std::string` bytes, and a partial multi-byte token piece is not valid
// UTF-8, so the library's `String` field cannot hold it. The C compiles every
// WORD/PATTERN/PATTERN_FULL trigger into an ECMAScript `std::regex`
// (`common_sampler_init`'s trigger loop, sampling.cpp:222-256: a WORD is
// `regex_escape(word)`, a PATTERN is taken verbatim, a PATTERN_FULL is
// wrapped in `^…$`); `llama_grammar_trigger_pattern::find`
// (llama-grammar.cpp:378-409) returns the first non-empty capture group's
// position, else the match start, trying a full match first when the raw
// pattern begins `^` and ends `$`. The port matches WORDs as literal bytes
// (a regex-escaped word searched leftmost is its earliest occurrence, no
// groups → the word's position) and PATTERN/PATTERN_FULL through
// `llama::regex_lite`, which reproduces `find` exactly.
// ---------------------------------------------------------------------------

/// the trigger fields of `llama_grammar` (llama-grammar.h:148-153) the engine
/// owns while the grammar awaits its trigger
pub struct LazyTriggers {
    /// `trigger_tokens` — TOKEN triggers (single-token words)
    pub tokens: Vec<i32>,
    /// `trigger_patterns` in declaration order — WORD triggers keep their
    /// literal word (the C's `regex_escape(word)`), PATTERN/PATTERN_FULL
    /// triggers carry the compiled regex (`std::regex`, llama-grammar.cpp:1297)
    pub patterns: Vec<LazyPattern>,
    /// `trigger_buffer` — the output buffered while awaiting
    pub buffer: Vec<u8>,
    /// `trigger_buffer_positions` — (token, [start, end) in the buffer)
    pub positions: Vec<(i32, (usize, usize))>,
}

/// one entry of the C's `trigger_patterns` (sampling.cpp:222-256)
pub enum LazyPattern {
    /// COMMON_GRAMMAR_TRIGGER_TYPE_WORD — `regex_escape(word)` searched
    /// leftmost = the earliest occurrence of the literal word
    Word(Vec<u8>),
    /// COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN (verbatim) or
    /// COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL (pre-anchored `^…$`,
    /// sampling.cpp:235-245)
    Regex(llama::regex_lite::RegexLite),
}

impl LazyPattern {
    /// `llama_grammar_trigger_pattern::find` (llama-grammar.cpp:378-409)
    fn find(&self, buffer: &[u8]) -> Option<usize> {
        match self {
            LazyPattern::Word(word) => find_bytes(buffer, word),
            LazyPattern::Regex(re) => re.find(buffer),
        }
    }
}

/// `std::string::find` over bytes
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// the PATTERN_FULL anchoring of `common_sampler_init` (sampling.cpp:235-245):
/// `"^$"` for the empty pattern, else prepend `^` / append `$` unless the raw
/// pattern already begins/ends with them
fn anchor_pattern_full(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let pre = if pattern.starts_with('^') { "" } else { "^" };
    let post = if pattern.ends_with('$') { "" } else { "$" };
    format!("{pre}{pattern}{post}")
}

/// `common_sampler_sample` (sampling.cpp:594-676) with the lazy-grammar accept
/// of `llama_grammar_accept_impl` (llama-grammar.cpp:1398-1455): while the
/// grammar awaits its trigger, `apply` constrains nothing and the accepted
/// tokens accumulate in the trigger buffer until a trigger fires (then the
/// overlapping tokens replay into the grammar).
fn sample_with_lazy_grammar(
    smpl: &mut SamplingContext,
    logits: &[f32],
    grammar: &mut GrammarSampler,
    triggers: &mut LazyTriggers,
) -> Result<i32, String> {
    use llama::sampling::TokenData;
    let mut cur_p = TokenDataArray::from_logits(logits);
    smpl.chain.apply(&mut cur_p);
    let id = cur_p.data[cur_p.selected as usize].id;
    smpl.cur = cur_p.data[..cur_p.size].to_vec();

    // the sampled id is checked against the grammar as a single candidate
    // (`llama_sampler_sample`'s grammar_first=false flow)
    let mut single = TokenDataArray {
        data: vec![TokenData { id, logit: 1.0, p: 0.0 }],
        size: 1,
        selected: -1,
        sorted: false,
    };
    grammar.apply_to(&mut single);
    let token = if single.data[0].logit != f32::NEG_INFINITY {
        id
    } else {
        // resample: grammar first, then the chain
        let mut cur_p = TokenDataArray::from_logits(logits);
        grammar.apply_to(&mut cur_p);
        smpl.chain.apply(&mut cur_p);
        let t = cur_p.data[cur_p.selected as usize].id;
        smpl.cur = cur_p.data[..cur_p.size].to_vec();
        t
    };

    smpl.chain.accept(token);
    grammar_accept_lazy(grammar, triggers, token)?;
    smpl.prev.push_back(token);
    Ok(token)
}

/// `llama_grammar_accept_impl` (llama-grammar.cpp:1398-1455) — the lazy
/// awaiting-trigger branch plus the plain accept once triggered.
fn grammar_accept_lazy(
    grammar: &mut GrammarSampler,
    triggers: &mut LazyTriggers,
    token: i32,
) -> Result<(), String> {
    use llama::grammar::GrammarVocab as _;
    let piece = grammar.vocab.token_piece(token).to_vec();
    let is_eog = grammar.vocab.is_eog(token);
    let g = &mut grammar.grammar;

    if g.awaiting_trigger {
        // a TOKEN trigger fires on the token itself, which is then accepted
        // as part of the grammar (llama-grammar.cpp:1403-1411)
        if triggers.tokens.contains(&token) {
            g.awaiting_trigger = false;
            triggers.buffer.clear();
            return g.accept_token(token, &piece);
        }
        // otherwise the piece is buffered (llama-grammar.cpp:1412-1415)
        let position = (triggers.buffer.len(), triggers.buffer.len() + piece.len());
        triggers.positions.push((token, position));
        triggers.buffer.extend_from_slice(&piece);

        // the trigger patterns in declaration order — the first one with a
        // match fires, at its own capture-group position (the C loop of
        // llama-grammar.cpp:1415-1439: `find` per pattern, fire on the first
        // hit, NOT the minimum position across patterns)
        let mut fire: Option<usize> = None;
        for p in &triggers.patterns {
            if let Some(start) = p.find(&triggers.buffer) {
                fire = Some(start);
                break;
            }
        }
        if let Some(start) = fire {
            g.awaiting_trigger = false;
            // "replay tokens that overlap with [start, end)" — a token that
            // straddles the trigger position replays only its tail
            // (llama-grammar.cpp:1422-1438)
            let positions = std::mem::take(&mut triggers.positions);
            for (tok, (tok_start, tok_end)) in &positions {
                if *tok_end <= start {
                    continue;
                }
                let piece_start = (*tok_start).max(start);
                let piece = triggers.buffer[piece_start..*tok_end].to_vec();
                g.accept_token(*tok, &piece)?;
            }
            triggers.buffer.clear();
            triggers.positions.clear();
        }
        return Ok(());
    }

    // the non-awaiting accept (llama-grammar.cpp:1446-1455)
    if is_eog {
        for stack in &g.stacks {
            if stack.is_empty() {
                return Ok(());
            }
        }
        // C: GGML_ABORT("fatal error")
        return Err("fatal error: EOG token accepted by an incomplete grammar".to_string());
    }
    g.accept_token(token, &piece)
}

/// Which llama context the server holds — a decoder (`llama_decode`, the
/// reference's `ctx_tgt`), an encoder-only model (`llama_encode`, reached
/// when the server starts with `--embeddings` on a BERT-family file), or the
/// lfm2 d1-omni decision model (`lfm2.decision.block_count > 0`, a657f7e98).
pub enum Core {
    Decode(DecodeContext),
    Encode(EncoderContext),
    /// `graph_decision` of lfm2.cpp:415-518 — the null-memory d1-omni trunk
    /// the reference drives through `llama_decode`'s "no memory -> encode()"
    /// reroute (llama-context.cpp:1729-1732). The port's `DecodeContext`
    /// builds the hybrid-memory decode graph, so the decision model rides its
    /// own stateless core: one `build_lfm2_decision_forward` pass per request
    /// (the same driver as `tests/lfm2_decision_e2e.rs`).
    Decision(D1OmniCore),
}

/// the d1-omni forward driver — `tests/lfm2_decision_e2e.rs`'s `D1Driver`
/// lifted into the engine: the no-cache trunk (media-aware masks, text-only
/// in the port), the [3, n_tokens] score output of the head.
pub struct D1OmniCore {
    pub gctx: ggml::Context,
    pub w: llama::graph_arch::Lfm2DecisionModelWeights,
    pub p: llama::graph_arch::Lfm2DecisionParams,
    /// `hparams.n_embd_out()` = the question-type count (3,
    /// lfm2.cpp:15-17 `N_DECISION_TYPES`)
    pub n_embd_out: usize,
    pub n_threads: usize,
}

impl D1OmniCore {
    /// one decision pass — the text-only mask shapes (single sequence, no
    /// media: everything visible, plain neighbor taps). Returns the flat
    /// score rows `[n_tokens][n_embd_out]` (ggml's [3, T] column-major
    /// layout is token-major in memory — exactly `llama_get_embeddings_ith`
    /// rows, llama-context.cpp's `outputs_embd`).
    fn decide(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
        use ggml::types::GgmlType;
        use llama::graph_arch;

        let n = tokens.len();
        let t = n as i64;
        let watermark = self.gctx.mark();
        self.gctx.reset_graph_to(watermark);

        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let enc_mask = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let head_mask = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let conv_left = self.gctx.new_tensor_2d(GgmlType::F32, 1, t);
        let conv_right = self.gctx.new_tensor_2d(GgmlType::F32, 1, t);
        for x in [tokens_t, pos_t, enc_mask, head_mask, conv_left, conv_right] {
            self.gctx.arena_resize_tensor(x);
        }
        self.gctx
            .with_i32_mut(tokens_t, |q| q.copy_from_slice(tokens))
            .ok_or("tensor access failed".to_string())?;
        self.gctx
            .with_i32_mut(pos_t, |q| {
                for (k, v) in q.iter_mut().enumerate() {
                    *v = k as i32;
                }
            })
            .ok_or("tensor access failed".to_string())?;
        // text-only single sequence: every position visible in both masks
        // (lfm2.cpp:352-384's set_input with no media rows)
        self.gctx.with_f32_mut(enc_mask, |q| q.fill(0.0)).ok_or("tensor access failed".to_string())?;
        self.gctx.with_f32_mut(head_mask, |q| q.fill(0.0)).ok_or("tensor access failed".to_string())?;
        // the neighbor taps: left[i+1] = right[i] = 1 within the sequence
        // (lfm2_conv_mask_rule, graph_arch.rs)
        self.gctx
            .with_f32_mut(conv_left, |q| {
                q[0] = 0.0;
                for v in q.iter_mut().skip(1) {
                    *v = 1.0;
                }
            })
            .ok_or("tensor access failed".to_string())?;
        self.gctx
            .with_f32_mut(conv_right, |q| {
                for v in q.iter_mut().take(n.saturating_sub(1)) {
                    *v = 1.0;
                }
                if n > 0 {
                    q[n - 1] = 0.0;
                }
            })
            .ok_or("tensor access failed".to_string())?;

        let inp = graph_arch::Lfm2DecisionInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask_enc: enc_mask,
            kq_mask_head: head_mask,
            conv_left,
            conv_right,
            out_ids: None,
        };
        let result =
            graph_arch::build_lfm2_decision_forward(&mut self.gctx, &self.w, &self.p, &inp, n);
        let scores = result.scores;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, self.n_threads);
        let bytes = self
            .gctx
            .data_bytes(scores)
            .ok_or("score tensor access failed".to_string())?;
        // the [3, T] F32 scores, token-major in memory — T rows of
        // n_embd_out (the `llama_get_embeddings_ith` layout)
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        debug_assert_eq!(flat.len(), n * self.n_embd_out);
        Ok(flat)
    }
}

/// `server_context_impl` — the model side of the server.
pub struct Engine {
    pub core: Core,
    /// shared with the connection threads (tokenize/detokenize) — `Vocab` has
    /// no interior mutability
    pub vocab: Arc<Vocab>,
    pub n_vocab: i32,
    pub slots: Vec<Slot>,
    pub pending: VecDeque<Task>,
    /// `queue_tasks.defer` (server-queue.cpp) — the /slots tasks whose slot is
    /// still processing wait here for the next engine pass
    pub deferred: VecDeque<Task>,
    /// `params_base.n_batch` — the logical batch cap of one decode call
    pub n_batch: usize,
    /// `params_base.n_predict`
    pub n_predict_default: i32,
    pub model_name: String,
    pub model_path: String,
    /// `--no-context-shift`: with `--context-shift` on (common/arg.cpp:
    /// 1737-1741, default **off** at this revision — common.h:571) a slot that
    /// runs out of context shifts its KV cells and continues
    /// (server-context.cpp:2909-2972); off, generation stops at the boundary
    /// (process_token's `truncated` + `stop_type=limit`,
    /// server-context.cpp:1887-1894).
    pub ctx_shift: bool,
    /// `params_base.n_keep` (common.h:453, `--keep N`) — the request schema's
    /// `n_keep` default (server-schema.cpp:529 `params.n_keep =
    /// params_base.n_keep`).
    pub n_keep_default: i32,
    /// `params_base.n_ctx_checkpoints` (common.h:637, `--ctx-checkpoints`)
    /// — the per-slot cap of the prompt-cache checkpoint list; 0 disables
    /// the machinery (`do_checkpoint = n_ctx_checkpoints > 0`,
    /// server-context.cpp:3865)
    pub n_ctx_checkpoints: i32,
    /// `params_base.checkpoint_min_step` (common.h:639,
    /// `--checkpoint-min-step`) — the minimum token spacing between
    /// checkpoints (0 = no minimum)
    pub checkpoint_min_step: i32,
    /// `n_swa` of the target memory (server-context.cpp:1337
    /// `swa_full ? 0 : llama_model_n_swa(model_tgt)`; the port has no
    /// `--swa-full` flag, so it is the raw hparams window). Drives the
    /// rollback threshold `pos_min_thold = max(0, pos_next - n_swa - …)`
    /// (server-context.cpp:3674)
    pub n_swa: i32,
    /// the memory cannot roll back a partial sequence —
    /// `common_context_can_seq_rm` resolved to FULL/RS (common.cpp:1553-1596;
    /// a recurrent half without snapshots refuses `seq_rm`, a rollback
    /// ring/dsv4 bounds it, llama-memory-recurrent.cpp:180-210). Checkpoints
    /// are only created for such memories, or when `n_swa > 0`
    /// (server-context.cpp:3870-3878)
    pub seq_rm_bounded: bool,
    pub shutdown: Arc<AtomicBool>,
    /// `server_context_impl::spec` (server-context.cpp:1259-1300) — the
    /// speculator of `common_speculative_init`, shared by every slot (`None`
    /// = `slot.can_speculate()` false).
    pub spec: Option<CommonSpeculative>,
    /// `server_context_impl::decision` (server-context.cpp:886) — the
    /// `/v1/systemone` model context, `None` for a non-decision model
    pub decision: Option<crate::server_decision::ServerDecisionContext>,
    /// `llama_pooling_type(ctx_tgt)` — resolved at load
    /// (llama-context.cpp:216-222)
    pub pooling: llama::hparams::LlamaPoolingType,

    // `/props` (`server_context_meta`, server-context.h:60-120)
    pub ftype: String,
    pub media_marker: String,
    pub chat_template: String,
    pub chat_template_caps: Json,
    pub chat_template_tool_use: String,
    pub bos_token: String,
    pub eos_token: String,
    /// `enable_thinking` of the chat params (server-context.cpp:1448-1453)
    pub enable_thinking: bool,
}

impl Engine {
    /// the decode context (`ctx_tgt`) — `None` on an encoder-only server
    fn dctx(&mut self) -> Option<&mut DecodeContext> {
        match &mut self.core {
            Core::Decode(d) => Some(d),
            Core::Encode(_) | Core::Decision(_) => None,
        }
    }
}

/// `queue_tasks` (server-queue.cpp) — the task queue shared between the
/// connection threads and the engine. The engine itself is *not* shared: it
/// owns the `DecodeContext` and the per-slot sampler chains, which are not
/// `Send` (llama::sampling's `Box<dyn Sampler>`), so the engine runs on the
/// thread that created the model and the handlers only see the queue, the
/// vocabulary and the precomputed `/props`.
pub struct Queue {
    tasks: Mutex<VecDeque<Task>>,
    cv: Condvar,
}

impl Queue {
    pub fn new() -> Self {
        Queue { tasks: Mutex::new(VecDeque::new()), cv: Condvar::new() }
    }

    /// `queue_tasks.post(task)` — enqueue and wake the engine.
    pub fn submit(&self, task: Task) {
        let mut q = self.tasks.lock().unwrap();
        q.push_back(task);
        drop(q);
        self.cv.notify_all();
    }

    /// Take every queued task (the engine's `on_new_task` drain).
    pub fn drain(&self) -> Vec<Task> {
        let mut q = self.tasks.lock().unwrap();
        q.drain(..).collect()
    }

    /// `queue_tasks.wait()` — block until a task arrives or the timeout (the
    /// timeout lets the loop notice the shutdown flag).
    pub fn wait(&self, timeout: std::time::Duration) {
        let q = self.tasks.lock().unwrap();
        let _ = self.cv.wait_timeout(q, timeout);
    }
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

/// What the connection threads can see (`Send + Sync`: `llama::vocab::Vocab`
/// holds no interior mutability, so it can be shared).
pub struct Server {
    pub queue: Queue,
    pub vocab: Arc<Vocab>,
    /// the `/props` body, computed once the model is up
    pub props: String,
    pub n_slots: usize,
    /// `/health`'s `slots_processing`
    pub n_processing: std::sync::atomic::AtomicUsize,
    /// `params_base.n_predict` — the default the request schema starts from
    pub n_predict_default: i32,
    /// `params_base.n_keep` (`--keep N`) — the request schema's `n_keep`
    /// default (server-schema.cpp:529)
    pub n_keep_default: i32,
    /// `params_base.speculative.types` rendered with
    /// `common_speculative_type_name_str` — what a request's
    /// `generation_settings.speculative.types` inherits (server-task.cpp:81)
    pub speculative_types: String,
    /// task ids (`server_response_reader::get_new_id`)
    pub next_task_id: std::sync::atomic::AtomicI64,
    // ---- the chat / embedding surface the handlers read ----
    /// `params.embedding` — whether the embedding endpoints are enabled
    pub embeddings: bool,
    /// `llama_pooling_type(ctx_tgt)` (resolved, llama-context.cpp:216-222)
    pub pooling: llama::hparams::LlamaPoolingType,
    /// the chat-template inputs of `/v1/chat/completions`
    /// (`meta->chat_params.tmpls`, server-common.h:83-110 — the product of
    /// `common_chat_templates_init`). The library's `ChatTemplates` holds
    /// mini-jinja values with `Rc`, so it is not `Sync`; the handler thread
    /// rebuilds it from these inputs (the reference re-runs the differential
    /// analysis per request anyway, chat.cpp:1335-1340)
    pub chat_templates_init: llama::chat_tools::ChatTemplatesInit,
    /// `opt.enable_thinking` of the chat params (server-context.cpp:1448-1453)
    pub enable_thinking: bool,
    /// `/models`'s model entry (`server_context_meta`, server-context.h)
    pub model_meta: crate::ModelMeta,
    /// `params.slot_save_path` (`--slot-save-path`, arg.cpp:3610-3620) —
    /// empty disables the /slots action endpoint
    pub slot_save_path: String,
    /// the slot status table of `GET /slots` (`server_slot::to_json`,
    /// server-task.cpp) — only mutated on the engine thread, published as a
    /// string snapshot after each pass
    pub slots_json: std::sync::RwLock<String>,
}

impl Server {
    pub fn new_task_id(&self) -> i64 {
        self.next_task_id.fetch_add(1, Ordering::Relaxed) + 1
    }
}

impl Engine {
    fn has_work(&self) -> bool {
        !self.pending.is_empty() || self.slots.iter().any(|s| s.is_processing())
    }

    /// The engine loop — `queue_tasks.on_new_task` + `on_update_slots`
    /// (server-context.cpp:1408-1413): pull queued tasks into idle slots, then
    /// run one batched decode of every active slot, until nothing is left.
    /// Embedding tasks (`process_single_task`, server-context.cpp:2382-2400)
    /// never touch the slots in the port: they encode synchronously, one at a
    /// time, on the engine thread.
    pub fn run(&mut self, server: &Server) -> ! {
        // pin both `now_us` epochs before the first request — the lazy
        // OnceLocks would otherwise start the clocks AT their first call,
        // whose elapsed is then 0 and the first request's `t_start` reads
        // as "not set" (the `server_slot_stats::is_set` gate and every
        // timing of the first request were wrong; ggml_time_us initializes
        // at library load in the reference — engine.rs and api.rs each own
        // a clock here)
        let _ = (now_us(), crate::api::now_us());
        loop {
            let mut queued: VecDeque<Task> = server.queue.drain().into();
            // the deferred /slots tasks first (queue_tasks.defer) — a slot
            // that is still busy re-defers them below
            queued.extend(self.deferred.drain(..));
            // embedding and slot tasks complete inline — they never touch the
            // slot assignment of `update_slots`
            while let Some(mut task) = queued.pop_front() {
                match task.kind {
                    TaskKind::Embedding => self.run_embedding(&mut task),
                    TaskKind::SlotSave | TaskKind::SlotRestore | TaskKind::SlotErase => {
                        // "if requested slot is unavailable, we defer this task
                        // for processing later" (server-context.cpp:2558-2562)
                        let busy = task
                            .slot_action
                            .as_ref()
                            .map(|a| {
                                self.slots
                                    .iter()
                                    .any(|s| s.id == a.id_slot && s.is_processing())
                            })
                            .unwrap_or(false);
                        if busy {
                            self.deferred.push_back(task);
                        } else {
                            self.run_slot_task(&mut task);
                        }
                    }
                    TaskKind::Completion => self.pending.push_back(task),
                    // the decision pipeline is inline like the embedding one
                    // (`process_single_task`, server-context.cpp:2382-2400) —
                    // the port's slots stay free
                    TaskKind::Decision => self.run_decision(&mut task),
                }
            }
            drop(queued);
            if self.shutdown.load(Ordering::Relaxed) {
                self.abort_all_slots("server is shutting down");
                std::process::exit(0);
            }
            if self.has_work() {
                if let Err(err) = self.update_slots() {
                    eprintln!("llama-server: decode failed: {err}");
                    self.abort_all_slots(&err);
                }
                let processing = self.slots.iter().filter(|s| s.is_processing()).count();
                server.n_processing.store(processing, Ordering::Relaxed);
                // publish the `GET /slots` snapshot (`SERVER_TASK_TYPE_SLOT_GET`,
                // server-context.cpp:2523-2540)
                let snapshot = self.slots_json();
                *server.slots_json.write().unwrap() = snapshot;
            } else {
                server.queue.wait(std::time::Duration::from_millis(200));
            }
        }
    }

    /// `server_slot::to_json` for every slot (server-context.cpp:686-721) —
    /// the metrics-only variant (`slots_debug == 0`).
    fn slots_json(&self) -> String {
        let entries: Vec<Json> = self
            .slots
            .iter()
            .map(|slot| {
                let mut fields = vec![
                    ("id".into(), Json::Int(slot.id as i64)),
                    ("n_ctx".into(), Json::Int(slot.n_ctx as i64)),
                    ("speculative".into(), Json::Bool(self.spec.is_some())),
                    ("is_processing".into(), Json::Bool(slot.is_processing())),
                ];
                // `const auto & ptask = task ? task : task_prev;`
                let live = slot.task.as_ref().map(|t| {
                    (
                        t.id,
                        &t.params,
                        slot.prompt_tokens.len(),
                        slot.stats.n_prompt_processed,
                        slot.stats.n_prompt_cached,
                        slot.stats.n_gen,
                        slot.has_next_token,
                        slot.has_new_line,
                        if slot.n_predict_max < 0 {
                            -1
                        } else {
                            slot.n_predict_max as i64 - slot.stats.n_gen as i64
                        },
                        t.prompt_text.clone(),
                        slot.generated.text.clone(),
                    )
                });
                if let Some((id, params, n_prompt, n_proc, n_cached, n_gen, has_next, has_new_line, n_remaining, prompt, generated)) =
                    live.or_else(|| {
                        slot.prev.as_ref().map(|p| {
                            (
                                p.id,
                                &p.params,
                                p.n_prompt_tokens,
                                p.n_prompt_processed,
                                p.n_prompt_cached,
                                p.n_gen,
                                p.has_next_token,
                                p.has_new_line,
                                p.n_remaining,
                                p.prompt.clone(),
                                p.generated.clone(),
                            )
                        })
                    })
                {
                    fields.push(("id_task".into(), Json::Int(id)));
                    fields.push(("n_prompt_tokens".into(), Json::Int(n_prompt as i64)));
                    fields.push(("n_prompt_tokens_processed".into(), Json::Uint(n_proc)));
                    fields.push(("n_prompt_tokens_cache".into(), Json::Uint(n_cached)));
                    fields.push(("params".into(), task_params_to_json(params, true)));
                    fields.push((
                        "next_token".into(),
                        Json::Array(vec![Json::Object(vec![
                            ("has_next_token".into(), Json::Bool(has_next)),
                            ("has_new_line".into(), Json::Bool(has_new_line)),
                            ("n_remain".into(), Json::Int(n_remaining)),
                            ("n_decoded".into(), Json::Uint(n_gen)),
                        ])]),
                    ));
                    // metrics-only (`only_metrics`) drops prompt/generated
                    let _ = (prompt, generated);
                }
                Json::Object(fields)
            })
            .collect();
        Json::Array(entries).dump()
    }

    /// `SERVER_TASK_TYPE_EMBEDDING` → `send_embedding` (server-context.cpp:
    /// 2152-2192): the pooled row (MEAN/CLS/LAST) normalized with
    /// `common_embd_normalize`, per-token rows for pooling NONE. The encoder
    /// core answers with one `llama_encode`; a decoder model runs the prompt
    /// through the decode path with `llama_set_embeddings(ctx_tgt, true)` —
    /// every token an output row, the pooling node mounted on `t_embd`
    /// (server-context.cpp:2864 + llama-context.cpp:1729).
    fn run_embedding(&mut self, task: &mut Task) {
        let err = |msg: &str, code: i64| {
            let type_str = if code == 400 { "invalid_request_error" } else { "server_error" };
            let _ = task
                .tx
                .send(StreamEvent::Frame(error_frame(crate::api::json_error(msg, type_str, code))));
            let _ = task.tx.send(StreamEvent::Done);
        };
        let tokens = task.tokens.clone();
        let res_type = task.params.res_type;
        let normalize = task.params.embd_normalize;
        // `llama_pooling_type(ctx_tgt)` — resolved at load for both cores
        let pooling = self.pooling;
        // (n_embd_out, n_rows, values) of the embedding computation
        let (n_embd_out, n_rows, values) = match &mut self.core {
            Core::Encode(enc) => match enc.encode(&tokens) {
                Ok(e) => (e.n_embd_out, e.n_rows, e.values),
                Err(e) => {
                    err(&format!("failed to encode: {e}"), 500);
                    return;
                }
            },
            Core::Decode(dctx) => {
                // `llama_set_embeddings(ctx_tgt, slot_batched->need_embd())`
                // (server-context.cpp:2864) — true for an embedding batch
                dctx.set_embeddings(true);
                // fresh-slot semantics: `slot.mem.seq_rm(slot.id, p0, -1)`
                // with an empty slot prompt (p0 = 0) drops whatever the
                // sequence still holds before the prompt decode
                // (server-context.cpp:3446)
                dctx.seq_rm(0, -1, -1);
                // `!slot.can_split()`: an embedding task that is not LAST
                // pooling cannot split (server-context.cpp:440-446) — "input
                // (%d tokens) is too large to process. increase the physical
                // batch size (current batch size: %d)"
                // (server-context.cpp:3187-3196). The port's decode_embed
                // needs one ubatch for every pooling mode.
                if tokens.len() > dctx.n_batch {
                    dctx.set_embeddings(false);
                    err(
                        &format!(
                            "input ({} tokens) is too large to process. increase the physical \
                             batch size (current batch size: {})",
                            tokens.len(),
                            dctx.n_batch
                        ),
                        500,
                    );
                    return;
                }
                // a fresh sequence: the prompt decodes at 0..n
                let pos: Vec<i32> = (0..tokens.len() as i32).collect();
                let res = dctx.decode_embed(&tokens, &pos);
                // the generation batches that follow run with embeddings off,
                // exactly like the C's next `llama_set_embeddings` toggle
                dctx.set_embeddings(false);
                match res {
                    Ok(e) => (e.n_embd_out, e.n_rows, e.values),
                    Err(e) => {
                        err(&format!("failed to decode: {e}"), 500);
                        return;
                    }
                }
            }
            // the d1-omni core: its scores ARE the embedding rows (pooling
            // NONE was forced at load, common.cpp:1285-1290) — an /embeddings
            // request against the decision model answers with the per-token
            // score rows like the reference's encode reroute
            Core::Decision(d) => match d.decide(&tokens) {
                Ok(v) => (d.n_embd_out, tokens.len(), v),
                Err(e) => {
                    err(&format!("failed to encode: {e}"), 500);
                    return;
                }
            },
        };
        // `send_embedding`: pooling NONE reports every token row; a pooled mode
        // reports one normalized row
        let embedding: Vec<Vec<f32>> = if pooling == llama::hparams::LlamaPoolingType::NONE {
            (0..n_rows)
                .map(|r| values[r * n_embd_out..(r + 1) * n_embd_out].to_vec())
                .collect()
        } else {
            vec![common_embd_normalize(&values, normalize)]
        };
        // `server_task_result_embd::to_json` (server-task.cpp:1466-1486)
        let result = if res_type == ResponseType::OaiEmbd {
            Json::Object(vec![
                ("index".into(), Json::Int(task.index as i64)),
                (
                    "embedding".into(),
                    Json::Array(
                        embedding
                            .first()
                            .map(|v| v.iter().map(|&x| Json::Double(x as f64)).collect())
                            .unwrap_or_default(),
                    ),
                ),
                ("tokens_evaluated".into(), Json::Int(task.n_tokens() as i64)),
            ])
        } else {
            Json::Object(vec![
                ("index".into(), Json::Int(task.index as i64)),
                (
                    "embedding".into(),
                    Json::Array(
                        embedding
                            .iter()
                            .map(|v| Json::Array(v.iter().map(|&x| Json::Double(x as f64)).collect()))
                            .collect(),
                    ),
                ),
            ])
        };
        let _ = task.tx.send(StreamEvent::Frame(result.dump()));
        let _ = task.tx.send(StreamEvent::Done);
    }

    /// `SERVER_TASK_TYPE_SLOT_SAVE` / `SLOT_RESTORE` / `SLOT_ERASE`
    /// (server-context.cpp:2545-2686) — the per-slot state round-trip of
    /// `POST /slots/{id}?action=save|restore|erase`.
    ///
    /// The state file is `llama_state_seq_save_file`'s format
    /// (llama-context.cpp:3329-3349): `[u32 'ggsq'][u32 version 3]
    /// [u32 n_tokens][packed tokens][state bytes]`. The reference stores the
    /// memory module's `state_write` output directly; the port stores its
    /// framed `DecodeContext::state_seq_get_data` blob (io_magic + seq_id +
    /// memory state) and restores it through `state_seq_set_data` — the
    /// 8-byte framing is the port's addition, so a port-saved file is not
    /// byte-compatible with a reference-saved one (PARITY.md). The state
    /// covers the plain/iswa KV rows, the full dsv4 frame, the dsa lid rows,
    /// the MSA idx cache and the recurrent conv/ssm cells
    /// (kv_cache.rs `state_seq_write` + context.rs `recurrent_state_seq_write`).
    /// `SERVER_TASK_TYPE_DECISION` — the `/v1/systemone` pipeline
    /// (server-context.cpp:5446-5534 route + :2264-2330 `send_decision`).
    /// The task assembly mirrors the route handler; the decode is inline on
    /// the engine thread like the port's embedding tasks (the reference runs
    /// the prompts through the slot machinery — the observable answers are
    /// the same, see the PARITY.md note on cache metrics).
    fn run_decision(&mut self, task: &mut Task) {
        use crate::server_decision::{
            server_decision_group_tasks, DecisionTaskGroup, DecisionTaskSpec,
        };

        let err = |task: &Task, msg: &str, code: i64| {
            let type_str = if code == 400 {
                "invalid_request_error"
            } else if code == 501 {
                "not_supported_error"
            } else {
                "server_error"
            };
            let _ = task.tx.send(StreamEvent::Frame(error_frame(crate::api::json_error(
                msg, type_str, code,
            ))));
            let _ = task.tx.send(StreamEvent::Done);
        };

        let Some(dc) = self.decision.as_ref() else {
            err(task, "This model is not a decision model", 501);
            return;
        };
        let body = std::mem::replace(&mut task.decision_body, Json::Null);

        // parse_questions / parse_state throw std::invalid_argument -> 400
        let questions = match dc.parse_questions(&body) {
            Ok(q) => q,
            Err(e) => {
                err(task, &e, 400);
                return;
            }
        };
        let (state, files) = match dc.parse_state(&body) {
            Ok(v) => v,
            Err(e) => {
                err(task, &e, 400);
                return;
            }
        };
        // `if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image))`
        // (server-context.cpp:5463-5467) — the port has no mmproj wiring, so
        // `has_inp_image` is always false and ANY image answers 501
        if !files.is_empty() {
            err(
                task,
                "This server does not support image input for decisions. For a model that \
                 supports it, start it with `--mmproj`",
                501,
            );
            return;
        }

        // `get_media_marker()` (server-common.cpp) — only reachable with
        // images in the reference; the port's constant marker
        let media_marker = llama::mtmd::mtmd_default_marker();

        // one task per variant of each question, or one task for all the
        // questions (server-context.cpp:5477-5493)
        let mut specs: Vec<DecisionTaskSpec> = Vec::new();
        if dc.is_joint() {
            match dc.fill_task_joint(&state, &questions) {
                Ok((tokens, spec)) => specs.push(DecisionTaskSpec { tokens, spec, index: 0 }),
                Err(e) => {
                    err(task, &e, 500);
                    return;
                }
            }
        } else {
            let mut flat = 0usize;
            for q in &questions {
                for variant in 0..dc.n_variants(q) {
                    match dc.fill_task(&state, &questions, q, variant, files.len(), &media_marker, &files) {
                        Ok((tokens, spec)) => {
                            specs.push(DecisionTaskSpec { tokens, spec, index: flat })
                        }
                        Err(e) => {
                            // "the instructions and the options of a question
                            // must not be empty" and d1's "no single-token
                            // label left" are the path's invalid_argument
                            let code = if e.contains("must not be empty")
                                || e.contains("no single-token label left")
                            {
                                400
                            } else {
                                500
                            };
                            err(task, &e, code);
                            return;
                        }
                    }
                    flat += 1;
                }
            }
        }

        // `server_decision_group_tasks(tasks, params.n_parallel)` when the
        // model can share its prompt prefix (server-context.cpp:5494-5497)
        let n_parallel = self.slots.len();
        let can_share = dc.can_share_prompt();
        let is_joint = dc.is_joint();
        let groups = if can_share {
            server_decision_group_tasks(specs, n_parallel)
        } else {
            specs
                .into_iter()
                .map(|t| DecisionTaskGroup {
                    parent: t,
                    n_tokens_shared: 0,
                    children: Vec::new(),
                })
                .collect()
        };

        // decode every group. A shared-prefix group mirrors the reference's
        // slot flow exactly (server-context.cpp:3828-3854 + :3926-3932): the
        // parent's batch stops at `n_tokens_shared`, `copy_prompt_to` hands
        // that state to the children (`seq_cp` of the KV cells + the
        // recurrent tail), the parent then continues its own tail and every
        // child decodes its tail from the boundary. The port has one
        // context, so the "copy" is a full-sequence state snapshot taken
        // right after the prefix decode and restored before each child (the
        // recurrent cell cannot be rewound any other way — the single live
        // cell has no per-position snapshots without n_rs_seq)
        let mut scores_by_index: std::collections::BTreeMap<usize, Vec<f32>> =
            std::collections::BTreeMap::new();
        let mut n_tokens_total: i64 = 0;
        for g in groups {
            // "the question and its options (%d tokens) are too large to
            // process..." (server-context.cpp:3362-3370)
            let n_first = if g.parent.spec.pos_first() >= 0 {
                g.parent.spec.pos_first()
            } else {
                g.parent.tokens.len() as i32
            };
            let span = g.parent.tokens.len() as i32 - n_first;
            let n_batch = self.n_batch();
            if span > n_batch {
                err(
                    task,
                    &format!(
                        "the question and its options ({span} tokens) are too large to process. \
                         increase the batch size (current batch size: {n_batch})"
                    ),
                    400,
                );
                return;
            }
            // the shared boundary state: decode the prefix, snapshot the
            // whole sequence (KV + recurrent cell) at that point
            let boundary: Option<Vec<u8>> = if g.n_tokens_shared > 0 {
                match self.decision_decode_prefix(&g.parent.tokens, g.n_tokens_shared) {
                    Ok(snap) => Some(snap),
                    Err(code_msg) => {
                        err(task, &code_msg.0, code_msg.1);
                        return;
                    }
                }
            } else {
                None
            };
            // the parent's own scores: with a shared prefix the parent
            // continues from the boundary (the reference's parent prompt is
            // also split — prefix batch, then its own tail); without one it
            // is one fresh decode
            let parent_scores = if boundary.is_some() {
                self.decision_decode_continue(&g.parent.tokens, g.n_tokens_shared, &g.parent.spec)
            } else {
                self.decision_decode_one(&g.parent.tokens, &g.parent.spec)
            };
            match parent_scores {
                Ok(s) => {
                    n_tokens_total += g.parent.tokens.len() as i64;
                    scores_by_index.insert(g.parent.index, s);
                }
                Err(code_msg) => {
                    err(task, &code_msg.0, code_msg.1);
                    return;
                }
            }
            for child in g.children {
                let shared = g.n_tokens_shared;
                let tail_len = (child.tokens.len() - shared) as i32;
                if tail_len > n_batch {
                    err(
                        task,
                        &format!(
                            "the question and its options ({tail_len} tokens) are too large to \
                             process. increase the batch size (current batch size: {n_batch})"
                        ),
                        400,
                    );
                    return;
                }
                // rewind to the shared boundary before the child's tail
                // (`copy_prompt_to`'s seq_cp on the parent's prefix cells)
                if let Some(snap) = boundary.as_ref() {
                    if let Err(code_msg) = self.decision_restore_state(snap) {
                        err(task, &code_msg.0, code_msg.1);
                        return;
                    }
                }
                match self.decision_decode_tail(&g.parent.tokens, &child.tokens, shared, &child.spec)
                {
                    Ok(s) => {
                        n_tokens_total += child.tokens.len() as i64;
                        scores_by_index.insert(child.index, s);
                    }
                    Err(code_msg) => {
                        err(task, &code_msg.0, code_msg.1);
                        return;
                    }
                }
            }
        }

        // the answers, in the request's question order
        // (server-context.cpp:5506-5527)
        let dc = self.decision.as_ref().unwrap();
        let mut answers: Vec<(String, Json)> = Vec::new();
        let mut i_score = 0usize;
        let mut flat = 0usize;
        for q in &questions {
            let mut q_scores: Vec<Vec<f32>> = Vec::new();
            if is_joint {
                // one result with the scores of all the questions, in order
                let all = scores_by_index.get(&0).cloned().unwrap_or_default();
                let n_opts = q.options.len();
                if i_score + n_opts > all.len() {
                    err(task, "the model could not evaluate the decision", 500);
                    return;
                }
                q_scores.push(all[i_score..i_score + n_opts].to_vec());
                i_score += n_opts;
            } else {
                for _ in 0..dc.n_variants(q) {
                    q_scores.push(scores_by_index.get(&flat).cloned().unwrap_or_default());
                    flat += 1;
                }
            }
            match dc.format_answer(q, &q_scores, !files.is_empty()) {
                Ok(a) => answers.push((q.id.clone(), a)),
                Err(e) => {
                    err(task, &e, 500);
                    return;
                }
            }
        }

        // `res->ok(json{model, answers, usage})` (server-context.cpp:5529-5534)
        let result = Json::Object(vec![
            ("model".into(), Json::String(self.model_name.clone())),
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::Int(n_tokens_total)),
                    ("output_tokens".into(), Json::Int(0)),
                ]),
            ),
        ]);
        let _ = task.tx.send(StreamEvent::Frame(result.dump()));
        let _ = task.tx.send(StreamEvent::Done);
    }

    /// `send_decision`'s label arm (server-context.cpp:2278-2286): the logits
    /// of the label tokens at the last prompt token.
    /// the `decision.label_groups` reduction of `send_decision`
    /// (server-context.cpp:2347-2355, 88dcc460d): one output per group, the
    /// max of the group's label scores
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    fn decision_decode_one(
        &mut self,
        tokens: &[i32],
        spec: &crate::server_decision::DecisionSpec,
    ) -> Result<Vec<f32>, (String, i64)> {
        if !spec.labels.is_empty() {
            let logits = self.decode_last_logits(tokens)?;
            let mut out = Vec::with_capacity(spec.labels.len());
            for &label in &spec.labels {
                let row = *logits
                    .get(label as usize)
                    .ok_or_else(|| ("failed to get logits".to_string(), 500i64))?;
                out.push(row);
            }
            // `decision.label_groups`: the output of a group is the max of
            // its labels' scores (server-context.cpp:2347-2355, 88dcc460d)
            reduce_label_groups(&mut out, spec);
            return Ok(out);
        }
        // markers / joint: the embeddings output (`llama_get_embeddings_ith`)
        let (n_embd_out, n_rows, values) = self.decode_rows(tokens)?;
        self.decision_scores_from_rows(spec, &values, n_rows, n_embd_out, tokens.len())
    }

    /// the parent of a shared-prefix group: the prefix is already in the
    /// sequence (`decision_decode_prefix` left it at the boundary), the
    /// parent's own tail decodes as its second batch and the scores read at
    /// its last token — exactly the reference's [prefix batch][parent tail
    /// batch] split (server-context.cpp:3926-3932 stops the parent's fill at
    /// `n_tokens_shared`)
    fn decision_decode_continue(
        &mut self,
        tokens: &[i32],
        shared: usize,
        spec: &crate::server_decision::DecisionSpec,
    ) -> Result<Vec<f32>, (String, i64)> {
        let tail = &tokens[shared..];
        let pos: Vec<i32> = (shared as i32..tokens.len() as i32).collect();
        if !spec.labels.is_empty() {
            let logits = self.decode_continue_logits_at(tail, &pos)?;
            let mut out = Vec::with_capacity(spec.labels.len());
            for &label in &spec.labels {
                let row = *logits
                    .get(label as usize)
                    .ok_or_else(|| ("failed to get logits".to_string(), 500i64))?;
                out.push(row);
            }
            reduce_label_groups(&mut out, spec);
            return Ok(out);
        }
        let _ = tail;
        let _ = pos;
        // the embeddings family shares only via kev, whose child path
        // re-decodes whole prompts anyway — unreachable in practice
        self.decision_decode_one(tokens, spec)
    }

    /// the shared-prefix boundary decode: run the parent's first `shared`
    /// tokens (one batch, exactly where the reference's batch fill stops —
    /// "stop at the end of the shared prefix", server-context.cpp:3926-3932)
    /// and return the full-sequence state snapshot at that point
    /// (`copy_prompt_to`'s `seq_cp` source state). The port's snapshot is
    /// `state_seq_get_data(0, false)` — the full state, because the
    /// recurrent half is not in the PARTIAL_ONLY blob.
    fn decision_decode_prefix(
        &mut self,
        parent_tokens: &[i32],
        shared: usize,
    ) -> Result<Vec<u8>, (String, i64)> {
        let prefix = &parent_tokens[..shared];
        let pos: Vec<i32> = (0..shared as i32).collect();
        match &mut self.core {
            Core::Decode(dctx) => {
                dctx.seq_rm(0, -1, -1);
                dctx
                    .decode(prefix, &pos)
                    .map_err(|e| (format!("failed to decode: {e}"), 500))?;
                Ok(dctx.state_seq_get_data(0, false))
            }
            // the encoder/decision cores have no sequence state to share (the
            // embeddings family never shares a prompt —
            // `can_share_prompt()`)
            Core::Encode(_) | Core::Decision(_) => Ok(Vec::new()),
        }
    }

    /// restore a `decision_decode_prefix` snapshot before a child's tail
    /// (`copy_prompt_to(other)`: `seq_rm` + `seq_cp` of the prefix cells)
    fn decision_restore_state(&mut self, snap: &[u8]) -> Result<(), (String, i64)> {
        match &mut self.core {
            Core::Decode(dctx) => {
                dctx.seq_rm(0, -1, -1);
                dctx
                    .state_seq_set_data(0, snap, false)
                    .map_err(|e| (format!("failed to restore the shared prefix: {e}"), 500))
            }
            Core::Encode(_) | Core::Decision(_) => Ok(()),
        }
    }

    /// the shared-prefix continuation: the parent's first `shared` tokens are
    /// already in the KV (the restored boundary snapshot), the child decodes
    /// only its tail (`copy_prompt_to` + the child's own prompt processing,
    /// server-context.cpp:3610-3627)
    fn decision_decode_tail(
        &mut self,
        parent_tokens: &[i32],
        child_tokens: &[i32],
        shared: usize,
        spec: &crate::server_decision::DecisionSpec,
    ) -> Result<Vec<f32>, (String, i64)> {
        if !spec.labels.is_empty() {
            let tail = &child_tokens[shared..];
            let pos: Vec<i32> = (shared as i32..child_tokens.len() as i32).collect();
            // no wipe: the sequence continues from the boundary state
            let logits = self.decode_continue_logits_at(tail, &pos)?;
            let mut out = Vec::with_capacity(spec.labels.len());
            for &label in &spec.labels {
                let row = *logits
                    .get(label as usize)
                    .ok_or_else(|| ("failed to get logits".to_string(), 500i64))?;
                out.push(row);
            }
            // the label_groups max of the parent path applies here too
            reduce_label_groups(&mut out, spec);
            return Ok(out);
        }
        // embeddings path: re-decode the whole child prompt (the encode core
        // has no shared-prefix continuation; answers are identical, only the
        // cache accounting differs)
        let _ = parent_tokens;
        let (n_embd_out, n_rows, values) = self.decode_rows(child_tokens)?;
        self.decision_scores_from_rows(spec, &values, n_rows, n_embd_out, child_tokens.len())
    }

    /// the markers / pointer / joint score extraction of `send_decision`
    /// (server-context.cpp:2287-2329)
    fn decision_scores_from_rows(
        &self,
        spec: &crate::server_decision::DecisionSpec,
        values: &[f32],
        n_rows: usize,
        n_embd_out: usize,
        n_tokens: usize,
    ) -> Result<Vec<f32>, (String, i64)> {
        // the outputs of this batch are the last tokens of the prompt
        // (`pos_first = n_tokens - idx.size()`, server-context.cpp:2290-2295)
        let get_embd = |pos: usize| -> Option<&[f32]> {
            let i = pos as i64 - (n_tokens as i64 - n_rows as i64);
            if i < 0 || i as usize >= n_rows {
                None
            } else {
                let r = i as usize * n_embd_out;
                values.get(r..r + n_embd_out)
            }
        };

        let mut out: Vec<f32> = Vec::new();

        // joint head (decision model): the scores are the first rows
        for i in 0..spec.n_scores as usize {
            let embd = get_embd(i).ok_or_else(|| ("failed to get embeddings".to_string(), 500))?;
            out.push(embd[0]);
        }

        let n_pointer = n_embd_out / 2;
        let embd_q = if spec.pointer >= 0 {
            get_embd(spec.pointer as usize)
        } else {
            None
        };
        if spec.pointer >= 0 && embd_q.is_none() {
            return Err((
                "failed to get embeddings, the question and its options must fit in one batch"
                    .to_string(),
                500,
            ));
        }
        for &marker in &spec.markers {
            let embd = get_embd(marker as usize).ok_or_else(|| {
                (
                    "failed to get embeddings, the question and its options must fit in one batch"
                        .to_string(),
                    500,
                )
            })?;
            if spec.pointer < 0 {
                out.push(embd[spec.column as usize]);
                continue;
            }
            let q = embd_q.unwrap();
            let mut dot = 0.0f32;
            for i in 0..n_pointer {
                dot += q[i] * embd[n_pointer + i];
            }
            out.push(dot / (n_pointer as f32).sqrt());
        }

        Ok(out)
    }

    /// the last prompt token's logits row (`llama_get_logits_ith(ctx, i_batch)`)
    fn decode_last_logits(&mut self, tokens: &[i32]) -> Result<Vec<f32>, (String, i64)> {
        let pos: Vec<i32> = (0..tokens.len() as i32).collect();
        self.decode_last_logits_at(tokens, &pos)
    }

    fn decode_last_logits_at(
        &mut self,
        tokens: &[i32],
        pos: &[i32],
    ) -> Result<Vec<f32>, (String, i64)> {
        match &mut self.core {
            Core::Decode(dctx) => {
                // fresh sequence (the inline embedding pattern,
                // server-context.cpp:3446)
                dctx.seq_rm(0, -1, -1);
                match dctx.decode(tokens, pos) {
                    Ok(row) => Ok(row.to_vec()),
                    Err(e) => Err((format!("failed to decode: {e}"), 500)),
                }
            }
            Core::Encode(_) | Core::Decision(_) => Err((
                "an encoder-only context has no logits output".to_string(),
                500,
            )),
        }
    }

    /// the continuation twin of [`Self::decode_last_logits_at`]: no wipe —
    /// the sequence already holds the shared prefix (the restored boundary
    /// snapshot) and the batch appends the child's tail at its positions
    fn decode_continue_logits_at(
        &mut self,
        tokens: &[i32],
        pos: &[i32],
    ) -> Result<Vec<f32>, (String, i64)> {
        match &mut self.core {
            Core::Decode(dctx) => match dctx.decode(tokens, pos) {
                Ok(row) => Ok(row.to_vec()),
                Err(e) => Err((format!("failed to decode: {e}"), 500)),
            },
            Core::Encode(_) | Core::Decision(_) => Err((
                "an encoder-only context has no logits output".to_string(),
                500,
            )),
        }
    }

    /// per-token embedding rows (pooling NONE) — the encoder core's
    /// `llama_encode`, the decode core's `decode_embed`, the d1-omni core's
    /// own stateless forward
    fn decode_rows(&mut self, tokens: &[i32]) -> Result<(usize, usize, Vec<f32>), (String, i64)> {
        let pos: Vec<i32> = (0..tokens.len() as i32).collect();
        match &mut self.core {
            Core::Encode(enc) => match enc.encode(tokens) {
                Ok(e) => Ok((e.n_embd_out, e.n_rows, e.values)),
                Err(e) => Err((format!("failed to encode: {e}"), 500)),
            },
            // the d1-omni model: one whole-prompt pass through
            // `build_lfm2_decision_forward` — the [3, T] scores read as T
            // rows of n_embd_out (the `llama_get_embeddings_ith` layout)
            Core::Decision(d) => match d.decide(tokens) {
                Ok(v) => Ok((d.n_embd_out, tokens.len(), v)),
                Err(e) => Err((format!("failed to encode: {e}"), 500)),
            },
            Core::Decode(dctx) => {
                dctx.set_embeddings(true);
                dctx.seq_rm(0, -1, -1);
                let res = dctx.decode_embed(tokens, &pos);
                dctx.set_embeddings(false);
                match res {
                    Ok(e) => Ok((e.n_embd_out, e.n_rows, e.values)),
                    Err(e) => Err((format!("failed to decode: {e}"), 500)),
                }
            }
        }
    }

    /// `n_batch` of the active core (the batch-size error message reads it)
    fn n_batch(&self) -> i32 {
        match &self.core {
            Core::Decode(d) => d.n_batch as i32,
            // the EncoderContext reserves with the server's n_ubatch
            Core::Encode(_) | Core::Decision(_) => 512,
        }
    }

    /// the checkpoint blob's `partial_only` flag. The reference writes
    /// `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (the swa half / the recurrent
    /// cells / the whole plain KV); the port's partial blob SKIPS the
    /// recurrent half (`state_write_attn_half`, context.rs:3306-3313), so a
    /// recurrent/hybrid memory snapshots the full state — a superset that
    /// restores the same cells (see `create_checkpoint`'s doc comment)
    fn ckpt_partial_only(&self) -> bool {
        !matches!(&self.core, Core::Decode(d) if d.weights.recurrent_dims().is_some())
    }

    /// `llama_memory_seq_pos_min` of the target memory, hybrid-aware:
    /// `llama_memory_hybrid::seq_pos_min` is the MAX of the halves' mins
    /// (llama-memory-hybrid.cpp:172-175), and the recurrent half's one live
    /// cell sits at the sequence's END (find_slot's `cell.pos`,
    /// llama-memory-recurrent.cpp:655-665 — mirrored by the port's
    /// `recurrent_live_seq` reading the KV's pos max, context.rs:3817-3822).
    /// A hybrid whose cell is past the new prompt's prefix therefore reads
    /// `pos_min` at the old end — exactly what makes the rollback fire.
    fn mem_pos_min(&self, id: i32) -> i32 {
        match &self.core {
            Core::Decode(d) => {
                let attn_min = d.seq_pos_min(id);
                if d.weights.recurrent_dims().is_some() && d.recurrent_state_live_seq() == Some(id)
                {
                    // the live cell's position — the sequence's pos max
                    attn_min.max(d.kv.seq_pos_max_of(id as usize))
                } else {
                    attn_min
                }
            }
            _ => -1,
        }
    }

    /// `llama_memory_seq_pos_max`, hybrid-aware: the MIN of the halves'
    /// maxes (llama-memory-hybrid.cpp:177-180)
    fn mem_pos_max(&self, id: i32) -> i32 {
        match &self.core {
            Core::Decode(d) => {
                let attn_max = d.seq_pos_max(id);
                if d.weights.recurrent_dims().is_some() && d.recurrent_state_live_seq() == Some(id)
                {
                    attn_max.min(d.kv.seq_pos_max_of(id as usize))
                } else {
                    attn_max
                }
            }
            _ => -1,
        }
    }

    /// `create_checkpoint` (server-context.cpp:2521-2584, PR #13194's
    /// machinery, the 033df86b6 T2 remainder) — snapshot the parts of the
    /// memory that cannot be rolled back, at the head of the current batch
    /// (the caller runs it BEFORE the batch decodes, so the batch's tokens
    /// are not in the snapshot). `n_tokens_cur` = the tokens this slot added
    /// to the batch; the checkpoint's `n_tokens` is the slot's prompt length
    /// below them.
    ///
    /// The reference saves `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (the swa half
    /// of an iswa cache, the recurrent cells of a hybrid, everything of a
    /// plain KV — llama-kv-cache-iswa.cpp:260-266 /
    /// llama-memory-hybrid.cpp:190-196). The port's partial blob skips the
    /// recurrent half (`state_write_attn_half`,
    /// context.rs:3306-3313), so on a recurrent/hybrid memory the port
    /// snapshots the FULL state instead — a superset that restores the same
    /// cells plus the (already identical) attention prefix.
    fn create_checkpoint(&mut self, si: usize, n_tokens_cur: usize) {
        let id_task = self.slots[si].task.as_ref().map(|t| t.id).unwrap_or(-1);
        let n_ctx_checkpoints = self.n_ctx_checkpoints.max(0) as usize;
        let min_step = self.checkpoint_min_step;

        // evict checkpoints within min-step of a previous one, unless they
        // were created by the current task; only when the list is full —
        // otherwise short prompts keep just the oldest checkpoint
        // (server-context.cpp:2527-2541)
        {
            let mut last: i64 = -1;
            let mut it = 0usize;
            while self.slots[si].checkpoints.len() + 1 >= n_ctx_checkpoints
                && it < self.slots[si].checkpoints.len()
            {
                let cur = &self.slots[si].checkpoints[it];
                if cur.id_task != id_task
                    && last >= 0
                    && cur.n_tokens as i64 <= last + min_step as i64
                {
                    self.slots[si].checkpoints.remove(it);
                    continue;
                }
                last = cur.n_tokens as i64;
                it += 1;
            }
        }
        // make room for the new checkpoint (server-context.cpp:2543-2551)
        while self.slots[si].checkpoints.len() >= n_ctx_checkpoints {
            self.slots[si].checkpoints.pop_front();
        }

        // replace an existing checkpoint at the same n_tokens instead of
        // appending a duplicate (server-context.cpp:2553-2564)
        let n_tokens_new = self.slots[si].prompt_tokens.len() - n_tokens_cur;
        self.slots[si].checkpoints.retain(|c| c.n_tokens != n_tokens_new);

        // `update_pos` + `update_tgt`/`update_dft` (PARTIAL_ONLY)
        let id = self.slots[si].id;
        let (pos_min, pos_max) = (self.mem_pos_min(id), self.mem_pos_max(id));
        let recurrent = !self.ckpt_partial_only();
        // the partial/full choice — see the doc comment above
        let partial_only = !recurrent;
        let (data_tgt, data_dft) = match &mut self.core {
            Core::Decode(d) => (
                d.state_seq_get_data(id, partial_only),
                match self.spec.as_mut().and_then(|s| s.ctx_dft()) {
                    Some(dft) => dft.state_seq_get_data(id, partial_only),
                    None => Vec::new(),
                },
            ),
            Core::Encode(_) | Core::Decision(_) => (Vec::new(), Vec::new()),
        };
        // `common_speculative_get_state` (server-context.cpp:2577) — the
        // draft's speculative stash (eagle3's deferred g_embd row). The
        // port's CommonSpeculative has no get/set-state surface, so the
        // stash stays empty; nothing reads it on the port's rollback path
        // (the draft sequence is rebuilt from the restored target state)
        let data_spec = Vec::new();

        self.slots[si].checkpoints.push_back(SlotCheckpoint {
            id_task,
            n_tokens: n_tokens_new,
            pos_min,
            pos_max,
            data_tgt,
            data_dft,
            data_spec,
        });
        let cur = self.slots[si].checkpoints.back().unwrap();
        // SLT_TRC — the trace log, behind the port's debug gate
        if std::env::var("LLAMA_SERVER_DEBUG").is_ok() {
            eprintln!(
                "slot {}: created context checkpoint {} of {} (pos_min = {}, pos_max = {}, \
                 n_tokens = {}, size = {:.3} MiB)",
                self.slots[si].id,
                self.slots[si].checkpoints.len(),
                n_ctx_checkpoints,
                cur.pos_min,
                cur.pos_max,
                cur.n_tokens,
                cur.size() as f64 / 1024.0 / 1024.0
            );
        }
    }

    /// `save_slot_checkpoints` (server-context.cpp:2623-2653, 033df86b6) —
    /// serialize `slot.prompt.checkpoints` as the "SCKP" appendix of the slot
    /// save file. The port builds the file in memory, so the appendix bytes
    /// extend `out` (the C appends to the already-written payload file);
    /// returns the number of appendix bytes. An empty checkpoint list writes
    /// nothing — the same byte layout as the reference.
    fn save_slot_checkpoints(&self, si: usize, out: &mut Vec<u8>) -> usize {
        let n = slot_checkpoints_appendix(&self.slots[si].checkpoints, out);
        if n > 0 {
            eprintln!(
                "slot {}: appended {} context checkpoint(s) ({:.3} MiB)",
                self.slots[si].id,
                self.slots[si].checkpoints.len(),
                n as f64 / 1024.0 / 1024.0
            );
        }
        n
    }

    /// `load_slot_checkpoints` (server-context.cpp:2655-2714, 033df86b6) —
    /// parse the checkpoint appendix starting at `offset` (the end of the
    /// llama state payload within the file). Returns the number of appendix
    /// bytes consumed, 0 if there is no usable appendix. When a draft context
    /// is active and the newest checkpoint carries draft data, one blob is
    /// test-loaded — a mismatch drops every checkpoint's draft data instead
    /// of crashing.
    fn load_slot_checkpoints(&mut self, bytes: &[u8], offset: usize, si: usize) -> usize {
        let (mut checkpoints, n_read) = parse_slot_checkpoints_appendix(bytes, offset, self.slots[si].id);
        if n_read == 0 {
            return 0;
        }
        // the slot file does not check the draft context - test-load one draft
        // checkpoint, drop the draft data if it does not fit
        if self.spec.is_some()
            && !checkpoints.is_empty()
            && !checkpoints.back().unwrap().data_dft.is_empty()
        {
            let id_slot = self.slots[si].id;
            let back = checkpoints.back().unwrap().clone();
            let mut ok = true;
            if let Some(dft) = self.spec.as_mut().unwrap().ctx_dft() {
                ok = dft.state_seq_set_data(id_slot, &back.data_dft, true).is_ok();
                dft.seq_rm(id_slot, -1, -1);
            }
            if !ok {
                eprintln!(
                    "slot {}: draft context checkpoint data does not match the draft context - dropped",
                    self.slots[si].id
                );
                for cur in checkpoints.iter_mut() {
                    cur.data_dft.clear();
                }
            }
        }
        eprintln!(
            "slot {}: restored {} context checkpoint(s)",
            self.slots[si].id,
            checkpoints.len()
        );
        self.slots[si].checkpoints = checkpoints;
        n_read
    }

    fn run_slot_task(&mut self, task: &mut Task) {
        let err = |task: &Task, msg: &str, code: i64| {
            let type_str = if code == 400 { "invalid_request_error" } else { "server_error" };
            let _ = task
                .tx
                .send(StreamEvent::Frame(error_frame(crate::api::json_error(msg, type_str, code))));
            let _ = task.tx.send(StreamEvent::Done);
        };
        let Some(action) = task.slot_action.clone() else {
            err(task, "Unable to save slot", 500);
            return;
        };
        // `get_slot_by_id` → "Invalid slot ID" (server-context.cpp:2547-2551)
        let Some(si) = self.slots.iter().position(|s| s.id == action.id_slot) else {
            err(task, "Invalid slot ID", 400);
            return;
        };

        // the state-coverage gate: `KvCache::state_seq_write` serializes the
        // plain/iswa KV rows and the full dsv4 frame, the dsa lid rows
        // (kv_cache.rs::lid_state_write), the MSA idx rows (context.rs::
        // msa_state_seq_write) and — since the recurrent-state round — the
        // per-layer conv/ssm cells of the mamba/jamba/nemotron-h/
        // qwen3next/kimi-linear/bailingmoe3 family (context.rs::
        // recurrent_state_seq_write, llama-memory-recurrent.cpp:766-1224).
        // One port limitation remains measurable: the recurrent memory keeps
        // a single live cell, so a slot whose sequence is not the live one
        // cannot save its recurrence (the reference serializes every
        // sequence's own cell) — that save answers the reference's failure
        // shape with the reason instead of silently dropping the state.
        let state_gap = match &self.core {
            Core::Decode(dctx) => {
                let id_slot = action.id_slot;
                if dctx.weights.recurrent_dims().is_some()
                    && dctx.recurrent_state_live_seq().is_some_and(|s| s != id_slot)
                {
                    Some(
                        "the recurrent state of another sequence is live (the port keeps a \
                         single recurrent cell)",
                    )
                } else {
                    None
                }
            }
            // an encoder-only context has no sequence state at all; the
            // d1-omni decision core is stateless the same way (one forward
            // per request, no memory module)
            Core::Encode(_) | Core::Decision(_) => {
                Some("an encoder-only context has no sequence state")
            }
        };

        match task.kind {
            TaskKind::SlotSave => {
                let t_start = now_us();
                // `llama_state_seq_save_file`'s 0-byte failure → "Unable to
                // save slot" (server-context.cpp:2580-2583)
                if let Some(reason) = state_gap {
                    err(task, &format!("Unable to save slot: {reason}"), 500);
                    return;
                }
                // `slot->prompt.tokens.serialize()` (server-common.cpp:570-588)
                let packed = server_tokens_serialize(&self.slots[si].prompt_tokens);
                // `llama_state_seq_save_file` + "Unable to save slot" on 0
                // bytes written (server-context.cpp:2573-2582)
                let n_saved = self.slots[si].prompt_tokens.len();
                let framed = match &self.core {
                    Core::Decode(dctx) => dctx.state_seq_get_data(action.id_slot, false),
                    // an encoder-only context has no sequence state to save
                    // (the state_gap above already rejected these)
                    Core::Encode(_) | Core::Decision(_) => {
                        err(task, "Unable to save slot", 500);
                        return;
                    }
                };
                // `state_seq_save_file` (llama-context.cpp:3481-3499) writes
                // `state_seq_write_data` DIRECTLY — the `[io_magic][seq_id]`
                // frame exists only in the in-memory `state_seq_get_data`
                // blob, so it is stripped for the file (byte-compatible with
                // the reference's slot files)
                let state: &[u8] = &framed[8..];
                let mut file: Vec<u8> = Vec::with_capacity(12 + packed.len() + state.len());
                file.extend_from_slice(&STATE_SEQ_MAGIC.to_le_bytes());
                file.extend_from_slice(&STATE_SEQ_VERSION.to_le_bytes());
                file.extend_from_slice(&(packed.len() as u32 / 4).to_le_bytes());
                file.extend_from_slice(&packed);
                file.extend_from_slice(&state);
                // the checkpoint appendix goes after the llama state payload
                // (`save_slot_checkpoints`, server-context.cpp:2925-2929,
                // 033df86b6) — an empty list appends nothing, matching the
                // reference's bytes; the write failure of the whole file
                // covers the C's "incomplete context checkpoints" error
                let _nwrite_ckpt = self.save_slot_checkpoints(si, &mut file);
                if std::fs::write(&action.filepath, &file).is_err() {
                    err(task, "Unable to save slot", 500);
                    return;
                }
                let t_ms = (now_us() - t_start) as f64 / 1000.0;
                // `server_task_result_slot_save_load::to_json` (is_save,
                // server-task.cpp:1622-1641)
                let result = Json::Object(vec![
                    ("id_slot".into(), Json::Int(action.id_slot as i64)),
                    ("filename".into(), Json::String(action.filename.clone())),
                    ("n_saved".into(), Json::Int(n_saved as i64)),
                    ("n_written".into(), Json::Int(file.len() as i64)),
                    ("timings".into(), save_ms_json("save_ms", t_ms)),
                ]);
                let _ = task.tx.send(StreamEvent::Frame(result.dump()));
                let _ = task.tx.send(StreamEvent::Done);
            }
            TaskKind::SlotRestore => {
                let t_start = now_us();
                if let Some(reason) = state_gap {
                    err(task, &format!("Unable to restore slot: {reason}"), 400);
                    return;
                }
                // `llama_state_seq_load_file` (llama-context.cpp:3275-3327) —
                // the magic/version check, then the packed prompt, then
                // `state_seq_read_data`. Any failure reads as 0 bytes → "No
                // available space in KV cache or invalid slot save file"
                // (server-context.cpp:2618-2621)
                let fail_invalid = |task: &Task| {
                    err(
                        task,
                        "Unable to restore slot: No available space in KV cache or invalid slot \
                         save file",
                        400,
                    );
                };
                let bytes = match std::fs::read(&action.filepath) {
                    Ok(b) => b,
                    Err(_) => {
                        fail_invalid(task);
                        return;
                    }
                };
                let mut rd = &bytes[..];
                let take_u32 = |rd: &mut &[u8]| -> Option<u32> {
                    let (head, rest) = rd.split_at_checked(4)?;
                    *rd = rest;
                    Some(u32::from_le_bytes(head.try_into().unwrap()))
                };
                if take_u32(&mut rd) != Some(STATE_SEQ_MAGIC)
                    || take_u32(&mut rd) != Some(STATE_SEQ_VERSION)
                {
                    fail_invalid(task);
                    return;
                }
                let Some(n_packed) = take_u32(&mut rd) else {
                    fail_invalid(task);
                    return;
                };
                // "token count in sequence state file exceeds the file size!"
                // (:3291-3295) — the null-buffer sizing pass
                if (n_packed as usize) * 4 > rd.len() {
                    fail_invalid(task);
                    return;
                }
                let packed_bytes: Vec<u8> = rd[..n_packed as usize * 4].to_vec();
                rd = &rd[n_packed as usize * 4..];
                let packed: Vec<i32> = packed_bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                // `server_tokens::deserialize` (:2632) — the errors throw out
                // of the try block as "Unable to restore slot: <what>"
                let restored = match server_tokens_deserialize(&packed) {
                    Ok(t) => t,
                    Err(e) => {
                        // a failed restore clears the slot prompt
                        // (server-context.cpp:2644-2646)
                        self.slot_prompt_clear(si);
                        err(task, &format!("Unable to restore slot: {e}"), 400);
                        return;
                    }
                };
                // "Restored prompt does not fit in the slot context" (:2639)
                if restored.len() > self.slots[si].n_ctx as usize {
                    self.slot_prompt_clear(si);
                    err(task, "Unable to restore slot: Restored prompt does not fit in the slot context", 400);
                    return;
                }
                // `restored.validate(ctx_tgt)` → "Invalid tokens in slot save
                // file" (:2642-2645; server-common.cpp:755-785: every token
                // must be inside the vocab — no mtmd, so no NULL markers)
                if restored.iter().any(|&t| t < 0 || t as usize >= self.n_vocab as usize) {
                    self.slot_prompt_clear(si);
                    err(task, "Unable to restore slot: Invalid tokens in slot save file", 400);
                    return;
                }
                // `slot->prompt.clear()` then the state read
                // (server-context.cpp:2647) — seq_rm drops the old cells
                // before the restored state lands
                self.slot_prompt_clear(si);
                let payload_end;
                match &mut self.core {
                    Core::Decode(dctx) => {
                        // the file stores the unframed `state_seq_write_data`
                        // stream (state_seq_load_file reads it through
                        // llama_io_read_file); the port's state_seq_set_data
                        // takes the framed blob, so the 8-byte header goes
                        // back on (the saved seq id is not matched on
                        // restore, the target id is what counts)
                        let mut framed = Vec::with_capacity(rd.len() + 8);
                        framed.extend_from_slice(&llama::context::DecodeContext::STATE_SEQ_IO_MAGIC.to_le_bytes());
                        framed.extend_from_slice(&(action.id_slot as i32).to_le_bytes());
                        framed.extend_from_slice(rd);
                        if let Err(e) = dctx.state_seq_set_data(action.id_slot, &framed, false) {
                            err(task, &format!("Unable to restore slot: {e}"), 400);
                            return;
                        }
                        // `nread` is the end offset of the llama state payload
                        // within the file (server-context.cpp:2996) — the C's
                        // `file.tell()` after `state_seq_read_data`; the port
                        // re-derives it from a dummy size pass over the
                        // restored sequence (minus the 8-byte frame the size
                        // pass includes but the file does not)
                        payload_end = 12 + packed_bytes.len() + dctx.state_seq_get_size(action.id_slot, false) - 8;
                    }
                    Core::Encode(_) | Core::Decision(_) => {
                        err(task, "Unable to restore slot: No available space in KV cache or invalid slot save file", 400);
                        return;
                    }
                }
                self.slots[si].prompt_tokens = restored;
                // the checkpoint appendix after the payload
                // (`load_slot_checkpoints`, server-context.cpp:2998-3000, 033df86b6)
                let nread_ckpt = self.load_slot_checkpoints(&bytes, payload_end, si);
                // `res->n_bytes = nread + nread_ckpt` (:3008) — the payload
                // end plus whatever appendix was consumed (an ignored
                // appendix reports just the payload)
                let n_read = payload_end + nread_ckpt;
                let t_ms = (now_us() - t_start) as f64 / 1000.0;
                // `server_task_result_slot_save_load::to_json` (restore,
                // server-task.cpp:1643-1655)
                let result = Json::Object(vec![
                    ("id_slot".into(), Json::Int(action.id_slot as i64)),
                    ("filename".into(), Json::String(action.filename.clone())),
                    (
                        "n_restored".into(),
                        Json::Int(self.slots[si].prompt_tokens.len() as i64),
                    ),
                    ("n_read".into(), Json::Int(n_read as i64)),
                    ("timings".into(), save_ms_json("restore_ms", t_ms)),
                ]);
                let _ = task.tx.send(StreamEvent::Frame(result.dump()));
                let _ = task.tx.send(StreamEvent::Done);
            }
            TaskKind::SlotErase => {
                // "Erase token cache" (server-context.cpp:2673-2676):
                // n_erased before the clear, `slot->prompt_clear()` after
                let n_erased = self.slots[si].prompt_tokens.len();
                self.slot_prompt_clear(si);
                // `server_task_result_slot_erase::to_json` (server-task.cpp:1649-1656)
                let result = Json::Object(vec![
                    ("id_slot".into(), Json::Int(action.id_slot as i64)),
                    ("n_erased".into(), Json::Int(n_erased as i64)),
                ]);
                let _ = task.tx.send(StreamEvent::Frame(result.dump()));
                let _ = task.tx.send(StreamEvent::Done);
            }
            _ => unreachable!("run_slot_task on a non-slot task"),
        }
    }

    /// `server_slot::prompt_clear` (server-context.cpp:334-339) —
    /// `mem.seq_rm(id, -1, -1)` + `prompt.clear()`
    fn slot_prompt_clear(&mut self, si: usize) {
        let id = self.slots[si].id;
        if let Core::Decode(dctx) = &mut self.core {
            dctx.seq_rm(id, -1, -1);
        }
        self.slots[si].prompt_tokens.clear();
        // `server_prompt::clear()` also drops the context checkpoints
        // (server-common.cpp:475-482)
        self.slots[si].checkpoints.clear();
    }

    /// `get_available_slot` (server-context.cpp:1279-1390): the requested
    /// `id_slot`, else the slot with the best prompt-similarity, else the least
    /// recently used one (ties resolve to the highest id, like the C's
    /// `slot.t_last_used <= t_last`).
    fn get_available_slot(&self, task: &Task) -> Option<usize> {
        if task.params.id_slot != -1 {
            let id = task.params.id_slot;
            if let Some(i) = self.slots.iter().position(|s| s.id == id) {
                if !self.slots[i].is_processing() {
                    return Some(i);
                }
            }
        }

        // prompt similarity (server-context.cpp:1291-1360), the C's default
        // `slot_prompt_similarity = 0.5`
        let mut best: Option<usize> = None;
        let mut f_sim_best = 0.0f32;
        for (i, slot) in self.slots.iter().enumerate() {
            if task.params.id_slot != -1 && slot.id != task.params.id_slot {
                continue;
            }
            if slot.is_processing() || slot.prompt_tokens.is_empty() {
                continue;
            }
            let lcp = common_prefix_len(&slot.prompt_tokens, &task.tokens);
            let f_sim = lcp as f32 / task.tokens.len().max(1) as f32;
            if f_sim > f_sim_best && f_sim > 0.5 {
                f_sim_best = f_sim;
                best = Some(i);
            }
        }
        if best.is_some() {
            return best;
        }

        // LRU (server-context.cpp:1362-1381)
        let mut ret: Option<usize> = None;
        let mut t_last = -1i64;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_processing() {
                continue;
            }
            if ret.is_none() || slot.t_last_used <= t_last {
                t_last = slot.t_last_used;
                ret = Some(i);
            }
        }
        ret
    }

    /// `launch_slot_with_task` (server-context.cpp:1707-1825) — the parts the
    /// port honours: the prompt validation, the sampler chain, the per-request
    /// generation limit.
    fn launch_slot(&mut self, si: usize, task: Task) {
        let n_predict = if task.params.n_predict != -1 {
            task.params.n_predict
        } else {
            self.n_predict_default
        };
        // common_sampler_init(vocab, params) — the vocab lowers the DRY
        // string sequence breakers into token sequences
        // (common/sampling.cpp:353 llama_sampler_init_dry(vocab, …))
        let mut sampling = task.params.sampling.clone();
        // `n_probs` rides on the sampling params (common.h:700) — it decides
        // the greedy/dist eligibility of the chain's tail (d0b490f25)
        sampling.n_probs = task.params.n_probs;
        let mut sampler = SamplingContext::new_with_vocab(
            self.n_vocab,
            Some(&self.vocab),
            sampling,
        );
        // `server_slot::init_sampler` (server-context.cpp:409-428): the full
        // prompt is fed into the chain with is_generated = false so the
        // stateful samplers (penalties ring, DRY ring) see prompt repetitions
        // before the first sample — without this, DRY / repeat penalties
        // would only react to generated tokens
        for &tok in task.tokens.iter() {
            if tok != -1 {
                // LLAMA_TOKEN_NULL guard (:420-423)
                sampler.accept(tok);
            }
        }
        let sampler = Some(sampler);
        // the triggers of a lazy grammar (`common_sampler_init`'s trigger
        // loop, sampling.cpp:222-256: TOKEN triggers are kept as tokens, WORD
        // triggers become regex-escaped words, PATTERN/PATTERN_FULL triggers
        // become regexes — a PATTERN_FULL is wrapped in `^…$`,
        // sampling.cpp:235-245)
        let mut lazy_triggers = None;
        let grammar = if task.params.grammar.is_empty() {
            None
        } else {
            match llama::sampling::init_grammar(&self.vocab, &task.params.grammar) {
                Ok(mut g) => {
                    if task.params.grammar_lazy {
                        // `llama_sampler_init_grammar_lazy_patterns`
                        // (sampling.cpp:265-268): the triggers gate when the
                        // grammar starts constraining
                        g.grammar.lazy = true;
                        g.grammar.awaiting_trigger = true;
                        let mut triggers = LazyTriggers {
                            tokens: Vec::new(),
                            patterns: Vec::new(),
                            buffer: Vec::new(),
                            positions: Vec::new(),
                        };
                        // an unsupported pattern regex fails the request
                        // loudly (the C's `std::regex` ctor throws)
                        let mut trigger_err: Option<String> = None;
                        for t in &task.params.grammar_triggers {
                            let ty = t.at("type").and_then(|v| v.get_i64().ok()).unwrap_or(-1);
                            let value = t.at("value").and_then(|v| v.get_str().ok());
                            match ty {
                                0 => {
                                    // COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
                                    if let Some(tok) = t.at("token").and_then(|v| v.get_i64().ok()) {
                                        triggers.tokens.push(tok as i32);
                                    }
                                }
                                1 => {
                                    // COMMON_GRAMMAR_TRIGGER_TYPE_WORD — the
                                    // engine matches the literal word
                                    if let Some(w) = value {
                                        triggers.patterns.push(LazyPattern::Word(w.as_bytes().to_vec()));
                                    }
                                }
                                2 => {
                                    // COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN —
                                    // compiled as-is (sampling.cpp:230-234)
                                    match value.map(llama::regex_lite::RegexLite::new) {
                                        Some(Ok(re)) => triggers.patterns.push(LazyPattern::Regex(re)),
                                        Some(Err(e)) => trigger_err = Some(e),
                                        None => {}
                                    }
                                }
                                3 => {
                                    // COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL —
                                    // anchored `^…$` (sampling.cpp:235-245)
                                    match value.map(|v| anchor_pattern_full(v)) {
                                        Some(anchored) => {
                                            match llama::regex_lite::RegexLite::new(&anchored) {
                                                Ok(re) => {
                                                    triggers.patterns.push(LazyPattern::Regex(re))
                                                }
                                                Err(e) => trigger_err = Some(e),
                                            }
                                        }
                                        None => {}
                                    }
                                }
                                // C: GGML_ASSERT(false && "unknown trigger type")
                                _ => {
                                    trigger_err =
                                        Some(format!("unknown lazy-grammar trigger type {ty}"));
                                }
                            }
                        }
                        if let Some(e) = trigger_err {
                            let _ = task.tx.send(StreamEvent::Frame(error_frame(
                                crate::api::json_error(
                                    &format!("Failed to initialize samplers: {e}"),
                                    crate::api::ERROR_TYPE_INVALID_REQUEST.0,
                                    crate::api::ERROR_TYPE_INVALID_REQUEST.1,
                                ),
                            )));
                            let _ = task.tx.send(StreamEvent::Done);
                            return;
                        }
                        lazy_triggers = Some(triggers);
                    } else if task.params.grammar_prefill && !task.params.generation_prompt.is_empty()
                    {
                        // the grammar prefill (common/sampling.cpp:294-308,
                        // only for non-lazy grammars): feed the generation
                        // prompt's tokens so the grammar advances past the
                        // literal the template already placed in the prompt —
                        // only for the wrapped chat grammars
                        // (`common_grammar_needs_prefill`), never for
                        // user-supplied ones
                        let prefill = self.vocab.tokenize(&task.params.generation_prompt, false, true);
                        for (i, &tok) in prefill.iter().enumerate() {
                            // "some tokenizers add a space before the first
                            // special token" (sampling.cpp:283-288)
                            if i == 0 {
                                let piece = self.vocab.token_to_piece(tok);
                                if piece.starts_with(' ')
                                    && !task.params.generation_prompt.starts_with(' ')
                                {
                                    continue;
                                }
                            }
                            if let Err(e) = g.try_accept(tok) {
                                let _ = task.tx.send(StreamEvent::Frame(error_frame(
                                    crate::api::json_error(
                                        &format!("error initializing grammar sampler for grammar:\n{}\n\nGeneration prompt:\n'{}'\n: {e}",
                                            task.params.grammar, task.params.generation_prompt),
                                        crate::api::ERROR_TYPE_INVALID_REQUEST.0,
                                        crate::api::ERROR_TYPE_INVALID_REQUEST.1,
                                    ),
                                )));
                                let _ = task.tx.send(StreamEvent::Done);
                                return;
                            }
                        }
                    }
                    Some(g)
                }
                Err(e) => {
                    let _ = task.tx.send(StreamEvent::Frame(error_frame(
                        crate::api::json_error(
                            &format!("Failed to initialize samplers: failed to parse grammar: {e}"),
                            crate::api::ERROR_TYPE_INVALID_REQUEST.0,
                            crate::api::ERROR_TYPE_INVALID_REQUEST.1,
                        ),
                    )));
                    let _ = task.tx.send(StreamEvent::Done);
                    self.pending.push_back(task);
                    let t = self.pending.pop_back().unwrap();
                    let _ = t.tx.send(StreamEvent::Done);
                    return;
                }
            }
        };

        // the OAI-chat parse state (`task::create_state`, server-queue.cpp:530)
        let chat = if task.params.res_type == ResponseType::OaiChat {
            match ChatStreamState::new(&task.params, &self.vocab) {
                Ok(state) => Some(state),
                Err(e) => {
                    let _ = task.tx.send(StreamEvent::Frame(error_frame(
                        crate::api::json_error(
                            &format!("Failed to initialize samplers: {e}"),
                            crate::api::ERROR_TYPE_INVALID_REQUEST.0,
                            crate::api::ERROR_TYPE_INVALID_REQUEST.1,
                        ),
                    )));
                    let _ = task.tx.send(StreamEvent::Done);
                    return;
                }
            }
        } else {
            None
        };

        let slot = &mut self.slots[si];
        slot.n_predict_max = n_predict;
        slot.sampled = 0;
        slot.i_batch = -1;
        slot.generated = llama::chat_tools::ChatInput::default();
        slot.generated_tokens.clear();
        slot.probs_output.clear();
        slot.n_sent_text = 0;
        slot.has_next_token = false;
        slot.has_new_line = false;
        slot.stop = StopType::None;
        slot.stopping_word.clear();
        slot.truncated = false;
        slot.sent_begin = false;
        slot.chat = chat;
        slot.lazy = lazy_triggers;
        slot.spec_draft.clear();
        slot.spec_i_batch.clear();
        slot.stats = GenStats::default();
        slot.grammar = grammar;
        slot.sampler = sampler;
        slot.task = Some(task);
        slot.state = SlotState::Started;
    }

    /// `abort_all_slots`
    pub fn abort_all_slots(&mut self, err: &str) {
        for i in 0..self.slots.len() {
            if self.slots[i].is_processing() {
                if let Some(t) = self.slots[i].task.take() {
                    let _ = t.tx.send(StreamEvent::Frame(error_frame(crate::api::json_error(
                        err,
                        crate::api::ERROR_TYPE_SERVER.0,
                        crate::api::ERROR_TYPE_SERVER.1,
                    ))));
                    let _ = t.tx.send(StreamEvent::Done);
                }
                self.slots[i].release();
            }
        }
    }

    /// `pre_decode`'s context-shift block for one slot
    /// (server-context.cpp:2934-2971): with the context full, keep `n_keep`
    /// tokens, drop `n_discard` more (`seq_rm`), shift everything above them
    /// by `-n_discard` positions (`seq_add` — the K rows are re-rotated by the
    /// K-shift graph on the next decode, llama-context.cpp:1805), and splice
    /// the slot's token list the same way so generation continues.
    fn context_shift(&mut self, si: usize) -> Result<(), String> {
        // :2914-2920 — "this check is redundant (for good)": process_token
        // already stopped a !ctx_shift slot at the boundary
        // (engine.rs process_token, server-context.cpp:1887-1894). The caller's
        // fail_slot is the port's `send_error(slot, ...) + slot.release()`
        if !self.ctx_shift {
            return Err("context shift is disabled".into());
        }

        // int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens()
        //              : slot.task->params.n_keep;  (:2935)
        let (n_keep_req, n_discard_req) = {
            let t = self.slots[si].task.as_ref().unwrap();
            (t.params.n_keep, t.params.n_discard)
        };
        let mut n_keep =
            if n_keep_req < 0 { self.slots[si].task.as_ref().unwrap().n_tokens() } else { n_keep_req };
        // if (add_bos_token) n_keep += 1;  (:2937-2939)
        if self.vocab.add_bos {
            n_keep += 1;
        }
        // n_keep = std::min(slot.n_ctx - 4, n_keep);  (:2941)
        n_keep = n_keep.min(self.slots[si].n_ctx as i32 - 4);

        // const int n_left = slot.prompt.n_tokens() - n_keep;
        // int n_discard = params.n_discard ? params.n_discard : (n_left / 2);
        // n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));
        // (:2943-2947, ref PR #24786)
        let n_left = self.slots[si].prompt_tokens.len() as i32 - n_keep;
        let n_discard = if n_discard_req != 0 { n_discard_req } else { n_left / 2 }
            .clamp(0, (n_left - 1).max(0));

        // SLT_WRN "slot context shift" (:2949)
        if std::env::var("LLAMA_SERVER_DEBUG").is_ok() {
            eprintln!(
                "llama-server: slot context shift, n_keep = {n_keep}, n_left = {n_left}, \
                 n_discard = {n_discard}"
            );
        }

        // slot.mem.seq_rm (slot.id, n_keep, n_keep + n_discard);
        // slot.mem.seq_add(slot.id, n_keep + n_discard,
        //                  slot.prompt.tokens.pos_next(), -n_discard);  (:2951-2952)
        let (slot_id, pos_next) = (self.slots[si].id, self.slots[si].pos_next());
        if let Some(d) = self.dctx() {
            d.seq_rm(slot_id, n_keep, n_keep + n_discard);
            // a model whose cache cannot K-shift (deepseek4's dsv4) surfaces
            // the reference's abort as a request error here
            d.seq_add(slot_id, n_keep + n_discard, pos_next, -n_discard)?;
        }

        // shift the slot's token list (:2956-2968 — "add generated tokens to
        // cache", ref PR #16818): tokens [n_keep + n_discard, ..) move down by
        // n_discard, the tail shrinks by n_discard. The C rebuilds the prompt
        // through `slot.prompt.clear()` + insert — clear() drops the context
        // checkpoints too (server-task.h:611-616), so the shifted positions
        // never collide with a stale checkpoint
        {
            let mut new_tokens = self.slots[si].prompt_tokens.clone();
            for i in (n_keep + n_discard) as usize..new_tokens.len() {
                new_tokens[i - n_discard as usize] = new_tokens[i];
            }
            new_tokens.truncate(self.slots[si].prompt_tokens.len() - n_discard as usize);
            self.slots[si].prompt_tokens = new_tokens;
            self.slots[si].checkpoints.clear();
        }

        // the speculative draft context still holds the sequence's *unshifted*
        // cells — the reference rebuilds it through the checkpoint machinery
        // the port does not carry (server-context.cpp:3610-3680), so drop the
        // slot's draft sequence + pending drafts and let the next round begin
        // from the shifted prompt (a documented deviation, see PARITY.md)
        if self.spec.is_some() {
            self.slots[si].spec_draft.clear();
            self.slots[si].spec_i_batch.clear();
            if let Some(dft) = self.spec.as_mut().unwrap().ctx_dft() {
                dft.seq_rm(slot_id, -1, -1);
            }
            let prompt = self.slots[si].prompt_tokens.clone();
            self.spec.as_mut().unwrap().begin(slot_id, &prompt);
        }

        self.slots[si].truncated = true;
        Ok(())
    }

    /// `pre_decode` + `post_decode` (server-context.cpp:2930-4120) for one
    /// iteration: build the batch from every active slot, decode it as one
    /// multi-sequence batch, sample per slot.
    fn update_slots(&mut self) -> Result<(), String> {
        // 1. assign queued tasks to slots (`get_available_slot`)
        while let Some(task) = self.pending.pop_front() {
            match self.get_available_slot(&task) {
                Some(i) => self.launch_slot(i, task),
                None => {
                    self.pending.push_front(task);
                    break;
                }
            }
        }

        // 1.5 `pre_decode`'s context-shift iterate (server-context.cpp:
        // 2909-2972) — before the batch is built, shift every generating slot
        // whose context is full. Runs ahead of the token collection like the
        // C's `iterate(slots, ...)` before `batch.clear()`.
        for si in 0..self.slots.len() {
            if self.slots[si].state != SlotState::Generating
                || self.slots[si].task.is_none()
                || self.slots[si].prompt_tokens.len() as i64 + 1 < self.slots[si].n_ctx as i64
            {
                continue;
            }
            // (the multimodal abort :2922-2926 and the shared-prompt error
            // :2928-2932 have no port surface: no mmproj, no n_cmpl children)
            if let Err(e) = self.context_shift(si) {
                self.fail_slot(si, &e, crate::api::ERROR_TYPE_SERVER.1);
            }
        }

        // 2. begin prompts / collect the tokens each slot contributes
        let n_batch = self.n_batch;
        let mut batch = LlamaBatch::default();
        let mut decode: Vec<(usize, usize)> = Vec::new(); // (slot, batch token idx of the sampled row)
        let mut prompt_done: Vec<bool> = vec![false; self.slots.len()];

        // speculative: generate the drafts for the generating slots before the
        // batch is built (server-context.cpp:2995-3067) — one draft() call for
        // every drafting slot, exactly like the C's `drafting` list
        if self.spec.is_some() {
            let mut want_draft: Vec<usize> = Vec::new();
            for si in 0..self.slots.len() {
                if !self.slots[si].is_processing()
                    || self.slots[si].state != SlotState::Generating
                    || self.slots[si].task.is_none()
                {
                    continue;
                }
                let id = self.slots[si].id;
                self.spec.as_mut().unwrap().get_draft_params(id).drafting = false;
                // `slot.get_n_draft_max() > 0` and no draft to reuse
                if self.slots[si].spec_draft.is_empty() && self.get_n_draft_max(si) > 0 {
                    want_draft.push(si);
                }
                // stale candidates: a replay never reads them, a new draft
                // refills them (server-context.cpp:3146-3148)
                self.slots[si].spec_draft_q.clear();
            }
            if !want_draft.is_empty() {
                // the dparams inputs are snapshotted first (the C points at
                // `slot.spec_prompt`; the port owns a copy)
                let infos: Vec<(i32, i32, i32, i32, Vec<i32>)> = want_draft
                    .iter()
                    .map(|&si| {
                        let slot = &self.slots[si];
                        (
                            slot.id,
                            self.get_n_draft_max(si),
                            slot.pos_next(),
                            slot.sampled,
                            slot.prompt_tokens.clone(),
                        )
                    })
                    .collect();
                {
                    // the target context rides along for the gemma4
                    // mem-shared draft-mtp mode
                    let Core::Decode(tgt) = &mut self.core else {
                        unreachable!("spec decoding needs a decode core");
                    };
                    let spec = self.spec.as_mut().unwrap();
                    for (id, n_max, pos0, id_last, prompt) in &infos {
                        // `use_spec_rejection()` = temp > 0: at temp 0 both p
                        // and q are point masses, so rejection is the same as
                        // sample-and-match (server-context.cpp:489-492 +
                        // :3168-3181, upstream a7b94df2c)
                        let si = self.slots.iter().position(|s| s.id == *id).unwrap();
                        let (temp, seed) = {
                            let t = self.slots[si].task.as_ref().unwrap();
                            (t.params.sampling.temp, t.params.sampling.seed)
                        };
                        let dp = spec.get_draft_params(*id);
                        dp.drafting = true;
                        dp.n_max = *n_max;
                        dp.pos0 = *pos0;
                        dp.id_last = *id_last;
                        dp.prompt = prompt.clone();
                        dp.result.clear();
                        if temp > 0.0 {
                            dp.result_q = Some(Vec::new());
                            dp.temp = temp;
                            dp.seed = seed;
                        }
                    }
                    spec.draft(tgt);
                }
                // pull the drafts back into the slots and reset the draft
                // context past the target's pos_max — the cells draft() just
                // wrote (server-context.cpp:3049-3064: `seq_rm(ctx_dft, id,
                // ckpt.pos_max + 1, -1)`)
                for (id, ..) in &infos {
                    let (draft, draft_q) = {
                        let dp = self.spec.as_mut().unwrap().get_draft_params(*id);
                        (std::mem::take(&mut dp.result), dp.result_q.take())
                    };
                    let si = self.slots.iter().position(|s| s.id == *id).unwrap();
                    self.slots[si].spec_draft = draft;
                    if let Some(q) = draft_q {
                        self.slots[si].spec_draft_q = q;
                    }
                    // `slot.stats.n_draft_tokens += draft.size()`
                    // (server-context.cpp:3057)
                    self.slots[si].stats.n_draft_tokens += self.slots[si].spec_draft.len() as u64;
                }
                for si in want_draft {
                    if self.slots[si].spec_draft.is_empty() {
                        continue;
                    }
                    let id = self.slots[si].id;
                    let pos_max = self.dctx().map(|d| d.seq_pos_max(id)).unwrap_or(0);
                    if let Some(dft) = self.spec.as_mut().unwrap().ctx_dft() {
                        dft.seq_rm(id, pos_max + 1, -1);
                    }
                }
            }
        }

        for si in 0..self.slots.len() {
            if !self.slots[si].is_processing() || batch.token.len() >= n_batch {
                continue;
            }
            let state = self.slots[si].state;
            if state == SlotState::Started {
                if let Err(e) = self.begin_prompt(si) {
                    self.fail_slot(si, &e.0, e.1);
                    continue;
                }
                if self.slots[si].state == SlotState::Idle {
                    continue; // the prompt was empty / rejected
                }
            }
            if self.slots[si].state == SlotState::ProcessingPrompt
                || self.slots[si].state == SlotState::Started
            {
                let slot_id = self.slots[si].id;
                let n_tokens_task = self.slots[si].task.as_ref().unwrap().tokens.len();

                // ---- the checkpoint gates (server-context.cpp:3865-3878) ----
                // `bool do_checkpoint = params_base.n_ctx_checkpoints > 0`
                // (only completion tasks reach the slot machinery; the
                // port's embedding/decision tasks decode inline)
                // `&& (seq_rm FULL || RS || n_swa > 0)` — only the memories
                // that cannot roll back a partial sequence need a checkpoint
                let mut do_checkpoint =
                    self.n_ctx_checkpoints > 0 && (self.seq_rm_bounded || self.n_swa > 0);
                // `spans.is_user_start` / `spans.last_user_message_pos`
                // (common/chat.h:168-182) over the task's message spans
                let user_starts: Vec<usize> = self.slots[si]
                    .task
                    .as_ref()
                    .unwrap()
                    .params
                    .message_user_starts
                    .clone();
                let last_user_pos = user_starts.last().map(|&p| p as i32).unwrap_or(-1);
                let is_user_start_pos = move |pos: usize| user_starts.binary_search(&pos).is_ok();
                // `n_ubatch` — the decode core's per-call cap
                let n_ubatch = match &self.core {
                    Core::Decode(d) => d.n_batch as i32,
                    _ => 512,
                };

                let n_tokens_prev = batch.token.len();
                let mut last = None;
                while self.slots[si].prompt_tokens.len() < n_tokens_task && batch.token.len() < n_batch {
                    let tok = self.slots[si].task.as_ref().unwrap().tokens
                        [self.slots[si].prompt_tokens.len()];
                    let pos = self.slots[si].pos_next();
                    // `batch.add(slot.id, tok, pos, /*output=*/false, /*is_prompt=*/true)`
                    batch.add(tok, pos, &[slot_id], false);
                    self.slots[si].prompt_tokens.push(tok);
                    last = Some(batch.token.len() - 1);

                    // break at the last user message, or at user messages at
                    // least min step past the last checkpoint
                    // (server-context.cpp:3977-3983)
                    if do_checkpoint && is_user_start_pos(self.slots[si].prompt_tokens.len()) {
                        let pos = self.slots[si].prompt_tokens.len() as i64;
                        let back = self.slots[si].checkpoints.back().map(|c| c.n_tokens as i64);
                        if pos as i32 == last_user_pos
                            || back.is_none()
                            || pos > back.unwrap() + self.checkpoint_min_step as i64
                        {
                            break;
                        }
                    }

                    // process the last few tokens of the prompt separately so
                    // a checkpoint can be created at their head —
                    // {4 + n_ubatch, 4} before the end (PR #20288,
                    // server-context.cpp:3985-3999)
                    if do_checkpoint {
                        let mut should_break = false;
                        for offset in [4 + n_ubatch, 4] {
                            let n_last = n_batch.min(offset as usize);
                            if n_tokens_task == self.slots[si].prompt_tokens.len() + n_last {
                                should_break = true;
                                break;
                            }
                        }
                        if should_break {
                            break;
                        }
                    }
                }
                let n_tokens_cur = batch.token.len() - n_tokens_prev;
                let n_tokens_start = self.slots[si].prompt_tokens.len() - n_tokens_cur;

                // `near_prompt_end` / `is_user_start` of this batch's head
                // (server-context.cpp:4028-4029)
                let near_prompt_end =
                    (n_tokens_task as i32) < (self.slots[si].prompt_tokens.len() as i32 + n_ubatch);
                let is_user_start = is_user_start_pos(n_tokens_start);
                let is_last_user_message = n_tokens_start as i32 == last_user_pos;

                // skip ordinary mid-prompt checkpoints, unless the batch
                // starts a user message or we are near the end of the prompt
                // (server-context.cpp:4031-4035)
                if self.slots[si].prompt_tokens.len() != n_tokens_task
                    && !is_user_start
                    && !near_prompt_end
                {
                    do_checkpoint = false;
                }

                // nothing to checkpoint yet (server-context.cpp:4041-4044);
                // no mtmd chunks in the port (the :4047 gate)
                if do_checkpoint && self.mem_pos_min(slot_id) < 0 {
                    do_checkpoint = false;
                }

                // no need to create checkpoints that are too close together,
                // unless it's the last user message or the prompt is ending
                // (server-context.cpp:4050-4055)
                let back_n = self.slots[si].checkpoints.back().map(|c| c.n_tokens as i64);
                if do_checkpoint
                    && back_n.is_some()
                    && !is_last_user_message
                    && !near_prompt_end
                    && (n_tokens_start as i64) <= back_n.unwrap() + self.checkpoint_min_step as i64
                {
                    do_checkpoint = false;
                }

                // note: the checkpoint is taken before the batch decodes, so
                // its state does not cover the batch's tokens
                // (server-context.cpp:4057-4061)
                if do_checkpoint {
                    self.create_checkpoint(si, n_tokens_cur);
                }

                if self.slots[si].prompt_tokens.len() == n_tokens_task {
                    if let Some(idx) = last {
                        // `batch.set_output(batch.size() - 1, true)` (server-context.cpp:3591)
                        batch.logits.as_mut().unwrap()[idx] = true;
                        prompt_done[si] = true;
                        self.slots[si].i_batch = idx as i32;
                        decode.push((si, idx));
                    }
                }
            } else if state == SlotState::Generating {
                // `slot.handle_last_sampled_token(batch)` (server-context.cpp:505-541)
                let slot_id = self.slots[si].id;
                let tok = self.slots[si].sampled;
                let pos = self.slots[si].pos_next();
                let n_spec = self.slots[si].spec_draft.len();
                if n_spec > 0 && batch.token.len() + 1 + n_spec <= n_batch {
                    // speculative round: [sampled, draft...] all carry output
                    // flags; spec_i_batch records their batch indices
                    // (server-context.cpp:521-541)
                    let draft = self.slots[si].spec_draft.clone();
                    debug_assert!(self.slots[si].spec_i_batch.is_empty());
                    let idx = batch.token.len();
                    self.slots[si].spec_i_batch.push(idx as i32);
                    for i in 0..draft.len() {
                        self.slots[si].spec_i_batch.push((idx + i + 1) as i32);
                    }
                    batch.add(tok, pos, &[slot_id], true);
                    let mut p = pos + 1;
                    for &d in &draft {
                        batch.add(d, p, &[slot_id], true);
                        p += 1;
                    }
                    // `prompt.tokens.push_back(sampled); prompt.tokens.insert(spec_draft)`
                    self.slots[si].prompt_tokens.push(tok);
                    self.slots[si].prompt_tokens.extend_from_slice(&draft);
                    self.slots[si].i_batch = -1;
                    decode.push((si, idx));
                } else {
                    // no speculative decoding (no draft, or it does not fit)
                    batch.add(tok, pos, &[slot_id], true);
                    self.slots[si].prompt_tokens.push(tok);
                    let idx = batch.token.len() - 1;
                    self.slots[si].i_batch = idx as i32;
                    decode.push((si, idx));
                }
            }
        }

        if batch.token.is_empty() {
            return Ok(());
        }

        // 3. decode — one batched call for every active slot (multi-sequence)
        // SRV_DBG-level diagnostic (LLAMA_SERVER_DEBUG=1): how many sequences
        // the batched decode carries — the reference reports the same as
        // "n_tokens = %d, n_seqs = %d" (server-context.cpp:1343)
        if std::env::var("LLAMA_SERVER_DEBUG").is_ok() {
            let mut seqs: Vec<i32> = batch
                .seq_id
                .as_ref()
                .map(|s| s.iter().map(|ids| ids[0]).collect())
                .unwrap_or_default();
            seqs.sort_unstable();
            seqs.dedup();
            eprintln!(
                "llama-server: decode n_tokens = {}, n_seqs = {:?}",
                batch.token.len(),
                seqs
            );
        }
        // `llama_set_embeddings(ctx_tgt, slot_batched->need_embd())`
        // (server-context.cpp:2864) — the slot-batched decode carries no
        // embedding task in the port (they complete inline through
        // `run_embedding`), so generation batches always decode with
        // embeddings off
        if let Some(d) = self.dctx() {
            d.set_embeddings(false);
        }
        let out = match self.dctx() {
            Some(d) => d.decode_batch(&batch)?,
            None => {
                return Err("this server holds an encoder-only model; /completion is unavailable"
                    .into())
            }
        };

        // the post-decode prompt-bookkeeping loop (server-context.cpp:4595-
        // 4606): every batch row of a prompt-processing slot bumps the
        // slot's `n_prompt_processed` — the `timings.prompt_n` the client
        // reads. The C reads the batch row's `is_prompt` flag (set by
        // `batch.add(..., /* is_prompt = */ true)`); the port's equivalent
        // is the slot's state — a ProcessingPrompt slot contributed only
        // prompt rows to this batch (the last one carries the output flag
        // and still counts, like the C's is_prompt row)
        {
            let seqs = batch.seq_id.as_ref();
            for i in 0..batch.token.len() {
                let Some(ids) = seqs.map(|s| &s[i]) else { continue };
                let Some(&slot_id) = ids.first() else { continue };
                let Some(si) = self.slots.iter().position(|s| s.id == slot_id) else {
                    continue;
                };
                if self.slots[si].state == SlotState::ProcessingPrompt
                    && self.slots[si].stats.is_set()
                {
                    self.slots[si].stats.n_prompt_processed += 1;
                }
            }
        }

        // `common_speculative_process(spec, batch_view)` — feed the decoded
        // batch (prompt or generation) to the speculator, which decodes it on
        // the draft context (server-context.cpp:3742-3757). The target context
        // rides along for draft-mtp's embeddings_nextn reads.
        if self.spec.is_some() {
            let Core::Decode(dctx) = &self.core else {
                return Err("speculative decoding needs a decoder model".into());
            };
            if !self.spec.as_mut().unwrap().process(&batch, dctx) {
                return Err("failed to process speculative batch".into());
            }
        }

        // 4. post_decode: sample + process per slot
        for (si, idx) in decode {
            if prompt_done[si] {
                self.slots[si].state = SlotState::Generating;
                self.slots[si].stats.n_gen = 0;
                // `common_speculative_begin(spec, slot.id, prompt.tokens)`
                // (server-context.cpp:3838-3841)
                if self.spec.is_some() {
                    let prompt = self.slots[si].prompt_tokens.clone();
                    let id = self.slots[si].id;
                    self.spec.as_mut().unwrap().begin(id, &prompt);
                }
            }
            self.post_decode(si, idx as i32, &out)?;
        }
        Ok(())
    }

    /// The `SLOT_STATE_STARTED` branch of `pre_decode` (server-context.cpp:
    /// 3142-3413): cache-prompt prefix reuse, the "evaluate at least 1 token"
    /// rule, the context-size checks and the truncation of the cached tail.
    fn begin_prompt(&mut self, si: usize) -> Result<(), (&'static str, i64)> {
        use crate::api::{ERROR_TYPE_EXCEED_CONTEXT_SIZE, ERROR_TYPE_INVALID_REQUEST};

        self.slots[si].state = SlotState::ProcessingPrompt;
        self.slots[si].stats.update_prompt_start();
        self.slots[si].stats.n_gen = 0;

        let n_task = self.slots[si].task.as_ref().unwrap().tokens.len();
        if n_task == 0 {
            // `empty prompt - releasing slot`
            self.send_final(si);
            self.slots[si].release();
            return Ok(());
        }
        if n_task as u32 > self.slots[si].n_ctx {
            return Err((
                "input is larger than the max context size",
                ERROR_TYPE_EXCEED_CONTEXT_SIZE.1,
            ));
        }

        let mut n_past = 0usize;
        if self.slots[si].task.as_ref().unwrap().params.cache_prompt {
            // `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`
            let cached = self.slots[si].prompt_tokens.clone();
            let input = self.slots[si].task.as_ref().unwrap().tokens.clone();
            n_past = common_prefix_len(&cached, &input);

            // ---- the checkpoint rollback (server-context.cpp:3666-3784, PR
            // #24110) ---- when the KV's oldest live cell sits past what the
            // new prompt needs (the swa window evicted the prefix, or the
            // memory cannot roll back at all), search the checkpoint list
            // newest-first for one the sequence can restart from
            let id = self.slots[si].id;
            let mut pos_next = n_past as i32;
            let has_new_tokens = n_past < n_task;
            // `pos_min_thold = max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1))`
            // (server-context.cpp:3674)
            let pos_min_thold =
                (pos_next - self.n_swa - if has_new_tokens { 0 } else { 1 }).max(0);

            if n_past > 0 {
                let pos_min = self.mem_pos_min(id);
                if pos_min >= pos_min_thold {
                    // find_if over the reversed list: `pos_max <= pos_next`
                    // and (`pos_min < pos_min_thold` or `pos_min == 0`) —
                    // guarantee at least one token gets processed
                    // ([TAG_PROMPT_LOGITS] workaround of
                    // [TAG_CHECKPOINTS_FIX_POS_MIN],
                    // server-context.cpp:3728-3740)
                    let restore = self.slots[si]
                        .checkpoints
                        .iter()
                        .rev()
                        .find(|c| c.pos_max <= pos_next && (c.pos_min < pos_min_thold || c.pos_min == 0))
                        .cloned();
                    let mut do_reset = true;
                    let partial_only = self.ckpt_partial_only();
                    if let Some(c) = restore.as_ref() {
                        let ok = match &mut self.core {
                            Core::Decode(d) => {
                                d.state_seq_set_data(id, &c.data_tgt, partial_only).is_ok()
                            }
                            _ => false,
                        } && match self.spec.as_mut().and_then(|s| s.ctx_dft()) {
                            Some(dft) if !c.data_dft.is_empty() => {
                                dft.state_seq_set_data(id, &c.data_dft, partial_only).is_ok()
                            }
                            _ => true,
                        };
                        if ok {
                            do_reset = false;
                            // `pos_next = min(pos_next, max(pos_min + 1, pos_max))`;
                            // `n_past = min(size_up_to_pos(pos_next), n_tokens)`
                            // (server-context.cpp:3755-3757 — the positions are
                            // the token indices in the port, so
                            // `size_up_to_pos(p) == p`)
                            pos_next = pos_next.min((c.pos_min + 1).max(c.pos_max));
                            n_past = (pos_next.max(0) as usize).min(c.n_tokens);
                        } else if c.id_task != -1 {
                            // `GGML_ABORT("failed to restore context checkpoint")`
                            // (server-context.cpp:3748-3750) — a checkpoint
                            // this process created must load
                            return Err((
                                "failed to restore context checkpoint",
                                crate::api::ERROR_TYPE_SERVER.1,
                            ));
                        } else {
                            // restored from a slot file, not guaranteed to
                            // load — fall back to full prompt re-processing
                            // (server-context.cpp:3751-3753)
                        }
                    }
                    if do_reset {
                        // "forcing full prompt re-processing due to lack of
                        // cache data" (server-context.cpp:3765-3767)
                        n_past = 0;
                    }
                }

                // erase any checkpoints with pos_max > pos_next — they cover
                // positions the new prompt no longer holds
                // (server-context.cpp:3775-3786)
                self.slots[si].checkpoints.retain(|c| c.pos_max <= pos_next);
            }
        }

        // `[TAG_PROMPT_LOGITS]` (server-context.cpp:3401-3406): the last prompt
        // token must be evaluated, otherwise there are no logits to sample from
        if n_past == n_task && n_past > 0 {
            n_past -= 1;
        }

        self.slots[si].stats.n_prompt_cached = n_past as u64;
        self.slots[si].stats.n_prompt_processed = 0;
        self.slots[si].prompt_tokens.truncate(n_past);

        // "this is to signal the client that the request has started": with
        // `return_progress` the reference sends a progress frame here; without
        // it the frame only flushes the HTTP headers (`is_begin`, whose
        // `to_json()` returns null and is therefore never written to the SSE
        // body — server-task.cpp:1026-1030), so nothing is emitted.
        if self.slots[si].task.as_ref().unwrap().params.stream
            && self.slots[si].task.as_ref().unwrap().params.return_progress
            && !self.slots[si].sent_begin
        {
            self.slots[si].sent_begin = true;
            self.send_partial(si, None, true).ok();
        }

        // `slot.mem.seq_rm(slot.id, p0, -1)` — drop the cached tail of the
        // sequence so the prompt is re-processed from n_past; `slot.mem` wraps
        // both the target and the draft context (server-context.cpp:546-549)
        let p0 = self.slots[si].pos_next();
        let id = self.slots[si].id;
        if let Some(d) = self.dctx() {
            d.seq_rm(id, p0, -1);
        }
        if let Some(spec) = self.spec.as_mut() {
            if let Some(dft) = spec.ctx_dft() {
                dft.seq_rm(id, p0, -1);
            }
        }

        let task_n = self.slots[si].task.as_ref().unwrap().n_tokens();
        if task_n as u32 >= self.slots[si].n_ctx {
            return Err((
                "request exceeds the available context size, try increasing it",
                ERROR_TYPE_EXCEED_CONTEXT_SIZE.1,
            ));
        }
        let _ = ERROR_TYPE_INVALID_REQUEST;
        Ok(())
    }

    fn fail_slot(&mut self, si: usize, msg: &str, code: i64) {
        let type_str = if code == 400 { "invalid_request_error" } else { "server_error" };
        if let Some(t) = self.slots[si].task.take() {
            let _ = t.tx.send(StreamEvent::Frame(error_frame(crate::api::json_error(
                msg, type_str, code,
            ))));
            let _ = t.tx.send(StreamEvent::Done);
        }
        self.slots[si].release();
    }

    /// `post_decode` (server-context.cpp:3784-3900) for one slot: sample from
    /// the row `tok_idx` of the batch's logits and run `process_token`.
    fn post_decode(&mut self, si: usize, tok_idx: i32, out: &BatchOutput) -> Result<(), String> {
        if self.slots[si].state != SlotState::Generating {
            return Ok(());
        }
        // "sample using speculative decoding" (server-context.cpp:3846-3848):
        // a slot with a pending draft skips the plain sample
        if !self.slots[si].spec_draft.is_empty() && !self.slots[si].spec_i_batch.is_empty() {
            return self.post_decode_spec(si, out);
        }
        let logits = out
            .logits_ith(tok_idx)
            .ok_or_else(|| format!("no logits for batch index {tok_idx}"))?;

        // `common_sampler_sample(slot.smpl, ctx, tok_idx)` — the grammar
        // sampler is applied like common_sampler_sample's `grammar_first=false`
        // path (sample, then re-sample under the grammar if rejected); a lazy
        // grammar awaiting its trigger runs through the port of
        // `llama_grammar_accept_impl`'s trigger branch instead
        let mut tok = {
            let slot = &mut self.slots[si];
            let awaiting =
                slot.grammar.as_ref().map(|g| g.grammar.awaiting_trigger).unwrap_or(false);
            match (slot.sampler.as_mut(), slot.grammar.as_mut(), slot.lazy.as_mut()) {
                (Some(smpl), Some(g), Some(triggers)) if awaiting => {
                    sample_with_lazy_grammar(smpl, logits, g, triggers)
                        .map_err(|e| format!("grammar sampling failed: {e}"))?
                }
                (Some(smpl), Some(g), _) => smpl
                    .sample_with_grammar(logits, g)
                    .map_err(|e| format!("grammar sampling failed: {e}"))?,
                (Some(smpl), None, _) => smpl.sample(logits),
                (None, _, _) => crate::api::argmax(logits),
            }
        };

        // `common_sampler_accept(smpl, id, true)` (server-context.cpp:3861) —
        // in the port every sample path already performs the accept inside
        // (`SamplingContext::sample` / `sample_with_grammar` /
        // `sample_with_lazy_grammar` mirror `llama_sampler_sample`, which
        // accepts the selected token), so an explicit accept here would feed
        // the ring-based samplers (penalties / DRY) twice per token
        // the sampled token is 0-based normalised like the C's token ids
        let _ = &mut tok;

        self.slots[si].stats.n_gen += 1;
        if self.slots[si].stats.n_gen == 1 {
            self.slots[si].stats.update_prompt_last();
        }
        self.slots[si].stats.update_gen_last();
        self.slots[si].sampled = tok;

        // per-token probabilities (server-context.cpp:1964-2010)
        let n_probs = self.slots[si].task.as_ref().unwrap().params.n_probs as usize;
        if n_probs > 0 {
            let post = self.slots[si].task.as_ref().unwrap().params.post_sampling_probs;
            let pieces: Vec<(i32, String, f32)> = if post {
                let cands = self.slots[si].sampler.as_ref().unwrap().cur.clone();
                let mut td = llama::sampling::TokenDataArray::from_logits(logits);
                td.data = cands;
                td.size = td.data.len();
                let p = probs_from_candidates(&td, tok, n_probs, &|id| {
                    self.vocab.token_to_piece(id).to_string()
                });
                p.probs
            } else {
                let probs = crate::api::get_token_probabilities(logits, n_probs);
                probs
                    .iter()
                    .take(n_probs)
                    .map(|&(id, p)| (id, self.vocab.token_to_piece(id).to_string(), p))
                    .collect()
            };
            let prob = pieces.iter().find(|(id, _, _)| *id == tok).map(|(_, _, p)| *p).unwrap_or(1.0);
            self.slots[si].probs_output.push(TokenProbs {
                tok,
                prob,
                probs: pieces,
                text_to_send: crate::api::token_piece(&self.vocab, tok, false),
            });
        }

        let has_next = self.process_token(si, tok)?;
        if !has_next {
            self.send_final(si);
            self.slots[si].release();
        }
        Ok(())
    }

    /// The speculative sample-and-accept round of `post_decode`
    /// (server-context.cpp:3897-4017): verify the draft against the target's
    /// batch rows, commit the accepted tokens, trim both KV caches.
    fn post_decode_spec(&mut self, si: usize, out: &BatchOutput) -> Result<(), String> {
        // save the original draft size (server-context.cpp:3904-3906)
        let n_draft = self.slots[si].spec_draft.len();
        assert!(n_draft > 0, "spec round without a draft");
        assert_eq!(self.slots[si].spec_i_batch.len(), n_draft + 1);
        // `GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size()
        // == slot.spec_draft.size()))` (server-context.cpp:4123-4124)
        assert!(
            self.slots[si].spec_draft_q.is_empty()
                || self.slots[si].spec_draft_q.len() == self.slots[si].spec_draft.len()
        );

        // `common_sampler_sample_and_accept_n(smpl, ctx, spec_i_batch,
        // spec_draft)` (server-context.cpp:3913-3919) — the C's idxs overload
        // reads the logits rows at the recorded batch indices; the port builds
        // the sub-`BatchOutput` those rows form and runs the same rule.
        //
        // The four-way dispatch of a7b94df2c (:4120-4140): synth first (the
        // port's server has no synth probe family), then rejection when the
        // drafter filled a distribution and the target samples at temp > 0,
        // then the plain sample-and-match. `spec_is_replay` (the checkpoint
        // restore's re-accept) is not ported — the port's cache always
        // supports partial seq_rm, the direct accept path is what the
        // reference takes for full-attention contexts.
        let use_rejection = {
            let slot = &self.slots[si];
            let temp = slot.task.as_ref().map(|t| t.params.sampling.temp).unwrap_or(0.0);
            temp > 0.0 && !slot.spec_draft_q.is_empty()
        };
        let accepted = {
            let idxs = std::mem::take(&mut self.slots[si].spec_i_batch);
            let n_vocab = out.n_vocab;
            let mut logits = Vec::with_capacity(idxs.len() * n_vocab);
            for &ib in &idxs {
                let row = out
                    .logits_ith(ib)
                    .ok_or_else(|| format!("no logits for batch index {ib}"))?;
                logits.extend_from_slice(row);
            }
            let n_rows = idxs.len();
            let sub = BatchOutput {
                logits,
                n_outputs: n_rows,
                output_ids: (0..n_rows as i64).collect(),
                n_tokens: n_rows,
                n_vocab,
            };
            let slot = &mut self.slots[si];
            let smpl = slot.sampler.as_mut().ok_or("slot sampler gone")?;
            if use_rejection {
                // `common_sampler_sample_and_accept_n_rejection(smpl, ctx,
                // spec_i_batch, spec_draft, spec_draft_q)`
                // (server-context.cpp:4133-4135) — grammar_first=false; the
                // port's slot grammar rides its own lazy path
                let draft_q = std::mem::take(&mut slot.spec_draft_q);
                smpl.sample_and_accept_n_rejection(&sub, &slot.spec_draft, &draft_q, None, false)
            } else {
                smpl.sample_and_accept_n(&self.vocab, &sub, &slot.spec_draft)
            }
        };
        assert!(!accepted.is_empty(), "there will always be at least one accepted token");

        // the checkpoint restore of partial acceptance (server-context.cpp:
        // 3927-3963) is not ported: it is only taken for
        // COMMON_CONTEXT_SEQ_RM_TYPE_FULL contexts, and the port's KV cache
        // always supports partial removal — the direct accept path below is
        // what the reference runs for its full-attention contexts

        // `common_speculative_accept(spec, slot.id, accepted.size() - 1)` (:3961)
        let id = self.slots[si].id;
        self.spec
            .as_mut()
            .ok_or("speculator gone")?
            .accept(id, (accepted.len() - 1) as u16);

        // update how many tokens out of those tested were accepted
        // (server-context.cpp:3970-3979)
        let n_accepted = accepted.len() - 1;
        self.slots[si].stats.n_draft_accepted += n_accepted as u64;
        self.slots[si].stats.n_draft_verif_steps += 1;

        // the accepted tokens become the new prompt tail; the last one is the
        // next "sampled" token (:3989-3996)
        let n_tokens = self.slots[si].prompt_tokens.len();
        self.slots[si].prompt_tokens.truncate(n_tokens - n_draft);
        self.slots[si]
            .prompt_tokens
            .extend_from_slice(&accepted[..accepted.len() - 1]);
        self.slots[si].sampled = *accepted.last().unwrap();
        self.slots[si].spec_draft.clear();
        self.slots[si].spec_i_batch.clear();

        // `slot.mem.seq_rm(slot.id, prompt.tokens.pos_next(), -1)` — drop the
        // unaccepted tail of *both* KV caches (:3998)
        let p1 = self.slots[si].pos_next();
        if let Some(d) = self.dctx() {
            d.seq_rm(id, p1, -1);
        }
        if let Some(dft) = self.spec.as_mut().unwrap().ctx_dft() {
            dft.seq_rm(id, p1, -1);
        }

        // process the accepted tokens (:3999-4016) — the standard token
        // post-processing, done for a group at once
        let n_probs = self.slots[si].task.as_ref().unwrap().params.n_probs as usize;
        self.slots[si].stats.update_gen_last();
        for &t in &accepted {
            self.slots[si].stats.n_gen += 1;
            // `result.prob = 1.0f; // set later` — the C leaves the spec
            // tokens' probabilities unset (its TODO at :4004-4006)
            if n_probs > 0 {
                self.slots[si].probs_output.push(TokenProbs {
                    tok: t,
                    prob: 1.0,
                    probs: Vec::new(),
                    text_to_send: crate::api::token_piece(&self.vocab, t, false),
                });
            }
            let has_next = self.process_token(si, t)?;
            if !has_next {
                self.send_final(si);
                self.slots[si].release();
                return Ok(());
            }
        }
        Ok(())
    }

    /// `server_slot::get_n_draft_max` (server-context.cpp:478-497): the max
    /// draft that fits the remaining context and generation budget. The slot's
    /// prompt is not yet expanded with the sampled token, and one extra cell is
    /// kept free for the context shift.
    fn get_n_draft_max(&self, si: usize) -> i32 {
        if self.spec.is_none() {
            return 0;
        }
        let slot = &self.slots[si];
        let mut n_draft_max = slot.n_ctx as i32 - slot.prompt_tokens.len() as i32 - 2;
        // `slot.n_remaining()` = `n_predict_max - stats.n_gen` (-1 = unlimited)
        let n_remaining = if slot.n_predict_max < 0 {
            -1
        } else {
            slot.n_predict_max - slot.stats.n_gen as i32
        };
        if n_remaining > 0 {
            n_draft_max = n_draft_max.min(n_remaining - 1);
        }
        n_draft_max
    }

    /// `process_token` (server-context.cpp:1833-1960): the generated text, the
    /// stop-word/UTF-8 handling and every stop condition. The partial frame is
    /// sent *before* the limit/EOS stops are computed (the C's
    /// `send_partial_response` sits right after `slot.add_token`, :1881-1884),
    /// so only a stop *word* (which `find_stopping_strings` sets earlier) can
    /// attach timings to a partial — the LIMIT/EOS stops land on the final.
    fn process_token(&mut self, si: usize, tok: i32) -> Result<bool, String> {
        // the token's text (`common_token_to_piece(ctx, tok, accept_special_token(slot, tok))`
        // — server-context.cpp:3800-3802/3878: a CONTROL token renders its
        // piece only for `params_base.special` (always false here) or when it
        // is one of the request's preserved markers, so the template's
        // `<tool_call>`-style markers survive detokenization)
        let accept_special = self.slots[si]
            .task
            .as_ref()
            .map(|t| t.params.preserved_tokens.contains(&tok))
            .unwrap_or(false);
        let token_str = crate::api::token_piece(&self.vocab, tok, accept_special);
        // `slot.generated.append(token_str, result.tok)` (server-context.cpp:1971,
        // 18b5f8b18) — the generated text *and* tokens accumulate together
        self.slots[si].generated.append_piece(&token_str, tok);
        if self.slots[si].task.as_ref().unwrap().params.return_tokens {
            self.slots[si].generated_tokens.push(tok);
        }
        self.slots[si].has_next_token = true;

        let mut text_to_send = String::new();
        let mut incomplete = false;
        if !self.slots[si].generated.text.is_empty() {
            incomplete =
                validate_utf8_len(&self.slots[si].generated.text) < self.slots[si].generated.text.len();
            if !incomplete {
                let mut pos = self.slots[si].n_sent_text.min(self.slots[si].generated.text.len());
                let str_test = self.slots[si].generated.text[pos..].to_string();
                let mut send_text = true;

                // `find_stopping_strings(str_test, token_str.size(), is_full_stop = true)`
                // (server-context.cpp:1986-1992): a full stop truncates the
                // generated input at the stop word (`truncate(pos + stop_pos)`,
                // 18b5f8b18) — `send_text` stays true so the text *before* the
                // stop word still flows out in this partial
                if let Some(stop_pos) =
                    self.find_stopping_strings(si, &str_test, token_str.len(), true)
                {
                    self.slots[si].generated.truncate(pos + stop_pos);
                    pos = self.slots[si].n_sent_text.min(self.slots[si].generated.text.len());
                } else if self.slots[si].has_next_token && !self.vocab.is_eog(tok) {
                    // a partial stop word at the end: hold the text back
                    send_text = self.find_stopping_strings(si, &str_test, token_str.len(), false)
                        .is_none();
                }

                if send_text {
                    // no send the stop word in the response
                    text_to_send = self.slots[si].generated.text[pos..].to_string();
                    self.slots[si].n_sent_text += text_to_send.len();
                }
            }
        }

        let stream = self.slots[si].task.as_ref().unwrap().params.stream;
        // the C sends the partial here (server-context.cpp:1881-1884) — an
        // incomplete UTF-8 tail suppresses the frame entirely
        if !incomplete && stream {
            self.send_partial(si, Some((tok, text_to_send.clone())), false)?;
        }
        if incomplete {
            self.slots[si].has_next_token = true;
        }

        // stop conditions (server-context.cpp:1887-1953)
        if !self.ctx_shift && self.slots[si].pos_next() + 1 >= self.slots[si].n_ctx as i32 {
            self.slots[si].truncated = true;
            self.slots[si].stop = StopType::Limit;
            self.slots[si].has_next_token = false;
        }
        if self.slots[si].stats.n_gen > 0
            && self.slots[si].has_next_token
            && !self.slots[si].has_budget()
        {
            self.slots[si].stop = StopType::Limit;
            self.slots[si].has_next_token = false;
        }
        if text_to_send.contains('\n') {
            self.slots[si].has_new_line = true;
            let t_max = self.slots[si].task.as_ref().unwrap().params.t_max_predict_ms;
            if t_max > 0 && self.slots[si].stats.t_gen_ms() > t_max as f64 {
                self.slots[si].stop = StopType::Limit;
                self.slots[si].has_next_token = false;
            }
        }
        let ignore_eos = self.slots[si].task.as_ref().unwrap().params.ignore_eos;
        if !ignore_eos && self.vocab.is_eog(tok) {
            self.slots[si].stop = StopType::Eos;
            self.slots[si].has_next_token = false;
        }

        Ok(self.slots[si].has_next_token)
    }

    /// `server_slot::find_stopping_strings` (server-context.cpp:466-497):
    /// `is_full_stop` also sets the stop type and the stopping word.
    fn find_stopping_strings(
        &mut self,
        si: usize,
        text: &str,
        last_token_size: usize,
        is_full_stop: bool,
    ) -> Option<usize> {
        let mut stop_pos: Option<usize> = None;
        let words: Vec<String> = self.slots[si].task.as_ref()?.params.antiprompt.clone();
        for word in words.iter() {
            let pos = if is_full_stop {
                let tmp = word.len() + last_token_size;
                let from_pos = text.len().saturating_sub(tmp);
                text[from_pos..].find(word).map(|p| p + from_pos)
            } else {
                string_find_partial_stop(text, word)
            };
            if let Some(pos) = pos {
                if stop_pos.is_none() || pos < stop_pos.unwrap() {
                    if is_full_stop {
                        self.slots[si].stop = StopType::Word;
                        self.slots[si].stopping_word = word.clone();
                        self.slots[si].has_next_token = false;
                    }
                    stop_pos = Some(pos);
                }
            }
        }
        stop_pos
    }

    /// `send_partial_response` (server-context.cpp:2046-2135) + the result's
    /// `to_json` (server-task.cpp:1026-1164). `tkn` is the token this event
    /// reports (None for a progress event). The res_type selects the shape:
    /// the port's native frames, the OAI text-completion chunk or the OAI chat
    /// chunks (`to_json_oaicompat_chat` returns one chunk per delta — the C
    /// sends them as consecutive SSE frames via `format_oai_sse`).
    fn send_partial(&mut self, si: usize, tkn: Option<(i32, String)>, is_progress: bool) -> Result<(), String> {
        // the task is cloned out first — the slot is mutated below (the chat
        // parse advances the per-slot parse state)
        let Some(task) = self.slots[si].task.clone() else { return Ok(()) };
        if !task.params.stream {
            return Ok(());
        }
        if is_progress && !task.params.return_progress {
            return Ok(());
        }
        // `res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(),
        // tkn.text_to_send.size())` (server-context.cpp:2192-2194, 18b5f8b18) —
        // the partial's content is the token-aligned chunk that was just sent
        // (`n_sent_text` was already advanced in `process_token`)
        let (content, tokens) = match tkn {
            Some((tok, text)) => {
                let start = self.slots[si].n_sent_text - text.len();
                (self.slots[si].generated.substr(start, Some(text.len())), vec![tok])
            }
            None => (llama::chat_tools::ChatInput::default(), Vec::new()),
        };
        // `state.update_chat_msg(content, true, oaicompat_msg_diffs)`
        // (server-task.cpp:988-993): re-parse the accumulated text and diff
        // against the previous message — the OAI chat deltas are the diffs
        // (`server_chat_msg_diff_to_json_oaicompat`)
        let chat_diffs = if task.params.res_type == ResponseType::OaiChat {
            let state = self.slots[si].chat.as_mut().expect("chat task without parse state");
            Some(state.update_chat_msg(&content, true)?)
        } else {
            None
        };
        let slot = &self.slots[si];
        // `if (slot.stop != STOP_TYPE_NONE || timings_per_token) res->stats = …`
        // (server-context.cpp:2130-2133) — evaluated at the C's send position,
        // so only a stop *word* (found before the send) can carry timings
        let stats_json = if slot.stats.is_set()
            && (slot.stop != StopType::None || task.params.timings_per_token)
        {
            Some(slot.stats.to_json())
        } else {
            None
        };
        let prob_output = slot.probs_output.last().filter(|p| !p.probs.is_empty());
        let post_sampling = task.params.post_sampling_probs;
        // the logprobs object of an OAI chunk (`prob_output.probs.size() > 0`)
        let logprobs_json = |p: &TokenProbs| {
            Json::Object(vec![(
                "content".into(),
                probs_vector_to_json(std::slice::from_ref(p), post_sampling),
            )])
        };

        match task.params.res_type {
            ResponseType::OaiCmpl => {
                // `to_json_oaicompat` (server-task.cpp:1073-1109)
                let t = unix_now();
                let mut fields: Vec<(String, Json)> = vec![
                    (
                        "choices".into(),
                        Json::Array(vec![Json::Object(vec![
                            ("text".into(), Json::String(content.text.clone())),
                            ("index".into(), Json::Int(task.index as i64)),
                            (
                                "logprobs".into(),
                                prob_output
                                    .map(logprobs_json)
                                    .unwrap_or(Json::Null),
                            ),
                            ("finish_reason".into(), Json::Null),
                        ])]),
                    ),
                    ("created".into(), Json::Int(t)),
                    ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                    (
                        "system_fingerprint".into(),
                        Json::String(crate::BUILD_INFO.to_string()),
                    ),
                    ("object".into(), Json::String("text_completion".into())),
                    ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                ];
                if task.params.verbose {
                    fields.push((
                        "__verbose".into(),
                        partial_non_oaicompat(&task, slot, &content.text, &tokens, is_progress, stats_json.as_ref(), prob_output),
                    ));
                }
                if let Some(s) = stats_json {
                    fields.push(("timings".into(), s));
                }
                if is_progress {
                    fields.push(("prompt_progress".into(), progress_json(&task, slot)));
                }
                let _ = task.tx.send(StreamEvent::Frame(Json::Object(fields).dump()));
            }
            ResponseType::OaiChat => {
                // `to_json_oaicompat_chat` (server-task.cpp:1111-1164)
                let t = unix_now();
                let mut deltas: Vec<Json> = Vec::new();
                let first = slot.stats.n_gen == 1;
                // "We have to send an initial update to conform to openai
                // behavior" (server-task.cpp:1128-1133)
                if first || is_progress {
                    deltas.push(Json::Object(vec![
                        ("role".into(), Json::String("assistant".into())),
                        ("content".into(), Json::Null),
                    ]));
                }
                // the diffs of this token's parse — one delta each
                // (`server_chat_msg_diff_to_json_oaicompat`); a content-only
                // parse yields a single content delta (or none)
                for diff in chat_diffs.unwrap_or_default() {
                    deltas.push(chat_msg_diff_to_json(&diff));
                }
                if deltas.is_empty() {
                    // an empty `deltas` array writes no SSE frame at all
                    // (`format_oai_sse` over an empty list)
                    return Ok(());
                }
                let chunks: Vec<Json> = deltas
                    .into_iter()
                    .map(|delta| {
                        Json::Object(vec![
                            (
                                "choices".into(),
                                Json::Array(vec![Json::Object(vec![
                                    ("finish_reason".into(), Json::Null),
                                    ("index".into(), Json::Int(task.index as i64)),
                                    ("delta".into(), delta),
                                ])]),
                            ),
                            ("created".into(), Json::Int(t)),
                            ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                            ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                            (
                                "system_fingerprint".into(),
                                Json::String(crate::BUILD_INFO.to_string()),
                            ),
                            ("object".into(), Json::String("chat.completion.chunk".into())),
                        ])
                    })
                    .collect();
                // the last chunk of the batch carries the extras
                let last = chunks.len() - 1;
                let chunks: Vec<Json> = chunks
                    .into_iter()
                    .enumerate()
                    .map(|(i, mut chunk)| {
                        if i == last {
                            if let Some(p) = prob_output {
                                // the choice gains `logprobs` (nlohmann's
                                // operator[] appends at the end)
                                let choice = chunk.at("choices").and_then(|c| c.at_idx(0)).cloned();
                                if let Some(mut c) = choice {
                                    if let Json::Object(pairs) = &mut c {
                                        pairs.push(("logprobs".into(), logprobs_json(p)));
                                    }
                                    if let Json::Object(pairs) = &mut chunk {
                                        pairs[0].1 = Json::Array(vec![c]);
                                    }
                                }
                            }
                            if let Some(s) = stats_json.clone() {
                                chunk.set("timings", s);
                            }
                            if is_progress {
                                chunk.set("prompt_progress", progress_json(&task, slot));
                            }
                        }
                        chunk
                    })
                    .collect();
                for c in chunks {
                    let _ = task.tx.send(StreamEvent::Frame(c.dump()));
                }
            }
            _ => {
                let mut fields: Vec<(String, Json)> = vec![
                    ("index".into(), Json::Int(task.index as i64)),
                    ("content".into(), Json::String(content.text)),
                    (
                        "tokens".into(),
                        Json::Array(tokens.into_iter().map(|t| Json::Int(t as i64)).collect()),
                    ),
                    ("stop".into(), Json::Bool(false)),
                    // `server_task_result_cmpl_partial::id_slot` is never assigned, so it
                    // keeps its -1 default (server-task.h:310)
                    ("id_slot".into(), Json::Int(-1)),
                    ("tokens_predicted".into(), Json::Uint(slot.stats.n_gen)),
                    ("tokens_evaluated".into(), Json::Int(task.n_tokens() as i64)),
                ];
                if let Some(s) = stats_json {
                    fields.push(("timings".into(), s));
                }
                if is_progress && task.params.return_progress {
                    fields.push(("prompt_progress".into(), progress_json(&task, slot)));
                }
                if let Some(last) = prob_output {
                    fields.push((
                        "completion_probabilities".into(),
                        probs_vector_to_json(std::slice::from_ref(last), task.params.post_sampling_probs),
                    ));
                }
                let _ = task.tx.send(StreamEvent::Frame(Json::Object(fields).dump()));
            }
        }
        Ok(())
    }

    /// `send_final_response` (server-context.cpp:2089-2145) + the result's
    /// `to_json` (server-task.cpp:320-527): the native 16-key object, the OAI
    /// text-completion response or the OAI chat response / chunk list.
    fn send_final(&mut self, si: usize) {
        let Some(task) = self.slots[si].task.clone() else { return };

        // the final `update_chat_msg` (server-task.h:367 —
        // `oaicompat_msg = state.update_chat_msg(content, false, diffs)`):
        // streams add nothing (their text already flowed through the
        // partials), a non-stream request parses its whole text at once.
        // A parse failure is the C's thrown runtime error — the request
        // answers with it instead of a completion.
        let chat_final: Option<(ChatMsg, Vec<ChatMsgDiff>)> = if task.params.res_type
            == ResponseType::OaiChat
        {
            let added = if task.params.stream {
                llama::chat_tools::ChatInput::default()
            } else {
                self.slots[si].generated.clone()
            };
            match self.slots[si]
                .chat
                .as_mut()
                .expect("chat task without parse state")
                .update_chat_msg(&added, false)
            {
                Ok(diffs) => {
                    let msg = self.slots[si].chat.as_ref().unwrap().msg.clone();
                    Some((msg, diffs))
                }
                Err(e) => {
                    let _ = task.tx.send(StreamEvent::Frame(error_frame(
                        crate::api::json_error(&e, crate::api::ERROR_TYPE_SERVER.0, crate::api::ERROR_TYPE_SERVER.1),
                    )));
                    let _ = task.tx.send(StreamEvent::Done);
                    return;
                }
            }
        } else {
            None
        };

        let slot = &self.slots[si];

        // `slot.print_timings()`'s speculative line (server-context.cpp:668-676):
        // "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f"
        if slot.stats.n_draft_tokens > 0 {
            let n_draft_total = slot.stats.n_draft_tokens;
            let n_draft_accepted = slot.stats.n_draft_accepted;
            let n_draft_verif_steps = slot.stats.n_draft_verif_steps;
            let draft_ratio = n_draft_accepted as f32 / n_draft_total as f32;
            let mean_acc_len = if n_draft_verif_steps > 0 {
                1.0 + n_draft_accepted as f64 / n_draft_verif_steps as f64
            } else {
                1.0
            };
            eprintln!(
                "slot print_timing: id {:2} | task {} | draft acceptance = {draft_ratio:.5} \
                 ({n_draft_accepted:5} accepted / {n_draft_total:5} generated), mean len = {mean_acc_len:5.2}",
                slot.id,
                task.id
            );
        }
        let stream = task.params.stream;
        // `json finish_reason = "length"; if (stop == WORD || stop == EOS)
        // finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls"`
        // (server-task.cpp:418-421, :464-467)
        let finish_reason = if slot.stop == StopType::Word || slot.stop == StopType::Eos {
            let has_tool_calls =
                chat_final.as_ref().map(|(m, _)| !m.tool_calls.is_empty()).unwrap_or(false);
            if task.params.res_type == ResponseType::OaiChat && has_tool_calls {
                "tool_calls"
            } else {
                "stop"
            }
        } else {
            "length"
        };
        let usage = usage_json_oaicompat(slot.stats.n_gen, task.n_tokens() as i64, slot.stats.n_prompt_cached);
        // non-stream requests report the whole text; streams already sent it
        // (`res->content = std::move(slot.generated)` — a `common_chat_input`
        // since 18b5f8b18; only `.text` reaches the JSON)
        let content = if stream {
            llama::chat_tools::ChatInput::default()
        } else {
            slot.generated.clone()
        };
        let has_probs = !slot.probs_output.is_empty();

        match task.params.res_type {
            ResponseType::OaiCmpl => {
                // `to_json_oaicompat` (server-task.cpp:374-411)
                let t = unix_now();
                let logprobs = if !stream && has_probs {
                    Json::Object(vec![(
                        "content".into(),
                        probs_vector_to_json(&slot.probs_output, task.params.post_sampling_probs),
                    )])
                } else {
                    Json::Null // OAI default to null
                };
                let mut fields: Vec<(String, Json)> = vec![
                    (
                        "choices".into(),
                        Json::Array(vec![Json::Object(vec![
                            ("text".into(), Json::String(content.text)),
                            ("index".into(), Json::Int(task.index as i64)),
                            ("logprobs".into(), logprobs),
                            ("finish_reason".into(), Json::String(finish_reason.into())),
                        ])]),
                    ),
                    ("created".into(), Json::Int(t)),
                    ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                    (
                        "system_fingerprint".into(),
                        Json::String(crate::BUILD_INFO.to_string()),
                    ),
                    ("object".into(), Json::String("text_completion".into())),
                    ("usage".into(), usage),
                    ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                ];
                if task.params.verbose {
                    fields.push(("__verbose".into(), final_non_oaicompat(self, si)));
                }
                if slot.stats.is_set() {
                    fields.push(("timings".into(), slot.stats.to_json()));
                }
                let _ = task.tx.send(StreamEvent::Frame(Json::Object(fields).dump()));
                let _ = task.tx.send(StreamEvent::Done);
            }
            ResponseType::OaiChat => {
                let t = unix_now();
                if stream {
                    // `to_json_oaicompat_chat_stream` (server-task.cpp:462-525):
                    // one chunk per diff of the final parse, then the finish
                    // chunk (and the usage chunk)
                    let mut deltas: Vec<Json> = Vec::new();
                    for diff in &chat_final.as_ref().expect("chat state").1 {
                        deltas.push(Json::Object(vec![
                            (
                                "choices".into(),
                                Json::Array(vec![Json::Object(vec![
                                    ("finish_reason".into(), Json::Null),
                                    ("index".into(), Json::Int(task.index as i64)),
                                    ("delta".into(), chat_msg_diff_to_json(diff)),
                                ])]),
                            ),
                            ("created".into(), Json::Int(t)),
                            ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                            ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                            (
                                "system_fingerprint".into(),
                                Json::String(crate::BUILD_INFO.to_string()),
                            ),
                            ("object".into(), Json::String("chat.completion.chunk".into())),
                        ]));
                    }
                    deltas.push(Json::Object(vec![
                        (
                            "choices".into(),
                            Json::Array(vec![Json::Object(vec![
                                ("finish_reason".into(), Json::String(finish_reason.into())),
                                ("index".into(), Json::Int(task.index as i64)),
                                ("delta".into(), Json::Object(Vec::new())),
                            ])]),
                        ),
                        ("created".into(), Json::Int(t)),
                        ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                        ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                        (
                            "system_fingerprint".into(),
                            Json::String(crate::BUILD_INFO.to_string()),
                        ),
                        ("object".into(), Json::String("chat.completion.chunk".into())),
                    ]));
                    if task.params.include_usage {
                        deltas.push(Json::Object(vec![
                            ("choices".into(), Json::Array(Vec::new())),
                            ("created".into(), Json::Int(t)),
                            ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                            ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                            (
                                "system_fingerprint".into(),
                                Json::String(crate::BUILD_INFO.to_string()),
                            ),
                            ("object".into(), Json::String("chat.completion.chunk".into())),
                            ("usage".into(), usage),
                        ]));
                    }
                    if slot.stats.is_set() {
                        if let Some(Json::Object(pairs)) = deltas.last_mut() {
                            pairs.push(("timings".into(), slot.stats.to_json()));
                        }
                    }
                    if task.params.verbose {
                        if let Some(Json::Object(pairs)) = deltas.first_mut() {
                            pairs.push(("__verbose".into(), final_non_oaicompat(self, si)));
                        }
                    }
                    for d in deltas {
                        let _ = task.tx.send(StreamEvent::Frame(d.dump()));
                    }
                    let _ = task.tx.send(StreamEvent::Done);
                } else {
                    // `to_json_oaicompat_chat` (server-task.cpp:414-457) —
                    // `msg.to_json_oaicompat()` of the final parse (a pure
                    // content parse yields `{role, content}`, a tool-call
                    // parse adds the `tool_calls` array), with the raw text
                    // as the fallback when nothing parsed
                    // (`common_chat_msg::to_json_oaicompat`, chat.cpp:187-232)
                    let (final_msg, _) = chat_final.as_ref().expect("chat state");
                    let message = if !final_msg.empty() {
                        final_msg.to_json_oaicompat(false).unwrap_or(Json::Null)
                    } else {
                        Json::Object(vec![
                            ("role".into(), Json::String("assistant".into())),
                            ("content".into(), Json::String(content.text)),
                        ])
                    };
                    let mut choice = vec![
                        ("finish_reason".into(), Json::String(finish_reason.into())),
                        ("index".into(), Json::Int(task.index as i64)),
                        ("message".into(), message),
                    ];
                    if has_probs {
                        choice.push((
                            "logprobs".into(),
                            Json::Object(vec![(
                                "content".into(),
                                probs_vector_to_json(&slot.probs_output, task.params.post_sampling_probs),
                            )]),
                        ));
                    }
                    let mut fields: Vec<(String, Json)> = vec![
                        ("choices".into(), Json::Array(vec![Json::Object(choice)])),
                        ("created".into(), Json::Int(t)),
                        ("model".into(), Json::String(task.params.oaicompat_model.clone())),
                        (
                            "system_fingerprint".into(),
                            Json::String(crate::BUILD_INFO.to_string()),
                        ),
                        ("object".into(), Json::String("chat.completion".into())),
                        ("usage".into(), usage),
                        ("id".into(), Json::String(task.params.oaicompat_cmpl_id.clone())),
                    ];
                    if task.params.verbose {
                        fields.push(("__verbose".into(), final_non_oaicompat(self, si)));
                    }
                    if slot.stats.is_set() {
                        fields.push(("timings".into(), slot.stats.to_json()));
                    }
                    let _ = task.tx.send(StreamEvent::Frame(Json::Object(fields).dump()));
                    let _ = task.tx.send(StreamEvent::Done);
                }
            }
            _ => {
                let mut params = task.params.clone();
                params.n_predict = slot.n_predict_max;
                let mut fields: Vec<(String, Json)> = vec![
                    ("index".into(), Json::Int(task.index as i64)),
                    (
                        "content".into(),
                        Json::String(if stream { String::new() } else { slot.generated.text.clone() }),
                    ),
                    (
                        "tokens".into(),
                        Json::Array(if !stream && task.params.return_tokens {
                            slot.generated_tokens.iter().map(|&t| Json::Int(t as i64)).collect()
                        } else {
                            Vec::new()
                        }),
                    ),
                    ("id_slot".into(), Json::Int(slot.id as i64)),
                    ("stop".into(), Json::Bool(true)),
                    ("model".into(), Json::String(self.model_name.clone())),
                    ("tokens_predicted".into(), Json::Uint(slot.stats.n_gen)),
                    ("tokens_evaluated".into(), Json::Int(task.n_tokens() as i64)),
                    ("generation_settings".into(), task_params_to_json(&params, false)),
                    // `res->prompt = slot.task->tokens.detokenize(ctx_tgt, true)`
                    // (server-context.cpp:2111) — the *processed* token ids
                    // re-rendered with special tokens, so a BOS-carrying vocab
                    // echoes "<s> …" here even though the request body didn't
                    ("prompt".into(), Json::String(self.vocab.detokenize(&task.tokens, true))),
                    ("has_new_line".into(), Json::Bool(slot.has_new_line)),
                    ("truncated".into(), Json::Bool(slot.truncated)),
                    ("stop_type".into(), Json::String(slot.stop.as_str().into())),
                    ("stopping_word".into(), Json::String(slot.stopping_word.clone())),
                    ("tokens_cached".into(), Json::Int(slot.prompt_tokens.len() as i64)),
                    ("timings".into(), slot.stats.to_json()),
                ];
                if !stream && !slot.probs_output.is_empty() {
                    fields.push((
                        "completion_probabilities".into(),
                        probs_vector_to_json(&slot.probs_output, task.params.post_sampling_probs),
                    ));
                }
                let _ = task.tx.send(StreamEvent::Frame(Json::Object(fields).dump()));
                let _ = task.tx.send(StreamEvent::Done);
                if task.params.timings_per_token {
                    // nothing else to report (per-token timings travel in the partials)
                }
            }
        }
    }
}

/// `std::time(0)` — the unix seconds the OAI responses stamp `created` with.
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `result_prompt_progress::to_json` (server-task.cpp:242-247)
fn progress_json(task: &Task, slot: &Slot) -> Json {
    Json::Object(vec![
        ("total".into(), Json::Int(task.n_tokens() as i64)),
        ("cache".into(), Json::Uint(slot.stats.n_prompt_cached)),
        ("processed".into(), Json::Int(slot.prompt_tokens.len() as i64)),
        ("time_ms".into(), Json::Uint(slot.stats.t_elapsed_us() as u64 / 1000)),
    ])
}

/// `server_task_result_cmpl_partial::to_json_non_oaicompat`
/// (server-task.cpp:1049-1071) — the `__verbose` object of OAI partials.
#[allow(clippy::too_many_arguments)]
fn partial_non_oaicompat(
    task: &Task,
    slot: &Slot,
    content: &str,
    tokens: &[i32],
    is_progress: bool,
    stats: Option<&Json>,
    prob_output: Option<&TokenProbs>,
) -> Json {
    let mut fields: Vec<(String, Json)> = vec![
        ("index".into(), Json::Int(task.index as i64)),
        ("content".into(), Json::String(content.to_string())),
        (
            "tokens".into(),
            Json::Array(tokens.iter().map(|&t| Json::Int(t as i64)).collect()),
        ),
        ("stop".into(), Json::Bool(false)),
        ("id_slot".into(), Json::Int(-1)),
        ("tokens_predicted".into(), Json::Uint(slot.stats.n_gen)),
        ("tokens_evaluated".into(), Json::Int(task.n_tokens() as i64)),
    ];
    if let Some(s) = stats {
        fields.push(("timings".into(), s.clone()));
    }
    if is_progress {
        fields.push(("prompt_progress".into(), progress_json(task, slot)));
    }
    if let Some(p) = prob_output {
        fields.push((
            "completion_probabilities".into(),
            probs_vector_to_json(std::slice::from_ref(p), task.params.post_sampling_probs),
        ));
    }
    Json::Object(fields)
}

/// `server_task_result_cmpl_final::to_json_non_oaicompat`
/// (server-task.cpp:340-362) — the `__verbose` object of OAI finals.
fn final_non_oaicompat(engine: &Engine, si: usize) -> Json {
    let slot = &engine.slots[si];
    let Some(task) = slot.task.as_ref() else { return Json::Null };
    let stream = task.params.stream;
    let mut params = task.params.clone();
    params.n_predict = slot.n_predict_max;
    let mut fields: Vec<(String, Json)> = vec![
        ("index".into(), Json::Int(task.index as i64)),
        (
            "content".into(),
            Json::String(if stream { String::new() } else { slot.generated.text.clone() }),
        ),
        (
            "tokens".into(),
            Json::Array(if !stream && task.params.return_tokens {
                slot.generated_tokens.iter().map(|&t| Json::Int(t as i64)).collect()
            } else {
                Vec::new()
            }),
        ),
        ("id_slot".into(), Json::Int(slot.id as i64)),
        ("stop".into(), Json::Bool(true)),
        ("model".into(), Json::String(task.params.oaicompat_model.clone())),
        ("tokens_predicted".into(), Json::Uint(slot.stats.n_gen)),
        ("tokens_evaluated".into(), Json::Int(task.n_tokens() as i64)),
        ("generation_settings".into(), task_params_to_json(&params, false)),
        // the task result's `prompt` member (server-context.cpp:2111) — the
        // processed tokens re-detokenized with specials
        ("prompt".into(), Json::String(engine.vocab.detokenize(&task.tokens, true))),
        ("has_new_line".into(), Json::Bool(slot.has_new_line)),
        ("truncated".into(), Json::Bool(slot.truncated)),
        ("stop_type".into(), Json::String(slot.stop.as_str().into())),
        ("stopping_word".into(), Json::String(slot.stopping_word.clone())),
        ("tokens_cached".into(), Json::Int(slot.prompt_tokens.len() as i64)),
        ("timings".into(), slot.stats.to_json()),
    ];
    if !stream && !slot.probs_output.is_empty() {
        fields.push((
            "completion_probabilities".into(),
            probs_vector_to_json(&slot.probs_output, task.params.post_sampling_probs),
        ));
    }
    Json::Object(fields)
}

/// `llama_tokens::get_common_prefix`
fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// an SSE frame that carries the error object — the stream generator's
/// `format_error` (server-context.cpp:4416-4424)
pub fn error_frame(error_json: String) -> String {
    let v = Json::parse(&error_json).unwrap_or(Json::Null);
    let inner = v.at("error").cloned().unwrap_or(Json::Null);
    Json::Object(vec![("error".into(), inner)]).dump()
}

// ---------------------------------------------------------------------------
// tests — the lazy PATTERN-trigger engine on a synthetic piece table
// ---------------------------------------------------------------------------

#[cfg(test)]
mod lazy_trigger_tests {
    use super::*;

    /// token id = index; `is_eog` false for all
    fn pieces(list: &[&str]) -> llama::grammar::VocabPieces {
        llama::grammar::VocabPieces::from_iter(
            list.iter().map(|p| (p.as_bytes().to_vec(), false)),
        )
    }

    /// a `LazyTriggers` holding only compiled PATTERN triggers (the
    /// specialized parsers' form)
    fn pattern_triggers(patterns: &[&str]) -> LazyTriggers {
        LazyTriggers {
            tokens: Vec::new(),
            patterns: patterns
                .iter()
                .map(|p| {
                    LazyPattern::Regex(llama::regex_lite::RegexLite::new(p).expect("pattern compiles"))
                })
                .collect(),
            buffer: Vec::new(),
            positions: Vec::new(),
        }
    }

    fn grammar_accept_all(
        g: &mut GrammarSampler,
        t: &mut LazyTriggers,
        stream: &[i32],
    ) -> Vec<bool> {
        let mut awaiting = Vec::new();
        for &tok in stream {
            grammar_accept_lazy(g, t, tok).expect("accept");
            awaiting.push(g.grammar.awaiting_trigger);
        }
        awaiting
    }

    /// gpt-oss `<|start|>assistant(\s+to)` (parsers/gpt-oss.cpp:152): the fire
    /// position is the capture start (byte 21 of "foo<|start|>assistant to"),
    /// so exactly the " to" token replays into the grammar — both when the
    /// capture aligns with a token boundary and when one token straddles it
    /// (llama-grammar.cpp:1420-1431)
    #[test]
    fn gptoss_pattern_fires_at_capture_start() {
        // 0:"foo" 1:"<|start|>" 2:"assistant" 3:" to" 4:"!"
        let vocab = pieces(&["foo", "<|start|>", "assistant", " to", "!"]);
        let mut g =
            GrammarSampler::from_pieces(r#"root ::= " to" "!""#, "root", vocab).expect("grammar");
        g.grammar.lazy = true;
        g.grammar.awaiting_trigger = true;

        // clean token boundaries
        let mut t = pattern_triggers(&[r"<\|start\|>assistant(\s+to)"]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1, 2, 3]);
        assert_eq!(awaiting, vec![true, true, true, false]);
        // the replay fed exactly " to": the trailing "!" completes the grammar
        grammar_accept_lazy(&mut g, &mut t, 4).expect("!");
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()), "grammar complete");
        assert!(t.buffer.is_empty() && t.positions.is_empty(), "buffer cleared");

        // one token straddles the capture start: the replay is its tail slice
        // (0:"foo" 5:"<|start|>assistant to" 4:"!")
        let vocab = pieces(&["foo", "<|start|>", "assistant", " to", "!", "<|start|>assistant to"]);
        let mut g =
            GrammarSampler::from_pieces(r#"root ::= " to" "!""#, "root", vocab).expect("grammar");
        g.grammar.lazy = true;
        g.grammar.awaiting_trigger = true;
        let mut t = pattern_triggers(&[r"<\|start\|>assistant(\s+to)"]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 5]);
        assert_eq!(awaiting, vec![true, false]);
        grammar_accept_lazy(&mut g, &mut t, 4).expect("!");
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()), "straddling replay");
    }

    /// the anchored gpt-oss trigger `^\s+to$` (parsers/gpt-oss.cpp:150) fires
    /// only while the whole buffer is whitespace + "to" (llama-grammar.cpp
    /// :394-400 full-match path); a buffer that already grew past it (e.g.
    /// " to=functions…") never fires through this trigger
    #[test]
    fn gptoss_anchored_pattern_full_match_only() {
        // 0:" " 1:"to" 2:"=functions"
        // the replay feeds the whole buffer from byte 0 (whitespace included)
        let vocab = pieces(&[" ", "to", "=functions"]);
        let mut g =
            GrammarSampler::from_pieces(r#"root ::= " to" [a-z]*"#, "root", vocab).expect("grammar");
        g.grammar.lazy = true;
        g.grammar.awaiting_trigger = true;
        let mut t = pattern_triggers(&[r"^\s+to$"]);
        // " " → " to": fires at the second token, replay feeds " to" from
        // byte 0 (both tokens)
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1]);
        assert_eq!(awaiting, vec![true, false]);

        // "  to" as a single token also fires; the replay includes both spaces
        let vocab2 = pieces(&["  to", "to"]);
        let mut g2 =
            GrammarSampler::from_pieces(r#"root ::= "  to" [a-z]*"#, "root", vocab2).expect("grammar");
        g2.grammar.lazy = true;
        g2.grammar.awaiting_trigger = true;
        let mut t2 = pattern_triggers(&[r"^\s+to$"]);
        let awaiting = grammar_accept_all(&mut g2, &mut t2, &[0, 1]);
        assert_eq!(awaiting, vec![false, false]);

        // the buffer grew past the trigger in one token (" to=functions"):
        // it was never exactly \s+to, so it never fires (grammar unreached)
        let vocab3 = pieces(&[" to=functions"]);
        let mut g3 =
            GrammarSampler::from_pieces(r#"root ::= "x""#, "root", vocab3).expect("grammar");
        g3.grammar.lazy = true;
        g3.grammar.awaiting_trigger = true;
        let mut t3 = pattern_triggers(&[r"^\s+to$"]);
        let awaiting = grammar_accept_all(&mut g3, &mut t3, &[0]);
        assert_eq!(awaiting, vec![true]);
    }

    /// functionary v3.2 `>>>(?!all)` (parsers/functionary-v3-2.cpp:91):
    /// ">>>x" fires (grammar constrains from the ">>>"), ">>>all" never does
    /// — even when ">>>" and "all" arrive in one token, so the buffer never
    /// passed through bare ">>>"
    #[test]
    fn functionary_lookahead_blocks_all() {
        // 0:">>" 1:">" 2:"x" 3:">>>all" 4:">>>"
        let vocab = pieces(&[">>", ">", "x", ">>>all", ">>>"]);
        let mk = |vocab: llama::grammar::VocabPieces| {
            let mut g =
                GrammarSampler::from_pieces(r#"root ::= [a-z_>]+"#, "root", vocab).expect("grammar");
            g.grammar.lazy = true;
            g.grammar.awaiting_trigger = true;
            g
        };

        // ">>" → ">>>" → "x": the trigger fires once the bare ">>>" is
        // buffered (nothing follows, so the lookahead passes)
        let mut g = mk(vocab.clone());
        let mut t = pattern_triggers(&[r">>>(?!all)"]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1, 2]);
        assert_eq!(awaiting, vec![true, false, false]);

        // ">>>all" in one token: the buffer is ">>>all" and never fires
        let mut g = mk(vocab.clone());
        let mut t = pattern_triggers(&[r">>>(?!all)"]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[3]);
        assert_eq!(awaiting, vec![true]);

        // ">>>" then "all": fires at ">>>" (as in the reference flow, where
        // the trigger evaluates after every token)
        let mut g = mk(vocab.clone());
        let mut t = pattern_triggers(&[r">>>(?!all)"]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[4, 3]);
        assert_eq!(awaiting, vec![false, false]);
    }

    /// muse-glimmer's trigger (parsers/muse-glimmer.cpp:131-134): the buffer
    /// grows token by token and the grammar attaches only when the closing
    /// `<|message|>` lands; the `self`/`user` recipients are blocked by the
    /// negative lookaheads
    #[test]
    fn muse_pattern_incremental_fire() {
        // 0:" to=functions" 1:"\n{\"city\":\"Tokyo\"}" 2:"<|message|>"
        // 3:" to=self" 4:"<|message|>"
        let vocab = pieces(&[
            " to=functions",
            "\n{\"city\":\"Tokyo\"}",
            "<|message|>",
            " to=self",
        ]);
        let p = r"(?:^|<\|start\|>assistant)( to=(?!self<\|message\|>)(?!user<\|message\|>)[^<]*?<\|message\|>)";
        let mk = || {
            let mut g = GrammarSampler::from_pieces(
                r#"root ::= " to=" [^<]+ "<|message|>""#,
                "root",
                pieces(&[" to=functions", "\n{\"city\":\"Tokyo\"}", "<|message|>", " to=self"]),
            )
            .expect("grammar");
            g.grammar.lazy = true;
            g.grammar.awaiting_trigger = true;
            g
        };

        let mut g = mk();
        let mut t = pattern_triggers(&[p]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1, 2]);
        assert_eq!(awaiting, vec![true, true, false], "fires when <|message|> lands");
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()), "replayed the whole buffer from 0");

        // `self` recipient: the lookaheads keep the grammar awaiting forever
        let mut g = mk();
        let mut t = pattern_triggers(&[p]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[3, 2]);
        assert_eq!(awaiting, vec![true, true]);
    }

    /// the C fires on the first pattern *in declaration order* that matches
    /// (llama-grammar.cpp:1415-1439) — not the earliest position across
    /// patterns. Both patterns complete on the same final token, but their
    /// captures sit 30 bytes apart.
    #[test]
    fn pattern_priority_is_declaration_order() {
        // 0:" to=functions<|start|>assistant" 1:" to" 2:"!"
        let vocab = pieces(&[" to=functions<|start|>assistant", " to", "!"]);
        // final buffer = " to=functions<|start|>assistant to" (34 bytes):
        //   A `…assistant(\s+to)` captures at 31 (just the trailing " to")
        //   B `\s(to)=functions…` captures at 1  (from "to=functions" on)
        const A: &str = r"<\|start\|>assistant(\s+to)";
        const B: &str = r"\s(to)=functions<\|start\|>assistant\s+to";

        // A first: fires at 31 — only the " to" token replays
        let mut g = GrammarSampler::from_pieces(r#"root ::= " to" "!""#, "root", vocab.clone())
            .expect("grammar");
        g.grammar.lazy = true;
        g.grammar.awaiting_trigger = true;
        let mut t = pattern_triggers(&[A, B]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1]);
        assert_eq!(awaiting, vec![true, false]);
        grammar_accept_lazy(&mut g, &mut t, 2).expect("!");
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()), "A's short replay");

        // B first: fires at 1 — the replay is token 0's tail plus token 1
        let mut g = GrammarSampler::from_pieces(
            r#"root ::= "to=functions<|start|>assistant to" "!""#,
            "root",
            vocab,
        )
        .expect("grammar");
        g.grammar.lazy = true;
        g.grammar.awaiting_trigger = true;
        let mut t = pattern_triggers(&[B, A]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 1]);
        assert_eq!(awaiting, vec![true, false]);
        grammar_accept_lazy(&mut g, &mut t, 2).expect("!");
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()), "B's long replay");
    }

    /// the PATTERN_FULL anchoring of `common_sampler_init`
    /// (sampling.cpp:235-245) — verbatim `^`/`$` splice on the raw string
    #[test]
    fn pattern_full_anchor_splice() {
        assert_eq!(anchor_pattern_full(""), "^$");
        assert_eq!(anchor_pattern_full("hello"), "^hello$");
        assert_eq!(anchor_pattern_full("^hello"), "^hello$");
        assert_eq!(anchor_pattern_full("hello$"), "^hello$");
        assert_eq!(anchor_pattern_full("^hello$"), "^hello$");
        // raw-byte checks: an escaped '$' already ends the pattern → no splice
        assert_eq!(anchor_pattern_full("a\\$"), "^a\\$");
        assert_eq!(anchor_pattern_full("\\^a"), "^\\^a$");
    }

    /// a PATTERN_FULL trigger end-to-end: `hello` anchors to `^hello$`, so
    /// the buffer must be exactly "hello" when the trigger fires
    #[test]
    fn pattern_full_end_to_end() {
        // 0:"hello" 1:"hell" 2:"o" 3:"!"
        let vocab = pieces(&["hello", "hell", "o", "!"]);
        let mk = || {
            let mut g =
                GrammarSampler::from_pieces(r#"root ::= "hello" "!""#, "root", vocab.clone())
                    .expect("grammar");
            g.grammar.lazy = true;
            g.grammar.awaiting_trigger = true;
            g
        };

        // "hello" as one token: fires immediately, replay from byte 0
        let mut g = mk();
        let mut t = pattern_triggers(&[&anchor_pattern_full("hello")]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[0, 3]);
        assert_eq!(awaiting, vec![false, false]);
        assert!(g.grammar.stacks.iter().all(|s| s.is_empty()));

        // "hell" then "o": fires only on the second token
        let mut g = mk();
        let mut t = pattern_triggers(&[&anchor_pattern_full("hello")]);
        let awaiting = grammar_accept_all(&mut g, &mut t, &[1, 2]);
        assert_eq!(awaiting, vec![true, false]);

        // "hello" embedded in a longer buffer: never fires (grammar unused)
        let vocab2 = pieces(&["x", "hello", "y"]);
        let mut g2 = GrammarSampler::from_pieces(r#"root ::= "x""#, "root", vocab2).expect("grammar");
        g2.grammar.lazy = true;
        g2.grammar.awaiting_trigger = true;
        let mut t2 = pattern_triggers(&[&anchor_pattern_full("hello")]);
        let awaiting = grammar_accept_all(&mut g2, &mut t2, &[0, 1, 2]);
        assert_eq!(awaiting, vec![true, true, true]);
    }

    /// bit-for-bit replay of `parity/lazy_trigger_ref.txt` —
    ///  * every FIND line carries the pattern and buffer as hex plus the fire
    ///    position the *reference* `llama_grammar_trigger_pattern::find`
    ///    returned (the probe drives libllama directly); the port must agree
    ///    on every line
    ///  * every CASE block is the reference's lazy accept flow
    ///    (`llama_grammar_accept_impl`) over real qwen2 tokenization; the
    ///    port replays the same tokens through `grammar_accept_lazy` and must
    ///    match the awaiting flag, the buffer length and the grammar stack
    ///    count after every step
    #[test]
    fn lazy_trigger_reference_replay() {
        const REF: &str =
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../parity/lazy_trigger_ref.txt");
        let Ok(text) = std::fs::read_to_string(REF) else {
            eprintln!("SKIP: {REF} missing (run parity/gen_lazy_trigger_ref.sh)");
            return;
        };
        let unhex = |s: &str| -> Vec<u8> {
            (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
        };

        let mut n_find = 0usize;
        let mut n_cases = 0usize;
        // the CASE block being accumulated
        let mut grammar_hex: Option<Vec<u8>> = None;
        let mut patterns: Vec<String> = Vec::new();
        let mut toks: Vec<Vec<u8>> = Vec::new();
        // (step, awaiting, buffer_len, n_stacks, accept_failed)
        let mut steps: Vec<(usize, bool, usize, usize, bool)> = Vec::new();

        let mut flush_case = |grammar_hex: &mut Option<Vec<u8>>,
                              patterns: &mut Vec<String>,
                              toks: &mut Vec<Vec<u8>>,
                              steps: &mut Vec<(usize, bool, usize, usize, bool)>,
                              n_cases: &mut usize| {
            let Some(gh) = grammar_hex.take() else { return };
            let grammar = String::from_utf8(gh).expect("grammar utf8");
            let vocab = llama::grammar::VocabPieces::from_iter(
                toks.iter().map(|p| (p.clone(), false)),
            );
            let mut g = GrammarSampler::from_pieces(&grammar, "root", vocab)
                .unwrap_or_else(|e| panic!("grammar {grammar:?}: {e}"));
            g.grammar.lazy = true;
            g.grammar.awaiting_trigger = true;
            let mut t = LazyTriggers {
                tokens: Vec::new(),
                patterns: patterns
                    .iter()
                    .map(|p| {
                        LazyPattern::Regex(
                            llama::regex_lite::RegexLite::new(p)
                                .unwrap_or_else(|e| panic!("pattern {p:?}: {e}")),
                        )
                    })
                    .collect(),
                buffer: Vec::new(),
                positions: Vec::new(),
            };
            for (i, want) in steps.iter().enumerate() {
                let (i_want, awaiting_want, buflen_want, stacks_want, failed_want) = *want;
                assert_eq!(i, i_want, "step order in CASE block");
                let got = grammar_accept_lazy(&mut g, &mut t, i as i32);
                if failed_want {
                    // the reference's accept threw at this step — the port's
                    // accept must fail too
                    assert!(
                        got.is_err(),
                        "step {i} of grammar {grammar:?}: reference failed, port did not"
                    );
                    break;
                }
                got.unwrap_or_else(|e| panic!("step {i} of grammar {grammar:?}: {e}"));
                assert_eq!(
                    g.grammar.awaiting_trigger, awaiting_want,
                    "awaiting after step {i} of grammar {grammar:?}"
                );
                assert_eq!(
                    t.buffer.len(),
                    buflen_want,
                    "buffer length after step {i} of grammar {grammar:?}"
                );
                assert_eq!(
                    g.grammar.stacks.len(),
                    stacks_want,
                    "stack count after step {i} of grammar {grammar:?}"
                );
            }
            patterns.clear();
            toks.clear();
            steps.clear();
            *n_cases += 1;
        };

        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("FIND ") {
                // single-space split: an empty buffer renders as a double
                // space (`FIND <phex>  <pos>`) and must not collapse
                let mut it = rest.split(' ');
                let phex = it.next().expect("pattern hex");
                let bhex = it.next().expect("buffer hex");
                let pos = it.next().expect("position");
                let pattern = String::from_utf8(unhex(phex)).expect("pattern utf8");
                let buffer = unhex(bhex);
                let re = llama::regex_lite::RegexLite::new(&pattern)
                    .unwrap_or_else(|e| panic!("line {line:?}: {e}"));
                let got = re.find(&buffer);
                let want = if pos == "npos" { None } else { Some(pos.parse::<usize>().unwrap()) };
                assert_eq!(got, want, "pattern {pattern:?} buffer {buffer:?}");
                n_find += 1;
            } else if let Some(rest) = line.strip_prefix("CASE ") {
                flush_case(&mut grammar_hex, &mut patterns, &mut toks, &mut steps, &mut n_cases);
                let _ = rest;
            } else if let Some(rest) = line.strip_prefix("GRAMMAR ") {
                grammar_hex = Some(unhex(rest));
            } else if let Some(rest) = line.strip_prefix("PATTERN ") {
                patterns.push(String::from_utf8(unhex(rest)).expect("pattern utf8"));
            } else if let Some(rest) = line.strip_prefix("TOK ") {
                let mut it = rest.split_whitespace();
                let _dense = it.next().unwrap();
                let _orig = it.next().unwrap();
                toks.push(unhex(it.next().expect("piece hex")));
            } else if let Some(rest) = line.strip_prefix("STEP ") {
                let mut it = rest.split_whitespace();
                let i: usize = it.next().unwrap().parse().unwrap();
                let awaiting = it.next().unwrap() == "1";
                let buflen: usize = it.next().unwrap().parse().unwrap();
                let stacks: usize = it.next().unwrap().parse().unwrap();
                let failed = it.next().map(|m| m == "ACCEPT_FAILED").unwrap_or(false);
                steps.push((i, awaiting, buflen, stacks, failed));
            }
        }
        flush_case(&mut grammar_hex, &mut patterns, &mut toks, &mut steps, &mut n_cases);
        assert!(n_find > 100, "expected a full case matrix, got {n_find} FIND lines");
        assert!(n_cases >= 5, "expected the ACCEPT cases, got {n_cases}");
    }

    // ---- the /slots save/restore byte formats (this file's helpers) ----

    /// `server_tokens::serialize`/`deserialize` (server-common.cpp:570-633):
    /// the round trip is exact, a plain token list (the old format) passes
    /// through, and the C's error texts come out of the malformed payloads.
    #[test]
    fn server_tokens_state_round_trip() {
        let toks = vec![5i32, 785, 6722, 0, 151665];
        let packed = server_tokens_serialize(&toks);
        // the format: [i32 -1][u32 1][u32 n][tokens][u32 0], 4-byte padded
        assert_eq!(packed.len() % 4, 0);
        assert_eq!(server_tokens_deserialize(&toks_of(&packed)).unwrap(), toks);
        // the old format (no marker word) is a plain token list
        assert_eq!(server_tokens_deserialize(&[1, 2, 3]).unwrap(), vec![1, 2, 3]);
        assert_eq!(server_tokens_deserialize(&[]).unwrap(), Vec::<i32>::new());
        // the error paths
        let mut bad_version = packed.clone();
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            server_tokens_deserialize(&toks_of(&bad_version)).unwrap_err(),
            "Unsupported server tokens state version"
        );
        let mut trailing = packed.clone();
        trailing.extend_from_slice(&7i32.to_le_bytes());
        assert_eq!(
            server_tokens_deserialize(&toks_of(&trailing)).unwrap_err(),
            "Trailing data in server tokens state"
        );
        // a truncated payload (the vector header promised more than remains)
        let mut short = packed.clone();
        short.truncate(short.len() - 4);
        assert!(server_tokens_deserialize(&toks_of(&short)).is_err());
    }

    /// bytes → i32 words (little-endian, `sizeof(llama_token)` = 4)
    fn toks_of(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// `fs_validate_filename(filename, /*allow_subdirs=*/false)`
    /// (common/common.cpp:825-897): the accepted shapes and every forbidden
    /// class the C rejects.
    #[test]
    fn filename_validation() {
        for good in ["a", "slot.bin", "slot-2026.bin", "sub dir.bin", "x".repeat(255).as_str()] {
            assert!(fs_validate_filename(good), "{good:?} must be valid");
        }
        for bad in [
            "",                       // empty
            "a/b",                    // path separator
            "a\\b",                   // windows separator
            "..",                     // traversal
            "a..b",                   // traversal substring
            ".",                      // dot
            " ",                      // leading space
            "a ",                     // trailing space
            "a.",                     // trailing dot
            "a:b",                    // illegal character
            "a*b", "a?b", "a\"b", "a<b", "a>b", "a|b",
            "a\u{1}b",                // C0 control
            "a\u{7F}b",               // DEL
            "a\u{85}b",               // C1 control
            "a\u{FF0E}b",             // fullwidth full stop
            "a\u{2215}b",             // division slash
            "a\u{2216}b",             // set minus
            "a\u{FFFD}b",             // replacement char
            "a\u{FEFF}b",             // BOM
            &"x".repeat(256),         // over the 255-byte limit
        ] {
            assert!(!fs_validate_filename(bad), "{bad:?} must be rejected");
        }
    }
}
/// the checkpoint-appendix format tests — the port's mirror of the reference's
/// `test_slot_save.py` appendix cases (tools/server/tests/unit/test_slot_save.py:596-735,
/// 033df86b6): the byte layout, the damaged-appendix tolerance (ignored, never
/// fatal) and the `n_ctx_checkpoints` cap. The checkpoint *reuse* assertions of
/// the upstream tests (a restored slot rolling back to a checkpoint instead of
/// re-processing the prompt) need the prompt-cache checkpoint machinery, which
/// the port does not have yet.
#[cfg(test)]
mod slot_ckpt_tests {
    use super::*;

    fn ckpt(tgt: &[u8], dft: &[u8], spec: &[u8]) -> SlotCheckpoint {
        SlotCheckpoint {
            id_task: 7,
            n_tokens: 11,
            pos_min: 3,
            pos_max: 9,
            data_tgt: tgt.to_vec(),
            data_dft: dft.to_vec(),
            data_spec: spec.to_vec(),
        }
    }

    /// `parse_ckpt_appendix`'s spec: magic(4) version(4) count(4), then per
    /// checkpoint n_tokens(8) pos_min(4) pos_max(4) and three blobs (target,
    /// draft, speculative), each size(8) + data
    #[test]
    fn appendix_byte_layout() {
        let list: VecDeque<SlotCheckpoint> = [
            ckpt(b"TGT", b"", b"SP"),
            ckpt(&[0xAA; 5], b"D", b""),
        ]
        .into_iter()
        .collect();
        let mut out = Vec::new();
        let n = slot_checkpoints_appendix(&list, &mut out);
        assert_eq!(n, out.len());
        let expect: Vec<u8> = [
            SLOT_CKPT_MAGIC.to_le_bytes().as_slice(),
            SLOT_CKPT_VERSION.to_le_bytes().as_slice(),
            2u32.to_le_bytes().as_slice(),
            // checkpoint 0
            11u64.to_le_bytes().as_slice(),
            3i32.to_le_bytes().as_slice(),
            9i32.to_le_bytes().as_slice(),
            3u64.to_le_bytes().as_slice(),
            b"TGT".as_slice(),
            0u64.to_le_bytes().as_slice(),
            2u64.to_le_bytes().as_slice(),
            b"SP".as_slice(),
            // checkpoint 1
            11u64.to_le_bytes().as_slice(),
            3i32.to_le_bytes().as_slice(),
            9i32.to_le_bytes().as_slice(),
            5u64.to_le_bytes().as_slice(),
            &[0xAA; 5].as_slice(),
            1u64.to_le_bytes().as_slice(),
            b"D".as_slice(),
            0u64.to_le_bytes().as_slice(),
        ]
        .concat();
        assert_eq!(out, expect);
    }

    #[test]
    fn empty_list_writes_nothing() {
        let mut out = Vec::new();
        assert_eq!(slot_checkpoints_appendix(&Default::default(), &mut out), 0);
        assert!(out.is_empty());
    }

    #[test]
    fn roundtrip_and_task_marker() {
        let list: VecDeque<SlotCheckpoint> =
            [ckpt(b"tgt-state", b"dft-state", b"spec-state"), ckpt(b"x", b"", b"")]
                .into_iter()
                .collect();
        let mut file = b"PAYLOAD".to_vec();
        let off = file.len();
        let n = slot_checkpoints_appendix(&list, &mut file);
        let (got, consumed) = parse_slot_checkpoints_appendix(&file, off, 0);
        assert_eq!(consumed, n);
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(list.iter()) {
            // id_task = -1 marks a checkpoint restored from a slot file
            assert_eq!(g.id_task, -1);
            assert_eq!(g.n_tokens, w.n_tokens);
            assert_eq!(g.pos_min, w.pos_min);
            assert_eq!(g.pos_max, w.pos_max);
            assert_eq!(g.data_tgt, w.data_tgt);
            assert_eq!(g.data_dft, w.data_dft);
            assert_eq!(g.data_spec, w.data_spec);
        }
    }

    #[test]
    fn no_appendix_consumes_nothing() {
        let file = b"just the llama state payload".to_vec();
        let (got, n) = parse_slot_checkpoints_appendix(&file, file.len(), 0);
        assert_eq!(n, 0);
        assert!(got.is_empty());
        // a payload that merely contains the magic bytes mid-stream is not an
        // appendix — only the bytes at the exact offset are looked at
        let mut file2 = b"state".to_vec();
        file2.extend_from_slice(&SLOT_CKPT_MAGIC.to_le_bytes());
        let (got2, n2) = parse_slot_checkpoints_appendix(&file2, 0, 0);
        assert_eq!(n2, 0);
        assert!(got2.is_empty());
    }

    /// the upstream "oversized_blob" damage: a size field that cannot be
    /// allocated must be rejected before allocating (n_read == off)
    #[test]
    fn oversized_blob_is_rejected() {
        let mut file = Vec::new();
        file.extend_from_slice(&SLOT_CKPT_MAGIC.to_le_bytes());
        file.extend_from_slice(&SLOT_CKPT_VERSION.to_le_bytes());
        file.extend_from_slice(&1u32.to_le_bytes());
        file.extend_from_slice(&11u64.to_le_bytes());
        file.extend_from_slice(&3i32.to_le_bytes());
        file.extend_from_slice(&9i32.to_le_bytes());
        file.extend_from_slice(&(1u64 << 62).to_le_bytes());
        let (got, n) = parse_slot_checkpoints_appendix(&file, 0, 0);
        assert_eq!(n, 0);
        assert!(got.is_empty());
    }

    /// the upstream "empty_target" damage: a valid save never writes an empty
    /// target state
    #[test]
    fn empty_target_state_is_rejected() {
        let list: VecDeque<SlotCheckpoint> = [ckpt(b"", b"", b"")].into_iter().collect();
        let mut file = Vec::new();
        slot_checkpoints_appendix(&list, &mut file);
        let (got, n) = parse_slot_checkpoints_appendix(&file, 0, 0);
        assert_eq!(n, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn wrong_version_and_truncation_are_ignored() {
        let mut file = Vec::new();
        file.extend_from_slice(&SLOT_CKPT_MAGIC.to_le_bytes());
        file.extend_from_slice(&9u32.to_le_bytes()); // version
        file.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse_slot_checkpoints_appendix(&file, 0, 0).1, 0);

        let list: VecDeque<SlotCheckpoint> = [ckpt(b"tgt", b"", b"")].into_iter().collect();
        let mut full = Vec::new();
        slot_checkpoints_appendix(&list, &mut full);
        let (got, n) = parse_slot_checkpoints_appendix(&full[..full.len() - 1], 0, 0);
        assert_eq!(n, 0);
        assert!(got.is_empty());
    }

    /// the upstream "many_checkpoints" damage: a flood of entries keeps only
    /// the last `n_ctx_checkpoints`, the appendix is still fully consumed
    #[test]
    fn many_checkpoints_keep_the_last_cap() {
        let mut list: VecDeque<SlotCheckpoint> = Default::default();
        // one-byte fillers that never match go first, the real checkpoints stay last
        for i in 0..(N_CTX_CHECKPOINTS + 8) {
            list.push_back(SlotCheckpoint {
                id_task: 0,
                n_tokens: i,
                pos_min: 0,
                pos_max: 1 << 30,
                data_tgt: vec![0],
                data_dft: Vec::new(),
                data_spec: Vec::new(),
            });
        }
        let mut file = Vec::new();
        let n = slot_checkpoints_appendix(&list, &mut file);
        let (got, consumed) = parse_slot_checkpoints_appendix(&file, 0, 0);
        assert_eq!(consumed, n);
        assert_eq!(got.len(), N_CTX_CHECKPOINTS);
        // the survivors are the *last* N_CTX_CHECKPOINTS
        assert_eq!(got.front().unwrap().n_tokens, 8);
        assert_eq!(got.back().unwrap().n_tokens, N_CTX_CHECKPOINTS + 7);
    }
}
