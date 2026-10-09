//! Request schema + response JSON — port of tools/server/server-schema.cpp
//! (`eval_llama_cmpl_schema`), server-task.cpp (`task_params::to_json`,
//! `to_json_non_oaicompat`) and server-common.cpp (`format_error_response`,
//! `server_slot_stats::to_json`, `get_token_probabilities`).
//!
//! JSON is `llama::json_schema::Json` — the port's nlohmann-compatible value
//! type (ordered objects, `dump_float` = nlohmann's shortest round-trip form),
//! so the responses serialise like the reference's.

use llama::json_schema::Json;
use llama::sampling::{LogitBias, SamplingParams, TokenDataArray};
use llama::vocab::Vocab;

/// `ggml_time_us` — a monotonic microsecond clock (std::time::Instant)
pub fn now_us() -> i64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_micros() as i64
}

/// The C's strict first-max argmax (`llama_sampler_temp_impl` with temp <= 0).
pub fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as i32
}

/// `common_token_to_piece(vocab, token, special)` (common/common.cpp:1174-1190)
/// — with `special == false` a CONTROL/UNKNOWN token renders empty
/// (llama-vocab.cpp:3655-3663).
pub fn token_piece(vocab: &Vocab, token: i32, special: bool) -> String {
    // `llama_token_to_piece` answers "" for an out-of-range id (the buf API's
    // guard) — a model without an EOS token reaches here with LLAMA_TOKEN_NULL
    if token < 0 || token as usize >= vocab.n_tokens() as usize {
        return String::new();
    }
    let attr = vocab.token_get_attr(token);
    if !special && (attr & (llama::vocab::ATTR_UNKNOWN | llama::vocab::ATTR_CONTROL)) != 0 {
        return String::new();
    }
    vocab.token_to_piece(token).to_string()
}

/// `random_string()` (server-common.cpp:108-124) — `n` alphanumeric characters
/// (the shared id generator of `gen_chatcmplid` and `gen_tool_call_id`).
pub fn random_string(n: usize) -> String {
    const CHARS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15)
        | 1;
    let mut out = String::with_capacity(n);
    for _ in 0..n {
        // xorshift64* — the marker is not a secret, only a delimiter
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = state.wrapping_mul(0x2545F4914F6CDD1D);
        out.push(CHARS[(v >> 33) as usize % CHARS.len()] as char);
    }
    out
}

/// `format_error_response` (server-common.cpp:36-60)
pub fn json_error(message: &str, type_str: &str, code: i64) -> String {
    Json::Object(vec![
        ("error".into(), Json::Object(vec![
            ("code".into(), Json::Int(code)),
            ("message".into(), Json::String(message.to_string())),
            ("type".into(), Json::String(type_str.to_string())),
        ])),
    ])
    .dump()
}

/// The `code` inside `{"error":{"code":…}}` — what the reference uses as the
/// HTTP status (server.cpp:71-77 `res->status = json_value(error_data, "code", 500)`)
pub fn json_error_code(body: &str) -> Option<u16> {
    let v = Json::parse(body).ok()?;
    let code = v.at("error")?.at("code")?.get_i64().ok()?;
    Some(code as u16)
}

/// `error_type` (server-common.h:52-60)
pub const ERROR_TYPE_INVALID_REQUEST: (&str, i64) = ("invalid_request_error", 400);
pub const ERROR_TYPE_SERVER: (&str, i64) = ("server_error", 500);
#[allow(dead_code)]
pub const ERROR_TYPE_NOT_FOUND: (&str, i64) = ("not_found_error", 404);
/// `ERROR_TYPE_NOT_SUPPORTED` — endpoints the port does not implement answer
/// with this type (the reference's own answer for `POST /props` without
/// `--props`, server-context.cpp:4817-4821)
pub const ERROR_TYPE_NOT_SUPPORTED: (&str, i64) = ("not_supported_error", 501);
pub const ERROR_TYPE_EXCEED_CONTEXT_SIZE: (&str, i64) = ("exceed_context_size_error", 400);

/// `task_response_type` (server-task.h:33-40). The Responses/ASR/Anthropic
/// members are not ported (PARITY.md).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResponseType {
    /// `TASK_RESPONSE_TYPE_NONE` — llama.cpp native format
    None,
    /// `TASK_RESPONSE_TYPE_OAI_CHAT`
    OaiChat,
    /// `TASK_RESPONSE_TYPE_OAI_CMPL`
    OaiCmpl,
    /// `TASK_RESPONSE_TYPE_OAI_EMBD`
    OaiEmbd,
}

/// `llama_model_ftype_name` (llama-model.cpp) for the GGUF values the local
/// files use; anything else falls back to the ggml type name of the same id
/// (the reference has one string per `LLAMA_FTYPE_*`).
pub fn ftype_name(ftype: u32) -> String {
    let named = match ftype {
        0 => "all F32",
        1 => "mostly F16",
        2 => "mostly Q4_0",
        3 => "mostly Q4_1",
        7 => "mostly Q8_0",
        8 => "mostly Q5_0",
        9 => "mostly Q5_1",
        10 => "mostly Q2_K",
        11 => "mostly Q3_K - Small",
        12 => "mostly Q3_K - Medium",
        13 => "mostly Q3_K - Large",
        14 => "mostly Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "mostly Q5_K - Small",
        17 => "mostly Q5_K - Medium",
        18 => "mostly Q6_K",
        19 => "mostly IQ2_XXS",
        23 => "mostly IQ4_NL",
        30 => "mostly BF16",
        _ => return ggml::types::GgmlType::from_u32(ftype).map(|t| t.name().to_string()).unwrap_or_else(|| format!("unknown ({ftype})")),
    };
    named.to_string()
}

/// `server_slot_stats::to_json` (server-common.cpp:84-106) — the timings
/// object of every completion response.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenStats {
    /// `n_prompt_cached` — tokens reused from the slot's cache
    pub n_prompt_cached: u64,
    /// `n_prompt_processed`
    pub n_prompt_processed: u64,
    /// `n_gen` — sampled tokens (the first one is free: it comes from the last
    /// prompt batch, so the decode-step count is `n_gen - 1`)
    pub n_gen: u64,
    pub t_start: i64,
    pub t_prompt_last: i64,
    pub t_gen_last: i64,
    // ---- speculative counters (server-common.h's slot stats,
    // server-context.cpp:3057/:3970-3979) ----
    /// `n_draft_tokens` — draft tokens generated
    pub n_draft_tokens: u64,
    /// `n_draft_accepted` — draft tokens accepted by the target
    pub n_draft_accepted: u64,
    /// `n_draft_verif_steps` — verification rounds
    pub n_draft_verif_steps: u64,
}

impl GenStats {
    /// `server_slot_stats::update_prompt_start` (server-common.h:369-372)
    pub fn update_prompt_start(&mut self) {
        self.t_start = now_us();
    }

    /// `set_prompt_last` / `update_prompt_last` (server-common.h:373-378)
    pub fn update_prompt_last(&mut self) {
        self.t_prompt_last = now_us();
    }

    /// `update_gen_last` (server-common.h:379-381)
    pub fn update_gen_last(&mut self) {
        self.t_gen_last = now_us();
    }

    /// `t_elapsed_us` (server-common.h:389-391)
    pub fn t_elapsed_us(&self) -> i64 {
        if self.t_start == 0 {
            0
        } else {
            now_us() - self.t_start
        }
    }

    fn t_prompt_ms(&self) -> f64 {
        if self.t_prompt_last == 0 {
            return 0.0;
        }
        (self.t_prompt_last - self.t_start) as f64 / 1000.0
    }

    fn t_gen_us(&self) -> i64 {
        if self.t_gen_last == 0 {
            return 0;
        }
        1.max(self.t_gen_last - self.t_prompt_last)
    }

    pub fn t_gen_ms(&self) -> f64 {
        self.t_gen_us() as f64 / 1000.0
    }

    fn n_gen_steps(&self) -> u64 {
        self.n_gen.saturating_sub(1)
    }

    pub fn is_set(&self) -> bool {
        self.t_start > 0
    }

    pub fn to_json(&self) -> Json {
        let prompt_ms = self.t_prompt_ms();
        let gen_ms = self.t_gen_ms();
        let mut fields = vec![
            ("cache_n".into(), Json::Uint(self.n_prompt_cached)),
            ("prompt_n".into(), Json::Uint(self.n_prompt_processed)),
            ("prompt_ms".into(), Json::Double(prompt_ms)),
            (
                "prompt_per_token_ms".into(),
                Json::Double(if self.n_prompt_processed > 0 {
                    prompt_ms / self.n_prompt_processed as f64
                } else {
                    0.0
                }),
            ),
            (
                "prompt_per_second".into(),
                Json::Double(if prompt_ms > 0.0 {
                    1e3 / prompt_ms * self.n_prompt_processed as f64
                } else {
                    0.0
                }),
            ),
            ("predicted_n".into(), Json::Uint(self.n_gen)),
            ("predicted_ms".into(), Json::Double(gen_ms)),
            (
                "predicted_per_token_ms".into(),
                Json::Double(if self.n_gen_steps() > 0 {
                    gen_ms / self.n_gen_steps() as f64
                } else {
                    0.0
                }),
            ),
            (
                "predicted_per_second".into(),
                Json::Double(if gen_ms > 0.0 {
                    1e3 / gen_ms * self.n_gen_steps() as f64
                } else {
                    0.0
                }),
            ),
        ];
        // `if (n_draft_tokens > 0) { base["draft_n"] = ...; base["draft_n_accepted"]
        // = ...; }` (server-common.cpp:97-101)
        if self.n_draft_tokens > 0 {
            fields.push(("draft_n".into(), Json::Uint(self.n_draft_tokens)));
            fields.push(("draft_n_accepted".into(), Json::Uint(self.n_draft_accepted)));
        }
        Json::Object(fields)
    }
}

/// `stop_type` (server-task.cpp:246-255)
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StopType {
    None,
    Eos,
    Word,
    Limit,
}

impl StopType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StopType::None => "none",
            StopType::Eos => "eos",
            StopType::Word => "word",
            StopType::Limit => "limit",
        }
    }
}

/// Per-token probabilities of one generated token — `completion_token_output`
/// (server-task.cpp:264-296).
#[derive(Clone, Debug)]
pub struct TokenProbs {
    pub tok: i32,
    pub prob: f32,
    /// top `n_probs` candidates (id, piece, prob)
    pub probs: Vec<(i32, String, f32)>,
    /// the text this token contributes (`text_to_send`)
    pub text_to_send: String,
}

/// `completion_token_output::to_json` (server-task.cpp:264-280):
/// `{"id","token","bytes", post_sampling_probs ? "prob" : "logprob"}`.
fn token_probs_to_json(p: &[(i32, String, f32)], post_sampling: bool) -> Vec<Json> {
    let mut arr = Vec::new();
    for (id, piece, prob) in p {
        arr.push(Json::Object(vec![
            ("id".into(), Json::Int(*id as i64)),
            ("token".into(), Json::String(validate_utf8_prefix(piece))),
            ("bytes".into(), Json::Array(piece.as_bytes().iter().map(|&b| Json::Int(b as i64)).collect())),
            (
                if post_sampling { "prob" } else { "logprob" }.into(),
                Json::Double(if post_sampling {
                    *prob as f64
                } else {
                    logarithm(*prob)
                }),
            ),
        ]));
    }
    arr
}

/// `completion_token_output::logarithm` (server-task.cpp:304-307): 0 maps to
/// the lowest float so the JSON never carries `-inf`.
pub fn logarithm(x: f32) -> f64 {
    if x == 0.0 {
        f32::MIN as f64
    } else {
        (x as f64).ln()
    }
}

/// `completion_token_output::probs_vector_to_json` (server-task.cpp:282-302)
pub fn probs_vector_to_json(probs: &[TokenProbs], post_sampling: bool) -> Json {
    Json::Array(
        probs
            .iter()
            .map(|p| {
                Json::Object(vec![
                    ("id".into(), Json::Int(p.tok as i64)),
                    ("token".into(), Json::String(validate_utf8_prefix(&p.text_to_send))),
                    (
                        "bytes".into(),
                        Json::Array(p.text_to_send.as_bytes().iter().map(|&b| Json::Int(b as i64)).collect()),
                    ),
                    (
                        if post_sampling { "prob" } else { "logprob" }.into(),
                        Json::Double(if post_sampling { p.prob as f64 } else { logarithm(p.prob) }),
                    ),
                    (
                        if post_sampling { "top_probs" } else { "top_logprobs" }.into(),
                        Json::Array(token_probs_to_json(&p.probs, post_sampling)),
                    ),
                ])
            })
            .collect(),
    )
}

/// `get_token_probabilities` (server-common.cpp:1526-1573): the raw logits row
/// is partially sorted by logit (top `n_top`), softmaxed over the whole row,
/// and the top `n_top` probabilities are returned.
pub fn get_token_probabilities(logits: &[f32], n_top: usize) -> Vec<(i32, f32)> {
    let mut cur: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as i32, l)).collect();
    let mut n_top = n_top.min(cur.len());
    if n_top > 0 {
        cur.sort_by(|a, b| b.1.total_cmp(&a.1));
    } else {
        n_top = 0;
    }
    let max_l = if n_top > 0 {
        cur[0].1
    } else {
        cur.iter().map(|t| t.1).fold(f32::NEG_INFINITY, f32::max)
    };
    let mut cum_sum = 0.0f32;
    let mut probs: Vec<(i32, f32)> = Vec::with_capacity(cur.len());
    for &(id, l) in cur.iter() {
        let p = (l - max_l).exp();
        cum_sum += p;
        probs.push((id, p));
    }
    for p in probs.iter_mut() {
        p.1 /= cum_sum;
    }
    probs
}

/// `completion_token_output` of a `post_sampling_probs` request
/// (server-context.cpp:1968-1988): the sampler's own candidate distribution.
pub fn probs_from_candidates(candidates: &TokenDataArray, tok: i32, n_probs: usize, vocab_piece: &dyn Fn(i32) -> String) -> TokenProbs {
    let max_probs = candidates.size;
    let mut prob = 1.0f32;
    for i in 0..max_probs {
        let d = &candidates.data[i];
        if d.id == tok {
            prob = d.p;
            break;
        }
    }
    let n = n_probs.min(max_probs);
    let mut probs = Vec::with_capacity(n);
    for i in 0..n {
        let d = &candidates.data[i];
        if d.p == 0.0 {
            break;
        }
        probs.push((d.id, vocab_piece(d.id), d.p));
    }
    TokenProbs { tok, prob, probs, text_to_send: String::new() }
}

// ---------------------------------------------------------------------------
// task_params — the request-derived generation parameters (server-task.h:300-430)
// ---------------------------------------------------------------------------

/// `task_params` (server-task.h) — everything a `POST /completion` can set.
#[derive(Clone, Debug)]
pub struct TaskParams {
    pub sampling: SamplingParams,
    /// `n_predict` — the per-request limit; `-1` = the server default
    pub n_predict: i32,
    pub n_keep: i32,
    pub n_discard: i32,
    pub n_cache_reuse: i32,
    /// `sampling.n_probs` — how many top probabilities to report
    pub n_probs: i32,
    pub ignore_eos: bool,
    /// `antiprompt` (the request's `stop`)
    pub antiprompt: Vec<String>,
    pub stream: bool,
    pub cache_prompt: bool,
    pub return_tokens: bool,
    pub return_progress: bool,
    pub timings_per_token: bool,
    pub post_sampling_probs: bool,
    pub sse_ping_interval: i32,
    pub n_indent: i32,
    pub t_max_predict_ms: i64,
    pub id_slot: i32,
    pub n_cmpl: i32,
    /// the grammar text (`grammar` or `json_schema` lowered)
    pub grammar: String,
    pub grammar_lazy: bool,
    /// `generation_settings.grammar_triggers` — the typed form
    /// `server_grammar_trigger::to_json` serialises: `{"type","value"[,"token"]}`
    /// with type 0 = TOKEN, 1 = WORD (server-schema.cpp:353-380 upgrades a
    /// single-token WORD to TOKEN); consumed by the engine's lazy grammar
    pub grammar_triggers: Vec<Json>,
    pub response_fields: Vec<String>,
    pub include_usage: bool,
    pub verbose: bool,
    /// `speculative.types` of the base params (server-task.cpp:81
    /// `common_speculative_type_name_str(speculative.types)`) — the request
    /// inherits the server's `-md`/`--spec-type` selection (the per-request
    /// `speculative.n_max` adjustments of server-schema.cpp:197-206 are not
    /// ported)
    pub speculative_types: String,
    // ---- OAI-compat (server-task.h:300-430 task_params) ----
    /// `res_type` — which response shape `send_partial`/`send_final` emit
    pub res_type: ResponseType,
    /// `oaicompat_cmpl_id` — the `chatcmpl-…` id of the request
    pub oaicompat_cmpl_id: String,
    /// `oaicompat_model` — `meta->model_name`
    pub oaicompat_model: String,
    /// `chat_parser_params.format` — the chat format name the request carries
    /// (`generation_settings.chat_format`; the chat path sets "peg-native",
    /// the plain completion path keeps "Content-only")
    pub chat_format: String,
    /// `chat_parser_params.generation_prompt` — the assistant turn prefix the
    /// template emitted (`generation_settings.generation_prompt`)
    pub generation_prompt: String,
    /// `chat_parser_params.parser` — the differential autoparser's serialized
    /// arena (`chat_params.parser.save()`, server-common.cpp:1388-1390); empty
    /// means the pure-content parser (chat.cpp:1447-1523)
    pub chat_parser: String,
    /// `common_grammar_needs_prefill(params.grammar)` — the grammar begins
    /// with the generation-prompt literal; the sampler consumes those tokens
    /// before the first sample (common/sampling.cpp:294-308)
    pub grammar_prefill: bool,
    /// `chat_parser_params.reasoning_format` — "none" for a default-constructed
    /// task_params (`/props`), the request base inherits common_params'
    /// "deepseek" (server-task.cpp:136 prints it verbatim)
    pub reasoning_format: String,
    /// `sampling.preserved_tokens` — ids the sampler must not split
    /// (`generation_settings.preserved_tokens`)
    pub preserved_tokens: Vec<i32>,
    /// `embd_normalize` (-1 none, 0 max-abs int16, 1 taxicab, 2 euclidean,
    /// >2 p-norm; common.h:614 defaults 2)
    pub embd_normalize: i32,
    /// `message_spans` (server-task.h:96) — the prompt positions where user
    /// messages start, from `server_tokens::find_message_spans` over the
    /// request's `message_delimiters` (server-context.cpp:4813-4829). The
    /// checkpoint machinery reads them: a prompt batch that starts at a user
    /// message is checkpointed even mid-prompt
    /// (server-context.cpp:3978-3983), and the last user message bypasses
    /// the min-step spacing (:4053-4055)
    pub message_user_starts: Vec<usize>,
}

impl Default for TaskParams {
    fn default() -> Self {
        TaskParams {
            sampling: SamplingParams::default(),
            n_predict: -1,
            n_keep: 0,
            n_discard: 0,
            n_cache_reuse: 0,
            n_probs: 0,
            ignore_eos: false,
            antiprompt: Vec::new(),
            stream: false,
            cache_prompt: true,
            return_tokens: false,
            return_progress: false,
            timings_per_token: false,
            post_sampling_probs: false,
            sse_ping_interval: -1,
            n_indent: 0,
            t_max_predict_ms: -1,
            id_slot: -1,
            n_cmpl: 1,
            grammar: String::new(),
            grammar_lazy: false,
            grammar_triggers: Vec::new(),
            response_fields: Vec::new(),
            include_usage: false,
            verbose: false,
            speculative_types: "none".into(),
            res_type: ResponseType::None,
            oaicompat_cmpl_id: String::new(),
            oaicompat_model: String::new(),
            chat_format: "Content-only".into(),
            generation_prompt: String::new(),
            chat_parser: String::new(),
            grammar_prefill: false,
            reasoning_format: "none".into(),
            preserved_tokens: Vec::new(),
            embd_normalize: 2,
            message_user_starts: Vec::new(),
        }
    }
}

fn field_num(data: &Json, name: &str) -> Result<Option<f64>, String> {
    match data.at(name) {
        None | Some(Json::Null) => Ok(None),
        Some(v) if v.is_number() => Ok(Some(v.get_f64()?)),
        Some(_) => Err(format!("field '{name}' must be a number")),
    }
}

fn field_int(data: &Json, name: &str) -> Result<Option<i64>, String> {
    match data.at(name) {
        None | Some(Json::Null) => Ok(None),
        Some(v) if v.is_number() => Ok(Some(v.get_i64()?)),
        Some(_) => Err(format!("field '{name}' must be an integer")),
    }
}

fn field_bool(data: &Json, name: &str) -> Result<bool, String> {
    match data.at(name) {
        None | Some(Json::Null) => Ok(false),
        Some(Json::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("field '{name}' must be a boolean")),
    }
}

fn field_str(data: &Json, name: &str) -> Option<String> {
    data.at(name).and_then(|v| v.get_str().ok()).map(|s| s.to_string())
}

/// `eval_llama_cmpl_schema` (server-schema.cpp:17-330) — the fields the port
/// honours. Names, aliases and defaults follow the C; out-of-range values are
/// clamped like the C's soft limits (hard-limit violations are reported as
/// invalid_request_error, matching `field::check_hard_limits`). `vocab` is the
/// token-level lowering the C reaches through `ctx.vocab` (the preserved-token
/// and grammar-trigger handlers).
pub fn eval_llama_cmpl_schema(
    data: &Json,
    base: &TaskParams,
    vocab: &Vocab,
) -> Result<TaskParams, String> {
    if !data.is_object() {
        return Err("request body must be a JSON object".into());
    }
    let mut p = base.clone();

    // --- server-level fields ---
    p.stream = field_bool(data, "stream")?;
    if let Some(v) = data.at("stream_options").and_then(|v| v.at("include_usage")) {
        p.include_usage = v.is_boolean() && v.get_str().is_err() && matches!(v, Json::Bool(true));
    }
    p.cache_prompt = match data.at("cache_prompt") {
        Some(Json::Bool(b)) => *b,
        _ => base.cache_prompt,
    };
    p.return_tokens = field_bool(data, "return_tokens")?;
    p.return_progress = field_bool(data, "return_progress")?;
    p.timings_per_token = field_bool(data, "timings_per_token")?;
    p.post_sampling_probs = field_bool(data, "post_sampling_probs")?;
    p.verbose = field_bool(data, "verbose")?;

    // n_predict with its two aliases (server-schema.cpp:45-48)
    for name in ["n_predict", "max_completion_tokens", "max_tokens"] {
        if let Some(v) = field_int(data, name)? {
            if v < -1 {
                return Err(format!("value {v} is below the hard limit for field '{name}'"));
            }
            p.n_predict = v as i32;
            break;
        }
    }
    if let Some(v) = field_int(data, "n_keep")? {
        p.n_keep = v as i32;
    }
    if let Some(v) = field_int(data, "n_discard")? {
        p.n_discard = v.max(0) as i32;
    }
    if let Some(v) = field_int(data, "n_cache_reuse")? {
        p.n_cache_reuse = v.max(0) as i32;
    }
    if let Some(v) = field_int(data, "n_indent")? {
        p.n_indent = v.max(0) as i32;
    }
    if let Some(v) = field_int(data, "t_max_predict_ms")? {
        p.t_max_predict_ms = v;
    }
    if let Some(v) = field_int(data, "sse_ping_interval")? {
        p.sse_ping_interval = v as i32;
    }
    if let Some(v) = field_int(data, "id_slot")? {
        p.id_slot = v as i32;
    }
    if let Some(v) = field_int(data, "n_cmpl")?.or(field_int(data, "n")?) {
        p.n_cmpl = v.max(1) as i32;
    }
    if let Some(v) = data.at("response_fields") {
        p.response_fields = v.iter().filter_map(|x| x.get_str().ok().map(|s| s.to_string())).collect();
    }
    if let Some(stop) = data.at("stop") {
        match stop {
            Json::String(s) => p.antiprompt = vec![s.clone()],
            Json::Array(a) => {
                p.antiprompt = a.iter().filter_map(|v| v.get_str().ok().map(|s| s.to_string())).collect()
            }
            Json::Null => {}
            _ => return Err("field 'stop' must be a string or an array of strings".into()),
        }
    }

    // --- sampling params (server-schema.cpp:96-300) ---
    if let Some(v) = field_num(data, "temperature")? {
        if v < 0.0 {
            return Err(format!("value {v} is below the hard limit for field 'temperature'"));
        }
        p.sampling.temp = v as f32;
    }
    if let Some(v) = field_num(data, "dynatemp_range")? {
        p.sampling.dynatemp_range = v.max(0.0) as f32;
    }
    if let Some(v) = field_num(data, "dynatemp_exponent")? {
        p.sampling.dynatemp_exponent = v as f32;
    }
    if let Some(v) = field_int(data, "top_k")? {
        p.sampling.top_k = v.max(0) as i32;
    }
    if let Some(v) = field_num(data, "top_p")? {
        p.sampling.top_p = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = field_num(data, "min_p")? {
        p.sampling.min_p = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = field_num(data, "top_n_sigma")? {
        p.sampling.top_n_sigma = v as f32;
    }
    if let Some(v) = field_num(data, "xtc_probability")? {
        p.sampling.xtc_probability = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = field_num(data, "xtc_threshold")? {
        p.sampling.xtc_threshold = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = field_num(data, "typical_p")? {
        p.sampling.typ_p = v as f32;
    }
    if let Some(v) = field_int(data, "repeat_last_n")? {
        p.sampling.penalty_last_n = v.max(0) as i32;
        p.sampling.n_prev = p.sampling.penalty_last_n;
    }
    if let Some(v) = field_num(data, "repeat_penalty")? {
        p.sampling.penalty_repeat = v as f32;
    }
    if let Some(v) = field_num(data, "presence_penalty")? {
        p.sampling.penalty_present = v as f32;
    }
    if let Some(v) = field_num(data, "frequency_penalty")? {
        p.sampling.penalty_freq = v as f32;
    }
    // dry_* (server-schema.cpp:139-156): dry_base has a custom handler that
    // falls back to the server base default when v < 1.0; the two ints are
    // hard-limited to [0, INT32_MAX]
    if let Some(v) = field_num(data, "dry_multiplier")? {
        p.sampling.dry_multiplier = v as f32;
    }
    if let Some(v) = field_num(data, "dry_base")? {
        p.sampling.dry_base = if (v as f32) < 1.0 { base.sampling.dry_base } else { v as f32 };
    }
    if let Some(v) = field_int(data, "dry_allowed_length")? {
        if v < 0 {
            return Err(format!("value {v} is below the hard limit for field 'dry_allowed_length'"));
        }
        p.sampling.dry_allowed_length = v as i32;
    }
    if let Some(v) = field_int(data, "dry_penalty_last_n")? {
        if v < 0 {
            return Err(format!("value {v} is below the hard limit for field 'dry_penalty_last_n'"));
        }
        p.sampling.dry_penalty_last_n = v as i32;
    }
    // adaptive-p (server-schema.cpp:167-174): target soft-limited to <= 1.0,
    // decay hard-limited to [0.0, 0.99]
    if let Some(v) = field_num(data, "adaptive_target")? {
        p.sampling.adaptive_target = v.min(1.0) as f32;
    }
    if let Some(v) = field_num(data, "adaptive_decay")? {
        if !(0.0..=0.99).contains(&v) {
            return Err(format!("value {v} is outside the hard limits for field 'adaptive_decay'"));
        }
        p.sampling.adaptive_decay = v as f32;
    }
    // samplers (server-schema.cpp:505-515): an array of sampler type names,
    // or a single string of sampler chars
    if let Some(v) = data.at("samplers") {
        if !v.is_null() {
            match v {
                Json::Array(names) => {
                    let names: Vec<String> = names
                        .iter()
                        .filter_map(|x| x.get_str().ok().map(|s| s.to_string()))
                        .collect();
                    p.sampling.samplers = llama::sampling::common_sampler_types_from_names(&names);
                }
                Json::String(seq) => {
                    p.sampling.samplers = llama::sampling::common_sampler_types_from_chars(seq);
                }
                _ => {
                    return Err(
                        "field 'samplers' must be an array of sampler names or a string of chars"
                            .into(),
                    )
                }
            }
        }
    }
    if let Some(v) = field_int(data, "mirostat")? {
        if !(0..=2).contains(&v) {
            return Err(format!("value {v} is outside the hard limits for field 'mirostat'"));
        }
        p.sampling.mirostat = v as i32;
    }
    if let Some(v) = field_num(data, "mirostat_tau")? {
        p.sampling.mirostat_tau = v as f32;
    }
    if let Some(v) = field_num(data, "mirostat_eta")? {
        p.sampling.mirostat_eta = v as f32;
    }
    if let Some(v) = field_int(data, "seed")? {
        p.sampling.seed = v as u32;
    }
    if let Some(v) = field_int(data, "n_probs")? {
        p.n_probs = v.max(0) as i32;
    }
    if let Some(v) = field_int(data, "min_keep")? {
        p.sampling.min_keep = v.max(0) as usize;
    }
    if let Some(b) = data.at("ignore_eos") {
        if b.is_boolean() {
            p.ignore_eos = matches!(b, Json::Bool(true));
        }
    }
    if let Some(b) = data.at("logit_bias") {
        match b {
            Json::Array(a) => {
                for e in a {
                    // `{{"token", int}, {"bias", float}}` or the OAI map form
                    let (tok, bias) = match e {
                        Json::Array(pair) if pair.len() == 2 => (pair[0].get_i64()?, pair[1].get_f64()?),
                        Json::Object(_) => {
                            let t = e.at("token").ok_or("logit_bias entry needs 'token'")?.get_i64()?;
                            let b = e.at("bias").map(|v| v.get_f64()).transpose()?.unwrap_or(0.0);
                            (t, b)
                        }
                        _ => return Err("invalid logit_bias entry".into()),
                    };
                    p.sampling.logit_bias.push(LogitBias { token: tok as i32, bias: bias as f32 });
                }
            }
            Json::Object(map) => {
                for (k, v) in map.iter() {
                    let tok: i32 = k.parse().map_err(|_| "logit_bias keys must be token ids")?;
                    p.sampling.logit_bias.push(LogitBias { token: tok, bias: v.get_f64()? as f32 });
                }
            }
            Json::Null => {}
            _ => return Err("field 'logit_bias' must be an array or object".into()),
        }
    }
    if let Some(v) = data.at("preserved_tokens") {
        // server-schema.cpp:341-351: each marker survives only when it
        // tokenizes to a single id (a std::set<llama_token> → sorted unique)
        let mut ids: Vec<i32> = Vec::new();
        if let Json::Array(items) = v {
            for t in items {
                let Ok(s) = t.get_str() else { continue };
                let toks = vocab.tokenize(s, false, true);
                if toks.len() == 1 {
                    ids.push(toks[0]);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        p.preserved_tokens = ids;
    }

    // sequence breakers for DRY (server-schema.cpp:242-249): only a JSON
    // array of (non-empty overall) strings is accepted
    if let Some(v) = data.at("dry_sequence_breakers") {
        if !v.is_null() {
            let Json::Array(items) = v else {
                return Err("Error: dry_sequence_breakers must be a non-empty array of strings".into());
            };
            let breakers: Vec<String> = items
                .iter()
                .map(|x| x.get_str().map(|s| s.to_string()))
                .collect::<Result<_, _>>()
                .map_err(|_| "Error: dry_sequence_breakers must be a non-empty array of strings".to_string())?;
            if breakers.is_empty() {
                return Err("Error: dry_sequence_breakers must be a non-empty array of strings".into());
            }
            p.sampling.dry_sequence_breakers = breakers;
        }
    }

    // --- grammar (server-schema.cpp:262-290): "json_schema" only applies when
    // "grammar" is absent — the chat parse's wrapped tool-call grammar arrives
    // as "grammar" with `grammar_type == "tool_calls"` and wins over the body's
    // json_schema copy-through ---
    if data.at("json_schema").is_some() && data.at("grammar").is_none() {
        let schema = data.at("json_schema").unwrap();
        if !schema.is_null() {
            // an empty schema means any object
            let mut schema = schema.clone();
            if let Json::Object(o) = &schema {
                if o.is_empty() {
                    schema = Json::Object(vec![("type".into(), Json::String("object".into()))]);
                }
            }
            p.grammar = json_schema_to_grammar_value(&schema)?;
            // COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT — prefilled
            // (common_grammar_needs_prefill, common.h:218-221)
            p.grammar_prefill = true;
        }
    } else if let Some(g) = field_str(data, "grammar") {
        if !g.is_empty() {
            p.grammar = g;
            // "grammar_type key is set by the server when converting chat
            // template grammars" — a tool_calls grammar begins with the
            // generation-prompt literal (prefilled); a plain user grammar does
            // not (COMMON_GRAMMAR_TYPE_USER)
            let gtype = field_str(data, "grammar_type").unwrap_or_default();
            p.grammar_prefill = gtype == "tool_calls";
        }
    }
    p.grammar_lazy = field_bool(data, "grammar_lazy")?;
    if let Some(Json::Array(a)) = data.at("grammar_triggers") {
        // server-schema.cpp:353-380: a WORD trigger whose word tokenizes to a
        // single token is upgraded to a TOKEN trigger (and must already be a
        // preserved token); multi-token words stay WORD
        let mut out: Vec<Json> = Vec::new();
        for t in a {
            let ty = t.at("type").and_then(|v| v.get_i64().ok()).unwrap_or(-1);
            let value = t.at("value").and_then(|v| v.get_str().ok()).unwrap_or("").to_string();
            if ty == 1 {
                // COMMON_GRAMMAR_TRIGGER_TYPE_WORD
                let ids = vocab.tokenize(&value, false, true);
                if ids.len() == 1 {
                    if !p.preserved_tokens.contains(&ids[0]) {
                        return Err(format!(
                            "Grammar trigger word should be marked as preserved token: {value}"
                        ));
                    }
                    let trig = vec![
                        ("type".to_string(), Json::Int(0)),
                        ("value".to_string(), Json::String(value)),
                        ("token".to_string(), Json::Int(ids[0] as i64)),
                    ];
                    out.push(Json::Object(trig));
                } else {
                    out.push(Json::Object(vec![
                        ("type".to_string(), Json::Int(1)),
                        ("value".to_string(), Json::String(value)),
                    ]));
                }
            } else {
                out.push(t.clone());
            }
        }
        p.grammar_triggers = out;
        if p.grammar_lazy && p.grammar_triggers.is_empty() {
            return Err("Error: no triggers set for lazy grammar!".to_string());
        }
    }

    Ok(p)
}

/// `json_schema_to_grammar` (server-schema.cpp:252-271 → json-schema-to-grammar.cpp)
pub fn json_schema_to_grammar_value(schema: &Json) -> Result<String, String> {
    llama::json_schema::json_schema_to_grammar(schema, false)
}

/// `task_params::to_json` (server-task.cpp:30-140). `only_metrics` is what
/// `/props` uses for its `default_generation_settings`.
pub fn task_params_to_json(p: &TaskParams, only_metrics: bool) -> Json {
    let f = |x: f32| Json::Double(x as f64);
    let mut fields: Vec<(String, Json)> = Vec::new();
    let mut push = |k: &str, v: Json| fields.push((k.to_string(), v));

    push("seed", Json::Uint(p.sampling.seed as u64));
    push("temperature", f(p.sampling.temp));
    push("dynatemp_range", f(p.sampling.dynatemp_range));
    push("dynatemp_exponent", f(p.sampling.dynatemp_exponent));
    push("top_k", Json::Int(p.sampling.top_k as i64));
    push("top_p", f(p.sampling.top_p));
    push("min_p", f(p.sampling.min_p));
    push("top_n_sigma", f(p.sampling.top_n_sigma));
    push("xtc_probability", f(p.sampling.xtc_probability));
    push("xtc_threshold", f(p.sampling.xtc_threshold));
    push("typical_p", f(p.sampling.typ_p));
    push("repeat_last_n", Json::Int(p.sampling.penalty_last_n as i64));
    push("repeat_penalty", f(p.sampling.penalty_repeat));
    push("presence_penalty", f(p.sampling.penalty_present));
    push("frequency_penalty", f(p.sampling.penalty_freq));
    push("dry_multiplier", f(p.sampling.dry_multiplier));
    push("dry_base", f(p.sampling.dry_base));
    push("dry_allowed_length", Json::Int(p.sampling.dry_allowed_length as i64));
    push("dry_penalty_last_n", Json::Int(p.sampling.dry_penalty_last_n as i64));
    if !only_metrics {
        push(
            "dry_sequence_breakers",
            Json::Array(
                p.sampling
                    .dry_sequence_breakers
                    .iter()
                    .map(|s| Json::String(s.clone()))
                    .collect(),
            ),
        );
    }
    push("mirostat", Json::Int(p.sampling.mirostat as i64));
    push("mirostat_tau", f(p.sampling.mirostat_tau));
    push("mirostat_eta", f(p.sampling.mirostat_eta));
    push("adaptive_target", f(p.sampling.adaptive_target));
    push("adaptive_decay", f(p.sampling.adaptive_decay));
    if !only_metrics {
        push(
            "stop",
            Json::Array(p.antiprompt.iter().map(|s| Json::String(s.clone())).collect()),
        );
    }
    // `n_predict` is the request's value or the server default (the caller
    // substitutes it before printing, like `slot.n_predict_max`)
    push("max_tokens", Json::Int(p.n_predict as i64));
    push("n_predict", Json::Int(p.n_predict as i64));
    push("n_keep", Json::Int(p.n_keep as i64));
    push("n_discard", Json::Int(p.n_discard as i64));
    push("ignore_eos", Json::Bool(p.ignore_eos));
    push("stream", Json::Bool(p.stream));
    if !only_metrics {
        push(
            "logit_bias",
            Json::Array(
                p.sampling
                    .logit_bias
                    .iter()
                    .map(|b| {
                        Json::Object(vec![
                            ("token".into(), Json::Int(b.token as i64)),
                            ("bias".into(), Json::Double(b.bias as f64)),
                        ])
                    })
                    .collect(),
            ),
        );
    }
    push("n_probs", Json::Int(p.n_probs as i64));
    push("min_keep", Json::Int(p.sampling.min_keep as i64));
    if !only_metrics {
        push("grammar", Json::String(p.grammar.clone()));
        push("grammar_lazy", Json::Bool(p.grammar_lazy));
        push("grammar_triggers", Json::Array(p.grammar_triggers.clone()));
        push(
            "preserved_tokens",
            Json::Array(p.preserved_tokens.iter().map(|&t| Json::Int(t as i64)).collect()),
        );
    }
    // `common_chat_format_name(chat_parser_params.format)` — "Content-only"
    // for plain completions, "peg-native" on the chat path (the differential
    // autoparser's label for a generic template, chat.cpp:863-864)
    push("chat_format", Json::String(p.chat_format.clone()));
    // `common_reasoning_format_name(chat_parser_params.reasoning_format)`
    // (server-task.cpp:136) — printed verbatim, no only_metrics distinction
    push("reasoning_format", Json::String(p.reasoning_format.clone()));
    push("reasoning_in_content", Json::Bool(false));
    // `chat_parser_params.generation_prompt` (server-schema.cpp:310-315)
    push("generation_prompt", Json::String(p.generation_prompt.clone()));
    push(
        "samplers",
        Json::Array(
            p.sampling
                .samplers
                .iter()
                .map(|s| Json::String(s.to_str().to_string()))
                .collect(),
        ),
    );
    push("speculative.types", Json::String(p.speculative_types.clone()));
    push("timings_per_token", Json::Bool(p.timings_per_token));
    push("post_sampling_probs", Json::Bool(p.post_sampling_probs));
    push("backend_sampling", Json::Bool(false));
    push("lora", Json::Array(Vec::new()));

    Json::Object(fields)
}

/// `validate_utf8` (server-common.cpp) — the length of the valid prefix: a
/// multi-byte character cut off at the end shortens the result, which is how
/// the server avoids emitting half a UTF-8 sequence.
pub fn validate_utf8_prefix(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return String::new();
    }
    let mut cut = len;
    for i in 1..=4.min(len) {
        let c = bytes[len - i];
        if c & 0xE0 == 0xC0 {
            if i < 2 {
                cut = len - i;
            }
            break;
        } else if c & 0xF0 == 0xE0 {
            if i < 3 {
                cut = len - i;
            }
            break;
        } else if c & 0xF8 == 0xF0 {
            if i < 4 {
                cut = len - i;
            }
            break;
        } else if c & 0xC0 == 0x80 {
            continue;
        } else {
            break;
        }
    }
    String::from_utf8_lossy(&bytes[..cut]).into_owned()
}

/// `validate_utf8(text)` — the byte length of the valid prefix
/// (server-context.cpp:1842-1844 compares it against `generated_text.size()`)
pub fn validate_utf8_len(text: &str) -> usize {
    validate_utf8_prefix(text).len()
}

/// `string_find_partial_stop` (common/common.h:858-872)
pub fn string_find_partial_stop(text: &str, stop: &str) -> Option<usize> {
    if text.is_empty() || stop.is_empty() {
        return None;
    }
    let max_len = text.len().min(stop.len());
    let last = text.as_bytes()[text.len() - 1];
    for len in (1..=max_len).rev() {
        if stop.as_bytes()[len - 1] == last && text.ends_with(&stop[..len]) {
            return Some(text.len() - len);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// OAI-compat response pieces (server-task.cpp:365-527, server-common.cpp:1429)
// ---------------------------------------------------------------------------

/// The counters shared by every OAI response —
/// `server_task_result_cmpl_final::usage_json_oaicompat` (server-task.cpp:365-372).
pub fn usage_json_oaicompat(n_decoded: u64, n_prompt_tokens: i64, n_prompt_tokens_cache: u64) -> Json {
    Json::Object(vec![
        ("completion_tokens".into(), Json::Uint(n_decoded)),
        ("prompt_tokens".into(), Json::Int(n_prompt_tokens)),
        ("total_tokens".into(), Json::Uint(n_decoded + n_prompt_tokens.max(0) as u64)),
        (
            "prompt_tokens_details".into(),
            Json::Object(vec![("cached_tokens".into(), Json::Uint(n_prompt_tokens_cache))]),
        ),
    ])
}

/// `common_embd_normalize` (common/common.cpp:1940-1972)
pub fn common_embd_normalize(inp: &[f32], embd_norm: i32) -> Vec<f32> {
    let mut sum;
    match embd_norm {
        -1 => sum = 1.0, // no normalisation
        0 => {
            // max absolute, scaled into an int16 range
            sum = 0.0f64;
            for &x in inp {
                if sum < (x as f64).abs() {
                    sum = (x as f64).abs();
                }
            }
            sum /= 32760.0;
        }
        2 => {
            // euclidean
            sum = 0.0f64;
            for &x in inp {
                sum += (x as f64) * (x as f64);
            }
            sum = sum.sqrt();
        }
        p => {
            // p-norm (euclidean is p-norm p=2)
            sum = 0.0f64;
            for &x in inp {
                sum += (x as f64).abs().powi(p);
            }
            sum = sum.powf(1.0 / p as f64);
        }
    }
    // `const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;` — narrowed to f32
    // first, then the multiply is f32 x f32 exactly like the C
    let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0f32 };
    inp.iter().map(|&x| x * norm).collect()
}

/// `format_embeddings_response_oaicompat` (server-common.cpp:1429-1467). The
/// port answers `encoding_format: "float"` only (base64 needs the encoder
/// wired the same way; requests asking for base64 are rejected earlier).
pub fn format_embeddings_response_oaicompat(
    request: &Json,
    model_name: &str,
    embeddings: &[Json],
) -> Json {
    let mut n_tokens = 0i64;
    let mut data = Vec::new();
    for (i, elem) in embeddings.iter().enumerate() {
        data.push(Json::Object(vec![
            ("embedding".into(), elem.at("embedding").cloned().unwrap_or(Json::Array(Vec::new()))),
            ("index".into(), Json::Int(i as i64)),
            ("object".into(), Json::String("embedding".into())),
        ]));
        n_tokens += elem.at("tokens_evaluated").and_then(|v| v.get_i64().ok()).unwrap_or(0);
    }
    Json::Object(vec![
        (
            "model".into(),
            request
                .at("model")
                .and_then(|v| v.get_str().ok())
                .map(|s| Json::String(s.to_string()))
                .unwrap_or(Json::String(model_name.to_string())),
        ),
        ("object".into(), Json::String("list".into())),
        (
            "usage".into(),
            Json::Object(vec![
                ("prompt_tokens".into(), Json::Int(n_tokens)),
                ("total_tokens".into(), Json::Int(n_tokens)),
            ]),
        ),
        ("data".into(), Json::Array(data)),
    ])
}