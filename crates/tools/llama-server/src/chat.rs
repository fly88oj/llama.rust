//! `/v1/chat/completions` request parsing — port of
//! `oaicompat_chat_params_parse` (server-common.cpp:1151-1420).
//!
//! The chat template is applied through the library's jinja path
//! (`llama::chat_tools::chat_templates_apply`, the port of common/chat.cpp's
//! jinja route + the differential autoparser): the messages/tools are parsed
//! with the oaicompat helpers, rendered with the model's template, and the
//! autoparser derives the PEG parser (tool-call extraction), the tool-call
//! grammar (lazy with WORD triggers for `tool_choice: auto`) and the
//! preserved-token markers. The generated tokens are parsed back in the engine
//! (`engine::ChatStreamState`, the port of `task_result_state`).

use llama::chat_tools::{
    chat_format_name, chat_templates_apply, msgs_parse_oaicompat, tool_choice_parse_oaicompat,
    tools_parse_oaicompat, ChatMsg, ChatTemplates, ChatToolChoice, ReasoningFormat,
    TemplatesInputs,
};
use llama::json_schema::Json;

use crate::api::{json_error, ERROR_TYPE_INVALID_REQUEST, ERROR_TYPE_SERVER};

/// The `server_chat_params` subset the port honours (server-common.h:83-110 —
/// the `tmpls` the model load built via `common_chat_templates_init`).
pub struct ChatOptions<'a> {
    pub templates: &'a ChatTemplates,
    /// `opt.enable_thinking` — `params_base.enable_reasoning != 0 && template
    /// supports thinking` (server-context.cpp:1448-1453)
    pub enable_thinking: bool,
    /// `opt.reasoning_format` — the common_params default (deepseek)
    pub reasoning_format: ReasoningFormat,
}

/// The C++ exception the parse throws, mapped the way `ex_wrapper`
/// (server.cpp:54-86) maps it: `std::invalid_argument` → 400,
/// `std::runtime_error` → 500.
#[derive(Debug)]
pub struct ChatError {
    pub message: String,
    pub type_str: &'static str,
    pub code: i64,
}

impl ChatError {
    fn invalid(msg: impl Into<String>) -> Self {
        ChatError { message: msg.into(), type_str: ERROR_TYPE_INVALID_REQUEST.0, code: ERROR_TYPE_INVALID_REQUEST.1 }
    }
    fn runtime(msg: impl Into<String>) -> Self {
        ChatError { message: msg.into(), type_str: ERROR_TYPE_SERVER.0, code: ERROR_TYPE_SERVER.1 }
    }
    pub fn to_response_json(&self) -> String {
        json_error(&self.message, self.type_str, self.code)
    }
}

/// The output of a successful parse: the `llama_params` body handed to
/// `eval_llama_cmpl_schema`, plus the chat metadata the schema does not carry.
pub struct ChatParams {
    /// the body for the completion schema (`llama_params`)
    pub llama_params: Json,
    /// `common_chat_format_name(chat_params.format)` — the differential
    /// autoparser labels every generic template "peg-native" (chat.cpp:863)
    pub chat_format: &'static str,
    /// the assistant-turn prefix the template emits (`generation_prompt`)
    pub generation_prompt: String,
    /// `common_grammar_needs_prefill(chat_params.grammar)` — the grammar
    /// carries the generation-prompt literal and the sampler must consume it
    /// first (sampling.cpp:294-308)
    pub grammar_needs_prefill: bool,
    /// `chat_params.parser` — the serialized PEG arena of the autoparser
    /// (empty = the pure-content parser)
    pub parser: String,
}

/// `oaicompat_chat_params_parse` (server-common.cpp:1151-1420). The port runs
/// the jinja path (`opt.use_jinja` is true), so the `--no-jinja` tool errors
/// of server-common.cpp:1167-1174 are never taken.
pub fn oaicompat_chat_params_parse(
    body: &Json,
    opt: &ChatOptions,
) -> Result<ChatParams, ChatError> {
    let tools = body.at("tools").cloned().unwrap_or(Json::Null);
    let has_tools = matches!(&tools, Json::Array(a) if !a.is_empty());
    let stream = body.at("stream").map(|v| matches!(v, Json::Bool(true))).unwrap_or(false);
    let tool_choice = body
        .at("tool_choice")
        .and_then(|v| v.get_str().ok())
        .unwrap_or("auto")
        .to_string();

    // Handle "stop" field — a single string becomes a one-element array
    let stop: Vec<String> = match body.at("stop") {
        Some(Json::String(s)) => vec![s.clone()],
        _ => body
            .at("stop")
            .and_then(|v| match v {
                Json::Array(a) => Some(
                    a.iter().filter_map(|x| x.get_str().ok().map(|s| s.to_string())).collect(),
                ),
                _ => None,
            })
            .unwrap_or_default(),
    };

    let mut json_schema = body.at("json_schema").cloned().unwrap_or(Json::Null);
    let grammar = body
        .at("grammar")
        .and_then(|v| v.get_str().ok())
        .unwrap_or("")
        .to_string();
    if !json_schema.is_null() && !grammar.is_empty() {
        return Err(ChatError::runtime("Cannot use both json_schema and grammar"));
    }

    // Handle "response_format" field (server-common.cpp:1186-1202)
    if let Some(response_format) = body.at("response_format") {
        let response_type =
            response_format.at("type").and_then(|v| v.get_str().ok()).unwrap_or("");
        if response_type == "json_object" {
            if response_format.at("schema").is_some() || json_schema.empty() {
                json_schema = response_format.at("schema").cloned().unwrap_or(Json::Object(Vec::new()));
            }
        } else if response_type == "json_schema" {
            let schema_wrapper =
                response_format.at("json_schema").cloned().unwrap_or(Json::Object(Vec::new()));
            json_schema = schema_wrapper.at("schema").cloned().unwrap_or(Json::Object(Vec::new()));
        } else if !response_type.is_empty() && response_type != "text" {
            return Err(ChatError::invalid(format!(
                "response_format type must be one of \"text\" or \"json_object\", but got: {response_type}"
            )));
        }
    }

    // an absent or empty schema means any object
    if let Json::Object(o) = &json_schema {
        if o.is_empty() {
            json_schema = Json::Object(vec![("type".into(), Json::String("object".into()))]);
        }
    }

    // messages (server-common.cpp:1206-1266): validation; the content parts
    // are flattened by `common_chat_msgs_parse_oaicompat` below
    let Some(messages) = body.at("messages") else {
        return Err(ChatError::invalid("'messages' is required"));
    };
    let Json::Array(_) = messages else {
        return Err(ChatError::invalid("Expected 'messages' to be an array"));
    };
    for msg in messages.iter() {
        let role = msg.at("role").and_then(|v| v.get_str().ok()).unwrap_or("");
        if role != "assistant" && msg.at("content").is_none() {
            return Err(ChatError::invalid("All non-assistant messages must contain 'content'"));
        }
        if role == "assistant" && msg.at("content").is_none() {
            if msg.at("tool_calls").is_none() {
                return Err(ChatError::invalid(
                    "Assistant message must contain either 'content' or 'tool_calls'!",
                ));
            }
            // "avoid errors with no content" — the message parses via tool_calls
            continue;
        }
        let Some(content) = msg.at("content") else { continue };
        if matches!(content, Json::String(_) | Json::Null) {
            continue;
        }
        let Json::Array(parts) = content else {
            return Err(ChatError::invalid("Expected 'content' to be a string or an array"));
        };
        for p in parts {
            let ptype = p.at("type").and_then(|v| v.get_str().ok()).unwrap_or("");
            if ptype == "image_url" {
                return Err(ChatError::runtime(
                    "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj",
                ));
            } else if ptype == "input_audio" {
                return Err(ChatError::runtime(
                    "audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj",
                ));
            } else if ptype == "input_video" || ptype == "video_url" {
                return Err(ChatError::runtime(
                    "video input is not supported - hint: if this is unexpected, you may need to provide the mmproj",
                ));
            } else if ptype != "text" && ptype != "media_marker" {
                return Err(ChatError::invalid("unsupported content[].type"));
            }
        }
    }

    // `auto caps = common_chat_templates_get_caps(opt.tmpls.get())`
    // (server-common.cpp:1268)
    let caps = opt.templates.get_caps();
    let cap = |name: &str| caps.iter().find(|(k, _)| k == name).map(|(_, v)| *v).unwrap_or(false);

    // the template inputs (server-common.cpp:1270-1330)
    let msgs: Vec<ChatMsg> = msgs_parse_oaicompat(messages).map_err(|e| ChatError::invalid(e))?;
    let parsed_tools = tools_parse_oaicompat(&tools).map_err(|e| ChatError::invalid(e))?;
    let parsed_tool_choice =
        tool_choice_parse_oaicompat(&tool_choice).map_err(|e| ChatError::invalid(e))?;
    let json_schema_text = if json_schema.is_null() { String::new() } else { json_schema.dump() };

    // `add_generation_prompt` / `continue_final_message`
    // (server-common.cpp:1285-1314; `opt.prefill_assistant` is false)
    let mut add_generation_prompt =
        body.at("add_generation_prompt").map(|v| matches!(v, Json::Bool(true))).unwrap_or(true);
    let continue_final = body
        .at("continue_final_message")
        .map(continuation)
        .unwrap_or(Continuation::None);
    if continue_final != Continuation::None {
        if msgs.len() >= 2 && msgs[msgs.len() - 2].role == "assistant" {
            return Err(ChatError::invalid(
                "Cannot have 2 or more assistant messages at the end of the list.",
            ));
        }
        if add_generation_prompt {
            return Err(ChatError::invalid(
                "Cannot set both add_generation_prompt and continue_final_message to true.",
            ));
        }
        add_generation_prompt = false;
    }
    if continue_final != Continuation::None
        && !msgs.is_empty()
        && msgs.last().unwrap().role == "assistant"
        && !msgs.last().unwrap().tool_calls.is_empty()
    {
        return Err(ChatError::invalid(
            "Cannot continue an assistant message that contains tool calls.",
        ));
    }

    // `reasoning_format` (server-common.cpp:1315-1318)
    let mut reasoning_format = opt.reasoning_format;
    if let Some(v) = body.at("reasoning_format").and_then(|v| v.get_str().ok()) {
        reasoning_format = llama::chat_tools::reasoning_format_from_name(v)
            .map_err(|e| ChatError::invalid(e))?;
    }

    let mut enable_thinking = opt.enable_thinking;
    // tool-call parsing is on whenever tools are present and not turned off
    // (server-common.cpp:1324-1329); a custom grammar conflicts with them
    if !parsed_tools.is_empty() && parsed_tool_choice != ChatToolChoice::None {
        if body.at("grammar").is_some() {
            return Err(ChatError::invalid(
                "Cannot use custom grammar constraints with tools.",
            ));
        }
    }

    // chat_template_kwargs: merge the body's object into the kwargs
    // (server-common.cpp:1333-1337)
    let mut kwargs: Vec<(String, String)> = Vec::new();
    if let Some(Json::Object(items)) = body.at("chat_template_kwargs") {
        for (k, v) in items {
            kwargs.push((k.clone(), v.dump()));
        }
    }
    // the "enable_thinking" kwarg overrides the default
    // (server-common.cpp:1340-1346)
    let enable_thinking_kwarg =
        kwargs.iter().find(|(k, _)| k == "enable_thinking").map(|(_, v)| v.clone());
    match enable_thinking_kwarg.as_deref() {
        Some("true") => enable_thinking = true,
        Some("false") => enable_thinking = false,
        Some(s) if !s.is_empty() && s.starts_with('"') => {
            return Err(ChatError::invalid(
                "invalid type for \"enable_thinking\" (expected boolean, got string)",
            ));
        }
        _ => {}
    }
    // the OAI "reasoning_effort" field; "none" disables reasoning
    // (server-common.cpp:1349-1358)
    if let Some(effort) = body.at("reasoning_effort").and_then(|v| v.get_str().ok()) {
        if effort == "none" {
            enable_thinking = false;
            kwargs.retain(|(k, _)| k != "reasoning_effort");
        } else if !effort.is_empty() {
            kwargs.retain(|(k, _)| k != "reasoning_effort");
            kwargs.push(("reasoning_effort".into(), Json::String(effort.to_string()).dump()));
        }
    }

    let inputs = TemplatesInputs {
        messages: msgs,
        tools: parsed_tools,
        tool_choice: parsed_tool_choice,
        grammar: grammar.clone(),
        json_schema: json_schema_text,
        add_generation_prompt,
        use_jinja: true,
        parallel_tool_calls: body
            .at("parallel_tool_calls")
            .map(|v| matches!(v, Json::Bool(b) if *b))
            .unwrap_or_else(|| cap("supports_parallel_tool_calls")),
        reasoning_format,
        enable_thinking,
        now: None,
        chat_template_kwargs: kwargs,
        ..TemplatesInputs::default()
    };
    // `common_chat_templates_apply` (server-common.cpp:1360, chat.cpp:1434-1439)
    let chat_params = chat_templates_apply(opt.templates, &inputs)
        .map_err(|e| ChatError::invalid(format!("Failed to apply chat template: {e}")))?;
    let chat_format = chat_format_name(chat_params.format).map_err(ChatError::runtime)?;

    // build `llama_params` (server-common.cpp:1362-1392)
    let mut llama_params: Vec<(String, Json)> = Vec::new();
    let mut stop_json: Vec<Json> = stop.iter().map(|s| Json::String(s.clone())).collect();
    // `for (const auto & stop : chat_params.additional_stops)`
    // (server-common.cpp:1380-1382)
    for s in &chat_params.additional_stops {
        stop_json.push(Json::String(s.clone()));
    }
    llama_params.push(("stop".into(), Json::Array(stop_json)));
    llama_params.push(("chat_format".into(), Json::Int(chat_params.format as i64)));
    llama_params.push(("prompt".into(), Json::String(chat_params.prompt.clone())));
    let mut grammar_needs_prefill = false;
    if !chat_params.grammar.is_empty() {
        llama_params.push(("grammar".into(), Json::String(chat_params.grammar.clone())));
        llama_params.push(("grammar_type".into(), Json::String("tool_calls".into())));
        grammar_needs_prefill = true;
    }
    llama_params.push(("grammar_lazy".into(), Json::Bool(chat_params.grammar_lazy)));
    // `server_grammar_trigger::to_json` (server-common.h:84-92): the type is
    // the enum's integer (common.h:143-148: TOKEN=0, WORD=1, PATTERN=2,
    // PATTERN_FULL=3). The autoparser emits WORD; the specialized parsers
    // (gpt-oss, functionary, muse-glimmer, …) emit PATTERN.
    let triggers: Vec<Json> = chat_params
        .grammar_triggers
        .iter()
        .map(|t| {
            let ty = match t.ty {
                llama::chat_tools::GrammarTriggerType::Token => 0,
                llama::chat_tools::GrammarTriggerType::Word => 1,
                llama::chat_tools::GrammarTriggerType::Pattern => 2,
                llama::chat_tools::GrammarTriggerType::PatternFull => 3,
            };
            Json::Object(vec![
                ("type".into(), Json::Int(ty)),
                ("value".into(), Json::String(t.word.clone())),
            ])
        })
        .collect();
    llama_params.push(("grammar_triggers".into(), Json::Array(triggers)));
    llama_params.push((
        "preserved_tokens".into(),
        Json::Array(chat_params.preserved_tokens.iter().map(|t| Json::String(t.clone())).collect()),
    ));
    llama_params.push((
        "generation_prompt".into(),
        Json::String(chat_params.generation_prompt.clone()),
    ));
    if !chat_params.parser.is_empty() {
        llama_params.push(("chat_parser".into(), Json::String(chat_params.parser.clone())));
    }
    // `chat_params.message_delimiters.to_json()` (chat.cpp:115-124)
    let delims: Vec<Json> = chat_params
        .message_delimiters
        .iter()
        .map(|(role, d)| {
            Json::Object(vec![
                ("role".into(), Json::String(role.clone())),
                ("delimiter".into(), Json::String(d.clone())),
            ])
        })
        .collect();
    llama_params.push(("message_delimiters".into(), Json::Array(delims)));

    // Handle "logprobs" field (server-common.cpp:1394-1401)
    let logprobs = body.at("logprobs").map(|v| matches!(v, Json::Bool(true))).unwrap_or(false);
    if logprobs {
        if has_tools && stream {
            return Err(ChatError::invalid("logprobs is not supported with tools + stream"));
        }
        let top = body
            .at("top_logprobs")
            .and_then(|v| v.get_i64().ok())
            .unwrap_or(20);
        llama_params.push(("n_probs".into(), Json::Int(top)));
    } else if matches!(body.at("top_logprobs"), Some(v) if !v.is_null()) {
        return Err(ChatError::invalid("top_logprobs requires logprobs to be set to true"));
    }

    // Copy remaining properties (server-common.cpp:1404-1410): everything the
    // chat parse has not set flows through, so OAI fields (max_tokens,
    // temperature, …) and llama.cpp-specific ones (mirostat, …) both reach the
    // completion schema; "n_predict" always overwrites.
    if let Json::Object(items) = body {
        for (k, v) in items {
            let already = llama_params.iter().any(|(n, _)| n == k);
            if !already || k == "n_predict" {
                llama_params.push((k.clone(), v.clone()));
            }
        }
    }

    Ok(ChatParams {
        llama_params: Json::Object(llama_params),
        chat_format,
        generation_prompt: chat_params.generation_prompt,
        grammar_needs_prefill,
        parser: chat_params.parser,
    })
}

/// `common_chat_continuation_parse` (chat.cpp:622-636) — bool or the named modes.
#[derive(PartialEq)]
enum Continuation {
    None,
    Auto,
}

fn continuation(v: &Json) -> Continuation {
    match v {
        Json::Bool(true) => Continuation::Auto,
        Json::String(s) if s == "auto" => Continuation::Auto,
        _ => Continuation::None,
    }
}

// ---------------------------------------------------------------------------
// prompt tokenization — `llama_tokenize(model, text, add_special=true,
// parse_special=true)`. The port's `crates/llama/src/vocab.rs::
// tokenizer_st_partition` mishandles a special token that occurs twice inside
// one raw fragment (its Vec splice keeps the already-pushed right piece, so
// the text after the second occurrence is emitted twice — see the
// `tokenize_probe` test). crates/llama is outside this crate's ownership, so
// the server carries a corrected port of the partition (llama-vocab.cpp:
// 3250-3395) and delegates each raw fragment to `Vocab::tokenize` for the
// BPE/SPM work itself.
// ---------------------------------------------------------------------------


/// `llama_tokenize` for a rendered chat prompt: split the raw text around
/// every special token (`tokenizer_st_partition`), then tokenize each raw
/// fragment. The BPE assembly (bos/eos flags, per-fragment escaping) is
/// delegated to `Vocab::tokenize(frag, false, false)`; the SPM space-prefix
/// rule spans fragments, so SPM vocabs fall back to the vocab's own entry
/// point (correct unless the same special token repeats — the vocab bug).
pub fn tokenize_prompt(vocab: &llama::vocab::Vocab, raw: &str, add_special: bool) -> Vec<i32> {
    // `llama_tokenize` with parse_special=true. This used to carry a local
    // re-implementation of `tokenizer_st_partition` because the vocab's own
    // partition duplicated the tail after the second occurrence of a repeated
    // special token (it pushed every right piece instead of only the final
    // remainder); that bug is fixed at the source (vocab.rs, see
    // `chat_prompt_special_tokens` below), so the plain entry point is correct.
    vocab.tokenize(raw, add_special, true)
}

/// `common_chat_templates_support_enable_thinking` (chat.cpp:358-370): probe
/// the templates with a synthetic user turn and check whether the autoparser
/// found a reasoning block.
pub fn templates_support_enable_thinking(templates: &ChatTemplates) -> bool {
    let inputs = TemplatesInputs {
        messages: vec![ChatMsg { role: "user".into(), content: "test".into(), ..Default::default() }],
        enable_thinking: true,
        reasoning_format: ReasoningFormat::Deepseek,
        add_generation_prompt: true,
        ..TemplatesInputs::default()
    };
    match chat_templates_apply(templates, &inputs) {
        Ok(p) => p.supports_thinking,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    fn vocab() -> Option<llama::vocab::Vocab> {
        if !std::path::Path::new(QWEN).exists() {
            eprintln!("skip: {QWEN} missing");
            return None;
        }
        let f = std::fs::File::open(QWEN).unwrap();
        // SAFETY: read-only model file
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
        let gguf = ggml::Gguf::from_bytes(mmap).unwrap();
        Some(llama::vocab::Vocab::load(&gguf).unwrap())
    }

    fn qwen_templates() -> Option<ChatTemplates> {
        if !std::path::Path::new(QWEN).exists() {
            eprintln!("skip: {QWEN} missing");
            return None;
        }
        let f = std::fs::File::open(QWEN).unwrap();
        // SAFETY: read-only model file
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
        let gguf = ggml::Gguf::from_bytes(mmap).unwrap();
        let src = gguf.get_str("tokenizer.chat_template").unwrap_or("").to_string();
        llama::chat_tools::ChatTemplates::init(&llama::chat_tools::ChatTemplatesInit {
            chat_template_override: src,
            chat_template_tool_use: String::new(),
            bos_token: "<|endoftext|>".into(),
            eos_token: "<|im_end|>".into(),
            add_bos: false,
            add_eos: false,
        })
        .ok()
    }

    fn weather_tools() -> Json {
        Json::parse(
            r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]"#,
        )
        .unwrap()
    }

    fn body(tools: Option<Json>, tool_choice: Option<&str>, user: &str) -> Json {
        let mut items: Vec<(String, Json)> = Vec::new();
        items.push((
            "messages".into(),
            Json::Array(vec![Json::Object(vec![
                ("role".into(), Json::String("user".into())),
                ("content".into(), Json::String(user.into())),
            ])]),
        ));
        if let Some(t) = tools {
            items.push(("tools".into(), t));
        }
        if let Some(c) = tool_choice {
            items.push(("tool_choice".into(), Json::String(c.into())));
        }
        Json::Object(items)
    }

    /// a rendered qwen2.5 chat prompt tokenizes with its `<|im_*|>` markers
    /// as single special ids — the reference `/tokenize` answers. The vocab's
    /// own entry point duplicates the text after the second occurrence of a
    /// repeated special token (its `tokenizer_st_partition` keeps the pushed
    /// right piece), which is what [`tokenize_prompt`] exists to avoid.
    #[test]
    fn chat_prompt_special_tokens() {
        let Some(v) = vocab() else { return };
        let got = tokenize_prompt(&v, "<|im_start|>A<|im_end|>\n<|im_start|>B", false);
        assert_eq!(got, vec![151644, 32, 151645, 198, 151644, 33], "got {got:?}");
    }

    #[test]
    fn chat_prompt_special_tokens_long() {
        let Some(v) = vocab() else { return };
        let s = "<|im_start|>system\nYou are Qwen.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let got = tokenize_prompt(&v, s, true);
        // reference answer (fresh llama-server /tokenize, add_special=true)
        let want = vec![
            151644, 8948, 198, 2610, 525, 1207, 16948, 13, 151645, 198, 151644, 872, 198, 6023,
            151645, 198, 151644, 77091, 198,
        ];
        assert_eq!(got, want, "got {got:?}");
    }

    /// the plain-string and no-repeat cases agree with the vocab's own
    /// tokenizer (the workaround must not change them)
    #[test]
    fn chat_prompt_matches_vocab_when_no_repeat() {
        let Some(v) = vocab() else { return };
        for s in ["hello world", "A<|im_end|>B<|im_start|>C", "<|im_start|>A<|im_end|>"] {
            assert_eq!(tokenize_prompt(&v, s, false), v.tokenize(s, false, true), "{s}");
        }
    }

    /// the tools request end-to-end at the parse level: the qwen2.5 template
    /// renders the tool list, the autoparser derives the lazy tool-call
    /// grammar with the `<tool_call>` WORD trigger, and the reference server's
    /// captured generation (parity/chat_tools_server_raw.json) parses to the
    /// same tool call the reference answered with.
    #[test]
    fn tools_apply_and_parse() {
        let Some(t) = qwen_templates() else { return };
        let opt = ChatOptions {
            templates: &t,
            enable_thinking: false,
            reasoning_format: ReasoningFormat::None,
        };
        let v = vocab().unwrap();
        let parsed = oaicompat_chat_params_parse(
            &body(Some(weather_tools()), None, "What is the weather in Tokyo right now? Use the tool."),
            &opt,
        )
        .unwrap();
        assert_eq!(parsed.chat_format, "peg-native");
        let lp = &parsed.llama_params;
        // the tool section is rendered into the system prompt
        let prompt = lp.at("prompt").unwrap().get_str().unwrap();
        assert!(prompt.contains("# Tools"), "{prompt}");
        assert!(prompt.contains("\"name\": \"get_weather\""), "{prompt}");
        // lazy grammar + the `<tool_call>` WORD trigger (the per-call marker
        // carries its trailing newline, so it is multi-token — the schema
        // upgrade to a TOKEN trigger does not apply)
        assert_eq!(lp.at("grammar_lazy"), Some(&Json::Bool(true)));
        assert!(!lp.at("grammar").unwrap().get_str().unwrap().is_empty());
        let trig = lp.at("grammar_triggers").unwrap();
        assert_eq!(trig.dump(), r#"[{"type":1,"value":"<tool_call>\n"}]"#);
        // preserved markers of the template (single-token only — the lowering
        // happens in the schema)
        let pt = lp.at("preserved_tokens").unwrap().dump();
        assert_eq!(pt, r#"["<tool_call>","</tool_call>"]"#);
        assert!(!parsed.parser.is_empty());
        assert_eq!(lp.at("generation_prompt").unwrap().get_str().unwrap(), "<|im_start|>assistant\n");
        assert!(parsed.grammar_needs_prefill);

        // the schema keeps the multi-token word as a WORD trigger
        // (server-schema.cpp:366-377 — only a single-token word upgrades)
        let base = crate::api::TaskParams::default();
        let schema = crate::api::eval_llama_cmpl_schema(lp, &base, &v).unwrap();
        assert!(schema.grammar_lazy);
        assert_eq!(schema.grammar_triggers.len(), 1);
        assert_eq!(schema.grammar_triggers[0].at("type"), Some(&Json::Int(1)));
        assert!(schema.preserved_tokens.contains(&151657));
        assert!(schema.grammar_prefill);

        // the engine's stream state parses the reference's captured output
        let raw = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"unit\": \"fahrenheit\"}}\n</tool_call>";
        let mut pp = llama::chat_tools::ChatParserParams::default();
        pp.format = llama::chat_tools::ChatFormat::PegNative;
        pp.generation_prompt = llama::chat_tools::ChatInput::from_plain(parsed.generation_prompt.clone());
        pp.parser = llama::peg::PegArena::default();
        pp.parser.load(&parsed.parser).unwrap();
        let msg = llama::chat_tools::chat_parse(&llama::chat_tools::ChatInput::from(raw), false, &pp).unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert_eq!(
            msg.tool_calls[0].arguments,
            r#"{"city": "Tokyo", "unit": "fahrenheit"}"#
        );

        // a partial parse of the same text yields the same tool call once the
        // closing tag is present, and diffs stream the name/arguments
        let partial = llama::chat_tools::chat_parse(
            &llama::chat_tools::ChatInput::from(
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tok",
            ),
            true,
            &pp,
        )
        .unwrap();
        assert_eq!(partial.tool_calls.len(), 1);
        assert_eq!(partial.tool_calls[0].name, "get_weather");
    }

    /// `tool_choice: required` swaps the lazy grammar for an eager one
    /// (chat-auto-parser-generator.cpp:78-80): grammar_lazy false, the grammar
    /// still present (prefilled with the generation prompt).
    #[test]
    fn tools_required_eager_grammar() {
        let Some(t) = qwen_templates() else { return };
        let opt = ChatOptions {
            templates: &t,
            enable_thinking: false,
            reasoning_format: ReasoningFormat::None,
        };
        let parsed = oaicompat_chat_params_parse(
            &body(Some(weather_tools()), Some("required"), "What is the weather in Tokyo?"),
            &opt,
        )
        .unwrap();
        assert_eq!(parsed.llama_params.at("grammar_lazy"), Some(&Json::Bool(false)));
        assert!(!parsed.llama_params.at("grammar").unwrap().get_str().unwrap().is_empty());
        assert_eq!(parsed.llama_params.at("grammar_triggers").unwrap().dump(), "[]");
        assert!(parsed.grammar_needs_prefill);
    }

    /// a tools request with a plain-text (no tool call) answer: no grammar is
    /// attached when the model answers without `<tool_call>`, and the parse
    /// yields plain content — the finish_reason "stop" side of the surface.
    #[test]
    fn tools_parse_plain_answer() {
        let Some(t) = qwen_templates() else { return };
        let opt = ChatOptions {
            templates: &t,
            enable_thinking: false,
            reasoning_format: ReasoningFormat::None,
        };
        let parsed = oaicompat_chat_params_parse(
            &body(Some(weather_tools()), None, "hi"),
            &opt,
        )
        .unwrap();
        let mut pp = llama::chat_tools::ChatParserParams::default();
        pp.format = llama::chat_tools::ChatFormat::PegNative;
        pp.generation_prompt = llama::chat_tools::ChatInput::from_plain(parsed.generation_prompt.clone());
        pp.parser = llama::peg::PegArena::default();
        pp.parser.load(&parsed.parser).unwrap();
        let msg =
            llama::chat_tools::chat_parse(&llama::chat_tools::ChatInput::from("Hello! How can I help?"), false, &pp)
                .unwrap();
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, "Hello! How can I help?");
    }
}
