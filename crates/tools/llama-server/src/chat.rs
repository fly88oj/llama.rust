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
    chat_continuation_parse, chat_format_name, chat_templates_apply, msgs_parse_oaicompat,
    tool_choice_parse_oaicompat, tools_parse_oaicompat, ChatContinuation, ChatMsg, ChatSession,
    ChatSessionParams, ChatTemplates, ChatToolChoice, ReasoningFormat, TemplatesInputs,
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
    /// `opt.reasoning_budget` — `params_base.sampling.reasoning_budget_tokens`
    /// (server-context.cpp:1619); the port has no `--reasoning-budget-tokens`
    /// CLI yet, so the server passes -1 and only per-request values apply
    pub reasoning_budget: i64,
    /// `opt.reasoning_budget_message` —
    /// `params_base.sampling.reasoning_budget_message` (server-context.cpp:1620)
    pub reasoning_budget_message: String,
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
/// `eval_llama_cmpl_schema`, plus the chat session that owns the prompt's
/// parsing state (8b5436102 — the grammar/triggers/parser no longer ride
/// the JSON body).
pub struct ChatParams {
    /// the body for the completion schema (`llama_params`)
    pub llama_params: Json,
    /// `out_session` — rendered prompt + PEG parser + sampling side, applied
    /// to the task via `Task::apply_chat_session` (server-task.h:261-271)
    pub session: ChatSession,
}

/// `oaicompat_chat_params_parse` (server-common.cpp:1266-1493, 8b5436102). The
/// port runs the jinja path (`opt.use_jinja` is true), so the `--no-jinja`
/// tool errors of server-common.cpp:1283-1290 are never taken.
pub fn oaicompat_chat_params_parse(
    vocab: Option<&llama::vocab::Vocab>,
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
    // (server-common.cpp:1368-1388; `opt.prefill_assistant` is false)
    let mut add_generation_prompt =
        body.at("add_generation_prompt").map(|v| matches!(v, Json::Bool(true))).unwrap_or(true);
    let continue_final = match body.at("continue_final_message") {
        Some(v) => chat_continuation_parse(v).map_err(ChatError::invalid)?,
        None => ChatContinuation::None,
    };
    if continue_final != ChatContinuation::None {
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
    if continue_final != ChatContinuation::None
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
    // Apply chat template to the list of messages (server-common.cpp:1430-1432,
    // 8b5436102): the session owns the prompt, the PEG parser and the sampling
    // side from here on
    let session_params = ChatSessionParams {
        echo: body.at("echo").map(|v| matches!(v, Json::Bool(true))).unwrap_or(false),
        debug: false,
    };
    let session = ChatSession::new(opt.templates, vocab, &inputs, &session_params)
        .map_err(|e| ChatError::invalid(format!("Failed to apply chat template: {e}")))?;

    // build `llama_params` (server-common.cpp:1434 + reasoning budget
    // 1437-1455): the chat metadata itself no longer rides the JSON
    let mut llama_params: Vec<(String, Json)> = Vec::new();
    let mut stop_json: Vec<Json> = stop.iter().map(|s| Json::String(s.clone())).collect();
    // only the body's stops land in "stop" — the template's additional stops
    // become task antiprompts via `apply_chat_session` (server-common.cpp:1434)
    llama_params.push(("stop".into(), Json::Array(stop_json)));
    llama_params.push(("prompt".into(), Json::String(session.prompt().to_string())));

    // Reasoning budget: pass parameters through to sampling layer
    // (server-common.cpp:1437-1455). The completion schema consumes the
    // reasoning_budget_* fields (api.rs `eval_llama_cmpl_schema`,
    // server-schema.cpp:333-383) and the budget sampler runs inside
    // `SamplingContext` (sampling.rs, common/reasoning-budget.cpp).
    {
        let mut reasoning_budget = body
            .at("reasoning_budget_tokens")
            .and_then(|v| v.get_i64().ok())
            .or_else(|| body.at("thinking_budget_tokens").and_then(|v| v.get_i64().ok()))
            .unwrap_or(-1);
        if reasoning_budget == -1 {
            reasoning_budget = opt.reasoning_budget;
        }
        if !session.thinking_end_tags().is_empty() {
            llama_params.push(("reasoning_budget_tokens".into(), Json::Int(reasoning_budget)));
            llama_params.push((
                "reasoning_budget_start_tag".into(),
                Json::String(session.thinking_start_tag().to_string()),
            ));
            llama_params.push((
                "reasoning_budget_end_tags".into(),
                Json::Array(
                    session
                        .thinking_end_tags()
                        .iter()
                        .map(|t| Json::String(t.clone()))
                        .collect(),
                ),
            ));
            let budget_message = body
                .at("reasoning_budget_message")
                .and_then(|v| v.get_str().ok())
                .map(|s| s.to_string())
                .unwrap_or_else(|| opt.reasoning_budget_message.clone());
            llama_params.push(("reasoning_budget_message".into(), Json::String(budget_message)));
            llama_params.push((
                "reasoning_control".into(),
                Json::Bool(
                    body.at("reasoning_control")
                        .map(|v| matches!(v, Json::Bool(true)))
                        .unwrap_or(false),
                ),
            ));
        }
    }

    // Handle "logprobs" field (server-common.cpp:1457-1466)
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

    // Copy remaining properties (server-common.cpp:1468-1477): everything the
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

    // the session owns these, the server applies them with
    // `Task::apply_chat_session()` (server-common.cpp:1479-1487)
    llama_params.retain(|(k, _)| {
        !matches!(k.as_str(), "grammar_lazy" | "grammar_triggers" | "preserved_tokens")
    });
    if !session.grammar().is_empty() {
        llama_params.retain(|(k, _)| !matches!(k.as_str(), "grammar" | "json_schema"));
    }

    Ok(ChatParams {
        llama_params: Json::Object(llama_params),
        session,
    })
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

    /// corpus file: <LLAMA_RUST_DATA_HOME>/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
    #[allow(non_snake_case)]
    fn QWEN() -> &'static str {
        crate::chat::__llama_rust_data_path("localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    }

    fn vocab() -> Option<llama::vocab::Vocab> {
        if !std::path::Path::new(QWEN()).exists() {
            eprintln!("skip: {} missing", QWEN());
            return None;
        }
        let f = std::fs::File::open(QWEN()).unwrap();
        // SAFETY: read-only model file
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
        let gguf = ggml::Gguf::from_bytes(mmap).unwrap();
        Some(llama::vocab::Vocab::load(&gguf).unwrap())
    }

    fn qwen_templates() -> Option<ChatTemplates> {
        if !std::path::Path::new(QWEN()).exists() {
            eprintln!("skip: {} missing", QWEN());
            return None;
        }
        let f = std::fs::File::open(QWEN()).unwrap();
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

    /// the Qwen3 template of the reference tree — the same file the upstream
    /// `test_reasoning_budget_*` tests read (tests/test-chat.cpp:5914-5981);
    /// its `<think>...</think>` markers are what the autoparser detects
    fn qwen3_templates() -> Option<ChatTemplates> {
        /// corpus file: <LLAMA_RUST_DATA_HOME>/llm/llama.cpp-next/models/templates/Qwen-Qwen3-0.6B.jinja
        #[allow(non_snake_case)]
        fn QWEN3_JINJA() -> &'static str {
            crate::chat::__llama_rust_data_path("llm/llama.cpp-next/models/templates/Qwen-Qwen3-0.6B.jinja")
        }
        if !std::path::Path::new(QWEN3_JINJA()).exists() {
            eprintln!("skip: {} missing", QWEN3_JINJA());
            return None;
        }
        let src = std::fs::read_to_string(QWEN3_JINJA()).unwrap();
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

    /// `test_reasoning_budget_tokens_per_request` +
    /// `test_reasoning_budget_message_per_request`
    /// (tests/test-chat.cpp:5914-5981, the 99f3dc322 tests): the per-request
    /// `reasoning_budget_tokens` / `reasoning_budget_message` win over the
    /// server defaults, the completion schema lowers the four keys into the
    /// sampling params (server-schema.cpp:333-383), and the budget sampler
    /// runs off them (common/sampling.cpp:313-326) — budget 0 forces the end
    /// tag immediately once the start tag is seen.
    #[test]
    fn reasoning_budget_per_request() {
        let (Some(v), Some(t)) = (vocab(), qwen3_templates()) else {
            return;
        };
        let opt = ChatOptions {
            templates: &t,
            enable_thinking: true,
            reasoning_format: ReasoningFormat::None,
            reasoning_budget: -1,
            reasoning_budget_message: "server default".into(),
        };

        // (1) the per-request token budget: 0 (suppress thinking) wins over
        // the server default (-1)
        let body = Json::parse(
            r#"{"messages":[{"role":"user","content":"hello"}],"reasoning_budget_tokens":0,
                "reasoning_budget_message":"per-request message"}"#,
        )
        .unwrap();
        let parsed = oaicompat_chat_params_parse(Some(&v), &body, &opt).unwrap();
        let lp = &parsed.llama_params;
        assert_eq!(lp.at("reasoning_budget_tokens").unwrap().get_i64().unwrap(), 0);
        let start_tag = lp
            .at("reasoning_budget_start_tag")
            .and_then(|x| x.get_str().ok())
            .map(|s| s.to_string())
            .expect("reasoning_budget_start_tag missing (thinking markers not detected)");
        let end_tag = lp
            .at("reasoning_budget_end_tags")
            .and_then(|x| match x {
                Json::Array(a) => a.first().and_then(|s| s.get_str().ok()).map(|s| s.to_string()),
                _ => None,
            })
            .expect("reasoning_budget_end_tags missing");
        assert_eq!(
            lp.at("reasoning_budget_message").unwrap().get_str().unwrap(),
            "per-request message"
        );

        // (2) the schema consumption (server-schema.cpp:333-383): the four
        // keys lower into the sampling params, and the message's tokens are
        // prepended to the first end tag as the forced sequence
        let base = crate::api::TaskParams::default();
        let p = crate::api::eval_llama_cmpl_schema(lp, &base, &v).unwrap();
        assert_eq!(p.sampling.reasoning_budget_tokens, 0);
        assert_eq!(p.sampling.reasoning_budget_start, v.tokenize(&start_tag, false, true));
        assert_eq!(p.sampling.reasoning_budget_end, vec![v.tokenize(&end_tag, false, true)]);
        assert_eq!(p.sampling.reasoning_budget_message, "per-request message");
        let mut forced = v.tokenize("per-request message", false, true);
        forced.extend_from_slice(&p.sampling.reasoning_budget_end[0]);
        assert_eq!(p.sampling.reasoning_budget_forced, forced);

        // (3) the budget sampler (common/sampling.cpp:313-326): built inside
        // SamplingContext; the start tag arms it, budget 0 forces immediately
        // (reasoning-budget.cpp:87-92), and the forced token is the only one
        // that survives apply()
        let mut sampling = p.sampling.clone();
        sampling.generation_prompt = String::new();
        let mut smpl = llama::sampling::SamplingContext::new_with_vocab(
            v.n_tokens() as i32,
            Some(&v),
            sampling,
        );
        let rb = smpl.rbudget.as_mut().expect("the budget sampler was not built");
        assert_eq!(rb.get_state(), llama::sampling::ReasoningBudgetState::Idle);
        for &tok in &p.sampling.reasoning_budget_start {
            rb.accept(tok);
        }
        assert_eq!(rb.get_state(), llama::sampling::ReasoningBudgetState::Forcing);

        // apply() masks everything but the first forced token: a greedy chain
        // whose argmax differs still returns it (reasoning-budget.cpp:166-186)
        let first_forced = p.sampling.reasoning_budget_forced[0];
        let mut logits = vec![-3.0f32; v.n_tokens() as usize];
        logits[5] = 2.0; // the would-be argmax
        let id = smpl.sample(&logits);
        assert_eq!(id, first_forced);

        // the server default still applies when the request carries no budget
        let body = Json::parse(r#"{"messages":[{"role":"user","content":"hello"}]}"#).unwrap();
        let parsed = oaicompat_chat_params_parse(Some(&v), &body, &opt).unwrap();
        let lp = &parsed.llama_params;
        assert_eq!(lp.at("reasoning_budget_tokens").unwrap().get_i64().unwrap(), -1);
        assert_eq!(
            lp.at("reasoning_budget_message").unwrap().get_str().unwrap(),
            "server default"
        );
        let p = crate::api::eval_llama_cmpl_schema(lp, &base, &v).unwrap();
        assert_eq!(p.sampling.reasoning_budget_tokens, -1);
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
        let v = vocab().unwrap();
        let opt = ChatOptions {
            templates: &t,
            enable_thinking: false,
            reasoning_format: ReasoningFormat::None,
            reasoning_budget: -1,
            reasoning_budget_message: String::new(),
        };
        let parsed = oaicompat_chat_params_parse(
            Some(&v),
            &body(Some(weather_tools()), None, "What is the weather in Tokyo right now? Use the tool."),
            &opt,
        )
        .unwrap();
        // the differential autoparser's label for a generic template
        // (chat.cpp:863-864)
        assert_eq!(llama::chat_tools::chat_format_name(parsed.session.format()).unwrap(), "peg-native");
        let lp = &parsed.llama_params;
        // the tool section is rendered into the system prompt
        let prompt = lp.at("prompt").unwrap().get_str().unwrap();
        assert!(prompt.contains("# Tools"), "{prompt}");
        assert!(prompt.contains("\"name\": \"get_weather\""), "{prompt}");
        // the session owns the grammar side now (8b5436102): the keys are
        // erased from llama_params and applied via Task::apply_chat_session
        assert!(lp.at("grammar_lazy").is_none());
        assert!(lp.at("grammar").is_none());
        assert!(lp.at("grammar_triggers").is_none());
        assert!(lp.at("preserved_tokens").is_none());
        assert!(lp.at("generation_prompt").is_none());
        assert!(parsed.session.grammar_lazy());
        assert!(!parsed.session.grammar().is_empty());
        assert_eq!(parsed.session.generation_prompt(), "<|im_start|>assistant\n");

        // `task.apply_chat_session(session)` (server-task.h:261-271): the
        // sampling side lands on the task — the `<tool_call>` WORD trigger
        // carries its trailing newline, so it is multi-token and stays WORD
        let mut task = crate::engine::Task {
            id: 1,
            index: 0,
            kind: crate::engine::TaskKind::Completion,
            tokens: Vec::new(),
            params: crate::api::TaskParams::default(),
            prompt_text: String::new(),
            slot_action: None,
            decision_body: Json::Null,
            chat_session: None,
            tx: std::sync::mpsc::channel().0,
        };
        task.apply_chat_session(&parsed.session);
        let schema = &task.params;
        assert!(schema.grammar_lazy);
        assert_eq!(schema.grammar_triggers.len(), 1);
        assert_eq!(schema.grammar_triggers[0].at("type"), Some(&Json::Int(1)));
        assert!(schema.grammar_prefill);
        assert_eq!(schema.chat_format, "peg-native");
        assert_eq!(schema.generation_prompt, "<|im_start|>assistant\n");
        assert_eq!(task.chat_session.as_ref().unwrap().prompt(), prompt);

        // the session parses the reference's captured output
        let raw = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"unit\": \"fahrenheit\"}}\n</tool_call>";
        let mut session = parsed.session.clone();
        let msg = session.finish(&llama::chat_tools::ChatInput::from(raw)).unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "get_weather");
        assert_eq!(
            msg.tool_calls[0].arguments,
            r#"{"city": "Tokyo", "unit": "fahrenheit"}"#
        );

        // a partial parse of the same text yields the same tool call once the
        // closing tag is present, and diffs stream the name/arguments
        let mut session = parsed.session;
        let partial = session
            .feed(&llama::chat_tools::ChatInput::from(
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tok",
            ))
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
            reasoning_budget: -1,
            reasoning_budget_message: String::new(),
        };
        let parsed = oaicompat_chat_params_parse(
            None,
            &body(Some(weather_tools()), Some("required"), "What is the weather in Tokyo?"),
            &opt,
        )
        .unwrap();
        assert!(!parsed.session.grammar_lazy());
        assert!(!parsed.session.grammar().is_empty());
        assert!(parsed.session.grammar_triggers().is_empty());
        // `tool_choice: required` swaps the lazy grammar for an eager one
        // (chat-auto-parser-generator.cpp:78-80) — applied as a prefilled
        // tool_calls grammar through apply_chat_session
        let mut side = llama::chat_tools::ChatSamplingSide::default();
        parsed.session.apply_sampling(&mut side);
        assert!(!side.grammar.is_empty());
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
            reasoning_budget: -1,
            reasoning_budget_message: String::new(),
        };
        let parsed = oaicompat_chat_params_parse(
            None,
            &body(Some(weather_tools()), None, "hi"),
            &opt,
        )
        .unwrap();
        let mut session = parsed.session;
        let msg = session
            .finish(&llama::chat_tools::ChatInput::from("Hello! How can I help?"))
            .unwrap();
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, "Hello! How can I help?");
    }
}

#[cfg(test)]
// llama-rust-data-home: corpus paths resolve against $LLAMA_RUST_DATA_HOME (fallback: $HOME).
fn __llama_rust_data_path(rel: &str) -> &'static str {
    let root = std::env::var_os("LLAMA_RUST_DATA_HOME")
        .or_else(|| std::env::var_os("HOME").map(std::ffi::OsString::from))
        .unwrap_or_else(|| std::ffi::OsString::from("/var/lib/llama-rust"));
    let joined = std::path::Path::new(&root).join(rel);
    Box::leak(joined.to_string_lossy().into_owned().into_boxed_str())
}
