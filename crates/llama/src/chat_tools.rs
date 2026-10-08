//! chat_tools.rs — port of the reference's chat tool-calling machinery,
//! baseline bd4f514db1:
//!
//!   * `common/chat.h` + `common/chat.cpp` — the tools/messages model
//!     (oaicompat parse/serialize), the template→tools binding
//!     (`common_chat_template_direct_apply`), the jinja path of
//!     `common_chat_templates_apply` (chat.cpp:1225-1366), `common_chat_parse`
//!     (chat.cpp:1441-1523) and the streaming message diff
//!     (`common_chat_msg_diff::compute_diffs`, chat.cpp:267-333).
//!   * `common/chat-peg-parser.cpp` — the chat PEG builder tags + AST→msg
//!     mapper (`common_chat_peg_mapper`), `standard_json_tools` and friends,
//!     `tagged_peg_parser`.
//!   * `common/chat-auto-parser-generator.cpp` — `autoparser::peg_generator`
//!     (parser + lazy grammar generation for constrained tool-call output).
//!   * `common/chat-diff-analyzer.cpp` + `common/chat-auto-parser-helpers.cpp`
//!     — the differential template analysis driving the automatic parser.
//!
//! The PEG kit itself lives in [`crate::peg`]; the mini-jinja engine (with
//! `tools`/`tool_calls` values, `tojson`, and the `caps` analysis) lives in
//! [`crate::chat`].
//!
//! Not ported here (see PARITY.md): the specialized per-template handlers
//! `common/parsers/*.cpp` (chat.cpp:1090-1223 dispatches to them; this port's
//! `try_specialized_template` always declines and falls through to the
//! differential autoparser), the gemma4/minimax-m3 mappers
//! (chat-peg-parser.cpp:956-1232), and the reasoning-budget/ASR presets.
//!
//! C++ → Rust symbol map (line numbers = pinned tree):
//!   * `common_chat_tool_call`            — chat.h:27-35    ⇒ [`ChatToolCall`]
//!   * `common_chat_msg`                  — chat.h:80-128   ⇒ [`ChatMsg`]
//!   * `common_chat_msg::to_json_oaicompat` — chat.cpp:187-265 ⇒ [`ChatMsg::to_json_oaicompat`]
//!   * `common_chat_msg_diff::compute_diffs` — chat.cpp:267-333 ⇒ [`ChatMsgDiff::compute_diffs`]
//!   * `common_chat_msgs_parse_oaicompat`   — chat.cpp:373-472 ⇒ [`msgs_parse_oaicompat`]
//!   * `messages_inp_normalizer`         — chat.cpp:474-536 ⇒ [`MessagesInpNormalizer`]
//!   * `common_chat_tools_*`              — chat.cpp:558-620 ⇒ [`tools_to_json_oaicompat`] etc.
//!   * `common_chat_templates_init`       — chat.cpp:757-855 ⇒ [`ChatTemplates::init`]
//!   * `common_chat_template_direct_apply_impl` — chat.cpp:905-966 ⇒ [`template_direct_apply_impl`]
//!   * `common_chat_template_generation_prompt_impl` — chat.cpp:974-994 ⇒ [`template_generation_prompt_impl`]
//!   * workaround namespace — chat.cpp:1002-1078 ⇒ [`workaround`]
//!   * `common_chat_try_specialized_template` — chat.cpp:1090-1223 ⇒ [`try_specialized_template`]
//!   * `common_chat_templates_apply_jinja` — chat.cpp:1225-1366 ⇒ [`chat_templates_apply_jinja`]
//!   * `common_chat_peg_parse`           — chat.cpp:1447-1523 ⇒ [`chat_peg_parse`]
//!   * `common_chat_peg_builder`          — chat-peg-parser.h:60-180 ⇒ [`ChatPegBuilder`]
//!   * `common_chat_peg_mapper`           — chat-peg-parser.cpp:276-456 ⇒ [`ChatPegMapper`]
//!   * `standard_json_tools`              — chat-peg-parser.cpp:905-954 ⇒ [`ChatPegBuilder::standard_json_tools`]
//!   * `tagged_peg_parser`                — chat-peg-parser.cpp:188-227 ⇒ [`TaggedPegParser`]
//!   * `calculate_diff_split` etc.        — chat-auto-parser-helpers.cpp ⇒ this module
//!   * `autoparser::autoparser`           — chat-diff-analyzer.cpp:257-459 ⇒ [`Autoparser`]
//!   * `peg_generator::generate_parser`   — chat-auto-parser-generator.cpp:23-97 ⇒ [`generate_parser`]
//!
//! Deviations (documented):
//!   * `common_chat_params::parser` is a serialized arena string in C++; here
//!     [`ChatParams::parser`] keeps the serialized string and consumers reload
//!     it with [`crate::peg::PegArena::load`], exactly like the reference
//!     (chat.cpp:1359-1360).
//!   * `now` (datetime/date_string binding) renders in UTC via the pinned
//!     clock, the documented chat.rs deviation from `std::localtime`.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::chat::{caps_get, mini_jinja, JinjaCaps};
use crate::json_schema::{schema_from_json, Json, SchemaDocument, SchemaKind};
use crate::peg::{
    build_peg_parser, AstNode, ParseContext, ParseFlags, ParseResult, ParserId, PegArena,
    PegBuilder, PARSE_FLAG_DEBUG, PARSE_FLAG_LENIENT, PARSE_FLAG_NONE,
};

// ---------------------------------------------------------------------------
// basic types (chat.h)
// ---------------------------------------------------------------------------

/// `common_chat_tool_call` (chat.h:27-35)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// `common_chat_msg_content_part` (chat.h:37-49)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatMsgContentPart {
    pub ty: String,
    pub text: String,
}

/// `common_chat_msg` (chat.h:80-128)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    pub content_parts: Vec<ChatMsgContentPart>,
    pub tool_calls: Vec<ChatToolCall>,
    pub reasoning_content: String,
    pub tool_name: String,
    pub tool_call_id: String,
}

impl ChatMsg {
    /// `empty()` (chat.h:93-96)
    pub fn empty(&self) -> bool {
        self.content.is_empty()
            && self.content_parts.is_empty()
            && self.tool_calls.is_empty()
            && self.reasoning_content.is_empty()
            && self.tool_name.is_empty()
            && self.tool_call_id.is_empty()
    }

    /// `contains_media()` (chat.h:98-105)
    pub fn contains_media(&self) -> bool {
        self.content_parts
            .iter()
            .any(|part| part.ty == "media_marker")
    }

    /// `set_tool_call_ids` (chat.h:107-119)
    pub fn set_tool_call_ids(
        &mut self,
        ids_cache: &mut Vec<String>,
        gen_tool_call_id: impl Fn() -> String,
    ) {
        for i in 0..self.tool_calls.len() {
            if ids_cache.len() <= i {
                let mut id = self.tool_calls[i].id.clone();
                if id.is_empty() {
                    id = gen_tool_call_id();
                }
                ids_cache.push(id);
            }
            self.tool_calls[i].id = ids_cache[i].clone();
        }
    }

    /// `render_content(delimiter)` (chat.cpp:76-94)
    pub fn render_content(&self, delimiter: &str) -> Result<String, String> {
        if !self.content.is_empty() && !self.content_parts.is_empty() {
            return Err("Cannot specify both content and content_parts".to_string());
        }
        if !self.content.is_empty() {
            return Ok(self.content.clone());
        }
        let mut text = String::new();
        for part in &self.content_parts {
            if part.ty == "text" {
                if !text.is_empty() {
                    text += delimiter;
                }
                text += &part.text;
            }
        }
        Ok(text)
    }

    /// `to_json_oaicompat(concat_typed_text)` (chat.cpp:187-265)
    pub fn to_json_oaicompat(&self, concat_typed_text: bool) -> Result<Json, String> {
        if !self.content.is_empty() && !self.content_parts.is_empty() {
            return Err("Cannot specify both content and content_parts".to_string());
        }
        let mut jmsg = Json::Object(vec![("role".to_string(), Json::String(self.role.clone()))]);
        if !self.content.is_empty() {
            jmsg.set("content", Json::String(self.content.clone()));
        } else if !self.content_parts.is_empty() {
            if concat_typed_text || self.contains_media() {
                let mut text = String::new();
                let mut last_was_media_marker = false;
                // join parts with newline, no newline before/after media markers
                for part in &self.content_parts {
                    let add_new_line;
                    if part.ty == "text" {
                        add_new_line = !last_was_media_marker && !text.is_empty();
                        last_was_media_marker = false;
                    } else if part.ty == "media_marker" {
                        add_new_line = false;
                        last_was_media_marker = true;
                    } else {
                        continue;
                    }
                    if add_new_line {
                        text.push('\n');
                    }
                    text += &part.text;
                }
                jmsg.set("content", Json::String(text));
            } else {
                let parts = Json::Array(
                    self.content_parts
                        .iter()
                        .map(|part| {
                            Json::Object(vec![
                                ("type".to_string(), Json::String(part.ty.clone())),
                                ("text".to_string(), Json::String(part.text.clone())),
                            ])
                        })
                        .collect(),
                );
                jmsg.set("content", parts);
            }
        } else {
            jmsg.set("content", Json::String(String::new()));
        }
        if !self.reasoning_content.is_empty() {
            jmsg.set(
                "reasoning_content",
                Json::String(self.reasoning_content.clone()),
            );
        }
        if !self.tool_name.is_empty() {
            jmsg.set("name", Json::String(self.tool_name.clone()));
        }
        if !self.tool_call_id.is_empty() {
            jmsg.set("tool_call_id", Json::String(self.tool_call_id.clone()));
        }
        if !self.tool_calls.is_empty() {
            let mut jtool_calls = Json::Array(Vec::new());
            for tool_call in &self.tool_calls {
                let mut tc = Json::Object(vec![
                    ("type".to_string(), Json::String("function".to_string())),
                    (
                        "function".to_string(),
                        Json::Object(vec![
                            ("name".to_string(), Json::String(tool_call.name.clone())),
                            (
                                "arguments".to_string(),
                                Json::String(tool_call.arguments.clone()),
                            ),
                        ]),
                    ),
                ]);
                if !tool_call.id.is_empty() {
                    tc.set("id", Json::String(tool_call.id.clone()));
                }
                if let Json::Array(v) = &mut jtool_calls {
                    v.push(tc);
                }
            }
            jmsg.set("tool_calls", jtool_calls);
        }
        Ok(jmsg)
    }
}

/// `common_chat_tool` (chat.h:216-220); `parameters` is the JSON schema text
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters: String,
}

/// `common_chat_tool_choice` (chat.h:222-226)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatToolChoice {
    #[default]
    Auto,
    Required,
    None,
}

/// `common_chat_format` (chat.h:228-238). The PEG_GEMMA4/PEG_MINIMAX_M3
/// variants only occur via the unported specialized parsers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatFormat {
    #[default]
    ContentOnly,
    PegSimple,
    PegNative,
    PegGemma4,
    PegMinimaxM3,
}

/// `common_chat_format_name` (chat.cpp:857-872)
pub fn chat_format_name(format: ChatFormat) -> Result<&'static str, String> {
    Ok(match format {
        ChatFormat::ContentOnly => "Content-only",
        ChatFormat::PegSimple => "peg-simple",
        ChatFormat::PegNative => "peg-native",
        ChatFormat::PegGemma4 => "peg-gemma4",
        ChatFormat::PegMinimaxM3 => "peg-minimax-m3",
    })
}

/// `common_reasoning_format` (common.h)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningFormat {
    #[default]
    None,
    Auto,
    Deepseek,
    DeepseekLegacy,
}

/// `common_reasoning_format_name` (chat.cpp:874-887)
pub fn reasoning_format_name(format: ReasoningFormat) -> Result<&'static str, String> {
    Ok(match format {
        ReasoningFormat::None => "none",
        ReasoningFormat::Auto => "auto",
        ReasoningFormat::Deepseek => "deepseek",
        ReasoningFormat::DeepseekLegacy => "deepseek-legacy",
    })
}

/// `common_reasoning_format_from_name` (chat.cpp:889-903)
pub fn reasoning_format_from_name(format: &str) -> Result<ReasoningFormat, String> {
    match format {
        "none" => Ok(ReasoningFormat::None),
        "auto" => Ok(ReasoningFormat::Auto),
        "deepseek" => Ok(ReasoningFormat::Deepseek),
        "deepseek-legacy" => Ok(ReasoningFormat::DeepseekLegacy),
        other => Err(format!("Unknown reasoning format: {other}")),
    }
}

/// `common_chat_continuation` (chat.h:242-247)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatContinuation {
    #[default]
    None,
    Auto,
    Reasoning,
    Content,
}

/// `common_chat_continuation_parse` (chat.cpp:622-636)
pub fn chat_continuation_parse(value: &Json) -> ChatContinuation {
    match value {
        Json::Bool(true) => ChatContinuation::Auto,
        Json::String(s) if s == "reasoning_content" => ChatContinuation::Reasoning,
        Json::String(s) if s == "content" => ChatContinuation::Content,
        _ => ChatContinuation::None,
    }
}

/// `common_grammar_trigger_type` (common.h:143-148)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarTriggerType {
    Token,
    Word,
    Pattern,
    PatternFull,
}

/// `common_grammar_trigger` (common.h:150-155) — `word` carries the trigger
/// value for every type (a word or a pattern, per `ty`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarTrigger {
    pub ty: GrammarTriggerType,
    pub word: String,
}

impl GrammarTrigger {
    /// `COMMON_GRAMMAR_TRIGGER_TYPE_WORD` form (the only form the autoparser
    /// emits; the specialized parsers under chat_parsers use the others)
    pub fn word(word: &str) -> Self {
        GrammarTrigger {
            ty: GrammarTriggerType::Word,
            word: word.to_string(),
        }
    }

    /// `COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN` form (gpt-oss, functionary v3.2,
    /// muse-glimmer)
    pub fn pattern(pattern: &str) -> Self {
        GrammarTrigger {
            ty: GrammarTriggerType::Pattern,
            word: pattern.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// string_diff + compute_diffs (chat.cpp:45-70, 267-333)
// ---------------------------------------------------------------------------

fn string_starts_with(haystack: &str, needle: &str) -> bool {
    haystack.as_bytes().starts_with(needle.as_bytes())
}

pub(crate) fn string_ends_with(haystack: &str, needle: &str) -> bool {
    haystack.as_bytes().ends_with(needle.as_bytes())
}

/// `string_diff` (chat.cpp:57-70)
fn string_diff(last: &str, current: &str) -> Result<String, String> {
    if last.is_empty() {
        return Ok(current.to_string());
    }
    if !string_starts_with(current, last) {
        if string_starts_with(last, current) {
            // last generation ended on a partial stop word, current on the
            // erased stop word
            return Ok(String::new());
        }
        return Err(format!(
            "Invalid diff: '{last}' not found at start of '{current}'"
        ));
    }
    Ok(current[last.len()..].to_string())
}

/// `common_chat_msg_diff` (chat.h:130-143)
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMsgDiff {
    pub reasoning_content_delta: String,
    pub content_delta: String,
    pub tool_call_index: usize,
    pub tool_call_delta: ChatToolCall,
}

impl Default for ChatMsgDiff {
    fn default() -> Self {
        ChatMsgDiff {
            reasoning_content_delta: String::new(),
            content_delta: String::new(),
            tool_call_index: ChatMsgDiff::NPOS,
            tool_call_delta: ChatToolCall::default(),
        }
    }
}

impl ChatMsgDiff {
    pub const NPOS: usize = usize::MAX;

    /// `compute_diffs` (chat.cpp:267-333)
    pub fn compute_diffs(msg_prv: &ChatMsg, msg_new: &ChatMsg) -> Result<Vec<ChatMsgDiff>, String> {
        let mut diffs: Vec<ChatMsgDiff> = Vec::new();

        if msg_prv.reasoning_content != msg_new.reasoning_content {
            let mut diff = ChatMsgDiff::default();
            diff.reasoning_content_delta =
                string_diff(&msg_prv.reasoning_content, &msg_new.reasoning_content)?;
            diffs.push(diff);
        }
        if msg_prv.content != msg_new.content {
            let mut diff = ChatMsgDiff::default();
            diff.content_delta = string_diff(&msg_prv.content, &msg_new.content)?;
            diffs.push(diff);
        }

        if msg_new.tool_calls.len() < msg_prv.tool_calls.len() {
            let mut err = String::from("Invalid diff: now finding less tool calls!\n");
            err += &format!("  Previous ({}):\n", msg_prv.tool_calls.len());
            for tc in &msg_prv.tool_calls {
                err += &format!("    - name: '{}', args: '{}'\n", tc.name, tc.arguments);
            }
            err += &format!("  Current ({}):\n", msg_new.tool_calls.len());
            for tc in &msg_new.tool_calls {
                err += &format!("    - name: '{}', args: '{}'\n", tc.name, tc.arguments);
            }
            err += &format!("  Current msg text content:\n{}\n", msg_new.content);
            return Err(err);
        }

        if !msg_prv.tool_calls.is_empty() {
            let idx = msg_prv.tool_calls.len() - 1;
            let pref = &msg_prv.tool_calls[idx];
            let newf = &msg_new.tool_calls[idx];
            // Allow tool name to change during incremental parsing:
            // empty → non-empty, or prefix → longer string
            if pref.name != newf.name && !pref.name.is_empty() && !newf.name.is_empty() {
                let is_prefix = newf.name.starts_with(pref.name.as_str());
                if !is_prefix {
                    return Err("Invalid diff: tool call mismatch!".to_string());
                }
            }
            let args_diff = string_diff(&pref.arguments, &newf.arguments)?;
            if !args_diff.is_empty() || pref.id != newf.id || pref.name != newf.name {
                let mut diff = ChatMsgDiff::default();
                diff.tool_call_index = idx;
                if pref.id != newf.id || pref.name != newf.name {
                    diff.tool_call_delta.id = newf.id.clone();
                    diff.tool_call_delta.name = newf.name.clone();
                }
                diff.tool_call_delta.arguments = args_diff;
                diffs.push(diff);
            }
        }
        for idx in msg_prv.tool_calls.len()..msg_new.tool_calls.len() {
            let mut diff = ChatMsgDiff::default();
            diff.tool_call_index = idx;
            diff.tool_call_delta = msg_new.tool_calls[idx].clone();
            diffs.push(diff);
        }

        Ok(diffs)
    }
}

// ---------------------------------------------------------------------------
// oaicompat parsing (chat.cpp:345-620)
// ---------------------------------------------------------------------------

/// `common_chat_tool_choice_parse_oaicompat` (chat.cpp:345-356)
pub fn tool_choice_parse_oaicompat(tool_choice: &str) -> Result<ChatToolChoice, String> {
    match tool_choice {
        "auto" => Ok(ChatToolChoice::Auto),
        "none" => Ok(ChatToolChoice::None),
        "required" => Ok(ChatToolChoice::Required),
        other => Err(format!("Invalid tool_choice: {other}")),
    }
}

/// `common_chat_msgs_parse_oaicompat` (chat.cpp:373-472)
pub fn msgs_parse_oaicompat(messages: &Json) -> Result<Vec<ChatMsg>, String> {
    let inner = || -> Result<Vec<ChatMsg>, String> {
        if !messages.is_array() {
            return Err(format!(
                "Expected 'messages' to be an array, got {}",
                messages.dump()
            ));
        }
        let mut msgs = Vec::new();
        for message in messages.iter() {
            if !message.is_object() {
                return Err(format!(
                    "Expected 'message' to be an object, got {}",
                    message.dump()
                ));
            }
            let mut msg = ChatMsg::default();
            let Some(role) = message.at("role") else {
                return Err(format!("Missing 'role' in message: {}", message.dump()));
            };
            msg.role = role.get_str()?.to_string();

            let has_content = message.at("content").is_some();
            let has_tool_calls = message.at("tool_calls").is_some();
            if let Some(content) = message.at("content") {
                if content.is_string() {
                    msg.content = content.get_str()?.to_string();
                } else if content.is_array() {
                    for part in content.iter() {
                        let Some(ty) = part.at("type") else {
                            return Err(format!("Missing content part type: {}", part.dump()));
                        };
                        let ty = ty.get_str()?;
                        if ty != "text" && ty != "media_marker" {
                            return Err(format!("Unsupported content part type: {ty}"));
                        }
                        msg.content_parts.push(ChatMsgContentPart {
                            ty: ty.to_string(),
                            text: part
                                .at("text")
                                .ok_or_else(|| "missing text".to_string())?
                                .get_str()?
                                .to_string(),
                        });
                    }
                } else if !content.is_null() {
                    return Err(format!(
                        "Invalid 'content' type: expected string or array, got {} (ref: https://github.com/ggml-org/llama.cpp/issues/8367)",
                        content.dump()
                    ));
                }
            }
            if has_tool_calls {
                let calls = message.at("tool_calls").ok_or("missing tool_calls")?;
                for tool_call in calls.iter() {
                    let mut tc = ChatToolCall::default();
                    let Some(ty) = tool_call.at("type") else {
                        return Err(format!("Missing tool call type: {}", tool_call.dump()));
                    };
                    if ty.get_str()? != "function" {
                        return Err(format!("Unsupported tool call type: {}", tool_call.dump()));
                    }
                    let fc = tool_call.at("function").ok_or_else(|| {
                        format!("Missing tool call function: {}", tool_call.dump())
                    })?;
                    let Some(name) = fc.at("name") else {
                        return Err(format!("Missing tool call name: {}", tool_call.dump()));
                    };
                    tc.name = name.get_str()?.to_string();
                    let args = fc.at("arguments").ok_or("missing arguments")?;
                    if args.is_string() {
                        tc.arguments = args.get_str()?.to_string();
                    } else {
                        tc.arguments = args.dump();
                    }
                    if let Some(id) = tool_call.at("id") {
                        tc.id = id.get_str()?.to_string();
                    }
                    msg.tool_calls.push(tc);
                }
            }
            if !has_content && !has_tool_calls {
                return Err(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & https://github.com/ggml-org/llama.cpp/issues/12279)"
                        .to_string(),
                );
            }
            if let Some(rc) = message.at("reasoning_content") {
                msg.reasoning_content = rc.get_str()?.to_string();
            }
            if let Some(n) = message.at("name") {
                msg.tool_name = n.get_str()?.to_string();
            }
            if let Some(id) = message.at("tool_call_id") {
                msg.tool_call_id = id.get_str()?.to_string();
            }
            msgs.push(msg);
        }
        Ok(msgs)
    }();
    inner.map_err(|e| format!("Failed to parse messages: {e}"))
}

/// `messages_inp_normalizer` (chat.cpp:474-536)
pub struct MessagesInpNormalizer<'a> {
    pub caps: &'a JinjaCaps,
}

impl<'a> MessagesInpNormalizer<'a> {
    pub fn new(caps: &'a JinjaCaps) -> Self {
        MessagesInpNormalizer { caps }
    }

    /// `normalize` (chat.cpp:483-508): if the template only supports string
    /// content, convert arrays to strings; if only typed, strings to arrays.
    pub fn normalize(&self, messages: &Json) -> Json {
        let only_string = self.caps.supports_string_content && !self.caps.supports_typed_content;
        let only_typed = !self.caps.supports_string_content && self.caps.supports_typed_content;
        if (!only_string && !only_typed) || !messages.is_array() {
            return messages.clone();
        }
        let mut normalized = Json::Array(Vec::new());
        for msg in messages.iter() {
            let mut copy = msg.clone();
            if let Some(it) = copy.at("content").cloned() {
                if only_typed && it.is_string() {
                    copy.set(
                        "content",
                        Json::Array(vec![Json::Object(vec![
                            ("type".to_string(), Json::String("text".to_string())),
                            ("text".to_string(), it.clone()),
                        ])]),
                    );
                } else if only_string && it.is_array() {
                    copy.set("content", Json::String(Self::concat_content_parts(&it)));
                }
            }
            if let Json::Array(v) = &mut normalized {
                v.push(copy);
            }
        }
        normalized
    }

    /// `concat_content_parts` (chat.cpp:511-535)
    fn concat_content_parts(parts: &Json) -> String {
        let mut text = String::new();
        let mut last_was_media_marker = false;
        for part in parts.iter() {
            let ty = part.at("type").and_then(|v| v.get_str().ok()).unwrap_or("");
            let add_new_line;
            if ty == "text" {
                add_new_line = !last_was_media_marker && !text.is_empty();
                last_was_media_marker = false;
            } else if ty == "media_marker" {
                add_new_line = false;
                last_was_media_marker = true;
            } else {
                continue;
            }
            if add_new_line {
                text.push('\n');
            }
            text += part.at("text").and_then(|v| v.get_str().ok()).unwrap_or("");
        }
        text
    }
}

/// `render_message_to_json` (chat.cpp:538-548)
fn render_message_to_json(msgs: &[ChatMsg], caps: &JinjaCaps) -> Json {
    let mut messages = Json::Array(Vec::new());
    for msg in msgs {
        if let Json::Array(v) = &mut messages {
            v.push(msg.to_json_oaicompat(false).unwrap_or(Json::Null));
        }
    }
    MessagesInpNormalizer::new(caps).normalize(&messages)
}

/// `common_chat_msgs_to_json_oaicompat` (chat.cpp:551-556, DEPRECATED: tests)
pub fn msgs_to_json_oaicompat(msgs: &[ChatMsg], concat_typed_text: bool) -> Json {
    let mut c = JinjaCaps::default();
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    render_message_to_json(msgs, &c)
}

/// `common_chat_tools_to_json_oaicompat` (chat.cpp:558-575)
pub fn tools_to_json_oaicompat(tools: &[ChatTool]) -> Json {
    if tools.is_empty() {
        return Json::Null;
    }
    let mut result = Json::Array(Vec::new());
    for tool in tools {
        let entry = Json::Object(vec![
            ("type".to_string(), Json::String("function".to_string())),
            (
                "function".to_string(),
                Json::Object(vec![
                    ("name".to_string(), Json::String(tool.name.clone())),
                    (
                        "description".to_string(),
                        Json::String(tool.description.clone()),
                    ),
                    (
                        "parameters".to_string(),
                        Json::parse(&tool.parameters).unwrap_or(Json::Null),
                    ),
                ]),
            ),
        ]);
        if let Json::Array(v) = &mut result {
            v.push(entry);
        }
    }
    result
}

/// `common_chat_tool_parameters` (chat.cpp:577-585): the parameters schema of a
/// function tool; a tool without parameters takes zero arguments.
pub fn tool_parameters(function: &Json) -> Json {
    if let Some(params) = function.at("parameters") {
        if !params.is_null() && !(params.is_object() && params.empty()) {
            return params.clone();
        }
    }
    Json::Object(vec![
        ("type".to_string(), Json::String("object".to_string())),
        ("properties".to_string(), Json::Object(Vec::new())),
    ])
}

/// `common_chat_tools_parse_oaicompat` (chat.cpp:587-620)
pub fn tools_parse_oaicompat(tools: &Json) -> Result<Vec<ChatTool>, String> {
    let inner = || -> Result<Vec<ChatTool>, String> {
        let mut result = Vec::new();
        if !tools.is_null() {
            if !tools.is_array() {
                return Err(format!(
                    "Expected 'tools' to be an array, got {}",
                    tools.dump()
                ));
            }
            for tool in tools.iter() {
                let Some(ty) = tool.at("type") else {
                    return Err(format!("Missing tool type: {}", tool.dump()));
                };
                if !ty.is_string() || ty.get_str()? != "function" {
                    return Err(format!("Unsupported tool type: {}", tool.dump()));
                }
                let function = tool
                    .at("function")
                    .ok_or_else(|| format!("Missing tool function: {}", tool.dump()))?;
                result.push(ChatTool {
                    name: function
                        .at("name")
                        .ok_or_else(|| "missing name".to_string())?
                        .get_str()?
                        .to_string(),
                    description: function
                        .at("description")
                        .and_then(|v| v.get_str().ok())
                        .unwrap_or("")
                        .to_string(),
                    parameters: function
                        .at("parameters")
                        .cloned()
                        .unwrap_or(Json::Object(Vec::new()))
                        .dump(),
                });
            }
        }
        Ok(result)
    }();
    inner.map_err(|e| format!("Failed to parse tools: {e}; tools = {}", tools.dump()))
}

// ---------------------------------------------------------------------------
// workaround namespace (chat.cpp:1002-1078)
// ---------------------------------------------------------------------------

pub mod workaround {
    use super::Json;

    /// `map_developer_role_to_system` (chat.cpp:1004-1012)
    pub fn map_developer_role_to_system(messages: &mut Json) {
        if let Json::Array(items) = messages {
            for message in items.iter_mut() {
                if let Some(role) = message.at("role") {
                    if role.is_string() && role.get_str() == Ok("developer") {
                        message.set("role", Json::String("system".to_string()));
                    }
                }
            }
        }
    }

    /// `system_message_not_supported` (chat.cpp:1016-1030)
    pub fn system_message_not_supported(messages: &mut Json) {
        if let Json::Array(items) = messages {
            if !items.is_empty()
                && items[0].at("role").and_then(|r| r.get_str().ok()) == Some("system")
            {
                if items.len() > 1 {
                    let first_content = items[0]
                        .at("content")
                        .and_then(|c| c.get_str().ok())
                        .unwrap_or("")
                        .to_string();
                    let second_content = items[1]
                        .at("content")
                        .and_then(|c| c.get_str().ok())
                        .unwrap_or("")
                        .to_string();
                    items[1].set(
                        "content",
                        Json::String(format!("{first_content}\n{second_content}")),
                    );
                    items.remove(0);
                } else {
                    items.remove(0);
                }
            }
        }
    }

    /// `requires_non_null_content` (chat.cpp:1032-1039)
    pub fn requires_non_null_content(messages: &mut Json) {
        if let Json::Array(items) = messages {
            for message in items.iter_mut() {
                if message.at("tool_calls").is_some() && message.at("content").is_none() {
                    message.set("content", Json::String(String::new()));
                }
            }
        }
    }

    /// `func_args_not_string` (chat.cpp:1041-1059)
    pub fn func_args_not_string(messages: &mut Json) -> Result<(), String> {
        if let Json::Array(items) = messages {
            for message in items.iter_mut() {
                if let Some(Json::Array(calls)) = message.at("tool_calls").cloned() {
                    let mut new_calls = calls;
                    for tool_call in new_calls.iter_mut() {
                        let mut args_slot: Option<&mut Json> = None;
                        if let Some(f) = tool_call.at_mut_object_field("function") {
                            args_slot = f.at_mut_object_field("arguments");
                        }
                        if let Some(args) = args_slot {
                            if args.is_string() {
                                *args = Json::parse(args.get_str()?).map_err(|e| {
                                    format!("Failed to parse tool call arguments as JSON: {e}")
                                })?;
                            }
                        }
                    }
                    message.set("tool_calls", Json::Array(new_calls));
                }
            }
        }
        Ok(())
    }

    /// `trim_all_content` (chat.cpp:1061-1076) — on ChatMsg, not json
    pub fn trim_all_content(messages: &mut [super::ChatMsg]) {
        for message in messages.iter_mut() {
            message.content = super::trim_whitespace(&message.content).to_string();
            message.reasoning_content =
                super::trim_whitespace(&message.reasoning_content).to_string();
            for part in message.content_parts.iter_mut() {
                if part.ty == "text" {
                    part.text = super::trim_whitespace(&part.text).to_string();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// diff helpers (chat-auto-parser-helpers.cpp)
// ---------------------------------------------------------------------------

/// `trim_whitespace` (helpers.cpp:8-27) — C isspace
pub fn trim_whitespace(s: &str) -> &str {
    crate::chat::trim_pub(s)
}

/// `trim_leading_whitespace` (helpers.cpp:29-37)
pub fn trim_leading_whitespace(s: &str) -> &str {
    let b = s.as_bytes();
    let mut start = 0;
    while start < b.len() && crate::chat::c_isspace_pub(b[start]) {
        start += 1;
    }
    &s[start..]
}

/// `trim_trailing_whitespace` (helpers.cpp:39-64)
pub fn trim_trailing_whitespace(s: &str) -> &str {
    let b = s.as_bytes();
    if b.is_empty() {
        return "";
    }
    let mut end = b.len() - 1;
    while end > 0 && crate::chat::c_isspace_pub(b[end]) {
        end -= 1;
    }
    // If first char is also whitespace, return empty string
    if end == 0 && crate::chat::c_isspace_pub(b[0]) {
        return "";
    }
    &s[..=end]
}

/// `trim_trailing_newlines` (helpers.cpp:59-64)
pub fn trim_trailing_newlines(s: &str) -> &str {
    let b = s.as_bytes();
    let mut end = b.len();
    while end > 0 && b[end - 1] == b'\n' {
        end -= 1;
    }
    &s[..end]
}

/// `segment_type`/`segment` (chat-auto-parser.h:427-453)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentType {
    Text,
    Marker,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub ty: SegmentType,
    pub value: String,
}

/// `diff_split` (chat-auto-parser.h:30-39)
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiffSplit {
    pub prefix: String,
    pub suffix: String,
    pub left: String,
    pub right: String,
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    // helpers.cpp:66-73
    let (l, r) = (left.as_bytes(), right.as_bytes());
    let min_len = l.len().min(r.len());
    let mut prefix_len = 0;
    while prefix_len < min_len && l[prefix_len] == r[prefix_len] {
        prefix_len += 1;
    }
    prefix_len
}

fn common_suffix_len(left: &str, right: &str) -> usize {
    // helpers.cpp:75-82
    let (l, r) = (left.as_bytes(), right.as_bytes());
    let min_len = l.len().min(r.len());
    let mut suffix_len = 0;
    while suffix_len < min_len && l[l.len() - 1 - suffix_len] == r[r.len() - 1 - suffix_len] {
        suffix_len += 1;
    }
    suffix_len
}

/// `calculate_diff_split` (helpers.cpp:84-204)
pub fn calculate_diff_split(left: &str, right: &str) -> DiffSplit {
    let mut result = DiffSplit::default();

    let left_seg = segmentize_markers(left);
    let right_seg = segmentize_markers(right);

    if left_seg.is_empty() {
        result.right = right.to_string();
        return result;
    }
    if right_seg.is_empty() {
        result.left = left.to_string();
        return result;
    }

    // The C++ walks with bidirectional iterators (left_start/left_end etc.);
    // emulate with inclusive indices.
    let mut ls = 0usize;
    let mut le = left_seg.len() - 1;
    let mut rs = 0usize;
    let mut re = right_seg.len() - 1;

    let mut left_fully_consumed = false;
    let mut right_fully_consumed = false;

    while ls != le && rs != re {
        let mut advanced = false;
        if left_seg[ls] == right_seg[rs] {
            result.prefix += &left_seg[ls].value;
            ls += 1;
            rs += 1;
            advanced = true;
        }
        if left_seg[le] == right_seg[re] {
            result.suffix = format!("{}{}", left_seg[le].value, result.suffix);
            if ls != le {
                le -= 1;
            } else {
                left_fully_consumed = true;
            }
            if rs != re {
                re -= 1;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if !advanced {
            break;
        }
    }

    if ls == le && rs != re {
        if left_seg[ls] == right_seg[re] {
            result.suffix = format!("{}{}", right_seg[re].value, result.suffix);
            re = re.saturating_sub(1);
            left_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix += &right_seg[rs].value;
            rs += 1;
            left_fully_consumed = true;
        }
    } else if rs == re && ls != le {
        if left_seg[le] == right_seg[rs] {
            result.suffix = format!("{}{}", left_seg[le].value, result.suffix);
            le = le.saturating_sub(1);
            right_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix += &left_seg[ls].value;
            ls += 1;
            right_fully_consumed = true;
        }
    } else if ls == le
        && rs == re
        && left_seg[ls] == right_seg[rs]
        && left_seg[ls].ty == SegmentType::Marker
    {
        result.prefix += &right_seg[rs].value;
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    // C++: std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, …)
    // — [start, end+1) when not fully consumed (the ++ makes it inclusive of
    // the end segment), [start, end) when fully consumed (the end segment was
    // already emitted as a matched suffix)
    let remainder_of =
        |segs: &[Segment], start: usize, end: usize, fully_consumed: bool| -> String {
            let stop = if fully_consumed { end } else { end + 1 };
            segs[start..stop.max(start)]
                .iter()
                .map(|s| s.value.as_str())
                .collect::<String>()
        };
    let remainder_left = remainder_of(&left_seg, ls, le, left_fully_consumed);
    let remainder_right = remainder_of(&right_seg, rs, re, right_fully_consumed);

    let can_have_text_suffix =
        left_seg[le].ty == SegmentType::Text && right_seg[re].ty == SegmentType::Text;
    let can_have_text_prefix =
        right_seg[rs].ty == SegmentType::Text && left_seg[ls].ty == SegmentType::Text;

    let suffix_len = if can_have_text_suffix {
        common_suffix_len(&remainder_left, &remainder_right)
    } else {
        0
    };
    // avoid overlaps between prefix and suffix
    let prefix_len = if can_have_text_prefix {
        let l = &remainder_left[..remainder_left.len() - suffix_len];
        let r = &remainder_right[..remainder_right.len() - suffix_len];
        common_prefix_len(l, r)
    } else {
        0
    };

    result.prefix += &remainder_left[..prefix_len];
    result.suffix = format!(
        "{}{}",
        &remainder_left[remainder_left.len() - suffix_len..],
        result.suffix
    );
    result.left = remainder_left[prefix_len..remainder_left.len() - suffix_len].to_string();
    result.right = remainder_right[prefix_len..remainder_right.len() - suffix_len].to_string();

    if result.left.is_empty() && result.right.is_empty() {
        // degenerate case, no diff — pick prefix = all as representation
        result.prefix = left.to_string();
        result.suffix = String::new();
    }

    // helpers.cpp:189-201: when left is fully shared with right, the
    // simultaneous prefix/suffix matching can rotate the diff; enforce that
    // left is a prefix of right directly.
    if result.left.is_empty()
        && !result.right.is_empty()
        && left.len() <= right.len()
        && &right[..left.len()] == left
    {
        result.prefix = left.to_string();
        result.suffix = String::new();
        result.right = right[left.len()..].to_string();
    }

    result
}

/// `until_common_prefix` (helpers.cpp:207-231)
pub fn until_common_prefix(full: &str, left: &str, right: &str) -> String {
    let common_prefix_len = common_prefix_len(left, right);
    if common_prefix_len == 0 {
        return String::new();
    }
    let common_prefix = &left[..common_prefix_len];
    match full.find(common_prefix) {
        Some(pos) => full[..pos].to_string(),
        None => String::new(),
    }
}

/// `after_common_suffix` (helpers.cpp:234-261)
pub fn after_common_suffix(full: &str, left: &str, right: &str) -> String {
    let common_suffix_len = common_suffix_len(left, right);
    if common_suffix_len == 0 {
        return String::new();
    }
    let common_suffix = &left[left.len() - common_suffix_len..];
    match full.rfind(common_suffix) {
        Some(pos) => full[pos + common_suffix.len()..].to_string(),
        None => String::new(),
    }
}

/// `segmentize_markers` (helpers.cpp:266-296)
pub fn segmentize_markers(text: &str) -> Vec<Segment> {
    let mut retval = Vec::new();
    let mut in_marker = false;
    let mut marker_opener = '\0';

    let is_marker_opener = |c: char| c == '<' || c == '[';
    let is_marker_closer = |op: char, c: char| (op == '<' && c == '>') || (op == '[' && c == ']');

    let mut last_border = 0usize;
    let mut byte_pos = 0usize;
    for c in text.chars() {
        let cpos = byte_pos;
        byte_pos += c.len_utf8();
        if !in_marker && is_marker_opener(c) {
            if last_border < cpos {
                retval.push(Segment {
                    ty: SegmentType::Text,
                    value: text[last_border..cpos].to_string(),
                });
            }
            last_border = cpos;
            in_marker = true;
            marker_opener = c;
        } else if in_marker && is_marker_closer(marker_opener, c) {
            retval.push(Segment {
                ty: SegmentType::Marker,
                value: text[last_border..byte_pos].to_string(),
            });
            last_border = byte_pos;
            in_marker = false;
            marker_opener = '\0';
        }
    }
    if last_border < text.len() {
        retval.push(Segment {
            ty: SegmentType::Text,
            value: text[last_border..].to_string(),
        });
    }
    retval
}

/// `prune_whitespace_segments` (helpers.cpp:298-306)
pub fn prune_whitespace_segments(segments: &[Segment]) -> Vec<Segment> {
    segments
        .iter()
        .filter(|seg| !trim_whitespace(&seg.value).is_empty())
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// tagged peg parser (chat-peg-parser.cpp:188-227)
// ---------------------------------------------------------------------------

/// `tagged_parse_result` (chat-peg-parser.h:196-199)
pub struct TaggedParseResult {
    pub result: ParseResult,
    pub tags: BTreeMap<String, String>,
}

/// `tagged_peg_parser` (chat-peg-parser.h:201-217)
pub struct TaggedPegParser {
    pub arena: PegArena,
    pub flags: ParseFlags,
}

impl TaggedPegParser {
    /// `parse_and_extract` (chat-peg-parser.cpp:196-204)
    pub fn parse_and_extract(&self, input: &str, extra_flags: ParseFlags) -> TaggedParseResult {
        let mut ctx = ParseContext::new(input, self.flags | extra_flags);
        let parse_result = self
            .arena
            .parse(&mut ctx, 0)
            .unwrap_or_else(|e| panic!("{e}"));

        // tag_based_peg_mapper::from_ast (chat-peg-parser.cpp:188-194)
        let mut tags: BTreeMap<String, String> = BTreeMap::new();
        let input_bytes = ctx.input.as_bytes().to_vec();
        let mut visitor = |node: &AstNode| {
            if !node.tag.is_empty() {
                tags.insert(
                    node.tag.clone(),
                    String::from_utf8_lossy(&input_bytes[node.start..node.end]).into_owned(),
                );
            }
        };
        ctx.ast.visit_result(&parse_result, &mut visitor);

        TaggedParseResult {
            result: parse_result,
            tags,
        }
    }

    /// `parse_anywhere_and_extract` (chat-peg-parser.cpp:206-220)
    pub fn parse_anywhere_and_extract(&self, input: &str) -> TaggedParseResult {
        if input.is_empty() {
            return self.parse_and_extract(input, PARSE_FLAG_NONE);
        }
        let n = input.len();
        for i in 0..n {
            let mut ctx = ParseContext::new(input, self.flags);
            let parse_result = self
                .arena
                .parse(&mut ctx, i)
                .unwrap_or_else(|e| panic!("{e}"));
            if parse_result.success() || i == n - 1 {
                let mut tags: BTreeMap<String, String> = BTreeMap::new();
                let input_bytes = ctx.input.as_bytes().to_vec();
                let mut visitor = |node: &AstNode| {
                    if !node.tag.is_empty() {
                        tags.insert(
                            node.tag.clone(),
                            String::from_utf8_lossy(&input_bytes[node.start..node.end])
                                .into_owned(),
                        );
                    }
                };
                ctx.ast.visit_result(&parse_result, &mut visitor);
                return TaggedParseResult {
                    result: parse_result,
                    tags,
                };
            }
        }
        unreachable!("parse_anywhere_and_extract: should not happen")
    }
}

/// `build_tagged_peg_parser` (chat-peg-parser.cpp:222-227)
pub fn build_tagged_peg_parser<F>(f: F) -> TaggedPegParser
where
    F: FnOnce(&mut PegBuilder) -> ParserId,
{
    let arena = build_peg_parser(f).expect("build_tagged_peg_parser");
    TaggedPegParser {
        arena,
        flags: PARSE_FLAG_NONE,
    }
}

// ---------------------------------------------------------------------------
// chat peg builder (chat-peg-parser.h:60-180)
// ---------------------------------------------------------------------------

/// Tag constants (`common_chat_peg_builder`, chat-peg-parser.h:63-79)
pub mod chat_tag {
    pub const REASONING_BLOCK: &str = "reasoning-block";
    pub const REASONING: &str = "reasoning";
    pub const CONTENT: &str = "content";
    pub const TOOL: &str = "tool";
    pub const TOOL_OPEN: &str = "tool-open";
    pub const TOOL_CLOSE: &str = "tool-close";
    pub const TOOL_ID: &str = "tool-id";
    pub const TOOL_NAME: &str = "tool-name";
    pub const TOOL_ARGS: &str = "tool-args";
    pub const TOOL_ARG: &str = "tool-arg";
    pub const TOOL_ARG_OPEN: &str = "tool-arg-open";
    pub const TOOL_ARG_CLOSE: &str = "tool-arg-close";
    pub const TOOL_ARG_NAME: &str = "tool-arg-name";
    pub const TOOL_ARG_VALUE: &str = "tool-arg-value";
    pub const TOOL_ARG_STRING_VALUE: &str = "tool-arg-string-value";
}

/// `COMMON_CHAT_MAX_PERMUTE` (chat-peg-parser.h:58)
pub const COMMON_CHAT_MAX_PERMUTE: usize = 6;

/// `common_chat_peg_builder` (chat-peg-parser.h:60-180) — a composition
/// wrapper over [`PegBuilder`] adding the chat tag helpers.
pub struct ChatPegBuilder {
    pub p: PegBuilder,
}

impl Default for ChatPegBuilder {
    fn default() -> Self {
        ChatPegBuilder {
            p: PegBuilder::default(),
        }
    }
}

impl ChatPegBuilder {
    // ---- low-level tag methods (chat-peg-parser.h:82-107) -------------------

    pub fn reasoning_block(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::REASONING_BLOCK, p)
    }

    pub fn reasoning(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::REASONING, p)
    }

    pub fn content(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::CONTENT, p)
    }

    pub fn tool(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL, p)
    }

    pub fn tool_open(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_OPEN, p);
        self.p.atomic(t)
    }

    pub fn tool_close(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_CLOSE, p);
        self.p.atomic(t)
    }

    pub fn tool_id(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_ID, p);
        self.p.atomic(t)
    }

    pub fn tool_name(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_NAME, p);
        self.p.atomic(t)
    }

    pub fn tool_args(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL_ARGS, p)
    }

    pub fn tool_arg(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL_ARG, p)
    }

    pub fn tool_arg_open(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_ARG_OPEN, p);
        self.p.atomic(t)
    }

    pub fn tool_arg_close(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_ARG_CLOSE, p);
        self.p.atomic(t)
    }

    pub fn tool_arg_name(&mut self, p: ParserId) -> ParserId {
        let t = self.p.tag(chat_tag::TOOL_ARG_NAME, p);
        self.p.atomic(t)
    }

    pub fn tool_arg_value(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL_ARG_VALUE, p)
    }

    /// schema-declared string types — not treated as a potential JSON container
    pub fn tool_arg_string_value(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL_ARG_STRING_VALUE, p)
    }

    pub fn tool_arg_json_value(&mut self, p: ParserId) -> ParserId {
        self.p.tag(chat_tag::TOOL_ARG_VALUE, p)
    }

    // ---- structured helpers ---------------------------------------------------

    /// `tag_with_safe_content` (chat-peg-parser.cpp:229-237)
    pub fn tag_with_safe_content(&mut self, tag_name: &str, marker: &str, p: ParserId) -> ParserId {
        if marker.is_empty() {
            let any = self.p.any();
            let c = self.content(any);
            let chunk = self.p.rule(tag_name, c, false);
            let choice = self.p.choice(&[p, chunk]);
            return self.p.zero_or_more(choice);
        }
        let m = self.p.literal(marker);
        let neg = self.p.negate(m);
        let until = self.p.until(marker);
        let seq = self.p.sequence(&[neg, until]);
        let inner = self.content(seq);
        let content_chunk = self.p.rule(tag_name, inner, false);
        let choice = self.p.choice(&[p, content_chunk]);
        self.p.zero_or_more(choice)
    }

    /// `permute` (chat-peg-parser.cpp:239-274): matches every parser exactly
    /// once, in any order.
    pub fn permute(&mut self, rule_prefix: &str, parsers: &[ParserId]) -> ParserId {
        if parsers.is_empty() {
            return self.p.eps();
        }
        if parsers.len() == 1 || parsers.len() > COMMON_CHAT_MAX_PERMUTE {
            return self.p.sequence(parsers);
        }

        fn remaining_of(
            b: &mut ChatPegBuilder,
            parsers: &[ParserId],
            rules: &mut BTreeMap<u32, ParserId>,
            rule_prefix: &str,
            remaining: u32,
        ) -> ParserId {
            if remaining == 0 {
                return b.p.eps();
            }
            if let Some(&cached) = rules.get(&remaining) {
                return cached;
            }
            let mut alternatives: Vec<ParserId> = Vec::new();
            for (i, &parser) in parsers.iter().enumerate() {
                let bit = 1u32 << i;
                if remaining & bit != 0 {
                    let rest = remaining_of(b, parsers, rules, rule_prefix, remaining & !bit);
                    let seq = b.p.sequence(&[parser, rest]);
                    alternatives.push(seq);
                }
            }
            let alternatives = b.p.choice(&alternatives);
            let name = format!("{rule_prefix}-{remaining}");
            let r = b.p.rule(&name, alternatives, false);
            rules.insert(remaining, r);
            r
        }

        let mut rules: BTreeMap<u32, ParserId> = BTreeMap::new();
        remaining_of(
            self,
            parsers,
            &mut rules,
            rule_prefix,
            (1u32 << parsers.len()) - 1,
        )
    }

    /// `prefix` (chat-peg-parser.cpp:869-877)
    pub fn prefix(&mut self, s: &str, delimiter: &str) -> ParserId {
        if s.is_empty() {
            return self.p.eps();
        }
        if delimiter.is_empty() {
            return self.p.literal(s);
        }
        let cut = &s[..s.find(delimiter).unwrap_or(s.len())];
        self.p.literal(cut)
    }

    /// `optspace` (chat-peg-parser.cpp:879-903): the tag's leading and trailing
    /// spaces are optional.
    pub fn optspace(&mut self, tag: &str) -> ParserId {
        let b = tag.as_bytes();
        let mut end_of_prefix_space = tag.len();
        let mut start_of_suffix_space = tag.len();
        for (i, c) in b.iter().enumerate() {
            if !crate::chat::c_isspace_pub(*c) {
                end_of_prefix_space = i;
                break;
            }
        }
        for i in (1..=tag.len()).rev() {
            if !crate::chat::c_isspace_pub(b[i - 1]) {
                start_of_suffix_space = i;
                break;
            }
        }
        let mut acc = self.p.eps();
        for i in 0..end_of_prefix_space {
            let lit = self.p.literal(&tag[i..i + 1]);
            let opt = self.p.optional(lit);
            acc = self.p.sequence(&[acc, opt]);
        }
        let mid = self
            .p
            .literal(&tag[end_of_prefix_space..start_of_suffix_space]);
        acc = self.p.sequence(&[acc, mid]);
        for i in start_of_suffix_space..tag.len() {
            let lit = self.p.literal(&tag[i..i + 1]);
            let opt = self.p.optional(lit);
            acc = self.p.sequence(&[acc, opt]);
        }
        acc
    }

    /// `json_member` (peg-parser.cpp:1371-1380) via the plain builder
    pub fn json_member(&mut self, key: &str, p: ParserId) -> ParserId {
        self.p.json_member(key, p)
    }

    // ---- standard_json_tools (chat-peg-parser.cpp:905-954 + the 3 modes) ----

    /// `standard_json_tools` (chat-peg-parser.cpp:905-954)
    #[allow(clippy::too_many_arguments)]
    pub fn standard_json_tools(
        &mut self,
        section_start: &str,
        section_end: &str,
        tools: &Json,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
        name_key: &str,
        args_key: &str,
        array_wrapped: bool,
        function_is_key: bool,
        call_id_key: &str,
        gen_call_id_key: &str,
        parameters_order: &[String],
        accept_openai_wrapper: bool,
    ) -> ParserId {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }

        let effective_name_key = if name_key.is_empty() {
            "name"
        } else {
            name_key
        };
        let effective_args_key = if args_key.is_empty() {
            "arguments"
        } else {
            args_key
        };

        // Dispatch to the appropriate builder based on the JSON layout mode
        let tool_choices = if function_is_key {
            self.build_json_tools_function_is_key(
                tools,
                args_key,
                effective_args_key,
                call_id_key,
                gen_call_id_key,
            )
        } else {
            let name_spec = parse_key_spec(effective_name_key);
            let args_spec = parse_key_spec(effective_args_key);
            if !name_spec.0.is_empty() || !args_spec.0.is_empty() {
                self.build_json_tools_nested_keys(
                    tools,
                    effective_name_key,
                    effective_args_key,
                    call_id_key,
                    gen_call_id_key,
                )
            } else {
                self.build_json_tools_flat_keys(
                    tools,
                    effective_name_key,
                    effective_args_key,
                    call_id_key,
                    gen_call_id_key,
                    parameters_order,
                    accept_openai_wrapper,
                )
            }
        };

        // Build the section with markers
        let mut tool_calls = tool_choices;
        if parallel_tool_calls {
            let sp = self.p.space();
            let comma = self.p.literal(",");
            let sp2 = self.p.space();
            let tail = self.p.sequence(&[sp, comma, sp2, tool_choices]);
            let tail = self.p.zero_or_more(tail);
            tool_calls = self.p.sequence(&[tool_calls, tail]);
        }

        if array_wrapped {
            let lb = self.p.literal("[");
            let sp = self.p.space();
            let sp2 = self.p.space();
            let rb = self.p.literal("]");
            tool_calls = self.p.sequence(&[lb, sp, tool_calls, sp2, rb]);
        }

        let lb = self.p.literal(section_start);
        let sp = self.p.space();
        let sp2 = self.p.space();
        let rb = self.p.literal(section_end);
        let body = self.p.sequence(&[lb, sp, tool_calls, sp2, rb]);
        let section = self.p.trigger_rule("tool-call", body);

        if force_tool_calls {
            section
        } else {
            self.p.optional(section)
        }
    }

    /// Mode 1 `build_json_tools_function_is_key` (chat-peg-parser.cpp:628-704):
    /// `{"function_name": {...}}`
    fn build_json_tools_function_is_key(
        &mut self,
        tools: &Json,
        args_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
    ) -> ParserId {
        let mut tool_choices: Vec<ParserId> = Vec::new();

        for tool_def in tools.iter() {
            let Some(function) = tool_def.at("function") else {
                continue;
            };
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);

            let mut inner_fields: Vec<ParserId> = Vec::new();

            if !call_id_key.is_empty() {
                let id_parser = {
                    let key = self.p.literal(&format!("\"{call_id_key}\""));
                    let sp = self.p.space();
                    let colon = self.p.literal(":");
                    let sp2 = self.p.space();
                    let q1 = self.p.literal("\"");
                    let content = self.p.string_content(b'"');
                    let id = self.tool_id(content);
                    let q2 = self.p.literal("\"");
                    let seq = self.p.sequence(&[key, sp, colon, sp2, q1, id, q2]);
                    self.p.atomic(seq)
                };
                let sp = self.p.space();
                let comma = self.p.literal(",");
                let sp2 = self.p.space();
                let opt_comma = self.p.optional(comma);
                let tail = self.p.sequence(&[sp, opt_comma, sp2]);
                let opt = self.p.optional(id_parser);
                inner_fields.push(self.p.sequence(&[opt, tail]));
            }

            if !gen_call_id_key.is_empty() {
                let gen_id_parser = {
                    let key = self.p.literal(&format!("\"{gen_call_id_key}\""));
                    let sp = self.p.space();
                    let colon = self.p.literal(":");
                    let sp2 = self.p.space();
                    let q1 = self.p.literal("\"");
                    let content = self.p.string_content(b'"');
                    let id1 = self.tool_id(content);
                    let q2 = self.p.literal("\"");
                    let s1 = self.p.sequence(&[q1, id1, q2]);
                    let num = self.p.json_number();
                    let id2 = self.tool_id(num);
                    let ch = self.p.choice(&[s1, id2]);
                    let seq = self.p.sequence(&[key, sp, colon, sp2, ch]);
                    self.p.atomic(seq)
                };
                let sp = self.p.space();
                let comma = self.p.literal(",");
                let sp2 = self.p.space();
                let opt_comma = self.p.optional(comma);
                let tail = self.p.sequence(&[sp, opt_comma, sp2]);
                let opt = self.p.optional(gen_id_parser);
                inner_fields.push(self.p.sequence(&[opt, tail]));
            }

            // Arguments — either wrapped in args_key or parsed directly
            let args_parser = if args_key.is_empty() {
                let j = self.p.json();
                let sch = self
                    .p
                    .schema(j, &format!("tool-{name}-schema"), &params, false);
                self.tool_args(sch)
            } else {
                let key = self.p.literal(&format!("\"{effective_args_key}\""));
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let j = self.p.json();
                let sch = self
                    .p
                    .schema(j, &format!("tool-{name}-schema"), &params, false);
                let ta = self.tool_args(sch);
                self.p.sequence(&[key, sp, colon, sp2, ta])
            };
            inner_fields.push(args_parser);

            // Build inner object parser
            let inner_object = if args_key.is_empty() && inner_fields.len() == 1 {
                inner_fields[0]
            } else {
                let lb = self.p.literal("{");
                let sp = self.p.space();
                let mut acc = self.p.sequence(&[lb, sp]);
                for (i, &f) in inner_fields.iter().enumerate() {
                    acc = self.p.sequence(&[acc, f]);
                    if i < inner_fields.len() - 1 {
                        let s = self.p.space();
                        acc = self.p.sequence(&[acc, s]);
                    }
                }
                let sp = self.p.space();
                let rb = self.p.literal("}");
                self.p.sequence(&[acc, sp, rb])
            };

            let tool_parser = {
                let lb = self.p.literal("{");
                let open = self.tool_open(lb);
                let sp = self.p.space();
                let q = self.p.literal("\"");
                let nm = self.p.literal(&name);
                let tn = self.tool_name(nm);
                let q2 = self.p.literal("\"");
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let rb = self.p.literal("}");
                let close = self.tool_close(rb);
                let seq = self.p.sequence(&[
                    open,
                    sp,
                    q,
                    tn,
                    q2,
                    sp2,
                    colon,
                    sp2,
                    inner_object,
                    sp2,
                    close,
                ]);
                self.tool(seq)
            };

            let r = self.p.rule(&format!("tool-{name}"), tool_parser, false);
            tool_choices.push(r);
        }

        if tool_choices.is_empty() {
            return self.p.eps();
        }
        self.p.choice(&tool_choices)
    }

    /// Mode 2 `build_json_tools_nested_keys` (chat-peg-parser.cpp:707-776):
    /// dot notation like `"function.name"`
    fn build_json_tools_nested_keys(
        &mut self,
        tools: &Json,
        effective_name_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
    ) -> ParserId {
        let mut tool_choices: Vec<ParserId> = Vec::new();

        let name_spec = parse_key_spec(effective_name_key);
        let args_spec = parse_key_spec(effective_args_key);

        let nested_prefix = if !name_spec.0.is_empty() {
            name_spec.0.clone()
        } else {
            args_spec.0.clone()
        };
        let nested_name_field = if !name_spec.0.is_empty() {
            name_spec.1.clone()
        } else {
            effective_name_key.to_string()
        };
        let nested_args_field = if !args_spec.0.is_empty() {
            args_spec.1.clone()
        } else {
            effective_args_key.to_string()
        };

        for tool_def in tools.iter() {
            let Some(function) = tool_def.at("function") else {
                continue;
            };
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);

            let nested_name = {
                let key = self.p.literal(&format!("\"{nested_name_field}\""));
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let q1 = self.p.literal("\"");
                let nm = self.p.literal(&name);
                let tn = self.tool_name(nm);
                let q2 = self.p.literal("\"");
                let seq = self.p.sequence(&[q1, tn, q2]);
                let a = self.p.atomic(seq);
                self.p.sequence(&[key, sp, colon, sp2, a])
            };
            let nested_args = {
                let key = self.p.literal(&format!("\"{nested_args_field}\""));
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let j = self.p.json();
                let sch = self
                    .p
                    .schema(j, &format!("tool-{name}-schema"), &params, false);
                let ta = self.tool_args(sch);
                self.p.sequence(&[key, sp, colon, sp2, ta])
            };
            let nested_object = {
                let lb = self.p.literal("{");
                let sp = self.p.space();
                let comma = self.p.literal(",");
                let sp2 = self.p.space();
                let sp3 = self.p.space();
                let rb = self.p.literal("}");
                self.p
                    .sequence(&[lb, sp, nested_name, sp2, comma, sp3, nested_args, sp2, rb])
            };

            // Format: { id?, "function": {...} }
            let lb = self.p.literal("{");
            let sp0 = self.p.space();
            let mut tool_parser_body = self.tool_open(lb);
            tool_parser_body = self.p.sequence(&[tool_parser_body, sp0]);

            if !call_id_key.is_empty() {
                let id_spec = parse_key_spec(call_id_key);
                if id_spec.0.is_empty() {
                    let id_parser = {
                        let key = self.p.literal(&format!("\"{call_id_key}\""));
                        let sp = self.p.space();
                        let colon = self.p.literal(":");
                        let sp2 = self.p.space();
                        let q1 = self.p.literal("\"");
                        let content = self.p.string_content(b'"');
                        let id = self.tool_id(content);
                        let q2 = self.p.literal("\"");
                        let seq = self.p.sequence(&[key, sp, colon, sp2, q1, id, q2]);
                        self.p.atomic(seq)
                    };
                    let sp = self.p.space();
                    let comma = self.p.literal(",");
                    let sp2 = self.p.space();
                    let opt = self.p.optional(id_parser);
                    tool_parser_body = self.p.sequence(&[tool_parser_body, opt, sp, comma, sp2]);
                }
            }

            if !gen_call_id_key.is_empty() {
                let gen_id_spec = parse_key_spec(gen_call_id_key);
                if gen_id_spec.0.is_empty() {
                    let gen_id_parser = {
                        let key = self.p.literal(&format!("\"{gen_call_id_key}\""));
                        let sp = self.p.space();
                        let colon = self.p.literal(":");
                        let sp2 = self.p.space();
                        let q1 = self.p.literal("\"");
                        let content = self.p.string_content(b'"');
                        let id1 = self.tool_id(content);
                        let s1 = self.p.sequence(&[q1, id1, q1]);
                        let num = self.p.json_number();
                        let id2 = self.tool_id(num);
                        let ch = self.p.choice(&[s1, id2]);
                        let seq = self.p.sequence(&[key, sp, colon, sp2, ch]);
                        self.p.atomic(seq)
                    };
                    let sp = self.p.space();
                    let comma = self.p.literal(",");
                    let sp2 = self.p.space();
                    let opt = self.p.optional(gen_id_parser);
                    tool_parser_body = self.p.sequence(&[tool_parser_body, opt, sp, comma, sp2]);
                }
            }

            let nested_field = {
                let key = self.p.literal(&format!("\"{nested_prefix}\""));
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                self.p.sequence(&[key, sp, colon, sp2, nested_object])
            };
            let sp = self.p.space();
            let rb = self.p.literal("}");
            let close = self.tool_close(rb);
            tool_parser_body = self
                .p
                .sequence(&[tool_parser_body, nested_field, sp, close]);

            let t = self.tool(tool_parser_body);
            let r = self.p.rule(&format!("tool-{name}"), t, false);
            tool_choices.push(r);
        }

        if tool_choices.is_empty() {
            return self.p.eps();
        }
        self.p.choice(&tool_choices)
    }

    /// Mode 3 `build_json_tools_flat_keys` (chat-peg-parser.cpp:779-867)
    #[allow(clippy::too_many_arguments)]
    fn build_json_tools_flat_keys(
        &mut self,
        tools: &Json,
        effective_name_key: &str,
        effective_args_key: &str,
        call_id_key: &str,
        gen_call_id_key: &str,
        parameters_order: &[String],
        accept_openai_wrapper: bool,
    ) -> ParserId {
        let mut tool_choices: Vec<ParserId> = Vec::new();
        let name_key_parser = self.p.literal(&format!("\"{effective_name_key}\""));
        let args_key_parser = self.p.literal(&format!("\"{effective_args_key}\""));

        for tool_def in tools.iter() {
            let Some(function) = tool_def.at("function") else {
                continue;
            };
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);

            let tool_name_ = {
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let q1 = self.p.literal("\"");
                let nm = self.p.literal(&name);
                let tn = self.tool_name(nm);
                let q2 = self.p.literal("\"");
                let seq = self.p.sequence(&[q1, tn, q2]);
                let a = self.p.atomic(seq);
                self.p.sequence(&[name_key_parser, sp, colon, sp2, a])
            };
            let tool_args_ = {
                let sp = self.p.space();
                let colon = self.p.literal(":");
                let sp2 = self.p.space();
                let j = self.p.json();
                let sch = self
                    .p
                    .schema(j, &format!("tool-{name}-schema"), &params, false);
                let ta = self.tool_args(sch);
                self.p.sequence(&[args_key_parser, sp, colon, sp2, ta])
            };

            // Build ID parsers if keys are provided
            let mut id_parser = self.p.eps();
            if !call_id_key.is_empty() {
                id_parser = {
                    let key = self.p.literal(&format!("\"{call_id_key}\""));
                    let sp = self.p.space();
                    let colon = self.p.literal(":");
                    let sp2 = self.p.space();
                    let q1 = self.p.literal("\"");
                    let content = self.p.string_content(b'"');
                    let id1 = self.tool_id(content);
                    let q2 = self.p.literal("\"");
                    let s1 = self.p.sequence(&[q1, id1, q2]);
                    let num = self.p.json_number();
                    let id2 = self.tool_id(num);
                    let ch = self.p.choice(&[s1, id2]);
                    let seq = self.p.sequence(&[key, sp, colon, sp2, ch]);
                    self.p.atomic(seq)
                };
            }

            let mut gen_id_parser = self.p.eps();
            if !gen_call_id_key.is_empty() {
                gen_id_parser = {
                    let key = self.p.literal(&format!("\"{gen_call_id_key}\""));
                    let sp = self.p.space();
                    let colon = self.p.literal(":");
                    let sp2 = self.p.space();
                    let q1 = self.p.literal("\"");
                    let content = self.p.string_content(b'"');
                    let id1 = self.tool_id(content);
                    let q2 = self.p.literal("\"");
                    let s1 = self.p.sequence(&[q1, id1, q2]);
                    let num = self.p.json_number();
                    let id2 = self.tool_id(num);
                    let ch = self.p.choice(&[s1, id2]);
                    let seq = self.p.sequence(&[key, sp, colon, sp2, ch]);
                    self.p.atomic(seq)
                };
            }

            // (parser, key) pairs sorted by parameters_order
            let mut parser_pairs: Vec<(ParserId, String)> = Vec::new();
            parser_pairs.push((tool_name_, effective_name_key.to_string()));
            parser_pairs.push((tool_args_, effective_args_key.to_string()));
            if !call_id_key.is_empty() {
                let opt = self.p.optional(id_parser);
                parser_pairs.push((opt, call_id_key.to_string()));
            }
            if !gen_call_id_key.is_empty() {
                let opt = self.p.optional(gen_id_parser);
                parser_pairs.push((opt, gen_call_id_key.to_string()));
            }

            parser_pairs.sort_by(|a, b| {
                let pos = |k: &str| parameters_order.iter().position(|p| p == k);
                let idx_a = pos(&a.1).unwrap_or(parameters_order.len());
                let idx_b = pos(&b.1).unwrap_or(parameters_order.len());
                idx_a.cmp(&idx_b)
            });

            // optional leading "type": "function" (OpenAI wrapper)
            let mut type_field = self.p.eps();
            if accept_openai_wrapper {
                type_field = {
                    let key = self.p.literal("\"type\"");
                    let sp = self.p.space();
                    let colon = self.p.literal(":");
                    let sp2 = self.p.space();
                    let v = self.p.literal("\"function\"");
                    let sp3 = self.p.space();
                    let comma = self.p.literal(",");
                    let sp4 = self.p.space();
                    let seq = self.p.sequence(&[key, sp, colon, sp2, v, sp3, comma, sp4]);
                    self.p.optional(seq)
                };
            }
            let lb = self.p.literal("{");
            let open = self.tool_open(lb);
            let sp = self.p.space();
            let mut ordered_body = self.p.sequence(&[open, sp, type_field]);
            for (i, &(parser, _)) in parser_pairs.iter().enumerate() {
                ordered_body = self.p.sequence(&[ordered_body, parser]);
                if i < parser_pairs.len() - 1 {
                    let sp = self.p.space();
                    let comma = self.p.literal(",");
                    let sp2 = self.p.space();
                    ordered_body = self.p.sequence(&[ordered_body, sp, comma, sp2]);
                }
            }
            let sp = self.p.space();
            let rb = self.p.literal("}");
            let close = self.tool_close(rb);
            let ordered_body = self.p.sequence(&[ordered_body, sp, close]);

            let t = self.tool(ordered_body);
            let r = self.p.rule(&format!("tool-{name}"), t, false);
            tool_choices.push(r);
        }

        if tool_choices.is_empty() {
            return self.p.eps();
        }
        self.p.choice(&tool_choices)
    }

    /// `standard_constructed_tools` (chat-peg-parser.cpp:458-526)
    pub fn standard_constructed_tools(
        &mut self,
        markers: &BTreeMap<String, String>,
        tools: &Json,
        parallel_tool_calls: bool,
        force_tool_calls: bool,
    ) -> ParserId {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }

        let get_marker = |key: &str, default_val: &str| -> String {
            markers
                .get(key)
                .cloned()
                .unwrap_or_else(|| default_val.to_string())
        };

        let section_start = get_marker("tool_call_start_marker", "<tool_call>");
        let section_end = get_marker("tool_call_end_marker", "</tool_call>");
        let func_opener = get_marker("function_opener", "<function=");
        let func_name_suffix = get_marker("function_name_suffix", ">");
        let func_closer = get_marker("function_closer", "</function>");
        let param_key_prefix = get_marker("parameter_key_prefix", "<param=");
        let param_key_suffix = get_marker("parameter_key_suffix", ">");
        let param_closer = get_marker("parameter_closer", "</param>");

        let mut tool_choice_alts: Vec<ParserId> = Vec::new();

        for tool_def in tools.iter() {
            let Some(function) = tool_def.at("function") else {
                continue;
            };
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);

            // Build argument parsers
            let mut args = self.p.eps();
            if let Some(props) = params.at("properties") {
                if !props.empty() {
                    let mut arg_alts: Vec<ParserId> = Vec::new();
                    for (prop_name, _) in props.items() {
                        let lit1 = self.p.literal(&prop_name);
                        let lit2 = self.p.literal(&format!("\"{prop_name}\""));
                        let lit3 = self.p.literal(&format!("'{prop_name}'"));
                        let arg_name_parser = self.p.choice(&[lit1, lit2, lit3]);

                        let open = self.p.literal(&param_key_prefix);
                        let tao = self.tool_arg_open(open);
                        let tan = self.tool_arg_name(arg_name_parser);
                        let suffix = self.p.literal(&param_key_suffix);
                        let until = self.p.until(&param_closer);
                        let tav = self.tool_arg_value(until);
                        let close = self.p.literal(&param_closer);
                        let tac = self.tool_arg_close(close);
                        let seq = self.p.sequence(&[tao, tan, suffix, tav, tac]);
                        let arg_rule = self.tool_arg(seq);
                        arg_alts.push(arg_rule);
                    }
                    let arg_choice = self.p.choice(&arg_alts);
                    let sp = self.p.space();
                    let seq = self.p.sequence(&[arg_choice, sp]);
                    args = self.p.zero_or_more(seq);
                }
            }

            // <function=name>args</function>
            let open = self.p.literal(&func_opener);
            let nm = self.p.literal(&name);
            let tn = self.tool_name(nm);
            let suffix = self.p.literal(&func_name_suffix);
            let oseq = self.p.sequence(&[open, tn, suffix]);
            let t_open = self.tool_open(oseq);
            let sp = self.p.space();
            let ta = self.tool_args(args);
            let sp2 = self.p.space();
            let close = self.p.literal(&func_closer);
            let t_close = self.tool_close(close);
            let seq = self.p.sequence(&[t_open, sp, ta, sp2, t_close]);
            let tool_parser = self.tool(seq);

            let r = self.p.rule(&format!("tool-{name}"), tool_parser, false);
            tool_choice_alts.push(r);
        }

        let tool_choices = self.p.choice(&tool_choice_alts);

        // Build the section with markers
        let section = if parallel_tool_calls {
            let lb = self.p.literal(&section_start);
            let sp = self.p.space();
            let seq = self.p.sequence(&[tool_choices, sp]);
            let one = self.p.one_or_more(seq);
            let rb = self.p.literal(&section_end);
            let body = self.p.sequence(&[lb, sp, one, rb]);
            self.p.trigger_rule("tool-call", body)
        } else {
            let lb = self.p.literal(&section_start);
            let sp = self.p.space();
            let sp2 = self.p.space();
            let rb = self.p.literal(&section_end);
            let body = self.p.sequence(&[lb, sp, tool_choices, sp2, rb]);
            self.p.trigger_rule("tool-call", body)
        };

        if force_tool_calls {
            section
        } else {
            self.p.optional(section)
        }
    }

    /// `python_or_json_value` (chat-peg-parser.cpp:529-548) — like
    /// python_value(), but the leaf also accepts JSON-cased true/false/null
    /// (used by LFM2/LFM2.5). Plain-builder-only, so it lives on PegBuilder.
    pub fn python_or_json_value(&mut self) -> ParserId {
        python_or_json_value_impl(&mut self.p)
    }

    /// `python_style_tool_calls` (chat-peg-parser.cpp:552-616): `name(arg1="v1")`
    pub fn python_style_tool_calls(
        &mut self,
        tools: &Json,
        parallel_tool_calls: bool,
        allow_json_literals: bool,
    ) -> ParserId {
        if !tools.is_array() || tools.empty() {
            return self.p.eps();
        }

        let mut tool_choice_alts: Vec<ParserId> = Vec::new();

        for tool_def in tools.iter() {
            let Some(function) = tool_def.at("function") else {
                continue;
            };
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);

            let mut args = self.p.eps();
            if let Some(props) = params.at("properties") {
                if !props.empty() {
                    let mut arg_alts: Vec<ParserId> = Vec::new();
                    for (prop_name, prop_def) in props.items() {
                        let is_string_type = prop_def
                            .at("type")
                            .and_then(|v| v.get_str().ok())
                            .map(|t| t == "string")
                            .unwrap_or(false);

                        let arg_name_parser = self.p.literal(&prop_name);

                        // Quoted literal as a value (escapes preserved)
                        let string_value_parser = {
                            let q1 = self.p.literal("\"");
                            let c1 = self.p.string_content(b'"');
                            let q2 = self.p.literal("\"");
                            let d = self.p.sequence(&[q1, c1, q2]);
                            let q3 = self.p.literal("'");
                            let c2 = self.p.string_content(b'\'');
                            let q4 = self.p.literal("'");
                            let s = self.p.sequence(&[q3, c2, q4]);
                            let ch = self.p.choice(&[d, s]);
                            self.tool_arg_value(ch)
                        };

                        let arg_value_parser = if is_string_type {
                            string_value_parser
                        } else {
                            let inner = if allow_json_literals {
                                self.python_or_json_value()
                            } else {
                                self.p.python_value()
                            };
                            self.tool_arg_value(inner)
                        };

                        // Full argument: name="value" or name=value
                        let eq = self.p.literal("=");
                        let tan = self.tool_arg_name(arg_name_parser);
                        let eseq = self.p.sequence(&[tan, eq]);
                        let tao = self.tool_arg_open(eseq);
                        let eps = self.p.eps();
                        let tac = self.tool_arg_close(eps);
                        let seq = self.p.sequence(&[tao, arg_value_parser, tac]);
                        let arg_rule = self.tool_arg(seq);
                        arg_alts.push(arg_rule);
                    }

                    let arg_choice = self.p.choice(&arg_alts);
                    let comma = self.p.literal(",");
                    let sp = self.p.space();
                    let tseq = self.p.sequence(&[comma, sp, arg_choice]);
                    let tail = self.p.zero_or_more(tseq);
                    args = self.p.sequence(&[arg_choice, tail]);
                }
            }

            let nm = self.p.literal(&name);
            let tn = self.tool_name(nm);
            let lp = self.p.literal("(");
            let oseq = self.p.sequence(&[tn, lp]);
            let t_open = self.tool_open(oseq);
            let sp = self.p.space();
            let ta = self.tool_args(args);
            let sp2 = self.p.space();
            let rp = self.p.literal(")");
            let t_close = self.tool_close(rp);
            let seq = self.p.sequence(&[t_open, sp, ta, sp2, t_close]);
            let tool_parser = self.tool(seq);

            let r = self.p.rule(&format!("tool-{name}"), tool_parser, false);
            tool_choice_alts.push(r);
        }

        let tool_choices = self.p.choice(&tool_choice_alts);

        let lb = self.p.literal("[");
        let sp = self.p.space();
        let comma = self.p.literal(",");
        let sp2 = self.p.space();
        let rb = self.p.literal("]");
        if parallel_tool_calls {
            let tseq = self.p.sequence(&[comma, sp, tool_choices]);
            let tail = self.p.zero_or_more(tseq);
            self.p.sequence(&[lb, sp, tool_choices, tail, sp2, rb])
        } else {
            self.p.sequence(&[lb, sp, tool_choices, sp2, rb])
        }
    }
}

/// `python_or_json_value` body (chat-peg-parser.cpp:529-548)
fn python_or_json_value_impl(p: &mut PegBuilder) -> ParserId {
    p.rule_fn("python-or-json-value", |b| {
        let ws = b.space();
        let value = python_or_json_value_impl(b);

        let member = {
            let key = b.python_string();
            let colon = b.literal(":");
            b.sequence(&[key, ws, colon, ws, value])
        };
        let members = {
            let tail = {
                let comma = b.literal(",");
                b.sequence(&[ws, comma, ws, member])
            };
            let tail = b.zero_or_more(tail);
            b.sequence(&[member, tail])
        };
        let dict = b.rule_fn("python-or-json-dict", |b| {
            let lb = b.literal("{");
            let rb1 = b.literal("}");
            let rb2 = b.literal("}");
            let seq = {
                let s = b.sequence(&[members, ws, rb2]);
                b.choice(&[rb1, s])
            };
            let w = b.space();
            b.sequence(&[lb, ws, seq, w])
        });

        let elements = {
            let tail = {
                let comma = b.literal(",");
                b.sequence(&[comma, ws, value])
            };
            let tail = b.zero_or_more(tail);
            b.sequence(&[value, tail])
        };
        let array = b.rule_fn("python-or-json-array", |b| {
            let lb = b.literal("[");
            let rb1 = b.literal("]");
            let rb2 = b.literal("]");
            let seq = {
                let s = b.sequence(&[elements, ws, rb2]);
                b.choice(&[rb1, s])
            };
            let w = b.space();
            b.sequence(&[lb, ws, seq, w])
        });

        let s = b.python_string();
        let n = b.python_number();
        let pb = b.python_bool();
        let pn = b.python_null();
        let jb = b.json_bool();
        let jn = b.json_null();
        b.choice(&[dict, array, s, n, pb, pn, jb, jn])
    })
}

/// `parse_key_spec` (chat-peg-parser.cpp:618-625)
fn parse_key_spec(key: &str) -> (String, String) {
    match key.find('.') {
        None => (String::new(), key.to_string()), // top-level field
        Some(dot_pos) => (key[..dot_pos].to_string(), key[dot_pos + 1..].to_string()),
    }
}

/// `build_chat_peg_parser` (chat-peg-parser.h:182-187)
pub fn build_chat_peg_parser<F>(f: F) -> Result<PegArena, String>
where
    F: FnOnce(&mut ChatPegBuilder) -> ParserId,
{
    let mut builder = ChatPegBuilder::default();
    let root = f(&mut builder);
    builder.p.set_root(root);
    builder.p.build()
}

// ---------------------------------------------------------------------------
// chat peg mapper (chat-peg-parser.cpp:12-456)
// ---------------------------------------------------------------------------

fn trim_trailing_space(sv: &str) -> &str {
    // chat-peg-parser.cpp:12-22 (max = -1)
    let b = sv.as_bytes();
    let mut end = b.len();
    while end > 0 && crate::chat::c_isspace_pub(b[end - 1]) {
        end -= 1;
    }
    &sv[..end]
}

fn trim_leading_space(sv: &str, max: i32) -> &str {
    // chat-peg-parser.cpp:24-34
    let b = sv.as_bytes();
    let mut start = 0usize;
    let mut count = 0;
    while start < b.len() && crate::chat::c_isspace_pub(b[start]) {
        if max != -1 && count >= max {
            break;
        }
        start += 1;
        count += 1;
    }
    &sv[start..]
}

fn trim_peg(sv: &str) -> &str {
    trim_trailing_space(trim_leading_space(sv, 1))
}

/// `json_brace_depth` (chat-peg-parser.cpp:42-68)
fn json_brace_depth(s: &str) -> i32 {
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    for c in s.bytes() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == b'\\' && in_string {
            escaped = true;
            continue;
        }
        if c == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
            }
        }
    }
    depth
}

/// `escape_json_string_inner` (chat-peg-parser.cpp:71-77)
pub(crate) fn escape_json_string_inner(s: &str) -> String {
    let escaped = Json::String(s.to_string()).dump();
    if escaped.len() >= 2 && escaped.starts_with('"') && escaped.ends_with('"') {
        return escaped[1..escaped.len() - 1].to_string();
    }
    escaped
}

/// `normalize_quotes_to_json` (chat-peg-parser.cpp:84-186)
fn normalize_quotes_to_json(input: &str) -> String {
    let b = input.as_bytes();
    let mut result = String::with_capacity(input.len() + 16);

    let mut in_single_quoted = false;
    let mut in_double_quoted = false;

    let is_word_char = |ch: u8| ch.is_ascii_alphanumeric() || ch == b'_';

    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;

        // Handle escape sequences
        if c == '\\' && i + 1 < b.len() {
            let next = b[i + 1] as char;
            if in_single_quoted {
                if next == '\'' {
                    result.push('\'');
                    i += 1;
                    continue;
                }
                if next == '"' {
                    result.push_str("\\\"");
                    i += 1;
                    continue;
                }
                result.push(c);
                result.push(next);
                i += 1;
                continue;
            }
            if in_double_quoted {
                result.push(c);
                result.push(next);
                i += 1;
                continue;
            }
            result.push(c);
            i += 1;
            continue;
        }

        if c == '"' {
            if in_single_quoted {
                result.push_str("\\\"");
            } else {
                in_double_quoted = !in_double_quoted;
                result.push(c);
            }
        } else if c == '\'' {
            if in_double_quoted {
                result.push(c);
            } else if in_single_quoted {
                in_single_quoted = false;
                result.push('"');
            } else {
                in_single_quoted = true;
                result.push('"');
            }
        } else if !in_single_quoted
            && !in_double_quoted
            && (c == 'T' || c == 'F' || c == 'N')
            && (i == 0 || !is_word_char(b[i - 1]))
        {
            // Python literals → JSON; prefix match keeps partials monotonic
            const LITERALS: [(&str, &str); 3] =
                [("True", "true"), ("False", "false"), ("None", "null")];
            let mut n = 0usize;
            while i + n < b.len() && is_word_char(b[i + n]) {
                n += 1;
            }
            let token = &input[i..i + n];
            let mut matched = false;
            for (py, js) in LITERALS {
                if py.len() >= n && &py[..n] == token {
                    result += &js[..n];
                    i += n - 1;
                    matched = true;
                    break;
                }
            }
            if !matched {
                result.push(c);
            }
        } else {
            result.push(c);
        }
        i += 1;
    }

    result
}

/// `common_chat_peg_mapper` (chat-peg-parser.h:10-33, .cpp:276-456)
pub struct ChatPegMapper<'a> {
    pub result: &'a mut ChatMsg,
    // Tool call handling state (chat-peg-parser.h:24-28)
    pending_tool_call: Option<ChatToolCall>,
    /// `current_tool != null`; when `current_is_pending` it points at the
    /// pending call, else at the last entry of `result.tool_calls`.
    have_current: bool,
    current_is_pending: bool,
    arg_count: i32,
    closing_quote_pending: bool,
    /// Buffer to delay arguments until the tool name is known
    args_buffer: String,
}

impl<'a> ChatPegMapper<'a> {
    pub fn new(result: &'a mut ChatMsg) -> Self {
        ChatPegMapper {
            result,
            pending_tool_call: None,
            have_current: false,
            current_is_pending: false,
            arg_count: 0,
            closing_quote_pending: false,
            args_buffer: String::new(),
        }
    }

    fn current_tool(&mut self) -> Option<&mut ChatToolCall> {
        if !self.have_current {
            return None;
        }
        if self.current_is_pending {
            self.pending_tool_call.as_mut()
        } else {
            self.result.tool_calls.last_mut()
        }
    }

    /// `normalize_container_value` (chat-peg-parser.cpp:280-282)
    fn normalize_container_value(input: &str) -> String {
        normalize_quotes_to_json(input)
    }

    /// `from_ast` (chat-peg-parser.cpp:284-313): pre-order walk of the parse
    /// result's AST nodes, then the pending-flush epilogue.
    pub fn from_ast(&mut self, ctx: &ParseContext, result: &ParseResult) {
        let mut stack: Vec<usize> = result.nodes.iter().rev().copied().collect();
        while let Some(id) = stack.pop() {
            let children: Vec<usize>;
            {
                let node = ctx.ast.get(id);
                self.map(node, &ctx.input);
                children = node.children.clone();
            }
            for &child in children.iter().rev() {
                stack.push(child);
            }
        }
        self.finish();
    }

    /// `map` (chat-peg-parser.cpp:315-456) — one AST node.
    pub fn map(&mut self, node: &AstNode, input: &str) {
        let node_text = |n: &AstNode| -> String {
            String::from_utf8_lossy(&input.as_bytes()[n.start..n.end]).into_owned()
        };

        let is_reasoning = node.tag == chat_tag::REASONING;
        let is_content = node.tag == chat_tag::CONTENT;

        if is_reasoning {
            // GPT OSS can have more than one reasoning block, concatenate
            self.result.reasoning_content += &node.sanitized_text(input.as_bytes());
        }
        if is_content {
            // Concatenate content from multiple content nodes
            self.result.content += &node.sanitized_text(input.as_bytes());
        }

        let is_tool_open = node.tag == chat_tag::TOOL_OPEN;
        let is_tool_close = node.tag == chat_tag::TOOL_CLOSE;
        let is_tool_name = node.tag == chat_tag::TOOL_NAME;
        let is_tool_id = node.tag == chat_tag::TOOL_ID;
        let is_tool_args = node.tag == chat_tag::TOOL_ARGS;
        let is_arg_open = node.tag == chat_tag::TOOL_ARG_OPEN;
        let is_arg_close = node.tag == chat_tag::TOOL_ARG_CLOSE;
        let is_arg_name = node.tag == chat_tag::TOOL_ARG_NAME;
        let is_arg_value = node.tag == chat_tag::TOOL_ARG_VALUE;
        let is_arg_string_value = node.tag == chat_tag::TOOL_ARG_STRING_VALUE;

        if is_tool_open {
            self.pending_tool_call = Some(ChatToolCall::default());
            self.have_current = true;
            self.current_is_pending = true;
            self.arg_count = 0;
            self.args_buffer.clear();
            self.closing_quote_pending = false;
        }

        if is_tool_id && self.have_current {
            let node_txt = node_text(node);
            let text = trim_trailing_space(&node_txt);
            let text = if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
                &text[1..text.len() - 1]
            } else {
                text
            };
            if let Some(tool) = self.current_tool() {
                tool.id = text.to_string();
            }
        }

        if is_tool_name && self.have_current {
            let node_txt = node_text(node);
            let name = trim_trailing_space(&node_txt).to_string();
            let buffer = std::mem::take(&mut self.args_buffer);
            {
                let tool = self.current_tool().unwrap();
                tool.name = name;
                // Now that we have the name, populate the arguments from the buffer
                if !buffer.is_empty() {
                    tool.arguments = buffer;
                } else if tool.arguments.is_empty() {
                    tool.arguments = "{".to_string();
                }
            }
            // Add the tool call to results so streaming can see it
            if let Some(pending) = self.pending_tool_call.take() {
                self.result.tool_calls.push(pending);
                self.current_is_pending = false;
            }
        }

        if is_tool_args && self.have_current {
            // JSON format: arguments come as a complete JSON object; tagged
            // format builds up from individual arg_name/arg_value nodes
            let text = trim_trailing_space(&node_text(node)).to_string();
            if !text.is_empty() && text.starts_with('{') {
                let target_is_buffer = self.args_target_is_buffer();
                if target_is_buffer {
                    self.args_buffer = text;
                } else if let Some(tool) = self.current_tool() {
                    tool.arguments = text;
                }
            }
        }

        if is_arg_open {
            self.closing_quote_pending = false;
        }

        if is_arg_name && self.have_current {
            let mut arg_entry = String::new();
            if self.arg_count > 0 {
                arg_entry = ",".to_string();
            }
            arg_entry += &format!(
                "{}:",
                Json::String(trim_peg(&node_text(node)).to_string()).dump()
            );
            self.arg_count += 1;

            let target_is_buffer = self.args_target_is_buffer();
            if target_is_buffer {
                if self.args_buffer.is_empty() {
                    self.args_buffer = "{".to_string();
                }
                self.args_buffer += &arg_entry;
            } else if let Some(tool) = self.current_tool() {
                if tool.arguments.is_empty() {
                    tool.arguments = "{".to_string();
                }
                tool.arguments += &arg_entry;
            }
        }

        if (is_arg_value || is_arg_string_value) && self.have_current {
            let value_content = node_text(node);

            let mut value_to_add = String::new();
            if value_content.is_empty() && is_arg_string_value {
                // Empty string value - arg_close will add the closing quote
                value_to_add = "\"".to_string();
                self.closing_quote_pending = true;
            } else if !value_content.is_empty() && is_arg_string_value {
                // Schema declares this as string type - literal string value
                if !self.closing_quote_pending {
                    value_to_add = "\"".to_string();
                    self.closing_quote_pending = true;
                }
                value_to_add += &escape_json_string_inner(&value_content);
            } else if !value_content.is_empty() {
                // Pythonic scalars/containers -> JSON
                value_to_add += &Self::normalize_container_value(&value_content);
            }

            let target_is_buffer = self.args_target_is_buffer();
            if target_is_buffer {
                self.args_buffer += &value_to_add;
            } else if let Some(tool) = self.current_tool() {
                tool.arguments += &value_to_add;
            }
        }

        if is_arg_close && self.have_current && self.closing_quote_pending {
            let target_is_buffer = self.args_target_is_buffer();
            if target_is_buffer {
                self.args_buffer += "\"";
            } else if let Some(tool) = self.current_tool() {
                tool.arguments += "\"";
            }
            self.closing_quote_pending = false;
        }

        if is_tool_close && self.have_current {
            // Flush buffer to arguments if tool name was never seen
            let name_empty = self
                .current_tool()
                .map(|t| t.name.is_empty())
                .unwrap_or(false);
            let buffer = std::mem::take(&mut self.args_buffer);
            if name_empty && !buffer.is_empty() {
                if let Some(tool) = self.current_tool() {
                    tool.arguments = buffer;
                }
            }
            // Close any pending string quote
            if self.closing_quote_pending {
                if let Some(tool) = self.current_tool() {
                    tool.arguments += "\"";
                }
                self.closing_quote_pending = false;
            }
            // Close any unclosed braces (accounts for nested objects)
            if let Some(tool) = self.current_tool() {
                for _ in 0..json_brace_depth(&tool.arguments) {
                    tool.arguments += "}";
                }
            }
            // Add tool call to results if named; otherwise discard
            if self.pending_tool_call.is_some() {
                if let Some(pending) = self.pending_tool_call.take() {
                    if !pending.name.is_empty() {
                        self.result.tool_calls.push(pending);
                    }
                }
                // `current_tool = nullptr` (chat-peg-parser.cpp:454, upstream
                // a7b94df2c) — the close of a not-yet-named tool must not
                // leave `current_tool` pointing at the reset pending slot;
                // after a named tool the C keeps it at the last result entry
                self.have_current = false;
                self.current_is_pending = false;
            }
        }
    }

    /// `args_target()` (chat-peg-parser.cpp:276-278): before the tool name is
    /// known, argument writes go to the buffer.
    fn args_target_is_buffer(&mut self) -> bool {
        self.current_is_pending
            && self
                .current_tool()
                .map(|t| t.name.is_empty())
                .unwrap_or(false)
    }

    /// `from_ast` epilogue (chat-peg-parser.cpp:286-313): flush any pending
    /// tool call, discard whitespace-only reasoning.
    pub fn finish(&mut self) {
        if let Some(mut pending) = self.pending_tool_call.take() {
            if !pending.name.is_empty() {
                if !self.args_buffer.is_empty() {
                    pending.arguments = self.args_buffer.clone();
                }
                if self.closing_quote_pending && !pending.arguments.is_empty() {
                    pending.arguments += "\"";
                }
                self.result.tool_calls.push(pending);
            }
            self.have_current = false;
            self.current_is_pending = false;
        }

        // Discard whitespace-only reasoning content (e.g. <think></think> prefill)
        if !self.result.reasoning_content.is_empty()
            && self
                .result
                .reasoning_content
                .bytes()
                .all(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.result.reasoning_content.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// templates — common_chat_template(s) (chat.h:51-78, chat.cpp:715-855, 905-1000)
// ---------------------------------------------------------------------------

/// `CHATML_TEMPLATE_SRC` (chat.cpp:715-721)
pub const CHATML_TEMPLATE_SRC: &str = "{%- for message in messages -%}\n  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n  {{- '<|im_start|>assistant\\n' -}}\n{%- endif -%}";

/// `common_chat_template` (chat.h:51-78): parsed program + tokens + caps.
#[derive(Clone)]
pub struct ChatTemplate {
    /// parsed mini-jinja program (lexer → parser, chat.h:58-62)
    pub prog: Vec<mini_jinja::Node>,
    pub bos_tok: String,
    pub eos_tok: String,
    pub src: String,
    pub caps: JinjaCaps,
}

impl ChatTemplate {
    /// the `common_chat_template` constructor (chat.h:58-69)
    pub fn new(src: &str, bos_token: &str, eos_token: &str) -> Result<ChatTemplate, String> {
        let toks = mini_jinja::lex(src)?;
        let prog = mini_jinja::parse(&toks)?;
        let caps = caps_get(src)?;
        Ok(ChatTemplate {
            prog,
            src: src.to_string(),
            bos_tok: bos_token.to_string(),
            eos_tok: eos_token.to_string(),
            caps,
        })
    }

    pub fn source(&self) -> &str {
        &self.src
    }
    pub fn bos_token(&self) -> &str {
        &self.bos_tok
    }
    pub fn eos_token(&self) -> &str {
        &self.eos_tok
    }
    /// `original_caps()` (chat.h:75-77)
    pub fn original_caps(&self) -> &JinjaCaps {
        &self.caps
    }
}

/// `struct common_chat_templates` (chat.cpp:337-343)
pub struct ChatTemplates {
    pub add_bos: bool,
    pub add_eos: bool,
    /// Model had builtin template or template override was specified
    pub has_explicit_template: bool,
    /// always set (defaults to chatml)
    pub template_default: ChatTemplate,
    pub template_tool_use: Option<ChatTemplate>,
}

/// Model-side inputs of [`ChatTemplates::init`] standing in for the metadata
/// reads (`llama_model_chat_template` / vocab token pieces); the integrator
/// supplies them from the loaded model.
pub struct ChatTemplatesInit {
    /// `tokenizer.chat_template` (or the `--chat-template` override)
    pub chat_template_override: String,
    /// `tokenizer.chat_template.tool_use`
    pub chat_template_tool_use: String,
    pub bos_token: String,
    pub eos_token: String,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl ChatTemplates {
    /// `common_chat_templates_init` (chat.cpp:757-855)
    pub fn init(init: &ChatTemplatesInit) -> Result<ChatTemplates, String> {
        let mut default_template_src = String::new();
        let template_tool_use_src = init.chat_template_tool_use.clone();

        let has_explicit_template = !init.chat_template_override.is_empty();
        let mut has_explicit_template = has_explicit_template;
        if init.chat_template_override.is_empty() {
            // model builtin default template arrives via this field in the
            // integrator; without a model there is nothing to read
        } else {
            default_template_src = init.chat_template_override.clone();
        }
        if !template_tool_use_src.is_empty() {
            has_explicit_template = true;
        }
        if default_template_src.is_empty() || default_template_src == "chatml" {
            if !template_tool_use_src.is_empty() {
                default_template_src = template_tool_use_src.clone();
            } else {
                default_template_src = CHATML_TEMPLATE_SRC.to_string();
            }
        }

        // chat.cpp:788-809: temporary template-source patches
        if default_template_src.contains("<|channel|>")
            && default_template_src.contains("in message.content or")
        {
            default_template_src = default_template_src.replace(
                "{%- if \"<|channel|>analysis<|message|>\" in message.content or \"<|channel|>final<|message|>\" in message.content %}",
                "{%- if false %}",
            );
        }
        if default_template_src.contains("[TOOL_CALLS]")
            && default_template_src.contains("if (message['content'] is none or")
        {
            default_template_src = default_template_src.replace(
                "{%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                "{%- if false %}",
            );
        }

        let template_default =
            ChatTemplate::new(&default_template_src, &init.bos_token, &init.eos_token)
                .map_err(|e| format!("failed to initialize chat template: {e}"))?;

        let template_tool_use = if !template_tool_use_src.is_empty() {
            match ChatTemplate::new(&template_tool_use_src, &init.bos_token, &init.eos_token) {
                Ok(t) => Some(t),
                Err(_) => None, // "failed to parse tool use chat template (ignoring it)"
            }
        } else {
            None
        };

        Ok(ChatTemplates {
            add_bos: init.add_bos,
            add_eos: init.add_eos,
            has_explicit_template,
            template_default,
            template_tool_use,
        })
    }

    /// `common_chat_templates_source` (chat.cpp:744-755)
    pub fn source(&self, variant: &str) -> String {
        if !variant.is_empty() && variant == "tool_use" {
            if let Some(t) = &self.template_tool_use {
                return t.source().to_string();
            }
            return String::new();
        }
        self.template_default.source().to_string()
    }

    /// `common_chat_templates_get_caps` (chat.cpp:1525-1533)
    pub fn get_caps(&self) -> Vec<(String, bool)> {
        if let Some(tool_use) = &self.template_tool_use {
            // take the more expressive template when available
            return tool_use.caps.to_map();
        }
        self.template_default.caps.to_map()
    }
}

/// `format_time` (chat.cpp:36-43) — UTC on the pinned clock (the documented
/// chat.rs deviation from `std::localtime`).
fn format_time(now: i64, format: &str) -> String {
    mini_jinja::strftime_utc(now, format)
}

/// `common_chat_extra_context` (chat.cpp:1080-1088) — reads the WALL CLOCK
/// (`std::chrono::system_clock::now()`), not `inputs.now`; the port matches
/// that (the parity comparison normalizes rendered dates).
fn chat_extra_context() -> Json {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let mut ctx = Json::Object(Vec::new());
    ctx.set("datetime", Json::String(format_time(now, "%b %d %Y")));
    ctx.set("date_string", Json::String(format_time(now, "%d %b %Y")));
    ctx
}

/// `autoparser::generation_params` (chat-auto-parser.h:54-78)
#[derive(Clone)]
pub struct GenerationParams {
    pub messages: Json,
    pub tools: Json,
    pub tool_choice: ChatToolChoice,
    pub json_schema: Json,
    pub parallel_tool_calls: bool,
    pub reasoning_format: ReasoningFormat,
    pub grammar: String,
    pub add_generation_prompt: bool,
    pub continue_final_message: ChatContinuation,
    pub continue_msg: ChatMsg,
    pub enable_thinking: bool,
    pub now: i64,
    pub extra_context: Json,
    pub add_bos: bool,
    pub add_eos: bool,
    /// `is_inference` (chat-auto-parser.h:71) — defaulted to true and never
    /// assigned anywhere in the reference; only gpt-oss reads it
    pub is_inference: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        GenerationParams {
            messages: Json::Array(Vec::new()),
            tools: Json::Null,
            tool_choice: ChatToolChoice::Auto,
            json_schema: Json::Null,
            parallel_tool_calls: true,
            reasoning_format: ReasoningFormat::Auto,
            grammar: String::new(),
            add_generation_prompt: false,
            continue_final_message: ChatContinuation::None,
            continue_msg: ChatMsg::default(),
            enable_thinking: true,
            now: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            extra_context: Json::Object(Vec::new()),
            add_bos: false,
            add_eos: false,
            is_inference: true,
        }
    }
}

impl GenerationParams {
    /// `has_continuation()` (chat-auto-parser.h:75-77)
    pub fn has_continuation(&self) -> bool {
        self.continue_final_message != ChatContinuation::None && !self.continue_msg.empty()
    }
}

/// `common_chat_template_direct_apply_impl` (chat.cpp:905-966)
pub(crate) fn template_direct_apply_impl(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
    messages_override: Option<&Json>,
    tools_override: Option<&Json>,
    additional_context: Option<&Json>,
) -> Result<String, String> {
    // messages_override is already built for this template; don't touch it
    let messages = match messages_override {
        Some(m) => m.clone(),
        None => MessagesInpNormalizer::new(tmpl.original_caps()).normalize(&inputs.messages),
    };

    let tools_val = match tools_override {
        Some(t) => mini_jinja::RenderInputs::val_from_json(t),
        None => mini_jinja::RenderInputs::val_from_json(&inputs.tools),
    };

    let mut inputs_ctx = mini_jinja::RenderInputs {
        messages: mini_jinja::RenderInputs::val_from_json(&messages),
        tools: tools_val,
        add_generation_prompt: inputs.add_generation_prompt,
        bos_token: tmpl.bos_token().to_string(),
        eos_token: tmpl.eos_token().to_string(),
        enable_thinking: Some(inputs.enable_thinking),
        extra: Vec::new(),
        now: Some(inputs.now),
    };

    // extra_context / additional_context as extra globals (chat.cpp:925-936)
    let mut extras: Vec<(String, mini_jinja::Val)> = Vec::new();
    if let Json::Object(fields) = &inputs.extra_context {
        for (k, v) in fields {
            extras.push((k.clone(), mini_jinja::RenderInputs::val_from_json(v)));
        }
    }
    if let Some(Json::Object(fields)) = additional_context {
        for (k, v) in fields {
            if let Some(slot) = extras.iter_mut().find(|(n, _)| n == k) {
                slot.1 = mini_jinja::RenderInputs::val_from_json(v);
            } else {
                extras.push((k.clone(), mini_jinja::RenderInputs::val_from_json(v)));
            }
        }
    }
    inputs_ctx.extra = extras;

    let result = mini_jinja::render_inputs(&tmpl.prog, &inputs_ctx)?;

    // chat.cpp:958-964: strip BOS/EOS the tokenizer will re-add
    let mut result = result;
    if inputs.add_bos && result.starts_with(tmpl.bos_token()) {
        result = result[tmpl.bos_token().len()..].to_string();
    }
    if inputs.add_eos && result.ends_with(tmpl.eos_token()) {
        let n = result.len();
        result = result[..n - tmpl.eos_token().len()].to_string();
    }
    Ok(result)
}

/// `common_chat_template_direct_apply` (chat.cpp:968-972)
pub fn template_direct_apply(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<String, String> {
    template_direct_apply_impl(tmpl, inputs, None, None, None)
}

/// `common_chat_template_generation_prompt_impl` (chat.cpp:974-994)
pub(crate) fn template_generation_prompt_impl(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
    messages_override: Option<&Json>,
    tools_override: Option<&Json>,
    additional_context: Option<&Json>,
) -> Result<String, String> {
    let mut params = inputs.clone();
    params.add_generation_prompt = false;
    params.continue_final_message = ChatContinuation::None;
    let no_gen_prompt = template_direct_apply_impl(
        tmpl,
        &params,
        messages_override,
        tools_override,
        additional_context,
    )?;
    params.add_generation_prompt = true;
    let gen_prompt = template_direct_apply_impl(
        tmpl,
        &params,
        messages_override,
        tools_override,
        additional_context,
    )?;

    let mut prefix_len = 0usize;
    let min_size = no_gen_prompt.len().min(gen_prompt.len());
    while prefix_len < min_size
        && no_gen_prompt.as_bytes()[prefix_len] == gen_prompt.as_bytes()[prefix_len]
    {
        prefix_len += 1;
    }
    Ok(gen_prompt[prefix_len..].to_string())
}

/// `common_chat_template_generation_prompt` (chat.cpp:996-1000)
pub fn template_generation_prompt(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<String, String> {
    template_generation_prompt_impl(tmpl, inputs, None, None, None)
}

// ---------------------------------------------------------------------------
// autoparser — differential analysis (chat-diff-analyzer.cpp +
// chat-auto-parser-helpers.cpp:308-362)
// ---------------------------------------------------------------------------

/// analysis sentinel strings (chat-diff-analyzer.cpp:24-34)
pub mod needles {
    pub const FUN_FIRST: &str = "FFF_FIRST_FUN_F";
    pub const FUN_SECOND: &str = "SSS_SECOND_FUN_S";
    pub const ARG_FIRST: &str = "AA_ARG_FST_AA";
    pub const ARG_SECOND: &str = "BB_ARG_SND_BB";
    pub const USER_MSG: &str = "U_USER_MSG Hello END_U";
    pub const USER_MSG_TWO: &str = "V_USER_MSG Hello END_V";
    pub const ASSISTANT_MSG: &str = "A_ASST_MSG I can help END_A";
    pub const THINKING_CONTENT: &str = "REASON_PART I am thinking END_R";
    pub const CALL_ID_001: &str = "call00001";
    pub const CALL_ID_002: &str = "call00002";
    pub const CALL_ID_999: &str = "call99999";
}

/// `reasoning_mode` (chat-auto-parser.h:85-89)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningMode {
    #[default]
    None,
    /// tag-based: <think>…</think> (start can be empty for delimiter style)
    TagBased,
    /// only reason on tool calls
    ToolsOnly,
}

/// `content_mode` (chat-auto-parser.h:104-109)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContentMode {
    #[default]
    Plain,
    AlwaysWrapped,
    WrappedWithReasoning,
}

/// `call_id_position` (chat-auto-parser.h:125-130)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallIdPosition {
    #[default]
    None,
    PreFuncName,
    BetweenFuncAndArgs,
    PostArgs,
}

/// `tool_format` (chat-auto-parser.h:148-153)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolFormat {
    #[default]
    None,
    JsonNative,
    TagWithJson,
    TagWithTagged,
}

/// `tool_format_analysis` (chat-auto-parser.h:174-192). Defaults mirror the
/// C++ member initializers (`function_field = "function"`, `name_field =
/// "name"`, `args_field = "arguments"`).
#[derive(Clone, Debug)]
pub struct ToolFormatAnalysis {
    pub mode: ToolFormat,
    pub section_start: String,  // e.g. "<tool_call>", "[TOOL_CALLS]", ""
    pub section_end: String,    // e.g. "</tool_call>", ""
    pub per_call_start: String, // e.g. "<|tool_call_begin|>", ""
    pub per_call_end: String,   // e.g. "<|tool_call_end|>", ""
    pub fun_name_is_key: bool,  // { "<funname>": { … } }
    pub tools_array_wrapped: bool,
    pub openai_wrapper_trigger: bool,
    pub function_field: String,
    pub name_field: String,
    pub args_field: String,
    pub id_field: String,
    pub gen_id_field: String,
    pub parameter_order: Vec<String>,
}

impl Default for ToolFormatAnalysis {
    fn default() -> Self {
        ToolFormatAnalysis {
            mode: ToolFormat::None,
            section_start: String::new(),
            section_end: String::new(),
            per_call_start: String::new(),
            per_call_end: String::new(),
            fun_name_is_key: false,
            tools_array_wrapped: false,
            openai_wrapper_trigger: false,
            function_field: "function".to_string(),
            name_field: "name".to_string(),
            args_field: "arguments".to_string(),
            id_field: String::new(),
            gen_id_field: String::new(),
            parameter_order: Vec::new(),
        }
    }
}

/// `tool_function_analysis` (chat-auto-parser.h:194-199)
#[derive(Clone, Debug, Default)]
pub struct ToolFunctionAnalysis {
    pub name_prefix: String,    // e.g. "<function=", "\"name\": \"", "functions."
    pub name_suffix: String,    // e.g. ">", "\"", ":0"
    pub args_separator: String, // e.g. "<tool_sep>"
    pub close: String,          // e.g. "</function>"
}

/// `tool_arguments_analysis` (chat-auto-parser.h:201-210)
#[derive(Clone, Debug, Default)]
pub struct ToolArgumentsAnalysis {
    pub start: String,        // e.g. "<|tool_call_argument_begin|>", "<args>"
    pub end: String,          // e.g. "<|tool_call_argument_end|>", "</args>"
    pub name_prefix: String,  // e.g. "<param=", "<arg_key>"
    pub name_suffix: String,  // e.g. ">", "</arg_key>"
    pub value_prefix: String, // e.g. "<arg_value>"
    pub value_suffix: String, // e.g. "</arg_value>"
    pub separator: String,    // e.g. "\n"
    pub tolerate_intertag_whitespace: bool,
}

/// `tool_id_analysis` (chat-auto-parser.h:212-217)
#[derive(Clone, Debug, Default)]
pub struct ToolIdAnalysis {
    pub pos: CallIdPosition,
    pub prefix: String,
    pub suffix: String,
}

/// `analyze_reasoning` (chat-auto-parser.h:256-277)
#[derive(Clone, Debug, Default)]
pub struct AnalyzeReasoning {
    pub mode: ReasoningMode,
    pub start: String,
    pub end: String,
}

/// `analyze_content` (chat-auto-parser.h:283-298)
#[derive(Clone, Debug, Default)]
pub struct AnalyzeContent {
    pub mode: ContentMode,
    pub start: String,
    pub end: String,
}

impl AnalyzeContent {
    /// `is_always_wrapped()` (chat-diff-analyzer.cpp:764-766)
    pub fn is_always_wrapped(&self) -> bool {
        self.mode == ContentMode::AlwaysWrapped && !self.start.is_empty() && !self.end.is_empty()
    }
}

/// `analyze_tools` (chat-auto-parser.h:304-375)
#[derive(Clone, Debug, Default)]
pub struct AnalyzeTools {
    pub format: ToolFormatAnalysis,
    pub function: ToolFunctionAnalysis,
    pub arguments: ToolArgumentsAnalysis,
    pub call_id: ToolIdAnalysis,
}

/// `template_params` (chat-auto-parser.h:22-28)
#[derive(Clone)]
pub struct TemplateParams {
    pub messages: Json,
    pub tools: Json,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub extra_context: Option<Json>,
}

impl Default for TemplateParams {
    fn default() -> Self {
        TemplateParams {
            messages: Json::Array(Vec::new()),
            tools: Json::Null,
            add_generation_prompt: false,
            enable_thinking: true,
            extra_context: None,
        }
    }
}

/// `compare_variants_result` (chat-auto-parser.h:42-46)
pub struct CompareVariantsResult {
    pub diff: DiffSplit,
    pub output_a: String,
    pub output_b: String,
}

/// `apply_template` (chat-auto-parser-helpers.cpp:310-330): render, mapping
/// failures to the `#**ERROR**#` sentinel. `now` is pinned to 0 — the
/// reference re-renders with the wall clock, which cancels out of every diff.
fn apply_template(tmpl: &ChatTemplate, params: &TemplateParams) -> String {
    const ERR_TMPL: &str = "#**ERROR**#";
    let mut tmpl_params = GenerationParams {
        messages: params.messages.clone(),
        tools: params.tools.clone(),
        add_generation_prompt: params.add_generation_prompt,
        enable_thinking: params.enable_thinking,
        now: 0,
        ..GenerationParams::default()
    };

    let mut extra = params
        .extra_context
        .clone()
        .unwrap_or(Json::Object(Vec::new()));
    if !extra.is_object() {
        extra = Json::Object(Vec::new());
    }
    extra.set("enable_thinking", Json::Bool(params.enable_thinking));
    tmpl_params.extra_context = extra;

    match template_direct_apply(tmpl, &tmpl_params) {
        Ok(s) => s,
        Err(_) => ERR_TMPL.to_string(),
    }
}

/// `compare_variants` (chat-auto-parser-helpers.cpp:332-360)
fn compare_variants(
    tmpl: &ChatTemplate,
    params_a: &TemplateParams,
    params_modifier: &dyn Fn(&mut TemplateParams),
) -> Option<CompareVariantsResult> {
    // Create variant B by copying A
    let mut params_b = params_a.clone();
    params_modifier(&mut params_b);

    // Apply template to both variants
    let output_a = apply_template(tmpl, params_a);
    let output_b = apply_template(tmpl, &params_b);

    // Check for template application failures
    const ERR_TMPL: &str = "#**ERROR**#";
    if output_a == ERR_TMPL || output_b == ERR_TMPL {
        return None;
    }

    let diff = calculate_diff_split(&output_a, &output_b);
    Some(CompareVariantsResult {
        diff,
        output_a,
        output_b,
    })
}

// -- synthetic conversation fixtures (chat-diff-analyzer.cpp:208-248) --------

fn analysis_tools_json() -> Json {
    let schema = Json::parse(&format!(
        r#"{{"type":"object","properties":{{"{}":{{"type":"string","description":"First argument"}},"{}":{{"type":"string","description":"Second argument"}}}},"required":[]}}"#,
        needles::ARG_FIRST,
        needles::ARG_SECOND
    ))
    .unwrap();
    Json::Array(vec![
        Json::Object(vec![
            ("type".to_string(), Json::String("function".to_string())),
            (
                "function".to_string(),
                Json::Object(vec![
                    (
                        "name".to_string(),
                        Json::String(needles::FUN_FIRST.to_string()),
                    ),
                    (
                        "description".to_string(),
                        Json::String("Test function foo".to_string()),
                    ),
                    ("parameters".to_string(), schema.clone()),
                ]),
            ),
        ]),
        Json::Object(vec![
            ("type".to_string(), Json::String("function".to_string())),
            (
                "function".to_string(),
                Json::Object(vec![
                    (
                        "name".to_string(),
                        Json::String(needles::FUN_SECOND.to_string()),
                    ),
                    (
                        "description".to_string(),
                        Json::String("Test function bar".to_string()),
                    ),
                    ("parameters".to_string(), schema),
                ]),
            ),
        ]),
    ])
}

fn user_msg_json(content: &str) -> Json {
    Json::Object(vec![
        ("role".to_string(), Json::String("user".to_string())),
        ("content".to_string(), Json::String(content.to_string())),
    ])
}

/// `build_tool_call` (chat-diff-analyzer.cpp:230-236)
fn build_tool_call(name: &str, args: Json, id: &str) -> Json {
    Json::Object(vec![
        ("id".to_string(), Json::String(id.to_string())),
        ("type".to_string(), Json::String("function".to_string())),
        (
            "function".to_string(),
            Json::Object(vec![
                ("name".to_string(), Json::String(name.to_string())),
                ("arguments".to_string(), args),
            ]),
        ),
    ])
}

fn args_xy() -> Json {
    Json::Object(vec![
        (
            needles::ARG_FIRST.to_string(),
            Json::String("XXXX".to_string()),
        ),
        (
            needles::ARG_SECOND.to_string(),
            Json::String("YYYY".to_string()),
        ),
    ])
}

fn assistant_msg_json(content: &str) -> Json {
    Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        ("content".to_string(), Json::String(content.to_string())),
    ])
}

fn assistant_with_tool_calls_json(calls: Vec<Json>) -> Json {
    Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        ("content".to_string(), Json::String(String::new())),
        ("tool_calls".to_string(), Json::Array(calls)),
    ])
}

/// `struct autoparser` (chat-auto-parser.h:380-409)
pub struct Autoparser {
    pub jinja_caps: JinjaCaps,
    pub user_start: String,
    pub assistant_start: String,
    pub reasoning: AnalyzeReasoning,
    pub content: AnalyzeContent,
    pub tools: AnalyzeTools,
    pub analysis_complete: bool,
    /// Preserved tokens for the tokenizer (union of all non-empty markers)
    pub preserved_tokens: Vec<String>,
    /// literal stop strings caught however tokenized
    pub additional_stops: Vec<String>,
}

impl Default for Autoparser {
    fn default() -> Self {
        Autoparser {
            jinja_caps: JinjaCaps::default(),
            user_start: String::new(),
            assistant_start: String::new(),
            reasoning: AnalyzeReasoning::default(),
            content: AnalyzeContent::default(),
            tools: AnalyzeTools::default(),
            analysis_complete: false,
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
        }
    }
}

impl Autoparser {
    /// `analyze_template` (chat-diff-analyzer.cpp:257-311)
    pub fn analyze_template(&mut self, tmpl: &ChatTemplate) {
        self.jinja_caps = *tmpl.original_caps();
        self.reasoning = analyze_reasoning_of(tmpl, self.jinja_caps.supports_tool_calls);
        self.content = analyze_content_of(tmpl, &self.reasoning);
        self.tools = if self.jinja_caps.supports_tool_calls {
            analyze_tools_of(tmpl, &self.jinja_caps, &self.reasoning)
        } else {
            AnalyzeTools::default()
        };
        self.assistant_start = detect_assistant_start_marker(tmpl, &self.reasoning);
        self.user_start = detect_user_start_marker(tmpl);
        self.collect_preserved_tokens();
        apply_workarounds(tmpl, self);
        self.analysis_complete = true;
    }

    /// `collect_preserved_tokens` (chat-diff-analyzer.cpp:313-345)
    fn collect_preserved_tokens(&mut self) {
        let mut tokens: Vec<String> = Vec::new();
        let mut add_token = |org_token: &str| {
            let token = trim_whitespace(org_token);
            if !token.is_empty() && !tokens.iter().any(|t| t == token) {
                tokens.push(token.to_string());
            }
        };
        add_token(&self.reasoning.start);
        add_token(&self.reasoning.end);
        add_token(&self.content.start);
        add_token(&self.content.end);
        add_token(&self.tools.format.section_start);
        add_token(&self.tools.format.section_end);
        add_token(&self.tools.format.per_call_start);
        add_token(&self.tools.format.per_call_end);
        add_token(&self.tools.function.name_prefix);
        add_token(&self.tools.function.name_suffix);
        add_token(&self.tools.function.args_separator);
        add_token(&self.tools.function.close);
        add_token(&self.tools.arguments.start);
        add_token(&self.tools.arguments.end);
        add_token(&self.tools.arguments.name_prefix);
        add_token(&self.tools.arguments.name_suffix);
        add_token(&self.tools.arguments.separator);
        add_token(&self.tools.arguments.value_prefix);
        add_token(&self.tools.arguments.value_suffix);
        add_token(&self.tools.call_id.prefix);
        add_token(&self.tools.call_id.suffix);
        self.preserved_tokens = tokens;
    }
}

/// `detect_assistant_start_marker` (chat-diff-analyzer.cpp:347-387)
fn detect_assistant_start_marker(tmpl: &ChatTemplate, reasoning: &AnalyzeReasoning) -> String {
    let user = user_msg_json(needles::USER_MSG);
    let assistant_no_reasoning = assistant_msg_json(needles::ASSISTANT_MSG);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone()]),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), assistant_no_reasoning.clone()]);
    });

    let Some(comparison) = comparison else {
        return String::new();
    };

    let usermsg = comparison.diff.right;
    if !usermsg.contains(needles::ASSISTANT_MSG) {
        // "Did not find assistant message in assistant message block, skipping detection"
    }

    let mut ast_prefix = usermsg[..usermsg
        .find(needles::ASSISTANT_MSG)
        .unwrap_or(usermsg.len())]
        .to_string();
    let rs = trim_whitespace(&reasoning.start).to_string();
    let re = trim_whitespace(&reasoning.end).to_string();
    if !rs.is_empty() {
        if let Some(pos) = ast_prefix.find(&rs) {
            ast_prefix = ast_prefix[..pos].to_string();
        }
    }
    if !re.is_empty() {
        if let Some(pos) = ast_prefix.find(&re) {
            ast_prefix = ast_prefix[..pos].to_string();
        }
    }
    trim_whitespace(&ast_prefix).to_string()
}

/// `detect_user_start_marker` (chat-diff-analyzer.cpp:389-459)
fn detect_user_start_marker(tmpl: &ChatTemplate) -> String {
    let user = user_msg_json(needles::USER_MSG);
    let assistant = assistant_msg_json(needles::ASSISTANT_MSG);
    let user_two = user_msg_json(needles::USER_MSG_TWO);

    let params = TemplateParams {
        messages: Json::Array(Vec::new()),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let mut comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone()]);
    });

    if comparison.is_none() {
        // unsupported empty messages — try the reserve variant
        let params = TemplateParams {
            messages: Json::Array(vec![user_two.clone(), assistant.clone()]),
            add_generation_prompt: false,
            enable_thinking: true,
            ..TemplateParams::default()
        };
        comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
            p.messages = Json::Array(vec![user_two.clone(), assistant.clone(), user.clone()]);
        });
        if comparison.is_none() {
            return String::new();
        }
    }
    let comparison = comparison.unwrap();

    let mut usermsg = comparison.diff.right;
    if !usermsg.contains(needles::USER_MSG) {
        // "Did not find user message in user message block, aborting detection"
    }
    if let Some(pos) = usermsg.find(needles::ASSISTANT_MSG) {
        usermsg = usermsg[pos + needles::ASSISTANT_MSG.len()..].to_string();
    }

    let candidate = usermsg[..usermsg.find(needles::USER_MSG).unwrap_or(usermsg.len())].to_string();
    let candidate_split = segmentize_markers(&candidate);
    let mut result = String::new();
    let mut encountered_marker = false;
    for mrk in &candidate_split {
        let lower_mrk = mrk.value.to_lowercase();
        // heuristic to weed out potential end markers, but only at the start
        if mrk.ty == SegmentType::Marker
            && !encountered_marker
            && (lower_mrk.contains("end") || lower_mrk.contains("close"))
        {
            continue;
        }
        if mrk.ty == SegmentType::Text
            && !encountered_marker
            && trim_whitespace(&mrk.value).is_empty()
        {
            continue;
        }
        encountered_marker |= mrk.ty == SegmentType::Marker;
        result += &mrk.value;
    }
    trim_whitespace(&result).to_string()
}

/// `analyze_reasoning` constructor (chat-diff-analyzer.cpp:461-471)
fn analyze_reasoning_of(tmpl: &ChatTemplate, supports_tools: bool) -> AnalyzeReasoning {
    let mut reasoning = AnalyzeReasoning::default();
    compare_reasoning_presence(tmpl, &mut reasoning);
    compare_thinking_enabled(tmpl, &mut reasoning);
    if supports_tools {
        compare_reasoning_scope(tmpl, &mut reasoning);
    }
    reasoning
}

/// `compare_reasoning_presence` (chat-diff-analyzer.cpp:473-530)
fn compare_reasoning_presence(tmpl: &ChatTemplate, reasoning: &mut AnalyzeReasoning) {
    let user = user_msg_json(needles::USER_MSG);
    let assistant_no_reasoning = assistant_msg_json(needles::ASSISTANT_MSG);
    let assistant_with_reasoning = Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        (
            "content".to_string(),
            Json::String(needles::ASSISTANT_MSG.to_string()),
        ),
        (
            "reasoning_content".to_string(),
            Json::String(needles::THINKING_CONTENT.to_string()),
        ),
    ]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), assistant_no_reasoning]),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), assistant_with_reasoning.clone()]);
    });
    let Some(comparison) = comparison else { return };

    let diff = &comparison.diff;
    let reasoning_content = needles::THINKING_CONTENT;

    if !diff.right.is_empty() && diff.right.contains(reasoning_content) {
        // chat-diff-analyzer.cpp:508-513
        let parser_delimiter = build_tagged_peg_parser(|p| {
            let lit = p.literal(reasoning_content);
            let sp = p.space();
            let marker = p.marker();
            let msp = p.space();
            let tagged = {
                let inner = p.sequence(&[marker, msp]);
                p.tag("post", inner)
            };
            let opt = p.optional(tagged);
            let rest = p.rest();
            p.sequence(&[lit, sp, opt, rest])
        });
        let parser_wrapped = build_tagged_peg_parser(|p| {
            let marker = p.marker();
            let msp = p.space();
            let pre = {
                let inner = p.sequence(&[marker, msp]);
                p.tag("pre", inner)
            };
            let lit = p.literal(reasoning_content);
            let sp1 = p.space();
            let marker2 = p.marker();
            let sp2 = p.space();
            let post = {
                let inner = p.sequence(&[sp1, marker2, sp2]);
                p.tag("post", inner)
            };
            let rest = p.rest();
            p.sequence(&[pre, lit, post, rest])
        });
        // try the more aggressive parse first, fall back to the delimiter one
        let mut result = parser_wrapped.parse_anywhere_and_extract(&comparison.output_b);
        if !result.result.success() {
            result = parser_delimiter.parse_anywhere_and_extract(&comparison.output_b);
        }
        if result.result.success() {
            let pre = result.tags.get("pre").cloned().unwrap_or_default();
            let post = result.tags.get("post").cloned().unwrap_or_default();
            if !pre.is_empty() && !post.is_empty() {
                reasoning.mode = ReasoningMode::TagBased;
                reasoning.start = pre;
                reasoning.end = post;
            } else if !post.is_empty() {
                reasoning.mode = ReasoningMode::TagBased;
                reasoning.end = post;
            }
        }
    }
}

/// `compare_thinking_enabled` (chat-diff-analyzer.cpp:532-614)
fn compare_thinking_enabled(tmpl: &ChatTemplate, reasoning: &mut AnalyzeReasoning) {
    let user = user_msg_json(needles::USER_MSG);

    let params = TemplateParams {
        messages: Json::Array(vec![user]),
        add_generation_prompt: true,
        enable_thinking: false,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.enable_thinking = true
    });
    let Some(comparison) = comparison else { return };

    let diff = &comparison.diff;
    let left_trimmed = trim_whitespace(&diff.left).to_string();
    let right_trimmed = trim_whitespace(&diff.right).to_string();

    if left_trimmed.is_empty() && !diff.right.is_empty() {
        if !right_trimmed.is_empty() && string_ends_with(&comparison.output_b, &right_trimmed) {
            if reasoning.start.is_empty() {
                reasoning.start = diff.right.clone();
                reasoning.mode = ReasoningMode::TagBased;
            }
        }
    } else if right_trimmed.is_empty() && !diff.left.is_empty() {
        if !left_trimmed.is_empty() && string_ends_with(&comparison.output_a, &left_trimmed) {
            if reasoning.end.is_empty() {
                let seg = prune_whitespace_segments(&segmentize_markers(&comparison.output_a));
                if seg.len() >= 2
                    && seg[seg.len() - 1].value == left_trimmed
                    && seg[seg.len() - 2].ty == SegmentType::Marker
                {
                    reasoning.start = seg[seg.len() - 2].value.clone();
                }
                reasoning.end = diff.left.clone();
                reasoning.mode = ReasoningMode::TagBased;
            }
        }
    } else if !left_trimmed.is_empty() && !right_trimmed.is_empty() {
        // Full-output diff is noisy; tail-anchor to find appended markers
        // (chat-diff-analyzer.cpp:573-609)
        const ANCHOR_LEN: usize = 64;
        for dir in 0..2 {
            let (base, extended) = if dir == 0 {
                (&comparison.output_b, &comparison.output_a)
            } else {
                (&comparison.output_a, &comparison.output_b)
            };
            let len = base.len().min(ANCHOR_LEN);
            if len == 0 || len > base.len() {
                continue;
            }
            let anchor = &base[base.len() - len..];
            let Some(pos) = extended.rfind(anchor) else {
                continue;
            };
            if pos + len >= extended.len() {
                continue;
            }
            let extra = trim_whitespace(&extended[pos + len..]).to_string();
            if extra.is_empty() {
                continue;
            }
            let seg = prune_whitespace_segments(&segmentize_markers(&extra));
            if seg.len() == 2
                && seg[0].ty == SegmentType::Marker
                && seg[1].ty == SegmentType::Marker
            {
                if reasoning.start.is_empty() {
                    reasoning.start = seg[0].value.clone();
                }
                if reasoning.end.is_empty() {
                    reasoning.end = seg[1].value.clone();
                }
                reasoning.mode = ReasoningMode::TagBased;
                break;
            }
        }
    }

    if reasoning.mode == ReasoningMode::None
        && reasoning.start.is_empty()
        && !reasoning.end.is_empty()
    {
        reasoning.mode = ReasoningMode::TagBased;
    }
}

/// `compare_reasoning_scope` (chat-diff-analyzer.cpp:616-675)
fn compare_reasoning_scope(tmpl: &ChatTemplate, reasoning: &mut AnalyzeReasoning) {
    let user = user_msg_json(needles::USER_MSG);
    let assistant_reasoning_content = Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        (
            "content".to_string(),
            Json::String(needles::ASSISTANT_MSG.to_string()),
        ),
        (
            "reasoning_content".to_string(),
            Json::String(needles::THINKING_CONTENT.to_string()),
        ),
    ]);
    let assistant_reasoning_tools = Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        ("content".to_string(), Json::Null),
        (
            "reasoning_content".to_string(),
            Json::String(needles::THINKING_CONTENT.to_string()),
        ),
        (
            "tool_calls".to_string(),
            Json::Array(vec![build_tool_call(
                needles::FUN_FIRST,
                Json::Object(vec![
                    (
                        needles::ARG_FIRST.to_string(),
                        Json::String("VVVV".to_string()),
                    ),
                    (
                        needles::ARG_SECOND.to_string(),
                        Json::String("XXXX".to_string()),
                    ),
                ]),
                needles::CALL_ID_001,
            )]),
        ),
    ]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), assistant_reasoning_content]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), assistant_reasoning_tools.clone()]);
    });
    let Some(comparison) = comparison else { return };

    let reasoning_content = needles::THINKING_CONTENT;
    let reasoning_in_a = comparison.output_a.contains(reasoning_content);
    let reasoning_in_b = comparison.output_b.contains(reasoning_content);

    if !reasoning_in_a && reasoning_in_b {
        reasoning.mode = ReasoningMode::ToolsOnly;

        let parser_wrapped = build_tagged_peg_parser(|p| {
            let marker = p.marker();
            let sp = p.space();
            let pre = {
                let inner = p.sequence(&[marker, sp]);
                p.tag("pre", inner)
            };
            let lit = p.literal(reasoning_content);
            let sp1 = p.space();
            let marker2 = p.marker();
            let sp2 = p.space();
            let post = {
                let inner = p.sequence(&[marker2, sp2]);
                p.tag("post", inner)
            };
            p.sequence(&[pre, lit, sp1, post])
        });
        let result = parser_wrapped.parse_anywhere_and_extract(&comparison.output_b);
        if result.result.success() {
            reasoning.start = result.tags.get("pre").cloned().unwrap_or_default();
            reasoning.end = result.tags.get("post").cloned().unwrap_or_default();
        } else {
            let parser_delimiter = build_tagged_peg_parser(|p| {
                let lit = p.literal(reasoning_content);
                let sp = p.space();
                let marker = p.marker();
                let msp = p.space();
                let post = {
                    let inner = p.sequence(&[marker, msp]);
                    p.tag("post", inner)
                };
                let opt = p.optional(post);
                p.sequence(&[lit, sp, opt])
            });
            let result = parser_delimiter.parse_anywhere_and_extract(&comparison.output_b);
            if result.result.success() {
                reasoning.end = result.tags.get("post").cloned().unwrap_or_default();
            } else {
                // "Unable to extract reasoning markers, falling back to NONE"
                reasoning.mode = ReasoningMode::None;
            }
        }
    }
}

/// `analyze_content` constructor (chat-diff-analyzer.cpp:677-762)
fn analyze_content_of(tmpl: &ChatTemplate, reasoning: &AnalyzeReasoning) -> AnalyzeContent {
    let mut content = AnalyzeContent::default();

    let user = user_msg_json(needles::USER_MSG);
    let assistant_content_only = assistant_msg_json(needles::ASSISTANT_MSG);
    let assistant_with_tools = assistant_with_tool_calls_json(vec![build_tool_call(
        "test_func",
        Json::Object(vec![(
            "arg1".to_string(),
            Json::String("value1".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);
    let assistant_with_reasoning = Json::Object(vec![
        ("role".to_string(), Json::String("assistant".to_string())),
        ("content".to_string(), Json::String(String::new())),
        (
            "reasoning_content".to_string(),
            Json::String(needles::THINKING_CONTENT.to_string()),
        ),
    ]);

    let params_content_only = TemplateParams {
        messages: Json::Array(vec![user.clone(), assistant_content_only]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison_with_tools =
        compare_variants(tmpl, &params_content_only, &|p: &mut TemplateParams| {
            p.messages = Json::Array(vec![user.clone(), assistant_with_tools.clone()]);
        });
    let comparison_with_reasoning =
        compare_variants(tmpl, &params_content_only, &|p: &mut TemplateParams| {
            p.messages = Json::Array(vec![user.clone(), assistant_with_reasoning.clone()]);
        });

    let (Some(comp_tools), Some(comp_reasoning)) =
        (comparison_with_tools, comparison_with_reasoning)
    else {
        return content;
    };

    let diff_tools = &comp_tools.diff;
    let diff_reasoning = &comp_reasoning.diff;
    let response = needles::ASSISTANT_MSG;

    let mut found_plain_content = false;
    if trim_whitespace(&diff_tools.left) == response {
        // chat-diff-analyzer.cpp:724-727 — the literal is the runtime
        // diff_reasoning.left, so build the parser per value
        let parser = build_content_probe_parser(&diff_reasoning.left);
        if parser
            .parse_and_extract(&diff_reasoning.left, PARSE_FLAG_NONE)
            .result
            .success()
        {
            // only the content text in the diff — no markers
            content.mode = ContentMode::Plain;
            found_plain_content = true;
        } else if reasoning.mode != ReasoningMode::None && !reasoning.end.is_empty() {
            let end_tag = reasoning.end.clone();
            let post_reasoning_parser = build_tagged_peg_parser(move |p| {
                let lit = p.literal(&end_tag);
                let sp = p.space();
                let resp = p.literal(response);
                p.sequence(&[lit, sp, resp])
            });
            if post_reasoning_parser
                .parse_anywhere_and_extract(&diff_reasoning.left)
                .result
                .success()
            {
                content.mode = ContentMode::Plain;
                found_plain_content = true;
            }
        }
    }
    if !found_plain_content {
        let mut rdiff = diff_reasoning.left.clone();
        if !reasoning.end.is_empty() {
            if let Some(pos) = rdiff.find(&reasoning.end) {
                rdiff = rdiff[pos + reasoning.end.len()..].to_string();
            }
        }
        // Take the more promising diff
        let pure_content = if rdiff.len() > diff_tools.left.len() {
            rdiff
        } else {
            diff_tools.left.clone()
        };
        let parser_wrapped = build_tagged_peg_parser(|p| {
            let marker = p.marker();
            let msp = p.space();
            let pre = {
                let inner = p.sequence(&[marker, msp]);
                p.tag("pre", inner)
            };
            let lit = p.literal(response);
            let sp = p.space();
            let marker2 = p.marker();
            let sp2 = p.space();
            let post = {
                let inner = p.sequence(&[marker2, sp2]);
                p.tag("post", inner)
            };
            let rest = p.rest();
            p.sequence(&[pre, lit, sp, post, rest])
        });
        let result = parser_wrapped.parse_anywhere_and_extract(&pure_content);
        content.start = result.tags.get("pre").cloned().unwrap_or_default();
        content.end = result.tags.get("post").cloned().unwrap_or_default();
        // TODO(upstream): WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if !content.start.is_empty() || !content.end.is_empty() {
        content.mode = ContentMode::AlwaysWrapped;
        // TODO(upstream): END_DELIMITED content mode
    }
    content
}

/// the `p.space() + diff_reasoning.left + p.space() + optional(marker) + space
/// + end` probe (chat-diff-analyzer.cpp:724-727) — literal depends on the diff
fn build_content_probe_parser(reasoning_left: &str) -> TaggedPegParser {
    let lit_s = reasoning_left.to_string();
    build_tagged_peg_parser(move |p| {
        let sp1 = p.space();
        let lit = p.literal(&lit_s);
        let sp2 = p.space();
        let m = p.marker();
        let marker = p.optional(m);
        let sp3 = p.space();
        let end = p.end();
        p.sequence(&[sp1, lit, sp2, marker, sp3, end])
    })
}

// ---------------------------------------------------------------------------
// tool analysis (chat-diff-analyzer.cpp:768-1633)
// ---------------------------------------------------------------------------

/// `analyze_tools` constructor (chat-diff-analyzer.cpp:768-791)
fn analyze_tools_of(
    tmpl: &ChatTemplate,
    caps: &JinjaCaps,
    reasoning: &AnalyzeReasoning,
) -> AnalyzeTools {
    let mut tools = AnalyzeTools::default();
    analyze_tool_calls(
        tmpl,
        &mut tools,
        reasoning,
        caps.supports_parallel_tool_calls,
    );

    if tools.format.mode != ToolFormat::None && tools.format.mode != ToolFormat::JsonNative {
        if caps.supports_parallel_tool_calls {
            check_per_call_markers(tmpl, &mut tools);
        }
        extract_function_markers(tmpl, &mut tools);
        if tools.format.mode == ToolFormat::TagWithTagged {
            extract_argument_name_markers(tmpl, &mut tools);
            extract_argument_value_markers(tmpl, &mut tools);
        }
        extract_argument_separator(tmpl, &mut tools);
        extract_args_markers(tmpl, &mut tools);
        extract_call_id_markers(tmpl, &mut tools);
    }
    tools
}

/// `analyze_tool_calls` (chat-diff-analyzer.cpp:793-828)
fn analyze_tool_calls(
    tmpl: &ChatTemplate,
    tools: &mut AnalyzeTools,
    reasoning: &AnalyzeReasoning,
    supports_parallel_tool_calls: bool,
) {
    let user = user_msg_json(needles::USER_MSG);
    let assistant_no_tools = assistant_msg_json(needles::ASSISTANT_MSG);
    let assistant_with_tools = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        args_xy(),
        needles::CALL_ID_001,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), assistant_no_tools]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), assistant_with_tools.clone()]);
    });
    let Some(comparison) = comparison else { return };

    let tool_section = comparison.diff.right;
    if tool_section.is_empty() {
        return;
    }

    analyze_tool_call_format(
        tmpl,
        tools,
        &tool_section,
        needles::FUN_FIRST,
        needles::ARG_FIRST,
        reasoning,
        supports_parallel_tool_calls,
    );
}

/// `analyze_tool_call_format` (chat-diff-analyzer.cpp:830-886)
fn analyze_tool_call_format(
    tmpl: &ChatTemplate,
    tools: &mut AnalyzeTools,
    haystack: &str,
    fun_name_needle: &str,
    arg_name_needle: &str,
    reasoning: &AnalyzeReasoning,
    supports_parallel_tool_calls: bool,
) {
    if fun_name_needle.is_empty() || arg_name_needle.is_empty() || haystack.is_empty() {
        return;
    }

    // in_json_haystack (chat-diff-analyzer.cpp:839-846)
    let in_json_haystack = |needle: &str| -> bool {
        let parser = build_in_json_parser(needle);
        let result = parser.parse_anywhere_and_extract(haystack);
        result.result.success()
    };

    let fun_quote = in_json_haystack(fun_name_needle);
    let arg_quote = in_json_haystack(arg_name_needle);

    if fun_quote {
        // no need to check further, we're in JSON land
        tools.format.mode = ToolFormat::JsonNative;
    } else if arg_quote {
        tools.format.mode = ToolFormat::TagWithJson;
    } else {
        tools.format.mode = ToolFormat::TagWithTagged;
    }

    // first, remove any reasoning markers
    let mut clean_haystack = haystack.to_string();
    if !reasoning.start.is_empty() {
        if let Some(pos) = haystack.find(&reasoning.start) {
            clean_haystack = format!(
                "{}{}",
                &haystack[..pos],
                &haystack[pos + reasoning.start.len()..]
            );
        }
    }
    if !reasoning.end.is_empty() {
        if let Some(pos) = clean_haystack.find(&reasoning.end) {
            clean_haystack = format!(
                "{}{}",
                &clean_haystack[..pos],
                &clean_haystack[pos + reasoning.end.len()..]
            );
        }
    }

    if tools.format.mode == ToolFormat::JsonNative {
        analyze_tool_call_format_json_native(
            tools,
            &clean_haystack,
            fun_name_needle,
            arg_name_needle,
        );
        if supports_parallel_tool_calls {
            analyze_json_native_parallel_calls(tmpl, tools);
        }
    } else {
        analyze_tool_call_format_non_json(tools, &clean_haystack, fun_name_needle);
    }
    // always relax whitespace requirements on ending markers
    tools.format.section_end = trim_whitespace(&tools.format.section_end).to_string();
    tools.format.per_call_end = trim_whitespace(&tools.format.per_call_end).to_string();
}

/// the `in_json_haystack` parser with the runtime needle (chat-diff-analyzer.cpp:840-845)
fn build_in_json_parser(needle: &str) -> TaggedPegParser {
    let needle = needle.to_string();
    build_tagged_peg_parser(move |p| {
        let lb = p.literal("{");
        let colon = p.literal(":");
        let choice1 = p.choice(&[lb, colon]);
        let q = p.literal("\"");
        let n = p.literal(&needle);
        let q2 = p.literal("\"");
        let dq = {
            let seq = p.sequence(&[q, n, q2]);
            p.tag("dq", seq)
        };
        let choice2 = p.choice(&[dq]);
        p.spaced(choice1, choice2)
    })
}

/// `analyze_json_native_parallel_calls` (chat-diff-analyzer.cpp:888-922)
fn analyze_json_native_parallel_calls(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let one = build_tool_call(needles::FUN_FIRST, args_xy(), needles::CALL_ID_001);
    let two = build_tool_call(needles::FUN_SECOND, args_xy(), needles::CALL_ID_002);

    let params = TemplateParams {
        messages: Json::Array(vec![
            user.clone(),
            assistant_with_tool_calls_json(vec![one]),
        ]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![
            user.clone(),
            assistant_with_tool_calls_json(vec![one2(), two.clone()]),
        ]);
    });
    let Some(comparison) = comparison else { return };

    let second_call = comparison.diff.right;
    if !tools.format.section_start.is_empty() && second_call.contains(&tools.format.section_start) {
        tools.format.per_call_start = tools.format.section_start.clone();
        tools.format.per_call_end = tools.format.section_end.clone();
        tools.format.section_start.clear();
        tools.format.section_end.clear();
    }
}

fn one2() -> Json {
    build_tool_call(needles::FUN_FIRST, args_xy(), needles::CALL_ID_001)
}

/// `analyze_tool_call_format_json_native` (chat-diff-analyzer.cpp:924-1001)
fn analyze_tool_call_format_json_native(
    tools: &mut AnalyzeTools,
    clean_haystack: &str,
    fun_name_needle: &str,
    arg_name_needle: &str,
) {
    // we might not have the typical OpenAI tool calling structure
    let json_start = clean_haystack.find('{').map(|v| v as i64).unwrap_or(-1);
    let json_end = clean_haystack.rfind('}').map(|v| v as i64).unwrap_or(-1);
    if json_start < 0 || json_end < json_start {
        return;
    }
    let cut = clean_haystack[json_start as usize..=json_end as usize].to_string();
    let call_struct = match Json::parse(&cut) {
        Ok(v) => v,
        Err(_) => return, // C++ json::parse throws; abort marker analysis
    };

    let register_field =
        |format: &mut ToolFormatAnalysis, prefix: &str, key: &str, value: &Json| {
            let prefixed = |k: &str| {
                if !prefix.is_empty() {
                    format!("{prefix}.{k}")
                } else {
                    k.to_string()
                }
            };
            if value.is_string()
                && value
                    .get_str()
                    .map(|s| s.contains("call0000"))
                    .unwrap_or(false)
            {
                format.id_field = prefixed(key);
            } else if value.is_string() && value.get_str() == Ok(fun_name_needle) {
                format.name_field = prefixed(key);
            } else if value.dump().contains(arg_name_needle) {
                format.args_field = prefixed(key);
            } else if key.contains("id") {
                // heuristics for generated id field
                format.gen_id_field = prefixed(key);
            }
        };

    for (key, el) in call_struct.items() {
        if key == fun_name_needle {
            tools.format.fun_name_is_key = true;
            tools.format.name_field.clear();
            tools.format.args_field.clear();
        } else {
            let is_args_object = el.is_object() && !el.dump().contains(arg_name_needle);
            if is_args_object {
                tools.format.function_field = key.clone();
                for (sub_key, sub_val) in el.items() {
                    register_field(&mut tools.format, &key, &sub_key, sub_val);
                }
            }
            register_field(&mut tools.format, "", &key, el);
        }
    }

    // array-wrapped check (chat-diff-analyzer.cpp:964-973)
    let mut json_start = json_start;
    let mut json_end = json_end;
    {
        let cut_clone = cut.clone();
        let array_parser = build_tagged_peg_parser(move |p| {
            let lb = p.literal("[");
            let sp = p.space();
            let pre = {
                let seq = p.sequence(&[lb, sp]);
                p.tag("pre", seq)
            };
            let lit = p.literal(&cut_clone);
            let sp2 = p.space();
            let rb = p.literal("]");
            let post = {
                let seq = p.sequence(&[sp2, rb]);
                p.tag("post", seq)
            };
            p.sequence(&[pre, lit, post])
        });
        let ar_parse_res = array_parser.parse_anywhere_and_extract(clean_haystack);
        if ar_parse_res.result.success() {
            tools.format.tools_array_wrapped = true;
            json_start -= ar_parse_res.tags.get("pre").map(|s| s.len()).unwrap_or(0) as i64;
            json_end += ar_parse_res.tags.get("post").map(|s| s.len()).unwrap_or(0) as i64;
        }
    }
    let json_end = (json_end + 1).max(0) as usize; // past the closing char
    let json_start = json_start.max(0) as usize;

    // parameter order by first occurrence (chat-diff-analyzer.cpp:976-992)
    let mut located_params: Vec<(usize, String)> = Vec::new();
    for field in [
        &tools.format.name_field,
        &tools.format.args_field,
        &tools.format.id_field,
        &tools.format.gen_id_field,
    ] {
        if !field.is_empty() {
            located_params.push((
                clean_haystack.find(field.as_str()).unwrap_or(usize::MAX),
                field.clone(),
            ));
        }
    }
    located_params.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, field) in located_params {
        tools.format.parameter_order.push(field);
    }

    // extract tool calling markers
    tools.format.section_start = trim_leading_whitespace(&clean_haystack[..json_start]).to_string();
    tools.format.section_end = if json_end <= clean_haystack.len() {
        trim_whitespace(&clean_haystack[json_end..]).to_string()
    } else {
        String::new()
    };
    // avoid duplicate closing brackets when array-wrapped
    if tools.format.tools_array_wrapped && tools.format.section_end == "]" {
        tools.format.section_end.clear();
    }
}

/// `analyze_tool_call_format_non_json` (chat-diff-analyzer.cpp:1003-1045)
fn analyze_tool_call_format_non_json(
    tools: &mut AnalyzeTools,
    clean_haystack: &str,
    fun_name_needle: &str,
) {
    // first: is the function inside a tag or standalone?
    let fun_res =
        build_fun_marker_parser(fun_name_needle).parse_anywhere_and_extract(clean_haystack);
    let mut fun_marker = fun_name_needle.to_string();
    if fun_res.result.success() {
        fun_marker = fun_res
            .tags
            .get("fun_marker")
            .cloned()
            .unwrap_or(fun_marker);
    }

    // consume up to two markers, then everything up to the function marker is
    // the function name prefix
    let per_tool_parser = build_per_tool_parser(&fun_marker);
    let section_parser = build_section_parser(&fun_marker);
    let mut result = per_tool_parser.parse_anywhere_and_extract(clean_haystack);
    let result_end;
    if result.result.success() {
        let double_closer_parser = build_tagged_peg_parser(|p| {
            let m1 = p.marker();
            let sp1 = p.space();
            let call_end = {
                let seq = p.sequence(&[m1, sp1]);
                p.tag("call_end", seq)
            };
            let m2 = p.marker();
            let sp2 = p.space();
            let sec_end = {
                let seq = p.sequence(&[m2, sp2]);
                p.tag("sec_end", seq)
            };
            let end = p.end();
            p.sequence(&[call_end, sec_end, end])
        });
        let rest = result.tags.get("rest").cloned().unwrap_or_default();
        result_end = double_closer_parser.parse_anywhere_and_extract(&rest);
        let fun_pre = fun_res.tags.get("fun_pre").cloned().unwrap_or_default();
        tools.function.name_prefix = format!("{fun_pre}{}", tools.function.name_prefix);
    } else {
        result = section_parser.parse_anywhere_and_extract(clean_haystack);
        let single_closer_parser = build_tagged_peg_parser(|p| {
            let m = p.marker();
            let sp = p.space();
            let sec_end = {
                let seq = p.sequence(&[m, sp]);
                p.tag("sec_end", seq)
            };
            let end = p.end();
            p.sequence(&[sec_end, end])
        });
        let rest = result.tags.get("rest").cloned().unwrap_or_default();
        result_end = single_closer_parser.parse_anywhere_and_extract(&rest);
    }
    tools.format.per_call_start = result.tags.get("call_start").cloned().unwrap_or_default();
    tools.format.per_call_end = result_end.tags.get("call_end").cloned().unwrap_or_default();
    tools.format.section_start = result.tags.get("sec_start").cloned().unwrap_or_default();
    tools.format.section_end = result_end.tags.get("sec_end").cloned().unwrap_or_default();
}

/// the `fun_marker_parser` with the runtime needle (chat-diff-analyzer.cpp:1006-1012)
fn build_fun_marker_parser(fun_name_needle: &str) -> TaggedPegParser {
    let needle = fun_name_needle.to_string();
    build_tagged_peg_parser(move |p| {
        // option 1: <…fun…> tag
        let opt1 = {
            let lt = p.literal("<");
            let needle2 = needle.clone();
            let until = p.until_one_of(&[">", &needle2]);
            let fun_pre = {
                let seq = p.sequence(&[lt, until]);
                p.tag("fun_pre", seq)
            };
            let lit = p.literal(&needle);
            let sp = p.space();
            let lt2 = p.literal("<");
            let nseq = p.sequence(&[sp, lt2]);
            let nsp = p.negate(nseq);
            let until2 = p.until(">");
            let gt = p.literal(">");
            let fun_post = {
                let seq = p.sequence(&[nsp, until2, gt]);
                p.tag("fun_post", seq)
            };
            let sp = p.space();
            p.sequence(&[fun_pre, lit, fun_post, sp])
        };
        // option 2: […] bracket tag
        let opt2 = {
            let lb = p.literal("[");
            let needle2 = needle.clone();
            let until = p.until_one_of(&["]", &needle2]);
            let fun_pre = {
                let seq = p.sequence(&[lb, until]);
                p.tag("fun_pre", seq)
            };
            let lit = p.literal(&needle);
            let sp = p.space();
            let lb2 = p.literal("[");
            let until2 = p.until("]");
            let rb = p.literal("]");
            let iseq = p.sequence(&[sp, lb2, until2, rb]);
            let inner_neg = p.negate(iseq);
            let sp = p.space();
            p.sequence(&[fun_pre, lit, inner_neg, sp])
        };
        let choice = p.choice(&[opt1, opt2]);
        p.tag("fun_marker", choice)
    })
}

/// the `per_tool_parser` (chat-diff-analyzer.cpp:1019-1022)
fn build_per_tool_parser(fun_marker: &str) -> TaggedPegParser {
    let fun_marker = fun_marker.to_string();
    build_tagged_peg_parser(move |p| {
        let m1 = p.marker();
        let sp1 = p.space();
        let sec_start = {
            let seq = p.sequence(&[m1, sp1]);
            p.tag("sec_start", seq)
        };
        let m2 = p.marker();
        let sp2 = p.space();
        let call_start = {
            let seq = p.sequence(&[m2, sp2]);
            p.tag("call_start", seq)
        };
        let until = p.until(&fun_marker);
        let fun_pre = p.tag("fun_pre", until);
        let lit = p.literal(&fun_marker);
        let r = p.rest();
        let rest = p.tag("rest", r);
        p.sequence(&[sec_start, call_start, fun_pre, lit, rest])
    })
}

/// the `section_parser` (chat-diff-analyzer.cpp:1023-1025)
fn build_section_parser(fun_marker: &str) -> TaggedPegParser {
    let fun_marker = fun_marker.to_string();
    build_tagged_peg_parser(move |p| {
        let m = p.marker();
        let sp = p.space();
        let sec_start = {
            let seq = p.sequence(&[m, sp]);
            p.tag("sec_start", seq)
        };
        let lit = p.literal(&fun_marker);
        let r = p.rest();
        let rest = p.tag("rest", r);
        p.sequence(&[sec_start, lit, rest])
    })
}

/// `check_per_call_markers` (chat-diff-analyzer.cpp:1047-1101)
fn check_per_call_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let one = build_tool_call(needles::FUN_FIRST, args_xy(), needles::CALL_ID_001);
    let two = build_tool_call(needles::FUN_SECOND, args_xy(), needles::CALL_ID_002);

    let params = TemplateParams {
        messages: Json::Array(vec![
            user.clone(),
            assistant_with_tool_calls_json(vec![one]),
        ]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let one_vs_two = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![
            user.clone(),
            assistant_with_tool_calls_json(vec![one2(), two.clone()]),
        ]);
    });
    let Some(one_vs_two) = one_vs_two else { return };

    let filter_common_call_part =
        calculate_diff_split(&one_vs_two.diff.suffix, &one_vs_two.diff.right);

    let second_tool_content = trim_leading_whitespace(&filter_common_call_part.right).to_string();
    if !tools.format.section_start.is_empty()
        && second_tool_content.starts_with(&tools.format.section_start)
    {
        tools.format.per_call_start = tools.format.section_start.clone();
        tools.format.per_call_end = tools.format.section_end.clone();
        tools.format.section_start.clear();
        tools.format.section_end.clear();
    }

    if !tools.format.per_call_end.is_empty() {
        let count_occurrences = |haystack: &str, needle: &str| -> usize {
            let mut count = 0;
            let mut from = 0usize;
            while let Some(rel) = haystack[from..].find(needle) {
                count += 1;
                from += rel + needle.len();
            }
            count
        };
        let calls_one = count_occurrences(&one_vs_two.output_a, &tools.format.per_call_end);
        let calls_two = count_occurrences(&one_vs_two.output_b, &tools.format.per_call_end);
        if calls_one > 0 && calls_one == calls_two {
            tools.format.section_end = tools.format.per_call_end.clone();
            tools.format.per_call_end.clear();
        }
    }
}

/// `extract_function_markers` (chat-diff-analyzer.cpp:1103-1244)
fn extract_function_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let assistant_foo = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        args_xy(),
        needles::CALL_ID_001,
    )]);
    let assistant_bar = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_SECOND,
        args_xy(),
        needles::CALL_ID_002,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), assistant_foo]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), assistant_bar.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if diff.left.contains(needles::FUN_FIRST) && diff.right.contains(needles::FUN_SECOND) {
        let prefix_marker = if !tools.format.per_call_start.is_empty() {
            tools.format.per_call_start.clone()
        } else {
            tools.format.section_start.clone()
        };
        if !prefix_marker.is_empty() {
            if let Some(pos) = diff.prefix.rfind(&prefix_marker) {
                tools.function.name_prefix = diff.prefix[pos + prefix_marker.len()..].to_string();
            }
        }

        // name prefix/suffix from diff.left (stop at the next marker boundary)
        let name_parser = build_tagged_peg_parser(|p| {
            let fun_first = p.literal(needles::FUN_FIRST);
            let pre = {
                let until = p.until(needles::FUN_FIRST);
                p.tag("pre", until)
            };
            let mk = p.marker();
            let neg = p.negate(mk);
            let any = p.any();
            let post = {
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                p.tag("post", z)
            };
            p.sequence(&[pre, fun_first, post])
        });
        let name_result = name_parser.parse_and_extract(&diff.left, PARSE_FLAG_NONE);
        if name_result.result.success() {
            tools.function.name_prefix += &name_result.tags.get("pre").cloned().unwrap_or_default();
            tools.function.name_suffix = name_result.tags.get("post").cloned().unwrap_or_default();
        }

        // Extend name_suffix with content from diff.suffix before args begin
        if tools.format.mode == ToolFormat::TagWithJson {
            let suffix_parser = build_tagged_peg_parser(|p| {
                let non_json = {
                    let m = p.marker();
                    let l1 = p.literal("{");
                    let n1 = p.negate(l1);
                    let l2 = p.literal("[");
                    let n2 = p.negate(l2);
                    let any = p.any();
                    let seq = p.sequence(&[n1, n2, any]);
                    p.choice(&[m, seq])
                };
                let after_json = {
                    let mk = p.marker();
                    let neg = p.negate(mk);
                    let any = p.any();
                    let zseq = p.sequence(&[neg, any]);
                    let z = p.zero_or_more(zseq);
                    let m = p.marker();
                    p.sequence(&[z, m])
                };
                let ext = {
                    let z = p.zero_or_more(non_json);
                    p.tag("ext", z)
                };
                p.sequence(&[ext, after_json])
            });
            let suf_result = suffix_parser.parse_and_extract(&diff.suffix, PARSE_FLAG_NONE);
            if suf_result.result.success() {
                tools.function.name_suffix +=
                    &suf_result.tags.get("ext").cloned().unwrap_or_default();
            }
        } else {
            // tagged: name_suffix extends to the first marker (arg marker)
            let suffix_parser = build_tagged_peg_parser(|p| {
                let mk = p.marker();
                let neg = p.negate(mk);
                let any = p.any();
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                p.tag("ext", z)
            });
            let suf_result = suffix_parser.parse_and_extract(&diff.suffix, PARSE_FLAG_NONE);
            if suf_result.result.success() {
                let ext = suf_result.tags.get("ext").cloned().unwrap_or_default();
                tools.function.name_suffix += &ext;

                // args separator: between ext and the first arg marker
                let sep_parser = build_tagged_peg_parser(|p| {
                    let arg_start = {
                        let m = p.marker();
                        let sp = p.space();
                        let a1 = p.literal(needles::ARG_FIRST);
                        let a2 = p.literal(needles::ARG_SECOND);
                        let ch = p.choice(&[a1, a2]);
                        p.sequence(&[m, sp, ch])
                    };
                    let neg = p.negate(arg_start);
                    let any = p.any();
                    let zseq = p.sequence(&[neg, any]);
                    let z = p.zero_or_more(zseq);
                    let sep = p.tag("sep", z);
                    let asx = {
                        let m = p.marker();
                        let sp = p.space();
                        let a1 = p.literal(needles::ARG_FIRST);
                        let a2 = p.literal(needles::ARG_SECOND);
                        let ch = p.choice(&[a1, a2]);
                        p.sequence(&[m, sp, ch])
                    };
                    p.sequence(&[sep, asx])
                });
                let tail = diff.suffix[ext.len()..].to_string();
                let sep_result = sep_parser.parse_and_extract(&tail, PARSE_FLAG_NONE);
                if sep_result.result.success() {
                    tools.function.args_separator =
                        trim_whitespace(&sep_result.tags.get("sep").cloned().unwrap_or_default())
                            .to_string();
                }
            }
        }

        // Extract the closer (between last arg and call/section end marker)
        let suffix_marker = if !tools.format.per_call_end.is_empty() {
            tools.format.per_call_end.clone()
        } else {
            tools.format.section_end.clone()
        };
        let closer_suffix = if suffix_marker.is_empty() {
            // rely on an extra diff with the no-calls version
            let assistant_nocall = assistant_msg_json(needles::ASSISTANT_MSG);
            let notool_comp = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
                p.messages = Json::Array(vec![user.clone(), assistant_nocall.clone()]);
            });
            match notool_comp {
                Some(c) => match c.diff.left.find("YYYY") {
                    Some(pos) => c.diff.left[pos + 4..].to_string(),
                    None => String::new(),
                },
                None => String::new(),
            }
        } else {
            match diff.suffix.find(&suffix_marker) {
                Some(pos) => diff.suffix[..pos].to_string(),
                None => diff.suffix.clone(),
            }
        };
        if !closer_suffix.is_empty() {
            if tools.format.mode == ToolFormat::TagWithTagged {
                let closer_parser = build_tagged_peg_parser(|p| {
                    let until = p.until("YYYY");
                    let lit = p.literal("YYYY");
                    let sp = p.space();
                    let m = p.marker();
                    let sp2 = p.space();
                    let r = p.rest();
                    let close = p.tag("close", r);
                    p.sequence(&[until, lit, sp, m, sp2, close])
                });
                let close_result = closer_parser.parse_and_extract(&closer_suffix, PARSE_FLAG_NONE);
                if close_result.result.success() {
                    tools.function.close =
                        close_result.tags.get("close").cloned().unwrap_or_default();
                }
            } else if tools.format.mode == ToolFormat::TagWithJson {
                let closer_parser = build_tagged_peg_parser(|p| {
                    let until = p.until("YYYY");
                    let lit = p.literal("YYYY");
                    let r = p.rest();
                    let post_val = p.tag("post_val", r);
                    p.sequence(&[until, lit, post_val])
                });
                let close_result = closer_parser.parse_and_extract(&closer_suffix, PARSE_FLAG_NONE);
                if close_result.result.success() {
                    let post = close_result
                        .tags
                        .get("post_val")
                        .cloned()
                        .unwrap_or_default();
                    if let Some(pos) = post.rfind(|c| c == '}' || c == ']') {
                        if pos < post.len() - 1 {
                            tools.function.close =
                                trim_leading_whitespace(&post[pos + 1..]).to_string();
                        }
                    }
                }
            }
        }
        tools.function.close = trim_leading_whitespace(&tools.function.close).to_string();
    }
}

/// `extract_argument_name_markers` (chat-diff-analyzer.cpp:1251-1323)
fn extract_argument_name_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let first_arg = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_FIRST.to_string(),
            Json::String("XXXX".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);
    let other_arg = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_SECOND.to_string(),
            Json::String("YYYY".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), first_arg]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), other_arg.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if !diff.left.is_empty() && !diff.right.is_empty() {
        let left_parser = build_tagged_peg_parser(|p| {
            let until = p.until(needles::ARG_FIRST);
            let pre = p.tag("pre", until);
            let lit = p.literal(needles::ARG_FIRST);
            let u = p.until_one_of(&["\"", "X"]);
            let suffix = p.tag("suffix", u);
            p.sequence(&[pre, lit, suffix])
        });
        let right_parser = build_tagged_peg_parser(|p| {
            let until = p.until(needles::ARG_SECOND);
            let pre = p.tag("pre", until);
            let lit = p.literal(needles::ARG_SECOND);
            let u = p.until_one_of(&["\"", "Y"]);
            let suffix = p.tag("suffix", u);
            p.sequence(&[pre, lit, suffix])
        });
        let left_result = left_parser.parse_anywhere_and_extract(&diff.left);
        let right_result = right_parser.parse_anywhere_and_extract(&diff.right);

        let lpre = left_result.tags.get("pre").cloned().unwrap_or_default();
        let lsuf = left_result.tags.get("suffix").cloned().unwrap_or_default();
        let rpre = right_result.tags.get("pre").cloned().unwrap_or_default();
        let rsuf = right_result.tags.get("suffix").cloned().unwrap_or_default();
        if left_result.result.success()
            && right_result.result.success()
            && !lpre.is_empty()
            && lpre == rpre
            && lsuf == rsuf
        {
            // Name inside a structure (e.g. JSON key): prefix is the shared wrapper
            tools.arguments.name_prefix = lpre;
            tools.arguments.name_suffix = lsuf;
        } else if diff.left.starts_with(needles::ARG_FIRST)
            && diff.right.starts_with(needles::ARG_SECOND)
        {
            // Name directly in the diff: prefix from the last marker of diff.prefix
            let pre_parser = build_tagged_peg_parser(|p| {
                let last_marker = {
                    let mk = p.marker();
                    let neg = p.negate(mk);
                    let any = p.any();
                    let zseq = p.sequence(&[neg, any]);
                    let z = p.zero_or_more(zseq);
                    let end = p.end();
                    p.sequence(&[mk, z, end])
                };
                let neg = p.negate(last_marker);
                let any = p.any();
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                let np = p.tag("name_prefix", last_marker);
                p.sequence(&[z, np])
            });
            let pre_result = pre_parser.parse_and_extract(&diff.prefix, PARSE_FLAG_NONE);
            tools.arguments.name_prefix = if pre_result.result.success() {
                pre_result
                    .tags
                    .get("name_prefix")
                    .cloned()
                    .unwrap_or_default()
            } else {
                diff.prefix.clone()
            };

            // Suffix extends past ARG_FIRST to the first marker (+ optional space)
            let after_first = format!("{}{}", &diff.left[needles::ARG_FIRST.len()..], diff.suffix);
            let suffix_parser = build_tagged_peg_parser(|p| {
                let mk = p.marker();
                let neg = p.negate(mk);
                let any = p.any();
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                let m = p.marker();
                let sp = p.space();
                let zseq = p.sequence(&[z, m, sp]);
                let suffix = p.tag("suffix", zseq);
                suffix
            });
            let suf_result = suffix_parser.parse_anywhere_and_extract(&after_first);
            if suf_result.result.success() {
                tools.arguments.name_suffix =
                    suf_result.tags.get("suffix").cloned().unwrap_or_default();
            }
        }
    }
}

/// `extract_argument_value_markers` (chat-diff-analyzer.cpp:1325-1388)
fn extract_argument_value_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let val_x = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_FIRST.to_string(),
            Json::String("XXXX".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);
    let val_y = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_FIRST.to_string(),
            Json::String("YYYY".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), val_x]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), val_y.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if diff.left == "XXXX" && diff.right == "YYYY" {
        let arg_name_ending = format!("{}{}", needles::ARG_FIRST, tools.arguments.name_suffix);
        let mut prefix = diff.prefix.clone();
        if let Some(pos) = prefix.rfind(&arg_name_ending) {
            prefix = prefix[pos + arg_name_ending.len()..].to_string();
        }
        if !prefix.is_empty() {
            let prefix_parser = build_tagged_peg_parser(|p| {
                let last_marker = {
                    let mk = p.marker();
                    let neg = p.negate(mk);
                    let any = p.any();
                    let zseq = p.sequence(&[neg, any]);
                    let z = p.zero_or_more(zseq);
                    let end = p.end();
                    p.sequence(&[mk, z, end])
                };
                let neg = p.negate(last_marker);
                let any = p.any();
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                let vp = p.tag("val_prefix", last_marker);
                p.sequence(&[z, vp])
            });
            let pre_result = prefix_parser.parse_and_extract(&prefix, PARSE_FLAG_NONE);
            tools.arguments.value_prefix = if pre_result.result.success() {
                pre_result
                    .tags
                    .get("val_prefix")
                    .cloned()
                    .unwrap_or_default()
            } else {
                prefix
            };
        }

        let mut value_suffix = diff.suffix.clone();
        if !tools.function.close.is_empty() {
            if let Some(pos) = value_suffix.find(&tools.function.close) {
                value_suffix = value_suffix[..pos].to_string();
            }
        } else if !tools.format.per_call_end.is_empty() || !tools.format.section_end.is_empty() {
            let end_marker = if !tools.format.per_call_end.is_empty() {
                tools.format.per_call_end.clone()
            } else {
                tools.format.section_end.clone()
            };
            if let Some(pos) = value_suffix.find(&end_marker) {
                value_suffix = value_suffix[..pos].to_string();
            }
        }
        if !trim_whitespace(&value_suffix).is_empty() {
            tools.arguments.value_suffix = value_suffix;
        }
    }
}

/// `extract_argument_separator` (chat-diff-analyzer.cpp:1390-1423)
fn extract_argument_separator(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let one_arg = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_FIRST.to_string(),
            Json::String("XXXX".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);
    let two_args = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        args_xy(),
        needles::CALL_ID_001,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), one_arg]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), two_args.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if !diff.right.is_empty() {
        tools.arguments.separator =
            until_common_prefix(&diff.right, needles::ARG_FIRST, needles::ARG_SECOND);
    }
}

/// `extract_args_markers` (chat-diff-analyzer.cpp:1425-1484)
fn extract_args_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let no_args = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(Vec::new()),
        needles::CALL_ID_001,
    )]);
    let with_args = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        Json::Object(vec![(
            needles::ARG_FIRST.to_string(),
            Json::String("XXXX".to_string()),
        )]),
        needles::CALL_ID_001,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), no_args]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), with_args.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if tools.format.mode == ToolFormat::JsonNative {
        let prefix_marker = if !tools.format.section_start.is_empty() {
            tools.format.section_start.clone()
        } else {
            tools.format.per_call_start.clone()
        };
        let suffix_marker = if !tools.format.section_end.is_empty() {
            tools.format.section_end.clone()
        } else {
            tools.format.per_call_end.clone()
        };
        // find the closest occurrences
        let prefix_pos = if prefix_marker.is_empty() {
            Some(0)
        } else {
            diff.prefix.rfind(&prefix_marker)
        };
        let suffix_pos = if suffix_marker.is_empty() {
            None
        } else {
            diff.suffix.find(&suffix_marker)
        };
        let prefix_pos = prefix_pos.unwrap_or(0);
        let suffix_end = suffix_pos.unwrap_or(diff.suffix.len());
        let prefix_cut = diff.prefix[prefix_pos + prefix_marker.len()..].to_string();
        let suffix_cut = diff.suffix[..suffix_end].to_string();
        let args_start = until_common_prefix(&prefix_cut, "{}", "{\"first\":");
        let args_end = after_common_suffix(&suffix_cut, "{}", "\"XXXX\"}");

        if !args_start.is_empty() || !args_end.is_empty() {
            let mut args_start = args_start;
            if let Some(pos) = args_start.find(needles::FUN_FIRST) {
                args_start = args_start[pos + needles::FUN_FIRST.len()..].to_string();
            }
            if let Some(pos) = args_start.find(needles::CALL_ID_001) {
                args_start = args_start[pos + needles::CALL_ID_001.len()..].to_string();
            }
            tools.arguments.start = args_start;
            tools.arguments.end = args_end;
        }
    }
}

/// `extract_call_id_markers` (chat-diff-analyzer.cpp:1486-1633)
fn extract_call_id_markers(tmpl: &ChatTemplate, tools: &mut AnalyzeTools) {
    let user = user_msg_json(needles::USER_MSG);
    let id1 = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        args_xy(),
        needles::CALL_ID_001,
    )]);
    let id2 = assistant_with_tool_calls_json(vec![build_tool_call(
        needles::FUN_FIRST,
        args_xy(),
        needles::CALL_ID_999,
    )]);

    let params = TemplateParams {
        messages: Json::Array(vec![user.clone(), id1]),
        tools: analysis_tools_json(),
        add_generation_prompt: false,
        enable_thinking: true,
        ..TemplateParams::default()
    };

    let comparison = compare_variants(tmpl, &params, &|p: &mut TemplateParams| {
        p.messages = Json::Array(vec![user.clone(), id2.clone()]);
    });
    let Some(comparison) = comparison else { return };
    let diff = &comparison.diff;

    if diff.left.is_empty() && diff.right.is_empty() {
        return;
    }

    let id_value_1 = needles::CALL_ID_001;
    let id_value_2 = needles::CALL_ID_999;

    let common_id_prefix_len = id_value_1
        .bytes()
        .zip(id_value_2.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let common_id_part = &id_value_1[..common_id_prefix_len];

    let func_name = needles::FUN_FIRST;
    let func_name_in_prefix = diff.prefix.rfind(func_name);
    let func_name_in_suffix = diff.suffix.find(func_name);

    let find_last_marker = |s: &str| -> String {
        let parser = build_tagged_peg_parser(|p| {
            let last = {
                let mk = p.marker();
                let neg = p.negate(mk);
                let any = p.any();
                let zseq = p.sequence(&[neg, any]);
                let z = p.zero_or_more(zseq);
                let end = p.end();
                p.sequence(&[mk, z, end])
            };
            let neg = p.negate(last);
            let any = p.any();
            let zseq = p.sequence(&[neg, any]);
            let z = p.zero_or_more(zseq);
            let mk = p.marker();
            let m = p.tag("m", mk);
            p.sequence(&[z, m])
        });
        let res = parser.parse_anywhere_and_extract(s);
        if res.result.success() {
            res.tags.get("m").cloned().unwrap_or_default()
        } else {
            String::new()
        }
    };

    let find_first_marker = |s: &str| -> String {
        let parser = build_tagged_peg_parser(|p| {
            let m = p.marker();
            p.tag("m", m)
        });
        let res = parser.parse_anywhere_and_extract(s);
        if res.result.success() {
            res.tags.get("m").cloned().unwrap_or_default()
        } else {
            String::new()
        }
    };

    if func_name_in_prefix.is_some() && func_name_in_suffix.is_none() {
        // call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        let fpos = func_name_in_prefix.unwrap();
        let args_in_prefix = diff.prefix[fpos..].find('{').map(|p| fpos + p);
        let args_in_suffix = diff.suffix.find('{');

        if let Some(_) = args_in_suffix {
            if args_in_prefix.is_none() {
                // Args in suffix ⇒ BETWEEN_FUNC_AND_ARGS
                tools.call_id.pos = CallIdPosition::BetweenFuncAndArgs;

                let after_func = diff.prefix[fpos + func_name.len()..].to_string();
                let id_prefix_parser = {
                    let cid = common_id_part.to_string();
                    build_tagged_peg_parser(move |p| {
                        let m = p.marker();
                        let prefix = p.tag("prefix", m);
                        let mk = p.marker();
                        let neg1 = p.negate(mk);
                        let lit = p.literal(&cid);
                        let neg2 = p.negate(lit);
                        let any = p.any();
                        let zseq = p.sequence(&[neg1, neg2, any]);
                        let z = p.zero_or_more(zseq);
                        let lit2 = p.literal(&cid);
                        p.sequence(&[prefix, z, lit2])
                    })
                };
                let id_res = id_prefix_parser.parse_anywhere_and_extract(&after_func);
                if id_res.result.success() {
                    tools.call_id.prefix = id_res.tags.get("prefix").cloned().unwrap_or_default();
                } else {
                    tools.call_id.prefix = find_last_marker(&after_func);
                }

                // call_id_suffix: the first marker in the suffix before args "{"
                let suffix_parser = {
                    let cid = common_id_part.to_string();
                    let _ = cid;
                    build_tagged_peg_parser(|p| {
                        let mk = p.marker();
                        let neg1 = p.negate(mk);
                        let brace = p.literal("{");
                        let neg2 = p.negate(brace);
                        let any = p.any();
                        let zseq = p.sequence(&[neg1, neg2, any]);
                        let z = p.zero_or_more(zseq);
                        let m = p.marker();
                        let sseq = p.sequence(&[z, m]);
                        p.tag("suffix", sseq)
                    })
                };
                let suf_res = suffix_parser.parse_anywhere_and_extract(&diff.suffix);
                if suf_res.result.success() {
                    tools.call_id.suffix = suf_res.tags.get("suffix").cloned().unwrap_or_default();
                }
            } else {
                // Args in prefix ⇒ POST_ARGS
                tools.call_id.pos = CallIdPosition::PostArgs;

                let after_args = diff.prefix[args_in_prefix.unwrap()..].to_string();
                if let Some(closing_brace) = after_args.rfind('}') {
                    let between_args_and_id = after_args[closing_brace + 1..].to_string();
                    tools.call_id.prefix = find_last_marker(&between_args_and_id);
                }
                tools.call_id.suffix = find_first_marker(&diff.suffix);
            }
        }
    } else if func_name_in_suffix.is_some() && func_name_in_prefix.is_none() {
        // call_id is PRE_FUNC_NAME
        tools.call_id.pos = CallIdPosition::PreFuncName;
        tools.call_id.prefix = find_last_marker(&diff.prefix);
        let before_func = diff.suffix[..func_name_in_suffix.unwrap()].to_string();
        tools.call_id.suffix = find_first_marker(&before_func);
    }

    if tools.call_id.prefix == tools.arguments.end {
        tools.call_id.prefix.clear();
    }
    if tools.call_id.suffix == tools.arguments.start {
        tools.call_id.suffix.clear();
    }

    // per_call_end may have swallowed the call_id_suffix + sample args
    if tools.call_id.pos != CallIdPosition::None
        && !tools.call_id.suffix.is_empty()
        && tools.format.per_call_end.starts_with(&tools.call_id.suffix)
    {
        tools.format.per_call_end.clear();
    }
}

/// the analysis workarounds table (chat-diff-analyzer.cpp:36-205)
fn apply_workarounds(tmpl: &ChatTemplate, analysis: &mut Autoparser) {
    let src = tmpl.source();
    // Old reasoning Qwen templates (chat-diff-analyzer.cpp:39-51)
    if src.contains("content.split('</think>')")
        && !src.contains("reasoning_content")
        && !src.contains("<SPECIAL_12>")
        && analysis.reasoning.mode == ReasoningMode::None
    {
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>".to_string();
        analysis.reasoning.end = "</think>".to_string();
        analysis.preserved_tokens.push("<think>".to_string());
        analysis.preserved_tokens.push("</think>".to_string());
    }
    // Granite 3.3 (chat-diff-analyzer.cpp:53-68)
    if src.contains(
        "Write your thoughts between <think></think> and write your response between <response></response>",
    ) {
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>".to_string();
        analysis.reasoning.end = "</think>".to_string();
        analysis.preserved_tokens.push("<think>".to_string());
        analysis.preserved_tokens.push("</think>".to_string());
        analysis.content.mode = ContentMode::WrappedWithReasoning;
        analysis.content.start = "<response>".to_string();
        analysis.content.end = "</response>".to_string();
        analysis.preserved_tokens.push("<response>".to_string());
        analysis.preserved_tokens.push("</response>".to_string());
    }
    // Cohere Command R+ (chat-diff-analyzer.cpp:70-81)
    if src.contains("<|CHATBOT_TOKEN|>")
        && src.contains("<|END_OF_TURN_TOKEN|>")
        && analysis.content.start.is_empty()
    {
        analysis.content.mode = ContentMode::AlwaysWrapped;
        analysis.content.start = "<|CHATBOT_TOKEN|>".to_string();
        analysis.content.end = "<|END_OF_TURN_TOKEN|>".to_string();
        analysis
            .preserved_tokens
            .push("<|CHATBOT_TOKEN|>".to_string());
        analysis
            .preserved_tokens
            .push("<|END_OF_TURN_TOKEN|>".to_string());
        analysis.user_start = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>".to_string();
    }
    // Functionary 3.1 (chat-diff-analyzer.cpp:83-102)
    if src.contains(
        "set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", \"code_interpreter\") | list | length > 0",
    ) {
        analysis.content.mode = ContentMode::Plain;
        analysis.content.end = String::new();
        analysis.tools.function.name_prefix = String::new();
        analysis.tools.format.section_start = String::new();
        analysis.tools.format.section_end = String::new();
        analysis.tools.format.per_call_start = "<function=".to_string();
        analysis.tools.format.per_call_end = "</function>".to_string();
        analysis.tools.function.close = String::new();
        analysis.preserved_tokens.clear();
        analysis.preserved_tokens.push("<|eot_id|>".to_string());
        analysis.preserved_tokens.push("<|eom_id|>".to_string());
        analysis.preserved_tokens.push("<function=".to_string());
        analysis.preserved_tokens.push(">".to_string());
        analysis.preserved_tokens.push("</function>".to_string());
    }
    // DeepSeek-R1-Distill-Qwen (chat-diff-analyzer.cpp:104-116)
    if src.contains(
        "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'",
    ) {
        analysis.tools.format.section_start = "<｜tool▁calls▁begin｜>".to_string();
        analysis.tools.format.section_end = "<｜tool▁calls▁end｜>".to_string();
        analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function".to_string();
        analysis.tools.function.name_prefix = "<｜tool▁sep｜>".to_string();
        analysis.tools.format.per_call_end = "<｜tool▁call▁end｜>".to_string();
        analysis.tools.function.close = "```".to_string();
    }
    // Nemotron Nano v2 (chat-diff-analyzer.cpp:118-143)
    if src.contains("<SPECIAL_10>")
        && src.contains("<SPECIAL_11>")
        && src.contains("<SPECIAL_12>")
        && src.contains("<TOOL_RESPONSE>")
    {
        analysis.tools.format.mode = ToolFormat::JsonNative;
        analysis.tools.format.section_start = String::new();
        analysis.tools.format.section_end = String::new();
        analysis.tools.format.per_call_start = "<TOOLCALL>".to_string();
        analysis.tools.format.per_call_end = "</TOOLCALL>".to_string();
        analysis.tools.format.tools_array_wrapped = true;
        analysis.content.mode = ContentMode::Plain;
        analysis.content.start = String::new();
        analysis.content.end = String::new();
        analysis.reasoning.mode = ReasoningMode::TagBased;
        analysis.reasoning.start = "<think>\n".to_string();
        analysis.reasoning.end = "</think>".to_string();
        analysis.assistant_start = "<SPECIAL_11>Assistant".to_string();
        analysis.user_start = "<SPECIAL_11>User".to_string();
        analysis.preserved_tokens.clear();
        analysis.preserved_tokens.push("<SPECIAL_11>".to_string());
        analysis.preserved_tokens.push("</think>".to_string());
        analysis.preserved_tokens.push("<TOOLCALL>".to_string());
        analysis.preserved_tokens.push("</TOOLCALL>".to_string());
    }
    // Fireworks (chat-diff-analyzer.cpp:145-152)
    if src.contains(
        "{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n' + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}",
    ) {
        analysis.assistant_start = "<|start_header_id|>assistant<|end_header_id|>".to_string();
        analysis.user_start = "<|start_header_id|>user<|end_header_id|>".to_string();
    }
    // Solar Open (chat-diff-analyzer.cpp:154-159)
    if src.contains("<|begin|>assistant<|think|><|end|>") {
        analysis.assistant_start = "<|begin|>assistant".to_string();
    }
    // Apriel 1.6 (chat-diff-analyzer.cpp:161-167)
    if src.contains("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text") {
        analysis.user_start = "<|begin_user|>".to_string();
        analysis.assistant_start = "<|begin_assistant|>".to_string();
    }
    // JSON {name, parameters} tool instruction w/ OpenAI wrapper (169-175)
    if src.contains("Respond in the format {\"name\": function name")
        && src.contains("Do not use variables.")
    {
        analysis.tools.format.openai_wrapper_trigger = true;
    }
    // Laguna (poolside) (chat-diff-analyzer.cpp:181-195)
    if src.contains("laguna_glm_thinking") {
        analysis.reasoning.start = trim_whitespace(&analysis.reasoning.start).to_string();
        analysis.reasoning.end = trim_whitespace(&analysis.reasoning.end).to_string();
        analysis.tools.arguments.value_prefix =
            trim_whitespace(&analysis.tools.arguments.value_prefix).to_string();
        analysis.tools.arguments.value_suffix =
            trim_whitespace(&analysis.tools.arguments.value_suffix).to_string();
        analysis.tools.arguments.separator =
            trim_whitespace(&analysis.tools.arguments.separator).to_string();
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
        analysis.additional_stops.push("</assistant>".to_string());
    }
    // Bailing V3 (chat-diff-analyzer.cpp:197-203)
    if src.contains("Bailing V3 chat template") {
        analysis.tools.arguments.value_suffix =
            trim_whitespace(&analysis.tools.arguments.value_suffix).to_string();
        analysis.tools.arguments.tolerate_intertag_whitespace = true;
    }
}

// ---------------------------------------------------------------------------
// parser generation (chat-auto-parser-generator.cpp)
// ---------------------------------------------------------------------------

/// `parser_build_context` (chat-auto-parser-generator.cpp:18-21 +
/// chat-auto-parser.h:226-235)
pub struct ParserBuildContext<'a, 'b> {
    pub p: &'a mut ChatPegBuilder,
    pub inputs: &'b GenerationParams,
    pub reasoning_parser: ParserId,
    pub extracting_reasoning: bool,
    pub reasoning: Option<&'b AnalyzeReasoning>,
    pub content: Option<&'b AnalyzeContent>,
}

/// `peg_generator::generate_parser` (chat-auto-parser-generator.cpp:23-29)
pub fn generate_parser(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    // Run differential analysis to extract template structure
    let mut autoparser = Autoparser::default();
    autoparser.analyze_template(tmpl);
    generate_parser_with(tmpl, inputs, &autoparser)
}

/// `peg_generator::generate_parser` overload (chat-auto-parser-generator.cpp:31-97)
pub fn generate_parser_with(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
    autoparser: &Autoparser,
) -> Result<ChatParams, String> {
    // Create the result structure
    let mut data = ChatParams::default();
    data.prompt = template_direct_apply(tmpl, inputs)?;
    data.generation_prompt = template_generation_prompt(tmpl, inputs)?;
    data.format = ChatFormat::PegNative;
    data.preserved_tokens = autoparser.preserved_tokens.clone();
    data.additional_stops
        .extend(autoparser.additional_stops.iter().cloned());

    let parser_generation_prompt = data.generation_prompt.clone();

    if inputs.continue_final_message != ChatContinuation::None && inputs.has_continuation() {
        // Build up generation prompt manually (chat-auto-parser-generator.cpp:45-62)
        let msg = &inputs.continue_msg;
        if !autoparser.reasoning.start.is_empty() {
            if let Some(pos) = data.generation_prompt.find(&autoparser.reasoning.start) {
                data.generation_prompt = data.generation_prompt[..pos].to_string();
            }
            data.generation_prompt += &autoparser.reasoning.start;
            data.generation_prompt += &msg.reasoning_content;
            if inputs.continue_final_message == ChatContinuation::Content {
                data.generation_prompt += &autoparser.reasoning.end;
            }
        }
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &msg.render_content("\n\n")?;
        }
        data.prompt += &data.generation_prompt;
    }

    let parser = autoparser_build_parser(autoparser, inputs, &parser_generation_prompt)?;
    data.parser = parser.save();

    // Build grammar if tools are present (chat-auto-parser-generator.cpp:67-94)
    let has_tools = autoparser.tools.format.mode != ToolFormat::None
        && inputs.tools.is_array()
        && !inputs.tools.empty();
    let trigger_marker = if !autoparser.tools.format.section_start.is_empty() {
        autoparser.tools.format.section_start.clone()
    } else {
        autoparser.tools.format.per_call_start.clone()
    };

    let has_response_format =
        !matches!(inputs.json_schema, Json::Null) && inputs.json_schema.is_object();
    let include_grammar = has_response_format
        || (has_tools
            && ((inputs.tool_choice == ChatToolChoice::Auto && !trigger_marker.is_empty())
                || inputs.tool_choice == ChatToolChoice::Required));

    if include_grammar {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == ChatToolChoice::Auto;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        // Set grammar triggers based on tool section markers
        if data.grammar_lazy {
            data.grammar_triggers = vec![GrammarTrigger::word(&trigger_marker)];
            if autoparser.tools.format.openai_wrapper_trigger {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers
                    .push(GrammarTrigger::word("{\"type\": \"function\","));
            }
        }
    }

    Ok(data)
}

/// `autoparser::build_parser` (chat-auto-parser-generator.cpp:99-136)
fn autoparser_build_parser(
    autoparser: &Autoparser,
    inputs: &GenerationParams,
    generation_prompt: &str,
) -> Result<PegArena, String> {
    if !autoparser.analysis_complete {
        return Err(
            "Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)"
                .to_string(),
        );
    }
    build_chat_peg_parser(|p| {
        let extracting_reasoning = inputs.reasoning_format != ReasoningFormat::None
            && autoparser.reasoning.mode != ReasoningMode::None;

        // Build reasoning parser
        let reasoning_parser =
            build_reasoning_parser(p, &autoparser.reasoning, extracting_reasoning);

        let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
        let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        let mut pure_content = autoparser.reasoning.mode == ReasoningMode::None;

        let parser;
        if has_response_format {
            // chat-auto-parser-generator.cpp:120-126
            let json = p.p.json();
            let schema =
                p.p.schema(json, "response-format-schema", &inputs.json_schema, false);
            let content = p.content(schema);
            let response_format = p.p.rule("response-format", content, false);
            let lit1 = p.p.literal("```json");
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let lit2 = p.p.literal("```");
            let branch1 = p.p.sequence(&[lit1, sp1, response_format, sp2, lit2]);
            let sp3 = p.p.space();
            let sp4 = p.p.space();
            let branch2 = p.p.sequence(&[sp3, response_format, sp4]);
            let choice = p.p.choice(&[branch1, branch2]);
            let sp0 = p.p.space();
            let end = p.p.end();
            parser = p.p.sequence(&[reasoning_parser, sp0, choice, end]);
            pure_content = false;
        } else if has_tools
            && inputs.tool_choice != ChatToolChoice::None
            && autoparser.jinja_caps.supports_tool_calls
        {
            parser = build_tool_parser(
                p,
                &autoparser.tools,
                &autoparser.content,
                inputs,
                reasoning_parser,
            );
            pure_content = false;
        } else {
            parser = build_content_parser(p, &autoparser.content, reasoning_parser);
        }
        let reasoning_start = trim_whitespace(&autoparser.reasoning.start).to_string();
        if pure_content {
            let pre = p.prefix(generation_prompt, &reasoning_start);
            p.p.sequence(&[pre, parser])
        } else {
            let pre = p.prefix(generation_prompt, &reasoning_start);
            p.p.spaced(pre, parser)
        }
    })
}

/// `analyze_reasoning::build_parser` (chat-auto-parser-generator.cpp:138-157)
fn build_reasoning_parser(
    p: &mut ChatPegBuilder,
    reasoning: &AnalyzeReasoning,
    extracting_reasoning: bool,
) -> ParserId {
    if !extracting_reasoning {
        return p.p.eps();
    }
    if reasoning.mode == ReasoningMode::TagBased || reasoning.mode == ReasoningMode::ToolsOnly {
        if !reasoning.end.is_empty() {
            if !reasoning.start.is_empty() {
                // Standard tag-based: optional(<think>reasoning</think>)
                let inner = {
                    let opt_start = p.optspace(&reasoning.start);
                    let end_trimmed = trim_whitespace(&reasoning.end).to_string();
                    let until = p.p.until(&end_trimmed);
                    let r = p.reasoning(until);
                    let opt_end = p.optspace(&end_trimmed);
                    p.p.sequence(&[opt_start, r, opt_end])
                };
                p.p.optional(inner)
            } else {
                // Delimiter-style (empty start)
                let end_trimmed = trim_whitespace(&reasoning.end).to_string();
                let until = p.p.until(&end_trimmed);
                let r = p.reasoning(until);
                let opt_end = p.optspace(&end_trimmed);
                let inner = p.p.sequence(&[r, opt_end]);
                p.p.optional(inner)
            }
        } else {
            p.p.eps()
        }
    } else {
        p.p.eps()
    }
}

/// `analyze_content::build_parser` (chat-auto-parser-generator.cpp:159-169)
fn build_content_parser(
    p: &mut ChatPegBuilder,
    content: &AnalyzeContent,
    reasoning_parser: ParserId,
) -> ParserId {
    if content.is_always_wrapped() {
        let until_end = p.p.until(&content.end);
        let c1 = p.content(until_end);
        let end_lit = p.p.literal(&content.end);
        let until_start = p.p.until(&content.start);
        let c0 = p.content(until_start);
        let start_lit = p.p.literal(&content.start);
        let end = p.p.end();
        // NOTE: the C++ distinguishes extracting_reasoning vs not
        // (chat-auto-parser-generator.cpp:162-166); the non-extracting variant
        // leads with a second content capture instead of the reasoning parser
        let _ = (c0, start_lit);
        p.p.sequence(&[reasoning_parser, c1, end_lit, end])
    } else {
        let rest = p.p.rest();
        let c = p.content(rest);
        let end = p.p.end();
        p.p.sequence(&[reasoning_parser, c, end])
    }
}

/// `analyze_content::build_optional_wrapped` (chat-auto-parser-generator.cpp:171-178)
#[allow(dead_code)]
fn build_optional_wrapped(p: &mut ChatPegBuilder, content: &AnalyzeContent) -> ParserId {
    if content.is_always_wrapped() {
        let start = p.p.literal(&content.start);
        let until = p.p.until(&content.end);
        let c = p.content(until);
        let end = p.p.literal(&content.end);
        let inner = p.p.sequence(&[start, c, end]);
        p.p.optional(inner)
    } else {
        p.p.eps()
    }
}

/// `analyze_tools::build_parser` (chat-auto-parser-generator.cpp:180-194)
fn build_tool_parser(
    p: &mut ChatPegBuilder,
    tools: &AnalyzeTools,
    content: &AnalyzeContent,
    inputs: &GenerationParams,
    reasoning_parser: ParserId,
) -> ParserId {
    match tools.format.mode {
        ToolFormat::JsonNative => {
            build_tool_parser_json_native(p, tools, content, inputs, reasoning_parser)
        }
        ToolFormat::TagWithJson => build_tool_parser_tag_json(p, tools, inputs, reasoning_parser),
        ToolFormat::TagWithTagged => {
            build_tool_parser_tag_tagged(p, tools, inputs, reasoning_parser)
        }
        ToolFormat::None => p.p.eps(),
    }
}

/// `analyze_tools::build_tool_parser_json_native` (chat-auto-parser-generator.cpp:196-238)
fn build_tool_parser_json_native(
    p: &mut ChatPegBuilder,
    tools: &AnalyzeTools,
    content: &AnalyzeContent,
    inputs: &GenerationParams,
    reasoning_parser: ParserId,
) -> ParserId {
    // Effective field names with dot notation if function_field is set
    let mut name_field = tools.format.name_field.clone();
    let mut args_field = tools.format.args_field.clone();

    if !tools.format.function_field.is_empty()
        && tools.format.function_field != "function"
        && !name_field.contains('.')
    {
        name_field = format!("{}.{}", tools.format.function_field, name_field);
        args_field = format!("{}.{}", tools.format.function_field, args_field);
    }

    let tools_parser;
    if tools.format.section_start.is_empty() && !tools.format.per_call_start.is_empty() {
        let single_tool_parser = p.standard_json_tools(
            &tools.format.per_call_start,
            &tools.format.per_call_end,
            &inputs.tools,
            inputs.parallel_tool_calls,
            inputs.tool_choice == ChatToolChoice::Required,
            &name_field,
            &args_field,
            tools.format.tools_array_wrapped,
            tools.format.fun_name_is_key,
            &tools.format.id_field,
            &tools.format.gen_id_field,
            &tools.format.parameter_order,
            tools.format.openai_wrapper_trigger,
        );
        let sp = p.p.space();
        let seq = p.p.sequence(&[single_tool_parser, sp]);
        let one = p.p.one_or_more(seq);
        tools_parser = p.p.trigger_rule("tool-calls", one);
    } else {
        tools_parser = p.standard_json_tools(
            &tools.format.section_start,
            &tools.format.section_end,
            &inputs.tools,
            inputs.parallel_tool_calls,
            inputs.tool_choice == ChatToolChoice::Required,
            &name_field,
            &args_field,
            tools.format.tools_array_wrapped,
            tools.format.fun_name_is_key,
            &tools.format.id_field,
            &tools.format.gen_id_field,
            &tools.format.parameter_order,
            tools.format.openai_wrapper_trigger,
        );
    }

    // Handle content wrappers if present (chat-auto-parser-generator.cpp:224-228)
    if content.is_always_wrapped() {
        let wrapped_content = {
            let start = p.p.literal(&content.start);
            let until = p.p.until(&content.end);
            let c = p.content(until);
            let end = p.p.literal(&content.end);
            let inner = p.p.sequence(&[start, c, end]);
            p.p.optional(inner)
        };
        let end = p.p.end();
        return p
            .p
            .sequence(&[reasoning_parser, wrapped_content, tools_parser, end]);
    }

    let mut tool_start = "{".to_string();
    if !tools.format.section_start.is_empty() {
        tool_start = tools.format.section_start.clone();
    } else if !tools.format.per_call_start.is_empty() {
        tool_start = tools.format.per_call_start.clone();
    }

    let until = p.p.until(&tool_start);
    let c = p.content(until);
    let opt = p.p.optional(c);
    let end = p.p.end();
    p.p.sequence(&[reasoning_parser, opt, tools_parser, end])
}

/// `analyze_tools::build_func_parser` (chat-auto-parser-generator.cpp:240-286):
/// shared helper assembling open + call_id + args with atomicity handling.
#[allow(clippy::too_many_arguments)]
fn build_func_parser(
    p: &mut ChatPegBuilder,
    tools: &AnalyzeTools,
    name: &str,
    call_id_section: ParserId,
    have_call_id: bool,
    args: ParserId,
    atomic_peek: Option<ParserId>,
) -> ParserId {
    let open = {
        let prefix = p.p.literal(&tools.function.name_prefix);
        let nm = p.p.literal(name);
        let tn = p.tool_name(nm);
        let inner = p.p.sequence(&[prefix, tn]);
        let suffix = p.p.literal(&tools.function.name_suffix);
        let oseq = p.p.sequence(&[inner, suffix]);
        p.tool_open(oseq)
    };
    let mut matched_atomic = false;
    let mut func_parser;

    if !tools.function.args_separator.is_empty() {
        let sp = p.p.space();
        let sep = p.p.literal(&tools.function.args_separator);
        let open = p.p.sequence(&[open, sp, sep]);
        let sp2 = p.p.space();
        func_parser = p.p.sequence(&[open, call_id_section, sp2, args]);
        matched_atomic = true;
    } else if !tools.function.name_suffix.is_empty() {
        // chat-auto-parser-generator.cpp:252-254
        let sp = p.p.space();
        func_parser = p.p.sequence(&[open, call_id_section, sp, args]);
        matched_atomic = true;
    } else if have_call_id {
        let seq = p.p.sequence(&[open, call_id_section]);
        let atom = p.p.atomic(seq);
        let sp = p.p.space();
        func_parser = p.p.sequence(&[atom, sp, args]);
        matched_atomic = true;
    } else if let Some(peek_expr) = atomic_peek {
        let sp = p.p.space();
        let seq = p.p.sequence(&[open, call_id_section, sp, peek_expr]);
        let atom = p.p.atomic(seq);
        func_parser = p.p.sequence(&[atom, args]);
        matched_atomic = true;
    } else {
        let sp = p.p.space();
        func_parser = p.p.sequence(&[open, call_id_section, sp, args]);
    }

    if !tools.function.close.is_empty() {
        let close = p.p.literal(&tools.function.close);
        let sp = p.p.space();
        let tc = p.tool_close(close);
        func_parser = p.p.sequence(&[func_parser, sp, tc]);
    } else if !tools.format.per_call_end.is_empty() {
        // peek() so tool_close only fires when the closing marker is visible
        // (chat-auto-parser-generator.cpp:268-278)
        let close_peek = if tools.arguments.tolerate_intertag_whitespace {
            let sp = p.p.space();
            let lit = p.p.literal(&tools.format.per_call_end);
            let pseq = p.p.sequence(&[sp, lit]);
            p.p.peek(pseq)
        } else {
            let lit = p.p.literal(&tools.format.per_call_end);
            p.p.peek(lit)
        };
        let tc = p.tool_close(close_peek);
        func_parser = p.p.sequence(&[func_parser, tc]);
    } else {
        let sp = p.p.space();
        let tc = p.tool_close(sp); // force tool closing callbacks in the mapper
        func_parser = p.p.sequence(&[func_parser, tc]);
    }
    if !matched_atomic {
        func_parser = p.p.atomic(func_parser);
    }
    func_parser
}

/// `foreach_function` (chat-peg-parser.h-adjacent helper): iterate a tools
/// array's function objects.
pub(crate) fn foreach_function<F: FnMut(&Json)>(tools: &Json, mut f: F) {
    if !tools.is_array() {
        return;
    }
    for tool in tools.iter() {
        if let Some(function) = tool.at("function") {
            f(function);
        }
    }
}

/// `foreach_parameter` (chat-auto-parser-generator.cpp:374-392 helper):
/// iterate a function's schema properties with their owning document.
pub(crate) fn foreach_parameter<
    F: FnMut(&crate::json_schema::SchemaProperty, &Rc<SchemaDocument>),
>(
    function: &Json,
    mut f: F,
) {
    let params = tool_parameters(function);
    let doc = match schema_from_json(&params) {
        Ok(d) => Rc::new(d),
        Err(_) => return,
    };
    if let Some(node) = doc.nodes.get(doc.root) {
        if let SchemaKind::Object { properties, .. } = &node.kind {
            for prop in properties {
                f(prop, &doc);
            }
        }
    }
}

/// `analyze_tools::build_tool_parser_tag_json` (chat-auto-parser-generator.cpp:288-357)
fn build_tool_parser_tag_json(
    p: &mut ChatPegBuilder,
    tools: &AnalyzeTools,
    inputs: &GenerationParams,
    reasoning_parser: ParserId,
) -> ParserId {
    let mut tool_choice_alts: Vec<ParserId> = Vec::new();

    foreach_function(&inputs.tools, |func| {
        let name = func
            .at("name")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("")
            .to_string();
        let schema = tool_parameters(func);

        // Build call_id parser based on position (if supported)
        let mut have_call_id = false;
        let mut call_id_section = p.p.eps();
        if tools.call_id.pos == CallIdPosition::BetweenFuncAndArgs
            && !tools.call_id.prefix.is_empty()
            && (!tools.call_id.suffix.is_empty() || !tools.arguments.start.is_empty())
        {
            if !tools.call_id.suffix.is_empty() {
                let pre = p.p.literal(&tools.call_id.prefix);
                let until = p.p.until(&tools.call_id.suffix);
                let id = p.tool_id(until);
                let oseq = p.p.sequence(&[pre, id]);
                let opt = p.p.optional(oseq);
                let suf = p.p.literal(&tools.call_id.suffix);
                call_id_section = p.p.sequence(&[opt, suf]);
            } else {
                let pre = p.p.literal(&tools.call_id.prefix);
                let until = p.p.until(&tools.arguments.start);
                let id = p.tool_id(until);
                let oseq = p.p.sequence(&[pre, id]);
                call_id_section = p.p.optional(oseq);
            }
            have_call_id = true;
        }
        let mut args_parser = {
            let j = p.p.json();
            let sch =
                p.p.schema(j, &format!("tool-{name}-schema"), &schema, false);
            p.tool_args(sch)
        };
        if !tools.arguments.start.is_empty() {
            let lit = p.p.literal(&tools.arguments.start);
            args_parser = p.p.sequence(&[lit, args_parser]);
        }
        if !tools.arguments.end.is_empty() {
            let lit = p.p.literal(&tools.arguments.end);
            args_parser = p.p.sequence(&[args_parser, lit]);
        }

        let atomic_peek = if !tools.arguments.start.is_empty() {
            let lit = p.p.literal(&tools.arguments.start);
            Some(p.p.peek(lit))
        } else {
            None
        };
        let func_parser = build_func_parser(
            p,
            tools,
            &name,
            call_id_section,
            have_call_id,
            args_parser,
            atomic_peek,
        );
        tool_choice_alts.push(p.p.rule(&format!("tool-{name}"), func_parser, false));
    });

    let tool_choice = p.p.choice(&tool_choice_alts);
    let require_calls = inputs.tool_choice == ChatToolChoice::Required;

    let mut tool_calls;
    if !tools.format.per_call_start.is_empty() {
        let wrapped_call = {
            let s = p.p.literal(&tools.format.per_call_start);
            let e = p.p.literal(&tools.format.per_call_end);
            p.p.sequence(&[s, tool_choice, e])
        };
        if inputs.parallel_tool_calls {
            let sp = p.p.space();
            let mseq = p.p.sequence(&[sp, wrapped_call]);
            let more = p.p.zero_or_more(mseq);
            let body = p.p.sequence(&[wrapped_call, more]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            tool_calls = p.p.trigger_rule("tool-call", wrapped_call);
        }
        if !tools.format.section_start.is_empty() {
            let lit = p.p.literal(&tools.format.section_start);
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let end = if tools.format.section_end.is_empty() {
                p.p.end()
            } else {
                p.p.literal(&tools.format.section_end)
            };
            let body = p.p.sequence(&[lit, sp1, tool_calls, sp2, end]);
            tool_calls = p.p.trigger_rule("tool-calls", body);
        }
    } else {
        let separator = ", "; // default
        if inputs.parallel_tool_calls {
            let s = p.p.literal(&tools.format.section_start);
            let lit = p.p.literal(separator);
            let mseq = p.p.sequence(&[lit, tool_choice]);
            let more = p.p.zero_or_more(mseq);
            let e = p.p.literal(&tools.format.section_end);
            let body = p.p.sequence(&[s, tool_choice, more, e]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            let s = p.p.literal(&tools.format.section_start);
            let e = p.p.literal(&tools.format.section_end);
            let body = p.p.sequence(&[s, tool_choice, e]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        }
    }

    if !require_calls {
        tool_calls = p.p.optional(tool_calls);
    }

    let trigger_marker = if !tools.format.section_start.is_empty() {
        tools.format.section_start.clone()
    } else {
        tools.format.per_call_start.clone()
    };
    let content_before_tools = if trigger_marker.is_empty() {
        p.p.eps()
    } else {
        p.p.until(&trigger_marker)
    };
    let c = p.content(content_before_tools);
    let opt = p.p.optional(c);
    let end = p.p.end();
    p.p.sequence(&[reasoning_parser, opt, tool_calls, end])
}

/// `analyze_tools::build_tool_parser_tag_tagged` (chat-auto-parser-generator.cpp:359-476)
fn build_tool_parser_tag_tagged(
    p: &mut ChatPegBuilder,
    tools: &AnalyzeTools,
    inputs: &GenerationParams,
    reasoning_parser: ParserId,
) -> ParserId {
    let value_suffix = tools.arguments.value_suffix.clone();
    let u = p.p.until(&value_suffix);
    // registers the named rule; the arg body below reaches it via ref_
    p.p.rule("until-suffix", u, false);

    let mut tool_choice_alts: Vec<ParserId> = Vec::new();

    foreach_function(&inputs.tools, |func| {
        let name = func
            .at("name")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("")
            .to_string();

        // Build parser for each argument, separating required and optional
        let mut required_parsers: Vec<ParserId> = Vec::new();
        let mut optional_parsers: Vec<ParserId> = Vec::new();
        foreach_parameter(func, |param, doc| {
            let arg = {
                let open_name = p.p.literal(&tools.arguments.name_prefix);
                let nm = p.p.literal(&param.name);
                let tan = p.tool_arg_name(nm);
                let inner = p.p.sequence(&[open_name, tan]);
                let n_suffix = p.p.literal(&tools.arguments.name_suffix);
                let oseq = p.p.sequence(&[inner, n_suffix]);
                let tao = p.tool_arg_open(oseq);
                let vp = p.p.literal(&tools.arguments.value_prefix);
                let may_be_string = doc.may_be_string(param.schema);
                let value_parser = if may_be_string {
                    let until = p.p.ref_("until-suffix");
                    let sv = p.tool_arg_string_value(until);
                    let vs = p.p.literal(&tools.arguments.value_suffix);
                    let tac = p.tool_arg_close(vs);
                    let child = p.p.sequence(&[sv, tac]);
                    p.p.ac(child, &[tools.arguments.value_suffix.clone()])
                } else {
                    let j = p.p.json();
                    let sch = p.p.schema_node(
                        j,
                        &format!("tool-{name}-arg-{}-schema", param.name),
                        Rc::clone(doc),
                        param.schema,
                        false,
                    );
                    let jv = p.tool_arg_json_value(sch);
                    let vs = p.p.literal(&tools.arguments.value_suffix);
                    let tac = p.tool_arg_close(vs);
                    p.p.sequence(&[jv, tac])
                };
                let aseq = p.p.sequence(&[tao, vp, value_parser]);
                p.tool_arg(aseq)
            };

            let named_arg =
                p.p.rule(&format!("tool-{name}-arg-{}", param.name), arg, false);
            if param.required {
                required_parsers.push(named_arg);
            } else {
                optional_parsers.push(named_arg);
            }
        });

        // Required arg sequence in definition order
        let mut args_seq = p.p.eps();
        for (i, &rp) in required_parsers.iter().enumerate() {
            if i > 0 {
                let sp = p.p.space();
                args_seq = p.p.sequence(&[args_seq, sp]);
            }
            args_seq = p.p.sequence(&[args_seq, rp]);
        }

        // Optional args with flexible ordering
        if !optional_parsers.is_empty() {
            let any_opt = p.p.choice(&optional_parsers);
            let sp = p.p.space();
            let rseq = p.p.sequence(&[sp, any_opt]);
            let tail = p.p.repeat3(rseq, 0, -1);
            args_seq = p.p.sequence(&[args_seq, tail]);
        }

        if !tools.arguments.start.is_empty() {
            let lit = p.p.literal(&tools.arguments.start);
            args_seq = p.p.sequence(&[lit, args_seq]);
        }
        if !tools.arguments.end.is_empty() {
            let lit = p.p.literal(&tools.arguments.end);
            args_seq = p.p.sequence(&[args_seq, lit]);
        }

        // call_id parser based on position (if supported)
        let mut call_id_section = p.p.eps();
        let mut have_call_id = false;
        if tools.call_id.pos == CallIdPosition::BetweenFuncAndArgs
            && !tools.call_id.prefix.is_empty()
            && (!tools.call_id.suffix.is_empty() || !tools.arguments.start.is_empty())
        {
            have_call_id = true;
            if !tools.call_id.suffix.is_empty() {
                let pre = p.p.literal(&tools.call_id.prefix);
                let until = p.p.until(&tools.call_id.suffix);
                let id = p.tool_id(until);
                let suf = p.p.literal(&tools.call_id.suffix);
                let inner = p.p.sequence(&[pre, id, suf]);
                call_id_section = p.p.optional(inner);
            } else {
                let pre = p.p.literal(&tools.call_id.prefix);
                let until = p.p.until(&tools.arguments.start);
                let id = p.tool_id(until);
                let oseq = p.p.sequence(&[pre, id]);
                call_id_section = p.p.optional(oseq);
            }
        }

        // Only peek for an arg tag when there are required args that must
        // follow (#20650)
        let atomic_peek = if !tools.arguments.name_prefix.is_empty() && !required_parsers.is_empty()
        {
            let lit = p.p.literal(&tools.arguments.name_prefix);
            Some(p.p.peek(lit))
        } else {
            None
        };
        let func_parser = build_func_parser(
            p,
            tools,
            &name,
            call_id_section,
            have_call_id,
            args_seq,
            atomic_peek,
        );
        tool_choice_alts.push(p.p.rule(&format!("tool-{name}"), func_parser, false));
    });

    let tool_choice = p.p.choice(&tool_choice_alts);
    let require_tools = inputs.tool_choice == ChatToolChoice::Required;

    let mut tool_calls;
    if !tools.format.per_call_start.is_empty() {
        let pcs = p.p.literal(&tools.format.per_call_start);
        let pce = p.p.literal(&tools.format.per_call_end);
        let wrapped_call = {
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            p.p.sequence(&[pcs, sp1, tool_choice, sp2, pce])
        };
        if inputs.parallel_tool_calls {
            let sp1 = p.p.space();
            let mseq = p.p.sequence(&[sp1, wrapped_call]);
            let more = p.p.zero_or_more(mseq);
            let sp2 = p.p.space();
            let body = p.p.sequence(&[wrapped_call, more, sp2]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            let sp = p.p.space();
            let body = p.p.sequence(&[wrapped_call, sp]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        }
        if !tools.format.section_start.is_empty() {
            let lit = p.p.literal(&tools.format.section_start);
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let end = if tools.format.section_end.is_empty() {
                p.p.end()
            } else {
                let e = p.p.literal(&tools.format.section_end);
                let sp = p.p.space();
                p.p.sequence(&[e, sp])
            };
            let body = p.p.sequence(&[lit, sp1, tool_calls, sp2, end]);
            tool_calls = p.p.trigger_rule("tool-calls", body);
        }
    } else {
        let separator = ", "; // default
        if inputs.parallel_tool_calls {
            let s = p.p.literal(&tools.format.section_start);
            let sp1 = p.p.space();
            let lit = p.p.literal(separator);
            let mseq = p.p.sequence(&[lit, tool_choice]);
            let more = p.p.zero_or_more(mseq);
            let sp2 = p.p.space();
            let e = p.p.literal(&tools.format.section_end);
            let body = p.p.sequence(&[s, sp1, tool_choice, more, sp2, e]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            let s = p.p.literal(&tools.format.section_start);
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let e = p.p.literal(&tools.format.section_end);
            let body = p.p.sequence(&[s, sp1, tool_choice, sp2, e]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        }
    }

    if !require_tools {
        tool_calls = p.p.optional(tool_calls);
    }

    let trigger_marker = if !tools.format.section_start.is_empty() {
        tools.format.section_start.clone()
    } else {
        tools.format.per_call_start.clone()
    };
    let content_before_tools = if trigger_marker.is_empty() {
        p.p.eps()
    } else {
        p.p.until(&trigger_marker)
    };
    let c = p.content(content_before_tools);
    let opt = p.p.optional(c);
    let end = p.p.end();
    p.p.sequence(&[reasoning_parser, opt, tool_calls, end])
}

// ---------------------------------------------------------------------------
// templates apply — common_chat_templates_apply (chat.cpp:1225-1439)
// ---------------------------------------------------------------------------

/// `common_chat_templates_inputs` (chat.h:249-266)
pub struct TemplatesInputs {
    pub messages: Vec<ChatMsg>,
    pub grammar: String,
    pub json_schema: String,
    pub add_generation_prompt: bool,
    /// `continue_final_message` (chat.h:254)
    pub continue_final_message: ChatContinuation,
    pub use_jinja: bool,
    pub tools: Vec<ChatTool>,
    pub tool_choice: ChatToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_format: ReasoningFormat,
    pub enable_thinking: bool,
    /// pinned clock for the datetime/date_string bindings (the reference reads
    /// `std::chrono::system_clock::now()`); None ⇒ current time
    pub now: Option<i64>,
    pub chat_template_kwargs: Vec<(String, String)>,
    pub add_bos: bool,
    pub add_eos: bool,
    pub force_pure_content: bool,
}

impl Default for TemplatesInputs {
    fn default() -> Self {
        TemplatesInputs {
            messages: Vec::new(),
            grammar: String::new(),
            json_schema: String::new(),
            add_generation_prompt: true,
            continue_final_message: ChatContinuation::None,
            use_jinja: true,
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Auto,
            parallel_tool_calls: false,
            reasoning_format: ReasoningFormat::None,
            enable_thinking: true,
            now: None,
            chat_template_kwargs: Vec::new(),
            add_bos: false,
            add_eos: false,
            force_pure_content: false,
        }
    }
}

/// `common_chat_params` (chat.h:269-283)
#[derive(Default)]
pub struct ChatParams {
    pub format: ChatFormat,
    pub prompt: String,
    pub grammar: String,
    pub grammar_lazy: bool,
    pub generation_prompt: String,
    pub supports_thinking: bool,
    pub thinking_start_tag: String,
    pub thinking_end_tags: Vec<String>,
    pub grammar_triggers: Vec<GrammarTrigger>,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
    /// serialized [`PegArena`] (the C++ stores `parser.save()`)
    pub parser: String,
    /// message delimiters for context reuse (chat.cpp:1341-1349); role +
    /// delimiter string (tokenization happens in the integrator)
    pub message_delimiters: Vec<(String, String)>,
}

/// `common_chat_try_specialized_template` (chat.cpp:1090-1223) — dispatch
/// into the specialized handlers under `common/parsers/*.cpp` (ported in
/// [`crate::chat_parsers`]); returns None when no handler matches so the
/// differential autoparser takes over.
fn try_specialized_template(
    tmpl: &ChatTemplate,
    src: &str,
    params: &mut GenerationParams,
) -> Result<Option<ChatParams>, String> {
    crate::chat_parsers::try_specialized_template(tmpl, src, params)
}

/// `common_chat_templates_apply_jinja` (chat.cpp:1225-1366)
pub fn chat_templates_apply_jinja(
    tmpls: &ChatTemplates,
    inputs: &TemplatesInputs,
) -> Result<ChatParams, String> {
    let mut params = GenerationParams {
        tools: tools_to_json_oaicompat(&inputs.tools),
        reasoning_format: inputs.reasoning_format,
        enable_thinking: inputs.enable_thinking,
        parallel_tool_calls: inputs.parallel_tool_calls,
        grammar: inputs.grammar.clone(),
        add_generation_prompt: inputs.add_generation_prompt,
        add_bos: tmpls.add_bos,
        add_eos: tmpls.add_eos,
        ..GenerationParams::default()
    };
    params.now = inputs.now.unwrap_or(params.now);

    let use_tool_template = params.tools.is_array() && tmpls.template_tool_use.is_some();
    let tmpl = if use_tool_template {
        tmpls.template_tool_use.as_ref().unwrap()
    } else {
        &tmpls.template_default
    };
    let src = tmpl.source();
    let caps = tmpl.original_caps().clone();

    let mut messages_to_render: Vec<ChatMsg> = inputs.messages.clone();
    // StepFun: trim message contents (chat.cpp:1235-1241)
    if src.contains("You have access to the following functions in JSONSchema format") {
        workaround::trim_all_content(&mut messages_to_render);
    }

    params.messages = render_message_to_json(&messages_to_render, tmpl.original_caps());
    params.tool_choice = inputs.tool_choice;

    // chat.cpp:1252-1271: continue_final_message handling
    params.continue_final_message = inputs.continue_final_message;
    if params.continue_final_message != ChatContinuation::None {
        params.add_generation_prompt = false;

        if !inputs.messages.is_empty() {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages.last().unwrap().clone();
            if let Json::Array(msgs) = &mut params.messages {
                msgs.pop();
            }
        }

        if params.continue_final_message == ChatContinuation::Auto && !inputs.messages.is_empty() {
            // Resolve based on message content
            params.continue_final_message = ChatContinuation::Content;
            if !params.continue_msg.reasoning_content.is_empty()
                && params.continue_msg.content.is_empty()
                && params.continue_msg.content_parts.is_empty()
            {
                params.continue_final_message = ChatContinuation::Reasoning;
            }
        }
    }

    if !src.contains("<|channel|>") {
        // map developer to system for all models except GPT-OSS
        workaround::map_developer_role_to_system(&mut params.messages);
    }

    if !caps.supports_system_role {
        workaround::system_message_not_supported(&mut params.messages);
    }

    if caps.supports_tool_calls {
        // some templates require a non-null content field in tool call messages
        workaround::requires_non_null_content(&mut params.messages);
    }

    if caps.supports_object_arguments {
        workaround::func_args_not_string(&mut params.messages)?;
    }

    params.extra_context = chat_extra_context();
    for (k, v) in &inputs.chat_template_kwargs {
        params
            .extra_context
            .set(k, Json::parse(v).unwrap_or(Json::String(v.clone())));
    }

    if !inputs.json_schema.is_empty() {
        params.json_schema = Json::parse(&inputs.json_schema)
            .map_err(|e| format!("Failed to parse json_schema: {e}"))?;
    }

    if params.tools.is_array() {
        if params.tool_choice != ChatToolChoice::None && !params.grammar.is_empty() {
            return Err("Cannot specify grammar with tools".to_string());
        }
        if caps.supports_tool_calls && !caps.supports_tools {
            // "Template supports tool calls but does not natively describe
            // tools. The fallback behaviour used may produce bad results…"
        }
    }

    if inputs.force_pure_content {
        // chat.cpp:1315-1329
        let mut data = ChatParams::default();
        let mut params_copy = params.clone();
        params_copy.reasoning_format = ReasoningFormat::None;
        data.prompt = template_direct_apply_impl(tmpl, &params_copy, None, None, None)?;
        data.generation_prompt = template_generation_prompt_impl(tmpl, &params, None, None, None)?;
        data.format = ChatFormat::PegNative;
        let gen_prompt = data.generation_prompt.clone();
        data.parser = build_chat_peg_parser(move |p| {
            let pre = p.prefix(&gen_prompt, "");
            let rest = p.p.rest();
            let c = p.content(rest);
            p.p.sequence(&[pre, c])
        })?
        .save();
        return Ok(data);
    }

    if let Some(result) = try_specialized_template(tmpl, src, &mut params)? {
        return Ok(result);
    }

    // chat.cpp:1335-1365: the differential autoparser
    let mut autoparser = Autoparser::default();
    autoparser.analyze_template(tmpl);

    let mut auto_params = generate_parser_with(tmpl, &params, &autoparser).map_err(|e| {
        format!(
            "Unable to generate parser for this template. Automatic parser generation failed: {e}"
        )
    })?;

    let mut delimiters: Vec<(String, String)> = Vec::new();
    if !autoparser.assistant_start.is_empty() {
        delimiters.push(("assistant".to_string(), autoparser.assistant_start.clone()));
    }
    if !autoparser.user_start.is_empty() {
        delimiters.push(("user".to_string(), autoparser.user_start.clone()));
    }
    auto_params.message_delimiters = delimiters;

    auto_params.supports_thinking = autoparser.reasoning.mode != ReasoningMode::None;
    if auto_params.supports_thinking {
        auto_params.thinking_start_tag = trim_whitespace(&autoparser.reasoning.start).to_string();
        let end_tag = trim_whitespace(&autoparser.reasoning.end).to_string();
        if !end_tag.is_empty() {
            auto_params.thinking_end_tags = vec![end_tag];
        }
    }
    // validate the serialized parser round-trips (chat.cpp:1359-1360)
    let mut arena = PegArena::default();
    arena.load(&auto_params.parser)?;
    Ok(auto_params)
}

/// `common_chat_templates_apply_legacy` (chat.cpp:1368-1432) — the `--no-jinja`
/// route via `llama_chat_apply_template`; tools are dropped, grammar kept.
fn chat_templates_apply_legacy(
    tmpls: &ChatTemplates,
    inputs: &TemplatesInputs,
) -> Result<ChatParams, String> {
    use crate::chat::{self, ChatMessage, Role};

    let mut chat: Vec<ChatMessage> = Vec::new();
    let mut contents: Vec<String> = Vec::new();
    for msg in &inputs.messages {
        let mut content = msg.content.clone();
        for part in &msg.content_parts {
            if part.ty != "text" && part.ty != "media_marker" {
                continue;
            }
            if !content.is_empty() {
                content += "\n";
            }
            content += &part.text;
        }
        contents.push(content);
    }
    for (i, content) in contents.iter().enumerate() {
        let msg = &inputs.messages[i];
        let role = Role::from_str(&msg.role).unwrap_or(Role::User);
        chat.push(ChatMessage::new(role, content.clone()));
    }

    let src = tmpls.template_default.source();
    let prompt = chat::apply_named(Some(src), &chat, inputs.add_generation_prompt)
        .map_err(|_| "this custom template is not supported, try using --jinja".to_string())?;

    let mut params = ChatParams::default();
    params.prompt = prompt;
    if !inputs.json_schema.is_empty() {
        params.grammar =
            crate::json_schema::json_schema_to_grammar(&Json::parse(&inputs.json_schema)?, true)?;
    } else {
        params.grammar = inputs.grammar.clone();
    }
    Ok(params)
}

/// `common_chat_templates_apply` (chat.cpp:1434-1439)
pub fn chat_templates_apply(
    tmpls: &ChatTemplates,
    inputs: &TemplatesInputs,
) -> Result<ChatParams, String> {
    if inputs.use_jinja {
        chat_templates_apply_jinja(tmpls, inputs)
    } else {
        chat_templates_apply_legacy(tmpls, inputs)
    }
}

// ---------------------------------------------------------------------------
// parsing model output — common_chat_parse / common_chat_peg_parse (chat.cpp:1441-1523)
// ---------------------------------------------------------------------------

/// `common_chat_parser_params` (chat.h:287-303)
pub struct ChatParserParams {
    pub format: ChatFormat,
    pub reasoning_format: ReasoningFormat,
    pub reasoning_in_content: bool,
    pub generation_prompt: String,
    pub parse_tool_calls: bool,
    pub is_continuation: bool,
    pub echo: bool,
    pub debug: bool,
    pub parser: PegArena,
}

impl ChatParserParams {
    /// from a [`ChatParams`] (chat.h:299-302) — loads the serialized parser
    pub fn from_chat_params(params: &ChatParams) -> Result<ChatParserParams, String> {
        let mut parser = PegArena::default();
        parser.load(&params.parser)?;
        Ok(ChatParserParams {
            format: params.format,
            reasoning_format: ReasoningFormat::None,
            reasoning_in_content: false,
            generation_prompt: params.generation_prompt.clone(),
            parse_tool_calls: true,
            is_continuation: false,
            echo: false,
            debug: false,
            parser,
        })
    }
}

impl Default for ChatParserParams {
    fn default() -> Self {
        ChatParserParams {
            format: ChatFormat::ContentOnly,
            reasoning_format: ReasoningFormat::None,
            reasoning_in_content: false,
            generation_prompt: String::new(),
            parse_tool_calls: true,
            is_continuation: false,
            echo: false,
            debug: false,
            parser: PegArena::default(),
        }
    }
}

/// `common_chat_peg_parse` (chat.cpp:1447-1523)
pub fn chat_peg_parse(
    src_parser: &PegArena,
    input: &str,
    is_partial: bool,
    params: &ChatParserParams,
) -> Result<ChatMsg, String> {
    let default_parser;
    let parser: &PegArena = if src_parser.is_empty() {
        // "No parser definition detected, assuming pure content parser."
        default_parser = build_chat_peg_parser(|p| {
            let rest = p.p.rest();
            let c = p.content(rest);
            let end = p.p.end();
            p.p.sequence(&[c, end])
        })?;
        &default_parser
    } else {
        src_parser
    };

    let effective_input = if params.generation_prompt.is_empty() {
        input.to_string()
    } else {
        format!("{}{}", params.generation_prompt, input)
    };

    let mut flags: ParseFlags = PARSE_FLAG_LENIENT;
    if params.debug {
        flags |= PARSE_FLAG_DEBUG;
    }

    let mut ctx = ParseContext::new(&effective_input, flags);
    let result = parser.parse(&mut ctx, 0).map_err(|e| {
        format!("The model produced output that does not match the expected format: {e}")
    })?;

    if result.fail() {
        // During partial parsing, return partial results from captured AST nodes
        if is_partial && result.end > 0 {
            let mut msg = ChatMsg {
                role: "assistant".to_string(),
                ..Default::default()
            };
            map_by_format(params.format, &mut msg, &ctx, &result);
            return Ok(msg);
        }
        let unparsed = &effective_input[result.end..];
        return Err(format!(
            "The model produced output that does not match the expected {} format (unparsed: {unparsed})",
            chat_format_name(params.format).unwrap_or("Content-only"),
        ));
    }

    let mut msg = ChatMsg {
        role: "assistant".to_string(),
        ..Default::default()
    };
    map_by_format(params.format, &mut msg, &ctx, &result);
    Ok(msg)
}

/// the format-dispatched mapper selection in `common_chat_peg_parse`
/// (chat.cpp:1468-1475 / 1494-1501): gemma4 and MiniMax-M3 use dedicated
/// mappers, everything else the generic tag mapper
fn map_by_format(format: ChatFormat, msg: &mut ChatMsg, ctx: &ParseContext, result: &ParseResult) {
    match format {
        ChatFormat::PegGemma4 => {
            crate::chat_parsers::ChatPegGemma4Mapper::new(msg).from_ast(ctx, result)
        }
        ChatFormat::PegMinimaxM3 => {
            crate::chat_parsers::ChatPegMinimaxM3Mapper::new(msg).from_ast(ctx, result)
        }
        _ => ChatPegMapper::new(msg).from_ast(ctx, result),
    }
}

/// `common_chat_parse` (chat.cpp:1441-1445)
pub fn chat_parse(
    input: &str,
    is_partial: bool,
    params: &ChatParserParams,
) -> Result<ChatMsg, String> {
    chat_peg_parse(&params.parser, input, is_partial, params)
}
