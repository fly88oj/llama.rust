//! chat_parsers.rs — literal port of the specialized chat-template handlers
//! under `common/parsers/*.cpp` plus their dispatch table
//! (`common_chat_try_specialized_template`, common/chat.cpp:1090-1223).
//!
//! Mapping (C symbol → Rust symbol here):
//!   * parsers/ministral3.cpp     `common_chat_params_init_ministral_3`       → [`chat_params_init_ministral_3`]
//!   * parsers/gpt-oss.cpp        `common_chat_params_init_gpt_oss`           → [`chat_params_init_gpt_oss`]
//!   * parsers/muse-glimmer.cpp   `common_chat_params_init_muse_glimmer`      → [`chat_params_init_muse_glimmer`]
//!   * parsers/functionary-v3-2.cpp `common_chat_params_init_functionary_v3_2` → [`chat_params_init_functionary_v3_2`]
//!   * parsers/kimi-k2.cpp        `common_chat_params_init_kimi_k2`           → [`chat_params_init_kimi_k2`]
//!   * parsers/kimi-k3.cpp        `common_chat_params_init_kimi_k3`           → [`chat_params_init_kimi_k3`]
//!   * parsers/ling3.cpp          `common_chat_params_init_ling3`             → [`chat_params_init_ling3`]
//!   * parsers/cohere2moe.cpp     `common_chat_params_init_cohere2moe`        → [`chat_params_init_cohere2moe`]
//!   * parsers/lfm2.cpp           `is_lfm2_template` / `common_chat_params_init_lfm2` → [`is_lfm2_template`] / [`chat_params_init_lfm2`]
//!   * parsers/gigachat-v3.cpp    `common_chat_params_init_gigachat_v3`       → [`chat_params_init_gigachat_v3`]
//!   * parsers/minimax-m3.cpp     `common_chat_params_init_minimax_m3`        → [`chat_params_init_minimax_m3`]
//!   * parsers/deepseek.cpp       `common_chat_params_init_deepseek_v3_2`     → [`chat_params_init_deepseek_v3_2`]
//!   * parsers/gemma4.cpp         `common_chat_params_init_gemma4`            → [`chat_params_init_gemma4`]
//!   * parsers/minicpm5.cpp       `common_chat_params_init_minicpm5`          → [`chat_params_init_minicpm5`]
//!   * parsers/qwen3-coder.cpp    `common_chat_params_init_qwen3_coder`       → [`chat_params_init_qwen3_coder`]
//!   * chat-peg-parser.cpp:956-1232 `common_chat_peg_{gemma4,minimax_m3}_mapper` → [`ChatPegGemma4Mapper`] / [`ChatPegMinimaxM3Mapper`]
//!
//! The dispatch order below is chat.cpp:1093-1221 verbatim; the first
//! detection whose needles match wins.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::chat_tools::{chat_tag, ChatMsg, ChatParams, ChatToolCall};
use crate::chat_tools::{
    escape_json_string_inner, foreach_function, foreach_parameter, string_ends_with,
    template_direct_apply_impl, template_generation_prompt_impl, tool_parameters, ChatContinuation,
    ChatFormat, ChatPegBuilder, ChatTemplate, ChatToolChoice, GenerationParams, GrammarTrigger,
    ReasoningFormat,
};
use crate::json_schema::{schema_from_json, Json, SchemaDocument, SchemaKind, ValueType};
use crate::peg::{AstNode, ParseContext, ParseResult, ParserId, INVALID_AST_ID};

// Local builder-call helpers: the C++ composes parsers as expressions
// (`p.literal("a") + p.space() + p.rule(...)`); in Rust each sub-parser must
// be evaluated before the receiver is borrowed, so calls that mix the builder
// into their arguments go through these (arguments are bound first, then the
// single receiver borrow happens). `pegN!` targets `p.p.<method>`, `pegcN!`
// targets the ChatPegBuilder `<method>` itself, `peg_seq!` builds a sequence.
macro_rules! peg1 {
    ($b:expr, $m:ident, $a:expr) => {{
        let t = ($a,);
        $b.p.$m(t.0)
    }};
}
macro_rules! pegc1 {
    ($b:expr, $m:ident, $a:expr) => {{
        let t = ($a,);
        $b.$m(t.0)
    }};
}
#[allow(unused_macros)]
macro_rules! peg2 {
    ($b:expr, $m:ident, $a:expr, $c:expr) => {{
        let t = ($a, $c);
        $b.p.$m(t.0, t.1)
    }};
}
#[allow(unused_macros)]
macro_rules! pegc2 {
    ($b:expr, $m:ident, $a:expr, $c:expr) => {{
        let t = ($a, $c);
        $b.$m(t.0, t.1)
    }};
}
macro_rules! peg3 {
    ($b:expr, $m:ident, $a:expr, $c:expr, $d:expr) => {{
        let t = ($a, $c, $d);
        $b.p.$m(t.0, t.1, t.2)
    }};
}
#[allow(unused_macros)]
macro_rules! peg4 {
    ($b:expr, $m:ident, $a:expr, $c:expr, $d:expr, $e:expr) => {{
        let t = ($a, $c, $d, $e);
        $b.p.$m(t.0, t.1, t.2, t.3)
    }};
}
#[allow(unused_macros)]
macro_rules! peg5 {
    ($b:expr, $m:ident, $a:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {{
        let t = ($a, $c, $d, $e, $f);
        $b.p.$m(t.0, t.1, t.2, t.3, t.4)
    }};
}
#[allow(unused_macros)]
macro_rules! peg_seq {
    ($b:expr, $($e:expr),+ $(,)?) => {{
        let ids = [$($e),+];
        $b.p.sequence(&ids)
    }};
}

/// `msg.erase(key)` on an object message (nlohmann `erase`)
fn json_erase(msg: &mut Json, key: &str) {
    if let Json::Object(fields) = msg {
        fields.retain(|(k, _)| k != key);
    }
}

/// miniMax-M3 mapper tag constants (chat-peg-parser.h:45-47)
mod mm3_tag {
    pub const TOOL_ARG_OBJECT: &str = "tool-arg-object";
    pub const TOOL_ARG_ARRAY: &str = "tool-arg-array";
    pub const TOOL_ARG_ITEM: &str = "tool-arg-item";
}

// ---------------------------------------------------------------------------
// LFM2 template detection (parsers/lfm2.cpp:3-8)
// ---------------------------------------------------------------------------

/// `is_lfm2_template` (parsers/lfm2.cpp:5-8)
pub fn is_lfm2_template(src: &str) -> bool {
    src.contains("<|tool_list_start|>") && src.contains("<|tool_list_end|>")
}

// ---------------------------------------------------------------------------
// Ministral / Magistral Large 3 (parsers/ministral3.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_ministral_3` (parsers/ministral3.cpp:3-126)
pub(crate) fn chat_params_init_ministral_3(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    let mut adjusted_messages = Json::Array(Vec::new());
    if let Json::Array(msgs) = &inputs.messages {
        for msg in msgs {
            let role = msg.at("role").and_then(|v| v.get_str().ok()).unwrap_or("");
            if role != "system" && role != "assistant" {
                // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
                if let Json::Array(a) = &mut adjusted_messages {
                    a.push(msg.clone());
                }
                continue;
            }

            let mut content = Json::Array(Vec::new());

            // If message contains `reasoning_content`, add it as a block of type `thinking`
            if msg
                .at("reasoning_content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                if let Json::Array(a) = &mut content {
                    a.push(Json::Object(vec![
                        ("type".to_string(), Json::String("thinking".to_string())),
                        (
                            "thinking".to_string(),
                            msg.at("reasoning_content").cloned().unwrap(),
                        ),
                    ]));
                }
            }

            // If message contains `content`, add it as a block of type `text`
            if let Some(c) = msg.at("content") {
                if c.is_string() {
                    if let Json::Array(a) = &mut content {
                        a.push(Json::Object(vec![
                            ("type".to_string(), Json::String("text".to_string())),
                            ("text".to_string(), c.clone()),
                        ]));
                    }
                } else if c.is_array() {
                    if let Json::Array(blocks) = c {
                        if let Json::Array(a) = &mut content {
                            a.extend(blocks.iter().cloned());
                        }
                    }
                }
            }

            let mut adjusted = msg.clone();
            adjusted.set("content", content);
            json_erase(&mut adjusted, "reasoning_content");
            if let Json::Array(a) = &mut adjusted_messages {
                a.push(adjusted);
            }
        }
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    // `include_grammar` starts true and is cleared in the content-only branch
    // of the parser builder below (parsers/ministral3.cpp:107)
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    data.supports_thinking = true;
    data.thinking_start_tag = "[THINK]".to_string();
    data.thinking_end_tags = vec!["[/THINK]".to_string()];
    data.prompt = template_direct_apply_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.generation_prompt =
        template_generation_prompt_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.format = ChatFormat::PegNative;
    data.preserved_tokens = vec![
        "[THINK]".to_string(),
        "[/THINK]".to_string(),
        "[TOOL_CALLS]".to_string(),
        "[ARGS]".to_string(),
    ];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("[THINK]{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("[/THINK]{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        // ChatPegBuilder deref
        let generation_prompt = p.p.eps();
        let reasoning = if extract_reasoning {
            let lit = p.p.literal("[THINK]");
            let until = p.p.until("[/THINK]");
            let r = p.reasoning(until);
            let close = p.p.literal("[/THINK]");
            let seq = p.p.sequence(&[lit, r, close]);
            p.p.optional(seq)
        } else {
            p.p.eps()
        };

        // Response format parser
        if has_response_format {
            // Ministral wants to emit json surrounded by code fences
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format", &inputs.json_schema, false);
            let c = p.content(sch);
            let fence1 = p.p.literal("```json");
            let a = p.p.spaced(reasoning, fence1);
            let b = p.p.spaced(a, c);
            let fence2 = p.p.literal("```");
            let seq = p.p.spaced(b, fence2);
            return p.p.sequence(&[generation_prompt, seq]);
        }

        // Tool call parser
        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            let mut tool_choice_alts: Vec<ParserId> = Vec::new();
            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                let schema = tool_parameters(function);

                // pegc1!(p, tool_open, pegc1!(p, tool_name, p.literal(name)) + "[ARGS]")"[ARGS]")
                let lit = p.p.literal(&name);
                let tn = p.tool_name(lit);
                let args_lit = p.p.literal("[ARGS]");
                let oseq = p.p.sequence(&[tn, args_lit]);
                let t_open = p.tool_open(oseq);
                let json_p = p.p.json();
                let sch =
                    p.p.schema(json_p, &format!("tool-{name}-schema"), &schema, false);
                let ta = p.tool_args(sch);
                // NB: no p.tool() wrapper here (parsers/ministral3.cpp:94-96):
                // tool_open(...) + tool_args(...)
                let seq = p.p.sequence(&[t_open, ta]);

                let r = p.p.rule(&format!("tool-{name}"), seq, false);
                tool_choice_alts.push(r);
            });
            let tool_choice = p.p.choice(&tool_choice_alts);

            let min_calls = if inputs.tool_choice == ChatToolChoice::Required {
                1
            } else {
                0
            };
            let max_calls = if inputs.parallel_tool_calls { -1 } else { 1 };
            // p.repeat("[TOOL_CALLS]" + tool_choice, min_calls, max_calls)
            let lit = p.p.literal("[TOOL_CALLS]");
            let seq = p.p.sequence(&[lit, tool_choice]);
            let rep = p.p.repeat3(seq, min_calls, max_calls);
            let tool_calls = p.p.trigger_rule("tool-call", rep);

            // reasoning << pegc1!(p, content, p.until("[TOOL_CALLS]")) << tool_calls
            let until = p.p.until("[TOOL_CALLS]");
            let c = p.content(until);
            let a = p.p.spaced(reasoning, c);
            let seq = p.p.spaced(a, tool_calls);
            return p.p.sequence(&[generation_prompt, seq]);
        }

        // Content only parser (include_grammar = false)
        let rest = p.p.rest();
        let c = p.content(rest);
        let seq = p.p.spaced(reasoning, c);
        p.p.sequence(&[generation_prompt, seq])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = has_tools && inputs.tool_choice == ChatToolChoice::Auto;

        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word("[TOOL_CALLS]")];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// GPT-OSS (parsers/gpt-oss.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_gpt_oss` (parsers/gpt-oss.cpp:3-158)
pub(crate) fn chat_params_init_gpt_oss(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    let mut adjusted_messages = Json::Array(Vec::new());
    if let Json::Array(msgs) = &inputs.messages {
        for msg in msgs {
            let mut msg = msg.clone();
            if msg
                .at("reasoning_content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                msg.set("thinking", msg.at("reasoning_content").cloned().unwrap());
                let has_calls = msg
                    .at("tool_calls")
                    .map(|t| t.is_array() && !t.empty())
                    .unwrap_or(false);
                if has_calls {
                    json_erase(&mut msg, "content");
                }
            }
            if let Json::Array(a) = &mut adjusted_messages {
                a.push(msg);
            }
        }
    }

    let mut prompt =
        template_direct_apply_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if inputs.is_inference && !inputs.add_generation_prompt {
        const RETURN_TOKEN: &str = "<|return|>";
        const END_TOKEN: &str = "<|end|>";
        if let Some(pos) = prompt.rfind(RETURN_TOKEN) {
            prompt.replace_range(pos..pos + RETURN_TOKEN.len(), END_TOKEN);
        }
    }

    data.prompt = prompt;
    data.generation_prompt =
        template_generation_prompt_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.message_delimiters = vec![
        ("assistant".to_string(), "<|start|>assistant".to_string()),
        ("user".to_string(), "<|start|>user".to_string()),
        ("system".to_string(), "<|start|>developer".to_string()),
        ("system".to_string(), "<|start|>system".to_string()),
        ("tool".to_string(), "<|start|>functions".to_string()),
    ];

    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>".to_string();
    data.thinking_end_tags = vec!["<|end|>".to_string()];

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = vec![
        "<|channel|>".to_string(),
        "<|constrain|>".to_string(),
        "<|message|>".to_string(),
        "<|start|>".to_string(),
        "<|end|>".to_string(),
    ];

    // Adjust prompt for continuation
    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!(
            "<|start|>assistant<|channel|>analysis<|message|>{}",
            msg.reasoning_content
        );
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!(
                "<|end|><|start|>assistant<|channel|>final<|message|>{}",
                msg.render_content("\n\n")?
            );
        }

        data.prompt += &data.generation_prompt;
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let start_lit = p.p.literal("<|start|>assistant");
        let start = p.p.rule("start", start_lit, false);
        let end_lit = p.p.literal("<|end|>");
        let end = p.p.rule("end", end_lit, false);
        let content_until = p.p.until("<|end|>");
        let content = p.p.rule("message-content", content_until, false);
        // auto channel = p.literal("<|channel|>") + (p.literal("commentary") | p.literal("analysis"));
        let ch_lit = p.p.literal("<|channel|>");
        let c1 = p.p.literal("commentary");
        let c2 = p.p.literal("analysis");
        let ch_alt = p.p.choice(&[c1, c2]);
        let channel = p.p.sequence(&[ch_lit, ch_alt]);
        let constrain_type = p.p.chars("[A-Za-z0-9_-]", 1, -1);

        // Occasionally, gpt-oss-20b will prefix channels with this commentary
        let sc_lit = p.p.literal("<|channel|>commentary");
        let ta_lit = p.p.literal(" to=assistant");
        let ta_opt = p.p.optional(ta_lit);
        let sc_seq = p.p.sequence(&[sc_lit, ta_opt]);
        let stray_commentary = p.p.optional(sc_seq);
        let an_lit = p.p.literal("<|channel|>analysis<|message|>");
        let start_analysis = p.p.sequence(&[stray_commentary, an_lit]);

        if extract_reasoning {
            let r = p.reasoning(content);
            let seq = p.p.sequence(&[start_analysis, r, end]);
            p.p.rule("analysis", seq, false);
        } else {
            let seq = p.p.sequence(&[start_analysis, content, end]);
            let c = p.content(seq);
            p.p.rule("analysis", c, false);
        }

        let analysis = p.p.ref_("analysis");
        let pm_lit = p.p.literal("<|channel|>commentary<|message|>");
        let pm_content = p.content(content);
        let pm_seq = p.p.sequence(&[pm_lit, pm_content, end]);
        let preamble = p.p.rule("preamble", pm_seq, false);
        let fm_lit = p.p.literal("<|channel|>final<|message|>");
        let fm_content = p.content(content);
        let fm_seq = p.p.sequence(&[stray_commentary, fm_lit, fm_content]);
        let final_msg = p.p.rule("final", fm_seq, false);

        // Consume any unsolicited tool calls, e.g. builtin functions
        let un_opt = p.p.optional(channel);
        let un_to = p.p.literal(" to=");
        let un_seq = p.p.sequence(&[un_opt, un_to, content, end]);
        let un_atomic = p.p.atomic(un_seq);
        let unsolicited = p.p.rule("unsolicited", un_atomic, false);

        let any_alt = p.p.choice(&[preamble, analysis]);
        let any = p.p.rule("any", any_alt, false);

        if has_response_format {
            // auto constraint = peg1!(p, optional, p.space() + peg1!(p, optional, p.literal("<|constrain|>")) + constrain_type)ain_type);
            let sp = p.p.space();
            let c_lit = p.p.literal("<|constrain|>");
            let c_opt = p.p.optional(c_lit);
            let seq = p.p.sequence(&[sp, c_opt, constrain_type]);
            let constraint = p.p.optional(seq);
            let rf_start = p.p.literal("<|channel|>final");
            let rf_msg = p.p.literal("<|message|>");
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
            let rf_content = p.content(sch);
            let rf_seq = p.p.sequence(&[rf_start, constraint, rf_msg, rf_content]);
            let response_format = p.p.rule("response-format", rf_seq, false);

            let zo_seq = p.p.sequence(&[start, analysis]);
            let zo = p.p.zero_or_more(zo_seq);
            return p.p.sequence(&[zo, start, response_format]);
        }

        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            let mut tool_choice_alts: Vec<ParserId> = Vec::new();

            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                let params = tool_parameters(function);

                let fn_lit = p.p.literal(" to=functions.");
                let nm_lit = p.p.literal(&name);
                let tn = p.tool_name(nm_lit);
                let func_name = p.p.sequence(&[fn_lit, tn]);
                let sp = p.p.space();
                let c_lit = p.p.literal("<|constrain|>");
                let c_opt = p.p.optional(c_lit);
                let cseq = p.p.sequence(&[sp, c_opt, constrain_type]);
                let constraint = p.p.optional(cseq);
                let json_p = p.p.json();
                let sch =
                    p.p.schema(json_p, &format!("tool-{name}-schema"), &params, false);
                let args = p.tool_args(sch);

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                let msg_lit = p.p.literal("<|message|>");
                let role_seq = p.p.sequence(&[func_name, channel, constraint, msg_lit]);
                let role_open = p.tool_open(role_seq);
                let tool_in_role = pegc1!(p, tool, p.p.sequence(&[role_open, args]));

                // recipient in channel header
                //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                let ch_seq = p.p.sequence(&[channel, func_name, constraint, msg_lit]);
                let ch_open = p.tool_open(ch_seq);
                let tool_in_channel = pegc1!(p, tool, p.p.sequence(&[ch_open, args]));

                let alt = p.p.choice(&[tool_in_role, tool_in_channel]);
                let r = p.p.rule(&format!("tool-{name}"), alt, false);
                tool_choice_alts.push(r);
            });

            let tool_choice = p.p.choice(&tool_choice_alts);

            let tool_call = p.p.trigger_rule("tool-call", tool_choice);

            let zo_seq = p.p.sequence(&[start, any]);
            let zo = p.p.zero_or_more(zo_seq);
            if inputs.tool_choice == ChatToolChoice::Required {
                return p.p.sequence(&[zo, start, tool_call]);
            }

            let alt = p.p.choice(&[tool_call, final_msg]);
            return p.p.sequence(&[zo, start, alt]);
        }

        let zo_seq = p.p.sequence(&[start, any]);
        let zo = p.p.zero_or_more(zo_seq);
        let alt = p.p.choice(&[final_msg, unsolicited]);
        p.p.sequence(&[zo, start, alt])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![
            GrammarTrigger::pattern("^\\s+to$"),
            GrammarTrigger::pattern("^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$"),
            GrammarTrigger::pattern("<\\|start\\|>assistant(\\s+to)"),
            GrammarTrigger::pattern(
                "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)",
            ),
        ];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// LLM-jp-4.1 Harmony (parsers/llm-jp-harmony.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_llm_jp_harmony` (parsers/llm-jp-harmony.cpp:8-164).
/// LLM-jp-4.1: the GPT-OSS (Harmony) format with two differences
///  - the tokenizer emits a space after every special token:
///    "<|channel|> analysis<|message|> ..."
///  - parallel tool calls are consecutive assistant messages, all but the
///    last closed by <|end|>
pub(crate) fn chat_params_init_llm_jp_harmony(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    // Copy reasoning to the "thinking" field as expected by the template
    let mut adjusted_messages = Json::Array(Vec::new());
    if let Json::Array(msgs) = &inputs.messages {
        for msg in msgs {
            let mut msg = msg.clone();
            if msg
                .at("reasoning_content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                msg.set("thinking", msg.at("reasoning_content").cloned().unwrap());
                let has_calls = msg
                    .at("tool_calls")
                    .map(|t| t.is_array() && !t.empty())
                    .unwrap_or(false);
                if has_calls {
                    json_erase(&mut msg, "content");
                }
            }
            if let Json::Array(a) = &mut adjusted_messages {
                a.push(msg);
            }
        }
    }

    let mut prompt =
        template_direct_apply_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if inputs.is_inference && !inputs.add_generation_prompt {
        const RETURN_TOKEN: &str = "<|return|>";
        const END_TOKEN: &str = "<|end|>";
        if let Some(pos) = prompt.rfind(RETURN_TOKEN) {
            prompt.replace_range(pos..pos + RETURN_TOKEN.len(), END_TOKEN);
        }
    }

    data.prompt = prompt;
    data.generation_prompt =
        template_generation_prompt_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.message_delimiters = vec![
        ("assistant".to_string(), "<|start|>assistant".to_string()),
        ("user".to_string(), "<|start|>user".to_string()),
        ("system".to_string(), "<|start|>developer".to_string()),
        ("system".to_string(), "<|start|>system".to_string()),
        ("tool".to_string(), "<|start|>functions".to_string()),
    ];

    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;

    data.thinking_start_tag = "<|channel|>analysis<|message|>".to_string();
    data.thinking_end_tags = vec!["<|end|>".to_string()];

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = vec![
        "<|channel|>".to_string(),
        "<|constrain|>".to_string(),
        "<|message|>".to_string(),
        "<|start|>".to_string(),
        "<|end|>".to_string(),
    ];

    // Adjust prompt for continuation
    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!(
            "<|start|>assistant<|channel|>analysis<|message|>{}",
            msg.reasoning_content
        );
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!(
                "<|end|><|start|>assistant<|channel|>final<|message|>{}",
                msg.render_content("\n\n")?
            );
        }

        data.prompt += &data.generation_prompt;
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        // tokenizer space after special tokens; not p.space() since GBNF
        // `space` allows one space only
        let sp = p.p.chars("[ ]", 0, -1);
        let ch_lit = p.p.literal("<|channel|>");
        let channel_tag = p.p.sequence(&[ch_lit, sp]);
        // one space only: keep an intentional leading space in the body
        let msg_lit = p.p.literal("<|message|>");
        let spc_lit = p.p.literal(" ");
        let spc_opt = p.p.optional(spc_lit);
        let message = p.p.sequence(&[msg_lit, spc_opt]);

        let s_lit = p.p.literal("<|start|>");
        let a_lit = p.p.literal("assistant");
        let start_seq = p.p.sequence(&[s_lit, sp, a_lit]);
        let start = p.p.rule("start", start_seq, false);
        let e_lit = p.p.literal("<|end|>");
        let end = p.p.rule("end", e_lit, false);
        let content_until = p.p.until("<|end|>");
        let content = p.p.rule("message-content", content_until, false);
        let c1 = p.p.literal("commentary");
        let c2 = p.p.literal("analysis");
        let ch_alt = p.p.choice(&[c1, c2]);
        let channel = p.p.sequence(&[channel_tag, ch_alt]);
        let constrain_type = p.p.chars("[A-Za-z0-9_-]", 1, -1);
        let sp_g = p.p.space();
        let cn_lit = p.p.literal("<|constrain|>");
        let cn_seq = p.p.sequence(&[cn_lit, sp]);
        let cn_opt = p.p.optional(cn_seq);
        let cn_full = p.p.sequence(&[sp_g, cn_opt, constrain_type]);
        let constraint = p.p.optional(cn_full);

        let an_lit = p.p.literal("analysis");
        let start_analysis = p.p.sequence(&[channel_tag, an_lit, message]);
        if extract_reasoning {
            let r = p.reasoning(content);
            let seq = p.p.sequence(&[start_analysis, r, end]);
            p.p.rule("analysis", seq, false);
        } else {
            let seq = p.p.sequence(&[start_analysis, content, end]);
            let c = p.content(seq);
            p.p.rule("analysis", c, false);
        }

        let analysis = p.p.ref_("analysis");
        let pb_lit = p.p.literal("commentary");
        let pb_content = p.content(content);
        let pb_seq = p.p.sequence(&[channel_tag, pb_lit, message, pb_content, end]);
        let preamble = p.p.rule("preamble", pb_seq, false);
        let fin_lit = p.p.literal("final");
        let fin_content = p.content(content);
        let fin_seq = p.p.sequence(&[channel_tag, fin_lit, message, fin_content]);
        let final_msg = p.p.rule("final", fin_seq, false);

        let any_alt = p.p.choice(&[preamble, analysis]);
        let any = p.p.rule("any", any_alt, false);

        if has_response_format {
            let f2_lit = p.p.literal("final");
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
            let rf_content = p.content(sch);
            let rf_seq = p.p.sequence(&[channel_tag, f2_lit, constraint, message, rf_content]);
            let response_format = p.p.rule("response-format", rf_seq, false);

            let zo_seq = p.p.sequence(&[start, analysis]);
            let zo = p.p.zero_or_more(zo_seq);
            return p.p.sequence(&[zo, start, response_format]);
        }

        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            let mut tool_choice_alts: Vec<ParserId> = Vec::new();

            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                let params = tool_parameters(function);

                let fn_lit = p.p.literal(" to=functions.");
                let nm_lit = p.p.literal(&name);
                let tn = p.tool_name(nm_lit);
                let func_name = p.p.sequence(&[fn_lit, tn]);
                let json_p = p.p.json();
                let sch =
                    p.p.schema(json_p, &format!("tool-{name}-schema"), &params, false);
                let args = p.tool_args(sch);

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                let role_seq = p.p.sequence(&[func_name, channel, constraint, message]);
                let role_open = p.tool_open(role_seq);
                let tool_in_role = pegc1!(p, tool, p.p.sequence(&[role_open, args]));

                // recipient in channel header
                //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                let ch_seq = p.p.sequence(&[channel, func_name, constraint, message]);
                let ch_open = p.tool_open(ch_seq);
                let tool_in_channel = pegc1!(p, tool, p.p.sequence(&[ch_open, args]));

                let alt = p.p.choice(&[tool_in_role, tool_in_channel]);
                let r = p.p.rule(&format!("tool-{name}"), alt, false);
                tool_choice_alts.push(r);
            });

            let tool_choice = p.p.choice(&tool_choice_alts);

            // parallel calls are separated by <|end|>; inside the trigger
            // rule so the lazy grammar covers all of them
            let tool_calls = if inputs.parallel_tool_calls {
                let es = p.p.sequence(&[end, start, tool_choice]);
                let zo = p.p.zero_or_more(es);
                p.p.sequence(&[tool_choice, zo])
            } else {
                tool_choice
            };
            let tool_call = p.p.trigger_rule("tool-call", tool_calls);

            let zo_seq = p.p.sequence(&[start, any]);
            let zo = p.p.zero_or_more(zo_seq);
            if inputs.tool_choice == ChatToolChoice::Required {
                return p.p.sequence(&[zo, start, tool_call]);
            }

            let alt = p.p.choice(&[tool_call, final_msg]);
            return p.p.sequence(&[zo, start, alt]);
        }

        let zo_seq = p.p.sequence(&[start, any]);
        let zo = p.p.zero_or_more(zo_seq);
        p.p.sequence(&[zo, start, final_msg])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![
            GrammarTrigger::pattern("^\\s+to$"),
            GrammarTrigger::pattern("^<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to=functions$"),
            GrammarTrigger::pattern("<\\|start\\|>\\s*assistant(\\s+to)"),
            GrammarTrigger::pattern(
                "<\\|start\\|>\\s*assistant(<\\|channel\\|>\\s*(?:commentary|analysis)\\s+to)",
            ),
        ];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Muse Glimmer (parsers/muse-glimmer.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_muse_glimmer` (parsers/muse-glimmer.cpp:10-138)
pub(crate) fn chat_params_init_muse_glimmer(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = "<|start|>assistant".to_string();
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;

    data.preserved_tokens = vec![
        "<|start|>".to_string(),
        "<|message|>".to_string(),
        "<|eom|>".to_string(),
        "<|eot|>".to_string(),
        // ATEM tool-call markup emitted on " to=<tool>" turns.
        "<atem:function_calls>".to_string(),
        "<atem:invoke".to_string(),
        "<atem:parameter".to_string(),
        "</atem:parameter>".to_string(),
        "</atem:invoke>".to_string(),
        "</atem:function_calls>".to_string(),
    ];

    data.message_delimiters = vec![
        ("assistant".to_string(), "<|start|>assistant".to_string()),
        ("user".to_string(), "<|start|>user".to_string()),
        ("system".to_string(), "<|start|>system".to_string()),
        ("tool".to_string(), "<|start|>tool".to_string()),
    ];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!(
            "<|start|>assistant to=self<|message|>{}",
            msg.reasoning_content
        );
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!(
                "<|eom|><|start|>assistant to=user<|message|>{}",
                msg.render_content("\n\n")?
            );
        }

        data.prompt += &data.generation_prompt;
    }

    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    // Constrained grammar whenever tools are offered or a response format is requested.
    let include_grammar = has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let start_lit = p.p.literal("<|start|>assistant");
        let start = p.p.rule("start", start_lit, false);

        if !extract_reasoning && !include_grammar {
            let rest = p.p.rest();
            let c = p.content(rest);
            return p.p.sequence(&[start, c]);
        }

        if extract_reasoning {
            let lit = p.p.literal(" to=self<|message|>");
            let until = p.p.until("<|eom|>");
            let r = p.reasoning(until);
            let eom = p.p.literal("<|eom|>");
            let seq = p.p.sequence(&[lit, r, eom]);
            p.p.rule("analysis", seq, false);
        } else {
            let lit = p.p.literal(" to=self<|message|>");
            let until = p.p.until("<|eom|>");
            let c = p.content(until);
            let eom = p.p.literal("<|eom|>");
            let seq = p.p.sequence(&[lit, c, eom]);
            p.p.rule("analysis", seq, false);
        }
        let analysis = p.p.ref_("analysis");

        let recipient_lit = p.p.literal(" to=user");
        let recipient = p.p.optional(recipient_lit);
        let msg_lit = p.p.literal("<|message|>");
        let until = p.p.until_one_of(&["<|eot|>", "<|eom|>"]);
        let c = p.content(until);
        let seq = p.p.sequence(&[recipient, msg_lit, c]);
        let final_msg = p.p.rule("final", seq, false);

        if has_response_format {
            let response_json = {
                let json_p = p.p.json();
                let sch = p.p.schema(
                    json_p,
                    "response-format-schema",
                    &inputs.json_schema,
                    false,
                );
                p.content(sch)
            };
            let jb = p.p.literal("```json");
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let je = p.p.literal("```");
            let fenced = p.p.sequence(&[jb, sp1, response_json, sp2, je]);
            let alt = p.p.choice(&[fenced, response_json]);
            let rf_seq = p.p.sequence(&[recipient, msg_lit, alt]);
            let response_format = p.p.rule("response-format", rf_seq, false);

            let zo_seq = p.p.sequence(&[start, analysis]);
            let zo = p.p.zero_or_more(zo_seq);
            return p.p.sequence(&[zo, start, response_format]);
        }

        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            // string_value = pegc2!(p, ac, //     pegc1!(p, tool_arg_string_value, p.until("</atem:parameter>")) + pegc1!(p, tool_arg_close, p.literal("</atem:parameter>")), //     "</atem:parameter>")/atem:parameter>")
            let until = p.p.until("</atem:parameter>");
            let sv = p.tool_arg_string_value(until);
            let close_lit = p.p.literal("</atem:parameter>");
            let tc = p.tool_arg_close(close_lit);
            let seq = p.p.sequence(&[sv, tc]);
            let delim = "</atem:parameter>".to_string();
            let string_value = p.p.ac(seq, &[delim]);

            let mut tool_choice_alts: Vec<ParserId> = Vec::new();
            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();

                let mut arg_rules: Vec<ParserId> = Vec::new();
                foreach_parameter(function, |prop, doc| {
                    // auto value_parser = p.eps();
                    let value_parser = if doc.may_be_string(prop.schema) {
                        string_value
                    } else {
                        let json_p = p.p.json();
                        let sch = p.p.schema_node(
                            json_p,
                            &format!("tool-{name}-arg-{}-schema", prop.name),
                            Rc::clone(doc),
                            prop.schema,
                            false,
                        );
                        let jv = p.tool_arg_json_value(sch);
                        let close_lit = p.p.literal("</atem:parameter>");
                        let tc = p.tool_arg_close(close_lit);
                        p.p.sequence(&[jv, tc])
                    };

                    // <atem:parameter name="KEY">
                    let open1 = p.p.literal("<atem:parameter name=\"");
                    let nm = p.p.literal(&prop.name);
                    let tan = p.tool_arg_name(nm);
                    let open2 = p.p.literal("\">");
                    let oseq = p.p.sequence(&[open1, tan, open2]);
                    let tao = p.tool_arg_open(oseq);
                    let arg_seq = p.p.sequence(&[tao, value_parser]);
                    arg_rules.push(p.tool_arg(arg_seq));
                });

                let mut args = p.p.eps();
                if !arg_rules.is_empty() {
                    let choice = p.p.choice(&arg_rules);
                    let sp = p.p.space();
                    let seq = p.p.sequence(&[choice, sp]);
                    args = p.p.zero_or_more(seq);
                }

                // to=<tool><|message|><atem:function_calls> <atem:invoke name="NAME">
                let to_lit = p.p.literal(" to=");
                let until = p.p.until("<|message|>");
                let to_seq = p.p.sequence(&[to_lit, until]);
                let fc_lit = p.p.literal("<|message|><atem:function_calls>");
                let sp1 = p.p.space();
                let inv_lit = p.p.literal("<atem:invoke name=\"");
                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let inv2 = p.p.literal("\">");
                let sp2 = p.p.space();
                let open_seq = p.p.sequence(&[to_seq, fc_lit, sp1, inv_lit, tn, inv2, sp2]);
                let t_open = p.tool_open(open_seq);
                let ta = p.tool_args(args);
                let cl1 = p.p.literal("</atem:invoke>");
                let cl_sp = p.p.space();
                let cl2 = p.p.literal("</atem:function_calls>");
                let cl_seq = p.p.sequence(&[cl1, cl_sp, cl2]);
                let t_close = p.tool_close(cl_seq);
                // tool_open << tool_args << tool_close
                let a = p.p.spaced(t_open, ta);
                let body = p.p.spaced(a, t_close);
                let tool_parser = p.tool(body);

                let r = p.p.rule(&format!("tool-{name}"), tool_parser, false);
                tool_choice_alts.push(r);
            });

            let tool_choice = p.p.choice(&tool_choice_alts);

            let tool_calls = if inputs.parallel_tool_calls {
                // tool_choice + zero_or_more("<|eom|>" + start + tool_choice)
                let eom = p.p.literal("<|eom|>");
                let seq = p.p.sequence(&[eom, start, tool_choice]);
                let more = p.p.zero_or_more(seq);
                let body = p.p.sequence(&[tool_choice, more]);
                p.p.trigger_rule("tool-call", body)
            } else {
                p.p.trigger_rule("tool-call", tool_choice)
            };

            let zo_seq = p.p.sequence(&[start, analysis]);
            let zo = p.p.zero_or_more(zo_seq);

            if inputs.tool_choice == ChatToolChoice::Required {
                return p.p.sequence(&[zo, start, tool_calls]);
            }
            let eom = p.p.literal("<|eom|>");
            let trailing_seq = p.p.sequence(&[eom, start, tool_calls]);
            let trailing_calls = p.p.optional(trailing_seq);
            let alt_body = p.p.sequence(&[final_msg, trailing_calls]);
            let alt = p.p.choice(&[tool_calls, alt_body]);
            return p.p.sequence(&[zo, start, alt]);
        }

        let zo_seq = p.p.sequence(&[start, analysis]);
        let zo = p.p.zero_or_more(zo_seq);
        p.p.sequence(&[zo, start, final_msg])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = !(has_response_format
            || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;
        data.grammar_triggers = vec![GrammarTrigger::pattern(
            "(?:^|<\\|start\\|>assistant)( to=(?!self<\\|message\\|>)(?!user<\\|message\\|>)[^<]*?<\\|message\\|>)",
        )];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Functionary v3.2 (parsers/functionary-v3-2.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_functionary_v3_2` (parsers/functionary-v3-2.cpp:4-96)
pub(crate) fn chat_params_init_functionary_v3_2(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.preserved_tokens = vec![">>>all".to_string()];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let include_grammar = has_tools && inputs.tool_choice != ChatToolChoice::None;

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;
        data.generation_prompt = format!(
            "<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n{}",
            msg.render_content("\n\n")?
        );
        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        // Functionary v3.2 format:
        // - Normal content: >>>all\n{content}
        // - Tool calls: >>>function_name\n{json_args}
        // Generation prompt ends with ">>>" so model outputs recipient immediately

        // Build content parser for >>>all\n{content}
        // When tools are present, content stops before the next ">>>" (tool call)
        // When no tools, content goes until end
        let all_lit = p.p.literal("all\n");
        let until = p.p.until(">>>");
        let c = p.content(until);
        let content_until_tool = p.p.sequence(&[all_lit, c]);
        let all_lit2 = p.p.literal("all\n");
        let rest = p.p.rest();
        let c2 = p.content(rest);
        let content_until_end = p.p.sequence(&[all_lit2, c2]);
        let generation_prompt =
            p.p.literal("<|start_header_id|>assistant<|end_header_id|>\n\n>>>");

        // If no tools or tool_choice is NONE, just parse content
        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            // When no tools, just match the prefix and capture everything after
            let end = p.p.end();
            let seq = p.p.sequence(&[generation_prompt, content_until_end, end]);
            return seq;
        }

        // Build tool call parsers for each available function
        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        foreach_function(&inputs.tools, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let schema = tool_parameters(function);

            // Tool format: >>>function_name\n{json_args}
            let nm = p.p.literal(&name);
            let tn = p.tool_name(nm);
            let nl = p.p.literal("\n");
            let oseq = p.p.sequence(&[tn, nl]);
            let t_open = p.tool_open(oseq);
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, &format!("tool-{name}-schema"), &schema, false);
            let ta = p.tool_args(sch);
            let seq = p.p.sequence(&[t_open, ta]);
            let tool_parser = p.tool(seq);

            let r = p.p.rule(&format!("tool-{name}"), tool_parser, false);
            tool_choice_alts.push(r);
        });

        let content_only = content_until_end;
        let tool_choice = p.p.choice(&tool_choice_alts);
        let one = p.p.one_or_more(tool_choice);
        let tools_only = p.p.trigger_rule("tools", one);
        let content_and_tools = p.p.sequence(&[content_until_tool, tools_only]);

        let ret;
        if inputs.tool_choice == ChatToolChoice::Required {
            if inputs.parallel_tool_calls {
                let alt = p.p.choice(&[content_and_tools, tools_only]);
                let end = p.p.end();
                ret = p.p.sequence(&[alt, end]);
            } else {
                let seq = p.p.sequence(&[content_until_tool, tool_choice]);
                let alt = p.p.choice(&[seq, tools_only]);
                let end = p.p.end();
                ret = p.p.sequence(&[alt, end]);
            }
        } else if inputs.parallel_tool_calls {
            let alt = p.p.choice(&[content_and_tools, content_only, tools_only]);
            let end = p.p.end();
            ret = p.p.sequence(&[alt, end]);
        } else {
            let content_and_tool = p.p.sequence(&[content_until_tool, tool_choice]);
            let alt = p.p.choice(&[content_and_tool, content_only, tool_choice]);
            let end = p.p.end();
            ret = p.p.sequence(&[alt, end]);
        }
        p.p.sequence(&[generation_prompt, ret])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = inputs.tool_choice == ChatToolChoice::Auto;

        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        // Grammar trigger for when the model starts outputting a tool call
        // (after the initial ">>>" in the generation prompt but recipient other than "all")
        data.grammar_triggers = vec![GrammarTrigger::pattern(">>>(?!all)")];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Kimi K2 Thinking (parsers/kimi-k2.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_kimi_k2` (parsers/kimi-k2.cpp:5-128)
pub(crate) fn chat_params_init_kimi_k2(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;
    data.preserved_tokens = vec![
        "<|tool_calls_section_begin|>".to_string(),
        "<|tool_calls_section_end|>".to_string(),
        "<|tool_call_begin|>".to_string(),
        "<|tool_call_argument_begin|>".to_string(),
        "<|tool_call_end|>".to_string(),
        "<think>".to_string(),
        "</think>".to_string(),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar = has_tools && inputs.tool_choice != ChatToolChoice::None;

    const SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
    const SECTION_END: &str = "<|tool_calls_section_end|>";
    const CALL_BEGIN: &str = "<|tool_call_begin|>";
    const ARGS_BEGIN: &str = "<|tool_call_argument_begin|>";
    const CALL_END: &str = "<|tool_call_end|>";

    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";
    const GEN_PROMPT: &str = "<|im_assistant|>assistant<|im_middle|>";

    data.thinking_start_tag = THINK_START.to_string();
    data.thinking_end_tags = vec![THINK_END.to_string()];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{GEN_PROMPT}{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("{THINK_END}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        // Kimi K2 Thinking format:
        // - Reasoning: <think>{reasoning}</think>
        // - Content: text after reasoning
        // - Tool calls section:
        //   <|tool_calls_section_begin|>
        //   <|tool_call_begin|>functions.<name>:<index><|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        //   ...
        //   <|tool_calls_section_end|>
        // The ID format is: functions.<function_name>:<counter> where counter is 0, 1, 2, ...

        let end = p.p.end();

        // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
        // For example, it can call tools at the end of reasoning without closing reasoning...
        let reasoning = if extract_reasoning {
            let lit = p.p.literal(THINK_START);
            let until = p.p.until_one_of(&[
                THINK_END,
                "<|tool_calls_section_begin|>",
                "<|tool_call_begin|>",
            ]);
            let r = p.reasoning(until);
            let te = p.p.literal(THINK_END);
            let close = p.p.optional(te);
            let seq = p.p.sequence(&[lit, r, close]);
            p.p.optional(seq)
        } else {
            p.p.eps()
        };
        let generation_prompt = p.p.literal(GEN_PROMPT);

        // Content only parser (no tools)
        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            let rest = p.p.rest();
            let c = p.content(rest);
            return p.p.sequence(&[generation_prompt, reasoning, c, end]);
        }

        // Build tool call parsers for each available function
        // The ID format is: functions.<name>:<index>
        // We need to match: functions.<name>:<digits>
        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        foreach_function(&inputs.tools, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let schema = tool_parameters(function);

            // Match: functions.<name>:<digits>
            // Capture the full call id (functions.<name>:<digits>) using tool_id tag
            let f_lit = p.p.literal("functions.");
            let nm = p.p.literal(&name);
            let tn = p.tool_name(nm);
            let colon = p.p.literal(":");
            let digits = p.p.chars("[0-9]", 1, -1);
            let id_seq = p.p.sequence(&[f_lit, tn, colon, digits]);
            let tool_id = p.tool_id(id_seq);
            let args_lit = p.p.literal(ARGS_BEGIN);
            let open_seq = p.p.sequence(&[tool_id, args_lit]);
            let t_open = p.tool_open(open_seq);
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, &format!("tool-{name}-schema"), &schema, false);
            let ta = p.tool_args(sch);
            let ce = p.p.literal(CALL_END);
            let close = p.p.optional(ce);
            let t_close = p.tool_close(close);
            let seq = p.p.sequence(&[t_open, ta, t_close]);
            let tool_parser = p.tool(seq);

            let r = p.p.rule(&format!("tool-{name}"), tool_parser, false);
            tool_choice_alts.push(r);
        });
        let tool_choice = p.p.choice(&tool_choice_alts);

        // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
        let min_calls = if inputs.tool_choice == ChatToolChoice::Required {
            1
        } else {
            0
        };
        let max_calls = if inputs.parallel_tool_calls { -1 } else { 1 };
        // Use trigger_rule so grammar generator knows where to start generating rules
        let cb_lit = p.p.literal(CALL_BEGIN);
        let cb_seq = p.p.sequence(&[cb_lit, tool_choice]);
        let cb_rep = p.p.repeat3(cb_seq, min_calls, max_calls);
        let se_lit = p.p.literal(SECTION_END);
        let sec_opt = p.p.optional(se_lit);
        let trig_body = p.p.sequence(&[cb_rep, sec_opt]);
        let trigger = p.p.trigger_rule("tool-call", trig_body);
        let sb_lit = p.p.literal(SECTION_BEGIN);
        let sb_opt = p.p.optional(sb_lit);
        let tc_seq = p.p.sequence(&[sb_opt, trigger]);
        let tool_calls = p.p.rule("tool-calls", tc_seq, false);

        let content_before_tools_until = p.p.until_one_of(&[SECTION_BEGIN, CALL_BEGIN]);
        let content_before_tools = p.content(content_before_tools_until);

        p.p.sequence(&[
            generation_prompt,
            reasoning,
            content_before_tools,
            tool_calls,
            end,
        ])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = inputs.tool_choice == ChatToolChoice::Auto;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word("<|tool_call_begin|>")];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Kimi K3 (parsers/kimi-k3.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_kimi_k3` (parsers/kimi-k3.cpp:8-167)
pub(crate) fn chat_params_init_kimi_k3(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;

    const SEP: &str = "<|sep|>";
    const MSG_START: &str = "<|open|>message role=\"assistant\"<|sep|>";
    const THINK_START: &str = "<|open|>think<|sep|>";
    const THINK_END: &str = "<|close|>think<|sep|>";
    const RESP_START: &str = "<|open|>response<|sep|>";
    const RESP_END: &str = "<|close|>response<|sep|>";
    const TOOLS_START: &str = "<|open|>tools<|sep|>";
    const TOOLS_END: &str = "<|close|>tools<|sep|>";
    const CALL_START: &str = "<|open|>call tool=\"";
    const CALL_END: &str = "<|close|>call<|sep|>";
    const ARG_START: &str = "<|open|>argument key=\"";
    const ARG_END: &str = "<|close|>argument<|sep|>";
    const MSG_END: &str = "<|close|>message<|sep|>";
    const EOM_TOKEN: &str = "<|end_of_msg|>";

    // only the markers are special tokens. tag names ("think", "response", ...) are
    // normal tokens and must not be preserved, or prose with those words is broken
    data.preserved_tokens = vec![
        "<|open|>".to_string(),
        "<|close|>".to_string(),
        "<|sep|>".to_string(),
        "<|end_of_msg|>".to_string(),
    ];

    data.thinking_start_tag = THINK_START.to_string();
    data.thinking_end_tags = vec![THINK_END.to_string()];

    // per-role message-start delimiters. user/assistant messages only have the role
    // attribute, so the full opener is used. system and tool messages have more
    // attributes, so those delimiters stop after the closing quote of the role
    data.message_delimiters = vec![
        (
            "assistant".to_string(),
            "<|open|>message role=\"assistant\"<|sep|>".to_string(),
        ),
        (
            "user".to_string(),
            "<|open|>message role=\"user\"<|sep|>".to_string(),
        ),
        (
            "tool".to_string(),
            "<|open|>message role=\"tool\"".to_string(),
        ),
        (
            "system".to_string(),
            "<|open|>message role=\"system\"".to_string(),
        ),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar = has_tools && inputs.tool_choice != ChatToolChoice::None;

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{MSG_START}{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt +=
                &format!("{THINK_END}{RESP_START}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let end = p.p.end();

        // auto start = p.optional(p.literal(MSG_START));  (no rule wrapper)
        let ms_lit = p.p.literal(MSG_START);
        let start = p.p.optional(ms_lit);

        // the think section is always consumed, even with reasoning extraction off:
        // the generation prompt ends with open_tag('think'), so it is always present.
        // reasoning stops at its own closer, or at the response opener if the model
        // skips the closer
        let think_until = p.p.until_one_of(&[THINK_END, RESP_START]);
        let think_body = if extract_reasoning {
            p.reasoning(think_until)
        } else {
            p.content(think_until)
        };

        let ts_lit = p.p.literal(THINK_START);
        let ts_opt = p.p.optional(ts_lit);
        let te_lit = p.p.literal(THINK_END);
        let te_opt = p.p.optional(te_lit);
        let inner = p.p.sequence(&[ts_opt, think_body, te_opt]);
        let reasoning = p.p.optional(inner);

        // content runs to the response closer, or to the next section if truncated
        let rs_lit = p.p.literal(RESP_START);
        let rs_opt = p.p.optional(rs_lit);
        let resp_until = p.p.until_one_of(&[RESP_END, TOOLS_START, MSG_END]);
        let resp_content = p.content(resp_until);
        let re_lit = p.p.literal(RESP_END);
        let re_opt = p.p.optional(re_lit);
        let response = p.p.sequence(&[rs_opt, resp_content, re_opt]);

        // the EOG token after the message closer reaches the parser as text,
        // so it must be consumed or the parse stays incomplete
        let me_lit = p.p.literal(MSG_END);
        let me_opt = p.p.optional(me_lit);
        let eom_lit = p.p.literal(EOM_TOKEN);
        let eom_opt = p.p.optional(eom_lit);
        let trailer = p.p.sequence(&[me_opt, eom_opt]);

        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            return p.p.sequence(&[start, reasoning, response, trailer, end]);
        }

        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        foreach_function(&inputs.tools, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let schema = tool_parameters(function);

            // arguments come one tag per key, with the JSON type in a type="..."
            // attribute. the type is taken from the tool schema instead, as it tells
            // us if the value is JSON or a literal string
            let mut args = p.p.eps();
            if let Some(props) = schema.at("properties") {
                if !props.empty() {
                    let mut arg_choice_alts: Vec<ParserId> = Vec::new();
                    for (key, prop_value) in props.items() {
                        let mut ty = "string".to_string();
                        if prop_value.is_object() {
                            if let Some(t) = prop_value.at("type") {
                                if let Ok(s) = t.get_str() {
                                    ty = s.to_string();
                                }
                            }
                        }

                        let until = p.p.until(ARG_END);
                        let value = if ty == "string" {
                            p.tool_arg_string_value(until)
                        } else {
                            p.tool_arg_value(until)
                        };

                        // skip the trailing type="..." attribute: anything up to <|sep|>
                        // p.tool_arg_open(p.literal(ARG_START)) + p.tool_arg_name(...) + ...
                        let as_lit = p.p.literal(ARG_START);
                        let tao = p.tool_arg_open(as_lit);
                        let km = p.p.literal(&key);
                        let tan = p.tool_arg_name(km);
                        let q = p.p.literal("\"");
                        let until_sep = p.p.until(SEP);
                        let sep1 = p.p.literal(SEP);
                        let ae_lit = p.p.literal(ARG_END);
                        let tac = p.tool_arg_close(ae_lit);
                        let arg_seq = p.p.sequence(&[tao, tan, q, until_sep, sep1, value, tac]);
                        let arg = p.tool_arg(arg_seq);
                        let r = p.p.rule(&format!("kimi-k3-arg-{name}-{key}"), arg, false);
                        arg_choice_alts.push(r);
                    }
                    let arg_choices = p.p.choice(&arg_choice_alts);
                    args = p.p.zero_or_more(arg_choices);
                }
            }

            // skip the trailing index="N" attribute the same way
            let cs_lit = p.p.literal(CALL_START);
            let nm = p.p.literal(&name);
            let tn = p.tool_name(nm);
            let q = p.p.literal("\"");
            let until_sep = p.p.until(SEP);
            let sep1 = p.p.literal(SEP);
            let open_seq = p.p.sequence(&[cs_lit, tn, q, until_sep, sep1]);
            let t_open = p.tool_open(open_seq);
            let ta = p.tool_args(args);
            let ce_lit = p.p.literal(CALL_END);
            let t_close = p.tool_close(ce_lit);
            let call = pegc1!(p, tool, p.p.sequence(&[t_open, ta, t_close]));

            let r = p.p.rule(&format!("kimi-k3-tool-{name}"), call, false);
            tool_choice_alts.push(r);
        });
        let tool_choices = p.p.choice(&tool_choice_alts);

        // all calls go inside one tools section, then the message is closed. the
        // message closer is part of the trigger rule, or else the lazy grammar
        // rejects it once tool calls have started
        let ts_lit = p.p.literal(TOOLS_START);
        let one = p.p.one_or_more(tool_choices);
        let te_lit = p.p.literal(TOOLS_END);
        let me2_lit = p.p.literal(MSG_END);
        let me2_opt = p.p.optional(me2_lit);
        let eom2_lit = p.p.literal(EOM_TOKEN);
        let eom2_opt = p.p.optional(eom2_lit);
        let body = p.p.sequence(&[ts_lit, one, te_lit, me2_opt, eom2_opt]);
        let tools_section = p.p.trigger_rule("kimi-k3-tool-call", body);

        let tools = if inputs.tool_choice == ChatToolChoice::Required {
            tools_section
        } else {
            p.p.optional(tools_section)
        };

        p.p.sequence(&[start, reasoning, response, tools, trailer, end])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = inputs.tool_choice != ChatToolChoice::Required;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(TOOLS_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Ling 3.0 / Bailing V3 (parsers/ling3.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_ling3` (parsers/ling3.cpp:11-194)
pub(crate) fn chat_params_init_ling3(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;

    const ROLE: &str = "<role>ASSISTANT</role>";
    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";
    const CALL_START: &str = "<tool_call>";
    const CALL_END: &str = "</tool_call>";
    const ARG_KEY: &str = "<arg_key>";
    const ARG_KEY_END: &str = "</arg_key>";
    const ARG_VAL: &str = "<arg_value>";
    const ROLE_END: &str = "<|role_end|>";
    const ARG_VAL_END: &str = "</arg_value>";

    data.preserved_tokens = vec![
        THINK_START.to_string(),
        THINK_END.to_string(),
        CALL_START.to_string(),
        CALL_END.to_string(),
        ARG_KEY.to_string(),
        ARG_KEY_END.to_string(),
        ARG_VAL.to_string(),
        ARG_VAL_END.to_string(),
        ROLE_END.to_string(),
    ];

    data.thinking_start_tag = THINK_START.to_string();
    // Support both </think> and <tool_call> as reasoning end sequences: a call
    // can be emitted before the think block is closed.
    data.thinking_end_tags = vec![THINK_END.to_string(), CALL_START.to_string()];

    data.message_delimiters = vec![
        (
            "assistant".to_string(),
            "<role>ASSISTANT</role>".to_string(),
        ),
        ("user".to_string(), "<role>HUMAN</role>".to_string()),
        ("tool".to_string(), "<role>OBSERVATION</role>".to_string()),
        ("system".to_string(), "<role>SYSTEM</role>".to_string()),
    ];

    // the model may spell the end-of-turn control token out as text tokens,
    // which does not stop generation; a literal stop string catches it either
    // way (as the Laguna patch does for its </assistant> token)
    data.additional_stops = vec![ROLE_END.to_string()];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{ROLE}\n{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("{THINK_END}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    // The generation prompt pre-opens the think block when thinking is on, so
    // the opening tag is optional here and reasoning runs until </think> or a
    // tool call start; with thinking off the template pre-closes the block and
    // everything the model emits is content.
    let think_open;
    if inputs.has_continuation() {
        think_open = inputs.continue_final_message != ChatContinuation::Content;
    } else {
        let last_open = data.generation_prompt.rfind(THINK_START);
        let last_close = data.generation_prompt.rfind(THINK_END);
        think_open = match (last_open, last_close) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let end = p.p.end();

        // the effective papeg1!(p, optional, p.p.space())rompt + model output, so the
        // assistant opener is optionally consumed here
        let role_lit = p.p.literal(ROLE);
        let sp_opt = peg1!(p, optional, p.p.space());
        let role_seq = p.p.sequence(&[role_lit, sp_opt]);
        let opener = p.p.optional(role_seq);

        // the generation prompt pre-opens the think block, so the opening tag
        // is optional; a missing close tag does not swallow a tool call
        let body_end = if think_open {
            p.p.until_one_of(&[THINK_END, CALL_START])
        } else {
            p.p.until_one_of(&[THINK_END])
        };
        let think_body = if extract_reasoning {
            p.reasoning(body_end)
        } else {
            p.content(body_end)
        };

        let ts_opt = peg1!(p, optional, p.p.literal(THINK_START));
        let te_opt = peg1!(p, optional, p.p.literal(THINK_END));
        let inner = p.p.sequence(&[ts_opt, think_body, te_opt]);
        let reasoning = p.p.optional(inner);

        // content between the think block and the first tool call, plus any
        // trailing text after the last tool call, are plain content
        let content_until = p.p.until_one_of(&[CALL_START]);
        let content_c = p.content(content_until);
        let content = p.p.optional(content_c);

        // a trailing end-of-turn token ipeg1!(p, optional, p.p.literal(ROLE_END))content
        let tail_until = p.p.until(ROLE_END);
        let tail_c = p.content(tail_until);
        let tail_opt = p.p.optional(tail_c);
        let tail_re = peg1!(p, optional, p.p.literal(ROLE_END));
        let tail = p.p.sequence(&[tail_opt, tail_re]);

        // the think block must close before the JSON, so the turn cannot end
        // inside the reasoning (parsers/ling3.cpp:105-110, upstream a7b94df2c)
        if has_response_format {
            let ts = p.p.literal(THINK_START);
            let te = p.p.literal(THINK_END);
            let closed_reasoning = p.p.sequence(&[ts, think_body, te]);
            let json_p = p.p.json();
            let sch = p.p.schema(json_p, "response-format", &inputs.json_schema, false);
            let response_format = p.content(sch);
            let spaced = p.p.spaced(closed_reasoning, response_format);
            return p.p.sequence(&[opener, spaced, end]);
        }

        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            return p.p.sequence(&[opener, reasoning, tail, end]);
        }

        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        let arg_close_lit = p.p.literal(ARG_VAL_END);
        let arg_close = p.tool_arg_close(arg_close_lit);
        let arg_string_until = p.p.until(ARG_VAL_END);
        let arg_string_v = p.tool_arg_string_value(arg_string_until);
        let arg_string_seq = p.p.sequence(&[arg_string_v, arg_close]);
        let arg_string = p.p.rule("ling3-arg-string", arg_string_seq, false);

        foreach_function(&inputs.tools, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();

            let mut required_args: Vec<ParserId> = Vec::new();
            let mut optional_args: Vec<ParserId> = Vec::new();

            // each argument may be preceded by whitespace: the model emits
            // newlines between arguments, the template history does not
            foreach_parameter(function, |param, doc| {
                let rule_name = format!("ling3-arg-{name}-{}", param.name);

                let types = doc.value_types(param.schema);

                // string arguments are raw text up to the closing tag, other
                // types parse as JSON per their schema; each alternative
                // consumes the closing tag itself so a JSON prefix can not
                // commit the choice before the tag matches
                let arg_value;
                if !types.has(ValueType::String) {
                    let json_p = p.p.json();
                    let sch = p.p.schema_node(
                        json_p,
                        &format!("{rule_name}-schema"),
                        Rc::clone(doc),
                        param.schema,
                        false,
                    );
                    let jv = p.tool_arg_json_value(sch);
                    arg_value = p.p.sequence(&[jv, arg_close]);
                } else if types.is_only(ValueType::String) {
                    arg_value = arg_string;
                } else {
                    // the parser tries the JSON alternative first to type the value
                    let json_p = p.p.json();
                    let sch = p.p.schema_node(
                        json_p,
                        &format!("{rule_name}-schema"),
                        Rc::clone(doc),
                        param.schema,
                        false,
                    );
                    let jv = p.tool_arg_json_value(sch);
                    let jseq = p.p.sequence(&[jv, arg_close]);
                    let jatomic = p.p.atomic(jseq);
                    let alt = p.p.choice(&[jatomic, arg_string]);
                    arg_value = p.p.gbnf(alt, "ling3-arg-string");
                }

                // p.optional(p.space()) + tool_arg(tool_arg_open(...) + p.optional(p.space()) + ARG_VAL + arg_value)
                let k_lit = p.p.literal(ARG_KEY);
                let nm = p.p.literal(&param.name);
                let tan = p.tool_arg_name(nm);
                let ke_lit = p.p.literal(ARG_KEY_END);
                let kseq = p.p.sequence(&[k_lit, tan, ke_lit]);
                let tao = p.tool_arg_open(kseq);
                let sp2_id = p.p.space();
                let sp2 = p.p.optional(sp2_id);
                let v_lit = p.p.literal(ARG_VAL);
                let arg_body = p.p.sequence(&[tao, sp2, v_lit, arg_value]);
                let arg = p.tool_arg(arg_body);
                let pre_id = p.p.space();
                let pre = p.p.optional(pre_id);
                let arg_rule = p.p.sequence(&[pre, arg]);
                let arg_rule = p.p.rule(&rule_name, arg_rule, false);

                if param.required {
                    required_args.push(arg_rule);
                } else {
                    optional_args.push(arg_rule);
                }
            });

            // required arguments in any order (as Qwen3-Coder does), then
            // optional ones in any order and number
            let mut args = p.permute(&format!("ling3-{name}-args"), &required_args);
            if !optional_args.is_empty() {
                let any_opt = p.p.choice(&optional_args);
                let zom = p.p.zero_or_more(any_opt);
                args = p.p.sequence(&[args, zom]);
            }

            let cs_lit = p.p.literal(CALL_START);
            let nm = p.p.literal(&name);
            let tn = p.tool_name(nm);
            let sp_id = p.p.space();
            let sp_pre = p.p.optional(sp_id);
            let oseq = p.p.sequence(&[cs_lit, tn, sp_pre]);
            let t_open = p.tool_open(oseq);
            let ta = p.tool_args(args);
            let ce_pre = {
                let a = p.p.space();
                p.p.optional(a)
            };
            let ce_lit = p.p.literal(CALL_END);
            let ce_seq = p.p.sequence(&[ce_pre, ce_lit]);
            let t_close = p.tool_close(ce_seq);
            let call = pegc1!(p, tool, p.p.sequence(&[t_open, ta, t_close]));

            let r = p.p.rule(&format!("ling3-tool-{name}"), call, false);
            tool_choice_alts.push(r);
        });
        let tool_choices = p.p.choice(&tool_choice_alts);

        let calls = if inputs.parallel_tool_calls {
            // tool_choices + zero_or_more(p.space() + tool_choices)
            let sp = p.p.space();
            let seq = p.p.sequence(&[sp, tool_choices]);
            let more = p.p.zero_or_more(seq);
            p.p.sequence(&[tool_choices, more])
        } else {
            tool_choices
        };

        let trailing_until = p.p.until(ROLE_END);
        let trailing_c = p.content(trailing_until);
        let trailing_opt = p.p.optional(trailing_c);
        let re_lit = p.p.literal(ROLE_END);
        let trailing_re = p.p.optional(re_lit);
        let trailing = p.p.sequence(&[trailing_opt, trailing_re]);
        let sp = p.p.space();
        let body = p.p.sequence(&[calls, sp, trailing]);
        let tools_section = p.p.trigger_rule("ling3-tool-call", body);

        let tools = if inputs.tool_choice == ChatToolChoice::Required {
            tools_section
        } else {
            p.p.optional(tools_section)
        };

        p.p.sequence(&[opener, reasoning, content, tools, tail, end])
    })?;

    data.parser = parser.save();

    if include_grammar {
        // `!has_response_format && tool_choice != REQUIRED`
        // (parsers/ling3.cpp:191, upstream a7b94df2c)
        data.grammar_lazy =
            !has_response_format && inputs.tool_choice != ChatToolChoice::Required;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(CALL_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Cohere2 MoE / North Code (parsers/cohere2moe.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_cohere2moe` (parsers/cohere2moe.cpp:19-141)
pub(crate) fn chat_params_init_cohere2moe(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    const TURN_START: &str = "<|START_OF_TURN_TOKEN|>";
    const TURN_END: &str = "<|END_OF_TURN_TOKEN|>";
    const CHATBOT: &str = "<|CHATBOT_TOKEN|>";
    const USER: &str = "<|USER_TOKEN|>";
    const SYSTEM: &str = "<|SYSTEM_TOKEN|>";
    const THINK_START: &str = "<|START_THINKING|>";
    const THINK_END: &str = "<|END_THINKING|>";
    const TEXT_START: &str = "<|START_TEXT|>";
    const TEXT_END: &str = "<|END_TEXT|>";
    const ACTION_START: &str = "<|START_ACTION|>";
    const ACTION_END: &str = "<|END_ACTION|>";
    const RESULT_START: &str = "<|START_TOOL_RESULT|>";
    const RESULT_END: &str = "<|END_TOOL_RESULT|>";

    // Stable prefix of the generation prompt that precedes the (forced) <|START_THINKING|> marker.
    let gen_prefix = format!("{TURN_START}{CHATBOT}");

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;
    data.thinking_start_tag = THINK_START.to_string();
    data.thinking_end_tags = vec![THINK_END.to_string()];
    data.preserved_tokens = vec![
        TURN_START.to_string(),
        TURN_END.to_string(),
        CHATBOT.to_string(),
        USER.to_string(),
        SYSTEM.to_string(),
        THINK_START.to_string(),
        THINK_END.to_string(),
        TEXT_START.to_string(),
        TEXT_END.to_string(),
        ACTION_START.to_string(),
        ACTION_END.to_string(),
        RESULT_START.to_string(),
        RESULT_END.to_string(),
    ];

    // Declare per-role message delimiters. Tool results are rendered with the
    // system token followed by <|START_TOOL_RESULT|>, so the "tool" delimiter must be listed before
    // the plain "system" one (it is a strict superset, and the role split tries delimiters in order).
    data.message_delimiters = vec![
        ("assistant".to_string(), gen_prefix.clone()),
        ("user".to_string(), format!("{TURN_START}{USER}")),
        (
            "tool".to_string(),
            format!("{TURN_START}{SYSTEM}{RESULT_START}"),
        ),
        ("system".to_string(), format!("{TURN_START}{SYSTEM}")),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{gen_prefix}{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt +=
                &format!("{THINK_END}{TEXT_START}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.p.literal(&gen_prefix);
        let end = p.p.end();

        // The thinking block is always present (the generation prompt forces <|START_THINKING|>).
        // When extracting reasoning, capture its body; otherwise keep the whole block (markers
        // included) inline as content, matching reasoning_format=NONE conventions.
        let reasoning;
        if extract_reasoning {
            let ts = p.p.literal(THINK_START);
            let until = p.p.until_one_of(&[THINK_END, TEXT_START, ACTION_START]);
            let r = p.reasoning(until);
            let te = {
                let a = p.p.literal(THINK_END);
                p.p.optional(a)
            };
            let seq = p.p.sequence(&[ts, r, te]);
            reasoning = p.p.optional(seq);
        } else {
            let ts = p.p.literal(THINK_START);
            let until = p.p.until_one_of(&[THINK_END, TEXT_START, ACTION_START]);
            let te = {
                let a = p.p.literal(THINK_END);
                p.p.optional(a)
            };
            let seq = p.p.sequence(&[ts, until, te]);
            let c = p.content(seq);
            reasoning = p.p.optional(c);
        }

        let text_content = if has_response_format {
            let ts = p.p.literal(TEXT_START);
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
            let c = p.content(sch);
            let te = {
                let a = p.p.literal(TEXT_END);
                p.p.optional(a)
            };
            p.p.sequence(&[ts, c, te])
        } else {
            let ts = p.p.literal(TEXT_START);
            let until = p.p.until(TEXT_END);
            let c = p.content(until);
            let te = {
                let a = p.p.literal(TEXT_END);
                p.p.optional(a)
            };
            p.p.sequence(&[ts, c, te])
        };

        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            let te = {
                let a = p.p.literal(TURN_END);
                p.p.optional(a)
            };
            return p
                .p
                .sequence(&[generation_prompt, reasoning, text_content, te, end]);
        }

        let require_tools = inputs.tool_choice == ChatToolChoice::Required;

        // <|START_ACTION|>[ {"tool_call_id": "0", "tool_name": "f", "parameters": {...}}, ... ]<|END_ACTION|>
        let tool_calls = p.standard_json_tools(
            ACTION_START,
            ACTION_END,
            &inputs.tools,
            inputs.parallel_tool_calls,
            /* force_tool_calls = */ true,
            /* name_key = */ "tool_name",
            /* args_key = */ "parameters",
            /* array_wrapped = */ true,
            /* function_is_key = */ false,
            /* call_id_key = */ "",
            /* gen_call_id_key = */ "tool_call_id",
            /* parameters_order = */
            &[
                "tool_call_id".to_string(),
                "tool_name".to_string(),
                "parameters".to_string(),
            ],
            /* accept_openai_wrapper = */ false,
        );

        // Content and tool calls are mutually exclusive in this format.
        let body = if require_tools {
            tool_calls
        } else {
            p.p.choice(&[tool_calls, text_content])
        };

        let te = {
            let a = p.p.literal(TURN_END);
            p.p.optional(a)
        };
        p.p.sequence(&[generation_prompt, reasoning, body, te, end])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == ChatToolChoice::Auto;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(ACTION_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// LFM2 / LFM2.5 (parsers/lfm2.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_lfm2` (parsers/lfm2.cpp:14-110); `tool_list_tokens`
/// preserves the LFM2 system tool-list markers.
pub(crate) fn chat_params_init_lfm2(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
    tool_list_tokens: bool,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    const TOOL_CALL_START: &str = "<|tool_call_start|>";
    const TOOL_CALL_END: &str = "<|tool_call_end|>";
    const TOOL_LIST_START: &str = "<|tool_list_start|>";
    const TOOL_LIST_END: &str = "<|tool_list_end|>";
    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";
    const GEN_PROMPT: &str = "<|im_start|>assistant\n";

    // Copy reasoning to the "thinking" field the template expects
    let mut adjusted_messages = Json::Array(Vec::new());
    if let Json::Array(msgs) = &inputs.messages {
        for msg in msgs {
            let mut msg = msg.clone();
            if msg
                .at("reasoning_content")
                .map(|v| v.is_string())
                .unwrap_or(false)
            {
                msg.set("thinking", msg.at("reasoning_content").cloned().unwrap());
            }
            if let Json::Array(a) = &mut adjusted_messages {
                a.push(msg);
            }
        }
    }

    data.prompt = template_direct_apply_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.generation_prompt =
        template_generation_prompt_impl(tmpl, inputs, Some(&adjusted_messages), None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;
    data.preserved_tokens = vec![
        TOOL_CALL_START.to_string(),
        TOOL_CALL_END.to_string(),
        THINK_START.to_string(),
        THINK_END.to_string(),
    ];
    if tool_list_tokens {
        data.preserved_tokens.push(TOOL_LIST_START.to_string());
        data.preserved_tokens.push(TOOL_LIST_END.to_string());
    }

    data.thinking_start_tag = THINK_START.to_string();
    data.thinking_end_tags = vec![THINK_END.to_string()];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    // Gate by reasoning format and whether the template supports <think>
    let extract_reasoning =
        inputs.reasoning_format != ReasoningFormat::None && tmpl.source().contains(THINK_START);
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{GEN_PROMPT}{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("{THINK_END}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.p.literal(GEN_PROMPT);
        let end = p.p.end();

        let mut reasoning = p.p.eps();
        if extract_reasoning {
            let ts = p.p.literal(THINK_START);
            let until = p.p.until(THINK_END);
            let r = p.reasoning(until);
            let te = p.p.literal(THINK_END);
            let seq = p.p.sequence(&[ts, r, te]);
            reasoning = p.p.optional(seq);
        }

        if !has_tools || inputs.tool_choice == ChatToolChoice::None {
            if has_response_format {
                let json_p = p.p.json();
                let sch =
                    p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
                let response_format = p.content(sch);
                return p
                    .p
                    .sequence(&[generation_prompt, reasoning, response_format, end]);
            }
            let rest = p.p.rest();
            let c = p.content(rest);
            return p.p.sequence(&[generation_prompt, reasoning, c, end]);
        }
        let py = p.python_style_tool_calls(
            &inputs.tools,
            inputs.parallel_tool_calls,
            /* allow_json_literals = */ true,
        );
        let tcs = p.p.literal(TOOL_CALL_START);
        let tce = p.p.literal(TOOL_CALL_END);
        let inner = p.p.sequence(&[tcs, py, tce]);
        let trigger = p.p.trigger_rule("tool-call", inner);
        let tool_calls = p.p.rule("tool-calls", trigger, false);

        let until = p.p.until(TOOL_CALL_START);
        let content = p.content(until);

        p.p.sequence(&[generation_prompt, reasoning, content, tool_calls, end])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(TOOL_CALL_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// GigaChatV3 (parsers/gigachat-v3.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_gigachat_v3` (parsers/gigachat-v3.cpp:3-76)
pub(crate) fn chat_params_init_gigachat_v3(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = false;
    data.preserved_tokens = vec![
        "<|message_sep|>\n\n".to_string(),
        "<|role_sep|>\n".to_string(),
    ];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;
        data.generation_prompt = format!("assistant<|role_sep|>\n{}", msg.render_content("\n\n")?);
        data.prompt += &data.generation_prompt;
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    // include_grammar starts true and is cleared in the content-only branch of
    // the parser builder (parsers/gigachat-v3.cpp:55)
    let include_grammar = has_tools && inputs.tool_choice != ChatToolChoice::None;
    let tool_call_start_prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let ret;
        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            // Build a choice of all available tools
            let mut tool_choice_alts: Vec<ParserId> = Vec::new();
            for tool in inputs.tools.iter() {
                let Some(function) = tool.at("function") else {
                    continue;
                };
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                let schema = tool_parameters(function);

                // tool_name = p.json_member("name", "\"" + p.tool_name(p.literal(name)) + "\"")
                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let q1 = p.p.literal("\"");
                let q2 = p.p.literal("\"");
                let quoted = p.p.sequence(&[q1, tn, q2]);
                let tool_name = p.p.json_member("name", quoted);

                // tool_args = p.json_member("arguments", p.tool_args(p.schema(...)))
                let json_p = p.p.json();
                let sch =
                    p.p.schema(json_p, &format!("tool-{name}-schema"), &schema, false);
                let ta = p.tool_args(sch);
                let tool_args = p.p.json_member("arguments", ta);

                // tool_open = p.tool_open(p.literal("{") << tool_name)
                let brace = p.p.literal("{");
                let open_inner = p.p.spaced(brace, tool_name);
                let tool_open = p.tool_open(open_inner);

                // p.rule("tool-" + name, tool_open << "," << tool_args << "}")
                let comma = p.p.literal(",");
                let s1 = p.p.spaced(tool_open, comma);
                let s2 = p.p.spaced(s1, tool_args);
                let rbrace = p.p.literal("}");
                let s3 = p.p.spaced(s2, rbrace);
                let r = p.p.rule(&format!("tool-{name}"), s3, false);
                tool_choice_alts.push(r);
            }
            let tool_choice = p.p.choice(&tool_choice_alts);

            // Define the tool call structure
            let min_calls = if inputs.tool_choice == ChatToolChoice::Required {
                1
            } else {
                0
            };
            let max_calls = 1; // parallel toolcalls are not supported
            let prefix_lit = p.p.literal(tool_call_start_prefix);
            let tc_body = p.p.sequence(&[prefix_lit, tool_choice]);
            let tool_call = p.p.rule("tool-call", tc_body, false);
            let rep = p.p.repeat3(tool_call, min_calls, max_calls);
            let tool_calls = p.p.trigger_rule("tool-call-root", rep);

            // ret = p.content(p.until("<|message_sep|>\n\n")) << tool_calls
            let until = p.p.until("<|message_sep|>\n\n");
            let c = p.content(until);
            ret = p.p.spaced(c, tool_calls);
        } else {
            // Content only parser (include_grammar = false)
            let rest = p.p.rest();
            ret = p.content(rest);
        }

        let prefix = p.p.literal("assistant<|role_sep|>\n");
        p.p.sequence(&[prefix, ret])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = has_tools && inputs.tool_choice == ChatToolChoice::Auto;

        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(tool_call_start_prefix)];
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// MiniMax-M3 (parsers/minimax-m3.cpp)
// ---------------------------------------------------------------------------

/// `value_of` helper (parsers/minimax-m3.cpp:109-151): the recursive
/// XML-expansion of a schema node.
fn minimax_m3_build_value_of(
    p: &mut ChatPegBuilder,
    doc: &Rc<SchemaDocument>,
    node: crate::json_schema::NodeId,
    rule_name: &str,
    close: &str,
) -> ParserId {
    let close_tag_lit = p.p.literal(close);
    let close_tag = p.tool_arg_close(close_tag_lit);

    // A string accepts anything, so a union with a string alternative is a string
    if doc.may_be_string(node) {
        let until = p.p.until(close);
        let sv = p.tool_arg_string_value(until);
        let seq = p.p.sequence(&[sv, close_tag]);
        return p.p.ac(seq, &[close.to_string()]);
    }

    match &doc.node(node).kind {
        SchemaKind::AnyOf { .. } => {
            let mut choices: Vec<ParserId> = Vec::new();
            let children = doc.node(node).children.clone();
            for (index, alternative) in children.iter().enumerate() {
                let alt_name = format!("{rule_name}-{index}");

                // There is a risk that this breaks streaming deltas, but that's a risk we
                // assume to provide tool arg streaming.
                choices.push(minimax_m3_build_value_of(
                    p,
                    doc,
                    *alternative,
                    &alt_name,
                    close,
                ));
            }
            p.p.choice(&choices)
        }
        SchemaKind::Object { properties, .. } if !properties.is_empty() => {
            let members = minimax_m3_members_of(p, doc, properties, rule_name);
            let tagged = p.p.tag(mm3_tag::TOOL_ARG_OBJECT, members);
            let sp = p.p.space();
            p.p.sequence(&[tagged, sp, close_tag])
        }
        SchemaKind::Array { items, .. } => {
            // item_close = NS + "</item>"
            let item_close = format!("]<]minimax[>[</item>");
            let item_open = p.p.literal("]<]minimax[>[<item>");
            let value = minimax_m3_build_value_of(
                p,
                doc,
                *items,
                &format!("{rule_name}-item"),
                &item_close,
            );
            let item_body = p.p.sequence(&[item_open, value]);
            let tagged_item = p.p.tag(mm3_tag::TOOL_ARG_ITEM, item_body);
            let item = p.p.rule(&format!("{rule_name}-item"), tagged_item, false);
            let sp = p.p.space();
            let seq = p.p.sequence(&[sp, item]);
            let rep = p.p.repeat3(seq, 0, -1);
            let tagged_arr = p.p.tag(mm3_tag::TOOL_ARG_ARRAY, rep);
            let sp2 = p.p.space();
            p.p.sequence(&[tagged_arr, sp2, close_tag])
        }
        _ => {
            let json_p = p.p.json();
            let sch = p.p.schema_node(
                json_p,
                &format!("{rule_name}-schema"),
                Rc::clone(doc),
                node,
                false,
            );
            let jv = p.tool_arg_json_value(sch);
            p.p.sequence(&[jv, close_tag])
        }
    }
}

/// `members_of` (parsers/minimax-m3.cpp:154-179)
fn minimax_m3_members_of(
    p: &mut ChatPegBuilder,
    doc: &Rc<SchemaDocument>,
    properties: &[crate::json_schema::SchemaProperty],
    rule_prefix: &str,
) -> ParserId {
    // `element_of` (parsers/minimax-m3.cpp:98-107)
    let element_of = |p: &mut ChatPegBuilder,
                      tag: &str,
                      schema: crate::json_schema::NodeId,
                      rule_name: &str|
     -> ParserId {
        // close = NS + "</" + tag + ">"
        let close = format!("]<]minimax[>[</{tag}>");
        let open1 = p.p.literal("]<]minimax[>[<");
        let tag_lit = p.p.literal(tag);
        let tan = p.tool_arg_name(tag_lit);
        let gt = p.p.literal(">");
        let oseq = p.p.sequence(&[open1, tan, gt]);
        let tao = p.tool_arg_open(oseq);
        let value = minimax_m3_build_value_of(p, doc, schema, rule_name, &close);
        let arg = p.p.sequence(&[tao, value]);
        let ta = p.tool_arg(arg);
        p.p.rule(rule_name, ta, false)
    };

    // Required properties in schema order, then any number of optional ones in any order.
    let mut required_elements: Vec<ParserId> = Vec::new();
    let mut optional_elements: Vec<ParserId> = Vec::new();
    for prop in properties {
        let element = element_of(
            p,
            &prop.name,
            prop.schema,
            &format!("{rule_prefix}-{}", prop.name),
        );
        if prop.required {
            required_elements.push(element);
        } else {
            optional_elements.push(element);
        }
    }

    let mut members = p.p.eps();
    for (i, &element) in required_elements.iter().enumerate() {
        if i > 0 {
            let sp = p.p.space();
            members = p.p.sequence(&[members, sp]);
        }
        members = p.p.sequence(&[members, element]);
    }

    if !optional_elements.is_empty() {
        let any_optional = p.p.choice(&optional_elements);
        let sp = p.p.space();
        let seq = p.p.sequence(&[sp, any_optional]);
        let rep = p.p.repeat3(seq, 0, -1);
        members = p.p.sequence(&[members, rep]);
    }

    members
}

/// `common_chat_params_init_minimax_m3` (parsers/minimax-m3.cpp:3-229)
pub(crate) fn chat_params_init_minimax_m3(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegMinimaxM3;
    data.supports_thinking = true;
    data.thinking_start_tag = "<mm:think>".to_string();
    data.thinking_end_tags = vec!["</mm:think>".to_string()];

    // M3 prefixes every tool tag with the namespace token "]<]minimax[>[";
    // params use the parameter name as the tag (<file_path>...</file_path>).
    const NS: &str = "]<]minimax[>[";
    const THINK_START: &str = "<mm:think>";
    const THINK_END: &str = "</mm:think>";
    const FC_START: &str = "]<]minimax[>[<tool_call>";
    const FC_END: &str = "]<]minimax[>[</tool_call>";
    const INVOKE_END: &str = "]<]minimax[>[</invoke>";

    data.preserved_tokens = vec![
        NS.to_string(),
        "<tool_call>".to_string(),
        "</tool_call>".to_string(),
        THINK_START.to_string(),
        THINK_END.to_string(),
    ];

    data.message_delimiters = vec![
        ("assistant".to_string(), "]~b]ai".to_string()),
        ("user".to_string(), "]~b]user".to_string()),
        ("tool".to_string(), "]~b]tool".to_string()),
        ("system".to_string(), "]~b]developer".to_string()),
        ("system".to_string(), "]~b]system".to_string()),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    let gen_prompt = data.generation_prompt.clone();

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = format!("{gen_prompt}{THINK_START}{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("{THINK_END}{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let enable_thinking = inputs.enable_thinking;
    let parallel_tool_calls = inputs.parallel_tool_calls;
    let tool_choice_required = inputs.tool_choice == ChatToolChoice::Required;
    let tool_choice_not_none = inputs.tool_choice != ChatToolChoice::None;
    let tools_json = inputs.tools.clone();
    let json_schema = inputs.json_schema.clone();

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.prefix(&gen_prompt, THINK_START);
        let end = p.p.end();

        let mut reasoning = p.p.eps();
        if extract_reasoning {
            let block = if enable_thinking {
                let ts = p.p.literal(THINK_START);
                let sp = p.p.space();
                let until = p.p.until(THINK_END);
                let r = p.reasoning(until);
                let te = p.p.literal(THINK_END);
                let inner = p.p.sequence(&[r, te]);
                let ac = p.p.ac(inner, &[THINK_END.to_string()]);
                p.p.sequence(&[ts, sp, ac])
            } else {
                let ts = p.p.literal(THINK_START);
                let until = p.p.until(THINK_END);
                let te = p.p.literal(THINK_END);
                let inner = p.p.sequence(&[until, te]);
                let ac = p.p.ac(inner, &[THINK_END.to_string()]);
                p.p.sequence(&[ts, ac])
            };

            // A turn without reasoning is prefixed with a bare </mm:think>, written either by the
            // generation prompt (thinking_mode = "disabled") or by the model itself.
            let bare = p.p.literal(THINK_END);
            let alt = p.p.choice(&[block, bare]);
            reasoning = p.p.optional(alt);
        }

        if has_response_format {
            let fence1 = p.p.literal("```json");
            let sp = p.p.space();
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &json_schema, false);
            let c = p.content(sch);
            let sp2 = p.p.space();
            let fence2 = p.p.literal("```");
            let body = p.p.sequence(&[fence1, sp, c, sp2, fence2]);
            let response_format = p.p.rule("response-format", body, false);
            return p
                .p
                .sequence(&[generation_prompt, reasoning, response_format, end]);
        }

        if !has_tools || !tool_choice_not_none {
            let rest = p.p.rest();
            let c = p.content(rest);
            return p.p.sequence(&[generation_prompt, reasoning, c, end]);
        }

        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        foreach_function(&tools_json, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            let params = tool_parameters(function);
            // the C++ common_chat_schema_from_json throws on bad schemas; keep
            // the builder total with an empty document (see PegBuilder::schema)
            let doc = Rc::new(
                schema_from_json(&params).unwrap_or_else(|_| SchemaDocument {
                    nodes: Vec::new(),
                    root: 0,
                    refs: BTreeMap::new(),
                }),
            );

            let mut invoke_body = p.p.eps();
            if let SchemaKind::Object { properties, .. } = &doc.node(doc.root).kind {
                invoke_body =
                    minimax_m3_members_of(p, &doc, properties, &format!("tool-{name}-arg"));
            }

            let inv_open1 = p.p.literal(&format!("{NS}<invoke name=\""));
            let nm = p.p.literal(&name);
            let tn = p.tool_name(nm);
            let inv_open2 = p.p.literal("\">");
            let oseq = p.p.sequence(&[inv_open1, tn, inv_open2]);
            let t_open = p.tool_open(oseq);
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let icl = p.p.literal(INVOKE_END);
            let t_close = p.tool_close(icl);
            let body = p.p.sequence(&[t_open, sp1, invoke_body, sp2, t_close]);
            let func_parser = p.tool(body);

            let r = p.p.rule(&format!("tool-{name}"), func_parser, false);
            tool_choice_alts.push(r);
        });
        let tool_choice = p.p.choice(&tool_choice_alts);

        let require_tools = tool_choice_required;

        let tool_calls;
        if parallel_tool_calls {
            let fcs = p.p.literal(FC_START);
            let sp = p.p.space();
            let more_seq = p.p.sequence(&[sp, tool_choice]);
            let more = p.p.zero_or_more(more_seq);
            let sp2 = p.p.space();
            let fce = p.p.literal(FC_END);
            let body = p.p.sequence(&[fcs, sp, tool_choice, more, sp2, fce]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            let fcs = p.p.literal(FC_START);
            let sp = p.p.space();
            let sp2 = p.p.space();
            let fce = p.p.literal(FC_END);
            let body = p.p.sequence(&[fcs, sp, tool_choice, sp2, fce]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        }

        let tool_calls = if !require_tools {
            p.p.optional(tool_calls)
        } else {
            tool_calls
        };

        let until = p.p.until(FC_START);
        let content_before_tools = p.content(until);
        p.p.sequence(&[
            generation_prompt,
            reasoning,
            content_before_tools,
            tool_calls,
            end,
        ])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(FC_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// DeepSeek V3.2 / V4 (parsers/deepseek.cpp)
// ---------------------------------------------------------------------------

/// `deepseek_v4_sort_tool_results` (parsers/deepseek.cpp:6-67)
fn deepseek_v4_sort_tool_results(messages: &Json) -> Json {
    let mut adjusted = messages.clone();
    let mut call_order: BTreeMap<String, usize> = BTreeMap::new();

    let Json::Array(msgs) = &mut adjusted else {
        return adjusted;
    };
    let mut i = 0usize;
    while i < msgs.len() {
        let role = msgs[i]
            .at("role")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("")
            .to_string();

        let is_assistant_with_calls = role == "assistant"
            && msgs[i]
                .at("tool_calls")
                .map(|t| t.is_array() && !t.empty())
                .unwrap_or(false);
        if is_assistant_with_calls {
            call_order.clear();
            if let Some(Json::Array(tool_calls)) = msgs[i].at("tool_calls") {
                for (idx, tc) in tool_calls.iter().enumerate() {
                    let id = tc.at("id").and_then(|v| v.get_str().ok()).unwrap_or("");
                    if !id.is_empty() {
                        call_order.insert(id.to_string(), idx);
                    }
                }
            }
            i += 1;
            continue;
        }

        if role != "user" && role != "tool" {
            i += 1;
            continue;
        }

        // collect a maximal run of user/tool messages - they render into one user block
        let mut tool_positions: Vec<usize> = Vec::new();
        let mut run_end = i;
        while run_end < msgs.len() {
            let r = msgs[run_end]
                .at("role")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            if r == "tool" {
                tool_positions.push(run_end);
            } else if r != "user" {
                break;
            }
            run_end += 1;
        }

        if tool_positions.len() > 1 && !call_order.is_empty() {
            let mut results: Vec<Json> = Vec::with_capacity(tool_positions.len());
            for &pos in &tool_positions {
                results.push(msgs[pos].clone());
            }
            // std::stable_sort by the preceding assistant message's call order
            results.sort_by_key(|m| {
                let id = m
                    .at("tool_call_id")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                call_order.get(&id).copied().unwrap_or(0)
            });
            for (k, &pos) in tool_positions.iter().enumerate() {
                msgs[pos] = results[k].clone();
            }
        }

        i = run_end;
    }

    adjusted
}

/// `common_chat_params_init_deepseek_v3_2` (parsers/deepseek.cpp:69-273)
#[allow(non_snake_case)]
pub(crate) fn chat_params_init_deepseek_v3_2(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    // V4 uses the same DSML markup as V3.2, but names the tool call block "tool_calls"
    // instead of "function_calls", renders tool results in tool call order and its
    // non-thinking generation prompt ends with a bare </think> instead of an empty
    // <think></think> pair.
    let is_v4 = !tmpl.source().contains("function_calls");

    let adjusted_messages = if is_v4 {
        Some(deepseek_v4_sort_tool_results(&inputs.messages))
    } else {
        None
    };

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    let mut additional_context: Option<Json> = None;
    if is_v4 && has_response_format {
        additional_context = Some(Json::Object(vec![(
            "response_format".to_string(),
            inputs.json_schema.clone(),
        )]));
    }

    let dsml: &str = "｜DSML｜";
    let THINK_START: &str = "<think>";
    let THINK_END: &str = "</think>";
    let TC_BLOCK: &str = if is_v4 {
        "tool_calls"
    } else {
        "function_calls"
    };
    let FC_START: String = format!("<{dsml}{TC_BLOCK}>");
    let FC_END: String = format!("</{dsml}{TC_BLOCK}>");
    let INVOKE_START: String = format!("<{dsml}invoke");
    let INVOKE_END: String = format!("</{dsml}invoke>");
    let PARAM_START: String = format!("<{dsml}parameter");
    let PARAM_END: String = format!("</{dsml}parameter>");
    let GEN_PROMPT: &str = "<｜Assistant｜>";
    let TC_SEPARATOR: &str = "\n\n";

    // lets the server find user turns in the prompt and place context checkpoints there
    data.message_delimiters = vec![
        ("assistant".to_string(), GEN_PROMPT.to_string()),
        ("user".to_string(), "<｜User｜>".to_string()),
    ];

    data.prompt = template_direct_apply_impl(
        tmpl,
        inputs,
        adjusted_messages.as_ref(),
        None,
        additional_context.as_ref(),
    )?;
    data.generation_prompt = template_generation_prompt_impl(
        tmpl,
        inputs,
        adjusted_messages.as_ref(),
        None,
        additional_context.as_ref(),
    )?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;
    data.thinking_start_tag = THINK_START.to_string();
    data.thinking_end_tags = vec![THINK_END.to_string(), FC_START.clone()];
    data.preserved_tokens = vec![
        dsml.to_string(),
        THINK_START.to_string(),
        THINK_END.to_string(),
    ];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        if is_v4 && msg.reasoning_content.is_empty() {
            data.generation_prompt = format!("{GEN_PROMPT}{THINK_END}");
            if inputs.continue_final_message == ChatContinuation::Content {
                data.generation_prompt += &msg.render_content("\n\n")?;
            }
        } else {
            data.generation_prompt = format!("{GEN_PROMPT}{THINK_START}{}", msg.reasoning_content);
            if inputs.continue_final_message == ChatContinuation::Content {
                data.generation_prompt += &format!("{THINK_END}{}", msg.render_content("\n\n")?);
            }
        }

        data.prompt += &data.generation_prompt;
    }

    let require_tools = inputs.tool_choice == ChatToolChoice::Required;
    let has_tool_calls = has_tools && inputs.tool_choice != ChatToolChoice::None;

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.p.literal(GEN_PROMPT);
        let end = p.p.end();

        // build tool call section first since we might need it in reasoning
        let mut tool_choice_alts: Vec<ParserId> = Vec::new();
        if has_tool_calls {
            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();

                let mut required_parsers: Vec<ParserId> = Vec::new();
                let mut optional_parsers: Vec<ParserId> = Vec::new();
                foreach_parameter(function, |param, doc| {
                    let is_string = doc.may_be_string(param.schema);

                    // <｜DSML｜parameter name="KEY" string="true|false">
                    let open1 = p.p.literal(&format!("{PARAM_START} name=\""));
                    let nm = p.p.literal(&param.name);
                    let tan = p.tool_arg_name(nm);
                    let flag = if is_string { "true" } else { "false" };
                    let open2 = p.p.literal(&format!("\" string=\"{flag}\">"));
                    let oseq = p.p.sequence(&[open1, tan, open2]);
                    let tao = p.tool_arg_open(oseq);

                    let value = if is_string {
                        let until = p.p.until(&PARAM_END);
                        p.tool_arg_string_value(until)
                    } else {
                        let json_p = p.p.json();
                        let sch = p.p.schema_node(
                            json_p,
                            &format!("tool-{name}-arg-{}-schema", param.name),
                            Rc::clone(doc),
                            param.schema,
                            false,
                        );
                        p.tool_arg_json_value(sch)
                    };
                    let close_lit = p.p.literal(&PARAM_END);
                    let tac = p.tool_arg_close(close_lit);
                    let arg_seq = p.p.sequence(&[tao, value, tac]);
                    let arg = p.tool_arg(arg_seq);

                    let named_arg =
                        p.p.rule(&format!("tool-{name}-arg-{}", param.name), arg, false);
                    if param.required {
                        required_parsers.push(named_arg);
                    } else {
                        optional_parsers.push(named_arg);
                    }
                });

                let mut args_seq = p.p.eps();
                for (i, &rp) in required_parsers.iter().enumerate() {
                    if i > 0 {
                        let sp = p.p.space();
                        args_seq = p.p.sequence(&[args_seq, sp]);
                    }
                    args_seq = p.p.sequence(&[args_seq, rp]);
                }

                if !optional_parsers.is_empty() {
                    let any_opt = p.p.choice(&optional_parsers);
                    let sp = p.p.space();
                    let seq = p.p.sequence(&[sp, any_opt]);
                    let rep = p.p.repeat3(seq, 0, -1);
                    args_seq = p.p.sequence(&[args_seq, rep]);
                }

                let invoke_body = args_seq;
                let open1 = p.p.literal(&format!("{INVOKE_START} name=\""));
                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let open2 = p.p.literal("\">\n");
                let oseq = p.p.sequence(&[open1, tn, open2]);
                let t_open = p.tool_open(oseq);
                let sp = p.p.space();
                let close_lit = p.p.literal(&INVOKE_END);
                let t_close = p.tool_close(close_lit);
                let body = p.p.sequence(&[t_open, invoke_body, sp, t_close]);
                let func_parser = p.tool(body);

                let r = p.p.rule(&format!("tool-{name}"), func_parser, false);
                tool_choice_alts.push(r);
            });
        }
        let tool_choice = p.p.choice(&tool_choice_alts);

        let mut tool_calls;
        if inputs.parallel_tool_calls {
            // p.literal(FC_START) + p.space() + tool_choice +
            //     pegc1!(p, zero_or_more, p.space() + tool_choice) + p.space() + p.literal(FC_END)
            let fcs = p.p.literal(&FC_START);
            let sp1 = p.p.space();
            let more_seq = p.p.sequence(&[sp1, tool_choice]);
            let more = p.p.zero_or_more(more_seq);
            let sp2 = p.p.space();
            let fce = p.p.literal(&FC_END);
            let body = p.p.sequence(&[fcs, tool_choice, more, sp2, fce]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        } else {
            let fcs = p.p.literal(&FC_START);
            let sp1 = p.p.space();
            let sp2 = p.p.space();
            let fce = p.p.literal(&FC_END);
            let body = p.p.sequence(&[fcs, sp1, tool_choice, sp2, fce]);
            tool_calls = p.p.trigger_rule("tool-call", body);
        }

        let obligatory_tool_calls = tool_calls;

        if !require_tools {
            tool_calls = p.p.optional(tool_calls);
        }

        let mut reasoning = p.p.eps();
        let mut reasoning_with_tc = p.p.eps();
        let mut allow_reasoning_with_tc = false;

        if extract_reasoning && inputs.enable_thinking {
            let ts = p.p.literal(THINK_START);
            let until = p.p.until(THINK_END);
            let r = p.reasoning(until);
            let te = p.p.literal(THINK_END);
            let seq = p.p.sequence(&[ts, r, te]);
            reasoning = p.p.optional(seq);

            let ts2 = p.p.literal(THINK_START);
            let tc_sep_start = format!("{TC_SEPARATOR}{FC_START}");
            let until2 =
                p.p.until_one_of(&[tc_sep_start.as_str(), FC_START.as_str(), THINK_END]);
            let r2 = p.reasoning(until2);
            let sp = p.p.space();
            let seq2 = p.p.sequence(&[ts2, r2, sp, obligatory_tool_calls]);
            reasoning_with_tc = seq2;
            allow_reasoning_with_tc = true;
        } else if extract_reasoning {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair (V3.2) or a bare </think> (V4) that
            // must still be consumed.
            if is_v4 {
                let te = p.p.literal(THINK_END);
                reasoning = p.p.optional(te);
            } else {
                let ts = p.p.literal(THINK_START);
                let until = p.p.until(THINK_END);
                let te = p.p.literal(THINK_END);
                let seq = p.p.sequence(&[ts, until, te]);
                reasoning = p.p.optional(seq);
            }
        }

        if has_response_format {
            let fence1 = p.p.literal("```json");
            let sp = p.p.space();
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
            let c = p.content(sch);
            let sp2 = p.p.space();
            let fence2 = p.p.literal("```");
            let body = p.p.sequence(&[fence1, sp, c, sp2, fence2]);
            let response_format = p.p.rule("response-format", body, false);
            return p
                .p
                .sequence(&[generation_prompt, reasoning, response_format, end]);
        }

        if !has_tool_calls {
            let rest = p.p.rest();
            let c = p.content(rest);
            return p.p.sequence(&[generation_prompt, reasoning, c, end]);
        }

        // content_before_tools = p.negate(p.literal(THINK_START)) +
        //     p.content(p.until_one_of({ TC_SEPARATOR + FC_START, FC_START })) + p.space()
        let ts_lit = p.p.literal(THINK_START);
        let neg = p.p.negate(ts_lit);
        let tc_sep_start = format!("{TC_SEPARATOR}{FC_START}");
        let until =
            p.p.until_one_of(&[tc_sep_start.as_str(), FC_START.as_str()]);
        let c = p.content(until);
        let sp = p.p.space();
        let content_before_tools = p.p.sequence(&[neg, c, sp]);
        if allow_reasoning_with_tc {
            let alt = p.p.sequence(&[reasoning, content_before_tools, tool_calls]);
            let choice = p.p.choice(&[reasoning_with_tc, alt]);
            return p.p.sequence(&[generation_prompt, choice, end]);
        }
        p.p.sequence(&[
            generation_prompt,
            reasoning,
            content_before_tools,
            tool_calls,
            end,
        ])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = has_tools && !require_tools;
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word(&FC_START)];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Gemma4 (parsers/gemma4.cpp)
// ---------------------------------------------------------------------------

/// `workaround::gemma4_model_turn_builder` (parsers/gemma4.cpp:15-125) —
/// flattens `assistant(tool_call+) tool+ [assistant(content)]` runs into one
/// assistant message with a `tool_responses` field.
struct Gemma4ModelTurnBuilder {
    tool_calls: Vec<Json>,
    tool_responses: Vec<Json>,
    content: Json,
    reasoning_content: Json,
    pos: usize,
}

impl Gemma4ModelTurnBuilder {
    fn has_content(msg: &Json) -> bool {
        let Some(content) = msg.at("content") else {
            return false;
        };
        if content.is_null() {
            return false;
        }
        if let Ok(s) = content.get_str() {
            return !s.is_empty();
        }
        if content.is_array() {
            return !content.empty();
        }
        false
    }

    fn has_tool_calls(msg: &Json) -> bool {
        msg.at("tool_calls")
            .map(|t| t.is_array() && !t.empty())
            .unwrap_or(false)
    }

    fn collect(&mut self, messages: &[Json]) {
        // Collect the first assistant message
        let msg = &messages[self.pos];
        if msg
            .at("reasoning_content")
            .map(|v| v.is_string())
            .unwrap_or(false)
        {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            self.reasoning_content = msg.at("reasoning_content").cloned().unwrap();
        }
        if let Some(Json::Array(tcs)) = msg.at("tool_calls") {
            for tc in tcs {
                self.tool_calls.push(tc.clone());
            }
        }
        self.pos += 1;

        // Collect tool call results
        while self.pos < messages.len() {
            let role = messages[self.pos]
                .at("role")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("");
            if role != "tool" {
                break;
            }
            self.collect_result(&messages[self.pos]);
            self.pos += 1;
        }

        // Check if the next assistant message is the final message
        if self.pos < messages.len() {
            let next = &messages[self.pos];
            let role = next.at("role").and_then(|v| v.get_str().ok()).unwrap_or("");
            if role == "assistant" && !Self::has_tool_calls(next) && Self::has_content(next) {
                self.content = next.at("content").cloned().unwrap();
                self.pos += 1;
            }
        }
    }

    fn collect_result(&mut self, curr: &Json) {
        // Try to parse the content as JSON; fall back to raw string
        let response = if let Some(content) = curr.at("content") {
            if let Ok(text) = content.get_str() {
                match Json::parse(text) {
                    Ok(parsed) => parsed,
                    Err(_) => content.clone(),
                }
            } else {
                content.clone()
            }
        } else {
            Json::Null
        };

        let mut name = String::new();

        // Match name with corresponding tool call
        let idx = self.tool_responses.len();
        if idx < self.tool_calls.len() {
            let tc = &self.tool_calls[idx];
            if let Some(function) = tc.at("function") {
                name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
            }
        }

        // Fallback to the tool call id
        if name.is_empty() {
            name = curr
                .at("tool_call_id")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
        }

        self.tool_responses.push(Json::Object(vec![
            ("name".to_string(), Json::String(name)),
            ("response".to_string(), response),
        ]));
    }

    fn build(mut self, messages: &[Json], pos: usize) -> (Json, usize) {
        self.pos = pos;
        self.collect(messages);

        let mut msg = Json::Object(vec![
            ("role".to_string(), Json::String("assistant".to_string())),
            (
                "tool_calls".to_string(),
                Json::Array(std::mem::take(&mut self.tool_calls)),
            ),
        ]);
        if !self.tool_responses.is_empty() {
            msg.set(
                "tool_responses",
                Json::Array(std::mem::take(&mut self.tool_responses)),
            );
        }
        if !self.content.is_null() {
            msg.set("content", self.content.clone());
        }
        if !self.reasoning_content.is_null() {
            msg.set("reasoning_content", self.reasoning_content.clone());
        }
        (msg, self.pos)
    }
}

/// `workaround::convert_tool_responses_gemma4` (parsers/gemma4.cpp:127-147)
pub(crate) fn convert_tool_responses_gemma4(messages: &mut Json) {
    let Json::Array(src) = messages.clone() else {
        return;
    };
    let mut result: Vec<Json> = Vec::new();
    let mut i = 0usize;

    while i < src.len() {
        let msg = &src[i];
        let role_is_assistant = msg
            .at("role")
            .and_then(|v| v.get_str().ok())
            .map(|r| r == "assistant")
            .unwrap_or(false);
        if !role_is_assistant
            || !msg
                .at("tool_calls")
                .map(|t| t.is_array() && !t.empty())
                .unwrap_or(false)
        {
            result.push(msg.clone());
            i += 1;
            continue;
        }

        let builder = Gemma4ModelTurnBuilder {
            tool_calls: Vec::new(),
            tool_responses: Vec::new(),
            content: Json::Null,
            reasoning_content: Json::Null,
            pos: 0,
        };
        let (built, pos) = builder.build(&src, i);
        result.push(built);
        i = pos;
    }

    *messages = Json::Array(result);
}

/// `common_chat_params_init_gemma4` (parsers/gemma4.cpp:151-307)
pub(crate) fn chat_params_init_gemma4(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;

    if inputs.add_generation_prompt && string_ends_with(&data.prompt, "<turn|>\n") {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.generation_prompt = "<|turn>model\n".to_string();
        data.prompt += &data.generation_prompt;
    }

    data.message_delimiters = vec![
        ("user".to_string(), "<|turn>user".to_string()),
        ("assistant".to_string(), "<|turn>model".to_string()),
    ];

    data.format = ChatFormat::PegGemma4;
    data.supports_thinking = true;
    data.thinking_start_tag = "<|channel>thought".to_string();
    data.thinking_end_tags = vec!["<channel|>".to_string()];

    data.preserved_tokens = vec![
        "<|channel>".to_string(),
        "<channel|>".to_string(),
        "<|tool_call>".to_string(),
        "<tool_call|>".to_string(),
        "<|turn>".to_string(),
    ];

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = if string_ends_with(&data.prompt, "<turn|>\n") {
            "<|turn>model\n".to_string()
        } else {
            String::new()
        };
        data.generation_prompt += &format!("<|channel>thought\n{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("<channel|>{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let start_opt = peg1!(p, optional, p.p.literal("<|turn>model\n"));
        let start = p.p.rule("start", start_opt, false);

        if extract_reasoning {
            let ch = p.p.literal("<|channel>thought");
            let sp = p.p.space();
            let until = p.p.until("<channel|>");
            let r = p.reasoning(until);
            let close = p.p.literal("<channel|>");
            let seq = p.p.sequence(&[ch, sp, r, close]);
            p.p.rule("thought", seq, false);
        } else {
            let ch = p.p.literal("<|channel>thought");
            let sp = p.p.space();
            let until = p.p.until("<channel|>");
            let close = p.p.literal("<channel|>");
            let seq = p.p.sequence(&[ch, sp, until, close]);
            let c = p.content(seq);
            p.p.rule("thought", c, false);
        }

        // consume_empty_channels = gbnf(zero_or_more(literal("<|channel>") + negate(literal("thought"))), "")
        let ch_lit = p.p.literal("<|channel>");
        let thought_lit = p.p.literal("thought");
        let neg = p.p.negate(thought_lit);
        let seq = p.p.sequence(&[ch_lit, neg]);
        let zom = p.p.zero_or_more(seq);
        let consume_empty_channels = p.p.gbnf(zom, "");

        // thought = (peek(literal("<|channel>")) + consume_empty_channels + ref("thought")) | negate(literal("<|channel>"))
        let peek_lit = p.p.literal("<|channel>");
        let pk = p.p.peek(peek_lit);
        let thought_ref = p.p.ref_("thought");
        let pos_seq = p.p.sequence(&[pk, consume_empty_channels, thought_ref]);
        let neg_lit = p.p.literal("<|channel>");
        let neg = p.p.negate(neg_lit);
        let thought = p.p.choice(&[pos_seq, neg]);

        if has_response_format {
            let fence1 = p.p.literal("```json");
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format-schema", &inputs.json_schema, false);
            let c = p.content(sch);
            let fence2 = p.p.literal("```");
            let a = p.p.spaced(fence1, c);
            let response_format = p.p.spaced(a, fence2);
            let opt = p.p.optional(thought);
            return p.p.sequence(&[start, opt, response_format]);
        }

        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            // Gemma4 tool calling syntax
            // Rules should match traversal logic in gemma4_to_json()
            let sc_content = p.p.until("<|\"|>");
            p.p.rule("gemma4-string-content", sc_content, false);
            let q1 = p.p.literal("<|\"|>");
            let sc_ref = p.p.ref_("gemma4-string-content");
            let q2 = p.p.literal("<|\"|>");
            let str_seq = p.p.sequence(&[q1, sc_ref, q2]);
            p.p.rule("gemma4-string", str_seq, false);
            let jb = p.p.json_bool();
            p.p.rule("gemma4-bool", jb, false);
            let jn = p.p.json_null();
            p.p.rule("gemma4-null", jn, false);
            let jnum = p.p.json_number();
            p.p.rule("gemma4-number", jnum, false);
            let key_name_chars = p.p.chars("[^:}]", 1, -1);
            let key_name = p.p.rule("gemma4-dict-key-name", key_name_chars, false);
            let colon = p.p.literal(":");
            let key_seq = p.p.sequence(&[key_name, colon]);
            p.p.rule("gemma4-dict-key", key_seq, false);
            {
                let key_ref = p.p.ref_("gemma4-dict-key");
                let sp = p.p.space();
                let value_ref = p.p.ref_("gemma4-value");
                let kv_seq = p.p.sequence(&[key_ref, sp, value_ref]);
                p.p.rule("gemma4-dict-kv", kv_seq, false);
            }
            {
                // gemma4-dict (parsers/gemma4.cpp:230-238): one ws and one
                // member ref, reused across sites (the C++ lambda captures)
                let ws = p.p.space();
                let member = p.p.ref_("gemma4-dict-kv");
                let comma = p.p.literal(",");
                let tail_seq = p.p.sequence(&[comma, ws, member]);
                let tail = p.p.zero_or_more(tail_seq);
                let members = p.p.sequence(&[member, tail]);
                let lb = p.p.literal("{");
                let rb1 = p.p.literal("}");
                let rb2 = p.p.literal("}");
                let inner_members = p.p.sequence(&[members, ws, rb2]);
                let inner = p.p.choice(&[rb1, inner_members]);
                let seq = p.p.sequence(&[lb, ws, inner]);
                p.p.rule("gemma4-dict", seq, false);
            }
            {
                // gemma4-array (parsers/gemma4.cpp:239-247)
                let ws = p.p.space();
                let value = p.p.ref_("gemma4-value");
                let comma = p.p.literal(",");
                let tail_seq = p.p.sequence(&[comma, ws, value]);
                let tail = p.p.zero_or_more(tail_seq);
                let elements = p.p.sequence(&[value, tail]);
                let lb = p.p.literal("[");
                let rb1 = p.p.literal("]");
                let rb2 = p.p.literal("]");
                let inner_elements = p.p.sequence(&[elements, ws, rb2]);
                let inner = p.p.choice(&[rb1, inner_elements]);
                let seq = p.p.sequence(&[lb, ws, inner]);
                p.p.rule("gemma4-array", seq, false);
            }
            {
                // gemma4-value
                let s = p.p.ref_("gemma4-string");
                let d = p.p.ref_("gemma4-dict");
                let a = p.p.ref_("gemma4-array");
                let n = p.p.ref_("gemma4-number");
                let b = p.p.ref_("gemma4-bool");
                let z = p.p.ref_("gemma4-null");
                let ch = p.p.choice(&[s, d, a, n, b, z]);
                p.p.rule("gemma4-value", ch, false);
            }

            let mut tool_choice_alts: Vec<ParserId> = Vec::new();

            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                // TODO @aldehir : need to extend json-schema-to-grammar to produce more than JSON rules
                // const auto & params = function.at("parameters");

                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let brace = p.p.literal("{");
                let pk = p.p.peek(brace);
                let oseq = p.p.sequence(&[tn, pk]);
                let t_open = p.tool_open(oseq);
                let dict_ref = p.p.ref_("gemma4-dict");
                let ta = p.tool_args(dict_ref);
                let seq = p.p.sequence(&[t_open, ta]);
                let tool_parser = p.tool(seq);

                let r = p.p.rule(&format!("tool-{name}"), tool_parser, false);
                tool_choice_alts.push(r);
            });
            let tool_choice = p.p.choice(&tool_choice_alts);

            let min = if inputs.tool_choice == ChatToolChoice::Required {
                1
            } else {
                0
            };
            let max = if inputs.parallel_tool_calls { -1 } else { 1 };
            // p.repeat("<|tool_call>call:" + tool_choice + "<tool_call|>", min, max)
            let tc_open = p.p.literal("<|tool_call>call:");
            let tc_close = p.p.literal("<tool_call|>");
            let tc_seq = p.p.sequence(&[tc_open, tool_choice, tc_close]);
            let rep = p.p.repeat3(tc_seq, min, max);
            let tool_call = p.p.trigger_rule("tool-call", rep);

            if inputs.tool_choice == ChatToolChoice::Required {
                return p.p.sequence(&[start, thought, tool_call]);
            }

            let scan_until = p.p.until("<|tool_call>");
            let scan_to_toolcall = p.p.rule("scan-to-toolcall", scan_until, false);
            let content_until =
                p.p.until_one_of(&["<|channel>", "<channel|>", "<|tool_call>"]);
            let content_c = p.content(content_until);
            let content = p.p.rule("content", content_c, false);
            let message_body = p.p.sequence(&[thought, content]);
            let message = p.p.rule("message", message_body, false);
            let zom = p.p.zero_or_more(message);
            return p.p.sequence(&[start, zom, scan_to_toolcall, tool_call]);
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        let content_until = p.p.until_one_of(&["<|channel>", "<channel|>"]);
        let content_c = p.content(content_until);
        let content = p.p.rule("content", content_c, false);
        let message_body = p.p.sequence(&[thought, content]);
        let message = p.p.rule("message", message_body, false);
        let oom = p.p.one_or_more(message);
        p.p.sequence(&[start, oom])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word("<|tool_call>")];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// MiniCPM5 (parsers/minicpm5.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_minicpm5` (parsers/minicpm5.cpp:6-130)
pub(crate) fn chat_params_init_minicpm5(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;
    data.supports_thinking = true;
    data.preserved_tokens = vec![
        "<function".to_string(),
        "<param".to_string(),
        "</function>".to_string(),
        "</param>".to_string(),
        "<think>".to_string(),
        "</think>".to_string(),
    ];

    data.thinking_start_tag = "<think>".to_string();
    data.thinking_end_tags = vec!["</think>".to_string()];

    data.message_delimiters = vec![
        ("assistant".to_string(), "<|im_start|>assistant".to_string()),
        (
            "tool".to_string(),
            "<|im_start|>user\n<tool_response>".to_string(),
        ),
        ("user".to_string(), "<|im_start|>user".to_string()),
        ("system".to_string(), "<|im_start|>system".to_string()),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt =
            format!("<|im_start|>assistant\n<think>\n{}", msg.reasoning_content);
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &format!("\n</think>\n\n{}", msg.render_content("\n\n")?);
        }

        data.prompt += &data.generation_prompt;
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.p.literal("<|im_start|>assistant\n");

        let mut reasoning = p.p.eps();
        if extract_reasoning {
            // ("<think>" << p.reasoning(p.until("</think>")) << "</think>") + p.space()
            let ts = p.p.literal("<think>");
            let until = p.p.until("</think>");
            let r = p.reasoning(until);
            let te = p.p.literal("</think>");
            let a = p.p.spaced(ts, r);
            let b = p.p.spaced(a, te);
            let sp = p.p.space();
            reasoning = p.p.sequence(&[b, sp]);
        }

        // Response format parser
        if has_response_format {
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format", &inputs.json_schema, false);
            let c = p.content(sch);
            return p.p.sequence(&[generation_prompt, reasoning, c]);
        }

        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            // CDATA lets a value carry characters that would otherwise close the tag (e.g.
            // </param>); capture the inner text only, excluding the CDATA markers.
            let cdata_branch = {
                let open = p.p.literal("<![CDATA[");
                let until = p.p.until("]]>");
                let sv = p.tool_arg_string_value(until);
                let close_ac = p.p.literal("]]>");
                let seq = p.p.sequence(&[sv, close_ac]);
                let ac = p.p.ac(seq, &["]]>".to_string()]);
                let close_lit = p.p.literal("</param>");
                let tac = p.tool_arg_close(close_lit);
                p.p.sequence(&[open, ac, tac])
            };
            let plain_branch = {
                let neg_lit = p.p.literal("<![CDATA[");
                let neg = p.p.negate(neg_lit);
                let until = p.p.until("</param>");
                let sv = p.tool_arg_string_value(until);
                let close_lit = p.p.literal("</param>");
                let tac = p.tool_arg_close(close_lit);
                let seq = p.p.sequence(&[sv, tac]);
                let ac = p.p.ac(seq, &["</param>".to_string()]);
                p.p.sequence(&[neg, ac])
            };
            let string_value = p.p.choice(&[cdata_branch, plain_branch]);

            let mut tool_choice_alts: Vec<ParserId> = Vec::new();
            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();

                let mut arg_rules: Vec<ParserId> = Vec::new();
                foreach_parameter(function, |prop, doc| {
                    let value_parser = if doc.may_be_string(prop.schema) {
                        string_value
                    } else {
                        let json_p = p.p.json();
                        let sch = p.p.schema_node(
                            json_p,
                            &format!("tool-{name}-arg-{}-schema", prop.name),
                            Rc::clone(doc),
                            prop.schema,
                            false,
                        );
                        let jv = p.tool_arg_json_value(sch);
                        let close_lit = p.p.literal("</param>");
                        let tac = p.tool_arg_close(close_lit);
                        p.p.sequence(&[jv, tac])
                    };

                    let open1 = p.p.literal("<param name=\"");
                    let nm = p.p.literal(&prop.name);
                    let tan = p.tool_arg_name(nm);
                    let open2 = p.p.literal("\">");
                    let oseq = p.p.sequence(&[open1, tan, open2]);
                    let tao = p.tool_arg_open(oseq);
                    let arg_seq = p.p.sequence(&[tao, value_parser]);
                    arg_rules.push(p.tool_arg(arg_seq));
                });

                let mut args = p.p.eps();
                if !arg_rules.is_empty() {
                    let choice = p.p.choice(&arg_rules);
                    let sp = p.p.space();
                    let seq = p.p.sequence(&[choice, sp]);
                    args = p.p.zero_or_more(seq);
                }

                let open1 = p.p.literal("<function name=\"");
                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let open2 = p.p.literal("\">");
                let oseq = p.p.sequence(&[open1, tn, open2]);
                let t_open = p.tool_open(oseq);
                let ta = p.tool_args(args);
                let close_lit = p.p.literal("</function>");
                let t_close = p.tool_close(close_lit);
                // p.tool(p.tool_open(...) << p.tool_args(args) << p.tool_close(...))
                let a = p.p.spaced(t_open, ta);
                let body = p.p.spaced(a, t_close);
                let tool_parser = p.tool(body);

                let r = p.p.rule(&format!("tool-{name}"), tool_parser, false);
                tool_choice_alts.push(r);
            });
            let tool_choice = p.p.choice(&tool_choice_alts);

            let max_calls = if inputs.parallel_tool_calls { -1 } else { 1 };
            // p.repeat(tool_choice + p.space(), 1, max_calls)
            let sp = p.p.space();
            let seq = p.p.sequence(&[tool_choice, sp]);
            let rep = p.p.repeat3(seq, 1, max_calls);
            let tool_calls = p.p.trigger_rule("tool-call", rep);

            let until = p.p.until("<function");
            let content = p.content(until);
            let end = p.p.end();

            return p
                .p
                .sequence(&[generation_prompt, reasoning, content, tool_calls, end]);
        }

        let rest = p.p.rest();
        let c = p.content(rest);
        let end = p.p.end();
        p.p.sequence(&[generation_prompt, reasoning, c, end])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy =
            !(has_response_format || (has_tools && inputs.tool_choice == ChatToolChoice::Required));
        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        data.grammar_triggers = vec![GrammarTrigger::word("<function")];
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Qwen3-Coder (parsers/qwen3-coder.cpp)
// ---------------------------------------------------------------------------

/// `common_chat_params_init_qwen3_coder` (parsers/qwen3-coder.cpp:3-194)
pub(crate) fn chat_params_init_qwen3_coder(
    tmpl: &ChatTemplate,
    inputs: &GenerationParams,
) -> Result<ChatParams, String> {
    let mut data = ChatParams::default();

    const GEN_PREFIX: &str = "<|im_start|>assistant\n";

    data.prompt = template_direct_apply_impl(tmpl, inputs, None, None, None)?;
    data.generation_prompt = template_generation_prompt_impl(tmpl, inputs, None, None, None)?;
    data.format = ChatFormat::PegNative;

    let supports_reasoning = tmpl.source().contains("<think>");

    data.supports_thinking = supports_reasoning;
    data.preserved_tokens = vec!["<tool_call>".to_string(), "</tool_call>".to_string()];

    let is_qwen3_coder = !supports_reasoning;

    if supports_reasoning {
        data.thinking_start_tag = "<think>".to_string();
        // Support both </think> and <tool_call> as reasoning end sequences.
        // The newline variant comes first so it is included in the forced message
        // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
        data.thinking_end_tags = vec![
            "\n</think>".to_string(),
            "</think>".to_string(),
            "<tool_call>".to_string(),
        ];
        data.preserved_tokens.push("<think>".to_string());
        data.preserved_tokens.push("</think>".to_string());
    }

    data.message_delimiters = vec![
        ("assistant".to_string(), "<|im_start|>assistant".to_string()),
        // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
        (
            "tool".to_string(),
            "<|im_start|>user\n<tool_response>".to_string(),
        ),
        // StepFun-3.5-Flash
        ("tool".to_string(), "<|im_start|>tool_response".to_string()),
        ("user".to_string(), "<|im_start|>user".to_string()),
        ("system".to_string(), "<|im_start|>system".to_string()),
    ];

    let has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    let has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    let extract_reasoning = inputs.reasoning_format != ReasoningFormat::None;
    let include_grammar =
        has_response_format || (has_tools && inputs.tool_choice != ChatToolChoice::None);

    if inputs.has_continuation() {
        let msg = &inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX.to_string();
        if supports_reasoning {
            data.generation_prompt += &format!("<think>\n{}", msg.reasoning_content);
            if inputs.continue_final_message == ChatContinuation::Content {
                data.generation_prompt += "\n</think>\n\n";
            }
        }
        if inputs.continue_final_message == ChatContinuation::Content {
            data.generation_prompt += &msg.render_content("\n\n")?;
        }

        data.prompt += &data.generation_prompt;
    }

    let mut tool_call_starts: Vec<String> = vec!["<tool_call>".to_string()];

    if is_qwen3_coder {
        // Match complete <function=name> opener for Qwen3-Coder models that occasionally omit the
        // starting <tool_call>. The model may hallucinate a tool name, but it is preferable over
        // constraining on <function which may occur in valid content generation, e.g. #include <functional>
        foreach_function(&inputs.tools, |function| {
            let name = function
                .at("name")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("")
                .to_string();
            tool_call_starts.push(format!("<function={name}>"));
        });
    }

    let parser = crate::chat_tools::build_chat_peg_parser(|p| {
        let generation_prompt = p.p.literal(GEN_PREFIX);

        let mut reasoning = p.p.eps();
        if supports_reasoning && extract_reasoning {
            let ts = p.p.literal("<think>");
            let sp = p.p.space();
            let until = p.p.until_one_of(&["</think>", "<tool_call>"]);
            let r = p.reasoning(until);
            // (p.literal("</think>") | p.peek(p.literal("<tool_call>")))
            let close = {
                let te = p.p.literal("</think>");
                let tc_lit = p.p.literal("<tool_call>");
                let pk = p.p.peek(tc_lit);
                p.p.choice(&[te, pk])
            };
            let seq = p.p.sequence(&[ts, sp, r, close]);
            reasoning = p.p.optional(seq);
        }

        // Response format parser
        if has_response_format {
            let json_p = p.p.json();
            let sch =
                p.p.schema(json_p, "response-format", &inputs.json_schema, false);
            let c = p.content(sch);
            let seq = p.p.spaced(reasoning, c);
            return p.p.sequence(&[generation_prompt, seq]);
        }

        // Tool call parser
        if has_tools && inputs.tool_choice != ChatToolChoice::None {
            let arg_close_lit = p.p.literal("\n</parameter>\n");
            let arg_close = p.tool_arg_close(arg_close_lit);
            let arg_string_until = p.p.until("\n</parameter>\n");
            let arg_string_v = p.tool_arg_string_value(arg_string_until);
            let arg_string_seq = p.p.sequence(&[arg_string_v, arg_close]);
            let arg_string = peg3!(
                p,
                rule,
                "xml-arg-string",
                p.p.ac(arg_string_seq, &["\n</parameter>\n".to_string()]),
                false
            );

            let mut tool_choice_alts: Vec<ParserId> = Vec::new();
            foreach_function(&inputs.tools, |function| {
                let name = function
                    .at("name")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();

                let mut required_args: Vec<ParserId> = Vec::new();
                let mut optional_args: Vec<ParserId> = Vec::new();

                foreach_parameter(function, |param, doc| {
                    let rule_name = format!("tool-{name}-arg-{}", param.name);

                    // p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">\n")
                    let open1 = p.p.literal("<parameter=");
                    let nm = p.p.literal(&param.name);
                    let tan = p.tool_arg_name(nm);
                    let open2 = p.p.literal(">\n");
                    let oseq = p.p.sequence(&[open1, tan, open2]);
                    let arg_open = p.tool_arg_open(oseq);

                    let types = doc.value_types(param.schema);

                    let arg_value;
                    if !types.has(ValueType::String) {
                        let json_p = p.p.json();
                        let sch = p.p.schema_node(
                            json_p,
                            &format!("{rule_name}-schema"),
                            Rc::clone(doc),
                            param.schema,
                            false,
                        );
                        let jv = p.tool_arg_json_value(sch);
                        arg_value = p.p.sequence(&[jv, arg_close]);
                    } else if types.is_only(ValueType::String) {
                        arg_value = arg_string;
                    } else {
                        // The string alternative accepts any text, so the grammar only keeps the raw string
                        // rule. The parser still tries the JSON alternatives first to type the value.
                        let mut json_value_alts: Vec<ParserId> = Vec::new();
                        if types.has(ValueType::Object) {
                            json_value_alts.push(p.p.json_object());
                        }
                        if types.has(ValueType::Array) {
                            json_value_alts.push(p.p.json_array());
                        }
                        if types.has(ValueType::Number) || types.has(ValueType::Integer) {
                            json_value_alts.push(p.p.json_number());
                        }
                        if types.has(ValueType::Boolean) {
                            json_value_alts.push(p.p.json_bool());
                        }
                        if types.has(ValueType::Null) {
                            json_value_alts.push(p.p.json_null());
                        }
                        let json_value = p.p.choice(&json_value_alts);
                        let jv = p.tool_arg_json_value(json_value);
                        let jseq = p.p.sequence(&[jv, arg_close]);
                        let jatomic = p.p.atomic(jseq);
                        let alt = p.p.choice(&[jatomic, arg_string]);
                        arg_value = p.p.gbnf(alt, "xml-arg-string");
                    }

                    let arg_rule_body = p.p.sequence(&[arg_open, arg_value]);
                    let arg_rule = p.tool_arg(arg_rule_body);
                    let arg_rule = p.p.rule(&rule_name, arg_rule, false);

                    if param.required {
                        required_args.push(arg_rule);
                    } else {
                        optional_args.push(arg_rule);
                    }
                });

                // Accept required arguments in any order, as Qwen does not always adhere to the
                // order provided.
                let mut args = p.permute(&format!("tool-{name}-args"), &required_args);
                if !optional_args.is_empty() {
                    let any_opt = p.p.choice(&optional_args);
                    let zom = p.p.zero_or_more(any_opt);
                    args = p.p.sequence(&[args, zom]);
                }

                let open1 = p.p.literal("<function=");
                let nm = p.p.literal(&name);
                let tn = p.tool_name(nm);
                let open2 = p.p.literal(">\n");
                let oseq = p.p.sequence(&[open1, tn, open2]);
                let t_open = p.tool_open(oseq);
                let ta = p.tool_args(args);
                let ce = p.p.literal("</function>\n");
                let t_close = p.tool_close(ce);
                let func = pegc1!(p, tool, p.p.sequence(&[t_open, ta, t_close]));

                let r = p.p.rule(&format!("tool-{name}"), func, false);
                tool_choice_alts.push(r);
            });
            let tool_choice = p.p.choice(&tool_choice_alts);

            let min_calls = if inputs.tool_choice == ChatToolChoice::Required {
                1
            } else {
                0
            };

            // tool_call_body = tool_choice + "</tool_call>" + p.space()
            let tc_end = p.p.literal("</tool_call>");
            let sp = p.p.space();
            let tool_call_body = p.p.sequence(&[tool_choice, tc_end, sp]);
            // tool_call = p.rule("tool-call", "<tool_call>\n" + tool_call_body)
            let tc_open = p.p.literal("<tool_call>\n");
            let tc_seq = p.p.sequence(&[tc_open, tool_call_body]);
            let tool_call = p.p.rule("tool-call", tc_seq, false);

            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            let tool_call_first = if is_qwen3_coder {
                let tc_open2 = p.p.literal("<tool_call>\n");
                let opt = p.p.optional(tc_open2);
                let seq = p.p.sequence(&[opt, tool_call_body]);
                p.p.rule("tool-call-first", seq, false)
            } else {
                tool_call
            };

            let calls = if inputs.parallel_tool_calls {
                let more = p.p.zero_or_more(tool_call);
                p.p.sequence(&[tool_call_first, more])
            } else {
                tool_call_first
            };
            // p.repeat(calls, min_calls, 1)
            let rep = p.p.repeat3(calls, min_calls, 1);
            let tool_calls = p.p.trigger_rule("tool-call-root", rep);

            // reasoning << p.content(p.until_one_of(tool_call_starts)) << tool_calls
            let starts: Vec<&str> = tool_call_starts.iter().map(|s| s.as_str()).collect();
            let until = p.p.until_one_of(&starts);
            let c = p.content(until);
            let a = p.p.spaced(reasoning, c);
            let seq = p.p.spaced(a, tool_calls);
            return p.p.sequence(&[generation_prompt, seq]);
        }

        // Content only parser
        let rest = p.p.rest();
        let c = p.content(rest);
        let seq = p.p.spaced(reasoning, c);
        p.p.sequence(&[generation_prompt, seq])
    })?;

    data.parser = parser.save();

    if include_grammar {
        data.grammar_lazy = has_tools && inputs.tool_choice == ChatToolChoice::Auto;

        data.grammar = parser.build_grammar(data.grammar_lazy)?;

        if data.grammar_lazy {
            for start in &tool_call_starts {
                data.grammar_triggers.push(GrammarTrigger::word(start));
            }
        }
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// dispatch — common_chat_try_specialized_template (chat.cpp:1090-1223)
// ---------------------------------------------------------------------------

/// `common_chat_try_specialized_template` (chat.cpp:1090-1223)
pub(crate) fn try_specialized_template(
    tmpl: &ChatTemplate,
    src: &str,
    params: &mut GenerationParams,
) -> Result<Option<ChatParams>, String> {
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if src.contains("[SYSTEM_PROMPT]")
        && src.contains("[TOOL_CALLS]")
        && src.contains("[ARGS]")
        && !src.contains("[CALL_ID]")
    {
        return Ok(Some(chat_params_init_ministral_3(tmpl, params)?));
    }

    // LLM-jp-4.1 - GPT-OSS dialect (spaces after special tokens, <|end|>-separated parallel calls)
    if src.contains("chat_format=llm-jp-harmony-v1") {
        return Ok(Some(chat_params_init_llm_jp_harmony(tmpl, params)?));
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if src.contains("<|channel|>") {
        return Ok(Some(chat_params_init_gpt_oss(tmpl, params)?));
    }

    // Muse Glimmer format using " to=<recipient>" recipients and <|eom|>/<|eot|> message terminators.
    if src.contains("<atem:function_calls>") && src.contains("<|eom|>") {
        return Ok(Some(chat_params_init_muse_glimmer(tmpl, params)?));
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if src.contains(">>>all") && src.contains(">>>${recipient}") {
        return Ok(Some(chat_params_init_functionary_v3_2(tmpl, params)?));
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if src.contains("<|tool_calls_section_begin|>") && src.contains("<|tool_call_begin|>") {
        return Ok(Some(chat_params_init_kimi_k2(tmpl, params)?));
    }

    // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
    if src.contains("<|open|>") && src.contains("<|close|>") && src.contains("<|end_of_msg|>") {
        return Ok(Some(chat_params_init_kimi_k3(tmpl, params)?));
    }

    // Ling 3.0 / Bailing V3 - <role>X</role> sections with <arg_key>/<arg_value> tagged
    // tool calls. <role> sections are unique to this family among the tagged-arg templates.
    if src.contains("<role>ASSISTANT</role>") && src.contains("<arg_key>") {
        return Ok(Some(chat_params_init_ling3(tmpl, params)?));
    }

    // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
    // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
    // Command-R templates use <|START_RESPONSE|>).
    if src.contains("<|START_TEXT|>") && src.contains("<|START_ACTION|>") {
        return Ok(Some(chat_params_init_cohere2moe(tmpl, params)?));
    }

    if is_lfm2_template(src) {
        return Ok(Some(chat_params_init_lfm2(
            tmpl, params, /* tool_list_tokens = */ true,
        )?));
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if src.contains("List of tools: [") && !src.contains("<|tool_list_start|>") {
        return Ok(Some(chat_params_init_lfm2(
            tmpl, params, /* tool_list_tokens = */ false,
        )?));
    }

    // GigaChatV3 format detection
    if src.contains("<|role_sep|>")
        && src.contains("<|message_sep|>")
        && !src.contains("<|function_call|>")
    {
        return Ok(Some(chat_params_init_gigachat_v3(tmpl, params)?));
    }

    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if src.contains("]<]minimax[>[") && src.contains("<tool_call>") && src.contains("<invoke name=")
    {
        return Ok(Some(chat_params_init_minimax_m3(tmpl, params)?));
    }

    // DeepSeek V3.2/V4 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    // V3.2 names the tool call block "function_calls", V4 names it "tool_calls".
    if src.contains("dsml_token")
        && src.contains("DSML")
        && (src.contains("function_calls") || src.contains("tool_calls"))
    {
        return Ok(Some(chat_params_init_deepseek_v3_2(tmpl, params)?));
    }

    // Gemma4 format detection
    if src.contains("'<|tool_call>call:'") {
        if !src.contains("{#- OpenAI Chat Completions:") {
            // apply workarounds if using the older gemma4 templates
            // LOG_WRN: detected an outdated gemma4 chat template, applying
            // compatibility workarounds. Consider updating to the official
            // template. (chat.cpp:1210-1213)
            eprintln!(
                "common_chat_try_specialized_template: detected an outdated gemma4 chat template, applying compatibility workarounds. Consider updating to the official template."
            );
            convert_tool_responses_gemma4(&mut params.messages);
        }
        return Ok(Some(chat_params_init_gemma4(tmpl, params)?));
    }

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
    if src.contains("Tool usage guidelines:")
        && src.contains("<function name=\"")
        && src.contains("<param name=\"")
    {
        return Ok(Some(chat_params_init_minicpm5(tmpl, params)?));
    }

    // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
    if src.contains("<tool_call>")
        && src.contains("<function=")
        && src.contains("<parameter=")
        // Exclude models that don't use \n between tags
        && !src.contains("'<tool_call><function=' ~ tool_call.name ~ '>'")
    {
        return Ok(Some(chat_params_init_qwen3_coder(tmpl, params)?));
    }

    Ok(None)
}

// ---------------------------------------------------------------------------
// gemma4 / minimax-m3 AST mappers (chat-peg-parser.cpp:956-1232)
// ---------------------------------------------------------------------------

/// `common_chat_peg_gemma4_mapper` (chat-peg-parser.h:35-41). `from_ast`
/// walks the result's root nodes (chat-peg-parser.cpp:956-960), `visit`
/// (1057-1094) maps tags; tool arguments are rebuilt from the gemma4-* rules
/// by [`gemma4_to_json`] (962-1055).
pub(crate) struct ChatPegGemma4Mapper<'a> {
    pub result: &'a mut ChatMsg,
}

impl<'a> ChatPegGemma4Mapper<'a> {
    pub fn new(result: &'a mut ChatMsg) -> Self {
        ChatPegGemma4Mapper { result }
    }

    /// `from_ast` (chat-peg-parser.cpp:956-960)
    pub fn from_ast(&mut self, ctx: &ParseContext, result: &ParseResult) {
        for &node_id in &result.nodes {
            self.visit(ctx, node_id);
        }
    }

    /// `visit` (chat-peg-parser.cpp:1057-1094)
    fn visit(&mut self, ctx: &ParseContext, id: usize) {
        let node = ctx.ast.get(id);

        if node.tag == chat_tag::REASONING {
            self.result.reasoning_content += &node.sanitized_text(ctx.input.as_bytes());
            return;
        }

        if node.tag == chat_tag::CONTENT {
            self.result.content += &node.sanitized_text(ctx.input.as_bytes());
            return;
        }

        if node.tag == chat_tag::TOOL {
            let name_id = ctx.ast.find_by_tag(node, chat_tag::TOOL_NAME, 3);
            let args_id = ctx.ast.find_by_tag(node, chat_tag::TOOL_ARGS, 3);

            if name_id != INVALID_AST_ID && args_id != INVALID_AST_ID {
                let name_node = ctx.ast.get(name_id);
                let args_node = ctx.ast.get(args_id);

                if !name_node.is_partial {
                    let mut call = ChatToolCall::default();
                    call.name = ctx.ast.node_text(name_id, &ctx.input).to_string();
                    if let Some(&first) = args_node.children.first() {
                        call.arguments = gemma4_to_json(ctx, first);
                    }
                    self.result.tool_calls.push(call);
                }
            }

            return;
        }

        for &child_id in &node.children {
            self.visit(ctx, child_id);
        }
    }
}

/// `gemma4_to_json` (chat-peg-parser.cpp:962-1055): rebuild JSON text from the
/// gemma4-* rule subtree.
fn gemma4_to_json(ctx: &ParseContext, id: usize) -> String {
    let node = ctx.ast.get(id);
    let text = ctx.ast.node_text(id, &ctx.input);

    if text.is_empty() {
        return String::new();
    }

    if node.rule == "gemma4-number" || node.rule == "gemma4-bool" || node.rule == "gemma4-null" {
        return text.to_string();
    }

    if node.rule == "gemma4-string-content" {
        return escape_json_string_inner(text);
    }

    if node.rule == "gemma4-string" {
        let mut result = String::from("\"");
        if let Some(&first) = node.children.first() {
            result.push_str(&gemma4_to_json(ctx, first));
            if !node.is_partial {
                result.push('"');
            }
        }
        return result;
    }

    if node.rule == "gemma4-array" {
        let mut result = String::from("[");
        let mut add_comma = false;
        for &child_id in &node.children {
            if add_comma {
                result.push(',');
            }
            add_comma = true;
            result.push_str(&gemma4_to_json(ctx, child_id));
        }
        if !node.is_partial {
            result.push(']');
        }
        return result;
    }

    if node.rule == "gemma4-dict-key-name" {
        return text.to_string();
    }

    if node.rule == "gemma4-dict-key" {
        let mut result = String::from("\"");
        if let Some(&first) = node.children.first() {
            result.push_str(&escape_json_string_inner(&gemma4_to_json(ctx, first)));
        }
        if !node.is_partial {
            result.push_str("\":");
        }
        return result;
    }

    if node.rule == "gemma4-dict-kv" {
        let mut result = String::new();
        for &child_id in &node.children {
            result.push_str(&gemma4_to_json(ctx, child_id));
        }
        return result;
    }

    if node.rule == "gemma4-dict" {
        let mut result = String::from("{");
        let mut add_comma = false;
        for &child_id in &node.children {
            if add_comma {
                result.push(',');
            }
            add_comma = true;
            result.push_str(&gemma4_to_json(ctx, child_id));
        }
        if !node.is_partial {
            result.push('}');
        }
        return result;
    }

    if node.rule == "gemma4-value" {
        if let Some(&first) = node.children.first() {
            return gemma4_to_json(ctx, first);
        }
        return String::new();
    }

    String::new()
}

/// `minimax_m3_collect` (chat-peg-parser.cpp:1096-1108)
fn minimax_m3_collect(ctx: &ParseContext, node: &AstNode, tag: &str, out: &mut Vec<usize>) {
    for &child_id in &node.children {
        let child = ctx.ast.get(child_id);
        if child.tag == tag {
            out.push(child_id);
        } else {
            minimax_m3_collect(ctx, child, tag, out);
        }
    }
}

/// `minimax_m3_value_of` (chat-peg-parser.cpp:1110-1121)
fn minimax_m3_value_of(ctx: &ParseContext, node: &AstNode) -> usize {
    for &child_id in &node.children {
        let tag = ctx.ast.get(child_id).tag.as_str();
        if tag == chat_tag::TOOL_ARG_VALUE
            || tag == chat_tag::TOOL_ARG_STRING_VALUE
            || tag == mm3_tag::TOOL_ARG_OBJECT
            || tag == mm3_tag::TOOL_ARG_ARRAY
        {
            return child_id;
        }
    }
    INVALID_AST_ID
}

/// `minimax_m3_value_to_json` (chat-peg-parser.cpp:1175-1196)
fn minimax_m3_value_to_json(ctx: &ParseContext, id: usize, closed: bool) -> String {
    if id == INVALID_AST_ID {
        return String::new();
    }

    let node = ctx.ast.get(id);

    if node.tag == mm3_tag::TOOL_ARG_OBJECT {
        return minimax_m3_container_to_json(ctx, node, /* is_object = */ true, closed);
    }

    if node.tag == mm3_tag::TOOL_ARG_ARRAY {
        return minimax_m3_container_to_json(ctx, node, /* is_object = */ false, closed);
    }

    if node.tag == chat_tag::TOOL_ARG_STRING_VALUE {
        let text = ctx.ast.node_text(id, &ctx.input);
        return format!(
            "\"{}{}",
            escape_json_string_inner(text),
            if closed { "\"" } else { "" }
        );
    }

    // Numbers and booleans are written verbatim by the template
    ctx.ast.node_text(id, &ctx.input).to_string()
}

/// `minimax_m3_member_to_json` (chat-peg-parser.cpp:1125-1133)
fn minimax_m3_member_to_json(ctx: &ParseContext, node: &AstNode) -> String {
    let name_id = ctx.ast.find_by_tag(node, chat_tag::TOOL_ARG_NAME, 3);
    if name_id == INVALID_AST_ID {
        return String::new();
    }

    let name = ctx.ast.node_text(name_id, &ctx.input);
    format!(
        "{}:{}",
        Json::String(name.to_string()).dump(),
        minimax_m3_value_to_json(ctx, minimax_m3_value_of(ctx, node), !node.is_partial)
    )
}

/// `minimax_m3_container_to_json` (chat-peg-parser.cpp:1135-1173)
fn minimax_m3_container_to_json(
    ctx: &ParseContext,
    node: &AstNode,
    is_object: bool,
    closed: bool,
) -> String {
    let tag = if is_object {
        chat_tag::TOOL_ARG
    } else {
        mm3_tag::TOOL_ARG_ITEM
    };

    let mut entries: Vec<usize> = Vec::new();
    minimax_m3_collect(ctx, node, tag, &mut entries);

    let mut result = if is_object {
        String::from("{")
    } else {
        String::from("[")
    };
    let mut add_comma = false;
    for entry_id in entries {
        let entry = ctx.ast.get(entry_id);
        let text = if is_object {
            minimax_m3_member_to_json(ctx, entry)
        } else {
            minimax_m3_value_to_json(ctx, minimax_m3_value_of(ctx, entry), !entry.is_partial)
        };

        if text.is_empty() {
            continue;
        }

        if add_comma {
            result.push(',');
        }
        add_comma = true;
        result.push_str(&text);
    }

    if closed {
        result.push(if is_object { '}' } else { ']' });
    }
    result
}

/// `common_chat_peg_minimax_m3_mapper` (chat-peg-parser.h:43-53)
pub(crate) struct ChatPegMinimaxM3Mapper<'a> {
    pub result: &'a mut ChatMsg,
}

impl<'a> ChatPegMinimaxM3Mapper<'a> {
    pub fn new(result: &'a mut ChatMsg) -> Self {
        ChatPegMinimaxM3Mapper { result }
    }

    /// `from_ast` (chat-peg-parser.cpp:1198-1203)
    pub fn from_ast(&mut self, ctx: &ParseContext, result: &ParseResult) {
        for &node_id in &result.nodes {
            self.visit(ctx, node_id);
        }
    }

    /// `visit` (chat-peg-parser.cpp:1205-1232)
    fn visit(&mut self, ctx: &ParseContext, id: usize) {
        let node = ctx.ast.get(id);

        if node.tag == chat_tag::REASONING {
            self.result.reasoning_content += &node.sanitized_text(ctx.input.as_bytes());
            return;
        }

        if node.tag == chat_tag::CONTENT {
            self.result.content += &node.sanitized_text(ctx.input.as_bytes());
            return;
        }

        if node.tag == chat_tag::TOOL {
            let name_id = ctx.ast.find_by_tag(node, chat_tag::TOOL_NAME, 3);
            if name_id != INVALID_AST_ID {
                let mut call = ChatToolCall::default();
                call.name = ctx.ast.node_text(name_id, &ctx.input).to_string();
                call.arguments = minimax_m3_container_to_json(
                    ctx,
                    node,
                    /* is_object = */ true,
                    !node.is_partial,
                );
                self.result.tool_calls.push(call);
            }
            return;
        }

        for &child_id in &node.children {
            self.visit(ctx, child_id);
        }
    }
}
