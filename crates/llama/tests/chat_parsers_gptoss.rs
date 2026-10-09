//! chat_parsers_gptoss.rs — gpt-oss end-to-end against the pinned reference
//! llama-server run on the local gpt-oss-20b (parity/gptoss_server_check.sh).
//!
//! The captured artifacts:
//!   * response.json      — the reference's own parsed chat answer
//!                          (reasoning_content / content / tool_calls split)
//!   * raw-completion.json — the raw bytes the model produced for the same
//!                          rendered prompt (a /completion replay)
//!   * prompt.txt / gen-prompt.txt — the reference's rendered prompt and the
//!                          generation prompt suffix
//!
//! Here the port's gpt-oss handler (chat_parsers.rs) builds the parser from a
//! detection-needle template (the port's mini-jinja cannot compile the real
//! harmony template — the documented jinja gap) with the same tools, parses
//! the raw bytes with the reference's generation prompt, and the split must
//! equal the reference's parsed fields (name + arguments; the call id is
//! generated server-side, not by the model, so it is only checked to exist).

use llama::chat_tools::*;
use llama::json_schema::Json;

const ARTIFACTS: &str = "/tmp/gptoss-srv";

/// the synthetic needle template from parity/chat_tools_cases.json
const SYNTH_GPT_OSS: &str = "{%- for m in messages %}<|start|>{{ m.role }}<|channel|>final<|message|>{{ m.content }}<|return|><|end|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant<|channel|>analysis<|message|>{% endif %}";

fn read(name: &str) -> Option<String> {
    std::fs::read_to_string(format!("{ARTIFACTS}/{name}")).ok()
}

#[test]
fn gpt_oss_server_split_matches_reference() {
    let Some(response) = read("response.json") else {
        eprintln!("gpt-oss server artifacts missing (run parity/gptoss_server_check.sh); skipping");
        return;
    };
    let response: Json = Json::parse(&response).expect("response json");
    let raw = read("raw-completion.json").expect("raw completion");
    let raw: Json = Json::parse(&raw).expect("raw json");
    let gen_prompt = read("gen-prompt.txt").expect("gen prompt");

    let raw_text = raw
        .at("content")
        .and_then(|v| v.get_str().ok())
        .expect("raw content")
        .to_string();
    assert!(!raw_text.is_empty(), "empty raw capture");

    let msg_ref = response
        .at("choices")
        .and_then(|c| c.at_idx(0))
        .and_then(|c| c.at("message"))
        .expect("reference message");

    // the same request the server served, through the port's gpt-oss handler
    let request: Json = Json::parse(&read("request.json").unwrap()).unwrap();
    let tools = tools_parse_oaicompat(request.at("tools").unwrap()).unwrap();
    let messages = msgs_parse_oaicompat(request.at("messages").unwrap()).unwrap();

    let tmpls = ChatTemplates::init(&ChatTemplatesInit {
        chat_template_override: SYNTH_GPT_OSS.to_string(),
        chat_template_tool_use: String::new(),
        bos_token: String::new(),
        eos_token: String::new(),
        add_bos: false,
        add_eos: false,
    })
    .unwrap();

    let params = chat_templates_apply(
        &tmpls,
        &TemplatesInputs {
            messages,
            tools,
            tool_choice: ChatToolChoice::Auto,
            add_generation_prompt: true,
            parallel_tool_calls: false,
            reasoning_format: reasoning_format_from_name("deepseek").unwrap(),
            ..TemplatesInputs::default()
        },
    )
    .unwrap();

    // the specialized handler must have engaged (the needle template contains
    // <|channel|>)
    assert_eq!(
        chat_format_name(params.format).unwrap_or_default(),
        "peg-native"
    );
    assert!(params.generation_prompt.contains("assistant"));

    let mut pparams = ChatParserParams::from_chat_params(&params).unwrap();
    pparams.generation_prompt = ChatInput::from_plain(gen_prompt);
    let msg = chat_parse(&llama::chat_tools::ChatInput::from(raw_text.as_str()), false, &pparams)
        .unwrap_or_else(|e| panic!("port parse failed: {e}\nraw: {raw_text:?}"));

    // the reasoning/final split the reference reported
    let exp_reasoning = msg_ref
        .at("reasoning_content")
        .and_then(|v| v.get_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        msg.reasoning_content, exp_reasoning,
        "reasoning split differs"
    );
    let exp_content = msg_ref
        .at("content")
        .and_then(|v| v.get_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(msg.content, exp_content, "final content differs");

    // tool calls: name + arguments byte-equal; the id is server-generated
    let ref_calls = msg_ref.at("tool_calls").cloned().unwrap_or(Json::Null);
    if ref_calls.is_array() && !ref_calls.empty() {
        assert_eq!(
            msg.tool_calls.len(),
            ref_calls.size(),
            "tool call count differs"
        );
        for (i, call) in msg.tool_calls.iter().enumerate() {
            let rc = ref_calls.at_idx(i).unwrap();
            let rf = rc.at("function").unwrap();
            assert_eq!(
                call.name,
                rf.at("name").and_then(|v| v.get_str().ok()).unwrap_or(""),
                "tool name differs"
            );
            assert_eq!(
                call.arguments,
                rf.at("arguments")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or(""),
                "tool arguments differ"
            );
        }
    } else {
        assert!(
            msg.tool_calls.is_empty(),
            "port found tool calls where the reference found none"
        );
    }
}
