//! chat_template_e2e.rs — end-to-end template-render comparison against the
//! pinned reference llama-server on REAL local models whose GGUFs carry the
//! vendor templates that were blocked on the old mini-jinja (`{% macro %}`,
//! block `{% set %}`, `namespace()`, slices, kwargs, `selectattr`):
//!
//!   * gpt-oss-20b  (gpt-oss template  — macros/namespace/slices; the model
//!     file's template is a LATER revision than the parity fixture, so this
//!     pins constructs the fixture does not even contain)
//!   * gemma-4-12B  (gemma4 template   — macros/dictsort/namespace)
//!   * Qwen3.5-9B   (qwen35 template   — `messages[::-1]` / render_content)
//!
//! Capture (reference side): `bash parity/chat_template_ref_render.sh gptoss
//! gemma4 qwen35` starts the reference server per model, dumps `/props`'
//! `chat_template` and `/apply-template` outputs for three probe bodies into
//! /tmp/mj-e2e. The port's llama-server cannot load these architectures yet
//! (FILE_MAP.md's architecture matrix), so the port side renders the very
//! same template string through `llama::chat_tools` — the exact code path
//! the server uses — and the prompts are compared byte-for-byte modulo the
//! wall-clock dates (`strftime_now`/`datetime`/`date_string` render the
//! server's clock, the documented chat.rs deviation).
//!
//! The test skips (returns) when the capture directory is absent, same
//! convention as tests/chat_parsers_gptoss.rs.

use llama::chat_tools::*;
use llama::json_schema::Json;

const ARTIFACTS: &str = "/tmp/mj-e2e";

const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]"#;

/// the three probe bodies of parity/chat_template_ref_render.sh, in order
const BODIES: [&str; 3] = [
    r#"{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"What is the weather in Tokyo?"}]}"#,
    r#"{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"What is the weather in Tokyo? Use the tool."}],"tools":TOOLS_JSON}"#,
    r#"{"messages":[{"role":"user","content":"Weather in Tokyo?"},{"role":"assistant","content":"","tool_calls":[{"id":"call1","type":"function","function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},{"role":"tool","content":"15C sunny","tool_call_id":"call1"},{"role":"user","content":"thanks"}],"tools":TOOLS_JSON}"#,
];

/// wall-clock dates in both formats the templates emit (see
/// tests/chat_tools_parity.rs::normalize_dates for the rationale)
fn normalize_dates(s: &str) -> String {
    let re_date = |s: &str| -> String {
        let b = s.as_bytes();
        let mut out = String::with_capacity(s.len());
        let mut i = 0usize;
        while i < b.len() {
            let rest = &s[i..];
            if rest.len() >= 10
                && rest.as_bytes()[..4].iter().all(|c| c.is_ascii_digit())
                && rest.as_bytes()[4] == b'-'
                && rest.as_bytes()[5..7].iter().all(|c| c.is_ascii_digit())
                && rest.as_bytes()[7] == b'-'
                && rest.as_bytes()[8..10].iter().all(|c| c.is_ascii_digit())
            {
                out.push_str("YYYY-MM-DD");
                i += 10;
                continue;
            }
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
        out
    };
    re_date(s)
}

fn render_port(template: &str, body: &str) -> Result<ChatParams, String> {
    let body = body.replace("TOOLS_JSON", TOOLS);
    let body = Json::parse(&body).map_err(|e| format!("body: {e}"))?;
    let tmpls = ChatTemplates::init(&ChatTemplatesInit {
        chat_template_override: template.to_string(),
        chat_template_tool_use: String::new(),
        bos_token: String::new(),
        eos_token: String::new(),
        add_bos: false,
        add_eos: false,
    })?;
    let inputs = TemplatesInputs {
        messages: msgs_parse_oaicompat(body.at("messages").unwrap())?,
        tools: tools_parse_oaicompat(body.at("tools").unwrap_or(&Json::Null))?,
        tool_choice: tool_choice_parse_oaicompat(
            body.at("tool_choice")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("auto"),
        )?,
        add_generation_prompt: true,
        parallel_tool_calls: true,
        reasoning_format: reasoning_format_from_name("none")?,
        json_schema: String::new(),
        enable_thinking: true,
        now: None,
        ..TemplatesInputs::default()
    };
    chat_templates_apply(&tmpls, &inputs)
}

#[test]
fn e2e_vendor_templates_match_reference_server() {
    let cases = [
        ("gptoss", "gpt-oss-20b"),
        ("gemma4", "gemma-4-12B"),
        ("qwen35", "Qwen3.5-9B"),
    ];
    let mut checked = 0usize;
    for (tag, desc) in cases {
        let Ok(template) = std::fs::read_to_string(format!("{ARTIFACTS}/{tag}.template.jinja"))
        else {
            eprintln!("{tag}: no capture (run parity/chat_template_ref_render.sh), skipping");
            continue;
        };
        for (i, body) in BODIES.iter().enumerate() {
            let ref_path = format!("{ARTIFACTS}/{tag}-{}-ref.json", i + 1);
            let Ok(ref_json) = std::fs::read_to_string(&ref_path) else {
                eprintln!("{tag}: missing {ref_path}, skipping");
                continue;
            };
            let expected: String = Json::parse(&ref_json)
                .expect("ref json")
                .at("prompt")
                .and_then(|v| v.get_str().ok())
                .expect("prompt")
                .to_string();
            let got = render_port(&template, body)
                .unwrap_or_else(|e| panic!("{tag} ({desc}) body {}: apply: {e}", i + 1));
            assert_eq!(
                normalize_dates(&got.prompt),
                normalize_dates(&expected),
                "{tag} ({desc}) body {} prompt differs",
                i + 1,
            );
            checked += 1;
        }
    }
    if checked == 0 {
        // /tmp is wiped across reboots on this box; the captures are cheap to
        // regenerate but need the reference server + the three real models.
        // Self-heal once; if regeneration is impossible (no REF binary), skip
        // with a loud note instead of failing a template-engine change.
        let rc = std::process::Command::new("bash")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/chat_template_ref_render.sh"))
            .args(["gptoss", "gemma4", "qwen35"])
            .status();
        if rc.map(|s| s.success()).unwrap_or(false) {
            panic!("captures were missing under {ARTIFACTS}; regenerated them — re-run the test");
        }
        eprintln!("SKIP: no captures under {ARTIFACTS} and regeneration failed (reference unavailable)");
        return;
    }
    assert!(checked >= 3, "no captures found under {ARTIFACTS}");
    eprintln!(
        "chat_template_e2e: {checked} real-model prompts byte-identical to the reference server"
    );
}
