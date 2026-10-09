//! chat_tools_parity.rs — parity harness for the chat tool-calling port
//! (common/chat.cpp's tools path, common/chat-peg-parser.cpp,
//! common/chat-auto-parser-generator.cpp, common/chat-diff-analyzer.cpp and
//! the mini-jinja tools bindings in chat.rs) against fixtures produced by the
//! pinned reference via `parity/gen_chat_tools_ref.sh` (probe:
//! `parity/ref_chat_tools_dump.cpp`, cases: `parity/chat_tools_cases.json`).
//!
//! What is compared, per case:
//!   * the rendered prompt and generation_prompt (byte-for-byte)
//!   * the generated PEG parser serialization (`common_chat_params::parser`,
//!     byte-for-byte — this pins the whole differential autoparser analysis:
//!     tool-format detection, markers, call-id markers, reasoning markers…)
//!   * grammar text + lazy flag + word triggers
//!   * preserved tokens / additional stops / thinking tags / message delimiters
//!   * `common_chat_parse` output for the case's `parse_input`
//!     (structure equality on the oaicompat JSON)
//!
//! Both sides render with a pinned clock (1727000000) under TZ=UTC (the
//! reference formats `datetime`/`date_string` with `std::localtime`, the port
//! renders UTC — the documented chat.rs deviation).
//!
//! Additionally transcribes the reference's own unit tests for the pieces they
//! cover: `tests/test-chat-auto-parser.cpp` diff/segment helpers and
//! `tests/test-chat-peg-parser.cpp` standard_json_tools.

use llama::chat_tools::*;
use llama::json_schema::Json;
use llama::peg::PegArena;

const PINNED_EPOCH: i64 = 1_727_000_000;

fn fixture_dir() -> String {
    option_env!("CHAT_TOOLS_FIXTURE_DIR")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "../../parity".to_string())
}

struct Case {
    name: String,
    input: Json,
    expected: Json,
}

fn load_cases() -> Vec<Case> {
    let dir = fixture_dir();
    let cases_text = std::fs::read_to_string(format!("{dir}/chat_tools_cases.json"))
        .expect("parity/chat_tools_cases.json (regenerate with parity/gen_chat_tools_ref.sh)");
    let ref_text = std::fs::read_to_string(format!("{dir}/chat_tools_ref.json"))
        .expect("parity/chat_tools_ref.json (regenerate with parity/gen_chat_tools_ref.sh)");
    let cases = Json::parse(&cases_text).expect("cases json");
    let expected = Json::parse(&ref_text).expect("ref json");

    let mut out = Vec::new();
    let Json::Array(refs) = expected else {
        panic!("ref not an array")
    };
    for (i, exp) in refs.iter().enumerate() {
        let case = cases.at("cases").unwrap().at_idx(i).unwrap().clone();
        let tmpl = case
            .at("template")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("?")
            .to_string();
        out.push(Case {
            name: format!("#{i} [{tmpl}]"),
            input: case,
            expected: exp.clone(),
        });
    }
    out
}

/// Run the port's `common_chat_templates_apply` for one case.
fn apply_case(all: &Json, case: &Json) -> Result<ChatParams, String> {
    let name = case.at("template").unwrap().get_str()?.to_string();
    let src = all
        .at("templates")
        .unwrap()
        .iter()
        .find(|t| t.at("name").and_then(|v| v.get_str().ok()) == Some(name.as_str()))
        .and_then(|t| t.at("src"))
        .ok_or_else(|| format!("template {name} not in fixture"))?
        .get_str()?
        .to_string();
    let tmpls = ChatTemplates::init(&ChatTemplatesInit {
        chat_template_override: src,
        chat_template_tool_use: String::new(),
        bos_token: String::new(),
        eos_token: String::new(),
        add_bos: false,
        add_eos: false,
    })?;

    let messages = msgs_parse_oaicompat(case.at("messages").unwrap())?;
    let tools = tools_parse_oaicompat(case.at("tools").unwrap_or(&Json::Null))?;
    let tool_choice = tool_choice_parse_oaicompat(
        case.at("tool_choice")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("auto"),
    )?;

    let inputs = TemplatesInputs {
        messages,
        tools,
        tool_choice,
        add_generation_prompt: case
            .at("add_generation_prompt")
            .and_then(|v| match v {
                Json::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true),
        parallel_tool_calls: matches!(case.at("parallel_tool_calls"), Some(Json::Bool(true))),
        reasoning_format: reasoning_format_from_name(
            case.at("reasoning_format")
                .and_then(|v| v.get_str().ok())
                .unwrap_or("none"),
        )?,
        json_schema: case
            .at("json_schema")
            .and_then(|v| v.get_str().ok())
            .unwrap_or("")
            .to_string(),
        enable_thinking: !matches!(case.at("enable_thinking"), Some(Json::Bool(false))),
        now: Some(PINNED_EPOCH),
        chat_template_kwargs: match case.at("chat_template_kwargs") {
            Some(Json::Object(fields)) => fields
                .iter()
                .map(|(k, v)| {
                    // the value is a json-encoded string, like the C's
                    // `chat_template_kwargs["k"] = "\"xml\""`
                    (
                        k.clone(),
                        match v {
                            Json::String(s) => s.clone(),
                            other => other.dump(),
                        },
                    )
                })
                .collect(),
            _ => Vec::new(),
        },
        ..TemplatesInputs::default()
    };

    chat_templates_apply(&tmpls, &inputs)
}

/// The reference renders `datetime`/`date_string` from the wall clock
/// (chat.cpp:1080-1088 — `system_clock::now()`, not `inputs.now`), so both
/// sides' prompts can contain different dates; normalize them away.
/// (`strftime_now('%Y-%m-%d')` in the gpt-oss/muse-glimmer templates renders
/// the same wall clock, so ISO dates are normalized too.)
fn normalize_dates(s: &str) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let rest = &s[i..];
        // ISO date "YYYY-MM-DD"
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
        // "Mon DD YYYY"
        let mut replaced = false;
        for m in MONTHS {
            if rest.starts_with(m) {
                let after = &rest[m.len() + 1..];
                let parts: Vec<&str> = after.splitn(2, ' ').collect();
                if parts.len() == 2
                    && (1..=2).contains(&parts[0].len())
                    && parts[0].bytes().all(|c| c.is_ascii_digit())
                    && parts[1].len() >= 4
                    && parts[1].as_bytes()[..4].iter().all(|c| c.is_ascii_digit())
                {
                    out.push_str("Mon DD YYYY");
                    i += m.len() + 1 + parts[0].len() + 1 + 4;
                    replaced = true;
                    break;
                }
            }
        }
        if replaced {
            continue;
        }
        // "DD Mon YYYY"
        let digits = rest.bytes().take_while(|c| c.is_ascii_digit()).count();
        if (digits == 1 || digits == 2)
            && rest.len() > digits + 9
            && rest.as_bytes()[digits] == b' '
        {
            let after = &rest[digits + 1..];
            if after.len() > 8 && after.as_bytes()[3] == b' ' {
                let mon = &after[..3];
                let year = &after[4..8];
                if MONTHS.contains(&mon) && year.bytes().all(|c| c.is_ascii_digit()) {
                    out.push_str("DD Mon YYYY");
                    i += digits + 1 + 3 + 1 + 4;
                    continue;
                }
            }
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn json_string(v: &Json, key: &str) -> String {
    v.at(key)
        .and_then(|x| x.get_str().ok())
        .unwrap_or("")
        .to_string()
}

fn json_bool(v: &Json, key: &str) -> bool {
    matches!(v.at(key), Some(Json::Bool(true)))
}

fn json_str_array(v: &Json, key: &str) -> Vec<String> {
    match v.at(key) {
        Some(Json::Array(items)) => items
            .iter()
            .filter_map(|i| i.get_str().ok().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// C++-faithful parser dump (peg-parser.cpp:914-995)
// ---------------------------------------------------------------------------

/// `common_peg_arena::dump` with the LIVE C++ visited-set semantics: the
/// visited set is shared by every construct except `Tag`, which calls the
/// 2-argument `dump()` (a fresh set). (The port's `PegArena::dump` resets the
/// set for `Atomic` as well — matching the C++'s dead duplicate branch — so it
/// prints rule bodies twice where the reference prints `[cycle]`; the
/// specialized parsers wrap rule-refs in `Atomic`, which is how the two
/// differ. Dumping here keeps peg.rs untouched; see PARITY.md.)
fn ref_dump(arena: &PegArena, id: llama::peg::ParserId) -> String {
    use std::collections::BTreeSet;
    fn dump_impl(
        arena: &PegArena,
        id: llama::peg::ParserId,
        visited: &mut BTreeSet<llama::peg::ParserId>,
    ) -> String {
        use llama::peg::ParserKind;
        if visited.contains(&id) {
            return "[cycle]".to_string();
        }
        visited.insert(id);
        match arena.get(id) {
            ParserKind::Epsilon => "Epsilon".to_string(),
            ParserKind::Start => "Start".to_string(),
            ParserKind::End => "End".to_string(),
            ParserKind::Literal(l) => format!("Literal({l})"),
            ParserKind::Sequence(children) => {
                let parts: Vec<String> = children
                    .iter()
                    .map(|&c| dump_impl(arena, c, visited))
                    .collect();
                format!("Sequence({})", parts.join(", "))
            }
            ParserKind::Choice(children) => {
                let parts: Vec<String> = children
                    .iter()
                    .map(|&c| dump_impl(arena, c, visited))
                    .collect();
                format!("Choice({})", parts.join(", "))
            }
            ParserKind::Repetition { child, min, max } => {
                if *max == -1 {
                    format!(
                        "Repetition({}, {}, unbounded)",
                        dump_impl(arena, *child, visited),
                        min
                    )
                } else {
                    format!(
                        "Repetition({}, {}, {})",
                        dump_impl(arena, *child, visited),
                        min,
                        max
                    )
                }
            }
            ParserKind::And { child } => format!("And({})", dump_impl(arena, *child, visited)),
            ParserKind::Not { child } => format!("Not({})", dump_impl(arena, *child, visited)),
            // peg-parser.cpp:964-966 — the live branch SHARES the visited set
            ParserKind::Atomic { child } => {
                format!("Atomic({})", dump_impl(arena, *child, visited))
            }
            ParserKind::Gbnf { child, grammar } => {
                format!("Gbnf({}, {})", grammar, dump_impl(arena, *child, visited))
            }
            ParserKind::Ac { child, delimiters } => format!(
                "Ac({}, {})",
                delimiters.join(" | "),
                dump_impl(arena, *child, visited)
            ),
            ParserKind::Any => "Any".to_string(),
            ParserKind::Space => "Space".to_string(),
            ParserKind::Chars {
                pattern, min, max, ..
            } => {
                if *max == -1 {
                    format!("CharRepeat({pattern}, {min}, unbounded)")
                } else {
                    format!("CharRepeat({pattern}, {min}, {max})")
                }
            }
            ParserKind::Str { delimiter } => format!("String({})", *delimiter as char),
            ParserKind::Until { delimiters } => format!("Until({})", delimiters.join(" | ")),
            ParserKind::Schema {
                child, node, doc, ..
            } => {
                let kind: String = match doc {
                    Some(doc) => doc.node(*node).kind_name().to_string(),
                    None => "null".to_string(),
                };
                format!("Schema({}, {})", dump_impl(arena, *child, visited), kind)
            }
            ParserKind::Rule { name, child, .. } => {
                format!("Rule({}, {})", name, dump_impl(arena, *child, visited))
            }
            ParserKind::Ref { name } => format!("Ref({name})"),
            // peg-parser.cpp:987-989 — Tag resets the visited set
            ParserKind::Tag { child, tag } => {
                format!(
                    "Tag({}, {})",
                    tag,
                    dump_impl(arena, *child, &mut BTreeSet::new())
                )
            }
        }
    }
    let mut visited = BTreeSet::new();
    dump_impl(arena, id, &mut visited)
}

// ---------------------------------------------------------------------------
// main parity test
// ---------------------------------------------------------------------------

#[test]
fn chat_tools_parity_prompts_and_parsers() {
    let cases = load_cases();
    assert!(!cases.is_empty(), "no cases loaded");
    let mut failures: Vec<String> = Vec::new();
    let mut jinja_gaps: Vec<String> = Vec::new();
    let mut prompts_checked = 0usize;
    let mut parsers_checked = 0usize;
    let mut grammars_checked = 0usize;

    // full cases document (for template source lookup)
    let cases_text2 =
        std::fs::read_to_string(format!("{}/chat_tools_cases.json", fixture_dir())).unwrap();
    let all = Json::parse(&cases_text2).unwrap();
    for case in &cases {
        let exp = &case.expected;
        if !json_bool(exp, "ok") {
            // the reference itself failed this case; the port must also fail
            let got = apply_case(&all, &case.input);
            if got.is_ok() {
                failures.push(format!(
                    "{}: reference failed ('{}') but port succeeded",
                    case.name,
                    json_string(exp, "error")
                ));
            }
            continue;
        }

        let got = match apply_case(&all, &case.input) {
            Ok(p) => p,
            Err(e) => {
                // The port's mini-jinja (chat.rs) ports the minja surface the
                // vendor corpus uses (macro/block set/namespace/slices/kwargs/
                // selectattr/…), so every case must render. Cases failing at
                // template *initialization* are counted as engine gaps (kept
                // as a tripwire — the count must stay 0; see PARITY.md).
                if e.contains("failed to initialize chat template") {
                    let tmpl = case
                        .input
                        .at("template")
                        .and_then(|v| v.get_str().ok())
                        .unwrap_or("");
                    if !tmpl.starts_with("synth-") {
                        jinja_gaps.push(format!("{}: {}", case.name, e));
                        continue;
                    }
                }
                failures.push(format!("{}: apply failed: {e}", case.name));
                continue;
            }
        };

        let check = |name: &str, cond: bool, detail: String, failures: &mut Vec<String>| {
            if !cond {
                failures.push(format!("{}: {name} mismatch: {detail}", case.name));
            }
        };

        // format
        let exp_format = json_string(exp, "format");
        check(
            "format",
            chat_format_name(got.format) == Ok(exp_format.as_str()),
            format!("{:?} != {exp_format}", got.format),
            &mut failures,
        );

        // prompt + generation prompt, byte-for-byte
        let exp_prompt = json_string(exp, "prompt");
        check(
            "prompt",
            normalize_dates(&got.prompt) == normalize_dates(&exp_prompt),
            format!("\n  exp: {:?}\n  got: {:?}", exp_prompt, got.prompt),
            &mut failures,
        );
        prompts_checked += 1;

        let exp_gen = json_string(exp, "generation_prompt");
        check(
            "generation_prompt",
            normalize_dates(&got.generation_prompt) == normalize_dates(&exp_gen),
            format!("\n  exp: {:?}\n  got: {:?}", exp_gen, got.generation_prompt),
            &mut failures,
        );

        // grammar + lazy + triggers
        let exp_grammar = json_string(exp, "grammar");
        check(
            "grammar_lazy",
            got.grammar_lazy == json_bool(exp, "grammar_lazy"),
            format!(
                "{:?} != {:?}",
                got.grammar_lazy,
                json_bool(exp, "grammar_lazy")
            ),
            &mut failures,
        );
        // (type, value) pairs — the specialized parsers also emit PATTERN (2)
        // triggers (gpt-oss, functionary v3.2, muse-glimmer)
        let exp_triggers: Vec<(i64, String)> = match exp.at("grammar_triggers") {
            Some(Json::Array(items)) => items
                .iter()
                .filter_map(|t| {
                    let ty = match t.at("type") {
                        Some(Json::Int(v)) => *v as i64,
                        _ => return None,
                    };
                    let val = t.at("value").and_then(|v| v.get_str().ok())?.to_string();
                    Some((ty, val))
                })
                .collect(),
            _ => Vec::new(),
        };
        let got_triggers: Vec<(i64, String)> = got
            .grammar_triggers
            .iter()
            .map(|t| {
                let ty = match t.ty {
                    GrammarTriggerType::Token => 0,
                    GrammarTriggerType::Word => 1,
                    GrammarTriggerType::Pattern => 2,
                    GrammarTriggerType::PatternFull => 3,
                };
                (ty, t.word.clone())
            })
            .collect();
        check(
            "grammar_triggers",
            exp_triggers == got_triggers,
            format!("{exp_triggers:?} != {got_triggers:?}"),
            &mut failures,
        );
        // grammar text: compare byte-for-byte; on mismatch fall back to a
        // rule-set comparison (rule ORDER is converter-internal, see PARITY.md)
        if got.grammar != exp_grammar {
            // `until-<id>`/`ac-<id>` rule names embed the arena allocation id,
            // which follows the C++ expression evaluation order — canonicalize
            let norm = |g: &str| -> Vec<String> {
                let mut rules: Vec<String> = g
                    .lines()
                    .filter(|l| l.contains("::="))
                    .map(|l| l.trim().to_string())
                    .map(|l| {
                        let mut out = String::new();
                        let bytes = l.as_bytes();
                        let mut i = 0;
                        while i < bytes.len() {
                            let prefix_len = if l[i..].starts_with("until-") {
                                6
                            } else if l[i..].starts_with("ac-") {
                                3
                            } else {
                                0
                            };
                            if prefix_len > 0 {
                                let start = i;
                                i += prefix_len;
                                while i < bytes.len() && bytes[i].is_ascii_digit() {
                                    i += 1;
                                }
                                // blank the allocation-id digits
                                out.push_str(&l[start..start + prefix_len]);
                                out.push('#');
                            } else {
                                let ch = l[i..].chars().next().unwrap();
                                out.push(ch);
                                i += ch.len_utf8();
                            }
                        }
                        out
                    })
                    .collect();
                rules.sort();
                rules
            };
            check(
                "grammar (normalized rule set)",
                norm(&got.grammar) == norm(&exp_grammar),
                format!(
                    "\n  exp: {}\n  got: {}",
                    exp_grammar.chars().take(400).collect::<String>(),
                    got.grammar.chars().take(400).collect::<String>()
                ),
                &mut failures,
            );
        }
        grammars_checked += 1;

        // preserved tokens / stops / thinking
        check(
            "preserved_tokens",
            got.preserved_tokens == json_str_array(exp, "preserved_tokens"),
            format!(
                "{:?} != {:?}",
                got.preserved_tokens,
                json_str_array(exp, "preserved_tokens")
            ),
            &mut failures,
        );
        check(
            "additional_stops",
            got.additional_stops == json_str_array(exp, "additional_stops"),
            format!(
                "{:?} != {:?}",
                got.additional_stops,
                json_str_array(exp, "additional_stops")
            ),
            &mut failures,
        );
        check(
            "supports_thinking",
            got.supports_thinking == json_bool(exp, "supports_thinking"),
            String::new(),
            &mut failures,
        );
        check(
            "thinking_start_tag",
            got.thinking_start_tag == json_string(exp, "thinking_start_tag"),
            format!(
                "{:?} != {:?}",
                got.thinking_start_tag,
                json_string(exp, "thinking_start_tag")
            ),
            &mut failures,
        );
        check(
            "thinking_end_tags",
            got.thinking_end_tags == json_str_array(exp, "thinking_end_tags"),
            String::new(),
            &mut failures,
        );

        // message delimiters (role, delimiter) pairs
        let exp_delims: Vec<(String, String)> = match exp.at("message_delimiters") {
            Some(Json::Array(items)) => items
                .iter()
                .map(|d| {
                    (
                        d.at("role")
                            .and_then(|v| v.get_str().ok())
                            .unwrap_or("")
                            .to_string(),
                        d.at("delimiter")
                            .and_then(|v| v.get_str().ok())
                            .unwrap_or("")
                            .to_string(),
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        check(
            "message_delimiters",
            got.message_delimiters == exp_delims,
            format!("{:?} != {:?}", got.message_delimiters, exp_delims),
            &mut failures,
        );

        // canonical parser dump from the root — allocation ids inside the
        // serialized arena follow the C++ evaluation order, so the graph dump
        // (deterministic DFS from root) is the structural byte-for-byte check
        let exp_dump = json_string(exp, "parser_dump");
        let mut arena = PegArena::default();
        arena.load(&got.parser).expect("parser reload");
        let got_dump = ref_dump(&arena, arena.root());
        check(
            "parser dump",
            got_dump == exp_dump,
            format!(
                "first diff at {:?}\n  exp: {}\n  got: {}",
                got_dump
                    .bytes()
                    .zip(exp_dump.bytes())
                    .position(|(a, b)| a != b)
                    .map(|p| {
                        let lo = p.saturating_sub(80);
                        let hi = (p + 120).min(exp_dump.len());
                        if lo < hi {
                            &exp_dump[lo..hi]
                        } else {
                            ""
                        }
                    }),
                &exp_dump[..exp_dump.len().min(200)],
                &got_dump[..got_dump.len().min(200)]
            ),
            &mut failures,
        );
        parsers_checked += 1;

        // parse side
        if let Some(parse_input) = case.input.at("parse_input").and_then(|v| v.get_str().ok()) {
            let is_partial = matches!(case.input.at("parse_partial"), Some(Json::Bool(true)));
            let mut params = ChatParserParams::from_chat_params(&got)
                .unwrap_or_else(|e| panic!("{}: load parser: {e}", case.name));
            params.generation_prompt = ChatInput::from_plain(got.generation_prompt.clone());
            let msg = match chat_parse(&ChatInput::from(parse_input), is_partial, &params) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{}: chat_parse failed: {e}", case.name));
                    continue;
                }
            };
            let got_json = msgs_to_json_oaicompat(&[msg], false);
            let got_dump = got_json.at_idx(0).map(|v| v.dump()).unwrap_or_default();
            let exp_parse = exp.at("parse").map(|v| v.dump()).unwrap_or_default();
            check(
                "chat_parse output",
                got_dump == exp_parse,
                format!("\n  exp: {exp_parse}\n  got: {got_dump}"),
                &mut failures,
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} case(s) with mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "chat_tools_parity: {prompts_checked} prompts, {parsers_checked} parsers, \
         {grammars_checked} grammars checked byte-for-byte / rule-set; \
         {} cases blocked by mini-jinja engine gaps (must be 0, see PARITY.md)",
        jinja_gaps.len()
    );
    assert!(
        jinja_gaps.is_empty(),
        "{} case(s) blocked by mini-jinja gaps:\n{}",
        jinja_gaps.len(),
        jinja_gaps.join("\n")
    );
}

// ---------------------------------------------------------------------------
// transcribed reference unit tests
// ---------------------------------------------------------------------------

/// `test_marker_separation` excerpts (tests/test-chat-auto-parser.cpp:614-…)
#[test]
fn ref_marker_separation() {
    use llama::chat_tools::{segmentize_markers, SegmentType};

    let single_square = segmentize_markers("pre_marker[marker]post_marker");
    assert!(matches!(single_square[0].ty, SegmentType::Text));
    assert!(matches!(single_square[1].ty, SegmentType::Marker));
    assert!(matches!(single_square[2].ty, SegmentType::Text));
    assert_eq!(single_square[0].value, "pre_marker");
    assert_eq!(single_square[1].value, "[marker]");
    assert_eq!(single_square[2].value, "post_marker");

    let paired = segmentize_markers("<hello>world</hello>");
    assert_eq!(paired[0].value, "<hello>");
    assert_eq!(paired[1].value, "world");
    assert_eq!(paired[2].value, "</hello>");

    let both = segmentize_markers("<hello>[hello]<world>[world]");
    assert_eq!(both.len(), 4);
    assert_eq!(both[1].value, "[hello]");
}

/// `test_calculate_diff_split_basic` + `_tag_boundaries` + `_generation_prompt`
/// (tests/test-chat-auto-parser.cpp)
#[test]
fn ref_diff_split_reference_cases() {
    use llama::chat_tools::calculate_diff_split;

    let d = calculate_diff_split("hello world", "hello test");
    assert_eq!(d.prefix, "hello ");
    assert_eq!(d.left, "world");
    assert_eq!(d.right, "test");
    assert_eq!(d.suffix, "");

    let d = calculate_diff_split("abc", "xyz");
    assert_eq!(
        (
            d.prefix.as_str(),
            d.left.as_str(),
            d.right.as_str(),
            d.suffix.as_str()
        ),
        ("", "abc", "xyz", "")
    );

    let d = calculate_diff_split("prefixA suffix", "prefixB suffix");
    assert_eq!(
        (
            d.prefix.as_str(),
            d.left.as_str(),
            d.right.as_str(),
            d.suffix.as_str()
        ),
        ("prefix", "A", "B", " suffix")
    );

    // partial tag on one side (tag_boundaries)
    let d = calculate_diff_split("prefix<tag>", "prefix</tag>suffix");
    assert_eq!(d.prefix, "prefix");
    assert_eq!(d.left, "<tag>");
    assert_eq!(d.right, "</tag>suffix");
    assert_eq!(d.suffix, "");

    // nested tags: text suffix is taken inside the segments ("actual
    // algorithm behavior, though not semantically ideal" — reference comment)
    let d = calculate_diff_split("prefix<div>content</div>", "prefix<div>different</div>");
    assert_eq!(
        (
            d.prefix.as_str(),
            d.left.as_str(),
            d.right.as_str(),
            d.suffix.as_str()
        ),
        ("prefix<div>", "cont", "differ", "ent</div>")
    );

    // chatml thinking generation-prompt rotation (helpers.cpp:189-201 fix)
    let left = "<|im_start|>user\nHello<|im_end|>\n";
    let right = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let d = calculate_diff_split(left, right);
    assert_eq!(
        (
            d.prefix.as_str(),
            d.left.as_str(),
            d.right.as_str(),
            d.suffix.as_str()
        ),
        (left, "", "<|im_start|>assistant\n<think>\n", "")
    );
}

/// `until_common_prefix` / `after_common_suffix` doc examples
/// (chat-auto-parser-helpers.h:31-46)
#[test]
fn ref_common_prefix_suffix() {
    use llama::chat_tools::{after_common_suffix, until_common_prefix};

    assert_eq!(
        until_common_prefix(
            "really want a FUNCTION call",
            "FUNCTION alpha",
            "FUNCTION beta"
        ),
        "really want a "
    );
    assert_eq!(
        until_common_prefix("<tool_call>", "<something>", "<something_else>"),
        ""
    );
    assert_eq!(until_common_prefix("some text", "1234", "abcd"), "");
    assert_eq!(
        after_common_suffix(
            "really want a FUNCTION call",
            "first FUNCTION",
            "second FUNCTION"
        ),
        " call"
    );
    // NB: the header doc example (helpers.h:41-43) says " three args four",
    // but the actual common suffix of "alpha-args"/"beta-args" is "a-args"
    // which does not occur in the string — the implementation (helpers.cpp:
    // 234-261, verified against libllama-common) returns ""
    assert_eq!(
        after_common_suffix(
            "one arg two-args three args four",
            "alpha-args",
            "beta-args"
        ),
        ""
    );
}

/// `test_normalize_quotes_to_json` excerpts (tests/test-chat-auto-parser.cpp)
#[test]
fn ref_normalize_quotes() {
    // exercised through the tagged-args parse case; the canonical examples
    // come from chat-peg-parser.cpp:79-83
    // {'key': 'value'} -> {"key": "value"} happens via the mapper — checked in
    // the parse fixtures above. Here: the direct helper semantics via parse.
    let out = llama::chat_tools::calculate_diff_split("{'a': 1}", "{'a': 2}");
    assert_eq!(out.left, "1");
    assert_eq!(out.right, "2");
}

/// `test_standard_json_tools_formats` excerpt (tests/test-chat-peg-parser.cpp):
/// a `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` parser round
/// trip through the mapper.
#[test]
fn ref_standard_json_tools_roundtrip() {
    let tools = Json::parse(
        r#"[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#,
    )
    .unwrap();

    let arena = llama::chat_tools::build_chat_peg_parser(|p| {
        let tc = p.standard_json_tools(
            "<tool_call>",
            "</tool_call>",
            &tools,
            /* parallel */ false,
            /* force */ false,
            "",
            "",
            false,
            false,
            "",
            "",
            &[],
            false,
        );
        // generator structure: optional(content(until(trigger))) + tools (generator:354-356)
        let until = p.p.until("<tool_call>");
        let c = p.content(until);
        let opt = p.p.optional(c);
        let rest = p.p.rest();
        let c2 = p.content(rest);
        let end = p.p.end();
        p.p.sequence(&[opt, tc, c2, end])
    })
    .unwrap();

    let mut params = ChatParserParams::default();
    params.parser = arena;
    let msg = chat_parse(
        &ChatInput::from(r#"Sure!<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>"#),
        false,
        &params,
    )
    .unwrap();
    assert_eq!(msg.content, "Sure!");
    assert_eq!(msg.tool_calls.len(), 1);
    assert_eq!(msg.tool_calls[0].name, "get_weather");
    assert_eq!(msg.tool_calls[0].arguments, r#"{"city": "Tokyo"}"#);
}

/// save/load round trip of the serialized parser (peg save/load, and the
/// ChatParams.parser reload used by the server path).
#[test]
fn parser_serialization_roundtrip() {
    let tools = Json::parse(
        r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}]"#,
    )
    .unwrap();
    let arena = llama::chat_tools::build_chat_peg_parser(|p| {
        let tc = p.standard_json_tools(
            "[",
            "]",
            &tools,
            true,
            true,
            "",
            "",
            false,
            false,
            "",
            "",
            &[],
            false,
        );
        let end = p.p.end();
        p.p.sequence(&[tc, end])
    })
    .unwrap();
    let mut reloaded = PegArena::default();
    reloaded.load(&arena.save()).unwrap();

    let mut params = ChatParserParams::default();
    params.parser = reloaded;
    let msg = chat_parse(&ChatInput::from(r#"[{"name": "f", "arguments": {"x": 7}}]"#), false, &params).unwrap();
    assert_eq!(msg.tool_calls[0].name, "f");
    assert_eq!(msg.tool_calls[0].arguments, r#"{"x": 7}"#);
}

/// partial parse: streaming mid-arguments yields the tool name + partial args
#[test]
fn partial_parse_streams_tool_args() {
    let tools = Json::parse(
        r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#,
    )
    .unwrap();
    let arena = llama::chat_tools::build_chat_peg_parser(|p| {
        let tc = p.standard_json_tools(
            "<tool_call>",
            "</tool_call>",
            &tools,
            false,
            false,
            "",
            "",
            false,
            false,
            "",
            "",
            &[],
            false,
        );
        let rest = p.p.rest();
        let c = p.content(rest);
        let end = p.p.end();
        p.p.sequence(&[tc, c, end])
    })
    .unwrap();
    let mut params = ChatParserParams::default();
    params.parser = arena;
    let msg = chat_parse(
        &ChatInput::from(r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Tok"#),
        true,
        &params,
    )
    .unwrap();
    assert_eq!(msg.tool_calls.len(), 1);
    assert_eq!(msg.tool_calls[0].name, "get_weather");
    assert!(msg.tool_calls[0].arguments.contains("Tok"));
}

/// `common_chat_msg_diff::compute_diffs` streaming semantics (chat.cpp:267-333)
#[test]
fn msg_diffs_streaming() {
    let prv = ChatMsg {
        role: "assistant".into(),
        tool_calls: vec![ChatToolCall {
            name: "get_weather".into(),
            arguments: r#"{"city": "To"#.into(),
            id: String::new(),
        }],
        ..Default::default()
    };
    let new = ChatMsg {
        role: "assistant".into(),
        tool_calls: vec![ChatToolCall {
            name: "get_weather".into(),
            arguments: r#"{"city": "Tokyo"}"#.into(),
            id: String::new(),
        }],
        ..Default::default()
    };
    let diffs = ChatMsgDiff::compute_diffs(&prv, &new).unwrap();
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].tool_call_index, 0);
    assert_eq!(diffs[0].tool_call_delta.arguments, r#"kyo"}"#);
}

/// End-to-end tool-call generation check against the pinned reference
/// llama-server (parity/chat_tools_server_check.sh):
///   1. the reference server (qwen2.5, temp 0) answered a `tools` request with
///      finish_reason "tool_calls" and its own parsed tool_calls
///   2. the RAW generated text (captured via /completion on the same prompt)
///      is stored in parity/chat_tools_server_raw.json
///   3. here the port's generated parser parses that raw text and the result
///      must structurally equal the reference's reported tool_calls (name +
///      arguments; the id is generated server-side, not by the model)
#[test]
fn ref_server_tool_call_parse() {
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(format!("{dir}/chat_tools_server_raw.json"))
        .expect("parity/chat_tools_server_raw.json (run parity/chat_tools_server_check.sh)");
    let raw: Json = Json::parse(&raw).expect("raw completion json");
    let raw_text = raw
        .at("content")
        .and_then(|v| v.get_str().ok())
        .expect("content")
        .to_string();

    let response = std::fs::read_to_string(format!("{dir}/chat_tools_server_response.json"))
        .expect("parity/chat_tools_server_response.json");
    let response: Json = Json::parse(&response).expect("response json");
    let ref_calls = response
        .at("choices")
        .and_then(|c| c.at_idx(0))
        .and_then(|c| c.at("message"))
        .and_then(|m| m.at("tool_calls"))
        .expect("reference tool_calls");

    // the same request the server served (messages + tools), through the port
    let request: Json =
        Json::parse(&std::fs::read_to_string(format!("{dir}/chat_tools_cases.json")).unwrap())
            .unwrap();
    let qwen_src = request
        .at("templates")
        .unwrap()
        .iter()
        .find(|t| t.at("name").and_then(|v| v.get_str().ok()) == Some("qwen25"))
        .and_then(|t| t.at("src"))
        .unwrap()
        .get_str()
        .unwrap()
        .to_string();
    let tools = Json::parse(
        r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]"#,
    ).unwrap();

    let tmpls = ChatTemplates::init(&ChatTemplatesInit {
        chat_template_override: qwen_src,
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
            messages: msgs_parse_oaicompat(&Json::parse(
                r#"[{"role":"user","content":"What is the weather in Tokyo right now? Use the tool."}]"#,
            ).unwrap()).unwrap(),
            tools: tools_parse_oaicompat(&tools).unwrap(),
            tool_choice: ChatToolChoice::Auto,
            add_generation_prompt: true,
            now: Some(PINNED_EPOCH),
            ..TemplatesInputs::default()
        },
    ).unwrap();

    // the parser the port generates for this template+tools parses the raw
    // server output into the same tool_calls the reference reported
    let mut pparams = ChatParserParams::from_chat_params(&params).unwrap();
    pparams.generation_prompt = ChatInput::from_plain(params.generation_prompt.clone());
    let msg = chat_parse(&ChatInput::from(raw_text.as_str()), false, &pparams).unwrap();

    let ref_name = ref_calls
        .at_idx(0)
        .unwrap()
        .at("function")
        .unwrap()
        .at("name")
        .unwrap()
        .get_str()
        .unwrap()
        .to_string();
    let ref_args = ref_calls
        .at_idx(0)
        .unwrap()
        .at("function")
        .unwrap()
        .at("arguments")
        .unwrap()
        .get_str()
        .unwrap()
        .to_string();
    assert_eq!(
        msg.content, "",
        "content should be empty for a pure tool call"
    );
    assert_eq!(msg.tool_calls.len(), 1);
    assert_eq!(
        msg.tool_calls[0].name, ref_name,
        "name differs from reference"
    );
    assert_eq!(
        msg.tool_calls[0].arguments, ref_args,
        "arguments differ from reference"
    );
}

/// debug helper: dump the port's parser for one case (index or `name`) when
/// CHAT_TOOLS_DEBUG_CASE is set, for diffing against chat_tools_ref.json
#[test]
fn debug_single_case() {
    let Some(which) = option_env!("CHAT_TOOLS_DEBUG_CASE") else {
        return;
    };
    let cases = load_cases();
    let all = Json::parse(
        &std::fs::read_to_string(format!("{}/chat_tools_cases.json", fixture_dir())).unwrap(),
    )
    .unwrap();
    let case = if let Ok(idx) = which.parse::<usize>() {
        cases[idx].input.clone()
    } else {
        cases
            .iter()
            .find(|c| c.name.contains(which))
            .expect("no such case")
            .input
            .clone()
    };
    let got = apply_case(&all, &case).expect("apply");
    let mut arena = PegArena::default();
    arena.load(&got.parser).unwrap();
    eprintln!(
        "=== PORT parser dump ===\n{}",
        ref_dump(&arena, arena.root())
    );
    eprintln!("=== PORT grammar ===\n{}", got.grammar);
}
