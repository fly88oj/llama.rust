//! JSON-schema → GBNF parity with the pinned reference build (`bd4f514db1`).
//!
//! `parity/gen_json_schema_ref.sh` dumps, for every schema in `parity/schemas/`
//! (the reference's own 81 `test-json-schema-to-grammar.cpp` cases, the two extra
//! schemas of its `main()`, 66 hand written fixtures and the 33
//! `test-grammar-integration.cpp` `test_schema()` cases), the GBNF the *reference*
//! converter produces — hex encoded, so the comparison below is byte for byte —
//! plus, for the integration cases, the reference matcher's accept/reject verdict
//! for every expected-pass/expected-fail string.
//!
//! This test regenerates the same sequence with `llama::json_schema` and compares:
//!
//! * the generated GBNF (and the failure message for the schemas the reference
//!   rejects) — `gbnf_matches_reference_byte_for_byte`;
//! * the `common_json::dump()` of the schema (the `SCHEMA` line), which is what
//!   `_generate_constant_rule` prints `const`/`enum` values through;
//! * the accept/reject verdicts of those generated grammars through the ported
//!   matcher — `integration_cases_match_reference_verdicts`;
//! * the `dotall` option of `build_grammar` (the `dotall-*.json` fixtures are
//!   dumped twice, once through `json_schema_to_grammar` and once through
//!   `build_grammar(..., {dotall: true})`).
//!
//! A final test drives grammar-constrained greedy generation from a schema through
//! `GrammarSampler` with the reference qwen2 piece table.

use std::path::{Path, PathBuf};

use llama::grammar::Grammar;
use llama::json_schema::{
    build_grammar, json_schema_to_grammar, json_schema_to_grammar_document, schema_from_json,
    GrammarOptions, Json,
};
use llama::unicode::cpt_from_utf8;

const REF_TXT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/json_schema_ref.txt"
);
const SCHEMAS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/schemas");
const REF_BIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/grammar_pieces_ref.bin"
);

// ---------------------------------------------------------------------------
// the dump
// ---------------------------------------------------------------------------

/// one `CASE` block of parity/json_schema_ref.txt
#[derive(Debug)]
struct Case {
    name: String,
    schema_hex: String,
    /// `None` when the reference rejected the schema (`GBNF_FAIL`)
    gbnf_hex: Option<String>,
    fail_hex: Option<String>,
    /// (expected kind: true = listed as passing, reference matched, string)
    verdicts: Vec<(bool, Option<bool>, Vec<u8>)>,
}

struct Dump {
    cases: Vec<Case>,
    /// `RT <name> <raw schema hex> <common_json::parse(...).dump() hex>`
    roundtrips: Vec<(String, Vec<u8>, Vec<u8>)>,
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

fn load_dump() -> Option<Dump> {
    let text = std::fs::read_to_string(REF_TXT).ok()?;
    let mut cases = Vec::new();
    let mut roundtrips = Vec::new();
    let mut cur: Option<Case> = None;
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (key, rest) = match line.split_once(' ') {
            Some((k, r)) => (k, r),
            None => (line, ""),
        };
        match key {
            "CASE" => {
                cur = Some(Case {
                    name: rest.to_string(),
                    schema_hex: String::new(),
                    gbnf_hex: None,
                    fail_hex: None,
                    verdicts: Vec::new(),
                });
            }
            "SCHEMA" => cur.as_mut().expect("SCHEMA inside CASE").schema_hex = rest.to_string(),
            "GBNF" => cur.as_mut().expect("GBNF inside CASE").gbnf_hex = Some(rest.to_string()),
            "GBNF_FAIL" => {
                cur.as_mut().expect("GBNF_FAIL inside CASE").fail_hex = Some(rest.to_string())
            }
            "ACCEPT" | "REJECT" => {
                let mut parts = rest.splitn(2, ' ');
                let matched = parts.next().unwrap();
                let string_hex = parts.next().unwrap_or("");
                cur.as_mut()
                    .expect("ACCEPT/REJECT inside CASE")
                    .verdicts
                    .push((
                        key == "ACCEPT",
                        if matched == "?" {
                            None
                        } else {
                            Some(matched == "1")
                        },
                        unhex(string_hex),
                    ));
            }
            "RT" => {
                let mut parts = rest.splitn(3, ' ');
                let name = parts.next().unwrap().to_string();
                let raw = unhex(parts.next().unwrap());
                let dumped = unhex(parts.next().unwrap());
                roundtrips.push((name, raw, dumped));
            }
            "END" => cases.push(cur.take().expect("END without CASE")),
            _ => panic!("unknown line in {REF_TXT}: {line}"),
        }
    }
    assert!(
        cases.len() > 100,
        "expected the full dump, got {} cases",
        cases.len()
    );
    Some(Dump { cases, roundtrips })
}

/// the fixture files the `gen_json_schema_ref.sh` dump loop walks over, in the
/// same sorted order bash's glob produces
fn schema_files() -> Vec<PathBuf> {
    let dir = Path::new(SCHEMAS_DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| {
            panic!("cannot read {SCHEMAS_DIR}: {e} (run parity/gen_json_schema_ref.sh)")
        })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            name.ends_with(".json") && name != "integration_cases.json" && !name.starts_with('_')
        })
        .collect();
    files.sort();
    files
}

fn file_stem(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().to_string()
}

// ---------------------------------------------------------------------------
// our side of the comparison
// ---------------------------------------------------------------------------

/// our converter for one fixture file: the parsed schema, its `dump()` and either
/// the GBNF or the failure message
fn convert_file(path: &Path) -> (String, String, Result<String, String>) {
    let text = std::fs::read_to_string(path).unwrap();
    let schema = Json::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (text, schema.dump(), json_schema_to_grammar(&schema, true))
}

/// first differing line with a little context, for the failure messages
fn first_diff(expected: &str, actual: &str) -> String {
    for (i, (e, a)) in expected.lines().zip(actual.lines()).enumerate() {
        if e != a {
            return format!("line {}:\n  expected: {e:?}\n  actual:   {a:?}", i + 1);
        }
    }
    let (e, a): (Vec<&str>, Vec<&str>) = (expected.lines().collect(), actual.lines().collect());
    format!(
        "identical for {} lines, then {:?} vs {:?} ({} vs {} lines)",
        e.len().min(a.len()),
        e.get(a.len()),
        a.get(e.len()),
        e.len(),
        a.len()
    )
}

/// for the file-based cases the dump's `SCHEMA` is the raw fixture text; for the
/// integration cases it is `common_json::dump()` of the parsed schema
fn compare_case(
    case: &Case,
    name: &str,
    schema_text: &str,
    ours_schema_dump: &str,
    result: &Result<String, String>,
) -> bool {
    assert_eq!(
        case.name, name,
        "dump/regeneration order diverged at {name}"
    );
    let mut ok = true;

    // the schema the reference was handed, byte for byte
    let expected_schema = String::from_utf8(unhex(&case.schema_hex)).unwrap();
    let schema_matches = if case.name.starts_with("cases/") {
        expected_schema == ours_schema_dump
    } else {
        expected_schema == schema_text
    };
    if !schema_matches {
        eprintln!("CASE {name}: SCHEMA differs\n  reference: {expected_schema:?}\n  ours:      {ours_schema_dump:?}");
        ok = false;
    }

    match (&case.gbnf_hex, result) {
        (Some(expected_hex), Ok(actual)) => {
            let expected = String::from_utf8(unhex(expected_hex)).unwrap();
            if expected != *actual {
                eprintln!(
                    "CASE {name}: GBNF differs — {}",
                    first_diff(&expected, actual)
                );
                eprintln!("  reference:\n{expected}\n  ours:\n{actual}");
                ok = false;
            }
        }
        (None, Err(actual)) => {
            let expected = String::from_utf8(unhex(case.fail_hex.as_ref().unwrap())).unwrap();
            if expected != *actual {
                eprintln!("CASE {name}: failure message differs\n  reference: {expected:?}\n  ours: {actual:?}");
                ok = false;
            }
        }
        (Some(_), Err(actual)) => {
            eprintln!("CASE {name}: the reference converted the schema, we failed: {actual}");
            ok = false;
        }
        (None, Ok(actual)) => {
            eprintln!("CASE {name}: the reference rejected the schema, we produced:\n{actual}");
            ok = false;
        }
    }
    ok
}

// ---------------------------------------------------------------------------
// 1. the GBNF bytes
// ---------------------------------------------------------------------------

#[test]
fn gbnf_matches_reference_byte_for_byte() {
    let dump = match load_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_json_schema_ref.sh)");
            return;
        }
    };

    let mut ours: Vec<(String, String, Result<String, String>)> = Vec::new();
    let mut dotalls: Vec<(String, String, Result<String, String>)> = Vec::new();
    for path in schema_files() {
        let body = convert_file(&path);
        if file_stem(&path).starts_with("dotall-") {
            // the same fixtures again through build_grammar(dotall = true)
            let text = std::fs::read_to_string(&path).unwrap();
            let schema = Json::parse(&text).unwrap();
            let doc = schema_from_json(&schema).unwrap();
            let gbnf = build_grammar(&doc, GrammarOptions { dotall: true }, |builder| {
                builder.add_schema("root", doc.root);
            });
            dotalls.push((file_stem(&path), text, gbnf));
        }
        ours.push((file_stem(&path), body.0, body.2));
    }

    // the file/dotall blocks come first; the `cases/` blocks are the integration
    // cases checked by `integration_cases_match_reference_verdicts`
    let blocks: Vec<&Case> = dump
        .cases
        .iter()
        .filter(|c| !c.name.starts_with("cases/"))
        .collect();

    let mut matched = 0usize;
    let mut total = 0usize;
    let mut i = 0usize;
    for (name, text, result) in &ours {
        let path = Path::new(SCHEMAS_DIR).join(format!("{name}.json"));
        let (_, schema_dump, _) = convert_file(&path);
        total += 1;
        if compare_case(blocks[i], name, text, &schema_dump, result) {
            matched += 1;
        }
        i += 1;
    }
    for (name, text, result) in &dotalls {
        let path = Path::new(SCHEMAS_DIR).join(format!("{name}.json"));
        let (_, schema_dump, _) = convert_file(&path);
        total += 1;
        if compare_case(blocks[i], name, text, &schema_dump, result) {
            matched += 1;
        }
        i += 1;
    }
    assert_eq!(
        i,
        blocks.len(),
        "the dump has {} schema blocks, we generated {i}",
        blocks.len()
    );

    eprintln!("json-schema → GBNF byte-for-byte: {matched}/{total} schemas matched");
    assert_eq!(
        matched,
        total,
        "{} schemas differ from the reference",
        total - matched
    );
}

// ---------------------------------------------------------------------------
// 2. the matcher verdicts of the generated grammars
// ---------------------------------------------------------------------------

/// `parse_tokens()` (tests/test-grammar-integration.cpp:56-85): the `0xff`-prefixed
/// token ids plus UTF-8 text, invalid UTF-8 becoming one U+FFFD per byte
fn parse_tokens(input: &[u8]) -> Vec<(i32, Vec<u8>)> {
    let mut out = Vec::with_capacity(input.len());
    let mut offset = 0usize;
    while offset < input.len() {
        if input[offset] == 0xff {
            assert!(offset + 5 <= input.len(), "not enough bytes for token id");
            let val = ((input[offset + 1] as u32) << 24)
                | ((input[offset + 2] as u32) << 16)
                | ((input[offset + 3] as u32) << 8)
                | (input[offset + 4] as u32);
            out.push((val as i32, format!("<[{val}]>").into_bytes()));
            offset += 5;
        } else {
            let mut off = offset;
            match cpt_from_utf8(input, &mut off) {
                Ok(cpt) => {
                    out.push((0, llama::unicode::cpt_to_utf8(cpt).into_bytes()));
                    offset = off;
                }
                Err(()) => {
                    offset += 1;
                    out.push((0, llama::unicode::cpt_to_utf8(0xFFFD).into_bytes()));
                }
            }
        }
    }
    out
}

/// `match_string()` (tests/test-grammar-integration.cpp:87-114)
fn match_string(input: &[u8], grammar: &mut Grammar) -> bool {
    for (id, piece) in parse_tokens(input) {
        if grammar.accept_token(id, &piece).is_err() {
            return false;
        }
        if grammar.stacks.is_empty() {
            return false;
        }
    }
    grammar.stacks.iter().any(|s| s.is_empty())
}

#[test]
fn integration_cases_match_reference_verdicts() {
    let dump = match load_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_json_schema_ref.sh)");
            return;
        }
    };
    // the integration cases are the last blocks of the dump, named `cases/<desc>`
    let cases: Vec<&Case> = dump
        .cases
        .iter()
        .filter(|c| c.name.starts_with("cases/"))
        .collect();
    assert!(!cases.is_empty(), "no integration cases in the dump");

    let cases_json =
        std::fs::read_to_string(Path::new(SCHEMAS_DIR).join("integration_cases.json")).unwrap();
    let parsed = Json::parse(&cases_json).unwrap();
    let entries = parsed.iter().collect::<Vec<_>>();
    assert_eq!(
        entries.len(),
        cases.len(),
        "integration fixture/dump size mismatch"
    );

    let mut verdicts_checked = 0usize;
    let mut cases_checked = 0usize;
    for (entry, case) in entries.iter().zip(&cases) {
        let name = entry.at("name").unwrap().get_str().unwrap();
        assert_eq!(
            case.name,
            format!("cases/{name}"),
            "integration case order diverged"
        );

        // the schema text of the dump is `common_json::dump()` of the parsed case
        let schema = entry.at("schema").unwrap();
        let schema_dump = schema.dump();
        let expected_schema = String::from_utf8(unhex(&case.schema_hex)).unwrap();
        assert_eq!(
            schema_dump, expected_schema,
            "SCHEMA dump differs for {name}"
        );

        let gbnf = json_schema_to_grammar(schema, true).unwrap_or_else(|e| panic!("{name}: {e}"));
        let expected_gbnf =
            String::from_utf8(unhex(case.gbnf_hex.as_ref().expect("GBNF"))).unwrap();
        assert_eq!(
            gbnf,
            expected_gbnf,
            "GBNF differs for {name} — {}",
            first_diff(&expected_gbnf, &gbnf)
        );
        assert!(
            Grammar::parse(None, &gbnf, "root").is_ok(),
            "{name}: our own grammar must parse"
        );

        for (expect_accept, reference_matched, text) in &case.verdicts {
            let reference_matched = reference_matched.expect("the reference built every grammar");
            // the reference matcher agrees with its own test expectations
            assert_eq!(
                *expect_accept,
                reference_matched,
                "{name}: the reference contradicts its expectation for {:?}",
                String::from_utf8_lossy(text)
            );
            // a fresh grammar per string, like the C test's stacks reset
            let mut grammar = Grammar::parse(None, &gbnf, "root").unwrap();
            let ours = match_string(text, &mut grammar);
            assert_eq!(
                ours,
                reference_matched,
                "{name}: {:?} — reference {}, ours {}",
                String::from_utf8_lossy(text),
                reference_matched,
                ours
            );
            verdicts_checked += 1;
        }
        cases_checked += 1;
    }
    eprintln!("integration cases: {cases_checked} grammars, {verdicts_checked} accept/reject verdicts matched");
}

// ---------------------------------------------------------------------------
// 3. the document entry point, the failure messages, `build_grammar`
// ---------------------------------------------------------------------------

#[test]
fn document_entry_point_agrees_with_json_entry_point() {
    let mut checked = 0usize;
    for path in schema_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        let schema = Json::parse(&text).unwrap();
        let Ok(doc) = schema_from_json(&schema) else {
            continue;
        };
        let via_document = json_schema_to_grammar_document(&doc);
        let via_json = json_schema_to_grammar(&schema, true);
        assert_eq!(
            via_document,
            via_json,
            "{}: the two entry points differ",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 100, "only {checked} schemas built");
    eprintln!("document/json entry points agree on {checked} schemas");
}

#[test]
fn failure_cases_report_the_reference_message() {
    // the exact reference messages (the same strings the dump records)
    let cases: &[(&str, &str)] = &[
        (
            r#"{"type": "kaboom"}"#,
            "JSON schema conversion failed:\nJSON schema error at #: unrecognized type kaboom",
        ),
        (
            r#"{"type": 42}"#,
            "JSON schema conversion failed:\nJSON schema error at #: type must be a string or an array of strings",
        ),
        (
            r#"{"type": "string", "pattern": "^(a$"}"#,
            "JSON schema conversion failed:\nInvalid pattern ^(a$: unbalanced parentheses",
        ),
        ("true", "JSON schema conversion failed:\nJSON schema error at #: schema must be an object"),
    ];
    for (schema, expected) in cases {
        let json = Json::parse(schema).unwrap();
        assert_eq!(json_schema_to_grammar(&json, true).unwrap_err(), *expected);
    }
}

/// `common_json::parse()` + `dump()` on every fixture must match nlohmann's, which
/// is what `_generate_constant_rule` prints `const`/`enum` values through (the
/// `RT` lines of the dump).
#[test]
fn json_parse_and_dump_match_nlohmann() {
    let dump = match load_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_json_schema_ref.sh)");
            return;
        }
    };
    assert!(dump.roundtrips.len() > 100, "no RT lines in the dump");
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for (name, raw, expected_dump) in &dump.roundtrips {
        let text = String::from_utf8(raw.clone()).unwrap();
        match Json::parse(&text) {
            Ok(json) => {
                let ours = json.dump().into_bytes();
                if ours != *expected_dump {
                    failures.push(format!(
                        "{name}: dump differs\n  reference: {:?}\n  ours:      {:?}",
                        String::from_utf8_lossy(expected_dump),
                        String::from_utf8_lossy(&ours)
                    ));
                }
            }
            Err(e) => failures.push(format!("{name}: our parser rejected it: {e}")),
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} JSON round trips differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("common_json parse+dump: {checked} fixtures round-tripped like nlohmann");
}

/// `verify_expectation_parseable()` (tests/test-json-schema-to-grammar.cpp:44-56):
/// the reference parses every grammar it expects to succeed, so the grammars we
/// generate for the reference's own test suite must be accepted by the ported
/// parser too (and contain a `root` symbol).
#[test]
fn reference_test_suite_grammars_are_parseable() {
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for path in schema_files() {
        let name = file_stem(&path);
        let text = std::fs::read_to_string(&path).unwrap();
        let schema = Json::parse(&text).unwrap();
        let Ok(gbnf) = json_schema_to_grammar(&schema, true) else {
            continue;
        };
        if !name.starts_with("tc-") || name.contains("unbalanced") {
            continue;
        }
        match Grammar::parse(None, &gbnf, "root") {
            Ok(_) => checked += 1,
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "grammars the reference expects to parse:\n{}",
        failures.join("\n")
    );
    assert!(checked >= 75, "only {checked} reference grammars parsed");
    eprintln!("reference test-suite grammars: {checked} parsed by the ported GBNF parser");
}

#[test]
fn build_grammar_adds_rules_and_schemas() {
    let schema =
        Json::parse(r#"{"type": "object", "properties": {"a": {"type": "string"}}}"#).unwrap();
    let doc = schema_from_json(&schema).unwrap();
    let gbnf = build_grammar(&doc, GrammarOptions::default(), |builder| {
        builder.add_rule("greeting", "\"hello\"");
        let prop = match &doc.node(doc.root).kind {
            llama::json_schema::SchemaKind::Object { properties, .. } => properties[0].schema,
            _ => unreachable!(),
        };
        builder.add_schema("root", prop);
    })
    .unwrap();
    assert!(gbnf.contains("greeting ::= \"hello\"\n"));
    assert!(
        gbnf.contains("root ::= \"\\\"\" char* \"\\\"\"\n"),
        "{gbnf}"
    );
    assert!(Grammar::parse(None, &gbnf, "root").is_ok());
    assert!(Grammar::parse(None, &gbnf, "greeting").is_ok());
}

// ---------------------------------------------------------------------------
// 4. grammar-constrained greedy generation from a schema (the llama-cli path)
// ---------------------------------------------------------------------------

/// reference qwen2 piece table (parity/grammar_pieces_ref.bin)
struct RefPieces {
    pieces: Vec<Vec<u8>>,
    is_eog: Vec<bool>,
}

impl RefPieces {
    fn load(path: &str) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut off = 4usize;
        let mut pieces = Vec::with_capacity(n);
        let mut is_eog = Vec::with_capacity(n);
        for _ in 0..n {
            let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            pieces.push(bytes[off..off + len].to_vec());
            off += len;
            is_eog.push(bytes[off] != 0);
            off += 1;
        }
        Some(RefPieces { pieces, is_eog })
    }
}

/// the `constrained_greedy` driver of crates/llama/tests/grammar_parity.rs:1448:
/// temp 0, the "wanted" token boosted below an illegal one, so the sampler has to
/// mask the boost away at every step
fn constrained_greedy(
    grammar: &mut llama::sampling::GrammarSampler,
    n_vocab: usize,
    wanted: &[i32],
    invalid: i32,
    eog: i32,
) -> Vec<i32> {
    use llama::sampling::{SamplingContext, SamplingParams};
    let mut ctx = SamplingContext::new(
        n_vocab as i32,
        SamplingParams {
            temp: 0.0,
            seed: 0,
            ..Default::default()
        },
    );
    let mut out = Vec::new();
    for &want in wanted {
        let mut logits = vec![0.0f32; n_vocab];
        logits[invalid as usize] = 200.0;
        logits[want as usize] = 100.0;
        let tok = ctx.sample_with_grammar(&logits, grammar).unwrap();
        assert_eq!(tok, want, "the boosted illegal token must not be sampled");
        out.push(tok);
    }
    // once complete only EOG may win over the boosted illegal token
    let mut logits = vec![0.0f32; n_vocab];
    logits[eog as usize] = 200.0;
    logits[invalid as usize] = 100.0;
    out.push(ctx.sample_with_grammar(&logits, grammar).unwrap());
    out
}

#[test]
fn constrained_greedy_generation_from_a_schema() {
    let vocab = match RefPieces::load(REF_BIN) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: {REF_BIN} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    let n_vocab = vocab.pieces.len();
    let pieces = llama::grammar::VocabPieces::from_iter(
        vocab
            .pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), vocab.is_eog[i])),
    );

    // an object with one enum property: after `{"a":` only `"yes"` or `"no"` fit
    let schema = Json::parse(
        r#"{"type": "object", "properties": {"a": {"enum": ["yes", "no"]}}, "required": ["a"], "additionalProperties": false}"#,
    )
    .unwrap();
    let gbnf = json_schema_to_grammar(&schema, true).unwrap();
    let mut grammar = llama::sampling::GrammarSampler::from_pieces(&gbnf, "root", pieces).unwrap();

    let find = |p: &[u8]| {
        vocab
            .pieces
            .iter()
            .position(|q| q == p)
            .unwrap_or_else(|| panic!("no token with piece {p:?}")) as i32
    };
    let wanted = [
        find(b"{"),
        find(b"\""),
        find(b"a"),
        find(b"\""),
        find(b":"),
        find(b"\""),
        find(b"y"),
        find(b"e"),
        find(b"s"),
        find(b"\""),
        find(b"}"),
    ];
    // boosted above the wanted token at every step: illegal inside the enum literal
    const QWEN2_EOG: i32 = 151643;
    let out = constrained_greedy(&mut grammar, n_vocab, &wanted, find(b"z"), QWEN2_EOG);
    assert_eq!(
        &out[..wanted.len()],
        &wanted,
        "greedy output must follow the enum"
    );
    let text: Vec<u8> = out[..wanted.len()]
        .iter()
        .flat_map(|&id| vocab.pieces[id as usize].clone())
        .collect();
    assert_eq!(String::from_utf8(text).unwrap(), r#"{"a":"yes"}"#);
    assert_eq!(
        out[wanted.len()],
        QWEN2_EOG,
        "the grammar must allow EOG once complete"
    );

    // the grammar must not accept a value outside the enum
    let mut bad = Grammar::parse(None, &gbnf, "root").unwrap();
    assert!(!match_string(br#"{"a":"maybe"}"#, &mut bad));
    let mut good = Grammar::parse(None, &gbnf, "root").unwrap();
    assert!(match_string(br#"{"a":"no"}"#, &mut good));
}
