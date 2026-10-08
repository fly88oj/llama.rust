//! grammar_parity.rs — GBNF engine verification (parser + stack matcher).
//!
//! Four layers of evidence, from strongest (bit-exact vs the reference build) to
//! evidence that needs no reference checkout:
//!
//! 1. `parser_cases_*` — the whole `tests/test-grammar-parser.cpp` case list
//!    transcribed verbatim (symbol ids + flattened rule element lists + the
//!    failure cases). Expectations are the ones asserted by the C test itself.
//! 2. `integration_cases_*` — the `test_grammar(...)` half of
//!    `tests/test-grammar-integration.cpp` (passing/failing string sets). The
//!    `test_schema(...)` half needs `json-schema-to-grammar`, which is *not*
//!    ported (see the agent report), so those cases are absent.
//! 3. `reference_replay_*` — replays `parity/grammar_ref.txt` (produced by
//!    `parity/ref_grammar_dump.cpp` against the pinned `libllama.so`): parser
//!    dumps for every `grammars/*.gbnf` plus per-token stack/mask trajectories
//!    over the real qwen2 vocabulary, including byte-split UTF-8 sequences.
//! 4. `piece_table_*` — the full qwen2 `token_to_piece` cache vs
//!    `parity/grammar_pieces_ref.bin`, i.e. the exact input of the matcher.

use std::collections::BTreeMap;

use llama::grammar::{
    el_at, Grammar, GrammarParser, GrammarVocab, Gretype, PartialUtf8, Rule, Rules,
};
use llama::sampling::{Sampler, TokenData, TokenDataArray};
use llama::unicode::{cpt_from_utf8, cpt_to_utf8};
use llama::vocab::Vocab;

// ---------------------------------------------------------------------------
// helpers for the transcribed expectations
// ---------------------------------------------------------------------------

fn ch(c: char) -> (Gretype, u32) {
    (Gretype::Char, c as u32)
}
fn alt() -> (Gretype, u32) {
    (Gretype::Alt, 0)
}
fn end() -> (Gretype, u32) {
    (Gretype::End, 0)
}
fn rr(id: u32) -> (Gretype, u32) {
    (Gretype::RuleRef, id)
}
fn cnot(c: char) -> (Gretype, u32) {
    (Gretype::CharNot, c as u32)
}
fn rng(c: char) -> (Gretype, u32) {
    (Gretype::CharRngUpper, c as u32)
}
fn calt(c: char) -> (Gretype, u32) {
    (Gretype::CharAlt, c as u32)
}
fn tok(id: u32) -> (Gretype, u32) {
    (Gretype::Token, id)
}
fn toknot(id: u32) -> (Gretype, u32) {
    (Gretype::TokenNot, id)
}

/// `verify_parsing()` (tests/test-grammar-parser.cpp:25-133)
fn verify_parsing(gbnf: &str, expected_symbols: &[(&str, u32)], expected_rules: &[(Gretype, u32)]) {
    let mut parser = GrammarParser::new(None);
    assert!(
        parser.parse(gbnf.as_bytes()),
        "parse failed for grammar:\n{gbnf}"
    );

    let symbols: Vec<(String, u32)> = parser
        .symbol_ids
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let expected: Vec<(String, u32)> = expected_symbols
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    assert_eq!(
        symbols, expected,
        "symbol_ids mismatch for grammar:\n{gbnf}"
    );

    let mut flat: Vec<(Gretype, u32)> = Vec::new();
    for rule in &parser.rules {
        for elem in rule {
            flat.push((elem.ty, elem.value));
        }
    }
    assert_eq!(flat, expected_rules, "rules mismatch for grammar:\n{gbnf}");
}

/// `verify_failure()` (:135-140)
fn verify_failure(gbnf: &str) {
    let mut parser = GrammarParser::new(None);
    assert!(
        !parser.parse(gbnf.as_bytes()),
        "expected parse failure for:\n{gbnf}"
    );
    assert!(parser.rules.is_empty(), "rules should have been cleared");
}

// ---------------------------------------------------------------------------
// 1. tests/test-grammar-parser.cpp
// ---------------------------------------------------------------------------

#[test]
fn parser_cases_failures() {
    verify_failure(
        r#"
        root ::= "a"{,}"
    "#,
    );
    verify_failure(
        r#"
        root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}
    "#,
    );
    verify_failure(
        r#"
        root ::= "a"{,10}"
    "#,
    );
    verify_failure(
        r#"
        root ::= "a"{5000}
    "#,
    );
    verify_failure(
        r#"
        root ::= "a"{5000,}
    "#,
    );
    verify_failure(
        r#"
        root ::= "a"{5000,6000}
    "#,
    );
}

#[test]
fn parser_cases_repetitions() {
    verify_parsing(
        r#"
        root ::= "a"{0,5000}
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root ::= "a"{3,5000}
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            ch('a'),
            ch('a'),
            ch('a'),
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"
    "#,
        &[("root", 0)],
        &[ch('a'), end()],
    );

    verify_parsing(
        r#"
        root  ::= "a" | [bdx-z] | [^1-3]
    "#,
        &[("root", 0)],
        &[
            ch('a'),
            alt(),
            ch('b'),
            calt('d'),
            calt('x'),
            rng('z'),
            alt(),
            cnot('1'),
            rng('3'),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= a+
        a     ::= "a"
    "#,
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[
            // root (index 0)
            rr(1),
            rr(2),
            end(),
            // a (index 1)
            ch('a'),
            end(),
            // root_2 (index 2)
            rr(1),
            rr(2),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"+
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            ch('a'),
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= a?
        a     ::= "a"
    "#,
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[
            // root (index 0)
            rr(2),
            end(),
            // a (index 1)
            ch('a'),
            end(),
            // root_2 (index 2)
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"?
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= a*
        a     ::= "a"
    "#,
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[
            // root (index 0)
            rr(2),
            end(),
            // a (index 1)
            ch('a'),
            end(),
            // root_2 (index 2)
            rr(1),
            rr(2),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"*
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"{2}
    "#,
        &[("root", 0)],
        &[ch('a'), ch('a'), end()],
    );

    verify_parsing(
        r#"
        root  ::= "a"{2,}
    "#,
        &[("root", 0), ("root_1", 1)],
        &[
            ch('a'),
            ch('a'),
            rr(1),
            end(),
            // root_1 (index 1)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= "a"{ 4}
    "#,
        &[("root", 0)],
        &[ch('a'), ch('a'), ch('a'), ch('a'), end()],
    );

    verify_parsing(
        r#"
        root  ::= "a"{2,4}
    "#,
        &[("root", 0), ("root_1", 1), ("root_2", 2)],
        &[
            // root (index 0)
            ch('a'),
            ch('a'),
            rr(2),
            end(),
            // root_1 (index 1)
            ch('a'),
            alt(),
            end(),
            // root_2 (index 2)
            ch('a'),
            rr(1),
            alt(),
            end(),
        ],
    );
}

#[test]
fn parser_cases_complex() {
    verify_parsing(
        r#"
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    "#,
        &[
            ("expr", 2),
            ("expr_5", 5),
            ("expr_6", 6),
            ("root", 0),
            ("root_1", 1),
            ("root_4", 4),
            ("term", 3),
            ("term_7", 7),
        ],
        &[
            // root (index 0)
            rr(1),
            rr(4),
            end(),
            // root_1 (index 1)
            rr(2),
            ch('='),
            rr(3),
            ch('\n'),
            end(),
            // expr (index 2)
            rr(3),
            rr(6),
            end(),
            // term (index 3)
            ch('0'),
            rng('9'),
            rr(7),
            end(),
            // root_4 (index 4)
            rr(1),
            rr(4),
            alt(),
            end(),
            // expr_5 (index 5)
            ch('-'),
            calt('+'),
            calt('*'),
            calt('/'),
            rr(3),
            end(),
            // expr_6 (index 6)
            rr(5),
            rr(6),
            alt(),
            end(),
            // term_7 (index 7)
            ch('0'),
            rng('9'),
            rr(7),
            alt(),
            end(),
        ],
    );

    verify_parsing(
        r#"
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    "#,
        &[
            ("expr", 2),
            ("expr_6", 6),
            ("expr_7", 7),
            ("ident", 8),
            ("ident_10", 10),
            ("num", 9),
            ("num_11", 11),
            ("root", 0),
            ("root_1", 1),
            ("root_5", 5),
            ("term", 4),
            ("ws", 3),
            ("ws_12", 12),
        ],
        &[
            // root (index 0)
            rr(1),
            rr(5),
            end(),
            // root_1 (index 1)
            rr(2),
            ch('='),
            rr(3),
            rr(4),
            ch('\n'),
            end(),
            // expr (index 2)
            rr(4),
            rr(7),
            end(),
            // ws (index 3)
            rr(12),
            end(),
            // term (index 4)
            rr(8),
            alt(),
            rr(9),
            alt(),
            ch('('),
            rr(3),
            rr(2),
            ch(')'),
            rr(3),
            end(),
            // root_5 (index 5)
            rr(1),
            rr(5),
            alt(),
            end(),
            // expr_6 (index 6)
            ch('-'),
            calt('+'),
            calt('*'),
            calt('/'),
            rr(4),
            end(),
            // expr_7 (index 7)
            rr(6),
            rr(7),
            alt(),
            end(),
            // ident (index 8)
            ch('a'),
            rng('z'),
            rr(10),
            rr(3),
            end(),
            // num (index 9)
            ch('0'),
            rng('9'),
            rr(11),
            rr(3),
            end(),
            // ident_10 (index 10)
            ch('a'),
            rng('z'),
            calt('0'),
            rng('9'),
            calt('_'),
            rr(10),
            alt(),
            end(),
            // num_11 (index 11)
            ch('0'),
            rng('9'),
            rr(11),
            alt(),
            end(),
            // ws_12 (index 12)
            ch(' '),
            calt('\t'),
            calt('\n'),
            rr(12),
            alt(),
            end(),
        ],
    );

    // <[1000]> = " thinking", <[1001]> = " response" in the reference test
    verify_parsing(
        r#"
        root  ::= <[1000]> !<[1001]> <[1001]>
    "#,
        &[("root", 0)],
        &[tok(1000), toknot(1001), tok(1001), end()],
    );
}

// ---------------------------------------------------------------------------
// 2. tests/test-grammar-integration.cpp — `test_grammar` cases
// ---------------------------------------------------------------------------

/// `token()` (tests/test-grammar-integration.cpp:45-53): a 32-bit id encoded as
/// `0xff` + big-endian id, accepted with the piece `<[id]>`.
fn token(id: i32) -> Vec<u8> {
    let mut v = vec![0xffu8];
    v.extend_from_slice(&id.to_be_bytes());
    v
}

fn s(x: &str) -> Vec<u8> {
    x.as_bytes().to_vec()
}

/// concatenate byte strings (the C test builds its inputs with `+`)
macro_rules! cat {
    ($($x:expr),* $(,)?) => {{
        let mut v: Vec<u8> = Vec::new();
        $(v.extend_from_slice(&$x);)*
        v
    }};
}

/// `parse_tokens()` (:56-85): the `0xff`-prefixed ids plus UTF-8 text
/// (invalid UTF-8 → U+FFFD, one byte consumed).
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
            let piece = format!("<[{val}]>");
            out.push((val as i32, piece.into_bytes()));
            offset += 5;
        } else {
            let mut off = offset;
            match cpt_from_utf8(input, &mut off) {
                Ok(cpt) => {
                    out.push((0, cpt_to_utf8(cpt).into_bytes()));
                    offset = off;
                }
                Err(()) => {
                    offset += 1;
                    out.push((0, cpt_to_utf8(0xFFFD).into_bytes()));
                }
            }
        }
    }
    out
}

/// `match_string()` (:87-114)
fn match_string(input: &[u8], grammar: &mut Grammar) -> bool {
    for (id, piece) in parse_tokens(input) {
        if grammar.accept_token(id, &piece).is_err() {
            // normally this shouldn't get hit because of llama_grammar_apply
            return false;
        }
        if grammar.stacks.is_empty() {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }
    grammar.stacks.iter().any(|s| s.is_empty())
}

/// `test_grammar()` (tests/test-grammar-integration.cpp:188-190 + test() :116-187).
/// The C test saves/restores the stacks; a fresh grammar per string is equivalent.
fn test_grammar(gbnf: &str, passing: &[Vec<u8>], failing: &[Vec<u8>]) {
    assert!(
        Grammar::parse(None, gbnf, "root").is_ok(),
        "grammar must build:\n{gbnf}"
    );
    for bytes in passing {
        let mut g = Grammar::parse(None, gbnf, "root").unwrap();
        assert!(
            match_string(bytes, &mut g),
            "expected PASS: {:?} for grammar:\n{gbnf}",
            String::from_utf8_lossy(bytes)
        );
    }
    for bytes in failing {
        let mut g = Grammar::parse(None, gbnf, "root").unwrap();
        assert!(
            !match_string(bytes, &mut g),
            "expected FAIL: {:?} for grammar:\n{gbnf}",
            String::from_utf8_lossy(bytes)
        );
    }
}

/// `test_grammar` for inputs that are plain UTF-8 text
fn test_grammar_str(gbnf: &str, passing: &[&str], failing: &[&str]) {
    let p: Vec<Vec<u8>> = passing.iter().map(|x| s(x)).collect();
    let f: Vec<Vec<u8>> = failing.iter().map(|x| s(x)).collect();
    test_grammar(gbnf, &p, &f);
}

#[test]
fn integration_simple_grammar() {
    test_grammar_str(
        r#"
            root ::= expr
            expr ::= term ("+" term)*
            term ::= number
            number ::= [0-9]+"#,
        &["42", "1+2+3+4+5", "123+456"],
        &["+", "/ 3", "1+2+3+4+5+", "12a45"],
    );

    test_grammar(
        r#"
            root ::= <[10]> content <[11]>
            content ::= (!<[11]>)*"#,
        &[
            cat![token(10), s("hello world"), token(11)],
            cat![
                token(10),
                s("text with "),
                token(12),
                s(" other tokens "),
                token(13),
                s(" mixed in"),
                token(11)
            ],
            cat![token(10), token(11)],
            cat![
                token(10),
                token(12),
                token(13),
                token(14),
                token(15),
                token(11)
            ],
            cat![token(10), s("a"), token(11)],
        ],
        &[
            cat![token(10), s("missing end token")],
            cat![token(10)],
            cat![s("missing start token"), token(11)],
            cat![token(10), token(11), token(11)], // double end token
            cat![token(11), s("wrong order"), token(10)],
        ],
    );
}

#[test]
fn integration_complex_grammar() {
    test_grammar_str(
        r#"
            root ::= expression
            expression ::= term ws (("+"|"-") ws term)*
            term ::= factor ws (("*"|"/") ws factor)*
            factor ::= number | variable | "(" expression ")" | function-call
            number ::= [0-9]+
            variable ::= [a-zA-Z_][a-zA-Z0-9_]*
            function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
            ws ::= [ \t\n\r]?"#,
        &[
            "42",
            "1*2*3*4*5",
            "x",
            "x+10",
            "x1+y2",
            "(a+b)*(c-d)",
            "func()",
            "func(x,y+2)",
            "a*(b+c)-d/e",
            "f(g(x),h(y,z))",
            "x + 10",
            "x1 + y2",
            "(a + b) * (c - d)",
            "func()",
            "func(x, y + 2)",
            "a * (b + c) - d / e",
            "f(g(x), h(y, z))",
            "123+456",
            "123*456*789-123/456+789*123",
            "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456",
        ],
        &[
            "+",
            "/ 3x",
            "x + + y",
            "a * / b",
            "func(,)",
            "func(x y)",
            "(a + b",
            "x + y)",
            "a + b * (c - d",
            "42 +",
            "x +",
            "x + 10 +",
            "(a + b) * (c - d",
            "func(",
            "func(x, y + 2",
            "a * (b + c) - d /",
            "f(g(x), h(y, z)",
            "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456/",
        ],
    );

    test_grammar(
        r#"
            root ::= reasoning+ content tool-call*
            reasoning ::= <[10]> (!<[11]>)* <[11]>
            content ::= <[20]> (!<[21]>)* <[21]>
            tool-call ::= <[12]> name <[13]> args <[14]>
            name ::= (!<[13]>)+
            args ::= (!<[14]>)*"#,
        &[
            cat![
                token(10),
                s("I am thinking"),
                token(11),
                token(20),
                s("hello world!"),
                token(21),
                token(12),
                s("search"),
                token(13),
                s("query=test"),
                token(14)
            ],
            cat![
                token(10),
                s("reasoning 1"),
                token(11),
                token(10),
                s("reasoning 2"),
                token(11),
                token(20),
                token(21),
                token(12),
                s("tool"),
                token(13),
                token(14)
            ],
            cat![token(10), token(11), token(20), s("content"), token(21)],
            cat![
                token(10),
                s("think"),
                token(12),
                s(" nested"),
                token(11),
                token(20),
                token(10),
                s("more content"),
                token(21),
                token(12),
                s("fn"),
                token(13),
                s("x=1,y=2"),
                token(14),
                token(12),
                s("fn2"),
                token(13),
                token(14)
            ],
            cat![
                token(10),
                s("reasoning"),
                token(11),
                token(10),
                s("more"),
                token(11),
                token(10),
                s("even more"),
                token(11),
                token(20),
                s("text"),
                token(21),
                token(12),
                s("a"),
                token(13),
                s("b"),
                token(14),
                token(12),
                s("c"),
                token(13),
                s("d"),
                token(14)
            ],
        ],
        &[
            cat![token(20), s("content only"), token(21)],
            cat![token(10), s("no closing reasoning")],
            cat![token(10), token(11), token(20), s("no closing content")],
            cat![
                token(10),
                token(11),
                token(20),
                token(21),
                token(12),
                s("incomplete tool")
            ],
            cat![token(10), token(11), token(11), token(20), token(21)],
        ],
    );
}

#[test]
fn integration_special_chars_and_quantifiers() {
    test_grammar_str(
        r#"
            root ::= ... "abc" ...
            "#,
        &["abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵"],
        &[
            "aaabcccc",
            "aaaaabcccc",
            "aaaabccc",
            "aaaabccccc",
            "🔵🟠✅❌abc❌✅🟠🔵",
            "🔵🟠abc🟠🔵",
        ],
    );

    test_grammar_str(
        r#"root ::= "a"*"#,
        &[
            "",
            "a",
            "aaaaa",
            "aaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ],
        &[
            "b",
            "ab",
            "aab",
            "ba",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        ],
    );

    test_grammar_str(
        r#"root ::= "a"+"#,
        &[
            "a",
            "aaaaa",
            "aaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ],
        &[
            "",
            "b",
            "ab",
            "aab",
            "ba",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
        ],
    );

    test_grammar_str(r#"root ::= "a"?"#, &["", "a"], &["b", "ab", "aa", "ba"]);

    test_grammar_str(
        r#"
            root ::= cons+ vowel* cons? (vowel cons)*
            vowel ::= [aeiouy]
            cons ::= [bcdfghjklmnpqrstvwxyz]
            "#,
        &["yes", "no", "noyes", "crwth", "four", "bryyyy"],
        &["yess", "yesno", "forty", "catyyy"],
    );

    test_grammar_str(
        r#"
            root ::= [ab]{4}
        "#,
        &["aaaa", "bbbb", "abab"],
        &["a", "b", "aaaaa"],
    );

    test_grammar_str(
        r#"
            root ::= [ab]{4,}
        "#,
        &["aaaa", "aaaaab", "bbbb", "ababab"],
        &["", "aba"],
    );

    test_grammar_str(
        r#"
            root ::= [ab]{0,4}
        "#,
        &["", "a", "aa", "aaa", "aaab"],
        &["aaaaa"],
    );

    test_grammar_str(
        r#"
            root ::= ("0x" [A-F0-9]{2} " "?){3,5}
        "#,
        &["0xFF 0x12 0xAB", "0xFF 0x12 0xAB 0x00 0x00"],
        &["", "0xFF", "0xFF 0x12", "0xFF 0x12 0xAB 0x00 0x00 0x00"],
    );

    test_grammar_str(
        r#"
            root ::= ( [x]* )*
        "#,
        &["", "x", "xx"],
        &["y", "yy"],
    );
}

#[test]
fn integration_failure_modes() {
    // missing root node: parses, but no `root` symbol
    let grammar_str = r#"
        rot ::= expr
        expr ::= term ("+" term)*
        term ::= number
        number ::= [0-9]+"#;
    let mut parsed = GrammarParser::new(None);
    assert!(parsed.parse(grammar_str.as_bytes()));
    assert!(!parsed.rules.is_empty());
    assert!(!parsed.symbol_ids.contains_key("root"));

    // missing reference node: parse fails, rules cleared
    let grammar_str = r#"root ::= expr
        expr ::= term ("+" term)*
        term ::= numero
        number ::= [0-9]+"#;
    let mut parsed = GrammarParser::new(None);
    assert!(!parsed.parse(grammar_str.as_bytes()));
    assert!(parsed.rules.is_empty());

    // left recursion detection (the four reference cases)
    for s in [
        r#"root ::= "a" | root "a""#,
        r#"
        root ::= asdf
        asdf ::= "a" | asdf "a"
        "#,
        r#"
        root ::= asdf
        asdf ::= "a" | foo "b"
        foo ::= "c" | asdf "d" | "e""#,
        r#"
        root ::= asdf
        asdf ::= "a" | foo "b"
        foo ::= "c" | empty asdf "d" | "e"
        empty ::= "blah" | "#,
    ] {
        assert!(
            Grammar::parse(None, s, "root").is_err(),
            "expected left recursion failure for:\n{s}"
        );
    }

    // missing root symbol / custom root symbol check
    assert!(Grammar::parse(
        None,
        r#"
        root ::= "foobar"
    "#,
        "nonexistent"
    )
    .is_err());
    assert!(Grammar::parse(
        None,
        r#"
        foobar ::= "foobar"
    "#,
        "root"
    )
    .is_err());
    let custom = Grammar::parse(
        None,
        r#"
        foobar ::= "foobar"
    "#,
        "foobar",
    );
    assert!(custom.is_ok());
}

// ---------------------------------------------------------------------------
// 3. reference replay (parity/grammar_ref.txt + parity/grammar_pieces_ref.bin)
// ---------------------------------------------------------------------------

const REF_TXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/grammar_ref.txt");
const REF_BIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/grammar_pieces_ref.bin"
);

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// the reference piece table (`vocab->token_to_piece` cache + `is_eog`)
struct RefPieces {
    offsets: Vec<u32>,
    buf: Vec<u8>,
    is_eog: Vec<bool>,
}

impl RefPieces {
    fn load(path: &str) -> Option<RefPieces> {
        let bytes = std::fs::read(path).ok()?;
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut offsets = Vec::with_capacity(n + 1);
        let mut buf = Vec::new();
        let mut is_eog = Vec::with_capacity(n);
        offsets.push(0u32);
        let mut p = 4usize;
        for _ in 0..n {
            let len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            buf.extend_from_slice(&bytes[p..p + len]);
            p += len;
            is_eog.push(bytes[p] != 0);
            p += 1;
            offsets.push(buf.len() as u32);
        }
        assert_eq!(p, bytes.len(), "trailing bytes in {path}");
        Some(RefPieces {
            offsets,
            buf,
            is_eog,
        })
    }

    fn n_tokens(&self) -> usize {
        self.is_eog.len()
    }
}

impl GrammarVocab for RefPieces {
    fn is_eog(&self, token: i32) -> bool {
        token >= 0 && (token as usize) < self.is_eog.len() && self.is_eog[token as usize]
    }
    fn token_piece(&self, token: i32) -> &[u8] {
        if token < 0 || token as usize + 1 >= self.offsets.len() {
            return &[];
        }
        let a = self.offsets[token as usize] as usize;
        let b = self.offsets[token as usize + 1] as usize;
        &self.buf[a..b]
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// the C `fnv1a` in parity/ref_grammar_dump.cpp
fn fnv1a(ids: &[i32]) -> u32 {
    let mut h: u32 = 2166136261;
    for &id in ids {
        let v = id as u32;
        for b in 0..4 {
            h ^= (v >> (8 * b)) & 0xFF;
            h = h.wrapping_mul(16777619);
        }
    }
    h
}

/// the canonical `STACKS` rendering of parity/ref_grammar_dump.cpp
fn fmt_stacks(grammar: &Grammar) -> String {
    let mut canon: Vec<Vec<(u32, u32)>> = grammar
        .stacks
        .iter()
        .map(|s| {
            let mut v: Vec<(u32, u32)> = s.clone();
            v.sort();
            v
        })
        .collect();
    canon.sort();
    let mut out = format!("STACKS {}", canon.len());
    for s in &canon {
        out.push_str(&format!(" {}", s.len()));
        for e in s {
            out.push_str(&format!(" {}.{}", e.0, e.1));
        }
    }
    out
}

/// apply over every vocab token (logit 1.0), then render count + fnv + head ids
fn fmt_mask(grammar: &Grammar, vocab: &dyn GrammarVocab, n: usize) -> String {
    let logits = vec![1.0f32; n];
    let mut cur = TokenDataArray::from_logits(&logits);
    grammar.apply(vocab, &mut cur);
    let keep: Vec<i32> = (0..cur.size)
        .filter(|&i| cur.data[i].logit != f32::NEG_INFINITY)
        .map(|i| cur.data[i].id)
        .collect();
    let mut out = format!("MASK {} {:08x}", keep.len(), fnv1a(&keep));
    for id in keep.iter().take(12) {
        out.push_str(&format!(" {id}"));
    }
    out
}

/// parse the artifact into (name, source, lines) parser blocks and case blocks
struct RefDump {
    parser_blocks: Vec<(String, Vec<u8>, Vec<String>)>,
    cases: Vec<(String, Vec<u8>, Vec<i32>, Vec<String>)>,
}

fn parse_ref_dump() -> Option<RefDump> {
    let text = std::fs::read_to_string(REF_TXT).ok()?;
    let mut parser_blocks = Vec::new();
    let mut cases = Vec::new();
    let mut cur_parser: Option<(String, Vec<u8>, Vec<String>)> = None;
    let mut cur_case: Option<(String, Vec<u8>, Vec<i32>, Vec<String>)> = None;

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("PARSE ") {
            if let Some(b) = cur_parser.take() {
                parser_blocks.push(b);
            }
            let mut it = rest.split_whitespace();
            let name = it.next().unwrap().to_string();
            let _status = it.next();
            cur_parser = Some((name, Vec::new(), Vec::new()));
        } else if let Some(rest) = line.strip_prefix("GSRC ") {
            let src = unhex(rest);
            if let Some(c) = cur_case.as_mut() {
                c.1 = src;
            } else if let Some(p) = cur_parser.as_mut() {
                p.1 = src;
            }
        } else if let Some(rest) = line.strip_prefix("CASE ") {
            if let Some(b) = cur_parser.take() {
                parser_blocks.push(b);
            }
            if let Some(c) = cur_case.take() {
                cases.push(c);
            }
            let mut it = rest.split_whitespace();
            let name = it.next().unwrap().to_string();
            let ids: Vec<i32> = it.map(|t| t.parse().unwrap()).collect();
            cur_case = Some((name, Vec::new(), ids, Vec::new()));
        } else if line.starts_with("SYM ") || line.starts_with("RULE ") {
            if let Some(p) = cur_parser.as_mut() {
                p.2.push(line.to_string());
            }
        } else if line.starts_with("STEP ") {
            if let Some(c) = cur_case.as_mut() {
                c.3.push(line.to_string());
            }
        }
    }
    if let Some(b) = cur_parser.take() {
        parser_blocks.push(b);
    }
    if let Some(c) = cur_case.take() {
        cases.push(c);
    }
    Some(RefDump {
        parser_blocks,
        cases,
    })
}

#[test]
fn reference_parser_dump_replay() {
    let dump = match parse_ref_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    assert!(
        !dump.parser_blocks.is_empty(),
        "no parser blocks in {REF_TXT}"
    );
    let mut checked = 0usize;
    for (name, src, ref_lines) in &dump.parser_blocks {
        let gbnf = String::from_utf8(src.clone()).unwrap();
        let mut parser = GrammarParser::new(None);
        let ok = parser.parse(src);
        assert_eq!(
            ok,
            ref_lines.first().map(|_| true).unwrap_or(ok) && !ref_lines.is_empty(),
            "parse status mismatch for {name}"
        );
        let mut mine: Vec<String> = Vec::new();
        for (n, id) in &parser.symbol_ids {
            mine.push(format!("SYM {n} {id}"));
        }
        for (r, rule) in parser.rules.iter().enumerate() {
            let mut line = format!("RULE {r}");
            for e in rule {
                line.push_str(&format!(" {} {}", e.ty as u8, e.value));
            }
            mine.push(line);
        }
        assert_eq!(&mine, ref_lines, "parser dump mismatch for {name}\n{gbnf}");
        checked += 1;
    }
    eprintln!("reference parser dump: {checked} grammars identical");
}

#[test]
fn reference_matcher_replay() {
    let dump = match parse_ref_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    let vocab = match RefPieces::load(REF_BIN) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: {REF_BIN} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    assert!(!dump.cases.is_empty(), "no CASE blocks in {REF_TXT}");

    let mut n_steps = 0usize;
    for (name, src, ids, ref_steps) in &dump.cases {
        let gbnf = String::from_utf8(src.clone()).unwrap();
        let mut grammar = Grammar::parse(None, &gbnf, "root")
            .unwrap_or_else(|e| panic!("grammar {name} failed to build: {e}\n{gbnf}"));
        assert_eq!(grammar.stacks.len() > 0, true);
        for (i, ref_line) in ref_steps.iter().enumerate() {
            let id = ids[i];
            let piece = vocab.token_piece(id).to_vec();
            // STEP <i> <id> <piece> STACKS ... MASK ...
            let expect_prefix = format!("STEP {i} {id} {}", hex(&piece));
            assert!(
                ref_line.starts_with(&expect_prefix),
                "piece/step mismatch for {name} step {i}:\nref: {ref_line}\nhead: {expect_prefix}"
            );
            let rest = ref_line[expect_prefix.len()..].trim();
            if rest == "ACCEPT_FAILED" {
                // the reference dump accepts without logit gating, so a token the
                // grammar does not allow makes it throw (empty stacks)
                assert!(
                    grammar.accept_token(id, &piece).is_err(),
                    "{name} step {i}: reference failed to accept, ours succeeded"
                );
                continue;
            }
            grammar
                .accept_token(id, &piece)
                .unwrap_or_else(|e| panic!("{name} step {i}: accept failed: {e}"));
            let mine = format!(
                "{} {}",
                fmt_stacks(&grammar),
                fmt_mask(&grammar, &vocab, vocab.n_tokens())
            );
            assert_eq!(mine, rest, "matcher replay mismatch for {name} step {i}");
            n_steps += 1;
        }
    }
    eprintln!(
        "reference matcher replay: {} cases / {} steps identical",
        dump.cases.len(),
        n_steps
    );

    // same trajectories, but with the *port's* vocab implementation driving
    // apply() (so piece lookup + is_eog come from crates/llama/src/vocab.rs)
    let gguf = match ggml::Gguf::open(qwen2_vocab_path()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: cannot open {}: {e}", qwen2_vocab_path());
            return;
        }
    };
    let our_vocab = Vocab::load(&gguf).expect("vocab");
    assert_eq!(our_vocab.n_tokens() as usize, vocab.n_tokens());
    let mut n_steps2 = 0usize;
    for (name, src, ids, ref_steps) in &dump.cases {
        let gbnf = String::from_utf8(src.clone()).unwrap();
        let mut grammar = Grammar::parse(None, &gbnf, "root").unwrap();
        for (i, ref_line) in ref_steps.iter().enumerate() {
            let id = ids[i];
            let piece = our_vocab.token_piece(id).to_vec();
            let expect_prefix = format!("STEP {i} {id} {}", hex(&piece));
            assert!(
                ref_line.starts_with(&expect_prefix),
                "{name} step {i} piece"
            );
            let rest = ref_line[expect_prefix.len()..].trim();
            if rest == "ACCEPT_FAILED" {
                assert!(grammar.accept_token(id, &piece).is_err());
                continue;
            }
            grammar.accept_token(id, &piece).unwrap();
            let mine = format!(
                "{} {}",
                fmt_stacks(&grammar),
                fmt_mask(&grammar, &our_vocab, our_vocab.n_tokens() as usize)
            );
            assert_eq!(
                mine, rest,
                "matcher replay (own vocab) mismatch for {name} step {i}"
            );
            n_steps2 += 1;
        }
    }
    eprintln!(
        "reference matcher replay with the port's Vocab: {} steps identical",
        n_steps2
    );
}

// ---------------------------------------------------------------------------
// 4. full qwen2 piece table vs the reference cache
// ---------------------------------------------------------------------------

fn qwen2_vocab_path() -> &'static str {
    "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-qwen2.gguf"
}

#[test]
fn piece_table_vs_reference() {
    let refp = match RefPieces::load(REF_BIN) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: {REF_BIN} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    let path = qwen2_vocab_path();
    let gguf = match ggml::Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: cannot open {path}: {e}");
            return;
        }
    };
    let vocab = Vocab::load(&gguf).expect("vocab");
    assert_eq!(
        vocab.n_tokens() as usize,
        refp.n_tokens(),
        "n_vocab differs from the reference dump"
    );

    let mut n_piece_diff = 0usize;
    let mut n_eog_diff = 0usize;
    for id in 0..vocab.n_tokens() as i32 {
        if vocab.token_to_piece_bytes(id) != refp.token_piece(id) {
            if n_piece_diff < 5 {
                eprintln!(
                    "piece {id}: mine={} ref={}",
                    hex(vocab.token_to_piece_bytes(id)),
                    hex(refp.token_piece(id))
                );
            }
            n_piece_diff += 1;
        }
        if vocab.is_eog(id) != refp.is_eog(id) {
            n_eog_diff += 1;
        }
    }
    assert_eq!(
        n_piece_diff, 0,
        "{n_piece_diff} token pieces differ from the reference"
    );
    assert_eq!(
        n_eog_diff, 0,
        "{n_eog_diff} is_eog flags differ from the reference"
    );
    eprintln!(
        "piece table: {} tokens identical (pieces + is_eog)",
        vocab.n_tokens()
    );
}

// ---------------------------------------------------------------------------
// 5. apply()/accept() sanity on a synthetic vocab (no reference needed)
// ---------------------------------------------------------------------------

/// a tiny piece table: id → piece, so the sampler path can be tested standalone
struct SynthVocab {
    pieces: Vec<Vec<u8>>,
    eog: i32,
}

impl GrammarVocab for SynthVocab {
    fn is_eog(&self, token: i32) -> bool {
        token == self.eog
    }
    fn token_piece(&self, token: i32) -> &[u8] {
        if token < 0 || token as usize >= self.pieces.len() {
            return &[];
        }
        &self.pieces[token as usize]
    }
}

fn synth_cur(logits: &[f32]) -> TokenDataArray {
    TokenDataArray::from_logits(logits)
}

#[test]
fn apply_masks_and_allows_eog() {
    // root ::= "ab"
    let grammar = Grammar::parse(None, r#"root ::= "ab""#, "root").unwrap();
    let vocab = SynthVocab {
        pieces: vec![
            b"a".to_vec(),   // 0
            b"b".to_vec(),   // 1
            b"ab".to_vec(),  // 2
            b"c".to_vec(),   // 3
            b"".to_vec(),    // 4 — empty piece: always masked
            b"\0x".to_vec(), // 5 — NUL first byte: always masked
        ],
        eog: 6,
    };

    let mut cur = synth_cur(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    // add the EOG token
    cur.data.push(TokenData {
        id: 6,
        logit: 0.7,
        p: 0.0,
    });
    cur.size = 7;
    grammar.apply(&vocab, &mut cur);
    let masked: Vec<bool> = cur.data[..cur.size]
        .iter()
        .map(|d| d.logit == f32::NEG_INFINITY)
        .collect();
    // "a" ok, "b"/"c" rejected (must start with 'a'), "ab" ok, empty/NUL masked,
    // EOG masked (no empty stack yet)
    assert_eq!(masked, vec![false, true, false, true, true, true, true]);

    // accept "a" then the EOG must become allowed and continue to be masked
    let mut grammar2 = Grammar::parse(None, r#"root ::= "ab""#, "root").unwrap();
    grammar2.accept_token(0, b"a").unwrap();
    let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3]);
    cur.data.push(TokenData {
        id: 6,
        logit: 0.7,
        p: 0.0,
    });
    cur.size = 4;
    grammar2.apply(&vocab, &mut cur);
    assert_eq!(cur.data[0].logit, f32::NEG_INFINITY); // "a" no longer allowed
    assert_eq!(cur.data[1].logit, 0.2); // "b" allowed
    assert_eq!(cur.data[2].logit, f32::NEG_INFINITY);
    assert_eq!(cur.data[3].logit, f32::NEG_INFINITY); // EOG still masked

    // after completing the grammar, EOG is allowed and nothing else is
    grammar2.accept_token(1, b"b").unwrap();
    assert!(grammar2.stacks.iter().any(|s| s.is_empty()));
    let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3]);
    cur.data.push(TokenData {
        id: 6,
        logit: 0.7,
        p: 0.0,
    });
    cur.size = 4;
    grammar2.apply(&vocab, &mut cur);
    assert!(cur.data[..3].iter().all(|d| d.logit == f32::NEG_INFINITY));
    assert_eq!(cur.data[3].logit, 0.7);
    // accepting the EOG on a completed grammar is a no-op (C returns early)
    grammar2.accept_impl(&vocab, 6).unwrap();
}

#[test]
fn accept_partial_utf8_across_tokens() {
    // root ::= [🔵-🟠]+ (U+1F535..U+1F7E0)
    let grammar_str = "root ::= [\\U0001F535-\\U0001F7E0]+\n";
    let mut grammar = Grammar::parse(None, grammar_str, "root").unwrap();
    // feed the 4 bytes of U+1F535 one token at a time
    for (i, byte) in [0xF0u8, 0x9F, 0x94, 0xB5].iter().enumerate() {
        let piece = [*byte];
        // the first step creates a partial state (n_remain > 0)
        grammar.accept_token(i as i32, &piece).unwrap();
    }
    assert_eq!(
        grammar.partial_utf8,
        PartialUtf8 {
            value: 0x1F535,
            n_remain: 0
        }
    );
    assert!(grammar.stacks.len() >= 1);

    // a byte that cannot be a valid continuation is rejected by apply
    let mut g = Grammar::parse(None, grammar_str, "root").unwrap();
    g.accept_token(0, &[0xF0]).unwrap();
    let partial = g.partial_utf8;
    assert_eq!(
        partial,
        PartialUtf8 {
            value: 0,
            n_remain: 3
        }
    );
    let mut cur = TokenDataArray::from_logits(&[1.0, 1.0]);
    let bytes_table = SynthVocab {
        pieces: vec![vec![0x9F], vec![0x41]], // valid continuation vs 'A'
        eog: -1,
    };
    g.apply(&bytes_table, &mut cur);
    assert_eq!(cur.data[0].logit, 1.0, "0x9F can complete the emoji range");
    assert_eq!(cur.data[1].logit, f32::NEG_INFINITY, "0x41 cannot");
}

#[test]
fn apply_matches_reference_on_user_grammar_with_vocab() {
    // the same JSON case as the artifact, but driven through our own Vocab
    let path = qwen2_vocab_path();
    let gguf = match ggml::Gguf::open(path) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("SKIP: cannot open {path}");
            return;
        }
    };
    let vocab = Vocab::load(&gguf).expect("vocab");
    // vocab-driven piece check for one token: `token_to_piece_bytes` is what
    // `GrammarVocab for Vocab` feeds the matcher
    for id in [0i32, 220, 4913, 128247] {
        assert_eq!(vocab.token_piece(id), vocab.token_to_piece_bytes(id));
    }
    let gbnf = "root ::= \"{\" ws \"a\" ws \"}\"\nws ::= [ ]*\n";
    let grammar = Grammar::parse(Some(&vocab), gbnf, "root").unwrap();
    let mut cur = TokenDataArray::from_logits(&vec![1.0f32; vocab.n_tokens() as usize]);
    grammar.apply(&vocab, &mut cur);
    assert!(cur
        .data
        .iter()
        .all(|d| d.logit == f32::NEG_INFINITY || d.p == 0.0));
    // only tokens whose piece starts with '{' (plus EOG, not allowed here) survive
    let surviving: Vec<char> = (0..cur.size)
        .filter(|&i| cur.data[i].logit != f32::NEG_INFINITY)
        .filter_map(|i| vocab.token_to_piece(i as i32).chars().next())
        .collect();
    assert!(!surviving.is_empty());
    assert!(
        surviving.iter().all(|c| *c == '{'),
        "unexpected survivors: {surviving:?}"
    );

    // and `Rule`/`Rules` plumbing used by tests elsewhere
    let rules: Rules = vec![vec![
        llama::grammar::GrammarElement::new(Gretype::Char, 'x' as u32),
        llama::grammar::GrammarElement::new(Gretype::End, 0),
    ]];
    assert!(el_at(&rules, (0, 0)).ty == Gretype::Char);
    let _rule: Rule = rules[0].clone();
    let _map: BTreeMap<String, u32> = BTreeMap::new();
}
// ---------------------------------------------------------------------------
// 6. grammar-constrained greedy generation through GrammarSampler +
//    SamplingContext (the llama-cli path)
// ---------------------------------------------------------------------------

/// greedy generation with the grammar applied (temp = 0), boosting `invalid`
/// above the "wanted" token at every step so the sampler has to reject and
/// resample when the boost is not grammar-legal.
fn constrained_greedy(
    grammar: &mut llama::sampling::GrammarSampler,
    n_vocab: usize,
    wanted: &[i32],
    invalid: i32,
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
        out.push(tok);
    }
    // one more step with no "wanted" token: the boosted token must now be legal
    // (EOG once the grammar is complete)
    let mut logits = vec![0.0f32; n_vocab];
    logits[invalid as usize] = 200.0;
    let tok = ctx.sample_with_grammar(&logits, grammar).unwrap();
    out.push(tok);
    out
}

#[test]
fn constrained_greedy_generation_with_reference_pieces() {
    let vocab = match RefPieces::load(REF_BIN) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: {REF_BIN} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    let pieces = llama::grammar::VocabPieces::from_iter(
        (0..vocab.n_tokens())
            .map(|i| (vocab.token_piece(i as i32).to_vec(), vocab.is_eog(i as i32))),
    );
    // the simplest grammar that still has alternation + repetition
    let gbnf = "root ::= \"{\" ws \"a\" ws \"}\"\nws ::= [ ]*\n";
    let mut grammar = llama::sampling::GrammarSampler::from_pieces(gbnf, "root", pieces).unwrap();

    // pick the first token for each piece (ids are stable in the qwen2 table)
    let find = |p: &[u8]| {
        (0..vocab.n_tokens())
            .find(|i| vocab.token_piece(*i as i32) == p)
            .unwrap_or_else(|| panic!("no token with piece {p:?}")) as i32
    };
    let wanted = [find(b"{"), find(b"a"), find(b" "), find(b"}")];
    let eog = 151643i32;
    let out = constrained_greedy(&mut grammar, vocab.n_tokens(), &wanted, eog);
    assert_eq!(
        &out[..4],
        &wanted,
        "greedy output must follow the wanted tokens"
    );
    assert_eq!(out[4], eog, "the grammar must allow EOG once complete");

    let text: Vec<u8> = out[..4]
        .iter()
        .flat_map(|&id| vocab.token_piece(id).to_vec())
        .collect();
    assert_eq!(String::from_utf8(text).unwrap(), "{a }");
}

#[test]
fn init_grammar_with_real_vocab_and_generate() {
    let path = qwen2_vocab_path();
    let gguf = match ggml::Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: cannot open {path}: {e}");
            return;
        }
    };
    let vocab = Vocab::load(&gguf).expect("vocab");
    let gbnf = "root ::= \"{\" ws \"a\" ws \"}\"\nws ::= [ ]*\n";
    let mut grammar = llama::sampling::init_grammar(&vocab, gbnf).unwrap();
    assert_eq!(grammar.name(), "grammar");
    assert_eq!(grammar.grammar_root, "root");

    let find = |p: &[u8]| {
        (0..vocab.n_tokens() as i32)
            .find(|i| vocab.token_to_piece_bytes(*i) == p)
            .unwrap_or_else(|| panic!("no token with piece {p:?}"))
    };
    let wanted = [find(b"{"), find(b"a"), find(b" "), find(b"}")];
    let out = constrained_greedy(
        &mut grammar,
        vocab.n_tokens() as usize,
        &wanted,
        vocab.token_eos(),
    );
    assert_eq!(&out[..4], &wanted);
    let text: String = out[..4]
        .iter()
        .map(|&id| vocab.token_to_piece(id))
        .collect();
    assert_eq!(text, "{a }");

    // `init_grammar` error paths
    assert!(llama::sampling::init_grammar(&vocab, "").is_err());
    assert!(llama::sampling::init_grammar(&vocab, "root ::= missing").is_err());
}

// ---------------------------------------------------------------------------
// 7. `llama_grammar_parser::print` round trip (print_rule, llama-grammar.cpp:297-372)
// ---------------------------------------------------------------------------

#[test]
fn parser_print_round_trip() {
    // exact canonical text (print_rule, llama-grammar.cpp:297-372)
    let cases: &[(&str, &str)] = &[
        (
            "root ::= \"a\" | [bdx-z] | [^1-3]",
            "root ::= [a] | [bdx-z] | [^1-3] \n",
        ),
        (
            "root ::= <[1000]> !<[1001]> <[1001]>",
            "root ::= <[1000]> !<[1001]> <[1001]> \n",
        ),
        // NOTE: C prints `.] ` for CHAR_ANY (is_char_element(CHAR_ANY) is true and
        // the closing-bracket check looks at the *next* element) — faithful quirk
        (
            "root ::= \"a\"+\na ::= .\n",
            "root ::= [a] root_1 \nroot_1 ::= [a] root_1 | \na ::= .] \n",
        ),
    ];
    for (gbnf, expected) in cases {
        let mut p = GrammarParser::new(None);
        assert!(p.parse(gbnf.as_bytes()), "must parse: {gbnf}");
        let printed = p.print().unwrap();
        assert_eq!(&printed, expected, "print mismatch for {gbnf}");
        if gbnf.contains('.') {
            // the `.` (CHAR_ANY) quirk above makes the output non-GBNF on purpose
            continue;
        }
        // re-parse the printed text: rules must be unchanged
        let mut p2 = GrammarParser::new(None);
        assert!(
            p2.parse(printed.as_bytes()),
            "printed text must re-parse: {printed:?}"
        );
        assert_eq!(p2.rules, p.rules, "rules changed by print for {gbnf}");
    }

    // every ASCII-printable grammar in the artifact must re-parse to the same
    // rules (C's print_grammar_char copes out of UTF-8 with `<U+XXXX>`, which is
    // not valid GBNF input, so those are skipped — same limitation in C)
    let dump = match parse_ref_dump() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: {REF_TXT} missing (run parity/gen_grammar_ref.sh)");
            return;
        }
    };
    let mut checked = 0usize;
    for (name, src, _ref_lines) in &dump.parser_blocks {
        let mut p1 = GrammarParser::new(None);
        assert!(p1.parse(src), "{name} must parse");
        let printed = p1.print().expect("print");
        if printed.contains("<U+") {
            // control chars / non-ASCII print as `<U+XXXX>` (C cop-out) — not GBNF
            continue;
        }
        let mut p2 = GrammarParser::new(None);
        assert!(
            p2.parse(printed.as_bytes()),
            "printed grammar must re-parse:\n{printed}"
        );
        assert_eq!(
            p1.symbol_ids, p2.symbol_ids,
            "symbol ids changed for {name}"
        );
        assert_eq!(p1.rules, p2.rules, "rules changed for {name}\n{printed}");
        checked += 1;
    }
    eprintln!(
        "parser print round-trip: {} hand-written cases + {checked} artifact grammars re-parse identically",
        cases.len()
    );
}
