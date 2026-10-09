//! K2-Horizon splitter + PLaMo-3 pre-segmentation parity — the port side of
//! `parity/ref_k2_plamo_split.c` (the probe compiles the NEW reference's own
//! unicode.cpp from /home/jeffrey/llm/llama.cpp-next @c35b66744; its output
//! is checked in as `parity/ref_k2_split.golden` / `parity/ref_plamo3_
//! segments.golden`, produced with `bash parity/ref_k2_plamo_split.sh`).
//!
//! The corpora are duplicated verbatim in the probe source — keeping them in
//! lockstep is the point of the test (a divergence in either file fails the
//! comparison here).

use llama::unicode;
use llama::vocab::plamo3_segments;

/// the K2-Horizon corpus (see parity/ref_k2_plamo_split.c K2_CORPUS)
const K2_CORPUS: &[&str] = &[
    "Amy\u{200C}khaham",
    "a\u{200D}b",
    "code\u{0301}e\u{0301}",
    "12345",
    "12",
    "'S 're 'Ll",
    "'\u{017F}",
    " !x??\r\n",
    "a  b   c",
    "\u{200C}leading",
    "\u{6C49}\u{5B57}abc123",
];

/// the PLaMo-3 corpus (see parity/ref_k2_plamo_split.c PLAMO3_CORPUS)
const PLAMO3_CORPUS: &[&str] = &[
    "<|plamo:chat|>hello<|plamo:eos|>",
    "<|plamo:|>x",
    "<|plamo:\u{001D}sep|>y",
    "<|plamo:notspecial",
    "<|plamo:a\u{1F600}|>z",
    "aaaa",
    "aaa",
    "ababab",
    "   x",
    "\u{FEFF}keep",
    "pre\u{EE00}post",
    "<|plamo:fim_prefix|>\u{EE00}<|plamo:fim_suffix|>",
];

const K2_REGEX: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

fn render_cpts(cpts: &[u32]) -> String {
    let mut s = String::new();
    for &c in cpts {
        s.push_str(&format!("U+{c:04X} "));
    }
    // the golden lines carry no trailing space (the probe's printf format)
    s.trim_end().to_string()
}

fn golden(rel: &str) -> Vec<String> {
    let path = format!(
        "{}/../../parity/{rel}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (regenerate with parity/ref_k2_plamo_split.sh)"))
        .lines()
        .map(|l| l.trim_end().to_string())
        .collect()
}

#[test]
fn k2_horizon_splitter_matches_reference() {
    let mut got: Vec<String> = Vec::new();
    for text in K2_CORPUS {
        let exprs = vec![K2_REGEX.to_string()];
        for word in unicode::regex_split(text, &exprs, false) {
            got.push(render_cpts(&unicode::cpts_from_utf8(word.as_bytes())));
        }
        got.push("---".into());
    }
    let want = golden("ref_k2_split.golden");
    assert_eq!(got, want, "K2-Horizon split diverged from the reference");
}

#[test]
fn plamo3_pre_segmentation_matches_reference() {
    let mut got: Vec<String> = Vec::new();
    for text in PLAMO3_CORPUS {
        for (a, b) in plamo3_segments(&unicode::cpts_from_utf8(text.as_bytes())) {
            let seg = &unicode::cpts_from_utf8(text.as_bytes())[a..b];
            got.push(render_cpts(seg));
        }
        got.push("---".into());
    }
    let want = golden("ref_plamo3_segments.golden");
    assert_eq!(
        got, want,
        "PLaMo-3 pre-segmentation diverged from the reference"
    );
}
