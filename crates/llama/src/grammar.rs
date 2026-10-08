//! `src/llama-grammar.cpp` + `src/llama-grammar.h` — GBNF grammar parser and
//! pushdown-stack matcher (1:1 port of llama.cpp `bd4f514db1`, 1526 + 194 lines).
//!
//! Every item carries the C line range it mirrors. The port is verbatim in
//! behaviour, including the corners that are easy to get wrong:
//!
//! * byte-level parsing: the grammar source is copied with a NUL sentinel, so
//!   `parse_space`/`parse_hex`/`parse_char` reproduce C's `*pos` / `pos[1]`
//!   semantics exactly (`decode_utf8` :19 and :34);
//! * the repetition rewrite `S{m,n} → S…S S'(n−m)` (`handle_repetitions`
//!   :463-528) and the `MAX_REPETITION_THRESHOLD` guard (:13, failure cases in
//!   the reference tests);
//! * the stack machine: `advance_stack` :856, `accept`/`accept_chr` :1020/:1046,
//!   `accept_token` :1472, `apply_impl` :1355, `reject_candidates*` :940/:1057
//!   — including the *candidate pointer walk* (C keeps `const uint32_t *
//!   code_points` and moves it `+1` / `-1` while recursing; here a candidate
//!   carries `(decoded_index, cp_offset)` which is exactly equivalent because
//!   only `index` (for masking) and the terminating `0` of the code point array
//!   are ever observed).
//!
//! Representation choices (documented where they can be observed):
//!
//! * a stack element is `(rule_index, element_index)` instead of
//!   `const llama_grammar_element *` — C pointer identity *is* element identity,
//!   so the two are isomorphic and every set/dedup operation (`seen` in
//!   `advance_stack`, `std::find` against `new_stacks`) keeps its meaning;
//! * C's pointer *ordering* only influences the iteration order inside
//!   `advance_stack`, never the resulting stack set (the mask is applied per
//!   candidate index and `allow_eog` only asks "is any stack empty"), so a
//!   deterministic `BTreeSet` order is used;
//! * `lazy` grammars need `std::regex` trigger patterns, which this workspace
//!   has no dependency for — the trigger fields are kept so `apply`/`accept`
//!   keep the reference structure, but the lazy path is not reachable
//!   (see the report: gap).
//!
//! Not ported here: `llama_grammar_trigger_pattern::find` (:378-409, needs
//! `std::regex`), the trigger-buffer replay in `llama_grammar_accept_impl`
//! (:1403-1443, lazy-only), `llama_grammar_clone_impl` (:1325, C rewrites
//! pointers — trivially `Clone` here).

use std::collections::{BTreeMap, BTreeSet};

use crate::sampling::{LlamaToken, TokenDataArray};
use crate::vocab::Vocab;

/// `MAX_REPETITION_THRESHOLD` (llama-grammar.cpp:13)
pub const MAX_REPETITION_THRESHOLD: u64 = 2000;

// ---------------------------------------------------------------------------
// llama-grammar.h types
// ---------------------------------------------------------------------------

/// `enum llama_gretype` (llama-grammar.h:13-45)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Gretype {
    /// end of rule definition
    End = 0,
    /// start of alternate definition for rule
    Alt = 1,
    /// non-terminal element: reference to rule
    RuleRef = 2,
    /// terminal element: character (code point)
    Char = 3,
    /// inverse char(s) (`[^a]`, `[^a-b]`, `[^abc]`)
    CharNot = 4,
    /// modifies a preceding CHAR/CHAR_ALT to be an inclusive range (`[a-z]`)
    CharRngUpper = 5,
    /// modifies a preceding CHAR/CHAR_RNG_UPPER to add an alternate char (`[ab]`)
    CharAlt = 6,
    /// any character (`.`)
    CharAny = 7,
    /// terminal element: token (`<[token-id]>`)
    Token = 8,
    /// inverse token (`!<[token-id]>`)
    TokenNot = 9,
}

impl Gretype {
    /// `print_rule_binary` / `type_str` spelling (llama-grammar.cpp:252-265)
    pub fn name(self) -> &'static str {
        match self {
            Gretype::End => "END",
            Gretype::Alt => "ALT",
            Gretype::RuleRef => "RULE_REF",
            Gretype::Char => "CHAR",
            Gretype::CharNot => "CHAR_NOT",
            Gretype::CharRngUpper => "CHAR_RNG_UPPER",
            Gretype::CharAlt => "CHAR_ALT",
            Gretype::CharAny => "CHAR_ANY",
            Gretype::Token => "TOKEN",
            Gretype::TokenNot => "TOKEN_NOT",
        }
    }
}

/// `struct llama_grammar_element` (llama-grammar.h:47-50)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GrammarElement {
    pub ty: Gretype,
    /// Unicode code point, rule ID, or token ID
    pub value: u32,
}

impl GrammarElement {
    pub fn new(ty: Gretype, value: u32) -> Self {
        GrammarElement { ty, value }
    }
}

/// `struct llama_partial_utf8` (llama-grammar.h:52-55)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PartialUtf8 {
    /// bit value so far (unshifted)
    pub value: u32,
    /// num bytes remaining; -1 indicates invalid sequence
    pub n_remain: i32,
}

/// `llama_grammar_rule` = `std::vector<llama_grammar_element>`
pub type Rule = Vec<GrammarElement>;
/// `llama_grammar_rules`
pub type Rules = Vec<Rule>;

/// `const llama_grammar_element *` — `(rule index, element index)`
pub type ElemRef = (u32, u32);
/// `llama_grammar_stack`
pub type Stack = Vec<ElemRef>;
/// `llama_grammar_stacks`
pub type Stacks = Vec<Stack>;

// ---------------------------------------------------------------------------
// helpers (:18-372)
// ---------------------------------------------------------------------------

/// NUL-terminated byte accessor: C reads `*pos` on a `c_str()` buffer, i.e. a
/// NUL byte (and everything past the end) reads as 0.
#[inline]
fn b(src: &[u8], pos: usize) -> u8 {
    if pos < src.len() {
        src[pos]
    } else {
        0
    }
}

#[inline]
fn next(p: ElemRef) -> ElemRef {
    (p.0, p.1 + 1)
}

#[inline]
fn next2(p: ElemRef) -> ElemRef {
    (p.0, p.1 + 2)
}

/// `elem_at` — element lookup that mirrors "the pointer is valid"; out of range
/// reads return `END` (unreachable for well-formed rules).
#[inline]
pub fn el_at(rules: &Rules, pos: ElemRef) -> GrammarElement {
    match rules
        .get(pos.0 as usize)
        .and_then(|r| r.get(pos.1 as usize))
    {
        Some(e) => *e,
        None => GrammarElement::new(Gretype::End, 0),
    }
}

/// `decode_utf8(const char *)` (llama-grammar.cpp:19-32) — single code point,
/// used by `parse_char`.
fn decode_utf8_cpt(src: &[u8], pos: usize) -> (u32, usize) {
    // NOTE: assumes valid utf8 (but checks for overrun)
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    let first_byte = b(src, pos);
    let highbits = first_byte >> 4;
    let len = LOOKUP[highbits as usize];
    let mask = ((1u32 << (8 - len)) - 1) as u8;
    let mut value = (first_byte & mask) as u32;
    let end = pos + len; // may overrun!
    let mut p = pos + 1;
    while p < end && b(src, p) != 0 {
        value = (value << 6) + ((b(src, p) & 0x3F) as u32);
        p += 1;
    }
    (value, p)
}

/// `decode_utf8(const std::string &, llama_partial_utf8)` (:34-92).
///
/// Returns the decoded code points with the terminating `0` appended (C does
/// `code_points.push_back(0)`) and the updated partial state. Note the C
/// iteration stops at an embedded NUL byte, reproduced here.
pub fn decode_utf8(src: &[u8], partial_start: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    const LOOKUP: [i32; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];
    let mut pos = 0usize;
    let mut code_points: Vec<u32> = Vec::with_capacity(src.len() + 1);

    let mut value = partial_start.value;
    let mut n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while b(src, pos) != 0 && n_remain > 0 {
        let next_byte = b(src, pos);
        if (next_byte >> 6) != 2 {
            // invalid sequence, abort
            code_points.push(0);
            return (
                code_points,
                PartialUtf8 {
                    value: 0,
                    n_remain: -1,
                },
            );
        }
        value = (value << 6) + ((next_byte & 0x3F) as u32);
        pos += 1;
        n_remain -= 1;
    }

    if partial_start.n_remain > 0 && n_remain == 0 {
        code_points.push(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while b(src, pos) != 0 {
        let first_byte = b(src, pos);
        let highbits = first_byte >> 4;
        n_remain = LOOKUP[highbits as usize] - 1;

        if n_remain < 0 {
            // invalid sequence, abort
            code_points.clear();
            code_points.push(0);
            return (code_points, PartialUtf8 { value: 0, n_remain });
        }

        let mask = ((1u32 << (7 - n_remain)) - 1) as u8;
        value = (first_byte & mask) as u32;

        pos += 1;
        while b(src, pos) != 0 && n_remain > 0 {
            value = (value << 6) + ((b(src, pos) & 0x3F) as u32);
            pos += 1;
            n_remain -= 1;
        }
        if n_remain == 0 {
            code_points.push(value);
        }
    }
    code_points.push(0);

    (code_points, PartialUtf8 { value, n_remain })
}

/// `is_digit_char` (:94)
#[inline]
fn is_digit_char(c: u8) -> bool {
    b'0' <= c && c <= b'9'
}

/// `is_word_char` (:98)
#[inline]
fn is_word_char(c: u8) -> bool {
    (b'a' <= c && c <= b'z') || (b'A' <= c && c <= b'Z') || c == b'-' || is_digit_char(c)
}

/// `parse_hex` (:102-123)
fn parse_hex(src: &[u8], pos: usize, size: usize) -> Result<(u32, usize), String> {
    let mut p = pos;
    let end = pos + size;
    let mut value: u32 = 0;
    while p < end && b(src, p) != 0 {
        value <<= 4;
        let c = b(src, p);
        if b'a' <= c && c <= b'f' {
            value += (c - b'a' + 10) as u32;
        } else if b'A' <= c && c <= b'F' {
            value += (c - b'A' + 10) as u32;
        } else if b'0' <= c && c <= b'9' {
            value += (c - b'0') as u32;
        } else {
            break;
        }
        p += 1;
    }
    if p != end {
        return Err(format!("expecting {size} hex chars at {}", show(src, pos)));
    }
    Ok((value, p))
}

/// `parse_space` (:125-138)
fn parse_space(src: &[u8], pos: usize, newline_ok: bool) -> usize {
    let mut p = pos;
    while b(src, p) == b' '
        || b(src, p) == b'\t'
        || b(src, p) == b'#'
        || (newline_ok && (b(src, p) == b'\r' || b(src, p) == b'\n'))
    {
        if b(src, p) == b'#' {
            while b(src, p) != 0 && b(src, p) != b'\r' && b(src, p) != b'\n' {
                p += 1;
            }
        } else {
            p += 1;
        }
    }
    p
}

/// `parse_name` (:140-149)
fn parse_name(src: &[u8], pos: usize) -> Result<usize, String> {
    let mut p = pos;
    while is_word_char(b(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(format!("expecting name at {}", show(src, pos)));
    }
    Ok(p)
}

/// `parse_int` (:151-160)
fn parse_int(src: &[u8], pos: usize) -> Result<usize, String> {
    let mut p = pos;
    while is_digit_char(b(src, p)) {
        p += 1;
    }
    if p == pos {
        return Err(format!("expecting integer at {}", show(src, pos)));
    }
    Ok(p)
}

/// `parse_char` (:162-184)
fn parse_char(src: &[u8], pos: usize) -> Result<(u32, usize), String> {
    if b(src, pos) == b'\\' {
        match b(src, pos + 1) {
            b'x' => return parse_hex(src, pos + 2, 2),
            b'u' => return parse_hex(src, pos + 2, 4),
            b'U' => return parse_hex(src, pos + 2, 8),
            b't' => return Ok((b'\t' as u32, pos + 2)),
            b'r' => return Ok((b'\r' as u32, pos + 2)),
            b'n' => return Ok((b'\n' as u32, pos + 2)),
            c @ (b'\\' | b'"' | b'[' | b']' | b'-') => return Ok((c as u32, pos + 2)),
            _ => return Err(format!("unknown escape at {}", show(src, pos))),
        }
    } else if b(src, pos) != 0 {
        return Ok(decode_utf8_cpt(src, pos));
    }
    Err("unexpected end of input".to_string())
}

/// `parse_token` (:186-230) — the `<token>` form needs the vocab tokenizer.
fn parse_token(vocab: Option<&Vocab>, src: &[u8], pos: usize) -> Result<(u32, usize), String> {
    let mut p = pos;
    if b(src, p) != b'<' {
        return Err(format!("expecting '<' at {}", show(src, p)));
    }
    p += 1;

    // Parse <[id]>
    if b(src, p) == b'[' {
        p += 1;
        let int_end = parse_int(src, p)?;
        let text = std::str::from_utf8(&src[p..int_end]).unwrap_or("");
        // 5cf3a3528 (upstream) fixed the numeric truncation here: the C parsed
        // with std::stoul into unsigned long and silently truncated
        // uint32_t token_id = stoul(...); it now rejects ids above u32::MAX
        // (llama-grammar.cpp:197-203). The port's `parse::<u32>` never had a
        // truncation path — values > u32::MAX fail the parse, the fixed C's
        // behavior — so this arm needs no change.
        let token_id: u32 = text
            .parse::<u32>()
            .map_err(|_| format!("invalid token id at {}", show(src, pos)))?;
        p = int_end;
        if b(src, p) != b']' {
            return Err(format!("expecting ']' at {}", show(src, p)));
        }
        p += 1;
        if b(src, p) != b'>' {
            return Err(format!("expecting '>' at {}", show(src, p)));
        }
        p += 1;
        return Ok((token_id, p));
    }

    let vocab = match vocab {
        Some(v) => v,
        None => return Err(format!("no vocab to parse token at {}", show(src, pos))),
    };

    // Parse <token> and tokenize to obtain the token id
    while b(src, p) != 0 && b(src, p) != b'>' {
        p += 1;
    }
    if b(src, p) != b'>' {
        return Err(format!("expecting '>' at {}", show(src, p)));
    }
    p += 1;

    let text = std::str::from_utf8(&src[pos..p]).unwrap_or("");
    let tokens = vocab.tokenize(text, false, true);
    if tokens.len() != 1 {
        // must tokenize to exactly 1 token
        return Err(format!("invalid token '{}'", show_range(src, pos, p)));
    }
    Ok((tokens[0] as u32, p))
}

/// `print_grammar_char` (:232-239) → string
fn grammar_char(c: u32) -> String {
    if 0x20 <= c && c <= 0x7f {
        format!("{}", char::from_u32(c).unwrap_or('?'))
    } else {
        // cop out of encoding UTF-8
        format!("<U+{c:04X}>")
    }
}

/// `is_char_element` (:241-250)
fn is_char_element(elem: GrammarElement) -> bool {
    matches!(
        elem.ty,
        Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAlt
            | Gretype::CharRngUpper
            | Gretype::CharAny
    )
}

/// a printable slice of the source for error messages (C prints the pointer's
/// string, i.e. up to the next NUL)
fn show(src: &[u8], pos: usize) -> String {
    let mut out = String::new();
    let mut p = pos;
    while b(src, p) != 0 {
        out.push_str(&grammar_char(b(src, p) as u32));
        p += 1;
    }
    out
}

/// `show` for a range
fn show_range(src: &[u8], from: usize, to: usize) -> String {
    let mut out = String::new();
    for p in from..to.min(src.len()) {
        out.push_str(&grammar_char(b(src, p) as u32));
    }
    out
}

// ---------------------------------------------------------------------------
// grammar parser (:416-748)
// ---------------------------------------------------------------------------

/// `struct llama_grammar_parser` (llama-grammar.h:86-117).
///
/// `symbol_ids` is a `std::map<std::string, uint32_t>` in C (ordered!) and is
/// reproduced with a `BTreeMap`: ids are handed out as `symbol_ids.size()`, so
/// the ordering is observable in rule numbering and in `print`.
pub struct GrammarParser<'a> {
    /// note: null vocab for testing (not great) — C allows `nullptr`
    pub vocab: Option<&'a Vocab>,
    pub symbol_ids: BTreeMap<String, u32>,
    pub rules: Rules,
}

impl<'a> GrammarParser<'a> {
    pub fn new(vocab: Option<&'a Vocab>) -> Self {
        GrammarParser {
            vocab,
            symbol_ids: BTreeMap::new(),
            rules: Vec::new(),
        }
    }

    /// `get_symbol_id` (:416-420)
    pub fn get_symbol_id(&mut self, name: &str) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        *self.symbol_ids.entry(name.to_string()).or_insert(next_id)
    }

    /// `generate_symbol_id` (:422-426)
    pub fn generate_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        self.symbol_ids
            .insert(format!("{base_name}_{next_id}"), next_id);
        next_id
    }

    /// `add_rule` (:428-433)
    pub fn add_rule(&mut self, rule_id: u32, rule: &Rule) {
        if self.rules.len() <= rule_id as usize {
            self.rules.resize(rule_id as usize + 1, Rule::new());
        }
        self.rules[rule_id as usize] = rule.clone();
    }

    /// `parse_alternates` (:435-450)
    fn parse_alternates(
        &mut self,
        src: &[u8],
        pos: usize,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
    ) -> Result<usize, String> {
        let mut rule = Rule::new();
        let mut pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested)?;
        while b(src, pos) == b'|' {
            rule.push(GrammarElement::new(Gretype::Alt, 0));
            pos = parse_space(src, pos + 1, true);
            pos = self.parse_sequence(src, pos, rule_name, &mut rule, is_nested)?;
        }
        rule.push(GrammarElement::new(Gretype::End, 0));
        self.add_rule(rule_id, &rule);
        Ok(pos)
    }

    /// `parse_sequence` (:452-664)
    fn parse_sequence(
        &mut self,
        src: &[u8],
        pos: usize,
        rule_name: &str,
        rule: &mut Rule,
        is_nested: bool,
    ) -> Result<usize, String> {
        let mut last_sym_start = rule.len();
        let mut pos = pos;
        let mut n_prev_rules: u64 = 1;

        // use UINT64_MAX as the empty value (C: aligned to uint64_t so -1 can't be used)
        macro_rules! handle_repetitions {
            ($min_times:expr, $max_times:expr) => {{
                let min_times: u64 = $min_times;
                let max_times: u64 = $max_times;
                let no_max = max_times == u64::MAX;
                if last_sym_start == rule.len() {
                    return Err(format!(
                        "expecting preceding item to */+/?/{{ at {}",
                        show(src, pos)
                    ));
                }

                // apply transformation to previous symbol (last_sym_start to end)
                // according to the following rewrite rules:
                // S{m,n} --> S S S (m times) S'(n-m)
                //            S'(x)   ::= S S'(x-1) |
                //            (... n-m definitions of these S' rules ...)
                //            S'(1)   ::= S |
                // S{m,} -->  S S S (m times) S'
                //            S'     ::= S S' |
                // S*     --> S{0,}
                //        --> S'     ::= S S' |
                // S+     --> S{1,}
                //        --> S S'
                //            S'     ::= S S' |
                // S?     --> S{0,1}
                //        --> S'
                //            S'     ::= S |

                let prev_rule: Rule = rule[last_sym_start..].to_vec();
                // calculate the total number of rules generated by this repetition
                let total_rules: u64 = if !no_max && max_times > 0 {
                    max_times
                } else if min_times > 0 {
                    min_times
                } else {
                    1
                };

                if n_prev_rules * total_rules > MAX_REPETITION_THRESHOLD {
                    return Err("number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please reduce the number of repetitions or rule complexity".to_string());
                }

                if min_times == 0 {
                    rule.resize(last_sym_start, GrammarElement::new(Gretype::End, 0));
                } else {
                    // repeat the previous elements (min_times - 1) times
                    for _ in 1..min_times {
                        rule.extend_from_slice(&prev_rule);
                    }
                }

                let mut last_rec_rule_id: u32 = 0;
                // NOTE: C computes `max_times - min_times` on uint64_t; `{5,2}`
                // underflows there (the reference implementation then allocates
                // until it dies). `saturating_sub` keeps the sane intent.
                let n_opt: u64 = if no_max { 1 } else { max_times.saturating_sub(min_times) };

                let mut rec_rule = prev_rule.clone();
                for i in 0..n_opt {
                    rec_rule.truncate(prev_rule.len());
                    let rec_rule_id = self.generate_symbol_id(rule_name);
                    if i > 0 || no_max {
                        rec_rule.push(GrammarElement::new(
                            Gretype::RuleRef,
                            if no_max { rec_rule_id } else { last_rec_rule_id },
                        ));
                    }
                    rec_rule.push(GrammarElement::new(Gretype::Alt, 0));
                    rec_rule.push(GrammarElement::new(Gretype::End, 0));
                    self.add_rule(rec_rule_id, &rec_rule);
                    last_rec_rule_id = rec_rule_id;
                }
                if n_opt > 0 {
                    rule.push(GrammarElement::new(Gretype::RuleRef, last_rec_rule_id));
                }
                n_prev_rules *= total_rules;
                debug_assert!(n_prev_rules >= 1);
            }};
        }

        while b(src, pos) != 0 {
            let c = b(src, pos);
            if c == b'"' {
                // literal string
                pos += 1;
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while b(src, pos) != b'"' {
                    if b(src, pos) == 0 {
                        return Err("unexpected end of input".to_string());
                    }
                    let (chr, p) = parse_char(src, pos)?;
                    pos = p;
                    rule.push(GrammarElement::new(Gretype::Char, chr));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if c == b'[' {
                // char range(s)
                pos += 1;
                let mut start_type = Gretype::Char;
                if b(src, pos) == b'^' {
                    pos += 1;
                    start_type = Gretype::CharNot;
                }
                last_sym_start = rule.len();
                n_prev_rules = 1;
                while b(src, pos) != b']' {
                    if b(src, pos) == 0 {
                        return Err("unexpected end of input".to_string());
                    }
                    let (chr, p) = parse_char(src, pos)?;
                    pos = p;
                    let ty = if last_sym_start < rule.len() {
                        Gretype::CharAlt
                    } else {
                        start_type
                    };

                    rule.push(GrammarElement::new(ty, chr));
                    if b(src, pos) == b'-' && b(src, pos + 1) != b']' {
                        if b(src, pos + 1) == 0 {
                            return Err("unexpected end of input".to_string());
                        }
                        let (endchr, p) = parse_char(src, pos + 1)?;
                        pos = p;
                        rule.push(GrammarElement::new(Gretype::CharRngUpper, endchr));
                    }
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if c == b'<' || c == b'!' {
                // token
                let mut ty = Gretype::Token;
                if c == b'!' {
                    // token inverse
                    ty = Gretype::TokenNot;
                    pos += 1;
                }
                let (token_id, p) = parse_token(self.vocab, src, pos)?;
                pos = parse_space(src, p, is_nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(ty, token_id));
            } else if is_word_char(c) {
                // rule reference
                let name_end = parse_name(src, pos)?;
                let name = String::from_utf8_lossy(&src[pos..name_end]).into_owned();
                let ref_rule_id = self.get_symbol_id(&name);
                pos = parse_space(src, name_end, is_nested);
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::RuleRef, ref_rule_id));
            } else if c == b'(' {
                // grouping
                // parse nested alternates into synthesized rule
                pos = parse_space(src, pos + 1, true);
                let n_rules_before = self.symbol_ids.len();
                let sub_rule_id = self.generate_symbol_id(rule_name);
                pos = self.parse_alternates(src, pos, rule_name, sub_rule_id, true)?;
                n_prev_rules = (self.symbol_ids.len() - n_rules_before).max(1) as u64;
                last_sym_start = rule.len();
                // output reference to synthesized rule
                rule.push(GrammarElement::new(Gretype::RuleRef, sub_rule_id));
                if b(src, pos) != b')' {
                    return Err(format!("expecting ')' at {}", show(src, pos)));
                }
                pos = parse_space(src, pos + 1, is_nested);
            } else if c == b'.' {
                // any char
                last_sym_start = rule.len();
                n_prev_rules = 1;
                rule.push(GrammarElement::new(Gretype::CharAny, 0));
                pos = parse_space(src, pos + 1, is_nested);
            } else if c == b'*' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions!(0u64, u64::MAX);
            } else if c == b'+' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions!(1u64, u64::MAX);
            } else if c == b'?' {
                pos = parse_space(src, pos + 1, is_nested);
                handle_repetitions!(0u64, 1u64);
            } else if c == b'{' {
                pos = parse_space(src, pos + 1, is_nested);

                if !is_digit_char(b(src, pos)) {
                    return Err(format!("expecting an int at {}", show(src, pos)));
                }
                let int_end = parse_int(src, pos)?;
                let min_times: u64 = String::from_utf8_lossy(&src[pos..int_end])
                    .parse()
                    .map_err(|_| format!("expecting an int at {}", show(src, pos)))?;
                pos = parse_space(src, int_end, is_nested);

                let mut max_times: u64 = u64::MAX; // default: no max limit

                if b(src, pos) == b'}' {
                    max_times = min_times;
                    pos = parse_space(src, pos + 1, is_nested);
                } else if b(src, pos) == b',' {
                    pos = parse_space(src, pos + 1, is_nested);

                    if is_digit_char(b(src, pos)) {
                        let int_end = parse_int(src, pos)?;
                        max_times = String::from_utf8_lossy(&src[pos..int_end])
                            .parse()
                            .map_err(|_| format!("expecting an int at {}", show(src, pos)))?;
                        pos = parse_space(src, int_end, is_nested);
                    }

                    if b(src, pos) != b'}' {
                        return Err(format!("expecting '}}' at {}", show(src, pos)));
                    }
                    pos = parse_space(src, pos + 1, is_nested);
                } else {
                    return Err(format!("expecting ',' at {}", show(src, pos)));
                }
                if min_times > MAX_REPETITION_THRESHOLD {
                    return Err("number of repetitions exceeds sane defaults, please reduce the number of repetitions".to_string());
                }
                if max_times != u64::MAX && max_times > MAX_REPETITION_THRESHOLD {
                    max_times = u64::MAX;
                }
                handle_repetitions!(min_times, max_times);
            } else {
                break;
            }
        }
        Ok(pos)
    }

    /// `parse` (:690-722). Returns false (and clears `rules`) on any error,
    /// exactly like the reference (which also prints the error to stderr).
    pub fn parse(&mut self, grammar_bytes: &[u8]) -> bool {
        // C parses a NUL-terminated `const char *` (grammar_str.c_str()): copy
        // the source with a NUL sentinel so `*pos`/`pos[1]`/`pos[2]` reads are
        // byte-exact (an embedded NUL truncates, as in C).
        let mut src: Vec<u8> = grammar_bytes.to_vec();
        src.extend_from_slice(&[0u8; 8]);

        match self.parse_inner(&src) {
            Ok(()) => true,
            Err(err) => {
                eprintln!("llama_grammar_parser::parse: error parsing grammar: {err}");
                self.rules.clear();
                false
            }
        }
    }

    fn parse_inner(&mut self, src: &[u8]) -> Result<(), String> {
        let mut pos = parse_space(src, 0, true);
        while b(src, pos) != 0 {
            pos = self.parse_rule(src, pos)?;
        }
        // validate the state to ensure that all rules are defined
        for rule in &self.rules {
            if rule.is_empty() {
                return Err("Undefined rule".to_string());
            }
            for elem in rule {
                if elem.ty == Gretype::RuleRef {
                    // ensure that the rule at that location exists
                    if elem.value as usize >= self.rules.len()
                        || self.rules[elem.value as usize].is_empty()
                    {
                        // get the name of the rule that is missing
                        for (name, id) in &self.symbol_ids {
                            if *id == elem.value {
                                return Err(format!("Undefined rule identifier '{name}'"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `parse_rule` (:666-688) — `pos` points at the rule name (C's `src`).
    fn parse_rule(&mut self, src: &[u8], pos: usize) -> Result<usize, String> {
        let name_start = pos;
        let name_end = parse_name(src, pos)?;
        let mut pos = parse_space(src, name_end, false);
        let name = String::from_utf8_lossy(&src[name_start..name_end]).into_owned();
        let rule_id = self.get_symbol_id(&name);

        if !(b(src, pos) == b':' && b(src, pos + 1) == b':' && b(src, pos + 2) == b'=') {
            return Err(format!("expecting ::= at {}", show(src, pos)));
        }
        pos = parse_space(src, pos + 3, true);

        pos = self.parse_alternates(src, pos, &name, rule_id, false)?;

        if b(src, pos) == b'\r' {
            pos += if b(src, pos + 1) == b'\n' { 2 } else { 1 };
        } else if b(src, pos) == b'\n' {
            pos += 1;
        } else if b(src, pos) != 0 {
            return Err(format!("expecting newline or end at {}", show(src, pos)));
        }
        Ok(parse_space(src, pos, true))
    }

    /// `c_rules` (:741-748) — trivially the rules themselves here.
    pub fn c_rules(&self) -> &Rules {
        &self.rules
    }

    /// `print` (:724-739) — GBNF text dump into a string (C writes to a FILE).
    pub fn print(&self) -> Result<String, String> {
        let mut symbol_id_names: BTreeMap<u32, String> = BTreeMap::new();
        for (name, id) in &self.symbol_ids {
            symbol_id_names.insert(*id, name.clone());
        }
        let mut out = String::new();
        for (i, rule) in self.rules.iter().enumerate() {
            self.print_rule(&mut out, i as u32, rule, &symbol_id_names)?;
        }
        Ok(out)
    }

    /// `print_rule` (:297-372)
    fn print_rule(
        &self,
        out: &mut String,
        rule_id: u32,
        rule: &Rule,
        symbol_id_names: &BTreeMap<u32, String>,
    ) -> Result<(), String> {
        use std::fmt::Write as _;
        if rule.is_empty() || rule.last().unwrap().ty != Gretype::End {
            return Err(format!(
                "malformed rule, does not end with LLAMA_GRETYPE_END: {rule_id}"
            ));
        }
        let name = symbol_id_names
            .get(&rule_id)
            .ok_or_else(|| format!("no symbol name for rule {rule_id}"))?;
        let _ = write!(out, "{name} ::= ");
        for i in 0..rule.len() - 1 {
            let elem = rule[i];
            match elem.ty {
                Gretype::End => {
                    return Err(format!("unexpected end of rule: {rule_id},{i}"));
                }
                Gretype::Alt => {
                    out.push_str("| ");
                }
                Gretype::RuleRef => {
                    let n = symbol_id_names
                        .get(&elem.value)
                        .ok_or_else(|| format!("no symbol name for rule {}", elem.value))?;
                    let _ = write!(out, "{n} ");
                }
                Gretype::Char => {
                    out.push('[');
                    out.push_str(&grammar_char(elem.value));
                }
                Gretype::CharNot => {
                    out.push_str("[^");
                    out.push_str(&grammar_char(elem.value));
                }
                Gretype::CharRngUpper => {
                    if i == 0 || !is_char_element(rule[i - 1]) {
                        return Err(format!(
                            "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: {rule_id},{i}"
                        ));
                    }
                    out.push('-');
                    out.push_str(&grammar_char(elem.value));
                }
                Gretype::CharAlt => {
                    if i == 0 || !is_char_element(rule[i - 1]) {
                        return Err(format!(
                            "LLAMA_GRETYPE_CHAR_ALT without preceding char: {rule_id},{i}"
                        ));
                    }
                    out.push_str(&grammar_char(elem.value));
                }
                Gretype::CharAny => {
                    out.push('.');
                }
                Gretype::Token => {
                    let _ = write!(out, "<[{}]> ", elem.value);
                }
                Gretype::TokenNot => {
                    let _ = write!(out, "!<[{}]> ", elem.value);
                }
            }
            if is_char_element(elem) {
                match rule[i + 1].ty {
                    Gretype::CharAlt | Gretype::CharRngUpper | Gretype::CharAny => {}
                    _ => out.push_str("] "),
                }
            }
        }
        out.push('\n');
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// matcher (:750-1126)
// ---------------------------------------------------------------------------

/// `llama_grammar_is_end_of_sequence` (:751-757)
#[inline]
pub fn is_end_of_sequence(rules: &Rules, pos: ElemRef) -> bool {
    matches!(el_at(rules, pos).ty, Gretype::End | Gretype::Alt)
}

/// `llama_grammar_match_char` (:761-786)
///
/// Returns `(matched, position after the char range)`; the second value is also
/// how C gets `stack_pos_after` (calling it with `chr = 0`, :1110).
pub fn match_char(rules: &Rules, pos: ElemRef, chr: u32) -> (bool, ElemRef) {
    let mut found = false;
    let mut pos = pos;
    let is_positive_char = matches!(el_at(rules, pos).ty, Gretype::Char | Gretype::CharAny);

    debug_assert!(is_positive_char || el_at(rules, pos).ty == Gretype::CharNot);

    loop {
        if el_at(rules, next(pos)).ty == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            found =
                found || (el_at(rules, pos).value <= chr && chr <= el_at(rules, next(pos)).value);
            pos = next2(pos);
        } else if el_at(rules, pos).ty == Gretype::CharAny {
            // any character matches "."
            found = true;
            pos = next(pos);
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || el_at(rules, pos).value == chr;
            pos = next(pos);
        }
        if el_at(rules, pos).ty != Gretype::CharAlt {
            break;
        }
    }

    (found == is_positive_char, pos)
}

/// `llama_grammar_match_partial_char` (:791-837)
pub fn match_partial_char(rules: &Rules, pos: ElemRef, partial_utf8: PartialUtf8) -> bool {
    let mut pos = pos;
    let is_positive_char = matches!(el_at(rules, pos).ty, Gretype::Char | Gretype::CharAny);
    debug_assert!(is_positive_char || el_at(rules, pos).ty == Gretype::CharNot);

    let partial_value = partial_utf8.value;
    let n_remain = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if n_remain < 0 || (n_remain == 1 && partial_value < 2) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    let low0 = partial_value << (n_remain as u32 * 6);
    let high = low0 | ((1u32 << (n_remain as u32 * 6)) - 1);
    let mut low = low0;

    if low == 0 {
        if n_remain == 2 {
            low = 1 << 11;
        } else if n_remain == 3 {
            low = 1 << 16;
        }
    }

    loop {
        if el_at(rules, next(pos)).ty == Gretype::CharRngUpper {
            // inclusive range, e.g. [a-z]
            if el_at(rules, pos).value <= high && low <= el_at(rules, next(pos)).value {
                return is_positive_char;
            }
            pos = next2(pos);
        } else if el_at(rules, pos).ty == Gretype::CharAny {
            // any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if low <= el_at(rules, pos).value && el_at(rules, pos).value <= high {
                return is_positive_char;
            }
            pos = next(pos);
        }
        if el_at(rules, pos).ty != Gretype::CharAlt {
            break;
        }
    }

    !is_positive_char
}

/// `llama_grammar_match_token` (:841-852)
pub fn match_token(rules: &Rules, pos: ElemRef, token: LlamaToken) -> bool {
    let elem = el_at(rules, pos);
    debug_assert!(elem.ty == Gretype::Token || elem.ty == Gretype::TokenNot);
    if elem.ty == Gretype::Token {
        return elem.value == token as u32;
    }
    if elem.ty == Gretype::TokenNot {
        return elem.value != token as u32;
    }
    false
}

/// `llama_grammar_advance_stack` (:856-938)
///
/// Transforms a grammar pushdown stack into N possible stacks, all ending at a
/// character range (terminal element). `seen` (C `std::set` ordered by pointer
/// address) is a `BTreeSet` here; membership is the only thing observed.
pub fn advance_stack(rules: &Rules, stack: &Stack, new_stacks: &mut Stacks) {
    let mut todo: Vec<Stack> = vec![stack.clone()];
    let mut seen: BTreeSet<Stack> = BTreeSet::new();

    while let Some(curr) = todo.pop() {
        if !seen.insert(curr.clone()) {
            continue;
        }

        if curr.is_empty() {
            if !new_stacks.contains(&curr) {
                new_stacks.push(curr);
            }
            continue;
        }

        let pos = *curr.last().unwrap();

        match el_at(rules, pos).ty {
            Gretype::RuleRef => {
                let rule_id = el_at(rules, pos).value as usize;
                let mut subpos: ElemRef = (rule_id as u32, 0);
                loop {
                    // init new stack without the top (pos)
                    let mut next_stack: Stack = curr[..curr.len() - 1].to_vec();
                    if !is_end_of_sequence(rules, next(pos)) {
                        // if this rule ref is followed by another element, add that to stack
                        next_stack.push(next(pos));
                    }
                    if !is_end_of_sequence(rules, subpos) {
                        // if alternate is nonempty, add to stack
                        next_stack.push(subpos);
                    }
                    todo.push(next_stack);
                    while !is_end_of_sequence(rules, subpos) {
                        // scan to end of alternate def
                        subpos = next(subpos);
                    }
                    if el_at(rules, subpos).ty == Gretype::Alt {
                        // there's another alternate def of this rule to process
                        subpos = next(subpos);
                    } else {
                        break;
                    }
                }
            }
            Gretype::Char
            | Gretype::CharNot
            | Gretype::CharAny
            | Gretype::Token
            | Gretype::TokenNot => {
                if !new_stacks.contains(&curr) {
                    // only add the stack if it's not a duplicate of one we already have
                    new_stacks.push(curr);
                }
            }
            _ => {
                // end of alternate (END, ALT) or middle of char range
                // (CHAR_ALT, CHAR_RNG_UPPER); stack should never be left on those
                panic!(
                    "fatal error: stack on {} / {}",
                    el_at(rules, pos).ty.name(),
                    pos.1
                );
            }
        }
    }
}

/// `llama_grammar_accept_chr` (:1020-1044)
fn accept_chr(rules: &Rules, stack: &Stack, chr: u32, new_stacks: &mut Stacks) {
    if stack.is_empty() {
        return;
    }

    let pos = *stack.last().unwrap();

    // ignore if this turns into a token
    if matches!(el_at(rules, pos).ty, Gretype::Token | Gretype::TokenNot) {
        return;
    }

    let (matched, after) = match_char(rules, pos, chr);
    if matched {
        let mut new_stack: Stack = stack[..stack.len() - 1].to_vec();
        if !is_end_of_sequence(rules, after) {
            new_stack.push(after);
        }
        advance_stack(rules, &new_stack, new_stacks);
    }
}

/// `llama_grammar_accept` (:1046-1055) — accept a single code point.
pub fn accept_chr_into(rules: &Rules, stacks: &Stacks, chr: u32) -> Stacks {
    let mut stacks_new: Stacks = Vec::with_capacity(stacks.len());
    for stack in stacks {
        accept_chr(rules, stack, chr, &mut stacks_new);
    }
    stacks_new
}

/// `llama_grammar_reject_candidates_for_stack` (:1057-1126)
///
/// `decoded` holds the code point arrays (NUL-terminated) the candidates walk;
/// `Candidate::cp` is C's `code_points` offset.
pub fn reject_candidates_for_stack(
    rules: &Rules,
    stack: &Stack,
    candidates: &[Candidate],
    decoded: &[Vec<u32>],
) -> Vec<Candidate> {
    let mut rejects: Vec<Candidate> = Vec::with_capacity(candidates.len());

    if stack.is_empty() {
        for tok in candidates {
            if decoded[tok.decoded][tok.cp] != 0 || tok.partial.n_remain != 0 {
                rejects.push(*tok);
            }
        }
        return rejects;
    }

    let stack_pos = *stack.last().unwrap();

    // if the top of the stack is a token rule, then we only need to check the token id
    if matches!(
        el_at(rules, stack_pos).ty,
        Gretype::Token | Gretype::TokenNot
    ) {
        for tok in candidates {
            if decoded[tok.decoded][tok.cp] == 0 {
                // reached the end of a token consumed by char rules, reject iff it ended
                // in a partial response
                if tok.partial.n_remain != 0 {
                    rejects.push(*tok);
                }
            } else if !match_token(rules, stack_pos, tok.id) {
                rejects.push(*tok);
            }
        }
        return rejects;
    }

    let mut next_candidates: Vec<Candidate> = Vec::with_capacity(candidates.len());

    for tok in candidates {
        if decoded[tok.decoded][tok.cp] == 0 {
            // reached end of full codepoints in token, reject iff it ended in a partial
            // sequence that cannot satisfy this position in grammar
            if tok.partial.n_remain != 0 && !match_partial_char(rules, stack_pos, tok.partial) {
                rejects.push(*tok);
            }
        } else if match_char(rules, stack_pos, decoded[tok.decoded][tok.cp]).0 {
            next_candidates.push(Candidate {
                cp: tok.cp + 1,
                ..*tok
            });
        } else {
            rejects.push(*tok);
        }
    }

    let stack_pos_after = match_char(rules, stack_pos, 0).1;

    // update top of stack to next element, if any
    let mut stack_after: Stack = stack[..stack.len() - 1].to_vec();
    if !is_end_of_sequence(rules, stack_pos_after) {
        stack_after.push(stack_pos_after);
    }
    let mut next_stacks: Stacks = Vec::new();
    advance_stack(rules, &stack_after, &mut next_stacks);

    let next_rejects = reject_candidates(rules, &next_stacks, &next_candidates, decoded);
    for tok in next_rejects {
        rejects.push(Candidate {
            cp: tok.cp - 1,
            ..tok
        });
    }

    rejects
}

/// `llama_grammar_reject_candidates` (:940-957)
pub fn reject_candidates(
    rules: &Rules,
    stacks: &Stacks,
    candidates: &[Candidate],
    decoded: &[Vec<u32>],
) -> Vec<Candidate> {
    assert!(!stacks.is_empty(), "grammar has no stacks left"); // REVIEW (C comment)

    if candidates.is_empty() {
        return Vec::new();
    }

    let mut rejects = reject_candidates_for_stack(rules, &stacks[0], candidates, decoded);

    for stack in &stacks[1..] {
        rejects = reject_candidates_for_stack(rules, stack, &rejects, decoded);
    }

    rejects
}

/// `struct llama_grammar_candidate` (llama-grammar.h:57-62) with
/// `code_points` → `(decoded index, offset)`.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// `index` into `llama_token_data_array::data` (used for masking)
    pub index: usize,
    /// index into the decoded code point arrays passed to the reject helpers
    pub decoded: usize,
    /// C's `code_points` offset
    pub cp: usize,
    pub partial: PartialUtf8,
    pub id: LlamaToken,
}

/// `llama_grammar_detect_left_recursion` (:959-1010)
pub fn detect_left_recursion(
    rules: &Rules,
    rule_index: usize,
    rules_visited: &mut [bool],
    rules_in_progress: &mut [bool],
    rules_may_be_empty: &mut [bool],
) -> bool {
    if rules_in_progress[rule_index] {
        return true;
    }

    rules_in_progress[rule_index] = true;

    let rule = &rules[rule_index];

    // First check if the rule might produce the empty string.
    let mut at_rule_start = true;
    for i in 0..rule.len() {
        if is_end_of_sequence(rules, (rule_index as u32, i as u32)) {
            if at_rule_start {
                rules_may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the
    // previous nonterminal may be empty)
    let mut recurse_into_nonterminal = true;
    for i in 0..rule.len() {
        if rule[i].ty == Gretype::RuleRef && recurse_into_nonterminal {
            if detect_left_recursion(
                rules,
                rule[i].value as usize,
                rules_visited,
                rules_in_progress,
                rules_may_be_empty,
            ) {
                return true;
            }
            if !rules_may_be_empty[rule[i].value as usize] {
                recurse_into_nonterminal = false;
            }
        } else if rule[i].ty == Gretype::End || rule[i].ty == Gretype::Alt {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    rules_in_progress[rule_index] = false;
    rules_visited[rule_index] = true;

    false
}

// ---------------------------------------------------------------------------
// grammar object (:1128-1525)
// ---------------------------------------------------------------------------

/// `struct llama_grammar` (llama-grammar.h:126-151).
#[derive(Clone, Debug, Default)]
pub struct Grammar {
    /// `const llama_grammar_rules rules`
    pub rules: Rules,
    /// `llama_grammar_stacks stacks`
    pub stacks: Stacks,

    /// buffer for partially generated UTF-8 sequence from accepted tokens
    pub partial_utf8: PartialUtf8,

    /// start symbol (C keeps it only inside `llama_grammar_init_impl`; stored
    /// here so `reset` does not have to re-parse)
    pub start_rule_index: usize,

    /// lazy grammars wait for trigger words or tokens before constraining sampling
    pub lazy: bool,
    /// initialized to true for lazy grammars only
    pub awaiting_trigger: bool,
    /// output buffered by lazy grammar (cleared once trigger is found)
    pub trigger_buffer: String,
    /// tokens buffered by lazy grammar (used to replay when a trigger is found)
    pub trigger_buffer_positions: Vec<(LlamaToken, (usize, usize))>,
    /// tokens that trigger a lazy grammar, or tokens to force printing of (even if special)
    pub trigger_tokens: Vec<LlamaToken>,
    /// trigger regexes (not ported: needs `std::regex`)
    pub trigger_patterns: Vec<String>,
}

impl Grammar {
    /// `llama_grammar_init_impl(vocab, rules, n_rules, start_rule_index)` (:1130-1209)
    ///
    /// `rules` are element lists without the trailing END (C copies each rule
    /// up to its END and appends one); pass complete rules here — they are
    /// normalized the same way.
    pub fn from_c_rules(rules: &Rules, start_rule_index: usize) -> Result<Grammar, String> {
        let mut vec_rules: Rules = Vec::with_capacity(rules.len());
        for rule in rules {
            let mut r: Rule = Vec::new();
            for elem in rule {
                if elem.ty == Gretype::End {
                    break;
                }
                r.push(*elem);
            }
            r.push(GrammarElement::new(Gretype::End, 0));
            vec_rules.push(r);
        }
        Grammar::from_parsed_rules(vec_rules, start_rule_index)
    }

    /// validation + left-recursion check + initial stacks (:1146-1208)
    pub fn from_parsed_rules(vec_rules: Rules, start_rule_index: usize) -> Result<Grammar, String> {
        let n_rules = vec_rules.len();

        // validate that all rule references point to valid rules
        for (i, rule) in vec_rules.iter().enumerate() {
            for elem in rule {
                if elem.ty == Gretype::RuleRef
                    && (elem.value as usize >= n_rules || vec_rules[elem.value as usize].is_empty())
                {
                    return Err(format!(
                        "invalid grammar: rule {i} references undefined rule {}",
                        elem.value
                    ));
                }
            }
        }

        // check for left recursion
        let mut rules_visited = vec![false; n_rules];
        let mut rules_in_progress = vec![false; n_rules];
        let mut rules_may_be_empty = vec![false; n_rules];
        for i in 0..n_rules {
            if rules_visited[i] {
                continue;
            }
            if detect_left_recursion(
                &vec_rules,
                i,
                &mut rules_visited,
                &mut rules_in_progress,
                &mut rules_may_be_empty,
            ) {
                return Err(format!(
                    "unsupported grammar, left recursion detected for nonterminal at index {i}"
                ));
            }
        }

        let stacks = initial_stacks(&vec_rules, start_rule_index);

        Ok(Grammar {
            rules: vec_rules,
            stacks,
            partial_utf8: PartialUtf8::default(),
            start_rule_index,
            lazy: false,
            awaiting_trigger: false,
            trigger_buffer: String::new(),
            trigger_buffer_positions: Vec::new(),
            trigger_tokens: Vec::new(),
            trigger_patterns: Vec::new(),
        })
    }

    /// `llama_grammar_init_impl(vocab, grammar_str, grammar_root, ...)` (:1211-1315)
    pub fn parse(
        vocab: Option<&Vocab>,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<Grammar, String> {
        let mut parser = GrammarParser::new(vocab);

        // if there is a grammar, parse it
        // rules will be empty (default) if there are parse errors
        if !parser.parse(grammar_str.as_bytes()) || parser.rules.is_empty() {
            return Err("failed to parse grammar".to_string());
        }

        // ensure that the grammar contains the start symbol
        if !parser.symbol_ids.contains_key(grammar_root) {
            return Err(format!(
                "grammar does not contain a '{grammar_root}' symbol"
            ));
        }

        let start_rule_index = parser.symbol_ids[grammar_root] as usize;
        let rules = parser.c_rules().clone();

        Grammar::from_parsed_rules(rules, start_rule_index)
    }

    /// `llama_sampler_grammar_reset` (llama-sampler.cpp:2700-2718) — re-init:
    /// rebuild the initial stacks from the (immutable) rules.
    pub fn reset(&mut self) {
        self.stacks = initial_stacks(&self.rules, self.start_rule_index);
        self.partial_utf8 = PartialUtf8::default();
        self.awaiting_trigger = self.lazy;
        self.trigger_buffer.clear();
        self.trigger_buffer_positions.clear();
    }

    /// `llama_grammar_accept_str` (:1457-1470)
    pub fn accept_str(&mut self, piece: &str) -> Result<(), String> {
        // Note terminating 0 in decoded string
        let (code_points, partial) = decode_utf8(piece.as_bytes(), self.partial_utf8);

        for &cp in &code_points[..code_points.len() - 1] {
            self.stacks = accept_chr_into(&self.rules, &self.stacks, cp);
        }

        self.partial_utf8 = partial;
        if self.stacks.is_empty() {
            return Err(format!(
                "Unexpected empty grammar stack after accepting piece: {piece}"
            ));
        }
        Ok(())
    }

    /// `llama_grammar_accept_token` (:1472-1525)
    pub fn accept_token(&mut self, token: LlamaToken, piece: &[u8]) -> Result<(), String> {
        // Note terminating 0 in decoded string
        let (code_points, partial) = decode_utf8(piece, self.partial_utf8);

        let mut stacks_new: Stacks = Vec::with_capacity(self.stacks.len());

        for stack in &self.stacks {
            if stack.is_empty() {
                continue;
            }

            let pos = *stack.last().unwrap();

            if matches!(
                el_at(&self.rules, pos).ty,
                Gretype::Token | Gretype::TokenNot
            ) {
                if match_token(&self.rules, pos, token) {
                    let mut new_stack: Stack = stack[..stack.len() - 1].to_vec();
                    if !is_end_of_sequence(&self.rules, next(pos)) {
                        new_stack.push(next(pos));
                    }
                    advance_stack(&self.rules, &new_stack, &mut stacks_new);
                }
            } else {
                let mut current_stacks: Stacks = vec![stack.clone()];

                for &cp in &code_points[..code_points.len() - 1] {
                    let mut next_stacks: Stacks = Vec::new();

                    for cur_stack in &current_stacks {
                        accept_chr(&self.rules, cur_stack, cp, &mut next_stacks);
                    }

                    current_stacks = next_stacks;
                    if current_stacks.is_empty() {
                        break;
                    }
                }

                for surviving_stack in current_stacks {
                    if !stacks_new.contains(&surviving_stack) {
                        stacks_new.push(surviving_stack);
                    }
                }
            }
        }

        self.stacks = stacks_new;
        self.partial_utf8 = partial;

        if self.stacks.is_empty() {
            return Err(format!(
                "Unexpected empty grammar stack after accepting piece: {} ({token})",
                String::from_utf8_lossy(piece)
            ));
        }
        Ok(())
    }

    /// `llama_grammar_apply_impl` (:1355-1396)
    pub fn apply(&self, vocab: &dyn GrammarVocab, cur_p: &mut TokenDataArray) {
        if self.awaiting_trigger {
            return;
        }

        let mut allow_eog = false;
        for stack in &self.stacks {
            if stack.is_empty() {
                allow_eog = true;
                break;
            }
        }

        let mut candidates_decoded: Vec<Vec<u32>> = Vec::with_capacity(cur_p.size);
        let mut candidates_grammar: Vec<Candidate> = Vec::with_capacity(cur_p.size);

        for i in 0..cur_p.size {
            let id = cur_p.data[i].id;
            let piece = vocab.token_piece(id);

            if vocab.is_eog(id) {
                if !allow_eog {
                    cur_p.data[i].logit = f32::NEG_INFINITY;
                }
            } else if piece.is_empty() || piece[0] == 0 {
                cur_p.data[i].logit = f32::NEG_INFINITY;
            } else {
                let (code_points, partial) = decode_utf8(piece, self.partial_utf8);
                candidates_decoded.push(code_points);
                candidates_grammar.push(Candidate {
                    index: i,
                    decoded: candidates_decoded.len() - 1,
                    cp: 0,
                    partial,
                    id,
                });
            }
        }

        let rejects = reject_candidates(
            &self.rules,
            &self.stacks,
            &candidates_grammar,
            &candidates_decoded,
        );
        for reject in rejects {
            cur_p.data[reject.index].logit = f32::NEG_INFINITY;
        }
    }

    /// `llama_grammar_accept_impl` (:1398-1455) — non-lazy path. Reads the piece
    /// and the EOG flag from `vocab`, exactly like C (`grammar.vocab`).
    ///
    /// The lazy/trigger branch (:1403-1443) needs `std::regex` and is not
    /// ported; `awaiting_trigger` is therefore always false here.
    pub fn accept_impl(
        &mut self,
        vocab: &dyn GrammarVocab,
        token: LlamaToken,
    ) -> Result<(), String> {
        let piece = vocab.token_piece(token).to_vec();

        debug_assert!(!self.awaiting_trigger);

        if vocab.is_eog(token) {
            for stack in &self.stacks {
                if stack.is_empty() {
                    return Ok(());
                }
            }
            // C: GGML_ABORT("fatal error")
            return Err("fatal error: EOG token accepted by an incomplete grammar".to_string());
        }

        self.accept_token(token, &piece)
    }
}

/// `llama_grammar_stacks` for the start rule alternates (:1172-1192, :1265-1285)
pub fn initial_stacks(rules: &Rules, start_rule_index: usize) -> Stacks {
    let mut stacks: Stacks = Vec::new();
    let mut pos: ElemRef = (start_rule_index as u32, 0);
    loop {
        let mut stack: Stack = Vec::new();
        if !is_end_of_sequence(rules, pos) {
            // if alternate is nonempty, add to stack
            stack.push(pos);
        }
        advance_stack(rules, &stack, &mut stacks);
        while !is_end_of_sequence(rules, pos) {
            // scan to end of alternate def
            pos = next(pos);
        }
        if el_at(rules, pos).ty == Gretype::Alt {
            // there's another alternate def of this rule to process
            pos = next(pos);
        } else {
            break;
        }
    }
    stacks
}

// ---------------------------------------------------------------------------
// vocab view used by apply/accept
// ---------------------------------------------------------------------------

/// `const llama_vocab *` access needed by the grammar (llama-grammar.cpp:1378-1384):
/// `vocab->token_to_piece(id)` and `vocab->is_eog(id)`.
pub trait GrammarVocab {
    fn is_eog(&self, token: LlamaToken) -> bool;
    /// `vocab->token_to_piece(token)` — cached piece, raw bytes (`special = true`)
    fn token_piece(&self, token: LlamaToken) -> &[u8];
}

impl GrammarVocab for Vocab {
    fn is_eog(&self, token: LlamaToken) -> bool {
        Vocab::is_eog(self, token)
    }

    fn token_piece(&self, token: LlamaToken) -> &[u8] {
        if token < 0 || token as usize >= self.n_tokens() as usize {
            return &[];
        }
        self.token_to_piece_bytes(token)
    }
}

/// Snapshot of `vocab->cache_token_to_piece` + `is_eog` (llama-vocab.cpp:3054)
/// so a [`Grammar`] can be driven without borrowing the whole vocab. Flat
/// buffers keep it at ~1 MB for a 152k-token vocab.
#[derive(Clone, Debug, Default)]
pub struct VocabPieces {
    offsets: Vec<u32>,
    buf: Vec<u8>,
    is_eog: Vec<bool>,
}

impl VocabPieces {
    pub fn from_vocab(vocab: &Vocab) -> Self {
        let n = vocab.n_tokens() as usize;
        let mut offsets = Vec::with_capacity(n + 1);
        let mut buf: Vec<u8> = Vec::new();
        let mut is_eog = Vec::with_capacity(n);
        offsets.push(0u32);
        for id in 0..n {
            buf.extend_from_slice(vocab.token_to_piece_bytes(id as LlamaToken));
            offsets.push(buf.len() as u32);
            is_eog.push(Vocab::is_eog(vocab, id as LlamaToken));
        }
        VocabPieces {
            offsets,
            buf,
            is_eog,
        }
    }

    /// build from `(piece, is_eog)` pairs in token-id order (tests / tools that
    /// have a piece dump but no vocab)
    pub fn from_iter<I: IntoIterator<Item = (Vec<u8>, bool)>>(it: I) -> Self {
        let mut offsets = vec![0u32];
        let mut buf: Vec<u8> = Vec::new();
        let mut is_eog = Vec::new();
        for (piece, eog) in it {
            buf.extend_from_slice(&piece);
            offsets.push(buf.len() as u32);
            is_eog.push(eog);
        }
        VocabPieces {
            offsets,
            buf,
            is_eog,
        }
    }

    /// raw constructor (offsets must be monotonic, `buf.len() == *offsets.last()`)
    pub fn from_parts(offsets: Vec<u32>, buf: Vec<u8>, is_eog: Vec<bool>) -> Self {
        assert_eq!(offsets.len(), is_eog.len() + 1);
        assert_eq!(*offsets.last().unwrap() as usize, buf.len());
        VocabPieces {
            offsets,
            buf,
            is_eog,
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.is_eog.len()
    }
}

impl GrammarVocab for VocabPieces {
    fn is_eog(&self, token: LlamaToken) -> bool {
        token >= 0 && (token as usize) < self.is_eog.len() && self.is_eog[token as usize]
    }

    /// C indexes the cache directly (out-of-range ids throw); an out-of-range id
    /// yields an empty piece, which `apply` masks with -inf anyway.
    fn token_piece(&self, token: LlamaToken) -> &[u8] {
        if token < 0 || token as usize + 1 >= self.offsets.len() {
            return &[];
        }
        let a = self.offsets[token as usize] as usize;
        let b = self.offsets[token as usize + 1] as usize;
        &self.buf[a..b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `decode_utf8` (:34) — the two reference implementations agree on ASCII and
    // the partial state survives across calls.
    #[test]
    fn decode_utf8_matches_reference_semantics() {
        // C returns the *last* decoded value with n_remain = 0
        let (cps, partial) = decode_utf8(b"abc", PartialUtf8::default());
        assert_eq!(cps, vec![b'a' as u32, b'b' as u32, b'c' as u32, 0]);
        assert_eq!(
            partial,
            PartialUtf8 {
                value: b'c' as u32,
                n_remain: 0
            }
        );

        // 0xF0 0x9F 0x94 0xB5 = U+1F535 (🔵): complete
        let (cps, partial) = decode_utf8(&[0xF0, 0x9F, 0x94, 0xB5], PartialUtf8::default());
        assert_eq!(cps, vec![0x1F535, 0]);
        assert_eq!(
            partial,
            PartialUtf8 {
                value: 0x1F535,
                n_remain: 0
            }
        );

        // split across two pieces: first half then second half
        let (cps, p1) = decode_utf8(&[0xF0, 0x9F], PartialUtf8::default());
        assert_eq!(cps, vec![0]);
        assert_eq!(
            p1,
            PartialUtf8 {
                value: 0x1F,
                n_remain: 2
            }
        );
        let (cps, p2) = decode_utf8(&[0x94, 0xB5], p1);
        assert_eq!(cps, vec![0x1F535, 0]);
        assert_eq!(
            p2,
            PartialUtf8 {
                value: 0x1F535,
                n_remain: 0
            }
        );

        // incomplete tail (U+2581 ▁ = E2 96 81)
        let (cps, p3) = decode_utf8(&[0xE2, 0x96], PartialUtf8::default());
        assert_eq!(cps, vec![0]);
        assert_eq!(
            p3,
            PartialUtf8 {
                value: 0x96,
                n_remain: 1
            }
        );
        let (cps, p3b) = decode_utf8(&[0x81], p3);
        assert_eq!(cps, vec![0x2581, 0]);
        assert_eq!(
            p3b,
            PartialUtf8 {
                value: 0x2581,
                n_remain: 0
            }
        );

        // invalid continuation byte with a pending partial → {0, -1}
        let (cps, p4) = decode_utf8(
            b"x",
            PartialUtf8 {
                value: 0x1F5,
                n_remain: 2,
            },
        );
        assert_eq!(cps, vec![0]);
        assert_eq!(
            p4,
            PartialUtf8 {
                value: 0,
                n_remain: -1
            }
        );

        // 0x80..0xB0 are continuation bytes: LOOKUP[8..11] == 0 → invalid
        let (cps, p5) = decode_utf8(&[0x80], PartialUtf8::default());
        assert_eq!(cps, vec![0]);
        assert_eq!(
            p5,
            PartialUtf8 {
                value: 0,
                n_remain: -1
            }
        );
    }

    // `match_char` (:761) — range/ALT/ANY/NOT walk
    #[test]
    fn match_char_ranges() {
        // [a-cx]
        let rules: Rules = vec![vec![
            GrammarElement::new(Gretype::Char, 'a' as u32),
            GrammarElement::new(Gretype::CharRngUpper, 'c' as u32),
            GrammarElement::new(Gretype::CharAlt, 'x' as u32),
            GrammarElement::new(Gretype::End, 0),
        ]];
        let pos = (0u32, 0u32);
        assert!(match_char(&rules, pos, 'b' as u32).0);
        assert!(match_char(&rules, pos, 'x' as u32).0);
        assert!(!match_char(&rules, pos, 'd' as u32).0);
        // the walk stops on the element after the cluster (the END here)
        assert_eq!(match_char(&rules, pos, 'b' as u32).1, (0, 3));

        // [^a]
        let rules: Rules = vec![vec![
            GrammarElement::new(Gretype::CharNot, 'a' as u32),
            GrammarElement::new(Gretype::End, 0),
        ]];
        assert!(!match_char(&rules, (0, 0), 'a' as u32).0);
        assert!(match_char(&rules, (0, 0), 'b' as u32).0);
        assert_eq!(match_char(&rules, (0, 0), 'b' as u32).1, (0, 1));

        // [a-c] followed by . — the cluster walk stops *at* the next element
        let rules: Rules = vec![vec![
            GrammarElement::new(Gretype::Char, 'a' as u32),
            GrammarElement::new(Gretype::CharRngUpper, 'c' as u32),
            GrammarElement::new(Gretype::CharAny, 0),
            GrammarElement::new(Gretype::End, 0),
        ]];
        let (m, after) = match_char(&rules, (0, 0), 'b' as u32);
        assert!(m);
        assert_eq!(after, (0, 2)); // the CHAR_ANY is the next stack top
        assert!(match_char(&rules, after, 'q' as u32).0);
    }

    // `parse_hex`/`parse_char` escapes (:102, :162)
    #[test]
    fn parse_char_escapes() {
        let src = b"\\x41\\u00e9\\t";
        let (c, p) = parse_char(src, 0).unwrap();
        assert_eq!((c, p), ('A' as u32, 4));
        let (c, p) = parse_char(src, p).unwrap();
        assert_eq!((c, p), (0xE9, 10));
        let (c, p) = parse_char(src, p).unwrap();
        assert_eq!((c, p), (b'\t' as u32, 12));
        assert_eq!(
            parse_char(b"\\q", 0),
            Err("unknown escape at \\q".to_string())
        );
        assert!(parse_char(b"", 0).is_err());
    }
}
