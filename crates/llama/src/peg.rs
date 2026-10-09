//! peg.rs — 1:1 port of llama.cpp `common/peg-parser.cpp` + `common/peg-parser.h`
//! and the small `common/trie.cpp` + `common/trie.h` kit it uses, baseline
//! bd4f514db1. This is the PGE/PEG parsing kit: parser combinators, the parse
//! executor with streaming (`NEED_MORE_INPUT`) semantics, the AST arena and the
//! GBNF grammar generation from a parser arena.
//!
//! C++ → Rust reference map (all line numbers = pinned tree):
//!   * `common_peg_parse_result_type`     — peg-parser.h:67-73
//!   * `common_peg_invalid_utf8`          — peg-parser.h:76-79
//!   * `common_peg_ast_node`/`_arena`     — peg-parser.h:81-134, peg-parser.cpp:137-203
//!   * `common_peg_parse_result`          — peg-parser.h:136-160
//!   * parser variants + `common_peg_arena` — peg-parser.h:203-368
//!   * `parser_executor`                  — peg-parser.cpp:225-816
//!   * builder + primitives               — peg-parser.cpp:1072-1387
//!   * GBNF generation (`build_grammar`)  — peg-parser.cpp:1389-1815
//!   * serialization (`to_json`/`from_json`/`save`/`load`) — peg-parser.cpp:1817-2118
//!   * `common_trie` / `common_aho_corasick` — trie.h / trie.cpp
//!
//! Deviations (documented, no C++ counterpart):
//!   * C++ `common_peg_parser` handles with operator `+` / `<<` / `|` map to
//!     builder methods: `p.sequence(&[a, b])` (`+`), `p.spaced(a, b)` (`<<`,
//!     i.e. `a space b`, peg-parser.cpp:1020-1022) and `p.choice(&[a, b])` (`|`).
//!   * AST nodes keep `(start, end)` offsets instead of a `std::string_view`
//!     into the input; the input is passed in by the consumer
//!     (`PegAstArena::node_text`). Same data, no dangling-view hazard.
//!   * `rules_` is insertion-ordered (C++ `std::unordered_map`), so generated
//!     GBNF rule *order* is deterministic here but may differ from the
//!     reference's; the grammar text is semantically identical.
//!   * The DEBUG flag is kept (`is_debug`) but the per-step `fprintf(stderr)`
//!     traces of peg-parser.cpp are not ported (log noise only).
//!   * `arena.parse` returns `Result<ParseResult, String>`: the C++
//!     `throw std::runtime_error("Rule not found: ...")` (peg-parser.cpp:217-223)
//!     becomes `Err`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use crate::json_schema::{
    build_grammar as schema_build_grammar, gbnf_format_literal, schema_from_json, GrammarOptions,
    Json, SchemaDocument, SchemaKind, SchemaNode,
};

pub type ParserId = usize;
pub const INVALID_PARSER_ID: ParserId = usize::MAX;
pub type AstId = usize;
pub const INVALID_AST_ID: AstId = usize::MAX;

// ---------------------------------------------------------------------------
// utf8 codepoint parse — `common_parse_utf8_codepoint` (common/unicode.cpp:17-71)
// ---------------------------------------------------------------------------

pub enum Utf8Status {
    Success,
    Incomplete,
    Invalid,
}

pub struct Utf8Result {
    pub status: Utf8Status,
    pub codepoint: u32,
    pub bytes_consumed: usize,
}

/// `common_parse_utf8_codepoint(input, offset)` (unicode.cpp:17). Returns the
/// codepoint at `offset`, or `Incomplete`/`Invalid` with the number of bytes
/// examined (`bytes_consumed`, 0 for a past-the-end offset).
pub fn parse_utf8_codepoint(input: &[u8], offset: usize) -> Utf8Result {
    let incomplete = |n| Utf8Result {
        status: Utf8Status::Incomplete,
        codepoint: 0,
        bytes_consumed: n,
    };
    let invalid = |n| Utf8Result {
        status: Utf8Status::Invalid,
        codepoint: 0,
        bytes_consumed: n,
    };
    let success = |cp: u32, n: usize| Utf8Result {
        status: Utf8Status::Success,
        codepoint: cp,
        bytes_consumed: n,
    };

    if offset >= input.len() {
        return incomplete(0);
    }
    // ASCII fast path
    if input[offset] & 0x80 == 0 {
        return success(input[offset] as u32, 1);
    }
    // Invalid: continuation byte as first byte
    if input[offset] & 0x40 == 0 {
        return invalid(1);
    }
    // 2-byte sequence
    if input[offset] & 0x20 == 0 {
        if offset + 1 >= input.len() {
            return incomplete(1);
        }
        if input[offset + 1] & 0xc0 != 0x80 {
            return invalid(1);
        }
        let result = ((input[offset] as u32 & 0x1f) << 6) | (input[offset + 1] as u32 & 0x3f);
        return success(result, 2);
    }
    // 3-byte sequence
    if input[offset] & 0x10 == 0 {
        // Check one byte at a time so a bad byte is reported before a short input
        for i in 1..3 {
            if offset + i >= input.len() {
                return incomplete(i);
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return invalid(i);
            }
        }
        let result = ((input[offset] as u32 & 0x0f) << 12)
            | ((input[offset + 1] as u32 & 0x3f) << 6)
            | (input[offset + 2] as u32 & 0x3f);
        return success(result, 3);
    }
    // 4-byte sequence
    if input[offset] & 0x08 == 0 {
        for i in 1..4 {
            if offset + i >= input.len() {
                return incomplete(i);
            }
            if input[offset + i] & 0xc0 != 0x80 {
                return invalid(i);
            }
        }
        let result = ((input[offset] as u32 & 0x07) << 18)
            | ((input[offset + 1] as u32 & 0x3f) << 12)
            | ((input[offset + 2] as u32 & 0x3f) << 6)
            | (input[offset + 3] as u32 & 0x3f);
        return success(result, 4);
    }
    // 5- and 6-byte sequences are not valid UTF-8 (llama.cpp's decoder accepts
    // them, but the pinned decoder rejects: unicode.cpp:71-75 returns INVALID)
    invalid(1)
}

// ---------------------------------------------------------------------------
// common_trie — trie.h / trie.cpp
// ---------------------------------------------------------------------------

/// `struct common_trie` (trie.h:15-56) — trie over UTF-8 codepoints, used by
/// the `until` parser and the GBNF exclusion grammar.
#[derive(Default)]
pub struct Trie {
    /// node → (codepoint → node index); `node.pattern >= 0` marks a word end.
    nodes: Vec<TrieNode>,
    n_patterns: i32,
}

struct TrieNode {
    children: BTreeMap<u32, usize>,
    /// index of the pattern ending at this node, -1 if none (trie.h:26)
    pattern: i32,
}

impl Default for TrieNode {
    fn default() -> Self {
        TrieNode {
            children: BTreeMap::new(),
            pattern: -1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrieMatch {
    NoMatch,
    PartialMatch,
    CompleteMatch,
}

impl Trie {
    pub fn new() -> Self {
        let mut t = Trie {
            nodes: Vec::new(),
            n_patterns: 0,
        };
        t.create_node(); // root
        t
    }

    pub fn from_words(words: &[String]) -> Self {
        let mut t = Trie::new();
        for w in words {
            t.insert(w);
        }
        t
    }

    fn create_node(&mut self) -> usize {
        let index = self.nodes.len();
        self.nodes.push(TrieNode::default());
        index
    }

    /// `check_at` (trie.cpp:12-49): does a delimiter start at `start_pos`?
    pub fn check_at(&self, sv: &[u8], start_pos: usize) -> TrieMatch {
        let mut current = 0usize; // root
        let mut pos = start_pos;
        while pos < sv.len() {
            let result = parse_utf8_codepoint(sv, pos);
            if !matches!(result.status, Utf8Status::Success) {
                break;
            }
            let Some(&next) = self.nodes[current].children.get(&result.codepoint) else {
                return TrieMatch::NoMatch;
            };
            current = next;
            pos += result.bytes_consumed;
            if self.nodes[current].pattern >= 0 {
                return TrieMatch::CompleteMatch;
            }
        }
        if current != 0 {
            // in the middle of a potential match
            return TrieMatch::PartialMatch;
        }
        TrieMatch::NoMatch
    }

    /// `insert(word)` (trie.cpp:51-63) — as a sequence of codepoints.
    pub fn insert(&mut self, word: &str) -> i32 {
        let mut symbols = Vec::new();
        let bytes = word.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let result = parse_utf8_codepoint(bytes, pos);
            if !matches!(result.status, Utf8Status::Success) {
                break;
            }
            symbols.push(result.codepoint);
            pos += result.bytes_consumed;
        }
        self.insert_symbols(&symbols)
    }

    /// `insert(symbols)` (trie.cpp:65-79)
    pub fn insert_symbols(&mut self, symbols: &[u32]) -> i32 {
        let mut current = 0usize;
        for &ch in symbols {
            if let Some(&next) = self.nodes[current].children.get(&ch) {
                current = next;
            } else {
                let child = self.create_node();
                self.nodes[current].children.insert(ch, child);
                current = child;
            }
        }
        if self.nodes[current].pattern < 0 {
            self.nodes[current].pattern = self.n_patterns;
            self.n_patterns += 1;
        }
        self.nodes[current].pattern
    }
}

// ---------------------------------------------------------------------------
// common_aho_corasick — trie.h:59-85 / trie.cpp:81-123
// ---------------------------------------------------------------------------

pub struct AhoCorasick {
    trie: Trie,
    fail: Vec<usize>,
    /// states in BFS order
    pub order: Vec<usize>,
    /// longest pattern ending at each state (directly or via a suffix link)
    pub m: Vec<i32>,
    /// every character with a transition
    pub alphabet: BTreeSet<u32>,
}

impl AhoCorasick {
    pub fn from_strings(strings: &[String]) -> Self {
        AhoCorasick::from_trie(Trie::from_words(strings))
    }

    pub fn from_trie(trie: Trie) -> Self {
        // trie.cpp:81-105: BFS failure links + BFS order
        let n = trie.nodes.len();
        let mut fail = vec![0usize; n];
        let mut order = Vec::with_capacity(n);
        let mut m = vec![-1i32; n];
        let mut alphabet = BTreeSet::new();

        // The C++ builds `order`/`fail` in one BFS pass, then resolves `match`
        // in a second pass over `order` (trie.cpp:107-123).
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        // edges snapshot in BFS order (children maps are BTreeMap = sorted by cp,
        // matching std::map iteration)
        let mut edges: Vec<Vec<(u32, usize)>> = Vec::new();
        while let Some(u) = queue.pop_front() {
            order.push(u);
            let kids: Vec<(u32, usize)> = trie.nodes[u]
                .children
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            for &(ch, v) in &kids {
                alphabet.insert(ch);
                if u != 0 {
                    let mut f = fail[u];
                    while f != 0 && !trie.nodes[f].children.contains_key(&ch) {
                        f = fail[f];
                    }
                    // trie.cpp:96-103
                    if let Some(&w) = trie.nodes[f].children.get(&ch) {
                        if w != v {
                            fail[v] = w;
                        }
                    }
                }
                queue.push_back(v);
            }
            while edges.len() < u {
                edges.push(Vec::new());
            }
            if edges.len() == u {
                edges.push(kids);
            } else {
                edges[u] = kids;
            }
        }
        let _ = &mut edges;

        // trie.cpp:107-123: match via suffix links in BFS order
        for &u in &order {
            m[u] = trie.nodes[u].pattern;
            if m[u] < 0 && u != 0 {
                m[u] = m[fail[u]];
            }
        }

        AhoCorasick {
            trie,
            fail,
            order,
            m,
            alphabet,
        }
    }

    pub fn num_states(&self) -> usize {
        self.trie.nodes.len()
    }

    pub fn is_terminal(&self, s: usize) -> bool {
        self.m[s] >= 0
    }

    /// `next(state, ch)` (trie.cpp:113-123): follow failure links until a
    /// transition on `ch` exists.
    pub fn next(&self, mut state: usize, ch: u32) -> usize {
        loop {
            if let Some(&v) = self.trie.nodes[state].children.get(&ch) {
                return v;
            }
            if state == 0 {
                return 0;
            }
            state = self.fail[state];
        }
    }
}

// ---------------------------------------------------------------------------
// parse result / AST (peg-parser.h:67-199, peg-parser.cpp:21-28,137-203)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParseResultType {
    Fail = 0,
    Success = 1,
    NeedMoreInput = 2,
}

pub fn parse_result_type_name(t: ParseResultType) -> &'static str {
    match t {
        ParseResultType::Fail => "fail",
        ParseResultType::Success => "success",
        ParseResultType::NeedMoreInput => "need_more_input",
    }
}

/// `common_peg_invalid_utf8` (peg-parser.h:76-79): a run of input bytes that
/// does not decode as UTF-8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidUtf8 {
    pub pos: usize,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct AstNode {
    pub id: AstId,
    // NB: kept for the C++ layout; the arena index is the id
    pub rule: String,
    pub tag: String,
    pub start: usize,
    pub end: usize,
    pub children: Vec<AstId>,
    pub is_partial: bool,
    /// Invalid UTF-8 inside the node, in ascending order
    pub invalid_utf8: Vec<InvalidUtf8>,
}

impl AstNode {
    /// `sanitized_text()` (peg-parser.cpp:169-186): text with every invalid
    /// run replaced by U+FFFD.
    pub fn sanitized_text(&self, input: &[u8]) -> String {
        if self.invalid_utf8.is_empty() {
            return String::from_utf8_lossy(&input[self.start..self.end]).into_owned();
        }
        let mut out = String::new();
        let mut seg_start = self.start;
        for invalid in &self.invalid_utf8 {
            out.push_str(&String::from_utf8_lossy(&input[seg_start..invalid.pos]));
            out.push('\u{FFFD}');
            seg_start = invalid.pos + invalid.len;
        }
        out.push_str(&String::from_utf8_lossy(&input[seg_start..self.end]));
        out
    }
}

/// `common_peg_ast_arena` (peg-parser.h:103-134)
#[derive(Default)]
pub struct PegAstArena {
    nodes: Vec<AstNode>,
}

impl PegAstArena {
    /// `add_node` (peg-parser.h:106-119)
    pub fn add_node(
        &mut self,
        rule: &str,
        tag: &str,
        start: usize,
        end: usize,
        children: Vec<AstId>,
        is_partial: bool,
        invalid_utf8: Vec<InvalidUtf8>,
    ) -> AstId {
        let id = self.nodes.len();
        self.nodes.push(AstNode {
            id,
            rule: rule.to_string(),
            tag: tag.to_string(),
            start,
            end,
            children,
            is_partial,
            invalid_utf8,
        });
        id
    }

    pub fn get(&self, id: AstId) -> &AstNode {
        &self.nodes[id]
    }

    /// node text — the C++ keeps a `std::string_view` into the context input
    pub fn node_text<'a>(&self, id: AstId, input: &'a str) -> &'a str {
        let n = self.get(id);
        &input[n.start..n.end]
    }

    /// `find_by_tag` (peg-parser.cpp:137-151)
    pub fn find_by_tag(&self, parent: &AstNode, tag: &str, max_depth: i32) -> AstId {
        for &child_id in &parent.children {
            let child = self.get(child_id);
            if child.tag == tag {
                return child_id;
            }
            if max_depth > 1 {
                let result = self.find_by_tag(child, tag, max_depth - 1);
                if result != INVALID_AST_ID {
                    return result;
                }
            }
        }
        INVALID_AST_ID
    }

    /// `find_by_rule` (peg-parser.cpp:153-167)
    pub fn find_by_rule(&self, parent: &AstNode, rule: &str, max_depth: i32) -> AstId {
        for &child_id in &parent.children {
            let child = self.get(child_id);
            if child.rule == rule {
                return child_id;
            }
            if max_depth > 1 {
                let result = self.find_by_rule(child, rule, max_depth - 1);
                if result != INVALID_AST_ID {
                    return result;
                }
            }
        }
        INVALID_AST_ID
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// `visit(id, visitor)` (peg-parser.cpp:188-197)
    pub fn visit<F: FnMut(&AstNode)>(&self, id: AstId, visitor: &mut F) {
        if id == INVALID_AST_ID {
            return;
        }
        let node = self.get(id);
        visitor(node);
        for i in 0..node.children.len() {
            let child = node.children[i];
            self.visit(child, visitor);
        }
    }

    /// `visit(result, visitor)` (peg-parser.cpp:199-203)
    pub fn visit_result<F: FnMut(&AstNode)>(&self, result: &ParseResult, visitor: &mut F) {
        for i in 0..result.nodes.len() {
            self.visit(result.nodes[i], visitor);
        }
    }

    /// `dump()` (peg-parser.cpp:840-863)
    pub fn dump(&self) -> String {
        let mut oss = String::new();
        for node in &self.nodes {
            self.bfs_node(node, 0, &mut oss);
        }
        oss
    }

    fn bfs_node(&self, node: &AstNode, indent: usize, oss: &mut String) {
        for _ in 0..indent {
            oss.push_str("  ");
        }
        oss.push_str(&format!("NODE {}", node.id));
        if !node.rule.is_empty() {
            oss.push_str(&format!(" (rule {})", node.rule));
        }
        if !node.tag.is_empty() {
            oss.push_str(&format!(" (tag {})", node.tag));
        }
        oss.push_str(" ['']\n"); // text omitted: needs input (deviation, debug only)
        for i in 0..node.children.len() {
            let child = self.get(node.children[i]);
            self.bfs_node(child, indent + 1, oss);
        }
    }
}

/// `common_peg_parse_result` (peg-parser.h:136-160)
#[derive(Clone, Debug)]
pub struct ParseResult {
    pub ty: ParseResultType,
    pub start: usize,
    pub end: usize,
    pub nodes: Vec<AstId>,
    /// Invalid UTF-8 consumed by this result, carried up to the enclosing AST nodes
    pub invalid_utf8: Vec<InvalidUtf8>,
}

impl Default for ParseResult {
    fn default() -> Self {
        ParseResult {
            ty: ParseResultType::Fail,
            start: 0,
            end: 0,
            nodes: Vec::new(),
            invalid_utf8: Vec::new(),
        }
    }
}

impl ParseResult {
    pub fn new(ty: ParseResultType, start: usize) -> Self {
        ParseResult {
            ty,
            start,
            end: start,
            nodes: Vec::new(),
            invalid_utf8: Vec::new(),
        }
    }

    pub fn new_span(ty: ParseResultType, start: usize, end: usize) -> Self {
        ParseResult {
            ty,
            start,
            end,
            nodes: Vec::new(),
            invalid_utf8: Vec::new(),
        }
    }

    pub fn new_full(
        ty: ParseResultType,
        start: usize,
        end: usize,
        nodes: Vec<AstId>,
        invalid_utf8: Vec<InvalidUtf8>,
    ) -> Self {
        ParseResult {
            ty,
            start,
            end,
            nodes,
            invalid_utf8,
        }
    }

    pub fn fail(&self) -> bool {
        self.ty == ParseResultType::Fail
    }
    pub fn need_more_input(&self) -> bool {
        self.ty == ParseResultType::NeedMoreInput
    }
    pub fn success(&self) -> bool {
        self.ty == ParseResultType::Success
    }
}

pub type ParseFlags = u32;
pub const PARSE_FLAG_NONE: ParseFlags = 0;
pub const PARSE_FLAG_LENIENT: ParseFlags = 1 << 0;
pub const PARSE_FLAG_DEBUG: ParseFlags = 1 << 1;

/// `common_peg_parse_context` (peg-parser.h:184-203)
pub struct ParseContext {
    /// `input` — the bytes being parsed (peg-parser.h:186)
    pub input: String,
    /// `tokens` — token ids aligned 1:1 with `input`'s bytes
    /// (`LLAMA_TOKEN_NULL` on continuation bytes), e.g.
    /// input  = [h,   e,  l,  l,  o,  _,  w,  o,  r,  l,  d]
    /// tokens = [id, -1, -1, -1, -1, id, -1, -1, -1, -1, -1]
    /// (peg-parser.h:187; 18b5f8b18)
    pub tokens: Vec<i32>,
    pub flags: ParseFlags,
    pub ast: PegAstArena,
    pub parse_depth: i32,
}

impl ParseContext {
    pub fn new(input: &str, flags: ParseFlags) -> Self {
        ParseContext {
            input: input.to_string(),
            tokens: Vec::new(),
            flags,
            ast: PegAstArena::default(),
            parse_depth: 0,
        }
    }

    /// `common_peg_parse_context(std::string, std::vector<llama_token>, flags)`
    /// (peg-parser.h:200-203, 18b5f8b18) — the token-aligned constructor;
    /// asserts the vectors are byte-aligned when tokens are present.
    pub fn new_with_tokens(input: String, tokens: Vec<i32>, flags: ParseFlags) -> Self {
        assert!(tokens.is_empty() || tokens.len() == input.len());
        ParseContext {
            input,
            tokens,
            flags,
            ast: PegAstArena::default(),
            parse_depth: 0,
        }
    }

    pub fn is_lenient(&self) -> bool {
        self.flags & PARSE_FLAG_LENIENT != 0
    }
    pub fn is_debug(&self) -> bool {
        self.flags & PARSE_FLAG_DEBUG != 0
    }
}

// ---------------------------------------------------------------------------
// parser variants (peg-parser.h:203-324)
// ---------------------------------------------------------------------------

/// `common_peg_chars_parser::char_range` (peg-parser.h:241-245)
#[derive(Clone, Debug, PartialEq)]
pub struct CharRange {
    pub start: u32,
    pub end: u32,
}

impl CharRange {
    pub fn contains(&self, codepoint: u32) -> bool {
        codepoint >= self.start && codepoint <= self.end
    }
}

#[derive(Clone)]
pub enum ParserKind {
    Epsilon,                 // peg-parser.h:204
    Start,                   // :206
    End,                     // :208
    Literal(String),         // :210-212
    Sequence(Vec<ParserId>), // :214-216
    Choice(Vec<ParserId>),   // :218-220
    Repetition {
        child: ParserId,
        min: i32,
        max: i32,
    }, // :222-225 (-1 = unbounded)
    And {
        child: ParserId,
    }, // :228
    Not {
        child: ParserId,
    }, // :232
    Any,                     // :236
    Space,                   // :238
    Chars {
        // :240-252
        pattern: String,
        ranges: Vec<CharRange>,
        negated: bool,
        min: i32,
        max: i32,
    },
    Str {
        delimiter: u8,
    }, // :254-256
    Until {
        delimiters: Vec<String>,
    }, // :258-260
    Schema {
        // :262-270
        child: ParserId,
        name: String,
        /// `common_chat_schema_document_ptr` — `None` after a serialization
        /// round trip (the reference's load() leaves `node == nullptr`)
        doc: Option<Rc<SchemaDocument>>,
        node: crate::json_schema::NodeId,
        raw: bool,
    },
    Rule {
        name: String,
        child: ParserId,
        trigger: bool,
    }, // :272-276
    Ref {
        name: String,
    }, // :278-280
    Atomic {
        child: ParserId,
    }, // :282-284
    Tag {
        child: ParserId,
        tag: String,
    }, // :286-289
    Gbnf {
        child: ParserId,
        grammar: String,
    }, // :291-294
    Ac {
        child: ParserId,
        delimiters: Vec<String>,
    }, // :296-299
}

impl std::fmt::Debug for ParserKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParserKind::Epsilon => write!(f, "Epsilon"),
            ParserKind::Start => write!(f, "Start"),
            ParserKind::End => write!(f, "End"),
            ParserKind::Literal(l) => write!(f, "Literal({l})"),
            ParserKind::Sequence(_) => write!(f, "Sequence(..)"),
            ParserKind::Choice(_) => write!(f, "Choice(..)"),
            ParserKind::Repetition { child, min, max } => {
                write!(f, "Repetition({child}, {min}, {max})")
            }
            ParserKind::And { child } => write!(f, "And({child})"),
            ParserKind::Not { child } => write!(f, "Not({child})"),
            ParserKind::Any => write!(f, "Any"),
            ParserKind::Space => write!(f, "Space"),
            ParserKind::Chars { pattern, .. } => write!(f, "CharRepeat({pattern})"),
            ParserKind::Str { delimiter } => write!(f, "String({})", *delimiter as char),
            ParserKind::Until { delimiters } => write!(f, "Until({delimiters:?})"),
            ParserKind::Schema { name, .. } => write!(f, "Schema({name})"),
            ParserKind::Rule { name, .. } => write!(f, "Rule({name})"),
            ParserKind::Ref { name } => write!(f, "Ref({name})"),
            ParserKind::Atomic { child } => write!(f, "Atomic({child})"),
            ParserKind::Tag { tag, .. } => write!(f, "Tag({tag})"),
            ParserKind::Gbnf { grammar, .. } => write!(f, "Gbnf({grammar})"),
            ParserKind::Ac { delimiters, .. } => write!(f, "Ac({delimiters:?})"),
        }
    }
}

// ---------------------------------------------------------------------------
// common_peg_arena (peg-parser.h:326-368)
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct PegArena {
    parsers: Vec<ParserKind>,
    /// insertion-ordered `rules_` (deviation from std::unordered_map, see docs)
    rules: Vec<(String, ParserId)>,
    root: ParserId,
}

impl PegArena {
    pub fn get(&self, id: ParserId) -> &ParserKind {
        &self.parsers[id]
    }

    pub fn size(&self) -> usize {
        self.parsers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parsers.is_empty()
    }

    pub fn root(&self) -> ParserId {
        self.root
    }

    pub fn set_root(&mut self, id: ParserId) {
        self.root = id;
    }

    fn add_parser(&mut self, parser: ParserKind) -> ParserId {
        let id = self.parsers.len();
        self.parsers.push(parser);
        id
    }

    fn add_rule(&mut self, name: &str, id: ParserId) {
        // std::unordered_map::operator[] overwrites existing entries
        if let Some(entry) = self.rules.iter_mut().find(|(n, _)| n == name) {
            entry.1 = id;
        } else {
            self.rules.push((name.to_string(), id));
        }
    }

    /// `get_rule` (peg-parser.cpp:217-223) — C++ throws; here `Err`.
    pub fn get_rule(&self, name: &str) -> Result<ParserId, String> {
        self.rules
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
            .ok_or_else(|| format!("Rule not found: {name}"))
    }

    pub fn has_rule(&self, name: &str) -> bool {
        self.rules.iter().any(|(n, _)| n == name)
    }

    /// `parse(ctx, start)` (peg-parser.cpp:818-823)
    pub fn parse(&self, ctx: &mut ParseContext, start: usize) -> Result<ParseResult, String> {
        if self.root == INVALID_PARSER_ID {
            return Err("No root parser set".to_string());
        }
        self.parse_id(self.root, ctx, start)
    }

    /// `parse(id, ctx, start)` (peg-parser.cpp:825-830)
    pub fn parse_id(
        &self,
        id: ParserId,
        ctx: &mut ParseContext,
        start: usize,
    ) -> Result<ParseResult, String> {
        let parser = &self.parsers[id];
        self.execute(id, parser, ctx, start)
    }

    // ---- parser_executor (peg-parser.cpp:225-816) --------------------------

    fn execute(
        &self,
        _id: ParserId, // NB: kept for the C++ layout (unused: parser is passed directly)
        parser: &ParserKind,
        ctx: &mut ParseContext,
        start_pos: usize,
    ) -> Result<ParseResult, String> {
        let input = ctx.input.as_bytes().to_vec();
        let r = match parser {
            ParserKind::Epsilon => ParseResult::new(ParseResultType::Success, start_pos),
            ParserKind::Start => ParseResult::new(
                if start_pos == 0 {
                    ParseResultType::Success
                } else {
                    ParseResultType::Fail
                },
                start_pos,
            ),
            ParserKind::End => ParseResult::new(
                if start_pos >= input.len() {
                    ParseResultType::Success
                } else {
                    ParseResultType::Fail
                },
                start_pos,
            ),
            ParserKind::Literal(literal) => {
                // peg-parser.cpp:277-293
                let mut pos = start_pos;
                let mut i = 0usize;
                while i < literal.len() {
                    if pos >= input.len() {
                        if !ctx.is_lenient() {
                            return Ok(ParseResult::new(ParseResultType::Fail, start_pos));
                        }
                        return Ok(ParseResult::new_span(
                            ParseResultType::NeedMoreInput,
                            start_pos,
                            pos,
                        ));
                    }
                    if input[pos] != literal.as_bytes()[i] {
                        return Ok(ParseResult::new(ParseResultType::Fail, start_pos));
                    }
                    pos += 1;
                    i += 1;
                }
                ParseResult::new_span(ParseResultType::Success, start_pos, pos)
            }
            ParserKind::Sequence(children) => {
                // peg-parser.cpp:295-347
                ctx.parse_depth += 1;
                let mut pos = start_pos;
                let mut nodes: Vec<AstId> = Vec::new();
                let mut invalid_utf8: Vec<InvalidUtf8> = Vec::new();
                let mut out = None;
                for &child_id in children {
                    let result = self.parse_id(child_id, ctx, pos)?;
                    if result.fail() {
                        out = Some(ParseResult::new_span(
                            ParseResultType::Fail,
                            start_pos,
                            result.end,
                        ));
                        break;
                    }
                    nodes.extend(result.nodes.iter().copied());
                    invalid_utf8.extend(result.invalid_utf8.iter().copied());
                    if result.need_more_input() {
                        out = Some(ParseResult::new_full(
                            ParseResultType::NeedMoreInput,
                            start_pos,
                            result.end,
                            nodes.clone(),
                            invalid_utf8.clone(),
                        ));
                        break;
                    }
                    pos = result.end;
                }
                ctx.parse_depth -= 1;
                out.unwrap_or(ParseResult::new_full(
                    ParseResultType::Success,
                    start_pos,
                    pos,
                    nodes,
                    invalid_utf8,
                ))
            }
            ParserKind::Choice(children) => {
                // peg-parser.cpp:349-382
                let pos = start_pos;
                let mut out = ParseResult::new(ParseResultType::Fail, start_pos);
                for &child_id in children {
                    let result = self.parse_id(child_id, ctx, pos)?;
                    if !result.fail() {
                        out = result;
                        break;
                    }
                }
                out
            }
            ParserKind::Repetition { child, min, max } => {
                // peg-parser.cpp:384-476
                ctx.parse_depth += 1;
                let mut pos = start_pos;
                let mut match_count = 0i32;
                let mut nodes: Vec<AstId> = Vec::new();
                let mut invalid_utf8: Vec<InvalidUtf8> = Vec::new();
                let mut out = None;
                while *max == -1 || match_count < *max {
                    if pos >= input.len() {
                        break;
                    }
                    let result = self.parse_id(*child, ctx, pos)?;
                    if result.success() {
                        // Prevent infinite loop on empty matches
                        if result.end == pos {
                            break;
                        }
                        nodes.extend(result.nodes.iter().copied());
                        invalid_utf8.extend(result.invalid_utf8.iter().copied());
                        pos = result.end;
                        match_count += 1;
                        continue;
                    }
                    if result.need_more_input() {
                        nodes.extend(result.nodes.iter().copied());
                        invalid_utf8.extend(result.invalid_utf8.iter().copied());
                        out = Some(ParseResult::new_full(
                            ParseResultType::NeedMoreInput,
                            start_pos,
                            result.end,
                            nodes.clone(),
                            invalid_utf8.clone(),
                        ));
                        break;
                    }
                    // Child failed - stop trying
                    break;
                }
                if out.is_none() {
                    if *min > 0 && match_count < *min {
                        if pos >= input.len() && ctx.is_lenient() {
                            out = Some(ParseResult::new_full(
                                ParseResultType::NeedMoreInput,
                                start_pos,
                                pos,
                                nodes,
                                invalid_utf8,
                            ));
                        } else {
                            out =
                                Some(ParseResult::new_span(ParseResultType::Fail, start_pos, pos));
                        }
                    } else {
                        out = Some(ParseResult::new_full(
                            ParseResultType::Success,
                            start_pos,
                            pos,
                            nodes,
                            invalid_utf8,
                        ));
                    }
                }
                ctx.parse_depth -= 1;
                out.unwrap()
            }
            ParserKind::And { child } => {
                // peg-parser.cpp:478-482
                let result = self.parse_id(*child, ctx, start_pos)?;
                ParseResult::new(result.ty, start_pos)
            }
            ParserKind::Not { child } => {
                // peg-parser.cpp:484-499
                let result = self.parse_id(*child, ctx, start_pos)?;
                if result.success() {
                    ParseResult::new(ParseResultType::Fail, start_pos)
                } else if result.need_more_input() {
                    ParseResult::new(ParseResultType::NeedMoreInput, start_pos)
                } else {
                    ParseResult::new(ParseResultType::Success, start_pos)
                }
            }
            ParserKind::Any => {
                // peg-parser.cpp:501-515 — a single UTF-8 codepoint
                let result = parse_utf8_codepoint(&input, start_pos);
                match result.status {
                    Utf8Status::Incomplete => {
                        if !ctx.is_lenient() {
                            ParseResult::new(ParseResultType::Fail, start_pos)
                        } else {
                            ParseResult::new(ParseResultType::NeedMoreInput, start_pos)
                        }
                    }
                    Utf8Status::Invalid => ParseResult::new(ParseResultType::Fail, start_pos),
                    Utf8Status::Success => ParseResult::new_span(
                        ParseResultType::Success,
                        start_pos,
                        start_pos + result.bytes_consumed,
                    ),
                }
            }
            ParserKind::Space => {
                // peg-parser.cpp:517-529
                let mut pos = start_pos;
                while pos < input.len() {
                    let c = input[pos];
                    if is_c_space(c) {
                        pos += 1;
                    } else {
                        break;
                    }
                }
                ParseResult::new_span(ParseResultType::Success, start_pos, pos)
            }
            ParserKind::Chars {
                ranges,
                negated,
                min,
                max,
                ..
            } => {
                // peg-parser.cpp:531-593
                self.exec_chars(&input, ranges, *negated, *min, *max, start_pos, ctx)
            }
            ParserKind::Str { delimiter } => {
                // peg-parser.cpp:643-683
                self.exec_string(&input, start_pos, *delimiter, ctx)?
            }
            ParserKind::Until { delimiters } => {
                // peg-parser.cpp:685-732
                self.exec_until(&input, delimiters, start_pos, ctx)
            }
            ParserKind::Schema { child, .. } => {
                // peg-parser.cpp:734-736
                self.parse_id(*child, ctx, start_pos)?
            }
            ParserKind::Rule { name, child, .. } => {
                // peg-parser.cpp:738-763
                let result = self.parse_id(*child, ctx, start_pos)?;
                if !result.fail() {
                    let node_id = ctx.ast.add_node(
                        name,
                        "",
                        result.start,
                        result.end,
                        result.nodes.clone(),
                        result.need_more_input(),
                        result.invalid_utf8.clone(),
                    );
                    // C++ moves invalid_utf8 into the node; the returned result's
                    // vector is moved-from (empty in practice)
                    return Ok(ParseResult::new_full(
                        result.ty,
                        result.start,
                        result.end,
                        vec![node_id],
                        Vec::new(),
                    ));
                }
                result
            }
            ParserKind::Tag { child, tag } => {
                // peg-parser.cpp:765-793
                let result = self.parse_id(*child, ctx, start_pos)?;
                if !result.fail() {
                    let node_id = ctx.ast.add_node(
                        "",
                        tag,
                        result.start,
                        result.end,
                        result.nodes.clone(),
                        result.need_more_input(),
                        result.invalid_utf8.clone(),
                    );
                    return Ok(ParseResult::new_full(
                        result.ty,
                        result.start,
                        result.end,
                        vec![node_id],
                        Vec::new(),
                    ));
                }
                result
            }
            ParserKind::Ref { name } => {
                // peg-parser.cpp:795-798
                let rule_id = self.get_rule(name)?;
                self.parse_id(rule_id, ctx, start_pos)?
            }
            ParserKind::Atomic { child } => {
                // peg-parser.cpp:800-807
                let mut result = self.parse_id(*child, ctx, start_pos)?;
                if result.need_more_input() {
                    // Clear nodes so they don't propagate up.
                    result.nodes.clear();
                }
                result
            }
            ParserKind::Gbnf { child, .. } => {
                // peg-parser.cpp:809-811
                self.parse_id(*child, ctx, start_pos)?
            }
            ParserKind::Ac { child, .. } => {
                // peg-parser.cpp:813-815
                self.parse_id(*child, ctx, start_pos)?
            }
        };
        Ok(r)
    }

    fn exec_chars(
        &self,
        input: &[u8],
        ranges: &[CharRange],
        negated: bool,
        min: i32,
        max: i32,
        start_pos: usize,
        ctx: &ParseContext,
    ) -> ParseResult {
        // peg-parser.cpp:531-593
        let mut pos = start_pos;
        let mut match_count = 0i32;
        while max == -1 || match_count < max {
            let result = parse_utf8_codepoint(input, pos);
            if matches!(result.status, Utf8Status::Incomplete) {
                if match_count >= min {
                    return ParseResult::new_span(ParseResultType::Success, start_pos, pos);
                }
                if !ctx.is_lenient() {
                    return ParseResult::new(ParseResultType::Fail, start_pos);
                }
                return ParseResult::new_span(ParseResultType::NeedMoreInput, start_pos, pos);
            }
            if matches!(result.status, Utf8Status::Invalid) {
                // Malformed UTF-8 in input
                if match_count >= min {
                    return ParseResult::new_span(ParseResultType::Success, start_pos, pos);
                }
                return ParseResult::new(ParseResultType::Fail, start_pos);
            }
            let mut matches = ranges.iter().any(|range| range.contains(result.codepoint));
            if negated {
                matches = !matches;
            }
            if matches {
                pos += result.bytes_consumed;
                match_count += 1;
            } else {
                break;
            }
        }
        if match_count < min {
            if pos >= input.len() && ctx.is_lenient() {
                return ParseResult::new_span(ParseResultType::NeedMoreInput, start_pos, pos);
            }
            return ParseResult::new_span(ParseResultType::Fail, start_pos, pos);
        }
        ParseResult::new_span(ParseResultType::Success, start_pos, pos)
    }

    // escape handling for the string parser (peg-parser.cpp:595-641)
    fn handle_escape_sequence(
        input: &[u8],
        start: usize,
        pos: &mut usize,
        delimiter: u8,
        lenient: bool,
    ) -> ParseResult {
        let save = *pos;
        *pos += 1; // consume '\'
        if *pos >= input.len() {
            if !lenient {
                return ParseResult::new(ParseResultType::Fail, start);
            }
            *pos = save; // suppress unmatched '\'
            return ParseResult::new_span(ParseResultType::NeedMoreInput, start, *pos);
        }
        let c = input[*pos];
        if c == delimiter
            || c == b'\\'
            || c == b'/'
            || c == b'b'
            || c == b'f'
            || c == b'n'
            || c == b'r'
            || c == b't'
        {
            *pos += 1;
            return ParseResult::new_span(ParseResultType::Success, start, *pos);
        }
        if c == b'u' {
            let result = Self::handle_unicode_escape(input, start, pos, lenient);
            if result.need_more_input() {
                *pos = save; // suppress incomplete sequence
                return ParseResult::new_span(ParseResultType::NeedMoreInput, start, *pos);
            }
            return result;
        }
        ParseResult::new(ParseResultType::Fail, start)
    }

    fn handle_unicode_escape(
        input: &[u8],
        start: usize,
        pos: &mut usize,
        lenient: bool,
    ) -> ParseResult {
        *pos += 1; // consume 'u'
        for _ in 0..4 {
            if *pos >= input.len() {
                if !lenient {
                    return ParseResult::new(ParseResultType::Fail, start);
                }
                return ParseResult::new_span(ParseResultType::NeedMoreInput, start, *pos);
            }
            if !input[*pos].is_ascii_hexdigit() {
                return ParseResult::new(ParseResultType::Fail, start);
            }
            *pos += 1;
        }
        ParseResult::new_span(ParseResultType::Success, start, *pos)
    }

    fn exec_string(
        &self,
        input: &[u8],
        start_pos: usize,
        delimiter: u8,
        ctx: &ParseContext,
    ) -> Result<ParseResult, String> {
        // peg-parser.cpp:643-683
        let mut pos = start_pos;
        while pos < input.len() {
            let c = input[pos];
            if c == delimiter {
                // Found closing delimiter - success (don't consume it)
                return Ok(ParseResult::new_span(
                    ParseResultType::Success,
                    start_pos,
                    pos,
                ));
            }
            if c == b'\\' {
                let result = Self::handle_escape_sequence(
                    input,
                    start_pos,
                    &mut pos,
                    delimiter,
                    ctx.is_lenient(),
                );
                if !result.success() {
                    return Ok(result);
                }
            } else {
                let utf8_result = parse_utf8_codepoint(input, pos);
                match utf8_result.status {
                    Utf8Status::Incomplete => {
                        if !ctx.is_lenient() {
                            return Ok(ParseResult::new(ParseResultType::Fail, start_pos));
                        }
                        return Ok(ParseResult::new_span(
                            ParseResultType::NeedMoreInput,
                            start_pos,
                            pos,
                        ));
                    }
                    Utf8Status::Invalid => {
                        return Ok(ParseResult::new(ParseResultType::Fail, start_pos));
                    }
                    Utf8Status::Success => {
                        pos += utf8_result.bytes_consumed;
                    }
                }
            }
        }
        // Reached end without finding closing quote
        if !ctx.is_lenient() {
            return Ok(ParseResult::new_span(ParseResultType::Fail, start_pos, pos));
        }
        Ok(ParseResult::new_span(
            ParseResultType::NeedMoreInput,
            start_pos,
            pos,
        ))
    }

    fn exec_until(
        &self,
        input: &[u8],
        delimiters: &[String],
        start_pos: usize,
        ctx: &ParseContext,
    ) -> ParseResult {
        // peg-parser.cpp:685-732
        let matcher = Trie::from_words(delimiters);

        let mut pos = start_pos;
        let mut last_valid_pos = start_pos;
        let mut invalid_utf8: Vec<InvalidUtf8> = Vec::new();

        while pos < input.len() {
            let utf8_result = parse_utf8_codepoint(input, pos);
            if matches!(utf8_result.status, Utf8Status::Incomplete) && ctx.is_lenient() {
                // The rest of the sequence may still arrive, return what we have so far
                return ParseResult::new_full(
                    ParseResultType::NeedMoreInput,
                    start_pos,
                    last_valid_pos,
                    Vec::new(),
                    invalid_utf8,
                );
            }
            if !matches!(utf8_result.status, Utf8Status::Success) {
                // Malformed UTF-8, or a sequence truncated by the end of a complete input.
                invalid_utf8.push(InvalidUtf8 {
                    pos,
                    len: utf8_result.bytes_consumed,
                });
                pos += utf8_result.bytes_consumed;
                last_valid_pos = pos;
                continue;
            }
            // Check if a delimiter starts at this position
            let m = matcher.check_at(input, pos);
            if m == TrieMatch::CompleteMatch {
                // Found a complete delimiter, return everything before it
                return ParseResult::new_full(
                    ParseResultType::Success,
                    start_pos,
                    pos,
                    Vec::new(),
                    invalid_utf8,
                );
            }
            if m == TrieMatch::PartialMatch {
                // Found a partial match extending to end of input, return everything before it
                return ParseResult::new_full(
                    ParseResultType::Success,
                    start_pos,
                    pos,
                    Vec::new(),
                    invalid_utf8,
                );
            }
            pos += utf8_result.bytes_consumed;
            last_valid_pos = pos;
        }

        if last_valid_pos == input.len() && ctx.is_lenient() {
            // Reached the end of a partial stream, there might still be more input.
            return ParseResult::new_full(
                ParseResultType::NeedMoreInput,
                start_pos,
                last_valid_pos,
                Vec::new(),
                invalid_utf8,
            );
        }
        ParseResult::new_full(
            ParseResultType::Success,
            start_pos,
            last_valid_pos,
            Vec::new(),
            invalid_utf8,
        )
    }

    // ---- resolve_refs (peg-parser.cpp:832-912) ------------------------------

    fn resolve_ref(&self, id: ParserId) -> Result<ParserId, String> {
        if let ParserKind::Ref { name } = &self.parsers[id] {
            return self.get_rule(name);
        }
        Ok(id)
    }

    pub fn resolve_refs(&mut self) -> Result<(), String> {
        // Walk through all parsers and replace refs with their corresponding rule IDs
        for i in 0..self.parsers.len() {
            let child_ids: Vec<ParserId> = match &self.parsers[i] {
                ParserKind::Sequence(children) | ParserKind::Choice(children) => children.clone(),
                ParserKind::Repetition { child, .. }
                | ParserKind::And { child }
                | ParserKind::Not { child }
                | ParserKind::Tag { child, .. }
                | ParserKind::Atomic { child }
                | ParserKind::Gbnf { child, .. }
                | ParserKind::Ac { child, .. }
                | ParserKind::Rule { child, .. }
                | ParserKind::Schema { child, .. } => vec![*child],
                _ => Vec::new(),
            };
            if child_ids.is_empty() {
                continue;
            }
            let resolved: Vec<ParserId> = child_ids
                .iter()
                .map(|&c| self.resolve_ref(c))
                .collect::<Result<_, _>>()?;
            let mut resolved = resolved.into_iter();
            match &mut self.parsers[i] {
                ParserKind::Sequence(children) => {
                    for child in children.iter_mut() {
                        *child = resolved.next().unwrap();
                    }
                }
                ParserKind::Choice(children) => {
                    for child in children.iter_mut() {
                        *child = resolved.next().unwrap();
                    }
                }
                ParserKind::Repetition { child, .. }
                | ParserKind::And { child }
                | ParserKind::Not { child }
                | ParserKind::Tag { child, .. }
                | ParserKind::Atomic { child }
                | ParserKind::Gbnf { child, .. }
                | ParserKind::Ac { child, .. }
                | ParserKind::Rule { child, .. }
                | ParserKind::Schema { child, .. } => {
                    *child = resolved.next().unwrap();
                }
                _ => {}
            }
        }
        // Also flatten root if it's a ref
        if self.root != INVALID_PARSER_ID {
            self.root = self.resolve_ref(self.root)?;
        }
        Ok(())
    }

    // ---- dump (peg-parser.cpp:914-995) --------------------------------------

    pub fn dump(&self, id: ParserId) -> String {
        let mut visited = BTreeSet::new();
        self.dump_impl(id, &mut visited)
    }

    fn dump_impl(&self, id: ParserId, visited: &mut BTreeSet<ParserId>) -> String {
        if visited.contains(&id) {
            return "[cycle]".to_string();
        }
        visited.insert(id);
        match &self.parsers[id] {
            ParserKind::Epsilon => "Epsilon".to_string(),
            ParserKind::Start => "Start".to_string(),
            ParserKind::End => "End".to_string(),
            ParserKind::Literal(l) => format!("Literal({l})"),
            ParserKind::Sequence(children) => {
                let parts: Vec<String> = children
                    .iter()
                    .map(|&c| self.dump_impl(c, visited))
                    .collect();
                format!("Sequence({})", parts.join(", "))
            }
            ParserKind::Choice(children) => {
                let parts: Vec<String> = children
                    .iter()
                    .map(|&c| self.dump_impl(c, visited))
                    .collect();
                format!("Choice({})", parts.join(", "))
            }
            ParserKind::Repetition { child, min, max } => {
                if *max == -1 {
                    format!(
                        "Repetition({}, {}, unbounded)",
                        self.dump_impl(*child, visited),
                        min
                    )
                } else {
                    format!(
                        "Repetition({}, {}, {})",
                        self.dump_impl(*child, visited),
                        min,
                        max
                    )
                }
            }
            ParserKind::And { child } => format!("And({})", self.dump_impl(*child, visited)),
            ParserKind::Not { child } => format!("Not({})", self.dump_impl(*child, visited)),
            ParserKind::Atomic { child } => {
                // the C++ has two atomic branches (peg-parser.cpp:963 and the
                // dead one after tag at :991-993) — the FIRST wins, and it
                // passes the SHARED visited set; only Tag (:988-990) calls
                // dump() with a fresh one
                format!("Atomic({})", self.dump_impl(*child, visited))
            }
            ParserKind::Gbnf { child, grammar } => {
                format!("Gbnf({}, {})", grammar, self.dump_impl(*child, visited))
            }
            ParserKind::Ac { child, delimiters } => format!(
                "Ac({}, {})",
                delimiters.join(" | "),
                self.dump_impl(*child, visited)
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
            ParserKind::Str { delimiter } => {
                format!("String({})", *delimiter as char)
            }
            ParserKind::Until { delimiters } => format!("Until({})", delimiters.join(" | ")),
            ParserKind::Schema {
                child, node, doc, ..
            } => {
                let kind: String = match doc {
                    Some(doc) => doc.node(*node).kind_name().to_string(),
                    None => "null".to_string(), // node == nullptr (peg-parser.cpp:982)
                };
                format!("Schema({}, {})", self.dump_impl(*child, visited), kind)
            }
            ParserKind::Rule { name, child, .. } => {
                format!("Rule({}, {})", name, self.dump_impl(*child, visited))
            }
            ParserKind::Ref { name } => format!("Ref({name})"),
            ParserKind::Tag { child, tag } => {
                format!("Tag({}, {})", tag, self.dump(*child))
            }
        }
    }

    // ---- GBNF generation (peg-parser.cpp:1389-1815) -------------------------

    /// `build_grammar(builder, lazy)` (peg-parser.cpp:1606-1815). Emits rules via
    /// the json-schema grammar builder; returns the final grammar text.
    pub fn build_grammar(&self, lazy: bool) -> Result<String, String> {
        // Merge every schema document referenced by a Schema parser into one
        // arena document (the C++ builder converts `common_chat_schema` nodes by
        // value; the port's converter is document-scoped).
        let mut merger = SchemaMerger::new(self);
        let document = std::mem::replace(
            &mut merger.document,
            SchemaDocument {
                nodes: Vec::new(),
                root: 0,
                refs: BTreeMap::new(),
            },
        );
        let doc = Rc::new(document);
        schema_build_grammar(&doc, GrammarOptions::default(), |builder| {
            let mut gen = GbnfGen {
                arena: self,
                builder,
                merger: &mut merger,
            };
            gen.run(lazy);
        })
    }

    // ---- serialization (peg-parser.cpp:1817-2118) ---------------------------

    /// `to_json()` (peg-parser.cpp:1899-1909)
    pub fn to_json(&self) -> Json {
        let parsers = Json::Array(self.parsers.iter().map(serialize_parser_variant).collect());
        let rules = Json::Object(
            self.rules
                .iter()
                .map(|(name, id)| (name.clone(), Json::Uint(*id as u64)))
                .collect(),
        );
        Json::Object(vec![
            ("parsers".to_string(), parsers),
            ("rules".to_string(), rules),
            ("root".to_string(), Json::Uint(self.root as u64)),
        ])
    }

    /// `from_json` (peg-parser.cpp:2077-2110)
    pub fn from_json(j: &Json) -> Result<PegArena, String> {
        let parsers_json = j
            .at("parsers")
            .filter(|v| v.is_array())
            .ok_or("JSON missing or invalid 'parsers' array")?;
        let rules_json = j
            .at("rules")
            .filter(|v| v.is_object())
            .ok_or("JSON missing or invalid 'rules' object")?;
        let root_json = j.at("root").ok_or("JSON missing 'root' field")?;

        let mut arena = PegArena::default();
        for parser_json in parsers_json.iter() {
            arena.parsers.push(deserialize_parser_variant(parser_json)?);
        }
        for (name, id) in rules_json.items() {
            let id = id.get_i64().map_err(|e| e.to_string())? as usize;
            if id >= arena.parsers.len() {
                return Err(format!("Rule '{name}' references invalid parser ID: {id}"));
            }
            arena.rules.push((name, id));
        }
        let root = match root_json {
            Json::Null => INVALID_PARSER_ID,
            other => other.get_i64().map_err(|e| e.to_string())? as usize,
        };
        if root != INVALID_PARSER_ID && root >= arena.parsers.len() {
            return Err(format!("Root references invalid parser ID: {root}"));
        }
        arena.root = root;
        Ok(arena)
    }

    /// `save()` (peg-parser.cpp:2112-2114)
    pub fn save(&self) -> String {
        self.to_json().dump()
    }

    /// `load(data)` (peg-parser.cpp:2116-2118)
    pub fn load(&mut self, data: &str) -> Result<(), String> {
        *self = PegArena::from_json(&Json::parse(data)?)?;
        Ok(())
    }
}

fn is_c_space(c: u8) -> bool {
    // std::isspace in the "C" locale
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

// ---------------------------------------------------------------------------
// char-class parsing (peg-parser.cpp:30-135)
// ---------------------------------------------------------------------------

fn is_hex_digit(c: u8) -> bool {
    c.is_ascii_digit() || (b'a'..=b'f').contains(&c) || (b'A'..=b'F').contains(&c)
}

/// `parse_hex_escape` (peg-parser.cpp:34-57)
fn parse_hex_escape(s: &[u8], pos: usize, hex_count: usize) -> (u32, usize) {
    if pos + hex_count > s.len() {
        return (0, 0);
    }
    let mut value: u32 = 0;
    for i in 0..hex_count {
        let c = s[pos + i];
        if !is_hex_digit(c) {
            return (0, 0);
        }
        value <<= 4;
        if (b'a'..=b'f').contains(&c) {
            value += (c - b'a') as u32 + 10;
        } else if (b'A'..=b'F').contains(&c) {
            value += (c - b'A') as u32 + 10;
        } else {
            value += (c - b'0') as u32;
        }
    }
    (value, hex_count)
}

/// `parse_char_class_char` (peg-parser.cpp:59-98)
fn parse_char_class_char(content: &[u8], pos: usize) -> (u32, usize) {
    if content[pos] == b'\\' && pos + 1 < content.len() {
        match content[pos + 1] {
            b'x' => {
                let (v, n) = parse_hex_escape(content, pos + 2, 2);
                if n > 0 {
                    return (v, 2 + n);
                }
                (b'x' as u32, 2) // invalid escape → literal 'x'
            }
            b'u' => {
                let (v, n) = parse_hex_escape(content, pos + 2, 4);
                if n > 0 {
                    return (v, 2 + n);
                }
                (b'u' as u32, 2)
            }
            b'U' => {
                let (v, n) = parse_hex_escape(content, pos + 2, 8);
                if n > 0 {
                    return (v, 2 + n);
                }
                (b'U' as u32, 2)
            }
            b'n' => (b'\n' as u32, 2),
            b't' => (b'\t' as u32, 2),
            b'r' => (b'\r' as u32, 2),
            b'\\' => (b'\\' as u32, 2),
            b']' => (b']' as u32, 2),
            b'[' => (b'[' as u32, 2),
            other => (other as u32, 2),
        }
    } else {
        (content[pos] as u32, 1)
    }
}

/// `parse_char_classes` (peg-parser.cpp:100-135)
fn parse_char_classes(classes: &str) -> (Vec<CharRange>, bool) {
    let mut ranges = Vec::new();
    let mut negated = false;

    let mut content = classes;
    if content.starts_with('[') {
        content = &content[1..];
    }
    if content.ends_with(']') {
        content = &content[..content.len() - 1];
    }
    if !content.is_empty() && content.starts_with('^') {
        negated = true;
        content = &content[1..];
    }

    let b = content.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let (start, start_len) = parse_char_class_char(b, i);
        i += start_len;
        if i + 1 < b.len() && b[i] == b'-' {
            // Range detected
            let (end, end_len) = parse_char_class_char(b, i + 1);
            ranges.push(CharRange { start, end });
            i += 1 + end_len;
        } else {
            ranges.push(CharRange { start, end: start });
        }
    }
    (ranges, negated)
}

// ---------------------------------------------------------------------------
// rule_name (peg-parser.cpp:1072-1075)
// ---------------------------------------------------------------------------

fn rule_name(name: &str) -> String {
    // std::regex_replace(name, "[^a-zA-Z0-9-]+", "-")
    let mut out = String::new();
    let mut in_run = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('-');
            in_run = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// common_peg_parser_builder (peg-parser.h:370-554, peg-parser.cpp:1072-1387)
// ---------------------------------------------------------------------------

pub struct PegBuilder {
    arena: PegArena,
    /// error slot for the C++ `throw`s that have no in-band channel here
    /// (`ac` with empty delimiters, peg-parser.cpp:1382-1387)
    error: Option<String>,
}

impl Default for PegBuilder {
    fn default() -> Self {
        PegBuilder {
            arena: PegArena::default(),
            error: None,
        }
    }
}

impl PegBuilder {
    fn add(&mut self, p: ParserKind) -> ParserId {
        self.arena.add_parser(p)
    }

    // Match nothing, always succeed.  S -> ε   (peg-parser.h:379-381)
    pub fn eps(&mut self) -> ParserId {
        self.add(ParserKind::Epsilon)
    }

    // Matches the start of the input.  S -> ^   (peg-parser.h:383-385)
    pub fn start(&mut self) -> ParserId {
        self.add(ParserKind::Start)
    }

    // Matches the end of the input.  S -> $   (peg-parser.h:387-389)
    pub fn end(&mut self) -> ParserId {
        self.add(ParserKind::End)
    }

    // Matches an exact literal string.  S -> "hello"   (peg-parser.h:391-393)
    pub fn literal(&mut self, literal: &str) -> ParserId {
        self.add(ParserKind::Literal(literal.to_string()))
    }

    /// `sequence(parsers)` (peg-parser.cpp:1079-1091) — flattens nested sequences.
    /// This is the C++ `operator+` (peg-parser.cpp:1012-1014).
    pub fn sequence(&mut self, parsers: &[ParserId]) -> ParserId {
        let mut flattened: Vec<ParserId> = Vec::new();
        for &p in parsers {
            if let ParserKind::Sequence(children) = self.arena.get(p) {
                flattened.extend(children.iter().copied());
            } else {
                flattened.push(p);
            }
        }
        self.add(ParserKind::Sequence(flattened))
    }

    /// The C++ `operator<<` (peg-parser.cpp:1020-1022): a sequence separated by
    /// a space parser, i.e. `sequence({a, space(), b})`.
    pub fn spaced(&mut self, a: ParserId, b: ParserId) -> ParserId {
        let sp = self.space();
        self.sequence(&[a, sp, b])
    }

    /// `choice(parsers)` (peg-parser.cpp:1111-1123) — flattens nested choices.
    /// This is the C++ `operator|` (peg-parser.cpp:1016-1018).
    pub fn choice(&mut self, parsers: &[ParserId]) -> ParserId {
        let mut flattened: Vec<ParserId> = Vec::new();
        for &p in parsers {
            if let ParserKind::Choice(children) = self.arena.get(p) {
                flattened.extend(children.iter().copied());
            } else {
                flattened.push(p);
            }
        }
        self.add(ParserKind::Choice(flattened))
    }

    /// Append to a choice (the C++ `operator|=` accumulates into the handle;
    /// here the caller threads the id through).
    pub fn choice_append(&mut self, acc: ParserId, p: ParserId) -> ParserId {
        self.choice(&[acc, p])
    }

    // S -> A+   (peg-parser.h:411)
    pub fn one_or_more(&mut self, p: ParserId) -> ParserId {
        self.repeat3(p, 1, -1)
    }

    // S -> A*   (peg-parser.h:415)
    pub fn zero_or_more(&mut self, p: ParserId) -> ParserId {
        self.repeat3(p, 0, -1)
    }

    // S -> A?   (peg-parser.h:419)
    pub fn optional(&mut self, p: ParserId) -> ParserId {
        self.repeat3(p, 0, 1)
    }

    // Positive lookahead:  S -> &A   (peg-parser.h:423)
    pub fn peek(&mut self, p: ParserId) -> ParserId {
        self.add(ParserKind::And { child: p })
    }

    // Negative lookahead:  S -> !A   (peg-parser.h:427)
    pub fn negate(&mut self, p: ParserId) -> ParserId {
        self.add(ParserKind::Not { child: p })
    }

    // Matches any single character.  S -> .   (peg-parser.h:431)
    pub fn any(&mut self) -> ParserId {
        self.add(ParserKind::Any)
    }

    // S -> [a-z]{m,n}   (peg-parser.h:437)
    pub fn chars(&mut self, classes: &str, min: i32, max: i32) -> ParserId {
        let (ranges, negated) = parse_char_classes(classes);
        self.add(ParserKind::Chars {
            pattern: classes.to_string(),
            ranges,
            negated,
            min,
            max,
        })
    }

    // Lightweight reference to a named rule.  S -> expr   (peg-parser.h:442)
    pub fn ref_(&mut self, name: &str) -> ParserId {
        self.add(ParserKind::Ref {
            name: name.to_string(),
        })
    }

    // S -> [ \t\n]*   (peg-parser.h:446)
    pub fn space(&mut self) -> ParserId {
        self.add(ParserKind::Space)
    }

    // S -> (!delim .)*   (peg-parser.h:451)
    pub fn until(&mut self, delimiter: &str) -> ParserId {
        self.add(ParserKind::Until {
            delimiters: vec![delimiter.to_string()],
        })
    }

    // S -> (!delim .)*   (peg-parser.h:455)
    pub fn until_one_of(&mut self, delimiters: &[&str]) -> ParserId {
        self.add(ParserKind::Until {
            delimiters: delimiters.iter().map(|s| s.to_string()).collect(),
        })
    }

    // Matches everything.  S -> .*   (peg-parser.h:459)
    pub fn rest(&mut self) -> ParserId {
        self.until_one_of(&[])
    }

    // S -> A{m,n}   (peg-parser.h:464)
    pub fn repeat3(&mut self, p: ParserId, min: i32, max: i32) -> ParserId {
        self.add(ParserKind::Repetition { child: p, min, max })
    }

    // S -> A{n}   (peg-parser.h:468)
    pub fn repeat(&mut self, p: ParserId, n: i32) -> ParserId {
        self.repeat3(p, n, n)
    }

    // S -> '"' content '"' space   (peg-parser.h:471)
    pub fn double_quoted_string(&mut self) -> ParserId {
        self.rule_fn("double-quoted-string", |p| {
            let a = p.literal("\"");
            let b = p.string_content(b'"');
            let c = p.literal("\"");
            p.sequence(&[a, b, c])
        })
    }

    // S -> "'" content "'" space   (peg-parser.h:474)
    pub fn single_quoted_string(&mut self) -> ParserId {
        self.rule_fn("single-quoted-string", |p| {
            let a = p.literal("'");
            let b = p.string_content(b'\'');
            let c = p.literal("'");
            p.sequence(&[a, b, c])
        })
    }

    // Both double-quoted and single-quoted styles.  (peg-parser.h:477)
    pub fn quoted_string(&mut self) -> ParserId {
        self.rule_fn("quoted-string", |p| {
            let a = p.double_quoted_string();
            let b = p.single_quoted_string();
            p.choice(&[a, b])
        })
    }

    // String content without the surrounding delimiter.  (peg-parser.h:480)
    pub fn string_content(&mut self, delimiter: u8) -> ParserId {
        self.add(ParserKind::Str { delimiter })
    }

    // ---- JSON parsers (peg-parser.cpp:1220-1295) ----------------------------

    pub fn json_number(&mut self) -> ParserId {
        self.rule_fn("json-number", |p| {
            let digit1_9 = p.chars("[1-9]", 1, 1);
            let digits = p.chars("[0-9]", 1, -1);
            let int_part = {
                let zero = p.literal("0");
                let rest = {
                    let d = p.chars("[0-9]", 0, -1);
                    p.sequence(&[digit1_9, d])
                };
                p.choice(&[zero, rest])
            };
            let frac = {
                let dot = p.literal(".");
                p.sequence(&[dot, digits])
            };
            let exp = {
                let e = p.literal("e");
                let e2 = p.literal("E");
                let e = p.choice(&[e, e2]);
                let sign_c = p.chars("[+-]", 1, 1);
                let sign = p.optional(sign_c);
                p.sequence(&[e, sign, digits])
            };
            // Negative lookahead: only commit the number when the next character
            // can't extend it (peg-parser.cpp:1227-1230).
            let nc = p.chars("[0-9.eE+-]", 1, 1);
            let not_number_continuation = p.negate(nc);
            let minus_lit = p.literal("-");
            let minus = p.optional(minus_lit);
            let frac = p.optional(frac);
            let exp = p.optional(exp);
            p.sequence(&[minus, int_part, frac, exp, not_number_continuation])
        })
    }

    pub fn json_string(&mut self) -> ParserId {
        self.rule_fn("json-string", |p| {
            let a = p.literal("\"");
            let b = p.string_content(b'"');
            let c = p.literal("\"");
            p.sequence(&[a, b, c])
        })
    }

    pub fn json_bool(&mut self) -> ParserId {
        self.rule_fn("json-bool", |p| {
            let a = p.literal("true");
            let b = p.literal("false");
            p.choice(&[a, b])
        })
    }

    pub fn json_null(&mut self) -> ParserId {
        self.rule_fn("json-null", |p| p.literal("null"))
    }

    pub fn json_object(&mut self) -> ParserId {
        self.rule_fn("json-object", |p| {
            let ws = p.space();
            let json = p.json();
            let member = {
                let key = p.json_string();
                let colon = p.literal(":");
                p.sequence(&[key, ws, colon, ws, json])
            };
            let members = {
                let tail = {
                    let comma = p.literal(",");
                    let m = member;
                    p.sequence(&[ws, comma, ws, m])
                };
                let tail = p.zero_or_more(tail);
                p.sequence(&[member, tail])
            };
            let close = {
                let rb1 = p.literal("}");
                let rb2 = p.literal("}");
                let seq = p.sequence(&[members, ws, rb2]);
                p.choice(&[rb1, seq])
            };
            let lb = p.literal("{");
            p.sequence(&[lb, ws, close])
        })
    }

    pub fn json_array(&mut self) -> ParserId {
        self.rule_fn("json-array", |p| {
            let ws = p.space();
            let json = p.json();
            let elements = {
                let tail = {
                    let comma = p.literal(",");
                    p.sequence(&[ws, comma, ws, json])
                };
                let tail = p.zero_or_more(tail);
                p.sequence(&[json, tail])
            };
            let close = {
                let rb1 = p.literal("]");
                let rb2 = p.literal("]");
                let seq = p.sequence(&[elements, ws, rb2]);
                p.choice(&[rb1, seq])
            };
            let lb = p.literal("[");
            p.sequence(&[lb, ws, close])
        })
    }

    pub fn json(&mut self) -> ParserId {
        self.rule_fn("json-value", |p| {
            let obj = p.json_object();
            let arr = p.json_array();
            let s = p.json_string();
            let n = p.json_number();
            let b = p.json_bool();
            let null = p.json_null();
            p.choice(&[obj, arr, s, n, b, null])
        })
    }

    // ---- Python value parsers (peg-parser.cpp:1297-1363) --------------------

    pub fn python_string(&mut self) -> ParserId {
        self.rule_fn("python-string", |p| {
            let d = p.double_quoted_string();
            let s = p.single_quoted_string();
            p.choice(&[d, s])
        })
    }

    pub fn python_number(&mut self) -> ParserId {
        self.json_number()
    }

    pub fn python_bool(&mut self) -> ParserId {
        self.rule_fn("python-bool", |p| {
            let t = p.literal("True");
            let f = p.literal("False");
            p.choice(&[t, f])
        })
    }

    pub fn python_null(&mut self) -> ParserId {
        self.rule_fn("python-none", |p| p.literal("None"))
    }

    pub fn python_dict(&mut self) -> ParserId {
        self.rule_fn("python-dict", |p| {
            let ws = p.space();
            let value = p.python_value();
            let member = {
                let key = p.python_string();
                let colon = p.literal(":");
                p.sequence(&[key, ws, colon, ws, value])
            };
            let members = {
                let tail = {
                    let comma = p.literal(",");
                    let m = member;
                    p.sequence(&[ws, comma, ws, m])
                };
                let tail = p.zero_or_more(tail);
                p.sequence(&[member, tail])
            };
            let close = {
                let rb1 = p.literal("}");
                let rb2 = p.literal("}");
                let seq = p.sequence(&[members, ws, rb2]);
                p.choice(&[rb1, seq])
            };
            let lb = p.literal("{");
            p.sequence(&[lb, ws, close, ws])
        })
    }

    pub fn python_array(&mut self) -> ParserId {
        self.rule_fn("python-array", |p| {
            let ws = p.space();
            let value = p.python_value();
            let elements = {
                let tail = {
                    let comma = p.literal(",");
                    p.sequence(&[comma, ws, value])
                };
                let tail = p.zero_or_more(tail);
                p.sequence(&[value, tail])
            };
            let close = {
                let rb1 = p.literal("]");
                let rb2 = p.literal("]");
                let seq = p.sequence(&[elements, ws, rb2]);
                p.choice(&[rb1, seq])
            };
            let lb = p.literal("[");
            p.sequence(&[lb, ws, close, ws])
        })
    }

    pub fn python_value(&mut self) -> ParserId {
        self.rule_fn("python-value", |p| {
            let dict = p.python_dict();
            let arr = p.python_array();
            let s = p.python_string();
            let n = p.python_number();
            let b = p.python_bool();
            let null = p.python_null();
            p.choice(&[dict, arr, s, n, b, null])
        })
    }

    // A marker: text delimited by a pair of <> or []  (peg-parser.cpp:1365-1369)
    pub fn marker(&mut self) -> ParserId {
        let sharp_bracket_parser = {
            let lt = p_literal(self, "<");
            let until = self.until(">");
            let gt = self.literal(">");
            self.sequence(&[lt, until, gt])
        };
        let square_bracket_parser = {
            let lb = self.literal("[");
            let until = self.until("]");
            let rb = self.literal("]");
            self.sequence(&[lb, until, rb])
        };
        self.choice(&[sharp_bracket_parser, square_bracket_parser])
    }

    // A JSON object member with a fixed key  (peg-parser.cpp:1371-1380)
    pub fn json_member(&mut self, key: &str, p: ParserId) -> ParserId {
        let ws = self.space();
        let k = self.literal(&format!("\"{key}\""));
        let colon = self.literal(":");
        self.sequence(&[k, ws, colon, ws, p])
    }

    // ---- schema / rules / wrappers -------------------------------------------

    /// `schema(p, name, doc, node, raw)` (peg-parser.cpp:1148-1150)
    pub fn schema_node(
        &mut self,
        p: ParserId,
        name: &str,
        doc: Rc<SchemaDocument>,
        node: crate::json_schema::NodeId,
        raw: bool,
    ) -> ParserId {
        self.add(ParserKind::Schema {
            child: p,
            name: name.to_string(),
            doc: Some(doc),
            node,
            raw,
        })
    }

    /// `schema(p, name, schema_json, raw)` (peg-parser.cpp:1152-1155)
    pub fn schema(&mut self, p: ParserId, name: &str, schema: &Json, raw: bool) -> ParserId {
        let doc = Rc::new(
            schema_from_json(schema)
                .map_err(|e| e.to_string())
                .unwrap_or_else(|_| {
                    // the C++ constructor propagates the throw; keep the builder total
                    // by installing an empty document and letting add_schema fail later
                    SchemaDocument {
                        nodes: Vec::new(),
                        root: 0,
                        refs: BTreeMap::new(),
                    }
                }),
        );
        let root = doc.root;
        self.schema_node(p, name, doc, root, raw)
    }

    /// `rule(name, p, trigger)` (peg-parser.cpp:1157-1162)
    pub fn rule(&mut self, name: &str, p: ParserId, trigger: bool) -> ParserId {
        let clean_name = rule_name(name);
        let rule_id = self.add(ParserKind::Rule {
            name: clean_name.clone(),
            child: p,
            trigger,
        });
        self.arena.add_rule(&clean_name, rule_id);
        self.ref_(&clean_name)
    }

    /// `rule(name, builder_fn)` — non-trigger form (C++ default argument
    /// `trigger=false`, peg-parser.h:524)
    pub fn rule_fn<F>(&mut self, name: &str, f: F) -> ParserId
    where
        F: FnOnce(&mut Self) -> ParserId,
    {
        self.rule_fn_trigger(name, f, false)
    }

    /// `rule(name, builder_fn, trigger)` (peg-parser.cpp:1164-1183): allows
    /// recursive rules via a placeholder.
    pub fn rule_fn_trigger<F>(&mut self, name: &str, f: F, trigger: bool) -> ParserId
    where
        F: FnOnce(&mut Self) -> ParserId,
    {
        let clean_name = rule_name(name);
        if self.arena.has_rule(&clean_name) {
            return self.ref_(&clean_name);
        }
        // Create placeholder rule to allow recursive references
        let placeholder = self.any(); // Temporary placeholder
        let placeholder_rule_id = self.add(ParserKind::Rule {
            name: clean_name.clone(),
            child: placeholder,
            trigger,
        });
        self.arena.add_rule(&clean_name, placeholder_rule_id);
        // Build the actual parser
        let parser = f(self);
        // Replace placeholder with actual rule
        let rule_id = self.add(ParserKind::Rule {
            name: clean_name.clone(),
            child: parser,
            trigger,
        });
        self.arena.add_rule(&clean_name, rule_id);
        self.ref_(&clean_name)
    }

    /// `trigger_rule(name, p)` (peg-parser.h:528)
    pub fn trigger_rule(&mut self, name: &str, p: ParserId) -> ParserId {
        self.rule(name, p, true)
    }

    /// `trigger_rule(name, builder)` (peg-parser.h:529)
    pub fn trigger_rule_fn<F>(&mut self, name: &str, f: F) -> ParserId
    where
        F: FnOnce(&mut Self) -> ParserId,
    {
        self.rule_fn_trigger(name, f, true)
    }

    /// `atomic(p)` (peg-parser.h:534): no AST node if the child is partial.
    pub fn atomic(&mut self, p: ParserId) -> ParserId {
        self.add(ParserKind::Atomic { child: p })
    }

    /// `tag(tag, p)` (peg-parser.h:538)
    pub fn tag(&mut self, tag: &str, p: ParserId) -> ParserId {
        self.add(ParserKind::Tag {
            child: p,
            tag: tag.to_string(),
        })
    }

    /// `gbnf(p, grammar)` (peg-parser.h:542)
    pub fn gbnf(&mut self, p: ParserId, grammar: &str) -> ParserId {
        self.add(ParserKind::Gbnf {
            child: p,
            grammar: grammar.to_string(),
        })
    }

    /// `ac(p, delimiters)` (peg-parser.cpp:1382-1387)
    pub fn ac(&mut self, p: ParserId, delimiters: &[String]) -> ParserId {
        if delimiters.is_empty() {
            self.error = Some("ac parser requires at least one delimiter".to_string());
            return self.eps();
        }
        self.add(ParserKind::Ac {
            child: p,
            delimiters: delimiters.to_vec(),
        })
    }

    pub fn set_root(&mut self, p: ParserId) {
        self.arena.set_root(p);
    }

    /// `build()` (peg-parser.cpp:1189-1192)
    pub fn build(&mut self) -> Result<PegArena, String> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        self.arena.resolve_refs()?;
        Ok(std::mem::take(&mut self.arena))
    }
}

fn p_literal(b: &mut PegBuilder, s: &str) -> ParserId {
    b.literal(s)
}

// ---------------------------------------------------------------------------
// build_peg_parser (peg-parser.cpp:2120-2124)
// ---------------------------------------------------------------------------

pub fn build_peg_parser<F>(f: F) -> Result<PegArena, String>
where
    F: FnOnce(&mut PegBuilder) -> ParserId,
{
    let mut builder = PegBuilder::default();
    let root = f(&mut builder);
    builder.set_root(root);
    builder.build()
}

// ---------------------------------------------------------------------------
// GBNF generation helpers (peg-parser.cpp:1389-1603)
// ---------------------------------------------------------------------------

/// `gbnf_escape_char_class` (peg-parser.cpp:1389-1437)
fn gbnf_escape_char_class(c: u32) -> String {
    let ch = char::from_u32(c).unwrap_or('\u{FFFD}');
    if ch == '-' || ch == ']' || ch == '[' || ch == '\\' {
        return format!("\\{ch}");
    }
    if ch == '\n' {
        return "\\n".to_string();
    }
    if ch == '\t' {
        return "\\t".to_string();
    }
    if ch == '\r' {
        return "\\r".to_string();
    }
    // Printable ASCII
    if (0x20..=0x7E).contains(&c) {
        return ch.to_string();
    }
    // Hex escape
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    if c <= 0xFF {
        format!(
            "\\x{}{}",
            HEX[((c >> 4) & 0xF) as usize] as char,
            HEX[(c & 0xF) as usize] as char
        )
    } else if c <= 0xFFFF {
        format!(
            "\\u{}{}{}{}",
            HEX[((c >> 12) & 0xF) as usize] as char,
            HEX[((c >> 8) & 0xF) as usize] as char,
            HEX[((c >> 4) & 0xF) as usize] as char,
            HEX[(c & 0xF) as usize] as char
        )
    } else {
        let mut s = String::from("\\U");
        for i in 0..8 {
            s.push(HEX[((c >> ((7 - i) * 4)) & 0xF) as usize] as char);
        }
        s
    }
}

/// `gbnf_char_class` (peg-parser.cpp:1439-1445)
fn gbnf_char_class(chars: &[u32], negate: bool) -> String {
    let mut s = if negate {
        "[^".to_string()
    } else {
        "[".to_string()
    };
    for &ch in chars {
        s += &gbnf_escape_char_class(ch);
    }
    s + "]"
}

/// `gbnf_ac_grammar` (peg-parser.cpp:1447-1495): emit an Aho-Corasick automaton
/// DFA; `build_rule` shapes each state's right-hand side.
fn gbnf_ac_grammar<F>(
    builder: &mut crate::json_schema::GrammarBuilder,
    prefix: &str,
    strings: &[String],
    build_rule: F,
) -> String
where
    F: Fn(&[u32], &BTreeMap<usize, Vec<u32>>, &[u32], &dyn Fn(usize) -> String) -> String,
{
    let ac = AhoCorasick::from_strings(strings);

    let state_name = |s: usize| -> String {
        if s == 0 {
            return prefix.to_string();
        }
        let num = s.to_string();
        if num.len() == 1 {
            format!("{prefix}-0{num}")
        } else {
            format!("{prefix}-{num}")
        }
    };

    for q in 0..ac.num_states() {
        if ac.is_terminal(q) {
            continue; // match states
        }
        let mut buckets: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut completing: Vec<u32> = Vec::new(); // chars that complete a delimiter
        let mut specific: Vec<u32> = Vec::new(); // chars with an explicit transition
        for &c in ac.alphabet.iter() {
            let d = ac.next(q, c);
            if ac.is_terminal(d) {
                completing.push(c);
                specific.push(c);
            } else if d != 0 {
                buckets.entry(d).or_default().push(c);
                specific.push(c);
            }
        }
        builder.add_rule(
            &state_name(q),
            &build_rule(&completing, &buckets, &specific, &state_name),
        );
    }

    // An empty delimiter makes the start state terminal (peg-parser.cpp:1489-1493)
    if ac.is_terminal(0) {
        builder.add_rule(prefix, "|");
    }

    state_name(0)
}

/// `gbnf_excluding_grammar` (peg-parser.cpp:1502-1519): matches strings that
/// contain no string in `strings` as a substring (complement automaton).
fn gbnf_excluding_grammar(
    builder: &mut crate::json_schema::GrammarBuilder,
    prefix: &str,
    strings: &[String],
) -> String {
    gbnf_ac_grammar(
        builder,
        prefix,
        strings,
        |_completing, buckets, specific, state_name| {
            // every state is accepting and completing chars get no
            // alternative, so a forbidden string can never be matched
            let mut rhs = "|".to_string();
            for (d, chars) in buckets {
                rhs += &format!(" {} {} |", gbnf_char_class(chars, false), state_name(*d));
            }
            rhs += &format!(" {} {}", gbnf_char_class(specific, true), state_name(0));
            rhs
        },
    )
}

/// `gbnf_including_grammar` (peg-parser.cpp:1524-1543): everything up to and
/// including the first occurrence of any string in `strings`.
fn gbnf_including_grammar(
    builder: &mut crate::json_schema::GrammarBuilder,
    prefix: &str,
    strings: &[String],
) -> String {
    gbnf_ac_grammar(
        builder,
        prefix,
        strings,
        |completing, buckets, specific, state_name| {
            let mut alts: Vec<String> = Vec::new();
            if !completing.is_empty() {
                alts.push(gbnf_char_class(completing, false)); // terminate on match
            }
            for (d, chars) in buckets {
                alts.push(format!(
                    "{} {}",
                    gbnf_char_class(chars, false),
                    state_name(*d)
                ));
            }
            // every other character keeps scanning from the start state
            alts.push(format!(
                "{} {}",
                gbnf_char_class(specific, true),
                state_name(0)
            ));
            alts.join(" | ")
        },
    )
}

/// `collect_reachable_rules` (peg-parser.cpp:1545-1603)
fn collect_reachable_rules(arena: &PegArena, rule: ParserId) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    let mut visited: BTreeSet<String> = BTreeSet::new();

    fn visit(
        arena: &PegArena,
        id: ParserId,
        reachable: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
    ) {
        match arena.get(id) {
            ParserKind::Epsilon
            | ParserKind::Start
            | ParserKind::End
            | ParserKind::Until { .. }
            | ParserKind::Literal(_)
            | ParserKind::Chars { .. }
            | ParserKind::Space
            | ParserKind::Any
            | ParserKind::Str { .. } => {}
            ParserKind::Sequence(children) | ParserKind::Choice(children) => {
                for &child in children {
                    visit(arena, child, reachable, visited);
                }
            }
            ParserKind::Repetition { child, .. }
            | ParserKind::And { child }
            | ParserKind::Not { child }
            | ParserKind::Tag { child, .. }
            | ParserKind::Atomic { child }
            | ParserKind::Gbnf { child, .. }
            | ParserKind::Ac { child, .. }
            | ParserKind::Schema { child, .. } => visit(arena, *child, reachable, visited),
            ParserKind::Rule { name, child, .. } => {
                if !visited.contains(name) {
                    visited.insert(name.clone());
                    reachable.insert(name.clone());
                    visit(arena, *child, reachable, visited);
                }
            }
            ParserKind::Ref { name } => {
                // Traverse rules so we pick up everything
                match arena.get_rule(name) {
                    Ok(referenced_rule) => visit(arena, referenced_rule, reachable, visited),
                    Err(_) => {}
                }
            }
        }
    }

    visit(arena, rule, &mut reachable, &mut visited);
    reachable
}

// ---------------------------------------------------------------------------
// schema merge — bridges the peg arena's per-parser schema documents into the
// json-schema converter's single-document interface
// ---------------------------------------------------------------------------

struct SchemaMerger {
    /// (doc rc ptr, node id in that doc) → merged node id
    map: HashMap<(usize, usize), usize>,
    document: SchemaDocument,
}

impl SchemaMerger {
    fn new(arena: &PegArena) -> Self {
        let mut merger = SchemaMerger {
            map: HashMap::new(),
            document: SchemaDocument {
                nodes: Vec::new(),
                root: 0,
                refs: BTreeMap::new(),
            },
        };
        // seed with a dummy root node (the converter always wants doc.root valid)
        merger.document.nodes.push(SchemaNode {
            kind: SchemaKind::Any,
            children: Vec::new(),
        });
        merger.document.root = 0;
        // append every schema document reachable from the arena, remapping ids
        for parser in &arena.parsers {
            if let ParserKind::Schema { doc, node, .. } = parser {
                if let Some(doc) = doc {
                    merger.merge(Rc::clone(doc), *node);
                }
            }
        }
        merger
    }

    fn merge(&mut self, doc: Rc<SchemaDocument>, node: usize) -> usize {
        let key = (Rc::as_ptr(&doc) as usize, node);
        if let Some(&merged) = self.map.get(&key) {
            return merged;
        }
        // copy the whole reachable subgraph depth-first
        let base = self.document.nodes.len();
        let _ = base;
        let merged = self.merge_node(&doc, node);
        self.map.insert(key, merged);
        merged
    }

    fn merge_node(&mut self, doc: &Rc<SchemaDocument>, node: usize) -> usize {
        let key = (Rc::as_ptr(doc) as usize, node);
        if let Some(&merged) = self.map.get(&key) {
            return merged;
        }
        // reserve the slot first so $ref cycles stay expressible
        let new_id = self.document.nodes.len();
        self.document.nodes.push(SchemaNode {
            kind: SchemaKind::Any,
            children: Vec::new(),
        });
        // register before recursing so cycles resolve to this slot
        self.map.insert(key, new_id);
        let source = doc.node(node);
        let mut children = Vec::new();
        for &child in &source.children {
            children.push(self.merge_node(doc, child));
        }
        let mut kind = source.kind.clone();
        match &mut kind {
            SchemaKind::Ref { target, .. } => {
                if let Some(t) = *target {
                    *target = Some(self.merge_node(doc, t));
                }
            }
            // property / item node ids live outside `children` — remap them too
            SchemaKind::Object {
                properties,
                additional_properties,
            } => {
                for prop in properties.iter_mut() {
                    prop.schema = self.merge_node(doc, prop.schema);
                }
                if let Some(a) = *additional_properties {
                    *additional_properties = Some(self.merge_node(doc, a));
                }
            }
            SchemaKind::Array { items, .. } => {
                *items = self.merge_node(doc, *items);
            }
            _ => {}
        }
        self.document.nodes[new_id] = SchemaNode { kind, children };
        // copy $ref names for the document-level refs map
        if let SchemaKind::Ref { ref_str, target } = &self.document.nodes[new_id].kind {
            if let Some(merged_target) = *target {
                self.document.refs.insert(ref_str.clone(), merged_target);
            }
        }
        new_id
    }
}

// ---------------------------------------------------------------------------
// GBNF generation driver (peg-parser.cpp:1606-1815)
// ---------------------------------------------------------------------------

struct GbnfGen<'a, 'b, 'c> {
    arena: &'a PegArena,
    builder: &'a mut crate::json_schema::GrammarBuilder<'b, 'c>,
    merger: &'a mut SchemaMerger,
}

impl GbnfGen<'_, '_, '_> {
    // A raw string value is parsed by the child rather than constrained by the
    // schema (peg-parser.cpp:1607-1610)
    fn schema_delegates(&self, s: &ParserKind) -> bool {
        match s {
            ParserKind::Schema {
                doc: Some(doc),
                node,
                raw,
                ..
            } => *raw && doc.may_be_string(*node),
            // !s.node (peg-parser.cpp:1608-1610)
            _ => true,
        }
    }

    // Unwrap the parser so we can check if it's a sequence or choice
    // (peg-parser.cpp:1612-1630)
    fn effective_parser_is(&self, start: ParserId, pred: fn(&ParserKind) -> bool) -> bool {
        let mut id = start;
        loop {
            match self.arena.get(id) {
                ParserKind::Tag { child, .. } => id = *child,
                ParserKind::Atomic { child } => id = *child,
                ParserKind::Schema { child, .. } if self.schema_delegates(self.arena.get(id)) => {
                    id = *child
                }
                p => return pred(p),
            }
        }
    }

    fn is_choice_or_seq(&self, id: ParserId) -> bool {
        self.effective_parser_is(id, |p| {
            matches!(p, ParserKind::Choice(_) | ParserKind::Sequence(_))
        })
    }

    fn is_choice(&self, id: ParserId) -> bool {
        self.effective_parser_is(id, |p| matches!(p, ParserKind::Choice(_)))
    }

    /// `to_gbnf` (peg-parser.cpp:1632-1762)
    fn to_gbnf(&mut self, id: ParserId) -> String {
        match self.arena.get(id) {
            ParserKind::Epsilon | ParserKind::Start | ParserKind::End => "".to_string(),
            ParserKind::Literal(literal) => gbnf_format_literal(literal),
            ParserKind::Sequence(children) => {
                let mut s = String::new();
                for &child in children {
                    let child_gbnf = self.to_gbnf(child);
                    if child_gbnf.is_empty() {
                        continue;
                    }
                    if !s.is_empty() {
                        s += " ";
                    }
                    if self.is_choice_or_seq(child) {
                        s += &format!("({child_gbnf})");
                    } else {
                        s += &child_gbnf;
                    }
                }
                s
            }
            ParserKind::Choice(children) => {
                let mut s = String::new();
                for &child in children {
                    if !s.is_empty() {
                        s += " | ";
                    }
                    let child_gbnf = self.to_gbnf(child);
                    if self.is_choice(child) {
                        s += &format!("({child_gbnf})");
                    } else {
                        s += &child_gbnf;
                    }
                }
                s
            }
            ParserKind::Repetition { child, min, max } => {
                let mut child_gbnf = self.to_gbnf(*child);
                if self.is_choice_or_seq(*child) {
                    child_gbnf = format!("({child_gbnf})");
                }
                if *min == 0 && *max == 1 {
                    return child_gbnf + "?";
                }
                if *min == 0 && *max == -1 {
                    return child_gbnf + "*";
                }
                if *min == 1 && *max == -1 {
                    return child_gbnf + "+";
                }
                if *max == -1 {
                    return format!("{child_gbnf}{{{min},}}");
                }
                if min == max {
                    if *min == 1 {
                        return child_gbnf;
                    }
                    return format!("{child_gbnf}{{{min}}}");
                }
                format!("{child_gbnf}{{{min},{max}}}")
            }
            ParserKind::And { .. } | ParserKind::Not { .. } => {
                "".to_string() // Lookahead not supported in GBNF
            }
            ParserKind::Any => ".".to_string(),
            ParserKind::Space => "space".to_string(),
            ParserKind::Chars {
                pattern, min, max, ..
            } => {
                let result = pattern.clone();
                if *min == 0 && *max == 1 {
                    return result + "?";
                }
                if *min == 0 && *max == -1 {
                    return result + "*";
                }
                if *min == 1 && *max == -1 {
                    return result + "+";
                }
                if *max == -1 {
                    return format!("{result}{{{min},}}");
                }
                if min == max {
                    if *min == 1 {
                        return result;
                    }
                    return format!("{result}{{{min}}}");
                }
                format!("{result}{{{min},{max}}}")
            }
            ParserKind::Str { delimiter } => {
                let delim = (*delimiter as char).to_string();
                format!(r#"( [^{delim}\\] | "\\" ( [{delim}\\/ bfnrt] | "u" [0-9a-fA-F]{{4}} ) )*"#)
            }
            ParserKind::Until { delimiters } => {
                if delimiters.is_empty() {
                    return ".*".to_string();
                }
                let prefix = format!("until-{id}");
                gbnf_excluding_grammar(self.builder, &prefix, delimiters)
            }
            ParserKind::Schema { .. } => {
                if self.schema_delegates(self.arena.get(id)) {
                    let child = match self.arena.get(id) {
                        ParserKind::Schema { child, .. } => *child,
                        _ => unreachable!(),
                    };
                    return self.to_gbnf(child);
                }
                let ParserKind::Schema {
                    name, doc, node, ..
                } = self.arena.get(id)
                else {
                    unreachable!()
                };
                match doc {
                    Some(doc) => {
                        let merged = self.mmerger(Rc::clone(doc), *node);
                        self.builder.add_schema(name, merged)
                    }
                    // deserialized parser without its schema document: the
                    // reference never reaches here (grammars are built from
                    // the live arena); fall back to the child's grammar
                    None => {
                        let child = match self.arena.get(id) {
                            ParserKind::Schema { child, .. } => *child,
                            _ => unreachable!(),
                        };
                        self.to_gbnf(child)
                    }
                }
            }
            ParserKind::Rule { name, .. } => name.clone(),
            ParserKind::Ref { name } => {
                // Refs should not exist after flattening, but kept just in case
                name.clone()
            }
            ParserKind::Tag { child, .. } => self.to_gbnf(*child),
            ParserKind::Atomic { child } => self.to_gbnf(*child),
            ParserKind::Gbnf { grammar, .. } => grammar.clone(),
            ParserKind::Ac { delimiters, .. } => {
                let prefix = format!("ac-{id}");
                gbnf_including_grammar(self.builder, &prefix, delimiters)
            }
        }
    }

    fn mmerger(&self, doc: Rc<SchemaDocument>, node: usize) -> usize {
        *self
            .merger
            .map
            .get(&(Rc::as_ptr(&doc) as usize, node))
            .unwrap_or(&node)
    }

    /// the body of `common_peg_arena::build_grammar` (peg-parser.cpp:1764-1815)
    fn run(&mut self, lazy: bool) {
        // Collect reachable rules
        let mut reachable_rules: BTreeSet<String> = BTreeSet::new();
        if lazy {
            // Collect rules reachable from trigger rules
            for (_, rule_id) in self.arena.rules.clone() {
                if let ParserKind::Rule {
                    name,
                    trigger: true,
                    ..
                } = self.arena.get(rule_id)
                {
                    reachable_rules.insert(name.clone());
                    let add_rules = collect_reachable_rules(self.arena, rule_id);
                    reachable_rules.extend(add_rules);
                }
            }
        } else {
            // Collect rules reachable from root
            reachable_rules = collect_reachable_rules(self.arena, self.arena.root);
        }

        // Create GBNF rules for all reachable rules
        for (name, rule_id) in self.arena.rules.clone() {
            if !reachable_rules.contains(&name) {
                continue;
            }
            if let ParserKind::Rule { name, child, .. } = self.arena.get(rule_id) {
                let rhs = self.to_gbnf(*child);
                self.builder.add_rule(name, &rhs);
            }
        }

        if lazy {
            // Generate root rule from trigger rules only
            let mut trigger_names: Vec<String> = Vec::new();
            for (_, rule_id) in self.arena.rules.clone() {
                if let ParserKind::Rule {
                    name,
                    trigger: true,
                    ..
                } = self.arena.get(rule_id)
                {
                    trigger_names.push(name.clone());
                }
            }
            // Sort for predictable order
            trigger_names.sort();
            self.builder.add_rule("root", &trigger_names.join(" | "));
        } else if self.arena.root != INVALID_PARSER_ID {
            let rhs = self.to_gbnf(self.arena.root);
            self.builder.add_rule("root", &rhs);
        }
    }
}

// ---------------------------------------------------------------------------
// serialization variants (peg-parser.cpp:1817-2075)
// ---------------------------------------------------------------------------

fn serialize_parser_variant(variant: &ParserKind) -> Json {
    fn obj(fields: Vec<(&str, Json)>) -> Json {
        Json::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }
    fn ids(v: &[ParserId]) -> Json {
        Json::Array(v.iter().map(|&i| Json::Uint(i as u64)).collect())
    }
    match variant {
        ParserKind::Epsilon => obj(vec![("type", Json::String("epsilon".into()))]),
        ParserKind::Start => obj(vec![("type", Json::String("start".into()))]),
        ParserKind::End => obj(vec![("type", Json::String("end".into()))]),
        ParserKind::Literal(literal) => obj(vec![
            ("type", Json::String("literal".into())),
            ("literal", Json::String(literal.clone())),
        ]),
        ParserKind::Sequence(children) => obj(vec![
            ("type", Json::String("sequence".into())),
            ("children", ids(children)),
        ]),
        ParserKind::Choice(children) => obj(vec![
            ("type", Json::String("choice".into())),
            ("children", ids(children)),
        ]),
        ParserKind::Repetition { child, min, max } => obj(vec![
            ("type", Json::String("repetition".into())),
            ("child", Json::Uint(*child as u64)),
            ("min_count", Json::Int(*min as i64)),
            ("max_count", Json::Int(*max as i64)),
        ]),
        ParserKind::And { child } => obj(vec![
            ("type", Json::String("and".into())),
            ("child", Json::Uint(*child as u64)),
        ]),
        ParserKind::Not { child } => obj(vec![
            ("type", Json::String("not".into())),
            ("child", Json::Uint(*child as u64)),
        ]),
        ParserKind::Any => obj(vec![("type", Json::String("any".into()))]),
        ParserKind::Space => obj(vec![("type", Json::String("space".into()))]),
        ParserKind::Chars {
            pattern,
            ranges,
            negated,
            min,
            max,
        } => {
            let ranges_json = Json::Array(
                ranges
                    .iter()
                    .map(|r| {
                        obj(vec![
                            ("start", Json::Uint(r.start as u64)),
                            ("end", Json::Uint(r.end as u64)),
                        ])
                    })
                    .collect(),
            );
            obj(vec![
                ("type", Json::String("chars".into())),
                ("pattern", Json::String(pattern.clone())),
                ("ranges", ranges_json),
                ("negated", Json::Bool(*negated)),
                ("min_count", Json::Int(*min as i64)),
                ("max_count", Json::Int(*max as i64)),
            ])
        }
        ParserKind::Str { delimiter } => obj(vec![
            ("type", Json::String("string".into())),
            ("delimiter", Json::String((*delimiter as char).to_string())),
        ]),
        ParserKind::Until { delimiters } => obj(vec![
            ("type", Json::String("until".into())),
            (
                "delimiters",
                Json::Array(delimiters.iter().map(|d| Json::String(d.clone())).collect()),
            ),
        ]),
        ParserKind::Schema {
            child, name, raw, ..
        } => obj(vec![
            ("type", Json::String("schema".into())),
            ("child", Json::Uint(*child as u64)),
            ("name", Json::String(name.clone())),
            ("raw", Json::Bool(*raw)),
        ]),
        ParserKind::Rule {
            name,
            child,
            trigger,
        } => obj(vec![
            ("type", Json::String("rule".into())),
            ("name", Json::String(name.clone())),
            ("child", Json::Uint(*child as u64)),
            ("trigger", Json::Bool(*trigger)),
        ]),
        ParserKind::Ref { name } => obj(vec![
            ("type", Json::String("ref".into())),
            ("name", Json::String(name.clone())),
        ]),
        ParserKind::Atomic { child } => obj(vec![
            ("type", Json::String("atomic".into())),
            ("child", Json::Uint(*child as u64)),
        ]),
        ParserKind::Tag { child, tag } => obj(vec![
            ("type", Json::String("tag".into())),
            ("child", Json::Uint(*child as u64)),
            ("tag", Json::String(tag.clone())),
        ]),
        ParserKind::Gbnf { child, grammar } => obj(vec![
            ("type", Json::String("gbnf".into())),
            ("child", Json::Uint(*child as u64)),
            ("grammar", Json::String(grammar.clone())),
        ]),
        ParserKind::Ac { child, delimiters } => obj(vec![
            ("type", Json::String("ac".into())),
            ("child", Json::Uint(*child as u64)),
            (
                "delimiters",
                Json::Array(delimiters.iter().map(|d| Json::String(d.clone())).collect()),
            ),
        ]),
    }
}

fn deserialize_parser_variant(j: &Json) -> Result<ParserKind, String> {
    let ty = j
        .at("type")
        .and_then(|v| if v.is_string() { Some(v) } else { None })
        .and_then(|v| v.get_str().ok())
        .ok_or("Parser variant JSON missing or invalid 'type' field")?
        .to_string();

    let uint = |key: &str| -> Result<usize, String> {
        j.at(key)
            .ok_or_else(|| format!("parser missing '{key}' field"))?
            .get_i64()
            .map(|v| v as usize)
            .map_err(|e| e.to_string())
    };
    let int = |key: &str| -> Result<i32, String> {
        j.at(key)
            .ok_or_else(|| format!("parser missing '{key}' field"))?
            .get_i64()
            .map(|v| v as i32)
            .map_err(|e| e.to_string())
    };
    let string = |key: &str| -> Result<String, String> {
        j.at(key)
            .and_then(|v| v.get_str().ok().map(|s| s.to_string()))
            .ok_or_else(|| format!("parser missing or invalid '{key}' field"))
    };
    match ty.as_str() {
        "epsilon" => Ok(ParserKind::Epsilon),
        "start" => Ok(ParserKind::Start),
        "end" => Ok(ParserKind::End),
        "literal" => Ok(ParserKind::Literal(string("literal")?)),
        "sequence" => {
            let children = j
                .at("children")
                .filter(|v| v.is_array())
                .ok_or("sequence parser missing or invalid 'children' field")?;
            let mut ids = Vec::new();
            for c in children.iter() {
                ids.push(c.get_i64().map_err(|e| e.to_string())? as usize);
            }
            Ok(ParserKind::Sequence(ids))
        }
        "choice" => {
            let children = j
                .at("children")
                .filter(|v| v.is_array())
                .ok_or("choice parser missing or invalid 'children' field")?;
            let mut ids = Vec::new();
            for c in children.iter() {
                ids.push(c.get_i64().map_err(|e| e.to_string())? as usize);
            }
            Ok(ParserKind::Choice(ids))
        }
        "repetition" => Ok(ParserKind::Repetition {
            child: uint("child")?,
            min: int("min_count")?,
            max: int("max_count")?,
        }),
        "and" => Ok(ParserKind::And {
            child: uint("child")?,
        }),
        "not" => Ok(ParserKind::Not {
            child: uint("child")?,
        }),
        "any" => Ok(ParserKind::Any),
        "space" => Ok(ParserKind::Space),
        "chars" => {
            let pattern = string("pattern")?;
            let negated = match j.at("negated") {
                Some(Json::Bool(b)) => *b,
                _ => return Err("chars parser missing required fields".to_string()),
            };
            let min = int("min_count")?;
            let max = int("max_count")?;
            let mut ranges = Vec::new();
            let ranges_json = j
                .at("ranges")
                .filter(|v| v.is_array())
                .ok_or("chars parser missing required fields")?;
            for r in ranges_json.iter() {
                let start = r
                    .at("start")
                    .and_then(|v| v.get_i64().ok())
                    .ok_or("char_range missing 'start' or 'end' field")?
                    as u32;
                let end = r
                    .at("end")
                    .and_then(|v| v.get_i64().ok())
                    .ok_or("char_range missing 'start' or 'end' field")?
                    as u32;
                ranges.push(CharRange { start, end });
            }
            Ok(ParserKind::Chars {
                pattern,
                ranges,
                negated,
                min,
                max,
            })
        }
        "string" => {
            let delimiter = string("delimiter")?;
            if delimiter.is_empty() {
                return Err("string parser delimiter is empty.".to_string());
            }
            Ok(ParserKind::Str {
                delimiter: delimiter.as_bytes()[0],
            })
        }
        "until" => {
            let delimiters = j
                .at("delimiters")
                .filter(|v| v.is_array())
                .ok_or("until parser missing or invalid 'delimiters' field")?;
            let mut out = Vec::new();
            for d in delimiters.iter() {
                out.push(d.get_str().map_err(|e| e.to_string())?.to_string());
            }
            Ok(ParserKind::Until { delimiters: out })
        }
        "schema" => Ok(ParserKind::Schema {
            child: uint("child")?,
            name: string("name")?,
            raw: match j.at("raw") {
                Some(Json::Bool(b)) => *b,
                _ => return Err("schema parser missing required fields".to_string()),
            },
            // schema documents are not serialized in the reference either:
            // deserialize leaves `node == nullptr` (peg-parser.cpp:2010-2018)
            doc: None,
            node: 0,
        }),
        "rule" => Ok(ParserKind::Rule {
            name: string("name")?,
            child: uint("child")?,
            trigger: match j.at("trigger") {
                Some(Json::Bool(b)) => *b,
                _ => return Err("rule parser missing required fields".to_string()),
            },
        }),
        "ref" => Ok(ParserKind::Ref {
            name: string("name")?,
        }),
        "atomic" => Ok(ParserKind::Atomic {
            child: uint("child")?,
        }),
        "tag" => Ok(ParserKind::Tag {
            child: uint("child")?,
            tag: string("tag")?,
        }),
        "gbnf" => Ok(ParserKind::Gbnf {
            child: uint("child")?,
            grammar: string("grammar")?,
        }),
        "ac" => {
            let child = uint("child")?;
            let delimiters = j
                .at("delimiters")
                .filter(|v| v.is_array())
                .ok_or("ac parser requires 'child' and a non-empty 'delimiters' array")?;
            let mut out = Vec::new();
            for d in delimiters.iter() {
                out.push(d.get_str().map_err(|e| e.to_string())?.to_string());
            }
            if out.is_empty() {
                return Err(
                    "ac parser requires 'child' and a non-empty 'delimiters' array".to_string(),
                );
            }
            Ok(ParserKind::Ac {
                child,
                delimiters: out,
            })
        }
        other => Err(format!("Unknown parser type: {other}")),
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(arena: &PegArena, input: &str) -> ParseResult {
        let mut ctx = ParseContext::new(input, PARSE_FLAG_LENIENT);
        arena.parse(&mut ctx, 0).unwrap()
    }

    #[test]
    fn literal_success_and_need_more() {
        let arena = build_peg_parser(|p| p.literal("hello")).unwrap();
        assert!(parse_str(&arena, "hello world").success());
        let r = parse_str(&arena, "hel");
        assert!(r.need_more_input());
        assert!(parse_str(&arena, "hexlo").fail());
    }

    #[test]
    fn sequence_choice_and_space() {
        let arena = build_peg_parser(|p| {
            let a = p.literal("foo");
            let b = p.literal("bar");
            p.spaced(a, b) // foo space bar
        })
        .unwrap();
        assert!(parse_str(&arena, "foo bar").success());
        assert!(parse_str(&arena, "foo   bar").success());
        // space() matches zero or more spaces (peg-parser.cpp:517-529), so the
        // spaced sequence also accepts the tight form
        assert!(parse_str(&arena, "foobar").success());
    }

    #[test]
    fn chars_and_until() {
        let arena = build_peg_parser(|p| {
            let ident = p.chars("[a-z]+", 1, -1);
            let u = p.until("<end>");
            p.sequence(&[ident, u])
        })
        .unwrap();
        assert!(parse_str(&arena, "abc stuff<end>").success());
        // until at end of lenient input → NEED_MORE
        let r = parse_str(&arena, "abc and more");
        assert!(r.need_more_input());
    }

    #[test]
    fn rule_tag_ast_and_mapper_tags() {
        // tagged parser used by the diff analyzer: pre marker + rest
        let arena = build_peg_parser(|p| {
            let m = p.marker();
            let sp = p.space();
            let seq = p.sequence(&[m, sp]);
            let t = p.tag("pre", seq);
            let rest = p.rest();
            p.sequence(&[t, rest])
        })
        .unwrap();
        let mut ctx = ParseContext::new("<think>\nsome text", PARSE_FLAG_NONE);
        let result = arena.parse(&mut ctx, 0).unwrap();
        assert!(result.success());
        let mut tags: Vec<(String, String)> = Vec::new();
        ctx.ast.visit_result(&result, &mut |node| {
            if !node.tag.is_empty() {
                tags.push((
                    node.tag.clone(),
                    ctx.ast.node_text(node.id, &ctx.input).to_string(),
                ));
            }
        });
        assert_eq!(tags, vec![("pre".to_string(), "<think>\n".to_string())]);
    }

    #[test]
    fn json_rule_set() {
        let arena = build_peg_parser(|p| {
            let j = p.json();
            let end = p.end();
            p.sequence(&[j, end])
        })
        .unwrap();
        assert!(parse_str(&arena, r#"{"a": [1, 2.5, true, null], "b": "x"}"#).success());
        assert!(parse_str(&arena, r#"{"a": }"#).fail());
        // partial JSON in lenient mode → NEED_MORE
        assert!(parse_str(&arena, r#"{"a": [1, 2"#).need_more_input());
    }

    #[test]
    fn save_load_roundtrip() {
        let arena = build_peg_parser(|p| {
            let j = p.json();
            let end = p.end();
            p.sequence(&[j, end])
        })
        .unwrap();
        let saved = arena.save();
        let loaded = PegArena::from_json(&Json::parse(&saved).unwrap()).unwrap();
        assert!(parse_str(&loaded, r#"{"k": true}"#).success());
    }

    #[test]
    fn build_grammar_from_simple_parser() {
        // schema-constrained json value → grammar contains the schema rules
        let schema = Json::parse(r#"{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}"#).unwrap();
        let arena = build_peg_parser(|p| {
            let j = p.json();
            p.schema(j, "test-schema", &schema, false)
        })
        .unwrap();
        let g = arena.build_grammar(false).unwrap();
        assert!(g.contains("root ::="), "grammar: {g}");
        assert!(g.contains("test-schema-city-kv"), "grammar: {g}");
    }

    #[test]
    fn aho_corasick_next_and_terminal() {
        let ac = AhoCorasick::from_strings(&["</param>".to_string(), "</function>".to_string()]);
        assert!(ac.is_terminal(0) == false);
        // walk "</param>"
        let mut s = 0usize;
        for ch in "</param>".chars() {
            s = ac.next(s, ch as u32);
        }
        assert!(ac.is_terminal(s));
    }
}
