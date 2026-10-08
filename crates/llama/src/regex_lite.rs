//! regex_lite.rs — a minimal backtracking regex matcher for lazy-grammar
//! PATTERN triggers (the `std::regex` subset those patterns use).
//!
//! The reference compiles every lazy-grammar trigger into an ECMAScript
//! `std::regex` (`llama_grammar_init_impl`, llama-grammar.cpp:1293-1298) and
//! fires the grammar at the first capture-group position
//! (`llama_grammar_trigger_pattern::find`, llama-grammar.cpp:378-409):
//!
//!   * `find` first tries a *full* match (`std::regex_match`) when the raw
//!     pattern bytes both begin with `^` and end with `$` (:394-400), else /
//!     afterwards searches anywhere (`std::regex_search`, :403-406);
//!   * the fire position is the first *capturing* group with a non-empty
//!     match (lowest group index, `find_start_pos` :379-392), falling back to
//!     the whole match's start.
//!
//! The matcher below reproduces exactly that `find` on bytes (std::string is
//! byte-oriented; the lazy trigger buffer holds raw token pieces, which may be
//! partial UTF-8). Supported constructs — the complete set emitted by the
//! specialized chat parsers (gpt-oss/functionary-v3.2/muse-glimmer) plus what
//! a PATTERN_FULL trigger needs:
//!
//!   * literals, escaped metacharacters (`\|`, `\.`, …: `\` + any
//!     non-alphanumeric ASCII byte is that byte literally)
//!   * `\s` = [09-0D 20] (libstdc++/libc++ "C"-locale isspace — pinned
//!     against the reference in parity/lazy_trigger_ref.txt), `\S`
//!   * `.` = any byte except `\n` and `\r` (ECMAScript line terminators)
//!   * anchors `^` (position 0 only) and `$` (end only; no multiline)
//!   * character classes `[...]` with `^` negation, literals and `a-z` ranges
//!     (a `]` as the first member is a literal, like std::regex/POSIX:
//!     `[]]` = the class {']'}, `[^]]` = "not ]")
//!   * groups: capturing `(...)`, non-capturing `(?:...)`
//!   * alternation `|`
//!   * quantifiers `*` `+` `?` `{m}` `{m,}` `{m,n}` (+ lazy `?` variants) on
//!     simple atoms; `*` `+` `?` also on groups that cannot match empty
//!   * lookahead `(?=...)` / `(?!...)` — zero-width, sub-match anchored at the
//!     current position
//!
//! Loud-fail policy: anything else is a compile-time `Err` naming the
//! construct (never a silent mismatch) — `\d`/`\w`/`\b`/backrefs/control
//! escapes, lookbehind, named groups, inline flags, quantified lookaheads or
//! anchors, `{m,n}` on a group, a quantifier on a group that can match the
//! empty string (an ECMAScript empty-iteration loop), `\s`/`\d`/… inside a
//! class, malformed input, and `{m,n}` with n > 1000. Known divergences from
//! full ECMAScript (all unreachable for the shipped trigger patterns,
//! documented in PARITY.md): a bare `{`/`}`/`]` outside a class is rejected
//! instead of being a literal (Annex B), and captures set inside a
//! *successful* positive lookahead are kept even if a later backtracking path
//! abandons that lookahead.
//!
//! Exponential-backtracking exposure is the same as the reference's
//! `std::regex` (both are backtracking engines over user-suppliable trigger
//! patterns).

// ---------------------------------------------------------------------------
// compiled form
// ---------------------------------------------------------------------------

/// a byte-oriented simple atom (no captures inside, consumes exactly one byte)
#[derive(Clone, Copy, Debug)]
enum SimpleAtom {
    Byte(u8),
    /// `.` — any byte except the ECMAScript line terminators `\n` and `\r`
    /// (libstdc++ byte-mode; 0x85/0xA0/0xFF do match — pinned in
    /// parity/lazy_trigger_ref.txt)
    Any,
    /// `classes` table index
    Class(u32),
}

/// one compiled instruction; `next`/`first`/`second`/`inner` are node indices
/// (`MATCH` terminates a chain)
#[derive(Clone, Copy, Debug)]
enum Node {
    Simple {
        atom: SimpleAtom,
        next: u32,
    },
    /// `atom{min,max}` over a simple atom — matched iteratively (no captures
    /// inside, so retries need no slot restore)
    Repeat {
        atom: SimpleAtom,
        min: u32,
        max: Option<u32>,
        greedy: bool,
        next: u32,
    },
    /// `^` — start of input only
    AssertStart {
        next: u32,
    },
    /// `$` — end of input only
    AssertEnd {
        next: u32,
    },
    /// record the position in `slots[slot]` (restored on backtrack)
    Save {
        slot: u32,
        next: u32,
    },
    /// try `first`, else `second` (greedy repetition: `first` = body)
    Split {
        first: u32,
        second: u32,
    },
    /// `(?=...)` / `(?!...)` — zero-width; `inner` is matched as a prefix at
    /// the current position
    Look {
        negated: bool,
        inner: u32,
        next: u32,
    },
    Match,
}

/// `[...]` — `ranges` (inclusive byte bounds), negated when `negated`
#[derive(Clone, Debug)]
struct ClassSet {
    negated: bool,
    ranges: Vec<(u8, u8)>,
}

impl ClassSet {
    fn contains(&self, b: u8) -> bool {
        let hit = self.ranges.iter().any(|&(lo, hi)| b >= lo && b <= hi);
        hit != self.negated
    }
}

/// the compiled trigger regex — `find` is `llama_grammar_trigger_pattern::find`
/// (llama-grammar.cpp:378-409)
#[derive(Debug)]
pub struct RegexLite {
    /// the raw pattern bytes — the `^`…`$` full-match fast path (:394) checks
    /// them, not the compiled form (an escaped `\$` at the end still takes the
    /// full-match path, exactly like the C)
    pattern: Vec<u8>,
    nodes: Vec<Node>,
    classes: Vec<ClassSet>,
    entry: u32,
    /// number of capturing groups (`(...)` only)
    n_groups: usize,
}

/// index of the shared terminal `Match` node
const MATCH: u32 = 0;

/// `std::regex(pattern)` + `llama_grammar_trigger_pattern::find` over `input`:
/// `Some(fire_position)` or `None` (`std::string::npos`).
///
/// Mirrors llama-grammar.cpp:378-409 exactly: the anchored full match is tried
/// first when the raw pattern begins with `^` and ends with `$`, then a search
/// anywhere; the position is the first non-empty capture group's start, else
/// the match start.
pub fn find(pattern: &str, input: &[u8]) -> Result<Option<usize>, String> {
    let re = RegexLite::new(pattern)?;
    Ok(re.find(input))
}

impl RegexLite {
    /// `std::regex(trigger.pattern)` (llama-grammar.cpp:1297) for the
    /// supported subset; `Err` names any unsupported construct (loud fail)
    pub fn new(pattern: &str) -> Result<RegexLite, String> {
        let mut c = Compiler {
            pat: pattern.as_bytes(),
            pos: 0,
            n_groups: 0,
            classes: Vec::new(),
        };
        let ast = c.parse_alt()?;
        if c.pos != c.pat.len() {
            // only an unbalanced ')' can stop the parser early
            return Err(err(pattern, c.pos, "unbalanced ')'"));
        }
        let n_groups = c.n_groups;

        let mut g = Codegen {
            nodes: vec![Node::Match], // nodes[MATCH]
            classes: c.classes,
        };
        let entry = g.seq(&ast, MATCH);
        Ok(RegexLite {
            pattern: pattern.as_bytes().to_vec(),
            nodes: g.nodes,
            classes: g.classes,
            entry,
            n_groups,
        })
    }

    /// `llama_grammar_trigger_pattern::find` (llama-grammar.cpp:378-409)
    pub fn find(&self, input: &[u8]) -> Option<usize> {
        // :394-400 — a pattern whose raw bytes begin with '^' and end with '$'
        // is first tried as a full match of the entire input
        let anchored = !self.pattern.is_empty()
            && self.pattern.first() == Some(&b'^')
            && self.pattern.last() == Some(&b'$');
        if anchored {
            if let Some(slots) = self.exec(input, true) {
                return Some(start_pos(&slots, self.n_groups));
            }
        }
        // :403-406 — search anywhere (leftmost match)
        self.exec(input, false)
            .map(|slots| start_pos(&slots, self.n_groups))
    }

    /// run the program: `full` = `std::regex_match` (must span the whole
    /// input), else `std::regex_search` (leftmost start wins). Returns the
    /// capture slots on success (slot 2g/2g+1 = group g start/end; slots
    /// 0/1 = whole match).
    fn exec(&self, input: &[u8], full: bool) -> Option<Vec<Option<u32>>> {
        let n_slots = 2 * (self.n_groups + 1);
        if full {
            let mut slots = vec![None; n_slots];
            slots[0] = Some(0);
            if let Some(end) = self.m(input, self.entry, 0, &mut slots) {
                if end == input.len() {
                    slots[1] = Some(end as u32);
                    return Some(slots);
                }
            }
            None
        } else {
            for start in 0..=input.len() {
                let mut slots = vec![None; n_slots];
                slots[0] = Some(start as u32);
                if let Some(end) = self.m(input, self.entry, start, &mut slots) {
                    slots[1] = Some(end as u32);
                    return Some(slots);
                }
            }
            None
        }
    }

    fn atom_matches(&self, atom: SimpleAtom, b: u8) -> bool {
        match atom {
            SimpleAtom::Byte(x) => b == x,
            SimpleAtom::Any => b != b'\n' && b != b'\r',
            SimpleAtom::Class(i) => self.classes[i as usize].contains(b),
        }
    }

    /// backtracking matcher — ECMAScript semantics: alternatives in source
    /// order, greedy quantifiers longest-first (lazy shortest-first).
    /// `Save` restores its slot when its continuation fails, so any failing
    /// path leaves `slots` exactly as it found them.
    fn m(&self, input: &[u8], node: u32, pos: usize, slots: &mut [Option<u32>]) -> Option<usize> {
        match self.nodes[node as usize] {
            Node::Match => Some(pos),
            Node::Simple { atom, next } => {
                if pos < input.len() && self.atom_matches(atom, input[pos]) {
                    self.m(input, next, pos + 1, slots)
                } else {
                    None
                }
            }
            Node::Repeat {
                atom,
                min,
                max,
                greedy,
                next,
            } => {
                // count the maximal run of the atom (bounded by max)
                let mut k = 0usize;
                let mut p = pos;
                let lim = max.map_or(usize::MAX, |m| m as usize);
                while k < lim && p < input.len() && self.atom_matches(atom, input[p]) {
                    p += 1;
                    k += 1;
                }
                if k < min as usize {
                    return None;
                }
                if greedy {
                    let mut kk = k;
                    loop {
                        if let Some(e) = self.m(input, next, pos + kk, slots) {
                            return Some(e);
                        }
                        if kk == min as usize {
                            return None;
                        }
                        kk -= 1;
                    }
                } else {
                    for kk in min as usize..=k {
                        if let Some(e) = self.m(input, next, pos + kk, slots) {
                            return Some(e);
                        }
                    }
                    None
                }
            }
            Node::AssertStart { next } => {
                if pos == 0 {
                    self.m(input, next, pos, slots)
                } else {
                    None
                }
            }
            Node::AssertEnd { next } => {
                if pos == input.len() {
                    self.m(input, next, pos, slots)
                } else {
                    None
                }
            }
            Node::Save { slot, next } => {
                let old = slots[slot as usize];
                slots[slot as usize] = Some(pos as u32);
                match self.m(input, next, pos, slots) {
                    Some(e) => Some(e),
                    None => {
                        slots[slot as usize] = old;
                        None
                    }
                }
            }
            Node::Split { first, second } => {
                if let Some(e) = self.m(input, first, pos, slots) {
                    Some(e)
                } else {
                    self.m(input, second, pos, slots)
                }
            }
            Node::Look {
                negated,
                inner,
                next,
            } => {
                let mut scratch = vec![None; slots.len()];
                let matched = self.m(input, inner, pos, &mut scratch).is_some();
                if matched != negated {
                    // ECMAScript keeps captures set inside a successful
                    // positive lookahead (negative discards them)
                    if matched {
                        for i in 0..slots.len() {
                            if scratch[i].is_some() {
                                slots[i] = scratch[i];
                            }
                        }
                    }
                    self.m(input, next, pos, slots)
                } else {
                    None
                }
            }
        }
    }
}

/// `find_start_pos` (llama-grammar.cpp:379-392): the first capturing group
/// (lowest index) whose match is non-empty, else the whole match's start
fn start_pos(slots: &[Option<u32>], n_groups: usize) -> usize {
    for g in 1..=n_groups {
        if let (Some(a), Some(b)) = (slots[2 * g], slots[2 * g + 1]) {
            if b > a {
                return a as usize;
            }
        }
    }
    slots[0].unwrap() as usize
}

fn err(pattern: &str, at: usize, what: &str) -> String {
    format!("regex_lite: unsupported regex trigger pattern {pattern:?} at byte {at}: {what}")
}

// ---------------------------------------------------------------------------
// parser (pattern bytes → AST)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Ast {
    Simple(SimpleAtom),
    Start,
    End,
    /// capturing (`idx = Some(g)`) or non-capturing group around an alternation
    Group {
        idx: Option<usize>,
        inner: Box<Ast>,
    },
    Look {
        negated: bool,
        inner: Box<Ast>,
    },
    Rep {
        atom: Box<Ast>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    /// alternation branches
    Alt(Vec<Ast>),
    /// sequence pieces
    Seq(Vec<Ast>),
}

/// can `ast` match the empty string? (a quantified group that can would loop
/// forever in the Split-loop compilation — ECMAScript breaks empty iterations,
/// the port rejects the pattern loudly instead)
fn nullable(ast: &Ast) -> bool {
    match ast {
        Ast::Simple(_) => false,
        Ast::Start | Ast::End | Ast::Look { .. } => true, // zero-width
        Ast::Group { inner, .. } => nullable(inner),
        Ast::Rep { atom, min, .. } => *min == 0 || nullable(atom),
        Ast::Alt(branches) => branches.iter().any(nullable),
        Ast::Seq(pieces) => pieces.iter().all(nullable),
    }
}

struct Compiler<'a> {
    pat: &'a [u8],
    pos: usize,
    n_groups: usize,
    classes: Vec<ClassSet>,
}

impl<'a> Compiler<'a> {
    fn peek(&self) -> Option<u8> {
        self.pat.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    /// `alt := seq ('|' seq)*`
    fn parse_alt(&mut self) -> Result<Ast, String> {
        let mut branches = vec![self.parse_seq()?];
        while self.peek() == Some(b'|') {
            self.pos += 1;
            branches.push(self.parse_seq()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap()
        } else {
            Ast::Alt(branches)
        })
    }

    /// `seq := rep*`
    fn parse_seq(&mut self) -> Result<Ast, String> {
        let mut pieces = Vec::new();
        while let Some(b) = self.peek() {
            if b == b'|' || b == b')' {
                break;
            }
            pieces.push(self.parse_rep()?);
        }
        Ok(Ast::Seq(pieces))
    }

    /// `rep := atom ('*'|'+'|'?'|'{m,n}') '?'?`
    fn parse_rep(&mut self) -> Result<Ast, String> {
        let atom = self.parse_atom()?;
        let (min, max) = match self.peek() {
            Some(b'*') => {
                self.pos += 1;
                (0, None)
            }
            Some(b'+') => {
                self.pos += 1;
                (1, None)
            }
            Some(b'?') => {
                self.pos += 1;
                (0, Some(1))
            }
            Some(b'{') => {
                let save = self.pos;
                match self.parse_bounds()? {
                    Some(mm) => mm,
                    None => {
                        // not a valid {m,n} — reject rather than fall back to a
                        // literal '{' (loud fail, see module docs)
                        return Err(err(
                            self.pat_str(),
                            save,
                            "'{' is not a quantifier and a literal '{' is not supported",
                        ));
                    }
                }
            }
            _ => return Ok(atom),
        };
        // lazy marker
        let greedy = if self.peek() == Some(b'?') {
            self.pos += 1;
            false
        } else {
            true
        };
        // a second quantifier is malformed (possessive `++` is not ECMAScript)
        match self.peek() {
            Some(b'*') | Some(b'+') | Some(b'?') => {
                return Err(err(self.pat_str(), self.pos, "double quantifier"));
            }
            Some(b'{') => {
                let save = self.pos;
                if self.parse_bounds()?.is_some() {
                    return Err(err(self.pat_str(), save, "double quantifier"));
                }
                self.pos = save;
            }
            _ => {}
        }
        if max.is_some_and(|m| m > 1000) {
            return Err(err(
                self.pat_str(),
                self.pos,
                "repetition bound {m,n} > 1000",
            ));
        }
        match atom {
            // simple atoms: compiled to the iterative Repeat node
            Ast::Simple(_) => Ok(Ast::Rep {
                atom: Box::new(atom),
                min,
                max,
                greedy,
            }),
            Ast::Group { .. } => {
                // `*` / `+` / `?` on a group compile to a Split loop; the
                // loop cannot enforce a bounded repeat count or a minimum
                // above 1, so the braced forms are rejected
                if max.is_some_and(|m| m > 1) || min > 1 {
                    return Err(err(
                        self.pat_str(),
                        self.pos,
                        "{m,n} on a group is not supported (use * / + / ?)",
                    ));
                }
                if nullable(&atom) {
                    // `X*`/`X+`/`X?` over a possibly-empty body: the Split
                    // loop could iterate forever (ECMAScript breaks empty
                    // iterations) — reject loudly instead
                    return Err(err(
                        self.pat_str(),
                        self.pos,
                        "quantifier on a group that can match the empty string",
                    ));
                }
                Ok(Ast::Rep {
                    atom: Box::new(atom),
                    min,
                    max,
                    greedy,
                })
            }
            // lookaheads are zero-width: `(?!x)*` is either a no-op or a
            // forever-loop — reject loudly
            Ast::Look { .. } => Err(err(
                self.pat_str(),
                self.pos,
                "quantified '(?=…)'/'(?!…)' lookahead",
            )),
            // `^*`, `$+` … are legal ECMAScript but meaningless; reject loudly
            Ast::Start | Ast::End => {
                Err(err(self.pat_str(), self.pos, "quantified '^'/'$' anchor"))
            }
            Ast::Alt(_) | Ast::Seq(_) | Ast::Rep { .. } => unreachable!(),
        }
    }

    /// `{m}` / `{m,}` / `{m,n}` — the `{` is already peeked; `None` when the
    /// bytes are not a quantifier
    fn parse_bounds(&mut self) -> Result<Option<(u32, Option<u32>)>, String> {
        let start = self.pos;
        debug_assert_eq!(self.peek(), Some(b'{'));
        self.pos += 1;
        let min = match self.parse_num() {
            Some(n) => n,
            None => {
                self.pos = start;
                return Ok(None);
            }
        };
        let max = if self.peek() == Some(b',') {
            self.pos += 1;
            match self.parse_num() {
                Some(n) => Some(n),
                None => None, // `{m,}`
            }
        } else {
            Some(min) // `{m}`
        };
        if self.peek() != Some(b'}') {
            self.pos = start;
            return Ok(None);
        }
        self.pos += 1;
        if let Some(m) = max {
            if m < min {
                return Err(err(self.pat_str(), start, "{m,n} with n < m"));
            }
        }
        Ok(Some((min, max)))
    }

    fn parse_num(&mut self) -> Option<u32> {
        let start = self.pos;
        let mut n = 0u32;
        while let Some(b) = self.peek() {
            if !b.is_ascii_digit() {
                break;
            }
            n = n.saturating_mul(10).saturating_add((b - b'0') as u32);
            self.pos += 1;
        }
        if self.pos == start {
            None
        } else {
            Some(n)
        }
    }

    fn parse_atom(&mut self) -> Result<Ast, String> {
        let b = self
            .bump()
            .ok_or_else(|| err(self.pat_str(), self.pos, "unexpected end of pattern"))?;
        match b {
            b'(' => {
                let (kind, idx) = if self.peek() == Some(b'?') {
                    self.pos += 1;
                    match self.peek() {
                        Some(b':') => {
                            self.pos += 1;
                            (GroupKind::NonCap, None)
                        }
                        Some(b'=') => {
                            self.pos += 1;
                            (GroupKind::Look(false), None)
                        }
                        Some(b'!') => {
                            self.pos += 1;
                            (GroupKind::Look(true), None)
                        }
                        Some(b'<') => {
                            return Err(err(
                                self.pat_str(),
                                self.pos,
                                "lookbehind '(?<='/'(?<!' and named groups '(?<name>' are not supported",
                            ));
                        }
                        other => {
                            return Err(err(
                                self.pat_str(),
                                self.pos,
                                &format!(
                                    "group modifier '(?{}' is not supported (inline flags, unicode escapes, …)",
                                    other.map_or('?', |c| c as char)
                                ),
                            ));
                        }
                    }
                } else {
                    self.n_groups += 1;
                    (GroupKind::Cap, Some(self.n_groups))
                };
                let inner = self.parse_alt()?;
                if self.peek() != Some(b')') {
                    return Err(err(self.pat_str(), self.pos, "unbalanced '('"));
                }
                self.pos += 1;
                Ok(match kind {
                    GroupKind::Cap => Ast::Group {
                        idx,
                        inner: Box::new(inner),
                    },
                    GroupKind::NonCap => Ast::Group {
                        idx: None,
                        inner: Box::new(inner),
                    },
                    GroupKind::Look(negated) => Ast::Look {
                        negated,
                        inner: Box::new(inner),
                    },
                })
            }
            b'[' => self.parse_class().map(Ast::Simple),
            b'.' => Ok(Ast::Simple(SimpleAtom::Any)),
            b'^' => Ok(Ast::Start),
            b'$' => Ok(Ast::End),
            b'\\' => self.parse_escape().map(Ast::Simple),
            b')' => unreachable!(), // parse_seq stops before ')'
            b'*' | b'+' | b'?' => Err(err(
                self.pat_str(),
                self.pos - 1,
                "quantifier with nothing to repeat",
            )),
            b']' | b'}' => Err(err(
                self.pat_str(),
                self.pos - 1,
                "literal ']'/'}' must be escaped",
            )),
            _ => Ok(Ast::Simple(SimpleAtom::Byte(b))),
        }
    }

    /// the escape after `\` — outside a class
    fn parse_escape(&mut self) -> Result<SimpleAtom, String> {
        let at = self.pos - 1;
        let b = self
            .bump()
            .ok_or_else(|| err(self.pat_str(), at, "trailing '\\'"))?;
        match b {
            b's' => {
                let idx = self.push_class(ClassSet {
                    negated: false,
                    // "C"-locale isspace (libstdc++/libc++ agree for ASCII;
                    // 0x85/0xA0 do NOT match — pinned in lazy_trigger_ref.txt)
                    ranges: vec![(0x09, 0x0D), (0x20, 0x20)],
                });
                Ok(SimpleAtom::Class(idx))
            }
            b'S' => {
                let idx = self.push_class(ClassSet {
                    negated: true,
                    ranges: vec![(0x09, 0x0D), (0x20, 0x20)],
                });
                Ok(SimpleAtom::Class(idx))
            }
            b if b.is_ascii_alphanumeric() => Err(err(
                self.pat_str(),
                at,
                &format!(
                    "escape '\\{}' is not supported (only \\s \\S and escaped metacharacters)",
                    b as char
                ),
            )),
            b if b.is_ascii() => Ok(SimpleAtom::Byte(b)), // `\|`, `\.`, `\\`, …
            _ => Err(err(self.pat_str(), at, "non-ASCII escape is not supported")),
        }
    }

    /// `[...]` — negation, literals, `a-z` ranges, escaped literals
    fn parse_class(&mut self) -> Result<SimpleAtom, String> {
        let open = self.pos - 1;
        let negated = if self.peek() == Some(b'^') {
            self.pos += 1;
            true
        } else {
            false
        };
        let mut ranges: Vec<(u8, u8)> = Vec::new();
        let mut first = true;
        loop {
            // std::regex (libstdc++/libc++) follows the POSIX convention: a
            // ']' as the first member (after the optional '^') is a literal,
            // so `[]]` is the class {']} and `[^]]` is "not ]"
            let b = match self.peek() {
                None => return Err(err(self.pat_str(), open, "unterminated '['")),
                Some(b']') if !first => {
                    self.pos += 1;
                    break;
                }
                Some(b) => b,
            };
            first = false;
            let lo = self.class_byte()?;
            // range `a-z` (a `-` at the very end / before `]` is literal)
            let hi = if self.peek() == Some(b'-')
                && self.pat.get(self.pos + 1).is_some_and(|&c| c != b']')
            {
                self.pos += 1;
                let at = self.pos;
                let h = self.class_byte()?;
                if h < lo {
                    return Err(err(self.pat_str(), at, "class range with hi < lo"));
                }
                h
            } else {
                lo
            };
            ranges.push((lo, hi));
        }
        if ranges.is_empty() && !negated {
            // ECMAScript `[]` never matches; reject loudly instead
            return Err(err(self.pat_str(), open, "empty class '[]'"));
        }
        let idx = self.push_class(ClassSet { negated, ranges });
        Ok(SimpleAtom::Class(idx))
    }

    /// one literal byte inside `[...]` (escaped metacharacters allowed,
    /// `\s`/`\d`/… rejected — loud fail)
    fn class_byte(&mut self) -> Result<u8, String> {
        let b = self
            .bump()
            .ok_or_else(|| err(self.pat_str(), self.pos, "unterminated '['"))?;
        if b != b'\\' {
            return Ok(b);
        }
        let at = self.pos - 1;
        let e = self
            .bump()
            .ok_or_else(|| err(self.pat_str(), at, "trailing '\\'"))?;
        if e.is_ascii_alphanumeric() {
            return Err(err(
                self.pat_str(),
                at,
                &format!(
                    "escape '\\{}' inside [...] is not supported (only escaped literals)",
                    e as char
                ),
            ));
        }
        Ok(e)
    }

    fn push_class(&mut self, set: ClassSet) -> u32 {
        self.classes.push(set);
        (self.classes.len() - 1) as u32
    }

    fn pat_str(&self) -> &str {
        // patterns are valid UTF-8 (they arrive as JSON strings); only used
        // in error messages
        std::str::from_utf8(self.pat).unwrap_or("<invalid utf8>")
    }
}

enum GroupKind {
    Cap,
    NonCap,
    Look(bool),
}

// ---------------------------------------------------------------------------
// codegen (AST → node graph, compiled back-to-front so `next` is known)
// ---------------------------------------------------------------------------

struct Codegen {
    nodes: Vec<Node>,
    classes: Vec<ClassSet>,
}

impl Codegen {
    fn push(&mut self, n: Node) -> u32 {
        self.nodes.push(n);
        (self.nodes.len() - 1) as u32
    }

    /// compile an alternation / sequence chain ending at `next`
    fn seq(&mut self, ast: &Ast, next: u32) -> u32 {
        match ast {
            Ast::Seq(pieces) => {
                let mut cur = next;
                for p in pieces.iter().rev() {
                    cur = self.piece(p, cur);
                }
                cur
            }
            Ast::Alt(branches) => {
                // Split chain in branch order (first alternative tried first)
                let entries: Vec<u32> = branches.iter().map(|b| self.seq(b, next)).collect();
                let mut it = entries.into_iter().rev();
                let mut cur = it.next().unwrap();
                for e in it {
                    cur = self.push(Node::Split {
                        first: e,
                        second: cur,
                    });
                }
                cur
            }
            other => self.piece(other, next),
        }
    }

    /// compile one sequence piece (already quantifier-wrapped by the parser)
    fn piece(&mut self, ast: &Ast, next: u32) -> u32 {
        match ast {
            Ast::Simple(atom) => self.push(Node::Simple { atom: *atom, next }),
            Ast::Start => self.push(Node::AssertStart { next }),
            Ast::End => self.push(Node::AssertEnd { next }),
            Ast::Group { idx, inner } => match idx {
                // capturing group: Save(start) body Save(end) — the C's
                // `find_start_pos` reads exactly these spans
                Some(g) => {
                    let start_save = self.push(Node::Save {
                        slot: (2 * g) as u32,
                        next: MATCH,
                    });
                    let end_save = self.push(Node::Save {
                        slot: (2 * g + 1) as u32,
                        next,
                    });
                    let body = self.seq(inner, end_save);
                    self.nodes[start_save as usize] = Node::Save {
                        slot: (2 * g) as u32,
                        next: body,
                    };
                    start_save
                }
                None => self.seq(inner, next),
            },
            Ast::Look { negated, inner } => {
                // the sub-pattern must match a *prefix* at the current
                // position (zero-width for the outer match)
                let inner_entry = self.seq(inner, MATCH);
                self.push(Node::Look {
                    negated: *negated,
                    inner: inner_entry,
                    next,
                })
            }
            Ast::Rep {
                atom,
                min,
                max,
                greedy,
            } => match &**atom {
                Ast::Simple(sa) => self.push(Node::Repeat {
                    atom: *sa,
                    min: *min,
                    max: *max,
                    greedy: *greedy,
                    next,
                }),
                Ast::Group { .. } => {
                    // `X*`/`X+`/`X?` over a group (never empty-matching —
                    // the parser rejects those): a Split loop
                    //   L0: Split(body → L0, next)   (greedy; lazy swaps)
                    // `+` compiles the body once in front of the loop so at
                    // least one iteration ran.
                    let split = self.push(Node::Split {
                        first: MATCH,
                        second: next,
                    });
                    let body = self.group(atom, split);
                    let (first, second) = if *greedy { (body, next) } else { (next, body) };
                    self.nodes[split as usize] = Node::Split { first, second };
                    if *min >= 1 {
                        body // X+: enter through the mandatory copy
                    } else {
                        split // X*/X?: enter through the loop split
                    }
                }
                _ => unreachable!("parser wrapped {:?} in Rep", atom),
            },
            Ast::Alt(_) | Ast::Seq(_) => unreachable!("seq() handles these"),
        }
    }

    /// one unquantified group copy whose tail loops back to `loop_to`
    fn group(&mut self, ast: &Ast, loop_to: u32) -> u32 {
        match ast {
            Ast::Group { idx, inner } => match idx {
                Some(g) => {
                    let start_save = self.push(Node::Save {
                        slot: (2 * g) as u32,
                        next: MATCH,
                    });
                    let end_save = self.push(Node::Save {
                        slot: (2 * g + 1) as u32,
                        next: loop_to,
                    });
                    let body = self.seq(inner, end_save);
                    self.nodes[start_save as usize] = Node::Save {
                        slot: (2 * g) as u32,
                        next: body,
                    };
                    start_save
                }
                None => self.seq(inner, loop_to),
            },
            other => unreachable!("group() on {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn f(pattern: &str, input: &str) -> Option<usize> {
        RegexLite::new(pattern).unwrap().find(input.as_bytes())
    }

    fn fb(pattern: &str, input: &[u8]) -> Option<usize> {
        RegexLite::new(pattern).unwrap().find(input)
    }

    fn bad(pattern: &str) -> String {
        RegexLite::new(pattern).unwrap_err()
    }

    // -- the six trigger patterns the chat parsers actually emit -------------

    // gpt-oss `^\s+to$` (parsers/gpt-oss.cpp:150) — anchored → full match only
    #[test]
    fn gptoss_anchored_to() {
        assert_eq!(f(r"^\s+to$", " to"), Some(0));
        assert_eq!(f(r"^\s+to$", "\t\n to"), Some(0)); // \s+ spans the tabs/newline
        assert_eq!(f(r"^\s+to$", "x to"), None); // ^: no search-anywhere fallback
        assert_eq!(f(r"^\s+to$", " to "), None); // $: trailing space
        assert_eq!(f(r"^\s+to$", " tox"), None);
        assert_eq!(f(r"^\s+to$", "to"), None); // \s+ needs one whitespace
        assert_eq!(f(r"^\s+to$", ""), None);
    }

    // gpt-oss `^<\|channel\|>(?:commentary|analysis)\s+to=functions$`
    // (parsers/gpt-oss.cpp:151) — no capturing group → whole-match start (0)
    #[test]
    fn gptoss_anchored_channel() {
        let p = r"^<\|channel\|>(?:commentary|analysis)\s+to=functions$";
        assert_eq!(f(p, "<|channel|>commentary to=functions"), Some(0));
        assert_eq!(f(p, "<|channel|>analysis to=functions"), Some(0));
        assert_eq!(f(p, "<|channel|>commentary  to=functions"), Some(0)); // \s+ greedy
        assert_eq!(f(p, "<|channel|>final to=functions"), None);
        assert_eq!(f(p, "x<|channel|>commentary to=functions"), None);
        assert_eq!(f(p, "<|channel|>commentary to=function"), None);
        assert_eq!(f(p, "<|channel|>commentary to=functions\n"), None);
    }

    // gpt-oss `<\|start\|>assistant(\s+to)` (parsers/gpt-oss.cpp:152) — the
    // capture decides the fire position
    #[test]
    fn gptoss_capture_to() {
        let p = r"<\|start\|>assistant(\s+to)";
        // "foo" (3) + "<|start|>assistant" (9+9=18 bytes) → capture at 21
        assert_eq!(f(p, "foo<|start|>assistant to"), Some(21));
        // two spaces: \s+ backtracks to swallow both, capture still at 18
        assert_eq!(f(p, "<|start|>assistant  to"), Some(18));
        assert_eq!(f(p, "zz<|start|>assistant to"), Some(20));
        // incomplete: no "to" buffered yet
        assert_eq!(f(p, "<|start|>assistant "), None);
        // the match need not reach the end of the buffer — "tox" still fires
        // at the capture start
        assert_eq!(f(p, "<|start|>assistant tox"), Some(18));
        assert_eq!(f(p, "<|start|>assistant"), None);
        // \s+ needs whitespace: "assistantto" does not fire
        assert_eq!(f(p, "<|start|>assistantto"), None);
    }

    // gpt-oss `<\|start\|>assistant(<\|channel\|>(?:commentary|analysis)\s+to)`
    // (parsers/gpt-oss.cpp:153)
    #[test]
    fn gptoss_capture_channel() {
        let p = r"<\|start\|>assistant(<\|channel\|>(?:commentary|analysis)\s+to)";
        // capture starts right after "assistant" (byte 18)
        assert_eq!(f(p, "<|start|>assistant<|channel|>commentary to"), Some(18));
        assert_eq!(f(p, "x<|start|>assistant<|channel|>analysis  to"), Some(19));
        assert_eq!(f(p, "<|start|>assistant<|channel|>final to"), None);
        assert_eq!(f(p, "<|start|>assistant<|channel|>commentary"), None);
    }

    // functionary v3.2 `>>>(?!all)` (parsers/functionary-v3-2.cpp:91)
    #[test]
    fn functionary_negative_lookahead() {
        let p = r">>>(?!all)";
        assert_eq!(f(p, ">>>"), Some(0));
        assert_eq!(f(p, ">>>all"), None);
        assert_eq!(f(p, ">>>x"), Some(0));
        assert_eq!(f(p, ">>> "), Some(0));
        assert_eq!(f(p, ">>>al"), Some(0)); // "all" not complete yet
        assert_eq!(f(p, ">>>allz"), None); // "all" is a prefix — still blocked
        assert_eq!(f(p, ">>>>all"), Some(0)); // lookahead at 3 sees ">all"
        assert_eq!(f(p, "x>>>all"), None);
        assert_eq!(f(p, "x>>>get_weather"), Some(1));
        assert_eq!(f(p, ">>"), None);
    }

    // muse-glimmer
    // `(?:^|<\|start\|>assistant)( to=(?!self<\|message\|>)(?!user<\|message\|>)[^<]*?<\|message\|>)`
    // (parsers/muse-glimmer.cpp:131-134) — lazy `[^<]*?` + two lookaheads
    #[test]
    fn muse_glimmer_trigger() {
        let p = r"(?:^|<\|start\|>assistant)( to=(?!self<\|message\|>)(?!user<\|message\|>)[^<]*?<\|message\|>)";
        // at position 0 the `^` alternative matches empty; capture starts at 0
        assert_eq!(f(p, " to=functions\n{\"a\":1}<|message|>"), Some(0));
        // after <|start|>assistant (18 bytes): capture at 18
        assert_eq!(f(p, "<|start|>assistant to=functions<|message|>"), Some(18));
        // lookaheads block the self/user recipients
        assert_eq!(f(p, " to=self<|message|>"), None);
        assert_eq!(f(p, " to=user<|message|>x"), None);
        // mid-buffer without the assistant marker cannot fire (^ only at 0)
        assert_eq!(f(p, "x to=functions<|message|>"), None);
        // lazy scan: [^<]*? crosses newlines but stops at the first '<'
        assert_eq!(f(p, " to=abc\ndef<|message|>tail<|message|>"), Some(0));
        // incomplete: the closing <|message|> has not been buffered yet
        assert_eq!(f(p, " to=functions"), None);
        // a '<' before <|message|> kills the lazy scan
        assert_eq!(f(p, " to=a<b<|message|>"), None);
        assert_eq!(f(p, "<|start|>assistant to=self<|message|>"), None);
    }

    // -- PATTERN_FULL anchoring shape (`^x$`) and the find fast path ---------

    #[test]
    fn pattern_full_shape() {
        // sampling.cpp:235-245 anchors a PATTERN_FULL trigger; the engine
        // builds the string, here we check the anchored form's behavior
        assert_eq!(f("^hello$", "hello"), Some(0));
        assert_eq!(f("^hello$", "hell"), None);
        assert_eq!(f("^hello$", "hellox"), None);
        assert_eq!(f("^hello$", "xhello"), None);
        assert_eq!(f("^$", ""), Some(0)); // empty trigger: fires on empty buffer
        assert_eq!(f("^$", "x"), None);
    }

    #[test]
    fn anchored_pattern_with_capture() {
        // a `^…$` pattern with a capture still reports the capture start
        assert_eq!(f(r"^x(\s+to)$", "x to"), Some(1));
        assert_eq!(f(r"^(\s+to)$", " to"), Some(0));
        assert_eq!(f(r"^x(\s+to)$", "x toto"), None);
    }

    // -- constructs ----------------------------------------------------------

    #[test]
    fn literals_and_escapes() {
        assert_eq!(f("abc", "xxabcyy"), Some(2)); // leftmost
        assert_eq!(f("abc", "ababc"), Some(2));
        assert_eq!(f("a.c", "abc"), Some(0)); // `.` matches 'b'
        assert_eq!(f("a.c", "a\nc"), None); // `.` is not `\n`
        assert_eq!(f("a.c", "a\rc"), None); // …nor `\r` (line terminator)
        assert_eq!(fb("^.$", &[0x85]), Some(0)); // NEL is not a terminator
        assert_eq!(f(r"a\.c", "a.c"), Some(0));
        assert_eq!(f(r"a\.c", "abc"), None);
        assert_eq!(f(r"\(\)\[\]\{\}\|\?\*\+\^\$\\", "()[]{}|?*+^$\\"), Some(0));
        assert_eq!(f(r"\/\-\:", "/-:"), Some(0));
        assert_eq!(f("a|ab", "ab"), Some(0)); // alternation: first branch wins
        assert_eq!(f("ab|a", "a"), Some(0));
    }

    #[test]
    fn classes() {
        assert_eq!(f("[abc]+", "zzcabz"), Some(2));
        assert_eq!(f("[a-c]+", "zzabcz"), Some(2));
        assert_eq!(f("[^abc]+", "abczzz"), Some(3));
        assert_eq!(f("[^]]+", "]]x]]"), Some(2));
        assert_eq!(f(r"[\]\[]+", "a[]b"), Some(1));
        assert_eq!(f("[^x]+", "x\ty\nz"), Some(1)); // negated crosses \n
        assert_eq!(f("[-x]+", "a-x"), Some(1)); // leading '-' is literal
        assert_eq!(f("[x-]+", "a-x"), Some(1)); // trailing '-' is literal
                                                // first-']'-literal (std::regex/POSIX): `[^]]` = "not ]"
        assert_eq!(f("[^]]+", "\n\x00"), Some(0));
        assert_eq!(f("[^]]+", "x]"), Some(0)); // stops before the ']'
                                               // non-ASCII bytes in classes/buffers are plain bytes
        assert_eq!(fb("[^\x00-\x7f]+", b"\xc3\xa9"), Some(0));
    }

    #[test]
    fn whitespace_classes() {
        // \s = {09..0D, 20} — everything else (incl. 0x85 NEL, 0xA0 NBSP)
        // does not match, matching the reference's std::regex on bytes
        for b in [0x09u8, 0x0A, 0x0B, 0x0C, 0x0D, 0x20] {
            assert_eq!(fb(r"^\s$", &[b]), Some(0), "byte {b:#x}");
        }
        for b in [0x00u8, 0x41, 0x7f, 0x85, 0xa0] {
            assert_eq!(fb(r"^\s$", &[b]), None, "byte {b:#x}");
        }
        for b in [0x09u8, 0x0A, 0x0B, 0x0C, 0x0D, 0x20] {
            assert_eq!(fb(r"^\S$", &[b]), None, "byte {b:#x}");
        }
        assert_eq!(fb(r"^\S$", &[0x85]), Some(0));
        assert_eq!(fb(r"^\S$", &[0xa0]), Some(0));
    }

    #[test]
    fn anchors() {
        assert_eq!(f("^abc", "abcd"), Some(0));
        assert_eq!(f("^abc", "xabc"), None);
        assert_eq!(f("abc$", "xxabc"), Some(2));
        assert_eq!(f("abc$", "abcx"), None);
        assert_eq!(f("^abc$", "abc"), Some(0));
        // no multiline: ^ only at 0, $ only at the very end
        assert_eq!(f("^a$", "a\n"), None);
        assert_eq!(f("^a$", "\na"), None);
        assert_eq!(f("a$", "a\n"), None);
    }

    #[test]
    fn quantifiers() {
        assert_eq!(f("^a*$", ""), Some(0));
        assert_eq!(f("^a*$", "aaa"), Some(0));
        assert_eq!(f("^a+$", ""), None);
        assert_eq!(f("^a?b$", "b"), Some(0));
        assert_eq!(f("^a?b$", "ab"), Some(0));
        assert_eq!(f("^a?b$", "aab"), None);
        assert_eq!(f("^a{3}$", "aaa"), Some(0));
        assert_eq!(f("^a{3}$", "aa"), None);
        assert_eq!(f("^a{2,}$", "aaaaa"), Some(0));
        assert_eq!(f("^a{2,}$", "a"), None);
        assert_eq!(f("^a{2,3}$", "aaa"), Some(0));
        assert_eq!(f("^a{2,3}$", "aaaa"), None);
        assert_eq!(f("^a{2,3}?$", "aa"), Some(0)); // lazy takes the minimum
                                                   // lazy class scan
        assert_eq!(f("^[^x]*?x", "aax"), Some(0));
        assert_eq!(f("^[^x]*?x$", "axxb"), None);
    }

    #[test]
    fn captures_and_groups() {
        // first non-empty group wins (lowest index), empty groups skipped
        assert_eq!(f("(x)?(y)z", "yz"), Some(0)); // group 1 absent → group 2 at 0
        assert_eq!(f("a(x*)(y)", "ay"), Some(1)); // group 1 empty at 1, group 2 at 1
        assert_eq!(f("a(b*)(c)", "abc"), Some(1));
        // nested: outer group is index 1
        assert_eq!(f("a((b)c)", "abc"), Some(1));
        // non-capturing groups do not count
        assert_eq!(f("a(?:b)(c)", "abc"), Some(2));
        // leftmost match wins over later alternatives
        assert_eq!(f("(b)|(ab)", "ab"), Some(0));
        assert_eq!(f("(ab)|(b)", "xb"), Some(1));
    }

    #[test]
    fn lookahead() {
        assert_eq!(f("a(?=b)", "ab"), Some(0));
        assert_eq!(f("a(?=b)", "ac"), None);
        assert_eq!(f("a(?!b)", "ac"), Some(0));
        assert_eq!(f("a(?!b)", "ab"), None);
        // the lookahead is zero-width: "a(?=b)c" demands 'b' AND 'c' after
        // 'a' — unsatisfiable
        assert_eq!(f("a(?=b)c", "abc"), None);
        assert_eq!(f("a(?=b)b", "ab"), Some(0));
        assert_eq!(f("x(?=y)z", "xyz"), None);
        assert_eq!(f("x(?=y)y", "xy"), Some(0));
        // lookahead at end of input: nothing follows, so (?=b) fails
        assert_eq!(f("a(?=b)", "a"), None);
        // alternation inside the lookahead
        assert_eq!(f("q(?!a|b)", "qc"), Some(0));
        assert_eq!(f("q(?!a|b)", "qb"), None);
    }

    #[test]
    fn group_quantifiers() {
        // a quantified capture reports its LAST iteration's span
        // (ECMAScript): (ab)+ over "ababab" ends with the 4..6 "ab"
        assert_eq!(f("^(ab)+$", "ababab"), Some(4));
        assert_eq!(f("^(ab)*c$", "c"), Some(0)); // group absent → match start
        assert_eq!(f("^(ab)+c$", "abc"), Some(0));
        assert_eq!(f("^(?:ab)+$", "abab"), Some(0)); // non-capturing
        assert_eq!(f("^(ab)?$", ""), Some(0));
        assert_eq!(f("^x(a.)+", "xatag"), Some(3));
        // lazy group quantifier: as few iterations as possible
        assert_eq!(f("^(a)+?b$", "aab"), Some(1));
    }

    #[test]
    fn empty_pattern_and_empty_matches() {
        assert_eq!(f("", ""), Some(0));
        assert_eq!(f("", "x"), Some(0));
        assert_eq!(f("^$", ""), Some(0));
        assert_eq!(f("x*", "abc"), Some(0)); // empty match at 0
        assert_eq!(f("^()$", ""), Some(0)); // empty group → whole-match start
    }

    // -- loud-fail policy -----------------------------------------------------

    #[test]
    fn unsupported_constructs_fail_loudly() {
        for p in [
            r"\d",
            r"\D",
            r"\w",
            r"\W",
            r"\b",
            r"\B",
            r"\n",
            r"\t",
            r"\r",
            r"\f",
            r"\v",
            r"\0",
            r"\1",
            r"\x41",
            r"\u0041",
            r"\p{L}",
            r"(?<=x)",
            r"(?<!x)",
            r"(?<name>x)",
            r"(?i)x",
            r"(?s:.)",
            r"(?=a)*",
            r"(?!a)+",
            r"^*",
            r"$+",
            r"a**",
            r"a*+",
            r"a{2,1}",
            r"a{1001}",
            r"a{2,1001}",
            r"a{",
            r"a{,2}",
            "a]",
            "a}",
            "(a",
            "a)",
            "[a",
            "[]",
            r"[\d]",
            r"[\s]",
            "[z-a]",
            r"\",
            "(a){2,3}", // {m,n} on a group
            "(a?)+",    // quantified group that can match empty
            "(?:x*)*",  // ditto, nested
        ] {
            assert!(RegexLite::new(p).is_err(), "{p:?} should be rejected");
        }
        // {m,n} on simple atoms IS supported (see quantifiers test)
        assert!(RegexLite::new(r"x{2,3}").is_ok());
        // the error must name the construct and carry the pattern
        let e = bad(r"\d+");
        assert!(e.contains("regex_lite") && e.contains(r"\d"), "{e}");
    }
}
