//! `common/json-schema-to-grammar.cpp` — JSON Schema → GBNF converter, plus the
//! schema model it consumes from `common/json-schema.{h,cpp}` and the `common_json`
//! subset of `common/json.{h,cpp}` it is driven by (1:1 port of llama.cpp
//! `bd4f514db1`, 1028 + 514 + 434 lines).
//!
//! Entry points (mirroring the reference):
//!
//! * [`json_schema_to_grammar`] — `json_schema_to_grammar(common_json, force_gbnf)`
//!   (json-schema-to-grammar.cpp:993), what `llama-cli -j/--json-schema` uses;
//! * [`json_schema_to_grammar_document`] — `json_schema_to_grammar(document)`
//!   (:1008), the already-built schema tree;
//! * [`build_grammar`] — the callback form (:1015) with the `dotall` option, used by
//!   the PEG parser builders;
//! * [`gbnf_format_literal`] (:313).
//!
//! Everything the reference prints is reproduced byte for byte: the rule order is
//! the `std::map` key order (here `BTreeMap`), the union/`-kv`/`-rest` rule names
//! follow `_add_rule`'s duplicate suffixing, and `_visit_primitive` keeps the
//! quirk that a *root* primitive rule is emitted under the name `root` instead of
//! its own (`:828`).
//!
//! Not ported (all unreachable from `bd4f514db1`'s GBNF path):
//!
//! * the `LLAMA_USE_LLGUIDANCE` branch of `json_schema_to_grammar` (:994-1000) —
//!   the workspace builds without llguidance so `force_gbnf` is ignored, exactly
//!   like the reference's `#else` branch;
//! * `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` (:277) — defined but never used;
//! * `common_json`'s container builders/mutators the converter never calls
//!   (`array()`, `push_back`, `insert`, `erase`, `iterator`/`items_view`, the
//!   `std::map`/`std::set`/`std::vector` `common_json_value` ctors, ...). The parser
//!   and `dump()` are ported in full because `_generate_constant_rule` prints
//!   `const`/`enum` values through them.
//!
//! Two places where C++ is UB and this port had to pick a behaviour are marked
//! `UB-C` in the code: the `int64_t` negation/`+1`/`-1` overflows of
//! `build_min_max_int`/`build_integer` (wrapping here, which is what the reference
//! binary does) and the `double` → `int64_t` casts of `get_bound` (x86 `cvttsd2si`,
//! i.e. `INT64_MIN` for NaN and out of range).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::unicode::{cpt_from_utf8, cpt_to_utf8};

// ---------------------------------------------------------------------------
// common_json (common/json.h + common/json.cpp)
// ---------------------------------------------------------------------------

/// `class common_json` — nlohmann's `ordered_json` behind a small interface
/// (json.cpp:14). Objects keep insertion order, numbers keep the distinction
/// between signed/unsigned integers and doubles (`is_number_integer()` is true
/// for both integer kinds, `get<int64_t>()` converts a double by truncation).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Double(f64),
    String(String),
    /// array, in order
    Array(Vec<Json>),
    /// object, keys in insertion order (json.cpp:176 `ordered_json::object()`)
    Object(Vec<(String, Json)>),
}

impl Json {
    // --- predicates (json.cpp:254-264) ---

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Json::Object(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Json::Array(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Json::String(_))
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, Json::Bool(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Json::Int(_) | Json::Uint(_) | Json::Double(_))
    }

    /// `is_number_integer()` is true for `number_integer_t` *and* `number_unsigned_t`
    pub fn is_number_integer(&self) -> bool {
        matches!(self, Json::Int(_) | Json::Uint(_))
    }

    pub fn is_number_float(&self) -> bool {
        matches!(self, Json::Double(_))
    }

    pub fn empty(&self) -> bool {
        match self {
            Json::Array(v) => v.is_empty(),
            Json::Object(v) => v.is_empty(),
            Json::String(s) => s.is_empty(),
            _ => true,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Json::Array(v) => v.len(),
            Json::Object(v) => v.len(),
            _ => 0,
        }
    }

    /// `contains(key)` (json.cpp:266) — false for everything but objects
    pub fn contains(&self, key: &str) -> bool {
        match self {
            Json::Object(v) => v.iter().any(|(k, _)| k == key),
            _ => false,
        }
    }

    /// `at(key)` — `None` stands for the `common_json_error` throw
    pub fn at(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `at(idx)`
    pub fn at_idx(&self, idx: usize) -> Option<&Json> {
        match self {
            Json::Array(v) => v.get(idx),
            _ => None,
        }
    }

    /// `begin()/end()` (json.cpp:348-374) — array by index, object in insertion
    /// order, a plain value once
    pub fn iter(&self) -> std::vec::IntoIter<&Json> {
        match self {
            Json::Array(v) => v.iter().collect::<Vec<_>>().into_iter(),
            Json::Object(v) => v.iter().map(|(_, v)| v).collect::<Vec<_>>().into_iter(),
            other => vec![other].into_iter(),
        }
    }

    /// `items()` (json.cpp:377-392)
    pub fn items(&self) -> Vec<(String, &Json)> {
        match self {
            Json::Object(v) => v.iter().map(|(k, v)| (k.clone(), v)).collect(),
            Json::Array(v) => v
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v))
                .collect(),
            other => vec![(String::new(), other)],
        }
    }

    /// `get<int64_t>()` — nlohmann converts a double by truncation and a `uint64`
    /// that does not fit throws `out_of_range` (json.hpp `get_arithmetic_value`)
    pub fn get_i64(&self) -> Result<i64, String> {
        match self {
            Json::Int(v) => Ok(*v),
            Json::Uint(v) => i64::try_from(*v).map_err(|_| {
                format!("[json.exception.out_of_range.401] number {v} is out of range")
            }),
            // UB-C: `(int64_t) double` is cvttsd2si on x86
            Json::Double(v) => Ok(double_to_i64_x86(*v)),
            _ => Err("[json.exception.type_error.302] type must be number".to_string()),
        }
    }

    /// `get<int>()` — the C++ side reaches it only for `is_number_integer()`
    /// values, where it truncates to the low 32 bits
    pub fn get_int(&self) -> Result<i32, String> {
        Ok(self.get_i64()? as i32)
    }

    /// `get<double>()`
    pub fn get_f64(&self) -> Result<f64, String> {
        match self {
            Json::Int(v) => Ok(*v as f64),
            Json::Uint(v) => Ok(*v as f64),
            Json::Double(v) => Ok(*v),
            _ => Err("[json.exception.type_error.302] type must be number".to_string()),
        }
    }

    /// `get<std::string>()`
    pub fn get_str(&self) -> Result<&str, String> {
        match self {
            Json::String(s) => Ok(s),
            _ => Err("[json.exception.type_error.302] type must be string".to_string()),
        }
    }

    /// mutable field access on an object (used by func_args_not_string's
    /// in-place argument rewriting, chat.cpp:1044-1054); `None` for missing
    /// keys and non-objects
    pub fn at_mut_object_field(&mut self, key: &str) -> Option<&mut Json> {
        if let Json::Object(v) = self {
            v.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    /// `operator[](key) = value` (json.cpp:287, `set`/`assign` :309-315): an
    /// existing key keeps its position, a new key is appended
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Object(v) = self {
            match v.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value,
                None => v.push((key.to_string(), value)),
            }
        }
    }

    // --- parse (json.cpp:201 `common_json::parse`) ---

    /// `common_json::parse(text)` — nlohmann `ordered_json::parse`. The error text
    /// is not nlohmann's, the accepted/rejected inputs are.
    pub fn parse(text: &str) -> Result<Json, String> {
        let bytes = text.as_bytes();
        let mut p = Parser { s: bytes, i: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != bytes.len() {
            return Err(format!(
                "[json.exception.parse_error.101] parse error at byte {}: unexpected trailing input",
                p.i
            ));
        }
        Ok(v)
    }

    // --- dump (json.cpp:339 `common_json::dump`, ensure_ascii = false) ---

    /// `dump()` — nlohmann's compact serialisation. Strings escape control
    /// characters (`\b \f \n \r \t`, otherwise `\u00xx`) plus `"` and `\`, and pass
    /// valid UTF-8 through unescaped (json.hpp `dump_escaped`, ensure_ascii=false).
    /// Doubles use the Grisu2 "shortest round-trip, `%g`-like" form (`dump_float`).
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.dump_into(&mut out);
        out
    }

    fn dump_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(v) => out.push_str(&v.to_string()),
            Json::Uint(v) => out.push_str(&v.to_string()),
            Json::Double(v) => out.push_str(&dump_float(*v)),
            Json::String(s) => dump_string_into(s, out),
            Json::Array(v) => {
                out.push('[');
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.dump_into(out);
                }
                out.push(']');
            }
            Json::Object(v) => {
                out.push('{');
                for (i, (k, item)) in v.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    dump_string_into(k, out);
                    out.push(':');
                    item.dump_into(out);
                }
                out.push('}');
            }
        }
    }
}

/// `dump_escaped` with `ensure_ascii = false` (json.hpp:19076+)
fn dump_string_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '\u{08}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{0C}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) <= 0x1F => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `dump_float` (json.hpp:19520) — nlohmann's Grisu2 `to_chars` + `format_buffer`
/// with `kMinExp = -4`, `kMaxExp = digits10 = 15` for doubles, and `0.0` for zero.
/// Rust's shortest round-trip digits are the same digits Grisu2 produces.
/// nlohmann-style float dump, exposed for the mini-jinja `tojson` filter
/// (value.cpp `value_to_json` uses `std::ostringstream << double`, which is the
/// same shortest-round-trip form nlohmann produces)
pub fn dump_float_pub(x: f64) -> String {
    dump_float(x)
}

fn dump_float(x: f64) -> String {
    if !x.is_finite() {
        return "null".to_string(); // json.hpp:19523
    }
    let negative = x.is_sign_negative();
    let x = if negative { -x } else { x };
    if x == 0.0 {
        return if negative {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    // Rust's shortest round-trip form: "d[.ddd]e±E" → digits + decimal point position
    let s = format!("{x:e}");
    let (mantissa, exp) = s.split_once('e').expect("Rust lower-exp format");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp + 1; // value = digits * 10^(n-k)
    let buf = format_buffer(&digits, n, k);
    if negative {
        format!("-{buf}")
    } else {
        buf
    }
}

/// `dtoa_impl::format_buffer` (json.hpp:18600-18667)
fn format_buffer(digits: &str, n: i32, k: i32) -> String {
    const MIN_EXP: i32 = -4;
    const MAX_EXP: i32 = 15;
    let d: Vec<char> = digits.chars().collect();
    if k <= n && n <= MAX_EXP {
        // digits[000] → make it look like a floating-point number
        let mut out: String = digits.to_string();
        out.push_str(&"0".repeat((n - k) as usize));
        out.push_str(".0");
        return out;
    }
    if 0 < n && n <= MAX_EXP {
        // dig.its
        let mut out: String = d[..n as usize].iter().collect();
        out.push('.');
        out.extend(&d[n as usize..]);
        return out;
    }
    if MIN_EXP < n && n <= 0 {
        // 0.[000]digits
        let mut out = String::from("0.");
        out.push_str(&"0".repeat((-n) as usize));
        out.push_str(digits);
        return out;
    }
    // d[.igits]E+123
    let mut out = if k == 1 {
        digits.to_string()
    } else {
        format!("{}.{}", d[0], d[1..].iter().collect::<String>())
    };
    out.push('e');
    out.push_str(&append_exponent(n - 1));
    out
}

/// `dtoa_impl::append_exponent` (json.hpp:18548) — always at least two digits
fn append_exponent(e: i32) -> String {
    let sign = if e < 0 { '-' } else { '+' };
    format!("{sign}{:02}", e.unsigned_abs())
}

/// `(int64_t) d` on x86-64: `cvttsd2si`, which yields `INT64_MIN` for NaN and for
/// anything outside `[INT64_MIN, INT64_MAX]`. `UB-C`: the C++ cast is UB there,
/// this is what the reference binary produces.
fn double_to_i64_x86(d: f64) -> i64 {
    if d.is_nan() || d >= 9223372036854775808.0 || d < -9223372036854775808.0 {
        i64::MIN
    } else {
        d as i64
    }
}

/// nlohmann's default parser, the subset `common_json::parse` can be handed
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err(&self, msg: &str) -> String {
        format!(
            "[json.exception.parse_error.101] parse error at byte {}: {msg}",
            self.i
        )
    }

    fn value(&mut self) -> Result<Json, String> {
        let c = *self
            .s
            .get(self.i)
            .ok_or_else(|| self.err("unexpected end of input"))?;
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::String(self.string()?)),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'n' => self.literal("null", Json::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn literal(&mut self, text: &str, value: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(text.as_bytes()) {
            self.i += text.len();
            Ok(value)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut out = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.skip_ws();
            out.push(self.value()?);
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(out));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut out: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.skip_ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("object key must be a string"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err(self.err("expected ':'"));
            }
            self.i += 1;
            self.skip_ws();
            let value = self.value()?;
            // duplicate keys keep the first position with the new value
            // (nlohmann `object[key] = value`)
            match out.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => out.push((key, value)),
            }
            self.skip_ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(out));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // '"'
        let mut out = String::new();
        loop {
            let c = *self
                .s
                .get(self.i)
                .ok_or_else(|| self.err("unterminated string"))?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let e = *self
                        .s
                        .get(self.i + 1)
                        .ok_or_else(|| self.err("unterminated escape"))?;
                    match e {
                        b'"' => {
                            out.push('"');
                            self.i += 2;
                        }
                        b'\\' => {
                            out.push('\\');
                            self.i += 2;
                        }
                        b'/' => {
                            out.push('/');
                            self.i += 2;
                        }
                        b'b' => {
                            out.push('\u{08}');
                            self.i += 2;
                        }
                        b'f' => {
                            out.push('\u{0C}');
                            self.i += 2;
                        }
                        b'n' => {
                            out.push('\n');
                            self.i += 2;
                        }
                        b'r' => {
                            out.push('\r');
                            self.i += 2;
                        }
                        b't' => {
                            out.push('\t');
                            self.i += 2;
                        }
                        b'u' => {
                            self.i += 2;
                            let cp = self.hex4()?;
                            let c = if (0xD800..0xDC00).contains(&cp) {
                                // surrogate pair
                                if self.s.get(self.i) != Some(&b'\\')
                                    || self.s.get(self.i + 1) != Some(&b'u')
                                {
                                    return Err(
                                        self.err("surrogate must be followed by another \\u")
                                    );
                                }
                                self.i += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                char::from_u32(0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00))
                                    .ok_or_else(|| self.err("invalid code point"))?
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return Err(self.err("unexpected low surrogate"));
                            } else {
                                char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))?
                            };
                            out.push(c);
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                _ => {
                    // copy the (possibly multi-byte) character
                    let mut off = self.i;
                    let cpt =
                        cpt_from_utf8(self.s, &mut off).map_err(|_| self.err("invalid UTF-8"))?;
                    out.push(char::from_u32(cpt).ok_or_else(|| self.err("invalid code point"))?);
                    self.i = off;
                }
            }
        }
    }

    /// 4 hex digits at the current position
    fn hex4(&mut self) -> Result<u32, String> {
        let hex = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| self.err("unterminated \\u escape"))?;
        let mut v = 0u32;
        for &h in hex {
            let d = (h as char)
                .to_digit(16)
                .ok_or_else(|| self.err("invalid \\u escape"))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let negative = self.s[self.i] == b'-';
        if negative {
            self.i += 1;
        }
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.') {
            is_float = true;
            self.i += 1;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
        }
        let text =
            std::str::from_utf8(&self.s[start..self.i]).map_err(|_| self.err("invalid number"))?;
        if !is_float {
            // nlohmann keeps an integer that fits, a `uint64` when only the
            // positive part fits, and falls back to `double` beyond that
            let magnitude: u128 = text.trim_start_matches('-').parse().unwrap_or(u128::MAX);
            if negative {
                if magnitude == i64::MAX as u128 + 1 {
                    return Ok(Json::Int(i64::MIN));
                }
                if magnitude <= i64::MAX as u128 {
                    return Ok(Json::Int(-(magnitude as i64)));
                }
            } else {
                if magnitude <= i64::MAX as u128 {
                    return Ok(Json::Int(magnitude as i64));
                }
                if magnitude <= u64::MAX as u128 {
                    return Ok(Json::Uint(magnitude as u64));
                }
            }
        }
        let v: f64 = text.parse().map_err(|_| self.err("invalid number"))?;
        Ok(Json::Double(v))
    }
}

// ---------------------------------------------------------------------------
// the schema model (common/json-schema.h)
// ---------------------------------------------------------------------------

/// `common_chat_schema::value_type`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueType {
    Null,
    Boolean,
    Number,
    Integer,
    String,
    Array,
    Object,
}

/// `common_chat_schema::type_set` (json-schema.h:50-76)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TypeSet(u32);

impl TypeSet {
    pub fn new(types: &[ValueType]) -> Self {
        let mut set = TypeSet::default();
        for &t in types {
            set.add(t);
        }
        set
    }

    pub fn all() -> Self {
        TypeSet::new(&[
            ValueType::Null,
            ValueType::Boolean,
            ValueType::Number,
            ValueType::Integer,
            ValueType::String,
            ValueType::Array,
            ValueType::Object,
        ])
    }

    pub fn add(&mut self, t: ValueType) {
        self.0 |= 1 << (t as u32);
    }

    pub fn has(&self, t: ValueType) -> bool {
        self.0 & (1 << (t as u32)) != 0
    }

    pub fn is_only(&self, t: ValueType) -> bool {
        self.0 == 1 << (t as u32)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn union(&mut self, other: TypeSet) {
        self.0 |= other.0;
    }

    pub fn intersect(&mut self, other: TypeSet) {
        self.0 &= other.0;
    }
}

/// `common_chat_schema::string_format`
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StringFormat {
    #[default]
    None,
    /// uuid, uuid1 .. uuid5
    Uuid,
    Date,
    Time,
    DateTime,
}

/// `node_kind` plus its payload; the reference spreads these over the
/// `common_chat_schema_*` subclasses, one enum keeps the same information
#[derive(Clone, Debug)]
pub enum SchemaKind {
    /// `KIND_ANY` — no constraint
    Any,
    /// `{"$ref": "#/..."}`; `target` is filled once every `$ref` is built
    Ref {
        ref_str: String,
        target: Option<NodeId>,
    },
    /// oneOf / anyOf, or a `"type"` array expanded to one alternative per type
    AnyOf,
    /// allOf
    AllOf,
    Const(Json),
    Enum(Vec<Json>),
    Null,
    Boolean,
    Number,
    /// `exclusiveMinimum` / `exclusiveMaximum` are folded into the bounds
    Integer {
        minimum: i64,
        maximum: i64,
    },
    String {
        pattern: String,
        format: StringFormat,
        min_length: i32,
        max_length: i32,
    },
    Array {
        items: NodeId,
        min_items: i32,
        max_items: i32,
    },
    /// `prefixItems` (or `items` given as an array)
    Tuple,
    Object {
        properties: Vec<SchemaProperty>,
        additional_properties: Option<NodeId>,
    },
}

/// index into [`SchemaDocument::nodes`]
pub type NodeId = usize;

/// `common_chat_schema_object::properties` (json-schema.h:176-180)
#[derive(Clone, Debug)]
pub struct SchemaProperty {
    pub name: String,
    pub schema: NodeId,
    pub required: bool,
}

/// one node of the tree; children live in the document arena so that `$ref`
/// cycles stay expressible (the reference uses raw pointers for the same reason)
#[derive(Clone, Debug)]
pub struct SchemaNode {
    pub kind: SchemaKind,
    /// `anyOf` / `oneOf` / `allOf` children, in schema order
    pub children: Vec<NodeId>,
}

impl SchemaNode {
    /// `kind()` + `common_chat_schema::kind_name()`
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            SchemaKind::Any => "any",
            SchemaKind::Ref { .. } => "ref",
            SchemaKind::AnyOf => "anyOf",
            SchemaKind::AllOf => "allOf",
            SchemaKind::Const(_) => "const",
            SchemaKind::Enum(_) => "enum",
            SchemaKind::Null => "null",
            SchemaKind::Boolean => "boolean",
            SchemaKind::Number => "number",
            SchemaKind::Integer { .. } => "integer",
            SchemaKind::String { .. } => "string",
            SchemaKind::Array { .. } => "array",
            SchemaKind::Tuple => "tuple",
            SchemaKind::Object { .. } => "object",
        }
    }
}

/// `common_chat_schema::type_name` (json-schema.cpp:503)
pub fn type_name(t: ValueType) -> &'static str {
    match t {
        ValueType::Null => "null",
        ValueType::Boolean => "boolean",
        ValueType::Number => "number",
        ValueType::Integer => "integer",
        ValueType::String => "string",
        ValueType::Array => "array",
        ValueType::Object => "object",
    }
}

/// `common_chat_schema_document` (json-schema.h:189-192)
#[derive(Clone, Debug)]
pub struct SchemaDocument {
    pub nodes: Vec<SchemaNode>,
    pub root: NodeId,
    /// `$ref` string → the node it resolves to
    pub refs: BTreeMap<String, NodeId>,
}

impl SchemaDocument {
    pub fn node(&self, id: NodeId) -> &SchemaNode {
        &self.nodes[id]
    }

    /// `common_chat_schema::value_types()` (json-schema.cpp:424)
    pub fn value_types(&self, id: NodeId) -> TypeSet {
        let mut visited = HashSet::new();
        self.value_types_impl(id, &mut visited)
    }

    fn value_types_impl(&self, id: NodeId, visited: &mut HashSet<NodeId>) -> TypeSet {
        let node = self.node(id);
        match &node.kind {
            SchemaKind::Any => TypeSet::all(),
            SchemaKind::Null => TypeSet::new(&[ValueType::Null]),
            SchemaKind::Boolean => TypeSet::new(&[ValueType::Boolean]),
            SchemaKind::Number => TypeSet::new(&[ValueType::Number, ValueType::Integer]),
            SchemaKind::Integer { .. } => TypeSet::new(&[ValueType::Integer]),
            SchemaKind::String { .. } => TypeSet::new(&[ValueType::String]),
            SchemaKind::Array { .. } | SchemaKind::Tuple => TypeSet::new(&[ValueType::Array]),
            SchemaKind::Object { .. } => TypeSet::new(&[ValueType::Object]),
            SchemaKind::Const(v) => TypeSet::new(&[json_type(v)]),
            SchemaKind::Enum(values) => {
                let mut types = TypeSet::default();
                for v in values {
                    types.add(json_type(v));
                }
                types
            }
            SchemaKind::Ref { target, .. } => {
                let Some(target) = *target else {
                    return TypeSet::default();
                };
                if !visited.insert(target) {
                    // a cycle contributes no type, to be safe
                    return TypeSet::default();
                }
                let types = self.value_types_impl(target, visited);
                visited.remove(&target);
                types
            }
            SchemaKind::AnyOf => {
                let mut types = TypeSet::default();
                for &child in &node.children {
                    types.union(self.value_types_impl(child, visited));
                }
                types
            }
            SchemaKind::AllOf => {
                let mut types = TypeSet::all();
                for &child in &node.children {
                    types.intersect(self.value_types_impl(child, visited));
                }
                types
            }
        }
    }

    /// `common_chat_schema::may_be_string()` (json-schema.cpp:478)
    pub fn may_be_string(&self, id: NodeId) -> bool {
        let mut visited = HashSet::new();
        self.may_be_string_impl(id, &mut visited)
    }

    fn may_be_string_impl(&self, id: NodeId, visited: &mut HashSet<NodeId>) -> bool {
        let node = self.node(id);
        match &node.kind {
            SchemaKind::String { .. } => true,
            SchemaKind::Const(v) => v.is_string(),
            SchemaKind::Enum(values) => values.iter().any(|v| v.is_string()),
            SchemaKind::Ref { target, .. } => {
                // a cycle is taken as not a string, to be safe
                let Some(target) = *target else { return false };
                if !visited.insert(target) {
                    return false;
                }
                let result = self.may_be_string_impl(target, visited);
                visited.remove(&target);
                result
            }
            SchemaKind::AnyOf => node
                .children
                .iter()
                .any(|&c| self.may_be_string_impl(c, visited)),
            SchemaKind::AllOf => {
                // every child must allow a string, an any child constrains nothing
                let mut any_string = false;
                for &child in &node.children {
                    if matches!(self.node(child).kind, SchemaKind::Any) {
                        continue;
                    }
                    if !self.may_be_string_impl(child, visited) {
                        return false;
                    }
                    any_string = true;
                }
                any_string
            }
            _ => false,
        }
    }
}

/// `json_type()` (json-schema.cpp:346)
fn json_type(value: &Json) -> ValueType {
    match value {
        Json::Null => ValueType::Null,
        Json::Bool(_) => ValueType::Boolean,
        Json::Int(_) | Json::Uint(_) => ValueType::Integer,
        Json::Double(_) => ValueType::Number,
        Json::String(_) => ValueType::String,
        Json::Array(_) => ValueType::Array,
        Json::Object(_) => ValueType::Object,
    }
}

// ---------------------------------------------------------------------------
// json-schema.cpp — building the document out of the JSON text
// ---------------------------------------------------------------------------

/// `common_chat_schema_builder` (json-schema.cpp:12-338)
struct SchemaBuilder {
    /// the document root; the C++ builder keeps a reference to the input tree,
    /// here it owns a copy so that the `"type": [...]` alternatives it
    /// synthesises can be handed to `build_node` by reference
    root: Json,
    nodes: Vec<SchemaNode>,
    /// `refs_`: `None` while a `$ref` target is being built (the cycle guard)
    refs: BTreeMap<String, Option<NodeId>>,
    /// `pending_`: (node, ref) pairs whose `target` is patched after the build
    pending: Vec<(NodeId, String)>,
}

/// `[[noreturn]] fail(path, msg)` (json-schema.cpp:22)
fn schema_fail(path: &str, msg: String) -> String {
    format!("JSON schema error at {path}: {msg}")
}

impl SchemaBuilder {
    fn push(&mut self, kind: SchemaKind) -> NodeId {
        self.nodes.push(SchemaNode {
            kind,
            children: Vec::new(),
        });
        self.nodes.len() - 1
    }

    /// `get_count()` (json-schema.cpp:26)
    fn get_count(schema: &Json, key: &str, path: &str, def: i32) -> Result<i32, String> {
        if !schema.contains(key) {
            return Ok(def);
        }
        let value = schema.at(key).unwrap();
        if !value.is_number_integer() {
            return Err(schema_fail(
                path,
                format!("{key} must be a non-negative integer"),
            ));
        }
        // UB-C: `get<int>()` on a value outside the int range truncates
        let count = value.get_int().unwrap_or(0);
        if count < 0 {
            return Err(schema_fail(
                path,
                format!("{key} must be a non-negative integer"),
            ));
        }
        Ok(count)
    }

    /// `get_bound()` (json-schema.cpp:38) — a fractional bound is rounded inwards
    fn get_bound(schema: &Json, key: &str, path: &str, round_up: bool) -> Result<i64, String> {
        let value = schema.at(key).unwrap();
        if value.is_number_integer() {
            return value.get_i64();
        }
        if !value.is_number() {
            return Err(schema_fail(path, format!("{key} must be a number")));
        }
        let d = value.get_f64().unwrap_or(0.0);
        Ok(double_to_i64_x86(if round_up {
            d.ceil()
        } else {
            d.floor()
        }))
    }

    /// `get_format()` (json-schema.cpp:50)
    fn get_format(schema: &Json, path: &str) -> Result<StringFormat, String> {
        if !schema.contains("format") {
            return Ok(StringFormat::None);
        }
        let value = schema.at("format").unwrap();
        if !value.is_string() {
            return Err(schema_fail(path, "format must be a string".to_string()));
        }
        let format = value.get_str().unwrap_or("");
        Ok(match format {
            "date" => StringFormat::Date,
            "time" => StringFormat::Time,
            "date-time" => StringFormat::DateTime,
            // "uuid" or "uuid1" .. "uuid5"
            f if f == "uuid"
                || (f.len() == 5
                    && f.as_bytes()[..4] == *b"uuid"
                    && (b'1'..=b'5').contains(&f.as_bytes()[4])) =>
            {
                StringFormat::Uuid
            }
            _ => StringFormat::None,
        })
    }

    /// `resolve_ref()` (json-schema.cpp:74) — returns a copy of the target, which
    /// is what `build_node` consumes
    fn resolve_ref(&self, r: &str, path: &str) -> Result<Json, String> {
        let mut target = &self.root;
        // string_split(ref.substr(1), "/") keeps the empty parts
        let rest = if r.is_empty() { r } else { &r[1..] };
        for sel in rest.split('/').skip(1) {
            if target.is_object() && target.contains(sel) {
                target = target.at(sel).unwrap();
            } else if target.is_array() {
                // std::stoull: a non-number is caught and taken as out of range
                let idx = sel.parse::<i64>().unwrap_or(-1);
                if idx < 0 || idx as usize >= target.size() {
                    return Err(schema_fail(
                        path,
                        format!("cannot resolve $ref {r}, {sel} is out of range"),
                    ));
                }
                target = target.at_idx(idx as usize).unwrap();
            } else {
                return Err(schema_fail(
                    path,
                    format!("cannot resolve $ref {r}, {sel} not found"),
                ));
            }
        }
        Ok(target.clone())
    }

    /// `build_ref()` (json-schema.cpp:99)
    fn build_ref(&mut self, value: &Json, path: &str) -> Result<NodeId, String> {
        if !value.is_string() {
            return Err(schema_fail(path, "$ref must be a string".to_string()));
        }
        let r = value.get_str().unwrap_or("").to_string();
        if !r.starts_with("#/") {
            return Err(schema_fail(
                path,
                format!(
                    "unsupported $ref {r}, only references into the same document are supported"
                ),
            ));
        }
        if !self.refs.contains_key(&r) {
            // reserve the key first, so that a cycle back to this $ref stops here
            self.refs.insert(r.clone(), None);
            let target = self.resolve_ref(&r, path)?;
            let built = self.build_node(&target, &r)?;
            self.refs.insert(r.clone(), Some(built));
        }
        let node = self.push(SchemaKind::Ref {
            ref_str: r.clone(),
            target: None,
        });
        self.pending.push((node, r));
        Ok(node)
    }

    /// `build_alternatives<T>()` (json-schema.cpp:117)
    fn build_alternatives(
        &mut self,
        alts: &Json,
        path: &str,
        all_of: bool,
    ) -> Result<NodeId, String> {
        if !alts.is_array() {
            return Err(schema_fail(path, "must be an array of schemas".to_string()));
        }
        if alts.empty() {
            return Err(schema_fail(path, "must not be empty".to_string()));
        }
        let node = self.push(if all_of {
            SchemaKind::AllOf
        } else {
            SchemaKind::AnyOf
        });
        for (i, alt) in alts.iter().enumerate() {
            let child = self.build_node(alt, &format!("{path}/{i}"))?;
            self.nodes[node].children.push(child);
        }
        Ok(node)
    }

    /// `schema.at("allOf")` — every caller checks `contains("allOf")` first
    fn all_of(schema: &Json) -> &Json {
        schema.at("allOf").unwrap()
    }

    /// `build_object()` (json-schema.cpp:133)
    fn build_object(&mut self, schema: &Json, path: &str) -> Result<NodeId, String> {
        let mut required: HashSet<String> = HashSet::new();
        if let Some(req) = schema.at("required") {
            if req.is_array() {
                for name in req.iter() {
                    if let Ok(name) = name.get_str() {
                        required.insert(name.to_string());
                    }
                }
            }
        }

        let mut properties: Vec<SchemaProperty> = Vec::new();
        if let Some(props) = schema.at("properties") {
            if !props.is_object() {
                return Err(schema_fail(
                    path,
                    "properties must be an object".to_string(),
                ));
            }
            for (name, prop) in props.items() {
                let child = self.build_node(prop, &format!("{path}/properties/{name}"))?;
                let is_required = required.contains(&name);
                properties.push(SchemaProperty {
                    name,
                    schema: child,
                    required: is_required,
                });
            }
        }

        let mut additional_properties = None;
        if let Some(additional) = schema.at("additionalProperties") {
            if additional.is_boolean() {
                if matches!(additional, Json::Bool(true)) {
                    additional_properties = Some(self.push(SchemaKind::Any));
                }
            } else if additional.is_object() {
                additional_properties =
                    Some(self.build_node(additional, &format!("{path}/additionalProperties"))?);
            } else {
                return Err(schema_fail(
                    path,
                    "additionalProperties must be a boolean or a schema".to_string(),
                ));
            }
        } else if !schema.contains("properties") {
            // {"type": "object"} on its own accepts any object
            additional_properties = Some(self.push(SchemaKind::Any));
        }

        Ok(self.push(SchemaKind::Object {
            properties,
            additional_properties,
        }))
    }

    /// `build_array()` (json-schema.cpp:174)
    fn build_array(&mut self, schema: &Json, path: &str) -> Result<NodeId, String> {
        let items;
        if schema.contains("items") || schema.contains("prefixItems") {
            // "items" wins when both are present; as in the converter, a schema
            // instead of an array is the item schema
            let key = if schema.contains("items") {
                "items"
            } else {
                "prefixItems"
            };
            let value = schema.at(key).unwrap();
            if value.is_array() {
                let node = self.push(SchemaKind::Tuple);
                for (i, item) in value.iter().enumerate() {
                    let child = self.build_node(item, &format!("{path}/{key}/{i}"))?;
                    self.nodes[node].children.push(child);
                }
                return Ok(node);
            }
            items = self.build_node(value, &format!("{path}/{key}"))?;
        } else {
            items = self.push(SchemaKind::Any);
        }
        let min_items = SchemaBuilder::get_count(schema, "minItems", path, 0)?;
        let max_items = SchemaBuilder::get_count(schema, "maxItems", path, -1)?;
        Ok(self.push(SchemaKind::Array {
            items,
            min_items,
            max_items,
        }))
    }

    /// `build_string()` (json-schema.cpp:197)
    fn build_string(&mut self, schema: &Json, path: &str) -> Result<NodeId, String> {
        let mut pattern = String::new();
        if let Some(value) = schema.at("pattern") {
            if !value.is_string() {
                return Err(schema_fail(path, "pattern must be a string".to_string()));
            }
            pattern = value.get_str().unwrap_or("").to_string();
        }
        let format = SchemaBuilder::get_format(schema, path)?;
        let min_length = SchemaBuilder::get_count(schema, "minLength", path, 0)?;
        let max_length = SchemaBuilder::get_count(schema, "maxLength", path, -1)?;
        Ok(self.push(SchemaKind::String {
            pattern,
            format,
            min_length,
            max_length,
        }))
    }

    /// `build_integer()` (json-schema.cpp:212)
    fn build_integer(&mut self, schema: &Json, path: &str) -> Result<NodeId, String> {
        let mut minimum = i64::MIN;
        let mut maximum = i64::MAX;
        if schema.contains("minimum") {
            minimum = SchemaBuilder::get_bound(schema, "minimum", path, true)?;
        } else if schema.contains("exclusiveMinimum") {
            // UB-C: `+ 1` can overflow, C++ wraps
            minimum =
                SchemaBuilder::get_bound(schema, "exclusiveMinimum", path, false)?.wrapping_add(1);
        }
        if schema.contains("maximum") {
            maximum = SchemaBuilder::get_bound(schema, "maximum", path, false)?;
        } else if schema.contains("exclusiveMaximum") {
            maximum =
                SchemaBuilder::get_bound(schema, "exclusiveMaximum", path, true)?.wrapping_sub(1);
        }
        Ok(self.push(SchemaKind::Integer { minimum, maximum }))
    }

    /// `build_node()` (json-schema.cpp:227)
    fn build_node(&mut self, schema: &Json, path: &str) -> Result<NodeId, String> {
        if !schema.is_object() {
            return Err(schema_fail(path, "schema must be an object".to_string()));
        }
        if schema.contains("$ref") {
            return self.build_ref(schema.at("$ref").unwrap(), path);
        }
        if schema.contains("oneOf") || schema.contains("anyOf") {
            let key = if schema.contains("oneOf") {
                "oneOf"
            } else {
                "anyOf"
            };
            return self.build_alternatives(
                schema.at(key).unwrap(),
                &format!("{path}/{key}"),
                false,
            );
        }

        if let Some(t) = schema.at("type") {
            if t.is_array() {
                // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
                if t.empty() {
                    return Err(schema_fail(path, "type must not be empty".to_string()));
                }
                let node = self.push(SchemaKind::AnyOf);
                for (i, ty) in t.iter().enumerate() {
                    let mut alt = schema.clone();
                    alt.set("type", ty.clone());
                    let child = self.build_node(&alt, &format!("{path}/type/{i}"))?;
                    self.nodes[node].children.push(child);
                }
                return Ok(node);
            }
        }
        if schema.contains("const") {
            let value = schema.at("const").unwrap().clone();
            return Ok(self.push(SchemaKind::Const(value)));
        }
        if schema.contains("enum") {
            let values = schema.at("enum").unwrap();
            if !values.is_array() || values.empty() {
                return Err(schema_fail(
                    path,
                    "enum must be a non-empty array".to_string(),
                ));
            }
            return Ok(self.push(SchemaKind::Enum(values.iter().cloned().collect())));
        }
        let type_value = schema.at("type");
        if let Some(t) = type_value {
            if !t.is_null() && !t.is_string() {
                return Err(schema_fail(
                    path,
                    "type must be a string or an array of strings".to_string(),
                ));
            }
        }

        let type_name = match type_value {
            Some(Json::String(s)) => s.clone(),
            _ => String::new(),
        };
        let has_properties = schema.contains("properties")
            || (schema.contains("additionalProperties")
                && schema.at("additionalProperties") != Some(&Json::Bool(true)));

        if type_name.is_empty() {
            // without a type the structural keywords decide, in the same order as
            // the converter
            if has_properties {
                return self.build_object(schema, path);
            }
            if schema.contains("allOf") {
                let alts = SchemaBuilder::all_of(schema).clone();
                return self.build_alternatives(&alts, &format!("{path}/allOf"), true);
            }
            if schema.contains("items") || schema.contains("prefixItems") {
                return self.build_array(schema, path);
            }
            if schema.contains("pattern")
                || schema.contains("minLength")
                || schema.contains("maxLength")
                || SchemaBuilder::get_format(schema, path)? != StringFormat::None
            {
                return self.build_string(schema, path);
            }
            return Ok(self.push(SchemaKind::Any));
        }
        if type_name == "object" {
            if !has_properties && schema.contains("allOf") {
                let alts = SchemaBuilder::all_of(schema).clone();
                return self.build_alternatives(&alts, &format!("{path}/allOf"), true);
            }
            return self.build_object(schema, path);
        }
        if type_name == "string" {
            if schema.contains("allOf") {
                let alts = SchemaBuilder::all_of(schema).clone();
                return self.build_alternatives(&alts, &format!("{path}/allOf"), true);
            }
            return self.build_string(schema, path);
        }
        if type_name == "array" {
            return self.build_array(schema, path);
        }
        if type_name == "integer" {
            return self.build_integer(schema, path);
        }
        if type_name == "number" {
            return Ok(self.push(SchemaKind::Number));
        }
        if type_name == "boolean" {
            return Ok(self.push(SchemaKind::Boolean));
        }
        if type_name == "null" {
            return Ok(self.push(SchemaKind::Null));
        }
        Err(schema_fail(path, format!("unrecognized type {type_name}")))
    }
}

/// `common_chat_schema_from_json()` (json-schema.cpp:340) — throws
/// `std::runtime_error` when the schema falls outside the supported subset.
pub fn schema_from_json(schema: &Json) -> Result<SchemaDocument, String> {
    let mut builder = SchemaBuilder {
        root: schema.clone(),
        nodes: Vec::new(),
        refs: BTreeMap::new(),
        pending: Vec::new(),
    };
    let root_json = builder.root.clone();
    let root = builder.build_node(&root_json, "#")?;
    let mut doc = SchemaDocument {
        nodes: std::mem::take(&mut builder.nodes),
        root,
        refs: BTreeMap::new(),
    };
    for (r, target) in builder.refs {
        doc.refs
            .insert(r, target.expect("every reserved $ref is built"));
    }
    // `ref->target = doc_.refs.at(ref->ref).get()` (json-schema.cpp:333-335)
    for (node, r) in builder.pending {
        let target = *doc.refs.get(&r).expect("every pending $ref was built");
        doc.nodes[node].kind = match &doc.nodes[node].kind {
            SchemaKind::Ref { ref_str, .. } => SchemaKind::Ref {
                ref_str: ref_str.clone(),
                target: Some(target),
            },
            _ => unreachable!("only ref nodes are pending"),
        };
    }
    Ok(doc)
}

// ---------------------------------------------------------------------------
// json-schema-to-grammar.cpp — the converter
// ---------------------------------------------------------------------------

/// `build_repetition()` (:18)
fn build_repetition(
    item_rule: &str,
    min_items: i32,
    max_items: i32,
    separator_rule: &str,
) -> String {
    let has_max = max_items != i32::MAX;

    if max_items == 0 {
        return String::new();
    }
    if min_items == 0 && max_items == 1 {
        return format!("{item_rule}?");
    }

    if separator_rule.is_empty() {
        if min_items == 1 && !has_max {
            return format!("{item_rule}+");
        }
        if min_items == 0 && !has_max {
            return format!("{item_rule}*");
        }
        return format!(
            "{item_rule}{{{},{}}}",
            min_items,
            if has_max {
                max_items.to_string()
            } else {
                String::new()
            }
        );
    }

    let mut result = format!(
        "{item_rule} {}",
        build_repetition(
            &format!("({separator_rule} {item_rule})"),
            if min_items == 0 { 0 } else { min_items - 1 },
            if has_max { max_items - 1 } else { max_items },
            ""
        )
    );
    if min_items == 0 {
        result = format!("({result})?");
    }
    result
}

/// `build_min_max_int()` (:45)
fn build_min_max_int(
    mut min_value: i64,
    max_value: i64,
    out: &mut String,
    decimals_left: i32,
    top_level: bool,
) {
    let has_min = min_value != i64::MIN;
    let has_max = max_value != i64::MAX;

    fn digit_range(out: &mut String, from: char, to: char) {
        out.push('[');
        if from == to {
            out.push(from);
        } else {
            out.push(from);
            out.push('-');
            out.push(to);
        }
        out.push(']');
    }

    fn more_digits(out: &mut String, min_digits: i32, max_digits: i32) {
        out.push_str("[0-9]");
        if min_digits == max_digits && min_digits == 1 {
            return;
        }
        out.push('{');
        out.push_str(&min_digits.to_string());
        if max_digits != min_digits {
            out.push(',');
            if max_digits != i32::MAX {
                out.push_str(&max_digits.to_string());
            }
        }
        out.push('}');
    }

    fn repeat(c: char, n: usize) -> String {
        std::iter::repeat(c).take(n).collect()
    }

    /// `uniform_range` (:73-127)
    fn uniform_range(out: &mut String, from: &str, to: &str) {
        let from_b = from.as_bytes();
        let to_b = to.as_bytes();
        let mut i = 0usize;
        while i < from_b.len() && i < to_b.len() && from_b[i] == to_b[i] {
            i += 1;
        }
        if i > 0 {
            out.push('"');
            out.push_str(&from[..i]);
            out.push('"');
        }
        if i < from_b.len() && i < to_b.len() {
            if i > 0 {
                out.push(' ');
            }
            let sub_len = from_b.len() - i - 1;
            if sub_len > 0 {
                let from_sub = &from[i + 1..];
                let to_sub = &to[i + 1..];
                let sub_zeros = repeat('0', sub_len);
                let sub_nines = repeat('9', sub_len);

                let mut to_reached = false;
                out.push('(');
                if from_sub == sub_zeros {
                    digit_range(out, from_b[i] as char, (to_b[i] - 1) as char);
                    out.push(' ');
                    more_digits(out, sub_len as i32, sub_len as i32);
                } else {
                    out.push('[');
                    out.push(from_b[i] as char);
                    out.push_str("] ");
                    out.push('(');
                    uniform_range(out, from_sub, &sub_nines);
                    out.push(')');
                    if from_b[i] < to_b[i] - 1 {
                        out.push_str(" | ");
                        if to_sub == sub_nines {
                            digit_range(out, (from_b[i] + 1) as char, to_b[i] as char);
                            to_reached = true;
                        } else {
                            digit_range(out, (from_b[i] + 1) as char, (to_b[i] - 1) as char);
                        }
                        out.push(' ');
                        more_digits(out, sub_len as i32, sub_len as i32);
                    }
                }
                if !to_reached {
                    out.push_str(" | ");
                    digit_range(out, to_b[i] as char, to_b[i] as char);
                    out.push(' ');
                    uniform_range(out, &sub_zeros, to_sub);
                }
                out.push(')');
            } else {
                out.push('[');
                out.push(from_b[i] as char);
                out.push('-');
                out.push(to_b[i] as char);
                out.push(']');
            }
        }
    }

    if has_min && has_max {
        if min_value < 0 && max_value < 0 {
            out.push_str("\"-\" (");
            // UB-C: `-max_value` / `-min_value` (C++ negates, both are outside the
            // sentinels here so no overflow can happen)
            build_min_max_int(
                max_value.wrapping_neg(),
                min_value.wrapping_neg(),
                out,
                decimals_left,
                /* top_level= */ true,
            );
            out.push(')');
            return;
        }

        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(
                0,
                min_value.wrapping_neg(),
                out,
                decimals_left,
                /* top_level= */ true,
            );
            out.push_str(") | ");
            min_value = 0;
        }

        let mut min_s = min_value.to_string();
        let max_s = max_value.to_string();
        let min_digits = min_s.len();
        let max_digits = max_s.len();

        for digits in min_digits..max_digits {
            uniform_range(out, &min_s, &repeat('9', digits));
            min_s = format!("1{}", repeat('0', digits));
            out.push_str(" | ");
        }
        uniform_range(out, &min_s, &max_s);
        return;
    }

    let less_decimals = std::cmp::max(decimals_left - 1, 1);

    if has_min {
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(
                i64::MIN,
                min_value.wrapping_neg(),
                out,
                decimals_left,
                /* top_level= */ false,
            );
            out.push_str(") | [0] | [1-9] ");
            more_digits(out, 0, decimals_left - 1);
        } else if min_value == 0 {
            if top_level {
                out.push_str("[0] | [1-9] ");
                more_digits(out, 0, less_decimals);
            } else {
                more_digits(out, 1, decimals_left);
            }
        } else if min_value <= 9 {
            let c = (b'0' + min_value as u8) as char;
            let range_start = if top_level { '1' } else { '0' };
            if c > range_start {
                digit_range(out, range_start, (c as u8 - 1) as char);
                out.push(' ');
                more_digits(out, 1, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, '9');
            out.push(' ');
            more_digits(out, 0, less_decimals);
        } else {
            let min_s = min_value.to_string();
            let len = min_s.len();
            let c = min_s.as_bytes()[0] as char;

            if c > '1' {
                digit_range(
                    out,
                    if top_level { '1' } else { '0' },
                    (c as u8 - 1) as char,
                );
                out.push(' ');
                more_digits(out, len as i32, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, c);
            out.push_str(" (");
            let rest: i64 = min_s[1..].parse().expect("digits of a positive integer");
            build_min_max_int(
                rest,
                i64::MAX,
                out,
                less_decimals,
                /* top_level= */ false,
            );
            out.push(')');
            if c < '9' {
                out.push_str(" | ");
                digit_range(out, (c as u8 + 1) as char, '9');
                out.push(' ');
                more_digits(out, len as i32 - 1, less_decimals);
            }
        }
        return;
    }

    if has_max {
        if max_value >= 0 {
            if top_level {
                out.push_str("\"-\" [1-9] ");
                more_digits(out, 0, less_decimals);
                out.push_str(" | ");
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out.push_str("\"-\" (");
            // UB-C: negating a negative maximum, without overflow as above
            build_min_max_int(
                max_value.wrapping_neg(),
                i64::MAX,
                out,
                decimals_left,
                /* top_level= */ false,
            );
            out.push(')');
        }
        return;
    }

    panic!("At least one of min_value or max_value must be set"); // :226
}

/// `SPACE_RULE` (:229)
const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

/// `struct BuiltinRule` (:231)
struct BuiltinRule {
    content: &'static str,
    deps: &'static [&'static str],
}

/// `PRIMITIVE_RULES` (:236)
const PRIMITIVE_RULES: &[(&str, BuiltinRule)] = &[
    ("boolean", BuiltinRule { content: "(\"true\" | \"false\")", deps: &[] }),
    ("decimal-part", BuiltinRule { content: "[0-9]{1,16}", deps: &[] }),
    ("integral-part", BuiltinRule { content: "[0] | [1-9] [0-9]{0,15}", deps: &[] }),
    (
        "number",
        BuiltinRule {
            content: "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?",
            deps: &["integral-part", "decimal-part"],
        },
    ),
    ("integer", BuiltinRule { content: "(\"-\"? integral-part)", deps: &["integral-part"] }),
    (
        "value",
        BuiltinRule {
            content: "object | array | string | number | boolean | null",
            deps: &["object", "array", "string", "number", "boolean", "null"],
        },
    ),
    (
        "object",
        BuiltinRule {
            content: "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
            deps: &["string", "value"],
        },
    ),
    ("array", BuiltinRule { content: "\"[\" space ( value (\",\" space value)* )? space \"]\"", deps: &["value"] }),
    (
        "uuid",
        BuiltinRule {
            content: "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"",
            deps: &[],
        },
    ),
    (
        "char",
        BuiltinRule {
            content: "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})",
            deps: &[],
        },
    ),
    ("string", BuiltinRule { content: "\"\\\"\" char* \"\\\"\"", deps: &["char"] }),
    ("null", BuiltinRule { content: "\"null\"", deps: &[] }),
];

/// `STRING_FORMAT_RULES` (:251)
const STRING_FORMAT_RULES: &[(&str, BuiltinRule)] = &[
    (
        "date",
        BuiltinRule {
            content: "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )",
            deps: &[],
        },
    ),
    (
        "time",
        BuiltinRule {
            content: "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )",
            deps: &[],
        },
    ),
    ("date-time", BuiltinRule { content: "date \"T\" time", deps: &["date", "time"] }),
    ("date-string", BuiltinRule { content: "\"\\\"\" date \"\\\"\"", deps: &["date"] }),
    ("time-string", BuiltinRule { content: "\"\\\"\" time \"\\\"\"", deps: &["time"] }),
    ("date-time-string", BuiltinRule { content: "\"\\\"\" date-time \"\\\"\"", deps: &["date-time"] }),
];

fn builtin_lookup(name: &str) -> Option<&'static BuiltinRule> {
    PRIMITIVE_RULES
        .iter()
        .chain(STRING_FORMAT_RULES.iter())
        .find(|(n, _)| *n == name)
        .map(|(_, r)| r)
}

/// `is_reserved_name()` (:260)
fn is_reserved_name(name: &str) -> bool {
    name == "root"
        || PRIMITIVE_RULES.iter().any(|(n, _)| *n == name)
        || STRING_FORMAT_RULES.iter().any(|(n, _)| *n == name)
}

/// `INVALID_RULE_CHARS_RE` / `nonalphanumeric_regex` replacement (:275, :685):
/// every maximal run of characters outside `[a-zA-Z0-9-]` becomes a single `-`
/// (`std::regex_replace` over bytes).
fn replace_invalid_rule_chars(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for &b in name.as_bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' {
            in_run = false;
            out.push(b as char);
        } else if !in_run {
            in_run = true;
            out.push('-');
        }
    }
    out
}

/// `format_literal()` (:305) with `GRAMMAR_LITERAL_ESCAPE_RE`/`_ESCAPES` (:276-280):
/// the RE matches `\r`, `\n`, `"` and `\` (the map also spells `-` and `]`, which
/// only the never-used range RE would have matched).
fn format_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() + 2);
    out.push('"');
    for c in literal.chars() {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `gbnf_format_literal()` (:313)
pub fn gbnf_format_literal(literal: &str) -> String {
    format_literal(literal)
}

/// `gbnf_escape_length()` (:315)
fn gbnf_escape_length(pattern: &[u8], pos: usize) -> usize {
    if pos + 1 >= pattern.len() || pattern[pos] != b'\\' {
        return 0;
    }
    let n_hex = match pattern[pos + 1] {
        b'x' => 2,
        b'u' => 4,
        b'U' => 8,
        // keep in sync with parse_char() in src/llama-grammar.cpp
        b't' | b'r' | b'n' | b'\\' | b'"' | b'[' | b']' | b'-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > pattern.len() {
        return 0;
    }
    for &h in &pattern[pos + 2..pos + 2 + n_hex] {
        if !(h.is_ascii_digit() || (b'a'..=b'f').contains(&h) || (b'A'..=b'F').contains(&h)) {
            return 0;
        }
    }
    2 + n_hex
}

/// `MAX_PATTERN_DEPTH` (:282)
const MAX_PATTERN_DEPTH: i32 = 100;

/// `NON_LITERAL_SET` (:284)
fn is_non_literal(c: u8) -> bool {
    matches!(
        c,
        b'|' | b'.' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'*' | b'+' | b'?' | b'^' | b'$'
    )
}

/// `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS` (:285)
fn escaped_in_regexps_but_not_in_literals(c: u8) -> bool {
    matches!(
        c,
        b'^' | b'$' | b'.' | b'[' | b']' | b'(' | b')' | b'|' | b'{' | b'}' | b'*' | b'+' | b'?'
    )
}

/// the two exception kinds `_visit_pattern` catches (:380-388)
#[derive(Debug)]
enum PatternError {
    /// `unsupported_pattern` — a valid regex with no grammar equivalent
    Unsupported(String),
    /// `invalid_pattern` — not a valid regex
    Invalid(String),
}

/// `literal_or_rule` (:417)
#[derive(Clone, Debug)]
enum LiteralOrRule {
    Literal(String),
    Rule(String),
}

impl LiteralOrRule {
    /// `to_rule()` (:418)
    fn to_rule(&self) -> String {
        match self {
            LiteralOrRule::Literal(s) => format!("\"{s}\""),
            LiteralOrRule::Rule(s) => s.clone(),
        }
    }

    fn is_literal(&self) -> bool {
        matches!(self, LiteralOrRule::Literal(_))
    }
}

/// `_pattern_to_rule`'s mutable scan state (`i`, `paren_depth`, `sub_rule_ids`)
struct PatternState {
    i: usize,
    paren_depth: i32,
    sub_rule_ids: HashMap<String, String>,
}

/// `common_trie` (common/trie.h:12-50) — codepoint trie, children in codepoint order
struct Trie {
    children: Vec<BTreeMap<u32, usize>>,
    pattern: Vec<i32>,
    n_patterns: i32,
}

impl Trie {
    fn new(words: &[String]) -> Self {
        let mut t = Trie {
            children: vec![BTreeMap::new()],
            pattern: vec![-1],
            n_patterns: 0,
        };
        for w in words {
            t.insert(w);
        }
        t
    }

    /// `common_trie::insert(const std::string &)` (trie.cpp:43): invalid UTF-8
    /// stops the walk, duplicates keep the first pattern index
    fn insert(&mut self, word: &str) -> i32 {
        let mut symbols = Vec::new();
        let bytes = word.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            match cpt_from_utf8(bytes, &mut pos) {
                Ok(cpt) => symbols.push(cpt),
                Err(()) => break,
            }
        }
        let mut current = 0usize;
        for ch in symbols {
            let existing = self.children[current].get(&ch).copied();
            current = match existing {
                Some(child) => child,
                None => {
                    let child = self.children.len();
                    self.children.push(BTreeMap::new());
                    self.pattern.push(-1);
                    self.children[current].insert(ch, child);
                    child
                }
            };
        }
        if self.pattern[current] < 0 {
            self.pattern[current] = self.n_patterns;
            self.n_patterns += 1;
        }
        self.pattern[current]
    }
}

/// `common_chat_schema_converter` (:342)
struct Converter<'a> {
    doc: &'a SchemaDocument,
    dotall: bool,
    /// `std::map<std::string, std::string> _rules` — printed in key order
    rules: BTreeMap<String, String>,
    refs_being_resolved: HashSet<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl<'a> Converter<'a> {
    /// `explicit common_chat_schema_converter(bool dotall)` (:816)
    fn new(doc: &'a SchemaDocument, dotall: bool) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Converter {
            doc,
            dotall,
            rules,
            refs_being_resolved: HashSet::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// `_add_rule()` (:356)
    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = replace_invalid_rule_chars(name);
        if self.rules.get(&esc_name).map(|r| r == rule).unwrap_or(true) {
            self.rules.insert(esc_name.clone(), rule.to_string());
            return esc_name;
        }
        let mut i = 0;
        loop {
            let key = format!("{esc_name}{i}");
            match self.rules.get(&key) {
                Some(existing) if existing != rule => i += 1,
                _ => {
                    self.rules.insert(key.clone(), rule.to_string());
                    return key;
                }
            }
        }
    }

    /// `_generate_union_rule()` (:371)
    fn generate_union_rule(&mut self, name: &str, alt_schemas: &[NodeId]) -> String {
        let mut rules = Vec::with_capacity(alt_schemas.len());
        for (i, &alt) in alt_schemas.iter().enumerate() {
            let suffix = if name.is_empty() { "alternative-" } else { "-" };
            rules.push(self.visit(alt, &format!("{name}{suffix}{i}")));
        }
        rules.join(" | ")
    }

    /// `_visit_pattern()` (:390)
    fn visit_pattern(&mut self, pattern: &str, name: &str) -> String {
        let rules_snapshot = self.rules.clone();
        match self.pattern_to_rule(pattern, name) {
            Ok(rule) => rule,
            Err(PatternError::Unsupported(err)) => {
                // revert rules
                self.rules = rules_snapshot;
                self.warnings.push(format!(
                    "pattern {pattern} is not supported ({err}), accepting any string"
                ));
                let primitive = self.add_primitive("string", builtin_lookup("string").unwrap());
                self.add_rule(name, &primitive)
            }
            Err(PatternError::Invalid(err)) => {
                self.rules = rules_snapshot;
                self.errors
                    .push(format!("Invalid pattern {pattern}: {err}"));
                String::new()
            }
        }
    }

    /// `_pattern_to_rule()` (:406)
    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> Result<String, PatternError> {
        let bytes = pattern.as_bytes();
        if bytes.len() < 2 || bytes[0] != b'^' || *bytes.last().unwrap() != b'$' {
            return Err(PatternError::Unsupported(
                "not anchored with '^' and '$'".to_string(),
            ));
        }
        let sub_pattern = &bytes[1..bytes.len() - 1];
        let mut st = PatternState {
            i: 0,
            paren_depth: 0,
            sub_rule_ids: HashMap::new(),
        };

        let rule = self.transform(sub_pattern, &mut st, name)?.to_rule();
        if st.paren_depth != 0 {
            return Err(PatternError::Invalid("unbalanced parentheses".to_string()));
        }

        Ok(self.add_rule(name, &format!("\"\\\"\" ({rule}) \"\\\"\"")))
    }

    /// the `get_dot` lambda of `transform` (:426)
    fn get_dot(&mut self) -> String {
        let rule = if self.dotall {
            "[\\U00000000-\\U0010FFFF]"
        } else {
            "[^\\x0A\\x0D]"
        };
        self.add_rule("dot", rule)
    }

    /// the `transform` lambda (:423-623)
    fn transform(
        &mut self,
        sub_pattern: &[u8],
        st: &mut PatternState,
        name: &str,
    ) -> Result<LiteralOrRule, PatternError> {
        let length = sub_pattern.len();
        let mut seq: Vec<LiteralOrRule> = Vec::new();

        while st.i < length {
            let c = sub_pattern[st.i];
            if c == b'.' {
                seq.push(LiteralOrRule::Rule(self.get_dot()));
                st.i += 1;
            } else if c == b'(' {
                st.i += 1;
                if st.i < length && sub_pattern[st.i] == b'?' {
                    if st.i + 1 < length && sub_pattern[st.i + 1] == b':' {
                        // skip "?:" for a non-capturing group, treat as a regular group
                        st.i += 2;
                    } else {
                        // lookaround, named group, inline flags, ...
                        return Err(PatternError::Unsupported(
                            "unsupported group syntax".to_string(),
                        ));
                    }
                }
                st.paren_depth += 1;
                if st.paren_depth > MAX_PATTERN_DEPTH {
                    return Err(PatternError::Unsupported(
                        "pattern nesting too deep".to_string(),
                    ));
                }
                let inner = self.transform(sub_pattern, st, name)?;
                seq.push(LiteralOrRule::Rule(format!("({})", inner.to_rule())));
            } else if c == b')' {
                st.i += 1;
                if st.paren_depth == 0 {
                    return Err(PatternError::Invalid("unbalanced parentheses".to_string()));
                }
                st.paren_depth -= 1;
                return Ok(join_seq(seq));
            } else if c == b'^' || c == b'$' {
                return Err(PatternError::Unsupported(
                    "anchor inside the pattern".to_string(),
                ));
            } else if c == b'[' {
                let mut square_brackets = String::from("[");
                st.i += 1;
                while st.i < length && sub_pattern[st.i] != b']' {
                    if sub_pattern[st.i] == b'\\' {
                        let escape_length = gbnf_escape_length(sub_pattern, st.i);
                        if escape_length == 0 {
                            return Err(PatternError::Unsupported(format!(
                                "unsupported escape in character class: {}",
                                String::from_utf8_lossy(&sub_pattern[st.i..(st.i + 2).min(length)])
                            )));
                        }
                        square_brackets.push_str(&String::from_utf8_lossy(
                            &sub_pattern[st.i..st.i + escape_length],
                        ));
                        st.i += escape_length;
                    } else {
                        square_brackets.push(sub_pattern[st.i] as char);
                        st.i += 1;
                    }
                }
                if st.i >= length {
                    return Err(PatternError::Invalid(
                        "unterminated character class".to_string(),
                    ));
                }
                square_brackets.push(']');
                st.i += 1;
                seq.push(LiteralOrRule::Rule(square_brackets));
            } else if c == b'|' {
                seq.push(LiteralOrRule::Rule("|".to_string()));
                st.i += 1;
            } else if c == b'*' || c == b'+' || c == b'?' {
                if seq.is_empty() {
                    return Err(PatternError::Invalid("nothing to repeat".to_string()));
                }
                let item = seq.last().unwrap().to_rule();
                *seq.last_mut().unwrap() = LiteralOrRule::Rule(format!("{item}{}", c as char));
                st.i += 1;
            } else if c == b'{' {
                let mut curly_brackets = String::from("{");
                st.i += 1;
                while st.i < length && sub_pattern[st.i] != b'}' {
                    curly_brackets.push(sub_pattern[st.i] as char);
                    st.i += 1;
                }
                if st.i >= length {
                    return Err(PatternError::Unsupported(
                        "unterminated curly brackets".to_string(),
                    ));
                }
                curly_brackets.push('}');
                st.i += 1;
                let nums: Vec<&str> = curly_brackets[1..curly_brackets.len() - 1]
                    .split(',')
                    .collect();
                let mut min_times: i32 = 0;
                let mut max_times: i32 = i32::MAX;
                if nums.len() != 1 && nums.len() != 2 {
                    return Err(PatternError::Unsupported(
                        "wrong number of values in curly brackets".to_string(),
                    ));
                }
                let bad_number =
                    || PatternError::Unsupported("invalid number in curly brackets".to_string());
                if nums.len() == 1 {
                    min_times = stoi_prefix(nums[0]).map_err(|_| bad_number())?;
                    max_times = min_times;
                } else {
                    if !nums[0].is_empty() {
                        min_times = stoi_prefix(nums[0]).map_err(|_| bad_number())?;
                    }
                    if !nums[1].is_empty() {
                        max_times = stoi_prefix(nums[1]).map_err(|_| bad_number())?;
                    }
                }
                if seq.is_empty() {
                    return Err(PatternError::Invalid("nothing to repeat".to_string()));
                }
                let last = seq.last().unwrap().clone();
                let mut sub = match &last {
                    LiteralOrRule::Literal(s) | LiteralOrRule::Rule(s) => s.clone(),
                };
                let sub_is_literal = last.is_literal();

                if !sub_is_literal {
                    // `sub_rule_ids[sub]` inserts the empty entry before the size
                    // is read, so the first sub-rule is `-1` (:569-572)
                    if !st.sub_rule_ids.contains_key(&sub) {
                        st.sub_rule_ids.insert(sub.clone(), String::new());
                    }
                    let existing = st.sub_rule_ids.get(&sub).unwrap().clone();
                    let sub_id = if existing.is_empty() {
                        let id = self.add_rule(&format!("{name}-{}", st.sub_rule_ids.len()), &sub);
                        st.sub_rule_ids.insert(sub.clone(), id.clone());
                        id
                    } else {
                        existing
                    };
                    sub = sub_id;
                }
                *seq.last_mut().unwrap() = LiteralOrRule::Rule(build_repetition(
                    &if sub_is_literal {
                        format!("\"{sub}\"")
                    } else {
                        sub
                    },
                    min_times,
                    max_times,
                    "",
                ));
            } else {
                let mut literal = String::new();
                while st.i < length {
                    if sub_pattern[st.i] == b'\\' {
                        if st.i == length - 1 {
                            return Err(PatternError::Invalid("trailing backslash".to_string()));
                        }
                        let next = sub_pattern[st.i + 1];
                        if escaped_in_regexps_but_not_in_literals(next) {
                            st.i += 1;
                            literal.push(sub_pattern[st.i] as char);
                            st.i += 1;
                        } else {
                            let escape_length = gbnf_escape_length(sub_pattern, st.i);
                            if escape_length == 0 {
                                return Err(PatternError::Unsupported(format!(
                                    "unsupported escape: {}",
                                    String::from_utf8_lossy(
                                        &sub_pattern[st.i..(st.i + 2).min(length)]
                                    )
                                )));
                            }
                            literal.push_str(&String::from_utf8_lossy(
                                &sub_pattern[st.i..st.i + escape_length],
                            ));
                            st.i += escape_length;
                        }
                    } else if sub_pattern[st.i] == b'"' {
                        literal.push_str("\\\"");
                        st.i += 1;
                    } else if !is_non_literal(sub_pattern[st.i])
                        && (st.i == length - 1
                            || literal.is_empty()
                            || sub_pattern[st.i + 1] == b'.'
                            || !is_non_literal(sub_pattern[st.i + 1]))
                    {
                        literal.push(sub_pattern[st.i] as char);
                        st.i += 1;
                    } else {
                        break;
                    }
                }
                if literal.is_empty() {
                    // nothing was consumed, ex. a stray ']' or '}'
                    return Err(PatternError::Unsupported(format!(
                        "unsupported character: {}",
                        c as char
                    )));
                }
                seq.push(LiteralOrRule::Literal(literal));
            }
        }
        Ok(join_seq(seq))
    }

    /// `_not_strings()` (:641)
    fn not_strings(&mut self, strings: &[String]) -> String {
        let trie = Trie::new(strings);

        let char_rule = self.add_primitive("char", builtin_lookup("char").unwrap());
        let mut out = String::from("[\"] ( ");
        visit_trie(&mut out, &trie, 0, &char_rule);
        out.push_str(" )");
        if trie.pattern[0] < 0 {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }

    /// `_resolve_ref()` (:682)
    fn resolve_ref(&mut self, node: NodeId) -> String {
        let (ref_str, target) = match &self.doc.node(node).kind {
            SchemaKind::Ref { ref_str, target } => (ref_str.clone(), *target),
            _ => unreachable!("resolve_ref on a non-ref node"),
        };
        let ref_fragment = match ref_str.find('#') {
            Some(it) => ref_str[it + 1..].to_string(),
            None => ref_str.clone(),
        };
        let ref_name = format!("ref{}", replace_invalid_rule_chars(&ref_fragment));
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(&ref_str) {
            let Some(target) = target else {
                self.errors.push(format!("Unresolved $ref {ref_str}"));
                return String::new();
            };
            self.refs_being_resolved.insert(ref_str.clone());
            let name = self.visit(target, &ref_name);
            self.refs_being_resolved.remove(&ref_str);
            return name;
        }
        ref_name
    }

    /// `_build_object_rule()` (:699)
    fn build_object_rule(
        &mut self,
        properties: &[(String, NodeId)],
        required: &HashSet<String>,
        name: &str,
        additional_properties: Option<NodeId>,
    ) -> String {
        let mut required_props: Vec<String> = Vec::new();
        let mut optional_props: Vec<String> = Vec::new();
        let mut prop_kv_rule_names: HashMap<String, String> = HashMap::new();
        let mut prop_names: Vec<String> = Vec::new();
        for (prop_name, prop_schema) in properties {
            let prop_rule_name = self.visit(
                *prop_schema,
                &format!(
                    "{name}{}{prop_name}",
                    if name.is_empty() { "" } else { "-" }
                ),
            );
            prop_kv_rule_names.insert(
                prop_name.clone(),
                self.add_rule(
                    &format!(
                        "{name}{}{prop_name}-kv",
                        if name.is_empty() { "" } else { "-" }
                    ),
                    &format!(
                        "{} space \":\" space {prop_rule_name}",
                        format_literal(&Json::String(prop_name.clone()).dump())
                    ),
                ),
            );
            if required.contains(prop_name) {
                required_props.push(prop_name.clone());
            } else {
                optional_props.push(prop_name.clone());
            }
            prop_names.push(prop_name.clone());
        }
        if let Some(additional) = additional_properties {
            let sub_name = format!("{name}{}additional", if name.is_empty() { "" } else { "-" });
            let value_rule = if !matches!(self.doc.node(additional).kind, SchemaKind::Any) {
                self.visit(additional, &format!("{sub_name}-value"))
            } else {
                self.add_primitive("value", builtin_lookup("value").unwrap())
            };

            let key_rule = if prop_names.is_empty() {
                self.add_primitive("string", builtin_lookup("string").unwrap())
            } else {
                let not = self.not_strings(&prop_names);
                self.add_rule(&format!("{sub_name}-k"), &not)
            };
            let kv_rule = self.add_rule(
                &format!("{sub_name}-kv"),
                &format!("{key_rule} \":\" space {value_rule}"),
            );
            prop_kv_rule_names.insert("*".to_string(), kv_rule);
            optional_props.push("*".to_string());
        }

        if required_props.is_empty() && optional_props.is_empty() {
            return "\"{\" space \"}\"".to_string();
        }

        let mut rule = String::from("\"{\" space ");
        for (i, prop) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(&prop_kv_rule_names[prop]);
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }

            // `get_recursive_refs` (:757-777), one entry per starting property
            for i in 0..optional_props.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                rule.push_str(&self.get_recursive_refs(
                    &optional_props[i..],
                    false,
                    name,
                    &prop_kv_rule_names,
                ));
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }

        rule.push_str(" space \"}\"");

        rule
    }

    /// the `get_recursive_refs` lambda of `_build_object_rule` (:757)
    fn get_recursive_refs(
        &mut self,
        ks: &[String],
        first_is_optional: bool,
        name: &str,
        prop_kv_rule_names: &HashMap<String, String>,
    ) -> String {
        let mut res = String::new();
        if ks.is_empty() {
            return res;
        }
        let k = &ks[0];
        let kv_rule_name = &prop_kv_rule_names[k];
        let comma_ref = format!("( \",\" space {kv_rule_name} )");
        if first_is_optional {
            res = format!("{comma_ref}{}", if k == "*" { "*" } else { "?" });
        } else {
            res = format!(
                "{kv_rule_name}{}",
                if k == "*" {
                    format!(" {comma_ref}*")
                } else {
                    String::new()
                }
            );
        }
        if ks.len() > 1 {
            let rest = self.get_recursive_refs(&ks[1..], true, name, prop_kv_rule_names);
            let rest_rule = self.add_rule(
                &format!("{name}{}{k}-rest", if name.is_empty() { "" } else { "-" }),
                &rest,
            );
            res.push(' ');
            res.push_str(&rest_rule);
        }
        res
    }

    /// `_add_primitive()` (:796)
    fn add_primitive(&mut self, name: &str, rule: &'static BuiltinRule) -> String {
        let n = self.add_rule(name, rule.content);
        for dep in rule.deps {
            let Some(dep_rule) = builtin_lookup(dep) else {
                self.errors.push(format!("Rule {dep} not known"));
                continue;
            };
            if !self.rules.contains_key(*dep) {
                self.add_primitive(dep, dep_rule);
            }
        }
        n
    }

    /// `add_schema()` (:820)
    fn add_schema(&mut self, name: &str, node: NodeId) -> String {
        self.visit(node, name)
    }

    /// `_visit_primitive()` (:828) — a root-level primitive rule is emitted under
    /// the name `root` rather than its own
    fn visit_primitive(&mut self, rule_name: &str, type_: &str) -> String {
        let name = if rule_name == "root" { "root" } else { type_ };
        self.add_primitive(name, builtin_lookup(type_).unwrap())
    }

    /// `_visit_all_of()` (:832)
    fn visit_all_of(&mut self, node: NodeId, name: &str, rule_name: &str) -> String {
        let mut required: HashSet<String> = HashSet::new();
        let mut properties: Vec<(String, NodeId)> = Vec::new();
        let mut enum_values: BTreeMap<String, usize> = BTreeMap::new();

        /// the `add_component` lambda (:836-853)
        fn add_component(
            doc: &SchemaDocument,
            id: NodeId,
            is_required: bool,
            required: &mut HashSet<String>,
            properties: &mut Vec<(String, NodeId)>,
            enum_values: &mut BTreeMap<String, usize>,
        ) {
            match &doc.node(id).kind {
                SchemaKind::Ref {
                    target: Some(target),
                    ..
                } => {
                    add_component(doc, *target, is_required, required, properties, enum_values);
                }
                SchemaKind::Object {
                    properties: props, ..
                } => {
                    for prop in props {
                        properties.push((prop.name.clone(), prop.schema));
                        if is_required {
                            required.insert(prop.name.clone());
                        }
                    }
                }
                SchemaKind::Enum(values) => {
                    for v in values {
                        *enum_values.entry(generate_constant_rule(v)).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
        }

        let children = self.doc.node(node).children.clone();
        for &child in &children {
            if matches!(self.doc.node(child).kind, SchemaKind::AnyOf) {
                for alt in self.doc.node(child).children.clone() {
                    add_component(
                        self.doc,
                        alt,
                        false,
                        &mut required,
                        &mut properties,
                        &mut enum_values,
                    );
                }
            } else {
                add_component(
                    self.doc,
                    child,
                    true,
                    &mut required,
                    &mut properties,
                    &mut enum_values,
                );
            }
        }
        if !enum_values.is_empty() {
            let mut enum_intersection: Vec<String> = Vec::new();
            for (p, count) in &enum_values {
                if *count == children.len() {
                    enum_intersection.push(p.clone());
                }
            }
            if !enum_intersection.is_empty() {
                return self.add_rule(rule_name, &format!("({})", enum_intersection.join(" | ")));
            }
        }
        let rule = self.build_object_rule(&properties, &required, name, None);
        self.add_rule(rule_name, &rule)
    }

    /// `visit()` (:877)
    fn visit(&mut self, node: NodeId, name: &str) -> String {
        let rule_name = if is_reserved_name(name) {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };
        let sub_name = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };

        let doc = self.doc; // a copy of the reference, so `&mut self` stays usable
        match &doc.node(node).kind {
            SchemaKind::Ref { .. } => {
                let resolved = self.resolve_ref(node);
                self.add_rule(&rule_name, &resolved)
            }
            SchemaKind::AnyOf => {
                let union = self.generate_union_rule(name, &doc.node(node).children);
                self.add_rule(&rule_name, &union)
            }
            SchemaKind::AllOf => self.visit_all_of(node, name, &rule_name),
            SchemaKind::Const(value) => {
                let rule = generate_constant_rule(value);
                self.add_rule(&rule_name, &rule)
            }
            SchemaKind::Enum(values) => {
                let enum_values: Vec<String> = values.iter().map(generate_constant_rule).collect();
                self.add_rule(&rule_name, &format!("({})", enum_values.join(" | ")))
            }
            SchemaKind::Object {
                properties,
                additional_properties,
            } => {
                if properties.is_empty()
                    && additional_properties
                        .map(|a| matches!(doc.node(a).kind, SchemaKind::Any))
                        .unwrap_or(false)
                {
                    let primitive = self.add_primitive("object", builtin_lookup("object").unwrap());
                    return self.add_rule(&rule_name, &primitive);
                }
                let props: Vec<(String, NodeId)> = properties
                    .iter()
                    .map(|p| (p.name.clone(), p.schema))
                    .collect();
                let required: HashSet<String> = properties
                    .iter()
                    .filter(|p| p.required)
                    .map(|p| p.name.clone())
                    .collect();
                let additional = *additional_properties;
                let rule = self.build_object_rule(&props, &required, name, additional);
                self.add_rule(&rule_name, &rule)
            }
            SchemaKind::Tuple => {
                let items = doc.node(node).children.clone();
                let mut rule = String::from("\"[\" space ");
                for (i, &item) in items.iter().enumerate() {
                    if i > 0 {
                        rule.push_str(" \",\" space ");
                    }
                    let item_rule = self.visit(item, &format!("{sub_name}tuple-{i}"));
                    rule.push_str(&item_rule);
                }
                rule.push_str(" space \"]\"");
                self.add_rule(&rule_name, &rule)
            }
            SchemaKind::Array {
                items,
                min_items,
                max_items,
            } => {
                let (items, min_items, max_items) = (*items, *min_items, *max_items);
                if matches!(doc.node(items).kind, SchemaKind::Any)
                    && min_items == 0
                    && max_items < 0
                {
                    return self.visit_primitive(&rule_name, "array");
                }
                let item_rule_name = self.visit(items, &format!("{sub_name}item"));
                let max_items = if max_items < 0 { i32::MAX } else { max_items };
                let rule = format!(
                    "\"[\" space {} space \"]\"",
                    build_repetition(&item_rule_name, min_items, max_items, "\",\" space")
                );
                self.add_rule(&rule_name, &rule)
            }
            SchemaKind::String {
                pattern,
                format,
                min_length,
                max_length,
            } => {
                let (pattern, format, min_length, max_length) =
                    (pattern.clone(), *format, *min_length, *max_length);
                if !pattern.is_empty() {
                    return self.visit_pattern(&pattern, &rule_name);
                }
                if format == StringFormat::Uuid {
                    return self.visit_primitive(&rule_name, "uuid");
                }
                if format != StringFormat::None {
                    let prim_name = format!(
                        "{}-string",
                        match format {
                            StringFormat::Date => "date",
                            StringFormat::Time => "time",
                            _ => "date-time",
                        }
                    );
                    let primitive =
                        self.add_primitive(&prim_name, builtin_lookup(&prim_name).unwrap());
                    return self.add_rule(&rule_name, &primitive);
                }
                if min_length > 0 || max_length >= 0 {
                    let char_rule = self.add_primitive("char", builtin_lookup("char").unwrap());
                    let max_len = if max_length < 0 { i32::MAX } else { max_length };
                    let rule = format!(
                        "\"\\\"\" {} \"\\\"\"",
                        build_repetition(&char_rule, min_length, max_len, "")
                    );
                    return self.add_rule(&rule_name, &rule);
                }
                self.visit_primitive(&rule_name, "string")
            }
            SchemaKind::Integer { minimum, maximum } => {
                let (minimum, maximum) = (*minimum, *maximum);
                if minimum == i64::MIN && maximum == i64::MAX {
                    return self.visit_primitive(&rule_name, "integer");
                }
                let mut out = String::from("(");
                build_min_max_int(minimum, maximum, &mut out, 16, true);
                out.push(')');
                self.add_rule(&rule_name, &out)
            }
            SchemaKind::Number => self.visit_primitive(&rule_name, "number"),
            SchemaKind::Boolean => self.visit_primitive(&rule_name, "boolean"),
            SchemaKind::Null => self.visit_primitive(&rule_name, "null"),
            SchemaKind::Any => {
                let primitive = self.add_primitive("value", builtin_lookup("value").unwrap());
                self.add_rule(&rule_name, &primitive)
            }
        }
    }

    /// `check_errors()` (:975)
    fn check_errors(&self) -> Result<(), String> {
        if !self.errors.is_empty() {
            return Err(format!(
                "JSON schema conversion failed:\n{}",
                self.errors.join("\n")
            ));
        }
        if !self.warnings.is_empty() {
            eprintln!(
                "WARNING: JSON schema conversion was incomplete: {}",
                self.warnings.join("; ")
            );
        }
        Ok(())
    }

    /// `format_grammar()` (:984)
    fn format_grammar(&self) -> String {
        let mut ss = String::new();
        for (name, rule) in &self.rules {
            ss.push_str(name);
            ss.push_str(" ::= ");
            ss.push_str(rule);
            ss.push('\n');
        }
        ss
    }
}

/// `_generate_constant_rule()` (:824)
fn generate_constant_rule(value: &Json) -> String {
    format_literal(&value.dump())
}

/// the `join_seq` lambda of `transform` (:437-467) — merges consecutive literals
fn join_seq(seq: Vec<LiteralOrRule>) -> LiteralOrRule {
    let mut ret: Vec<LiteralOrRule> = Vec::new();
    let mut literal = String::new();
    for item in seq {
        match item {
            LiteralOrRule::Literal(s) => literal.push_str(&s),
            rule => {
                if !literal.is_empty() {
                    ret.push(LiteralOrRule::Literal(std::mem::take(&mut literal)));
                }
                ret.push(rule);
            }
        }
    }
    if !literal.is_empty() {
        ret.push(LiteralOrRule::Literal(literal));
    }
    let results: Vec<String> = ret.iter().map(|item| item.to_rule()).collect();
    LiteralOrRule::Rule(results.join(" "))
}

/// the recursive `visit` of `_not_strings` (:647-672)
fn visit_trie(out: &mut String, trie: &Trie, idx: usize, char_rule: &str) {
    let mut rejects = String::new();
    let mut first = true;
    for (&cpt, &child) in &trie.children[idx] {
        let c = cpt_to_utf8(cpt);
        rejects.push_str(&c);
        if first {
            first = false;
        } else {
            out.push_str(" | ");
        }
        out.push('[');
        out.push_str(&c);
        out.push(']');
        if !trie.children[child].is_empty() {
            out.push_str(" (");
            visit_trie(out, trie, child, char_rule);
            out.push(')');
        } else {
            out.push(' ');
            out.push_str(char_rule);
            out.push('+');
        }
    }
    if !trie.children[idx].is_empty() {
        out.push_str(" | [^\"");
        out.push_str(&rejects);
        out.push_str("] ");
        out.push_str(char_rule);
        out.push('*');
    }
}

/// `std::stoi` on the front of the text: leading whitespace, optional sign, at
/// least one digit; everything after the digits is ignored, a value outside the
/// `int` range is an error (`out_of_range`, which the caller treats the same as
/// `invalid_argument`, both are `logic_error`s).
fn stoi_prefix(s: &str) -> Result<i32, ()> {
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() && b[i].is_ascii_whitespace() && b[i] < 0x80 {
        i += 1;
    }
    let negative = if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        let neg = b[i] == b'-';
        i += 1;
        neg
    } else {
        false
    };
    let digits_start = i;
    let mut value: i128 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        value = value * 10 + (b[i] - b'0') as i128;
        // saturate like strtol's ERANGE, the range check below then fails
        value = value.min(i32::MAX as i128 + 1);
        i += 1;
    }
    if i == digits_start {
        return Err(());
    }
    let signed = if negative { -value } else { value };
    if signed < i32::MIN as i128 || signed > i32::MAX as i128 {
        return Err(());
    }
    Ok(signed as i32)
}

// ---------------------------------------------------------------------------
// entry points (json-schema-to-grammar.cpp:993-1028)
// ---------------------------------------------------------------------------

/// `json_schema_to_grammar(const common_json & schema, bool force_gbnf)` (:993).
///
/// `force_gbnf` is ignored: this workspace has no `LLAMA_USE_LLGUIDANCE`, exactly
/// like the reference's `#else` branch (:998-1000).
pub fn json_schema_to_grammar(schema: &Json, force_gbnf: bool) -> Result<String, String> {
    let _ = force_gbnf;
    let doc =
        schema_from_json(schema).map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    json_schema_to_grammar_document(&doc)
}

/// `json_schema_to_grammar(const common_chat_schema_document &)` (:1008)
pub fn json_schema_to_grammar_document(schema: &SchemaDocument) -> Result<String, String> {
    let mut converter = Converter::new(schema, false);
    converter.visit(schema.root, "");
    converter.check_errors()?;
    Ok(converter.format_grammar())
}

/// `struct common_grammar_options` (json-schema-to-grammar.h:17)
#[derive(Clone, Copy, Debug, Default)]
pub struct GrammarOptions {
    pub dotall: bool,
}

/// the `common_grammar_builder` callbacks (json-schema-to-grammar.h:12-15)
pub struct GrammarBuilder<'a, 'd> {
    converter: &'a mut Converter<'d>,
}

impl GrammarBuilder<'_, '_> {
    /// `.add_rule`
    pub fn add_rule(&mut self, name: &str, rule: &str) -> String {
        self.converter.add_rule(name, rule)
    }

    /// `.add_schema` — `name == "root"` becomes the unnamed root of the grammar
    /// (:1022)
    pub fn add_schema(&mut self, name: &str, node: NodeId) -> String {
        let name = if name == "root" { "" } else { name };
        self.converter.add_schema(name, node)
    }
}

/// `build_grammar()` (:1015) — `cb` gets the builder for one schema document
pub fn build_grammar<F>(
    schema: &SchemaDocument,
    options: GrammarOptions,
    cb: F,
) -> Result<String, String>
where
    F: FnOnce(&mut GrammarBuilder<'_, '_>),
{
    let mut converter = Converter::new(schema, options.dotall);
    {
        let mut builder = GrammarBuilder {
            converter: &mut converter,
        };
        cb(&mut builder);
    }
    converter.check_errors()?;
    Ok(converter.format_grammar())
}

// ---------------------------------------------------------------------------
// unit tests (the byte-for-byte parity with the reference lives in
// crates/llama/tests/json_schema_parity.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parse_and_dump_round_trip() {
        let v =
            Json::parse(r#"{"a": [1, 2.5, -3, true, null, "x\ny"], "b": {"c": 1e100}}"#).unwrap();
        assert_eq!(
            v.dump(),
            r#"{"a":[1,2.5,-3,true,null,"x\ny"],"b":{"c":1e+100}}"#
        );
    }

    #[test]
    fn json_object_keeps_insertion_order() {
        let v = Json::parse(r#"{"z":1,"a":2,"m":{"q":1,"b":2}}"#).unwrap();
        assert_eq!(v.dump(), r#"{"z":1,"a":2,"m":{"q":1,"b":2}}"#);
    }

    #[test]
    fn dump_float_matches_nlohmann_grisu2() {
        assert_eq!(dump_float(0.0), "0.0");
        assert_eq!(dump_float(-0.0), "-0.0");
        assert_eq!(dump_float(1.5), "1.5");
        assert_eq!(dump_float(100000.0), "100000.0");
        assert_eq!(dump_float(1e100), "1e+100");
        assert_eq!(dump_float(1e-7), "1e-07");
        assert_eq!(dump_float(3.14), "3.14");
    }

    #[test]
    fn schema_builder_failures_match_the_reference_messages() {
        let err = schema_from_json(&Json::parse(r#"{"type": "kaboom"}"#).unwrap()).unwrap_err();
        assert_eq!(err, "JSON schema error at #: unrecognized type kaboom");
        let err = schema_from_json(&Json::parse("true").unwrap()).unwrap_err();
        assert_eq!(err, "JSON schema error at #: schema must be an object");
    }

    #[test]
    fn simple_and_range_grammars() {
        let g = json_schema_to_grammar(
            &Json::parse(r#"{"type":"integer","minimum":0}"#).unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(
            g,
            "root ::= ([0] | [1-9] [0-9]{0,15})\nspace ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n"
        );
    }

    #[test]
    fn recursive_refs_build_once() {
        let schema = Json::parse(
            r##"{"$ref": "#/$defs/node", "$defs": {"node": {"type": "object",
                "properties": {"next": {"$ref": "#/$defs/node"}, "leaf": {}},
                "additionalProperties": false}}}"##,
        )
        .unwrap();
        let doc = schema_from_json(&schema).unwrap();
        // the document entry point and the JSON entry point must agree (:1554-1566)
        assert_eq!(
            json_schema_to_grammar_document(&doc).unwrap(),
            json_schema_to_grammar(&schema, true).unwrap()
        );
        assert!(json_schema_to_grammar(&schema, true)
            .unwrap()
            .contains("ref-defs-node ::= "));
    }
}
