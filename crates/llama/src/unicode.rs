//! unicode.rs — port of llama.cpp `src/unicode.cpp` (owner: agent B).
//!
//! Provides UTF-8 helpers, per-codepoint category flags (built from
//! `unicode_data.rs`), the GPT-2 byte↔unicode mapping, simple lowercase
//! mapping, NFD base-codepoint normalization and the pre-tokenizer regex
//! splitter used by the BPE tokenizer.
//!
//! The "generic" regex fallback replicates llama.cpp's strategy exactly:
//! custom hand-written splitters for the hot patterns (gpt2/llama3/qwen2/
//! qwen35/kimi-k2/afmoe/newlines), a byte-collapsed regex path when the
//! pattern contains `\p{...}` categories, and a codepoint regex path
//! otherwise (with non-ASCII whitespace folded to U+000B so `\s` matches
//! it, exactly like the C++ `std::wregex` workaround). `fancy-regex` stands
//! in for `std::regex` (ECMAScript) — it supports the lookaheads that
//! several pre-tokenizer patterns use (`\s+(?!\S)` etc.).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use fancy_regex::Regex;

use crate::unicode_data::{
    UNICODE_MAP_LOWERCASE, UNICODE_MAP_UPPERCASE, UNICODE_RANGES_FLAGS, UNICODE_RANGES_NFD,
    UNICODE_SET_WHITESPACE,
};

pub const MAX_CODEPOINTS: usize = 0x110000;

/// `struct unicode_cpt_flags` — bit flags for one codepoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CptFlags(pub u16);

impl CptFlags {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002; // regex: \p{N}
    pub const LETTER: u16 = 0x0004; // regex: \p{L}
    pub const SEPARATOR: u16 = 0x0008; // regex: \p{Z}
    pub const ACCENT_MARK: u16 = 0x0010; // regex: \p{M}
    pub const PUNCTUATION: u16 = 0x0020; // regex: \p{P}
    pub const SYMBOL: u16 = 0x0040; // regex: \p{S}
    pub const CONTROL: u16 = 0x0080; // regex: \p{C}
    pub const MASK_CATEGORIES: u16 = 0x00FF;
    pub const WHITESPACE: u16 = 0x0100; // regex: \s
    pub const LOWERCASE: u16 = 0x0200;
    pub const UPPERCASE: u16 = 0x0400;
    pub const NFD: u16 = 0x0800;

    #[inline]
    pub fn is_undefined(self) -> bool {
        self.0 & Self::UNDEFINED != 0
    }
    #[inline]
    pub fn is_number(self) -> bool {
        self.0 & Self::NUMBER != 0
    }
    #[inline]
    pub fn is_letter(self) -> bool {
        self.0 & Self::LETTER != 0
    }
    #[inline]
    pub fn is_separator(self) -> bool {
        self.0 & Self::SEPARATOR != 0
    }
    #[inline]
    pub fn is_accent_mark(self) -> bool {
        self.0 & Self::ACCENT_MARK != 0
    }
    #[inline]
    pub fn is_punctuation(self) -> bool {
        self.0 & Self::PUNCTUATION != 0
    }
    #[inline]
    pub fn is_symbol(self) -> bool {
        self.0 & Self::SYMBOL != 0
    }
    #[inline]
    pub fn is_control(self) -> bool {
        self.0 & Self::CONTROL != 0
    }
    #[inline]
    pub fn is_whitespace(self) -> bool {
        self.0 & Self::WHITESPACE != 0
    }
    #[inline]
    pub fn is_lowercase(self) -> bool {
        self.0 & Self::LOWERCASE != 0
    }
    #[inline]
    pub fn is_uppercase(self) -> bool {
        self.0 & Self::UPPERCASE != 0
    }
    #[inline]
    pub fn is_nfd(self) -> bool {
        self.0 & Self::NFD != 0
    }
    /// `category_flag()` — category bits only.
    #[inline]
    pub fn category_flag(self) -> u16 {
        self.0 & Self::MASK_CATEGORIES
    }
    /// `as_uint()`.
    #[inline]
    pub fn as_uint(self) -> u16 {
        self.0
    }
}

/// `unicode_len_utf8(src)` — expected byte length from the first byte.
#[inline]
pub fn len_utf8(src: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(src >> 4) as usize]
}

/// `unicode_cpt_from_utf8(utf8, offset)` — decode one codepoint at `*offset`,
/// advancing it. `Err` mirrors the C++ `std::invalid_argument` throw.
pub fn cpt_from_utf8(bytes: &[u8], offset: &mut usize) -> Result<u32, ()> {
    debug_assert!(*offset < bytes.len());
    let b0 = bytes[*offset];
    if b0 & 0x80 == 0 {
        let result = b0 as u32;
        *offset += 1;
        return Ok(result);
    }
    if b0 & 0x40 == 0 {
        return Err(());
    }
    if b0 & 0x20 == 0 {
        if *offset + 1 >= bytes.len() || (bytes[*offset + 1] & 0xc0) != 0x80 {
            return Err(());
        }
        let result = ((b0 as u32 & 0x1f) << 6) | (bytes[*offset + 1] as u32 & 0x3f);
        *offset += 2;
        return Ok(result);
    }
    if b0 & 0x10 == 0 {
        if *offset + 2 >= bytes.len()
            || (bytes[*offset + 1] & 0xc0) != 0x80
            || (bytes[*offset + 2] & 0xc0) != 0x80
        {
            return Err(());
        }
        let result = ((b0 as u32 & 0x0f) << 12)
            | ((bytes[*offset + 1] as u32 & 0x3f) << 6)
            | (bytes[*offset + 2] as u32 & 0x3f);
        *offset += 3;
        return Ok(result);
    }
    if b0 & 0x08 == 0 {
        if *offset + 3 >= bytes.len()
            || (bytes[*offset + 1] & 0xc0) != 0x80
            || (bytes[*offset + 2] & 0xc0) != 0x80
            || (bytes[*offset + 3] & 0xc0) != 0x80
        {
            return Err(());
        }
        let result = ((b0 as u32 & 0x07) << 18)
            | ((bytes[*offset + 1] as u32 & 0x3f) << 12)
            | ((bytes[*offset + 2] as u32 & 0x3f) << 6)
            | (bytes[*offset + 3] as u32 & 0x3f);
        *offset += 4;
        return Ok(result);
    }
    Err(())
}

/// `unicode_cpt_to_utf8(cpt)` — encode a codepoint. Invalid codepoints
/// (unreachable with GGUF-loaded text) map to U+FFFD instead of throwing.
pub fn cpt_to_utf8(cpt: u32) -> String {
    char::from_u32(cpt).unwrap_or('\u{FFFD}').to_string()
}

/// single-`char` variant of `unicode_cpt_to_utf8` (all in-house uses are
/// single codepoints).
pub fn cpt_to_char(cpt: u32) -> char {
    char::from_u32(cpt).unwrap_or('\u{FFFD}')
}

/// `unicode_cpts_from_utf8(utf8)` — invalid sequences become U+FFFD, one per
/// skipped byte (mirrors the C++ catch-and-skip loop).
pub fn cpts_from_utf8(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(bytes.len());
    let mut offset = 0usize;
    while offset < bytes.len() {
        match cpt_from_utf8(bytes, &mut offset) {
            Ok(cpt) => result.push(cpt),
            Err(()) => {
                offset += 1;
                result.push(0xFFFD);
            }
        }
    }
    result
}

/// `unicode_cpts_normalize_nfd(cpts)` — map codepoints to their NFD base
/// codepoint (accent stripping for the WPM/BERT normalizer).
pub fn cpts_normalize_nfd(cpts: &[u32]) -> Vec<u32> {
    cpts.iter()
        .map(|&cpt| {
            // std::upper_bound(unicode_ranges_nfd, cpt) - 1
            let idx = UNICODE_RANGES_NFD
                .partition_point(|r| r.0 <= cpt)
                .saturating_sub(1);
            let r = &UNICODE_RANGES_NFD[idx];
            if r.0 <= cpt && cpt <= r.1 {
                r.2
            } else {
                cpt
            }
        })
        .collect()
}

fn cpt_flags_table() -> &'static [u16; MAX_CODEPOINTS] {
    static TABLE: OnceLock<Box<[u16; MAX_CODEPOINTS]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0u16; MAX_CODEPOINTS];
        debug_assert_eq!(UNICODE_RANGES_FLAGS[0].0, 0);
        debug_assert_eq!(
            UNICODE_RANGES_FLAGS[UNICODE_RANGES_FLAGS.len() - 1].0,
            MAX_CODEPOINTS as u32
        );
        for i in 1..UNICODE_RANGES_FLAGS.len() {
            let (range_ini, flags) = UNICODE_RANGES_FLAGS[i - 1];
            let (range_end, _) = UNICODE_RANGES_FLAGS[i];
            for cpt in range_ini..range_end {
                t[cpt as usize] = flags;
            }
        }
        for &cpt in UNICODE_SET_WHITESPACE {
            t[cpt as usize] |= CptFlags::WHITESPACE;
        }
        for &(_, to) in UNICODE_MAP_LOWERCASE {
            t[to as usize] |= CptFlags::LOWERCASE;
        }
        for &(_, to) in UNICODE_MAP_UPPERCASE {
            t[to as usize] |= CptFlags::UPPERCASE;
        }
        for &(_, _, nfd) in UNICODE_RANGES_NFD {
            t[nfd as usize] |= CptFlags::NFD;
        }
        t.into_boxed_slice().try_into().ok().expect("table size")
    })
}

/// `unicode_cpt_flags_from_cpt(cpt)`.
pub fn cpt_flags_from_cpt(cpt: u32) -> CptFlags {
    let t = cpt_flags_table();
    if (cpt as usize) < t.len() {
        CptFlags(t[cpt as usize])
    } else {
        CptFlags(CptFlags::UNDEFINED)
    }
}

/// `unicode_cpt_flags_from_utf8(utf8)` — flags of the first codepoint.
pub fn cpt_flags_from_utf8(utf8: &str) -> CptFlags {
    if utf8.is_empty() {
        return CptFlags(CptFlags::UNDEFINED);
    }
    let mut offset = 0usize;
    match cpt_from_utf8(utf8.as_bytes(), &mut offset) {
        Ok(cpt) => cpt_flags_from_cpt(cpt),
        Err(()) => CptFlags(CptFlags::UNDEFINED),
    }
}

fn byte_to_utf8_table() -> &'static [String; 256] {
    static TABLE: OnceLock<Box<[String; 256]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map: Vec<Option<String>> = vec![None; 256];
        for ch in 0x21..=0x7Eu32 {
            map[ch as usize] = Some(cpt_to_utf8(ch));
        }
        for ch in 0xA1..=0xACu32 {
            map[ch as usize] = Some(cpt_to_utf8(ch));
        }
        for ch in 0xAE..=0xFFu32 {
            map[ch as usize] = Some(cpt_to_utf8(ch));
        }
        let mut n = 0u32;
        let mut out: Vec<String> = Vec::with_capacity(256);
        for ch in 0..256usize {
            match &map[ch] {
                Some(s) => out.push(s.clone()),
                None => {
                    out.push(cpt_to_utf8(256 + n));
                    n += 1;
                }
            }
        }
        out.into_boxed_slice().try_into().ok().expect("256 entries")
    })
}

fn utf8_to_byte_table() -> &'static HashMap<&'static str, u8> {
    static TABLE: OnceLock<HashMap<&'static str, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let fwd = byte_to_utf8_table();
        let mut map = HashMap::with_capacity(256);
        for (ch, s) in fwd.iter().enumerate() {
            map.insert(s.as_str(), ch as u8);
        }
        map
    })
}

/// `unicode_byte_to_utf8(byte)` — GPT-2 byte-level encoding of one byte.
pub fn byte_to_utf8(byte: u8) -> &'static str {
    &byte_to_utf8_table()[byte as usize]
}

/// `unicode_utf8_to_byte(utf8)` — inverse mapping; `None` mirrors the C++
/// `std::out_of_range` throw from `map.at()`.
pub fn utf8_to_byte(utf8: &str) -> Option<u8> {
    utf8_to_byte_table().get(utf8).copied()
}

/// `unicode_tolower(cpt)` — simple lowercase mapping (binary search).
pub fn tolower(cpt: u32) -> u32 {
    match UNICODE_MAP_LOWERCASE.binary_search_by(|p| p.0.cmp(&cpt)) {
        Ok(idx) => UNICODE_MAP_LOWERCASE[idx].1,
        Err(_) => cpt,
    }
}

/// `unicode_cpt_is_han(cpt)`.
pub fn cpt_is_han(cpt: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cpt)
        || (0x3400..=0x4DBF).contains(&cpt)
        || (0x20000..=0x2A6DF).contains(&cpt)
        || (0x2A700..=0x2B73F).contains(&cpt)
        || (0x2B740..=0x2B81F).contains(&cpt)
        || (0x2B820..=0x2CEAF).contains(&cpt)
        || (0x2CEB0..=0x2EBEF).contains(&cpt)
        || (0xF900..=0xFAFF).contains(&cpt)
        || (0x2F800..=0x2FA1F).contains(&cpt)
}

// ---------------------------------------------------------------------------
// custom (hand-written) regex splitters — direct ports from unicode.cpp
// ---------------------------------------------------------------------------

const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;

/// Shared per-segment splitter state (the `_get_cpt`/`_get_flags`/`_add_token`
/// closures of the C++ implementations).
struct Splitter<'a> {
    cpts: &'a [u32],
    offset_ini: usize,
    offset_end: usize,
    prev_end: usize,
    out: Vec<usize>,
}

impl<'a> Splitter<'a> {
    #[inline]
    fn cpt(&self, pos: usize) -> u32 {
        if self.offset_ini <= pos && pos < self.offset_end {
            self.cpts[pos]
        } else {
            OUT_OF_RANGE
        }
    }
    #[inline]
    fn flags(&self, pos: usize) -> CptFlags {
        if self.offset_ini <= pos && pos < self.offset_end {
            cpt_flags_from_cpt(self.cpts[pos])
        } else {
            CptFlags(0)
        }
    }
    #[inline]
    fn add_token(&mut self, end: usize) -> usize {
        debug_assert!(self.prev_end <= end && end <= self.offset_end);
        let len = end - self.prev_end;
        if len > 0 {
            self.out.push(len);
        }
        self.prev_end = end;
        len
    }
}

/// case-insensitive contraction check `(?i:'s|'t|'re|'ve|'m|'ll|'d)` shared by
/// the llama3/qwen2/qwen35 splitters. Returns match length (0 = no match).
fn match_contraction_ci(sp: &Splitter, pos: usize) -> usize {
    if sp.cpt(pos) == '\'' as u32 && pos + 1 < sp.offset_end {
        let cpt_next = tolower(sp.cpt(pos + 1));
        if cpt_next == 's' as u32
            || cpt_next == 't' as u32
            || cpt_next == 'm' as u32
            || cpt_next == 'd' as u32
        {
            return 2;
        }
        if pos + 2 < sp.offset_end {
            let cpt_next_next = tolower(sp.cpt(pos + 2));
            if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
            {
                return 3;
            }
        }
    }
    0
}

/// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn regex_split_custom_gpt2(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if cpt == '\'' as u32 && pos + 1 < offset_end {
                let cpt_next = sp.cpt(pos + 1);
                if cpt_next == 's' as u32
                    || cpt_next == 't' as u32
                    || cpt_next == 'm' as u32
                    || cpt_next == 'd' as u32
                {
                    pos += sp.add_token(pos + 2);
                    continue;
                }
                if pos + 2 < offset_end {
                    let cpt_next_next = sp.cpt(pos + 2);
                    if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                        || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
                    {
                        pos += sp.add_token(pos + 3);
                        continue;
                    }
                }
            }

            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            // regex: <space>?\p{L}+
            if flags2.is_letter() {
                pos += usize::from(cpt == ' ' as u32);
                while flags2.is_letter() {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                sp.add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if flags2.is_number() {
                pos += usize::from(cpt == ' ' as u32);
                while flags2.is_number() {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                sp.add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                && flags2.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                num_whitespaces += 1;
            }

            // regex: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// LLAMA3 system regex:
/// "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
fn regex_split_custom_llama3(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            let contraction = match_contraction_ci(&sp, pos);
            if contraction > 0 {
                pos += sp.add_token(pos + contraction);
                continue;
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number()) {
                if flags.is_letter() || sp.flags(pos + 1).is_letter() {
                    pos += 1;
                    while sp.flags(pos).is_letter() {
                        pos += 1;
                    }
                    sp.add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if flags.is_number() {
                let mut ini = pos;
                while sp.flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        sp.add_token(pos);
                        ini = pos;
                    }
                }
                sp.add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                && flags.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                let mut cpt2 = sp.cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            let mut last_end_r_or_n = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = sp.cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // regex: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// K2-Horizon system regex (462524043, unicode.cpp:473-622):
/// "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
///
/// The existing llama3 splitter with a single rule widened: a letter run
/// also takes marks, ZWNJ and ZWJ (the generic fallback cannot serve this
/// regex — U+200C/U+200D collapse to the 0xD0 fallback byte there, so every
/// ZWNJ/ZWJ would end a letter run).
fn regex_split_custom_k2_horizon(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        // K2-Horizon: letter runs are (?:\p{L}|\p{M}|\u200C|\u200D)+
        let is_k2_letter = |sp: &Splitter, pos: usize| -> bool {
            let c = sp.cpt(pos);
            if c == 0x200C || c == 0x200D {
                return true;
            }
            let f = sp.flags(pos);
            f.is_letter() || f.is_accent_mark()
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) — see `contraction` above
            if contraction_at(&sp, pos, offset_end) > 0 {
                pos += sp.add_token(pos + contraction_at(&sp, pos, offset_end));
                continue;
            }

            // regex: [^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number()) {
                if is_k2_letter(&sp, pos) || is_k2_letter(&sp, pos + 1) {
                    pos += 1;
                    while is_k2_letter(&sp, pos) {
                        pos += 1;
                    }
                    sp.add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if flags.is_number() {
                let mut ini = pos;
                while sp.flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        sp.add_token(pos);
                        ini = pos;
                    }
                }
                sp.add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                && flags.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                let mut cpt2 = sp.cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            let mut last_end_r_or_n = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = sp.cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // regex: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// the K2-Horizon contraction check with long-s folding — `match_contraction_ci`
/// plus the 0x017F → 's' fold (unicode.cpp:520-525)
fn contraction_at(sp: &Splitter, pos: usize, offset_end: usize) -> usize {
    if sp.cpt(pos) == '\'' as u32 && pos + 1 < offset_end {
        let mut cpt_next = tolower(sp.cpt(pos + 1));
        if cpt_next == 0x017F {
            cpt_next = 's' as u32; // Unicode case-folding of long s
        }
        if cpt_next == 's' as u32
            || cpt_next == 't' as u32
            || cpt_next == 'm' as u32
            || cpt_next == 'd' as u32
        {
            return 2;
        }
        if pos + 2 < offset_end {
            let cpt_next_next = tolower(sp.cpt(pos + 2));
            if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
            {
                return 3;
            }
        }
    }
    0
}

/// Qwen2 system regex:
/// "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
fn regex_split_custom_qwen2(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            let contraction = match_contraction_ci(&sp, pos);
            if contraction > 0 {
                pos += sp.add_token(pos + contraction);
                continue;
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number()) {
                if flags.is_letter() || sp.flags(pos + 1).is_letter() {
                    pos += 1;
                    while sp.flags(pos).is_letter() {
                        pos += 1;
                    }
                    sp.add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if flags.is_number() {
                pos += 1;
                sp.add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                && flags.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace() | flags2.is_letter() | flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                let mut cpt2 = sp.cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            let mut last_end_r_or_n = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = sp.cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // regex: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// Qwen3.5 system regex — like Qwen2 but letter runs also consume \p{M}:
/// "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
fn regex_split_custom_qwen35(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            let contraction = match_contraction_ci(&sp, pos);
            if contraction > 0 {
                pos += sp.add_token(pos + contraction);
                continue;
            }

            // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number()) {
                if flags.is_letter()
                    || flags.is_accent_mark()
                    || sp.flags(pos + 1).is_accent_mark()
                    || sp.flags(pos + 1).is_letter()
                {
                    pos += 1;
                    while sp.flags(pos).is_letter() || sp.flags(pos).is_accent_mark() {
                        pos += 1;
                    }
                    sp.add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if flags.is_number() {
                pos += 1;
                sp.add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace()
                | flags2.is_letter()
                | flags2.is_accent_mark()
                | flags2.is_number())
                && flags.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace()
                    | flags2.is_letter()
                    | flags2.is_accent_mark()
                    | flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                let mut cpt2 = sp.cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            let mut last_end_r_or_n = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = sp.cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // regex: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // regex: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// K2 system regex patterns (see unicode.cpp for the full pattern).
fn regex_split_custom_kimi_k2(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let cpt = sp.cpt(pos);
            let flags = sp.flags(pos);

            // Pattern 1: [\p{Han}]+ (Chinese characters)
            if cpt_is_han(cpt) {
                while cpt_is_han(sp.cpt(pos)) {
                    pos += 1;
                }
                sp.add_token(pos);
                continue;
            }

            // Pattern 2 & 3: letter words excluding Han characters
            let is_letter_pattern = (flags.is_letter() && !cpt_is_han(cpt))
                || (!(cpt == '\r' as u32
                    || cpt == '\n' as u32
                    || flags.is_letter()
                    || flags.is_number())
                    && sp.flags(pos + 1).is_letter()
                    && !cpt_is_han(sp.cpt(pos + 1)));

            if is_letter_pattern {
                // optional leading non-letter/non-number character
                let mut has_leading_char = false;
                if !(cpt == '\r' as u32
                    || cpt == '\n' as u32
                    || flags.is_letter()
                    || flags.is_number())
                {
                    has_leading_char = true;
                    pos += 1;
                }

                let mut has_letters = false;
                while sp.flags(pos).is_letter() && !cpt_is_han(sp.cpt(pos)) {
                    has_letters = true;
                    pos += 1;
                }

                if has_letters
                    || (!has_leading_char && sp.flags(pos).is_letter() && !cpt_is_han(sp.cpt(pos)))
                {
                    if !has_letters {
                        pos += 1; // consume the first letter
                    }
                    while sp.flags(pos).is_letter() && !cpt_is_han(sp.cpt(pos)) {
                        pos += 1;
                    }

                    // optional contraction (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if sp.cpt(pos) == '\'' as u32 && pos + 1 < offset_end {
                        let cpt_next = tolower(sp.cpt(pos + 1));
                        if cpt_next == 's' as u32
                            || cpt_next == 't' as u32
                            || cpt_next == 'm' as u32
                            || cpt_next == 'd' as u32
                        {
                            pos += 2;
                        } else if pos + 2 < offset_end {
                            let cpt_next_next = tolower(sp.cpt(pos + 2));
                            if (cpt_next == 'r' as u32 && cpt_next_next == 'e' as u32)
                                || (cpt_next == 'v' as u32 && cpt_next_next == 'e' as u32)
                                || (cpt_next == 'l' as u32 && cpt_next_next == 'l' as u32)
                            {
                                pos += 3;
                            }
                        }
                    }

                    sp.add_token(pos);
                    continue;
                } else if has_leading_char {
                    // consumed a leading char but found no letters — backtrack
                    pos -= 1;
                }
            }

            // Pattern 4: \p{N}{1,3}
            if flags.is_number() {
                let mut ini = pos;
                while sp.flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        sp.add_token(pos);
                        ini = pos;
                    }
                }
                sp.add_token(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]*
            let mut flags2 = if cpt == ' ' as u32 {
                sp.flags(pos + 1)
            } else {
                flags
            };
            if !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_number())
                && flags2.as_uint() != 0
            {
                pos += usize::from(cpt == ' ' as u32);
                while !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_number())
                    && flags2.as_uint() != 0
                {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                let mut cpt2 = sp.cpt(pos);
                while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                sp.add_token(pos);
                continue;
            }

            let mut num_whitespaces = 0usize;
            let mut last_end_r_or_n = 0usize;
            while sp.flags(pos + num_whitespaces).is_whitespace() {
                let cpt2 = sp.cpt(pos + num_whitespaces);
                if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces += 1;
            }

            // Pattern 6: \s*[\r\n]+
            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                sp.add_token(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S)
            if num_whitespaces > 1 && sp.cpt(pos + num_whitespaces) != OUT_OF_RANGE {
                pos += num_whitespaces - 1;
                sp.add_token(pos);
                continue;
            }

            // Pattern 8: \s+
            if num_whitespaces > 0 {
                pos += num_whitespaces;
                sp.add_token(pos);
                continue;
            }

            // no matches
            pos += 1;
            sp.add_token(pos);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// AFMOE digit handling: splits digits with leading 1-2 based on total length
/// modulo 3 (also used for the tiny_aya digit-grouping pattern).
fn regex_split_custom_afmoe(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;
        let out = std::mem::take(&mut bpe_offsets);
        let mut sp = Splitter {
            cpts,
            offset_ini,
            offset_end,
            prev_end: offset_ini,
            out,
        };

        let mut pos = offset_ini;
        while pos < offset_end {
            let flags = sp.flags(pos);

            if flags.is_number() {
                let digit_start = pos;
                let mut digit_count = 0usize;

                while sp.flags(pos).is_number() && pos < offset_end {
                    digit_count += 1;
                    pos += 1;
                }

                let remainder = digit_count % 3;
                let mut current = digit_start;

                if remainder > 0 {
                    sp.add_token(current + remainder);
                    current += remainder;
                }

                while current < digit_start + digit_count {
                    sp.add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            pos += 1;
        }

        if sp.prev_end < offset_end {
            sp.add_token(offset_end);
        }

        bpe_offsets = sp.out;
    }
    bpe_offsets
}

/// regex: [^\n]+|[\n]+ — runs of non-newline / newline codepoints.
fn regex_split_custom_newlines(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let offset_ini = start;
        let offset_end = start + offset;
        start = offset_end;

        let mut pos = offset_ini;
        while pos < offset_end {
            let is_newline = cpts[pos] == '\n' as u32;
            let run_start = pos;
            while pos < offset_end && (cpts[pos] == '\n' as u32) == is_newline {
                pos += 1;
            }
            bpe_offsets.push(pos - run_start);
        }
    }
    bpe_offsets
}

/// `unicode_regex_split_custom` — dispatch to a hand-written splitter when the
/// regex string matches one of the known patterns (empty result = unhandled).
fn regex_split_custom(cpts: &[u32], regex_expr: &str, offsets: &[usize]) -> Vec<usize> {
    match regex_expr {
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" => {
            regex_split_custom_gpt2(cpts, offsets)
        }
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        | "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" => {
            regex_split_custom_llama3(cpts, offsets)
        }
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" => {
            regex_split_custom_qwen2(cpts, offsets)
        }
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" => {
            regex_split_custom_qwen35(cpts, offsets)
        }
        // K2-Horizon: llama3 splitter with marks + ZWNJ/ZWJ inside letter
        // runs (462524043, unicode.cpp:1212-1216)
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" => {
            regex_split_custom_k2_horizon(cpts, offsets)
        }
        "\\p{Han}+" => regex_split_custom_kimi_k2(cpts, offsets),
        "\\p{AFMoE_digits}" => regex_split_custom_afmoe(cpts, offsets),
        "[^\\n]+|[\\n]+" => regex_split_custom_newlines(cpts, offsets),
        "\\d{1,3}(?=(?:\\d{3})*\\b)" => regex_split_custom_afmoe(cpts, offsets),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// generic regex fallback (std::regex / std::wregex equivalent)
// ---------------------------------------------------------------------------

/// `\p{...}` categories recognized by the collapsed-representation trick
/// (k_ucat_enum in unicode.cpp). Note llama.cpp maps Lu/Ll/Lt/Lm/Lo all to
/// the generic LETTER flag — we do the same for 1:1 behavior.
const K_UCAT_ENUM: &[(&str, u16)] = &[
    ("\\p{N}", CptFlags::NUMBER),
    ("\\p{L}", CptFlags::LETTER),
    ("\\p{P}", CptFlags::PUNCTUATION),
    ("\\p{M}", CptFlags::ACCENT_MARK),
    ("\\p{S}", CptFlags::SYMBOL),
    ("\\p{Lu}", CptFlags::LETTER), // Uppercase letter
    ("\\p{Ll}", CptFlags::LETTER), // Lowercase letter
    ("\\p{Lt}", CptFlags::LETTER), // Titlecase letter
    ("\\p{Lm}", CptFlags::LETTER), // Modifier letter
    ("\\p{Lo}", CptFlags::LETTER), // Other letter
];

/// collapsed byte assigned to each category (k_ucat_cpt in unicode.cpp).
fn ucat_cpt(flag: u16) -> Option<u8> {
    match flag {
        CptFlags::NUMBER => Some(0xD1),
        CptFlags::LETTER => Some(0xD2),
        CptFlags::PUNCTUATION => Some(0xD3),
        CptFlags::ACCENT_MARK => Some(0xD4),
        CptFlags::SYMBOL => Some(0xD5),
        _ => None,
    }
}

/// ASCII class content added next to the collapsed category byte
/// (k_ucat_map in unicode.cpp, after C++ escape processing).
fn ucat_map(flag: u16) -> &'static str {
    match flag {
        CptFlags::NUMBER => "\u{30}-\u{39}",              // 0-9
        CptFlags::LETTER => "\u{41}-\u{5A}\u{61}-\u{7A}", // A-Za-z
        CptFlags::PUNCTUATION => "!-#%-*,-/:-;?-@\\[-\\]_\\{\\}",
        CptFlags::ACCENT_MARK => "", // no sub-128 codepoints
        CptFlags::SYMBOL => "\\$\\+<=>^`\\|~",
        _ => "",
    }
}

/// One codepoint → one "collapsed" byte (see unicode_regex_split in unicode.cpp).
fn collapse_cpt(cpt: u32) -> u8 {
    if cpt < 128 {
        return cpt as u8;
    }
    let flags = cpt_flags_from_cpt(cpt);
    if flags.is_whitespace() {
        // std::regex \s does not match 0x85 — vertical tab is the whitespace fallback
        0x0B
    } else if let Some(b) = ucat_cpt(flags.category_flag()) {
        b
    } else {
        0xD0 // fallback
    }
}

/// Build the collapsed regex: replace `\p{...}` with byte classes. Bytes are
/// represented as chars U+0000..U+00FF (Latin-1) so the result is valid UTF-8
/// for the regex engine while remaining byte-equivalent to the C++ original.
fn collapse_regex(regex_expr: &str) -> String {
    let mut out = String::with_capacity(regex_expr.len() + 16);
    let mut inside = false;
    let bytes = regex_expr.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' && (i == 0 || bytes[i - 1] != b'\\') {
            out.push('[');
            inside = true;
            i += 1;
            continue;
        }
        if inside && bytes[i] == b']' && bytes[i - 1] != b'\\' {
            out.push(']');
            inside = false;
            i += 1;
            continue;
        }
        if bytes[i] == b'\\' && i + 3 < bytes.len() && bytes[i + 1] == b'p' && bytes[i + 2] == b'{'
        {
            if let Some(rel) = regex_expr[i + 3..].find('}') {
                let closing_brace = rel + i + 3;
                if closing_brace <= i + 10 {
                    // reasonable limit, as in unicode.cpp
                    let pat = &regex_expr[i..closing_brace + 1];
                    if let Some((_, flag)) = K_UCAT_ENUM.iter().find(|(k, _)| *k == pat) {
                        if !inside {
                            out.push('[');
                        }
                        out.push(ucat_cpt(*flag).unwrap() as char);
                        out.push_str(ucat_map(*flag));
                        if !inside {
                            out.push(']');
                        }
                        i = closing_brace + 1;
                        continue;
                    }
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn compiled_regex_cache() -> &'static Mutex<HashMap<String, Regex>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn compile_regex(expr: &str) -> Regex {
    if let Some(re) = compiled_regex_cache().lock().unwrap().get(expr) {
        return re.clone();
    }
    match Regex::new(expr) {
        Ok(re) => {
            compiled_regex_cache()
                .lock()
                .unwrap()
                .insert(expr.to_string(), re.clone());
            re
        }
        // C++ prints and throws here; the shipped pre-tokenizer patterns never
        // fail to compile, so treat this as unreachable
        Err(e) => panic!("Failed to process regex: '{expr}': {e}"),
    }
}

/// `unicode_regex_split_stl` — std::regex_iterator semantics: find successive
/// matches in each segment; gaps between matches become their own tokens.
/// `text` must have exactly one char per codepoint (wtext or collapsed text).
fn regex_split_stl(text: &str, regex_expr: &str, offsets: &[usize]) -> Vec<usize> {
    let re = compile_regex(regex_expr);

    // char index → byte offset table for slicing at codepoint boundaries
    let mut char_starts: Vec<usize> = Vec::with_capacity(text.len());
    let mut b = 0usize;
    for c in text.chars() {
        char_starts.push(b);
        b += c.len_utf8();
    }
    let n_chars = char_starts.len();
    // byte position (always on a char boundary) → char index; `len()` maps to
    // n_chars (one past the last char)
    let char_idx = |byte_pos: usize| -> usize { char_starts.partition_point(|&s| s < byte_pos) };

    let mut bpe_offsets: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &offset in offsets {
        let seg_start = if start < n_chars {
            char_starts[start]
        } else {
            text.len()
        };
        let seg_end = if start + offset < n_chars {
            char_starts[start + offset]
        } else {
            text.len()
        };
        let seg = &text[seg_start..seg_end];

        let mut start_idx = 0usize; // in chars, relative to segment start
        let mut search_pos = 0usize; // byte position within segment
        loop {
            let m = match re.find_from_pos(seg, search_pos) {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(e) => panic!("regex match error: {e}"),
            };
            // match byte offsets are segment-relative; convert to char
            // indices relative to the segment start
            let m_start_c = char_idx(seg_start + m.start()) - start;
            let m_end_c = char_idx(seg_start + m.end()) - start;
            let m_len_c = m_end_c - m_start_c;

            if m_start_c > start_idx {
                bpe_offsets.push(m_start_c - start_idx);
            }
            bpe_offsets.push(m_len_c);
            start_idx = m_start_c + m_len_c;

            if m_len_c == 0 {
                // std::regex_iterator: after an empty match, resume one char
                // past it (terminating if at the end of the range)
                if m.end() >= seg.len() {
                    break;
                }
                let mut next = m.end() + 1;
                while next < seg.len() && !seg.is_char_boundary(next) {
                    next += 1;
                }
                search_pos = next;
            } else {
                search_pos = m.end();
            }
        }

        if start_idx < offset {
            bpe_offsets.push(offset - start_idx);
        }
        start += offset;
    }

    bpe_offsets
}

/// GPT-2 byte-level encoding of already-split words
/// (`unicode_byte_encoding_process` in unicode.cpp).
fn byte_encoding_process(bpe_words: Vec<String>) -> Vec<String> {
    bpe_words
        .into_iter()
        .map(|word| {
            let mut encoded = String::with_capacity(word.len() * 2);
            for b in word.as_bytes() {
                encoded.push_str(byte_to_utf8(*b));
            }
            encoded
        })
        .collect()
}

/// `unicode_regex_split(text, regex_exprs, byte_encode)` — split text into
/// words by applying each regex in sequence.
pub fn regex_split(text: &str, regex_exprs: &[String], byte_encode: bool) -> Vec<String> {
    // 462524043 (unicode.cpp:1369-1371): the empty input returns early —
    // the collapsed-representation pass below would otherwise emit one
    // empty word
    if text.is_empty() {
        return Vec::new();
    }
    let cpts = cpts_from_utf8(text.as_bytes());

    // compute collapsed codepoints only if needed by at least one regex
    let need_collapse = regex_exprs.iter().any(|regex_expr| {
        K_UCAT_ENUM
            .iter()
            .any(|(ucat, _)| regex_expr.contains(ucat))
    });

    // collapsed representation: one byte (as a Latin-1 char) per codepoint
    let text_collapsed: String = if need_collapse {
        cpts.iter().map(|&cpt| collapse_cpt(cpt) as char).collect()
    } else {
        String::new()
    };

    let mut bpe_offsets: Vec<usize> = vec![cpts.len()];

    for regex_expr in regex_exprs {
        // first, see if we have an efficient custom regex implementation
        let tmp = regex_split_custom(&cpts, regex_expr, &bpe_offsets);
        if !tmp.is_empty() {
            bpe_offsets = tmp;
            continue;
        }

        // fallback to general-purpose regex
        let use_collapsed = K_UCAT_ENUM
            .iter()
            .any(|(ucat, _)| regex_expr.contains(ucat));
        if use_collapsed {
            // sanity-check that the original regex does not contain any non-ASCII characters
            let cpts_regex = cpts_from_utf8(regex_expr.as_bytes());
            if cpts_regex.iter().any(|&cpt| cpt >= 128) {
                panic!("Regex includes both unicode categories and non-ASCII characters - not supported");
            }
            let regex_expr_collapsed = collapse_regex(regex_expr);
            bpe_offsets = regex_split_stl(&text_collapsed, &regex_expr_collapsed, &bpe_offsets);
        } else {
            // no unicode category used — run the regex over the codepoints,
            // with non-ASCII whitespace folded to U+000B (std::wregex \s does
            // not match non-ASCII whitespace)
            let wtext: String = cpts
                .iter()
                .map(|&cpt| {
                    if cpt > 0x7F && cpt_flags_from_cpt(cpt).is_whitespace() {
                        '\u{0B}'
                    } else {
                        cpt_to_char(cpt)
                    }
                })
                .collect();
            bpe_offsets = regex_split_stl(&wtext, regex_expr, &bpe_offsets);
        }
    }

    let mut bpe_words: Vec<String> = Vec::with_capacity(bpe_offsets.len());
    let mut start = 0usize;
    for &offset in &bpe_offsets {
        let mut word = String::new();
        for i in start..start + offset {
            word.push(cpt_to_char(cpts[i]));
        }
        bpe_words.push(word);
        start += offset;
    }

    if byte_encode {
        return byte_encoding_process(bpe_words);
    }

    bpe_words
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len_utf8() {
        assert_eq!(len_utf8(b'a'), 1);
        assert_eq!(len_utf8(0x7f), 1);
        assert_eq!(len_utf8(0xC0), 2);
        assert_eq!(len_utf8(0xDF), 2);
        assert_eq!(len_utf8(0xE0), 3);
        assert_eq!(len_utf8(0xEF), 3);
        assert_eq!(len_utf8(0xF0), 4);
        assert_eq!(len_utf8(0xF7), 4);
    }

    #[test]
    fn test_cpt_roundtrip() {
        for &cpt in &[0x24u32, 0xA2, 0x20AC, 0x10348] {
            let s = cpt_to_utf8(cpt);
            let mut off = 0usize;
            assert_eq!(cpt_from_utf8(s.as_bytes(), &mut off), Ok(cpt));
            assert_eq!(off, s.len());
        }
        // invalid continuation
        let mut off = 0usize;
        assert_eq!(cpt_from_utf8(&[0xC3, 0x28], &mut off), Err(()));
    }

    #[test]
    fn test_cpts_from_utf8_invalid() {
        // invalid byte → U+FFFD, one per skipped byte
        let cpts = cpts_from_utf8(&[b'a', 0xFF, b'b']);
        assert_eq!(cpts, vec!['a' as u32, 0xFFFD, 'b' as u32]);
    }

    #[test]
    fn test_flags() {
        let f = cpt_flags_from_cpt('a' as u32);
        assert!(f.is_letter() && f.is_lowercase() && !f.is_uppercase());
        let f = cpt_flags_from_cpt('A' as u32);
        assert!(f.is_letter() && f.is_uppercase() && !f.is_lowercase());
        let f = cpt_flags_from_cpt('0' as u32);
        assert!(f.is_number());
        let f = cpt_flags_from_cpt(' ' as u32);
        assert!(f.is_separator() && f.is_whitespace());
        let f = cpt_flags_from_cpt(0x0B);
        assert!(f.is_whitespace() && f.is_control());
        // CJK is a letter (Lo)
        assert!(cpt_flags_from_cpt(0x4E2D).is_letter());
        // emoji is a symbol
        assert!(cpt_flags_from_cpt(0x1F999).is_symbol());
        // out of range → undefined
        assert!(cpt_flags_from_cpt(0x110000).is_undefined());
    }

    #[test]
    fn test_tolower() {
        assert_eq!(tolower('A' as u32), 'a' as u32);
        assert_eq!(tolower('a' as u32), 'a' as u32);
        assert_eq!(tolower('1' as u32), '1' as u32);
        assert_eq!(tolower(0x00C0), 0x00E0); // À → à
    }

    #[test]
    fn test_byte_to_utf8() {
        assert_eq!(byte_to_utf8(b'!'), "!");
        assert_eq!(byte_to_utf8(0x7E), "~");
        // printable ASCII maps to itself; unmapped bytes map to U+0100+
        // 0x20 is the 33rd unmapped byte (0x00..=0x20) → U+0120 'Ġ'
        assert_eq!(byte_to_utf8(b' '), "\u{0120}");
        assert_eq!(byte_to_utf8(0x00), "\u{0100}");
        assert_eq!(byte_to_utf8(0xA1), "\u{A1}");
        // round trip
        for b in 0..=255u8 {
            assert_eq!(utf8_to_byte(byte_to_utf8(b)), Some(b));
        }
    }

    #[test]
    fn test_nfd() {
        assert_eq!(cpts_normalize_nfd(&[0x00C0]), vec![0x0041]); // À → A
        assert_eq!(cpts_normalize_nfd(&[0x0041]), vec![0x0041]);
        assert_eq!(cpts_normalize_nfd(&[0x00E4]), vec![0x0061]); // ä → a
    }

    #[test]
    fn test_is_han() {
        assert!(cpt_is_han(0x4E2D));
        assert!(cpt_is_han(0x9FFF));
        assert!(cpt_is_han(0x3400));
        assert!(!cpt_is_han(0x3042)); // hiragana
        assert!(!cpt_is_han('a' as u32));
    }

    const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    #[test]
    fn test_regex_split_qwen2_custom() {
        let exprs = vec![QWEN2_RE.to_string()];
        assert_eq!(
            regex_split("Hello world", &exprs, false),
            vec!["Hello", " world"]
        );
        assert_eq!(
            regex_split("中文 test", &exprs, false),
            vec!["中文", " test"]
        );
        assert_eq!(
            regex_split("12345", &exprs, false),
            vec!["1", "2", "3", "4", "5"]
        );
        assert_eq!(regex_split("don't", &exprs, false), vec!["don", "'t"]);
        assert_eq!(regex_split("a \n b", &exprs, false), vec!["a", " \n", " b"]);
        assert_eq!(regex_split("42!", &exprs, false), vec!["4", "2", "!"]);
        assert_eq!(
            regex_split("你好，世界！", &exprs, false),
            vec!["你好", "，世界", "！"]
        );
    }

    #[test]
    fn test_regex_split_gpt2_custom() {
        let exprs = vec![
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)".to_string(),
        ];
        assert_eq!(
            regex_split("Hello world", &exprs, false),
            vec!["Hello", " world"]
        );
        assert_eq!(
            regex_split("w048 7tuijk", &exprs, false),
            vec!["w", "048", " 7", "tuijk"]
        );
        assert_eq!(regex_split("   x", &exprs, false), vec!["  ", " x"]);
        assert_eq!(regex_split("   ", &exprs, false), vec!["   "]);
    }

    #[test]
    fn test_regex_split_llama3_custom() {
        let exprs = vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".to_string()];
        assert_eq!(
            regex_split("1234567", &exprs, false),
            vec!["123", "456", "7"]
        );
        assert_eq!(
            regex_split("Hello World", &exprs, false),
            vec!["Hello", " World"]
        );
        assert_eq!(
            regex_split("世界 hello", &exprs, false),
            vec!["世界", " hello"]
        );
    }

    #[test]
    fn test_regex_split_generic_fallback() {
        // non-collapsed path: plain regex without \p{}
        let exprs = vec!["[0-9][0-9][0-9]".to_string()];
        assert_eq!(
            regex_split("a12345b", &exprs, false),
            vec!["a", "123", "45b"]
        );

        // collapsed path: \p{N}+ on unicode digits
        let exprs = vec![r"\p{N}+".to_string()];
        assert_eq!(regex_split("a12٣4b", &exprs, false), vec!["a", "12٣4", "b"]);

        // collapsed path with lookahead: \s+$
        let exprs = vec![r"\s+$".to_string()];
        assert_eq!(regex_split("ab  ", &exprs, false), vec!["ab", "  "]);

        // deepseek-llm style: \s?\p{L}+ (collapsed, optional space prefix)
        let exprs = vec![r"\s?\p{L}+".to_string()];
        assert_eq!(regex_split("a b", &exprs, false), vec!["a", " b"]);
        assert_eq!(regex_split("αβγ", &exprs, false), vec!["αβγ"]);
    }

    #[test]
    fn test_byte_encoding() {
        let exprs = vec![
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)".to_string(),
        ];
        let words = regex_split(" Hello", &exprs, true);
        assert_eq!(words, vec!["\u{0120}Hello"]); // ĠHello
        let words = regex_split("你好", &exprs, true);
        // CJK bytes map to U+01xx chars
        let expected: String = "你好".as_bytes().iter().map(|&b| byte_to_utf8(b)).collect();
        assert_eq!(words, vec![expected]);
    }
}
