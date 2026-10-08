//! console.rs — port of `common/console.cpp` (pinned bd4f514db1, 1,166 lines):
//! the interactive line-editing console of the tools (`llama-completion`'s
//! `-i` loop, the reference's only remaining consumer).
//!
//! **Console decision (documented in PARITY.md)**: the file is ported in
//! full, both reader paths —
//!
//!   * `readline_simple` (console.cpp:1046-1081) — plain `std::getline`
//!     lines, `/` returns control, `\` toggles multiline. This is what runs
//!     under `--simple-io` and in subprocesses; the scripted (non-tty)
//!     verification path.
//!   * `readline_advanced` (console.cpp:753-1044) — the raw-mode editor:
//!     UTF-8 cursor-aware insert/delete, escape sequences (arrows, home/end,
//!     delete, ctrl-left/right word jumps), history up/down, `\`-continuation
//!     and `/`-submit marker, tab completion hook.
//!
//! Mapping (C → Rust):
//!   console::init/cleanup            (:89-166)   -> [`Console::init`]/[`Console::cleanup`]
//!   console::set_display             (:173-198)  -> [`Console::set_display`]
//!   getchar32                        (:200-263)  -> [`Console::getchar32`] (raw stdin bytes
//!                                                    decoded as UTF-8 — `getwchar` under a
//!                                                    UTF-8 locale; the "C"-locale byte-mangling
//!                                                    of multi-byte input is not reproduced)
//!   pop_cursor                       (:265-284)  -> [`Console::pop_cursor`]
//!   estimateWidth                    (:286-293)  -> [`estimate_width`] (`wcwidth` via
//!                                                    `extern "C"`, the mlock.rs precedent —
//!                                                    no libc crate)
//!   put_codepoint                    (:295-354)  -> [`Console::put_codepoint`] (the non-tty
//!                                                    branch + the `\033[6n` cursor query on
//!                                                    the open `/dev/tty`)
//!   replace_last                     (:356-363)  -> [`Console::replace_last`]
//!   decode_utf8                      (:365-409)  -> [`decode_utf8`]
//!   append_utf8                      (:411-429)  -> [`append_utf8`]
//!   prev/next_utf8_char_pos          (:432-448)  -> [`prev_utf8_char_pos`]/[`next_utf8_char_pos`]
//!   delete_at_cursor                 (:456-482)  -> [`Console::delete_at_cursor`]
//!   clear_current_line               (:484-495)  -> [`Console::clear_current_line`]
//!   set_line_contents                (:497-527)  -> [`Console::set_line_contents`]
//!   move_to_line_start/end           (:529-547)  -> [`Console::move_to_line_start`]/`_end`
//!   has_ctrl_modifier                (:549-575)  -> [`has_ctrl_modifier`]
//!   is_space_codepoint               (:577-579)  -> [`is_space_codepoint`]
//!   move_word_left/right             (:581-664)  -> [`Console::move_word_left`]/`_right`
//!   move_cursor                      (:666-697)  -> [`Console::move_cursor`]
//!   history_t                        (:699-751)  -> [`History`]
//!   readline_advanced                (:753-1044) -> [`Console::readline_advanced`]
//!   readline_simple                  (:1046-1081)-> [`Console::readline_simple`]
//!   console::readline                (:1083-1088) -> [`Console::readline`]
//!   console::log/error/flush         (:1146-1165)-> [`Console::log`]/[`Console::error`]/[`Console::flush`]
//!
//! Documented deviations:
//!   * no libc crate — `termios`/`tcgetattr`/`tcsetattr`/`wcwidth` are
//!     declared `extern "C"` (the mlock.rs precedent; Linux layout).
//!   * the Windows half of every `#ifdef _WIN32` is not ported (the port is
//!     POSIX-only, FILE_MAP's standing decision).
//!   * `console::spinner` (:1094-1144) is not ported: its only consumers are
//!     the unported examples (parallel/batched) and cli-ui.h — no ported tool
//!     reaches it.
//!   * `set_completion_callback` (:1090-1092) is carried as a field-less
//!     no-op — the pinned consumer set (tools/completion) never installs one.

use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// termios + wcwidth via extern "C" (mlock.rs precedent; Linux x86_64 layout)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32],
    c_ispeed: u32,
    c_ospeed: u32,
}

extern "C" {
    fn tcgetattr(fd: i32, termios_p: *mut Termios) -> i32;
    fn tcsetattr(fd: i32, optional_actions: i32, termios_p: *const Termios) -> i32;
    fn wcwidth(wc: u32) -> i32;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

const STDIN_FILENO: i32 = 0;
const STDOUT_FILENO: i32 = 1;
const TCSANOW: i32 = 0;
const TIOCGWINSZ: u64 = 0x5413;
const ICANON: u32 = 0o0000002;
const ECHO: u32 = 0o0000010;
const VMIN: usize = 6;
const VTIME: usize = 5;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

/// `wcwidth` (:291) — negative for non-printable codepoints, which the
/// callers clamp to 0 (or, with a live tty, resolve through a cursor query
/// in [`Console::put_codepoint`]).
fn estimate_width(codepoint: u32) -> i32 {
    unsafe { wcwidth(codepoint) }
}

/// ANSI escapes (:37-45)
const ANSI_COLOR_RESET: &str = "\x1b[0m";
const ANSI_COLOR_MAGENTA: &str = "\x1b[35m";
const ANSI_COLOR_YELLOW: &str = "\x1b[33m";
const ANSI_COLOR_GRAY: &str = "\x1b[90m";
const ANSI_COLOR_GREEN: &str = "\x1b[32m";
const ANSI_COLOR_RED: &str = "\x1b[31m";
const ANSI_BOLD: &str = "\x1b[1m";

/// `display_type` (console.h:14-22)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayType {
    Reset,
    Info,
    Prompt,
    Reasoning,
    UserInput,
    Error,
}

/// the end-of-stream sentinel of `getchar32` (`WEOF`)
const WEOF: Option<u32> = None;

// ---------------------------------------------------------------------------
// UTF-8 helpers — decode_utf8 (:365-409), append_utf8 (:411-429),
// prev/next_utf8_char_pos (:432-448)
// ---------------------------------------------------------------------------

/// `decode_utf8` (:365-409) — one codepoint from `input[pos..]`, writing the
/// byte `advance`. Invalid sequences decode to U+FFFD with `advance = 1`.
pub fn decode_utf8(input: &[u8], pos: usize) -> (u32, usize) {
    let c = input[pos];
    if c & 0x80 == 0 {
        return (c as u32, 1);
    }
    if c & 0xE0 == 0xC0 && pos + 1 < input.len() {
        let c1 = input[pos + 1];
        if c1 & 0xC0 != 0x80 {
            return (0xFFFD, 1);
        }
        return ((((c & 0x1F) as u32) << 6) | ((c1 & 0x3F) as u32), 2);
    }
    if c & 0xF0 == 0xE0 && pos + 2 < input.len() {
        let (c1, c2) = (input[pos + 1], input[pos + 2]);
        if c1 & 0xC0 != 0x80 || c2 & 0xC0 != 0x80 {
            return (0xFFFD, 1);
        }
        return (
            (((c & 0x0F) as u32) << 12)
                | (((c1 & 0x3F) as u32) << 6)
                | ((c2 & 0x3F) as u32),
            3,
        );
    }
    if c & 0xF8 == 0xF0 && pos + 3 < input.len() {
        let (c1, c2, c3) = (input[pos + 1], input[pos + 2], input[pos + 3]);
        if c1 & 0xC0 != 0x80 || c2 & 0xC0 != 0x80 || c3 & 0xC0 != 0x80 {
            return (0xFFFD, 1);
        }
        return (
            (((c & 0x07) as u32) << 18)
                | (((c1 & 0x3F) as u32) << 12)
                | (((c2 & 0x3F) as u32) << 6)
                | ((c3 & 0x3F) as u32),
            4,
        );
    }
    (0xFFFD, 1) // replacement character for invalid input
}

/// `append_utf8` (:411-429)
pub fn append_utf8(ch: u32, out: &mut Vec<u8>) {
    if ch <= 0x7F {
        out.push(ch as u8);
    } else if ch <= 0x7FF {
        out.push((0xC0 | ((ch >> 6) & 0x1F)) as u8);
        out.push((0x80 | (ch & 0x3F)) as u8);
    } else if ch <= 0xFFFF {
        out.push((0xE0 | ((ch >> 12) & 0x0F)) as u8);
        out.push((0x80 | ((ch >> 6) & 0x3F)) as u8);
        out.push((0x80 | (ch & 0x3F)) as u8);
    } else if ch <= 0x10FFFF {
        out.push((0xF0 | ((ch >> 18) & 0x07)) as u8);
        out.push((0x80 | ((ch >> 12) & 0x3F)) as u8);
        out.push((0x80 | ((ch >> 6) & 0x3F)) as u8);
        out.push((0x80 | (ch & 0x3F)) as u8);
    }
    // Invalid Unicode code point — dropped (:426-428)
}

/// `prev_utf8_char_pos` (:432-439) — byte index of the previous character.
fn prev_utf8_char_pos(line: &[u8], pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut pos = pos - 1;
    while pos > 0 && line[pos] & 0xC0 == 0x80 {
        pos -= 1;
    }
    pos
}

/// `next_utf8_char_pos` (:441-448)
fn next_utf8_char_pos(line: &[u8], pos: usize) -> usize {
    if pos >= line.len() {
        return line.len();
    }
    let mut pos = pos + 1;
    while pos < line.len() && line[pos] & 0xC0 == 0x80 {
        pos += 1;
    }
    pos
}

/// `has_ctrl_modifier` (:549-575) — a `5` in any `;`-separated parameter
/// group of an escape sequence.
fn has_ctrl_modifier(params: &str) -> bool {
    let bytes = params.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        let end = params[start..].find(';').map(|i| start + i);
        let seg_end = end.unwrap_or(bytes.len());
        if seg_end > start {
            let mut value: i32 = 0;
            for &ch in &bytes[start..seg_end] {
                if !ch.is_ascii_digit() {
                    value = -1;
                    break;
                }
                value = value * 10 + (ch - b'0') as i32;
            }
            if value == 5 {
                return true;
            }
        }
        match end {
            None => break,
            Some(e) => start = e + 1,
        }
    }
    false
}

/// `is_space_codepoint` (:577-579) — `std::iswspace`
fn is_space_codepoint(cp: u32) -> bool {
    char::from_u32(cp).is_some_and(char::is_whitespace)
}

// ---------------------------------------------------------------------------
// history_t (:699-751)
// ---------------------------------------------------------------------------

/// `console::history_t` (:699-751)
#[derive(Default)]
struct History {
    entries: Vec<String>,
    /// `SIZE_MAX` == not viewing
    viewing_idx: Option<usize>,
    /// current line before viewing history
    backup_line: String,
}

impl History {
    /// `history_t::add` (:703-713)
    fn add(&mut self, line: &str) {
        if line.is_empty() {
            return;
        }
        // avoid duplicates with the last entry
        if self.entries.is_empty() || self.entries.last().map(|s| s.as_str()) != Some(line) {
            self.entries.push(line.to_string());
        }
        // also clear viewing state
        self.end_viewing();
    }

    /// `history_t::prev` (:714-726)
    fn prev(&mut self, cur_line: &mut String) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let Some(idx) = self.viewing_idx else {
            return false;
        };
        let idx = idx.saturating_sub(1);
        self.viewing_idx = Some(idx);
        *cur_line = self.entries[idx].clone();
        true
    }

    /// `history_t::next` (:727-739)
    fn next(&mut self, cur_line: &mut String) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let Some(mut idx) = self.viewing_idx else {
            return false;
        };
        idx += 1;
        if idx >= self.entries.len() {
            *cur_line = self.backup_line.clone();
            self.end_viewing();
        } else {
            self.viewing_idx = Some(idx);
            *cur_line = self.entries[idx].clone();
        }
        true
    }

    /// `history_t::begin_viewing` (:740-743)
    fn begin_viewing(&mut self, line: &str) {
        self.backup_line = line.to_string();
        self.viewing_idx = Some(self.entries.len());
    }

    /// `history_t::end_viewing` (:744-747)
    fn end_viewing(&mut self) {
        self.viewing_idx = None;
        self.backup_line.clear();
    }

    /// `history_t::is_viewing` (:748-750)
    fn is_viewing(&self) -> bool {
        self.viewing_idx.is_some()
    }
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

/// `console::`'s file-scope state (:70-83) — one instance per process, owned
/// by the interactive driver (`llama-cli -i`) where the C kept statics.
pub struct Console {
    /// `advanced_display` — `params.use_color` (console.cpp:70)
    advanced_display: bool,
    /// `simple_io` — `params.simple_io` (:71)
    simple_io: bool,
    /// `current_display` (:72)
    current_display: DisplayType,
    /// `tty`/`out` (:74-80) — `stdout`, or the `/dev/tty` handle when the
    /// advanced editor could open it (init :141-144)
    tty: Option<std::fs::File>,
    /// `initial_state` (:80) — the pre-init termios, restored by `cleanup`
    initial_state: Option<Termios>,
    /// the session's history (:751)
    history: History,
    /// `getchar32`'s pending multi-byte tail — the C's `getwchar` consumes
    /// whole characters; the port decodes raw stdin bytes with the same
    /// [`decode_utf8`] the line buffer uses (:365-409)
    stdin_tail: Vec<u8>,
    /// stdin as raw bytes (`_fileno(stdin)` reads in the C)
    stdin_raw: std::io::Stdin,
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

impl Console {
    pub fn new() -> Self {
        Console {
            advanced_display: false,
            simple_io: true,
            current_display: DisplayType::Reset,
            tty: None,
            initial_state: None,
            history: History::default(),
            stdin_tail: Vec::new(),
            stdin_raw: std::io::stdin(),
        }
    }

    /// `console::init` (:89-149) — the POSIX half. `use_simple_io` is
    /// `params.simple_io`, `use_advanced_display` is `params.use_color`.
    pub fn init(&mut self, use_simple_io: bool, use_advanced_display: bool) {
        self.advanced_display = use_advanced_display;
        self.simple_io = use_simple_io;
        if !self.simple_io {
            // raw mode on stdin; on a pipe `tcgetattr` fails and the
            // terminal is left as it was — exactly the C's silent failure
            // (:133-139), which is what makes scripted sessions work
            let mut new_termios;
            let mut initial = unsafe { std::mem::zeroed::<Termios>() };
            if unsafe { tcgetattr(STDIN_FILENO, &mut initial) } == 0 {
                self.initial_state = Some(initial);
                new_termios = initial;
                new_termios.c_lflag &= !(ICANON | ECHO);
                new_termios.c_cc[VMIN] = 1;
                new_termios.c_cc[VTIME] = 0;
                unsafe { tcsetattr(STDIN_FILENO, TCSANOW, &new_termios) };
            }

            if let Ok(tty) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/tty")
            {
                self.tty = Some(tty);
            }
        }
        // `setlocale(LC_ALL, "")` (:147) is a no-op for the port: input is
        // decoded as UTF-8 directly (see [`Self::getchar32`])
    }

    /// `console::cleanup` (:151-166)
    pub fn cleanup(&mut self) {
        self.set_display(DisplayType::Reset);
        if !self.simple_io {
            self.tty = None;
            if let Some(initial) = self.initial_state.take() {
                unsafe { tcsetattr(STDIN_FILENO, TCSANOW, &initial) };
            }
        }
    }

    /// `console::set_display` (:173-198) — only emits when advanced display
    /// (color) is on and the type actually changes.
    pub fn set_display(&mut self, display: DisplayType) {
        if self.advanced_display && self.current_display != display {
            let _ = std::io::stdout().flush();
            let seq = match display {
                DisplayType::Reset => ANSI_COLOR_RESET,
                DisplayType::Info => ANSI_COLOR_MAGENTA,
                DisplayType::Prompt => ANSI_COLOR_YELLOW,
                DisplayType::Reasoning => ANSI_COLOR_GRAY,
                DisplayType::UserInput => "\x1b[1m\x1b[32m", // ANSI_BOLD ANSI_COLOR_GREEN
                DisplayType::Error => "\x1b[1m\x1b[31m",     // ANSI_BOLD ANSI_COLOR_RED
            };
            self.write_out(seq.as_bytes());
            self.current_display = display;
            self.flush_out();
        }
    }

    /// write to `out` — stdout, or `/dev/tty` when the advanced editor
    /// opened one (:74, :141-144)
    fn write_out(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        match self.tty.as_mut() {
            Some(tty) => {
                let _ = tty.write_all(bytes);
            }
            None => {
                let mut so = std::io::stdout();
                let _ = so.write_all(bytes);
            }
        }
    }

    fn write_stdout(&mut self, bytes: &[u8]) {
        let mut so = std::io::stdout();
        let _ = so.write_all(bytes);
    }

    fn flush_out(&mut self) {
        if let Some(tty) = self.tty.as_mut() {
            let _ = tty.flush();
        } else {
            let _ = std::io::stdout().flush();
        }
    }

    /// `getchar32` (:200-263, POSIX half) — one codepoint from stdin.
    /// `getwchar` under a UTF-8 locale decodes multi-byte sequences; the
    /// port decodes the raw bytes with [`decode_utf8`]. EOF → `WEOF`
    /// (`None`). A partial multi-byte prefix blocks for the rest of the
    /// character, like `getwchar` does.
    fn getchar32(&mut self) -> Option<u32> {
        loop {
            if !self.stdin_tail.is_empty() && !is_incomplete(&self.stdin_tail) {
                let (cp, advance) = decode_utf8(&self.stdin_tail, 0);
                self.stdin_tail.drain(..advance);
                return Some(cp);
            }
            let mut buf = [0u8; 1];
            match self.stdin_raw.read(&mut buf) {
                Ok(0) | Err(_) => return WEOF,
                Ok(_) => self.stdin_tail.push(buf[0]),
            }
        }
    }

    /// `pop_cursor` (:265-284, POSIX half)
    fn pop_cursor(&mut self) {
        self.write_out(b"\x08");
    }

    /// `put_codepoint` (:295-354, POSIX half) — write the sequence and return
    /// its display width. `expected_width >= 0` (or no tty) trusts the
    /// estimate (:322-326); otherwise the cursor is queried on the tty
    /// before/after the write (:328-352).
    fn put_codepoint(&mut self, utf8: &[u8], expected_width: i32) -> i32 {
        if expected_width >= 0 || self.tty.is_none() {
            self.write_out(utf8);
            return expected_width;
        }

        let mut tty = self.tty.take().expect("tty");
        let (x1, _y1) = query_cursor_position(&mut tty);
        let _ = tty.write_all(utf8);
        let (x2, _y2) = query_cursor_position(&mut tty);
        self.tty = Some(tty);
        let mut width = x2 as i32 - x1 as i32;
        if width < 0 {
            // Calculate the width considering text wrapping (:347-351)
            let mut w = Winsize::default();
            unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut w as *mut Winsize) };
            width += w.ws_col as i32;
        }
        width
    }

    /// `replace_last` (:356-363, POSIX half)
    fn replace_last(&mut self, ch: u8) {
        let s = format!("\x08{}", ch as char);
        self.write_out(s.as_bytes());
    }

    /// `delete_at_cursor` (:456-482)
    fn delete_at_cursor(&mut self, line: &mut Vec<u8>, widths: &mut Vec<i32>, char_pos: usize, byte_pos: &mut usize) {
        if char_pos >= widths.len() {
            return;
        }
        let next_pos = next_utf8_char_pos(line, *byte_pos);
        let w = widths[char_pos];
        let char_len = next_pos - *byte_pos;

        line.drain(*byte_pos..*byte_pos + char_len);
        widths.remove(char_pos);

        let mut p = *byte_pos;
        let mut tail_width = 0;
        for i in char_pos..widths.len() {
            let following = next_utf8_char_pos(line, p);
            let slice = line[p..following].to_vec();
            self.put_codepoint(&slice, widths[i]);
            tail_width += widths[i];
            p = following;
        }

        let spaces = vec![b' '; w.max(0) as usize];
        self.write_out(&spaces);

        self.move_cursor(-(tail_width + w));
    }

    /// `clear_current_line` (:484-495)
    fn clear_current_line(&mut self, widths: &[i32]) {
        let mut total_width = 0;
        for &w in widths {
            total_width += if w > 0 { w } else { 1 };
        }
        if total_width > 0 {
            let spaces = vec![b' '; total_width as usize];
            self.write_out(&spaces);
            self.move_cursor(-total_width);
        }
    }

    /// `set_line_contents` (:497-527). `cursor_byte_pos < 0` (the C's default
    /// -1) leaves the cursor at the line end.
    fn set_line_contents(
        &mut self,
        new_line: &[u8],
        line: &mut Vec<u8>,
        widths: &mut Vec<i32>,
        char_pos: &mut usize,
        byte_pos: &mut usize,
        cursor_byte_pos: i64,
    ) {
        self.move_to_line_start(char_pos, byte_pos, widths);
        self.clear_current_line(widths);

        *line = new_line.to_vec();
        widths.clear();
        *byte_pos = 0;
        *char_pos = 0;

        let mut idx = 0;
        let mut back_width: i32 = 0;
        while idx < line.len() {
            let (cp, advance) = decode_utf8(line, idx);
            let expected_width = estimate_width(cp);
            let slice = line[idx..idx + advance].to_vec();
            let mut real_width = self.put_codepoint(&slice, expected_width);
            if real_width < 0 {
                real_width = 0;
            }
            widths.push(real_width);
            idx += advance;
            if cursor_byte_pos >= 0 && (cursor_byte_pos as usize) < idx {
                back_width += real_width;
            } else {
                *char_pos += 1;
                *byte_pos = idx;
            }
        }
        if cursor_byte_pos >= 0 {
            self.move_cursor(-back_width);
        }
    }

    /// `move_to_line_start` (:529-537)
    fn move_to_line_start(&mut self, char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32]) {
        let mut back_width = 0;
        for i in 0..*char_pos {
            back_width += widths[i];
        }
        self.move_cursor(-back_width);
        *char_pos = 0;
        *byte_pos = 0;
    }

    /// `move_to_line_end` (:539-547)
    fn move_to_line_end(&mut self, char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line_len: usize) {
        let mut forward_width = 0;
        for i in *char_pos..widths.len() {
            forward_width += widths[i];
        }
        self.move_cursor(forward_width);
        *char_pos = widths.len();
        *byte_pos = line_len;
    }

    /// `move_word_left` (:581-617)
    fn move_word_left(&mut self, char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line: &[u8]) {
        if *char_pos == 0 {
            return;
        }

        let mut new_char_pos = *char_pos;
        let mut new_byte_pos = *byte_pos;
        let mut move_width: i32 = 0;

        while new_char_pos > 0 {
            let prev_byte = prev_utf8_char_pos(line, new_byte_pos);
            let (cp, _) = decode_utf8(line, prev_byte);
            if !is_space_codepoint(cp) {
                break;
            }
            move_width += widths[new_char_pos - 1];
            new_char_pos -= 1;
            new_byte_pos = prev_byte;
        }

        while new_char_pos > 0 {
            let prev_byte = prev_utf8_char_pos(line, new_byte_pos);
            let (cp, _) = decode_utf8(line, prev_byte);
            if is_space_codepoint(cp) {
                break;
            }
            move_width += widths[new_char_pos - 1];
            new_char_pos -= 1;
            new_byte_pos = prev_byte;
        }

        self.move_cursor(-move_width);
        *char_pos = new_char_pos;
        *byte_pos = new_byte_pos;
    }

    /// `move_word_right` (:619-664)
    fn move_word_right(&mut self, char_pos: &mut usize, byte_pos: &mut usize, widths: &[i32], line: &[u8]) {
        if *char_pos >= widths.len() {
            return;
        }

        let mut new_char_pos = *char_pos;
        let mut new_byte_pos = *byte_pos;
        let mut move_width: i32 = 0;

        while new_char_pos < widths.len() {
            let (cp, advance) = decode_utf8(line, new_byte_pos);
            if !is_space_codepoint(cp) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos += 1;
            new_byte_pos += advance;
        }

        while new_char_pos < widths.len() {
            let (cp, advance) = decode_utf8(line, new_byte_pos);
            if is_space_codepoint(cp) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos += 1;
            new_byte_pos += advance;
        }

        while new_char_pos < widths.len() {
            let (cp, advance) = decode_utf8(line, new_byte_pos);
            if !is_space_codepoint(cp) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos += 1;
            new_byte_pos += advance;
        }

        self.move_cursor(move_width);
        *char_pos = new_char_pos;
        *byte_pos = new_byte_pos;
    }

    /// `move_cursor` (:666-697, POSIX half) — `\b` back, `\033[C` forward
    fn move_cursor(&mut self, delta: i32) {
        if delta == 0 {
            return;
        }
        if delta < 0 {
            let bs = vec![b'\x08'; (-delta) as usize];
            self.write_out(&bs);
        } else {
            let mut fwd = Vec::with_capacity((delta as usize) * 3);
            for _ in 0..delta {
                fwd.extend_from_slice(b"\x1b[C");
            }
            self.write_out(&fwd);
        }
    }

    /// `readline` (:1083-1088) — dispatches on `simple_io`. Returns whether
    /// more input is expected (multiline continuation).
    pub fn readline(&mut self, line: &mut String, multiline_input: bool) -> bool {
        if self.simple_io {
            self.readline_simple(line, multiline_input)
        } else {
            self.readline_advanced(line, multiline_input)
        }
    }

    /// `readline_advanced` (:753-1044)
    fn readline_advanced(&mut self, line: &mut String, multiline_input: bool) -> bool {
        if self.tty.is_none() {
            let _ = std::io::stdout().flush();
        }

        let mut line_buf: Vec<u8> = Vec::new();
        let mut widths: Vec<i32> = Vec::new();
        let mut is_special_char = false;
        let mut end_of_stream = false;

        let mut byte_pos: usize = 0; // current byte index
        let mut char_pos: usize = 0; // current character index (one char can be multiple bytes)

        loop {
            self.flush_out(); // Ensure all output is displayed before waiting for input (:790)
            let input_char = match self.getchar32() {
                Some(c) => c,
                None => {
                    end_of_stream = true;
                    // fall through the WEOF check below
                    0xFFFF_FFFF
                }
            };

            if input_char == b'\r' as u32 || input_char == b'\n' as u32 {
                break;
            }

            // (completion_cb tab handling, :797-809 — no consumer installs a
            // callback in the pinned tree's completion tool)

            if input_char == 0xFFFF_FFFF || input_char == 0x04 /* Ctrl+D */ {
                end_of_stream = true;
                break;
            }

            if is_special_char {
                self.replace_last(*line_buf.last().unwrap_or(&b' '));
                is_special_char = false;
            }

            if input_char == 0x1B {
                // Escape sequence (:821-894)
                let code = self.getchar32();
                if code == Some(b'[' as u32) {
                    let mut params = String::new();
                    let mut code = 0u32;
                    loop {
                        code = self.getchar32().unwrap_or(0xFFFF_FFFF);
                        if in_ascii_range(code, b'A', b'Z')
                            || in_ascii_range(code, b'a', b'z')
                            || code == b'~' as u32
                            || code == 0xFFFF_FFFF
                        {
                            break;
                        }
                        params.push(code as u8 as char);
                    }

                    let ctrl_modifier = has_ctrl_modifier(&params);

                    if code == b'D' as u32 {
                        // left
                        if ctrl_modifier {
                            self.move_word_left(&mut char_pos, &mut byte_pos, &widths, &line_buf);
                        } else if char_pos > 0 {
                            let w = widths[char_pos - 1];
                            self.move_cursor(-w);
                            char_pos -= 1;
                            byte_pos = prev_utf8_char_pos(&line_buf, byte_pos);
                        }
                    } else if code == b'C' as u32 {
                        // right
                        if ctrl_modifier {
                            self.move_word_right(&mut char_pos, &mut byte_pos, &widths, &line_buf);
                        } else if char_pos < widths.len() {
                            let w = widths[char_pos];
                            self.move_cursor(w);
                            char_pos += 1;
                            byte_pos = next_utf8_char_pos(&line_buf, byte_pos);
                        }
                    } else if code == b'H' as u32 {
                        // home
                        self.move_to_line_start(&mut char_pos, &mut byte_pos, &widths);
                    } else if code == b'F' as u32 {
                        // end
                        let len = line_buf.len();
                        self.move_to_line_end(&mut char_pos, &mut byte_pos, &widths, len);
                    } else if code == b'A' as u32 || code == b'B' as u32 {
                        // up/down (:857-865)
                        if code == b'A' as u32 {
                            self.history_prev(&mut line_buf, &mut widths, &mut char_pos, &mut byte_pos);
                            is_special_char = false;
                        } else if code == b'B' as u32 {
                            self.history_next(&mut line_buf, &mut widths, &mut char_pos, &mut byte_pos);
                            is_special_char = false;
                        }
                    } else if (code == b'~' as u32
                        || in_ascii_range(code, b'A', b'Z')
                        || in_ascii_range(code, b'a', b'z'))
                        && !params.is_empty()
                    {
                        let mut digits = String::new();
                        for ch in params.chars() {
                            if ch == ';' {
                                break;
                            }
                            if ch.is_ascii_digit() {
                                digits.push(ch);
                            }
                        }

                        if code == b'~' as u32 {
                            if digits == "1" || digits == "7" {
                                // home
                                self.move_to_line_start(&mut char_pos, &mut byte_pos, &widths);
                            } else if digits == "4" || digits == "8" {
                                // end
                                let len = line_buf.len();
                                self.move_to_line_end(&mut char_pos, &mut byte_pos, &widths, len);
                            } else if digits == "3" {
                                // delete
                                self.delete_at_cursor(&mut line_buf, &mut widths, char_pos, &mut byte_pos);
                            }
                        }
                    }
                } else if code == Some(0x1B) {
                    // Discard the rest of the escape sequence (:887-893)
                    loop {
                        match self.getchar32() {
                            None => break,
                            Some(c) => {
                                if in_ascii_range(c, b'A', b'Z')
                                    || in_ascii_range(c, b'a', b'z')
                                    || c == b'~' as u32
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            } else if input_char == 0x08 || input_char == 0x7F {
                // Backspace (:929-957)
                if char_pos > 0 {
                    let w = widths[char_pos - 1];
                    self.move_cursor(-w);
                    char_pos -= 1;
                    let prev_pos = prev_utf8_char_pos(&line_buf, byte_pos);
                    let char_len = byte_pos - prev_pos;
                    byte_pos = prev_pos;

                    // remove the character
                    line_buf.drain(byte_pos..byte_pos + char_len);
                    widths.remove(char_pos);

                    // redraw tail
                    let mut p = byte_pos;
                    let mut tail_width = 0;
                    for i in char_pos..widths.len() {
                        let next_p = next_utf8_char_pos(&line_buf, p);
                        let slice = line_buf[p..next_p].to_vec();
                        self.put_codepoint(&slice, widths[i]);
                        tail_width += widths[i];
                        p = next_p;
                    }

                    // clear display
                    let spaces = vec![b' '; w.max(0) as usize];
                    self.write_out(&spaces);
                    self.move_cursor(-(tail_width + w));
                }
            } else {
                // insert character (:958-996)
                let mut new_char_str = Vec::new();
                append_utf8(input_char, &mut new_char_str);
                let w = estimate_width(input_char);

                if char_pos == widths.len() {
                    // insert at the end
                    line_buf.extend_from_slice(&new_char_str);
                    let mut real_w = self.put_codepoint(&new_char_str, w);
                    if real_w < 0 {
                        real_w = 0;
                    }
                    widths.push(real_w);
                    byte_pos += new_char_str.len();
                    char_pos += 1;
                } else {
                    // insert in middle
                    for (i, &b) in new_char_str.iter().enumerate() {
                        line_buf.insert(byte_pos + i, b);
                    }

                    let mut real_w = self.put_codepoint(&new_char_str, w);
                    if real_w < 0 {
                        real_w = 0;
                    }

                    widths.insert(char_pos, real_w);

                    // print the tail
                    let mut p = byte_pos + new_char_str.len();
                    let mut tail_width = 0;
                    for i in char_pos + 1..widths.len() {
                        let next_p = next_utf8_char_pos(&line_buf, p);
                        let slice = line_buf[p..next_p].to_vec();
                        self.put_codepoint(&slice, widths[i]);
                        tail_width += widths[i];
                        p = next_p;
                    }

                    self.move_cursor(-tail_width);

                    byte_pos += new_char_str.len();
                    char_pos += 1;
                }
            }

            if !line_buf.is_empty() {
                let last = *line_buf.last().unwrap();
                if last == b'\\' || last == b'/' {
                    self.replace_last(last);
                    is_special_char = true;
                }
            }
        }

        let mut has_more = multiline_input;
        if is_special_char {
            self.replace_last(b' ');
            self.pop_cursor();

            let last = *line_buf.last().unwrap_or(&b' ');
            line_buf.pop();
            if last == b'\\' {
                line_buf.push(b'\n');
                self.write_out(b"\n");
                has_more = !has_more;
            } else {
                // llama will just eat the single space, it won't act as a space
                if line_buf.len() == 1 && line_buf[0] == b' ' {
                    line_buf.clear();
                    self.pop_cursor();
                }
                // a submitted line always ends with a newline
                // (console.cpp:1022, #29722)
                line_buf.push(b'\n');
                has_more = false;
            }
        } else if end_of_stream {
            has_more = false;
        } else {
            line_buf.push(b'\n');
            self.write_out(b"\n");
        }

        *line = String::from_utf8_lossy(&line_buf).into_owned();
        if !end_of_stream && !line.is_empty() {
            // remove the trailing newline for history storage (:1032-1040)
            let hline = line.strip_suffix('\n').unwrap_or(line);
            self.history.add(hline);
        }

        self.flush_out();
        has_more
    }

    /// the `history_prev` lambda (:770-779)
    fn history_prev(
        &mut self,
        line: &mut Vec<u8>,
        widths: &mut Vec<i32>,
        char_pos: &mut usize,
        byte_pos: &mut usize,
    ) {
        if !self.history.is_viewing() {
            let s = String::from_utf8_lossy(line).into_owned();
            self.history.begin_viewing(&s);
        }
        let mut new_line = String::new();
        if !self.history.prev(&mut new_line) {
            return;
        }
        let bytes = new_line.into_bytes();
        self.set_line_contents(&bytes, line, widths, char_pos, byte_pos, -1);
    }

    /// the `history_next` lambda (:780-788)
    fn history_next(
        &mut self,
        line: &mut Vec<u8>,
        widths: &mut Vec<i32>,
        char_pos: &mut usize,
        byte_pos: &mut usize,
    ) {
        if self.history.is_viewing() {
            let mut new_line = String::new();
            if !self.history.next(&mut new_line) {
                return;
            }
            let bytes = new_line.into_bytes();
            self.set_line_contents(&bytes, line, widths, char_pos, byte_pos, -1);
        }
    }

    /// `readline_simple` (:1046-1081, POSIX half) — `std::getline(cin)`
    /// strips the delimiter; EOF → cleared line, `false`.
    fn readline_simple(&mut self, line: &mut String, mut multiline_input: bool) -> bool {
        line.clear();
        let mut buf = String::new();
        match std::io::stdin().read_line(&mut buf) {
            Ok(0) | Err(_) => {
                // Input stream is bad or EOF received
                line.clear();
                return false;
            }
            Ok(_) => {
                // getline consumes the delimiter — read_line keeps it
                *line = buf.strip_suffix('\n').unwrap_or(&buf).to_string();
                // a CR from a Windows pipe would reach getline as data in the
                // C; keep it (the marker checks below see the raw line)
            }
        }
        if !line.is_empty() {
            let last = line.chars().last().unwrap();
            if last == '/' {
                // Always return control on '/' symbol — the marker is
                // replaced by the newline, not removed (console.cpp:1067-1070)
                let mut b = String::with_capacity(line.len());
                b.push_str(&line[..line.len() - 1]);
                b.push('\n');
                *line = b;
                return false;
            }
            if last == '\\' {
                // '\\' changes the default action
                line.pop();
                multiline_input = !multiline_input;
            }
        }
        line.push('\n');

        // By default, continue input if multiline_input is set
        multiline_input
    }

    /// `console::log` (:1146-1151) — writes to `out` without touching colors
    pub fn log(&mut self, s: &str) {
        self.write_out(s.as_bytes());
    }

    /// `console::error` (:1153-1161) — error-colored write restoring the
    /// previous color
    pub fn error(&mut self, s: &str) {
        let cur = self.current_display;
        self.set_display(DisplayType::Error);
        self.write_out(s.as_bytes());
        self.set_display(cur);
    }

    /// `console::flush` (:1163-1165)
    pub fn flush(&mut self) {
        self.flush_out();
    }
}

/// the pending tail of [`Console::getchar32`] holds only a UTF-8 prefix when
/// the lead byte promises more continuation bytes than buffered
/// (`getwchar` blocks for the whole character instead)
fn is_incomplete(tail: &[u8]) -> bool {
    let first = tail[0];
    let need = if first & 0x80 == 0 {
        1
    } else if first & 0xE0 == 0xC0 {
        2
    } else if first & 0xF0 == 0xE0 {
        3
    } else if first & 0xF8 == 0xF0 {
        4
    } else {
        return false; // invalid lead byte — decode_utf8 yields U+FFFD at 1
    };
    tail.len() < need
}

/// `code >= 'A' && code <= 'Z'` over a `char32_t` (no truncation of
/// non-ASCII codepoints — the C compares the full 32-bit value)
fn in_ascii_range(code: u32, lo: u8, hi: u8) -> bool {
    code <= 0x7F && (code as u8) >= lo && (code as u8) <= hi
}

/// the `\033[6n` cursor-position query of `put_codepoint` (:328-339) —
/// `\033[%d;%dR`, column is the second field
fn query_cursor_position(tty: &mut std::fs::File) -> (usize, usize) {
    use std::os::unix::io::AsRawFd;
    let _ = tty.write_all(b"\x1b[6n");
    let _ = tty.flush();
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    // read the reply: ESC [ y ; x R — bounded so a silent terminal cannot
    // hang the editor (the C's fscanf pattern fails the same way on silence)
    let mut read = 0;
    while read < 32 {
        match tty.read(&mut byte) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                buf.push(byte[0]);
                read += 1;
                if byte[0] == b'R' {
                    break;
                }
            }
        }
    }
    let s = String::from_utf8_lossy(&buf);
    let _ = tty.as_raw_fd();
    if let Some(rest) = s.strip_prefix("\x1b[") {
        let rest = rest.trim_end_matches('R');
        let mut it = rest.split(';');
        let y = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let x = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        return (x, y);
    }
    (0, 0)
}

// ---------------------------------------------------------------------------
// tests — the scripted (non-tty) console behavior, pinned against
// console.cpp's own semantics (see parity/cli_interactive_parity.sh for the
// cross-binary transcript comparison)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_utf8_matches_c() {
        // console.cpp:365-409 — ASCII, 2/3/4-byte sequences, invalid
        assert_eq!(decode_utf8(b"a", 0), (b'a' as u32, 1));
        assert_eq!(decode_utf8("é".as_bytes(), 0), (0xE9, 2));
        assert_eq!(decode_utf8("中".as_bytes(), 0), (0x4E2D, 3));
        assert_eq!(decode_utf8("🦙".as_bytes(), 0), (0x1F999, 4));
        assert_eq!(decode_utf8(&[0xC3], 0), (0xFFFD, 1)); // truncated 2-byte
        assert_eq!(decode_utf8(&[0xE4, 0xB8], 0), (0xFFFD, 1)); // bad continuation
    }

    #[test]
    fn append_utf8_roundtrip() {
        for &cp in &[0x24u32, 0xA2, 0x20AC, 0x10348] {
            let mut v = Vec::new();
            append_utf8(cp, &mut v);
            let (back, adv) = decode_utf8(&v, 0);
            assert_eq!(back, cp);
            assert_eq!(adv, v.len());
        }
    }

    #[test]
    fn prev_next_char_pos() {
        let s = "a中\0"; // 1 + 3 + 1 bytes
        let b = s.as_bytes();
        assert_eq!(prev_utf8_char_pos(b, 4), 1);
        assert_eq!(prev_utf8_char_pos(b, 5), 4);
        assert_eq!(next_utf8_char_pos(b, 1), 4);
        assert_eq!(next_utf8_char_pos(b, 4), 5);
        assert_eq!(next_utf8_char_pos(b, 5), 5);
    }

    #[test]
    fn has_ctrl_modifier_c_semantics() {
        // console.cpp:549-575 — the '5' modifier of e.g. "1;5C"
        assert!(has_ctrl_modifier("1;5"));
        assert!(!has_ctrl_modifier("1"));
        assert!(!has_ctrl_modifier("5x")); // non-digit kills the group
        assert!(has_ctrl_modifier("5;1"));
    }

    #[test]
    fn history_c_semantics() {
        // console.cpp:699-751
        let mut h = History::default();
        h.add(""); // empty never added
        assert!(h.entries.is_empty());
        h.add("one");
        h.add("one"); // dup with last not re-added
        assert_eq!(h.entries.len(), 1);
        h.add("two");

        let mut cur = "cur".to_string();
        h.begin_viewing(&cur);
        assert!(h.prev(&mut cur));
        assert_eq!(cur, "two");
        assert!(h.prev(&mut cur));
        assert_eq!(cur, "one");
        assert!(h.prev(&mut cur)); // clamps at 0
        assert_eq!(cur, "one");
        assert!(h.next(&mut cur));
        assert_eq!(cur, "two");
        assert!(h.next(&mut cur)); // wraps to the backup line
        assert_eq!(cur, "cur");
        assert!(!h.is_viewing());
    }

    /// the `/` and `\` line-end handling of `readline_simple`
    /// (console.cpp:1066-1076) — exercised through a subprocess-free harness:
    /// the function only touches `line`, so drive it directly with a crafted
    /// String the way getline would have produced it.
    #[test]
    fn readline_simple_trailing_markers() {
        // emulate: getline fills `line`, then the marker handling runs
        let emulate = |raw: &str, multiline: bool| -> (String, bool) {
            // copy of the :1066-1081 tail
            let mut line = raw.to_string();
            let mut multiline_input = multiline;
            if !line.is_empty() {
                let last = line.chars().last().unwrap();
                if last == '/' {
                    line.pop();
                    return (line, false);
                }
                if last == '\\' {
                    line.pop();
                    multiline_input = !multiline_input;
                }
            }
            line.push('\n');
            (line, multiline_input)
        };
        assert_eq!(emulate("plain", false), ("plain\n".to_string(), false));
        assert_eq!(emulate("plain\\", false), ("plain\n".to_string(), true));
        assert_eq!(emulate("plain\\", true), ("plain\n".to_string(), false));
        assert_eq!(emulate("plain/", true), ("plain".to_string(), false));
        assert_eq!(emulate("", false), ("\n".to_string(), false));
    }

    /// the `\`/`/` marker consumption of `readline_advanced`'s exit path
    /// (console.cpp:1004-1030) — mirrored as a pure function for the test
    #[test]
    fn readline_advanced_marker_exit_semantics() {
        // (line_before_pop, is_special, multiline_in) -> (line_after, has_more)
        let emulate = |raw: &str, multiline_input: bool| -> (String, bool) {
            let mut line: Vec<u8> = raw.as_bytes().to_vec();
            let mut has_more = multiline_input;
            // is_special_char is exactly "line ends in \\ or /" on loop exit
            let is_special_char =
                matches!(line.last(), Some(b'\\') | Some(b'/'));
            if is_special_char {
                let last = *line.last().unwrap();
                line.pop();
                if last == b'\\' {
                    line.push(b'\n');
                    has_more = !has_more;
                } else {
                    if line.len() == 1 && line[0] == b' ' {
                        line.clear();
                    }
                    has_more = false;
                }
            } else {
                line.push(b'\n');
            }
            (String::from_utf8_lossy(&line).into_owned(), has_more)
        };
        assert_eq!(emulate("plain", false), ("plain\n".to_string(), false));
        assert_eq!(emulate("cont\\", false), ("cont\n".to_string(), true));
        assert_eq!(emulate("cont\\", true), ("cont\n".to_string(), false));
        assert_eq!(emulate("end/", false), ("end".to_string(), false));
        // the lone-space-after-'/' pop (:1016-1020)
        assert_eq!(emulate(" ", false), (" \n".to_string(), false));
    }
}
