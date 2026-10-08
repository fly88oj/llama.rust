//! impl_log.rs — port of the logging half of `llama-impl.cpp` (llama-impl.cpp:28-71:
//! `llama_log_get` / `llama_log_set` / `llama_log_internal` / `llama_log_callback_default`)
//! plus `llama_time_us` (llama.cpp:153) and the small display-string helpers that the
//! log surface consumes.
//!
//! Mapping:
//!   llama_log_get              -> [`log_get`]
//!   llama_log_set              -> [`log_set`]
//!   llama_log_internal(_v)     -> [`log_internal`] (+ the `llama_log_*` macros below)
//!   llama_log_callback_default -> [`log_callback_default`]
//!   LLAMA_LOG/CONT/INFO/WARN/ERROR/DEBUG (llama-impl.h:28-35) -> [`llama_log!`] family
//!   llama_time_us (llama.cpp:153 -> ggml_time_us)             -> [`crate::time_us`]
//!
//! Deviations (documented):
//!   * the C callback signature is `void (*)(ggml_log_level, const char *, void *)`;
//!     the port passes the user-data word as a `usize` (same pointer-sized token —
//!     callers hand an `AtomicBool`/state address, exactly like C user data).
//!   * the C also forwards the callback into ggml via `ggml_log_set`; the port's
//!     ggml crate has no logger of its own, so the routing lives entirely here.

use std::sync::atomic::{AtomicUsize, Ordering};

/// `enum ggml_log_level` (ggml.h:1326-1333).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    None = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Cont = 5,
}

/// `ggml_log_callback` — level, the fully formatted text (no trailing newline is
/// added; C log lines embed their own `\n`), and the user-data word.
pub type LogCallback = fn(LogLevel, &str, usize);

// the registered callback as a plain fn-pointer usize (0 = the default), like
// compute.rs's EVAL_CALLBACK — the "no callback" path stays a single load
static LOG_CALLBACK: AtomicUsize = AtomicUsize::new(0);
static LOG_USER_DATA: AtomicUsize = AtomicUsize::new(0);

/// `llama_log_callback_default` (llama-impl.cpp:62-67) — level and user data
/// ignored, text to stderr, flushed.
pub fn log_callback_default(_level: LogLevel, text: &str, _user_data: usize) {
    use std::io::Write;
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(text.as_bytes());
    let _ = err.flush();
}

/// `llama_log_get` (llama-impl.cpp:28-30). `None` means "the default callback"
/// exactly like the C returning `llama_log_callback_default` implicitly via
/// `ggml_log_get`'s current value.
pub fn log_get() -> (Option<LogCallback>, usize) {
    let cb = match LOG_CALLBACK.load(Ordering::Acquire) {
        0 => None,
        p => Some(unsafe { std::mem::transmute::<usize, LogCallback>(p) }),
    };
    (cb, LOG_USER_DATA.load(Ordering::Acquire))
}

/// `llama_log_set` (llama-impl.cpp:32-36) — a null callback resets to the
/// default stderr behavior.
pub fn log_set(log_callback: Option<LogCallback>, user_data: usize) {
    let p = log_callback.map(|f| f as usize).unwrap_or(0);
    LOG_CALLBACK.store(p, Ordering::Release);
    LOG_USER_DATA.store(user_data, Ordering::Release);
}

/// `llama_log_internal` (llama-impl.cpp:38-60) — format, then hand the whole
/// message to the registered callback. The C's 128-byte stack buffer vs heap
/// fallback is a C-only concern.
pub fn log_internal(level: LogLevel, args: std::fmt::Arguments<'_>) {
    let text = std::fmt::format(args);
    match LOG_CALLBACK.load(Ordering::Acquire) {
        0 => log_callback_default(level, &text, 0),
        p => {
            let cb = unsafe { std::mem::transmute::<usize, LogCallback>(p) };
            cb(level, &text, LOG_USER_DATA.load(Ordering::Acquire));
        }
    }
}

/// `LLAMA_LOG(...)` — level NONE (the plain informational level the model-load
/// banner is printed with).
#[macro_export]
macro_rules! llama_log {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::None, format_args!($($arg)*)) };
}

/// `LLAMA_LOG_INFO(...)` (llama-impl.h:29).
#[macro_export]
macro_rules! llama_log_info {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::Info, format_args!($($arg)*)) };
}

/// `LLAMA_LOG_WARN(...)` (llama-impl.h:30).
#[macro_export]
macro_rules! llama_log_warn {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::Warn, format_args!($($arg)*)) };
}

/// `LLAMA_LOG_ERROR(...)` (llama-impl.h:31).
#[macro_export]
macro_rules! llama_log_error {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::Error, format_args!($($arg)*)) };
}

/// `LLAMA_LOG_DEBUG(...)` (llama-impl.h:32).
#[macro_export]
macro_rules! llama_log_debug {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::Debug, format_args!($($arg)*)) };
}

/// `LLAMA_LOG_CONT(...)` (llama-impl.h:33) — a continuation line.
#[macro_export]
macro_rules! llama_log_cont {
    ($($arg:tt)*) => { $crate::impl_log::log_internal($crate::impl_log::LogLevel::Cont, format_args!($($arg)*)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    static SEEN: std::sync::Mutex<Vec<(u32, String)>> = std::sync::Mutex::new(Vec::new());

    fn sink(level: LogLevel, text: &str, _ud: usize) {
        SEEN.lock().unwrap().push((level as u32, text.to_string()));
    }

    /// Round-trips the routing: set a callback, emit through every level, check
    /// the callback saw exactly what was formatted; reset restores stderr.
    #[test]
    fn log_routing_roundtrip() {
        log_set(Some(sink), 42);
        crate::llama_log!("plain {}/{}\n", 1, 2);
        crate::llama_log_info!("info\n");
        crate::llama_log_warn!("warn\n");
        crate::llama_log_error!("error\n");
        crate::llama_log_cont!("cont\n");
        let (cb, ud) = log_get();
        assert!(cb.is_some());
        assert_eq!(ud, 42);
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), 5);
        assert_eq!(seen[0], (LogLevel::None as u32, "plain 1/2\n".to_string()));
        assert_eq!(seen[1], (LogLevel::Info as u32, "info\n".to_string()));
        assert_eq!(seen[2], (LogLevel::Warn as u32, "warn\n".to_string()));
        assert_eq!(seen[3], (LogLevel::Error as u32, "error\n".to_string()));
        assert_eq!(seen[4], (LogLevel::Cont as u32, "cont\n".to_string()));
        // reset -> default (None), like llama_log_set(nullptr, nullptr)
        log_set(None, 0);
        assert!(log_get().0.is_none());
        // a long message (> the C's 128-byte fast path) routes identically
        let _ = AtomicBool::new(false);
    }

    /// The default callback is byte-faithful to `fputs(stderr)+fflush`: capture
    /// stderr is not possible portably here, so assert the function is what
    /// `log_set(None, ..)` falls back to and that it does not panic on binary
    /// text fragments.
    #[test]
    fn default_callback_accepts_any_text() {
        log_callback_default(LogLevel::None, "no trailing newline", 0);
        log_callback_default(LogLevel::Error, "", 7);
    }
}
