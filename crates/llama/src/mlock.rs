//! mlock.rs — port of `llama_mlock` (llama-mmap.cpp:654-818: the POSIX
//! `impl` + the pimpl wrapper at :810-816) — the `--mlock` workflow.
//!
//! Mapping:
//!   llama_mlock::impl::lock_granularity (:669-671) -> page size (4 KiB x86;
//!       the C rounds up to whole pages, the grow_to round-up below covers it)
//!   llama_mlock::impl::raw_lock      (:683-719) -> [`Mlock::raw_lock`]
//!   llama_mlock::impl::raw_unlock    (:723-727) -> [`Mlock::raw_unlock`]
//!   llama_mlock::impl::init/grow_to  (:799-818) -> [`Mlock::init`]/[`Mlock::grow_to`]
//!   llama_mlock::SUPPORTED           (:811-816) -> [`Mlock::SUPPORTED`]
//!
//! Deviations (documented):
//!   * no libc crate in the workspace — the mlock/munlock/getrlimit symbols
//!     are declared `extern "C"` directly (same lib the C links).
//!   * the C keeps `addr`/`size` as raw pointers of *backend buffers or
//!     mappings*; the port's only lockable memory is the weights `Mmap`, so
//!     init takes the mapping's base pointer/length. `--mlock` on the
//!     reference with mmap (the port's constant load mode, PARITY:748) locks
//!     the whole mapping progressively as `load_all_data` reads each tensor
//!     (llama-model-loader.cpp:1661-1663 `grow_to(weight->offs + n_size)` —
//!     monotonically growing to the mapping end); the port locks the mapping
//!     once, to its full length, at the same point in the load order.

use std::ffi::c_void;

// llama-mmap.cpp:700-702 (POSIX branch)
extern "C" {
    fn mlock(addr: *const c_void, len: usize) -> i32;
    fn munlock(addr: *const c_void, len: usize) -> i32;
    fn __errno_location() -> *mut i32;
}

fn errno() -> i32 {
    unsafe { *__errno_location() }
}

fn strerror(err: i32) -> String {
    // std::strerror equivalent — the C uses strerror(3) text verbatim
    extern "C" {
        fn strerror(errnum: i32) -> *const std::ffi::c_char;
    }
    unsafe { std::ffi::CStr::from_ptr(strerror(err)).to_string_lossy().into_owned() }
}

// the resource-limit suggestion text (llama-mmap.cpp:695-698)
const MLOCK_SUGGESTION: &str = "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n";

const RLIMIT_MEMLOCK: i32 = 8;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct RLimit {
    rlim_cur: u64,
    rlim_max: u64,
}

extern "C" {
    fn getrlimit(resource: i32, rlim: *mut RLimit) -> i32;
}

/// `llama_mlock` — pins an address range into RAM, growing the locked span
/// as more of it is needed. Mirrors the C's `failed_already` latch: after a
/// failed lock the object goes inert instead of retrying.
pub struct Mlock {
    addr: *mut u8,
    size: usize,
    failed_already: bool,
}

// SAFETY: the C mlock object is just (addr, size); the port hands it across
// threads exactly like the C's llama_mlocks vector.
unsafe impl Send for Mlock {}

impl Mlock {
    /// `llama_mlock::SUPPORTED` (llama-mmap.cpp:811-816) — `_POSIX_MEMLOCK_RANGE`.
    pub const SUPPORTED: bool = true;

    pub fn new() -> Self {
        Mlock { addr: std::ptr::null_mut(), size: 0, failed_already: false }
    }

    /// `impl::init` (llama-mmap.cpp:799-802).
    pub fn init(&mut self, ptr: *mut u8) {
        assert!(self.addr.is_null() && self.size == 0);
        self.addr = ptr;
    }

    fn page_size() -> usize {
        // impl::lock_granularity (:669-671) = sysconf(_SC_PAGESIZE)
        4096
    }

    /// `impl::raw_lock` (llama-mmap.cpp:683-719) — locks `[addr, addr+len)`
    /// and, on failure, logs the C's warning (with the RLIMIT_MEMLOCK
    /// suggestion when errno == ENOMEM and the soft limit could cover it).
    fn raw_lock(&mut self, addr: *const c_void, len: usize) -> bool {
        if unsafe { mlock(addr, len) } == 0 {
            return true;
        }
        let err = errno();
        let errmsg = strerror(err);
        let mut suggest = err == 12 /* ENOMEM */;
        if suggest {
            let mut lim = RLimit::default();
            if unsafe { getrlimit(RLIMIT_MEMLOCK, &mut lim) } != 0 {
                suggest = false;
            }
            if suggest && lim.rlim_max > lim.rlim_cur + len as u64 {
                suggest = false;
            }
        }
        crate::llama_log_warn!(
            "warning: failed to mlock {}-byte buffer (after previously locking {} bytes): {}\n{}",
            len,
            self.size,
            errmsg,
            if suggest { MLOCK_SUGGESTION } else { "" }
        );
        false
    }

    /// `impl::raw_unlock` (llama-mmap.cpp:723-727).
    #[allow(dead_code)]
    fn raw_unlock(&mut self, addr: *const c_void, len: usize) {
        if unsafe { munlock(addr, len) } != 0 {
            crate::llama_log_warn!(
                "warning: failed to munlock buffer: {}\n",
                strerror(errno())
            );
        }
    }

    /// `impl::grow_to` (llama-mmap.cpp:804-818): round the target up to the
    /// lock granularity and lock the extension; a failure latches
    /// `failed_already`.
    pub fn grow_to(&mut self, target_size: usize) {
        assert!(!self.addr.is_null());
        if self.failed_already {
            return;
        }
        let granularity = Self::page_size();
        let target_size = (target_size + granularity - 1) & !(granularity - 1);
        if target_size > self.size {
            if self.raw_lock(unsafe { self.addr.add(self.size) } as *const c_void, target_size - self.size) {
                self.size = target_size;
            } else {
                self.failed_already = true;
            }
        }
    }
}

impl Default for Mlock {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Mlock {
    /// The C unlocks in `impl::~impl` via the destructor's member destructors
    /// (raw_unlock of the locked span).
    fn drop(&mut self) {
        if self.size > 0 {
            self.raw_unlock(self.addr as *const c_void, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock a private buffer and grow it — the plain success path; verifies
    /// the granularity round-up and the size accounting (llama-mmap.cpp:804-818).
    #[test]
    fn mlock_grow_to_locks_pages() {
        let mut buf = vec![0u8; 10000];
        let mut m = Mlock::new();
        m.init(buf.as_mut_ptr());
        m.grow_to(5000);
        assert!(m.size >= 8192, "rounded to two 4-KiB pages, got {}", m.size);
        m.grow_to(9000);
        assert!(m.size >= 12288);
        // a failed_already latch is only reachable without CAP_IPC_LOCK room;
        // not asserted here (machine-dependent).
    }
}
