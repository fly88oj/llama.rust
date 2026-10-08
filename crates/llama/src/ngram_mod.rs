//! ngram-mod — port of `common/ngram-mod.cpp` / `common/ngram-mod.h`
//! (llama.cpp bd4f514db1): the modular-hash n-gram container of ref
//! https://github.com/ggml-org/llama.cpp/pull/19164 — a fixed-size open
//! table keyed by the LCG-style hash of `n` tokens, each slot holding the
//! token that followed the n-gram (`ngram-mod`'s `--spec-type ngram-mod`).
//!
//! Used by the `ngram-mod` speculative implementation
//! (`common_speculative_impl_ngram_mod`, speculative.cpp:1869-2042, ported
//! in `crate::speculative`).

/// `struct common_ngram_mod` (ngram-mod.h:13-38). `entry_t` is `int32_t`
/// (the port's `llama_token`), `EMPTY = -1`.
pub type Entry = i32;

/// `common_ngram_mod::EMPTY` (ngram-mod.h:16)
pub const EMPTY: Entry = -1;

pub struct CommonNgramMod {
    /// ngram size to hash
    n: usize,
    /// number of used slots
    used: usize,
    /// the open table — `EMPTY` when free
    entries: Vec<Entry>,
}

impl CommonNgramMod {
    /// `common_ngram_mod(uint16_t n, size_t size)` (ngram-mod.cpp:9-13) —
    /// the reference sizes it `4*1024*1024` slots (speculative.cpp:1896)
    pub fn new(n: u16, size: usize) -> Self {
        let mut m = CommonNgramMod {
            n: n as usize,
            used: 0,
            entries: vec![EMPTY; size],
        };
        m.reset();
        m
    }

    /// `idx` (ngram-mod.cpp:15-25):
    /// `res = (res*6364136223846793005 + tokens[i]) % entries.size()` over
    /// the first `n` tokens
    pub fn idx(&self, tokens: &[Entry]) -> usize {
        let mut res: u64 = 0;
        for i in 0..self.n {
            res = res
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(tokens[i] as i64 as u64);
        }

        (res % self.entries.len() as u64) as usize
    }

    /// `add` (ngram-mod.cpp:27-35): store `tokens[n]` (the token after the
    /// n-gram) at the n-gram's slot
    pub fn add(&mut self, tokens: &[Entry]) {
        let i = self.idx(tokens);

        if self.entries[i] == EMPTY {
            self.used += 1;
        }

        self.entries[i] = tokens[self.n];
    }

    /// `get` (ngram-mod.cpp:37-41): the stored follower, `EMPTY` (the C's
    /// -1) when the slot is free
    pub fn get(&self, tokens: &[Entry]) -> Entry {
        let i = self.idx(tokens);

        self.entries[i]
    }

    /// `reset` (ngram-mod.cpp:43-46)
    pub fn reset(&mut self) {
        self.entries.fill(EMPTY);
        self.used = 0;
    }

    /// `get_n` (ngram-mod.cpp:48-50)
    pub fn get_n(&self) -> usize {
        self.n
    }

    /// `get_used` (ngram-mod.cpp:52-54)
    pub fn get_used(&self) -> usize {
        self.used
    }

    /// `size` (ngram-mod.cpp:56-58)
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// `size_bytes` (ngram-mod.cpp:60-62)
    pub fn size_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<Entry>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_get_roundtrip_and_reset() {
        // ngram-mod.cpp:9-62 — a size-2 container: the n-gram (5, 6) stores
        // its follower 7; a different n-gram misses; reset clears
        let mut m = CommonNgramMod::new(2, 1024);
        assert_eq!(m.get_n(), 2);
        assert_eq!(m.get_used(), 0);

        m.add(&[5, 6, 7]);
        assert_eq!(m.get(&[5, 6, 0]), 7, "the follower of the key (5, 6)");
        assert_eq!(m.get(&[5, 0, 0]), EMPTY, "a different key misses");
        assert_eq!(m.get_used(), 1);

        // re-adding the same key keeps one used slot
        m.add(&[5, 6, 8]);
        assert_eq!(m.get_used(), 1);
        assert_eq!(m.get(&[5, 6, 0]), 8, "the latest follower wins");

        m.reset();
        assert_eq!(m.get_used(), 0);
        assert_eq!(m.get(&[5, 6, 0]), EMPTY);
        assert_eq!(m.size_bytes(), 1024 * 4);
    }
}
