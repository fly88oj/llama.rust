//! ngram-cache — port of `common/ngram-cache.cpp` / `common/ngram-cache.h`
//! (llama.cpp bd4f514db1): the lookup-decoding n-gram cache
//! (`--spec-type ngram-cache`), an empirical map from n-grams (1..4 tokens)
//! to the distribution of the tokens that followed them.
//!
//! Used by the `ngram-cache` speculative implementation
//! (`common_speculative_impl_ngram_cache`, speculative.cpp:2044-2181, ported
//! in `crate::speculative`). Serialization keeps the reference's binary
//! format byte-for-byte (`common_ngram_cache_save` / `_load`,
//! ngram-cache.cpp:200-258), so a cache file written by the reference's
//! `llama-lookup-*` tools loads here and vice versa.

use std::collections::HashMap;
use std::io::{Read, Write};

/// `LLAMA_NGRAM_MIN` (ngram-cache.h:9)
pub const LLAMA_NGRAM_MIN: usize = 1;
/// `LLAMA_NGRAM_MAX` (ngram-cache.h:10)
pub const LLAMA_NGRAM_MAX: usize = 4;
/// `LLAMA_NGRAM_STATIC` (ngram-cache.h:11)
pub const LLAMA_NGRAM_STATIC: usize = 2;

/// `LLAMA_TOKEN_NULL` (llama.h) — the port's token ids are plain i32
pub const LLAMA_TOKEN_NULL: i32 = -1;

/// `struct common_ngram` (ngram-cache.h:15-38): LLAMA_NGRAM_MAX tokens,
/// `LLAMA_TOKEN_NULL`-padded past `ngram_size`. `PartialEq`/`Hash` follow the
/// C's `operator==` / `common_ngram_hash_function` (the Fibonacci-hashing
/// XOR fold over `token * 11400714819323198485`).
#[derive(Clone, Debug)]
pub struct CommonNgram {
    pub tokens: [i32; LLAMA_NGRAM_MAX],
}

impl CommonNgram {
    /// `common_ngram(const llama_token * input, const int ngram_size)`
    /// (ngram-cache.h:24-28)
    pub fn new(input: &[i32], ngram_size: usize) -> Self {
        let mut tokens = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        for (i, t) in tokens.iter_mut().enumerate() {
            *t = if i < ngram_size {
                input[i]
            } else {
                LLAMA_TOKEN_NULL
            };
        }
        CommonNgram { tokens }
    }
}

impl PartialEq for CommonNgram {
    /// `operator==` (ngram-cache.h:30-37)
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens
    }
}
impl Eq for CommonNgram {}

/// `common_token_hash_function` (ngram-cache.h:40-45):
/// `token * 11400714819323198485llu` — the 64-bit product truncated like the
/// C's `size_t` (LP64) assignment.
fn token_hash(token: i32) -> u64 {
    (token as i64 as u64).wrapping_mul(11400714819323198485u64)
}

impl std::hash::Hash for CommonNgram {
    /// `common_ngram_hash_function` (ngram-cache.h:47-55)
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut hash = token_hash(self.tokens[0]);
        for &t in &self.tokens[1..] {
            hash ^= token_hash(t);
        }
        state.write_u64(hash);
    }
}

/// `common_ngram_cache_part` (ngram-cache.h:58): token -> number of times
/// seen
pub type CommonNgramCachePart = HashMap<i32, i32>;

/// `common_ngram_cache` (ngram-cache.h:61): n-gram -> empirical distribution
/// of following tokens
pub type CommonNgramCache = HashMap<CommonNgram, CommonNgramCachePart>;

/// `common_ngram_cache_update` (ngram-cache.cpp:12-52): fold every n-gram of
/// `ngram_min..=ngram_max` size that ends inside the last `nnew` tokens of
/// `inp_data` into the cache. `inp_data` may only be appended to between
/// calls (ngram-cache.h:71-72); the `print_progress` ETA branch (:42-49) is
/// ported as a no-op print every 10M n-grams.
pub fn common_ngram_cache_update(
    ngram_cache: &mut CommonNgramCache,
    ngram_min: usize,
    ngram_max: usize,
    inp_data: &[i32],
    nnew: usize,
    print_progress: bool,
) {
    let inp_size = inp_data.len() as i64;

    let mut n_done: i64 = 0;
    let mut last_report: i64 = 0;

    for ngram_size in ngram_min as i64..=ngram_max as i64 {
        // :21 — start so the n-gram fits before the new region
        let i_start = (inp_size - nnew as i64).max(ngram_size);
        for i in i_start..inp_size {
            let ngram_start = (i - ngram_size) as usize;
            let ngram = CommonNgram::new(&inp_data[ngram_start..], ngram_size as usize);
            let token = inp_data[i as usize];

            // :27-39 — find-or-insert the part (:29-31), then the token
            // count (:33-38)
            let part = ngram_cache.entry(ngram).or_default();
            *part.entry(token).or_insert(0) += 1;
            n_done += 1;

            // :42-49 — the reference's ETA print (silent unless asked)
            if print_progress && n_done - last_report >= 10_000_000 {
                last_report = n_done;
                eprintln!(
                    "common_ngram_cache_update: {n_done}/{} done",
                    inp_size * (ngram_max as i64 - ngram_min as i64 + 1)
                );
            }
        }
    }
}

/// `get_token` (ngram-cache.cpp:55-57): token `i` of the combined
/// `[inp | draft[1..]]` sequence — `draft[0]` is the previously sampled
/// token and belongs to `inp`'s continuation, so the draft side is indexed
/// with a +1 shift.
fn get_token(inp: &[i32], draft: &[i32], i: usize) -> i32 {
    if i < inp.len() {
        inp[i]
    } else {
        draft[1 + i - inp.len()]
    }
}

/// If sample size or percentage are below these thresholds the draft is
/// aborted early (ngram-cache.cpp:60-63). Indexed by n-gram size - 1.
const DRAFT_MIN_SAMPLE_SIZE_LAX: [i32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];
const DRAFT_MIN_PERCENT_LAX: [i32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];
const DRAFT_MIN_SAMPLE_SIZE_STRICT: [i32; LLAMA_NGRAM_MAX] = [4, 3, 2, 2];
const DRAFT_MIN_PERCENT_STRICT: [i32; LLAMA_NGRAM_MAX] = [75, 66, 66, 66];

/// `try_draft` from only the static ngram cache (ngram-cache.cpp:66-95):
/// the argmax token of `ngram_static`'s part, gated by the lax thresholds of
/// `LLAMA_NGRAM_STATIC`-sized n-grams.
fn try_draft_static(nc_static: &CommonNgramCache, ngram_static: &CommonNgram) -> i32 {
    let Some(part_static) = nc_static.get(ngram_static) else {
        return LLAMA_TOKEN_NULL;
    };

    let mut max_count_static = 0;
    let mut sum_count_static = 0;
    let mut max_token = LLAMA_TOKEN_NULL;

    // :77-86 — first strict-max count wins ties (the iteration order of the
    // C's unordered_map; only the counts matter for the thresholds, and the
    // max token is deterministic when a unique maximum exists)
    for (&token, &count_static) in part_static {
        if count_static > max_count_static {
            max_token = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    // :88-93
    if sum_count_static < DRAFT_MIN_SAMPLE_SIZE_LAX[LLAMA_NGRAM_STATIC - 1] {
        return LLAMA_TOKEN_NULL;
    }
    if 100 * max_count_static < DRAFT_MIN_PERCENT_LAX[LLAMA_NGRAM_STATIC - 1] * sum_count_static {
        return LLAMA_TOKEN_NULL;
    }
    max_token
}

/// `try_draft` from a primary cache (context/dynamic) validated against the
/// static part (ngram-cache.cpp:98-144): walk the candidate n-grams from
/// longest to shortest, pick the token maximizing
/// `count_primary * count_static` (static count defaults to 1 — the C's
/// `100*count` / the matching `100*max_count_static` scale cancels), gated
/// by the caller's thresholds.
fn try_draft_primary(
    nc_primary: &CommonNgramCache,
    ngrams_primary: &[CommonNgram],
    part_static: &CommonNgramCachePart,
    min_sample_size: &[i32; LLAMA_NGRAM_MAX],
    min_percent: &[i32; LLAMA_NGRAM_MAX],
) -> i32 {
    let mut drafted_token = LLAMA_TOKEN_NULL;

    // :104 — longest n-gram first, stop at the first accepted candidate
    for i in (0..ngrams_primary.len()).rev() {
        if drafted_token != LLAMA_TOKEN_NULL {
            break;
        }
        let ngram_primary = &ngrams_primary[i];

        let Some(part_primary) = nc_primary.get(ngram_primary) else {
            continue;
        };

        let mut max_count_primary = 0;
        let mut max_count_static = 0;
        let mut sum_count_primary = 0;
        let mut max_token = LLAMA_TOKEN_NULL;

        for (&token, &count_primary) in part_primary {
            // :124 — the static count scaled by 100 (1 when unseen)
            let count_static = part_static.get(&token).map(|&c| 100 * c).unwrap_or(1);

            if count_primary * count_static > max_count_primary * max_count_static {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static = count_static;
            }
            sum_count_primary += count_primary;
        }

        // :134-139
        if sum_count_primary < min_sample_size[i] {
            continue;
        }
        if 100 * max_count_primary < min_percent[i] * sum_count_primary {
            continue;
        }
        drafted_token = max_token;
    }

    drafted_token
}

/// `common_ngram_cache_draft` (ngram-cache.cpp:146-198): extend `draft`
/// (initially `[id_last]`) up to `n_draft` tokens from the context, dynamic
/// and static caches.
pub fn common_ngram_cache_draft(
    inp: &[i32],
    draft: &mut Vec<i32>,
    n_draft: i32,
    ngram_min: usize,
    ngram_max: usize,
    nc_context: &mut CommonNgramCache,
    nc_dynamic: &mut CommonNgramCache,
    nc_static: &mut CommonNgramCache,
) {
    // GGML_ASSERT(draft.size() == 1) (:150)
    assert_eq!(draft.len(), 1);
    let inp_size = inp.len();

    if inp_size < LLAMA_NGRAM_STATIC {
        return; // :153-155
    }

    while draft.len() as i32 - 1 < n_draft {
        let mut drafted_token = LLAMA_TOKEN_NULL;

        // :160-169 — the static (size-2) n-gram ending at the draft head
        let ngram_start_static =
            inp_size as i64 - LLAMA_NGRAM_STATIC as i64 + draft.len() as i64 - 1;
        let mut ngram_static = CommonNgram {
            tokens: [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX],
        };
        for j in ngram_start_static..ngram_start_static + LLAMA_NGRAM_STATIC as i64 {
            ngram_static.tokens[(j - ngram_start_static) as usize] =
                get_token(inp, draft, j as usize);
        }
        let part_static: CommonNgramCachePart =
            nc_static.get(&ngram_static).cloned().unwrap_or_default();

        // :171-180 — the context+dynamic n-grams of every size
        let mut ngrams_cd: Vec<CommonNgram> = Vec::with_capacity(ngram_max - ngram_min + 1);
        for ngram_size_cd in ngram_min as i64..=ngram_max as i64 {
            let ngram_start_cd = inp_size as i64 - ngram_size_cd + draft.len() as i64 - 1;
            let mut ngram_cd = CommonNgram {
                tokens: [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX],
            };
            for j in ngram_start_cd..ngram_start_cd + ngram_size_cd {
                ngram_cd.tokens[(j - ngram_start_cd) as usize] = get_token(inp, draft, j as usize);
            }
            ngrams_cd.push(ngram_cd);
        }
        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_primary(
                nc_context,
                &ngrams_cd,
                &part_static,
                &DRAFT_MIN_SAMPLE_SIZE_LAX,
                &DRAFT_MIN_PERCENT_LAX,
            );
        }
        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_primary(
                nc_dynamic,
                &ngrams_cd,
                &part_static,
                &DRAFT_MIN_SAMPLE_SIZE_STRICT,
                &DRAFT_MIN_PERCENT_STRICT,
            );
        }
        if drafted_token == LLAMA_TOKEN_NULL {
            drafted_token = try_draft_static(nc_static, &ngram_static);
        }

        if drafted_token == LLAMA_TOKEN_NULL {
            break; // :191-193
        }

        draft.push(drafted_token); // :196
    }
}

/// `common_ngram_cache_save` (ngram-cache.cpp:200-220): the reference's
/// binary layout — per entry `[common_ngram (16 bytes) | i32 ntokens |
/// ntokens × (i32 token, i32 count)]`, little-endian, hash-map iteration
/// order (Rust's HashMap order differs from the C's unordered_map order, so
/// the byte stream's *entry order* is not reproducible — every entry is
/// byte-identical and `_load` accepts either).
pub fn common_ngram_cache_save(
    ngram_cache: &CommonNgramCache,
    filename: &str,
) -> std::io::Result<()> {
    let mut file_out = std::fs::File::create(filename)?;

    for (ngram, token_counts) in ngram_cache {
        assert!(!token_counts.is_empty()); // :205
        let ntokens = token_counts.len() as i32;
        assert!(ntokens > 0); // :207

        file_out.write_all(bytemuck::cast_slice(&ngram.tokens))?;
        file_out.write_all(&ntokens.to_le_bytes())?;
        for (&token, &count) in token_counts {
            assert!(count > 0); // :214
            file_out.write_all(&token.to_le_bytes())?;
            file_out.write_all(&count.to_le_bytes())?;
        }
    }

    Ok(())
}

/// the metadata of one stored tensor block (`dsv4_state_write_tensor_streams`
/// style) — here simply the read-side asserts of `common_ngram_cache_load`
/// (ngram-cache.cpp:222-258), reported as `Err` instead of GGML_ASSERT
fn read_exact_or_eof(buf: &mut impl Read, out: &mut [u8]) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < out.len() {
        match buf.read(&mut out[done..]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// `common_ngram_cache_load` (ngram-cache.cpp:222-258): read a cache written
/// by [`common_ngram_cache_save`] (or the reference's `llama-lookup-create`).
/// The C throws `std::ifstream::failure` on a missing file and GGML_ASSERTs
/// on truncation; the port reports both as `Err`.
pub fn common_ngram_cache_load(filename: &str) -> std::io::Result<CommonNgramCache> {
    let mut file = std::fs::File::open(filename)?;
    let mut ngram_cache = CommonNgramCache::new();

    loop {
        // :238 — a short ngram read (not EOF) is a truncated file
        let mut ngramc = [0u8; std::mem::size_of::<CommonNgram>()];
        if read_exact_or_eof(&mut file, &mut ngramc)? == 0 {
            break; // clean EOF
        }
        let ngram = CommonNgram {
            tokens: bytemuck::cast(ngramc),
        };

        let mut b4 = [0u8; 4];
        if read_exact_or_eof(&mut file, &mut b4)? != 4 {
            return Err(truncated(filename));
        }
        let ntokens = i32::from_le_bytes(b4);
        if ntokens <= 0 {
            return Err(truncated(filename)); // :241
        }

        let mut token_counts = CommonNgramCachePart::new();
        for _ in 0..ntokens {
            if read_exact_or_eof(&mut file, &mut b4)? != 4 {
                return Err(truncated(filename)); // :246
            }
            let token = i32::from_le_bytes(b4);
            if read_exact_or_eof(&mut file, &mut b4)? != 4 {
                return Err(truncated(filename)); // :248
            }
            let count = i32::from_le_bytes(b4);
            if count <= 0 {
                return Err(truncated(filename)); // :249
            }
            token_counts.insert(token, count);
        }

        ngram_cache.insert(ngram, token_counts);
    }

    Ok(ngram_cache)
}

fn truncated(filename: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Unable to parse ngram cache file {filename}"),
    )
}

/// `common_ngram_cache_merge` (ngram-cache.cpp:260-285): fold
/// `ngram_cache_add` into `ngram_cache_target` (counts add up).
pub fn common_ngram_cache_merge(
    ngram_cache_target: &mut CommonNgramCache,
    ngram_cache_add: &CommonNgramCache,
) {
    for (ngram, part) in ngram_cache_add {
        let part_merged = ngram_cache_target
            .entry(ngram.clone())
            .or_insert_with(CommonNgramCachePart::new);

        for (&token, &count) in part {
            assert!(count > 0); // :274
            *part_merged.entry(token).or_insert(0) += count;
        }
    }
}

// ---------------------------------------------------------------------------
// unit tests — hand-computed cache/draft behavior over the C's thresholds
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_counts_ngrams() {
        // ngram-cache.cpp:12-52 — "1 2 1 2 1" at sizes 1..2: the 1-gram (1)
        // is followed by 2 twice (pos 0 and 2; pos 4 ends the input), the
        // 2-gram (1,2) by 1 twice
        let inp = [1i32, 2, 1, 2, 1];
        let mut cache = CommonNgramCache::new();
        common_ngram_cache_update(&mut cache, 1, 2, &inp, inp.len(), false);

        let one_a = CommonNgram::new(&[1], 1);
        let part = &cache[&one_a];
        assert_eq!(part[&2], 2, "token 2 followed token 1 twice");
        assert!(!part.contains_key(&1), "the final 1 has no follower");
        let bigram = CommonNgram::new(&[1, 2], 2);
        assert_eq!(cache[&bigram][&1], 2);

        // appending continues, does not rebuild: one more "1" at the end
        let inp2 = [1i32, 2, 1, 2, 1, 1];
        common_ngram_cache_update(&mut cache, 1, 2, &inp2, 1, false);
        assert_eq!(cache[&one_a][&1], 1, "pos 5's 1 follows a 1");
    }

    #[test]
    fn draft_from_context_cache() {
        // a perfectly repetitive context: every "2" is followed by "3" and
        // every "3" by "2" — the draft chains the repetition (the size-1
        // lax thresholds pass: 2 samples, 100%)
        let mut cache = CommonNgramCache::new();
        let inp = [1i32, 2, 3, 2, 3, 2];
        common_ngram_cache_update(&mut cache, 1, 2, &inp, inp.len(), false);
        let mut dynamic = CommonNgramCache::new();
        let mut statics = CommonNgramCache::new();

        let mut draft = vec![2i32];
        common_ngram_cache_draft(
            &inp,
            &mut draft,
            3,
            1,
            2,
            &mut cache,
            &mut dynamic,
            &mut statics,
        );
        assert_eq!(draft, vec![2, 3, 2, 3], "the draft chains the repetition");
    }

    #[test]
    fn save_load_roundtrip_and_merge() {
        let mut cache = CommonNgramCache::new();
        let inp = [5i32, 6, 7, 5, 6, 7];
        common_ngram_cache_update(&mut cache, 1, 2, &inp, inp.len(), false);

        // pid-unique temp dir — a fixed path races with a concurrent test run
        // of the same suite (the 2026-10-01 batch-18 gate hit exactly that:
        // a parallel agent's run rewrote cache.bin between save and load)
        let dir = std::env::temp_dir()
            .join(format!("llama-rust-ngram-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        common_ngram_cache_save(&cache, path.to_str().unwrap()).unwrap();

        let loaded = common_ngram_cache_load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), cache.len());
        for (ngram, part) in &cache {
            let got = &loaded[ngram];
            assert_eq!(got.len(), part.len());
            for (&t, &c) in part {
                assert_eq!(got[&t], c);
            }
        }

        // merge doubles the counts
        let mut merged = cache.clone();
        common_ngram_cache_merge(&mut merged, &loaded);
        let one = CommonNgram::new(&[5], 1);
        assert_eq!(merged[&one][&6], 2 * cache[&one][&6]);

        // a truncated file errors (the C's GGML_ASSERTs)
        let short = dir.join("short.bin");
        std::fs::write(&short, &[0u8; 8]).unwrap();
        assert!(common_ngram_cache_load(short.to_str().unwrap()).is_err());
    }
}
