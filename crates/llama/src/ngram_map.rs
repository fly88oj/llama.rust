//! ngram-map — port of `common/ngram-map.cpp` / `common/ngram-map.h`
//! (llama.cpp bd4f514db1): structures that map n-grams to a list of m-grams
//! for lookup in the model's own token history — the two self-speculation
//! algorithms of ref https://github.com/ggml-org/llama.cpp/pull/18471:
//!
//!   * `ngram_simple` — a backward pattern search over the history
//!     ([`common_ngram_simple_draft`], ngram-map.cpp:49-112);
//!   * `ngram_map` — a map from key n-grams to up to
//!     [`COMMON_NGRAM_MAX_VALUES`] following m-grams with hit statistics
//!     ([`CommonNgramMap`], ngram-map.cpp:121-536).
//!
//! Used by the `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` speculative
//! implementations (`common_speculative_impl_ngram_simple` /
//! `common_speculative_impl_ngram_map_k`, speculative.cpp:1769-1867, ported
//! in `crate::speculative`).

/// `COMMON_NGRAM_MAX_VALUES` (ngram-map.h:39): maximum number of m-gram
/// values stored for each key n-gram.
pub const COMMON_NGRAM_MAX_VALUES: usize = 4;

/// `COMMON_NGRAM_HASH_MAP_SIZE` (ngram-map.h:42): entries in the (optional,
/// size 0 disables) hash → ngram-index map.
pub const COMMON_NGRAM_HASH_MAP_SIZE: usize = 262144;

/// `COMMON_NGRAM_MAX_VALUE_COUNT` (ngram-map.cpp:119): maximum counted
/// occurrences of one value.
pub const COMMON_NGRAM_MAX_VALUE_COUNT: u32 = 16380;

/// prime number used for LCG hash function (32 bit), it is near
/// (sqrt(5) - 1)/2 * 2^32 — `LCG_FACTOR` (ngram-map.cpp:11)
const LCG_FACTOR: u32 = 2654435761;

/// `common_ngram_map_hash` (ngram-map.cpp:14-20): the LCG hash of the n-gram
/// of size `len` at offset `start`.
fn common_ngram_map_hash(tokens: &[i32], start: usize, len: usize) -> u32 {
    let mut hash: u32 = 0;
    for i in 0..len {
        hash = hash
            .wrapping_mul(LCG_FACTOR)
            .wrapping_add(tokens[start + i] as u32);
    }
    hash
}

/// `common_ngram_simple_config` (ngram-map.h:24-27)
#[derive(Clone, Copy, Debug)]
pub struct CommonNgramSimpleConfig {
    /// size of n-grams to lookup in self-mode
    pub size_ngram: u16,
    /// size of m-grams to draft in self-mode
    pub size_mgram: u16,
}

/// `common_ngram_simple_draft` (ngram-map.cpp:49-112): search the last
/// `size_ngram` tokens (the pattern) backwards in the history and return the
/// `≤ size_mgram` tokens that followed the newest other occurrence.
pub fn common_ngram_simple_draft(
    config: &CommonNgramSimpleConfig,
    tokens: &[i32],
    sampled: i32,
) -> Vec<i32> {
    // Simple implementation of self-speculative decoding without a draft
    // model. (ngram-map.cpp:53)
    let cur_len = tokens.len();

    let n_draft_min = config.size_ngram as usize; // :57
    let n_draft_max = config.size_mgram as usize; // :58

    // return an empty vector if there is no match (:62)
    let mut draft_tokens: Vec<i32> = Vec::new();

    // We need at least n_draft_min + n_draft_max + 1 tokens. (:64-67)
    if cur_len <= n_draft_min + n_draft_max + 1 {
        return draft_tokens;
    }

    // pattern search (:69-75) — the last n_draft_min-1 tokens + `sampled`
    let mut pattern: Vec<i32> = Vec::with_capacity(n_draft_min);
    for j in cur_len - n_draft_min + 1..cur_len {
        pattern.push(tokens[j]);
    }
    pattern.push(sampled); // add the last token to the pattern

    // we ignore position 0, position 0 == no match; search backwards, but
    // skip the current match (we are currently there) (:77-91)
    let mut match_pos = 0usize;
    let mut j = cur_len as i64 - n_draft_min as i64 - 1;
    while j > 0 {
        let ju = j as usize;
        let mut m = true;
        for (k, &p) in pattern.iter().enumerate() {
            if tokens[ju + k] != p {
                m = false;
                break;
            }
        }
        if m {
            match_pos = ju;
            break;
        }
        j -= 1;
    }
    if match_pos == 0 {
        return draft_tokens; // :92-94
    }

    // :96-102
    let copy_max = n_draft_max.min(cur_len - (match_pos + n_draft_min));
    if copy_max < n_draft_min {
        return draft_tokens;
    }

    // :107-111
    draft_tokens.reserve(copy_max);
    for j in 0..copy_max {
        draft_tokens.push(tokens[match_pos + n_draft_min + j]);
    }
    draft_tokens
}

/// `common_ngram_map_value` (ngram-map.h:45-49): statistics of an m-gram
/// after a known n-gram.
#[derive(Clone, Copy, Debug)]
pub struct CommonNgramMapValue {
    /// index of the value m-gram in token-history (0 if unused)
    pub value_idx: usize,
    /// number of occurrences of this value m-gram after the key n-gram
    /// (0 in an unused values-slot)
    pub value_num: u16,
    /// number of accepted tokens at last draft (-1 if unused)
    pub n_accepted: i16,
}

impl Default for CommonNgramMapValue {
    fn default() -> Self {
        CommonNgramMapValue {
            value_idx: 0,
            value_num: 0,
            n_accepted: -1,
        }
    }
}

/// `common_ngram_map_key` (ngram-map.h:52-58): statistics of an n-gram.
#[derive(Clone, Debug)]
pub struct CommonNgramMapKey {
    /// index of the key n-gram in token-history
    pub key_idx: usize,
    /// index of the last token of statistics computation (key_num, values)
    pub stat_idx: usize,
    /// number of occurrences of this key n-gram in token-history
    pub key_num: u16,
    /// some known values after the key
    pub values: Vec<CommonNgramMapValue>,
}

/// `common_ngram_map` (ngram-map.h:61-95): the map from n-grams to following
/// m-grams in token-history, plus the rollback/rebuild bookkeeping of
/// reasoning chats (the previous reasoning block is removed from context
/// history → `size_last_begin` shrinks and the next `begin` refreshes).
#[derive(Clone, Debug)]
pub struct CommonNgramMap {
    /// size of key n-grams
    pub size_key: u16,
    /// size of value m-grams
    pub size_value: u16,
    /// true if only key n-grams are used, no values (the `ngram-map-k` mode)
    pub key_only: bool,
    /// key n-grams which occur several times in token-history
    pub keys: Vec<CommonNgramMapKey>,
    /// minimum number of key hits to consider a draft
    pub min_hits: u16,
    /// number of tokens at the previous start of generation (:82)
    pub size_last_begin: usize,
    /// true if a draft was created at the last call (:84)
    pub last_draft_created: bool,
    /// index of the last key used for draft generation (0 = no draft) (:85)
    pub last_draft_key_idx: usize,
    /// index of the last value used for draft generation (:86)
    pub last_draft_value_idx: u16,
    /// index of the last check in context history (:88)
    pub idx_last_check: usize,
    /// key_map[hash] = index of the n-gram in the context window (:93) —
    /// the optional uint32 map of ngram-map.h:42 (empty disables it, as the
    /// port does; the linear search paths below always run)
    pub key_map: Vec<u32>,
    /// index of the last n-gram added to key_map (:94)
    pub key_map_last_idx: u32,
}

impl CommonNgramMap {
    /// `get_common_ngram_map` (speculative.cpp:2195-2204) over a
    /// `common_params_speculative_ngram_map`
    pub fn new(size_key: u16, size_value: u16, key_only: bool, min_hits: u16) -> Self {
        CommonNgramMap {
            size_key,
            size_value,
            key_only,
            keys: Vec::new(),
            min_hits,
            size_last_begin: 0,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
            idx_last_check: 0,
            key_map: Vec::new(), // size 0 disables the hash map (ngram-map.h:42)
            key_map_last_idx: 0,
        }
    }

    /// `common_ngram_map_begin` (ngram-map.cpp:121-219): start a generation —
    /// record the new begin and refresh (shrink-clean) the map when the
    /// history got shorter than the last begin (the reasoning-chat rebuild).
    pub fn begin(&mut self, tokens: &[i32]) {
        let size_begin = tokens.len();

        // :128-135 — the cleanup start when the history shrank
        let mut idx_begin_cleanup = self.size_last_begin;
        if idx_begin_cleanup > size_begin {
            if size_begin > (self.size_key + self.size_value) as usize {
                idx_begin_cleanup = size_begin - self.size_key as usize - self.size_value as usize;
            } else {
                idx_begin_cleanup = 0;
            }
        }

        // :161-169 — the key_map cleanup (disabled here: empty key_map)
        if !self.key_map.is_empty() && size_begin < self.idx_last_check {
            let mut count_map_entries_upd = 0usize;
            for v in self.key_map.iter_mut() {
                if *v != 0 && *v as usize >= idx_begin_cleanup {
                    *v = 0;
                    count_map_entries_upd += 1;
                }
            }
            let _ = count_map_entries_upd;
            self.key_map_last_idx = if idx_begin_cleanup > 0 {
                (idx_begin_cleanup - 1) as u32
            } else {
                0
            };
        }

        // :171-215 — delete the keys / values at or past the cleanup start
        if size_begin < self.idx_last_check && !self.keys.is_empty() {
            let count_keys = self.keys.len();
            let mut count_keys_del = 0usize;
            let mut count_values_del = 0usize;

            // iterate backwards so erase() keeps the remaining indices valid
            for i in (0..self.keys.len()).rev() {
                if self.keys[i].key_idx >= idx_begin_cleanup {
                    // Delete the key. (:177-182)
                    self.keys.remove(i);
                    count_keys_del += 1;
                    continue;
                }
                if self.key_only {
                    continue; // :184-186
                }

                // Check the indices of the values. (:188-203) — delete
                // value slots at/past the cleanup start, shifting the rest
                // left and clearing the tail
                let mut j = COMMON_NGRAM_MAX_VALUES as i64 - 1;
                while j >= 0 {
                    if self.keys[i].values[j as usize].value_idx != 0
                        && self.keys[i].values[j as usize].value_idx >= idx_begin_cleanup
                    {
                        count_values_del += 1;
                        // Move all values after this value to the left.
                        for k in j as usize..COMMON_NGRAM_MAX_VALUES - 1 {
                            self.keys[i].values[k] = self.keys[i].values[k + 1];
                        }
                        // Clear the last value.
                        let last = COMMON_NGRAM_MAX_VALUES - 1;
                        self.keys[i].values[last].value_idx = 0;
                        self.keys[i].values[last].value_num = 0;
                    }
                    j -= 1;
                }
                if self.keys[i].values[0].value_idx == 0 {
                    // No values left, delete the key. (:204-209)
                    self.keys.remove(i);
                    count_keys_del += 1;
                }
            }

            let _ = (count_keys, count_keys_del, count_values_del);
        }

        self.idx_last_check = size_begin; // :217
        self.size_last_begin = size_begin; // :218
    }

    /// `common_ngram_map_draft` (ngram-map.cpp:221-516): search the key
    /// n-gram (the last `size_key` tokens ending in `sampled`) in the map /
    /// history, update the key statistics and fill `draft` with the most
    /// frequent following m-gram.
    pub fn draft(&mut self, inp: &[i32], sampled: i32, draft: &mut Vec<i32>) {
        // reset last key and value. (:224-227)
        self.last_draft_created = false;
        self.last_draft_key_idx = 0;
        self.last_draft_value_idx = 0;

        let cur_len = inp.len();
        let n = self.size_key as usize;
        let m = self.size_value as usize;
        if cur_len < 2 * n + m {
            return; // :232-234
        }

        // :240-244 — map.idx_last_check follows the history
        assert!(
            self.idx_last_check <= cur_len,
            "idx_last_check > cur_len (should not happen because of begin())"
        );
        self.idx_last_check = cur_len;

        // search pattern, the key n-gram (:246-252)
        let mut key_tokens: Vec<i32> = Vec::with_capacity(n);
        for j in cur_len - n + 1..cur_len {
            key_tokens.push(inp[j]);
        }
        key_tokens.push(sampled);

        // search for the key in the map (:254-312)
        let mut match_pos = 0usize;
        assert!(self.size_last_begin <= cur_len, "size_last_begin > cur_len");
        // (the key_map fast path is disabled — the port always searches)
        if match_pos == 0 && self.size_last_begin > n + m + 1 {
            // Search in [1, size_last_begin - n - m - 1], descending (:280-294)
            let mut j = self.size_last_begin - n - m - 1;
            while j > self.key_map_last_idx as usize {
                let mut mmatch = true;
                for k in 0..n {
                    if inp[j + k] != key_tokens[k] {
                        mmatch = false;
                        break;
                    }
                }
                if mmatch {
                    match_pos = j;
                    break;
                }
                j -= 1;
            }
        }
        if match_pos == 0 {
            // In case of a reasoning chat, the part after size_last_begin may
            // be deleted/reordered later. Search in
            // [size_last_begin, cur_len - n - m - 1], descending. (:295-312)
            let mut j = cur_len as i64 - n as i64 - m as i64 - 1;
            while j > self.size_last_begin as i64 && j > self.key_map_last_idx as i64 {
                let ju = j as usize;
                let mut mmatch = true;
                for k in 0..n {
                    if inp[ju + k] != key_tokens[k] {
                        mmatch = false;
                        break;
                    }
                }
                if mmatch {
                    match_pos = ju;
                    break;
                }
                j -= 1;
            }
        }

        // We have a match, now we look for the statistics of the key.
        // (:347-361) — the first key whose n-gram matches
        let mut key_offset = self.keys.len();
        for (i, key) in self.keys.iter().enumerate() {
            let mut mmatch = true;
            for j in 0..n {
                if inp[key.key_idx + j] != key_tokens[j] {
                    mmatch = false;
                    break;
                }
            }
            if mmatch {
                key_offset = i;
                break;
            }
        }
        if key_offset == self.keys.len() {
            // We create a new key-entry, it will get offset key_offset.
            // (:362-373)
            let mut new_key = CommonNgramMapKey {
                key_idx: match_pos,
                stat_idx: 0,
                key_num: 0,
                values: vec![CommonNgramMapValue::default(); COMMON_NGRAM_MAX_VALUES],
            };
            for v in new_key.values.iter_mut() {
                v.value_num = 0;
                v.n_accepted = m as i16;
            }
            self.keys.push(new_key);
        }

        // update number of key hits (:378-380)
        let key = &mut self.keys[key_offset];
        key.key_num = (key.key_num + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT as u16);

        if self.key_only {
            // simple mode (:382-399): fill the draft with the m tokens
            // following the key, values[0] only
            let n_draft_tokens = m.min(key.values[0].n_accepted.max(0) as usize);
            for i in 0..n_draft_tokens {
                draft.push(inp[match_pos + n + i]);
            }

            self.last_draft_created = true;
            self.last_draft_key_idx = key_offset;
            self.last_draft_value_idx = 0; // value 0 is used for simple mode
            return;
        }

        if key.key_num < self.min_hits {
            // not enough hits to consider this a good draft (:401-406)
            return;
        }

        // complex mode: examine the different m-grams after this key n-gram.
        // (:408-456) — scan [stat_idx, match_pos] for the key, dedup the
        // following m-grams into the value slots
        let key_tokens_snapshot = key_tokens.clone();
        for i in key.stat_idx..=match_pos {
            // begins the key n-gram at index i? (:413-422)
            let mut match_key = true;
            for k in 0..n {
                if inp[i + k] != key_tokens_snapshot[k] {
                    match_key = false;
                    break;
                }
            }
            if !match_key {
                continue;
            }

            // existing or new value m-gram after the key at index i?
            // (:425-450)
            let idx_begin_value_key = i + n;
            let mut idx_value: i64 = -1;
            for (v, value) in key.values.iter().enumerate() {
                let idx_begin_value_v = value.value_idx;
                if idx_begin_value_v == 0 {
                    // an empty slot => a new value m-gram (:430-436)
                    key.values[v].value_idx = idx_begin_value_key;
                    key.values[v].value_num = 0;
                    key.values[v].n_accepted = m as i16;
                    idx_value = v as i64;
                    break;
                }
                let mut mmatch = true;
                for j in 0..m {
                    if inp[idx_begin_value_key + j] != inp[idx_begin_value_v + j] {
                        mmatch = false;
                        break;
                    }
                }
                if mmatch {
                    // an existing value m-gram (:445-449)
                    idx_value = v as i64;
                    break;
                }
            }
            if idx_value >= 0 {
                // We found a value m-gram of the key n-gram. (:451-455)
                let v = idx_value as usize;
                key.values[v].value_num =
                    (key.values[v].value_num + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT as u16);
            }
        }
        // the statistics are updated up to match_pos. (:457-458)
        key.stat_idx = match_pos;

        // Do we have a value we could use for the draft? (:460-478)
        let mut max_occur: u16 = 0;
        let mut slot_max = 0usize;
        for (v, value) in key.values.iter().enumerate() {
            if value.value_num > max_occur {
                max_occur = value.value_num;
                slot_max = v;
            }
        }
        // What is the sum of the other occurrences? (:470-478)
        let mut sum_occur: u32 = 0;
        for (v, value) in key.values.iter().enumerate() {
            if v == slot_max {
                continue;
            }
            sum_occur += value.value_num as u32;
        }

        if sum_occur > 0 && (max_occur as u32) < 2 * sum_occur {
            // The most frequent value is not much more frequent than the
            // other values. We do not use the draft. (:495-499)
            return;
        }

        // We use the most frequent value values[slot_max] for the draft.
        // (:501-507)
        let n_draft_tokens = m.min(key.values[slot_max].n_accepted.max(0) as usize);
        for i in 0..n_draft_tokens {
            draft.push(inp[match_pos + n + i]);
        }

        self.last_draft_created = true; // :513-515
        self.last_draft_key_idx = key_offset;
        self.last_draft_value_idx = slot_max as u16;
    }

    /// `common_ngram_map_accept` (ngram-map.cpp:518-536): record how many
    /// tokens of the last draft the target accepted (the value's draft
    /// length adapts).
    pub fn accept(&mut self, n_accepted: u16) {
        if !self.last_draft_created {
            return; // :519-521
        }

        // find the key and its chosen value. (:523-530)
        let key_idx = self.last_draft_key_idx;
        let val_idx = self.last_draft_value_idx as usize;

        // update the value statistics (:532-535)
        self.keys[key_idx].values[val_idx].n_accepted = n_accepted as i16;
    }
}

// ---------------------------------------------------------------------------
// unit tests — the reference's own examples (hand-computed from the C)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_draft_finds_repeat() {
        // ngram-map.cpp:49-112 — the pattern (n=2) is the last token [6] +
        // sampled 7 → [6, 7]; its newest other occurrence is at index 2,
        // followed by [10, 11, 12] (m=3)
        let config = CommonNgramSimpleConfig {
            size_ngram: 2,
            size_mgram: 3,
        };
        let tokens = vec![1, 2, 6, 7, 10, 11, 12, 3, 4, 6];
        let d = common_ngram_simple_draft(&config, &tokens, 7);
        assert_eq!(d, vec![10, 11, 12]);

        // too-short history: the early-out (:65-67)
        let short = vec![1, 2, 3, 4, 5, 6];
        assert!(common_ngram_simple_draft(&config, &short, 7).is_empty());

        // no second occurrence: empty
        let once = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert!(common_ngram_simple_draft(&config, &once, 5).is_empty());
    }

    #[test]
    fn map_draft_key_only_mode() {
        // ngram-map-k: no value statistics — the m tokens after the newest
        // key match are drafted (min_hits is not consulted in key_only mode)
        let mut map = CommonNgramMap::new(2, 3, true, 1);
        let inp: Vec<i32> = vec![1, 2, 6, 7, 10, 11, 12, 3, 4, 6];
        map.begin(&inp);
        let mut draft = Vec::new();
        map.draft(&inp, 7, &mut draft);
        assert_eq!(draft, vec![10, 11, 12]);
        assert!(map.last_draft_created);
        assert_eq!(map.last_draft_value_idx, 0);

        // the accept feedback only changes n_accepted of value 0
        map.accept(2);
        let key = &map.keys[map.last_draft_key_idx];
        assert_eq!(key.values[0].n_accepted, 2);

        // a new draft is then cut to the accepted length (m min n_accepted)
        let mut draft2 = Vec::new();
        map.draft(&inp, 7, &mut draft2);
        assert_eq!(draft2, vec![10, 11]);
    }

    #[test]
    fn map_draft_complex_mode_needs_min_hits() {
        // ngram-map-k4v: the first hit only counts statistics (key_num 1 <
        // min_hits 2 → no draft); the second hit drafts
        let mut map = CommonNgramMap::new(2, 3, false, 2);
        let inp: Vec<i32> = vec![1, 2, 6, 7, 10, 11, 12, 3, 4, 6];
        map.begin(&inp);

        let mut draft = Vec::new();
        map.draft(&inp, 7, &mut draft);
        assert!(draft.is_empty(), "first hit: key_num 1 < min_hits 2");
        assert!(!map.last_draft_created);

        // same key again (the driver re-drafts after accepting): key_num 2
        let mut draft = Vec::new();
        map.draft(&inp, 7, &mut draft);
        assert_eq!(draft, vec![10, 11, 12], "second hit: the only value drafts");
    }

    #[test]
    fn map_begin_shrinks_after_history_removal() {
        // the reasoning-chat rebuild (:128-135 + :171-215): the history
        // shrinks below size_last_begin → keys at/past the cleanup start are
        // deleted
        let mut map = CommonNgramMap::new(2, 3, false, 1);
        let long: Vec<i32> = vec![1, 2, 6, 7, 10, 11, 12, 3, 4, 6];
        map.begin(&long);
        let mut draft = Vec::new();
        map.draft(&long, 7, &mut draft);
        assert_eq!(map.keys.len(), 1);

        // the history drops its tail — the cleanup start is 10 - 2 - 3 = 5,
        // so a key at index 2 survives (:128-135) and one at index 7 dies
        assert_eq!(map.keys[0].key_idx, 2);
        map.keys[0].key_idx = 7;
        let short: Vec<i32> = long[..5].to_vec();
        map.begin(&short);
        assert!(
            map.keys.is_empty(),
            "the key past the cleanup start is deleted"
        );
    }
}
