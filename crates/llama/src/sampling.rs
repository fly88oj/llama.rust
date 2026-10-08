//! sampling.rs — sampler chain, 1:1 port of llama.cpp `src/llama-sampler.cpp` (CPU paths)
//! plus the convenience API from `common/sampling.cpp` / `common/common.h`
//! (the old `llama-sampling.cpp` / `llama_sampling_context` API no longer exists in
//! the pinned tree — see the module report for details).
//!
//! Baseline: llama.cpp bd4f514db1, libstdc++ (GCC 13) `<random>` semantics.
//!
//! Bit-exactness notes (all verified against a C++ reference program compiled with
//! g++ 13.3 against the pinned sources; see the port report for the fixtures):
//!
//! * [`Mt19937`] replicates `std::mt19937` bit-for-bit (verified against the canonical
//!   mt19937ar test vector for the default seed 5489 and against libstdc++ for seed 42).
//! * [`Mt19937::canonical_double`] replicates `std::generate_canonical<double, 53>`
//!   as implemented in libstdc++ (`random.tcc`): two 32-bit draws combined as
//!   `(u1 + u2 * 2^32) / 2^64` in f64, with the LWG-2524 `nextafter(1, 0)` clamp.
//!   `uniform_real_distribution<double>(0, 1)` is exactly this value (`x * 1.0 + 0.0`).
//! * [`Mt19937::canonical_float`] replicates `std::generate_canonical<float, 24>`
//!   (one draw, `u / 2^32` as f32) — used by the XTC sampler.
//! * [`DiscreteDistribution`] replicates libstdc++ `std::discrete_distribution<int>`
//!   (`_M_initialize` normalize/partial-sum + `lower_bound` sampling).
//! * `std::sort` / `std::partial_sort` on candidate arrays are replicated through the
//!   libstdc++ introsort / heap-select algorithms ([`std_sort_by`], [`std_partial_sort_by`])
//!   so that tie-order of equal logits matches the C++ build exactly.
//!
//! Deviations from the pinned C++ (intentional, documented):
//! * `greedy` breaks ties by taking the *last* maximum (matches `ggml_vec_argmax_f32`);
//!   the pinned CPU `llama_sampler_greedy_apply` uses a strict `>` and takes the *first*.
//! * The `selected` field of `llama_token_data_array` is kept (new-style API) — the
//!   samplers operate on [`TokenDataArray`], not on a bare `&mut [f32]`, because
//!   top-p/typical/dist semantics need `id`/`p`/`sorted`. Use
//!   [`SamplerChain::apply_logits`] for a logits-only convenience wrapper.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// llama.h types (sampling subset)
// ---------------------------------------------------------------------------

pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

/// `llama_token`
pub type LlamaToken = i32;

/// `llama_token_data` (NOTE: distinct from `vocab::TokenData`, which is
/// `llama_vocab::token_data` — text/score/attr).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenData {
    pub id: LlamaToken,
    pub logit: f32,
    pub p: f32,
}

/// `llama_token_data_array`
#[derive(Debug, Clone, Default)]
pub struct TokenDataArray {
    pub data: Vec<TokenData>,
    pub size: usize,
    /// index into `data[..size]` of the selected candidate (-1 = none yet)
    pub selected: i64,
    /// do not assume sorted — always check this flag
    pub sorted: bool,
}

impl TokenDataArray {
    /// Build candidates from a raw logits vector (id == index, p == 0, unsorted).
    /// This mirrors what `llama_sampler_sample` does with `llama_get_logits_ith`.
    pub fn from_logits(logits: &[f32]) -> Self {
        TokenDataArray {
            data: logits
                .iter()
                .enumerate()
                .map(|(i, &l)| TokenData {
                    id: i as LlamaToken,
                    logit: l,
                    p: 0.0,
                })
                .collect(),
            size: logits.len(),
            selected: -1,
            sorted: false,
        }
    }

    pub fn selected_token(&self) -> Option<LlamaToken> {
        if self.selected >= 0 && (self.selected as usize) < self.size {
            Some(self.data[self.selected as usize].id)
        } else {
            None
        }
    }
}

/// `llama_logit_bias`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitBias {
    pub token: LlamaToken,
    pub bias: f32,
}

// ---------------------------------------------------------------------------
// ring_buffer (verbatim port of the `ring_buffer` template in llama-sampler.cpp)
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer, port of `ring_buffer<T>` in llama-sampler.cpp.
/// Only the members used by the samplers are provided.
#[derive(Debug, Clone, Default)]
pub struct RingBuffer<T: Copy + Default> {
    capacity: usize,
    sz: usize,
    first: usize,
    pos: usize,
    data: Vec<T>,
}

impl<T: Copy + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        RingBuffer {
            capacity,
            sz: 0,
            first: 0,
            pos: 0,
            data: vec![T::default(); capacity],
        }
    }

    pub fn front(&self) -> T {
        self.data[self.first]
    }

    pub fn push_back(&mut self, value: T) {
        if self.capacity == 0 {
            panic!("ring buffer: capacity is zero");
        }
        if self.sz == self.capacity {
            // advance the start when buffer is full
            self.first = (self.first + 1) % self.capacity;
        } else {
            self.sz += 1;
        }
        self.data[self.pos] = value;
        self.pos = (self.pos + 1) % self.capacity;
    }

    pub fn clear(&mut self) {
        self.sz = 0;
        self.first = 0;
        self.pos = 0;
    }

    /// `rat(i)` (llama-sampler.cpp:85-90) — reverse access: `rat(0)` is the
    /// most recently pushed value, `rat(sz-1)` the oldest.
    pub fn rat(&self, i: usize) -> T {
        if i >= self.sz {
            panic!("ring buffer: index out of bounds");
        }
        self.data[(self.first + self.sz - i - 1) % self.capacity]
    }

    pub fn len(&self) -> usize {
        self.sz
    }

    pub fn is_empty(&self) -> bool {
        self.sz == 0
    }
}

// ---------------------------------------------------------------------------
// Mt19937 — bit-exact std::mt19937 + libstdc++ generate_canonical
// ---------------------------------------------------------------------------

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// Bit-exact replica of `std::mt19937` (libstdc++ `mersenne_twister_engine`).
#[derive(Debug, Clone)]
pub struct Mt19937 {
    mt: [u32; MT_N],
    mti: usize,
}

impl Mt19937 {
    /// `std::mt19937(seed)`
    pub fn new(seed: u32) -> Self {
        let mut s = Mt19937 {
            mt: [0; MT_N],
            mti: MT_N,
        };
        s.seed(seed);
        s
    }

    /// `rng.seed(s)` — the standard MT19937 initialization by `1812433253 * (x ^ (x >> 30)) + i`.
    pub fn seed(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..MT_N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = MT_N;
    }

    fn twist(&mut self) {
        for i in 0..MT_N {
            let y = (self.mt[i] & MT_UPPER_MASK) | (self.mt[(i + 1) % MT_N] & MT_LOWER_MASK);
            self.mt[i] =
                self.mt[(i + MT_M) % MT_N] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        }
        self.mti = 0;
    }

    /// `rng()` — tempered 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= MT_N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `std::generate_canonical<double, 53>(rng)` — libstdc++ semantics:
    /// two draws, `sum = u1 * 1 + u2 * 2^32` (f64), `ret = sum / 2^64`,
    /// clamped below 1.0 via `nextafter(1, 0)` (LWG 2524).
    /// This is also exactly `uniform_real_distribution<double>(0, 1)(rng)`.
    pub fn canonical_double(&mut self) -> f64 {
        const R: f64 = 4294967296.0; // 2^32
        let mut sum: f64 = 0.0;
        let mut tmp: f64 = 1.0;
        for _ in 0..2 {
            sum += (self.next_u32() as f64) * tmp;
            tmp *= R;
        }
        let mut ret = sum / tmp;
        if ret >= 1.0 {
            // std::nextafter(1.0, 0.0) == largest f64 below 1.0
            ret = f64::from_bits(1.0f64.to_bits() - 1);
        }
        ret
    }

    /// `std::generate_canonical<float, 24>(rng)` — libstdc++ semantics:
    /// one draw, `ret = u / 2^32` (f32), clamped below 1.0.
    /// This is also exactly `uniform_real_distribution<float>(0, 1)(rng)`.
    pub fn canonical_float(&mut self) -> f32 {
        const R: f32 = 4294967296.0; // 2^32
        let mut sum: f32 = 0.0;
        let mut tmp: f32 = 1.0;
        for _ in 0..1 {
            sum += (self.next_u32() as f32) * tmp;
            tmp *= R;
        }
        let mut ret = sum / tmp;
        if ret >= 1.0 {
            // std::nextafter(1.0f, 0.0f)
            ret = f32::from_bits(1.0f32.to_bits() - 1);
        }
        ret
    }
}

/// `get_rng_seed` (llama-sampler.cpp). `LLAMA_DEFAULT_SEED` maps to a
/// non-deterministic seed (C++: `std::random_device` / system clock; here:
/// nanosecond clock — bit-exactness is meaningless for a random seed).
pub fn get_rng_seed(seed: u32) -> u32 {
    if seed == LLAMA_DEFAULT_SEED {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        now.subsec_nanos() ^ (now.as_secs() as u32)
    } else {
        seed
    }
}

// ---------------------------------------------------------------------------
// DiscreteDistribution — libstdc++ std::discrete_distribution<int> replica
// ---------------------------------------------------------------------------

/// Bit-exact replica of libstdc++ `std::discrete_distribution<int>` as used by
/// `llama_sample_dist` (mirostat v1/v2): probabilities are converted to f64,
/// normalized by their f64 sum, accumulated with `partial_sum` into `_M_cp`
/// (last entry forced to exactly 1.0); sampling draws a canonical double and
/// returns the `lower_bound` index (first `_M_cp[i] >= p`).
pub struct DiscreteDistribution {
    cp: Vec<f64>,
}

impl DiscreteDistribution {
    pub fn new(probs: &[f32]) -> Self {
        let mut cp = Vec::new();
        if probs.len() >= 2 {
            // _M_prob: vector<double> built from the f32 iterator range (in order)
            let prob: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
            // std::accumulate(begin, end, 0.0)
            let sum: f64 = prob.iter().fold(0.0f64, |a, &b| a + b);
            debug_assert!(sum > 0.0);
            // __normalize: each /= sum, in order
            let mut prob = prob;
            for p in prob.iter_mut() {
                *p /= sum;
            }
            // std::partial_sum
            let mut acc = prob[0];
            cp.push(acc);
            for &p in prob.iter().skip(1) {
                acc = acc + p;
                cp.push(acc);
            }
            // make sure the last cumulative probability is one
            let last = cp.len() - 1;
            cp[last] = 1.0;
        }
        DiscreteDistribution { cp }
    }

    /// `dist(rng)` — returns the selected index.
    pub fn sample(&self, rng: &mut Mt19937) -> usize {
        if self.cp.is_empty() {
            return 0;
        }
        let p = rng.canonical_double();
        // std::lower_bound(cp.begin(), cp.end(), p)
        let mut lo = 0usize;
        let mut hi = self.cp.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cp[mid] < p {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

// ---------------------------------------------------------------------------
// libstdc++ std::sort / std::partial_sort replication
//
// The CPU samplers sort `llama_token_data` by descending logit (and `typical`
// sorts indices). std::sort is *unstable*, so tie order between equal logits
// depends on the exact algorithm. To reproduce the C++ build bit-for-bit we
// replicate the libstdc++ introsort (std::sort) and heap-select (std::partial_sort)
// algorithms (bits/stl_algo.h / bits/stl_heap.h, GCC 13).
// ---------------------------------------------------------------------------

fn lg_floor(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// libstdc++ `std::sort` (introsort: quicksort with median-of-3, insertion-sort
/// threshold 16, heapsort fallback at depth `2*log2(n)`).
pub fn std_sort_by<T: Copy>(v: &mut [T], comp: &mut impl FnMut(&T, &T) -> bool) {
    if v.len() > 1 {
        introsort_loop(v, 0, v.len(), lg_floor(v.len()) * 2, comp);
        final_insertion_sort(v, 0, v.len(), comp);
    }
}

fn introsort_loop<T: Copy>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth_limit: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    while last - first > 16 {
        if depth_limit == 0 {
            // __partial_sort(first, last, last) — full heapsort of the range
            heap_select(v, first, last, last, comp);
            sort_heap_range(v, first, last, comp);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, comp);
        introsort_loop(v, cut, last, depth_limit, comp);
        last = cut;
    }
}

fn unguarded_partition_pivot<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, comp);
    unguarded_partition(v, first + 1, last, first, comp)
}

fn move_median_to_first<T: Copy>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    if comp(&v[a], &v[b]) {
        if comp(&v[b], &v[c]) {
            v.swap(result, b);
        } else if comp(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if comp(&v[a], &v[c]) {
        v.swap(result, a);
    } else if comp(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) -> usize {
    loop {
        while comp(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while comp(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn final_insertion_sort<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    if last - first > 16 {
        insertion_sort(v, first, first + 16, comp);
        unguarded_insertion_sort(v, first + 16, last, comp);
    } else {
        insertion_sort(v, first, last, comp);
    }
}

fn insertion_sort<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if comp(&v[i], &v[first]) {
            let val = v[i];
            // move_backward(first, i, i + 1)
            let mut j = i;
            while j > first {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, comp);
        }
    }
}

fn unguarded_insertion_sort<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    for i in (first + 1)..last {
        unguarded_linear_insert(v, i, comp);
    }
}

fn unguarded_linear_insert<T: Copy>(
    v: &mut [T],
    last_idx: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    let val = v[last_idx];
    let mut pos = last_idx;
    let mut next = last_idx - 1;
    while comp(&val, &v[next]) {
        v[pos] = v[next];
        pos = next;
        next -= 1;
    }
    v[pos] = val;
}

// ---- heap machinery (bits/stl_heap.h) ----

fn push_heap_at<T: Copy>(
    v: &mut [T],
    first: usize,
    hole_index: usize,
    top_index: usize,
    value: T,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    let mut hole = hole_index;
    let mut parent = if hole > 0 { (hole - 1) / 2 } else { 0 };
    while hole > top_index && comp(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = if hole > 0 { (hole - 1) / 2 } else { 0 };
    }
    v[first + hole] = value;
}

fn adjust_heap<T: Copy>(
    v: &mut [T],
    first: usize,
    hole_index: usize,
    len: usize,
    value: T,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    debug_assert!(len >= 1);
    let top_index = hole_index;
    let mut hole = hole_index;
    let mut second_child = hole_index;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if comp(&v[first + second_child], &v[first + second_child - 1]) {
            second_child -= 1;
        }
        v[first + hole] = v[first + second_child];
        hole = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        v[first + hole] = v[first + second_child - 1];
        hole = second_child - 1;
    }
    push_heap_at(v, first, hole, top_index, value, comp);
}

/// `__pop_heap(first, last, result)` — all indices absolute into `v`.
fn pop_heap_at<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    result: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    let value = v[result];
    v[result] = v[first];
    adjust_heap(v, first, 0, last - first, value, comp);
}

fn make_heap_range<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, comp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn sort_heap_range<T: Copy>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    while last - first > 1 {
        last -= 1;
        pop_heap_at(v, first, last, last, comp);
    }
}

/// `__heap_select(first, middle, last)` — all indices absolute into `v`.
fn heap_select<T: Copy>(
    v: &mut [T],
    first: usize,
    middle: usize,
    last: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    make_heap_range(v, first, middle, comp);
    for i in middle..last {
        if comp(&v[i], &v[first]) {
            pop_heap_at(v, first, middle, i, comp);
        }
    }
}

/// libstdc++ `std::partial_sort(v.begin(), v.begin() + middle, v.end())`.
/// `middle` must be >= 1 (callers guarantee this; middle == 0 is UB in libstdc++).
pub fn std_partial_sort_by<T: Copy>(
    v: &mut [T],
    middle: usize,
    comp: &mut impl FnMut(&T, &T) -> bool,
) {
    debug_assert!(middle >= 1 && middle <= v.len());
    let last = v.len();
    heap_select(v, 0, middle, last, comp);
    sort_heap_range(v, 0, middle, comp);
}

// ---------------------------------------------------------------------------
// token-data sorting helpers (llama-sampler.cpp)
// ---------------------------------------------------------------------------

/// C `(int)float` cast with x86-64 `cvttss2si` semantics: NaN and out-of-range
/// values produce INT_MIN ("integer indefinite"). The pinned llama.cpp code
/// performs such casts in mirostat (`int(k)` where k can overflow) and in the
/// bucket-sort index computation; Rust's `as` cast saturates instead, so we
/// replicate the observed C behavior bit-for-bit.
#[inline]
fn c_int_cast(x: f32) -> i32 {
    if x.is_nan() || x >= 2147483648.0f32 || x <= -2147483649.0f32 {
        i32::MIN
    } else {
        x as i32
    }
}

fn sort_desc(a: &TokenData, b: &TokenData) -> bool {
    a.logit > b.logit
}

/// `llama_token_data_array_partial_sort` (bucket radix + std::sort/partial_sort).
/// Writes the result into `res` (which is resized to `nhave >= npartial`);
/// does not mutate `cur`.
fn token_data_array_partial_sort(cur: &[TokenData], npartial: usize, res: &mut Vec<TokenData>) {
    const NBUCKETS: usize = 128;
    const BUCKET_LOW: f32 = -10.0;
    const BUCKET_HIGH: f32 = 10.0;
    const BUCKET_SCALE: f32 = NBUCKETS as f32 / (BUCKET_HIGH - BUCKET_LOW);
    const BUCKET_INTER: f32 = -BUCKET_LOW * BUCKET_SCALE;

    let n = cur.len();
    let mut bucket_idx = Vec::with_capacity(n);
    let mut histo = [0i32; NBUCKETS];

    for d in cur.iter().take(n) {
        let val = d.logit;
        // int(bucket_scale * val + bucket_inter), clamped to [0, nbuckets-1]
        let mut ib = c_int_cast(BUCKET_SCALE * val + BUCKET_INTER);
        ib = ib.max(0).min(NBUCKETS as i32 - 1);
        bucket_idx.push(ib);
        histo[ib as usize] += 1;
    }
    let mut nhave: i32 = 0;
    let mut ib: i32 = NBUCKETS as i32 - 1;
    while ib >= 0 {
        nhave += histo[ib as usize];
        if nhave >= npartial as i32 {
            break;
        }
        ib -= 1;
    }
    debug_assert!(
        ib >= 0,
        "npartial > cur.len() is not supported (callers clamp k)"
    );
    res.resize(nhave.max(0) as usize, TokenData::default());

    // bucket_ptrs: running write offsets, top bucket first
    let mut bucket_ptrs = Vec::with_capacity((NBUCKETS as i32 - ib) as usize);
    {
        let mut ptr = 0usize;
        for j in (ib..NBUCKETS as i32).rev() {
            bucket_ptrs.push(ptr);
            ptr += histo[j as usize] as usize;
        }
    }
    for i in 0..n {
        let j = bucket_idx[i];
        if j >= ib {
            let idx = (NBUCKETS as i32 - 1 - j) as usize;
            res[bucket_ptrs[idx]] = cur[i];
            bucket_ptrs[idx] += 1;
        }
    }

    let mut ptr = 0usize;
    let mut ndone: i32 = 0;
    let mut j = NBUCKETS as i32 - 1;
    while j > ib {
        let h = histo[j as usize] as usize;
        std_sort_by(&mut res[ptr..ptr + h], &mut sort_desc);
        ptr += h;
        ndone += histo[j as usize];
        j -= 1;
    }
    let h_last = histo[ib as usize] as usize;
    let mid = (npartial as i32 - ndone) as usize;
    debug_assert!(mid >= 1 && mid <= h_last);
    std_partial_sort_by(&mut res[ptr..ptr + h_last], mid, &mut sort_desc);
}

/// `llama_token_data_array_partial_sort_inplace` — reduces `cur_p` to `npartial`
/// top entries (descending), sets `sorted = true`.
fn token_data_array_partial_sort_inplace(cur_p: &mut TokenDataArray, npartial: usize) {
    debug_assert!(npartial >= 1 && npartial <= cur_p.size);
    if npartial <= 128 {
        let size = cur_p.size;
        std_partial_sort_by(&mut cur_p.data[..size], npartial, &mut sort_desc);
        cur_p.size = npartial;
        cur_p.sorted = true;
        return;
    }

    let mut tmp = Vec::new();
    {
        let size = cur_p.size;
        let src: Vec<TokenData> = cur_p.data[..size].to_vec();
        token_data_array_partial_sort(&src, npartial, &mut tmp);
    }
    cur_p.data[..npartial].copy_from_slice(&tmp[..npartial]);

    cur_p.size = npartial;
    cur_p.sorted = true;
}

// ---------------------------------------------------------------------------
// shared sampler impls (static functions of llama-sampler.cpp)
// ---------------------------------------------------------------------------

/// `llama_sample_dist` — discrete_distribution sampling over `data[..size].p`.
pub fn sample_dist(cur_p: &TokenDataArray, rng: &mut Mt19937) -> usize {
    let probs: Vec<f32> = cur_p.data[..cur_p.size].iter().map(|d| d.p).collect();
    DiscreteDistribution::new(&probs).sample(rng)
}

/// `llama_sampler_temp_impl`
pub fn temp_impl(cur_p: &mut TokenDataArray, temp: f32) {
    if cur_p.size == 0 {
        return;
    }

    if temp <= 0.0 {
        // find the token with the highest logit and set the rest to -inf
        // NOTE: strict `>` — ties keep the FIRST occurrence (verbatim C behavior)
        let mut max_i = 0usize;
        let mut max_l = cur_p.data[0].logit;

        for i in 1..cur_p.size {
            if cur_p.data[i].logit > max_l {
                cur_p.data[max_i].logit = f32::NEG_INFINITY;
                max_i = i;
                max_l = cur_p.data[i].logit;
            } else {
                cur_p.data[i].logit = f32::NEG_INFINITY;
            }
        }

        return;
    }

    for i in 0..cur_p.size {
        cur_p.data[i].logit /= temp;
    }
}

/// `llama_sampler_softmax_impl`
pub fn softmax_impl(cur_p: &mut TokenDataArray, do_sort: bool) {
    assert!(cur_p.size > 0);

    // sort the logits in descending order if requested
    if do_sort && !cur_p.sorted {
        token_data_array_partial_sort_inplace(cur_p, cur_p.size);
    }

    let mut max_l = cur_p.data[0].logit;
    if !cur_p.sorted {
        for i in 1..cur_p.size {
            // std::max(a, b) == (a < b) ? b : a
            max_l = if max_l < cur_p.data[i].logit {
                cur_p.data[i].logit
            } else {
                max_l
            };
        }
    }

    let mut cum_sum = 0.0f32;

    for i in 0..cur_p.size {
        let p = (cur_p.data[i].logit - max_l).exp();
        cur_p.data[i].p = p;
        cum_sum += p;
    }

    for i in 0..cur_p.size {
        cur_p.data[i].p /= cum_sum;
    }
}

/// `llama_sampler_top_k_impl`
pub fn top_k_impl(cur_p: &mut TokenDataArray, k: i32) {
    if k <= 0 {
        return;
    }

    let k = (k as usize).min(cur_p.size);

    // sort scores in descending order
    if !cur_p.sorted {
        token_data_array_partial_sort_inplace(cur_p, k);
    }

    cur_p.size = k;
}

// ---------------------------------------------------------------------------
// Sampler trait + chain (llama_sampler API)
// ---------------------------------------------------------------------------

/// Port of the `llama_sampler` vtable subset used by CPU sampling.
///
/// NOTE on the `apply` signature: the C `llama_sampler_apply` operates on a
/// `llama_token_data_array` (logits + probabilities + sorted flag), and so does
/// this trait. [`SamplerChain::apply_logits`] offers a logits-slice wrapper.
pub trait Sampler {
    fn name(&self) -> &'static str;

    /// `llama_sampler_apply` (required in C)
    fn apply(&mut self, cur: &mut TokenDataArray);

    /// `llama_sampler_accept` (optional in C — updates e.g. penalty ring buffers)
    fn accept(&mut self, _token: LlamaToken) {}

    /// `llama_sampler_reset` (optional in C)
    fn reset(&mut self) {}

    /// `llama_sampler_get_seed` contribution — only dist / mirostat report a seed
    fn get_seed(&self) -> u32 {
        LLAMA_DEFAULT_SEED
    }

    /// Swap in an external RNG state (dist / mirostat only); returns true if handled.
    fn set_rng(&mut self, _rng: &mut Mt19937) -> bool {
        false
    }
}

/// `llama_sampler_init_empty` — no-op placeholder used for disabled samplers
/// (e.g. `init_top_k(0)` yields `Empty { name: "?top-k" }`).
pub struct EmptySampler {
    pub name: &'static str,
}

impl Sampler for EmptySampler {
    fn name(&self) -> &'static str {
        self.name
    }
    fn apply(&mut self, _cur: &mut TokenDataArray) {}
}

/// `struct llama_perf_sampler_data` (include/llama.h:1594-1598).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerfSamplerData {
    /// time needed for sampling in ms
    pub t_sample_ms: f64,
    /// number of sampled tokens
    pub n_sample: i32,
}

/// `llama_sampler_chain` (CPU path — the backend_* graph hooks are not ported).
#[derive(Default)]
pub struct SamplerChain {
    pub samplers: Vec<Box<dyn Sampler>>,
    /// `llama_sampler_chain::t_sample_us` / `n_sample`
    /// (llama-sampler.cpp:672's time_meas + :676 n_sample++).
    pub t_sample_us: i64,
    pub n_sample: i32,
}

impl SamplerChain {
    /// `llama_sampler_chain_init`
    pub fn new() -> Self {
        SamplerChain {
            samplers: Vec::new(),
            t_sample_us: 0,
            n_sample: 0,
        }
    }

    /// `llama_perf_sampler` (llama-sampler.cpp:4355-4368) —
    /// `llama_perf_sampler_data`.
    pub fn perf_sampler(&self) -> PerfSamplerData {
        PerfSamplerData {
            t_sample_ms: 1e-3 * self.t_sample_us as f64,
            n_sample: self.n_sample.max(0),
        }
    }

    /// `llama_perf_sampler_print` (llama-sampler.cpp:4370-4374).
    pub fn perf_sampler_print(&self) {
        let data = self.perf_sampler();
        crate::llama_log_info!(
            "llama_perf_sampler_print:    samplers time = {:10.2} ms / {:5} runs\n",
            data.t_sample_ms,
            data.n_sample
        );
    }

    /// `llama_perf_sampler_reset` (llama-sampler.cpp:4376-4385).
    pub fn perf_sampler_reset(&mut self) {
        self.t_sample_us = 0;
        self.n_sample = 0;
    }

    /// `llama_sampler_chain_add`
    pub fn add(&mut self, smpl: Box<dyn Sampler>) {
        self.samplers.push(smpl);
    }

    /// `llama_sampler_chain_apply` (llama-sampler.cpp:672 — a time_meas
    /// around the chain's apply, the perf counters behind
    /// `llama_perf_sampler`).
    pub fn apply(&mut self, cur_p: &mut TokenDataArray) {
        let t0 = crate::time_us();
        for smpl in self.samplers.iter_mut() {
            smpl.apply(cur_p);
        }
        self.t_sample_us += crate::time_us() - t0;
        self.n_sample += 1;
    }

    /// Convenience wrapper for logits-only callers (per the port spec):
    /// builds candidates with id == index, applies the chain, writes the
    /// surviving logits back in candidate order.
    pub fn apply_logits(&mut self, logits: &mut [f32]) {
        let mut cur = TokenDataArray::from_logits(logits);
        self.apply(&mut cur);
        for (i, d) in cur.data[..cur.size].iter().enumerate() {
            logits[i] = d.logit;
        }
    }

    /// `llama_sampler_chain_accept`
    pub fn accept(&mut self, token: LlamaToken) {
        for smpl in self.samplers.iter_mut() {
            smpl.accept(token);
        }
    }

    /// `llama_sampler_chain_reset`
    pub fn reset(&mut self) {
        for smpl in self.samplers.iter_mut() {
            smpl.reset();
        }
    }

    /// `llama_sampler_chain_n`
    pub fn n(&self) -> i32 {
        self.samplers.len() as i32
    }

    /// `llama_sampler_chain_get` (i >= 0; i == -1 "self" is not modeled)
    pub fn get(&self, i: i32) -> Option<&dyn Sampler> {
        if i < 0 || (i as usize) >= self.samplers.len() {
            return None;
        }
        Some(self.samplers[i as usize].as_ref())
    }

    /// `llama_sampler_chain_remove`
    pub fn remove(&mut self, i: i32) -> Option<Box<dyn Sampler>> {
        if i < 0 || (i as usize) >= self.samplers.len() {
            return None;
        }
        Some(self.samplers.remove(i as usize))
    }

    /// `llama_sampler_get_seed` on a chain — reverse order, first non-default seed
    pub fn get_seed(&self) -> u32 {
        for smpl in self.samplers.iter().rev() {
            let seed = smpl.get_seed();
            if seed != LLAMA_DEFAULT_SEED {
                return seed;
            }
        }
        LLAMA_DEFAULT_SEED
    }

    /// Swap an external RNG into the last RNG-owning sampler (dist/mirostat),
    /// returning it afterwards — used by `SamplingContext::sample_with_rng`.
    pub fn set_rng(&mut self, rng: &mut Mt19937) {
        for smpl in self.samplers.iter_mut().rev() {
            if smpl.set_rng(rng) {
                return;
            }
        }
    }

    /// `llama_sampler_sample` (CPU path): build candidates from `logits`,
    /// apply the chain, return the selected token id and `accept` it.
    /// The chain must end with a *selecting* sampler (greedy / dist / mirostat).
    pub fn sample(&mut self, logits: &[f32]) -> LlamaToken {
        let mut cur_p = TokenDataArray::from_logits(logits);

        self.apply(&mut cur_p);

        assert!(
            cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
            "llama_sampler_sample: chain did not select a token (end with greedy/dist/mirostat)"
        );

        let token = cur_p.data[cur_p.selected as usize].id;

        self.accept(token);

        token
    }
}

// ---------------------------------------------------------------------------
// samplers
// ---------------------------------------------------------------------------

// ---- greedy ----

/// `llama_sampler_greedy`.
///
/// DEVATION (per port spec): ties are broken by taking the LAST maximum, which
/// matches `ggml_vec_argmax_f32` (the argmax used by the ggml backend path).
/// The pinned CPU `llama_sampler_greedy_apply` uses a strict `>` comparison and
/// would keep the FIRST maximum — see the module report for the C reference
/// output (`logits [1,3,3,2]` selects index 1 in C++, index 2 here).
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        cur_p.selected = 0;
        for i in 1..cur_p.size {
            if cur_p.data[i].logit >= cur_p.data[cur_p.selected as usize].logit {
                cur_p.selected = i as i64;
            }
        }
    }
}

/// `llama_sampler_init_greedy`
pub fn init_greedy() -> Box<dyn Sampler> {
    Box::new(GreedySampler)
}

// ---- dist ----

/// `llama_sampler_dist` (CPU apply path — the `#if 1` cumulative-sampling fast
/// path with `std::uniform_real_distribution<double>`).
pub struct DistSampler {
    pub seed: u32,
    pub seed_cur: u32,
    pub rng: Mt19937,
}

impl Sampler for DistSampler {
    fn name(&self) -> &'static str {
        "dist"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        // edge cases
        if cur_p.size == 0 {
            cur_p.selected = -1;
            return;
        }

        cur_p.selected = 0;

        if cur_p.size == 1 {
            // keep the RNG state aligned with backend sampling (one draw per output)
            self.rng.canonical_double();
            cur_p.data[0].p = 1.0;
            return;
        }

        // max logit for numerical stability
        let mut max_l = cur_p.data[0].logit;
        if !cur_p.sorted {
            for i in 1..cur_p.size {
                max_l = if max_l < cur_p.data[i].logit {
                    cur_p.data[i].logit
                } else {
                    max_l
                };
            }
        }

        // apply softmax to obtain the probabilities
        let mut sum_cum = 0.0f64;
        for i in 0..cur_p.size {
            let p = (cur_p.data[i].logit - max_l).exp();
            cur_p.data[i].p = p;
            sum_cum += p as f64;
        }

        // sample from the obtained probabilities and normalize in a single pass
        let rnd = self.rng.canonical_double();

        let mut sum_run: f64 = 0.0;
        let sum_tgt: f64 = sum_cum * rnd;

        let mut found = false;
        for i in 0..cur_p.size {
            if !found {
                // accumulate probs until we reach the target sum
                sum_run += cur_p.data[i].p as f64;
                if sum_run >= sum_tgt {
                    cur_p.selected = i as i64;
                    found = true;
                }
            }

            // normalize probs (float /= double: computed in f64, narrowed to f32)
            cur_p.data[i].p = ((cur_p.data[i].p as f64) / sum_cum) as f32;
        }

        // fallback to the last token (don't think this can happen)
        debug_assert!(found);
        if !found {
            cur_p.selected = cur_p.size as i64 - 1;
        }
    }

    fn reset(&mut self) {
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.seed(self.seed_cur);
    }

    fn get_seed(&self) -> u32 {
        self.seed_cur
    }

    fn set_rng(&mut self, rng: &mut Mt19937) -> bool {
        std::mem::swap(&mut self.rng, rng);
        true
    }
}

/// `llama_sampler_init_dist`
pub fn init_dist(seed: u32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(DistSampler {
        seed,
        seed_cur,
        rng: Mt19937::new(seed_cur),
    })
}

// ---- top-k ----

/// `llama_sampler_top_k`
pub struct TopKSampler {
    pub k: i32,
}

impl Sampler for TopKSampler {
    fn name(&self) -> &'static str {
        "top-k"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        top_k_impl(cur_p, self.k);
    }
}

/// `llama_sampler_init_top_k` (k <= 0 -> empty "?top-k")
pub fn init_top_k(k: i32) -> Box<dyn Sampler> {
    if k <= 0 {
        return Box::new(EmptySampler { name: "?top-k" });
    }
    Box::new(TopKSampler { k })
}

// ---- top-p ----

/// `llama_sampler_top_p`
pub struct TopPSampler {
    pub p: f32,
    pub min_keep: usize,
    buf_sort: Vec<TokenData>,
}

impl Sampler for TopPSampler {
    fn name(&self) -> &'static str {
        "top-p"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.p >= 1.0 {
            return;
        }

        softmax_impl(cur_p, false);

        let mut k = cur_p.size;
        // if not sorted, try adaptive top-k sorting
        let mut use_buf = false;
        if !cur_p.sorted && cur_p.size > 1024 {
            k = 256.min(cur_p.size);
            let size = cur_p.size;
            let src: Vec<TokenData> = cur_p.data[..size].to_vec();
            token_data_array_partial_sort(&src, k, &mut self.buf_sort);
            use_buf = true;
        } else if !cur_p.sorted {
            // small candidates -> sort inplace
            token_data_array_partial_sort_inplace(cur_p, k);
        }

        // compute the cumulative probabilities
        let mut cum_sum = 0.0f32;
        let mut last_idx = cur_p.size;

        let mut i = 0usize;
        while i < cur_p.size {
            let pi = if use_buf {
                self.buf_sort[i].p
            } else {
                cur_p.data[i].p
            };
            cum_sum += pi;

            // check if the running sum is at least p or if we have kept at least min_keep tokens
            if cum_sum >= self.p && i + 1 >= self.min_keep {
                last_idx = i + 1;
                break;
            }

            // we exceeded the current top-k heuristic -> increase k and continue
            if !cur_p.sorted && i == k - 1 {
                k = cur_p.size;
                let size = cur_p.size;
                let src: Vec<TokenData> = cur_p.data[..size].to_vec();
                token_data_array_partial_sort(&src, k, &mut self.buf_sort);
                use_buf = true;
            }
            i += 1;
        }

        // resize the output vector to keep only the top-p tokens
        if !cur_p.sorted {
            cur_p.data[..last_idx].copy_from_slice(&self.buf_sort[..last_idx]);
            cur_p.sorted = true;
        }

        cur_p.size = last_idx;
    }
}

/// `llama_sampler_init_top_p` (p >= 1.0 -> empty "?top-p")
pub fn init_top_p(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p >= 1.0 {
        return Box::new(EmptySampler { name: "?top-p" });
    }
    Box::new(TopPSampler {
        p,
        min_keep,
        buf_sort: Vec::new(),
    })
}

// ---- min-p ----

/// `llama_sampler_min_p`
pub struct MinPSampler {
    pub p: f32,
    pub min_keep: usize,
}

impl Sampler for MinPSampler {
    fn name(&self) -> &'static str {
        "min-p"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.p <= 0.0 || cur_p.size == 0 {
            return;
        }

        let mut min_p_applied = false;

        // if the candidates aren't sorted, try the unsorted implementation first
        if !cur_p.sorted {
            let mut filtered_tokens: Vec<TokenData> = Vec::new();

            let mut max_logit = f32::MIN; // -FLT_MAX
            for i in 0..cur_p.size {
                max_l_set(&mut max_logit, cur_p.data[i].logit);
            }
            let min_logit = max_logit + self.p.ln(); // min logit for p_i >= p * p_max

            for i in 0..cur_p.size {
                if cur_p.data[i].logit >= min_logit {
                    filtered_tokens.push(cur_p.data[i]);
                }
            }

            // if we have enough values the operation was a success
            if !filtered_tokens.is_empty() && filtered_tokens.len() >= self.min_keep {
                cur_p.data[..filtered_tokens.len()].copy_from_slice(&filtered_tokens);
                cur_p.size = filtered_tokens.len();
                min_p_applied = true;
            }
        }

        // if the candidates are sorted or the unsorted implementation failed
        if !min_p_applied {
            // sort the logits in descending order
            if !cur_p.sorted {
                token_data_array_partial_sort_inplace(cur_p, cur_p.size);
            }

            let min_logit = cur_p.data[0].logit + self.p.ln();

            let mut i = 1usize; // first token always matches
            while i < cur_p.size {
                if cur_p.data[i].logit < min_logit && i >= self.min_keep {
                    break; // prob too small
                }
                i += 1;
            }

            // resize the output vector to keep only the matching tokens
            cur_p.size = i;
        }
    }
}

#[inline]
fn max_l_set(cur: &mut f32, x: f32) {
    // std::max(cur, x) == (cur < x) ? x : cur
    if *cur < x {
        *cur = x;
    }
}

/// `llama_sampler_init_min_p` (p <= 0 -> empty "?min-p")
pub fn init_min_p(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p <= 0.0 {
        return Box::new(EmptySampler { name: "?min-p" });
    }
    Box::new(MinPSampler { p, min_keep })
}

// ---- typical ----

/// `llama_sampler_typical`
pub struct TypicalSampler {
    pub p: f32,
    pub min_keep: usize,
}

impl Sampler for TypicalSampler {
    fn name(&self) -> &'static str {
        "typical"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.p >= 1.0 {
            return;
        }

        // compute the softmax of logits and calculate entropy
        softmax_impl(cur_p, true);

        let mut entropy = 0.0f32;
        for i in 0..cur_p.size {
            entropy += -cur_p.data[i].p * cur_p.data[i].p.ln();
        }

        // compute the absolute difference between negative log probability and entropy
        let mut shifted_scores = Vec::with_capacity(cur_p.size);
        for i in 0..cur_p.size {
            let shifted_score = (-cur_p.data[i].p.ln() - entropy).abs();
            shifted_scores.push(shifted_score);
        }

        // sort tokens based on the shifted_scores and their corresponding indices
        let mut indices: Vec<usize> = (0..cur_p.size).collect();
        std_sort_by(&mut indices, &mut |a, b| {
            shifted_scores[*a] < shifted_scores[*b]
        });

        // compute the cumulative probabilities
        let mut cum_sum = 0.0f32;
        let mut last_idx = indices.len();

        for (i, &idx) in indices.iter().enumerate() {
            cum_sum += cur_p.data[idx].p;

            // check if the running sum is greater than typical or min_keep is satisfied
            if cum_sum > self.p && (self.min_keep == 0 || i >= self.min_keep - 1) {
                last_idx = i + 1;
                break;
            }
        }

        // resize the output vector to keep only the locally typical tokens
        let mut cur_p_new = Vec::with_capacity(last_idx);
        for &idx in indices.iter().take(last_idx) {
            cur_p_new.push(cur_p.data[idx]);
        }

        // replace the data in cur_p with the cur_p_new data
        let n = cur_p_new.len();
        cur_p.data[..n].copy_from_slice(&cur_p_new);
        cur_p.size = n;
        cur_p.sorted = false;
    }
}

/// `llama_sampler_init_typical` (p >= 1.0 -> empty "?typical")
pub fn init_typical(p: f32, min_keep: usize) -> Box<dyn Sampler> {
    if p >= 1.0 {
        return Box::new(EmptySampler { name: "?typical" });
    }
    Box::new(TypicalSampler { p, min_keep })
}

// ---- temp ----

/// `llama_sampler_temp`
pub struct TempSampler {
    pub temp: f32,
}

impl Sampler for TempSampler {
    fn name(&self) -> &'static str {
        "temp"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        temp_impl(cur_p, self.temp);
    }
}

/// `llama_sampler_init_temp` (temp == 1.0 -> empty "?temp")
pub fn init_temp(temp: f32) -> Box<dyn Sampler> {
    if temp == 1.0 {
        return Box::new(EmptySampler { name: "?temp" });
    }
    Box::new(TempSampler { temp })
}

// ---- temp-ext ----

/// `llama_sampler_temp_ext` (dynamic entropy-mapped temperature)
pub struct TempExtSampler {
    pub temp: f32,
    pub delta: f32,
    pub exponent: f32,
}

impl Sampler for TempExtSampler {
    fn name(&self) -> &'static str {
        "temp-ext"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.delta > 0.0 {
            let min_temp = (self.temp - self.delta).max(0.0);
            let max_temp = self.temp + self.delta;

            let exponent_val = self.exponent;

            // no need to do anything if there is only one (or zero) candidates
            if cur_p.size <= 1 {
                return;
            }

            // calculate maximum possible entropy
            let max_entropy = -(1.0f32 / cur_p.size as f32).ln();

            softmax_impl(cur_p, true);

            // calculate entropy of the softmax probabilities
            let mut entropy = 0.0f32;
            for i in 0..cur_p.size {
                let prob = cur_p.data[i].p;
                if prob > 0.0 {
                    entropy -= prob * prob.ln();
                }
            }

            // normalize the entropy
            let normalized_entropy = entropy / max_entropy;

            // map the normalized entropy to the desired temperature range
            let dyn_temp = min_temp + (max_temp - min_temp) * normalized_entropy.powf(exponent_val);

            // apply the dynamically calculated temperature scaling
            temp_impl(cur_p, dyn_temp);

            // re-compute softmax probabilities after scaling
            let max_l_double = cur_p.data[0].logit as f64;

            let mut cum_sum_double = 0.0f64;
            for i in 0..cur_p.size {
                let p = ((cur_p.data[i].logit as f64) - max_l_double).exp();
                cur_p.data[i].p = p as f32; // store the scaled probability
                cum_sum_double += p;
            }

            for i in 0..cur_p.size {
                // p /= cum_sum (float /= double: f64 division, narrowed)
                cur_p.data[i].p = ((cur_p.data[i].p as f64) / cum_sum_double) as f32;
            }
        } else {
            temp_impl(cur_p, self.temp);
        }
    }
}

/// `llama_sampler_init_temp_ext` (temp == 1 && delta <= 0 -> empty "?temp-ext")
pub fn init_temp_ext(temp: f32, delta: f32, exponent: f32) -> Box<dyn Sampler> {
    if temp == 1.0 && delta <= 0.0 {
        return Box::new(EmptySampler { name: "?temp-ext" });
    }
    Box::new(TempExtSampler {
        temp,
        delta,
        exponent,
    })
}

// ---- xtc ----

/// `llama_sampler_xtc`
pub struct XtcSampler {
    pub probability: f32,
    pub threshold: f32,
    pub min_keep: usize,
    pub seed: u32,
    pub seed_cur: u32,
    pub rng: Mt19937,
}

impl Sampler for XtcSampler {
    fn name(&self) -> &'static str {
        "xtc"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.probability <= 0.0 || self.threshold > 0.5 || cur_p.size < 2 {
            return;
        }

        // std::uniform_real_distribution<float>(0, 1)
        let chance = self.rng.canonical_float();
        if chance > self.probability {
            return;
        }

        softmax_impl(cur_p, true);

        let mut pos_last = 0usize;

        for i in 0..cur_p.size {
            if cur_p.data[i].p >= self.threshold {
                pos_last = i;
            } else {
                break;
            }
        }

        if cur_p.size - pos_last >= self.min_keep && pos_last > 0 {
            // cur_p->data += pos_last
            cur_p.data.drain(0..pos_last);
            cur_p.size -= pos_last;
            if cur_p.selected >= 0 {
                // keep the selected index consistent with the shifted data
                // (C does not adjust it; mirror C exactly — leave as-is)
            }
        }
    }

    fn reset(&mut self) {
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.seed(self.seed_cur);
    }
}

/// `llama_sampler_init_xtc` (p <= 0 || t > 0.5 -> empty "?xtc")
pub fn init_xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Box<dyn Sampler> {
    if p <= 0.0 || t > 0.5 {
        return Box::new(EmptySampler { name: "?xtc" });
    }
    let seed_cur = get_rng_seed(seed);
    Box::new(XtcSampler {
        probability: p,
        threshold: t,
        min_keep,
        seed,
        seed_cur,
        rng: Mt19937::new(seed_cur),
    })
}

// ---- top-n-sigma ----

/// `llama_sampler_top_n_sigma`
pub struct TopNSigmaSampler {
    pub n: f32,
}

impl Sampler for TopNSigmaSampler {
    fn name(&self) -> &'static str {
        "top-n-sigma"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.n <= 0.0 || cur_p.size <= 1 {
            return;
        }

        // find max logit and calculate mean (skip -INF)
        let mut max = cur_p.data[0].logit;
        let mut logits_sum = 0.0f32;
        let mut valid_count = 0usize;
        for i in 0..cur_p.size {
            if cur_p.data[i].logit != f32::NEG_INFINITY {
                max_l_set(&mut max, cur_p.data[i].logit);
                logits_sum += cur_p.data[i].logit;
                valid_count += 1;
            }
        }
        let mean = if valid_count > 0 {
            logits_sum / valid_count as f32
        } else {
            0.0
        };

        // calculate standard deviation
        // (C: pow(x, 2) on float args computes in double and narrows — identical
        //  to a float multiply because the double square of an f32 is exact)
        let mut acc = 0.0f32;
        for i in 0..cur_p.size {
            if cur_p.data[i].logit != f32::NEG_INFINITY {
                let d = cur_p.data[i].logit - mean;
                acc += d * d;
            }
        }
        let std_dev = if valid_count > 0 {
            (acc / valid_count as f32).sqrt()
        } else {
            0.0
        };

        // apply mask
        for i in 0..cur_p.size {
            if cur_p.data[i].logit < max - (self.n * std_dev) {
                cur_p.data[i].logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// `llama_sampler_init_top_n_sigma` (n <= 0 -> empty "?top-n-sigma")
pub fn init_top_n_sigma(n: f32) -> Box<dyn Sampler> {
    if n <= 0.0 {
        return Box::new(EmptySampler {
            name: "?top-n-sigma",
        });
    }
    Box::new(TopNSigmaSampler { n })
}

// ---- penalties ----

/// `llama_sampler_penalties`
pub struct PenaltiesSampler {
    pub n_vocab: i32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    /// ring of recently accepted tokens (capacity penalty_last_n)
    pub prev: RingBuffer<LlamaToken>,
    /// token id -> occurrence count within the ring window
    pub token_count: HashMap<LlamaToken, i32>,
}

impl PenaltiesSampler {
    /// `llama_sampler_penalties::is_disabled`
    pub fn is_disabled(&self) -> bool {
        Self::params_disabled(
            self.penalty_last_n,
            self.penalty_repeat,
            self.penalty_freq,
            self.penalty_present,
        )
    }

    pub fn params_disabled(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> bool {
        penalty_last_n == 0
            || (penalty_repeat == 1.0 && penalty_freq == 0.0 && penalty_present == 0.0)
    }
}

impl Sampler for PenaltiesSampler {
    fn name(&self) -> &'static str {
        "penalties"
    }

    /// `llama_sampler_penalties_accept` — ring buffer + count map update
    fn accept(&mut self, token: LlamaToken) {
        if self.penalty_last_n == 0 {
            return;
        }

        *self.token_count.entry(token).or_insert(0) += 1;

        // if the ring buffer is full, remove the oldest token
        if self.prev.len() >= self.penalty_last_n as usize {
            let old = self.prev.front();
            let c = self.token_count.get_mut(&old).unwrap();
            *c -= 1;
            if *c == 0 {
                self.token_count.remove(&old);
            }
        }

        self.prev.push_back(token);
    }

    /// `llama_sampler_penalties_apply`
    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.is_disabled() {
            return;
        }

        for i in 0..cur_p.size {
            let Some(&count) = self.token_count.get(&cur_p.data[i].id) else {
                continue;
            };

            debug_assert!(count > 0 && count <= self.penalty_last_n);

            // multiply for non-positive logits (dividing would boost negative logits)
            if cur_p.data[i].logit <= 0.0 {
                cur_p.data[i].logit *= self.penalty_repeat;
            } else {
                cur_p.data[i].logit /= self.penalty_repeat;
            }

            cur_p.data[i].logit -=
                count as f32 * self.penalty_freq + (count > 0) as i32 as f32 * self.penalty_present;
        }

        cur_p.sorted = false;
    }

    fn reset(&mut self) {
        self.prev.clear();
        self.token_count.clear();
    }
}

/// `llama_sampler_init_penalties` (disabled -> empty "?penalties")
pub fn init_penalties(
    n_vocab: i32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
) -> Box<dyn Sampler> {
    let penalty_last_n = penalty_last_n.max(0);

    if PenaltiesSampler::params_disabled(
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
    ) {
        return Box::new(EmptySampler { name: "?penalties" });
    }

    Box::new(PenaltiesSampler {
        n_vocab,
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present,
        prev: RingBuffer::new(penalty_last_n as usize),
        token_count: HashMap::new(),
    })
}

// ---- dry ----

/// `struct llama_sampler_dry` (llama-sampler.cpp:3317-3327).
///
/// `std::unordered_multimap<llama_token, std::vector<llama_token>>` is
/// represented as `head -> [tails]` (an empty tail = single-token breaker);
/// all uses (`equal_range` scans in `_apply` step 1/4 and the dedup in
/// `get_overlapping_token_sequences`) are order-independent over the equal
/// range, so the HashMap-of-Vec is behaviorally identical.
#[derive(Clone)]
pub struct DrySampler {
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    /// `dry_processed_breakers`
    pub dry_processed_breakers: HashMap<LlamaToken, Vec<Vec<LlamaToken>>>,
    pub dry_repeat_count: Vec<i32>,
    pub dry_max_token_repeat: HashMap<LlamaToken, i32>,
    pub last_tokens: RingBuffer<LlamaToken>,
}

/// `get_overlapping_token_sequences` (llama-sampler.cpp:3331-3369, ported
/// from Koboldcpp PR#982) — for every vocab token whose piece contains or
/// starts the breaker string `str`, record the head token plus the
/// tokenization of the unmatched remainder (clamped to `max_tail_len`).
fn get_overlapping_token_sequences(
    vocab: &crate::vocab::Vocab,
    str_: &[u8],
    token_sequences: &mut HashMap<LlamaToken, Vec<Vec<LlamaToken>>>,
    max_tail_len: i32,
) {
    for token_id in 0..vocab.n_tokens() as LlamaToken {
        // std::string word = vocab.detokenize({token_id}, true) — byte-exact
        // form of the C wrapper (remove_special=false, unparse_special=true)
        let word: Vec<u8> = vocab.detokenize_impl(&[token_id], false, true);
        if find_subslice(&word, str_).is_some() {
            token_sequences
                .entry(token_id)
                .or_default()
                .push(Vec::new());
        } else {
            let word_len = word.len();
            let str_len = str_.len();
            // size_t pos = -1; while ((pos = word.find(str[0], pos + 1)) != npos)
            let mut pos: i64 = -1;
            while let Some(next) = find_byte_from(&word, str_[0], (pos + 1) as usize) {
                pos = next as i64;
                let mut matched = true;
                let mut i: usize = 1;
                while i < str_len && (i + pos as usize) < word_len {
                    if word[pos as usize + i] != str_[i] {
                        matched = false;
                        break;
                    }
                    i += 1;
                }
                if matched {
                    // vocab.tokenize(str.substr(i), false, false)
                    let mut tokenization: Vec<LlamaToken> =
                        vocab.tokenize_bytes(&str_[i..], false, false);
                    if max_tail_len >= 0 && tokenization.len() > max_tail_len as usize {
                        tokenization.truncate(max_tail_len as usize);
                    }

                    // ensure we don't already have a duplicate matching tokenization
                    let tails = token_sequences.entry(token_id).or_default();
                    if !tails.iter().any(|t| *t == tokenization) {
                        tails.push(tokenization);
                    }
                }
            }
        }
    }
}

/// `std::string::find` on byte slices.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// `word.find(byte, from)` — first occurrence of `byte` at or after `from`.
fn find_byte_from(word: &[u8], byte: u8, from: usize) -> Option<usize> {
    if from >= word.len() {
        return None;
    }
    word[from..]
        .iter()
        .position(|&b| b == byte)
        .map(|i| i + from)
}

impl DrySampler {
    /// whether the sampler is active (mirrors the guard in `_accept`/`_apply`)
    fn enabled(&self) -> bool {
        self.dry_multiplier != 0.0 && self.dry_base >= 1.0 && self.dry_penalty_last_n != 0
    }
}

impl Sampler for DrySampler {
    /// `llama_sampler_dry_name` (:3372-3374)
    fn name(&self) -> &'static str {
        "dry"
    }

    /// `llama_sampler_dry_accept` (:3376-3383)
    fn accept(&mut self, token: LlamaToken) {
        if !self.enabled() {
            return;
        }
        self.last_tokens.push_back(token);
    }

    /// `llama_sampler_dry_apply` (:3385-3564, Koboldcpp PR#982) — the four
    /// steps: restart-sequence scan, reverse Z-algorithm, max-repeat map,
    /// logit penalty.
    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if !self.enabled() {
            return;
        }

        let last_n_repeat = (self.last_tokens.len() as i32).min(self.dry_penalty_last_n) as usize;

        if last_n_repeat as i32 <= self.dry_allowed_length {
            return;
        }

        self.dry_repeat_count = vec![0; last_n_repeat];
        self.dry_max_token_repeat.clear();

        // Step 1: look for restart sequences to limit the maximum repetition length.
        let mut rep_limit = last_n_repeat as i32;
        for i in 0..last_n_repeat as i32 {
            let token = self.last_tokens.rat(i as usize);
            let Some(tails) = self.dry_processed_breakers.get(&token) else {
                continue;
            };
            let mut longest_match: i32 = -1;
            for tail in tails.iter() {
                // (*it) does not contain the head character, so seq_len is
                // the restart sequence length minus 1
                let seq_len = tail.len() as i32;
                if seq_len > longest_match && seq_len <= i {
                    let mut matched = true;
                    for offset in 0..seq_len {
                        // the -1 when indexing last_tokens: we already matched the head
                        if tail[offset as usize] != self.last_tokens.rat((i - offset - 1) as usize)
                        {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        longest_match = seq_len;
                    }
                }
            }
            if longest_match >= 0 {
                // restart sequence starting `i` tokens from the end and
                // continuing for `longest_match` tokens
                rep_limit = i - longest_match;
                break;
            }
        }
        if rep_limit < self.dry_allowed_length {
            return;
        }

        // Step 2: reverse Z-algorithm over the last N tokens
        // (https://ivanyu.me/blog/2014/10/15/z-algorithm/)
        {
            let last = last_n_repeat as i32 - 1;

            let mut rt: i32 = 0;
            let mut lt: i32 = 0;

            for k in 1..last_n_repeat as i32 {
                if k > rt {
                    // if k is outside the current Z-box, do naive computation
                    let mut n = 0i32;
                    while (n + k) < last_n_repeat as i32
                        && self.last_tokens.rat(n as usize)
                            == self.last_tokens.rat((n + k) as usize)
                    {
                        n += 1;
                    }
                    self.dry_repeat_count[(last - k) as usize] = n.min(rep_limit);
                    if n > 0 {
                        lt = k;
                        rt = k + n - 1;
                    }
                } else {
                    // if k is inside the current Z-box, consider two cases
                    let p = k - lt; // pair index
                    let right_part_len = rt - k + 1;

                    if self.dry_repeat_count[(last - p) as usize] < right_part_len {
                        let n = self.dry_repeat_count[(last - p) as usize].min(rep_limit);
                        self.dry_repeat_count[(last - k) as usize] = n;
                    } else {
                        let mut i = rt + 1;
                        while i < last_n_repeat as i32
                            && self.last_tokens.rat(i as usize)
                                == self.last_tokens.rat((i - k) as usize)
                        {
                            i += 1;
                        }

                        let n = (i - k).min(rep_limit);
                        self.dry_repeat_count[(last - k) as usize] = n;
                        lt = k;
                        rt = i - 1;
                    }
                }
            }
        }

        // Step 3: maximum repeat length that each new token would extend
        for i in 0..(last_n_repeat as i32 - 1) {
            let repeat_len = self.dry_repeat_count[i as usize];
            if repeat_len >= self.dry_allowed_length {
                // this token ends a repeat, so the next token would continue one
                let token = self
                    .last_tokens
                    .rat((last_n_repeat as i32 - 2 - i) as usize);
                // track the maximum sequence ending in this token
                let e = self.dry_max_token_repeat.entry(token).or_insert(i32::MIN);
                if *e < repeat_len {
                    *e = repeat_len;
                }
            }
        }

        // Step 4: apply logit penalties based on the maximum repeat length
        //
        // clamp the exponent to avoid fp overflow in pow(base, exponent):
        // computed from base and the approximate log of FLT_MAX
        const FLOAT_MAX_LOG: f32 = 88.722_839_1;
        let mut max_exponent: i32 = 0;
        if self.dry_base > 1.000_001 {
            // int(FLOAT_MAX_LOG / std::log(float)) — float division, C cast
            max_exponent = c_int_cast(FLOAT_MAX_LOG / self.dry_base.ln());
        }

        for i in 0..cur_p.size {
            if let Some(&max_repeat) = self.dry_max_token_repeat.get(&cur_p.data[i].id) {
                // check all sequence breakers starting with this token
                let is_single_token_breaker = self
                    .dry_processed_breakers
                    .get(&cur_p.data[i].id)
                    .map(|tails| tails.iter().any(|t| t.is_empty()))
                    .unwrap_or(false);

                // apply penalty only if it's not a single-token sequence breaker
                if !is_single_token_breaker {
                    let mut repeat_exp = max_repeat - self.dry_allowed_length;
                    if max_exponent > 0 && repeat_exp > max_exponent {
                        repeat_exp = max_exponent;
                    }
                    // std::pow(float, int) promotes to double: the product and
                    // the pow are f64, narrowed once into the f32 penalty
                    let penalty = (self.dry_multiplier as f64
                        * f64::powf(self.dry_base as f64, repeat_exp as f64))
                        as f32;
                    cur_p.data[i].logit -= penalty;
                }
            }
        }

        cur_p.sorted = false;
    }

    /// `llama_sampler_dry_reset` (:3566-3571)
    fn reset(&mut self) {
        self.last_tokens.clear();
        self.dry_repeat_count.clear();
        self.dry_max_token_repeat.clear();
    }
}

/// `llama_sampler_init_dry` (llama-sampler.cpp:3640-3699). `vocab` is needed
/// only to process the raw string breakers (C always passes the model vocab);
/// a `None` vocab skips that processing — equivalent to passing
/// `seq_breakers == NULL` — and is only reachable through the legacy
/// [`SamplingContext::new`] constructor.
pub fn init_dry(
    vocab: Option<&crate::vocab::Vocab>,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    seq_breakers: &[String],
) -> Box<dyn Sampler> {
    let dry_penalty_last_n = dry_penalty_last_n.max(0);
    let mut processed_breakers: HashMap<LlamaToken, Vec<Vec<LlamaToken>>> = HashMap::new();

    const MAX_CHAR_LEN: usize = 40;
    const MAX_SEQ_LEN: i32 = 20;

    let dry_enabled = dry_multiplier != 0.0 && dry_base >= 1.0 && dry_penalty_last_n != 0;

    if !dry_enabled {
        return Box::new(EmptySampler { name: "?dry" });
    }

    if dry_enabled && !seq_breakers.is_empty() {
        if let Some(vocab) = vocab {
            // process sequence breakers
            for sb in seq_breakers {
                if sb.is_empty() {
                    // LLAMA_LOG_WARN("skipping null or empty DRY sequence breaker")
                    continue;
                }

                let mut sequence_break: Vec<u8> = sb.as_bytes().to_vec();
                if sequence_break.len() > MAX_CHAR_LEN {
                    // LLAMA_LOG_WARN("truncating DRY sequence breaker …")
                    sequence_break.truncate(MAX_CHAR_LEN);
                }

                get_overlapping_token_sequences(
                    vocab,
                    &sequence_break,
                    &mut processed_breakers,
                    MAX_SEQ_LEN,
                );
            }
        }
    }

    Box::new(DrySampler {
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        dry_processed_breakers: processed_breakers,
        dry_repeat_count: if dry_enabled {
            vec![0; dry_penalty_last_n as usize]
        } else {
            Vec::new()
        },
        dry_max_token_repeat: HashMap::new(),
        last_tokens: RingBuffer::new(if dry_enabled {
            dry_penalty_last_n as usize
        } else {
            0
        }),
    })
}

/// `llama_sampler_init_dry_testing` (llama-sampler.cpp:3702-3724) — the
/// wrapper used by test-sampling.cpp: token-based sequence breakers
/// (`breaker[0]` is the head, the rest the tail), no vocab needed.
pub fn init_dry_testing(
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_penalty_last_n: i32,
    seq_breakers: &[&[LlamaToken]],
) -> DrySampler {
    // llama_sampler_init_dry(&dummy_vocab, …, NULL, 0) — the C then writes
    // through result->ctx even for the disabled case (UB); require enabled
    assert!(
        dry_multiplier != 0.0 && dry_base >= 1.0 && dry_penalty_last_n != 0,
        "init_dry_testing requires dry enabled"
    );
    let mut ctx = DrySampler {
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n: dry_penalty_last_n.max(0),
        dry_processed_breakers: HashMap::new(),
        dry_repeat_count: Vec::new(),
        dry_max_token_repeat: HashMap::new(),
        last_tokens: RingBuffer::new(dry_penalty_last_n.max(0) as usize),
    };
    ctx.dry_repeat_count = vec![0; ctx.dry_penalty_last_n as usize];

    // process the token-based sequence breakers
    for breaker in seq_breakers {
        if breaker.is_empty() {
            continue;
        }
        let head_token = breaker[0];
        let tail_tokens: Vec<LlamaToken> = breaker[1..].to_vec();
        ctx.dry_processed_breakers
            .entry(head_token)
            .or_default()
            .push(tail_tokens);
    }

    ctx
}

// ---- adaptive-p ----

/// adaptive probability transformation constants
/// (llama-sampler.cpp:3753-3756)
const DISTRIBUTION_WIDTH: f32 = 0.3;
const PEAK_LOGIT_VALUE: f32 = 5.0;
const SHARPNESS: f32 = 10.0;
const INV_WIDTH: f32 = 1.0 / DISTRIBUTION_WIDTH;

/// `struct llama_sampler_adaptive_p` (llama-sampler.cpp:3727-3751) — EMA of
/// the *original* probabilities of selected tokens, used to compute an
/// adapted target at each sampling step (ref: llama.cpp PR#17927).
pub struct AdaptivePSampler {
    pub target: f32,   // negative = disabled
    pub decay: f32,    // EMA decay (0.0 - 0.99)
    pub seed: u32,     // original RNG seed
    pub seed_cur: u32, // actual RNG seed
    pub rng: Mt19937,
    pub weighted_sum: f32,        // sum(p_i * decay^i)
    pub total_weight: f32,        // sum(decay^i), converges to 1/(1-decay)
    pub original_probs: Vec<f32>, // pre-transform probs, cached for EMA update
    pub pending_token_id: LlamaToken,
    pub pending_token_idx: i32,
}

impl Sampler for AdaptivePSampler {
    /// `llama_sampler_adaptive_p_name` (:3758-3760)
    fn name(&self) -> &'static str {
        "adaptive-p"
    }

    /// `llama_sampler_adaptive_p_apply` (:3762-3804)
    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        softmax_impl(cur_p, false);

        if self.target < 0.0 {
            // at negative target values, adaptive-p is no-op: simply sample
            // from the existing distribution
            cur_p.selected = sample_dist(cur_p, &mut self.rng) as i64;
            return;
        }

        // store the original probabilities
        self.original_probs.resize(cur_p.size, 0.0);
        for i in 0..cur_p.size {
            self.original_probs[i] = cur_p.data[i].p;
        }

        // using the EMA, compute the adapted target probability
        let target = self.target.clamp(0.0, 1.0);
        let adapted_target = (if self.total_weight == 0.0 {
            target
        } else {
            2.0 * target - (self.weighted_sum / self.total_weight)
        })
        .clamp(0.0, 1.0);

        // adaptive probability transform: quadratic near target for fine
        // differentiation, transitioning to linear decay in the tails;
        // unbounded negative logits suppress far-from-target tokens after
        // the softmax
        for i in 0..cur_p.size {
            if cur_p.data[i].logit == f32::NEG_INFINITY {
                // don't transform logits that are -INFINITY (masked out by
                // e.g. min-p and top-p when backend sampling)
                continue;
            }
            let dist = ((cur_p.data[i].p - adapted_target) * INV_WIDTH).abs();
            cur_p.data[i].logit = PEAK_LOGIT_VALUE - SHARPNESS * dist * dist / (1.0 + dist);
        }

        // softmax and sample from the transformed distribution
        softmax_impl(cur_p, false);
        let idx = sample_dist(cur_p, &mut self.rng);
        cur_p.selected = idx as i64;

        // store the selected token ID for acceptance later
        self.pending_token_id = cur_p.data[idx].id;
        self.pending_token_idx = idx as i32;
    }

    /// `llama_sampler_adaptive_p_accept` (:3806-3818)
    fn accept(&mut self, token: LlamaToken) {
        if self.pending_token_id == token {
            assert!(
                self.pending_token_id != -1,
                "pending_token_id is LLAMA_TOKEN_NULL"
            );
            assert!(self.pending_token_idx != -1, "pending_token_idx is -1");
            // update EMA with the original probability of the selected token
            let p = self.original_probs[self.pending_token_idx as usize];
            self.weighted_sum = p + self.decay * self.weighted_sum;
            self.total_weight = 1.0 + self.decay * self.total_weight;
        }
        self.pending_token_id = -1;
        self.pending_token_idx = -1;
    }

    /// `llama_sampler_adaptive_p_reset` (:3820-3830)
    fn reset(&mut self) {
        // target/decay never change; original_probs is overwritten each apply
        self.weighted_sum = self.target / (1.0 - self.decay);
        self.total_weight = 1.0 / (1.0 - self.decay);
        self.pending_token_id = -1;
        self.pending_token_idx = -1;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.seed(self.seed_cur);
    }

    fn get_seed(&self) -> u32 {
        self.seed_cur
    }

    fn set_rng(&mut self, rng: &mut Mt19937) -> bool {
        std::mem::swap(&mut self.rng, rng);
        true
    }
}

/// `llama_sampler_init_adaptive_p` (llama-sampler.cpp:3883-3905)
pub fn init_adaptive_p(target: f32, decay: f32, seed: u32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    let clamped_decay = decay.clamp(0.0, 0.99);
    Box::new(AdaptivePSampler {
        target,
        decay: clamped_decay,
        seed,
        seed_cur,
        rng: Mt19937::new(seed_cur),
        weighted_sum: target / (1.0 - clamped_decay),
        total_weight: 1.0 / (1.0 - clamped_decay),
        original_probs: Vec::new(),
        pending_token_id: -1, // LLAMA_TOKEN_NULL
        pending_token_idx: -1,
    })
}

// ---- infill ----

/// `struct llama_sampler_infill` (llama-sampler.cpp:4070-4075) — the vocab
/// face (`token_to_piece(special=false)`, `is_eog`, `token_eot/eos`) is
/// snapshotted at init like the grammar sampler's `VocabPieces`.
pub struct InfillSampler {
    /// `vocab->token_to_piece(id, lstrip=0, special=false)` per token id
    pieces: Vec<Vec<u8>>,
    is_eog: Vec<bool>,
    token_eot: LlamaToken,
    token_eos: LlamaToken,
}

impl InfillSampler {
    fn is_eog(&self, id: LlamaToken) -> bool {
        self.is_eog.get(id as usize).copied().unwrap_or(false)
    }
}

impl Sampler for InfillSampler {
    /// `llama_sampler_infill_name` (:4077-4079)
    fn name(&self) -> &'static str {
        "infill"
    }

    /// `llama_sampler_infill_apply` (:4081-4250)
    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        softmax_impl(cur_p, true);

        let mut p_txt_sum = 0.0f32;
        let mut p_eog_sum = 0.0f32;

        for i in 0..cur_p.size {
            if self.is_eog(cur_p.data[i].id) {
                p_eog_sum += cur_p.data[i].p;
            } else {
                p_txt_sum += cur_p.data[i].p;
            }
        }

        if 3.0 * p_eog_sum * cur_p.size as f32 > p_txt_sum {
            // ratio p_txt/p_eog too low -> keep just the EOG tokens
            let size_org = cur_p.size;
            cur_p.size = 0;
            let mut p_sum = 0.0f32;

            for i in 0..size_org {
                if self.is_eog(cur_p.data[i].id) {
                    p_sum += cur_p.data[i].p;
                    let n = cur_p.size;
                    cur_p.data[n] = cur_p.data[i];
                    cur_p.size = n + 1;
                }
            }

            // normalize probs
            for i in 0..cur_p.size {
                cur_p.data[i].p /= p_sum;
            }

            return;
        }

        // combine tokens with common prefix
        for i0 in 0..cur_p.size {
            for i1 in 0..cur_p.size {
                if cur_p.data[i0].logit == f32::NEG_INFINITY {
                    break;
                }
                if i0 == i1 || cur_p.data[i1].logit == f32::NEG_INFINITY {
                    continue;
                }

                let piece0: &[u8] = &self.pieces[cur_p.data[i0].id as usize];
                let piece1: &[u8] = &self.pieces[cur_p.data[i1].id as usize];

                // token i0 is a prefix of token i1
                if !piece0.is_empty()
                    && piece0.len() <= piece1.len()
                    && &piece1[..piece0.len()] == piece0
                {
                    // merge into the token with higher probability
                    let (dst, src) = if cur_p.data[i1].p > cur_p.data[i0].p {
                        (i1, i0)
                    } else {
                        (i0, i1)
                    };

                    cur_p.data[dst].p += cur_p.data[src].p;
                    cur_p.data[src].logit = f32::NEG_INFINITY;
                    cur_p.data[src].p = 0.0;
                }
            }
        }

        let mut n_non_eog: usize = 0;

        let mut size_org = cur_p.size;
        let mut p_sum = 0.0f32;
        let mut thold = 0.2f32;

        cur_p.size = 0;
        for i in 0..size_org {
            let is_eog = self.is_eog(cur_p.data[i].id);

            if cur_p.data[i].p < thold && !is_eog {
                continue;
            }
            if !is_eog {
                n_non_eog += 1;
            }
            p_sum += cur_p.data[i].p;

            // keep this token
            let n = cur_p.size;
            cur_p.data[n] = cur_p.data[i];
            cur_p.size = n + 1;
        }

        // if no non-EOG tokens are left -> reduce cur_p to single EOT token
        if n_non_eog == 0 {
            cur_p.size = 1;
            cur_p.data[0].id = self.token_eot;
            if cur_p.data[0].id == -1 {
                // LLAMA_TOKEN_NULL
                cur_p.data[0].id = self.token_eos;
            }
            cur_p.data[0].logit = 1.0;
            assert!(cur_p.data[0].id != -1);
            return;
        }

        // normalize probs
        for i in 0..cur_p.size {
            cur_p.data[i].p /= p_sum;
        }

        size_org = cur_p.size;
        p_sum = 0.0;
        // C: `thold = 1.0/(n_non_eog + 1)` — double division, narrowed to f32
        thold = (1.0 / ((n_non_eog + 1) as f64)) as f32;

        cur_p.size = 0;
        for i in 0..size_org {
            let is_eog = self.is_eog(cur_p.data[i].id);

            if cur_p.data[i].p < thold && !is_eog {
                continue;
            }
            p_sum += cur_p.data[i].p;

            let n = cur_p.size;
            cur_p.data[n] = cur_p.data[i];
            cur_p.size = n + 1;
        }

        // normalize probs
        for i in 0..cur_p.size {
            cur_p.data[i].p /= p_sum;
        }
    }
}

/// `llama_sampler_init_infill` (llama-sampler.cpp:4288-4299) — snapshots the
/// vocab face at init (`buf0`/`buf1` are transient scratch in the C).
pub fn init_infill(vocab: &crate::vocab::Vocab) -> Box<dyn Sampler> {
    let n = vocab.n_tokens() as usize;
    let mut pieces = Vec::with_capacity(n);
    let mut is_eog = Vec::with_capacity(n);
    for id in 0..n {
        // vocab->token_to_piece(id, lstrip = 0, special = false)
        pieces.push(vocab.token_to_piece_special(id as LlamaToken, false));
        is_eog.push(vocab.is_eog(id as LlamaToken));
    }
    Box::new(InfillSampler {
        pieces,
        is_eog,
        token_eot: vocab.token_eot(),
        token_eos: vocab.token_eos(),
    })
}

// ---- logit-bias ----

/// `llama_sampler_logit_bias`
pub struct LogitBiasSampler {
    pub n_vocab: i32,
    pub logit_bias: Vec<LogitBias>,
    to_search: Vec<LogitBias>,
}

impl Sampler for LogitBiasSampler {
    fn name(&self) -> &'static str {
        "logit-bias"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        if self.logit_bias.is_empty() {
            return;
        }

        self.to_search.clear();

        // update the candidates that have not been shuffled (idx == id)
        for lb in self.logit_bias.iter() {
            let t = lb.token;
            if t >= 0 && cur_p.size > t as usize && cur_p.data[t as usize].id == t {
                cur_p.data[t as usize].logit += lb.bias;
            } else {
                self.to_search.push(*lb);
            }
        }

        if self.to_search.is_empty() {
            return;
        }

        // search for the remaining candidates
        for i in 0..cur_p.size {
            for lb in self.to_search.iter() {
                if cur_p.data[i].id == lb.token {
                    cur_p.data[i].logit += lb.bias;
                    break;
                }
            }
        }
    }
}

/// `llama_sampler_init_logit_bias` (n <= 0 -> empty "?logit-bias")
pub fn init_logit_bias(n_vocab: i32, logit_bias: &[LogitBias]) -> Box<dyn Sampler> {
    if logit_bias.is_empty() {
        return Box::new(EmptySampler {
            name: "?logit-bias",
        });
    }
    Box::new(LogitBiasSampler {
        n_vocab,
        logit_bias: logit_bias.to_vec(),
        to_search: Vec::new(),
    })
}

// ---- mirostat ----

/// `llama_sampler_mirostat` (v1)
pub struct MirostatSampler {
    pub n_vocab: i32,
    pub seed: u32,
    pub seed_cur: u32,
    pub tau: f32,
    pub eta: f32,
    pub m: i32,
    pub mu: f32,
    pub rng: Mt19937,
}

impl Sampler for MirostatSampler {
    fn name(&self) -> &'static str {
        "mirostat"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        softmax_impl(cur_p, true);

        // estimate s_hat using the most probable m tokens
        let mut sum_ti_bi = 0.0f32;
        let mut sum_ti_sq = 0.0f32;
        let lim = ((self.m - 1) as usize).min(cur_p.size - 1);
        for i in 0..lim {
            let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b_i = (cur_p.data[i].p / cur_p.data[i + 1].p).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        let s_hat = sum_ti_bi / sum_ti_sq;

        // compute k from the estimated s_hat and target surprise value
        let epsilon_hat = s_hat - 1.0;
        let k = ((epsilon_hat * 2.0f32.powf(self.mu))
            / (1.0 - (self.n_vocab as f32).powf(-epsilon_hat)))
        .powf(1.0 / s_hat);

        // int(k): k may overflow i32 (near-uniform distributions) — the C
        // conversion yields INT_MIN there, and std::max(int(k), 1) selects 1
        top_k_impl(cur_p, c_int_cast(k).max(1));

        softmax_impl(cur_p, true);

        let idx = sample_dist(cur_p, &mut self.rng);

        cur_p.selected = idx as i64;

        let observed_surprise = -cur_p.data[idx].p.log2();
        let e = observed_surprise - self.tau;

        // update mu using the learning rate and error
        self.mu = self.mu - self.eta * e;
    }

    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.seed(self.seed_cur);
    }

    fn get_seed(&self) -> u32 {
        self.seed_cur
    }

    fn set_rng(&mut self, rng: &mut Mt19937) -> bool {
        std::mem::swap(&mut self.rng, rng);
        true
    }
}

/// `llama_sampler_init_mirostat`
pub fn init_mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(MirostatSampler {
        n_vocab,
        seed,
        seed_cur,
        tau,
        eta,
        m,
        mu: 2.0 * tau,
        rng: Mt19937::new(seed_cur),
    })
}

// ---- mirostat v2 ----

/// `llama_sampler_mirostat_v2`
pub struct MirostatV2Sampler {
    pub seed: u32,
    pub seed_cur: u32,
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
    pub rng: Mt19937,
}

impl Sampler for MirostatV2Sampler {
    fn name(&self) -> &'static str {
        "mirostat-v2"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        softmax_impl(cur_p, true);

        // truncate the words with surprise values greater than mu
        let mut new_size = cur_p.size;
        for i in 0..cur_p.size {
            if -cur_p.data[i].p.log2() > self.mu {
                new_size = i;
                break;
            }
        }
        cur_p.size = new_size;

        if cur_p.size == 0 {
            cur_p.size = 1;
        }

        // normalize the probabilities of the remaining words
        softmax_impl(cur_p, true);

        let idx = sample_dist(cur_p, &mut self.rng);

        cur_p.selected = idx as i64;

        let observed_surprise = -cur_p.data[idx].p.log2();
        let e = observed_surprise - self.tau;

        // update mu using the learning rate and error
        self.mu = self.mu - self.eta * e;
    }

    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
        self.seed_cur = get_rng_seed(self.seed);
        self.rng.seed(self.seed_cur);
    }

    fn get_seed(&self) -> u32 {
        self.seed_cur
    }

    fn set_rng(&mut self, rng: &mut Mt19937) -> bool {
        std::mem::swap(&mut self.rng, rng);
        true
    }
}

/// `llama_sampler_init_mirostat_v2`
pub fn init_mirostat_v2(seed: u32, tau: f32, eta: f32) -> Box<dyn Sampler> {
    let seed_cur = get_rng_seed(seed);
    Box::new(MirostatV2Sampler {
        seed,
        seed_cur,
        tau,
        eta,
        mu: 2.0 * tau,
        rng: Mt19937::new(seed_cur),
    })
}

// ---------------------------------------------------------------------------
// common_sampler_type (common/common.h:220-232 + common/sampling.cpp:795-919)
// ---------------------------------------------------------------------------

/// `enum common_sampler_type` (common/common.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommonSamplerType {
    Dry,         // COMMON_SAMPLER_TYPE_DRY
    TopK,        // COMMON_SAMPLER_TYPE_TOP_K
    TypicalP,    // COMMON_SAMPLER_TYPE_TYPICAL_P
    TopP,        // COMMON_SAMPLER_TYPE_TOP_P
    TopNSigma,   // COMMON_SAMPLER_TYPE_TOP_N_SIGMA
    MinP,        // COMMON_SAMPLER_TYPE_MIN_P
    Temperature, // COMMON_SAMPLER_TYPE_TEMPERATURE
    Xtc,         // COMMON_SAMPLER_TYPE_XTC
    Infill,      // COMMON_SAMPLER_TYPE_INFILL
    Penalties,   // COMMON_SAMPLER_TYPE_PENALTIES
    AdaptiveP,   // COMMON_SAMPLER_TYPE_ADAPTIVE_P
}

impl CommonSamplerType {
    /// `common_sampler_type_to_chr` (common/sampling.cpp:795-810)
    pub fn to_chr(self) -> char {
        match self {
            CommonSamplerType::Dry => 'd',
            CommonSamplerType::TopK => 'k',
            CommonSamplerType::TypicalP => 'y',
            CommonSamplerType::TopP => 'p',
            CommonSamplerType::TopNSigma => 's',
            CommonSamplerType::MinP => 'm',
            CommonSamplerType::Temperature => 't',
            CommonSamplerType::Xtc => 'x',
            CommonSamplerType::Infill => 'i',
            CommonSamplerType::Penalties => 'e',
            CommonSamplerType::AdaptiveP => 'a',
        }
    }

    /// `common_sampler_type_to_str` (common/sampling.cpp:812-827)
    pub fn to_str(self) -> &'static str {
        match self {
            CommonSamplerType::Dry => "dry",
            CommonSamplerType::TopK => "top_k",
            CommonSamplerType::TypicalP => "typ_p",
            CommonSamplerType::TopP => "top_p",
            CommonSamplerType::TopNSigma => "top_n_sigma",
            CommonSamplerType::MinP => "min_p",
            CommonSamplerType::Temperature => "temperature",
            CommonSamplerType::Xtc => "xtc",
            CommonSamplerType::Infill => "infill",
            CommonSamplerType::Penalties => "penalties",
            CommonSamplerType::AdaptiveP => "adaptive_p",
        }
    }

    /// the default `common_params_sampling::samplers` chain
    /// (common/common.h:265-275)
    pub fn default_chain() -> Vec<CommonSamplerType> {
        vec![
            CommonSamplerType::Penalties,
            CommonSamplerType::Dry,
            CommonSamplerType::TopNSigma,
            CommonSamplerType::TopK,
            CommonSamplerType::TypicalP,
            CommonSamplerType::TopP,
            CommonSamplerType::MinP,
            CommonSamplerType::Xtc,
            CommonSamplerType::Temperature,
        ]
    }
}

/// `common_sampler_types_from_names` (common/sampling.cpp:829-889) — accepts
/// canonical names, kebab-case ("top-k"), no-dash ("topk") and the misc
/// aliases ("nucleus"/"temp"/"typ"), lower-cased; unmatched names are
/// skipped with a warning in the C.
pub fn common_sampler_types_from_names(names: &[String]) -> Vec<CommonSamplerType> {
    use std::collections::HashMap as Map;
    use std::sync::OnceLock;

    // canonical sampler name mapping (sampling.cpp:833-845)
    static MAP: OnceLock<Map<String, CommonSamplerType>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let canonical: Vec<(&str, CommonSamplerType)> = vec![
            ("dry", CommonSamplerType::Dry),
            ("top_k", CommonSamplerType::TopK),
            ("top_p", CommonSamplerType::TopP),
            ("top_n_sigma", CommonSamplerType::TopNSigma),
            ("typ_p", CommonSamplerType::TypicalP),
            ("min_p", CommonSamplerType::MinP),
            ("temperature", CommonSamplerType::Temperature),
            ("xtc", CommonSamplerType::Xtc),
            ("infill", CommonSamplerType::Infill),
            ("penalties", CommonSamplerType::Penalties),
            ("adaptive_p", CommonSamplerType::AdaptiveP),
        ];
        let mut m: Map<String, CommonSamplerType> = Map::new();
        for (name, ty) in &canonical {
            // kebab-case: "top-k", "min-p", etc.
            m.insert(name.replace('_', "-"), *ty);
            // no dash: "topk", "minp", etc.
            m.insert(name.replace('_', ""), *ty);
            // canonical names win (alias_name_map.merge(canonical_name_map))
            m.insert(name.to_string(), *ty);
        }
        // misc. aliases (sampling.cpp:866-868)
        m.insert("nucleus".into(), CommonSamplerType::TopP);
        m.insert("temp".into(), CommonSamplerType::Temperature);
        m.insert("typ".into(), CommonSamplerType::TypicalP);
        m
    });

    let mut samplers = Vec::with_capacity(names.len());
    for name in names {
        let name_lower = name.to_lowercase();
        if let Some(&t) = map.get(&name_lower) {
            samplers.push(t);
        }
        // else: LOG_WRN "unable to match sampler by name" — skipped
    }
    samplers
}

/// `common_sampler_types_from_chars` (common/sampling.cpp:891-919) — the
/// one-char-per-sampler form ("kfypm" == top_k,typical,top_p,min_p,temp);
/// unmatched chars are skipped with a warning in the C.
pub fn common_sampler_types_from_chars(chars: &str) -> Vec<CommonSamplerType> {
    let mut samplers = Vec::with_capacity(chars.len());
    for c in chars.chars() {
        let t = match c {
            'd' => CommonSamplerType::Dry,
            'k' => CommonSamplerType::TopK,
            'y' => CommonSamplerType::TypicalP,
            'p' => CommonSamplerType::TopP,
            's' => CommonSamplerType::TopNSigma,
            'm' => CommonSamplerType::MinP,
            't' => CommonSamplerType::Temperature,
            'x' => CommonSamplerType::Xtc,
            'i' => CommonSamplerType::Infill,
            'e' => CommonSamplerType::Penalties,
            'a' => CommonSamplerType::AdaptiveP,
            _ => continue, // LOG_WRN "unable to match sampler by char"
        };
        samplers.push(t);
    }
    samplers
}

// ---------------------------------------------------------------------------
// top-level convenience API (common/common.h common_params_sampling +
// common/sampling.cpp common_sampler — the replacement for the removed
// llama_sampling_context / llama_sampling_params API)
// ---------------------------------------------------------------------------

/// Subset of `common_params_sampling` (common/common.h) with the field defaults
/// of the pinned tree. `n_vocab` is passed to `SamplingContext::new` instead
/// (mirrors `common_sampler_init(vocab, params)`).
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub seed: u32,              // LLAMA_DEFAULT_SEED
    pub n_prev: i32,            // 64
    pub min_keep: usize,        // 0
    pub top_k: i32,             // 40 (<= 0 to use vocab size)
    pub top_p: f32,             // 0.95 (1.0 = disabled)
    pub min_p: f32,             // 0.05 (0.0 = disabled)
    pub xtc_probability: f32,   // 0.0 = disabled
    pub xtc_threshold: f32,     // 0.10 (> 0.5 disables)
    pub typ_p: f32,             // 1.0 = disabled
    pub temp: f32,              // 0.80 (<= 0 = greedy)
    pub dynatemp_range: f32,    // 0.0 = disabled
    pub dynatemp_exponent: f32, // 1.0
    pub penalty_last_n: i32,    // 64 (0 = disabled)
    pub penalty_repeat: f32,    // 1.0 = disabled
    pub penalty_freq: f32,      // 0.0 = disabled
    pub penalty_present: f32,   // 0.0 = disabled
    pub mirostat: i32,          // 0 = disabled, 1 = v1, 2 = v2
    pub top_n_sigma: f32,       // -1.0 = disabled
    pub mirostat_tau: f32,      // 5.0
    pub mirostat_eta: f32,      // 0.1
    /// `dry_multiplier` (common.h:239, 0.0 = disabled)
    pub dry_multiplier: f32,
    /// `dry_base` (common.h:240, default 1.75)
    pub dry_base: f32,
    /// `dry_allowed_length` (common.h:241, default 2)
    pub dry_allowed_length: i32,
    /// `dry_penalty_last_n` (common.h:242, default 64, 0 = disabled)
    pub dry_penalty_last_n: i32,
    /// `dry_sequence_breakers` (common.h:259 — default `["\n", ":", "\"", "*"]`)
    pub dry_sequence_breakers: Vec<String>,
    /// `adaptive_target` (common.h:243, -1.0 = disabled)
    pub adaptive_target: f32,
    /// `adaptive_decay` (common.h:244, default 0.90)
    pub adaptive_decay: f32,
    /// `samplers` (common.h:265-275) — the chain order; the C default is
    /// penalties, dry, top-n-sigma, top-k, typical, top-p, min-p, xtc, temperature
    pub samplers: Vec<CommonSamplerType>,
    /// user logit biases (common: also merged with vocab suppress tokens — the
    /// caller can pre-merge by pushing `{token, -inf}` entries here)
    pub logit_bias: Vec<LogitBias>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            seed: LLAMA_DEFAULT_SEED,
            n_prev: 64,
            min_keep: 0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            xtc_probability: 0.0,
            xtc_threshold: 0.10,
            typ_p: 1.0,
            temp: 0.80,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            mirostat: 0,
            top_n_sigma: -1.0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_sequence_breakers: vec![
                "\n".to_string(),
                ":".to_string(),
                "\"".to_string(),
                "*".to_string(),
            ],
            adaptive_target: -1.0,
            adaptive_decay: 0.90,
            samplers: CommonSamplerType::default_chain(),
            logit_bias: Vec::new(),
        }
    }
}

/// `common_sampler` — the convenience sampling context (replacement for the
/// removed `llama_sampling_context`). The chain is built in `params.samplers`
/// order exactly like `common_sampler_init` (common/sampling.cpp:340-413):
/// logit_bias (if any) first, then the samplers list (penalties, dry,
/// top-n-sigma, top-k, typical, top-p, min-p, xtc, temp-ext by default),
/// then `dist` — or `adaptive-p` at the very end when the user included it
/// in the list (replacing `dist`, sampling.cpp:383-400); mirostat v1/v2
/// replace the whole list with `temp` + mirostat.
pub struct SamplingContext {
    pub params: SamplingParams,
    pub chain: SamplerChain,
    /// `common_sampler::prev` — ring of recent tokens (max(32, n_prev))
    pub prev: RingBuffer<LlamaToken>,
    /// `common_sampler::cur` — candidates of the last sample() call
    pub cur: Vec<TokenData>,
    /// `common_sampler::rng` (common/sampling.cpp:125-126) — for rejection
    /// sampling; independent of the draft, or the target distribution is not
    /// preserved (upstream a7b94df2c)
    pub rng: Mt19937,
}

impl SamplingContext {
    /// `common_sampler_init` (subset: no grammar / reasoning-budget).
    /// Kept for the pre-DRY call sites: string DRY sequence breakers cannot
    /// be processed without a vocab (equivalent to passing `NULL` breakers).
    pub fn new(n_vocab: i32, params: SamplingParams) -> Self {
        Self::new_with_vocab(n_vocab, None, params)
    }

    /// `common_sampler_init` (common/sampling.cpp:187-438) with the model
    /// vocab — required to lower the DRY string sequence breakers into
    /// token sequences (`get_overlapping_token_sequences`).
    pub fn new_with_vocab(
        n_vocab: i32,
        vocab: Option<&crate::vocab::Vocab>,
        params: SamplingParams,
    ) -> Self {
        let mut samplers: Vec<Box<dyn Sampler>> = Vec::new();

        // logit bias: user biases + model suppress tokens are pre-merged into
        // params.logit_bias by the caller (sampling.cpp:325-338)
        if !params.logit_bias.is_empty() {
            samplers.push(init_logit_bias(n_vocab, &params.logit_bias));
        }

        if params.mirostat == 0 {
            // see below
            let mut use_adaptive_p = false;

            for cnstr in params.samplers.iter() {
                match cnstr {
                    CommonSamplerType::Dry => {
                        samplers.push(init_dry(
                            vocab,
                            params.dry_multiplier,
                            params.dry_base,
                            params.dry_allowed_length,
                            params.dry_penalty_last_n,
                            &params.dry_sequence_breakers,
                        ));
                    }
                    CommonSamplerType::TopK => {
                        samplers.push(init_top_k(params.top_k));
                    }
                    CommonSamplerType::TopP => {
                        samplers.push(init_top_p(params.top_p, params.min_keep));
                    }
                    CommonSamplerType::TopNSigma => {
                        samplers.push(init_top_n_sigma(params.top_n_sigma));
                    }
                    CommonSamplerType::MinP => {
                        samplers.push(init_min_p(params.min_p, params.min_keep));
                    }
                    CommonSamplerType::Xtc => {
                        samplers.push(init_xtc(
                            params.xtc_probability,
                            params.xtc_threshold,
                            params.min_keep,
                            params.seed,
                        ));
                    }
                    CommonSamplerType::TypicalP => {
                        samplers.push(init_typical(params.typ_p, params.min_keep));
                    }
                    CommonSamplerType::Temperature => {
                        samplers.push(init_temp_ext(
                            params.temp,
                            params.dynatemp_range,
                            params.dynatemp_exponent,
                        ));
                    }
                    CommonSamplerType::Infill => {
                        // the infill sampler needs the vocab face
                        // (llama_sampler_init_infill, llama-sampler.cpp:4288)
                        let Some(v) = vocab else {
                            panic!("the infill sampler requires a vocab");
                        };
                        samplers.push(init_infill(v));
                    }
                    CommonSamplerType::Penalties => {
                        samplers.push(init_penalties(
                            n_vocab,
                            params.penalty_last_n,
                            params.penalty_repeat,
                            params.penalty_freq,
                            params.penalty_present,
                        ));
                    }
                    CommonSamplerType::AdaptiveP => {
                        // the `adaptive-p` sampler is like `dist` and `mirostat`
                        // in that it selects a single token, so it is added at
                        // the end of the chain by default unless the user
                        // specifically included `adaptive-p` (sampling.cpp:383-389)
                        use_adaptive_p = true;
                    }
                }
            }
            if use_adaptive_p {
                // only if user explicitly included adaptive-p sampler
                samplers.push(init_adaptive_p(
                    params.adaptive_target,
                    params.adaptive_decay,
                    params.seed,
                ));
            } else {
                // default: sample from distribution
                samplers.push(init_dist(params.seed));
            }
        } else if params.mirostat == 1 {
            samplers.push(init_temp(params.temp));
            samplers.push(init_mirostat(
                n_vocab,
                params.seed,
                params.mirostat_tau,
                params.mirostat_eta,
                100,
            ));
        } else if params.mirostat == 2 {
            samplers.push(init_temp(params.temp));
            samplers.push(init_mirostat_v2(
                params.seed,
                params.mirostat_tau,
                params.mirostat_eta,
            ));
        } else {
            panic!("unknown mirostat version");
        }

        let mut chain = SamplerChain::new();
        for s in samplers {
            chain.add(s);
        }

        // `/* .rng = */ std::mt19937(llama_sampler_get_seed(chain) ^
        // 0x9e3779b9u)` (common/sampling.cpp:439-441) — mix it, the chain and
        // the draft are seeded from this one too
        let rng = Mt19937::new(chain.get_seed() ^ 0x9e37_79b9);

        SamplingContext {
            prev: RingBuffer::new(32.max(params.n_prev as usize) as usize),
            params,
            chain,
            cur: Vec::new(),
            rng,
        }
    }

    /// `common_sampler_sample` -> `llama_sampler_sample` (CPU path).
    /// The chain's `dist` (or mirostat) sampler advances its own RNG.
    pub fn sample(&mut self, logits: &[f32]) -> LlamaToken {
        let mut cur_p = TokenDataArray::from_logits(logits);

        self.chain.apply(&mut cur_p);

        assert!(
            cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
            "SamplingContext::sample: chain did not select a token"
        );

        let token = cur_p.data[cur_p.selected as usize].id;
        self.cur = cur_p.data[..cur_p.size].to_vec();

        self.chain.accept(token);
        self.prev.push_back(token);

        token
    }

    /// `common_sampler_get_candidates` (common/sampling.cpp:861-881): the
    /// candidates of the last sample, sorted by `p` descending when requested
    /// (the C sorts in place through `cur_p.sorted`; the port re-sorts the
    /// `cur` snapshot, same ordering `data[0]` reads see).
    pub fn get_candidates(&mut self, do_sort: bool) -> &[TokenData] {
        if do_sort {
            self.cur.sort_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(std::cmp::Ordering::Equal));
        }
        &self.cur
    }

    /// `common_sampler_sample` (common/sampling.cpp:594-676) WITHOUT the
    /// trailing accept — the C splits sampling from `common_sampler_accept`,
    /// and the rejection verifier needs that split (it samples a fallback
    /// token but accepts the drafted one). Grammar variant of the
    /// grammar_first=false flow: the sampled token is checked against the
    /// grammar as a single candidate, and an invalid token is resampled with
    /// the grammar applied before the chain (:650-678).
    fn sample_no_accept(
        &mut self,
        logits: &[f32],
        grmr: Option<&mut GrammarSampler>,
        grammar_first: bool,
    ) -> LlamaToken {
        let sample_once = |chain: &mut SamplerChain, cur_p: &mut TokenDataArray| {
            chain.apply(cur_p);
            assert!(
                cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
                "no selected token during sampling - check your sampling configuration"
            );
            cur_p.data[cur_p.selected as usize].id
        };

        if grmr.is_none() {
            // `if (grammar_first || !grammar_should_apply(gsmpl)) return id;`
            let mut cur_p = TokenDataArray::from_logits(logits);
            let id = sample_once(&mut self.chain, &mut cur_p);
            self.cur = cur_p.data[..cur_p.size].to_vec();
            return id;
        }

        let mut grmr = grmr.unwrap();

        let mut cur_p = TokenDataArray::from_logits(logits);
        if grammar_first {
            // `llama_sampler_apply(grmr, &cur_p)` before the chain (:642-644)
            grmr.apply_to(&mut cur_p);
        }
        let id = sample_once(&mut self.chain, &mut cur_p);
        self.cur = cur_p.data[..cur_p.size].to_vec();
        if grammar_first {
            return id;
        }

        // check if the sampled token fits the grammar (grammar-based
        // rejection sampling, :650-663)
        {
            let mut single = TokenDataArray {
                data: vec![TokenData { id, logit: 1.0, p: 0.0 }],
                size: 1,
                selected: -1,
                sorted: false,
            };
            grmr.apply_to(&mut single);
            if single.data[0].logit != f32::NEG_INFINITY {
                return id;
            }
        }

        // resample: grammar first, then the chain (:666-681)
        let mut cur_p = TokenDataArray::from_logits(logits);
        grmr.apply_to(&mut cur_p);
        let id = sample_once(&mut self.chain, &mut cur_p);
        self.cur = cur_p.data[..cur_p.size].to_vec();
        id
    }

    /// `sample` variant that draws from a caller-provided RNG: the external
    /// RNG state is swapped into the chain's dist/mirostat sampler for the
    /// draw (and keeps the drawn state afterwards).
    pub fn sample_with_rng(&mut self, logits: &[f32], rng: &mut Mt19937) -> LlamaToken {
        self.chain.set_rng(rng);
        let token = self.sample(logits);
        self.chain.set_rng(rng); // swap back out
        token
    }

    /// `common_sampler_accept` (feed a prompt token, no sampling)
    pub fn accept(&mut self, token: LlamaToken) {
        self.chain.accept(token);
        self.prev.push_back(token);
    }

    /// `common_sampler_reset`
    pub fn reset(&mut self) {
        self.chain.reset();
        self.prev.clear();
    }

    /// `common_sampler_sample_and_accept_n` (common/sampling.cpp:678-715) — the
    /// speculative-verification rule, owned by the sampler module exactly like
    /// the C owns it (the driver reaches it through its `common_sampler`).
    /// Delegates to `crate::speculative::common_sampler_sample_and_accept_n`.
    pub fn sample_and_accept_n(
        &mut self,
        vocab: &crate::vocab::Vocab,
        out: &crate::context::BatchOutput,
        draft: &[LlamaToken],
    ) -> Vec<LlamaToken> {
        crate::speculative::common_sampler_sample_and_accept_n(self, vocab, out, draft)
    }

    /// `common_sampler_sample_and_accept_n_rejection`
    /// (common/sampling.cpp:720-836, upstream a7b94df2c) — as
    /// [`Self::sample_and_accept_n`], but verifies by rejection sampling:
    /// accept a drafted token with probability min(1, p/q), else draw from
    /// norm(max(0, p - q)). Preserves the target distribution exactly, and
    /// accepts more often than matching does when the draft samples instead
    /// of taking its argmax. `draft_q` holds the draft's candidates per token.
    ///
    /// `grmr` mirrors `gsmpl->grmr` (`grammar_should_apply` ⇔ `Some`);
    /// `grammar_first` is the C's flag of the same name. Note the rejection
    /// rule has no EOG check — it takes the drafted token whenever the coin
    /// passes, EOG included.
    pub fn sample_and_accept_n_rejection(
        &mut self,
        out: &crate::context::BatchOutput,
        draft: &[LlamaToken],
        draft_q: &[Vec<TokenData>],
        mut grmr: Option<&mut GrammarSampler>,
        grammar_first: bool,
    ) -> Vec<LlamaToken> {
        assert!(
            out.n_outputs == draft.len() + 1,
            "idxs.size() must be draft.size() + 1"
        );
        assert_eq!(
            draft_q.len(),
            draft.len(),
            "draft_q must have one entry per draft token"
        );

        let mut result: Vec<LlamaToken> = Vec::with_capacity(draft.len() + 1);

        // `prob_of` (common/sampling.cpp:720-727)
        fn prob_of(data: &[TokenData], id: LlamaToken) -> f32 {
            for d in data {
                if d.id == id {
                    return d.p;
                }
            }
            0.0
        }

        // the port's BatchOutput carries whole logits rows; the C reads rows
        // through `idxs` — row i here is `idxs[i]`
        let logits_row =
            |i: usize| -> &[f32] { out.logits_ith(i as i32).expect("verify logits row") };

        // `cand` — candidate array masked by the grammar, if there is one
        let mut cand: Vec<TokenData> = Vec::new();
        let mut residual: Vec<TokenData> = Vec::new();

        let mut i = 0usize;
        while i < draft.len() {
            // leaves the target distribution in the candidate array
            let id_tgt = self.sample_no_accept(logits_row(i), grmr.as_deref_mut(), grammar_first);

            let cur_p = self.get_candidates(true).to_vec();
            let q = &draft_q[i];

            let masked = !grammar_first && grmr.is_some();
            if masked {
                // `cand.assign(cur_p->data, ...)` + `llama_sampler_apply(grmr, &arr)`
                let mut arr = TokenDataArray {
                    data: cur_p.clone(),
                    size: cur_p.len(),
                    selected: -1,
                    sorted: false,
                };
                grmr.as_deref_mut().unwrap().apply_to(&mut arr);
                cand = arr.data[..arr.size].to_vec();
            }

            // a candidate the grammar rejects carries no probability, whatever
            // the target thinks
            let p_raw = |k: usize| -> f32 {
                if masked && cand[k].logit == f32::NEG_INFINITY {
                    0.0
                } else {
                    cur_p[k].p
                }
            };

            // masking drops probability mass, so rescale what is left or the
            // residual is over-weighted
            let mut p_sum = 0.0f32;
            if masked {
                for k in 0..cur_p.len() {
                    p_sum += p_raw(k);
                }
            }
            let p_norm = if masked && p_sum > 0.0 { 1.0 / p_sum } else { 1.0 };
            let p_of = |k: usize| -> f32 { p_raw(k) * p_norm };

            // q_x is never 0 for a token the draft produced, but guard the divide
            let q_x = prob_of(q, draft[i]);

            let mut p_x = 0.0f32;
            for (k, c) in cur_p.iter().enumerate() {
                if c.id == draft[i] {
                    p_x = p_of(k);
                    break;
                }
            }

            // draws come from the sampler's own stream (`uni(gsmpl->rng)`, a
            // `uniform_real_distribution<float>` = `Mt19937::canonical_float`)
            if q_x > 0.0 && (p_x >= q_x || self.rng.canonical_float() < p_x / q_x) {
                self.accept_grammar_or_plain(draft[i], grmr.as_deref_mut());
                result.push(draft[i]);
                i += 1;
                continue;
            }

            // rejected: tokens outside q's support keep all of p
            residual.clear();
            let mut sum = 0.0f32;
            for (k, c) in cur_p.iter().enumerate() {
                let r = p_of(k) - prob_of(q, c.id);
                if r > 0.0 {
                    residual.push(TokenData { id: c.id, logit: 0.0, p: r });
                    sum += r;
                }
            }

            let mut id = id_tgt;
            if sum > 0.0 {
                let mut u = self.rng.canonical_float() * sum;
                id = residual.last().unwrap().id;
                for e in &residual {
                    u -= e.p;
                    if u <= 0.0 {
                        id = e.id;
                        break;
                    }
                }
            }

            self.accept_grammar_or_plain(id, grmr.as_deref_mut());
            result.push(id);

            break;
        }

        if i == draft.len() {
            let id = self.sample_no_accept(logits_row(i), grmr.as_deref_mut(), grammar_first);
            self.accept_grammar_or_plain(id, grmr.as_deref_mut());
            result.push(id);
        }

        result
    }

    /// `common_sampler_accept(..., accept_grammar = true)` — the chain accept
    /// plus the grammar accept when a grammar is attached
    /// (common/sampling.cpp:475-489).
    fn accept_grammar_or_plain(&mut self, token: LlamaToken, grmr: Option<&mut GrammarSampler>) {
        if let Some(g) = grmr {
            // C throws std::runtime_error from `llama_sampler_accept`; the
            // port's try_accept propagates the same failure
            g.try_accept(token).expect("grammar accept failed");
        }
        self.accept(token);
    }
}

// ---------------------------------------------------------------------------
// grammar sampler (llama-sampler.cpp:2658-2853) — added by the GBNF task;
// nothing above this line was modified.
// ---------------------------------------------------------------------------

/// `struct llama_sampler_grammar` (llama-sampler.cpp:2660-2666).
///
/// Ports `llama_sampler_init_grammar` (:2825-2830) / `llama_sampler_init_grammar_impl`
/// (:2766-2823) and the sampler vtable (:2669-2760). The lazy variants
/// (`llama_sampler_init_grammar_lazy` :2832, `_lazy_patterns` :2843) need
/// `std::regex` and are not ported (documented gap) — `lazy` stays false, so
/// `awaiting_trigger` is never entered.
#[derive(Clone)]
pub struct GrammarSampler {
    /// `struct llama_grammar * grammar` (:2666)
    pub grammar: crate::grammar::Grammar,
    /// `const llama_vocab * vocab` — the grammar only needs
    /// `vocab->token_to_piece(id)` + `vocab->is_eog(id)` (llama-grammar.cpp:1378-1384),
    /// so the piece cache is snapshotted here (see `VocabPieces`)
    pub vocab: crate::grammar::VocabPieces,
    /// `std::string grammar_str` (:2663) — kept for `reset`
    pub grammar_str: String,
    /// `std::string grammar_root` (:2664) — kept for `reset`
    pub grammar_root: String,
}

impl GrammarSampler {
    /// `llama_grammar_init_impl(vocab, grammar_str, grammar_root, lazy=false, ...)`
    pub fn new(
        vocab: &crate::vocab::Vocab,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<GrammarSampler, String> {
        let grammar = crate::grammar::Grammar::parse(Some(vocab), grammar_str, grammar_root)?;
        Ok(GrammarSampler {
            grammar,
            vocab: crate::grammar::VocabPieces::from_vocab(vocab),
            grammar_str: grammar_str.to_string(),
            grammar_root: grammar_root.to_string(),
        })
    }

    /// Construction from an existing piece table (no `Vocab` needed). The thin
    /// `<token>` grammar form requires a real vocab, so it is rejected here.
    pub fn from_pieces(
        grammar_str: &str,
        grammar_root: &str,
        pieces: crate::grammar::VocabPieces,
    ) -> Result<GrammarSampler, String> {
        let grammar = crate::grammar::Grammar::parse(None, grammar_str, grammar_root)?;
        Ok(GrammarSampler {
            grammar,
            vocab: pieces,
            grammar_str: grammar_str.to_string(),
            grammar_root: grammar_root.to_string(),
        })
    }

    /// `llama_sampler_grammar_apply` (:2680-2685) → `llama_grammar_apply_impl`
    pub fn apply_to(&self, cur_p: &mut TokenDataArray) {
        self.grammar.apply(&self.vocab, cur_p);
    }

    /// `llama_sampler_grammar_accept_impl` (:2673-2678) → `llama_grammar_accept_impl`
    /// (only the non-lazy path; C throws on an empty stack after accepting)
    pub fn try_accept(&mut self, token: LlamaToken) -> Result<(), String> {
        self.grammar.accept_impl(&self.vocab, token)
    }

    /// `vocab->token_to_piece(token)` for the snaphotted table
    pub fn piece(&self, token: LlamaToken) -> &[u8] {
        use crate::grammar::GrammarVocab as _;
        self.vocab.token_piece(token)
    }
}

impl Sampler for GrammarSampler {
    /// `llama_sampler_grammar_name` (:2669-2671)
    fn name(&self) -> &'static str {
        "grammar"
    }

    fn apply(&mut self, cur_p: &mut TokenDataArray) {
        self.apply_to(cur_p);
    }

    fn accept(&mut self, token: LlamaToken) {
        // C throws std::runtime_error (llama-grammar.cpp:1522) — propagate as a
        // panic for the vtable path; `try_accept` is the non-panicking form.
        if let Err(e) = self.try_accept(token) {
            panic!("grammar sampler: {e}");
        }
    }

    fn reset(&mut self) {
        // `llama_sampler_grammar_reset` (:2700-2718) re-runs
        // llama_grammar_init_impl with the stored string/root; since the parsed
        // rules are immutable, rebuilding the initial stacks is equivalent.
        self.grammar.reset();
    }
}

/// `llama_sampler_init_grammar` (:2825-2830) with the default root `"root"`.
pub fn init_grammar(
    vocab: &crate::vocab::Vocab,
    grammar_str: &str,
) -> Result<GrammarSampler, String> {
    init_grammar_root(vocab, grammar_str, "root")
}

/// `llama_sampler_init_grammar_impl` (:2766-2823) with an explicit root.
/// C returns a no-op sampler when `grammar_str` is empty; here that is an
/// explicit error (the caller decides whether a grammar exists).
pub fn init_grammar_root(
    vocab: &crate::vocab::Vocab,
    grammar_str: &str,
    grammar_root: &str,
) -> Result<GrammarSampler, String> {
    if grammar_str.is_empty() {
        return Err("failed to parse grammar: empty grammar string".to_string());
    }
    GrammarSampler::new(vocab, grammar_str, grammar_root)
}

impl SamplingContext {
    /// `common_sampler_sample(gsmpl, ctx, idx, /* grammar_first = */ false)`
    /// (common/sampling.cpp:594-676) with the grammar sampler kept *outside*
    /// the chain, exactly like `common_sampler`.
    ///
    /// Flow (grammar_first = false):
    ///  1. apply the chain and take the sampled token;
    ///  2. re-apply the grammar to a single-token array to test validity
    ///     (:646-658) — valid → done, so the RNG is consumed once;
    ///  3. invalid → re-read the logits, apply the grammar *then* the chain and
    ///     sample again (:660-674), i.e. a second RNG draw.
    ///
    /// Returns the token and accepts it into the chain + grammar + `prev`
    /// (`common_sampler_accept(..., is_generated = true)`, :471-499).
    pub fn sample_with_grammar(
        &mut self,
        logits: &[f32],
        grammar: &mut GrammarSampler,
    ) -> Result<LlamaToken, String> {
        let mut cur_p = TokenDataArray::from_logits(logits);
        self.chain.apply(&mut cur_p);
        assert!(
            cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
            "sample_with_grammar: chain did not select a token"
        );
        let id = cur_p.data[cur_p.selected as usize].id;
        self.cur = cur_p.data[..cur_p.size].to_vec();

        // check if the sampled token fits the grammar (single-candidate apply)
        let mut single = TokenDataArray {
            data: vec![TokenData {
                id,
                logit: 1.0,
                p: 0.0,
            }],
            size: 1,
            selected: -1,
            sorted: false,
        };
        grammar.apply_to(&mut single);
        let valid = single.data[0].logit != f32::NEG_INFINITY;

        let token = if valid {
            id
        } else {
            // resample: grammar first, then the chain
            let mut cur_p = TokenDataArray::from_logits(logits);
            grammar.apply_to(&mut cur_p);
            self.chain.apply(&mut cur_p);
            assert!(
                cur_p.selected >= 0 && (cur_p.selected as usize) < cur_p.size,
                "sample_with_grammar: no selected token during resampling"
            );
            let t = cur_p.data[cur_p.selected as usize].id;
            self.cur = cur_p.data[..cur_p.size].to_vec();
            t
        };

        self.chain.accept(token);
        grammar.try_accept(token)?;
        self.prev.push_back(token);

        Ok(token)
    }
}

// ---------------------------------------------------------------------------
// tests — fixtures verified against the C++ reference program
// (/tmp/smplref/sampler_ref.cpp, g++ 13.3, pinned llama.cpp bd4f514db1)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hexf32(v: f32) -> u32 {
        v.to_bits()
    }

    fn cur_from(logits: &[f32]) -> TokenDataArray {
        TokenDataArray::from_logits(logits)
    }

    fn ids(cur: &TokenDataArray) -> Vec<i32> {
        cur.data[..cur.size].iter().map(|d| d.id).collect()
    }

    fn logits_bits(cur: &TokenDataArray) -> Vec<u32> {
        cur.data[..cur.size]
            .iter()
            .map(|d| hexf32(d.logit))
            .collect()
    }

    fn p_bits(cur: &TokenDataArray) -> Vec<u32> {
        cur.data[..cur.size].iter().map(|d| hexf32(d.p)).collect()
    }

    // ---- Mt19937 / distributions ------------------------------------------

    /// Known mt19937 vector: the canonical mt19937ar output for the default
    /// seed 5489 (first value 3499211612) — proves the engine core.
    #[test]
    fn mt19937_default_seed_5489_vector() {
        let mut rng = Mt19937::new(5489);
        let expected = [3499211612u32, 581869302, 3890346734, 3586334585, 545404204];
        for &e in &expected {
            assert_eq!(rng.next_u32(), e);
        }
    }

    /// libstdc++ (g++ 13.3) std::mt19937(42) — first 5 raw outputs.
    #[test]
    fn mt19937_seed42_vector() {
        let mut rng = Mt19937::new(42);
        let expected = [1608637542u32, 3421126067, 4083286876, 787846414, 3143890026];
        for &e in &expected {
            assert_eq!(rng.next_u32(), e);
        }
    }

    /// uniform_real_distribution<double>(0,1) bit patterns, seed 42 (C++ ref:
    /// URD64 3fe97d47b66bfc3c 3fc77aca8779b102 3fe8f33a88f76c7f).
    #[test]
    fn uniform_real_double_bits() {
        let mut rng = Mt19937::new(42);
        assert_eq!(rng.canonical_double().to_bits(), 0x3fe97d47b66bfc3c);
        assert_eq!(rng.canonical_double().to_bits(), 0x3fc77aca8779b102);
        assert_eq!(rng.canonical_double().to_bits(), 0x3fe8f33a88f76c7f);
    }

    /// uniform_real_distribution<float>(0,1) bit patterns, seed 42 (C++ ref:
    /// URD32 3ebfc3b9 3f4bea3e 3f736203) and seeds 1..=8 (XTC_CHANCE).
    #[test]
    fn uniform_real_float_bits() {
        let mut rng = Mt19937::new(42);
        assert_eq!(rng.canonical_float().to_bits(), 0x3ebfc3b9);
        assert_eq!(rng.canonical_float().to_bits(), 0x3f4bea3e);
        assert_eq!(rng.canonical_float().to_bits(), 0x3f736203);

        let expected = [
            0x3ed583e8u32,
            0x3edf3ab9,
            0x3f0d0117,
            0x3f778f44,
            0x3e63522e,
            0x3f64927c,
            0x3d9c4785,
            0x3f5f9912,
        ];
        for (i, &e) in expected.iter().enumerate() {
            let mut r = Mt19937::new(i as u32 + 1);
            assert_eq!(r.canonical_float().to_bits(), e);
        }
    }

    /// libstdc++ discrete_distribution replica: single-probability table
    /// (size < 2 -> cp empty -> always index 0); used by mirostat fixtures below.
    #[test]
    fn discrete_distribution_small_table() {
        let mut rng = Mt19937::new(1);
        assert_eq!(DiscreteDistribution::new(&[0.5]).sample(&mut rng), 0);
        assert_eq!(DiscreteDistribution::new(&[]).sample(&mut rng), 0);
    }

    // ---- greedy -------------------------------------------------------------

    /// Task spec: ties take the LAST maximum (ggml argmax semantics).
    /// (Pinned CPU llama_sampler_greedy_apply with `>` would pick token 1 —
    /// documented divergence, see module docs.)
    #[test]
    fn greedy_tie_takes_last() {
        let mut s = init_greedy();
        let mut cur = cur_from(&[1.0, 3.0, 3.0, 2.0]);
        s.apply(&mut cur);
        assert_eq!(cur.selected, 2);
        assert_eq!(cur.selected_token(), Some(2));
    }

    #[test]
    fn greedy_basic() {
        let mut s = init_greedy();
        let mut cur = cur_from(&[-1.0, 5.0, 3.0]);
        s.apply(&mut cur);
        assert_eq!(cur.selected_token(), Some(1));
    }

    // ---- dist ---------------------------------------------------------------

    /// Bit-exact dist fixture: logits [1,2,3], seed 42 (C++ ref DIST_123).
    #[test]
    fn dist_fixture_123_seed42() {
        let mut s = init_dist(42);
        let mut cur = cur_from(&[1.0, 2.0, 3.0]);
        s.apply(&mut cur);
        assert_eq!(cur.selected, 2);
        assert_eq!(cur.selected_token(), Some(2));
        // softmax probs normalized in f64 then narrowed (C++ ref DIST_123_P)
        assert_eq!(p_bits(&cur), vec![0x3db861f3, 0x3e7a9a1a, 0x3f2a4d3b]);
    }

    /// Six sequential draws over a 6-token vocab with one shared rng
    /// (C++ ref DIST_SEQ6 5 1 4 3 2 0).
    #[test]
    fn dist_sequential_draws() {
        let mut s = init_dist(42);
        let logits = [0.1f32, -0.2, 0.3, 0.25, -0.05, 0.4];
        let got: Vec<i32> = (0..6)
            .map(|_| {
                let mut cur = cur_from(&logits);
                s.apply(&mut cur);
                cur.selected_token().unwrap()
            })
            .collect();
        assert_eq!(got, vec![5, 1, 4, 3, 2, 0]);
    }

    /// size == 1: one rng draw consumed, p forced to 1.0 (C++ ref DIST_SIZE1).
    /// Also verifies the draw actually advanced the RNG stream (2 words).
    #[test]
    fn dist_single_candidate() {
        let mut s = DistSampler {
            seed: 99,
            seed_cur: 99,
            rng: Mt19937::new(99),
        };
        let mut cur = cur_from(&[7.0]);
        s.apply(&mut cur);
        assert_eq!(cur.selected, 0);
        assert_eq!(hexf32(cur.data[0].p), 0x3f800000); // 1.0f
                                                       // the size==1 branch consumed exactly one canonical-double (= 2 draws)
        let mut check = Mt19937::new(99);
        check.canonical_double();
        assert_eq!(s.rng.next_u32(), check.next_u32());
    }

    // ---- temp / temp-ext -----------------------------------------------------

    /// Task spec: logits [0, ln2] with temp=1 -> softmax = [1/3, 2/3].
    /// (Verified bits from the C++ ref TEMP_LN2: 3eaaaaab / 3f2aaaab.)
    #[test]
    fn temp_softmax_ln2() {
        let mut cur = cur_from(&[0.0, 2.0f32.ln()]);
        temp_impl(&mut cur, 1.0);
        let mut d = init_dist(123);
        d.apply(&mut cur);
        assert_eq!(hexf32(cur.data[0].p), 0x3eaaaaab); // 0.33333334
        assert_eq!(hexf32(cur.data[1].p), 0x3f2aaaab); // 0.6666667
        assert_eq!(cur.selected, 1);
        // softmax([0, ln2]) = [1/3, 2/3]; probabilities sum to 1
        let sum: f32 = cur.data[..2].iter().map(|d| d.p).sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    /// temp = 0.5 divides logits (C++ ref TEMP_HALF: logit1 = 2*ln2 bit-exact).
    #[test]
    fn temp_half_division() {
        let mut cur = cur_from(&[0.0, 2.0f32.ln()]);
        temp_impl(&mut cur, 0.5);
        assert_eq!(logits_bits(&cur), vec![0x00000000, 0x3fb17218]);
    }

    /// temp <= 0 keeps only the max logit (FIRST max on ties — verbatim C;
    /// C++ ref TEMP_ZERO: only index 1 survives).
    #[test]
    fn temp_zero_argmax() {
        let mut cur = cur_from(&[1.0, 3.0, 3.0, 2.0]);
        temp_impl(&mut cur, 0.0);
        assert_eq!(
            logits_bits(&cur),
            vec![0xff800000, 0x40400000, 0xff800000, 0xff800000]
        );
    }

    /// init_temp(1.0) -> empty sampler "?temp"; init_temp_ext(1,0,1) -> "?temp-ext".
    #[test]
    fn temp_init_disabled() {
        assert_eq!(init_temp(1.0).name(), "?temp");
        assert_eq!(init_temp_ext(1.0, 0.0, 1.0).name(), "?temp-ext");
        assert_eq!(init_temp(0.8).name(), "temp");
        assert_eq!(init_temp_ext(0.8, 0.0, 1.0).name(), "temp-ext");
    }

    // ---- top-k ----------------------------------------------------------------

    /// C++ ref TOPK3: descending order after partial sort, size truncated to k.
    #[test]
    fn top_k_basic() {
        let mut cur = cur_from(&[0.1, 0.4, 0.2, 0.3, 0.05, 0.25]);
        top_k_impl(&mut cur, 3);
        assert_eq!(cur.size, 3);
        assert_eq!(ids(&cur), vec![1, 3, 5]);
        assert_eq!(logits_bits(&cur), vec![0x3ecccccd, 0x3e99999a, 0x3e800000]);
    }

    /// Tie-order replication of libstdc++ std::partial_sort (heap path, k <= 128):
    /// C++ ref TOPK_TIES ids [8,10,2,14,3,7,1,9,11,0].
    #[test]
    fn top_k_tie_order_partial_sort() {
        let logits: Vec<f32> = vec![
            2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.5, 2.0, 4.0, 2.0, 3.5, 2.0, 1.5, 2.0, 2.5, 2.0, 0.75,
            2.0,
        ];
        let mut cur = cur_from(&logits);
        top_k_impl(&mut cur, 10);
        assert_eq!(ids(&cur), vec![8, 10, 2, 14, 3, 7, 1, 9, 11, 0]);
    }

    /// Tie-order replication for a full-range partial sort (softmax do_sort path):
    /// C++ ref SOFTMAX_SORT_TIES ids [8,10,2,14,1,7,17,3,9,11,0,5,15,13,12,4,16,6].
    #[test]
    fn softmax_sort_tie_order() {
        let logits: Vec<f32> = vec![
            2.0, 2.0, 3.0, 2.0, 1.0, 2.0, 0.5, 2.0, 4.0, 2.0, 3.5, 2.0, 1.5, 2.0, 2.5, 2.0, 0.75,
            2.0,
        ];
        let mut cur = cur_from(&logits);
        softmax_impl(&mut cur, true);
        assert_eq!(
            ids(&cur),
            vec![8, 10, 2, 14, 1, 7, 17, 3, 9, 11, 0, 5, 15, 13, 12, 4, 16, 6]
        );
        assert_eq!(
            p_bits(&cur),
            vec![
                0x3e88785c, 0x3e258bf3, 0x3dc8d17a, 0x3d739ad9, 0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa,
                0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa, 0x3d13c0fa,
                0x3cb33c05, 0x3c596c20, 0x3c295426, 0x3c03df93,
            ]
        );
    }

    /// Bucket-sort path (n > 128): C++ ref TOPK200 (200 logits from mt19937(7),
    /// k=40) — checks ids + logit bits of the first 12 survivors.
    #[test]
    fn top_k_bucket_sort_large_n() {
        let mut lg = Mt19937::new(7);
        let logits: Vec<f32> = (0..200)
            .map(|_| ((lg.next_u32() % 1000) as i32 - 500) as f32 / 100.0)
            .collect();
        let mut cur = cur_from(&logits);
        top_k_impl(&mut cur, 40);
        assert_eq!(cur.size, 40);
        assert_eq!(
            ids(&cur)[..12].to_vec(),
            vec![124, 159, 183, 190, 8, 18, 40, 174, 155, 144, 85, 81]
        );
        assert_eq!(
            logits_bits(&cur)[..12].to_vec(),
            vec![
                0x409fae14, 0x409eb852, 0x409c7ae1, 0x409c28f6, 0x409c28f6, 0x409a8f5c, 0x409947ae,
                0x409947ae, 0x4098a3d7, 0x40980000, 0x4096147b, 0x4093851f,
            ]
        );
    }

    /// top-k with k >= size keeps everything sorted; k <= 0 via init is empty.
    #[test]
    fn top_k_init_disabled() {
        assert_eq!(init_top_k(0).name(), "?top-k");
        let mut s = init_top_k(3);
        let mut cur = cur_from(&[1.0, 2.0, 3.0]);
        s.apply(&mut cur);
        assert_eq!(cur.size, 3);
        assert_eq!(ids(&cur), vec![2, 1, 0]);
    }

    // ---- top-p -----------------------------------------------------------------

    /// C++ ref TOPP06: probs [0.3,0.25,0.2,0.15,0.1], p=0.6 -> first 3 tokens
    /// (cum_sum 0.75 >= 0.6 at i=2), min_keep=1.
    #[test]
    fn top_p_basic() {
        let mut s = init_top_p(0.6, 1);
        let probs = [0.30f32, 0.25, 0.20, 0.15, 0.10];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 3);
        assert!(cur.sorted);
        assert_eq!(ids(&cur), vec![0, 1, 2]);
        assert_eq!(p_bits(&cur), vec![0x3e99999a, 0x3e800000, 0x3e4ccccc]);
        // kept mass: 0.3 + 0.25 + 0.2 = 0.75 (>= p = 0.6, one token past the boundary)
        let sum: f32 = cur.data[..cur.size].iter().map(|d| d.p).sum();
        assert!((sum - 0.75).abs() < 1e-6, "sum = {sum}");
    }

    /// min_keep extends the kept set: C++ ref TOPP06B (2 kept) vs TOPP06B_MK3 (3 kept).
    #[test]
    fn top_p_min_keep() {
        let probs = [0.5f32, 0.3, 0.1, 0.1];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();

        let mut s = init_top_p(0.6, 1);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 2);
        assert_eq!(p_bits(&cur), vec![0x3f000000, 0x3e999999]);

        let mut s2 = init_top_p(0.6, 3);
        let mut cur2 = cur_from(&logits);
        s2.apply(&mut cur2);
        assert_eq!(cur2.size, 3);
        assert_eq!(ids(&cur2), vec![0, 1, 3]); // tie between id 2 and 3 -> 3 first
        assert_eq!(p_bits(&cur2), vec![0x3f000000, 0x3e999999, 0x3dcccccc]);
    }

    /// Adaptive bucket path for >1024 unsorted candidates: C++ ref TOPP1100
    /// (1100 logits from mt19937(9), p=0.9, min_keep=1) -> 251 kept.
    #[test]
    fn top_p_adaptive_large_n() {
        let mut lg = Mt19937::new(9);
        let logits: Vec<f32> = (0..1100)
            .map(|_| ((lg.next_u32() % 1000) as i32 - 500) as f32 / 100.0)
            .collect();
        let mut s = init_top_p(0.9, 1);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 251);
        assert_eq!(
            ids(&cur)[..10].to_vec(),
            vec![158, 711, 814, 818, 677, 338, 231, 365, 115, 800]
        );
        assert_eq!(
            p_bits(&cur)[..10].to_vec(),
            vec![
                0x3c13da25, 0x3c126186, 0x3c10ecaa, 0x3c10ecaa, 0x3c0e0e06, 0x3c09db3d, 0x3c07206d,
                0x3c05c83b, 0x3c05c83b, 0x3c047373,
            ]
        );
    }

    #[test]
    fn top_p_init_disabled() {
        assert_eq!(init_top_p(1.0, 1).name(), "?top-p");
    }

    // ---- min-p -------------------------------------------------------------------

    /// Unsorted filter path: p=0.1 on [0.5,0.2,0.1,0.05,0.02] keeps 4 tokens
    /// (threshold = 0.5*0.1 = 0.05; 0.02 dropped). C++ ref MINP01B.
    #[test]
    fn min_p_filter() {
        let probs = [0.5f32, 0.2, 0.1, 0.05, 0.02];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut s = init_min_p(0.1, 1);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 4);
        assert_eq!(ids(&cur), vec![0, 1, 2, 3]);
        // logit bits (C++ ref MINP01B): order preserved, logits unchanged
        assert_eq!(
            logits_bits(&cur),
            vec![0xbf317218, 0xbfce0210, 0xc0135d8e, 0xc03fba14]
        );
        assert!(!cur.sorted);
    }

    /// min_keep forces the sorted fallback path when the filter keeps too few:
    /// C++ ref MINP06_MK3_FALLBACK -> sorted, 3 kept.
    #[test]
    fn min_p_min_keep_fallback() {
        let probs = [0.5f32, 0.2, 0.1, 0.05, 0.02];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut s = init_min_p(0.6, 3);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 3);
        assert_eq!(ids(&cur), vec![0, 1, 2]);
        assert!(cur.sorted);
        assert_eq!(hexf32(cur.data[0].logit), 0xbf317218);
    }

    #[test]
    fn min_p_init_disabled() {
        assert_eq!(init_min_p(0.0, 1).name(), "?min-p");
    }

    // ---- typical -------------------------------------------------------------------

    /// C++ ref TYPICAL07: p=0.7, min_keep=1 on logits [0.1,0.4,0.2,0.3,0.25].
    #[test]
    fn typical_fixture() {
        let mut s = init_typical(0.7, 1);
        let mut cur = cur_from(&[0.1, 0.4, 0.2, 0.3, 0.25]);
        s.apply(&mut cur);
        assert_eq!(cur.size, 4);
        assert_eq!(ids(&cur), vec![4, 3, 2, 1]);
        assert_eq!(
            p_bits(&cur),
            vec![0x3e4bc785, 0x3e563a34, 0x3e41d748, 0x3e6cc202]
        );
    }

    #[test]
    fn typical_init_disabled() {
        assert_eq!(init_typical(1.0, 1).name(), "?typical");
    }

    // ---- xtc -------------------------------------------------------------------------

    /// Removal path: probability=1 always rolls in; tokens with p >= threshold
    /// (0.2) are removed from the front. C++ ref XTC_CUT.
    #[test]
    fn xtc_cut() {
        let probs = [0.45f32, 0.25, 0.15, 0.15];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut s = init_xtc(1.0, 0.2, 2, 42);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 3);
        assert_eq!(ids(&cur), vec![1, 3, 2]);
        assert_eq!(p_bits(&cur), vec![0x3e800001, 0x3e19999a, 0x3e19999a]);
    }

    /// No-roll path: seed 42 chance = 0.3745 > probability 0.3 -> early return,
    /// probabilities untouched (still 0), unsorted. C++ ref XTC_NOROLL.
    #[test]
    fn xtc_no_roll() {
        let probs = [0.45f32, 0.25, 0.15, 0.15];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut s = init_xtc(0.3, 0.2, 2, 42);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 4);
        assert!(!cur.sorted);
        assert_eq!(p_bits(&cur), vec![0, 0, 0, 0]);
    }

    /// min_keep blocks the cut. C++ ref XTC_MINKEEP.
    #[test]
    fn xtc_min_keep_blocks() {
        let probs = [0.45f32, 0.25, 0.15, 0.15];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut s = init_xtc(1.0, 0.2, 5, 42);
        let mut cur = cur_from(&logits);
        s.apply(&mut cur);
        assert_eq!(cur.size, 4);
        assert!(cur.sorted);
        assert_eq!(ids(&cur), vec![0, 1, 3, 2]);
        assert_eq!(
            p_bits(&cur),
            vec![0x3ee66667, 0x3e800001, 0x3e19999a, 0x3e19999a]
        );
    }

    #[test]
    fn xtc_init_disabled() {
        assert_eq!(init_xtc(0.0, 0.2, 1, 42).name(), "?xtc");
        assert_eq!(init_xtc(1.0, 0.6, 1, 42).name(), "?xtc");
    }

    // ---- top-n-sigma ---------------------------------------------------------------------

    /// C++ ref TOPNSIGMA1: n=1.0 on [5,4,3,2,1,0.5,0.25,0.1] -> mask below max - std.
    #[test]
    fn top_n_sigma_fixture() {
        let mut s = init_top_n_sigma(1.0);
        let mut cur = cur_from(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1]);
        s.apply(&mut cur);
        assert_eq!(cur.size, 8);
        assert_eq!(
            logits_bits(&cur),
            vec![
                0x40a00000, 0x40800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000, 0xff800000,
                0xff800000,
            ]
        );
    }

    #[test]
    fn top_n_sigma_init_disabled() {
        assert_eq!(init_top_n_sigma(-1.0).name(), "?top-n-sigma");
    }

    // ---- penalties --------------------------------------------------------------------------

    /// Penalties formula, hand-checked:
    /// after accepting {2, 2, 0} (last_n=8, repeat=1.2, freq=0.3, present=0.25):
    ///   token 2 (count 2, logit 3 > 0): 3/1.2 - 2*0.3 - 0.25 = 1.65
    ///   token 0 (count 1, logit 1 > 0): 1/1.2 - 0.3 - 0.25 = 0.28333
    ///   token 1 (count 0): unchanged (-2.0); token 3: unchanged (0.5)
    /// C++ ref PEN_AFTER3 logits bits.
    #[test]
    fn penalties_after_3_accepts() {
        let mut s = init_penalties(4, 8, 1.2, 0.3, 0.25);
        for t in [2, 2, 0] {
            s.accept(t);
        }
        let mut cur = cur_from(&[1.0, -2.0, 3.0, 0.5]);
        s.apply(&mut cur);
        assert_eq!(
            logits_bits(&cur),
            vec![0x3e911110, 0xc0000000, 0x3fd33333, 0x3f000000]
        );
        assert!(!cur.sorted);
    }

    /// Ring overflow: 8 accepts with last_n=8 -> oldest duplicated token 2 leaves
    /// the window (counts: 2->0, 0->1, 1->5). C++ ref PEN_AFTER8.
    #[test]
    fn penalties_ring_overflow() {
        let mut s = init_penalties(4, 8, 1.2, 0.3, 0.25);
        for t in [2, 2, 0, 1, 1, 1, 1, 1] {
            s.accept(t);
        }
        // token 1: logit -2 <= 0 -> *1.2 = -2.4; count 5: -2.4 - 5*0.3 - 0.25 = -4.15
        let mut cur = cur_from(&[1.0, -2.0, 3.0, 0.5]);
        s.apply(&mut cur);
        assert_eq!(
            logits_bits(&cur),
            vec![0x3e911110, 0xc084cccd, 0x3fd33333, 0x3f000000]
        );
    }

    #[test]
    fn penalties_init_disabled() {
        assert_eq!(init_penalties(4, 0, 1.2, 0.3, 0.25).name(), "?penalties");
        assert_eq!(init_penalties(4, 64, 1.0, 0.0, 0.0).name(), "?penalties");
        // negative last_n clamps to 0 -> disabled
        assert_eq!(init_penalties(4, -5, 1.2, 0.3, 0.25).name(), "?penalties");
    }

    // ---- logit-bias -----------------------------------------------------------------------------

    /// C++ ref LOGBIAS: bias id1 +0.5 (direct index), id2 -inf (direct),
    /// id3 +10 (also direct here).
    #[test]
    fn logit_bias_direct() {
        let bias = vec![
            LogitBias {
                token: 1,
                bias: 0.5,
            },
            LogitBias {
                token: 2,
                bias: f32::NEG_INFINITY,
            },
            LogitBias {
                token: 3,
                bias: 10.0,
            },
        ];
        let mut s = init_logit_bias(4, &bias);
        let mut cur = cur_from(&[1.0, 2.0, 3.0, 4.0]);
        s.apply(&mut cur);
        assert_eq!(
            logits_bits(&cur),
            vec![0x3f800000, 0x40200000, 0xff800000, 0x41600000]
        );
    }

    /// Shuffled-candidates path: after top-k reorder, ids no longer match indices,
    /// so the linear-search branch applies the bias.
    #[test]
    fn logit_bias_after_reorder() {
        let bias = vec![LogitBias {
            token: 0,
            bias: -1.0,
        }];
        let mut s = init_logit_bias(3, &bias);
        // candidates already reordered: id 2 at index 0 -> token 0 must be found by search
        let mut cur = TokenDataArray {
            data: vec![
                TokenData {
                    id: 2,
                    logit: 0.3,
                    p: 0.0,
                },
                TokenData {
                    id: 0,
                    logit: 0.1,
                    p: 0.0,
                },
                TokenData {
                    id: 1,
                    logit: 0.2,
                    p: 0.0,
                },
            ],
            size: 3,
            selected: -1,
            sorted: true,
        };
        s.apply(&mut cur);
        assert_eq!(hexf32(cur.data[1].logit), hexf32(0.1 - 1.0));
        assert_eq!(hexf32(cur.data[0].logit), hexf32(0.3));
    }

    #[test]
    fn logit_bias_init_disabled() {
        assert_eq!(init_logit_bias(4, &[]).name(), "?logit-bias");
    }

    // ---- mirostat -----------------------------------------------------------------------------

    /// C++ ref MIRO1: n_vocab=6, tau=5, eta=0.1, m=100, seed 42 ->
    /// selects id 1, mu bit pattern 41280000.
    #[test]
    fn mirostat_v1_fixture() {
        let mut s = MirostatSampler {
            n_vocab: 6,
            seed: 42,
            seed_cur: 42,
            tau: 5.0,
            eta: 0.1,
            m: 100,
            mu: 2.0 * 5.0,
            rng: Mt19937::new(42),
        };
        let mut cur = cur_from(&[0.1, 0.4, 0.2, 0.3, 0.25, 0.05]);
        s.apply(&mut cur);
        assert_eq!(cur.selected, 0);
        assert_eq!(cur.selected_token(), Some(1));
        assert_eq!(hexf32(s.mu), 0x41280000);
        assert_eq!(cur.size, 1);
        assert_eq!(hexf32(cur.data[0].p), 0x3f800000);
    }

    /// C++ ref MIRO2: tau=5, eta=0.1, seed 42 -> selects idx 4 (id 0),
    /// mu bit pattern 41239427.
    #[test]
    fn mirostat_v2_fixture() {
        let mut s = MirostatV2Sampler {
            seed: 42,
            seed_cur: 42,
            tau: 5.0,
            eta: 0.1,
            mu: 2.0 * 5.0,
            rng: Mt19937::new(42),
        };
        let mut cur = cur_from(&[0.1, 0.4, 0.2, 0.3, 0.25, 0.05]);
        s.apply(&mut cur);
        assert_eq!(cur.selected, 4);
        assert_eq!(cur.selected_token(), Some(0));
        assert_eq!(hexf32(s.mu), 0x41239427);
        assert_eq!(ids(&cur), vec![1, 3, 4, 2, 0, 5]);
        assert_eq!(
            p_bits(&cur),
            vec![0x3e4b965c, 0x3e3836a4, 0x3e2f3ab0, 0x3e26aee8, 0x3e16d23b, 0x3e0f7730,]
        );
    }

    #[test]
    fn mirostat_reset_restores_mu_and_seed() {
        let mut s = MirostatV2Sampler {
            seed: 42,
            seed_cur: 42,
            tau: 5.0,
            eta: 0.1,
            mu: 2.0 * 5.0,
            rng: Mt19937::new(42),
        };
        let mut cur = cur_from(&[0.1, 0.4, 0.2]);
        s.apply(&mut cur);
        assert_ne!(hexf32(s.mu), hexf32(10.0));
        s.reset();
        assert_eq!(hexf32(s.mu), hexf32(10.0)); // 2 * tau
        assert_eq!(s.get_seed(), 42);
    }

    // ---- chain / SamplingContext ------------------------------------------------------------------

    /// C++ ref CHAIN: penalties(10,4,1.1,0.25,0.1) -> top_k(5) -> top_p(0.9,1)
    /// -> min_p(0.05,1) -> temp_ext(0.8,0,1) -> dist(1234), 6 steps over a
    /// fixed logits vector, tokens [3, 8, 6, 9, 7, 5].
    #[test]
    fn chain_reference_6_steps() {
        let logits_arr = [0.1f32, -0.2, 0.3, 0.25, -0.05, 0.4, 0.2, -0.1, 0.15, 0.05];

        let mut chain = SamplerChain::new();
        chain.add(init_penalties(10, 4, 1.1, 0.25, 0.1));
        let mut topk = init_top_k(5);
        let _ = &mut topk;
        chain.add(init_top_k(5));
        chain.add(init_top_p(0.9, 1));
        chain.add(init_min_p(0.05, 1));
        chain.add(init_temp_ext(0.8, 0.0, 1.0));
        chain.add(init_dist(1234));

        let mut tokens = Vec::new();
        for _step in 0..6 {
            tokens.push(chain.sample(&logits_arr));
        }
        assert_eq!(tokens, vec![3, 8, 6, 9, 7, 5]);
    }

    /// Step-0 candidate snapshot from the C++ ref CHAIN_P step=0 (rebuilt by
    /// running the identical chain once, without the extra apply of the test above).
    #[test]
    fn chain_reference_step0_probs() {
        let logits_arr = [0.1f32, -0.2, 0.3, 0.25, -0.05, 0.4, 0.2, -0.1, 0.15, 0.05];
        let mut chain = SamplerChain::new();
        chain.add(init_penalties(10, 4, 1.1, 0.25, 0.1));
        chain.add(init_top_k(5));
        chain.add(init_top_p(0.9, 1));
        chain.add(init_min_p(0.05, 1));
        chain.add(init_temp_ext(0.8, 0.0, 1.0));
        chain.add(init_dist(1234));

        let mut cur = cur_from(&logits_arr);
        chain.apply(&mut cur);
        assert_eq!(cur.selected, 2);
        assert_eq!(ids(&cur), vec![5, 2, 3, 6, 8]);
        assert_eq!(
            p_bits(&cur),
            vec![0x3e728ada, 0x3e560afc, 0x3e491320, 0x3e3ce467, 0x3e3172a3]
        );
        let tok = cur.selected_token().unwrap();
        assert_eq!(tok, 3);
    }

    /// SamplingContext with common.h defaults: deterministic for a fixed seed.
    #[test]
    fn sampling_context_defaults_deterministic() {
        let params = SamplingParams {
            seed: 42,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(10, params);

        // defaults from common/common.h
        assert_eq!(SamplingParams::default().temp, 0.80);
        assert_eq!(SamplingParams::default().top_k, 40);
        assert_eq!(SamplingParams::default().top_p, 0.95);
        assert_eq!(SamplingParams::default().min_p, 0.05);

        // chain: penalties(defaults disable it -> "?penalties"), dry(disabled
        // -> "?dry", common.h chain position 2), top-n-sigma, top-k, typical,
        // top-p, min-p, xtc, temp-ext, dist — the reference default chain
        // (common.h:265-275) exactly
        assert_eq!(ctx.chain.n(), 10);
        let names: Vec<&str> = (0..ctx.chain.n())
            .map(|i| ctx.chain.get(i).unwrap().name())
            .collect();
        assert_eq!(
            names,
            vec![
                "?penalties",
                "?dry",
                "?top-n-sigma",
                "top-k",
                "?typical",
                "top-p",
                "min-p",
                "?xtc",
                "temp-ext",
                "dist"
            ]
        );
        assert_eq!(ctx.chain.get_seed(), 42);

        let logits: Vec<f32> = (0..10).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let t0 = ctx.sample(&logits);
        let t1 = ctx.sample(&logits);
        assert_eq!(ctx.prev.len(), 2);

        // replay with a fresh context -> identical stream
        let params = SamplingParams {
            seed: 42,
            ..Default::default()
        };
        let mut ctx2 = SamplingContext::new(10, params);
        assert_eq!(ctx2.sample(&logits), t0);
        assert_eq!(ctx2.sample(&logits), t1);
    }

    /// sample_with_rng injects an external RNG into the dist sampler.
    #[test]
    fn sampling_context_sample_with_external_rng() {
        let params = SamplingParams {
            seed: 7,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            penalty_last_n: 0,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(4, params);
        let logits = [0.25f32, 0.6, 0.1, 0.05];

        let mut ext = Mt19937::new(42);
        let t = ctx.sample_with_rng(&logits, &mut ext);
        // a plain dist sampler seeded with the same state picks the same token
        let mut d = init_dist(1);
        let mut rng = Mt19937::new(42);
        d.set_rng(&mut rng);
        let mut cur = cur_from(&logits);
        d.apply(&mut cur);
        assert_eq!(t, cur.selected_token().unwrap());
        // external rng advanced by exactly one canonical double draw
        let mut check = Mt19937::new(42);
        check.canonical_double();
        assert_eq!(ext.next_u32(), check.next_u32());
    }

    /// Chain with greedy end: token == argmax; apply_logits round-trips logits.
    #[test]
    fn chain_greedy_end_and_apply_logits() {
        let mut chain = SamplerChain::new();
        chain.add(init_top_k(2));
        chain.add(init_greedy());
        assert_eq!(chain.sample(&[0.1, 5.0, 3.0, 0.2]), 1);

        let mut logits = [1.0, 4.0, 2.0];
        chain.apply_logits(&mut logits);
        assert_eq!(logits[0], 4.0); // top-2 sorted descending
        assert_eq!(logits[1], 2.0);
    }

    /// Chain without a selecting sampler must panic (C: GGML_ASSERT).
    #[test]
    #[should_panic]
    fn chain_without_selector_panics() {
        let mut chain = SamplerChain::new();
        chain.add(init_top_k(2));
        let _ = chain.sample(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn ring_buffer_semantics() {
        let mut rb = RingBuffer::new(3);
        for v in [1, 2, 3, 4, 5] {
            rb.push_back(v);
        }
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.front(), 3); // 1,2 evicted; front is oldest survivor
        rb.clear();
        assert!(rb.is_empty());
    }

    #[test]
    fn empty_chain_no_op() {
        let mut chain = SamplerChain::new();
        let mut cur = cur_from(&[1.0, 2.0]);
        chain.apply(&mut cur);
        assert_eq!(cur.size, 2);
        chain.accept(1);
        chain.reset();
    }
}

// ---------------------------------------------------------------------------
// grammar sampler tests (appended by the GBNF task — the module above is
// untouched, so the pre-existing 51 test behaviours are unchanged)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod grammar_sampler_tests {
    use super::*;
    use crate::grammar::VocabPieces;

    /// synthetic piece table: ids in order, (piece, is_eog)
    fn pieces(items: &[(&[u8], bool)]) -> VocabPieces {
        VocabPieces::from_iter(items.iter().map(|(p, e)| (p.to_vec(), *e)))
    }

    /// ids: 0="a", 1="b", 2="ab", 3="c", 4=eog
    fn ab_pieces() -> VocabPieces {
        pieces(&[
            (b"a", false),
            (b"b", false),
            (b"ab", false),
            (b"c", false),
            (b"", true),
        ])
    }

    fn greedy_ctx(n_vocab: i32) -> SamplingContext {
        // temp <= 0 = greedy (llama_sampler_temp_impl); same chain order as the
        // CLI / common_sampler_init
        SamplingContext::new(
            n_vocab,
            SamplingParams {
                temp: 0.0,
                seed: 0,
                ..Default::default()
            },
        )
    }

    #[test]
    fn init_grammar_errors_and_name() {
        assert!(GrammarSampler::from_pieces("", "root", ab_pieces()).is_err());
        assert!(GrammarSampler::from_pieces("root ::= \"a\"", "nope", ab_pieces()).is_err());
        assert!(
            GrammarSampler::from_pieces("root ::= root \"a\"", "root", ab_pieces()).is_err(),
            "left recursion must be rejected"
        );
        // the `<token>` form needs a real vocab (`<[id]>` does not)
        assert!(GrammarSampler::from_pieces("root ::= <a>", "root", ab_pieces()).is_err());
        assert!(GrammarSampler::from_pieces("root ::= <[0]>", "root", ab_pieces()).is_ok());
        let g = GrammarSampler::from_pieces("root ::= \"a\"", "root", ab_pieces()).unwrap();
        assert_eq!(g.name(), "grammar");
        assert_eq!(g.piece(0), b"a");
        assert_eq!(g.piece(99), b"");
    }

    #[test]
    fn sampler_apply_accept_reset() {
        let mut g = GrammarSampler::from_pieces("root ::= \"ab\"", "root", ab_pieces()).unwrap();

        let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3, 0.4, 1.0]);
        g.apply(&mut cur);
        assert_eq!(cur.data[0].logit, 0.1); // "a"
        assert_eq!(cur.data[1].logit, f32::NEG_INFINITY); // "b"
        assert_eq!(cur.data[2].logit, 0.3); // "ab"
        assert_eq!(cur.data[3].logit, f32::NEG_INFINITY); // "c"
        assert_eq!(cur.data[4].logit, f32::NEG_INFINITY); // EOG (no empty stack)

        g.accept(0); // "a"
        let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3, 0.4, 1.0]);
        g.apply(&mut cur);
        assert_eq!(cur.data[0].logit, f32::NEG_INFINITY);
        assert_eq!(cur.data[1].logit, 0.2); // only "b" continues
        assert_eq!(cur.data[2].logit, f32::NEG_INFINITY);

        g.accept(1); // "b" — grammar complete
        assert!(g.grammar.stacks.iter().any(|s| s.is_empty()));
        let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3, 0.4, 1.0]);
        g.apply(&mut cur);
        assert!(cur.data[..4].iter().all(|d| d.logit == f32::NEG_INFINITY));
        assert_eq!(cur.data[4].logit, 1.0); // EOG now allowed

        g.accept(4); // EOG on a complete grammar is a no-op (llama-grammar.cpp:1445-1452)

        g.reset(); // llama_sampler_grammar_reset
        let mut cur = TokenDataArray::from_logits(&[0.1, 0.2, 0.3, 0.4, 1.0]);
        g.apply(&mut cur);
        assert_eq!(cur.data[0].logit, 0.1);
        assert_eq!(cur.data[2].logit, 0.3);
        assert_eq!(cur.data[1].logit, f32::NEG_INFINITY);
    }

    #[test]
    fn sampler_panics_on_invalid_accept() {
        // "ab" then a second accept of "a" cannot continue the grammar
        let mut g = GrammarSampler::from_pieces("root ::= \"ab\"", "root", ab_pieces()).unwrap();
        g.accept(2); // "ab"
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.accept(0); // "a" — C throws "Unexpected empty grammar stack"
        }));
        assert!(r.is_err(), "expected a panic mirroring the C throw");
    }

    #[test]
    fn sample_with_grammar_resamples_invalid_token() {
        // argmax is the EOG token, which the grammar rejects (no empty stack):
        // the resample path must return the grammar-valid argmax ("ab")
        let mut g = GrammarSampler::from_pieces("root ::= \"ab\"", "root", ab_pieces()).unwrap();
        let mut ctx = greedy_ctx(5);
        let tok = ctx
            .sample_with_grammar(&[0.1, 0.2, 0.9, 1.0, 2.0], &mut g)
            .unwrap();
        assert_eq!(tok, 2);
        assert!(g.grammar.stacks.iter().any(|s| s.is_empty()));
        assert_eq!(ctx.prev.front(), 2);

        // valid argmax → returned without resampling
        let mut g2 = GrammarSampler::from_pieces("root ::= \"ab\"", "root", ab_pieces()).unwrap();
        let mut ctx2 = greedy_ctx(5);
        let tok = ctx2
            .sample_with_grammar(&[0.1, 0.2, 1.0, 0.5, 0.0], &mut g2)
            .unwrap();
        assert_eq!(tok, 2);

        // grammar-constrained argmax differs from the raw argmax
        let mut g3 = GrammarSampler::from_pieces("root ::= \"a\"", "root", ab_pieces()).unwrap();
        let mut ctx3 = greedy_ctx(5);
        let tok = ctx3
            .sample_with_grammar(&[0.1, 0.9, 0.8, 0.7, 2.0], &mut g3)
            .unwrap();
        assert_eq!(tok, 0, "only \"a\" satisfies root ::= \"a\"");
    }
}
