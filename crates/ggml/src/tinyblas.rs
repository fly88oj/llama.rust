//! tinyblas.rs — llamafile tinyBLAS, the reference's production GEMM path
//! (ggml/src/ggml-cpu/llamafile/sgemm.cpp) for multi-column `mul_mat`.
//!
//! `ggml_compute_forward_mul_mat` tries `llamafile_sgemm` **twice**
//! (ggml-cpu.c:1306 and :1389) and only falls back to the row-wise `vec_dot`
//! loop when both return false:
//!
//! * attempt 1 (`src1` contiguous) passes `Btype = src1->type`; `src1` is
//!   always F32 in this port, so only F32 weights are accepted there;
//! * attempt 2 runs after `src1` was quantized into `wdata` for the `vec_dot`
//!   path, i.e. only when `src1->type != vec_dot_type`, and passes
//!   `Btype = vec_dot_type(src0->type)` — that is what routes Q5_0/Q8_0
//!   weights (`vec_dot_type == Q8_0`) and F16/BF16 weights
//!   (`vec_dot_type == self`) into tinyBLAS.
//!
//! `llamafile_sgemm` (sgemm.cpp:3805) is a switch over `Atype` with a
//! `Btype`-pairing guard per case, then the per-class gate:
//!
//! | Atype  | Btype required | class                                  | gate                       |
//! |--------|----------------|----------------------------------------|----------------------------|
//! | F32    | F32            | `tinyBLAS<16,__m512>` (:3833)          | `m%4==0` (K tails ok)      |
//! | F16    | F16            | `tinyBLAS<16,__m512>` (:3960)          | `m%4==0` (K tails ok)      |
//! | BF16   | BF16           | `tinyBLAS<32,__m512bh>` (:3896)        | `m%4==0` (K tails ok)      |
//! | Q8_0   | Q8_0           | `tinyBLAS_Q0_AVX` (:4045)              | none (any m, n, k)         |
//! | Q4_0   | Q8_0           | `tinyBLAS_Q0_AVX` (:4082)              | none                       |
//! | Q5_0   | Q8_0           | `tinyBLAS_Q0_AVX` (:4119)              | none                       |
//! | IQ4_NL | Q8_0           | `tinyBLAS_Q0_AVX` (:4135)              | none                       |
//! | other  | —              | —                                      | `return false`             |
//!
//! with two guards ahead of the switch: `n < 2` (:3820, prompt processing only)
//! and `Ctype != F32` (:3824; the port always writes f32). `m`, `n`, `k` are the
//! *per-(i12,i13) slice* values `ne01`, `ne11`, `ne00/blck_size(src0->type)`:
//! the call is made once per broadcast plane, so a 3D `mul_mat` with
//! `ne11 == 1` never reaches tinyBLAS no matter how many columns it has
//! overall. The BF16 `KN` is 32 only under `__AVX512BF16__` (this host, see
//! parity/tinyblas_ref.bin's build word); the `#elif defined(__AVX512F__)`
//! branch would instantiate `KN = 16` (:3904).
//!
//! ## K tails (upstream #29806, sync batch D2)
//!
//! Since a7b94df2c `tinyBLAS::matmul` only keeps the `k % KN != 0` bail on
//! non-x86 builds (`#if !defined(__AVX__) && !defined(__AVX2__) &&
//! !defined(__AVX512F__)`, sgemm.cpp:569-572), and `gemm_bloc` walks full
//! blocks (`l + KN <= k`, :627) and then handles the `rem = k % KN` remainder
//! with masked `load_partial` vectors + one extra `madd` per accumulator
//! (:653-668) before `hsum`. Masked-off lanes load as zero, whose products add
//! exactly 0.0 to the accumulator, so the tail contributes only the elements
//! `k-rem..k-1` into lanes `0..rem` — the value stays the C's exact op
//! sequence and the port mirrors it bit-for-bit (`f::panel*`'s epilogue).
//!
//! ## Values
//!
//! Every tinyBLAS microkernel keeps **one accumulator per output element** and
//! walks `k` in order, so an element's value does not depend on the tile (or
//! the thread) that computes it:
//!
//! * `tinyBLAS::gemm_bloc` (sgemm.cpp:625-671) accumulates
//!   `Cv[j][i] = madd(Av[i], Bv, Cv[j][i])` for `l += KN` with `V = __m512`
//!   (`__m512bh` for BF16) — plus the K-tail `madd` after the last full block
//!   (#29806, :653-668) — and ends in `hsum` = `_mm512_reduce_add_ps` (fold
//!   8/4/2/1 = `vec_dot::reduce_add16`): for F32/F16 lane `t` sums the values
//!   `k ≡ t (mod 16)` plus the tail elements in lanes `< rem`, for BF16 the
//!   VDPBF16PS pair `(2t, 2t+1)`.
//! * `tinyBLAS_Q0_AVX::{gemm,gemm4xN,gemmMx4}` (sgemm.cpp:1634/1524/1578) do
//!   `Cv[j][i] = madd(set1(dA*dB), updot(sign(A,A), sign(B,A)), Cv[j][i])`
//!   once per 32-value block and end in `hsum` (quants.c `hsum_float_8`).
//!   `updot`/`load` are literally the reference's AVX2 `vec_dot` helpers —
//!   with `updot` compiled to `vpdpbusd` on an `__AVX512VNNI__ &&
//!   __AVX512VL__` host (sgemm.cpp:1756-1757), which the port's `vnni_lane`
//!   tiles mirror (`dpbusd` is value-identical to the `maddubsw + maddwd`
//!   pair for these operand ranges — no int16 saturation is reachable; see
//!   `vnni_lane`'s doc) — which is why the port's `vec_dot_row` already
//!   reproduces the reference's tinyBLAS output for Q4_0/Q5_0/Q8_0
//!   (PARITY.md).
//!
//! This port keeps that arithmetic and reorganizes only the *scheduling*: a row
//! panel of `RM = 4` weight rows is decoded once and reused across every column
//! block (the C re-decodes it once per `RN`-wide column tile — sgemm.cpp:554-
//! 557), and the caller splits the (row block × column block) grid across
//! threads. Both are partitioning-only changes: each output element still runs
//! the C's exact operation sequence, so results stay bit-exact and independent
//! of the thread count. `parity/tinyblas_ref.bin` + `dump_tests` pin both the
//! routing decision and the values against the shipped reference binary.

use crate::types::GgmlType;

/// The `llamafile_sgemm` case a (Atype, Btype) pair selects on this build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// `tinyBLAS<16, __m512, __m512, float, float, float>` (sgemm.cpp:3833).
    F32,
    /// `tinyBLAS<16, __m512, __m512, ggml_fp16_t, ...>` (sgemm.cpp:3960).
    F16,
    /// `tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ...>` under `__AVX512BF16__`
    /// (sgemm.cpp:3896), else `tinyBLAS<16, __m512, __m512, ggml_bf16_t, ...>`
    /// (:3904).
    Bf16,
    /// `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0>` (sgemm.cpp:4045).
    Q8xQ8,
    /// `tinyBLAS_Q0_AVX<block_q4_0, block_q8_0>` (sgemm.cpp:4082).
    Q4xQ8,
    /// `tinyBLAS_Q0_AVX<block_q5_0, block_q8_0>` (sgemm.cpp:4119).
    Q5xQ8,
}

impl Op {
    /// The C's `KN` template parameter, in *elements* for the float classes and
    /// in *blocks* for the Q0 class (one 32-value block per iteration).
    pub fn kn(self) -> usize {
        match self {
            Op::F32 | Op::F16 => 16,
            Op::Bf16 => 32, // __AVX512BF16__ host: tinyBLAS<32, __m512bh>
            Op::Q8xQ8 | Op::Q4xQ8 | Op::Q5xQ8 => 32,
        }
    }

    /// `tinyBLAS::matmul`'s gate (sgemm.cpp:568-612): the `k % KN != 0` bail is
    /// x86-only removed (`#if !defined(__AVX__) && !defined(__AVX2__) &&
    /// !defined(__AVX512F__)`, :569-572 — since #29806 the K remainder is
    /// handled by `gemm_bloc`'s masked-load epilogue), then the
    /// `m % 16 / % 8 / % 4` chain picks BM and `m % 4 != 0` bails. The port's
    /// float lanes exist only under AVX512 (`resolve`), i.e. exactly the builds
    /// whose `__AVX512F__` drops the k gate. The Q0 class has no gate at all
    /// (sgemm.cpp:1371 `mnpack(0, m, 0, n)`).
    pub fn eligible(self, m: usize, n: usize, _k: usize) -> bool {
        if n < 2 {
            // sgemm.cpp:3819-3822 — "only enable sgemm for prompt processing"
            return false;
        }
        match self {
            // K tails handled in-kernel since a7b94df2c (sgemm.cpp:653-668)
            Op::F32 | Op::F16 | Op::Bf16 => m % 4 == 0,
            Op::Q8xQ8 | Op::Q4xQ8 | Op::Q5xQ8 => true,
        }
    }
}

/// How many `gemm` calls each `Op` served *on this thread*, as a *wiring*
/// probe: the dump tests pin the kernel and the predicate, this pins that
/// `compute.rs`'s `mul_mat` actually routes there (and nowhere else) for a real
/// graph. Thread-local so that cargo's parallel test threads cannot pollute a
/// measurement; `Team::run` always executes shard 0 on the calling thread, so a
/// routed `mul_mat` shows up here.
std::thread_local! {
    static CALLS: [std::cell::Cell<usize>; 6] = [const { std::cell::Cell::new(0) }; 6];
}

impl Op {
    fn idx(self) -> usize {
        match self {
            Op::F32 => 0,
            Op::F16 => 1,
            Op::Bf16 => 2,
            Op::Q8xQ8 => 3,
            Op::Q4xQ8 => 4,
            Op::Q5xQ8 => 5,
        }
    }
}

/// Snapshot of this thread's per-`Op` `gemm` call count (see the docs above).
pub fn calls() -> [usize; 6] {
    CALLS.with(|c| {
        let mut out = [0usize; 6];
        for (i, v) in out.iter_mut().enumerate() {
            *v = c[i].get();
        }
        out
    })
}

#[inline]
fn count_call(op: Op) {
    CALLS.with(|c| c[op.idx()].set(c[op.idx()].get() + 1));
}

/// The `switch (Atype)` of `llamafile_sgemm` (sgemm.cpp:3827-4149) plus each
/// case's `Btype` guard, restricted to the host features this port implements.
/// `None` is the reference's `return false`.
///
/// The reference binary is `-march=native` on an AVX512 + AVX512BF16 host
/// (build-rust-ref/CMakeCache.txt: GGML_NATIVE=ON, all GGML_AVX* off ⇒ the
/// compiler's own macros), so its `__AVX512F__`/`__AVX512BF16__` branches are
/// the ones compiled in; the port takes a path only when the host has the same
/// features and otherwise keeps the `vec_dot` fallback (a non-AVX512 reference
/// build instantiates `tinyBLAS<8,__m256>`, whose lane structure differs).
pub fn resolve(ty0: GgmlType, ty1: GgmlType) -> Option<Op> {
    use GgmlType::*;
    let op = match ty0 {
        F32 if ty1 == F32 => Op::F32,
        F16 if ty1 == F16 => Op::F16,
        Bf16 if ty1 == Bf16 => Op::Bf16,
        Q8_0 if ty1 == Q8_0 => Op::Q8xQ8,
        Q4_0 if ty1 == Q8_0 => Op::Q4xQ8,
        Q5_0 if ty1 == Q8_0 => Op::Q5xQ8,
        // IQ4_NL (sgemm.cpp:4131) has no quantizer / vec_dot in this port, so
        // it cannot appear as a weight type here (the dump tests record it as
        // the one case the reference accepts and the port does not).
        _ => return None,
    };
    let ok = match op {
        Op::F32 | Op::F16 | Op::Bf16 => avx512(),
        Op::Q8xQ8 | Op::Q4xQ8 | Op::Q5xQ8 => crate::simd_x86::avx2(),
    };
    ok.then_some(op)
}

/// `llamafile_sgemm`'s complete decision for one slice: the type switch
/// (`resolve`) plus the `n >= 2` / `m % 4` gates (k remainders are in-kernel
/// since #29806 — see [`Op::eligible`]).
pub fn accepts(ty0: GgmlType, ty1: GgmlType, m: usize, n: usize, k: usize) -> bool {
    match resolve(ty0, ty1) {
        Some(op) => op.eligible(m, n, k),
        None => false,
    }
}

/// Host has the AVX512 baseline the float instantiations need.
pub fn avx512() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    })
}

/// Host has the bf16 dot product: selects `tinyBLAS<32,__m512bh>` (sgemm.cpp:3894)
/// over the widen-to-f32 fallback (sgemm.cpp:3902).
pub fn avx512bf16() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| avx512() && is_x86_feature_detected!("avx512bf16"))
}

/// Rows of A per row panel: `tinyBLAS::mnpack<4, ...>` (sgemm.cpp:501) and the
/// `tinyBLAS_Q0_AVX::mnpack` `case 0x44` tile (sgemm.cpp:1380).
pub const RM: usize = 4;

/// Columns of B per column block: the `RN` of `mnpack<.., 6, ..>` /
/// `gemm4xN<RN>` (sgemm.cpp:501/1384).
pub const RN: usize = 4;

/// One `llamafile_sgemm` slice: `C = Aᵀ·B`, `A` = `m` weight rows of `k` values
/// (`lda` values per row — in *elements* for the float ops, in *blocks* for the
/// Q0 ops, exactly the C call site's `nb01/type_size`), `B` = `n` activation
/// rows already in `Btype` (`ldb` per row), `C` = `m × n` f32 with row stride
/// `ldc` (the C passes `nb1/4 == ne01`).
///
/// `i0..i1` / `j0..j1` are the caller's share of the (row block × column block)
/// grid. Each output element is one independent accumulator chain, so the split
/// cannot change a value.
///
/// # Safety
/// `a` must address `m` rows of `lda` units (`ggml_type_size(ty0)` bytes each),
/// `b` `n` rows of `ldb` units, and `c` must be valid for `ldc * n` f32;
/// `i1 <= m`, `j1 <= n`, and the host must have the features `accepts` gates on.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    op: Op,
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    lda: usize,
    b: *const u8,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    debug_assert!(i0 <= i1 && i1 <= m && j0 <= j1 && j1 <= n);
    count_call(op);
    #[cfg(target_arch = "x86_64")]
    match op {
        Op::F32 | Op::F16 | Op::Bf16 => f::gemm(op, k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
        Op::Q8xQ8 | Op::Q4xQ8 | Op::Q5xQ8 => {
            q0::gemm(op, k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (op, m, n, k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1);
        unreachable!("tinyblas: no SIMD kernel on this arch");
    }
}

// ======================================================================
// tinyBLAS<KN, __m512[, __m512bh]> — F32 / F16 / BF16
// (sgemm.cpp:3829-4039's `__AVX512F__` / `__AVX512BF16__` instantiations)
// ======================================================================

#[cfg(target_arch = "x86_64")]
mod f {
    use super::{avx512bf16, Op, RM, RN};
    use core::arch::x86_64::*;

    /// `load<V>` for the 16-lane float classes (sgemm.cpp:360-369), selected by
    /// the C's `TA`: 0 = `float` (`_mm512_loadu_ps`), 1 = `ggml_fp16_t`
    /// (`_mm512_cvtph_ps`), 2 = `ggml_bf16_t` in the non-`__AVX512BF16__` build
    /// (`cvtepu16_epi32` + `slli 16`).
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")]
    unsafe fn load16<const TA: u8>(p: *const u8) -> __m512 {
        match TA {
            0 => _mm512_loadu_ps(p as *const f32),
            1 => _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i)),
            _ => _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i)),
                16,
            )),
        }
    }

    /// `load_partial_u16<__m256i>` (sgemm.cpp:425-438): the reference build's
    /// `__AVX512BW__ && __AVX512VL__` branch `_mm256_maskz_loadu_epi16((1u<<n)-1,
    /// p)` — the first `n` u16 lanes loaded, the rest zero. (The `#else`
    /// maskload-with-pairs spelling is value-identical: both fill exactly n
    /// u16 slots, so one branch suffices for bit-parity.)
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")]
    unsafe fn load_partial_u16_512(p: *const u8, n: usize) -> __m256i {
        _mm256_maskz_loadu_epi16(((1u16 << n) - 1) as __mmask16, p as *const _)
    }

    /// `load_partial<__m512>(const TA *, int)` for the 16-lane float classes
    /// (sgemm.cpp:441-451): f32 = `_mm512_maskz_loadu_ps` (:441-442), f16 =
    /// `cvtph` of the masked u16 load (:445-446), bf16 (non-`__AVX512BF16__`) =
    /// `cvtepu16_epi32` + `slli 16` of it (:449-450).
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")]
    unsafe fn load_partial16<const TA: u8>(p: *const u8, n: usize) -> __m512 {
        match TA {
            0 => _mm512_maskz_loadu_ps(((1u16 << n) - 1) as __mmask16, p as *const f32),
            1 => _mm512_cvtph_ps(load_partial_u16_512(p, n)),
            _ => _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(load_partial_u16_512(p, n)),
                16,
            )),
        }
    }

    /// `tinyBLAS::gemm_bloc<RM, RN>` (sgemm.cpp:625-671) for the 16-lane
    /// classes, `madd` = `_mm512_fmadd_ps` (sgemm.cpp:147):
    ///
    /// ```text
    /// D Cv[RN][RM] = {};
    /// for (l = 0; l + KN <= k; l += KN) {
    ///     V Av[RM]; for (i) Av[i] = load(A + lda*(ii+i) + l);
    ///     for (j) { V Bv = load(B + ldb*(jj+j) + l);
    ///               for (i) Cv[j][i] = madd(Av[i], Bv, Cv[j][i]); }
    /// }
    /// rem = k % KN;                       // K tail (#29806, sgemm.cpp:653-668)
    /// if (rem) { for (i) Av[i] = load_partial(A + lda*(ii+i) + k - rem, rem);
    ///            for (j) { V Bv = load_partial(B + ldb*(jj+j) + k - rem, rem);
    ///                      for (i) Cv[j][i] = madd(Av[i], Bv, Cv[j][i]); } }
    /// for (j) for (i) C[ldc*(jj+j) + ii+i] = hsum(Cv[j][i]);
    /// ```
    ///
    /// The C's `mnpack<4,6,BM>` (sgemm.cpp:611-624) only chooses which block a
    /// thread walks; with one accumulator per element the value is tiling
    /// independent, so this walks `RM`-row panels against `RN`-wide column
    /// blocks. The tail's masked loads zero the lanes `>= rem`; their products
    /// add exactly `c + 0*0` to the accumulator, so the tail contributes only
    /// elements `k-rem..k-1` into lanes `0..rem` — the C's op order verbatim.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")]
    unsafe fn panel<const TA: u8>(
        k: usize,
        a: *const u8,
        lda: usize,
        b: *const u8,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        let asz = match TA {
            0 => 4,
            _ => 2,
        };
        let mut ii = i0;
        while ii < i1 {
            let rme = RM.min(i1 - ii);
            let mut jj = j0;
            while jj < j1 {
                let rne = RN.min(j1 - jj);
                let mut cv = [[_mm512_setzero_ps(); RM]; RN];
                let mut l = 0;
                while l + 16 <= k {
                    let mut av = [_mm512_setzero_ps(); RM];
                    for i in 0..rme {
                        av[i] = load16::<TA>(a.add((ii + i) * lda * asz + l * asz));
                    }
                    for j in 0..rne {
                        let bv = load16::<TA>(b.add((jj + j) * ldb * asz + l * asz));
                        for i in 0..rme {
                            cv[j][i] = _mm512_fmadd_ps(av[i], bv, cv[j][i]);
                        }
                    }
                    l += 16;
                }
                // K tail (sgemm.cpp:653-668): rem masked loads, one extra madd
                // per accumulator, before hsum. The C loops the full RM/RN
                // (gemm_bloc only ever sees whole tiles); the port's runtime
                // rme/rne bounds are the same elements.
                let rem = k % 16;
                if rem != 0 {
                    let base = (k - rem) * asz;
                    let mut av = [_mm512_setzero_ps(); RM];
                    for i in 0..rme {
                        av[i] = load_partial16::<TA>(a.add((ii + i) * lda * asz + base), rem);
                    }
                    for j in 0..rne {
                        let bv = load_partial16::<TA>(b.add((jj + j) * ldb * asz + base), rem);
                        for i in 0..rme {
                            cv[j][i] = _mm512_fmadd_ps(av[i], bv, cv[j][i]);
                        }
                    }
                }
                for j in 0..rne {
                    for i in 0..rme {
                        *c.add((jj + j) * ldc + ii + i) = _mm512_reduce_add_ps(cv[j][i]);
                    }
                }
                jj += rne;
            }
            ii += rme;
        }
    }

    /// `(const __m512bh)_mm512_loadu_ps((const float *)p)` (sgemm.cpp:373-378).
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")]
    unsafe fn bf16_load(p: *const u8) -> __m512bh {
        core::mem::transmute(_mm512_loadu_si512(p as *const _))
    }

    /// Zero of the `bf16` vector type (`acc_t` init in `gemm_bloc`).
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")]
    unsafe fn bf16_zero() -> __m512bh {
        core::mem::transmute::<__m512i, __m512bh>(_mm512_setzero_si512())
    }

    /// `load_partial<__m512bh>(const ggml_bf16_t *, int)` (sgemm.cpp:455-457):
    /// `(const __m512bh)_mm512_maskz_loadu_epi16((1<<n)-1, p)` — the first `n`
    /// bf16 lanes loaded (rem <= KN-1 = 31, so the C's `uint64_t(1)` shift
    /// cannot overflow; `__mmask32`/u32 here), the rest zero.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")]
    unsafe fn bf16_load_partial(p: *const u8, n: usize) -> __m512bh {
        core::mem::transmute(_mm512_maskz_loadu_epi16((1u32 << n) - 1, p as *const _))
    }

    /// `tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ...>` (sgemm.cpp:3896):
    /// `KN = 32`, `load<__m512bh>` is a raw 64-byte load and `madd` =
    /// `_mm512_dpbf16_ps` (sgemm.cpp:152-155). VDPBF16PS adds `a·b` into `c`
    /// with lane `t` taking the bf16 pair `(2t, 2t+1)` as
    /// `fma(a1,b1, fma(a0,b0, c))` in f32 — see `vec_dot_tinyblas_bf16`.
    /// The K tail (#29806, sgemm.cpp:653-668) uses `bf16_load_partial` — the
    /// zeroed upper pairs add exactly `c + 0*0 + 0*0` per lane.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")]
    unsafe fn panel_bf16(
        k: usize,
        a: *const u8,
        lda: usize,
        b: *const u8,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        let mut ii = i0;
        while ii < i1 {
            let rme = RM.min(i1 - ii);
            let mut jj = j0;
            while jj < j1 {
                let rne = RN.min(j1 - jj);
                let mut cv = [[_mm512_setzero_ps(); RM]; RN];
                let mut l = 0;
                while l + 32 <= k {
                    let mut av = [bf16_zero(); RM];
                    for i in 0..rme {
                        av[i] = bf16_load(a.add((ii + i) * lda * 2 + l * 2));
                    }
                    for j in 0..rne {
                        let bv = bf16_load(b.add((jj + j) * ldb * 2 + l * 2));
                        for i in 0..rme {
                            cv[j][i] = _mm512_dpbf16_ps(cv[j][i], av[i], bv);
                        }
                    }
                    l += 32;
                }
                // K tail (sgemm.cpp:653-668)
                let rem = k % 32;
                if rem != 0 {
                    let base = (k - rem) * 2;
                    let mut av = [bf16_zero(); RM];
                    for i in 0..rme {
                        av[i] = bf16_load_partial(a.add((ii + i) * lda * 2 + base), rem);
                    }
                    for j in 0..rne {
                        let bv = bf16_load_partial(b.add((jj + j) * ldb * 2 + base), rem);
                        for i in 0..rme {
                            cv[j][i] = _mm512_dpbf16_ps(cv[j][i], av[i], bv);
                        }
                    }
                }
                for j in 0..rne {
                    for i in 0..rme {
                        *c.add((jj + j) * ldc + ii + i) = _mm512_reduce_add_ps(cv[j][i]);
                    }
                }
                jj += rne;
            }
            ii += rme;
        }
    }

    pub unsafe fn gemm(
        op: Op,
        k: usize,
        a: *const u8,
        lda: usize,
        b: *const u8,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        match op {
            Op::F32 => panel::<0>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
            Op::F16 => panel::<1>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
            Op::Bf16 if avx512bf16() => panel_bf16(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
            Op::Bf16 => panel::<2>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
            _ => unreachable!("tinyblas float gemm: {op:?}"),
        }
    }
}

// ======================================================================
// tinyBLAS_Q0_AVX — Q8_0×Q8_0, Q4_0×Q8_0, Q5_0×Q8_0
// (sgemm.cpp:1351-1795, instantiated at :4041-4145)
// ======================================================================

#[cfg(target_arch = "x86_64")]
mod q0 {
    use super::{Op, RM, RN};
    use crate::simd_x86 as sx;
    use core::arch::x86_64::*;

    /// Column-tile width of the Q0 kernel: the `nc` of the C's
    /// `mnpack` `case 0x44` under `VECTOR_REGISTERS == 32` (sgemm.cpp:1380-
    /// 1384), i.e. `gemm4xN<4>` — 4x4 tiles. (The 6-wide tile belongs to the
    /// *float* class's `mnpack<4,6,BM>` (:499-513); the Q0 class picks 4.)
    /// Any width gives the same values — one accumulator per output element —
    /// so this is a pure throughput knob; `bench_tests` compares widths.
    /// Covers both 32-register lanes (VNNI and plain EVEX); live vectors:
    /// 4*4 (Cv) + 4 (av) + dvec + bq + db + temps ≈ 24-26 of 32.
    const RN_TILE: usize = 4;

    /// The AVX2 fallback's column tile: the C's 16-register build picks
    /// `nc = 2` (`#else` of `VECTOR_REGISTERS == 32`, sgemm.cpp:1435-1441) —
    /// `4*2 + 4 = 12 ≤ 16` accumulators fit without spilling (RN_TILE = 6 on
    /// a 16-register host spills most of the 28 live vectors every k step).
    const AVX2_RN_TILE: usize = 2;

    /// `tinyBLAS_Q0_AVX::load(block_q8_0)` (sgemm.cpp:1684).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_q8_0(b: *const u8) -> __m256i {
        _mm256_loadu_si256(b.add(2) as *const __m256i)
    }

    /// `load(block_q4_0)` (sgemm.cpp:1696): `denibble(qs) - 8`.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_q4_0(b: *const u8) -> __m256i {
        _mm256_sub_epi8(sx::bytes_from_nibbles_32(b.add(2)), _mm256_set1_epi8(8))
    }

    /// `load(block_q5_0)` (sgemm.cpp:1710): `denibble(qs) | bittobyte(qh)`,
    /// where `bittobyte` sets `0xF0` (= -16 as int8) on every element whose
    /// 5th bit is *clear*.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_q5_0(b: *const u8) -> __m256i {
        _mm256_or_si256(
            sx::bytes_from_nibbles_32(b.add(6)),
            _mm256_andnot_si256(sx::bytes_from_bits_32(b.add(2)), _mm256_set1_epi8(-16)),
        )
    }

    /// Size of one weight block (`ggml_type_size`): q8_0 34, q4_0 18, q5_0 22.
    #[inline]
    const fn block_bytes(ty: u8) -> usize {
        match ty {
            0 => 34,
            1 => 18,
            _ => 22,
        }
    }

    /// The A-side `load<T>` of `tinyBLAS_Q0_AVX` for `TA` = 0/1/2 =
    /// q8_0/q4_0/q5_0 (sgemm.cpp:1684-1712).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_a<const TA: u8>(b: *const u8) -> __m256i {
        match TA {
            0 => load_q8_0(b),
            1 => load_q4_0(b),
            _ => load_q5_0(b),
        }
    }

    /// `tinyBLAS_Q0_AVX::gemm<RM,RN>` (sgemm.cpp:1634-1682), i.e.
    ///
    /// ```text
    /// __m256 Cv[RN][RM] = {};
    /// for (l = 0; l < k; ++l)                       // l walks 32-value blocks
    ///     for (j) for (i)
    ///         Cv[j][i] = madd(set1(dA*dB),
    ///                         updot(sign(A,A), sign(B,A)), Cv[j][i]);
    /// for (j) for (i) C[ldc*(jj+j) + ii+i] = hsum(Cv[j][i]);
    /// ```
    ///
    /// `updot` here is the AVX2 `mul_sum_i8_pairs_float` body the port already
    /// uses in `vec_dot_row` (the C takes its `__AVX512VNNI__` `dpbusd` branch
    /// on this host; the two are value-identical for these operand ranges —
    /// see simd_x86.rs's header). `hsum` is `hsum_float_8`.
    ///
    /// `NROW`/`NCOL` are compile-time so the `NROW*NCOL` accumulators and the
    /// unrolled body stay in registers (the C's `gemm4xN<RN>` has the same
    /// shape); the C re-runs `load(A)` inside the `j` loop (sgemm.cpp:1652),
    /// this reads the pre-decoded row panel instead.
    /// Instantiated twice from one body — plain AVX2 (VEX, 16 ymm) and the
    /// EVEX flavor (`avx512f/vl/bw/dq`: the extended ymm16-31 register file,
    /// which is the instruction selection the reference build gets on this
    /// host). Same lanes, same op order, bit-identical by construction (see
    /// `simd_x86.rs`'s q6_K pair for the precedent).
    macro_rules! q0_tile {
        ($(#[$attr:meta])* $name:ident) => {
            $(#[$attr])*
            unsafe fn $name<const NROW: usize, const NCOL: usize>(
                k: usize,
                panel: *const u8,
                adelta: *const f32,
                b: *const u8,
                ldb: usize,
                c: *mut f32,
                ldc: usize,
                ii: usize,
                jj: usize,
            ) {
                let mut cv = [[_mm256_setzero_ps(); NROW]; NCOL];
                for l in 0..k {
                    let mut av = [_mm256_setzero_si256(); NROW];
                    for (i, a) in av.iter_mut().enumerate() {
                        *a = _mm256_loadu_si256(panel.add((i * k + l) * 32) as *const __m256i);
                    }
                    for (j, cvj) in cv.iter_mut().enumerate() {
                        // B is always Q8_0 (`Btype != Q8_0` is rejected in the
                        // dispatcher, sgemm.cpp:4042/4079/4116).
                        let pb = b.add((jj + j) * ldb * 34 + l * 34);
                        let bq = load_q8_0(pb);
                        let db = sx::d_f32(pb);
                        for (i, cvi) in cvj.iter_mut().enumerate() {
                            let q = sx::mul_sum_i8_pairs_float(av[i], bq);
                            let d = _mm256_set1_ps(*adelta.add(l * RM + i) * db);
                            *cvi = _mm256_fmadd_ps(d, q, *cvi);
                        }
                    }
                }
                for (j, cvj) in cv.iter().enumerate() {
                    for (i, cvi) in cvj.iter().enumerate() {
                        *c.add((jj + j) * ldc + ii + i) = sx::hsum8(*cvi);
                    }
                }
            }
        };
    }

    q0_tile!(
        #[target_feature(enable = "avx2,fma,f16c")]
        tile_avx2
    );
    q0_tile!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq")]
        tile_evex
    );

    /// Row- or column-tail tile: the same body with runtime bounds. Only the
    /// last panel/block of a GEMM whose `m`/`n` is not a tile multiple takes
    /// this (the C's `mnpack` reaches `gemm<3,2>`, `gemm<1,1>`, … the same way).
    macro_rules! q0_tile_tail {
        ($(#[$attr:meta])* $name:ident) => {
            $(#[$attr])*
            unsafe fn $name<const NCOL: usize>(
                k: usize,
                panel: *const u8,
                adelta: *const f32,
                b: *const u8,
                ldb: usize,
                c: *mut f32,
                ldc: usize,
                ii: usize,
                jj: usize,
                rme: usize,
                rne: usize,
            ) {
                let mut cv = [[_mm256_setzero_ps(); RM]; NCOL];
                for l in 0..k {
                    let mut av = [_mm256_setzero_si256(); RM];
                    for i in 0..rme {
                        av[i] = _mm256_loadu_si256(panel.add((i * k + l) * 32) as *const __m256i);
                    }
                    for j in 0..rne {
                        let pb = b.add((jj + j) * ldb * 34 + l * 34);
                        let bq = load_q8_0(pb);
                        let db = sx::d_f32(pb);
                        for i in 0..rme {
                            let q = sx::mul_sum_i8_pairs_float(av[i], bq);
                            let d = _mm256_set1_ps(*adelta.add(l * RM + i) * db);
                            cv[j][i] = _mm256_fmadd_ps(d, q, cv[j][i]);
                        }
                    }
                }
                for (j, cvj) in cv.iter().enumerate().take(rne) {
                    for (i, cvi) in cvj.iter().enumerate().take(rme) {
                        *c.add((jj + j) * ldc + ii + i) = sx::hsum8(*cvi);
                    }
                }
            }
        };
    }

    q0_tile_tail!(
        #[target_feature(enable = "avx2,fma,f16c")]
        tile_tail_avx2
    );
    q0_tile_tail!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq")]
        tile_tail_evex
    );

    /// The VNNI lane of `tinyBLAS_Q0_AVX::gemm4xN` — what the reference build
    /// runs on an `__AVX512VNNI__ && __AVX512VL__` host (this one): `updot` is
    /// `_mm256_dpbusd_epi32(zero, u, s)` (sgemm.cpp:1756-1757) instead of the
    /// `maddubsw + maddwd` pair, and the four rows' `d` values are handled as
    /// one packed product: `da = cvtph(d0..d3)` per k (sgemm.cpp:1537-1540),
    /// `dvec = da * db` per (j, k), then one lane `vshufps` per (i, j, k)
    /// (sgemm.cpp:1545-1560). Per (i,j,k) step: shuffle + 2×vpsignb + vpdpbusd
    /// + vcvtdq2ps + vfmadd — the exact instruction mix of the reference's
    /// compiled `gemm4xN<4>`; the port's pre-decoded row panel only replaces
    /// the C's per-column-tile A re-decode.
    ///
    /// `adelta` here is the f32 delta array in *interleaved* layout — the d of
    /// row `i`, block `l` lives at `adelta[l*RM + i]` — so one `_mm_loadu_ps`
    /// fetches all four rows' deltas of block l. The port converts the fp16 at
    /// panel-decode time (`d_f32` == the C's `unhalf`, see simd_x86.rs's
    /// exhaustive f16 proof), so `da` skips the C's per-k `_mm_cvtph_ps` —
    /// same values, one op less per k.
    macro_rules! q0_tile_vnni {
        ($(#[$attr:meta])* $name:ident) => {
            $(#[$attr])*
            unsafe fn $name<const NCOL: usize>(
                k: usize,
                panel: *const u8,
                adelta: *const f32,
                b: *const u8,
                ldb: usize,
                c: *mut f32,
                ldc: usize,
                ii: usize,
                jj: usize,
            ) {
                let mut cv = [[_mm256_setzero_ps(); RM]; NCOL];
                for l in 0..k {
                    let mut av = [_mm256_setzero_si256(); RM];
                    for (i, a) in av.iter_mut().enumerate() {
                        *a = _mm256_loadu_si256(panel.add((i * k + l) * 32) as *const __m256i);
                    }
                    // sgemm.cpp:1537-1540: the four rows' d's of block l in one
                    // xmm (the C packs the raw fp16s; we keep the converted f32s)
                    let da = _mm_loadu_ps(adelta.add(l * RM));
                    for (j, cvj) in cv.iter_mut().enumerate() {
                        // B is always Q8_0 (`Btype != Q8_0` is rejected in the
                        // dispatcher, sgemm.cpp:4042/4079/4116).
                        let pb = b.add((jj + j) * ldb * 34 + l * 34);
                        let bq = load_q8_0(pb);
                        // sgemm.cpp:1542-1545: dvec = (da * set1(db)) replicated
                        // across both 128-bit halves
                        let d256 =
                            _mm256_castps128_ps256(_mm_mul_ps(da, _mm_set1_ps(sx::d_f32(pb))));
                        let dvec = _mm256_permute2f128_ps(d256, d256, 0);
                        for (i, cvi) in cvj.iter_mut().enumerate() {
                            // sgemm.cpp:1547-1559: sign(A,A)/sign(B,A) then updot
                            // (= dpbusd on this lane), madd with lane i's d
                            let u = _mm256_sign_epi8(av[i], av[i]);
                            let s = _mm256_sign_epi8(bq, av[i]);
                            let q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(
                                _mm256_setzero_si256(),
                                u,
                                s,
                            ));
                            // the C's shuffle immediates (sgemm.cpp:1549-1559):
                            // _MM_SHUFFLE(i,i,i,i) = 0 / 85 / 170 / 255
                            let d = match i {
                                0 => _mm256_shuffle_ps(dvec, dvec, 0),
                                1 => _mm256_shuffle_ps(dvec, dvec, 85),
                                2 => _mm256_shuffle_ps(dvec, dvec, 170),
                                _ => _mm256_shuffle_ps(dvec, dvec, 255),
                            };
                            *cvi = _mm256_fmadd_ps(d, q, *cvi);
                        }
                    }
                }
                for (j, cvj) in cv.iter_mut().enumerate() {
                    for (i, cvi) in cvj.iter_mut().enumerate() {
                        *c.add((jj + j) * ldc + ii + i) = sx::hsum8(*cvi);
                    }
                }
            }
        };
    }

    q0_tile_vnni!(
        #[target_feature(enable = "avx2,fma,f16c,avx512vl,avx512vnni")]
        tile_vnni
    );

    /// Row/column-tail version of the VNNI tile (runtime bounds, only the
    /// last panel/block of a non-tile-multiple m/n takes it — the C's
    /// `mnpack` reaches `gemm<3,2>` etc. the same way).
    macro_rules! q0_tile_tail_vnni {
        ($(#[$attr:meta])* $name:ident) => {
            $(#[$attr])*
            unsafe fn $name<const NCOL: usize>(
                k: usize,
                panel: *const u8,
                adelta: *const f32,
                b: *const u8,
                ldb: usize,
                c: *mut f32,
                ldc: usize,
                ii: usize,
                jj: usize,
                rme: usize,
                rne: usize,
            ) {
                let mut cv = [[_mm256_setzero_ps(); RM]; NCOL];
                for l in 0..k {
                    let mut av = [_mm256_setzero_si256(); RM];
                    for i in 0..rme {
                        av[i] = _mm256_loadu_si256(panel.add((i * k + l) * 32) as *const __m256i);
                    }
                    // rows >= rme keep delta 0 (adelta is zero-initialized), so
                    // their (discarded) accumulators stay well-defined
                    let da = _mm_loadu_ps(adelta.add(l * RM));
                    for j in 0..rne {
                        let pb = b.add((jj + j) * ldb * 34 + l * 34);
                        let bq = load_q8_0(pb);
                        let d256 =
                            _mm256_castps128_ps256(_mm_mul_ps(da, _mm_set1_ps(sx::d_f32(pb))));
                        let dvec = _mm256_permute2f128_ps(d256, d256, 0);
                        for i in 0..rme {
                            let u = _mm256_sign_epi8(av[i], av[i]);
                            let s = _mm256_sign_epi8(bq, av[i]);
                            let q = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(
                                _mm256_setzero_si256(),
                                u,
                                s,
                            ));
                            let d = match i {
                                0 => _mm256_shuffle_ps(dvec, dvec, 0),
                                1 => _mm256_shuffle_ps(dvec, dvec, 85),
                                2 => _mm256_shuffle_ps(dvec, dvec, 170),
                                _ => _mm256_shuffle_ps(dvec, dvec, 255),
                            };
                            cv[j][i] = _mm256_fmadd_ps(d, q, cv[j][i]);
                        }
                    }
                }
                for (j, cvj) in cv.iter_mut().enumerate().take(rne) {
                    for (i, cvi) in cvj.iter_mut().enumerate().take(rme) {
                        *c.add((jj + j) * ldc + ii + i) = sx::hsum8(*cvi);
                    }
                }
            }
        };
    }

    q0_tile_tail_vnni!(
        #[target_feature(enable = "avx2,fma,f16c,avx512vl,avx512vnni")]
        tile_tail_vnni
    );

    /// The C's `updot` lane on this host: `#if defined(__AVX512VNNI__) &&
    /// defined(__AVX512VL__)` (sgemm.cpp:1756) selects `_mm256_dpbusd_epi32`
    /// over the `maddubsw + maddwd` pair. Value-identical for every operand
    /// this class feeds it: `maddubs`'s int16 saturation cannot engage
    /// (largest pair magnitude is q8_0's 2·127·127 = 32258 < 32767) and the
    /// int32 pair sums stay < 2^31, so both spellings produce the same i32
    /// lanes — hence the same f32 after conversion and the same fma chain.
    #[inline]
    fn vnni_lane() -> bool {
        sx::avx512vnni() && sx::avx512vl()
    }

    /// Decode `NROW` A rows of `k` blocks into the panel + delta arrays (the
    /// C's `load(TA)` and `unhalf(block.d)`, sgemm.cpp:1684-1712 / :80).
    /// `adelta` is *interleaved*: the f32 delta of row `i`, block `l` lives at
    /// `adelta[l*RM + i]`, so a VNNI tile fetches all four rows' deltas of a
    /// block with one `_mm_loadu_ps` (the C packs the four raw fp16s into a
    /// u64 and converts them with one `_mm_cvtph_ps`, sgemm.cpp:1537-1540).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn panel_decode<const TA: u8, const NROW: usize>(
        a: *const u8,
        lda: usize,
        k: usize,
        ii: usize,
        panel: *mut u8,
        adelta: *mut f32,
    ) {
        let xsz = block_bytes(TA);
        for i in 0..NROW {
            let row = a.add((ii + i) * lda * xsz);
            for l in 0..k {
                let pa = row.add(l * xsz);
                _mm256_storeu_si256(
                    panel.add((i * k + l) * 32) as *mut __m256i,
                    load_a::<TA>(pa),
                );
                *adelta.add(l * RM + i) = sx::d_f32(pa);
            }
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn gemm_ty<const TA: u8, const NCOL: usize>(
        k: usize,
        a: *const u8,
        lda: usize,
        b: *const u8,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        // Decoded row panel: RM rows × k blocks of the C's `load(TA)` result,
        // plus the f32 A delta of each block (`unhalf(A.d)`).
        let mut panel = vec![0u8; RM * k * 32];
        let mut adelta = vec![0f32; RM * k];
        let pp = panel.as_mut_ptr();
        let ap = adelta.as_mut_ptr();

        let mut ii = i0;
        while ii < i1 {
            let rme = RM.min(i1 - ii);
            if rme == RM {
                panel_decode::<TA, RM>(a, lda, k, ii, pp, ap);
            } else {
                // row tail: same decode with a runtime row count (deltas in
                // the interleaved layout, rows >= rme stay 0)
                let xsz = block_bytes(TA);
                for i in 0..rme {
                    let row = a.add((ii + i) * lda * xsz);
                    for l in 0..k {
                        let pa = row.add(l * xsz);
                        _mm256_storeu_si256(
                            pp.add((i * k + l) * 32) as *mut __m256i,
                            load_a::<TA>(pa),
                        );
                        *ap.add(l * RM + i) = sx::d_f32(pa);
                    }
                }
            }
            let mut jj = j0;
            while jj < j1 {
                let rne = NCOL.min(j1 - jj);
                if rme == RM && rne == NCOL {
                    if vnni_lane() {
                        tile_vnni::<NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj);
                    } else if sx::avx512vl() {
                        tile_evex::<RM, NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj);
                    } else {
                        tile_avx2::<RM, NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj);
                    }
                } else if vnni_lane() {
                    tile_tail_vnni::<NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj, rme, rne);
                } else if sx::avx512vl() {
                    tile_tail_evex::<NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj, rme, rne);
                } else {
                    tile_tail_avx2::<NCOL>(k, pp, ap, b, ldb, c, ldc, ii, jj, rme, rne);
                }
                jj += rne;
            }
            ii += rme;
        }
    }

    pub unsafe fn gemm(
        op: Op,
        k: usize,
        a: *const u8,
        lda: usize,
        b: *const u8,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
    ) {
        // the host's register file picks the column tile exactly like the C's
        // `VECTOR_REGISTERS` compile-time switch (sgemm.cpp:1379/1435): the
        // 32-register builds (VNNI or plain EVEX) take the `case 0x44` tile
        // (nc = 4), the 16-register AVX2 build `nc = 2`. `gemm_ty` then picks
        // the kernel flavor per tile — `dpbusd` under VNNI+VL (sgemm.cpp:1756)
        // — so both lanes share this dispatch.
        if sx::avx512vl() {
            match op {
                Op::Q8xQ8 => gemm_ty::<0, RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                Op::Q4xQ8 => gemm_ty::<1, RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                Op::Q5xQ8 => gemm_ty::<2, RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                _ => unreachable!("tinyblas Q0 gemm: {op:?}"),
            }
        } else {
            match op {
                Op::Q8xQ8 => gemm_ty::<0, AVX2_RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                Op::Q4xQ8 => gemm_ty::<1, AVX2_RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                Op::Q5xQ8 => gemm_ty::<2, AVX2_RN_TILE>(k, a, lda, b, ldb, c, ldc, i0, i1, j0, j1),
                _ => unreachable!("tinyblas Q0 gemm: {op:?}"),
            }
        }
    }
}
// ======================================================================
// routing + value parity vs the reference binary
// (parity/tinyblas_ref.bin, built by parity/ref_tinyblas_dump.c, which calls
//  the *exported* `llamafile_sgemm` of the shipped libggml-cpu.so directly)
// ======================================================================

#[cfg(test)]
mod dump_tests {
    use super::*;
    use crate::types::GgmlType;

    const REF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/tinyblas_ref.bin");

    /// One `VTB1` section. A bailed case carries no payload (the reference
    /// never reads A/B), so `a`/`b`/`c` are empty there.
    struct Case<'a> {
        aty: GgmlType,
        bty: GgmlType,
        m: usize,
        n: usize,
        k: usize,
        accepted: bool,
        a: &'a [u8],
        b: &'a [u8],
        c: Vec<f32>,
    }

    fn parse(bytes: &[u8]) -> (u32, Vec<Case<'_>>) {
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 0x3052_5456, "VTR0 header");
        let flags = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let mut c = &bytes[8..];
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x3142_5456, "VTB1 section #{}", out.len());
            let u = |i: usize| u32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap());
            let (aty, bty, m, n, k, ret) = (
                GgmlType::from_u32(u(0)).unwrap(),
                GgmlType::from_u32(u(1)).unwrap(),
                u(2) as usize,
                u(3) as usize,
                u(4) as usize,
                u(5) != 0,
            );
            c = &c[24..]; // 6 header fields after the magic
            let (a, b, cc) = if ret {
                assert!(
                    aty.row_size(m * k * aty.blck_size()) + bty.row_size(n * k * aty.blck_size())
                        + m * n * 4
                        <= c.len(),
                    "VTB1 section #{i}: payload overruns (a {aty:?} b {bty:?} m {m} n {n} k {k})",
                    i = out.len()
                );
                let (a, r) = c.split_at(aty.row_size(m * k * aty.blck_size()).min(c.len()));
                let (b, r) = r.split_at(bty.row_size(n * k * aty.blck_size()).min(r.len()));
                let (cv, r) = r.split_at(m * n * 4);
                c = r;
                let cv = cv
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                (a, b, cv)
            } else {
                (&[][..], &[][..], Vec::new())
            };
            out.push(Case { aty, bty, m, n, k, accepted: ret, a, b, c: cc });
        }
        (flags, out)
    }

    /// The A/B payload layout the dumper uses: `k` is in `Atype` blocks, both
    /// rows carry `k * blck(Atype)` values, `lda = ldb = k` units.
    fn row_bytes(rows: usize, ty: GgmlType, k: usize, aty: GgmlType) -> usize {
        rows * ty.row_size(k * aty.blck_size())
    }

    /// Bit-exactness of the ported kernels and of every bail condition, against
    /// the shipped reference's own `llamafile_sgemm` (parity/tinyblas_ref.bin).
    ///
    /// The artifact's build word is asserted first: it records the macros the
    /// reference was compiled with (`-march=native`), which are what select the
    /// `tinyBLAS<16,__m512>` / `tinyBLAS<32,__m512bh>` / `tinyBLAS_Q0_AVX`
    /// instantiations and the `KN` values this port mirrors. On a host that
    /// does not match them the port deliberately keeps the vec_dot fallback,
    /// so the test skips.
    #[test]
    fn tinyblas_routing_and_values_vs_reference() {
        let bytes = std::fs::read(REF)
            .unwrap_or_else(|_| panic!("missing {REF}: build via parity/ref_tinyblas_dump.c"));
        let (flags, cases) = parse(&bytes);
        let bit = |i: u32| flags >> i & 1 != 0;
        println!(
            "reference build: AVX2={} AVX512F={} AVX512DQ={} AVX512BW={} AVX512VNNI={} \
             AVX512VL={} AVX512BF16={} F16C={} FMA={}",
            bit(0),
            bit(1),
            bit(2),
            bit(3),
            bit(4),
            bit(5),
            bit(6),
            bit(7),
            bit(8)
        );
        // the KN=32 bf16 variant and the AVX512 float kernels are only the
        // reference's choice when those macros are on
        assert_eq!(avx512(), bit(1) && bit(2), "AVX512F/DQ mismatch with the artifact");
        assert_eq!(avx512bf16(), bit(6), "AVX512BF16 mismatch with the artifact");
        if !avx512() {
            eprintln!("SKIP: host has no AVX512 — the port keeps the vec_dot path by design");
            return;
        }

        let mut failures = Vec::new();
        let mut accepted = 0usize;
        let mut exact = 0usize;
        let mut routed = 0usize;
        let mut iq4nl = 0usize;
        for s in &cases {
            // The one sgemm case the port does not implement: `case
            // GGML_TYPE_IQ4_NL` (sgemm.cpp:4131) accepts B == Q8_0, but IQ4_NL
            // has no quantizer/`vec_dot_type`/`vec_dot` in this port, so it can
            // never appear as a weight here. Verified as the *only* difference.
            if s.aty == GgmlType::Iq4Nl {
                assert_eq!(
                    (s.bty == GgmlType::Q8_0),
                    s.accepted,
                    "IQ4_NL x {:?}: the reference accepts exactly IQ4_NL x Q8_0",
                    s.bty
                );
                assert!(!accepts(s.aty, s.bty, s.m, s.n, s.k));
                iq4nl += 1;
                continue;
            }
            let want = accepts(s.aty, s.bty, s.m, s.n, s.k);
            if want != s.accepted {
                failures.push(format!(
                    "{:?}x{:?} m={} n={} k={}: routing mine {} reference {}",
                    s.aty, s.bty, s.m, s.n, s.k, want, s.accepted
                ));
                continue;
            }
            routed += 1;
            if !s.accepted {
                continue;
            }
            accepted += 1;
            let op = resolve(s.aty, s.bty).unwrap();
            assert_eq!(s.a.len(), row_bytes(s.m, s.aty, s.k, s.aty), "A payload size");
            assert_eq!(s.b.len(), row_bytes(s.n, s.bty, s.k, s.aty), "B payload size");
            assert_eq!(s.c.len(), s.m * s.n, "C payload size");
            let mut mine = vec![0f32; s.m * s.n];
            // SAFETY: the payload slices are exactly the sizes the kernel reads
            // (m/n rows of k units, lda = ldb = k) and `accepts` just held.
            unsafe {
                gemm(
                    op,
                    s.m,
                    s.n,
                    s.k,
                    s.a.as_ptr(),
                    s.k,
                    s.b.as_ptr(),
                    s.k,
                    mine.as_mut_ptr(),
                    s.m,
                    0,
                    s.m,
                    0,
                    s.n,
                );
            }
            if mine.iter().zip(&s.c).any(|(x, y)| x.to_bits() != y.to_bits()) {
                let (i, x, y) = mine
                    .iter()
                    .zip(&s.c)
                    .enumerate()
                    .find(|(_, (x, y))| x.to_bits() != y.to_bits())
                    .map(|(i, (x, y))| (i, *x, *y))
                    .unwrap();
                let worst = mine
                    .iter()
                    .zip(&s.c)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0f32, f32::max);
                failures.push(format!(
                    "{:?}x{:?} m={} n={} k={}: element {i} {x} != {} (worst rel {worst:.2e})",
                    s.aty, s.bty, s.m, s.n, s.k, y
                ));
            } else {
                exact += 1;
            }
            // the row-panel split must not move a single bit
            let mut split = vec![0f32; s.m * s.n];
            let rm = RM.max(1);
            for i0 in (0..s.m).step_by(rm) {
                let i1 = (i0 + rm).min(s.m);
                for j0 in (0..s.n).step_by(RN) {
                    let j1 = (j0 + RN).min(s.n);
                    unsafe {
                        gemm(
                            op,
                            s.m,
                            s.n,
                            s.k,
                            s.a.as_ptr(),
                            s.k,
                            s.b.as_ptr(),
                            s.k,
                            split.as_mut_ptr(),
                            s.m,
                            i0,
                            i1,
                            j0,
                            j1,
                        );
                    }
                }
            }
            if split.iter().zip(&mine).any(|(x, y)| x.to_bits() != y.to_bits()) {
                failures.push(format!(
                    "{:?}x{:?} m={} n={} k={}: tiled run != full run",
                    s.aty, s.bty, s.m, s.n, s.k
                ));
            }
        }
        assert!(failures.is_empty(), "tinyblas parity failures:\n{}", failures.join("\n"));
        println!(
            "routing: {routed} cases match, values: {exact}/{accepted} bit-exact, \
             IQ4_NL cases: {iq4nl} (unimplemented by design, verdict matches)"
        );
        assert!(accepted > 300, "artifact looks stale ({accepted} accepted cases)");
        assert!(routed > 3000, "artifact looks stale ({routed} routing cases)");
    }

    /// The switch/rejection table of `llamafile_sgemm` that the artifact's
    /// type pairs cannot all carry (weight types without a quantizer in the
    /// port, B types without a `from_float`) — transcribed from
    /// sgemm.cpp:3827-4149.
    #[test]
    fn accepts_table_matches_the_reference_switch() {
        use GgmlType::*;
        let table: &[(GgmlType, GgmlType, bool)] = &[
            (F32, F32, true),
            (F32, F16, false),
            (F16, F16, true),
            (F16, F32, false),
            (Bf16, Bf16, true),
            (Q8_0, Q8_0, true),
            (Q8_0, F32, false),
            (Q4_0, Q8_0, true),
            (Q4_0, F32, false),
            (Q5_0, Q8_0, true),
            (Q4_1, Q8_1, false),
            (Q5_1, Q8_1, false),
            (Q2K, Q8K, false),
            (Q3K, Q8K, false),
            (Q4K, Q8K, false),
            (Q5K, Q8K, false),
            (Q6K, Q8K, false),
            (Iq4Nl, Q8_0, false),
            (Mxfp4, Q8_0, false),
        ];
        for &(a, b, want) in table {
            assert_eq!(resolve(a, b).is_some(), want, "resolve({a:?}, {b:?})");
            assert_eq!(
                accepts(a, b, 64, 8, 32 / a.blck_size().max(1) * 8 / a.blck_size().max(1)),
                want && avx512(),
                "accepts({a:?}, {b:?})"
            );
        }
        // the numeric gates: n >= 2, m % 4 (sgemm.cpp:3820, :568-612). The
        // k % KN bail is x86-only removed since #29806 (sgemm.cpp:569-572):
        // remainders go through gemm_bloc's masked-load tail (:653-668), so
        // k % 16 / k % 32 != 0 shapes are *accepted* now — the dump test's
        // regenerated artifact carries their values.
        assert!(!accepts(F32, F32, 64, 1, 64), "n < 2 must bail");
        assert!(!accepts(F32, F32, 62, 8, 64), "m % 4 != 0 must bail");
        assert!(accepts(F32, F32, 64, 8, 60), "k % 16 != 0: K tail (accepted)");
        assert!(accepts(F16, F16, 64, 8, 60), "k % 16 != 0: K tail (f16, accepted)");
        assert!(accepts(F32, F32, 64, 8, 1), "k < KN: tail-only, accepted");
        // __AVX512BF16__ host: tinyBLAS<32,__m512bh> => K tail rem = k % 32
        assert!(accepts(Bf16, Bf16, 64, 8, 16), "bf16 KN=32: k % 32 != 0 is a K tail now");
        assert!(accepts(Bf16, Bf16, 64, 8, 33));
        assert!(accepts(Bf16, Bf16, 64, 8, 32));
        // the Q0 class has no k/m gate at all
        assert!(accepts(Q8_0, Q8_0, 1, 2, 1));
        assert!(accepts(Q5_0, Q8_0, 3, 2, 3));
        assert!(!accepts(Q5_0, Q8_0, 3, 1, 3), "n < 2 must bail (q5_0)");
        assert_eq!(Op::Bf16.kn(), 32);
        assert_eq!(Op::F32.kn(), 16);
        assert_eq!(Op::Q5xQ8.kn(), 32);
    }

/// The gpt-oss MoE router GEMM — the one real-model shape the per-type dump
/// (`parity/tinyblas_dump_ref.bin`, k <= 1024) does not cover. `ffn_gate_inp`
/// is F32 [2880 x 32] and `cur` [2880 x T], so the reference's first
/// `llamafile_sgemm` attempt accepts it and its production path is
/// `tinyBLAS<16,__m512>`.
///
/// Ground truth: `parity/ref_tinyblas_router.c` calls `llamafile_sgemm`
/// directly on an LCG-filled A/B with a 1-thread pool (the only way to drive it
/// outside a graph dispatch — see that file's header; at nth>1 with an
/// undriven pool the C's chunk protocol runs job 0 only) and dumps C.
#[test]
fn f32_router_shape_matches_reference() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/");
    let Ok(ab) = std::fs::read(format!("{base}tinyblas_router_ab.bin")) else {
        eprintln!("skipping: build it via parity/ref_tinyblas_router.c");
        return;
    };
    let Ok(reference) = std::fs::read(format!("{base}tinyblas_router_ref.bin")) else {
        eprintln!("skipping: build it via parity/ref_tinyblas_router.c");
        return;
    };
    const M: usize = 32;
    const N: usize = 5;
    const K: usize = 2880;
    let a: &[f32] = bytemuck::cast_slice(&ab[..M * K * 4]);
    let b: &[f32] = bytemuck::cast_slice(&ab[M * K * 4..]);
    assert_eq!(a.len(), M * K);
    assert_eq!(b.len(), N * K);
    let want: &[f32] = bytemuck::cast_slice(&reference[4..4 + M * N * 4]);

    let mut c = vec![0f32; M * N];
    assert!(accepts(GgmlType::F32, GgmlType::F32, M, N, K), "router shape must accept");
    // SAFETY: both buffers are exactly the sizes the kernel reads/writes for
    // this (m, n, k), and the host feature check below gates the AVX512 body.
    unsafe {
        gemm(
            Op::F32,
            M,
            N,
            K,
            a.as_ptr() as *const u8,
            K,
            b.as_ptr() as *const u8,
            K,
            c.as_mut_ptr(),
            M,
            0,
            M,
            0,
            N,
        );
    }
    let bad = c.iter().zip(want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(bad, 0, "router-shaped F32 tinyBLAS: {bad}/{} elements differ", M * N);
}
}
// ======================================================================
// dispatch wiring: does `compute::forward_mul_mat` route where the reference
// does? (parity/tinyblas_ref.bin pins the predicate and the kernels; this pins
// the call site, including the `src1_cont` / `vec_dot_type` conditions the
// per-type dumps cannot see)
// ======================================================================

#[cfg(test)]
mod wire_tests {
    use super::*;
    use crate::compute::graph_compute;
    use crate::graph::Graph;
    use crate::tensor::Context;
    use crate::types::GgmlType;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as i32 as f32 / (1u32 << 28) as f32) * 1.8 - 0.9
            })
            .collect()
    }

    fn quantize(ty: GgmlType, xf: &[f32], out: &mut [u8]) {
        use crate::quants::*;
        use crate::quants_k::*;
        match ty {
            GgmlType::Q4_0 => quantize_row_q4_0_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q5_0 => quantize_row_q5_0_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q8_0 => quantize_row_q8_0(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q4K => quantize_row_q4_K_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q5K => quantize_row_q5_K_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q6K => quantize_row_q6_K_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::F32 => out.copy_from_slice(bytemuck::cast_slice(xf)),
            GgmlType::F16 => {
                let o: &mut [half::f16] = bytemuck::cast_slice_mut(out);
                for (d, s) in o.iter_mut().zip(xf) {
                    *d = half::f16::from_f32(*s);
                }
            }
            GgmlType::Bf16 => {
                let o: &mut [half::bf16] = bytemuck::cast_slice_mut(out);
                for (d, s) in o.iter_mut().zip(xf) {
                    *d = half::bf16::from_f32(*s);
                }
            }
            other => unimplemented!("quantize {other:?}"),
        }
    }

    /// The exact weight byte layout the graph handed to the kernels (the same
    /// quantizer `run` used).
    fn ctx_bytes(ty: GgmlType, wf: &[f32], rows: usize) -> Vec<u8> {
        let mut out = vec![0u8; ty.row_size(wf.len() / rows.max(1)) * rows];
        quantize(ty, wf, &mut out);
        out
    }

    /// One graph `mul_mat([n, rows], [n, cols])` through the real dispatch.
    fn run(ty: GgmlType, n: usize, rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(ty, n as i64, rows as i64);
        let b = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
        let d = ctx.mul_mat(a, b);
        for t in [a, b] {
            ctx.arena_resize_tensor(t);
        }
        let wf = lcg(n * rows, 11);
        let af = lcg(n * cols, 22);
        quantize(ty, &wf, ctx.data_bytes_mut(a).unwrap());
        ctx.data_bytes_mut(b)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&af));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        graph_compute(&mut ctx, &mut g, 4);
        let out = bytemuck::cast_slice(ctx.data_bytes(d).unwrap()).to_vec();
        (wf, af, out)
    }

    /// The gate reads the *per-plane* `ne11` (ggml-cpu.c:1307/:1390), not the
    /// flattened column count: a 3D `mul_mat` whose `ne11 == 1` must stay on the
    /// `vec_dot` fallback for every plane even though it has many columns
    /// overall. This is the shape the old `ne11*ne12*ne13 >= 2` wiring got
    /// wrong (it also lacked the `k % KN` / `m % 4` gates).
    #[test]
    fn ne11_one_bails_on_every_plane() {
        for (ty, ne11, planes, want) in [
            (GgmlType::Q5_0, 1usize, 4usize, None),               // n=1 per plane
            (GgmlType::Q5_0, 2, 4, Some(Op::Q5xQ8)),              // n=2 per plane
            (GgmlType::F16, 1, 3, None),                          // attention-shaped
            (GgmlType::F16, 3, 3, Some(Op::F16)),
        ] {
            let n = 128usize;
            let rows = 16usize;
            let mut ctx = Context::new();
            let a = ctx.new_tensor_2d(ty, n as i64, rows as i64);
            let b = ctx.new_tensor_3d(GgmlType::F32, n as i64, ne11 as i64, planes as i64);
            let d = ctx.mul_mat(a, b);
            for t in [a, b] {
                ctx.arena_resize_tensor(t);
            }
            let wf = lcg(n * rows, 5);
            quantize(ty, &wf, ctx.data_bytes_mut(a).unwrap());
            let af = lcg(n * ne11 * planes, 6);
            ctx.data_bytes_mut(b)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&af));
            let mut g = Graph::new(8);
            g.build_forward(&ctx, d);
            let before = calls();
            graph_compute(&mut ctx, &mut g, 4);
            let after = calls();
            let delta: Vec<usize> = (0..6).map(|i| after[i] - before[i]).collect();
            match want {
                Some(op) => assert!(delta[op.idx()] >= 1, "{ty:?} ne11={ne11} planes={planes}"),
                None => assert_eq!(delta, [0; 6], "{ty:?} ne11={ne11} planes={planes}"),
            }
            assert_eq!(ctx.data_bytes(d).unwrap().len(), rows * ne11 * planes * 4);
        }
    }

    /// The routing table of `compute.rs`'s two `llamafile_sgemm` attempts, per
    /// (weight type, per-slice n, m, k). `None` = the reference's
    /// `vec_dot` fallback, which the counter must confirm too.
    #[test]
    fn mul_mat_routes_like_llamafile_sgemm() {
        let cases: &[(GgmlType, usize, usize, usize, Option<Op>)] = &[
            // (ty, n=k, rows=m, cols per slice, expected Op)
            (GgmlType::Q5_0, 256, 32, 4, Some(Op::Q5xQ8)),
            (GgmlType::Q5_0, 256, 32, 1, None), // n < 2
            (GgmlType::Q5_0, 256, 30, 4, Some(Op::Q5xQ8)), // q5_0 has no m gate
            // Q4_0 with ne[1] % 8 == 0 is intercepted by the CPU_REPACK traits
            // *before* the op switch (ggml-cpu.c:1751-1753), so no llamafile Op
            // may serve it (repack.rs's q4_0 section handles the whole mul_mat)
            (GgmlType::Q4_0, 256, 32, 3, None),
            // ne[1] % 8 != 0 keeps the reference's plain (non-repack) tensor
            // routing: llamafile Q4xQ8 for n >= 2, vec_dot otherwise
            (GgmlType::Q4_0, 256, 30, 3, Some(Op::Q4xQ8)),
            (GgmlType::Q4_0, 256, 30, 1, None),
            (GgmlType::Q8_0, 256, 32, 4, Some(Op::Q8xQ8)),
            (GgmlType::Q4K, 256, 32, 4, None), // no sgemm case at all
            (GgmlType::Q5K, 256, 32, 4, None),
            (GgmlType::Q6K, 512, 32, 4, None),
            (GgmlType::F32, 64, 64, 8, Some(Op::F32)),
            (GgmlType::F32, 60, 64, 8, Some(Op::F32)), // k % 16 != 0: K tail
            (GgmlType::F32, 64, 62, 8, None),  // m % 4 != 0
            (GgmlType::F32, 64, 64, 1, None),  // n < 2
            (GgmlType::F16, 64, 64, 8, Some(Op::F16)),
            (GgmlType::F16, 64, 64, 1, None),
            (GgmlType::Bf16, 64, 64, 16, Some(Op::Bf16)),
            (GgmlType::Bf16, 48, 64, 8, Some(Op::Bf16)), // K tail: rem = 48 % 32 = 16
        ];
        for &(ty, n, rows, cols, want) in cases {
            let before = calls();
            let (wf, af, got) = run(ty, n, rows, cols);
            let after = calls();
            let mut delta = [0usize; 6];
            for i in 0..6 {
                delta[i] = after[i] - before[i];
            }
            let served: Vec<usize> = (0..6).filter(|&i| delta[i] != 0).collect();
            match want {
                Some(op) => assert_eq!(
                    served,
                    vec![op.idx()],
                    "{ty:?} n={n} m={rows} n_cols={cols}: expected {op:?} to serve, delta {delta:?}"
                ),
                None => assert_eq!(
                    delta,
                    [0; 6],
                    "{ty:?} n={n} m={rows} n_cols={cols}: the reference bails, no Op may run"
                ),
            }
            // value sanity per case, against the kernel the reference uses on
            // the *other* side of the gate:
            //   * Q4_0/Q5_0/Q8_0: the row-wise `vec_dot` on the same quantized
            //     bytes — the port's two routes agree bit-for-bit for the Q0
            //     class (mulmat_ref.bin / kquant_real_tensor_tests pin both),
            //     so a wrong ld/offset in the wiring shows up here;
            //   * F32/F16/BF16: the naive f64 dot over the values the tinyBLAS
            //     instantiation actually sees.
            if let Some(op) = want {
                let wk: Vec<f32> = match ty {
                    GgmlType::F16 => wf.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect(),
                    GgmlType::Bf16 => wf.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect(),
                    _ => wf.clone(),
                };
                let ak: Vec<f32> = match ty {
                    GgmlType::F16 => af.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect(),
                    GgmlType::Bf16 => af.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect(),
                    _ => af.clone(),
                };
                let q8 = |c: usize| -> Vec<u8> {
                    let mut out = vec![0u8; GgmlType::Q8_0.row_size(n)];
                    crate::quants::quantize_row_q8_0(
                        &af[c * n..(c + 1) * n],
                        bytemuck::cast_slice_mut(&mut out),
                    );
                    out
                };
                let wbytes = ctx_bytes(ty, &wf, rows);
                for c in 0..cols {
                    for r in 0..rows {
                        let g = got[c * rows + r];
                        match op {
                            Op::Q8xQ8 | Op::Q4xQ8 | Op::Q5xQ8 => {
                                // the port's row-wise kernel (bit-identical to
                                // the reference's tinyBLAS for the Q0 class)
                                let want_v = crate::vec_dot::vec_dot_row(
                                    ty,
                                    n,
                                    &wbytes[r * ty.row_size(n)..],
                                    &q8(c),
                                );
                                assert_eq!(
                                    g.to_bits(),
                                    want_v.to_bits(),
                                    "{ty:?}: r={r} c={c} gemm {g} vs vec_dot {want_v}"
                                );
                            }
                            _ => {
                                let mut refv = 0f64;
                                for j in 0..n {
                                    refv += wk[r * n + j] as f64 * ak[c * n + j] as f64;
                                }
                                let rel = (g as f64 - refv).abs() / refv.abs().max(1.0);
                                assert!(
                                    rel < 1e-4,
                                    "{ty:?}: r={r} c={c} got={g} naive={refv} rel={rel:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Ad-hoc throughput probe for the real prefill shapes (run with
/// `cargo test --release -p ggml bench_tinyblas_shapes -- --ignored --nocapture`).
/// Not a correctness test: it exists to compare the ported kernels' cost against
/// the reference's own GEMM throughput (~1060 t/s prefill on qwen2.5-0.5b).
#[cfg(test)]
mod bench_tests {
    use super::*;
    use crate::types::GgmlType;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as i32 as f32 / (1u32 << 28) as f32) * 1.8 - 0.9
            })
            .collect()
    }

    #[test]
    #[ignore = "manual: tinyBLAS GEMM throughput probe"]
    fn bench_tinyblas_shapes() {
        // (weight type, k, rows m, cols n) — qwen2.5-0.5b Q4_K_M prefill shapes
        // plus K-tail pairs since #29806 (895/897 = 896±1: one masked tail vs
        // the clean multiple — K tails exist for speed, so the tail lane must
        // not cost more than the vec_dot fallback it replaced would have)
        let cases: &[(GgmlType, usize, usize, usize)] = &[
            (GgmlType::Q5_0, 896, 896, 64),    // attn_output
            (GgmlType::Q8_0, 896, 128, 64),    // attn_v
            (GgmlType::Q5_0, 896, 4864, 64),   // ffn_gate
            (GgmlType::Q8_0, 896, 4864, 64),   // ffn_up (if Q8_0)
            (GgmlType::Q5_0, 896, 4864, 5),    // short prompt
            (GgmlType::F32, 64, 64, 64),       // attention kqv (T=64)
            (GgmlType::F32, 896, 4864, 64),    // f32 panel, k%16==0
            (GgmlType::F32, 895, 4864, 64),    // K tail (rem 15)
            (GgmlType::F32, 897, 4864, 64),    // K tail (rem 1)
            (GgmlType::F16, 896, 4864, 64),    // f16 panel, k%16==0
            (GgmlType::F16, 895, 4864, 64),    // f16 K tail
            (GgmlType::Bf16, 896, 4864, 64),   // bf16 KN=32: 896%32==0
            (GgmlType::Bf16, 895, 4864, 64),   // bf16 K tail (rem 31)
        ];
        for &(ty, k, m, n) in cases {
            let wf = lcg(k * m, 11);
            let af = lcg(k * n, 22);
            let mut w = vec![0u8; ty.row_size(k * m)];
            let bty = if ty == GgmlType::F32 { GgmlType::F32 } else { GgmlType::Q8_0 };
            let mut b = vec![0u8; bty.row_size(k * n)];
            match ty {
                GgmlType::Q5_0 => {
                    crate::quants::quantize_row_q5_0_ref(&wf, bytemuck::cast_slice_mut(&mut w))
                }
                GgmlType::Q8_0 => {
                    crate::quants::quantize_row_q8_0(&wf, bytemuck::cast_slice_mut(&mut w))
                }
                GgmlType::F32 => w.copy_from_slice(bytemuck::cast_slice(&wf)),
                GgmlType::F16 => {
                    let o: &mut [half::f16] = bytemuck::cast_slice_mut(&mut w);
                    for (d, s) in o.iter_mut().zip(&wf) {
                        *d = half::f16::from_f32(*s);
                    }
                    // f16/f32 share Btype = self
                    b.clear();
                    b.resize(k * n * 2, 0);
                    let ob: &mut [half::f16] = bytemuck::cast_slice_mut(&mut b);
                    for (d, s) in ob.iter_mut().zip(&af) {
                        *d = half::f16::from_f32(*s);
                    }
                }
                GgmlType::Bf16 => {
                    let o: &mut [half::bf16] = bytemuck::cast_slice_mut(&mut w);
                    for (d, s) in o.iter_mut().zip(&wf) {
                        *d = half::bf16::from_f32(*s);
                    }
                    b.clear();
                    b.resize(k * n * 2, 0);
                    let ob: &mut [half::bf16] = bytemuck::cast_slice_mut(&mut b);
                    for (d, s) in ob.iter_mut().zip(&af) {
                        *d = half::bf16::from_f32(*s);
                    }
                }
                other => unimplemented!("{other:?}"),
            }
            let bty = match ty {
                GgmlType::F32 => GgmlType::F32,
                GgmlType::F16 => GgmlType::F16,
                GgmlType::Bf16 => GgmlType::Bf16,
                _ => GgmlType::Q8_0,
            };
            match ty {
                GgmlType::F32 => b.copy_from_slice(bytemuck::cast_slice(&af)),
                GgmlType::Q5_0 | GgmlType::Q8_0 => {
                    crate::quants::quantize_row_q8_0(&af, bytemuck::cast_slice_mut(&mut b))
                }
                _ => {}
            }
            let Some(op) = resolve(ty, bty) else { continue };
            let kb = k / ty.blck_size();
            let mut c = vec![0f32; m * n];
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                unsafe {
                    gemm(
                        op,
                        m,
                        n,
                        kb,
                        w.as_ptr(),
                        kb,
                        b.as_ptr(),
                        kb,
                        c.as_mut_ptr(),
                        m,
                        0,
                        m,
                        0,
                        n,
                    )
                };
                best = best.min(t.elapsed().as_secs_f64());
            }
            let macs = (m * n * k) as f64;
            // and with the row panels split over 8 threads (the real mul_mat
            // partitioning), to separate kernel cost from parallel scaling
            let mut best8 = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                std::thread::scope(|sc| {
                    for th in 0..8 {
                        let (w, b, c) = (&w, &b, &c);
                        let lo = m / 8 * th;
                        let hi = if th == 7 { m } else { m / 8 * (th + 1) };
                        sc.spawn(move || unsafe {
                            gemm(
                                op,
                                m,
                                n,
                                kb,
                                w.as_ptr(),
                                kb,
                                b.as_ptr(),
                                kb,
                                c.as_ptr() as *mut f32,
                                m,
                                lo,
                                hi,
                                0,
                                n,
                            )
                        });
                    }
                });
                best8 = best8.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{ty:?} k={k} m={m} n={n}: 1t {:.2} ms {:.1} GMAC/s | 8t {:.2} ms {:.1} GMAC/s",
                best * 1e3,
                macs / best / 1e9,
                best8 * 1e3,
                macs / best8 / 1e9
            );
        }
    }
}
