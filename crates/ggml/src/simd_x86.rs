//! simd_x86.rs — AVX2/AVX512 implementations of the hot ggml CPU kernels.
//!
//! Every kernel here is the SIMD body the reference build actually runs for
//! that type: the reference is compiled with `-march=native` but *without*
//! `GGML_AVX512`/`GGML_AVX512_VNNI` (build-rust-ref/CMakeCache.txt), so
//! `arch/x86/quants.c` selects its `#if defined(__AVX2__)` bodies for the
//! quantized types — `-march=native` still defines `__AVX512F__`/`__AVX512DQ__`
//! on this host, so `vec.cpp`/`vec.h` select their 16-lane f32 bodies for the
//! elementwise/vector ops (silu, softmax, FA's `simd_gemm`), ported in the
//! AVX512 section below. Reproducing those bodies instruction-for-instruction is what keeps
//! the port bit-exact: same lane assignment, same integer ops, same
//! `vfmadd*` per block. The scalar kernels in `vec_dot.rs` are the verified
//! spec (they already mirror the same lane structure) and are kept as the
//! fallback for non-AVX2 hosts; `vec_dot.rs`'s tests assert SIMD == scalar
//! bit-for-bit on random data, and the reference dumps (`parity/*.bin`) assert
//! scalar == reference, which transitively pins the SIMD path.
//!
//! **Kernels must carry `#[target_feature(enable = "avx2,fma,f16c")]` on the
//! function that contains the intrinsics.** `core::arch` intrinsics are
//! themselves `#[target_feature]` functions, so in a caller without the feature
//! they cannot be inlined: the compiler emits a real call per intrinsic and
//! passes the vectors through the stack (measured: 7 ns per `vpaddb`, i.e. the
//! q5_0 kernel ran only 1.4x faster than scalar). Each public kernel here is a
//! safe one-line wrapper around an `*_avx2` body that carries the attribute.
//!
//! Two deliberate, value-identical substitutions (both noted at their site):
//! * `_mm256_dpbusd_epi32` (the reference's `__AVX512VNNI__ && __AVX512VL__`
//!   shortcut in `mul_sum_us8_pairs_float`) is not reachable from stable Rust;
//!   the `maddubs + madd` form is used instead. It is exact whenever the
//!   pairwise int16 sum cannot saturate, which the value range checks below
//!   establish for every call site.
//! * `get_scale_shuffle*` pointer arithmetic over `__m128i`/`__m256i` is
//!   reproduced by building the same broadcast pattern directly.
#![allow(non_snake_case)] // keep the C kernel names (vec_dot.rs convention)

use half::f16;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Host has AVX2 + FMA (cached; the dispatch gate for every kernel below).
pub fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *HAS.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Host has AVX512F + AVX512BW + AVX512DQ — the `__AVX512BW__ &&
/// __AVX512DQ__` gate of the x86 repack gemm bodies (arch/x86/repack.cpp:662,
/// the branch a `-march=native` build of an AVX512 host compiles in).
pub fn avx512bw() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *HAS.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512dq")
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Host has AVX512F + AVX512VL + AVX512BW — enough for EVEX encodings of the
/// 256-bit integer ops, the xmm16-31 register file and `vpternlogq`, i.e. the
/// instruction selection a reference build compiled with AVX512 enabled uses
/// for the plain AVX2 kernel sources (there is no AVX512 #elif for
/// `ggml_vec_dot_q6_K_q8_K`; the same AVX2 body just gets better encodings —
/// the `evex` instantiation of the port's q6_K kernel mirrors exactly that).
pub fn avx512vl() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *HAS.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512bw")
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Host has AVX512 VNNI — the `__AVX512VNNI__` gate of the repack gemm's
/// dot op (`mul_sum_i8_pairs_acc_int32x16` → `_mm512_dpbusd_epi32`,
/// arch/x86/repack.cpp:125/:140-147), what the `-march=native` reference
/// build compiles.
pub fn avx512vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *HAS.get_or_init(|| {
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni")
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Host has AVX512F + AVX512DQ — the `__AVX512F__ && __AVX512DQ__` gate of the
/// reference's f32 vector paths (`ggml_v_expf`/`ggml_v_silu`/`ggml_vec_silu_f32`/
/// `ggml_vec_soft_max_f32`, vec.h/vec.cpp). DQ is only needed for completeness
/// with the C gate; every intrinsic the kernels use is F (the `|n|` test below
/// is written as an integer AND, not `_mm512_abs_ps`).
pub fn avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *HAS
            .get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

#[cfg(target_arch = "x86_64")]
mod kernels {
    use super::*;
    use crate::blocks::QK_K;

    // ===================== shared helpers =====================

    /// `GGML_CPU_FP16_TO_FP32` on a raw LE f16 byte pair: the hardware F16C
    /// conversion (`vcvtph2ps`). This is the exact instruction the `half`
    /// crate's runtime F16C detection already lowers to on this class of
    /// host — with a per-call detection branch tree in front (two loads, a
    /// TLS test and a `bt` before every block's `d`); the reference instead
    /// pays one `ggml_table_f32_f16[]` load (table filled from
    /// `ggml_compute_fp16_to_fp32`, ggml-cpu.c:3886-3890). Identical bits for
    /// every f16 — `f16c_cvtph_matches_portable` below proves it
    /// exhaustively over all 65536 bit patterns on this host. Only called
    /// from `f16c`-gated kernels.
    #[inline]
    #[target_feature(enable = "f16c")]
    pub(crate) unsafe fn d_f32(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
            u16::from_le_bytes([*p, *p.add(1)]) as i32,
        )))
    }

    /// The f32 `d` of a `block_q8_K`.
    #[inline]
    unsafe fn q8k_d(p: *const u8) -> f32 {
        f32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)])
    }

    /// hsum_float_8 (arch/x86/quants.c:43): extract high 128, add, movehl-add,
    /// movehdup-addss.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn hsum8(x: __m256) -> f32 {
        let mut res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    /// `quantize_row_q8_0` (arch/x86/quants.c:302, `__AVX2__` body) — the
    /// activation quantizer of every Q4_0/Q5_0/Q8_0 mul_mat's wdata pass, i.e.
    /// the body the reference build actually runs. amax via the
    /// andnot/max network, `id = 127/amax` (NOT the `_ref`'s
    /// `1/(amax/127)`), `_mm256_round_ps(_MM_FROUND_TO_NEAREST_INT)` =
    /// ties-to-even, `cvtps_epi32` + the two `packs` with the
    /// `permutevar8x32(0,4,1,5,2,6,3,7)` order fix (quants.c:355-367).
    /// Bit-identical to the scalar spec in quants.rs on finite inputs (the
    /// scalar `as i8` saturates NaN lanes to 0 where the C body's
    /// `cvtps_epi32`→`packs` chain saturates to -128; activations never feed
    /// NaN into a quantizer — RMS-normed tensors are finite).
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn quantize_row_q8_0_avx2(x: *const f32, y: *mut u8, nb: usize) {
        let sign_bit = _mm256_set1_ps(-0.0f32);
        let mut x = x;
        let mut y = y;
        for _ in 0..nb {
            // Load elements into 4 AVX vectors (quants.c:309-312)
            let v0 = _mm256_loadu_ps(x);
            let v1 = _mm256_loadu_ps(x.add(8));
            let v2 = _mm256_loadu_ps(x.add(16));
            let v3 = _mm256_loadu_ps(x.add(24));
            x = x.add(32);

            // Compute max(abs(e)) for the block (quants.c:315-320)
            let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
            max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

            let mut max4 =
                _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
            max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            let max_scalar = _mm_cvtss_f32(max4);

            // Quantize these floats (quants.c:332-337)
            let d = max_scalar / 127.0f32;
            *(y as *mut u16) = f16::from_f32(d).to_bits(); // y[i].d
            let id = if max_scalar != 0.0f32 { 127.0f32 / max_scalar } else { 0.0f32 };
            let mul = _mm256_set1_ps(id);

            let v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT);
            let v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT);
            let v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT);
            let v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT);

            // Convert floats to integers, then int32→int16→int8 with the
            // in-lane pack order fixed by the permute (quants.c:347-367)
            let mut i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let mut i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);
            _mm256_storeu_si256(y.add(2) as *mut __m256i, i0); // y[i].qs
            y = y.add(34);
        }
    }

    /// Safe entry: dispatch for `quants::quantize_row_q8_0`.
    pub fn quantize_row_q8_0(x: &[f32], y: &mut [crate::blocks::BlockQ8_0]) {
        debug_assert_eq!(x.len(), y.len() * 32);
        unsafe { quantize_row_q8_0_avx2(x.as_ptr(), y.as_mut_ptr().cast(), y.len()) }
    }

    /// `mul_sum_us8_pairs_float` (quants.c:115) — unsigned x signed, pairwise
    /// into int16 then pairwise into int32, converted to f32 lanes.
    ///
    /// Exactness: `maddubs` saturates at the int16 pair sum. Every call site
    /// bounds the per-byte operand magnitudes so the pair sum stays in range:
    /// q4 (|q|<=8 x |q8|<=127 = 2032), q5 (16x127 = 4064), q8 (128x127 pairs =
    /// 32512), K-quants (|sc|<=127 and products of <=64x127). See each kernel.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// `mul_sum_i8_pairs_float` (quants.c:139).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
        let ax = _mm256_sign_epi8(x, x); // |x|
        let sy = _mm256_sign_epi8(y, x); // y carrying x's sign
        mul_sum_us8_pairs_float(ax, sy)
    }

    /// `bytes_from_nibbles_32` (quants.c:82): low nibbles of qs[0..16] into
    /// elements 0..15, high nibbles into 16..31.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn bytes_from_nibbles_32(p: *const u8) -> __m256i {
        let tmp = _mm_loadu_si128(p as *const __m128i);
        let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
        _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F))
    }

    /// `bytes_from_bits_32` (quants.c:75): element e becomes 0xFF iff bit e of
    /// the little-endian u32 at `p` is set.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn bytes_from_bits_32(p: *const u8) -> __m256i {
        let x32 = (p as *const i32).read_unaligned();
        let shuf = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0x0000_0000_0000_0000,
        );
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf);
        let bit_mask = _mm256_set1_epi64x(0x7fbf_dfeff7fb_fdfeu64 as i64);
        let bytes = _mm256_or_si256(bytes, bit_mask);
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
    }

    /// `get_scale_shuffle_k4(i)` (quants.c:240): byte mask [i*2, i*2+1]
    /// repeated over 32 bytes ⇒ the u16 lane `i` of the scale vector
    /// broadcast to all 8 u16 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_bcast_u16(scales: __m256i, i: usize) -> __m256i {
        let m = _mm256_setr_epi8(
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
            (2 * i) as i8,
            (2 * i + 1) as i8,
        );
        _mm256_shuffle_epi8(scales, m)
    }

    /// The `k_shuffle` mask table of `get_scale_shuffle(i)` (quants.c:540-555
    /// — byte j of mask i selects `scales[2i + j/8]`, so `_mm_shuffle_epi8`
    /// yields `[sc[2i] x8, sc[2i+1] x8]` and `_mm256_cvtepi8_epi16` makes i16
    /// lanes 0..7 = sc[2i], 8..15 = sc[2i+1]).
    ///
    /// The reference build keeps all 8 masks in xmm11-xmm18 registers for the
    /// whole kernel (GCC hoists the constant loads); one `vpshufb` per scale
    /// pair. A plain Rust `static` re-load folds back into constant lane
    /// knowledge once the j-loop unroll makes `i` a compile-time constant
    /// (LLVM then materializes each mask with 3-4 generic shuffles instead of
    /// one `vpshufb`), so the q6_K kernels load the 8 masks once through a
    /// `black_box`ed base pointer — opaque enough to stay real loads, cheap
    /// enough to sit in registers for the row loop.
    #[inline]
    unsafe fn q6k_scale_masks() -> [__m128i; 8] {
        static K_SHUFFLE: [u8; 128] = {
            // byte j selects scales[j/8]: mask i (bytes i*16..i*16+16) is
            // `scales[2i]` repeated 8 then `scales[2i+1]` repeated 8
            let mut t = [0u8; 128];
            let mut j = 0;
            while j < 128 {
                t[j] = (j / 8) as u8;
                j += 1;
            }
            t
        };
        let base = std::hint::black_box(K_SHUFFLE.as_ptr());
        [
            _mm_loadu_si128(base as *const __m128i),
            _mm_loadu_si128(base.add(16) as *const __m128i),
            _mm_loadu_si128(base.add(32) as *const __m128i),
            _mm_loadu_si128(base.add(48) as *const __m128i),
            _mm_loadu_si128(base.add(64) as *const __m128i),
            _mm_loadu_si128(base.add(80) as *const __m128i),
            _mm_loadu_si128(base.add(96) as *const __m128i),
            _mm_loadu_si128(base.add(112) as *const __m128i),
        ]
    }

    // ===================== q4_0 x q8_0 =====================

    /// `ggml_vec_dot_q4_0_q8_0` AVX2 body (arch/x86/quants.c:718-745).
    pub fn vec_dot_q4_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 32, 0, "vec_dot_q4_0_q8_0: n must be a multiple of 32");
        debug_assert!(
            x.len() >= n / 32 * 18 && y.len() >= n / 32 * 34,
            "vec_dot_q4_0_q8_0: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/32 * 18` / `* 34` bytes per row).
        unsafe { vec_dot_q4_0_q8_0_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q4_0_q8_0`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q4_0_q8_0_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / 32;
        // SAFETY: reached only through the wrapper above, which is only called
        // when `avx2()` holds and asserts the slice lengths every load needs.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let off = _mm256_set1_epi8(8);
            for ib in 0..nb {
                let bx = x.as_ptr().add(ib * 18);
                let by = y.as_ptr().add(ib * 34);
                let d = _mm256_set1_ps(d_f32(bx) * d_f32(by));
                let qx = _mm256_sub_epi8(bytes_from_nibbles_32(bx.add(2)), off);
                let qy = _mm256_loadu_si256(by.add(2) as *const __m256i);
                let q = mul_sum_i8_pairs_float(qx, qy);
                acc = _mm256_fmadd_ps(d, q, acc);
            }
            hsum8(acc)
        }
    }

    // ===================== q4_1 x q8_1 =====================

    /// `ggml_vec_dot_q4_1_q8_1` AVX2 body (arch/x86/quants.c:875-917).
    pub fn vec_dot_q4_1_q8_1(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 32, 0, "vec_dot_q4_1_q8_1: n must be a multiple of 32");
        debug_assert!(
            x.len() >= n / 32 * 20 && y.len() >= n / 32 * 36,
            "vec_dot_q4_1_q8_1: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/32 * 20` / `* 36` bytes per row).
        unsafe { vec_dot_q4_1_q8_1_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q4_1_q8_1`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q4_1_q8_1_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / 32;
        // SAFETY: as in `vec_dot_q4_0_q8_0_avx2`.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut summs = 0f32;
            for ib in 0..nb {
                let bx = x.as_ptr().add(ib * 20);
                let by = y.as_ptr().add(ib * 36);
                let d0 = d_f32(bx);
                let d1 = d_f32(by);
                // `summs += x.m*y.s` (contracted to fma in the reference build)
                summs = d_f32(bx.add(2)).mul_add(d_f32(by.add(2)), summs);
                let d0d1 = _mm256_mul_ps(_mm256_set1_ps(d0), _mm256_set1_ps(d1));
                let qx = bytes_from_nibbles_32(bx.add(4));
                let qy = _mm256_loadu_si256(by.add(4) as *const __m256i);
                let q = mul_sum_us8_pairs_float(qx, qy);
                acc = _mm256_fmadd_ps(d0d1, q, acc);
            }
            hsum8(acc) + summs
        }
    }

    // ===================== q5_0 x q8_0 =====================

    /// `ggml_vec_dot_q5_0_q8_0` AVX2 body (arch/x86/quants.c:1163-1190): the
    /// 5th bit is OR'd in as `0xF0` (= -16 in int8) where the bit is *clear*,
    /// which folds the -16 bias into the signed byte, then int8 x int8.
    pub fn vec_dot_q5_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 32, 0, "vec_dot_q5_0_q8_0: n must be a multiple of 32");
        debug_assert!(
            x.len() >= n / 32 * 22 && y.len() >= n / 32 * 34,
            "vec_dot_q5_0_q8_0: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/32 * 22` / `* 34` bytes per row).
        unsafe { vec_dot_q5_0_q8_0_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q5_0_q8_0`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q5_0_q8_0_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / 32;
        // SAFETY: see `vec_dot_q4_0_q8_0`; |qx| <= 16 and |qy| <= 127, so the
        // maddubs pairs stay far inside int16.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for ib in 0..nb {
                let bx = x.as_ptr().add(ib * 22);
                let by = y.as_ptr().add(ib * 34);
                let d = _mm256_set1_ps(d_f32(bx) * d_f32(by));
                let qx = bytes_from_nibbles_32(bx.add(6));
                let bxhi = _mm256_andnot_si256(
                    bytes_from_bits_32(bx.add(2)),
                    _mm256_set1_epi8(-16), // 0xF0
                );
                let qx = _mm256_or_si256(qx, bxhi);
                let qy = _mm256_loadu_si256(by.add(2) as *const __m256i);
                let q = mul_sum_i8_pairs_float(qx, qy);
                acc = _mm256_fmadd_ps(d, q, acc);
            }
            hsum8(acc)
        }
    }

    // ===================== q5_1 x q8_1 =====================

    /// `ggml_vec_dot_q5_1_q8_1` AVX2 body (arch/x86/quants.c:1239-1268).
    pub fn vec_dot_q5_1_q8_1(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 32, 0, "vec_dot_q5_1_q8_1: n must be a multiple of 32");
        debug_assert!(
            x.len() >= n / 32 * 24 && y.len() >= n / 32 * 36,
            "vec_dot_q5_1_q8_1: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/32 * 24` / `* 36` bytes per row).
        unsafe { vec_dot_q5_1_q8_1_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q5_1_q8_1`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q5_1_q8_1_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / 32;
        // SAFETY: see `vec_dot_q4_0_q8_0`; unsigned 5-bit values <= 31.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut summs = 0f32;
            for ib in 0..nb {
                let bx = x.as_ptr().add(ib * 24);
                let by = y.as_ptr().add(ib * 36);
                let dx = _mm256_set1_ps(d_f32(bx));
                summs = d_f32(bx.add(2)).mul_add(d_f32(by.add(2)), summs);
                let qx = bytes_from_nibbles_32(bx.add(8));
                let bxhi = _mm256_and_si256(bytes_from_bits_32(bx.add(4)), _mm256_set1_epi8(0x10));
                let qx = _mm256_or_si256(qx, bxhi);
                let dy = _mm256_set1_ps(d_f32(by));
                let qy = _mm256_loadu_si256(by.add(4) as *const __m256i);
                let q = mul_sum_us8_pairs_float(qx, qy);
                acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
            }
            hsum8(acc) + summs
        }
    }

    // ===================== q8_0 x q8_0 =====================

    /// `ggml_vec_dot_q8_0_q8_0` AVX2 body (arch/x86/quants.c:1325-1345).
    pub fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 32, 0, "vec_dot_q8_0_q8_0: n must be a multiple of 32");
        debug_assert!(
            x.len() >= n / 32 * 34 && y.len() >= n / 32 * 34,
            "vec_dot_q8_0_q8_0: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/32 * 34` / `* 34` bytes per row).
        unsafe { vec_dot_q8_0_q8_0_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q8_0_q8_0`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q8_0_q8_0_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / 32;
        // SAFETY: see `vec_dot_q4_0_q8_0`; int8 x int8 pairs reach at most
        // 128*127*2 = 32512 < 32767, so maddubs never saturates.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for ib in 0..nb {
                let bx = x.as_ptr().add(ib * 34);
                let by = y.as_ptr().add(ib * 34);
                let d = _mm256_set1_ps(d_f32(bx) * d_f32(by));
                let qx = _mm256_loadu_si256(bx.add(2) as *const __m256i);
                let qy = _mm256_loadu_si256(by.add(2) as *const __m256i);
                let q = mul_sum_i8_pairs_float(qx, qy);
                acc = _mm256_fmadd_ps(d, q, acc);
            }
            hsum8(acc)
        }
    }

    // ===================== q4_K x q8_K =====================

    /// Decode the 12 packed 6-bit (scale, min) pairs exactly like the C
    /// `utmp` shuffle (arch/x86/quants.c:2068-2073) — byte layout:
    /// `[sc0..sc7][min0..min7]`.
    #[inline]
    fn q4k_scale_bytes(packed: &[u8; 12]) -> [u8; 16] {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes(packed[0..4].try_into().unwrap());
        utmp[1] = u32::from_le_bytes(packed[4..8].try_into().unwrap());
        utmp[2] = u32::from_le_bytes(packed[8..12].try_into().unwrap());
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        let mut out = [0u8; 16];
        for (i, w) in utmp.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// `ggml_vec_dot_q4_K_q8_K` AVX2 body (arch/x86/quants.c:2057-2135):
    /// 8-lane `acc` for the 4-bit term, separate 4-lane `acc_m` for the min
    /// term, `_mm_add_ps(acc_m, movehl)` + `_mm_add_ss(movehdup)` at the end.
    pub fn vec_dot_q4_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 256, 0, "vec_dot_q4_K_q8_K: n must be a multiple of 256");
        debug_assert!(
            x.len() >= n / 256 * 144 && y.len() >= n / 256 * 292,
            "vec_dot_q4_K_q8_K: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/256 * 144` / `* 292` bytes per row).
        unsafe { vec_dot_q4_K_q8_K_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q4_K_q8_K`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q4_K_q8_K_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / QK_K;
        // SAFETY: as in `vec_dot_q4_0_q8_0_avx2`. `maddubs(q4l, q8l)` pairs a 4-bit
        // value (<=15) with int8 (<=127) => <=3810; the scale madd pairs
        // 6-bit scales (<=63) with those => <= 480060 < 2^31.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();
            let m4 = _mm256_set1_epi8(0x0F);

            for i in 0..nb {
                let bx = x.as_ptr().add(i * 144);
                let by = y.as_ptr().add(i * 292); // d f32 + qs 256 + bsums 32
                let d = _mm256_set1_ps(q8k_d(by) * d_f32(bx));
                let dmin = -q8k_d(by) * d_f32(bx.add(2));

                let bytes = q4k_scale_bytes(&*(bx.add(4) as *const [u8; 12]));
                // mins_and_scales: 16 u16 = [sc0..sc7][min0..min7]
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_loadu_si128(bytes.as_ptr() as *const __m128i));

                // min term: hadd_epi16 of bsums, madd against the 8 mins
                let q8sums = _mm256_loadu_si256(by.add(260) as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_set_m128i(sc128, sc128);

                let mut sumi = _mm256_setzero_si256();
                let mut q4 = bx.add(16); // qs[128]
                let mut q8 = by.add(4);

                for j in 0..4 {
                    let scale_l = scale_bcast_u16(scales, 2 * j);
                    let scale_h = scale_bcast_u16(scales, 2 * j + 1);

                    let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                    q4 = q4.add(32);
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                    let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let p16l = _mm256_madd_epi16(
                        scale_l,
                        _mm256_maddubs_epi16(q4l, q8l),
                    );

                    let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let p16h = _mm256_madd_epi16(
                        scale_h,
                        _mm256_maddubs_epi16(q4h, q8h),
                    );

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }

                acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
            }

            let mut acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
            acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
            hsum8(acc) + _mm_cvtss_f32(acc_m)
        }
    }

    // ===================== q5_K x q8_K =====================

    /// `ggml_vec_dot_q5_K_q8_K` AVX2 body (arch/x86/quants.c:2235-2300): same
    /// lane shape as q4_K, 5th bit from `hbits` (one 32-byte load per block),
    /// and the min term horizontally folded to a scalar fma chain.
    pub fn vec_dot_q5_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 256, 0, "vec_dot_q5_K_q8_K: n must be a multiple of 256");
        debug_assert!(
            x.len() >= n / 256 * 176 && y.len() >= n / 256 * 292,
            "vec_dot_q5_K_q8_K: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/256 * 176` / `* 292` bytes per row).
        unsafe { vec_dot_q5_K_q8_K_avx2(n, x, y) }
    }

    /// AVX2 body of [`vec_dot_q5_K_q8_K`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn vec_dot_q5_K_q8_K_avx2(n: usize, x: &[u8], y: &[u8]) -> f32 {
        let nb = n / QK_K;
        // SAFETY: see `vec_dot_q4_K_q8_K`; `q5 = q5l + (bit<<4) <= 31` keeps the
        // maddubs pairs (31*127*2 = 7874) inside int16.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut summs = 0f32;
            let m4 = _mm256_set1_epi8(0x0F);
            let mzero = _mm_setzero_si128();

            for i in 0..nb {
                let bx = x.as_ptr().add(i * 176);
                let by = y.as_ptr().add(i * 292);
                let d = _mm256_set1_ps(q8k_d(by) * d_f32(bx));
                let dmin = -q8k_d(by) * d_f32(bx.add(2));

                let bytes = q4k_scale_bytes(&*(bx.add(4) as *const [u8; 12]));
                let mins_and_scales = _mm256_cvtepu8_epi16(_mm_loadu_si128(bytes.as_ptr() as *const __m128i));

                let q8sums = _mm256_loadu_si256(by.add(260) as *const __m256i);
                let q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1),
                );
                let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
                let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                let mf = _mm_cvtsi128_si32(hsum) as f32;
                summs = mf.mul_add(dmin, summs);

                let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
                let scales = _mm256_set_m128i(sc128, sc128);

                let hbits = _mm256_loadu_si256(bx.add(16) as *const __m256i); // qh[32]
                let mut hmask = _mm256_set1_epi8(1);

                let mut sumi = _mm256_setzero_si256();
                let mut bit = 0i32;
                let mut q5 = bx.add(48); // qs[128]
                let mut q8 = by.add(4);

                for j in 0..4 {
                    let scale_0 = scale_bcast_u16(scales, 2 * j);
                    let scale_1 = scale_bcast_u16(scales, 2 * j + 1);

                    let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
                    q5 = q5.add(32);

                    let q5l_0 = _mm256_and_si256(q5bits, m4);
                    let sh = _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit));
                    bit += 1;
                    let q5h_0 = _mm256_slli_epi16(sh, 4);
                    let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    let sh = _mm256_srl_epi16(_mm256_and_si256(hbits, hmask), _mm_cvtsi32_si128(bit));
                    bit += 1;
                    let q5h_1 = _mm256_slli_epi16(sh, 4);
                    let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);

                    let p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
                    let p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                }

                acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
            }

            hsum8(acc) + summs
        }
    }

    // ===================== q6_K x q8_K =====================

    /// `ggml_vec_dot_q6_K_q8_K` AVX2 body (arch/x86/quants.c:2439-2510): four
    /// 32-value streams per 128-value half, scales via `get_scale_shuffle`
    /// (i16 lanes 0..7 = sc[2i], 8..15 = sc[2i+1]), the -32 bias subtracted
    /// once as `32*bsums*scales` after the k loop.
    ///
    /// Instantiated twice from one source body: the plain AVX2 build matches
    /// a reference compiled without AVT512, and the `evex` flavor matches the
    /// reference build of this machine (compiled with AVX512 enabled, so GCC
    /// selects EVEX encodings, the xmm16-31 register file and `vpternlog`
    /// AND/OR fusions for the very same AVX2 source, cf. the disasm note in
    /// `q6k_scale_masks`). Same lanes, same op order - pure instruction
    /// selection, bit-identical by construction.
    pub fn vec_dot_q6_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
        debug_assert_eq!(n % 256, 0, "vec_dot_q6_K_q8_K: n must be a multiple of 256");
        debug_assert!(
            x.len() >= n / 256 * 210 && y.len() >= n / 256 * 292,
            "vec_dot_q6_K_q8_K: short rows"
        );
        // SAFETY: the dispatcher (vec_dot.rs) only reaches this function when
        // `avx2()` is true, and the asserts above bound every load the kernel
        // performs (`n/256 * 210` / `* 292` bytes per row).
        unsafe {
            if avx512vl() {
                vec_dot_q6_K_q8_K_evex(n, x, y)
            } else {
                vec_dot_q6_K_q8_K_avx2(n, x, y)
            }
        }
    }

    macro_rules! q6k_vec_dot {
        ($(#[$attr:meta])* $name:ident) => {
            $(#[$attr])*
            unsafe fn $name(n: usize, x: &[u8], y: &[u8]) -> f32 {
                let nb = n / QK_K;
                // SAFETY: as in `vec_dot_q4_0_q8_0_avx2`. `maddubs(q4, q8)` takes the 6-bit
                // value (<=63, unsigned) against int8 => pair <= 8001; the scale madd
                // pairs int8 scales (<=127) => <= 2.03e6 < 2^31.
                unsafe {
                    let m3 = _mm256_set1_epi8(3);
                    let m15 = _mm256_set1_epi8(0x0F);
                    let m12 = _mm256_set1_epi8(12);
                    let m48 = _mm256_set1_epi8(48);
                    let mc0 = _mm256_set1_epi8(-64);
                    // quants.c:2490 get_scale_shuffle(is+k): the 8 constant
                    // masks, loaded once and kept in registers
                    let [sk0, sk1, sk2, sk3, sk4, sk5, sk6, sk7] = q6k_scale_masks();
                    let mut acc = _mm256_setzero_ps();

                    for i in 0..nb {
                        let bx = x.as_ptr().add(i * 210);
                        let by = y.as_ptr().add(i * 292);
                        let d = _mm256_set1_ps(q8k_d(by) * d_f32(bx.add(208)));

                        let q8sums = _mm256_loadu_si256(by.add(260) as *const __m256i);
                        let scales = _mm_loadu_si128(bx.add(192) as *const __m128i);
                        let scales_16 = _mm256_cvtepi8_epi16(scales);
                        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

                        let mut sumi = _mm256_setzero_si256();

                        let q4 = bx; // ql[128]
                        let qh = bx.add(128);
                        let q8 = by.add(4);

                        for j in 0..2 {
                            let q4bits1 = _mm256_loadu_si256(q4.add(j * 64) as *const __m256i);
                            let q4bits2 = _mm256_loadu_si256(q4.add(j * 64 + 32) as *const __m256i);
                            let q4bitsh = _mm256_loadu_si256(qh.add(j * 32) as *const __m256i);

                            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m3), 4);
                            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsh, m12), 2);
                            let q4h_2 = _mm256_and_si256(q4bitsh, m48);
                            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsh, mc0), 2);

                            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                            let q4_2 = _mm256_or_si256(
                                _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),
                                q4h_2,
                            );
                            let q4_3 = _mm256_or_si256(
                                _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),
                                q4h_3,
                            );

                            // unrolled k=0..4 exactly like the C (quants.c:2484-2497):
                            // keeping the four p16 accumulators as named registers (not
                            // a `[__m256i; 4]` indexed in a loop) measurably changes
                            // codegen for the whole kernel body
                            let q8_0 = _mm256_loadu_si256(q8.add(j * 128) as *const __m256i);
                            let q8_1 = _mm256_loadu_si256(q8.add(j * 128 + 32) as *const __m256i);
                            let q8_2 = _mm256_loadu_si256(q8.add(j * 128 + 64) as *const __m256i);
                            let q8_3 = _mm256_loadu_si256(q8.add(j * 128 + 96) as *const __m256i);

                            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                            // is = j*4 + k once the loop is unrolled: sk0..sk3
                            // for the first 128-value half, sk4..sk7 for the
                            // second (quants.c:2489-2492)
                            let (scale_0, scale_1, scale_2, scale_3) = if j == 0 {
                                (
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk0)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk1)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk2)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk3)),
                                )
                            } else {
                                (
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk4)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk5)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk6)),
                                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sk7)),
                                )
                            };

                            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                            p16_2 = _mm256_madd_epi16(scale_2, p16_2);
                            p16_3 = _mm256_madd_epi16(scale_3, p16_3);

                            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                        }

                        let sumi = _mm256_sub_epi32(sumi, q8sclsub);
                        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
                    }

                    hsum8(acc)
                }
            }
        };
    }

    /// AVX2 body of [`vec_dot_q6_K_q8_K`] (see the module docs for why the
    /// `#[target_feature]` attribute has to live on this function).
    q6k_vec_dot!(
        #[target_feature(enable = "avx2,fma,f16c")]
        vec_dot_q6_K_q8_K_avx2
    );

    /// Same body compiled with AVX512F/VL/BW enabled - the instruction
    /// selection the reference build gets on this machine (EVEX encodings,
    /// the xmm16-31 register file, `vpternlog`-fused AND/OR pairs). Lanes
    /// and op order identical to the AVX2 body; see
    /// [`vec_dot_q6_K_q8_K`].
    q6k_vec_dot!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw")]
        vec_dot_q6_K_q8_K_evex
    );

    // ===================== AVX512 f32 vector kernels =====================
    //
    // The lane-parallel f32 bodies the reference build runs (vec.h's
    // `__AVX512F__ && __AVX512DQ__` branches). Per-lane ops are value-identical
    // to the scalar ports in ops.rs/flash_attn.rs by construction (same ops,
    // same order, one lane per element), so SIMD == scalar bit-for-bit; the
    // tests below assert it and the reference dumps pin scalar == reference.
    // No AVX2 middle tier: vec.h's AVX2 `ggml_v_expf` (vec.h:1215) is a
    // *different polynomial* (integer `k` scaling, `fma(j,k,k)` tail), so an
    // AVX2 lane would not be bit-exact against the AVX512 reference — hosts
    // without AVX512 keep the scalar ports.

    /// Exact f32 bit patterns of the C hex-float literals (vec.h:1175-1190).
    mod vexp_consts {
        pub const R: f32 = 12582912.0; // 0x1.8p23
        pub const LOG2E: f32 = f32::from_bits(0x3FB8AA3B); // 0x1.715476p+0
        pub const C1: f32 = f32::from_bits(0x35BFBE8E); // 0x1.7f7d1cp-20
        pub const C2: f32 = f32::from_bits(0x3F317200); // 0x1.62e4p-1
        pub const P1: f32 = f32::from_bits(0x3C072010); // 0x1.0e4020p-7
        pub const P2: f32 = f32::from_bits(0x3D2B9F17); // 0x1.573e2ep-5
        pub const P3: f32 = f32::from_bits(0x3E2AAF33); // 0x1.555e66p-3
        pub const P4: f32 = f32::from_bits(0x3EFFFEDB); // 0x1.fffdb6p-2
        pub const P5: f32 = f32::from_bits(0x3F7FFFF6); // 0x1.ffffecp-1
    }

    /// `ggml_v_expf(__m512)` — vec.h:1172-1198 (AVX512F+DQ branch), the
    /// polynomial the reference binary runs on every SIMD softmax/silu chunk.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) unsafe fn v_expf_16(x: __m512) -> __m512 {
        use vexp_consts::*;
        let r = _mm512_set1_ps(R);
        let z = _mm512_fmadd_ps(x, _mm512_set1_ps(LOG2E), r);
        let n = _mm512_sub_ps(z, r);
        // b = fnmadd(n, C1, fnmadd(n, C2, x));  `_mm512_abs_ps(n)` spelled as
        // an integer AND (AVX512F only).
        let b = _mm512_fnmadd_ps(
            n,
            _mm512_set1_ps(C1),
            _mm512_fnmadd_ps(n, _mm512_set1_ps(C2), x),
        );
        let n_abs = _mm512_and_ps(n, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFF)));
        // d = |n| > 192 (ordered: NaN -> mask bit clear, like the scalar test)
        let d = _mm512_cmp_ps_mask(n_abs, _mm512_set1_ps(192.0), _CMP_GT_OQ);
        let u = _mm512_mul_ps(b, b);
        let j = _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_fmadd_ps(_mm512_set1_ps(P1), b, _mm512_set1_ps(P2)),
                u,
                _mm512_fmadd_ps(_mm512_set1_ps(P3), b, _mm512_set1_ps(P4)),
            ),
            u,
            _mm512_fmadd_ps(_mm512_set1_ps(P5), b, _mm512_set1_ps(1.0)),
        );
        // res = j * 2^n, one rounding (subnormals included)
        let res = _mm512_scalef_ps(j, n);
        // `if (_mm512_kortestz(d, d)) return res;` — kortestz(d, d) == d == 0
        if d == 0 {
            return res;
        }
        // alt = (n <= 0) ? 0 : +inf, then blend d ? alt : res
        let zero = _mm512_setzero_ps();
        let alt = _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ),
            _mm512_set1_ps(f32::INFINITY),
            zero,
        );
        _mm512_mask_blend_ps(d, res, alt)
    }

    /// `ggml_v_silu(__m512)` — vec.h:1200-1208: `x / (1 + v_expf(0 - x))`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) unsafe fn v_silu_16(x: __m512) -> __m512 {
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        let exp_neg_x = v_expf_16(neg_x);
        let one_plus = _mm512_add_ps(_mm512_set1_ps(1.0), exp_neg_x);
        _mm512_div_ps(x, one_plus)
    }

    /// `ggml_vec_silu_f32` — vec.cpp:380-413: 16-wide `ggml_v_silu` chunks and
    /// the scalar `ggml_silu_f32` tail (`for (; i < n; ++i) y[i] = ...` —
    /// libm `expf`, NOT the v512 polynomial; the two differ by 1 ulp on ~a
    /// quarter of tail lanes, which the batch-15 MoE node dump pinned: every
    /// differing `ffn_moe_swiglu` element sat at k >= 16 of an n_ff_exp 24
    /// row).
    pub fn vec_silu_f32(y: &mut [f32], x: &[f32]) {
        assert_eq!(y.len(), x.len(), "vec_silu_f32: length mismatch");
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the split.
            unsafe { vec_silu_f32_avx512(y, x) }
        } else {
            for (y, &x) in y.iter_mut().zip(x) {
                *y = crate::ops::ggml_silu_scalar_f32(x);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn vec_silu_f32_avx512(y: &mut [f32], x: &[f32]) {
        let n = y.len();
        let mut i = 0;
        while i + 16 <= n {
            let vx = _mm512_loadu_ps(x[i..].as_ptr());
            _mm512_storeu_ps(y[i..].as_mut_ptr(), v_silu_16(vx));
            i += 16;
        }
        // scalar tail — vec.h's ggml_silu_f32 = x/(1 + expf(-x)), libm expf
        while i < n {
            y[i] = crate::ops::ggml_silu_scalar_f32(x[i]);
            i += 1;
        }
    }

    /// `ggml_vec_swiglu_f32` — vec.cpp:417-447: per row, 16-wide
    /// `_mm512_mul_ps(ggml_v_silu(x), g)` chunks then the scalar tail
    /// `ggml_silu_f32(x[i]) * g[i]` (libm `expf`). Lane-wise, so storing the
    /// v512 silu to a stack array and multiplying per lane is bit-identical.
    pub fn vec_swiglu_f32(y: &mut [f32], x: &[f32], g: &[f32]) {
        assert_eq!(y.len(), x.len(), "vec_swiglu_f32: length mismatch");
        assert_eq!(y.len(), g.len(), "vec_swiglu_f32: length mismatch");
        if avx512() {
            let n = y.len();
            let mut i = 0;
            let mut silu = [0f32; 16];
            while i + 16 <= n {
                // SAFETY: `avx512()` holds; the block is fully in bounds.
                unsafe {
                    let vx = _mm512_loadu_ps(x[i..].as_ptr());
                    _mm512_storeu_ps(silu.as_mut_ptr(), v_silu_16(vx));
                }
                for j in 0..16 {
                    y[i + j] = silu[j] * g[i + j];
                }
                i += 16;
            }
            while i < n {
                y[i] = crate::ops::ggml_silu_scalar_f32(x[i]) * g[i];
                i += 1;
            }
        } else {
            for ((y, &x), &g) in y.iter_mut().zip(x).zip(g) {
                *y = crate::ops::ggml_silu_scalar_f32(x) * g;
            }
        }
    }

    /// `ggml_vec_soft_max_f32` — vec.cpp:531-597: per-16-lane chunk
    /// `v_expf(x - max)` stored back with `sum += (ggml_float)_mm512_reduce_add_ps(val)`,
    /// then the scalar libm `expf` tail. Returns the f64 partial-sum total.
    pub fn vec_soft_max_f32(y: &mut [f32], x: &[f32], max: f32) -> f64 {
        assert_eq!(y.len(), x.len(), "vec_soft_max_f32: length mismatch");
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the splits.
            unsafe { vec_soft_max_f32_avx512(y, x, max) }
        } else {
            // scalar lane-port (the same chunking the C's #else would produce):
            // 16-wide v512-polynomial chunks reduced with the GCC reduce tree,
            // scalar expf tail
            let n = y.len();
            let mut i = 0;
            let mut sum = 0f64;
            while i + 16 <= n {
                let mut v = [0f32; 16];
                for (slot, &xi) in v.iter_mut().zip(&x[i..i + 16]) {
                    *slot = crate::ops::ggml_expf_v512(xi - max);
                }
                y[i..i + 16].copy_from_slice(&v);
                sum += crate::vec_dot::reduce_add16(&v) as f64;
                i += 16;
            }
            while i < n {
                let val = (x[i] - max).exp(); // expf
                y[i] = val;
                sum += val as f64;
                i += 1;
            }
            sum
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn vec_soft_max_f32_avx512(y: &mut [f32], x: &[f32], max: f32) -> f64 {
        let n = y.len();
        let vmax = _mm512_set1_ps(max);
        let mut i = 0;
        let mut sum = 0f64;
        while i + 16 <= n {
            let val = v_expf_16(_mm512_sub_ps(_mm512_loadu_ps(x[i..].as_ptr()), vmax));
            _mm512_storeu_ps(y[i..].as_mut_ptr(), val);
            sum += _mm512_reduce_add_ps(val) as f64;
            i += 16;
        }
        // scalar tail — C: `float val = expf(x[i] - max);`
        while i < n {
            let val = (x[i] - max).exp();
            y[i] = val;
            sum += val as f64;
            i += 1;
        }
        sum
    }

    /// The same kernel with `y == x` — how the tiled FA kernel calls it
    /// (`ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew)`,
    /// ops.cpp:9077). The caller gates on [`avx512`] (the scalar lane-port
    /// fallback lives in `flash_attn::vec_soft_max_f32_inplace`).
    pub fn vec_soft_max_f32_inplace(y: &mut [f32], max: f32) -> f64 {
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: `avx512()` documented as checked by the caller; the loop
        // keeps every load/store inside the slice.
        unsafe {
            let n = y.len();
            let vmax = _mm512_set1_ps(max);
            let mut i = 0;
            let mut sum = 0f64;
            while i + 16 <= n {
                let val = v_expf_16(_mm512_sub_ps(_mm512_loadu_ps(y[i..].as_ptr()), vmax));
                _mm512_storeu_ps(y[i..].as_mut_ptr(), val);
                sum += _mm512_reduce_add_ps(val) as f64;
                i += 16;
            }
            while i < n {
                let val = (y[i] - max).exp();
                y[i] = val;
                sum += val as f64;
                i += 1;
            }
            sum
        }
    }

    /// `ggml_vec_add_f32(n, y, y, x)` — the aliasing form the tiled FA mask
    /// add uses (`ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32)`,
    /// ops.cpp:9056; vec.h:89 with z == x, i.e. vec.h:108's
    /// `ggml_vec_acc_f32` shape `y[i] += x[i]`). Caller gates on [`avx512`];
    /// per-lane addition, bit-identical to the scalar loop at any width.
    pub fn vec_acc_f32(y: &mut [f32], x: &[f32]) {
        assert_eq!(y.len(), x.len(), "vec_acc_f32: length mismatch");
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: `avx512()` documented as checked by the caller; the loop
        // keeps every load/store inside the slices.
        unsafe {
            let n = y.len();
            let mut i = 0;
            while i + 16 <= n {
                let a = _mm512_loadu_ps(y[i..].as_ptr());
                let b = _mm512_loadu_ps(x[i..].as_ptr());
                _mm512_storeu_ps(y[i..].as_mut_ptr(), _mm512_add_ps(a, b));
                i += 16;
            }
            while i < n {
                y[i] += x[i];
                i += 1;
            }
        }
    }

    /// `ggml_cpu_fp16_to_fp32(DV, v_f16, V32 + tk * DV, DV)` (ops.cpp:9088,
    /// the tiled FA V-tile pack) as one `vcvtph2ps` stream per 16 values.
    /// f16→f32 is exact, so the lane conversion is bit-identical to the
    /// scalar `GGML_CPU_FP16_TO_FP32` loop. `dst.len()` must be a multiple of
    /// 16 and `src` at least `dst.len() * 2` bytes. Caller gates on [`avx512`].
    pub fn fp16_to_f32_row(dst: &mut [f32], src: &[u8]) {
        assert!(dst.len() % 16 == 0, "fp16_to_f32_row: dst must be 16-multiple");
        assert!(src.len() >= dst.len() * 2, "fp16_to_f32_row: short src");
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: `avx512()` documented as checked by the caller; the loop
        // reads 32 bytes and writes 64 bytes per iteration, both inside the
        // slices by the asserts above.
        unsafe {
            let mut i = 0;
            while i < dst.len() {
                let h = _mm256_loadu_si256(src[i * 2..].as_ptr() as *const __m256i);
                _mm512_storeu_ps(dst[i..].as_mut_ptr(), _mm512_cvtph_ps(h));
                i += 16;
            }
        }
    }

    /// `ggml_vec_dot_f16` (vec.cpp:264, the x86 AVX512 `F32Cx16` branch —
    /// `GGML_F16_STEP 64`, `EPR 16`, `ARR 4`, simd-mappings.h:533-576): 4×16
    /// lanes of `cvtph2ps` + `vfmadd` per 64-value chunk, the
    /// `GGML_F16_VEC_REDUCE` pairwise fold (`sum0+=sum2; sum1+=sum3;
    /// sum0+=sum1`), the `_mm512_reduce_add_ps` `{8,4,2,1}` halving tree, and
    /// the `ggml_float` (f64) scalar tail (vec.cpp:366-368). Bit-identical to
    /// the scalar lane emulation `vec_dot::vec_dot_f16_c` (pinned against
    /// parity/vec_ref.bin); this is the speed form the reference's
    /// `one_chunk` flash attention calls per KV row (ops.cpp:8757).
    ///
    /// Caller gates on [`avx512`].
    ///
    /// # Safety
    /// `x` and `y` must each address `n` readable f16 values.
    pub fn vec_dot_f16(n: usize, x: *const u16, y: *const u16) -> f32 {
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: see above.
        unsafe { vec_dot_f16_avx512(n, x, y) }
    }

    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn vec_dot_f16_avx512(n: usize, x: *const u16, y: *const u16) -> f32 {
        let np = n & !63; // GGML_F16_STEP
        let mut sum = [_mm512_setzero_ps(); 4]; // GGML_F16_ARR
        let mut i = 0;
        while i < np {
            for (j, s) in sum.iter_mut().enumerate() {
                let ax = _mm512_cvtph_ps(_mm256_loadu_si256(x.add(i + j * 16) as *const __m256i));
                let ay = _mm512_cvtph_ps(_mm256_loadu_si256(y.add(i + j * 16) as *const __m256i));
                *s = _mm512_fmadd_ps(ax, ay, *s);
            }
            i += 64;
        }
        // GGML_F16_VEC_REDUCE (F32Cx16_REDUCE): halve the accumulator array
        sum[0] = _mm512_add_ps(sum[0], sum[2]);
        sum[1] = _mm512_add_ps(sum[1], sum[3]);
        sum[0] = _mm512_add_ps(sum[0], sum[1]);
        // `_mm512_reduce_add_ps` — the {8,4,2,1} halving order of
        // `vec_dot::reduce_add16`, spelled with explicit shuffles
        let lo = _mm512_castps512_ps256(sum[0]);
        let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(sum[0])));
        let mut t = _mm256_add_ps(lo, hi); // half = 8
        t = _mm256_add_ps(t, _mm256_permute2f128_ps::<1>(t, t)); // half = 4
        t = _mm256_add_ps(t, _mm256_shuffle_ps::<0xEE>(t, t)); // half = 2
        t = _mm256_add_ps(t, _mm256_shuffle_ps::<0x55>(t, t)); // half = 1
        let mut res = f64::from(_mm_cvtss_f32(_mm256_castps256_ps128(t)));
        // leftovers, accumulated in ggml_float like the C
        while i < n {
            let xv = half::f16::from_bits(x.add(i).read_unaligned()).to_f32();
            let yv = half::f16::from_bits(y.add(i).read_unaligned()).to_f32();
            res += f64::from(xv * yv);
            i += 1;
        }
        res as f32
    }

    /// `ggml_vec_mad_f16` (vec.h:439, the x86 `F32Cx16` branch): per 16 lanes
    /// `y = cvtps_ph(vfmadd(cvtph2ps(x), set1(v), cvtph2ps(y)))` (the
    /// `GGML_F16_STEP` unroll is value-identical), scalar tail
    /// `y = f16(f32(y) + f32(x)*v)` fused (`-ffp-contract=fast`). f16→f32 is
    /// exact and `cvtps_ph`/`from_f32` are both round-to-nearest-even, so
    /// every lane equals the scalar loop bit-for-bit — the `one_chunk` V
    /// accumulation (ops.cpp:8793).
    ///
    /// Caller gates on [`avx512`].
    ///
    /// # Safety
    /// `y` and `x` must each address `n` f16 values; `y` writable.
    pub fn vec_mad_f16(y: *mut u16, x: *const u16, v: f32, n: usize) {
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: see above.
        unsafe { vec_mad_f16_avx512(y, x, v, n) }
    }

    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn vec_mad_f16_avx512(y: *mut u16, x: *const u16, v: f32, n: usize) {
        let vv = _mm512_set1_ps(v);
        let mut i = 0;
        while i + 16 <= n {
            let ax = _mm512_cvtph_ps(_mm256_loadu_si256(x.add(i) as *const __m256i));
            let ay = _mm512_cvtph_ps(_mm256_loadu_si256(y.add(i) as *const __m256i));
            let r = _mm512_fmadd_ps(ax, vv, ay);
            _mm256_storeu_si256(y.add(i) as *mut __m256i, _mm512_cvtps_ph(r, 0));
            i += 16;
        }
        while i < n {
            let yv = half::f16::from_bits(y.add(i).read_unaligned()).to_f32();
            let xv = half::f16::from_bits(x.add(i).read_unaligned()).to_f32();
            let b = half::f16::from_f32(xv.mul_add(v, yv)).to_bits();
            y.add(i).write_unaligned(b);
            i += 1;
        }
    }

    /// `ggml_vec_scale_f16` (vec.h:769, the x86 `F32Cx16` branch): per 16
    /// lanes `y = cvtps_ph(cvtph2ps(y) * set1(v))` — one rounded multiply per
    /// lane like the scalar loop (the `one_chunk` rescale, ops.cpp:8783).
    ///
    /// Caller gates on [`avx512`].
    ///
    /// # Safety
    /// `y` must address `n` writable f16 values.
    pub fn vec_scale_f16(y: *mut u16, v: f32, n: usize) {
        debug_assert!(avx512(), "caller must gate on avx512()");
        // SAFETY: see above.
        unsafe { vec_scale_f16_avx512(y, v, n) }
    }

    #[target_feature(enable = "avx512f,avx512dq,fma")]
    unsafe fn vec_scale_f16_avx512(y: *mut u16, v: f32, n: usize) {
        let vv = _mm512_set1_ps(v);
        let mut i = 0;
        while i + 16 <= n {
            let ay = _mm512_cvtph_ps(_mm256_loadu_si256(y.add(i) as *const __m256i));
            let r = _mm512_mul_ps(ay, vv);
            _mm256_storeu_si256(y.add(i) as *mut __m256i, _mm512_cvtps_ph(r, 0));
            i += 16;
        }
        while i < n {
            let yv = half::f16::from_bits(y.add(i).read_unaligned()).to_f32();
            let b = half::f16::from_f32(yv * v).to_bits();
            y.add(i).write_unaligned(b);
            i += 1;
        }
    }

    /// `ggml_vec_scale_f32` — vec.h:703-766: `y[i] *= v`, one rounded multiply
    /// per lane (the AVX512 GGML_F32_STEP unroll is value-identical to any
    /// width) and the scalar tail.
    pub fn vec_scale_f32(y: &mut [f32], v: f32) {
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the splits.
            unsafe { vec_scale_f32_avx512(y, v) }
        } else {
            for y in y.iter_mut() {
                *y *= v;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn vec_scale_f32_avx512(y: &mut [f32], v: f32) {
        let n = y.len();
        let vv = _mm512_set1_ps(v);
        let mut i = 0;
        while i + 16 <= n {
            let a = _mm512_loadu_ps(y[i..].as_ptr());
            _mm512_storeu_ps(y[i..].as_mut_ptr(), _mm512_mul_ps(a, vv));
            i += 16;
        }
        while i < n {
            y[i] *= v;
            i += 1;
        }
    }

    /// `z[i] = x[i] * v` — `ggml_vec_scale_f32`'s per-lane multiply applied
    /// from a source buffer (the GGML_OP_MUL broadcast row, binary-ops.cpp's
    /// `z[i] = f32_to_dst(op(src0_to_f32(x[i]), y0))` with `ne10 == 1`).
    /// One rounded multiply per lane, bit-identical to the scalar loop.
    pub fn vec_mul1_f32(z: &mut [f32], x: &[f32], v: f32) {
        assert_eq!(z.len(), x.len(), "vec_mul1_f32: length mismatch");
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the splits.
            unsafe {
                let n = z.len();
                let vv = _mm512_set1_ps(v);
                let mut i = 0;
                while i + 16 <= n {
                    let a = _mm512_loadu_ps(x[i..].as_ptr());
                    _mm512_storeu_ps(z[i..].as_mut_ptr(), _mm512_mul_ps(a, vv));
                    i += 16;
                }
                while i < n {
                    z[i] = x[i] * v;
                    i += 1;
                }
            }
        } else {
            for (z, &x) in z.iter_mut().zip(x) {
                *z = x * v;
            }
        }
    }

    /// `ggml_vec_add_f32` — vec.h:89-99 (the C chunk is 8-wide AVX2; per-lane
    /// addition makes any width value-identical).
    pub fn vec_add_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
        assert_eq!(z.len(), x.len());
        assert_eq!(z.len(), y.len());
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the splits.
            unsafe { vec_addmul_f32_avx512(z, x, y, true) }
        } else {
            for ((z, &x), &y) in z.iter_mut().zip(x).zip(y) {
                *z = x + y;
            }
        }
    }

    /// `ggml_vec_mul_f32` — vec.h:128. The C loop is plain `z[i] = x[i]*y[i]`
    /// (the reference's GCC auto-vectorizes it; per-lane product either way).
    pub fn vec_mul_f32(z: &mut [f32], x: &[f32], y: &[f32]) {
        assert_eq!(z.len(), x.len());
        assert_eq!(z.len(), y.len());
        if avx512() {
            // SAFETY: `avx512()` holds; slice bounds are kept by the splits.
            unsafe { vec_addmul_f32_avx512(z, x, y, false) }
        } else {
            for ((z, &x), &y) in z.iter_mut().zip(x).zip(y) {
                *z = x * y;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn vec_addmul_f32_avx512(z: &mut [f32], x: &[f32], y: &[f32], add: bool) {
        let n = z.len();
        let mut i = 0;
        while i + 16 <= n {
            let a = _mm512_loadu_ps(x[i..].as_ptr());
            let b = _mm512_loadu_ps(y[i..].as_ptr());
            let r = if add { _mm512_add_ps(a, b) } else { _mm512_mul_ps(a, b) };
            _mm512_storeu_ps(z[i..].as_mut_ptr(), r);
            i += 16;
        }
        while i < n {
            z[i] = if add { x[i] + y[i] } else { x[i] * y[i] };
            i += 1;
        }
    }

    /// `simd_gemm` — simd-gemm.h:60-131 (AVX512 build: `GEMM_RM = 4`,
    /// `GEMM_RN = 4`, `KN = GGML_F32_EPR = 16`):
    /// `C[M x N] += A[M x K] * B[K x N]`.
    ///
    /// Every output element is one independent `_mm512_fmadd_ps` chain over
    /// `kk` ascending (the RM/RN register blocking only assigns elements to
    /// lanes), so this is bit-identical to the scalar `#else` body with
    /// `f32::mul_add` — which `flash_attn::simd_gemm` keeps as the fallback.
    pub fn simd_gemm_avx512(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        debug_assert!(c.len() >= m * n && a.len() >= m * k && b.len() >= k * n);
        // SAFETY: `avx512()` is checked by the caller (flash_attn::simd_gemm);
        // every load/store the ukernels perform stays inside `m*n`/`m*k`/`k*n`.
        unsafe { simd_gemm_avx512_impl(c, a, b, m, k, n) }
    }

    /// `simd_gemm_ukernel<RM, RN>` — simd-gemm.h:24-55.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn gemm_ukernel<const RM: usize, const RN: usize>(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        k: usize,
        n: usize,
    ) {
        const KN: usize = 16;
        let mut acc = [[_mm512_setzero_ps(); RN]; RM];
        for i in 0..RM {
            for r in 0..RN {
                acc[i][r] = _mm512_loadu_ps(c.add(i * n + r * KN));
            }
        }
        for kk in 0..k {
            let mut bv = [_mm512_setzero_ps(); RN];
            for r in 0..RN {
                bv[r] = _mm512_loadu_ps(b.add(kk * n + r * KN));
            }
            for i in 0..RM {
                let p = _mm512_set1_ps(a.add(i * k + kk).read());
                for r in 0..RN {
                    acc[i][r] = _mm512_fmadd_ps(bv[r], p, acc[i][r]);
                }
            }
        }
        for i in 0..RM {
            for r in 0..RN {
                _mm512_storeu_ps(c.add(i * n + r * KN), acc[i][r]);
            }
        }
    }

    /// `simd_gemm_ukernel_tail<RM>` — simd-gemm.h:59-107 (AVX512F branch,
    /// sync batch D): the `N % KN` tail columns as one masked pass
    /// (`maskz_loadu` + `mask3_fmadd` + `mask_storeu`). Bit-identical to the
    /// per-element `f32::mul_add` chain it replaces (each lane keeps the same
    /// fused `acc + A*B` sequence over kk ascending); transcribed in the
    /// masked form for 1:1 fidelity with the new upstream tail.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn gemm_ukernel_tail<const RM: usize>(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        k: usize,
        n: usize,
        cols: usize,
    ) {
        let mask: __mmask16 = ((1u32 << cols) - 1) as u16;
        let mut acc = [_mm512_setzero_ps(); RM];
        for i in 0..RM {
            acc[i] = _mm512_maskz_loadu_ps(mask, c.add(i * n));
        }
        for kk in 0..k {
            let bv = _mm512_maskz_loadu_ps(mask, b.add(kk * n));
            for i in 0..RM {
                acc[i] = _mm512_mask3_fmadd_ps(
                    _mm512_set1_ps(a.add(i * k + kk).read()),
                    bv,
                    acc[i],
                    mask,
                );
            }
        }
        for i in 0..RM {
            _mm512_mask_storeu_ps(c.add(i * n), mask, acc[i]);
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn simd_gemm_avx512_impl(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        // GEMM_RM = 4, GEMM_RN = 4, KN = 16 (simd-gemm.h:11-13 AVX512 branch)
        let mut c = c.as_mut_ptr();
        let mut a = a.as_ptr();
        let mut ii = 0usize;
        while ii + 4 <= m {
            let mut jj = 0usize;
            while jj + 4 * 16 <= n {
                gemm_ukernel::<4, 4>(c.add(jj), a, b.as_ptr().add(jj), k, n);
                jj += 4 * 16;
            }
            while jj + 16 <= n {
                gemm_ukernel::<4, 1>(c.add(jj), a, b.as_ptr().add(jj), k, n);
                jj += 16;
            }
            // simd-gemm.h:127-130: the jj tail as one masked pass
            if jj < n {
                gemm_ukernel_tail::<4>(c.add(jj), a, b.as_ptr().add(jj), k, n, n - jj);
            }
            a = a.add(4 * k);
            c = c.add(4 * n);
            ii += 4;
        }
        // simd-gemm.h:133-151: tail rows, one at a time
        while ii < m {
            let mut jj = 0usize;
            while jj + 4 * 16 <= n {
                gemm_ukernel::<1, 4>(c.add(jj), a, b.as_ptr().add(jj), k, n);
                jj += 4 * 16;
            }
            while jj + 16 <= n {
                gemm_ukernel::<1, 1>(c.add(jj), a, b.as_ptr().add(jj), k, n);
                jj += 16;
            }
            if jj < n {
                gemm_ukernel_tail::<1>(c.add(jj), a, b.as_ptr().add(jj), k, n, n - jj);
            }
            a = a.add(k);
            c = c.add(n);
            ii += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use kernels::*;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::blocks::*;
    use crate::vec_dot;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 28) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `ggml_compute_fp16_to_fp32` (ggml-impl.h:396-418, the FBGEMM bit
    /// algorithm) — the function whose values fill the reference's
    /// `ggml_table_f32_f16` (ggml-cpu.c:3886-3890), i.e. what
    /// `GGML_CPU_FP16_TO_FP32` returns in the reference lane.
    fn fp16_to_f32_portable(h: u16) -> u32 {
        let w = (h as u32) << 16;
        let sign = w & 0x8000_0000;
        let two_w = w.wrapping_add(w);
        let exp_offset = 0xE0u32 << 23;
        let normalized_value =
            (f32::from_bits((two_w >> 4).wrapping_add(exp_offset)) * f32::from_bits(0x7800_000)).to_bits();
        let denormalized_value =
            (f32::from_bits((two_w >> 17) | (126u32 << 23)) - 0.5f32).to_bits();
        let result = sign
            | if two_w < (1u32 << 27) {
                denormalized_value
            } else {
                normalized_value
            };
        result
    }

    /// `d_f32`'s `vcvtph2ps` must return the same bits as the reference's
    /// `ggml_table_f32_f16` entry — exhaustively over **all 65536 f16 bit
    /// patterns**, zero differences (measured on this machine: this Zen5 core
    /// leaves f16 sNaN payloads un-quieted through `vcvtph2ps`, exactly like
    /// the portable bit algorithm, so even the theoretical divergence class
    /// is empty here).
    #[test]
    fn f16c_cvtph_matches_portable() {
        if !avx2() {
            return; // no f16c without the avx2() gate's f16c part
        }
        let hw = |h: u16| -> u32 {
            #[target_feature(enable = "f16c")]
            unsafe fn conv(h: u16) -> u32 {
                _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32))).to_bits()
            }
            unsafe { conv(h) }
        };
        for h in 0u16..=u16::MAX {
            assert_eq!(hw(h), fp16_to_f32_portable(h), "f16 0x{h:04x}");
        }
    }

    /// `quantize_row_q8_0`'s AVX2 body must be bit-identical to the scalar
    /// spec (quants.rs) — same `id = 127/amax`, ties-even rounding, same
    /// pack/permute element order — over the activation lengths the wdata
    /// pass sees (k multiples of 32, incl. 896/4864/151936-row model shapes),
    /// with tie-heavy and tiny-magnitude inputs.
    #[test]
    fn quantize_row_q8_0_avx2_matches_scalar_bit_exact() {
        if !avx2() {
            return;
        }
        let scalar = |x: &[f32], y: &mut [BlockQ8_0]| {
            for (i, b) in y.iter_mut().enumerate() {
                let mut amax = 0.0f32;
                for j in 0..QK8_0 {
                    amax = amax.max(x[i * QK8_0 + j].abs());
                }
                let d = amax / 127.0;
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                let mut qs = [0i8; QK8_0];
                for j in 0..QK8_0 {
                    let r = (x[i * QK8_0 + j] * id).round_ties_even();
                    // cvtps_epi32's INT_MIN saturation, mirrored (see quants.rs)
                    qs[j] = if r.is_nan() || r.abs() >= 2147483648.0 {
                        i8::MIN
                    } else {
                        r as i8
                    };
                }
                *b = BlockQ8_0 { d: f16::from_f32(d), qs };
            }
        };
        for n in [32, 64, 96, 256, 896, 1024, 4864] {
            let nb = n / QK8_0;
            // random, tie-heavy (x*id exactly at .5), near-zero (amax==0 → id=0)
            let mut inputs = lcg(n, 11 + n as u32);
            inputs.extend(lcg(n, 5).iter().map(|v| (v * 127.0).round() / 127.0)); // exact .5 ties
            inputs.extend(std::iter::repeat(0.0).take(n));
            inputs.extend(lcg(n, 9).iter().map(|v| v * 1e-40)); // subnormal-ish scale
            for c in 0..4 {
                let x = &inputs[c * n..(c + 1) * n];
                let mut a = vec![BlockQ8_0 { d: f16::ZERO, qs: [0; QK8_0] }; nb];
                let mut b = vec![BlockQ8_0 { d: f16::ZERO, qs: [0; QK8_0] }; nb];
                scalar(x, &mut a);
                crate::simd_x86::quantize_row_q8_0(x, &mut b);
                let (ab, bb) = (
                    bytemuck::cast_slice::<BlockQ8_0, u8>(&a),
                    bytemuck::cast_slice::<BlockQ8_0, u8>(&b),
                );
                assert_eq!(ab, bb, "n={n} case={c}: SIMD q8_0 quantizer diverged");
            }
        }
    }

    /// `nb` random blocks of `bsize` bytes; the 2 bytes at each offset in
    /// `d_at` are overwritten with a finite f16 (the `d` fields both paths
    /// convert). Random *raw* payloads are the stronger test here: they cover
    /// every qs/qh/scale bit pattern, including ones no quantizer emits.
    fn rand_blocks(nb: usize, bsize: usize, d_at: &[usize], seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 13) as u8
        };
        let mut v = vec![0u8; nb * bsize];
        for b in 0..nb {
            for j in 0..bsize {
                v[b * bsize + j] = next();
            }
            for &o in d_at {
                // Finite (and non-zero) f16: keep the sign and mantissa, force
                // the exponent into 10..25. NaN/Inf inputs are excluded on
                // purpose: with a NaN `d`/`m`/`s` the *operand order* of the
                // block fma becomes observable (x86 fma returns the first NaN
                // operand quieted), and the reference's C bodies order the
                // multiplicands differently from the scalar lane-port for
                // q5_1 (`fmadd(q, dx*dy, acc)` vs `dxdy.mul_add(q, acc)`).
                // Real quantizers never emit a non-finite scale.
                let raw = u16::from_le_bytes([v[b * bsize + o], v[b * bsize + o + 1]]);
                let e = ((raw >> 10) & 0x0F) + 10; // 10..25
                let bits = (raw & 0x8000) | (e << 10) | (raw & 0x03FF);
                v[b * bsize + o..b * bsize + o + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        v
    }

    /// Rewrite `-128` to `-127` in the q8 payloads of `v` (offsets `from..` of
    /// every `bsize`-byte block).
    ///
    /// [quants.c `mul_sum_i8_pairs_float`] computes `|x| * y*sign(x)` via
    /// `_mm256_sign_epi8`, and `-(-128)` wraps back to -128 in int8. For an
    /// activation byte of -128 and a *negative* weight byte the reference's
    /// AVX2 (and its `dpbusd` variant — same operands) therefore yields
    /// `|x|*(-128)` where exact math wants `|x|*128`; the reference's own
    /// *generic* kernel (what the scalar fallback mirrors) gives the exact
    /// value. That input is unreachable for real data: both q8_0/q8_1
    /// quantizers use `id = 127/amax` (`d = amax/127`), so activations stay in
    /// `[-127, 127]`; q8_K (`iscale = -128/max`, which *can* emit -128) is only
    /// ever consumed by the K-quant kernels, which use plain `maddubs`.
    /// The kernel keeps the reference's behavior; the raw-input tests below
    /// stay in the reachable range so they pin SIMD == scalar.
    fn clamp_q8_act(v: &mut [u8], bsize: usize, from: usize) {
        for b in 0..v.len() / bsize {
            for j in from..bsize {
                if v[b * bsize + j] == 0x80 {
                    v[b * bsize + j] = 0x81;
                }
            }
        }
    }

    /// Random q8_0 activation blocks within the reachable range (see
    /// [`clamp_q8_act`]).
    fn raw_q8(nb: usize, seed: u32) -> Vec<u8> {
        let mut v = rand_blocks(nb, 34, &[0], seed);
        clamp_q8_act(&mut v, 34, 2); // keep the f16 d field untouched
        v
    }

    /// Random `block_q8_K` blocks carrying the *derived* fields a real
    /// quantizer produces: `bsums[i] = sum(qs[16i..16i+16])` (computed from qs
    /// by `quantize_row_q8_K`; |bsums| <= 16*127 = 2032 keeps the kernels'
    /// `hadd_epi16` inside int16) and a finite f32 `d`. Independently random
    /// bsums would be an unreachable input where the reference's *saturating*
    /// hadd legitimately differs from the scalar port's i32 sum.
    fn raw_q8k(nb: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let mut v = vec![0u8; nb * 292];
        for b in 0..nb {
            let blk = &mut v[b * 292..(b + 1) * 292];
            let d = ((next() >> 9) as f32) * 1e-6 + 0.001; // finite positive f32
            blk[0..4].copy_from_slice(&d.to_le_bytes());
            for j in 0..256 {
                blk[4 + j] = (next() >> 13) as u8;
            }
            for i in 0..16 {
                let mut s = 0i16;
                for j in 0..16 {
                    s = s.wrapping_add(blk[4 + 16 * i + j] as i8 as i16);
                }
                blk[260 + 2 * i..262 + 2 * i].copy_from_slice(&s.to_le_bytes());
            }
        }
        v
    }

    /// Every SIMD kernel must be bit-identical to the scalar lane-port it
    /// replaces, both on blocks produced by the crate's quantizers (the `d`/
    /// scale fields the reference would see) and on random raw blocks.
    #[test]
    fn simd_matches_scalar_bit_exact() {
        assert!(avx2(), "test host must have AVX2");
        for &n in &[32usize, 64, 96, 160, 320, 480, 960, 4896] {
            let nb = n / 32;
            for seed in [1u32, 2, 3] {
                let xq = lcg(n, seed);
                let yq = lcg(n, seed + 100);

                let mut x4 = vec![0u8; nb * 18];
                crate::quants::quantize_row_q4_0_ref(&xq, bytemuck::cast_slice_mut(&mut x4));
                let mut x5 = vec![0u8; nb * 22];
                crate::quants::quantize_row_q5_0_ref(&xq, bytemuck::cast_slice_mut(&mut x5));
                let mut x8 = vec![0u8; nb * 34];
                crate::quants::quantize_row_q8_0(&xq, bytemuck::cast_slice_mut(&mut x8));
                let mut y8 = vec![0u8; nb * 34];
                crate::quants::quantize_row_q8_0(&yq, bytemuck::cast_slice_mut(&mut y8));
                let mut x41 = vec![0u8; nb * 20];
                crate::quants::quantize_row_q4_1_ref(&xq, bytemuck::cast_slice_mut(&mut x41));
                let mut x51 = vec![0u8; nb * 24];
                crate::quants::quantize_row_q5_1_ref(&xq, bytemuck::cast_slice_mut(&mut x51));
                let mut y81 = vec![0u8; nb * 36];
                crate::quants::quantize_row_q8_1(&yq, bytemuck::cast_slice_mut(&mut y81));

                let raw4 = (rand_blocks(nb, 18, &[0], seed + 3), raw_q8(nb, seed + 4));
                let raw5 = (rand_blocks(nb, 22, &[0], seed + 5), raw_q8(nb, seed + 6));
                let raw8 = (rand_blocks(nb, 34, &[0], seed + 7), raw_q8(nb, seed + 8));
                let raw41 =
                    (rand_blocks(nb, 20, &[0, 2], seed + 9), rand_blocks(nb, 36, &[0, 2], seed + 10));
                let raw51 =
                    (rand_blocks(nb, 24, &[0, 2], seed + 11), rand_blocks(nb, 36, &[0, 2], seed + 12));

                macro_rules! eq {
                    ($name:expr, $simd:expr, $scalar:expr, $x:expr, $y:expr) => {{
                        let a = $simd(n, $x, $y);
                        let b = $scalar(n, $x, $y);
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            concat!($name, ": n={} seed={} simd {} != scalar {}"),
                            n,
                            seed,
                            a,
                            b
                        );
                    }};
                }

                eq!("q4_0", vec_dot_q4_0_q8_0, vec_dot::vec_dot_q4_0_q8_0_scalar, &x4, &y8);
                eq!("q4_0 raw", vec_dot_q4_0_q8_0, vec_dot::vec_dot_q4_0_q8_0_scalar, &raw4.0, &raw4.1);
                eq!("q5_0", vec_dot_q5_0_q8_0, vec_dot::vec_dot_q5_0_q8_0_scalar, &x5, &y8);
                eq!("q5_0 raw", vec_dot_q5_0_q8_0, vec_dot::vec_dot_q5_0_q8_0_scalar, &raw5.0, &raw5.1);
                eq!("q8_0", vec_dot_q8_0_q8_0, vec_dot::vec_dot_q8_0_q8_0_scalar, &x8, &y8);
                eq!("q8_0 raw", vec_dot_q8_0_q8_0, vec_dot::vec_dot_q8_0_q8_0_scalar, &raw8.0, &raw8.1);
                eq!("q4_1", vec_dot_q4_1_q8_1, vec_dot::vec_dot_q4_1_q8_1_scalar, &x41, &y81);
                eq!("q4_1 raw", vec_dot_q4_1_q8_1, vec_dot::vec_dot_q4_1_q8_1_scalar, &raw41.0, &raw41.1);
                eq!("q5_1", vec_dot_q5_1_q8_1, vec_dot::vec_dot_q5_1_q8_1_scalar, &x51, &y81);
                eq!("q5_1 raw", vec_dot_q5_1_q8_1, vec_dot::vec_dot_q5_1_q8_1_scalar, &raw51.0, &raw51.1);
            }
        }

        for &n in &[256usize, 512, 768, 1024, 4864] {
            let nb = n / QK_K;
            for seed in [5u32, 6, 7] {
                let xq = lcg(n, seed);
                let yq = lcg(n, seed + 200);
                let mut yk = vec![0u8; nb * 292];
                crate::quants::quantize_row_q8_K(&yq, bytemuck::cast_slice_mut(&mut yk));
                let yrk = raw_q8k(nb, seed + 21);

                macro_rules! keq {
                    ($name:expr, $bsize:expr, $d_at:expr, $simd:expr, $scalar:expr, $quant:expr) => {{
                        let mut xb = vec![0u8; nb * $bsize];
                        $quant(&xq, bytemuck::cast_slice_mut(&mut xb));
                        let a = $simd(n, &xb, &yk);
                        let b = $scalar(n, &xb, &yk);
                        assert_eq!(
                            a.to_bits(), b.to_bits(),
                            "{} real: n={} seed={} simd {} != scalar {}", $name, n, seed, a, b
                        );
                        let xr = rand_blocks(nb, $bsize, $d_at, seed + 31);
                        for (label, act) in [("raw=quant-act", &yk), ("raw=raw-act", &yrk)] {
                            let a = $simd(n, &xr, act);
                            let b = $scalar(n, &xr, act);
                            assert_eq!(
                                a.to_bits(), b.to_bits(),
                                "{} {}: n={} seed={} simd {} != scalar {}",
                                $name, label, n, seed, a, b
                            );
                        }
                    }};
                }

                keq!("q4_K", 144, &[0, 2], vec_dot_q4_K_q8_K,
                     vec_dot::vec_dot_q4_K_q8_K_scalar, crate::quants_k::quantize_row_q4_K_ref);
                keq!("q5_K", 176, &[0, 2], vec_dot_q5_K_q8_K,
                     vec_dot::vec_dot_q5_K_q8_K_scalar, crate::quants_k::quantize_row_q5_K_ref);
                keq!("q6_K", 210, &[208], vec_dot_q6_K_q8_K,
                     vec_dot::vec_dot_q6_K_q8_K_scalar, crate::quants_k::quantize_row_q6_K_ref);
            }
        }
    }

    /// Throughput of the SIMD kernels as `mul_mat` calls them (one call per
    /// weight row). `cargo test --release -p ggml --lib kernel_throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kernel_throughput() {
        use std::time::Instant;
        fn bench(name: &str, n: usize, rows: usize, mut f: impl FnMut() -> f32) {
            for _ in 0..64 {
                std::hint::black_box(f());
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let mut acc = 0f32;
                for _ in 0..rows {
                    acc += f();
                }
                std::hint::black_box(acc);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{name:22} n={n:5} rows={rows:6}  {:7.1} ns/row  {:6.2} G el/s",
                best / rows as f64 * 1e9,
                rows as f64 * n as f64 / best / 1e9
            );
        }
        let n = 896usize;
        let nb8 = n / 32;
        let xf = lcg(n, 1);
        let yf = lcg(n, 2);
        let mut x50 = vec![0u8; nb8 * 22];
        crate::quants::quantize_row_q5_0_ref(&xf, bytemuck::cast_slice_mut(&mut x50));
        let mut x80 = vec![0u8; nb8 * 34];
        crate::quants::quantize_row_q8_0(&xf, bytemuck::cast_slice_mut(&mut x80));
        let mut y80 = vec![0u8; nb8 * 34];
        crate::quants::quantize_row_q8_0(&yf, bytemuck::cast_slice_mut(&mut y80));
        let rows = 8192;
        // black_box the slices: with a pure f16 conversion the whole call is
        // loop-invariant and LLVM hoists it out of the timing loop otherwise
        bench("q5_0 simd", n, rows, || {
            vec_dot_q5_0_q8_0(n, std::hint::black_box(&x50), &y80)
        });
        bench("q5_0 scalar", n, rows, || {
            vec_dot::vec_dot_q5_0_q8_0_scalar(n, std::hint::black_box(&x50), &y80)
        });
        bench("q8_0 simd", n, rows, || {
            vec_dot_q8_0_q8_0(n, std::hint::black_box(&x80), &y80)
        });
        bench("q8_0 scalar", n, rows, || {
            vec_dot::vec_dot_q8_0_q8_0_scalar(n, std::hint::black_box(&x80), &y80)
        });
        bench("q5_0 via vec_dot_row", n, rows, || {
            vec_dot::vec_dot_row(crate::types::GgmlType::Q5_0, n, std::hint::black_box(&x50), &y80)
        });

        // q6_K x q8_K with the exact protocol of parity/ref_vecdot_q6k_bench.c
        // (n = ffn_down row length 4864, 64 distinct x/y rows cycled so the
        // working set is L2-resident — compares instruction throughput, not
        // DRAM streaming; reference lane: the AVX2 body quants.c:2426-2510).
        use bytemuck::Zeroable;
        let n = 4864usize;
        let nbk = n / 256;
        let x6k: Vec<Vec<u8>> = (0..64)
            .map(|i| {
                let xf = lcg(n, 100 + i as u32);
                let mut b = vec![unsafe { crate::blocks::BlockQ6K::zeroed() }; nbk];
                crate::quants_k::quantize_row_q6_K_ref(&xf, &mut b);
                bytemuck::cast_slice(&b).to_vec()
            })
            .collect::<Vec<_>>();
        let y8k: Vec<Vec<u8>> = (0..64)
            .map(|i| {
                let yf = lcg(n, 200 + i as u32);
                let mut b = vec![unsafe { crate::blocks::BlockQ8K::zeroed() }; nbk];
                crate::quants::quantize_row_q8_K(&yf, &mut b);
                bytemuck::cast_slice(&b).to_vec()
            })
            .collect::<Vec<_>>();
        {
            let mut i = 0usize;
            bench("q6_K x q8_K simd", n, rows, || {
                let s = vec_dot_q6_K_q8_K(n, &x6k[i & 63], &y8k[(i >> 6) & 63]);
                i = i.wrapping_add(1);
                s
            });
        }
    }

    // ===================== AVX512 f32 kernels =====================

    /// Deterministic f32 inputs spanning the ranges the kernels see in
    /// practice: LCG values in (-1, 1), scaled mantissa/exponent products
    /// (covers the `|n| > 192` saturation of `ggml_v_expf`), and specials.
    fn f32_sweep(n: usize, seed: u32) -> Vec<f32> {
        let mut v = lcg(n, seed);
        for (i, x) in v.iter_mut().enumerate() {
            let e = ((i * 7 + seed as usize) % 260) as i32 - 130; // 2^e within f32 range
            *x *= (2.0f32).powi(e);
        }
        v.extend([0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, 88.5, -88.5, -104.0, 104.0]);
        v
    }

    /// `ggml_v_silu`/`ggml_vec_silu_f32` (vec.h:1200 / vec.cpp:380): the
    /// 16-lane chunks must be bit-identical to the scalar `ggml_silu_f32`
    /// lane-port (same polynomial, per-lane div), tail included.
    #[test]
    fn avx512_f16_vec_kernels_match_scalar_bit_exact() {
        // ggml_vec_dot_f16 / ggml_vec_mad_f16 / ggml_vec_scale_f16 (the
        // F32Cx16 bodies) vs their scalar lane emulations — the dumps pin the
        // scalar forms, this pins SIMD == scalar on every width incl. tails.
        if !avx512() {
            eprintln!("skip: no AVX512F/DQ on this host");
            return;
        }
        let mut st = 0x51ce_f16du32;
        let mut lcg = move || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            st
        };
        let bits = |g: &mut dyn FnMut() -> u32| -> u16 {
            // spread f16 patterns over the normal range with some subnormals
            let r = g();
            (if r % 16 == 0 { r % 0x0400 } else { 0x3000 + r % 0x2400 }) as u16
        };
        for &n in &[1usize, 2, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 100, 128, 130] {
            let x: Vec<u16> = (0..n).map(|_| bits(&mut lcg)).collect();
            let y: Vec<u16> = (0..n).map(|_| bits(&mut lcg)).collect();
            // dot
            let xf: Vec<half::f16> = x.iter().map(|&b| half::f16::from_bits(b)).collect();
            let yf: Vec<half::f16> = y.iter().map(|&b| half::f16::from_bits(b)).collect();
            let want = crate::vec_dot::vec_dot_f16_c(n, &xf, &yf);
            // SAFETY: `x`/`y` hold `n` u16 f16 bit patterns each.
            let got = unsafe { vec_dot_f16(n, x.as_ptr(), y.as_ptr()) };
            assert_eq!(got.to_bits(), want.to_bits(), "vec_dot_f16 n={n}");
            // mad: y += x*v (scalar reference = the fused per-element form)
            let v = (lcg() as i32 as f32 / (1u32 << 28) as f32) * 3.0;
            let mut ym = yf.clone();
            let mut ym2 = yf.clone();
            for (ym, &xv) in ym.iter_mut().zip(&xf) {
                *ym = half::f16::from_f32(xv.to_f32().mul_add(v, ym.to_f32()));
            }
            // SAFETY: `ym2`/`x` hold `n` f16 values.
            unsafe { vec_mad_f16(ym2.as_mut_ptr() as *mut u16, x.as_ptr(), v, n) };
            for i in 0..n {
                assert_eq!(ym2[i].to_bits(), ym[i].to_bits(), "vec_mad_f16 n={n} i={i}");
            }
            // scale: y *= v
            let vs = (lcg() as i32 as f32 / (1u32 << 28) as f32) * 2.0;
            let mut ys = yf.clone();
            let mut ys2 = yf.clone();
            for ys in ys.iter_mut() {
                *ys = half::f16::from_f32(ys.to_f32() * vs);
            }
            // SAFETY: `ys2` holds `n` f16 values.
            unsafe { vec_scale_f16(ys2.as_mut_ptr() as *mut u16, vs, n) };
            for i in 0..n {
                assert_eq!(ys2[i].to_bits(), ys[i].to_bits(), "vec_scale_f16 n={n} i={i}");
            }
        }
    }

    #[test]
    fn avx512_silu_matches_scalar_bit_exact() {
        assert!(avx512(), "test host must have AVX512F+DQ");
        for n in [0usize, 1, 15, 16, 17, 31, 48, 896, 4864] {
            for seed in [1u32, 2, 3] {
                let x = f32_sweep(n, seed); // length n + 10 (specials appended)
                let mut y = vec![0f32; x.len()];
                vec_silu_f32(&mut y, &x);
                // the 16-lane chunks run the v512-polynomial lane form; the
                // tail runs vec.h's scalar ggml_silu_f32 with libm expf — a
                // different function (1 ulp apart on ~a quarter of inputs;
                // pinned by the batch-15 MoE node dump: every diverging
                // ffn_moe_swiglu element sat at k >= 16 of an n_ff_exp 24 row)
                let n_chunk = (x.len() / 16) * 16;
                for (i, (&got, &want)) in y.iter().zip(&x).enumerate() {
                    let oracle = if i < n_chunk {
                        crate::ops::ggml_silu_f32(want) // the v512 lane form
                    } else {
                        crate::ops::ggml_silu_scalar_f32(want) // libm expf tail
                    };
                    assert_eq!(
                        got.to_bits(),
                        oracle.to_bits(),
                        "silu n={n} seed={seed} i={i}: {got} vs {oracle}"
                    );
                }
            }
        }
    }

    /// `ggml_vec_soft_max_f32` (vec.cpp:531): the AVX512 chunk path (v_expf +
    /// `_mm512_reduce_add_ps` into the f64 sum, libm `expf` tail) must be
    /// bit-identical to the elementwise lane-port of the same structure.
    #[test]
    fn avx512_soft_max_matches_scalar_bit_exact() {
        assert!(avx512(), "test host must have AVX512F+DQ");
        for n in [1usize, 15, 16, 17, 31, 33, 64, 896] {
            for seed in [5u32, 6] {
                let x = f32_sweep(n, seed); // length n + 10 (specials appended)
                let n = x.len();
                let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut y = vec![0f32; n];
                let got = vec_soft_max_f32(&mut y, &x, max);
                // elementwise reference: same chunking, same reduce tree
                let mut want = vec![0f32; n];
                let mut sum = 0f64;
                let mut i = 0;
                while i + 16 <= n {
                    let mut chunk = [0f32; 16];
                    for l in 0..16 {
                        chunk[l] = crate::ops::ggml_expf_v512(x[i + l] - max);
                    }
                    want[i..i + 16].copy_from_slice(&chunk);
                    sum += vec_dot::reduce_add16(&chunk) as f64;
                    i += 16;
                }
                while i < n {
                    let e = (x[i] - max).exp();
                    want[i] = e;
                    sum += e as f64;
                    i += 1;
                }
                assert_eq!(got.to_bits(), sum.to_bits(), "soft_max sum n={n} seed={seed}");
                for (i, (a, b)) in y.iter().zip(&want).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "soft_max y n={n} seed={seed} i={i}");
                }
            }
        }
    }

    /// `ggml_vec_scale_f32` / `ggml_vec_add_f32` / `ggml_vec_mul_f32`: one
    /// rounded op per lane — any width is bit-identical to the scalar loop.
    #[test]
    fn avx512_lane_ops_match_scalar_bit_exact() {
        assert!(avx512(), "test host must have AVX512F+DQ");
        for n in [0usize, 1, 15, 16, 17, 896, 4864] {
            let x = f32_sweep(n, 7); // length n + 10 (specials appended)
            let y = f32_sweep(n, 8);
            let n = x.len();
            let s = 1.7320508f32;
            let mut z1 = x.clone();
            vec_scale_f32(&mut z1, s);
            for i in 0..n {
                assert_eq!(z1[i].to_bits(), (x[i] * s).to_bits(), "scale n={n} i={i}");
            }
            let mut z2 = vec![0f32; n];
            vec_add_f32(&mut z2, &x, &y);
            let mut z3 = vec![0f32; n];
            vec_mul_f32(&mut z3, &x, &y);
            for i in 0..n {
                assert_eq!(z2[i].to_bits(), (x[i] + y[i]).to_bits(), "add n={n} i={i}");
                assert_eq!(z3[i].to_bits(), (x[i] * y[i]).to_bits(), "mul n={n} i={i}");
            }
        }
    }

    /// `simd_gemm` (simd-gemm.h:24-131): the AVX512 register-blocked
    /// microkernels and their scalar tails must be bit-identical to the
    /// scalar `#else` body (per-element FMA chain over `kk` ascending) on
    /// every tile split — including M/N not divisible by the block sizes.
    #[test]
    fn avx512_gemm_matches_scalar_bit_exact() {
        assert!(avx512(), "test host must have AVX512F+DQ");
        let gemm_scalar = |c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize| {
            for i in 0..m {
                for kk in 0..k {
                    let av = a[i * k + kk];
                    for j in 0..n {
                        c[i * n + j] = av.mul_add(b[kk * n + j], c[i * n + j]);
                    }
                }
            }
        };
        for (m, k, n) in [
            (1usize, 1, 1),
            (3, 5, 7),
            (4, 16, 64),
            (5, 16, 64),   // M tail row
            (8, 3, 17),    // N scalar tail
            (16, 33, 33),
            (64, 64, 64),  // the FA KQ/VKQ shapes
            (64, 64, 128),
            (13, 100, 80),
        ] {
            for seed in [11u32, 12] {
                let a = f32_sweep(m * k, seed).into_iter().map(|v| v * 0.25).collect::<Vec<_>>();
                let b = f32_sweep(k * n, seed + 1).into_iter().map(|v| v * 0.25).collect::<Vec<_>>();
                let c0 = f32_sweep(m * n, seed + 2).into_iter().map(|v| v * 0.5).collect::<Vec<_>>();
                let mut c1 = c0.clone();
                let mut c2 = c0.clone();
                simd_gemm_avx512(&mut c1, &a, &b, m, k, n);
                gemm_scalar(&mut c2, &a, &b, m, k, n);
                for (i, (x, y)) in c1.iter().zip(&c2).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "gemm m={m} k={k} n={n} seed={seed} i={i}");
                }
            }
        }
    }
}
