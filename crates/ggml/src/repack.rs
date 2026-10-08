//! repack.rs — CPU_REPACK port for MXFP4 (`GGML_USE_CPU_REPACK`).
//!
//! Reference: ggml/src/ggml-cpu/repack.cpp (5253 lines) +
//! ggml/src/ggml-cpu/arch/x86/repack.cpp (6407 lines), pinned bd4f514db1.
//! Build context: the reference binary is `GGML_NATIVE=ON` +
//! `GGML_USE_CPU_REPACK=ON` on AVX512F/BW/DQ/VNNI + AVX2, so MXFP4 tensors it
//! loads take exactly one path:
//!
//! ```text
//!   ggml_repack_get_optimal_repack_type (repack.cpp:5093) -> mxfp4_8x8_q8_0
//!     (ggml_cpu_has_avx2() && ne[1] % 8 == 0)
//!   -> the tensor lives in the CPU_REPACK buffer, whose `set_tensor`
//!      (repack.cpp:5150) runs repack_mxfp4_to_mxfp4_8_bl (repack.cpp:4202)
//!      -> make_block_mxfp4x8 (repack.cpp:4178)  [8x8 interleave, 136 B blocks]
//!   -> MUL_MAT      (repack.cpp:4650 forward_mul_mat)      -> gemm/gemv
//!      MUL_MAT_ID   (repack.cpp:4783 forward_mul_mat_id)   -> gemv (nr == 1)
//!   -> ggml_gemv_mxfp4_8x8_q8_0 (arch/x86/repack.cpp:1700)
//!      ggml_gemm_mxfp4_8x8_q8_0 (arch/x86/repack.cpp:3511)
//!      both tie into gemv|gemm_q4_b32_8x8_q8_0_lut_avx
//!      (arch/x86/repack.cpp:522 / 641) with the mxfp4 sign-extend LUT
//!      `kvalues_mxfp4` and the E8M0 "half" scale.
//! ```
//!
//! ## Arithmetic structure of the 8x8 kernels (what makes parity possible)
//!
//! For one output element (weight row `r`, activation row `a`) and one 32-element
//! k-block `b`, the kernels compute
//!
//! ```text
//!   acc = fma(f32(Σ_k LUT[nibble(w[r,k,b])] * i8(a[k,b])),   // exact int32
//!             e8m0_to_fp32_half(e[r,b]) * f32(a.d[b]),        // exact (pow2 * fp16)
//!             acc)
//! ```
//!
//! with `b` ascending (one fma per k-block, `iacc` is zeroed per block:
//! arch/x86/repack.cpp:610-629 for gemv, 867-870 for gemm). Consequences, all
//! verified bit-for-bit against the reference in the `ref_parity` tests below:
//!
//!   * the integer dot is computed exactly (all 32 products in int32: |max| =
//!     12*127*32 = 48768 < 2^24, so `cvtepi32_ps` is exact too);
//!   * the scale product rounds once, then a *fused* multiply-add accumulates it;
//!     the SIMD lane permutation (columns `[0,4,1,5,2,6,3,7]`, see
//!     arch/x86/repack.cpp:614-626) only decides *which* lane holds which row, so
//!     a scalar re-implementation reproduces the reference's f32 result exactly.
//!
//! The `_generic` C kernels (repack.cpp:1236 / 2363) are the arch-fallback scalar
//! versions and are NOT what the reference runs; they accumulate
//! `sumf[j] += d * (v0*a0 + v1*a1) * e` per element, i.e. a different rounding
//! order (and indeed they disagree with the AVX kernels, see
//! `parity/ref_repack_dump.cpp`'s own report).
//!
//! ## What is ported / not ported (x86_64 view of `ggml_repack_get_optimal_repack_type`,
//! repack.cpp:4925-5140)
//!
//! * **MXFP4 8x8** (AVX2 gate, repack.cpp:5093-5099) — ported below.
//! * **Q4_K 8x8** (AVX2 gate, repack.cpp:5006-5011) — ported below (this file's
//!   Q4_K section): `repack_q4_K_8_bl` layout, AVX2 gemv + gemm kernels, and
//!   the `compute.rs` wiring for both 2D `mul_mat` (gemm for whole 4-row
//!   groups + gemv tails, repack.cpp:4638-4647) and 3D `mul_mat_id` (gemv
//!   only, repack.cpp:4893-4908).
//! * **Q2_K 8x8** (AVX512 gate, repack.cpp:5028-5036) — NOT ported. This host
//!   has AVX512, so the reference *would* repack Q2_K; no local model carries
//!   Q2_K tensors, so the row-wise `vec_dot_q2_K_q8_K` path is what the port
//!   runs and the divergence is unreachable on this machine's models.
//! * **Q4_0 8x8** (AVX2 gate, repack.cpp:4987-4993) — ported (this file's
//!   Q4_0 section): `repack_q4_0_to_q4_0_8_bl` layout, AVX2 gemv + gemm
//!   kernels (the shared `gemv|gemm_q4_b32_8x8_q8_0_lut_avx` templates), and
//!   the `compute.rs` wiring for both 2D `mul_mat` and 3D `mul_mat_id`. The
//!   intercept happens *before* llamafile tinyBLAS
//!   (`ggml_compute_forward` runs `ggml_cpu_extra_compute_forward`,
//!   ggml-cpu.c:1751-1753), so a Q4_0 model (e.g. gemma-4-*-QAT) now takes
//!   the same route as the reference for every qualifying tensor (n=1 decode
//!   included) — closing the recorded bit-level routing divergence.
//! * **IQ4_NL 8x8** (repack.cpp:5072-5078) — NOT ported: the instance exists
//!   on this host but the port has no IQ4_NL quantizer/vec_dot at all (the
//!   type cannot appear as a weight), and no local model carries IQ4_NL
//!   tensors (swept every local GGUF), so the divergence is unreachable here.
//! * **Q5_K / Q6_K** — no x86 instance at all (repack.cpp:5050-5071: both
//!   gates are `neon && matmul_int8/dotprod`), so the reference runs its
//!   row-wise `vec_dot` and the port does the same (`vec_dot.rs`, SIMD);
//!   `vec_dot.rs::kquant_real_tensor_tests` pins this on real tensors.
//! * `nrows == 2` multi-row dots (`type_traits_cpu[].nrows`,
//!   ggml-cpu.c:243-348) are NEON-only (`__ARM_FEATURE_MATMUL_INT8`); on x86
//!   every type has `nrows == 1`, so the port's one-row-at-a-time `vec_dot`
//!   loop matches the reference.
//!
//! Coverage consequence for the local GGUFs: every Q4_K/MXFP4 tensor in them
//! has `ne[1] % 8 == 0` (measured: qwen2.5-0.5b/7b, gpt-oss-20b Q4_K_M and
//! MXFP4, LFM2-8B-A1B incl. the 56 3D expert tensors, Qwen3.6-35B MoE,
//! granite's ssm tensors), i.e. the port's coverage equals the reference's.
//!
//! * f16/bf16 tinyBLAS-style repack variants: N/A for MXFP4.

use crate::blocks::{BlockQ8_0, QK8_0, QK_MXFP4};
use crate::quants::{e8m0_to_fp32_half, KVALUES_MXFP4};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// `sizeof(block_mxfp4)` (ggml-common.h:214-219)
pub const BLOCK_MXFP4_SIZE: usize = QK_MXFP4 / 2 + 1; // 17
/// `sizeof(block_mxfp4x8)` (repack.h:135-139) — one 8-row x 1-block tile
pub const BLOCK_MXFP4X8_SIZE: usize = 8 + QK_MXFP4 * 4; // 136
/// `sizeof(block_q8_0)` (ggml-common.h) — the PARAM_TYPE of the mxfp4 traits
pub const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0; // 34
/// `sizeof(block_q8_0x4)` (repack.h:44, block<8,4>) — gemm's interleaved Lhs
pub const BLOCK_Q8_0X4_SIZE: usize = 4 * 2 + QK8_0 * 4; // 136
/// `nrows_interleaved` (repack.cpp:4205) == `ncols_interleaved` (repack.h:135)
pub const NROWS_INTERLEAVED: usize = 8;
/// `NB_COLS` — the tensor_traits NB_COLS template arg of mxfp4_8x8_q8_0
pub const NB_COLS: usize = 8;
/// `INTER_SIZE` — the tensor_traits INTER_SIZE template arg of mxfp4_8x8_q8_0
/// (repack.cpp:4956). It is also the gemm Lhs interleave granularity
/// (`ggml_quantize_mat_t<INTER_SIZE, _>`, repack.cpp:325) — see
/// [`quantize_mat_q8_0_4x8`].
pub const INTER_SIZE: usize = 8;

/// `ggml_repack_get_optimal_repack_type` (repack.cpp:5093-5099) for MXFP4 on
/// x86_64: repack iff `ggml_cpu_has_avx2() && ne[1] % 8 == 0`. `ne1` here is
/// `tensor->ne[1]` (dim 1 of the weight), not `ggml_nrows` — for a 3D MoE tensor
/// `ne[1]` is the per-expert row count.
///
/// `repack_mxfp4_to_mxfp4_8_bl` additionally requires `ne[0] % QK_MXFP4 == 0`
/// (implicitly true for any MXFP4 tensor: `ne[0] % blck_size == 0` is enforced by
/// ggml's tensor constructor) — asserted here anyway.
pub fn repack_supported_mxfp4(ne1: i64, ne0: i64) -> bool {
    ne0 % QK_MXFP4 as i64 == 0 && ne1 % NROWS_INTERLEAVED as i64 == 0
}

/// `repack.cpp:4178 make_block_mxfp4x8` — interleave eight `block_mxfp4`s
/// (rows `in[0..8]` of the same k-block) into one 136-byte `block_mxfp4x8`.
///
/// Layout: `e[8]` = the 8 per-row E8M0 bytes, then `qs[128]` where
/// `qs[8*i .. 8*i+8] = in[i % 8].qs[(i / 8) * 8 .. (i / 8) * 8 + 8]`
/// (`blck_size_interleave == 8`). I.e. rows 0..7 of the *first* half of the
/// 16 nibble bytes (elements 0..15) first, then rows 0..7 of the second half
/// (elements 16..31). `src_id = i % 8` is the **row** index — the interleave is
/// over rows, not over nibbles.
pub fn make_block_mxfp4x8(in8: &[u8], out: &mut [u8]) {
    debug_assert_eq!(in8.len(), 8 * BLOCK_MXFP4_SIZE);
    debug_assert_eq!(out.len(), BLOCK_MXFP4X8_SIZE);
    let (out_e, out_qs) = out.split_at_mut(8);
    for i in 0..8 {
        out_e[i] = in8[i * BLOCK_MXFP4_SIZE];
    }
    // C: for (i = 0; i < end; ++i) { src_id = i % 8; src_offset = (i / 8) * 8;
    //                                dst_offset = i * 8; memcpy(8); }  end = 32*4/8 = 16
    for i in 0..(QK_MXFP4 * 4 / 8) {
        let src_id = i % 8;
        let src_offset = (i / 8) * 8;
        let dst_offset = i * 8;
        let src = &in8[src_id * BLOCK_MXFP4_SIZE + 1 + src_offset..src_id * BLOCK_MXFP4_SIZE + 1 + src_offset + 8];
        out_qs[dst_offset..dst_offset + 8].copy_from_slice(src);
    }
}

/// `repack.cpp:4202 repack_mxfp4_to_mxfp4_8_bl` — whole-tensor repack.
///
/// `src` is the plain row-major MXFP4 data (`nrows * n_per_row/32 * 17` bytes);
/// rows are grouped in 8s (`src += nrows_interleaved * nblocks` per group) and
/// within a group the tiles are written in k-block order. The output is
/// byte-identical in size, so the repacked tensor keeps the plan row stride
/// `nb[1] = nblocks * 17` (`ggml_backend_cpu_repack_buffer_type` has a NULL
/// `get_alloc_size`, i.e. `ggml_nbytes`), which is what lets the kernels address
/// 8-row groups as `src0 + row/8 * nb1 * 8`.
pub fn repack_mxfp4_8x8_into(src: &[u8], nrows: usize, n_per_row: usize, dst: &mut [u8]) {
    let nb = n_per_row / QK_MXFP4;
    let row = nb * BLOCK_MXFP4_SIZE;
    assert_eq!(n_per_row % QK_MXFP4, 0, "repack: ne[0] % 32 != 0");
    assert_eq!(nrows % NROWS_INTERLEAVED, 0, "repack: ne[1] % 8 != 0");
    assert!(src.len() >= nrows * row, "repack: src too small");
    assert!(dst.len() >= nrows * row, "repack: dst too small");
    debug_assert!(src.len() == nrows * row && dst.len() == nrows * row);

    for g in 0..nrows / NROWS_INTERLEAVED {
        let sbase = g * NROWS_INTERLEAVED * row;
        let dbase = sbase; // same total size per group: 8 rows x nb x 17
        for x in 0..nb {
            let tile = &mut dst[dbase + x * BLOCK_MXFP4X8_SIZE..dbase + (x + 1) * BLOCK_MXFP4X8_SIZE];
            let mut tmp = [0u8; 8 * BLOCK_MXFP4_SIZE];
            for i in 0..NROWS_INTERLEAVED {
                // dst_tmp[i] = src[x + i*nblocks]  (row i of this group, block x)
                let so = sbase + i * row + x * BLOCK_MXFP4_SIZE;
                tmp[i * BLOCK_MXFP4_SIZE..(i + 1) * BLOCK_MXFP4_SIZE]
                    .copy_from_slice(&src[so..so + BLOCK_MXFP4_SIZE]);
            }
            make_block_mxfp4x8(&tmp, tile);
        }
    }
}

/// `repack_mxfp4_8x8_into` into a fresh buffer.
pub fn repack_mxfp4_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let len = nrows * (n_per_row / QK_MXFP4) * BLOCK_MXFP4_SIZE;
    let mut out = vec![0u8; len];
    repack_mxfp4_8x8_into(&src[..len], nrows, n_per_row, &mut out);
    out
}

/// Inverse of [`repack_mxfp4_8x8`] (used by the round-trip test and debugging;
/// the reference has no such function).
pub fn unrepack_mxfp4_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let nb = n_per_row / QK_MXFP4;
    let row = nb * BLOCK_MXFP4_SIZE;
    assert_eq!(nrows % NROWS_INTERLEAVED, 0);
    let mut out = vec![0u8; nrows * row];
    for g in 0..nrows / NROWS_INTERLEAVED {
        let base = g * NROWS_INTERLEAVED * row;
        for x in 0..nb {
            let tile = &src[base + x * BLOCK_MXFP4X8_SIZE..base + (x + 1) * BLOCK_MXFP4X8_SIZE];
            for i in 0..NROWS_INTERLEAVED {
                let so = base + i * row + x * BLOCK_MXFP4_SIZE;
                out[so] = tile[i];
            }
            for i in 0..(QK_MXFP4 * 4 / 8) {
                let src_id = i % 8;
                let src_offset = (i / 8) * 8;
                let dst_offset = i * 8;
                let so = base + src_id * row + x * BLOCK_MXFP4_SIZE + 1 + src_offset;
                out[so..so + 8].copy_from_slice(&tile[8 + dst_offset..8 + dst_offset + 8]);
            }
        }
    }
    out
}

/// `ggml_quantize_mat_q8_0_4x4` (x86 form ~arch/x86/repack.cpp:201+ [repack.h:145], generic form
/// repack.cpp:135) — quantize 4 activation rows into `block_q8_0x4` tiles,
/// interleave granularity 4 (`blck_size_interleave = 4`).
///
/// The *values* are plain `quantize_row_q8_0` values (the x86 kernel uses
/// `id = 127/amax`, `_mm256_round_ps(_MM_ROUND_NEAREST)`, `d = amax/127`, which
/// is exactly `crate::quants::quantize_row_q8_0`); only the storage is
/// interleaved: `qs[j] = row[src_id].qs[src_offset]` with
/// `src_id = (j % (4*i)) / i`, `src_offset = (j / (4*i)) * i + j % i`.
pub fn quantize_mat_q8_0_4x4(x: &[f32], k: usize, n_rows: usize, out: &mut [u8]) {
    quantize_mat_q8_0_4xn(x, k, n_rows, out, 4);
}

/// `ggml_quantize_mat_q8_0_4x8` (arch/x86/repack.cpp:178, generic form
/// repack.cpp:173) — same 136-byte `block_q8_0x4` tile, interleave granularity 8.
///
/// **This is the one the 8x8 traits use**: `forward_mul_mat` calls
/// `ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>` (repack.cpp:4698) and
/// `INTER_SIZE == NB_COLS == 8` for `mxfp4_8x8_q8_0` (repack.cpp:4956,
/// specialization at repack.cpp:325), so the gemm's Lhs is the *4x8* layout
/// while the gemv's is a plain `block_q8_0` row.
pub fn quantize_mat_q8_0_4x8(x: &[f32], k: usize, n_rows: usize, out: &mut [u8]) {
    quantize_mat_q8_0_4xn(x, k, n_rows, out, 8);
}

fn quantize_mat_q8_0_4xn(x: &[f32], k: usize, n_rows: usize, out: &mut [u8], interleave: usize) {
    assert_eq!(k % QK8_0, 0, "quantize_mat_q8_0_4xN: k % 32 != 0");
    assert_eq!(n_rows % 4, 0, "quantize_mat_q8_0_4xN: rows must be a multiple of 4");
    let nb = k / QK8_0;
    let mut q8 = vec![0u8; 4 * nb * BLOCK_Q8_0_SIZE];
    for r in 0..n_rows / 4 {
        let xrows = &x[r * 4 * k..(r + 1) * 4 * k];
        let dst = &mut out[r * nb * BLOCK_Q8_0X4_SIZE..(r + 1) * nb * BLOCK_Q8_0X4_SIZE];
        crate::quants::quantize_row_q8_0(xrows, bytemuck::cast_slice_mut(&mut q8[..4 * nb * BLOCK_Q8_0_SIZE]));
        for ib in 0..nb {
            let (d, qs) = dst[ib * BLOCK_Q8_0X4_SIZE..(ib + 1) * BLOCK_Q8_0X4_SIZE].split_at_mut(8);
            for rr in 0..4 {
                let src = &q8[(rr * nb + ib) * BLOCK_Q8_0_SIZE..];
                d[rr * 2..rr * 2 + 2].copy_from_slice(&src[..2]);
            }
            for j in 0..QK8_0 * 4 {
                let src_id = (j % (4 * interleave)) / interleave;
                let src_offset = (j / (4 * interleave)) * interleave + (j % interleave);
                let src = &q8[(src_id * nb + ib) * BLOCK_Q8_0_SIZE + 2..];
                qs[j] = src[src_offset];
            }
        }
    }
}


// ---------------------------------------------------------------------------
// AVX2 lane kernels (arch/x86/repack.cpp:522 gemv / :641 gemm)
// ---------------------------------------------------------------------------

/// The reference build (`GGML_NATIVE=ON`, AVX512F/BW/DQ/VNNI + AVX2) runs
/// `gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>` with the AVX2 body verbatim
/// (`ggml_gemv_mxfp4_8x8_q8_0` gates on `__AVX2__` only, arch/x86/repack.cpp:1700)
/// and `gemm_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>` with its AVX512BW/DQ body
/// (arch/x86/repack.cpp:3511 → :663-1096). Reproducing the *lane networks* of
/// those kernels needs SIMD, and AVX512 intrinsics are unstable in Rust, so:
///
///   * [`gemv_mxfp4_8x8_q8_0`] is a 1:1 port of the reference's AVX2 gemv body:
///     same shuffle network, same `iacc` grouping, same per-block
///     `fma(cvtepi32_ps(iacc), col_scale*row_scale, acc)`, same
///     `permutevar8x32_ps` store order.
///   * the gemm shares that column network, 4 activation rows at a time
///     (`acc_rows[4]`), each row's lhs built straight from the 4x8-interleaved
///     tile (`qs[(el/8)*32 + row*8 + el%8]`, wrapped into the gemv's
///     `[A(row,0-15) | A(row,0-15)]` layout).
///
/// Bit-exactness of that choice is structural, not empirical: every output
/// element of *every* variant of these kernels is
/// `acc = fma(f32(Σ_32 LUT[w]*q8), e8m0_half*row_d, acc)` once per k-block, with
/// the Σ in exact int32 (the lane regrouping only assigns which lane holds which
/// element — the C AVX512 gemm sums the same 32 products of the same element
/// pairs, just laid out over 2x2 lanes). The verified reference dump
/// (`gemv|gemm_matches_reference_dump_bitexact`) pins this.
///
/// `_mm256_dpbusd_epi32` (the reference's `__AVX512VNNI__ && __AVX512VL__`
/// shortcut, arch/x86/repack.cpp:153) is unavailable from stable Rust; the
/// `maddubs+madd` form used here is value-identical because |LUT|*|q8| ≤ 1524
/// keeps the pairwise int16 sum within range (≤ 3048 > −32768/32767), so the
/// saturating maddubs never saturates.
#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use super::{
        e8m0_to_fp32_half, BlockQ8_0, BLOCK_MXFP4X8_SIZE, BLOCK_Q8_0X4_SIZE, BLOCK_Q8_0_SIZE,
        KVALUES_MXFP4, NROWS_INTERLEAVED, QK8_0,
    };
    use core::arch::x86_64::*;

    /// `mul_sum_i8_pairs_acc_int32x8` (`arch/x86/repack.cpp:165`), non-VNNI form:
    /// `acc + Σ_{4 bytes per dword} x*y`, the products exact in int32.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_i8_pairs_acc_int32x8(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(x, x); // |x| carrying x's sign (LUT has no -128)
        let sy = _mm256_sign_epi8(y, x); // y with x's sign
        let dot = _mm256_maddubs_epi16(ax, sy); // u8*i8 pairwise -> i16
        _mm256_add_epi32(acc, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// The VNNI form the reference build compiles on this host (its inner loop
    /// is 8× `vpdpbusd`, no `vpmaddubsw`/`vpmaddwd` — see parity/asm/
    /// ref_mxfp4_gemv_loop.asm): `acc + Σ_{4/dword} u·s` with `u` unsigned and
    /// `s` signed. Value-identical to the maddubs+madd pair for every operand
    /// this kernel feeds it: `u` = |LUT| ≤ 12, `s` = ±q8 ≤ 127, so the i16
    /// pair sums stay ≤ 2·12·127 = 3048 < 32767 — `maddubs`' saturation cannot
    /// engage and the i32 dword sums are exact in both spellings (the same
    /// proof tinyblas.rs's `vnni_lane` carries for the Q0 class).
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn mul_sum_i8_pairs_acc_int32x8_vnni(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(acc, _mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x))
    }

    /// The 8 shuffle/blend triples of the gemv inner body (arch/x86/repack.cpp:616-626):
    /// rhs operand + the matching lhs dword broadcast.
    /// `lhs0` = `[A(0-15) | A(0-15)]`, `lhs1` = `[A(16-31) | A(16-31)]`.
    /// Generated once per `mul_sum` flavor so the VNNI `vpdpbusd` spelling gets
    /// its own target-feature gate (an intrinsic's enclosing fn must carry the
    /// feature); the two bodies are otherwise token-identical.
    macro_rules! mxfp4_iacc_row {
        ($(#[$attr:meta])* $name:ident, $msum:ident) => {
            $(#[$attr])*
            #[inline]
            unsafe fn $name(
                lut: __m256i,
                m4b: __m256i,
                tile: *const u8,
                lhs0: __m256i,
                lhs1: __m256i,
            ) -> __m256i {
        // `block_mxfp4x8` = {e[8], qs[128]} — the C kernels index `b_ptr[b].qs`,
        // i.e. the tile payload starts 8 bytes into the block (repack.h:135).
        let tile = tile.add(8);
        let raw_0123_0 = _mm256_loadu_si256(tile as *const __m256i);
        let raw_4567_0 = _mm256_loadu_si256(tile.add(32) as *const __m256i);
        let raw_0123_1 = _mm256_loadu_si256(tile.add(64) as *const __m256i);
        let raw_4567_1 = _mm256_loadu_si256(tile.add(96) as *const __m256i);
        // 4-bit -> 8-bit, sign maintained (pshufb through the i8 LUT)
        let r0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, m4b));
        let r4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, m4b));
        let r0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, m4b));
        let r4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, m4b));
        let r0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_0, 4), m4b));
        let r4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_0, 4), m4b));
        let r0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_1, 4), m4b));
        let r4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_1, 4), m4b));

        let mut iacc = _mm256_setzero_si256();
        // lane order after accumulation: B0 B4 B1 B5 B2 B6 B3 B7
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(r0123_0, _mm256_shuffle_epi32::<177>(r4567_0)),
            _mm256_shuffle_epi32::<0>(lhs0),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_0), r4567_0),
            _mm256_shuffle_epi32::<85>(lhs0),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(r0123_1, _mm256_shuffle_epi32::<177>(r4567_1)),
            _mm256_shuffle_epi32::<170>(lhs0),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_1), r4567_1),
            _mm256_shuffle_epi32::<255>(lhs0),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(r0123_2, _mm256_shuffle_epi32::<177>(r4567_2)),
            _mm256_shuffle_epi32::<0>(lhs1),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_2), r4567_2),
            _mm256_shuffle_epi32::<85>(lhs1),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(r0123_3, _mm256_shuffle_epi32::<177>(r4567_3)),
            _mm256_shuffle_epi32::<170>(lhs1),
        );
        iacc = $msum(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_3), r4567_3),
            _mm256_shuffle_epi32::<255>(lhs1),
        );
        iacc
            }
        };
    }

    mxfp4_iacc_row!(
        #[target_feature(enable = "avx2")]
        iacc_row,
        mul_sum_i8_pairs_acc_int32x8
    );
    mxfp4_iacc_row!(
        #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
        iacc_row_vnni,
        mul_sum_i8_pairs_acc_int32x8_vnni
    );



    /// `col_scale_f32` of the mxfp4 branch (arch/x86/repack.cpp:576-586):
    /// `set_ps(e7,e3,e6,e2,e5,e1,e4,e0)` — the (B0,B4,B1,B5,B2,B6,B3,B7) lane order.
    #[inline]
    unsafe fn col_scale_mxfp4(e: *const u8) -> __m256 {
        let g = |i: usize| e8m0_to_fp32_half(*e.add(i));
        _mm256_set_ps(
            g(7),
            g(3),
            g(6),
            g(2),
            g(5),
            g(1),
            g(4),
            g(0),
        )
    }

    /// The gemv's lhs pair for one `block_q8_0` row (arch/x86/repack.cpp:597-604).
    #[inline]
    unsafe fn lhs_pair(qs: *const u8) -> (__m256i, __m256i) {
        let lo = _mm_loadu_si128(qs as *const __m128i);
        let hi = _mm_loadu_si128(qs.add(16) as *const __m128i);
        let lhs0 = _mm256_broadcastsi128_si256(lo); // [A(0-15) | A(0-15)]
        let lhs1 = _mm256_broadcastsi128_si256(hi); // [A(16-31) | A(16-31)]
        (lhs0, lhs1)
    }

    /// Row `row` (0..4) of a `block_q8_0x4` qs payload in the gemv lhs layout:
    /// element `el` sits at `qs[(el/8)*32 + row*8 + el%8]`
    /// (inverse of [`super::quantize_mat_q8_0_4x8`]'s store loop).
    #[inline]
    unsafe fn lhs_pair_q8_0x4(qs: *const u8, row: usize) -> (__m256i, __m256i) {
        let a = _mm_loadl_epi64(qs.add(row * 8) as *const __m128i);
        let b = _mm_loadl_epi64(qs.add(32 + row * 8) as *const __m128i);
        let c = _mm_loadl_epi64(qs.add(64 + row * 8) as *const __m128i);
        let d = _mm_loadl_epi64(qs.add(96 + row * 8) as *const __m128i);
        let lo = _mm_unpacklo_epi64(a, b); // A(row,0-15)
        let hi = _mm_unpacklo_epi64(c, d); // A(row,16-31)
        (_mm256_broadcastsi128_si256(lo), _mm256_broadcastsi128_si256(hi))
    }

    /// `GGML_CPU_FP16_TO_FP32` for one `block_q8_0.d` / `block_q8_0x4.d[i]`.
    /// The `vcvtph2ps` hardware instruction — value-identical to the `half`
    /// crate's bit algorithm for all 65536 bit patterns (pinned exhaustively
    /// by `simd_x86`'s `f16c_cvtph_matches_portable`; the reference build gets
    /// the same instruction from its `__F16C__` GGML_CPU_FP16_TO_FP32). The
    /// `half::f16::to_f32` this replaces carries a runtime F16C detection
    /// tree per call — one call per k-block, ~32k calls per gemv on the
    /// gpt-oss expert shape.
    #[inline]
    #[target_feature(enable = "f16c")]
    unsafe fn d_f16(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
            u16::from_le_bytes([*p, *p.add(1)]) as i32,
        )))
    }

    #[inline]
    unsafe fn load_lut() -> __m256i {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i))
    }

    /// AVX2 body of `gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>`
    /// (arch/x86/repack.cpp:522-637), `nr == 1` (all C call sites).
    /// Instantiated twice from one body — plain AVX2 (VEX, 16 ymm) and the
    /// EVEX flavor (`avx512f/vl/bw/dq`: the extended ymm16-31 register file,
    /// which is the instruction selection the `-march=native` reference build
    /// gets on this host). Same intrinsics, same op order ⇒ bit-identical by
    /// construction (the q6_K / tinyBLAS dual instantiations are the
    /// precedent); the gemv body holds ~25 live vectors and spills most of
    /// them in the 16-register VEX build.
    macro_rules! mxfp4_gemv_body {
        ($(#[$attr:meta])* $name:ident, $iacc:ident) => {
            $(#[$attr])*
            pub unsafe fn $name(
                n: usize,
                s: *mut f32,
                vx: *const u8,
                vy: *const u8,
                nc: usize,
            ) {
                let nb = n / QK8_0;
                let b_nb = n / 32;
                let lut = load_lut();
                let m4b = _mm256_set1_epi8(0x0F);
                let finalpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                for x in 0..nc / NROWS_INTERLEAVED {
                    let b_ptr = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                    let mut acc = _mm256_setzero_ps();
                    for b in 0..nb {
                        let tile = b_ptr.add(b * BLOCK_MXFP4X8_SIZE);
                        let a_blk = vy.add(b * BLOCK_Q8_0_SIZE);
                        let (lhs0, lhs1) = lhs_pair(a_blk.add(2));
                        let iacc = $iacc(lut, m4b, tile, lhs0, lhs1);
                        let d = d_f16(a_blk);
                        acc = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc),
                            _mm256_mul_ps(col_scale_mxfp4(tile), _mm256_set1_ps(d)),
                            acc,
                        );
                    }
                    let out = _mm256_permutevar8x32_ps(acc, finalpermute);
                    _mm256_storeu_ps(s.add(x * 8), out);
                }
            }
        };
    }

    mxfp4_gemv_body!(
        #[target_feature(enable = "avx2,fma,f16c")]
        gemv_vex,
        iacc_row
    );
    mxfp4_gemv_body!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq")]
        gemv_evex,
        iacc_row
    );
    mxfp4_gemv_body!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
        gemv_evex_vnni,
        iacc_row_vnni
    );

    /// `updot`'s lane selection (sgemm.cpp:1756): the VNNI `vpdpbusd` spelling
    /// whenever the host has AVX512VNNI+VL, which is what the reference's
    /// `-march=native` build runs on this machine.
    fn vnni_lane() -> bool {
        crate::simd_x86::avx512vnni() && crate::simd_x86::avx512vl()
    }

    /// Feature dispatch: VNNI+EVEX > EVEX > plain AVX2 (mirrors what the
    /// reference's `-march=native` build compiles in unconditionally).
    pub unsafe fn gemv(n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nc: usize) {
        if vnni_lane() {
            gemv_evex_vnni(n, s, vx, vy, nc);
        } else if crate::simd_x86::avx512vl() {
            gemv_evex(n, s, vx, vy, nc);
        } else {
            gemv_vex(n, s, vx, vy, nc);
        }
    }

    /// The reference's AVX2 gemm body (`gemm_q4_b32_8x8_q8_0_lut_avx`, `#else`
    /// branch of arch/x86/repack.cpp:1097-1290) with the gemv column network:
    /// 16 activation rows per pass (4 `block_q8_0x4` tiles, rhs network shared),
    /// `acc_rows[16]` then a 4-row tail — same arithmetic as the AVX512 branch
    /// the reference actually executes on this machine. Triple-instantiated
    /// like `gemv` above (VEX / EVEX / EVEX+VNNI).
    macro_rules! mxfp4_gemm_body {
        ($(#[$attr:meta])* $name:ident, $iacc:ident) => {
            $(#[$attr])*
            pub unsafe fn $name(n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize, xstart: usize) {
        let nb = n / QK8_0;
        let b_nb = n / 32;
        let lut = load_lut();
        let m4b = _mm256_set1_epi8(0x0F);
        let finalpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        let anr = nr - nr % 16; // C: rows handled with acc_rows[16]

        let mut y = 0usize;
        while y < anr / 4 {
            // a_ptrs[0..4] = four consecutive block_q8_0x4 tiles = 16 rows
            // (C: `a_ptrs[rp] = a_ptr_start + (y + rp)*nb`, arch/x86/repack.cpp:1279)
            for x in xstart..nc / NROWS_INTERLEAVED {
                let b_ptr = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                let mut acc = [_mm256_setzero_ps(); 16];
                for b in 0..nb {
                    let mut iacc = [_mm256_setzero_si256(); 16];
                    let mut ds = [0f32; 16];
                    for rp in 0..4 {
                        let a_tile = vy.add(((y + rp) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        for i in 0..4 {
                            let (lhs0, lhs1) = lhs_pair_q8_0x4(a_tile.add(8), i);
                            iacc[rp * 4 + i] = $iacc(lut, m4b, b_ptr.add(b * BLOCK_MXFP4X8_SIZE), lhs0, lhs1);
                            ds[rp * 4 + i] = d_f16(a_tile.add(i * 2));
                        }
                    }
                    let col = col_scale_mxfp4(b_ptr.add(b * BLOCK_MXFP4X8_SIZE));
                    for i in 0..16 {
                        acc[i] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc[i]),
                            _mm256_mul_ps(col, _mm256_set1_ps(ds[i])),
                            acc[i],
                        );
                    }
                }
                for i in 0..16 {
                    let out = _mm256_permutevar8x32_ps(acc[i], finalpermute);
                    _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), out);
                }
            }
            y += 4;
        }
        // 4-row tail (C: `for (; y < nr/4; y++)` with acc_rows[4])
        while y < nr / 4 {
            for x in xstart..nc / NROWS_INTERLEAVED {
                let b_ptr = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                let mut acc = [_mm256_setzero_ps(); 4];
                for b in 0..nb {
                    let at = vy.add((y * nb + b) * BLOCK_Q8_0X4_SIZE);
                    let mut iacc = [_mm256_setzero_si256(); 4];
                    let mut ds = [0f32; 4];
                    for i in 0..4 {
                        let (lhs0, lhs1) = lhs_pair_q8_0x4(at.add(8), i);
                        iacc[i] = $iacc(lut, m4b, b_ptr.add(b * BLOCK_MXFP4X8_SIZE), lhs0, lhs1);
                        ds[i] = d_f16(at.add(i * 2));
                    }
                    let col = col_scale_mxfp4(b_ptr.add(b * BLOCK_MXFP4X8_SIZE));
                    for i in 0..4 {
                        acc[i] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc[i]),
                            _mm256_mul_ps(col, _mm256_set1_ps(ds[i])),
                            acc[i],
                        );
                    }
                }
                for i in 0..4 {
                    let out = _mm256_permutevar8x32_ps(acc[i], finalpermute);
                    _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), out);
                }
            }
            y += 1;
        }
        }
    };
    }

    mxfp4_gemm_body!(
        #[target_feature(enable = "avx2,fma,f16c")]
        gemm_vex,
        iacc_row
    );
    mxfp4_gemm_body!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq")]
        gemm_evex,
        iacc_row
    );
    mxfp4_gemm_body!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
        gemm_evex_vnni,
        iacc_row_vnni
    );

    /// `mul_sum_i8_pairs_acc_int32x16` (arch/x86/repack.cpp:134), VNNI spelling:
    /// the zmm twin of [`mul_sum_i8_pairs_acc_int32x8_vnni`], same value
    /// proof (|LUT| ≤ 12, |q8| ≤ 127, no saturation possible).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn dpb512(acc: __m512i, x: __m512i, y: __m512i) -> __m512i {
        // the C's exact spelling (arch/x86/repack.cpp:133-141): |x|, then y
        // negated under x's negative-byte mask, then vpdpbusd. The byte where
        // x == 0 needs no zeroing (|x| = 0 makes the product 0), so this is
        // value-identical to the 256-bit vpsignb form.
        let ax = _mm512_abs_epi8(x);
        let blt0 = _mm512_movepi8_mask(x);
        let sy = _mm512_mask_sub_epi8(y, blt0, _mm512_setzero_si512(), y);
        _mm512_dpbusd_epi32(acc, ax, sy)
    }

    /// The gemm's four dword-shuffle patterns (arch/x86/repack.cpp:734-758):
    /// 160/245 select the even/odd lhs dword pairs, 136/221 the rhs ones.
    #[inline(always)]
    fn shuf160(x: __m512i) -> __m512i {
        // SAFETY: only invoked (transitively, after full inlining) from
        // `gemm_avx512`, which carries avx512f.
        unsafe { _mm512_shuffle_epi32::<160>(x) }
    }
    #[inline(always)]
    fn shuf245(x: __m512i) -> __m512i {
        // SAFETY: only invoked (transitively, after full inlining) from
        // `gemm_avx512`, which carries avx512f.
        unsafe { _mm512_shuffle_epi32::<245>(x) }
    }
    #[inline(always)]
    fn shuf136(x: __m512i) -> __m512i {
        // SAFETY: only invoked (transitively, after full inlining) from
        // `gemm_avx512`, which carries avx512f.
        unsafe { _mm512_shuffle_epi32::<136>(x) }
    }
    #[inline(always)]
    fn shuf221(x: __m512i) -> __m512i {
        // SAFETY: only invoked (transitively, after full inlining) from
        // `gemm_avx512`, which carries avx512f.
        unsafe { _mm512_shuffle_epi32::<221>(x) }
    }

    /// The reference's AVX512BW/DQ gemm section (arch/x86/repack.cpp:663-1096),
    /// which is what the pinned build runs on this host: two 8-column tiles per
    /// x-pass packed into zmm (014589CD / 2367ABEF byte interleave), the
    /// activation rows in 2x2 "matrix-matrix" form, 16 `acc_rows` live in the
    /// 32-entry zmm register file. Bit-exact by the same argument as the
    /// 256-bit body: every output element of every k-block is
    /// `fma(f32(exact int32 Σ32 LUT[w]·q8), e8m0_half·row_d, acc)` — the lane
    /// regrouping only redistributes which zmm lane holds which element, and
    /// the sp1+sp2 split is an exact integer partition of the 32 products.
    ///
    /// Covers `x < anc/8` (`anc = nc - nc%16`); the ragged 8-column group and
    /// the row tails delegate to the 256-bit body with `xstart` (C:1092-1095).
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
    pub unsafe fn gemm_avx512(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
    ) {
        let nb = n / QK8_0;
        let b_nb = n / 32;
        let anc = nc - nc % 16;
        let anr = nr - nr % 16;
        if anc == 0 || anr == 0 {
            // degenerate: everything falls to the 256-bit body from column 0
            gemm_evex_vnni(n, s, bs, vx, vy, nr, nc, 0);
            return;
        }
        let lut256 = load_lut();
        // signextendlutexpanded + m4bexpanded (arch/x86/repack.cpp:666-668)
        let lut = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(lut256), lut256);
        let m4b = _mm512_set1_epi8(0x0F);
        // requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4) (:660)
        let required_order = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
        // loadMask = low two dwords (block_q8_0x4.d[0..4] = 4 x f16)
        let load_mask = _mm_set_epi32(0, 0, -1, -1);

        // one block_q8_0x4 (4 rows) against the decoded rhs pair + col
        // scale; fma into acc[BASE..BASE+4]. C:788-876 (16-row rp body) ==
        // C:902-1083 (4-row body) modulo the tile. BASE must be a literal:
        // runtime-indexed acc[] and the closure form this replaced both put
        // the dpb operands on the stack (the 32-entry zmm file needs the
        // chains interleaved so the shuffle patterns are transient).
        macro_rules! zmm_rows {
            ($a_tile:expr, $acc:expr, $BASE:expr, $col:expr,
             $r0_0:expr, $r0_1:expr, $r0_2:expr, $r0_3:expr,
             $r1_0:expr, $r1_1:expr, $r1_2:expr, $r1_3:expr) => {{
                let qs = ($a_tile).add(8);
                // lhs: four 32B loads (elements 0-7/8-15/16-23/24-31 of
                // A0..A3), each split [A0A1|A0A1] / [A2A3|A2A3] (C:792-812)
                let lift = |v: __m256i| {
                    let lo = _mm256_permute2x128_si256::<0>(v, v);
                    let hi = _mm256_permute2x128_si256::<17>(v, v);
                    (
                        _mm512_inserti32x8::<1>(_mm512_castsi256_si512(lo), lo),
                        _mm512_inserti32x8::<1>(_mm512_castsi256_si512(hi), hi),
                    )
                };
                let (l01_0, l23_0) = lift(_mm256_loadu_si256(qs as *const __m256i));
                let (l01_1, l23_1) = lift(_mm256_loadu_si256(qs.add(32) as *const __m256i));
                let (l01_2, l23_2) = lift(_mm256_loadu_si256(qs.add(64) as *const __m256i));
                let (l01_3, l23_3) = lift(_mm256_loadu_si256(qs.add(96) as *const __m256i));
                let zero = _mm512_setzero_epi32();
                // the 2x2 chains (C:845-852): [lhs 01|lhs 23] x [rhs 0145|rhs
                // 2367], sp1 (lhs 160 / rhs 136) + sp2 (245 / 221) summed with
                // an exact integer add (C:855-858)
                let m00 = _mm512_add_epi32(
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf160(l01_3), shuf136($r0_3)), shuf160(l01_2), shuf136($r0_2)),
                        shuf160(l01_1), shuf136($r0_1)), shuf160(l01_0), shuf136($r0_0)),
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf245(l01_3), shuf221($r0_3)), shuf245(l01_2), shuf221($r0_2)),
                        shuf245(l01_1), shuf221($r0_1)), shuf245(l01_0), shuf221($r0_0)));
                let m01 = _mm512_add_epi32(
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf160(l01_3), shuf136($r1_3)), shuf160(l01_2), shuf136($r1_2)),
                        shuf160(l01_1), shuf136($r1_1)), shuf160(l01_0), shuf136($r1_0)),
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf245(l01_3), shuf221($r1_3)), shuf245(l01_2), shuf221($r1_2)),
                        shuf245(l01_1), shuf221($r1_1)), shuf245(l01_0), shuf221($r1_0)));
                let m10 = _mm512_add_epi32(
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf160(l23_3), shuf136($r0_3)), shuf160(l23_2), shuf136($r0_2)),
                        shuf160(l23_1), shuf136($r0_1)), shuf160(l23_0), shuf136($r0_0)),
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf245(l23_3), shuf221($r0_3)), shuf245(l23_2), shuf221($r0_2)),
                        shuf245(l23_1), shuf221($r0_1)), shuf245(l23_0), shuf221($r0_0)));
                let m11 = _mm512_add_epi32(
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf160(l23_3), shuf136($r1_3)), shuf160(l23_2), shuf136($r1_2)),
                        shuf160(l23_1), shuf136($r1_1)), shuf160(l23_0), shuf136($r1_0)),
                    dpb512(dpb512(dpb512(dpb512(zero,
                        shuf245(l23_3), shuf221($r1_3)), shuf245(l23_2), shuf221($r1_2)),
                        shuf245(l23_1), shuf221($r1_1)), shuf245(l23_0), shuf221($r1_0)));
                // straighten to 4 row vectors (C:862-865)
                let ir0 = _mm512_mask_blend_epi32(0xCCCC, m00, _mm512_shuffle_epi32::<78>(m01));
                let ir1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(m00), m01);
                let ir2 = _mm512_mask_blend_epi32(0xCCCC, m10, _mm512_shuffle_epi32::<78>(m11));
                let ir3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(m10), m11);
                // row scales: the tile's 4 f16 d -> f32 -> broadcast (C:868-869)
                let d4 = _mm_shuffle_epi32::<68>(_mm_maskload_epi32(($a_tile) as *const i32, load_mask));
                let rs = _mm512_broadcast_f32x4(_mm_cvtph_ps(d4));
                $acc[$BASE] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ir0),
                    _mm512_mul_ps($col, _mm512_shuffle_ps::<0>(rs, rs)), $acc[$BASE]);
                $acc[$BASE + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ir1),
                    _mm512_mul_ps($col, _mm512_shuffle_ps::<85>(rs, rs)), $acc[$BASE + 1]);
                $acc[$BASE + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ir2),
                    _mm512_mul_ps($col, _mm512_shuffle_ps::<170>(rs, rs)), $acc[$BASE + 2]);
                $acc[$BASE + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ir3),
                    _mm512_mul_ps($col, _mm512_shuffle_ps::<255>(rs, rs)), $acc[$BASE + 3]);
            }};
        }
        // ---- 16-row section (C:671-883) ----
        // arch batch 18 note: a loop-fission experiment (four b-loops, one
        // per 4-row group, decode re-run per loop — the shape GCC's own
        // distribution produces for the reference) measured 8036→9884 µs on
        // the nr=128 isolated bench: the 4x re-decode costs more than the
        // removed zmm spills. A 2-way split was ~2.4% better but under the
        // concurrent-agent load the run-to-run spread was ±20%, below any
        // keep threshold. The fused form below stays (perf9's shape);
        // residual 1.31-1.4x vs GCC remains LLVM scheduling — see
        // parity/asm/{port,ref}_mxfp4_gemm_avx512.asm.
        let mut y = 0usize;
        while y < anr / 4 {
            for x in (0..anc / NROWS_INTERLEAVED).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_MXFP4X8_SIZE);
                let mut acc = [_mm512_setzero_ps(); 16];
                {
                    for b in 0..nb {
                        let q0 = b0.add(b * BLOCK_MXFP4X8_SIZE).add(8);
                        let q1 = b1.add(b * BLOCK_MXFP4X8_SIZE).add(8);
                        // raw 2-tile loads (C:694-702)
                        let raw = |q: *const u8| {
                            [
                                _mm256_loadu_si256(q as *const __m256i),
                                _mm256_loadu_si256(q.add(32) as *const __m256i),
                                _mm256_loadu_si256(q.add(64) as *const __m256i),
                                _mm256_loadu_si256(q.add(96) as *const __m256i),
                            ]
                        };
                        let r0 = raw(q0);
                        let r1 = raw(q1);
                        // 0145/2367 blends (C:705-713)
                        let bl = |a: __m256i, c: __m256i| -> (__m256i, __m256i) {
                            let pc = _mm256_permutevar8x32_epi32(c, required_order);
                            let pa = _mm256_permutevar8x32_epi32(a, required_order);
                            (_mm256_blend_epi32::<240>(a, pc), _mm256_blend_epi32::<240>(pa, c))
                        };
                        let (a0145_0, a2367_0) = bl(r0[0], r0[1]);
                        let (a0145_1, a2367_1) = bl(r0[2], r0[3]);
                        let (a89cd_0, aabef_0) = bl(r1[0], r1[1]);
                        let (a89cd_1, aabef_1) = bl(r1[2], r1[3]);
                        let up = |lo: __m256i, hi: __m256i| _mm512_inserti32x8::<1>(_mm512_castsi256_si512(lo), hi);
                        let raw0145 = [up(a0145_0, a89cd_0), up(a0145_1, a89cd_1)];
                        let raw2367 = [up(a2367_0, aabef_0), up(a2367_1, aabef_1)];
                        // nibble -> LUT (C:721-731)
                        let rhs_0145 = [
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(raw0145[0], m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(raw0145[1], m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw0145[0]), m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw0145[1]), m4b)),
                        ];
                        let rhs_2367 = [
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(raw2367[0], m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(raw2367[1], m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw2367[0]), m4b)),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw2367[1]), m4b)),
                        ];
                        // col scale: 16 E8M0 gathers, tile1 in the high half
                        // (C:766-784 mxfp4 branch)
                        let e0 = b0.add(b * BLOCK_MXFP4X8_SIZE);
                        let e1 = b1.add(b * BLOCK_MXFP4X8_SIZE);
                        let g = |p: *const u8, i: usize| crate::quants::E8M0_HALF_LUT[*p.add(i) as usize];
                        let col = _mm512_set_ps(
                            g(e1, 7), g(e1, 6), g(e1, 5), g(e1, 4), g(e1, 3), g(e1, 2), g(e1, 1), g(e1, 0),
                            g(e0, 7), g(e0, 6), g(e0, 5), g(e0, 4), g(e0, 3), g(e0, 2), g(e0, 1), g(e0, 0),
                        );
                        let a0 = vy.add((y * nb + b) * BLOCK_Q8_0X4_SIZE);
                        let a1 = vy.add(((y + 1) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        let a2 = vy.add(((y + 2) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        let a3 = vy.add(((y + 3) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        zmm_rows!(a0, acc, 0, col, rhs_0145[0], rhs_0145[1], rhs_0145[2], rhs_0145[3],
                                  rhs_2367[0], rhs_2367[1], rhs_2367[2], rhs_2367[3]);
                        zmm_rows!(a1, acc, 4, col, rhs_0145[0], rhs_0145[1], rhs_0145[2], rhs_0145[3],
                                  rhs_2367[0], rhs_2367[1], rhs_2367[2], rhs_2367[3]);
                        zmm_rows!(a2, acc, 8, col, rhs_0145[0], rhs_0145[1], rhs_0145[2], rhs_0145[3],
                                  rhs_2367[0], rhs_2367[1], rhs_2367[2], rhs_2367[3]);
                        zmm_rows!(a3, acc, 12, col, rhs_0145[0], rhs_0145[1], rhs_0145[2], rhs_0145[3],
                                  rhs_2367[0], rhs_2367[1], rhs_2367[2], rhs_2367[3]);
                    }
                }
                for i in 0..16 {
                    _mm512_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 4;
        }
        // ---- 4-row section (C:887-1091) ----
        while y < nr / 4 {
            for x in (0..anc / NROWS_INTERLEAVED).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_MXFP4X8_SIZE);
                let mut acc = [_mm512_setzero_ps(); 4];
                for b in 0..nb {
                    let q0 = b0.add(b * BLOCK_MXFP4X8_SIZE).add(8);
                    let q1 = b1.add(b * BLOCK_MXFP4X8_SIZE).add(8);
                    let raw = |q: *const u8| {
                        [
                            _mm256_loadu_si256(q as *const __m256i),
                            _mm256_loadu_si256(q.add(32) as *const __m256i),
                            _mm256_loadu_si256(q.add(64) as *const __m256i),
                            _mm256_loadu_si256(q.add(96) as *const __m256i),
                        ]
                    };
                    let r0 = raw(q0);
                    let r1 = raw(q1);
                    let bl = |a: __m256i, c: __m256i| -> (__m256i, __m256i) {
                        let pc = _mm256_permutevar8x32_epi32(c, required_order);
                        let pa = _mm256_permutevar8x32_epi32(a, required_order);
                        (_mm256_blend_epi32::<240>(a, pc), _mm256_blend_epi32::<240>(pa, c))
                    };
                    let (a0145_0, a2367_0) = bl(r0[0], r0[1]);
                    let (a0145_1, a2367_1) = bl(r0[2], r0[3]);
                    let (a89cd_0, aabef_0) = bl(r1[0], r1[1]);
                    let (a89cd_1, aabef_1) = bl(r1[2], r1[3]);
                    let up = |lo: __m256i, hi: __m256i| _mm512_inserti32x8::<1>(_mm512_castsi256_si512(lo), hi);
                    let raw0145 = [up(a0145_0, a89cd_0), up(a0145_1, a89cd_1)];
                    let raw2367 = [up(a2367_0, aabef_0), up(a2367_1, aabef_1)];
                    let rhs_0145 = [
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(raw0145[0], m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(raw0145[1], m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw0145[0]), m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw0145[1]), m4b)),
                    ];
                    let rhs_2367 = [
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(raw2367[0], m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(raw2367[1], m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw2367[0]), m4b)),
                        _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(raw2367[1]), m4b)),
                    ];
                    let e0 = b0.add(b * BLOCK_MXFP4X8_SIZE);
                    let e1 = b1.add(b * BLOCK_MXFP4X8_SIZE);
                    let g = |p: *const u8, i: usize| crate::quants::E8M0_HALF_LUT[*p.add(i) as usize];
                    let col = _mm512_set_ps(
                        g(e1, 7), g(e1, 6), g(e1, 5), g(e1, 4), g(e1, 3), g(e1, 2), g(e1, 1), g(e1, 0),
                        g(e0, 7), g(e0, 6), g(e0, 5), g(e0, 4), g(e0, 3), g(e0, 2), g(e0, 1), g(e0, 0),
                    );
                    let a_tile = vy.add((y * nb + b) * BLOCK_Q8_0X4_SIZE);
                    zmm_rows!(a_tile, acc, 0, col, rhs_0145[0], rhs_0145[1], rhs_0145[2], rhs_0145[3],
                              rhs_2367[0], rhs_2367[1], rhs_2367[2], rhs_2367[3]);
                }
                for i in 0..4 {
                    _mm512_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 1;
        }
        // ---- ragged 8-column group (C:1092-1095 -> the 256-bit body) ----
        let xstart = if anc != nc { anc / NROWS_INTERLEAVED } else { 0 };
        if xstart != 0 {
            gemm_evex_vnni(n, s, bs, vx, vy, nr, nc, xstart);
        }
    }

    /// Feature dispatch, same ladder as `gemv`, with the reference's AVX512BW/DQ
    /// zmm gemm section on top (the pinned build's code path on this host,
    /// arch/x86/repack.cpp:663).
    pub unsafe fn gemm(n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize) {
        if vnni_lane() && crate::simd_x86::avx512bw() && is_x86_feature_detected!("avx512dq") {
            gemm_avx512(n, s, bs, vx, vy, nr, nc);
        } else if vnni_lane() {
            gemm_evex_vnni(n, s, bs, vx, vy, nr, nc, 0);
        } else if crate::simd_x86::avx512vl() {
            gemm_evex(n, s, bs, vx, vy, nr, nc, 0);
        } else {
            gemm_vex(n, s, bs, vx, vy, nr, nc, 0);
        }
    }

    /// `ggml_cpu_has_avx2()`-equivalent runtime gate (the reference's
    /// `ggml_repack_get_optimal_repack_type` requires AVX2 for mxfp4_8x8).
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    /// The scalar path's activation view type; kept imported so the `avx2` module
    /// documents the same `block_q8_0` layout the SIMD kernels read.
    #[allow(dead_code)]
    fn _ty(_: &BlockQ8_0) {}
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

/// Accumulate one 32-element k-block into `acc[0..8]` for the 8 weight rows of
/// `w_blk` (one `block_mxfp4x8`, 136 bytes) against activation block `a`.
///
/// This is the arithmetic of `gemv_q4_b32_8x8_q8_0_lut_avx`'s inner body
/// (arch/x86/repack.cpp:566-629): an exact int32 dot over the whole block, then
/// `_mm256_fmadd_ps(cvtepi32_ps(iacc), _mm256_mul_ps(col_scale, row_scale), acc)`.
#[inline(always)]
fn accum_block_mxfp4x8(w_blk: &[u8], a: &BlockQ8_0, acc: &mut [f32; 8]) {
    let d = a.d.to_f32();
    for j in 0..8 {
        // row j of the tile: nibble bytes 0..8 (elements 0..7 lo / 16..23 hi) at
        // qs[8j], and bytes 8..16 (elements 8..15 lo / 24..31 hi) at qs[64+8j];
        // the tile's qs payload starts at byte 8, hence 8 + 8j / 8 + 64 + 8j.
        let lo = &w_blk[8 + 8 * j..8 + 8 * j + 8];
        let hi = &w_blk[72 + 8 * j..72 + 8 * j + 8];
        let mut dot = 0i32;
        for t in 0..8 {
            // element t: low nibble of lo[t];  element 16+t: high nibble of lo[t]
            dot += KVALUES_MXFP4[(lo[t] & 0x0F) as usize] as i32 * a.qs[t] as i32;
            dot += KVALUES_MXFP4[(lo[t] >> 4) as usize] as i32 * a.qs[16 + t] as i32;
            // element 8+t: low nibble of hi[t]; element 24+t: high nibble of hi[t]
            dot += KVALUES_MXFP4[(hi[t] & 0x0F) as usize] as i32 * a.qs[8 + t] as i32;
            dot += KVALUES_MXFP4[(hi[t] >> 4) as usize] as i32 * a.qs[24 + t] as i32;
        }
        let scale = e8m0_to_fp32_half(w_blk[j]) * d;
        acc[j] = (dot as f32).mul_add(scale, acc[j]);
    }
}

/// `ggml_gemv_mxfp4_8x8_q8_0` (repack.cpp:1236 dispatch →
/// arch/x86/repack.cpp:1700 `gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>`).
///
/// C signature: `(int n, float * s, size_t bs, const void * vx, const void * vy,
/// int nr, int nc)`. Every call site passes `nr == 1`
/// (repack.cpp:4644 forward_mul_mat_one_chunk's tail rows, and repack.cpp:4905
/// forward_mul_mat_id), where the store index `s + (y*nr + x*8)` degenerates to
/// `s + x*8` and `bs` is unused — this port asserts `nr == 1` and drops both.
///
/// * `n`   — row length in elements (`ne00`)
/// * `s`   — dst floats starting at the first produced row (C passes
///   `dst + i1*nb1 + i2*nb2 + src0_start`); `nc/8` groups of 8, written in order
/// * `vx`  — repacked weights; C passes `src0 + src0_start * nb01`, i.e. the
///   first 8-row group *containing* `src0_start` (callers align to 8)
/// * `vy`  — one activation row in `block_q8_0` form (`from_float`, plain layout)
/// * `nc`  — number of weight rows to produce (multiple of 8)
pub fn gemv_mxfp4_8x8_q8_0(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    let nb = n / QK_MXFP4;
    assert_eq!(n % QK_MXFP4, 0, "gemv: n % 32 != 0");
    assert_eq!(nc % NB_COLS, 0, "gemv: nc % 8 != 0");
    assert!(vy.len() >= nb * BLOCK_Q8_0_SIZE, "gemv: vy too small");
    assert!(vx.len() >= (nc / 8) * nb * BLOCK_MXFP4X8_SIZE, "gemv: vx too small");
    assert!(s.len() >= nc, "gemv: s too small");
    STAT_GEMV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // reference AVX2 kernel when the host has it (see `simd_x86`); the scalar
    // body below is the same arithmetic and stays as the fallback
    #[cfg(target_arch = "x86_64")]
    if simd_enabled() && simd_x86::available() {
        // SAFETY: bounds asserted above; the kernel only reads `nc/8 * nb` tiles,
        // `nb` q8_0 blocks and writes `nc` floats.
        unsafe {
            simd_x86::gemv(n, s.as_mut_ptr(), vx.as_ptr(), vy.as_ptr(), nc);
        }
        return;
    }

    gemv_mxfp4_8x8_q8_0_scalar(n, s, vx, vy, nc);
}

/// Portable (scalar) body of [`gemv_mxfp4_8x8_q8_0`] — the arithmetic the SIMD
/// kernel must reproduce bit-for-bit, and the fallback on non-AVX2 hosts.
pub fn gemv_mxfp4_8x8_q8_0_scalar(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    let nb = n / QK_MXFP4;
    let a: &[BlockQ8_0] = bytemuck::cast_slice(&vy[..nb * BLOCK_Q8_0_SIZE]);

    for x in 0..nc / 8 {
        let b_ptr = &vx[x * nb * BLOCK_MXFP4X8_SIZE..];
        let mut acc = [0f32; 8];
        for b in 0..nb {
            accum_block_mxfp4x8(&b_ptr[b * BLOCK_MXFP4X8_SIZE..], &a[b], &mut acc);
        }
        s[x * 8..x * 8 + 8].copy_from_slice(&acc);
    }
}

/// `ggml_gemm_mxfp4_8x8_q8_0` (repack.cpp:2363 dispatch →
/// arch/x86/repack.cpp:3511 `gemm_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>`).
///
/// C signature: `(int n, float * s, size_t bs, const void * vx, const void * vy,
/// int nr, int nc)`; stores `s + ((y*4 + i) * bs + x*8)`
/// (arch/x86/repack.cpp:873/1789). `vy` is the 4-row activation tile written by
/// [`quantize_mat_q8_0_4x8`] (`INTER_SIZE == 8` for the mxfp4 8x8 traits — see
/// that function's note), `nr` must be a multiple of 4, `nc` a multiple of 8.
/// Per output element this is the *same* fma-per-block accumulation as
/// [`gemv_mxfp4_8x8_q8_0`] — only the lane assignment and the Lhs layout differ.
pub fn gemm_mxfp4_8x8_q8_0(n: usize, s: &mut [f32], bs: usize, vx: &[u8], vy: &[u8], nr: usize, nc: usize) {
    let nb = n / QK_MXFP4;
    assert_eq!(n % QK_MXFP4, 0, "gemm: n % 32 != 0");
    assert_eq!(nr % 4, 0, "gemm: nr % 4 != 0");
    assert_eq!(nc % NB_COLS, 0, "gemm: nc % 8 != 0");
    assert!(vy.len() >= (nr / 4) * nb * BLOCK_Q8_0X4_SIZE, "gemm: vy too small");
    assert!(vx.len() >= (nc / 8) * nb * BLOCK_MXFP4X8_SIZE, "gemm: vx too small");
    assert!(
        s.len() >= (nr - 1) * bs + nc,
        "gemm: s too small for nr={nr} bs={bs} nc={nc}"
    );

    #[cfg(target_arch = "x86_64")]
    if simd_enabled() && simd_x86::available() {
        // SAFETY: bounds asserted above (C stores `s + ((y*4+i)*bs + x*8)`).
        unsafe {
            simd_x86::gemm(n, s.as_mut_ptr(), bs, vx.as_ptr(), vy.as_ptr(), nr, nc);
        }
        return;
    }

    gemm_mxfp4_8x8_q8_0_scalar(n, s, bs, vx, vy, nr, nc);
}

/// Portable (scalar) body of [`gemm_mxfp4_8x8_q8_0`] — see the doc note on the
/// SIMD kernel for why the two agree bit-for-bit.
pub fn gemm_mxfp4_8x8_q8_0_scalar(n: usize, s: &mut [f32], bs: usize, vx: &[u8], vy: &[u8], nr: usize, nc: usize) {
    let nb = n / QK_MXFP4;
    for y in 0..nr / 4 {
        // one block_q8_0x4 group = 4 activation rows, nb tiles of 136 bytes
        let a_grp = &vy[y * nb * BLOCK_Q8_0X4_SIZE..(y + 1) * nb * BLOCK_Q8_0X4_SIZE];
        for x in 0..nc / 8 {
            let b_ptr = &vx[x * nb * BLOCK_MXFP4X8_SIZE..];
            let mut acc = [[0f32; 8]; 4];
            for b in 0..nb {
                let tile = &b_ptr[b * BLOCK_MXFP4X8_SIZE..];
                let a_tile = &a_grp[b * BLOCK_Q8_0X4_SIZE..];
                for i in 0..4 {
                    let a = BlockQ8_0 {
                        d: f16_from_bytes(&a_tile[i * 2..i * 2 + 2]),
                        qs: deinterleave_q8_0x4_row(&a_tile[8..], i, INTER_SIZE),
                    };
                    accum_block_mxfp4x8(tile, &a, &mut acc[i]);
                }
            }
            for i in 0..4 {
                let o = (y * 4 + i) * bs + x * 8;
                s[o..o + 8].copy_from_slice(&acc[i]);
            }
        }
    }
}

#[inline]
fn f16_from_bytes(b: &[u8]) -> half::f16 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]]))
}

/// Extract activation row `row` (0..4) of a `block_q8_0x4` qs payload as the
/// plain q8_0 element order, for interleave granularity `interleave`
/// (`INTER_SIZE`): inverse of [`quantize_mat_q8_0_4x4`]/[`quantize_mat_q8_0_4x8`]'s
/// store loop, i.e. `qs[(el/interleave)*4*interleave + row*interleave + el%interleave]`
/// holds element `el` of row `row`.
#[inline]
fn deinterleave_q8_0x4_row(qs: &[u8], row: usize, interleave: usize) -> [i8; QK8_0] {
    let mut out = [0i8; QK8_0];
    for j in 0..QK8_0 * 4 {
        let src_id = (j % (4 * interleave)) / interleave;
        let src_offset = (j / (4 * interleave)) * interleave + (j % interleave);
        if src_id == row {
            out[src_offset] = qs[j] as i8;
        }
    }
    out
}

// ===========================================================================
// Q4_K 8x8 — the reference's x86 production path for Q4_K
// ===========================================================================
//
// Reference: `ggml_repack_get_optimal_repack_type` (repack.cpp:5006-5011)
//
// ```text
//   if (ggml_cpu_has_avx2()) { if (cur->ne[1] % 8 == 0) return &q4_K_8x8_q8_K; }
// ```
//
// (the two other Q4_K gates in that function need NEON or RISC-V; on x86 the
// AVX2 branch is the only one, so every 2D Q4_K tensor whose `ne[1]` is a
// multiple of 8 gets a CPU_REPACK trait). Q5_K/Q6_K have **no x86 instance at
// all** (repack.cpp:5050-5071: both gates are `neon && matmul_int8/dotprod`), so
// the reference runs its row-wise `vec_dot` for them and the port must too
// (vec_dot.rs's `kquant_real_tensor_tests` pins that).
//
//   * `set_tensor` -> `repack_q4_K_to_q4_K_8_bl` (repack.cpp:3573-3601) ->
//     `make_block_q4_Kx8` (repack.cpp:3177-3252): 8 rows x 12 scale bytes +
//     1024 nibble bytes per k-super-block.
//   * `forward_mul_mat` (repack.cpp:4637-4647) splits the activation rows:
//     `if (nrows > 3) gemm(..., nrows - nrows%4, ncols)` and then one `gemv`
//     per leftover row. `forward_mul_mat`'s wdata pass (repack.cpp:4697-4705)
//     quantizes whole groups of 4 rows with `ggml_quantize_mat_t<8, Q8_K>` =
//     `ggml_quantize_mat_q8_K_4x8` (repack.cpp:336 -> arch/x86/repack.cpp:290)
//     and the `ne11 % 4` tail rows with the plain
//     `type_traits_cpu[Q8_K].from_float` = `quantize_row_q8_K_ref` — so the two
//     kernels consume two *different* LHS layouts (interleaved 4x8 tiles vs
//     plain 292-byte rows).
//   * kernels: `ggml_gemv_q4_K_8x8_q8_K` (arch/x86/repack.cpp:1464; the AVX2
//     body at :1486-1677 is what a `-march=native` build links) and
//     `ggml_gemm_q4_K_8x8_q8_K` (arch/x86/repack.cpp:2042; the AVX512BW+DQ
//     body at :2077-2815 — 16-row passes, a 4-row tail, then the AVX2 body
//     at :3158-3486 for the `nc % 16` tail columns — is ported in
//     `simd_x86_q4k::gemm_avx512`, the AVX2 body in `gemm_256_section`).
//
// ## Arithmetic (what the port must reproduce)
//
// For one output element (weight row `c`, activation row `r`) and one
// super-block `b`, every body of both kernels computes
//
// ```text
//   iacc   = Σ_{g<4} [ s_2g(c) * Σ_{m<32} lo_nib(m) * q8[64g+m]
//                    + s_2g+1(c) * Σ_{m<32} hi_nib(m) * q8[64g+32+m] ]   // exact int32
//   imin   = Σ_{g<4} [ m_2g(c) * (bsums[4g] + bsums[4g+1])
//                    + m_2g+1(c) * (bsums[4g+2] + bsums[4g+3]) ]         // exact int32
//   acc    = fma(f32(iacc), f32(d_col[c]) * f32(d_row), acc)             // ONE fma per b
//   accmin = fma(f32(imin), f32(dmin_col[c]) * f32(d_row), accmin)
//   out    = acc - accmin
// ```
//
// with `b` ascending. The two properties that make a re-implementation
// bit-exact are (a) the integer sums are exact (all products ≤ 15*127, the
// int16 lane sums stay < 2^15, the scaled sums < 2^31) so *any* lane grouping
// gives the same int32, and (b) the float side is one rounded `d_col*d_row`
// product followed by one fused multiply-add per super-block — the C's SIMD
// lane permutations only decide which register lane holds which output. The
// row-wise `vec_dot_q4_K_q8_K` kernels do NOT have property (b) (they fma
// *partial* per-lane ints with the same product, then hsum), which is why the
// reference's own repack-vs-plain gap is non-zero and why the port had to
// switch Q4_K to this path to match the reference (`vec_dot.rs`'s
// `kquant_real_tensor_tests` recorded the old, plain-path agreement).
//
// The C's `*_generic` kernels (repack.cpp:958 gemv / :2032 gemm) are the
// non-AVX2 arch fallbacks; they accumulate `sumf[j] += f32(sumi) * d * d_row`
// per k-group (a different rounding chain) and are NOT what the reference
// build on this host runs. The port's scalar kernels below therefore port the
// *SIMD* arithmetic (property (b)); `q4k_simd_matches_scalar_bit_exact` pins
// SIMD == scalar, and the dump test pins both against the reference binary.

use crate::blocks::QK_K;

/// `sizeof(block_q4_K)` (ggml-common.h)
pub const BLOCK_Q4_K_SIZE: usize = 4 + 12 + QK_K / 2; // 144
/// `sizeof(block_q4_Kx8)` (repack.h:48-55) — one 8-row x 1-super-block tile
pub const BLOCK_Q4_KX8_SIZE: usize = 16 + 16 + 96 + QK_K * 4; // 1152
/// `K_SCALE_SIZE` — the 6-bit (scale, min) bytes of one `block_q4_K`
pub const K_SCALE_SIZE: usize = 12;
/// `sizeof(block_q8_K)` (ggml-common.h) — f32 d + 256 i8 + 16 i16
pub const BLOCK_Q8_K_SIZE: usize = 4 + QK_K + 32; // 292
/// `sizeof(block_q8_Kx4)` (repack.h:101-107) — the gemm's 4x8 LHS tile
pub const BLOCK_Q8_KX4_SIZE: usize = 16 + QK_K * 4 + QK_K / 4 * 2; // 1168
/// `qk` of the Q4_K path (the Q4_K super-block is QK_K elements)
pub const Q4K_QK: usize = QK_K;
/// `nrows_interleaved` of `repack_q4_K_to_q4_K_8_bl` (repack.cpp:3577)
const NROWS_INTERLEAVED_Q4K: usize = 8;

/// `ggml_repack_get_optimal_repack_type` (repack.cpp:5006-5011) for Q4_K on
/// x86_64: repack iff `ggml_cpu_has_avx2() && ne[1] % 8 == 0`.
/// `repack_q4_K_to_q4_K_8_bl` additionally requires `ne[0] % 8 == 0`
/// (repack.cpp:3585), which `ne[0] % QK_K == 0` implies.
pub fn repack_supported_q4k(ne1: i64, ne0: i64) -> bool {
    ne0 % QK_K as i64 == 0 && ne1 % NROWS_INTERLEAVED_Q4K as i64 == 0
}

/// Byte offset of scale group `sb` (0..8) inside `block_q4_Kx8.scales`
/// (repack.cpp:3209-3249: the first four groups at `12*i`, the last four at
/// `12*i + 48`).
#[inline]
pub const fn q4k_scale_off(sb: usize) -> usize {
    if sb < 4 {
        12 * sb
    } else {
        12 * (sb - 4) + 48
    }
}

/// Decode the 12 packed 6-bit `(scale, min)` pairs of one sub-block into the
/// 16 bytes `[s(col0..4), s(col4..8), m(col0..4), m(col4..8)]` — the C's `utmp`
/// dance verbatim (repack.cpp:990-995 in the gemv, :3289-3302 in the gemm).
#[inline]
pub fn unpack_q4k_scales_bytes(packed: &[u8; 12]) -> [u8; 16] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut utmp = [
        u32::from_le_bytes(packed[0..4].try_into().unwrap()),
        u32::from_le_bytes(packed[4..8].try_into().unwrap()),
        u32::from_le_bytes(packed[8..12].try_into().unwrap()),
        0u32,
    ];
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    let mut out = [0u8; 16];
    for i in 0..4 {
        out[4 * i..4 * i + 4].copy_from_slice(&utmp[i].to_le_bytes());
    }
    out
}

/// `repack.cpp:3177 make_block_q4_Kx8` — interleave eight `block_q4_K`s (rows
/// `in[0..8]` of the same k-super-block) into one 1152-byte `block_q4_Kx8`.
///
/// Layout: `d[8]`/`dmin[8]` copied per row, `scales[96]` = 8 unpacked 12-byte
/// groups (group `sb` holds sub-block `sb`'s 8 columns x (scale|min)), and
/// `qs[1024]` where byte `8*i + j` is column `i%8`'s byte `8*(i/8) + j`
/// (`blck_size_interleave == 8`, repack.cpp:3188-3200) — i.e. 8-byte chunks of
/// the eight columns in turn.
pub fn make_block_q4_Kx8(in8: &[u8], out: &mut [u8]) {
    debug_assert_eq!(in8.len(), 8 * BLOCK_Q4_K_SIZE);
    debug_assert_eq!(out.len(), BLOCK_Q4_KX8_SIZE);
    // d[8]: in[i].d  (repack.cpp:3180-3182)
    for i in 0..8 {
        out[2 * i..2 * i + 2].copy_from_slice(&in8[i * BLOCK_Q4_K_SIZE..i * BLOCK_Q4_K_SIZE + 2]);
    }
    // dmin[8]: in[i].dmin  (repack.cpp:3184-3186)
    for i in 0..8 {
        out[16 + 2 * i..16 + 2 * i + 2]
            .copy_from_slice(&in8[i * BLOCK_Q4_K_SIZE + 2..i * BLOCK_Q4_K_SIZE + 4]);
    }
    // qs: 8-byte interleave over the 8 rows (repack.cpp:3188-3200).
    // `in[src_id].qs` starts at byte 16 of a plain `block_q4_K`
    // (d 2 + dmin 2 + scales 12).
    let end = QK_K * 4 / 8; // 128
    for i in 0..end {
        let src_id = i % 8;
        let src_offset = (i / 8) * 8;
        let dst_offset = 128 + i * 8;
        let s = src_id * BLOCK_Q4_K_SIZE + 16 + src_offset;
        out[dst_offset..dst_offset + 8].copy_from_slice(&in8[s..s + 8]);
    }
    // scales: 8 x 12-byte groups, each holding one sub-block's 8 columns
    // (repack.cpp:3202-3249)
    let scales = &mut out[32..32 + 96];
    for i in 0..4 {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            let sc = &in8[j * BLOCK_Q4_K_SIZE + 4..j * BLOCK_Q4_K_SIZE + 16]; // in[j].scales
            s[j] = sc[i] & 63;
            m[j] = sc[i + 4] & 63;
        }
        let g = q4k_scale_off(i);
        scales[g] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales[g + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales[g + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales[g + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales[g + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales[g + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales[g + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales[g + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales[g + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales[g + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales[g + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales[g + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
    for i in 0..4 {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            let sc = &in8[j * BLOCK_Q4_K_SIZE + 4..j * BLOCK_Q4_K_SIZE + 16];
            s[j] = ((sc[i] & 192) >> 2) | (sc[i + 8] & 15);
            m[j] = ((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4);
        }
        let g = q4k_scale_off(4 + i);
        scales[g] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales[g + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales[g + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales[g + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales[g + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales[g + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales[g + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales[g + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales[g + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales[g + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales[g + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales[g + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
}

/// `repack.cpp:3573 repack_q4_K_to_q4_K_8_bl` — whole-tensor repack. Rows are
/// grouped in 8s; within a group the tiles are written in k-block order, so the
/// output keeps the source size and a row group occupies `nb * 1152` bytes.
///
/// Row groups are independent, so the loop is parallelized with rayon when the
/// tensor has enough of them (the reference does the equivalent work inside its
/// model *load*, where it is not part of any measured forward; the port
/// materializes lazily on the first use, so any unparallelized time here lands
/// in the first forward of every process). The output is byte-identical either
/// way.
pub fn repack_q4_K_8x8_into(src: &[u8], nrows: usize, n_per_row: usize, dst: &mut [u8]) {
    use rayon::prelude::*;
    let nb = n_per_row / QK_K;
    let row = nb * BLOCK_Q4_K_SIZE;
    assert_eq!(n_per_row % QK_K, 0, "repack_q4_K: ne[0] % QK_K != 0");
    assert_eq!(nrows % NROWS_INTERLEAVED_Q4K, 0, "repack_q4_K: ne[1] % 8 != 0");
    assert!(src.len() >= nrows * row, "repack_q4_K: src too small");
    assert!(dst.len() >= nrows * row, "repack_q4_K: dst too small");
    let group = NROWS_INTERLEAVED_Q4K * row;
    let groups = nrows / NROWS_INTERLEAVED_Q4K;
    let one = |srcg: &[u8], dstg: &mut [u8]| {
        let mut tmp = [0u8; 8 * BLOCK_Q4_K_SIZE];
        let mut d = 0usize;
        for x in 0..nb {
            for i in 0..NROWS_INTERLEAVED_Q4K {
                let so = (x + i * nb) * BLOCK_Q4_K_SIZE;
                tmp[i * BLOCK_Q4_K_SIZE..(i + 1) * BLOCK_Q4_K_SIZE]
                    .copy_from_slice(&srcg[so..so + BLOCK_Q4_K_SIZE]);
            }
            make_block_q4_Kx8(&tmp, &mut dstg[d..d + BLOCK_Q4_KX8_SIZE]);
            d += BLOCK_Q4_KX8_SIZE;
        }
    };
    let (src, dst) = (&src[..groups * group], &mut dst[..groups * group]);
    if groups >= 4 {
        src.par_chunks(group)
            .zip(dst.par_chunks_mut(group))
            .for_each(|(sg, dg)| one(sg, dg));
    } else {
        for (sg, dg) in src.chunks(group).zip(dst.chunks_mut(group)) {
            one(sg, dg);
        }
    }
}

/// [`repack_q4_K_8x8_into`] into a fresh buffer.
pub fn repack_q4_K_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let len = nrows * (n_per_row / QK_K) * BLOCK_Q4_K_SIZE;
    let mut out = vec![0u8; len];
    repack_q4_K_8x8_into(&src[..len], nrows, n_per_row, &mut out);
    out
}

/// Inverse of [`repack_q4_K_8x8`] (round-trip test / debugging only; the
/// reference has no such function).
///
/// The scale groups are decoded with the kernel's own
/// [`unpack_q4k_scales_bytes`] (which yields the 8 columns' 6-bit scales and
/// mins in column order) and re-packed into `block_q4_K.scales` with the
/// standard `get_scale_min_k4` packing (ggml-quants.c:880), so the round trip
/// is a true byte-level inverse.
pub fn unrepack_q4_K_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let nb = n_per_row / QK_K;
    let row = nb * BLOCK_Q4_K_SIZE;
    assert_eq!(nrows % NROWS_INTERLEAVED_Q4K, 0);
    let mut out = vec![0u8; nrows * row];
    for g in 0..nrows / NROWS_INTERLEAVED_Q4K {
        let base = g * NROWS_INTERLEAVED_Q4K * row;
        for x in 0..nb {
            let tile = &src[base + x * BLOCK_Q4_KX8_SIZE..base + (x + 1) * BLOCK_Q4_KX8_SIZE];
            // d[8]/dmin[8] -> the plain blocks' first four bytes
            for i in 0..8 {
                let so = base + (x + i * nb) * BLOCK_Q4_K_SIZE;
                out[so..so + 2].copy_from_slice(&tile[2 * i..2 * i + 2]);
                out[so + 2..so + 4].copy_from_slice(&tile[16 + 2 * i..16 + 2 * i + 2]);
            }
            // per sub-block sb, decode the 8 columns' (scale, min) and re-pack
            for sb in 0..8 {
                let mut packed = [0u8; 12];
                packed.copy_from_slice(&tile[32 + q4k_scale_off(sb)..32 + q4k_scale_off(sb) + 12]);
                let dec = unpack_q4k_scales_bytes(&packed);
                for i in 0..8 {
                    let so = base + (x + i * nb) * BLOCK_Q4_K_SIZE + 4;
                    let (s, m) = (dec[i], dec[8 + i]);
                    if sb < 4 {
                        out[so + sb] = (out[so + sb] & !63) | (s & 63);
                        out[so + sb + 4] = (out[so + sb + 4] & !63) | (m & 63);
                    } else {
                        let j = sb - 4;
                        out[so + j] = (out[so + j] & !192) | ((s >> 4) << 6);
                        out[so + j + 4] = (out[so + j + 4] & !192) | ((m >> 4) << 6);
                        out[so + j + 8] = (out[so + j + 8] & !15) | (s & 15);
                        out[so + j + 8] = (out[so + j + 8] & !240) | ((m & 15) << 4);
                    }
                }
            }
            for i in 0..(QK_K * 4 / 8) {
                let src_id = i % 8;
                let src_offset = (i / 8) * 8;
                let dst_offset = i * 8;
                let so = base + (x + src_id * nb) * BLOCK_Q4_K_SIZE + 16 + src_offset;
                out[so..so + 8]
                    .copy_from_slice(&tile[128 + dst_offset..128 + dst_offset + 8]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// activation quantization for the repack path
// ---------------------------------------------------------------------------

/// `ggml_quantize_mat_q8_K_4x8` (arch/x86/repack.cpp:290-512, AVX2 body) —
/// quantize 4 activation rows into one `block_q8_Kx4`, the LHS the gemm reads.
///
/// The *values* are the reference's: per-row `iscale = ±127/max|v|` over the
/// whole 256-element super-block (arch/x86/repack.cpp:371-389),
/// `round-nearest-even` (the C's `_mm256_round_ps(.., _MM_ROUND_NEAREST)`;
/// note the plain `quantize_row_q8_K_ref` uses `nearest_int`'s ties-away
/// rounding instead), saturation to i8 through the two `packs`, and
/// `d = 1/iscale` — a *reciprocal of a reciprocal*, so the f32 bytes match the
/// C's `y[i].d[row_iter] = 1/iscale` exactly.
///
/// The sign convention differs from the C in one case: `arch/x86/repack.cpp:384`
/// negates `iscale` (and hence `d`) when its lane mask trips, which negates
/// `qs` and `d` *together* — the dequantized values `d*q` and the dot products
/// are unaffected (round-to-nearest-even is odd-symmetric and the negation is
/// exact), so the kernels' outputs are identical; only those two byte patterns
/// can differ in sign. `q4k_interleaved_activation_matches_reference` pins the
/// dequantized values against the reference's own q8_Kx4 bytes.
///
/// Storage (arch/x86/repack.cpp:393-424): element `el` of row `r` sits at
/// `qs[(el/8)*32 + r*8 + el%8]`; bsums are the 16-element sums of the *stored*
/// values and sit at `bsums[16*(el/64) + 4*r + (el/16)%4]`
/// (arch/x86/repack.cpp:426-503, generic form repack.cpp:304).
pub fn quantize_mat_q8_K_4x8(x: &[f32], k: usize, n_rows: usize, out: &mut [u8]) {
    assert_eq!(k % QK_K, 0, "quantize_mat_q8_K_4x8: k % QK_K != 0");
    assert_eq!(n_rows % 4, 0, "quantize_mat_q8_K_4x8: rows must be a multiple of 4");
    let nb = k / QK_K;
    for r4 in 0..n_rows / 4 {
        let rows = &x[r4 * 4 * k..(r4 + 1) * 4 * k];
        let tile = &mut out[r4 * nb * BLOCK_Q8_KX4_SIZE..(r4 + 1) * nb * BLOCK_Q8_KX4_SIZE];
        for ib in 0..nb {
            let blk = &mut tile[ib * BLOCK_Q8_KX4_SIZE..(ib + 1) * BLOCK_Q8_KX4_SIZE];
            let (d, rest) = blk.split_at_mut(16);
            let (qs, bsums) = rest.split_at_mut(QK_K * 4);
            for r in 0..4 {
                let xb = &rows[r * k + ib * QK_K..r * k + (ib + 1) * QK_K];
                let mut max_abs = 0.0f32;
                for &v in xb {
                    let a = v.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                let iscale = if max_abs != 0.0 { 127.0f32 / max_abs } else { 0.0 };
                d[4 * r..4 * r + 4].copy_from_slice(
                    &if max_abs != 0.0 { 1.0f32 / iscale } else { 0.0 }.to_le_bytes(),
                );
                for (el, &v) in xb.iter().enumerate() {
                    // `_mm256_cvtps_epi32` after the round, saturated by the
                    // two `packs`: clamp to i8.
                    let q = (iscale * v).round_ties_even().clamp(-128.0, 127.0) as i8;
                    qs[(el / 8) * 32 + r * 8 + el % 8] = q as u8;
                }
            }
            // bsums: 16-element sums of the stored quants, in the interleaved
            // order `16*(el/64) + 4*r + (el/16)%4`
            bsums.fill(0);
            for r in 0..4 {
                for el in 0..QK_K {
                    let q = qs[(el / 8) * 32 + r * 8 + el % 8] as i8 as i32;
                    let idx = 16 * (el / 64) + 4 * r + (el / 16) % 4;
                    let cur = i16::from_le_bytes([bsums[2 * idx], bsums[2 * idx + 1]]);
                    let nv = (cur as i32 + q) as i16;
                    bsums[2 * idx..2 * idx + 2].copy_from_slice(&nv.to_le_bytes());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K 8x8 gemv / gemm (scalar bodies)
// ---------------------------------------------------------------------------

/// Per-(8-column, sub-block) decoded `block_q4_Kx8` scales/mins: the 8 columns'
/// 6-bit `(scale, min)` pairs of sub-block `sb`.
#[inline]
fn q4k_block_scales(tile: &[u8], sb: usize) -> ([u8; 8], [u8; 8]) {
    let mut packed = [0u8; K_SCALE_SIZE];
    packed.copy_from_slice(&tile[32 + q4k_scale_off(sb)..32 + q4k_scale_off(sb) + K_SCALE_SIZE]);
    let dec = unpack_q4k_scales_bytes(&packed);
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    s.copy_from_slice(&dec[0..8]);
    m.copy_from_slice(&dec[8..16]);
    (s, m)
}

#[inline]
fn q4k_half_f32(b: &[u8]) -> f32 {
    f16_from_bytes(b).to_f32()
}

/// Scalar body of the 8x8 gemv for one activation row: the arithmetic of the
/// C's AVX2 `ggml_gemv_q4_K_8x8_q8_K` (see the section header), which is also
/// what [`gemv_q4_K_8x8_q8_K`] runs on AVX2 hosts.
///
/// `vy` points at one plain `block_q8_K` row (`nb` blocks of 292 bytes, the
/// layout `quantize_row_q8_K_ref` writes); `nc` must be a multiple of 8 and the
/// 8-column groups are addressed as `vx + x*nb*1152`.
pub fn gemv_q4_K_8x8_q8_K_scalar(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    let nb = n / QK_K;
    assert_eq!(n % QK_K, 0, "gemv_q4_K: n % QK_K != 0");
    assert_eq!(nc % 8, 0, "gemv_q4_K: nc % 8 != 0");
    for x in 0..nc / 8 {
        let bmat = &vx[x * nb * BLOCK_Q4_KX8_SIZE..];
        let mut sumf = [0f32; 8];
        let mut summin = [0f32; 8];
        for b in 0..nb {
            let tile = &bmat[b * BLOCK_Q4_KX8_SIZE..(b + 1) * BLOCK_Q4_KX8_SIZE];
            let q8 = &vy[b * BLOCK_Q8_K_SIZE..(b + 1) * BLOCK_Q8_K_SIZE];
            let d_row = f32::from_le_bytes(q8[0..4].try_into().unwrap());
            let q8s = &q8[4..4 + QK_K];
            let bsums: Vec<i16> = (0..16)
                .map(|j| i16::from_le_bytes([q8[260 + 2 * j], q8[261 + 2 * j]]))
                .collect();
            let mut iacc = [0i32; 8];
            let mut imin = [0i32; 8];
            for g in 0..4 {
                let (s_lo, m_lo) = q4k_block_scales(tile, 2 * g);
                let (s_hi, m_hi) = q4k_block_scales(tile, 2 * g + 1);
                let mut d_lo = [0i32; 8];
                let mut d_hi = [0i32; 8];
                for m in 0..32 {
                    let ym = q8s[64 * g + m] as i8 as i32;
                    let yh = q8s[64 * g + 32 + m] as i8 as i32;
                    for c in 0..8 {
                        let byte = tile[128 + 256 * g + 8 * c + 64 * (m / 8) + (m % 8)];
                        d_lo[c] += (byte & 0xF) as i32 * ym;
                        d_hi[c] += (byte >> 4) as i32 * yh;
                    }
                }
                let b_lo = bsums[4 * g] as i32 + bsums[4 * g + 1] as i32;
                let b_hi = bsums[4 * g + 2] as i32 + bsums[4 * g + 3] as i32;
                for c in 0..8 {
                    iacc[c] += d_lo[c] * s_lo[c] as i32 + d_hi[c] * s_hi[c] as i32;
                    imin[c] += m_lo[c] as i32 * b_lo + m_hi[c] as i32 * b_hi;
                }
            }
            for c in 0..8 {
                let d_col = q4k_half_f32(&tile[2 * c..2 * c + 2]);
                let dmin_col = q4k_half_f32(&tile[16 + 2 * c..16 + 2 * c + 2]);
                sumf[c] = (iacc[c] as f32).mul_add(d_col * d_row, sumf[c]);
                summin[c] = (imin[c] as f32).mul_add(dmin_col * d_row, summin[c]);
            }
        }
        for c in 0..8 {
            s[x * 8 + c] = sumf[c] - summin[c];
        }
    }
}

/// Scalar body of the 8x8 gemm (4 activation rows per `block_q8_Kx4`, `nr`
/// rows total, row stride `bs`, output row `y*4+i` at `s[(y*4+i)*bs + x*8]`).
pub fn gemm_q4_K_8x8_q8_K_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[u8],
    vy: &[u8],
    nr: usize,
    nc: usize,
) {
    let nb = n / QK_K;
    assert_eq!(n % QK_K, 0, "gemm_q4_K: n % QK_K != 0");
    assert_eq!(nr % 4, 0, "gemm_q4_K: nr % 4 != 0");
    assert_eq!(nc % 8, 0, "gemm_q4_K: nc % 8 != 0");
    for y in 0..nr / 4 {
        let ablk = &vy[y * nb * BLOCK_Q8_KX4_SIZE..];
        for x in 0..nc / 8 {
            let bmat = &vx[x * nb * BLOCK_Q4_KX8_SIZE..];
            let mut acc = [[0f32; 8]; 4];
            let mut accmin = [[0f32; 8]; 4];
            for b in 0..nb {
                let tile = &bmat[b * BLOCK_Q4_KX8_SIZE..(b + 1) * BLOCK_Q4_KX8_SIZE];
                let a = &ablk[b * BLOCK_Q8_KX4_SIZE..(b + 1) * BLOCK_Q8_KX4_SIZE];
                let q8s = &a[16..16 + QK_K * 4];
                let mut iacc = [[0i32; 8]; 4];
                let mut imin = [[0i32; 8]; 4];
                for g in 0..4 {
                    let (s_lo, m_lo) = q4k_block_scales(tile, 2 * g);
                    let (s_hi, m_hi) = q4k_block_scales(tile, 2 * g + 1);
                    let mut d_lo = [[0i32; 8]; 4];
                    let mut d_hi = [[0i32; 8]; 4];
                    for m in 0..32 {
                        for r in 0..4 {
                            // element (64g+m) of row r: qs[(el/8)*32 + r*8 + el%8]
                            let p = 256 * g + 32 * (m / 8) + 8 * r + (m % 8);
                            let ym = q8s[p] as i8 as i32;
                            let yh = q8s[p + 128] as i8 as i32;
                            for c in 0..8 {
                                let byte = tile[128 + 256 * g + 8 * c + 64 * (m / 8) + (m % 8)];
                                d_lo[r][c] += (byte & 0xF) as i32 * ym;
                                d_hi[r][c] += (byte >> 4) as i32 * yh;
                            }
                        }
                    }
                    for r in 0..4 {
                        // bsums[16*(el/64) + 4*r + (el/16)%4]
                        let bs_at = |t: usize| -> i32 {
                            let j = 16 * g + 4 * r + t;
                            i16::from_le_bytes([a[1040 + 2 * j], a[1041 + 2 * j]]) as i32
                        };
                        let (b_lo, b_hi) = (bs_at(0) + bs_at(1), bs_at(2) + bs_at(3));
                        let d_row = f32::from_le_bytes(a[4 * r..4 * r + 4].try_into().unwrap());
                        for c in 0..8 {
                            iacc[r][c] = d_lo[r][c] * s_lo[c] as i32 + d_hi[r][c] * s_hi[c] as i32;
                            imin[r][c] = m_lo[c] as i32 * b_lo + m_hi[c] as i32 * b_hi;
                            // one fma per (super block, sub block pair) — the
                            // C's gemm body (arch/x86/repack.cpp:3464-3477) fma's
                            // inside its `sb` loop, unlike the gemv's single
                            // per-`b` fma (repack.cpp:1668). The two kernels round
                            // differently; the port mirrors each.
                            let d_col = q4k_half_f32(&tile[2 * c..2 * c + 2]);
                            let dmin_col = q4k_half_f32(&tile[16 + 2 * c..16 + 2 * c + 2]);
                            acc[r][c] = (iacc[r][c] as f32).mul_add(d_col * d_row, acc[r][c]);
                            accmin[r][c] =
                                (imin[r][c] as f32).mul_add(dmin_col * d_row, accmin[r][c]);
                        }
                    }
                }
            }
            for r in 0..4 {
                let o = (y * 4 + r) * bs + x * 8;
                for c in 0..8 {
                    s[o + c] = acc[r][c] - accmin[r][c];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K 8x8 gemv / gemm (AVX2 lane kernels)
// ---------------------------------------------------------------------------

/// `gemm_q4_K_8x8_q8_K`'s per-(4 rows, 8 columns) body split into the two
/// sub-block halves: see [`simd_x86_q4k`] for the lane network.
#[cfg(target_arch = "x86_64")]
mod simd_x86_q4k {
    use super::{
        BLOCK_Q4_KX8_SIZE, BLOCK_Q8_KX4_SIZE, BLOCK_Q8_K_SIZE, QK_K,
    };
    use core::arch::x86_64::*;

    /// The six shuffle/blend constants of the C bodies that are plain data
    /// (`_mm_set_epi8` / `_mm256_set_epi32` in that order, arch/x86/repack.cpp:
    /// 1488-1497 for the gemv).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn gemv_masks() -> (__m128i, __m128i, __m256i, __m256i) {
        (
            _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0),
            _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0),
            _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0),
            _mm256_set1_epi8(0x0F),
        )
    }

    /// `GGML_F32Cx8_LOAD` — 8 ggml_half values widened to f32, in order
    /// (arch/x86/repack.cpp:36; needs F16C).
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn f32x8_load(p: *const u8) -> __m256 {
        _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
    }

    /// `GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)` (arch/x86/repack.cpp:38).
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn f32x8_rearrange_load(p: *const u8, mask: __m128i) -> __m256 {
        _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128(p as *const __m128i), mask))
    }

    /// `ggml_gemv_q4_K_8x8_q8_K` AVX2 body (arch/x86/repack.cpp:1486-1677),
    /// transcribed instruction by instruction. `nr` is 1 in every call the
    /// reference makes (forward_mul_mat's tail rows, repack.cpp:4643-4647), so
    /// the `y` loop and the `y*nr + x*8` output index collapse to row 0 —
    /// kept explicit to mirror the C.
    ///
    /// # Safety
    /// `vx` must address `nc * nb * 1152` repacked bytes, `vy` one `block_q8_K`
    /// row of `nb * 292` bytes, and `s` `nc` floats.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemv(n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nc: usize) {
        let nb = n / QK_K;
        let b_nb = n / QK_K;
        let (deltamask, scalemask, finalpermutemask, m4b) = gemv_masks();
        let kmask1: u32 = 0x3f3f_3f3f;
        let kmask2: u32 = 0x0f0f_0f0f;
        let kmask3: u32 = 0x0303_0303;

        for y in 0..1usize {
            let a_ptr = vy;
            for x in 0..nc / 8 {
                let b_ptr = vx.add(x * b_nb * BLOCK_Q4_KX8_SIZE);
                let mut acc_row = _mm256_setzero_ps();
                let mut acc_min_rows = _mm256_setzero_ps();
                for b in 0..nb {
                    let blk = b_ptr.add(b * BLOCK_Q4_KX8_SIZE);
                    let a_blk = a_ptr.add(b * BLOCK_Q8_K_SIZE);
                    let row_scale_f32 =
                        _mm256_set1_ps(f32::from_le_bytes(*(a_blk as *const [u8; 4]).cast()));
                    let col_scale_f32 = f32x8_rearrange_load(blk, deltamask);
                    let col_dmin_f32 = f32x8_load(blk.add(16));
                    let mut iacc_b = _mm256_setzero_si256();
                    let mut iacc_min_b = _mm256_setzero_si256();
                    let q8sums =
                        _mm256_loadu_si256(a_blk.add(4 + QK_K) as *const __m256i);
                    let mut q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
                        _mm256_castsi256_si128(q8sums),
                        _mm256_extracti128_si256::<1>(q8sums),
                    ));
                    q8s = _mm256_permute2f128_si256::<0>(q8s, q8s);
                    for sb in 0..QK_K / 64 {
                        let qs = blk.add(128 + sb * 256);
                        let rhs_raw_0123_0 = _mm256_loadu_si256(qs as *const __m256i);
                        let rhs_raw_4567_0 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
                        let rhs_raw_0123_1 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
                        let rhs_raw_4567_1 = _mm256_loadu_si256(qs.add(96) as *const __m256i);
                        let rhs_raw_0123_2 = _mm256_loadu_si256(qs.add(128) as *const __m256i);
                        let rhs_raw_4567_2 = _mm256_loadu_si256(qs.add(160) as *const __m256i);
                        let rhs_raw_0123_3 = _mm256_loadu_si256(qs.add(192) as *const __m256i);
                        let rhs_raw_4567_3 = _mm256_loadu_si256(qs.add(224) as *const __m256i);
                        // 4-bit -> 8-bit
                        let r0123_00 = _mm256_and_si256(rhs_raw_0123_0, m4b);
                        let r4567_00 = _mm256_and_si256(rhs_raw_4567_0, m4b);
                        let r0123_01 = _mm256_and_si256(rhs_raw_0123_1, m4b);
                        let r4567_01 = _mm256_and_si256(rhs_raw_4567_1, m4b);
                        let r0123_02 = _mm256_and_si256(rhs_raw_0123_2, m4b);
                        let r4567_02 = _mm256_and_si256(rhs_raw_4567_2, m4b);
                        let r0123_03 = _mm256_and_si256(rhs_raw_0123_3, m4b);
                        let r4567_03 = _mm256_and_si256(rhs_raw_4567_3, m4b);
                        let r0123_10 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0123_0), m4b);
                        let r4567_10 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_4567_0), m4b);
                        let r0123_11 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0123_1), m4b);
                        let r4567_11 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_4567_1), m4b);
                        let r0123_12 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0123_2), m4b);
                        let r4567_12 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_4567_2), m4b);
                        let r0123_13 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0123_3), m4b);
                        let r4567_13 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_4567_3), m4b);

                        let mut utmp_0 = [0u32; 4];
                        let mut utmp_1 = [0u32; 4];
                        let sc = blk.add(32);
                        unpack_utmp(sc.add(24 * sb), &mut utmp_0, kmask1, kmask2, kmask3);
                        unpack_utmp(sc.add(12 + sb * 24), &mut utmp_1, kmask1, kmask2, kmask3);
                        let ms0 = _mm_set_epi32(
                            utmp_0[3] as i32,
                            utmp_0[2] as i32,
                            utmp_0[1] as i32,
                            utmp_0[0] as i32,
                        );
                        let scales_rearrange_0 = _mm_shuffle_epi8(ms0, scalemask);
                        let scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);
                        let ms1 = _mm_set_epi32(
                            utmp_1[3] as i32,
                            utmp_1[2] as i32,
                            utmp_1[1] as i32,
                            utmp_1[0] as i32,
                        );
                        let scales_rearrange_1 = _mm_shuffle_epi8(ms1, scalemask);
                        let scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);
                        let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                            _mm_shuffle_epi32::<78>(ms0),
                            _mm_shuffle_epi32::<78>(ms1),
                        ));

                        let aq = a_blk.add(4 + sb * 64);
                        let mut lhs_vec_00 = _mm256_castsi128_si256(
                            _mm_loadu_si128(aq as *const __m128i),
                        );
                        let mut lhs_vec_01 = _mm256_castsi128_si256(
                            _mm_loadu_si128(aq.add(16) as *const __m128i),
                        );
                        let mut lhs_vec_10 = _mm256_castsi128_si256(
                            _mm_loadu_si128(aq.add(32) as *const __m128i),
                        );
                        let mut lhs_vec_11 = _mm256_castsi128_si256(
                            _mm_loadu_si128(aq.add(48) as *const __m128i),
                        );
                        lhs_vec_00 = _mm256_permute2f128_si256::<0>(lhs_vec_00, lhs_vec_00);
                        lhs_vec_01 = _mm256_permute2f128_si256::<0>(lhs_vec_01, lhs_vec_01);
                        lhs_vec_10 = _mm256_permute2f128_si256::<0>(lhs_vec_10, lhs_vec_10);
                        lhs_vec_11 = _mm256_permute2f128_si256::<0>(lhs_vec_11, lhs_vec_11);

                        let mut iacc_0 = _mm256_setzero_si256();
                        let mut iacc_1 = _mm256_setzero_si256();
                        macro_rules! macc {
                            ($acc:ident, $rhs:expr, $lhs:expr) => {
                                $acc = _mm256_add_epi16(
                                    $acc,
                                    _mm256_maddubs_epi16($rhs, $lhs),
                                );
                            };
                        }
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(r0123_00, _mm256_shuffle_epi32::<177>(r4567_00)),
                            _mm256_shuffle_epi32::<0>(lhs_vec_00)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_00), r4567_00),
                            _mm256_shuffle_epi32::<85>(lhs_vec_00)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(r0123_01, _mm256_shuffle_epi32::<177>(r4567_01)),
                            _mm256_shuffle_epi32::<170>(lhs_vec_00)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_01), r4567_01),
                            _mm256_shuffle_epi32::<255>(lhs_vec_00)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(r0123_02, _mm256_shuffle_epi32::<177>(r4567_02)),
                            _mm256_shuffle_epi32::<0>(lhs_vec_01)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_02), r4567_02),
                            _mm256_shuffle_epi32::<85>(lhs_vec_01)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(r0123_03, _mm256_shuffle_epi32::<177>(r4567_03)),
                            _mm256_shuffle_epi32::<170>(lhs_vec_01)
                        );
                        macc!(
                            iacc_0,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_03), r4567_03),
                            _mm256_shuffle_epi32::<255>(lhs_vec_01)
                        );
                        iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(r0123_10, _mm256_shuffle_epi32::<177>(r4567_10)),
                            _mm256_shuffle_epi32::<0>(lhs_vec_10)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_10), r4567_10),
                            _mm256_shuffle_epi32::<85>(lhs_vec_10)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(r0123_11, _mm256_shuffle_epi32::<177>(r4567_11)),
                            _mm256_shuffle_epi32::<170>(lhs_vec_10)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_11), r4567_11),
                            _mm256_shuffle_epi32::<255>(lhs_vec_10)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(r0123_12, _mm256_shuffle_epi32::<177>(r4567_12)),
                            _mm256_shuffle_epi32::<0>(lhs_vec_11)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_12), r4567_12),
                            _mm256_shuffle_epi32::<85>(lhs_vec_11)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(r0123_13, _mm256_shuffle_epi32::<177>(r4567_13)),
                            _mm256_shuffle_epi32::<170>(lhs_vec_11)
                        );
                        macc!(
                            iacc_1,
                            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_13), r4567_13),
                            _mm256_shuffle_epi32::<255>(lhs_vec_11)
                        );
                        iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                        let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);
                        let q8s_sb = _mm256_shuffle_epi32::<0>(q8s);
                        let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                        q8s = _mm256_bsrli_epi128::<4>(q8s);
                        iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                        iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                    }
                    acc_row = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_b),
                        _mm256_mul_ps(col_scale_f32, row_scale_f32),
                        acc_row,
                    );
                    acc_min_rows = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_min_b),
                        _mm256_mul_ps(col_dmin_f32, row_scale_f32),
                        acc_min_rows,
                    );
                }
                acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
                _mm256_storeu_ps(
                    s.add(y * 1 + x * 8),
                    _mm256_sub_ps(acc_row, acc_min_rows),
                );
                super::note_q4k_gemv_call();
            }
        }
    }

    /// The C's 12-byte `utmp` unpacking (repack.cpp:990-995).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_utmp(p: *const u8, utmp: &mut [u32; 4], kmask1: u32, kmask2: u32, kmask3: u32) {
        core::ptr::copy_nonoverlapping(p, utmp.as_mut_ptr() as *mut u8, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
    }

    /// The 8x8 gemm body of arch/x86/repack.cpp:3158-3486 (the AVX2 path the
    /// reference takes when `__AVX512BW__`/`__AVX512DQ__` are absent, and the
    /// tail body of the AVX512 build): 4 activation rows of one `block_q8_Kx4`
    /// x 8 weight columns per pass, 4 sub-block pairs per super block.
    ///
    /// The lane network is the C's; every lane position, blend mask and
    /// shuffle immediate is transcribed from the listed lines. The port's
    /// `scales_0`/`mins_01` construction is shared with the gemv above (the C
    /// writes the same two lines in both bodies).
    ///
    /// `xstart` is the C's `xstart` (arch/x86/repack.cpp:2076): 0 on an
    /// AVX2-only host (the whole matrix) and `anc/8` on an AVX512 host, where
    /// the 512-bit section above handled the `nc % 16 == 0` prefix and this
    /// body serves the `nc % 16 == 8` tail columns for every row
    /// (:2811-2813 resets `y = 0` before these loops).
    ///
    /// # Safety
    /// `vx` must address `nc * nb * 1152` repacked bytes, `vy` `nr/4 * nb`
    /// `block_q8_Kx4` tiles, and `s` at least `nr * bs` floats.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm_256_section(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
        xstart: usize,
    ) {
        let nb = n / QK_K;
        let b_nb = n / QK_K;
        let kmask1: u32 = 0x3f3f_3f3f;
        let kmask2: u32 = 0x0f0f_0f0f;
        let kmask3: u32 = 0x0303_0303;
        let m4b = _mm256_set1_epi8(0x0F);
        let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
        for y in 0..nr / 4 {
            let a_ptr = vy.add(y * nb * BLOCK_Q8_KX4_SIZE);
            for x in xstart..nc / 8 {
                let b_ptr = vx.add(x * b_nb * BLOCK_Q4_KX8_SIZE);
                let mut acc_rows = [_mm256_setzero_ps(); 4];
                let mut acc_min_rows = [_mm256_setzero_ps(); 4];
                for b in 0..nb {
                    let blk = b_ptr.add(b * BLOCK_Q4_KX8_SIZE);
                    let a_blk = a_ptr.add(b * BLOCK_Q8_KX4_SIZE);
                    let col_scale_f32 = f32x8_load(blk);
                    let col_dmin_f32 = f32x8_load(blk.add(16));
                    for sb in 0..QK_K / 64 {
                        let qs = blk.add(128 + sb * 256);
                        let rhs_raw_0123_0 = _mm256_loadu_si256(qs as *const __m256i);
                        let rhs_raw_4567_0 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
                        let rhs_raw_0123_1 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
                        let rhs_raw_4567_1 = _mm256_loadu_si256(qs.add(96) as *const __m256i);
                        let rhs_raw_0123_2 = _mm256_loadu_si256(qs.add(128) as *const __m256i);
                        let rhs_raw_4567_2 = _mm256_loadu_si256(qs.add(160) as *const __m256i);
                        let rhs_raw_0123_3 = _mm256_loadu_si256(qs.add(192) as *const __m256i);
                        let rhs_raw_4567_3 = _mm256_loadu_si256(qs.add(224) as *const __m256i);
                        let rhs_raw_0145_0 = _mm256_blend_epi32::<240>(
                            rhs_raw_0123_0,
                            _mm256_permutevar8x32_epi32(rhs_raw_4567_0, required_order),
                        );
                        let rhs_raw_2367_0 = _mm256_blend_epi32::<240>(
                            _mm256_permutevar8x32_epi32(rhs_raw_0123_0, required_order),
                            rhs_raw_4567_0,
                        );
                        let rhs_raw_0145_1 = _mm256_blend_epi32::<240>(
                            rhs_raw_0123_1,
                            _mm256_permutevar8x32_epi32(rhs_raw_4567_1, required_order),
                        );
                        let rhs_raw_2367_1 = _mm256_blend_epi32::<240>(
                            _mm256_permutevar8x32_epi32(rhs_raw_0123_1, required_order),
                            rhs_raw_4567_1,
                        );
                        let rhs_raw_0145_2 = _mm256_blend_epi32::<240>(
                            rhs_raw_0123_2,
                            _mm256_permutevar8x32_epi32(rhs_raw_4567_2, required_order),
                        );
                        let rhs_raw_2367_2 = _mm256_blend_epi32::<240>(
                            _mm256_permutevar8x32_epi32(rhs_raw_0123_2, required_order),
                            rhs_raw_4567_2,
                        );
                        let rhs_raw_0145_3 = _mm256_blend_epi32::<240>(
                            rhs_raw_0123_3,
                            _mm256_permutevar8x32_epi32(rhs_raw_4567_3, required_order),
                        );
                        let rhs_raw_2367_3 = _mm256_blend_epi32::<240>(
                            _mm256_permutevar8x32_epi32(rhs_raw_0123_3, required_order),
                            rhs_raw_4567_3,
                        );
                        // 4-bit -> 8-bit, per sub block half
                        let r0145_00 = _mm256_and_si256(rhs_raw_0145_0, m4b);
                        let r2367_00 = _mm256_and_si256(rhs_raw_2367_0, m4b);
                        let r0145_01 = _mm256_and_si256(rhs_raw_0145_1, m4b);
                        let r2367_01 = _mm256_and_si256(rhs_raw_2367_1, m4b);
                        let r0145_02 = _mm256_and_si256(rhs_raw_0145_2, m4b);
                        let r2367_02 = _mm256_and_si256(rhs_raw_2367_2, m4b);
                        let r0145_03 = _mm256_and_si256(rhs_raw_0145_3, m4b);
                        let r2367_03 = _mm256_and_si256(rhs_raw_2367_3, m4b);
                        let r0145_10 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0145_0), m4b);
                        let r2367_10 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_2367_0), m4b);
                        let r0145_11 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0145_1), m4b);
                        let r2367_11 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_2367_1), m4b);
                        let r0145_12 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0145_2), m4b);
                        let r2367_12 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_2367_2), m4b);
                        let r0145_13 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_0145_3), m4b);
                        let r2367_13 =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_raw_2367_3), m4b);
                        // shuffle pattern one: elements 0-3 of each 8-byte slice
                        let s0145_00 = _mm256_shuffle_epi32::<136>(r0145_00);
                        let s2367_00 = _mm256_shuffle_epi32::<136>(r2367_00);
                        let s0145_01 = _mm256_shuffle_epi32::<136>(r0145_01);
                        let s2367_01 = _mm256_shuffle_epi32::<136>(r2367_01);
                        let s0145_02 = _mm256_shuffle_epi32::<136>(r0145_02);
                        let s2367_02 = _mm256_shuffle_epi32::<136>(r2367_02);
                        let s0145_03 = _mm256_shuffle_epi32::<136>(r0145_03);
                        let s2367_03 = _mm256_shuffle_epi32::<136>(r2367_03);
                        let s0145_10 = _mm256_shuffle_epi32::<136>(r0145_10);
                        let s2367_10 = _mm256_shuffle_epi32::<136>(r2367_10);
                        let s0145_11 = _mm256_shuffle_epi32::<136>(r0145_11);
                        let s2367_11 = _mm256_shuffle_epi32::<136>(r2367_11);
                        let s0145_12 = _mm256_shuffle_epi32::<136>(r0145_12);
                        let s2367_12 = _mm256_shuffle_epi32::<136>(r2367_12);
                        let s0145_13 = _mm256_shuffle_epi32::<136>(r0145_13);
                        let s2367_13 = _mm256_shuffle_epi32::<136>(r2367_13);
                        // shuffle pattern two: elements 4-7 of each 8-byte slice
                        let s0145_00b = _mm256_shuffle_epi32::<221>(r0145_00);
                        let s2367_00b = _mm256_shuffle_epi32::<221>(r2367_00);
                        let s0145_01b = _mm256_shuffle_epi32::<221>(r0145_01);
                        let s2367_01b = _mm256_shuffle_epi32::<221>(r2367_01);
                        let s0145_02b = _mm256_shuffle_epi32::<221>(r0145_02);
                        let s2367_02b = _mm256_shuffle_epi32::<221>(r2367_02);
                        let s0145_03b = _mm256_shuffle_epi32::<221>(r0145_03);
                        let s2367_03b = _mm256_shuffle_epi32::<221>(r2367_03);
                        let s0145_10b = _mm256_shuffle_epi32::<221>(r0145_10);
                        let s2367_10b = _mm256_shuffle_epi32::<221>(r2367_10);
                        let s0145_11b = _mm256_shuffle_epi32::<221>(r0145_11);
                        let s2367_11b = _mm256_shuffle_epi32::<221>(r2367_11);
                        let s0145_12b = _mm256_shuffle_epi32::<221>(r0145_12);
                        let s2367_12b = _mm256_shuffle_epi32::<221>(r2367_12);
                        let s0145_13b = _mm256_shuffle_epi32::<221>(r0145_13);
                        let s2367_13b = _mm256_shuffle_epi32::<221>(r2367_13);

                        let mut utmp_0 = [0u32; 4];
                        let mut utmp_1 = [0u32; 4];
                        let sc = blk.add(32);
                        unpack_utmp(sc.add(24 * sb), &mut utmp_0, kmask1, kmask2, kmask3);
                        unpack_utmp(sc.add(12 + sb * 24), &mut utmp_1, kmask1, kmask2, kmask3);
                        let ms0 = _mm_set_epi32(
                            utmp_0[3] as i32,
                            utmp_0[2] as i32,
                            utmp_0[1] as i32,
                            utmp_0[0] as i32,
                        );
                        let scales_0 =
                            _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(ms0, ms0));
                        let ms1 = _mm_set_epi32(
                            utmp_1[3] as i32,
                            utmp_1[2] as i32,
                            utmp_1[1] as i32,
                            utmp_1[0] as i32,
                        );
                        let scales_1 =
                            _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(ms1, ms1));
                        let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                            _mm_shuffle_epi32::<78>(ms0),
                            _mm_shuffle_epi32::<78>(ms1),
                        ));
                        let scale_0145_0 = _mm256_shuffle_epi32::<68>(scales_0);
                        let scale_2367_0 = _mm256_shuffle_epi32::<238>(scales_0);
                        let scale_0145_1 = _mm256_shuffle_epi32::<68>(scales_1);
                        let scale_2367_1 = _mm256_shuffle_epi32::<238>(scales_1);

                        // LHS: one block_q8_Kx4 holds 4 rows x 64 elements per
                        // 256-byte k-group; `_01` = rows 0,1 (both lanes),
                        // `_23` = rows 2,3 (arch/x86/repack.cpp:3323-3346)
                        let aq = a_blk.add(16 + sb * 256);
                        let l_0123_00 = _mm256_loadu_si256(aq as *const __m256i);
                        let l_01_00 = _mm256_permute2f128_si256::<0>(l_0123_00, l_0123_00);
                        let l_23_00 = _mm256_permute2f128_si256::<17>(l_0123_00, l_0123_00);
                        let l_0123_01 = _mm256_loadu_si256(aq.add(32) as *const __m256i);
                        let l_01_01 = _mm256_permute2f128_si256::<0>(l_0123_01, l_0123_01);
                        let l_23_01 = _mm256_permute2f128_si256::<17>(l_0123_01, l_0123_01);
                        let l_0123_02 = _mm256_loadu_si256(aq.add(64) as *const __m256i);
                        let l_01_02 = _mm256_permute2f128_si256::<0>(l_0123_02, l_0123_02);
                        let l_23_02 = _mm256_permute2f128_si256::<17>(l_0123_02, l_0123_02);
                        let l_0123_03 = _mm256_loadu_si256(aq.add(96) as *const __m256i);
                        let l_01_03 = _mm256_permute2f128_si256::<0>(l_0123_03, l_0123_03);
                        let l_23_03 = _mm256_permute2f128_si256::<17>(l_0123_03, l_0123_03);
                        let l_0123_10 = _mm256_loadu_si256(aq.add(128) as *const __m256i);
                        let l_01_10 = _mm256_permute2f128_si256::<0>(l_0123_10, l_0123_10);
                        let l_23_10 = _mm256_permute2f128_si256::<17>(l_0123_10, l_0123_10);
                        let l_0123_11 = _mm256_loadu_si256(aq.add(160) as *const __m256i);
                        let l_01_11 = _mm256_permute2f128_si256::<0>(l_0123_11, l_0123_11);
                        let l_23_11 = _mm256_permute2f128_si256::<17>(l_0123_11, l_0123_11);
                        let l_0123_12 = _mm256_loadu_si256(aq.add(192) as *const __m256i);
                        let l_01_12 = _mm256_permute2f128_si256::<0>(l_0123_12, l_0123_12);
                        let l_23_12 = _mm256_permute2f128_si256::<17>(l_0123_12, l_0123_12);
                        let l_0123_13 = _mm256_loadu_si256(aq.add(224) as *const __m256i);
                        let l_01_13 = _mm256_permute2f128_si256::<0>(l_0123_13, l_0123_13);
                        let l_23_13 = _mm256_permute2f128_si256::<17>(l_0123_13, l_0123_13);

                        // bsums: 16 i16 for the 4 rows of this k-group; the SSE
                        // `hadd` pairs the 16-element sums and the broadcast +
                        // dword select then pick rows 0..4
                        // (arch/x86/repack.cpp:3349-3351, :3469-3472).
                        // NB the C's pointer arithmetic is in `int16_t` units:
                        // `bsums + 16 * sb` is 32 * sb bytes.
                        let bsums = _mm256_loadu_si256(
                            a_blk.add(1040 + 32 * sb) as *const __m256i
                        );
                        let bsums_hsum = _mm256_castsi128_si256(_mm_hadd_epi16(
                            _mm256_castsi256_si128(bsums),
                            _mm256_extracti128_si256::<1>(bsums),
                        ));
                        let bsums_hsum =
                            _mm256_permute2x128_si256::<0>(bsums_hsum, bsums_hsum);

                        // shuffle pattern one / two of the LHS (rows 0,1 vs 2,3)
                        macro_rules! sh {
                            ($v:expr, $i:literal) => {
                                _mm256_shuffle_epi32::<$i>($v)
                            };
                        }
                        let l01_00_sp1 = sh!(l_01_00, 160);
                        let l23_00_sp1 = sh!(l_23_00, 160);
                        let l01_01_sp1 = sh!(l_01_01, 160);
                        let l23_01_sp1 = sh!(l_23_01, 160);
                        let l01_02_sp1 = sh!(l_01_02, 160);
                        let l23_02_sp1 = sh!(l_23_02, 160);
                        let l01_03_sp1 = sh!(l_01_03, 160);
                        let l23_03_sp1 = sh!(l_23_03, 160);
                        let l01_10_sp1 = sh!(l_01_10, 160);
                        let l23_10_sp1 = sh!(l_23_10, 160);
                        let l01_11_sp1 = sh!(l_01_11, 160);
                        let l23_11_sp1 = sh!(l_23_11, 160);
                        let l01_12_sp1 = sh!(l_01_12, 160);
                        let l23_12_sp1 = sh!(l_23_12, 160);
                        let l01_13_sp1 = sh!(l_01_13, 160);
                        let l23_13_sp1 = sh!(l_23_13, 160);
                        let l01_00_sp2 = sh!(l_01_00, 245);
                        let l23_00_sp2 = sh!(l_23_00, 245);
                        let l01_01_sp2 = sh!(l_01_01, 245);
                        let l23_01_sp2 = sh!(l_23_01, 245);
                        let l01_02_sp2 = sh!(l_01_02, 245);
                        let l23_02_sp2 = sh!(l_23_02, 245);
                        let l01_03_sp2 = sh!(l_01_03, 245);
                        let l23_03_sp2 = sh!(l_23_03, 245);
                        let l01_10_sp2 = sh!(l_01_10, 245);
                        let l23_10_sp2 = sh!(l_23_10, 245);
                        let l01_11_sp2 = sh!(l_01_11, 245);
                        let l23_11_sp2 = sh!(l_23_11, 245);
                        let l01_12_sp2 = sh!(l_01_12, 245);
                        let l23_12_sp2 = sh!(l_23_12, 245);
                        let l01_13_sp2 = sh!(l_01_13, 245);
                        let l23_13_sp2 = sh!(l_23_13, 245);

                        // 4 maddubs per accumulator (the four 8-byte slices of
                        // a sub block), accumulated in int16 (max 4*3810 < 2^15)
                        macro_rules! acc4 {
                            ($r03:expr, $r02:expr, $r01:expr, $r00:expr,
                             $l03:expr, $l02:expr, $l01:expr, $l00:expr) => {{
                                let a = _mm256_maddubs_epi16($r03, $l03);
                                let b = _mm256_maddubs_epi16($r02, $l02);
                                let c = _mm256_maddubs_epi16($r01, $l01);
                                let d = _mm256_maddubs_epi16($r00, $l00);
                                _mm256_add_epi16(_mm256_add_epi16(a, b), _mm256_add_epi16(c, d))
                            }};
                        }
                        let m00_0 = _mm256_add_epi16(
                            acc4!(s0145_03, s0145_02, s0145_01, s0145_00,
                                  l01_03_sp1, l01_02_sp1, l01_01_sp1, l01_00_sp1),
                            acc4!(s0145_03b, s0145_02b, s0145_01b, s0145_00b,
                                  l01_03_sp2, l01_02_sp2, l01_01_sp2, l01_00_sp2),
                        );
                        let m01_0 = _mm256_add_epi16(
                            acc4!(s2367_03, s2367_02, s2367_01, s2367_00,
                                  l01_03_sp1, l01_02_sp1, l01_01_sp1, l01_00_sp1),
                            acc4!(s2367_03b, s2367_02b, s2367_01b, s2367_00b,
                                  l01_03_sp2, l01_02_sp2, l01_01_sp2, l01_00_sp2),
                        );
                        let m10_0 = _mm256_add_epi16(
                            acc4!(s0145_03, s0145_02, s0145_01, s0145_00,
                                  l23_03_sp1, l23_02_sp1, l23_01_sp1, l23_00_sp1),
                            acc4!(s0145_03b, s0145_02b, s0145_01b, s0145_00b,
                                  l23_03_sp2, l23_02_sp2, l23_01_sp2, l23_00_sp2),
                        );
                        let m11_0 = _mm256_add_epi16(
                            acc4!(s2367_03, s2367_02, s2367_01, s2367_00,
                                  l23_03_sp1, l23_02_sp1, l23_01_sp1, l23_00_sp1),
                            acc4!(s2367_03b, s2367_02b, s2367_01b, s2367_00b,
                                  l23_03_sp2, l23_02_sp2, l23_01_sp2, l23_00_sp2),
                        );
                        let m00_1 = _mm256_add_epi16(
                            acc4!(s0145_13, s0145_12, s0145_11, s0145_10,
                                  l01_13_sp1, l01_12_sp1, l01_11_sp1, l01_10_sp1),
                            acc4!(s0145_13b, s0145_12b, s0145_11b, s0145_10b,
                                  l01_13_sp2, l01_12_sp2, l01_11_sp2, l01_10_sp2),
                        );
                        let m01_1 = _mm256_add_epi16(
                            acc4!(s2367_13, s2367_12, s2367_11, s2367_10,
                                  l01_13_sp1, l01_12_sp1, l01_11_sp1, l01_10_sp1),
                            acc4!(s2367_13b, s2367_12b, s2367_11b, s2367_10b,
                                  l01_13_sp2, l01_12_sp2, l01_11_sp2, l01_10_sp2),
                        );
                        let m10_1 = _mm256_add_epi16(
                            acc4!(s0145_13, s0145_12, s0145_11, s0145_10,
                                  l23_13_sp1, l23_12_sp1, l23_11_sp1, l23_10_sp1),
                            acc4!(s0145_13b, s0145_12b, s0145_11b, s0145_10b,
                                  l23_13_sp2, l23_12_sp2, l23_11_sp2, l23_10_sp2),
                        );
                        let m11_1 = _mm256_add_epi16(
                            acc4!(s2367_13, s2367_12, s2367_11, s2367_10,
                                  l23_13_sp1, l23_12_sp1, l23_11_sp1, l23_10_sp1),
                            acc4!(s2367_13b, s2367_12b, s2367_11b, s2367_10b,
                                  l23_13_sp2, l23_12_sp2, l23_11_sp2, l23_10_sp2),
                        );
                        // scale, then straighten into 4 rows x 8 columns
                        let m00_0 = _mm256_madd_epi16(m00_0, scale_0145_0);
                        let m01_0 = _mm256_madd_epi16(m01_0, scale_2367_0);
                        let m10_0 = _mm256_madd_epi16(m10_0, scale_0145_0);
                        let m11_0 = _mm256_madd_epi16(m11_0, scale_2367_0);
                        let m00_1 = _mm256_madd_epi16(m00_1, scale_0145_1);
                        let m01_1 = _mm256_madd_epi16(m01_1, scale_2367_1);
                        let m10_1 = _mm256_madd_epi16(m10_1, scale_0145_1);
                        let m11_1 = _mm256_madd_epi16(m11_1, scale_2367_1);
                        let iacc_row_0_0 = _mm256_blend_epi32::<204>(
                            m00_0,
                            _mm256_shuffle_epi32::<78>(m01_0),
                        );
                        let iacc_row_1_0 = _mm256_blend_epi32::<204>(
                            _mm256_shuffle_epi32::<78>(m00_0),
                            m01_0,
                        );
                        let iacc_row_2_0 = _mm256_blend_epi32::<204>(
                            m10_0,
                            _mm256_shuffle_epi32::<78>(m11_0),
                        );
                        let iacc_row_3_0 = _mm256_blend_epi32::<204>(
                            _mm256_shuffle_epi32::<78>(m10_0),
                            m11_0,
                        );
                        let iacc_row_0_1 = _mm256_blend_epi32::<204>(
                            m00_1,
                            _mm256_shuffle_epi32::<78>(m01_1),
                        );
                        let iacc_row_1_1 = _mm256_blend_epi32::<204>(
                            _mm256_shuffle_epi32::<78>(m00_1),
                            m01_1,
                        );
                        let iacc_row_2_1 = _mm256_blend_epi32::<204>(
                            m10_1,
                            _mm256_shuffle_epi32::<78>(m11_1),
                        );
                        let iacc_row_3_1 = _mm256_blend_epi32::<204>(
                            _mm256_shuffle_epi32::<78>(m10_1),
                            m11_1,
                        );
                        let iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        let iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        let iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        let iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // row scales: `_mm_load_ps(a_ptr[b].d)` = rows 0..4,
                        // duplicated into both 128-bit lanes
                        // (arch/x86/repack.cpp:3460-3461)
                        let row_scale_sse = _mm_loadu_ps(a_blk as *const f32);
                        let row_scale_f32 = _mm256_set_m128(row_scale_sse, row_scale_sse);
                        acc_rows[0] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc_row_0),
                            _mm256_mul_ps(
                                col_scale_f32,
                                _mm256_shuffle_ps::<0>(row_scale_f32, row_scale_f32),
                            ),
                            acc_rows[0],
                        );
                        acc_rows[1] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc_row_1),
                            _mm256_mul_ps(
                                col_scale_f32,
                                _mm256_shuffle_ps::<85>(row_scale_f32, row_scale_f32),
                            ),
                            acc_rows[1],
                        );
                        acc_rows[2] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc_row_2),
                            _mm256_mul_ps(
                                col_scale_f32,
                                _mm256_shuffle_ps::<170>(row_scale_f32, row_scale_f32),
                            ),
                            acc_rows[2],
                        );
                        acc_rows[3] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc_row_3),
                            _mm256_mul_ps(
                                col_scale_f32,
                                _mm256_shuffle_ps::<255>(row_scale_f32, row_scale_f32),
                            ),
                            acc_rows[3],
                        );

                        // min term: the same 4 shuffle patterns over the
                        // 16-element bsums (arch/x86/repack.cpp:3469-3477)
                        let min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32::<0>(bsums_hsum), mins_01);
                        let min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32::<85>(bsums_hsum), mins_01);
                        let min_2 =
                            _mm256_madd_epi16(_mm256_shuffle_epi32::<170>(bsums_hsum), mins_01);
                        let min_3 =
                            _mm256_madd_epi16(_mm256_shuffle_epi32::<255>(bsums_hsum), mins_01);
                        acc_min_rows[0] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(min_0),
                            _mm256_mul_ps(
                                col_dmin_f32,
                                _mm256_shuffle_ps::<0>(row_scale_f32, row_scale_f32),
                            ),
                            acc_min_rows[0],
                        );
                        acc_min_rows[1] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(min_1),
                            _mm256_mul_ps(
                                col_dmin_f32,
                                _mm256_shuffle_ps::<85>(row_scale_f32, row_scale_f32),
                            ),
                            acc_min_rows[1],
                        );
                        acc_min_rows[2] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(min_2),
                            _mm256_mul_ps(
                                col_dmin_f32,
                                _mm256_shuffle_ps::<170>(row_scale_f32, row_scale_f32),
                            ),
                            acc_min_rows[2],
                        );
                        acc_min_rows[3] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(min_3),
                            _mm256_mul_ps(
                                col_dmin_f32,
                                _mm256_shuffle_ps::<255>(row_scale_f32, row_scale_f32),
                            ),
                            acc_min_rows[3],
                        );
                    }
                }
                for i in 0..4 {
                    let o = s.add((y * 4 + i) * bs + x * 8);
                    _mm256_storeu_ps(o, _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
                }
                super::note_q4k_gemm_call();
            }
        }
    }

    // -------------------------------------------------------------------
    // AVX512BW+DQ gemm body (arch/x86/repack.cpp:2077-2815)
    // -------------------------------------------------------------------

    /// `GGML_F32Cx8x2_LOAD(x, y)` (arch/x86/repack.cpp:30) — sixteen ggml_half
    /// values, eight from each of two `block_q4_Kx8` tiles, widened to f32
    /// with tile 0 in lanes 0..8 and tile 1 in lanes 8..16. The gemm's
    /// `col_scale_f32`/`col_dmin_f32` (repack.cpp:2111, :2114, :2474, :2477).
    #[inline]
    #[target_feature(enable = "avx2,f16c,avx512f")]
    unsafe fn f32x8x2_load(x: *const u8, y: *const u8) -> __m512 {
        _mm512_cvtph_ps(_mm256_set_m128i(
            _mm_loadu_si128(y as *const __m128i),
            _mm_loadu_si128(x as *const __m128i),
        ))
    }

    /// The right-hand (weight) lane network of the AVX512 gemm for one (super
    /// block, k-group `sb`): the eight 32-byte `qs` chunks of two
    /// `block_q4_Kx8` tiles (arch/x86/repack.cpp:2119-2135), dword-blended
    /// into the `0145`/`2367` (tile 0) and `89CD`/`ABEF` (tile 1) orders
    /// (:2137-2153), fused across the two tiles (:2155-2163), nibble-split
    /// (:2166-2184) and dword-shuffled into the sp1 (136) / sp2 (221)
    /// patterns (:2187-2222).
    ///
    /// Index layout: `cd` = the `014589CD` column set, `ef` = `2367ABEF`;
    /// `[0..4]` = chunks 0..3 low nibbles (`_00.._03`), `[4..8]` = chunks
    /// 0..3 high nibbles (`_10.._13`).
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512dq")]
    unsafe fn rhs_512(
        qs0: *const u8, // b_ptr_0[b].qs + sb*256
        qs1: *const u8, // b_ptr_1[b].qs + sb*256
        m4b: __m512i,
        required_order: __m256i,
    ) -> ([__m512i; 8], [__m512i; 8], [__m512i; 8], [__m512i; 8]) {
        // raw chunk loads (arch/x86/repack.cpp:2119-2135)
        let load = |p: *const u8| _mm256_loadu_si256(p as *const __m256i);
        let raw_0123 = [
            load(qs0),
            load(qs0.add(64)),
            load(qs0.add(128)),
            load(qs0.add(192)),
        ];
        let raw_4567 = [
            load(qs0.add(32)),
            load(qs0.add(96)),
            load(qs0.add(160)),
            load(qs0.add(224)),
        ];
        let raw_89ab = [
            load(qs1),
            load(qs1.add(64)),
            load(qs1.add(128)),
            load(qs1.add(192)),
        ];
        let raw_cdef = [
            load(qs1.add(32)),
            load(qs1.add(96)),
            load(qs1.add(160)),
            load(qs1.add(224)),
        ];
        // 0145/2367/89CD/ABEF dwords (arch/x86/repack.cpp:2137-2153)
        let mut raw_0145 = [_mm256_setzero_si256(); 4];
        let mut raw_2367 = [_mm256_setzero_si256(); 4];
        let mut raw_89cd = [_mm256_setzero_si256(); 4];
        let mut raw_abef = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            raw_0145[j] = _mm256_blend_epi32::<240>(
                raw_0123[j],
                _mm256_permutevar8x32_epi32(raw_4567[j], required_order),
            );
            raw_2367[j] = _mm256_blend_epi32::<240>(
                _mm256_permutevar8x32_epi32(raw_0123[j], required_order),
                raw_4567[j],
            );
            raw_89cd[j] = _mm256_blend_epi32::<240>(
                raw_89ab[j],
                _mm256_permutevar8x32_epi32(raw_cdef[j], required_order),
            );
            raw_abef[j] = _mm256_blend_epi32::<240>(
                _mm256_permutevar8x32_epi32(raw_89ab[j], required_order),
                raw_cdef[j],
            );
        }
        // fuse the two tiles into 512 bits (arch/x86/repack.cpp:2155-2163)
        let mut raw_cd = [_mm512_setzero_si512(); 4];
        let mut raw_ef = [_mm512_setzero_si512(); 4];
        for j in 0..4 {
            raw_cd[j] = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_0145[j]), raw_89cd[j]);
            raw_ef[j] = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_2367[j]), raw_abef[j]);
        }
        // 4-bit -> 8-bit + the two dword shuffle patterns
        // (arch/x86/repack.cpp:2166-2222)
        let mut cd_sp1 = [_mm512_setzero_si512(); 8];
        let mut cd_sp2 = [_mm512_setzero_si512(); 8];
        let mut ef_sp1 = [_mm512_setzero_si512(); 8];
        let mut ef_sp2 = [_mm512_setzero_si512(); 8];
        for j in 0..4 {
            let lo_cd = _mm512_and_si512(raw_cd[j], m4b);
            let hi_cd = _mm512_and_si512(_mm512_srli_epi16::<4>(raw_cd[j]), m4b);
            let lo_ef = _mm512_and_si512(raw_ef[j], m4b);
            let hi_ef = _mm512_and_si512(_mm512_srli_epi16::<4>(raw_ef[j]), m4b);
            cd_sp1[j] = _mm512_shuffle_epi32::<136>(lo_cd);
            cd_sp2[j] = _mm512_shuffle_epi32::<221>(lo_cd);
            cd_sp1[j + 4] = _mm512_shuffle_epi32::<136>(hi_cd);
            cd_sp2[j + 4] = _mm512_shuffle_epi32::<221>(hi_cd);
            ef_sp1[j] = _mm512_shuffle_epi32::<136>(lo_ef);
            ef_sp2[j] = _mm512_shuffle_epi32::<221>(lo_ef);
            ef_sp1[j + 4] = _mm512_shuffle_epi32::<136>(hi_ef);
            ef_sp2[j + 4] = _mm512_shuffle_epi32::<221>(hi_ef);
        }
        (cd_sp1, cd_sp2, ef_sp1, ef_sp2)
    }

    /// The 6-bit scale/min network of the AVX512 gemm for one (super block,
    /// k-group): the four 12-byte `utmp` decodes of the two tiles' scale
    /// groups for sub-blocks `2*sb`/`2*sb+1` (arch/x86/repack.cpp:2224-2256),
    /// the 16-lane `scales`/`mins` vectors (:2258-2267) and the four
    /// column-scale dword-broadcasts (:2269-2273). Returns
    /// `(scale_014589CD_0, scale_2367ABEF_0, scale_014589CD_1,
    /// scale_2367ABEF_1, mins_01)`.
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    unsafe fn scales_512(
        sc0: *const u8, // b_ptr_0[b].scales (tile base + 32)
        sc1: *const u8,
        sb: usize,
        kmask1: u32,
        kmask2: u32,
        kmask3: u32,
    ) -> (__m512i, __m512i, __m512i, __m512i, __m512i) {
        let mut utmp_00 = [0u32; 4];
        let mut utmp_01 = [0u32; 4];
        let mut utmp_10 = [0u32; 4];
        let mut utmp_11 = [0u32; 4];
        unpack_utmp(sc0.add(24 * sb), &mut utmp_00, kmask1, kmask2, kmask3);
        unpack_utmp(sc0.add(12 + sb * 24), &mut utmp_01, kmask1, kmask2, kmask3);
        unpack_utmp(sc1.add(sb * 24), &mut utmp_10, kmask1, kmask2, kmask3);
        unpack_utmp(sc1.add(12 + sb * 24), &mut utmp_11, kmask1, kmask2, kmask3);
        // mins_and_scales_0/1 (arch/x86/repack.cpp:2259, :2263): tile 0's two
        // sub blocks in the low 128 bits, tile 1's in the high
        let ms0 = _mm256_set_epi32(
            utmp_10[3] as i32,
            utmp_10[2] as i32,
            utmp_10[1] as i32,
            utmp_10[0] as i32,
            utmp_00[3] as i32,
            utmp_00[2] as i32,
            utmp_00[1] as i32,
            utmp_00[0] as i32,
        );
        let ms1 = _mm256_set_epi32(
            utmp_11[3] as i32,
            utmp_11[2] as i32,
            utmp_11[1] as i32,
            utmp_11[0] as i32,
            utmp_01[3] as i32,
            utmp_01[2] as i32,
            utmp_01[1] as i32,
            utmp_01[0] as i32,
        );
        // scales/mins (arch/x86/repack.cpp:2260, :2264, :2267)
        let scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(ms0, ms0));
        let scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(ms1, ms1));
        let mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(
            _mm256_shuffle_epi32::<78>(ms0),
            _mm256_shuffle_epi32::<78>(ms1),
        ));
        // (arch/x86/repack.cpp:2269-2273)
        (
            _mm512_shuffle_epi32::<68>(scales_0),
            _mm512_shuffle_epi32::<238>(scales_0),
            _mm512_shuffle_epi32::<68>(scales_1),
            _mm512_shuffle_epi32::<238>(scales_1),
            mins_01,
        )
    }

    /// The left-hand (activation) lane network of the AVX512 gemm for one
    /// `block_q8_Kx4` k-group `sb`: the eight 32-byte `qs` chunks
    /// (arch/x86/repack.cpp:2279-2302), each 128-bit half broadcast to 256
    /// then 512 bits (`_01` = activation rows 0,1; `_23` = rows 2,3,
    /// :2304-2320), dword-shuffled into the sp1 (160) / sp2 (245) patterns
    /// (:2329-2363). Index layout mirrors [`rhs_512`]: `[0..4]` = chunks
    /// 0..3 (`_00.._03`), `[4..8]` = chunks 4..7 (`_10.._13`).
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512dq")]
    unsafe fn lhs_512(qs: *const u8) -> ([__m512i; 8], [__m512i; 8], [__m512i; 8], [__m512i; 8]) {
        let mut l01 = [_mm512_setzero_si512(); 8];
        let mut l23 = [_mm512_setzero_si512(); 8];
        for j in 0..8 {
            let v = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
            let y01 = _mm256_permute2f128_si256::<0>(v, v);
            let y23 = _mm256_permute2f128_si256::<17>(v, v);
            l01[j] = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y01), y01);
            l23[j] = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y23), y23);
        }
        let mut l01_sp1 = [_mm512_setzero_si512(); 8];
        let mut l01_sp2 = [_mm512_setzero_si512(); 8];
        let mut l23_sp1 = [_mm512_setzero_si512(); 8];
        let mut l23_sp2 = [_mm512_setzero_si512(); 8];
        for j in 0..8 {
            l01_sp1[j] = _mm512_shuffle_epi32::<160>(l01[j]);
            l01_sp2[j] = _mm512_shuffle_epi32::<245>(l01[j]);
            l23_sp1[j] = _mm512_shuffle_epi32::<160>(l23[j]);
            l23_sp2[j] = _mm512_shuffle_epi32::<245>(l23[j]);
        }
        (l01_sp1, l01_sp2, l23_sp1, l23_sp2)
    }

    /// `lhs_bsums_hsum_0123_01` (arch/x86/repack.cpp:2323-2326, :2686-2689):
    /// the 16 i16 bsums of one `block_q8_Kx4` k-group (`bsums + 16*sb` in
    /// int16 units), pairwise hadd-ed so each dword holds
    /// `(bsums[4r+t0], bsums[4r+t1])` pairs per row r, duplicated to 512
    /// bits. (NB the C's pointer arithmetic is in `int16_t` units: 16*sb
    /// elements = 32*sb bytes.)
    #[inline]
    #[target_feature(enable = "avx2,avx512dq")]
    unsafe fn bsums_hsum_512(p: *const u8) -> __m512i {
        let bsums = _mm256_loadu_si256(p as *const __m256i);
        let hsum = _mm256_castsi128_si256(_mm_hadd_epi16(
            _mm256_castsi256_si128(bsums),
            _mm256_extracti128_si256::<1>(bsums),
        ));
        let hsum = _mm256_permute2x128_si256::<0>(hsum, hsum);
        _mm512_inserti32x8::<1>(_mm512_castsi256_si512(hsum), hsum)
    }

    /// The AVX512 gemm's per-(row-pair set, k-group) epilogue
    /// (arch/x86/repack.cpp:2366-2439): the eight int16 `iacc_mat` chains,
    /// the sp1+sp2 sums, the scale madds, the `0xCCCC` straightening into
    /// row vectors, the f32 row scales and the eight fmadds into
    /// `acc_rows[base..base+4]` / `acc_min_rows[base..base+4]`.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq")]
    unsafe fn acc_rows_512(
        a_tile: *const u8,
        bsums_hsum: __m512i,
        l01_sp1: &[__m512i; 8],
        l01_sp2: &[__m512i; 8],
        l23_sp1: &[__m512i; 8],
        l23_sp2: &[__m512i; 8],
        cd_sp1: &[__m512i; 8],
        cd_sp2: &[__m512i; 8],
        ef_sp1: &[__m512i; 8],
        ef_sp2: &[__m512i; 8],
        scale_cd_0: __m512i,
        scale_ef_0: __m512i,
        scale_cd_1: __m512i,
        scale_ef_1: __m512i,
        mins_01: __m512i,
        col_scale_f32: __m512,
        col_dmin_f32: __m512,
        acc_rows: &mut [__m512],
        acc_min_rows: &mut [__m512],
        base: usize,
    ) {
        // 4 maddubs summed in int16, innermost chunk first
        // (arch/x86/repack.cpp:2366-2382; products <= 15*127, the 4-term
        // lane sums stay < 2^15 — exact for any grouping)
        macro_rules! iacc16 {
            ($r:expr, $l:expr) => {{
                let a = _mm512_maddubs_epi16($r[3], $l[3]);
                let a = _mm512_add_epi16(a, _mm512_maddubs_epi16($r[2], $l[2]));
                let a = _mm512_add_epi16(a, _mm512_maddubs_epi16($r[1], $l[1]));
                _mm512_add_epi16(a, _mm512_maddubs_epi16($r[0], $l[0]))
            }};
        }
        // shuffle pattern one (arch/x86/repack.cpp:2366-2373)
        let i00_0_sp1 = iacc16!(&cd_sp1[0..4], &l01_sp1[0..4]);
        let i01_0_sp1 = iacc16!(&ef_sp1[0..4], &l01_sp1[0..4]);
        let i10_0_sp1 = iacc16!(&cd_sp1[0..4], &l23_sp1[0..4]);
        let i11_0_sp1 = iacc16!(&ef_sp1[0..4], &l23_sp1[0..4]);
        let i00_1_sp1 = iacc16!(&cd_sp1[4..8], &l01_sp1[4..8]);
        let i01_1_sp1 = iacc16!(&ef_sp1[4..8], &l01_sp1[4..8]);
        let i10_1_sp1 = iacc16!(&cd_sp1[4..8], &l23_sp1[4..8]);
        let i11_1_sp1 = iacc16!(&ef_sp1[4..8], &l23_sp1[4..8]);
        // shuffle pattern two (arch/x86/repack.cpp:2375-2382)
        let i00_0_sp2 = iacc16!(&cd_sp2[0..4], &l01_sp2[0..4]);
        let i01_0_sp2 = iacc16!(&ef_sp2[0..4], &l01_sp2[0..4]);
        let i10_0_sp2 = iacc16!(&cd_sp2[0..4], &l23_sp2[0..4]);
        let i11_0_sp2 = iacc16!(&ef_sp2[0..4], &l23_sp2[0..4]);
        let i00_1_sp2 = iacc16!(&cd_sp2[4..8], &l01_sp2[4..8]);
        let i01_1_sp2 = iacc16!(&ef_sp2[4..8], &l01_sp2[4..8]);
        let i10_1_sp2 = iacc16!(&cd_sp2[4..8], &l23_sp2[4..8]);
        let i11_1_sp2 = iacc16!(&ef_sp2[4..8], &l23_sp2[4..8]);
        // sp1 + sp2 (arch/x86/repack.cpp:2385-2393)
        let i00_0 = _mm512_add_epi16(i00_0_sp1, i00_0_sp2);
        let i01_0 = _mm512_add_epi16(i01_0_sp1, i01_0_sp2);
        let i10_0 = _mm512_add_epi16(i10_0_sp1, i10_0_sp2);
        let i11_0 = _mm512_add_epi16(i11_0_sp1, i11_0_sp2);
        let i00_1 = _mm512_add_epi16(i00_1_sp1, i00_1_sp2);
        let i01_1 = _mm512_add_epi16(i01_1_sp1, i01_1_sp2);
        let i10_1 = _mm512_add_epi16(i10_1_sp1, i10_1_sp2);
        let i11_1 = _mm512_add_epi16(i11_1_sp1, i11_1_sp2);
        // scale madd (arch/x86/repack.cpp:2395-2403)
        let i00_0 = _mm512_madd_epi16(i00_0, scale_cd_0);
        let i01_0 = _mm512_madd_epi16(i01_0, scale_ef_0);
        let i10_0 = _mm512_madd_epi16(i10_0, scale_cd_0);
        let i11_0 = _mm512_madd_epi16(i11_0, scale_ef_0);
        let i00_1 = _mm512_madd_epi16(i00_1, scale_cd_1);
        let i01_1 = _mm512_madd_epi16(i01_1, scale_ef_1);
        let i10_1 = _mm512_madd_epi16(i10_1, scale_cd_1);
        let i11_1 = _mm512_madd_epi16(i11_1, scale_ef_1);
        // straighten out to 4 row vectors (arch/x86/repack.cpp:2406-2413)
        let row_0_0 = _mm512_mask_blend_epi32(0xCCCC, i00_0, _mm512_shuffle_epi32::<78>(i01_0));
        let row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i00_0), i01_0);
        let row_2_0 = _mm512_mask_blend_epi32(0xCCCC, i10_0, _mm512_shuffle_epi32::<78>(i11_0));
        let row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i10_0), i11_0);
        let row_0_1 = _mm512_mask_blend_epi32(0xCCCC, i00_1, _mm512_shuffle_epi32::<78>(i01_1));
        let row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i00_1), i01_1);
        let row_2_1 = _mm512_mask_blend_epi32(0xCCCC, i10_1, _mm512_shuffle_epi32::<78>(i11_1));
        let row_3_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i10_1), i11_1);
        // (arch/x86/repack.cpp:2415-2418)
        let row_0 = _mm512_add_epi32(row_0_0, row_0_1);
        let row_1 = _mm512_add_epi32(row_1_0, row_1_1);
        let row_2 = _mm512_add_epi32(row_2_0, row_2_1);
        let row_3 = _mm512_add_epi32(row_3_0, row_3_1);
        // the 4 Q8_K row scales repeated across lanes
        // (arch/x86/repack.cpp:2421-2423)
        let sse = _mm_loadu_ps(a_tile as *const f32);
        let ymm = _mm256_set_m128(sse, sse);
        let rs = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(ymm), ymm);
        // (arch/x86/repack.cpp:2426-2429)
        acc_rows[base] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row_0),
            _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps::<0>(rs, rs)),
            acc_rows[base],
        );
        acc_rows[base + 1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row_1),
            _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps::<85>(rs, rs)),
            acc_rows[base + 1],
        );
        acc_rows[base + 2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row_2),
            _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps::<170>(rs, rs)),
            acc_rows[base + 2],
        );
        acc_rows[base + 3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row_3),
            _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps::<255>(rs, rs)),
            acc_rows[base + 3],
        );
        // min term (arch/x86/repack.cpp:2431-2439)
        let min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32::<0>(bsums_hsum), mins_01);
        let min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32::<85>(bsums_hsum), mins_01);
        let min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32::<170>(bsums_hsum), mins_01);
        let min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32::<255>(bsums_hsum), mins_01);
        acc_min_rows[base] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(min_0),
            _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps::<0>(rs, rs)),
            acc_min_rows[base],
        );
        acc_min_rows[base + 1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(min_1),
            _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps::<85>(rs, rs)),
            acc_min_rows[base + 1],
        );
        acc_min_rows[base + 2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(min_2),
            _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps::<170>(rs, rs)),
            acc_min_rows[base + 2],
        );
        acc_min_rows[base + 3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(min_3),
            _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps::<255>(rs, rs)),
            acc_min_rows[base + 3],
        );
    }

    /// The `__AVX512BW__ && __AVX512DQ__` section of `ggml_gemm_q4_K_8x8_q8_K`
    /// (arch/x86/repack.cpp:2077-2815) — the body the reference's
    /// `-march=native` build runs on this host: 16 activation rows (four
    /// `block_q8_Kx4` tiles) x 16 weight columns (two `block_q4_Kx8` tiles)
    /// per pass (:2086-2448), then the 4-row tail over the same 16-column
    /// pairs (:2450-2810), handing the `nc % 16 == 8` tail columns to the
    /// AVX2 section via `xstart = anc/8` (:2811-2814). Unlike the Q4_0 gemm
    /// there is no VNNI branch in this body (its dot op is
    /// `maddubs`+`madd`, never `dpbusd`), so no const-generic variant; the
    /// runtime AVX512BW+DQ check lives in the [`gemm`] dispatcher, outside
    /// every loop.
    ///
    /// # Safety
    /// as [`gemm_256_section`].
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq")]
    pub unsafe fn gemm_avx512(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
    ) {
        let nb = n / QK_K;
        let b_nb = n / QK_K;
        let kmask1: u32 = 0x3f3f_3f3f;
        let kmask2: u32 = 0x0f0f_0f0f;
        let kmask3: u32 = 0x0303_0303;
        // m4bexpanded (arch/x86/repack.cpp:2084)
        let m4b = _mm512_set1_epi8(0x0F);
        // requiredOrder (arch/x86/repack.cpp:2072)
        let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
        // anr/anc align nr/nc to 16 (arch/x86/repack.cpp:2074, :2079)
        let anr = nr - nr % 16;
        let anc = nc - nc % 16;
        let mut y = 0usize;

        // 16 activation rows x 16 columns (arch/x86/repack.cpp:2086-2448)
        while y < anr / 4 {
            // a_ptrs[0..4] (arch/x86/repack.cpp:2088-2093)
            for x in (0..anc / 8).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_Q4_KX8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_Q4_KX8_SIZE);
                // master FP accumulators (arch/x86/repack.cpp:2097-2106)
                let mut acc_rows = [_mm512_setzero_ps(); 16];
                let mut acc_min_rows = [_mm512_setzero_ps(); 16];
                for b in 0..nb {
                    let t0 = b0.add(b * BLOCK_Q4_KX8_SIZE);
                    let t1 = b1.add(b * BLOCK_Q4_KX8_SIZE);
                    // col_scale / col_dmin (arch/x86/repack.cpp:2111, :2114)
                    let col_scale_f32 = f32x8x2_load(t0, t1);
                    let col_dmin_f32 = f32x8x2_load(t0.add(16), t1.add(16));
                    for sb in 0..QK_K / 64 {
                        let (cd_sp1, cd_sp2, ef_sp1, ef_sp2) = rhs_512(
                            t0.add(128 + 256 * sb),
                            t1.add(128 + 256 * sb),
                            m4b,
                            required_order,
                        );
                        let (scd0, sef0, scd1, sef1, mins_01) =
                            scales_512(t0.add(32), t1.add(32), sb, kmask1, kmask2, kmask3);
                        for rp in 0..4 {
                            // a_ptrs[rp] (arch/x86/repack.cpp:2090-2093)
                            let a_tile = vy.add(((y + rp) * nb + b) * BLOCK_Q8_KX4_SIZE);
                            let (l01_sp1, l01_sp2, l23_sp1, l23_sp2) =
                                lhs_512(a_tile.add(16 + 256 * sb));
                            let bsums_hsum = bsums_hsum_512(a_tile.add(1040 + 32 * sb));
                            acc_rows_512(
                                a_tile,
                                bsums_hsum,
                                &l01_sp1,
                                &l01_sp2,
                                &l23_sp1,
                                &l23_sp2,
                                &cd_sp1,
                                &cd_sp2,
                                &ef_sp1,
                                &ef_sp2,
                                scd0,
                                sef0,
                                scd1,
                                sef1,
                                mins_01,
                                col_scale_f32,
                                col_dmin_f32,
                                &mut acc_rows,
                                &mut acc_min_rows,
                                rp * 4,
                            );
                        }
                    }
                }
                // store (arch/x86/repack.cpp:2443-2446)
                for i in 0..16 {
                    _mm512_storeu_ps(
                        s.add((y * 4 + i) * bs + x * 8),
                        _mm512_sub_ps(acc_rows[i], acc_min_rows[i]),
                    );
                }
                super::note_q4k_gemm_call();
            }
            y += 4;
        }

        // 4-row tail x 16 columns (arch/x86/repack.cpp:2450-2810)
        while y < nr / 4 {
            for x in (0..anc / 8).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_Q4_KX8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_Q4_KX8_SIZE);
                let mut acc_rows = [_mm512_setzero_ps(); 4];
                let mut acc_min_rows = [_mm512_setzero_ps(); 4];
                for b in 0..nb {
                    let t0 = b0.add(b * BLOCK_Q4_KX8_SIZE);
                    let t1 = b1.add(b * BLOCK_Q4_KX8_SIZE);
                    // (arch/x86/repack.cpp:2474, :2477)
                    let col_scale_f32 = f32x8x2_load(t0, t1);
                    let col_dmin_f32 = f32x8x2_load(t0.add(16), t1.add(16));
                    for sb in 0..QK_K / 64 {
                        let (cd_sp1, cd_sp2, ef_sp1, ef_sp2) = rhs_512(
                            t0.add(128 + 256 * sb),
                            t1.add(128 + 256 * sb),
                            m4b,
                            required_order,
                        );
                        let (scd0, sef0, scd1, sef1, mins_01) =
                            scales_512(t0.add(32), t1.add(32), sb, kmask1, kmask2, kmask3);
                        let a_tile = vy.add((y * nb + b) * BLOCK_Q8_KX4_SIZE);
                        let (l01_sp1, l01_sp2, l23_sp1, l23_sp2) = lhs_512(a_tile.add(16 + 256 * sb));
                        let bsums_hsum = bsums_hsum_512(a_tile.add(1040 + 32 * sb));
                        acc_rows_512(
                            a_tile,
                            bsums_hsum,
                            &l01_sp1,
                            &l01_sp2,
                            &l23_sp1,
                            &l23_sp2,
                            &cd_sp1,
                            &cd_sp2,
                            &ef_sp1,
                            &ef_sp2,
                            scd0,
                            sef0,
                            scd1,
                            sef1,
                            mins_01,
                            col_scale_f32,
                            col_dmin_f32,
                            &mut acc_rows,
                            &mut acc_min_rows,
                            0,
                        );
                    }
                }
                // store (arch/x86/repack.cpp:2805-2808)
                for i in 0..4 {
                    _mm512_storeu_ps(
                        s.add((y * 4 + i) * bs + x * 8),
                        _mm512_sub_ps(acc_rows[i], acc_min_rows[i]),
                    );
                }
                super::note_q4k_gemm_call();
            }
            y += 1;
        }

        // the `nc % 16` tail columns go through the AVX2 section
        // (arch/x86/repack.cpp:2811-2814: xstart = anc/8, y = 0)
        if anc != nc {
            gemm_256_section(n, s, bs, vx, vy, nr, nc, anc / 8);
        }
    }

    /// `ggml_gemm_q4_K_8x8_q8_K`'s AVX2-or-better body (arch/x86/
    /// repack.cpp:2065): on an AVX512BW+DQ host the 512-bit section
    /// (`gemm_avx512`, the `-march=native` reference build's compiled form)
    /// with the AVX2 section serving the `nc % 16` tail columns; otherwise
    /// the AVX2 section over the whole matrix — the C's `#else` build. The
    /// runtime check runs once per call, outside every loop.
    ///
    /// # Safety
    /// `vx` must address `nc * nb * 1152` repacked bytes, `vy` `nr/4 * nb`
    /// `block_q8_Kx4` tiles, and `s` at least `nr * bs` floats.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
    ) {
        if crate::simd_x86::avx512bw() {
            gemm_avx512(n, s, bs, vx, vy, nr, nc);
            return;
        }
        gemm_256_section(n, s, bs, vx, vy, nr, nc, 0);
    }
}

/// `ggml_gemv_q4_K_8x8_q8_K` (repack.cpp:4379 -> arch/x86/repack.cpp:1464).
///
/// `nc` is the number of weight rows of the 8-row group matrix (a multiple of
/// 8); the kernel writes `nc` floats (8 columns per call, in the C's
/// `finalpermutemask` order). AVX2 hosts take the transcribed C body, others
/// the scalar body with identical arithmetic; `LLAMA_RUST_REPACK_SIMD=0`
/// forces the scalar body.
pub fn gemv_q4_K_8x8_q8_K(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    assert_eq!(n % QK_K, 0, "gemv_q4_K: n % QK_K != 0");
    assert_eq!(nc % 8, 0, "gemv_q4_K: nc % 8 != 0");
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() && simd_enabled() {
        // SAFETY: the caller passes the repacked 8x8 matrix (`nc/8 * nb *
        // 1152` bytes), one `block_q8_K` row (`nb * 292` bytes) and `nc`
        // output floats; `nc % 8 == 0` and `n % QK_K == 0` are asserted above.
        unsafe {
            simd_x86_q4k::gemv(n, s.as_mut_ptr(), vx.as_ptr(), vy.as_ptr(), nc);
        }
        return;
    }
    gemv_q4_K_8x8_q8_K_scalar(n, s, vx, vy, nc);
}

/// `ggml_gemm_q4_K_8x8_q8_K` (repack.cpp:4484 -> arch/x86/repack.cpp:2042).
///
/// `nr` activation rows must be a multiple of 4 (the C asserts it); the
/// activation rows come from `block_q8_Kx4` tiles (`ggml_quantize_mat_q8_K_4x8`).
/// AVX2 hosts take the transcribed C body — on AVX512BW+DQ hosts the 512-bit
/// section (`simd_x86_q4k::gemm_avx512`, arch/x86/repack.cpp:2077-2815) with
/// the AVX2 body serving the `nc % 16` tail columns — others the scalar body
/// with identical arithmetic; `LLAMA_RUST_REPACK_SIMD=0` forces the scalar
/// body.
pub fn gemm_q4_K_8x8_q8_K(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[u8],
    vy: &[u8],
    nr: usize,
    nc: usize,
) {
    assert_eq!(n % QK_K, 0, "gemm_q4_K: n % QK_K != 0");
    assert_eq!(nr % 4, 0, "gemm_q4_K: nr % 4 != 0");
    assert_eq!(nc % 8, 0, "gemm_q4_K: nc % 8 != 0");
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() && simd_enabled() {
        // SAFETY: as in `gemv_q4_K_8x8_q8_K`; `vy` holds `nr/4 * nb` tiles of
        // 1168 bytes and `s` at least `nr * bs` floats.
        unsafe {
            simd_x86_q4k::gemm(n, s.as_mut_ptr(), bs, vx.as_ptr(), vy.as_ptr(), nr, nc);
        }
        return;
    }
    gemm_q4_K_8x8_q8_K_scalar(n, s, bs, vx, vy, nr, nc);
}

// ===========================================================================
// Q4_0 8x8 — the reference's x86 production path for Q4_0
// ===========================================================================
//
// Reference: `ggml_repack_get_optimal_repack_type` (repack.cpp:4987-4993)
//
// ```text
//   if (ggml_cpu_has_avx2()) { if (cur->ne[1] % 8 == 0) return &q4_0_8x8_q8_0; }
// ```
//
// (the other Q4_0 gates need NEON or RISC-V; on x86 the AVX2 branch is the
// only one, so every 2D Q4_0 tensor whose `ne[1]` is a multiple of 8 gets a
// CPU_REPACK trait — and, unlike the pre-repack port's routing, that tensor's
// mul_mats then never reach llamafile tinyBLAS: `ggml_compute_forward` runs
// `ggml_cpu_extra_compute_forward` (ggml-cpu.c:1751-1753) *before* the op
// switch, and the repack traits handle GGML_OP_MUL_MAT unconditionally).
//
//   * `set_tensor` -> `repack_q4_0_to_q4_0_8_bl` (repack.cpp:3790-3822) ->
//     `make_block_q4_0x8` (repack.cpp:3128-3152): 8 fp16 deltas first, then
//     the nibble bytes interleaved 8-at-a-time with `^ 0x8888..88` (the
//     offset-binary -> signed-nibble flip, folded into the layout).
//   * `forward_mul_mat_one_chunk` (repack.cpp:4638-4647) splits the activation
//     rows exactly like Q4_K: `nrows > 3` -> gemm over `nrows - nrows % 4`
//     rows, then one gemv per tail row; the wdata pass (repack.cpp:4697-4705)
//     quantizes 4-row groups with `ggml_quantize_mat_t<8, Q8_0>` =
//     `ggml_quantize_mat_q8_0_4x8` and tail rows with the plain
//     `quantize_row_q8_0`.
//   * kernels: `ggml_gemv_q4_0_8x8_q8_0` (arch/x86/repack.cpp:1448) and
//     `ggml_gemm_q4_0_8x8_q8_0` (arch/x86/repack.cpp:2022), both thin wrappers
//     over the *shared* `gemv|gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`
//     templates (:522/:641) with the sign-extend LUT
//     `set_epi8(-1,..,-8, 7,6,..,0)` and the fp16 `col_scale_f32` loaded
//     through `GGML_F32Cx8_REARRANGE_LOAD` in the (B0,B4,B1,B5,B2,B6,B3,B7)
//     lane order (arch/x86/repack.cpp:581-583).
//
// ## Arithmetic (what the port must reproduce)
//
// For one output element (weight row `c`, activation row `r`) and one 32-element
// k-block `b`, every body of both kernels computes
//
// ```text
//   iacc = Σ_{k<32} LUT[nibble(w[c,k,b])] * q8[k]        // exact int32
//   acc  = fma(f32(iacc), f32(d_col[c]) * f32(d_row), acc) // ONE fma per b
// ```
//
// with `b` ascending — the same two properties as the MXFP4 kernels (integer
// dot exact: |LUT| ≤ 8, |q8| ≤ 127, so |iacc| ≤ 32512 < 2^24 makes
// `cvtepi32_ps` exact as well; one rounded `d_col*d_row` product then one fused
// multiply-add per block, lane permutations only assign lanes). The
// `*_generic` kernels (repack.cpp:843/1895) accumulate `sumf += d*d_row*
// (f32)sumi` per block — a *non-fused* chain — and are not what the reference
// build runs; `parity/ref_repack_q4_0dump.cpp`'s own report shows them
// disagreeing with the AVX kernels.

use crate::blocks::QK4_0;

/// `sizeof(block_q4_0)` (ggml-common.h)
pub const BLOCK_Q4_0_SIZE: usize = 2 + QK4_0 / 2; // 18
/// `sizeof(block_q4_0x8)` (repack.h) — one 8-row x 1-block tile: 8 fp16 deltas
/// then `QK4_0 * 4` interleaved nibble bytes. Same byte size as 8 plain
/// blocks, so the repacked tensor keeps the plan row stride.
pub const BLOCK_Q4_0X8_SIZE: usize = 8 * 2 + QK4_0 * 4; // 144
/// `nrows_interleaved` of `repack_q4_0_to_q4_0_8_bl` (repack.cpp:3793)
const NROWS_INTERLEAVED_Q4_0: usize = 8;

/// `ggml_repack_get_optimal_repack_type` (repack.cpp:4987-4993) for Q4_0 on
/// x86_64: repack iff `ggml_cpu_has_avx2() && ne[1] % 8 == 0`.
/// `repack_q4_0_to_q4_0_8_bl` additionally requires `ne[0] % 8 == 0`
/// (repack.cpp:3804), which `ne[0] % QK4_0 == 0` implies.
pub fn repack_supported_q4_0(ne1: i64, ne0: i64) -> bool {
    ne0 % QK4_0 as i64 == 0 && ne1 % NROWS_INTERLEAVED_Q4_0 as i64 == 0
}

/// `repack.cpp:3128 make_block_q4_0x8` — interleave eight `block_q4_0`s (rows
/// `in[0..8]` of the same k-block) into one 144-byte `block_q4_0x8`.
///
/// Layout: `d[8]` = the 8 per-row fp16 deltas, then `qs[128]` where
/// `qs[dst..dst+8] = in[i % 8].qs[src_offset..src_offset+8] ^ 0x88...88` with
/// `dst = i * 8`, `src_offset = (i / 8) * 8` (`blck_size_interleave == 8`,
/// `end = QK4_0 * 4 / 8 = 16`): rows 0..7 of nibble bytes 0..7 first, then
/// rows 0..7 of nibble bytes 8..15 — the interleave is over rows.
pub fn make_block_q4_0x8(in8: &[u8], out: &mut [u8]) {
    debug_assert_eq!(in8.len(), 8 * BLOCK_Q4_0_SIZE);
    debug_assert_eq!(out.len(), BLOCK_Q4_0X8_SIZE);
    let (out_d, out_qs) = out.split_at_mut(16);
    for i in 0..8 {
        out_d[2 * i..2 * i + 2].copy_from_slice(&in8[i * BLOCK_Q4_0_SIZE..i * BLOCK_Q4_0_SIZE + 2]);
    }
    // C: for (i = 0; i < end; ++i) { src_id = i % 8; src_offset = (i / 8) * 8;
    //     dst_offset = i * 8; memcpy 8 bytes; elems ^= 0x8888888888888888; }
    for i in 0..(QK4_0 * 4 / 8) {
        let src_id = i % 8;
        let src_offset = (i / 8) * 8;
        let dst_offset = i * 8;
        let src = &in8[src_id * BLOCK_Q4_0_SIZE + 2 + src_offset..src_id * BLOCK_Q4_0_SIZE + 2 + src_offset + 8];
        for (o, &b) in out_qs[dst_offset..dst_offset + 8].iter_mut().zip(src) {
            *o = b ^ 0x88;
        }
    }
}

/// `repack.cpp:3790 repack_q4_0_to_q4_0_8_bl` — whole-tensor repack. Rows are
/// grouped in 8s; within a group the tiles are written in k-block order. The
/// output is byte-identical in size (`block_q4_0x8` == 8 * `block_q4_0`), so
/// the repacked copy keeps the plan row stride — kernels address 8-row groups
/// as `src0 + row/8 * nb * 144`.
pub fn repack_q4_0_8x8_into(src: &[u8], nrows: usize, n_per_row: usize, dst: &mut [u8]) {
    let nb = n_per_row / QK4_0;
    let row = nb * BLOCK_Q4_0_SIZE;
    assert_eq!(n_per_row % QK4_0, 0, "repack_q4_0: ne[0] % 32 != 0");
    assert_eq!(nrows % NROWS_INTERLEAVED_Q4_0, 0, "repack_q4_0: ne[1] % 8 != 0");
    assert!(src.len() >= nrows * row, "repack_q4_0: src too small");
    assert!(dst.len() >= nrows * row, "repack_q4_0: dst too small");
    let group = NROWS_INTERLEAVED_Q4_0 * row;
    for g in 0..nrows / NROWS_INTERLEAVED_Q4_0 {
        let sbase = g * group;
        for x in 0..nb {
            let d = sbase + x * BLOCK_Q4_0X8_SIZE;
            let mut tmp = [0u8; 8 * BLOCK_Q4_0_SIZE];
            for i in 0..NROWS_INTERLEAVED_Q4_0 {
                // dst_tmp[i] = src[x + i*nblocks]  (row i of this group, block x)
                let so = sbase + i * row + x * BLOCK_Q4_0_SIZE;
                tmp[i * BLOCK_Q4_0_SIZE..(i + 1) * BLOCK_Q4_0_SIZE]
                    .copy_from_slice(&src[so..so + BLOCK_Q4_0_SIZE]);
            }
            make_block_q4_0x8(&tmp, &mut dst[d..d + BLOCK_Q4_0X8_SIZE]);
        }
    }
}

/// [`repack_q4_0_8x8_into`] into a fresh buffer.
pub fn repack_q4_0_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let len = nrows * (n_per_row / QK4_0) * BLOCK_Q4_0_SIZE;
    let mut out = vec![0u8; len];
    repack_q4_0_8x8_into(&src[..len], nrows, n_per_row, &mut out);
    out
}

/// Inverse of [`repack_q4_0_8x8`] (round-trip test / debugging only; the
/// reference has no such function).
pub fn unrepack_q4_0_8x8(src: &[u8], nrows: usize, n_per_row: usize) -> Vec<u8> {
    let nb = n_per_row / QK4_0;
    let row = nb * BLOCK_Q4_0_SIZE;
    assert_eq!(nrows % NROWS_INTERLEAVED_Q4_0, 0);
    let mut out = vec![0u8; nrows * row];
    for g in 0..nrows / NROWS_INTERLEAVED_Q4_0 {
        let base = g * NROWS_INTERLEAVED_Q4_0 * row;
        for x in 0..nb {
            let tile = &src[base + x * BLOCK_Q4_0X8_SIZE..base + (x + 1) * BLOCK_Q4_0X8_SIZE];
            for i in 0..NROWS_INTERLEAVED_Q4_0 {
                let so = base + i * row + x * BLOCK_Q4_0_SIZE;
                out[so..so + 2].copy_from_slice(&tile[2 * i..2 * i + 2]);
            }
            for i in 0..(QK4_0 * 4 / 8) {
                let src_id = i % 8;
                let src_offset = (i / 8) * 8;
                let dst_offset = i * 8;
                let so = base + src_id * row + x * BLOCK_Q4_0_SIZE + 2 + src_offset;
                for (o, &b) in out[so..so + 8]
                    .iter_mut()
                    .zip(&tile[16 + dst_offset..16 + dst_offset + 8])
                {
                    *o = b ^ 0x88;
                }
            }
        }
    }
    out
}

/// Byte of `block_q4_0x8.qs` holding element `e` of column `c`:
/// `qs[(e%16/8)*64 + c*8 + e%8]` (the inverse of `make_block_q4_0x8`'s store
/// loop). q4_0 packs split-half — byte `j`'s low nibble is element `j` and its
/// high nibble is element `16+j` (ggml-quants.c:138-145: `qs[j] = x0;
/// qs[j] |= x1 << 4` with `x0 = x[0+j]`, `x1 = x[qk/2+j]`) — so the low nibble
/// carries `e % 16` and the high nibble `16 + e % 16`, exactly like mxfp4.
#[inline]
fn q4_0x8_nib(tile: &[u8], c: usize, e: usize) -> i32 {
    let j = e % 16;
    let byte = tile[16 + (j / 8) * 64 + c * 8 + j % 8];
    let nib = if e < 16 { byte & 0xF } else { byte >> 4 };
    q4_0_lut(nib)
}

/// Signed value of nibble `n` through the kernels' sign-extend LUT
/// (`set_epi8(-1,..,-8, 7,..,0)`: `n < 8 -> n`, `n >= 8 -> n - 16`).
#[inline]
fn q4_0_lut(n: u8) -> i32 {
    if n < 8 { n as i32 } else { n as i32 - 16 }
}

/// Scalar body of the 8x8 gemv for one activation row: the arithmetic of the
/// C's `gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>` (arch/x86/repack.cpp:522),
/// which is also what [`gemv_q4_0_8x8_q8_0`] runs on AVX2 hosts.
///
/// `vy` points at one plain `block_q8_0` row (`nb * 34` bytes); `nc` must be a
/// multiple of 8, the 8-column groups addressed as `vx + x*nb*144`.
pub fn gemv_q4_0_8x8_q8_0_scalar(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    let nb = n / QK4_0;
    assert_eq!(n % QK4_0, 0, "gemv_q4_0: n % QK4_0 != 0");
    assert_eq!(nc % 8, 0, "gemv_q4_0: nc % 8 != 0");
    for x in 0..nc / 8 {
        let bmat = &vx[x * nb * BLOCK_Q4_0X8_SIZE..];
        let mut sumf = [0f32; 8];
        for b in 0..nb {
            let tile = &bmat[b * BLOCK_Q4_0X8_SIZE..(b + 1) * BLOCK_Q4_0X8_SIZE];
            let q8 = &vy[b * BLOCK_Q8_0_SIZE..(b + 1) * BLOCK_Q8_0_SIZE];
            let d_row = f16_from_bytes(&q8[0..2]).to_f32();
            let q8s = &q8[2..2 + QK4_0];
            for c in 0..8 {
                let d_col = f16_from_bytes(&tile[2 * c..2 * c + 2]).to_f32();
                let mut iacc = 0i32;
                for e in 0..QK4_0 {
                    iacc += q4_0x8_nib(tile, c, e) * (q8s[e] as i8 as i32);
                }
                // acc = fma(cvtepi32(iacc), col_scale * row_scale, acc)
                // (arch/x86/repack.cpp:629)
                sumf[c] = (iacc as f32).mul_add(d_col * d_row, sumf[c]);
            }
        }
        s[x * 8..x * 8 + 8].copy_from_slice(&sumf);
    }
}

/// Scalar body of the 8x8 gemm (4 activation rows per `block_q8_0x4`, `nr`
/// rows total, row stride `bs`, output row `y*4+i` at `s[(y*4+i)*bs + x*8]`).
pub fn gemm_q4_0_8x8_q8_0_scalar(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[u8],
    vy: &[u8],
    nr: usize,
    nc: usize,
) {
    let nb = n / QK4_0;
    assert_eq!(n % QK4_0, 0, "gemm_q4_0: n % QK4_0 != 0");
    assert_eq!(nr % 4, 0, "gemm_q4_0: nr % 4 != 0");
    assert_eq!(nc % 8, 0, "gemm_q4_0: nc % 8 != 0");
    for y in 0..nr / 4 {
        let ablk = &vy[y * nb * BLOCK_Q8_0X4_SIZE..];
        for x in 0..nc / 8 {
            let bmat = &vx[x * nb * BLOCK_Q4_0X8_SIZE..];
            let mut acc = [[0f32; 8]; 4];
            for b in 0..nb {
                let tile = &bmat[b * BLOCK_Q4_0X8_SIZE..(b + 1) * BLOCK_Q4_0X8_SIZE];
                let a = &ablk[b * BLOCK_Q8_0X4_SIZE..(b + 1) * BLOCK_Q8_0X4_SIZE];
                for r in 0..4 {
                    let d_row = f16_from_bytes(&a[2 * r..2 * r + 2]).to_f32();
                    for c in 0..8 {
                        let d_col = f16_from_bytes(&tile[2 * c..2 * c + 2]).to_f32();
                        let mut iacc = 0i32;
                        for e in 0..QK4_0 {
                            // element e of row r: qs[(e/8)*32 + r*8 + e%8]
                            let qv = a[8 + (e / 8) * 32 + r * 8 + e % 8] as i8 as i32;
                            iacc += q4_0x8_nib(tile, c, e) * qv;
                        }
                        acc[r][c] = (iacc as f32).mul_add(d_col * d_row, acc[r][c]);
                    }
                }
            }
            for r in 0..4 {
                let o = (y * 4 + r) * bs + x * 8;
                s[o..o + 8].copy_from_slice(&acc[r]);
            }
        }
    }
}

/// `gemv|gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`'s AVX2 bodies — the same
/// lane networks as the MXFP4 instantiation (the templates are shared,
/// arch/x86/repack.cpp:522/:641) with the Q4_0 specifics:
///   * `signextendlut` = `_mm_set_epi8(-1,..,-8, 7,6,..,0)` broadcast
///     (arch/x86/repack.cpp:1450-1452);
///   * `col_scale_f32` = `GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask)`
///     (:581-583) — the fp16 deltas in the (B0,B4,B1,B5,B2,B6,B3,B7) order;
///   * the tile payload starts 16 bytes in (8 fp16 deltas before `qs`).
#[cfg(target_arch = "x86_64")]
mod simd_x86_q4_0 {
    use super::{BLOCK_Q4_0X8_SIZE, BLOCK_Q8_0X4_SIZE, BLOCK_Q8_0_SIZE, QK4_0};
    use core::arch::x86_64::*;

    /// `mul_sum_i8_pairs_acc_int32x8` (arch/x86/repack.cpp:165), non-VNNI form:
    /// the `sign`+`maddubs`+`madd` chain the port runs everywhere in the
    /// 256-bit bodies (the reference's `-march=native` build would take its
    /// `_mm256_dpbusd_epi32` VNNI form; both compute the same exact int32
    /// sums, `|x| <= 8 x |y| <= 127` pairs never saturate int16 — see
    /// [`msda16`] for where the port does use VNNI).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_sum_i8_pairs_acc_int32x8(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(x, x); // |x| carrying x's sign
        let sy = _mm256_sign_epi8(y, x); // y with x's sign
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_add_epi32(acc, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// The 8 shuffle/blend triples of the gemv inner body
    /// (arch/x86/repack.cpp:616-626): rhs operand + lhs dword broadcast.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn iacc_row(lut: __m256i, m4b: __m256i, tile: *const u8, lhs0: __m256i, lhs1: __m256i) -> __m256i {
        // `block_q4_0x8` = {d[8], qs[128]} — payload starts 16 bytes in.
        let tile = tile.add(16);
        let raw_0123_0 = _mm256_loadu_si256(tile as *const __m256i);
        let raw_4567_0 = _mm256_loadu_si256(tile.add(32) as *const __m256i);
        let raw_0123_1 = _mm256_loadu_si256(tile.add(64) as *const __m256i);
        let raw_4567_1 = _mm256_loadu_si256(tile.add(96) as *const __m256i);
        let r0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, m4b));
        let r4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, m4b));
        let r0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, m4b));
        let r4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, m4b));
        let r0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_0), m4b));
        let r4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_0), m4b));
        let r0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_1), m4b));
        let r4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_1), m4b));

        let mut iacc = _mm256_setzero_si256();
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(r0123_0, _mm256_shuffle_epi32::<177>(r4567_0)),
            _mm256_shuffle_epi32::<0>(lhs0),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_0), r4567_0),
            _mm256_shuffle_epi32::<85>(lhs0),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(r0123_1, _mm256_shuffle_epi32::<177>(r4567_1)),
            _mm256_shuffle_epi32::<170>(lhs0),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_1), r4567_1),
            _mm256_shuffle_epi32::<255>(lhs0),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(r0123_2, _mm256_shuffle_epi32::<177>(r4567_2)),
            _mm256_shuffle_epi32::<0>(lhs1),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_2), r4567_2),
            _mm256_shuffle_epi32::<85>(lhs1),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(r0123_3, _mm256_shuffle_epi32::<177>(r4567_3)),
            _mm256_shuffle_epi32::<170>(lhs1),
        );
        iacc = mul_sum_i8_pairs_acc_int32x8(
            iacc,
            _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_3), r4567_3),
            _mm256_shuffle_epi32::<255>(lhs1),
        );
        iacc
    }

    /// `col_scale_f32` of the q4_0 branch (arch/x86/repack.cpp:581-583):
    /// `GGML_F32Cx8_REARRANGE_LOAD(d, changemask)` — fp16 deltas [d0..d7] in
    /// the (B0,B4,B1,B5,B2,B6,B3,B7) lane order.
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn col_scale_q4_0(d: *const u8) -> __m256 {
        let changemask =
            _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
        _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128(d as *const __m128i), changemask))
    }

    /// The gemv's lhs pair for one `block_q8_0` row (arch/x86/repack.cpp:597-604).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lhs_pair(qs: *const u8) -> (__m256i, __m256i) {
        let lo = _mm_loadu_si128(qs as *const __m128i);
        let hi = _mm_loadu_si128(qs.add(16) as *const __m128i);
        (_mm256_broadcastsi128_si256(lo), _mm256_broadcastsi128_si256(hi))
    }

    /// `GGML_CPU_FP16_TO_FP32` for one `block_q8_0.d` / `block_q8_0x4.d[i]`.
    /// The `vcvtph2ps` hardware instruction — value-identical to the `half`
    /// crate's bit algorithm for all 65536 bit patterns (pinned exhaustively
    /// by `simd_x86`'s `f16c_cvtph_matches_portable`; the reference build gets
    /// the same instruction from its `__F16C__` GGML_CPU_FP16_TO_FP32). The
    /// `half::f16::to_f32` this replaces carries a runtime F16C detection
    /// tree per call — one call per k-block, ~32k calls per gemv on the
    /// gpt-oss expert shape.
    #[inline]
    #[target_feature(enable = "f16c")]
    unsafe fn d_f16(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
            u16::from_le_bytes([*p, *p.add(1)]) as i32,
        )))
    }

    /// AVX2 body of `gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`
    /// (arch/x86/repack.cpp:522-637), `nr == 1` (all C call sites).
    ///
    /// # Safety
    /// `vx` must address `nc/8 * nb * 144` repacked bytes, `vy` one
    /// `block_q8_0` row (`nb * 34` bytes), `s` at least `nc` floats.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemv(n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nc: usize) {
        let nb = n / QK4_0;
        let b_nb = n / 32;
        // signextendlut (arch/x86/repack.cpp:1450): bytes [0,1,..,7,-8,..,-1]
        let lut = {
            let lo = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
            _mm256_castsi128_si256(lo)
        };
        let lut = _mm256_permute2f128_si256::<0>(lut, lut);
        let m4b = _mm256_set1_epi8(0x0F);
        let finalpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

        for x in 0..nc / 8 {
            let b_ptr = vx.add(x * b_nb * BLOCK_Q4_0X8_SIZE);
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let tile = b_ptr.add(b * BLOCK_Q4_0X8_SIZE);
                let a_blk = vy.add(b * BLOCK_Q8_0_SIZE);
                let (lhs0, lhs1) = lhs_pair(a_blk.add(2));
                let iacc = iacc_row(lut, m4b, tile, lhs0, lhs1);
                let d = d_f16(a_blk);
                acc = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc),
                    _mm256_mul_ps(col_scale_q4_0(tile), _mm256_set1_ps(d)),
                    acc,
                );
            }
            let out = _mm256_permutevar8x32_ps(acc, finalpermute);
            _mm256_storeu_ps(s.add(x * 8), out);
        }
    }

    /// `mul_sum_i8_pairs_acc_int32x16` (arch/x86/repack.cpp:140-147): int8
    /// multiply, pairwise into int32, plus the accumulator. `VNNI` selects
    /// the `__AVX512VNNI__` `_mm512_dpbusd_epi32` form of
    /// `mul_sum_us8_pairs_acc_int32x16` (:125) — what the `-march=native`
    /// reference build compiles — or the `abs`/`mask_sub` +
    /// `maddubs`/`madd` chain (`sum_i16_pairs_acc_int32x16`, :118-121).
    /// Both compute the same exact int32 sums (`|x| <= 8 x |y| <= 127`
    /// pairs never saturate int16), so the choice is speed-only; it is a
    /// const parameter (not a runtime `avx512vnni()` check inside the loop)
    /// so the selected form inlines cleanly into the caller.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")]
    unsafe fn msda16<const VNNI: bool>(acc: __m512i, x: __m512i, y: __m512i) -> __m512i {
        let zero = _mm512_setzero_si512();
        let ax = _mm512_abs_epi8(x); // |x|
        let blt0 = _mm512_movepi8_mask(x); // x's sign mask
        let sy = _mm512_mask_sub_epi8(y, blt0, zero, y); // y signed by x
        if VNNI {
            return _mm512_dpbusd_epi32(acc, ax, sy);
        }
        let dot = _mm512_maddubs_epi16(ax, sy);
        _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_set1_epi16(1), dot))
    }

    /// The per-`b` right-hand (weight) lane network shared by both AVX512
    /// loops (arch/x86/repack.cpp:700-766, computed once per `b` *outside*
    /// the row-pair loop — the sharing the port's old per-row gemv-network
    /// body lacked): two `block_q4_0x8` tiles (16 columns) nibble-expanded
    /// through the sign-extend LUT and dword-shuffled into the sp1 (imm 136)
    /// / sp2 (imm 221) operand sets, plus the two tiles' fused
    /// `col_scale_f32` (`GGML_F32Cx8x2_LOAD`, :765 = arch/x86/repack.cpp:30).
    ///
    /// Index layout: `[0..4]` = the `014589CD` set (`_j` = k-octet j),
    /// `[4..8]` = the `2367ABEF` set.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl")]
    unsafe fn rhs_512(
        b0: *const u8,
        b1: *const u8,
        lut: __m512i,
        m4b: __m512i,
        required_order: __m256i,
    ) -> ([__m512i; 8], [__m512i; 8], __m512) {
        // (arch/x86/repack.cpp:700-712) qs starts 16 bytes into the tile.
        let q0 = b0.add(16);
        let q1 = b1.add(16);
        let raw_0123_0 = _mm256_loadu_si256(q0 as *const __m256i);
        let raw_4567_0 = _mm256_loadu_si256(q0.add(32) as *const __m256i);
        let raw_0123_1 = _mm256_loadu_si256(q0.add(64) as *const __m256i);
        let raw_4567_1 = _mm256_loadu_si256(q0.add(96) as *const __m256i);
        let raw_89AB_0 = _mm256_loadu_si256(q1 as *const __m256i);
        let raw_CDEF_0 = _mm256_loadu_si256(q1.add(32) as *const __m256i);
        let raw_89AB_1 = _mm256_loadu_si256(q1.add(64) as *const __m256i);
        let raw_CDEF_1 = _mm256_loadu_si256(q1.add(96) as *const __m256i);
        // (arch/x86/repack.cpp:714-723)
        let raw_0145_0 =
            _mm256_blend_epi32::<240>(raw_0123_0, _mm256_permutevar8x32_epi32(raw_4567_0, required_order));
        let raw_2367_0 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_0123_0, required_order), raw_4567_0);
        let raw_0145_1 =
            _mm256_blend_epi32::<240>(raw_0123_1, _mm256_permutevar8x32_epi32(raw_4567_1, required_order));
        let raw_2367_1 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_0123_1, required_order), raw_4567_1);
        let raw_89CD_0 =
            _mm256_blend_epi32::<240>(raw_89AB_0, _mm256_permutevar8x32_epi32(raw_CDEF_0, required_order));
        let raw_ABEF_0 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_89AB_0, required_order), raw_CDEF_0);
        let raw_89CD_1 =
            _mm256_blend_epi32::<240>(raw_89AB_1, _mm256_permutevar8x32_epi32(raw_CDEF_1, required_order));
        let raw_ABEF_1 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_89AB_1, required_order), raw_CDEF_1);
        // (arch/x86/repack.cpp:725-728)
        let raw_014589CD_0 = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_0145_0), raw_89CD_0);
        let raw_2367ABEF_0 = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_2367_0), raw_ABEF_0);
        let raw_014589CD_1 = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_0145_1), raw_89CD_1);
        let raw_2367ABEF_1 = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(raw_2367_1), raw_ABEF_1);
        // 4-bit -> 8-bit, sign maintained (arch/x86/repack.cpp:730-740)
        let lut_ = |v: __m512i| _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4b));
        let srl4 = |v: __m512i| _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(v), m4b));
        let mat_014589CD = [lut_(raw_014589CD_0), lut_(raw_014589CD_1), srl4(raw_014589CD_0), srl4(raw_014589CD_1)];
        let mat_2367ABEF = [lut_(raw_2367ABEF_0), lut_(raw_2367ABEF_1), srl4(raw_2367ABEF_0), srl4(raw_2367ABEF_1)];
        // dword shuffle patterns (arch/x86/repack.cpp:742-766)
        let mut sp1 = [_mm512_setzero_si512(); 8];
        let mut sp2 = [_mm512_setzero_si512(); 8];
        for j in 0..4 {
            sp1[j] = _mm512_shuffle_epi32::<136>(mat_014589CD[j]);
            sp2[j] = _mm512_shuffle_epi32::<221>(mat_014589CD[j]);
            sp1[j + 4] = _mm512_shuffle_epi32::<136>(mat_2367ABEF[j]);
            sp2[j + 4] = _mm512_shuffle_epi32::<221>(mat_2367ABEF[j]);
        }
        // col_scale_f32 (arch/x86/repack.cpp:765, macro at :30)
        let col = _mm512_cvtph_ps(_mm256_set_m128i(
            _mm_loadu_si128(b1 as *const __m128i),
            _mm_loadu_si128(b0 as *const __m128i),
        ));
        (sp1, sp2, col)
    }

    /// The per-row-pair left-hand (activation) lane network of the AVX512
    /// body (arch/x86/repack.cpp:805-851): one `block_q8_0x4` tile's qs
    /// payload, each 128-bit half duplicated to 512 bits, dword-shuffled
    /// into the sp1 (160) / sp2 (245) patterns. `[0..4]` = row pair
    /// `(A0,A1)`, `[4..8]` = `(A2,A3)`.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl")]
    unsafe fn lhs_512(a: *const u8) -> ([__m512i; 8], [__m512i; 8]) {
        // qs starts 8 bytes into the tile (arch/x86/repack.cpp:807-821).
        let qs = a.add(8);
        let raw0 = _mm256_loadu_si256(qs as *const __m256i);
        let raw1 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
        let raw2 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
        let raw3 = _mm256_loadu_si256(qs.add(96) as *const __m256i);
        let y01_0 = _mm256_permute2f128_si256::<0>(raw0, raw0);
        let y23_0 = _mm256_permute2f128_si256::<17>(raw0, raw0);
        let y01_1 = _mm256_permute2f128_si256::<0>(raw1, raw1);
        let y23_1 = _mm256_permute2f128_si256::<17>(raw1, raw1);
        let y01_2 = _mm256_permute2f128_si256::<0>(raw2, raw2);
        let y23_2 = _mm256_permute2f128_si256::<17>(raw2, raw2);
        let y01_3 = _mm256_permute2f128_si256::<0>(raw3, raw3);
        let y23_3 = _mm256_permute2f128_si256::<17>(raw3, raw3);
        let lhs01 = [
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y01_0), y01_0),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y01_1), y01_1),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y01_2), y01_2),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y01_3), y01_3),
        ];
        let lhs23 = [
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y23_0), y23_0),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y23_1), y23_1),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y23_2), y23_2),
            _mm512_inserti32x8::<1>(_mm512_castsi256_si512(y23_3), y23_3),
        ];
        // (arch/x86/repack.cpp:823-851)
        let mut sp1 = [_mm512_setzero_si512(); 8];
        let mut sp2 = [_mm512_setzero_si512(); 8];
        for j in 0..4 {
            sp1[j] = _mm512_shuffle_epi32::<160>(lhs01[j]);
            sp1[j + 4] = _mm512_shuffle_epi32::<160>(lhs23[j]);
            sp2[j] = _mm512_shuffle_epi32::<245>(lhs01[j]);
            sp2[j + 4] = _mm512_shuffle_epi32::<245>(lhs23[j]);
        }
        (sp1, sp2)
    }

    /// The 4-deep `mul_sum_i8_pairs_acc_int32x16` chain of one `iacc_mat`
    /// cell (arch/x86/repack.cpp:855-862), innermost `j = 3` first.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")]
    unsafe fn iacc_mat_cell_512<const VNNI: bool>(lhs: &[__m512i], rhs: &[__m512i]) -> __m512i {
        let zero = _mm512_setzero_si512();
        let a = msda16::<VNNI>(zero, lhs[3], rhs[3]);
        let a = msda16::<VNNI>(a, lhs[2], rhs[2]);
        let a = msda16::<VNNI>(a, lhs[1], rhs[1]);
        msda16::<VNNI>(a, lhs[0], rhs[0])
    }

    /// The AVX512 gemm's per-(row pair set, b) epilogue: the four
    /// `iacc_mat_XX = sp1 + sp2` sums, the `0xCCCC`-mask straightening into
    /// row vectors, the fp16 row scales and the four fmadds
    /// (arch/x86/repack.cpp:864-888).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")]
    unsafe fn acc_rows_512<const VNNI: bool>(
        a_tile: *const u8,
        l_sp1: &[__m512i; 8],
        l_sp2: &[__m512i; 8],
        r_sp1: &[__m512i; 8],
        r_sp2: &[__m512i; 8],
        col: __m512,
        load_mask: __m128i,
        acc: &mut [__m512],
        base: usize,
    ) {
        let i00 = _mm512_add_epi32(
            iacc_mat_cell_512::<VNNI>(&l_sp1[..4], &r_sp1[..4]),
            iacc_mat_cell_512::<VNNI>(&l_sp2[..4], &r_sp2[..4]),
        );
        let i01 = _mm512_add_epi32(
            iacc_mat_cell_512::<VNNI>(&l_sp1[..4], &r_sp1[4..]),
            iacc_mat_cell_512::<VNNI>(&l_sp2[..4], &r_sp2[4..]),
        );
        let i10 = _mm512_add_epi32(
            iacc_mat_cell_512::<VNNI>(&l_sp1[4..], &r_sp1[..4]),
            iacc_mat_cell_512::<VNNI>(&l_sp2[4..], &r_sp2[..4]),
        );
        let i11 = _mm512_add_epi32(
            iacc_mat_cell_512::<VNNI>(&l_sp1[4..], &r_sp1[4..]),
            iacc_mat_cell_512::<VNNI>(&l_sp2[4..], &r_sp2[4..]),
        );
        // straighten out to 4 row vectors (arch/x86/repack.cpp:872-875)
        let row0 = _mm512_mask_blend_epi32(0xCCCC, i00, _mm512_shuffle_epi32::<78>(i01));
        let row1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i00), i01);
        let row2 = _mm512_mask_blend_epi32(0xCCCC, i10, _mm512_shuffle_epi32::<78>(i11));
        let row3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32::<78>(i10), i11);
        // the 4 Q8_0 row scales repeated across lanes
        // (arch/x86/repack.cpp:880-881 = :31 `GGML_F32Cx16_REPEAT_LOAD`)
        let row_scale_f16 = _mm_shuffle_epi32::<68>(_mm_maskload_epi32(a_tile as *const i32, load_mask));
        let rs = _mm512_cvtph_ps(_mm256_set_m128i(row_scale_f16, row_scale_f16));
        // (arch/x86/repack.cpp:885-888)
        acc[base] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row0),
            _mm512_mul_ps(col, _mm512_shuffle_ps::<0>(rs, rs)),
            acc[base],
        );
        acc[base + 1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row1),
            _mm512_mul_ps(col, _mm512_shuffle_ps::<85>(rs, rs)),
            acc[base + 1],
        );
        acc[base + 2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row2),
            _mm512_mul_ps(col, _mm512_shuffle_ps::<170>(rs, rs)),
            acc[base + 2],
        );
        acc[base + 3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(row3),
            _mm512_mul_ps(col, _mm512_shuffle_ps::<255>(rs, rs)),
            acc[base + 3],
        );
    }

    /// The per-`b` right-hand lane network of the AVX2 gemm body
    /// (arch/x86/repack.cpp:1112-1182): one `block_q4_0x8` tile (8 columns)
    /// through the same sp1/sp2 construction at 256 bits, plus the tile's
    /// `col_scale_f32` (`GGML_F32Cx8_LOAD`, :1177 = :34). Same index layout
    /// as [`rhs_512`].
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn rhs_256(
        b: *const u8,
        lut: __m256i,
        m4b: __m256i,
        required_order: __m256i,
    ) -> ([__m256i; 8], [__m256i; 8], __m256) {
        // (arch/x86/repack.cpp:1112-1115)
        let qs = b.add(16);
        let raw_0123_0 = _mm256_loadu_si256(qs as *const __m256i);
        let raw_4567_0 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
        let raw_0123_1 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
        let raw_4567_1 = _mm256_loadu_si256(qs.add(96) as *const __m256i);
        // (arch/x86/repack.cpp:1118-1121)
        let raw_0145_0 =
            _mm256_blend_epi32::<240>(raw_0123_0, _mm256_permutevar8x32_epi32(raw_4567_0, required_order));
        let raw_2367_0 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_0123_0, required_order), raw_4567_0);
        let raw_0145_1 =
            _mm256_blend_epi32::<240>(raw_0123_1, _mm256_permutevar8x32_epi32(raw_4567_1, required_order));
        let raw_2367_1 =
            _mm256_blend_epi32::<240>(_mm256_permutevar8x32_epi32(raw_0123_1, required_order), raw_4567_1);
        // 4-bit -> 8-bit (arch/x86/repack.cpp:1123-1133)
        let lut_ = |v: __m256i| _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4b));
        let srl4 = |v: __m256i| _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4b));
        let mat_0145 = [lut_(raw_0145_0), lut_(raw_0145_1), srl4(raw_0145_0), srl4(raw_0145_1)];
        let mat_2367 = [lut_(raw_2367_0), lut_(raw_2367_1), srl4(raw_2367_0), srl4(raw_2367_1)];
        // (arch/x86/repack.cpp:1135-1165)
        let mut sp1 = [_mm256_setzero_si256(); 8];
        let mut sp2 = [_mm256_setzero_si256(); 8];
        for j in 0..4 {
            sp1[j] = _mm256_shuffle_epi32::<136>(mat_0145[j]);
            sp2[j] = _mm256_shuffle_epi32::<221>(mat_0145[j]);
            sp1[j + 4] = _mm256_shuffle_epi32::<136>(mat_2367[j]);
            sp2[j + 4] = _mm256_shuffle_epi32::<221>(mat_2367[j]);
        }
        // (arch/x86/repack.cpp:1177, macro at :34)
        let col = _mm256_cvtph_ps(_mm_loadu_si128(b as *const __m128i));
        (sp1, sp2, col)
    }

    /// The per-row-pair left-hand lane network of the AVX2 gemm body
    /// (arch/x86/repack.cpp:1191-1241), same construction as [`lhs_512`]
    /// but staying in 256 bits.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn lhs_256(a: *const u8) -> ([__m256i; 8], [__m256i; 8]) {
        let qs = a.add(8);
        let raw0 = _mm256_loadu_si256(qs as *const __m256i);
        let raw1 = _mm256_loadu_si256(qs.add(32) as *const __m256i);
        let raw2 = _mm256_loadu_si256(qs.add(64) as *const __m256i);
        let raw3 = _mm256_loadu_si256(qs.add(96) as *const __m256i);
        let lhs01 = [
            _mm256_permute2f128_si256::<0>(raw0, raw0),
            _mm256_permute2f128_si256::<0>(raw1, raw1),
            _mm256_permute2f128_si256::<0>(raw2, raw2),
            _mm256_permute2f128_si256::<0>(raw3, raw3),
        ];
        let lhs23 = [
            _mm256_permute2f128_si256::<17>(raw0, raw0),
            _mm256_permute2f128_si256::<17>(raw1, raw1),
            _mm256_permute2f128_si256::<17>(raw2, raw2),
            _mm256_permute2f128_si256::<17>(raw3, raw3),
        ];
        // (arch/x86/repack.cpp:1221-1241)
        let mut sp1 = [_mm256_setzero_si256(); 8];
        let mut sp2 = [_mm256_setzero_si256(); 8];
        for j in 0..4 {
            sp1[j] = _mm256_shuffle_epi32::<160>(lhs01[j]);
            sp1[j + 4] = _mm256_shuffle_epi32::<160>(lhs23[j]);
            sp2[j] = _mm256_shuffle_epi32::<245>(lhs01[j]);
            sp2[j + 4] = _mm256_shuffle_epi32::<245>(lhs23[j]);
        }
        (sp1, sp2)
    }

    /// The AVX2 `iacc_mat` cell chain (arch/x86/repack.cpp:1247-1254).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn iacc_mat_cell_256(lhs: &[__m256i], rhs: &[__m256i]) -> __m256i {
        let zero = _mm256_setzero_si256();
        let a = mul_sum_i8_pairs_acc_int32x8(zero, lhs[3], rhs[3]);
        let a = mul_sum_i8_pairs_acc_int32x8(a, lhs[2], rhs[2]);
        let a = mul_sum_i8_pairs_acc_int32x8(a, lhs[1], rhs[1]);
        mul_sum_i8_pairs_acc_int32x8(a, lhs[0], rhs[0])
    }

    /// The AVX2 gemm's per-(row pair set, b) epilogue
    /// (arch/x86/repack.cpp:1256-1280): the mirror of [`acc_rows_512`] with
    /// 204-mask blends and `GGML_F32Cx8_REPEAT_LOAD` (:35) row scales.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn acc_rows_256(
        a_tile: *const u8,
        l_sp1: &[__m256i; 8],
        l_sp2: &[__m256i; 8],
        r_sp1: &[__m256i; 8],
        r_sp2: &[__m256i; 8],
        col: __m256,
        load_mask: __m128i,
        acc: &mut [__m256],
        base: usize,
    ) {
        let i00 = _mm256_add_epi32(
            iacc_mat_cell_256(&l_sp1[..4], &r_sp1[..4]),
            iacc_mat_cell_256(&l_sp2[..4], &r_sp2[..4]),
        );
        let i01 = _mm256_add_epi32(
            iacc_mat_cell_256(&l_sp1[..4], &r_sp1[4..]),
            iacc_mat_cell_256(&l_sp2[..4], &r_sp2[4..]),
        );
        let i10 = _mm256_add_epi32(
            iacc_mat_cell_256(&l_sp1[4..], &r_sp1[..4]),
            iacc_mat_cell_256(&l_sp2[4..], &r_sp2[..4]),
        );
        let i11 = _mm256_add_epi32(
            iacc_mat_cell_256(&l_sp1[4..], &r_sp1[4..]),
            iacc_mat_cell_256(&l_sp2[4..], &r_sp2[4..]),
        );
        // (arch/x86/repack.cpp:1258-1261)
        let row0 = _mm256_blend_epi32::<204>(i00, _mm256_shuffle_epi32::<78>(i01));
        let row1 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(i00), i01);
        let row2 = _mm256_blend_epi32::<204>(i10, _mm256_shuffle_epi32::<78>(i11));
        let row3 = _mm256_blend_epi32::<204>(_mm256_shuffle_epi32::<78>(i10), i11);
        // (arch/x86/repack.cpp:1264, macro at :35)
        let rs = _mm256_cvtph_ps(_mm_shuffle_epi32::<68>(_mm_maskload_epi32(a_tile as *const i32, load_mask)));
        // (arch/x86/repack.cpp:1266-1270)
        acc[base] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row0),
            _mm256_mul_ps(col, _mm256_shuffle_ps::<0>(rs, rs)),
            acc[base],
        );
        acc[base + 1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row1),
            _mm256_mul_ps(col, _mm256_shuffle_ps::<85>(rs, rs)),
            acc[base + 1],
        );
        acc[base + 2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row2),
            _mm256_mul_ps(col, _mm256_shuffle_ps::<170>(rs, rs)),
            acc[base + 2],
        );
        acc[base + 3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(row3),
            _mm256_mul_ps(col, _mm256_shuffle_ps::<255>(rs, rs)),
            acc[base + 3],
        );
    }

    /// The AVX2 section of `gemm_q4_b32_8x8_q8_0_lut_avx` (arch/x86/
    /// repack.cpp:1100-1445): the 2x2 `iacc_mat` dot-product network at 256
    /// bits, groups of four `block_q8_0x4` (16 rows) then the 4-row tail, 8
    /// columns per pass, `x` starting at `xstart` — 0 on an AVX2-only host
    /// (the whole matrix) and `anc/8` on an AVX512 host (the `nc % 16` tail
    /// columns).
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn gemm_256_section(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
        xstart: usize,
    ) {
        let nb = n / QK4_0;
        let b_nb = n / 32;
        // signextendlut (arch/x86/repack.cpp:2027-2029)
        let lut = {
            let lo = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
            let full = _mm256_castsi128_si256(lo);
            _mm256_permute2f128_si256::<0>(full, full)
        };
        let m4b = _mm256_set1_epi8(0x0F);
        let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
        let load_mask = _mm_set_epi32(0, 0, -1, -1);
        let anr = nr - nr % 16;
        let mut y = 0usize;
        // groups of four block_q8_0x4 x 8 columns (arch/x86/repack.cpp:1100-1283)
        while y < anr / 4 {
            for x in xstart..nc / 8 {
                let b_ptr = vx.add(x * b_nb * BLOCK_Q4_0X8_SIZE);
                let mut acc = [_mm256_setzero_ps(); 16];
                for b in 0..nb {
                    let (r_sp1, r_sp2, col) =
                        rhs_256(b_ptr.add(b * BLOCK_Q4_0X8_SIZE), lut, m4b, required_order);
                    for rp in 0..4 {
                        let a_tile = vy.add(((y + rp) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        let (l_sp1, l_sp2) = lhs_256(a_tile);
                        acc_rows_256(a_tile, &l_sp1, &l_sp2, &r_sp1, &r_sp2, col, load_mask, &mut acc, rp * 4);
                    }
                }
                // (arch/x86/repack.cpp:1275-1279)
                for i in 0..16 {
                    _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 4;
        }
        // one block_q8_0x4 x 8 columns (arch/x86/repack.cpp:1285-1445)
        while y < nr / 4 {
            for x in xstart..nc / 8 {
                let b_ptr = vx.add(x * b_nb * BLOCK_Q4_0X8_SIZE);
                let mut acc = [_mm256_setzero_ps(); 4];
                for b in 0..nb {
                    let (r_sp1, r_sp2, col) =
                        rhs_256(b_ptr.add(b * BLOCK_Q4_0X8_SIZE), lut, m4b, required_order);
                    let a_tile = vy.add((y * nb + b) * BLOCK_Q8_0X4_SIZE);
                    let (l_sp1, l_sp2) = lhs_256(a_tile);
                    acc_rows_256(a_tile, &l_sp1, &l_sp2, &r_sp1, &r_sp2, col, load_mask, &mut acc, 0);
                }
                for i in 0..4 {
                    _mm256_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 1;
        }
    }

    /// The `__AVX512BW__ && __AVX512DQ__` section of
    /// `gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>` (arch/x86/repack.cpp:
    /// 660-1110): the 512-bit 2x2 network, 16 activation rows x 16 columns
    /// per pass then the 4-row tail, handing the `nc % 16` tail columns to
    /// the 256-bit section via `xstart = anc/8`. `VNNI` picks the dot op
    /// (see [`msda16`]); the runtime check lives in the [`gemm`] dispatcher
    /// so each specialization is branch-free.
    #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")]
    unsafe fn gemm_avx512<const VNNI: bool>(
        n: usize,
        s: *mut f32,
        bs: usize,
        vx: *const u8,
        vy: *const u8,
        nr: usize,
        nc: usize,
    ) {
        let nb = n / QK4_0;
        let b_nb = n / 32;
        // signextendlut (arch/x86/repack.cpp:2027-2029), expanded to 512 bits
        // (:671 `signextendlutexpanded`)
        let lut = {
            let lo = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
            let full = _mm256_castsi128_si256(lo);
            _mm256_permute2f128_si256::<0>(full, full)
        };
        let lut512 = _mm512_inserti32x8::<1>(_mm512_castsi256_si512(lut), lut);
        let m4b512 = _mm512_set1_epi8(0x0F);
        // (arch/x86/repack.cpp:667-668)
        let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
        let load_mask = _mm_set_epi32(0, 0, -1, -1);
        let anr = nr - nr % 16;
        let anc = nc - nc % 16;
        let mut y = 0usize;

        // groups of four block_q8_0x4 (16 activation rows) x 16 columns
        // (arch/x86/repack.cpp:660-908)
        while y < anr / 4 {
            for x in (0..anc / 8).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_Q4_0X8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_Q4_0X8_SIZE);
                let mut acc = [_mm512_setzero_ps(); 16];
                for b in 0..nb {
                    let (r_sp1, r_sp2, col) = rhs_512(
                        b0.add(b * BLOCK_Q4_0X8_SIZE),
                        b1.add(b * BLOCK_Q4_0X8_SIZE),
                        lut512,
                        m4b512,
                        required_order,
                    );
                    for rp in 0..4 {
                        let a_tile = vy.add(((y + rp) * nb + b) * BLOCK_Q8_0X4_SIZE);
                        let (l_sp1, l_sp2) = lhs_512(a_tile);
                        acc_rows_512::<VNNI>(a_tile, &l_sp1, &l_sp2, &r_sp1, &r_sp2, col, load_mask, &mut acc, rp * 4);
                    }
                }
                // (arch/x86/repack.cpp:904-906)
                for i in 0..16 {
                    _mm512_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 4;
        }
        // one block_q8_0x4 (4 activation rows) x 16 columns
        // (arch/x86/repack.cpp:910-1105)
        while y < nr / 4 {
            for x in (0..anc / 8).step_by(2) {
                let b0 = vx.add(x * b_nb * BLOCK_Q4_0X8_SIZE);
                let b1 = vx.add((x + 1) * b_nb * BLOCK_Q4_0X8_SIZE);
                let mut acc = [_mm512_setzero_ps(); 4];
                for b in 0..nb {
                    let (r_sp1, r_sp2, col) = rhs_512(
                        b0.add(b * BLOCK_Q4_0X8_SIZE),
                        b1.add(b * BLOCK_Q4_0X8_SIZE),
                        lut512,
                        m4b512,
                        required_order,
                    );
                    let a_tile = vy.add((y * nb + b) * BLOCK_Q8_0X4_SIZE);
                    let (l_sp1, l_sp2) = lhs_512(a_tile);
                    acc_rows_512::<VNNI>(a_tile, &l_sp1, &l_sp2, &r_sp1, &r_sp2, col, load_mask, &mut acc, 0);
                }
                for i in 0..4 {
                    _mm512_storeu_ps(s.add((y * 4 + i) * bs + x * 8), acc[i]);
                }
            }
            y += 1;
        }
        // the `nc % 16` tail columns go through the AVX2 section
        // (arch/x86/repack.cpp:1107-1110)
        if anc != nc {
            gemm_256_section(n, s, bs, vx, vy, nr, nc, anc / 8);
        }
    }

    /// `gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>` (arch/x86/repack.cpp:641),
    /// the gemm body `ggml_gemm_q4_0_8x8_q8_0` (:2022-2035) runs on every
    /// AVX2-or-better host: on an AVX512BW+DQ host the 512-bit section
    /// (`gemm_avx512`, the `-march=native` reference build's compiled form)
    /// with the VNNI dot op when the host has AVX512 VNNI; otherwise the
    /// 256-bit section over the whole matrix — the C template's `#else`
    /// build.
    ///
    /// The port's previous body replicated the *gemv's* per-row lane network
    /// (each of the 16 rows re-ran the weight loads/LUT shuffles of its
    /// column tile); the C's structure computes the weight network once per
    /// `b` and shares it across the row pairs, which is where its throughput
    /// lives. Per-element arithmetic (exact int32 dot, one rounded
    /// `d_col*d_row` product, one fma per block) is unchanged — bit-exact
    /// with the scalar body and the reference's dump.
    ///
    /// # Safety
    /// `vy` must address `nr/4 * nb` tiles of 136 bytes and `s` at least
    /// `nr * bs` floats.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm(n: usize, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: usize, nc: usize) {
        if crate::simd_x86::avx512bw() {
            if crate::simd_x86::avx512vnni() {
                gemm_avx512::<true>(n, s, bs, vx, vy, nr, nc);
            } else {
                gemm_avx512::<false>(n, s, bs, vx, vy, nr, nc);
            }
            return;
        }
        gemm_256_section(n, s, bs, vx, vy, nr, nc, 0);
    }
}

/// `ggml_gemv_q4_0_8x8_q8_0` (repack.cpp:4361 -> arch/x86/repack.cpp:1448).
///
/// `nc` is the number of weight rows (a multiple of 8); the kernel writes `nc`
/// floats. AVX2 hosts take the transcribed C body, others the scalar body with
/// identical arithmetic; `LLAMA_RUST_REPACK_SIMD=0` forces the scalar body.
pub fn gemv_q4_0_8x8_q8_0(n: usize, s: &mut [f32], vx: &[u8], vy: &[u8], nc: usize) {
    assert_eq!(n % QK4_0, 0, "gemv_q4_0: n % QK4_0 != 0");
    assert_eq!(nc % 8, 0, "gemv_q4_0: nc % 8 != 0");
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() && simd_enabled() {
        // SAFETY: the caller passes the repacked 8x8 matrix (`nc/8 * nb * 144`
        // bytes), one `block_q8_0` row (`nb * 34` bytes) and `nc` output
        // floats; `nc % 8 == 0` and `n % QK4_0 == 0` are asserted above.
        unsafe {
            simd_x86_q4_0::gemv(n, s.as_mut_ptr(), vx.as_ptr(), vy.as_ptr(), nc);
        }
        note_q4_0_gemv_call();
        return;
    }
    note_q4_0_gemv_call();
    gemv_q4_0_8x8_q8_0_scalar(n, s, vx, vy, nc);
}

/// `ggml_gemm_q4_0_8x8_q8_0` (repack.cpp:4473 -> arch/x86/repack.cpp:2022).
///
/// `nr` activation rows must be a multiple of 4 (the C asserts it); the
/// activation rows come from `block_q8_0x4` tiles (`ggml_quantize_mat_q8_0_4x8`).
pub fn gemm_q4_0_8x8_q8_0(
    n: usize,
    s: &mut [f32],
    bs: usize,
    vx: &[u8],
    vy: &[u8],
    nr: usize,
    nc: usize,
) {
    assert_eq!(n % QK4_0, 0, "gemm_q4_0: n % QK4_0 != 0");
    assert_eq!(nr % 4, 0, "gemm_q4_0: nr % 4 != 0");
    assert_eq!(nc % 8, 0, "gemm_q4_0: nc % 8 != 0");
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() && simd_enabled() {
        // SAFETY: as in `gemv_q4_0_8x8_q8_0`; `vy` holds `nr/4 * nb` tiles of
        // 136 bytes and `s` at least `nr * bs` floats.
        unsafe {
            simd_x86_q4_0::gemm(n, s.as_mut_ptr(), bs, vx.as_ptr(), vy.as_ptr(), nr, nc);
        }
        note_q4_0_gemm_call();
        return;
    }
    note_q4_0_gemm_call();
    gemm_q4_0_8x8_q8_0_scalar(n, s, bs, vx, vy, nr, nc);
}

// ---------------------------------------------------------------------------
// lazy repack cache (the port's stand-in for the CPU_REPACK buffer)
// ---------------------------------------------------------------------------

/// Process-wide repack cache.
///
/// The reference materializes the 8x8 bytes once at load time into an anonymous
/// `CPU_REPACK` buffer (gpt-oss-20b MXFP4: ~10.1 GiB, matching the server's
/// measured ~10 GiB anonymous RSS). Our weights stay mmap'd and zero-copy, so
/// the equivalent bytes are built on first use and kept for the process
/// lifetime, keyed by `(data pointer, byte length, ne[1], ne[0])` plus a
/// content fingerprint of the first 64 bytes — the fingerprint makes a stale hit
/// impossible when a freed mmap's address is reused by a *different* tensor
/// (the shape alone would not distinguish e.g. two requantizations of the same
/// model).
///
/// `LLAMA_RUST_REPACK=0` disables the path (falls back to the row-wise
/// `vec_dot_mxfp4_q8_0`, i.e. the pre-repack port behaviour).
/// `LLAMA_RUST_REPACK_MAX_MB` caps the total cached bytes (default 16384 MiB);
/// a tensor that would exceed the cap is not cached (and the caller falls back).
pub fn repack_enabled() -> bool {
    match REPACK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        x if x == RepackOverride::On as u8 => return true,
        x if x == RepackOverride::Off as u8 => return false,
        _ => {}
    }
    match std::env::var("LLAMA_RUST_REPACK") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true, // the reference defaults to CPU_REPACK ON
    }
}

/// `mparams.use_extra_bufts` override surface (llama-bench `--repack`,
/// default `llama_model_default_params().use_extra_bufts` = true,
/// llama-model.cpp:2924): None = follow `LLAMA_RUST_REPACK`
static REPACK_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[derive(Clone, Copy, PartialEq)]
enum RepackOverride {
    Env = 0,
    On = 1,
    Off = 2,
}

/// force the repack path on/off regardless of `LLAMA_RUST_REPACK`
pub fn set_repack_override(on: Option<bool>) {
    let v = match on {
        None => RepackOverride::Env,
        Some(true) => RepackOverride::On,
        Some(false) => RepackOverride::Off,
    };
    REPACK_OVERRIDE.store(v as u8, std::sync::atomic::Ordering::Relaxed);
}

/// Bench-only switch: `LLAMA_RUST_REPACK_SIMD=0` forces the scalar kernels so the
/// A/B of the two implementations can be measured end-to-end (they are
/// bit-identical, so this never changes results, only speed).
pub fn simd_enabled() -> bool {
    match std::env::var("LLAMA_RUST_REPACK_SIMD") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn cache_budget() -> usize {
    std::env::var("LLAMA_RUST_REPACK_MAX_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16 * 1024)
        * 1024
        * 1024
}

type CacheKey = (usize, usize, usize, usize, u64);

fn cache() -> &'static Mutex<HashMap<CacheKey, Arc<Vec<u8>>>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<Vec<u8>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// (tensors materialized, bytes materialized) — how much the repack cache has
/// built so far (hits do not count). Lets tests assert the path was actually
/// taken, and accounts for the RSS the reference spends on its CPU_REPACK buffer.
pub fn repack_stats() -> (usize, usize) {
    (
        STAT_TENSORS.load(std::sync::atomic::Ordering::Relaxed),
        STAT_BYTES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Milliseconds spent building 8x8 copies (the reference spends the equivalent
/// inside its model load; ours lands in the first forward pass, so timing code
/// must subtract it).
pub fn repack_materialize_ms() -> f64 {
    STAT_MS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3 // micros -> ms
}

/// Number of `gemv_mxfp4_8x8_q8_0` calls issued — a direct witness that the
/// compute.rs repack branch (and not the row-wise vec_dot fallback) ran.
pub fn repack_gemv_calls() -> usize {
    STAT_GEMV_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Number of Q4_K 8x8 `gemv` calls issued (witness for the Q4_K routing).
pub fn repack_q4k_gemv_calls() -> usize {
    STAT_Q4K_GEMV_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Number of Q4_K 8x8 `gemm` calls issued (witness for the Q4_K routing).
pub fn repack_q4k_gemm_calls() -> usize {
    STAT_Q4K_GEMM_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Number of Q4_0 8x8 `gemv` calls issued (witness for the Q4_0 routing).
pub fn repack_q4_0_gemv_calls() -> usize {
    STAT_Q4_0_GEMV_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Number of Q4_0 8x8 `gemm` calls issued (witness for the Q4_0 routing).
pub fn repack_q4_0_gemm_calls() -> usize {
    STAT_Q4_0_GEMM_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

static STAT_TENSORS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STAT_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// microseconds spent in materialization
static STAT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STAT_GEMV_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STAT_Q4K_GEMV_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STAT_Q4K_GEMM_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STAT_Q4_0_GEMV_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static STAT_Q4_0_GEMM_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn note_q4k_gemv_call() {
    STAT_Q4K_GEMV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_q4k_gemm_call() {
    STAT_Q4K_GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_q4_0_gemv_call() {
    STAT_Q4_0_GEMV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_q4_0_gemm_call() {
    STAT_Q4_0_GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn note_materialized(bytes: usize, micros: u64) {
    STAT_TENSORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAT_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    STAT_MS.fetch_add(micros, std::sync::atomic::Ordering::Relaxed);
}

/// FNV-1a over the first 64 bytes (a cheap "is this really the same tensor"
/// guard; the mmap address+length alone can be recycled by the allocator).
fn fingerprint(src: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in src.iter().take(64) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Repack a whole MXFP4 weight tensor (all rows, all experts) into the 8x8
/// layout, reusing a cached copy when the same tensor was seen before.
///
/// Returns `None` when the tensor does not qualify
/// ([`repack_supported_mxfp4`]) or the cache budget is exhausted.
pub fn repack_mxfp4_8x8_cached(
    data_ptr: usize,
    src: &[u8],
    ne1: i64,
    ne0: i64,
) -> Option<Arc<Vec<u8>>> {
    if !repack_supported_mxfp4(ne1, ne0) {
        return None;
    }
    let nrows = src.len() / ((ne0 as usize / QK_MXFP4) * BLOCK_MXFP4_SIZE);
    let key: CacheKey = (data_ptr, src.len(), ne1 as usize, ne0 as usize, fingerprint(src));
    {
        let map = cache().lock().unwrap();
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
        let used: usize = map.values().map(|v| v.len()).sum();
        if used + src.len() > cache_budget() {
            return None;
        }
    }
    let t_start = std::time::Instant::now();
    let out = Arc::new(repack_mxfp4_8x8(src, nrows, ne0 as usize));
    note_materialized(out.len(), t_start.elapsed().as_micros() as u64);
    let mut map = cache().lock().unwrap();
    // another thread may have won the race; keep whichever copy is there
    Some(map.entry(key).or_insert(out).clone())
}

/// [`repack_mxfp4_8x8_cached`] for Q4_K (`repack_q4_K_to_q4_K_8_bl`,
/// repack.cpp:3573). Same cache, same `LLAMA_RUST_REPACK*` switches.
pub fn repack_q4_K_8x8_cached(
    data_ptr: usize,
    src: &[u8],
    ne1: i64,
    ne0: i64,
) -> Option<Arc<Vec<u8>>> {
    if !repack_supported_q4k(ne1, ne0) {
        return None;
    }
    let nrows = src.len() / ((ne0 as usize / QK_K) * BLOCK_Q4_K_SIZE);
    let key: CacheKey = (data_ptr, src.len(), ne1 as usize, ne0 as usize, fingerprint(src));
    {
        let map = cache().lock().unwrap();
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
        let used: usize = map.values().map(|v| v.len()).sum();
        if used + src.len() > cache_budget() {
            return None;
        }
    }
    let t_start = std::time::Instant::now();
    let out = Arc::new(repack_q4_K_8x8(src, nrows, ne0 as usize));
    note_materialized(out.len(), t_start.elapsed().as_micros() as u64);
    let mut map = cache().lock().unwrap();
    Some(map.entry(key).or_insert(out).clone())
}

/// [`repack_q4_K_8x8_cached`] for Q4_0 (`repack_q4_0_to_q4_0_8_bl`,
/// repack.cpp:3790). Same cache, same `LLAMA_RUST_REPACK*` switches.
pub fn repack_q4_0_8x8_cached(
    data_ptr: usize,
    src: &[u8],
    ne1: i64,
    ne0: i64,
) -> Option<Arc<Vec<u8>>> {
    if !repack_supported_q4_0(ne1, ne0) {
        return None;
    }
    let nrows = src.len() / ((ne0 as usize / QK4_0) * BLOCK_Q4_0_SIZE);
    let key: CacheKey = (data_ptr, src.len(), ne1 as usize, ne0 as usize, fingerprint(src));
    {
        let map = cache().lock().unwrap();
        if let Some(v) = map.get(&key) {
            return Some(v.clone());
        }
        let used: usize = map.values().map(|v| v.len()).sum();
        if used + src.len() > cache_budget() {
            return None;
        }
    }
    let t_start = std::time::Instant::now();
    let out = Arc::new(repack_q4_0_8x8(src, nrows, ne0 as usize));
    note_materialized(out.len(), t_start.elapsed().as_micros() as u64);
    let mut map = cache().lock().unwrap();
    Some(map.entry(key).or_insert(out).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack stats/counters are process-global (like the reference's
    /// CPU_REPACK buffer), so the tests that assert on them serialize here.
    /// Kernel-parity tests that merely *call* the kernels take the same lock
    /// because the gemv call counter is global too.
    pub(super) fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The SIMD kernels must equal the scalar ones bit-for-bit on *every* shape
    /// (the 16-row gemm body, the 4-row tail, ragged `nc`, and nr < 4).
    #[test]
    fn simd_kernels_match_scalar_bit_exact() {
        let _serial = serial();
        for &n in &[32usize, 64, 96, 2880] {
            for &nc in &[8usize, 16, 24, 40, 2880.min(8 * 45)] {
                let src = rand_mxfp4(nc, n, 0x1111_0000 + (n * 31 + nc) as u32);
                let rep = repack_mxfp4_8x8(&src, nc, n);
                let act = lcg_f32(n, 0x2222_0000 + (n * 17 + nc) as u32);
                let mut q8 = vec![0u8; (n / QK8_0) * BLOCK_Q8_0_SIZE];
                crate::quants::quantize_row_q8_0(&act, bytemuck::cast_slice_mut(&mut q8));
                let (mut a, mut b) = (vec![f32::NAN; nc], vec![f32::NAN; nc]);
                gemv_mxfp4_8x8_q8_0_scalar(n, &mut a, &rep, &q8, nc);
                gemv_mxfp4_8x8_q8_0(n, &mut b, &rep, &q8, nc);
                assert_eq!(a, b, "gemv n={n} nc={nc}");

                for &nr in &[4usize, 8, 12, 16, 20, 24, 32, 128] {
                    let act2 = lcg_f32(nr * n, 0x3333_0000 + (n * 13 + nr) as u32);
                    let mut q8x4 = vec![0u8; (nr / 4) * (n / QK8_0) * BLOCK_Q8_0X4_SIZE];
                    quantize_mat_q8_0_4x8(&act2, n, nr, &mut q8x4);
                    let bs = nc;
                    let (mut ga, mut gb) = (vec![f32::NAN; nr * bs], vec![f32::NAN; nr * bs]);
                    gemm_mxfp4_8x8_q8_0_scalar(n, &mut ga, bs, &rep, &q8x4, nr, nc);
                    gemm_mxfp4_8x8_q8_0(n, &mut gb, bs, &rep, &q8x4, nr, nc);
                    for r in 0..nr {
                        for c in 0..nc {
                            assert_eq!(
                                ga[r * bs + c].to_bits(),
                                gb[r * bs + c].to_bits(),
                                "gemm n={n} nc={nc} nr={nr} [{r},{c}]: {} vs {}",
                                ga[r * bs + c],
                                gb[r * bs + c]
                            );
                        }
                    }
                    assert!(gb.iter().all(|v| v.is_finite()), "gemm n={n} nc={nc} nr={nr}: non-finite");
                }
            }
        }
    }

    /// A/B of the two kernel implementations on the gpt-oss-20b expert shape
    /// (n = 2880, nc = 2880 = n_ff) — the decode hot path — plus the prefill
    /// gemm shapes. `cargo test --release -p ggml --lib bench_repack_kernels --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "manual: repack gemv/gemm A/B throughput"]
    fn bench_repack_kernels() {
        let _serial = serial();
        let (n, nrows) = (2880usize, 2880usize); // gpt-oss expert: 2880x2880
        let src = rand_mxfp4(nrows, n, 0x1234_9999);
        let rep = repack_mxfp4_8x8(&src, nrows, n);
        let act = lcg_f32(n, 0xabcd_1111);
        let mut q8 = vec![0u8; (n / QK8_0) * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(&act, bytemuck::cast_slice_mut(&mut q8));

        let time = |label: &str, f: &mut dyn FnMut() -> f32| {
            let mut best = f64::INFINITY;
            let mut sink = 0f32;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                sink = f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            // GB/s of weight traffic (136 B per 8-row tile per 32-element block)
            let bytes = nrows as f64 * (n / QK_MXFP4) as f64 * BLOCK_MXFP4X8_SIZE as f64;
            println!(
                "{label:>26}: {:8.3} ms  {:7.2} GB/s (weights)  [{sink:.4}]",
                best * 1e3,
                bytes / best / 1e9
            );
            best
        };

        let mut out = vec![0f32; nrows];
        let a = time("gemv scalar (nr=1)", &mut || {
            gemv_mxfp4_8x8_q8_0_scalar(n, &mut out, &rep, &q8, nrows);
            out.iter().copied().sum()
        });
        let b = time("gemv AVX2 (nr=1)", &mut || {
            gemv_mxfp4_8x8_q8_0(n, &mut out, &rep, &q8, nrows);
            out.iter().copied().sum()
        });
        println!("gemv speedup: {:.2}x", a / b);

        let nr = 128usize;
        let act2 = lcg_f32(nr * n, 0x5150_2222);
        let mut q8x4 = vec![0u8; (nr / 4) * (n / QK8_0) * BLOCK_Q8_0X4_SIZE];
        quantize_mat_q8_0_4x8(&act2, n, nr, &mut q8x4);
        let mut gout = vec![0f32; nr * nrows];
        let c = time("gemm scalar (nr=128)", &mut || {
            gemm_mxfp4_8x8_q8_0_scalar(n, &mut gout, nrows, &rep, &q8x4, nr, nrows);
            gout.iter().copied().sum()
        });
        let d = time("gemm AVX2 (nr=128)", &mut || {
            gemm_mxfp4_8x8_q8_0(n, &mut gout, nrows, &rep, &q8x4, nr, nrows);
            gout.iter().copied().sum()
        });
        println!("gemm speedup: {:.2}x", c / d);
        assert!(d < c, "the SIMD kernel must be faster");
    }

    /// Rough cost of materializing one gpt-oss expert tensor (141 MiB), for the
    /// report's "repack materialization is a one-time ~X s" claim.
    #[test]
    #[ignore = "manual: 141 MiB repack timing"]
    fn bench_repack_one_expert() {
        let (nrows, n_per_row) = (2880 * 32, 2880);
        let src = rand_mxfp4(nrows, n_per_row, 0x1234);
        let t = std::time::Instant::now();
        let out = repack_mxfp4_8x8(&src, nrows, n_per_row);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("repack {} MiB: {:.1} ms ({:.2} GB/s)", out.len() / 1024 / 1024, ms, out.len() as f64 / ms / 1e6);
        assert_eq!(out.len(), src.len());
    }

    fn first_diff(a: &[u8], b: &[u8]) -> Option<(usize, u8, u8)> {
        a.iter().zip(b).enumerate().find(|(_, (x, y))| x != y).map(|(i, (&x, &y))| (i, x, y))
    }

    fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn lcg_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as i32 as f32 / (1u32 << 28) as f32) * 1.5 - 0.75
            })
            .collect()
    }

    /// random but well-formed MXFP4 bytes: `e` spread over normals + denormals
    fn rand_mxfp4(nrows: usize, n_per_row: usize, seed: u32) -> Vec<u8> {
        let nb = n_per_row / QK_MXFP4;
        let mut s = seed;
        let mut out = vec![0u8; nrows * nb * BLOCK_MXFP4_SIZE];
        for (ib, blk) in out.chunks_mut(BLOCK_MXFP4_SIZE).enumerate() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            blk[0] = if ib % 16 == 15 { (s % 3) as u8 } else { 118 + (s % 16) as u8 };
            for q in &mut blk[1..] {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *q = (s >> 24) as u8;
            }
        }
        out
    }

    // ---- layout ----

    #[test]
    fn repack_roundtrip_identity() {
        for &(nrows, n_per_row) in &[(8usize, 32usize), (16, 96), (24, 64), (8, 2880)] {
            let src = rand_mxfp4(nrows, n_per_row, 0x1234_5678 ^ nrows as u32);
            let rep = repack_mxfp4_8x8(&src, nrows, n_per_row);
            assert_eq!(rep.len(), src.len(), "repack keeps the byte size");
            assert_ne!(rep, src, "repack must actually reorder (n_per_row >= 64)");
            let back = unrepack_mxfp4_8x8(&rep, nrows, n_per_row);
            assert_eq!(back, src, "unrepack(repack(x)) == x for {nrows}x{n_per_row}");
        }
    }

    #[test]
    fn repack_layout_is_row_interleave() {
        // hand check of make_block_mxfp4x8 against the C loops: 8 rows, 1 block
        let nrows = 8;
        let n_per_row = 32;
        let mut src = vec![0u8; nrows * BLOCK_MXFP4_SIZE];
        for r in 0..8 {
            src[r * BLOCK_MXFP4_SIZE] = 100 + r as u8; // e of row r
            for j in 0..16 {
                src[r * BLOCK_MXFP4_SIZE + 1 + j] = (r * 16 + j) as u8; // qs
            }
        }
        let rep = repack_mxfp4_8x8(&src, nrows, n_per_row);
        assert_eq!(&rep[..8], &[100, 101, 102, 103, 104, 105, 106, 107]);
        for i in 0..16 {
            // chunk i = row i%8, qs[(i/8)*8 .. +8]
            let expect: Vec<u8> = ((i % 8) * 16 + (i / 8) * 8..(i % 8) * 16 + (i / 8) * 8 + 8)
                .map(|v| v as u8)
                .collect();
            assert_eq!(&rep[8 + i * 8..8 + i * 8 + 8], &expect[..], "chunk {i}");
        }
    }

    #[test]
    fn repack_qualification_matches_c() {
        // repack.cpp:5093 — ne[1] % 8 == 0 (dim 1, not ggml_nrows)
        assert!(repack_supported_mxfp4(2880, 2880));
        assert!(repack_supported_mxfp4(8, 32));
        assert!(!repack_supported_mxfp4(4, 32));
        assert!(!repack_supported_mxfp4(16, 33)); // ne0 % 32 != 0
        assert!(!repack_supported_mxfp4(15, 32));
    }

    #[test]
    fn quantize_mat_4xn_matches_plain_q8_0_values() {
        let k = 64;
        let n_rows = 8;
        let x = lcg_f32(n_rows * k, 0xdead_beef);
        for interleave in [4usize, 8] {
            let mut out = vec![0u8; (n_rows / 4) * (k / QK8_0) * BLOCK_Q8_0X4_SIZE];
            if interleave == 4 {
                quantize_mat_q8_0_4x4(&x, k, n_rows, &mut out);
            } else {
                quantize_mat_q8_0_4x8(&x, k, n_rows, &mut out);
            }
            // plain q8_0 rows for comparison
            let mut q8 = vec![0u8; n_rows * (k / QK8_0) * BLOCK_Q8_0_SIZE];
            crate::quants::quantize_row_q8_0(&x, bytemuck::cast_slice_mut(&mut q8));
            for r in 0..n_rows {
                for ib in 0..k / QK8_0 {
                    let tile =
                        &out[(r / 4) * (k / QK8_0) * BLOCK_Q8_0X4_SIZE + ib * BLOCK_Q8_0X4_SIZE..];
                    assert_eq!(
                        f16_from_bytes(&tile[(r % 4) * 2..(r % 4) * 2 + 2]).to_f32(),
                        f16_from_bytes(&q8[(r * (k / QK8_0) + ib) * BLOCK_Q8_0_SIZE..][..2]).to_f32()
                    );
                    assert_eq!(deinterleave_q8_0x4_row(&tile[8..], r % 4, interleave), {
                        let src = &q8[(r * (k / QK8_0) + ib) * BLOCK_Q8_0_SIZE + 2..];
                        let mut a = [0i8; QK8_0];
                        for (j, v) in a.iter_mut().enumerate() {
                            *v = src[j] as i8;
                        }
                        a
                    });
                }
            }
        }
    }

    // ---- arithmetic ----

    /// Dequantized naive reference (f64) for the same products the kernel sums:
    /// `Σ_b (Σ_k LUT*w * q8) * (e_half * d)`. Tolerance-only companion to the
    /// exact C-dump parity test, so the kernel is checked even without the dump.
    fn naive_dot(src: &[u8], n_per_row: usize, wrow: usize, act: &[f32]) -> f64 {
        use crate::quants::dequantize_row_mxfp4;
        let nb = n_per_row / QK_MXFP4;
        let row = nb * BLOCK_MXFP4_SIZE;
        let blocks: &[crate::blocks::BlockMxfp4] =
            bytemuck::cast_slice(&src[wrow * row..wrow * row + nb * BLOCK_MXFP4_SIZE]);
        let mut w = vec![0f32; n_per_row];
        dequantize_row_mxfp4(blocks, &mut w);
        // q8_0 activations (the kernels' PARAM_TYPE)
        let mut q8 = vec![0u8; nb * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(act, bytemuck::cast_slice_mut(&mut q8));
        let qb: &[BlockQ8_0] = bytemuck::cast_slice(&q8);
        let mut sum = 0f64;
        for ib in 0..nb {
            let d = qb[ib].d.to_f32();
            for j in 0..QK_MXFP4 {
                sum += (w[ib * QK_MXFP4 + j] as f64) * (qb[ib].qs[j] as f64) * (d as f64);
            }
        }
        sum
    }

    #[test]
    fn gemv_matches_dequantized_reference() {
        let _serial = serial();
        let n_per_row = 96;
        let nrows = 16;
        let src = rand_mxfp4(nrows, n_per_row, 0x5eed_1234);
        let rep = repack_mxfp4_8x8(&src, nrows, n_per_row);
        let act = lcg_f32(n_per_row, 0xabcd_ef01);
        let mut q8 = vec![0u8; (n_per_row / QK8_0) * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(&act, bytemuck::cast_slice_mut(&mut q8));

        let mut s = vec![0f32; nrows];
        gemv_mxfp4_8x8_q8_0(n_per_row, &mut s, &rep, &q8, nrows);
        for r in 0..nrows {
            let want = naive_dot(&src, n_per_row, r, &act);
            assert!(
                (s[r] as f64 - want).abs() <= 1e-5 * want.abs().max(1.0),
                "row {r}: {} vs {}",
                s[r],
                want
            );
        }
    }

    #[test]
    fn gemm_matches_gemv_elementwise() {
        let _serial = serial();
        let n_per_row = 96;
        let nrows = 16;
        let nr = 8;
        let src = rand_mxfp4(nrows, n_per_row, 0x1357_9bdf);
        let rep = repack_mxfp4_8x8(&src, nrows, n_per_row);
        let act = lcg_f32(nr * n_per_row, 0x2468_ace0);
        let mut q8x4 = vec![0u8; (nr / 4) * (n_per_row / QK8_0) * BLOCK_Q8_0X4_SIZE];
        quantize_mat_q8_0_4x8(&act, n_per_row, nr, &mut q8x4);

        let mut out = vec![f32::NAN; nr * nrows];
        gemm_mxfp4_8x8_q8_0(n_per_row, &mut out, nrows, &rep, &q8x4, nr, nrows);
        for r in 0..nr {
            let mut q8 = vec![0u8; (n_per_row / QK8_0) * BLOCK_Q8_0_SIZE];
            crate::quants::quantize_row_q8_0(
                &act[r * n_per_row..(r + 1) * n_per_row],
                bytemuck::cast_slice_mut(&mut q8),
            );
            let mut sg = vec![0f32; nrows];
            gemv_mxfp4_8x8_q8_0(n_per_row, &mut sg, &rep, &q8, nrows);
            for c in 0..nrows {
                // both go through the same per-block fma, so this is bit-exact
                assert_eq!(
                    out[r * nrows + c].to_bits(),
                    sg[c].to_bits(),
                    "gemm({r},{c}) != gemv"
                );
            }
        }
    }

    // ---- reference dump parity (parity/ref_repack_dump.cpp) ----

    struct Dump {
        n_per_row: usize,
        nrows: usize,
        nr_gemv: usize,
        nr_gemm: usize,
        nc: usize,
        src: Vec<u8>,
        repacked: Vec<u8>,
        q8: Vec<u8>,
        q8x4: Vec<u8>,
        gemv: Vec<f32>,
        gemm: Vec<f32>,
        gemv_gen: Vec<f32>,
        gemm_gen: Vec<f32>,
        act_gemv: Vec<f32>,
        act_gemm: Vec<f32>,
    }

    fn read_dump() -> Option<Dump> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/repack_ref.bin");
        let buf = std::fs::read(path).ok()?;
        let mut o = 0usize;
        let mut next = |o: &mut usize| -> Vec<u8> {
            let len = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap()) as usize;
            *o += 4;
            let v = buf[*o..*o + len].to_vec();
            *o += len;
            v
        };
        let hdr = next(&mut o);
        let g = |i: usize| u32::from_le_bytes(hdr[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
        let f = |v: &[u8]| -> Vec<f32> {
            v.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let (src, repacked, q8, q8x4, gemv, gemm, gemv_gen, gemm_gen, act_gemv, act_gemm) = (
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
            next(&mut o),
        );
        Some(Dump {
            n_per_row: g(0),
            nrows: g(1),
            nr_gemv: g(2),
            nr_gemm: g(3),
            nc: g(4),
            src,
            repacked,
            q8,
            q8x4,
            gemv: f(&gemv),
            gemm: f(&gemm),
            gemv_gen: f(&gemv_gen),
            gemm_gen: f(&gemm_gen),
            act_gemv: f(&act_gemv),
            act_gemm: f(&act_gemm),
        })
    }

    /// Byte-for-byte layout parity against the reference's CPU_REPACK buffer
    /// (`ggml_backend_cpu_repack_buffer_type` + `set_tensor`).
    ///
    /// Regenerate with `parity/ref_repack_dump.cpp` (see its header) — the test
    /// skips when the dump is absent.
    #[test]
    fn layout_bytes_match_reference_dump() {
        let Some(d) = read_dump() else {
            eprintln!("skip: parity/ref_repack_ref.bin missing (build parity/ref_repack_dump.cpp)");
            return;
        };
        let mine = repack_mxfp4_8x8(&d.src, d.nrows, d.n_per_row);
        assert_eq!(mine.len(), d.repacked.len());
        if mine != d.repacked {
            let (i, _) = mine.iter().zip(&d.repacked).enumerate().find(|(_, (a, b))| a != b).unwrap();
            panic!(
                "repacked layout differs at byte {i} (mine {:#04x}, ref {:#04x}) of {}",
                mine[i],
                d.repacked[i],
                mine.len()
            );
        }
    }

    /// `ggml_gemv_mxfp4_8x8_q8_0` (AVX2/AVX512 kernel the reference runs) must
    /// match this port bit-for-bit.
    #[test]
    fn gemv_matches_reference_dump_bitexact() {
        let _serial = serial();
        let Some(d) = read_dump() else {
            eprintln!("skip: parity/repack_ref.bin missing");
            return;
        };
        assert_eq!(d.repacked, repack_mxfp4_8x8(&d.src, d.nrows, d.n_per_row), "layout first");
        // C's store index is `s + (y*nr + x*8)`; the dump used nr = 1
        assert_eq!(d.nr_gemv, 1, "dump must call gemv with nr = 1 (all C call sites do)");
        let sg_len = d.nr_gemv * d.nr_gemv + d.nc + 64;
        let mut mine = vec![f32::NAN; sg_len];
        gemv_mxfp4_8x8_q8_0(d.n_per_row, &mut mine, &d.repacked, &d.q8, d.nc);
        for x in 0..d.nc / 8 {
            for j in 0..8 {
                let i = x * 8 + j;
                assert_eq!(
                    mine[i].to_bits(),
                    d.gemv[i].to_bits(),
                    "gemv[{i}] (weight row {}) mine {} ref {}",
                    x * 8 + j,
                    mine[i],
                    d.gemv[i]
                );
            }
        }
    }

    /// `ggml_gemm_mxfp4_8x8_q8_0` with reference-quantized `block_q8_0x4`
    /// activations — bit-for-bit.
    #[test]
    fn gemm_matches_reference_dump_bitexact() {
        let Some(d) = read_dump() else {
            eprintln!("skip: parity/repack_ref.bin missing");
            return;
        };
        let bs = d.nc; // dump used bs = nc
        let mut mine = vec![f32::NAN; d.nr_gemm * bs + 64];
        gemm_mxfp4_8x8_q8_0(d.n_per_row, &mut mine, bs, &d.repacked, &d.q8x4, d.nr_gemm, d.nc);
        for r in 0..d.nr_gemm {
            for c in 0..d.nc {
                let i = r * bs + c;
                assert_eq!(
                    mine[i].to_bits(),
                    d.gemm[i].to_bits(),
                    "gemm[{r},{c}] mine {} ref {}",
                    mine[i],
                    d.gemm[i]
                );
            }
        }
    }

/// Both activation quantizers the reference's repack path uses must match
    /// bit-for-bit: `from_float` (= x86 `quantize_row_q8_0`, plain rows) for gemv
    /// and `ggml_quantize_mat_t<INTER_SIZE=8>` (= `ggml_quantize_mat_q8_0_4x8`)
    /// for gemm.
    #[test]
    fn activation_quantizers_match_reference_dump() {
        let Some(d) = read_dump() else {
            eprintln!("skip: parity/repack_ref.bin missing");
            return;
        };
        // plain q8_0 rows
        let nb = d.n_per_row / QK8_0;
        let mut q8 = vec![0u8; d.nr_gemv * nb * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(&d.act_gemv, bytemuck::cast_slice_mut(&mut q8));
        assert_eq!(
            q8,
            d.q8,
            "plain q8_0 activation bytes differ from the reference: {:?}",
            first_diff(&q8, &d.q8)
        );
        // 4x8-interleaved q8_0x4 rows (the gemm Lhs layout)
        let mut q8x4 = vec![0u8; (d.nr_gemm / 4) * nb * BLOCK_Q8_0X4_SIZE];
        quantize_mat_q8_0_4x8(&d.act_gemm, d.n_per_row, d.nr_gemm, &mut q8x4);
        assert_eq!(
            q8x4,
            d.q8x4,
            "block_q8_0x4 (4x8) activation bytes differ from the reference: {:?}",
            first_diff(&q8x4, &d.q8x4)
        );
    }

    #[test]
    fn generic_kernel_differs_from_avx_kernel() {
        let Some(d) = read_dump() else {
            eprintln!("skip: parity/repack_ref.bin missing");
            return;
        };
        let diff = (0..d.nc).filter(|&i| d.gemv[i].to_bits() != d.gemv_gen[i].to_bits()).count();
        eprintln!(
            "gemv AVX vs generic: {diff}/{} lanes differ (max |Δ| = {:.3e})",
            d.nc,
            (0..d.nc).map(|i| (d.gemv[i] - d.gemv_gen[i]).abs() as f64).fold(0.0, f64::max)
        );
        assert!(diff > 0, "expected the two C kernels to differ (dump captures both)");
    }

    // ---- graph-level wiring (compute.rs mul_mat/mul_mat_id repack branch) ----

    /// End-to-end through `graph_compute`: an MXFP4 3D expert tensor is the
    /// gpt-oss MoE shape that the reference repacks, so this must take the
    /// repack branch (asserted via `repack_stats`) and land within 1e-5 of a
    /// plain f64 reference built from `dequantize_row_mxfp4` + q8_0 activations.
    #[test]
    fn mul_mat_id_repack_wiring_matches_f64_reference() {
        let _serial = serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;

        let (n_embd, n_ff, n_experts, n_used, n_tok) = (64usize, 32usize, 4usize, 2usize, 3usize);
        let mut ctx = Context::new();
        let as_ = ctx.new_tensor_3d(GgmlType::Mxfp4, n_embd as i64, n_ff as i64, n_experts as i64);
        let b = ctx.new_tensor_3d(GgmlType::F32, n_embd as i64, n_used as i64, n_tok as i64);
        let ids = ctx.new_tensor_2d(GgmlType::I32, n_used as i64, n_tok as i64);
        let dst = ctx.mul_mat_id(as_, b, ids);
        for id in [as_, b, ids, dst] {
            ctx.arena_resize_tensor(id);
        }

        let wsrc = rand_mxfp4(n_ff * n_experts, n_embd, 0x1bad_c0de);
        ctx.data_bytes_mut(as_).unwrap().copy_from_slice(&wsrc);
        let bs = lcg_f32(n_embd * n_used * n_tok, 0xfeed_f00d);
        ctx.with_f32_mut(b, |p| p.copy_from_slice(&bs)).unwrap();
        // ids[slot, token] — deterministic experts, excercising the slot-index
        // src1 row selector (`i11 = slot`, C ggml-cpu.c:1510-1519)
        let idv: Vec<i32> = (0..n_used * n_tok).map(|i| ((i * 3 + 1) % n_experts) as i32).collect();
        ctx.with_i32_mut(ids, |p| p.copy_from_slice(&idv)).unwrap();

        let stats_before = repack_stats();
        let calls_before = repack_gemv_calls();
        let mut g = Graph::new(16);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 2);
        let stats_after = repack_stats();
        // (>= 1 and not == 1: other tests in this crate materialize copies
        // concurrently in the same process)
        assert!(
            stats_after.0 > stats_before.0,
            "the repack path was not taken for the 3D MXFP4 weight"
        );
        // 8 weight rows per gemv call x (n_ids * n_tok) activation rows
        assert_eq!(
            repack_gemv_calls() - calls_before,
            (n_ff / 8) * n_used * n_tok,
            "repack gemv was not called for every (slot, token) row"
        );
        assert!(stats_after.1 - stats_before.1 >= wsrc.len());

        // f64 reference: dequantized weights x q8_0-quantized activations
        let nb = n_embd / QK_MXFP4;
        let deq = {
            let blocks: &[crate::blocks::BlockMxfp4] = bytemuck::cast_slice(&wsrc);
            let mut v = vec![0f32; n_ff * n_experts * n_embd];
            crate::quants::dequantize_row_mxfp4(blocks, &mut v);
            v
        };
        let out = ctx.f32s(dst).unwrap().to_vec();
        for i2 in 0..n_tok {
            for i1 in 0..n_used {
                let expert = idv[i1 + i2 * n_used] as usize;
                let act = &bs[(i1 + i2 * n_used) * n_embd..][..n_embd];
                let mut q8 = vec![0u8; nb * BLOCK_Q8_0_SIZE];
                crate::quants::quantize_row_q8_0(act, bytemuck::cast_slice_mut(&mut q8));
                let qb: &[BlockQ8_0] = bytemuck::cast_slice(&q8);
                for i0 in 0..n_ff {
                    let mut want = 0f64;
                    for ib in 0..nb {
                        let d = qb[ib].d.to_f32() as f64;
                        for j in 0..QK_MXFP4 {
                            let k = ib * QK_MXFP4 + j;
                            want += (deq[(expert * n_ff + i0) * n_embd + k] as f64)
                                * (qb[ib].qs[j] as f64)
                                * d;
                        }
                    }
                    let got = out[i0 + i1 * n_ff + i2 * n_ff * n_used] as f64;
                    assert!(
                        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "dst[{i0},{i1},{i2}] = {got} vs f64 {want}"
                    );
                }
            }
        }
    }

    /// 2D MXFP4 `mul_mat` through `graph_compute` — C would run
    /// `gemm` for the first `nrows - nrows%4` activation rows and `gemv` for the
    /// tail (repack.cpp:4638-4647); the port walks every row through gemv, which
    /// is the same arithmetic per element (see `gemm_matches_gemv_elementwise`).
    #[test]
    fn mul_mat_repack_wiring_matches_f64_reference() {
        let _serial = serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;

        let (n_per_row, n_rows, n_act) = (64usize, 16usize, 6usize);
        let mut ctx = Context::new();
        let wgt = ctx.new_tensor_2d(GgmlType::Mxfp4, n_per_row as i64, n_rows as i64);
        let act = ctx.new_tensor_2d(GgmlType::F32, n_per_row as i64, n_act as i64);
        let dst = ctx.mul_mat(wgt, act);
        for id in [wgt, act, dst] {
            ctx.arena_resize_tensor(id);
        }
        let wsrc = rand_mxfp4(n_rows, n_per_row, 0x0d15_ea5e);
        ctx.data_bytes_mut(wgt).unwrap().copy_from_slice(&wsrc);
        let a = lcg_f32(n_per_row * n_act, 0xc0ff_ee00);
        ctx.with_f32_mut(act, |p| p.copy_from_slice(&a)).unwrap();

        let stats_before = repack_stats();
        let calls_before = repack_gemv_calls();
        let mut g = Graph::new(8);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 3);
        assert!(
            repack_stats().0 > stats_before.0,
            "the repack path was not taken for the 2D MXFP4 weight"
        );
        assert_eq!(
            repack_gemv_calls() - calls_before,
            (n_rows / 8) * n_act,
            "2D mul_mat must run one gemv per 8 weight rows per activation row"
        );

        let nb = n_per_row / QK_MXFP4;
        let out = ctx.f32s(dst).unwrap().to_vec();
        for col in 0..n_act {
            let mut q8 = vec![0u8; nb * BLOCK_Q8_0_SIZE];
            crate::quants::quantize_row_q8_0(&a[col * n_per_row..][..n_per_row], bytemuck::cast_slice_mut(&mut q8));
            let qb: &[BlockQ8_0] = bytemuck::cast_slice(&q8);
            for row in 0..n_rows {
                let want = naive_dot(&wsrc, n_per_row, row, &a[col * n_per_row..][..n_per_row]);
                let got = out[row + col * n_rows] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "dst[{row},{col}] = {got} vs f64 {want}"
                );
                let _ = qb;
            }
        }
    }

    /// 3D Q4_K `mul_mat_id` through `graph_compute` — the MoE expert shape the
    /// reference repacks on x86 (`ggml_repack_get_optimal_repack_type` looks at
    /// the *per-expert* `ne[1]`, repack.cpp:5006-5011; LFM2-8B-A1B Q4_K_M has 56
    /// such tensors): must take the repack branch (witnessed by
    /// `repack_q4k_gemv_calls`) and land within 1e-5 of a plain f64 reference
    /// built from `dequantize_row_q4_K` x `quantize_row_q8_K_ref` activations.
    #[test]
    fn mul_mat_id_q4k_repack_wiring_matches_f64_reference() {
        let _serial = serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;
        use bytemuck::Zeroable;

        let (n_embd, n_ff, n_experts, n_used, n_tok) = (512usize, 16usize, 4usize, 2usize, 3usize);
        let mut ctx = Context::new();
        let as_ = ctx.new_tensor_3d(GgmlType::Q4K, n_embd as i64, n_ff as i64, n_experts as i64);
        let b = ctx.new_tensor_3d(GgmlType::F32, n_embd as i64, n_used as i64, n_tok as i64);
        let ids = ctx.new_tensor_2d(GgmlType::I32, n_used as i64, n_tok as i64);
        let dst = ctx.mul_mat_id(as_, b, ids);
        for id in [as_, b, ids, dst] {
            ctx.arena_resize_tensor(id);
        }

        // well-formed Q4_K weights: LCG rows through the reference quantizer
        let mut st = 0x1bad_c0deu32;
        let mut lcg = move || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (st as i32 as f32 / (1u32 << 28) as f32) * 1.5 - 0.75
        };
        let wsrc_f: Vec<f32> = (0..n_ff * n_experts * n_embd).map(|_| lcg()).collect();
        let mut wsrc = vec![0u8; n_ff * n_experts * (n_embd / QK_K) * BLOCK_Q4_K_SIZE];
        for r in 0..n_ff * n_experts {
            let mut blocks = vec![crate::blocks::BlockQ4K::zeroed(); n_embd / QK_K];
            crate::quants_k::quantize_row_q4_K_ref(&wsrc_f[r * n_embd..][..n_embd], &mut blocks);
            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
            wsrc[r * bytes.len()..][..bytes.len()].copy_from_slice(bytes);
        }
        ctx.data_bytes_mut(as_).unwrap().copy_from_slice(&wsrc);
        let bs: Vec<f32> = (0..n_embd * n_used * n_tok).map(|_| lcg()).collect();
        ctx.with_f32_mut(b, |p| p.copy_from_slice(&bs)).unwrap();
        // ids[slot, token] — deterministic experts
        let idv: Vec<i32> = (0..n_used * n_tok).map(|i| ((i * 3 + 1) % n_experts) as i32).collect();
        ctx.with_i32_mut(ids, |p| p.copy_from_slice(&idv)).unwrap();

        let stats_before = repack_stats();
        let calls_before = repack_q4k_gemv_calls();
        let mut g = Graph::new(16);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 2);
        assert!(
            repack_stats().0 > stats_before.0,
            "the repack path was not taken for the 3D Q4_K weight"
        );
        assert_eq!(
            repack_q4k_gemv_calls() - calls_before,
            (n_ff / 8) * n_used * n_tok,
            "one q4_K 8x8 gemv per 8 weight rows per (slot, token) row"
        );

        // f64 reference: dequantized weights x q8_K-quantized activations
        let nb = n_embd / QK_K;
        let out = ctx.f32s(dst).unwrap().to_vec();
        for i2 in 0..n_tok {
            for i1 in 0..n_used {
                let expert = idv[i1 + i2 * n_used] as usize;
                let act = &bs[(i1 + i2 * n_used) * n_embd..][..n_embd];
                let mut qb = vec![crate::blocks::BlockQ8K::zeroed(); nb];
                crate::quants::quantize_row_q8_K_ref(act, &mut qb);
                for i0 in 0..n_ff {
                    let blocks: &[crate::blocks::BlockQ4K] = bytemuck::cast_slice(
                        &wsrc[(expert * n_ff + i0) * nb * BLOCK_Q4_K_SIZE..][..nb * BLOCK_Q4_K_SIZE],
                    );
                    let mut w = vec![0f32; n_embd];
                    crate::quants::dequantize_row_q4_K(blocks, &mut w);
                    let mut want = 0f64;
                    for ib in 0..nb {
                        let d = qb[ib].d as f64;
                        for j in 0..QK_K {
                            want += w[ib * QK_K + j] as f64 * qb[ib].qs[j] as f64 * d;
                        }
                    }
                    let got = out[i0 + i1 * n_ff + i2 * n_ff * n_used] as f64;
                    // 1e-4, not 1e-5: the kernel accumulates the value and min
                    // terms separately in f32 (`acc - accmin`, each with one
                    // fma per super block) so the naive f64 sum lands ~1e-5
                    // relative off after that subtraction's cancellation — the
                    // *bit-exact* check is `q4k_ref_tests`, not this test.
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "dst[{i0},{i1},{i2}] = {got} vs f64 {want}"
                    );
                }
            }
        }
    }

    /// 2D Q4_K `mul_mat` with a src1 that has **more planes than the weight**
    /// (`ne12 > ne02`, the GQA broadcast): each dst plane must read its own
    /// weight plane selection `i02 = i12 / r2` (repack.cpp:4622-4628). Before
    /// that was wired the repack branch used plane 0's weights everywhere.
    #[test]
    fn mul_mat_q4k_repack_broadcast_planes_match_f64_reference() {
        let _serial = serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;
        use bytemuck::Zeroable;

        let (n_per_row, n_rows, n_act, n_planes) = (256usize, 8usize, 4usize, 3usize);
        let mut ctx = Context::new();
        let wgt = ctx.new_tensor_2d(GgmlType::Q4K, n_per_row as i64, n_rows as i64);
        let act = ctx.new_tensor_3d(GgmlType::F32, n_per_row as i64, n_act as i64, n_planes as i64);
        let dst = ctx.mul_mat(wgt, act);
        for id in [wgt, act, dst] {
            ctx.arena_resize_tensor(id);
        }

        let mut st = 0x0b12_3456u32;
        let mut lcg = move || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (st as i32 as f32 / (1u32 << 28) as f32) * 1.5 - 0.75
        };
        let wf: Vec<f32> = (0..n_rows * n_per_row).map(|_| lcg()).collect();
        let mut wsrc = vec![0u8; n_rows * (n_per_row / QK_K) * BLOCK_Q4_K_SIZE];
        let mut blocks = vec![crate::blocks::BlockQ4K::zeroed(); n_per_row / QK_K];
        for r in 0..n_rows {
            crate::quants_k::quantize_row_q4_K_ref(&wf[r * n_per_row..][..n_per_row], &mut blocks);
            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
            wsrc[r * bytes.len()..][..bytes.len()].copy_from_slice(bytes);
        }
        ctx.data_bytes_mut(wgt).unwrap().copy_from_slice(&wsrc);
        let a: Vec<f32> = (0..n_per_row * n_act * n_planes).map(|_| lcg()).collect();
        ctx.with_f32_mut(act, |p| p.copy_from_slice(&a)).unwrap();

        let mut g = Graph::new(8);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 2);

        // f64 reference per (plane, activation row, weight row)
        let nb = n_per_row / QK_K;
        let out = ctx.f32s(dst).unwrap().to_vec();
        for p in 0..n_planes {
            for c in 0..n_act {
                let mut qb = vec![crate::blocks::BlockQ8K::zeroed(); nb];
                crate::quants::quantize_row_q8_K_ref(&a[(p * n_act + c) * n_per_row..][..n_per_row], &mut qb);
                for r in 0..n_rows {
                    let blocks: &[crate::blocks::BlockQ4K] =
                        bytemuck::cast_slice(&wsrc[r * nb * BLOCK_Q4_K_SIZE..][..nb * BLOCK_Q4_K_SIZE]);
                    let mut w = vec![0f32; n_per_row];
                    crate::quants::dequantize_row_q4_K(blocks, &mut w);
                    let mut want = 0f64;
                    for ib in 0..nb {
                        let d = qb[ib].d as f64;
                        for j in 0..QK_K {
                            want += w[ib * QK_K + j] as f64 * qb[ib].qs[j] as f64 * d;
                        }
                    }
                    let got = out[r + c * n_rows + p * n_rows * n_act] as f64;
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "dst[{r},{c},{p}] = {got} vs f64 {want}"
                    );
                }
            }
        }
    }

    /// Tensors the reference would not repack (`ne[1] % 8 != 0`) and tensors that
    /// are not plain row arrays must fall back to the row-wise vec_dot.
    #[test]
    fn mul_mat_id_repack_skips_unqualified_tensor() {
        let _serial = serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;

        let mut ctx = Context::new();
        // 12 % 8 != 0 -> ggml_repack_get_optimal_repack_type returns nullptr
        let as_ = ctx.new_tensor_3d(GgmlType::Mxfp4, 64, 12, 2);
        let b = ctx.new_tensor_3d(GgmlType::F32, 64, 1, 1);
        let ids = ctx.new_tensor_2d(GgmlType::I32, 1, 1);
        let dst = ctx.mul_mat_id(as_, b, ids);
        for id in [as_, b, ids, dst] {
            ctx.arena_resize_tensor(id);
        }
        let wsrc = rand_mxfp4(12 * 2, 64, 0x5a5a_1234);
        ctx.data_bytes_mut(as_).unwrap().copy_from_slice(&wsrc);
        ctx.with_f32_mut(b, |p| p.fill(0.25)).unwrap();
        ctx.with_i32_mut(ids, |p| p[0] = 1).unwrap();

        let stats_before = repack_stats();
        let calls_before = repack_gemv_calls();
        let mut g = Graph::new(8);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 1);
        assert_eq!(
            repack_stats(),
            stats_before,
            "an unqualified tensor must not be repacked (fallback path)"
        );
        assert_eq!(repack_gemv_calls(), calls_before, "fallback must not call the 8x8 kernel");
        assert!(ctx.f32s(dst).unwrap().iter().any(|v| *v != 0.0));
    }

    /// The two f32 accumulation orders the port can take are:
    ///   * repack gemv/gemm (this file): `acc = fma(int_dot, e8m0_half*row_d, acc)`
    ///   * row-wise `vec_dot_mxfp4_q8_0` (ggml-cpu/quants.c:298):
    ///     `acc += (e8m0_half*row_d) * int_dot` (two roundings per block)
    ///
    /// On MXFP4 x Q8_0 they coincide: `int_dot` fits in <=16 bits and the scale
    /// (E8M0 power of two x fp16) carries <=11, so the product needs <=27 bits and
    /// its rounding error stays below the accumulator's ulp. Measured 0/368,640
    /// random rows here and **0/16x201088 logit differences** in the real
    /// gpt-oss-20b run (LLAMA_RUST_REPACK=0 vs 1 logits dumps compared
    /// bit-for-bit) — i.e. this particular accumulation difference is *not* the
    /// cause of the port's residual logit gap vs the reference (PARITY.md).
    #[test]
    fn fma_accumulation_vs_mul_add_coincide_on_mxfp4() {
        let _serial = serial();
        let (n_per_row, nrows) = (2880usize, 4096usize);
        let nb = n_per_row / QK_MXFP4;
        let src = rand_mxfp4(nrows, n_per_row, 0xfeed_beef);
        let rep = repack_mxfp4_8x8(&src, nrows, n_per_row);
        let act = lcg_f32(n_per_row, 0x600d_f00d);
        let mut q8 = vec![0u8; nb * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(&act, bytemuck::cast_slice_mut(&mut q8));

        let mut fma_out = vec![0f32; nrows];
        gemv_mxfp4_8x8_q8_0(n_per_row, &mut fma_out, &rep, &q8, nrows);

        // same integer dots, mul-then-add accumulation (the row-wise vec_dot order)
        let qb: &[BlockQ8_0] = bytemuck::cast_slice(&q8);
        let mut diffs = 0;
        for row in 0..nrows {
            let mut acc = 0f32;
            for b in 0..nb {
                let tile = &rep[(row / 8) * nb * BLOCK_MXFP4X8_SIZE + b * BLOCK_MXFP4X8_SIZE..];
                let mut dot = 0i32;
                let lo = &tile[8 + 8 * (row % 8)..];
                let hi = &tile[72 + 8 * (row % 8)..];
                for t in 0..8 {
                    dot += KVALUES_MXFP4[(lo[t] & 0x0F) as usize] as i32 * qb[b].qs[t] as i32;
                    dot += KVALUES_MXFP4[(lo[t] >> 4) as usize] as i32 * qb[b].qs[16 + t] as i32;
                    dot += KVALUES_MXFP4[(hi[t] & 0x0F) as usize] as i32 * qb[b].qs[8 + t] as i32;
                    dot += KVALUES_MXFP4[(hi[t] >> 4) as usize] as i32 * qb[b].qs[24 + t] as i32;
                }
                let d = e8m0_to_fp32_half(tile[row % 8]) * qb[b].d.to_f32();
                acc += d * dot as f32; // mul, then add (two roundings)
            }
            if acc.to_bits() != fma_out[row].to_bits() {
                diffs += 1;
            }
        }
        assert_eq!(
            diffs, 0,
            "fma vs mul+add disagreed on {diffs}/{nrows} rows — if this fires, the \
             claim in the doc comment (and the e2e bit-identity) needs revisiting"
        );
    }

    /// The same tensor twice must reuse one cached copy (the reference pays for
    /// its CPU_REPACK buffer once at load time; ours is lazy but equally stable).
    #[test]
    fn repack_cache_reuses_copy() {
        let _serial = serial();
        let src = rand_mxfp4(16, 64, 0x0bad_f00d);
        let a = repack_mxfp4_8x8_cached(src.as_ptr() as usize, &src, 16, 64).unwrap();
        let (n1, b1) = repack_stats();
        let b = repack_mxfp4_8x8_cached(src.as_ptr() as usize, &src, 16, 64).unwrap();
        let (n2, b2) = repack_stats();
        assert!(Arc::ptr_eq(&a, &b), "cache miss on the second lookup");
        assert_eq!((n1, b1), (n2, b2), "no new bytes may be materialized");
    }
}
/// Reference-dump parity for the Q4_K 8x8 path
/// (`parity/ref_repack_kdump.cpp` -> `parity/q4k_repack_ref.bin`).
#[cfg(test)]
mod q4k_ref_tests {
    use super::*;
    use bytemuck::Zeroable;

    struct Dump {
        n: usize,
        nrows: usize,
        nr_gemv: usize,
        nr_gemm: usize,
        nc: usize,
        src: Vec<u8>,
        repacked: Vec<u8>,
        q8k: Vec<u8>,
        q8kx4: Vec<u8>,
        gemv_avx: Vec<f32>,
        gemm_avx: Vec<f32>,
        gemv_gen: Vec<f32>,
        gemm_gen: Vec<f32>,
        act_g: Vec<f32>,
        act_m: Vec<f32>,
        trait_present: u32,
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect()
    }

    fn parse(bytes: &[u8]) -> Dump {
        let mut c = bytes;
        let mut section = |c: &mut &[u8]| -> Vec<u8> {
            let len = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            *c = &c[4..];
            let (s, rest) = c.split_at(len);
            *c = rest;
            s.to_vec()
        };
        let hdr = section(&mut c);
        let u = |i: usize| u32::from_le_bytes(hdr[4 * i..4 * i + 4].try_into().unwrap());
        let (n, nrows, nr_gemv, nr_gemm, nc) =
            (u(1) as usize, u(2) as usize, u(3) as usize, u(4) as usize, u(5) as usize);
        let src = section(&mut c);
        let repacked = section(&mut c);
        let q8k = section(&mut c);
        let q8kx4 = section(&mut c);
        let gemv_avx = f32s(&section(&mut c));
        let gemm_avx = f32s(&section(&mut c));
        let gemv_gen = f32s(&section(&mut c));
        let gemm_gen = f32s(&section(&mut c));
        let act_g = f32s(&section(&mut c));
        let act_m = f32s(&section(&mut c));
        let tr = section(&mut c);
        Dump {
            n,
            nrows,
            nr_gemv,
            nr_gemm,
            nc,
            src,
            repacked,
            q8k,
            q8kx4,
            gemv_avx,
            gemm_avx,
            gemv_gen,
            gemm_gen,
            act_g,
            act_m,
            trait_present: u32::from_le_bytes(tr[0..4].try_into().unwrap()),
        }
    }

    fn load() -> Dump {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/q4k_repack_ref.bin");
        let bytes = std::fs::read(path).unwrap_or_else(|_| {
            panic!("missing {path}: build via parity/ref_repack_kdump.cpp (see its header)")
        });
        parse(&bytes)
    }

    /// The reference build on this host must route Q4_K to the 8x8 traits; if
    /// this fails the whole routing table in the report is void.
    #[test]
    fn reference_q4k_trait_is_the_8x8_repack() {
        let d = load();
        assert_eq!(d.trait_present, 1, "CPU_REPACK gave Q4_K no trait (repack.cpp:5006)");
    }

    /// The port's `make_block_q4_Kx8`/`repack_q4_K_8x8_into` must reproduce the
    /// reference's CPU_REPACK buffer byte for byte.
    #[test]
    fn repack_layout_matches_reference_bytes() {
        let d = load();
        let mine = repack_q4_K_8x8(&d.src, d.nrows, d.n);
        assert_eq!(mine.len(), d.repacked.len(), "repacked size");
        if mine != d.repacked {
            let i = mine.iter().zip(&d.repacked).position(|(a, b)| a != b).unwrap();
            let w = |s: &[u8]| s[..(i + 16).min(s.len())].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
            panic!("repacked bytes differ at {i} of {}\nmine: {}\nref : {}",
                   mine.len(), w(&mine), w(&d.repacked));
        }
        // and the round trip
        let back = unrepack_q4_K_8x8(&mine, d.nrows, d.n);
        assert_eq!(back, d.src, "unrepack round trip");
    }

    /// The port's activation quantizer must agree with the reference's
    /// `ggml_quantize_mat_q8_K_4x8` on the dequantized values (`d*q` per
    /// element) — the sign convention may differ (see the function doc).
    #[test]
    fn q4k_interleaved_activation_matches_reference() {
        let d = load();
        let nb = d.n / QK_K;
        let rows = d.nr_gemm;
        let mut mine = vec![0u8; (rows / 4) * nb * BLOCK_Q8_KX4_SIZE];
        quantize_mat_q8_K_4x8(&d.act_m[..rows * d.n], d.n, rows, &mut mine);
        assert_eq!(mine.len(), d.q8kx4.len());
        for ib in 0..nb * (rows / 4) {
            let a = &mine[ib * BLOCK_Q8_KX4_SIZE..];
            let b = &d.q8kx4[ib * BLOCK_Q8_KX4_SIZE..];
            for r in 0..4 {
                let da = f32::from_le_bytes(a[4 * r..4 * r + 4].try_into().unwrap());
                let db = f32::from_le_bytes(b[4 * r..4 * r + 4].try_into().unwrap());
                assert_eq!(da.abs(), db.abs(), "q8_Kx4 d magnitude (row {r}, block {ib})");
                for el in 0..QK_K {
                    let qa = a[16 + (el / 8) * 32 + r * 8 + el % 8] as i8 as i32;
                    let qb = b[16 + (el / 8) * 32 + r * 8 + el % 8] as i8 as i32;
                    assert_eq!(
                        (da * qa as f32).to_bits(),
                        (db * qb as f32).to_bits(),
                        "dequantized activation differs (row {r}, element {el}, block {ib})"
                    );
                }
                // bsums are the sums of the stored quants; with the possible
                // global sign flip they can differ in sign only
                for j in 0..16 {
                    let sa = i16::from_le_bytes([a[1040 + 2 * j], a[1041 + 2 * j]]);
                    let sb = i16::from_le_bytes([b[1040 + 2 * j], b[1041 + 2 * j]]);
                    assert_eq!(sa.abs(), sb.abs(), "bsums differ (row {r}, j {j}, block {ib})");
                }
            }
        }
    }

    /// The port's plain `q8_K` rows (the gemv's LHS) must equal the reference's
    /// `from_float` bytes — the same quantizer `vec_dot` already matches.
    #[test]
    fn q4k_gemv_activation_bytes_match_reference() {
        let d = load();
        let nb = d.n / QK_K;
        let mut mine = vec![0u8; d.nr_gemv * nb * BLOCK_Q8_K_SIZE];
        for r in 0..d.nr_gemv {
            let row = &d.act_g[r * d.n..(r + 1) * d.n];
            let mut blocks = vec![crate::blocks::BlockQ8K::zeroed(); nb];
            crate::quants::quantize_row_q8_K_ref(row, &mut blocks);
            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
            mine[r * bytes.len()..(r + 1) * bytes.len()].copy_from_slice(bytes);
        }
        assert_eq!(mine, d.q8k);
    }

    fn run_gemv(d: &Dump, repacked: &[u8], q8k: &[u8]) -> Vec<f32> {
        let out_len = d.nr_gemv * d.nr_gemv + d.nc + 64;
        let mut out = vec![1234.5f32; out_len];
        for r in 0..d.nr_gemv {
            gemv_q4_K_8x8_q8_K(
                d.n,
                &mut out[r * d.nr_gemv..],
                repacked,
                &q8k[r * (d.n / QK_K) * BLOCK_Q8_K_SIZE..],
                d.nc,
            );
        }
        out
    }

    fn run_gemm(d: &Dump, repacked: &[u8], q8kx4: &[u8]) -> Vec<f32> {
        let mut out = vec![1234.5f32; d.nr_gemm * d.nc + 64];
        gemm_q4_K_8x8_q8_K(d.n, &mut out, d.nc, repacked, q8kx4, d.nr_gemm, d.nc);
        out
    }

    /// The kernels must reproduce the reference's AVX bodies bit for bit on
    /// every value the dump covers (and not the `_generic` ones, which use a
    /// different rounding chain — asserted so the test cannot pass by accident).
    #[test]
    fn q4k_kernels_match_reference_avx_bit_exact() {
        let _serial = super::tests::serial();
        let d = load();
        let mine_g = run_gemv(&d, &d.repacked, &d.q8k);
        let mine_m = run_gemm(&d, &d.repacked, &d.q8kx4);
        let bad_g: Vec<usize> = (0..d.gemv_avx.len())
            .filter(|&i| mine_g[i].to_bits() != d.gemv_avx[i].to_bits())
            .collect();
        let bad_m: Vec<usize> = (0..d.gemm_avx.len())
            .filter(|&i| mine_m[i].to_bits() != d.gemm_avx[i].to_bits())
            .collect();
        assert!(
            bad_g.is_empty(),
            "gemv differs from the reference AVX body at {} of {} slots, first {:?} \
             (mine {:?} vs ref {:?}); *_generic agreement: {}",
            bad_g.len(),
            d.gemv_avx.len(),
            bad_g.first(),
            bad_g.first().map(|&i| mine_g[i]),
            bad_g.first().map(|&i| d.gemv_avx[i]),
            mine_g.iter().zip(&d.gemv_gen).all(|(a, b)| a.to_bits() == b.to_bits()),
        );
        assert!(
            bad_m.is_empty(),
            "gemm differs from the reference AVX body at {} of {} slots, first {:?} \
             (mine {:?} vs ref {:?})",
            bad_m.len(),
            d.gemm_avx.len(),
            bad_m.first(),
            bad_m.first().map(|&i| mine_m[i]),
            bad_m.first().map(|&i| d.gemm_avx[i]),
        );
        // the two reference bodies really do differ here, so the assertion above
        // is a statement about the AVX body, not about both
        let n_gen_diff = d
            .gemv_avx
            .iter()
            .zip(&d.gemv_gen)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "q4_K 8x8: {} gemv + {} gemm values bit-exact vs the reference AVX body; \
             reference AVX-vs-generic mismatches: gemv {n_gen_diff}, gemm {}",
            d.gemv_avx.len() - bad_g.len(),
            d.gemm_avx.len() - bad_m.len(),
            d.gemm_avx.iter().zip(&d.gemm_gen).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
        );
    }

    /// SIMD == scalar on the dump's own data (both must be the AVX body's
    /// arithmetic; `LLAMA_RUST_REPACK_SIMD=0` only switches speed).
    #[test]
    fn q4k_simd_matches_scalar_bit_exact() {
        let _serial = super::tests::serial();
        let d = load();
        if std::env::var_os("LLAMA_RUST_Q4K_DEBUG").is_some() {
            let mut s_m = vec![0f32; d.nr_gemm * d.nc];
            gemm_q4_K_8x8_q8_K_scalar(d.n, &mut s_m, d.nc, &d.repacked, &d.q8kx4, d.nr_gemm, d.nc);
            let mut v_m = vec![0f32; d.nr_gemm * d.nc];
            gemm_q4_K_8x8_q8_K(d.n, &mut v_m, d.nc, &d.repacked, &d.q8kx4, d.nr_gemm, d.nc);
            for i in 0..d.nr_gemm * d.nc {
                if s_m[i].to_bits() != v_m[i].to_bits() || v_m[i].to_bits() != d.gemm_avx[i].to_bits()
                {
                    println!(
                        "gemm ({}, {}): scalar {:?} simd {:?} ref {:?}",
                        i / d.nc, i % d.nc, s_m[i], v_m[i], d.gemm_avx[i]
                    );
                }
            }
        }
        let mut simd_g = vec![0f32; d.nc];
        gemv_q4_K_8x8_q8_K(d.n, &mut simd_g, &d.repacked, &d.q8k, d.nc);
        assert_eq!(super::repack_q4k_gemv_calls() >= 1, super::simd_enabled());
        let mut scal_g = vec![0f32; d.nc];
        gemv_q4_K_8x8_q8_K_scalar(d.n, &mut scal_g, &d.repacked, &d.q8k, d.nc);
        for i in 0..d.nc {
            assert_eq!(simd_g[i].to_bits(), scal_g[i].to_bits(), "gemv value {i}");
        }
        let mut simd_m = vec![0f32; d.nr_gemm * d.nc];
        gemm_q4_K_8x8_q8_K(d.n, &mut simd_m, d.nc, &d.repacked, &d.q8kx4, d.nr_gemm, d.nc);
        let mut scal_m = vec![0f32; d.nr_gemm * d.nc];
        gemm_q4_K_8x8_q8_K_scalar(d.n, &mut scal_m, d.nc, &d.repacked, &d.q8kx4, d.nr_gemm, d.nc);
        for i in 0..d.nr_gemm * d.nc {
            assert_eq!(simd_m[i].to_bits(), scal_m[i].to_bits(), "gemm value {i}");
        }
    }

    /// Random Q4_K weights + activations over a shape grid, shared by the two
    /// shape-grid tests below: `nc` sweeps both `nc % 16` residues so the
    /// AVX512 section's full-matrix and tail-columns hand-off are both
    /// covered, and `nr` sweeps the 16-row groups, the 4-row tail and odd
    /// remainders.
    fn shape_grid_case(
        lcg: &mut impl FnMut() -> f32,
        n: usize,
        nc: usize,
        nr: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let nb = n / QK_K;
        let mut wf = vec![0f32; nc * n];
        for v in wf.iter_mut() {
            *v = lcg();
        }
        let mut wsrc = vec![0u8; nc * nb * BLOCK_Q4_K_SIZE];
        let mut blocks = vec![crate::blocks::BlockQ4K::zeroed(); nb];
        for r in 0..nc {
            crate::quants_k::quantize_row_q4_K_ref(&wf[r * n..][..n], &mut blocks);
            let b: &[u8] = bytemuck::cast_slice(&blocks);
            wsrc[r * nb * BLOCK_Q4_K_SIZE..][..nb * BLOCK_Q4_K_SIZE].copy_from_slice(b);
        }
        let rep = repack_q4_K_8x8(&wsrc, nc, n);
        let act: Vec<f32> = (0..nr * n).map(|_| lcg()).collect();
        let mut q8kx4 = vec![0u8; (nr / 4) * nb * BLOCK_Q8_KX4_SIZE];
        quantize_mat_q8_K_4x8(&act, n, nr, &mut q8kx4);
        (rep, q8kx4)
    }

    /// SIMD == scalar across a shape grid, so every loop boundary of the
    /// gemm's lane sections is covered on any given host: on an AVX512BW+DQ
    /// host `nc % 16 == 0` shapes run the AVX512 section (16-row groups and
    /// its 4-row tail) while `nc % 16 == 8` shapes also cross into the AVX2
    /// tail-columns section (`xstart == anc/8`, arch/x86/repack.cpp:2811);
    /// an AVX2-only host runs the AVX2 section for all of them.
    #[test]
    fn q4k_simd_matches_scalar_shape_grid() {
        let _serial = super::tests::serial();
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let mut lcg = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as i32 as f32 / (1u32 << 31) as f32 * 1.5 - 0.75
        };
        for &n in &[256usize, 512, 768] {
            for &nc in &[8usize, 16, 24, 32, 40] {
                for &nr in &[4usize, 8, 16, 20, 36, 64] {
                    let (rep, q8kx4) = shape_grid_case(&mut lcg, n, nc, nr);
                    let mut simd_m = vec![0f32; nr * nc];
                    gemm_q4_K_8x8_q8_K(n, &mut simd_m, nc, &rep, &q8kx4, nr, nc);
                    let mut scal_m = vec![0f32; nr * nc];
                    gemm_q4_K_8x8_q8_K_scalar(n, &mut scal_m, nc, &rep, &q8kx4, nr, nc);
                    for i in 0..nr * nc {
                        assert_eq!(
                            simd_m[i].to_bits(),
                            scal_m[i].to_bits(),
                            "gemm n={n} nc={nc} nr={nr} value {i}"
                        );
                    }
                }
            }
        }
    }

    /// AVX512 == AVX2 == scalar across the same shape grid, forced per body:
    /// on an AVX512BW+DQ host `simd_x86_q4k::gemm_avx512` (the 512-bit
    /// section plus the AVX2 tail columns) must agree bit-for-bit with
    /// `gemm_256_section` over the whole matrix — the body an AVX2-only host
    /// runs — and with the scalar body: the three bodies are the same
    /// arithmetic in different lane assignments. Skipped (not failed) on
    /// hosts without AVX512BW+DQ.
    #[test]
    fn q4k_avx512_matches_avx2_and_scalar_shape_grid() {
        let _serial = super::tests::serial();
        if !crate::simd_x86::avx512bw() {
            eprintln!("host lacks AVX512BW+DQ: AVX512-vs-AVX2 check skipped");
            return;
        }
        let mut st = 0x8F3C_A291_77B5_14E6u64;
        let mut lcg = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as i32 as f32 / (1u32 << 31) as f32 * 1.5 - 0.75
        };
        for &n in &[256usize, 512, 768] {
            for &nc in &[8usize, 16, 24, 32, 40] {
                for &nr in &[4usize, 8, 16, 20, 36, 64] {
                    let (rep, q8kx4) = shape_grid_case(&mut lcg, n, nc, nr);
                    let mut a512 = vec![0f32; nr * nc];
                    // SAFETY: the repacked buffer holds nc*nb 1152-byte tiles
                    // and vy holds nr/4*nb 1168-byte tiles; nr*nc <= the
                    // output slots, bs = nc.
                    unsafe {
                        simd_x86_q4k::gemm_avx512(
                            n,
                            a512.as_mut_ptr(),
                            nc,
                            rep.as_ptr(),
                            q8kx4.as_ptr(),
                            nr,
                            nc,
                        );
                    }
                    let mut a2 = vec![0f32; nr * nc];
                    unsafe {
                        simd_x86_q4k::gemm_256_section(
                            n,
                            a2.as_mut_ptr(),
                            nc,
                            rep.as_ptr(),
                            q8kx4.as_ptr(),
                            nr,
                            nc,
                            0,
                        );
                    }
                    let mut scal = vec![0f32; nr * nc];
                    gemm_q4_K_8x8_q8_K_scalar(n, &mut scal, nc, &rep, &q8kx4, nr, nc);
                    for i in 0..nr * nc {
                        assert_eq!(
                            a512[i].to_bits(),
                            a2[i].to_bits(),
                            "avx512 vs avx2 n={n} nc={nc} nr={nr} value {i}"
                        );
                        assert_eq!(
                            a2[i].to_bits(),
                            scal[i].to_bits(),
                            "avx2 vs scalar n={n} nc={nc} nr={nr} value {i}"
                        );
                    }
                }
            }
        }
    }
}

/// Reference-dump parity for the Q4_0 8x8 path
/// (`parity/ref_repack_q4_0dump.cpp` -> `parity/q4_0_repack_ref.bin`).
#[cfg(test)]
mod q4_0_ref_tests {
    use super::*;
    use bytemuck::Zeroable;

    struct Dump {
        n: usize,
        nrows: usize,
        nr_gemv: usize,
        nr_gemm: usize,
        nc: usize,
        src: Vec<u8>,
        repacked: Vec<u8>,
        q8: Vec<u8>,
        q8x4: Vec<u8>,
        gemv_avx: Vec<f32>,
        gemm_avx: Vec<f32>,
        gemv_gen: Vec<f32>,
        gemm_gen: Vec<f32>,
        act_g: Vec<f32>,
        act_m: Vec<f32>,
        trait_present: u32,
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect()
    }

    fn parse(bytes: &[u8]) -> Dump {
        let mut c = bytes;
        let mut section = |c: &mut &[u8]| -> Vec<u8> {
            let len = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            *c = &c[4..];
            let (s, rest) = c.split_at(len);
            *c = rest;
            s.to_vec()
        };
        let hdr = section(&mut c);
        let u = |i: usize| u32::from_le_bytes(hdr[4 * i..4 * i + 4].try_into().unwrap());
        let (n, nrows, nr_gemv, nr_gemm, nc) =
            (u(1) as usize, u(2) as usize, u(3) as usize, u(4) as usize, u(5) as usize);
        let src = section(&mut c);
        let repacked = section(&mut c);
        let q8 = section(&mut c);
        let q8x4 = section(&mut c);
        let gemv_avx = f32s(&section(&mut c));
        let gemm_avx = f32s(&section(&mut c));
        let gemv_gen = f32s(&section(&mut c));
        let gemm_gen = f32s(&section(&mut c));
        let act_g = f32s(&section(&mut c));
        let act_m = f32s(&section(&mut c));
        let tr = section(&mut c);
        Dump {
            n,
            nrows,
            nr_gemv,
            nr_gemm,
            nc,
            src,
            repacked,
            q8,
            q8x4,
            gemv_avx,
            gemm_avx,
            gemv_gen,
            gemm_gen,
            act_g,
            act_m,
            trait_present: u32::from_le_bytes(tr[0..4].try_into().unwrap()),
        }
    }

    fn load() -> Dump {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/q4_0_repack_ref.bin");
        let bytes = std::fs::read(path).unwrap_or_else(|_| {
            panic!("missing {path}: build via parity/ref_repack_q4_0dump.cpp (see its header)")
        });
        parse(&bytes)
    }

    /// The reference build on this host must route Q4_0 to the 8x8 traits
    /// (repack.cpp:4987-4993); the dump's own log line says
    /// "repack tensor with q4_0_8x8".
    #[test]
    fn reference_q4_0_trait_is_the_8x8_repack() {
        let d = load();
        assert_eq!(d.trait_present, 1, "CPU_REPACK gave Q4_0 no trait (repack.cpp:4987)");
    }

    /// `make_block_q4_0x8` / `repack_q4_0_8x8_into` must reproduce the
    /// reference's CPU_REPACK buffer byte for byte (deltas + `^0x88` nibble
    /// interleave).
    #[test]
    fn repack_layout_matches_reference_bytes() {
        let d = load();
        let mine = repack_q4_0_8x8(&d.src, d.nrows, d.n);
        assert_eq!(mine.len(), d.repacked.len(), "repacked size");
        if mine != d.repacked {
            let i = mine.iter().zip(&d.repacked).position(|(a, b)| a != b).unwrap();
            let w = |s: &[u8]| s[..(i + 16).min(s.len())].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
            panic!("repacked bytes differ at {i} of {}\nmine: {}\nref : {}",
                   mine.len(), w(&mine), w(&d.repacked));
        }
        let back = unrepack_q4_0_8x8(&mine, d.nrows, d.n);
        assert_eq!(back, d.src, "unrepack round trip");
    }

    /// The port's plain `q8_0` rows (the gemv's LHS) must equal the reference's
    /// `from_float` bytes.
    #[test]
    fn q4_0_gemv_activation_bytes_match_reference() {
        let d = load();
        let nb = d.n / QK4_0;
        let mut mine = vec![0u8; d.nr_gemv * nb * BLOCK_Q8_0_SIZE];
        for r in 0..d.nr_gemv {
            let row = &d.act_g[r * d.n..(r + 1) * d.n];
            let mut blocks = vec![crate::blocks::BlockQ8_0::zeroed(); nb];
            crate::quants::quantize_row_q8_0(row, &mut blocks);
            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
            mine[r * bytes.len()..(r + 1) * bytes.len()].copy_from_slice(bytes);
        }
        assert_eq!(mine, d.q8);
    }

    /// The port's `quantize_mat_q8_0_4x8` must equal the reference's
    /// `ggml_quantize_mat_q8_0_4x8` bytes (same quantizer, same interleave).
    #[test]
    fn q4_0_interleaved_activation_matches_reference() {
        let d = load();
        let nb = d.n / QK4_0;
        let rows = d.nr_gemm;
        let mut mine = vec![0u8; (rows / 4) * nb * BLOCK_Q8_0X4_SIZE];
        quantize_mat_q8_0_4x8(&d.act_m[..rows * d.n], d.n, rows, &mut mine);
        assert_eq!(mine, d.q8x4);
    }

    /// The kernels must reproduce the reference's AVX bodies bit for bit on
    /// every value the dump covers (and not the `_generic` ones, which use a
    /// different rounding chain — asserted so the test cannot pass by accident).
    #[test]
    fn q4_0_kernels_match_reference_avx_bit_exact() {
        let _serial = super::tests::serial();
        let d = load();
        let mut mine_g = vec![1234.5f32; d.gemv_avx.len()];
        gemv_q4_0_8x8_q8_0(d.n, &mut mine_g, &d.repacked, &d.q8, d.nc);
        let mut mine_m = vec![1234.5f32; d.gemm_avx.len()];
        gemm_q4_0_8x8_q8_0(d.n, &mut mine_m, d.nc, &d.repacked, &d.q8x4, d.nr_gemm, d.nc);
        let bad_g: Vec<usize> = (0..d.gemv_avx.len())
            .filter(|&i| mine_g[i].to_bits() != d.gemv_avx[i].to_bits())
            .collect();
        let bad_m: Vec<usize> = (0..d.gemm_avx.len())
            .filter(|&i| mine_m[i].to_bits() != d.gemm_avx[i].to_bits())
            .collect();
        assert!(
            bad_g.is_empty(),
            "gemv differs from the reference AVX body at {} of {} slots, first {:?} \
             (mine {:?} vs ref {:?})",
            bad_g.len(),
            d.gemv_avx.len(),
            bad_g.first(),
            bad_g.first().map(|&i| mine_g[i]),
            bad_g.first().map(|&i| d.gemv_avx[i]),
        );
        assert!(
            bad_m.is_empty(),
            "gemm differs from the reference AVX body at {} of {} slots, first {:?} \
             (mine {:?} vs ref {:?})",
            bad_m.len(),
            d.gemm_avx.len(),
            bad_m.first(),
            bad_m.first().map(|&i| mine_m[i]),
            bad_m.first().map(|&i| d.gemm_avx[i]),
        );
        // the two reference bodies really do differ (the dump's own report
        // says so), so the assertions above are about the AVX body
        let n_gen_diff = d
            .gemv_avx
            .iter()
            .zip(&d.gemv_gen)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(n_gen_diff > 0, "expected reference AVX != generic on gemv");
        let n_gen_diff_m = d
            .gemm_avx
            .iter()
            .zip(&d.gemm_gen)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(n_gen_diff_m > 0, "expected reference AVX != generic on gemm");
    }

    /// SIMD == scalar on the dump's own data (both must be the AVX body's
    /// arithmetic; `LLAMA_RUST_REPACK_SIMD=0` only switches speed).
    #[test]
    fn q4_0_simd_matches_scalar_bit_exact() {
        let _serial = super::tests::serial();
        let d = load();
        let mut simd_g = vec![0f32; d.nc];
        gemv_q4_0_8x8_q8_0(d.n, &mut simd_g, &d.repacked, &d.q8, d.nc);
        assert!(super::repack_q4_0_gemv_calls() >= 1);
        let mut scal_g = vec![0f32; d.nc];
        gemv_q4_0_8x8_q8_0_scalar(d.n, &mut scal_g, &d.repacked, &d.q8, d.nc);
        for i in 0..d.nc {
            assert_eq!(simd_g[i].to_bits(), scal_g[i].to_bits(), "gemv value {i}");
        }
        let mut simd_m = vec![0f32; d.nr_gemm * d.nc];
        gemm_q4_0_8x8_q8_0(d.n, &mut simd_m, d.nc, &d.repacked, &d.q8x4, d.nr_gemm, d.nc);
        let mut scal_m = vec![0f32; d.nr_gemm * d.nc];
        gemm_q4_0_8x8_q8_0_scalar(d.n, &mut scal_m, d.nc, &d.repacked, &d.q8x4, d.nr_gemm, d.nc);
        for i in 0..d.nr_gemm * d.nc {
            assert_eq!(simd_m[i].to_bits(), scal_m[i].to_bits(), "gemm value {i}");
        }
    }

    /// SIMD == scalar across a shape grid, so every loop boundary of the
    /// gemm's lane sections is covered on any given host: on an AVX512 host
    /// `nc % 16 == 0` shapes run the AVX512 section while `nc % 16 == 8`
    /// shapes also cross into the AVX2 tail-columns section (its 16-row
    /// group and 4-row loops, `xstart != 0`); an AVX2-only host runs the
    /// AVX2 section for all of them. `nr` sweeps the 16-row groups, the
    /// 4-row tail and both gemv tail rows.
    #[test]
    fn q4_0_simd_matches_scalar_shape_grid() {
        let _serial = super::tests::serial();
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let mut lcg = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as i32 as f32 / (1u32 << 31) as f32 * 1.5 - 0.75
        };
        for &n in &[96usize, 256, 160] {
            let nb = n / QK4_0;
            for &nc in &[8usize, 16, 24, 32, 40] {
                // random Q4_0 weights, repacked
                let mut wf = vec![0f32; nc * n];
                for v in wf.iter_mut() {
                    *v = lcg();
                }
                let mut wsrc = vec![0u8; nc * nb * BLOCK_Q4_0_SIZE];
                let mut blocks = vec![crate::blocks::BlockQ4_0::zeroed(); nb];
                for r in 0..nc {
                    crate::quants::quantize_row_q4_0_ref(&wf[r * n..][..n], &mut blocks);
                    let b: &[u8] = bytemuck::cast_slice(&blocks);
                    wsrc[r * nb * BLOCK_Q4_0_SIZE..][..nb * BLOCK_Q4_0_SIZE].copy_from_slice(b);
                }
                let rep = repack_q4_0_8x8(&wsrc, nc, n);
                for &nr in &[4usize, 8, 16, 20, 36, 64] {
                    let act: Vec<f32> = (0..nr * n).map(|_| lcg()).collect();
                    let mut q8x4 = vec![0u8; (nr / 4) * nb * BLOCK_Q8_0X4_SIZE];
                    quantize_mat_q8_0_4x8(&act, n, nr, &mut q8x4);
                    let mut simd_m = vec![0f32; nr * nc];
                    gemm_q4_0_8x8_q8_0(n, &mut simd_m, nc, &rep, &q8x4, nr, nc);
                    let mut scal_m = vec![0f32; nr * nc];
                    gemm_q4_0_8x8_q8_0_scalar(n, &mut scal_m, nc, &rep, &q8x4, nr, nc);
                    for i in 0..nr * nc {
                        assert_eq!(
                            simd_m[i].to_bits(),
                            scal_m[i].to_bits(),
                            "gemm n={n} nc={nc} nr={nr} value {i}"
                        );
                    }
                }
            }
        }
    }

    /// 2D Q4_0 `mul_mat` through `graph_compute`: a qualifying tensor
    /// (`ne[1] % 8 == 0`) must take the repack branch (asserted via the call
    /// counters) for both the multi-column prefill shape (gemm + gemv tail)
    /// and the single-column decode shape (gemv only), and land on the f64
    /// reference built from `dequantize_row_q4_0` + q8_0 activations.
    #[test]
    fn mul_mat_q4_0_repack_wiring_matches_f64_reference() {
        let _serial = super::tests::serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;

        let (n_per_row, n_rows, n_act) = (96usize, 16usize, 6usize); // 6 = 4-row gemm group + 2-row gemv tail
        let mut ctx = Context::new();
        let wgt = ctx.new_tensor_2d(GgmlType::Q4_0, n_per_row as i64, n_rows as i64);
        let act = ctx.new_tensor_2d(GgmlType::F32, n_per_row as i64, n_act as i64);
        let dst = ctx.mul_mat(wgt, act);
        for id in [wgt, act, dst] {
            ctx.arena_resize_tensor(id);
        }

        let mut st = 0x0f0f_0f0fu32;
        let mut lcg = move || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (st as i32 as f32 / (1u32 << 28) as f32) * 1.5 - 0.75
        };
        let wf: Vec<f32> = (0..n_rows * n_per_row).map(|_| lcg()).collect();
        let mut wsrc = vec![0u8; n_rows * (n_per_row / QK4_0) * BLOCK_Q4_0_SIZE];
        let mut blocks = vec![crate::blocks::BlockQ4_0::zeroed(); n_per_row / QK4_0];
        for r in 0..n_rows {
            crate::quants::quantize_row_q4_0_ref(&wf[r * n_per_row..][..n_per_row], &mut blocks);
            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
            wsrc[r * bytes.len()..][..bytes.len()].copy_from_slice(bytes);
        }
        ctx.data_bytes_mut(wgt).unwrap().copy_from_slice(&wsrc);
        let a: Vec<f32> = (0..n_per_row * n_act).map(|_| lcg()).collect();
        ctx.with_f32_mut(act, |p| p.copy_from_slice(&a)).unwrap();

        let stats_before = repack_stats();
        let (gv_before, gm_before) = (super::repack_q4_0_gemv_calls(), super::repack_q4_0_gemm_calls());
        let mut g = Graph::new(8);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 2);
        assert_eq!(repack_stats().0, stats_before.0 + 1, "the weight must be repacked once");
        // gemm served the 4-row group, gemv the 2-row tail
        assert!(super::repack_q4_0_gemm_calls() > gm_before, "the gemm branch must run");
        assert!(super::repack_q4_0_gemv_calls() > gv_before, "the gemv tail must run");

        // f64 reference per (activation row, weight row)
        let nb = n_per_row / QK4_0;
        let out = ctx.f32s(dst).unwrap().to_vec();
        for c in 0..n_act {
            let mut qb = vec![crate::blocks::BlockQ8_0::zeroed(); nb];
            crate::quants::quantize_row_q8_0(&a[c * n_per_row..][..n_per_row], &mut qb);
            for r in 0..n_rows {
                let blocks: &[crate::blocks::BlockQ4_0] = bytemuck::cast_slice(
                    &wsrc[r * nb * BLOCK_Q4_0_SIZE..][..nb * BLOCK_Q4_0_SIZE],
                );
                let mut w = vec![0f32; n_per_row];
                crate::quants::dequantize_row_q4_0(blocks, &mut w);
                let mut want = 0f64;
                for ib in 0..nb {
                    let d = qb[ib].d.to_f64();
                    for j in 0..QK4_0 {
                        want += w[ib * QK4_0 + j] as f64 * qb[ib].qs[j] as f64 * d;
                    }
                }
                let got = out[r + c * n_rows] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "dst[{r},{c}] = {got} vs f64 {want}"
                );
            }
        }
    }

    /// A Q4_0 tensor that does not qualify (`ne[1] % 8 != 0`) must keep the
    /// pre-repack routing (tinyBLAS / row-wise vec_dot), like a reference
    /// tensor the CPU_REPACK buffer never hosts.
    #[test]
    fn mul_mat_q4_0_repack_skips_unqualified_tensor() {
        let _serial = super::tests::serial();
        use crate::graph::Graph;
        use crate::tensor::Context;
        use crate::types::GgmlType;

        let mut ctx = Context::new();
        let wgt = ctx.new_tensor_2d(GgmlType::Q4_0, 96, 12); // 12 % 8 != 0
        let act = ctx.new_tensor_2d(GgmlType::F32, 96, 3);
        let dst = ctx.mul_mat(wgt, act);
        for id in [wgt, act, dst] {
            ctx.arena_resize_tensor(id);
        }
        let stats_before = repack_stats();
        let (gv, gm) = (super::repack_q4_0_gemv_calls(), super::repack_q4_0_gemm_calls());
        let mut g = Graph::new(8);
        g.build_forward(&ctx, dst);
        crate::compute::graph_compute(&mut ctx, &mut g, 1);
        assert_eq!(repack_stats(), stats_before, "unqualified tensor must not be repacked");
        assert_eq!((super::repack_q4_0_gemv_calls(), super::repack_q4_0_gemm_calls()), (gv, gm));
        assert!(ctx.f32s(dst).unwrap().iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod q4k_diag_tests {
    use super::*;
    use bytemuck::Zeroable;

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    /// Build a synthetic 8-row Q4_K group, repack it, and compare the SIMD
    /// gemm against the scalar one for three weight regimes: full, mins only
    /// (scales zero), scales only (dmin zero).
    #[test]
    fn q4k_gemm_simd_vs_scalar_regimes() {
        let n = 256usize;
        let nb = 1usize;
        let nrows = 8usize;
        let nc = nrows;
        let mut st = 0x1234_5678u32;
        for regime in ["full", "dmin=0", "scale=0", "d=0"] {
            let mut src = vec![0u8; nrows * nb * BLOCK_Q4_K_SIZE];
            for r in 0..nrows {
                let b = &mut src[r * BLOCK_Q4_K_SIZE..(r + 1) * BLOCK_Q4_K_SIZE];
                let dh = if regime == "d=0" { 0u16 } else { (0x2000u32 + lcg(&mut st) % 0x2000) as u16 };
                let dm = if regime == "dmin=0" { 0u16 } else { (0x2000u32 + lcg(&mut st) % 0x2000) as u16 };
                b[0..2].copy_from_slice(&dh.to_le_bytes());
                b[2..4].copy_from_slice(&dm.to_le_bytes());
                for j in 4..16 {
                    let v = if regime == "scale=0" { 0u8 } else { lcg(&mut st) as u8 };
                    b[j] = v;
                }
                for j in 16..144 {
                    b[j] = lcg(&mut st) as u8;
                }
            }
            let rep = repack_q4_K_8x8(&src, nrows, n);
            // 4 activation rows -> one q8_Kx4 tile
            let mut act = vec![0f32; 4 * n];
            for v in act.iter_mut() {
                *v = ((lcg(&mut st) as i32 as f32) / (1u32 << 28) as f32) - 1.0;
            }
            let mut q8x4 = vec![0u8; nb * BLOCK_Q8_KX4_SIZE];
            quantize_mat_q8_K_4x8(&act, n, 4, &mut q8x4);
            let mut a = vec![0f32; 4 * nc];
            let mut b = vec![0f32; 4 * nc];
            gemm_q4_K_8x8_q8_K_scalar(n, &mut a, nc, &rep, &q8x4, 4, nc);
            gemm_q4_K_8x8_q8_K(n, &mut b, nc, &rep, &q8x4, 4, nc);
            let bad: Vec<usize> = (0..4 * nc).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
            println!(
                "regime {regime:>8}: {} of {} differ; first {:?} scalar {:?} simd {:?}",
                bad.len(),
                4 * nc,
                bad.first(),
                bad.first().map(|&i| a[i]),
                bad.first().map(|&i| b[i]),
            );
            for &i in bad.iter().take(4) {
                println!("    ({}, {}): scalar {:?} simd {:?}", i / nc, i % nc, a[i], b[i]);
            }
        }
    }
}

#[cfg(test)]
mod q4k_min_probe_tests {
    use super::*;

    /// Single nonzero min (sub-block 0, column 0) + `d = 0`: every output is
    /// `-(B(0) of some row) * dmin_col * d_row`, so the printed tables show
    /// exactly which (row, column, sub-block) the SIMD min term picks.
    #[test]
    fn q4k_gemm_min_term_mapping() {
        for sb in 0..8usize {
            probe_one_min(sb);
        }
    }

    fn probe_one_min(sb: usize) {
        let n = 256usize;
        let nrows = 8usize;
        let mut st = 0x9e37_79b9u32;
        let mut lcg = move || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            st
        };
        let mut src = vec![0u8; nrows * BLOCK_Q4_K_SIZE];
        for r in 0..nrows {
            let b = &mut src[r * BLOCK_Q4_K_SIZE..(r + 1) * BLOCK_Q4_K_SIZE];
            b[0..2].copy_from_slice(&0u16.to_le_bytes()); // d = 0
            let dm = half::f16::from_f32(1.0).to_bits();
            b[2..4].copy_from_slice(&dm.to_le_bytes()); // dmin = 1
            for j in 4..144 {
                b[j] = 0;
            }
            if r == 0 {
                // get_scale_min_k4(sb) reads q[sb]&63 / q[sb+4]&63 for sb<4 and
                // q[sb+4]&0xF | q[sb-4]>>6<<4 / q[sb+4]>>4 | q[sb]>>6<<4 above
                if sb < 4 {
                    b[4 + sb + 4] = 1;
                } else {
                    b[4 + sb + 4] = 16;
                }
            }
            for j in 16..144 {
                b[j] = lcg() as u8;
            }
        }
        let rep = repack_q4_K_8x8(&src, nrows, n);
        let mut act = vec![0f32; 4 * n];
        for v in act.iter_mut() {
            *v = ((lcg() as i32) as f32 / (1u32 << 28) as f32) - 1.0;
        }
        let mut q8x4 = vec![0u8; BLOCK_Q8_KX4_SIZE];
        quantize_mat_q8_K_4x8(&act, n, 4, &mut q8x4);
        println!("act rows' d: {:?}", (0..4).map(|r| f32::from_le_bytes(q8x4[4 * r..4 * r + 4].try_into().unwrap())).collect::<Vec<_>>());
        for r in 0..4 {
            let bs: Vec<i16> = (0..16)
                .map(|j| i16::from_le_bytes([q8x4[1040 + 2 * (16 * 0 + 4 * r + j % 4 + 4 * (j / 4))], q8x4[1041 + 2 * (16 * 0 + 4 * r + j % 4 + 4 * (j / 4))]]))
                .collect();
            println!("  row {r} bsums[0..4] (sub-block 0 pair) {bs:?}");
        }
        let mut a = vec![0f32; 4 * nrows];
        let mut b = vec![0f32; 4 * nrows];
        gemm_q4_K_8x8_q8_K_scalar(n, &mut a, nrows, &rep, &q8x4, 4, nrows);
        gemm_q4_K_8x8_q8_K(n, &mut b, nrows, &rep, &q8x4, 4, nrows);
        // raw min values: out = -min (acc = 0)
        for i in 0..4 {
            println!(
                "row {i}: scalar {:?}",
                (0..8).map(|c| -a[i * nrows + c]).collect::<Vec<_>>()
            );
            println!(
                "row {i}: simd   {:?}",
                (0..8).map(|c| -b[i * nrows + c]).collect::<Vec<_>>()
            );
        }
    }
}

/// Kernel throughput microbench (not a correctness test): `cargo test
/// --release -p ggml --lib q4k_kernel_speed -- --ignored --nocapture`.
#[cfg(test)]
mod q4k_speed_tests {
    use super::*;
    use bytemuck::Zeroable;

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    #[test]
    #[ignore]
    fn q4k_repack_materialize_speed() {
        // the local 0.5B's Q4_K payload: 12 x [4864 x 896]
        let (n, nc) = (4864usize, 896usize);
        let nb = n / QK_K;
        let mut st = 0x1234_5678u32;
        let mut src = vec![0u8; nc * nb * BLOCK_Q4_K_SIZE];
        for b in src.iter_mut() {
            *b = lcg(&mut st) as u8;
        }
        let t = std::time::Instant::now();
        let out = repack_q4_K_8x8(&src, nc, n);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "repack_q4_K_8x8: {} MiB in {:.1} ms ({:.2} GiB/s) -> 12 tensors = {:.1} ms",
            src.len() >> 20,
            dt * 1e3,
            src.len() as f64 / dt / (1u64 << 30) as f64,
            dt * 1e3 * 12.0
        );
        let _ = out;
    }

    /// The production mixture at realistic chunk sizes + thread scaling.
    #[test]
    #[ignore]
    fn q4k_gemm_chunk_and_threads() {
        let (n, nc, nr) = (3584usize, 18944usize, 8usize);
        let nb = n / QK_K;
        let mut st = 0x5eed_1234u32;
        let mut src = vec![0u8; nc * nb * BLOCK_Q4_K_SIZE];
        for b in src.iter_mut() {
            *b = lcg(&mut st) as u8;
        }
        for blk in src.chunks_mut(BLOCK_Q4_K_SIZE) {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());
            blk[2..4].copy_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
        }
        let rep = repack_q4_K_8x8(&src, nc, n);
        let mut act = vec![0f32; nr * n];
        for v in act.iter_mut() {
            *v = ((lcg(&mut st) as i32) as f32 / (1u32 << 28) as f32) - 1.0;
        }
        let mut q8x4 = vec![0u8; nr / 4 * nb * BLOCK_Q8_KX4_SIZE];
        for g in 0..nr / 4 {
            quantize_mat_q8_K_4x8(&act[g * 4 * n..(g * 4 + 4) * n], n, 4,
                                  &mut q8x4[g * nb * BLOCK_Q8_KX4_SIZE..]);
        }
        let macs = (nr * nc * n) as f64;
        let mut out = vec![0f32; nr * nc];
        for &chunk_rows in &[nc, 1024, 512, 256, 128, 32] {
            let nchunk0 = nc.div_ceil(chunk_rows);
            let t = std::time::Instant::now();
            for c in 0..nchunk0 {
                let c0 = c * chunk_rows;
                let c1 = (c0 + chunk_rows).min(nc);
                let bmat = &rep[c0 / 8 * nb * BLOCK_Q4_KX8_SIZE..];
                let d = &mut out[c0 * 0 + 0..];
                let _ = d;
                gemm_q4_K_8x8_q8_K(n, &mut out[c0..], nc, bmat, &q8x4, nr, c1 - c0);
            }
            let dt = t.elapsed().as_secs_f64();
            println!("1 thread, chunks of {chunk_rows:>5}: {:7.2} ms  {:6.1} GMAC/s",
                     dt * 1e3, macs / dt / 1e9);
        }
        // thread scaling at the production chunking (128 rows)
        for &nt in &[1usize, 2, 4, 8] {
            let nchunk0 = nc.div_ceil(128);
            let t = std::time::Instant::now();
            let rep = &rep;
            let q8x4 = &q8x4;
            std::thread::scope(|sc| {
                let mut handles = Vec::new();
                for w in 0..nt {
                    let (rep, q8x4) = (rep, q8x4);
                    let out_ptr = out.as_mut_ptr() as usize;
                    handles.push(sc.spawn(move || {
                        for c in (w..nchunk0).step_by(nt) {
                            let c0 = c * 128;
                            let c1 = (c0 + 128).min(nc);
                            let bmat = &rep[c0 / 8 * nb * BLOCK_Q4_KX8_SIZE..];
                            let s = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (out_ptr as *mut f32).add(c0), nc - c0)
                            };
                            gemm_q4_K_8x8_q8_K(n, s, nc, bmat, q8x4, nr, c1 - c0);
                        }
                    }));
                }
                for h in handles {
                    h.join().unwrap();
                }
            });
            let dt = t.elapsed().as_secs_f64();
            println!("chunks of 128, {nt} threads: {:7.2} ms  {:6.1} GMAC/s",
                     dt * 1e3, macs / dt / 1e9);
        }
    }

    #[test]
    #[ignore]
    fn q4k_kernel_speed() {
        // the local qwen2.5-0.5B's ffn_down geometry
        for &(n, nc) in &[(4864usize, 896usize), (3584, 3584)] {
            let nb = n / QK_K;
            let nrows = nc;
            let mut st = 0x2468_ace1u32;
            let mut src = vec![0u8; nrows * nb * BLOCK_Q4_K_SIZE];
            for b in src.iter_mut() {
                *b = lcg(&mut st) as u8;
            }
            // keep d/dmin sane (f16 ~0.1)
            for blk in src.chunks_mut(BLOCK_Q4_K_SIZE) {
                blk[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());
                blk[2..4].copy_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
            }
            let rep = repack_q4_K_8x8(&src, nrows, n);
            for &nr in &[1usize, 4, 8, 64] {
                let mut act = vec![0f32; nr * n];
                for v in act.iter_mut() {
                    *v = ((lcg(&mut st) as i32) as f32 / (1u32 << 28) as f32) - 1.0;
                }
                let nr4 = nr - nr % 4;
                // gemv (1 row at a time) — the port's row-wise equivalent
                let mut out = vec![0f32; nr * nc];
                let mut q8 = vec![0u8; nr * nb * BLOCK_Q8_K_SIZE];
                for r in 0..nr {
                    let mut blocks = vec![crate::blocks::BlockQ8K::zeroed(); nb];
                    crate::quants::quantize_row_q8_K_ref(&act[r * n..(r + 1) * n], &mut blocks);
                    let bytes: &[u8] = bytemuck::cast_slice(&blocks);
                    q8[r * bytes.len()..(r + 1) * bytes.len()].copy_from_slice(bytes);
                }
                let t = std::time::Instant::now();
                for r in 0..nr {
                    gemv_q4_K_8x8_q8_K(n, &mut out[r * nc..], &rep, &q8[r * nb * BLOCK_Q8_K_SIZE..], nc);
                }
                let t_gemv = t.elapsed().as_secs_f64();
                // gemm (4-row groups) + gemv tail, the production mixture
                let mut out2 = vec![0f32; nr * nc];
                let mut q8x4 = vec![0u8; nr4 / 4 * nb * BLOCK_Q8_KX4_SIZE];
                for g in 0..nr4 / 4 {
                    quantize_mat_q8_K_4x8(
                        &act[g * 4 * n..(g * 4 + 4) * n],
                        n,
                        4,
                        &mut q8x4[g * nb * BLOCK_Q8_KX4_SIZE..],
                    );
                }
                let mut q8t = vec![0u8; (nr - nr4) * nb * BLOCK_Q8_K_SIZE];
                for r in nr4..nr {
                    let mut blocks = vec![crate::blocks::BlockQ8K::zeroed(); nb];
                    crate::quants::quantize_row_q8_K_ref(&act[r * n..(r + 1) * n], &mut blocks);
                    let bytes: &[u8] = bytemuck::cast_slice(&blocks);
                    q8t[(r - nr4) * bytes.len()..][..bytes.len()].copy_from_slice(bytes);
                }
                let t = std::time::Instant::now();
                if nr4 > 0 {
                    gemm_q4_K_8x8_q8_K(n, &mut out2, nc, &rep, &q8x4, nr4, nc);
                }
                for r in nr4..nr {
                    gemv_q4_K_8x8_q8_K(
                        n,
                        &mut out2[r * nc..],
                        &rep,
                        &q8t[(r - nr4) * nb * BLOCK_Q8_K_SIZE..],
                        nc,
                    );
                }
                let t_gemm = t.elapsed().as_secs_f64();
                let macs = (nr * nc * n) as f64;
                println!(
                    "n={n} nc={nc} nr={nr:>2}: gemv-only {:7.3} ms ({:6.1} GMAC/s) | gemm+tail {:7.3} ms \
                     ({:6.1} GMAC/s)  x{:.2}",
                    t_gemv * 1e3,
                    macs / t_gemv / 1e9,
                    t_gemm * 1e3,
                    macs / t_gemm / 1e9,
                    t_gemv / t_gemm,
                );
            }
        }
    }

    /// AVX512 (`gemm_avx512`, arch/x86/repack.cpp:2077-2815) vs AVX2
    /// (`gemm_256_section` over the whole matrix) gemm sections on the
    /// production geometries — the round-5 kernel A/B. Requires an
    /// AVX512BW+DQ host (prints and returns otherwise).
    #[test]
    #[ignore]
    fn q4k_gemm_avx512_vs_avx2_speed() {
        if !crate::simd_x86::avx512bw() {
            eprintln!("host lacks AVX512BW+DQ: nothing to A/B");
            return;
        }
        for &(n, nc) in &[(4864usize, 896usize), (3584, 3584)] {
            let nb = n / QK_K;
            let mut st = 0x1357_9BDFu32;
            let mut src = vec![0u8; nc * nb * BLOCK_Q4_K_SIZE];
            for b in src.iter_mut() {
                *b = lcg(&mut st) as u8;
            }
            for blk in src.chunks_mut(BLOCK_Q4_K_SIZE) {
                blk[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());
                blk[2..4].copy_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
            }
            let rep = repack_q4_K_8x8(&src, nc, n);
            for &nr in &[16usize, 64, 256] {
                let act: Vec<f32> = (0..nr * n)
                    .map(|_| ((lcg(&mut st) as i32) as f32 / (1u32 << 28) as f32) - 1.0)
                    .collect();
                let mut q8x4 = vec![0u8; (nr / 4) * nb * BLOCK_Q8_KX4_SIZE];
                quantize_mat_q8_K_4x8(&act, n, nr, &mut q8x4);
                let mut out = vec![0f32; nr * nc];
                let reps = 3;
                // SAFETY: same argument contract as `gemm_q4_K_8x8_q8_K`.
                unsafe {
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        simd_x86_q4k::gemm_256_section(
                            n,
                            out.as_mut_ptr(),
                            nc,
                            rep.as_ptr(),
                            q8x4.as_ptr(),
                            nr,
                            nc,
                            0,
                        );
                    }
                    let t_avx2 = t.elapsed().as_secs_f64() / reps as f64;
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        simd_x86_q4k::gemm_avx512(
                            n,
                            out.as_mut_ptr(),
                            nc,
                            rep.as_ptr(),
                            q8x4.as_ptr(),
                            nr,
                            nc,
                        );
                    }
                    let t_avx512 = t.elapsed().as_secs_f64() / reps as f64;
                    let macs = (nr * nc * n) as f64;
                    println!(
                        "n={n} nc={nc} nr={nr:>3}: avx2 {:7.3} ms ({:6.1} GMAC/s) | avx512 {:7.3} ms \
                         ({:6.1} GMAC/s)  x{:.2}",
                        t_avx2 * 1e3,
                        macs / t_avx2 / 1e9,
                        t_avx512 * 1e3,
                        macs / t_avx512 / 1e9,
                        t_avx2 / t_avx512,
                    );
                }
                let _ = out;
            }
        }
    }
}

/// Round-9 gemv/gemm codegen lab (perf9): the A/B harness that localized the
/// MXFP4 gap. Findings (see PARITY.md perf9):
///   * the pre-perf9 isolated "1.98x gemv gap" was a *harness-data artifact*:
///     `tests::rand_mxfp4` seeds E8M0 exponents in {0,1,2} every 16th block,
///     so ~40% of tiles have a col×row product that underflows to a denormal
///     and the vmulps takes FP assists (~70c each) — real gpt-oss weights
///     (block scales ≈ 2^-4) never produce them. On clean data
///     (LAB_CLEAN_DATA=1) the production gemv is FASTER than the reference
///     (≈137 vs 158 µs on the 2880×2880 expert shape).
///   * the real gap was the gemm: the port re-decoded each weight tile per
///     activation row (16×) where the reference shares one decode per tile
///     pair (AVX512 zmm 2x2 scheme, arch/x86/repack.cpp:663-1096, ported in
///     this round as `gemm_avx512`).
/// Diagnostic-only; run with
/// `cargo test --release -p ggml --lib mxfp4_gemv_lab -- --ignored --nocapture`.
#[cfg(test)]
mod mxfp4_gemv_lab {
    use super::{BLOCK_MXFP4_SIZE, BLOCK_MXFP4X8_SIZE, BLOCK_Q8_0_SIZE, NROWS_INTERLEAVED, QK_MXFP4, QK8_0};
    use crate::quants::E8M0_HALF_LUT;
    use core::arch::x86_64::*;

    fn lcg_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as i32 as f32) / (1u32 << 28) as f32 - 1.0
            })
            .collect()
    }

    /// same generator as tests::rand_mxfp4 (kept local: the lab must not
    /// depend on the test module's private items)
    fn rand_mxfp4(nrows: usize, n_per_row: usize, seed: u32) -> Vec<u8> {
        let nb = n_per_row / QK_MXFP4;
        let mut s = seed;
        let mut out = vec![0u8; nrows * nb * BLOCK_MXFP4_SIZE];
        for (ib, blk) in out.chunks_mut(BLOCK_MXFP4_SIZE).enumerate() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            blk[0] = if ib % 16 == 15 { (s % 3) as u8 } else { 118 + (s % 16) as u8 };
            for q in &mut blk[1..] {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *q = (s >> 24) as u8;
            }
        }
        out
    }

    /// `mul_sum_i8_pairs_acc_int32x8`, VNNI spelling (arch/x86/repack.cpp:153) —
    /// the port's `iacc_row_vnni` core, split out so the lab can steer the
    /// accumulator. Value-identical proof shared with the production kernel.
    #[inline(always)]
    unsafe fn dpb(acc: __m256i, x: __m256i, y: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(acc, _mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x))
    }

    #[inline(always)]
    unsafe fn d_f16(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(
            u16::from_le_bytes([*p, *p.add(1)]) as i32,
        )))
    }

    /// Scale-path knob: "vec" = production arithmetic decode; "lut" = the
    /// reference's 8 scalar gathers from `ggml_table_f32_e8m0_half`;
    /// "one" = constant 1.0 (DIAGNOSTIC ONLY, wrong values).
    #[inline(always)]
    unsafe fn col_scale(tile: *const u8, mode: u8) -> __m256 {
        match mode {
            1 => {
                let g = |i: usize| E8M0_HALF_LUT[*tile.add(i) as usize];
                _mm256_set_ps(g(7), g(3), g(6), g(2), g(5), g(1), g(4), g(0))
            }
            2 => _mm256_set1_ps(1.0),
            _ => {
                // hand-vectorized e8m0_to_fp32_half: x>=2 -> (x-1)<<23;
                // x in {0,1} -> the subnormal patterns 0x00200000<<x =
                // 0x00200000 + (x<<21). The x=0 lane's (0-1)<<23 = 0xFF800000
                // is blended out (and integer SIMD wraps, so no UB).
                // bytes pre-permuted (0,4,1,5,2,6,3,7) so the lanes land in
                // the iacc order (B0,B4,B1,B5,B2,B6,B3,B7).
                let bytes = _mm_shuffle_epi8(
                    _mm_loadl_epi64(tile as *const __m128i),
                    _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0),
                );
                let idx = _mm256_cvtepu8_epi32(bytes);
                let shifted = _mm256_slli_epi32::<23>(_mm256_sub_epi32(idx, _mm256_set1_epi32(1)));
                let denorm = _mm256_add_epi32(
                    _mm256_slli_epi32::<21>(idx),
                    _mm256_set1_epi32(0x0020_0000),
                );
                let is_denorm = _mm256_cmpgt_epi32(_mm256_set1_epi32(2), idx);
                _mm256_castsi256_ps(_mm256_blendv_epi8(shifted, denorm, is_denorm))
            }
        }
    }

    /// One lab kernel instantiation.
    ///   $split  — number of independent iacc accumulators (1/2/4)
    ///   $scale  — 0 vec / 1 lut / 2 one(diagnostic)
    ///   $unroll — blocks per iteration (1/2)
    macro_rules! lab_gemv {
        ($(#[$attr:meta])* $name:ident, $split:expr, $scale:expr, $unroll:expr) => {
            lab_gemv!($(#[$attr])* $name, $split, $scale, $unroll, 0);
        };
        ($(#[$attr:meta])* $name:ident, $split:expr, $scale:expr, $unroll:expr, $tail:expr) => {
            $(#[$attr])*
            pub unsafe fn $name(n: usize, s: *mut f32, vx: *const u8, vy: *const u8, nc: usize) {
                const S: usize = $split;
                const U: usize = $unroll;
                let nb = n / QK8_0;
                let b_nb = n / 32;
                let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    crate::quants::KVALUES_MXFP4.as_ptr() as *const __m128i));
                let m4b = _mm256_set1_epi8(0x0F);
                let finalpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                for x in 0..nc / NROWS_INTERLEAVED {
                    let b_ptr = vx.add(x * b_nb * BLOCK_MXFP4X8_SIZE);
                    let mut acc = _mm256_setzero_ps();
                    let mut acc2 = _mm256_set1_ps(1.0);
                    let b_end = nb - nb % U;
                    let mut b = 0usize;
                    while b < b_end {
                        let mut u = 0usize;
                        while u < U {
                            let tile = b_ptr.add((b + u) * BLOCK_MXFP4X8_SIZE);
                            let a_blk = vy.add((b + u) * BLOCK_Q8_0_SIZE);
                            let qs = a_blk.add(2);
                            let (lhs0lo, lhs1lo) = (
                                _mm_loadu_si128(qs as *const __m128i),
                                _mm_loadu_si128(qs.add(16) as *const __m128i),
                            );
                            let lhs0 = _mm256_broadcastsi128_si256(lhs0lo);
                            let lhs1 = _mm256_broadcastsi128_si256(lhs1lo);

                            let tileq = tile.add(8);
                            let raw_0123_0 = _mm256_loadu_si256(tileq as *const __m256i);
                            let raw_4567_0 = _mm256_loadu_si256(tileq.add(32) as *const __m256i);
                            let raw_0123_1 = _mm256_loadu_si256(tileq.add(64) as *const __m256i);
                            let raw_4567_1 = _mm256_loadu_si256(tileq.add(96) as *const __m256i);
                            let r0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, m4b));
                            let r4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, m4b));
                            let r0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, m4b));
                            let r4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, m4b));
                            let r0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_0), m4b));
                            let r4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_0), m4b));
                            let r0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_1), m4b));
                            let r4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_1), m4b));

                            let t0 = _mm256_blend_epi32::<170>(r0123_0, _mm256_shuffle_epi32::<177>(r4567_0));
                            let t1 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_0), r4567_0);
                            let t2 = _mm256_blend_epi32::<170>(r0123_1, _mm256_shuffle_epi32::<177>(r4567_1));
                            let t3 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_1), r4567_1);
                            let t4 = _mm256_blend_epi32::<170>(r0123_2, _mm256_shuffle_epi32::<177>(r4567_2));
                            let t5 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_2), r4567_2);
                            let t6 = _mm256_blend_epi32::<170>(r0123_3, _mm256_shuffle_epi32::<177>(r4567_3));
                            let t7 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_3), r4567_3);
                            let l0 = _mm256_shuffle_epi32::<0>(lhs0);
                            let l1 = _mm256_shuffle_epi32::<85>(lhs0);
                            let l2 = _mm256_shuffle_epi32::<170>(lhs0);
                            let l3 = _mm256_shuffle_epi32::<255>(lhs0);
                            let l4 = _mm256_shuffle_epi32::<0>(lhs1);
                            let l5 = _mm256_shuffle_epi32::<85>(lhs1);
                            let l6 = _mm256_shuffle_epi32::<170>(lhs1);
                            let l7 = _mm256_shuffle_epi32::<255>(lhs1);

                            let mut iacc = [_mm256_setzero_si256(); S];
                            iacc[0 % S] = dpb(iacc[0 % S], t0, l0);
                            iacc[1 % S] = dpb(iacc[1 % S], t1, l1);
                            iacc[2 % S] = dpb(iacc[2 % S], t2, l2);
                            iacc[3 % S] = dpb(iacc[3 % S], t3, l3);
                            iacc[4 % S] = dpb(iacc[4 % S], t4, l4);
                            iacc[5 % S] = dpb(iacc[5 % S], t5, l5);
                            iacc[6 % S] = dpb(iacc[6 % S], t6, l6);
                            iacc[7 % S] = dpb(iacc[7 % S], t7, l7);
                            let mut iacc_tot = iacc[S - 1];
                            let mut k = S as isize - 2;
                            while k >= 0 {
                                iacc_tot = _mm256_add_epi32(iacc_tot, iacc[k as usize]);
                                k -= 1;
                            }
                            let d = d_f16(a_blk);
                            let col = col_scale(tile, $scale);
                            if $tail == 0 {
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    _mm256_mul_ps(col, _mm256_set1_ps(d)),
                                    acc,
                                );
                            } else if $tail == 1 {
                                // compute the scale, black-hole it (keeps it live
                                // through a loop-carried mul chain)
                                acc2 = _mm256_mul_ps(acc2, col);
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    _mm256_set1_ps(d),
                                    acc,
                                );
                            } else if $tail == 2 {
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    col,
                                    acc,
                                );
                            } else if $tail == 3 {
                                acc2 = _mm256_mul_ps(acc2, col);
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    col,
                                    acc,
                                );
                            } else if $tail == 4 {
                                // mul-shape only: col * constant -> fma (no d path)
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    _mm256_mul_ps(col, _mm256_set1_ps(0.25)),
                                    acc,
                                );
                            } else if $tail == 5 {
                                // d-path only: set1(d) * constant -> fma (no col)
                                acc = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(iacc_tot),
                                    _mm256_mul_ps(_mm256_set1_ps(d), _mm256_set1_ps(0.25)),
                                    acc,
                                );
                            } else {
                                // d-path blackholed via a non-underflowing add chain;
                                // col used directly
                                acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(d));
                                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_tot), col, acc);
                            }
                            u += 1;
                        }
                        b += U;
                    }
                    while b < nb {
                        let tile = b_ptr.add(b * BLOCK_MXFP4X8_SIZE);
                        let a_blk = vy.add(b * BLOCK_Q8_0_SIZE);
                        let qs = a_blk.add(2);
                        let lhs0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(qs as *const __m128i));
                        let lhs1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(qs.add(16) as *const __m128i));
                        let tileq = tile.add(8);
                        let raw_0123_0 = _mm256_loadu_si256(tileq as *const __m256i);
                        let raw_4567_0 = _mm256_loadu_si256(tileq.add(32) as *const __m256i);
                        let raw_0123_1 = _mm256_loadu_si256(tileq.add(64) as *const __m256i);
                        let raw_4567_1 = _mm256_loadu_si256(tileq.add(96) as *const __m256i);
                        let r0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, m4b));
                        let r4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, m4b));
                        let r0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, m4b));
                        let r4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, m4b));
                        let r0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_0), m4b));
                        let r4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_0), m4b));
                        let r0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_0123_1), m4b));
                        let r4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16::<4>(raw_4567_1), m4b));
                        let t0 = _mm256_blend_epi32::<170>(r0123_0, _mm256_shuffle_epi32::<177>(r4567_0));
                        let t1 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_0), r4567_0);
                        let t2 = _mm256_blend_epi32::<170>(r0123_1, _mm256_shuffle_epi32::<177>(r4567_1));
                        let t3 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_1), r4567_1);
                        let t4 = _mm256_blend_epi32::<170>(r0123_2, _mm256_shuffle_epi32::<177>(r4567_2));
                        let t5 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_2), r4567_2);
                        let t6 = _mm256_blend_epi32::<170>(r0123_3, _mm256_shuffle_epi32::<177>(r4567_3));
                        let t7 = _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(r0123_3), r4567_3);
                        let iacc = dpb(dpb(dpb(dpb(dpb(dpb(dpb(dpb(
                            _mm256_setzero_si256(), t0, _mm256_shuffle_epi32::<0>(lhs0)),
                            t1, _mm256_shuffle_epi32::<85>(lhs0)),
                            t2, _mm256_shuffle_epi32::<170>(lhs0)),
                            t3, _mm256_shuffle_epi32::<255>(lhs0)),
                            t4, _mm256_shuffle_epi32::<0>(lhs1)),
                            t5, _mm256_shuffle_epi32::<85>(lhs1)),
                            t6, _mm256_shuffle_epi32::<170>(lhs1)),
                            t7, _mm256_shuffle_epi32::<255>(lhs1));
                        let d = d_f16(a_blk);
                        let col = col_scale(tile, $scale);
                        if $tail == 0 {
                            acc = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(iacc),
                                _mm256_mul_ps(col, _mm256_set1_ps(d)),
                                acc,
                            );
                        } else if $tail == 1 {
                            acc2 = _mm256_mul_ps(acc2, col);
                            acc = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(iacc),
                                _mm256_set1_ps(d),
                                acc,
                            );
                        } else if $tail == 2 {
                            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), col, acc);
                        } else if $tail == 3 {
                            acc2 = _mm256_mul_ps(acc2, col);
                            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), col, acc);
                        } else if $tail == 4 {
                            acc = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(iacc),
                                _mm256_mul_ps(col, _mm256_set1_ps(0.25)),
                                acc,
                            );
                        } else if $tail == 5 {
                            acc = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(iacc),
                                _mm256_mul_ps(_mm256_set1_ps(d), _mm256_set1_ps(0.25)),
                                acc,
                            );
                        } else {
                            acc2 = _mm256_add_ps(acc2, _mm256_set1_ps(d));
                            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), col, acc);
                        }
                        b += 1;
                    }
                    let mixed = _mm256_mul_ps(acc, acc2);
                    let out = _mm256_permutevar8x32_ps(mixed, finalpermute);
                    _mm256_storeu_ps(s.add(x * 8), out);
                }
            }
        };
    }

    // DIAGNOSTIC ONLY (wrong values): isolate the e8m0 scale path's cost.

    lab_gemv!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
        lab_v_s1_sc_u1, 1, 0, 1
    );
    lab_gemv!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
        lab_l_s1_sc_u1, 1, 1, 1
    );
    // DIAGNOSTIC ONLY (wrong values): the network floor without any e8m0
    // scale — on clean data the production kernel sits ~10% above this.
    lab_gemv!(
        #[target_feature(enable = "avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni")]
        lab_diag_one_scale, 1, 2, 1
    );

    /// A/B all lab variants against the production kernel on the gpt-oss
    /// expert shape: bit-equality for real variants, timing for all.
    #[test]
    #[ignore = "manual: perf9 gemv lab"]
    fn mxfp4_gemv_lab_ab() {
        let (n, nc) = (2880usize, 2880usize);
        // CLEAN data: no denormal-range e8m0 scales (real gpt-oss weights have
        // block scales around 2^-4, never 2^-128 — only the old rand_mxfp4
        // seeds e in {0,1,2} every 16th block, whose col*row products
        // underflow and take FP assists). Gate on an env knob so both
        // regimes can be measured.
        let clean = std::env::var_os("LAB_CLEAN_DATA").is_some();
        let mut src = rand_mxfp4(nc, n, 0x1234_9999);
        if clean {
            for blk in src.chunks_mut(BLOCK_MXFP4_SIZE) {
                if blk[0] < 120 {
                    blk[0] = 120;
                }
            }
        }
        let rep = super::repack_mxfp4_8x8(&src, nc, n);
        let act = lcg_f32(n, 0xabcd_1111);
        let mut q8 = vec![0u8; (n / QK8_0) * BLOCK_Q8_0_SIZE];
        crate::quants::quantize_row_q8_0(&act, bytemuck::cast_slice_mut(&mut q8));

        let mut ref_out = vec![0f32; nc];
        super::gemv_mxfp4_8x8_q8_0(n, &mut ref_out, &rep, &q8, nc);

        let time = |label: &str, f: &mut dyn FnMut() -> f32| {
            let mut best = f64::INFINITY;
            let mut sink = 0f32;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                sink = f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{label:>22}: {:8.1} us/call  [{sink:.4}]", best * 1e6);
            best
        };

        let base = time("prod gemv (vnni)", &mut || {
            super::gemv_mxfp4_8x8_q8_0(n, &mut ref_out, &rep, &q8, nc);
            ref_out.iter().copied().sum()
        });

        let mut out = vec![0f32; nc];
        let mut run = |label: &str, k: unsafe fn(usize, *mut f32, *const u8, *const u8, usize), check: bool| {
            for v in out.iter_mut() {
                *v = f32::NAN;
            }
            // SAFETY: lab kernels honor the production gemv contract.
            unsafe { k(n, out.as_mut_ptr(), rep.as_ptr(), q8.as_ptr(), nc) };
            if check {
                for c in 0..nc {
                    assert_eq!(
                        ref_out[c].to_bits(),
                        out[c].to_bits(),
                        "{label}: lane {c} differs: {} vs {}",
                        ref_out[c], out[c]
                    );
                }
            }
            let mut best = f64::INFINITY;
            let mut sink = 0f32;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                for _ in 0..4 {
                    // SAFETY: same contract.
                    unsafe { k(n, out.as_mut_ptr(), rep.as_ptr(), q8.as_ptr(), nc) };
                }
                sink = out.iter().copied().sum();
                best = best.min(t.elapsed().as_secs_f64() / 4.0);
            }
            println!("{label:>22}: {:8.1} us/call  ({:.3}x)  [{sink:.4}]", best * 1e6, base / best);
        };

        run("lab_v_s1_sc_u1", lab_v_s1_sc_u1, true);
        run("lab_l_s1_sc_u1", lab_l_s1_sc_u1, true);
        run("lab_diag_one_scale", lab_diag_one_scale, false);

        // production gemm on the same weights (nr=128, the pp64 MoE shape)
        let nr = 128usize;
        let act2 = lcg_f32(nr * n, 0x5150_2222);
        let mut q8x4 = vec![0u8; (nr / 4) * (n / QK8_0) * super::BLOCK_Q8_0X4_SIZE];
        super::quantize_mat_q8_0_4x8(&act2, n, nr, &mut q8x4);
        let mut gout = vec![0f32; nr * nc];
        let mut best = f64::INFINITY;
        let mut sink = 0f32;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..3 {
                super::gemm_mxfp4_8x8_q8_0(n, &mut gout, nc, &rep, &q8x4, nr, nc);
            }
            sink = gout[0] + gout[nr * nc - 1];
            best = best.min(t.elapsed().as_secs_f64() / 3.0);
        }
        println!("prod gemm nr=128 ({}): {:8.1} us/call   [{sink:.4}]",
                 if clean { "CLEAN" } else { "rand" }, best * 1e6);
    }
}
