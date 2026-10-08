//! vec_dot.rs — dot products (ports of ggml-cpu/quants.c, vec.cpp and the
//! llamafile tinyBLAS path). Owner: agent I.
//!
//! Two families live here:
//!
//! * `*_generic` — line-by-line ports of the scalar reference kernels in
//!   ggml/src/ggml-cpu/quants.c (the code before the SIMD variants) and of the
//!   f64-accumulating vec.cpp scalar tails. Kept as the reference the
//!   production kernels are diffed against.
//! * the production kernels — the paths the reference binary actually runs
//!   (`-march=native`, GGML_AVX512 off ⇒ arch/x86/quants.c `__AVX2__` bodies,
//!   and vec.cpp's AVX512 branches for f32/f16/bf16). They reproduce the
//!   reference's SIMD lane structure exactly, i.e. `mul_add` for every
//!   `vfmadd*` and the same horizontal-sum trees, so results are bit-for-bit
//!   equal to the reference (see the parity tests at the bottom).
#![allow(non_snake_case)] // keep the C kernel names (quants.rs convention)

use crate::blocks::*;
use half::f16;

/// Reinterpret `nb` blocks starting at `bytes`. The caller guarantees the
/// underlying buffer holds at least `nb * size_of::<B>()` bytes (rows may be
/// slices that run to the end of a larger storage).
#[inline]
fn blocks_of<B>(bytes: &[u8], nb: usize) -> &[B] {
    debug_assert!(nb * std::mem::size_of::<B>() <= bytes.len());
    debug_assert!(bytes.as_ptr() as usize % std::mem::align_of::<B>() == 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const B, nb) }
}

// ===================== simple quants (x * q8_0) =====================

/// verify against ggml_vec_dot_q4_0_q8_0_generic (quants.c:225)

/// cast the first `nb` blocks of `bytes` (rd_rest slices run to arena end)
fn cast_prefix<B: bytemuck::Pod>(bytes: &[u8], nb: usize) -> &[B] {
    bytemuck::cast_slice(&bytes[..nb * size_of::<B>()])
}

pub fn vec_dot_q4_0_q8_0_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);

    let x = blocks_of::<BlockQ4_0>(x, nb);
    let y = blocks_of::<BlockQ8_0>(y, nb);

    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let v0 = (x[ib].qs[j] & 0x0F) as i32 - 8;
            let v1 = (x[ib].qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * y[ib].qs[j] as i32;
            sumi1 += v1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += sumi as f32 * x[ib].d.to_f32() * y[ib].d.to_f32();
    }
    sumf
}

/// verify against ggml_vec_dot_q5_0_q8_0_generic (quants.c:365)
pub fn vec_dot_q5_0_q8_0_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);

    let x = blocks_of::<BlockQ5_0>(x, nb);
    let y = blocks_of::<BlockQ8_0>(y, nb);

    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let qh = u32::from_le_bytes(x[ib].qh);

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..qk / 2 {
            let xh_0 = (((qh & (1u32 << j)) >> j) << 4) as u8;
            let xh_1 = ((qh & (1u32 << (j + 16))) >> (j + 12)) as u8;

            let x0 = ((x[ib].qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((x[ib].qs[j] >> 4) | xh_1) as i32 - 16;

            sumi0 += x0 * y[ib].qs[j] as i32;
            sumi1 += x1 * y[ib].qs[j + qk / 2] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += (x[ib].d.to_f32() * y[ib].d.to_f32()) * sumi as f32;
    }
    sumf
}

/// verify against ggml_vec_dot_q8_0_q8_0_generic (quants.c:451)
/// ggml_vec_dot_q4_1_q8_1 (generic, quants.c) — f16 d/m·d/s float tail
pub fn vec_dot_q4_1_q8_1_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::blocks::{BlockQ4_1, BlockQ8_1, QK8_1};
    let nb = n / QK8_1;
    let xb: &[BlockQ4_1] = cast_prefix(x, nb);
    let yb: &[BlockQ8_1] = cast_prefix(y, nb);
    debug_assert_eq!(xb.len(), nb);
    debug_assert_eq!(yb.len(), nb);
    let mut sumf = 0f32;
    for ib in 0..nb {
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..QK8_1 / 2 {
            let v0 = (xb[ib].qs[j] & 0x0F) as i32;
            let v1 = (xb[ib].qs[j] >> 4) as i32;
            sumi0 += v0 * yb[ib].qs[j] as i32;
            sumi1 += v1 * yb[ib].qs[j + QK8_1 / 2] as i32;
        }
        let sumi = sumi0 + sumi1;
        sumf += (xb[ib].d.to_f32() * yb[ib].d.to_f32()) * sumi as f32
            + xb[ib].m.to_f32() * yb[ib].s.to_f32();
    }
    sumf
}

/// ggml_vec_dot_q5_1_q8_1 (generic, quants.c)
pub fn vec_dot_q5_1_q8_1_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::blocks::{BlockQ5_1, BlockQ8_1, QK8_1};
    let nb = n / QK8_1;
    let xb: &[BlockQ5_1] = cast_prefix(x, nb);
    let yb: &[BlockQ8_1] = cast_prefix(y, nb);
    debug_assert_eq!(xb.len(), nb);
    debug_assert_eq!(yb.len(), nb);
    let mut sumf = 0f32;
    for ib in 0..nb {
        let qh = u32::from_le_bytes(xb[ib].qh);
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..QK8_1 / 2 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = (xb[ib].qs[j] & 0xF) as i32 | xh_0;
            let x1 = (xb[ib].qs[j] >> 4) as i32 | xh_1;
            sumi0 += x0 * yb[ib].qs[j] as i32;
            sumi1 += x1 * yb[ib].qs[j + QK8_1 / 2] as i32;
        }
        let sumi = sumi0 + sumi1;
        sumf += (xb[ib].d.to_f32() * yb[ib].d.to_f32()) * sumi as f32
            + xb[ib].m.to_f32() * yb[ib].s.to_f32();
    }
    sumf
}

pub fn vec_dot_q8_0_q8_0_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);

    let x = blocks_of::<BlockQ8_0>(x, nb);
    let y = blocks_of::<BlockQ8_0>(y, nb);

    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi = 0i32;
        for j in 0..qk {
            sumi += x[ib].qs[j] as i32 * y[ib].qs[j] as i32;
        }
        sumf += sumi as f32 * (x[ib].d.to_f32() * y[ib].d.to_f32());
    }
    sumf
}

// ===================== K-quants (x * q8_K) =====================

/// verify against ggml_vec_dot_q2_K_q8_K_generic (quants.c:565)
pub fn vec_dot_q2_K_q8_K_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);

    let x = blocks_of::<BlockQ2K>(x, nb);
    let y = blocks_of::<BlockQ8K>(y, nb);

    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q2 = &x[i].qs;
        let q8 = &y[i].qs;
        let sc = &x[i].scales;

        let mut summs = 0i32;
        for j in 0..16 {
            summs += y[i].bsums[j] as i32 * (sc[j] >> 4) as i32;
        }

        let dall = y[i].d * x[i].d.to_f32();
        let dmin = y[i].d * x[i].dmin.to_f32();

        let mut isum = 0i32;
        let mut is = 0usize;
        let mut q8i = 0usize; // q8 advances by 32 per 16-value group
        let mut q2i = 0usize; // q2 advances by 32 per 128-value group
        for _k in 0..QK_K / 128 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let mut d = (sc[is] & 0xF) as i32;
                is += 1;
                let mut isuml = 0i32;
                for l in 0..16 {
                    isuml += q8[q8i + l] as i32 * ((q2[q2i + l] >> shift) & 3) as i32;
                }
                isum += d * isuml;
                d = (sc[is] & 0xF) as i32;
                is += 1;
                isuml = 0i32;
                for l in 16..32 {
                    isuml += q8[q8i + l] as i32 * ((q2[q2i + l] >> shift) & 3) as i32;
                }
                isum += d * isuml;
                shift += 2;
                q8i += 32;
            }
            q2i += 32;
        }
        sumf += dall * isum as f32 - dmin * summs as f32;
    }
    sumf
}

/// verify against ggml_vec_dot_q3_K_q8_K_generic (quants.c:617)
pub fn vec_dot_q3_K_q8_K_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);

    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let x = blocks_of::<BlockQ3K>(x, nb);
    let y = blocks_of::<BlockQ8K>(y, nb);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q3 = &x[i].qs;
        let hm = &x[i].hmask;
        let q8 = &y[i].qs;

        let mut aux8 = [0i8; QK_K];
        let mut aux32 = [0i32; 8];

        // unpack 2-bit values + sign bits into aux8 (4 shifts per 128 values)
        let mut a = 0usize;
        let mut m: u8 = 1;
        let mut q3i = 0usize;
        for _j in 0..QK_K / 128 {
            for shift in [0u32, 2, 4, 6] {
                for l in 0..32 {
                    aux8[a + l] = ((q3[q3i + l] >> shift) & 3) as i8;
                }
                for l in 0..32 {
                    aux8[a + l] -= if hm[l] & m != 0 { 0 } else { 4 };
                }
                a += 32;
                m <<= 1;
            }
            q3i += 32;
        }

        // 12 packed 6-bit scales -> 16 int8 scales
        let mut auxs = [0u32; 4];
        auxs[0] = u32::from_le_bytes(x[i].scales[0..4].try_into().unwrap());
        auxs[1] = u32::from_le_bytes(x[i].scales[4..8].try_into().unwrap());
        auxs[2] = u32::from_le_bytes(x[i].scales[8..12].try_into().unwrap());
        let tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        auxs[0] = (auxs[0] & KMASK2) | ((tmp & KMASK1) << 4);
        auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales: [i8; 16] = bytemuck::cast(auxs);

        let mut q8i = 0usize;
        let mut ai = 0usize;
        let mut aux16 = [0i16; 8];
        for j in 0..QK_K / 16 {
            for _ in 0..2 {
                for l in 0..8 {
                    aux16[l] = q8[q8i + l] as i16 * aux8[ai + l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += (scales[j] as i32 - 32) * aux16[l] as i32;
                }
                q8i += 8;
                ai += 8;
            }
        }

        let d = x[i].d.to_f32() * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

/// Decode the 12-byte packed 6-bit scales of Q4_K/Q5_K into (scales[8], mins[8])
/// verify against ggml_vec_dot_q4_K_q8_K_generic (quants.c:736-741)
#[inline]
fn decode_q4k_scales(packed: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
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

    let bytes: [u8; 16] = bytemuck::cast(utmp);
    let scales: [u8; 8] = bytes[0..8].try_into().unwrap();
    let mins: [u8; 8] = bytes[8..16].try_into().unwrap();
    (scales, mins)
}

/// verify against ggml_vec_dot_q4_K_q8_K_generic (quants.c:696)
pub fn vec_dot_q4_K_q8_K_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);

    let x = blocks_of::<BlockQ4K>(x, nb);
    let y = blocks_of::<BlockQ8K>(y, nb);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].qs;
        let q8 = &y[i].qs;

        let mut aux8 = [0i8; QK_K];
        let mut aux32 = [0i32; 8];

        // unpack nibbles: low halves first, then high halves, per 64 values
        let mut a = 0usize;
        let mut q4i = 0usize;
        for _j in 0..QK_K / 64 {
            for l in 0..32 {
                aux8[a + l] = (q4[q4i + l] & 0xF) as i8;
            }
            a += 32;
            for l in 0..32 {
                aux8[a + l] = (q4[q4i + l] >> 4) as i8;
            }
            a += 32;
            q4i += 32;
        }

        let (scales, mins) = decode_q4k_scales(&x[i].scales);

        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            sumi += y[i].bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut q8i = 0usize;
        let mut ai = 0usize;
        let mut aux16 = [0i16; 8];
        let mut is = 0usize;
        for _j in 0..QK_K / 32 {
            let scale = scales[is] as i32;
            is += 1;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = q8[q8i + l] as i16 * aux8[ai + l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8i += 8;
                ai += 8;
            }
        }

        let d = x[i].d.to_f32() * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = x[i].dmin.to_f32() * y[i].d;
        sumf -= dmin * sumi as f32;
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

/// verify against ggml_vec_dot_q5_K_q8_K_generic (quants.c:771)
pub fn vec_dot_q5_K_q8_K_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);

    let x = blocks_of::<BlockQ5K>(x, nb);
    let y = blocks_of::<BlockQ8K>(y, nb);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].qs;
        let hm = &x[i].qh;
        let q8 = &y[i].qs;

        let mut aux8 = [0i8; QK_K];
        let mut aux32 = [0i32; 8];

        let mut a = 0usize;
        let mut m: u8 = 1;
        let mut q4i = 0usize;
        for _j in 0..QK_K / 64 {
            for l in 0..32 {
                aux8[a + l] = (q4[q4i + l] & 0xF) as i8;
            }
            for l in 0..32 {
                aux8[a + l] += if hm[l] & m != 0 { 16 } else { 0 };
            }
            a += 32;
            m <<= 1;
            for l in 0..32 {
                aux8[a + l] = (q4[q4i + l] >> 4) as i8;
            }
            for l in 0..32 {
                aux8[a + l] += if hm[l] & m != 0 { 16 } else { 0 };
            }
            a += 32;
            m <<= 1;
            q4i += 32;
        }

        let (scales, mins) = decode_q4k_scales(&x[i].scales);

        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            sumi += y[i].bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut q8i = 0usize;
        let mut ai = 0usize;
        let mut aux16 = [0i16; 8];
        let mut is = 0usize;
        for _j in 0..QK_K / 32 {
            let scale = scales[is] as i32;
            is += 1;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = q8[q8i + l] as i16 * aux8[ai + l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8i += 8;
                ai += 8;
            }
        }

        let d = x[i].d.to_f32() * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        let dmin = x[i].dmin.to_f32() * y[i].d;
        sumf -= dmin * sumi as f32;
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

/// verify against ggml_vec_dot_q6_K_q8_K_generic (quants.c:851)
pub fn vec_dot_q6_K_q8_K_generic(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);

    let x = blocks_of::<BlockQ6K>(x, nb);
    let y = blocks_of::<BlockQ8K>(y, nb);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let q4 = &x[i].ql;
        let qh = &x[i].qh;
        let q8 = &y[i].qs;

        let mut aux8 = [0i8; QK_K];
        let mut aux32 = [0i32; 8];

        let mut a = 0usize;
        let mut q4i = 0usize;
        let mut qhi = 0usize;
        for _j in 0..QK_K / 128 {
            for l in 0..32 {
                aux8[a + l] = ((((q4[q4i + l] & 0xF) | (((qh[qhi + l]) & 3) << 4)) as i32) - 32) as i8;
                aux8[a + l + 32] = ((((q4[q4i + l + 32] & 0xF) | (((qh[qhi + l] >> 2) & 3) << 4)) as i32) - 32) as i8;
                aux8[a + l + 64] = ((((q4[q4i + l] >> 4) | (((qh[qhi + l] >> 4) & 3) << 4)) as i32) - 32) as i8;
                aux8[a + l + 96] = ((((q4[q4i + l + 32] >> 4) | (((qh[qhi + l] >> 6) & 3) << 4)) as i32) - 32) as i8;
            }
            a += 128;
            q4i += 64;
            qhi += 32;
        }

        let mut q8i = 0usize;
        let mut ai = 0usize;
        let mut aux16 = [0i16; 8];
        let mut is = 0usize;
        for _j in 0..QK_K / 16 {
            let scale = x[i].scales[is] as i32;
            is += 1;
            for _ in 0..2 {
                for l in 0..8 {
                    aux16[l] = q8[q8i + l] as i16 * aux8[ai + l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8i += 8;
                ai += 8;
            }
        }

        let d = x[i].d.to_f32() * y[i].d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

// ======================================================================
// AVX2 lane-order quantized kernels — the production path.
//
// The reference .so is compiled with -march=native (host has AVX512, but
// GGML_AVX512 is OFF), so `arch/x86/quants.c` takes its
// `#if defined(__AVX2__)` bodies: every quantized type accumulates in ONE
// __m256 (8 f32 lanes) with one `_mm256_fmadd_ps` per block and reduces with
// `hsum_float_8`. Which element lands in which lane therefore decides the
// final rounding, and that differs from the generic scalar ports above.
// These kernels reproduce the AVX2 lane assignment exactly, with
// `f32::mul_add` for every `vfmadd*`.
//
// Ground truth: parity/mulmat_ref.bin (production graph mul_mat, ne11 = 3) and
// parity/mulmat_ref_c1.bin (ne11 = 1) — see mulmat_tests.
// ======================================================================

/// hsum_float_8 (arch/x86/quants.c:43) — the AVX2 horizontal add tree:
/// `xv + xl` (elementwise), then movhl→vaddps (stride 2), movshdup→vaddss.
#[inline]
fn hsum8(a: &[f32; 8]) -> f32 {
    let mut r = [0f32; 4];
    for i in 0..4 {
        r[i] = a[i + 4] + a[i];
    }
    let s0 = r[0] + r[2];
    let s1 = r[1] + r[3];
    s0 + s1
}

/// The elements of `bytes_from_nibbles_32`: low nibbles of `qs[0..16]` land in
/// elements 0..15, the high nibbles in elements 16..31.
#[inline]
fn nibble32(qs: &[u8; 16], e: usize, bias: i32) -> i32 {
    let v = if e < 16 { qs[e] & 0x0F } else { qs[e - 16] >> 4 };
    v as i32 + bias
}

/// ggml_vec_dot_q4_0_q8_0 (arch/x86/quants.c:701, __AVX2__): per block
/// `acc = fmadd(x.d*y.d, madd4(qx-8, qy), acc)`, lane l = elements 4l..4l+3.
///
/// Dispatches to the AVX2 body (`simd_x86`) when the host has it; the scalar
/// lane-port below is the same arithmetic and the non-AVX2 fallback.
pub fn vec_dot_q4_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q4_0_q8_0(n, x, y);
    }
    vec_dot_q4_0_q8_0_scalar(n, x, y)
}

pub fn vec_dot_q4_0_q8_0_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);
    let xb: &[BlockQ4_0] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for ib in 0..nb {
        let d = xb[ib].d.to_f32() * yb[ib].d.to_f32();
        for l in 0..8 {
            let mut q = 0i32;
            for i in 0..4 {
                let e = 4 * l + i;
                q += nibble32(&xb[ib].qs, e, -8) * yb[ib].qs[e] as i32;
            }
            acc[l] = d.mul_add(q as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

/// ggml_vec_dot_q4_1_q8_1 (arch/x86/quants.c:859, __AVX2__): unsigned nibbles
/// (no bias), `acc = fmadd(d0*d1, madd4(qx, qy), acc)`; the `m*s` term is a
/// separate scalar `vfmadd231ss` chain added after `hsum_float_8`.
pub fn vec_dot_q4_1_q8_1(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q4_1_q8_1(n, x, y);
    }
    vec_dot_q4_1_q8_1_scalar(n, x, y)
}

pub fn vec_dot_q4_1_q8_1_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::blocks::{BlockQ4_1, BlockQ8_1, QK8_1};
    let nb = n / QK8_1;
    let xb: &[BlockQ4_1] = cast_prefix(x, nb);
    let yb: &[BlockQ8_1] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    let mut summs = 0f32;
    for ib in 0..nb {
        let d0d1 = xb[ib].d.to_f32() * yb[ib].d.to_f32();
        summs = xb[ib].m.to_f32().mul_add(yb[ib].s.to_f32(), summs);
        for l in 0..8 {
            let mut q = 0i32;
            for i in 0..4 {
                let e = 4 * l + i;
                q += nibble32(&xb[ib].qs, e, 0) * yb[ib].qs[e] as i32;
            }
            acc[l] = d0d1.mul_add(q as f32, acc[l]);
        }
    }
    hsum8(&acc) + summs
}

/// ggml_vec_dot_q5_0_q8_0 (arch/x86/quants.c:1142, __AVX2__): nibble plus the
/// 5th bit from `qh` (bit `e` of the little-endian u32) then -16.
pub fn vec_dot_q5_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q5_0_q8_0(n, x, y);
    }
    vec_dot_q5_0_q8_0_scalar(n, x, y)
}

pub fn vec_dot_q5_0_q8_0_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);
    let xb: &[BlockQ5_0] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for ib in 0..nb {
        let d = xb[ib].d.to_f32() * yb[ib].d.to_f32();
        let qh = u32::from_le_bytes(xb[ib].qh);
        for l in 0..8 {
            let mut q = 0i32;
            for i in 0..4 {
                let e = 4 * l + i;
                let lo = if e < 16 { xb[ib].qs[e] & 0x0F } else { xb[ib].qs[e - 16] >> 4 };
                let v = (lo as i32 | (((qh >> e) & 1) as i32) << 4) - 16;
                q += v * yb[ib].qs[e] as i32;
            }
            acc[l] = d.mul_add(q as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

/// ggml_vec_dot_q5_1_q8_1 (arch/x86/quants.c:1222, __AVX2__): unsigned 5-bit
/// values (nibble | bit<<4), `acc = fmadd(madd4(qx,qy), dx*dy, acc)`.
pub fn vec_dot_q5_1_q8_1(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q5_1_q8_1(n, x, y);
    }
    vec_dot_q5_1_q8_1_scalar(n, x, y)
}

pub fn vec_dot_q5_1_q8_1_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::blocks::{BlockQ5_1, BlockQ8_1, QK8_1};
    let nb = n / QK8_1;
    let xb: &[BlockQ5_1] = cast_prefix(x, nb);
    let yb: &[BlockQ8_1] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    let mut summs = 0f32;
    for ib in 0..nb {
        let dxdy = xb[ib].d.to_f32() * yb[ib].d.to_f32();
        summs = xb[ib].m.to_f32().mul_add(yb[ib].s.to_f32(), summs);
        let qh = u32::from_le_bytes(xb[ib].qh);
        for l in 0..8 {
            let mut q = 0i32;
            for i in 0..4 {
                let e = 4 * l + i;
                let lo = if e < 16 { xb[ib].qs[e] & 0x0F } else { xb[ib].qs[e - 16] >> 4 };
                let v = lo as i32 | (((qh >> e) & 1) as i32) << 4;
                q += v * yb[ib].qs[e] as i32;
            }
            acc[l] = dxdy.mul_add(q as f32, acc[l]);
        }
    }
    hsum8(&acc) + summs
}

/// ggml_vec_dot_q8_0_q8_0 (arch/x86/quants.c:1308, __AVX2__): lane l covers
/// the 4 contiguous qs elements 4l..4l+3 (no nibble unpacking).
pub fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q8_0_q8_0(n, x, y);
    }
    vec_dot_q8_0_q8_0_scalar(n, x, y)
}

pub fn vec_dot_q8_0_q8_0_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let qk = QK8_0;
    let nb = n / qk;
    debug_assert_eq!(n % qk, 0);
    let xb: &[BlockQ8_0] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for ib in 0..nb {
        let d = xb[ib].d.to_f32() * yb[ib].d.to_f32();
        for l in 0..8 {
            let mut q = 0i32;
            for i in 0..4 {
                let e = 4 * l + i;
                q += xb[ib].qs[e] as i32 * yb[ib].qs[e] as i32;
            }
            acc[l] = d.mul_add(q as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

/// ggml_vec_dot_q2_K_q8_K (arch/x86/quants.c:1574, __AVX2__): per block the min
/// term is fma-added FIRST (`madd_epi16(mins, bsums)` over 8 lanes), then the
/// 2-bit term; lane l of a 32-value shift group covers elements 4l..4l+3,
/// scaled by `sc[8j + 2k + (l>=4)]`.
pub fn vec_dot_q2_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockQ2K] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for i in 0..nb {
        let d = yb[i].d * xb[i].d.to_f32();
        let dmin = -yb[i].d * xb[i].dmin.to_f32();

        // `_mm256_madd_epi16(mins, bsums)`: lane l pairs bsums 2l, 2l+1.
        for l in 0..8 {
            let prod = (xb[i].scales[2 * l] >> 4) as i32 * yb[i].bsums[2 * l] as i32
                + (xb[i].scales[2 * l + 1] >> 4) as i32 * yb[i].bsums[2 * l + 1] as i32;
            acc[l] = dmin.mul_add(prod as f32, acc[l]);
        }

        let mut sumi = [0i32; 8];
        for j in 0..QK_K / 128 {
            for k in 0..4 {
                for l in 0..8 {
                    let sc = (xb[i].scales[8 * j + 2 * k + usize::from(l >= 4)] & 0x0F) as i32;
                    let mut s = 0i32;
                    for ii in 0..4 {
                        let e = 4 * l + ii;
                        let q2 = ((xb[i].qs[32 * j + e] >> (2 * k)) & 3) as i32;
                        s += q2 * yb[i].qs[128 * j + 32 * k + e] as i32;
                    }
                    sumi[l] += sc * s;
                }
            }
        }
        for l in 0..8 {
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

/// ggml_vec_dot_q3_K_q8_K (arch/x86/quants.c:1766, __AVX2__): the 6-bit scales
/// are biased -32 and the hmask correction is folded in as `-4*q8` (the AVX2
/// subtracts `maddubs(4·h, q8)` from `maddubs(lo2, q8)`).
pub fn vec_dot_q3_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockQ3K] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for i in 0..nb {
        let d = yb[i].d * xb[i].d.to_f32();
        let scales = q3k_scales_avx2(&xb[i].scales);

        let mut sumi = [0i32; 8];
        for j in 0..QK_K / 128 {
            for k in 0..4 {
                let bit = 4 * j + k;
                for l in 0..8 {
                    let sc = scales[8 * j + 2 * k + usize::from(l >= 4)] as i32;
                    let mut s = 0i32;
                    for ii in 0..4 {
                        let e = 4 * l + ii;
                        let lo = ((xb[i].qs[32 * j + e] >> (2 * k)) & 3) as i32;
                        let v = if xb[i].hmask[e] & (1 << bit) != 0 { lo } else { lo - 4 };
                        s += v * yb[i].qs[128 * j + 32 * k + e] as i32;
                    }
                    sumi[l] += sc * s;
                }
            }
        }
        for l in 0..8 {
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

/// The 16 signed 6-bit scales of Q3_K as the AVX2 kernel builds them
/// (`_mm_set_epi32` of the four de-packed words, then `- 32` in int8).
#[inline]
fn q3k_scales_avx2(packed: &[u8; 12]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let aux0 = u32::from_le_bytes(packed[0..4].try_into().unwrap());
    let aux1 = u32::from_le_bytes(packed[4..8].try_into().unwrap());
    let aux2 = u32::from_le_bytes(packed[8..12].try_into().unwrap());

    // _mm_set_epi32(e3, e2, e1, e0) — e0 lands in the low 32 bits.
    let e0 = (aux0 & KMASK2) | ((aux2 & KMASK1) << 4);
    let e1 = (aux1 & KMASK2) | (((aux2 >> 2) & KMASK1) << 4);
    let e2 = ((aux0 >> 4) & KMASK2) | (((aux2 >> 4) & KMASK1) << 4);
    let e3 = ((aux1 >> 4) & KMASK2) | (((aux2 >> 6) & KMASK1) << 4);
    let bytes: [u8; 16] = bytemuck::cast([e0, e1, e2, e3]);
    let mut scales = [0i8; 16];
    for (l, sc) in scales.iter_mut().enumerate() {
        *sc = bytes[l].wrapping_sub(32) as i8;
    }
    scales
}

/// ggml_vec_dot_q4_K_q8_K (arch/x86/quants.c:2038, __AVX2__): the min term
/// accumulates in a separate 4-lane `acc_m` (reduced last), the 6-bit term in
/// the 8-lane `acc`; lane l covers elements 4l..4l+3 of each 32-value
/// half-chunk with scale `sc[2j]` (low nibbles) / `sc[2j+1]` (high nibbles).
pub fn vec_dot_q4_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q4_K_q8_K(n, x, y);
    }
    vec_dot_q4_K_q8_K_scalar(n, x, y)
}

pub fn vec_dot_q4_K_q8_K_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockQ4K] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    let mut acc_m = [0f32; 4];
    for i in 0..nb {
        let d = yb[i].d * xb[i].d.to_f32();
        let dmin = -yb[i].d * xb[i].dmin.to_f32();
        let (scales, mins) = decode_q4k_scales(&xb[i].scales);

        // `_mm_hadd_epi16` of bsums, then madd against mins[0..8].
        for l in 0..4 {
            let q8s_lo = yb[i].bsums[4 * l] as i32 + yb[i].bsums[4 * l + 1] as i32;
            let q8s_hi = yb[i].bsums[4 * l + 2] as i32 + yb[i].bsums[4 * l + 3] as i32;
            let prod = mins[2 * l] as i32 * q8s_lo + mins[2 * l + 1] as i32 * q8s_hi;
            acc_m[l] = dmin.mul_add(prod as f32, acc_m[l]);
        }

        let mut sumi = [0i32; 8];
        for j in 0..QK_K / 64 {
            let sc_lo = scales[2 * j] as i32;
            let sc_hi = scales[2 * j + 1] as i32;
            for l in 0..8 {
                let mut a = 0i32;
                let mut b = 0i32;
                for ii in 0..4 {
                    let e = 4 * l + ii;
                    a += (xb[i].qs[32 * j + e] & 0x0F) as i32 * yb[i].qs[64 * j + e] as i32;
                    b += (xb[i].qs[32 * j + e] >> 4) as i32 * yb[i].qs[64 * j + 32 + e] as i32;
                }
                sumi[l] += sc_lo * a + sc_hi * b;
            }
        }
        for l in 0..8 {
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }
    // acc_m = add_ps(acc_m, movehl(acc_m)) then add_ss with movehdup.
    let m01 = [acc_m[0] + acc_m[2], acc_m[1] + acc_m[3]];
    hsum8(&acc) + (m01[0] + m01[1])
}

/// ggml_vec_dot_q5_K_q8_K (arch/x86/quants.c:2216, __AVX2__): same lane shape
/// as q4_K with the 5th bit (bit `e/32` of `qh[e]`) added, and the min term
/// horizontally folded to one int then accumulated as a scalar `vfmadd231ss`.
pub fn vec_dot_q5_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q5_K_q8_K(n, x, y);
    }
    vec_dot_q5_K_q8_K_scalar(n, x, y)
}

pub fn vec_dot_q5_K_q8_K_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockQ5K] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    let mut summs = 0f32;
    for i in 0..nb {
        let d = yb[i].d * xb[i].d.to_f32();
        let dmin = -yb[i].d * xb[i].dmin.to_f32();
        let (scales, mins) = decode_q4k_scales(&xb[i].scales);

        // prod (4 lanes) horizontally folded with `hadd_epi32` → one scalar.
        let mut m = 0i32;
        for l in 0..4 {
            let q8s_lo = yb[i].bsums[4 * l] as i32 + yb[i].bsums[4 * l + 1] as i32;
            let q8s_hi = yb[i].bsums[4 * l + 2] as i32 + yb[i].bsums[4 * l + 3] as i32;
            m += mins[2 * l] as i32 * q8s_lo + mins[2 * l + 1] as i32 * q8s_hi;
        }
        let mf = m as f32;
        summs = mf.mul_add(dmin, summs);

        let mut sumi = [0i32; 8];
        for j in 0..QK_K / 64 {
            let sc_lo = scales[2 * j] as i32;
            let sc_hi = scales[2 * j + 1] as i32;
            let bit_lo = 2 * j; // one bit per 32-value group, hmask <<= 1
            let bit_hi = 2 * j + 1;
            for l in 0..8 {
                let mut a = 0i32;
                let mut b = 0i32;
                for ii in 0..4 {
                    let e = 4 * l + ii;
                    let lo = (xb[i].qs[32 * j + e] & 0x0F) as i32
                        | (((xb[i].qh[e] >> bit_lo) & 1) as i32) << 4;
                    let hi = (xb[i].qs[32 * j + e] >> 4) as i32
                        | (((xb[i].qh[e] >> bit_hi) & 1) as i32) << 4;
                    a += lo * yb[i].qs[64 * j + e] as i32;
                    b += hi * yb[i].qs[64 * j + 32 + e] as i32;
                }
                sumi[l] += sc_lo * a + sc_hi * b;
            }
        }
        for l in 0..8 {
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }
    hsum8(&acc) + summs
}

/// ggml_vec_dot_q6_K_q8_K (arch/x86/quants.c:2426, __AVX2__): the four 32-value
/// streams are lane-split 4-at-a-time with scale `sc[8j+2k+(l>=4)]`; the -32
/// bias is NOT folded per element but subtracted once as `32*bsums*scales`.
pub fn vec_dot_q6_K_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        return crate::simd_x86::vec_dot_q6_K_q8_K(n, x, y);
    }
    vec_dot_q6_K_q8_K_scalar(n, x, y)
}

pub fn vec_dot_q6_K_q8_K_scalar(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockQ6K] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut acc = [0f32; 8];
    for i in 0..nb {
        let d = yb[i].d * xb[i].d.to_f32();
        let (ql, qh) = (&xb[i].ql, &xb[i].qh);
        let sc = &xb[i].scales;

        let mut sumi = [0i32; 8];
        for j in 0..QK_K / 128 {
            for k in 0..4 {
                for ii in 0..32 {
                    let val: i32 = match k {
                        0 => ((ql[64 * j + ii] & 0x0F) | ((qh[32 * j + ii] & 0x03) << 4)) as i32,
                        1 => {
                            ((ql[64 * j + 32 + ii] & 0x0F)
                                | (((qh[32 * j + ii] >> 2) & 0x03) << 4)) as i32
                        }
                        2 => ((ql[64 * j + ii] >> 4) | (qh[32 * j + ii] & 0x30)) as i32,
                        _ => {
                            ((ql[64 * j + 32 + ii] >> 4) | ((qh[32 * j + ii] & 0xC0) >> 2)) as i32
                        }
                    };
                    let s = sc[8 * j + 2 * k + usize::from(ii >= 16)] as i32;
                    sumi[ii / 4] += s * yb[i].qs[128 * j + 32 * k + ii] as i32 * val;
                }
            }
        }
        // sumi -= q8sclsub = `slli_epi32(madd_epi16(bsums, scales), 5)`.
        for l in 0..8 {
            let sub = (yb[i].bsums[2 * l] as i32 * sc[2 * l] as i32
                + yb[i].bsums[2 * l + 1] as i32 * sc[2 * l + 1] as i32)
                << 5;
            sumi[l] -= sub;
            acc[l] = d.mul_add(sumi[l] as f32, acc[l]);
        }
    }
    hsum8(&acc)
}

// ======================================================================
// Q1_0 / Q2_0 / NVFP4 / IQ family — the AUDIT_ggml.md §5-A.1 wiring round
// (2026-09-27).
//
// The reference build compiles ggml/src/ggml-cpu/arch/x86/quants.c with
// -march=native and GGML_AVX512 off, so these types take the `#if
// defined(__AVX2__)` bodies: one __m256i accumulator of 8 i32 lanes plus one
// __m256 of 8 f32 lanes, `_mm256_fmadd_ps` per block and `hsum_float_8` at the
// end. The kernels below reproduce that lane structure with `[i32; 8]` /
// `[f32; 8]` and `f32::mul_add` for every `vfmadd*` — the same convention the
// q2_K/q3_K lane ports above use. The integer parts are exact, so only the
// f32 accumulation order decides the bits; the per-lane element grouping
// (i32 lane l = vector elements 4l..4l+3) is what the maddubs/madd pairs
// impose.
//
// Q2_0 is the one exception: arch/x86/quants.c has NO ggml_vec_dot_q2_0_q8_0,
// so arch-fallback.h maps it to the generic scalar kernel — that is what the
// reference dispatches on every x86 build, and the port keeps the scalar form.
//
// Ground truth: parity/vecdot3_ref.bin (parity/ref_vecdot_dump3.c, dlsym of
// the kernels from libggml-cpu.so.0 — see vecdot3_tests at the bottom).
//
// NOTE (tiled/, sync batch D): since the iqp removal the reference dispatches
// the `tiled` K-quant matmul for every supported type at batch >= 8
// (ggml-cpu.c:1269 / :1678 — ported 1:1 in tiled.rs); batches < 8 (decode)
// and tensors the tiled gate refuses still fall through to these vec_dot
// kernels, exactly like the reference. The old iqp.cpp panel (deleted
// upstream) was never ported — PARITY.md's former ➖ lane is superseded by
// the tiled port.
// ======================================================================

/// ggml-common.h:1132 `IQ1S_DELTA` (iq1_m reuses it as IQ1M_DELTA, :1133).
const IQ1S_DELTA: f32 = 0.125;

/// ggml_vec_dot_q1_0_q8_0 (arch/x86/quants.c:555, __AVX2__): QK1_0 = 128 maps
/// one Q1_0 block onto FOUR q8_0 blocks; per K the sign bits of `qs32[K]`
/// conditionally negate the q8 lanes (`sy = (qy ^ sm) - sm`), the +1/-1 sum
/// reduces through maddubs+madd into 8 i32 lanes, and the q8_0 scales chain
/// with one `vfmadd` each (a plain `vmul` for K = 0) before the single fma by
/// the Q1_0 `d`.
pub fn vec_dot_q1_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK1_0;
    debug_assert_eq!(n % QK1_0, 0);
    let xb: &[BlockQ1_0] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb * 4);

    let mut acc = [0f32; 8];
    for ib in 0..nb {
        let d0 = xb[ib].d.to_f32();
        let mut acc_block = [0f32; 8];
        for k in 0..4usize {
            let yk = &yb[ib * 4 + k];
            let dy = yk.d.to_f32();
            let qs32 = u32::from_le_bytes(xb[ib].qs[4 * k..4 * k + 4].try_into().unwrap());
            let mut s32 = [0i32; 8];
            for l in 0..8 {
                let mut s = 0i32;
                for i in 0..4 {
                    let e = 4 * l + i;
                    let q = yk.qs[e] as i32;
                    s += if (qs32 >> e) & 1 != 0 { q } else { -q };
                }
                s32[l] = s;
            }
            if k == 0 {
                // K = 0 is a plain `vmulps` (quants.c:593), not an fma
                for l in 0..8 {
                    acc_block[l] = dy * s32[l] as f32;
                }
            } else {
                for l in 0..8 {
                    acc_block[l] = dy.mul_add(s32[l] as f32, acc_block[l]);
                }
            }
        }
        for l in 0..8 {
            acc[l] = d0.mul_add(acc_block[l], acc[l]);
        }
    }
    hsum8(&acc)
}

/// ggml_vec_dot_q2_0_q8_0_generic (quants.c:177) — verbatim scalar port: this
/// is the kernel the x86 reference actually dispatches (arch/x86/quants.c has
/// no q2_0 variant; arch-fallback.h maps the name to `_generic`). One Q2_0
/// block (QK2_0 = 64) maps onto two q8_0 blocks.
pub fn vec_dot_q2_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    let nb = n / QK2_0;
    debug_assert_eq!(n % QK2_0, 0);
    let xb: &[BlockQ2_0] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb * 2);

    let mut sumf = 0f32;
    for i in 0..nb {
        let d0 = xb[i].d.to_f32();
        let mut sumi = 0f32;
        for k in 0..2usize {
            let yb2 = &yb[i * 2 + k];
            let d1 = yb2.d.to_f32();
            let mut sumi_block = 0i32;
            for b in 0..8usize {
                let byte = xb[i].qs[k * 8 + b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_block += (((byte >> 0) & 3) as i32 - 1) * yb2.qs[b * 4] as i32;
                sumi_block += (((byte >> 2) & 3) as i32 - 1) * yb2.qs[b * 4 + 1] as i32;
                sumi_block += (((byte >> 4) & 3) as i32 - 1) * yb2.qs[b * 4 + 2] as i32;
                sumi_block += (((byte >> 6) & 3) as i32 - 1) * yb2.qs[b * 4 + 3] as i32;
            }
            sumi += d1 * sumi_block as f32;
        }
        sumf += d0 * sumi;
    }
    sumf
}

/// ggml_vec_dot_nvfp4_q8_0 (arch/x86/quants.c:1004, __AVX2__): QK_NVFP4 = 64
/// maps one NVFP4 block (4 UE4M3 sub-block scales + 32 packed E2M1 nibbles)
/// onto two q8_0 blocks; `mul_add_epi8` (sign-splitting maddubs) times the
/// nibble LUT `kvalues_fp4`, one fma per sub-block pair with the sub-block
/// scale in lanes 0-3 / 4-7.
pub fn vec_dot_nvfp4_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants::{ue4m3_to_fp32, KVALUES_MXFP4};
    let nb = n / QK_NVFP4;
    debug_assert_eq!(n % QK_NVFP4, 0);
    let xb: &[BlockNvfp4] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb * 2);

    let mut accum = [0f32; 8];
    for ib in 0..nb {
        // dequant value of element e of the first/second 32-value half; the
        // unpacklo/unpackhi_epi64 reordering in the C lands the nibbles in
        // exactly this natural order: within each 16-value sub-block the low
        // nibbles of the 8 qs bytes are elements 0..8, the high nibbles 8..16
        let val = |half: usize, e: usize| -> i32 {
            let j = e % 16;
            let byte = xb[ib].qs[half * 16 + 8 * (e / 16) + j % 8];
            let nib = if j < 8 { byte & 0x0F } else { byte >> 4 };
            KVALUES_MXFP4[nib as usize] as i32
        };
        let dy0 = yb[2 * ib].d.to_f32();
        let dy1 = yb[2 * ib + 1].d.to_f32();
        // scales01 = [s0 x4 | s1 x4], scales23 = [s2 x4 | s3 x4] (quants.c:1057)
        let s = [
            ue4m3_to_fp32(xb[ib].d[0]) * dy0,
            ue4m3_to_fp32(xb[ib].d[1]) * dy0,
            ue4m3_to_fp32(xb[ib].d[2]) * dy1,
            ue4m3_to_fp32(xb[ib].d[3]) * dy1,
        ];
        let mut p1 = [0i32; 8];
        let mut p2 = [0i32; 8];
        for l in 0..8 {
            let e0 = 4 * l; // p_1 lanes 0-3 = sub-block 0, lanes 4-7 = sub-block 1
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for i in 0..4 {
                let e = e0 + i;
                s1 += val(0, e) * yb[2 * ib].qs[e] as i32;
                s2 += val(1, e) * yb[2 * ib + 1].qs[e] as i32;
            }
            p1[l] = s1;
            p2[l] = s2;
        }
        // accum = fmadd(scales01, p_1, accum); then fmadd(scales23, p_2, …)
        for l in 0..8 {
            let sc = if l < 4 { s[0] } else { s[1] };
            accum[l] = sc.mul_add(p1[l] as f32, accum[l]);
        }
        for l in 0..8 {
            let sc = if l < 4 { s[2] } else { s[3] };
            accum[l] = sc.mul_add(p2[l] as f32, accum[l]);
        }
    }
    hsum8(&accum)
}

/// ggml_vec_dot_iq2_xxs_q8_K (arch/x86/quants.c:2660, __AVX2__): per 32-value
/// group the 4 grid u64s come from `iq2xxs_grid[qs-bytes]`, the signs from
/// `keven_signs_q2xs` (the ±1 view of `ksigns_iq2xs` — byte j of group idx is
/// -1 iff bit j of `ksigns_iq2xs[idx]` is set) taken from bits 7l..7l+7 of
/// aux32[odd], and the block scale is `2*(aux32[odd] >> 28)+1`; final
/// `0.125 * hsum`.
pub fn vec_dot_iq2_xxs_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::{IQ2XXS_GRID, KSIGNS_IQ2XS};
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq2Xxs] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accumf = [0f32; 8];
    for i in 0..nb {
        let d = xb[i].d.to_f32() * yb[i].d;
        let mut sumi = [0i32; 8]; // sumi1+sumi2 (integer-exact fold)
        let mut q8i = 0usize;
        for it in 0..QK_K / 64 {
            let w = |k: usize| xb[i].qs[8 * it + k] as u32;
            let aux32 = [w(0) | (w(1) << 16), w(2) | (w(3) << 16), w(4) | (w(5) << 16), w(6) | (w(7) << 16)];
            let aux8 = |j: usize| ((aux32[j / 4] >> (8 * (j % 4))) & 0xFF) as usize;
            for h in 0..2usize {
                let signs_word = aux32[1 + 2 * h];
                let ls = 2 * (signs_word >> 28) as i32 + 1;
                for l in 0..8usize {
                    let mut s = 0i32;
                    for t in 0..4usize {
                        let e = 4 * l + t;
                        let grid = ((IQ2XXS_GRID[aux8(8 * h + e / 8)] >> (8 * (e % 8))) & 0xFF) as i32;
                        let sgn = if (KSIGNS_IQ2XS[((signs_word >> (7 * (e / 8))) & 127) as usize] >> (e % 8)) & 1
                            != 0
                        {
                            -1i32
                        } else {
                            1
                        };
                        s += grid * sgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += ls * s;
                }
                q8i += 32;
            }
        }
        for l in 0..8 {
            accumf[l] = d.mul_add(sumi[l] as f32, accumf[l]);
        }
    }
    0.125f32 * hsum8(&accumf)
}

/// ggml_vec_dot_iq2_xs_q8_K (arch/x86/quants.c:2778, __AVX2__): grid index =
/// `qs[u16] & 511`, sign byte = `ksigns_iq2xs[qs >> 9]` (the AVX2 rebuilds it
/// from bits 9..15 plus the odd-parity bit via k_bit_helper — the same
/// even-parity encoding), scale pair = nibbles of `scales[block]` (the
/// nibble-interleaved `scales` vector + `get_scale_shuffle` select bytes
/// {2*block, 2*block+1}); final `0.125 * hsum`.
pub fn vec_dot_iq2_xs_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::{IQ2XS_GRID, KSIGNS_IQ2XS};
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq2Xs] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accumf = [0f32; 8];
    for i in 0..nb {
        let d = xb[i].d.to_f32() * yb[i].d;
        let mut sumi = [0i32; 8];
        let mut q8i = 0usize;
        for it in 0..QK_K / 128 {
            for c in 0..4usize {
                let b = 4 * it + c; // the 32-value block = its scale byte pair
                let sc_lo = 2 * (xb[i].scales[b] & 0x0F) as i32 + 1;
                let sc_hi = 2 * (xb[i].scales[b] >> 4) as i32 + 1;
                for l in 0..8usize {
                    let sc = if l < 4 { sc_lo } else { sc_hi };
                    let mut s = 0i32;
                    for t in 0..4usize {
                        let e = 4 * l + t;
                        let q2 = xb[i].qs[16 * it + 4 * c + e / 8] as u32;
                        let grid = ((IQ2XS_GRID[(q2 & 511) as usize] >> (8 * (e % 8))) & 0xFF) as i32;
                        let sgn = if (KSIGNS_IQ2XS[((q2 >> 9) & 127) as usize] >> (e % 8)) & 1 != 0 {
                            -1i32
                        } else {
                            1
                        };
                        s += grid * sgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += sc * s;
                }
                q8i += 32;
            }
        }
        for l in 0..8 {
            accumf[l] = d.mul_add(sumi[l] as f32, accumf[l]);
        }
    }
    0.125f32 * hsum8(&accumf)
}

/// ggml_vec_dot_iq2_s_q8_K (arch/x86/quants.c:3075, __AVX2__): grid index =
/// `qs[byte] | (qh[block] << (8-2k) & 0x300)` per u64 lane k, signs are the
/// four bytes at `qs[QK_K/8 + 4*block + l]` (the second half of the qs array),
/// scales = nibbles of `scales[block]` via `get_scale_shuffle_k4`; final
/// `0.125 * hsum`.
pub fn vec_dot_iq2_s_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::IQ2S_GRID;
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq2S] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accumf = [0f32; 8];
    for i in 0..nb {
        let d = xb[i].d.to_f32() * yb[i].d;
        let mut sumi = [0i32; 8];
        let mut q8i = 0usize;
        for t in 0..QK_K / 64 {
            for h in 0..2usize {
                let v = 2 * t + h; // 32-value block
                let sc_lo = 2 * (xb[i].scales[v] & 0x0F) as i32 + 1;
                let sc_hi = 2 * (xb[i].scales[v] >> 4) as i32 + 1;
                for l in 0..8usize {
                    let sc = if l < 4 { sc_lo } else { sc_hi };
                    let mut s = 0i32;
                    for tt in 0..4usize {
                        let e = 4 * l + tt;
                        let k = e / 8;
                        let idx = (xb[i].qs[8 * t + 4 * h + k] as usize)
                            | (((xb[i].qh[v] as usize) << (8 - 2 * k)) & 0x300);
                        let grid = ((IQ2S_GRID[idx] >> (8 * (e % 8))) & 0xFF) as i32;
                        let sign_byte = xb[i].qs[QK_K / 8 + 8 * t + 4 * h + k];
                        let sgn = if sign_byte & (1 << (e % 8)) != 0 { -1i32 } else { 1 };
                        s += grid * sgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += sc * s;
                }
                q8i += 32;
            }
        }
        for l in 0..8 {
            accumf[l] = d.mul_add(sumi[l] as f32, accumf[l]);
        }
    }
    0.125f32 * hsum8(&accumf)
}

/// ggml_vec_dot_iq3_xxs_q8_K (arch/x86/quants.c:3260, __AVX2__): grid index
/// bytes `qs[0..QK_K/4]`, sign words `qs[QK_K/4 + 4*block + h]` (ls in the top
/// nibble), `iq3xxs_grid` entries are u32s (4 values each); final
/// `0.25 * hsum`.
pub fn vec_dot_iq3_xxs_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::{IQ3XXS_GRID, KSIGNS_IQ2XS};
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq3Xxs] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accumf = [0f32; 8];
    for i in 0..nb {
        let d = xb[i].d.to_f32() * yb[i].d;
        let mut sumi = [0i32; 8];
        let mut q8i = 0usize;
        for it in 0..QK_K / 64 {
            for h in 0..2usize {
                let gas = QK_K / 4 + 8 * it + 4 * h;
                let signs_word = u32::from_le_bytes(xb[i].qs[gas..gas + 4].try_into().unwrap());
                let ls = 2 * (signs_word >> 28) as i32 + 1;
                for l in 0..8usize {
                    let mut s = 0i32;
                    for t in 0..4usize {
                        let e = 4 * l + t;
                        let idx = xb[i].qs[16 * it + 8 * h + e / 4] as usize;
                        let grid = ((IQ3XXS_GRID[idx] >> (8 * (e % 4))) & 0xFF) as i32;
                        let sgn = if (KSIGNS_IQ2XS[((signs_word >> (7 * (e / 8))) & 127) as usize] >> (e % 8)) & 1
                            != 0
                        {
                            -1i32
                        } else {
                            1
                        };
                        s += grid * sgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += ls * s;
                }
                q8i += 32;
            }
        }
        for l in 0..8 {
            accumf[l] = d.mul_add(sumi[l] as f32, accumf[l]);
        }
    }
    0.25f32 * hsum8(&accumf)
}

/// ggml_vec_dot_iq3_s_q8_K (arch/x86/quants.c:3384, __AVX2__): grid index =
/// `qs[byte] | (qh[block] << (k+1) & 256)` per u32 lane k (8 lanes of 4
/// values), signs = the four `signs[block*4 + h*2 + l]` bytes, scale =
/// `2*(scales[block/2] >> 4*(block%2)) + 1` uniform per block; plain hsum.
pub fn vec_dot_iq3_s_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::IQ3S_GRID;
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq3S] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accumf = [0f32; 8];
    for i in 0..nb {
        let d = xb[i].d.to_f32() * yb[i].d;
        let mut sumi = [0i32; 8];
        let mut q8i = 0usize;
        let mut qsi = 0usize;
        for t in 0..QK_K / 64 {
            for h in 0..2usize {
                let v = 2 * t + h; // 32-value block
                let qhv = xb[i].qh[v] as u32;
                let ls = if h == 0 {
                    2 * (xb[i].scales[t] & 0x0F) as i32 + 1
                } else {
                    2 * (xb[i].scales[t] >> 4) as i32 + 1
                };
                for l in 0..8usize {
                    let mut s = 0i32;
                    for tt in 0..4usize {
                        let e = 4 * l + tt;
                        let k = e / 4;
                        // `_mm256_set_epi32(1,2,…,8)` puts 8 in lane 0 — the
                        // shift for byte k is 8-k, matching the generic's
                        // `qh << (8-2l)` / `qh << (7-2l)` pair (quants.c:1120)
                        let idx = (xb[i].qs[qsi + 8 * h + k] as u32) | ((qhv << (8 - k as u32)) & 256);
                        let grid = ((IQ3S_GRID[idx as usize] >> (8 * (e % 4))) & 0xFF) as i32;
                        let sign_byte = xb[i].signs[8 * t + 4 * h + e / 8];
                        let sgn = if sign_byte & (1 << (e % 8)) != 0 { -1i32 } else { 1 };
                        s += grid * sgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += ls * s;
                }
                q8i += 32;
            }
            qsi += 16;
        }
        for l in 0..8 {
            accumf[l] = d.mul_add(sumi[l] as f32, accumf[l]);
        }
    }
    hsum8(&accumf)
}

/// ggml_vec_dot_iq1_s_q8_K (arch/x86/quants.c:3594, __AVX2__): grid index =
/// `qs[byte] | (qh bits 3k..3k+3) << 8` per u64 lane k of ±1 bytes — the
/// `-march=native` build takes the `#ifdef __BMI2__` `_pdep_u64` body, which
/// the fallback's `{<<8, <<5, <<2, >>1}` shifts reproduce exactly; block
/// scale `2*((qh >> 12) & 7)+1`, plus a scalar delta term
/// `(bsums pair) * ±1 * ls` accumulated in `accum1`; final
/// `hsum + IQ1S_DELTA * accum1`.
pub fn vec_dot_iq1_s_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::IQ1S_GRID;
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq1S] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accum = [0f32; 8];
    let mut accum1 = 0f32;
    for i in 0..nb {
        let mut sumi = [0i32; 8];
        let mut sumi1 = 0i32;
        let mut q8i = 0usize;
        let mut qsi = 0usize;
        for t in 0..QK_K / 64 {
            for j in 0..2usize {
                let qhw = xb[i].qh[2 * t + j];
                let ls = 2 * ((qhw >> 12) & 7) as i32 + 1;
                for l in 0..8usize {
                    let mut s = 0i32;
                    for tt in 0..4usize {
                        let e = 4 * l + tt;
                        let k = e / 8;
                        // lane k's index takes qh bits 3k..3k+3: shifts
                        // {<<8, <<5, <<2, >>1} (quants.c:3628 — identical to
                        // the BMI2 `_pdep_u64` form the .so actually runs)
                        let shift = 8 - 3 * k as i32;
                        let qhi = if shift >= 0 { (qhw as u32) << shift } else { qhw as u32 >> 1 };
                        let idx = (xb[i].qs[qsi + 4 * j + k] as u32) | (qhi & 0x700);
                        let v = ((IQ1S_GRID[idx as usize] >> (8 * (e % 8))) & 0xFF) as u8 as i8 as i32;
                        s += v * yb[i].qs[q8i + e] as i32;
                    }
                    sumi[l] += ls * s;
                }
                let sg = if qhw & 0x8000 != 0 { -1i32 } else { 1 };
                sumi1 += (yb[i].bsums[4 * t + 2 * j] as i32 + yb[i].bsums[4 * t + 2 * j + 1] as i32)
                    * sg
                    * ls;
                q8i += 32;
            }
            qsi += 8;
        }
        let d = yb[i].d * xb[i].d.to_f32();
        for l in 0..8 {
            accum[l] = d.mul_add(sumi[l] as f32, accum[l]);
        }
        // the reference build contracts `accum1 += d * sumi1` into a fused
        // multiply-add (vfmadd231ss in the shipped libggml-cpu.so,
        // ggml_vec_dot_iq1_s_q8_K) — a plain mul+add diverges by 1 ulp on
        // data-dependent rounding (found by the batch-D tiled artifact's
        // stock-path sections, 2026-10-01)
        accum1 = d.mul_add(sumi1 as f32, accum1);
    }
    hsum8(&accum) + IQ1S_DELTA * accum1
}

/// ggml_vec_dot_iq1_m_q8_K (arch/x86/quants.c:3713, __AVX2__; the BMI2 and
/// non-BMI2 bodies agree — the port follows the byte-cast form):
/// grid indices `qs[byte] | (qh[byte 2j + k/2] << (8-4*(k%2)) & 0x700)` on
/// lane k; the 16 group scales are the 3-bit fields of the four `scales`
/// u16s (srlv shift lanes {0,6,3,9} = `_mm256_set_epi64x(9,3,6,0)` lane0..3),
/// so 32-block 2t+j uses `2*((sc[t] >> (6j)) & 7)+1` for elements
/// 0..16 and `2*((sc[t] >> (6j+3)) & 7)+1` for 16..32; the delta ±1 comes from
/// bits 3/7 of `qh[4t+2j + (lane/2)]`; two separate 8-lane accumulators with
/// final `hsum + IQ1M_DELTA * hsum`. `d = y.d * fp16(scale_u16)` with the
/// nibble-assembled u16 reinterpreted as an f16.
pub fn vec_dot_iq1_m_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::IQ1S_GRID;
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq1M] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accum1 = [0f32; 8];
    let mut accum2 = [0f32; 8];
    for i in 0..nb {
        let sc = |m: usize| u16::from_le_bytes([xb[i].scales[2 * m], xb[i].scales[2 * m + 1]]);
        let scale_u16 =
            (sc(0) >> 12) | ((sc(1) >> 8) & 0x00f0) | ((sc(2) >> 4) & 0x0f00) | (sc(3) & 0xf000);
        let d = yb[i].d * f16::from_bits(scale_u16).to_f32();
        let mut sumi1 = [0i32; 8];
        let mut sumi2 = [0i32; 8];
        let mut q8i = 0usize;
        let mut qsi = 0usize;
        let mut qhi = 0usize;
        for t in 0..QK_K / 64 {
            let sct = sc(t);
            let scales_v = [
                [2 * (sct & 7) as i32 + 1, 2 * ((sct >> 3) & 7) as i32 + 1],
                [2 * ((sct >> 6) & 7) as i32 + 1, 2 * ((sct >> 9) & 7) as i32 + 1],
            ];
            for j in 0..2usize {
                for l in 0..8usize {
                    let half = usize::from(l >= 4);
                    let scl = scales_v[j][half];
                    let mut s = 0i32;
                    let mut sdelta = 0i32;
                    for tt in 0..4usize {
                        let e = 4 * l + tt;
                        let k = e / 8;
                        // qh is bytes here: `(uint16_t)qh[2j + k/2] << …`
                        // casts the byte value, it is not a u16 load
                        let qhb = xb[i].qh[qhi + 2 * j + k / 2] as u32;
                        let idx = (xb[i].qs[qsi + 4 * j + k] as u32) | ((qhb << (8 - 4 * (k % 2))) & 0x700);
                        let v = ((IQ1S_GRID[idx as usize] >> (8 * (e % 8))) & 0xFF) as u8 as i8 as i32;
                        s += v * yb[i].qs[q8i + e] as i32;
                        // delta lane: qh byte (4t+2j + lane/2), bit 3 for even
                        // lanes, bit 7 for odd ones (quants.c:3783-3790)
                        let dbyte = xb[i].qh[qhi + 2 * j + k / 2];
                        let dbit = if k % 2 == 0 { 0x08 } else { 0x80 };
                        let dsgn = if dbyte & dbit != 0 { -1i32 } else { 1 };
                        sdelta += dsgn * yb[i].qs[q8i + e] as i32;
                    }
                    sumi1[l] += scl * s;
                    sumi2[l] += scl * sdelta;
                }
                q8i += 32;
            }
            qsi += 8;
            qhi += 4;
        }
        for l in 0..8 {
            accum1[l] = d.mul_add(sumi1[l] as f32, accum1[l]);
            accum2[l] = d.mul_add(sumi2[l] as f32, accum2[l]);
        }
    }
    hsum8(&accum1) + IQ1S_DELTA * hsum8(&accum2)
}

/// ggml_vec_dot_iq4_nl_q8_0 (arch/x86/quants.c:3920, __AVX2__): block pairs —
/// two separate 8-lane accumulators, one fma per block with
/// `d = y.d * x.d`, elementwise `add_ps` before the single hsum, and the
/// scalar `sumf += d*(sumi1+sumi2)` tail for an odd block count.
pub fn vec_dot_iq4_nl_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::KVALUES_IQ4NL;
    let nb = n / QK4_NL;
    debug_assert_eq!(n % QK4_NL, 0);
    let xb: &[BlockIq4Nl] = cast_prefix(x, nb);
    let yb: &[BlockQ8_0] = cast_prefix(y, nb);

    let mut accum1 = [0f32; 8];
    let mut accum2 = [0f32; 8];
    let mut ib = 0usize;
    let lane = |j: usize, l: usize| -> i32 {
        let mut s = 0i32;
        for t in 0..4usize {
            let e = 4 * l + t;
            let nib = if e < 16 {
                (xb[j].qs[e] & 0x0F) as usize
            } else {
                (xb[j].qs[e - 16] >> 4) as usize
            };
            s += KVALUES_IQ4NL[nib] as i32 * yb[j].qs[e] as i32;
        }
        s
    };
    while ib + 1 < nb {
        let d0 = yb[ib].d.to_f32() * xb[ib].d.to_f32();
        let d1 = yb[ib + 1].d.to_f32() * xb[ib + 1].d.to_f32();
        for l in 0..8 {
            accum1[l] = d0.mul_add(lane(ib, l) as f32, accum1[l]);
            accum2[l] = d1.mul_add(lane(ib + 1, l) as f32, accum2[l]);
        }
        ib += 2;
    }
    let mut sumf = {
        let mut a = [0f32; 8];
        for l in 0..8 {
            a[l] = accum1[l] + accum2[l];
        }
        hsum8(&a)
    };
    // odd tail (quants.c:3992-4000)
    for j in ib..nb {
        let d = yb[j].d.to_f32() * xb[j].d.to_f32();
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j2 in 0..QK4_NL / 2 {
            sumi1 += yb[j].qs[j2] as i32 * KVALUES_IQ4NL[(xb[j].qs[j2] & 0xf) as usize] as i32;
            sumi2 += yb[j].qs[j2 + QK4_NL / 2] as i32
                * KVALUES_IQ4NL[(xb[j].qs[j2] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

/// ggml_vec_dot_iq4_xs_q8_K (arch/x86/quants.c:4004, __AVX2__): the iq4_nl
/// lane shape with per-32-value-block 6-bit scales
/// `((scales_l[b/2] >> 4(b%2)) | (sh << (4-2(b%2)) & 0x30)) - 32` where `sh`
/// walks 4 bits of `scales_h` per block pair; single accumulator.
pub fn vec_dot_iq4_xs_q8_K(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::quants_k::KVALUES_IQ4NL;
    let nb = n / QK_K;
    debug_assert_eq!(n % QK_K, 0);
    let xb: &[BlockIq4Xs] = cast_prefix(x, nb);
    let yb: &[BlockQ8K] = cast_prefix(y, nb);

    let mut accum = [0f32; 8];
    for ibl in 0..nb {
        let d = xb[ibl].d.to_f32() * yb[ibl].d;
        let mut sumi = [0i32; 8];
        let mut q8i = 0usize;
        let mut qsi = 0usize;
        for t in 0..QK_K / 64 {
            let sh = (xb[ibl].scales_h >> (4 * t)) as u32;
            let ls = [
                ((xb[ibl].scales_l[t] & 0xf) as i32 | ((sh << 4) & 0x30) as i32) - 32,
                ((xb[ibl].scales_l[t] >> 4) as i32 | ((sh << 2) & 0x30) as i32) - 32,
            ];
            for h in 0..2usize {
                for l in 0..8usize {
                    let mut s = 0i32;
                    for tt in 0..4usize {
                        let e = 4 * l + tt;
                        let nib = if e < 16 {
                            (xb[ibl].qs[qsi + e] & 0x0F) as usize
                        } else {
                            (xb[ibl].qs[qsi + e - 16] >> 4) as usize
                        };
                        s += KVALUES_IQ4NL[nib] as i32 * yb[ibl].qs[q8i + e] as i32;
                    }
                    sumi[l] += ls[h] * s;
                }
                q8i += 32;
                qsi += 16;
            }
        }
        for l in 0..8 {
            accum[l] = d.mul_add(sumi[l] as f32, accum[l]);
        }
    }
    hsum8(&accum)
}

// ===================== unquantized types =====================

/// verify against ggml_vec_dot_f32 scalar tail (vec.cpp:11; ggml_float = f64 accumulation)
pub fn vec_dot_f32(n: usize, x: &[f32], y: &[f32]) -> f32 {
    debug_assert!(x.len() >= n);
    debug_assert!(y.len() >= n);
    let mut sumf = 0.0f64;
    for i in 0..n {
        sumf += (x[i] * y[i]) as f64;
    }
    sumf as f32
}

/// verify against ggml_vec_dot_f16 scalar tail (vec.cpp:264)
pub fn vec_dot_f16(n: usize, x: &[f16], y: &[f16]) -> f32 {
    let mut sumf = 0.0f64;
    for i in 0..n {
        sumf += (x[i].to_f32() * y[i].to_f32()) as f64;
    }
    sumf as f32
}

/// verify against ggml_vec_dot_bf16 scalar tail (vec.cpp:139)
pub fn vec_dot_bf16(n: usize, x: &[half::bf16], y: &[half::bf16]) -> f32 {
    let mut sumf = 0.0f64;
    for i in 0..n {
        sumf += (x[i].to_f32() * y[i].to_f32()) as f64;
    }
    sumf as f32
}

/// The vec_dot companion type for a weight type, mirroring
/// `type_traits_cpu[].vec_dot_type` (ggml-cpu.c). F16/BF16 intentionally differ
/// from C: we keep activations in f32 and widen the weight row instead of
/// quantizing activations down to f16 (see compute::forward_mul_mat).
#[inline]
pub fn vec_dot_type(ty: crate::types::GgmlType) -> Option<crate::types::GgmlType> {
    use crate::types::GgmlType::*;
    Some(match ty {
        F32 => F32,
        F16 => F16,
        Bf16 => Bf16,
        Q1_0 | Q2_0 | Q4_0 | Q5_0 | Q8_0 | Nvfp4 | Iq4Nl => Q8_0,
        Q4_1 | Q5_1 => crate::types::GgmlType::Q8_1,
        Q2K | Q3K | Q4K | Q5K | Q6K => Q8K,
        // ggml-cpu.c:407-516: every IQ type but IQ4_NL pairs with q8_K
        Iq2Xxs | Iq2Xs | Iq2S | Iq3Xxs | Iq3S | Iq1S | Iq1M | Iq4Xs => Q8K,
        // TQ1_0/TQ2_0: refused — see vec_dot_row.
        _ => return None,
    })
}

/// Dispatch a single row dot product `x · y` for weight type `ty`.
/// `x` is `n` weight values of type `ty` (already widened to f32 rows for
/// F16/BF16 by the caller), `y` is the converted activation row
/// (type = vec_dot_type(ty)).
pub fn vec_dot_row(ty: crate::types::GgmlType, n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::types::GgmlType::*;
    match ty {
        // F32: ggml_vec_dot_f32 — the AVX512 FMA-chain emulation, exactly the
        // dot the reference's mul_mat_id one_chunk runs (type_traits_cpu
        // .vec_dot, ggml-cpu.c:218). The f64 scalar twin (`vec_dot_f32`) is
        // NOT bit-equal and drifted deepseek4's MoE experts (arch batch 7).
        F32 => vec_dot_f32_c(n, bytemuck::cast_slice(x), bytemuck::cast_slice(y)),
        Bf16 => vec_dot_f32(n, bytemuck::cast_slice(x), bytemuck::cast_slice(y)),
        F16 => vec_dot_f32(n, bytemuck::cast_slice(x), bytemuck::cast_slice(y)),
        Q4_0 => vec_dot_q4_0_q8_0(n, x, y),
        Q4_1 => vec_dot_q4_1_q8_1(n, x, y),
        Q5_0 => vec_dot_q5_0_q8_0(n, x, y),
        Q5_1 => vec_dot_q5_1_q8_1(n, x, y),
        Q8_0 => vec_dot_q8_0_q8_0(n, x, y),
        Q2K => vec_dot_q2_K_q8_K(n, x, y),
        Q3K => vec_dot_q3_K_q8_K(n, x, y),
        Q4K => vec_dot_q4_K_q8_K(n, x, y),
        Q5K => vec_dot_q5_K_q8_K(n, x, y),
        Q6K => vec_dot_q6_K_q8_K(n, x, y),
        Q1_0 => vec_dot_q1_0_q8_0(n, x, y),
        Q2_0 => vec_dot_q2_0_q8_0(n, x, y),
        Nvfp4 => vec_dot_nvfp4_q8_0(n, x, y),
        Iq2Xxs => vec_dot_iq2_xxs_q8_K(n, x, y),
        Iq2Xs => vec_dot_iq2_xs_q8_K(n, x, y),
        Iq2S => vec_dot_iq2_s_q8_K(n, x, y),
        Iq3Xxs => vec_dot_iq3_xxs_q8_K(n, x, y),
        Iq3S => vec_dot_iq3_s_q8_K(n, x, y),
        Iq1S => vec_dot_iq1_s_q8_K(n, x, y),
        Iq1M => vec_dot_iq1_m_q8_K(n, x, y),
        Iq4Nl => vec_dot_iq4_nl_q8_0(n, x, y),
        Iq4Xs => vec_dot_iq4_xs_q8_K(n, x, y),
        // Documented refusal (AUDIT_ggml.md §5-B.4): the port carries the
        // TQ1_0/TQ2_0 block layouts and type-table rows but none of their
        // chain — no quantizer (ggml-quants.c quantize_row_tq1_0/tq2_0 not
        // ported), no dequantizer (dequantize_row_tq1_0/tq2_0,
        // ggml-quants.c:2428/2467 not ported) and no vec_dot (quants.c:481/533
        // generic, arch/x86/quants.c:1376/1508 AVX2). The reference's own
        // consumers are test fixtures only; a GGUF that does carry ternary
        // weights is refused loudly here instead of computing wrong values.
        Tq1_0 | Tq2_0 => panic!(
            "vec_dot: TQ1_0/TQ2_0 (ternary) weights are not supported — the whole ternary \
             chain is unported (quantize/dequantize/vec_dot; see PARITY.md)"
        ),
        other => unimplemented!("vec_dot_row for {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 28) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Integer accumulation of the quantized kernels must be bit-exact against
    /// an independent straightforward reference implementation.
    #[test]
    fn q4_0_q8_0_bit_exact() {
        let n = 256;
        let xf = lcg(n, 42);
        let nb = n / QK4_0;
        let mut xq = vec![0u8; nb * std::mem::size_of::<BlockQ4_0>()];
        crate::quants::quantize_row_q4_0_ref(&xf, bytemuck::cast_slice_mut(&mut xq));
        let mut yq8 = vec![0u8; nb * std::mem::size_of::<BlockQ8_0>()];
        crate::quants::quantize_row_q8_0(&xf, bytemuck::cast_slice_mut(&mut yq8));

        let xb: &[BlockQ4_0] = bytemuck::cast_slice(&xq);
        let yb: &[BlockQ8_0] = bytemuck::cast_slice(&yq8);
        let mut acc = 0.0f32;
        for ib in 0..nb {
            let mut si = 0i32;
            for j in 0..16 {
                let v0 = (xb[ib].qs[j] & 0xF) as i32 - 8;
                let v1 = (xb[ib].qs[j] >> 4) as i32 - 8;
                si += v0 * yb[ib].qs[j] as i32;
                si += v1 * yb[ib].qs[j + 16] as i32;
            }
            acc += (si as f32 * xb[ib].d.to_f32()) * yb[ib].d.to_f32();
        }
        let ours = vec_dot_q4_0_q8_0_generic(n, &xq, &yq8);
        assert_eq!(ours.to_bits(), acc.to_bits(), "q4_0 dot must be bit exact");
    }

    /// q8_0 × q8_0 kernel: bit-exact vs a straightforward integer reference
    /// (the K-quant kernels are exercised against naive f64 matmuls in
    /// compute.rs's mul_mat tests).
    #[test]
    fn q8_0_q8_0_bit_exact() {
        let n2 = 128;
        let xf2 = lcg(n2, 9);
        let nb2 = n2 / QK8_0;
        let mut a8 = vec![0u8; nb2 * std::mem::size_of::<BlockQ8_0>()];
        let mut b8 = vec![0u8; nb2 * std::mem::size_of::<BlockQ8_0>()];
        crate::quants::quantize_row_q8_0(&xf2, bytemuck::cast_slice_mut(&mut a8));
        crate::quants::quantize_row_q8_0(lcg(n2, 10).as_slice(), bytemuck::cast_slice_mut(&mut b8));
        let ab: &[BlockQ8_0] = bytemuck::cast_slice(&a8);
        let bb: &[BlockQ8_0] = bytemuck::cast_slice(&b8);
        let mut acc = 0.0f32;
        for ib in 0..nb2 {
            let mut si = 0i32;
            for j in 0..32 {
                si += ab[ib].qs[j] as i32 * bb[ib].qs[j] as i32;
            }
            acc += si as f32 * (ab[ib].d.to_f32() * bb[ib].d.to_f32());
        }
        assert_eq!(vec_dot_q8_0_q8_0_generic(n2, &a8, &b8).to_bits(), acc.to_bits());
    }

    #[test]
    fn f32_dot_matches_f64_reference() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin()).collect();
        let y: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut r = 0.0f64;
        for i in 0..x.len() {
            r += (x[i] * y[i]) as f64;
        }
        assert_eq!(vec_dot_f32(x.len(), &x, &y).to_bits(), (r as f32).to_bits());
    }
}


// ======================================================================
// AVX512 lane-order kernels — bit-exact ports of the reference build's
// F32Cx16 paths (STEP 64, EPR 16, ARR 4; simd-mappings.h:533+). Ground truth:
// parity/vec_ref.bin (parity/ref_vec_dump.c against libggml-cpu.so).
// ======================================================================

/// `_mm512_reduce_add_ps` fold order: t[i] += t[i+half] for half in
/// {8,4,2,1} (vextractf64x4 → 128 → 64 → horizontal).
#[inline]
pub fn reduce_add16(x: &[f32; 16]) -> f32 {
    let mut t = *x;
    let mut half = 8;
    while half >= 1 {
        for i in 0..half {
            t[i] += t[i + half];
        }
        half >>= 1;
    }
    t[0]
}

/// ggml_vec_dot_f32 (AVX512): 4×16 lanes, fused multiply-add, then the
/// GGML_F32x16_REDUCE pairwise fold + reduce_add16, then the `n % 64` leftovers.
///
/// The leftovers keep summing into the *f32* `sumf` (vec.cpp declares `float
/// sumf` in the SIMD branch; only the scalar `#else` branch uses `ggml_float`),
/// and GCC 13.3 vectorizes that loop for this build (`-O3 -march=native`, the
/// shipped libggml-cpu.so, `ggml_vec_dot_f32+0x1da`):
///
///   * 16-wide chunks: `vmulps` (products *rounded*, no FMA) then the lane
///     products added into `sumf` one by one in lane order 0..15;
///   * one 8-wide chunk the same way when ≥ 8 remain;
///   * the last 0..7 elements as a scalar `vfmadd231ss` chain (fused).
///
/// So n % 64 == 0 is not the only bit-exact region: this reproduces the tail
/// too. parity/mulmat_q8_bert_ref.bin pins all of it (its F32 attention
/// sections run k = 2/3/4/8/14/16/64).
pub fn vec_dot_f32_c(n: usize, x: &[f32], y: &[f32]) -> f32 {
    let np = n & !63;
    let mut sum = [[0f32; 16]; 4];
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            let s = &mut sum[j];
            for l in 0..16 {
                let k = i + j * 16 + l;
                s[l] = y[k].mul_add(x[k], s[l]);
            }
        }
        i += 64;
    }
    // REDUCE (offset ARR/2=2, then 1)
    let mut a = sum[0];
    for l in 0..16 {
        a[l] += sum[2][l];
    }
    let mut b = sum[1];
    for l in 0..16 {
        b[l] += sum[3][l];
    }
    for l in 0..16 {
        a[l] += b[l];
    }
    let mut res = reduce_add16(&a);
    let mut left = n - i;
    // unfused products, sequential lane-order adds
    while left >= 16 {
        for j in 0..16 {
            let p = x[i + j] * y[i + j];
            res += p;
        }
        i += 16;
        left -= 16;
    }
    if left >= 8 {
        for j in 0..8 {
            let p = x[i + j] * y[i + j];
            res += p;
        }
        i += 8;
        left -= 8;
    }
    // scalar tail, contracted into an FMA
    while left > 0 {
        res = x[i].mul_add(y[i], res);
        i += 1;
        left -= 1;
    }
    res
}

/// ggml_vec_dot_f16 (AVX512 F32Cx16 fallback): halves widened to f32, same
/// lane structure as vec_dot_f32_c.
///
/// NOTE: the reference's mul_mat reaches this kernel only for ne11 < 2
/// (`llamafile_sgemm` bails out below n == 2); with ne11 >= 2 it converts the
/// activations to f16 and hands the whole GEMM to llamafile tinyBLAS, whose
/// per-element accumulation differs (see mulmat_tests).
pub fn vec_dot_f16_c(n: usize, x: &[f16], y: &[f16]) -> f32 {
    let np = n & !63;
    let mut sum = [[0f32; 16]; 4];
    let mut xf = [0f32; 64];
    let mut yf = [0f32; 64];
    let mut i = 0;
    while i < np {
        for k in 0..64 {
            xf[k] = x[i + k].to_f32();
            yf[k] = y[i + k].to_f32();
        }
        for j in 0..4 {
            let s = &mut sum[j];
            for l in 0..16 {
                let k = j * 16 + l;
                s[l] = yf[k].mul_add(xf[k], s[l]);
            }
        }
        i += 64;
    }
    let mut a = sum[0];
    for l in 0..16 {
        a[l] += sum[2][l];
    }
    let mut b = sum[1];
    for l in 0..16 {
        b[l] += sum[3][l];
    }
    for l in 0..16 {
        a[l] += b[l];
    }
    let mut res = reduce_add16(&a) as f64;
    while i < n {
        res += (x[i].to_f32() * y[i].to_f32()) as f64;
        i += 1;
    }
    res as f32
}

/// ggml_vec_dot_bf16 (vec.cpp:139, __AVX512BF16__ — the host/defconfig path):
/// two 16-lane f32 accumulators over 64-value chunks driven by
/// `_mm512_dpbf16_ps`, i.e. lane t takes the bf16 pair (2t, 2t+1) of each
/// 32-value block with VDPBF16PS ordering `fma(a0,b0, fma(a1,b1, acc))`
/// (products are exact, each add rounds once — verified against the
/// instruction over 3e6 random cases). The two lane vectors are then
/// `_mm512_reduce_add_ps`-folded and summed in f64, leftovers in f64 too.
///
/// NOTE: the reference's mul_mat only reaches this function for ne11 < 2
/// (`llamafile_sgemm` bails out below n == 2); for ne11 >= 2 it hands F16/BF16
/// weights to llamafile tinyBLAS instead (see mulmat_tests).
pub fn vec_dot_bf16_c(n: usize, x: &[half::bf16], y: &[half::bf16]) -> f32 {
    let mut c1 = [0f32; 16];
    let mut c2 = [0f32; 16];
    let mut i = 0;
    while i + 64 <= n {
        for t in 0..16 {
            let (a0, a1) = (x[i + 2 * t].to_f32(), x[i + 2 * t + 1].to_f32());
            let (b0, b1) = (y[i + 2 * t].to_f32(), y[i + 2 * t + 1].to_f32());
            c1[t] = a0.mul_add(b0, a1.mul_add(b1, c1[t]));
        }
        for t in 0..16 {
            let (a0, a1) = (x[i + 32 + 2 * t].to_f32(), x[i + 32 + 2 * t + 1].to_f32());
            let (b0, b1) = (y[i + 32 + 2 * t].to_f32(), y[i + 32 + 2 * t + 1].to_f32());
            c2[t] = a0.mul_add(b0, a1.mul_add(b1, c2[t]));
        }
        i += 64;
    }
    let mut sumf = reduce_add16(&c1) as f64;
    sumf += reduce_add16(&c2) as f64;
    while i < n {
        sumf += (x[i].to_f32() * y[i].to_f32()) as f64;
        i += 1;
    }
    sumf as f32
}

/// ggml_vec_soft_max_f32 accumulation order (AVX512): per 16-lane chunk
/// `sum += reduce_add16(exp_chunk)` in f64, leftovers scalar. Writes the exp
/// values to `out` like the C kernel.
pub fn soft_max_sum_c(n: usize, x: &[f32], out: &mut [f32], expf: impl Fn(f32) -> f32 + Copy) -> f32 {
    let mut sum = 0f64;
    let mut i = 0;
    let mut chunk = [0f32; 16];
    while i + 16 <= n {
        for l in 0..16 {
            chunk[l] = expf(x[i + l]);
        }
        out[i..i + 16].copy_from_slice(&chunk);
        sum += reduce_add16(&chunk) as f64;
        i += 16;
    }
    while i < n {
        // C tail = plain libm expf (vec.cpp:593), NOT the SIMD polynomial
        let e = x[i].exp();
        out[i] = e;
        sum += e as f64;
        i += 1;
    }
    sum as f32
}


// ======================================================================
// llamafile tinyBLAS dot products — the reference routes F16/BF16 weights
// with ne11 >= 2 to llamafile_sgemm (llamafile/sgemm.cpp; `if (n < 2) return
// false`), whose gemm_bloc keeps one accumulator lane per vector k-lane:
//   f16 : KN=16, vfmadd → lane t sums k ≡ t (mod 16), hsum = reduce_add16
//   bf16: KN=32, VDPBF16PS → lane t sums pairs (2t, 2t+1) via nested fma
// Bit-exact vs the reference (parity/mulmat_ref.bin); ne11 == 1 uses the
// regular vec_dot_*_c kernels instead.
// ======================================================================

pub fn vec_dot_tinyblas_f16(n: usize, x: &[half::f16], y: &[half::f16]) -> f32 {
    let mut acc = [0f32; 16];
    let mut i = 0;
    while i + 16 <= n {
        for t in 0..16 {
            acc[t] = x[i + t].to_f32().mul_add(y[i + t].to_f32(), acc[t]);
        }
        i += 16;
    }
    reduce_add16(&acc)
}

pub fn vec_dot_tinyblas_bf16(n: usize, x: &[half::bf16], y: &[half::bf16]) -> f32 {
    let mut acc = [0f32; 16];
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..16 {
            let (a0, a1) = (x[i + 2 * t].to_f32(), x[i + 2 * t + 1].to_f32());
            let (b0, b1) = (y[i + 2 * t].to_f32(), y[i + 2 * t + 1].to_f32());
            acc[t] = a0.mul_add(b0, a1.mul_add(b1, acc[t]));
        }
        i += 32;
    }
    reduce_add16(&acc)
}

#[cfg(test)]
mod avx512_tests {
    use super::*;

    /// LCG identical to parity/ref_vec_dump.c
    fn ref_input(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x9e37_79b9;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 28) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Bit-exact parity vs the reference AVX512 kernels (parity/vec_ref.bin).
    #[test]
    fn vec_lane_order_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/vec_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            panic!("missing {path}: build via parity/ref_vec_dump.c");
        };
        let mut c = &bytes[..];
        let mut sections = 0usize;
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x3143_4556, "section magic");
            let kind = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            let (xy, rest) = c.split_at(n * 8);
            c = rest;
            let res_ref = f32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];

            // x,y come straight from the dump (x then y)
            let x: Vec<f32> = xy[..n * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let y: Vec<f32> = xy[n * 4..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

            // Non-multiple-of-64 tails: reference tails are auto-vectorized
            // differently by GCC (1-ulp drift, see PARITY.md). Real inference
            // uses 64-multiple dot lengths (head_dim), so strict-compare those.
            // SOFTMAX (kind 2) runs at odd n in decode (n_kv grows by 1) — do
            // NOT skip it; any mismatch here is a real decode bug.
            if n % 64 != 0 && kind != 2 {
                sections += 1;
                continue;
            }
            let mine = match kind {
                0 => vec_dot_f32_c(n, &x, &y),
                1 => {
                    // PARITY.md: the reference .so drops n%64 leftovers in the
                    // f16 dot (binary-side anomaly, source includes them) and
                    // its SIMD fp32→fp16 leaves tail elements unconverted.
                    // Real inference uses dot lengths == head_dim (64), so
                    // restrict strict comparison to multiples of 64.
                    if n % 64 != 0 {
                        sections += 1;
                        continue;
                    }
                    let xh: Vec<half::f16> = x.iter().map(|v| half::f16::from_f32(*v)).collect();
                    let yh: Vec<half::f16> = y.iter().map(|v| half::f16::from_f32(*v)).collect();
                    vec_dot_f16_c(n, &xh, &yh)
                }
                3 => {
                    if n % 64 != 0 {
                        sections += 1;
                        continue;
                    }
                    let xh: Vec<half::f16> = x
                        .iter()
                        .map(|v| half::f16::from_bits((v.to_bits() & 0xFFFF) as u16))
                        .collect();
                    let yh: Vec<half::f16> = y
                        .iter()
                        .map(|v| half::f16::from_bits((v.to_bits() & 0xFFFF) as u16))
                        .collect();
                    vec_dot_f16_c(n, &xh, &yh)
                }
                2 => {
                    let mut out = vec![0f32; n];
                    soft_max_sum_c(n, &x, &mut out, crate::ops::ggml_expf_v512)
                }
                k => panic!("unknown kind {k}"),
            };
            assert_eq!(
                mine.to_bits(),
                res_ref.to_bits(),
                "kind {kind} n {n}: lane order mismatch (mine {mine} ref {res_ref})"
            );
            sections += 1;
        }
        assert_eq!(sections, 40, "10 lengths x 4 kinds");
    }
}


/// Bit-exact parity vs the reference production mul_mat path
/// (parity/mulmat_ref.bin from parity/ref_mulmat_dump.c: real graph compute
/// through libggml-cpu). Covers quantized weights × f32 activations with the
/// runtime activation quantizers, plus F16/BF16 weights.
#[cfg(test)]
mod mulmat_tests {
    use super::reduce_add16;
    use crate::compute::graph_compute;
    use crate::graph::Graph;
    use crate::tensor::Context;
    use crate::types::GgmlType;
    use half::{bf16, f16};

    /// One section of a VMM1 artifact (see parity/ref_mulmat_dump.c).
    struct Section<'a> {
        ty: GgmlType,
        n: usize,
        rows: usize,
        cols: usize,
        xq: &'a [u8],
        y: &'a [f32],
        dst: &'a [f32],
    }

    fn parse(bytes: &[u8]) -> Vec<Section<'_>> {
        let mut c = bytes;
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x314D_4D56, "VMM1 magic");
            let tid = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            let rows = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            let cols = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            let ty = GgmlType::from_u32(tid).unwrap();
            let (xq, rest) = c.split_at(ty.row_size(n) * rows);
            c = rest;
            let (y, rest) = c.split_at(n * cols * 4);
            c = rest;
            let (dst, rest) = c.split_at(rows * cols * 4);
            c = rest;
            out.push(Section {
                ty,
                n,
                rows,
                cols,
                xq,
                y: bytemuck::cast_slice(y),
                dst: bytemuck::cast_slice(dst),
            });
        }
        out
    }

    /// Run our production mul_mat over one artifact section.
    fn mul_mat(s: &Section<'_>) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(s.ty, s.n as i64, s.rows as i64);
        let b = ctx.new_tensor_2d(GgmlType::F32, s.n as i64, s.cols as i64);
        let d = ctx.mul_mat(a, b);
        for t in [a, b] {
            ctx.arena_resize_tensor(t);
        }
        ctx.data_bytes_mut(a).unwrap().copy_from_slice(s.xq);
        ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(s.y));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        graph_compute(&mut ctx, &mut g, 4);
        bytemuck::cast_slice(ctx.data_bytes(d).unwrap()).to_vec()
    }

    /// Per-element accumulation of llamafile tinyBLAS, which the reference
    /// reaches for F16/BF16 weights whenever ne11 >= 2: `ggml_compute_forward_mul_mat`
    /// converts the activations and then hands the whole GEMM to
    /// `llamafile_sgemm` (llamafile/sgemm.cpp; `if (n < 2) return false` makes
    /// it a no-op for a single column, and its `case GGML_TYPE_F16/BF16`
    /// accepts the converted B). `gemm_bloc` keeps `V Cv[] = {}` — one
    /// accumulator lane per k-lane of the vector type — runs
    /// `Cv = madd(Av, Bv, Cv)` for l = 0, KN, 2KN, ... and ends with
    /// `hsum(Cv)` = `_mm512_reduce_add_ps`:
    ///   f16 : KN = 16, madd = vfmadd_ps → lane t sums k ≡ t (mod 16);
    ///   bf16: KN = 32, madd = VDPBF16PS → lane t sums the pairs (2t, 2t+1)
    ///         with `fma(a0,b0, fma(a1,b1, acc))` per step.
    /// Verified bit-exact against both artifacts (see the tests below).
    fn llamafile_f16_dot(n: usize, x: &[f16], y: &[f16]) -> f32 {
        super::vec_dot_tinyblas_f16(n, x, y)
    }

    fn llamafile_bf16_dot(n: usize, x: &[bf16], y: &[bf16]) -> f32 {
        super::vec_dot_tinyblas_bf16(n, x, y)
    }

    /// The reference's F16/BF16 mul_mat result for the streaming shape, i.e.
    /// the tinyBLAS kernels applied to the dumped weights and the f32→f16/bf16
    /// converted activations (the same conversion `from_float` performs).
    fn llamafile_rows(s: &Section<'_>) -> Vec<f32> {
        let mut out = Vec::with_capacity(s.rows * s.cols);
        for col in 0..s.cols {
            let yrow = &s.y[col * s.n..(col + 1) * s.n];
            for row in 0..s.rows {
                let xrow = &s.xq[row * s.ty.row_size(s.n)..(row + 1) * s.ty.row_size(s.n)];
                out.push(match s.ty {
                    GgmlType::F16 => {
                        let xh: Vec<f16> = xrow
                            .chunks_exact(2)
                            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                            .collect();
                        let yh: Vec<f16> = yrow.iter().map(|&v| f16::from_f32(v)).collect();
                        llamafile_f16_dot(s.n, &xh, &yh)
                    }
                    GgmlType::Bf16 => {
                        let xb: Vec<bf16> = xrow
                            .chunks_exact(2)
                            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                            .collect();
                        let yb: Vec<bf16> = yrow.iter().map(|&v| bf16::from_f32(v)).collect();
                        llamafile_bf16_dot(s.n, &xb, &yb)
                    }
                    other => panic!("llamafile_rows: {other:?}"),
                });
            }
        }
        out
    }

    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0f32, f32::max)
    }

    /// Streaming shape (ne11 = 3, parity/mulmat_ref.bin).
    ///
    /// The ten quantized types are bit-exact through our mul_mat: the
    /// reference rejects them in llamafile_sgemm (`if (Btype != GGML_TYPE_<A>)`
    /// — the activation type is q8_0/q8_1/q8_K, never the weight type), so it
    /// runs the arch/x86/quants.c AVX2 kernels that the lane ports above
    /// reproduce lane-for-lane.
    ///
    /// F16/BF16 are the two types where the reference does NOT use
    /// `ggml_vec_dot_f16/bf16` for this shape: with two or more columns the
    /// whole GEMM goes to llamafile tinyBLAS (see `llamafile_*_dot`), which
    /// compute.rs does not model (it calls vec_dot_f16_c/vec_dot_bf16_c for
    /// every shape — the kernel the reference uses for ne11 = 1). So this test
    /// asserts (a) the tinyBLAS kernels are bit-exact against the reference
    /// and (b) our mul_mat stays within the structural gap, which
    /// `mulmat_c1_bit_exact_vs_reference` pins down at ne11 = 1.
    #[test]
    fn mulmat_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/mulmat_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            panic!("missing {path}: build via parity/ref_mulmat_dump.c");
        };
        let sections = parse(&bytes);
        assert_eq!(sections.len(), 12, "12 types");
        let mut failures = Vec::new();
        let mut checked = 0usize;
        for s in &sections {
            let ours = mul_mat(s);
            let bits = |v: &[f32], s: &Section<'_>| -> bool {
                v.len() == s.dst.len() && v.iter().zip(s.dst).all(|(a, b)| a.to_bits() == b.to_bits())
            };
            match s.ty {
                GgmlType::F16 | GgmlType::Bf16 => {
                    let ref_path = llamafile_rows(s);
                    if !bits(&ref_path, s) {
                        failures.push(format!(
                            "{:?}: llamafile tinyBLAS emulation mismatch (worst rel err {:.2e})",
                            s.ty,
                            rel_err(&ref_path, s.dst)
                        ));
                    }
                    let gap = rel_err(&ours, s.dst);
                    if gap > 1e-6 {
                        failures.push(format!(
                            "{:?}: mul_mat deviates {gap:.2e} from the reference's tinyBLAS path \
                             (expected <= 1e-6 structural gap)",
                            s.ty
                        ));
                    }
                    checked += 1;
                }
                _ => {
                    if !bits(&ours, s) {
                        let worst = rel_err(&ours, s.dst);
                        let (i, a, b) = ours
                            .iter()
                            .zip(s.dst)
                            .enumerate()
                            .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                            .map(|(i, (a, b))| (i, *a, *b))
                            .unwrap();
                        failures.push(format!(
                            "{:?} n {} r {} c {}: worst rel err {worst:.2e}; \
                             first at {i}: mine {a} ({:08x}) ref {b} ({:08x})",
                            s.ty,
                            s.n,
                            s.rows,
                            s.cols,
                            a.to_bits(),
                            b.to_bits()
                        ));
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 12);
        assert!(failures.is_empty(), "mul_mat parity failures:\n{}", failures.join("\n"));
    }

    /// Single-column shape (ne11 = 1, parity/mulmat_ref_c1.bin).
    ///
    /// `llamafile_sgemm` returns false for n < 2, so the reference runs
    /// `ggml_vec_dot_*` for every type, F16/BF16 included. All twelve sections
    /// must therefore be bit-exact through our mul_mat dispatch — this is what
    /// pins `vec_dot_f16_c` / `vec_dot_bf16_c` (and the ten quantized kernels)
    /// to the reference for the decode path.
    #[test]
    fn mulmat_c1_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/mulmat_ref_c1.bin");
        let Ok(bytes) = std::fs::read(path) else {
            panic!("missing {path}: build via parity/ref_mulmat_c1_dump.c");
        };
        let sections = parse(&bytes);
        assert_eq!(sections.len(), 12, "12 types");
        let mut failures = Vec::new();
        for s in &sections {
            let ours = mul_mat(s);
            if ours.len() != s.dst.len() || ours.iter().zip(s.dst).any(|(a, b)| a.to_bits() != b.to_bits()) {
                let worst = rel_err(&ours, s.dst);
                failures.push(format!("{:?}: worst rel err {worst:.2e}", s.ty));
            }
        }
        assert!(failures.is_empty(), "mul_mat (ne11 = 1) parity failures:\n{}", failures.join("\n"));
    }
}

// ======================================================================
// Q5_K kernel verification (agent W, 2026-09-24)
//
// Two independent questions, because "Q5_K is wrong" was the standing
// hypothesis for the granite-hybrid residual (agent T) while the mulmat
// artifacts claim bit-exactness (agent I):
//
//   (a) `q5k_lane_vs_generic_vs_integer_reference` — the pure kernel
//       question: the production lane kernel, the scalar `_generic` port and
//       an *independently transcribed* exact integer reference must agree:
//       the integer accumulators bit-for-bit, the f32 result with `_generic`,
//       and both within the 1-2 ulp float-rounding class of the f64-exact
//       value. A lane/shuffle mistake in the AVX2 port would show up here as
//       a *value* error (wrong scale applied to a byte group), not as ulp.
//
//   (b) `kquant_real_tensor_dump_vs_reference` — the production-path question:
//       `parity/ref_q5k_dump` runs the *reference* graph mul_mat over the real
//       GGUF bytes of granite's Q5_K ffn_{gate,up}_shexp and Q4_K ssm_{in,out}
//       plus gpt-oss Q4_K attn_output, and records whether the CPU_REPACK
//       buffer type gives the tensor a repack trait. Reading tensor->extra back
//       from the real `ggml_repack_get_optimal_repack_type` gate is what
//       settles it: on this x86 host (AVX512, GGML_NATIVE=ON,
//       GGML_USE_CPU_REPACK=ON) the Q5_K/Q6_K branches require
//       `ggml_cpu_has_neon()` (repack.cpp:5050/5061) and come back NULL, i.e.
//       the reference computes Q5_K with exactly the vec_dot kernels the
//       artifacts above pin down, while Q4_K/Q2_K/MXFP4 *do* get a repack
//       instance (the reference prints "repack tensor with q4_K_8x8" at load).
// ======================================================================

#[cfg(test)]
mod q5k_kernel_tests {
    use super::*;
    use bytemuck::Zeroable;
    use crate::blocks::{BlockQ5K, BlockQ8K, K_SCALE_SIZE, QK_K};
    use crate::quants::quantize_row_q8_K_ref;
    use crate::quants_k::quantize_row_q5_K_ref;

    /// LCG shared with the C dumps (parity/ref_mulmat_dump.c).
    fn lcg(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as i32 as f32 / (1u32 << 28) as f32) * scale
            })
            .collect()
    }

    /// Transcription of ggml-quants.c `get_scale_min_k4` (quants.c:736) — kept
    /// separate from `decode_q4k_scales` on purpose: this is the *independent*
    /// decode the kernel's scale/min unpacking is checked against.
    fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
        }
    }

    /// Exact integer terms of one Q5_K block dot, from the independent decode:
    /// per 32-value group g the integer part scaled by `scale_g` and the min
    /// part scaled by `min_g` (both i32, no rounding anywhere).
    struct Terms {
        /// Σ_g scale_g * Σ_{e in g} q5[e]*q8[e]
        iacc: i32,
        /// Σ_g min_g * Σ_{e in g} q8[e]   (the `bsums` term)
        macc: i32,
        /// the same iacc split into the 8 f32 accumulator lanes the AVX2
        /// kernel uses: lane l takes the 4 low-nibble products of bytes 4l..4l+3
        /// with `scale[2j]` and the 4 high-nibble ones with `scale[2j+1]`
        lanes: [i32; 8],
    }

    fn exact_terms(x: &BlockQ5K, y: &BlockQ8K) -> Terms {
        let mut terms = Terms { iacc: 0, macc: 0, lanes: [0i32; 8] };
        for j in 0..QK_K / 64 {
            let (sc_lo, m_lo) = get_scale_min_k4(2 * j, &x.scales);
            let (sc_hi, m_hi) = get_scale_min_k4(2 * j + 1, &x.scales);
            // group 2j: low nibbles of qs[32j..32j+32], qh bit 2j
            // group 2j+1: high nibbles, qh bit 2j+1, paired with q8[32..64]
            let mut g_lo = 0i32;
            let mut g_hi = 0i32;
            let mut bs_lo = 0i32;
            let mut bs_hi = 0i32;
            for e in 0..32 {
                let lo = ((x.qs[32 * j + e] & 0x0F) as i32) | ((((x.qh[e] >> (2 * j)) & 1) as i32) << 4);
                let hi = ((x.qs[32 * j + e] >> 4) as i32) | ((((x.qh[e] >> (2 * j + 1)) & 1) as i32) << 4);
                let q8_lo = y.qs[64 * j + e] as i32;
                let q8_hi = y.qs[64 * j + 32 + e] as i32;
                g_lo += lo * q8_lo;
                g_hi += hi * q8_hi;
                bs_lo += q8_lo;
                bs_hi += q8_hi;
                terms.lanes[e / 4] += sc_lo as i32 * lo * q8_lo + sc_hi as i32 * hi * q8_hi;
            }
            terms.iacc += sc_lo as i32 * g_lo + sc_hi as i32 * g_hi;
            terms.macc += m_lo as i32 * bs_lo + m_hi as i32 * bs_hi;
        }
        terms
    }

    fn bsums_of(y: &BlockQ8K) -> [i32; QK_K / 16] {
        let mut out = [0i32; QK_K / 16];
        for (i, c) in y.qs.chunks_exact(16).enumerate() {
            out[i] = c.iter().map(|&v| v as i32).sum();
        }
        out
    }

    /// (a) lane kernel, `_generic` and an independent exact integer reference.
    ///
    /// What is *provable* and what is not:
    ///   * the integer accumulators are exact in both kernels, so the lane
    ///     kernel's per-lane i32 sums must add up to the independently
    ///     transcribed integer dot (a lane/shuffle/scale mixup in the AVX2 port
    ///     would corrupt a *value*, i.e. a huge error, not ulps);
    ///   * the f32 tails are different summation trees by construction (the
    ///     AVX2 kernel folds the min term into `summs` via one fma per block and
    ///     hsums 8 lanes; the scalar generic keeps `sums[8]` and subtracts
    ///     `dmin*sumi` per block), so lane vs generic are only compared as
    ///     "same value within a small ulp band". Bit-identity to the reference
    ///     is a different claim, proven against the dumped AVX2 output
    ///     (mulmat_ref.bin and `kquant_real_tensor_dump_vs_reference` below).
    ///
    /// `nb` spans one block, the real granite shape (1536/256 = 6) and a long
    /// row; the patterns cover the 5-bit ceiling (31), a zero block and a tiny
    /// f16 scale.
    #[test]
    fn q5k_lane_vs_generic_vs_integer_reference() {
        let mut worst_ulp_lane = 0f64;
        let mut worst_ulp_generic = 0f64;
        let mut worst_lane_vs_generic = 0f64;
        for (name, seed, scale) in [
            ("uniform", 0x5eed_1234u32, 0.75f32),
            ("wide", 0x51ce_5678, 6.0),
            ("narrow", 0x1234_9876, 0.02),
        ] {
            for &nb in &[1usize, 6, 64] {
                let n = nb * QK_K;
                let w = lcg(n, seed, scale);
                let a = lcg(n, seed ^ 0xdead_beef, scale * 1.3);
                let mut x = vec![BlockQ5K::zeroed(); nb];
                let mut y = vec![BlockQ8K::zeroed(); nb];
                quantize_row_q5_K_ref(&w, &mut x);
                quantize_row_q8_K_ref(&a, &mut y);

                let xb: &[u8] = bytemuck::cast_slice(&x);
                let yb: &[u8] = bytemuck::cast_slice(&y);
                let lane = vec_dot_q5_K_q8_K(n, xb, yb);
                let generic = vec_dot_q5_K_q8_K_generic(n, xb, yb);

                // exact integer terms (independent decode) + f64-exact value
                let mut exact = 0f64;
                let mut iacc_total = 0i64;
                let mut macc_total = 0i64;
                let mut max_lane = 0i32;
                for i in 0..nb {
                    let t = exact_terms(&x[i], &y[i]);
                    let d = x[i].d.to_f32() as f64 * y[i].d as f64;
                    let dmin = x[i].dmin.to_f32() as f64 * y[i].d as f64;
                    exact += d * t.iacc as f64 - dmin * t.macc as f64;
                    let split: i64 = t.lanes.iter().map(|&v| v as i64).sum();
                    for &v in t.lanes.iter() {
                        max_lane = max_lane.max(v.abs());
                    }
                    // the 8 AVX2 accumulator lanes must partition the integer
                    // dot exactly (no product lost, none counted twice)
                    assert_eq!(split, t.iacc as i64, "{name} nb {nb}: lane split != exact i32");
                    iacc_total += t.iacc as i64;
                    macc_total += t.macc as i64;
                    let bs = bsums_of(&y[i]);
                    let mut bsum_sum = 0i32;
                    for (g, &b) in bs.iter().enumerate() {
                        bsum_sum += get_scale_min_k4(g / 2, &x[i].scales).1 as i32 * b;
                    }
                    assert_eq!(bsum_sum, t.macc, "{name} nb {nb}: bsums term mismatch");
                    for (g, &b) in bs.iter().enumerate() {
                        assert_eq!(y[i].bsums[g] as i32, b, "{name} nb {nb}: stored bsums[{g}] mismatch");
                    }
                }
                assert!(iacc_total.abs() < (i32::MAX as i64) / 2, "i32 headroom");
                assert!(max_lane < i32::MAX / 4, "{name} nb {nb}: lane i32 overflow risk {max_lane}");

                // f32 ulp of the result magnitude (the kernels accumulate in f32)
                let ulp = |v: f32| (v as f64 - exact).abs() / (exact.abs() * f32::EPSILON as f64).max(f64::MIN_POSITIVE);
                let (ul_lane, ul_gen) = (ulp(lane), ulp(generic));
                let ul_gap = (lane as f64 - generic as f64).abs()
                    / (generic.abs() as f64 * f32::EPSILON as f64).max(f64::MIN_POSITIVE);
                worst_ulp_lane = worst_ulp_lane.max(ul_lane);
                worst_ulp_generic = worst_ulp_generic.max(ul_gen);
                worst_lane_vs_generic = worst_lane_vs_generic.max(ul_gap);
                assert!(
                    ul_lane < 64.0 && ul_gen < 64.0,
                    "{name} nb {nb}: kernel outside the float-rounding class \
                     (lane {lane} {ul_lane:.1} ulp, generic {generic} {ul_gen:.1} ulp, exact {exact})"
                );
                println!(
                    "{name:7} nb {nb:3}: lane {lane:+.9} generic {generic:+.9} exact {exact:+.9} \
                     (lane {ul_lane:.2} ulp, generic {ul_gen:.2} ulp, lane-vs-generic {ul_gap:.2} ulp, |i32| max {max_lane})"
                );
            }
        }
        println!(
            "worst: lane {worst_ulp_lane:.2} ulp / generic {worst_ulp_generic:.2} ulp / \
             lane-vs-generic {worst_lane_vs_generic:.2} ulp (a pure summation-tree difference)"
        );
        assert!(worst_lane_vs_generic < 16.0, "lane vs generic gap {worst_lane_vs_generic:.1} ulp");

        // 5-bit ceiling: weights that quantize to the max code everywhere,
        // and a zero block (d == 0) must not break the lane structure.
        let n = QK_K;
        let mut x = vec![BlockQ5K::zeroed(); 1];
        let mut y = vec![BlockQ8K::zeroed(); 1];
        quantize_row_q5_K_ref(&vec![100.0f32; n], &mut x);
        quantize_row_q8_K_ref(&lcg(n, 7, 1.0), &mut y);
        let xb: &[u8] = bytemuck::cast_slice(&x);
        let yb: &[u8] = bytemuck::cast_slice(&y);
        let t = exact_terms(&x[0], &y[0]);
        assert_eq!(t.lanes.iter().sum::<i32>(), t.iacc, "saturated block lane split");
        let lane = vec_dot_q5_K_q8_K(n, xb, yb);
        let generic = vec_dot_q5_K_q8_K_generic(n, xb, yb);
        let exact = x[0].d.to_f32() as f64 * y[0].d as f64 * t.iacc as f64
            - x[0].dmin.to_f32() as f64 * y[0].d as f64 * t.macc as f64;
        println!("saturated block: lane {lane:+.6} generic {generic:+.6} exact {exact:+.6}");
        assert!(
            ((lane as f64 - exact).abs() / exact.abs()) < 1e-6,
            "saturated block: lane {lane} vs exact {exact}"
        );
        let mut z = vec![BlockQ5K::zeroed(); 1];
        let mut zy = vec![BlockQ8K::zeroed(); 1];
        quantize_row_q5_K_ref(&vec![0.0f32; n], &mut z);
        quantize_row_q8_K_ref(&vec![0.0f32; n], &mut zy);
        let zb: &[u8] = bytemuck::cast_slice(&z);
        let zyb: &[u8] = bytemuck::cast_slice(&zy);
        let zl = vec_dot_q5_K_q8_K(n, zb, zyb);
        assert_eq!(zl.to_bits(), 0.0f32.to_bits(), "zero block must give exactly 0.0");
        assert_eq!(zl.to_bits(), vec_dot_q5_K_q8_K_generic(n, zb, zyb).to_bits());
    }
}


/// (b) Real GGUF tensors × the reference production path
/// (parity/{q5k_real,q4k_granite,q4k_goss,q6k_granite}_ref.bin, built by
/// parity/ref_q5k_dump.c — see its header for the artifact layout).
///
/// For each tensor the artifact carries the reference's own graph mul_mat
/// output twice: once with the weight in a plain buffer (the vec_dot path) and,
/// when the CPU_REPACK buffer type assigned a repack trait to the tensor, once
/// with the weight in the repack buffer (the 8x8 outer-product path the
/// reference really runs for it). Our mul_mat must equal the former bit for
/// bit; the latter is the *production* output, and the gap is the residual that
/// no amount of Q5_K kernel fixing can remove.
#[cfg(test)]
mod kquant_real_tensor_tests {
    use bytemuck::Zeroable;
    use crate::compute::graph_compute;
    use crate::graph::Graph;
    use crate::tensor::Context;
    use crate::types::GgmlType;

    struct Dump {
        ty: GgmlType,
        n: usize,
        rows: usize,
        cols: usize,
        flags: u32,
        xq: Vec<u8>,
        act: Vec<f32>,
        plain: Vec<f32>,
        repacked: Vec<u8>,
        repack_out: Vec<f32>,
        q8k: Vec<u8>,
        sgemm_out: Vec<f32>,
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
        let (ty, n, rows, cols, flags) =
            (GgmlType::from_u32(u(0)).unwrap(), u(1) as usize, u(2) as usize, u(3) as usize, u(4));
        Dump {
            ty,
            n,
            rows,
            cols,
            flags,
            xq: section(&mut c),
            act: f32s(&section(&mut c)),
            plain: f32s(&section(&mut c)),
            repacked: section(&mut c),
            repack_out: f32s(&section(&mut c)),
            q8k: section(&mut c),
            sgemm_out: f32s(&section(&mut c)),
        }
    }

    fn mul_mat(d: &Dump) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(d.ty, d.n as i64, d.rows as i64);
        let b = ctx.new_tensor_2d(GgmlType::F32, d.n as i64, d.cols as i64);
        let out = ctx.mul_mat(a, b);
        for t in [a, b] {
            ctx.arena_resize_tensor(t);
        }
        ctx.data_bytes_mut(a).unwrap().copy_from_slice(&d.xq);
        ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(&d.act));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, 4);
        bytemuck::cast_slice(ctx.data_bytes(out).unwrap()).to_vec()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0f32, f32::max)
    }

    #[test]
    fn kquant_real_tensor_dump_vs_reference() {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/");
        // (file, human name, expected repack trait on this x86 host)
        let cases = [
            ("q5k_real_ref.bin", "granite blk.0.ffn_gate_shexp Q5_K [1536 x 1024]", false, false),
            ("q6k_granite_ref.bin", "granite blk.0.ffn_down_shexp Q6_K [1024 x 8]", false, false),
            ("q5_0_goss_ref.bin", "gpt-oss Q4_K_M blk.0.attn_q Q5_0 [2880 x 8]", false, true),
            ("q8_0_goss_ref.bin", "gpt-oss Q4_K_M blk.0.attn_v Q8_0 [2880 x 8]", false, true),
            ("q4k_granite_ref.bin", "granite blk.0.ssm_out Q4_K [3072 x 8]", true, false),
            ("q4k_goss_ref.bin", "gpt-oss Q4_K_M blk.0.attn_output Q4_K [4096 x 8]", true, false),
        ];
        let mut report = Vec::new();
        for (file, name, expect_repack, expect_sgemm) in cases {
            let path = format!("{base}{file}");
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_q5k_dump.c"));
            let d = parse(&bytes);
            let ours = mul_mat(&d);
            assert_eq!(ours.len(), d.plain.len(), "{name}: output size");

            // the reference's own two paths: does repack change the result?
            let has_repack = d.flags & 1 != 0;
            assert_eq!(
                has_repack, expect_repack,
                "{name}: repack trait {has_repack} != expected {expect_repack} \
                 (repack.cpp:4925 gate, x86 has no Q5_K/Q6_K instance)"
            );
            assert_eq!(
                d.flags & 4 != 0,
                expect_sgemm,
                "{name}: llamafile_sgemm routing != expected (ggml-cpu.c:1383 second attempt with the \
                 converted Q8_0 activation rows)"
            );
            let sgemm_gap = if expect_sgemm {
                assert_eq!(d.sgemm_out.len(), d.plain.len());
                rel_err(&d.sgemm_out, &d.plain)
            } else {
                0.0
            };
            let repack_gap = if has_repack {
                assert_eq!(d.repack_out.len(), d.plain.len());
                rel_err(&d.repack_out, &d.plain)
            } else {
                assert!(d.repacked.is_empty() && d.repack_out.is_empty(), "{name}: unexpected repack sections");
                0.0
            };

            // Our mul_mat must be the reference's *production* path bit for bit:
            // the plain vec_dot where the CPU_REPACK gate gives the tensor no
            // trait (Q5_K/Q6_K on x86, repack.cpp:5050-5071), and the Q4_K 8x8
            // kernels where it does (`ggml_gemm/gemv_q4_K_8x8_q8_K`,
            // repack.cpp:5006-5011).
            let want = if has_repack { &d.repack_out } else { &d.plain };
            let plain_eq = ours.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits());
            let plain_gap = rel_err(&ours, want);
            let line = if plain_eq {
                format!(
                    "{name}: production path BIT-EXACT ({} values); repack trait {}, sgemm {}, \
                     reference repack-vs-plain gap {repack_gap:.2e}, sgemm-vs-plain gap {sgemm_gap:.2e}",
                    ours.len(),
                    if has_repack { "YES (8x8)" } else { "none" },
                    if expect_sgemm { "tinyBLAS" } else { "vec_dot" },
                )
            } else {
                format!(
                    "{name}: MISMATCH vs the reference production path (worst rel {plain_gap:.2e})"
                )
            };
            println!("{line}");
            report.push((name, plain_eq, plain_gap, has_repack, repack_gap));

            // the activation quantization inside the reference's mul_mat must be
            // ours as well (same Q8_K bytes for the same f32 rows). The dump sizes the
            // section as nact * (n/QK_K) blocks and calls the reference quantizer with
            // k = n (parity/ref_q5k_dump.c:218); with n = 2880 the trailing n%QK_K = 64
            // values are dropped because the reference's `assert(k % QK_K == 0)` is
            // compiled out of the release build. Reproduce exactly that prefix.
            if !d.q8k.is_empty() {
                let mut mine = vec![0u8; d.q8k.len()];
                for r in 0..d.cols {
                    let row = &d.act[r * d.n..(r + 1) * d.n];
                    let nblk = d.n / crate::blocks::QK_K;
                    let covered = nblk * crate::blocks::QK_K;
                    let mut blocks = vec![crate::blocks::BlockQ8K::zeroed(); nblk];
                    crate::quants::quantize_row_q8_K_ref(&row[..covered], &mut blocks);
                    let bytes: &[u8] = bytemuck::cast_slice(&blocks);
                    mine[r * bytes.len()..(r + 1) * bytes.len()].copy_from_slice(bytes);
                }
                assert_eq!(mine, d.q8k, "{name}: Q8_K activation bytes differ from the reference");
            }
        }

        let failed: Vec<_> = report.iter().filter(|r| !r.1).collect();
        assert!(failed.is_empty(), "our mul_mat != the reference production path for: {failed:?}");

        let affected: Vec<_> = report.iter().filter(|r| r.3).collect();
        assert!(
            !affected.is_empty(),
            "no repack case in the set — the Q4_K production-path asymmetry would be untested"
        );
        for a in affected {
            println!(
                "  => {name}: reference production path is the Q4_K 8x8 repack kernel, which this \
                 port now runs bit-exactly; its gap to the reference's own plain vec_dot is \
                 {gap:.2e} relative ({kind})",
                name = a.0,
                gap = a.4,
                kind = if a.4 > 0.0 { "a real summation-tree difference" } else { "bit-identical" }
            );
        }
    }
}

/// Round-trip guard for the *thread partitioning* of the compute kernels:
/// splitting a node's work across threads must not change a single bit (the
/// chunk boundaries only decide who computes an output element, never how).
/// This is the regression test for the `par_mul_mat` bug where the sequential
/// fallback only ran the first chunk, i.e. skipped 75 of 76 row blocks at
/// `n_threads == 1`.
#[cfg(test)]
mod thread_invariance_tests {
    use crate::compute::graph_compute;
    use crate::graph::Graph;
    use crate::tensor::Context;
    use crate::types::GgmlType;

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 28) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// `[rows x n] · [n x cols]` through the real graph, at several thread
    /// counts, for a quantized type (integer vec_dot path), F32 and F16.
    #[test]
    fn mul_mat_bit_identical_across_thread_counts() {
        for (ty, n, rows, cols) in [
            (GgmlType::Q5_0, 256usize, 151usize, 3usize),
            (GgmlType::Q4K, 256, 129, 2),
            (GgmlType::Q6K, 512, 33, 3),
            (GgmlType::F32, 40, 67, 3),
            (GgmlType::F16, 40, 35, 2),
        ] {
            let run = |nth: usize| -> Vec<f32> {
                let mut ctx = Context::new();
                let a = ctx.new_tensor_2d(ty, n as i64, rows as i64);
                let b = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
                let out = ctx.mul_mat(a, b);
                for t in [a, b] {
                    ctx.arena_resize_tensor(t);
                }
                let wf = lcg(n * rows, 11);
                let raw = ctx.data_bytes_mut(a).unwrap();
                quantize(ty, &wf, raw);
                let af = lcg(n * cols, 22);
                ctx.data_bytes_mut(b)
                    .unwrap()
                    .copy_from_slice(bytemuck::cast_slice(&af));
                let mut g = Graph::new(8);
                g.build_forward(&ctx, out);
                graph_compute(&mut ctx, &mut g, nth);
                bytemuck::cast_slice(ctx.data_bytes(out).unwrap()).to_vec()
            };
            let base = run(1);
            for nth in [2usize, 3, 4, 8] {
                let got = run(nth);
                assert_eq!(got.len(), base.len(), "{ty:?}: size");
                for (i, (x, y)) in got.iter().zip(&base).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{ty:?} nth={nth}: element {i} differs ({x} vs {y})"
                    );
                }
            }
        }
    }

    fn quantize(ty: GgmlType, xf: &[f32], out: &mut [u8]) {
        use crate::quants::*;
        use crate::quants_k::*;
        match ty {
            GgmlType::Q5_0 => quantize_row_q5_0_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q4K => quantize_row_q4_K_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::Q6K => quantize_row_q6_K_ref(xf, bytemuck::cast_slice_mut(out)),
            GgmlType::F32 => out.copy_from_slice(bytemuck::cast_slice(xf)),
            GgmlType::F16 => {
                let o: &mut [half::f16] = bytemuck::cast_slice_mut(out);
                for (d, s) in o.iter_mut().zip(xf) {
                    *d = half::f16::from_f32(*s);
                }
            }
            other => unimplemented!("quantize {other:?}"),
        }
    }
}

/// Bit-exact parity of the audit-round kernels (Q1_0/Q2_0/NVFP4/IQ x9) vs the
/// reference production kernels (parity/vecdot3_ref.bin, built by
/// parity/ref_vecdot_dump3.c against libggml-cpu.so.0 — the AVX2 lane bodies).
#[cfg(test)]
mod vecdot3_tests {
    use super::*;
    use crate::types::GgmlType;

    #[test]
    fn vecdot3_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/vecdot3_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_vecdot_dump3.c"));
        let mut c = &bytes[..];
        let mut sections = 0usize;
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x3344_4456, "VDD3 section magic");
            let tid = u32::from_le_bytes(c[..4].try_into().unwrap());
            let n = u32::from_le_bytes(c[4..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let ty = GgmlType::from_u32(tid).unwrap();
            let (xq, rest) = c.split_at(ty.row_size(n));
            c = rest;
            let vdt = vec_dot_type(ty).unwrap();
            let (yq, rest) = c.split_at(vdt.row_size(n));
            c = rest;
            let res_ref = f32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];

            let mine = vec_dot_row(ty, n, xq, yq);
            assert_eq!(
                mine.to_bits(),
                res_ref.to_bits(),
                "{ty:?} n {n}: mine {mine} ({:08x}) ref {res_ref} ({:08x})",
                mine.to_bits(),
                res_ref.to_bits()
            );
            sections += 1;
        }
        assert_eq!(sections, 12, "12 audit-round types");
    }

    /// The same LCG byte-for-byte as parity/ref_vecdot_dump3.c — used to
    /// rebuild synthetic weight rows for the wiring test below.
    fn lcg_bytes(n: usize) -> Vec<u8> {
        let mut state: u32 = 0x5eed_1234;
        // the C probe consumes 2*1024 floats for x/y first (next_val), i.e.
        // 2048 draws, then bytes; mirror that stream when parity matters
        let mut next = |state: &mut u32| -> u32 {
            *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *state
        };
        for _ in 0..2048 {
            next(&mut state);
        }
        (0..n).map(|_| (next(&mut state) >> 13) as u8).collect()
    }

    /// Synthetic tensors per type through the real graph mul_mat: this is the
    /// dispatch-level wiring the audit flagged (compute.rs's vec_dot_type +
    /// wdata activation quantization + vec_dot_row). The per-kernel bits are
    /// pinned to the reference by `vecdot3_bit_exact_vs_reference`; here the
    /// graph output must equal a hand-rolled quantize+vec_dot_row per element,
    /// which fails if any routing arm is missing or mis-typed.
    #[test]
    fn mul_mat_wiring_synthetic_weights() {
        use crate::compute::graph_compute;
        use crate::graph::Graph;
        use crate::tensor::Context;

        let types = [
            GgmlType::Q1_0,
            GgmlType::Q2_0,
            GgmlType::Nvfp4,
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
        ];
        let (n, rows, cols) = (1024usize, 3usize, 2usize);
        for ty in types {
            let mut ctx = Context::new();
            let a = ctx.new_tensor_2d(ty, n as i64, rows as i64);
            let b = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
            let out = ctx.mul_mat(a, b);
            for t in [a, b] {
                ctx.arena_resize_tensor(t);
            }
            // weights: LCG bytes with the fp16 d forced finite (same patch as
            // the C probe); iq1_m's nibble-assembled scale -> fp16 1.0
            let mut wb = lcg_bytes(ty.row_size(n) * rows);
            let (blck, tsz) = (ty.blck_size(), ty.type_size());
            for ib in 0..n / blck * rows {
                let off = ib * tsz;
                if ty == GgmlType::Iq1M {
                    let sc = |j: usize| u16::from_le_bytes([wb[off + 2 * j], wb[off + 2 * j + 1]]);
                    let mut sc16 = [sc(0), sc(1), sc(2), sc(3)];
                    sc16[0] = (sc16[0] & 0x0fff) | 0x3000;
                    sc16[1] = (sc16[1] & 0xf0ff) | 0x0c00;
                    sc16[2] &= 0xff0f;
                    sc16[3] &= 0xfff0;
                    for (j, v) in sc16.iter().enumerate() {
                        wb[off + 2 * j..off + 2 * j + 2].copy_from_slice(&v.to_le_bytes());
                    }
                } else if ty != GgmlType::Nvfp4 {
                    wb[off + 1] = (wb[off + 1] & 0x03) | 0x38;
                }
            }
            ctx.data_bytes_mut(a).unwrap().copy_from_slice(&wb);
            // activations: plain LCG floats
            let mut state: u32 = 0x5eed_1234;
            let acts: Vec<f32> = (0..n * cols)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state as i32 as f32 / (1u32 << 28) as f32) * 1.8 - 0.9
                })
                .collect();
            ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(&acts));

            let mut g = Graph::new(8);
            g.build_forward(&ctx, out);
            graph_compute(&mut ctx, &mut g, 3);
            let got: Vec<f32> = bytemuck::cast_slice(ctx.data_bytes(out).unwrap()).to_vec();

            // expected: the port's own activation quantizer + vec_dot_row
            let vdt = vec_dot_type(ty).unwrap();
            let rs = vdt.row_size(n);
            let mut exp = Vec::with_capacity(rows * cols);
            for col in 0..cols {
                let mut yq = vec![0u8; rs];
                let rowf = &acts[col * n..(col + 1) * n];
                match vdt {
                    GgmlType::Q8_0 => crate::quants::quantize_row_q8_0(
                        rowf,
                        bytemuck::cast_slice_mut(&mut yq),
                    ),
                    GgmlType::Q8K => crate::quants::quantize_row_q8_K(
                        rowf,
                        bytemuck::cast_slice_mut(&mut yq),
                    ),
                    other => unreachable!("vec_dot_type {other:?}"),
                }
                for row in 0..rows {
                    let xr = &wb[row * ty.row_size(n)..(row + 1) * ty.row_size(n)];
                    exp.push(vec_dot_row(ty, n, xr, &yq));
                }
            }
            assert_eq!(got.len(), exp.len(), "{ty:?}: output size");
            for (i, (x, y)) in got.iter().zip(&exp).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{ty:?}: element {i} differs ({x} vs {y})"
                );
            }
        }
    }
}
