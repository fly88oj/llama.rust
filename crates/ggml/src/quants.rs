//! Quantization algorithms — port of ggml/src/ggml-quants.c.
//!
//! Priority order: dequantize (weight loading) → q8 activation quant → vec_dot
//! → weight quantize (llama-quantize parity). All math mirrors the C scalar
//! reference (`*_ref` implementations) exactly, incl. float rounding modes.

use crate::blocks::*;
use crate::types::GgmlType;
use half::f16;

/// ggml-quants.c `nearest_int` — magic-number round-to-nearest-even for |f| < 2^22.
#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

#[inline]
fn fp16(x: f32) -> f16 {
    f16::from_f32(x)
}
#[inline]
fn f32_of(x: f16) -> f32 {
    x.to_f32()
}
#[inline]
fn minmax(x: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = -f32::MAX;
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

// ===================== Q1_0

pub fn quantize_row_q1_0_ref(x: &[f32], y: &mut [BlockQ1_0]) {
    let qk = QK1_0;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        // ggml-quants.c:40-70: d = sum|x| / qk (mean absolute value), sign bits
        // stored directly (no normalization).
        let mut sum_abs = 0.0f32;
        for j in 0..qk {
            sum_abs += x[i * qk + j].abs();
        }
        let d = sum_abs / qk as f32;
        let mut qs = [0u8; QK1_0 / 8];
        for j in 0..qk {
            let bit = (x[i * qk + j] >= 0.0) as u8;
            qs[j / 8] |= bit << (j % 8);
        }
        *b = BlockQ1_0 { d: fp16(d), qs };
    }
}

pub fn dequantize_row_q1_0(x: &[BlockQ1_0], y: &mut [f32]) {
    let qk = QK1_0;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        let neg = -d;
        for j in 0..qk {
            let bit = (b.qs[j / 8] >> (j % 8)) & 1;
            y[i * qk + j] = if bit != 0 { d } else { neg };
        }
    }
}

// ===================== Q2_0

pub fn quantize_row_q2_0_ref(x: &[f32], y: &mut [BlockQ2_0]) {
    let qk = QK2_0;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        // ggml-quants.c:74-110: d = max|x|; q = clamp(roundf(w/d) + 1, 0, 3)
        // (roundf = half away from zero, like the other _ref quantizers).
        let mut amax = 0.0f32;
        for j in 0..qk {
            let a = x[i * qk + j].abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; QK2_0 / 4];
        for j in 0..qk {
            let mut q = (x[i * qk + j] * id).round() as i32 + 1;
            q = q.clamp(0, 3);
            qs[j / 4] |= (q as u8) << ((j % 4) * 2);
        }
        *b = BlockQ2_0 { d: fp16(d), qs };
    }
}

pub fn dequantize_row_q2_0(x: &[BlockQ2_0], y: &mut [f32]) {
    let qk = QK2_0;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        for j in 0..qk {
            let q = (b.qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i * qk + j] = (q as i32 - 1) as f32 * d;
        }
    }
}

// ===================== Q4_0 / Q4_1

pub fn quantize_row_q4_0_ref(x: &[f32], y: &mut [BlockQ4_0]) {
    let qk = QK4_0;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        let mut amax = 0.0;
        let mut max = 0.0;
        for j in 0..qk {
            let v = x[i * qk + j];
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; QK4_0 / 2];
        for j in 0..qk / 2 {
            let x0 = x[i * qk + j] * id;
            let x1 = x[i * qk + qk / 2 + j] * id;
            let xi0 = (x0 + 8.5) as i8; // C: (int8_t)(x0 + 8.5f) then MIN(15, .)
            let xi1 = (x1 + 8.5) as i8;
            let xi0 = (xi0.min(15) as u8) & 0x0F;
            let xi1 = (xi1.min(15) as u8) & 0x0F;
            qs[j] = xi0 | (xi1 << 4);
        }
        *b = BlockQ4_0 { d: fp16(d), qs };
    }
}

pub fn dequantize_row_q4_0(x: &[BlockQ4_0], y: &mut [f32]) {
    let qk = QK4_0;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        for j in 0..qk / 2 {
            let x0 = (b.qs[j] & 0x0F) as i32 - 8;
            let x1 = (b.qs[j] >> 4) as i32 - 8;
            y[i * qk + j] = x0 as f32 * d;
            y[i * qk + qk / 2 + j] = x1 as f32 * d;
        }
    }
}

pub fn quantize_row_q4_1_ref(x: &[f32], y: &mut [BlockQ4_1]) {
    let qk = QK4_1;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        let (min, max) = minmax(&x[i * qk..i * qk + qk]);
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0u8; QK4_1 / 2];
        for j in 0..qk / 2 {
            let x0 = (x[i * qk + j] - min) * id;
            let x1 = (x[i * qk + qk / 2 + j] - min) * id;
            let xi0 = (((x0 + 0.5) as i32).min(15).max(0)) as u8;
            let xi1 = (((x1 + 0.5) as i32).min(15).max(0)) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }
        *b = BlockQ4_1 { d: fp16(d), m: fp16(min), qs };
    }
}

pub fn dequantize_row_q4_1(x: &[BlockQ4_1], y: &mut [f32]) {
    let qk = QK4_1;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        let m = f32_of(b.m);
        for j in 0..qk / 2 {
            let x0 = (b.qs[j] & 0x0F) as i32;
            let x1 = (b.qs[j] >> 4) as i32;
            y[i * qk + j] = x0 as f32 * d + m;
            y[i * qk + qk / 2 + j] = x1 as f32 * d + m;
        }
    }
}

// ===================== Q5_0 / Q5_1

pub fn quantize_row_q5_0_ref(x: &[f32], y: &mut [BlockQ5_0]) {
    let qk = QK5_0;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        let mut amax = 0.0;
        let mut max = 0.0;
        for j in 0..qk {
            let v = x[i * qk + j];
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_0 / 2];
        for j in 0..qk / 2 {
            let x0 = x[i * qk + j] * id;
            let x1 = x[i * qk + qk / 2 + j] * id;
            let xi0 = (((x0 + 16.5) as i8).min(31) as u8) & 0x1F;
            let xi1 = (((x1 + 16.5) as i8).min(31) as u8) & 0x1F;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + qk / 2);
        }
        *b = BlockQ5_0 {
            d: fp16(d),
            qh: qh.to_le_bytes(),
            qs,
        };
    }
}

pub fn dequantize_row_q5_0(x: &[BlockQ5_0], y: &mut [f32]) {
    let qk = QK5_0;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        let qh = u32::from_le_bytes(b.qh);
        for j in 0..qk / 2 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((b.qs[j] & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((b.qs[j] >> 4) | xh_1) as i32 - 16;
            y[i * qk + j] = x0 as f32 * d;
            y[i * qk + qk / 2 + j] = x1 as f32 * d;
        }
    }
}

pub fn quantize_row_q5_1_ref(x: &[f32], y: &mut [BlockQ5_1]) {
    let qk = QK5_1;
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * qk);
    for (i, b) in y.iter_mut().enumerate() {
        let (min, max) = minmax(&x[i * qk..i * qk + qk]);
        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_1 / 2];
        for j in 0..qk / 2 {
            let x0 = (x[i * qk + j] - min) * id;
            let x1 = (x[i * qk + qk / 2 + j] - min) * id;
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + qk / 2);
        }
        *b = BlockQ5_1 {
            d: fp16(d),
            m: fp16(min),
            qh: qh.to_le_bytes(),
            qs,
        };
    }
}

pub fn dequantize_row_q5_1(x: &[BlockQ5_1], y: &mut [f32]) {
    let qk = QK5_1;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        let m = f32_of(b.m);
        let qh = u32::from_le_bytes(b.qh);
        for j in 0..qk / 2 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((b.qs[j] & 0x0F) | xh_0) as i32;
            let x1 = ((b.qs[j] >> 4) | xh_1) as i32;
            y[i * qk + j] = x0 as f32 * d + m;
            y[i * qk + qk / 2 + j] = x1 as f32 * d + m;
        }
    }
}

// ===================== Q8_0 / Q8_1 (activations & deterministic files)

pub fn quantize_row_q8_0_ref(x: &[f32], y: &mut [BlockQ8_0]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK8_0);
    for (i, b) in y.iter_mut().enumerate() {
        let mut amax = 0.0f32;
        for j in 0..QK8_0 {
            amax = amax.max(x[i * QK8_0 + j].abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_0];
        for j in 0..QK8_0 {
            // C _ref uses roundf = round-half-away-from-zero
            qs[j] = (x[i * QK8_0 + j] * id).round() as i8;
        }
        *b = BlockQ8_0 { d: fp16(d), qs };
    }
}

/// Runtime fast path — 1:1 with the x86 AVX kernel (arch/x86/quants.c:302):
/// `id = 127/amax` (NOT the ref's 1/(amax/127)) and round-to-nearest-even
/// (`_mm256_round_ps`), which differ from `_ref` on ties.
pub fn quantize_row_q8_0(x: &[f32], y: &mut [BlockQ8_0]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_x86::avx2() {
        // the AVX2 body itself (simd_x86), same arithmetic lane-for-lane
        crate::simd_x86::quantize_row_q8_0(x, y);
        return;
    }
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK8_0);
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
            // `_mm256_cvtps_epi32` maps NaN/±inf/int32-overflow lanes to
            // INT_MIN (quants.c:347) and `packs` then saturates to -128 —
            // reachable only when `id` itself overflows (amax deep-subnormal);
            // mirror it so the scalar spec stays == the AVX2/SIMD body.
            qs[j] = if r.is_nan() || r.abs() >= 2147483648.0 {
                i8::MIN
            } else {
                r as i8
            };
        }
        *b = BlockQ8_0 { d: fp16(d), qs };
    }
}

pub fn dequantize_row_q8_0(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK8_0);
    for (i, b) in x.iter().enumerate() {
        let d = f32_of(b.d);
        for j in 0..QK8_0 {
            y[i * QK8_0 + j] = b.qs[j] as f32 * d;
        }
    }
}

pub fn quantize_row_q8_1_ref(x: &[f32], y: &mut [BlockQ8_1]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK8_1);
    for (i, b) in y.iter_mut().enumerate() {
        let mut amax = 0.0f32;
        for j in 0..QK8_1 {
            amax = amax.max(x[i * QK8_1 + j].abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK8_1];
        let mut sum: i32 = 0;
        for j in 0..QK8_1 / 2 {
            let v0 = x[i * QK8_1 + j] * id;
            let v1 = x[i * QK8_1 + QK8_1 / 2 + j] * id;
            qs[j] = v0.round_ties_even() as i8;
            qs[QK8_1 / 2 + j] = v1.round_ties_even() as i8;
            sum += qs[j] as i32;
            sum += qs[QK8_1 / 2 + j] as i32;
        }
        *b = BlockQ8_1 { d: fp16(d), s: fp16(sum as f32 * d), qs };
    }
}

pub fn quantize_row_q8_1(x: &[f32], y: &mut [BlockQ8_1]) {
    quantize_row_q8_1_ref(x, y)
}

// ===================== row conversion for unquantized types

pub fn dequantize_row_f16(x: &[f16], y: &mut [f32]) {
    for (dst, src) in y.iter_mut().zip(x) {
        *dst = src.to_f32();
    }
}

pub fn dequantize_row_bf16(x: &[half::bf16], y: &mut [f32]) {
    for (dst, src) in y.iter_mut().zip(x) {
        *dst = src.to_f32();
    }
}

// ===================== K-quants (super-blocks of QK_K = 256)

/// ggml-quants.c `get_scale_min_k4`
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub fn dequantize_row_q2_K(x: &[BlockQ2K], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK_K);
    let mut yi = 0usize;
    for b in x.iter() {
        let d = f32_of(b.d);
        let min = f32_of(b.dmin);
        let q = &b.qs;
        let mut is = 0usize;
        let mut qi = 0usize; // q advances by 32 per 128-group
        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let sc = b.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    y[yi] = dl * ((q[qi + l] >> shift) & 3) as i8 as f32 - ml;
                    yi += 1;
                }
                let sc = b.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    y[yi] = dl * ((q[qi + l + 16] >> shift) & 3) as i8 as f32 - ml;
                    yi += 1;
                }
                shift += 2;
            }
            qi += 32;
        }
    }
}

pub fn dequantize_row_q3_K(x: &[BlockQ3K], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK_K);
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut yi = 0usize;
    for b in x.iter() {
        let d_all = f32_of(b.d);
        // unpack 12-byte scales into 16 i8 scales (same aux[] trick as C)
        let mut aux = [0u32; 4];
        for (i, chunk) in b.scales.chunks_exact(4).enumerate() {
            aux[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales: [i8; 16] = bytemuck::cast(aux);

        let q = &b.qs;
        let hm = &b.hmask;
        let mut m: u8 = 1;
        let mut is = 0usize;
        let mut qi = 0usize; // q index within this block
        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((q[qi + l] >> shift) & 3) as i8;
                    let sub = if hm[l] & m != 0 { 0 } else { 4 };
                    y[yi] = dl * (qv - sub) as f32;
                    yi += 1;
                }
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((q[qi + l + 16] >> shift) & 3) as i8;
                    let sub = if hm[l + 16] & m != 0 { 0 } else { 4 };
                    y[yi] = dl * (qv - sub) as f32;
                    yi += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qi += 32;
        }
    }
}

pub fn dequantize_row_q4_K(x: &[BlockQ4K], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK_K);
    let mut yi = 0usize;
    for b in x.iter() {
        let d = f32_of(b.d);
        let min = f32_of(b.dmin);
        let mut is = 0usize;
        let mut qi = 0usize;
        for _j in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, &b.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &b.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                y[yi + l] = d1 * (b.qs[qi + l] & 0xF) as f32 - m1;
            }
            yi += 32;
            for l in 0..32 {
                y[yi + l] = d2 * (b.qs[qi + l] >> 4) as f32 - m2;
            }
            yi += 32;
            qi += 32;
            is += 2;
        }
    }
}

pub fn dequantize_row_q5_K(x: &[BlockQ5K], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK_K);
    let mut yi = 0usize;
    for b in x.iter() {
        let d = f32_of(b.d);
        let min = f32_of(b.dmin);
        let mut is = 0usize;
        let mut qi = 0usize; // ql index
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, &b.scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &b.scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                let hi = if b.qh[l] & u1 != 0 { 16 } else { 0 };
                y[yi + l] = d1 * ((b.qs[qi + l] & 0xF) as i32 + hi) as f32 - m1;
            }
            yi += 32;
            for l in 0..32 {
                let hi = if b.qh[l] & u2 != 0 { 16 } else { 0 };
                y[yi + l] = d2 * ((b.qs[qi + l] >> 4) as i32 + hi) as f32 - m2;
            }
            yi += 32;
            qi += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

pub fn dequantize_row_q6_K(x: &[BlockQ6K], y: &mut [f32]) {
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * QK_K);
    let mut yi = 0usize;
    for b in x.iter() {
        let d = f32_of(b.d);
        let mut ql = 0usize;
        let mut qh = 0usize;
        let mut sc = 0usize;
        for _n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((b.ql[ql + l] & 0xF) | (((b.qh[qh + l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((b.ql[ql + l + 32] & 0xF) | (((b.qh[qh + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((b.ql[ql + l] >> 4) | (((b.qh[qh + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((b.ql[ql + l + 32] >> 4) | (((b.qh[qh + l] >> 6) & 3) << 4)) as i8 - 32;
                y[yi + l] = d * b.scales[sc + is + 0] as f32 * q1 as f32;
                y[yi + l + 32] = d * b.scales[sc + is + 2] as f32 * q2 as f32;
                y[yi + l + 64] = d * b.scales[sc + is + 4] as f32 * q3 as f32;
                y[yi + l + 96] = d * b.scales[sc + is + 6] as f32 * q4 as f32;
            }
            yi += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/// ggml-quants.c `quantize_row_q8_K_ref`
pub fn quantize_row_q8_K_ref(x: &[f32], y: &mut [BlockQ8K]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);
    for (i, b) in y.iter_mut().enumerate() {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            b.d = 0.0;
            b.qs = [0; QK_K];
            b.bsums = [0; QK_K / 16];
            continue;
        }
        // not -128.f/max: needed for IQ2_XXS AVX, kept for parity
        let iscale = -127.0 / max;
        for j in 0..QK_K {
            let v = nearest_int(iscale * xb[j]);
            b.qs[j] = v.min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let mut sum: i32 = 0;
            for ii in 0..16 {
                sum += b.qs[j * 16 + ii] as i32;
            }
            b.bsums[j] = sum as i16;
        }
        b.d = 1.0 / iscale;
    }
}

pub fn quantize_row_q8_K(x: &[f32], y: &mut [BlockQ8K]) {
    quantize_row_q8_K_ref(x, y)
}

// ===================== MXFP4 / NVFP4 (OCP microscaling FP4)

/// ggml-common.h:1126-1129 `kvalues_fp4` (aliased as `kvalues_mxfp4`).
/// The 16 signed E2M1 codes scaled by 2 (so the halved scales below line up).
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `ggml_table_f32_e8m0_half` (ggml.c / ggml-impl.h:489) — the 256-entry LUT
/// the reference's x86 repack gemv/gemm gathers the 8 per-tile E8M0 scales
/// through (`vmovss (%rdi,%rax,4)` in parity/asm/ref_mxfp4_gemv_loop.asm).
/// Entry x is bit-identical to [`e8m0_to_fp32_half(x)`] by construction (the
/// same bit pattern, evaluated at compile time), so gather and arithmetic
/// decodes are interchangeable.
pub const E8M0_HALF_LUT: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = e8m0_to_fp32_half(i as u8);
        i += 1;
    }
    t
};

/// ggml-impl.h:489 `ggml_e8m0_to_fp32_half` — E8M0 (bias 127) → f32, times 0.5.
/// Exact bit construction: `2^(x-128)`, with the two denormals x=0,1 handled by
/// precomputed patterns. NaNs are not handled (mirrors the C note).
#[inline]
pub const fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 {
        // 0x00200000 = 2^-128, 0x00400000 = 2^-127
        0x0020_0000u32 << x
    } else {
        // 0.5 * 2^(x-127) = 2^(x-128): normalized, exponent field (x - 1)
        ((x - 1) as u32) << 23
    };
    f32::from_bits(bits)
}

/// ggml-impl.h:514 `ggml_ue4m3_to_fp32` — UE4M3 (4 exp bits, bias 7, 3 mantissa
/// bits) → f32, times 0.5 (to match the doubled `KVALUES_MXFP4` convention).
/// 0x00 and 0x7F are special-cased to 0.0. `ldexpf(x, n)` is exact for every
/// value this encoder can produce (all results are normals with a 3-bit
/// mantissa), so the shift is done with plain exponent arithmetic.
#[inline]
pub fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = (x >> 3) & 0xF;
    let man = x & 0x7;
    let raw = if exp == 0 {
        // subnormal: man * 2^-9
        ldexp_exact(man as f32, -9)
    } else {
        ldexp_exact(1.0 + man as f32 / 8.0, exp as i32 - 7)
    };
    raw * 0.5
}

/// `ldexpf(x, n)` for the inputs `ue4m3_to_fp32` can produce: x is 0.0 or a
/// normal f32 (1..1.875 or a small integer) and `x * 2^n` is again normal, so
/// the scaling is exact and reduces to integer exponent arithmetic.
#[inline]
fn ldexp_exact(x: f32, n: i32) -> f32 {
    debug_assert!(x >= 0.0 && x.is_finite());
    if x == 0.0 {
        return x; // byte 0x80: exp == 0, man == 0 -> ldexpf(0, -9) == 0
    }
    debug_assert!(x.is_normal());
    let m = x.to_bits();
    let e = ((m >> 23) & 0xFF) as i32 - 127 + n;
    debug_assert!((1..=254).contains(&(e + 127)), "ldexp overflow x={x} n={n}");
    f32::from_bits((m & 0x807F_FFFF) | (((e + 127) as u32) << 23))
}

/// ggml-quants.c:569 `dequantize_row_mxfp4` — QK_MXFP4 = 32, 17-byte blocks
/// (1 E8M0 scale + 16 packed E2M1 nibbles).
pub fn dequantize_row_mxfp4(x: &[BlockMxfp4], y: &mut [f32]) {
    let qk = QK_MXFP4;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        let d = e8m0_to_fp32_half(b.e);
        for j in 0..qk / 2 {
            let x0 = KVALUES_MXFP4[(b.qs[j] & 0x0F) as usize] as f32;
            let x1 = KVALUES_MXFP4[(b.qs[j] >> 4) as usize] as f32;
            y[i * qk + j] = x0 * d;
            y[i * qk + qk / 2 + j] = x1 * d;
        }
    }
}

/// ggml-quants.c:589 `dequantize_row_nvfp4` — QK_NVFP4 = 64, 36-byte blocks
/// (4 UE4M3 sub-block scales + 32 packed E2M1 nibbles), 16-element sub-blocks.
pub fn dequantize_row_nvfp4(x: &[BlockNvfp4], y: &mut [f32]) {
    let qk = QK_NVFP4;
    let qk_sub = QK_NVFP4_SUB;
    let n_sub = qk / qk_sub;
    let nb = x.len();
    debug_assert_eq!(y.len(), nb * qk);
    for (i, b) in x.iter().enumerate() {
        for s in 0..n_sub {
            let d = ue4m3_to_fp32(b.d[s]);
            let yb = &mut y[i * qk + s * qk_sub..i * qk + (s + 1) * qk_sub];
            for j in 0..qk_sub / 2 {
                let v0 = KVALUES_MXFP4[(b.qs[s * (qk_sub / 2) + j] & 0x0F) as usize] as f32;
                let v1 = KVALUES_MXFP4[(b.qs[s * (qk_sub / 2) + j] >> 4) as usize] as f32;
                yb[j] = v0 * d;
                yb[j + qk_sub / 2] = v1 * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgmlType;

    /// LCG identical to parity/ref_quants_dump.c
    fn ref_input(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 30) as f32) * 0.75 - 0.125
            })
            .collect()
    }

    /// Bit-exact parity vs the pinned C reference (parity/quants_ref.bin,
    /// generated by parity/ref_quants_dump.c against ggml-base bd4f514db1).
    ///
    /// For every dumped type:
    ///  - our quantize_*_ref must produce identical block bytes
    ///  - our dequantize on the reference's block bytes must produce identical f32s
    #[test]
    fn quants_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/quants_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            panic!("missing {path}: build it via parity/ref_quants_dump.c (see FILE_MAP)");
        };
        let x = ref_input(1024);
        let mut c = &bytes[..];
        let mut sections_checked = 0usize;
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF {
                break; // EOF marker (written in place of a section header)
            }
            assert_eq!(magic, 0x5443_4553, "section magic");
            let tid = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let qlen = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let (qbytes, rest) = c.split_at(qlen);
            let (deq, rest2) = rest.split_at(n * 4);
            c = rest2;
            assert_eq!(n, 1024);
            let ty = GgmlType::from_u32(tid).unwrap();

            // 1) our quantize == reference quantize (only for implemented types)
            let quant_impl: Option<fn(&[f32], &mut [u8])> = match ty {
                GgmlType::Q4_0 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ4_0]| quantize_row_q4_0_ref(x, y))),
                GgmlType::Q4_1 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ4_1]| quantize_row_q4_1_ref(x, y))),
                GgmlType::Q5_0 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ5_0]| quantize_row_q5_0_ref(x, y))),
                GgmlType::Q5_1 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ5_1]| quantize_row_q5_1_ref(x, y))),
                GgmlType::Q8_0 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ8_0]| quantize_row_q8_0_ref(x, y))),
                GgmlType::Q8_1 => Some(|x, out| fill(x, out, |x, y: &mut [BlockQ8_1]| quantize_row_q8_1_ref(x, y))),
                _ => None, // K-quant quantizers: next batch
            };
            if let Some(q) = quant_impl {
                let mut mine = vec![0u8; qlen];
                q(&x, &mut mine);
                assert_eq!(mine.as_slice(), qbytes, "{ty:?}: quantized bytes differ from reference");
            }

            // 2) our dequant(reference bytes) == reference dequant
            let all_zero = deq.iter().all(|b| *b == 0);
            if !all_zero {
                let ref_deq: &[f32] = bytemuck::cast_slice(deq);
                let mut mine = vec![0f32; n];
                dequantize_row(ty, qbytes, &mut mine);
                let mut diff_idx = usize::MAX;
                let mismatch = mine
                    .iter()
                    .zip(ref_deq.iter())
                    .enumerate()
                    .find(|(i, (a, b))| {
                        if a.to_bits() != b.to_bits() {
                            diff_idx = *i;
                            true
                        } else {
                            false
                        }
                    })
                    .is_some();
                assert!(
                    !mismatch,
                    "{ty:?}: dequant differs from reference at index {diff_idx}: ours={} ref={}",
                    if diff_idx < n { mine[diff_idx] } else { f32::NAN },
                    if diff_idx < n { ref_deq[diff_idx] } else { f32::NAN },
                );
            }
            sections_checked += 1;
        }
        assert_eq!(sections_checked, 12, "expected 12 sections");
    }

    /// Bit-exact parity for the two types added in this revision, Q1_0 and Q2_0
    /// (parity/quants_ref_q1q2.bin, same generator + same input LCG as
    /// quants_ref.bin — kept in its own file so the 12-section count above holds).
    ///
    /// These are the only ggml quantizers the port had gated off
    /// (tools/quantize/src/rows.rs::has_quantizer): Q1_0 stores d = sum|x|/QK
    /// with a raw sign bit per weight, Q2_0 stores d = max|x| and
    /// q = clamp(roundf(w/d) + 1, 0, 3) — both differ from the obvious
    /// "normalize by max" formulation.
    #[test]
    fn quants_q1_0_q2_0_bit_exact_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/quants_ref_q1q2.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build it via parity/ref_quants_dump.c"));
        let x = ref_input(1024);
        let mut c = &bytes[..];
        let mut seen = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF {
                break;
            }
            assert_eq!(magic, 0x5443_4553, "section magic");
            let tid = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let qlen = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let (qbytes, rest) = c.split_at(qlen);
            let (deq, rest2) = rest.split_at(n * 4);
            c = rest2;
            let ty = GgmlType::from_u32(tid).unwrap();

            let quant: fn(&[f32], &mut [u8]) = match ty {
                GgmlType::Q1_0 => |x, out| fill(x, out, |x, y: &mut [BlockQ1_0]| quantize_row_q1_0_ref(x, y)),
                GgmlType::Q2_0 => |x, out| fill(x, out, |x, y: &mut [BlockQ2_0]| quantize_row_q2_0_ref(x, y)),
                other => panic!("unexpected type {other:?} in quants_ref_q1q2.bin"),
            };
            let mut mine = vec![0u8; qlen];
            quant(&x[..n], &mut mine);
            assert_eq!(mine.as_slice(), qbytes, "{ty:?}: quantized bytes differ from reference");

            let ref_deq: &[f32] = bytemuck::cast_slice(deq);
            let mut got = vec![0f32; n];
            dequantize_row(ty, qbytes, &mut got);
            for (i, (a, b)) in got.iter().zip(ref_deq).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{ty:?}: dequant differs from reference at {i}: ours={a} ref={b}"
                );
            }
            seen.push(ty);
        }
        assert_eq!(seen, vec![GgmlType::Q1_0, GgmlType::Q2_0], "section set");
    }

    fn fill<B: bytemuck::Pod>(x: &[f32], out: &mut [u8], f: impl Fn(&[f32], &mut [B])) {
        let nb = out.len() / size_of::<B>();
        let blocks: &mut [B] = bytemuck::cast_slice_mut(out);
        debug_assert_eq!(blocks.len(), nb);
        f(x, blocks);
    }

    // ===================== MXFP4 / NVFP4 parity

    /// Bit-exact parity vs the pinned C reference for MXFP4/NVFP4
    /// (parity/quants_ref_fp4.bin, generated by parity/ref_fp4_dump.c).
    ///
    /// 8 sections: 3 row lengths per type produced by the reference
    /// `quantize_row_mxfp4_ref` / `quantize_row_nvfp4_ref` (1024 = full rows,
    /// plus 96/320 for MXFP4 and 192/320 for NVFP4), then 2 hand-built raw
    /// block sections that sweep every E8M0 exponent, every UE4M3 sub-scale
    /// (incl. the 0x00 / 0x7F special cases and the subnormal range) and all
    /// 16 E2M1 nibble codes. Every f32 is compared via `to_bits`, so -0.0 and
    /// ±inf must match exactly too.
    #[test]
    fn fp4_dequant_bit_exact_vs_reference() {
        const MAGIC_SECT: u32 = 0x5443_4553; // quantizer-produced bytes
        const MAGIC_RAW: u32 = 0x5241_5753; // hand-built block bytes

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/quants_ref_fp4.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build it via parity/ref_fp4_dump.c"));
        // (magic, type id, n elements)
        let expected: [(u32, u32, usize); 8] = [
            (MAGIC_SECT, 39, 1024),
            (MAGIC_SECT, 39, 96),
            (MAGIC_SECT, 39, 320),
            (MAGIC_SECT, 40, 1024),
            (MAGIC_SECT, 40, 192),
            (MAGIC_SECT, 40, 320),
            (MAGIC_RAW, 39, 1024),
            (MAGIC_RAW, 40, 1024),
        ];

        let mut c = &bytes[..];
        let mut seen = Vec::new();
        let mut n_mismatch_total = 0usize;
        let mut neg_zeroes = 0usize;
        let mut infinities = 0usize;
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF {
                break; // EOF marker
            }
            assert!(magic == MAGIC_SECT || magic == MAGIC_RAW, "section magic {magic:#x}");
            let tid = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let qlen = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
            c = &c[8..];
            let (qbytes, rest) = c.split_at(qlen);
            let (deq, rest2) = rest.split_at(n * 4);
            c = rest2;
            let ty = GgmlType::from_u32(tid).unwrap();
            assert!(ty == GgmlType::Mxfp4 || ty == GgmlType::Nvfp4, "unexpected {ty:?}");
            assert_eq!(qlen, n / ty.blck_size() * ty.type_size(), "{ty:?} row size");
            seen.push((magic, tid, n));

            // qlen is not 4-byte aligned for every section, so decode the
            // reference f32s through an explicit LE read instead of cast_slice.
            let ref_deq: Vec<f32> = deq
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut mine = vec![0f32; n];
            dequantize_row(ty, qbytes, &mut mine);

            let mut mismatches = 0usize;
            let mut first = usize::MAX;
            for (i, (a, b)) in mine.iter().zip(ref_deq.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    if first == usize::MAX {
                        first = i;
                    }
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "{ty:?} n={n}: {mismatches}/{n} f32 differ; first at {first}: ours={:#010x} ({}) ref={:#010x} ({})",
                mine[first].to_bits(),
                mine[first],
                ref_deq[first].to_bits(),
                ref_deq[first],
            );
            n_mismatch_total += mismatches;
            neg_zeroes += ref_deq.iter().filter(|v| v.to_bits() == 0x8000_0000).count();
            infinities += ref_deq.iter().filter(|v| v.is_infinite()).count();
        }
        assert_eq!(seen.len(), expected.len(), "section count");
        assert_eq!(seen.as_slice(), expected.as_slice(), "section sequence");
        assert_eq!(n_mismatch_total, 0);
        // the raw sections must actually have exercised the signed-zero and
        // overflow paths, otherwise the sweep above proves nothing there.
        assert!(neg_zeroes >= 32, "expected -0.0 samples in the dump, got {neg_zeroes}");
        assert!(infinities >= 32, "expected ±inf samples in the dump, got {infinities}");
    }

    /// Every E8M0 byte maps to exactly 2^(x-128) (the C macro is a bit pattern).
    #[test]
    fn e8m0_scale_boundaries() {
        for x in 0..=255u32 {
            let got = e8m0_to_fp32_half(x as u8);
            // f64 powi keeps the subnormal exponents exact; the cast to f32 is
            // exact for every 2^(x-128), x in 0..=255.
            let want = 2f64.powi(x as i32 - 128) as f32;
            assert_eq!(got.to_bits(), want.to_bits(), "e8m0 byte {x}");
            assert!(got > 0.0 && got.is_finite(), "e8m0 byte {x} should be finite positive");
        }
        assert_eq!(e8m0_to_fp32_half(0).to_bits(), 0x0020_0000); // 2^-128, subnormal
        assert_eq!(e8m0_to_fp32_half(1).to_bits(), 0x0040_0000); // 2^-127 = f32::MIN_POSITIVE
        assert_eq!(e8m0_to_fp32_half(127), 0.5);
        assert_eq!(e8m0_to_fp32_half(128), 1.0);
        assert_eq!(e8m0_to_fp32_half(254).to_bits(), 0x7E80_0000); // 2^126
        assert_eq!(e8m0_to_fp32_half(255).to_bits(), 0x7F00_0000); // 2^127, still finite
        // kvalue 12 * 2^127 overflows to inf; the same kvalue at 2^126 does not.
        assert!((KVALUES_MXFP4[15] as f32 * e8m0_to_fp32_half(255)).is_infinite());
        assert_eq!(KVALUES_MXFP4[15] as f32 * e8m0_to_fp32_half(254), -12.0 * 2f32.powi(126));
        assert_eq!(KVALUES_MXFP4, [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]);
    }

    /// Every UE4M3 byte vs an independent f64 evaluation of the decoder.
    #[test]
    fn ue4m3_scale_boundaries() {
        for x in 0..=255u32 {
            let got = ue4m3_to_fp32(x as u8);
            let exp = (x >> 3) & 0xF;
            let man = (x & 0x7) as f64;
            let want64 = if x == 0 || x == 0x7F {
                0.0
            } else if exp == 0 {
                man * 2f64.powi(-10) // subnormal: man * 2^-9, then * 0.5
            } else {
                (1.0 + man / 8.0) * 2f64.powi(exp as i32 - 8) // ldexp(., exp-7) * 0.5
            };
            let want = want64 as f32;
            assert_eq!(got.to_bits(), want.to_bits(), "ue4m3 byte {x:#04x}");
            assert!(got >= 0.0 && got.is_finite());
        }
        assert_eq!(ue4m3_to_fp32(0x00), 0.0);
        assert_eq!(ue4m3_to_fp32(0x7F), 0.0);
        assert_eq!(ue4m3_to_fp32(0x01).to_bits(), 0x3A80_0000); // subnormal 2^-10
        assert_eq!(ue4m3_to_fp32(0x07).to_bits(), 0x3BE0_0000); // subnormal 7*2^-10
        assert_eq!(ue4m3_to_fp32(0x08).to_bits(), 0x3C00_0000); // smallest normal 2^-7
        assert_eq!(ue4m3_to_fp32(0x40), 1.0); // 1.0 * 2^1 * 0.5
        assert_eq!(ue4m3_to_fp32(0x7E), 224.0); // largest finite: 1.75 * 2^8 * 0.5
        // bit 7 is ignored by the C decoder: 0x80 has exp == 0, man == 0 -> 0.0
        assert_eq!(ue4m3_to_fp32(0x80), 0.0);
        assert_eq!(ue4m3_to_fp32(0x81).to_bits(), ue4m3_to_fp32(0x01).to_bits());
    }

    /// Layout assertions used by `dequantize_row`'s length checks.
    #[test]
    fn fp4_block_geometry() {
        assert_eq!(GgmlType::Mxfp4.blck_size(), 32);
        assert_eq!(GgmlType::Mxfp4.type_size(), 17); // 1 + 32/2
        assert_eq!(GgmlType::Nvfp4.blck_size(), 64);
        assert_eq!(GgmlType::Nvfp4.type_size(), 36); // 4 + 64/2
        assert_eq!(size_of::<BlockMxfp4>(), 17);
        assert_eq!(size_of::<BlockNvfp4>(), 36);
    }
}

/// Dequantize `y.len()` elements of `ty` from `bytes`.
/// `bytes.len()` must equal `n / blck_size * type_size`.
pub fn dequantize_row(ty: GgmlType, bytes: &[u8], y: &mut [f32]) {
    let n = y.len();
    let blck = ty.blck_size();
    assert_eq!(n % blck, 0, "dequantize_row: n not multiple of blck_size");
    let nb = n / blck;
    assert_eq!(
        bytes.len(),
        nb * ty.type_size(),
        "dequantize_row: byte length mismatch for {ty:?}"
    );
    match ty {
        GgmlType::F32 => y.copy_from_slice(bytemuck::cast_slice(bytes)),
        GgmlType::F16 => dequantize_row_f16(bytemuck::cast_slice(bytes), y),
        GgmlType::Bf16 => dequantize_row_bf16(bytemuck::cast_slice(bytes), y),
        GgmlType::Q4_0 => dequantize_row_q4_0(bytemuck::cast_slice(bytes), y),
        GgmlType::Q4_1 => dequantize_row_q4_1(bytemuck::cast_slice(bytes), y),
        GgmlType::Q5_0 => dequantize_row_q5_0(bytemuck::cast_slice(bytes), y),
        GgmlType::Q5_1 => dequantize_row_q5_1(bytemuck::cast_slice(bytes), y),
        GgmlType::Q8_0 => dequantize_row_q8_0(bytemuck::cast_slice(bytes), y),
        GgmlType::Q1_0 => dequantize_row_q1_0(bytemuck::cast_slice(bytes), y),
        GgmlType::Q2_0 => dequantize_row_q2_0(bytemuck::cast_slice(bytes), y),
        GgmlType::Q2K => dequantize_row_q2_K(bytemuck::cast_slice(bytes), y),
        GgmlType::Q3K => dequantize_row_q3_K(bytemuck::cast_slice(bytes), y),
        GgmlType::Q4K => dequantize_row_q4_K(bytemuck::cast_slice(bytes), y),
        GgmlType::Q5K => dequantize_row_q5_K(bytemuck::cast_slice(bytes), y),
        GgmlType::Q6K => dequantize_row_q6_K(bytemuck::cast_slice(bytes), y),
        GgmlType::Mxfp4 => dequantize_row_mxfp4(bytemuck::cast_slice(bytes), y),
        GgmlType::Nvfp4 => dequantize_row_nvfp4(bytemuck::cast_slice(bytes), y),
        // ggml.c:632 type_traits.to_float rows for the IQ family — the
        // functions have been in quants_k.rs since the first IQ round; this
        // dispatch is what `to_float`-consumers (get_rows/dup/quantize
        // conversion, ggml-cpu.c:215's traits table) call.
        GgmlType::Iq2Xxs => crate::quants_k::dequantize_row_iq2_xxs(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq2Xs => crate::quants_k::dequantize_row_iq2_xs(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq2S => crate::quants_k::dequantize_row_iq2_s(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq3Xxs => crate::quants_k::dequantize_row_iq3_xxs(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq3S => crate::quants_k::dequantize_row_iq3_s(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq1S => crate::quants_k::dequantize_row_iq1_s(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq1M => crate::quants_k::dequantize_row_iq1_m(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq4Nl => crate::quants_k::dequantize_row_iq4_nl(bytemuck::cast_slice(bytes), y),
        GgmlType::Iq4Xs => crate::quants_k::dequantize_row_iq4_xs(bytemuck::cast_slice(bytes), y),
        // Documented refusal (AUDIT_ggml.md §5-B.4): the ternary chain is
        // unported end to end (no quantizer, no dequantizer, no vec_dot); the
        // reference implementations live at ggml-quants.c:2428/2467
        // (dequantize_row_tq1_0/tq2_0). Refuse loudly instead of computing
        // wrong values — see PARITY.md's ternary row.
        GgmlType::Tq1_0 | GgmlType::Tq2_0 => panic!(
            "dequantize_row: TQ1_0/TQ2_0 (ternary) blocks are not supported — the whole \
             ternary chain is unported (quantize/dequantize/vec_dot; see PARITY.md)"
        ),
        other => unimplemented!("dequantize_row for {other:?}"),
    }
}

/// The AUDIT_ggml.md §5-A.2 wiring: `dequantize_row`'s dispatch must reach the
/// nine IQ functions that have lived in quants_k.rs since the first IQ round
/// (they are the `type_traits.to_float` rows of ggml.c:632 for IQ types), and
/// must refuse ternary blocks loudly.
#[cfg(test)]
mod iq_dispatch_tests {
    use super::*;
    use crate::quants_k;

    fn lcg_bytes(n: usize) -> Vec<u8> {
        let mut state: u32 = 0x1234_abcd;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 13) as u8
            })
            .collect()
    }

    #[test]
    fn dequantize_row_dispatch_reaches_iq_family() {
        for ty in [
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
        ] {
            let n = ty.blck_size() * 3; // 3 blocks
            let bytes = lcg_bytes(ty.row_size(n));
            let mut via_dispatch = vec![0f32; n];
            dequantize_row(ty, &bytes, &mut via_dispatch);
            let mut direct = vec![0f32; n];
            // same bytes through the quants_k functions the arms call
            match ty {
                GgmlType::Iq2Xxs => {
                    quants_k::dequantize_row_iq2_xxs(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq2Xs => {
                    quants_k::dequantize_row_iq2_xs(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq2S => {
                    quants_k::dequantize_row_iq2_s(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq3Xxs => {
                    quants_k::dequantize_row_iq3_xxs(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq3S => {
                    quants_k::dequantize_row_iq3_s(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq1S => {
                    quants_k::dequantize_row_iq1_s(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq1M => {
                    quants_k::dequantize_row_iq1_m(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq4Nl => {
                    quants_k::dequantize_row_iq4_nl(bytemuck::cast_slice(&bytes), &mut direct)
                }
                GgmlType::Iq4Xs => {
                    quants_k::dequantize_row_iq4_xs(bytemuck::cast_slice(&bytes), &mut direct)
                }
                _ => unreachable!(),
            }
            for (i, (a, b)) in via_dispatch.iter().zip(&direct).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{ty:?}: element {i} differs");
            }
        }
    }

    /// TQ1_0/TQ2_0 are the documented refusal: no dequantizer exists in the
    /// port (ggml-quants.c:2428/2467 unported), so the dispatch must panic
    /// instead of producing values.
    #[test]
    #[should_panic(expected = "ternary")]
    fn dequantize_row_refuses_ternary() {
        let bytes = lcg_bytes(GgmlType::Tq1_0.row_size(GgmlType::Tq1_0.blck_size()));
        let mut y = vec![0f32; GgmlType::Tq1_0.blck_size()];
        dequantize_row(GgmlType::Tq1_0, &bytes, &mut y);
    }
}
