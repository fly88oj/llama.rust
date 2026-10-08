//! quants_impl.rs — the `quant_weights != NULL` half of ggml-quants.c @
//! bd4f514db1: the `*_impl` quantizers that `llama-quantize --imatrix` selects
//! (llama-quant.cpp:747-808 `llama_tensor_quantize_impl` →
//! `ggml_quantize_chunk` → `quantize_<type>(..., imatrix)`).
//!
//! Every `quantize_row_*` entry point keeps the C name and the C's
//! `(x, y, n, quant_weights)` shape; the dispatcher
//! [`quantize_row_weighted`] is the port of the C's per-type wrapper
//! (`quantize_q2_K`, `quantize_iq4_xs`, ... ggml-quants.c:1213-1227 etc.),
//! which is just "use `*_impl` when weights are given, else the `*_ref`".
//!
//! Scope: the types `llama_tensor_get_type` can ask for on a CPU-only port —
//! Q4_0/Q4_1/Q5_0/Q5_1, Q2_K..Q6_K (all weighted variants) and IQ4_NL/IQ4_XS.
//! The IQ2_*/IQ3_*/IQ1_* quantizers additionally need the runtime
//! `iq2xs_init_impl`/`iq3xs_init_impl` neighbour tables; they are NOT ported
//! (see PARITY.md "imatrix" — that is what gates the remaining IQ ftypes).
//!
//! The K-quant helpers (`make_qx_quants`, `make_qkx3_quants`, `make_qp_quants`,
//! `get_scale_min_k4`) are private in `quants_k.rs` and are copied here
//! verbatim (same C origin, already verified bit-exact there) rather than made
//! `pub`, to keep the quantizer files single-owner.

use crate::blocks::*;
use crate::quants::nearest_int;
use crate::types::GgmlType;
use half::f16;

/// ggml-quants.c `GROUP_MAX_EPS`
const GROUP_MAX_EPS: f32 = 1e-15;

#[inline]
fn fp16(x: f32) -> f16 {
    f16::from_f32(x)
}
#[inline]
fn f32_of(x: f16) -> f32 {
    x.to_f32()
}

/// ggml-quants.c `best_index_int8` (ggml-quants.c:28-36): binary search over a
/// sorted `int8` codebook; ties pick the smaller magnitude.
fn best_index_int8(n: usize, val: &[i8], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[n - 1] as f32 {
        return n - 1;
    }
    let mut ml = 0usize;
    let mut mu = n - 1;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (val[mu - 1] as f32) < (val[mu] as f32) - x {
        mu - 1
    } else {
        mu
    }
}

// ===========================================================================
// K-quant helpers (copies of the private ones in quants_k.rs)
// ===========================================================================

/// ggml-quants.c `make_qx_quants`
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], rmse_type: i32, qw: Option<&[f32]>) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        // all zero
        for v in &mut l[..n] {
            *v = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            l[i] = (nmax + li.min(nmax - 1).max(-nmax)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let weight = |i: usize| match qw {
        Some(q) => q[i],
        None => match rmse_type {
            1 => x[i] * x[i],
            2 => 1.0,
            3 => x[i].abs(),
            _ => x[i].abs().sqrt(),
        },
    };
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        li = li.min(nmax - 1).max(-nmax);
        l[i] = (li + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 { 0.5 * (scale + 1.0 / iscale) } else { 1.0 / iscale };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1f32 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let mut li = nearest_int(iscale * x[i]);
            li = li.min(nmax - 1).max(-nmax);
            let w = weight(i);
            sumlx += w * x[i] * li as f32;
            suml2 += w * li as f32 * li as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                l[i] = (nmax + li.min(nmax - 1).max(-nmax)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml-quants.c `make_qkx3_quants`
#[allow(clippy::too_many_arguments)]
fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let w_of = |i: usize| weights.map_or(x[i] * x[i], |w| w[i]);
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = w_of(0);
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = w_of(i);
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        for v in &mut l[..n] {
            *v = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        l[i] = li.min(nmax).max(0) as u8;
        let mut diff = scale * l[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = w_of(i);
        best_mad += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).min(nmax).max(0);
            laux[i] = li as u8;
            let w = w_of(i);
            sum_l += w * li as f32;
            sum_l2 += w * li as f32 * li as f32;
            sum_xl += w * li as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = w_of(i);
                mad += w * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// ggml-quants.c `make_qp_quants`
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        max = max.max(x[i]);
    }
    if max < GROUP_MAX_EPS {
        // all zero
        for v in &mut l[..n] {
            *v = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1f32 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).min(nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let mut slx = sumlx - w * x[i] * l[i] as f32;
            let mut sl2 = suml2 - w * l[i] as f32 * l[i] as f32;
            if slx > 0.0 && sl2 > 0.0 {
                let mut new_l = nearest_int(x[i] * sl2 / slx);
                new_l = new_l.min(nmax);
                if new_l != l[i] as i32 {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// ggml-quants.c `get_scale_min_k4`
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ===========================================================================
// Q2_K..Q6_K, imatrix variants (ggml-quants.c:1149-2067)
// ===========================================================================

/// ggml-quants.c `quantize_row_q2_K_impl` (ggml-quants.c:1149-1210).
///
/// Note the C's `for (int l = 0; l < QK_K/16; ++l) sw[j] += weight[l];` — the
/// 16 per-quant weights are accumulated into `sw[j]`, not `sw[l]` (kept
/// literally).
pub fn quantize_row_q2_K_impl(x: &[f32], y: &mut [BlockQ2K], quant_weights: &[f32]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 16];
    let mut mins = [0f32; QK_K / 16];
    let mut scales = [0f32; QK_K / 16];
    // the C resets `sw` with memset at the top of every block
    #[allow(unused_assignments)]
    let mut sw = [0f32; QK_K / 16];
    let mut weight = [0f32; 16];
    let mut ls = [0u8; QK_K / 16];
    let mut lm = [0u8; QK_K / 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        sw = [0f32; QK_K / 16];
        let mut sumx2 = 0f32;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = sumx2 / QK_K as f32;
        for j in 0..QK_K / 16 {
            let qw = &quant_weights[QK_K * i + 16 * j..];
            for li in 0..16 {
                weight[li] = qw[li] * (sigma2 + xb[16 * j + li] * xb[16 * j + li]).sqrt();
            }
            for li in 0..QK_K / 16 {
                sw[j] += weight[li];
            }
            scales[j] = make_qkx3_quants(
                16,
                3,
                &xb[16 * j..16 * j + 16],
                Some(&weight),
                &mut l[16 * j..16 * j + 16],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let mut dm = make_qp_quants(QK_K / 16, 15, &scales, &mut ls, &sw);
        let mut mm = make_qp_quants(QK_K / 16, 15, &mins, &mut lm, &sw);

        y[i].d = fp16(dm);
        y[i].dmin = fp16(mm);
        dm = f32_of(y[i].d);
        mm = f32_of(y[i].dmin);

        let mut scales_b = [0u8; QK_K / 16];
        for j in 0..QK_K / 16 {
            scales_b[j] = ls[j] | (lm[j] << 4);
        }

        // requantize (the C's `const bool requantize = true`)
        for j in 0..QK_K / 16 {
            let d = dm * (scales_b[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let m = mm * (scales_b[j] >> 4) as f32;
            for ii in 0..16 {
                let li = nearest_int((xb[16 * j + ii] + m) / d);
                l[16 * j + ii] = li.min(3).max(0) as u8;
            }
        }

        let mut qs = [0u8; QK_K / 4];
        for j in (0..QK_K).step_by(128) {
            for li in 0..32 {
                qs[j / 4 + li] =
                    l[j + li] | (l[j + li + 32] << 2) | (l[j + li + 64] << 4) | (l[j + li + 96] << 6);
            }
        }
        y[i].scales = scales_b;
        y[i].qs = qs;
    }
}

/// ggml-quants.c `quantize_row_q3_K_impl` (ggml-quants.c:1355-1428).
pub fn quantize_row_q3_K_impl(x: &[f32], y: &mut [BlockQ3K], quant_weights: &[f32]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);

    let mut l = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];
    let mut weight = [0f32; 16];
    let mut sw = [0f32; QK_K / 16];
    let mut ls = [0i8; QK_K / 16];

    for (i, out) in y.iter_mut().enumerate() {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut sumx2 = 0f32;
        for j in 0..QK_K {
            sumx2 += xb[j] * xb[j];
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for j in 0..QK_K / 16 {
            let qw = &quant_weights[QK_K * i + 16 * j..];
            for li in 0..16 {
                weight[li] = qw[li] * (sigma2 + xb[16 * j + li] * xb[16 * j + li]).sqrt();
            }
            let mut sumw = 0f32;
            for li in 0..16 {
                sumw += weight[li];
            }
            sw[j] = sumw;
            scales[j] = make_qx_quants(16, 4, &xb[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16], 1, Some(&weight));
        }

        let mut scales_b = [0u8; 12];
        let d_block = make_qx_quants(QK_K / 16, 32, &scales, &mut ls, 1, Some(&sw));
        for j in 0..QK_K / 16 {
            let mut lj = ls[j] as i32;
            if j < 8 {
                scales_b[j] = (lj & 0xF) as u8;
            } else {
                scales_b[j - 8] |= ((lj & 0xF) << 4) as u8;
            }
            lj >>= 4;
            scales_b[j % 4 + 8] |= (lj << (2 * (j / 4))) as u8;
        }
        let d = fp16(d_block);

        for j in 0..QK_K / 16 {
            let mut sc: i32 = if j < 8 {
                (scales_b[j] & 0xF) as i32
            } else {
                (scales_b[j - 8] >> 4) as i32
            };
            sc = (sc | ((((scales_b[8 + j % 4] >> (2 * (j / 4))) & 3) as i32) << 4)) - 32;
            let dd = f32_of(d) * sc as f32;
            if dd == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let li = nearest_int(xb[16 * j + ii] / dd);
                l[16 * j + ii] = (li.min(3).max(-4) + 4) as i8;
            }
        }

        let mut hmask = [0u8; QK_K / 8];
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into
        // bit 1, etc.
        let mut m = 0usize;
        let mut hm: u8 = 1;
        for j in 0..QK_K {
            if l[j] > 3 {
                hmask[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        let mut qs = [0u8; QK_K / 4];
        for j in (0..QK_K).step_by(128) {
            for li in 0..32 {
                qs[j / 4 + li] = ((l[j + li] as i32)
                    | ((l[j + li + 32] as i32) << 2)
                    | ((l[j + li + 64] as i32) << 4)
                    | ((l[j + li + 96] as i32) << 6)) as u8;
            }
        }
        *out = BlockQ3K { hmask, qs, scales: scales_b, d };
    }
}

/// ggml-quants.c `quantize_row_q4_K_impl` (ggml-quants.c:1553-1621).
pub fn quantize_row_q4_K_impl(x: &[f32], y: &mut [BlockQ4K], quant_weights: &[f32]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut ls = [0u8; QK_K / 32];
    let mut lm = [0u8; QK_K / 32];
    let mut weights = [0f32; 32];
    let mut sw = [0f32; QK_K / 32];
    let mut mins = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut sum_x2 = 0f32;
        for li in 0..QK_K {
            sum_x2 += xb[li] * xb[li];
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;

        for j in 0..QK_K / 32 {
            let qw = &quant_weights[QK_K * i + 32 * j..];
            for li in 0..32 {
                weights[li] = qw[li] * (sigma2 + xb[32 * j + li] * xb[32 * j + li]).sqrt();
            }
            let mut sumw = 0f32;
            for li in 0..32 {
                sumw += weights[li];
            }
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(
                32,
                15,
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);
        let mut scales_b = [0u8; K_SCALE_SIZE];
        for j in 0..QK_K / 32 {
            let lsj = ls[j];
            let lmj = lm[j];
            if j < 4 {
                scales_b[j] = lsj;
                scales_b[j + 4] = lmj;
            } else {
                scales_b[j + 4] = (lsj & 0xF) | ((lmj & 0xF) << 4);
                scales_b[j - 4] |= (lsj >> 4) << 6;
                scales_b[j] |= (lmj >> 4) << 6;
            }
        }
        let d = fp16(d_block);
        let dmin = fp16(m_block);

        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &scales_b);
            let dd = f32_of(d) * sc as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = f32_of(dmin) * m as f32;
            for ii in 0..32 {
                let li = nearest_int((xb[32 * j + ii] + dm) / dd);
                l[32 * j + ii] = li.min(15).max(0) as u8;
            }
        }
        let mut qs = [0u8; QK_K / 2];
        let mut q = 0usize;
        for j in (0..QK_K).step_by(64) {
            for li in 0..32 {
                qs[q + li] = l[j + li] | (l[j + li + 32] << 4);
            }
            q += 32;
        }

        y[i].d = d;
        y[i].dmin = dmin;
        y[i].scales = scales_b;
        y[i].qs = qs;
    }
}

/// ggml-quants.c `quantize_row_q5_K_impl` (ggml-quants.c:1758-1843).
pub fn quantize_row_q5_K_impl(x: &[f32], y: &mut [BlockQ5K], quant_weights: &[f32]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);

    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut ls = [0u8; QK_K / 32];
    let mut lm = [0u8; QK_K / 32];
    let mut mins = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];
    let mut sw = [0f32; QK_K / 32];
    let mut weights = [0f32; 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut sum_x2 = 0f32;
        for li in 0..QK_K {
            sum_x2 += xb[li] * xb[li];
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;

        for j in 0..QK_K / 32 {
            let qw = &quant_weights[QK_K * i + 32 * j..];
            for li in 0..32 {
                weights[li] = qw[li] * (sigma2 + xb[32 * j + li] * xb[32 * j + li]).sqrt();
            }
            let mut sumw = 0f32;
            for li in 0..32 {
                sumw += weights[li];
            }
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(
                32,
                31,
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);

        let mut scales_b = [0u8; K_SCALE_SIZE];
        for j in 0..QK_K / 32 {
            let lsj = ls[j].min(63);
            let lmj = lm[j].min(63);
            if j < 4 {
                scales_b[j] = lsj;
                scales_b[j + 4] = lmj;
            } else {
                scales_b[j + 4] = (lsj & 0xF) | ((lmj & 0xF) << 4);
                scales_b[j - 4] |= (lsj >> 4) << 6;
                scales_b[j] |= (lmj >> 4) << 6;
            }
        }
        let d = fp16(d_block);
        let dmin = fp16(m_block);

        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &scales_b);
            let dd = f32_of(d) * sc as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = f32_of(dmin) * m as f32;
            for ii in 0..32 {
                let li = nearest_int((xb[32 * j + ii] + dm) / dd);
                l[32 * j + ii] = li.min(31).max(0) as u8;
            }
        }

        let mut qh = [0u8; QK_K / 8];
        let mut qs = [0u8; QK_K / 2];
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        let mut ql = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[ql + j] = (l1 | (l2 << 4)) as u8;
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }

        y[i].d = d;
        y[i].dmin = dmin;
        y[i].scales = scales_b;
        y[i].qh = qh;
        y[i].qs = qs;
    }
}

/// ggml-quants.c `quantize_row_q6_K_impl` (ggml-quants.c:1970-2052).
pub fn quantize_row_q6_K_impl(x: &[f32], y: &mut [BlockQ6K], quant_weights: &[f32]) {
    let nb = y.len();
    debug_assert_eq!(x.len(), nb * QK_K);

    let mut l = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;

        for ib in 0..QK_K / 16 {
            let qw = &quant_weights[QK_K * i + 16 * ib..];
            let scale =
                make_qx_quants(16, 32, &xb[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16], 1, Some(qw));
            scales[ib] = scale;

            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if max_abs_scale < GROUP_MAX_EPS {
            y[i] = BlockQ6K {
                ql: [0; QK_K / 2],
                qh: [0; QK_K / 4],
                scales: [0; QK_K / 16],
                d: fp16(0.0),
            };
            continue;
        }

        let iscale = -128.0f32 / max_scale;
        let d = fp16(1.0 / iscale);
        let mut scales_b = [0i8; QK_K / 16];
        for ib in 0..QK_K / 16 {
            scales_b[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        }

        for j in 0..QK_K / 16 {
            let dd = f32_of(d) * scales_b[j] as f32;
            if dd == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let li = nearest_int(xb[16 * j + ii] / dd);
                l[16 * j + ii] = (li.min(31).max(-32) + 32) as i8;
            }
        }

        let mut ql = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 4];
        let mut qli = 0usize;
        let mut qhi = 0usize;
        for j in (0..QK_K).step_by(128) {
            for li in 0..32usize {
                let q1 = (l[j + li] & 0xF) as u8;
                let q2 = (l[j + li + 32] & 0xF) as u8;
                let q3 = (l[j + li + 64] & 0xF) as u8;
                let q4 = (l[j + li + 96] & 0xF) as u8;
                ql[qli + li] = q1 | (q3 << 4);
                ql[qli + 32 + li] = q2 | (q4 << 4);
                qh[qhi + li] = (((l[j + li] as i32) >> 4)
                    | (((l[j + li + 32] as i32) >> 4) << 2)
                    | (((l[j + li + 64] as i32) >> 4) << 4)
                    | (((l[j + li + 96] as i32) >> 4) << 6)) as u8;
            }
            qli += 64;
            qhi += 32;
        }

        y[i] = BlockQ6K { ql, qh, scales: scales_b, d };
    }
}

// ===========================================================================
// Q4_0/Q4_1/Q5_0/Q5_1, imatrix variants (ggml-quants.c:2070-2296)
// ===========================================================================

/// ggml-quants.c `quantize_row_q4_0_impl` (ggml-quants.c:2070-2097).
pub fn quantize_row_q4_0_impl(x: &[f32], y: &mut [BlockQ4_0], quant_weights: &[f32]) {
    let n_per_row = x.len();
    let mut weight = [0f32; QK4_0];
    let mut l = [0i8; QK4_0];

    let mut sum_x2 = 0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = y.len();
    for ib in 0..nb {
        let xb = &x[QK4_0 * ib..QK4_0 * ib + QK4_0];
        let qw = &quant_weights[QK4_0 * ib..];
        for j in 0..QK4_0 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let d = make_qx_quants(QK4_0, 8, xb, &mut l, 1, Some(&weight));
        let mut qs = [0u8; QK4_0 / 2];
        for j in 0..16 {
            qs[j] = (l[j] as u8) | ((l[j + 16] as u8) << 4);
        }
        y[ib] = BlockQ4_0 { d: fp16(d), qs };
    }
}

/// ggml-quants.c `quantize_row_q4_1_impl` (ggml-quants.c:2143-2172).
pub fn quantize_row_q4_1_impl(x: &[f32], y: &mut [BlockQ4_1], quant_weights: &[f32]) {
    let n_per_row = x.len();
    let mut weight = [0f32; QK4_1];
    let mut l = [0u8; QK4_1];
    let mut laux = [0u8; QK4_1];

    let mut sum_x2 = 0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = y.len();
    for ib in 0..nb {
        let xb = &x[QK4_1 * ib..QK4_1 * ib + QK4_1];
        let qw = &quant_weights[QK4_1 * ib..];
        for j in 0..QK4_1 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let mut min = 0f32;
        let d = make_qkx3_quants(QK4_1, 15, xb, Some(&weight), &mut l, &mut min, &mut laux, -0.9, 0.05, 36, false);
        let mut qs = [0u8; QK4_1 / 2];
        for j in 0..16 {
            qs[j] = l[j] | (l[j + 16] << 4);
        }
        y[ib] = BlockQ4_1 { d: fp16(d), m: fp16(-min), qs };
    }
}

/// ggml-quants.c `quantize_row_q5_0_impl` (ggml-quants.c:2188-2225).
pub fn quantize_row_q5_0_impl(x: &[f32], y: &mut [BlockQ5_0], quant_weights: &[f32]) {
    let n_per_row = x.len();
    let mut weight = [0f32; QK5_0];
    let mut l = [0i8; QK5_0];

    let mut sum_x2 = 0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = y.len();
    for ib in 0..nb {
        let xb = &x[QK5_0 * ib..QK5_0 * ib + QK5_0];
        let qw = &quant_weights[QK5_0 * ib..];
        for j in 0..QK5_0 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let d = make_qx_quants(QK5_0, 16, xb, &mut l, 1, Some(&weight));

        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_0 / 2];
        for j in 0..16 {
            let xi0 = l[j] as u8;
            let xi1 = l[j + 16] as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_0 / 2);
        }
        y[ib] = BlockQ5_0 { d: fp16(d), qh: qh.to_le_bytes(), qs };
    }
}

/// ggml-quants.c `quantize_row_q5_1_impl` (ggml-quants.c:2242-2273).
pub fn quantize_row_q5_1_impl(x: &[f32], y: &mut [BlockQ5_1], quant_weights: &[f32]) {
    let n_per_row = x.len();
    let mut weight = [0f32; QK5_1];
    let mut l = [0u8; QK5_1];
    let mut laux = [0u8; QK5_1];

    let mut sum_x2 = 0f32;
    for j in 0..n_per_row {
        sum_x2 += x[j] * x[j];
    }
    let sigma2 = sum_x2 / n_per_row as f32;

    let nb = y.len();
    for ib in 0..nb {
        let xb = &x[QK5_1 * ib..QK5_1 * ib + QK5_1];
        let qw = &quant_weights[QK5_1 * ib..];
        for j in 0..QK5_1 {
            weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
        }
        let mut min = 0f32;
        let d = make_qkx3_quants(QK5_1, 31, xb, Some(&weight), &mut l, &mut min, &mut laux, -0.9, 0.05, 36, false);

        let mut qh: u32 = 0;
        let mut qs = [0u8; QK5_1 / 2];
        for j in 0..16 {
            let xi0 = l[j];
            let xi1 = l[j + 16];
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK5_0 / 2);
        }
        y[ib] = BlockQ5_1 { d: fp16(d), m: fp16(-min), qh: qh.to_le_bytes(), qs };
    }
}

// ===========================================================================
// IQ4_NL / IQ4_XS (ggml-quants.c:4937-5104)
// ===========================================================================

/// ggml-quants.c `quantize_row_iq4_nl_impl` (ggml-quants.c:4937-5046).
///
/// Used for both IQ4_NL (`super_block_size == QK4_NL`, no scales) and IQ4_XS
/// (`QK_K` with per-32 `scales_h`/`scales_l`).
#[allow(clippy::too_many_arguments)]
fn quantize_row_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],
    dh: &mut f16,
    q4: &mut [u8],
    scales_h: Option<&mut [u16]>,
    scales_l: Option<&mut [u8]>,
    scales: &mut [f32],
    weight: &mut [f32],
    l: &mut [u8],
    values: &[i8],
    quant_weights: Option<&[f32]>,
    ntry: i32,
) {
    let mut sigma2 = 0f32;
    for j in 0..super_block_size {
        sigma2 += x[j] * x[j];
    }
    sigma2 *= 2.0 / super_block_size as f32;

    for v in q4[..super_block_size / 2].iter_mut() {
        *v = 0;
    }
    *dh = fp16(0.0);

    let mut max_scale = 0f32;
    let mut amax_scale = 0f32;
    for ib in 0..super_block_size / block_size {
        let xb = &x[ib * block_size..ib * block_size + block_size];
        let lb = &mut l[ib * block_size..ib * block_size + block_size];
        match quant_weights {
            Some(qw) => {
                let qw = &qw[ib * block_size..];
                for j in 0..block_size {
                    weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                }
            }
            None => {
                for j in 0..block_size {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let mut amax = 0f32;
        let mut max = 0f32;
        for j in 0..block_size {
            let ax = xb[j].abs();
            if ax > amax {
                amax = ax;
                max = xb[j];
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let mut d = if ntry > 0 { -max / values[0] as f32 } else { max / values[0] as f32 };
        let mut id = 1.0 / d;
        let mut sumqx = 0f32;
        let mut sumq2 = 0f32;
        for j in 0..block_size {
            let al = id * xb[j];
            let li = best_index_int8(16, values, al);
            lb[j] = li as u8;
            let q = values[li] as f32;
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -ntry..=ntry {
            id = (itry as f32 + values[0] as f32) / max;
            sumqx = 0.0;
            sumq2 = 0.0;
            for j in 0..block_size {
                let al = id * xb[j];
                let li = best_index_int8(16, values, al);
                let q = values[li] as f32;
                let w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if super_block_size / block_size > 1 {
        let nb = super_block_size / block_size;
        let scales_h = scales_h.expect("IQ4_XS needs scales_h");
        let scales_l = scales_l.expect("IQ4_XS needs scales_l");
        for v in scales_h[..(nb + 7) / 8].iter_mut() {
            *v = 0;
        }
        let d = -max_scale / 32.0;
        *dh = fp16(d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..super_block_size / block_size {
            let mut lj = nearest_int(id * scales[ib]);
            lj = lj.min(31).max(-32);
            let dl = d * lj as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let lb = &mut l[ib * block_size..ib * block_size + block_size];
            let xb = &x[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                lb[j] = best_index_int8(16, values, idl * xb[j]) as u8;
            }
            lj += 32;
            let l_l = (lj & 0xf) as u8;
            let l_h = (lj >> 4) as u16;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h[ib / 8] |= l_h << (2 * (ib % 8));
        }
    } else {
        *dh = fp16(scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 { 1.0 / scales[0] } else { 0.0 };
            for j in 0..super_block_size {
                l[j] = best_index_int8(16, values, id * x[j]) as u8;
            }
        }
    }

    for i in 0..super_block_size / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
}

/// `quantize_iq4_nl` (ggml-quants.c:5048-5069): one 32-block at a time, with
/// its own slice of the weights.
pub fn quantize_iq4_nl(x: &[f32], y: &mut [BlockIq4Nl], quant_weights: Option<&[f32]>) {
    let n_per_row = x.len();
    let nblock = n_per_row / QK4_NL;
    let mut l = [0u8; QK4_NL];
    let mut weight = [0f32; QK4_NL];
    // C passes `&scale` (a single float) — the impl only writes it
    let mut scale = [0f32; 1];
    for ib in 0..nblock {
        let qw = quant_weights.map(|qw| &qw[QK4_NL * ib..]);
        let mut dh = f16::ZERO;
        let mut qs = [0u8; QK4_NL / 2];
        let mut unused_h = [0u16; 1];
        let mut unused_l = [0u8; 1];
        quantize_row_iq4_nl_impl(
            QK4_NL,
            32,
            &x[QK4_NL * ib..],
            &mut dh,
            &mut qs,
            Some(&mut unused_h),
            Some(&mut unused_l),
            &mut scale,
            &mut weight,
            &mut l,
            &crate::quants_k::KVALUES_IQ4NL,
            qw,
            7,
        );
        y[ib] = BlockIq4Nl { d: dh, qs };
    }
}

/// `quantize_iq4_xs` (ggml-quants.c:5086-5104).
pub fn quantize_iq4_xs(x: &[f32], y: &mut [BlockIq4Xs], quant_weights: Option<&[f32]>) {
    let n_per_row = x.len();
    let nblock = n_per_row / QK_K;
    let mut l = [0u8; QK_K];
    let mut weight = [0f32; 32];
    let mut scales = [0f32; QK_K / 32];
    for ib in 0..nblock {
        let qw = quant_weights.map(|qw| &qw[QK_K * ib..]);
        let mut dh = f16::ZERO;
        let mut qs = [0u8; QK_K / 2];
        let mut scales_h = [0u16; QK_K / 64];
        let mut scales_l = [0u8; QK_K / 64];
        quantize_row_iq4_nl_impl(
            QK_K,
            32,
            &x[QK_K * ib..],
            &mut dh,
            &mut qs,
            Some(&mut scales_h),
            Some(&mut scales_l),
            &mut scales,
            &mut weight,
            &mut l,
            &crate::quants_k::KVALUES_IQ4NL,
            qw,
            7,
        );
        y[ib] = BlockIq4Xs { d: dh, scales_h: scales_h[0], scales_l, qs };
    }
}

// ===========================================================================
// dispatcher — the C's per-type wrapper (ggml-quants.c:1213-1227 etc.)
// ===========================================================================

/// Quantize one row (or `n_per_row`-element chunk) of `ty`, using the imatrix
/// weights when the type has a weighted variant — the port of
/// `quantize_<type>(src, dst, nrow=1, n_per_row, quant_weights)` dispatched by
/// `ggml_quantize_chunk` (ggml.c:8057-8139).
///
/// `dst` must be at least `ty.row_size(n_per_row)` bytes.
pub fn quantize_row_weighted(
    ty: GgmlType,
    x: &[f32],
    n_per_row: i64,
    dst: &mut [u8],
    quant_weights: Option<&[f32]>,
) -> Result<usize, String> {
    let npr = n_per_row as usize;
    let row_size = ty.row_size(npr);
    debug_assert!(dst.len() >= row_size);
    debug_assert!(x.len() >= npr);

    macro_rules! as_blocks {
        ($bt:ty) => {
            bytemuck::cast_slice_mut::<u8, $bt>(&mut dst[..row_size])
        };
    }

    match ty {
        GgmlType::Q4_0 => match quant_weights {
            Some(qw) => quantize_row_q4_0_impl(&x[..npr], as_blocks!(BlockQ4_0), qw),
            None => crate::quants::quantize_row_q4_0_ref(&x[..npr], as_blocks!(BlockQ4_0)),
        },
        GgmlType::Q4_1 => match quant_weights {
            Some(qw) => quantize_row_q4_1_impl(&x[..npr], as_blocks!(BlockQ4_1), qw),
            None => crate::quants::quantize_row_q4_1_ref(&x[..npr], as_blocks!(BlockQ4_1)),
        },
        GgmlType::Q5_0 => match quant_weights {
            Some(qw) => quantize_row_q5_0_impl(&x[..npr], as_blocks!(BlockQ5_0), qw),
            None => crate::quants::quantize_row_q5_0_ref(&x[..npr], as_blocks!(BlockQ5_0)),
        },
        GgmlType::Q5_1 => match quant_weights {
            Some(qw) => quantize_row_q5_1_impl(&x[..npr], as_blocks!(BlockQ5_1), qw),
            None => crate::quants::quantize_row_q5_1_ref(&x[..npr], as_blocks!(BlockQ5_1)),
        },
        GgmlType::Q2K => match quant_weights {
            Some(qw) => quantize_row_q2_K_impl(&x[..npr], as_blocks!(BlockQ2K), qw),
            None => crate::quants_k::quantize_row_q2_K_ref(&x[..npr], as_blocks!(BlockQ2K)),
        },
        GgmlType::Q3K => match quant_weights {
            Some(qw) => quantize_row_q3_K_impl(&x[..npr], as_blocks!(BlockQ3K), qw),
            None => crate::quants_k::quantize_row_q3_K_ref(&x[..npr], as_blocks!(BlockQ3K)),
        },
        GgmlType::Q4K => match quant_weights {
            Some(qw) => quantize_row_q4_K_impl(&x[..npr], as_blocks!(BlockQ4K), qw),
            None => crate::quants_k::quantize_row_q4_K_ref(&x[..npr], as_blocks!(BlockQ4K)),
        },
        GgmlType::Q5K => match quant_weights {
            Some(qw) => quantize_row_q5_K_impl(&x[..npr], as_blocks!(BlockQ5K), qw),
            None => crate::quants_k::quantize_row_q5_K_ref(&x[..npr], as_blocks!(BlockQ5K)),
        },
        GgmlType::Q6K => match quant_weights {
            Some(qw) => quantize_row_q6_K_impl(&x[..npr], as_blocks!(BlockQ6K), qw),
            None => crate::quants_k::quantize_row_q6_K_ref(&x[..npr], as_blocks!(BlockQ6K)),
        },
        GgmlType::Iq4Nl => quantize_iq4_nl(&x[..npr], as_blocks!(BlockIq4Nl), quant_weights),
        GgmlType::Iq4Xs => quantize_iq4_xs(&x[..npr], as_blocks!(BlockIq4Xs), quant_weights),
        other => {
            return Err(format!(
                "no imatrix-aware quantizer for type {} in this port",
                other.name()
            ))
        }
    }
    Ok(row_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every weighted quantizer must be a *pure* function of (x, weights): the
    /// no-weights path must reproduce the already-verified `*_ref` bytes, and
    /// the weighted path must differ (the weights actually matter) — the smoke
    /// test that the dispatch is wired to two distinct code paths.
    #[test]
    fn weighted_and_ref_paths_are_distinct() {
        let n = 256usize;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let qw: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32).collect();

        for ty in [GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let row = ty.row_size(n);
            let mut a = vec![0u8; row];
            let mut b = vec![0u8; row];
            quantize_row_weighted(ty, &x, n as i64, &mut a, None).unwrap();
            quantize_row_weighted(ty, &x, n as i64, &mut b, Some(&qw)).unwrap();
            assert_ne!(a, b, "{ty:?}: imatrix weights must change the output");
            // the no-weights path is the ref quantizer
            let mut c = vec![0u8; row];
            match ty {
                GgmlType::Q2K => crate::quants_k::quantize_row_q2_K_ref(&x, bytemuck::cast_slice_mut(&mut c)),
                GgmlType::Q3K => crate::quants_k::quantize_row_q3_K_ref(&x, bytemuck::cast_slice_mut(&mut c)),
                GgmlType::Q4K => crate::quants_k::quantize_row_q4_K_ref(&x, bytemuck::cast_slice_mut(&mut c)),
                GgmlType::Q5K => crate::quants_k::quantize_row_q5_K_ref(&x, bytemuck::cast_slice_mut(&mut c)),
                GgmlType::Q6K => crate::quants_k::quantize_row_q6_K_ref(&x, bytemuck::cast_slice_mut(&mut c)),
                _ => unreachable!(),
            }
            assert_eq!(a, c, "{ty:?}: no weights must equal the *_ref path");
        }
    }

    /// IQ4_NL / IQ4_XS: the weights change the codes. (There is no `*_ref` entry
    /// point for these types in the pinned C — `ggml_quantize_chunk` always
    /// calls `quantize_iq4_nl`/`quantize_iq4_xs`, which pass the weights
    /// through, so both branches are this impl; its byte-exactness is pinned by
    /// the `llama-quantize --imatrix` parity run, not here.)
    #[test]
    fn iq4_uses_weights() {
        let n = 256usize;
        let x: Vec<f32> = (0..n).map(|i| ((i * 53 % 97) as f32 - 48.0) / 40.0).collect();
        let qw: Vec<f32> = (0..n).map(|i| 0.25 + (i % 5) as f32).collect();

        for ty in [GgmlType::Iq4Nl, GgmlType::Iq4Xs] {
            let row = ty.row_size(n);
            let mut a = vec![0u8; row];
            let mut b = vec![0u8; row];
            quantize_row_weighted(ty, &x, n as i64, &mut a, None).unwrap();
            quantize_row_weighted(ty, &x, n as i64, &mut b, Some(&qw)).unwrap();
            assert_ne!(a, b, "{ty:?}: imatrix weights must change the output");
            // a weight vector of ones is *not* the unweighted path (the impl
            // branches on the pointer, ggml-quants.c:4949-4955)
            let ones = vec![1.0f32; n];
            let mut c = vec![0u8; row];
            quantize_row_weighted(ty, &x, n as i64, &mut c, Some(&ones)).unwrap();
            assert_ne!(a, c, "{ty:?}: the weighted branch is a different code path");
        }
    }
}