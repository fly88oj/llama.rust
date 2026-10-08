//! rows.rs — row-level dequantization / quantization / validation helpers.
//!
//! Ports of:
//! * `llama_tensor_dequantize_impl`   (llama-quant.cpp:216-281)
//! * `llama_tensor_quantize_impl`     (llama-quant.cpp:747-816)
//! * `ggml_quantize_chunk` dispatch   (ggml.c:8057-8140) — one `*_ref` call
//!   per row; with `imatrix == NULL` every `quantize_*` wrapper reduces to the
//!   `*_ref` function (e.g. ggml-quants.c:2128-2141), which is what the file
//!   format uses.
//! * `ggml_validate_row_data`         (ggml-quants.c:5380-5640)
//!
//! The Rust quantizers live in `ggml::quants` / `ggml::quants_k` and were
//! verified bit-exact against the reference (Q4_0..Q8_0, Q2_K..Q6_K).

use bytemuck::cast_slice_mut;
use ggml::blocks::*;
use ggml::quants::{
    dequantize_row, quantize_row_q1_0_ref, quantize_row_q2_0_ref, quantize_row_q4_0_ref,
    quantize_row_q4_1_ref, quantize_row_q5_0_ref, quantize_row_q5_1_ref, quantize_row_q8_0_ref,
};
use ggml::quants_k::{
    quantize_row_q2_K_ref, quantize_row_q3_K_ref, quantize_row_q4_K_ref, quantize_row_q5_K_ref,
    quantize_row_q6_K_ref,
};
use ggml::types::GgmlType;
use half::{bf16, f16};
use rayon::prelude::*;

/// Types this port can write (`quantize_*` dispatch, ggml.c:8093-8125).
///
/// Missing vs. the reference:
/// * MXFP4 / NVFP4 (ggml.c:8103-8104) and the whole IQ*/TQ* family — no
///   quantizers in `ggml::quants` at all (dequantizers only).
///
/// Q1_0 / Q2_0 used to be gated off here: their `*_ref` ports were built from a
/// different revision and disagreed with the pinned bd4f514db1 (Q1_0 wanted
/// `d = sum(|x|)/QK1_0`, Q2_0 wanted `d = max|x|` with `roundf`). Both are fixed
/// and bit-exact now (ggml::quants::tests::quants_q1_0_q2_0_bit_exact_vs_reference
/// against parity/quants_ref_q1q2.bin), so they are enabled.
pub fn has_quantizer(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(
        ty,
        F32 | F16 | Bf16 | Q1_0 | Q2_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K
            | Iq4Nl
            | Iq4Xs
    )
}

/// Does this port have the *imatrix-weighted* variant of `ty`?
/// (`quantize_<type>(..., imatrix != NULL)` — ggml.c:8093-8125 dispatch.)
///
/// Types without one are still quantized byte-identically with an imatrix
/// present: the C's wrappers for Q8_0/Q1_0/Q2_0 and the F16/BF16/F32 casts
/// ignore `quant_weights` entirely (ggml-quants.c:2298-2315, 2121-2139).
pub fn has_weighted_quantizer(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(ty, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q2K | Q3K | Q4K | Q5K | Q6K | Iq4Nl | Iq4Xs)
}

/// One `ggml_quantize_chunk` call (ggml.c:8057-8139):
/// `quantize_<type>(src, dst, nrows, n_per_row, imatrix)`.
fn quantize_chunk(
    ty: GgmlType,
    src: &[f32],
    nrows: i64,
    n_per_row: i64,
    dst: &mut [u8],
    imatrix: Option<&[f32]>,
) -> Result<usize, String> {
    if !has_quantizer(ty) {
        return Err(format!(
            "quantizer for type {} is not implemented in this port",
            ty.name()
        ));
    }
    let npr = n_per_row as usize;
    let row_size = ty.row_size(npr);
    // The C's `quantize_<type>` wrappers call the `*_impl` row function once
    // per row when `quant_weights != NULL` (ggml-quants.c:1213-1227). Rows are
    // independent and the weights are constant across the chunk (chunks never
    // cross an expert boundary, llama-quant.cpp:746), so the port runs that row
    // loop on rayon — the same split the C's per-chunk threads use.
    if let (Some(qw), true) = (imatrix, has_weighted_quantizer(ty)) {
        let work = &mut dst[..nrows as usize * row_size];
        work.par_chunks_mut(row_size)
            .zip(src.par_chunks(npr))
            .enumerate()
            .for_each(|(row, (y, x))| {
                let _ = row;
                ggml::quants_impl::quantize_row_weighted(ty, x, n_per_row, y, Some(qw))
                    .expect("weighted quantizer failed");
            });
        // llama-quant.cpp:761-764 `ggml_validate_row_data`
        if !validate_row_data(ty, work) {
            return Err("quantized data validation failed".to_string());
        }
        return Ok(nrows as usize * row_size);
    }
    quantize_rows(ty, src, nrows, n_per_row, dst)
}

/// `llama_tensor_quantize_impl` (llama-quant.cpp:747-816), single-thread arm:
/// quantize `nrows` rows starting at `first_row` (the tensor's global row
/// index), giving every expert slice its own imatrix row block
/// (`imatrix_for_row`, llama-quant.cpp:750-752).
///
/// `imatrix` is the *normalized* importance matrix of this tensor
/// (`ne[0] * ne[2]` floats, common/imatrix-loader + quantize.cpp:196-215);
/// `nrows_per_expert` is `tensor->ne[1]` (llama-quant.cpp:1300).
pub fn quantize_rows_imatrix(
    ty: GgmlType,
    src: &[f32],
    nrows: i64,
    n_per_row: i64,
    dst: &mut [u8],
    first_row: i64,
    nrows_per_expert: i64,
    imatrix: Option<&[f32]>,
) -> Result<usize, String> {
    let mut new_size = 0usize;
    let mut row = 0i64;
    while row < nrows {
        let row_global = first_row + row;
        // stop at the expert boundary (llama-quant.cpp:759)
        let this_nrow = std::cmp::min(nrows - row, nrows_per_expert - row_global % nrows_per_expert);
        let qw = imatrix.map(|im| {
            let off = ((row_global / nrows_per_expert) * n_per_row) as usize;
            &im[off..]
        });
        let this_size = quantize_chunk(
            ty,
            &src[row as usize * n_per_row as usize..],
            this_nrow,
            n_per_row,
            &mut dst[new_size..],
            qw,
        )?;
        new_size += this_size;
        row += this_nrow;
    }
    Ok(new_size)
}

/// `llama_tensor_dequantize_impl` (llama-quant.cpp:216-281).
///
/// The reference splits the work into `nthread` equal block ranges; blocks are
/// independent so any split is bit-identical. This port uses rayon with whole
/// blocks per job.
///
/// `src` is the raw tensor payload, `dst` the f32 row buffer.
pub fn dequantize_into(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), String> {
    if !ty.is_quantized() && !matches!(ty, GgmlType::F16 | GgmlType::Bf16) {
        // llama-quant.cpp:225-228 (F32 is never routed here by the caller)
        return Err(format!(
            "cannot dequantize/convert tensor type {}",
            ty.name()
        ));
    }
    if ty.is_quantized() && !llama::quant::has_dequantize(ty) {
        return Err(format!(
            "type {} unsupported for integer quantization: no dequantization available",
            ty.name()
        ));
    }
    let blck = ty.blck_size();
    let type_size = ty.type_size();
    assert_eq!(dst.len() % blck, 0, "dequantize: element count not a multiple of block");
    assert_eq!(src.len(), dst.len() / blck * type_size, "dequantize: byte length mismatch");

    if dst.len() <= blck * 256 {
        dequantize_row(ty, src, dst);
        return Ok(());
    }
    // whole blocks per job, one job per thread (plus the remainder in the tail)
    let nblocks = dst.len() / blck;
    let per_job = nblocks.div_ceil(rayon::current_num_threads()).max(1);
    dst.par_chunks_mut(per_job * blck)
        .zip(src.par_chunks(per_job * type_size))
        .for_each(|(dy, sx)| dequantize_row(ty, sx, dy));
    Ok(())
}

/// `llama_tensor_quantize_impl` (llama-quant.cpp:747-816) / the
/// `ggml_quantize_chunk` row loop: quantize `nrows` rows of `n_per_row`
/// elements from `src` into `dst`.
///
/// Returns the number of bytes written (`nrows * ggml_row_size(ty, n_per_row)`),
/// or an error exactly where C throws (`ggml_validate_row_data` failure,
/// llama-quant.cpp:762-764, 799-803).
pub fn quantize_rows(
    ty: GgmlType,
    src: &[f32],
    nrows: i64,
    n_per_row: i64,
    dst: &mut [u8],
) -> Result<usize, String> {
    if !has_quantizer(ty) {
        return Err(format!(
            "quantizer for type {} is not implemented in this port",
            ty.name()
        ));
    }
    let npr = n_per_row as usize;
    let blck = ty.blck_size();
    assert_eq!(npr % blck, 0, "row not divisible by block size");
    assert!(src.len() >= nrows as usize * npr);
    let row_size = ty.row_size(npr);
    assert!(dst.len() >= nrows as usize * row_size);

    let rows = nrows as usize;
    let bpr = npr / blck;

    macro_rules! per_row {
        ($bt:ty, $f:path) => {{
            let out = cast_slice_mut::<u8, $bt>(&mut dst[..rows * row_size]);
            // rows are independent: the reference hands disjoint row ranges to
            // its threads (llama-quant.cpp:775-808), rayon does the same
            out.par_chunks_mut(bpr)
                .zip(src.par_chunks(npr))
                .for_each(|(y, x)| $f(x, y));
        }};
    }

    match ty {
        GgmlType::Q1_0 => per_row!(BlockQ1_0, quantize_row_q1_0_ref),
        GgmlType::Q2_0 => per_row!(BlockQ2_0, quantize_row_q2_0_ref),
        GgmlType::Q4_0 => per_row!(BlockQ4_0, quantize_row_q4_0_ref),
        GgmlType::Q4_1 => per_row!(BlockQ4_1, quantize_row_q4_1_ref),
        GgmlType::Q5_0 => per_row!(BlockQ5_0, quantize_row_q5_0_ref),
        GgmlType::Q5_1 => per_row!(BlockQ5_1, quantize_row_q5_1_ref),
        GgmlType::Q8_0 => per_row!(BlockQ8_0, quantize_row_q8_0_ref),
        GgmlType::Q2K => per_row!(BlockQ2K, quantize_row_q2_K_ref),
        GgmlType::Q3K => per_row!(BlockQ3K, quantize_row_q3_K_ref),
        GgmlType::Q4K => per_row!(BlockQ4K, quantize_row_q4_K_ref),
        GgmlType::Q5K => per_row!(BlockQ5K, quantize_row_q5_K_ref),
        GgmlType::Q6K => per_row!(BlockQ6K, quantize_row_q6_K_ref),
        GgmlType::F16 => {
            // ggml.c:8122-8127 `ggml_fp32_to_fp16_row`
            let out = cast_slice_mut::<u8, f16>(&mut dst[..rows * row_size]);
            out.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = f16::from_f32(*s));
        }
        GgmlType::Bf16 => {
            // ggml.c:8128-8133 `ggml_fp32_to_bf16_row_ref`
            let out = cast_slice_mut::<u8, bf16>(&mut dst[..rows * row_size]);
            out.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = bf16::from_f32(*s));
        }
        GgmlType::F32 => {
            // ggml.c:8134-8139 memcpy
            let out = cast_slice_mut::<u8, f32>(&mut dst[..rows * row_size]);
            out.copy_from_slice(&src[..rows * npr]);
        }
        GgmlType::Iq4Nl | GgmlType::Iq4Xs => {
            // `ggml_quantize_chunk` always routes these through the weighted
            // entry points, with the weights possibly NULL
            // (ggml-quants.c:5048-5104)
            let out = &mut dst[..rows * row_size];
            out.par_chunks_mut(row_size)
                .zip(src.par_chunks(npr))
                .for_each(|(y, x)| {
                    ggml::quants_impl::quantize_row_weighted(ty, x, n_per_row, y, None)
                        .expect("iq4 quantizer failed");
                });
        }
        other => unreachable!("has_quantizer({other:?}) returned true"),
    }

    // llama-quant.cpp:762-764 / 799-803, ggml_validate_row_data
    if !validate_row_data(ty, &dst[..rows * row_size]) {
        return Err("quantized data validation failed".to_string());
    }
    Ok(rows * row_size)
}

// ---------------------------------------------------------------------------
// ggml_validate_row_data (ggml-quants.c:5380-5640)
// ---------------------------------------------------------------------------

fn isinf_fp16(f: f16) -> bool {
    let bits = f.to_bits();
    bits & 0x7c00 == 0x7c00 && bits & 0x03ff == 0
}

fn isnan_fp16(f: f16) -> bool {
    let bits = f.to_bits();
    bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0
}

fn validate_fp16(f: f16, i: usize) -> bool {
    if isinf_fp16(f) {
        eprintln!("ggml_validate_row_data: found inf value at block {i}");
        return false;
    }
    if isnan_fp16(f) {
        eprintln!("ggml_validate_row_data: found nan value at block {i}");
        return false;
    }
    true
}

fn validate_float(f: f32, i: usize) -> bool {
    if f.is_infinite() {
        eprintln!("ggml_validate_row_data: found inf value at block {i}");
        return false;
    }
    if f.is_nan() {
        eprintln!("ggml_validate_row_data: found nan value at block {i}");
        return false;
    }
    true
}

/// `ggml_validate_row_data` (ggml-quants.c:5380-5640): only the diagnostic
/// checks that matter for the types this port can produce. Types the reference
/// validates but this port never writes fall through unchanged.
pub fn validate_row_data(ty: GgmlType, data: &[u8]) -> bool {
    if data.len() % ty.type_size() != 0 {
        eprintln!(
            "ggml_validate_row_data: invalid size {} for type {} (type size = {})",
            data.len(),
            ty.name(),
            ty.type_size()
        );
        return false;
    }
    let nb = data.len() / ty.type_size();

    macro_rules! d_check {
        ($bt:ty, $get:expr) => {{
            let q: &[$bt] = bytemuck::cast_slice(&data[..nb * ty.type_size()]);
            for i in 0..nb {
                if !$get(q[i], i) {
                    return false;
                }
            }
        }};
    }

    match ty {
        GgmlType::F32 => {
            let f: &[f32] = bytemuck::cast_slice(data);
            for (i, v) in f.iter().enumerate() {
                if !validate_float(*v, i) {
                    return false;
                }
            }
        }
        GgmlType::F16 => {
            let f: &[f16] = bytemuck::cast_slice(data);
            for (i, v) in f.iter().enumerate() {
                if !validate_fp16(*v, i) {
                    return false;
                }
            }
        }
        GgmlType::Bf16 => {
            // ggml-quants.c:5394-5411
            let f: &[bf16] = bytemuck::cast_slice(data);
            let mut nans = 0;
            let mut infs = 0;
            for v in f {
                let bits = v.to_bits() & 0x7fff;
                nans += (bits > 0x7f80) as usize;
                infs += (bits == 0x7f80) as usize;
            }
            if nans > 0 {
                eprintln!("ggml_validate_row_data: found {nans} NaNs in row of {nb} BF16 values");
                return false;
            }
            if infs > 0 {
                eprintln!(
                    "ggml_validate_row_data: found {infs} infinities in row of {nb} BF16 values"
                );
                return false;
            }
        }
        GgmlType::Q1_0 => d_check!(BlockQ1_0, |b: BlockQ1_0, i| validate_fp16(b.d, i)),
        GgmlType::Q2_0 => d_check!(BlockQ2_0, |b: BlockQ2_0, i| validate_fp16(b.d, i)),
        GgmlType::Q4_0 => d_check!(BlockQ4_0, |b: BlockQ4_0, i| validate_fp16(b.d, i)),
        GgmlType::Q4_1 => {
            d_check!(BlockQ4_1, |b: BlockQ4_1, i| validate_fp16(b.d, i)
                && validate_fp16(b.m, i))
        }
        GgmlType::Q5_0 => d_check!(BlockQ5_0, |b: BlockQ5_0, i| validate_fp16(b.d, i)),
        GgmlType::Q5_1 => {
            d_check!(BlockQ5_1, |b: BlockQ5_1, i| validate_fp16(b.d, i)
                && validate_fp16(b.m, i))
        }
        GgmlType::Q8_0 => d_check!(BlockQ8_0, |b: BlockQ8_0, i| validate_fp16(b.d, i)),
        GgmlType::Q2K => {
            d_check!(BlockQ2K, |b: BlockQ2K, i| validate_fp16(b.d, i)
                && validate_fp16(b.dmin, i))
        }
        GgmlType::Q3K => d_check!(BlockQ3K, |b: BlockQ3K, i| validate_fp16(b.d, i)),
        GgmlType::Q4K => {
            d_check!(BlockQ4K, |b: BlockQ4K, i| validate_fp16(b.d, i)
                && validate_fp16(b.dmin, i))
        }
        GgmlType::Q5K => {
            d_check!(BlockQ5K, |b: BlockQ5K, i| validate_fp16(b.d, i)
                && validate_fp16(b.dmin, i))
        }
        GgmlType::Q6K => d_check!(BlockQ6K, |b: BlockQ6K, i| validate_fp16(b.d, i)),
        GgmlType::Mxfp4 => {
            // VALIDATE_ROW_DATA_E_E8M0_IMPL
            let q: &[BlockMxfp4] = bytemuck::cast_slice(&data[..nb * ty.type_size()]);
            for (i, b) in q.iter().enumerate() {
                if b.e == 0xff {
                    eprintln!(
                        "ggml_validate_row_data: found invalid e value {} at block {i}",
                        b.e
                    );
                    return false;
                }
            }
        }
        GgmlType::Nvfp4 => {} // "UE4M3 scales are uint8_t — all byte values are valid"
        _ => {
            // IQ*/TQ*/Q8_K/...: the reference validates these too, but this
            // port cannot produce them (no quantizer), so they never appear.
        }
    }
    true
}