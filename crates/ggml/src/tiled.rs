//! Tiled K-quant matmul — 1:1 port of ggml/src/ggml-cpu/tiled/
//! {tiled.cpp, tiled.h, tiled-kernel.cpp, tiled-kernel.h} (upstream def4d406a,
//! sync batch D). This is NEW CPU surface that replaces the deleted
//! `iqp.{cpp,h}` "IQ panel gemm" lane: it takes over the whole
//! `GGML_OP_MUL_MAT` when src0 is a supported K-quant / IQ format and the
//! batch is large enough (`rows >= 8`, tiled.cpp:1202-1205), and per-expert
//! `GGML_OP_MUL_MAT_ID` the same way (tiled.cpp:1227-1245).
//!
//! ## Which kernel the reference build runs
//! The reference build is `-march=native` (`GGML_NATIVE=ON`) on a Zen 5 host
//! with AVX512F/VL/DQ/BW/VNNI and **no AVX512FP16**, so the C preprocessor
//! selects `tiled_run_micro_vnni_8x16` (tiled-kernel.cpp:94, selected by
//! `__AVX512VNNI__ && __AVX512VL__ && __AVX512DQ__`) and the VNNI
//! 16x16-int32-transpose `tiled_repack_src1` (tiled-kernel.cpp:577). Those two
//! are the only bodies transcribed numerically; the AVX2/AVX/scalar bodies in
//! the same file are alternative ISA compilations of the same templates that
//! this reference never dispatches — the same policy the port uses for the
//! arch/x86 vec_dot lane ports. The epilogue float ops were pinned against
//! the shipped libggml-cpu.so disassembly (tiled_run_micro_vnni_8x16):
//! `vcvtdq2ps; vmulps d; vfnmadd132ps dmin; vfmadd213ps (buf, d1)` — i.e.
//! `result = d0 * (f32)s1_acc`, `result = fnmadd(dmin, f2, result)`,
//! `buf = fma(result, d1, buf)` — the C source's separate
//! `_mm512_add_ps(load, _mm512_mul_ps(result, d1_vec))` was contracted by
//! GCC's default `-ffp-contract=fast`, so the port uses `f32::mul_add` there.
//!
//! ## Thread scheduling
//! The C's chunk work-stealing (`ggml_threadpool_chunk_set/add`,
//! tiled.cpp:1082-1143) only decides which thread computes which
//! (ir0 x ir1) chunk; every output element is computed inside exactly one
//! chunk by a fixed op sequence, so the port distributes the same chunk grid
//! through the port's Team and stays bit-identical for any thread count
//! (asserted by the tests). The MUL_MAT_ID expert windows are likewise
//! per-(k-window, row-group) independent: each zeroes its acc window before
//! accumulating (tiled.cpp:775-777).
//!
//! ## Ground truth
//! `parity/tiled_ref.bin` (`parity/ref_tiled_dump.c`, linked against the NEW
//! reference build): synthetic mul_mat / mul_mat_id sections per type covering
//! the standard (>16 src1 rows), narrow (<=16, k_extent) and ragged window
//! paths; the `tiled_ref_tests` module at the bottom replays them through the
//! port's own `graph_compute`.

use crate::blocks::{
    BlockIq1M, BlockIq1S, BlockIq2S, BlockIq2Xs, BlockIq2Xxs, BlockIq3S, BlockIq3Xxs, BlockIq4Xs,
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
};
use crate::quants_k::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
    KVALUES_IQ4NL,
};
use crate::types::GgmlType;

/// tiled-kernel.h:22-25
pub const TILED_TILE_K: usize = 256; // one QK_K block
pub const TILED_TILE_ROWS: usize = 256; // max window rows, ragged at edges
pub const TILED_MICRO: usize = 16; // microtile edge (also the bsums granularity)
/// Per-thread workspace slot (tiled-kernel.h:26); `sizeof(tiled_ws)` rounds up
/// to this. The C reserves `64 + n_tasks * TILED_WS_SLOT` bytes of graph work
/// buffer (tiled.cpp:654-659); the port has no shared work buffer and keeps
/// one workspace per OS thread instead (see `TiledWs`).
pub const TILED_WS_SLOT: usize = 512 * 1024;
/// tiled-kernel.h:33 `NB_MAX` — max subblocks per 256-elem block (SUBBLK=16).
const NB_MAX: usize = TILED_TILE_K / 16;
/// tiled.cpp:719 `TILED_MMID_GROUP` — src0 rows per MUL_MAT_ID group.
const TILED_MMID_GROUP: usize = 64;

/// `block_q8_K` byte size (static_assert'd 292 in tiled-kernel.cpp:569).
const Q8K_BLOCK: usize = core::mem::size_of::<BlockQ8K>();

// ======================================================================
// tiles + per-thread workspace (tiled-kernel.h:29-54)
// ======================================================================

/// `tiled_tile_src0` — weight side, shared by all formats.
struct TiledTileSrc0 {
    /// `uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]` — unsigned quants widened
    /// to u8 (+128-biased for the BIAS != 0 formats).
    q: Box<[u8; TILED_TILE_ROWS * TILED_TILE_K]>,
    d: Box<[f32; TILED_TILE_ROWS]>,
    dmin: Box<[f32; TILED_TILE_ROWS]>,
    scales: Box<[i32; TILED_TILE_ROWS * NB_MAX]>,
    mins: Box<[i32; TILED_TILE_ROWS * NB_MAX]>,
}

/// `tiled_tile_src1` — built from the q8_K work rows.
struct TiledTileSrc1 {
    /// `int8_t q[...]` — q8 codes, repacked in place on the VNNI path.
    q: Box<[u8; TILED_TILE_ROWS * TILED_TILE_K]>,
    /// Per-16 code sums from q8_K widened to i32.
    bsums: Box<[i32; (TILED_TILE_K / TILED_MICRO) * TILED_TILE_ROWS]>,
    d: Box<[f32; TILED_TILE_ROWS]>,
}

/// `tiled_ws` (tiled-kernel.h:48-52). All state is re-initialised per tile —
/// the ragged src1 tail is explicitly zero-padded by the unpacker
/// (tiled.cpp:543-569) — so scratch carries nothing between chunks and one
/// workspace per OS thread is a faithful stand-in for the C's per-thread
/// wdata slots.
struct TiledWs {
    src0: TiledTileSrc0,
    src1: TiledTileSrc1,
    acc: Box<[f32; TILED_TILE_ROWS * TILED_TILE_ROWS]>,
}

thread_local! {
    static WS: std::cell::RefCell<TiledWs> = std::cell::RefCell::new(TiledWs {
        src0: TiledTileSrc0 {
            q: Box::new([0; TILED_TILE_ROWS * TILED_TILE_K]),
            d: Box::new([0.0; TILED_TILE_ROWS]),
            dmin: Box::new([0.0; TILED_TILE_ROWS]),
            scales: Box::new([0; TILED_TILE_ROWS * NB_MAX]),
            mins: Box::new([0; TILED_TILE_ROWS * NB_MAX]),
        },
        src1: TiledTileSrc1 {
            q: Box::new([0; TILED_TILE_ROWS * TILED_TILE_K]),
            bsums: Box::new([0; (TILED_TILE_K / TILED_MICRO) * TILED_TILE_ROWS]),
            d: Box::new([0.0; TILED_TILE_ROWS]),
        },
        acc: Box::new([0.0; TILED_TILE_ROWS * TILED_TILE_ROWS]),
    });
}

// ======================================================================
// unpack primitives (tiled-kernel.h:60-155)
// ======================================================================
// The C defines these twice (`#if defined(__AVX2__)` intrinsics / scalar
// `#else`); both are the same pure byte permutation, so the scalar bodies are
// a faithful port of either.

/// `tiled_unpk_nib4` — packed 4-bit codes -> low nibbles (lo) + high (hi),
/// 32 bytes each. The AVX2 body masks before the lane shift
/// (`and(v, 0xF0)` then `srli 4`), which for byte lanes equals `src >> 4`.
#[inline]
fn unpk_nib4(src: &[u8], lo: &mut [u8], hi: &mut [u8]) {
    for l in 0..32 {
        lo[l] = src[l] & 0xF;
        hi[l] = src[l] >> 4;
    }
}

/// `tiled_unpk_2bit::<S>` — 2-bit values at bit offset S, 32 bytes.
#[inline]
fn unpk_2bit<const S: u32>(src: &[u8], dst: &mut [u8]) {
    for l in 0..32 {
        dst[l] = (src[l] >> S) & 3;
    }
}

/// `tiled_unpk_or::<S, D, M>` — OR the M-bit value at bit offset S of src into
/// bit offset D of dst, 32 lanes.
#[inline]
fn unpk_or<const S: u32, const D: u32, const M: u8>(dst: &mut [u8], src: &[u8]) {
    for l in 0..32 {
        dst[l] |= ((src[l] >> S) & M) << D;
    }
}

/// `tiled_lut8` — 16-entry byte LUT: `dst[j] = lut[src[j]]`.
#[inline]
fn lut8(lut: &[u8; 16], src: &[u8], dst: &mut [u8]) {
    for j in 0..16 {
        dst[j] = lut[src[j] as usize];
    }
}

/// `tiled_unpk_sign32` — 32 grid magnitudes (4 x 8 bytes, LE) + 4 sign bytes
/// -> biased codes `128 ± v` (u8 wrapping; mirrors the SIMD identity
/// `(v ^ mask) - mask + 128`).
fn unpk_sign32(g0: u64, g1: u64, g2: u64, g3: u64, signs: &[u8; 4], dst32: &mut [u8]) {
    let g = [g0.to_le_bytes(), g1.to_le_bytes(), g2.to_le_bytes(), g3.to_le_bytes()];
    for l in 0..4 {
        let s = signs[l];
        for j in 0..8 {
            dst32[8 * l + j] = if s & (1 << j) != 0 {
                (128i32 - g[l][j] as i32) as u8
            } else {
                (128i32 + g[l][j] as i32) as u8
            };
        }
    }
}

/// `tiled_unpk_tern8` — 8 ternary grid bytes (0 / 1 / 0xFF as -1):
/// `dst[j] = 128 + delta + 8 * (int8_t) src[j]` (u8 wrapping).
fn unpk_tern8(src: &[u8], delta: i8, dst: &mut [u8]) {
    for j in 0..8 {
        dst[j] = (128 + delta as i32 + 8 * (src[j] as i8 as i32)) as u8;
    }
}

// ======================================================================
// src0 formats (tiled.cpp:21-535 unpackers, :1170-1200 kernel constants)
// ======================================================================

/// The 13 supported src0 formats.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fmt {
    Q4K,
    Q5K,
    Q6K,
    Q3K,
    Q2K,
    Iq4Xs,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
}

impl Fmt {
    pub fn from_type(ty: GgmlType) -> Option<Fmt> {
        Some(match ty {
            GgmlType::Q4K => Fmt::Q4K,
            GgmlType::Q5K => Fmt::Q5K,
            GgmlType::Q6K => Fmt::Q6K,
            GgmlType::Q3K => Fmt::Q3K,
            GgmlType::Q2K => Fmt::Q2K,
            GgmlType::Iq4Xs => Fmt::Iq4Xs,
            GgmlType::Iq2Xxs => Fmt::Iq2Xxs,
            GgmlType::Iq2Xs => Fmt::Iq2Xs,
            GgmlType::Iq2S => Fmt::Iq2S,
            GgmlType::Iq3Xxs => Fmt::Iq3Xxs,
            GgmlType::Iq3S => Fmt::Iq3S,
            GgmlType::Iq1S => Fmt::Iq1S,
            GgmlType::Iq1M => Fmt::Iq1M,
            _ => return None,
        })
    }

    /// (SUBBLK, HAS_MIN, BIAS, ACTBIAS) — tiled.cpp:1171-1196, the explicit
    /// instantiation list at tiled-kernel.cpp:541-563.
    fn kernel_consts(self) -> (usize, bool, i32, bool) {
        match self {
            Fmt::Q6K => (16, false, 32, true),
            Fmt::Q5K => (32, true, 0, false),
            Fmt::Q4K => (32, true, 0, false),
            Fmt::Q3K => (16, false, 4, true),
            Fmt::Q2K => (16, true, 0, false),
            Fmt::Iq4Xs => (32, false, 128, false),
            Fmt::Iq2Xxs => (32, false, 128, true),
            Fmt::Iq2Xs => (16, false, 128, true),
            Fmt::Iq2S => (16, false, 128, true),
            Fmt::Iq3Xxs => (32, false, 128, true),
            Fmt::Iq3S => (32, false, 128, true),
            Fmt::Iq1S => (32, false, 128, true),
            Fmt::Iq1M => (16, false, 128, true),
        }
    }
}

/// Read block `i` of a block array at `rows` (unaligned pod read).
#[inline]
fn blk<T: bytemuck::Pod>(rows: *const u8, i: usize) -> T {
    let sz = core::mem::size_of::<T>();
    let bytes = unsafe { core::slice::from_raw_parts(rows.add(i * sz), sz) };
    bytemuck::pod_read_unaligned(bytes)
}

/// The q4_K/q5_K shared 12-byte scale/min decode (tiled.cpp:25-27, :69-71,
/// "same extraction as the reference kernels" — the ggml kmask trick).
fn unpack_k_scales_12(scales: &[u8; 12]) -> ([i32; 8], [i32; 8]) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut utmp = [0u32; 4];
    for (w, c) in utmp.iter_mut().zip(scales.chunks_exact(4)) {
        *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    // C: `scales = (const uint8_t *) &utmp[0]; mins = &utmp[2];` — 8 bytes
    // each, i.e. the byte spans of utmp[0..2] and utmp[2..4]
    let mut out_s = [0i32; 8];
    let mut out_m = [0i32; 8];
    for i in 0..4 {
        out_s[i] = utmp[0].to_le_bytes()[i] as i32;
        out_s[i + 4] = utmp[1].to_le_bytes()[i] as i32;
        out_m[i] = utmp[2].to_le_bytes()[i] as i32;
        out_m[i + 4] = utmp[3].to_le_bytes()[i] as i32;
    }
    (out_s, out_m)
}

/// Common dst offsets for the src0 unpackers.
#[allow(clippy::too_many_arguments)]
fn unpack_src0(
    fmt: Fmt,
    rows: *const u8,
    row_stride: usize,
    n_rows: usize,
    tile: &mut TiledTileSrc0,
    num_k: usize,
) {
    debug_assert!(n_rows <= TILED_TILE_ROWS);
    let qk_stride = num_k * TILED_TILE_K;
    for slab in 0..num_k {
        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let q_off = r * qk_stride + slab * TILED_TILE_K;
            let q = &mut tile.q[q_off..q_off + TILED_TILE_K];
            match fmt {
                Fmt::Q4K => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockQ4K = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    tile.dmin[d_off] = x.dmin.to_f32();
                    let (scales, mins) = unpack_k_scales_12(&x.scales);
                    tile.scales[s_off..s_off + NB].copy_from_slice(&scales);
                    tile.mins[s_off..s_off + NB].copy_from_slice(&mins);
                    { let (lo, hi) = q[0..64].split_at_mut(32); unpk_nib4(&x.qs[0..], lo, hi); }
                    { let (lo, hi) = q[64..128].split_at_mut(32); unpk_nib4(&x.qs[32..], lo, hi); }
                    { let (lo, hi) = q[128..192].split_at_mut(32); unpk_nib4(&x.qs[64..], lo, hi); }
                    { let (lo, hi) = q[192..256].split_at_mut(32); unpk_nib4(&x.qs[96..], lo, hi); }
                }
                Fmt::Q5K => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockQ5K = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    tile.dmin[d_off] = x.dmin.to_f32();
                    let (scales, mins) = unpack_k_scales_12(&x.scales);
                    tile.scales[s_off..s_off + NB].copy_from_slice(&scales);
                    tile.mins[s_off..s_off + NB].copy_from_slice(&mins);
                    { let (lo, hi) = q[0..64].split_at_mut(32); unpk_nib4(&x.qs[0..], lo, hi); }
                    { let (lo, hi) = q[64..128].split_at_mut(32); unpk_nib4(&x.qs[32..], lo, hi); }
                    { let (lo, hi) = q[128..192].split_at_mut(32); unpk_nib4(&x.qs[64..], lo, hi); }
                    { let (lo, hi) = q[192..256].split_at_mut(32); unpk_nib4(&x.qs[96..], lo, hi); }
                    unpk_or::<0, 4, 1>(&mut q[0..], &x.qh);
                    unpk_or::<1, 4, 1>(&mut q[32..], &x.qh);
                    unpk_or::<2, 4, 1>(&mut q[64..], &x.qh);
                    unpk_or::<3, 4, 1>(&mut q[96..], &x.qh);
                    unpk_or::<4, 4, 1>(&mut q[128..], &x.qh);
                    unpk_or::<5, 4, 1>(&mut q[160..], &x.qh);
                    unpk_or::<6, 4, 1>(&mut q[192..], &x.qh);
                    unpk_or::<7, 4, 1>(&mut q[224..], &x.qh);
                }
                Fmt::Q6K => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockQ6K = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    for half in 0..2 {
                        let out = &mut q[128 * half..];
                        { let (l, h) = out.split_at_mut(64); unpk_nib4(&x.ql[64 * half..], &mut l[0..32], &mut h[0..32]); }
                        { let (l, h) = out.split_at_mut(64); unpk_nib4(&x.ql[64 * half + 32..], &mut l[32..64], &mut h[32..64]); }
                        let qh = &x.qh[32 * half..];
                        unpk_or::<0, 4, 3>(&mut out[0..], qh);
                        unpk_or::<2, 4, 3>(&mut out[32..], qh);
                        unpk_or::<4, 4, 3>(&mut out[64..], qh);
                        unpk_or::<6, 4, 3>(&mut out[96..], qh);
                    }
                    for (s, &sc) in tile.scales[s_off..s_off + NB].iter_mut().zip(x.scales.iter()) {
                        *s = sc as i32;
                    }
                }
                Fmt::Q3K => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockQ3K = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    let s0 = &x.qs[0..];
                    let s1 = &x.qs[32..];
                    let (o0, o1) = q.split_at_mut(128);
                    unpk_2bit::<0>(s0, &mut o0[0..32]);
                    unpk_or::<0, 2, 1>(&mut o0[0..], &x.hmask);
                    unpk_2bit::<2>(s0, &mut o0[32..64]);
                    unpk_or::<1, 2, 1>(&mut o0[32..], &x.hmask);
                    unpk_2bit::<4>(s0, &mut o0[64..96]);
                    unpk_or::<2, 2, 1>(&mut o0[64..], &x.hmask);
                    unpk_2bit::<6>(s0, &mut o0[96..128]);
                    unpk_or::<3, 2, 1>(&mut o0[96..], &x.hmask);
                    unpk_2bit::<0>(s1, &mut o1[0..32]);
                    unpk_or::<4, 2, 1>(&mut o1[0..], &x.hmask);
                    unpk_2bit::<2>(s1, &mut o1[32..64]);
                    unpk_or::<5, 2, 1>(&mut o1[32..], &x.hmask);
                    unpk_2bit::<4>(s1, &mut o1[64..96]);
                    unpk_or::<6, 2, 1>(&mut o1[64..], &x.hmask);
                    unpk_2bit::<6>(s1, &mut o1[96..128]);
                    unpk_or::<7, 2, 1>(&mut o1[96..], &x.hmask);
                    const KMASK1: u32 = 0x0303_0303;
                    const KMASK2: u32 = 0x0f0f_0f0f;
                    let mut auxs = [0u32; 4];
                    for (w, c) in auxs.iter_mut().zip(x.scales.chunks_exact(4)) {
                        *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                    let tmp = auxs[2];
                    auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                    auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                    auxs[0] = (auxs[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
                    auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                    let bytes = [
                        auxs[0].to_le_bytes(),
                        auxs[1].to_le_bytes(),
                        auxs[2].to_le_bytes(),
                        auxs[3].to_le_bytes(),
                    ];
                    for (s, b) in tile.scales[s_off..s_off + NB].iter_mut().zip(bytes.iter().flatten()) {
                        *s = *b as i8 as i32 - 32;
                    }
                }
                Fmt::Q2K => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockQ2K = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    tile.dmin[d_off] = x.dmin.to_f32();
                    for half in 0..2 {
                        let s = &x.qs[32 * half..];
                        let out = &mut q[128 * half..];
                        unpk_2bit::<0>(s, &mut out[0..32]);
                        unpk_2bit::<2>(s, &mut out[32..64]);
                        unpk_2bit::<4>(s, &mut out[64..96]);
                        unpk_2bit::<6>(s, &mut out[96..128]);
                    }
                    for s in 0..NB {
                        tile.scales[s_off + s] = (x.scales[s] & 0xF) as i32;
                        tile.mins[s_off + s] = (x.scales[s] >> 4) as i32;
                    }
                }
                Fmt::Iq4Xs => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq4Xs = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    // kvalues + 128 up front so the expansion is a plain LUT
                    // (tiled.cpp:242-251)
                    let lut: [u8; 16] = core::array::from_fn(|i| (KVALUES_IQ4NL[i] as i32 + 128) as u8);
                    for s in 0..NB {
                        let ls = ((x.scales_l[s / 2] >> (4 * (s % 2))) & 0xf) as i32
                            | (((x.scales_h >> (2 * s)) & 3) as i32) << 4;
                        tile.scales[s_off + s] = ls - 32;
                    }
                    let mut lo = [0u8; 32];
                    let mut hi = [0u8; 32];
                    for u in 0..4 {
                        unpk_nib4(&x.qs[32 * u..], &mut lo, &mut hi);
                        lut8(&lut, &lo[0..], &mut q[64 * u..]);
                        lut8(&lut, &hi[0..], &mut q[64 * u + 16..]);
                        lut8(&lut, &lo[16..], &mut q[64 * u + 32..]);
                        lut8(&lut, &hi[16..], &mut q[64 * u + 48..]);
                    }
                }
                Fmt::Iq2Xxs => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq2Xxs = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32() * 0.125;
                    let mut g = [0u64; 4];
                    let mut signs4 = [0u8; 4];
                    for ib32 in 0..NB {
                        // C: memcpy(aux32, x.qs + 4*ib32, 8) — the 4 u16s of
                        // this subblock as 8 LE bytes; grid indices are the
                        // bytes, signs come from the high u32
                        let mut aux = [0u8; 8];
                        for (k, w) in x.qs[4 * ib32..4 * ib32 + 4].iter().enumerate() {
                            aux[2 * k..2 * k + 2].copy_from_slice(&w.to_le_bytes());
                        }
                        let aux32_1 = u32::from_le_bytes(aux[4..8].try_into().unwrap());
                        tile.scales[s_off + ib32] = (2 * (aux32_1 >> 28) + 1) as i32;
                        for l in 0..4 {
                            g[l] = IQ2XXS_GRID[aux[l] as usize];
                            signs4[l] = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                        }
                        unpk_sign32(g[0], g[1], g[2], g[3], &signs4, &mut q[32 * ib32..]);
                    }
                }
                Fmt::Iq2Xs => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq2Xs = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32() * 0.125;
                    let mut g = [0u64; 4];
                    let mut signs4 = [0u8; 4];
                    for ib32 in 0..QK_K / 32 {
                        tile.scales[s_off + 2 * ib32] = (2 * (x.scales[ib32] & 0xf) as i32 + 1) as i32;
                        tile.scales[s_off + 2 * ib32 + 1] = (2 * (x.scales[ib32] >> 4) as i32 + 1) as i32;
                        let qq = &x.qs[4 * ib32..];
                        for l in 0..4 {
                            g[l] = IQ2XS_GRID[(qq[l] & 511) as usize];
                            signs4[l] = KSIGNS_IQ2XS[(qq[l] >> 9) as usize];
                        }
                        unpk_sign32(g[0], g[1], g[2], g[3], &signs4, &mut q[32 * ib32..]);
                    }
                }
                Fmt::Iq2S => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq2S = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32() * 0.125;
                    // packed sign bytes share the qs array (tiled.cpp:365)
                    let qs = &x.qs[0..];
                    let signs = &x.qs[QK_K / 8..];
                    let mut g = [0u64; 4];
                    let mut off = 0usize;
                    for ib32 in 0..QK_K / 32 {
                        tile.scales[s_off + 2 * ib32] = (2 * (x.scales[ib32] & 0xf) as i32 + 1) as i32;
                        tile.scales[s_off + 2 * ib32 + 1] = (2 * (x.scales[ib32] >> 4) as i32 + 1) as i32;
                        for l in 0..4 {
                            g[l] = IQ2S_GRID[(qs[off + l] as u32
                                | (((x.qh[ib32] as u32) << (8 - 2 * l)) & 0x300))
                                as usize];
                        }
                        unpk_sign32(
                            g[0],
                            g[1],
                            g[2],
                            g[3],
                            signs[off..off + 4].try_into().unwrap(),
                            &mut q[32 * ib32..],
                        );
                        off += 4;
                    }
                }
                Fmt::Iq3Xxs => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq3Xxs = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32() * 0.25;
                    let qs = &x.qs[0..];
                    let scales_and_signs = &x.qs[QK_K / 4..];
                    let mut g = [0u64; 4];
                    let mut signs4 = [0u8; 4];
                    let mut off = 0usize;
                    for ib32 in 0..QK_K / 32 {
                        let aux32 = u32::from_le_bytes(
                            scales_and_signs[4 * ib32..4 * ib32 + 4].try_into().unwrap(),
                        );
                        tile.scales[s_off + ib32] = (2 * (aux32 >> 28) + 1) as i32;
                        for l in 0..4 {
                            g[l] = ((IQ3XXS_GRID[qs[off + 2 * l + 1] as usize] as u64) << 32)
                                | IQ3XXS_GRID[qs[off + 2 * l] as usize] as u64;
                            signs4[l] = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        }
                        unpk_sign32(g[0], g[1], g[2], g[3], &signs4, &mut q[32 * ib32..]);
                        off += 8;
                    }
                }
                Fmt::Iq3S => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq3S = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32();
                    let mut g = [0u64; 4];
                    let mut qs_off = 0usize;
                    let mut qh_off = 0usize;
                    let mut signs_off = 0usize;
                    let mut ib32 = 0usize;
                    while ib32 < QK_K / 32 {
                        tile.scales[s_off + ib32] = (1 + 2 * (x.scales[ib32 / 2] & 0xf) as i32) as i32;
                        tile.scales[s_off + ib32 + 1] = (1 + 2 * (x.scales[ib32 / 2] >> 4) as i32) as i32;
                        for h in 0..2 {
                            for l in 0..4 {
                                g[l] = ((IQ3S_GRID[(x.qs[qs_off + 2 * l + 1] as u32
                                    | (((x.qh[qh_off + h] as u32) << (7 - 2 * l)) & 256))
                                    as usize]
                                    as u64)
                                    << 32)
                                    | IQ3S_GRID[(x.qs[qs_off + 2 * l] as u32
                                        | (((x.qh[qh_off + h] as u32) << (8 - 2 * l)) & 256))
                                        as usize]
                                        as u64;
                            }
                            unpk_sign32(
                                g[0],
                                g[1],
                                g[2],
                                g[3],
                                x.signs[signs_off..signs_off + 4].try_into().unwrap(),
                                &mut q[(ib32 + h) * 32..],
                            );
                            qs_off += 8;
                            signs_off += 4;
                        }
                        qh_off += 2;
                        ib32 += 2;
                    }
                }
                Fmt::Iq1S => {
                    const NB: usize = 8;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq1S = blk(rows, r * row_stride + slab);
                    tile.d[d_off] = x.d.to_f32() * 0.125;
                    let mut off = 0usize;
                    for ib in 0..QK_K / 32 {
                        let hw = x.qh[ib];
                        tile.scales[s_off + ib] = (2 * ((hw >> 12) & 7) as i32 + 1) as i32;
                        let delta: i8 = if hw & 0x8000 != 0 { -1 } else { 1 };
                        for l in 0..4 {
                            let entry = IQ1S_GRID
                                [(x.qs[off + l] as u32 | (((hw >> (3 * l)) & 7) as u32) << 8) as usize];
                            unpk_tern8(&entry.to_le_bytes(), delta, &mut q[32 * ib + 8 * l..]);
                        }
                        off += 4;
                    }
                }
                Fmt::Iq1M => {
                    const NB: usize = 16;
                    let s_off = r * (NB * num_k) + slab * NB;
                    let x: BlockIq1M = blk(rows, r * row_stride + slab);
                    // iq1m_scale_t: fp16 scale packed across the 4 scale u16
                    let sc = [
                        u16::from_le_bytes([x.scales[0], x.scales[1]]),
                        u16::from_le_bytes([x.scales[2], x.scales[3]]),
                        u16::from_le_bytes([x.scales[4], x.scales[5]]),
                        u16::from_le_bytes([x.scales[6], x.scales[7]]),
                    ];
                    let f16bits = (sc[0] >> 12)
                        | ((sc[1] >> 8) & 0x00f0)
                        | ((sc[2] >> 4) & 0x0f00)
                        | (sc[3] & 0xf000);
                    tile.d[d_off] = half::f16::from_bits(f16bits).to_f32() * 0.125;
                    let mut qs_off = 0usize;
                    let mut qh_off = 0usize;
                    for ib in 0..QK_K / 32 {
                        let hw = sc[ib / 2];
                        let sh = 6 * (ib % 2);
                        tile.scales[s_off + 2 * ib] = (2 * ((hw >> sh) & 7) as i32 + 1) as i32;
                        tile.scales[s_off + 2 * ib + 1] =
                            (2 * ((hw >> (sh + 3)) & 7) as i32 + 1) as i32;
                        let qs = [
                            x.qs[qs_off],
                            x.qs[qs_off + 1],
                            x.qs[qs_off + 2],
                            x.qs[qs_off + 3],
                        ];
                        let idx = [
                            ((qs[0] as u16) | (((x.qh[qh_off] as u16) << 8) & 0x0700)) as usize,
                            ((qs[1] as u16) | (((x.qh[qh_off] as u16) << 4) & 0x0700)) as usize,
                            ((qs[2] as u16) | (((x.qh[qh_off + 1] as u16) << 8) & 0x0700)) as usize,
                            ((qs[3] as u16) | (((x.qh[qh_off + 1] as u16) << 4) & 0x0700)) as usize,
                        ];
                        let delta = [
                            if x.qh[qh_off] & 0x08 != 0 { -1i8 } else { 1i8 },
                            if x.qh[qh_off] & 0x80 != 0 { -1i8 } else { 1i8 },
                            if x.qh[qh_off + 1] & 0x08 != 0 { -1i8 } else { 1i8 },
                            if x.qh[qh_off + 1] & 0x80 != 0 { -1i8 } else { 1i8 },
                        ];
                        for l in 0..4 {
                            let entry = IQ1S_GRID[idx[l]];
                            unpk_tern8(&entry.to_le_bytes(), delta[l], &mut q[32 * ib + 8 * l..]);
                        }
                        qs_off += 4;
                        qh_off += 2;
                    }
                }
            }
        }
    }
}

/// `tiled_unpack_src1_q8_K` (tiled.cpp:540-570). `rows[r]` points at
/// activation row r's first q8_K block; block `kblk + slab` is read at
/// `rows[r] + (kblk + slab) * 292`.
fn unpack_src1_q8k(
    rows: &[*const u8],
    n_rows: usize,
    tile: &mut TiledTileSrc1,
    kblk: usize,
    num_k: usize,
) {
    debug_assert!(n_rows <= TILED_TILE_ROWS);
    let n_padded = (n_rows + TILED_MICRO - 1) & !(TILED_MICRO - 1);
    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };
    let n16 = TILED_TILE_K / TILED_MICRO;
    for slab in 0..num_k {
        let q_off = slab * TILED_TILE_K;
        for r in 0..n_padded {
            let dst = &mut tile.q[r * (num_k * TILED_TILE_K) + q_off..][..TILED_TILE_K];
            if r < n_rows {
                let b: BlockQ8K = blk(rows[r], kblk + slab);
                dst.copy_from_slice(bytemuck::bytes_of(&b.qs));
            } else {
                dst.fill(0);
            }
        }
        for r in 0..n_padded {
            if r < n_rows {
                let b: BlockQ8K = blk(rows[r], kblk + slab);
                for s in 0..n16 {
                    tile.bsums[(slab * n16 + s) * bs_stride + r] = b.bsums[s] as i32;
                }
                tile.d[slab * TILED_MICRO + r] = b.d;
            } else {
                for s in 0..n16 {
                    tile.bsums[(slab * n16 + s) * bs_stride + r] = 0;
                }
                tile.d[slab * TILED_MICRO + r] = 0.0;
            }
        }
    }
}

// ======================================================================
// VNNI microkernel (tiled-kernel.cpp:94-205)
// ======================================================================

/// `_mm512_dpbusd_epi32(acc, a, b)` per 32-bit lane: acc[l] += sum of
/// (unsigned byte i of `a`) * (signed byte 4l+i of `b`). `b` is the 64 bytes
/// of repacked activation codes for lanes 0..16.
#[inline]
fn dpbusd_64(acc: &mut [i32; 16], a: u32, b: &[u8]) {
    debug_assert_eq!(b.len(), 64);
    for (l, bb) in b.chunks_exact(4).enumerate() {
        let mut s = 0i32;
        for (i, &byte) in bb.iter().enumerate() {
            s = s.wrapping_add(((a >> (8 * i)) & 0xff) as i32 * (byte as i8 as i32));
        }
        acc[l] = acc[l].wrapping_add(s);
    }
}

/// `tiled_run_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS, NK>` (tiled-kernel.cpp:
/// 94-197) — one 8x16 band (8 src0 rows x 16 src1 cols), exact int32 math;
/// the float epilogue per the shipped disassembly (module header).
#[allow(clippy::too_many_arguments)]
fn micro_vnni_8x16<const SUBBLK: usize, const HAS_MIN: bool, const BIAS: i32, const NK: i32>(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    i0: usize,
    j0: usize,
    num_k: usize,
    slab: usize,
    buf: &mut [f32],
    buf_stride: usize,
) {
    const NUM_ROWS: usize = 8; // band width (register-pressure note, C:102)
    let nb = TILED_TILE_K / SUBBLK;
    let ns = SUBBLK / 16;
    let ng = SUBBLK / 4;

    // num_k/slab stride selection (C:104-116)
    let nkr = if NK > 0 { NK as usize } else { num_k };
    let seff = if NK > 0 { 0 } else { slab };
    let qk_stride = nkr * TILED_TILE_K;
    let qk_off = seff * TILED_TILE_K;
    let nb_stride = nb * nkr;
    let nb_off = seff * nb;
    let bs_stride = if nkr == 1 { TILED_TILE_ROWS } else { TILED_MICRO };
    let bs_off = seff * TILED_MICRO;

    let d1_vec: [f32; 16] = src1.d[seff * TILED_MICRO + j0..][..16].try_into().unwrap();

    let mut s1_acc = [[0i32; 16]; NUM_ROWS];
    let mut s2_acc = [[0i32; 16]; NUM_ROWS];

    // band pass: dpbusd over every 4-byte group of every subblock
    for s in 0..nb {
        let mut bsums32 = [0i32; 16];
        for l in 0..16 {
            bsums32[l] = src1.bsums[(bs_off + s * ns) * bs_stride + j0 + l];
        }
        for u in 1..ns {
            for l in 0..16 {
                bsums32[l] =
                    bsums32[l].wrapping_add(src1.bsums[(bs_off + s * ns + u) * bs_stride + j0 + l]);
            }
        }
        let bias32: [i32; 16] = if BIAS != 0 {
            core::array::from_fn(|l| bsums32[l].wrapping_mul(BIAS))
        } else {
            [0i32; 16]
        };

        let mut acc16 = [[0i32; 16]; NUM_ROWS];
        for g in 0..ng {
            let kg = s * ng + g;
            // in-place interleave layout: [kg%16 @ TILED_TILE_K][kg/16 @ 64][row @ 4]
            let codes_base = (j0 / TILED_MICRO) * (TILED_MICRO * qk_stride)
                + qk_off
                + (kg % TILED_MICRO) * qk_stride
                + (kg / TILED_MICRO) * (TILED_MICRO * 4);
            for t in 0..NUM_ROWS {
                let u4 = u32::from_le_bytes(
                    src0.q[(i0 + t) * qk_stride + qk_off + kg * 4..][..4].try_into().unwrap(),
                );
                dpbusd_64(&mut acc16[t], u4, &src1.q[codes_base..][..64]);
            }
        }

        // s1_acc += scales * (raw - BIAS*bsums)
        for t in 0..NUM_ROWS {
            let scale = src0.scales[(i0 + t) * nb_stride + nb_off + s];
            for l in 0..16 {
                let mut rawi = acc16[t][l];
                if BIAS != 0 {
                    rawi = rawi.wrapping_sub(bias32[l]);
                }
                s1_acc[t][l] = s1_acc[t][l].wrapping_add(rawi.wrapping_mul(scale));
            }
        }
    }

    // s2_acc += mins * bsums (C:171-182)
    if HAS_MIN {
        for s in 0..nb {
            let mut bsums32 = [0i32; 16];
            for l in 0..16 {
                bsums32[l] = src1.bsums[(bs_off + s * ns) * bs_stride + j0 + l];
            }
            for u in 1..ns {
                for l in 0..16 {
                    bsums32[l] = bsums32[l]
                        .wrapping_add(src1.bsums[(bs_off + s * ns + u) * bs_stride + j0 + l]);
                }
            }
            for t in 0..NUM_ROWS {
                let m = src0.mins[(i0 + t) * nb_stride + nb_off + s];
                for l in 0..16 {
                    s2_acc[t][l] = s2_acc[t][l].wrapping_add(bsums32[l].wrapping_mul(m));
                }
            }
        }
    }

    // epilogue: int->float, per-row scales, fused buf update
    for t in 0..NUM_ROWS {
        let ar = i0 + t;
        let d_off = seff * TILED_MICRO + ar;
        let d0 = src0.d[d_off];
        let dmin0 = src0.dmin[d_off];
        let p = (i0 + t) * buf_stride + j0;
        for l in 0..16 {
            let f1 = s1_acc[t][l] as f32; // vcvtdq2ps
            let mut result = d0 * f1; // vmulps
            if HAS_MIN {
                let f2 = s2_acc[t][l] as f32;
                result = (-dmin0).mul_add(f2, result); // vfnmadd132ps
            }
            buf[p + l] = result.mul_add(d1_vec[l], buf[p + l]); // vfmadd213ps
        }
    }
}

/// `tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>` (tiled-kernel.cpp:
/// 510-539), the reference build's VNNI branch: a 16x16 microtile as two
/// explicit 8x16 band passes. `num_k == 1` takes the NK=1 instantiation
/// (compile-time offsets), else the NK=0 runtime-strided one.
#[allow(clippy::too_many_arguments)]
fn run_microtile<const SUBBLK: usize, const HAS_MIN: bool, const BIAS: i32, const ACTBIAS: bool>(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    i0: usize,
    j0: usize,
    num_k: usize,
    slab: usize,
    buf: &mut [f32],
    buf_stride: usize,
) {
    if num_k == 1 {
        micro_vnni_8x16::<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
        micro_vnni_8x16::<SUBBLK, HAS_MIN, BIAS, 1>(src0, src1, i0 + 8, j0, num_k, slab, buf, buf_stride);
    } else {
        micro_vnni_8x16::<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
        micro_vnni_8x16::<SUBBLK, HAS_MIN, BIAS, 0>(src0, src1, i0 + 8, j0, num_k, slab, buf, buf_stride);
    }
}

/// `tiled_repack_src1` (tiled-kernel.cpp:577-642) — the VNNI build's in-place
/// 16x16 int32 transpose of the natural `[row][k]` codes into the group-local
/// `[kg%16][kg/16][row][4]` layout, four butterfly phases plus the
/// `col_order` store. `bias` is unused on this ISA branch (GGML_UNUSED in C).
fn repack_src1(src1: &mut TiledTileSrc1, row0: usize, num_k: usize, _bias: bool) {
    const IDX_A1: [u32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
    const IDX_B1: [u32; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
    const IDX_A2: [u32; 16] = [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29];
    const IDX_B2: [u32; 16] = [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31];
    const IDX_A3: [u32; 16] = [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23];
    const IDX_B3: [u32; 16] = [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31];
    const IDX_A4: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
    const IDX_B4: [u32; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
    const COL_ORDER: [usize; 16] = [0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15];

    let n_tiles = num_k * (TILED_TILE_K / (TILED_MICRO * 4));
    let row_stride = num_k * TILED_TILE_K;
    let base = row0 * row_stride;
    for t in 0..n_tiles {
        let cbase = base + t * (TILED_MICRO * 4);
        let mut v = [[0u32; 16]; 16];
        for r in 0..16 {
            let bytes = &src1.q[cbase + r * row_stride..][..64];
            for (w, c) in v[r].iter_mut().zip(bytes.chunks_exact(4)) {
                *w = u32::from_le_bytes(c.try_into().unwrap());
            }
        }
        // phase 1: 1-element interleave, pairs (0,1), (2,3), ..., (14,15)
        let mut i = 0;
        while i < 16 {
            let (a, b) = (v[i], v[i + 1]);
            v[i] = perm2var(&a, &IDX_A1, &b);
            v[i + 1] = perm2var(&a, &IDX_B1, &b);
            i += 2;
        }
        // phase 2: 2-element interleave
        let mut i = 0;
        while i < 16 {
            let (a, b) = (v[i], v[i + 2]);
            v[i] = perm2var(&a, &IDX_A2, &b);
            v[i + 2] = perm2var(&a, &IDX_B2, &b);
            let (a, b) = (v[i + 1], v[i + 3]);
            v[i + 1] = perm2var(&a, &IDX_A2, &b);
            v[i + 3] = perm2var(&a, &IDX_B2, &b);
            i += 4;
        }
        // phase 3: 4-element interleave
        let mut i = 0;
        while i < 16 {
            for j in 0..4 {
                let (a, b) = (v[i + j], v[i + 4 + j]);
                v[i + j] = perm2var(&a, &IDX_A3, &b);
                v[i + 4 + j] = perm2var(&a, &IDX_B3, &b);
            }
            i += 8;
        }
        // phase 4: 8-element interleave
        for i in 0..8 {
            let (a, b) = (v[i], v[i + 8]);
            v[i] = perm2var(&a, &IDX_A4, &b);
            v[i + 8] = perm2var(&a, &IDX_B4, &b);
        }
        // store: v[g] holds 16 int32s for k-group col_order[g], rows 0..15
        for (g, &co) in COL_ORDER.iter().enumerate() {
            let dst = &mut src1.q[cbase + co * row_stride..][..64];
            for (c, w) in dst.chunks_exact_mut(4).zip(v[g]) {
                c.copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

/// `_mm512_permutex2var_epi32(a, idx, b)` — lane i takes `a[idx[i]]` when
/// `idx[i] < 16`, else `b[idx[i] - 16]`.
#[inline]
fn perm2var(a: &[u32; 16], idx: &[u32; 16], b: &[u32; 16]) -> [u32; 16] {
    let mut out = [0u32; 16];
    for (o, &i) in out.iter_mut().zip(idx) {
        *o = if i < 16 { a[i as usize] } else { b[i as usize - 16] };
    }
    out
}

/// `tiled_repack_src0<SUBBLK>` (tiled-kernel.cpp:692-714) — the act-bias
/// correction `corr = 128*sum_s scales[s]*(qsum_s - SUBBLK*BIAS)` stored in
/// `mins[r][0]`. Only the AVX2 kernel reads it (its epilogue subtracts the
/// precomputed value, tiled-kernel.cpp:374-377); the VNNI kernel this
/// reference dispatches never does, so the port keeps the documented no-op.
fn repack_src0<const SUBBLK: usize>(
    _tile: &mut TiledTileSrc0,
    _n_rows: usize,
    _num_k: usize,
    _bias: i32,
    _corr: bool,
) {
    debug_assert!(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");
}

// ======================================================================
// gates + window helpers (tiled.cpp:572-754)
// ======================================================================

/// `GGML_CPU_TILED_MM` master switch, default on (tiled.cpp:573-581). C's
/// `atoi`: a non-zero numeric prefix enables.
pub fn matmul_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("GGML_CPU_TILED_MM") {
        None => true,
        Some(v) => atoi(&v.to_string_lossy()) != 0,
    })
}

/// `GGML_CPU_TILED_MM_FORCE` (tiled.cpp:584-592) — test/bench only: take the
/// tiled path even when unprofitable.
fn matmul_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| match std::env::var_os("GGML_CPU_TILED_MM_FORCE") {
        None => false,
        Some(v) => atoi(&v.to_string_lossy()) == 1,
    })
}

/// C `atoi` (whitespace prefix, optional sign, digit prefix).
fn atoi(s: &str) -> i32 {
    let t = s.trim_start();
    let mut it = t.chars().peekable();
    let neg = match it.peek() {
        Some('-') => {
            it.next();
            true
        }
        Some('+') => {
            it.next();
            false
        }
        _ => false,
    };
    let digits: String = it.take_while(char::is_ascii_digit).collect();
    let v: i32 = digits.parse().unwrap_or(0);
    if neg {
        -v
    } else {
        v
    }
}

/// `ggml_tiled_supported` (tiled.cpp:605-643) minus the dead tail: the C's
/// switch returns from every case, so the `src1->type`/contiguity checks
/// after it are unreachable (kept upstream; noted for the 1:1 record).
/// `repacked` stands in for `src0->extra != NULL` (the CPU_REPACK buffer).
pub fn supported(ty: GgmlType, repacked: bool) -> bool {
    matmul_enabled() && !repacked && Fmt::from_type(ty).is_some()
}

/// `ggml_tiled_min_batch` (tiled.cpp:1202-1205).
fn min_batch(rows: i64) -> bool {
    rows >= 8 || matmul_forced()
}

/// `ggml_tiled_narrow_k_extent` (tiled.cpp:664-677) — K chunk for the
/// long/skinny narrow tiles (31k L1 budget, multiple of 256, divides ne00).
fn narrow_k_extent(n_rows: usize, ne00: usize) -> usize {
    let l1 = 31 * 1024;
    let rows = n_rows.max(TILED_MICRO);
    let mut ke = l1 / (TILED_MICRO + rows);
    ke &= !(TILED_TILE_K - 1);
    let ke_max = TILED_TILE_ROWS * TILED_TILE_K / TILED_MICRO;
    if ke > ke_max {
        ke = ke_max;
    }
    while ke >= TILED_TILE_K && ne00 % ke != 0 {
        ke -= TILED_TILE_K;
    }
    if ke < TILED_TILE_K { 0 } else { ke }
}

/// `tiled_store_window` (tiled.cpp:680-711) — j-major buffer -> i-major dst
/// (dst column stride `dst_stride` floats).
fn store_window(
    buf: &[f32],
    n_src0: usize,
    n_src1: usize,
    buf_stride: usize,
    dst: *mut f32,
    dst_stride: usize,
) {
    let mut ri = 0;
    while ri + 16 <= n_src0 {
        let mut rj = 0;
        while rj + 8 <= n_src1 {
            let mut r = [[0f32; 8]; 16];
            for (t, rt) in r.iter_mut().enumerate() {
                rt.copy_from_slice(&buf[(ri + t) * buf_stride + rj..][..8]);
            }
            for u in 0..8 {
                for t in 0..16 {
                    unsafe {
                        *dst.add(ri + t + (rj + u) * dst_stride) = r[t][u];
                    }
                }
            }
            rj += 8;
        }
        while rj < n_src1 {
            for t in 0..16 {
                unsafe {
                    *dst.add(ri + t + rj * dst_stride) = buf[(ri + t) * buf_stride + rj];
                }
            }
            rj += 1;
        }
        ri += 16;
    }
    while ri < n_src0 {
        for j in 0..n_src1 {
            unsafe {
                *dst.add(ri + j * dst_stride) = buf[ri * buf_stride + j];
            }
        }
        ri += 1;
    }
}

/// `tiled_store_window_scatter` (tiled.cpp:723-754) — the MUL_MAT_ID twin;
/// `col_ptrs[j]` points at dst column j's first element.
fn store_window_scatter(buf: &[f32], n_src0: usize, n_src1: usize, buf_stride: usize, col_ptrs: &[*mut f32]) {
    let mut ri = 0;
    while ri + 16 <= n_src0 {
        let mut rj = 0;
        while rj + 8 <= n_src1 {
            let mut r = [[0f32; 8]; 16];
            for (t, rt) in r.iter_mut().enumerate() {
                rt.copy_from_slice(&buf[(ri + t) * buf_stride + rj..][..8]);
            }
            for u in 0..8 {
                for t in 0..16 {
                    unsafe {
                        *col_ptrs[rj + u].add(ri + t) = r[t][u];
                    }
                }
            }
            rj += 8;
        }
        while rj < n_src1 {
            for t in 0..16 {
                unsafe {
                    *col_ptrs[rj].add(ri + t) = buf[(ri + t) * buf_stride + rj];
                }
            }
            rj += 1;
        }
        ri += 16;
    }
    while ri < n_src0 {
        for j in 0..n_src1 {
            unsafe {
                *col_ptrs[j].add(ri) = buf[ri * buf_stride + j];
            }
        }
        ri += 1;
    }
}

/// Resolve `(SUBBLK, HAS_MIN, BIAS, ACTBIAS)` to one monomorphized method
/// call — the closed set of tiled.cpp:1171-1196.
macro_rules! with_kernel {
    ($fmt:expr, $f:ident ( $c:expr, $run:expr )) => {
        match $fmt.kernel_consts() {
            (16, false, 32, true) => $f::<16, false, 32, true>($c, $run),
            (32, true, 0, false) => $f::<32, true, 0, false>($c, $run),
            (16, false, 4, true) => $f::<16, false, 4, true>($c, $run),
            (32, false, 128, false) => $f::<32, false, 128, false>($c, $run),
            (32, false, 128, true) => $f::<32, false, 128, true>($c, $run),
            (16, false, 128, true) => $f::<16, false, 128, true>($c, $run),
            (16, true, 0, false) => $f::<16, true, 0, false>($c, $run),
            other => unreachable!("tiled: no instantiation for {other:?}"),
        }
    };
    ($fmt:expr, $self:ident . $method:ident ( $run:expr )) => {
        match $fmt.kernel_consts() {
            (16, false, 32, true) => $self.$method::<16, false, 32, true>($run),
            (32, true, 0, false) => $self.$method::<32, true, 0, false>($run),
            (16, false, 4, true) => $self.$method::<16, false, 4, true>($run),
            (32, false, 128, false) => $self.$method::<32, false, 128, false>($run),
            (32, false, 128, true) => $self.$method::<32, false, 128, true>($run),
            (16, false, 128, true) => $self.$method::<16, false, 128, true>($run),
            (16, true, 0, false) => $self.$method::<16, true, 0, false>($run),
            other => unreachable!("tiled: no instantiation for {other:?}"),
        }
    };
}

// ======================================================================
// MUL_MAT driver (tiled.cpp:895-1223)
// ======================================================================

/// Resolved tensor view for the mul_mat driver. Raw pointers follow the
/// port's kernel-side convention (compute.rs's `loc` plumbing); bounds are
/// asserted by the caller exactly like the C driver's own GGML_ASSERTs.
pub(crate) struct MmArgs {
    pub fmt: Fmt,
    /// `src0->data`
    pub src0: *const u8,
    pub src0_bs: usize,
    pub nb01: usize,
    pub src0_nb2: usize,
    pub src0_nb3: usize,
    pub ne00: usize,
    pub ne01: usize,
    pub ne02: usize,
    pub ne03: usize,
    /// src1 F32 base; rows at `i11*nb11 + i12*nb12 + i13*nb13`.
    pub src1: *const u8,
    pub nb11: usize,
    pub nb12: usize,
    pub nb13: usize,
    pub ne11: usize,
    pub ne12: usize,
    pub ne13: usize,
    /// Prebuilt q8_K activation rows (only when src1 already is Q8_K —
    /// `src1->type == vec_dot_type` in the C); else the driver quantizes.
    pub prebuilt_wd: Option<(*const u8, usize)>,
    pub dst: *mut u8,
    pub dst_nb1: usize,
    pub dst_nb2: usize,
    pub dst_nb3: usize,
    pub nth: usize,
}

impl MmArgs {
    /// `ggml_compute_forward_mul_mat_tiled` (tiled.cpp:1209-1223) — full-op
    /// entry; false falls through to the stock path.
    pub(crate) fn compute(&self, run: &dyn Fn(usize, &(dyn Fn(usize, usize) + Sync))) -> bool {
        if !min_batch(self.ne11 as i64) {
            return false;
        }
        with_kernel!(self.fmt, self.driver(run))
    }

    /// `ggml_compute_forward_mul_mat_tiled_driver` (tiled.cpp:1027-1144).
    fn driver<const S: usize, const H: bool, const B: i32, const A: bool>(
        &self,
        run: &dyn Fn(usize, &(dyn Fn(usize, usize) + Sync)),
    ) -> bool {
        let ne00 = self.ne00;
        let (ne11, ne12, ne13) = (self.ne11, self.ne12, self.ne13);
        debug_assert!(ne00 % TILED_TILE_K == 0, "tiled: ne00 % 256");
        let nblocks = ne00 / QK_K;
        let wd_row = nblocks * Q8K_BLOCK;
        let rows_total = ne11 * ne12 * ne13;
        let nth = self.nth.max(1);

        // from_float pass (tiled.cpp:1057-1080): quantize src1 to q8_K with
        // the C's per-thread *block slices* of every row — block-granular, so
        // the bytes equal a whole-row quantize. One AtomicPtr hands the
        // disjoint slices to the team (the port's wdata idiom).
        let wd_buf = vec![0u8; rows_total * wd_row];
        let wd: usize = match self.prebuilt_wd {
            Some((p, _)) => p as usize,
            None => {
                let wd_ptr = std::sync::atomic::AtomicPtr::new(wd_buf.as_ptr() as *mut u8);
                let src1 = self.src1 as usize;
                let (nb11, nb12, nb13) = (self.nb11, self.nb12, self.nb13);
                let do_thread = move |th: usize| {
                    let b0 = th * nblocks / nth;
                    let b1 = (th + 1) * nblocks / nth;
                    if b1 <= b0 {
                        return;
                    }
                    let base = wd_ptr.load(std::sync::atomic::Ordering::Relaxed);
                    for i13 in 0..ne13 {
                        for i12 in 0..ne12 {
                            for i11 in 0..ne11 {
                                let ir = (i13 * ne12 + i12) * ne11 + i11;
                                let src_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                                // SAFETY: the row slice lives inside src1's
                                // storage (bounds asserted by the caller); the
                                // out slice is this thread's block slice of
                                // wd_buf (slices across threads are disjoint).
                                let row: &[f32] = unsafe {
                                    core::slice::from_raw_parts(
                                        (src1 + src_off + b0 * QK_K * 4) as *const f32,
                                        (b1 - b0) * QK_K,
                                    )
                                };
                                let out: &mut [u8] = unsafe {
                                    core::slice::from_raw_parts_mut(
                                        base.add(ir * wd_row + b0 * Q8K_BLOCK),
                                        (b1 - b0) * Q8K_BLOCK,
                                    )
                                };
                                let blocks: &mut [BlockQ8K] = bytemuck::cast_slice_mut(out);
                                crate::quants::quantize_row_q8_K(row, blocks);
                            }
                        }
                    }
                };
                if nth <= 1 {
                    do_thread(0);
                } else {
                    run(nth, &|lo, hi| {
                        // one shard per thread id (lo..hi is [th, th+1))
                        for th in lo..hi {
                            do_thread(th);
                        }
                    });
                }
                wd_buf.as_ptr() as usize
            }
        };

        // chunk grid (tiled.cpp:1095-1112): 256-row chunks, halved while the
        // grid is too coarse for nth*4 (floor 16 = the microtile edge)
        let nr0 = self.ne01;
        let nr1 = rows_total;
        let mut chunk_size: usize = 256;
        let mut nchunk0 = nr0.div_ceil(chunk_size);
        let mut nchunk1 = nr1.div_ceil(chunk_size);
        while nchunk0 * nchunk1 < nth * 4 && chunk_size > 16 {
            chunk_size /= 2;
            nchunk0 = nr0.div_ceil(chunk_size);
            nchunk1 = nr1.div_ceil(chunk_size);
        }
        let dr0 = nr0.div_ceil(nchunk0);
        let dr1 = nr1.div_ceil(nchunk1);
        let jobs = nchunk0 * nchunk1;

        let this = MmChunk {
            fmt: self.fmt,
            src0: self.src0 as usize,
            src0_bs: self.src0_bs,
            nb01: self.nb01,
            src0_nb2: self.src0_nb2,
            src0_nb3: self.src0_nb3,
            src0_ne02: self.ne02,
            src0_ne03: self.ne03,
            ne00,
            wd,
            nbw1: wd_row,
            dst: self.dst as usize,
            dst_nb1: self.dst_nb1,
            dst_nb2: self.dst_nb2,
            dst_nb3: self.dst_nb3,
            ne11,
            ne12,
            ne13,
        };
        let run_chunk = |c: usize| {
            let ith0 = c % nchunk0;
            let ith1 = c / nchunk0;
            let ir0_start = dr0 * ith0;
            let ir0_end = (ir0_start + dr0).min(nr0);
            let ir1_start = dr1 * ith1;
            let ir1_end = (ir1_start + dr1).min(nr1);
            if ir0_start >= ir0_end || ir1_start >= ir1_end {
                return;
            }
            WS.with_borrow_mut(|ws| {
                let TiledWs { src0, src1, acc } = ws;
                one_chunk::<S, H, B, A>(
                    &this,
                    src0,
                    src1,
                    &mut acc[..],
                    ir0_start,
                    ir0_end,
                    ir1_start,
                    ir1_end,
                );
            });
        };
        if jobs == 1 {
            run_chunk(0);
        } else {
            run(jobs, &|lo, hi| {
                for c in lo..hi {
                    run_chunk(c);
                }
            });
        }
        true
    }
}

/// Per-chunk resolved view (one_chunk re-derives these from the tensors;
/// hoisting keeps the address math in one place). Pointers travel as `usize`
/// so the struct is `Copy + Send + Sync` through the Team closures.
#[derive(Clone, Copy)]
struct MmChunk {
    fmt: Fmt,
    src0: usize,
    src0_bs: usize,
    nb01: usize,
    src0_nb2: usize,
    src0_nb3: usize,
    src0_ne02: usize,
    src0_ne03: usize,
    ne00: usize,
    wd: usize,
    nbw1: usize,
    dst: usize,
    dst_nb1: usize,
    dst_nb2: usize,
    dst_nb3: usize,
    ne11: usize,
    ne12: usize,
    ne13: usize,
}

/// `ggml_compute_forward_mul_mat_tiled_one_chunk` (tiled.cpp:895-1025).
#[allow(clippy::too_many_arguments)]
fn one_chunk<const S: usize, const H: bool, const B: i32, const A: bool>(
    c: &MmChunk,
    src0_tile: &mut TiledTileSrc0,
    src1_tile: &mut TiledTileSrc1,
    acc: &mut [f32],
    ir0_start: usize,
    ir0_end: usize,
    ir1_start: usize,
    ir1_end: usize,
) {
    const TILE: usize = 256;
    const MICRO: usize = 16;
    let ne00 = c.ne00;
    let ne11 = c.ne11;
    let r2 = c.ne12 / c.src0_ne02;
    let r3 = c.ne13 / c.src0_ne03;
    debug_assert!(r2 >= 1 && r3 >= 1, "tiled: ne12 % ne02");
    let stride = c.nb01 / c.src0_bs;

    // 256-wide windows over the chunk; the iir1 window is clamped at the src1
    // batch (ne11) boundary (tiled.cpp:935-952)
    let mut iir1 = ir1_start;
    while iir1 < ir1_end {
        let mut iir1_end = (iir1 + TILE).min(ir1_end);
        let bnd = (iir1 / ne11 + 1) * ne11;
        if bnd < iir1_end {
            iir1_end = bnd;
        }
        let n_src1 = iir1_end - iir1;
        let i13 = iir1 / (c.ne12 * ne11);
        let i12 = (iir1 - i13 * c.ne12 * ne11) / ne11;
        let i11 = iir1 - i13 * c.ne12 * ne11 - i12 * ne11;
        let i02 = i12 / r2;
        let i03 = i13 / r3;
        // SAFETY: plane base pointers inside src0/dst as bound by the caller.
        let src0_row = c.src0 + i02 * c.src0_nb2 + i03 * c.src0_nb3;
        let dst_col = c.dst + i12 * c.dst_nb2 + i13 * c.dst_nb3;

        // rows[r]: the window's r-th q8_K work row base (tiled.cpp:963-966)
        let mut rows: [*const u8; TILED_TILE_ROWS] = [core::ptr::null(); TILED_TILE_ROWS];
        for (r, rp) in rows.iter_mut().enumerate().take(n_src1) {
            *rp = (c.wd + (iir1 + r) * c.nbw1) as *const u8;
        }

        let mut iir0 = ir0_start;
        while iir0 < ir0_end {
            let iir0_end = (iir0 + TILE).min(ir0_end);
            let n_src0 = iir0_end - iir0;

            acc.fill(0.0);

            if n_src1 <= MICRO {
                // narrow path (tiled.cpp:975-999): longer, skinny tiles
                let k_extent = narrow_k_extent(n_src1, ne00);
                let num_k = k_extent / TILED_TILE_K;
                let mut k0 = 0usize;
                while k0 < ne00 {
                    let kstart = k0 / TILED_TILE_K;
                    unpack_src1_q8k(&rows[..n_src1], n_src1, src1_tile, kstart, num_k);
                    repack_src1(src1_tile, 0, num_k, A);
                    let mut ir0 = iir0;
                    while ir0 < iir0_end {
                        let n0 = MICRO.min(iir0_end - ir0);
                        // SAFETY: the band's k_extent chunk of src0.
                        let wbase = (src0_row + ir0 * c.nb01 + kstart * c.src0_bs) as *const u8;
                        unpack_src0(c.fmt, wbase, stride, n0, src0_tile, num_k);
                        repack_src0::<S>(src0_tile, n0, num_k, B, A);
                        let buf_off = (ir0 - iir0) * TILED_TILE_ROWS;
                        for slab in 0..num_k {
                            run_microtile::<S, H, B, A>(
                                src0_tile,
                                src1_tile,
                                0,
                                0,
                                num_k,
                                slab,
                                &mut acc[buf_off..],
                                TILED_TILE_ROWS,
                            );
                        }
                        ir0 += MICRO;
                    }
                    k0 += k_extent;
                }
            } else {
                // standard per-slab path (tiled.cpp:1000-1017)
                let mut ib = 0usize;
                while ib < ne00 {
                    let kblk = ib / TILE;
                    // SAFETY: the 256-row window at (iir0, kblk).
                    let wbase = (src0_row + iir0 * c.nb01 + kblk * c.src0_bs) as *const u8;
                    unpack_src0(c.fmt, wbase, stride, n_src0, src0_tile, 1);
                    repack_src0::<S>(src0_tile, n_src0, 1, B, A);
                    unpack_src1_q8k(&rows[..n_src1], n_src1, src1_tile, kblk, 1);
                    let mut ir1 = iir1;
                    while ir1 < iir1_end {
                        repack_src1(src1_tile, ir1 - iir1, 1, A);
                        let mut ir0 = iir0;
                        while ir0 < iir0_end {
                            run_microtile::<S, H, B, A>(
                                src0_tile,
                                src1_tile,
                                ir0 - iir0,
                                ir1 - iir1,
                                1,
                                0,
                                acc,
                                TILED_TILE_ROWS,
                            );
                            ir0 += MICRO;
                        }
                        ir1 += MICRO;
                    }
                    ib += TILE;
                }
            }
            // write acc back out (tiled.cpp:1019-1021)
            // SAFETY: dst window (n_src0 x n_src1) at the chunk's base.
            let dst_ptr = (dst_col + iir0 * 4 + i11 * c.dst_nb1) as *mut f32;
            store_window(acc, n_src0, n_src1, TILED_TILE_ROWS, dst_ptr, c.dst_nb1 / 4);
            iir0 += TILE;
        }
        iir1 = iir1_end;
    }
}

// ======================================================================
// MUL_MAT_ID driver (tiled.cpp:713-1245)
// ======================================================================

/// Resolved tensor view for one expert of MUL_MAT_ID. Pointers travel as
/// `usize` (see `MmChunk`).
#[derive(Clone, Copy)]
pub(crate) struct MmidArgs {
    pub fmt: Fmt,
    /// `src0->data + cur_a * nb02` (the expert's plane base)
    pub src0: usize,
    pub src0_bs: usize,
    pub nb01: usize,
    pub ne00: usize,
    pub ne01: usize,
    /// q8_K work rows (`wdata`, or src1->data when it already is Q8_K)
    pub wd: usize,
    pub nbw1: usize,
    pub ne11: usize,
    pub dst: usize,
    pub dst_nb1: usize,
    pub dst_nb2: usize,
    /// `cne1` routed rows of `expert_rows` (slot, token) int32 pairs
    pub expert_rows: usize,
    pub cne1: usize,
    pub nth: usize,
}

/// `ggml_compute_forward_mul_mat_id_tiled_one_expert` (tiled.cpp:838-893).
/// The C strides experts across the team (`g0..g1` row groups per thread,
/// all k windows serial inside); windows are value-independent (each zeroes
/// its acc region before accumulating, tiled.cpp:775-777), so the port
/// schedules (k window x group) pairs through the Team instead.
#[allow(clippy::too_many_arguments)]
fn one_expert<const S: usize, const H: bool, const B: i32, const A: bool>(
    c: &MmidArgs,
    run: &dyn Fn(usize, &(dyn Fn(usize, usize) + Sync)),
) -> bool {
    let (ne00, ne01) = (c.ne00, c.ne01);
    let cne1 = c.cne1;
    let ngroups = ne01.div_ceil(TILED_MMID_GROUP);
    let nk = cne1.div_ceil(TILED_TILE_K);

    let job = *c;
    let run_job = move |j: usize| {
        let c = &job;
        let kw = j / ngroups;
        let g = j % ngroups;
        let k = kw * TILED_TILE_K;
        let nrows = TILED_TILE_K.min(cne1 - k);
        // routed src1 rows of this k window (tiled.cpp:880-885)
        let mut rows: [*const u8; TILED_TILE_ROWS] = [core::ptr::null(); TILED_TILE_ROWS];
        // SAFETY: expert_rows holds 2*cne1 i32s (the caller's mapping table);
        // the wd row addresses are bound by the caller's tensor views.
        for i in 0..nrows {
            let i11 = unsafe { *(c.expert_rows as *const i32).add(2 * (k + i)) } as usize % c.ne11;
            let i12 = unsafe { *(c.expert_rows as *const i32).add(2 * (k + i) + 1) } as usize;
            rows[i] = (c.wd + (i11 + i12 * c.ne11) * c.nbw1) as *const u8;
        }
        WS.with_borrow_mut(|ws| {
            let TiledWs { src0, src1, acc } = ws;
            mmid_gemm_window::<S, H, B, A>(
                c,
                src0,
                src1,
                &mut acc[..],
                g * TILED_MMID_GROUP,
                k,
                nrows,
                rows,
            );
        });
    };
    let jobs = nk * ngroups;
    if jobs == 1 {
        run_job(0);
    } else {
        run(jobs, &|lo, hi| {
            for j in lo..hi {
                run_job(j);
            }
        });
    }
    true
}

impl MmidArgs {
    /// `ggml_compute_forward_mul_mat_id_tiled` (tiled.cpp:1227-1245) — one
    /// expert; false falls through to the stock per-row path.
    pub(crate) fn compute(&self, run: &dyn Fn(usize, &(dyn Fn(usize, usize) + Sync))) -> bool {
        if !min_batch(self.cne1 as i64) {
            return false;
        }
        one_expert_entry(self, run)
    }
}

fn one_expert_entry(c: &MmidArgs, run: &dyn Fn(usize, &(dyn Fn(usize, usize) + Sync))) -> bool {
    with_kernel!(c.fmt, one_expert(c, run))
}

/// `tiled_mmid_gemm_window` (tiled.cpp:759-832) — one (g, k) macrotile.
#[allow(clippy::too_many_arguments)]
fn mmid_gemm_window<const S: usize, const H: bool, const B: i32, const A: bool>(
    c: &MmidArgs,
    src0_tile: &mut TiledTileSrc0,
    src1_tile: &mut TiledTileSrc1,
    acc: &mut [f32],
    r: usize,
    k: usize,
    nrows: usize,
    rows: [*const u8; TILED_TILE_ROWS],
) {
    let (ne00, ne01) = (c.ne00, c.ne01);
    let r_end = (r + TILED_MMID_GROUP).min(ne01);
    let n_src0 = r_end - r;
    let stride = c.nb01 / c.src0_bs;

    // zero only the window region of acc (tiled.cpp:775-777)
    for i in 0..n_src0 {
        acc[i * TILED_TILE_ROWS..i * TILED_TILE_ROWS + nrows].fill(0.0);
    }

    // scattered writeback columns (tiled.cpp:780-784): col m goes to its
    // routed dst row, r*nb0 is the window row offset (nb0 == 4, F32 dst)
    let mut col_ptrs: [*mut f32; TILED_TILE_ROWS] = [core::ptr::null_mut(); TILED_TILE_ROWS];
    for m in 0..nrows {
        // SAFETY: expert_rows pairs + dst strides bound by the caller.
        let slot = unsafe { *(c.expert_rows as *const i32).add(2 * (k + m)) } as usize;
        let token = unsafe { *(c.expert_rows as *const i32).add(2 * (k + m) + 1) } as usize;
        col_ptrs[m] = (c.dst + r * 4 + slot * c.dst_nb1 + token * c.dst_nb2) as *mut f32;
    }

    if nrows <= TILED_MICRO {
        // narrow path (tiled.cpp:788-810)
        let k_extent = narrow_k_extent(nrows, ne00);
        let num_k = k_extent / TILED_TILE_K;
        let mut k0 = 0usize;
        while k0 < ne00 {
            let kstart = k0 / TILED_TILE_K;
            unpack_src1_q8k(&rows[..nrows], nrows, src1_tile, kstart, num_k);
            repack_src1(src1_tile, 0, num_k, A);
            let mut ir0 = r;
            while ir0 < r_end {
                let n0 = TILED_MICRO.min(r_end - ir0);
                // SAFETY: the band's k_extent chunk of the expert plane.
                let wbase = (c.src0 + ir0 * c.nb01 + kstart * c.src0_bs) as *const u8;
                unpack_src0(c.fmt, wbase, stride, n0, src0_tile, num_k);
                repack_src0::<S>(src0_tile, n0, num_k, B, A);
                let buf_off = (ir0 - r) * TILED_TILE_ROWS;
                for slab in 0..num_k {
                    run_microtile::<S, H, B, A>(
                        src0_tile,
                        src1_tile,
                        0,
                        0,
                        num_k,
                        slab,
                        &mut acc[buf_off..],
                        TILED_TILE_ROWS,
                    );
                }
                ir0 += TILED_MICRO;
            }
            k0 += k_extent;
        }
    } else {
        // standard per-slab path (tiled.cpp:812-829)
        let mut ib = 0usize;
        while ib < ne00 {
            let kblk = ib / TILED_TILE_K;
            // SAFETY: the (<=64)-row window at (r, kblk) of the expert plane.
            let wbase = (c.src0 + r * c.nb01 + kblk * c.src0_bs) as *const u8;
            unpack_src0(c.fmt, wbase, stride, n_src0, src0_tile, 1);
            repack_src0::<S>(src0_tile, n_src0, 1, B, A);
            unpack_src1_q8k(&rows[..nrows], nrows, src1_tile, kblk, 1);
            let mut ir1 = 0usize;
            while ir1 < nrows {
                repack_src1(src1_tile, ir1, 1, A);
                let mut ir0 = r;
                while ir0 < r_end {
                    run_microtile::<S, H, B, A>(
                        src0_tile,
                        src1_tile,
                        ir0 - r,
                        ir1,
                        1,
                        0,
                        acc,
                        TILED_TILE_ROWS,
                    );
                    ir0 += TILED_MICRO;
                }
                ir1 += TILED_MICRO;
            }
            ib += TILED_TILE_K;
        }
    }

    store_window_scatter(acc, n_src0, nrows, TILED_TILE_ROWS, &col_ptrs[..nrows]);
}

// ======================================================================
// ground-truth replay — parity/tiled_ref.bin (parity/ref_tiled_dump.c)
// ======================================================================

#[cfg(test)]
mod tiled_ref_tests {
    use crate::compute::graph_compute;
    use crate::graph::Graph;
    use crate::tensor::Context;
    use crate::types::GgmlType;

    enum Section<'a> {
        Mm {
            ty: GgmlType,
            n: usize,
            rows: usize,
            cols: usize,
            xq: &'a [u8],
            y: &'a [f32],
            dst: &'a [f32],
        },
        Mmid {
            ty: GgmlType,
            n: usize,
            rows: usize,
            ne11: usize,
            n_as: usize,
            n_ids: usize,
            ne12: usize,
            xq: &'a [u8],
            y: &'a [f32],
            ids: &'a [i32],
            dst: &'a [f32],
        },
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
            assert_eq!(magic, 0x3154_4D56, "VMT1 magic");
            let mut h = [0u32; 5];
            for v in h.iter_mut() {
                *v = u32::from_le_bytes(c[..4].try_into().unwrap());
                c = &c[4..];
            }
            let (kind, ty, n, rows, cols) = (h[0], h[1], h[2] as usize, h[3] as usize, h[4] as usize);
            let ty = GgmlType::from_u32(ty).unwrap();
            if kind == 0 {
                let xq_len = ty.row_size(n) * rows;
                let (xq, rest) = c.split_at(xq_len);
                c = rest;
                let (y, rest) = c.split_at(n * cols * 4);
                c = rest;
                let (dst, rest) = c.split_at(rows * cols * 4);
                c = rest;
                out.push(Section::Mm {
                    ty,
                    n,
                    rows,
                    cols,
                    xq,
                    y: bytemuck::cast_slice(y),
                    dst: bytemuck::cast_slice(dst),
                });
            } else {
                let mut h2 = [0u32; 3];
                for v in h2.iter_mut() {
                    *v = u32::from_le_bytes(c[..4].try_into().unwrap());
                    c = &c[4..];
                }
                let (n_as, n_ids, ne12) = (h2[0] as usize, h2[1] as usize, h2[2] as usize);
                let xq_len = ty.row_size(n) * rows * n_as;
                let (xq, rest) = c.split_at(xq_len);
                c = rest;
                let (y, rest) = c.split_at(n * cols * ne12 * 4);
                c = rest;
                let (ids, rest) = c.split_at(n_ids * ne12 * 4);
                c = rest;
                let (dst, rest) = c.split_at(rows * n_ids * ne12 * 4);
                c = rest;
                out.push(Section::Mmid {
                    ty,
                    n,
                    rows,
                    ne11: cols,
                    n_as,
                    n_ids,
                    ne12,
                    xq,
                    y: bytemuck::cast_slice(y),
                    ids: bytemuck::cast_slice(ids),
                    dst: bytemuck::cast_slice(dst),
                });
            }
        }
        out
    }

    fn ne11_of(cols: u32) -> usize {
        cols as usize
    }

    fn load() -> Option<Vec<Section<'static>>> {
        let path = std::env::var("SYNCG_TILED_BIN").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/tiled_ref.bin").to_string());
        let bytes = std::fs::read(path).ok()?;
        // SAFETY-free: leak the artifact so the sections can borrow it; test
        // binaries are short-lived.
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Some(parse(bytes))
    }

    fn run_mm(ty: GgmlType, n: usize, rows: usize, cols: usize, xq: &[u8], y: &[f32]) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(ty, n as i64, rows as i64);
        let b = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
        let d = ctx.mul_mat(a, b);
        for t in [a, b] {
            ctx.arena_resize_tensor(t);
        }
        ctx.data_bytes_mut(a).unwrap().copy_from_slice(xq);
        ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(y));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        graph_compute(&mut ctx, &mut g, 4);
        bytemuck::cast_slice(ctx.data_bytes(d).unwrap()).to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mmid(
        ty: GgmlType,
        n: usize,
        rows: usize,
        ne11: usize,
        n_as: usize,
        n_ids: usize,
        ne12: usize,
        xq: &[u8],
        y: &[f32],
        ids: &[i32],
    ) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(ty, n as i64, rows as i64, n_as as i64);
        let b = ctx.new_tensor_3d(GgmlType::F32, n as i64, ne11 as i64, ne12 as i64);
        let iid = ctx.new_tensor_2d(GgmlType::I32, n_ids as i64, ne12 as i64);
        let d = ctx.mul_mat_id(a, b, iid);
        for t in [a, b, iid] {
            ctx.arena_resize_tensor(t);
        }
        ctx.data_bytes_mut(a).unwrap().copy_from_slice(xq);
        ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(y));
        ctx.data_bytes_mut(iid).unwrap().copy_from_slice(bytemuck::cast_slice(ids));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        graph_compute(&mut ctx, &mut g, 4);
        bytemuck::cast_slice(ctx.data_bytes(d).unwrap()).to_vec()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Every MUL_MAT section bit-exact vs the NEW reference build (tiled path
    /// active: cols >= 8 everywhere).
    #[test]
    fn tiled_mulmat_matches_reference_bit_exact() {
        let Some(sections) = load() else {
            eprintln!("skip: parity/tiled_ref.bin missing (build parity/ref_tiled_dump.c)");
            return;
        };
        let mut checked = 0usize;
        for s in &sections {
            let Section::Mm { ty, n, rows, cols, xq, y, dst } = s else { continue };
            let got = run_mm(*ty, *n, *rows, *cols, xq, y);
            assert_eq!(bits(&got), bits(dst), "type {ty:?} n={n} R={rows} C={cols}");
            checked += 1;
        }
        assert_eq!(checked, 52, "52 mm sections in the artifact");
    }

    /// Every MUL_MAT_ID section bit-exact (mixed tiled/stock expert batches).
    #[test]
    fn tiled_mulmat_id_matches_reference_bit_exact() {
        let Some(sections) = load() else {
            eprintln!("skip: parity/tiled_ref.bin missing");
            return;
        };
        let mut checked = 0usize;
        for s in &sections {
            let Section::Mmid { ty, n, rows, ne11, n_as, n_ids, ne12, xq, y, ids, dst } = s else {
                continue;
            };
            let got = run_mmid(*ty, *n, *rows, *ne11, *n_as, *n_ids, *ne12, xq, y, ids);
            assert_eq!(
                bits(&got),
                bits(dst),
                "type {ty:?} n={n} R={rows} ne11={ne11} as={n_as} ids={n_ids} t={ne12}"
            );
            checked += 1;
        }
        assert_eq!(checked, 13, "13 mmid sections in the artifact");
    }

    /// The C's chunk grid only assigns work; every element must be
    /// thread-count invariant (mirrors mulmat's
    /// `mul_mat_bit_identical_across_thread_counts`).
    #[test]
    fn tiled_mulmat_thread_invariant() {
        let Some(sections) = load() else { return };
        let Section::Mm { ty, n, rows, cols, xq, y, dst } = &sections[0] else { return };
        let t1 = run_mm(*ty, *n, *rows, *cols, xq, y);
        // graph_compute with a different team size
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(*ty, *n as i64, *rows as i64);
        let b = ctx.new_tensor_2d(GgmlType::F32, *n as i64, *cols as i64);
        let d = ctx.mul_mat(a, b);
        for t in [a, b] {
            ctx.arena_resize_tensor(t);
        }
        ctx.data_bytes_mut(a).unwrap().copy_from_slice(xq);
        ctx.data_bytes_mut(b).unwrap().copy_from_slice(bytemuck::cast_slice(y));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, d);
        graph_compute(&mut ctx, &mut g, 1);
        let t2: Vec<f32> = bytemuck::cast_slice(ctx.data_bytes(d).unwrap()).to_vec();
        assert_eq!(bits(&t1), bits(&t2));
        assert_eq!(bits(&t1), bits(dst));
    }
}


