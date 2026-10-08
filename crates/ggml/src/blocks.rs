//! Quantization block layouts — 1:1 port of ggml-common.h (`GGML_COMMON_DECL` section).
//!
//! Byte layouts must match exactly (GGUF files are read as these structs).
//! `size_of` checks below mirror the C `static_assert`s.

use bytemuck::{Pod, Zeroable};
use half::f16;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = QK_K / 64 * 3; // 12

// ---- legacy quads/octets (QK = 32..128) ----

pub const QK1_0: usize = 128;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ1_0 {
    pub d: f16,
    pub qs: [u8; QK1_0 / 8],
}

pub const QK2_0: usize = 64;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ2_0 {
    pub d: f16,
    pub qs: [u8; QK2_0 / 4],
}

pub const QK4_0: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

pub const QK4_1: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK4_1 / 2],
}

pub const QK_MXFP4: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockMxfp4 {
    pub e: u8, // E8M0 scale
    pub qs: [u8; QK_MXFP4 / 2],
}

pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockNvfp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB], // UE4M3 sub-block scales
    pub qs: [u8; QK_NVFP4 / 2],
}

pub const QK5_0: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_0 {
    pub d: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}

pub const QK5_1: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}

pub const QK8_0: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

pub const QK8_1: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_1 {
    pub d: f16,
    pub s: f16, // d * sum(qs[i])
    pub qs: [i8; QK8_1],
}

// ---- ternary ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockTq1_0 {
    pub qs: [u8; (QK_K - 4 * QK_K / 64) / 5], // 48
    pub qh: [u8; QK_K / 64],                  // 4
    pub d: f16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockTq2_0 {
    pub qs: [u8; QK_K / 4],
    pub d: f16,
}

// ---- K-quants (super-blocks of QK_K=256) ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16], // scales and mins, 4-bit
    pub qs: [u8; QK_K / 4],
    pub d: f16,
    pub dmin: f16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12], // 6-bit
    pub d: f16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE], // 6-bit
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: f16,
}

/// Intermediate format for dot products (never stored in GGUF as weights).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

// ---- i-quants ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2Xxs {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2Xs {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
    pub scales: [u8; QK_K / 32],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq3Xxs {
    pub d: f16,
    pub qs: [u8; 3 * QK_K / 8],
}

pub const IQ3S_N_SCALE: usize = QK_K / 64;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq3S {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; IQ3S_N_SCALE],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq1S {
    pub d: f16,
    pub qs: [u8; QK_K / 8],
    pub qh: [u16; QK_K / 32],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq1M {
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
    pub scales: [u8; QK_K / 32],
}

pub const QK4_NL: usize = 32;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq4Nl {
    pub d: f16,
    pub qs: [u8; QK4_NL / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq4Xs {
    pub d: f16,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

// static_assert equivalents — these pin the byte layouts to the C reference.
const _: () = {
    assert!(size_of::<BlockQ1_0>() == 2 + QK1_0 / 8);
    assert!(size_of::<BlockQ2_0>() == 2 + QK2_0 / 4);
    assert!(size_of::<BlockQ4_0>() == 2 + QK4_0 / 2);
    assert!(size_of::<BlockQ4_1>() == 4 + QK4_1 / 2);
    assert!(size_of::<BlockMxfp4>() == 1 + QK_MXFP4 / 2);
    assert!(size_of::<BlockNvfp4>() == QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2);
    assert!(size_of::<BlockQ5_0>() == 2 + 4 + QK5_0 / 2);
    assert!(size_of::<BlockQ5_1>() == 4 + 4 + QK5_1 / 2);
    assert!(size_of::<BlockQ8_0>() == 2 + QK8_0);
    assert!(size_of::<BlockQ8_1>() == 4 + QK8_1);
    assert!(size_of::<BlockTq1_0>() == 2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5);
    assert!(size_of::<BlockTq2_0>() == 2 + QK_K / 4);
    assert!(size_of::<BlockQ2K>() == 4 + QK_K / 16 + QK_K / 4);
    assert!(size_of::<BlockQ3K>() == 2 + QK_K / 4 + QK_K / 8 + 12);
    assert!(size_of::<BlockQ4K>() == 4 + K_SCALE_SIZE + QK_K / 2);
    assert!(size_of::<BlockQ5K>() == 4 + K_SCALE_SIZE + QK_K / 2 + QK_K / 8);
    assert!(size_of::<BlockQ6K>() == 2 + QK_K / 16 + 3 * QK_K / 4);
    assert!(size_of::<BlockQ8K>() == 4 + QK_K + 2 * QK_K / 16);
    assert!(size_of::<BlockIq2Xxs>() == 2 + 2 * QK_K / 8);
    assert!(size_of::<BlockIq2Xs>() == 2 + 2 * QK_K / 8 + QK_K / 32);
    assert!(size_of::<BlockIq2S>() == 2 + QK_K / 4 + QK_K / 16);
    assert!(size_of::<BlockIq3Xxs>() == 2 + 3 * QK_K / 8);
    assert!(size_of::<BlockIq3S>() == 2 + 13 * (QK_K / 32) + IQ3S_N_SCALE);
    assert!(size_of::<BlockIq1S>() == 2 + QK_K / 8 + QK_K / 16);
    assert!(size_of::<BlockIq1M>() == QK_K / 8 + QK_K / 16 + QK_K / 32);
    assert!(size_of::<BlockIq4Nl>() == 2 + QK4_NL / 2);
    assert!(size_of::<BlockIq4Xs>() == 2 + 2 + QK_K / 64 + QK_K / 2);
};
