//! ops.rs — op builders (port of ggml.c constructors). Owner: agent A.
//!
//! Each method mirrors the corresponding `ggml_*` constructor in
//! ggml/src/ggml.c (line numbers refer to the pinned worktree @ bd4f514db1).
//! Results carry `op`, `src`, `op_params` exactly like the C tensors so that
//! compute.rs can dispatch 1:1.
//!
//! Notable encoding differences forced by tensor.rs's current (shared) API —
//! see the final report:
//!   * RMS_NORM is stored as GgmlOp::Norm with op_params[1] == 1
//!     (`norm` writes op_params[1] == 0). op_params[0] holds the f32 bits of
//!     eps in both cases, identical to C.
//!   * GELU is stored as GgmlOp::Silu (= GGML_OP_UNARY in ggml.h) with
//!     op_params[0] = unary op code: 4 = TANH, 8 = GELU, 10 = SILU (ggml.h
//!     enum).
//!   * GGML_OP_CONT reuses GgmlOp::Dup (compute paths are identical in C).
//!   * GET_ROWS_BACK is stored as GgmlOp::GetRows with op_params[15] == 1.
//!   * rope/soft_max src[2] (freq factors / attention sinks) is not exposed by
//!     the ported builder signatures — they pass `None` (the slot itself exists
//!     since MAX_SRC grew to 4 for flash_attn_ext).

use crate::tensor::{Context, GgmlOp, Storage, TensorId, TensorMeta};
use crate::types::{GgmlType, MAX_DIMS};
use half::f16;

/// op_params[1] flag distinguishing rms_norm from plain norm while tensor.rs's
/// GgmlOp lacks an RmsNorm variant.
pub const OP_FLAG_NORM_IS_RMS: i32 = 1;
/// op_params[1] flag marking a Norm node as ggml_l2_norm (ggml.c:3260) —
/// clef's decision head (a7b94df2c clef.cpp:604/611-612)
pub const OP_FLAG_NORM_IS_L2: i32 = 2;
/// op_params[15] flag marking a GetRows node as GET_ROWS_BACK.
pub const OP_FLAG_GET_ROWS_BACK: i32 = 1;

/// ggml.h enum ggml_unary_op (subset)
pub const GGML_UNARY_OP_TANH: i32 = 4;
pub const GGML_UNARY_OP_SIGMOID: i32 = 7;
/// ggml.h:609 GGML_UNARY_OP_NEG — the 3rd entry of the unary enum (ABS=0,
/// SGN=1, NEG=2); kernel unary-ops.cpp:11 `op_neg` → `-x` (vec.h:844
/// ggml_vec_neg_f32 is the scalar loop). The chunked delta-net's decay
/// differences `-g_cs` / negated attn before the triangular solve
/// (delta-net-base.cpp:167/189).
pub const GGML_UNARY_OP_NEG: i32 = 2;
/// ggml.h:613 GGML_UNARY_OP_ELU — the 6th entry of the unary enum; kernel
/// `(x > 0) ? x : expm1f(x)` (vec.h:915 `ggml_vec_elu_f32`). The SEANet
/// conv stack of pocket-tts uses it between every conv (pockettts-seanet.cpp:104).
pub const GGML_UNARY_OP_ELU: i32 = 5;
pub const GGML_UNARY_OP_GELU: i32 = 8;
pub const GGML_UNARY_OP_SILU: i32 = 10;
/// ggml.h:614 GGML_UNARY_OP_RELU — the 7th entry of the unary enum; the
/// conformer-family conv stems (models/conformer.cpp:26). Kernel: vec.h:922
/// `ggml_vec_relu_f32` — a plain `(x > 0) ? x : 0` loop (no SIMD variant).
pub const GGML_UNARY_OP_RELU: i32 = 6;
/// arch batch 10 (minimax-01): GGML_UNARY_OP_EXP — ggml.h:621, the 13th of
/// the enum (ABS..TRUNC); consumed by the lightning-attention decays
/// (`ggml_exp`, ggml.c:2892).
pub const GGML_UNARY_OP_EXP: i32 = 13;
pub const GGML_UNARY_OP_SOFTPLUS: i32 = 15;
/// arch batch 10 (graniteswitch): GGML_UNARY_OP_ROUND — ggml.h:627, the
/// router lane's float→slot rounding (`ggml_round`, ggml.c:2952).
pub const GGML_UNARY_OP_ROUND: i32 = 20;
/// ggml.h:624 GGML_UNARY_OP_GELU_ERF — the 17th entry of the unary enum
/// (ABS..TRUNC), the erf-form GELU of the whisper-enc/clip audio graphs
/// (`ggml_gelu_erf`, ggml.c:2796).
pub const GGML_UNARY_OP_GELU_ERF: i32 = 16;
/// arch batch 11a (apertus): GGML_UNARY_OP_XIELU — ggml.h:625, the 17th of
/// the enum; the FFN activation of Apertus (`ggml_xielu`, ggml.c:2838). Not a
/// plain unary: op_params[0] carries the op id, [1..5] the folded f32
/// constants (see [`Context::xielu`]).
pub const GGML_UNARY_OP_XIELU: i32 = 17;
/// audio round 5 (TTS generators): ggml.h:611 GGML_UNARY_OP_STEP — the 4th
/// entry of the unary enum; kernel vec.h:903 `ggml_vec_step_f32` =
/// `(x > 0.f) ? 1.f : 0.f`. The on-graph sampling masks of qwen3tts's
/// code_gen and the banded causal masks of both generators
/// (qwen3tts-gen.cpp:36/:57/:474, pockettts-gen.cpp:223).
pub const GGML_UNARY_OP_STEP: i32 = 3;
/// ggml.h:609 GGML_UNARY_OP_ABS — the 1st entry of the unary enum; kernel
/// unary-ops.cpp:3 `op_abs` → `fabsf(x)`. The PLE gate's magnitude
/// (`ggml_abs`, qwen4exp.cpp:1246).
pub const GGML_UNARY_OP_ABS: i32 = 0;
/// ggml.h:610 GGML_UNARY_OP_SGN — the 2nd entry of the unary enum; kernel
/// unary-ops.cpp:7 `op_sgn` → `(x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f)`.
/// The PLE gate's signed sqrt (`ggml_sgn`, qwen4exp.cpp:1247).
pub const GGML_UNARY_OP_SGN: i32 = 1;

/// ggml.h:671-674 enum ggml_tri_type (kept in the reference's own order)
pub const GGML_TRI_TYPE_UPPER_DIAG: i32 = 0;
pub const GGML_TRI_TYPE_UPPER: i32 = 1;
pub const GGML_TRI_TYPE_LOWER_DIAG: i32 = 2;
pub const GGML_TRI_TYPE_LOWER: i32 = 3;

/// ggml.h sort order
pub const GGML_SORT_ORDER_ASC: i32 = 0;
pub const GGML_SORT_ORDER_DESC: i32 = 1;

/// ggml.h enum ggml_glu_op
pub const GGML_GLU_OP_REGLU: i32 = 0;
pub const GGML_GLU_OP_GEGLU: i32 = 1;
pub const GGML_GLU_OP_SWIGLU: i32 = 2;
pub const GGML_GLU_OP_SWIGLU_OAI: i32 = 3;
pub const GGML_GLU_OP_GEGLU_ERF: i32 = 4;
pub const GGML_GLU_OP_GEGLU_QUICK: i32 = 5;
/// arch batch 7 (deepseek4): GGML_GLU_OP_SWIGLU_CLAMP (ggml.h:641) — the
/// swiglu_clamp variant of the expert/shared FFNs
/// (`ggml_swiglu_clamp`, ggml.c:3123-3132).
pub const GGML_GLU_OP_SWIGLU_CLAMP: i32 = 6;

/// ggml.h enum ggml_scale_mode — low byte of op_params[0] of GGML_OP_UPSCALE
/// (the node `ggml_interpolate` builds)
pub const GGML_SCALE_MODE_NEAREST: u32 = 0;
pub const GGML_SCALE_MODE_BILINEAR: u32 = 1;
pub const GGML_SCALE_MODE_BICUBIC: u32 = 2;

/// ggml.h ggml_scale_flag bits — high byte of the same op_params[0]
pub const GGML_SCALE_FLAG_ALIGN_CORNERS: u32 = 0x100;
pub const GGML_SCALE_FLAG_ANTIALIAS: u32 = 0x200;

/// ggml.h:456 GGML_HINT_SRC0_IS_HADAMARD — rides a MUL_MAT node's
/// op_params[1] (ggml_mul_mat_set_hint, ggml.c:3368). The CPU kernel
/// dispatches the hinted gemm to the fast Walsh-Hadamard transform instead
/// (ggml-cpu.c:1263-1266 → ops.cpp:12027), whose butterfly accumulation
/// order differs from the gemm dot — deepseek4's `llama_mul_mat_hadamard`
/// (llama-impl.h:70) always sets it.
pub const GGML_HINT_SRC0_IS_HADAMARD: i32 = 1;

/// ggml.h rope types
pub const GGML_ROPE_TYPE_NORMAL: i32 = 0;
pub const GGML_ROPE_TYPE_NEOX: i32 = 2;
pub const GGML_ROPE_TYPE_MROPE: i32 = 8;
pub const GGML_ROPE_TYPE_VISION: i32 = 24;
pub const GGML_ROPE_TYPE_IMROPE: i32 = 40;

pub const GGML_MROPE_SECTIONS: usize = 4;

// ======================================================================
// ggml_v_expf — scalar port of the AVX2 polynomial (vec.h:1215).
// Used by softmax and silu so their numerics match the C build.
// ======================================================================
/// `ggml_v_expf` — AVX512 variant (vec.h:1172). Differs from the AVX2 one:
/// polynomial tail is `c5*b + 1.0` and scaling uses exact `scalef` semantics
/// (`j * 2^n`, one rounding), overflow path blends INF/0 for |n| > 192.
/// The reference binary runs the AVX512 paths — hot kernels use this one.
#[inline]
pub fn ggml_expf_v512(x: f32) -> f32 {
    const R: f32 = 12582912.0; // 0x1.8p23 = 1.5 * 2^23
    const LOG2E: f32 = f32::from_bits(0x3FB8AA3B); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3F317200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35BFBE8E); // 0x1.7f7d1cp-20
    const C1: f32 = f32::from_bits(0x3C072010); // 0x1.0e4020p-7
    const C2: f32 = f32::from_bits(0x3D2B9F17); // 0x1.573e2ep-5
    const C3: f32 = f32::from_bits(0x3E2AAF33); // 0x1.555e66p-3
    const C4: f32 = f32::from_bits(0x3EFFFEDB); // 0x1.fffdb6p-2
    const C5: f32 = f32::from_bits(0x3F7FFFF6); // 0x1.ffffecp-1

    let z = x.mul_add(LOG2E, R);
    let n = z - R;
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let d = n.abs() > 192.0;
    if !d {
        let u = b * b;
        let mid = (C1.mul_add(b, C2)).mul_add(u, C3.mul_add(b, C4));
        let j = mid.mul_add(u, C5.mul_add(b, 1.0));
        // scalef: j * 2^n with a single f32 rounding — `j` carries 24 bits
        // and 2^n is a power of two, so the f64 product is exact and the
        // cast rounds once (subnormal results included), exactly like
        // `_mm512_scalef_ps` (VSCALEFPS). |n| <= 192 here, so 2^n is a
        // normal f64.
        return ((j as f64) * f64::from_bits(((n as i32 + 1023) as u64) << 52)) as f32;
    }
    if n <= 0.0 {
        0.0
    } else {
        f32::INFINITY
    }
}

pub fn ggml_expf(x: f32) -> f32 {
    // exact bit patterns of the C hex-float constants (Rust has no 0x1.xp±y literals)
    const R: f32 = 12582912.0; // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3F31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35BF_BE8E); // 0x1.7f7d1cp-20
    const C1: f32 = f32::from_bits(0x3C07_2010); // 0x1.0e4020p-7
    const C2: f32 = f32::from_bits(0x3D2B_9F17); // 0x1.573e2ep-5
    const C3: f32 = f32::from_bits(0x3E2A_AF33); // 0x1.555e66p-3
    const C4: f32 = f32::from_bits(0x3EFF_FEDB); // 0x1.fffdb6p-2
    const C5: f32 = f32::from_bits(0x3F7F_FFF6); // 0x1.ffffecp-1

    let z = x.mul_add(LOG2E, R);
    let n = z - R;
    // fnmadd(n, ln2_hi, x) = x - n*ln2_hi with a single rounding
    let b = (-LN2_HI).mul_add(n, x);
    let b = (-LN2_LO).mul_add(n, b);
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let c = n.abs() > 126.0;
    let u = b * b;
    let m = C1.mul_add(b, C2).mul_add(u, C3.mul_add(b, C4));
    let j = m.mul_add(u, C5 * b);
    if !c {
        return j.mul_add(k, k);
    }
    // overflow / underflow branch
    let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    let d = n.abs() > 192.0;
    if d {
        s1 * s1
    } else if c {
        s2.mul_add(j, s2) * s1
    } else {
        k.mul_add(j, k)
    }
}

/// The AVX512 `ggml_v_silu` lane form — `x / (1 + v_expf(-x))` with the
/// vector polynomial exp. This is what the reference binary's 16-wide chunks
/// compute (`ggml_vec_silu_f32` / `ggml_vec_swiglu_f32`, vec.cpp:380/:417),
/// NOT the C scalar tail — see [`ggml_silu_scalar_f32`].
#[inline]
pub fn ggml_silu_f32(x: f32) -> f32 {
    // reference binary runs the AVX512 silu (x / (1 + ggml_v_expf(-x)))
    x / (1.0 + ggml_expf_v512(-x))
}

/// vec.h:1046 `ggml_silu_f32` as the reference binary compiles it for the
/// scalar tails: `x/(1 + expf(-x))` with the SYSTEM libm expf (no
/// simd-mappings remap). The v512 polynomial above differs from libm `expf`
/// by 1 ulp on ~a quarter of inputs, so the tails are bit-visible whenever a
/// row length is not a multiple of 16 (the batch-15 MoE `ffn_moe_swiglu`
/// rows, n_ff_exp 24).
#[inline]
pub fn ggml_silu_scalar_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 对照 ggml_gelu_f32 (vec.h:968) — the *polynomial*. Its only consumer is the
/// f16 table build (ggml-cpu.c:3891); the GELU op itself reads the table.
///
/// `1.0f + GELU_COEF_A*x*x` is contracted into an FMA: that is what the
/// reference build does when it compiles the (vectorisable) table-init loop
/// with -O3 -march=native, and it is bit-visible — index 0xbfff differs by one
/// f16 ulp otherwise (verified: the explicit-FMA form reproduces all 65536
/// reference table entries, the plain mul/add form only 65535).
#[inline]
pub fn ggml_gelu_f32(x: f32) -> f32 {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (GELU_COEF_A * x).mul_add(x, 1.0)).tanh())
}

/// `ggml_table_gelu_f16` (vec.cpp:6, filled in ggml_cpu_init at ggml-cpu.c:3891):
/// `table[i] = fp16(gelu_poly(fp16_to_fp32(i)))` for all 65536 f16 bit
/// patterns. Built once, lazily, exactly like the C init.
///
/// Both conversions are the *scalar* ggml_compute_* ones, because that is what
/// the reference build resolves to: simd-mappings.h's `__F16C__` branch only
/// defines `GGML_CPU_COMPUTE_{FP16,FP32}_TO_*` (line 106055 in the
/// preprocessed ggml-cpu.c), so the plain `GGML_CPU_FP32_TO_FP16` falls
/// through to `GGML_COMPUTE_FP32_TO_FP16` (simd-mappings.h:157). The scalar
/// converter canonicalises NaN to 0x7E00 where `half::f16::from_f32` keeps the
/// payload — that is the only difference, and it is bit-visible in 2045 of the
/// 65536 table entries.
pub fn ggml_table_gelu_f16() -> &'static [u16; 1 << 16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Box<[u16; 1 << 16]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0u16; 1 << 16].into_boxed_slice();
        for (i, slot) in t.iter_mut().enumerate() {
            let f = ggml_compute_fp16_to_fp32(i as u16);
            *slot = ggml_compute_fp32_to_fp16(ggml_gelu_f32(f));
        }
        t.try_into().ok().expect("65536 entries")
    })
}

/// 对照 ggml-impl.h:395 `ggml_compute_fp16_to_fp32` (= GGML_FP16_TO_FP32, and
/// the values stored in ggml_table_f32_f16 by ggml-cpu.c:3889). Exact bit
/// trick port; note it *preserves* the NaN payload.
#[inline]
pub fn ggml_compute_fp16_to_fp32(h: u16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);

    let exp_offset: u32 = 0xE0 << 23;
    let exp_scale = f32::from_bits(0x0780_0000); // 0x1.0p-112
    let normalized_value = f32::from_bits((two_w >> 4).wrapping_add(exp_offset)) * exp_scale;

    let magic_mask: u32 = 126 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;

    let denormalized_cutoff: u32 = 1 << 27;
    let result = sign
        | if two_w < denormalized_cutoff {
            denormalized_value.to_bits()
        } else {
            normalized_value.to_bits()
        };
    f32::from_bits(result)
}

/// 对照 ggml-impl.h:420 `ggml_compute_fp32_to_fp16` (= GGML_FP32_TO_FP16, and
/// what GGML_CPU_FP32_TO_FP16 expands to in this build). NaN/overflow collapse
/// to 0x7E00; denormals go through the 2^±112 scaling trick.
#[inline]
pub fn ggml_compute_fp32_to_fp16(f: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x7780_0000); // 0x1.0p+112
    let scale_to_zero = f32::from_bits(0x0880_0000); // 0x1.0p-110
    let mut base = (f.abs() * scale_to_inf) * scale_to_zero;

    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let mut bias = shl1_w & 0xFF00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }

    base = f32::from_bits((bias >> 1).wrapping_add(0x0780_0000)) + base;
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x0000_7C00;
    let mantissa_bits = bits & 0x0000_0FFF;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | if shl1_w > 0xFF00_0000 { 0x7E00 } else { nonsign }) as u16
}

/// 对照 ggml_vec_gelu_f32 (vec.h:987) — the path the reference actually runs:
/// `#define GGML_GELU_FP16` is unconditional (vec.h:46), so F32 GELU goes
/// through the f16 lookup table with a |x| <= 10 shortcut. The output is
/// f16-quantised, so it differs from the polynomial by ~1e-3 relative.
#[inline]
pub fn ggml_vec_gelu_f32(x: f32) -> f32 {
    if x <= -10.0 {
        0.0
    } else if x >= 10.0 {
        x
    } else {
        // vec.h:998-999: t = GGML_CPU_FP32_TO_FP16(x[i]) (scalar here),
        // y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]) (f32 table)
        let t = ggml_compute_fp32_to_fp16(x) as usize;
        ggml_compute_fp16_to_fp32(ggml_table_gelu_f16()[t])
    }
}

/// 对照 ggml_vec_gelu_f16 (vec.h:973) — direct bit-pattern table lookup.
#[inline]
pub fn ggml_vec_gelu_f16(x: f16) -> f16 {
    f16::from_bits(ggml_table_gelu_f16()[x.to_bits() as usize])
}

// ---- BF16 vec helpers (vec.h sync batch D: ggml_vec_*_bf16, the BF16
// ---- unary/GLU dispatch added in ops.cpp) ----
// All five are plain scalar loops over the f32 functors with
// GGML_BF16_TO_FP32 in / GGML_FP32_TO_BF16 out (vec.h:979-1052); the half
// crate's round-to-nearest-even bf16 conversion is the port's established
// stand-in for GGML_FP32_TO_BF16 (verified on the wdata conversion path).

/// 对照 ggml_vec_gelu_bf16 (vec.h:979): the *polynomial* gelu (not the f16
/// table the F32 path uses — the bf16 loop calls `ggml_gelu_f32` directly).
#[inline]
pub fn ggml_vec_gelu_bf16(x: half::bf16) -> half::bf16 {
    half::bf16::from_f32(ggml_gelu_f32(x.to_f32()))
}

/// 对照 ggml_vec_gelu_erf_bf16 (vec.h:993).
#[inline]
pub fn ggml_vec_gelu_erf_bf16(x: half::bf16) -> half::bf16 {
    let xi = x.to_f32();
    half::bf16::from_f32(0.5 * xi * (1.0 + unsafe { erff(xi * SQRT_2_INV) }))
}

/// 对照 ggml_vec_silu_bf16 (vec.h:1397): libm-expf silu scalar.
#[inline]
pub fn ggml_vec_silu_bf16(x: half::bf16) -> half::bf16 {
    half::bf16::from_f32(ggml_silu_scalar_f32(x.to_f32()))
}

/// 对照 ggml_vec_reglu_bf16 (vec.h:1439).
#[inline]
pub fn ggml_vec_reglu_bf16(x: half::bf16, g: half::bf16) -> half::bf16 {
    let v = x.to_f32();
    half::bf16::from_f32(if v > 0.0 { v * g.to_f32() } else { 0.0 })
}

/// 对照 ggml_vec_geglu_bf16 (vec.h:1477).
#[inline]
pub fn ggml_vec_geglu_bf16(x: half::bf16, g: half::bf16) -> half::bf16 {
    half::bf16::from_f32(ggml_gelu_f32(x.to_f32()) * g.to_f32())
}

/// 对照 ggml_vec_swiglu_bf16 (vec.h:1493).
#[inline]
pub fn ggml_vec_swiglu_bf16(x: half::bf16, g: half::bf16) -> half::bf16 {
    let xi = x.to_f32();
    let gi = g.to_f32();
    half::bf16::from_f32((xi / (1.0 + (-xi).exp())) * gi)
}

/// glibc `erff` — the same libm symbol the reference build's
/// `ggml_vec_gelu_erf_f32` calls (vec.h:1010-1015 is a plain scalar loop;
/// no SIMD `ggml_v_gelu_erf` exists at this commit, and erf is not in
/// libmvec's default -O3 auto-vectorization set — the dump's kind-3 raw
/// section proves the op == raw erff). Rust std has no `erf`, so the port
/// binds the libc symbol directly: values are bit-identical by construction
/// and no new dependency is introduced (std already links libm on Linux).
#[link(name = "m")]
extern "C" {
    fn erff(x: f32) -> f32;
}

/// `SQRT_2_INV` (vec.h:967): `0.70710678118654752440084436210484f`.
const SQRT_2_INV: f32 = 0.707_106_781_186_547_524_400_844_362_104_84f32;

/// 对照 ggml_vec_gelu_erf_f32 (vec.h:1010): `0.5f*x*(1.0f + erff(x*SQRT_2_INV))`
/// — all plain f32 multiplies and one add off the erff result, so GCC's
/// default -ffp-contract cannot fuse anything; the port reproduces the exact
/// evaluation.
#[inline]
pub fn ggml_vec_gelu_erf_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + unsafe { erff(x * SQRT_2_INV) })
}

/// 对照 ggml_vec_gelu_erf_f16 (vec.h:979):
/// `res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV)); y = FP32_TO_FP16(res)` with
/// `xi = GGML_CPU_FP16_TO_FP32(x)` — the scalar conversion on the way out.
#[inline]
pub fn ggml_vec_gelu_erf_f16(x: f16) -> f16 {
    let xi = x.to_f32();
    f16::from_bits(ggml_compute_fp32_to_fp16(ggml_vec_gelu_erf_f32(xi)))
}

/// 对照 op_tanh (unary-ops.cpp:19) == `tanhf`. The reference has NO SIMD tanh
/// at this commit: ggml_vec_tanh_f32 is a plain scalar loop (vec.h:909) and
/// unary_op<op_tanh> calls tanhf directly (unary-ops.cpp:253). Verified
/// bit-identical to the dumped op output (parity/tanh_ref.bin kinds 0 vs 3).
#[inline]
pub fn ggml_tanh_f32(x: f32) -> f32 {
    x.tanh()
}

// ======================================================================
// shared builder helpers (ggml.c internals)
// ======================================================================

/// tensor.rs's `TensorMeta::type_aware_strides` is module-private; local copy
/// (ggml.c ggml_new_tensor_impl stride computation).
fn type_aware_strides(ne: &[i64; MAX_DIMS], ty: GgmlType) -> [u64; MAX_DIMS] {
    let nb0 = ty.type_size() as u64;
    let blck = ty.blck_size() as i64;
    let nb1 = nb0 * (ne[0] / blck) as u64;
    let nb2 = nb1 * ne[1] as u64;
    let nb3 = nb2 * ne[2] as u64;
    [nb0, nb1, nb2, nb3]
}

/// ggml_can_repeat (ggml.c:1589): can t1 be represented as a repetition of t0?
pub fn can_repeat(ctx: &Context, t0: TensorId, t1: TensorId) -> bool {
    let a = &ctx.tensors[t0.0 as usize];
    let b = &ctx.tensors[t1.0 as usize];
    let empty = |t: &TensorMeta| t.ne.iter().all(|&n| n == 0);
    if empty(a) {
        return empty(b);
    }
    (0..MAX_DIMS).all(|i| b.ne[i] % a.ne[i] == 0)
}

/// ggml_can_mul_mat (ggml.c:3333)
pub fn can_mul_mat(ctx: &Context, t0: TensorId, t1: TensorId) -> bool {
    let a = &ctx.tensors[t0.0 as usize];
    let b = &ctx.tensors[t1.0 as usize];
    a.ne[0] == b.ne[0] && b.ne[2] % a.ne[2] == 0 && b.ne[3] % a.ne[3] == 0
}

/// ggml_is_transposed: ggml.c ggml_is_permuted-style check t->nb[0] > t->nb[1]
pub fn is_transposed(ctx: &Context, t: TensorId) -> bool {
    let t = &ctx.tensors[t.0 as usize];
    t.nb[0] > t.nb[1]
}

impl Context {
    /// ggml_dup_tensor (ggml.c:1895): fresh tensor with same type & shape.
    fn dup_tensor_of(&mut self, a: TensorId) -> TensorId {
        let (ty, ne) = {
            let t = &self.tensors[a.0 as usize];
            (t.ty, t.ne)
        };
        self.new_tensor(ty, ne)
    }

    /// ggml_view_tensor (ggml.c:1963): alias of `a` with identical ne/nb.
    fn view_tensor_of(&mut self, a: TensorId) -> TensorId {
        let (ty, ne, nb, name) = {
            let t = &self.tensors[a.0 as usize];
            (t.ty, t.ne, t.nb, t.name.clone())
        };
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorMeta {
            ty,
            ne,
            nb,
            op: GgmlOp::None,
            op_params: [0; crate::types::MAX_OP_PARAMS / 4],
            src: [None; crate::types::MAX_SRC],
            view_src: Some(a),
            view_offs: 0,
            storage: Storage::None, // resolved through the view chain
            name: format!("{name} (view)"),
            flags: 0,
        });
        id
    }

    /// ggml_new_tensor_impl(..., base=a, offset): view with a new shape.
    fn view_of(&mut self, a: TensorId, ne: [i64; MAX_DIMS], offset: usize) -> TensorId {
        let ty = self.tensors[a.0 as usize].ty;
        let nb = type_aware_strides(&ne, ty);
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorMeta {
            ty,
            ne,
            nb,
            op: GgmlOp::None,
            op_params: [0; crate::types::MAX_OP_PARAMS / 4],
            src: [None; crate::types::MAX_SRC],
            view_src: Some(a),
            view_offs: offset,
            storage: Storage::None,
            name: String::new(),
            flags: 0,
        });
        id
    }

    /// Wire `op` + `src` into a tensor. Const-generic over the array length so ops
    /// with more sources than the legacy 3-slot literals (e.g. flash_attn_ext's
    /// q/k/v/mask, once MAX_SRC grows) build without touching existing call sites.
    fn init_op<const N: usize>(&mut self, id: TensorId, op: GgmlOp, src: [Option<TensorId>; N]) -> TensorId {
        let t = &mut self.tensors[id.0 as usize];
        t.op = op;
        t.src = [None; crate::types::MAX_SRC];
        for (i, s) in src.into_iter().enumerate().take(crate::types::MAX_SRC) {
            t.src[i] = s;
        }
        id
    }

    fn params_f32(&mut self, id: TensorId, vals: &[f32]) {
        let bits: Vec<i32> = vals.iter().map(|v| v.to_bits() as i32).collect();
        self.set_op_params_i32(id, &bits);
    }

    // ===================== dup / add / mul =====================

    /// 对照 ggml_dup_impl / ggml_dup (ggml.c:2032)
    pub fn dup(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Dup, [Some(a), None, None])
    }
    /// 对照 ggml_dup_inplace (ggml.c:2050)
    pub fn dup_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Dup, [Some(a), None, None])
    }

    /// 对照 ggml_add_impl / ggml_add (ggml.c:2058)
    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(can_repeat(self, b, a), "ggml_add: !can_repeat");
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Add, [Some(a), Some(b), None])
    }
    /// 对照 ggml_add_inplace (ggml.c:2081)
    pub fn add_inplace(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(can_repeat(self, b, a), "ggml_add_inplace: !can_repeat");
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Add, [Some(a), Some(b), None])
    }

    /// 对照 ggml_mul_impl / ggml_mul (ggml.c:2259)
    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(
            can_repeat(self, b, a),
            "ggml_mul: !can_repeat: a.ne {:?} b.ne {:?}",
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne
        );
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Mul, [Some(a), Some(b), None])
    }
    /// 对照 ggml_mul_inplace (ggml.c:2282)
    pub fn mul_inplace(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(can_repeat(self, b, a), "ggml_mul_inplace: !can_repeat");
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Mul, [Some(a), Some(b), None])
    }

    /// 对照 ggml_div_impl / ggml_div (ggml.c:2291) — elementwise a / b with
    /// repeat-broadcast of `b`. Used by the MoE weight normalization
    /// (llama-graph.cpp:2144).
    pub fn div(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(
            can_repeat(self, b, a),
            "ggml_div: !can_repeat: a.ne {:?} b.ne {:?}",
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne
        );
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Div, [Some(a), Some(b), None])
    }

    /// 对照 ggml_sub_impl / ggml_sub (ggml.c:2225-2246) — elementwise a - b
    /// with repeat-broadcast of `b` (binary-ops.cpp:144
    /// `ggml_compute_forward_sub` → `op_sub`, vec.h:112 `z[i] = x[i] - y[i]`).
    /// Audio round 4: parakeet's folded batch-norm centers with it
    /// (parakeet.cpp:377), mimo's RVQ residual loop (mimo-audio.cpp:103) and
    /// the qwen3tts speaker encoder's statistics pooling (qwen3tts-spkenc.cpp:110).
    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(
            can_repeat(self, b, a),
            "ggml_sub: !can_repeat: a.ne {:?} b.ne {:?}",
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne
        );
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Sub, [Some(a), Some(b), None])
    }

    /// 对照 ggml_clamp_impl / ggml_clamp (ggml.c:4110) — op_params[0]/[1] hold
    /// the f32 bits of min/max; compute is `MAX(MIN(x, max), min)`
    /// (ops.cpp:5851).
    pub fn clamp(&mut self, a: TensorId, min: f32, max: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.params_f32(r, &[min, max]);
        self.init_op(r, GgmlOp::Clamp, [Some(a), None, None])
    }
    /// 对照 ggml_clamp_inplace (ggml.c:4135)
    pub fn clamp_inplace(&mut self, a: TensorId, min: f32, max: f32) -> TensorId {
        let r = self.view_tensor_of(a);
        self.params_f32(r, &[min, max]);
        self.init_op(r, GgmlOp::Clamp, [Some(a), None, None])
    }

    /// 对照 ggml_set_rows (ggml.c:4000): scatter rows of `b` into `a` at row
    /// indices `c` (I64/I32). src slots follow the C legacy order:
    /// src[0]=b (rows), src[1]=c (idx), src[2]=a (dst).
    pub fn set_rows(&mut self, a: TensorId, b: TensorId, c: TensorId) -> TensorId {
        let ta = &self.tensors[a.0 as usize];
        let tb = &self.tensors[b.0 as usize];
        let tc = &self.tensors[c.0 as usize];
        assert_eq!(ta.ne[0], tb.ne[0], "set_rows: ne0");
        assert_eq!(ta.ne[2], tb.ne[2], "set_rows: ne2");
        assert_eq!(ta.ne[3], tb.ne[3], "set_rows: ne3");
        assert_eq!(tb.ne[1], tc.ne[0], "set_rows: b.ne1 == c.ne0");
        assert_eq!(tb.ne[2] % tc.ne[1], 0, "set_rows: ne2 % c.ne1");
        assert_eq!(tb.ne[3] % tc.ne[2], 0, "set_rows: ne3 % c.ne2");
        assert_eq!(tc.ne[3], 1, "set_rows: c.ne3 == 1");
        assert!(
            matches!(tb.ty, GgmlType::F32 | GgmlType::F16),
            "set_rows: src rows must be F32/F16, got {:?}",
            tb.ty
        );
        assert!(
            matches!(tc.ty, GgmlType::I64 | GgmlType::I32),
            "set_rows: idx must be I64/I32, got {:?}",
            tc.ty
        );
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::SetRows, [Some(b), Some(c), Some(a)])
    }

    // ===================== mul_mat =====================

    /// 对照 ggml_mul_mat (ggml.c:3341)
    // ===================== fill / lightning indexer (DSA, batch 6) =========

    /// 对照 ggml_fill (ggml.c:5407, via ggml_fill_impl :5389): dst is a dup
    /// of `a` (F32/F16, contiguous required like C) whose every element the
    /// kernel sets to `c`.
    pub fn fill(&mut self, a: TensorId, c: f32) -> TensorId {
        assert!(
            matches!(self.ty(a), GgmlType::F32 | GgmlType::F16),
            "ggml_fill: a must be F32/F16"
        );
        assert!(is_contiguous_ctx(self, a), "ggml_fill: a must be contiguous");
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Fill, [Some(a), None, None]);
        self.params_f32(r, &[c]);
        r
    }

    /// 对照 ggml_lightning_indexer (ggml.c:6423-6451) — the DeepSeek DSA
    /// fused indexer score. Shapes (C asserts, ggml.c:6429-6445):
    ///   q  [n_embd, n_head, n_tokens, n_stream] F32
    ///   k  [n_embd, 1, n_kv, n_stream] (the lid F16/F32 cache view)
    ///   w  [n_head, n_tokens, 1, n_stream] F32 (pre-scaled indexer weights)
    ///   m  [n_kv, n_tokens, mask_ne2, n_stream-divisor] F16
    /// dst [n_kv, n_tokens, 1, n_stream] F32.
    pub fn lightning_indexer(
        &mut self,
        q: TensorId,
        k: TensorId,
        weights: TensorId,
        mask: TensorId,
    ) -> TensorId {
        let (qne, kne, wne, mne, qty, wty, mty) = {
            let (qt, kt, wt, mt) = (
                &self.tensors[q.0 as usize],
                &self.tensors[k.0 as usize],
                &self.tensors[weights.0 as usize],
                &self.tensors[mask.0 as usize],
            );
            (qt.ne, kt.ne, wt.ne, mt.ne, qt.ty, wt.ty, mt.ty)
        };
        assert_eq!(qty, GgmlType::F32, "lightning_indexer: q->type == F32");
        assert_eq!(wty, GgmlType::F32, "lightning_indexer: weights->type == F32");
        assert_eq!(mty, GgmlType::F16, "lightning_indexer: mask->type == F16");
        assert_eq!(qne[0], kne[0], "lightning_indexer: q->ne[0] == k->ne[0]");
        assert_eq!(mne[0], kne[2], "lightning_indexer: mask->ne[0] == k->ne[2]");
        assert_eq!(qne[1], wne[0], "lightning_indexer: q->ne[1] == weights->ne[0]");
        assert_eq!(kne[1], 1, "lightning_indexer: k->ne[1] == 1");
        assert_eq!(mne[1], qne[2], "lightning_indexer: mask->ne[1] == q->ne[2]");
        assert_eq!(qne[2], wne[1], "lightning_indexer: q->ne[2] == weights->ne[1]");
        assert_eq!(wne[2], 1, "lightning_indexer: weights->ne[2] == 1");
        assert_eq!(mne[2], 1, "lightning_indexer: mask->ne[2] == 1");
        assert_eq!(qne[3], kne[3], "lightning_indexer: q->ne[3] == k->ne[3]");
        assert_eq!(kne[3], wne[3], "lightning_indexer: k->ne[3] == weights->ne[3]");
        assert_eq!(wne[3] % mne[3], 0, "lightning_indexer: weights->ne[3] % mask->ne[3]");

        let r = self.new_tensor(GgmlType::F32, [kne[2], qne[2], 1, qne[3]]);
        self.init_op(r, GgmlOp::LightningIndexer, [Some(q), Some(k), Some(weights), Some(mask)]);
        r
    }

    /// 对照 ggml_top_k (ggml.c:5459-5470): I32 `[k, ne1, ne2, ne3]` of the k
    /// largest rows of `a` (per row of dims 1..3). NOT the same tie behavior
    /// as [`Context::argsort_top_k`] — the kernel ports the reference's
    /// `std::partial_sort` heap-select (ops.cpp:8550-8604) exactly, which is
    /// what the deepseek32 DSA mask-select consumes.
    pub fn top_k(&mut self, a: TensorId, k: i32) -> TensorId {
        let ane = self.tensors[a.0 as usize].ne;
        assert!(ane[0] >= k as i64, "ggml_top_k: a->ne[0] >= k");
        let r =
            self.new_tensor(GgmlType::I32, [k as i64, ane[1], ane[2], ane[3]]);
        self.init_op(r, GgmlOp::TopK, [Some(a), None, None]);
        r
    }

    // ===================== sqrt / rope-back / swiglu-clamp (arch batch 7, dsv4) =====

    /// 对照 ggml_sqrt (ggml.c:2347-2364, GGML_OP_SQRT): per-element `sqrtf`
    /// (unary-ops.cpp:51 op_sqrt). Consumed by the deepseek4 SQRT_SOFTPLUS
    /// MoE gating (`sqrt(softplus(logits))`, llama-graph.cpp:2053).
    pub fn sqrt(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Sqrt, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_sin (ggml.c:2423-2441, GGML_OP_SIN): per-element `sinf`
    /// (unary-ops.cpp:289 `op_sin` → vec.h:878 `ggml_vec_sin_f32`, a plain
    /// scalar libm loop). Audio round 4: parakeet builds the sinusoidal RPE
    /// in-graph (parakeet.cpp:104).
    pub fn sin(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Sin, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_cos (ggml.c:2449-2467, GGML_OP_COS): per-element `cosf`
    /// (unary-ops.cpp:293 `op_cos` → vec.h:884, scalar). parakeet.cpp:105.
    pub fn cos(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Cos, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_sqr (ggml.c:2321-2339, GGML_OP_SQR): per-element `x*x`
    /// (unary-ops.cpp:281 `op_sqr` → vec.h:859 `y[i] = x[i]*x[i]`, scalar).
    /// mimo's RVQ codebook norms (mimo-audio.cpp:93) and the qwen3tts
    /// speaker variance (qwen3tts-spkenc.cpp:111).
    pub fn sqr(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Sqr, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_mean (ggml.c:2521-2531, GGML_OP_MEAN): row-wise mean over
    /// ne[0] — `ggml_vec_sum_f32` (double accumulator) then `/= ne00`
    /// (ops.cpp:1513-1543), single-threaded (ith==0 only, ggml-cpu.c). The
    /// qwen3tts speaker encoder's temporal means (qwen3tts-spkenc.cpp:76).
    pub fn mean(&mut self, a: TensorId) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        let r = self.new_tensor(GgmlType::F32, [1, ne[1], ne[2], ne[3]]);
        self.init_op(r, GgmlOp::Mean, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_pad_reflect_1d (ggml.c:5288-5315, GGML_OP_PAD_REFLECT_1D):
    /// reflect-pad dim0 by p0 (left) / p1 (right) — `left[-i] = left[i]`,
    /// `right[i] = right[-i]` (ops.cpp:8282-8318). F32 + contiguous input
    /// only. The qwen3tts speaker encoder's "same" convs (qwen3tts-spkenc.cpp:17).
    pub fn pad_reflect_1d(&mut self, a: TensorId, p0: i32, p1: i32) -> TensorId {
        let t = &self.tensors[a.0 as usize];
        assert!((p0 as i64) < t.ne[0] && (p1 as i64) < t.ne[0], "pad_reflect_1d: pad < ne0");
        assert_eq!(t.ty, GgmlType::F32, "pad_reflect_1d: F32 only");
        assert!(is_contiguous_ctx(self, a), "pad_reflect_1d: contiguous input only");
        let r = self.new_tensor(
            GgmlType::F32,
            [t.ne[0] + p0 as i64 + p1 as i64, t.ne[1], t.ne[2], t.ne[3]],
        );
        self.set_op_params_i32(r, &[p0, p1]);
        self.init_op(r, GgmlOp::PadReflect1d, [Some(a), None, None]);
        r
    }

    // ===================== audio round 5 (TTS generators) =====================
    // qwen3tts-gen.cpp / pockettts-gen.cpp op surface: sum / cumsum / tri /
    // log / step / col2im_1d. Owner: agent GEN5.

    /// 对照 ggml_step (ggml.c:2695 → ggml_unary(GGML_UNARY_OP_STEP)):
    /// per-element `(x > 0.f) ? 1.f : 0.f` (vec.h:903, no SIMD variant).
    pub fn step(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_STEP]);
        r
    }

    /// 对照 ggml_log (ggml.c:2387, GGML_OP_LOG — its own op in the pinned
    /// enum, not UNARY): per-element `logf` (unary-ops.cpp:297
    /// `unary_op<op_log>` → vec.h:872).
    pub fn log(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Log, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_sum (ggml.c:2477, GGML_OP_SUM): reduce the whole tensor to
    /// one scalar of the same type — f32 rows accumulated in ggml_float
    /// (double) then summed in double (ops.cpp:1282-1310), ith==0 only.
    pub fn sum(&mut self, a: TensorId) -> TensorId {
        let ty = self.tensors[a.0 as usize].ty;
        let r = self.new_tensor_1d(ty, 1);
        self.init_op(r, GgmlOp::Sum, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_cumsum (ggml.c:2508, GGML_OP_CUMSUM): running sum along
    /// dim 0 — `y[i] = y[i-1] + x[i]` in f32 (vec.h:1507), F32 only.
    pub fn cumsum(&mut self, a: TensorId) -> TensorId {
        assert_eq!(
            self.tensors[a.0 as usize].ty,
            GgmlType::F32,
            "ggml_cumsum: F32 only"
        );
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Cumsum, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_tri (ggml.c:5368, GGML_OP_TRI): keep the half of each row
    /// the tri-type predicate passes over the row index, zero the rest
    /// (ops.cpp:2289). F32 + contiguous + square only.
    pub fn tri(&mut self, a: TensorId, tri_type: i32) -> TensorId {
        let t = &self.tensors[a.0 as usize];
        assert_eq!(t.ty, GgmlType::F32, "ggml_tri: F32 only");
        assert_eq!(t.ne[0], t.ne[1], "ggml_tri: ne0 == ne1");
        let r = self.dup_tensor_of(a);
        self.set_op_params_i32(r, &[tri_type]);
        self.init_op(r, GgmlOp::Tri, [Some(a), None, None]);
        r
    }

    // ===================== arch batch 18: the chunked delta-net ops ==========
    // GGML_OP_NEG rides the UNARY arm above; SET / DIAG / SOLVE_TRI are their
    // own nodes. Owner: agent GDN.

    /// 对照 ggml_neg (ggml.c:2681 → ggml_unary(GGML_UNARY_OP_NEG)):
    /// per-element `-x` (unary-ops.cpp:11 `op_neg`, scalar — no SIMD variant).
    pub fn neg(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_NEG]);
        r
    }

    /// 对照 ggml_abs (unary-ops.cpp:3 `op_abs` → :238): per-element
    /// `fabsf(x)`, F32 only. The PLE gate magnitude (qwen4exp.cpp:1246).
    pub fn abs(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_ABS]);
        r
    }

    /// 对照 ggml_sgn (unary-ops.cpp:7 `op_sgn` → :242): per-element
    /// `(x > 0) ? 1 : ((x < 0) ? -1 : 0)`, F32 only. The PLE gate's signed
    /// square root (qwen4exp.cpp:1247).
    pub fn sgn(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_SGN]);
        r
    }

    /// 对照 ggml_set_impl / ggml_set_inplace (ggml.c:3505-3542): dst is a
    /// view of `a` whose `b`-shaped region, viewed through the param strides
    /// at `offset`, is overwritten with `b`'s rows (F32, kernel
    /// ops.cpp:4769-4834 — n_tasks = 1, ggml-cpu.c:2379). `nb1/nb2/nb3` are
    /// the *view* strides of the destination region; nb0 is implicitly the
    /// element size because `a` must be contiguous.
    pub fn set_inplace(
        &mut self,
        a: TensorId,
        b: TensorId,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> TensorId {
        let ta = &self.tensors[a.0 as usize];
        let tb = &self.tensors[b.0 as usize];
        assert_eq!(ta.ty, GgmlType::F32, "ggml_set: F32 only");
        assert_eq!(tb.ty, GgmlType::F32, "ggml_set: F32 only");
        assert!(is_contiguous_ctx(self, a), "ggml_set: a must be contiguous");
        let numel = |t: &TensorMeta| t.ne.iter().map(|&n| n.max(1)).product::<i64>();
        assert!(
            numel(ta) >= numel(tb),
            "ggml_set: nelements(a) >= nelements(b)"
        );
        assert!(
            (nb1 | nb2 | nb3 | offset) < (1 << 30),
            "ggml_set: strides/offset fit i32 op_params"
        );
        let params = [nb1 as i32, nb2 as i32, nb3 as i32, offset as i32, 1];
        let r = self.view_tensor_of(a);
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Set, [Some(a), Some(b), None]);
        r
    }

    /// 对照 ggml_diag (ggml.c:4027, GGML_OP_DIAG): scatter the single row of
    /// `a` (ne1 == 1) onto the diagonal of an [n, n, ne2, ne3] square —
    /// `d[i][j] = (i == j) ? s[j] : 0`. F32 only (kernel ops.cpp:5434-5472,
    /// single-tasked). The chunked delta-net's `I + attn` identity
    /// (delta-net-base.cpp:162-165).
    pub fn diag(&mut self, a: TensorId) -> TensorId {
        let t = &self.tensors[a.0 as usize];
        assert_eq!(t.ne[1], 1, "ggml_diag: ne1 == 1");
        assert_eq!(t.ty, GgmlType::F32, "ggml_diag: F32 only");
        let ne = t.ne;
        let r = self.new_tensor(GgmlType::F32, [ne[0], ne[0], ne[2], ne[3]]);
        self.init_op(r, GgmlOp::Diag, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_solve_tri (ggml.c:6329, GGML_OP_SOLVE_TRI): dst = A⁻¹B by
    /// forward substitution where A is a square lower-triangular F32 matrix
    /// and B packs the right-hand sides column-wise. The reference supports
    /// exactly `left && lower && !uni` (asserted in the builder, ggml.c:6350)
    /// — the flags ride along for signature parity and are asserted here.
    /// Kernel ops.cpp:10824-10880: per solve column c,
    /// `X[i][c] = (B[i][c] - Σ_{t<i} A[i][t]·X[t][c]) / A[i][i]`.
    pub fn solve_tri(
        &mut self,
        a: TensorId,
        b: TensorId,
        left: bool,
        lower: bool,
        uni: bool,
    ) -> TensorId {
        let ta = &self.tensors[a.0 as usize];
        let tb = &self.tensors[b.0 as usize];
        assert_eq!(ta.ty, GgmlType::F32, "ggml_solve_tri: A must be F32");
        assert_eq!(tb.ty, GgmlType::F32, "ggml_solve_tri: B must be F32");
        assert_eq!(ta.ne[0], ta.ne[1], "ggml_solve_tri: A must be square");
        assert_eq!(ta.ne[1], tb.ne[1], "ggml_solve_tri: A.ne1 == B.ne1");
        assert_eq!(ta.ne[2], tb.ne[2], "ggml_solve_tri: A.ne2 == B.ne2");
        assert_eq!(ta.ne[3], tb.ne[3], "ggml_solve_tri: A.ne3 == B.ne3");
        assert!(is_contiguous_ctx(self, a), "ggml_solve_tri: A contiguous");
        assert!(is_contiguous_ctx(self, b), "ggml_solve_tri: B contiguous");
        assert!(left && lower && !uni, "ggml_solve_tri: left && lower && !uni only");
        let bne = tb.ne;
        let r = self.new_tensor(
            GgmlType::F32,
            [bne[0], bne[1], bne[2], bne[3]],
        );
        self.init_op(r, GgmlOp::SolveTri, [Some(a), Some(b), None]);
        r
    }

    /// 对照 ggml_col2im_1d (ggml.c:4679, GGML_OP_COL2IM_1D): scatter-add the
    /// columns `a` [K*OC, T_in] into a signal [T_out, OC] where
    /// `T_out = (T_in-1)*s0 + K - 2*p0` — the gather-form kernel ops.cpp:7017.
    /// The causal ConvTranspose1d core of both TTS generators.
    pub fn col2im_1d(&mut self, a: TensorId, s0: i32, oc: i32, p0: i32) -> TensorId {
        let t = &self.tensors[a.0 as usize];
        assert!(
            t.ne[2] == 1 && t.ne[3] == 1,
            "ggml_col2im_1d: a must be a matrix"
        );
        assert!(
            is_contiguous_ctx(self, a),
            "ggml_col2im_1d: contiguous input only"
        );
        assert!(
            matches!(t.ty, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16),
            "ggml_col2im_1d: F32/F16/BF16 only"
        );
        assert!(s0 > 0 && oc > 0 && p0 >= 0);
        let k_oc = t.ne[0];
        let t_in = t.ne[1];
        let k = k_oc / oc as i64;
        let t_out = (t_in - 1) * s0 as i64 + k - 2 * p0 as i64;
        assert_eq!(k_oc, k * oc as i64, "ggml_col2im_1d: ne0 % oc == 0");
        assert!(k > 0 && t_out > 0);
        let r = self.new_tensor_2d(t.ty, t_out, oc as i64);
        self.set_op_params_i32(r, &[s0, oc, p0]);
        self.init_op(r, GgmlOp::Col2Im1d, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_pad (ggml.c:5212 → ggml_pad_ext :5166): F32 tensor widened
    /// by `p0..p3` on dim 0..3 (right padding only through this constructor,
    /// exactly the graniteswitch router-lane use). op_params = the 8 pad
    /// widths + circular = 0.
    pub fn pad(&mut self, a: TensorId, p0: i32, p1: i32, p2: i32, p3: i32) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        assert_eq!(
            self.tensors[a.0 as usize].ty,
            GgmlType::F32,
            "ggml_pad: F32 only (ops.cpp:8261-8277)"
        );
        let r = self.new_tensor(
            GgmlType::F32,
            [ne[0] + p0 as i64, ne[1] + p1 as i64, ne[2] + p2 as i64, ne[3] + p3 as i64],
        );
        let params = [0, p0, 0, p1, 0, p2, 0, p3, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Pad, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_pad_ext (ggml.c:5166-5210): the full 8-width form of
    /// [`Context::pad`] — left/right pads on all four dims, F32, zero-filled
    /// margins (circular stays unported, asserted in compute). Audio round 4:
    /// parakeet's local-attention chunk padding pads dim2 on both sides
    /// (parakeet.cpp:180-244) and pocket-tts's SEANet causal convs pad dim0
    /// left+right (pockettts-seanet.cpp:36-38).
    #[allow(clippy::too_many_arguments)]
    pub fn pad_ext(
        &mut self,
        a: TensorId,
        lp0: i32,
        rp0: i32,
        lp1: i32,
        rp1: i32,
        lp2: i32,
        rp2: i32,
        lp3: i32,
        rp3: i32,
    ) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        assert_eq!(
            self.tensors[a.0 as usize].ty,
            GgmlType::F32,
            "ggml_pad_ext: F32 only (ops.cpp:8261-8277)"
        );
        let r = self.new_tensor(
            GgmlType::F32,
            [
                ne[0] + lp0 as i64 + rp0 as i64,
                ne[1] + lp1 as i64 + rp1 as i64,
                ne[2] + lp2 as i64 + rp2 as i64,
                ne[3] + lp3 as i64 + rp3 as i64,
            ],
        );
        let params = [lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Pad, [Some(a), None, None]);
        r
    }

    /// 对照 ggml_rope_ext_back (ggml.c:4485-4502): builds ggml_rope_ext then
    /// flips the op to GGML_OP_ROPE_BACK — the kernel is the *same* rope_flt
    /// body with `sin_sign = -1` (ops.cpp:6264-6285), i.e. the inverse
    /// rotation. deepseek4 de-rotates the attention output through this
    /// (deepseek4.cpp:1205).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext_back(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorId {
        let r = self.rope_ext(
            a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow,
        );
        self.tensors[r.0 as usize].op = GgmlOp::RoPEBack;
        r
    }

    /// 对照 ggml_swiglu_clamp (ggml.c:3123-3132 → ggml_glu_impl :2906 with
    /// b != NULL, GGML_GLU_OP_SWIGLU_CLAMP): op_params[0] = op, [1] =
    /// swapped (0), [3] = limit f32 bits. Kernel (ops.cpp:3408-3465):
    /// `min(gate, limit) / (1 + exp(-gate)) * clamp(up, -limit, limit)` — note
    /// the gate's sigmoid uses the *unclamped* min(gate,limit).
    pub fn swiglu_clamp(&mut self, a: TensorId, b: TensorId, limit: f32) -> TensorId {
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        assert_eq!(ane, bne, "ggml_swiglu_clamp: shape mismatch");
        let ty = self.tensors[a.0 as usize].ty;
        assert_eq!(ty, self.tensors[b.0 as usize].ty, "ggml_swiglu_clamp: type mismatch");
        let r = self.new_tensor(ty, ane);
        let mut params = [0i32; 4];
        params[0] = GGML_GLU_OP_SWIGLU_CLAMP;
        params[1] = 0; // swapped = false (both halves passed)
        params[3] = limit.to_bits() as i32;
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Glu, [Some(a), Some(b), None])
    }

    /// 对照 ggml_dsv4_hc_comb (ggml.c:6459-6505): F32 `[hc, hc, n_tokens]`
    /// — the fused hyper-connection mixing matrix. `hc` is recovered from
    /// `hc_mix_dim = (2 + hc)*hc` exactly like the C (:6470-6476, asserted
    /// == 4); op_params[0] = eps (f32 bits), [1] = n_iter (the sinkhorn
    /// iteration count). src = [mixes, scale, base].
    pub fn dsv4_hc_comb(
        &mut self,
        mixes: TensorId,
        scale: TensorId,
        base: TensorId,
        eps: f32,
        n_iter: i32,
    ) -> TensorId {
        let (mne, sne, bne) = (
            self.tensors[mixes.0 as usize].ne,
            self.tensors[scale.0 as usize].ne,
            self.tensors[base.0 as usize].ne,
        );
        assert_eq!(self.ty(mixes), GgmlType::F32, "dsv4_hc_comb: mixes->type == F32");
        assert_eq!(self.ty(scale), GgmlType::F32, "dsv4_hc_comb: scale->type == F32");
        assert_eq!(self.ty(base), GgmlType::F32, "dsv4_hc_comb: base->type == F32");
        assert!(n_iter > 0, "dsv4_hc_comb: n_iter > 0");

        let hc_mix_dim = mne[0];
        let n_tokens = mne[1];

        // ggml.c:6470-6476 — hc = the i with (2+i)*i == hc_mix_dim
        let mut hc = 0i64;
        let mut i = 1i64;
        while i * i + 2 * i <= hc_mix_dim {
            if (2 + i) * i == hc_mix_dim {
                hc = i;
                break;
            }
            i += 1;
        }
        assert!(hc > 0, "dsv4_hc_comb: hc > 0");
        assert_eq!(hc, 4, "dsv4_hc_comb: hc == 4");
        assert_eq!(mne[2], 1);
        assert_eq!(mne[3], 1);
        assert!(sne[0] >= 3);
        assert_eq!(sne[1], 1);
        assert_eq!(sne[2], 1);
        assert_eq!(sne[3], 1);
        assert_eq!(bne[0], hc_mix_dim);
        assert_eq!(bne[1], 1);
        assert_eq!(bne[2], 1);
        assert_eq!(bne[3], 1);

        let r = self.new_tensor(GgmlType::F32, [hc, hc, n_tokens, 1]);
        // ggml.c:6497-6498 — params[0] = eps (f32 bits), [1] = n_iter
        self.set_op_params_i32(r, &[eps.to_bits() as i32, n_iter]);
        self.init_op(r, GgmlOp::Dsv4HcComb, [Some(mixes), Some(scale), Some(base)]);
        r
    }

    /// 对照 ggml_dsv4_hc_pre (ggml.c:6548-6553 → ggml_dsv4_hc_pre_impl :6510,
    /// scale = 1.0, gated = false): F32 `[n_embd, n_tokens]` — the fused
    /// hyper-connection stream mix. x is `[n_embd, hc, n_tokens]`, weights
    /// `[hc, n_tokens]`; op_params[0] = scale (f32 bits), [1] = gated (0).
    pub fn dsv4_hc_pre(&mut self, x: TensorId, weights: TensorId) -> TensorId {
        let (xne, wne) = (self.tensors[x.0 as usize].ne, self.tensors[weights.0 as usize].ne);
        assert_eq!(self.ty(x), GgmlType::F32, "dsv4_hc_pre: x->type == F32");
        assert_eq!(self.ty(weights), GgmlType::F32, "dsv4_hc_pre: weights->type == F32");
        let n_embd = xne[0];
        let hc = xne[1];
        let n_tokens = xne[2];
        assert!(hc > 0);
        assert_eq!(xne[3], 1);
        // gated = false branch (ggml.c:6531-6534)
        assert_eq!(wne[0], hc);
        assert_eq!(wne[1], n_tokens);
        assert_eq!(wne[2], 1);
        assert_eq!(wne[3], 1);

        let r = self.new_tensor(GgmlType::F32, [n_embd, n_tokens, 1, 1]);
        let params = [1.0f32.to_bits() as i32, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Dsv4HcPre, [Some(x), Some(weights), None]);
        r
    }

    /// 对照 ggml_dsv4_hc_pre_gated (ggml.c:6555-6586 → the `gated = true`
    /// arm of ggml_dsv4_hc_pre_impl, :6490-6540). The weights are the full
    /// `[n_embd, hc, n_tokens]` gate tensor and the kernel folds the sigmoid
    /// gate and the mean over the streams in one op (ops.cpp gated branch).
    /// arch batch 11a (qwen4exp): `build_hc_mix`'s fused default
    /// (`cparams.fused_dsv4_hc_pre`, qwen4exp.cpp:291).
    pub fn dsv4_hc_pre_gated(
        &mut self,
        x: TensorId,
        weights: TensorId,
        scale: f32,
    ) -> TensorId {
        let (xne, wne) = (self.tensors[x.0 as usize].ne, self.tensors[weights.0 as usize].ne);
        assert_eq!(self.ty(x), GgmlType::F32, "dsv4_hc_pre_gated: x->type == F32");
        assert_eq!(self.ty(weights), GgmlType::F32, "dsv4_hc_pre_gated: weights->type == F32");
        let n_embd = xne[0];
        let hc = xne[1];
        let n_tokens = xne[2];
        assert!(hc > 0);
        assert_eq!(xne[3], 1);
        // gated = true branch (ggml.c:6536-6540)
        assert_eq!(wne[0], n_embd);
        assert_eq!(wne[1], hc);
        assert_eq!(wne[2], n_tokens);
        assert_eq!(wne[3], 1);

        let r = self.new_tensor(GgmlType::F32, [n_embd, n_tokens, 1, 1]);
        let params = [scale.to_bits() as i32, 1];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Dsv4HcPre, [Some(x), Some(weights), None]);
        r
    }

    /// 对照 ggml_dsv4_hc_post (ggml.c:6565-6607): F32
    /// `[n_embd, hc, n_tokens]` — the fused hyper-connection output mix.
    /// x `[n_embd, n_tokens]`, residual `[n_embd, hc, n_tokens]`, post
    /// `[hc, n_tokens]`, comb `[hc, hc, n_tokens]` (None = the NULL branch —
    /// identity mixing, each stream keeps its own residual).
    pub fn dsv4_hc_post(
        &mut self,
        x: TensorId,
        residual: TensorId,
        post: TensorId,
        comb: Option<TensorId>,
    ) -> TensorId {
        let (xne, rne, pne) = (
            self.tensors[x.0 as usize].ne,
            self.tensors[residual.0 as usize].ne,
            self.tensors[post.0 as usize].ne,
        );
        assert_eq!(self.ty(x), GgmlType::F32, "dsv4_hc_post: x->type == F32");
        assert_eq!(self.ty(residual), GgmlType::F32, "dsv4_hc_post: residual->type == F32");
        assert_eq!(self.ty(post), GgmlType::F32, "dsv4_hc_post: post->type == F32");
        let n_embd = xne[0];
        let n_tokens = xne[1];
        let hc = rne[1];
        assert!(hc > 0);
        assert_eq!(xne[2], 1);
        assert_eq!(xne[3], 1);
        assert_eq!(rne[0], n_embd);
        assert_eq!(rne[2], n_tokens);
        assert_eq!(rne[3], 1);
        assert_eq!(pne[0], hc);
        assert_eq!(pne[1], n_tokens);
        assert_eq!(pne[2], 1);
        assert_eq!(pne[3], 1);
        if let Some(c) = comb {
            let cne = self.tensors[c.0 as usize].ne;
            assert_eq!(self.ty(c), GgmlType::F32, "dsv4_hc_post: comb->type == F32");
            assert_eq!(cne[0], hc);
            assert_eq!(cne[1], hc);
            assert_eq!(cne[2], n_tokens);
            assert_eq!(cne[3], 1);
        }

        let r = self.new_tensor(GgmlType::F32, [n_embd, hc, n_tokens, 1]);
        self.init_op(r, GgmlOp::Dsv4HcPost, [Some(x), Some(residual), Some(post), comb]);
        r
    }

    pub fn mul_mat(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(can_mul_mat(self, a, b), "ggml_mul_mat: !can_mul_mat");
        assert!(!is_transposed(self, a), "ggml_mul_mat: a is transposed");
        let (an1, bn1, bn2, bn3) = {
            let at = &self.tensors[a.0 as usize];
            let bt = &self.tensors[b.0 as usize];
            (at.ne[1], bt.ne[1], bt.ne[2], bt.ne[3])
        };
        let r = self.new_tensor(GgmlType::F32, [an1, bn1, bn2, bn3]);
        self.init_op(r, GgmlOp::MulMat, [Some(a), Some(b), None])
    }

    /// 对照 ggml_mul_mat_set_hint(ctx, GGML_HINT_SRC0_IS_HADAMARD)
    /// (ggml.c:3368-3373): mark an existing MUL_MAT node's src0 as a Hadamard
    /// matrix — the CPU kernel then computes the fast Walsh-Hadamard
    /// transform (ggml-cpu.c:1263-1266), not the gemm dot. `llama_mul_mat_
    /// hadamard` (llama-impl.h:70) always sets it, so the port must too to
    /// keep the accumulation order (arch batch 7).
    pub fn mul_mat_set_hint_hadamard(&mut self, r: TensorId) {
        assert_eq!(self.op(r), GgmlOp::MulMat, "mul_mat_set_hint_hadamard: not a MUL_MAT");
        self.set_op_params_i32(r, &[0, GGML_HINT_SRC0_IS_HADAMARD]);
    }

    /// 对照 ggml_mul_mat_id (ggml.c:3354) — MoE expert matmul: `as` is a stack of
    /// n_expert matrices {ne00, ne01, ne02}, `b` the activations {ne00, ne11,
    /// n_tokens} and `ids` the per-(slot, token) expert indices {n_ids, n_tokens}
    /// (I32, may be a strided view — see ggml_argsort_top_k). Result
    /// {as->ne[1], ids->ne[0], b->ne[2]} — i.e. [n_ff, n_expert_used, n_tokens].
    pub fn mul_mat_id(&mut self, as_: TensorId, b: TensorId, ids: TensorId) -> TensorId {
        assert!(!is_transposed(self, as_), "ggml_mul_mat_id: as is transposed");
        let (ane, bne, ine) = (
            self.tensors[as_.0 as usize].ne,
            self.tensors[b.0 as usize].ne,
            self.tensors[ids.0 as usize].ne,
        );
        assert_eq!(self.tensors[ids.0 as usize].ty, GgmlType::I32, "mul_mat_id: ids must be I32");
        assert_eq!(ane[3], 1, "ggml_mul_mat_id: as must be 3d");
        assert_eq!(bne[3], 1, "ggml_mul_mat_id: b must be 3d");
        assert!(ine[2] == 1 && ine[3] == 1, "ggml_mul_mat_id: ids must be 2d");
        assert_eq!(ine[1], bne[2], "ggml_mul_mat_id: ids->ne[1] != b->ne[2]");
        assert_eq!(ane[0], bne[0], "ggml_mul_mat_id: !can_mul_mat");
        assert_eq!(ine[0] % bne[1], 0, "ggml_mul_mat_id: ids->ne[0] % b->ne[1]");

        let ne = [ane[1], ine[0], bne[2], 1];
        let r = self.new_tensor(GgmlType::F32, ne);
        self.init_op(r, GgmlOp::MulMatId, [Some(as_), Some(b), Some(ids)])
    }

    /// 对照 ggml_add_id (ggml.c:2150): dst[i0, i1, i2] = a[...] + b[i0, ids[i1, i2]].
    /// `b` is the per-expert bias {ne0, n_expert}; a/dst are {ne0, n_ids, n_tokens}.
    pub fn add_id(&mut self, a: TensorId, b: TensorId, ids: TensorId) -> TensorId {
        let (ane, bne, ine) = (
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne,
            self.tensors[ids.0 as usize].ne,
        );
        assert_eq!(self.tensors[ids.0 as usize].ty, GgmlType::I32, "add_id: ids must be I32");
        assert_eq!(ane[0], bne[0], "ggml_add_id: a->ne[0] != b->ne[0]");
        assert_eq!(ane[1], ine[0], "ggml_add_id: a->ne[1] != ids->ne[0]");
        assert_eq!(ane[2], ine[1], "ggml_add_id: a->ne[2] != ids->ne[1]");
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::AddId, [Some(a), Some(b), Some(ids)])
    }

    // ===================== norm / rms_norm / scale =====================

    /// 对照 ggml_norm (ggml.c:3151). op_params[0] = eps (f32 bits), [1] = 0.
    pub fn norm(&mut self, a: TensorId, eps: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Norm, [Some(a), None, None]);
        self.set_op_params_i32(r, &[eps.to_bits() as i32, 0]); // [1] == 0 => plain norm
        r
    }
    /// 对照 ggml_norm_inplace (ggml.c:3158)
    pub fn norm_inplace(&mut self, a: TensorId, eps: f32) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Norm, [Some(a), None, None]);
        self.set_op_params_i32(r, &[eps.to_bits() as i32, 0]);
        r
    }

    /// 对照 ggml_rms_norm (ggml.c:3182). Encoded as Norm + params[1]==1 until
    /// tensor.rs grows a GgmlOp::RmsNorm variant.
    pub fn rms_norm(&mut self, a: TensorId, eps: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Norm, [Some(a), None, None]);
        self.set_op_params_i32(r, &[eps.to_bits() as i32, OP_FLAG_NORM_IS_RMS]);
        r
    }
    /// 对照 ggml_rms_norm_inplace (ggml.c:3189)
    pub fn rms_norm_inplace(&mut self, a: TensorId, eps: f32) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Norm, [Some(a), None, None]);
        self.set_op_params_i32(r, &[eps.to_bits() as i32, OP_FLAG_NORM_IS_RMS]);
        r
    }

    /// 对照 ggml_l2_norm (ggml.c:3275): dst = x / max(||x||_2, eps), the
    /// row sum of squares accumulated in f64 like the CPU kernel
    /// (ggml-cpu/ops.cpp:4486). Encoded as Norm + params[1]==2 (the RMS
    /// precedent). F32 only, like the C.
    pub fn l2_norm(&mut self, a: TensorId, eps: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Norm, [Some(a), None, None]);
        self.set_op_params_i32(r, &[eps.to_bits() as i32, OP_FLAG_NORM_IS_L2]);
        r
    }

    /// 对照 ggml_scale (ggml.c:3467). op_params = {s, b=0} f32 bits.
    pub fn scale(&mut self, a: TensorId, s: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Scale, [Some(a), None, None]);
        self.params_f32(r, &[s, 0.0]);
        r
    }
    /// 对照 ggml_scale_bias (ggml.c:3481) — op_params = {s, b}, the kernel's
    /// `ggml_vec_mad1_f32` path (y = x*s + b). deepseek4's hc pre gates use
    /// it with s = 1 (deepseek4.cpp:384: `pre + dsv4_hc_eps`).
    pub fn scale_bias(&mut self, a: TensorId, s: f32, b: f32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Scale, [Some(a), None, None]);
        self.params_f32(r, &[s, b]);
        r
    }
    /// 对照 ggml_scale_inplace (ggml.c:3474)
    pub fn scale_inplace(&mut self, a: TensorId, s: f32) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Scale, [Some(a), None, None]);
        self.params_f32(r, &[s, 0.0]);
        r
    }

    // ===================== soft_max =====================

    /// 对照 ggml_soft_max (ggml.c:4179): scale=1, max_bias=0.
    pub fn soft_max(&mut self, a: TensorId) -> TensorId {
        self.soft_max_ext(a, None, 1.0, 0.0)
    }

    /// 对照 ggml_soft_max_ext (ggml.c:4191).
    /// `mask` is src[1] (F32 or F16). op_params = {scale, max_bias} f32 bits.
    pub fn soft_max_ext(
        &mut self,
        a: TensorId,
        mask: Option<TensorId>,
        scale: f32,
        max_bias: f32,
    ) -> TensorId {
        let ane = self.tensors[a.0 as usize].ne;
        if let Some(m) = mask {
            let mt = &self.tensors[m.0 as usize];
            assert!(
                mt.ty == GgmlType::F16 || mt.ty == GgmlType::F32,
                "soft_max mask must be F16 or F32"
            );
            assert_eq!(mt.ne[0], ane[0], "soft_max mask ne0");
            assert!(mt.ne[1] >= ane[1], "soft_max mask ne1");
            assert!(ane[2] % mt.ne[2] == 0 && ane[3] % mt.ne[3] == 0, "soft_max mask broadcast");
        }
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::SoftMax, [Some(a), mask, None]);
        self.params_f32(r, &[scale, max_bias]);
        r
    }

    /// 对照 ggml_soft_max_add_sinks (ggml.c:4209) — attach the attention-sink
    /// logits (F32, one per head = src[0]->ne[2]) as src[2] of an existing
    /// softmax node. The CPU kernel then folds `expf(sk[h] - max)` into the
    /// denominator and includes sk[h] in the row max (ops.cpp:5625-5680).
    pub fn soft_max_add_sinks(&mut self, a: TensorId, sinks: Option<TensorId>) {
        let (op, ne2) = {
            let t = &self.tensors[a.0 as usize];
            (t.op, t.ne[2])
        };
        let Some(s) = sinks else {
            self.tensors[a.0 as usize].src[2] = None;
            return;
        };
        assert_eq!(op, GgmlOp::SoftMax, "ggml_soft_max_add_sinks: not a softmax node");
        let ts = &self.tensors[s.0 as usize];
        assert!(self.tensors[a.0 as usize].src[2].is_none(), "sinks already set");
        assert_eq!(ne2, ts.ne[0], "ggml_soft_max_add_sinks: ne[2] != sinks->ne[0]");
        assert_eq!(ts.ty, GgmlType::F32, "sinks must be F32");
        self.tensors[a.0 as usize].src[2] = Some(s);
    }

    // ===================== glu / top-k =====================

    /// 对照 ggml_swiglu_oai (ggml.c:2802 → ggml_glu_impl, ggml.c:2760) with
    /// alpha/limit in op_params[2]/[3] (f32 bits); params[0] = GGML_GLU_OP_SWIGLU_OAI,
    /// params[1] = swapped (0). src = [gate, up].
    pub fn swiglu_oai(&mut self, a: TensorId, b: TensorId, alpha: f32, limit: f32) -> TensorId {
        let (ane, bne, ty) = (
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne,
            self.tensors[a.0 as usize].ty,
        );
        assert_eq!(ane, bne, "ggml_swiglu_oai: shape mismatch");
        assert_eq!(ty, self.tensors[b.0 as usize].ty, "ggml_swiglu_oai: type mismatch");
        let r = self.new_tensor(ty, ane);
        let mut params = [0i32; 4];
        params[0] = GGML_GLU_OP_SWIGLU_OAI;
        params[1] = 0; // swapped = false (ggml_glu_impl `swapped` arg)
        params[2] = alpha.to_bits() as i32;
        params[3] = limit.to_bits() as i32;
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Glu, [Some(a), Some(b), None])
    }

    /// 对照 ggml_argsort_top_k (ggml.c:5454): argsort DESC then a strided view of
    /// the first k columns. The view keeps the full-row stride nb[1] (= ne0*4),
    /// exactly like C — downstream get_rows/add_id/mul_mat_id read ids through nb.
    pub fn argsort_top_k(&mut self, a: TensorId, k: i32) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        assert!(ne[0] >= k as i64, "ggml_argsort_top_k: a->ne[0] < k");
        let full = self.argsort(a, GGML_SORT_ORDER_DESC);
        let nb1 = self.tensors[full.0 as usize].nb[1] as usize;
        let nb2 = self.tensors[full.0 as usize].nb[2] as usize;
        let nb3 = self.tensors[full.0 as usize].nb[3] as usize;
        self.view_4d(full, k as i64, ne[1], ne[2], ne[3], nb1, nb2, nb3, 0)
    }

    // ===================== tanh / silu / gelu =====================

    /// 对照 ggml_tanh (ggml.c:2709). op = UNARY(GgmlOp::Silu), params[0] = 4.
    pub fn tanh(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_TANH]);
        r
    }
    /// 对照 ggml_tanh_inplace (ggml.c:2715)
    pub fn tanh_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_TANH]);
        r
    }

    /// 对照 ggml_silu (ggml.c:2824). op = UNARY(GgmlOp::Silu), params[0] = 10.
    pub fn silu(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_SILU]);
        r
    }
    /// 对照 ggml_silu_inplace (ggml.c:2830)
    pub fn silu_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_SILU]);
        r
    }

    /// 对照 ggml_sigmoid (ggml.c:2731 → ggml_unary). op = UNARY(GgmlOp::Silu),
    /// params[0] = 7 (GGML_UNARY_OP_SIGMOID). Kernel: `1/(1 + expf(-x))` with
    /// libm expf (vec.h:936 `ggml_vec_sigmoid_f32` has no SIMD variant) — used
    /// by the MoE router of LFM2 (`expert_gating_func == 2` = SIGMOID).
    pub fn sigmoid(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_SIGMOID]);
        r
    }
    /// 对照 ggml_relu (ggml.c:2737 → ggml_unary). op = UNARY(GgmlOp::Silu),
    /// params[0] = 6 (GGML_UNARY_OP_RELU). Kernel: vec.h:922
    /// `ggml_vec_relu_f32` = `(x > 0) ? x : 0` (no SIMD variant).
    pub fn relu(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_RELU]);
        r
    }
    /// 对照 ggml_relu_inplace (ggml.c:2743)
    pub fn relu_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_RELU]);
        r
    }
    /// 对照 ggml_sigmoid_inplace (ggml.c:2737)
    pub fn sigmoid_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_SIGMOID]);
        r
    }

    /// 对照 ggml_elu (ggml.c:2721-2727 → ggml_unary(GGML_UNARY_OP_ELU)). op =
    /// UNARY, params[0] = 5. Kernel: `(x > 0) ? x : expm1f(x)` (vec.h:915
    /// `ggml_vec_elu_f32`, scalar). The pocket-tts SEANet activation
    /// (pockettts-seanet.cpp:104).
    pub fn elu(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_ELU]);
        r
    }

    /// 对照 ggml_gelu (ggml.c:2782). op = UNARY, params[0] = 8 (GELU).
    pub fn gelu(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_GELU]);
        r
    }
    /// 对照 ggml_gelu_inplace (ggml.c:2788)
    pub fn gelu_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_GELU]);
        r
    }

    /// 对照 ggml_gelu_erf (ggml.c:2796 → ggml_unary(GELU_ERF)). op = UNARY,
    /// params[0] = 16. Kernel: `0.5f*x*(1.0f + erff(x*SQRT_2_INV))`
    /// (`ggml_vec_gelu_erf_f32`, vec.h:1010 — a plain scalar libm loop, no
    /// SIMD variant at this commit); F32 and F16 only
    /// (`ggml_compute_forward_gelu_erf`, ops.cpp:2436-2453, else abort).
    /// The whisper-enc audio graph's activation (whisper-enc.cpp:20-21).
    pub fn gelu_erf(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_GELU_ERF]);
        r
    }
    /// 对照 ggml_gelu_erf_inplace (ggml.c:2802)
    pub fn gelu_erf_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_GELU_ERF]);
        r
    }

    /// 对照 ggml_exp (ggml.c:2892 → GGML_UNARY_OP_EXP). op = UNARY,
    /// params[0] = 13. Kernel: per-element `expf` — the reference has no SIMD
    /// exp at this commit (unary-ops.cpp:37 `op_exp` through
    /// `unary_op<op_exp>` :273, and vec.h:956 `ggml_vec_exp_f32` is a plain
    /// scalar loop too). minimax-01's lightning attention applies it to the
    /// q/k/diag decays (minimax-01.cpp:326-331/368).
    pub fn exp(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_EXP]);
        r
    }
    /// 对照 ggml_exp_inplace (ggml.c:2898)
    pub fn exp_inplace(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_EXP]);
        r
    }

    /// 对照 ggml_round (ggml.c:2963 → GGML_UNARY_OP_ROUND). op = UNARY,
    /// params[0] = 20. Kernel: per-element `roundf` (unary-ops.cpp:92 `op_round`
    /// → `unary_op<op_round>` :317); graniteswitch's router lane rounds the
    /// attended slot value before the I32 cast (granite-switch.cpp:285).
    pub fn round(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(r, &[GGML_UNARY_OP_ROUND]);
        r
    }

    /// 对照 ggml_xielu (ggml.c:2838 → GGML_UNARY_OP_XIELU). op = UNARY with a
    /// parameter block instead of the bare op id at [0]:
    ///   [0] = GGML_UNARY_OP_XIELU (i32)
    ///   [1] = beta + softplus(alpha_n)   (f32 bits, folded at build time)
    ///   [2] = softplus(alpha_p)          (f32 bits)
    ///   [3] = beta                       (f32 bits)
    ///   [4] = eps                        (f32 bits)
    /// `ggml_compute_softplus_f32` is ggml-impl.h:107-109:
    /// `x > 20 ? x : logf(1 + expf(x))`. Kernel: unary-ops.cpp:55-62 `op_xielu`
    /// + :325-336 (a per-element functor, no SIMD) — apertus's FFN activation.
    pub fn xielu(
        &mut self,
        a: TensorId,
        alpha_n: f32,
        alpha_p: f32,
        beta: f32,
        eps: f32,
    ) -> TensorId {
        let sp = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None]);
        self.set_op_params_i32(
            r,
            &[
                GGML_UNARY_OP_XIELU,
                (beta + sp(alpha_n)).to_bits() as i32,
                sp(alpha_p).to_bits() as i32,
                beta.to_bits() as i32,
                eps.to_bits() as i32,
            ],
        );
        r
    }

    // `ggml_op_pool` (ggml.h:2277-2280): MAX = 0, AVG = 1 (the port originally
    // had these inverted — internally consistent, so no numeric comparison was
    // affected — but the POOL_1D C-probe dump compares op_params verbatim, so
    // the discriminants now match ggml.h exactly)
    /// ggml.h:2277 `GGML_OP_POOL_MAX`
    pub const GGML_OP_POOL_MAX: i32 = 0;
    /// ggml.h:2278 `GGML_OP_POOL_AVG`
    pub const GGML_OP_POOL_AVG: i32 = 1;

    /// 对照 ggml_pool_2d (ggml.c:4928 → GGML_OP_POOL_2D). op_params =
    /// {op, k0, k1, s0, s1, p0, p1} — the C builder stores the float pads
    /// through an `int32_t params[]` (implicit truncation); dst is always F32
    /// `[calc_pool_out(ne0,k0,s0,p0), calc_pool_out(ne1,k1,s1,p1), ne2, ne3]`.
    /// Kernel: ops.cpp:7767-7852 — single-threaded (ith==0 only), the pool
    /// window slides over ne0/ne1 with boundary skips.
    #[allow(clippy::too_many_arguments)]
    pub fn pool_2d(
        &mut self,
        a: TensorId,
        op: i32,
        k0: i32,
        k1: i32,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
    ) -> TensorId {
        // ggml_calc_pool_output_size (ggml.c:5065): (ins + 2*p - ks)/s + 1
        // (C integer division truncates toward zero; p already int here)
        let out_size = move |ins: i64, k: i32, s: i32, p: i32| {
            (ins + 2 * p as i64 - k as i64) / s as i64 + 1
        };
        let ane = self.tensors[a.0 as usize].ne;
        let ne = [
            out_size(ane[0], k0, s0, p0),
            out_size(ane[1], k1, s1, p1),
            ane[2],
            ane[3],
        ];
        assert!(ne[0] > 0 && ne[1] > 0, "ggml_pool_2d: empty output");
        let r = self.new_tensor(GgmlType::F32, ne);
        self.init_op(r, GgmlOp::Pool2d, [Some(a), None, None]);
        self.set_op_params_i32(r, &[op, k0, k1, s0, s1, p0, p1]);
        r
    }

    /// 对照 ggml_pool_1d (ggml.c:5071 → GGML_OP_POOL_1D, ggml.h:2281).
    /// op_params = {op, k0, s0, p0}; dst is always F32
    /// `[calc_pool_out(ne0,k0,s0,p0), ne1, ne2, ne3]`.
    /// Kernel: ops.cpp:7690-7754 (`ggml_compute_forward_pool_1d_ksp`) —
    /// single-threaded (ith==0 only), per-row window with boundary skips;
    /// AVG divides by the in-range `count` (NOT k0 — the pool_2d kernel
    /// divides by ka=k0*k1, the behavioral difference between the two ops).
    pub fn pool_1d(&mut self, a: TensorId, op: i32, k0: i32, s0: i32, p0: i32) -> TensorId {
        // ggml_calc_pool_output_size (ggml.c:5065): `(ins + 2*p - ks)/s + 1`
        // where p is FLOAT — the numerator and the division happen in f32 and
        // the result is truncated toward zero AFTER the +1 (differs from
        // integer division when |ins+2p-k| < s and ins+2p-k < 0)
        let ane = self.tensors[a.0 as usize].ne;
        let ne0 = (((ane[0] + 2 * p0 as i64 - k0 as i64) as f32) / (s0 as f32) + 1.0) as i64;
        let ne = [ne0, ane[1], ane[2], ane[3]];
        assert!(ne[0] > 0, "ggml_pool_1d: empty output");
        let r = self.new_tensor(GgmlType::F32, ne);
        self.init_op(r, GgmlOp::Pool1d, [Some(a), None, None]);
        self.set_op_params_i32(r, &[op, k0, s0, p0]);
        r
    }

    /// 对照 ggml_arange (ggml.c:5475 → GGML_OP_ARANGE): a fresh F32 1-D
    /// tensor of `ceilf((stop-start)/step)` elements, `value = start + step*i`
    /// (ops.cpp:8386-8405).
    pub fn arange(&mut self, start: f32, stop: f32, step: f32) -> TensorId {
        assert!(stop > start, "ggml_arange: stop must be > start");
        let steps = ((stop - start) / step).ceil() as i64;
        let r = self.new_tensor_1d(GgmlType::F32, steps);
        self.init_op(r, GgmlOp::Arange, [None, None, None]);
        self.params_f32(r, &[start, stop, step]);
        r
    }

    // ===================== rope =====================

    /// 对照 ggml_rope_impl (ggml.c:4266) — the single implementation shared by
    /// every rope flavor. op_params layout (16 i32):
    ///   [0]=n_past(0) [1]=n_dims [2]=mode [3]=n_ctx(0) [4]=n_ctx_orig
    ///   [5..11]=freq_base/freq_scale/ext_factor/attn_factor/beta_fast/beta_slow
    ///           (f32 bit patterns)
    ///   [11..15]=mrope sections [15]=n_offs
    /// `c` is the optional rope freq-factor tensor (src[2], ggml.c:4318-4320).
    #[allow(clippy::too_many_arguments)]
    fn rope_impl(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        sections: Option<[i32; GGML_MROPE_SECTIONS]>,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        inplace: bool,
    ) -> TensorId {
        assert_eq!(mode & 1, 0, "rope mode & 1 == 1 is no longer supported");

        let bt = &self.tensors[b.0 as usize];
        assert!(bt.is_vector(), "rope: b must be a vector");
        assert_eq!(bt.ty, GgmlType::I32, "rope: b must be I32");

        let ane = self.tensors[a.0 as usize].ne;
        let mrope_used = mode & GGML_ROPE_TYPE_MROPE != 0;
        if mrope_used {
            assert_eq!(ane[2] * 4, bt.ne[0], "rope mrope: expect 4 position ids per token");
        } else {
            assert_eq!(ane[2], bt.ne[0], "rope: a->ne[2] != b->ne[0]");
        }
        if let Some(c) = c {
            // ggml.c:4312-4315
            assert_eq!(self.tensors[c.0 as usize].ty, GgmlType::F32, "rope: c must be F32");
            assert!(self.tensors[c.0 as usize].ne[0] >= (n_dims / 2) as i64, "rope: c->ne[0] >= n_dims/2");
        }

        let r = if inplace { self.view_tensor_of(a) } else { self.dup_tensor_of(a) };

        let mut params = [0i32; 16];
        params[1] = n_dims;
        params[2] = mode;
        params[4] = n_ctx_orig;
        params[5] = freq_base.to_bits() as i32;
        params[6] = freq_scale.to_bits() as i32;
        params[7] = ext_factor.to_bits() as i32;
        params[8] = attn_factor.to_bits() as i32;
        params[9] = beta_fast.to_bits() as i32;
        params[10] = beta_slow.to_bits() as i32;
        // ggml.c:4296-4300: sections land only for mrope modes
        if mrope_used {
            if let Some(s) = sections {
                params[11..15].copy_from_slice(&s);
            }
        }
        params[15] = 0; // n_offs, set via rope_set_offset() below (ggml.c:4313)
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::RoPE, [Some(a), Some(b), c])
    }

    // ===================== rope offset (deepseek2 MLA batch 6) =============

    /// 对照 ggml_rope_set_offset (ggml.c:4528-4536): set op_params[15] = n_offs
    /// on an existing ROPE node, so the rotation covers the *trailing* pairs
    /// `[n_offs, n_offs + n_dims)` of each row (deepseek2.cpp:617 — the
    /// non-MLA path ropes q's trailing qk_rope segment past the qk_nope
    /// prefix). The CPU kernel (compute.rs forward_rope) has read n_offs
    /// since the rope port; only this builder setter was missing.
    pub fn rope_set_offset(&mut self, a: TensorId, n_offs: i32) -> TensorId {
        assert!(n_offs >= 0);
        assert!(is_contiguous_ctx(self, a), "rope_set_offset: a must be contiguous");
        // ggml.c:4529 — ROPE or ROPE_BACK (deepseek4 de-ropes then offsets,
        // deepseek4.cpp:1206-1207)
        assert!(
            self.op(a) == GgmlOp::RoPE || self.op(a) == GgmlOp::RoPEBack,
            "rope_set_offset: a->op == GGML_OP_ROPE || GGML_OP_ROPE_BACK"
        );
        // ggml_set_op_params_i32(a, 15, n_offs) — only slot 15 changes; the
        // port's setter writes a prefix, so re-emit the whole 16-slot array
        let mut params = *self.op_params(a);
        params[15] = n_offs;
        self.set_op_params_i32(a, &params);
        a
    }

    /// 对照 ggml_rope (ggml.c:4325)
    pub fn rope(&mut self, a: TensorId, b: TensorId, n_dims: i32, mode: i32) -> TensorId {
        self.rope_impl(a, b, None, n_dims, None, mode, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0, false)
    }

    /// 对照 ggml_rope_inplace (ggml.c:4378)
    pub fn rope_inplace(&mut self, a: TensorId, b: TensorId, n_dims: i32, mode: i32) -> TensorId {
        self.rope_impl(a, b, None, n_dims, None, mode, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0, true)
    }

    /// 对照 ggml_rope_multi (ggml.c:4336): mrope with `sections` and the
    /// optional freq-factor tensor (src[2]) — gemma4's per-layer `rope_freqs`
    /// (gemma4.cpp:129/142, proportional rope) and qwen35's IMROPE
    /// (qwen35.cpp:299-309, sections = hparams.rope_sections).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_multi(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        sections: [i32; GGML_MROPE_SECTIONS],
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorId {
        self.rope_impl(
            a, b, c, n_dims, Some(sections), mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow, false,
        )
    }

    /// 对照 ggml_rope_ext (ggml.c:4389) with an optional freq-factor tensor
    /// (src[2]) — the gemma4 full-attention layers.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext_c(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorId {
        self.rope_ext(
            a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow,
        )
    }

    /// 对照 ggml_rope_ext (ggml.c:4389). `c` (freq factors, src[2]) is accepted
    /// verbatim (see `rope_ext_c`); the callers that pass None are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorId {
        self.rope_impl(
            a, b, c, n_dims, None, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow, false,
        )
    }

    /// 对照 ggml_rope_ext_inplace (ggml.c:4409)
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext_inplace(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: Option<TensorId>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorId {
        self.rope_impl(
            a, b, c, n_dims, None, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow, true,
        )
    }

    // ===================== get_rows =====================

    /// 对照 ggml_get_rows (ggml.c:3954)
    pub fn get_rows(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ane, bne) = (
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne,
        );
        let bt = &self.tensors[b.0 as usize];
        assert_eq!(bt.ty, GgmlType::I32, "get_rows: b must be I32");
        assert_eq!(ane[2], bne[1], "get_rows: ne02 != ne11");
        assert_eq!(ane[3], bne[2], "get_rows: ne03 != ne12");
        assert_eq!(bne[3], 1, "get_rows: b ne3 != 1");

        let ty = if self.tensors[a.0 as usize].ty == GgmlType::I32 {
            GgmlType::I32
        } else {
            GgmlType::F32
        };
        let r = self.new_tensor(ty, [ane[0], bne[0], bne[1], bne[2]]);
        self.init_op(r, GgmlOp::GetRows, [Some(a), Some(b), None])
    }

    /// 对照 ggml_get_rows_back (ggml.c:3979). `c` only contributes the output
    /// shape in C. Encoded as GetRows + op_params[15] == 1.
    pub fn get_rows_back(&mut self, a: TensorId, b: TensorId, c: TensorId) -> TensorId {
        let (ane, bne, cne) = (
            self.tensors[a.0 as usize].ne,
            self.tensors[b.0 as usize].ne,
            self.tensors[c.0 as usize].ne,
        );
        assert_eq!(self.tensors[b.0 as usize].ty, GgmlType::I32, "get_rows_back: b must be I32");
        assert!(bne[1] == 1 && bne[2] == 1 && bne[3] == 1, "get_rows_back: b must be vector");
        assert!(cne[2] == 1 && cne[3] == 1, "get_rows_back: c must be matrix");
        assert_eq!(ane[0], cne[0], "get_rows_back: ne00 != c ne0");

        let r = self.new_tensor_2d(GgmlType::F32, cne[0], cne[1]);
        self.set_op_params_i32(r, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, OP_FLAG_GET_ROWS_BACK]);
        self.init_op(r, GgmlOp::GetRows, [Some(a), Some(b), None])
    }

    // ===================== cpy / cont =====================

    /// 对照 ggml_cpy (ggml.c:3603): result is a view of the destination `b`.
    pub fn cpy(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert_eq!(
            self.tensors[a.0 as usize].n_elements(),
            self.tensors[b.0 as usize].n_elements(),
            "cpy: nelements mismatch"
        );
        let r = self.view_tensor_of(b);
        self.init_op(r, GgmlOp::Cpy, [Some(a), Some(b), None])
    }

    /// 对照 ggml_cont (ggml.c:3639). GGML_OP_CONT maps to GgmlOp::Dup here
    /// (C's forward_cont calls forward_dup unchanged).
    pub fn cont(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Dup, [Some(a), None, None])
    }

    /// 对照 ggml_cont_4d (ggml.c:3670): fresh **contiguous** tensor of the given
    /// shape filled with `a`'s elements in `a`'s own dimension order (the CPU
    /// kernel is forward_dup, ops.cpp:326 — a permuted `a` is linearised here).
    pub fn cont_4d(&mut self, a: TensorId, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> TensorId {
        let ty = self.tensors[a.0 as usize].ty;
        assert_eq!(
            self.tensors[a.0 as usize].n_elements(),
            ne0 * ne1 * ne2 * ne3,
            "ggml_cont_4d: nelements mismatch"
        );
        let r = self.new_tensor(ty, [ne0, ne1, ne2, ne3]);
        self.init_op(r, GgmlOp::Dup, [Some(a), None, None])
    }

    /// 对照 ggml_cont_3d (ggml.c:3661)
    pub fn cont_3d(&mut self, a: TensorId, ne0: i64, ne1: i64, ne2: i64) -> TensorId {
        self.cont_4d(a, ne0, ne1, ne2, 1)
    }

    /// 对照 ggml_cont_2d (ggml.c:3653)
    pub fn cont_2d(&mut self, a: TensorId, ne0: i64, ne1: i64) -> TensorId {
        self.cont_4d(a, ne0, ne1, 1, 1)
    }

    // ===================== views / reshape / permute / transpose =====================

    /// 对照 ggml_view_1d (ggml.c:3800)
    pub fn view_1d(&mut self, a: TensorId, ne0: i64, offset: usize) -> TensorId {
        let r = self.view_of(a, [ne0, 1, 1, 1], offset);
        self.set_op_params_i32(
            r,
            &[offset as u32 as i32, (offset as u64 >> 32) as u32 as i32],
        );
        self.init_op(r, GgmlOp::View, [Some(a), None, None])
    }

    /// 对照 ggml_view_2d (ggml.c:3812)
    pub fn view_2d(
        &mut self,
        a: TensorId,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> TensorId {
        let r = self.view_of(a, [ne0, ne1, 1, 1], offset);
        {
            let t = &mut self.tensors[r.0 as usize];
            t.nb[1] = nb1 as u64;
            t.nb[2] = t.nb[1] * ne1 as u64;
            t.nb[3] = t.nb[2];
        }
        self.set_op_params_i32(
            r,
            &[offset as u32 as i32, (offset as u64 >> 32) as u32 as i32],
        );
        self.init_op(r, GgmlOp::View, [Some(a), None, None])
    }

    /// 对照 ggml_view_3d (ggml.c:3832)
    pub fn view_3d(
        &mut self,
        a: TensorId,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> TensorId {
        let r = self.view_of(a, [ne0, ne1, ne2, 1], offset);
        {
            let t = &mut self.tensors[r.0 as usize];
            t.nb[1] = nb1 as u64;
            t.nb[2] = nb2 as u64;
            t.nb[3] = t.nb[2] * ne2 as u64;
        }
        self.set_op_params_i32(
            r,
            &[offset as u32 as i32, (offset as u64 >> 32) as u32 as i32],
        );
        self.init_op(r, GgmlOp::View, [Some(a), None, None])
    }

    /// 对照 ggml_view_4d (ggml.c:3854)
    #[allow(clippy::too_many_arguments)]
    pub fn view_4d(
        &mut self,
        a: TensorId,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> TensorId {
        let r = self.view_of(a, [ne0, ne1, ne2, ne3], offset);
        {
            let t = &mut self.tensors[r.0 as usize];
            t.nb[1] = nb1 as u64;
            t.nb[2] = nb2 as u64;
            t.nb[3] = nb3 as u64;
        }
        self.set_op_params_i32(
            r,
            &[offset as u32 as i32, (offset as u64 >> 32) as u32 as i32],
        );
        self.init_op(r, GgmlOp::View, [Some(a), None, None])
    }

    /// 对照 ggml_reshape (ggml.c:3690): takes the shape from `b`.
    pub fn reshape(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let bne = self.tensors[b.0 as usize].ne;
        assert!(is_contiguous_ctx(self, a), "reshape: a must be contiguous");
        assert_eq!(
            self.tensors[a.0 as usize].n_elements(),
            self.tensors[b.0 as usize].n_elements(),
            "reshape: nelements mismatch"
        );
        let r = self.view_of(a, bne, 0);
        self.init_op(r, GgmlOp::Reshape, [Some(a), None, None])
    }

    /// 对照 ggml_reshape_1d (ggml.c:3707)
    pub fn reshape_1d(&mut self, a: TensorId, ne0: i64) -> TensorId {
        assert!(is_contiguous_ctx(self, a), "reshape_1d: a must be contiguous");
        assert_eq!(self.tensors[a.0 as usize].n_elements(), ne0);
        let r = self.view_of(a, [ne0, 1, 1, 1], 0);
        self.init_op(r, GgmlOp::Reshape, [Some(a), None, None])
    }

    /// 对照 ggml_reshape_2d (ggml.c:3724)
    pub fn reshape_2d(&mut self, a: TensorId, ne0: i64, ne1: i64) -> TensorId {
        assert!(is_contiguous_ctx(self, a), "reshape_2d: a must be contiguous");
        assert_eq!(self.tensors[a.0 as usize].n_elements(), ne0 * ne1);
        let r = self.view_of(a, [ne0, ne1, 1, 1], 0);
        self.init_op(r, GgmlOp::Reshape, [Some(a), None, None])
    }

    /// 对照 ggml_reshape_3d (ggml.c:3742)
    pub fn reshape_3d(&mut self, a: TensorId, ne0: i64, ne1: i64, ne2: i64) -> TensorId {
        assert!(is_contiguous_ctx(self, a), "reshape_3d: a must be contiguous");
        assert_eq!(self.tensors[a.0 as usize].n_elements(), ne0 * ne1 * ne2);
        let r = self.view_of(a, [ne0, ne1, ne2, 1], 0);
        self.init_op(r, GgmlOp::Reshape, [Some(a), None, None])
    }

    /// 对照 ggml_reshape_4d (ggml.c:3761)
    pub fn reshape_4d(&mut self, a: TensorId, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> TensorId {
        assert!(is_contiguous_ctx(self, a), "reshape_4d: a must be contiguous");
        assert_eq!(self.tensors[a.0 as usize].n_elements(), ne0 * ne1 * ne2 * ne3);
        let r = self.view_of(a, [ne0, ne1, ne2, ne3], 0);
        self.init_op(r, GgmlOp::Reshape, [Some(a), None, None])
    }

    /// 对照 ggml_permute (ggml.c:3878)
    pub fn permute(&mut self, a: TensorId, axis0: usize, axis1: usize, axis2: usize, axis3: usize) -> TensorId {
        let mut axes = [axis0, axis1, axis2, axis3];
        axes.sort_unstable();
        assert_eq!(axes, [0, 1, 2, 3], "permute axes must be a permutation");

        let r = self.view_tensor_of(a);
        {
            let at = &self.tensors[a.0 as usize];
            let ne = at.ne;
            let nb = at.nb;
            let t = &mut self.tensors[r.0 as usize];
            for (dst_axis, src_axis) in [(axis0, 0usize), (axis1, 1), (axis2, 2), (axis3, 3)] {
                t.ne[dst_axis] = ne[src_axis];
                t.nb[dst_axis] = nb[src_axis];
            }
        }
        self.set_op_params_i32(r, &[axis0 as i32, axis1 as i32, axis2 as i32, axis3 as i32]);
        self.init_op(r, GgmlOp::Permute, [Some(a), None, None])
    }

    /// 对照 ggml_transpose (ggml.c:3934)
    pub fn transpose(&mut self, a: TensorId) -> TensorId {
        let r = self.view_tensor_of(a);
        {
            let at = &self.tensors[a.0 as usize];
            let (ne0, ne1, nb0, nb1) = (at.ne[0], at.ne[1], at.nb[0], at.nb[1]);
            let t = &mut self.tensors[r.0 as usize];
            t.ne[0] = ne1;
            t.ne[1] = ne0;
            t.nb[0] = nb1;
            t.nb[1] = nb0;
        }
        self.init_op(r, GgmlOp::Transpose, [Some(a), None, None])
    }

    // ===================== argmax / argsort =====================

    /// 对照 ggml_argmax (ggml.c:2537)
    pub fn argmax(&mut self, a: TensorId) -> TensorId {
        let at = &self.tensors[a.0 as usize];
        assert!(at.is_matrix(), "argmax: a must be matrix");
        let ne1 = at.ne[1];
        let r = self.new_tensor_1d(GgmlType::I32, ne1);
        self.init_op(r, GgmlOp::ArgMax, [Some(a), None, None])
    }

    /// 对照 ggml_argsort (ggml.c:5423). order: GGML_SORT_ORDER_ASC/DESC.
    pub fn argsort(&mut self, a: TensorId, order: i32) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        let r = self.new_tensor(GgmlType::I32, ne);
        self.set_op_params_i32(r, &[order]);
        self.init_op(r, GgmlOp::Argsort, [Some(a), None, None])
    }

    // ===================== diag_mask_inf =====================

    /// 对照 ggml_diag_mask_inf (ggml.c:4062)
    pub fn diag_mask_inf(&mut self, a: TensorId, n_past: i32) -> TensorId {
        let r = self.dup_tensor_of(a);
        self.set_op_params_i32(r, &[n_past]);
        self.init_op(r, GgmlOp::DiagMaskInf, [Some(a), None, None])
    }
    /// 对照 ggml_diag_mask_inf_inplace (ggml.c:4069)
    pub fn diag_mask_inf_inplace(&mut self, a: TensorId, n_past: i32) -> TensorId {
        let r = self.view_tensor_of(a);
        self.set_op_params_i32(r, &[n_past]);
        self.init_op(r, GgmlOp::DiagMaskInf, [Some(a), None, None])
    }

    // ===================== concat / sum_rows / repeat =====================

    /// 对照 ggml_concat (ggml.c:2622). dim: 0..=3.
    pub fn concat(&mut self, a: TensorId, b: TensorId, dim: usize) -> TensorId {
        assert!(dim < MAX_DIMS, "concat dim");
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        assert_eq!(self.tensors[a.0 as usize].ty, self.tensors[b.0 as usize].ty, "concat types");
        let mut ne = [1i64; MAX_DIMS];
        for d in 0..MAX_DIMS {
            if d == dim {
                ne[d] = ane[d] + bne[d];
                continue;
            }
            assert_eq!(ane[d], bne[d], "concat shape mismatch on dim {d}");
            ne[d] = ane[d];
        }
        let ty = self.tensors[a.0 as usize].ty;
        let r = self.new_tensor(ty, ne);
        self.set_op_params_i32(r, &[dim as i32]);
        self.init_op(r, GgmlOp::Concat, [Some(a), Some(b), None])
    }

    /// 对照 ggml_sum_rows (ggml.c:2490)
    pub fn sum_rows(&mut self, a: TensorId) -> TensorId {
        let ane = self.tensors[a.0 as usize].ne;
        let ty = self.tensors[a.0 as usize].ty;
        let r = self.new_tensor(ty, [1, ane[1], ane[2], ane[3]]);
        self.init_op(r, GgmlOp::SumRows, [Some(a), None, None])
    }

    /// 对照 ggml_repeat (ggml.c:2570): repeat `a` to the shape of `b`.
    pub fn repeat(&mut self, a: TensorId, b: TensorId) -> TensorId {
        assert!(can_repeat(self, a, b), "repeat: !can_repeat");
        let bne = self.tensors[b.0 as usize].ne;
        let ty = self.tensors[a.0 as usize].ty;
        let r = self.new_tensor(ty, bne);
        self.init_op(r, GgmlOp::Repeat, [Some(a), None, None])
    }

    // ===================== flash_attn_ext =====================

    /// 对照 ggml_flash_attn_ext (ggml.c:5497).
    ///
    /// src = [q, k, v, mask]; q [DK, T, H, S], k/v [D, S_kv, H_kv, S],
    /// mask [S_kv, T, mask_ne2, mask_ne3] F16 contiguous (mask optional).
    /// Result [DV, H, T, S] F32 (builder's `permute(0,2,1,3)`, ggml.c:5526).
    /// op_params: [0]=scale, [1]=max_bias, [2]=logit_softcap (f32 bits),
    /// [3]=prec (`ggml_flash_attn_ext_set_prec`, default 0 = PREC_UNDEFINED),
    /// [4]=n_kv_max (0 = dense attention, ggml.c:5561). `sinks` (src[4]) is not
    /// representable/needed here.
    /// `sinks` (C `ggml_flash_attn_ext_add_sinks`, ggml.c:5560) is per-head
    /// [n_head] F32 stored in src[4]; the CPU kernel applies it on the first
    /// kv chunk (ops.cpp:8811, flash_attn.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_ext(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        mask: Option<TensorId>,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> TensorId {
        self.flash_attn_ext_sinks(q, k, v, mask, None, scale, max_bias, logit_softcap)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_ext_sinks(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        mask: Option<TensorId>,
        sinks: Option<TensorId>,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> TensorId {
        assert!(can_mul_mat(self, k, q), "ggml_flash_attn_ext: !can_mul_mat (k, q)");
        let (qne, vne) = (
            self.tensors[q.0 as usize].ne,
            self.tensors[v.0 as usize].ne,
        );
        assert_eq!(qne[3], self.tensors[k.0 as usize].ne[3], "ggml_flash_attn_ext: q->ne[3] == k->ne[3]");
        assert_eq!(qne[3], vne[3], "ggml_flash_attn_ext: q->ne[3] == v->ne[3]");
        if let Some(m) = mask {
            let mt = &self.tensors[m.0 as usize];
            assert_eq!(mt.ty, GgmlType::F16, "ggml_flash_attn_ext: mask must be F16");
            assert!(is_contiguous_ctx(self, m), "ggml_flash_attn_ext: mask must be contiguous");
            assert_eq!(qne[2] % mt.ne[2], 0, "ggml_flash_attn_ext: q->ne[2] % mask->ne[2]");
            assert_eq!(qne[3] % mt.ne[3], 0, "ggml_flash_attn_ext: q->ne[3] % mask->ne[3]");
        }
        if max_bias > 0.0 {
            assert!(mask.is_some(), "ggml_flash_attn_ext: max_bias > 0 requires mask");
        }
        // permute(0, 2, 1, 3)
        let r = self.new_tensor(GgmlType::F32, [vne[0], qne[2], qne[1], qne[3]]);
        let mut src = [None; crate::types::MAX_SRC];
        src[0] = Some(q);
        src[1] = Some(k);
        src[2] = Some(v);
        src[3] = mask;
        if let Some(sk) = sinks {
            let st = &self.tensors[sk.0 as usize];
            assert_eq!(st.ty, GgmlType::F32, "ggml_flash_attn_ext: sinks must be F32");
            assert_eq!(st.ne[0], qne[2], "ggml_flash_attn_ext: sinks->ne[0] == q->ne[2]");
        }
        src[4] = sinks;
        self.init_op(r, GgmlOp::FlashAttnExt, src);
        // params[] = { scale, max_bias, logit_softcap } (ggml.c:5529)
        self.params_f32(r, &[scale, max_bias, logit_softcap]);
        r
    }

    // ===================== ssm (mamba2 / shortconv) =====================

    /// 对照 ggml_ssm_conv (ggml.c:5659-5683) — causal 1D convolution.
    ///
    /// `sx` = conv_x {d_conv - 1 + n_t, d_inner, n_seqs} (F32, 3d, nb[1] ==
    /// ne[0]*4), `c` = conv1d.weight {d_conv, d_inner} (F32, matrix).
    /// Result {d_inner, n_t, n_s} F32. Bias is *not* part of this op — C adds it
    /// as a separate ggml_add (mamba-base.cpp:235).
    pub fn ssm_conv(&mut self, sx: TensorId, c: TensorId) -> TensorId {
        let (sx_ty, sx_ne, c_ne, c_ty) = {
            let t = &self.tensors[sx.0 as usize];
            let cc = &self.tensors[c.0 as usize];
            (t.ty, t.ne, cc.ne, cc.ty)
        };
        // C asserts ggml_is_3d(sx) (n_dims == 3) — TensorMeta does not carry
        // n_dims (all shapes are normalised to 4), so only the shape contract
        // is checked here.
        assert!(sx_ne[2] >= 1 && sx_ne[3] == 1, "ggml_ssm_conv: sx must be 3d");
        assert!(c_ne[2] == 1 && c_ne[3] == 1, "ggml_ssm_conv: c must be a matrix");
        assert_eq!(sx_ty, GgmlType::F32, "ggml_ssm_conv: F32 only");
        assert_eq!(c_ty, GgmlType::F32, "ggml_ssm_conv: F32 only");

        let d_conv = c_ne[0];
        let d_inner = c_ne[1];
        let n_t = sx_ne[0] - d_conv + 1; // tokens per sequence
        let n_s = sx_ne[2];
        assert_eq!(sx_ne[0], d_conv - 1 + n_t, "ggml_ssm_conv: sx->ne[0]");
        assert_eq!(sx_ne[1], d_inner, "ggml_ssm_conv: sx->ne[1] == d_inner");
        assert!(n_t >= 0, "ggml_ssm_conv: n_t >= 0");

        let r = self.new_tensor(GgmlType::F32, [d_inner, n_t, n_s, 1]);
        self.init_op(r, GgmlOp::SsmConv, [Some(sx), Some(c), None])
    }

    /// 对照 ggml_ssm_scan (ggml.c:5729-5790) — mamba2 selective scan.
    ///
    /// src = [s, x, dt, A, B, C, ids]; op_params[0] = K. Result is the
    /// concatenation of y (nelements(x)) and K state snapshots per sequence:
    /// `y + ssm_states` (ggml.c:5773).
    /// Shape rules (all asserted in C): x {head_dim, n_head, n_seq_tokens,
    /// n_seqs} dim0-contiguous; s {d_state, head_dim, n_head, n_slots}
    /// contiguous; dt {n_head, n_seq_tokens, n_seqs} 3d; A {d_state, n_head}
    /// (or {1, n_head} for the mamba2 scalar decay, then K must be 1);
    /// B/C same shape {d_state, n_group, n_seq_tokens, n_seqs}; ids I32
    /// {n_seqs} vector.
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_scan(
        &mut self,
        s: TensorId,
        x: TensorId,
        dt: TensorId,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        ids: TensorId,
        k: i64,
    ) -> TensorId {
        assert!(k >= 1 && k <= i32::MAX as i64, "ggml_ssm_scan: K out of range");
        let (s_ne, s_ty) = (self.tensors[s.0 as usize].ne, self.tensors[s.0 as usize].ty);
        let (x_ne, x_ty) = (self.tensors[x.0 as usize].ne, self.tensors[x.0 as usize].ty);
        let dt_ne = self.tensors[dt.0 as usize].ne;
        let a_ne = self.tensors[a.0 as usize].ne;
        let b_ne = self.tensors[b.0 as usize].ne;
        let c_ne = self.tensors[c.0 as usize].ne;
        let ids_ty = self.tensors[ids.0 as usize].ty;
        let ids_ne = self.tensors[ids.0 as usize].ne;

        assert!(is_contiguous_ctx(self, s), "ggml_ssm_scan: s must be contiguous");
        assert!(is_contiguous_ctx(self, dt), "ggml_ssm_scan: dt must be contiguous");
        assert!(is_contiguous_ctx(self, a), "ggml_ssm_scan: A must be contiguous");
        for (t, name) in [(x, "x"), (b, "B"), (c, "C")] {
            let tt = &self.tensors[t.0 as usize];
            assert_eq!(tt.nb[0], tt.ty.type_size() as u64, "ggml_ssm_scan: {name}->nb[0]");
            assert_eq!(tt.nb[1], tt.ne[0] as u64 * tt.nb[0], "ggml_ssm_scan: {name}->nb[1]");
        }
        assert_eq!(b_ne, c_ne, "ggml_ssm_scan: B/C must have the same shape");
        assert_eq!(ids_ty, GgmlType::I32, "ggml_ssm_scan: ids must be I32");

        let d_state = s_ne[0];
        let head_dim = x_ne[0];
        let n_head = x_ne[1];
        let n_seq_tokens = x_ne[2];
        let n_seqs = x_ne[3];
        assert_eq!(dt_ne[0], n_head, "ggml_ssm_scan: dt->ne[0]");
        assert_eq!(dt_ne[1], n_seq_tokens, "ggml_ssm_scan: dt->ne[1]");
        assert_eq!(dt_ne[2], n_seqs, "ggml_ssm_scan: dt->ne[2]");
        assert_eq!(dt_ne[3], 1, "ggml_ssm_scan: dt must be 3d");
        assert_eq!(s_ne[1], head_dim, "ggml_ssm_scan: s->ne[1]");
        assert_eq!(s_ne[2], n_head, "ggml_ssm_scan: s->ne[2]");
        assert_eq!(b_ne[0], d_state, "ggml_ssm_scan: B->ne[0]");
        assert_eq!(b_ne[2], n_seq_tokens, "ggml_ssm_scan: B->ne[2]");
        assert_eq!(b_ne[3], n_seqs, "ggml_ssm_scan: B->ne[3]");
        assert_eq!(ids_ne[0], n_seqs, "ggml_ssm_scan: ids->ne[0]");
        assert!(ids_ne[1] == 1 && ids_ne[2] == 1 && ids_ne[3] == 1, "ggml_ssm_scan: ids must be a vector");
        assert_eq!(a_ne[1], n_head, "ggml_ssm_scan: A->ne[1] == n_head");
        assert!(a_ne[2] == 1 && a_ne[3] == 1, "ggml_ssm_scan: A must be a matrix");
        if a_ne[0] != 1 {
            // Mamba-1 has more granular decay factors
            assert_eq!(a_ne[0], d_state, "ggml_ssm_scan: A->ne[0] == d_state");
            assert_eq!(k, 1, "ggml_ssm_scan: A->ne[0] != 1 requires K == 1");
        }
        assert_eq!(x_ty, GgmlType::F32, "ggml_ssm_scan: x must be F32");
        assert_eq!(s_ty, GgmlType::F32, "ggml_ssm_scan: s must be F32");

        let nelem_x = x_ne.iter().product::<i64>();
        let r = self.new_tensor(
            GgmlType::F32,
            [nelem_x + k * s_ne[0] * s_ne[1] * s_ne[2] * ids_ne[0], 1, 1, 1],
        );
        let mut src = [None; crate::types::MAX_SRC];
        src[0] = Some(s);
        src[1] = Some(x);
        src[2] = Some(dt);
        src[3] = Some(a);
        src[4] = Some(b);
        src[5] = Some(c);
        src[6] = Some(ids);
        self.init_op(r, GgmlOp::SsmScan, src);
        self.set_op_params_i32(r, &[k as i32]);
        r
    }

    /// 对照 ggml_swiglu_split (ggml.c:3061 → ggml_glu_impl with b != NULL):
    /// `a` and `b` are the *already split* halves, result = silu(a) * b with the
    /// shape of `a` (ggml.c:2918-2919: ne0 = a->ne0 since b is not NULL).
    /// op_params[0] = GGML_GLU_OP_SWIGLU, [1] = swapped (0).
    pub fn swiglu_split(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        assert_eq!(ane, bne, "ggml_swiglu_split: shape mismatch");
        let ty = self.tensors[a.0 as usize].ty;
        assert_eq!(ty, self.tensors[b.0 as usize].ty, "ggml_swiglu_split: type mismatch");
        let r = self.new_tensor(ty, ane);
        let params = [GGML_GLU_OP_SWIGLU, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Glu, [Some(a), Some(b), None])
    }

    /// 对照 ggml_geglu_split (ggml.c:3043 → ggml_glu_impl with b != NULL):
    /// result = gelu(a) * b (ggml-cpu/ops.cpp:3035 `ggml_compute_forward_geglu_f32`
    /// with src1 → `ggml_vec_geglu_f32`, the f16-table gelu). Used by the
    /// gemma4 MoE expert FFN (gemma4.cpp:190 `LLM_FFN_GELU` with gate_exps).
    pub fn geglu_split(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        assert_eq!(ane, bne, "ggml_geglu_split: shape mismatch");
        let ty = self.tensors[a.0 as usize].ty;
        assert_eq!(ty, self.tensors[b.0 as usize].ty, "ggml_geglu_split: type mismatch");
        assert_eq!(ty, GgmlType::F32, "ggml_geglu_split: F32 only");
        let r = self.new_tensor(ty, ane);
        let params = [GGML_GLU_OP_GEGLU, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Glu, [Some(a), Some(b), None])
    }

    /// 对照 ggml_reglu_split (ggml.c:3019 → ggml_glu_impl with b != NULL):
    /// result = relu(a) * b — `(x > 0) ? x * g : 0` (ops.cpp:2892
    /// `ggml_compute_forward_reglu_f32` → vec.h:1401 ggml_vec_reglu_f32, a
    /// plain scalar loop). Used by smallthinker's MoE experts
    /// (smallthinker.cpp:156 `LLM_FFN_RELU` + gate_exps).
    pub fn reglu_split(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        assert_eq!(ane, bne, "ggml_reglu_split: shape mismatch");
        let ty = self.tensors[a.0 as usize].ty;
        assert_eq!(ty, self.tensors[b.0 as usize].ty, "ggml_reglu_split: type mismatch");
        assert_eq!(ty, GgmlType::F32, "ggml_reglu_split: F32 only");
        let r = self.new_tensor(ty, ane);
        let params = [GGML_GLU_OP_REGLU, 0];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Glu, [Some(a), Some(b), None])
    }

    /// 对照 ggml_softplus (ggml.c:2860 → GGML_UNARY_OP_SOFTPLUS).
    /// The CPU kernel is `ggml_compute_softplus_f32` = `(x > 20) ? x :
    /// logf(1 + expf(x))` (ggml-impl.h:107, see `ssm::softplus_f32`).
    pub fn softplus(&mut self, a: TensorId) -> TensorId {
        let r = self.dup_tensor_of(a);
        let params = [GGML_UNARY_OP_SOFTPLUS];
        self.set_op_params_i32(r, &params);
        self.init_op(r, GgmlOp::Silu, [Some(a), None, None])
    }

    /// 对照 ggml_repeat_4d (ggml.c:2596): repeat `a` to the given shape. C
    /// builds a throwaway tensor carrying the target dims; the port does the
    /// same (the dummy is never a graph leaf, so it costs no storage).
    pub fn repeat_4d(&mut self, a: TensorId, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> TensorId {
        let ty = self.tensors[a.0 as usize].ty;
        let dummy = self.new_tensor(ty, [ne0, ne1, ne2, ne3]);
        self.repeat(a, dummy)
    }

    /// 对照 ggml_gated_delta_net (ggml.c:6364-6413) — the fused gated delta net
    /// (delta-net-base.cpp:831-874 `build_delta_net_fused`).
    ///
    /// shapes (S_k == S_v, H_v % H_k == 0):
    ///   q, k  : [S_k, H_k, n_tokens, n_seqs]   (contiguous rows)
    ///   v     : [S_v, H_v, n_tokens, n_seqs]
    ///   g     : [1, H_v, n_tokens, n_seqs] (scalar gate) or [S_v, H_v, ...] (KDA)
    ///   beta  : [1, H_v, n_tokens, n_seqs]
    ///   state : [S_v, S_v, H_v, n_seqs] (initial s0 only)
    /// result: [S_v*H_v, n_tokens*n_seqs + K*S_v*n_seqs] — attention scores
    /// followed by K state snapshots, most recent first (slot 0 = final state).
    /// Only K == 1 is ported (the graphs never request rollback snapshots).
    pub fn gated_delta_net(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        g: TensorId,
        beta: TensorId,
        state: TensorId,
        kk: i64,
    ) -> TensorId {
        let tt = |id: TensorId| (&self.tensors[id.0 as usize].ty, self.tensors[id.0 as usize].ne, self.tensors[id.0 as usize].nb);
        for (id, name) in [(q, "q"), (k, "k"), (v, "v"), (g, "g"), (beta, "beta"), (state, "state")] {
            let (ty, _, nb) = tt(id);
            assert_eq!(*ty, GgmlType::F32, "ggml_gated_delta_net: {name} must be F32");
            assert_eq!(nb[0], 4, "ggml_gated_delta_net: {name} nb[0] (contiguous rows)");
        }
        let (vne, gne, bne, sne) = (tt(v).1, tt(g).1, tt(beta).1, tt(state).1);
        let (s_v, h_v, n_tokens, n_seqs) = (vne[0], vne[1], vne[2], vne[3]);
        assert!(gne[0] == 1 || gne[0] == s_v, "ggml_gated_delta_net: g->ne[0]");
        assert_eq!(bne[0], 1, "ggml_gated_delta_net: beta->ne[0]");
        assert_eq!((sne[0], sne[1], sne[2], sne[3]), (s_v, s_v, h_v, n_seqs), "ggml_gated_delta_net: state shape");
        assert!(kk >= 1, "ggml_gated_delta_net: K >= 1");

        let state_rows = kk * s_v * n_seqs;
        let r = self.new_tensor(
            GgmlType::F32,
            [s_v * h_v, n_tokens * n_seqs + state_rows, 1, 1],
        );
        let mut src = [None; crate::types::MAX_SRC];
        src[0] = Some(q);
        src[1] = Some(k);
        src[2] = Some(v);
        src[3] = Some(g);
        src[4] = Some(beta);
        src[5] = Some(state);
        self.init_op(r, GgmlOp::Gdn, src);
        self.set_op_params_i32(r, &[kk as i32]);
        r
    }

    // ===================== rwkv wkv (batch 14 round 2, agent WKV) =====================

    /// The assertions every ggml_rwkv_wkv* builder makes for its {S, H, T}
    /// operands (ggml.c:5877-5892 / :5920-5935 / :5963-5992):
    /// ggml_is_contiguous plus the shape equalities against k's S/H/T.
    fn wkv_check(&self, s: i64, heads: i64, n_tokens: i64, ops: &[(&str, TensorId)]) {
        for (name, id) in ops {
            let t = &self.tensors[id.0 as usize];
            assert_eq!(t.ty, GgmlType::F32, "ggml_rwkv_wkv: {name} must be F32");
            // ggml_is_contiguous
            assert_eq!(t.nb[0], 4, "ggml_rwkv_wkv: {name} nb[0]");
            let mut want = t.ne[0] as u64 * 4;
            for i in 1..MAX_DIMS {
                if t.ne[i] != 1 {
                    assert_eq!(t.nb[i], want, "ggml_rwkv_wkv: {name} nb[{i}] (contiguous)");
                }
                want *= t.ne[i] as u64;
            }
            assert!(
                (t.ne[0], t.ne[1], t.ne[2]) == (s, heads, n_tokens),
                "ggml_rwkv_wkv: {name} shape {{{}, {}, {}}} != {{{s}, {heads}, {n_tokens}}}",
                t.ne[0], t.ne[1], t.ne[2]
            );
        }
    }

    /// 对照 ggml_rwkv_wkv6 (ggml.c:5873-5912 → GGML_OP_RWKV_WKV6). k/v/r/td
    /// are {S, H, T}, tf is {S, H}, state is {S*S*H, n_seqs}; the packed
    /// result is {S*H, T + S*n_seqs} — the T output rows followed by the new
    /// state. Kernel: ops.cpp:10413-10603 (wkv.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv_wkv6(
        &mut self,
        k: TensorId,
        v: TensorId,
        r: TensorId,
        tf: TensorId,
        td: TensorId,
        state: TensorId,
    ) -> TensorId {
        let (s, heads, n_tokens) = {
            let t = &self.tensors[k.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2])
        };
        let n_seqs = self.tensors[state.0 as usize].ne[1];
        self.wkv_check(s, heads, n_tokens, &[
            ("k", k), ("v", v), ("r", r), ("td", td),
        ]);
        // tf: {S, H} contiguous
        let t = &self.tensors[tf.0 as usize];
        assert_eq!((t.ne[0], t.ne[1]), (s, heads), "ggml_rwkv_wkv6: tf shape");
        assert_eq!(t.nb[0], 4, "ggml_rwkv_wkv6: tf nb[0]");
        assert_eq!(t.nb[1], (s * 4) as u64, "ggml_rwkv_wkv6: tf nb[1] (contiguous)");
        assert_eq!(
            self.tensors[state.0 as usize].n_elements(),
            s * s * heads * n_seqs,
            "ggml_rwkv_wkv6: nelements(state)"
        );
        let res = self.new_tensor(GgmlType::F32, [s * heads, n_tokens + s * n_seqs, 1, 1]);
        let src = [Some(k), Some(v), Some(r), Some(tf), Some(td), Some(state)];
        self.init_op(res, GgmlOp::RwkvWkv6, src)
    }

    /// 对照 ggml_gated_linear_attn (ggml.c:5916-5954 →
    /// GGML_OP_GATED_LINEAR_ATTN). k/v/q/g are {S, H, T}, state is
    /// {S*S*H, n_seqs}; `scale` rides op_params[0] as f32 bits. Kernel:
    /// ops.cpp:10623-11418 (wkv.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_linear_attn(
        &mut self,
        k: TensorId,
        v: TensorId,
        q: TensorId,
        g: TensorId,
        state: TensorId,
        scale: f32,
    ) -> TensorId {
        let (s, heads, n_tokens) = {
            let t = &self.tensors[k.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2])
        };
        let n_seqs = self.tensors[state.0 as usize].ne[1];
        self.wkv_check(s, heads, n_tokens, &[
            ("k", k), ("v", v), ("q", q), ("g", g),
        ]);
        assert_eq!(
            self.tensors[state.0 as usize].n_elements(),
            s * s * heads * n_seqs,
            "ggml_gated_linear_attn: nelements(state)"
        );
        let res = self.new_tensor(GgmlType::F32, [s * heads, n_tokens + s * n_seqs, 1, 1]);
        let src = [Some(k), Some(v), Some(q), Some(g), Some(state)];
        self.init_op(res, GgmlOp::GatedLinearAttn, src);
        self.params_f32(res, &[scale]);
        res
    }

    /// 对照 ggml_rwkv_wkv7 (ggml.c:5959-6000 → GGML_OP_RWKV_WKV7). r/w/k/v/a/b
    /// are {S, H, T}, state is {S*S*H, n_seqs}; the packed result is
    /// {S*H, T + S*n_seqs}. Kernel: ops.cpp:11422-11617 (wkv.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv_wkv7(
        &mut self,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        state: TensorId,
    ) -> TensorId {
        let (s, heads, n_tokens) = {
            let t = &self.tensors[k.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2])
        };
        let n_seqs = self.tensors[state.0 as usize].ne[1];
        self.wkv_check(s, heads, n_tokens, &[
            ("r", r), ("w", w), ("k", k), ("v", v), ("a", a), ("b", b),
        ]);
        assert_eq!(
            self.tensors[state.0 as usize].n_elements(),
            s * s * heads * n_seqs,
            "ggml_rwkv_wkv7: nelements(state)"
        );
        let res = self.new_tensor(GgmlType::F32, [s * heads, n_tokens + s * n_seqs, 1, 1]);
        let src = [Some(r), Some(w), Some(k), Some(v), Some(a), Some(b), Some(state)];
        self.init_op(res, GgmlOp::RwkvWkv7, src)
    }

    // ===================== vision: cast / im2col / conv_2d / interpolate =====================

    /// 对照 ggml_cast (ggml.c:1970). Same node as C: `ggml_dup_tensor(ctx, a,
    /// type)` + GGML_OP_CAST, which the CPU backend dispatches to
    /// `ggml_compute_forward_dup` (ggml-cpu.c:1883 / ops.cpp:526) — i.e. this
    /// port's GgmlOp::Dup with a different dst type (compute.rs forward_dup,
    /// `dup_flt` f32→f16 is the path the vision graph takes).
    pub fn cast(&mut self, a: TensorId, ty: GgmlType) -> TensorId {
        let ne = self.tensors[a.0 as usize].ne;
        let r = self.new_tensor(ty, ne);
        self.init_op(r, GgmlOp::Dup, [Some(a), None, None])
    }

    /// 对照 ggml_calc_conv_output_size (ggml.c:4539)
    fn calc_conv_output_size(ins: i64, ks: i64, s: i32, p: i32, d: i32) -> i64 {
        (ins + 2 * p as i64 - d as i64 * (ks - 1) - 1) / s as i64 + 1
    }

    /// 对照 ggml_im2col (ggml.c:5547).
    ///
    /// `a` = kernel [KW, KH, IC, OC], `b` = image [IW, IH, IC, N]; result
    /// [IC*KH*KW, OW, OH, N] (`dst_type` is what the caller wants to read it
    /// as — `ggml_conv_2d` picks F16 for an F32 kernel, ggml.c:5678).
    /// op_params = {s0, s1, p0, p1, d0, d1, is_2d} (ggml.c:4576).
    #[allow(clippy::too_many_arguments)]
    pub fn im2col(
        &mut self,
        a: TensorId,
        b: TensorId,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
        d0: i32,
        d1: i32,
        is_2d: bool,
        dst_type: GgmlType,
    ) -> TensorId {
        let (ane, bne) = (self.tensors[a.0 as usize].ne, self.tensors[b.0 as usize].ne);
        if is_2d {
            assert_eq!(ane[2], bne[2], "ggml_im2col: a->ne[2] == b->ne[2]");
        } else {
            assert_eq!(bne[1], ane[1], "ggml_im2col: b->ne[1] == a->ne[1]");
            assert_eq!(bne[3], 1, "ggml_im2col: b->ne[3] == 1");
        }
        let oh = if is_2d { Self::calc_conv_output_size(bne[1], ane[1], s1, p1, d1) } else { 1 };
        let ow = Self::calc_conv_output_size(bne[0], ane[0], s0, p0, d0);
        assert!(!is_2d || oh > 0, "b too small compared to a");
        assert!(ow > 0, "b too small compared to a");

        let ne = [
            if is_2d { ane[2] * ane[1] * ane[0] } else { ane[1] * ane[0] },
            ow,
            if is_2d { oh } else { bne[2] },
            if is_2d { bne[3] } else { 1 },
        ];
        let r = self.new_tensor(dst_type, ne);
        self.init_op(r, GgmlOp::Im2col, [Some(a), Some(b), None]);
        self.set_op_params_i32(r, &[s0, s1, p0, p1, d0, d1, if is_2d { 1 } else { 0 }]);
        r
    }

    /// 对照 ggml_conv_2d (ggml.c:5678) — im2col + mul_mat against the reshaped
    /// kernel, returned as [OW, OH, OC, N]. `a` = [KW, KH, IC, OC], `b` =
    /// [IW, IH, IC, N].
    #[allow(clippy::too_many_arguments)]
    pub fn conv_2d(
        &mut self,
        a: TensorId,
        b: TensorId,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
        d0: i32,
        d1: i32,
    ) -> TensorId {
        let a_ty = self.tensors[a.0 as usize].ty;
        // ggml.c:5681 — an F32 kernel reads the patches as F16
        let dst_type = if a_ty == GgmlType::Bf16 { GgmlType::F32 } else { GgmlType::F16 };
        let im2col = self.im2col(a, b, s0, s1, p0, p1, d0, d1, true, dst_type);
        let (in0, in1, in2, in3) = {
            let t = &self.tensors[im2col.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2], t.ne[3])
        };
        let (ane0, ane1, ane2, ane3) = {
            let t = &self.tensors[a.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2], t.ne[3])
        };
        // ggml_reshape_2d(im2col, ne0, ne1*ne2*ne3) × ggml_reshape_2d(a, ne0, ne3)
        let lhs = self.reshape_2d(im2col, in0, in1 * in2 * in3);
        let rhs = self.reshape_2d(a, ane0 * ane1 * ane2, ane3);
        let result = self.mul_mat(lhs, rhs);
        // [OW*OH*N, OC] -> [OW, OH, N, OC] -> permute -> [OW, OH, OC, N]
        let result = self.reshape_4d(result, in1, in2, in3, ane3);
        let result = self.permute(result, 0, 1, 3, 2);
        self.cont(result)
    }

    /// 对照 ggml_roll (ggml.c:5321 → GGML_OP_ROLL): circularly shift each dim
    /// by `shift{i}` (positive = data moves up in index space — the kernel
    /// reads from `i - s`). src must have contiguous rows (nb[0] == 4, F32).
    /// Kernel: ops.cpp:8328-8373.
    pub fn roll(&mut self, a: TensorId, s0: i32, s1: i32, s2: i32, s3: i32) -> TensorId {
        let t = &self.tensors[a.0 as usize];
        assert_eq!(t.nb[0], 4, "ggml_roll: contiguous rows (F32) required");
        assert!((s0.abs() as i64) < t.ne[0], "ggml_roll: |shift0| < ne[0]");
        assert!((s1.abs() as i64) < t.ne[1], "ggml_roll: |shift1| < ne[1]");
        assert!((s2.abs() as i64) < t.ne[2], "ggml_roll: |shift2| < ne[2]");
        assert!((s3.abs() as i64) < t.ne[3], "ggml_roll: |shift3| < ne[3]");
        let r = self.dup_tensor_of(a);
        self.init_op(r, GgmlOp::Roll, [Some(a), None, None]);
        self.set_op_params_i32(r, &[s0, s1, s2, s3]);
        r
    }

    /// 对照 ggml_conv_2d_direct (ggml.c:4947 → GGML_OP_CONV_2D, single node):
    /// `a` = kernel [KW, KH, IC, OC], `b` = input [W, H, C, N] →
    /// [OW, OH, OC, N]. dst type = b's type.
    ///
    /// The C kernel (ops.cpp:7084-7217) im2cols each output patch into a
    /// [patch, IC*KW*KH] scratch and contracts against the contiguous kernel
    /// via `ggml_call_mul_mat` — the SAME contiguous-2D mul_mat the composite
    /// `ggml_conv_2d` ends in (the patch and kernel k-orders are the identical
    /// linearization `ic*KW*KH + ky*KW + kx`), so this builder expresses it as
    /// the port's im2col(F16 patches) + mul_mat composition: bit-identical
    /// (proven by parity/pool-conform dump, see PARITY.md).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_2d_direct(
        &mut self,
        a: TensorId,
        b: TensorId,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
        d0: i32,
        d1: i32,
    ) -> TensorId {
        let a_ty = self.tensors[a.0 as usize].ty;
        let b_ty = self.tensors[b.0 as usize].ty;
        let bne = self.tensors[b.0 as usize].ne;
        let ane = self.tensors[a.0 as usize].ne;
        assert_eq!(ane[2], bne[2], "ggml_conv_2d_direct: a->ne[2] == b->ne[2]");
        let ow = Self::calc_conv_output_size(bne[0], ane[0], s0, p0, d0);
        let oh = Self::calc_conv_output_size(bne[1], ane[1], s1, p1, d1);
        let _ = (ow, oh);
        // the direct kernel's patch scratch element type IS the kernel type
        // (ops.cpp:7119 `kernel_type = src0->type` — an F32 kernel keeps F32
        // patches and accumulates via vec_dot_f32, unlike the composite
        // ggml_conv_2d which halves the patches to F16)
        let im2col = self.im2col(a, b, s0, s1, p0, p1, d0, d1, true, a_ty);
        let (in0, in1, in2, in3) = {
            let t = &self.tensors[im2col.0 as usize];
            (t.ne[0], t.ne[1], t.ne[2], t.ne[3])
        };
        let lhs = self.reshape_2d(im2col, in0, in1 * in2 * in3);
        let rhs = self.reshape_2d(a, ane[0] * ane[1] * ane[2], ane[3]);
        let result = self.mul_mat(lhs, rhs);
        // [OW*OH*N, OC] -> [OW, OH, N, OC] -> permute -> [OW, OH, OC, N]
        let result = self.reshape_4d(result, in1, in2, in3, ane[3]);
        let result = self.permute(result, 0, 1, 3, 2);
        // the mul_mat result is F32; the C direct dst is b's type (== the
        // graph's F32 everywhere these archs run) — cast only if ever needed
        if self.tensors[result.0 as usize].ty != b_ty {
            return self.cast(result, b_ty);
        }
        self.cont(result)
    }

    /// 对照 ggml_conv_2d_dw_direct (ggml.c:4907 → GGML_OP_CONV_2D_DW):
    /// depthwise 2-D convolution, `a` = per-channel kernels [KW, KH, 1, C],
    /// `b` = [W, H, C, N] → [OW, OH, C, N]. Kernel: ops.cpp:7524-7619
    /// (whcn — the contiguous layout every audio graph feeds).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_2d_dw_direct(
        &mut self,
        a: TensorId,
        b: TensorId,
        s0: i32,
        s1: i32,
        p0: i32,
        p1: i32,
        d0: i32,
        d1: i32,
    ) -> TensorId {
        let ane = self.tensors[a.0 as usize].ne;
        let bne = self.tensors[b.0 as usize].ne;
        assert_eq!(ane[2], 1, "ggml_conv_2d_dw_direct: a->ne[2] == 1");
        assert_eq!(ane[3], bne[2], "ggml_conv_2d_dw_direct: a->ne[3] == b->ne[2]");
        let ne = [
            Self::calc_conv_output_size(bne[0], ane[0], s0, p0, d0),
            Self::calc_conv_output_size(bne[1], ane[1], s1, p1, d1),
            bne[2],
            bne[3],
        ];
        let ty = self.tensors[b.0 as usize].ty;
        let r = self.new_tensor(ty, ne);
        self.init_op(r, GgmlOp::Conv2dDw, [Some(a), Some(b), None]);
        self.set_op_params_i32(r, &[s0, s1, p0, p1, d0, d1]);
        r
    }

    /// 对照 ggml_interpolate (ggml.c:5199, `ggml_interpolate_impl` at :5156).
    /// op_params[0] = `mode` (ggml_scale_mode + ggml_scale_flag bits);
    /// the CPU kernel is ggml_compute_forward_upscale_f32 (ops.cpp:7978),
    /// reached through GGML_OP_UPSCALE.
    pub fn interpolate(
        &mut self,
        a: TensorId,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        mode: u32,
    ) -> TensorId {
        let ty = self.tensors[a.0 as usize].ty;
        let r = self.new_tensor(ty, [ne0, ne1, ne2, ne3]);
        self.init_op(r, GgmlOp::Upscale, [Some(a), None, None]);
        self.set_op_params_i32(r, &[mode as i32]);
        r
    }
}

/// ggml_is_contiguous (ggml.c:1474 ggml_is_contiguous_m_n with m=0..4)
pub fn is_contiguous_ctx(ctx: &Context, t: TensorId) -> bool {
    let t = &ctx.tensors[t.0 as usize];
    let mut next_nb = t.ty.type_size() as u64;
    if t.ne[0] != t.ty.blck_size() as i64 && t.nb[0] != next_nb {
        return false;
    }
    next_nb *= (t.ne[0] / t.ty.blck_size() as i64) as u64;
    for i in 1..MAX_DIMS {
        if t.ne[i] != 1 && t.nb[i] != next_nb {
            return false;
        }
        next_nb *= t.ne[i] as u64;
    }
    true
}

/// ggml_is_contiguous_rows (ggml.c:1539)
pub fn is_contiguous_rows_ctx(ctx: &Context, t: TensorId) -> bool {
    let t = &ctx.tensors[t.0 as usize];
    t.ne[0] == t.ty.blck_size() as i64 || t.nb[0] == t.ty.type_size() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expf_matches_libm_closely() {
        // The polynomial must track libm expf within a few ulps.
        for i in -60..60 {
            let x = i as f32 * 0.9;
            let a = ggml_expf(x);
            let b = x.exp();
            let rel = ((a - b) / b).abs();
            assert!(rel < 1e-5, "expf({x}) = {a} vs {b}");
        }
        assert_eq!(ggml_expf(-200.0), 0.0);
        assert!(ggml_expf(200.0).is_infinite());
    }

    #[test]
    fn op_params_layout() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(GgmlType::F32, 8, 2, 4); // [dim, head, seq]
        let pos = ctx.new_tensor_1d(GgmlType::I32, 4);
        let r = ctx.rope_ext(a, pos, None, 8, GGML_ROPE_TYPE_NEOX, 4096, 1_000_000.0, 1.0, 0.0, 1.0, 32.0, 1.0);
        let p = ctx.op_params(r);
        assert_eq!(p[0], 0); // n_past
        assert_eq!(p[1], 8); // n_dims
        assert_eq!(p[2], GGML_ROPE_TYPE_NEOX); // mode
        assert_eq!(p[3], 0); // n_ctx
        assert_eq!(p[4], 4096); // n_ctx_orig
        assert_eq!(f32::from_bits(p[5] as u32), 1_000_000.0);
        assert_eq!(f32::from_bits(p[6] as u32), 1.0);
        assert_eq!(f32::from_bits(p[7] as u32), 0.0);
        assert_eq!(f32::from_bits(p[8] as u32), 1.0);
        assert_eq!(f32::from_bits(p[9] as u32), 32.0);
        assert_eq!(f32::from_bits(p[10] as u32), 1.0);
        assert_eq!(&p[11..15], &[0, 0, 0, 0]); // sections
        assert_eq!(p[15], 0); // n_offs
        assert_eq!(ctx.op(r), GgmlOp::RoPE);

        // softmax params
        let s = ctx.soft_max_ext(a, None, 0.125, 8.0);
        let p = ctx.op_params(s);
        assert_eq!(f32::from_bits(p[0] as u32), 0.125);
        assert_eq!(f32::from_bits(p[1] as u32), 8.0);

        // rms_norm flag encoding
        let n = ctx.rms_norm(a, 1e-5);
        assert_eq!(ctx.op(n), GgmlOp::Norm);
        assert_eq!(f32::from_bits(ctx.op_params(n)[0] as u32), 1e-5);
        assert_eq!(ctx.op_params(n)[1], OP_FLAG_NORM_IS_RMS);
        let n2 = ctx.norm(a, 1e-5);
        assert_eq!(ctx.op_params(n2)[1], 0);
    }

    #[test]
    fn builders_shapes() {
        let mut ctx = Context::new();
        let w = ctx.new_tensor_2d(GgmlType::F32, 16, 8); // [16, 8]
        let x = ctx.new_tensor_2d(GgmlType::F32, 16, 3); // [16, 3]
        let mm = ctx.mul_mat(w, x);
        assert_eq!(ctx.ne(mm), &[8, 3, 1, 1]);

        // views keep parent storage
        let v = ctx.view_2d(w, 16, 4, 16 * 4, 0);
        assert_eq!(ctx.ne(v), &[16, 4, 1, 1]);
        assert!(ctx.tensors[v.0 as usize].view_src.is_some());

        let t = ctx.transpose(w);
        assert_eq!(ctx.ne(t), &[8, 16, 1, 1]);
        assert_eq!(ctx.nb(t)[0], 16 * 4);
        assert_eq!(ctx.nb(t)[1], 4);

        let p = ctx.permute(x, 1, 0, 2, 3);
        assert_eq!(ctx.ne(p), &[3, 16, 1, 1]);
        assert_eq!(ctx.op_params(p)[..4], [1, 0, 2, 3]);

        let c = ctx.concat(w, w, 1);
        assert_eq!(ctx.ne(c), &[16, 16, 1, 1]);

        let sr = ctx.sum_rows(x);
        assert_eq!(ctx.ne(sr), &[1, 3, 1, 1]);

        let ids = ctx.new_tensor_1d(GgmlType::I32, 5);
        let g = ctx.get_rows(w, ids);
        assert_eq!(ctx.ne(g), &[16, 5, 1, 1]);
    }
}
