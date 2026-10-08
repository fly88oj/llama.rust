//! wkv.rs — the three fused RWKV attention kernels.
//!
//! 对照 ggml/src/ggml-cpu/ops.cpp (pinned worktree @ bd4f514db1):
//!   * `ggml_compute_forward_rwkv_wkv6_f32` (ops.cpp:10413-10603) —
//!     GGML_OP_RWKV_WKV6 (ggml.c:5873), the rwkv6 / rwkv6qwen2… no: the
//!     rwkv6 WKV;
//!   * `ggml_compute_forward_gla_f32`       (ops.cpp:10623-11418) —
//!     GGML_OP_GATED_LINEAR_ATTN (ggml.c:5916), the rwkv6qwen2 WKV;
//!   * `ggml_compute_forward_rwkv_wkv7_f32` (ops.cpp:11422-11617) —
//!     GGML_OP_RWKV_WKV7 (ggml.c:5959), the rwkv7 / arwkv7 WKV.
//!
//! Geometry (all three kernels, identical): k/v/r(/q/g)/td(/w/a/b) are the
//! contiguous {S, H, T} tensors the graphs hand the op (asserted contiguous by
//! the builders, ggml.c:5877-5882 etc.), indexed FLAT with
//! `t_stride = HEADS*head_size = C` — element (t, h, d) at `t*C + h*S + d`;
//! `tf` is {S, H}; `state` is {S*S*H, n_seqs} (build_rs's reshape,
//! llama-graph.cpp:3497) — element (seq, h, i, j) at
//! `seq*S*S*H + h*S*S + i*S + j`. dst packs the output rows first
//! ({C, T}, zeroed by `ith == 0` + barrier in wkv6/gla) and then the new state
//! (`dst + C*T`, `state_offset = head_size*C*(t/(T/n_seqs))`); `state_prev`
//! is the dst state area for every token after the first of a sequence
//! (`t % (T/n_seqs) != 0`) and the src state tensor otherwise — the in-place
//! recurrence.
//!
//! Numerics: the reference build is `-march=native` on an AVX512F host, so
//! the kernels run their widest bodies: wkv6/gla the `WKV_VECTOR_SIZE ==
//! GLA_VECTOR_SIZE == 16` branch (GGML_F32x16 macros, simd-mappings.h:447-495)
//! with an explicit scalar tail for `head_size % 16`, wkv7 the generic
//! `GGML_F32_VEC` x86 branch (`GGML_F32_STEP 64 / GGML_F32_EPR 16 /
//! GGML_F32_ARR 4`, simd-mappings.h:452-453,1313). Every `GGML_F32_VEC_FMA`
//! is `_mm512_fmadd_ps(b, c, a)` — a single-rounded fused multiply-add —
//! reproduced here as `f32::mul_add` in the spec (scalar) bodies and as real
//! `vfmadd` in the `#[target_feature(enable = "avx512f")]` bodies; the two
//! are bit-identical (mul_add is always fused) and `wkv_simd_equals_scalar`
//! pins that.
//!
//! **The gla scalar tail's contraction is the OPPOSITE of the vec body's.**
//! The vec body's intrinsics pin `kv = v·k` as a separate multiply and fuse
//! `prev·g + kv`. GCC compiles the tail's C (`kv_val + prev_state_val *
//! g_val`) by fusing the LEFT product instead: `temp = fma(v, k, prev·g)`
//! with `prev·g` a separately-rounded multiply (verified by disassembling
//! the reference lib's rolled tail — `vmulss` prev·g then `vfmadd231ss`
//! v·k+that — and by gdb operand traces; both port tails carry this
//! pairing and the dump's S=20 section is bit-exact). wkv6's tail fuses the
//! left product too (`kv·tf + prev` — which is also its written form), so
//! its port tail needed no such correction.
//!
//! **wkv7 head_size < 64 is UB in the reference — not a convention to
//! reproduce.** The x86 body steps `j += GGML_F32_STEP` (64) with an
//! unbounded inner `kk < GGML_F32_ARR` (4) register sweep
//! (ops.cpp:11514-11556), so each (head, row, token) loads a full 64-float
//! window of r/w/k/a/b and the state row regardless of head_size. For
//! head_size 64 (every public RWKV7 GGUF) the window provably never leaves
//! the head row (`i*S + 63 ≤ S²−1 ⟺ S ≥ 64`) — fully deterministic,
//! bit-exact, and the only shape the dump's wkv7 ladder pins. For head_size
//! < 64 the window reads AND writes past the tensors into ggml context-pool
//! memory: 32-byte `ggml_object` headers (whose `next` field is a heap
//! pointer — one float lane reads an ASLR-dependent value) and, past dst,
//! the graph/hash-set objects, which the kernel itself overwrites with NaNs
//! mid-recurrence. The reference's sub-64 output therefore differs from
//! process run to process run — `parity/wkv7_oob_proof.c` prints the
//! per-run checksum table (S=64 constant across runs; S=16/48/32 all
//! different, mostly NaN) — so no bit-exact oracle can exist and the port
//! keeps its deterministic zero-fill approximation (out-of-tensor reads →
//! 0.0, writes past the state area → dropped) for those shapes only. The
//! in-tensor part of the overrun (the cross-head row mixing) is reproduced
//! exactly.
//!
//! Threading: the C kernels shard `HEADS` via
//! `h_start = HEADS*ith/nth, h_end = min(HEADS*(ith+1)/nth, HEADS)`
//! (ops.cpp:10434-10437) — contiguous ascending per-thread head ranges; the
//! port hands the same body `h0..h1` ranges through `par_rows`'s
//! `ceil(HEADS/nth)` split (identical ranges whenever HEADS % nth == 0, and
//! contiguous-ascending either way, so the nth == 1 order — the only one the
//! wkv7 sub-64 overrun makes observable — is preserved).

use crate::vec_dot::reduce_add16;

/// The AVX512 F32 geometry of the reference build for the wkv7 kernel
/// (simd-mappings.h:452-453): 64 elements per step, 16 per register, 4
/// registers — the unrolled `kk` sweep the x86 body runs per `j` step.
pub const F32_STEP: usize = 64;
pub const F32_EPR: usize = 16;
pub const F32_ARR: usize = 4;

/// The wkv6/gla chunk width (`WKV_VECTOR_SIZE`/`GLA_VECTOR_SIZE`, ops.cpp:10452
/// /10662): GGML_F32x16 on the AVX512 reference build.
const WKV_VECTOR_SIZE: usize = 16;

// ---------------------------------------------------------------------
// shared [f32; 16] spec bodies — the lane-exact reference for the intrinsics
// ---------------------------------------------------------------------

/// `GGML_F32x16_FMA(a, b, c) = _mm512_fmadd_ps(b, c, a)` = `b*c + a`
/// (simd-mappings.h:460). Argument order kept identical to the C call sites.
#[inline(always)]
fn fma16(a: [f32; 16], b: [f32; 16], c: [f32; 16]) -> [f32; 16] {
    let mut r = a;
    for l in 0..16 {
        r[l] = b[l].mul_add(c[l], a[l]);
    }
    r
}

/// `GGML_F32x16_MUL` (ops.cpp's `GGML_F32X_MUL`).
#[inline(always)]
fn mul16(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut r = a;
    for l in 0..16 {
        r[l] = a[l] * b[l];
    }
    r
}

/// `GGML_F32x16_SET1`.
#[inline(always)]
fn set116(x: f32) -> [f32; 16] {
    [x; 16]
}

/// 16-float load with the OOB convention of the wkv7 window: lanes at
/// `idx + l >= len` read 0.0 (the zero-guard convention the dump pins; see
/// the module header).
#[inline(always)]
fn load16z(p: *const f32, idx: usize, len: usize) -> [f32; 16] {
    let mut v = [0f32; 16];
    if idx + 16 <= len {
        // SAFETY: caller guarantees the 64 readable bytes at p+idx.
        unsafe { std::ptr::copy_nonoverlapping(p.add(idx), v.as_mut_ptr(), 16) };
    } else {
        for l in 0..16 {
            if idx + l < len {
                // SAFETY: idx + l < len
                v[l] = unsafe { *p.add(idx + l) };
            }
        }
    }
    v
}

/// 16-float store dropping the lanes past `len` (the wkv7 window's state
/// writes past the state area — see the module header).
#[inline(always)]
fn store16z(p: *mut f32, idx: usize, len: usize, v: [f32; 16]) {
    if idx + 16 <= len {
        // SAFETY: caller guarantees the 64 writable bytes at p+idx.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p.add(idx), 16) };
    } else {
        for l in 0..16 {
            if idx + l < len {
                // SAFETY: idx + l < len
                unsafe { *p.add(idx + l) = v[l] };
            }
        }
    }
}

/// `GGML_F32x16_REDUCE(res, x)` for the 4-register wkv7 accumulators
/// (simd-mappings.h:466-480): pairwise `x[0]+=x[2]`, `x[1]+=x[3]`,
/// `x[0]+=x[1]`, then `_mm512_reduce_add_ps(x[0])` — the pinned
/// `t[i] += t[i+half]` fold for half in {8,4,2,1} (vec_dot.rs `reduce_add16`).
#[inline(always)]
fn reduce4(sum: &[[f32; 16]; 4]) -> f32 {
    let mut x0 = sum[0];
    let mut x1 = sum[1];
    for l in 0..16 {
        x0[l] += sum[2][l];
    }
    for l in 0..16 {
        x1[l] += sum[3][l];
    }
    for l in 0..16 {
        x0[l] += x1[l];
    }
    reduce_add16(&x0)
}

/// Where the dst state area and the src state tensor start / how long they
/// are, precomputed by the dispatch (per-op constants).
#[derive(Clone, Copy)]
struct WkvGeom {
    c: usize,       // C = S*H (t_stride)
    s: usize,       // head_size
    heads: usize,   // H
    t: usize,       // T
    n_seqs: usize,  // state->ne[1]
    per_seq: usize, // T / n_seqs
    src_len: usize, // S*H*T — the {S,H,T} tensors' element count
    state_len: usize, // S*S*H*n_seqs
}

impl WkvGeom {
    fn new(s: usize, heads: usize, t: usize, n_seqs: usize) -> Self {
        WkvGeom {
            c: s * heads,
            s,
            heads,
            t,
            n_seqs,
            per_seq: t / n_seqs,
            src_len: s * heads * t,
            state_len: s * s * heads * n_seqs,
        }
    }

    /// `state_offset = head_size * C * (t / (T / n_seqs))` (ops.cpp:10481).
    #[inline(always)]
    fn state_offset(&self, t: usize) -> usize {
        self.s * self.c * (t / self.per_seq)
    }
}

// ---------------------------------------------------------------------
// ggml_compute_forward_rwkv_wkv6_f32 (ops.cpp:10413-10603)
// ---------------------------------------------------------------------

/// The kernel body for one thread's head range `h0..h1` (C's
/// `h_start..h_end`). The `ith == 0` memset of dst's output area plus the
/// barrier is the *caller's* (dispatch) job — same happens-before.
///
/// SAFETY: `dst` must point at `C*T + S*S*H*n_seqs` writable f32s; k/v/r/td
/// at `S*H*T` readable f32s each; tf at `S*H`; state at `S*S*H*n_seqs`. All
/// must stay live for the call. Concurrent calls must carry disjoint head
/// ranges (per-head state blocks and dst rows are disjoint).
#[allow(clippy::too_many_arguments)]
pub unsafe fn rwkv_wkv6_f32(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    r: *const f32,
    tf: *const f32,
    td: *const f32,
    state: *const f32,
    h0: usize,
    h1: usize,
    s: usize,
    heads: usize,
    t_len: usize,
    n_seqs: usize,
) {
    let g = WkvGeom::new(s, heads, t_len, n_seqs);
    let dst_state = dst.add(g.c * g.t);
    if crate::simd_x86::avx512() {
        #[cfg(target_arch = "x86_64")]
        {
            rwkv_wkv6_avx512(dst, k, v, r, tf, td, state, dst_state, h0, h1, &g);
            return;
        }
    }
    rwkv_wkv6_scalar(dst, k, v, r, tf, td, state, dst_state, h0, h1, &g);
}

/// The spec body: C's `#ifdef WKV_VECTOR_SIZE` branch with
/// WKV_VECTOR_SIZE = 16, lanes as `[f32; 16]` + `mul_add`.
#[allow(clippy::too_many_arguments)]
unsafe fn rwkv_wkv6_scalar(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    r: *const f32,
    tf: *const f32,
    td: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    h0: usize,
    h1: usize,
    g: &WkvGeom,
) {
    let vec_count = g.s / WKV_VECTOR_SIZE;
    for t in 0..g.t {
        let t_offset = t * g.c; // ops.cpp:10480
        let state_offset = g.state_offset(t);
        // SAFETY: pointer arithmetic stays inside dst (state area) / state.
        let state_cur = dst_state.add(state_offset);
        let state_prev: *const f32 = if t % g.per_seq != 0 {
            state_cur
        } else {
            state.add(state_offset)
        };
        for h in h0..h1 {
            let h_offset = h * g.s;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * g.s * g.s;
            for i in 0..g.s {
                let t_h_i_offset = t_h_offset + i;
                let h_i_offset = h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * g.s;

                // SAFETY: all scalar reads in-bounds by construction.
                let k_val = *k.add(t_h_i_offset);
                let r_val = *r.add(t_h_i_offset);
                let time_faaaa_val = *tf.add(h_i_offset);
                // RWKV v6: different time_decay for each token.
                let time_decay_val = *td.add(t_h_i_offset);

                let k_vec = set116(k_val);
                let r_vec = set116(r_val);
                let tff_vec = set116(time_faaaa_val);
                let td_vec = set116(time_decay_val);

                for j in 0..vec_count {
                    let base_j = j * WKV_VECTOR_SIZE;
                    let t_h_j_offset = t_h_offset + base_j;
                    let h_2d_i_j_offset = h_2d_i_offset + base_j;

                    // SAFETY: base_j + 16 <= s (vec_count chunks), every
                    // access below stays inside the {S,H,T} plane / head
                    // block / dst row.
                    let v_vec: [f32; 16] = std::ptr::read(v.add(t_h_j_offset) as *const _);
                    let prev_state_vec: [f32; 16] =
                        std::ptr::read(state_prev.add(h_2d_i_j_offset) as *const _);
                    let mut dst_vec: [f32; 16] =
                        std::ptr::read(dst.add(t_h_j_offset) as *const _);

                    // kv = v * k
                    let kv_vec = mul16(v_vec, k_vec);
                    // temp = kv * time_faaaa + prev_state
                    let temp_vec = fma16(prev_state_vec, kv_vec, tff_vec);
                    // dst += temp * r
                    dst_vec = fma16(dst_vec, temp_vec, r_vec);
                    std::ptr::write(dst.add(t_h_j_offset) as *mut _, dst_vec);
                    // state = prev_state * time_decay + kv
                    let new_state_vec = fma16(kv_vec, prev_state_vec, td_vec);
                    std::ptr::write(state_cur.add(h_2d_i_j_offset) as *mut _, new_state_vec);
                }

                // Handle remaining elements (ops.cpp:10562-10574) — GCC's
                // -ffp-contract=fast folds each a*b+c into one fma.
                for j in (vec_count * WKV_VECTOR_SIZE)..g.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    // SAFETY: j < s.
                    let v_val = *v.add(t_h_j_offset);
                    let kv_val = v_val * k_val;
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let temp_val = kv_val.mul_add(time_faaaa_val, prev_state_val);
                    *dst.add(t_h_j_offset) = temp_val.mul_add(r_val, *dst.add(t_h_j_offset));
                    *state_cur.add(h_2d_i_j_offset) =
                        prev_state_val.mul_add(time_decay_val, kv_val);
                }
            }
        }
    }
}

/// The `#[target_feature]` AVX512 body — same lane structure, real
/// `vfmadd`/`vmulps`/`vaddps` (bit-identical to the spec body; pinned by
/// `wkv_simd_equals_scalar`).
///
/// SAFETY: as `rwkv_wkv6_scalar`; additionally the caller must have checked
/// `avx512()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn rwkv_wkv6_avx512(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    r: *const f32,
    tf: *const f32,
    td: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    h0: usize,
    h1: usize,
    g: &WkvGeom,
) {
    use core::arch::x86_64::*;
    let vec_count = g.s / WKV_VECTOR_SIZE;
    for t in 0..g.t {
        let t_offset = t * g.c;
        let state_offset = g.state_offset(t);
        let state_cur = dst_state.add(state_offset);
        let state_prev: *const f32 = if t % g.per_seq != 0 { state_cur } else { state.add(state_offset) };
        for h in h0..h1 {
            let h_offset = h * g.s;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * g.s * g.s;
            for i in 0..g.s {
                let t_h_i_offset = t_h_offset + i;
                let h_i_offset = h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * g.s;

                // SAFETY: scalar reads in-bounds.
                let k_vec = _mm512_set1_ps(*k.add(t_h_i_offset));
                let r_vec = _mm512_set1_ps(*r.add(t_h_i_offset));
                let tff_vec = _mm512_set1_ps(*tf.add(h_i_offset));
                let td_vec = _mm512_set1_ps(*td.add(t_h_i_offset));

                for j in 0..vec_count {
                    let base_j = j * WKV_VECTOR_SIZE;
                    let t_h_j_offset = t_offset + h_offset + base_j;
                    let h_2d_i_j_offset = h_2d_offset + i * g.s + base_j;

                    // SAFETY: 16-lane windows inside the head row / block.
                    let v_vec = _mm512_loadu_ps(v.add(t_h_j_offset));
                    let prev_state_vec = _mm512_loadu_ps(state_prev.add(h_2d_i_j_offset));
                    let mut dst_vec = _mm512_loadu_ps(dst.add(t_h_j_offset));

                    // GGML_F32x16_FMA(a, b, c) == _mm512_fmadd_ps(b, c, a)
                    let kv_vec = _mm512_mul_ps(v_vec, k_vec);
                    let temp_vec = _mm512_fmadd_ps(kv_vec, tff_vec, prev_state_vec);
                    dst_vec = _mm512_fmadd_ps(temp_vec, r_vec, dst_vec);
                    _mm512_storeu_ps(dst.add(t_h_j_offset), dst_vec);
                    let new_state_vec = _mm512_fmadd_ps(prev_state_vec, td_vec, kv_vec);
                    _mm512_storeu_ps(state_cur.add(h_2d_i_j_offset), new_state_vec);
                }

                // scalar tail: identical to the spec body (fmaf contraction)
                for j in (vec_count * WKV_VECTOR_SIZE)..g.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    // SAFETY: j < s.
                    let v_val = *v.add(t_h_j_offset);
                    let k_val = *k.add(t_h_i_offset);
                    let r_val = *r.add(t_h_i_offset);
                    let time_faaaa_val = *tf.add(h_i_offset);
                    let time_decay_val = *td.add(t_h_i_offset);
                    let kv_val = v_val * k_val;
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let temp_val = kv_val.mul_add(time_faaaa_val, prev_state_val);
                    *dst.add(t_h_j_offset) = temp_val.mul_add(r_val, *dst.add(t_h_j_offset));
                    *state_cur.add(h_2d_i_j_offset) =
                        prev_state_val.mul_add(time_decay_val, kv_val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// ggml_compute_forward_gla_f32 (ops.cpp:10623-11418)
// ---------------------------------------------------------------------

/// GGML_OP_GATED_LINEAR_ATTN body for one thread's head range. The dst
/// memset is the caller's job.
///
/// SAFETY: as `rwkv_wkv6_f32` (q/g instead of tf/td).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gla_f32(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    q: *const f32,
    g: *const f32,
    state: *const f32,
    scale: f32,
    h0: usize,
    h1: usize,
    s: usize,
    heads: usize,
    t_len: usize,
    n_seqs: usize,
) {
    let g_geom = WkvGeom::new(s, heads, t_len, n_seqs);
    let dst_state = dst.add(g_geom.c * g_geom.t);
    if crate::simd_x86::avx512() {
        #[cfg(target_arch = "x86_64")]
        {
            gla_avx512(dst, k, v, q, g, state, dst_state, scale, h0, h1, &g_geom);
            return;
        }
    }
    gla_scalar(dst, k, v, q, g, state, dst_state, scale, h0, h1, &g_geom);
}

/// The spec body (GLA_VECTOR_SIZE = 16 branch, ops.cpp:10704-11390).
#[allow(clippy::too_many_arguments)]
unsafe fn gla_scalar(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    q: *const f32,
    g: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    scale: f32,
    h0: usize,
    h1: usize,
    geom: &WkvGeom,
) {
    let vec_count = geom.s / WKV_VECTOR_SIZE;
    for t in 0..geom.t {
        let t_offset = t * geom.c;
        let state_offset = geom.state_offset(t);
        // SAFETY: pointer arithmetic inside dst's state area / state tensor.
        let state_cur = dst_state.add(state_offset);
        let state_prev: *const f32 = if t % geom.per_seq != 0 {
            state_cur
        } else {
            state.add(state_offset)
        };
        for h in h0..h1 {
            let h_offset = h * geom.s;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * geom.s * geom.s;
            for i in 0..geom.s {
                let t_h_i_offset = t_h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * geom.s;

                // SAFETY: scalar reads in-bounds.
                let k_val = *k.add(t_h_i_offset);
                let q_val = *q.add(t_h_i_offset) * scale; // ops.cpp:10729
                let g_val = *g.add(t_h_i_offset);

                let k_vec = set116(k_val);
                let q_vec = set116(q_val);
                let g_vec = set116(g_val);

                for j in 0..vec_count {
                    let base_j = j * WKV_VECTOR_SIZE;
                    let t_h_j_offset = t_h_offset + base_j;
                    let h_2d_i_j_offset = h_2d_i_offset + base_j;

                    // SAFETY: windows inside the head row / block.
                    let v_vec: [f32; 16] = std::ptr::read(v.add(t_h_j_offset) as *const _);
                    let prev_state_vec: [f32; 16] =
                        std::ptr::read(state_prev.add(h_2d_i_j_offset) as *const _);
                    let mut dst_vec: [f32; 16] = std::ptr::read(dst.add(t_h_j_offset) as *const _);

                    let kv_vec = mul16(v_vec, k_vec);
                    // temp = prev_state * g + kv
                    let temp_vec = fma16(kv_vec, prev_state_vec, g_vec);
                    // dst += temp * q
                    dst_vec = fma16(dst_vec, temp_vec, q_vec);
                    std::ptr::write(dst.add(t_h_j_offset) as *mut _, dst_vec);
                    // state = temp
                    std::ptr::write(state_cur.add(h_2d_i_j_offset) as *mut _, temp_vec);
                }

                // scalar tail (ops.cpp:10759-10770). GCC's contraction of
                // `kv_val + prev_state_val * g_val` fuses the LEFT product —
                // the v*k multiply rides the fma and prev*g stays a separate
                // rounded multiply (verified against the reference's compiled
                // rolled tail: vmulss prev·g, then vfmadd231ss v·k + that;
                // the gdb trace lives in the PARITY.md batch-14 notes). The
                // vec body's explicit intrinsics pair the OTHER way (kv =
                // separate MUL, prev*g fused) — the two bodies genuinely
                // differ, and the dump pins both.
                for j in (vec_count * WKV_VECTOR_SIZE)..geom.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    // SAFETY: j < s.
                    let v_val = *v.add(t_h_j_offset);
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let pg_val = prev_state_val * g_val;
                    let temp_val = v_val.mul_add(k_val, pg_val);
                    *dst.add(t_h_j_offset) = temp_val.mul_add(q_val, *dst.add(t_h_j_offset));
                    *state_cur.add(h_2d_i_j_offset) = temp_val;
                }
            }
        }
    }
}

/// SAFETY: as `gla_scalar`; additionally the caller must have checked
/// `avx512()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gla_avx512(
    dst: *mut f32,
    k: *const f32,
    v: *const f32,
    q: *const f32,
    g: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    scale: f32,
    h0: usize,
    h1: usize,
    geom: &WkvGeom,
) {
    use core::arch::x86_64::*;
    let vec_count = geom.s / WKV_VECTOR_SIZE;
    for t in 0..geom.t {
        let t_offset = t * geom.c;
        let state_offset = geom.state_offset(t);
        let state_cur = dst_state.add(state_offset);
        let state_prev: *const f32 =
            if t % geom.per_seq != 0 { state_cur } else { state.add(state_offset) };
        for h in h0..h1 {
            let h_offset = h * geom.s;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * geom.s * geom.s;
            for i in 0..geom.s {
                let t_h_i_offset = t_h_offset + i;
                let h_2d_i_offset = h_2d_offset + i * geom.s;

                // SAFETY: scalar reads in-bounds.
                let k_vec = _mm512_set1_ps(*k.add(t_h_i_offset));
                let q_vec = _mm512_set1_ps(*q.add(t_h_i_offset) * scale);
                let g_vec = _mm512_set1_ps(*g.add(t_h_i_offset));

                for j in 0..vec_count {
                    let base_j = j * WKV_VECTOR_SIZE;
                    let t_h_j_offset = t_offset + h_offset + base_j;
                    let h_2d_i_j_offset = h_2d_offset + i * geom.s + base_j;

                    // SAFETY: windows inside the head row / block.
                    let v_vec = _mm512_loadu_ps(v.add(t_h_j_offset));
                    let prev_state_vec = _mm512_loadu_ps(state_prev.add(h_2d_i_j_offset));
                    let mut dst_vec = _mm512_loadu_ps(dst.add(t_h_j_offset));

                    let kv_vec = _mm512_mul_ps(v_vec, k_vec);
                    let temp_vec = _mm512_fmadd_ps(prev_state_vec, g_vec, kv_vec);
                    dst_vec = _mm512_fmadd_ps(temp_vec, q_vec, dst_vec);
                    _mm512_storeu_ps(dst.add(t_h_j_offset), dst_vec);
                    _mm512_storeu_ps(state_cur.add(h_2d_i_j_offset), temp_vec);
                }

                // scalar tail: the reference's compiled contraction (see
                // gla_scalar) — prev·g a separate mul, v·k fused in the fma
                for j in (vec_count * WKV_VECTOR_SIZE)..geom.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    // SAFETY: j < s.
                    let v_val = *v.add(t_h_j_offset);
                    let k_val = *k.add(t_h_i_offset);
                    let q_val = *q.add(t_h_i_offset) * scale;
                    let g_val = *g.add(t_h_i_offset);
                    let prev_state_val = *state_prev.add(h_2d_i_j_offset);
                    let pg_val = prev_state_val * g_val;
                    let temp_val = v_val.mul_add(k_val, pg_val);
                    *dst.add(t_h_j_offset) = temp_val.mul_add(q_val, *dst.add(t_h_j_offset));
                    *state_cur.add(h_2d_i_j_offset) = temp_val;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// ggml_compute_forward_rwkv_wkv7_f32 (ops.cpp:11422-11617)
// ---------------------------------------------------------------------

/// The x86 `GGML_SIMD` body for one thread's head range — the 4×16 window
/// sweep. At head_size ≥ 64 the window never leaves the head row / block
/// (exact, dump-pinned). Below that the reference's behavior is UB — see the
/// module header and parity/wkv7_oob_proof.c — and this port computes the
/// deterministic approximation (in-tensor cross-head mixing reproduced;
/// past-tensor reads 0.0 / writes dropped).
///
/// SAFETY: `dst` at `C*T + S*S*H*n_seqs` writable f32s; r/w/k/v/a/b at
/// `S*H*T` readable f32s each; state at `S*S*H*n_seqs`. Disjoint head ranges
/// across concurrent calls *only* keep wkv7 race-free when head_size is a
/// multiple of 64 (otherwise the window's tail writes cross head blocks —
/// the reference has the same race).
#[allow(clippy::too_many_arguments)]
pub unsafe fn rwkv_wkv7_f32(
    dst: *mut f32,
    r: *const f32,
    w: *const f32,
    k: *const f32,
    v: *const f32,
    a: *const f32,
    b: *const f32,
    state: *const f32,
    h0: usize,
    h1: usize,
    s: usize,
    heads: usize,
    t_len: usize,
    n_seqs: usize,
) {
    let g = WkvGeom::new(s, heads, t_len, n_seqs);
    let dst_state = dst.add(g.c * g.t);
    if crate::simd_x86::avx512() {
        #[cfg(target_arch = "x86_64")]
        {
            rwkv_wkv7_avx512(dst, r, w, k, v, a, b, state, dst_state, h0, h1, &g);
            return;
        }
    }
    rwkv_wkv7_scalar(dst, r, w, k, v, a, b, state, dst_state, h0, h1, &g);
}

/// The spec body — lanes as `[f32; 16]`, the 64-wide window over
/// `r/w/k/v/a/b` and `state_prev` with the zero-fill OOB convention.
#[allow(clippy::too_many_arguments)]
unsafe fn rwkv_wkv7_scalar(
    dst: *mut f32,
    r: *const f32,
    w: *const f32,
    k: *const f32,
    v: *const f32,
    a: *const f32,
    b: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    h0: usize,
    h1: usize,
    g: &WkvGeom,
) {
    for t in 0..g.t {
        let t_offset = t * g.c;
        let state_offset = g.state_offset(t);
        // SAFETY: inside dst's state area; state_prev may alias state_cur.
        let state_cur = dst_state.add(state_offset);
        let state_cur_len = g.state_len - state_offset;
        let state_prev: *const f32 = if t % g.per_seq != 0 {
            state_cur
        } else {
            state.add(state_offset)
        };
        let state_prev_len = state_cur_len;
        for h in h0..h1 {
            let h_offset = h * g.s;
            let t_h_offset = t_offset + h_offset;
            let h_2d_offset = h * g.s * g.s;
            for ii in 0..g.s {
                let t_h_i_offset = t_h_offset + ii;
                let h_2d_i_offset = h_2d_offset + ii * g.s;

                // SAFETY: ii < s — in-bounds.
                let v_vec = set116(*v.add(t_h_i_offset));

                // sa = Σ_j a[j]·state_prev[j] — 4×16 lanes (ops.cpp:11516-11535)
                let mut sum = [[0f32; 16]; F32_ARR];
                let mut j = 0usize;
                while j < g.s {
                    for kk in 0..F32_ARR {
                        let base = j + kk * F32_EPR;
                        let ax = load16z(a, t_h_offset + base, g.src_len);
                        let ay = load16z(state_prev, h_2d_i_offset + base, state_prev_len);
                        sum[kk] = fma16(sum[kk], ax, ay);
                    }
                    j += F32_STEP;
                }
                let sa = reduce4(&sum);
                let sa_vec = set116(sa);

                // state update + result accumulation (ops.cpp:11537-11560)
                let mut result_vec = [[0f32; 16]; F32_ARR];
                let mut j = 0usize;
                while j < g.s {
                    for kk in 0..F32_ARR {
                        let base = j + kk * F32_EPR;
                        let r_vec = load16z(r, t_h_offset + base, g.src_len);
                        let w_vec = load16z(w, t_h_offset + base, g.src_len);
                        let k_vec = load16z(k, t_h_offset + base, g.src_len);
                        let b_vec = load16z(b, t_h_offset + base, g.src_len);

                        // k = v * k (per lane)
                        let kv_vec = mul16(v_vec, k_vec);
                        // kv + s * decay + sa * b
                        let mut state_vec = load16z(state_prev, h_2d_i_offset + base, state_prev_len);
                        state_vec = fma16(kv_vec, state_vec, w_vec);
                        state_vec = fma16(state_vec, sa_vec, b_vec);
                        store16z(state_cur, h_2d_i_offset + base, state_cur_len, state_vec);
                        result_vec[kk] = fma16(result_vec[kk], state_vec, r_vec);
                    }
                    j += F32_STEP;
                }
                // SAFETY: t_h_i_offset < C*T.
                *dst.add(t_h_i_offset) = reduce4(&result_vec);

                // "There shouldn't be left-overs though." (ops.cpp:11562) —
                // the x86 geometry makes this loop unreachable (j exits the
                // window loop at the first multiple of STEP ≥ head_size);
                // kept literal, with the C's fmaf-contracted scalar form.
                while j < g.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    // SAFETY: guarded by the len checks inside load16z /
                    // store16z; the plain scalar reads below mirror C (they
                    // only run for head_size > 64 windows, in-tensor).
                    let r_val = load16z(r, t_h_j_offset, g.src_len)[0];
                    let w_val = load16z(w, t_h_j_offset, g.src_len)[0];
                    let k_val = load16z(k, t_h_j_offset, g.src_len)[0];
                    let b_val = load16z(b, t_h_j_offset, g.src_len)[0];
                    let kv_val = v_vec[0] * k_val;
                    let prev_state_val = load16z(state_prev, h_2d_i_j_offset, state_prev_len)[0];
                    // prev*w + kv + sa*b, the first pair contracted
                    let ns = prev_state_val.mul_add(w_val, kv_val) + sa * b_val;
                    store16z(state_cur, h_2d_i_j_offset, state_cur_len, [ns; 16]);
                    *dst.add(t_h_i_offset) = ns.mul_add(r_val, *dst.add(t_h_i_offset));
                    j += 1;
                }
            }
        }
    }
}

/// SAFETY: as `rwkv_wkv7_scalar`; additionally the caller must have checked
/// `avx512()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn rwkv_wkv7_avx512(
    dst: *mut f32,
    r: *const f32,
    w: *const f32,
    k: *const f32,
    v: *const f32,
    a: *const f32,
    b: *const f32,
    state: *const f32,
    dst_state: *mut f32,
    h0: usize,
    h1: usize,
    g: &WkvGeom,
) {
    use core::arch::x86_64::*;
    // load16z/store16z for __m512: the in-bounds fast path is a plain
    // loadu/storeu; the overrun path stages a zero-padded [f32;16].
    #[inline(always)]
    unsafe fn loadz(p: *const f32, idx: usize, len: usize) -> __m512 {
        if idx + 16 <= len {
            _mm512_loadu_ps(p.add(idx))
        } else {
            let v = load16z(p, idx, len);
            _mm512_loadu_ps(v.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn storez(p: *mut f32, idx: usize, len: usize, v: __m512) {
        if idx + 16 <= len {
            _mm512_storeu_ps(p.add(idx), v);
        } else {
            let mut buf = [0f32; 16];
            _mm512_storeu_ps(buf.as_mut_ptr(), v);
            store16z(p, idx, len, buf);
        }
    }
    /// GGML_F32x16_REDUCE(res, x) for 4 registers (simd-mappings.h:466-480):
    /// vector pairwise fold, then the pinned `_mm512_reduce_add_ps` order
    /// (vec_dot.rs `reduce_add16`) on a stack spill — both compilers'
    /// expansions agree with it (vec_ref.bin).
    #[inline(always)]
    unsafe fn reduce4x(x: &[__m512; 4]) -> f32 {
        let x0 = _mm512_add_ps(x[0], x[2]);
        let x1 = _mm512_add_ps(x[1], x[3]);
        let x0 = _mm512_add_ps(x0, x1);
        let mut buf = [0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), x0);
        reduce_add16(&buf)
    }

    for t in 0..g.t {
        let t_offset = t * g.c;
        let state_offset = g.state_offset(t);
        // SAFETY: inside dst's state area.
        let state_cur = dst_state.add(state_offset);
        let state_cur_len = g.state_len - state_offset;
        let state_prev: *const f32 = if t % g.per_seq != 0 {
            state_cur
        } else {
            state.add(state_offset)
        };
        let state_prev_len = state_cur_len;
        for h in h0..h1 {
            let t_h_offset = t_offset + h * g.s;
            let h_2d_offset = h * g.s * g.s;
            for ii in 0..g.s {
                let t_h_i_offset = t_h_offset + ii;
                let h_2d_i_offset = h_2d_offset + ii * g.s;

                // SAFETY: ii < s.
                let v_vec = _mm512_set1_ps(*v.add(t_h_i_offset));

                let mut sum = [_mm512_setzero_ps(); F32_ARR];
                let mut j = 0usize;
                while j < g.s {
                    for kk in 0..F32_ARR {
                        let base = j + kk * F32_EPR;
                        let ax = loadz(a, t_h_offset + base, g.src_len);
                        let ay = loadz(state_prev, h_2d_i_offset + base, state_prev_len);
                        sum[kk] = _mm512_fmadd_ps(ax, ay, sum[kk]);
                    }
                    j += F32_STEP;
                }
                let sa = reduce4x(&sum);
                let sa_vec = _mm512_set1_ps(sa);

                let mut result_vec = [_mm512_setzero_ps(); F32_ARR];
                let mut j = 0usize;
                while j < g.s {
                    for kk in 0..F32_ARR {
                        let base = j + kk * F32_EPR;
                        let r_vec = loadz(r, t_h_offset + base, g.src_len);
                        let w_vec = loadz(w, t_h_offset + base, g.src_len);
                        let k_vec = loadz(k, t_h_offset + base, g.src_len);
                        let b_vec = loadz(b, t_h_offset + base, g.src_len);

                        let kv_vec = _mm512_mul_ps(v_vec, k_vec);
                        let mut state_vec = loadz(state_prev, h_2d_i_offset + base, state_prev_len);
                        // kv + s*decay + sa*b
                        state_vec = _mm512_fmadd_ps(state_vec, w_vec, kv_vec);
                        state_vec = _mm512_fmadd_ps(sa_vec, b_vec, state_vec);
                        storez(state_cur, h_2d_i_offset + base, state_cur_len, state_vec);
                        result_vec[kk] = _mm512_fmadd_ps(state_vec, r_vec, result_vec[kk]);
                    }
                    j += F32_STEP;
                }
                // SAFETY: t_h_i_offset < C*T.
                *dst.add(t_h_i_offset) = reduce4x(&result_vec);

                // unreachable on the x86 geometry (see the spec body)
                while j < g.s {
                    let t_h_j_offset = t_h_offset + j;
                    let h_2d_i_j_offset = h_2d_i_offset + j;
                    let r_val = load16z(r, t_h_j_offset, g.src_len)[0];
                    let w_val = load16z(w, t_h_j_offset, g.src_len)[0];
                    let k_val = load16z(k, t_h_j_offset, g.src_len)[0];
                    let b_val = load16z(b, t_h_j_offset, g.src_len)[0];
                    let kv_val = *v.add(t_h_i_offset) * k_val;
                    let prev_state_val = load16z(state_prev, h_2d_i_j_offset, state_prev_len)[0];
                    let ns = sa.mul_add(b_val, prev_state_val.mul_add(w_val, kv_val));
                    store16z(state_cur, h_2d_i_j_offset, state_cur_len, [ns; 16]);
                    *dst.add(t_h_i_offset) = ns.mul_add(r_val, *dst.add(t_h_i_offset));
                    j += 1;
                }
            }
        }
    }
}

// ======================================================================
// tests — the reference dump (parity/wkv_ref.bin) is the gate
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgmlType;
    use crate::{compute::graph_compute, Context, Graph};

    fn mk3(ctx: &mut Context, data: &[f32], s: usize, h: usize, t: usize) -> crate::TensorId {
        let id = ctx.new_tensor_3d(GgmlType::F32, s as i64, h as i64, t as i64);
        ctx.arena_resize_tensor(id);
        ctx.with_f32_mut(id, |p| p.copy_from_slice(data)).unwrap();
        id
    }

    fn mk2(ctx: &mut Context, data: &[f32], ne0: usize, ne1: usize) -> crate::TensorId {
        let id = ctx.new_tensor_2d(GgmlType::F32, ne0 as i64, ne1 as i64);
        ctx.arena_resize_tensor(id);
        ctx.with_f32_mut(id, |p| p.copy_from_slice(data)).unwrap();
        id
    }

    fn compute(mut ctx: Context, out: crate::TensorId, nth: usize) -> Vec<f32> {
        let mut g = Graph::new(8);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, nth);
        ctx.f32s(out).unwrap().to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_wkv6(
        k: &[f32], v: &[f32], r: &[f32], tf: &[f32], td: &[f32], st: &[f32], s: usize, h: usize,
        t: usize, n_seqs: usize, nth: usize,
    ) -> Vec<f32> {
        let mut ctx = Context::new();
        let kt = mk3(&mut ctx, k, s, h, t);
        let vt = mk3(&mut ctx, v, s, h, t);
        let rt = mk3(&mut ctx, r, s, h, t);
        let tft = mk2(&mut ctx, tf, s, h);
        let tdt = mk3(&mut ctx, td, s, h, t);
        let stt = mk2(&mut ctx, st, s * s * h, n_seqs);
        let out = ctx.rwkv_wkv6(kt, vt, rt, tft, tdt, stt);
        compute(ctx, out, nth)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gla(
        k: &[f32], v: &[f32], q: &[f32], g: &[f32], st: &[f32], scale: f32, s: usize, h: usize,
        t: usize, n_seqs: usize, nth: usize,
    ) -> Vec<f32> {
        let mut ctx = Context::new();
        let kt = mk3(&mut ctx, k, s, h, t);
        let vt = mk3(&mut ctx, v, s, h, t);
        let qt = mk3(&mut ctx, q, s, h, t);
        let gt = mk3(&mut ctx, g, s, h, t);
        let stt = mk2(&mut ctx, st, s * s * h, n_seqs);
        let out = ctx.gated_linear_attn(kt, vt, qt, gt, stt, scale);
        compute(ctx, out, nth)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_wkv7(
        r: &[f32], w: &[f32], k: &[f32], v: &[f32], a: &[f32], b: &[f32], st: &[f32], s: usize,
        h: usize, t: usize, n_seqs: usize, nth: usize,
    ) -> Vec<f32> {
        let mut ctx = Context::new();
        let rt = mk3(&mut ctx, r, s, h, t);
        let wt = mk3(&mut ctx, w, s, h, t);
        let kt = mk3(&mut ctx, k, s, h, t);
        let vt = mk3(&mut ctx, v, s, h, t);
        let at = mk3(&mut ctx, a, s, h, t);
        let bt = mk3(&mut ctx, b, s, h, t);
        let stt = mk2(&mut ctx, st, s * s * h, n_seqs);
        let out = ctx.rwkv_wkv7(rt, wt, kt, vt, at, bt, stt);
        compute(ctx, out, nth)
    }

    fn assert_bits(got: &[f32], want: &[f32], tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag}: length");
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{tag}: element {i}: got {a} want {b}");
        }
    }

    /// Bit-exactness against parity/wkv_ref.bin (dumped by
    /// parity/ref_wkv_dump.c from the reference build's own
    /// ggml_rwkv_wkv6 / ggml_gated_linear_attn / ggml_rwkv_wkv7 through the
    /// real graph path at n_threads == 1). This is THE pin of the three
    /// kernels.
    ///
    /// Every section is bit-exact. The gla scalar tail carries the
    /// reference's compiled contraction — GCC fused the v·k product into
    /// the addend (`temp = fma(v, k, prev·g)`, prev·g a separate rounded
    /// multiply), the opposite pairing of the vec body's explicit
    /// intrinsics; pinned by the S=20 section (and the gdb disassembly
    /// evidence in PARITY.md's WKV notes).
    ///
    /// The wkv7 ladder is head_size ≥ 64 only: below that the reference's
    /// 64-wide window leaves the tensors (allocator metadata reads, NaN
    /// self-poisoning) and its output is ASLR-dependent — see
    /// parity/wkv7_oob_proof.c and the module header. Every section
    /// re-runs at nth = 3 and must be bit-equal (all shapes in the ladder
    /// have the per-head state blocks disjoint from the window's writes).
    #[test]
    fn wkv_kernels_match_reference_dump() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/wkv_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("SKIP: {path} missing (build parity/ref_wkv_dump.c)");
            return;
        };
        let mut c = &bytes[..];
        let u32at = |c: &mut &[u8]| -> u32 {
            let v = u32::from_le_bytes(c[..4].try_into().unwrap());
            *c = &c[4..];
            v
        };
        let f32s = |c: &mut &[u8], n: usize| -> Vec<f32> {
            let (a, rest) = c.split_at(n * 4);
            *c = rest;
            a.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        };
        let mut cases = 0usize;
        let mut total = 0usize;
        loop {
            let magic = u32at(&mut c);
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x31564B57, "section magic 'WKV1'");
            let kind = u32at(&mut c);
            let nd = u32at(&mut c) as usize;
            let dims: Vec<usize> = (0..nd).map(|_| u32at(&mut c) as usize).collect();
            match kind {
                20 => {
                    let (s, h, t, n_seqs) = (dims[0], dims[1], dims[2], dims[3]);
                    let nel = s * h * t;
                    let kv = f32s(&mut c, nel);
                    let vv = f32s(&mut c, nel);
                    let rr = f32s(&mut c, nel);
                    let tf = f32s(&mut c, s * h);
                    let td = f32s(&mut c, nel);
                    let st = f32s(&mut c, s * s * h * n_seqs);
                    let want = f32s(&mut c, nel + s * s * h * n_seqs);
                    let got = run_wkv6(&kv, &vv, &rr, &tf, &td, &st, s, h, t, n_seqs, 1);
                    assert_bits(&got, &want, &format!("wkv6 S={s} H={h} T={t} seqs={n_seqs}"));
                    if n_seqs == 1 {
                        let got3 = run_wkv6(&kv, &vv, &rr, &tf, &td, &st, s, h, t, n_seqs, 3);
                        assert_eq!(got, got3, "wkv6 S={s} H={h} T={t} nth-invariance");
                    }
                    cases += 1;
                    total += want.len();
                }
                21 => {
                    let (s, h, t, n_seqs) = (dims[0], dims[1], dims[2], dims[3]);
                    let scale = f32::from_bits(dims[4] as u32);
                    let nel = s * h * t;
                    let kv = f32s(&mut c, nel);
                    let vv = f32s(&mut c, nel);
                    let qq = f32s(&mut c, nel);
                    let gg = f32s(&mut c, nel);
                    let st = f32s(&mut c, s * s * h * n_seqs);
                    let want = f32s(&mut c, nel + s * s * h * n_seqs);
                    let got = run_gla(&kv, &vv, &qq, &gg, &st, scale, s, h, t, n_seqs, 1);
                    // every shape bit-exact — S=20's scalar tail now carries
                    // the reference's compiled contraction (see the test doc)
                    assert_bits(&got, &want, &format!("gla S={s} H={h} T={t} seqs={n_seqs}"));
                    if n_seqs == 1 {
                        let got3 = run_gla(&kv, &vv, &qq, &gg, &st, scale, s, h, t, n_seqs, 3);
                        assert_eq!(got, got3, "gla S={s} H={h} T={t} nth-invariance");
                    }
                    cases += 1;
                    total += want.len();
                }
                22 => {
                    let (s, h, t, n_seqs) = (dims[0], dims[1], dims[2], dims[3]);
                    let nel = s * h * t;
                    let rr = f32s(&mut c, nel);
                    let ww = f32s(&mut c, nel);
                    let kk = f32s(&mut c, nel);
                    let vv = f32s(&mut c, nel);
                    let aa = f32s(&mut c, nel);
                    let bb = f32s(&mut c, nel);
                    let st = f32s(&mut c, s * s * h * n_seqs);
                    let want = f32s(&mut c, nel + s * s * h * n_seqs);
                    let got = run_wkv7(&rr, &ww, &kk, &vv, &aa, &bb, &st, s, h, t, n_seqs, 1);
                    assert_bits(&got, &want, &format!("wkv7 S={s} H={h} T={t} seqs={n_seqs}"));
                    // S ≥ 64: the window never leaves the head row / block →
                    // nth-invariant, multi-seq included
                    if s % F32_STEP == 0 {
                        let got3 = run_wkv7(&rr, &ww, &kk, &vv, &aa, &bb, &st, s, h, t, n_seqs, 3);
                        assert_eq!(got, got3, "wkv7 S={s} H={h} T={t} seqs={n_seqs} nth-invariance");
                    }
                    cases += 1;
                    total += want.len();
                }
                other => panic!("unknown kind {other}"),
            }
        }
        eprintln!("wkv dump: {cases} sections, {total} elements, all bit-exact");
        assert!(cases >= 11, "expected the full section ladder, got {cases}");
    }

    /// The spec (scalar, `mul_add` lane) bodies must be bit-identical to the
    /// dispatched AVX512 bodies — the simd_x86.rs convention, transitively
    /// pinning the intrinsics path to the reference dump above.
    #[test]
    fn wkv_simd_equals_scalar() {
        if !crate::simd_x86::avx512() {
            eprintln!("SKIP: no AVX512 on this host");
            return;
        }
        for (s, h, t) in [(64usize, 2usize, 5usize), (16, 4, 3), (20, 2, 4)] {
            let nel = s * h * t;
            let mut it = lcg(0x5eed + s as u32);
            let mut rnd = |v: &mut Vec<f32>, n: usize| v.extend((0..n).map(|_| it.next().unwrap()));
            let (mut r, mut w, mut k, mut v, mut a, mut b, mut st) =
                (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
            rnd(&mut r, nel);
            rnd(&mut w, nel);
            rnd(&mut k, nel);
            rnd(&mut v, nel);
            rnd(&mut a, nel);
            rnd(&mut b, nel);
            rnd(&mut st, s * s * h);
            let via_ctx = run_wkv7(&r, &w, &k, &v, &a, &b, &st, s, h, t, 1, 1);
            let out_len = s * h * t + s * s * h;

            // scalar body directly (the dispatched entry picked avx512 above)
            let mut dst2 = vec![0f32; out_len];
            let g = WkvGeom::new(s, h, t, 1);
            // SAFETY: c*t + s*s*h == dst2.len()
            let dst_state = unsafe { dst2.as_mut_ptr().add(g.c * g.t) };
            unsafe {
                rwkv_wkv7_scalar(
                    dst2.as_mut_ptr(),
                    r.as_ptr(),
                    w.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    a.as_ptr(),
                    b.as_ptr(),
                    st.as_ptr(),
                    dst_state,
                    0,
                    h,
                    &g,
                )
            };
            assert_bits(&dst2, &via_ctx, &format!("wkv7 scalar==avx512 S={s}"));

            // wkv6 / gla likewise
            let mut it2 = lcg(0xBEEF + s as u32);
            let mut rnd2 =
                |v: &mut Vec<f32>, n: usize| v.extend((0..n).map(|_| it2.next().unwrap()));
            let (mut k6, mut v6, mut r6, mut tf, mut td, mut q, mut gg) =
                (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
            rnd2(&mut k6, nel);
            rnd2(&mut v6, nel);
            rnd2(&mut r6, nel);
            rnd2(&mut tf, s * h);
            rnd2(&mut td, nel);
            rnd2(&mut q, nel);
            rnd2(&mut gg, nel);
            let via6 = run_wkv6(&k6, &v6, &r6, &tf, &td, &st, s, h, t, 1, 1);
            let mut d6 = vec![0f32; out_len];
            // SAFETY: c*t + s*s*h == d6.len()
            let d6_state = unsafe { d6.as_mut_ptr().add(g.c * g.t) };
            unsafe {
                d6[..s * h * t].fill(0.0);
                rwkv_wkv6_scalar(
                    d6.as_mut_ptr(),
                    k6.as_ptr(),
                    v6.as_ptr(),
                    r6.as_ptr(),
                    tf.as_ptr(),
                    td.as_ptr(),
                    st.as_ptr(),
                    d6_state,
                    0,
                    h,
                    &g,
                )
            };
            assert_bits(&d6, &via6, &format!("wkv6 scalar==avx512 S={s}"));

            let scale = (s as f32).powf(-0.5);
            let via_gla = run_gla(&k6, &v6, &q, &gg, &st, scale, s, h, t, 1, 1);
            let mut dg = vec![0f32; out_len];
            // SAFETY: c*t + s*s*h == dg.len()
            let dg_state = unsafe { dg.as_mut_ptr().add(g.c * g.t) };
            unsafe {
                dg[..s * h * t].fill(0.0);
                gla_scalar(
                    dg.as_mut_ptr(),
                    k6.as_ptr(),
                    v6.as_ptr(),
                    q.as_ptr(),
                    gg.as_ptr(),
                    st.as_ptr(),
                    dg_state,
                    scale,
                    0,
                    h,
                    &g,
                )
            };
            // both port tails now carry the reference's compiled contraction
            // (prev·g separate, v·k fused), so scalar==SIMD holds for every
            // shape
            assert_bits(&dg, &via_gla, &format!("gla scalar==avx512 S={s}"));
        }
    }

    fn lcg(seed: u32) -> impl Iterator<Item = f32> {
        let mut s = seed;
        std::iter::repeat_with(move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as i32 as f32) / (1u32 << 30) as f32 * 0.5
        })
    }

    /// wkv6 / gla against an f64 per-token model — the independent-math
    /// check on top of the reference dump. The harness originally modeled
    /// wkv6's temp as `tf[h,i]·s + kv`; the kernel (ops.cpp:10533's
    /// `kv * time_faaaa + prev_state`) computes `kv·tf + s` — tf multiplies
    /// the kv product, not the state. The kernel was never wrong (bit-exact
    /// vs the reference in `wkv_kernels_match_reference_dump`); the harness
    /// formula was. Fixed here and re-enabled.
    #[test]
    fn wkv6_gla_match_naive() {
        let (s, h, t) = (16usize, 4usize, 3usize);
        let nel = s * h * t;
        let mut it = lcg(99);
        let mut rnd = |n: usize| (0..n).map(|_| it.next().unwrap()).collect::<Vec<f32>>();
        let k = rnd(nel);
        let v = rnd(nel);
        let r = rnd(nel);
        let tf = rnd(s * h);
        let td = rnd(nel);
        let st0 = rnd(s * s * h);
        let got = run_wkv6(&k, &v, &r, &tf, &td, &st0, s, h, t, 1, 1);

        // naive f64 recurrence (the kernel's actual formula, ops.cpp:10533):
        //   out[t,h,j] = Σ_i r[t,h,i] * (k[t,h,i]·v[t,h,j]·tf[h,i] + s[h,i,j])
        //   s[h,i,j] ← s[h,i,j]·td[t,h,i] + k[t,h,i]·v[t,h,j]
        // state laid out [h][i][j]; each (i,j) cell updates once per token.
        let mut state = st0.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let mut want = vec![0f64; nel];
        for tt in 0..t {
            for hh in 0..h {
                let t_h = tt * s * h + hh * s;
                for j in 0..s {
                    let mut acc = 0f64;
                    for i in 0..s {
                        let h2_ij = hh * s * s + i * s + j;
                        let kv = v[t_h + j] as f64 * k[t_h + i] as f64;
                        let temp = kv * tf[hh * s + i] as f64 + state[h2_ij];
                        acc += r[t_h + i] as f64 * temp;
                        state[h2_ij] = state[h2_ij] * td[t_h + i] as f64 + kv;
                    }
                    want[t_h + j] = acc;
                }
            }
        }
        for i in 0..nel {
            let err = (got[i] - want[i] as f32).abs();
            assert!(err < 1e-3, "wkv6 out[{i}]: {} vs {}", got[i], want[i]);
        }
        for i in 0..s * s * h {
            let err = (got[nel + i] - state[i] as f32).abs();
            assert!(err < 1e-3, "wkv6 state[{i}]: {} vs {}", got[nel + i], state[i]);
        }

        // gla
        let q = rnd(nel);
        let g = rnd(nel);
        let scale = (s as f32).powf(-0.5);
        let got = run_gla(&k, &v, &q, &g, &st0, scale, s, h, t, 1, 1);
        let mut state = st0.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let mut want = vec![0f64; nel];
        for tt in 0..t {
            for hh in 0..h {
                let t_h = tt * s * h + hh * s;
                // out[t,h,j] = Σ_i q[i]·(s[i,j]·g[i] + k[i]·v[j]) — the
                // output row is j-indexed; s[i,j] ← temp (the kernel writes
                // temp, not s·g + kv, into the state)
                for j in 0..s {
                    let mut acc = 0f64;
                    for i in 0..s {
                        let h2_ij = hh * s * s + i * s + j;
                        let qs = q[t_h + i] as f64 * scale as f64;
                        let kv = v[t_h + j] as f64 * k[t_h + i] as f64;
                        let temp = state[h2_ij] * g[t_h + i] as f64 + kv;
                        acc += temp * qs;
                        state[h2_ij] = temp;
                    }
                    want[t_h + j] = acc;
                }
            }
        }
        for i in 0..nel {
            let err = (got[i] - want[i] as f32).abs();
            assert!(err < 1e-3, "gla out[{i}]: {} vs {}", got[i], want[i]);
        }
    }

    /// wkv7 at head_size == 64 against the naive f64 recurrence (the exact
    /// geometry — no window overrun); at head_size < 64 the kernel's window
    /// semantics deliberately deviate from the naive model (module header),
    /// so only boundedness is asserted there.
    #[test]
    fn wkv7_matches_naive_at_head_size_64() {
        let (s, h, t) = (64usize, 2usize, 5usize);
        let nel = s * h * t;
        let mut it = lcg(7);
        let mut rnd = |n: usize| (0..n).map(|_| it.next().unwrap()).collect::<Vec<f32>>();
        let r = rnd(nel);
        let w = rnd(nel);
        let k = rnd(nel);
        let v = rnd(nel);
        let a = rnd(nel);
        let b = rnd(nel);
        let st0 = rnd(s * s * h);
        let got = run_wkv7(&r, &w, &k, &v, &a, &b, &st0, s, h, t, 1, 1);

        let mut state = st0.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let mut want = vec![0f64; nel];
        for tt in 0..t {
            for hh in 0..h {
                for i in 0..s {
                    let t_h = tt * s * h + hh * s;
                    let h2_i = hh * s * s + i * s;
                    let mut sa = 0f64;
                    for j in 0..s {
                        sa += a[t_h + j] as f64 * state[h2_i + j];
                    }
                    let mut res = 0f64;
                    for j in 0..s {
                        let kv = v[t_h + i] as f64 * k[t_h + j] as f64;
                        let ns = state[h2_i + j] * w[t_h + j] as f64 + kv + sa * b[t_h + j] as f64;
                        state[h2_i + j] = ns;
                        res += ns * r[t_h + j] as f64;
                    }
                    want[t_h + i] = res;
                }
            }
        }
        for i in 0..nel {
            let err = (got[i] - want[i] as f32).abs();
            assert!(err < 1e-2, "wkv7 out[{i}]: {} vs {}", got[i], want[i]);
        }
        for i in 0..s * s * h {
            let err = (got[nel + i] - state[i] as f32).abs();
            assert!(err < 1e-2, "wkv7 state[{i}]: {} vs {}", got[nel + i], state[i]);
        }
    }
}
