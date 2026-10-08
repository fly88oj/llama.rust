//! gdn.rs — gated delta net kernel (GGML_OP_GATED_DELTA_NET).
//!
//! 对照 ggml/src/ggml-cpu/ops.cpp:10894-11105 (`ggml_compute_forward_gated_delta_net`
//! / `_one_chunk`) in the pinned worktree @ bd4f514db1, restricted to the
//! **K == 1** slice (the only shape the port's graphs build: K is
//! `cparams.n_rs_seq + 1` and the port models a single sequence with no
//! rollback snapshots — see RecurrentState).
//!
//! Semantics (identical to C's per-token recurrence):
//!   for each (head v1, seq v3) and each token t (in order):
//!     S *= exp(g)                              (scalar gate: g->ne[0] == 1)
//!     delta[j] = (v[j] - dot(row_j(S), k)) * beta
//!     S[i][j] += k[i] * delta[j]                (outer product)
//!     out[j]   = dot(row_j(S), q) * scale
//! The state is kept **transposed** in memory (`s_out[j*S_v + i] = S[i][j]`),
//! exactly like the C kernel, so the dots read contiguous rows.
//!
//! Numerics: the two dots go through `vec_dot_f32_c` (the AVX512 lane
//! structure of the reference build's `ggml_vec_dot_f32`, 4x16 lanes + pairwise
//! reduce — see vec_dot.rs); `ggml_vec_scale_f32` / `ggml_vec_mad_f32` are
//! element-wise (FMA for the mad: C's AVX512 branch is `_mm512_fmadd_ps`, tail
//! `y += x*v`; S_v is a multiple of GGML_F32_STEP = 64 for every ported model),
//! and the gate exponential is plain libm `expf`, as in C.

use crate::vec_dot::vec_dot_f32_c;

/// One operand of the kernel: raw base pointer + the ggml strides/dims.
/// `ne` is only read for the broadcast factors (q/k heads, token counts).
pub struct GdnSrc {
    pub ptr: *const u8,
    pub nb: [u64; 4],
    pub ne: [i64; 4],
}

/// Polarity/geometry checks mirroring the asserts in `ggml_gated_delta_net`
/// (ggml.c:6366-6405) and the kernel prologue (ops.cpp:10904-10930).
#[allow(clippy::too_many_arguments)]
pub fn gdn_check(
    q: &GdnSrc,
    k: &GdnSrc,
    v: &GdnSrc,
    g: &GdnSrc,
    beta: &GdnSrc,
    state: &GdnSrc,
    k_slots: i64,
) {
    for (t, name) in [(q, "q"), (k, "k"), (v, "v")] {
        assert_eq!(t.nb[0], 4, "gdn: {name}->nb[0] == sizeof(float) (contiguous rows)");
    }
    for (t, name) in [(g, "g"), (beta, "beta"), (state, "state")] {
        // ggml_is_contiguous(src) — a vector / plain contiguous tensor
        assert_eq!(t.nb[0], 4, "gdn: {name} must be F32 contiguous");
        assert_eq!(t.nb[1], t.ne[0] as u64 * 4, "gdn: {name}->nb[1]");
        assert_eq!(t.nb[2], t.nb[1] * t.ne[1] as u64, "gdn: {name}->nb[2]");
        assert_eq!(t.nb[3], t.nb[2] * t.ne[2] as u64, "gdn: {name}->nb[3]");
    }
    let s_v = v.ne[0];
    let h_v = v.ne[1];
    let n_tokens = v.ne[2];
    let n_seqs = v.ne[3];
    assert_eq!(q.ne[0], s_v, "gdn: q->ne[0] == S_v");
    assert_eq!(k.ne[0], s_v, "gdn: k->ne[0] == S_v");
    assert_eq!(q.ne[2], n_tokens, "gdn: q tokens");
    assert_eq!(k.ne[2], n_tokens, "gdn: k tokens");
    assert_eq!(q.ne[3], n_seqs, "gdn: q seqs");
    assert_eq!(k.ne[3], n_seqs, "gdn: k seqs");
    assert!(h_v % k.ne[1] == 0, "gdn: H_v % H_k == 0");
    assert!(g.ne[0] == 1 || g.ne[0] == s_v, "gdn: g->ne[0] == 1 || S_v");
    assert_eq!(g.ne[1], h_v, "gdn: g->ne[1] == H_v");
    assert_eq!(g.ne[2], n_tokens, "gdn: g->ne[2]");
    assert_eq!(g.ne[3], n_seqs, "gdn: g->ne[3]");
    assert_eq!(beta.ne[0], 1, "gdn: beta->ne[0] == 1");
    assert_eq!(beta.ne[1], h_v, "gdn: beta->ne[1] == H_v");
    assert_eq!(beta.ne[2], n_tokens, "gdn: beta->ne[2]");
    assert_eq!(beta.ne[3], n_seqs, "gdn: beta->ne[3]");
    assert_eq!(state.ne[0], s_v, "gdn: state->ne[0] == S_v");
    assert_eq!(state.ne[1], s_v, "gdn: state->ne[1] == S_v");
    assert_eq!(state.ne[2], h_v, "gdn: state->ne[2] == H_v");
    assert_eq!(state.ne[3], n_seqs, "gdn: state->ne[3] == n_seqs");
    assert_eq!(k_slots, 1, "gdn: only K == 1 (no rollback snapshots) is ported");
}

/// `ggml_compute_forward_gated_delta_net_one_chunk` (ops.cpp:10895-11088) for
/// K == 1: rows `ir0..ir1` of the `H_v * n_seqs` work items, one (head, seq)
/// pair each — the exact row split C's `ggml_compute_forward_gated_delta_net_f32`
/// hands out (ops.cpp:11049-11088 shards `nr = V->ne[1]*V->ne[3]`).
///
/// `dst` is the op's F32 output: `[S_v*H_v, n_tokens*n_seqs + S_v*n_seqs]`,
/// i.e. attn scores first (element (d, h, t, seq) at
/// `((seq*n_tokens + t)*H_v + h)*S_v + d`), then the final state per (h, seq).
///
/// SAFETY: raw pointers into the context storage; the caller guarantees F32
/// operands with the asserted layouts and that concurrent calls (different
/// `ir` ranges) touch disjoint rows.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_f32(
    dst: *mut u8,
    src_q: &GdnSrc,
    src_k: &GdnSrc,
    src_v: &GdnSrc,
    src_g: &GdnSrc,
    src_beta: &GdnSrc,
    src_state: &GdnSrc,
    ir0: usize,
    ir1: usize,
) {
    let s_v = src_v.ne[0] as usize;
    let h_v = src_v.ne[1] as usize;
    let n_tokens = src_v.ne[2] as usize;
    let n_seqs = src_v.ne[3] as usize;

    let _h_k = src_k.ne[1] as usize;
    let kda = src_g.ne[0] == src_v.ne[0];

    // per-seq stride in floats (seq iv3 starts at state + iv3*state_seq_stride)
    let state_seq_stride = (src_state.nb[3] / 4) as usize;

    // output layout: [attn_scores | new_state]
    let attn_score_elems = s_v * h_v * n_tokens * n_seqs;
    let attn_out_base = dst;
    let state_out_base = dst.add(attn_score_elems * 4);
    let state_in_base = src_state.ptr;

    let (nbq1, nbq2, nbq3) = (src_q.nb[1] as usize, src_q.nb[2] as usize, src_q.nb[3] as usize);
    let (nbk1, nbk2, nbk3) = (src_k.nb[1] as usize, src_k.nb[2] as usize, src_k.nb[3] as usize);
    let (nbv1, nbv2, nbv3) = (src_v.nb[1] as usize, src_v.nb[2] as usize, src_v.nb[3] as usize);
    let (nbg1, nbg2, nbg3) = (src_g.nb[1] as usize, src_g.nb[2] as usize, src_g.nb[3] as usize);
    let (nbb1, nbb2, nbb3) =
        (src_beta.nb[1] as usize, src_beta.nb[2] as usize, src_beta.nb[3] as usize);

    let rq3 = (src_v.ne[3] / src_q.ne[3]) as usize;
    let rk3 = (src_v.ne[3] / src_k.ne[3]) as usize;

    let scale = 1.0f32 / (s_v as f32).sqrt();

    let mut delta = vec![0f32; s_v];

    for ir in ir0..ir1 {
        let iv1 = ir % h_v; // head index
        let iv3 = ir / h_v; // sequence

        let iq1 = iv1 % src_q.ne[1] as usize;
        let ik1 = iv1 % src_k.ne[1] as usize;
        let iq3 = iv3 / rq3;
        let ik3 = iv3 / rk3;

        // K == 1: write directly into the single output slot (ops.cpp:10966-10968)
        let s_out = state_out_base.add((iv3 * h_v + iv1) * s_v * s_v * 4) as *mut f32;
        let s_in = state_in_base.add((iv3 * state_seq_stride + iv1 * s_v * s_v) * 4) as *const f32;
        std::ptr::copy_nonoverlapping(s_in, s_out, s_v * s_v);

        let mut attn_data = attn_out_base.add((iv3 * n_tokens * h_v + iv1) * s_v * 4) as *mut f32;

        for t in 0..n_tokens {
            let q_d = src_q.ptr.add(iq3 * nbq3 + t * nbq2 + iq1 * nbq1) as *const f32;
            let k_d = src_k.ptr.add(ik3 * nbk3 + t * nbk2 + ik1 * nbk1) as *const f32;
            let v_d = src_v.ptr.add(iv3 * nbv3 + t * nbv2 + iv1 * nbv1) as *const f32;
            let beta_val =
                *(src_beta.ptr.add(iv3 * nbb3 + t * nbb2 + iv1 * nbb1) as *const f32);
            let g_d = src_g.ptr.add(iv3 * nbg3 + t * nbg2 + iv1 * nbg1) as *const f32;

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            if kda {
                // precompute exp(g) into delta scratch, then S[i][:] *= exp(g[i])
                for i in 0..s_v {
                    delta[i] = (*g_d.add(i)).exp();
                }
                for j in 0..s_v {
                    let row = s_out.add(j * s_v);
                    for i in 0..s_v {
                        *row.add(i) *= delta[i];
                    }
                }
            } else {
                let gv = (*g_d).exp();
                for i in 0..s_v * s_v {
                    *s_out.add(i) *= gv;
                }
            }

            // delta[j] = (v[j] - dot(row j of M, k)) * beta
            for j in 0..s_v {
                let row = std::slice::from_raw_parts(s_out.add(j * s_v), s_v);
                let kk = std::slice::from_raw_parts(k_d, s_v);
                let sum = vec_dot_f32_c(s_v, row, kk);
                delta[j] = (*v_d.add(j) - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j]  =>  M[j][i] += delta[j]*k[i]
            for j in 0..s_v {
                let row = s_out.add(j * s_v);
                let dv = delta[j];
                for i in 0..s_v {
                    *row.add(i) = (*k_d.add(i)).mul_add(dv, *row.add(i));
                }
            }

            // attn_out[j] = dot(row j of M, q) * scale
            for j in 0..s_v {
                let row = std::slice::from_raw_parts(s_out.add(j * s_v), s_v);
                let qq = std::slice::from_raw_parts(q_d, s_v);
                *attn_data.add(j) = vec_dot_f32_c(s_v, row, qq) * scale;
            }

            attn_data = attn_data.add(s_v * h_v); // advance to next token
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive per-token reference (plain f64 accumulation, no SIMD lane
    /// structure) — catches layout/indexing bugs against the kernel.
    #[test]
    fn gdn_kernel_matches_naive() {
        let s_v = 8usize;
        let h_k = 2usize;
        let h_v = 4usize; // H_v % H_k == 0 (tiled broadcast)
        let t = 3usize;
        let n_seqs = 1usize;

        let mut rng = 12345u32;
        let mut rnd = move || {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng >> 8) as f32 / 16777216.0) * 1.0 - 0.5
        };

        let mut q = vec![0f32; s_v * h_k * t * n_seqs];
        let mut k = vec![0f32; s_v * h_k * t * n_seqs];
        let mut v = vec![0f32; s_v * h_v * t * n_seqs];
        let mut g = vec![0f32; 1 * h_v * t * n_seqs];
        let mut beta = vec![0f32; 1 * h_v * t * n_seqs];
        let mut state = vec![0f32; s_v * s_v * h_v * n_seqs];
        for x in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
            *x = rnd();
        }
        for x in g.iter_mut() {
            *x = -rnd().abs() * 0.5;
        }
        for x in beta.iter_mut() {
            *x = 0.2 + rnd().abs() * 0.3;
        }
        for x in state.iter_mut() {
            *x = rnd();
        }

        let attn_elems = s_v * h_v * t * n_seqs;
        let mut out = vec![f32::NAN; attn_elems + s_v * s_v * h_v * n_seqs];

        let plain = |slice: &[f32], ne: [i64; 4]| GdnSrc {
            ptr: slice.as_ptr() as *const u8,
            nb: [
                4,
                ne[0] as u64 * 4,
                (ne[0] * ne[1]) as u64 * 4,
                (ne[0] * ne[1] * ne[2]) as u64 * 4,
            ],
            ne,
        };
        let qs = plain(&q, [s_v as i64, h_k as i64, t as i64, n_seqs as i64]);
        let ks = plain(&k, [s_v as i64, h_k as i64, t as i64, n_seqs as i64]);
        let vs = plain(&v, [s_v as i64, h_v as i64, t as i64, n_seqs as i64]);
        let gs = plain(&g, [1, h_v as i64, t as i64, n_seqs as i64]);
        let bs = plain(&beta, [1, h_v as i64, t as i64, n_seqs as i64]);
        let ss = plain(&state, [s_v as i64, s_v as i64, h_v as i64, n_seqs as i64]);
        gdn_check(&qs, &ks, &vs, &gs, &bs, &ss, 1);

        unsafe {
            gdn_f32(out.as_mut_ptr() as *mut u8, &qs, &ks, &vs, &gs, &bs, &ss, 0, h_v * n_seqs);
        }

        // ---- naive (state in [S_v, S_v, H_v, n_seqs] row-major: S[i][j] at i + j*S_v)
        let scale = 1.0f32 / (s_v as f32).sqrt();
        let mut st = state.clone();
        let mut want = vec![0f32; attn_elems];
        for seq in 0..n_seqs {
            for hv in 0..h_v {
                let hk = hv % h_k;
                let s_base = (seq * h_v + hv) * s_v * s_v;
                let mut m = st[s_base..s_base + s_v * s_v].to_vec();
                for tt in 0..t {
                    let qo = (seq * t + tt) * h_k * s_v + hk * s_v;
                    let ko = qo;
                    let vo = (seq * t + tt) * h_v * s_v + hv * s_v;
                    let go = (seq * t + tt) * h_v + hv;
                    let qd = &q[qo..qo + s_v];
                    let kd = &k[ko..ko + s_v];
                    let vd = &v[vo..vo + s_v];
                    let bv = beta[go];
                    let gv = g[go].exp();
                    for x in m.iter_mut() {
                        *x *= gv;
                    }
                    let mut d = vec![0f64; s_v];
                    for j in 0..s_v {
                        // dot(row j of M, k) with M[i][j] at i + j*S_v
                        let mut sum = 0f64;
                        for i in 0..s_v {
                            sum += (m[i + j * s_v] * kd[i]) as f64;
                        }
                        d[j] = ((vd[j] as f64) - sum) as f32 as f64 * bv as f64;
                    }
                    for j in 0..s_v {
                        for i in 0..s_v {
                            m[i + j * s_v] += kd[i] * d[j] as f32;
                        }
                    }
                    for j in 0..s_v {
                        let mut sum = 0f64;
                        for i in 0..s_v {
                            sum += (m[i + j * s_v] * qd[i]) as f64;
                        }
                        want[(seq * t + tt) * h_v * s_v + hv * s_v + j] = sum as f32 * scale;
                    }
                }
                // the kernel writes the state slots in the same transposed
                // [S_v,S_v,H_v,seq] layout the graph views them with
                st[s_base..s_base + s_v * s_v].copy_from_slice(&m);
            }
        }

        for (i, (a, b)) in out.iter().zip(&want).enumerate() {
            let err = (a - b).abs();
            assert!(err < 1e-4, "attn[{i}] kernel {a} vs naive {b} (err {err})");
        }
        for (i, want_s) in st.iter().enumerate() {
            let got_s = out[attn_elems + i];
            let err = (got_s - want_s).abs();
            assert!(err < 1e-4, "state[{i}] kernel {got_s} vs naive {want_s} (err {err})");
        }
    }
}