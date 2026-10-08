//! ssm.rs — SSM / short-conv kernels. Owner: agent T.
//!
//! 对照 ggml-cpu/ops.cpp (pinned worktree @ bd4f514db1):
//!   * `ggml_compute_forward_ssm_conv`  (ops.cpp:9701-9768)  — 因果 1D 卷积
//!   * `ggml_compute_forward_ssm_scan`  (ops.cpp:9771-10008) — mamba2 选择性扫描
//!
//! Both are scalar ports of the C reference: the C code only takes the SIMD
//! path when GGML_SIMD is defined, and the reference build (-march=native)
//! does; the SIMD path differs from the scalar tail only in the *order* of the
//! d_state accumulation (lanes + hsum), i.e. the usual 1-2 ulp class recorded
//! in PARITY.md item 1. The scalar loop order below is exactly C's `for (i0 =
//! np; i0 < nc; ++i0)` tail with np = 0.
//!
//! Threading: the C kernels shard by rows (ssm_conv: `nr` rows of d_inner) and
//! by heads (ssm_scan: `nh`), which is what the callers in compute.rs reproduce
//! via `par_rows`.

/// AVX512 F32 SIMD geometry of the reference build (simd-mappings.h:452-453,
/// `GGML_F32_STEP 64` / `GGML_F32_EPR 16`, `GGML_F32_ARR = STEP/EPR = 4`).
/// The port reproduces that lane structure where the C kernels branch on
/// `GGML_SIMD` (ssm_scan's mamba2 loop).
pub const F32_STEP: usize = 64;
pub const F32_EPR: usize = 16;
pub const F32_ARR: usize = 4;

/// `ggml_compute_softplus_f32` (ggml-impl.h:107):
/// `(input > 20.0f) ? input : logf(1 + expf(input))`.
#[inline]
pub fn softplus_f32(input: f32) -> f32 {
    if input > 20.0f32 {
        input
    } else {
        (1.0f32 + input.exp()).ln()
    }
}

/// ggml_compute_forward_ssm_conv_f32 (ops.cpp:9703-9754).
///
/// src0 `sx` = conv_x  {d_conv - 1 + n_t, d_inner, n_seqs} (row-contiguous:
/// nb[1] == ne[0]*4, asserted in C); src1 `c` = conv1d.weight {d_conv, d_inner};
/// dst {d_inner, n_t, n_seqs}.
///
/// `ir0..ir1` are the row range (of `nr` = d_inner rows) this thread handles,
/// exactly C's `dr = (nr + nth - 1)/nth; ir0 = dr*ith; ir1 = MIN(ir0+dr, nr)`.
///
/// SAFETY: raw pointers into the context storage; the caller guarantees the
/// tensors are F32 with the asserted layout and that the row ranges written by
/// concurrent calls are disjoint (heads/rows are independent).
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_conv_f32(
    dst: *mut u8,
    dst_nb: [u64; 4],
    sx: *const u8,
    sx_nb: [u64; 4],
    c: *const u8,
    c_nb: [u64; 4],
    ir0: usize,
    ir1: usize,
    nc: usize,  // d_conv
    ncs: usize, // src0->ne[0] == d_conv - 1 + n_t
    _nr: usize, // d_inner (row count; the shard range already bounds the work)
    n_t: usize, // tokens per sequence
    n_s: usize, // number of sequences
) {
    let ir = ir1.saturating_sub(ir0);
    for i3 in 0..n_s {
        for i2 in 0..n_t {
            // {d_conv - 1 + n_t, d_inner, n_seqs} sliding window
            let s = sx.add(ir0 * sx_nb[1] as usize + i2 * sx_nb[0] as usize + i3 * sx_nb[2] as usize) as *const f32;
            let cc = c.add(ir0 * c_nb[1] as usize) as *const f32;
            let x = dst.add(ir0 * dst_nb[0] as usize + i2 * dst_nb[1] as usize + i3 * dst_nb[2] as usize) as *mut f32;
            for i1 in 0..ir {
                // rowwise dot product (C explicitly avoids ggml_vec_dot_f32:
                // its sum is in double precision). GCC's default
                // -ffp-contract=fast turns `sumf += s*c` into an FMA in the
                // reference build (-march=native → FMA available), verified
                // against parity/ssm_ref.bin: fmaf reproduces all 4 ssm_conv
                // cases bit-exactly, the unfused form misses 6/20 elements of
                // the first. f32::mul_add lowers to llvm.fma → libm fmaf
                // (correctly rounded) on non-FMA targets, bit-identical to the
                // hardware FMA the reference emits.
                let mut sumf = 0.0f32;
                for i0 in 0..nc {
                    sumf = (*s.add(i0 + i1 * ncs)).mul_add(*cc.add(i0 + i1 * nc), sumf);
                }
                *x.add(i1) = sumf;
            }
        }
    }
}

/// ggml_compute_forward_ssm_scan_f32 (ops.cpp:9773-10004).
///
/// src0 `s`   {d_state, dim, n_head, n_slots} (state cache, F32 contiguous)
/// src1 `x`   {dim, n_head, n_seq_tokens, n_seqs} (dim0-contiguous view)
/// src2 `dt`  {n_head, n_seq_tokens, n_seqs}
/// src3 `A`   {d_state, n_head} or {1, n_head}
/// src4 `B`   {d_state, n_group, n_seq_tokens, n_seqs}
/// src5 `C`   {d_state, n_group, n_seq_tokens, n_seqs}
/// src6 `ids` {n_seqs} I32
/// dst: 1-D {nelements(x) + K*d_state*dim*n_head*n_seqs}: y first
///      ({dim, n_head, n_seq_tokens, n_seqs} contiguous), then K snapshots of
///      the state ({d_state, dim, n_head} per seq, slot-major).
///
/// `ih0..ih1` is the head range of this thread (`dh = (nh + nth - 1)/nth`).
///
/// SAFETY: see `ssm_conv_f32`; heads are independent (state and y slices
/// disjoint per head).
#[allow(clippy::too_many_arguments)]
pub unsafe fn ssm_scan_f32(
    // dst + the per-seq state stride inside dst (== src0->nb[3] in C)
    dst: *mut u8,
    s: *const u8,
    s_nb3: u64,
    x: *const u8,
    x_nb: [u64; 4],
    dt: *const u8,
    dt_nb: [u64; 4],
    a: *const u8,
    a_ne0: usize, // A->ne[0] == 1 → scalar decay (mamba2)
    b: *const u8,
    b_nb: [u64; 4],
    cc: *const u8,
    cc_nb: [u64; 4],
    ids: *const i32,
    ih0: usize,
    ih1: usize,
    nc: usize, // d_state
    nr: usize, // dim (head_dim)
    nh: usize, // n_head
    ng: usize, // n_group
    nt: usize, // n_seq_tokens
    ns: usize, // n_seqs
    k: usize,  // op_params[0]
) {
    // can't use ggml_nbytes because src1 is not necessarily contiguous
    let s_off = nr * nh * nt * ns * std::mem::size_of::<f32>();

    for i3 in 0..ns {
        let ids3 = *ids.add(i3) as isize;
        let mut s0 = s.add(ids3 as usize * s_nb3 as usize);
        let s_seq = dst.add(i3 * s_nb3 as usize + s_off) as *mut u8;

        for i2 in 0..nt {
            let xp = x.add(i2 * x_nb[2] as usize + i3 * x_nb[3] as usize) as *const f32;
            let dtp = dt.add(i2 * dt_nb[1] as usize + i3 * dt_nb[2] as usize) as *const f32;
            let ap = a as *const f32;
            let bp = b.add(i2 * b_nb[2] as usize + i3 * b_nb[3] as usize) as *const f32;
            let cp = cc.add(i2 * cc_nb[2] as usize + i3 * cc_nb[3] as usize) as *const f32;
            let y = dst.add(i2 * (nh * nr * 4) + i3 * (nt * nh * nr * 4)) as *mut f32;

            if a_ne0 == 1 {
                // Mamba-2: scalar decay per head — dA hoisted out of the
                // state-wise loop
                for h in ih0..ih1 {
                    let dt_soft_plus = softplus_f32(*dtp.add(h));
                    let da = (dt_soft_plus * *ap.add(h)).exp();
                    let g = h / (nh / ng); // repeat_interleave

                    for i1 in 0..nr {
                        let ii = i1 + h * nr;
                        let x_dt = *xp.add(ii) * dt_soft_plus;
                        // C's GGML_SIMD branch (ops.cpp:9846-9907): with
                        // d_state >= GGML_F32_STEP (64 for the AVX512 build of
                        // the reference) the first `np = nc & ~63` states are
                        // accumulated in 4 AVX512 lane vectors, then reduced
                        // (`GGML_F32x16_REDUCE` + `_mm512_reduce_add_ps`) and
                        // finished by the scalar FMA tail. d_state == 128
                        // (granite-4.0-h-*) takes exactly that path, so the
                        // lane structure is reproduced here — verified against
                        // parity/ssm_ref.bin (bit-exact).
                        let np = nc & !(F32_STEP - 1);
                        let mut sumf;
                        if np > 0 {
                            let mut sum = [[0.0f32; F32_EPR]; F32_ARR];
                            let mut i = 0;
                            while i < np {
                                for j in 0..F32_ARR {
                                    let base = i + j * F32_EPR + ii * nc;
                                    let bg = i + j * F32_EPR + g * nc;
                                    for l in 0..F32_EPR {
                                        // C spells this MUL/MUL/ADD, but GCC
                                        // contracts the vector builtins under
                                        // -ffp-contract=fast: ax = fma(s0, dA,
                                        // B*x_dt) — verified against the dump
                                        // (the plain mul/add form differs 1 ulp)
                                        let ax = (*(s0.add((base + l) * 4) as *const f32))
                                            .mul_add(da, *bp.add(bg + l) * x_dt);
                                        let az = *cp.add(bg + l);
                                        // GGML_F32x16_FMA(sum, ax, az) = fmadd(ax, az, sum)
                                        sum[j][l] = ax.mul_add(az, sum[j][l]);
                                        *(s_seq.add((base + l) * 4) as *mut f32) = ax;
                                    }
                                }
                                i += F32_STEP;
                            }
                            // GGML_F32x16_REDUCE: (sum0+sum2) + (sum1+sum3)
                            let mut v = [0.0f32; F32_EPR];
                            for l in 0..F32_EPR {
                                v[l] = (sum[0][l] + sum[2][l]) + (sum[1][l] + sum[3][l]);
                            }
                            // _mm512_reduce_add_ps(v): a[i]=v[i]+v[i+8],
                            // b[i]=a[i]+a[i+4], res=(b0+b2)+(b1+b3)
                            let mut a8 = [0.0f32; 8];
                            for i in 0..8 {
                                a8[i] = v[i] + v[i + 8];
                            }
                            let mut b4 = [0.0f32; 4];
                            for i in 0..4 {
                                b4[i] = a8[i] + a8[i + 4];
                            }
                            sumf = (b4[0] + b4[2]) + (b4[1] + b4[3]);
                        } else {
                            sumf = 0.0f32;
                        }
                        // scalar tail (np..nc), FMA-contracted as in the C build
                        for i0 in np..nc {
                            let i = i0 + ii * nc;
                            let ig = i0 + g * nc;
                            // state = prev_state * dA + dB * x
                            let state = (*(s0.add(i * 4) as *const f32)).mul_add(da, *bp.add(ig) * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf = state.mul_add(*cp.add(ig), sumf);
                            *(s_seq.add(i * 4) as *mut f32) = state;
                        }
                        *y.add(ii) = sumf;
                    }
                }
            } else {
                // Mamba-1: element-wise decay factor for the states
                for h in ih0..ih1 {
                    let dt_soft_plus = softplus_f32(*dtp.add(h));
                    let g = h / (nh / ng);

                    for i1 in 0..nr {
                        let ii = i1 + h * nr;
                        let x_dt = *xp.add(ii) * dt_soft_plus;
                        let mut sumf = 0.0f32;
                        for i0 in 0..nc {
                            let i = i0 + ii * nc;
                            let ig = i0 + g * nc;
                            // C: `state = (s0[i] * expf(dt_sp*A[i0+h*nc])) + B[ig]*x_dt`,
                            // `sumf += state*C[ig]` — both FMA-contracted by the
                            // reference build (GCC -ffp-contract=fast default)
                            let state = (*(s0.add(i * 4) as *const f32))
                                .mul_add((dt_soft_plus * *ap.add(i0 + h * nc)).exp(), *bp.add(ig) * x_dt);
                            sumf = state.mul_add(*cp.add(ig), sumf);
                            *(s_seq.add(i * 4) as *mut f32) = state;
                        }
                        *y.add(ii) = sumf;
                    }
                }
            }

            // rollback snapshots (K > 1; K == 1 in this port — recurrent
            // rollback / n_rs_seq is not modelled, see graph_arch.rs)
            let slot = nt - 1 - i2;
            if k > 1 && slot > 0 && slot < k {
                let s_snapshot = dst.add(s_off + (slot * ns + i3) * s_nb3 as usize);
                for h in ih0..ih1 {
                    // C: memcpy(s_snapshot + h*src0->nb[2], s + h*src0->nb[2],
                    // src0->nb[2]) — the per-head stride is nb[2] (== nc*nr*4),
                    // NOT nb[3]
                    std::ptr::copy_nonoverlapping(
                        s_seq.add(h * nc * nr * 4),
                        s_snapshot.add(h * nc * nr * 4),
                        nc * nr * 4,
                    );
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s_seq as *const u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exactness report against parity/ssm_ref.bin (dumped by
    /// parity/ref_ssm_dump.c from the reference build through the real graph
    /// path, `ggml_graph_compute_with_ctx`). Kind 10 = ssm_conv (scalar even in
    /// the SIMD build — C explicitly avoids ggml_vec_dot_f32 → expect 0 bit
    /// diffs), kind 11 = ssm_scan (the reference .so runs the AVX512 lane
    /// accumulation, this port the scalar tail → the 1-2 ulp class of
    /// PARITY.md item 1).
    fn max_ulp_diff(got: &[f32], want: &[f32]) -> (usize, u32, f32) {
        let mut diffs = 0usize;
        let mut worst = 0u32;
        let mut worst_rel = 0f32;
        for (a, b) in got.iter().zip(want) {
            if a.to_bits() != b.to_bits() {
                diffs += 1;
                let d = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                worst = worst.max(d);
            }
            worst_rel = worst_rel.max((a - b).abs() / b.abs().max(1e-6));
        }
        (diffs, worst, worst_rel)
    }

    #[test]
    fn ssm_kernels_match_reference_dump() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/ssm_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("SKIP: {path} missing (build parity/ref_ssm_dump.c)");
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
        let mut cases = 0;
        loop {
            let magic = u32at(&mut c);
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x314D_5353, "section magic 'SSM1'");
            let kind = u32at(&mut c);
            let nd = u32at(&mut c) as usize;
            let mut dims = Vec::new();
            for _ in 0..nd {
                dims.push(u32at(&mut c) as usize);
            }
            match kind {
                10 => {
                    let (d_conv, d_inner, n_t, n_s) = (dims[0], dims[1], dims[2], dims[3]);
                    let ncs = d_conv - 1 + n_t;
                    let sx = f32s(&mut c, ncs * d_inner * n_s);
                    let cv = f32s(&mut c, d_conv * d_inner);
                    let want = f32s(&mut c, d_inner * n_t * n_s);

                    let mut ctx = crate::Context::new();
                    let sx_t = ctx.new_tensor_3d(
                        crate::types::GgmlType::F32,
                        ncs as i64,
                        d_inner as i64,
                        n_s as i64,
                    );
                    ctx.arena_resize_tensor(sx_t);
                    ctx.with_f32_mut(sx_t, |p| p.copy_from_slice(&sx)).unwrap();
                    let c_t = ctx.new_tensor_2d(
                        crate::types::GgmlType::F32,
                        d_conv as i64,
                        d_inner as i64,
                    );
                    ctx.arena_resize_tensor(c_t);
                    ctx.with_f32_mut(c_t, |p| p.copy_from_slice(&cv)).unwrap();
                    let r = ctx.ssm_conv(sx_t, c_t);
                    let mut g = crate::Graph::new(8);
                    g.build_forward(&ctx, r);
                    crate::compute::graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap().to_vec();
                    let (diffs, worst, rel) = max_ulp_diff(&got, &want);
                    println!(
                        "ssm_conv d_conv={d_conv} d_inner={d_inner} n_t={n_t} n_s={n_s}: \
                         {diffs}/{} differ, worst {worst} ulp, max rel {rel:.2e}",
                        want.len()
                    );
                    assert!(rel < 1e-6, "ssm_conv far off: rel {rel:.3e}");
                }
                11 => {
                    let (d_state, hd, nh, ng, nt, ns, k, a_ne0) = (
                        dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7],
                    );
                    let slots = ns + 1;
                    let s = f32s(&mut c, d_state * hd * nh * slots);
                    let x = f32s(&mut c, hd * nh * nt * ns);
                    let dt = f32s(&mut c, nh * nt * ns);
                    let a = f32s(&mut c, a_ne0 * nh);
                    let b = f32s(&mut c, d_state * ng * nt * ns);
                    let cc = f32s(&mut c, d_state * ng * nt * ns);
                    let ids: Vec<i32> = {
                        let (a2, rest) = c.split_at(ns * 4);
                        c = rest;
                        a2.chunks_exact(4).map(|w| i32::from_le_bytes(w.try_into().unwrap())).collect()
                    };
                    let want = f32s(&mut c, x.len() + k * d_state * hd * nh * ns);

                    let mut ctx = crate::Context::new();
                    let mk = |ctx: &mut crate::Context, ne: [i64; 4], vals: &[f32]| {
                        let t = ctx.new_tensor(crate::types::GgmlType::F32, ne);
                        ctx.arena_resize_tensor(t);
                        ctx.with_f32_mut(t, |p| p.copy_from_slice(vals)).unwrap();
                        t
                    };
                    let s_t = mk(&mut ctx, [d_state as i64, hd as i64, nh as i64, slots as i64], &s);
                    let x_t = mk(&mut ctx, [hd as i64, nh as i64, nt as i64, ns as i64], &x);
                    let dt_t = mk(&mut ctx, [nh as i64, nt as i64, ns as i64, 1], &dt);
                    let a_t = mk(&mut ctx, [a_ne0 as i64, nh as i64, 1, 1], &a);
                    let b_t = mk(&mut ctx, [d_state as i64, ng as i64, nt as i64, ns as i64], &b);
                    let c_t = mk(&mut ctx, [d_state as i64, ng as i64, nt as i64, ns as i64], &cc);
                    let ids_t = ctx.new_tensor_1d(crate::types::GgmlType::I32, ns as i64);
                    ctx.arena_resize_tensor(ids_t);
                    ctx.with_i32_mut(ids_t, |p| p.copy_from_slice(&ids)).unwrap();

                    let r = ctx.ssm_scan(s_t, x_t, dt_t, a_t, b_t, c_t, ids_t, k as i64);
                    let mut g = crate::Graph::new(8);
                    g.build_forward(&ctx, r);
                    crate::compute::graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap().to_vec();
                    let (diffs, worst, rel) = max_ulp_diff(&got, &want);
                    let ny = x.len();
                    let (dy, ds) = (
                        got[..ny].iter().zip(&want[..ny]).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
                        got[ny..].iter().zip(&want[ny..]).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
                    );
                    let max_abs = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    println!(
                        "ssm_scan d_state={d_state} hd={hd} nh={nh} ng={ng} nt={nt} ns={ns} K={k} \
                         A_ne0={a_ne0}: {diffs}/{} differ (y {dy}/{ny}, state {ds}/{}), \
                         worst {worst} ulp, max rel {rel:.2e}, max abs {max_abs:.3e} (|want|max {scale:.3e})",
                        want.len(),
                        want.len() - ny
                    );
                    assert_eq!(diffs, 0, "ssm_scan must be bit-exact vs the reference dump: {diffs} differ, max rel {rel:.3e}");
                }
                other => panic!("unknown kind {other}"),
            }
            cases += 1;
        }
        assert!(cases >= 8, "expected the full dumper output, got {cases} sections");
    }

    /// naive ssm_conv: y[i1, i2, i3] = sum_{i0} sx[i2+i0, i1, i3] * c[i0, i1]
    #[test]
    fn ssm_conv_matches_naive() {
        let (nc, d_inner, n_t, n_s) = (3usize, 5usize, 4usize, 2usize);
        let ncs = nc - 1 + n_t;
        let sx: Vec<f32> = (0..ncs * d_inner * n_s).map(|i| (i as f32 * 0.31).sin()).collect();
        let c: Vec<f32> = (0..nc * d_inner).map(|i| (i as f32 * 0.77).cos()).collect();
        let mut dst = vec![0f32; d_inner * n_t * n_s];

        // C layout: sx {d_conv-1+n_t, d_inner, n_seqs} → nb = [4, ne0*4, ...]
        let sx_nb = [4u64, (ncs * 4) as u64, (ncs * d_inner * 4) as u64, 0];
        let c_nb = [4u64, (nc * 4) as u64, 0, 0];
        let dst_nb = [4u64, (d_inner * 4) as u64, (d_inner * n_t * 4) as u64, 0];
        unsafe {
            ssm_conv_f32(
                dst.as_mut_ptr() as *mut u8,
                dst_nb,
                sx.as_ptr() as *const u8,
                sx_nb,
                c.as_ptr() as *const u8,
                c_nb,
                0,
                d_inner,
                nc,
                ncs,
                d_inner,
                n_t,
                n_s,
            )
        };

        for i3 in 0..n_s {
            for i2 in 0..n_t {
                for i1 in 0..d_inner {
                    let want: f32 = (0..nc)
                        .map(|i0| sx[(i2 + i0) + i1 * ncs + i3 * ncs * d_inner] * c[i0 + i1 * nc])
                        .sum();
                    let got = dst[i1 + i2 * d_inner + i3 * d_inner * n_t];
                    assert!((got - want).abs() < 1e-6, "({i1},{i2},{i3}): {got} vs {want}");
                }
            }
        }
    }

    /// naive ssm_scan (mamba2 branch, A->ne[0] == 1), K = 1:
    /// state = state*dA + B*x*dt ; y = state·C ; final state written after y.
    #[test]
    fn ssm_scan_mamba2_matches_naive() {
        let (d_state, hd, nh, ng, nt, ns, t_slots) = (4usize, 3usize, 2usize, 1usize, 3usize, 2usize, 2usize);
        let d_inner = hd * nh;
        let mut st = vec![0f32; d_state * d_inner * t_slots];
        for (i, v) in st.iter_mut().enumerate() {
            *v = (i as f32 * 0.13).sin() * 0.5;
        }
        let x: Vec<f32> = (0..d_inner * nh * nt * ns).map(|i| (i as f32 * 0.19).cos()).collect();
        let dt: Vec<f32> = (0..nh * nt * ns).map(|i| 0.1 + (i as f32 * 0.05)).collect();
        let a: Vec<f32> = (0..nh).map(|i| -(1.0 + i as f32 * 0.1)).collect();
        let b: Vec<f32> = (0..d_state * ng * nt * ns).map(|i| (i as f32 * 0.23).sin()).collect();
        let cvals: Vec<f32> = (0..d_state * ng * nt * ns).map(|i| (i as f32 * 0.29).cos()).collect();
        let ids = [0i32, 1];
        // dst = y (nelements(x) = hd*nh*nt*ns) + one state block per seq
        let dst_len = hd * nh * nt * ns + d_state * d_inner * ns;
        let mut dst = vec![0f32; dst_len];

        // x is a {hd, nh, nt, ns} view with nb0 = 4, nb1 = hd*4 and the token
        // stride of the parent (the C xBC layout)
        let x_nb = [4u64, (hd * 4) as u64, (d_inner * 4) as u64, (d_inner * nt * 4) as u64];
        let dt_nb = [4u64, (nh * 4) as u64, (nh * nt * 4) as u64, 0];
        let b_nb = [4u64, (d_state * 4) as u64, (d_state * ng * 4) as u64, (d_state * ng * nt * 4) as u64];
        // the dst state stride == src0->nb[3]
        let s_nb3 = (d_state * d_inner * 4) as u64;
        unsafe {
            ssm_scan_f32(
                dst.as_mut_ptr() as *mut u8,
                st.as_ptr() as *const u8,
                s_nb3,
                x.as_ptr() as *const u8,
                x_nb,
                dt.as_ptr() as *const u8,
                dt_nb,
                a.as_ptr() as *const u8,
                1,
                b.as_ptr() as *const u8,
                b_nb,
                cvals.as_ptr() as *const u8,
                b_nb,
                ids.as_ptr(),
                0,
                nh,
                d_state,
                hd,
                nh,
                ng,
                nt,
                ns,
                1,
            )
        };
        // naive
        for seq in 0..ns {
            let mut state = st[seq * d_state * d_inner..(seq + 1) * d_state * d_inner].to_vec();
            for t in 0..nt {
                for h in 0..nh {
                    let dtv = softplus_f32(dt[h + t * nh + seq * nh * nt]);
                    let da = (dtv * a[h]).exp();
                    // C: g = h / (nh / ng) — repeat_interleave
                    let g = h / (nh / ng);
                    for d in 0..hd {
                        let ii = d + h * hd;
                        let xv = x[ii + t * d_inner + seq * d_inner * nt];
                        let x_dt = xv * dtv;
                        let mut sumf = 0.0f32;
                        for s0 in 0..d_state {
                            let i = s0 + ii * d_state;
                            let ig = s0 + g * d_state;
                            let sv = state[i] * da + b[ig + t * d_state * ng + seq * d_state * ng * nt] * x_dt;
                            sumf += sv * cvals[ig + t * d_state * ng + seq * d_state * ng * nt];
                            state[i] = sv;
                        }
                        let got = dst[ii + t * d_inner + seq * d_inner * nt];
                        assert!((got - sumf).abs() < 1e-6, "seq {seq} t {t} ii {ii}: {got} vs {sumf}");
                    }
                }
            }
            // final state after the y block (offset s_off = nelements(x)), the
            // per-seq stride inside dst is src0->nb[3] (K == 1 here)
            let base = hd * nh * nt * ns + seq * d_state * d_inner;
            for i in 0..d_state * d_inner {
                let got = dst[base + i];
                // state is {d_state, dim, nh}: same flat index as ii*d_state+s0
                assert!((got - state[i]).abs() < 1e-6, "state seq {seq} [{i}]: {got} vs {}", state[i]);
            }
        }
    }
}