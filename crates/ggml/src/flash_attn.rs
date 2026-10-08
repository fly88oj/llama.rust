//! flash_attn.rs — `GGML_OP_FLASH_ATTN_EXT` (CPU, scalar/generic path).
//!
//! 1:1 port of the reference CPU kernel (pinned worktree @ bd4f514db1):
//!   * builder      `ggml_flash_attn_ext`                                ggml.c:5497
//!   * dispatch     `ggml_compute_forward_flash_attn_ext`                ops.cpp:9348
//!   * path select  `ggml_compute_forward_flash_attn_ext_f16`            ops.cpp:9212
//!   * core kernel  `ggml_compute_forward_flash_attn_ext_f16_one_chunk`  ops.cpp:8614
//!   * chunk merge  `ggml_flash_attn_ext_reduce_partials`                ops.cpp:9142
//!
//! ## Registration / wiring
//! Compiled as `crate::flash_attn` (`lib.rs: pub mod flash_attn;`).
//! Companion edits (verified together with the tests below):
//!   * `types.rs`: `MAX_SRC` 3 → 4 (the node carries q/k/v/mask in src[0..3];
//!     ggml.h's GGML_MAX_SRC is 10 — 4 is the minimum, larger works too since
//!     the builder fills a `[None; MAX_SRC]` array).
//!   * `ops.rs`: `Context::flash_attn_ext` (builder, mirrors ggml.c:5497) and
//!     `init_op` made const-generic over the src array length so the ~44 legacy
//!     3-slot call sites stay unchanged.
//!   * `compute.rs`: `GgmlOp::FlashAttnExt => self.forward_flash_attn_ext(...)`
//!     plus that forward fn (`par_rows` sharding of `one_chunk`).
//!   * `graph.rs` test `forward_order`: two `src = [...]` literals need a 4th
//!     `None`.
//!
//! ## Layouts (C `ne`/`nb` convention, dim 0 contiguous)
//! Careful: in the C text `q`'s dim1 is the **token** axis and dim2 the **head**
//! axis (the `mul_mat(k, q)` order — `dst->ne[2] == neq1 == N` and
//! `dst = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }` pin this down):
//! ```text
//!   q    [DK, T,    H,    S]      k [DK, S_kv, H_kv, S]   v [DV, S_kv, H_kv, S]
//!   mask [S_kv, T, mask_ne2, mask_ne3]  (F16, contiguous; broadcast to q heads by
//!                                        `iq2 % mask_ne2`, to sequences by `iq3 % mask_ne3`)
//!   dst  [DV, H, T, S]            (builder's permute(0,2,1,3), ggml.c:5526)
//!   row index ir = iq3*T*H + iq2*T + iq1   (iq1 = token, fastest varying)
//! ```
//! GQA/MQA: k/v heads are broadcast to q heads by `ik2 = iq2 / (H/H_kv)`
//! (`rk2 = neq2/nek2 = H/H_kv`); similarly for sequences (`rk3`).
//! `mask[t][kv]` is indexed `iq1*mask.nb[1] + (iq2 % mask.ne[2])*mask.nb[2]`.
//!
//! ## Numerics (what this port reproduces)
//! The C kernel is an **online softmax** over the KV dimension: a running max
//! `M`, a running sum `S` and an unnormalized output `VKQ`, updated per KV
//! element `ic` (ops.cpp:8744-8802) with
//! ```text
//!   s = dot(k[ic], q)                             // f32 (f64 accumulator)
//!   s = s*scale;  s = logit_softcap*tanhf(s) if logit_softcap != 0
//!   mv = slope * fp16(mask[ic])                   // 0 without mask
//!   if mv == -INFINITY: continue                  // masked entry: skipped entirely
//!   s += mv
//!   ms = 1; vs = 1
//!   if s > M { M = s; ms = expf(Mold - M); VKQ *= ms } else { vs = expf(s - M) }
//!   VKQ += v[ic]*vs;  S = S*ms + vs
//! ```
//! then `VKQ /= S` with `S_inv = S == 0 ? 0 : 1/S` (ops.cpp:8805), so a
//! fully-masked row yields **zeros**, not NaN. `scale` was pre-divided by
//! `logit_softcap` (ops.cpp:8675-8677), and the ALiBi slope is
//! `m0^(h+1)` / `m1^(2*(h-n_head_log2)+1)` with `n_head_log2 = 1 << floor(log2(H))`,
//! `m0 = 2^(-max_bias/n_head_log2)`, `m1 = 2^(-(max_bias/2)/n_head_log2)`
//! (ops.cpp:8685-8697; same formula as `soft_max_ext`).
//!
//! ## Path selection (ops.cpp:9257-9340)
//! ```text
//!   use_split_kv_path = (neq1 == 1 && neq3 == 1) && kv is F32/F16 && k.type == v.type
//!                       && q.type == F32 && nek1 >= 512        (ops.cpp:9261)
//!   use_tiled         = q.type == F32 && kv is F32/F16 && k.type == v.type
//!                       && neq1 >= Q_TILE_SZ (64) && DV % f32_epr == 0   (ops.cpp:9318)
//! ```
//! `use_tiled` is what a prefill hits (T >= 64); everything else falls back to
//! `one_chunk`. `!use_ref` guards both in C — `use_ref` is a debug hook
//! (`ggml_threadpool`/env, default false) and is not modelled here.
//!
//! ## Divergences from the C text (deliberate; see report)
//! 1. **VKQ16 accumulator (F16 V)**: modelled exactly as the reference's
//!    `GGML_F16_VEC` (AVX512 `F32Cx16`): f32 arithmetic on `vcvtph2ps` loads
//!    with one RNE `vcvtps2ph $0` store rounding per scale/mad
//!    (ops.cpp:8712-8716/8773-8795; see the contraction profile below); the
//!    `_tiled` kernel accumulates in F32 exactly as C does there.
//! 2. **KQ dot (one_chunk)**: f32 products accumulated in f64 (the C scalar
//!    vec_dot tails, `ggml_float sumf`). C additionally quantizes Q to the K
//!    `vec_dot_type` first (`q_to_vec_dot`: Q→F16 when K is F16) and its SIMD
//!    builds accumulate in the lane type; the AVX512 tiled path packs Q/K to F32
//!    and uses `simd_gemm` instead — the [tiled] kernel below matches that.
//! 3. **expf**: scalar libm `expf()` → Rust `f32::exp` (glibc expf). The tiled
//!    kernel's vector softmax instead uses the reference's `ggml_v_expf`
//!    (AVX512F+DQ polynomial, vec.h:1172) and `_mm512_reduce_add_ps` reduction
//!    tree, both replicated elementwise in [`v_expf`]/[`reduce_add_ps`] — see
//!    the parity table in the report.
//! 4. **K type**: F16/F32 only. C also supports quantized K through `vec_dot_type`
//!    + `from_float` (quantized KV cache) — asserted unsupported here.
//! 5. `n_kv_max` (op_params[4]), `use_ref` and the NUMA/chunk work-stealing of
//!    the C thread loop are not represented (`one_chunk`/`tiled` row ranges are
//!    split by the caller; the split-KV path's chunk *count* — the only part
//!    its values depend on — is passed through as `n_chunks = nth`).
//!    `sinks` (src[4]) are supported in both kernels.
//! 6. **tiled SIMD**: on AVX512 hosts the tile's vector loops run as the
//!    reference's 16-lane kernels (`simd_gemm`'s `_mm512_fmadd_ps` register
//!    tiles, `ggml_v_expf`, `_mm512_reduce_add_ps`, `ggml_vec_scale_f32`,
//!    `ggml_vec_add_f32`, `vcvtph2ps` — see `simd_x86.rs`'s AVX512 section);
//!    every one of them is per-lane, so the elementwise lane-port kept here as
//!    the non-AVX512 fallback (`tiled_impl(use_simd=false)`) is bit-identical
//!    by construction — asserted by `tiled_simd_matches_scalar_bit_exact` and
//!    pinned against the reference by the dump tests below.
//!
//! ## Measured against the reference binary
//! (AVX512 `-march=native` build, through the *public* API:
//! `ggml_flash_attn_ext` + `ggml_graph_compute_with_ctx(..., 1)`; harness
//! `parity/ref_fa_dump.c`, tests `ref_dump_matches_all_cases` /
//! `ref_dump_bitexact` / `ref_dump_path_affinity`, plus the row-shape probe
//! `parity/ref_fa_probe.c` / `ref_probe_dump_bitexact` / `row_shape_invariance`).
//! "bitexact" is the count of f32 outputs whose bit pattern equals the
//! reference's.
//!
//! | # | shape | C path | bitexact |
//! |---|---|---|---|
//! | 0 | D=16 T=3 S_kv=5 **F32** KV + mask | one_chunk | **192/192** |
//! | 1 | D=64 T=64 S_kv=64 F16 KV + mask | tiled | **16384/16384** |
//! | 2 | D=64 T=128 S_kv=128 F16 KV + mask | tiled | **32768/32768** |
//! | 3 | D=32 T=65 S_kv=130 F16 KV + mask | tiled | **8320/8320** |
//! | 4 | D=64 T=64 S_kv=64 **F32** KV + mask | tiled | **16384/16384** |
//! | 5 | D=64 T=128 S_kv=200 F16 KV + mask + **sinks** | tiled | **32768/32768** |
//! | 6 | D=64 T=128 S_kv=200 F16 KV + mask + softcap=15 | tiled | **32768/32768** |
//! | 7 | D=64 **DV=128** T=64 S_kv=96 F16 KV + mask | tiled | **32768/32768** |
//! | 8 | D=32 T=32 S_kv=96 F16 KV + mask | one_chunk | **4096/4096** |
//! | 9 | D=64 T=64 S_kv=64 F16 KV, no mask | tiled | **16384/16384** |
//! | 10 | D=64 T=64 S_kv=64 F16 KV + mask, max_bias=8 (ALiBi) | tiled | **16384/16384** |
//! | 11 | D=64 T=64 S_kv=64 F16 KV + mask, window off=0 | tiled | **16384/16384** |
//! | 12 | D=64 T=64 S_kv=70 F16, **no mask + softcap=1e-4** (padded KV tail) | tiled | **16384/16384** |
//! | 13 | D=64 H=64 H_kv=8 T=1 S_kv=21 F16 + **sinks** (gpt-oss decode shape) | one_chunk | **4096/4096** |
//! | 14 | D=64 H=64 H_kv=8 T=5 S_kv=5 F16 + mask + sinks (gpt-oss prefill) | one_chunk | **20480/20480** |
//!
//! Every dump case — tiled **and** one_chunk — is bit-identical to the
//! reference, including the `S[-1] = -softcap` padded-KV-tail quirk of case 12
//! (which is also what identifies the path: case 12 differs from one_chunk by
//! ~5.6x, i.e. only the tiled kernel explains the reference's output).
//!
//! ## Contraction profile of the reference binary (what makes it reproducible)
//! GCC `-ffp-contract=fast` fuses *some* of the C's `a*b + c` lines and not
//! others, per inlined body — each of the following was read off the .so
//! (`ggml_compute_forward_flash_attn_ext`, objdump) and is reproduced exactly:
//!   * inner-loop `S = S*ms + vs` (ops.cpp:8802): **fused** — `vfmadd132ss`
//!     (.so c1a8b);
//!   * sinks `S = S*ms + vs` (ops.cpp:8824): **NOT fused** — the inlined
//!     `ggml_vec_scale_f32` boundary keeps `S*ms` a `vmulss` (.so c1ba8) and
//!     the `+ vs` a `vaddss` (.so c1cd7). The 1-2 ulp "FA row-shape tail"
//!     (gpt-oss teacher-forced band, Ds4 MTP verify argmax) was exactly this
//!     line computed as an FMA in the port;
//!   * split-KV merge (ops.cpp:9178-9181): **fused as
//!     `fma(old, scale_old, chunk*scale_new)`** — the `chunk*scale_new`
//!     product rounds once (`vmulss`), then one fused multiply-add
//!     (`vfmadd231ss` .so c345e / AVX512 lanes c648d / S line c65e2);
//!   * `ggml_vec_mad_f32` (F32 V): `GGML_F32_VEC_FMA = _mm512_fmadd_ps` —
//!     fused per lane (vfmadd213ps .so c27d0), scalar tail likewise;
//!   * `ggml_vec_mad_f16` / `ggml_vec_scale_f16` (F16 V): f32 FMA / mul on
//!     cvtph2ps loads with one RNE `vcvtps2ph $0` store rounding (.so c0ef0).
//!
//! ## Row-count (in)variance — per shape, like the reference
//! `one_chunk` and `tiled` are per-row kernels: a T-row batch and its 1-row
//! slices are bit-identical (asserted by `row_shape_invariance`; proven of the
//! reference by `parity/ref_fa_probe.c`). The **split-KV decode path**
//! (`neq1 == 1 && nek1 >= 512`, ops.cpp:9261) is the one place the reference's
//! values depend on the *thread count*: its KV-chunk merge is a different
//! rounding than the whole-range scan, so at `nth >= 2` a 1-row decode and a
//! T-row batch legitimately differ (measured on the reference: T=4 S_kv=600
//! batch vs its 1-row slices at nth=8 differ in every element). The port
//! dispatches that path per shape with `n_chunks = nth`
//! (`flash_attn_ext_dispatch_n` / `compute.rs`), so it matches the reference
//! *per shape* rather than being invariant there.

use crate::types::GgmlType;
use half::f16;

// ======================================================================
// byte-addressed element reads — C dereferences via (char *) + nb strides
// ======================================================================

#[inline]
fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_ne_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_ne_bytes([b[off], b[off + 1]])).to_f32()
}

// ======================================================================
// Src / Dst — strided byte views (the Rust stand-in for ggml_tensor data/nb)
// ======================================================================

/// Read-only view of one operand. `data` starts at the tensor's *first element*
/// (storage base + view offset); elements are addressed with the C `nb` strides.
#[derive(Clone, Copy)]
pub struct Src<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub ne: [i64; 4],
    pub nb: [u64; 4],
}

impl<'a> Src<'a> {
    /// Dense view, nb derived exactly like `ggml_new_tensor_impl`.
    pub fn dense(data: &'a [u8], ty: GgmlType, ne: [i64; 4]) -> Src<'a> {
        let nb0 = ty.type_size() as u64;
        let blck = ty.blck_size() as i64;
        let nb1 = nb0 * (ne[0] / blck) as u64;
        let nb2 = nb1 * ne[1] as u64;
        let nb3 = nb2 * ne[2] as u64;
        Src { data, ty, ne, nb: [nb0, nb1, nb2, nb3] }
    }

    /// Strided view (KV-cache / view tensors carry their own nb).
    pub fn strided(data: &'a [u8], ty: GgmlType, ne: [i64; 4], nb: [u64; 4]) -> Src<'a> {
        Src { data, ty, ne, nb }
    }

    /// C: `(const char *) t->data + i1*nb1 + i2*nb2 + i3*nb3`
    #[inline]
    pub fn row_off(&self, i1: i64, i2: i64, i3: i64) -> usize {
        (i1 as u64 * self.nb[1] + i2 as u64 * self.nb[2] + i3 as u64 * self.nb[3]) as usize
    }

    /// Element `i0` (walked with nb[0]) of the row that starts at `row_off`.
    #[inline]
    pub fn row_elem(&self, row_off: usize, i0: usize) -> f32 {
        let off = row_off + i0 * self.nb[0] as usize;
        match self.ty {
            GgmlType::F32 => f32_at(self.data, off),
            GgmlType::F16 => f16_at(self.data, off),
            other => panic!("fattn: unsupported type {other:?}"),
        }
    }

    fn is_contiguous(&self) -> bool {
        self.nb[0] == self.ty.type_size() as u64
            && self.nb[1] == self.nb[0] * self.ne[0] as u64
            && self.nb[2] == self.nb[1] * self.ne[1] as u64
            && self.nb[3] == self.nb[2] * self.ne[2] as u64
    }
}

/// Writable view of the destination: nb[0] (== 4) and nb[1] (head stride) are
/// the only strides entering the C write formula.
pub struct Dst<'a> {
    pub data: &'a mut [u8],
    pub ne: [i64; 4],
    pub nb: [u64; 4],
}

impl<'a> Dst<'a> {
    /// Dense F32 destination of shape `ne`.
    pub fn dense_f32(data: &'a mut [u8], ne: [i64; 4]) -> Dst<'a> {
        let nb1 = 4 * ne[0] as u64;
        let nb2 = nb1 * ne[1] as u64;
        let nb3 = nb2 * ne[2] as u64;
        Dst { data, ne, nb: [4, nb1, nb2, nb3] }
    }

    /// C (ops.cpp:8815-8820):
    /// `memcpy(dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ, nb1)`
    /// — the dst `ne1`/`ne2` are the *permuted* (heads, tokens) dims.
    pub fn write_row(&mut self, i1: i64, i2: i64, i3: i64, row: &[f32]) {
        assert_eq!(self.nb[0], 4, "fattn: dst nb0 != sizeof(float)");
        let off = ((i3 * self.ne[2] * self.ne[1] + i2 + i1 * self.ne[1]) as u64 * self.nb[1]) as usize;
        // memcpy of nb1 bytes — a bytewise copy of the same f32 values
        self.data[off..off + row.len() * 4].copy_from_slice(bytemuck::cast_slice(row));
    }
}

// ======================================================================
// FlashAttn — inputs + op_params (dst->src[0..4], op_params[0..2])
// ======================================================================

/// All inputs of one `GGML_OP_FLASH_ATTN_EXT` node.
pub struct FlashAttn<'a> {
    pub q: Src<'a>,
    pub k: Src<'a>,
    pub v: Src<'a>,
    pub mask: Option<Src<'a>>,
    /// op_params[0] f32
    pub scale: f32,
    /// op_params[1] f32
    pub max_bias: f32,
    /// op_params[2] f32
    pub logit_softcap: f32,
    /// per-head attention sinks (C src[4], `ggml_flash_attn_ext_add_sinks`).
    /// `None` = no sinks. Applied only on the FIRST kv chunk (ops.cpp:8812).
    pub sinks: Option<&'a [f32]>,
}

impl<'a> FlashAttn<'a> {
    /// The `GGML_ASSERT`s of the builder (ggml.c:5506-5518) plus the index-range
    /// preconditions the kernel's addressing implies.
    pub fn validate(&self) {
        let (q, k, v) = (&self.q, &self.k, &self.v);
        assert_eq!(k.ne[0], q.ne[0], "fattn: neq0 == nek0 (ggml_can_mul_mat)");
        // C asserts `neq0 == DK && nek0 == DK && nev0 == DV` (ops.cpp:8646-8651) —
        // the value head dim DV is NOT tied to DK (T=64/D=64/DV=128 is legal).
        assert_eq!(v.ne[0], self.dv(), "fattn: nev0 == DV");
        assert!(q.ne[3] == k.ne[3] && q.ne[3] == v.ne[3], "fattn: ne3 broadcast");
        // input tensor rows must be contiguous (ops.cpp asserts nb0 == type size)
        for s in [q, k, v] {
            assert_eq!(s.nb[0], s.ty.type_size() as u64, "fattn: nb0 != type size");
        }
        assert_eq!(q.ty, GgmlType::F32, "fattn: q must be F32");
        assert!(
            matches!(k.ty, GgmlType::F32 | GgmlType::F16),
            "fattn: unsupported K-type {:?}",
            k.ty
        );
        assert!(
            matches!(v.ty, GgmlType::F32 | GgmlType::F16),
            "fattn: unsupported V-type {:?}",
            v.ty
        );
        assert!(q.ne[2] % k.ne[2] == 0 && q.ne[2] % v.ne[2] == 0, "fattn: head broadcast");
        if let Some(m) = &self.mask {
            assert_eq!(m.ty, GgmlType::F16, "fattn: mask must be F16 (ggml.c:5515)");
            assert!(m.is_contiguous(), "fattn: mask must be contiguous");
            assert_eq!(q.ne[2] % m.ne[2], 0, "fattn: q->ne[2] % mask->ne[2] (ggml.c:5517)");
            assert_eq!(q.ne[3] % m.ne[3], 0, "fattn: q->ne[3] % mask->ne[3] (ggml.c:5518)");
            // defensive (C would read out of bounds): the mask must span k/v and q
            assert!(m.ne[0] >= self.n_kv(), "fattn: mask ne0 < n_kv");
            assert!(m.ne[1] >= q.ne[1], "fattn: mask ne1 < n_tokens");
        }
        if self.max_bias > 0.0 {
            assert!(self.mask.is_some(), "fattn: max_bias > 0 requires mask (ggml.c:5520)");
        }
    }

    pub fn dk(&self) -> i64 {
        self.k.ne[0]
    }
    pub fn dv(&self) -> i64 {
        self.v.ne[0]
    }
    /// Number of query tokens (neq1).
    pub fn n_tokens(&self) -> i64 {
        self.q.ne[1]
    }
    /// Number of query heads (neq2).
    pub fn n_head(&self) -> i64 {
        self.q.ne[2]
    }
    /// Number of KV positions (nek1).
    pub fn n_kv(&self) -> i64 {
        self.k.ne[1]
    }
    /// Total rows = neq1*neq2*neq3 (C: `nr`).
    pub fn n_rows(&self) -> i64 {
        self.q.ne[1] * self.q.ne[2] * self.q.ne[3]
    }
    /// Destination shape `{ v->ne[0], q->ne[2], q->ne[1], q->ne[3] }` (ggml.c:5526).
    pub fn dst_ne(&self) -> [i64; 4] {
        [self.v.ne[0], self.q.ne[2], self.q.ne[1], self.q.ne[3]]
    }
    pub fn dst_n_elements(&self) -> i64 {
        self.dst_ne().iter().product()
    }

    /// ops.cpp:8681: `1u << (uint32_t) floor(log2(n_head))`.
    fn n_head_log2(&self) -> u32 {
        1u32 << (self.n_head() as f64).log2().floor() as u32
    }
    /// ops.cpp:8685-8697 — ALiBi slope for query head `h` (1.0 when max_bias <= 0).
    fn slope(&self, h: u32) -> f32 {
        if self.max_bias <= 0.0 {
            return 1.0;
        }
        let nl2 = self.n_head_log2();
        let m0 = 2.0f32.powf(-self.max_bias / nl2 as f32);
        let m1 = 2.0f32.powf(-(self.max_bias / 2.0) / nl2 as f32);
        if h < nl2 {
            m0.powi(h as i32 + 1)
        } else {
            m1.powi((2 * (h - nl2) + 1) as i32)
        }
    }
    /// ops.cpp:8675-8677 — C pre-divides `scale` by `logit_softcap`.
    fn kq_scale(&self) -> f32 {
        if self.logit_softcap != 0.0 {
            self.scale / self.logit_softcap
        } else {
            self.scale
        }
    }
}

/// C `ir -> (iq1, iq2, iq3)` row decomposition (ops.cpp:8745-8747).
#[inline]
fn row_indices(fa: &FlashAttn<'_>, ir: usize) -> (i64, i64, i64) {
    let n1 = fa.n_tokens();
    let n2 = fa.n_head();
    let ir = ir as i64;
    let iq3 = ir / (n2 * n1);
    let iq2 = (ir - iq3 * n2 * n1) / n1;
    let iq1 = ir - iq3 * n2 * n1 - iq2 * n1;
    (iq1, iq2, iq3)
}

/// C: `kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1)` (ops.cpp:8756).
///
/// Scalar generic semantics of ggml-cpu's vec_dot tails: f32 products summed in
/// an f64 `ggml_float`, narrowed to f32 by the caller (`*s = sumf`).
/// C `kq_vec_dot` (ops.cpp:8754): `type_traits_cpu[K].vec_dot` with Q converted
/// by `q_to_vec_dot` (ops.cpp:8602):
///   * K F32: ggml_vec_dot_f32, Q stays F32
///   * K F16: from_float converts Q to F16, then ggml_vec_dot_f16
/// Both kernels are the already-bit-exact vec_dot ports (f32/f16 lane order),
/// so the score matches the reference kernel-for-kernel. Strided rows are
/// gathered into contiguous scratch first.
fn f16_bits_at(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

fn dot_kq(fa: &FlashAttn<'_>, k_row_off: usize, q_row_off: usize) -> f32 {
    let dk = fa.dk() as usize;
    let knb0 = fa.k.nb[0] as usize;
    let qnb0 = fa.q.nb[0] as usize;
    match fa.k.ty {
        GgmlType::F32 => {
            let mut kbuf = vec![0f32; dk];
            let mut qbuf = vec![0f32; dk];
            for d in 0..dk {
                kbuf[d] = f32_at(fa.k.data, k_row_off + d * knb0);
                qbuf[d] = f32_at(fa.q.data, q_row_off + d * qnb0);
            }
            crate::vec_dot::vec_dot_f32_c(dk, &kbuf, &qbuf)
        }
        GgmlType::F16 => {
            let mut kbuf = vec![half::f16::ZERO; dk];
            let mut qbuf = vec![half::f16::ZERO; dk];
            for d in 0..dk {
                kbuf[d] = half::f16::from_bits(f16_bits_at(fa.k.data, k_row_off + d * knb0));
                qbuf[d] = half::f16::from_f32(f32_at(fa.q.data, q_row_off + d * qnb0));
            }
            crate::vec_dot::vec_dot_f16_c(dk, &kbuf, &qbuf)
        }
        other => panic!("fattn: unsupported K-type {other:?}"),
    }
}

/// `kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1)` (ops.cpp:8757) with the Q row
/// already converted (`q_to_vec_dot` runs once per query row, ops.cpp:8746 —
/// the C does not re-convert Q for every KV token) and a *contiguous* K row
/// (`nbk0 == ggml_type_size(k->type)` is asserted for the whole kernel, so the
/// f16/f32 rows can be handed to the vec_dot kernels as slices, no gather).
/// The AVX512 `ggml_vec_dot_f16` (simd_x86) is the actual kernel the reference
/// runs on this host; `vec_dot_f16_c` is its bit-exact scalar emulation.
#[inline]
fn dot_kq_row(fa: &FlashAttn<'_>, k_row_off: usize, q_f16: &[half::f16], q_f32: &[f32]) -> f32 {
    let dk = fa.dk() as usize;
    match fa.k.ty {
        GgmlType::F32 => {
            let krow: &[f32] = bytemuck::cast_slice(&fa.k.data[k_row_off..k_row_off + dk * 4]);
            crate::vec_dot::vec_dot_f32_c(dk, krow, q_f32)
        }
        GgmlType::F16 => {
            let krow: &[half::f16] =
                bytemuck::cast_slice(&fa.k.data[k_row_off..k_row_off + dk * 2]);
            #[cfg(target_arch = "x86_64")]
            if crate::simd_x86::avx512() {
                // SAFETY: `krow`/`q_f16` are `dk`-element slices; the kernel
                // reads `dk` f16 values from each.
                return unsafe {
                    crate::simd_x86::vec_dot_f16(dk, krow.as_ptr() as *const u16, q_f16.as_ptr() as *const u16)
                };
            }
            crate::vec_dot::vec_dot_f16_c(dk, krow, q_f16)
        }
        other => panic!("fattn: unsupported K-type {other:?}"),
    }
}

// ======================================================================
// core row kernel — one (iq1, iq2, iq3) row over KV range [ic_start, ic_end)
// ======================================================================

/// The body of C's `one_chunk` inner loop for a single row: online softmax over
/// the KV range, writing the *unnormalized* accumulator into `vkq` (len DV) and
/// returning `(M, S)`. Callers normalize by `S`.
///
/// `vkq` is zeroed here, mirroring C's `memset(VKQ32/VKQ16, 0, ...)` (ops.cpp:8698-8702).
pub fn attn_row(
    fa: &FlashAttn<'_>,
    iq1: i64,
    iq2: i64,
    iq3: i64,
    ic_start: i64,
    ic_end: i64,
    vkq: &mut [f32],
) -> (f32, f32) {
    let dv = fa.dv() as usize;
    assert!(vkq.len() >= dv, "fattn: vkq scratch too small");

    // broadcast factors (ops.cpp:8651-8657)
    let rk2 = fa.q.ne[2] / fa.k.ne[2];
    let rk3 = fa.q.ne[3] / fa.k.ne[3];
    let rv2 = fa.q.ne[2] / fa.v.ne[2];
    let rv3 = fa.q.ne[3] / fa.v.ne[3];

    let slope = fa.slope(iq2 as u32);

    let mut s_sum = 0.0f32; // C: S
    let mut m = f32::NEG_INFINITY; // C: M
    for x in vkq[..dv].iter_mut() {
        *x = 0.0;
    }

    // row base offsets (ops.cpp:8703-8741)
    let q_row_off = fa.q.row_off(iq1, iq2, iq3);
    let (ik2, ik3, iv2, iv3) = (iq2 / rk2, iq3 / rk3, iq2 / rv2, iq3 / rv3);
    let k_base = (ik2 as u64 * fa.k.nb[2] + ik3 as u64 * fa.k.nb[3]) as usize;
    let v_base = (iv2 as u64 * fa.v.nb[2] + iv3 as u64 * fa.v.nb[3]) as usize;
    // mask row: mask->data + iq1*nb1 + (iq2%ne2)*nb2 + (iq3%ne3)*nb3 (ops.cpp:8749)
    let mask_base = fa.mask.as_ref().map(|m| {
        (iq1 as u64 * m.nb[1]
            + (iq2 % m.ne[2]) as u64 * m.nb[2]
            + (iq3 % m.ne[3]) as u64 * m.nb[3]) as usize
    });

    let kq_scale = fa.kq_scale(); // ops.cpp:8675-8677
    let (knb1, vnb1) = (fa.k.nb[1] as usize, fa.v.nb[1] as usize);
    let vnb0 = fa.v.nb[0] as usize;
    let softcap = fa.logit_softcap;
    // F16 V uses the fp16 VKQ accumulator (ops.cpp:8712-8716)
    let v_f16 = fa.v.ty == GgmlType::F16;
    let mut vkq16 = vec![half::f16::ZERO; dv];

    // `q_to_vec_dot(pq, Q_q, DK)` — once per query row (ops.cpp:8746), not per
    // KV token: for an F16 K cache the Q row widens to f16, for F32 it is
    // passed through (vec_dot_type(F32) == F32).
    let dk = fa.dk() as usize;
    let q_row = q_row_off;
    let q_f16: Vec<half::f16> = if fa.k.ty == GgmlType::F16 {
        (0..dk)
            .map(|d| half::f16::from_f32(f32_at(fa.q.data, q_row + d * fa.q.nb[0] as usize)))
            .collect()
    } else {
        Vec::new()
    };
    let q_f32: Vec<f32> = if fa.k.ty == GgmlType::F32 {
        (0..dk)
            .map(|d| f32_at(fa.q.data, q_row + d * fa.q.nb[0] as usize))
            .collect()
    } else {
        Vec::new()
    };
    // contiguous-row fast paths (the C asserts nb0 == type size for q/k/v)
    let k_contig = fa.k.nb[0] as usize == fa.k.ty.type_size();
    let v_contig = vnb0 == fa.v.ty.type_size();
    #[cfg(target_arch = "x86_64")]
    let v_f16_simd =
        v_f16 && v_contig && dv % 16 == 0 && crate::simd_x86::avx512();
    #[cfg(not(target_arch = "x86_64"))]
    let v_f16_simd = false;

    for ic in ic_start..ic_end {
        // ops.cpp:8749-8752
        let mv = match (&fa.mask, mask_base) {
            (Some(mm), Some(mb)) => slope * f16_at(mm.data, mb + ic as usize * mm.nb[0] as usize),
            _ => 0.0,
        };
        if mv == f32::NEG_INFINITY {
            continue; // masked entry: skipped entirely (keeps fully-masked rows at S = 0)
        }

        // ops.cpp:8754-8763: dot → scale → softcap → mask
        let k_off = k_base + ic as usize * knb1;
        let mut s = if k_contig {
            dot_kq_row(fa, k_off, &q_f16, &q_f32) * kq_scale
        } else {
            dot_kq(fa, k_off, q_row) * kq_scale
        };
        if softcap != 0.0 {
            s = softcap * s.tanh(); // C: logit_softcap*tanhf(s)
        }
        s += mv;

        // ops.cpp:8765-8800: online softmax step + V accumulation
        let mold = m;
        let mut ms = 1.0f32;
        let mut vs = 1.0f32;
        let v_row_off = v_base + ic as usize * vnb1;
        if v_f16 {
            // F16 V: VKQ16 accumulator (ops.cpp:8773-8795). x86 GGML_F16_VEC
            // resolves to the F32Cx16 fallback: f32 FMA + f16 store rounding —
            // ggml_vec_scale_f16 = f16(f32(y)*ms), ggml_vec_mad_f16 =
            // f16(fma(v, vs, y)). The AVX512 16-lane kernels are that literal
            // body (simd_x86::{vec_scale_f16,vec_mad_f16}); the element loop
            // below is the bit-identical fallback.
            if s > m {
                m = s;
                ms = (mold - m).exp();
                #[cfg(target_arch = "x86_64")]
                if v_f16_simd {
                    // SAFETY: `vkq16` holds `dv` f16 values and ms is finite.
                    unsafe {
                        crate::simd_x86::vec_scale_f16(
                            vkq16.as_mut_ptr() as *mut u16,
                            ms,
                            dv,
                        )
                    }
                } else {
                    for x in vkq16.iter_mut() {
                        *x = half::f16::from_f32(x.to_f32() * ms);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for x in vkq16.iter_mut() {
                        *x = half::f16::from_f32(x.to_f32() * ms);
                    }
                }
            } else {
                vs = (s - m).exp();
            }
            #[cfg(target_arch = "x86_64")]
            if v_f16_simd {
                // SAFETY: `vkq16` holds `dv` f16 values; the V row is
                // contiguous (`v_contig`) with `dv` f16 values at v_row_off.
                unsafe {
                    crate::simd_x86::vec_mad_f16(
                        vkq16.as_mut_ptr() as *mut u16,
                        fa.v.data.as_ptr().add(v_row_off) as *const u16,
                        vs,
                        dv,
                    )
                }
            } else {
                for (d, x) in vkq16.iter_mut().enumerate() {
                    let vv = f16_at(fa.v.data, v_row_off + d * vnb0);
                    *x = half::f16::from_f32(vv.mul_add(vs, x.to_f32()));
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for (d, x) in vkq16.iter_mut().enumerate() {
                    let vv = f16_at(fa.v.data, v_row_off + d * vnb0);
                    *x = half::f16::from_f32(vv.mul_add(vs, x.to_f32()));
                }
            }
        } else {
            if s > m {
                m = s;
                ms = (mold - m).exp(); // expf(-inf) == 0 on the first finite score
                for x in vkq[..dv].iter_mut() {
                    *x *= ms;
                }
            } else {
                vs = (s - m).exp();
            }
            for (d, x) in vkq[..dv].iter_mut().enumerate() {
                // ggml_vec_mad_f32: GGML_F32_VEC_FMA = _mm512_fmadd_ps — a
                // fused multiply-add per lane (the scalar tail contracts too);
                // the AVX512 binary's F32-V mad is vfmadd213ps (.so c27d0).
                *x = fa.v.row_elem(v_row_off, d).mul_add(vs, *x);
            }
        }
        // C `S = S*ms + vs`; gcc -ffp-contract=fast turns it into one FMA in the
        // reference binary (dump mode 13 is bit-exact only with the FMA).
        s_sum = s_sum.mul_add(ms, vs);
    }
    if v_f16 {
        // ops.cpp:8805-8807: VKQ32[d] = FP16_TO_FP32(VKQ16[d]) before sinks /
        // normalization (which then run in f32).
        for (d, x) in vkq[..dv].iter_mut().enumerate() {
            *x = vkq16[d].to_f32();
        }
    }

    // sinks (ops.cpp:8811-8825): apply only on the first kv chunk; per head h
    // with `s = sinks[h]`:
    //   s > M:  ms = expf(M - s); M = s; VKQ *= ms
    //   else :  vs = expf(s - M)
    //   S = S*ms + vs   (ms = 1 when the else branch runs)
    if let Some(sk) = fa.sinks {
        if ic_start == 0 {
            let s_h = sk[iq2 as usize];
            let (mut ms, mut vs) = (1.0f32, 1.0f32);
            if s_h > m {
                ms = (m - s_h).exp(); // libm expf, like the C tail
                m = s_h;
                for x in vkq[..dv].iter_mut() {
                    *x *= ms;
                }
            } else {
                vs = (s_h - m).exp();
            }
            // C `S = S*ms + vs` — the reference binary does NOT contract this
            // one: the inlined `ggml_vec_scale_f32` call boundary keeps the
            // `S*ms` as a separate vmulss (ops.cpp:9185 → .so c1ba8) and the
            // `+ vs` as a vaddss (.so c1cd7), two roundings. The inner-loop
            // S line above IS contracted (vfmadd132ss, .so c1a8b); this sinks
            // line is the one place they differ — fma vs mul+add disagree on
            // rows where the sink wins the max (`s > M`), by 1 ulp of S
            // (pinned bit-for-bit by parity/fa_probe.bin's T=1/T=5 records).
            s_sum = s_sum * ms + vs;
        }
    }

    (m, s_sum)
}

/// C ops.cpp:8805-8820: `S_inv = S == 0 ? 0 : 1/S; V *= S_inv;` then store.
fn write_normalized(
    fa: &FlashAttn<'_>,
    dst: &mut Dst<'_>,
    iq1: i64,
    iq2: i64,
    iq3: i64,
    vkq: &mut [f32],
    s: f32,
) {
    let dv = fa.dv() as usize;
    let s_inv = if s == 0.0 { 0.0 } else { 1.0 / s };
    for x in vkq[..dv].iter_mut() {
        *x *= s_inv;
    }
    dst.write_row(iq1, iq2, iq3, &vkq[..dv]);
}

// ======================================================================
// public kernels
// ======================================================================

/// C: `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (ops.cpp:8614) without
/// partials — rows `ir0..ir1` over the whole KV range, normalized into `dst`.
pub fn one_chunk(fa: &FlashAttn<'_>, ir0: usize, ir1: usize, dst: &mut Dst<'_>) {
    let dv = fa.dv() as usize;
    let mut vkq = vec![0f32; dv];
    for ir in ir0..ir1.min(fa.n_rows() as usize) {
        let (iq1, iq2, iq3) = row_indices(fa, ir);
        let (_m, s) = attn_row(fa, iq1, iq2, iq3, 0, fa.n_kv(), &mut vkq);
        write_normalized(fa, dst, iq1, iq2, iq3, &mut vkq, s);
    }
}

/// C: `one_chunk` with `write_partials != NULL` (split-KV path, ops.cpp:8803-8810):
/// rows `ir0..ir1` over the KV range `[ic_start, ic_end)`, storing
/// `[M, S, VKQ[DV]]` per row at `partials[ir*partial_stride ..]`.
///
/// Out-of-range chunks (`ic_start >= nek1`, C ops.cpp:9268-9273) write
/// `M = -inf, S = 0` and no VKQ, exactly like the C `else` branch.
pub fn one_chunk_partials(
    fa: &FlashAttn<'_>,
    ir0: usize,
    ir1: usize,
    ic_start: i64,
    ic_end: i64,
    partials: &mut [f32],
    partial_stride: usize,
) {
    let dv = fa.dv() as usize;
    if ic_start >= fa.n_kv() {
        for ir in ir0..ir1.min(fa.n_rows() as usize) {
            let p = &mut partials[ir * partial_stride..ir * partial_stride + 2];
            p[0] = f32::NEG_INFINITY;
            p[1] = 0.0;
        }
        return;
    }
    let mut vkq = vec![0f32; dv];
    for ir in ir0..ir1.min(fa.n_rows() as usize) {
        let (iq1, iq2, iq3) = row_indices(fa, ir);
        let (m, s) = attn_row(fa, iq1, iq2, iq3, ic_start, ic_end, &mut vkq);
        let p = &mut partials[ir * partial_stride..ir * partial_stride + 2 + dv];
        p[0] = m;
        p[1] = s;
        p[2..].copy_from_slice(&vkq);
    }
}

/// C: `ggml_flash_attn_ext_reduce_partials` (ops.cpp:9142).
///
/// `partials` layout: `[(q_head*n_chunks + chunk_idx) * (2+DV) + {M, S, VKQ[DV]}]`
/// (`partial_size = 2 + DV`, matching C's `partial_stride = nth*partial_size`
/// indexing). Only valid for the split-KV shapes the C dispatcher selects
/// (`neq1 == 1 && neq3 == 1`); writes row `(iq1=0, iq2=q_head, iq3=0)`.
pub fn reduce_partials(
    fa: &FlashAttn<'_>,
    partials: &[f32],
    n_chunks: usize,
    chunk_size: i64,
    dst: &mut Dst<'_>,
) {
    let dv = fa.dv() as usize;
    let partial_size = 2 + dv;
    let n_kv = fa.n_kv();
    assert_eq!(fa.n_tokens(), 1, "fattn reduce: neq1 == 1 (C split-KV eligibility)");
    assert_eq!(fa.q.ne[3], 1, "fattn reduce: neq3 == 1");

    let mut vkq_final = vec![0f32; dv]; // C: thread_wdata VKQ_final (zeroed)
    // C: for (q_head = ith; q_head < n_q_heads; q_head += nth)
    for q_head in 0..fa.n_head() as usize {
        let mut m_final = f32::NEG_INFINITY;
        let mut s_final = 0.0f32;
        for x in vkq_final.iter_mut() {
            *x = 0.0;
        }
        for chunk_idx in 0..n_chunks {
            if chunk_idx as i64 * chunk_size >= n_kv {
                continue; // ops.cpp:9167-9168
            }
            let p = &partials[(q_head * n_chunks + chunk_idx) * partial_size..][..partial_size];
            let (m_chunk, s_chunk) = (p[0], p[1]);
            if s_chunk == 0.0 {
                continue; // ops.cpp:9175
            }
            let m_new = m_final.max(m_chunk);
            let scale_old = (m_final - m_new).exp();
            let scale_new = (m_chunk - m_new).exp();
            // ops.cpp:9178-9180 — the reference contracts the merge into
            // `fma(old, scale_old, chunk*scale_new)` (scalar tail .so c3458,
            // AVX512 lanes c648d; the `chunk*scale_new` product rounds once,
            // then one fused multiply-add), so the port must fuse the same way.
            for (d, x) in vkq_final.iter_mut().enumerate() {
                *x = x.mul_add(scale_old, p[2 + d] * scale_new);
            }
            // ops.cpp:9181 — `S_final*scale_old + S_chunk*scale_new` likewise
            // contracted (.so c65e2: vmulss S_chunk*scale_new + vfmadd132ss).
            s_final = s_final.mul_add(scale_old, s_chunk * scale_new);
            m_final = m_new;
        }
        if s_final != 0.0 {
            let s_inv = 1.0 / s_final; // ops.cpp:9193-9196
            for x in vkq_final.iter_mut() {
                *x *= s_inv;
            }
        }
        dst.write_row(0, q_head as i64, 0, &vkq_final);
    }
}

/// C: the `use_split_kv_path` branch of `ggml_compute_forward_flash_attn_ext_f16`
/// (ops.cpp:9245-9291): split `[0, nek1)` into `n_chunks` chunks of
/// `chunk_size = ceil(nek1/n_chunks)`, compute per-chunk partials per head, then
/// merge with `reduce_partials`. C additionally gates this on `nek1 >= 512`
/// (and `k.type == v.type`, F16/F32 KV); the shape asserts mirror the rest.
///
/// Single-threaded equivalent of the C thread-pool + barrier run.
pub fn split_kv(fa: &FlashAttn<'_>, n_chunks: usize, dst: &mut Dst<'_>) {
    assert!(n_chunks >= 1);
    assert_eq!(fa.n_tokens(), 1, "fattn split-kv: neq1 == 1");
    assert_eq!(fa.q.ne[3], 1, "fattn split-kv: neq3 == 1");
    assert_eq!(fa.k.ty, fa.v.ty, "fattn split-kv: k.type == v.type");
    let dv = fa.dv() as usize;
    let partial_size = 2 + dv;
    let n_head = fa.n_head() as usize;
    let chunk_size = (fa.n_kv() + n_chunks as i64 - 1) / n_chunks as i64;
    let stride = n_chunks * partial_size;
    let mut partials = vec![0f32; stride * n_head];
    for c in 0..n_chunks {
        let ic_start = c as i64 * chunk_size;
        let ic_end = (ic_start + chunk_size).min(fa.n_kv());
        // C: per-thread view = partials_base + c*partial_size, stride nth*partial_size
        one_chunk_partials(fa, 0, n_head, ic_start, ic_end, &mut partials[c * partial_size..], stride);
    }
    reduce_partials(fa, &partials, n_chunks, chunk_size, dst);
}

// ======================================================================
// tiled kernel — C `ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8852)
// ======================================================================

/// `ggml_fa_tile_config::Q` / `::KV` (ggml-cpu/common.h:9-10, 90-92).
pub const Q_TILE: usize = 64;
pub const KV_TILE: usize = 64;
/// `GGML_F32_EPR` of the reference build: `-march=native` on AVX512 → the
/// `__AVX512F__` branch of simd-mappings.h:453 (`GGML_F32_EPR 16`). Sync
/// batch D: the upstream `DV % f32_epr == 0` constraint now applies only on
/// non-x86-64 (`#if defined(GGML_SIMD) && !defined(__x86_64__) &&
/// !defined(_M_X64)`, ops.cpp:9552) — on x86-64 the tiled path takes any DV,
/// which is what the masked `simd_gemm_ukernel_tail` (simd-gemm.h:59) handles.
const F32_EPR: i64 = 16;

/// ops.cpp:9546-9560 — the reference's `use_tiled` predicate
/// (`!use_ref && q F32 && kv F32/F16 && k.type == v.type && neq1 >= 64`);
/// the x86 build drops the `DV % f32_epr == 0` term (see F32_EPR's note), so
/// non-16-multiple DVs now take the tiled path with masked N-tails.
pub fn use_tiled(fa: &FlashAttn<'_>) -> bool {
    let kv_ok = matches!(fa.k.ty, GgmlType::F32 | GgmlType::F16);
    fa.q.ty == GgmlType::F32
        && kv_ok
        && fa.k.ty == fa.v.ty
        && fa.n_tokens() >= Q_TILE as i64
        && (fa.dv() % F32_EPR == 0 || cfg!(target_arch = "x86_64"))
}

/// ops.cpp:9245-9340 dispatch order: `use_split_kv_path` first (decode,
/// `nek1 >= 512`), then `use_tiled`, then `one_chunk`.
///
/// **split-KV is shape-active in the reference and is dispatched here**: for
/// `neq1 == 1 && neq3 == 1 && nek1 >= 512` (a plain decode with a long KV) the
/// reference splits the KV range into `n_chunks` chunks (`chunk_size =
/// ceil(nek1/nth)`), folds each chunk's `[M, S, VKQ]` partials and merges with
/// `reduce_partials` (ops.cpp:9245-9291). The chunk count is the reference
/// run's thread count, and the merged rounding genuinely differs from the
/// whole-range `one_chunk` scan at `n_chunks >= 2` — measured on the reference
/// itself (`parity/ref_fa_probe.c`): a T=4 S_kv=600 batch (one_chunk) vs its
/// T=1 slices (split-KV, 8 chunks) differ in **every** output element, while
/// both are invariant under their own shape. Bit-faithfulness therefore means
/// reproducing each shape's path, not invariance: pass the caller's thread
/// count as `n_chunks` (`nth == 1` degenerates to a single chunk, which is
/// bit-identical to `one_chunk`).
///
/// Rows `ir0..ir1` (the `one_chunk`/`tiled` fallbacks) are computed into `dst`.
pub fn flash_attn_ext_dispatch(fa: &FlashAttn<'_>, ir0: usize, ir1: usize, dst: &mut Dst<'_>) {
    if use_tiled(fa) {
        tiled(fa, ir0, ir1, dst);
    } else {
        one_chunk(fa, ir0, ir1, dst);
    }
}

/// The full C dispatch of `ggml_compute_forward_flash_attn_ext_f16`
/// (ops.cpp:9245-9340) for one whole tensor, including the split-KV branch
/// with `n_chunks` (= the reference run's thread count) KV chunks. This is
/// what a decode step must call; `flash_attn_ext_dispatch` remains for row
/// ranges of the non-split paths.
pub fn flash_attn_ext_dispatch_n(fa: &FlashAttn<'_>, n_chunks: usize, dst: &mut Dst<'_>) {
    if use_split_kv(fa) {
        split_kv(fa, n_chunks, dst);
    } else {
        flash_attn_ext_dispatch(fa, 0, fa.n_rows() as usize, dst);
    }
}

/// ops.cpp:9261: `!use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16
/// && k->type == v->type && q->type == F32 && nek1 >= 512` (`use_ref` is the
/// debug hook, not modelled).
pub fn use_split_kv(fa: &FlashAttn<'_>) -> bool {
    let kv_ok = matches!(fa.k.ty, GgmlType::F32 | GgmlType::F16);
    fa.n_tokens() == 1
        && fa.q.ne[3] == 1
        && kv_ok
        && fa.k.ty == fa.v.ty
        && fa.q.ty == GgmlType::F32
        && fa.n_kv() >= 512
}

/// `ggml_v_expf` AVX512F+DQ branch (vec.h:1172-1200), replicated elementwise
/// with single-rounding FMA (`_mm512_*_ps` are all elementwise, so one scalar
/// lane is a faithful stand-in; `_mm512_scalef_ps(j, n)` is a single-rounded
/// `j * 2^n`, which the f64 evaluation here reproduces exactly — `j` needs 24
/// bits and `2^n` is a power of two, so the f64 product is exact and the final
/// f32 cast does the one rounding, subnormals included).
#[inline]
fn v_expf(x: f32) -> f32 {
    // C hex-float literals -> exact f32 bit patterns (Rust has no 0x1.pN form)
    const R: f32 = f32::from_bits(0x4b40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const C1: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C2: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const P1: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    const P2: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const P3: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const P4: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const P5: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    let z = x.mul_add(LOG2E, R);
    let n = z - R;
    // b = fnmadd(n, C1, fnmadd(n, C2, x))  [fnmadd(a,b,c) = -(a*b) + c]
    let b = n.mul_add(-C1, n.mul_add(-C2, x));
    let u = b * b;
    // j = fma(fma(fma(P1,b,P2), u, fma(P3,b,P4)), u, fma(P5,b,1.0))
    let j = b.mul_add(P1, P2)
        .mul_add(u, b.mul_add(P3, P4))
        .mul_add(u, b.mul_add(P5, 1.0));
    // __mmask16 d = |n| > 192  (ordered compare: NaN -> false)
    if n.abs() > 192.0 {
        // alt = (n <= 0) ? 0 : +inf
        return if n <= 0.0 { 0.0 } else { f32::INFINITY };
    }
    // res = scalef(j, n) = j * 2^n, single-rounded
    ((j as f64) * exp2_scalar(n)) as f32
}

/// `2^n` for the integer `n` produced by the `z - R` split (exact in f64).
#[inline]
fn exp2_scalar(n: f32) -> f64 {
    let e = n as i32; // exact: n is already an integer (R = 1.5 * 2^23)
    if e > 1023 {
        f64::INFINITY
    } else if e < -1074 {
        0.0
    } else if (-1022..=1023).contains(&e) {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        // subnormal 2^e
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// `_mm512_reduce_add_ps` as emitted by GCC (avx512fintrin.h `__MM512_REDUCE_OP`):
/// `T3[j] = A[j+8]+A[j]`, `T6[j] = T3[j+4]+T3[j]`, then the 4-lane fold
/// `(T6[0]+T6[2]) + (T6[1]+T6[3])`.
#[inline]
fn reduce_add_ps(v: &[f32; 16]) -> f32 {
    let mut t3 = [0f32; 8];
    for j in 0..8 {
        t3[j] = v[8 + j] + v[j];
    }
    let t6 = [t3[4] + t3[0], t3[5] + t3[1], t3[6] + t3[2], t3[7] + t3[3]];
    (t6[0] + t6[2]) + (t6[1] + t6[3])
}

/// `ggml_vec_max_f32` (vec.h:1541): `max = MAX(max, x[i])` from `-INFINITY`.
#[inline]
fn vec_max_f32(x: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        max = if max > v { max } else { v }; // MAX(a, b) = a > b ? a : b
    }
    max
}

/// `ggml_vec_soft_max_f32` (vec.cpp:531) AVX512F+DQ branch, in place: writes
/// `v_expf(x[i] - max)` over `y` and returns the f64 (`ggml_float`) sum of the
/// 16-lane `_mm512_reduce_add_ps` partials, then the scalar `expf` tail.
///
/// With `use_simd` the 16-wide chunks run as the real vector kernel
/// (`simd_x86::vec_soft_max_f32_inplace` — the same per-lane ops as the
/// elementwise body below, so the two are bit-identical; asserted in
/// `simd_x86`'s tests and by the tiled scalar-vs-SIMD equality test). The
/// elementwise body is the non-AVX512 fallback.
fn vec_soft_max_f32_inplace(y: &mut [f32], max: f32, use_simd: bool) -> f64 {
    if use_simd && crate::simd_x86::avx512() {
        return crate::simd_x86::vec_soft_max_f32_inplace(y, max);
    }
    let n = y.len();
    let mut i = 0;
    let mut sum = 0f64;
    while i + 15 < n {
        let mut v = [0f32; 16];
        for (j, slot) in v.iter_mut().enumerate() {
            *slot = v_expf(y[i + j] - max);
        }
        y[i..i + 16].copy_from_slice(&v);
        sum += reduce_add_ps(&v) as f64;
        i += 16;
    }
    while i < n {
        let val = (y[i] - max).exp(); // expf
        sum += val as f64;
        y[i] = val;
        i += 1;
    }
    sum
}

/// `simd_gemm` (simd-gemm.h:60) — `C[M x N] += A[M x K] * B[K x N]`.
///
/// The reference build's `GGML_SIMD` branch (GEMM_RM=4, GEMM_RN=4, KN=16 on
/// AVX512) runs `simd_gemm_ukernel` register blocks of `_mm512_fmadd_ps`
/// (simd-gemm.h:24-55) plus scalar `jj` tails (simd-gemm.h:77-85, 100-106).
/// Every output element is an independent FMA chain over `kk` ascending —
/// the lane block only assigns elements to lanes, never the arithmetic — so
/// the AVX512 kernel (`simd_x86::simd_gemm_avx512`, a literal port of the C
/// tiles) and the elementwise scalar port below are **bit-identical by
/// construction**; `simd_x86`'s tests assert it and the reference dump
/// (`ref_dump_bitexact`, all 11 tiled cases) pins the result to the reference.
fn simd_gemm(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, use_simd: bool) {
    if use_simd && crate::simd_x86::avx512() {
        crate::simd_x86::simd_gemm_avx512(c, a, b, m, k, n);
    } else {
        simd_gemm_scalar(c, a, b, m, k, n);
    }
}

/// The C scalar `#else` body of `simd_gemm` (simd-gemm.h:134-145) with the
/// same FMA-chain evaluation the reference's tails (and GCC's contraction of
/// them) produce: `acc = fma(a, b, acc)` per element, `kk` ascending.
fn simd_gemm_scalar(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk];
            let (brow, crow) = (&b[kk * n..kk * n + n], &mut c[i * n..i * n + n]);
            for (x, &bv) in crow.iter_mut().zip(brow) {
                *x = av.mul_add(bv, *x);
            }
        }
    }
}

/// C `ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8852-9140): rows
/// `ir0..ir1` over the whole KV range, Q/KV tiles of 64.
///
/// Structure per 64-row Q tile (tiles never cross a head boundary):
///   * `KQ = Q_tile * K_tileᵀ` via `simd_gemm` (F32 accumulation, one FMA chain
///     per element), scaled, padded columns set to `-INFINITY`;
///   * vector softmax per row (`v_expf` + reduce tree) updating the running
///     `M`/`S`, with the tile's max used for every row of the tile;
///   * `VKQ32 += KQ * V_tile` via `simd_gemm` (C's accumulation order: the ms
///     rescale is applied once per tile, not per KV element);
///   * sinks, `S_inv` normalization and store, exactly as one_chunk.
///
/// On AVX512 hosts the per-lane loops run as the reference's 16-lane kernels
/// (`simd_gemm`'s `_mm512_fmadd_ps` tiles, `ggml_vec_scale_f32`,
/// `ggml_vec_add_f32`, `ggml_vec_soft_max_f32`'s `ggml_v_expf` +
/// `_mm512_reduce_add_ps`, `ggml_cpu_fp16_to_fp32`'s `vcvtph2ps`) — all
/// per-lane ops, so the scalar body below (kept verbatim as the fallback) is
/// bit-identical by construction; `tiled_simd_matches_scalar_bit_exact`
/// asserts it on the same inputs.
pub fn tiled(fa: &FlashAttn<'_>, ir0: usize, ir1: usize, dst: &mut Dst<'_>) {
    tiled_impl(fa, ir0, ir1, dst, crate::simd_x86::avx512());
}

/// The tile body; `use_simd == false` forces every vector kernel to its
/// elementwise lane-port (the non-AVX512 fallback — and the reference the
/// scalar-vs-SIMD equality test compares against).
#[allow(clippy::too_many_lines)]
fn tiled_impl(fa: &FlashAttn<'_>, ir0: usize, ir1: usize, dst: &mut Dst<'_>, use_simd: bool) {
    let dk = fa.dk() as usize;
    let dv = fa.dv() as usize;
    let n_tok = fa.n_tokens();
    let rk2 = fa.q.ne[2] / fa.k.ne[2];
    let rk3 = fa.q.ne[3] / fa.k.ne[3];
    let rv2 = fa.q.ne[2] / fa.v.ne[2];
    let rv3 = fa.q.ne[3] / fa.v.ne[3];
    let kq_scale = fa.kq_scale();
    let softcap = fa.logit_softcap;
    let kv_f16 = fa.k.ty == GgmlType::F16;

    // per-thread scratch (C: params->wdata, ops.cpp:8966-8968)
    let mut q_f32 = vec![0f32; Q_TILE * dk];
    let mut kq = vec![0f32; Q_TILE * KV_TILE];
    let mut mask32 = vec![0f32; Q_TILE * KV_TILE];
    let mut vkq32 = vec![0f32; Q_TILE * dv];
    let mut v32 = vec![0f32; KV_TILE * dv];
    let mut k_f32 = vec![0f32; KV_TILE * dk]; // K_f32[dk * KV_TILE + tk]

    let nrows = fa.n_rows() as usize;
    let ir_end = ir1.min(nrows);
    let mut ir = ir0;
    while ir < ir_end {
        let (iq1, iq2, iq3) = row_indices(fa, ir);
        // limit by tile size, chunk boundary and the head boundary (ops.cpp:8938)
        let tile_rows = Q_TILE.min(ir_end - ir).min((n_tok - iq1) as usize);
        assert!(tile_rows > 0, "fattn tiled: empty tile");
        let h = iq2 as u32;
        let slope = fa.slope(h);

        let mut s_sum = [0f32; Q_TILE];
        let mut m_row = [f32::NEG_INFINITY; Q_TILE];
        vkq32.fill(0.0);
        mask32.fill(0.0);

        // Q tile (ops.cpp:8970-8980): copied to contiguous rows, zero-padded
        // (C memcpys DK*sizeof(float) per row — F32 q, nb0 == 4 by validate)
        for tq in 0..tile_rows {
            let q_off = fa.q.row_off(iq1 + tq as i64, iq2, iq3);
            if fa.q.nb[0] == 4 {
                q_f32[tq * dk..(tq + 1) * dk]
                    .copy_from_slice(bytemuck::cast_slice(&fa.q.data[q_off..q_off + dk * 4]));
            } else {
                for d in 0..dk {
                    q_f32[tq * dk + d] = f32_at(fa.q.data, q_off + d * fa.q.nb[0] as usize);
                }
            }
        }
        for tq in tile_rows..Q_TILE {
            for d in 0..dk {
                q_f32[tq * dk + d] = 0.0;
            }
        }

        let ik2 = iq2 / rk2;
        let ik3 = iq3 / rk3;
        let iv2 = iq2 / rv2;
        let iv3 = iq3 / rv3;
        let (knb0, knb1) = (fa.k.nb[0] as usize, fa.k.nb[1] as usize);
        let (vnb0, vnb1) = (fa.v.nb[0] as usize, fa.v.nb[1] as usize);
        let k_base = (ik2 as u64 * fa.k.nb[2] + ik3 as u64 * fa.k.nb[3]) as usize;
        let v_base = (iv2 as u64 * fa.v.nb[2] + iv3 as u64 * fa.v.nb[3]) as usize;

        // (ops.cpp:8987-8988) — K_f32/V32 are zeroed once, so a partial tile
        // keeps the *previous* tile's tail (C relies on the KQ padding fixup).
        k_f32.fill(0.0);
        v32.fill(0.0);

        let mut ic = 0i64;
        while ic < fa.n_kv() {
            let kv_tile = KV_TILE.min((fa.n_kv() - ic) as usize);

            // ops.cpp:8992-9012: mask tile + all-masked-tile skip
            if let Some(mk) = &fa.mask {
                let mut can_skip = true;
                for tq in 0..tile_rows {
                    let mrow = (iq1 + tq as i64) as u64 * mk.nb[1]
                        + (iq2 % mk.ne[2]) as u64 * mk.nb[2]
                        + (iq3 % mk.ne[3]) as u64 * mk.nb[3];
                    for tk in 0..kv_tile {
                        let mv = slope * f16_at(mk.data, (mrow as usize) + (ic as usize + tk) * mk.nb[0] as usize);
                        mask32[tq * KV_TILE + tk] = mv;
                        if mv != f32::NEG_INFINITY {
                            can_skip = false;
                        }
                    }
                    for tk in kv_tile..KV_TILE {
                        mask32[tq * KV_TILE + tk] = f32::NEG_INFINITY;
                    }
                }
                if can_skip {
                    ic += KV_TILE as i64;
                    continue;
                }
            }

            // ops.cpp:9014-9030: pack the K tile transposed (KV contiguous)
            for tk in 0..kv_tile {
                let koff = k_base + (ic as usize + tk) * knb1;
                for d in 0..dk {
                    k_f32[d * KV_TILE + tk] = if kv_f16 {
                        f16_at(fa.k.data, koff + d * knb0)
                    } else {
                        f32_at(fa.k.data, koff + d * knb0)
                    };
                }
            }
            // ops.cpp:9031-9038: KQ = Q_tile * K_tile^T, then scale
            // (ggml_vec_scale_f32 — one rounded multiply per lane)
            kq.fill(0.0);
            simd_gemm(&mut kq, &q_f32, &k_f32, Q_TILE, dk, KV_TILE, use_simd);
            if use_simd {
                crate::simd_x86::vec_scale_f32(&mut kq, kq_scale);
            } else {
                for x in kq.iter_mut() {
                    *x *= kq_scale;
                }
            }
            // ops.cpp:9040-9052 (sync batch D): softcap FIRST (tanh over the
            // full Q_TILE x KV_TILE buffer), then the -INFINITY pad — the
            // upstream order swap means a padded KV tail stays -inf (zero
            // softmax weight) instead of tanh(-inf)*softcap = -softcap
            if softcap != 0.0 {
                for x in kq.iter_mut() {
                    *x = softcap * x.tanh();
                }
            }
            // ops.cpp:9054-9059: pad the KQ tail with -INFINITY
            if kv_tile < KV_TILE {
                for tq in 0..Q_TILE {
                    for tk in kv_tile..KV_TILE {
                        kq[tq * KV_TILE + tk] = f32::NEG_INFINITY;
                    }
                }
            }
            // ops.cpp:9054-9056: mask add — only the valid rows
            // (ggml_vec_add_f32(tile_rows*KV_TILE_SZ, KQ, KQ, mask32))
            if fa.mask.is_some() {
                let n = tile_rows * KV_TILE;
                if use_simd {
                    crate::simd_x86::vec_acc_f32(&mut kq[..n], &mask32[..n]);
                } else {
                    for i in 0..n {
                        kq[i] += mask32[i];
                    }
                }
            }

            // ops.cpp:9058-9083: online softmax over the tile (per row)
            let mut skip = [false; Q_TILE];
            for tq in 0..Q_TILE {
                let row = &mut kq[tq * KV_TILE..(tq + 1) * KV_TILE];
                let tile_max = vec_max_f32(row);
                if tile_max == f32::NEG_INFINITY {
                    skip[tq] = true;
                    continue;
                }
                let m_old = m_row[tq];
                let m_new = if m_old > tile_max { m_old } else { tile_max }; // fmaxf
                if m_new > m_old {
                    let ms = (m_old - m_new).exp(); // expf
                    let vrow = &mut vkq32[tq * dv..(tq + 1) * dv];
                    if use_simd {
                        crate::simd_x86::vec_scale_f32(vrow, ms);
                    } else {
                        for x in vrow.iter_mut() {
                            *x *= ms;
                        }
                    }
                    s_sum[tq] *= ms;
                }
                m_row[tq] = m_new;
                let sum = vec_soft_max_f32_inplace(row, m_new, use_simd);
                s_sum[tq] = ((s_sum[tq] as f64) + sum) as f32; // ff ... += ggml_float
            }

            // ops.cpp:9085-9092: pack V to contiguous F32 (C memcpy for F32 KV,
            // ggml_cpu_fp16_to_fp32 for F16 — vcvtph2ps lanes here)
            for tk in 0..kv_tile {
                let voff = v_base + (ic as usize + tk) * vnb1;
                if kv_f16 {
                    if use_simd && vnb0 == 2 && dv % 16 == 0 {
                        crate::simd_x86::fp16_to_f32_row(&mut v32[tk * dv..(tk + 1) * dv], &fa.v.data[voff..voff + dv * 2]);
                    } else {
                        for d in 0..dv {
                            v32[tk * dv + d] = f16_at(fa.v.data, voff + d * vnb0);
                        }
                    }
                } else if vnb0 == 4 {
                    v32[tk * dv..(tk + 1) * dv]
                        .copy_from_slice(bytemuck::cast_slice(&fa.v.data[voff..voff + dv * 4]));
                } else {
                    for d in 0..dv {
                        v32[tk * dv + d] = f32_at(fa.v.data, voff + d * vnb0);
                    }
                }
            }
            // ops.cpp:9093-9098: a skipped row contributes nothing
            for tq in 0..Q_TILE {
                if skip[tq] {
                    kq[tq * KV_TILE..(tq + 1) * KV_TILE].fill(0.0);
                }
            }
            // ops.cpp:9100: VKQ32 += weights * V_tile
            simd_gemm(&mut vkq32, &kq, &v32, Q_TILE, KV_TILE, dv, use_simd);

            ic += KV_TILE as i64;
        }

        // ops.cpp:9103-9119: sinks, valid rows only
        if let Some(sk) = fa.sinks {
            let sv = sk[h as usize];
            for tq in 0..tile_rows {
                let (mut ms, mut vs) = (1.0f32, 1.0f32);
                if sv > m_row[tq] {
                    ms = (m_row[tq] - sv).exp(); // expf
                    let vrow = &mut vkq32[tq * dv..(tq + 1) * dv];
                    if use_simd {
                        crate::simd_x86::vec_scale_f32(vrow, ms);
                    } else {
                        for x in vrow.iter_mut() {
                            *x *= ms;
                        }
                    }
                } else {
                    vs = (sv - m_row[tq]).exp();
                }
                    // C `S[tq]*ms + vs` contracts to one FMA (gcc -ffp-contract=fast).
            // Verified against the reference dump (mode 5, sinks + tiled).
            s_sum[tq] = s_sum[tq].mul_add(ms, vs);
            }
        }

        // ops.cpp:9121-9137: V /= S and store
        for tq in 0..tile_rows {
            let s_inv = if s_sum[tq] == 0.0 { 0.0 } else { 1.0 / s_sum[tq] };
            let vrow = &mut vkq32[tq * dv..(tq + 1) * dv];
            if use_simd {
                crate::simd_x86::vec_scale_f32(vrow, s_inv); // ggml_vec_scale_f32
            } else {
                for x in vrow.iter_mut() {
                    *x *= s_inv;
                }
            }
            dst.write_row(iq1 + tq as i64, iq2, iq3, vrow);
        }

        ir += tile_rows;
    }
}

/// Convenience reference entry point: single thread, whole KV range, dense F32
/// output, C's full path dispatch (split-KV with `n_chunks == 1` — bit-identical
/// to `one_chunk` since the single-chunk merge is exact; the shape-active
/// multi-chunk case is [`flash_attn_ext_dispatch_n`]). Returns the destination
/// buffer in C's permuted layout `[DV, H, T, S]` (offset of `(d, h, t, s)` =
/// `s*T*H*DV + t*H*DV + h*DV + d`).
pub fn flash_attn_ext_ref(fa: &FlashAttn<'_>) -> Vec<f32> {
    fa.validate();
    let ne = fa.dst_ne();
    let mut out = vec![0f32; fa.dst_n_elements() as usize];
    {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
        let mut dst = Dst::dense_f32(bytes, ne);
        flash_attn_ext_dispatch_n(fa, 1, &mut dst);
    }
    out
}

// ======================================================================
// tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (no external crates); samples in [-0.5, 0.5).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed)
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }
    }

    fn nb_dense(ty: GgmlType, ne0: usize, ne1: usize, ne2: usize) -> [u64; 4] {
        let nb0 = ty.type_size() as u64;
        let nb1 = nb0 * ne0 as u64;
        let nb2 = nb1 * ne1 as u64;
        let nb3 = nb2 * ne2 as u64;
        [nb0, nb1, nb2, nb3]
    }

    fn bytes_of(v: &[f32], ty: GgmlType) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * ty.type_size());
        for &x in v {
            match ty {
                GgmlType::F32 => out.extend_from_slice(&x.to_ne_bytes()),
                GgmlType::F16 => out.extend_from_slice(&f16::from_f32(x).to_bits().to_ne_bytes()),
                other => panic!("test: type {other:?}"),
            }
        }
        out
    }

    #[derive(Clone, Copy, Debug)]
    struct Shape {
        dk: usize,
        dv: usize,
        h: usize,
        h_kv: usize,
        t: usize,
        s_kv: usize,
    }

    struct Case {
        sh: Shape,
        qty: GgmlType,
        kty: GgmlType,
        vty: GgmlType,
        scale: f32,
        max_bias: f32,
        softcap: f32,
        /// per-head attention sinks (gpt-oss); applied on the first kv chunk
        sinks: Option<Vec<f32>>,
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        knb: [u64; 4],
        vnb: [u64; 4],
        /// [s_kv, t, 1, 1] F16 causal mask (f16 bit patterns)
        mask: Option<Vec<u8>>,
    }

    const SMALL: Shape = Shape { dk: 16, dv: 16, h: 4, h_kv: 2, t: 3, s_kv: 5 };

    impl Case {
        fn new(sh: Shape, kty: GgmlType, vty: GgmlType, seed: u64) -> Case {
            let mut rng = Rng::new(seed);
            let f32s = |rng: &mut Rng, n: usize| (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();
            let q_f = f32s(&mut rng, sh.dk * sh.h * sh.t);
            let k_f = f32s(&mut rng, sh.dk * sh.h_kv * sh.s_kv);
            let v_f = f32s(&mut rng, sh.dv * sh.h_kv * sh.s_kv);
            Case {
                sh,
                qty: GgmlType::F32,
                kty,
                vty,
                scale: 1.0 / (sh.dk as f32).sqrt(),
                max_bias: 0.0,
                softcap: 0.0,
                sinks: None,
                q: bytes_of(&q_f, GgmlType::F32),
                k: bytes_of(&k_f, kty),
                v: bytes_of(&v_f, vty),
                knb: nb_dense(kty, sh.dk, sh.s_kv, sh.h_kv),
                vnb: nb_dense(vty, sh.dv, sh.s_kv, sh.h_kv),
                mask: None,
            }
        }

        /// Causal f16 mask with window offset `s_kv - t` (as in a decode step
        /// where the last token sees all `s_kv` positions): row `tt` attends to
        /// `s <= tt + (s_kv - t)`.
        fn causal_mask(sh: Shape) -> Vec<u8> {
            let off = sh.s_kv - sh.t;
            let mut m = Vec::with_capacity(sh.s_kv * sh.t * 2);
            for tt in 0..sh.t {
                for s in 0..sh.s_kv {
                    let bits = if s <= tt + off { 0u16 } else { f16::NEG_INFINITY.to_bits() };
                    m.extend_from_slice(&bits.to_ne_bytes());
                }
            }
            m
        }

        fn uniform_mask(sh: Shape, value: f16) -> Vec<u8> {
            let bits = value.to_bits();
            let mut m = Vec::with_capacity(sh.s_kv * sh.t * 2);
            for _ in 0..sh.s_kv * sh.t {
                m.extend_from_slice(&bits.to_ne_bytes());
            }
            m
        }

        fn with_mask(mut self, m: Vec<u8>) -> Case {
            self.mask = Some(m);
            self
        }

        /// Pad the K/V row stride by `pad` bytes (KV-cache-style strided view,
        /// e.g. `view_3d` into a bigger cache buffer).
        fn with_padded_kv(mut self, pad: usize) -> Case {
            let sh = self.sh;
            let krow = sh.dk * self.kty.type_size();
            let vrow = sh.dv * self.vty.type_size();
            let mut k = Vec::new();
            for hh in 0..sh.h_kv {
                for ic in 0..sh.s_kv {
                    let off = (ic as u64 * self.knb[1] + hh as u64 * self.knb[2]) as usize;
                    k.extend_from_slice(&self.k[off..off + krow]);
                    k.extend(std::iter::repeat(0u8).take(pad));
                }
            }
            let mut v = Vec::new();
            for hh in 0..sh.h_kv {
                for ic in 0..sh.s_kv {
                    let off = (ic as u64 * self.vnb[1] + hh as u64 * self.vnb[2]) as usize;
                    v.extend_from_slice(&self.v[off..off + vrow]);
                    v.extend(std::iter::repeat(0u8).take(pad));
                }
            }
            self.k = k;
            self.v = v;
            // ne = [D, S_kv, H_kv]: nb1 = one (padded) KV row, nb2 = one head plane
            self.knb = [self.kty.type_size() as u64, (krow + pad) as u64, (krow + pad) as u64 * sh.s_kv as u64, 0];
            self.vnb = [self.vty.type_size() as u64, (vrow + pad) as u64, (vrow + pad) as u64 * sh.s_kv as u64, 0];
            self.knb[3] = self.knb[2] * sh.h_kv as u64;
            self.vnb[3] = self.vnb[2] * sh.h_kv as u64;
            self
        }

        /// Materialize every KV head h as `h / (H/H_kv)` and re-run with h_kv == h
        /// (independent check of the rk2/rv2 broadcast indexing).
        fn expand_kv_heads(&self) -> Case {
            let sh = Shape { h_kv: self.sh.h, ..self.sh };
            let krow = sh.dk * self.kty.type_size();
            let vrow = sh.dv * self.vty.type_size();
            let mut k = Vec::new();
            let mut v = Vec::new();
            for hh in 0..sh.h {
                for ic in 0..sh.s_kv {
                    let src_h = hh * self.sh.h_kv / sh.h;
                    let ko = (ic as u64 * self.knb[1] + src_h as u64 * self.knb[2]) as usize;
                    k.extend_from_slice(&self.k[ko..ko + krow]);
                    let vo = (ic as u64 * self.vnb[1] + src_h as u64 * self.vnb[2]) as usize;
                    v.extend_from_slice(&self.v[vo..vo + vrow]);
                }
            }
            Case {
                sh,
                qty: self.qty,
                kty: self.kty,
                vty: self.vty,
                scale: self.scale,
                max_bias: self.max_bias,
                softcap: self.softcap,
                sinks: self.sinks.clone(),
                q: self.q.clone(),
                k,
                v,
                knb: nb_dense(self.kty, sh.dk, sh.s_kv, sh.h),
                vnb: nb_dense(self.vty, sh.dv, sh.s_kv, sh.h),
                mask: self.mask.clone(),
            }
        }

        fn fa(&self) -> FlashAttn<'_> {
            let sh = self.sh;
            // C layout: q [DK, T, H, S], k/v [D, S_kv, H_kv, S], mask [S_kv, T, ...]
            FlashAttn {
                q: Src::strided(&self.q, self.qty, [sh.dk as i64, sh.t as i64, sh.h as i64, 1], nb_dense(GgmlType::F32, sh.dk, sh.t, sh.h)),
                k: Src::strided(&self.k, self.kty, [sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1], self.knb),
                v: Src::strided(&self.v, self.vty, [sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1], self.vnb),
                mask: self.mask.as_ref().map(|m| Src::dense(m, GgmlType::F16, [sh.s_kv as i64, sh.t as i64, 1, 1])),
                scale: self.scale,
                max_bias: self.max_bias,
                logit_softcap: self.softcap,
                sinks: self.sinks.as_deref(),
            }
        }

        fn run(&self) -> Vec<f32> {
            flash_attn_ext_ref(&self.fa())
        }

        fn run_split(&self, n_chunks: usize) -> Vec<f32> {
            let fa = self.fa();
            let ne = fa.dst_ne();
            let mut out = vec![0f32; fa.dst_n_elements() as usize];
            {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                let mut dst = Dst::dense_f32(bytes, ne);
                split_kv(&fa, n_chunks, &mut dst);
            }
            out
        }

        fn qnb(&self) -> [u64; 4] {
            nb_dense(GgmlType::F32, self.sh.dk, self.sh.t, self.sh.h)
        }

        fn dst_row<'a>(&self, out: &'a [f32], iq1: usize, iq2: usize) -> &'a [f32] {
            let sh = self.sh;
            let off = iq1 * sh.h * sh.dv + iq2 * sh.dv;
            &out[off..off + sh.dv]
        }

        /// Element read with the operand's strides.
        fn elem(&self, buf: &[u8], nb: &[u64; 4], ty: GgmlType, i1: usize, i2: usize, i0: usize) -> f32 {
            let off = (i1 as u64 * nb[1] + i2 as u64 * nb[2]) as usize + i0 * nb[0] as usize;
            match ty {
                GgmlType::F32 => f32_at(buf, off),
                GgmlType::F16 => f16_at(buf, off),
                other => panic!("test: type {other:?}"),
            }
        }

        /// Independent naive two-pass softmax reference for row (iq1 token,
        /// iq2 head): scores formed exactly like the kernel, then the textbook
        /// max/exp/sum softmax (structurally different from the online update).

        /// Comparison tolerance for the naive two-pass reference. F16 V uses the
        /// fp16 VKQ accumulator (ops.cpp:8773), so the kernel legitimately
        /// rounds every accumulation step to f16 — ~1e-3 relative.
        fn tol_for(vty: GgmlType) -> f32 {
            // fp16 VKQ accumulator (ops.cpp:8773): every accumulation step
            // rounds to f16, so the naive two-pass comparison legitimately
            // drifts ~1e-2 for F16 V. Kernel-vs-kernel tests stay bit-exact.
            if vty == GgmlType::F16 { 1e-2 } else { 1e-4 }
        }

        fn naive_row(&self, iq1: usize, iq2: usize) -> Vec<f32> {
            let sh = self.sh;
            let hk = iq2 / (sh.h / sh.h_kv);
            // ALiBi slope, recomputed from the C formula (ops.cpp:8685-8697)
            let (nl2, hh) = (1u32 << (sh.h as f64).log2().floor() as u32, iq2 as u32);
            let slope = if self.max_bias > 0.0 {
                let m0 = 2.0f32.powf(-self.max_bias / nl2 as f32);
                let m1 = 2.0f32.powf(-(self.max_bias / 2.0) / nl2 as f32);
                if hh < nl2 {
                    m0.powi(hh as i32 + 1)
                } else {
                    m1.powi((2 * (hh - nl2) + 1) as i32)
                }
            } else {
                1.0
            };

            let mut scores: Vec<(usize, f32)> = Vec::new();
            for ic in 0..sh.s_kv {
                let mv = match &self.mask {
                    Some(m) => slope * f16_at(m, (iq1 * sh.s_kv + ic) * 2),
                    None => 0.0,
                };
                if mv == f32::NEG_INFINITY {
                    continue; // C skips -inf mask entries before touching the accumulators
                }
                let mut acc = 0f64;
                for d in 0..sh.dk {
                    let kx = self.elem(&self.k, &self.knb, self.kty, ic, hk, d);
                    let qx = self.elem(&self.q, &self.qnb(), GgmlType::F32, iq1, iq2, d);
                    acc += (kx * qx) as f64;
                }
                let mut s = acc as f32 * if self.softcap != 0.0 { self.scale / self.softcap } else { self.scale };
                if self.softcap != 0.0 {
                    s = self.softcap * s.tanh();
                }
                s += mv;
                scores.push((ic, s));
            }
            if scores.is_empty() {
                return vec![0f32; sh.dv]; // S == 0 → zeros (C one_chunk)
            }
            // sinks = a virtual KV entry that only contributes to the denom:
            //   max includes sk; sum += exp(sk - max); no value term (ops.cpp:8811)
            let sk = self.sinks.as_ref().map(|v| v[iq2]);
            let mut mx = scores.iter().map(|&(_, s)| s).fold(f32::NEG_INFINITY, f32::max);
            if let Some(sk) = sk {
                mx = mx.max(sk);
            }
            let mut ex: Vec<f32> = scores.iter().map(|&(_, s)| (s - mx).exp()).collect();
            let mut sum: f32 = ex.iter().sum();
            if let Some(sk) = sk {
                sum += (sk - mx).exp();
            }
            let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            let mut out = vec![0f32; sh.dv];
            let _ = &mut ex;
            for (j, &(ic, _)) in scores.iter().enumerate() {
                let w = ex[j] * inv;
                for d in 0..sh.dv {
                    let vv = self.elem(&self.v, &self.vnb, self.vty, ic, hk, d);
                    out[d] += w * vv;
                }
            }
            out
        }

        fn naive_all(&self) -> Vec<f32> {
            let sh = self.sh;
            let mut out = vec![0f32; sh.dv * sh.h * sh.t];
            for iq1 in 0..sh.t {
                for iq2 in 0..sh.h {
                    let row = self.naive_row(iq1, iq2);
                    let off = iq1 * sh.h * sh.dv + iq2 * sh.dv;
                    out[off..off + sh.dv].copy_from_slice(&row);
                }
            }
            out
        }
    }

    /// Scaled relative error: `max |x-y| / max(|y|, 0.1*max|y|)`.
    /// The floor guards against near-zero outputs produced by cancellation
    /// between the weighted V terms (where a plain relative error is meaningless).
    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let ymax = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
        let floor = 0.1 * ymax;
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs() / y.abs().max(floor))
            .fold(0f32, f32::max)
    }

    // ---- 1. structure: online kernel vs naive two-pass softmax -------------

    #[test]
    fn matches_naive_two_pass_f32_kv() {
        let c = Case::new(SMALL, GgmlType::F32, GgmlType::F32, 1).with_mask(Case::causal_mask(SMALL));
        let (got, want) = (c.run(), c.naive_all());
        let e = rel_err(&got, &want);
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
        assert!(e < 1e-5, "expected near-exact agreement for f32 K/V, got {e}");
    }

    #[test]
    fn matches_naive_two_pass_f16_kv() {
        let c = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 2).with_mask(Case::causal_mask(SMALL));
        let e = rel_err(&c.run(), &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    #[test]
    fn matches_naive_two_pass_mixed_v_f32_k_f16() {
        let c = Case::new(SMALL, GgmlType::F16, GgmlType::F32, 3).with_mask(Case::causal_mask(SMALL));
        let e = rel_err(&c.run(), &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    // ---- 2. mask semantics -------------------------------------------------

    /// Fully masked row: every `mv == -INFINITY` → skipped → `S == 0` →
    /// `S_inv = 0` → output is exactly zero (C ops.cpp:8805; NOT NaN).
    /// attention sinks act as a virtual KV entry that only contributes to the
    /// softmax denominator (gpt-oss). Compared against the extended naive_row.
    #[test]
    fn sinks_match_naive_virtual_entry() {
        let sh = Shape { dk: 16, dv: 16, t: 3, s_kv: 5, h: 4, h_kv: 2 };
        let mut c = Case::new(sh, GgmlType::F16, GgmlType::F16, 0x5e7);
        c.mask = Some(Case::causal_mask(sh));
        c.sinks = Some(vec![0.7, -2.5, 3.1, 0.0]);
        let mine = c.run();
        let want = c.naive_all();
        let worst = mine
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / b.abs().max(0.1))
            .fold(0f32, f32::max);
        assert!(worst < 1e-2, "sinks mismatch: worst rel {worst} (f16 VKQ accumulator)");

        // a very dominant sink must shrink every output toward 0
        let mut c2 = Case::new(sh, GgmlType::F16, GgmlType::F16, 0x5e8);
        c2.mask = Some(Case::causal_mask(sh));
        c2.sinks = Some(vec![60.0; 4]);
        let a = c2.run();
        c2.sinks = None;
        let b = c2.run();
        let na: f32 = a.iter().map(|v| v.abs()).sum();
        let nb: f32 = b.iter().map(|v| v.abs()).sum();
        assert!(na < nb * 0.01, "dominant sink must dominate: {na} vs {nb}");
    }

    #[test]
    fn fully_masked_row_yields_zeros() {
        let c = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 4)
            .with_mask(Case::uniform_mask(SMALL, f16::NEG_INFINITY));
        let out = c.run();
        for (t, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "row {t}: {v} (expected exact zeros, no NaN)");
            assert_eq!(*v, 0.0, "row {t}: {v}");
        }
        let e = rel_err(&out, &c.naive_all());
        assert!(e <= 0.0, "naive mismatch {e}");
    }

    /// A *finite* very negative mask entry is NOT skipped (only exact `-inf` is,
    /// ops.cpp:8751). C therefore still produces a softmax over all entries —
    /// this is the semantics `n_kv_max` sparse attention relies on.
    #[test]
    fn finite_negative_mask_is_not_skipped() {
        let c = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 5)
            .with_mask(Case::uniform_mask(SMALL, f16::from_f32(-65504.0)));
        let out = c.run();
        assert!(out.iter().any(|&v| v != 0.0), "finite mask must not be skipped");
        let e = rel_err(&out, &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    /// One token row fully masked inside an otherwise normal causal mask.
    #[test]
    fn partially_masked_batch_matches_naive() {
        let sh = SMALL;
        let mut m = Case::causal_mask(sh);
        // token row 0: mask everything (f16 -inf over s_kv entries)
        for s in 0..sh.s_kv {
            m[s * 2..s * 2 + 2].copy_from_slice(&f16::NEG_INFINITY.to_bits().to_ne_bytes());
        }
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 6).with_mask(m);
        let out = c.run();
        for v in c.dst_row(&out, 0, 0) {
            assert_eq!(*v, 0.0);
        }
        let e = rel_err(&out, &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    // ---- 3. ALiBi slopes / logit softcap -----------------------------------

    #[test]
    fn alibi_max_bias_matches_naive() {
        let mut c = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 7).with_mask(Case::causal_mask(SMALL));
        c.max_bias = 4.0; // requires mask (ggml.c:5520)
        assert!(c.max_bias > 0.0);
        let e = rel_err(&c.run(), &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    #[test]
    fn logit_softcap_matches_naive() {
        let mut c = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 8).with_mask(Case::causal_mask(SMALL));
        c.softcap = 15.0;
        let e = rel_err(&c.run(), &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    // ---- 4. GQA broadcast / strides / split-KV -----------------------------

    /// h_kv = H/2 broadcast must equal an explicitly materialized h_kv = H run.
    #[test]
    fn gqa_broadcast_matches_expanded_heads() {
        let a = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 9).with_mask(Case::causal_mask(SMALL));
        let b = a.expand_kv_heads();
        assert_eq!(b.sh.h_kv, b.sh.h);
        let (x, y) = (a.run(), b.run());
        // identical inputs per head → bit-identical outputs
        assert_eq!(x, y, "GQA broadcast mismatch");
    }

    /// Strided K/V rows (KV-cache `view_3d`-style) must give identical results.
    #[test]
    fn padded_kv_views_match_dense() {
        let a = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 10).with_mask(Case::causal_mask(SMALL));
        let b = Case::new(SMALL, GgmlType::F16, GgmlType::F16, 10)
            .with_mask(Case::causal_mask(SMALL))
            .with_padded_kv(32);
        let (x, y) = (a.run(), b.run());
        assert_eq!(x, y, "strided K/V mismatch");
    }

    /// The split-KV path (chunked online softmax + `reduce_partials` merge) must
    /// agree with the single-chunk kernel. `n_chunks = 4` with s_kv = 5 also
    /// exercises the empty-chunk (S == 0) branch of the merge.
    #[test]
    fn split_kv_merge_matches_single_chunk() {
        let sh = Shape { dk: 16, dv: 16, h: 4, h_kv: 2, t: 1, s_kv: 5 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 11).with_mask(Case::causal_mask(sh));
        let whole = c.run();
        for n_chunks in [1usize, 2, 3, 4, 5] {
            let split = c.run_split(n_chunks);
            let e = rel_err(&split, &whole);
            let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
            assert!(e < tol, "n_chunks={n_chunks}: rel err {e}");
        }
        let e = rel_err(&whole, &c.naive_all());
        let tol = Case::tol_for(GgmlType::F16);
        assert!(e < tol, "rel err {e} (tol {tol})");
    }

    /// The split-KV path with an f32 KV cache and an empty tail chunk.
    #[test]
    fn split_kv_empty_chunk_matches() {
        let sh = Shape { dk: 8, dv: 8, h: 2, h_kv: 1, t: 1, s_kv: 3 };
        let c = Case::new(sh, GgmlType::F32, GgmlType::F32, 12).with_mask(Case::causal_mask(sh));
        let e = rel_err(&c.run_split(4), &c.run()); // chunk_size = 1 → 1 empty chunk
        assert!(e < 1e-6, "rel err {e}");
    }

    /// End-to-end graph path: `Context::flash_attn_ext` → `GgmlOp::FlashAttnExt`
/// dispatch → kernel. Must be bit-identical to `flash_attn_ext_ref` (same
/// kernel, same bytes/strides) for both thread counts, and match the naive
/// reference — this pins builder + dispatch + arena addressing together.
    #[test]
    /// graph path with src[4] sinks: same fixture plus per-head sinks, must be
    /// bit-identical to `one_chunk` with the same sinks (agent integration).
    #[test]
    fn graph_path_with_sinks_matches_ref() {
        use crate::graph::Graph;
        let sh = Shape { dk: 8, dv: 8, h: 4, h_kv: 2, t: 3, s_kv: 5 };
        let mut c = Case::new(sh, GgmlType::F16, GgmlType::F16, 22).with_mask(Case::causal_mask(sh));
        c.sinks = Some(vec![0.5, -1.25, 2.0, -0.75]);
        let mut ctx = crate::tensor::Context::new();
        let q = ctx.new_tensor_4d(GgmlType::F32, sh.dk as i64, sh.t as i64, sh.h as i64, 1);
        let k = ctx.new_tensor_4d(GgmlType::F16, sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        let v = ctx.new_tensor_4d(GgmlType::F16, sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        let mask = ctx.new_tensor_4d(GgmlType::F16, sh.s_kv as i64, sh.t as i64, 1, 1);
        let sinks = ctx.new_tensor_1d(GgmlType::F32, sh.h as i64);
        for id in [q, k, v, mask, sinks] {
            ctx.arena_resize_tensor(id);
        }
        ctx.data_bytes_mut(q).unwrap().copy_from_slice(&c.q);
        ctx.data_bytes_mut(k).unwrap().copy_from_slice(&c.k);
        ctx.data_bytes_mut(v).unwrap().copy_from_slice(&c.v);
        ctx.data_bytes_mut(mask).unwrap().copy_from_slice(c.mask.as_ref().unwrap());
        ctx.data_bytes_mut(sinks).unwrap().copy_from_slice(bytemuck::cast_slice(c.sinks.as_ref().unwrap()));
        let out = ctx.flash_attn_ext_sinks(q, k, v, Some(mask), Some(sinks), c.scale, 0.0, 0.0);
        for nth in [1usize, 4] {
            let mut g = Graph::new(8);
            g.build_forward(&ctx, out);
            crate::compute::graph_compute(&mut ctx, &mut g, nth);
            let got = ctx.f32s(out).unwrap().to_vec();
            let want = c.run();
            assert_eq!(got, want, "nth={nth}: sinks graph path != ref");
            assert!(
                rel_err(&got, &c.naive_all()) < Case::tol_for(GgmlType::F16),
                "nth={nth}"
            );
        }
    }

    /// Graph path across the split-KV threshold: a 1-token decode with
    /// `S_kv >= 512` takes `use_split_kv_path` in C (ops.cpp:9261) and in
    /// `compute.rs`'s dispatch — with `n_chunks == nth`, the only parameter
    /// the reference's values depend on. Pins that wiring: the graph result
    /// must be bit-identical to `flash_attn_ext_dispatch_n(fa, nth)` for
    /// several thread counts (and, per `ref_probe_dump_bitexact`, to the
    /// reference .so for the same nth).
    #[test]
    fn graph_path_split_kv_dispatches_with_nth() {
        use crate::graph::Graph;
        let sh = Shape { dk: 16, dv: 16, h: 4, h_kv: 2, t: 1, s_kv: 600 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 23);
        let mut ctx = crate::tensor::Context::new();
        let q = ctx.new_tensor_4d(GgmlType::F32, sh.dk as i64, sh.t as i64, sh.h as i64, 1);
        let k = ctx.new_tensor_4d(GgmlType::F16, sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        let v = ctx.new_tensor_4d(GgmlType::F16, sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        for id in [q, k, v] {
            ctx.arena_resize_tensor(id);
        }
        ctx.data_bytes_mut(q).unwrap().copy_from_slice(&c.q);
        ctx.data_bytes_mut(k).unwrap().copy_from_slice(&c.k);
        ctx.data_bytes_mut(v).unwrap().copy_from_slice(&c.v);
        let out = ctx.flash_attn_ext(q, k, v, None, c.scale, 0.0, 0.0);
        let fa = c.fa();
        assert!(use_split_kv(&fa), "S_kv=600 T=1 must be split-KV eligible");
        for nth in [1usize, 2, 4, 8] {
            let mut g = Graph::new(8);
            g.build_forward(&ctx, out);
            crate::compute::graph_compute(&mut ctx, &mut g, nth);
            let got = ctx.f32s(out).unwrap();
            let mut want = vec![0f32; fa.dst_n_elements() as usize];
            {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut want);
                flash_attn_ext_dispatch_n(&fa, nth, &mut Dst::dense_f32(bytes, fa.dst_ne()));
            }
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "nth={nth}: graph split-KV dispatch differs"
            );
        }
    }

    fn graph_path_matches_ref() {
        use crate::graph::Graph;
        let sh = Shape { dk: 8, dv: 8, h: 4, h_kv: 2, t: 3, s_kv: 5 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 21).with_mask(Case::causal_mask(sh));
        let mut ctx = crate::tensor::Context::new();
        let q = ctx.new_tensor_4d(GgmlType::F32, sh.dk as i64, sh.t as i64, sh.h as i64, 1);
        let k = ctx.new_tensor_4d(GgmlType::F16, sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        let v = ctx.new_tensor_4d(GgmlType::F16, sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1);
        let mask = ctx.new_tensor_4d(GgmlType::F16, sh.s_kv as i64, sh.t as i64, 1, 1);
        for id in [q, k, v, mask] {
            ctx.arena_resize_tensor(id);
        }
        // the Case buffers are already dense in exactly these ne/nb layouts
        ctx.data_bytes_mut(q).unwrap().copy_from_slice(&c.q);
        ctx.data_bytes_mut(k).unwrap().copy_from_slice(&c.k);
        ctx.data_bytes_mut(v).unwrap().copy_from_slice(&c.v);
        ctx.data_bytes_mut(mask).unwrap().copy_from_slice(c.mask.as_ref().unwrap());
        let out = ctx.flash_attn_ext(q, k, v, Some(mask), c.scale, 0.0, 0.0);
        assert_eq!(ctx.ne(out), &[sh.dv as i64, sh.h as i64, sh.t as i64, 1]);
        for nth in [1usize, 4] {
            let mut g = Graph::new(8);
            g.build_forward(&ctx, out);
            crate::compute::graph_compute(&mut ctx, &mut g, nth);
            let got = ctx.f32s(out).unwrap().to_vec();
            let want = c.run();
            assert_eq!(got, want, "nth={nth}: graph path != ref (bit-exact expected)");
            let e = rel_err(&got, &c.naive_all());
            let tol = Case::tol_for(GgmlType::F16); // F16 V fp16 VKQ accumulator (ops.cpp:8773)
            assert!(e < tol, "nth={nth}: rel err {e}");
        }
    }

    /// Output layout is C's `permute(0,2,1,3)`: `[DV, H, T, S]` with token-major
    /// rows (`offset = s*T*H*DV + t*H*DV + h*DV + d`).
    #[test]
    fn dst_layout_is_permuted() {
        let sh = Shape { dk: 8, dv: 8, h: 4, h_kv: 4, t: 2, s_kv: 2 };
        let c = Case::new(sh, GgmlType::F32, GgmlType::F32, 13); // no mask: full attention
        let fa = c.fa();
        assert_eq!(fa.dst_ne(), [8, 4, 2, 1]);
        let out = c.run();
        assert_eq!(out.len(), 8 * 4 * 2);
        // token 1 / head 3 row sits at 1*H*DV + 3*DV
        let row = c.dst_row(&out, 1, 3);
        let want = c.naive_row(1, 3);
        assert!(rel_err(row, &want) < 1e-6);
    }

    // ---- 5. tiled kernel vs the reference binary (parity/ref_fa_dump.c) -----

    /// One reference case: inputs as generated by `parity/ref_fa_dump.c`.
    struct RefCase {
        mode: i64,
        sh: Shape,
        kty: GgmlType,
        vty: GgmlType,
        mask: Option<Vec<u8>>,
        sinks: Option<Vec<f32>>,
        scale: f32,
        max_bias: f32,
        softcap: f32,
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        out: Vec<f32>,
    }

    fn read_ref_dump() -> Option<Vec<RefCase>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fa_ref.bin");
        let buf = std::fs::read(path).ok()?;
        let mut o = 0usize;
        let mut next = |o: &mut usize| -> Vec<u8> {
            let len = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap()) as usize;
            *o += 4;
            let v = buf[*o..*o + len].to_vec();
            *o += len;
            v
        };
        let mut cases = Vec::new();
        while o < buf.len() {
            let hdr = next(&mut o);
            let g = |i: usize| i64::from_le_bytes(hdr[i * 8..i * 8 + 8].try_into().unwrap());
            let (q, k, v, mask, sinks, out, extra) = (
                next(&mut o),
                next(&mut o),
                next(&mut o),
                next(&mut o),
                next(&mut o),
                next(&mut o),
                next(&mut o),
            );
            let f = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            let ty = |x: i64| if x == 0 { GgmlType::F32 } else { GgmlType::F16 };
            cases.push(RefCase {
                mode: g(0),
                sh: Shape {
                    dk: g(1) as usize,
                    dv: g(2) as usize,
                    h: g(3) as usize,
                    h_kv: g(4) as usize,
                    t: g(5) as usize,
                    s_kv: g(6) as usize,
                },
                kty: ty(g(7)),
                vty: ty(g(8)),
                mask: (!mask.is_empty()).then_some(mask),
                sinks: (!sinks.is_empty()).then(|| (0..sinks.len() / 4).map(|i| f(&sinks, i)).collect()),
                scale: f(&extra, 0),
                max_bias: f(&extra, 1),
                softcap: f(&extra, 2),
                q,
                k,
                v,
                out: (0..out.len() / 4).map(|i| f(&out, i)).collect(),
            });
        }
        Some(cases)
    }

    /// The reference dump's K/V tensors are dense in *its* layout; this builds the
    /// same `Src`s the port uses (nb derived exactly like ggml_new_tensor_impl).
    fn ref_case_fa(c: &RefCase) -> FlashAttn<'_> {
        let sh = c.sh;
        FlashAttn {
            q: Src::dense(&c.q, GgmlType::F32, [sh.dk as i64, sh.t as i64, sh.h as i64, 1]),
            k: Src::dense(&c.k, c.kty, [sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1]),
            v: Src::dense(&c.v, c.vty, [sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1]),
            mask: c
                .mask
                .as_ref()
                .map(|m| Src::dense(m, GgmlType::F16, [sh.s_kv as i64, sh.t as i64, 1, 1])),
            scale: c.scale,
            max_bias: c.max_bias,
            logit_softcap: c.softcap,
            sinks: c.sinks.as_deref(),
        }
    }

    /// Path affinity: which C kernel the reference ran for this case.
    fn expected_tiled(c: &RefCase) -> bool {
        c.sh.t >= Q_TILE && c.sh.dv % F32_EPR as usize == 0
    }

    /// The full kernel-vs-reference table: `parity/ref_fa_dump.c` through the
    /// *public* reference API (AVX512 build, 1 thread) vs this port's
    /// [`flash_attn_ext_ref`] (which dispatches exactly like C).
    ///
    /// Regenerate the dump with `parity/ref_fa_dump.c` (see its header); the test
    /// skips when it is absent.
    #[test]
    fn ref_dump_matches_all_cases() {
        let Some(cases) = read_ref_dump() else {
            eprintln!("skip: parity/fa_ref.bin missing (build parity/ref_fa_dump.c)");
            return;
        };
        assert!(!cases.is_empty());
        let mut failures: Vec<String> = Vec::new();
        println!(
            "{:>4} {:>9} {:>4} {:>4} {:>4} {:>5} {:>5} {:>4} {:>9} {:>10} {:>10}",
            "mode", "kernel", "dk", "dv", "T", "S_kv", "H_kv", "KV", "bitexact", "max|abs|", "max|rel|"
        );
        for c in &cases {
            let got = flash_attn_ext_ref(&ref_case_fa(c));
            assert_eq!(got.len(), c.out.len(), "mode {}: length", c.mode);
            let bits = got.iter().zip(&c.out).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            let max_abs = got.iter().zip(&c.out).map(|(a, b)| (a - b).abs() as f64).fold(0.0f64, f64::max);
            let rel = rel_err(&got, &c.out);
            println!(
                "{:>4} {:>9} {:>4} {:>4} {:>4} {:>5} {:>5} {:>4} {bits:>4}/{:<4} {:>10.3e} {:>10.3e}",
                c.mode,
                if expected_tiled(c) { "tiled" } else { "one_chunk" },
                c.sh.dk,
                c.sh.dv,
                c.sh.t,
                c.sh.s_kv,
                c.sh.h_kv,
                if c.kty == GgmlType::F32 { "f32" } else { "f16" },
                got.len(),
                max_abs,
                rel
            );
            if rel >= 1e-6 {
                failures.push(format!("mode {}: rel err {rel:.3e}", c.mode));
            }
        }
        assert!(failures.is_empty(), "reference mismatch:\n{}", failures.join("\n"));
    }

    /// Bit-level claim, per case: **every** case — tiled *and* one_chunk,
    /// incl. sinks (5/13/14), softcap, ALiBi, DV != DK, partial KV tiles and
    /// the padded-tail quirk — must be bit-identical to the reference. The
    /// historical 1-2 ulp one_chunk residual (gpt-oss FA-tail band) was the
    /// sinks `S = S*ms + vs` line computed as an FMA where the reference
    /// binary keeps it unfused, plus the unfused F32-V `ggml_vec_mad_f32`;
    /// both are reproduced now (see the module header's contraction profile),
    /// and the row-shape behaviour is pinned separately by
    /// `ref_probe_dump_bitexact` / `row_shape_invariance`.
    #[test]
    fn ref_dump_bitexact() {
        let Some(cases) = read_ref_dump() else {
            eprintln!("skip: parity/fa_ref.bin missing");
            return;
        };
        let mut bad: Vec<String> = Vec::new();
        for c in &cases {
            let got = flash_attn_ext_ref(&ref_case_fa(c));
            let bits = got.iter().zip(&c.out).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            if bits == got.len() {
                continue;
            }
            let rel = rel_err(&got, &c.out);
            let strict = expected_tiled(c);
            let msg = format!(
                "mode {} (kernel {}): {bits}/{} bits, rel {rel:.3e}",
                c.mode,
                if expected_tiled(c) { "tiled" } else { "one_chunk" },
                got.len()
            );
            if strict || rel >= 1e-6 {
                bad.push(msg);
            } else {
                eprintln!("{msg} [one_chunk, documented <1e-6 residual]");
            }
        }
        assert!(bad.is_empty(), "not bit-exact vs reference:\n{}", bad.join("\n"));
    }

    /// Minimal tiled sanity: one KV position -> the output *is* v (softmax = 1).
    #[test]
    fn tiled_single_kv_returns_v() {
        let sh = Shape { dk: 16, dv: 16, h: 4, h_kv: 2, t: 64, s_kv: 1 };
        let c = Case::new(sh, GgmlType::F32, GgmlType::F32, 41); // no mask
        let fa = c.fa();
        let ne = fa.dst_ne();
        let mut out = vec![0f32; fa.dst_n_elements() as usize];
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
            tiled(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne));
        }
        for h in 0..sh.h {
            let vh = h / (sh.h / sh.h_kv);
            for d in 0..sh.dv {
                let want = c.elem(&c.v, &c.vnb, c.vty, 0, vh, d);
                let got = out[0 * sh.h * sh.dv + h * sh.dv + d];
                assert_eq!(got, want, "h={h} d={d}");
            }
        }
    }

    /// Full softmax through the tiled kernel vs the naive two-pass reference.
    #[test]
    fn tiled_uniform_mask_matches_naive() {
        let sh = Shape { dk: 16, dv: 16, h: 4, h_kv: 2, t: 64, s_kv: 64 };
        let c = Case::new(sh, GgmlType::F32, GgmlType::F32, 42).with_mask(Case::uniform_mask(sh, f16::from_f32(0.0)));
        let fa = c.fa();
        let ne = fa.dst_ne();
        let mut out = vec![0f32; fa.dst_n_elements() as usize];
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
            tiled(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne));
        }
        let want = c.naive_all();
        for t in 0..sh.t {
            for h in 0..sh.h {
                let got = c.dst_row(&out, t, h);
                let w = c.dst_row(&want, t, h);
                for d in 0..sh.dv {
                    let rel = (got[d] - w[d]).abs() / w[d].abs().max(1e-3);
                    assert!(rel < 1e-4, "t={t} h={h} d={d}: got {} want {} (rel {rel:.2e})", got[d], w[d]);
                }
            }
        }
    }

    /// Diagnostic: for every dump case, report how well *both* kernels match the
    /// reference — identifies which C path the reference actually took.
    #[test]
    fn ref_dump_path_affinity() {
        let Some(cases) = read_ref_dump() else { return };
        println!("{:>4} {:>4} {:>4} {:>9} {:>10} {:>10}", "mode", "T", "dv", "expect", "tiled_rel", "onechunk_rel");
        for c in &cases {
            let fa = ref_case_fa(c);
            let ne = fa.dst_ne();
            let n = fa.dst_n_elements() as usize;
            let run = |which: u8| {
                let mut out = vec![0f32; n];
                {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                    let mut d = Dst::dense_f32(bytes, ne);
                    if which == 0 { tiled(&fa, 0, fa.n_rows() as usize, &mut d) }
                    else { one_chunk(&fa, 0, fa.n_rows() as usize, &mut d) }
                }
                out
            };
            let t = rel_err(&run(0), &c.out);
            let o = rel_err(&run(1), &c.out);
            println!("{:>4} {:>4} {:>4} {:>9} {t:>10.3e} {o:>10.3e}",
                c.mode, c.sh.t, c.sh.dv, if expected_tiled(c) { "tiled" } else { "one_chunk" });
        }
    }

    /// The tiled kernel must be reachable exactly under C's condition
    /// (ops.cpp:9317-9330).
    #[test]
    fn tiled_selection_matches_c_condition() {
        // T = 63: below Q_TILE -> one_chunk
        let sh = Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 63, s_kv: 63 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 31).with_mask(Case::causal_mask(sh));
        assert!(!use_tiled(&c.fa()));
        // T = 64: tiled
        let sh = Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 64, s_kv: 64 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 32).with_mask(Case::causal_mask(sh));
        assert!(use_tiled(&c.fa()));
        // sync batch D (ops.cpp:9552): the `DV % f32_epr == 0` term now
        // applies only on non-x86-64 — on x86 a ragged DV takes the tiled
        // path and the masked simd_gemm tail (simd-gemm.h:59) handles it
        let sh = Shape { dk: 32, dv: 40, h: 4, h_kv: 2, t: 64, s_kv: 64 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 33).with_mask(Case::causal_mask(sh));
        assert!(use_tiled(&c.fa()), "DV % 16 != 0 still tiles on x86-64 (ops.cpp:9552)");
    }

    /// `tiled` and `one_chunk` implement the same math with different
    /// accumulation orders; on the same F32-KV inputs they must agree well within
    /// the f32 noise floor of a 64-wide softmax.
    #[test]
    fn tiled_agrees_with_one_chunk_f32_kv() {
        let sh = Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 71, s_kv: 71 };
        let c = Case::new(sh, GgmlType::F32, GgmlType::F32, 34).with_mask(Case::causal_mask(sh));
        let fa = c.fa();
        assert!(use_tiled(&fa));
        let ne = fa.dst_ne();
        let mut t = vec![0f32; fa.dst_n_elements() as usize];
        let mut o = vec![0f32; fa.dst_n_elements() as usize];
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut t);
            tiled(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne));
        }
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut o);
            one_chunk(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne));
        }
        let e = rel_err(&t, &o);
        assert!(e < 1e-6, "tiled vs one_chunk rel err {e}");
        let e = rel_err(&t, &c.naive_all());
        assert!(e < 1e-4, "tiled vs naive rel err {e}");
    }

    /// The tiled kernel must honour the KV-cache strides: a padded (view_3d
    /// style) KV buffer gives bit-identical output to a dense one.
    #[test]
    fn tiled_padded_kv_views_match_dense() {
        let sh = Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 96, s_kv: 96 };
        let a = Case::new(sh, GgmlType::F16, GgmlType::F16, 51).with_mask(Case::causal_mask(sh));
        let b = Case::new(sh, GgmlType::F16, GgmlType::F16, 51)
            .with_mask(Case::causal_mask(sh))
            .with_padded_kv(48);
        assert!(use_tiled(&a.fa()));
        assert_eq!(a.run(), b.run(), "tiled: strided K/V mismatch");
        // and the strided run must agree with the naive reference too
        let e = rel_err(&b.run(), &b.naive_all());
        assert!(e < 1e-2, "tiled strided vs naive rel err {e}");
    }

    /// Row sharding must not change any output bit (the C kernel partitions rows
    /// across threads; tiles never cross a head boundary).
    #[test]
    fn tiled_row_sharding_is_bit_exact() {
        let sh = Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 130, s_kv: 130 };
        let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 35).with_mask(Case::causal_mask(sh));
        let fa = c.fa();
        assert!(use_tiled(&fa));
        let ne = fa.dst_ne();
        let n = fa.dst_n_elements() as usize;
        let whole = flash_attn_ext_ref(&fa); // dispatches to tiled
        for splits in [1usize, 2, 3, 5, 8, 17, 64] {
            let mut out = vec![0f32; n];
            let nr = fa.n_rows() as usize;
            let rows = (nr + splits - 1) / splits;
            {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                let mut d = Dst::dense_f32(bytes, ne);
                for s in 0..splits {
                    let ir0 = s * rows;
                    if ir0 >= nr {
                        break;
                    }
                    flash_attn_ext_dispatch(&fa, ir0, (ir0 + rows).min(nr), &mut d);
                }
            }
            assert_eq!(out, whole, "splits={splits}: sharding changed a bit");
        }
    }

    /// Kernel-level A/B of the two paths the C dispatcher chooses between, on
    /// realistic prefill shapes (`cargo test --release -p ggml --lib
    /// bench_fa_paths -- --ignored --nocapture`). The e2e prefill is dominated by
    /// mul_mat/repack, so this is the honest measure of the FA switch itself.
    #[test]
    #[ignore = "manual: prefill-shape FA path A/B"]
    fn bench_fa_paths() {
        let cases: [(&str, Shape); 4] = [
            ("qwen2.5-0.5b D=64 H=14/2 T=128", Shape { dk: 64, dv: 64, h: 14, h_kv: 2, t: 128, s_kv: 128 }),
            ("qwen2.5-0.5b D=64 H=14/2 T=512", Shape { dk: 64, dv: 64, h: 14, h_kv: 2, t: 512, s_kv: 512 }),
            ("gpt-oss-20b D=64 H=64/8 T=128", Shape { dk: 64, dv: 64, h: 64, h_kv: 8, t: 128, s_kv: 128 }),
            ("gpt-oss-20b D=64 H=64/8 T=512", Shape { dk: 64, dv: 64, h: 64, h_kv: 8, t: 512, s_kv: 512 }),
        ];
        for (label, sh) in cases {
            let c = Case::new(sh, GgmlType::F16, GgmlType::F16, 77).with_mask(Case::causal_mask(sh));
            let fa = c.fa();
            assert!(use_tiled(&fa));
            println!("{label}:");
            for (name, which) in [("tiled", 0u8), ("one_chunk", 1u8)] {
                let ne = fa.dst_ne();
                let mut best = f64::INFINITY;
                let mut out = vec![0f32; fa.dst_n_elements() as usize];
                for _ in 0..3 {
                    let t = std::time::Instant::now();
                    {
                        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                        let mut d = Dst::dense_f32(bytes, ne);
                        if which == 0 {
                            tiled(&fa, 0, fa.n_rows() as usize, &mut d)
                        } else {
                            one_chunk(&fa, 0, fa.n_rows() as usize, &mut d)
                        }
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                println!(
                    "  {name:>9}: {:8.1} ms  ({:7.1} tok/s)",
                    best * 1e3,
                    fa.n_tokens() as f64 / best
                );
            }
        }
    }

    /// Row-count invariance of the port's own kernels (the parity gap the
    /// gpt-oss FA-tail band and the Ds4 MTP verify batch exposed): the same
    /// logical attention — byte-identical q/k/v/mask/sinks values — must give
    /// bit-identical rows whether computed as one T-row batch (a speculative
    /// verify batch), T separate 1-row batches (plain decode), or a partial
    /// (T-1)-row batch. `parity/ref_fa_probe.c` proves the reference is
    /// invariant the same way for every `one_chunk`/`tiled` shape it probes
    /// (the split-KV decode shapes, `T == 1 && S_kv >= 512` at `nth >= 2`,
    /// are deliberately excluded — the reference itself is only *per-shape*
    /// faithful there; see `ref_probe_dump_bitexact`).
    #[test]
    fn row_shape_invariance() {
        let shapes = [
            // gpt-oss verify batch geometry: 64/8 heads, sinks, causal mask
            (Shape { dk: 64, dv: 64, h: 64, h_kv: 8, t: 5, s_kv: 21 }, 0x5aa1),
            // gpt-oss prompt prefill (mode 14's shape)
            (Shape { dk: 64, dv: 64, h: 64, h_kv: 8, t: 5, s_kv: 5 }, 0x5aa2),
            // 4-row verify batch without sinks
            (Shape { dk: 64, dv: 64, h: 16, h_kv: 2, t: 4, s_kv: 9 }, 0x5aa3),
            // T=63: the largest batch that still dispatches one_chunk (T >= 64
            // takes the tiled kernel, whose tile accumulation order is a
            // *different reference kernel* — a tiled batch legitimately
            // differs from its 1-row one_chunk slices, on both sides)
            (Shape { dk: 32, dv: 32, h: 8, h_kv: 2, t: 63, s_kv: 70 }, 0x5aa4),
        ];
        for (sh, seed) in shapes {
            let mut c = Case::new(sh, GgmlType::F16, GgmlType::F16, seed).with_mask(Case::causal_mask(sh));
            if sh.h == 64 {
                c.sinks = Some((0..sh.h as u32).map(|i| (i as f32) * 0.031 - 1.0).collect());
            }
            // slice runner: rows [t0, t1) of the same case (fresh Case buffers
            // re-sliced, exactly like parity/ref_fa_probe.c's run_slice)
            let run_rows = |t0: usize, t1: usize| -> Vec<f32> {
                let rows = t1 - t0;
                let mut qbuf = vec![0f32; sh.dk * rows * sh.h];
                for hh in 0..sh.h {
                    for tt in 0..rows {
                        let off = ((hh * sh.t + t0 + tt) * sh.dk * 4) as usize;
                        for d in 0..sh.dk {
                            qbuf[(hh * rows + tt) * sh.dk + d] = f32_at(&c.q, off + d * 4);
                        }
                    }
                }
                let mut mbuf: Option<Vec<u8>> = None;
                if let Some(mask) = &c.mask {
                    let mut m = Vec::with_capacity(sh.s_kv * rows * 2);
                    for tt in 0..rows {
                        let off = (t0 + tt) * sh.s_kv * 2;
                        m.extend_from_slice(&mask[off..off + sh.s_kv * 2]);
                    }
                    mbuf = Some(m);
                }
                let mask = mbuf.as_ref().map(|m| Src::dense(m, GgmlType::F16, [sh.s_kv as i64, rows as i64, 1, 1]));
                let fa2 = FlashAttn {
                    q: Src::dense(bytemuck::cast_slice(&qbuf), GgmlType::F32, [sh.dk as i64, rows as i64, sh.h as i64, 1]),
                    k: Src::dense(&c.k, GgmlType::F16, [sh.dk as i64, sh.s_kv as i64, sh.h_kv as i64, 1]),
                    v: Src::dense(&c.v, GgmlType::F16, [sh.dv as i64, sh.s_kv as i64, sh.h_kv as i64, 1]),
                    mask,
                    scale: c.scale,
                    max_bias: c.max_bias,
                    logit_softcap: c.softcap,
                    sinks: c.sinks.as_deref(),
                };
                let ne = fa2.dst_ne();
                let mut out = vec![0f32; fa2.dst_n_elements() as usize];
                {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                    let mut d = Dst::dense_f32(bytes, ne);
                    flash_attn_ext_dispatch_n(&fa2, 1, &mut d);
                }
                out
            };
            let batch = run_rows(0, sh.t);
            for t in 0..sh.t {
                let one = run_rows(t, t + 1);
                let row = &batch[t * sh.h * sh.dv..(t + 1) * sh.h * sh.dv];
                assert_eq!(
                    one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "T={} shape {sh:?}: 1-row run of token {t} differs from the batch row",
                    sh.t
                );
            }
            if sh.t >= 2 {
                let rest = run_rows(1, sh.t);
                assert_eq!(
                    rest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    batch[sh.h * sh.dv..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "T={} shape {sh:?}: (T-1)-row run differs from the batch tail",
                    sh.t
                );
            }
        }
    }

    /// The reference row-shape probe gate: `parity/ref_fa_probe.c` runs the
    /// *reference* (AVX512 build) on byte-identical inputs as a T-row batch
    /// [tag 0], T 1-row slices [tag 1] and a (T-1)-row batch [tag 2], at
    /// `nth = 1` and `nth = 8`, and writes every record to
    /// `parity/fa_probe.bin`. This replays each record through the port with
    /// the C-matching per-shape dispatch (1-row `S_kv >= 512` at `nth >= 2`
    /// takes `split_kv` with `n_chunks = nth`, everything else the
    /// row-invariant `one_chunk`/`tiled`) and requires **every** element
    /// bit-exact — the closure of the FA one_chunk tail that used to show
    /// 1-2 ulp (the sinks `S = S*ms + vs` line and the split-KV merge
    /// contraction; see the module header).
    #[test]
    fn ref_probe_dump_bitexact() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fa_probe.bin");
        let Ok(buf) = std::fs::read(path) else {
            eprintln!("skip: parity/fa_probe.bin missing (build & run parity/ref_fa_probe.c)");
            return;
        };
        let rd_f32 = |b: &[u8], i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let mut o = 0usize;
        let mut cases = 0usize;
        while o + 4 <= buf.len() {
            let magic = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
            assert_eq!(magic, 0xF17E0001, "bad probe record magic at {o}");
            o += 4;
            let next = |o: &mut usize| -> Vec<u8> {
                let len = u32::from_le_bytes(buf[*o..*o + 4].try_into().unwrap()) as usize;
                *o += 4;
                let v = buf[*o..*o + len].to_vec();
                *o += len;
                v
            };
            let geom = next(&mut o);
            let knobs = next(&mut o);
            let (qb, kb, vb, mb, sb) =
                (next(&mut o), next(&mut o), next(&mut o), next(&mut o), next(&mut o));
            let g = |i: usize| i64::from_le_bytes(geom[i * 8..i * 8 + 8].try_into().unwrap());
            let (dk, dv, h, h_kv, t, s_kv) = (
                g(0) as usize,
                g(1) as usize,
                g(2) as usize,
                g(3) as usize,
                g(4) as usize,
                g(5) as usize,
            );
            let (scale, max_bias, softcap) = (rd_f32(&knobs, 0), rd_f32(&knobs, 1), rd_f32(&knobs, 2));
            let sinks: Option<Vec<f32>> =
                (!sb.is_empty()).then(|| (0..sb.len() / 4).map(|i| rd_f32(&sb, i)).collect());
            // one record group per block: tag-0 batch, t tag-1 slices, tag-2
            // (T-1)-rows — the probe writes a fresh input block per nth
            {
                let label_head = format!("case{cases} T={t} S={s_kv}");
                // the group's thread count comes from the first record's header
                let first_nth =
                    u32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap()) as usize;
                let nth = first_nth;
                let run_rows = |t0: usize, t1: usize| -> Vec<f32> {
                    let rows = t1 - t0;
                    let mut qbuf = vec![0f32; dk * rows * h];
                    for hh in 0..h {
                        for tt in 0..rows {
                            let src = ((hh * t + t0 + tt) * dk) * 4;
                            for d in 0..dk {
                                qbuf[(hh * rows + tt) * dk + d] = rd_f32(&qb, src / 4 + d);
                            }
                        }
                    }
                    let mut mbuf: Option<Vec<u8>> = None;
                    if !mb.is_empty() {
                        let mut m = Vec::with_capacity(s_kv * rows * 2);
                        for tt in 0..rows {
                            let off = (t0 + tt) * s_kv * 2;
                            m.extend_from_slice(&mb[off..off + s_kv * 2]);
                        }
                        mbuf = Some(m);
                    }
                    let fa = FlashAttn {
                        q: Src::dense(
                            bytemuck::cast_slice(&qbuf),
                            GgmlType::F32,
                            [dk as i64, rows as i64, h as i64, 1],
                        ),
                        k: Src::dense(&kb, GgmlType::F16, [dk as i64, s_kv as i64, h_kv as i64, 1]),
                        v: Src::dense(&vb, GgmlType::F16, [dv as i64, s_kv as i64, h_kv as i64, 1]),
                        mask: mbuf
                            .as_ref()
                            .map(|m| Src::dense(m, GgmlType::F16, [s_kv as i64, rows as i64, 1, 1])),
                        scale,
                        max_bias,
                        logit_softcap: softcap,
                        sinks: sinks.as_deref(),
                    };
                    let ne = fa.dst_ne();
                    let mut out = vec![0f32; fa.dst_n_elements() as usize];
                    {
                        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
                        let mut d = Dst::dense_f32(bytes, ne);
                        flash_attn_ext_dispatch_n(&fa, nth, &mut d);
                    }
                    out
                };
                let label = format!("{label_head} nth={nth}");
                for _rec in 0..1 + t + usize::from(t >= 2) {
                    let (tag, tt) = (
                        u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()),
                        u32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()),
                    );
                    let rec_nth = u32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap());
                    o += 12;
                    let d = next(&mut o);
                    let ref_out = (0..d.len() / 4).map(|i| rd_f32(&d, i)).collect::<Vec<f32>>();
                    assert_eq!(rec_nth as usize, nth, "{label}: record group order");
                    let port = match tag {
                        0 => run_rows(0, t),
                        1 => run_rows(tt as usize, tt as usize + 1),
                        _ => run_rows(1, t),
                    };
                    let n = port.len();
                    let bits = port
                        .iter()
                        .zip(ref_out.iter())
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count();
                    assert_eq!(
                        bits,
                        n,
                        "{label} tag={tag} t={tt}: only {bits}/{n} bit-exact vs reference"
                    );
                }
            }
            cases += 1;
        }
        assert!(cases >= 10, "probe dump truncated: {cases} blocks");
    }

    /// `ggml_v_expf` (AVX512 polynomial, vec.h:1172) must reproduce the
    /// reference's variant: accuracy within its documented bound plus the two
    /// saturation branches.
    #[test]
    fn v_expf_saturates_like_the_c_polynomial() {
        assert_eq!(v_expf(0.0), 1.0);
        assert_eq!(v_expf(f32::NEG_INFINITY), 0.0);
        assert_eq!(v_expf(f32::INFINITY), f32::INFINITY);
        assert_eq!(v_expf(-1000.0), 0.0); // |n| > 192 and n <= 0
        assert_eq!(v_expf(1000.0), f32::INFINITY);
        for &x in &[-87.0f32, -20.0, -1.0, -0.5, 0.5, 1.0, 5.0, 88.0] {
            let want = x.exp();
            let got = v_expf(x);
            let rel = ((got - want).abs() as f64) / (want.abs() as f64).max(1e-30);
            // documented bound: max error 1.45358 + 0.5 ulps (vec.h:1168)
            assert!(rel < 4e-7, "x={x}: v_expf {got} vs expf {want} (rel {rel:.2e})");
        }
    }

    /// The AVX512 `ggml_v_expf(__m512)` body (`simd_x86::v_expf_16`) must be
    /// bit-identical to the elementwise scalar ports on a dense exponent sweep
    /// — both `v_expf` (f64 `scalef` emulation) and `ops::ggml_expf_v512`
    /// (powi scaling) — including the `|n| > 192` saturation region and the
    /// boundary inputs where `n` crosses 126/192.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn v_expf_vector_matches_scalar_bit_exact() {
        if !crate::simd_x86::avx512() {
            eprintln!("skip: host has no AVX512F+DQ");
            return;
        }
        let mut xs: Vec<f32> = Vec::new();
        for e in -140i32..=140 {
            for m in [0.5f32, 0.7, 0.9, 1.1, 1.3, 1.4990234] {
                xs.push(m * (2.0f32).powi(e));
                xs.push(-m * (2.0f32).powi(e));
            }
        }
        xs.extend([
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            88.37,
            -103.98,
            700.0,
            -700.0,
        ]);
        // n = round(x * log2e): the |n| boundaries sit near these exponents
        for chunk in xs.chunks(16) {
            let mut buf = [0f32; 16];
            buf[..chunk.len()].copy_from_slice(chunk);
            unsafe {
                let v = crate::simd_x86::v_expf_16(core::arch::x86_64::_mm512_loadu_ps(buf.as_ptr()));
                core::arch::x86_64::_mm512_storeu_ps(buf.as_mut_ptr(), v);
            }
            for (j, &g) in buf[..chunk.len()].iter().enumerate() {
                let x = chunk[j];
                let want1 = v_expf(x);
                let want2 = crate::ops::ggml_expf_v512(x);
                assert_eq!(g.to_bits(), want1.to_bits(), "x={x} lane {j} vs v_expf");
                assert_eq!(g.to_bits(), want2.to_bits(), "x={x} lane {j} vs ggml_expf_v512");
            }
        }
    }

    /// The AVX512 tiled kernel vs the forced-scalar elementwise tile body on
    /// the same inputs: every vector kernel in the tile (simd_gemm's FMA
    /// tiles, vec_scale, vec_add, the v_expf softmax chunks, the vcvtph2ps V
    /// pack) is per-lane, so the two must agree bit-for-bit. This is the
    /// scalar-vs-SIMD half of the bit-exactness evidence; the reference-dump
    /// tests (`ref_dump_bitexact`) are the kernel-vs-reference half.
    #[test]
    fn tiled_simd_matches_scalar_bit_exact() {
        if !crate::simd_x86::avx512() {
            eprintln!("skip: host has no AVX512F+DQ (both paths are scalar)");
            return;
        }
        let shapes = [
            (Shape { dk: 64, dv: 64, h: 4, h_kv: 2, t: 64, s_kv: 64 }, 0), // plain
            (Shape { dk: 64, dv: 64, h: 4, h_kv: 2, t: 128, s_kv: 128 }, 1), // two tiles
            (Shape { dk: 32, dv: 32, h: 4, h_kv: 2, t: 96, s_kv: 130 }, 2), // partial Q+KV tiles
            (Shape { dk: 64, dv: 128, h: 2, h_kv: 1, t: 96, s_kv: 96 }, 3), // DV != DK
            (Shape { dk: 64, dv: 64, h: 4, h_kv: 2, t: 64, s_kv: 200 }, 4), // sinks + softcap
        ];
        for (sh, kind) in shapes {
            for kty in [GgmlType::F16, GgmlType::F32] {
                let mut c = Case::new(sh, kty, kty, 60 + kind).with_mask(Case::causal_mask(sh));
                match kind {
                    1 => c.max_bias = 8.0,      // ALiBi slopes
                    2 => c.softcap = 1e-4,      // padded-KV-tail softcap quirk
                    4 => c.sinks = Some(vec![0.5, -1.25, 2.0, -0.75]),
                    _ => {}
                }
                let fa = c.fa();
                assert!(use_tiled(&fa));
                let ne = fa.dst_ne();
                let n = fa.dst_n_elements() as usize;
                let mut simd_out = vec![0f32; n];
                let mut scalar_out = vec![0f32; n];
                {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut simd_out);
                    tiled(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne));
                }
                {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut scalar_out);
                    tiled_impl(&fa, 0, fa.n_rows() as usize, &mut Dst::dense_f32(bytes, ne), false);
                }
                for (j, (a, b)) in simd_out.iter().zip(&scalar_out).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "kind={kind} kty={kty:?} elem {j}: simd {a} scalar {b}"
                    );
                }
            }
        }
    }
}
