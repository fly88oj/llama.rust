//! Tensor/context/graph builder — port of ggml.c's tensor ops constructors and
//! `ggml_build_forward` (compute graph).
//!
//! C uses raw pointers into a context arena; here tensors are indices
//! (`TensorId`) into `Context::tensors`, with storage either in the context
//! arena or referencing external (mmap) buffers. This mirrors the C builder API
//! shape while making graph construction safe.

use crate::types::{GgmlType, MAX_DIMS, MAX_NAME, MAX_OP_PARAMS, MAX_SRC};

pub const GGML_MAX_DIMS: usize = MAX_DIMS;
pub const GGML_MAX_NAME: usize = MAX_NAME;

// `madvise(2)`, declared locally (std links libc; no crate dependency): the
// arena's backing allocation is a large anonymous mmap, and MADV_HUGEPAGE
// there makes the kernel satisfy first-touch faults with 2 MiB pages — the
// first forward faults ~hundreds of pages instead of ~50k 4 KiB ones.
// MADV_HUGEPAGE == 14 on Linux (mman-linux.h, stable across archs we target).
#[cfg(target_os = "linux")]
extern "C" {
    fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
}

/// Ask for transparent huge pages over `[ptr, ptr+len)`. The malloc user
/// pointer is 16-byte aligned, not page aligned, and `madvise` rejects
/// unaligned addresses with EINVAL — so round the range out to whole pages.
/// A no-op (returns nonzero, ignored) on VMAs the kernel will not collapse,
/// safe to call opportunistically after every arena growth.
#[cfg(target_os = "linux")]
unsafe fn arena_advise_hugepage(ptr: *mut u8, len: usize) {
    const PAGE: usize = 4096; // x86_64/linux page size (vm.mmap_min_addr scale)
    let start = (ptr as usize) & !(PAGE - 1);
    let end = (ptr as usize + len + PAGE - 1) & !(PAGE - 1);
    if end <= start {
        return;
    }
    let _ = unsafe { madvise(start as *mut core::ffi::c_void, end - start, 14) };
}

#[cfg(not(target_os = "linux"))]
unsafe fn arena_advise_hugepage(_ptr: *mut u8, _len: usize) {}

/// `enum ggml_op` (subset grows as ops are ported; discriminants match ggml.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlOp {
    None = 0,
    Dup = 1,
    Add = 2,
    Mul = 3,
    Div = 4,
    Sub = 5,
    Norm = 6,
    // SquaredMulMat (= 8) and MulView (= 53) below are DEAD VARIANTS kept for
    // a concrete reason (AUDIT_ggml.md §5-C): the pinned bd4f514db1 enum has
    // neither op (they are legacy-revision leftovers), this crate never
    // dispatches them, but crates/llama's op-name mappings (tests/*_dump.rs,
    // tests/arch_batch*_e2e.rs, clip.rs) match on them exhaustively and those
    // files are owned by the porting agents, not this crate — removing the
    // variants would break their builds. No graph this crate builds ever
    // carries these tags.
    SquaredMulMat = 8,
    MulMat = 11,
    Scale = 14,
    Cpy = 15,
    Reshape = 19,
    View = 20,
    Permute = 21,
    Transpose = 22,
    GetRows = 23,
    DiagMaskInf = 26,
    SoftMax = 27,
    RoPE = 28,
    RoPEBack = 29,
    MulMatId = 30,
    Argsort = 33,
    ArgMax = 34,
    Repeat = 36,
    Concat = 42,
    Silu = 35,
    SumRows = 39,
    MulView = 53,
    /// GGML_OP_SET_ROWS — src[0]=src rows, src[1]=idx, src[2]=dst (C order is
    /// legacy-wired, see ggml.c:4000). Discriminant internal-only.
    SetRows = 54,
    /// GGML_OP_FLASH_ATTN_EXT — src = [q, k, v, mask]. Discriminant internal-only.
    FlashAttnExt = 55,
    /// GGML_OP_ADD_ID — src = [a, b, ids] (ggml.c:2150). Discriminant internal-only.
    AddId = 56,
    /// GGML_OP_GLU — src = [a, b?], op_params[0] = ggml_glu_op, [1] = swapped,
    /// [2]/[3] = alpha/limit f32 bits for SWIGLU_OAI (ggml.c:2802).
    /// Discriminant internal-only.
    Glu = 57,
    /// GGML_OP_SSM_CONV — src = [sx, conv1d.weight] (ggml.c:5659).
    /// Discriminant internal-only.
    SsmConv = 58,
    /// GGML_OP_SSM_SCAN — src = [s, x, dt, A, B, C, ids], op_params[0] = K
    /// (ggml.c:5729). Discriminant internal-only.
    SsmScan = 59,
    /// GGML_OP_CLAMP — src = [a], op_params[0]/[1] = min/max f32 bits
    /// (ggml.c:4110). Discriminant internal-only.
    Clamp = 60,
    /// GGML_OP_GATED_DELTA_NET — src = [q, k, v, g, beta, state],
    /// op_params[0] = K (snapshot slots; this port implements K == 1). The
    /// output packs `[S_v*H_v, n_tokens*n_seqs + K*S_v*n_seqs]` — the attention
    /// scores followed by the new state (ggml.c:6364, ops.cpp:10894).
    /// Discriminant internal-only.
    Gdn = 61,
    /// GGML_OP_IM2COL — src = [kernel, image], op_params = {s0,s1,p0,p1,d0,d1,
    /// is_2d} (ggml.c:4584). Vision patch-embedding conv only.
    /// Discriminant internal-only.
    Im2col = 62,
    /// GGML_OP_UPSCALE — the node `ggml_interpolate` builds (ggml.c:5199),
    /// src = [a], op_params[0] = ggml_scale_mode | ggml_scale_flag bits.
    /// Discriminant internal-only.
    Upscale = 63,
    /// GGML_OP_FILL — src = [a], op_params[0] = value f32 bits (ggml.c:5407).
    /// Used by the DSA top-k KQ-mask composition (llama-graph.cpp:3046/3058).
    /// Discriminant internal-only.
    Fill = 64,
    /// GGML_OP_LIGHTNING_INDEXER — src = [q, k, weights, mask], the DeepSeek
    /// DSA lightning indexer fused score (ggml.c:6423): per (token, kv)
    /// score = sum_h max(dot(q_h, k), 0) * w_h + mask. Discriminant
    /// internal-only.
    LightningIndexer = 65,
    /// GGML_OP_TOP_K — src = [a], dst I32 `[k, ne1, ne2, ne3]` (ggml.c:5459);
    /// `k` rides dst->ne[0]. The kernel is a faithful port of the reference's
    /// `std::partial_sort` heap-select (ops.cpp:8550-8604), whose *tie*
    /// behavior (deepseek32's ReLU-zeroed indexer scores tie massively)
    /// decides which kv rows the DSA mask unmasks. Discriminant internal-only.
    TopK = 66,
    /// GGML_OP_SQRT — src = [a], per-element `sqrtf` (unary-ops.cpp:285
    /// `op_sqrt`). Consumed by the deepseek4 SQRT_SOFTPLUS MoE gating
    /// (llama-graph.cpp:2053). Discriminant internal-only.
    Sqrt = 67,
    // arch batch 7 (2026-09-24), deepseek4: the fused hyper-connection ops.
    // ggml.h:585-587 GGML_OP_DSV4_HC_COMB/_PRE/_POST — the cparams defaults
    // (fused_dsv4_hc_* = true, llama-context.cpp:240-242) make these the
    // reference's default graph path.
    /// GGML_OP_DSV4_HC_COMB — src = [mixes, scale, base], op_params[0] = eps
    /// (f32 bits), [1] = n_iter (ggml.c:6459). Discriminant internal-only.
    Dsv4HcComb = 68,
    /// GGML_OP_DSV4_HC_PRE — src = [x, weights], op_params[0] = scale (f32
    /// bits), [1] = gated (ggml.c:6510 impl; only the non-gated
    /// `ggml_dsv4_hc_pre` builder is ported — that is what deepseek4 calls,
    /// deepseek4.cpp:301). Discriminant internal-only.
    Dsv4HcPre = 69,
    /// GGML_OP_DSV4_HC_POST — src = [x, residual, post, comb?] (ggml.c:6565;
    /// comb may be absent — the NULL branch keeps each stream's own residual).
    /// Discriminant internal-only.
    Dsv4HcPost = 70,
    /// GGML_OP_PAD — src = [a], op_params = {lp0, rp0, lp1, rp1, lp2, rp2,
    /// lp3, rp3, circular} (ggml.c:5166 ggml_pad_ext). F32 only; the padded
    /// region reads 0. graniteswitch's router lane pads the 1-wide signal to
    /// the head width (granite-switch.cpp:267-270). Discriminant internal-only.
    Pad = 71,
    /// GGML_OP_POOL_2D — src = [a], op_params = {op, k0, k1, s0, s1, p0, p1}
    /// (ggml.c:4928 ggml_pool_2d); the reference value is ggml.h:557 but the
    /// discriminant is internal-only. arch batch 11a (minimax-m3): the MSA
    /// block-score max-pool over the indexer scores.
    Pool2d = 72,
    /// GGML_OP_ARANGE — no src, op_params = {start, stop, step} f32 bits
    /// (ggml.c:5475). arch batch 11a (minimax-m3): the MSA decode-path
    /// position/cell index arithmetic.
    Arange = 73,
    /// GGML_OP_POOL_1D — src = [a], op_params = {op, k0, s0, p0}
    /// (ggml.c:5071 ggml_pool_1d; ggml.h:2281). The whisper-enc family's
    /// nn.AvgPool1d(2, stride=2) (clip.cpp:546-556). Discriminant
    /// internal-only (reference value ggml.h:556).
    Pool1d = 74,
    /// GGML_OP_ROLL — src = [a], op_params = {shift0..3} (ggml.c:5321).
    /// The conformer-family relative-shift / causal-pad trick. Discriminant
    /// internal-only (reference value ggml.h:563).
    Roll = 75,
    /// GGML_OP_CONV_2D — the *direct* single-node convolution
    /// (ggml.c:4947 ggml_conv_2d_direct; kernel ops.cpp:7084). The im2col
    /// composite `ggml_conv_2d` is a MulMat node chain, so this variant is
    /// only reached by the direct builder.
    ///
    /// DEAD VARIANT (AUDIT_ggml.md §5-C): the port's `conv_2d_direct` builder
    /// (ops.rs) composes im2col+mul_mat+permute and never emits this node
    /// (bit-equal to the reference per parity/conformops_ref.bin), and
    /// `forward_op` never dispatches it — but crates/llama's op-name mappings
    /// match on it (tests/clip.rs), so it stays declared. The discriminant is
    /// internal-only.
    Conv2dDirect = 76,
    /// GGML_OP_CONV_2D_DW — depthwise direct convolution (ggml.c:4907
    /// ggml_conv_2d_dw_direct; kernel ops.cpp:7524-7681). Discriminant
    /// internal-only.
    Conv2dDw = 77,
    // audio round 4: the parakeet/mimo/spkenc op surface.
    /// GGML_OP_SIN — src = [a], per-element `sinf` (ggml.c:2423; kernel
    /// unary-ops.cpp:289 through vec.h:878). parakeet's in-graph sinusoidal
    /// RPE (parakeet.cpp:104). Discriminant internal-only.
    Sin = 78,
    /// GGML_OP_COS — src = [a], per-element `cosf` (ggml.c:2449; kernel
    /// unary-ops.cpp:293). parakeet.cpp:105. Discriminant internal-only.
    Cos = 79,
    /// GGML_OP_SQR — src = [a], per-element `x*x` (ggml.c:2321; kernel
    /// unary-ops.cpp:281 through vec.h:859). mimo's RVQ norms
    /// (mimo-audio.cpp:93) and qwen3tts-spkenc's variance (:111).
    /// Discriminant internal-only.
    Sqr = 80,
    /// GGML_OP_MEAN — src = [a], row-wise mean over ne[0] (ggml.c:2521;
    /// kernel ops.cpp:1513 — vec_sum double accumulator / ne00, ith==0
    /// only). qwen3tts-spkenc's temporal means (:76). Discriminant
    /// internal-only.
    Mean = 81,
    /// GGML_OP_PAD_REFLECT_1D — src = [a], op_params = {p0, p1} (ggml.c:5290;
    /// kernel ops.cpp:8282 — reflect the first/last rows into the margins).
    /// qwen3tts-spkenc's "same"-padding convs (:17). Discriminant
    /// internal-only.
    PadReflect1d = 82,
    // audio round 5 (TTS generators, agent GEN5): the remaining op surface of
    // qwen3tts-gen.cpp / pockettts-gen.cpp. Discriminants are internal-only
    // (the reference values are the GGML_OP_* enum order, which this crate
    // does not reproduce — see SquaredMulMat above).
    /// GGML_OP_SUM — src = [a], scalar dst (a->type) via a ggml_float (double)
    /// row accumulator, ith==0 only (ops.cpp:1282). The on-graph sampling
    /// masks of qwen3tts's code_gen.
    Sum = 83,
    /// GGML_OP_CUMSUM — src = [a], exclusive-less prefix sum along dim 0
    /// (`y[i] = y[i-1] + x[i]`, vec.h:1507). qwen3tts's top-p / inverse-CDF
    /// sampling (qwen3tts-gen.cpp:32/:54).
    Cumsum = 84,
    /// GGML_OP_TRI — src = [a], op_params[0] = ggml_tri_type; keeps the
    /// predicate-passing half of each row, zeros the rest (ops.cpp:2289).
    /// qwen3tts's causal cache mask (qwen3tts-gen.cpp:109).
    Tri = 85,
    /// GGML_OP_LOG — src = [a], per-element `logf` (ggml.c:2375; kernel
    /// unary-ops.cpp:297 `unary_op<op_log>`). Its own op in the pinned enum
    /// (not part of GGML_OP_UNARY). The log-mask trick of the TTS graphs.
    Log = 86,
    /// GGML_OP_COL2IM_1D — src = [a], op_params = {s0, oc, p0}; scatter-add
    /// columns [K*OC, T_in] -> signal [(T_in-1)*s0 + K - 2*p0, OC]
    /// (ggml.c:4679; kernel ops.cpp:7017). The causal ConvTranspose1d of
    /// both generators (pockettts-seanet.cpp:75, qwen3tts-gen.cpp:348).
    Col2Im1d = 87,
    // arch batch 14 round 2 (2026-09, agent WKV): the three fused RWKV
    // kernels (ggml.h:579-581) — previously composed per token from
    // elementwise ops (the mul+add vs GGML_F32_VEC_FMA rounding gap was the
    // sole reason rwkv7-long / arwkv7 sat outside the batch-14 parity
    // defaults). The packed dst is {S*H, n_tokens + S*n_seqs}: output rows
    // first, then the new state per sequence.
    /// GGML_OP_RWKV_WKV6 — src = [k, v, r, tf, td, state] (ggml.c:5873;
    /// kernel ops.cpp:10413-10603, ported in wkv.rs). Discriminant
    /// internal-only.
    RwkvWkv6 = 88,
    /// GGML_OP_GATED_LINEAR_ATTN — src = [k, v, q, g, state], op_params[0] =
    /// scale f32 bits (ggml.c:5916; kernel ops.cpp:10623-11418, wkv.rs).
    /// Discriminant internal-only.
    GatedLinearAttn = 89,
    /// GGML_OP_RWKV_WKV7 — src = [r, w, k, v, a, b, state] (ggml.c:5959;
    /// kernel ops.cpp:11422-11617, wkv.rs — mind the head_size < 64 window
    /// overrun semantics documented there). Discriminant internal-only.
    RwkvWkv7 = 90,
    // arch batch 18 (2026-10, agent GDN): the two ops the chunked delta-net
    // half needs (delta-net-base.cpp:17-287) plus the two tiny ones its chunk
    // loop and decay arithmetic lean on. The reference CPU backend ships all
    // of them (ops.cpp / ggml-cpu.c) but the reference *model* graph never
    // reaches the chunked path — fused_gdn_ch is hard-true (llama-context.cpp:234),
    // so the acceptance is op/chain-level probes, not a model e2e.
    /// GGML_OP_SET — src = [a, b], op_params = {nb1, nb2, nb3, offset,
    /// inplace} as i32 (ggml.c:3505 ggml_set_impl). dst is a view of `a`
    /// (inplace) whose b-shaped region at `offset` is overwritten with b's
    /// rows, viewed through the param strides. The chunked delta-net's chunk
    /// loop writes each chunk's output slice back into the padded output
    /// tensor with it (delta-net-base.cpp:262). Discriminant internal-only.
    Set = 91,
    /// GGML_OP_DIAG — src = [a] (ne1 == 1, F32); dst [n, n, ne2, ne3] carries
    /// a's row i on dst row i's diagonal, zeros elsewhere (ggml.c:4027;
    /// kernel ops.cpp:5434-5472, n_tasks = 1). The chunked delta-net's I+attn
    /// identity term (delta-net-base.cpp:163-165). Discriminant internal-only.
    Diag = 92,
    /// GGML_OP_SOLVE_TRI — src = [A, B] F32; dst = A⁻¹B by forward
    /// substitution, A square lower-triangular (ggml.c:6329 asserts
    /// left && lower && !uni — no other variant exists at this commit;
    /// kernel ops.cpp:10824-10880, n_tasks = n_threads split over
    /// ne02*ne03*ne10 solve columns). The chunked delta-net's UT transform
    /// (delta-net-base.cpp:169-171). Discriminant internal-only.
    SolveTri = 93,
}

/// Where a tensor's bytes live.
#[derive(Clone)]
pub enum Storage {
    /// offset into Context::arena
    Arena { offset: usize, len: usize },
    /// external immutable buffer (mmap weights) + offset
    External { offset: usize, len: usize },
    /// no backing store yet
    None,
}

pub struct TensorMeta {
    pub ty: GgmlType,
    /// number of elements per dim; ne[0] is row length
    pub ne: [i64; MAX_DIMS],
    /// byte strides
    pub nb: [u64; MAX_DIMS],
    pub op: GgmlOp,
    pub op_params: [i32; MAX_OP_PARAMS / 4],
    pub src: [Option<TensorId>; MAX_SRC],
    pub view_src: Option<TensorId>,
    pub view_offs: usize,
    pub storage: Storage,
    pub name: String,
    pub flags: u32,
}

impl TensorMeta {
    pub fn n_elements(&self) -> i64 {
        self.ne.iter().product()
    }
    pub fn is_scalar(&self) -> bool {
        self.ne.iter().all(|&n| n == 1)
    }
    pub fn is_matrix(&self) -> bool {
        self.ne[2] == 1 && self.ne[3] == 1
    }
    pub fn is_vector(&self) -> bool {
        self.ne[1] == 1 && self.ne[2] == 1 && self.ne[3] == 1
    }
    pub fn nrows(&self) -> i64 {
        self.ne[1] * self.ne[2] * self.ne[3]
    }
    fn type_aware_strides(ne: &[i64; MAX_DIMS], ty: GgmlType) -> [u64; MAX_DIMS] {
        let nb0 = ty.type_size() as u64;
        let blck = ty.blck_size() as i64;
        let nb1 = nb0 * (ne[0] / blck) as u64;
        let nb2 = nb1 * ne[1] as u64;
        let nb3 = nb2 * ne[2] as u64;
        [nb0, nb1, nb2, nb3]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u32);

pub struct Context {
    /// Process-unique instance id (monotonic). backend.rs's process-wide
    /// tensor↔buffer side table keys its entries by the Context *identity*;
    /// keying by the raw address instead aliases a fresh Context placed at a
    /// dead Context's stack/heap slot with the dead one's stale entries (the
    /// gallocr then treats its tensors as externally allocated and
    /// `init_tensor` indexes buffer_id −1 — pinned by
    /// `compute::tests::perf8_repro`).
    pub uid: u64,
    pub tensors: Vec<TensorMeta>,
    /// arena for intermediate results
    pub arena: Vec<u8>,
    /// Bytes at the front of `arena`'s current allocation that are initialized
    /// (≥ `arena.len()` only in the sense that `reset_graph_to` may truncate the
    /// `Vec` while the tail bytes stay initialized in the retained capacity).
    /// The gallocr this mirrors never zeroes its buffer (ggml-alloc.c reserves
    /// once and hands out raw pointers), so arena growth must not re-zero
    /// already-initialized bytes — see `arena_resize_tensor`.
    pub arena_zeroed: usize,
    /// external buffers registered by the model loader; Storage::External indexes these
    pub external: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Monotonic Context instance counter (see `Context::uid`).
static CTX_UID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Context {
    pub fn new() -> Self {
        Context {
            uid: CTX_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            tensors: Vec::new(),
            arena: Vec::new(),
            arena_zeroed: 0,
            external: Vec::new(),
        }
    }

    /// The TensorId of context tensor slot `i` (0..tensors.len) — additive
    /// accessor for the foreign-backend executor, which walks every slot.
    pub fn id_at(&self, i: usize) -> TensorId {
        TensorId(i as u32)
    }

    pub fn new_tensor(&mut self, ty: GgmlType, ne: [i64; MAX_DIMS]) -> TensorId {
        assert!(ne.iter().all(|&n| n >= 0), "negative ne");
        // ggml.c:1783-1790 (sync batch D): validate the element count fits
        // int64_t before the row-size math multiplies it out
        if cfg!(debug_assertions) {
            let mut cur = ne[0];
            for &n in ne.iter().skip(1) {
                if n > 1 {
                    assert!(
                        i64::MAX / n > cur,
                        "tensor element count overflows int64_t"
                    );
                    cur *= n;
                }
            }
        }
        if ty.is_quantized() {
            assert_eq!(ne[0] % ty.blck_size() as i64, 0, "ne[0] not multiple of blck_size");
        }
        let nb = TensorMeta::type_aware_strides(&ne, ty);
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorMeta {
            ty,
            ne,
            nb,
            op: GgmlOp::None,
            op_params: [0; MAX_OP_PARAMS / 4],
            src: [None; crate::types::MAX_SRC],
            view_src: None,
            view_offs: 0,
            storage: Storage::None,
            name: String::new(),
            flags: 0,
        });
        id
    }

    /// `ggml_set_input` (ggml.c:8004) — sync batch D adds
    /// `GGML_ASSERT(tensor->op == GGML_OP_NONE)`; the flag itself is what the
    /// graph builder and sched key on. The C-FFI graph path asserts on the C
    /// side (backend_emit.rs calls the real ggml_set_input).
    pub fn set_input(&mut self, t: TensorId) {
        assert_eq!(
            self.tensors[t.0 as usize].op,
            GgmlOp::None,
            "ggml_set_input: tensor has an op"
        );
        self.tensors[t.0 as usize].flags |= crate::backend::GGML_TENSOR_FLAG_INPUT;
    }

    pub fn new_tensor_1d(&mut self, ty: GgmlType, ne0: i64) -> TensorId {
        self.new_tensor(ty, [ne0, 1, 1, 1])
    }
    pub fn new_tensor_2d(&mut self, ty: GgmlType, ne0: i64, ne1: i64) -> TensorId {
        self.new_tensor(ty, [ne0, ne1, 1, 1])
    }
    pub fn new_tensor_3d(&mut self, ty: GgmlType, ne0: i64, ne1: i64, ne2: i64) -> TensorId {
        self.new_tensor(ty, [ne0, ne1, ne2, 1])
    }
    pub fn new_tensor_4d(&mut self, ty: GgmlType, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> TensorId {
        self.new_tensor(ty, [ne0, ne1, ne2, ne3])
    }

    pub fn new_f32(&mut self, value: f32) -> TensorId {
        let id = self.new_tensor_1d(GgmlType::F32, 1);
        self.arena_resize_tensor(id);
        self.with_f32_mut(id, |p| p[0] = value);
        id
    }

    pub fn new_i32(&mut self, value: i32) -> TensorId {
        let id = self.new_tensor_1d(GgmlType::I32, 1);
        self.arena_resize_tensor(id);
        self.with_i32_mut(id, |p| p[0] = value);
        id
    }

    /// Point a tensor at external (mmap) storage.
    pub fn set_external_storage(&mut self, id: TensorId, ext: std::sync::Arc<dyn std::any::Any + Send + Sync>, offset: usize) {
        let len = self.nbytes(id);
        let idx = self.external.len();
        self.external.push(ext);
        let t = &mut self.tensors[id.0 as usize];
        t.storage = Storage::External { offset: offset + idx * 0, len };
        // encode buffer index in the high bits of offset to avoid a separate field
        // (offsets are < 2^48 in practice; buffers < 2^16)
        if let Storage::External { offset, .. } = &mut t.storage {
            *offset = (*offset as u64 | ((idx as u64) << 48)) as usize;
        }
    }

    fn ext_index_and_offset(&self, storage: &Storage) -> (usize, usize) {
        match storage {
            Storage::External { offset, .. } => (offset >> 48, offset & ((1 << 48) - 1)),
            _ => unreachable!(),
        }
    }

    pub fn nbytes(&self, id: TensorId) -> usize {
        let t = &self.tensors[id.0 as usize];
        (t.n_elements() as u64 / t.ty.blck_size() as u64 * t.ty.type_size() as u64) as usize
    }

    /// Allocate arena storage for a tensor (idempotent).
    ///
    /// Growth zeroes only bytes this allocation has never initialized
    /// (`arena_zeroed`), and `reset_graph_to` keeps the capacity, so a
    /// steady-state forward grows the `Vec` back to its high-water mark with
    /// two length stores instead of a full re-zero — the cost profile the C
    /// gallocr has (ggml-alloc.c:238-262 reserves the buffer once at the
    /// max-graph size and then only assigns offsets). Every byte at or below
    /// `arena.len()` is still guaranteed initialized, so handing out
    /// `&[u8]`/`&mut [u8]` views stays sound; bytes beyond the first graph's
    /// zeroing hold the previous graph's values, exactly like the C's reused
    /// gallocr buffer (every graph tensor is fully written by its op before
    /// any consumer reads it — the dataflow invariant both engines rely on).
    pub fn arena_resize_tensor(&mut self, id: TensorId) {
        let len = self.nbytes(id);
        let t = &mut self.tensors[id.0 as usize];
        if matches!(t.storage, Storage::None) {
            // keep 64-byte alignment per tensor
            let aligned = (self.arena.len() + 63) & !63;
            let new_len = aligned + len;
            if new_len > self.arena.len() {
                // Lift the Vec's len to the initialized watermark first: a
                // realloc inside `reserve` copies only `[0, len)`, so a
                // still-initialized tail `[len, arena_zeroed)` (left over from
                // a `reset_graph_to` truncate) must be inside the copied span
                // or it would be lost to the fresh allocation. The lift never
                // passes `new_len` — `reserve`'s `additional` argument must
                // stay positive.
                //
                // SAFETY (both set_len calls): bytes `[len, lift)` are
                // initialized in this allocation and `capacity >= arena_zeroed
                // >= lift` (invariants of this pair of fields: `arena_zeroed`
                // only grows to a length that was reserved first, and
                // `truncate` keeps capacity). The final `set_len(new_len)` is
                // covered by the reserve + zero below.
                let lift = self.arena_zeroed.min(new_len);
                if lift > self.arena.len() {
                    unsafe { self.arena.set_len(lift) };
                }
                if new_len > self.arena_zeroed {
                    self.arena.reserve(new_len - self.arena_zeroed);
                    // The large-size allocations backing a grown arena are
                    // anonymous mmaps (glibc realloc → mremap above the mmap
                    // threshold); asking the kernel for transparent huge pages
                    // there turns the first forward's ~50k 4 KiB faults into
                    // ~hundreds of 2 MiB faults. The advice is a no-op on
                    // non-anonymous VMAs and ignored where THP is off.
                    unsafe {
                        arena_advise_hugepage(self.arena.as_mut_ptr(), self.arena.capacity());
                    }
                    // SAFETY: `reserve` made room for `new_len` bytes, and
                    // `[0, arena_zeroed)` is initialized; the freshly zeroed
                    // span `[arena_zeroed, new_len)` completes initialization.
                    unsafe {
                        std::ptr::write_bytes(
                            self.arena.as_mut_ptr().add(self.arena_zeroed),
                            0,
                            new_len - self.arena_zeroed,
                        );
                    }
                    self.arena_zeroed = new_len;
                }
                unsafe { self.arena.set_len(new_len) };
            }
            t.storage = Storage::Arena { offset: aligned, len };
        }
    }

    /// Assign arena storage at a *planned* offset (perf9's gallocr-style
    /// reuse): the offset comes from the graph-wide allocator in
    /// compute.rs (`plan_arena_reuse`), which only hands out space whose
    /// previous owner's last consumer already ran — the same lifetime
    /// discipline as ggml-alloc.c's dyn_tallocr. Growth/zeroing follows the
    /// `arena_zeroed` watermark exactly like [`arena_resize_tensor`].
    pub fn arena_assign_tensor(&mut self, id: TensorId, offset: usize) {
        let len = self.nbytes(id);
        let t = &mut self.tensors[id.0 as usize];
        debug_assert!(matches!(t.storage, Storage::None));
        let new_len = offset + len;
        if new_len > self.arena.len() {
            // same lift/reserve/zero sequence as arena_resize_tensor, for the
            // same watermark reasons (see that fn's SAFETY notes)
            let lift = self.arena_zeroed.min(new_len);
            if lift > self.arena.len() {
                unsafe { self.arena.set_len(lift) };
            }
            if new_len > self.arena_zeroed {
                self.arena.reserve(new_len - self.arena_zeroed);
                unsafe {
                    arena_advise_hugepage(self.arena.as_mut_ptr(), self.arena.capacity());
                    std::ptr::write_bytes(
                        self.arena.as_mut_ptr().add(self.arena_zeroed),
                        0,
                        new_len - self.arena_zeroed,
                    );
                }
                self.arena_zeroed = new_len;
            }
            unsafe { self.arena.set_len(new_len) };
        }
        t.storage = Storage::Arena { offset, len };
    }

    /// `ggml_set_output` (ggml.h tensor flags) — perf9's reuse planner never
    /// frees or reuses an output tensor's storage, mirroring
    /// ggml-alloc.c:646/693. Additive; nothing else reads the flag on the CPU
    /// path.
    pub fn set_output(&mut self, id: TensorId) {
        self.tensors[id.0 as usize].flags |= crate::backend::GGML_TENSOR_FLAG_OUTPUT;
    }

    /// Mark the current tensor count — graph tensors created after this can be
    /// dropped by `reset_graph_to` (llama.cpp rebuilds its compute graph each
    /// ubatch; weights/cache/input tensors stay alive across steps).
    pub fn mark(&self) -> usize {
        self.tensors.len()
    }

    /// Drop graph tensors above `watermark` and reclaim arena tail storage.
    /// Storage referenced by surviving tensors is preserved.
    pub fn reset_graph_to(&mut self, watermark: usize) {
        assert!(watermark <= self.tensors.len());
        self.tensors.truncate(watermark);
        let mut live_end = 0usize;
        for t in &self.tensors {
            if let Storage::Arena { offset, len } = &t.storage {
                live_end = live_end.max(offset + len);
            }
        }
        // follow view chains: views resolve to base storage, already covered
        // The tail capacity is deliberately retained (no shrink_to_fit): the
        // next forward regrows to a similar high-water mark, and shrinking
        // would both copy the live prefix and re-fault the tail pages every
        // step. This mirrors the C gallocr, whose buffer stays at the max
        // graph's reservation across steps (ggml-alloc.c:238-262).
        self.arena.truncate(live_end);
    }

    pub fn set_name(&mut self, id: TensorId, name: &str) {
        self.tensors[id.0 as usize].name = name.to_string();
    }
    pub fn name(&self, id: TensorId) -> &str {
        &self.tensors[id.0 as usize].name
    }

    pub fn ne(&self, id: TensorId) -> &[i64; MAX_DIMS] {
        &self.tensors[id.0 as usize].ne
    }
    pub fn nb(&self, id: TensorId) -> &[u64; MAX_DIMS] {
        &self.tensors[id.0 as usize].nb
    }
    pub fn ty(&self, id: TensorId) -> GgmlType {
        self.tensors[id.0 as usize].ty
    }
    pub fn op(&self, id: TensorId) -> GgmlOp {
        self.tensors[id.0 as usize].op
    }
    pub fn src(&self, id: TensorId) -> [Option<TensorId>; MAX_SRC] {
        self.tensors[id.0 as usize].src
    }
    pub fn op_params(&self, id: TensorId) -> &[i32; MAX_OP_PARAMS / 4] {
        &self.tensors[id.0 as usize].op_params
    }
    pub fn set_op_params_i32(&mut self, id: TensorId, params: &[i32]) {
        let n = params.len().min(MAX_OP_PARAMS / 4);
        self.tensors[id.0 as usize].op_params[..n].copy_from_slice(&params[..n]);
    }

    // ---- data accessors (arena only; views through view_src chain) ----

    fn resolve(&self, id: TensorId) -> (TensorId, usize) {
        // follow view chain to base tensor and add offsets
        let mut base = id;
        let mut off = 0usize;
        loop {
            let t = &self.tensors[base.0 as usize];
            match t.view_src {
                Some(v) => {
                    off += t.view_offs;
                    base = v;
                }
                None => break,
            }
        }
        (base, off)
    }

    pub fn data_bytes(&self, id: TensorId) -> Option<&[u8]> {
        let (base, off) = self.resolve(id);
        let t = &self.tensors[base.0 as usize];
        match &t.storage {
            Storage::Arena { offset, len } => Some(&self.arena[offset + off..offset + off + len]),
            Storage::External { offset, len } => {
                let (idx, off2) = self.ext_index_and_offset(&Storage::External { offset: *offset, len: *len });
                let buf = self.external.get(idx)?;
                let bytes: &[u8] = buf
                    .downcast_ref::<Vec<u8>>()
                    .map(|v| v.as_slice())
                    .or_else(|| buf.downcast_ref::<memmap2::Mmap>().map(|m| m.as_ref()))?;
                Some(&bytes[off2 + off..off2 + off + len])
            }
            Storage::None => None,
        }
    }

    pub fn data_bytes_mut(&mut self, id: TensorId) -> Option<&mut [u8]> {
        let (base, off) = self.resolve(id);
        let t_storage = match &self.tensors[base.0 as usize].storage {
            Storage::Arena { offset, len } => (*offset, *len),
            _ => return None, // external is read-only
        };
        Some(&mut self.arena[t_storage.0 + off..t_storage.0 + off + t_storage.1])
    }

    pub fn f32s(&self, id: TensorId) -> Option<&[f32]> {
        bytemuck::cast_slice(self.data_bytes(id)?).into()
    }
    pub fn f32s_mut(&mut self, id: TensorId) -> Option<&mut [f32]> {
        let bytes = self.data_bytes_mut(id)?;
        Some(bytemuck::cast_slice_mut(bytes))
    }
    pub fn with_f32_mut<R>(&mut self, id: TensorId, f: impl FnOnce(&mut [f32]) -> R) -> Option<R> {
        let bytes = self.data_bytes_mut(id)?;
        let slice: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        Some(f(slice))
    }
    pub fn with_i32_mut<R>(&mut self, id: TensorId, f: impl FnOnce(&mut [i32]) -> R) -> Option<R> {
        let bytes = self.data_bytes_mut(id)?;
        let slice: &mut [i32] = bytemuck::cast_slice_mut(bytes);
        Some(f(slice))
    }
    pub fn i32s(&self, id: TensorId) -> Option<&[i32]> {
        Some(bytemuck::cast_slice(self.data_bytes(id)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_basics() {
        let mut ctx = Context::new();
        let t = ctx.new_tensor_2d(GgmlType::F32, 64, 8);
        assert_eq!(ctx.nbytes(t), 64 * 8 * 4);
        assert_eq!(ctx.nb(t)[1], 64 * 4);
        ctx.arena_resize_tensor(t);
        ctx.with_f32_mut(t, |p| p.fill(1.5)).unwrap();
        assert_eq!(ctx.f32s(t).unwrap()[100], 1.5);

        let q = ctx.new_tensor_2d(GgmlType::Q4_0, 32, 4);
        assert_eq!(ctx.nbytes(q), 4 * 18);
    }
}
