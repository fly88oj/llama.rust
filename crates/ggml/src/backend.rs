//! backend — port of ggml's backend abstraction layer (task ① of the GPU
//! enablement plan).
//!
//! Reference (llama.cpp pinned @ bd4f514db1):
//!   ggml/include/ggml-backend.h      — public API surface
//!   ggml/src/ggml-backend.cpp        — buffer type/buffer/backend/device
//!                                      plumbing, multi-buffer, events, utils
//!   ggml/src/ggml-backend-reg.cpp    — the registry + dynamic loading
//!   ggml/src/ggml-cpu/ggml-cpu.cpp   — the CPU backend adapter this file
//!                                      registers as the first backend
//!   ggml/src/ggml-backend-impl.h     — the vtable struct definitions
//!
//! The port keeps the C shape one-to-one: every iface struct below is a struct
//! of optional plain function pointers (not trait objects) so a dlopen'ed
//! backend (task ③) can be driven through the same vtables, and every function
//! carries a `ref file:line` comment pointing at the C original.
//!
//! The deliberate adaptations forced by the port's tensor model (tensors are
//! `TensorId`s into a `Context`, data lives in `Context::arena`/external
//! buffers — see tensor.rs) are:
//!   * every tensor-touching entry point threads `&mut Context` (C reads
//!     `tensor->data` through raw pointers instead);
//!   * "tensor->buffer" (a pointer field in C) becomes a process-wide side
//!     table keyed by (Context address, TensorId) — `tensor_buffer()` follows
//!     view chains exactly like C's `tensor->view_src ? … : tensor->buffer`;
//!   * buffer "base addresses" are arena offsets (`usize`), not pointers; the
//!     CPU buffer type's `alloc_buffer` claims a tail span of the Context
//!     arena where C does `ggml_aligned_malloc` (ggml-backend.cpp:2447);
//!   * object identity (C pointer equality, e.g. `sched->bufts[b] == buft`)
//!     is `Arc::ptr_eq` on the shared handles.

use crate::graph::Graph;
use crate::tensor::{Context, GgmlOp, Storage, TensorId};
use crate::types::GgmlType;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, OnceLock};

use crate::sysffi;

// ggml-backend-impl.h:11
pub const GGML_BACKEND_API_VERSION: i32 = 2;

// ggml-impl.h:44
pub const TENSOR_ALIGNMENT: usize = 32;

// ggml.h:663-667 (subset the scheduler uses)
pub const GGML_TENSOR_FLAG_INPUT: u32 = 1;
pub const GGML_TENSOR_FLAG_OUTPUT: u32 = 2;

// ggml.h:655-661 enum ggml_status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlStatus {
    Success = 0,
    AllocFailed = 1,
    OutOfMemory = 2,
    BackendError = 3,
    Aborted = 4,
}

// ===========================================================================
// Backend buffer type — ggml-backend-impl.h:17-35
// ===========================================================================

/// `struct ggml_backend_buffer_type_i` (ggml-backend-impl.h:17-29).
pub struct BackendBufferTypeI {
    pub get_name: fn(&BackendBufferType) -> String,
    /// allocate a buffer of this type (the port threads `&mut Context` — the
    /// CPU buffer type claims an arena span where C mallocs)
    pub alloc_buffer: fn(&BackendBufferTypeRef, &mut Context, usize) -> Option<BackendBufferRef>,
    /// tensor alignment
    pub get_alignment: fn(&BackendBufferType) -> usize,
    /// (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
    pub get_max_size: Option<fn(&BackendBufferType) -> usize>,
    /// (optional) data size needed to allocate the tensor, including padding
    /// (defaults to ggml_nbytes)
    pub get_alloc_size: Option<fn(&BackendBufferType, &Context, TensorId) -> usize>,
    /// (optional) check if tensor data is in host memory and uses standard
    /// ggml tensor layout (defaults to false)
    pub is_host: Option<fn(&BackendBufferType) -> bool>,
}

/// `struct ggml_backend_buffer_type` (ggml-backend-impl.h:31-35).
pub struct BackendBufferType {
    pub iface: BackendBufferTypeI,
    pub device: Option<BackendDeviceRef>,
    /// C `void *context`
    pub context: Arc<dyn std::any::Any + Send + Sync>,
}

pub type BackendBufferTypeRef = Arc<BackendBufferType>;

/// C pointer identity of buffer types (`buft_a == buft_b`).
pub fn buft_eq(a: &BackendBufferTypeRef, b: &BackendBufferTypeRef) -> bool {
    Arc::ptr_eq(a, b)
}

/// `ggml_backend_buft_name` (ggml-backend.cpp:34)
pub fn backend_buft_name(buft: &BackendBufferTypeRef) -> String {
    (buft.iface.get_name)(buft)
}

/// `ggml_backend_buft_alloc_buffer` (ggml-backend.cpp:39) — a zero-sized
/// request returns a dummy buffer.
pub fn backend_buft_alloc_buffer(
    buft: &BackendBufferTypeRef,
    ctx: &mut Context,
    size: usize,
) -> Option<BackendBufferRef> {
    if size == 0 {
        // return a dummy buffer for zero-sized allocations
        return Some(backend_buffer_init(buft.clone(), BackendBufferI::default(), BufferContext::None, 0));
    }
    (buft.iface.alloc_buffer)(buft, ctx, size)
}

/// `ggml_backend_buft_get_alignment` (ggml-backend.cpp:48)
pub fn backend_buft_get_alignment(buft: &BackendBufferTypeRef) -> usize {
    (buft.iface.get_alignment)(buft)
}

/// `ggml_backend_buft_get_max_size` (ggml-backend.cpp:53) — optional,
/// defaults to SIZE_MAX.
pub fn backend_buft_get_max_size(buft: &BackendBufferTypeRef) -> usize {
    // get_max_size is optional, defaults to SIZE_MAX
    if let Some(f) = buft.iface.get_max_size {
        return f(buft);
    }
    usize::MAX
}

/// `ggml_backend_buft_get_alloc_size` (ggml-backend.cpp:62) — optional,
/// defaults to ggml_nbytes (= `Context::nbytes`).
pub fn backend_buft_get_alloc_size(buft: &BackendBufferTypeRef, ctx: &Context, tensor: TensorId) -> usize {
    // get_alloc_size is optional, defaults to ggml_nbytes
    if let Some(f) = buft.iface.get_alloc_size {
        let size = f(buft, ctx, tensor);
        assert!(size >= ctx.nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND] (ggml-backend.cpp:69-74)
        assert!(
            size <= ctx.nbytes(tensor)
                || op_is_empty(ctx.op(tensor))
                || ctx.ty(tensor).is_quantized()
                || op_alloc_size_may_expand(ctx.op(tensor))
        );
        return size;
    }
    ctx.nbytes(tensor)
}

/// `ggml_backend_buft_is_host` (ggml-backend.cpp:81) — optional, defaults false.
pub fn backend_buft_is_host(buft: &BackendBufferTypeRef) -> bool {
    if let Some(f) = buft.iface.is_host {
        return f(buft);
    }
    false
}

/// `ggml_backend_buft_get_device` (ggml-backend.cpp:89)
pub fn backend_buft_get_device(buft: &BackendBufferTypeRef) -> Option<BackendDeviceRef> {
    buft.device.clone()
}

// ===========================================================================
// Backend buffer — ggml-backend-impl.h:46-75 / ggml-backend.h:49-53
// ===========================================================================

/// `enum ggml_backend_buffer_usage` (ggml-backend.h:49-53)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendBufferUsage {
    Any = 0,
    Weights = 1,
    Compute = 2,
}

/// `struct ggml_backend_buffer_i` (ggml-backend-impl.h:46-67). The 2d and
/// async variants are `Option`s exactly as in C (NULL = unsupported).
pub struct BackendBufferI {
    /// (optional) free the buffer
    pub free_buffer: Option<fn(&BackendBufferRef)>,
    /// base address of the buffer — in the port, the arena offset (usize)
    pub get_base: Option<fn(&BackendBufferRef) -> usize>,
    /// (optional) initialize a tensor in the buffer (eg. add tensor extras)
    pub init_tensor: Option<fn(&BackendBufferRef, &mut Context, TensorId) -> GgmlStatus>,
    /// tensor data access
    pub memset_tensor: Option<fn(&BackendBufferRef, &mut Context, TensorId, u8, usize, usize)>,
    pub set_tensor: fn(&BackendBufferRef, &mut Context, TensorId, &[u8], usize),
    pub get_tensor: fn(&BackendBufferRef, &Context, TensorId, &mut [u8], usize),
    /// (optional) 2d data copies
    pub set_tensor_2d: Option<fn(&BackendBufferRef, &mut Context, TensorId, &[u8], usize, usize, usize, usize, usize)>,
    pub get_tensor_2d: Option<fn(&BackendBufferRef, &Context, TensorId, &mut [u8], usize, usize, usize, usize, usize)>,
    /// (optional) tensor copy: dst is in the buffer, src may be in any buffer
    /// (return false if not supported)
    pub cpy_tensor: Option<fn(&BackendBufferRef, &mut Context, TensorId, TensorId) -> bool>,
    /// clear the entire buffer
    pub clear: Option<fn(&BackendBufferRef, &mut Context, u8)>,
    /// (optional) reset any internal state due to tensor initialization
    pub reset: Option<fn(&BackendBufferRef, &mut Context)>,
}

impl Default for BackendBufferI {
    fn default() -> Self {
        // unreachable placeholders for the two required slots — every real
        // buffer type fills them (mirrors the C zeroed vtable being invalid)
        fn no_set(_b: &BackendBufferRef, _c: &mut Context, _t: TensorId, _d: &[u8], _o: usize) {
            panic!("backend buffer has no set_tensor")
        }
        fn no_get(_b: &BackendBufferRef, _c: &Context, _t: TensorId, _d: &mut [u8], _o: usize) {
            panic!("backend buffer has no get_tensor")
        }
        BackendBufferI {
            free_buffer: None,
            get_base: None,
            init_tensor: None,
            memset_tensor: None,
            set_tensor: no_set,
            get_tensor: no_get,
            set_tensor_2d: None,
            get_tensor_2d: None,
            cpy_tensor: None,
            clear: None,
            reset: None,
        }
    }
}

/// The C `void *context` of a buffer — what the iface fns downcast to read.
#[derive(Clone)]
pub enum BufferContext {
    None,
    /// CPU host buffer backed by a `[base, base+size)` span of the Context
    /// arena (the port's `ggml_aligned_malloc`)
    ArenaSpan { base: usize },
    /// `ggml_backend_cpu_buffer_from_ptr` — external (mmap) storage
    ExternalHost { ext_index: usize, offset: usize },
    /// multi-buffer (ggml-backend.cpp:680)
    Multi { buffers: Vec<BackendBufferRef> },
    /// foreign (DL) buffer, task ③
    Foreign,
}

/// `struct ggml_backend_buffer` (ggml-backend-impl.h:69-75).
pub struct BackendBuffer {
    pub iface: BackendBufferI,
    pub buft: BackendBufferTypeRef,
    pub context: BufferContext,
    pub size: usize,
    /// C `enum ggml_backend_buffer_usage usage` — mutated through the shared
    /// handle (`ggml_backend_buffer_set_usage`), hence the lock
    pub usage: Mutex<BackendBufferUsage>,
}

pub type BackendBufferRef = Arc<BackendBuffer>;

/// `ggml_backend_buffer_init` (ggml-backend.cpp:96)
pub fn backend_buffer_init(
    buft: BackendBufferTypeRef,
    iface: BackendBufferI,
    context: BufferContext,
    size: usize,
) -> BackendBufferRef {
    Arc::new(BackendBuffer {
        iface,
        buft,
        context,
        size,
        usage: Mutex::new(BackendBufferUsage::Any), // GGML_BACKEND_BUFFER_USAGE_ANY
    })
}

/// `ggml_backend_buffer_name` (ggml-backend.cpp:112)
pub fn backend_buffer_name(buffer: &BackendBufferRef) -> String {
    backend_buft_name(&backend_buffer_get_type(buffer))
}

/// `ggml_backend_buffer_free` (ggml-backend.cpp:116) — explicit free like C;
/// dropping the last handle frees the object.
pub fn backend_buffer_free(buffer: Option<BackendBufferRef>) {
    let Some(buffer) = buffer else { return };
    if let Some(free_buffer) = buffer.iface.free_buffer {
        free_buffer(&buffer);
    }
}

/// `ggml_backend_buffer_get_size` (ggml-backend.cpp:127)
pub fn backend_buffer_get_size(buffer: &BackendBufferRef) -> usize {
    buffer.size
}

/// `ggml_backend_buffer_get_base` (ggml-backend.cpp:132) — the port's "base"
/// is an arena offset; zero-sized buffers report no base.
pub fn backend_buffer_get_base(buffer: &BackendBufferRef) -> Option<usize> {
    // get_base is optional if the buffer is zero-sized
    if !backend_buffer_is_multi_buffer(buffer) && buffer.size == 0 {
        return None;
    }
    buffer.iface.get_base.map(|f| f(buffer))
}

/// `ggml_backend_buffer_init_tensor` (ggml-backend.cpp:152) — optional.
pub fn backend_buffer_init_tensor(
    buffer: &BackendBufferRef,
    ctx: &mut Context,
    tensor: TensorId,
) -> GgmlStatus {
    // init_tensor is optional
    if let Some(f) = buffer.iface.init_tensor {
        return f(buffer, ctx, tensor);
    }
    GgmlStatus::Success
}

/// `ggml_backend_buffer_clear` (ggml-backend.cpp:161) — optional if the
/// buffer is zero-sized.
pub fn backend_buffer_clear(buffer: &BackendBufferRef, ctx: &mut Context, value: u8) {
    // clear is optional if the buffer is zero-sized
    if buffer.size == 0 {
        return;
    }
    if let Some(f) = buffer.iface.clear {
        f(buffer, ctx, value);
    }
}

/// `ggml_backend_buffer_get_alignment` (ggml-backend.cpp:171)
pub fn backend_buffer_get_alignment(buffer: &BackendBufferRef) -> usize {
    backend_buft_get_alignment(&backend_buffer_get_type(buffer))
}

/// `ggml_backend_buffer_get_max_size` (ggml-backend.cpp:175)
pub fn backend_buffer_get_max_size(buffer: &BackendBufferRef) -> usize {
    backend_buft_get_max_size(&backend_buffer_get_type(buffer))
}

/// `ggml_backend_buffer_get_alloc_size` (ggml-backend.cpp:179)
pub fn backend_buffer_get_alloc_size(buffer: &BackendBufferRef, ctx: &Context, tensor: TensorId) -> usize {
    backend_buft_get_alloc_size(&backend_buffer_get_type(buffer), ctx, tensor)
}

/// `ggml_backend_buffer_is_host` (ggml-backend.cpp:183)
pub fn backend_buffer_is_host(buffer: &BackendBufferRef) -> bool {
    backend_buft_is_host(&backend_buffer_get_type(buffer))
}

/// `ggml_backend_buffer_set_usage` (ggml-backend.cpp:187)
pub fn backend_buffer_set_usage(buffer: &BackendBufferRef, usage: BackendBufferUsage) {
    *buffer.usage.lock().unwrap() = usage;

    // FIXME: add a generic callback to the buffer interface (ggml-backend.cpp:191)
    if backend_buffer_is_multi_buffer(buffer) {
        backend_multi_buffer_set_usage(buffer, usage);
    }
}

/// `ggml_backend_buffer_get_usage` (ggml-backend.cpp:199)
pub fn backend_buffer_get_usage(buffer: &BackendBufferRef) -> BackendBufferUsage {
    *buffer.usage.lock().unwrap()
}

/// `ggml_backend_buffer_get_type` (ggml-backend.cpp:204)
pub fn backend_buffer_get_type(buffer: &BackendBufferRef) -> BackendBufferTypeRef {
    buffer.buft.clone()
}

/// `ggml_backend_buffer_reset` (ggml-backend.cpp:209) — optional.
pub fn backend_buffer_reset(buffer: &BackendBufferRef, ctx: &mut Context) {
    if let Some(f) = buffer.iface.reset {
        f(buffer, ctx);
    }
}

/// `ggml_backend_buffer_copy_tensor` (ggml-backend.cpp:216) — do not use
/// directly, use [`backend_tensor_copy`].
fn backend_buffer_copy_tensor(ctx: &mut Context, src: TensorId, dst: TensorId) -> bool {
    let Some(dst_buf) = tensor_buffer(ctx, dst) else { return false };
    if let Some(cpy) = dst_buf.iface.cpy_tensor {
        return cpy(&dst_buf, ctx, src, dst);
    }
    false
}

// ===========================================================================
// multi-buffer buffer — ggml-backend.cpp:678-746
// ===========================================================================

/// `ggml_backend_multi_buffer_free_buffer` (ggml-backend.cpp:685)
fn multi_buffer_free_buffer(buffer: &BackendBufferRef) {
    let BufferContext::Multi { buffers } = &buffer.context else { return };
    for b in buffers {
        backend_buffer_free(Some(b.clone()));
    }
}

/// `ggml_backend_multi_buffer_clear` (ggml-backend.cpp:696)
fn multi_buffer_clear(buffer: &BackendBufferRef, ctx: &mut Context, value: u8) {
    let BufferContext::Multi { buffers } = &buffer.context else { return };
    for b in buffers {
        backend_buffer_clear(b, ctx, value);
    }
}

/// `ggml_backend_multi_buffer_i` (ggml-backend.cpp:704-716)
fn multi_buffer_i() -> BackendBufferI {
    BackendBufferI {
        free_buffer: Some(multi_buffer_free_buffer),
        clear: Some(multi_buffer_clear),
        ..BackendBufferI::default()
    }
}

/// `ggml_backend_multi_buffer_alloc_buffer` (ggml-backend.cpp:718)
pub fn backend_multi_buffer_alloc_buffer(buffers: &[BackendBufferRef]) -> BackendBufferRef {
    let mut total_size = 0;
    for b in buffers {
        total_size += backend_buffer_get_size(b);
    }
    backend_buffer_init(
        buffers[0].buft.clone(),
        multi_buffer_i(),
        BufferContext::Multi { buffers: buffers.to_vec() },
        total_size,
    )
}

/// `ggml_backend_buffer_is_multi_buffer` (ggml-backend.cpp:734) — compares
/// the free_buffer fn pointer exactly like C
/// (`buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer`).
#[allow(unpredictable_function_pointer_comparisons)]
pub fn backend_buffer_is_multi_buffer(buffer: &BackendBufferRef) -> bool {
    buffer.iface.free_buffer == Some(multi_buffer_free_buffer as fn(&BackendBufferRef))
}

/// `ggml_backend_multi_buffer_set_usage` (ggml-backend.cpp:739)
pub fn backend_multi_buffer_set_usage(buffer: &BackendBufferRef, usage: BackendBufferUsage) {
    assert!(backend_buffer_is_multi_buffer(buffer));
    let BufferContext::Multi { buffers } = &buffer.context else { return };
    for b in buffers {
        backend_buffer_set_usage(b, usage);
    }
}

// ===========================================================================
// tensor ↔ buffer association — the port's `tensor->buffer`
// ===========================================================================

/// The C `struct ggml_tensor` carries `buffer` and `data` pointer fields; the
/// port's `TensorMeta` has neither (see tensor.rs), so the association lives
/// in this process-wide side table keyed by (Context address, TensorId).
/// Entries dangle after a buffer is freed exactly like C's `tensor->buffer`
/// would — callers are expected to discard such tensors.
type TensorKey = (usize, u32);

fn tensor_buffers() -> &'static Mutex<HashMap<TensorKey, BackendBufferRef>> {
    static TENSOR_BUFFERS: OnceLock<Mutex<HashMap<TensorKey, BackendBufferRef>>> = OnceLock::new();
    TENSOR_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ctx_key(ctx: &Context) -> usize {
    // PERF8 fix: the raw address aliases across Context lifetimes — a fresh
    // Context allocated at a dead Context's stack/heap slot inherits its
    // stale tensor_buffers entries, so the gallocr treats its tensors as
    // externally allocated, never reserves room for them, and
    // ggml_gallocr_init_tensor indexes buffer_id −1 (the C cannot hit this:
    // its `tensor->buffer` lives in the tensor, not a process-wide table).
    // Serial `--test-threads 1` hit it as backend_sched::tests::
    // sched_cpu_bit_identity → sched_repeated_compute; pinned by
    // compute::tests::perf8_repro. The monotonic `uid` can never alias.
    ctx.uid as usize
}

/// C `tensor->buffer` — None for tensors allocated outside ggml-backend.
/// Views resolve through the view chain to the base tensor's buffer
/// (ggml-backend.cpp:337 `tensor->view_src ? tensor->view_src->buffer : tensor->buffer`).
pub fn tensor_buffer(ctx: &Context, tensor: TensorId) -> Option<BackendBufferRef> {
    let key = ctx_key(ctx);
    let map = tensor_buffers().lock().unwrap();
    // follow the view chain (Context::resolve)
    let mut id = tensor;
    loop {
        if let Some(b) = map.get(&(key, id.0)) {
            return Some(b.clone());
        }
        match ctx.tensors[id.0 as usize].view_src {
            Some(v) => id = v,
            None => return None,
        }
    }
}

fn tensor_set_buffer(ctx: &Context, tensor: TensorId, buffer: BackendBufferRef) {
    tensor_buffers()
        .lock()
        .unwrap()
        .insert((ctx_key(ctx), tensor.0), buffer);
}

// ===========================================================================
// tensor data access — ggml-backend.cpp:265-423
// ===========================================================================

fn tensor_has_storage(ctx: &Context, tensor: TensorId) -> bool {
    matches!(
        ctx.tensors[tensor.0 as usize].storage,
        Storage::Arena { .. } | Storage::External { .. }
    )
}

/// `ggml_backend_tensor_set` (ggml-backend.cpp:335)
pub fn backend_tensor_set(ctx: &mut Context, tensor: TensorId, data: &[u8], offset: usize) {
    let buf = tensor_buffer(ctx, tensor).expect("tensor buffer not set");

    if data.is_empty() {
        return;
    }

    assert!(tensor_has_storage(ctx, tensor), "tensor not allocated");
    assert!(offset + data.len() <= ctx.nbytes(tensor), "tensor write out of bounds");
    (buf.iface.set_tensor)(&buf, ctx, tensor, data, offset);
}

/// `ggml_backend_tensor_get` (ggml-backend.cpp:350)
pub fn backend_tensor_get(ctx: &Context, tensor: TensorId, data: &mut [u8], offset: usize) {
    let buf = tensor_buffer(ctx, tensor).expect("tensor buffer not set");

    if data.is_empty() {
        return;
    }

    assert!(tensor_has_storage(ctx, tensor), "tensor not allocated");
    assert!(offset + data.len() <= ctx.nbytes(tensor), "tensor read out of bounds");
    (buf.iface.get_tensor)(&buf, ctx, tensor, data, offset);
}

/// `ggml_backend_tensor_memset` (ggml-backend.cpp:409)
pub fn backend_tensor_memset(ctx: &mut Context, tensor: TensorId, value: u8, offset: usize, size: usize) {
    let buf = tensor_buffer(ctx, tensor).expect("tensor buffer not set");

    if size == 0 {
        return;
    }

    assert!(tensor_has_storage(ctx, tensor), "tensor not allocated");
    assert!(offset + size <= ctx.nbytes(tensor), "tensor write out of bounds");
    let memset = buf
        .iface
        .memset_tensor
        .expect("memset not implemented by backend buffer");
    memset(&buf, ctx, tensor, value, offset, size);
}

/// `ggml_backend_tensor_set_2d` (ggml-backend.cpp:365) — falls back to the
/// 1d path when the iface entry is NULL (the CPU case).
pub fn backend_tensor_set_2d(
    ctx: &mut Context,
    tensor: TensorId,
    data: &[u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) {
    let buf = tensor_buffer(ctx, tensor).expect("tensor buffer not set");
    if let Some(set2d) = buf.iface.set_tensor_2d {
        if n_copies > 1 && size > 0 {
            assert!(
                offset + (n_copies - 1) * stride_tensor + size <= ctx.nbytes(tensor),
                "tensor write out of bounds"
            );
            set2d(&buf, ctx, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
            return;
        }
    }
    for i in 0..n_copies {
        backend_tensor_set(ctx, tensor, &data[i * stride_data..i * stride_data + size], offset + i * stride_tensor);
    }
}

/// `ggml_backend_tensor_get_2d` (ggml-backend.cpp:387)
pub fn backend_tensor_get_2d(
    ctx: &Context,
    tensor: TensorId,
    data: &mut [u8],
    offset: usize,
    size: usize,
    n_copies: usize,
    stride_tensor: usize,
    stride_data: usize,
) {
    let buf = tensor_buffer(ctx, tensor).expect("tensor buffer not set");
    if let Some(get2d) = buf.iface.get_tensor_2d {
        if n_copies > 1 && size > 0 {
            assert!(
                offset + (n_copies - 1) * stride_tensor + size <= ctx.nbytes(tensor),
                "tensor read out of bounds"
            );
            get2d(&buf, ctx, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
            return;
        }
    }
    for i in 0..n_copies {
        backend_tensor_get(
            ctx,
            tensor,
            &mut data[i * stride_data..i * stride_data + size],
            offset + i * stride_tensor,
        );
    }
}

/// `ggml_backend_tensor_set_async` (ggml-backend.cpp:265) — without an async
/// entry the C code synchronizes then does the blocking set; the CPU backend
/// has no async path, so this is the sync set.
pub fn backend_tensor_set_async(backend: &BackendRef, ctx: &mut Context, tensor: TensorId, data: &[u8], offset: usize) {
    assert!(tensor_has_storage(ctx, tensor));
    assert!(offset + data.len() <= ctx.nbytes(tensor), "tensor write out of bounds");
    if backend.iface.set_tensor_async.is_none() {
        backend_synchronize(backend);
        backend_tensor_set(ctx, tensor, data, offset);
    }
}

/// `ggml_backend_tensor_get_async` (ggml-backend.cpp:279)
pub fn backend_tensor_get_async(backend: &BackendRef, ctx: &Context, tensor: TensorId, data: &mut [u8], offset: usize) {
    assert!(tensor_has_storage(ctx, tensor));
    assert!(offset + data.len() <= ctx.nbytes(tensor), "tensor read out of bounds");
    if backend.iface.get_tensor_async.is_none() {
        // CPU backend: no queued work, straight read
        backend_tensor_get(ctx, tensor, data, offset);
    }
}

// ===========================================================================
// Backend (stream) — ggml-backend-impl.h:121-163
// ===========================================================================

/// `struct ggml_backend_graph_optimize_params` (ggml-backend-impl.h:114-119) —
/// passed to graph_optimize so the backend can add allocation dependencies.
#[derive(Default)]
pub struct GraphOptimizeParams {
    /// keep `tensor` allocated at least until `until` (a node of the same
    /// graph) has been computed
    pub add_alloc_dep: Option<Box<dyn FnMut(&mut Context, TensorId, TensorId) + Send>>,
}

/// `struct ggml_backend_i` (ggml-backend-impl.h:121-156). The plan/event/2d
/// hooks are `Option`s exactly like the C NULL slots.
pub struct BackendI {
    pub get_name: fn(&Backend) -> String,
    pub free: Option<fn(&BackendRef)>,
    /// (optional) asynchronous tensor data access
    pub set_tensor_async: Option<fn(&BackendRef, &mut Context, TensorId, &[u8], usize)>,
    pub get_tensor_async: Option<fn(&BackendRef, &Context, TensorId, &mut [u8], usize)>,
    pub cpy_tensor_async: Option<fn(&BackendRef, &BackendRef, &mut Context, TensorId, TensorId) -> bool>,
    /// (optional) complete all pending operations (required if the backend
    /// supports async operations)
    pub synchronize: Option<fn(&BackendRef)>,
    /// compute graph (always async if supported by the backend)
    pub graph_compute: fn(&BackendRef, &mut Context, &mut Graph) -> GgmlStatus,
    /// (optional) event synchronization
    pub event_record: Option<fn(&BackendRef, &BackendEvent)>,
    pub event_wait: Option<fn(&BackendRef, &BackendEvent)>,
    /// (optional) sort/optimize the nodes in the graph
    pub graph_optimize: Option<fn(&BackendRef, &mut Context, &mut Graph, &mut GraphOptimizeParams)>,
}

/// The C `void *context` of a backend.
pub enum BackendCtx {
    None,
    /// the port's CPU adapter (ggml-cpu.cpp:96 `ggml_backend_cpu_context`)
    Cpu(CpuBackendCtx),
    /// test mock backend (see backend_sched.rs tests / tests/test-alloc.cpp:15)
    Mock(MockBackendCtx),
    /// foreign (DL) backend, task ③
    Foreign,
}

/// `struct ggml_backend_cpu_context` (ggml-cpu.cpp:96-99, subset the port
/// needs: the thread count the engine is driven with; atomic because the
/// reg proc `ggml_backend_set_n_threads` mutates it through a shared handle).
pub struct CpuBackendCtx {
    pub n_threads: AtomicUsize,
}

/// ggml.h GGML_DEFAULT_N_THREADS
pub const GGML_DEFAULT_N_THREADS: usize = 4;

/// test mock state — mirrors the reference's own `dummy_backend`
/// (tests/test-alloc.cpp:15-33) with the addition of one rejected op.
pub struct MockBackendCtx {
    pub name: &'static str,
    pub reject_op: Option<GgmlOp>,
}

/// `struct ggml_backend` (ggml-backend-impl.h:158-163). `guid` identifies the
/// backend kind (`ggml_backend_is_cpu`).
pub struct Backend {
    pub guid: [u8; 16],
    pub iface: BackendI,
    pub device: BackendDeviceRef,
    pub context: BackendCtx,
}

pub type BackendRef = Arc<Backend>;

/// `ggml_backend_guid` (ggml-backend.cpp:226)
pub fn backend_guid(backend: &BackendRef) -> [u8; 16] {
    backend.guid
}

/// `ggml_backend_name` (ggml-backend.cpp:233)
pub fn backend_name(backend: Option<&BackendRef>) -> String {
    match backend {
        None => "NULL".to_string(),
        Some(b) => (b.iface.get_name)(b),
    }
}

/// `ggml_backend_free` (ggml-backend.cpp:240)
pub fn backend_free(backend: Option<BackendRef>) {
    if let Some(b) = backend {
        if let Some(free) = b.iface.free {
            free(&b);
        }
    }
}

/// `ggml_backend_get_default_buffer_type` (ggml-backend.cpp:248)
pub fn backend_get_default_buffer_type(backend: &BackendRef) -> BackendBufferTypeRef {
    backend_dev_buffer_type(&backend.device)
}

/// `ggml_backend_alloc_buffer` (ggml-backend.cpp:253)
pub fn backend_alloc_buffer(backend: &BackendRef, ctx: &mut Context, size: usize) -> Option<BackendBufferRef> {
    backend_buft_alloc_buffer(&backend_get_default_buffer_type(backend), ctx, size)
}

/// `ggml_backend_get_alignment` (ggml-backend.cpp:257)
pub fn backend_get_alignment(backend: &BackendRef) -> usize {
    backend_buft_get_alignment(&backend_get_default_buffer_type(backend))
}

/// `ggml_backend_get_max_size` (ggml-backend.cpp:261)
pub fn backend_get_max_size(backend: &BackendRef) -> usize {
    backend_buft_get_max_size(&backend_get_default_buffer_type(backend))
}

/// `ggml_backend_synchronize` (ggml-backend.cpp:425) — optional (the CPU
/// backend has none: nothing is ever queued).
pub fn backend_synchronize(backend: &BackendRef) {
    if let Some(f) = backend.iface.synchronize {
        f(backend);
    }
}

/// `ggml_backend_graph_compute` (ggml-backend.cpp:455)
pub fn backend_graph_compute(backend: &BackendRef, ctx: &mut Context, cgraph: &mut Graph) -> GgmlStatus {
    let err = backend_graph_compute_async(backend, ctx, cgraph);
    backend_synchronize(backend);
    err
}

/// `ggml_backend_graph_compute_async` (ggml-backend.cpp:461)
pub fn backend_graph_compute_async(backend: &BackendRef, ctx: &mut Context, cgraph: &mut Graph) -> GgmlStatus {
    (backend.iface.graph_compute)(backend, ctx, cgraph)
}

/// `ggml_backend_supports_op` (ggml-backend.cpp:466)
pub fn backend_supports_op(backend: &BackendRef, ctx: &Context, op: TensorId) -> bool {
    backend_dev_supports_op(&backend.device, ctx, op)
}

/// `ggml_backend_supports_buft` (ggml-backend.cpp:471)
pub fn backend_supports_buft(backend: &BackendRef, buft: &BackendBufferTypeRef) -> bool {
    backend_dev_supports_buft(&backend.device, buft)
}

/// `ggml_backend_offload_op` (ggml-backend.cpp:476)
pub fn backend_offload_op(backend: &BackendRef, ctx: &Context, op: TensorId) -> bool {
    backend_dev_offload_op(&backend.device, ctx, op)
}

/// `ggml_backend_get_device` (ggml-backend.cpp:481)
pub fn backend_get_device(backend: &BackendRef) -> BackendDeviceRef {
    backend.device.clone()
}

// ===========================================================================
// backend copy — ggml-backend.cpp:488-530
// ===========================================================================

/// `ggml_are_same_layout` (ggml.cpp) — type + shape + strides equal.
pub fn are_same_layout(ctx: &Context, a: TensorId, b: TensorId) -> bool {
    let (ta, tb) = (&ctx.tensors[a.0 as usize], &ctx.tensors[b.0 as usize]);
    ta.ty == tb.ty && ta.ne == tb.ne && ta.nb == tb.nb
}

/// `ggml_backend_tensor_copy` (ggml-backend.cpp:488). The port has one
/// Context, so src and dst live in it; the buffer ifaces route the bytes.
pub fn backend_tensor_copy(ctx: &mut Context, src: TensorId, dst: TensorId) {
    assert!(are_same_layout(ctx, src, dst), "cannot copy tensors with different layouts");

    if src == dst {
        return;
    }

    let src_buf = tensor_buffer(ctx, src);
    let dst_buf = tensor_buffer(ctx, dst);

    if src_buf.as_ref().map(|b| backend_buffer_is_host(b)).unwrap_or(false) {
        let bytes = ctx.data_bytes(src).expect("tensor not allocated").to_vec();
        backend_tensor_set(ctx, dst, &bytes, 0);
    } else if dst_buf.as_ref().map(|b| backend_buffer_is_host(b)).unwrap_or(false) {
        let mut bytes = vec![0u8; ctx.nbytes(src)];
        backend_tensor_get(ctx, src, &mut bytes, 0);
        if let Some(d) = ctx.data_bytes_mut(dst) {
            d.copy_from_slice(&bytes);
        }
    } else if !backend_buffer_copy_tensor(ctx, src, dst) {
        // slow path through a staging buffer (ggml-backend.cpp:503-507)
        let mut bytes = vec![0u8; ctx.nbytes(src)];
        backend_tensor_get(ctx, src, &mut bytes, 0);
        backend_tensor_set(ctx, dst, &bytes, 0);
    }
}

/// `ggml_backend_tensor_copy_async` (ggml-backend.cpp:511)
pub fn backend_tensor_copy_async(
    backend_src: &BackendRef,
    backend_dst: &BackendRef,
    ctx: &mut Context,
    src: TensorId,
    dst: TensorId,
) {
    assert!(are_same_layout(ctx, src, dst), "cannot copy tensors with different layouts");
    if src == dst {
        return;
    }
    if let Some(cpy) = backend_dst.iface.cpy_tensor_async {
        if cpy(backend_src, backend_dst, ctx, src, dst) {
            return;
        }
    }
    // an async copy would normally happen after all the queued operations on
    // both backends are completed — to simulate the same behavior, synchronize
    // both backends first, and do a blocking copy
    backend_synchronize(backend_src);
    backend_synchronize(backend_dst);
    backend_tensor_copy(ctx, src, dst);
}

// ===========================================================================
// Events — ggml-backend.h:122-128 / ggml-backend.cpp:532-568
// ===========================================================================

/// `struct ggml_backend_event` (ggml-backend-impl.h:165-168)
pub struct BackendEvent {
    pub device: BackendDeviceRef,
    pub context: Arc<dyn std::any::Any + Send + Sync>,
}
pub type BackendEventRef = Arc<BackendEvent>;

/// `ggml_backend_event_new` (ggml-backend.cpp:534) — null device is allowed;
/// the CPU device provides no events.
pub fn backend_event_new(device: Option<&BackendDeviceRef>) -> Option<BackendEventRef> {
    let device = device?;
    let f = device.iface.event_new?;
    Some(f(device))
}

/// `ggml_backend_event_free` (ggml-backend.cpp:542)
pub fn backend_event_free(event: Option<BackendEventRef>) {
    let Some(event) = event else { return };
    if let Some(free) = event.device.iface.event_free {
        free(&event.device, &event);
    }
}

/// `ggml_backend_event_record` (ggml-backend.cpp:549)
pub fn backend_event_record(event: &BackendEventRef, backend: &BackendRef) {
    let record = backend.iface.event_record.expect("backend has no event_record");
    record(backend, event);
}

/// `ggml_backend_event_synchronize` (ggml-backend.cpp:556)
pub fn backend_event_synchronize(event: &BackendEventRef) {
    let f = event.device.iface.event_synchronize.expect("device has no event_synchronize");
    f(&event.device, event);
}

/// `ggml_backend_event_wait` (ggml-backend.cpp:563)
pub fn backend_event_wait(backend: &BackendRef, event: &BackendEventRef) {
    let f = backend.iface.event_wait.expect("backend has no event_wait");
    f(backend, event);
}

/// `ggml_backend_graph_optimize` (ggml-backend.cpp:570) — optional.
pub fn backend_graph_optimize(
    backend: &BackendRef,
    ctx: &mut Context,
    cgraph: &mut Graph,
    params: &mut GraphOptimizeParams,
) {
    if let Some(f) = backend.iface.graph_optimize {
        f(backend, ctx, cgraph, params);
    }
}

// ===========================================================================
// Backend device — ggml-backend-impl.h:176-224 / ggml-backend.h:134-179
// ===========================================================================

/// `enum ggml_backend_dev_type` (ggml-backend.h:134-145)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDevType {
    /// CPU device using system memory
    Cpu,
    /// GPU device using dedicated memory
    Gpu,
    /// integrated GPU device using host memory
    Igpu,
    /// accelerator devices intended to be used together with the CPU backend
    Accel,
    /// "meta" device wrapping multiple other devices for tensor parallelism
    Meta,
}

/// `struct ggml_backend_dev_caps` (ggml-backend.h:148-159)
#[derive(Debug, Clone, Copy)]
pub struct BackendDevCaps {
    /// asynchronous operations
    pub async_: bool,
    /// pinned host buffer
    pub host_buffer: bool,
    /// creating buffers from host ptr
    pub buffer_from_host_ptr: bool,
    /// event synchronization
    pub events: bool,
    /// mmap is supported for loading
    pub mmap_support: bool,
}

/// `struct ggml_backend_dev_props` (ggml-backend.h:162-179)
pub struct BackendDevProps {
    pub name: String,
    pub description: String,
    pub memory_free: usize,
    pub memory_total: usize,
    pub ty: BackendDevType,
    /// None when unknown
    pub device_id: Option<String>,
    pub caps: BackendDevCaps,
}

/// `struct ggml_backend_device_i` (ggml-backend-impl.h:176-218)
pub struct BackendDeviceI {
    pub get_name: fn(&BackendDevice) -> String,
    pub get_description: fn(&BackendDevice) -> String,
    /// device memory in bytes: 0 bytes to indicate no memory to report
    pub get_memory: fn(&BackendDevice) -> (usize, usize),
    pub get_type: fn(&BackendDevice) -> BackendDevType,
    pub get_props: fn(&BackendDevice, &mut BackendDevProps),
    /// backend (stream) initialization
    pub init_backend: fn(&BackendDeviceRef, Option<&str>) -> Option<BackendRef>,
    /// preferred buffer type
    pub get_buffer_type: fn(&BackendDeviceRef) -> BackendBufferTypeRef,
    /// (optional) host buffer type
    pub get_host_buffer_type: Option<fn(&BackendDeviceRef) -> Option<BackendBufferTypeRef>>,
    /// (optional) buffer from pointer — the port passes external-storage
    /// indices instead of raw pointers (see [`BufferContext::ExternalHost`])
    pub buffer_from_host_ptr:
        Option<fn(&BackendDeviceRef, &mut Context, usize, usize, usize, usize) -> Option<BackendBufferRef>>,
    /// check if the backend can compute an operation
    pub supports_op: fn(&BackendDeviceRef, &Context, TensorId) -> bool,
    /// check if the backend can use tensors allocated in a buffer type
    pub supports_buft: fn(&BackendDeviceRef, &BackendBufferTypeRef) -> bool,
    /// (optional) check if the backend wants to run an operation, even if the
    /// weights are allocated in an incompatible buffer
    pub offload_op: Option<fn(&BackendDeviceRef, &Context, TensorId) -> bool>,
    /// (optional) event synchronization
    pub event_new: Option<fn(&BackendDeviceRef) -> BackendEventRef>,
    pub event_free: Option<fn(&BackendDeviceRef, &BackendEventRef)>,
    pub event_synchronize: Option<fn(&BackendDeviceRef, &BackendEventRef)>,
}

/// The C `void *context` of a device.
pub enum BackendDeviceCtx {
    None,
    /// the port's CPU device (ggml-cpu.cpp:340 `ggml_backend_cpu_device_context`)
    Cpu,
    /// test mock (tests/test-alloc.cpp `dummy_backend_context` shape) —
    /// carries the mock's static params for the fn-pointer ifaces
    Mock(Arc<dyn std::any::Any + Send + Sync>),
    /// foreign (DL) device — raw C-ABI pointer into the loaded .so; the
    /// registry keeps the DlHandle alive while the reg is registered
    /// (ggml-backend-reg.cpp:110-113 `dl_handle_ptr handle`)
    Foreign(*mut sysffi::GgmlBackendDevice),
}

// SAFETY: the raw pointer is only dereferenced through the FFI wrappers below
// while the owning registry entry (which holds the DlHandle) is alive.
unsafe impl Send for BackendDevice {}
unsafe impl Sync for BackendDevice {}

/// `struct ggml_backend_device` (ggml-backend-impl.h:220-224)
pub struct BackendDevice {
    pub iface: BackendDeviceI,
    pub reg: Mutex<Option<BackendRegRef>>,
    pub context: BackendDeviceCtx,
}

pub type BackendDeviceRef = Arc<BackendDevice>;

/// device identity — `Arc` identity, plus raw-pointer identity for foreign
/// devices (each FFI `get_device` call yields a fresh wrapper).
pub fn dev_eq(a: &BackendDeviceRef, b: &BackendDeviceRef) -> bool {
    if Arc::ptr_eq(a, b) {
        return true;
    }
    matches!(
        (&a.context, &b.context),
        (BackendDeviceCtx::Foreign(p), BackendDeviceCtx::Foreign(q)) if p == q
    )
}

/// `ggml_backend_dev_name` (ggml-backend.cpp:579)
pub fn backend_dev_name(device: &BackendDeviceRef) -> String {
    (device.iface.get_name)(device)
}

/// `ggml_backend_dev_description` (ggml-backend.cpp:584)
pub fn backend_dev_description(device: &BackendDeviceRef) -> String {
    (device.iface.get_description)(device)
}

/// `ggml_backend_dev_memory` (ggml-backend.cpp:589)
pub fn backend_dev_memory(device: &BackendDeviceRef) -> (usize, usize) {
    (device.iface.get_memory)(device)
}

/// `ggml_backend_dev_type` (ggml-backend.cpp:594)
pub fn backend_dev_type(device: &BackendDeviceRef) -> BackendDevType {
    (device.iface.get_type)(device)
}

/// `ggml_backend_dev_get_props` (ggml-backend.cpp:599)
pub fn backend_dev_get_props(device: &BackendDeviceRef) -> BackendDevProps {
    let mut props = BackendDevProps {
        name: String::new(),
        description: String::new(),
        memory_free: 0,
        memory_total: 0,
        ty: BackendDevType::Cpu,
        device_id: None,
        caps: BackendDevCaps {
            async_: false,
            host_buffer: false,
            buffer_from_host_ptr: false,
            events: false,
            mmap_support: false,
        },
    };
    (device.iface.get_props)(device, &mut props);
    props
}

/// `ggml_backend_dev_backend_reg` (ggml-backend.cpp:605)
pub fn backend_dev_backend_reg(device: &BackendDeviceRef) -> Option<BackendRegRef> {
    device.reg.lock().unwrap().clone()
}

/// `ggml_backend_dev_init` (ggml-backend.cpp:610)
pub fn backend_dev_init(device: &BackendDeviceRef, params: Option<&str>) -> Option<BackendRef> {
    (device.iface.init_backend)(device, params)
}

/// `ggml_backend_dev_buffer_type` (ggml-backend.cpp:615)
pub fn backend_dev_buffer_type(device: &BackendDeviceRef) -> BackendBufferTypeRef {
    (device.iface.get_buffer_type)(device)
}

/// `ggml_backend_dev_host_buffer_type` (ggml-backend.cpp:620) — optional.
pub fn backend_dev_host_buffer_type(device: &BackendDeviceRef) -> Option<BackendBufferTypeRef> {
    device.iface.get_host_buffer_type.as_ref()?(&device)
}

/// `ggml_backend_dev_buffer_from_host_ptr` (ggml-backend.cpp:629)
pub fn backend_dev_buffer_from_host_ptr(
    device: &BackendDeviceRef,
    ctx: &mut Context,
    ext_index: usize,
    offset: usize,
    size: usize,
    max_tensor_size: usize,
) -> Option<BackendBufferRef> {
    let f = device.iface.buffer_from_host_ptr.as_ref()?;
    f(device, ctx, ext_index, offset, size, max_tensor_size)
}

/// `ggml_backend_dev_supports_op` (ggml-backend.cpp:634)
pub fn backend_dev_supports_op(device: &BackendDeviceRef, ctx: &Context, op: TensorId) -> bool {
    (device.iface.supports_op)(device, ctx, op)
}

/// `ggml_backend_dev_supports_buft` (ggml-backend.cpp:639)
pub fn backend_dev_supports_buft(device: &BackendDeviceRef, buft: &BackendBufferTypeRef) -> bool {
    (device.iface.supports_buft)(device, buft)
}

/// `ggml_backend_dev_offload_op` (ggml-backend.cpp:644) — optional, false.
pub fn backend_dev_offload_op(device: &BackendDeviceRef, ctx: &Context, op: TensorId) -> bool {
    if let Some(f) = device.iface.offload_op {
        return f(device, ctx, op);
    }
    false
}

// ===========================================================================
// Backend (reg) — ggml-backend-impl.h:230-246
// ===========================================================================

/// `ggml_backend_reg_get_proc_address` results the CPU reg hands out
/// (ggml-cpu.cpp:543-570, subset — the proc-address mechanism itself).
#[derive(Clone)]
pub enum RegProc {
    /// `ggml_backend_set_n_threads_t`
    SetNThreads(fn(&BackendRef, usize)),
}

/// `struct ggml_backend_reg_i` (ggml-backend-impl.h:230-240)
pub struct BackendRegI {
    pub get_name: fn(&BackendReg) -> String,
    /// enumerate available devices
    pub get_device_count: fn(&BackendReg) -> usize,
    pub get_device: fn(&BackendRegRef, usize) -> Option<BackendDeviceRef>,
    /// (optional) get a pointer to a function in the backend
    pub get_proc_address: Option<fn(&BackendReg, &str) -> Option<RegProc>>,
}

/// The C `void *context` of a reg.
pub enum BackendRegCtx {
    None,
    Cpu,
    /// foreign (DL) reg — raw C-ABI pointer; the owning registry entry holds
    /// the DlHandle (ggml-backend-reg.cpp:110-113)
    Foreign(*mut sysffi::GgmlBackendReg),
}

// SAFETY: see BackendDevice above.
unsafe impl Send for BackendReg {}
unsafe impl Sync for BackendReg {}

/// `struct ggml_backend_reg` (ggml-backend-impl.h:242-246)
pub struct BackendReg {
    /// initialize to GGML_BACKEND_API_VERSION
    pub api_version: i32,
    pub iface: BackendRegI,
    pub context: BackendRegCtx,
}

pub type BackendRegRef = Arc<BackendReg>;

/// `ggml_backend_reg_name` (ggml-backend.cpp:655)
pub fn backend_reg_name(reg: &BackendRegRef) -> String {
    (reg.iface.get_name)(reg)
}

/// `ggml_backend_reg_dev_count` (ggml-backend.cpp:660)
pub fn backend_reg_dev_count(reg: &BackendRegRef) -> usize {
    (reg.iface.get_device_count)(reg)
}

/// `ggml_backend_reg_dev_get` (ggml-backend.cpp:665)
pub fn backend_reg_dev_get(reg: &BackendRegRef, index: usize) -> Option<BackendDeviceRef> {
    (reg.iface.get_device)(reg, index)
}

/// `ggml_backend_reg_get_proc_address` (ggml-backend.cpp:670) — optional.
pub fn backend_reg_get_proc_address(reg: &BackendRegRef, name: &str) -> Option<RegProc> {
    reg.iface.get_proc_address.as_ref()?(&reg, name)
}

// ===========================================================================
// Backend registry — ggml-backend-reg.cpp
// ===========================================================================

/// `struct ggml_backend_reg_entry` (ggml-backend-reg.cpp:110-113)
struct RegEntry {
    reg: BackendRegRef,
    /// RAII keepalive of the dlopen handle — dropping the entry dlcloses the
    /// backend (the C `dl_handle_ptr handle` member)
    #[allow(dead_code)] // read only by Drop
    handle: Option<sysffi::DlHandle>,
}

/// `struct ggml_backend_registry` (ggml-backend-reg.cpp:115-290, i.e. its
/// `backends` + `devices` vectors)
#[derive(Default)]
struct RegistryState {
    backends: Vec<RegEntry>,
    devices: Vec<BackendDeviceRef>,
}

/// `get_reg()` (ggml-backend-reg.cpp:292-295) — process-wide singleton; the
/// built-in CPU backend is registered on first use like the C constructor
/// registers `ggml_backend_cpu_reg()` (ggml-backend-reg.cpp:171-173).
fn registry() -> &'static Mutex<RegistryState> {
    static REG: OnceLock<Mutex<RegistryState>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut state = RegistryState::default();
        // ggml-backend-reg.cpp:171-173 (#ifdef GGML_USE_CPU)
        register_backend_locked(&mut state, cpu_backend_reg(), None);
        Mutex::new(state)
    })
}

/// `ggml_backend_register` (ggml-backend-reg.cpp:298)
pub fn backend_register(reg: BackendRegRef) {
    let mut g = registry().lock().unwrap();
    register_backend_locked(&mut g, reg, None);
}

/// `ggml_backend_registry::register_backend` (ggml-backend-reg.cpp:186-205)
fn register_backend_locked(g: &mut RegistryState, reg: BackendRegRef, handle: Option<sysffi::DlHandle>) {
    for entry in &g.backends {
        if Arc::ptr_eq(&entry.reg, &reg) {
            return;
        }
    }
    let n_devs = backend_reg_dev_count(&reg);
    g.backends.push(RegEntry { reg: reg.clone(), handle });
    for i in 0..n_devs {
        if let Some(dev) = backend_reg_dev_get(&reg, i) {
            register_device_locked(g, dev);
        }
    }
}

/// `ggml_backend_device_register` (ggml-backend-reg.cpp:302)
pub fn backend_device_register(device: BackendDeviceRef) {
    let mut g = registry().lock().unwrap();
    register_device_locked(&mut g, device);
}

/// `ggml_backend_registry::register_device` (ggml-backend-reg.cpp:207-218)
fn register_device_locked(g: &mut RegistryState, device: BackendDeviceRef) {
    for dev in &g.devices {
        if dev_eq(dev, &device) {
            return;
        }
    }
    g.devices.push(device);
}

/// `striequals` (ggml-backend-reg.cpp:307-314)
fn striequals(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// `ggml_backend_reg_count` (ggml-backend-reg.cpp:316)
pub fn backend_reg_count() -> usize {
    registry().lock().unwrap().backends.len()
}

/// `ggml_backend_reg_get` (ggml-backend-reg.cpp:320)
pub fn backend_reg_get(index: usize) -> Option<BackendRegRef> {
    let g = registry().lock().unwrap();
    g.backends.get(index).map(|e| e.reg.clone())
}

/// `ggml_backend_reg_by_name` (ggml-backend-reg.cpp:325)
pub fn backend_reg_by_name(name: &str) -> Option<BackendRegRef> {
    for i in 0..backend_reg_count() {
        let reg = backend_reg_get(i).unwrap();
        if striequals(&backend_reg_name(&reg), name) {
            return Some(reg);
        }
    }
    None
}

/// `ggml_backend_dev_count` (ggml-backend-reg.cpp:336)
pub fn backend_dev_count() -> usize {
    registry().lock().unwrap().devices.len()
}

/// `ggml_backend_dev_get` (ggml-backend-reg.cpp:340)
pub fn backend_dev_get(index: usize) -> Option<BackendDeviceRef> {
    let g = registry().lock().unwrap();
    g.devices.get(index).cloned()
}

/// `ggml_backend_dev_by_name` (ggml-backend-reg.cpp:345)
pub fn backend_dev_by_name(name: &str) -> Option<BackendDeviceRef> {
    for i in 0..backend_dev_count() {
        let dev = backend_dev_get(i).unwrap();
        if striequals(&backend_dev_name(&dev), name) {
            return Some(dev);
        }
    }
    None
}

/// `ggml_backend_dev_by_type` (ggml-backend-reg.cpp:355)
pub fn backend_dev_by_type(ty: BackendDevType) -> Option<BackendDeviceRef> {
    for i in 0..backend_dev_count() {
        let dev = backend_dev_get(i).unwrap();
        if backend_dev_type(&dev) == ty {
            return Some(dev);
        }
    }
    None
}

/// `ggml_backend_init_by_name` (ggml-backend-reg.cpp:366)
pub fn backend_init_by_name(name: &str, params: Option<&str>) -> Option<BackendRef> {
    let dev = backend_dev_by_name(name)?;
    backend_dev_init(&dev, params)
}

/// `ggml_backend_init_by_type` (ggml-backend-reg.cpp:374)
pub fn backend_init_by_type(ty: BackendDevType, params: Option<&str>) -> Option<BackendRef> {
    let dev = backend_dev_by_type(ty)?;
    backend_dev_init(&dev, params)
}

/// `ggml_backend_init_best` (ggml-backend-reg.cpp:382) — the default device
/// selection: prefer a discrete GPU, then an integrated GPU, then the CPU.
pub fn backend_init_best() -> Option<BackendRef> {
    let dev = backend_dev_by_type(BackendDevType::Gpu)
        .or_else(|| backend_dev_by_type(BackendDevType::Igpu))
        .or_else(|| backend_dev_by_type(BackendDevType::Cpu))?;
    backend_dev_init(&dev, None)
}

// -------------------------------------------------------------------------
// Dynamic loading — ggml-backend-reg.cpp:392-605 + ggml-backend-dl.cpp
// -------------------------------------------------------------------------

/// `ggml_backend_load` (ggml-backend-reg.cpp:393 + load_backend:220-264).
/// Loads, scores and version-checks the backend, registers it, and keeps the
/// dl handle alive in the registry entry. Returns None on failure (the C
/// function returns nullptr and logs).
pub fn backend_load(path: &Path) -> Option<BackendRegRef> {
    let mut g = registry().lock().unwrap();
    load_backend_locked(&mut g, path, false)
}

/// `ggml_backend_registry::load_backend` (ggml-backend-reg.cpp:220-264)
fn load_backend_locked(g: &mut RegistryState, path: &Path, silent: bool) -> Option<BackendRegRef> {
    let handle = sysffi::dl_load_library(path);
    if handle.is_null() {
        if !silent {
            eprintln!("ggml_backend_load: failed to load {}: {}", path.display(), sysffi::dl_error());
        }
        return None;
    }

    // optional score: 0 means "not supported on this system" (impl.h:252-254)
    if let Some(sym) = handle.get_sym("ggml_backend_score") {
        let score_fn: sysffi::GgmlBackendScoreFn =
            unsafe { std::mem::transmute::<*mut c_void, sysffi::GgmlBackendScoreFn>(sym) };
        if unsafe { score_fn() } == 0 {
            if !silent {
                eprintln!("ggml_backend_load: backend {} is not supported on this system", path.display());
            }
            return None;
        }
    }

    let Some(init_sym) = handle.get_sym("ggml_backend_init") else {
        if !silent {
            eprintln!("ggml_backend_load: failed to find ggml_backend_init in {}", path.display());
        }
        return None;
    };
    let init_fn: sysffi::GgmlBackendInitFn =
        unsafe { std::mem::transmute::<*mut c_void, sysffi::GgmlBackendInitFn>(init_sym) };

    let raw = unsafe { init_fn() };
    if raw.is_null() {
        if !silent {
            eprintln!(
                "ggml_backend_load: failed to initialize backend from {}: ggml_backend_init returned NULL",
                path.display()
            );
        }
        return None;
    }
    // ggml-backend-reg.cpp:246-257
    let api_version = unsafe { (*raw).api_version };
    if api_version != GGML_BACKEND_API_VERSION {
        if !silent {
            eprintln!(
                "ggml_backend_load: failed to initialize backend from {}: incompatible API version (backend: {}, current: {})",
                path.display(),
                api_version,
                GGML_BACKEND_API_VERSION
            );
        }
        return None;
    }

    let reg = foreign_reg_wrap(raw);
    eprintln!(
        "ggml_backend_load: loaded {} backend from {}",
        backend_reg_name(&reg),
        path.display()
    );
    register_backend_locked(g, reg.clone(), Some(handle));
    Some(reg)
}

/// Wrap a foreign C-ABI `ggml_backend_reg*` in the Rust registry object. The
/// iface fns are non-capturing (they resolve the raw pointer out of
/// `context`), and only the name + device enumeration are driven over FFI
/// (task ③ extends this).
fn foreign_reg_wrap(raw: *mut sysffi::GgmlBackendReg) -> BackendRegRef {
    let api_version = unsafe { (*raw).api_version };
    Arc::new(BackendReg {
        api_version,
        iface: BackendRegI {
            get_name: foreign_reg_name,
            get_device_count: foreign_reg_dev_count,
            get_device: foreign_reg_dev_get,
            get_proc_address: None,
        },
        context: BackendRegCtx::Foreign(raw),
    })
}

/// the foreign reg's `get_name` (ggml-backend-impl.h:232)
fn foreign_reg_name(reg: &BackendReg) -> String {
    match &reg.context {
        BackendRegCtx::Foreign(r) => unsafe {
            (**r).iface.get_name.map(|f| sysffi::cstr_ptr_to_string(f(*r))).unwrap_or_default()
        },
        _ => String::new(),
    }
}

/// the foreign reg's `get_device_count` (impl.h:235)
fn foreign_reg_dev_count(reg: &BackendReg) -> usize {
    match &reg.context {
        BackendRegCtx::Foreign(r) => unsafe { (**r).iface.get_device_count.map(|f| f(*r)).unwrap_or(0) },
        _ => 0,
    }
}

/// the foreign reg's `get_device` (impl.h:236), wrapped for enumeration
fn foreign_reg_dev_get(reg: &BackendRegRef, index: usize) -> Option<BackendDeviceRef> {
    let BackendRegCtx::Foreign(r) = &reg.context else { return None };
    // SAFETY: the pointer came from ggml_backend_init() of a library whose
    // handle the registry entry keeps alive for as long as the reg is
    // registered (ggml-backend-reg.cpp:261).
    let dev = unsafe { (**r).iface.get_device.map(|f| f(*r, index)) }?;
    if dev.is_null() {
        return None;
    }
    Some(foreign_dev_wrap(dev, reg.clone()))
}

/// Wrap a foreign C-ABI `ggml_backend_device*` — name/description/type/memory
/// over FFI; everything else unsupported until task ③ (a foreign device
/// cannot yet produce a backend stream or buffer).
fn foreign_dev_wrap(raw: *mut sysffi::GgmlBackendDevice, reg: BackendRegRef) -> BackendDeviceRef {
    Arc::new(BackendDevice {
        reg: Mutex::new(Some(reg)),
        context: BackendDeviceCtx::Foreign(raw),
        iface: BackendDeviceI {
            get_name: foreign_dev_name,
            get_description: foreign_dev_description,
            get_type: foreign_dev_type,
            get_memory: foreign_dev_memory,
            get_props: foreign_dev_props,
            init_backend: |_, _| None,
            get_buffer_type: foreign_dev_buffer_type,
            get_host_buffer_type: None,
            buffer_from_host_ptr: None,
            supports_op: |_, _, _| false,
            supports_buft: |_, _| false,
            offload_op: None,
            event_new: None,
            event_free: None,
            event_synchronize: None,
        },
    })
}

fn foreign_dev_name(dev: &BackendDevice) -> String {
    match &dev.context {
        BackendDeviceCtx::Foreign(p) => unsafe {
            (**p).iface.get_name.map(|f| sysffi::cstr_ptr_to_string(f(*p))).unwrap_or_default()
        },
        _ => String::new(),
    }
}

fn foreign_dev_description(dev: &BackendDevice) -> String {
    match &dev.context {
        BackendDeviceCtx::Foreign(p) => unsafe {
            (**p).iface.get_description.map(|f| sysffi::cstr_ptr_to_string(f(*p))).unwrap_or_default()
        },
        _ => String::new(),
    }
}

/// the C `enum ggml_backend_dev_type` discriminants (ggml-backend.h:134-145)
fn foreign_dev_type(dev: &BackendDevice) -> BackendDevType {
    match &dev.context {
        BackendDeviceCtx::Foreign(p) => match unsafe { (**p).iface.get_type.map(|f| f(*p)) } {
            Some(1) => BackendDevType::Gpu,
            Some(2) => BackendDevType::Igpu,
            Some(3) => BackendDevType::Accel,
            Some(4) => BackendDevType::Meta,
            _ => BackendDevType::Cpu,
        },
        _ => BackendDevType::Cpu,
    }
}

fn foreign_dev_memory(dev: &BackendDevice) -> (usize, usize) {
    match &dev.context {
        BackendDeviceCtx::Foreign(p) => {
            let (mut free, mut total) = (0usize, 0usize);
            unsafe {
                if let Some(f) = (**p).iface.get_memory {
                    f(*p, &mut free, &mut total);
                }
            }
            (free, total)
        }
        _ => (0, 0),
    }
}

fn foreign_dev_props(dev: &BackendDevice, props: &mut BackendDevProps) {
    props.name = (dev.iface.get_name)(dev);
    props.description = (dev.iface.get_description)(dev);
    props.ty = (dev.iface.get_type)(dev);
    (props.memory_free, props.memory_total) = (dev.iface.get_memory)(dev);
}

fn foreign_dev_buffer_type(dev: &BackendDeviceRef) -> BackendBufferTypeRef {
    panic!(
        "foreign device {} has no buffer type support until task ③",
        (dev.iface.get_name)(dev)
    )
}

/// `ggml_backend_unload` (ggml-backend-reg.cpp:397 + unload_backend:266-289)
pub fn backend_unload(reg: &BackendRegRef) {
    let mut g = registry().lock().unwrap();
    // unload_backend(reg, silent = true)
    let pos = g.backends.iter().position(|e| Arc::ptr_eq(&e.reg, reg));
    let Some(pos) = pos else { return };
    // remove devices
    g.devices
        .retain(|dev| !matches!(backend_dev_backend_reg(dev), Some(r) if Arc::ptr_eq(&r, reg)));
    // remove backend — dropping the entry drops the DlHandle (dlclose)
    g.backends.remove(pos);
}

/// `get_executable_path` (ggml-backend-reg.cpp:401-462, linux branch)
fn get_executable_path() -> PathBuf {
    match std::fs::read_link("/proc/self/exe") {
        Ok(p) => p.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")),
        Err(_) => PathBuf::from("."),
    }
}

/// `backend_filename_prefix` (ggml-backend-reg.cpp:464)
fn backend_filename_prefix() -> &'static str {
    "libggml-"
}

/// `backend_filename_extension` (ggml-backend-reg.cpp:472)
fn backend_filename_extension() -> &'static str {
    ".so"
}

/// `ggml_backend_load_best` (ggml-backend-reg.cpp:480-572) — enumerate
/// `[lib]ggml-name-*.[so|dll]`, pick the highest scorer, else the base name.
fn backend_load_best(name: &str, silent: bool, user_search_path: Option<&str>) -> Option<BackendRegRef> {
    let file_prefix = format!("{}{}-", backend_filename_prefix(), name);
    let file_extension = backend_filename_extension();

    let mut search_paths: Vec<PathBuf> = Vec::new();
    match user_search_path {
        None => {
            // default search paths: executable directory, current directory
            search_paths.push(get_executable_path());
            if let Ok(cwd) = std::env::current_dir() {
                search_paths.push(cwd);
            }
        }
        Some(p) => search_paths.push(PathBuf::from(p)),
    }

    let mut best_score = 0;
    let mut best_path: Option<PathBuf> = None;

    for search_path in &search_paths {
        let Ok(entries) = std::fs::read_dir(search_path) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(filename) = path.file_name().and_then(|f| f.to_str()) else { continue };
            // (std::filesystem's extension() keeps the leading dot — Rust's
            // Path::extension() strips it, hence the trim)
            let ext_matches =
                path.extension().and_then(|e| e.to_str()) == Some(file_extension.trim_start_matches('.'));
            if !filename.starts_with(&file_prefix) || !ext_matches {
                continue;
            }
            let handle = sysffi::dl_load_library(&path);
            if handle.is_null() {
                if !silent {
                    eprintln!(
                        "ggml_backend_load_best: failed to load {}: {}",
                        path.display(),
                        sysffi::dl_error()
                    );
                }
                continue;
            }
            let score_sym = match handle.get_sym("ggml_backend_score") {
                Some(p) => p,
                None => {
                    if !silent {
                        eprintln!(
                            "ggml_backend_load_best: failed to find ggml_backend_score in {}",
                            path.display()
                        );
                    }
                    continue;
                }
            };
            let score_fn: sysffi::GgmlBackendScoreFn =
                unsafe { std::mem::transmute::<*mut c_void, sysffi::GgmlBackendScoreFn>(score_sym) };
            let s = unsafe { score_fn() };
            if s > best_score {
                best_score = s;
                best_path = Some(path.clone());
            }
            // handle dropped = dlclose (C unique_ptr scope)
        }
    }

    let mut g = registry().lock().unwrap();
    if best_score == 0 {
        // try to load the base backend
        let filename = format!("{}{}{}", backend_filename_prefix(), name, file_extension);
        for search_path in &search_paths {
            let path = search_path.join(&filename);
            if path.exists() {
                return load_backend_locked(&mut g, &path, silent);
            }
        }
        return None;
    }
    load_backend_locked(&mut g, &best_path.unwrap(), silent)
}

/// `ggml_backend_load_all` (ggml-backend-reg.cpp:574)
pub fn backend_load_all() {
    backend_load_all_from_path(None);
}

/// `ggml_backend_load_all_from_path` (ggml-backend-reg.cpp:578-605)
pub fn backend_load_all_from_path(dir_path: Option<&str>) {
    for name in [
        "blas", "zendnn", "cann", "cuda", "hip", "metal", "rpc", "sycl", "vulkan", "virtgpu", "opencl", "hexagon",
        "musa", "openvino", "cpu",
    ] {
        backend_load_best(name, true, dir_path);
    }
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    if let Ok(backend_path) = std::env::var("GGML_BACKEND_PATH") {
        if !backend_path.is_empty() {
            let _ = backend_load(Path::new(&backend_path));
        }
    }
}

// ===========================================================================
// utils — ggml-backend.cpp:2106-2147
// ===========================================================================

/// `ggml_is_view_op` (ggml-backend.cpp:757)
pub fn is_view_op(op: GgmlOp) -> bool {
    matches!(op, GgmlOp::View | GgmlOp::Reshape | GgmlOp::Permute | GgmlOp::Transpose)
}

/// `ggml_op_is_empty` (ggml.cpp)
fn op_is_empty(op: GgmlOp) -> bool {
    matches!(op, GgmlOp::None | GgmlOp::View | GgmlOp::Reshape | GgmlOp::Permute | GgmlOp::Transpose)
}

/// `ggml_op_alloc_size_may_expand` (ggml-backend.cpp:2108-2120)
pub fn op_alloc_size_may_expand(op: GgmlOp) -> bool {
    matches!(
        op,
        GgmlOp::FlashAttnExt | GgmlOp::MulMat | GgmlOp::MulMatId | GgmlOp::Cumsum | GgmlOp::Argsort | GgmlOp::TopK
    )
}

/// `ggml_backend_view_init` (ggml-backend.cpp:2122). In the port a view has
/// no data pointer to patch — resolution goes through the view chain — so
/// this only asserts the base is allocated (has a buffer through the side
/// table).
pub fn backend_view_init(ctx: &Context, tensor: TensorId) -> GgmlStatus {
    let view_src = ctx.tensors[tensor.0 as usize].view_src.expect("view_init on non-view");
    assert!(tensor_buffer(ctx, view_src).is_some(), "view_src buffer not set");
    GgmlStatus::Success
}

/// `ggml_backend_tensor_alloc` (ggml-backend.cpp:2134). `addr` is the arena
/// byte offset (the port's `void *addr`). The arena is grown to cover the
/// span so a later arena truncation can never leave storage out of bounds.
pub fn backend_tensor_alloc(buffer: &BackendBufferRef, ctx: &mut Context, tensor: TensorId, addr: usize) -> GgmlStatus {
    assert!(matches!(ctx.tensors[tensor.0 as usize].storage, Storage::None), "tensor already allocated");
    assert!(ctx.tensors[tensor.0 as usize].view_src.is_none(), "tensor is a view");
    let base = backend_buffer_get_base(buffer);
    if let Some(base) = base {
        assert!(addr >= base);
        assert!(addr + backend_buffer_get_alloc_size(buffer, ctx, tensor) <= base + buffer.size);
        // keep the Context arena covering the claimed span
        if matches!(buffer.context, BufferContext::ArenaSpan { .. }) {
            let arena_end = base + buffer.size;
            if ctx.arena.len() < arena_end {
                ctx.arena.resize(arena_end, 0);
            }
        }
    }

    let len = ctx.nbytes(tensor);
    ctx.tensors[tensor.0 as usize].storage = Storage::Arena { offset: addr, len };
    tensor_set_buffer(ctx, tensor, buffer.clone());
    backend_buffer_init_tensor(buffer, ctx, tensor)
}

/// The mmap-loading path (task ② integrator item): associate a pre-allocated
/// (external-storage) tensor with a buffer built by
/// [`backend_cpu_buffer_from_ptr`]. C does this implicitly — the loader's
/// `ggml_backend_dev_buffer_from_host_ptr` buffer and the tensors' data
/// pointers into the mapping are linked when `tensor->buffer` is assigned.
pub fn backend_tensor_attach_buffer(ctx: &Context, tensor: TensorId, buffer: BackendBufferRef) {
    assert!(matches!(ctx.tensors[tensor.0 as usize].storage, Storage::External { .. }));
    tensor_set_buffer(ctx, tensor, buffer);
}

// ===========================================================================
// CPU backend — ggml-backend.cpp:2353-2513 (buffers) + ggml-cpu.cpp
// (device/reg/backend stream)
// ===========================================================================

/// `ggml_backend_cpu_buffer_get_base` (ggml-backend.cpp:2355) — the arena
/// span start (or the external-storage offset for from_ptr buffers).
fn cpu_buffer_get_base(buffer: &BackendBufferRef) -> usize {
    match &buffer.context {
        BufferContext::ArenaSpan { base } => *base,
        BufferContext::ExternalHost { offset, .. } => *offset,
        _ => 0,
    }
}

/// `ggml_backend_cpu_buffer_free_buffer` (ggml-backend.cpp:2367) — the arena
/// span is reclaimed with the Context; nothing to free.
fn cpu_buffer_free_buffer(_buffer: &BackendBufferRef) {}

/// `ggml_backend_cpu_buffer_memset_tensor` (ggml-backend.cpp:2372)
fn cpu_buffer_memset_tensor(
    _buffer: &BackendBufferRef,
    ctx: &mut Context,
    tensor: TensorId,
    value: u8,
    offset: usize,
    size: usize,
) {
    if let Some(d) = ctx.data_bytes_mut(tensor) {
        d[offset..offset + size].fill(value);
    }
}

/// `ggml_backend_cpu_buffer_set_tensor` (ggml-backend.cpp:2379)
fn cpu_buffer_set_tensor(_buffer: &BackendBufferRef, ctx: &mut Context, tensor: TensorId, data: &[u8], offset: usize) {
    let d = ctx.data_bytes_mut(tensor).expect("tensor not allocated");
    d[offset..offset + data.len()].copy_from_slice(data);
}

/// `ggml_backend_cpu_buffer_get_tensor` (ggml-backend.cpp:2386)
fn cpu_buffer_get_tensor(_buffer: &BackendBufferRef, ctx: &Context, tensor: TensorId, data: &mut [u8], offset: usize) {
    let s = ctx.data_bytes(tensor).expect("tensor not allocated");
    data.copy_from_slice(&s[offset..offset + data.len()]);
}

/// `ggml_backend_cpu_buffer_cpy_tensor` (ggml-backend.cpp:2393)
fn cpu_buffer_cpy_tensor(buffer: &BackendBufferRef, ctx: &mut Context, src: TensorId, dst: TensorId) -> bool {
    if let Some(src_buf) = tensor_buffer(ctx, src) {
        if backend_buffer_is_host(&src_buf) {
            let bytes = ctx.data_bytes(src).expect("tensor not allocated").to_vec();
            (buffer.iface.set_tensor)(buffer, ctx, dst, &bytes, 0);
            return true;
        }
    }
    false
}

/// `ggml_backend_cpu_buffer_clear` (ggml-backend.cpp:2404) —
/// `memset(buffer->context, value, buffer->size)`. External (mmap) buffers
/// are read-only in the port and skip the clear.
fn cpu_buffer_clear(buffer: &BackendBufferRef, ctx: &mut Context, value: u8) {
    if let BufferContext::ArenaSpan { base } = &buffer.context {
        let end = (base + buffer.size).min(ctx.arena.len());
        ctx.arena[*base..end].fill(value);
    }
}

/// `ggml_backend_cpu_buffer_i` (ggml-backend.cpp:2409-2421)
pub(crate) fn cpu_buffer_i() -> BackendBufferI {
    BackendBufferI {
        free_buffer: Some(cpu_buffer_free_buffer),
        get_base: Some(cpu_buffer_get_base),
        init_tensor: None, // no initialization required
        memset_tensor: Some(cpu_buffer_memset_tensor),
        set_tensor: cpu_buffer_set_tensor,
        get_tensor: cpu_buffer_get_tensor,
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: Some(cpu_buffer_cpy_tensor),
        clear: Some(cpu_buffer_clear),
        reset: None,
    }
}

/// `ggml_backend_cpu_buffer_from_ptr_i` (ggml-backend.cpp:2423-2435) — ptr is
/// not owned by the buffer, so it does not need to be freed
fn cpu_buffer_from_ptr_i() -> BackendBufferI {
    BackendBufferI {
        free_buffer: None,
        get_base: Some(cpu_buffer_get_base),
        init_tensor: None,
        memset_tensor: Some(cpu_buffer_memset_tensor),
        set_tensor: cpu_buffer_set_tensor,
        get_tensor: cpu_buffer_get_tensor,
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: Some(cpu_buffer_cpy_tensor),
        clear: Some(cpu_buffer_clear),
        reset: None,
    }
}

/// `ggml_backend_cpu_buffer_type_get_name` (ggml-backend.cpp:2441)
fn cpu_buffer_type_get_name(_buft: &BackendBufferType) -> String {
    "CPU".to_string()
}

/// `ggml_backend_cpu_buffer_type_alloc_buffer` (ggml-backend.cpp:2447) — the
/// port's `ggml_aligned_malloc(size)` is a claimed, zero-filled tail span of
/// the Context arena, TENSOR_ALIGNMENT-aligned like C.
pub(crate) fn cpu_buffer_type_alloc_buffer(
    buft: &BackendBufferTypeRef,
    ctx: &mut Context,
    size: usize,
) -> Option<BackendBufferRef> {
    let base = (ctx.arena.len() + TENSOR_ALIGNMENT - 1) & !(TENSOR_ALIGNMENT - 1);
    ctx.arena.resize(base + size, 0);
    Some(backend_buffer_init(
        buft.clone(),
        cpu_buffer_i(),
        BufferContext::ArenaSpan { base },
        size,
    ))
}

/// `ggml_backend_cpu_buffer_type_get_alignment` (ggml-backend.cpp:2458)
fn cpu_buffer_type_get_alignment(_buft: &BackendBufferType) -> usize {
    TENSOR_ALIGNMENT
}

/// `ggml_backend_cpu_buffer_type_is_host` (ggml-backend.cpp:2464)
fn cpu_buffer_type_is_host(_buft: &BackendBufferType) -> bool {
    true
}

/// `ggml_backend_cpu_buffer_type` (ggml-backend.cpp:2470-2485) — function-local
/// static in C; a global OnceLock here (identity must be stable).
pub fn backend_cpu_buffer_type() -> BackendBufferTypeRef {
    static BUFT: OnceLock<BackendBufferTypeRef> = OnceLock::new();
    BUFT.get_or_init(|| {
        Arc::new(BackendBufferType {
            iface: BackendBufferTypeI {
                get_name: cpu_buffer_type_get_name,
                alloc_buffer: cpu_buffer_type_alloc_buffer,
                get_alignment: cpu_buffer_type_get_alignment,
                get_max_size: None,    // defaults to SIZE_MAX
                get_alloc_size: None,  // defaults to ggml_nbytes
                is_host: Some(cpu_buffer_type_is_host),
            },
            device: None, // FIXME (ggml-backend.cpp:2480) reg_dev_get(cpu_reg, 0)
            context: Arc::new(()),
        })
    })
    .clone()
}

/// `ggml_backend_cpu_buffer_from_ptr_type_get_name` (ggml-backend.cpp:2487)
fn cpu_buffer_from_ptr_type_get_name(_buft: &BackendBufferType) -> String {
    "CPU_Mapped".to_string()
}

/// `ggml_backend_cpu_buffer_from_ptr_type` (ggml-backend.cpp:2493-2508)
fn backend_cpu_buffer_from_ptr_type() -> BackendBufferTypeRef {
    static BUFT: OnceLock<BackendBufferTypeRef> = OnceLock::new();
    BUFT.get_or_init(|| {
        Arc::new(BackendBufferType {
            iface: BackendBufferTypeI {
                get_name: cpu_buffer_from_ptr_type_get_name,
                alloc_buffer: cpu_buffer_type_alloc_buffer,
                get_alignment: cpu_buffer_type_get_alignment,
                get_max_size: None,
                get_alloc_size: None,
                is_host: Some(cpu_buffer_type_is_host),
            },
            device: None,
            context: Arc::new(()),
        })
    })
    .clone()
}

/// `ggml_backend_cpu_buffer_from_ptr` (ggml-backend.cpp:2510) — the port's
/// "host pointer" is a registered external buffer (mmap) plus offset.
pub fn backend_cpu_buffer_from_ptr(ctx: &Context, ext_index: usize, offset: usize, size: usize) -> BackendBufferRef {
    assert!(offset % TENSOR_ALIGNMENT == 0, "buffer pointer must be aligned");
    let _ = ctx;
    backend_buffer_init(
        backend_cpu_buffer_from_ptr_type(),
        cpu_buffer_from_ptr_i(),
        BufferContext::ExternalHost { ext_index, offset },
        size,
    )
}

// -------------------------------------------------------------------------
// CPU device — ggml-cpu.cpp:340-505
// -------------------------------------------------------------------------

/// `ggml_backend_cpu_device_get_name` (ggml-cpu.cpp:353)
fn cpu_device_get_name(_dev: &BackendDevice) -> String {
    "CPU".to_string()
}

/// `ggml_backend_cpu_device_get_description` (ggml-cpu.cpp:359) — C builds a
/// feature string; the port keeps the plain name.
fn cpu_device_get_description(_dev: &BackendDevice) -> String {
    "CPU".to_string()
}

/// `ggml_backend_cpu_device_get_memory` (ggml-cpu.cpp:365) — C uses
/// sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGE_SIZE) and reports all of it free;
/// the port reads /proc/meminfo (no libc dependency).
fn cpu_device_get_memory(_dev: &BackendDevice) -> (usize, usize) {
    let read = |key: &str| -> Option<usize> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix(key) {
                let kb: usize = rest.trim_start().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    };
    let total = read("MemTotal:").unwrap_or(0);
    // "free" system memory is ill-defined, for practical purposes assume that
    // all of it is free (ggml-cpu.cpp:380-381)
    (total, total)
}

/// `ggml_backend_cpu_device_get_type` (ggml-cpu.cpp:383)
fn cpu_device_get_type(_dev: &BackendDevice) -> BackendDevType {
    BackendDevType::Cpu
}

/// `ggml_backend_cpu_device_get_props` (ggml-cpu.cpp:390-403)
fn cpu_device_get_props(dev: &BackendDevice, props: &mut BackendDevProps) {
    props.name = cpu_device_get_name(dev);
    props.description = cpu_device_get_description(dev);
    props.ty = cpu_device_get_type(dev);
    (props.memory_free, props.memory_total) = cpu_device_get_memory(dev);
    props.caps = BackendDevCaps {
        async_: false,
        host_buffer: false,
        buffer_from_host_ptr: true,
        events: false,
        mmap_support: true,
    };
}

/// `ggml_backend_cpu_device_init_backend` (ggml-cpu.cpp:404) →
/// `ggml_backend_cpu_init` (ggml-cpu.cpp:210-232)
fn cpu_device_init_backend(_dev: &BackendDeviceRef, _params: Option<&str>) -> Option<BackendRef> {
    Some(cpu_backend_init())
}

/// `ggml_backend_cpu_device_get_buffer_type` (ggml-cpu.cpp:411)
fn cpu_device_get_buffer_type(_dev: &BackendDeviceRef) -> BackendBufferTypeRef {
    backend_cpu_buffer_type()
}

/// `ggml_backend_cpu_device_buffer_from_host_ptr` (ggml-cpu.cpp:417)
fn cpu_device_buffer_from_host_ptr(
    _dev: &BackendDeviceRef,
    ctx: &mut Context,
    ext_index: usize,
    offset: usize,
    size: usize,
    _max_tensor_size: usize,
) -> Option<BackendBufferRef> {
    Some(backend_cpu_buffer_from_ptr(ctx, ext_index, offset, size))
}

/// `ggml_backend_cpu_device_supports_op` (ggml-cpu.cpp:424-491) — ported for
/// the port's op surface; ops C checks that the port does not implement
/// (SOFT_MAX_BACK, IM2COL_BACK, GET_ROWS_BACK, OUT_PROD, single-op CONV_2D)
/// default to `true` like C's `default:` and would fail at compute time.
/// The MUL_MAT `GGML_HINT_SRC0_IS_HADAMARD` special case is dropped — the
/// port never emits that hint.
fn cpu_device_supports_op(_dev: &BackendDeviceRef, ctx: &Context, op: TensorId) -> bool {
    let t = &ctx.tensors[op.0 as usize];

    if matches!(t.op, GgmlOp::None | GgmlOp::Reshape | GgmlOp::View | GgmlOp::Permute | GgmlOp::Transpose) {
        return true;
    }

    match t.op {
        // ggml-cpu.cpp:447-456 — missing type_traits.from_float
        GgmlOp::Cpy | GgmlOp::SetRows => !matches!(
            t.ty,
            GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq2S
                | GgmlType::Iq1S
                | GgmlType::Iq1M
        ),
        // ggml-cpu.cpp:457-463
        GgmlOp::MulMat => match t.src[1] {
            None => true,
            Some(src1) => {
                let src1_ty = ctx.tensors[src1.0 as usize].ty;
                src1_ty == GgmlType::F32
                    || crate::vec_dot::vec_dot_type(ctx.tensors[t.src[0].unwrap().0 as usize].ty) == Some(src1_ty)
            }
        },
        // ggml-cpu.cpp:487-488
        GgmlOp::SsmScan => t.op_params[0] == 1 || ctx.ne(t.src[3].unwrap())[0] == 1,
        _ => true,
    }
}

/// `ggml_backend_cpu_device_supports_buft` (ggml-cpu.cpp:492) — the port has
/// no CPU extra buffer types.
fn cpu_device_supports_buft(_dev: &BackendDeviceRef, buft: &BackendBufferTypeRef) -> bool {
    backend_buft_is_host(buft)
}

/// `ggml_backend_cpu_device_i` (ggml-cpu.cpp:493-505)
fn cpu_device_i() -> BackendDeviceI {
    BackendDeviceI {
        get_name: cpu_device_get_name,
        get_description: cpu_device_get_description,
        get_memory: cpu_device_get_memory,
        get_type: cpu_device_get_type,
        get_props: cpu_device_get_props,
        init_backend: cpu_device_init_backend,
        get_buffer_type: cpu_device_get_buffer_type,
        get_host_buffer_type: None,
        buffer_from_host_ptr: Some(cpu_device_buffer_from_host_ptr),
        supports_op: cpu_device_supports_op,
        supports_buft: cpu_device_supports_buft,
        offload_op: None,
        event_new: None,
        event_free: None,
        event_synchronize: None,
    }
}

// -------------------------------------------------------------------------
// CPU reg — ggml-cpu.cpp:508-705
// -------------------------------------------------------------------------

/// `ggml_backend_cpu_reg_get_name` (ggml-cpu.cpp:511)
fn cpu_reg_get_name(_reg: &BackendReg) -> String {
    "CPU".to_string()
}

/// `ggml_backend_cpu_reg_get_device_count` (ggml-cpu.cpp:516)
fn cpu_reg_get_device_count(_reg: &BackendReg) -> usize {
    1
}

/// `ggml_backend_cpu_reg_get_device` (ggml-cpu.cpp:521) — static device in C;
/// stable identity via OnceLock here.
fn cpu_reg_get_device(reg: &BackendRegRef, index: usize) -> Option<BackendDeviceRef> {
    assert!(index == 0);
    static DEV: OnceLock<BackendDeviceRef> = OnceLock::new();
    Some(
        DEV.get_or_init(|| {
            Arc::new(BackendDevice {
                iface: cpu_device_i(),
                reg: Mutex::new(Some(reg.clone())),
                context: BackendDeviceCtx::Cpu,
            })
        })
        .clone(),
    )
}

/// `ggml_backend_cpu_get_proc_address` (ggml-cpu.cpp:543-570, subset)
fn cpu_reg_get_proc_address(_reg: &BackendReg, name: &str) -> Option<RegProc> {
    if name == "ggml_backend_set_n_threads" {
        return Some(RegProc::SetNThreads(|backend, n_threads| {
            cpu_backend_set_n_threads(backend, n_threads);
        }));
    }
    None
}

/// `ggml_backend_cpu_reg` (ggml-cpu.cpp:703-715) — static in C.
pub fn cpu_backend_reg() -> BackendRegRef {
    static REG: OnceLock<BackendRegRef> = OnceLock::new();
    REG.get_or_init(|| {
        Arc::new(BackendReg {
            api_version: GGML_BACKEND_API_VERSION,
            iface: BackendRegI {
                get_name: cpu_reg_get_name,
                get_device_count: cpu_reg_get_device_count,
                get_device: cpu_reg_get_device,
                get_proc_address: Some(cpu_reg_get_proc_address),
            },
            context: BackendRegCtx::Cpu,
        })
    })
    .clone()
}

// -------------------------------------------------------------------------
// CPU backend stream — ggml-cpu.cpp:112-232
// -------------------------------------------------------------------------

/// `ggml_backend_cpu_get_name` (ggml-cpu.cpp:112)
fn cpu_backend_get_name(_backend: &Backend) -> String {
    "CPU".to_string()
}

/// `ggml_backend_cpu_graph_compute` (ggml-cpu.cpp:175-197) — drives the
/// port's existing CPU engine (compute.rs::graph_compute); the engine is not
/// modified, only wrapped.
fn cpu_backend_graph_compute(backend: &BackendRef, ctx: &mut Context, cgraph: &mut Graph) -> GgmlStatus {
    let n_threads = match &backend.context {
        BackendCtx::Cpu(c) => c.n_threads.load(std::sync::atomic::Ordering::Relaxed),
        _ => unreachable!(),
    };
    crate::compute::graph_compute(ctx, cgraph, n_threads);
    GgmlStatus::Success
}

/// `ggml_backend_cpu_i` (ggml-cpu.cpp:199-216) — the graph-plan fns are not
/// ported (the port's engine has no separate plan stage), everything else
/// matches (all optional slots NULL).
fn cpu_backend_i() -> BackendI {
    BackendI {
        get_name: cpu_backend_get_name,
        free: None,
        set_tensor_async: None,
        get_tensor_async: None,
        cpy_tensor_async: None,
        synchronize: None,
        graph_compute: cpu_backend_graph_compute,
        event_record: None,
        event_wait: None,
        graph_optimize: None,
    }
}

/// `ggml_backend_cpu_guid` (ggml-cpu.cpp:218)
fn cpu_backend_guid() -> [u8; 16] {
    [
        0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89,
    ]
}

/// `ggml_backend_cpu_init` (ggml-cpu.cpp:210-232)
pub fn cpu_backend_init() -> BackendRef {
    Arc::new(Backend {
        guid: cpu_backend_guid(),
        iface: cpu_backend_i(),
        device: cpu_reg_get_device(&cpu_backend_reg(), 0).unwrap(),
        context: BackendCtx::Cpu(CpuBackendCtx {
            n_threads: AtomicUsize::new(GGML_DEFAULT_N_THREADS),
        }),
    })
}

/// `ggml_backend_is_cpu` (ggml-cpu.cpp:234)
pub fn backend_is_cpu(backend: &BackendRef) -> bool {
    backend.guid == cpu_backend_guid()
}

/// `ggml_backend_cpu_set_n_threads` (ggml-cpu.cpp:268)
pub fn cpu_backend_set_n_threads(backend: &BackendRef, n_threads: usize) {
    assert!(backend_is_cpu(backend));
    match &backend.context {
        BackendCtx::Cpu(c) => c.n_threads.store(n_threads, std::sync::atomic::Ordering::Relaxed),
        _ => unreachable!(),
    }
}

/// Registry reset for tests — unloads every dynamically loaded backend,
/// leaving exactly the built-in CPU entry (mirrors a fresh process). Does
/// NOT clear the tensor↔buffer side table: other tests may hold entries.
#[cfg(test)]
pub(crate) fn reset_registry_for_tests() {
    let mut g = registry().lock().unwrap();
    g.backends.retain(|e| matches!(e.reg.context, BackendRegCtx::Cpu));
    g.devices.retain(|dev| matches!(dev.context, BackendDeviceCtx::Cpu));
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgmlType;

    /// the registry is process-global — the tests that reset and count it
    /// must not overlap (cargo test runs tests in parallel threads)
    static REGISTRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// mirror of the reference's own backend test block: buffer
    /// alloc/get/set/clear round-trip through the CPU backend adapter
    /// (tests/test-alloc.cpp's dummy backend checks + ggml-backend.cpp's
    /// buffer API).
    #[test]
    fn cpu_buffer_roundtrip() {
        let mut ctx = Context::new();
        let buft = backend_cpu_buffer_type();
        assert_eq!(backend_buft_name(&buft), "CPU");
        assert_eq!(backend_buft_get_alignment(&buft), TENSOR_ALIGNMENT);
        assert_eq!(backend_buft_get_max_size(&buft), usize::MAX);
        assert!(backend_buft_is_host(&buft));

        // zero-size allocation returns a dummy buffer (ggml-backend.cpp:41-44)
        let dummy = backend_buft_alloc_buffer(&buft, &mut ctx, 0).unwrap();
        assert_eq!(backend_buffer_get_size(&dummy), 0);
        assert!(backend_buffer_get_base(&dummy).is_none());
        backend_buffer_clear(&dummy, &mut ctx, 0xAA); // optional for zero size
        backend_buffer_free(Some(dummy));

        // real buffer + tensor alloc/set/get/clear round-trip
        let buf = backend_buft_alloc_buffer(&buft, &mut ctx, 4096).unwrap();
        assert_eq!(backend_buffer_get_size(&buf), 4096);
        let base = backend_buffer_get_base(&buf).unwrap();
        assert_eq!(base % TENSOR_ALIGNMENT, 0);

        let t = ctx.new_tensor_2d(GgmlType::F32, 16, 8);
        assert_eq!(ctx.nbytes(t), 16 * 8 * 4);
        assert_eq!(backend_tensor_alloc(&buf, &mut ctx, t, base), GgmlStatus::Success);
        assert_eq!(backend_buffer_get_usage(&buf), BackendBufferUsage::Any);
        backend_buffer_set_usage(&buf, BackendBufferUsage::Weights);
        assert_eq!(backend_buffer_get_usage(&buf), BackendBufferUsage::Weights);

        let src: Vec<u8> = (0..ctx.nbytes(t)).map(|i| i as u8).collect();
        backend_tensor_set(&mut ctx, t, &src, 0);
        let mut dst = vec![0u8; ctx.nbytes(t)];
        backend_tensor_get(&ctx, t, &mut dst, 0);
        assert_eq!(src, dst);

        // offset write (ggml-backend.cpp:335 offset semantics)
        backend_tensor_set(&mut ctx, t, &[1, 2, 3, 4], 16);
        let mut four = [0u8; 4];
        backend_tensor_get(&ctx, t, &mut four, 16);
        assert_eq!(four, [1, 2, 3, 4]);

        // memset + clear (buffer-wide)
        backend_tensor_memset(&mut ctx, t, 0x5A, 0, 4);
        let mut four = [0u8; 4];
        backend_tensor_get(&ctx, t, &mut four, 0);
        assert_eq!(four, [0x5A; 4]);
        backend_buffer_clear(&buf, &mut ctx, 0);
        let mut all = vec![0u8; ctx.nbytes(t)];
        backend_tensor_get(&ctx, t, &mut all, 0);
        assert!(all.iter().all(|&b| b == 0));

        backend_buffer_free(Some(buf));
    }

    /// multi-buffer (ggml-backend.cpp:678-746)
    #[test]
    fn multi_buffer() {
        let mut ctx = Context::new();
        let buft = backend_cpu_buffer_type();
        let b0 = backend_buft_alloc_buffer(&buft, &mut ctx, 128).unwrap();
        let b1 = backend_buft_alloc_buffer(&buft, &mut ctx, 256).unwrap();
        let multi = backend_multi_buffer_alloc_buffer(&[b0.clone(), b1.clone()]);
        assert!(backend_buffer_is_multi_buffer(&multi));
        assert!(!backend_buffer_is_multi_buffer(&b0));
        assert_eq!(backend_buffer_get_size(&multi), 384);
        // set_usage propagates to children (ggml-backend.cpp:187-197)
        backend_buffer_set_usage(&multi, BackendBufferUsage::Compute);
        assert_eq!(backend_buffer_get_usage(&b0), BackendBufferUsage::Compute);
        assert_eq!(backend_buffer_get_usage(&b1), BackendBufferUsage::Compute);
    }

    /// device enumeration lists exactly 1 device (CPU); registry resolution
    /// by name/type works (ggml-backend-reg.cpp:316-390)
    #[test]
    fn registry_enumeration() {
        let _guard = REGISTRY_TEST_LOCK.lock().unwrap();
        reset_registry_for_tests();
        assert_eq!(backend_reg_count(), 1);
        assert_eq!(backend_dev_count(), 1);
        let reg = backend_reg_get(0).unwrap();
        assert_eq!(backend_reg_name(&reg), "CPU");
        assert_eq!(backend_reg_dev_count(&reg), 1);
        assert!(backend_reg_by_name("cpu").is_some(), "case-insensitive (striequals)");
        assert!(backend_reg_by_name("CUDA").is_none());
        let dev = backend_dev_get(0).unwrap();
        assert_eq!(backend_dev_name(&dev), "CPU");
        assert_eq!(backend_dev_type(&dev), BackendDevType::Cpu);
        assert!(backend_dev_by_name("CPU").is_some());
        assert!(backend_dev_by_type(BackendDevType::Gpu).is_none());
        assert!(backend_dev_by_type(BackendDevType::Cpu).is_some());

        // props (ggml-cpu.cpp:390-403)
        let props = backend_dev_get_props(&dev);
        assert_eq!(props.name, "CPU");
        assert_eq!(props.ty, BackendDevType::Cpu);
        assert!(props.caps.buffer_from_host_ptr && props.caps.mmap_support);
        assert!(!props.caps.async_ && !props.caps.events);
        assert!(props.memory_total > 0, "reads /proc/meminfo");

        // backends init through the device / by name / best
        let b = backend_dev_init(&dev, None).unwrap();
        assert!(backend_is_cpu(&b));
        assert_eq!(backend_name(Some(&b)), "CPU");
        assert!(backend_is_cpu(&backend_init_by_name("cpu", None).unwrap()));
        assert!(backend_is_cpu(&backend_init_by_type(BackendDevType::Cpu, None).unwrap()));
        assert!(backend_is_cpu(&backend_init_best().unwrap()));
        assert_eq!(backend_name(None), "NULL");

        // the CPU device's buffer type is the CPU buffer type; n_threads proc
        let bt = backend_dev_buffer_type(&dev);
        assert!(buft_eq(&bt, &backend_cpu_buffer_type()));
        match backend_reg_get_proc_address(&reg, "ggml_backend_set_n_threads") {
            Some(RegProc::SetNThreads(f)) => f(&b, 2),
            None => panic!("cpu reg proc address"),
        }
        let mut ctx = Context::new();
        let buf = backend_alloc_buffer(&b, &mut ctx, 64).unwrap();
        assert_eq!(backend_buffer_name(&buf), "CPU");
    }

    /// tensor copy between two CPU-buffer tensors through the host fast path
    /// (ggml-backend.cpp:488-509) — the tensors must live in backend buffers
    /// (`tensor->buffer` in C)
    #[test]
    fn tensor_copy_host() {
        let mut ctx = Context::new();
        let buft = backend_cpu_buffer_type();
        let buf = backend_buft_alloc_buffer(&buft, &mut ctx, 256).unwrap();
        let base = backend_buffer_get_base(&buf).unwrap();
        let x = ctx.new_tensor_1d(GgmlType::F32, 4);
        let y = ctx.new_tensor_1d(GgmlType::F32, 4);
        assert_eq!(backend_tensor_alloc(&buf, &mut ctx, x, base), GgmlStatus::Success);
        assert_eq!(backend_tensor_alloc(&buf, &mut ctx, y, base + 64), GgmlStatus::Success);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        backend_tensor_set(&mut ctx, x, &data, 0);
        backend_tensor_copy(&mut ctx, x, y);
        assert_eq!(ctx.f32s(y).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // src == dst is a no-op (ggml-backend.cpp:491)
        backend_tensor_copy(&mut ctx, x, x);
        assert_eq!(ctx.f32s(x).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// `ggml_backend_cpu_buffer_from_ptr` (ggml-backend.cpp:2510) over a
    /// registered external buffer.
    #[test]
    fn buffer_from_ptr() {
        let mut ctx = Context::new();
        let data: std::sync::Arc<Vec<u8>> = std::sync::Arc::new((0u8..=255).collect());
        let t = ctx.new_tensor_1d(GgmlType::I8, 256);
        ctx.set_external_storage(t, data, 0);
        let ext_index = match ctx.tensors[t.0 as usize].storage {
            Storage::External { offset, .. } => offset >> 48,
            _ => unreachable!(),
        };
        let buf = backend_cpu_buffer_from_ptr(&ctx, ext_index, 0, 256);
        assert_eq!(backend_buffer_name(&buf), "CPU_Mapped");
        assert!(backend_buft_is_host(&backend_buffer_get_type(&buf)));
        backend_tensor_attach_buffer(&ctx, t, buf);
        // reads go through the external storage
        let mut out = [0u8; 4];
        backend_tensor_get(&ctx, t, &mut out, 0);
        assert_eq!(out, [0, 1, 2, 3]);
    }

    /// device supports_op checks ported from ggml-cpu.cpp:424-491
    #[test]
    fn cpu_supports_op() {
        let dev = cpu_reg_get_device(&cpu_backend_reg(), 0).unwrap();
        let mut ctx = Context::new();
        // view-ish ops always supported
        let a = ctx.new_tensor_1d(GgmlType::F32, 8);
        ctx.tensors[a.0 as usize].op = GgmlOp::Reshape;
        assert!(backend_dev_supports_op(&dev, &ctx, a));
        // mul_mat: F32 x F32 ok (w [8,2] x x [8,3] -> [3,2])
        let w = ctx.new_tensor_2d(GgmlType::F32, 8, 2);
        let x = ctx.new_tensor_2d(GgmlType::F32, 8, 3);
        let mm = ctx.mul_mat(w, x);
        assert!(backend_dev_supports_op(&dev, &ctx, mm));
        // mul_mat with I32 rhs: not supported (vec_dot_type mismatch)
        let xi = ctx.new_tensor_2d(GgmlType::I32, 8, 3);
        let mm2 = ctx.mul_mat(w, xi);
        assert!(!backend_dev_supports_op(&dev, &ctx, mm2));
        // CPY onto IQ1_S: not supported (from_float missing)
        let iq = ctx.new_tensor_2d(GgmlType::Iq1S, 256, 1);
        let cpy = ctx.dup(iq);
        ctx.tensors[cpy.0 as usize].op = GgmlOp::Cpy;
        assert!(!backend_dev_supports_op(&dev, &ctx, cpy));
        // supports_buft: host only (ggml-cpu.cpp:492)
        assert!(backend_dev_supports_buft(&dev, &backend_cpu_buffer_type()));
    }

    /// The DL path (ggml-backend-reg.cpp:220-264 load_backend + :480-572
    /// load_best): fails gracefully with no .so present, and succeeds when
    /// pointed at a stub .so built at test time. The stub mirrors the C-ABI
    /// reg/device layout of sysffi.rs (source also kept in
    /// parity/backend_dl_stub.c) and is compiled with the system cc — no
    /// crate dependency is involved.
    #[test]
    fn backend_dl_stub() {
        let _guard = REGISTRY_TEST_LOCK.lock().unwrap();
        // the cc compiler must be available; skip gracefully otherwise
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        if std::process::Command::new(&cc).arg("--version").output().is_err() {
            eprintln!("skipping: no C compiler");
            return;
        }

        reset_registry_for_tests();

        // graceful failure with no .so present (dlerror path, reg.cpp:222-227)
        assert!(backend_load(Path::new("/nonexistent/dir/libggml-nosuch.so")).is_none());

        let dir = std::env::temp_dir().join(format!("ggml_dl_stub_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let make_stub = |so_name: &str, reg_name: &str, score: Option<i32>| {
            // Some(s) — exports ggml_backend_score returning s (0 = "not
            // supported"); None — no score symbol at all (reg.cpp:229)
            let score_fn = match score {
                Some(s) => format!("int ggml_backend_score(void) {{ return {s}; }}"),
                None => String::new(),
            };
            let src = format!(
                r#"
#include <stddef.h>
// mirrors sysffi.rs GgmlBackendRegI / GgmlBackendReg / GgmlBackendDeviceI /
// GgmlBackendDevice (repr(C) field order)
typedef const char* (*get_name_fn)(const void*);
typedef size_t       (*get_count_fn)(const void*);
typedef void*        (*get_dev_fn)(const void*, size_t);
typedef struct {{ get_name_fn get_name; get_count_fn get_device_count; get_dev_fn get_device; void* (*get_proc_address)(const void*, const char*); }} reg_i_t;
typedef struct {{ int api_version; reg_i_t iface; void* context; }} reg_t;
typedef struct {{ get_name_fn get_name; get_name_fn get_description; void (*get_memory)(const void*, size_t*, size_t*); int (*get_type)(const void*); }} dev_i_t;
typedef struct {{ dev_i_t iface; const reg_t* reg; void* context; }} dev_t;
static const char* reg_name_fn(const void* r) {{ (void)r; return "{reg_name}"; }}
static size_t reg_count_fn(const void* r) {{ (void)r; return 1; }}
static const char* dev0_name_fn(const void* d) {{ (void)d; return "{reg_name}0"; }}
static void dev0_mem_fn(const void* d, size_t* f, size_t* t) {{ (void)d; *f = 1<<30; *t = 1<<30; }}
static int dev0_type_fn(const void* d) {{ (void)d; return 0; }}
static dev_t dev0 = {{ {{ dev0_name_fn, dev0_name_fn, dev0_mem_fn, dev0_type_fn }}, NULL, NULL }};
static void* reg_get_dev_fn(const void* r, size_t i) {{ (void)r; return i == 0 ? (void*)&dev0 : NULL; }}
static reg_t the_reg = {{ 2 /* GGML_BACKEND_API_VERSION */, {{ reg_name_fn, reg_count_fn, reg_get_dev_fn, NULL }}, NULL }};
{score_fn}
void* ggml_backend_init(void) {{ return (void*)&the_reg; }}
"#,
                reg_name = reg_name,
                score_fn = score_fn,
            );
            let c_path = dir.join(format!("{}.c", so_name.trim_end_matches(".so")));
            let so_path = dir.join(so_name);
            std::fs::write(&c_path, src).unwrap();
            let out = std::process::Command::new(&cc)
                .args(["-shared", "-fPIC"])
                .arg("-o")
                .arg(&so_path)
                .arg(&c_path)
                .output()
                .expect("cc invocation");
            assert!(out.status.success(), "stub build failed: {:?}", out);
            so_path
        };

        // --- direct load (ggml_backend_load) ---
        let so = make_stub("libggml-direct.so", "DIRECT", None);
        let reg = backend_load(&so).expect("stub .so must load");
        assert_eq!(backend_reg_name(&reg), "DIRECT");
        assert_eq!(backend_reg_api_version(&reg), GGML_BACKEND_API_VERSION);
        assert_eq!(backend_reg_dev_count(&reg), 1);
        // device enumeration through the FFI wrapper
        let dev = backend_reg_dev_get(&reg, 0).expect("stub device");
        assert_eq!(backend_dev_name(&dev), "DIRECT0");
        assert_eq!(backend_dev_type(&dev), BackendDevType::Cpu);
        assert_eq!(backend_dev_memory(&dev), (1 << 30, 1 << 30));
        // registered globally (reg.cpp:261)
        assert!(backend_reg_by_name("direct").is_some());
        assert_eq!(backend_dev_count(), 2);
        // unload restores the registry (reg.cpp:266-289)
        backend_unload(&reg);
        assert_eq!(backend_reg_count(), 1);
        assert_eq!(backend_dev_count(), 1);

        // --- score gating: score 0 = not supported (reg.cpp:229-235) ---
        let so0 = make_stub("libggml-zero-0.so", "ZERO", Some(0));
        assert!(backend_load(&so0).is_none(), "score 0 backends are rejected");

        // --- load_best: pick the highest score, else the base name ---
        // (ggml-backend-reg.cpp:480-572)
        let _ = make_stub("libggml-vk-5.so", "VK5", Some(5));
        let _ = make_stub("libggml-vk-2.so", "VK2", Some(2));
        let best = backend_load_best("vk", true, Some(dir.to_str().unwrap())).expect("best-scored backend");
        assert_eq!(backend_reg_name(&best), "VK5");
        backend_unload(&best);
        // base fallback when no scored candidate beats 0
        let base = make_stub("libggml-bf.so", "BFBASE", None);
        let got = backend_load_best("bf", true, Some(dir.to_str().unwrap())).expect("base backend");
        assert_eq!(backend_reg_name(&got), "BFBASE");
        let _ = base;
        backend_unload(&got);
        // nothing present -> None
        assert!(backend_load_best("nosuchbackend", true, Some(dir.to_str().unwrap())).is_none());

        std::fs::remove_dir_all(&dir).ok();
        reset_registry_for_tests();
    }

    /// api_version accessor used by the DL test (reads the foreign reg
    /// header — the C check lives in load_backend, reg.cpp:246)
    fn backend_reg_api_version(reg: &BackendRegRef) -> i32 {
        reg.api_version
    }
}
