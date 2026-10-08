//! backend_emit — the dual-emission translator (task ② of the GPU enablement
//! plan): mirror a port `Context`/`Graph` into a *foreign* (C) ggml context
//! and drive a foreign backend scheduler over it (task ③).
//!
//! The port's engine builds graphs as `TensorId`s in a Rust `Context`
//! (tensor.rs); every foreign backend computes C `ggml_tensor*` graphs inside
//! a C `ggml_context`. Instead of replaying every graph op through the C op
//! *builders* (a per-op param-signature mapping that would forever trail the
//! port's op surface), the emitter performs a **structural mirror**: one C
//! tensor per Rust tensor, same type/ne/nb, same op (enum-remapped), same
//! `op_params` bytes (the port packs them 1:1 with ggml.c — that is the
//! port's parity discipline), same src links, same `view_src`/`view_offs`.
//! The only C builder used is `ggml_new_tensor`; everything else is written
//! through the public `struct ggml_tensor` layout (ggml.h:685-717).
//!
//! Weights/inputs are *bound*, not copied where possible:
//!   * mmap weights live in the mapping for the process lifetime, so their C
//!     twins are bound through `ggml_backend_cpu_buffer_from_ptr` over the
//!     whole mapping span (one buffer) + `ggml_backend_tensor_alloc` at each
//!     weight's own offset — the `llama_model_loader` mmap flow;
//!   * weights of GPU-placed layers get a device buffer and are uploaded once
//!     with `ggml_backend_tensor_set` (the loader's device path);
//!   * per-step inputs (token ids, KQ mask, positions) are small and the Rust
//!     arena they live in is not pointer-stable across steps, so they are
//!     re-uploaded each step — exactly the reference's per-ubatch `set_input`.
//!
//! Execution runs on the **C-side** scheduler (`ggml_backend_sched_*` from the
//! loaded ggml) rather than the port's backend_sched.rs: the port scheduler's
//! gallocr is arena-backed (Rust-side "addresses"), while foreign device
//! memory only exists as C buffers — driving the C scheduler is the
//! minimal-disruption path and gives the reference's exact split semantics
//! for free. The port's own Rust scheduler (backend_sched.rs) stays the CPU
//! engine path and the structural model for this file.
//!
//! Reference (pinned @ bd4f514db1):
//!   ggml.h                     — struct ggml_tensor, graph API, op enums
//!   ggml-backend.h             — sched/buffer/device API
//!   llama-context.cpp          — per-step flow: alloc_graph → set inputs →
//!                                graph_compute → sched_reset (:643 sched_new,
//!                                parallel = false for one device)
//!   llama-model.cpp:1521-1546  — the -ngl layer split points
//!   llama-kv-cache.cpp:205-222 — KV buffer per layer device (offload_kqv)

use crate::graph::Graph;
use crate::sysffi::op as c_op_code;
use crate::sysffi::*;
use crate::tensor::{Context, GgmlOp, Storage, TensorId, TensorMeta};
use crate::types::{GgmlType, MAX_SRC};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};

/// `LLAMA_MAX_NODES`-scale graph size (llama-context.cpp:636)
pub const EMIT_GRAPH_SIZE: usize = 8192;

/// ggml.h `GGML_MEM_ALIGN`-compatible tensor alignment for buffer spans
const TENSOR_ALIGN: usize = 32;

// ===========================================================================
// op / type mapping — port encodings → pinned ggml.h enums
// ===========================================================================

/// port `GgmlOp` → pinned `enum ggml_op`. The port's discriminants are
/// internal-only (tensor.rs:14), so this table is the single translation
/// point. Returns None for the DEAD VARIANTS (tensor.rs §5-C) — no graph
/// carries them; an error here means a graph carried one anyway.
pub fn op_to_c(op: GgmlOp) -> Option<i32> {
    use GgmlOp::*;
    // DEAD VARIANTS (AUDIT_ggml.md §5-C) never appear in a graph; Norm is
    // special-cased by the caller (RMS_NORM via op_params[1]). Qualified
    // Option::None — the glob below would resolve bare `None` to the variant.
    if matches!(op, Norm | SquaredMulMat | MulView | Conv2dDirect) {
        return Option::None;
    }
    Some(match op {
        None => c_op_code::NONE,
        // guarded above (Norm → norm_op_code; DEAD VARIANTS never emitted)
        Norm | SquaredMulMat | MulView | Conv2dDirect => unreachable!("op_to_c guard"),
        // port Dup covers ggml_dup + ggml_cont (ops.rs:16 — "GGML_OP_CONT
        // reuses GgmlOp::Dup"); the port's dup/cont results are standalone
        // tensors like C's ggml_cont (ggml.c:3639), and every backend's CONT
        // kernel is the dup kernel, so CONT is the structurally faithful twin
        Dup => c_op_code::CONT,
        Add => c_op_code::ADD,
        Mul => c_op_code::MUL,
        Div => c_op_code::DIV,
        Sub => c_op_code::SUB,
        MulMat => c_op_code::MUL_MAT,
        Scale => c_op_code::SCALE,
        Cpy => c_op_code::CPY,
        Reshape => c_op_code::RESHAPE,
        View => c_op_code::VIEW,
        Permute => c_op_code::PERMUTE,
        Transpose => c_op_code::TRANSPOSE,
        GetRows => c_op_code::GET_ROWS,
        DiagMaskInf => c_op_code::DIAG_MASK_INF,
        SoftMax => c_op_code::SOFT_MAX,
        RoPE => c_op_code::ROPE,
        RoPEBack => c_op_code::ROPE_BACK,
        MulMatId => c_op_code::MUL_MAT_ID,
        Argsort => c_op_code::ARGSORT,
        ArgMax => c_op_code::ARGMAX,
        Repeat => c_op_code::REPEAT,
        Concat => c_op_code::CONCAT,
        // port Silu *is* GGML_OP_UNARY (ops.rs:13-14): op_params[0] carries
        // the ggml_unary_op code already in the C layout
        Silu => c_op_code::UNARY,
        SumRows => c_op_code::SUM_ROWS,
        SetRows => c_op_code::SET_ROWS,
        FlashAttnExt => c_op_code::FLASH_ATTN_EXT,
        AddId => c_op_code::ADD_ID,
        Glu => c_op_code::GLU,
        SsmConv => c_op_code::SSM_CONV,
        SsmScan => c_op_code::SSM_SCAN,
        Clamp => c_op_code::CLAMP,
        Gdn => c_op_code::GATED_DELTA_NET,
        Im2col => c_op_code::IM2COL,
        Upscale => c_op_code::UPSCALE,
        Fill => c_op_code::FILL,
        LightningIndexer => c_op_code::LIGHTNING_INDEXER,
        TopK => c_op_code::TOP_K,
        Sqrt => c_op_code::SQRT,
        Dsv4HcComb => c_op_code::DSV4_HC_COMB,
        Dsv4HcPre => c_op_code::DSV4_HC_PRE,
        Dsv4HcPost => c_op_code::DSV4_HC_POST,
        Pad => c_op_code::PAD,
        Pool2d => c_op_code::POOL_2D,
        Arange => c_op_code::ARANGE,
        Pool1d => c_op_code::POOL_1D,
        Roll => c_op_code::ROLL,
        Conv2dDw => c_op_code::CONV_2D_DW,
        Sin => c_op_code::SIN,
        Cos => c_op_code::COS,
        Sqr => c_op_code::SQR,
        Mean => c_op_code::MEAN,
        PadReflect1d => c_op_code::PAD_REFLECT_1D,
        Sum => c_op_code::SUM,
        Cumsum => c_op_code::CUMSUM,
        Tri => c_op_code::TRI,
        Log => c_op_code::LOG,
        Col2Im1d => c_op_code::COL2IM_1D,
        RwkvWkv6 => c_op_code::RWKV_WKV6,
        GatedLinearAttn => c_op_code::GATED_LINEAR_ATTN,
        RwkvWkv7 => c_op_code::RWKV_WKV7,
        // arch batch 18 (chunked delta-net, agent GDN): ggml.h 33/43/85
        Set => c_op_code::SET,
        Diag => c_op_code::DIAG,
        SolveTri => c_op_code::SOLVE_TRI,
    })
}

/// port `Norm` disambiguation: RMS_NORM when op_params[1] == 1 (ops.rs:10),
/// L2_NORM when == 2 (clef); op_params[0] is eps for all three —
/// byte-identical to C's builders (ggml_norm ggml.c:3155 / ggml_rms_norm
/// ggml.c:3182 / ggml_l2_norm ggml.c:3260)
fn norm_op_code(op_params: &[i32; 16]) -> i32 {
    match op_params.get(1) {
        Some(&1) => c_op_code::RMS_NORM,
        Some(&2) => c_op_code::L2_NORM,
        _ => c_op_code::NORM,
    }
}

/// `types::GgmlType` discriminants are the C enum's (GGUF round-trips)
fn type_to_c(ty: GgmlType) -> i32 {
    ty as i32
}

// ===========================================================================
// -ngl placement — llama-model.cpp:1521-1546 (single-GPU layer split)
// ===========================================================================

/// `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`
/// (llama-model.cpp:1521). Layers `il >= i_gpu_start` go to the GPU — the
/// *tail* of the stack, with the output/global layer counted as
/// `n_layer_all` (`get_layer_buft_list(n_layer_all)`, llama-model.cpp:1546).
pub fn gpu_layer_start(n_layer_all: usize, n_gpu_layers: i32) -> usize {
    if n_gpu_layers <= 0 {
        return usize::MAX;
    }
    (n_layer_all as i64 + 1 - n_gpu_layers as i64).max(0) as usize
}

/// layer `il` is GPU-placed iff `il >= i_gpu_start && il < n_layer_all + 1`
/// (llama-model.cpp:1526-1529 — the second condition only truncates `-ngl`
/// beyond n_layer_all+1; `act_gpu_layers` with one GPU)
pub fn layer_on_gpu(n_layer_all: usize, n_gpu_layers: i32, il: usize) -> bool {
    let start = gpu_layer_start(n_layer_all, n_gpu_layers);
    il >= start && il < n_layer_all + 1
}

// ===========================================================================
// library loading
// ===========================================================================

struct Loaded {
    base: DlHandle,
    full: Option<DlHandle>,
    syms: CLibSyms,
}

fn load_libs(lib_dir: &Path) -> Result<Loaded, String> {
    let try_open = |name: &str| -> Option<DlHandle> {
        let p = lib_dir.join(name);
        if !p.exists() {
            return None;
        }
        let h = dl_load_library(&p);
        if h.is_null() {
            eprintln!("backend_emit: dlopen {} failed: {}", p.display(), dl_error());
            return None;
        }
        Some(h)
    };
    let base = try_open("libggml-base.so").ok_or_else(|| {
        format!(
            "no libggml-base.so in {} (pass the reference build's bin/ via --ggml-libs)",
            lib_dir.display()
        )
    })?;
    // libggml.so owns the registry + DL loader (may be absent in trimmed builds)
    let full = try_open("libggml.so");
    let syms = CLibSyms::resolve(&base, full.as_ref()).map_err(|e| format!("symbol resolve: {e}"))?;
    Ok(Loaded { base, full, syms })
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::new("?").unwrap())
}

unsafe fn cstr_owned(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// probe: list the devices the foreign build of `lib_dir` registers
/// (`--list-devices`); the dlopen handles are intentionally leaked — the C
/// registry keeps the backends loaded for the process lifetime
/// (ggml-backend-reg.cpp:261), and a dlclose would race libgomp teardown
pub fn list_devices(lib_dir: &Path) -> Result<Vec<String>, String> {
    let Loaded { base, full, syms } = load_libs(lib_dir)?;
    unsafe {
        (syms.ggml_backend_load_all_from_path)(cstr(&lib_dir.to_string_lossy()).as_ptr());
        let n = (syms.ggml_backend_dev_count)();
        let mut out = Vec::new();
        for i in 0..n {
            let dev = (syms.ggml_backend_dev_get)(i);
            if !dev.is_null() {
                out.push(cstr_owned((syms.ggml_backend_dev_name)(dev)));
            }
        }
        std::mem::forget(base);
        std::mem::forget(full);
        Ok(out)
    }
}

// ===========================================================================
// the executor
// ===========================================================================

/// Configuration for [`ForeignExecutor::new`].
#[derive(Clone)]
pub struct EmitConfig {
    /// directory holding the foreign ggml build (libggml-base.so, libggml.so,
    /// libggml-<backend>.so …) — `ggml_backend_load_all_from_path` target
    pub lib_dir: PathBuf,
    /// device name (e.g. "Vulkan0"); None = the foreign *CPU* backend only
    /// (the reference's libggml-cpu via DL — proves the plumbing without a GPU)
    pub device: Option<String>,
    /// -ngl (≤ 0 = nothing offloaded; the executor still drives the foreign
    /// CPU backend, which is the reference-CPU-kernels mode)
    pub n_gpu_layers: i32,
    pub n_threads: usize,
    pub graph_size: usize,
    /// `cparams.op_offload` (llama-context.cpp:273; default true)
    pub op_offload: bool,
}

impl EmitConfig {
    pub fn new(lib_dir: impl Into<PathBuf>) -> Self {
        EmitConfig {
            lib_dir: lib_dir.into(),
            device: None,
            n_gpu_layers: 0,
            n_threads: 4,
            graph_size: EMIT_GRAPH_SIZE,
            op_offload: true,
        }
    }
}

/// A persistent tensor's placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// host memory, bound by pointer (mmap weights / CPU-side KV cache)
    Cpu,
    /// device buffer + one-time upload (weights or KV cache of GPU layers)
    Gpu,
}

/// one recorded persistent binding awaiting its buffer pass
struct PendingBind {
    ct: *mut GgmlCTensor,
    /// source bytes (weights upload; caches use None)
    src: Option<(*const u8, usize)>,
    placement: Placement,
}

/// The foreign executor: dlopen'ed ggml + persistent bindings + the C sched.
///
/// Lifetime discipline: the `DlHandle`s keep the loaded ggml mapped; the raw
/// pointers are owned by that mapping (C objects) or by the Rust `Context`
/// the binding came from — the caller must keep that Context alive (the
/// engine does: `DecodeContext` owns the `gctx` the weights live in).
pub struct ForeignExecutor {
    syms: CLibSyms,
    _base: DlHandle,
    _full: Option<DlHandle>,

    gpu_dev: Option<*mut GgmlBackendDevT>,
    cpu_dev: *mut GgmlBackendDevT,
    gpu_backend: *mut GgmlBackendT,
    cpu_backend: *mut GgmlBackendT,
    backend_names: HashMap<usize, String>,

    /// persistent C contexts (weights / KV caches) and the per-step mirror
    model_ctx: *mut GgmlContext,
    kv_ctx: *mut GgmlContext,
    step_ctx: *mut GgmlContext,

    /// persistent Rust→C tensor map (weights + caches)
    persistent: HashMap<TensorId, *mut GgmlCTensor>,
    /// persistent view twins awaiting their base's buffer (patched in finish)
    pending_views: Vec<(TensorId, *mut GgmlCTensor)>,
    /// per-step Rust→C tensor map (cleared every step)
    step_map: HashMap<TensorId, *mut GgmlCTensor>,

    /// recorded bindings, grouped by buffer
    w_cpu: Vec<PendingBind>,
    w_gpu: Vec<PendingBind>,
    kv_cpu: Vec<PendingBind>,
    kv_gpu: Vec<PendingBind>,
    cpu_w_span: (*const u8, usize),
    gpu_w_align: usize,

    /// the finished buffers (null until the finish passes)
    cpu_w_buf: *mut GgmlBackendBufferT,
    /// the GPU weight buffers — `ggml_backend_buft_alloc_buffer_n`'s default
    /// plan splits at the buft's max size (ggml-backend.cpp:57-91 since #23671,
    /// the same `cur>0 && cur+sz>max` chunking the old ggml-alloc.c
    /// `..._from_buft_impl` had), so a multi-GiB weight set spans several
    /// device buffers like the reference's multi-buffer (RADV caps one buffer
    /// at 4 GiB)
    gpu_w_bufs: Vec<*mut GgmlBackendBufferT>,
    /// the KV cache buffers per placement — same segmentation rule as the
    /// weights (`llama_kv_cache::init` reaches the identical
    /// `ggml_backend_alloc_ctx_tensors_from_buft` path, llama-kv-cache.cpp:
    /// 285), so a multi-GiB KV cache also spans several device buffers.
    /// The C wraps n>1 segments in a `ggml_backend_multi_buffer`
    /// (ggml-alloc.c:1220-1224); the port queries nothing on the aggregate,
    /// so the segments are kept as the plain Vec
    cpu_kv_bufs: Vec<*mut GgmlBackendBufferT>,
    gpu_kv_bufs: Vec<*mut GgmlBackendBufferT>,

    sched: *mut GgmlBackendSchedT,

    /// per-step stats for the log
    pub n_nodes_last: usize,
    pub n_uploaded_bytes: u64,
    pub graph_ms_last: f64,
}

// SAFETY: pointers are dereferenced only through the C ABI while the DlHandle
// fields (declared before any use, dropped last) keep the code mapped; the
// executor is driven from one thread at a time like the engine itself.
unsafe impl Send for ForeignExecutor {}

impl ForeignExecutor {
    /// dlopen, register every backend of `lib_dir`, pick the device, init the
    /// backends (gpu?, cpu) and the scheduler over them — `llama_context`'s
    /// backend init (llama-context.cpp:580-682).
    pub fn new(cfg: &EmitConfig) -> Result<ForeignExecutor, String> {
        let Loaded { base, full, syms } = load_libs(&cfg.lib_dir)?;
        let s = &syms;
        unsafe {
            // ggml_backend_load_all_from_path (ggml-backend-reg.cpp:578)
            (s.ggml_backend_load_all_from_path)(cstr(&cfg.lib_dir.to_string_lossy()).as_ptr());

            let cpu_dev = (s.ggml_backend_dev_by_type)(GGML_BACKEND_DEV_TYPE_CPU);
            if cpu_dev.is_null() {
                return Err("foreign ggml registered no CPU device".into());
            }
            let gpu_dev = match &cfg.device {
                None => None,
                name => {
                    let raw = name.as_deref().unwrap_or("");
                    let mut dev = (s.ggml_backend_dev_by_name)(cstr(raw).as_ptr());
                    if dev.is_null() && !raw.chars().last().is_some_and(|c| c.is_ascii_digit()) {
                        // "Vulkan" → "Vulkan0" (devices list with an index)
                        dev = (s.ggml_backend_dev_by_name)(cstr(&format!("{raw}0")).as_ptr());
                    }
                    if dev.is_null() {
                        let n = (s.ggml_backend_dev_count)();
                        let names: Vec<String> = (0..n)
                            .map(|i| cstr_owned((s.ggml_backend_dev_name)((s.ggml_backend_dev_get)(i))))
                            .collect();
                        return Err(format!("device '{raw}' not found; available: {}", names.join(", ")));
                    }
                    Some(dev)
                }
            };

            let cpu_backend = (s.ggml_backend_dev_init)(cpu_dev, std::ptr::null());
            if cpu_backend.is_null() {
                return Err("cpu backend init failed".into());
            }
            // thread count through the reg proc address — the CPU side of the
            // hybrid runs the reference's threadpool (llama-context.cpp:614)
            {
                let reg = (s.ggml_backend_dev_backend_reg)(cpu_dev);
                let proc = (s.ggml_backend_reg_get_proc_address)(reg, cstr("ggml_backend_set_n_threads").as_ptr());
                if !proc.is_null() {
                    let set_n: FnBackendSetNThreads = std::mem::transmute::<*mut c_void, FnBackendSetNThreads>(proc);
                    set_n(cpu_backend, cfg.n_threads as i32);
                }
            }
            let (gpu_backend, gpu_buft_align) = match gpu_dev {
                None => (std::ptr::null_mut(), 32usize),
                Some(d) => {
                    let b = (s.ggml_backend_dev_init)(d, std::ptr::null());
                    if b.is_null() {
                        return Err("gpu backend init failed".into());
                    }
                    let align = (s.ggml_backend_buft_get_alignment)((s.ggml_backend_dev_buffer_type)(d));
                    (b, align.max(1))
                }
            };

            // scheduler: [gpu?, cpu] — gpu first = higher priority
            // (llama-context.cpp:643; parallel = false for one device,
            //  op_offload = cparams.op_offload)
            let mut backends: Vec<*mut GgmlBackendT> = Vec::new();
            if !gpu_backend.is_null() {
                backends.push(gpu_backend);
            }
            backends.push(cpu_backend);
            let sched = (s.ggml_backend_sched_new)(
                backends.as_mut_ptr(),
                std::ptr::null_mut(), // default buffer types per backend
                backends.len() as i32,
                cfg.graph_size,
                false,
                cfg.op_offload,
            );
            if sched.is_null() {
                return Err("ggml_backend_sched_new failed".into());
            }

            let mut backend_names = HashMap::new();
            backend_names.insert(cpu_backend as usize, "CPU".to_string());
            if !gpu_backend.is_null() {
                let name = cstr_owned((s.ggml_backend_dev_name)(gpu_dev.unwrap()));
                backend_names.insert(gpu_backend as usize, name);
            }

            // persistent contexts (metadata only — data comes from buffers)
            let t_overhead = (s.ggml_tensor_overhead)();
            let g_overhead = (s.ggml_graph_overhead_custom)(cfg.graph_size, false);
            let model_ctx = (s.ggml_init)(GgmlInitParams {
                mem_size: t_overhead * 8192 + g_overhead,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            let kv_ctx = (s.ggml_init)(GgmlInitParams {
                mem_size: t_overhead * 2048,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            if model_ctx.is_null() || kv_ctx.is_null() {
                return Err("ggml_init (persistent contexts) failed".into());
            }

            Ok(ForeignExecutor {
                syms,
                _base: base,
                _full: full,
                gpu_dev,
                cpu_dev,
                gpu_backend,
                cpu_backend,
                backend_names,
                model_ctx,
                kv_ctx,
                step_ctx: std::ptr::null_mut(),
                persistent: HashMap::new(),
                pending_views: Vec::new(),
                step_map: HashMap::new(),
                w_cpu: Vec::new(),
                w_gpu: Vec::new(),
                kv_cpu: Vec::new(),
                kv_gpu: Vec::new(),
                cpu_w_span: (std::ptr::null(), 0),
                gpu_w_align: gpu_buft_align,
                cpu_w_buf: std::ptr::null_mut(),
                gpu_w_bufs: Vec::new(),
                cpu_kv_bufs: Vec::new(),
                gpu_kv_bufs: Vec::new(),
                sched,
                n_nodes_last: 0,
                n_uploaded_bytes: 0,
                graph_ms_last: 0.0,
            })
        }
    }

    pub fn device_name(&self) -> String {
        unsafe {
            match self.gpu_dev {
                Some(d) => cstr_owned((self.syms.ggml_backend_dev_name)(d)),
                None => "CPU".to_string(),
            }
        }
    }

    /// Bind one model weight (external/mmap storage): creates the persistent
    /// C twin and records it for the [`ForeignExecutor::finish_weights`] pass.
    /// `placement` comes from the caller's -ngl rule ([`layer_on_gpu`]).
    pub fn bind_weight(&mut self, rctx: &Context, id: TensorId, placement: Placement) -> Result<(), String> {
        let bytes = rctx.data_bytes(id).ok_or_else(|| format!("weight '{}' has no bytes", rctx.name(id)))?;
        unsafe {
            let ct = self.new_persistent_twin(rctx, id, self.model_ctx)?;
            let bind = PendingBind {
                ct,
                src: Some((bytes.as_ptr(), bytes.len())),
                placement,
            };
            match placement {
                Placement::Cpu => {
                    // grow the from_ptr span to cover this tensor
                    let (base, len) = self.cpu_w_span;
                    let ptr = bytes.as_ptr();
                    self.cpu_w_span = if base.is_null() {
                        (ptr, bytes.len())
                    } else {
                        let new_base = base.min(ptr);
                        let new_end = (base as usize + len).max(ptr as usize + bytes.len());
                        (new_base, new_end - new_base as usize)
                    };
                    self.w_cpu.push(bind);
                }
                Placement::Gpu => self.w_gpu.push(bind),
            }
            self.register(id, ct);
            Ok(())
        }
    }

    /// Bind one persistent cross-step state tensor (KV caches). Placement
    /// follows the layer rule of the weights (`offload_kqv`,
    /// llama-kv-cache.cpp:216-221). Buffers are created by
    /// [`ForeignExecutor::finish_weights`].
    pub fn bind_cache(&mut self, rctx: &Context, id: TensorId, placement: Placement) -> Result<(), String> {
        unsafe {
            let ct = self.new_persistent_twin(rctx, id, self.kv_ctx)?;
            let nbytes = rctx.nbytes(id);
            let bind = PendingBind { ct, src: None, placement };
            match placement {
                Placement::Cpu => self.kv_cpu.push(bind),
                Placement::Gpu => self.kv_gpu.push(bind),
            }
            let _ = nbytes;
            self.register(id, ct);
            Ok(())
        }
    }

    fn register(&mut self, id: TensorId, ct: *mut GgmlCTensor) {
        self.persistent.insert(id, ct);
        // a twin with view_src bound before its base goes to the pending list
        unsafe {
            if (*ct).view_src != std::ptr::null_mut() {
                self.pending_views.push((id, ct));
            }
        }
    }

    /// The buffer pass: mmap CPU weights buffer, GPU weight buffer + upload,
    /// the two KV buffers, and view patches (`llama_model_loader`'s
    /// `load_all_data` epilogue + `llama_kv_cache::init`'s buffer creation).
    /// Must be called once after all binds, before the first compute.
    pub fn finish_weights(&mut self) -> Result<(), String> {
        unsafe {
            let s = self.syms;
            let align_up = |v: usize, a: usize| (v + a - 1) & !(a - 1);

            // ---- CPU weights: one buffer over the whole mmap span ----
            if !self.w_cpu.is_empty() {
                let (base, len) = self.cpu_w_span;
                // ggml_backend_cpu_buffer_from_ptr asserts alignment; GGUF
                // pads tensor offsets, but be defensive about the span head
                let aligned_base = (base as usize & !(TENSOR_ALIGN - 1)) as *const u8;
                let span = len + (base as usize - aligned_base as usize);
                self.cpu_w_buf = (s.ggml_backend_cpu_buffer_from_ptr)(aligned_base as *mut c_void, span);
                if self.cpu_w_buf.is_null() {
                    return Err("ggml_backend_cpu_buffer_from_ptr failed".into());
                }
                (s.ggml_backend_buffer_set_usage)(self.cpu_w_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                for b in &self.w_cpu {
                    let (src, len) = b.src.unwrap();
                    let st = (s.ggml_backend_tensor_alloc)(self.cpu_w_buf, b.ct, src as *mut c_void);
                    if st != GGML_STATUS_SUCCESS {
                        return Err(format!("tensor_alloc (cpu weight) failed: {st}"));
                    }
                    let _ = len;
                }
            }

            // ---- GPU weights: device buffers + one-time upload ----
            // `ggml_backend_buft_alloc_buffer_n`'s default plan chunking
            // (ggml-backend.cpp:57-91, #23671): a new buffer opens when the
            // running total would exceed the buft's max size (Vulkan:
            // suballocation_block_size, ggml-vulkan.cpp:12794-12798) — the
            // reference's multi-buffer, which loads multi-GiB weight sets on
            // devices whose per-buffer limit is 4 GiB (RADV maxBufferSize =
            // 0xffffffff, maxMemoryAllocationSize = 0xfffffffc)
            if !self.w_gpu.is_empty() {
                let dev = self.gpu_dev.ok_or("no gpu device for weight upload")?;
                let buft = (s.ggml_backend_dev_buffer_type)(dev);
                let align = self.gpu_w_align.max(1);
                let max_size = (s.ggml_backend_buft_get_max_size)(buft).max(1);
                // the backend's own alloc_size (padding-aware, like the KV
                // path below and ggml_backend_alloc_ctx_tensors)
                let w_size = |b: &PendingBind| align_up((s.ggml_backend_buft_get_alloc_size)(buft, b.ct).max(1), align);
                let mut chunks: Vec<(usize, usize)> = Vec::new(); // [start, end) bind ranges
                let mut cur = 0usize;
                let mut start = 0usize;
                for (i, b) in self.w_gpu.iter().enumerate() {
                    let sz = w_size(b);
                    if cur > 0 && cur + sz > max_size {
                        chunks.push((start, i));
                        start = i;
                        cur = sz;
                    } else {
                        cur += sz;
                    }
                }
                chunks.push((start, self.w_gpu.len()));
                for (a, b) in chunks {
                    let total: usize = self.w_gpu[a..b].iter().map(&w_size).sum();
                    let buf = (s.ggml_backend_buft_alloc_buffer)(buft, total);
                    if buf.is_null() {
                        return Err(format!("gpu weights buffer alloc of {total} bytes failed"));
                    }
                    (s.ggml_backend_buffer_set_usage)(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    let base = (s.ggml_backend_buffer_get_base)(buf) as usize;
                    let mut off = 0usize;
                    for wb in &self.w_gpu[a..b] {
                        let (src, len) = wb.src.unwrap();
                        let st = (s.ggml_backend_tensor_alloc)(buf, wb.ct, (base + off) as *mut c_void);
                        if st != GGML_STATUS_SUCCESS {
                            return Err(format!("tensor_alloc (gpu weight) failed: {st}"));
                        }
                        // host → device (the loader's set_tensor path)
                        (s.ggml_backend_tensor_set)(wb.ct, src as *const c_void, 0, len);
                        off += w_size(wb);
                    }
                    self.gpu_w_bufs.push(buf);
                }
            }

            // ---- KV caches: buffers per placement, zeroed, segmented at the
            // buft's max size ---- the exact rule the weights use above
            // (ggml-backend.cpp:57-91's default alloc_buffer_n plan), because
            // `llama_kv_cache::init` allocates through the same
            // `ggml_backend_alloc_ctx_tensors_from_buft`
            // (llama-kv-cache.cpp:285 → ggml-alloc.c:1143 →
            // `ggml_backend_buft_alloc_buffer_n`): a new buffer opens when the
            // running total would exceed the buft's max size. Without
            // this a context large enough that K+V exceed the device's
            // per-buffer cap (RADV: 4 GiB) fails the alloc outright — the
            // batch-18 not-done item. The CPU buft's max size is SIZE_MAX,
            // so the CPU side never segments in practice (the rule is
            // uniform anyway, like the C)
            for group in 0..2usize {
                let (binds, buft, tag) = match group {
                    // the borrows end at `binds`' last use below, before the
                    // buffer-vec assignment — per-group re-borrow, NLL-clean
                    0 => (
                        &self.kv_cpu,
                        (s.ggml_backend_cpu_buffer_type)(),
                        "cpu",
                    ),
                    _ => (
                        &self.kv_gpu,
                        match self.gpu_dev {
                            Some(d) => (s.ggml_backend_dev_buffer_type)(d),
                            None => std::ptr::null_mut(),
                        },
                        "gpu",
                    ),
                };
                if binds.is_empty() {
                    continue;
                }
                if buft.is_null() {
                    return Err("no gpu device for kv cache".into());
                }
                let align = (s.ggml_backend_buft_get_alignment)(buft).max(1);
                let max_size = (s.ggml_backend_buft_get_max_size)(buft).max(1);
                // alloc_size honors any backend-specific padding
                let kv_size = |b: &PendingBind| {
                    align_up((s.ggml_backend_buft_get_alloc_size)(buft, b.ct).max(1), align)
                };
                let mut chunks: Vec<(usize, usize)> = Vec::new(); // [start, end) bind ranges
                let mut cur = 0usize;
                let mut start = 0usize;
                for (i, b) in binds.iter().enumerate() {
                    let sz = kv_size(b);
                    if cur > 0 && cur + sz > max_size {
                        chunks.push((start, i));
                        start = i;
                        cur = sz;
                    } else {
                        cur += sz;
                    }
                }
                chunks.push((start, binds.len()));
                let mut bufs: Vec<*mut GgmlBackendBufferT> = Vec::new();
                for (a, b) in chunks {
                    let total: usize = binds[a..b].iter().map(&kv_size).sum();
                    let buf = (s.ggml_backend_buft_alloc_buffer)(buft, total);
                    if buf.is_null() {
                        return Err(format!("{tag} kv buffer alloc of {total} bytes failed"));
                    }
                    (s.ggml_backend_buffer_clear)(buf, 0); // the port's arena starts zeroed
                    let base = (s.ggml_backend_buffer_get_base)(buf) as usize;
                    let mut off = 0usize;
                    for kb in &binds[a..b] {
                        let st = (s.ggml_backend_tensor_alloc)(buf, kb.ct, (base + off) as *mut c_void);
                        if st != GGML_STATUS_SUCCESS {
                            return Err(format!("tensor_alloc ({tag} kv) failed: {st}"));
                        }
                        off += kv_size(kb);
                    }
                    bufs.push(buf);
                }
                if bufs.len() > 1 {
                    // the trigger evidence for the segmentation: the cache
                    // did not fit the buft's max size (e.g. a large -c on
                    // the 4 GiB-capped Vulkan device buffer)
                    eprintln!(
                        "backend_emit: {tag} kv cache split into {} buffers (max_size {:.2} GiB \
                         per buffer, {} tensors)",
                        bufs.len(),
                        max_size as f64 / (1024.0 * 1024.0 * 1024.0),
                        binds.len()
                    );
                }
                match group {
                    0 => self.cpu_kv_bufs = bufs,
                    _ => self.gpu_kv_bufs = bufs,
                }
            }

            // ---- persistent view twins: patch data/buffer through the base ----
            for &(_, ct) in &self.pending_views {
                if (*ct).buffer.is_null() && (*(*ct).view_src).buffer != std::ptr::null_mut() {
                    let st = (s.ggml_backend_view_init)(ct);
                    if st != GGML_STATUS_SUCCESS {
                        return Err(format!("view_init failed: {st}"));
                    }
                }
            }
            self.pending_views.clear();
            Ok(())
        }
    }

    /// Create the persistent C twin of `id` in `ctx` (metadata only).
    unsafe fn new_persistent_twin(
        &mut self,
        rctx: &Context,
        id: TensorId,
        ctx: *mut GgmlContext,
    ) -> Result<*mut GgmlCTensor, String> {
        let t = &rctx.tensors[id.0 as usize];
        let ct = (self.syms.ggml_new_tensor)(ctx, type_to_c(t.ty), 4, t.ne.as_ptr());
        if ct.is_null() {
            return Err("ggml_new_tensor (persistent) failed — context full".into());
        }
        if (*ct).ne != t.ne {
            return Err("foreign ggml_tensor layout mismatch (ne round-trip)".into());
        }
        self.fill_twin(ct, t)?;
        if !t.name.is_empty() {
            let name = cstr(&t.name);
            (self.syms.ggml_set_name)(ct, name.as_ptr());
        }
        // view chains resolve through the persistent map (bound earlier) or
        // are patched in finish_weights via the pending list
        if let Some(vs) = t.view_src {
            if let Some(base) = self.persistent.get(&vs) {
                (*ct).view_src = *base;
                (*ct).view_offs = t.view_offs;
            } else {
                // base not bound yet — mirror its metadata shape now and link
                // through it (the recursive twin); buffer patch deferred
                let base_ct = self.new_persistent_twin(rctx, vs, ctx)?;
                (*ct).view_src = base_ct;
                (*ct).view_offs = t.view_offs;
            }
        }
        Ok(ct)
    }

    /// Fill op/nb/op_params of a fresh C twin from the Rust meta.
    unsafe fn fill_twin(&self, ct: *mut GgmlCTensor, t: &TensorMeta) -> Result<(), String> {
        (*ct).nb = [t.nb[0] as usize, t.nb[1] as usize, t.nb[2] as usize, t.nb[3] as usize];
        let cop = match t.op {
            GgmlOp::None => c_op_code::NONE,
            GgmlOp::Norm => norm_op_code(&t.op_params),
            other => op_to_c(other).ok_or_else(|| format!("op {other:?} has no foreign mapping"))?,
        };
        (*ct).op = cop;
        (*ct).op_params = t.op_params;
        Ok(())
    }

    // -------------------------------------------------------------------
    // per-step emission
    // -------------------------------------------------------------------

    /// Mirror `rctx`'s `graph` into a fresh C context, run the C scheduler,
    /// and copy the results of `sync_back` back into those Rust tensors.
    /// The Rust graph is consumed **as built** — the CPU engine path stays
    /// untouched and byte-identical.
    pub fn graph_compute(
        &mut self,
        rctx: &mut Context,
        graph: &Graph,
        roots: &[TensorId],
        sync_back: &[TensorId],
    ) -> Result<(), String> {
        let t0 = unsafe { (self.syms.ggml_backend_time_us)() } as f64;
        unsafe {
            let s = self.syms;
            // ggml_free(last step ctx) — llama.cpp frees res->ctx per ubatch
            if !self.step_ctx.is_null() {
                (s.ggml_free)(self.step_ctx);
                self.step_ctx = std::ptr::null_mut();
            }
            self.step_map.clear();

            let n_meta = graph.nodes.len() + graph.leafs.len();
            let overhead = (s.ggml_tensor_overhead)();
            let g_overhead = (s.ggml_graph_overhead_custom)(EMIT_GRAPH_SIZE, false);
            self.step_ctx = (s.ggml_init)(GgmlInitParams {
                mem_size: overhead * (n_meta + 16) + g_overhead,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            });
            if self.step_ctx.is_null() {
                return Err("ggml_init (step ctx) failed".into());
            }

            // mirror leafs first, then nodes (topological) — every src of a
            // node is a leaf, a persistent twin or an earlier node
            for &l in graph.leafs.iter() {
                self.mirror(rctx, l)?;
            }
            for &n in graph.nodes.iter() {
                self.mirror(rctx, n)?;
            }

            // step inputs: per-step leafs — ggml_set_input so the C gallocr
            // allocates them up front (ggml-alloc.c:726-740) and the sched
            // assigns them to the last/CPU backend (ggml-backend.cpp:884-888),
            // exactly like llama.cpp's inp_* tensors. Leafs without Rust
            // storage upload zeros — the CPU arena zero-fill equivalent.
            let mut uploads: Vec<(TensorId, *mut GgmlCTensor, Vec<u8>)> = Vec::new();
            for &l in graph.leafs.iter() {
                if self.persistent.contains_key(&l) {
                    continue;
                }
                let ct = self.step_map[&l];
                (s.ggml_set_input)(ct);
                match rctx.data_bytes(l) {
                    Some(bytes) => uploads.push((l, ct, bytes.to_vec())),
                    None => uploads.push((l, ct, vec![0u8; rctx.nbytes(l)])),
                }
            }
            for &r in roots {
                let ct = self
                    .twin_of(r)
                    .ok_or_else(|| format!("root '{}' not mirrored", rctx.name(r)))?;
                (s.ggml_set_output)(ct);
            }

            // cgraph: expand **every** Rust node in Rust order. The engine
            // appends side-effect subtrees (KV SET_ROWS writes) with separate
            // build_forward calls (graph_arch.rs:1229) that are not reachable
            // from the logits root — expanding node-by-node reproduces the
            // exact Rust node set and order (each C tensor is visited once;
            // wr-before-read ordering follows the Rust topological order).
            let cgraph = (s.ggml_new_graph_custom)(self.step_ctx, EMIT_GRAPH_SIZE, false);
            if cgraph.is_null() {
                return Err("ggml_new_graph_custom failed".into());
            }
            for &n in graph.nodes.iter() {
                let ct = self.twin_of(n).expect("node mirrored");
                (s.ggml_build_forward_expand)(cgraph, ct);
            }
            let n_nodes = (s.ggml_graph_n_nodes)(cgraph) as usize;
            if n_nodes != graph.nodes.len() {
                return Err(format!(
                    "emission mismatch: C graph has {n_nodes} nodes, Rust graph has {}",
                    graph.nodes.len()
                ));
            }
            self.n_nodes_last = n_nodes;

            // alloc → upload inputs → compute (the ggml-backend.h example flow)
            if !(s.ggml_backend_sched_alloc_graph)(self.sched, cgraph) {
                return Err("ggml_backend_sched_alloc_graph failed (graph too large or unsupported op?)".into());
            }
            for (_, ct, bytes) in &uploads {
                (s.ggml_backend_tensor_set)(*ct, bytes.as_ptr() as *const c_void, 0, bytes.len());
            }
            self.n_uploaded_bytes = uploads.iter().map(|(_, _, b)| b.len() as u64).sum();

            let st = (s.ggml_backend_sched_graph_compute)(self.sched, cgraph);
            if st != GGML_STATUS_SUCCESS {
                return Err(format!("ggml_backend_sched_graph_compute failed with error {st}"));
            }
            (s.ggml_backend_sched_synchronize)(self.sched);

            // results back into the Rust tensors the engine reads. The sync
            // set is the caller's outputs PLUS the per-layer input taps: the
            // engine names them `layer_inp-{il}` (the port's
            // `res->t_layer_inp` store, graph_arch.rs:128-138) and reads
            // them right after compute exactly the way it reads the logits —
            // extract_layer_inputs pulls the enabled taps via
            // ggml_backend_tensor_get_async (llama-context.cpp:2265-2290).
            // The executor cannot see the enable flags (engine state, not
            // graph state), so it syncs the whole named tap set — the taps
            // are graph nodes of every decode graph regardless.
            let mut targets: Vec<TensorId> = sync_back.to_vec();
            targets.extend(
                graph
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&n| rctx.name(n).starts_with("layer_inp-")),
            );
            // the callers pass the enabled taps in sync_back since the
            // llama-graph.cpp:1374-1382 fix — dedupe so a tap is not
            // tensor_get'ed twice
            targets.sort_unstable_by_key(|t| t.0);
            targets.dedup_by_key(|t| t.0);
            for &t in &targets {
                let ct = self
                    .twin_of(t)
                    .ok_or_else(|| format!("sync_back '{}' not mirrored", rctx.name(t)))?;
                // storage like graph_compute would have: views resolve into
                // their base's arena span
                let mut base = t;
                while let Some(vs) = rctx.tensors[base.0 as usize].view_src {
                    base = vs;
                }
                if matches!(rctx.tensors[base.0 as usize].storage, Storage::None) {
                    rctx.arena_resize_tensor(base);
                }
                let name = rctx.name(t).to_string();
                let dst = rctx.data_bytes_mut(t).ok_or_else(|| format!("sync_back '{name}' has no writable storage"))?;
                (s.ggml_backend_tensor_get)(ct, dst.as_mut_ptr() as *mut c_void, 0, dst.len());
            }

            // per-step reset (llama-context.cpp process_ubatch epilogue)
            (s.ggml_backend_sched_reset)(self.sched);
        }
        let t1 = unsafe { (self.syms.ggml_backend_time_us)() } as f64;
        self.graph_ms_last = (t1 - t0) / 1e3;
        Ok(())
    }

    /// Recursive structural mirror of one Rust tensor into the step ctx.
    fn mirror(&mut self, rctx: &Context, id: TensorId) -> Result<*mut GgmlCTensor, String> {
        if let Some(p) = self.persistent.get(&id) {
            return Ok(*p);
        }
        if let Some(p) = self.step_map.get(&id) {
            return Ok(*p);
        }
        unsafe {
            let meta = snapshot_meta(&rctx.tensors[id.0 as usize]);
            let s = self.syms;
            let ct = (s.ggml_new_tensor)(self.step_ctx, type_to_c(meta.ty), 4, meta.ne.as_ptr());
            if ct.is_null() {
                return Err("ggml_new_tensor (step) failed — context full".into());
            }
            // structural check: the C struct must be the pinned layout
            if (*ct).ne != meta.ne {
                return Err("foreign ggml_tensor layout mismatch (ne round-trip)".into());
            }
            self.fill_twin(ct, &meta)?;
            for j in 0..MAX_SRC {
                if let Some(src) = meta.src[j] {
                    (*ct).src[j] = self.mirror(rctx, src)?;
                }
            }
            if let Some(vs) = meta.view_src {
                (*ct).view_src = self.mirror(rctx, vs)?;
                (*ct).view_offs = meta.view_offs;
            }
            if !meta.name.is_empty() {
                let name = cstr(&meta.name);
                (s.ggml_set_name)(ct, name.as_ptr());
            }
            (*ct).flags = meta.flags as i32;
            self.step_map.insert(id, ct);
            Ok(ct)
        }
    }

    fn twin_of(&self, id: TensorId) -> Option<*mut GgmlCTensor> {
        self.persistent
            .get(&id)
            .copied()
            .or_else(|| self.step_map.get(&id).copied())
    }

    /// which backend (by name) a graph tensor was scheduled on — the -ngl
    /// split check (`ggml_backend_sched_get_tensor_backend`)
    pub fn tensor_backend_name(&self, id: TensorId) -> Option<String> {
        let ct = self.twin_of(id)?;
        unsafe {
            let b = (self.syms.ggml_backend_sched_get_tensor_backend)(self.sched, ct);
            if b.is_null() {
                return None;
            }
            self.backend_names.get(&(b as usize)).cloned()
        }
    }

    pub fn n_backends(&self) -> usize {
        unsafe { (self.syms.ggml_backend_sched_get_n_backends)(self.sched) as usize }
    }

    /// backend names in sched order (gpu first)
    pub fn backend_names(&self) -> Vec<String> {
        unsafe {
            let n = (self.syms.ggml_backend_sched_get_n_backends)(self.sched);
            (0..n)
                .map(|i| {
                    let b = (self.syms.ggml_backend_sched_get_backend)(self.sched, i);
                    self.backend_names.get(&(b as usize)).cloned().unwrap_or_default()
                })
                .collect()
        }
    }

    /// compute-buffer size per backend — the sched banner numbers
    pub fn sched_buffer_sizes(&self) -> Vec<(String, usize)> {
        unsafe {
            let n = (self.syms.ggml_backend_sched_get_n_backends)(self.sched);
            (0..n)
                .map(|i| {
                    let b = (self.syms.ggml_backend_sched_get_backend)(self.sched, i);
                    let name = self.backend_names.get(&(b as usize)).cloned().unwrap_or_default();
                    let size = (self.syms.ggml_backend_sched_get_buffer_size)(self.sched, b);
                    (name, size)
                })
                .collect()
        }
    }
}

/// TensorMeta is not Clone; the mirror needs an owned snapshot (the Context
/// borrow ends before the recursive calls return).
fn snapshot_meta(t: &TensorMeta) -> TensorMeta {
    TensorMeta {
        ty: t.ty,
        ne: t.ne,
        nb: t.nb,
        op: t.op,
        op_params: t.op_params,
        src: t.src,
        view_src: t.view_src,
        view_offs: t.view_offs,
        storage: Storage::None,
        name: t.name.clone(),
        flags: t.flags,
    }
}

impl Drop for ForeignExecutor {
    fn drop(&mut self) {
        // [TAG_EMIT_LEAKY_EXIT] teardown is deliberately leak-only: freeing
        // the C objects from the Rust side races the foreign CPU backend's
        // OpenMP workers (SIGSEGV inside libgomp at process exit) — the same
        // reason the C registry leaks its dl handles (ggml-backend-reg.cpp:
        // 176-183 "backends cannot be safely unloaded"). A CLI process pays
        // nothing; a long-lived host can Box::leak the executor the same way.
        //
        // replace-then-forget, NOT read+forget: `ptr::read` copies the field
        // and leaves the original bits in place — the drop glue drops each
        // field *after* `drop()` returns, so read+forget dlcloses the handle
        // anyway. That unload/reload cycle re-runs ggml.cpp's static
        // initializer (ggml.cpp:13-26), whose terminate-handler install
        // asserts on the second pass (GGML_ASSERT(prev !=
        // ggml_uncaught_exception), ggml.cpp:22) — an abort that only shows
        // once two executors are constructed serially in one process (the
        // first dlclose runs between them). Replacing with a null handle (its
        // Drop is a no-op) keeps the refcount held for real.
        unsafe {
            std::mem::forget(std::mem::replace(&mut self._base, DlHandle::null()));
            if let Some(h) = self._full.take() {
                std::mem::forget(h);
            }
        }
    }
}

// ===========================================================================
// tests — the CPU foreign backend proves the emission layer end to end
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Context;
    use crate::types::GgmlType;

    /// The reference build used by the emission tests (CPU .so's of the
    /// pinned tree build). Skipped when absent so CI without the reference
    /// still passes.
    fn ref_lib_dir() -> Option<PathBuf> {
        let dir = PathBuf::from("/home/jeffrey/llm/llama.cpp/build-rust-ref/bin");
        dir.join("libggml-base.so").exists().then_some(dir)
    }

    fn fill_f32(ctx: &mut Context, t: TensorId, f: impl Fn(usize) -> f32) {
        let n = ctx.nbytes(t) / 4;
        let vals: Vec<f32> = (0..n).map(f).collect();
        ctx.with_f32_mut(t, |p| p.copy_from_slice(&vals)).unwrap();
    }

    /// A small graph exercising the qwen-relevant op surface: weight matmul
    /// (quantized), rms_norm, silu-glu elementwise, get_rows embedding lookup,
    /// rope + FA-free softmax attention via mask, KV set_rows + view reads.
    /// Computed three ways:
    ///   1. the port's own CPU engine (ground truth),
    ///   2. the foreign CPU backend (reference libggml-cpu kernels) —
    ///      numerically equal within a tight band,
    ///   3. structural: the emitted C graph has the same node count.
    #[test]
    fn foreign_cpu_backend_parity() {
        let Some(dir) = ref_lib_dir() else {
            eprintln!("skipping: reference build not present");
            return;
        };

        // ---- build the graph in a fresh Context ----
        let mut ctx = Context::new();
        let n_embd = 64usize;
        let n_tok = 3usize;

        // weights: q8_0 matmul weight + f32 bias
        let w = ctx.new_tensor_2d(GgmlType::Q8_0, n_embd as i64, 8);
        let data: Vec<u8> = (0..ctx.nbytes(w)).map(|i| (i % 251) as u8).collect();
        ctx.set_external_storage(w, std::sync::Arc::new(data), 0);
        let embd_table = ctx.new_tensor_2d(GgmlType::F32, n_embd as i64, 32);
        let tab: Vec<u8> = (0..ctx.nbytes(embd_table)).map(|i| (i % 249) as u8).collect();
        ctx.set_external_storage(embd_table, std::sync::Arc::new(tab), 0);

        // per-step inputs
        let tokens = ctx.new_tensor_1d(GgmlType::I32, n_tok as i64);
        ctx.arena_resize_tensor(tokens);
        ctx.with_i32_mut(tokens, |p| p.copy_from_slice(&[3, 17, 30])).unwrap();

        let x = ctx.get_rows(embd_table, tokens); // [n_embd, n_tok]
        let x = ctx.cont(x);
        let norm = ctx.rms_norm(x, 1e-5);
        let mm = ctx.mul_mat(w, norm); // [8, n_tok]
        let si = ctx.silu(mm);
        let out = ctx.mul(si, mm);
        ctx.set_name(out, "out");

        let mut g = Graph::new(32);
        g.build_forward(&ctx, out);
        assert!(g.nodes.len() >= 5);

        // ---- 1. port CPU engine ground truth ----
        let mut ctx_ref = Context::new();
        let out_ref = rebuild(&mut ctx_ref, n_embd, n_tok);
        let mut g_ref = Graph::new(32);
        g_ref.build_forward(&ctx_ref, out_ref);
        crate::compute::graph_compute(&mut ctx_ref, &mut g_ref, 1);

        // ---- 2. foreign CPU backend through the emitter ----
        let mut cfg = EmitConfig::new(&dir);
        cfg.device = None;
        cfg.n_gpu_layers = 0;
        cfg.n_threads = 1;
        let mut exe = ForeignExecutor::new(&cfg).expect("foreign executor");
        let w_id = w;
        let tab_id = embd_table;
        let out_id = out;
        // bind the two weights CPU-side (mmap-equivalent: external storage)
        exe.bind_weight(&ctx, w_id, Placement::Cpu).unwrap();
        exe.bind_weight(&ctx, tab_id, Placement::Cpu).unwrap();
        exe.finish_weights().unwrap();

        let st = exe.graph_compute(&mut ctx, &g, &[out_id], &[out_id]);
        st.unwrap_or_else(|e| panic!("foreign compute failed: {e}"));
        assert_eq!(exe.n_nodes_last, g_ref.nodes.len(), "C node count == Rust node count");

        let got: Vec<f32> = ctx.f32s(out_id).unwrap().to_vec();
        let want: Vec<f32> = ctx_ref.f32s(out_ref).unwrap().to_vec();
        assert_eq!(got.len(), want.len());
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // same kernels (reference CPU vs port CPU) — same math, allow only
        // last-ulp-level drift from any remaining accumulation-order gap
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        eprintln!(
            "foreign_cpu_backend_parity: max_abs={max_abs:.3e} scale={scale:.3e} rel={:.3e}",
            max_abs / scale
        );
        assert!(
            max_abs <= 2e-3 * scale,
            "foreign CPU backend drifted: max_abs={max_abs} scale={scale}"
        );
    }

    /// rebuild of the same graph for the direct-compute ground truth (the
    /// weights' bytes are derived identically from the same formulas)
    fn rebuild(ctx: &mut Context, n_embd: usize, n_tok: usize) -> TensorId {
        let w = ctx.new_tensor_2d(GgmlType::Q8_0, n_embd as i64, 8);
        let data: Vec<u8> = (0..ctx.nbytes(w)).map(|i| (i % 251) as u8).collect();
        ctx.set_external_storage(w, std::sync::Arc::new(data), 0);
        let embd_table = ctx.new_tensor_2d(GgmlType::F32, n_embd as i64, 32);
        let tab: Vec<u8> = (0..ctx.nbytes(embd_table)).map(|i| (i % 249) as u8).collect();
        ctx.set_external_storage(embd_table, std::sync::Arc::new(tab), 0);
        let tokens = ctx.new_tensor_1d(GgmlType::I32, n_tok as i64);
        ctx.arena_resize_tensor(tokens);
        ctx.with_i32_mut(tokens, |p| p.copy_from_slice(&[3, 17, 30])).unwrap();
        let x = ctx.get_rows(embd_table, tokens);
        let x = ctx.cont(x);
        let norm = ctx.rms_norm(x, 1e-5);
        let mm = ctx.mul_mat(w, norm);
        let si = ctx.silu(mm);
        ctx.mul(si, mm)
    }

    /// The KV-cache path: a persistent F16 cache bound in the foreign backend,
    /// written through SET_ROWS views and read back through views across two
    /// *separate* graph computes (state must survive the per-step ctx free) —
    /// the decode loop's exact shape. Ground truth = the port CPU engine.
    #[test]
    fn foreign_cpu_kv_cache_persistence() {
        let Some(dir) = ref_lib_dir() else {
            eprintln!("skipping: reference build not present");
            return;
        };
        let n_embd = 32usize; // head dim
        let n_heads = 2usize;
        let cache_size = 16u32; // slots
        let row = (n_embd * n_heads) as i64;

        // ---- port ground truth: two steps through the CPU engine ----
        let mut ctx_g = Context::new();
        let k_cache_g = ctx_g.new_tensor_2d(GgmlType::F16, row, cache_size as i64);
        ctx_g.arena_resize_tensor(k_cache_g);
        let mut want_out = Vec::new();
        for (step, n_tok) in [(1usize, 2usize), (2, 2)] {
            let (wr, kq) = build_kv_step(&mut ctx_g, k_cache_g, n_embd, n_heads, cache_size, n_tok, step);
            let mut g = Graph::new(64);
            g.build_forward(&ctx_g, wr);
            g.build_forward(&ctx_g, kq);
            crate::compute::graph_compute(&mut ctx_g, &mut g, 1);
            want_out.extend_from_slice(ctx_g.f32s(kq).unwrap());
            // also snapshot the cache content after each step
            want_out.extend_from_slice(&f16s_as_f32(&ctx_g, k_cache_g));
        }

        // ---- the same two steps through the foreign CPU backend ----
        let mut ctx = Context::new();
        let k_cache = ctx.new_tensor_2d(GgmlType::F16, row, cache_size as i64);
        ctx.arena_resize_tensor(k_cache); // CPU-engine parity needs storage; the emitter binds its own twin
        let mut cfg = EmitConfig::new(&dir);
        cfg.n_threads = 1;
        let mut exe = ForeignExecutor::new(&cfg).unwrap();
        exe.bind_cache(&ctx, k_cache, Placement::Cpu).unwrap();
        exe.finish_weights().unwrap();
        let mut got_out = Vec::new();
        for (step, n_tok) in [(1usize, 2usize), (2, 2)] {
            let (wr, kq) = build_kv_step(&mut ctx, k_cache, n_embd, n_heads, cache_size, n_tok, step);
            let mut g = Graph::new(64);
            g.build_forward(&ctx, wr);
            g.build_forward(&ctx, kq);
            exe.graph_compute(&mut ctx, &g, &[kq], &[kq]).unwrap();
            got_out.extend_from_slice(ctx.f32s(kq).unwrap());
            // read the persistent foreign cache back through a fresh Rust view
            let bytes = {
                let mut b = vec![0u8; ctx.nbytes(k_cache)];
                unsafe {
                    let ct = exe.twin_of(k_cache).unwrap();
                    (exe.syms.ggml_backend_tensor_get)(ct, b.as_mut_ptr() as *mut c_void, 0, b.len());
                }
                b
            };
            let f16: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let f32s: Vec<f32> = f16.iter().map(|h| half_f16::f16_to_f32(*h)).collect();
            got_out.extend_from_slice(&f32s);
        }

        assert_eq!(got_out.len(), want_out.len());
        let max_abs = got_out
            .iter()
            .zip(&want_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want_out.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        eprintln!("foreign_cpu_kv_cache_persistence: max_abs={max_abs:.3e} scale={scale:.3e}");
        assert!(max_abs <= 4e-3 * scale, "kv persistence drifted: {max_abs} vs scale {scale}");
    }

    /// one attention-ish step: rope the K projection, write rows
    /// [n_kv_prev .. n_kv_prev+n_tok) of the cache via SET_ROWS, read the
    /// whole live prefix back as F32, softmax over a causal mask, and sum —
    /// exercises ROPE, SET_ROWS, VIEW-of-persistent, SOFT_MAX and CPY.
    fn build_kv_step(
        ctx: &mut Context,
        k_cache: TensorId,
        n_embd: usize,
        n_heads: usize,
        cache_size: u32,
        n_tok: usize,
        step: usize,
    ) -> (TensorId, TensorId) {
        let row = (n_embd * n_heads) as i64;
        let n_kv_prev = ((step - 1) * n_tok) as i64;
        // "projection": deterministic input rows — 3d [n_embd, heads, tokens]
        // (rope wants the token count in ne[2], rotation over ne[0] pairs)
        let kq = ctx.new_tensor_3d(GgmlType::F32, n_embd as i64, n_heads as i64, n_tok as i64);
        ctx.arena_resize_tensor(kq);
        ctx.with_f32_mut(kq, |p| {
            for i in 0..p.len() {
                p[i] = ((i % 13) as f32 - 6.0) * 0.25 + step as f32;
            }
        })
        .unwrap();
        // rope over the row (2d: n_embd * n_heads treated as one row pair-wise
        // is not valid rope — use it purely as an op-shape exercise)
        let pos = ctx.new_tensor_1d(GgmlType::I32, n_tok as i64);
        ctx.arena_resize_tensor(pos);
        ctx.with_i32_mut(pos, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = n_kv_prev as i32 + i as i32;
            }
        })
        .unwrap();
        let freq = ctx.new_tensor_1d(GgmlType::F32, (n_embd / 2) as i64);
        ctx.arena_resize_tensor(freq);
        ctx.with_f32_mut(freq, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = 1.0 / (i as f32 + 1.0).powf(0.125);
            }
        })
        .unwrap();
        let roped = ctx.rope_ext(
            kq,
            pos,
            Some(freq),
            n_embd as i32, // n_dims — per-head rotation (llama-graph.cpp's n_rot)
            0,             // mode
            128,           // n_ctx
            10000.0,       // freq_base
            1.0,           // freq_scale
            0.0,           // ext_factor
            0.0,           // attn_factor
            0.0,           // beta_fast
            0.0,           // beta_slow
        );
        // to F16 for the cache write — contiguous [row, n_tok] rows
        let k16 = ctx.new_tensor_2d(GgmlType::F16, row, n_tok as i64);
        let roped_flat = ctx.reshape(roped, k16);
        let cpy = ctx.cpy(roped_flat, k16);
        // indices of the rows to write
        let idx = ctx.new_tensor_1d(GgmlType::I64, n_tok as i64);
        ctx.arena_resize_tensor(idx);
        {
            let b = ctx.data_bytes_mut(idx).unwrap();
            let vals: Vec<i64> = (n_kv_prev..n_kv_prev + n_tok as i64).collect();
            b.copy_from_slice(bytemuck::cast_slice(&vals));
        }
        // SET_ROWS into the persistent cache — dst first (ops.rs:583
        // legacy src order: src[0]=rows, src[1]=idx, src[2]=dst)
        let wr = ctx.set_rows(k_cache, cpy, idx);
        // read the live prefix [row, n_kv] back
        let n_kv = n_kv_prev + n_tok as i64;
        let live = ctx.view_2d(k_cache, row, n_kv, (row * 2) as usize, 0);
        let live_f32 = ctx.new_tensor_2d(GgmlType::F32, row, n_kv);
        let readback = ctx.cpy(live, live_f32);
        // mask for soft_max over the readback rows (a.ne0 = row): the C
        // contract is mask.ne0 == a.ne0, mask.ne1 >= a.ne1 (ops.rs:1257) —
        // cover the full cache height so both steps fit
        let mask = ctx.new_tensor_2d(GgmlType::F32, row, cache_size as i64);
        ctx.arena_resize_tensor(mask);
        ctx.with_f32_mut(mask, |p| {
            for i in 0..p.len() {
                let col = i % row as usize;
                p[i] = if col % 8 < 4 + step { 0.0 } else { f32::NEG_INFINITY };
            }
        })
        .unwrap();
        let sm = ctx.soft_max_ext(readback, Some(mask), 0.1, 0.0);
        let out = ctx.sum_rows(sm);
        ctx.set_name(out, "kv_step_out");
        let _ = cache_size;
        (wr, out)
    }

    fn f16s_as_f32(ctx: &Context, t: TensorId) -> Vec<f32> {
        let bytes = ctx.data_bytes(t).unwrap();
        bytes
            .chunks_exact(2)
            .map(|c| half_f16::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    /// minimal f16 → f32 (reference ggml fp16_ieee_from_fp16_value inverse)
    mod half_f16 {
        pub fn f16_to_f32(h: u16) -> f32 {
            let sign = ((h >> 15) & 1) as u32;
            let exp = ((h >> 10) & 0x1f) as i32;
            let frac = (h & 0x3ff) as u32;
            let bits: u32 = if exp == 0 {
                if frac == 0 {
                    sign << 31
                } else {
                    // subnormal
                    let e = -14 - 10;
                    let mut f = frac;
                    let mut shift = 0;
                    while f & 0x400 == 0 {
                        f <<= 1;
                        shift += 1;
                    }
                    f &= 0x3ff;
                    (sign << 31) | (((127 + e - shift + 1) as u32) << 23) | (f << 13)
                }
            } else if exp == 31 {
                (sign << 31) | (0xff << 23) | (frac << 13)
            } else {
                (sign << 31) | (((exp + 127 - 15) as u32) << 23) | (frac << 13)
            };
            f32::from_bits(bits)
        }
    }
}
