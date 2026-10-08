//! sysffi — the C-ABI layer needed for dynamic backend loading (task ① of the
//! GPU enablement plan).
//!
//! Reference (pinned @ bd4f514db1):
//!   ggml/src/ggml-backend-dl.h  — `dl_handle` / `dl_handle_ptr` / RTLD flags
//!   ggml/src/ggml-backend-dl.cpp — `dl_load_library` / `dl_get_sym` / `dl_error`
//!   ggml/src/ggml-backend-impl.h:248-287 — the DL entry points
//!     `ggml_backend_init_t` / `ggml_backend_score_t` + GGML_BACKEND_DL macros
//!
//! The C side uses `<dlfcn.h>` directly (no libltdl); this module declares the
//! same three functions via `extern "C"` so no `libc` crate dependency is
//! needed (std already links the C runtime; glibc >= 2.34 exposes dl* in libc,
//! older glibc resolves them from the `dl` link below).
//!
//! Only the POSIX path is ported (`#elif defined(__APPLE__)` in C is the same
//! dlfcn API; the _WIN32 LoadLibraryW path is not — this port targets the same
//! platforms the engine does).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

// ggml-backend-dl.h (via dlfcn.h)
pub const RTLD_NOW: c_int = 0x2;
pub const RTLD_LOCAL: c_int = 0x0;

extern "C" {
    // ggml-backend-dl.cpp:35 `dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)`
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    // ggml-backend-dl.cpp:40
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    // ggml-backend-dl.h dl_handle_deleter
    fn dlclose(handle: *mut c_void) -> c_int;
    // ggml-backend-dl.cpp:44
    fn dlerror() -> *mut c_char;
    // [TAG_EMIT_EXIT] _exit for the foreign-backend CLI epilogue: a dlopen'ed
    // libgomp (the foreign CPU backend's OpenMP) leaves worker threads whose
    // dynamic TLS is torn down by glibc's exit handlers before the threads
    // stop (the glibc "dlopen'd module with TLS + threads crash at exit"
    // class) — _exit(0) skips those handlers entirely. std links libc already.
    fn _exit(status: c_int) -> !;
}

/// The CLI's foreign-backend epilogue — see [TAG_EMIT_EXIT].
pub fn exit_now(status: i32) -> ! {
    unsafe { _exit(status) }
}

/// `dl_handle_ptr` (ggml-backend-dl.h:40): owning handle, dlclose on drop.
/// A null handle (the C `nullptr`) is allowed — `is_null` mirrors that test.
pub struct DlHandle(*mut c_void);

// SAFETY: a dlopen handle is an opaque token (glibc dl* are thread-safe);
// it is only used for dlsym here and dropped (dlclose) by the registry entry
// that owns it — the same lifetime discipline as the C dl_handle_ptr.
unsafe impl Send for DlHandle {}
unsafe impl Sync for DlHandle {}

impl DlHandle {
    /// the C `nullptr` dl_handle (dl_handle_ptr default) — a non-owning
    /// stand-in whose Drop is a no-op
    pub fn null() -> DlHandle {
        DlHandle(std::ptr::null_mut())
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// `dl_get_sym` (ggml-backend-dl.cpp:39). Returns None for the C NULL.
    pub fn get_sym(&self, name: &str) -> Option<*mut c_void> {
        if self.0.is_null() {
            return None;
        }
        let cname = CString::new(name).ok()?;
        let p = unsafe { dlsym(self.0, cname.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }
}

impl Drop for DlHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { dlclose(self.0) };
        }
    }
}

/// `dl_load_library` (ggml-backend-dl.cpp:34) — `RTLD_NOW | RTLD_LOCAL`.
pub fn dl_load_library(path: &Path) -> DlHandle {
    // the C side passes path.c_str(); non-UTF8 paths fail closed like a
    // failed CString conversion would
    let Some(s) = path.to_str() else {
        return DlHandle(std::ptr::null_mut());
    };
    let Ok(cname) = CString::new(s) else {
        return DlHandle(std::ptr::null_mut());
    };
    DlHandle(unsafe { dlopen(cname.as_ptr(), RTLD_NOW | RTLD_LOCAL) })
}

/// `dl_error` (ggml-backend-dl.cpp:43) — "" when dlerror() returns NULL.
pub fn dl_error() -> String {
    let e = unsafe { dlerror() };
    if e.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

// ---------------------------------------------------------------------------
// ggml-backend-impl.h:250-254 — the DL entry-point signatures
// ---------------------------------------------------------------------------

/// `typedef ggml_backend_reg_t (*ggml_backend_init_t)(void)` (impl.h:251)
pub type GgmlBackendInitFn = unsafe extern "C" fn() -> *mut GgmlBackendReg;
/// `typedef int (*ggml_backend_score_t)(void)` (impl.h:254)
pub type GgmlBackendScoreFn = unsafe extern "C" fn() -> c_int;

// ---------------------------------------------------------------------------
// C-ABI mirrors of the backend registry objects (ggml-backend-impl.h
// struct definitions) — only what task ① needs to *read* from a loaded
// backend: the reg header (api_version + name + device enumeration). Driving
// a foreign backend's compute (the full tensor/gbuffer ABI translation) is
// task ③ and intentionally not declared here.
// ---------------------------------------------------------------------------

/// `struct ggml_backend_reg_i` (ggml-backend-impl.h:230-240), the fields the
/// registry reads through.
#[repr(C)]
pub struct GgmlBackendRegI {
    pub get_name: Option<unsafe extern "C" fn(*const GgmlBackendReg) -> *const c_char>,
    pub get_device_count: Option<unsafe extern "C" fn(*const GgmlBackendReg) -> usize>,
    pub get_device: Option<unsafe extern "C" fn(*const GgmlBackendReg, usize) -> *mut GgmlBackendDevice>,
    pub get_proc_address: Option<unsafe extern "C" fn(*const GgmlBackendReg, *const c_char) -> *mut c_void>,
}

/// `struct ggml_backend_reg` (ggml-backend-impl.h:242-246)
#[repr(C)]
pub struct GgmlBackendReg {
    /// initialize to GGML_BACKEND_API_VERSION
    pub api_version: c_int,
    pub iface: GgmlBackendRegI,
    pub context: *mut c_void,
}

/// `struct ggml_backend_device_i` (ggml-backend-impl.h:176-218) — the fields
/// device enumeration reads through.
#[repr(C)]
pub struct GgmlBackendDeviceI {
    pub get_name: Option<unsafe extern "C" fn(*const GgmlBackendDevice) -> *const c_char>,
    pub get_description: Option<unsafe extern "C" fn(*const GgmlBackendDevice) -> *const c_char>,
    pub get_memory: Option<unsafe extern "C" fn(*const GgmlBackendDevice, *mut usize, *mut usize)>,
    pub get_type: Option<unsafe extern "C" fn(*const GgmlBackendDevice) -> c_int>,
}

/// `struct ggml_backend_device` (ggml-backend-impl.h:220-224)
#[repr(C)]
pub struct GgmlBackendDevice {
    pub iface: GgmlBackendDeviceI,
    pub reg: *const GgmlBackendReg,
    pub context: *mut c_void,
}

/// Read a `const char*` the usual ggml way ("" when NULL).
pub(crate) unsafe fn cstr_ptr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

// ===========================================================================
// Foreign ggml driving ABI (task ② of the GPU enablement plan)
// ---------------------------------------------------------------------------
// The port's engine builds graphs as `TensorId`s inside a Rust `Context`; a
// foreign (C) backend can only compute C `ggml_tensor*` graphs inside a C
// `ggml_context`. This section declares the C ABI needed to *build* a mirror
// graph and drive a foreign backend's scheduler. Every declaration below is
// resolved by symbol name from a dlopen'ed libggml (see backend_emit.rs) —
// nothing is link-time bound, so the port still builds on hosts without a
// reference build.
//
// Reference (pinned @ bd4f514db1):
//   ggml/include/ggml.h         — struct ggml_tensor (:685-717, fully public),
//                                  struct ggml_init_params (:677), enum ggml_op,
//                                  enum ggml_type, graph API (:2868-2904)
//   ggml/include/ggml-backend.h — the buffer/backend/sched API surface
// ===========================================================================

use std::ffi::c_uint;

/// opaque C handles — only ever passed back through the fn pointers below
#[repr(C)] pub struct GgmlContext { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlCgraph { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendBufferTypeT { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendBufferT { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendT { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendDevT { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendRegT { _priv: [u8; 0] }
#[repr(C)] pub struct GgmlBackendSchedT { _priv: [u8; 0] }

/// `struct ggml_tensor` (ggml.h:685-717) — byte-for-byte mirror; the emitter
/// writes `nb`/`op`/`op_params`/`src`/`view_src`/`view_offs` fields directly
/// because the C API only ever builds tensors through op builders.
#[repr(C)]
pub struct GgmlCTensor {
    pub ty: c_int,
    pub buffer: *mut GgmlBackendBufferT,
    pub ne: [i64; 4],
    pub nb: [usize; 4],
    pub op: c_int,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: [*mut GgmlCTensor; 10],
    pub view_src: *mut GgmlCTensor,
    pub view_offs: usize,
    pub data: *mut c_void,
    pub name: [c_char; 64],
    pub extra: *mut c_void,
    pub padding: [c_char; 8],
}

/// `struct ggml_init_params` (ggml.h:677-681)
#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

// enum ggml_type discriminants are mirrored 1:1 by types::GgmlType (GGUF
// round-trips prove the mapping); ggml_op values are remitted by the emitter
// because the port's GgmlOp discriminants are internal-only.

// ---- core graph-building API (ggml.h) ----
pub type FnGgmlInit = unsafe extern "C" fn(GgmlInitParams) -> *mut GgmlContext;
pub type FnGgmlFree = unsafe extern "C" fn(*mut GgmlContext);
pub type FnGgmlNewTensor =
    unsafe extern "C" fn(*mut GgmlContext, c_int, c_int, *const i64) -> *mut GgmlCTensor;
pub type FnGgmlSetName = unsafe extern "C" fn(*mut GgmlCTensor, *const c_char) -> *mut GgmlCTensor;
pub type FnGgmlSetInput = unsafe extern "C" fn(*mut GgmlCTensor);
pub type FnGgmlSetOutput = unsafe extern "C" fn(*mut GgmlCTensor);
pub type FnGgmlTensorOverhead = unsafe extern "C" fn() -> usize;
pub type FnGgmlGraphOverheadCustom = unsafe extern "C" fn(usize, bool) -> usize;
pub type FnGgmlNewGraphCustom = unsafe extern "C" fn(*mut GgmlContext, usize, bool) -> *mut GgmlCgraph;
pub type FnGgmlBuildForwardExpand = unsafe extern "C" fn(*mut GgmlCgraph, *mut GgmlCTensor);
pub type FnGgmlGraphNNodes = unsafe extern "C" fn(*mut GgmlCgraph) -> c_int;

// ---- buffer / tensor transport (ggml-backend.h) ----
pub type FnBuftAllocBuffer =
    unsafe extern "C" fn(*mut GgmlBackendBufferTypeT, usize) -> *mut GgmlBackendBufferT;
pub type FnBuftGetAlignment = unsafe extern "C" fn(*mut GgmlBackendBufferTypeT) -> usize;
pub type FnBuftGetMaxSize = unsafe extern "C" fn(*mut GgmlBackendBufferTypeT) -> usize;
pub type FnBuftGetAllocSize = unsafe extern "C" fn(*mut GgmlBackendBufferTypeT, *const GgmlCTensor) -> usize;
pub type FnBufferFree = unsafe extern "C" fn(*mut GgmlBackendBufferT);
pub type FnBufferGetBase = unsafe extern "C" fn(*mut GgmlBackendBufferT) -> *mut c_void;
pub type FnBufferClear = unsafe extern "C" fn(*mut GgmlBackendBufferT, u8);
pub type FnBufferSetUsage = unsafe extern "C" fn(*mut GgmlBackendBufferT, c_int);
pub type FnTensorSet =
    unsafe extern "C" fn(*mut GgmlCTensor, *const c_void, usize, usize);
pub type FnTensorGet =
    unsafe extern "C" fn(*const GgmlCTensor, *mut c_void, usize, usize);
pub type FnTensorAlloc = unsafe extern "C" fn(
    *mut GgmlBackendBufferT,
    *mut GgmlCTensor,
    *mut c_void,
) -> c_int;
pub type FnViewInit = unsafe extern "C" fn(*mut GgmlCTensor) -> c_int;
pub type FnCpuBufferFromPtr =
    unsafe extern "C" fn(*mut c_void, usize) -> *mut GgmlBackendBufferT;
pub type FnCpuBufferType = unsafe extern "C" fn() -> *mut GgmlBackendBufferTypeT;

// ---- device / registry (ggml-backend.h) ----
pub type FnBackendFree = unsafe extern "C" fn(*mut GgmlBackendT);
pub type FnDevCount = unsafe extern "C" fn() -> usize;
pub type FnDevGet = unsafe extern "C" fn(usize) -> *mut GgmlBackendDevT;
pub type FnDevByName = unsafe extern "C" fn(*const c_char) -> *mut GgmlBackendDevT;
pub type FnDevByType = unsafe extern "C" fn(c_int) -> *mut GgmlBackendDevT;
pub type FnDevName = unsafe extern "C" fn(*mut GgmlBackendDevT) -> *const c_char;
pub type FnDevInit = unsafe extern "C" fn(*mut GgmlBackendDevT, *const c_char) -> *mut GgmlBackendT;
pub type FnDevBufferType =
    unsafe extern "C" fn(*mut GgmlBackendDevT) -> *mut GgmlBackendBufferTypeT;
pub type FnDevReg = unsafe extern "C" fn(*mut GgmlBackendDevT) -> *mut GgmlBackendRegT;
pub type FnRegGetProcAddress =
    unsafe extern "C" fn(*mut GgmlBackendRegT, *const c_char) -> *mut c_void;
pub type FnLoadAllFromPath = unsafe extern "C" fn(*const c_char);

/// `ggml_backend_set_n_threads_t` (ggml-cpu.cpp:543 proc-address slot) — the
/// fn pointer a CPU reg hands out for thread control.
pub type FnBackendSetNThreads = unsafe extern "C" fn(*mut GgmlBackendT, c_int);

// ---- scheduler (ggml-backend.h:319-351) ----
pub type FnSchedNew = unsafe extern "C" fn(
    *mut *mut GgmlBackendT,          // backends
    *mut *mut GgmlBackendBufferTypeT, // bufts (may be NULL)
    c_int,                            // n_backends
    usize,                            // graph_size
    bool,                             // parallel
    bool,                             // op_offload
) -> *mut GgmlBackendSchedT;
pub type FnSchedFree = unsafe extern "C" fn(*mut GgmlBackendSchedT);
pub type FnSchedReset = unsafe extern "C" fn(*mut GgmlBackendSchedT);
pub type FnSchedAllocGraph = unsafe extern "C" fn(*mut GgmlBackendSchedT, *mut GgmlCgraph) -> bool;
pub type FnSchedGraphCompute = unsafe extern "C" fn(*mut GgmlBackendSchedT, *mut GgmlCgraph) -> c_int;
pub type FnSchedSynchronize = unsafe extern "C" fn(*mut GgmlBackendSchedT);
pub type FnSchedSetTensorBackend =
    unsafe extern "C" fn(*mut GgmlBackendSchedT, *mut GgmlCTensor, *mut GgmlBackendT);
pub type FnSchedGetTensorBackend =
    unsafe extern "C" fn(*mut GgmlBackendSchedT, *mut GgmlCTensor) -> *mut GgmlBackendT;
pub type FnSchedGetNBackends = unsafe extern "C" fn(*mut GgmlBackendSchedT) -> c_int;
pub type FnSchedGetBackend =
    unsafe extern "C" fn(*mut GgmlBackendSchedT, c_int) -> *mut GgmlBackendT;
pub type FnSchedGetBufferSize =
    unsafe extern "C" fn(*mut GgmlBackendSchedT, *mut GgmlBackendT) -> usize;

/// enum ggml_backend_dev_type (ggml-backend.h:134-145) for `ggml_backend_dev_by_type`
pub const GGML_BACKEND_DEV_TYPE_CPU: c_int = 0;
pub const GGML_BACKEND_DEV_TYPE_GPU: c_int = 1;
pub const GGML_BACKEND_DEV_TYPE_IGPU: c_int = 2;

/// enum ggml_backend_buffer_usage (ggml-backend.h:49-53)
pub const GGML_BACKEND_BUFFER_USAGE_ANY: c_int = 0;
pub const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub const GGML_BACKEND_BUFFER_USAGE_COMPUTE: c_int = 2;

/// `enum ggml_status` (ggml.h:655-661)
pub const GGML_STATUS_SUCCESS: c_int = 0;
pub const GGML_STATUS_ALLOC_FAILED: c_int = 1;
pub const GGML_STATUS_BACKEND_ERROR: c_int = 3;
pub const GGML_STATUS_ABORTED: c_int = 4;

// enum ggml_op values of the pinned ggml.h (verified against the pinned
// header's declaration order — the port's GgmlOp discriminants are
// internal-only, backend_emit.rs owns the mapping table)
pub mod op {
    pub const NONE: i32 = 0;
    pub const DUP: i32 = 1;
    pub const ADD: i32 = 2;
    pub const ADD_ID: i32 = 3;
    pub const SUB: i32 = 6;
    pub const MUL: i32 = 7;
    pub const DIV: i32 = 8;
    pub const SQR: i32 = 9;
    pub const SQRT: i32 = 10;
    pub const LOG: i32 = 11;
    pub const SIN: i32 = 12;
    pub const COS: i32 = 13;
    pub const SUM: i32 = 14;
    pub const SUM_ROWS: i32 = 15;
    pub const CUMSUM: i32 = 16;
    pub const MEAN: i32 = 17;
    pub const ARGMAX: i32 = 18;
    pub const REPEAT: i32 = 20;
    pub const CONCAT: i32 = 22;
    pub const NORM: i32 = 24;
    pub const RMS_NORM: i32 = 25;
    /// ggml.h GGML_OP_L2_NORM (position 28: NORM 24 / RMS_NORM 25 /
    /// RMS_NORM_BACK 26 / GROUP_NORM 27) — clef (a7b94df2c)
    pub const L2_NORM: i32 = 28;
    pub const MUL_MAT: i32 = 29;
    pub const MUL_MAT_ID: i32 = 30;
    pub const SCALE: i32 = 32;
    /// ggml.h GGML_OP_SET (position 33, right after SCALE) — arch batch 18
    pub const SET: i32 = 33;
    pub const CPY: i32 = 34;
    pub const CONT: i32 = 35;
    pub const RESHAPE: i32 = 36;
    pub const VIEW: i32 = 37;
    pub const PERMUTE: i32 = 38;
    pub const TRANSPOSE: i32 = 39;
    pub const GET_ROWS: i32 = 40;
    pub const SET_ROWS: i32 = 42;
    /// ggml.h GGML_OP_DIAG (position 43, right after SET_ROWS) — arch batch 18
    pub const DIAG: i32 = 43;
    pub const DIAG_MASK_INF: i32 = 44;
    pub const SOFT_MAX: i32 = 46;
    pub const ROPE: i32 = 48;
    pub const ROPE_BACK: i32 = 49;
    pub const CLAMP: i32 = 50;
    pub const IM2COL: i32 = 52;
    pub const COL2IM_1D: i32 = 55;
    pub const CONV_2D: i32 = 56;
    pub const CONV_2D_DW: i32 = 58;
    pub const POOL_1D: i32 = 60;
    pub const POOL_2D: i32 = 61;
    pub const UPSCALE: i32 = 63;
    pub const PAD: i32 = 64;
    pub const PAD_REFLECT_1D: i32 = 65;
    pub const ROLL: i32 = 66;
    pub const ARANGE: i32 = 67;
    pub const ARGSORT: i32 = 69;
    pub const TOP_K: i32 = 70;
    pub const TRI: i32 = 72;
    pub const FILL: i32 = 73;
    pub const FLASH_ATTN_EXT: i32 = 74;
    pub const SSM_CONV: i32 = 76;
    pub const SSM_SCAN: i32 = 77;
    pub const RWKV_WKV6: i32 = 82;
    pub const GATED_LINEAR_ATTN: i32 = 83;
    pub const RWKV_WKV7: i32 = 84;
    /// ggml.h GGML_OP_SOLVE_TRI (position 85, right after RWKV_WKV7) — arch
    /// batch 18 (the chunked delta-net's UT transform)
    pub const SOLVE_TRI: i32 = 85;
    pub const GATED_DELTA_NET: i32 = 86;
    pub const LIGHTNING_INDEXER: i32 = 87;
    pub const DSV4_HC_COMB: i32 = 88;
    pub const DSV4_HC_PRE: i32 = 89;
    pub const DSV4_HC_POST: i32 = 90;
    pub const UNARY: i32 = 91;
    /// ggml.h:601 GGML_OP_GLU — enum position 100, the LAST real op before
    /// GGML_OP_COUNT (101); the previous value 101 collided with COUNT and
    /// aborted the foreign CPU backend's ggml_graph_plan on gpt-oss (GLU is
    /// the MoE FFN gate) — batch 18
    pub const GLU: i32 = 100;
}

/// All dynamically-resolved symbols of one loaded ggml, grouped. Built by
/// `CLibSyms::resolve`; every field is Some after a successful load. `Copy`
/// so callers can snapshot it out of a borrowed owner.
#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct CLibSyms {
    // core
    pub ggml_init: FnGgmlInit,
    pub ggml_free: FnGgmlFree,
    pub ggml_new_tensor: FnGgmlNewTensor,
    pub ggml_set_name: FnGgmlSetName,
    pub ggml_set_input: FnGgmlSetInput,
    pub ggml_set_output: FnGgmlSetOutput,
    pub ggml_tensor_overhead: FnGgmlTensorOverhead,
    pub ggml_graph_overhead_custom: FnGgmlGraphOverheadCustom,
    pub ggml_new_graph_custom: FnGgmlNewGraphCustom,
    pub ggml_build_forward_expand: FnGgmlBuildForwardExpand,
    pub ggml_graph_n_nodes: FnGgmlGraphNNodes,
    // buffers / transport
    pub ggml_backend_buft_alloc_buffer: FnBuftAllocBuffer,
    pub ggml_backend_buft_get_alignment: FnBuftGetAlignment,
    pub ggml_backend_buft_get_max_size: FnBuftGetMaxSize,
    pub ggml_backend_buft_get_alloc_size: FnBuftGetAllocSize,
    pub ggml_backend_buffer_free: FnBufferFree,
    pub ggml_backend_buffer_get_base: FnBufferGetBase,
    pub ggml_backend_buffer_clear: FnBufferClear,
    pub ggml_backend_buffer_set_usage: FnBufferSetUsage,
    pub ggml_backend_tensor_set: FnTensorSet,
    pub ggml_backend_tensor_get: FnTensorGet,
    pub ggml_backend_tensor_alloc: FnTensorAlloc,
    pub ggml_backend_view_init: FnViewInit,
    pub ggml_backend_cpu_buffer_from_ptr: FnCpuBufferFromPtr,
    pub ggml_backend_cpu_buffer_type: FnCpuBufferType,
    // devices / registry
    pub ggml_backend_dev_count: FnDevCount,
    pub ggml_backend_dev_get: FnDevGet,
    pub ggml_backend_dev_by_name: FnDevByName,
    pub ggml_backend_dev_by_type: FnDevByType,
    pub ggml_backend_dev_name: FnDevName,
    pub ggml_backend_free: FnBackendFree,
    pub ggml_backend_dev_init: FnDevInit,
    pub ggml_backend_dev_buffer_type: FnDevBufferType,
    pub ggml_backend_dev_backend_reg: FnDevReg,
    pub ggml_backend_reg_get_proc_address: FnRegGetProcAddress,
    pub ggml_backend_load_all_from_path: FnLoadAllFromPath,
    // scheduler
    pub ggml_backend_sched_new: FnSchedNew,
    pub ggml_backend_sched_free: FnSchedFree,
    pub ggml_backend_sched_reset: FnSchedReset,
    pub ggml_backend_sched_alloc_graph: FnSchedAllocGraph,
    pub ggml_backend_sched_graph_compute: FnSchedGraphCompute,
    pub ggml_backend_sched_synchronize: FnSchedSynchronize,
    pub ggml_backend_sched_set_tensor_backend: FnSchedSetTensorBackend,
    pub ggml_backend_sched_get_tensor_backend: FnSchedGetTensorBackend,
    pub ggml_backend_sched_get_n_backends: FnSchedGetNBackends,
    pub ggml_backend_sched_get_backend: FnSchedGetBackend,
    pub ggml_backend_sched_get_buffer_size: FnSchedGetBufferSize,
    pub ggml_backend_time_us: unsafe extern "C" fn() -> i64,
}

impl CLibSyms {
    /// Resolve every symbol of [`CLibSyms`] from `base` (libggml-base.so) and
    /// `full` (libggml.so — owns the registry / DL loading; the two share the
    /// same build so their types are one ABI). Missing symbols → error string
    /// naming the symbol (the loader logs it like ggml-backend-reg.cpp:222).
    pub fn resolve(
        base: &DlHandle,
        full: Option<&DlHandle>,
    ) -> Result<CLibSyms, String> {
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // most symbols live in ggml-base; the registry/DL-loading ones
                // live in the full ggml lib — try base, then full
                let mut found: Option<$ty> = base.get_sym($name).map(|p| unsafe {
                    std::mem::transmute::<*mut c_void, $ty>(p)
                });
                if found.is_none() {
                    if let Some(f) = full {
                        found = f.get_sym($name).map(|p| unsafe {
                            std::mem::transmute::<*mut c_void, $ty>(p)
                        });
                    }
                }
                found.ok_or_else(|| concat!($name, " missing").to_string())?
            }};
        }
        let _ = c_uint::MAX; // keep the import honest
        Ok(CLibSyms {
            ggml_init: sym!("ggml_init", FnGgmlInit),
            ggml_free: sym!("ggml_free", FnGgmlFree),
            ggml_new_tensor: sym!("ggml_new_tensor", FnGgmlNewTensor),
            ggml_set_name: sym!("ggml_set_name", FnGgmlSetName),
            ggml_set_input: sym!("ggml_set_input", FnGgmlSetInput),
            ggml_set_output: sym!("ggml_set_output", FnGgmlSetOutput),
            ggml_tensor_overhead: sym!("ggml_tensor_overhead", FnGgmlTensorOverhead),
            ggml_graph_overhead_custom: sym!("ggml_graph_overhead_custom", FnGgmlGraphOverheadCustom),
            ggml_new_graph_custom: sym!("ggml_new_graph_custom", FnGgmlNewGraphCustom),
            ggml_build_forward_expand: sym!("ggml_build_forward_expand", FnGgmlBuildForwardExpand),
            ggml_graph_n_nodes: sym!("ggml_graph_n_nodes", FnGgmlGraphNNodes),
            ggml_backend_buft_alloc_buffer: sym!("ggml_backend_buft_alloc_buffer", FnBuftAllocBuffer),
            ggml_backend_buft_get_alignment: sym!("ggml_backend_buft_get_alignment", FnBuftGetAlignment),
            ggml_backend_buft_get_max_size: sym!("ggml_backend_buft_get_max_size", FnBuftGetMaxSize),
            ggml_backend_buft_get_alloc_size: sym!("ggml_backend_buft_get_alloc_size", FnBuftGetAllocSize),
            ggml_backend_buffer_free: sym!("ggml_backend_buffer_free", FnBufferFree),
            ggml_backend_buffer_get_base: sym!("ggml_backend_buffer_get_base", FnBufferGetBase),
            ggml_backend_buffer_clear: sym!("ggml_backend_buffer_clear", FnBufferClear),
            ggml_backend_buffer_set_usage: sym!("ggml_backend_buffer_set_usage", FnBufferSetUsage),
            ggml_backend_tensor_set: sym!("ggml_backend_tensor_set", FnTensorSet),
            ggml_backend_tensor_get: sym!("ggml_backend_tensor_get", FnTensorGet),
            ggml_backend_tensor_alloc: sym!("ggml_backend_tensor_alloc", FnTensorAlloc),
            ggml_backend_view_init: sym!("ggml_backend_view_init", FnViewInit),
            ggml_backend_cpu_buffer_from_ptr: sym!("ggml_backend_cpu_buffer_from_ptr", FnCpuBufferFromPtr),
            ggml_backend_cpu_buffer_type: sym!("ggml_backend_cpu_buffer_type", FnCpuBufferType),
            ggml_backend_dev_count: sym!("ggml_backend_dev_count", FnDevCount),
            ggml_backend_dev_get: sym!("ggml_backend_dev_get", FnDevGet),
            ggml_backend_dev_by_name: sym!("ggml_backend_dev_by_name", FnDevByName),
            ggml_backend_dev_by_type: sym!("ggml_backend_dev_by_type", FnDevByType),
            ggml_backend_dev_name: sym!("ggml_backend_dev_name", FnDevName),
            ggml_backend_free: sym!("ggml_backend_free", FnBackendFree),
            ggml_backend_dev_init: sym!("ggml_backend_dev_init", FnDevInit),
            ggml_backend_dev_buffer_type: sym!("ggml_backend_dev_buffer_type", FnDevBufferType),
            ggml_backend_dev_backend_reg: sym!("ggml_backend_dev_backend_reg", FnDevReg),
            ggml_backend_reg_get_proc_address: sym!("ggml_backend_reg_get_proc_address", FnRegGetProcAddress),
            ggml_backend_load_all_from_path: sym!("ggml_backend_load_all_from_path", FnLoadAllFromPath),
            ggml_backend_sched_new: sym!("ggml_backend_sched_new", FnSchedNew),
            ggml_backend_sched_free: sym!("ggml_backend_sched_free", FnSchedFree),
            ggml_backend_sched_reset: sym!("ggml_backend_sched_reset", FnSchedReset),
            ggml_backend_sched_alloc_graph: sym!("ggml_backend_sched_alloc_graph", FnSchedAllocGraph),
            ggml_backend_sched_graph_compute: sym!("ggml_backend_sched_graph_compute", FnSchedGraphCompute),
            ggml_backend_sched_synchronize: sym!("ggml_backend_sched_synchronize", FnSchedSynchronize),
            ggml_backend_sched_set_tensor_backend: sym!("ggml_backend_sched_set_tensor_backend", FnSchedSetTensorBackend),
            ggml_backend_sched_get_tensor_backend: sym!("ggml_backend_sched_get_tensor_backend", FnSchedGetTensorBackend),
            ggml_backend_sched_get_n_backends: sym!("ggml_backend_sched_get_n_backends", FnSchedGetNBackends),
            ggml_backend_sched_get_backend: sym!("ggml_backend_sched_get_backend", FnSchedGetBackend),
            ggml_backend_sched_get_buffer_size: sym!("ggml_backend_sched_get_buffer_size", FnSchedGetBufferSize),
            ggml_backend_time_us: sym!("ggml_time_us", unsafe extern "C" fn() -> i64),
        })
    }
}
