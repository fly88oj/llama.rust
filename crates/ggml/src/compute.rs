//! compute.rs — graph execution (port of ggml-cpu compute dispatch). Owner: agent A.
//!
//! `graph_compute` mirrors ggml-cpu.c's threadpool loop: allocate storage for
//! every graph tensor, then run each node's `ggml_compute_forward_*` in
//! topological order. Parallelism follows the C kernels' row-partitioning
//! (contiguous row ranges per thread, dispatched through a spinning `Team` —
//! see its docs), so results are independent of the thread count (asserted by
//! `vec_dot::thread_invariance_tests`). Each kernel is annotated with its C
//! reference.
//!
//! Memory model: unlike C (one pointer per tensor), tensors live in
//! `Context::arena` or external mmap buffers. Kernels address rows through
//! byte offsets computed from `nb` strides, exactly like the C code; the raw
//! arena pointer lets disjoint rows be written in parallel.

use crate::ops::{
    ggml_compute_fp32_to_fp16, ggml_silu_scalar_f32, ggml_tanh_f32,
    ggml_vec_gelu_f16, ggml_vec_gelu_erf_f16, ggml_vec_gelu_erf_f32, ggml_vec_gelu_f32,
    OP_FLAG_GET_ROWS_BACK,
    OP_FLAG_NORM_IS_RMS, OP_FLAG_NORM_IS_L2, GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_NEOX,
    GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_VISION, GGML_SCALE_FLAG_ALIGN_CORNERS,
    GGML_SCALE_MODE_BICUBIC, GGML_SCALE_MODE_BILINEAR, GGML_SCALE_MODE_NEAREST,
    GGML_SORT_ORDER_ASC, GGML_UNARY_OP_EXP, GGML_UNARY_OP_GELU, GGML_UNARY_OP_GELU_ERF,
    GGML_UNARY_OP_ROUND, GGML_UNARY_OP_XIELU,
    GGML_UNARY_OP_ELU,
    GGML_UNARY_OP_RELU,
    GGML_UNARY_OP_SIGMOID, GGML_UNARY_OP_SILU, GGML_UNARY_OP_SOFTPLUS, GGML_UNARY_OP_TANH,
    // NB: without these the match arms below are *binding patterns*
    // (catch-alls), not constants — STEP had been silently swallowing every
    // unimported unary op since audio round 5. arch batch 18 (agent GDN)
    // caught it via NEG dispatching to forward_step.
    GGML_UNARY_OP_STEP, GGML_UNARY_OP_NEG, GGML_UNARY_OP_ABS, GGML_UNARY_OP_SGN,
};
use crate::quants;
use crate::tensor::{Context, GgmlOp, Storage, TensorId, TensorMeta};
use crate::tinyblas;
use crate::types::{GgmlType, MAX_DIMS, MAX_OP_PARAMS, MAX_SRC};
use crate::vec_dot;
use half::{bf16, f16};

use crate::graph::Graph;

// ======================================================================
// Thread team — 对照 ggml-cpu.c 的 threadpool + ggml_barrier
// ======================================================================

/// A fixed team of `n` worker threads for the duration of one `graph_compute`
/// call. Workers spin on a generation counter published by the main thread
/// (like ggml's barrier) instead of going through rayon for every op: rayon's
/// `install` costs ~45 us per region here (~250 regions per token), which ate
/// the entire parallel speedup (8 threads gave only 2.3x on qwen generation).
///
/// Protocol (one region at a time, single-producer):
///  1. the main thread stores the job pointer + region size, then bumps `gen`
///     (`Release`) and immediately runs the last shard itself;
///  2. every worker spins until `gen` changes (`Acquire`), reads the job, runs
///     its shard, then bumps `done` (`Release`);
///  3. the main thread spins until `done == n-1`, so no worker is inside the
///     closure when the next region (or the closure's stack frame) is reused.
struct Team<'s> {
    shared: &'s TeamShared,
    n: usize,
}

struct TeamShared {
    gen: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
    quit: std::sync::atomic::AtomicBool,
    nr: std::sync::atomic::AtomicUsize,
    /// Only read/written under the protocol above; see `Team`.
    job: std::cell::UnsafeCell<Option<*const (dyn Fn(usize, usize) + Sync + 'static)>>,
}

// SAFETY: the `UnsafeCell` is only touched by the single-producer protocol
// documented on `Team` (store before the `gen` release, read after the `gen`
// acquire, never concurrent with a store).
unsafe impl Sync for TeamShared {}

impl<'s> Team<'s> {
    /// Spawn `n-1` workers into `scope`; the caller's thread is worker 0 and
    /// runs shard 0 of every region. `shared` must outlive the scope (it is a
    /// local of `graph_compute`), which is what lets the workers borrow it.
    fn new(scope: &'s std::thread::Scope<'s, '_>, shared: &'s TeamShared, n: usize) -> Self {
        for idx in 1..n {
            scope.spawn(move || worker_loop(shared, idx, n));
        }
        Team { shared, n }
    }

    /// Run `f(shard_lo, shard_hi)` over `jobs` units, split into `n` contiguous
    /// shards (`dr*(i+1) - dr*i` like the C `dr` partitioning).
    fn run(&self, jobs: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        if jobs == 0 {
            return;
        }
        let n = self.n;
        if n <= 1 {
            f(0, jobs);
            return;
        }
        // Shards are always computed over the *full* team size (shards past
        // `jobs` are empty) so that the main thread and every worker agree on
        // the split; `n = min(nth, jobs)` would give the same boundaries
        // (`ceil(jobs/nth)` vs `ceil(jobs/jobs)`) but would let the main thread
        // stop waiting while a worker was still inside an empty shard.
        let dr = (jobs + n - 1) / n;
        self.shared.nr.store(jobs, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: the closure outlives the region (`run` does not return until
        // every worker has reported `done`), and erasing its lifetime is a
        // lifetime-only change to the trait object.
        let p: *const (dyn Fn(usize, usize) + Sync + 'static) =
            unsafe { std::mem::transmute::<&(dyn Fn(usize, usize) + Sync), _>(f) };
        unsafe { *self.shared.job.get() = Some(p) };
        self.shared.done.store(0, std::sync::atomic::Ordering::Relaxed);
        self.shared.gen.fetch_add(1, std::sync::atomic::Ordering::Release);

        // shard 0 is this thread's own (the workers took 1..n-1)
        let hi = dr.min(jobs);
        if 0 < hi {
            f(0, hi);
        }
        while self.shared.done.load(std::sync::atomic::Ordering::Acquire) < self.n - 1 {
            std::hint::spin_loop();
        }
    }

    /// Publish a null job before waking the workers so nobody can dereference a
    /// stale closure pointer after the last region.
    fn shutdown(&self) {
        if self.n <= 1 {
            return;
        }
        unsafe { *self.shared.job.get() = None };
        self.shared.quit.store(true, std::sync::atomic::Ordering::Release);
        self.shared.gen.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

fn worker_loop(shared: &TeamShared, idx: usize, n: usize) {
    use std::sync::atomic::Ordering;
    let mut seen = 0usize;
    let mut spins = 0u32;
    loop {
        let g = shared.gen.load(Ordering::Acquire);
        if g == seen {
            // wrapping: `spins` is only a parking hint (never read); a debug
            // build's overflow check must not kill a worker that has been
            // spinning through a long single-threaded region.
            spins = spins.wrapping_add(1);
            // Spin, never yield: a worker that is descheduled costs the *next*
            // region a full scheduler wake-up (tens of microseconds), which is
            // more than the region's own work. `ggml_barrier` does the same
            // (ggml_threadpool_active_waits / sched_yield budget).
            let _ = spins;
            std::hint::spin_loop();
            continue;
        }
        seen = g;
        spins = 0;
        if shared.quit.load(Ordering::Acquire) {
            return;
        }
        let jobs = shared.nr.load(Ordering::Relaxed);
        // SAFETY: `gen` was observed after the producer's release store, and
        // the producer only advances `gen` again after every worker reported
        // `done` — so the pointer is still the live closure of this region.
        let job = unsafe { *shared.job.get() };
        if let Some(p) = job {
            let dr = (jobs + n - 1) / n;
            let lo = dr * idx;
            let hi = (lo + dr).min(jobs);
            if lo < hi {
                // SAFETY: see above — the referent is alive for the region and
                // this thread is the only one running this shard.
                let f: &(dyn Fn(usize, usize) + Sync) = unsafe {
                    std::mem::transmute::<*const (dyn Fn(usize, usize) + Sync + 'static), _>(p)
                };
                f(lo, hi);
            }
        }
        shared.done.fetch_add(1, Ordering::Release);
    }
}

// ======================================================================
// eval callback — port of `ggml_backend_sched_set_eval_callback`
// (ggml-backend.cpp:1798-1835, ggml-backend.h:309-316) and its use by the
// imatrix collector (`params.cb_eval`, llama-context.cpp:1421)
// ======================================================================

/// One `src[i]` of a node, resolved for [`EvalCallback`].
#[derive(Clone, Copy)]
pub struct EvalSrc<'a> {
    pub name: &'a str,
    pub ty: GgmlType,
    pub ne: [i64; MAX_DIMS],
    pub nb: [u64; MAX_DIMS],
    /// Bytes from the tensor's view start to the end of its backing storage
    /// (stride-addressed, exactly like the C kernels). `None` = no storage.
    pub data: Option<&'a [u8]>,
}

/// Read-only view of one graph node handed to [`EvalCallback`] — the port of
/// the `struct ggml_tensor * t` the C scheduler passes (the callback dereferences
/// `t->op`, `t->src[0]->name`, `t->src[1]` / `t->src[2]` data, `t->ne`).
#[derive(Clone, Copy)]
pub struct EvalNode<'a> {
    pub op: GgmlOp,
    pub name: &'a str,
    pub ty: GgmlType,
    pub ne: [i64; MAX_DIMS],
    pub nb: [u64; MAX_DIMS],
    /// the node's op_params (needed by ggml_op_desc's UNARY/GLU variants)
    pub op_params: [i32; MAX_OP_PARAMS / 4],
    /// The node's own bytes from its view start to the end of its backing
    /// storage (`t->data` post-compute, ask == false). `None` = no storage.
    pub data: Option<&'a [u8]>,
    pub src: [Option<EvalSrc<'a>>; MAX_SRC],
}

/// `ggml_backend_sched_eval_callback` (ggml-backend.h:316). Called twice per
/// node: `ask == true` before the node is computed ("do you want this data?"),
/// `ask == false` after it (the data hand-off). Returning false from the ask
/// call only means "not interested" — the node is computed either way, since
/// this port has no scheduler that could skip it.
pub type EvalCallback = fn(node: &EvalNode<'_>, ask: bool) -> bool;

/// A plain function pointer, so the "no callback registered" path is a single
/// null test outside the node loop (see `graph_compute`; the C scheduler takes
/// the same branch at ggml-backend.cpp:1798 with `sched->callback_eval` unset).
static EVAL_CALLBACK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `ggml_backend_sched_set_eval_callback` (ggml-backend.cpp:2045-2049).
pub fn set_eval_callback(callback: Option<EvalCallback>) {
    let p = callback.map(|f| f as usize).unwrap_or(0);
    EVAL_CALLBACK.store(p, std::sync::atomic::Ordering::Release);
}

#[inline]
fn eval_callback() -> Option<EvalCallback> {
    match EVAL_CALLBACK.load(std::sync::atomic::Ordering::Acquire) {
        0 => None,
        // SAFETY: only `set_eval_callback` writes the static, and only ever
        // with the usize of a `fn(&EvalNode, bool) -> bool`.
        p => Some(unsafe { std::mem::transmute::<usize, EvalCallback>(p) }),
    }
}

/// `op_add`/`op_mul`/`op_div` (binary-ops.cpp:8-17) — the per-lane scalar the
/// C `vec_binary_op_*` templates apply; the enum (instead of a closure) lets
/// the hot F32×F32→F32 path of `forward_bin_op` dispatch to vector kernels.
#[derive(Clone, Copy, PartialEq)]
enum BinOp {
    Add,
    Mul,
    Div,
    Sub,
}

/// Per-op wall-clock profile of `graph_compute`, diagnostic only: with
/// `LLAMA_RUST_OPPROF=<interval>` set, every node's `forward` is timed and the
/// cumulative table (per tensor key `name|op|ne`) is printed to stderr every
/// `<interval>`-th call (call 1 included, so a single prompt forward dumps
/// itself). `forward` ends in a team barrier, so main-thread wall time is the
/// op's true duration. Zero cost when the env var is absent.
static OPPROF: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, (u64, u64)>>,
> = std::sync::OnceLock::new();
static OPPROF_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn opprof_interval() -> Option<u64> {
    std::env::var("LLAMA_RUST_OPPROF").ok()?.parse::<u64>().ok()
}

/// `LLAMA_RUST_PHASEPROF=1`: per-`graph_compute` phase split (arena pass vs
/// node loop) on stderr, diagnostic only.
fn phaseprof_interval() -> Option<u64> {
    std::env::var("LLAMA_RUST_PHASEPROF").ok()?.parse::<u64>().ok()
}

fn opprof_record(key: String, ns: u64) {
    let m = OPPROF.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = m.lock().unwrap();
    let e = guard.entry(key).or_insert((0, 0));
    e.0 += 1;
    e.1 += ns;
}

fn opprof_maybe_dump() {
    let Some(interval) = opprof_interval() else { return };
    let n = OPPROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n != 1 && n % interval != 0 {
        return;
    }
    let m = OPPROF.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut rows: Vec<(String, u64, u64)> =
        m.lock().unwrap().iter().map(|(k, (c, ns))| (k.clone(), *c, *ns)).collect();
    rows.sort_by_key(|(_, _, ns)| std::cmp::Reverse(*ns));
    let total: u64 = rows.iter().map(|(_, _, ns)| *ns).sum();
    eprintln!("[OPPROF] call #{n}: {total} ns total, {} rows", rows.len());
    for (k, c, ns) in rows {
        eprintln!("[OPPROF] {ns:>12} ns  {c:>6}x  {:>10} ns/op  {k}", ns / c.max(1));
    }
}

pub fn graph_compute(ctx: &mut Context, g: &mut Graph, n_threads: usize) {
    // ggml_gallocr: every non-view graph tensor without storage gets arena room
    // (views resolve into their parent's storage). This is the port's
    // pre-perf9 shape — a pure bump, no block reuse.
    let phase_t0 = std::time::Instant::now();
    for &id in g.nodes.iter().chain(g.leafs.iter()) {
        if ctx.tensors[id.0 as usize].view_src.is_none() {
            ctx.arena_resize_tensor(id);
        }
    }
    let arena_ns = phase_t0.elapsed().as_nanos() as u64;
    graph_execute(ctx, g, n_threads, arena_ns, phase_t0);
}

/// perf9's gallocr-style CPU path: the same allocation discipline the
/// reference's `ggml_backend_sched` runs (ggml-alloc.c's dyn_tallocr —
/// best-fit free blocks, parents freed at their last consumer, in-place
/// parent reuse for `op_can_inplace` nodes), applied to the Context arena in
/// one planning pass before execution. `outputs` is the caller's
/// `ggml_set_output` set (llama-graph.cpp:1364-1401): those tensors are
/// flagged and never freed nor reused, so reading them after this returns is
/// safe. Everything else about the run is identical to [`graph_compute`].
pub fn graph_compute_reusing(
    ctx: &mut Context,
    g: &mut Graph,
    n_threads: usize,
    outputs: &[TensorId],
) {
    let phase_t0 = std::time::Instant::now();
    // LLAMA_RUST_NO_REUSE=1: fall back to the pre-perf9 bump (A/B knob for
    // benchmarking the reuse's effect within one binary; empty/0 = unset)
    let no_reuse = std::env::var("LLAMA_RUST_NO_REUSE").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
    if no_reuse {
        for &id in g.nodes.iter().chain(g.leafs.iter()) {
            if ctx.tensors[id.0 as usize].view_src.is_none() {
                ctx.arena_resize_tensor(id);
            }
        }
    } else {
        for &o in outputs {
            ctx.set_output(o);
        }
        plan_arena_reuse(ctx, g);
    }
    let arena_ns = phase_t0.elapsed().as_nanos() as u64;
    graph_execute(ctx, g, n_threads, arena_ns, phase_t0);
}

/// The free-block list + bump frontier of `ggml_dyn_tallocr`
/// (ggml-alloc.c:110-350), single-buffer degenerate case: `top` plays the
/// C's always-sufficient last block, `free` holds only the interior holes.
struct ArenaPlan {
    free: Vec<(usize, usize)>, // (offset, size), sorted by offset
    top: usize,
}

impl ArenaPlan {
    /// `ggml_dyn_tallocr_alloc` (ggml-alloc.c:202-269): sizes rounded to the
    /// 64-byte tensor alignment, best fit among the interior blocks
    /// (:215-224, `<=` so the last of equal sizes wins), bump the frontier
    /// otherwise.
    fn alloc(&mut self, size: usize) -> usize {
        let size = (size + 63) & !63;
        let mut best: Option<usize> = None;
        for (i, &(_, sz)) in self.free.iter().enumerate() {
            if sz >= size && best.is_none_or(|b| sz <= self.free[b].1) {
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                let (off, sz) = self.free[i];
                if sz == size {
                    self.free.remove(i);
                } else {
                    self.free[i] = (off + size, sz - size);
                }
                off
            }
            None => {
                let off = self.top;
                self.top += size;
                off
            }
        }
    }

    /// `ggml_dyn_tallocr_free_bytes` (ggml-alloc.c:312-350): insert sorted,
    /// merge with both neighbors; a block that reaches the frontier folds
    /// back into it (the C's last block just grows).
    fn free(&mut self, offset: usize, size: usize) {
        let size = (size + 63) & !63;
        if size == 0 {
            return;
        }
        let mut idx = self.free.len();
        for (i, &(off, _)) in self.free.iter().enumerate() {
            if offset < off {
                idx = i;
                break;
            }
        }
        self.free.insert(idx, (offset, size));
        if idx + 1 < self.free.len() && self.free[idx].0 + self.free[idx].1 == self.free[idx + 1].0 {
            self.free[idx].1 += self.free[idx + 1].1;
            self.free.remove(idx + 1);
        }
        if idx > 0 && self.free[idx - 1].0 + self.free[idx - 1].1 == self.free[idx].0 {
            self.free[idx - 1].1 += self.free[idx].1;
            self.free.remove(idx);
        }
        if let Some(&(off, sz)) = self.free.last() {
            if off + sz == self.top {
                self.top = off;
                self.free.pop();
            }
        }
    }
}

/// One `struct hash_node` (ggml-alloc.c:459-466) per planned tensor.
#[derive(Default, Clone, Copy)]
struct PlanNode {
    n_children: u32,
    n_views: u32,
    allocated: bool,
    offset: usize,
}

struct PlanWalk<'a> {
    ctx: &'a Context,
    hash: std::collections::HashMap<TensorId, PlanNode>,
    plan: ArenaPlan,
    assign: Vec<(TensorId, usize)>,
}

impl<'a> PlanWalk<'a> {
    fn hn(&mut self, t: TensorId) -> &mut PlanNode {
        self.hash.entry(t).or_default()
    }

    fn is_own(&mut self, t: TensorId) -> bool {
        self.hn(t).allocated
    }

    /// `ggml_gallocr_is_allocated` (ggml-alloc.c:524) for the plain CPU path:
    /// no backend buffer side table exists here, so "already allocated" is
    /// having storage (weights/KV/inputs) or being planned by us.
    fn is_allocated(&mut self, t: TensorId) -> bool {
        !matches!(self.ctx.tensors[t.0 as usize].storage, crate::tensor::Storage::None) || self.is_own(t)
    }

    fn out_flag(&self, t: TensorId) -> bool {
        let base = match self.ctx.tensors[t.0 as usize].view_src {
            Some(v) => v,
            None => t,
        };
        self.ctx.tensors[base.0 as usize].flags & crate::backend::GGML_TENSOR_FLAG_OUTPUT != 0
    }

    /// `ggml_gallocr_free_extra_space` (ggml-alloc.c:532-553) — same layout
    /// ⇒ same alignment, so only the size difference is freed.
    fn free_extra_space(&mut self, node: TensorId, parent: TensorId) {
        let parent_size = self.ctx.nbytes(parent);
        let node_size = self.ctx.nbytes(node);
        debug_assert!(parent_size >= node_size);
        let parent_size = (parent_size + 63) & !63;
        let node_size = (node_size + 63) & !63;
        if parent_size > node_size {
            let p_offset = self.hn(parent).offset;
            self.plan.free(p_offset + node_size, parent_size - node_size);
        }
    }

    /// `ggml_gallocr_allocate_node` (ggml-alloc.c:623-689).
    fn allocate_node(&mut self, node: TensorId) {
        if self.is_allocated(node) || self.ctx.tensors[node.0 as usize].view_src.is_some() {
            return;
        }
        self.hn(node).allocated = true;

        // try to reuse a parent's buffer (inplace)
        if crate::backend_sched::op_can_inplace(self.ctx.op(node)) {
            for i in 0..crate::types::MAX_SRC {
                let Some(parent) = self.ctx.tensors[node.0 as usize].src[i] else { continue };
                if !self.is_own(parent) {
                    continue;
                }
                if self.out_flag(parent) {
                    continue;
                }
                if !crate::backend::are_same_layout(self.ctx, node, parent) {
                    continue;
                }
                let p_hn = *self.hn(parent);
                if p_hn.n_children == 1 && p_hn.n_views == 0 {
                    if self.ctx.tensors[parent.0 as usize].view_src.is_some() {
                        let view_src = self.ctx.tensors[parent.0 as usize].view_src.unwrap();
                        let view_src_hn = *self.hn(view_src);
                        if view_src_hn.n_views == 1 && view_src_hn.n_children == 0 && view_src_hn.allocated {
                            debug_assert_eq!(view_src_hn.offset, p_hn.offset);
                            self.hn(node).offset = p_hn.offset;
                            self.hn(parent).allocated = false;
                            self.hn(view_src).allocated = false;
                            self.free_extra_space(node, view_src);
                            self.assign.push((node, p_hn.offset));
                            return;
                        }
                    } else {
                        self.hn(node).offset = p_hn.offset;
                        self.hn(parent).allocated = false;
                        self.free_extra_space(node, parent);
                        self.assign.push((node, p_hn.offset));
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        let size = self.ctx.nbytes(node);
        let offset = self.plan.alloc(size);
        self.hn(node).offset = offset;
        self.assign.push((node, offset));
    }

    /// `ggml_gallocr_free_node` (ggml-alloc.c:691-712): outputs never freed.
    fn free_node(&mut self, node: TensorId) {
        if self.out_flag(node) {
            return;
        }
        let offset = self.hn(node).offset;
        let size = self.ctx.nbytes(node);
        self.plan.free(offset, size);
        self.hn(node).allocated = false;
    }
}

/// `ggml_gallocr_alloc_graph_impl` (ggml-alloc.c:718-823) over the Context
/// arena. Assigns every planned tensor's storage; pre-allocated tensors
/// (weights, caches, inputs) are untouched, and view tensors resolve into
/// their parents as usual.
fn plan_arena_reuse(ctx: &mut Context, g: &Graph) {
    let mut w = PlanWalk {
        ctx,
        hash: std::collections::HashMap::new(),
        plan: ArenaPlan { free: Vec::new(), top: (ctx.arena.len() + 63) & !63 },
        assign: Vec::new(),
    };

    // allocate leafs (alloc.c:725-728)
    for &leaf in g.leafs.iter() {
        w.allocate_node(leaf);
    }

    // count children/views; allocate flagged inputs first (alloc.c:732-761)
    for &node in g.nodes.iter() {
        let t = &w.ctx.tensors[node.0 as usize];
        if t.view_src.is_some() && t.op != GgmlOp::None {
            w.hn(t.view_src.unwrap()).n_views += 1;
        }
        if t.flags & crate::backend::GGML_TENSOR_FLAG_INPUT != 0 {
            w.allocate_node(node);
        }
        for j in 0..crate::types::MAX_SRC {
            let Some(src) = t.src[j] else { continue };
            w.hn(src).n_children += 1;
            if w.ctx.tensors[src.0 as usize].flags & crate::backend::GGML_TENSOR_FLAG_INPUT != 0 {
                w.allocate_node(src);
            }
        }
    }

    // allocate/free in execution order (alloc.c:764-822)
    for &node in g.nodes.iter() {
        for j in 0..crate::types::MAX_SRC {
            let Some(parent) = w.ctx.tensors[node.0 as usize].src[j] else { continue };
            w.allocate_node(parent);
        }
        w.allocate_node(node);
        for j in 0..crate::types::MAX_SRC {
            let Some(parent) = w.ctx.tensors[node.0 as usize].src[j] else { continue };
            {
                let p_hn = w.hn(parent);
                p_hn.n_children -= 1;
            }
            let p_hn = *w.hn(parent);
            if p_hn.n_children == 0 && p_hn.n_views == 0 {
                if w.ctx.tensors[parent.0 as usize].view_src.is_some() {
                    let view_src = w.ctx.tensors[parent.0 as usize].view_src.unwrap();
                    let v_hn = w.hn(view_src);
                    v_hn.n_views -= 1;
                    if v_hn.n_views == 0 && v_hn.n_children == 0 && v_hn.allocated {
                        w.free_node(view_src);
                    }
                } else if p_hn.allocated {
                    w.free_node(parent);
                }
            }
        }
    }

    for (id, offset) in w.assign {
        ctx.arena_assign_tensor(id, offset);
    }
    debug_assert!(
        g.nodes.iter().chain(g.leafs.iter()).all(|&id| {
            ctx.tensors[id.0 as usize].view_src.is_some()
                || !matches!(ctx.tensors[id.0 as usize].storage, crate::tensor::Storage::None)
        }),
        "planner left a non-view graph tensor without storage"
    );
}

fn graph_execute(ctx: &mut Context, g: &mut Graph, n_threads: usize, arena_ns: u64, phase_t0: std::time::Instant) {

    let nth = n_threads.max(1);
    let callback = eval_callback();
    let prof = opprof_interval();

    // 对照 ggml-cpu.c 的 threadpool: fixed worker set, one spin barrier per op
    let shared = TeamShared {
        gen: std::sync::atomic::AtomicUsize::new(0),
        done: std::sync::atomic::AtomicUsize::new(0),
        quit: std::sync::atomic::AtomicBool::new(false),
        nr: std::sync::atomic::AtomicUsize::new(0),
        job: std::cell::UnsafeCell::new(None),
    };
    std::thread::scope(|scope| {
        let team = Team::new(scope, &shared, nth);
        let cpu = Cpu::new(ctx);
        let node_key = |cpu: &Cpu, node: TensorId| {
            let t = cpu.t(node);
            // src[0] carries the weight's model name ("blk.N.attn_q.weight")
            // and its quant type when the dst node itself is unnamed
            let (src0_name, src0_ty) = t
                .src[0]
                .map(|s| (cpu.t(s).name.clone(), format!("{:?}", cpu.t(s).ty)))
                .unwrap_or_default();
            format!(
                "{}|{}|{}|{:?}|{}x{}x{}",
                t.name, src0_name, src0_ty, t.op, t.ne[0], t.ne[1], t.ne[2]
            )
        };
        // no callback: one branch per graph, nothing added to the node loop
        if callback.is_none() {
            let nodes_t0 = std::time::Instant::now();
            for i in 0..g.nodes.len() {
                let node = g.nodes[i];
                let t0 = std::time::Instant::now();
                cpu.forward(node, nth, &team);
                if prof.is_some() {
                    opprof_record(node_key(&cpu, node), t0.elapsed().as_nanos() as u64);
                }
            }
            if let Some(interval) = phaseprof_interval() {
                eprintln!(
                    "[PHASEPROF] nodes={} arena={} us nodes={} us spawn_to_end={} us arena_len={} MiB cap={} MiB",
                    g.nodes.len(),
                    arena_ns / 1000,
                    nodes_t0.elapsed().as_nanos() as u64 / 1000,
                    phase_t0.elapsed().as_nanos() as u64 / 1000,
                    ctx.arena.len() / 1048576,
                    ctx.arena.capacity() / 1048576
                );
                let _ = interval;
            }
            team.shutdown();
            opprof_maybe_dump();
            return;
        }
        let callback = callback.unwrap();
        for i in 0..g.nodes.len() {
            let node = g.nodes[i];
            // ggml-backend.cpp:1806-1831 with a callback set: ask before the
            // node runs, hand the computed data over after it does.
            let need = callback(&cpu.eval_node(node), true);
            let t0 = std::time::Instant::now();
            cpu.forward(node, nth, &team);
            let dt = t0.elapsed();
            if prof.is_some() {
                opprof_record(node_key(&cpu, node), dt.as_nanos() as u64);
            }
            if need {
                callback(&cpu.eval_node(node), false);
            }
        }
        team.shutdown();
        opprof_maybe_dump();
    });
}

/// Load-time warm-up of the CPU_REPACK caches: materialize the 8x8 interleaved
/// copies of every context tensor that qualifies (the same predicates the
/// forward path applies — `repacked_q4_k` / `repacked_q4_0` /
/// `repacked_mxfp4`), spread over the machine's cores like the reference's
/// multithreaded load-time repack (`ggml_backend_cpu_repack_buffer_type`
/// converts every qualifying weight when the model loads, before any timed
/// forward). The forward path then hits the cache on its first use, so a
/// fresh process's *first* forward no longer pays the repack inside the
/// measurement — mirroring the reference's load-vs-forward cost split.
/// Returns the bytes materialized (0 when `LLAMA_RUST_REPACK=0`).
pub fn warm_repack(ctx: &mut Context) -> u64 {
    // Warm-pass budget: the repack cache's own budget (LLAMA_RUST_REPACK_MAX_MB,
    // 16 GiB default) is sized for the forward path's steady state; the *warm*
    // pass additionally materializes heap while the loader's big mmaps are
    // still resident, which can push a loaded machine past its overcommit
    // limit (observed as ENOMEM with two multi-10-GB test models loading
    // concurrently). Default 1 GiB covers the ≤7 B Q4_K models' full repack;
    // heavier cases (gemma-4-12B ~6 GiB, gpt-oss ~9.7 GiB) raise it via
    // LLAMA_RUST_REPACK_WARM_MAX_MB — parity/bench.sh sets 16 GiB so the
    // benchmark protocol stays reference-faithful.
    let warm_budget: u64 = std::env::var("LLAMA_RUST_REPACK_WARM_MAX_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1024)
        * 1024
        * 1024;
    let cpu = Cpu::new(ctx);
    let n = cpu.tensors.len();
    // candidates: allocated tensors of the repackable types (the helpers
    // re-check geometry; this filter only skips guaranteed misses and
    // Storage::None tensors, whose `rd_rest` would panic)
    let cands: Vec<u32> = (0..n as u32)
        .filter(|&i| {
            let t = cpu.t(TensorId(i));
            matches!(t.ty, GgmlType::Q4K | GgmlType::Q4_0 | GgmlType::Mxfp4)
                && t.view_src.is_none()
                && cpu.view_root_has_storage(TensorId(i))
        })
        .collect();
    let nth = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16).max(1);
    let total = std::sync::atomic::AtomicU64::new(0);
    if cands.is_empty() {
        return 0;
    }
    std::thread::scope(|scope| {
        let chunk = cands.len().div_ceil(nth);
        for shard in cands.chunks(chunk.max(1)) {
            // SAFETY: `Cpu` is a raw view (Send + Sync by design); every
            // worker only *reads* the weights and appends to the repack
            // cache's Mutex-guarded map (repack.rs). Disjoint tensors per
            // shard; the cache itself tolerates concurrent misses.
            let cpu = Cpu {
                tensors: cpu.tensors,
                external: cpu.external,
                arena: cpu.arena,
                arena_len: cpu.arena_len,
            };
            let total = &total;
            scope.spawn(move || {
                for &i in shard {
                    if total.load(std::sync::atomic::Ordering::Relaxed) >= warm_budget {
                        break;
                    }
                    let id = TensorId(i);
                    let rep = cpu
                        .repacked_q4_k(id)
                        .or_else(|| cpu.repacked_q4_0(id, true))
                        .or_else(|| cpu.repacked_q4_0(id, false))
                        .or_else(|| cpu.repacked_mxfp4(id));
                    if let Some(r) = rep {
                        total.fetch_add(r.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    total.load(std::sync::atomic::Ordering::Relaxed)
}

/// Load-time prefault of every external (mmap) weight buffer: touch one byte
/// per page so the *first* forward does not pay the minor-fault storm. The
/// reference's loader reads/walks the whole file while materializing tensors
/// (plus `posix_fadvise(WILLNEED)`), so its first timed forward starts with
/// resident pages; the port's pure-mmap load leaves them untouched. The pass
/// reads file-backed pages that are already in the page cache (the loader
/// just read the file), so it costs one cheap sweep at load, not disk I/O.
/// Returns the bytes swept.
pub fn prefault_weights(ctx: &Context) -> u64 {
    let mut total = 0u64;
    for buf in &ctx.external {
        let bytes = buf
            .downcast_ref::<Vec<u8>>()
            .map(|v| v.as_slice())
            .or_else(|| buf.downcast_ref::<memmap2::Mmap>().map(|m| m.as_ref()));
        if let Some(b) = bytes {
            // one touch per 4 KiB page (black_box keeps the read un-elided)
            for page in b.chunks(4096) {
                if let Some(&first) = page.first() {
                    std::hint::black_box(first);
                }
            }
            total += b.len() as u64;
        }
    }
    total
}

/// Raw view over the context storages. All kernels work through this: reads
/// may hit the arena or external mmap memory, writes only the arena.
struct Cpu<'a> {
    tensors: &'a [TensorMeta],
    external: &'a [std::sync::Arc<dyn std::any::Any + Send + Sync>],
    arena: *mut u8,
    arena_len: usize,
}

unsafe impl Send for Cpu<'_> {}
unsafe impl Sync for Cpu<'_> {}

impl<'a> Cpu<'a> {
    fn new(ctx: &'a mut Context) -> Self {
        let arena = ctx.arena.as_mut_ptr();
        let arena_len = ctx.arena.len();
        Cpu { tensors: &ctx.tensors, external: &ctx.external, arena, arena_len }
    }

    #[inline]
    fn t(&self, id: TensorId) -> &TensorMeta {
        &self.tensors[id.0 as usize]
    }

    /// Follow the view chain (tensor.rs Context::resolve) → (base storage ptr,
    /// writable, accumulated view byte offset, bytes available from the base
    /// storage pointer to the end of its allocation).
    fn loc(&self, id: TensorId) -> (*mut u8, bool, usize, usize) {
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
        let t = &self.tensors[base.0 as usize];
        match &t.storage {
            Storage::Arena { offset, .. } => {
                (unsafe { self.arena.add(*offset) }, true, off, self.arena_len - offset)
            }
            Storage::External { offset, len } => {
                let (idx, off2) = (*offset >> 48, *offset & ((1usize << 48) - 1));
                let buf = &self.external[idx];
                let ptr: *const u8 = buf
                    .downcast_ref::<Vec<u8>>()
                    .map(|v| v.as_ptr())
                    .or_else(|| buf.downcast_ref::<memmap2::Mmap>().map(|m| m.as_ptr()))
                    .expect("external storage buffer vanished") as *const u8;
                (unsafe { ptr.add(off2) } as *mut u8, false, off, *len)
            }
            Storage::None => panic!("tensor {} has no storage (not allocated?)", t.name),
        }
    }

    /// read `len` bytes at byte offset `off` (relative to the view start)
    #[inline]
    fn rd(&self, id: TensorId, off: usize, len: usize) -> &[u8] {
        let (base, _writable, voff, rest) = self.loc(id);
        debug_assert!(voff + off + len <= rest, "read out of storage bounds");
        unsafe { std::slice::from_raw_parts(base.add(voff + off), len) }
    }

    /// read from byte offset `off` to the end of the tensor's storage
    /// (rows of strided views may run past the view's own nbytes)
    #[inline]
    fn rd_rest(&self, id: TensorId, off: usize) -> &[u8] {
        let (base, _writable, voff, rest) = self.loc(id);
        debug_assert!(voff + off <= rest);
        unsafe { std::slice::from_raw_parts(base.add(voff + off), rest - voff - off) }
    }

    /// write `len` bytes at byte offset `off` (relative to the view start)
    #[inline]
    fn wr(&self, id: TensorId, off: usize, len: usize) -> &mut [u8] {
        let (base, writable, voff, rest) = self.loc(id);
        assert!(writable, "attempt to write external (mmap) storage");
        debug_assert!(voff + off + len <= rest, "write out of storage bounds");
        unsafe { std::slice::from_raw_parts_mut(base.add(voff + off), len) }
    }

    #[inline]
    fn rd_f32s(&self, id: TensorId, off: usize, n: usize) -> &[f32] {
        bytemuck::cast_slice(self.rd(id, off, n * 4))
    }
    #[inline]
    fn wr_f32s(&self, id: TensorId, off: usize, n: usize) -> &mut [f32] {
        bytemuck::cast_slice_mut(self.wr(id, off, n * 4))
    }
    #[inline]
    fn rd_i32s(&self, id: TensorId, off: usize, n: usize) -> &[i32] {
        bytemuck::cast_slice(self.rd(id, off, n * 4))
    }

    /// Build the [`EvalNode`] view of a node for the eval callback. Sources whose
    /// *view root* has no storage (never allocated) yield `data: None` — a view
    /// borrows its root's buffer, exactly as `->data` would in C.
    fn eval_node(&self, id: TensorId) -> EvalNode<'_> {
        let t = self.t(id);
        let mut src: [Option<EvalSrc<'_>>; MAX_SRC] = Default::default();
        for (i, s) in t.src.iter().enumerate() {
            let Some(sid) = *s else { continue };
            let st = self.t(sid);
            let data = if self.view_root_has_storage(sid) { Some(self.rd_rest(sid, 0)) } else { None };
            src[i] = Some(EvalSrc { name: &st.name, ty: st.ty, ne: st.ne, nb: st.nb, data });
        }
        let data = if self.view_root_has_storage(id) { Some(self.rd_rest(id, 0)) } else { None };
        EvalNode {
            op: t.op,
            name: &t.name,
            ty: t.ty,
            ne: t.ne,
            nb: t.nb,
            op_params: t.op_params,
            data,
            src,
        }
    }

    /// Follow `view_src` to the tensor that owns the buffer (`loc`'s walk) and
    /// report whether that root is allocated.
    fn view_root_has_storage(&self, id: TensorId) -> bool {
        let mut base = id;
        loop {
            let t = self.t(base);
            match t.view_src {
                Some(v) => base = v,
                None => break,
            }
        }
        !matches!(self.t(base).storage, Storage::None)
    }

    /// Run `f(ir0, ir1)` over contiguous row ranges split across `nth` shards,
    /// mirroring the C `dr/ir0/ir1` thread partitioning.
    fn par_rows(
        &self,
        nr: usize,
        nth: usize,
        team: &Team<'_>,
        f: impl Fn(usize, usize) + Send + Sync,
    ) {
        if nr == 0 {
            return;
        }
        if nth <= 1 || nr == 1 {
            f(0, nr);
            return;
        }
        team.run(nr, &f);
    }

    /// The C `mul_mat` / `mul_mat_id` work split (ggml-cpu.c:1404-1450): a
    /// `ceil(nr0/chunk_size) x ceil(nr1/chunk_size)` grid of chunks, re-chunked
    /// per thread when the grid is too coarse (`nchunk0*nchunk1 < nth*4`).
    /// That fallback is what gives the single-column (token generation) case
    /// its `nth`-way *row* split — without it a one-token mul_mat would run on
    /// a single thread. `rowchunk` keeps `ir0` blocks aligned for the 8-row
    /// group kernels (pass 1 otherwise).
    ///
    /// Chunk boundaries only decide *which* thread computes an output element,
    /// never how: every element is still one `vec_dot`/`gemv` call, so results
    /// stay independent of the thread count (asserted by the mul_mat tests).
    fn par_mul_mat(
        &self,
        nr0: usize,
        nr1: usize,
        rowchunk: usize,
        nth: usize,
        team: &Team<'_>,
        f: impl Fn(usize, usize, usize, usize) + Send + Sync,
    ) {
        if nr0 == 0 || nr1 == 0 {
            return;
        }
        let nth = nth.max(1);
        let chunk_size = if nr0 == 1 || nr1 == 1 { 64 } else { 16 };
        let mut nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        let mut nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
        if nchunk0 * nchunk1 < nth * 4 {
            nchunk0 = if nr0 > nr1 { nth } else { 1 };
            nchunk1 = if nr0 > nr1 { 1 } else { nth };
        }
        let rowchunk = rowchunk.max(1);
        let dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        let dr0 = (dr0 + rowchunk - 1) / rowchunk * rowchunk;
        let nchunk0 = (nr0 + dr0 - 1) / dr0;
        let dr1 = (nr1 + nchunk1 - 1) / nchunk1;
        let jobs = nchunk0 * nchunk1;

        let run_chunk = |c: usize| {
            let ith0 = c % nchunk0;
            let ith1 = c / nchunk0;
            let ir0 = (dr0 * ith0).min(nr0);
            let ir1 = (ir0 + dr0).min(nr0);
            let ic0 = (dr1 * ith1).min(nr1);
            let ic1 = (ic0 + dr1).min(nr1);
            if ir0 < ir1 && ic0 < ic1 {
                f(ir0, ir1, ic0, ic1);
            }
        };
        let run = |lo: usize, hi: usize| {
            for c in lo..hi {
                run_chunk(c);
            }
        };
        if jobs == 1 {
            run_chunk(0);
            return;
        }
        // NB: `Team::run` walks every shard, so the single-thread case (n == 1)
        // still runs *all* chunks of the C grid, not just the first.
        team.run(jobs, &run);
    }

    // ==================================================================
    // dispatch — 对照 ggml_graph_compute (ggml-cpu.c:2360+) op switch
    // ==================================================================

    fn forward(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        self.forward_op(dst, nth, team, self.t(dst).op);
    }

    fn forward_op(&self, dst: TensorId, nth: usize, team: &Team<'_>, op: GgmlOp) {
        match op {
            GgmlOp::Dup => self.forward_dup(dst, nth, team), // also GGML_OP_CONT (forward_cont == forward_dup)
            GgmlOp::Cpy => self.forward_dup(dst, nth, team), // ggml_compute_forward_cpy → dup
            GgmlOp::Add => self.forward_bin_op(dst, nth, team, BinOp::Add),
            GgmlOp::Mul => self.forward_bin_op(dst, nth, team, BinOp::Mul),
            GgmlOp::Div => self.forward_bin_op(dst, nth, team, BinOp::Div),
            // audio round 4: ggml_sub (binary-ops.cpp:144)
            GgmlOp::Sub => self.forward_bin_op(dst, nth, team, BinOp::Sub),
            GgmlOp::Clamp => self.forward_clamp(dst, nth, team),
            GgmlOp::MulMat => self.forward_mul_mat(dst, nth, team),
            GgmlOp::MulMatId => self.forward_mul_mat_id(dst, nth, team),
            GgmlOp::AddId => self.forward_add_id(dst, nth, team),
            GgmlOp::Glu => self.forward_glu(dst, nth, team),
            GgmlOp::Norm => {
                if self.t(dst).op_params[1] == OP_FLAG_NORM_IS_RMS {
                    self.forward_rms_norm(dst, nth, team);
                } else if self.t(dst).op_params[1] == OP_FLAG_NORM_IS_L2 {
                    self.forward_l2_norm(dst, nth, team);
                } else {
                    self.forward_norm(dst, nth, team);
                }
            }
            GgmlOp::Scale => self.forward_scale(dst, nth, team),
            GgmlOp::SoftMax => self.forward_soft_max(dst, nth, team),
            GgmlOp::Silu => {
                // GgmlOp::Silu == GGML_OP_UNARY; params[0] picks the unary op
                match self.t(dst).op_params[0] {
                    GGML_UNARY_OP_TANH => self.forward_tanh(dst, nth, team),
                    GGML_UNARY_OP_SIGMOID => self.forward_sigmoid(dst, nth, team),
                    GGML_UNARY_OP_RELU => self.forward_relu(dst, nth, team),
                    // audio round 4 (pocket-tts SEANet): (x>0)?x:expm1f(x)
                    GGML_UNARY_OP_ELU => self.forward_elu(dst, nth, team),
                    GGML_UNARY_OP_GELU => self.forward_gelu(dst, nth, team),
                    GGML_UNARY_OP_GELU_ERF => self.forward_gelu_erf(dst, nth, team),
                    GGML_UNARY_OP_SILU => self.forward_silu(dst, nth, team),
                    GGML_UNARY_OP_EXP => self.forward_exp(dst, nth, team),
                    GGML_UNARY_OP_SOFTPLUS => self.forward_softplus(dst, nth, team),
                    // arch batch 10 (graniteswitch): roundf per element
                    GGML_UNARY_OP_ROUND => self.forward_round(dst, nth, team),
                    // arch batch 11a (apertus): the parameterized xielu functor
                    GGML_UNARY_OP_XIELU => self.forward_xielu(dst, nth, team),
                    // audio round 5 (TTS generators): (x > 0) ? 1 : 0
                    GGML_UNARY_OP_STEP => self.forward_step(dst, nth, team),
                    // arch batch 18 (chunked delta-net): -x (unary-ops.cpp:11)
                    GGML_UNARY_OP_NEG => self.forward_neg(dst, nth, team),
                    // qwen4exp PLE (batch 19): fabsf (unary-ops.cpp:3)
                    GGML_UNARY_OP_ABS => self.forward_abs(dst, nth, team),
                    // qwen4exp PLE (batch 19): sign (unary-ops.cpp:7)
                    GGML_UNARY_OP_SGN => self.forward_sgn(dst, nth, team),
                    other => unimplemented!("unary op {other}"),
                }
            }
            GgmlOp::RoPE => self.forward_rope(dst, nth, team),
            GgmlOp::GetRows => {
                if self.t(dst).op_params[15] == OP_FLAG_GET_ROWS_BACK {
                    self.forward_get_rows_back(dst);
                } else {
                    self.forward_get_rows(dst, nth, team);
                }
            }
            GgmlOp::DiagMaskInf => self.forward_diag_mask_inf(dst, nth, team, f32::NEG_INFINITY),
            GgmlOp::Argsort => self.forward_argsort(dst),
            GgmlOp::ArgMax => self.forward_argmax(dst),
            GgmlOp::SumRows => self.forward_sum_rows(dst),
            GgmlOp::Repeat => self.forward_repeat(dst),
            GgmlOp::Concat => self.forward_concat(dst, nth, team),
            GgmlOp::SetRows => self.forward_set_rows(dst, nth, team),
            GgmlOp::FlashAttnExt => self.forward_flash_attn_ext(dst, nth, team),
            GgmlOp::SsmConv => self.forward_ssm_conv(dst, nth, team),
            GgmlOp::SsmScan => self.forward_ssm_scan(dst, nth, team),
            GgmlOp::Gdn => self.forward_gated_delta_net(dst, nth, team),
            // batch 14 round 2 (agent WKV): the fused RWKV kernels (wkv.rs)
            GgmlOp::RwkvWkv6 => self.forward_rwkv_wkv6(dst, nth, team),
            GgmlOp::GatedLinearAttn => self.forward_gated_linear_attn(dst, nth, team),
            GgmlOp::RwkvWkv7 => self.forward_rwkv_wkv7(dst, nth, team),
            GgmlOp::Im2col => self.forward_im2col(dst, nth, team),
            GgmlOp::Upscale => self.forward_upscale(dst, nth, team),
            GgmlOp::Fill => self.forward_fill(dst, nth, team),
            GgmlOp::LightningIndexer => self.forward_lightning_indexer(dst, nth, team),
            GgmlOp::TopK => self.forward_top_k(dst),
            GgmlOp::RoPEBack => self.forward_rope(dst, nth, team),
            // arch batch 7 (deepseek4): sqrt + the fused hyper-connection ops
            GgmlOp::Sqrt => self.forward_sqrt(dst, nth, team),
            // arch batch 10 (graniteswitch): the router lane's right-pad
            GgmlOp::Pad => self.forward_pad(dst, nth, team),
            // arch batch 11a (minimax-m3): the MSA block-score pooling + the
            // decode-path index arithmetic
            GgmlOp::Pool2d => self.forward_pool_2d(dst),
            // audio round 2 (whisper-enc family): the literal nn.AvgPool1d op
            GgmlOp::Pool1d => self.forward_pool_1d(dst),
            // audio round 3 (conformer family): the rel-shift roll + the
            // direct convolutions
            GgmlOp::Roll => self.forward_roll(dst, nth, team),
            GgmlOp::Conv2dDw => self.forward_conv_2d_dw(dst, nth, team),
            // audio round 4 (parakeet/mimo/spkenc): the new op surface
            GgmlOp::Sin => self.forward_sin_cos(dst, nth, team, false),
            GgmlOp::Cos => self.forward_sin_cos(dst, nth, team, true),
            GgmlOp::Sqr => self.forward_sqr(dst, nth, team),
            GgmlOp::Mean => self.forward_mean(dst),
            GgmlOp::PadReflect1d => self.forward_pad_reflect_1d(dst, nth, team),
            // audio round 5 (TTS generators): sum / cumsum / tri / log /
            // col2im_1d (GGML_OP_LOG is its own op; `step` is a UNARY variant
            // handled inside the Silu arm above)
            GgmlOp::Sum => self.forward_sum(dst),
            GgmlOp::Cumsum => self.forward_cumsum(dst, nth, team),
            GgmlOp::Tri => self.forward_tri(dst, nth, team),
            // arch batch 18 (chunked delta-net): the GGML_OP_SET scatter +
            // GGML_OP_DIAG identity + GGML_OP_SOLVE_TRI UT transform
            // (delta-net-base.cpp:162-171/262)
            GgmlOp::Set => self.forward_set(dst),
            GgmlOp::Diag => self.forward_diag(dst),
            GgmlOp::SolveTri => self.forward_solve_tri(dst, nth, team),
            GgmlOp::Log => self.forward_log(dst),
            GgmlOp::Col2Im1d => self.forward_col2im_1d(dst, nth, team),
            GgmlOp::Arange => self.forward_arange(dst, nth, team),
            GgmlOp::Dsv4HcComb => self.forward_dsv4_hc_comb(dst, nth, team),
            GgmlOp::Dsv4HcPre => self.forward_dsv4_hc_pre(dst, nth, team),
            GgmlOp::Dsv4HcPost => self.forward_dsv4_hc_post(dst, nth, team),
            // layout-only ops: data already shared/positioned via views
            GgmlOp::View | GgmlOp::Reshape | GgmlOp::Permute | GgmlOp::Transpose => {}
            other => unimplemented!("compute_forward for {other:?}"),
        }
    }

    // ==================================================================
    // add / mul — 对照 binary-ops.cpp apply_binary_op (can_repeat broadcast)
    //
    // Documented skip (AUDIT_ggml.md §5-A.3): the reference's *quantized*
    // src0/dst add path — `ggml_compute_forward_add_q_f32` (ops.cpp:578, a
    // generic dequantize-row → ggml_vec_acc_f32 → requantize-row loop,
    // dispatched from ggml_compute_forward_add's quantized cases at
    // ops.cpp:654-693) — is NOT ported. It is not a binary-ops.cpp kernel
    // family: sub/mul/div GGML_ABORT on quantized types there (binary-ops.cpp
    // `binary_op<op>`'s final else). Its sole consumer is control vectors
    // (`ggml_add` on a quantized tensor, llama.cpp cvec), which the llama
    // layer already skips (FILE_MAP:120); sub/mul/div never see quantized src0
    // anywhere in src/. When cvec is ported this site must grow the
    // dequant-acc-requant loop (and the `type_traits_cpu.from_float` requant
    // entry points it needs).
    // ==================================================================

    fn forward_bin_op(&self, dst: TensorId, nth: usize, team: &Team<'_>, op: BinOp) {
        let s0 = self.t(dst).src[0].expect("bin op src0");
        let s1 = self.t(dst).src[1].expect("bin op src1");
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);

        let ty = t0.ty;
        let esz = ty.type_size();
        let to_f32: fn(&[u8]) -> f32 = match ty {
            GgmlType::F32 => |p| f32::from_le_bytes(p.try_into().unwrap()),
            GgmlType::F16 => |p| f16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            GgmlType::Bf16 => |p| bf16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            other => unimplemented!("bin op src0 type {other:?}"),
        };
        let ty1 = t1.ty;
        let to_f32_1: fn(&[u8]) -> f32 = match ty1 {
            GgmlType::F32 => |p| f32::from_le_bytes(p.try_into().unwrap()),
            GgmlType::F16 => |p| f16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            GgmlType::Bf16 => |p| bf16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            other => unimplemented!("bin op src1 type {other:?}"),
        };
        let tyd = td.ty;
        let from_f32: fn(f32, &mut [u8]) = match tyd {
            GgmlType::F32 => |v, p| p.copy_from_slice(&v.to_le_bytes()),
            GgmlType::F16 => |v, p| p.copy_from_slice(&f16::from_f32(v).to_le_bytes()),
            GgmlType::Bf16 => |v, p| p.copy_from_slice(&bf16::from_f32(v).to_le_bytes()),
            other => unimplemented!("bin op dst type {other:?}"),
        };

        let (ne00, ne01, ne02, ne03) = (t0.ne[0], t0.ne[1], t0.ne[2], t0.ne[3]);
        let (ne10, ne11, ne12, ne13) = (t1.ne[0], t1.ne[1], t1.ne[2], t1.ne[3]);
        let (nb00, nb01, nb02, nb03) = (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb10, nb11, nb12, nb13) = (t1.nb[0] as usize, t1.nb[1] as usize, t1.nb[2] as usize, t1.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let dst_blck = tyd.blck_size();

        let nr = (ne01 * ne02 * ne03) as usize;
        let is_src1_cont_rows = is_contiguous_ctx_of(self, s1) && nb10 == ty1.type_size();
        let ne0 = td.ne[0] as usize;

        // per-lane scalar of binary-ops.cpp's templates (`op_add` etc.)
        let op_fn = match op {
            BinOp::Add => (|a, b| a + b) as fn(f32, f32) -> f32,
            BinOp::Mul => (|a, b| a * b) as fn(f32, f32) -> f32,
            BinOp::Div => (|a, b| a / b) as fn(f32, f32) -> f32,
            // vec.h:112 ggml_vec_sub_f32 (scalar loop, no SIMD variant)
            BinOp::Sub => (|a, b| a - b) as fn(f32, f32) -> f32,
        };

        // Hot path: F32 × F32 → F32 with contiguous element strides runs the
        // same per-lane op as the reference's auto-vectorized
        // `vec_binary_op_contiguous` loop — add/mul via the AVX512 lane
        // kernels (ggml_vec_add_f32 vec.h:89 / ggml_vec_mul_f32 vec.h:128;
        // one rounded op per lane, bit-identical at any width), div scalar
        // per element (vdivps == f32 division either way). A one-element src1
        // row broadcasts (nrep repeats of ne10 == 1): add is ggml_vec_add1_f32
        // (vec.h:105), mul is exactly ggml_vec_scale_f32 (vec.h:703).
        let fast_f32 = ty == GgmlType::F32
            && ty1 == GgmlType::F32
            && tyd == GgmlType::F32
            && nb00 == 4
            && nb10 == 4
            && nb0 == 4
            && is_src1_cont_rows;

        // element-wise over the dst row (vec_binary_op_non_contiguous)
        let run = move |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * ne01) as usize;
                let i02 = (ir - i03 * (ne02 * ne01) as usize) / ne01 as usize;
                let i01 = ir - i03 * (ne02 * ne01) as usize - i02 * ne01 as usize;

                let i13 = i03 % ne13 as usize;
                let i12 = i02 % ne12 as usize;
                let i11 = i01 % ne11 as usize;

                let dst_row = i03 * nb3 + i02 * nb2 + i01 * nb1;
                let src0_row = i03 * nb03 + i02 * nb02 + i01 * nb01;
                let src1_row = i13 * nb13 + i12 * nb12 + i11 * nb11;

                if fast_f32 && nb00 == esz {
                    // src1 row broadcast across the row (nr0 = ne00/ne10 repeats)
                    let nrep = (ne00 as usize) / (ne10 as usize);
                    if ne10 as usize == 1 {
                        let y0 = f32::from_le_bytes(
                            self.rd(s1, src1_row, 4).try_into().unwrap(),
                        );
                        let xs: &[f32] =
                            bytemuck::cast_slice(self.rd(s0, src0_row, ne00 as usize * 4));
                        let zs: &mut [f32] =
                            bytemuck::cast_slice_mut(self.wr(dst, dst_row, ne00 as usize * 4));
                        match op {
                            BinOp::Add => {
                                for (z, &x) in zs.iter_mut().zip(xs) {
                                    *z = x + y0;
                                }
                            }
                            BinOp::Mul => crate::simd_x86::vec_mul1_f32(zs, xs, y0),
                            BinOp::Div => {
                                for (z, &x) in zs.iter_mut().zip(xs) {
                                    *z = x / y0;
                                }
                            }
                            // vsubps lane == scalar f32 subtraction; the
                            // reference's vec loop is auto-vectorized the
                            // same way (binary-ops.cpp op_sub)
                            BinOp::Sub => {
                                for (z, &x) in zs.iter_mut().zip(xs) {
                                    *z = x - y0;
                                }
                            }
                        }
                        continue;
                    }
                    for r in 0..nrep {
                        let xs: &[f32] = bytemuck::cast_slice(
                            self.rd(s0, src0_row + r * ne10 as usize * 4, ne10 as usize * 4),
                        );
                        let ys: &[f32] =
                            bytemuck::cast_slice(self.rd(s1, src1_row, ne10 as usize * 4));
                        let zs: &mut [f32] = bytemuck::cast_slice_mut(
                            self.wr(dst, dst_row + r * ne10 as usize * 4, ne10 as usize * 4),
                        );
                        match op {
                            BinOp::Add => crate::simd_x86::vec_add_f32(zs, xs, ys),
                            BinOp::Mul => crate::simd_x86::vec_mul_f32(zs, xs, ys),
                            BinOp::Div => {
                                for ((z, &x), &y) in zs.iter_mut().zip(xs).zip(ys) {
                                    *z = x / y;
                                }
                            }
                            BinOp::Sub => {
                                for ((z, &x), &y) in zs.iter_mut().zip(xs).zip(ys) {
                                    *z = x - y;
                                }
                            }
                        }
                    }
                } else if is_src1_cont_rows && nb00 == esz {
                    // src1 row broadcast across the row (nr0 = ne00/ne10 repeats)
                    let nrep = (ne00 as usize) / (ne10 as usize);
                    for r in 0..nrep {
                        for j in 0..ne10 as usize {
                            let x = to_f32(&self.rd(s0, src0_row + (r * ne10 as usize + j) * nb00, esz));
                            let y = to_f32_1(&self.rd(s1, src1_row + j * nb10, ty1.type_size()));
                            let v = op_fn(x, y);
                            from_f32(v, self.wr(dst, dst_row + (r * ne10 as usize + j) * nb0, nb0));
                        }
                    }
                } else {
                    for i in 0..ne0 {
                        let i10 = i % ne10 as usize;
                        let x = to_f32(&self.rd(s0, src0_row + i * nb00, esz));
                        let y = to_f32_1(&self.rd(s1, src1_row + i10 * nb10, ty1.type_size()));
                        let v = op_fn(x, y);
                        from_f32(v, self.wr(dst, dst_row + (i / dst_blck) * nb0, nb0));
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // mul_mat — 对照 ggml-cpu.c:1255 ggml_compute_forward_mul_mat
    // ==================================================================

    /// One `llamafile_sgemm` attempt over every `(i12, i13)` broadcast plane
    /// (ggml-cpu.c:1304-1318 = attempt 1, :1387-1401 = attempt 2): for each
    /// plane the `ne01 × ne11` GEMM over `ne00/blck_size` values is handed to
    /// `tinyblas::gemm`, split across the team into `RM`-row panels (every
    /// output element is one independent accumulator, so the split cannot
    /// change a value — see tinyblas.rs's header).
    ///
    /// `b` holds the activation planes in `Btype` form with `b_ldb` units per
    /// row and `b_off2`/`b_off3` their stride per `i12`/`i13`: attempt 1 passes
    /// `src1`'s own storage, attempt 2 the converted `vec_dot_type` rows
    /// (`src1->data` / `wdata` in the C).
    #[allow(clippy::too_many_arguments)]
    fn sgemm_planes(
        &self,
        dst: TensorId,
        s0: TensorId,
        op: crate::tinyblas::Op,
        b: &[u8],
        b_ldb: usize,
        b_off2: usize,
        b_off3: usize,
        nth: usize,
        team: &Team<'_>,
    ) {
        use crate::tinyblas;
        let t0 = self.t(s0);
        let td = self.t(dst);
        let ty0 = t0.ty;
        let k = (t0.ne[0] as usize) / ty0.blck_size();
        let m = t0.ne[1] as usize;
        let lda = (t0.nb[1] as usize) / ty0.type_size();
        let n = td.ne[1] as usize;
        let (ne12, ne13) = (td.ne[2] as usize, td.ne[3] as usize);
        assert_eq!(ne12, td.ne[2] as usize, "sgemm: ne12 != ne2");
        let (ne02, ne03) = (t0.ne[2] as usize, t0.ne[3] as usize);
        let (nb02, nb03) = (t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb2, nb3) = (td.nb[2] as usize, td.nb[3] as usize);
        // broadcast factors (ggml-cpu.c:1298-1299)
        let (r2, r3) = (ne12 / ne02, ne13 / ne03);
        let jobs = m.div_ceil(tinyblas::RM);
        for i13 in 0..ne13 {
            for i12 in 0..ne12 {
                let a_base = i12 / r2 * nb02 + i13 / r3 * nb03;
                let b_base = i12 * b_off2 + i13 * b_off3;
                let d_base = i12 * nb2 + i13 * nb3;
                self.par_rows(jobs, nth, team, |j0, j1| {
                    let a = self.rd_rest(s0, a_base);
                    let c = self.wr_f32s(dst, d_base, m * n);
                    let i0 = j0 * tinyblas::RM;
                    let i1 = (j1 * tinyblas::RM).min(m);
                    // SAFETY: `a` reaches the end of src0's storage from the
                    // plane base (rows are `lda` units apart, all within it),
                    // `c` covers the whole plane and `b_base + n*b_ldb` stays
                    // inside `b` (the caller sized it per plane).
                    unsafe {
                        tinyblas::gemm(
                            op,
                            m,
                            n,
                            k,
                            a.as_ptr(),
                            lda,
                            b.as_ptr().add(b_base),
                            b_ldb,
                            c.as_mut_ptr(),
                            m,
                            i0,
                            i1,
                            0,
                            n,
                        );
                    }
                });
            }
        }
    }

    fn forward_mul_mat(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        // ggml-cpu.c:1262-1267 — a hinted Hadamard src0 routes to the fast
        // Walsh-Hadamard transform kernel (ops.cpp:12027) instead of the gemm
        // dot; the two accumulate in different orders, so the hint is a
        // numeric contract, not a scheduling one
        if self.t(dst).op_params[1] == crate::ops::GGML_HINT_SRC0_IS_HADAMARD {
            self.forward_fwht(dst, nth, team);
            return;
        }
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);

        let ty0 = t0.ty;
        debug_assert_eq!(t0.nb[0] as usize, ty0.type_size(), "mul_mat: permuted src0");
        debug_assert_eq!(t1.nb[0] as usize, t1.ty.type_size(), "mul_mat: permuted src1");
        debug_assert_eq!(td.nb[0] as usize, 4);

        let (ne00, ne01, ne02, ne03) = (t0.ne[0], t0.ne[1], t0.ne[2], t0.ne[3]);
        let (ne10, ne11, ne12, ne13) = (t1.ne[0], t1.ne[1], t1.ne[2], t1.ne[3]);
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb11, nb12, nb13) = (t1.nb[1] as usize, t1.nb[2] as usize, t1.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let (ne1, ne2, ne3) = (td.ne[1], td.ne[2], td.ne[3]);

        assert_eq!(ne00, ne10, "mul_mat: ne00 != ne10");
        // broadcast factors (ggml-cpu.c:1186)
        let r2 = ne12 / ne02;
        let r3 = ne13 / ne03;

        let n = ne00 as usize;
        // type_traits_cpu[].vec_dot_type (ggml-cpu.c:215-416): everything but
        // MXFP4 (compute.rs-local, see mxfp4_vec_dot_type) and the ternary
        // pair is wired through vec_dot::vec_dot_type since the audit round —
        // TQ1_0/TQ2_0 remain a documented refusal (AUDIT_ggml.md §5-B.4: the
        // whole ternary chain — quantizer, dequantizer and the quants.c:481/
        // 533 + arch/x86/quants.c:1376/1508 vec_dot kernels — is unported,
        // and the reference's own consumers are test fixtures only).
        let vdt = vec_dot::vec_dot_type(ty0)
            .or_else(|| mxfp4_vec_dot_type(ty0))
            .unwrap_or_else(|| {
                panic!(
                    "mul_mat: no vec_dot for {ty0:?} — ternary (TQ1_0/TQ2_0) weights are a \
                     documented refusal: the full TQ chain is unported (see PARITY.md)"
                )
            });
        assert_eq!(t1.ty, GgmlType::F32, "mul_mat: src1 must be F32");

        // ---- CPU_REPACK: Q4_K 8x8 (repack.rs) ----
        // The reference routes *every* Q4_K tensor with `ne[1] % 8 == 0` through
        // the CPU_REPACK buffer (repack.cpp:5006-5011, `ggml_cpu_has_avx2() &&
        // ne[1] % 8 == 0`) and then runs `ggml_gemm_q4_K_8x8_q8_K` for whole
        // groups of 4 activation rows plus `ggml_gemv_q4_K_8x8_q8_K` for the
        // `nrows % 4` tail (repack.cpp:4638-4647). Q5_K/Q6_K have no x86 repack
        // instance (repack.cpp:5050-5071), so they keep the row-wise vec_dot
        // below, exactly like the reference. The decision is taken here because
        // the repack route builds its own activation buffer (the C's wdata
        // layout with 4-row interleaved `block_q8_Kx4` tiles), which makes the
        // generic plain q8_K pass below unnecessary for it.
        let repacked_q4k = if ty0 == GgmlType::Q4K { self.repacked_q4_k(s0) } else { None };

        // ---- CPU_REPACK: Q4_0 8x8 (repack.rs) — the intercept BEFORE llamafile ----
        // Unlike Q4_K (which llamafile never accepts), Q4_0 *is* a llamafile
        // type, so the routing order is observable: in the reference the
        // repack traits intercept inside `ggml_compute_forward` *before* the
        // op switch (ggml-cpu.c:1751-1753 `ggml_cpu_extra_compute_forward`),
        // i.e. before `ggml_compute_forward_mul_mat`'s two `llamafile_sgemm`
        // attempts. Every 2D Q4_0 tensor with `ne[1] % 8 == 0` therefore takes
        // the 8x8 gemm/gemv for *all* its mul_mats (n == 1 decode included —
        // the reference decodes the repacked weights there too); a tensor that
        // does not qualify keeps the tinyBLAS/vec_dot routes below, exactly
        // like a reference tensor that never entered the CPU_REPACK buffer.
        let repacked_q4_0 = if ty0 == GgmlType::Q4_0 {
            self.repacked_q4_0(s0, /*require_2d*/ true)
        } else {
            None
        };

        // ---- step 1: quantize src1 rows into contiguous wdata (if needed) ----
        let row_size = |ty: GgmlType| (ty.type_size() * n / ty.blck_size()) as usize;
        // F16/BF16 weights convert activations inside their branch below
        let uses_wd = !matches!(ty0, GgmlType::F16 | GgmlType::Bf16);
        let wdata: Option<Vec<u8>> = if uses_wd && t1.ty != vdt && repacked_q4k.is_none() && repacked_q4_0.is_none() {
            let rs = row_size(vdt);
            let total = (ne11 * ne12 * ne13) as usize * rs;
            let mut wd = vec![0u8; total];
            let nrows = (ne11 * ne12 * ne13) as usize;
            {
                let wd_ptr = std::sync::atomic::AtomicPtr::new(wd.as_mut_ptr());
                self.par_rows(nrows, nth, team, |ir0, ir1| {
                    for ir in ir0..ir1 {
                        let i13 = ir / (ne11 * ne12) as usize;
                        let i12 = (ir - i13 * (ne11 * ne12) as usize) / ne11 as usize;
                        let i11 = ir - i13 * (ne11 * ne12) as usize - i12 * ne11 as usize;
                        let src_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                        let row: &[f32] = self.rd_f32s(s1, src_off, n);
                        let dst_off = ir * rs;
                        let p = wd_ptr.load(std::sync::atomic::Ordering::Relaxed);
                        let out = unsafe { std::slice::from_raw_parts_mut(p.add(dst_off), rs) };
                        match vdt {
                            GgmlType::Q8_0 => {
                                quants::quantize_row_q8_0(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::Q8_1 => {
                                quants::quantize_row_q8_1(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::Q8K => {
                                quants::quantize_row_q8_K(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::F32 => {
                                out.copy_from_slice(self.rd(s1, src_off, n * 4));
                            }
                            GgmlType::Bf16 => {
                                let row: &[f32] = self.rd_f32s(s1, src_off, n);
                                let o: &mut [half::bf16] = bytemuck::cast_slice_mut(out);
                                for j in 0..n {
                                    o[j] = half::bf16::from_f32(row[j]);
                                }
                            }
                            other => unimplemented!("wdata conversion to {other:?}"),
                        }
                    }
                });
            }
            Some(wd)
        } else {
            None
        };

        let nr0 = ne01 as usize; // weight rows (dst dim0)
        let nr1 = (ne1 * ne2 * ne3) as usize; // activation columns

        let wd = wdata.as_deref();

        // ---- llamafile tinyBLAS: the reference's production GEMM path ----
        // ggml-cpu.c:1303-1325 tries `llamafile_sgemm` with `Btype = src1->type`
        // first (only reachable for F32 weights: every other case needs the
        // converted activations) and :1386-1409 retries with
        // `Btype = vec_dot_type` when `src1->type != vec_dot_type`, i.e. when the
        // port built `wdata` — plus F16/BF16, whose `vec_dot_type` is their own
        // type and which the port converts lazily. Both attempts are guarded by
        // `!params->use_ref` since #29806 (ggml-cpu.c:1308/:1387 — skip tinyBLAS
        // when the test harness wants the vec_dot reference path); the port's
        // analog is the `LLAMA_RUST_NO_TINYBLAS` switch below (`tb_off`).
        // Both calls see the *per-slice* m/n/k (`ne01`, `ne11`,
        // `ne00/blck`), so a 3D mul_mat with `ne11 == 1` never reaches
        // tinyBLAS; the C's `goto` fallback also makes the decision
        // plane-independent. See tinyblas.rs for the full predicate table.
        let sgemm_k = (ne00 as usize) / ty0.blck_size();
        // `ggml_is_contiguous(src1)` verbatim (ggml.c) — the first attempt is
        // skipped for strided views (e.g. the permuted q of build_attn_mha),
        // and (like attempt 2) when the reference-harness `use_ref` is on;
        // `LLAMA_RUST_NO_TINYBLAS` is the port's analog (ggml-cpu.c:1308).
        let src1_cont = t1.nb[1] as usize == n * 4
            && t1.nb[2] as usize == t1.nb[1] as usize * ne11 as usize
            && t1.nb[3] as usize == t1.nb[2] as usize * ne12 as usize;
        // Diagnostic switch (same convention as `LLAMA_RUST_REPACK`): force the
        // row-wise `vec_dot` fallback to compare the two routes on a real model.
        let tb_off = std::env::var_os("LLAMA_RUST_NO_TINYBLAS").is_some();
        // the extra-buffer intercept (ggml-cpu.c:1751-1753) precedes the
        // llamafile attempts — see the Q4_0 block above
        if let Some(rep) = &repacked_q4_0 {
            self.mul_mat_q4_0_repack(dst, s0, s1, rep, nth, team);
            return;
        }

        // ---- tiled K-quant matmul (tiled/tiled.cpp, sync batch D) ----
        // ggml-cpu.c:1269-1272: right after the hadamard check (and after the
        // repack intercepts, which in the C happen even earlier, in
        // ggml_compute_forward) `ggml_compute_forward_mul_mat_tiled` takes
        // the whole op when the type is supported and the batch >= 8 — its
        // numerics are its own (int32 subblock-exact + per-slab f32), so the
        // port must take it at the same routing point. A repacked Q4_K never
        // gets here (`src0->extra != NULL` refuses, tiled.cpp:612).
        if repacked_q4k.is_none() && crate::tiled::supported(ty0, /*repacked*/ false) {
            let took = self.tiled_mul_mat(dst, s0, s1, nth, team);
            if took {
                return;
            }
        }
        if !tb_off && src1_cont && tinyblas::accepts(ty0, t1.ty, nr0, ne11 as usize, sgemm_k) {
            let b = self.rd_rest(s1, 0);
            self.sgemm_planes(
                dst,
                s0,
                tinyblas::resolve(ty0, t1.ty).unwrap(),
                b,
                nb11 / t1.ty.type_size(),
                nb12,
                nb13,
                nth,
                team,
            );
            return;
        }
        if !tb_off && t1.ty != vdt && tinyblas::accepts(ty0, vdt, nr0, ne11 as usize, sgemm_k) {
            // attempt 2 (ggml-cpu.c:1387, `!params->use_ref && src1->type !=
            // vec_dot_type`)'s B rows are the converted activations: `wdata`
            // when the port already built it, else (F16/BF16 weights) a local
            // conversion of the `ne11*ne12*ne13` rows, exactly the C's
            // `from_float` pass.
            let convert = wd.is_none();
            let rs = row_size(vdt);
            let rows = (ne11 * ne12 * ne13) as usize;
            let conv: Option<Vec<u8>> = if convert {
                let mut buf = vec![0u8; rows * rs];
                for ir in 0..rows {
                    let i13 = ir / (ne11 * ne12) as usize;
                    let i12 = (ir - i13 * (ne11 * ne12) as usize) / ne11 as usize;
                    let i11 = ir - i13 * (ne11 * ne12) as usize - i12 * ne11 as usize;
                    let src_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                    let row: &[f32] = self.rd_f32s(s1, src_off, n);
                    let out = &mut buf[ir * rs..(ir + 1) * rs];
                    match vdt {
                        GgmlType::F16 => {
                            let o: &mut [f16] = bytemuck::cast_slice_mut(out);
                            for j in 0..n {
                                o[j] = f16::from_f32(row[j]);
                            }
                        }
                        GgmlType::Bf16 => {
                            let o: &mut [bf16] = bytemuck::cast_slice_mut(out);
                            for j in 0..n {
                                o[j] = bf16::from_f32(row[j]);
                            }
                        }
                        other => unimplemented!("tinyBLAS activation conversion to {other:?}"),
                    }
                }
                Some(buf)
            } else {
                None
            };
            let b: &[u8] = conv.as_deref().or(wd).unwrap();
            let b_row = rs; // `row_size(vec_dot_type, ne10)`, ggml-cpu.c:1385
            self.sgemm_planes(
                dst,
                s0,
                tinyblas::resolve(ty0, vdt).unwrap(),
                b,
                b_row / vdt.type_size(),
                ne11 as usize * b_row,
                ne12 as usize * ne11 as usize * b_row,
                nth,
                team,
            );
            return;
        }

        match ty0 {
            // ---- F32 weights: dot against src1 rows in place ----
            GgmlType::F32 => {
                let run = |ir0_0: usize, ir0_1: usize, ir1_0: usize, ir1_1: usize| {
                    for ir1 in ir1_0..ir1_1 {
                        let i13 = ir1 / (ne12 * ne1) as usize;
                        let i12 = (ir1 - i13 * (ne12 * ne1) as usize) / ne1 as usize;
                        let i11 = ir1 - i13 * (ne12 * ne1) as usize - i12 * ne1 as usize;
                        let i02 = i12 / r2 as usize;
                        let i03 = i13 / r3 as usize;
                        let src0_base = i02 * nb02 + i03 * nb03;
                        let src1_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                        let dst_off = i11 * nb1 + i12 * nb2 + i13 * nb3;
                        let yrow: &[f32] = self.rd_f32s(s1, src1_off, n);
                        for ir0 in ir0_0..ir0_1 {
                            let xrow: &[f32] = self.rd_f32s(s0, src0_base + ir0 * nb01, n);
                            let v = vec_dot::vec_dot_f32_c(n, xrow, yrow);
                            // dst_col[ir0] advances by nb0 (== 4) bytes
                            self.wr_f32s(dst, dst_off + ir0 * 4, 1)[0] = v;
                        }
                    }
                };
                self.par_mul_mat(nr0, nr1, 1, nth, team, run);
            }
            // ---- F16/BF16 weights: widen the weight row to f32, f64-accumulated
            //      dot (C converts src1 to f16 instead; we keep full f32
            //      activations). Broadcast over ne2 (GQA) like ggml-cpu.c:1186.
            //      Only reached when `llamafile_sgemm` bailed (n < 2 or
            //      m % 4 != 0 — K remainders are in-kernel since #29806) — the
            //      tinyBLAS attempt above handles the rest.
            GgmlType::F16 | GgmlType::Bf16 => {
                let run = |ir0_0: usize, ir0_1: usize, ir1_0: usize, ir1_1: usize| {
                    let mut yh = vec![half::f16::ZERO; n];
                    let mut ybh = vec![bf16::ZERO; n];
                    for ir1 in ir1_0..ir1_1 {
                        let i13 = ir1 / (ne12 * ne1) as usize;
                        let i12 = (ir1 - i13 * (ne12 * ne1) as usize) / ne1 as usize;
                        let i11 = ir1 - i13 * (ne12 * ne1) as usize - i12 * ne1 as usize;
                        let i02 = i12 / r2 as usize;
                        let i03 = i13 / r3 as usize;
                        let src0_base = i02 * nb02 + i03 * nb03;
                        let src1_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                        let dst_off = i11 * nb1 + i12 * nb2 + i13 * nb3;
                        let yrow: &[f32] = self.rd_f32s(s1, src1_off, n);
                        match ty0 {
                            GgmlType::F16 => {
                                // fp32→fp16 RN (same as C from_float)
                                for j in 0..n {
                                    yh[j] = half::f16::from_f32(yrow[j]);
                                }
                                for ir0 in ir0_0..ir0_1 {
                                    let xrow: &[f16] =
                                        bytemuck::cast_slice(self.rd(s0, src0_base + ir0 * nb01, n * 2));
                                    let v = vec_dot::vec_dot_f16_c(n, xrow, &yh);
                                    self.wr_f32s(dst, dst_off + ir0 * 4, 1)[0] = v;
                                }
                            }
                            _ => {
                                // vec_dot_type(BF16) = BF16: activations quantized
                                // to bf16 like C, then the bf16×bf16 lane dot
                                for j in 0..n {
                                    ybh[j] = bf16::from_f32(yrow[j]);
                                }
                                for ir0 in ir0_0..ir0_1 {
                                    let xrow: &[bf16] =
                                        bytemuck::cast_slice(self.rd(s0, src0_base + ir0 * nb01, n * 2));
                                    let v = vec_dot::vec_dot_bf16_c(n, xrow, &ybh);
                                    self.wr_f32s(dst, dst_off + ir0 * 4, 1)[0] = v;
                                }
                            }
                        }
                    }
                };
                self.par_mul_mat(nr0, nr1, 1, nth, team, run);
            }
            // ---- quantized weights: integer vec_dot against quantized src1 ----
            _ => {
                let rs = row_size(vdt);
                // ---- CPU_REPACK: Q4_K 8x8 (decided above) ----
                // The two kernels are *not* interchangeable: the gemm fma's once
                // per (super block, sub block pair) while the gemv fma's once per
                // super block, so both must be mirrored (pinned bit-exactly by
                // `repack::q4k_ref_tests::q4k_kernels_match_reference_avx_bit_exact`,
                // and on real GGUF tensors by `vec_dot.rs`'s
                // `kquant_real_tensor_tests`).
                if let Some(rep) = &repacked_q4k {
                    self.mul_mat_q4k_repack(dst, s0, s1, rep, nth, team);
                    return;
                }
                // ---- CPU_REPACK: MXFP4 8x8 (repack.rs) ----
                // C's forward_mul_mat uses gemm for whole groups of 4 activation
                // rows and gemv for the `nrows % 4` tail (repack.cpp:4638-4647).
                // Both accumulate identically per output element (one fma per
                // 32-element block, same scale product — asserted by repack.rs's
                // `gemm_matches_gemv_elementwise`), so the port walks activation
                // rows one at a time through the gemv entry point.
                let repacked = self.repacked_mxfp4(s0);
                let run = |ir0_0: usize, ir0_1: usize, ir1_0: usize, ir1_1: usize| {
                    // the C's one_chunk walks raw data pointers
                    // (`(char *) src0_cur + ir0*nb01`, ggml-cpu.c:1345); resolve
                    // the view chain / storage *once per chunk range* instead of
                    // per output element — `rd_rest`/`wr_f32s` re-derive the
                    // base pointer (a view-chain walk + storage downcast) on
                    // every call, which costs ~2% of the Q6_K row-wise gemm.
                    // Same bytes, same vec_dot calls — only address derivation
                    // is hoisted. `voff` is each view's byte offset into its
                    // storage (what `rd_rest`/`wr_f32s` add on every access).
                    let (s0_ptr, _, s0_voff, s0_rest) = self.loc(s0);
                    let (d_ptr, d_writable, d_voff, _) = self.loc(dst);
                    debug_assert!(d_writable);
                    for ir1 in ir1_0..ir1_1 {
                        let i13 = ir1 / (ne12 * ne1) as usize;
                        let i12 = (ir1 - i13 * (ne12 * ne1) as usize) / ne1 as usize;
                        let i11 = ir1 - i13 * (ne12 * ne1) as usize - i12 * ne1 as usize;
                        let i02 = i12 / r2 as usize;
                        let i03 = i13 / r3 as usize;
                        let src0_base = i02 * nb02 + i03 * nb03;
                        let dst_off = i11 * nb1 + i12 * nb2 + i13 * nb3;
                        let yrow: &[u8] = match wd {
                            Some(w) => &w[ir1 * rs..],
                            None => self.rd_rest(s1, i11 * nb11 + i12 * nb12 + i13 * nb13),
                        };
                        if let Some(rep) = &repacked {
                            let nb = n / crate::blocks::QK_MXFP4;
                            let rowg = nb * crate::repack::BLOCK_MXFP4X8_SIZE;
                            let mat = &rep[src0_base..];
                            // 8-row groups: `par_mul_mat` gets rowchunk = 8,
                            // so ir0_0 is always group-aligned here.
                            for row0 in (ir0_0..ir0_1).step_by(8) {
                                let s = self.wr_f32s(dst, dst_off + row0 * 4, 8);
                                crate::repack::gemv_mxfp4_8x8_q8_0(n, s, &mat[row0 / 8 * rowg..], yrow, 8);
                            }
                            continue;
                        }
                        for ir0 in ir0_0..ir0_1 {
                            debug_assert!(s0_voff + src0_base + ir0 * nb01 <= s0_rest);
                            // SAFETY: the row lives `voff + src0_base +
                            // ir0*nb01` bytes into s0's storage; the slice runs
                            // to the end of that storage exactly like `rd_rest`
                            // (rows of strided views may legally run past the
                            // view).
                            let xrow: &[u8] = unsafe {
                                std::slice::from_raw_parts(
                                    s0_ptr.add(s0_voff + src0_base + ir0 * nb01),
                                    s0_rest - s0_voff - src0_base - ir0 * nb01,
                                )
                            };
                            let v = match ty0 {
                                // MXFP4 has no vec_dot_row arm (it lives in
                                // ggml-cpu/quants.c, not the type-traits table
                                // dispatch that vec_dot.rs mirrors).
                                GgmlType::Mxfp4 => vec_dot_mxfp4_q8_0(n, xrow, yrow),
                                _ => vec_dot::vec_dot_row(ty0, n, xrow, yrow),
                            };
                            // SAFETY: one f32 at `voff + dst_off + ir0*4` inside
                            // dst's arena storage (the address `wr_f32s`
                            // returned — view offset included).
                            unsafe {
                                (d_ptr.add(d_voff + dst_off + ir0 * 4) as *mut f32)
                                    .write_unaligned(v);
                            }
                        }
                    }
                };
                // rowchunk 8 only matters for the 8-row-group gemv; the
                // row-wise vec_dot path accepts any split.
                let rowchunk = if repacked.is_some() { 8 } else { 1 };
                self.par_mul_mat(nr0, nr1, rowchunk, nth, team, run);
            }
        }
    }

    // ==================================================================
    // mul_mat_id — 对照 ggml-cpu.c:1470 ggml_compute_forward_mul_mat_id
    // ==================================================================

    /// C reference (ggml-cpu.c:1550-1740): for every expert matrix `cur_a` the
    /// (slot, token) pairs whose `ids` entry is `cur_a` are looked up
    /// (`matrix_row_counts` / `MMID_MATRIX_ROW`), then each dst column
    /// `dst(i1, i2)[ir0] = vec_dot(src0_cur + ir0*nb01, src1_row(i11, i12))`
    /// with `i11 = id % ne11`, `i12 = token`. The per-expert grouping only
    /// reorders work between threads; every output element is a single
    /// independent `vec_dot`, so iterating (slot, token) directly is
    /// numerically identical (and keeps the result thread-count invariant).
    ///
    /// src1 is quantized to vec_dot_type up front exactly like C's wdata pass
    /// (ggml-cpu.c:1611-1644); the row index in wdata is `i11 + i12*ne11`.
    /// `ggml_compute_forward_mul_mat_tiled` adapter (tiled.cpp:1209-1223):
    /// resolves the Worker's view plumbing into `tiled::MmArgs`. Returns
    /// false when the tiled driver declines (batch < 8, env off); the type
    /// gate is the caller's (`tiled::supported`, mirroring how the C checks
    /// `src0->extra` and the type switch inside `ggml_tiled_supported`).
    fn tiled_mul_mat(
        &self,
        dst: TensorId,
        s0: TensorId,
        s1: TensorId,
        nth: usize,
        team: &Team<'_>,
    ) -> bool {
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);
        let fmt = crate::tiled::Fmt::from_type(t0.ty).unwrap();
        let vdt = vec_dot::vec_dot_type(t0.ty).unwrap();
        assert_eq!(vdt, GgmlType::Q8K, "tiled: vec_dot_type must be Q8_K");
        let (s0_ptr, _, s0_voff, _) = self.loc(s0);
        let (s1_ptr, _, s1_voff, _) = self.loc(s1);
        let (d_ptr, d_writable, d_voff, _) = self.loc(dst);
        debug_assert!(d_writable);
        let prebuilt = if t1.ty == vdt {
            Some((
                unsafe { s1_ptr.add(s1_voff) } as *const u8,
                t1.nb[1] as usize,
            ))
        } else {
            None
        };
        let args = crate::tiled::MmArgs {
            fmt,
            src0: unsafe { s0_ptr.add(s0_voff) },
            src0_bs: t0.ty.type_size(),
            nb01: t0.nb[1] as usize,
            src0_nb2: t0.nb[2] as usize,
            src0_nb3: t0.nb[3] as usize,
            ne00: t0.ne[0] as usize,
            ne01: t0.ne[1] as usize,
            ne02: t0.ne[2] as usize,
            ne03: t0.ne[3] as usize,
            src1: unsafe { s1_ptr.add(s1_voff) },
            nb11: t1.nb[1] as usize,
            nb12: t1.nb[2] as usize,
            nb13: t1.nb[3] as usize,
            ne11: t1.ne[1] as usize,
            ne12: t1.ne[2] as usize,
            ne13: t1.ne[3] as usize,
            prebuilt_wd: prebuilt,
            dst: unsafe { d_ptr.add(d_voff) },
            dst_nb1: td.nb[1] as usize,
            dst_nb2: td.nb[2] as usize,
            dst_nb3: td.nb[3] as usize,
            nth,
        };
        args.compute(&|jobs, f| team.run(jobs, f))
    }

    fn forward_mul_mat_id(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap(); // as  [ne00, ne01, n_as]
        let s1 = self.t(dst).src[1].unwrap(); // b   [ne10, ne11, ne12]
        let ids = self.t(dst).src[2].unwrap(); // ids [n_ids, ne12]
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let ti = self.t(ids);
        let td = self.t(dst);

        let ty = t0.ty;
        let vdt = vec_dot::vec_dot_type(ty)
            .or_else(|| mxfp4_vec_dot_type(ty))
            .unwrap_or_else(|| panic!("mul_mat_id: no vec_dot for {ty:?}"));

        assert_eq!(t0.nb[0] as usize, ty.type_size(), "mul_mat_id: permuted src0");
        assert_eq!(t1.nb[0] as usize, t1.ty.type_size(), "mul_mat_id: permuted src1");
        assert_eq!(td.nb[0] as usize, 4, "mul_mat_id: dst must be contiguous F32");
        assert_eq!(ti.ty, GgmlType::I32, "mul_mat_id: ids must be I32");

        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let n_as = t0.ne[2] as usize;
        let ne11 = t1.ne[1] as usize;
        let ne12 = t1.ne[2] as usize;
        let n_ids = ti.ne[0] as usize;
        let (nb01, nb02) = (t0.nb[1] as usize, t0.nb[2] as usize);
        let (nb11, nb12) = (t1.nb[1] as usize, t1.nb[2] as usize);
        let (inb0, inb1) = (ti.nb[0] as usize, ti.nb[1] as usize);
        let (nb1, nb2) = (td.nb[1] as usize, td.nb[2] as usize);
        assert_eq!(ne00, t1.ne[0] as usize, "mul_mat_id: ne00 != ne10");
        assert_eq!(n_ids, td.ne[1] as usize, "mul_mat_id: dst ne1");
        assert_eq!(ne12, td.ne[2] as usize, "mul_mat_id: dst ne2");

        // ---- wdata: quantize every src1 row to vec_dot_type (if needed) ----
        let rs = (vdt.type_size() * ne00 / vdt.blck_size()) as usize;
        let wdata: Option<Vec<u8>> = if t1.ty != vdt {
            assert_eq!(t1.ty, GgmlType::F32, "mul_mat_id: src1 must be F32");
            let nrows = ne11 * ne12;
            let mut wd = vec![0u8; nrows * rs];
            {
                let wd_ptr = std::sync::atomic::AtomicPtr::new(wd.as_mut_ptr());
                self.par_rows(nrows, nth, team, |ir0, ir1| {
                    for ir in ir0..ir1 {
                        let i12 = ir / ne11;
                        let i11 = ir - i12 * ne11;
                        let src_off = i11 * nb11 + i12 * nb12;
                        let row: &[f32] = self.rd_f32s(s1, src_off, ne00);
                        let p = wd_ptr.load(std::sync::atomic::Ordering::Relaxed);
                        let out = unsafe { std::slice::from_raw_parts_mut(p.add(ir * rs), rs) };
                        match vdt {
                            GgmlType::Q8_0 => {
                                quants::quantize_row_q8_0(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::Q8_1 => {
                                quants::quantize_row_q8_1(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::Q8K => {
                                quants::quantize_row_q8_K(row, bytemuck::cast_slice_mut(out));
                            }
                            GgmlType::F32 => out.copy_from_slice(self.rd(s1, src_off, ne00 * 4)),
                            other => unimplemented!("mul_mat_id wdata conversion to {other:?}"),
                        }
                    }
                });
            }
            Some(wd)
        } else {
            None
        };
        let wd = wdata.as_deref();

        // ---- CPU_REPACK: MXFP4 8x8 outer-product gemv (repack.rs) ----
        // The reference loads MXFP4 experts into the CPU_REPACK buffer
        // (repack.cpp:5093 mxfp4_8x8_q8_0) and runs
        // ggml_gemv_mxfp4_8x8_q8_0 on them (repack.cpp:4905). That kernel
        // accumulates `acc = fma(int_dot_block, e8m0_half(e)*q8_d, acc)` once per
        // 32-element block — the accumulation order the reference binary uses and
        // the port's row-wise vec_dot does not, so the repacked copy is built
        // (once, cached) whenever the tensor qualifies. See repack.rs's header.
        let repacked = self.repacked_mxfp4(s0);

        // ---- CPU_REPACK: Q4_K 8x8 (repack.rs) ----
        // A 3D Q4_K expert tensor whose *per-expert* `ne[1] % 8 == 0` gets the
        // q4_K_8x8_q8_K trait in the reference too (repack.cpp:5006-5011 looks
        // at `cur->ne[1]` — for a [ne00, ne01, n_as] tensor that is ne01, the
        // per-expert row count), and its `forward_mul_mat_id` (repack.cpp:4783)
        // runs `ggml_gemv_q4_K_8x8_q8_K` — always `nr == 1`, one call per
        // 8-row group of the expert matrix per (slot, token) pair
        // (repack.cpp:4893-4908: `gemv(..., 1, src0_cur_end - src0_cur_start)`).
        // The wdata pass above already wrote exactly the plain `block_q8_K`
        // rows that kernel consumes (`from_float` for Q8_K is
        // arch/x86/quants.c:505 `quantize_row_q8_K`, a plain delegate of
        // `_ref`). Local models reaching this: LFM2-8B-A1B Q4_K_M's 56 expert
        // tensors ([2048|1792, 1792|2048, 32]) and the Qwen3.6-35B MoE files.
        let repacked_q4k = if ty == GgmlType::Q4K { self.repacked_q4_k(s0) } else { None };

        // ---- CPU_REPACK: Q4_0 8x8 (repack.rs) ----
        // A 3D Q4_0 expert tensor whose *per-expert* `ne[1] % 8 == 0` gets the
        // q4_0_8x8_q8_0 trait in the reference too (repack.cpp:4987-4993 looks
        // at `cur->ne[1]`), and its `forward_mul_mat_id` (repack.cpp:4783)
        // runs `ggml_gemv_q4_0_8x8_q8_0` — always `nr == 1`, one call per
        // 8-row group per (slot, token) pair (repack.cpp:4893-4908). The
        // wdata pass above already wrote exactly the plain `block_q8_0` rows
        // that kernel consumes. Local models reaching this: gemma-4 MoE
        // ([704|2816, 2816|1408, 128] Q4_0 experts).
        let repacked_q4_0 = if ty == GgmlType::Q4_0 {
            self.repacked_q4_0(s0, /*require_2d*/ false)
        } else {
            None
        };

        // ---- tiled per-expert path (tiled/tiled.cpp, sync batch D) ----
        // ggml-cpu.c:1678-1681: for each expert with routed rows,
        // `ggml_compute_forward_mul_mat_id_tiled` takes the whole expert when
        // the type is supported and the routed row count >= 8. The C builds
        // the (slot, token) row-mapping table with ith == 0
        // (ggml-cpu.c:1646-1657, `iid1` outer / slot `id` inner); the port
        // builds the same table and skips the mapped pairs in the stock loop.
        // A repacked expert never gets here (src0->extra, tiled.cpp:612).
        let mut consumed: Vec<bool> = vec![false; n_ids * ne12];
        if repacked.is_none()
            && repacked_q4k.is_none()
            && repacked_q4_0.is_none()
            && crate::tiled::supported(ty, /*repacked*/ false)
        {
            let mut by_expert: Vec<Vec<(i32, i32)>> = vec![Vec::new(); n_as];
            for iid1 in 0..ne12 {
                for id in 0..n_ids {
                    let i02 = self.rd_i32s(ids, id * inb0 + iid1 * inb1, 1)[0];
                    assert!((0..n_as as i32).contains(&i02), "mul_mat_id: id out of range");
                    by_expert[i02 as usize].push((id as i32, iid1 as i32));
                }
            }
            // flat (slot, token) pair table, per-expert contiguous (the C's
            // MMID_MATRIX_ROW rows)
            let mut flat: Vec<i32> = Vec::with_capacity(2 * n_ids * ne12);
            let mut offs = vec![0usize; n_as + 1];
            for e in 0..n_as {
                offs[e] = flat.len();
                for &(a, b) in &by_expert[e] {
                    flat.push(a);
                    flat.push(b);
                }
            }
            offs[n_as] = flat.len();

            let fmt = crate::tiled::Fmt::from_type(ty).unwrap();
            assert_eq!(vdt, GgmlType::Q8K, "tiled: vec_dot_type must be Q8_K");
            let (s0_ptr, _, s0_voff, _) = self.loc(s0);
            let (d_ptr, d_writable, d_voff, _) = self.loc(dst);
            debug_assert!(d_writable);
            let prebuilt = if t1.ty == vdt {
                let (s1_ptr, _, s1_voff, _) = self.loc(s1);
                (unsafe { s1_ptr.add(s1_voff) } as *const u8, nb11)
            } else {
                (wd.unwrap().as_ptr(), rs)
            };
            let (wd_ptr, nbw1) = prebuilt;
            for cur_a in 0..n_as {
                let cne1 = (offs[cur_a + 1] - offs[cur_a]) / 2;
                if cne1 == 0 {
                    continue;
                }
                let args = crate::tiled::MmidArgs {
                    fmt,
                    src0: (unsafe { s0_ptr.add(s0_voff) } as usize) + cur_a * nb02,
                    src0_bs: ty.type_size(),
                    nb01,
                    ne00,
                    ne01,
                    wd: wd_ptr as usize,
                    nbw1,
                    ne11,
                    dst: (unsafe { d_ptr.add(d_voff) } as usize),
                    dst_nb1: nb1,
                    dst_nb2: nb2,
                    expert_rows: flat[offs[cur_a]..offs[cur_a + 1]].as_ptr() as usize,
                    cne1,
                    nth,
                };
                let took = args.compute(&|jobs, f| team.run(jobs, f));
                if took {
                    for p in by_expert[cur_a].iter() {
                        consumed[p.1 as usize * n_ids + p.0 as usize] = true;
                    }
                }
            }
        }

        // ---- main loop over (slot, token) pairs ----
        let run = |ir0_0: usize, ir0_1: usize, ic0: usize, ic1: usize| {
            for ir in ic0..ic1 {
                // experts consumed by the tiled path above (ggml-cpu.c:1678's
                // `continue` after ggml_compute_forward_mul_mat_id_tiled)
                if consumed[ir] {
                    continue;
                }
                let i1 = ir % n_ids; // expert slot (dst dim1)
                let i2 = ir / n_ids; // token       (dst dim2)
                let id = self.rd_i32s(ids, i1 * inb0 + i2 * inb1, 1)[0];
                assert!((0..n_as as i32).contains(&id), "mul_mat_id: id {id} out of range");
                let id = id as usize;
                // src1 row selector: C uses the *row-mapping* slot index here
                // (`row_mapping.i1`, ggml-cpu.c:1510-1519 — `const int id =
                // row_mapping.i1` is the ids->ne[0] slot despite the comment),
                // which matters for the down projection where src1 is
                // [n_ff, n_expert_used, n_tokens] (ne11 = n_expert_used > 1).
                let i11 = i1 % ne11;
                let src0_cur = id * nb02;
                let dst_col = i1 * nb1 + i2 * nb2;
                let yrow: &[u8] = match wd {
                    Some(w) => &w[(i11 + i2 * ne11) * rs..],
                    None => self.rd_rest(s1, i11 * nb11 + i2 * nb12),
                };
                if let Some(rep) = &repacked {
                    // C: gemv(ne00, dst + i1*nb1 + i2*nb2 + src0_cur_start, ne01,
                    //         src0_cur + src0_cur_start*nb01, src1_col, 1, ncols)
                    // with src0_row chunks aligned to NB_COLS == 8 — walked here as
                    // one 8-weight-row group per call, in order.
                    let nb = ne00 / crate::blocks::QK_MXFP4;
                    let rowg = nb * crate::repack::BLOCK_MXFP4X8_SIZE;
                    let mat = &rep[src0_cur..];
                    // 8-row groups: par_mul_mat gets rowchunk = 8, so ir0_0
                    // is always group-aligned here.
                    for row0 in (ir0_0..ir0_1).step_by(8) {
                        let s = self.wr_f32s(dst, dst_col + row0 * 4, 8);
                        crate::repack::gemv_mxfp4_8x8_q8_0(
                            ne00,
                            s,
                            &mat[row0 / 8 * rowg..],
                            yrow,
                            8,
                        );
                    }
                    continue;
                }
                if let Some(rep) = &repacked_q4k {
                    // C: gemv(ne00, (dst + i1*nb1 + i2*nb2) + src0_cur_start,
                    //         ne01, src0_cur + src0_cur_start*nb01, src1_col,
                    //         1, src0_cur_end - src0_cur_start)
                    // with `src0_cur` the expert's base — walked here as one
                    // 8-weight-row group per call, in order (`par_mul_mat`'s
                    // rowchunk 8 keeps ir0_0 group-aligned).
                    let nb = ne00 / crate::blocks::QK_K;
                    let rowg = nb * crate::repack::BLOCK_Q4_KX8_SIZE;
                    let mat = &rep[id * nb02..];
                    for row0 in (ir0_0..ir0_1).step_by(8) {
                        let s = self.wr_f32s(dst, dst_col + row0 * 4, 8);
                        crate::repack::gemv_q4_K_8x8_q8_K(ne00, s, &mat[row0 / 8 * rowg..], yrow, 8);
                    }
                    continue;
                }
                if let Some(rep) = &repacked_q4_0 {
                    // C: gemv(ne00, (dst + i1*nb1 + i2*nb2) + src0_cur_start,
                    //         ne01, src0_cur + src0_cur_start*nb01, src1_col,
                    //         1, src0_cur_end - src0_cur_start) — the Q4_0
                    // twin of the Q4_K branch above.
                    let nb = ne00 / crate::blocks::QK4_0;
                    let rowg = nb * crate::repack::BLOCK_Q4_0X8_SIZE;
                    let mat = &rep[id * nb02..];
                    for row0 in (ir0_0..ir0_1).step_by(8) {
                        let s = self.wr_f32s(dst, dst_col + row0 * 4, 8);
                        crate::repack::gemv_q4_0_8x8_q8_0(ne00, s, &mat[row0 / 8 * rowg..], yrow, 8);
                    }
                    continue;
                }
                for row in ir0_0..ir0_1 {
                    let xrow: &[u8] = self.rd_rest(s0, src0_cur + row * nb01);
                    let v = match ty {
                        GgmlType::Mxfp4 => vec_dot_mxfp4_q8_0(ne00, xrow, yrow),
                        _ => vec_dot::vec_dot_row(ty, ne00, xrow, yrow),
                    };
                    self.wr_f32s(dst, dst_col + row * 4, 1)[0] = v;
                }
            }
        };
        // C's mul_mat_id chunks each expert's (rows x used slots) the same way
        // `mul_mat` does (ggml-cpu.c:1727-1758): a one-token step with 4 used
        // experts still gets a full `nth`-way split over the expert rows.
        let rowchunk = if repacked.is_some() || repacked_q4k.is_some() || repacked_q4_0.is_some() {
            8
        } else {
            1
        };
        self.par_mul_mat(ne01, n_ids * ne12, rowchunk, nth, team, run);
    }

    /// 对照 ggml-cpu/ops.cpp:ggml_compute_forward_add_id_f32 — per-expert bias:
    /// dst(i0, i1, i2) = a(i0, i1, i2) + b(i0, ids(i1, i2)). `ids` may be the
    /// strided argsort_top_k view, hence the nb-based reads (ops.cpp uses nb20/nb21).
    fn forward_add_id(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let ids = self.t(dst).src[2].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let ti = self.t(ids);
        let td = self.t(dst);

        assert_eq!(td.ty, GgmlType::F32, "add_id: dst must be F32");
        assert_eq!(t0.ty, GgmlType::F32, "add_id: src0 must be F32");
        assert_eq!(t1.ty, GgmlType::F32, "add_id: src1 must be F32");
        assert_eq!(ti.ty, GgmlType::I32, "add_id: src2 must be I32");
        // ops.cpp:721-722 — contiguous F32 rows on src0/src1 (the row-slice
        // add below assumes element addressing)
        debug_assert_eq!(t0.nb[0] as usize, 4, "add_id: src0 row not contiguous");
        debug_assert_eq!(t1.nb[0] as usize, 4, "add_id: src1 row not contiguous");

        let ne0 = t0.ne[0] as usize;
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let nb11 = t1.nb[1] as usize;
        let nb20 = ti.nb[0] as usize;
        let nb21 = ti.nb[1] as usize;
        let (nb1, nb2) = (td.nb[1] as usize, td.nb[2] as usize);

        let run = |ir0: usize, ir1: usize| {
            for i1 in ir0..ir1 {
                for i2 in 0..ne2 {
                    let i11 = self.rd_i32s(ids, i1 * nb20 + i2 * nb21, 1)[0];
                    assert!(
                        (0..t1.ne[1] as i32).contains(&i11),
                        "add_id: expert {i11} out of range"
                    );
                    let bias = i11 as usize * nb11;
                    // one `ggml_vec_add_f32` over the row (ops.cpp:747-752):
                    // hoisting the three row slices out of the element loop
                    // lets this vectorize exactly like the C's row call —
                    // per-lane `a[i] + b[i]` in the same order, so the values
                    // are bit-identical (the per-element `rd/wr` spelling this
                    // replaces paid a `loc()` storage walk 3x per float).
                    let a: &[f32] = self.rd_f32s(s0, i1 * nb01 + i2 * nb02, ne0);
                    let b: &[f32] = self.rd_f32s(s1, bias, ne0);
                    let d: &mut [f32] = self.wr_f32s(dst, i1 * nb1 + i2 * nb2, ne0);
                    for (d, (a, b)) in d.iter_mut().zip(a.iter().zip(b.iter())) {
                        *d = a + b;
                    }
                }
            }
        };
        self.par_rows(ne1, nth, team, run);
    }

    // ==================================================================
    // glu — 对照 ggml-cpu/ops.cpp:3319 ggml_compute_forward_swiglu_oai_f32
    // ==================================================================

    /// SWIGLU_OAI (ops.cpp:3321-3385): x = min(gate, limit); y = clamp(up, -limit,
    /// limit); dst = x / (1 + expf(alpha * -x)) * (y + 1). F32 only, contiguous
    /// rows (GGML_ASSERT(ggml_is_contiguous_1)) — the reference uses libm expf,
    /// not the vec.h polynomial.
    fn forward_glu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1];
        let op = self.t(dst).op_params[0];
        let t0 = self.t(s0);
        let td = self.t(dst);

        // ---- BF16 GLU (ops.cpp sync batch D: ggml_compute_forward_glu_bf16_
        // rows, :2969-3030, over ggml_vec_{reglu,geglu,swiglu}_bf16) ----
        // The split-GLU form (src1 = the other half) is the only one the port
        // models, same as the F32 arms below.
        if t0.ty == GgmlType::Bf16 {
            assert_eq!(t0.nb[0] as usize, 2, "glu bf16: src0 must be contiguous_1");
            assert_eq!(td.nb[0] as usize, 2, "glu bf16: dst must be contiguous_1");
            let nb01 = t0.nb[1] as usize;
            let nb1 = td.nb[1] as usize;
            let nr = t0.nrows() as usize;
            let elem = |v: half::bf16, g: half::bf16| match op {
                crate::ops::GGML_GLU_OP_REGLU => crate::ops::ggml_vec_reglu_bf16(v, g),
                crate::ops::GGML_GLU_OP_GEGLU => crate::ops::ggml_vec_geglu_bf16(v, g),
                crate::ops::GGML_GLU_OP_SWIGLU => crate::ops::ggml_vec_swiglu_bf16(v, g),
                other => unimplemented!("glu bf16 op {other}"),
            };
            assert!(s1.is_some(), "glu bf16: split form only (mirrors the F32 arms)");
            let s1 = s1.unwrap();
            let t1 = self.t(s1);
            assert_eq!(t1.ty, GgmlType::Bf16, "glu bf16: src1 must be BF16");
            assert_eq!(t1.nb[0] as usize, 2, "glu bf16: src1 must be contiguous_1");
            assert_eq!(t0.ne[0], t1.ne[0], "glu bf16: shape mismatch");
            let nc = t0.ne[0] as usize;
            let nb11 = t1.nb[1] as usize;
            let run = |ir0: usize, ir1: usize| {
                for i1 in ir0..ir1 {
                    let x: &[half::bf16] = bytemuck::cast_slice(self.rd(s0, i1 * nb01, nc * 2));
                    let g: &[half::bf16] = bytemuck::cast_slice(self.rd(s1, i1 * nb11, nc * 2));
                    let y: &mut [half::bf16] =
                        bytemuck::cast_slice_mut(self.wr(dst, i1 * nb1, nc * 2));
                    for k in 0..nc {
                        y[k] = elem(x[k], g[k]);
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        assert_eq!(t0.ty, GgmlType::F32, "glu: F32 only");
        assert_eq!(t0.nb[0] as usize, 4, "glu: src0 must be contiguous_1");
        assert_eq!(td.nb[0] as usize, 4, "glu: dst must be contiguous_1");
        let nb01 = t0.nb[1] as usize;
        let nb1 = td.nb[1] as usize;
        let nr = t0.nrows() as usize;

        // ggml_swiglu_split with both halves passed (ggml.c:3061 →
        // ggml_compute_forward_swiglu_f32, ops.cpp:3178): nc = src0->ne[0]
        // (src1 present), y = silu(src0) * src1. Used by the granite mamba2
        // mixer (mamba-base.cpp:288).
        if op == crate::ops::GGML_GLU_OP_SWIGLU {
            assert!(s1.is_some(), "glu: swiglu requires src1 (the split halves)");
            let s1 = s1.unwrap();
            let t1 = self.t(s1);
            assert_eq!(t1.ty, GgmlType::F32, "swiglu: src1 must be F32");
            assert_eq!(t1.nb[0] as usize, 4, "swiglu: src1 must be contiguous_1");
            assert_eq!(t0.ne[0], t1.ne[0], "swiglu: shape mismatch");
            let nc = t0.ne[0] as usize;
            let nb11 = t1.nb[1] as usize;
            let run = |ir0: usize, ir1: usize| {
                let mut xs = vec![0f32; nc];
                let mut gs = vec![0f32; nc];
                let mut ys = vec![0f32; nc];
                for i1 in ir0..ir1 {
                    for k in 0..nc {
                        xs[k] = self.rd_f32s(s0, i1 * nb01 + k * 4, 1)[0];
                        gs[k] = self.rd_f32s(s1, i1 * nb11 + k * 4, 1)[0];
                    }
                    // ggml_vec_swiglu_f32 (vec.cpp:417): 16-wide ggml_v_silu
                    // chunks + the scalar libm-expf tail — n_ff_exp rows are
                    // often NOT a multiple of 16 (batch-15's 24), and the tail
                    // form is bit-visible (1 ulp)
                    crate::simd_x86::vec_swiglu_f32(&mut ys, &xs, &gs);
                    for k in 0..nc {
                        self.wr_f32s(dst, i1 * nb1 + k * 4, 1)[0] = ys[k];
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        // ggml_geglu_split with both halves passed (ggml.c:3043 →
        // ggml_compute_forward_geglu_f32, ops.cpp:3035-3087): y = gelu(src0) *
        // src1 with the f16-table gelu (ggml_vec_geglu_f32, vec.h:1415). gemma4
        // MoE experts (gemma4.cpp:190 LLM_FFN_GELU + gate_up).
        if op == crate::ops::GGML_GLU_OP_GEGLU {
            assert!(s1.is_some(), "glu: geglu requires src1 (the split halves)");
            let s1 = s1.unwrap();
            let t1 = self.t(s1);
            assert_eq!(t1.ty, GgmlType::F32, "geglu: src1 must be F32");
            assert_eq!(t1.nb[0] as usize, 4, "geglu: src1 must be contiguous_1");
            assert_eq!(t0.ne[0], t1.ne[0], "geglu: shape mismatch");
            let nc = t0.ne[0] as usize;
            let nb11 = t1.nb[1] as usize;
            let run = |ir0: usize, ir1: usize| {
                for i1 in ir0..ir1 {
                    for k in 0..nc {
                        let x = self.rd_f32s(s0, i1 * nb01 + k * 4, 1)[0];
                        let g = self.rd_f32s(s1, i1 * nb11 + k * 4, 1)[0];
                        self.wr_f32s(dst, i1 * nb1 + k * 4, 1)[0] = ggml_vec_gelu_f32(x) * g;
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        // ggml_reglu_split with both halves passed (ggml.c:3019 →
        // ggml_compute_forward_reglu_f32, ops.cpp:2892-2947): y =
        // (x > 0) ? x * g : 0 (vec.h:1401 ggml_vec_reglu_f32, a scalar loop).
        // smallthinker's MoE experts (smallthinker.cpp:156 LLM_FFN_RELU).
        if op == crate::ops::GGML_GLU_OP_REGLU {
            assert!(s1.is_some(), "glu: reglu requires src1 (the split halves)");
            let s1 = s1.unwrap();
            let t1 = self.t(s1);
            assert_eq!(t1.ty, GgmlType::F32, "reglu: src1 must be F32");
            assert_eq!(t1.nb[0] as usize, 4, "reglu: src1 must be contiguous_1");
            assert_eq!(t0.ne[0], t1.ne[0], "reglu: shape mismatch");
            let nc = t0.ne[0] as usize;
            let nb11 = t1.nb[1] as usize;
            let run = |ir0: usize, ir1: usize| {
                for i1 in ir0..ir1 {
                    for k in 0..nc {
                        let x = self.rd_f32s(s0, i1 * nb01 + k * 4, 1)[0];
                        let g = self.rd_f32s(s1, i1 * nb11 + k * 4, 1)[0];
                        self.wr_f32s(dst, i1 * nb1 + k * 4, 1)[0] =
                            if x > 0.0 { x * g } else { 0.0 };
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        // ggml_swiglu_clamp (ggml.c:3123 → ggml_compute_forward_swiglu_clamp_f32,
        // ops.cpp:3408-3465): y = min(gate, limit)/(1 + expf(-gate)) *
        // clamp(up, -limit, limit). Both halves are passed (the split-GLU form,
        // swapped = 0), deepseek4's expert/shared FFNs (llama-graph.cpp:2229/:1835).
        if op == crate::ops::GGML_GLU_OP_SWIGLU_CLAMP {
            assert!(s1.is_some(), "glu: swiglu_clamp requires src1 (the split halves)");
            let s1 = s1.unwrap();
            let t1 = self.t(s1);
            assert_eq!(t1.ty, GgmlType::F32, "swiglu_clamp: src1 must be F32");
            assert_eq!(t1.nb[0] as usize, 4, "swiglu_clamp: src1 must be contiguous_1");
            assert_eq!(t0.ne[0], t1.ne[0], "swiglu_clamp: shape mismatch");
            let nc = t0.ne[0] as usize;
            let nb11 = t1.nb[1] as usize;
            let limit = f32::from_bits(td.op_params[3] as u32);
            let run = |ir0: usize, ir1: usize| {
                for i1 in ir0..ir1 {
                    for k in 0..nc {
                        let x = self.rd_f32s(s0, i1 * nb01 + k * 4, 1)[0];
                        let u = self.rd_f32s(s1, i1 * nb11 + k * 4, 1)[0];
                        // ops.cpp:3445-3448 — std::min(gate, limit) and
                        // std::clamp(up, -limit, limit), sigmoid of the clamped gate
                        // (scalar `expf` in C — Rust's f32::exp is the same libm)
                        let gate = x.min(limit);
                        let up = u.clamp(-limit, limit);
                        self.wr_f32s(dst, i1 * nb1 + k * 4, 1)[0] =
                            gate / (1.0 + (-gate).exp()) * up;
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        assert_eq!(op, crate::ops::GGML_GLU_OP_SWIGLU_OAI, "glu: unsupported op {op}");
        assert!(s1.is_some(), "glu: swiglu_oai requires src1 (non-split GLU)");
        let s1 = s1.unwrap();
        let t1 = self.t(s1);
        assert_eq!(t1.ty, GgmlType::F32, "swiglu_oai: src1 must be F32");
        assert_eq!(t1.nb[0] as usize, 4, "swiglu_oai: src1 must be contiguous_1");

        let swapped = self.t(dst).op_params[1];
        let alpha = f32::from_bits(self.t(dst).op_params[2] as u32);
        let limit = f32::from_bits(self.t(dst).op_params[3] as u32);

        let nc = t0.ne[0] as usize;
        let nb11 = t1.nb[1] as usize;
        // C applies the swapped offsets only when src1 is NULL (the split-GLU
        // case): `if (!src1) { src0_p += swapped ? nc : 0; src1_p += swapped ? 0 : nc; }`
        // (ops.cpp:3355-3358). gpt-oss always passes both gate and up, so the
        // operands stay row-aligned; `swapped` is accepted and unused.
        let _ = swapped;
        let run = |ir0: usize, ir1: usize| {
            for i1 in ir0..ir1 {
                // hoist the row slices (ops.cpp:3365-3367 keeps one pointer
                // triple per row); the element loop below keeps the C's exact
                // expression order, so the values are bit-identical
                let a: &[f32] = self.rd_f32s(s0, i1 * nb01, nc);
                let b: &[f32] = self.rd_f32s(s1, i1 * nb11, nc);
                let d: &mut [f32] = self.wr_f32s(dst, i1 * nb1, nc);
                for k in 0..nc {
                    let x = a[k].min(limit);
                    let yv = b[k].clamp(-limit, limit);
                    d[k] = x / (1.0 + (alpha * (-x)).exp()) * (yv + 1.0);
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // norm / rms_norm — 对照 ops.cpp:3827 / ops.cpp:3924
    // ==================================================================

    fn forward_norm(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let eps = f32::from_bits(self.t(dst).op_params[0] as u32);

        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let ne02 = t0.ne[2] as usize;
        let ne03 = t0.ne[3] as usize;
        let nb00 = t0.nb[0] as usize;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let nb03 = t0.nb[3] as usize;
        let nb0 = td.nb[0] as usize;
        let nb1 = td.nb[1] as usize;
        let nb2 = td.nb[2] as usize;
        let nb3 = td.nb[3] as usize;

        let nr = ne01 * ne02 * ne03;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * ne01);
                let i02 = (ir - i03 * ne02 * ne01) / ne01;
                let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let dst_off = i01 * nb1 + i02 * nb2 + i03 * nb3;

                // ggml_vec_sum_f32: f64 accumulation, f32 result
                let mut sum = 0f64;
                for i00 in 0..ne00 {
                    sum += self.rd_f32s(s0, src_off + i00 * nb00, 1)[0] as f64;
                }
                let sum = sum as f32;
                let mean = sum / ne00 as f32;

                // ggml_vec_cvar_f32 (vec.cpp:455, `__AVX512F__ && __AVX512DQ__`
                // branch): center x-mean into dst, then per 16-value chunk
                // `sum += (ggml_float)_mm512_reduce_add_ps(val*val)` — the AVX512
                // reduce tree on the f32 squares accumulated into a double — and
                // the `n % 16` leftovers one value at a time. The f64-exact
                // variance this used to compute differs by 1 ulp, which the Q8_0
                // activation quantizer amplifies (PARITY.md, bert/t5 section).
                let mut vsum = 0f64;
                let mut i00 = 0;
                while i00 + 16 <= ne00 {
                    let mut sq = [0f32; 16];
                    for l in 0..16 {
                        let v = self.rd_f32s(s0, src_off + (i00 + l) * nb00, 1)[0] - mean;
                        self.wr_f32s(dst, dst_off + (i00 + l) * nb0, 1)[0] = v;
                        sq[l] = v * v;
                    }
                    vsum += vec_dot::reduce_add16(&sq) as f64;
                    i00 += 16;
                }
                while i00 < ne00 {
                    let v = self.rd_f32s(s0, src_off + i00 * nb00, 1)[0] - mean;
                    self.wr_f32s(dst, dst_off + i00 * nb0, 1)[0] = v;
                    vsum += (v * v) as f64;
                    i00 += 1;
                }
                let variance = (vsum / ne00 as f64) as f32;
                let scale = 1.0f32 / (variance + eps).sqrt();
                let y: &mut [f32] = self.wr_f32s(dst, dst_off, ne00);
                crate::simd_x86::vec_scale_f32(y, scale); // ggml_vec_scale_f32
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// ggml_compute_forward_l2_norm_f32 (ggml-cpu/ops.cpp:4486) — the row
    /// L2 norm of clef's decision head: `sum = Σ xi²` in f64, then
    /// `dst = xi / fmaxf(sqrtf(sum), eps)`. F32 rows only, like the C.
    fn forward_l2_norm(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let eps = f32::from_bits(self.t(dst).op_params[0] as u32);

        assert_eq!(t0.nb[0] as usize, 4, "l2_norm: src0 must be F32 contiguous");

        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let ne02 = t0.ne[2] as usize;
        let ne03 = t0.ne[3] as usize;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let nb03 = t0.nb[3] as usize;
        let nb1 = td.nb[1] as usize;
        let nb2 = td.nb[2] as usize;
        let nb3 = td.nb[3] as usize;

        let nr = ne01 * ne02 * ne03;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * ne01);
                let i02 = (ir - i03 * ne02 * ne01) / ne01;
                let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let dst_off = i01 * nb1 + i02 * nb2 + i03 * nb3;

                // the C accumulates the squares one value at a time into a
                // ggml_float (f64); sqrtf takes the f32-cast sum
                let x: &[f32] = self.rd_f32s(s0, src_off, ne00);
                let mut sum = 0f64;
                for &v in x {
                    sum += (v * v) as f64;
                }
                let scale = 1.0f32 / (sum as f32).sqrt().max(eps);

                let y: &mut [f32] = self.wr_f32s(dst, dst_off, ne00);
                for (yv, &v) in y.iter_mut().zip(x.iter()) {
                    *yv = v * scale;
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    fn forward_rms_norm(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let eps = f32::from_bits(self.t(dst).op_params[0] as u32);

        assert_eq!(t0.nb[0] as usize, 4, "rms_norm: src0 must be F32 contiguous");

        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let ne02 = t0.ne[2] as usize;
        let ne03 = t0.ne[3] as usize;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let nb03 = t0.nb[3] as usize;
        let nb1 = td.nb[1] as usize;
        let nb2 = td.nb[2] as usize;
        let nb3 = td.nb[3] as usize;

        let nr = ne01 * ne02 * ne03;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * ne01);
                let i02 = (ir - i03 * ne02 * ne01) / ne01;
                let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let dst_off = i01 * nb1 + i02 * nb2 + i03 * nb3;

                let x: &[f32] = self.rd_f32s(s0, src_off, ne00);
                let mut sum = 0f64;
                for &v in x {
                    sum += (v * v) as f64;
                }
                let mean = (sum / ne00 as f64) as f32;
                let scale = 1.0f32 / (mean + eps).sqrt();

                let y: &mut [f32] = self.wr_f32s(dst, dst_off, ne00);
                // memcpy + ggml_vec_scale_f32 (vec.h:703) — one rounded
                // multiply per lane, bit-identical to `x[i] * scale`.
                // In-place graphs (dst aliases src0) make the C's memcpy
                // self-copy — a semantic no-op the port skips instead of
                // tripping copy_from_slice's overlap precondition (found by
                // the debug-build UB precheck, sync batch D)
                if !std::ptr::eq(x.as_ptr(), y.as_ptr()) {
                    y.copy_from_slice(x);
                }
                crate::simd_x86::vec_scale_f32(y, scale);
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // scale — 对照 ops.cpp:4697 ggml_compute_forward_scale_f32
    // ==================================================================

    fn forward_scale(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        #[inline]
        fn nc_dbg(t: &TensorMeta) -> usize {
            t.ne[0] as usize
        }

        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let s = f32::from_bits(self.t(dst).op_params[0] as u32);
        let b = f32::from_bits(self.t(dst).op_params[1] as u32);

        // ggml_compute_forward_scale_bf16 (ops.cpp sync batch D, :4932):
        // y = bf16(s * bf16_to_f32(x) + b) — contiguous src0/dst, plain
        // mul+add off the op_params pair (fused by the reference build's
        // -ffp-contract=fast, hence mul_add here)
        if t0.ty == GgmlType::Bf16 {
            // ggml_is_contiguous (ops.cpp:4937-4939 asserts)
            debug_assert!(
                (t0.nb[0] as usize, t0.nb[1] as usize) == (2, nc_dbg(t0) * 2)
                    && td.nb[0] as usize == 2
                    && td.ne == t0.ne,
                "scale_bf16: src0/dst must be contiguous"
            );
            let nc = t0.ne[0] as usize;
            let nr = t0.nrows() as usize;
            let nb01 = t0.nb[1] as usize;
            let nb1 = td.nb[1] as usize;
            let run = |ir0: usize, ir1: usize| {
                for i1 in ir0..ir1 {
                    let x: &[half::bf16] = bytemuck::cast_slice(self.rd(s0, i1 * nb01, nc * 2));
                    let y: &mut [half::bf16] =
                        bytemuck::cast_slice_mut(self.wr(dst, i1 * nb1, nc * 2));
                    for k in 0..nc {
                        y[k] = half::bf16::from_f32(s.mul_add(x[k].to_f32(), b));
                    }
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let nb01 = t0.nb[1] as usize;
        let nb1 = td.nb[1] as usize;

        let same_loc = self.loc(s0) == self.loc(dst);
        let run = |ir0: usize, ir1: usize| {
            for i1 in ir0..ir1 {
                let dst_off = i1 * nb1;
                if b == 0.0 {
                    if !same_loc {
                        let src = self.rd(s0, i1 * nb01, nc * 4).to_vec();
                        self.wr(dst, dst_off, nc * 4).copy_from_slice(&src);
                    }
                    let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                    for v in y.iter_mut() {
                        *v *= s;
                    }
                } else {
                    // ggml_vec_mad1_f32: y = x*s + b, one fused multiply-add
                    // (vec.h:679 GGML_F32_VEC_FMA(vb, ay, vs) on x86)
                    for j in 0..nc {
                        let x = self.rd_f32s(s0, i1 * nb01 + j * 4, 1)[0];
                        self.wr_f32s(dst, dst_off + j * 4, 1)[0] = x.mul_add(s, b);
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // soft_max — 对照 ops.cpp:5584 ggml_compute_forward_soft_max_f32
    // ==================================================================

    fn forward_soft_max(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let mask = self.t(dst).src[1];
        // attention sinks (ggml_soft_max_add_sinks): one F32 logit per head
        // (src[0]->ne[2]); ops.cpp:5625-5680 folds them into the row max and
        // into `sum` as `expf(sk - max)` — the extra sink column of the softmax.
        let sinks = self.t(dst).src[2];
        let t0 = self.t(s0);
        let td = self.t(dst);

        let scale = f32::from_bits(self.t(dst).op_params[0] as u32);
        let max_bias = f32::from_bits(self.t(dst).op_params[1] as u32);

        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let ne02 = t0.ne[2] as usize;
        let ne03 = t0.ne[3] as usize;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let nb03 = t0.nb[3] as usize;
        let nb1 = td.nb[1] as usize;
        let nb2 = td.nb[2] as usize;
        let nb3 = td.nb[3] as usize;

        let (mne12, mne13, mnb11, mnb12, mnb13, mty) = match mask {
            Some(m) => {
                let tm = self.t(m);
                (tm.ne[2] as usize, tm.ne[3] as usize, tm.nb[1] as usize, tm.nb[2] as usize, tm.nb[3] as usize, tm.ty)
            }
            None => (1, 1, 0, 0, 0, GgmlType::F32),
        };

        // ALiBi slopes (ops.cpp:5615-5619)
        let n_head = ne02 as u32;
        let n_head_log2 = 1u32 << (n_head as f64).log2().floor() as u32;
        let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);

        let nr = ne01 * ne02 * ne03;
        let run = |ir0: usize, ir1: usize| {
            let mut wp = vec![0f32; ne00];
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * ne01);
                let i02 = (ir - i03 * ne02 * ne01) / ne01;
                let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                let i12 = i02 % mne12;
                let i13 = i03 % mne13;

                let h = i02 as u32;
                let slope = if max_bias > 0.0 {
                    if h < n_head_log2 {
                        m0.powi(h as i32 + 1)
                    } else {
                        m1.powi((2 * (h - n_head_log2) + 1) as i32)
                    }
                } else {
                    1.0
                };

                let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                let dst_off = i01 * nb1 + i02 * nb2 + i03 * nb3;

                let sp: &[f32] = self.rd_f32s(s0, src_off, ne00);
                wp[..ne00].copy_from_slice(sp);
                for v in wp[..ne00].iter_mut() {
                    *v *= scale;
                }
                if let Some(m) = mask {
                    let m_off = i01 * mnb11 + i12 * mnb12 + i13 * mnb13;
                    match mty {
                        GgmlType::F16 => {
                            let mp: &[f16] = bytemuck::cast_slice(self.rd(m, m_off, ne00 * 2));
                            for i in 0..ne00 {
                                wp[i] += slope * mp[i].to_f32();
                            }
                        }
                        GgmlType::F32 => {
                            let mp: &[f32] = self.rd_f32s(m, m_off, ne00);
                            for i in 0..ne00 {
                                wp[i] += slope * mp[i];
                            }
                        }
                        other => unimplemented!("soft_max mask type {other:?}"),
                    }
                }

                // max (ggml_vec_max_f32)
                let mut max = f32::NEG_INFINITY;
                for &v in &wp[..ne00] {
                    max = if v > max { v } else { max };
                }
                // ops.cpp:5670-5673 `if (sk) max = MAX(max, sk[i02]);`
                let sk = sinks.map(|sk| self.rd_f32s(sk, i02 * 4, 1)[0]);
                if let Some(sk) = sk {
                    if sk > max {
                        max = sk;
                    }
                }

                // ggml_vec_soft_max_f32 (AVX512F+DQ, vec.cpp:531): per-16-lane
                // chunk, exp via the v512 polynomial, `sum += reduce_add(chunk)`
                // in f64, scalar libm tail — then scale by 1/sum. The vector
                // kernel is bit-identical to the elementwise chunk loop it
                // replaces (asserted in simd_x86's tests).
                let dp: &mut [f32] = self.wr_f32s(dst, dst_off, ne00);
                let mut sum = crate::simd_x86::vec_soft_max_f32(dp, &wp[..ne00], max);
                // ops.cpp:5678-5680 `if (sk) sum += (ggml_float) expf(sk[i02] - max);`
                if let Some(sk) = sk {
                    sum += (sk - max).exp() as f64;
                }
                let inv = (1.0 / sum) as f32;
                // ops.cpp:5682: ggml_vec_scale_f32(ne00, dp, inv)
                crate::simd_x86::vec_scale_f32(dp, inv);
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // tanh / silu / gelu — 对照 unary-ops.cpp:253 (tanh) / ops.cpp:2579
    // (silu) / ops.cpp:2206 (gelu) + their vec.h scalar tails
    // ==================================================================

    /// 对照 ggml_compute_forward_tanh (unary-ops.cpp:253 → op_tanh at line 19).
    /// The reference has no SIMD tanh at this commit: `unary_op<op_tanh>` calls
    /// `tanhf` per element (vec.h:909 ggml_vec_tanh_f32 is a plain scalar loop
    /// too). Rust's f32::tanh resolves to the same glibc tanhf — verified
    /// bit-identical over 8M inputs and against parity/tanh_ref.bin.
    fn forward_tanh(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        for i in 0..nc {
                            y[i] = ggml_tanh_f32(x[i]);
                        }
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            // unary_op uses the type-conversion table: the dst
                            // conversion is GGML_CPU_FP32_TO_FP16 == the scalar
                            // ggml_compute_fp32_to_fp16 in this build, which
                            // canonicalises NaN (0x7E00) where half::from_f32
                            // keeps the payload.
                            y[i] = f16::from_bits(ggml_compute_fp32_to_fp16(ggml_tanh_f32(
                                x[i].to_f32(),
                            )));
                        }
                    }
                    other => unimplemented!("tanh for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// unary-ops.cpp:266 `unary_op<op_sigmoid>` with vec.h:936
    /// `ggml_vec_sigmoid_f32` — a plain libm loop: `1/(1 + expf(-x))`.
    fn forward_sigmoid(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        for i in 0..nc {
                            y[i] = 1.0 / (1.0 + (-x[i]).exp());
                        }
                    }
                    other => unimplemented!("sigmoid type {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// unary-ops.cpp:273 `unary_op<op_exp>` with vec.h:956
    /// `ggml_vec_exp_f32` — a plain libm loop: `expf(x)`. Like tanh/sigmoid the
    /// reference has no SIMD exp at this commit (no exp_ps in simd-mappings.h),
    /// so Rust's `f32::exp` (the same glibc expf the .so calls) is
    /// bit-identical — pinned over the full length sweep + edge cases against
    /// parity/exp_ref.bin.
    fn forward_exp(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        for i in 0..nc {
                            y[i] = x[i].exp();
                        }
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            // unary_op's type-conversion table — the dst
                            // conversion is the scalar ggml_compute_fp32_to_fp16
                            // (NaN canonicalisation), see the tanh F16 note
                            y[i] = f16::from_bits(ggml_compute_fp32_to_fp16(x[i].to_f32().exp()));
                        }
                    }
                    other => unimplemented!("exp for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// unary-ops.cpp:317 `unary_op<op_round>` — per-element `roundf`
    /// (round-half-away-from-zero; Rust `f32::round` is the same IEEE
    /// roundTiesAway). graniteswitch's router lane only (granite-switch.cpp:285).
    fn forward_round(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "round: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "round: F32 only");
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                for i in 0..nc {
                    y[i] = x[i].round();
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_xielu` (unary-ops.cpp:325-336): the folded
    /// constants come from op_params[1..5] (f32 bits), the element functor is
    /// `op_xielu` (:55-62) — `x > 0 ? alpha_p*x*x + beta*x :
    /// (expm1f(min(x,eps)) - x)*alpha_n + beta*x`, no SIMD.
    fn forward_xielu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let p = td.op_params;
        let alpha_n = f32::from_bits(p[1] as u32);
        let alpha_p = f32::from_bits(p[2] as u32);
        let beta = f32::from_bits(p[3] as u32);
        let eps = f32::from_bits(p[4] as u32);
        assert_eq!(t0.ty, GgmlType::F32, "xielu: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "xielu: F32 only");
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                for i in 0..nc {
                    let v = x[i];
                    // C: `alpha_p * x * x + beta * x` / `(expm1f(min_x_eps) -
                    // x) * alpha_n + beta * x` under -ffp-contract=fast — GCC
                    // contracts the left product chain into the fma's (a, b)
                    // and pre-rounds the right product: fma(alpha_p*x, x,
                    // beta*x); the negative branch factors as
                    // fma(expm1-x, alpha_n, beta*x) — multiplication operands
                    // commute inside the fma, so the ordering below is exact
                    let beta_v = beta * v;
                    y[i] = if v > 0.0 {
                        (alpha_p * v).mul_add(v, beta_v)
                    } else {
                        (v.min(eps).exp_m1() - v).mul_add(alpha_n, beta_v)
                    };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    fn forward_silu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        // ggml_vec_silu_f32 (vec.cpp:380): 16-wide ggml_v_silu
                        // chunks (AVX512F+DQ) + the scalar ggml_silu_f32 tail —
                        // the vector kernel is bit-identical to the per-element
                        // loop it replaces (same polynomial, per-lane div).
                        crate::simd_x86::vec_silu_f32(y, x);
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        // ggml_vec_silu_f16 (vec.h:1050) is a scalar loop with
                        // libm expf — never the v512 polynomial
                        for i in 0..nc {
                            y[i] = f16::from_f32(ggml_silu_scalar_f32(x[i].to_f32()));
                        }
                    }
                    // ggml_compute_forward_silu_bf16 (ops.cpp sync batch D) →
                    // ggml_vec_silu_bf16 (vec.h:1397) over
                    // ggml_compute_forward_unary_bf16_rows
                    GgmlType::Bf16 => {
                        let x: &[half::bf16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [half::bf16] =
                            bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            y[i] = crate::ops::ggml_vec_silu_bf16(x[i]);
                        }
                    }
                    other => unimplemented!("silu for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    fn forward_gelu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        for i in 0..nc {
                            y[i] = ggml_vec_gelu_f32(x[i]);
                        }
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            y[i] = ggml_vec_gelu_f16(x[i]);
                        }
                    }
                    // ggml_compute_forward_gelu_bf16 (ops.cpp sync batch D) →
                    // ggml_vec_gelu_bf16 (vec.h:979): the polynomial gelu,
                    // NOT the f16 table the F32 path takes
                    GgmlType::Bf16 => {
                        let x: &[half::bf16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [half::bf16] =
                            bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            y[i] = crate::ops::ggml_vec_gelu_bf16(x[i]);
                        }
                    }
                    other => unimplemented!("gelu for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// 对照 ggml_compute_forward_gelu_erf (ops.cpp:2436-2453): F32 via
    /// `ggml_vec_gelu_erf_f32` (ops.cpp:2341-2379, vec.h:1010's plain scalar
    /// erff loop), F16 via `ggml_vec_gelu_erf_f16` (ops.cpp:2388-2426,
    /// vec.h:979), anything else aborts in C. Row split identical to the
    /// gelu/softplus bodies (`dr = ceil(nr/nth)` rows per thread).
    fn forward_gelu_erf(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                match t0.ty {
                    GgmlType::F32 => {
                        let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                        let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                        for i in 0..nc {
                            y[i] = ggml_vec_gelu_erf_f32(x[i]);
                        }
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            y[i] = ggml_vec_gelu_erf_f16(x[i]);
                        }
                    }
                    // ggml_compute_forward_gelu_erf_bf16 (ops.cpp sync batch
                    // D) → ggml_vec_gelu_erf_bf16 (vec.h:993)
                    GgmlType::Bf16 => {
                        let x: &[half::bf16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [half::bf16] =
                            bytemuck::cast_slice_mut(self.wr(dst, dst_off, nc * 2));
                        for i in 0..nc {
                            y[i] = crate::ops::ggml_vec_gelu_erf_bf16(x[i]);
                        }
                    }
                    other => unimplemented!("gelu_erf for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// ops.cpp:3783 `ggml_compute_forward_softplus_f32` →
    /// `ggml_compute_softplus_f32` (ggml-impl.h:107):
    /// `(x > 20.0f) ? x : logf(1 + expf(x))`. qwen35's gated delta net applies
    /// it to the decay bias (qwen35.cpp:368).
    fn forward_softplus(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "softplus: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "softplus: F32 only");
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                for i in 0..nc {
                    y[i] = crate::ssm::softplus_f32(x[i]);
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// 对照 ggml_compute_forward_gated_delta_net (ops.cpp:11090) → gdn.rs.
    /// Rows split exactly like C's `_f32` entry (`nr = V->ne[1]*V->ne[3]`,
    /// ops.cpp:11049-11088).
    fn forward_gated_delta_net(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let src: Vec<TensorId> = (0..6).map(|i| td.src[i].expect("gdn src")).collect();
        let k_slots = td.op_params[0] as i64;
        // raw addresses (usize keeps the closure Send+Sync, like the other kernels)
        let addr = |id: TensorId| self.rd(id, 0, 0).as_ptr() as usize;
        let wrap = |id: TensorId| {
            let t = self.t(id);
            crate::gdn::GdnSrc { ptr: addr(id) as *const u8, nb: t.nb, ne: t.ne }
        };
        let (q, k, v, g, beta, state) = (
            wrap(src[0]),
            wrap(src[1]),
            wrap(src[2]),
            wrap(src[3]),
            wrap(src[4]),
            wrap(src[5]),
        );
        crate::gdn::gdn_check(&q, &k, &v, &g, &beta, &state, k_slots);

        let h_v = v.ne[1] as usize;
        let n_seqs = v.ne[3] as usize;
        let nr = h_v * n_seqs;
        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        let (qp, kp, vp, gp, bp, sp) = (
            q.ptr as usize,
            k.ptr as usize,
            v.ptr as usize,
            g.ptr as usize,
            beta.ptr as usize,
            state.ptr as usize,
        );
        let (q_nb, k_nb, v_nb, g_nb, b_nb, s_nb) = (q.nb, k.nb, v.nb, g.nb, beta.nb, state.nb);
        let (q_ne, k_ne, v_ne, g_ne, b_ne, s_ne) = (q.ne, k.ne, v.ne, g.ne, beta.ne, state.ne);
        let run = move |ir0: usize, ir1: usize| unsafe {
            let q = crate::gdn::GdnSrc { ptr: qp as *const u8, nb: q_nb, ne: q_ne };
            let k = crate::gdn::GdnSrc { ptr: kp as *const u8, nb: k_nb, ne: k_ne };
            let v = crate::gdn::GdnSrc { ptr: vp as *const u8, nb: v_nb, ne: v_ne };
            let g = crate::gdn::GdnSrc { ptr: gp as *const u8, nb: g_nb, ne: g_ne };
            let beta = crate::gdn::GdnSrc { ptr: bp as *const u8, nb: b_nb, ne: b_ne };
            let state = crate::gdn::GdnSrc { ptr: sp as *const u8, nb: s_nb, ne: s_ne };
            crate::gdn::gdn_f32(
                dst_p as *mut u8, &q, &k, &v, &g, &beta, &state, ir0, ir1,
            )
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // rwkv wkv — 对照 ops.cpp:10413-10603 / 10623-11418 / 11422-11617
    // (batch 14 round 2, agent WKV; kernels in wkv.rs)
    // ==================================================================

    /// 对照 ggml_compute_forward_rwkv_wkv6 (ops.cpp:10408) → wkv.rs. The
    /// heads are sharded like C's h_start/h_end (ops.cpp:10434-10437) —
    /// contiguous ascending ranges via par_rows.
    fn forward_rwkv_wkv6(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let (k, v, r, tf, td_src, state) = (
            td.src[0].expect("wkv6: k"),
            td.src[1].expect("wkv6: v"),
            td.src[2].expect("wkv6: r"),
            td.src[3].expect("wkv6: tf"),
            td.src[4].expect("wkv6: td"),
            td.src[5].expect("wkv6: state"),
        );
        let (kt, vt, rt) = (self.t(k), self.t(v), self.t(r));
        let (s, heads, t_len) = (kt.ne[0] as usize, kt.ne[1] as usize, kt.ne[2] as usize);
        let n_seqs = self.t(state).ne[1] as usize;
        debug_assert_eq!((vt.ne[0], vt.ne[1], vt.ne[2]), (kt.ne[0], kt.ne[1], kt.ne[2]));
        debug_assert_eq!((rt.ne[0], rt.ne[1], rt.ne[2]), (kt.ne[0], kt.ne[1], kt.ne[2]));

        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        // ith == 0 memset(dst_data, 0, T*C) + ggml_barrier (ops.cpp:10456-10458)
        unsafe { std::ptr::write_bytes(dst_p as *mut u8, 0, t_len * s * heads * 4) };

        let (kp, vp, rp, tfp, tdp, sp) = (
            self.rd(k, 0, 0).as_ptr() as usize,
            self.rd(v, 0, 0).as_ptr() as usize,
            self.rd(r, 0, 0).as_ptr() as usize,
            self.rd(tf, 0, 0).as_ptr() as usize,
            self.rd(td_src, 0, 0).as_ptr() as usize,
            self.rd(state, 0, 0).as_ptr() as usize,
        );
        let run = move |h0: usize, h1: usize| unsafe {
            crate::wkv::rwkv_wkv6_f32(
                dst_p as *mut f32,
                kp as *const f32,
                vp as *const f32,
                rp as *const f32,
                tfp as *const f32,
                tdp as *const f32,
                sp as *const f32,
                h0,
                h1,
                s,
                heads,
                t_len,
                n_seqs,
            )
        };
        self.par_rows(heads, nth, team, run);
    }

    /// 对照 ggml_compute_forward_gla (ops.cpp:10618) → wkv.rs.
    fn forward_gated_linear_attn(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let (k, v, q, g, state) = (
            td.src[0].expect("gla: k"),
            td.src[1].expect("gla: v"),
            td.src[2].expect("gla: q"),
            td.src[3].expect("gla: g"),
            td.src[4].expect("gla: state"),
        );
        let kt = self.t(k);
        let (s, heads, t_len) = (kt.ne[0] as usize, kt.ne[1] as usize, kt.ne[2] as usize);
        let n_seqs = self.t(state).ne[1] as usize;
        // ggml_get_op_params_f32(dst, 0) (ops.cpp:10631)
        let scale = f32::from_bits(td.op_params[0] as u32);

        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        // ith == 0 memset + barrier (ops.cpp:10666-10668)
        unsafe { std::ptr::write_bytes(dst_p as *mut u8, 0, t_len * s * heads * 4) };

        let (kp, vp, qp, gp, sp) = (
            self.rd(k, 0, 0).as_ptr() as usize,
            self.rd(v, 0, 0).as_ptr() as usize,
            self.rd(q, 0, 0).as_ptr() as usize,
            self.rd(g, 0, 0).as_ptr() as usize,
            self.rd(state, 0, 0).as_ptr() as usize,
        );
        let run = move |h0: usize, h1: usize| unsafe {
            crate::wkv::gla_f32(
                dst_p as *mut f32,
                kp as *const f32,
                vp as *const f32,
                qp as *const f32,
                gp as *const f32,
                sp as *const f32,
                scale,
                h0,
                h1,
                s,
                heads,
                t_len,
                n_seqs,
            )
        };
        self.par_rows(heads, nth, team, run);
    }

    /// 对照 ggml_compute_forward_rwkv_wkv7 (ops.cpp:11417) → wkv.rs. No
    /// memset: the kernel assigns every dst element (REDUCE per (t, h, i)).
    fn forward_rwkv_wkv7(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let (r, w, k, v, a, b, state) = (
            td.src[0].expect("wkv7: r"),
            td.src[1].expect("wkv7: w"),
            td.src[2].expect("wkv7: k"),
            td.src[3].expect("wkv7: v"),
            td.src[4].expect("wkv7: a"),
            td.src[5].expect("wkv7: b"),
            td.src[6].expect("wkv7: state"),
        );
        let kt = self.t(k);
        let (s, heads, t_len) = (kt.ne[0] as usize, kt.ne[1] as usize, kt.ne[2] as usize);
        let n_seqs = self.t(state).ne[1] as usize;

        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        let (rp, wp, kp, vp, ap, bp, sp) = (
            self.rd(r, 0, 0).as_ptr() as usize,
            self.rd(w, 0, 0).as_ptr() as usize,
            self.rd(k, 0, 0).as_ptr() as usize,
            self.rd(v, 0, 0).as_ptr() as usize,
            self.rd(a, 0, 0).as_ptr() as usize,
            self.rd(b, 0, 0).as_ptr() as usize,
            self.rd(state, 0, 0).as_ptr() as usize,
        );
        let run = move |h0: usize, h1: usize| unsafe {
            crate::wkv::rwkv_wkv7_f32(
                dst_p as *mut f32,
                rp as *const f32,
                wp as *const f32,
                kp as *const f32,
                vp as *const f32,
                ap as *const f32,
                bp as *const f32,
                sp as *const f32,
                h0,
                h1,
                s,
                heads,
                t_len,
                n_seqs,
            )
        };
        if s < crate::wkv::F32_STEP {
            // head_size < 64: the 64-wide window's state writes cross head
            // blocks (wkv.rs module header — the reference's own race); the
            // head loop runs single-threaded in ascending order so the port
            // stays deterministic (the C's output is nth==1-only there too)
            run(0, heads);
        } else {
            // S >= 64: the window never leaves the head's own block — the
            // per-head shards are disjoint, parallel is race-free
            self.par_rows(heads, nth, team, run);
        }
    }

    // ==================================================================
    // rope — 对照 ops.cpp:5951-6237
    // ==================================================================

    /// ops.cpp:5951 rope_yarn_ramp
    fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
        let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
        1.0 - y.clamp(0.0, 1.0)
    }

    /// ops.cpp:5958 rope_yarn
    fn rope_yarn(
        theta_extrap: f32,
        freq_scale: f32,
        corr_dims: &[f32; 2],
        i0: usize,
        ext_factor: f32,
        mut mscale: f32,
    ) -> (f32, f32) {
        let theta_interp = freq_scale * theta_extrap;
        let mut theta = theta_interp;
        if ext_factor != 0.0 {
            let ramp_mix = Self::rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        }
        (theta.cos() * mscale, theta.sin() * mscale)
    }

    /// ops.cpp:5975 ggml_rope_cache_init — `freq_factors` is src[2] (gemma4's
    /// per-layer `rope_freqs`, gemma4.cpp:140/168).
    fn rope_cache_init(
        theta_base: f32,
        freq_scale: f32,
        freq_factors: Option<&[f32]>,
        corr_dims: &[f32; 2],
        ne0: usize,
        ext_factor: f32,
        mscale: f32,
        cache: &mut [f32],
        sin_sign: f32,
        theta_scale: f32,
    ) {
        let mut theta = theta_base;
        let mut i0 = 0;
        while i0 < ne0 {
            let ff = freq_factors.map(|f| f[i0 / 2]).unwrap_or(1.0f32);
            let (c, s) =
                Self::rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale);
            cache[i0] = c;
            cache[i0 + 1] = s * sin_sign;
            theta *= theta_scale;
            i0 += 2;
        }
    }

    /// ops.cpp:5991 ggml_mrope_cache_init (indep_sects == false in C unless vision)
    /// `is_imrope` selects qwen3vl's interleaved sector walk (qwen35's rope type).
    #[allow(clippy::too_many_arguments)]
    fn mrope_cache_init(
        theta_base: [f32; 4],
        sections: &[i32; 4],
        is_imrope: bool,
        // C's `indep_sects` (ops.cpp:5994): vision-mode rope restarts each
        // section's theta at that section's base position.
        indep_sects: bool,
        freq_factors: Option<&[f32]>,
        freq_scale: f32,
        corr_dims: &[f32; 2],
        ne0: usize,
        ext_factor: f32,
        mscale: f32,
        cache: &mut [f32],
        sin_sign: f32,
        theta_scale: f32,
    ) {
        let mut theta = theta_base;
        let sect_dims = (sections[0] + sections[1] + sections[2] + sections[3]) as usize;
        let sec_w = (sections[1] + sections[0]) as usize;
        let sec_e = (sections[2] as usize) + sec_w;
        assert!(sect_dims <= ne0);

        let mut i0 = 0;
        while i0 < ne0 {
            let ff = freq_factors.map(|f| f[i0 / 2]).unwrap_or(1.0f32);
            let sector = (i0 / 2) % sect_dims;
            if indep_sects {
                // ops.cpp:6003-6014: reset the section's theta when the sector
                // changes (vision encoder)
                if sector == 0 {
                    theta[0] = theta_base[0];
                } else if sector == sections[0] as usize {
                    theta[1] = theta_base[1];
                } else if sector == sec_w {
                    theta[2] = theta_base[2];
                } else if sector == sec_e {
                    theta[3] = theta_base[3];
                }
            }
            let t = if is_imrope {
                // ops.cpp:6027-6036: qwen3vl interleaved mrope
                if sector % 3 == 1 && sector < 3 * sections[1] as usize {
                    theta[1]
                } else if sector % 3 == 2 && sector < 3 * sections[2] as usize {
                    theta[2]
                } else if sector % 3 == 0 && sector < 3 * sections[0] as usize {
                    theta[0]
                } else {
                    theta[3]
                }
            } else if sector >= sections[0] as usize && sector < sec_w {
                theta[1]
            } else if sector >= sec_w && sector < sec_w + sections[2] as usize {
                theta[2]
            } else if sector >= sec_w + sections[2] as usize {
                theta[3]
            } else {
                theta[0]
            };
            let (c, s) = Self::rope_yarn(t / ff, freq_scale, corr_dims, i0, ext_factor, mscale);
            cache[i0] = c;
            cache[i0 + 1] = s * sin_sign;
            theta[0] *= theta_scale;
            theta[1] *= theta_scale;
            theta[2] *= theta_scale;
            theta[3] *= theta_scale;
            i0 += 2;
        }
    }

    /// ggml.c:4469 ggml_rope_yarn_corr_dim / ggml_rope_yarn_corr_dims
    fn rope_yarn_corr_dims(n_dims: i32, n_ctx_orig: i32, freq_base: f32, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
        let corr_dim = |n_rot: f32| -> f32 {
            n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * freq_base.ln())
        };
        let start = corr_dim(beta_fast).floor();
        let end = corr_dim(beta_slow).ceil();
        [start.max(0.0), end.min((n_dims - 1) as f32)]
    }

    /// ops.cpp:6063 rotate_pairs (scalar): scale=1 for NORMAL (ic = i0),
    /// scale=2 otherwise (ic = i0/2).
    ///
    /// ops.cpp:6076-6077 `x0*cos_theta - x1*sin_theta` / `x0*sin_theta +
    /// x1*cos_theta` are FMA-contracted by the reference build (GCC 13.3
    /// `-O3 -march=native`, default `-ffp-contract=fast`): the right product
    /// is materialized by `vmulss` (one rounding) and the left one fused —
    /// `.so` disassembly `vmulss/vfmsub231ss` + `vmulss/vfmadd132ss` right
    /// after the two cache loads, and every one of the 8192 Q-rope elements
    /// of the qwen3-embedding layer-0 dump (parity/ref_decode_dump.c) equals
    /// `fma(x0, cos, -(x1*sin))` / `fma(x0, sin, x1*cos)` bit-for-bit (the
    /// plain two-rounding form missed 8192-6608 of them). `f32::mul_add` is
    /// the same fused single-rounding op.
    fn rotate_pairs_f32(n: usize, n_offset: usize, cache: &[f32], src: &[f32], dst: &mut [f32], scale: usize) {
        let mut i0 = 0;
        while i0 < n {
            let ic = i0 / scale;
            let cos_theta = cache[i0];
            let sin_theta = cache[i0 + 1];
            let x0 = src[ic];
            let x1 = src[ic + n_offset];
            dst[ic] = x0.mul_add(cos_theta, -(x1 * sin_theta));
            dst[ic + n_offset] = x0.mul_add(sin_theta, x1 * cos_theta);
            i0 += 2;
        }
    }

    fn forward_rope(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);

        let params = &self.t(dst).op_params;
        let n_dims = params[1] as usize;
        let mode = params[2];
        let n_ctx_orig = params[4];
        let freq_base = f32::from_bits(params[5] as u32);
        let freq_scale = f32::from_bits(params[6] as u32);
        let ext_factor = f32::from_bits(params[7] as u32);
        let attn_factor = f32::from_bits(params[8] as u32);
        let beta_fast = f32::from_bits(params[9] as u32);
        let beta_slow = f32::from_bits(params[10] as u32);
        let sections: [i32; 4] = [params[11], params[12], params[13], params[14]];
        let n_offs = params[15] as usize;

        assert_eq!(t0.nb[0] as usize, td.nb[0] as usize);
        assert_eq!(t0.nb[0] as usize, t0.ty.type_size());

        let ne0 = td.ne[0] as usize;
        let ne1 = td.ne[1] as usize;
        let ne2 = td.ne[2] as usize;
        let ne3 = td.ne[3] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        assert!(n_dims <= ne0);
        assert_eq!(n_dims % 2, 0);
        assert_eq!(n_offs % 2, 0);
        assert!(n_offs + n_dims <= ne0);

        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let corr_dims = Self::rope_yarn_corr_dims(n_dims as i32, n_ctx_orig, freq_base, beta_fast, beta_slow);

        let is_vision = mode == GGML_ROPE_TYPE_VISION;
        let mrope_used = mode & GGML_ROPE_TYPE_MROPE != 0;
        let is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

        let pos: &[i32] = self.rd_i32s(s1, 0, t1.n_elements() as usize);

        // ops.cpp:6171 `const float sin_sign = forward ? 1.0f : -1.0f;` — the
        // GGML_OP_ROPE_BACK entry point runs this same body with forward=false
        // (ops.cpp:6264-6285), i.e. the negated sine of the inverse rotation
        // (deepseek4's attn de-rope, deepseek4.cpp:1205)
        let sin_sign = if td.op == GgmlOp::RoPEBack { -1.0 } else { 1.0 };

        // src[2] = optional rope freq factors (ggml.c:4312-4320)
        let ff_id = self.t(dst).src[2];
        let ff_buf: Option<Vec<f32>> =
            ff_id.map(|id| self.rd_f32s(id, 0, self.t(id).ne[0] as usize).to_vec());
        let freq_factors: Option<&[f32]> = ff_buf.as_deref();

        let nr = ne1 * ne2 * ne3;

        let run = |ir0: usize, ir1: usize| {
            let mut cache = vec![0f32; ne0];
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;

                if !mrope_used {
                    let p = pos[i2] as f32;
                    Self::rope_cache_init(
                        p, freq_scale, freq_factors, &corr_dims, ne0, ext_factor, attn_factor,
                        &mut cache, sin_sign, theta_scale,
                    );
                } else {
                    let pt = pos[i2] as f32;
                    let ph = pos[i2 + ne2] as f32;
                    let pw = pos[i2 + ne2 * 2] as f32;
                    let pe = pos[i2 + ne2 * 3] as f32;
                    Self::mrope_cache_init(
                        [pt, ph, pw, pe], &sections, is_imrope, is_vision, freq_factors,
                        freq_scale, &corr_dims, ne0, ext_factor, attn_factor, &mut cache,
                        sin_sign, theta_scale,
                    );
                }

                let src_off = i3 * nb03 + i2 * nb02 + i1 * nb01;
                let dst_off = i3 * nb3 + i2 * nb2 + i1 * nb1;

                match t0.ty {
                    GgmlType::F32 => {
                        let src: &[f32] = self.rd_f32s(s0, src_off, ne0);
                        let out: &mut [f32] = self.wr_f32s(dst, dst_off, ne0);
                        match mode {
                            GGML_ROPE_TYPE_NORMAL => {
                                Self::rotate_pairs_f32(n_dims, 1, &cache, &src[n_offs..], &mut out[n_offs..], 1);
                            }
                            GGML_ROPE_TYPE_NEOX | GGML_ROPE_TYPE_MROPE | GGML_ROPE_TYPE_IMROPE => {
                                Self::rotate_pairs_f32(n_dims, n_dims / 2, &cache, &src[n_offs..], &mut out[n_offs..], 2);
                            }
                            GGML_ROPE_TYPE_VISION => {
                                Self::rotate_pairs_f32(ne0, n_dims, &cache, src, out, 2);
                            }
                            _ => panic!("rope type not supported"),
                        }
                        if !is_vision {
                            // copy the non-rotated channels
                            let mut i0 = 0;
                            while i0 < ne0 {
                                if i0 == n_offs {
                                    i0 += n_dims - 2; // skip rotated channels
                                    i0 += 2;
                                    continue;
                                }
                                out[i0] = src[i0];
                                out[i0 + 1] = src[i0 + 1];
                                i0 += 2;
                            }
                        }
                    }
                    GgmlType::F16 => {
                        let src: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, ne0 * 2));
                        let out: &mut [f16] = bytemuck::cast_slice_mut(self.wr(dst, dst_off, ne0 * 2));
                        let mut xs = vec![0f32; ne0];
                        let mut xo = vec![0f32; ne0];
                        for j in 0..ne0 {
                            xs[j] = src[j].to_f32();
                        }
                        match mode {
                            GGML_ROPE_TYPE_NORMAL => {
                                Self::rotate_pairs_f32(n_dims, 1, &cache, &xs[n_offs..], &mut xo[n_offs..], 1);
                            }
                            GGML_ROPE_TYPE_NEOX | GGML_ROPE_TYPE_MROPE | GGML_ROPE_TYPE_IMROPE => {
                                Self::rotate_pairs_f32(n_dims, n_dims / 2, &cache, &xs[n_offs..], &mut xo[n_offs..], 2);
                            }
                            GGML_ROPE_TYPE_VISION => {
                                Self::rotate_pairs_f32(ne0, n_dims, &cache, &xs, &mut xo, 2);
                            }
                            _ => panic!("rope type not supported"),
                        }
                        if !is_vision {
                            let mut i0 = 0;
                            while i0 < ne0 {
                                if i0 == n_offs {
                                    i0 += n_dims - 2;
                                    i0 += 2;
                                    continue;
                                }
                                xo[i0] = xs[i0];
                                xo[i0 + 1] = xs[i0 + 1];
                                i0 += 2;
                            }
                        }
                        for j in 0..ne0 {
                            out[j] = f16::from_f32(xo[j]);
                        }
                    }
                    other => unimplemented!("rope for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // get_rows — 对照 ops.cpp:4979-5144
    // ==================================================================

    fn forward_get_rows(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);

        let nc = t0.ne[0] as usize;
        let nr = t1.n_elements() as usize;
        let (ne10, ne11) = (t1.ne[0] as usize, t1.ne[1] as usize);
        let (nb10, nb11, nb12) = (t1.nb[0] as usize, t1.nb[1] as usize, t1.nb[2] as usize);
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let run = |i0: usize, i1: usize| {
            for i in i0..i1 {
                let i12 = i / (ne11 * ne10);
                let i11 = (i - i12 * ne11 * ne10) / ne10;
                let i10 = i - i12 * ne11 * ne10 - i11 * ne10;
                let row = self.rd_i32s(s1, i10 * nb10 + i11 * nb11 + i12 * nb12, 1)[0] as usize;
                assert!(row < t0.ne[1] as usize, "get_rows: index out of range");

                let src_off = row * nb01 + i11 * nb02 + i12 * nb03;
                let dst_off = i10 * nb1 + i11 * nb2 + i12 * nb3;

                let dst_bytes = self.wr(dst, dst_off, ty_row_bytes(td.ty, nc));
                match t0.ty {
                    GgmlType::F32 | GgmlType::I32 => {
                        let src_bytes = self.rd(s0, src_off, ty_row_bytes(t0.ty, nc));
                        dst_bytes.copy_from_slice(src_bytes);
                    }
                    GgmlType::F16 => {
                        let x: &[f16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                        for j in 0..nc {
                            y[j] = x[j].to_f32();
                        }
                    }
                    GgmlType::Bf16 => {
                        let x: &[bf16] = bytemuck::cast_slice(self.rd(s0, src_off, nc * 2));
                        let y: &mut [f32] = bytemuck::cast_slice_mut(dst_bytes);
                        for j in 0..nc {
                            y[j] = x[j].to_f32();
                        }
                    }
                    q => {
                        quants::dequantize_row(q, self.rd(s0, src_off, ty_row_bytes(q, nc)), bytemuck::cast_slice_mut(dst_bytes));
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// 对照 ops.cpp:5361 ggml_compute_forward_get_rows_back_f32
    fn forward_get_rows_back(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);

        assert_eq!(t0.ty, GgmlType::F32);
        let nc = t0.ne[0] as usize;
        let nr = t1.n_elements() as usize;
        let nb01 = t0.nb[1] as usize;
        let nb1 = self.t(dst).nb[1] as usize;

        // memset dst
        let total = self.t(dst).n_elements() as usize * 4;
        for b in self.wr(dst, 0, total) {
            *b = 0;
        }

        let pos: &[i32] = self.rd_i32s(s1, 0, nr);
        let dst_ne1 = self.t(dst).ne[1];
        for (i, &r) in pos.iter().enumerate() {
            // ops.cpp sync batch D (:5579/:5614): GGML_ASSERT(r >= 0 && r <
            // dst->ne[1]) in both get_rows_back_f32_f16 and _f32
            assert!(r >= 0 && (r as i64) < dst_ne1, "get_rows_back: row {r} out of range");
            let r = r as usize;
            let x: &[f32] = self.rd_f32s(s0, i * nb01, nc);
            let y: &mut [f32] = self.wr_f32s(dst, r * nb1, nc);
            for j in 0..nc {
                y[j] += x[j];
            }
        }
    }

    // ==================================================================
    // dup / cpy / cont — 对照 ops.cpp:17-574
    // ==================================================================

    fn forward_dup(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.n_elements(), td.n_elements(), "dup: nelements mismatch");

        if t0.ty == td.ty {
            self.dup_bytes(dst, s0, nth, team);
            return;
        }

        match (t0.ty, td.ty) {
            // non-quantized conversions via f32 (ops.cpp:47 ggml_compute_forward_dup_flt)
            (GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::I32) => {
                self.dup_flt(dst, s0, nth, team);
            }
            // ops.cpp:570-575: I32 src — dup_flt<int32_t, float>, any other dst aborts
            (GgmlType::I32, GgmlType::F32) => {
                self.dup_flt(dst, s0, nth, team);
            }
            // quantized src -> f32 dst (ops.cpp:475 ggml_compute_forward_dup_from_q)
            (q, GgmlType::F32) if q.is_quantized() => {
                self.dup_from_q(dst, s0);
            }
            // f32 src -> quantized dst (ops.cpp:269 ggml_compute_forward_dup_to_q)
            (GgmlType::F32, q) if q.is_quantized() => {
                self.dup_to_q(dst, s0, q);
            }
            (a, b) => unimplemented!("dup {a:?} -> {b:?}"),
        }
    }

    /// ops.cpp:326 ggml_compute_forward_dup_bytes
    fn dup_bytes(&self, dst: TensorId, s0: TensorId, nth: usize, team: &Team<'_>) {
        let t0 = self.t(s0);
        let td = self.t(dst);
        let ty = t0.ty;
        let ts = ty.type_size();
        let blck = ty.blck_size();

        let (ne00, ne01, ne02, ne03) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (ne0, ne1, ne2, ne3) = (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
        let (nb00, nb01, nb02, nb03) = (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        // both fully contiguous: single block copy
        if is_contiguous_ctx_of(self, s0) && is_contiguous_ctx_of(self, dst) {
            let total = ty_row_bytes(ty, t0.n_elements() as usize);
            let src = self.rd(s0, 0, total);
            let d = self.wr(dst, 0, total);
            d.copy_from_slice(src);
            return;
        }

        let same_shape = ne00 == ne0 && ne01 == ne1 && ne02 == ne2 && ne03 == ne3;
        let row_bytes = ty_row_bytes(ty, ne00);

        if same_shape && nb00 == ts && nb0 == ts {
            // copy by rows (ops.cpp:354)
            let nr = ne01 * ne02 * ne03;
            let run = |ir0: usize, ir1: usize| {
                for ir in ir0..ir1 {
                    let i03 = ir / (ne02 * ne01);
                    let i02 = (ir - i03 * ne02 * ne01) / ne01;
                    let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                    let src = self.rd(s0, i01 * nb01 + i02 * nb02 + i03 * nb03, row_bytes);
                    self.wr(dst, i01 * nb1 + i02 * nb2 + i03 * nb3, row_bytes).copy_from_slice(src);
                }
            };
            self.par_rows(nr, nth, team, run);
            return;
        }

        if is_contiguous_ctx_of(self, dst) {
            // contiguous dst: linear cursor over dst, row-major src (ops.cpp:372)
            let mut id = 0usize; // dst byte cursor
            for i03 in 0..ne03 {
                for i02 in 0..ne02 {
                    for i01 in 0..ne01 {
                        if nb00 == ts {
                            let src = self.rd(s0, i01 * nb01 + i02 * nb02 + i03 * nb03, row_bytes);
                            self.wr(dst, id, row_bytes).copy_from_slice(src);
                            id += row_bytes;
                        } else {
                            for i00 in 0..ne00 {
                                let src = self.rd(s0, i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03, ts);
                                self.wr(dst, id, ts).copy_from_slice(src);
                                id += ts;
                            }
                        }
                    }
                }
            }
            return;
        }

        // general: walk both shapes block-by-block (ops.cpp:412 counters);
        // element k of src (row-major) maps to element k of dst (row-major)
        let total_blocks = t0.n_elements() as usize / blck;
        let _ = (ne1, ne2, ne3);
        for kb in 0..total_blocks {
            // src position (block units)
            let b_per_row = ne00 / blck;
            let i03 = kb / (b_per_row * ne01 * ne02);
            let i02 = (kb - i03 * b_per_row * ne01 * ne02) / (b_per_row * ne01);
            let i01 = (kb - i03 * b_per_row * ne01 * ne02 - i02 * b_per_row * ne01) / b_per_row;
            let k00 = kb - i03 * b_per_row * ne01 * ne02 - i02 * b_per_row * ne01 - i01 * b_per_row;
            let src_off = k00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

            // dst position (block units)
            let db_per_row = ne0 / blck;
            let j03 = kb / (db_per_row * ne1 * ne2);
            let j02 = (kb - j03 * db_per_row * ne1 * ne2) / (db_per_row * ne1);
            let j01 = (kb - j03 * db_per_row * ne1 * ne2 - j02 * db_per_row * ne1) / db_per_row;
            let j00 = kb - j03 * db_per_row * ne1 * ne2 - j02 * db_per_row * ne1 - j01 * db_per_row;
            let dst_off = j00 * nb0 + j01 * nb1 + j02 * nb2 + j03 * nb3;

            let src = self.rd(s0, src_off, ts);
            self.wr(dst, dst_off, ts).copy_from_slice(src);
        }
    }

    /// ops.cpp:47 ggml_compute_forward_dup_flt (f32/f16/bf16/i32 conversions)
    fn dup_flt(&self, dst: TensorId, s0: TensorId, nth: usize, team: &Team<'_>) {
        let t0 = self.t(s0);
        let td = self.t(dst);
        let (ne00, ne01, ne02, ne03) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (ne0, ne1, ne2, ne3) = (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
        let (nb00, nb01, nb02, nb03) = (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let sty = t0.ty;
        let dty = td.ty;
        let ssz = sty.type_size();
        let dsz = dty.type_size();

        let to_f32: fn(&[u8]) -> f32 = match sty {
            GgmlType::F32 => |p| f32::from_le_bytes(p.try_into().unwrap()),
            GgmlType::F16 => |p| f16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            GgmlType::Bf16 => |p| bf16::from_le_bytes(p.try_into().unwrap()).to_f32(),
            // ops.cpp:570-575: GGML_TYPE_I32 src — only the F32 dst is legal
            // (dup_flt<int32_t, float>, the grovemoe chunk-expert selection)
            GgmlType::I32 => |p| i32::from_le_bytes(p.try_into().unwrap()) as f32,
            other => unimplemented!("dup_flt src {other:?}"),
        };
        let from_f32: fn(f32, &mut [u8]) = match dty {
            GgmlType::F32 => |v, p| p.copy_from_slice(&v.to_le_bytes()),
            GgmlType::F16 => |v, p| p.copy_from_slice(&f16::from_f32(v).to_le_bytes()),
            GgmlType::Bf16 => |v, p| p.copy_from_slice(&bf16::from_f32(v).to_le_bytes()),
            GgmlType::I32 => |v, p| p.copy_from_slice(&(v as i32).to_le_bytes()),
            other => unimplemented!("dup_flt dst {other:?}"),
        };

        if is_contiguous_ctx_of(self, dst) {
            if nb00 == ssz {
                // row-wise conversion into contiguous dst (ops.cpp:108)
                let nr = ne01 * ne02 * ne03;
                let run = |ir0: usize, ir1: usize| {
                    let mut id = ir0 * ne00 * dsz; // rows before this thread's range
                    for ir in ir0..ir1 {
                        let i03 = ir / (ne02 * ne01);
                        let i02 = (ir - i03 * ne02 * ne01) / ne01;
                        let i01 = ir - i03 * ne02 * ne01 - i02 * ne01;
                        let src_off = i01 * nb01 + i02 * nb02 + i03 * nb03;
                        for i00 in 0..ne00 {
                            let v = to_f32(self.rd(s0, src_off + i00 * nb00, ssz));
                            from_f32(v, self.wr(dst, id, dsz));
                            id += dsz;
                        }
                    }
                };
                self.par_rows(nr, nth, team, run);
            } else {
                // element-strided src into contiguous dst (ops.cpp:128)
                let total = t0.n_elements() as usize;
                let run = |k0: usize, k1: usize| {
                    let mut id = k0 * dsz;
                    for k in k0..k1 {
                        let i03 = k / (ne02 * ne01 * ne00);
                        let i02 = (k - i03 * ne02 * ne01 * ne00) / (ne01 * ne00);
                        let i01 = (k - i03 * ne02 * ne01 * ne00 - i02 * ne01 * ne00) / ne00;
                        let i00 = k % ne00;
                        let src_off = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                        let v = to_f32(self.rd(s0, src_off, ssz));
                        from_f32(v, self.wr(dst, id, dsz));
                        id += dsz;
                    }
                };
                self.par_rows(total, nth, team, run);
            }
            return;
        }

        // general: row-major element k of src -> row-major element k of dst
        let total = t0.n_elements() as usize;
        let _ = (nb1, nb2, nb3);
        for k in 0..total {
            let i03 = k / (ne02 * ne01 * ne00);
            let i02 = (k - i03 * ne02 * ne01 * ne00) / (ne01 * ne00);
            let i01 = (k - i03 * ne02 * ne01 * ne00 - i02 * ne01 * ne00) / ne00;
            let i00 = k % ne00;
            let j03 = k / (ne3 * ne2 * ne1 * ne0);
            let j02 = (k - j03 * ne3 * ne2 * ne1 * ne0) / (ne2 * ne1 * ne0);
            let j01 = (k - j03 * ne3 * ne2 * ne1 * ne0 - j02 * ne2 * ne1 * ne0) / (ne1 * ne0);
            let j00 = k % ne0;
            let v = to_f32(self.rd(s0, i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03, ssz));
            from_f32(v, self.wr(dst, j00 * nb0 + j01 * td.nb[1] as usize + j02 * td.nb[2] as usize + j03 * td.nb[3] as usize, dsz));
        }
    }

    /// ops.cpp:475 ggml_compute_forward_dup_from_q (quantized -> f32)
    fn dup_from_q(&self, dst: TensorId, s0: TensorId) {
        let t0 = self.t(s0);
        let td = self.t(dst);
        let ty = t0.ty;
        let qk = ty.blck_size();
        let ts = ty.type_size();
        let (ne00, ne01, ne02) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize);
        let (nb00, nb01, nb02, nb03) = (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);

        assert_eq!(td.nb[0] as usize, 4, "dup_from_q: dst must be f32-contiguous in dim0");

        let nblocks = t0.n_elements() as usize / qk;
        let mut tmp = vec![0f32; qk];
        for kb in 0..nblocks {
            let e = kb * qk; // first element of this block (linear)
            let i03 = e / (ne02 * ne01 * ne00);
            let i02 = (e - i03 * ne02 * ne01 * ne00) / (ne01 * ne00);
            let i01 = (e - i03 * ne02 * ne01 * ne00 - i02 * ne01 * ne00) / ne00;
            let i00 = e - i03 * ne02 * ne01 * ne00 - i02 * ne01 * ne00 - i01 * ne00;
            let src_off = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

            quants::dequantize_row(ty, self.rd(s0, src_off, ts), &mut tmp);
            let y: &mut [f32] = self.wr_f32s(dst, e * 4, qk);
            y.copy_from_slice(&tmp);
        }
    }

    /// ops.cpp:269 ggml_compute_forward_dup_to_q (f32 -> quantized)
    fn dup_to_q(&self, dst: TensorId, s0: TensorId, q: GgmlType) {
        let t0 = self.t(s0);
        assert!(is_contiguous_ctx_of(self, dst), "dup_to_q: dst must be contiguous");
        assert_eq!(t0.nb[0] as usize, 4, "dup_to_q: src0 must be f32-contiguous in dim0");

        let (ne00, ne01, ne02, ne03) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let rs = ty_row_bytes(q, ne00);

        let quantize: fn(&[f32], &mut [u8]) = match q {
            GgmlType::Q4_0 => |x, o| quants::quantize_row_q4_0_ref(x, bytemuck::cast_slice_mut(o)),
            GgmlType::Q4_1 => |x, o| quants::quantize_row_q4_1_ref(x, bytemuck::cast_slice_mut(o)),
            GgmlType::Q5_0 => |x, o| quants::quantize_row_q5_0_ref(x, bytemuck::cast_slice_mut(o)),
            GgmlType::Q5_1 => |x, o| quants::quantize_row_q5_1_ref(x, bytemuck::cast_slice_mut(o)),
            GgmlType::Q8_0 => |x, o| quants::quantize_row_q8_0(x, bytemuck::cast_slice_mut(o)),
            GgmlType::Q8_1 => |x, o| quants::quantize_row_q8_1(x, bytemuck::cast_slice_mut(o)),
            other => unimplemented!("dup_to_q for {other:?} (K-quant quantizers pending)"),
        };

        let mut id = 0usize;
        for i03 in 0..ne03 {
            for i02 in 0..ne02 {
                for i01 in 0..ne01 {
                    let row: &[f32] = self.rd_f32s(s0, i01 * nb01 + i02 * nb02 + i03 * nb03, ne00);
                    quantize(row, self.wr(dst, id, rs));
                    id += rs;
                }
            }
        }
    }

    // ==================================================================
    // argmax / argsort — 对照 ops.cpp:1570 / ops.cpp:8489
    // ==================================================================

    fn forward_argmax(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let ne00 = t0.ne[0] as usize;
        let ne01 = t0.ne[1] as usize;
        let nb01 = t0.nb[1] as usize;

        for i1 in 0..ne01 {
            let x: &[f32] = self.rd_f32s(s0, i1 * nb01, ne00);
            // ggml_vec_argmax_f32: ties -> last occurrence
            let mut max = f32::NEG_INFINITY;
            let mut idx = 0usize;
            for (i, &v) in x.iter().enumerate() {
                max = if v > max { v } else { max };
                if max == v {
                    idx = i;
                }
            }
            self.wr(dst, i1 * 4, 4).copy_from_slice(&(idx as i32).to_le_bytes());
        }
    }

    fn forward_argsort(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let order = self.t(dst).op_params[0];
        let ne00 = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let nb01 = t0.nb[1] as usize;
        let nb1 = td.nb[1] as usize;

        for i in 0..nr {
            let src: &[f32] = self.rd_f32s(s0, i * nb01, ne00);
            let out: &mut [i32] = bytemuck::cast_slice_mut(self.wr(dst, i * nb1, ne00 * 4));
            for (j, o) in out.iter_mut().enumerate() {
                *o = j as i32;
            }
            match order {
                GGML_SORT_ORDER_ASC => {
                    out.sort_unstable_by(|&a, &b| src[a as usize].partial_cmp(&src[b as usize]).unwrap_or(std::cmp::Ordering::Equal))
                }
                _ => {
                    out.sort_unstable_by(|&a, &b| src[b as usize].partial_cmp(&src[a as usize]).unwrap_or(std::cmp::Ordering::Equal))
                }
            }
        }
    }

    // ==================================================================
    // sum_rows / repeat — 对照 ops.cpp:1460 / ops.cpp:1698
    // ==================================================================

    /// ops.cpp:5221 ggml_compute_forward_set_rows_impl — scatter rows of
    /// src[0] into dst at row indices src[1] (I64/I32). src[2] is the base
    /// tensor dst views (legacy C wiring). F32 rows convert to the dst type
    /// (from_float, RN); F16 rows memcpy when dst is F16.
    fn forward_set_rows(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let b = self.t(dst).src[0].expect("set_rows rows");
        let c = self.t(dst).src[1].expect("set_rows idx");
        let tb = self.t(b);
        let tc = self.t(c);
        let td = self.t(dst);
        let nc = tb.ne[0] as usize;
        let nr = tb.ne[1] as usize;
        assert_eq!(td.ne[0] as usize, nc);
        let ne02 = tb.ne[2] as usize;
        let ne03 = tb.ne[3] as usize;
        let ne11 = tc.ne[1].max(1) as usize;
        let ne12 = tc.ne[2].max(1) as usize;
        let nb01 = tb.nb[1] as usize;
        let nb02 = tb.nb[2] as usize;
        let nb03 = tb.nb[3] as usize;
        let nb10 = tc.nb[0] as usize;
        let nb11 = tc.nb[1] as usize;
        let nb12 = tc.nb[2] as usize;
        let nd1 = td.nb[1] as usize;
        let nd2 = td.nb[2] as usize;
        let nd3 = td.nb[3] as usize;
        let src_f16 = tb.ty == GgmlType::F16;
        let dst_f16 = td.ty == GgmlType::F16;
        let dst_ty = td.ty;
        debug_assert!(ne02 % ne11 == 0 && ne03 % ne12 == 0);

        self.par_rows(nr, nth, team, |ir0, ir1| {
            for i03 in 0..ne03 {
                for i02 in 0..ne02 {
                    let i12 = i03 % ne12;
                    let i11 = i02 % ne11;
                    for i in ir0..ir1 {
                        let idx_off = i * nb10 + i11 * nb11 + i12 * nb12;
                        let i1 = match tc.ty {
                            GgmlType::I64 => {
                                i64::from_le_bytes(self.rd(c, idx_off, 8).try_into().unwrap()) as usize
                            }
                            GgmlType::I32 => {
                                i32::from_le_bytes(self.rd(c, idx_off, 4).try_into().unwrap()) as usize
                            }
                            t => unreachable!("set_rows idx type {t:?}"),
                        };
                        assert!(i1 < td.ne[1] as usize, "set_rows: row index {i1} out of range");
                        let src_off = i * nb01 + i02 * nb02 + i03 * nb03;
                        let dst_off = i1 * nd1 + i02 * nd2 + i03 * nd3;
                        match (src_f16, dst_f16) {
                            (false, true) => {
                                // f32 rows -> f16 dst (from_float, round-to-nearest)
                                let xs: &[f32] = bytemuck::cast_slice(self.rd(b, src_off, nc * 4));
                                let ys = self.wr(dst, dst_off, nc * 2);
                                let hs: &mut [f16] = bytemuck::cast_slice_mut(ys);
                                for (h, &x) in hs.iter_mut().zip(xs) {
                                    *h = f16::from_f32(x);
                                }
                            }
                            (true, true) => {
                                let xs = self.rd(b, src_off, nc * 2);
                                let ys = self.wr(dst, dst_off, nc * 2);
                                ys.copy_from_slice(xs);
                            }
                            (_, false) => {
                                let rs = dst_ty.type_size() * nc;
                                let xs = self.rd(b, src_off, rs);
                                let ys = self.wr(dst, dst_off, rs);
                                ys.copy_from_slice(xs);
                            }
                        }
                    }
                }
            }
        });
    }

    fn forward_sum_rows(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        let ne00 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        for i3 in 0..t0.ne[3] as usize {
            for i2 in 0..t0.ne[2] as usize {
                for i1 in 0..t0.ne[1] as usize {
                    let x: &[f32] = self.rd_f32s(s0, i1 * nb01 + i2 * nb02 + i3 * nb03, ne00);
                    let mut sum = 0f64; // ggml_vec_sum_f32
                    for &v in x {
                        sum += v as f64;
                    }
                    self.wr_f32s(dst, i1 * nb1 + i2 * nb2 + i3 * nb3, 1)[0] = sum as f32;
                }
            }
        }
    }

    fn forward_repeat(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        // ops.cpp:1789-1812 ggml_compute_forward_repeat dispatches by element
        // size: F16/BF16/I16 → repeat_f16, F32/I32 → repeat_f32 — both are the
        // same tile copy of ne00 elements, so branch on the width only
        let es = t0.ty.type_size();
        assert_eq!(td.ty, t0.ty, "repeat: same type only");
        assert!(
            es == 2 || es == 4,
            "repeat: F16/BF16/I16 (2B) and F32/I32 (4B) only"
        );
        assert_eq!(t0.nb[0] as usize, es, "repeat: contiguous rows");

        let (ne00, ne01, ne02, ne03) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (ne0, ne1, ne2, ne3) = (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let row = ne00 * es;

        let nr0 = ne0 / ne00;
        let nr1 = ne1 / ne01;
        let nr2 = ne2 / ne02;
        let nr3 = ne3 / ne03;

        for i3 in 0..nr3 {
            for k3 in 0..ne03 {
                for i2 in 0..nr2 {
                    for k2 in 0..ne02 {
                        for i1 in 0..nr1 {
                            for k1 in 0..ne01 {
                                for i0 in 0..nr0 {
                                    let src = self.rd(s0, k1 * nb01 + k2 * nb02 + k3 * nb03, row);
                                    let off = (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + i0 * row;
                                    let y = self.wr(dst, off, row);
                                    y.copy_from_slice(src);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ==================================================================
    // diag_mask_inf — 对照 ops.cpp:5496 ggml_compute_forward_diag_mask_f32
    // ==================================================================

    fn forward_diag_mask_inf(&self, dst: TensorId, nth: usize, team: &Team<'_>, value: f32) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let n_past = self.t(dst).op_params[0];

        // copy (C does this on thread 0 before the barrier)
        let same = self.loc(s0) == self.loc(dst);
        if !same {
            let total = self.t(dst).n_elements() as usize * 4;
            let src = self.rd(s0, 0, total).to_vec();
            self.wr(dst, 0, total).copy_from_slice(&src);
        }

        let nc = t0.ne[0] as usize;
        let nr = t0.ne[1] as usize;
        let n = t0.nrows() as usize;
        let nz = n / nr;
        let (nb0, nb1, nb2) = (self.t(dst).nb[0] as usize, self.t(dst).nb[1] as usize, self.t(dst).nb[2] as usize);

        let run = |j0: usize, j1: usize| {
            for k in 0..nz {
                for j in j0..j1 {
                    for i in n_past as usize..nc {
                        if i > n_past as usize + j {
                            self.wr_f32s(dst, k * nb2 + j * nb1 + i * nb0, 1)[0] = value;
                        }
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // concat — 对照 ops.cpp:2037 ggml_compute_forward_concat_f32
    // ==================================================================

    /// C's F32 concat is **element-wise** (ops.cpp:2061-2076): element
    /// (i0,i1,i2,i3) comes from src0 when it is in range, else from src1
    /// (offset by src0->ne in `dim`). That is what makes a *strided* source
    /// correct — the granite/lfm2 conv prepend concatenates a raw
    /// `transpose()` view (`concat(state, transpose(xbc), 0)`), whose dim-1
    /// stride is nb[0] == 4 while the dim-0 stride is the token stride; the
    /// previous region-memcpy formulation silently used contiguous rows for
    /// src1 and read the wrong elements whenever the prepended source had more
    /// than one dim-0 element (verified: `conv_prepend_concat_probe` /
    /// `concat_strided_source_probe` in hybrid_e2e.rs — T == 1 happened to be
    /// unaffected, which is why only the prefill of the hybrid archs broke).
    ///
    /// Quantized types keep the row-region path (`ggml_compute_forward_concat_any`,
    /// ops.cpp:1899) — no current graph concatenates quantized tensors.
    fn forward_concat(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);
        let dim = self.t(dst).op_params[0] as usize;

        let mut o = [0usize; 4];
        if dim == 0 {
            o[0] = t0.ne[0] as usize / t0.ty.blck_size();
        } else {
            o[dim] = t0.ne[dim] as usize;
        }

        let (ne00, ne01, ne02, ne03) = (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (ne10, ne11, ne12, ne13) = (t1.ne[0] as usize, t1.ne[1] as usize, t1.ne[2] as usize, t1.ne[3] as usize);
        let (nb00, nb01, nb02, nb03) =
            (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb10, nb11, nb12, nb13) =
            (t1.nb[0] as usize, t1.nb[1] as usize, t1.nb[2] as usize, t1.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        if !t0.ty.is_quantized() {
            // element-wise, C's F32 kernel
            let ts = td.ty.type_size();
            let (ne0, ne1, ne2, ne3) =
                (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
            let run = |i20: usize, i21: usize| {
                for i3 in 0..ne3 {
                    for i2 in i20..i21 {
                        for i1 in 0..ne1 {
                            for i0 in 0..ne0 {
                                let src = if i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03 {
                                    let off = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
                                    self.rd(s0, off, ts)
                                } else {
                                    let off = (i0 - o[0]) * nb10
                                        + (i1 - o[1]) * nb11
                                        + (i2 - o[2]) * nb12
                                        + (i3 - o[3]) * nb13;
                                    self.rd(s1, off, ts)
                                };
                                let off = i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
                                self.wr(dst, off, ts).copy_from_slice(src);
                            }
                        }
                    }
                }
            };
            self.par_rows(ne2, nth, team, run);
            return;
        }

        // quantized fallback: the historical row-region path (concat_any)
        let rs0 = ty_row_bytes(t0.ty, ne00);
        let rs1 = ty_row_bytes(t1.ty, ne10);
        let region = |i3: usize, i2: usize, i1: usize, first: bool| {
            if first {
                let src = self.rd(s0, i1 * nb01 + i2 * nb02 + i3 * nb03, rs0);
                self.wr(dst, i1 * nb1 + i2 * nb2 + i3 * nb3, rs0).copy_from_slice(src);
            } else {
                let src = self.rd(s1, i1 * nb11 + i2 * nb12 + i3 * nb13, rs1);
                let off = (i1 + o[1]) * nb1 + (i2 + o[2]) * nb2 + (i3 + o[3]) * nb3 + o[0] * nb0;
                self.wr(dst, off, rs1).copy_from_slice(src);
            }
        };
        let max_i2 = ne02.max(ne12);
        self.par_rows(max_i2, nth, team, |i20, i21| {
            for i3 in 0..ne03 {
                for i2 in i20..i21.min(ne02) {
                    for i1 in 0..ne01 {
                        region(i3, i2, i1, true);
                    }
                }
            }
            for i3 in 0..ne13 {
                for i2 in i20..i21.min(ne12) {
                    for i1 in 0..ne11 {
                        region(i3, i2, i1, false);
                    }
                }
            }
        });
    }

    // ==================================================================
    // flash_attn_ext — 对照 ggml_compute_forward_flash_attn_ext (ops.cpp:9348)
    // → ..._f16 (ops.cpp:9212) → ..._f16_one_chunk (ops.cpp:8614)
    // ==================================================================

    /// FA rows are fully independent, so `par_rows` sharding reproduces the C
    /// per-row loop exactly — **except** the `use_split_kv_path` branch
    /// (ops.cpp:9245-9291): for a 1-token, 1-sequence decode with `nek1 >= 512`
    /// the C splits the KV range into `nth` chunks and merges the partials
    /// (`ggml_flash_attn_ext_reduce_partials`), and that merged rounding is
    /// *not* the whole-range scan's rounding (the reference's own T=1 vs T>=2
    /// outputs differ there — parity/ref_fa_probe.c). The chunk count = thread
    /// count, so the split run must be issued once with `n_chunks = nth`, not
    /// row-sharded; `split_kv`'s sequential chunk loop is value-identical to
    /// the C's threaded one (per-chunk rows are thread-independent).
    fn forward_flash_attn_ext(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let (q, k, v) = (td.src[0].unwrap(), td.src[1].unwrap(), td.src[2].unwrap());
        let mask = td.src[3];
        let (scale, max_bias, logit_softcap) = (
            f32::from_bits(td.op_params[0] as u32),
            f32::from_bits(td.op_params[1] as u32),
            f32::from_bits(td.op_params[2] as u32),
        );
        // ops.cpp:9349-9357 aborts for any prec other than UNDEFINED/DEFAULT (0)
        // or F32 (10); both select the same F32-accumulator kernel ported here.
        assert!(
            td.op_params[3] == 0 || td.op_params[3] == 10,
            "fattn: unsupported prec {} (C aborts, ops.cpp:9350)",
            td.op_params[3]
        );
        // C addresses the sources through `(char *) data + nb`; KV-cache tensors
        // are views with their own strides, so ne/nb go straight into the kernel.
        let view = |id: TensorId| {
            let t = self.t(id);
            crate::flash_attn::Src::strided(self.rd_rest(id, 0), t.ty, t.ne, t.nb)
        };
        // sinks live in src[4] (ggml_flash_attn_ext_add_sinks); the slice is the
        // raw F32 [n_head] tensor storage (ops.cpp:8626 reads it per-head).
        let sinks_t = self.t(dst).src[4];
        let sinks_buf: Option<&[f32]> = sinks_t.map(|t| {
            let n = self.t(t).n_elements() as usize;
            self.rd_f32s(t, 0, n)
        });
        let fa = crate::flash_attn::FlashAttn {
            q: view(q),
            k: view(k),
            v: view(v),
            mask: mask.map(view),
            scale,
            max_bias,
            logit_softcap,
            sinks: sinks_buf,
        };
        fa.validate();
        if crate::flash_attn::use_split_kv(&fa) {
            // single issue, all chunks + reduce (ops.cpp:9262-9290)
            let dlen = td.n_elements() as usize * td.ty.type_size();
            let mut d = crate::flash_attn::Dst { data: self.wr(dst, 0, dlen), ne: td.ne, nb: td.nb };
            crate::flash_attn::split_kv(&fa, nth, &mut d);
            return;
        }
        let nr = fa.n_rows() as usize;
        let dlen = td.n_elements() as usize * td.ty.type_size();
        let run = |ir0: usize, ir1: usize| {
            let mut d = crate::flash_attn::Dst { data: self.wr(dst, 0, dlen), ne: td.ne, nb: td.nb };
            crate::flash_attn::flash_attn_ext_dispatch(&fa, ir0, ir1, &mut d);
        };
        self.par_rows(nr, nth, team, run);
    }

    /// ggml_compute_forward_clamp_f32 (ops.cpp:5821): `MAX(MIN(x, max), min)`, one
    /// row per thread iteration (`j = ith; j < n; j += nth`).
    fn forward_clamp(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "clamp: F32 only");
        assert_eq!(td.nb[0], 4, "clamp: nb0 == sizeof(float)");
        assert_eq!(t0.nb[0], 4, "clamp: src0 nb0 == sizeof(float)");
        let min = f32::from_bits(td.op_params[0] as u32);
        let max = f32::from_bits(td.op_params[1] as u32);
        let nc = t0.ne[0] as usize;
        let n = t0.nrows() as usize;
        let nb01 = t0.nb[1] as usize;
        let nb1 = td.nb[1] as usize;
        let run = |ir0: usize, ir1: usize| {
            for j in ir0..ir1 {
                for i in 0..nc {
                    let x = self.rd_f32s(s0, j * nb01 + i * 4, 1)[0];
                    self.wr_f32s(dst, j * nb1 + i * 4, 1)[0] = x.min(max).max(min);
                }
            }
        };
        self.par_rows(n, nth, team, run);
    }

    // ==================================================================
    // vision: im2col / upscale(interpolate) — 对照 ops.cpp:6871 / ops.cpp:7978
    // ==================================================================

    /// ggml_compute_forward_im2col (ops.cpp:6986) → ..._f16 (ops.cpp:6907) /
    /// ..._f32 (ops.cpp:6871). `src0` = kernel [KW, KH, IC, OC], `src1` = image
    /// [IW, IH, IC, N], dst = patches [IC*KH*KW, OW, OH, N].
    ///
    /// The F16 dst of `ggml_conv_2d` takes the ..._f16 kernel — the float source
    /// is rounded once per patch element (the only rounding of the patch-embed
    /// input; the following mul_mat then runs the F16 vec_dot over it).
    /// Thread split is C's: `for (iic = ith; iic < IC; iic += nth)`.
    fn forward_im2col(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap(); // kernel
        let s1 = self.t(dst).src[1].unwrap(); // image
        let (t0, t1, td) = (self.t(s0), self.t(s1), self.t(dst));
        let p = td.op_params;
        let (s0_, s1_, p0, p1, d0, d1) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let is_2d = p[6] == 1;

        let (ne10, ne11, ne12, ne13) = (t1.ne[0], t1.ne[1], t1.ne[2], t1.ne[3]);
        let (ne00, ne01) = (t0.ne[0], t0.ne[1]);
        let n = if is_2d { ne13 } else { ne12 };
        let ic = if is_2d { ne12 } else { ne11 };
        let ih = if is_2d { ne11 } else { 1 };
        let iw = ne10;
        let kh = if is_2d { ne01 } else { 1 };
        let kw = ne00;
        let oh = if is_2d { td.ne[2] } else { 1 };
        let ow = td.ne[1];
        assert_eq!(t1.ty, GgmlType::F32, "im2col: image must be F32");
        assert!(
            matches!(td.ty, GgmlType::F16 | GgmlType::F32),
            "im2col: dst must be F16/F32"
        );
        assert_eq!(t1.nb[0], 4, "im2col: nb10 == sizeof(float)");

        let (nb11, nb12, nb13) = (t1.nb[1] as usize, t1.nb[2] as usize, t1.nb[3] as usize);
        // C's `ofs0/ofs1` are the per-N and per-channel strides (nb13/nb12 for 2D)
        let (ofs0, ofs1) = if is_2d { (nb13, nb12) } else { (nb12, nb11) };
        let dst16 = td.ty == GgmlType::F16;
        assert_eq!(td.nb[0], if dst16 { 2 } else { 4 }, "im2col: dst nb0");

        let run = |ic0: i64, ic1: i64| {
            for in_ in 0..n {
                for ioh in 0..oh {
                    for iow in 0..ow {
                        for iic in ic0..ic1 {
                            // micro kernel: dst[(in*OH + ioh)*OW + iow][iic][kh][kw]
                            let base = (((in_ * oh + ioh) * ow + iow) * (ic * kh * kw)
                                + iic * (kh * kw)) as usize;
                            let src_off = in_ as usize * ofs0 + iic as usize * ofs1;
                            for ikh in 0..kh {
                                for ikw in 0..kw {
                                    let iiw = iow * s0_ as i64 + ikw * d0 as i64 - p0 as i64;
                                    let iih = ioh * s1_ as i64 + ikh * d1 as i64 - p1 as i64;
                                    let idx = base + (ikh * kw + ikw) as usize;
                                    let v = if iih < 0 || iih >= ih || iiw < 0 || iiw >= iw {
                                        0.0f32
                                    } else {
                                        self.rd_f32s(
                                            s1,
                                            src_off + (iih * iw + iiw) as usize * 4,
                                            1,
                                        )[0]
                                    };
                                    if dst16 {
                                        self.wr(dst, idx * 2, 2)
                                            .copy_from_slice(&f16::from_f32(v).to_le_bytes());
                                    } else {
                                        self.wr_f32s(dst, idx * 4, 1)[0] = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        self.par_rows(ic as usize, nth, team, move |a, b| run(a as i64, b as i64));
    }

    /// ggml_compute_forward_upscale_f32 (ops.cpp:7978) — the kernel behind
    /// `ggml_interpolate`. `mode_flags` = op_params[0]; the low byte selects
    /// nearest / bilinear / bicubic, GGML_SCALE_FLAG_ALIGN_CORNERS switches the
    /// sample grid to the corner-aligned one (ne-1)/(ne0-1) with no half-pixel
    /// offset — the mode the vision position-embedding resize uses.
    fn forward_upscale(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "upscale: F32 only");

        let (ne00, ne01, ne02, ne03) =
            (t0.ne[0] as usize, t0.ne[1] as usize, t0.ne[2] as usize, t0.ne[3] as usize);
        let (ne0, ne1, ne2, ne3) = (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
        let (nb00, nb01, nb02, nb03) = (t0.nb[0] as usize, t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb0, nb1, nb2, nb3) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        // C shards over i2 and loops i3 outermost; every node this port builds
        // is 3D (ggml_interpolate over the (w, h, n_embd) position embedding),
        // so i3 == 0 is the only reachable slice.
        assert_eq!(ne3, 1, "upscale: only 3d inputs are reachable here");

        let mode_flags = td.op_params[0] as u32;
        let mode = mode_flags & 0xFF;
        let mut sf0 = ne0 as f32 / ne00 as f32;
        let mut sf1 = ne1 as f32 / ne01 as f32;
        let sf2 = ne2 as f32 / ne02 as f32;
        let sf3 = ne3 as f32 / ne03 as f32;
        let mut pixel_offset = 0.5f32;
        if mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS != 0 {
            pixel_offset = 0.0f32;
            sf0 = if ne0 > 1 && ne00 > 1 { (ne0 - 1) as f32 / (ne00 - 1) as f32 } else { sf0 };
            sf1 = if ne1 > 1 && ne01 > 1 { (ne1 - 1) as f32 / (ne01 - 1) as f32 } else { sf1 };
        }
        let src_at = |i00: usize, i01: usize, i02: usize, i03: usize| -> f32 {
            self.rd_f32s(s0, i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03, 1)[0]
        };

        let run = |i2: usize, i2_end: usize| {
            for i2 in i2..i2_end {
                let i02 = (i2 as f32 / sf2) as usize;
                // the source's dim-3 index: i3 == 0 because ne3 == 1 (asserted
                // above) — the C's `i03 = i3 / sf3` (ops.cpp:8005)
                let i03 = 0usize;
                let _ = sf3;
                for i1 in 0..ne1 {
                    for i0 in 0..ne0 {
                        let y = ((i1 as f32 + pixel_offset) / sf1 - pixel_offset) as f32;
                        let x = ((i0 as f32 + pixel_offset) / sf0 - pixel_offset) as f32;
                        let v = if mode == GGML_SCALE_MODE_NEAREST {
                            src_at((i0 as f32 / sf0) as usize, (i1 as f32 / sf1) as usize, i02, i03)
                        } else if mode == GGML_SCALE_MODE_BILINEAR {
                            let y0f = y.floor();
                            let x0f = x.floor();
                            let mut y0 = y0f as i64;
                            let mut y1 = y0 + 1;
                            y0 = y0.clamp(0, ne01 as i64 - 1);
                            y1 = y1.clamp(0, ne01 as i64 - 1);
                            let dy = (y - y0f).clamp(0.0, 1.0);
                            let mut x0 = x0f as i64;
                            let mut x1 = x0 + 1;
                            x0 = x0.clamp(0, ne00 as i64 - 1);
                            x1 = x1.clamp(0, ne00 as i64 - 1);
                            let dx = (x - x0f).clamp(0.0, 1.0);
                            // i3 == 0 (asserted above), C ops.cpp:8005 `i03 = i3 / sf3`
                            let a = src_at(x0 as usize, y0 as usize, i02, i03);
                            let b = src_at(x1 as usize, y0 as usize, i02, i03);
                            let c = src_at(x0 as usize, y1 as usize, i02, i03);
                            let d = src_at(x1 as usize, y1 as usize, i02, i03);
                            a * (1.0 - dx) * (1.0 - dy) + b * dx * (1.0 - dy) + c * (1.0 - dx) * dy + d * dx * dy
                        } else if mode == GGML_SCALE_MODE_BICUBIC {
                            // a = -0.75 (PyTorch's alpha, ops.cpp:8061)
                            const A: f32 = -0.75;
                            let w1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
                            let w2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
                            let bicubic = |p0: f32, p1: f32, p2: f32, p3: f32, x: f32| {
                                p0 * w2(x + 1.0) + p1 * w1(x) + p2 * w1(1.0 - x) + p3 * w2(2.0 - x)
                            };
                            let y0 = y.floor();
                            let dy = y - y0;
                            let x0 = x.floor();
                            let dx = x - x0;
                            // i3 == 0 (asserted above), C ops.cpp:8005 `i03 = i3 / sf3`
                            let p = |xo: i64, yo: i64| -> f32 {
                                let i00 = (x0 as i64 + xo).clamp(0, ne00 as i64 - 1) as usize;
                                let i01 = (y0 as i64 + yo).clamp(0, ne01 as i64 - 1) as usize;
                                src_at(i00, i01, i02, i03)
                            };
                            bicubic(
                                bicubic(p(-1, -1), p(0, -1), p(1, -1), p(2, -1), dx),
                                bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                                bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                                bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx),
                                dy,
                            )
                        } else {
                            unimplemented!("upscale: unsupported mode {mode}");
                        };
                        // i3 == 0: the C writes `i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3` (ops.cpp:8015)
                        self.wr_f32s(dst, i0 * nb0 + i1 * nb1 + i2 * nb2, 1)[0] = v;
                    }
                }
            }
        };
        self.par_rows(ne2, nth, team, run);
    }

    // ==================================================================
    // ssm_conv / ssm_scan — 对照 ops.cpp:9701 / ops.cpp:9771
    // ==================================================================

    /// ggml_compute_forward_ssm_conv (ops.cpp:9756) → ..._f32 (ops.cpp:9703).
    /// Row sharding is C's (`dr = (nr + nth - 1)/nth` over d_inner rows).
    fn forward_ssm_conv(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let s1 = self.t(dst).src[1].unwrap();
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);

        assert_eq!(td.ty, GgmlType::F32, "ssm_conv: F32 only");
        let nc = t1.ne[0] as usize; // d_conv
        let ncs = t0.ne[0] as usize; // d_conv - 1 + n_t
        let nr = t0.ne[1] as usize; // d_inner
        let n_t = td.ne[1] as usize;
        let n_s = td.ne[2] as usize;
        assert_eq!(td.ne[0] as usize, nr, "ssm_conv: dst->ne[0] == nr");
        assert_eq!(t0.nb[0], 4, "ssm_conv: src0->nb[0] == sizeof(float)");
        assert_eq!(t1.nb[0], 4, "ssm_conv: src1->nb[0] == sizeof(float)");
        assert_eq!(t0.nb[1], t0.ne[0] as u64 * 4, "ssm_conv: src0->nb[1]");

        let dst_nb = td.nb;
        let sx_nb = t0.nb;
        let c_nb = t1.nb;
        // raw addresses (usize keeps the closure Send+Sync, like the other kernels)
        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        let sx_p = self.rd(s0, 0, 0).as_ptr() as usize;
        let c_p = self.rd(s1, 0, 0).as_ptr() as usize;

        let run = |ir0: usize, ir1: usize| unsafe {
            crate::ssm::ssm_conv_f32(
                dst_p as *mut u8,
                dst_nb,
                sx_p as *const u8,
                sx_nb,
                c_p as *const u8,
                c_nb,
                ir0,
                ir1,
                nc,
                ncs,
                nr,
                n_t,
                n_s,
            )
        };
        self.par_rows(nr, nth, team, run);
    }

    /// ggml_compute_forward_ssm_scan (ops.cpp:9996) → ..._f32 (ops.cpp:9773).
    /// Head sharding is C's (`dh = (nh + nth - 1)/nth`).
    fn forward_ssm_scan(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let ids = td.src[6].expect("ssm_scan: src[6] (ids)");
        let (s, x, dt, a, b, c) = (
            td.src[0].unwrap(),
            td.src[1].unwrap(),
            td.src[2].unwrap(),
            td.src[3].unwrap(),
            td.src[4].unwrap(),
            td.src[5].unwrap(),
        );
        let (ts, tx, tdt, ta, tb, tc) =
            (self.t(s), self.t(x), self.t(dt), self.t(a), self.t(b), self.t(c));
        for (t, name) in [(ts, "s"), (tx, "x"), (tdt, "dt"), (ta, "A"), (tb, "B"), (tc, "C")] {
            assert_eq!(t.nb[0], 4, "ssm_scan: {name}->nb[0] == sizeof(float)");
        }
        assert_eq!(self.t(ids).ty, GgmlType::I32, "ssm_scan: ids must be I32");

        let nc = ts.ne[0] as usize; // d_state
        let nr = ts.ne[1] as usize; // dim
        let nh = tx.ne[1] as usize; // n_head
        let ng = tb.ne[1] as usize;
        let nt = tx.ne[2] as usize;
        let ns = tx.ne[3] as usize;
        let k = td.op_params[0] as usize;
        assert!(k >= 1, "ssm_scan: K >= 1");
        assert_eq!(
            tx.n_elements() as usize + k * nc * nr * nh * ns,
            td.n_elements() as usize,
            "ssm_scan: dst is y + K states"
        );
        assert_eq!(nh % ng, 0, "ssm_scan: nh % ng");

        let dst_p = self.wr(dst, 0, 0).as_mut_ptr() as usize;
        let s_p = self.rd(s, 0, 0).as_ptr() as usize;
        let x_p = self.rd(x, 0, 0).as_ptr() as usize;
        let dt_p = self.rd(dt, 0, 0).as_ptr() as usize;
        let a_p = self.rd(a, 0, 0).as_ptr() as usize;
        let b_p = self.rd(b, 0, 0).as_ptr() as usize;
        let c_p = self.rd(c, 0, 0).as_ptr() as usize;
        let ids_p = self.rd(ids, 0, 0).as_ptr() as usize;

        let (s_nb3, x_nb, dt_nb, b_nb, c_nb, a_ne0) =
            (ts.nb[3], tx.nb, tdt.nb, tb.nb, tc.nb, ta.ne[0] as usize);

        let run = |ih0: usize, ih1: usize| unsafe {
            crate::ssm::ssm_scan_f32(
                dst_p as *mut u8,
                s_p as *const u8,
                s_nb3,
                x_p as *const u8,
                x_nb,
                dt_p as *const u8,
                dt_nb,
                a_p as *const u8,
                a_ne0,
                b_p as *const u8,
                b_nb,
                c_p as *const u8,
                c_nb,
                ids_p as *const i32,
                ih0,
                ih1,
                nc,
                nr,
                nh,
                ng,
                nt,
                ns,
                k,
            )
        };
        self.par_rows(nh, nth, team, run);
    }

    /// Test for "the whole tensor is a flat array of plain MXFP4 rows"
    /// (nb0 = 17 B block, nb[i>0] = the product of the lower dims) — what
    /// repack.rs's 8-row grouping and the reference's CPU_REPACK buffer assume.
    fn is_plain_mxfp4_rows(&self, id: TensorId) -> bool {
        let t = self.t(id);
        if t.ty != GgmlType::Mxfp4 || t.nb[0] as usize != t.ty.type_size() {
            return false;
        }
        let row =
            t.ne[0] as usize / crate::blocks::QK_MXFP4 * crate::repack::BLOCK_MXFP4_SIZE;
        let mut want = row as u64;
        for i in 1..MAX_DIMS {
            if t.ne[i] == 1 {
                continue;
            }
            if t.nb[i] != want {
                return false;
            }
            want *= t.ne[i] as u64;
        }
        true
    }

    /// CPU_REPACK (repack.rs) for MXFP4 weight tensors: return the lazily built
    /// and cached 8x8 interleaved copy the reference keeps in its CPU_REPACK
    /// buffer (`ggml_backend_cpu_repack_buffer_type`), or `None` when the path is
    /// disabled (`LLAMA_RUST_REPACK=0`), the geometry is not repackable
    /// (repack.cpp:5093: `ne[1] % 8 == 0`), the tensor is not a plain row array,
    /// or the cache budget is exhausted — callers then fall back to the row-wise
    /// `vec_dot_mxfp4_q8_0` (the pre-repack port behaviour).
    fn repacked_mxfp4(&self, id: TensorId) -> Option<std::sync::Arc<Vec<u8>>> {
        if !crate::repack::repack_enabled() || !self.is_plain_mxfp4_rows(id) {
            return None;
        }
        let t = self.t(id);
        // `rd_rest` runs to the end of the *storage* (arena tail or mmap), so
        // cut it down to this tensor's own rows (nrows * nb * 17 bytes).
        let row = t.ne[0] as usize / crate::blocks::QK_MXFP4 * crate::repack::BLOCK_MXFP4_SIZE;
        let nrows = (t.ne[1] * t.ne[2] * t.ne[3]) as usize;
        let rest = self.rd_rest(id, 0);
        assert!(rest.len() >= nrows * row, "repacked_mxfp4: storage shorter than the tensor");
        let src = &rest[..nrows * row];
        let ptr = self.rd(id, 0, 0).as_ptr() as usize;
        crate::repack::repack_mxfp4_8x8_cached(ptr, src, t.ne[1], t.ne[0])
    }

    /// [`is_plain_mxfp4_rows`] for Q4_K (`nb0 = 144 B block`, qs at byte 16).
    fn is_plain_q4k_rows(&self, id: TensorId) -> bool {
        let t = self.t(id);
        if t.ty != GgmlType::Q4K || t.nb[0] as usize != t.ty.type_size() {
            return false;
        }
        let row = t.ne[0] as usize / crate::blocks::QK_K * crate::repack::BLOCK_Q4_K_SIZE;
        let mut want = row as u64;
        for i in 1..MAX_DIMS {
            if t.ne[i] == 1 {
                continue;
            }
            if t.nb[i] != want {
                return false;
            }
            want *= t.ne[i] as u64;
        }
        true
    }

    /// CPU_REPACK (repack.rs) for Q4_K weight tensors: the lazily built 8x8 copy
    /// the reference keeps in its CPU_REPACK buffer
    /// (`ggml_backend_cpu_repack_buffer_type` + `repack_q4_K_to_q4_K_8_bl`), or
    /// `None` when the path is disabled (`LLAMA_RUST_REPACK=0`), the geometry is
    /// not repackable (repack.cpp:5006-5011: `ne[1] % 8 == 0`), the tensor is not
    /// a plain row array, or the budget is exhausted — callers then keep the
    /// row-wise `vec_dot_q4_K_q8_K` (the reference's *non*-repack path).
    fn repacked_q4_k(&self, id: TensorId) -> Option<std::sync::Arc<Vec<u8>>> {
        if !crate::repack::repack_enabled() || !self.is_plain_q4k_rows(id) {
            return None;
        }
        let t = self.t(id);
        let row = t.ne[0] as usize / crate::blocks::QK_K * crate::repack::BLOCK_Q4_K_SIZE;
        let nrows = (t.ne[1] * t.ne[2] * t.ne[3]) as usize;
        let rest = self.rd_rest(id, 0);
        assert!(rest.len() >= nrows * row, "repacked_q4_k: storage shorter than the tensor");
        let src = &rest[..nrows * row];
        let ptr = self.rd(id, 0, 0).as_ptr() as usize;
        crate::repack::repack_q4_K_8x8_cached(ptr, src, t.ne[1], t.ne[0])
    }

    /// [`is_plain_q4k_rows`] for Q4_0 (`nb0 = 18 B block`). `require_2d`
    /// mirrors the repack traits' `supports_op` gates (repack.cpp:5187-5210):
    /// `ggml_n_dims(src0) == 2` for MUL_MAT, `== 3` for MUL_MAT_ID.
    fn is_plain_q4_0_rows(&self, id: TensorId, require_2d: bool) -> bool {
        let t = self.t(id);
        if t.ty != GgmlType::Q4_0 || t.nb[0] as usize != t.ty.type_size() {
            return false;
        }
        if require_2d && (t.ne[2] != 1 || t.ne[3] != 1) {
            return false;
        }
        if !require_2d && (t.ne[3] != 1 || t.ne[2] == 1) {
            return false;
        }
        let row = t.ne[0] as usize / crate::blocks::QK4_0 * crate::repack::BLOCK_Q4_0_SIZE;
        let mut want = row as u64;
        for i in 1..MAX_DIMS {
            if t.ne[i] == 1 {
                continue;
            }
            if t.nb[i] != want {
                return false;
            }
            want *= t.ne[i] as u64;
        }
        true
    }

    /// CPU_REPACK (repack.rs) for Q4_0 weight tensors: the lazily built 8x8
    /// copy the reference keeps in its CPU_REPACK buffer
    /// (`ggml_backend_cpu_repack_buffer_type` + `repack_q4_0_to_q4_0_8_bl`),
    /// or `None` under the same conditions as [`Self::repacked_q4_k`] — plus
    /// the `ggml_n_dims` gate the traits' `supports_op` applies per op kind.
    fn repacked_q4_0(&self, id: TensorId, require_2d: bool) -> Option<std::sync::Arc<Vec<u8>>> {
        if !crate::repack::repack_enabled() || !self.is_plain_q4_0_rows(id, require_2d) {
            return None;
        }
        let t = self.t(id);
        let row = t.ne[0] as usize / crate::blocks::QK4_0 * crate::repack::BLOCK_Q4_0_SIZE;
        let nrows = (t.ne[1] * t.ne[2] * t.ne[3]) as usize;
        let rest = self.rd_rest(id, 0);
        assert!(rest.len() >= nrows * row, "repacked_q4_0: storage shorter than the tensor");
        let src = &rest[..nrows * row];
        let ptr = self.rd(id, 0, 0).as_ptr() as usize;
        crate::repack::repack_q4_0_8x8_cached(ptr, src, t.ne[1], t.ne[0])
    }

    /// The reference's repack `forward_mul_mat` (repack.cpp:4650-4781) +
    /// `forward_mul_mat_one_chunk` (:4600-4648) for Q4_0 — the exact Q4_K
    /// structure with PARAM_TYPE = Q8_0: per `(i12)` plane the activation rows
    /// are quantized into the C's wdata layout (4-row groups as `block_q8_0x4`
    /// tiles via `ggml_quantize_mat_q8_0_4x8`, the `ne11 % 4` tail as plain
    /// `block_q8_0` rows), the 4-row groups go through `gemm_q4_0_8x8_q8_0`
    /// and the tail through `gemv_q4_0_8x8_q8_0` (repack.cpp:4638-4647).
    ///
    /// The work split (src0 column chunks aligned to NB_COLS = 8, full planes)
    /// matches the C's `nchunk0` chunking; it only decides which thread
    /// computes which output element, never how.
    #[allow(clippy::too_many_arguments)]
    fn mul_mat_q4_0_repack(
        &self,
        dst: TensorId,
        s0: TensorId,
        s1: TensorId,
        rep: &[u8],
        nth: usize,
        team: &Team<'_>,
    ) {
        use crate::repack;
        use bytemuck::Zeroable;
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);
        let n = t0.ne[0] as usize; // ne00: k in elements
        let nc = t0.ne[1] as usize; // ne01: weight rows
        let ne11 = t1.ne[1] as usize; // activation rows (ne11)
        let nb = n / crate::blocks::QK4_0;
        let rs = nb * repack::BLOCK_Q8_0_SIZE; // one plain q8_0 row, 34 B/block
        let (nb11, nb12) = (t1.nb[1] as usize, t1.nb[2] as usize);
        let (nb1, nb2) = (td.nb[1] as usize, td.nb[2] as usize);
        assert_eq!(nc % 8, 0, "mul_mat_q4_0_repack: ne01 % 8 != 0");
        assert_eq!(nb1 % 4, 0, "mul_mat_q4_0_repack: dst nb1 % 4 != 0");
        // C asserts ne03 == ne13 == ne3 == 1 (repack.cpp:4672-4674)
        let planes = td.ne[2] as usize;
        assert_eq!(td.ne[3], 1, "mul_mat_q4_0_repack: ne3 != 1");
        assert_eq!(t0.ne[2], 1, "mul_mat_q4_0_repack: ne02 != 1");

        // ---- wdata: the C's per-plane blocks (repack.cpp:4697-4705) ----
        // A 4-row `block_q8_0x4` tile is exactly 4 plain q8_0 rows in size
        // (136 B), so the buffer is `ne11 * rs` bytes per plane with the tiles
        // written at `i11 * rs` — the addressing one_chunk relies on. The
        // `ne11 % 4` tail rows keep the plain `quantize_row_q8_0`
        // (`from_float` in the C). Work split: the C strides the 4-row groups
        // over the threads inside every plane (`i11 = ith*4; i11 += nth*4`,
        // repack.cpp:4703).
        let mut wd = vec![0u8; planes * ne11 * rs];
        {
            let wd_ptr = std::sync::atomic::AtomicPtr::new(wd.as_mut_ptr());
            let groups = planes * (ne11 / 4 + usize::from(ne11 % 4 != 0));
            self.par_rows(groups, nth, team, |g0, g1| {
                let mut rows: Vec<f32> = vec![0.0; 4 * n];
                let wd = wd_ptr.load(std::sync::atomic::Ordering::Relaxed);
                for g in g0..g1 {
                    let i12 = g / (ne11 / 4 + usize::from(ne11 % 4 != 0));
                    let i11 = (g % (ne11 / 4 + usize::from(ne11 % 4 != 0))) * 4;
                    let base = i12 * ne11 * rs;
                    if i11 + 4 <= ne11 {
                        let out =
                            unsafe { std::slice::from_raw_parts_mut(wd.add(base + i11 * rs), 4 * rs) };
                        for r in 0..4 {
                            let src = self.rd_f32s(s1, i12 * nb12 + (i11 + r) * nb11, n);
                            rows[r * n..(r + 1) * n].copy_from_slice(src);
                        }
                        repack::quantize_mat_q8_0_4x8(&rows, n, 4, out);
                    } else {
                        // the `ne11 % 4` tail: plain rows, one group of <4
                        let out = unsafe {
                            std::slice::from_raw_parts_mut(
                                wd.add(base + i11 * rs),
                                (ne11 - i11) * rs,
                            )
                        };
                        for (r, o) in (i11..ne11).zip(out.chunks_mut(rs)) {
                            let row = self.rd_f32s(s1, i12 * nb12 + r * nb11, n);
                            let mut blocks =
                                vec![crate::blocks::BlockQ8_0::zeroed(); nb];
                            quants::quantize_row_q8_0(row, &mut blocks);
                            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
                            o[..bytes.len()].copy_from_slice(bytes);
                        }
                    }
                }
            });
        }

        // ---- kernels: chunk the weight rows 8-aligned (NB_COLS), all
        //      activation rows of the chunk's plane in one gemm, then one gemv
        //      per `ne11 % 4` tail row (repack.cpp:4638-4647) ----
        // Chunk grain: the C makes `nth*4` chunks (`nth_scaled`, :4713) and
        // keeps every boundary a multiple of NB_COLS (:4730-4739) — same grain
        // as the Q4_K wiring above.
        let chunk_rows = (nc.div_ceil(nth * 4)).div_ceil(8) * 8;
        let nchunk0 = nc.div_ceil(chunk_rows);
        let jobs = nchunk0 * planes;
        let bs = nb1 / 4; // dst activation-column stride in floats
        let run = |lo: usize, hi: usize| {
            for job in lo..hi {
                let c0 = (job % nchunk0) * chunk_rows;
                let c1 = (c0 + chunk_rows).min(nc);
                let i12 = job / nchunk0;
                let ncols = c1 - c0;
                let a = &wd[i12 * ne11 * rs..];
                // the 8-column groups of weight rows c0..c1 (src0 is 2D, so
                // every plane reads weight plane 0; the repacked copy keeps
                // the byte size, `nc * nb * 18` bytes)
                let bmat = &rep[c0 / 8 * nb * repack::BLOCK_Q4_0X8_SIZE..];
                let dbase = i12 * (nb2 / 4) + c0;
                let nr4 = ne11 - ne11 % 4;
                if nr4 > 0 {
                    // exact store bound of the gemm: the last row's store ends
                    // at `(nr4-1)*bs + ncols` floats — one row less than
                    // `nr4*bs` (same slice-size fix as the Q4_K gemm).
                    let d = self.wr_f32s(dst, dbase * 4, (nr4 - 1) * bs + ncols);
                    repack::gemm_q4_0_8x8_q8_0(n, d, bs, bmat, a, nr4, ncols);
                }
                for r in nr4..ne11 {
                    let d = self.wr_f32s(dst, dbase * 4 + r * nb1, ncols);
                    repack::gemv_q4_0_8x8_q8_0(n, d, bmat, &a[r * rs..], ncols);
                }
            }
        };
        if jobs == 1 {
            run(0, 1);
        } else {
            team.run(jobs, &run);
        }
    }

    /// The reference's repack `forward_mul_mat` (repack.cpp:4650-4781) +
    /// `forward_mul_mat_one_chunk` (:4600-4648) for Q4_K: per `(i12, i13)` plane
    /// the activation rows are quantized into the C's wdata layout (4-row groups
    /// as `block_q8_Kx4` tiles via `ggml_quantize_mat_q8_K_4x8`, the `ne11 % 4`
    /// tail as plain `block_q8_K` rows), the 4-row groups go through
    /// `gemm_q4_K_8x8_q8_K` and the tail through `gemv_q4_K_8x8_q8_K` — the same
    /// mixture the C dispatches on `nrows > 3` (repack.cpp:4638).
    ///
    /// The work split (src0 column chunks aligned to NB_COLS = 8, full planes)
    /// matches the C's `nchunk0` chunking; it only decides which thread computes
    /// which output element, never how.
    fn mul_mat_q4k_repack(
        &self,
        dst: TensorId,
        s0: TensorId,
        s1: TensorId,
        rep: &[u8],
        nth: usize,
        team: &Team<'_>,
    ) {
        use crate::repack;
        use bytemuck::Zeroable;
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);
        let n = t0.ne[0] as usize; // ne00: k in elements
        let nc = t0.ne[1] as usize; // ne01: weight rows
        let ne11 = t1.ne[1] as usize; // activation rows (ne11)
        let nb = n / crate::blocks::QK_K;
        let rs = nb * repack::BLOCK_Q8_K_SIZE; // one plain q8_K row, 292 B/block
        let (nb11, nb12) = (t1.nb[1] as usize, t1.nb[2] as usize);
        let (nb1, nb2) = (td.nb[1] as usize, td.nb[2] as usize);
        assert_eq!(nc % 8, 0, "mul_mat_q4k_repack: ne01 % 8 != 0");
        assert_eq!(nb1 % 4, 0, "mul_mat_q4k_repack: dst nb1 % 4 != 0");
        assert_eq!(rs, nb * repack::BLOCK_Q8_K_SIZE, "mul_mat_q4k_repack: row size");
        let planes = td.ne[2] as usize; // C asserts ne03 == ne13 == 1 (repack.cpp:4673)
        assert_eq!(td.ne[3], 1, "mul_mat_q4k_repack: ne3 != 1");
        // src0-plane broadcast factor (repack.cpp:4627 `GGML_ASSERT(ne12 %
        // ne02 == 0)` + :4628 `i02 = i12 / r2`): a dst plane i12 reads weight
        // plane i02 — without this, a GQA-style broadcast would use plane 0's
        // weights for every plane.
        let r2 = (planes / t0.ne[2] as usize).max(1);
        debug_assert_eq!(planes % t0.ne[2] as usize, 0, "mul_mat_q4k_repack: ne12 % ne02");

        // ---- wdata: the C's per-plane blocks (repack.cpp:4697-4705) ----
        // A 4-row `block_q8_Kx4` tile is exactly 4 plain q8_K rows in size, so
        // the buffer is `ne11 * rs` bytes per plane with the tiles written at
        // `i11 * rs` — the addressing `forward_mul_mat_one_chunk` relies on.
        // The `ne11 % 4` tail rows keep the plain `quantize_row_q8_K_ref`
        // (`from_float` in the C). Work split: the C strides the 4-row groups
        // over the threads *inside* every plane (`i11 = ith*4; i11 += nth*4`,
        // repack.cpp:4703) — a 1-plane prefill still quantizes on all threads.
        let mut wd = vec![0u8; planes * ne11 * rs];
        {
            // same idiom as the wdata pass above: one AtomicPtr, disjoint rows
            let wd_ptr = std::sync::atomic::AtomicPtr::new(wd.as_mut_ptr());
            let groups = planes * (ne11 / 4 + usize::from(ne11 % 4 != 0));
            self.par_rows(groups, nth, team, |g0, g1| {
                let mut rows: Vec<f32> = vec![0.0; 4 * n];
                let wd = wd_ptr.load(std::sync::atomic::Ordering::Relaxed);
                for g in g0..g1 {
                    let i12 = g / (ne11 / 4 + usize::from(ne11 % 4 != 0));
                    let i11 = (g % (ne11 / 4 + usize::from(ne11 % 4 != 0))) * 4;
                    let base = i12 * ne11 * rs;
                    if i11 + 4 <= ne11 {
                        let out =
                            unsafe { std::slice::from_raw_parts_mut(wd.add(base + i11 * rs), 4 * rs) };
                        for r in 0..4 {
                            let src = self.rd_f32s(s1, i12 * nb12 + (i11 + r) * nb11, n);
                            rows[r * n..(r + 1) * n].copy_from_slice(src);
                        }
                        repack::quantize_mat_q8_K_4x8(&rows, n, 4, out);
                    } else {
                        // the `ne11 % 4` tail: plain rows, one group of <4
                        let out = unsafe {
                            std::slice::from_raw_parts_mut(
                                wd.add(base + i11 * rs),
                                (ne11 - i11) * rs,
                            )
                        };
                        for (r, o) in (i11..ne11).zip(out.chunks_mut(rs)) {
                            let row = self.rd_f32s(s1, i12 * nb12 + r * nb11, n);
                            let mut blocks = vec![crate::blocks::BlockQ8K::zeroed(); nb];
                            quants::quantize_row_q8_K_ref(row, &mut blocks);
                            let bytes: &[u8] = bytemuck::cast_slice(&blocks);
                            o[..bytes.len()].copy_from_slice(bytes);
                        }
                    }
                }
            });
        }

        // ---- kernels: chunk the weight rows 8-aligned (NB_COLS), all
        //      activation rows of the chunk's plane in one gemm (the C's
        //      `src1_start..src1_end = ith1*ne11 .. (ith1+1)*ne11`), then one
        //      gemv per `ne11 % 4` tail row (repack.cpp:4638-4647) ----
        // Chunk grain: the C makes `nth*4` chunks (`nth_scaled`, repack.cpp:4713)
        // and keeps every boundary a multiple of NB_COLS so the 8-row groups
        // never straddle chunks (:4730-4739) — a fixed 128-row grain would give
        // the 896-row qwen ffn_down tensors only 7 jobs on 8 threads.
        let chunk_rows = (nc.div_ceil(nth * 4)).div_ceil(8) * 8;
        let nchunk0 = nc.div_ceil(chunk_rows);
        let jobs = nchunk0 * planes;
        let bs = nb1 / 4; // dst activation-column stride in floats
        let run = |lo: usize, hi: usize| {
            for job in lo..hi {
                let c0 = (job % nchunk0) * chunk_rows;
                let c1 = (c0 + chunk_rows).min(nc);
                let i12 = job / nchunk0;
                let ncols = c1 - c0;
                let a = &wd[i12 * ne11 * rs..];
                // the 8-column groups of weight rows c0..c1 of weight plane
                // `i02 = i12 / r2` (repack.cpp:4622-4628); the repacked copy
                // keeps the byte size, so a plane is `nc * nb * 144` bytes.
                let i02 = i12 / r2;
                let bmat = &rep[i02 * nc * nb * repack::BLOCK_Q4_K_SIZE + c0 / 8 * nb * repack::BLOCK_Q4_KX8_SIZE..];
                let dbase = i12 * (nb2 / 4) + c0;
                let nr4 = ne11 - ne11 % 4;
                if nr4 > 0 {
                    // exact store bound of the gemm: the last row's store ends
                    // at `(nr4-1)*bs + ncols` floats (`s + ((y*4+i)*bs + x*8)`,
                    // arch/x86/repack.cpp:873) — one row less than `nr4*bs`.
                    let d = self.wr_f32s(dst, dbase * 4, (nr4 - 1) * bs + ncols);
                    repack::gemm_q4_K_8x8_q8_K(n, d, bs, bmat, a, nr4, ncols);
                }
                for r in nr4..ne11 {
                    let d = self.wr_f32s(dst, dbase * 4 + r * nb1, ncols);
                    repack::gemv_q4_K_8x8_q8_K(n, d, bmat, &a[r * rs..], ncols);
                }
            }
        };
        if jobs == 1 {
            run(0, 1);
        } else {
            team.run(jobs, &run);
        }
    }

    // ==================================================================
    // fill / lightning indexer — 对照 ops.cpp fill (via ggml_compute_forward_fill)
    // 与 ops.cpp:12124 ggml_compute_forward_lightning_indexer (DSA, batch 6)
    // ==================================================================

    /// ggml_compute_forward_fill_f32/_f16 (ops.cpp): every row of dst gets the
    /// constant `c` (op_params[0] f32 bits).
    fn forward_fill(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        let c = f32::from_bits(td.op_params[0] as u32);
        let (ne0, ne1, ne2, ne3) =
            (td.ne[0] as usize, td.ne[1] as usize, td.ne[2] as usize, td.ne[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let esz = td.ty.type_size();
        let nr = ne1 * ne2 * ne3;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let off = i3 * nb3 + i2 * nb2 + i1 * nb1;
                let row = self.wr(dst, off, ne0 * esz);
                match td.ty {
                    GgmlType::F32 => {
                        let p: &mut [f32] = bytemuck::cast_slice_mut(row);
                        p.fill(c);
                    }
                    GgmlType::F16 => {
                        let p: &mut [half::f16] = bytemuck::cast_slice_mut(row);
                        p.fill(half::f16::from_f32(c));
                    }
                    other => unimplemented!("fill for {other:?}"),
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_fwht_impl<float>` (ops.cpp:12027-12101): the
    /// fast Walsh-Hadamard transform behind the SRC0_IS_HADAMARD mul_mat
    /// hint. Per row: scale = 1/sqrt(n), x[j] = src[j]*scale, then butterfly
    /// passes len = 1, 2, 4, ... n/2 (the AVX lanes process 16 j's in
    /// lockstep but each lane is an independent u±v, so the plain loop is
    /// bit-identical). Threads split the rows (ops.cpp:12044-12047).
    fn forward_fwht(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s1 = self.t(dst).src[1].unwrap();
        let t1 = self.t(s1);
        let td = self.t(dst);
        assert_eq!(td.ty, GgmlType::F32, "fwht: dst F32");
        assert!(matches!(t1.ty, GgmlType::F32 | GgmlType::F16), "fwht: src1 F32/F16");

        let (ne10, ne11, ne12, ne13) = (t1.ne[0], t1.ne[1], t1.ne[2], t1.ne[3]);
        let (nb11, nb12, nb13) = (t1.nb[1] as usize, t1.nb[2] as usize, t1.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);

        let n = ne10;
        assert!(n & (n - 1) == 0, "fwht: n must be a power of 2");

        let nr = (ne11 * ne12 * ne13) as usize;
        let scale = 1.0f32 / (n as f32).sqrt();

        let run = |ir0: usize, ir1: usize| {
            for r in ir0..ir1 {
                let i13 = r / (ne11 * ne12) as usize;
                let i12 = (r - i13 * (ne11 * ne12) as usize) / ne11 as usize;
                let i11 = r - i13 * (ne11 * ne12) as usize - i12 * ne11 as usize;

                let src_off = i11 * nb11 + i12 * nb12 + i13 * nb13;
                let dst_off = i11 * nb1 + i12 * nb2 + i13 * nb3;

                // x[j] = load(src[j]) * scale
                let mut row = vec![0f32; n as usize];
                match t1.ty {
                    GgmlType::F32 => {
                        let src = self.rd_f32s(s1, src_off, n as usize);
                        for (o, &v) in row.iter_mut().zip(src) {
                            *o = v * scale;
                        }
                    }
                    _ => {
                        let src: &[half::f16] =
                            bytemuck::cast_slice(self.rd(s1, src_off, n as usize * 2));
                        for (o, &v) in row.iter_mut().zip(src) {
                            *o = v.to_f32() * scale;
                        }
                    }
                }

                // butterfly passes (len 1 .. n/2)
                let mut len = 1usize;
                while len < n as usize {
                    let mut i = 0usize;
                    while i < n as usize {
                        for j in 0..len {
                            let u = row[i + j];
                            let v = row[i + len + j];
                            row[i + j] = u + v;
                            row[i + len + j] = u - v;
                        }
                        i += 2 * len;
                    }
                    len <<= 1;
                }

                let out = self.wr_f32s(dst, dst_off, n as usize);
                out.copy_from_slice(&row);
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// ggml_compute_forward_lightning_indexer (ops.cpp:12126-12214): for every
    /// (token t, kv ik): score = sum_h max(dot(q[h,t], k[ik]), 0)*w[h,t] +
    /// fp16(mask[t,ik]). Threads split the *kv* rows (nr = n_kv), exactly like
    /// the C; the q·k dot goes through the bit-exact AVX512 `ggml_vec_dot_f32`
    /// reproduction (`vec_dot_f32_c`).
    fn forward_lightning_indexer(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let sq = self.t(dst).src[0].unwrap();
        let sk = self.t(dst).src[1].unwrap();
        let sw = self.t(dst).src[2].unwrap();
        let sm = self.t(dst).src[3].unwrap();
        let (tq, tk, tw, tm) = (self.t(sq), self.t(sk), self.t(sw), self.t(sm));

        assert_eq!(self.t(dst).ty, GgmlType::F32);
        assert_eq!(tq.ty, GgmlType::F32);
        assert_eq!(tw.ty, GgmlType::F32);
        assert_eq!(tm.ty, GgmlType::F16);

        let n_embd = tq.ne[0] as usize;
        let n_head = tq.ne[1] as usize;
        let n_tokens = tq.ne[2] as usize;
        let n_stream = tq.ne[3] as usize;
        let n_kv = tk.ne[2] as usize;

        let (nbq1, nbq2, nbq3) = (tq.nb[1] as usize, tq.nb[2] as usize, tq.nb[3] as usize);
        let (nbk2, nbk3) = (tk.nb[2] as usize, tk.nb[3] as usize);
        let (nbw1, nbw3) = (tw.nb[1] as usize, tw.nb[3] as usize);
        let (nbm1, nbm3, nem3) = (tm.nb[1] as usize, tm.nb[3] as usize, tm.ne[3] as usize);
        let (nb1, nb3) = (self.t(dst).nb[1] as usize, self.t(dst).nb[3] as usize);

        let k_f16 = tk.ty == GgmlType::F16;
        assert!(
            k_f16 || tk.ty == GgmlType::F32,
            "lightning indexer: unsupported K-type {:?}",
            tk.ty
        );

        let nr = n_kv;
        let run = |ir0: usize, ir1: usize| {
            // (temporary) buffer for K converted to float (params->wdata in C)
            let mut k_row_f32 = vec![0f32; n_embd];
            for s in 0..n_stream {
                for t in 0..n_tokens {
                    let w_row: &[f32] = self.rd_f32s(sw, t * nbw1 + s * nbw3, n_head);
                    let m_row: &[half::f16] =
                        bytemuck::cast_slice(self.rd(sm, t * nbm1 + (s % nem3) * nbm3, n_kv * 2));
                    for ik in ir0..ir1 {
                        let k_off = ik * nbk2 + s * nbk3;
                        let k_row: &[f32] = if k_f16 {
                            let raw: &[half::f16] =
                                bytemuck::cast_slice(self.rd(sk, k_off, n_embd * 2));
                            for (dst_v, src_v) in k_row_f32.iter_mut().zip(raw) {
                                *dst_v = src_v.to_f32();
                            }
                            &k_row_f32
                        } else {
                            self.rd_f32s(sk, k_off, n_embd)
                        };
                        let mut score = 0.0f32;
                        for h in 0..n_head {
                            let q_row: &[f32] =
                                self.rd_f32s(sq, h * nbq1 + t * nbq2 + s * nbq3, n_embd);
                            let qk = crate::vec_dot::vec_dot_f32_c(n_embd, q_row, k_row);
                            // ReLU and weights (prescaled)
                            score += qk.max(0.0) * w_row[h];
                        }
                        // apply mask
                        let out = self.wr_f32s(dst, t * nb1 + s * nb3 + ik * 4, 1);
                        out[0] = score + m_row[ik].to_f32();
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_top_k_f32` (ops.cpp:8550-8604) — a faithful port
    /// of the reference's `std::partial_sort` over an index array with
    /// `cmp_top_k{src_data}` (`data[a] > data[b]`, :8550-8555). The heap
    /// mechanics below mirror libstdc++'s `stl_heap.h` (`__make_heap` /
    /// `__adjust_heap` / `__push_heap` / `__pop_heap` / `__sort_heap`) so that
    /// *tied* values select exactly the same index set as the reference —
    /// deepseek32's indexer scores tie massively at ReLU-zero, and the
    /// selected set decides which kv rows the DSA top-k mask unmasks.
    /// Finally the [0]<->[1] swap of ops.cpp:8597-8600 ("emphasize that the
    /// order is not important") is reproduced.
    fn forward_top_k(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "top_k: F32 only");
        assert_eq!(t0.nb[0] as usize, 4, "top_k: nb0 == sizeof(float)");

        let ne00 = t0.ne[0] as usize;
        let nb01 = t0.nb[1] as usize;
        let nr = t0.nrows() as usize;
        let top_k = td.ne[0] as usize;
        let nb1 = td.nb[1] as usize;

        // libstdc++ stl_heap.h __push_heap: sift `value` up from holeIndex
        // toward topIndex while `cmp(parent, value)`.
        fn push_heap_idx(
            a: &mut [usize],
            mut hole: usize,
            top: usize,
            value: usize,
            cmp: &dyn Fn(usize, usize) -> bool,
        ) {
            while hole > top {
                let parent = (hole - 1) / 2;
                if !cmp(a[parent], value) {
                    break;
                }
                a[hole] = a[parent];
                hole = parent;
            }
            a[hole] = value;
        }

        // __adjust_heap: sift the hole down to the fringe, then push `value`.
        // The `(len - 1) / 2` bounds use the C's signed truncating division
        // ((0 - 1) / 2 == 0), so len == 0 degenerates to a bare push.
        fn adjust_heap_idx(
            a: &mut [usize],
            hole: usize,
            len: usize,
            value: usize,
            cmp: &dyn Fn(usize, usize) -> bool,
        ) {
            let top = hole;
            let mut hole = hole;
            let mut second_child = hole;
            let bound = ((len as isize) - 1) / 2;
            while second_child < bound as usize {
                second_child = 2 * (second_child + 1);
                if cmp(a[second_child], a[second_child - 1]) {
                    second_child -= 1;
                }
                a[hole] = a[second_child];
                hole = second_child;
            }
            if len & 1 == 0 && second_child == (((len as isize) - 2) / 2) as usize {
                second_child = 2 * (second_child + 1);
                a[hole] = a[second_child - 1];
                hole = second_child - 1;
            }
            push_heap_idx(a, hole, top, value, cmp);
        }

        // __make_heap over a[..len] (stl_heap.h: no-op for len < 2)
        fn make_heap_idx(a: &mut [usize], len: usize, cmp: &dyn Fn(usize, usize) -> bool) {
            if len < 2 {
                return;
            }
            let mut parent = (len - 2) / 2;
            loop {
                let value = a[parent];
                adjust_heap_idx(&mut a[..len], parent, len, value, cmp);
                if parent == 0 {
                    break;
                }
                parent -= 1;
            }
        }

        // __sort_heap over a[..len]: --last; pop_heap(first, last, last)
        fn sort_heap_idx(a: &mut [usize], mut len: usize, cmp: &dyn Fn(usize, usize) -> bool) {
            while len > 1 {
                len -= 1;
                let value = a[len];
                a[len] = a[0];
                adjust_heap_idx(&mut a[..len], 0, len, value, cmp);
            }
        }

        for i in 0..nr {
            let src: &[f32] = self.rd_f32s(s0, i * nb01, ne00);
            // cmp_top_k: data[a] > data[b] — "a before b" (descending)
            let cmp = |a: usize, b: usize| -> bool { src[a] > src[b] };

            let mut tmp: Vec<usize> = (0..ne00).collect();
            if top_k == 0 {
                continue;
            }

            // std::partial_sort(tmp, tmp+top_k, tmp+ne00, cmp) =
            // __heap_select (make_heap over [0,k) + the scan) + __sort_heap
            if top_k < ne00 {
                make_heap_idx(&mut tmp[..top_k], top_k, &cmp);
                // scan: if cmp(*i, *first) → __pop_heap(first, middle, i):
                // value = *i; *i = *first; adjust_heap over the FULL heap
                // (len = middle - first = top_k — the heap keeps k elements)
                for idx in top_k..ne00 {
                    if cmp(tmp[idx], tmp[0]) {
                        let value = tmp[idx];
                        tmp[idx] = tmp[0];
                        adjust_heap_idx(&mut tmp[..top_k], 0, top_k, value, &cmp);
                    }
                }
                sort_heap_idx(&mut tmp[..top_k], top_k, &cmp);
            } else {
                // middle == last: heap-sort of the whole range
                make_heap_idx(&mut tmp, ne00, &cmp);
                sort_heap_idx(&mut tmp, ne00, &cmp);
            }

            let out: &mut [i32] = bytemuck::cast_slice_mut(self.wr(dst, i * nb1, top_k * 4));
            for (o, &t) in out.iter_mut().zip(tmp[..top_k].iter()) {
                *o = t as i32;
            }
            // emphasize that the order is not important (ops.cpp:8598-8600)
            if top_k > 1 {
                out.swap(0, 1);
            }
        }
    }

    // ==================================================================
    // sqrt + the fused DeepSeek-V4 hyper-connection ops — 对照
    // unary-ops.cpp:285 ggml_compute_forward_sqrt 与 ops.cpp:11108-11419
    // ggml_compute_forward_dsv4_hc_{comb,pre,post} (arch batch 7)
    // ==================================================================

    /// `ggml_compute_forward_sqrt` (unary-ops.cpp:285, `unary_op<op_sqrt>`):
    /// per-element `sqrtf` with the C row split.
    fn forward_sqrt(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "sqrt: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "sqrt: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = v.sqrt();
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_sin/cos` (unary-ops.cpp:289/293 →
    /// `unary_op<op_sin/op_cos>`): per-element `sinf`/`cosf` through the
    /// scalar `ggml_vec_{sin,cos}_f32` loops (vec.h:878/884 — no SIMD
    /// variant at this commit). F32 and F16 (F16 goes through
    /// fp16→fp32→libm→fp16, unary-ops.cpp:106-136); this port carries the
    /// audio graphs' F32 case, F32-only assert otherwise.
    fn forward_sin_cos(&self, dst: TensorId, nth: usize, team: &Team<'_>, is_cos: bool) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "sin/cos: F32 only (audio graphs)");
        assert_eq!(td.ty, GgmlType::F32, "sin/cos: F32 only (audio graphs)");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                if is_cos {
                    for (o, &v) in out.iter_mut().zip(src) {
                        *o = v.cos();
                    }
                } else {
                    for (o, &v) in out.iter_mut().zip(src) {
                        *o = v.sin();
                    }
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_sqr` (unary-ops.cpp:281 `op_sqr` → vec.h:859
    /// `y[i] = x[i]*x[i]`, scalar loop).
    fn forward_sqr(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "sqr: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "sqr: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = v * v;
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_mean_f32` (ops.cpp:1513-1543): row-wise
    /// `ggml_vec_sum_f32` (double accumulator, sequential over ne00) then
    /// `/= (float) ne00`. ith == 0 only — single-threaded in the reference,
    /// same here.
    fn forward_mean(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "mean: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "mean: F32 only");
        let ne00 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        for i3 in 0..t0.ne[3] as usize {
            for i2 in 0..t0.ne[2] as usize {
                for i1 in 0..t0.ne[1] as usize {
                    let x: &[f32] = self.rd_f32s(s0, i1 * nb01 + i2 * nb02 + i3 * nb03, ne00);
                    let mut sum = 0f64; // ggml_vec_sum_f32
                    for &v in x {
                        sum += v as f64;
                    }
                    let s = sum as f32 / ne00 as f32;
                    self.wr_f32s(dst, i1 * nb1 + i2 * nb2 + i3 * nb3, 1)[0] = s;
                }
            }
        }
    }

    /// `ggml_compute_forward_pad_reflect_1d` (ops.cpp:8282-8318): copy each
    /// source row into the middle of the dst row, then reflect —
    /// `left[-i0] = left[i0]` for i0 in 1..=p0, `right[i0] = right[-i0]` for
    /// i0 in 1..=p1. F32 only; the i1 loop strides by nth in the C, the
    /// rows are disjoint so the port's contiguous split is equivalent.
    fn forward_pad_reflect_1d(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "pad_reflect_1d: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "pad_reflect_1d: F32 only");
        let p0 = td.op_params[0] as usize;
        let p1 = td.op_params[1] as usize;
        let ne0 = td.ne[0] as usize; // padded width
        let n_rows = (t0.ne[1] * t0.ne[2] * t0.ne[3]) as usize;
        let ne01 = t0.ne[1] as usize;
        let ne02 = t0.ne[2] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |r0: usize, r1: usize| {
            for ir in r0..r1 {
                let i3 = ir / (ne02 * ne01);
                let i2 = (ir - i3 * ne02 * ne01) / ne01;
                let i1 = ir - i3 * ne02 * ne01 - i2 * ne01;
                let src_row = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_row = i1 * nb1 + i2 * nb2 + i3 * nb3;
                let ne00 = ne0 - p0 - p1;
                let mid: &[f32] = self.rd_f32s(s0, src_row, ne00);
                let out = self.wr_f32s(dst, dst_row, ne0);
                out[p0..p0 + ne00].copy_from_slice(mid);
                for i in 1..=p0 {
                    out[p0 - i] = out[p0 + i];
                }
                for i in 1..=p1 {
                    out[p0 + ne00 - 1 + i] = out[p0 + ne00 - 1 - i];
                }
            }
        };
        self.par_rows(n_rows, nth, team, run);
    }

    /// `ggml_compute_forward_elu` — `ggml_vec_elu_f32` (vec.h:915):
    /// `(x > 0) ? x : expm1f(x)`, scalar libm loop.
    fn forward_elu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "elu: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "elu: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = if v > 0.0 { v } else { v.exp_m1() };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    // ==================================================================
    // audio round 5 — the TTS generator op surface. Owner: agent GEN5.
    // qwen3tts-gen.cpp / pockettts-gen.cpp consume these.
    // ==================================================================

    /// `ggml_compute_forward_sum_f32` (ops.cpp:1282-1310): every row through
    /// `ggml_vec_sum_f32_ggf` (a ggml_float = double row accumulator), the
    /// row sums accumulated in double, one scalar store. ith == 0 only
    /// (GGML_OP_SUM is n_tasks == 1, ggml-cpu.c:2282).
    fn forward_sum(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(td.ne[0] * td.ne[1] * td.ne[2] * td.ne[3], 1, "sum: scalar dst");
        assert_eq!(t0.ty, GgmlType::F32, "sum: F32 only (TTS graphs)");
        let ne00 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let mut sum = 0f64;
        for i3 in 0..t0.ne[3] as usize {
            for i2 in 0..t0.ne[2] as usize {
                for i1 in 0..t0.ne[1] as usize {
                    let x: &[f32] = self.rd_f32s(s0, i1 * nb01 + i2 * nb02 + i3 * nb03, ne00);
                    let mut row_sum = 0f64;
                    for &v in x {
                        row_sum += v as f64;
                    }
                    sum += row_sum;
                }
            }
        }
        self.wr_f32s(dst, 0, 1)[0] = sum as f32;
    }

    /// `ggml_compute_forward_cumsum_f32` (ops.cpp:1410-1438): per row
    /// `y[i] = y[i-1] + x[i]` in f32 (vec.h:1507 — float accumulator, the
    /// first element is a plain copy). Rows split like the C
    /// `get_thread_range`.
    fn forward_cumsum(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "cumsum: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "cumsum: F32 only");
        assert_eq!(t0.nb[0], 4, "cumsum: src0->nb[0] == sizeof(float)");
        assert_eq!(td.nb[0], 4, "cumsum: dst->nb[0] == sizeof(float)");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i1 * nb01 + i2 * nb02 + i3 * nb03, ne0);
                let out = self.wr_f32s(dst, i1 * nb1 + i2 * nb2 + i3 * nb3, ne0);
                let mut prev = src[0];
                out[0] = prev;
                for i in 1..ne0 {
                    prev = prev + src[i];
                    out[i] = prev;
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_tri_f32` (ops.cpp:2289-2318): keep
    /// `bipred(i0, i1)` elements of each row, zero the rest. Row-parallel
    /// (n_tasks == n_threads, ggml-cpu.c:2271).
    fn forward_tri(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "tri: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "tri: F32 only");
        let bipred = |i0: usize, r: usize| -> bool {
            // ops.cpp:2300-2305 — the row index of the square matrix is i1
            match td.op_params[0] {
                crate::ops::GGML_TRI_TYPE_LOWER_DIAG => i0 <= r,
                crate::ops::GGML_TRI_TYPE_LOWER => i0 < r,
                crate::ops::GGML_TRI_TYPE_UPPER_DIAG => i0 >= r,
                crate::ops::GGML_TRI_TYPE_UPPER => i0 > r,
                other => unimplemented!("invalid tri type {other}"),
            }
        };
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (i0, o) in out.iter_mut().enumerate() {
                    *o = if bipred(i0, i1) { src[i0] } else { 0.0 };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_neg` (unary-ops.cpp:11 `op_neg` → :246
    /// `unary_op<op_neg>`): per-element `-x`, F32 only, row-parallel like
    /// every unary op (n_tasks == n_threads). The chunked delta-net's decay
    /// differences and pre-solve attn negation (delta-net-base.cpp:167/189).
    fn forward_neg(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "neg: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "neg: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = -v;
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_abs` (unary-ops.cpp:3 `op_abs` → :238
    /// `unary_op<op_abs>`): per-element `fabsf(x)`, F32 only, row-parallel
    /// like every unary op. The PLE gate's magnitude (qwen4exp.cpp:1246).
    fn forward_abs(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "abs: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "abs: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = v.abs();
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_sgn` (unary-ops.cpp:7 `op_sgn` → :242
    /// `unary_op<op_sgn>`): per-element `(x > 0) ? 1 : ((x < 0) ? -1 : 0)`,
    /// F32 only, row-parallel. The PLE gate's signed sqrt (qwen4exp.cpp:1247).
    fn forward_sgn(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "sgn: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "sgn: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_diag_f32` (ops.cpp:5434-5472): the single src row
    /// (ne01 == 1) scattered onto the diagonal of each dst square —
    /// `d[i1][i0] = (i0 == i1) ? s[i1] : 0`. n_tasks == 1 (ggml-cpu.c:2385,
    /// the `ith != 0` early return), so single-threaded here.
    fn forward_diag(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "diag: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "diag: F32 only");
        assert_eq!(t0.nb[0], 4, "diag: src0->nb[0] == sizeof(float)");
        assert_eq!(td.nb[0], 4, "diag: dst->nb[0] == sizeof(float)");
        assert_eq!(t0.ne[0], td.ne[0], "diag: ne00 == ne0");
        assert_eq!(t0.ne[0], td.ne[1], "diag: ne00 == ne1");
        assert_eq!(t0.ne[1], 1, "diag: ne01 == 1");
        assert_eq!(t0.ne[2], td.ne[2], "diag: ne02 == ne2");
        assert_eq!(t0.ne[3], td.ne[3], "diag: ne03 == ne3");
        let (ne0, ne1, ne2, ne3) = (
            td.ne[0] as usize,
            td.ne[1] as usize,
            td.ne[2] as usize,
            td.ne[3] as usize,
        );
        let (nb03, nb02) = (t0.nb[3] as usize, t0.nb[2] as usize);
        let (nb3, nb2, nb1) = (td.nb[3] as usize, td.nb[2] as usize, td.nb[1] as usize);
        for i3 in 0..ne3 {
            for i2 in 0..ne2 {
                // src0 has a single row per (i2, i3); its ne00 == ne1 values
                // become the successive diagonal entries (ops.cpp:5463)
                let s = self.rd_f32s(s0, i3 * nb03 + i2 * nb02, ne1);
                for i1 in 0..ne1 {
                    let d = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                    for (i0, o) in d.iter_mut().enumerate() {
                        *o = if i0 == i1 { s[i1] } else { 0.0 };
                    }
                }
            }
        }
    }

    /// `ggml_compute_forward_solve_tri_f32` (ops.cpp:10824-10880): dst = A⁻¹B
    /// by forward substitution — A square lower-triangular [n, n, B1, B2],
    /// B/X with `k` as B's fast dim [k, n, B1, B2]. The parallel unit is one
    /// solve *column* `i01` inside one batch (i02, i03): `nr = ne02*ne03*k`
    /// units split dr-per-thread (ggml-cpu.c:2290 n_tasks = n_threads). The
    /// substitution sum is a strictly sequential scalar loop in the C — kept
    /// scalar here, no reassociation, so the result is bit-exact regardless
    /// of the column split.
    fn forward_solve_tri(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let (s0, s1) = (self.t(dst).src[0].unwrap(), self.t(dst).src[1].unwrap());
        let t0 = self.t(s0);
        let t1 = self.t(s1);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "solve_tri: A F32");
        assert_eq!(t1.ty, GgmlType::F32, "solve_tri: B F32");
        assert_eq!(td.ty, GgmlType::F32, "solve_tri: dst F32");
        assert_eq!(t0.ne[0], t0.ne[1], "solve_tri: A square");
        assert_eq!(td.ne[0], t1.ne[0], "solve_tri: ne0 == ne10");
        assert_eq!(td.ne[1], t1.ne[1], "solve_tri: ne1 == ne11");
        let k = t1.ne[0] as usize; // number of solve columns (B's fast dim)
        let n = t1.ne[1] as usize; // A is n×n
        let ne02 = t0.ne[2] as usize;
        let (nb02, nb03) = (t0.nb[2] as usize, t0.nb[3] as usize); // A batch strides
        let (nb12, nb13) = (t1.nb[2] as usize, t1.nb[3] as usize); // B batch strides
        let (nb2, nb3) = (td.nb[2] as usize, td.nb[3] as usize); // dst batch strides
        let nr = ne02 * t0.ne[3] as usize * k;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i03 = ir / (ne02 * k);
                let i02 = (ir - i03 * ne02 * k) / k;
                let i01 = ir - i03 * ne02 * k - i02 * k;
                // whole-batch slices: A [n*n], B [n*k], X [n*k] — one unit
                // owns its (i02, i03, i01) column exclusively, so the &mut X
                // never crosses a thread (ops.cpp:10857-10861)
                let a_b = self.rd_f32s(s0, i02 * nb02 + i03 * nb03, n * n);
                let b_b = self.rd_f32s(s1, i02 * nb12 + i03 * nb13, n * k);
                let x_b = self.wr_f32s(dst, i02 * nb2 + i03 * nb3, n * k);
                for i00 in 0..n {
                    let mut sum = 0.0f32;
                    for t in 0..i00 {
                        // C source semantics (ops.cpp:10871): a strictly
                        // sequential two-rounding accumulate. NB the
                        // reference .so (GCC 13.3 -O3, default
                        // -ffp-contract=fast) FMA-contracts *parts* of this
                        // loop per trip count (zmm column-vector bodies with
                        // in-order vmulps/vaddss + vfmadd231ss scalar
                        // remainders — a codegen mix that matches neither a
                        // pure strict nor a pure fma form; verified per-row
                        // against /tmp dumps: small-k shapes are 100% strict,
                        // k≥4 shapes are mixed). No single Rust form
                        // reproduces that mix, so the port is source-faithful
                        // and the parity tests bit-compare the strict shapes
                        // + ulp-bound the vectorized ones (see
                        // mtp2_tri_ops_bitcompare's notes).
                        sum += a_b[i00 * n + t] * x_b[t * k + i01];
                    }
                    let diag = a_b[i00 * n + i00];
                    x_b[i00 * k + i01] = (b_b[i00 * k + i01] - sum) / diag;
                }
            }
        };
        if nr == 0 {
            return;
        }
        if nth <= 1 || nr == 1 {
            run(0, nr);
        } else {
            team.run(nr, &run);
        }
    }

    /// `ggml_compute_forward_set_f32` (ops.cpp:4769-4834), inplace arm only:
    /// copy `b`'s rows into dst viewed through the op_params strides at the
    /// op_params offset. n_tasks == 1 (ggml-cpu.c:2379) — single-threaded, no
    /// barrier needed for the !inplace memcpy arm (never built here).
    fn forward_set(&self, dst: TensorId) {
        let (a, b) = (self.t(dst).src[0].unwrap(), self.t(dst).src[1].unwrap());
        let tb = self.t(b);
        let td = self.t(dst);
        assert_eq!(self.t(a).ty, GgmlType::F32, "set: F32 only");
        assert_eq!(tb.ty, GgmlType::F32, "set: F32 only");
        let p = &td.op_params;
        assert_eq!(p[4], 1, "set: only the inplace arm is ported");
        // C stores the strides/offset as int32 params (ggml.c:3523)
        let (nb1, nb2, nb3, offset) = (
            p[0] as u32 as usize,
            p[1] as u32 as usize,
            p[2] as u32 as usize,
            p[3] as u32 as usize,
        );
        let nc = tb.ne[0] as usize;
        let nr = tb.nrows() as usize;
        let (ne11, ne12) = (tb.ne[1].max(1) as usize, tb.ne[2].max(1) as usize);
        let (nb11, nb12, nb13) = (tb.nb[1] as usize, tb.nb[2] as usize, tb.nb[3] as usize);
        assert_eq!(tb.nb[0], 4, "set: nb10 == sizeof(float)");
        for ir in 0..nr {
            let i3 = ir / (ne12 * ne11);
            let i2 = (ir - i3 * ne12 * ne11) / ne11;
            let i1 = ir - i3 * ne12 * ne11 - i2 * ne11;
            let src = self.rd(b, i3 * nb13 + i2 * nb12 + i1 * nb11, nc * 4);
            let o = self.wr(dst, i3 * nb3 + i2 * nb2 + i1 * nb1 + offset, nc * 4);
            o.copy_from_slice(src);
        }
    }

    /// `ggml_compute_forward_log` (unary-ops.cpp:297, `unary_op<op_log>` →
    /// vec.h:872 `ggml_vec_log_f32`): per-element `logf`. GGML_OP_LOG is
    /// n_tasks == 1 (ggml-cpu.c:2279) — whole tensor on one task.
    fn forward_log(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "log: F32 only (TTS graphs)");
        assert_eq!(td.ty, GgmlType::F32, "log: F32 only (TTS graphs)");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        for ir in 0..nr {
            let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
            let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
            let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
            let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
            let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
            for (o, &v) in out.iter_mut().zip(src) {
                *o = v.ln();
            }
        }
    }

    /// `ggml_compute_forward_step` — `ggml_vec_step_f32` (vec.h:903):
    /// `(x[i] > 0.f) ? 1.f : 0.f`, scalar loop through the unary_op template.
    fn forward_step(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "step: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "step: F32 only");
        let nr = t0.nrows() as usize;
        let ne0 = t0.ne[0] as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (t0.ne[2] as usize * t0.ne[1] as usize);
                let i2 = (ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize) / t0.ne[1] as usize;
                let i1 = ir - i3 * t0.ne[2] as usize * t0.ne[1] as usize - i2 * t0.ne[1] as usize;
                let src = self.rd_f32s(s0, i3 * nb03 + i2 * nb02 + i1 * nb01, ne0);
                let out = self.wr_f32s(dst, i3 * nb3 + i2 * nb2 + i1 * nb1, ne0);
                for (o, &v) in out.iter_mut().zip(src) {
                    *o = if v > 0.0 { 1.0 } else { 0.0 };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_col2im_1d_impl` (ops.cpp:7017-7068): each dst
    /// element gathers the ceil(K/s) columns whose window covers it:
    /// t_in ranges over `ceil((t_abs-K+1)/s) ..= t_abs/s` (clamped), each
    /// with `k = t_abs - t_in*s`. F32 accumulator; F16/BF16 go through the
    /// conversion table exactly like the C template. Parallelized over the
    /// time axis (the C's `dr` bands over T_out).
    fn forward_col2im_1d(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.nb[0], td.nb[0], "col2im_1d: matching element size");
        assert_eq!(
            t0.ty, td.ty,
            "col2im_1d: src and dst share the type (ops.cpp:7070)"
        );
        let s0p = td.op_params[0] as i64;
        let oc = td.op_params[1] as i64;
        let p0 = td.op_params[2] as i64;
        let k_oc = t0.ne[0];
        let t_in_max = t0.ne[1] - 1;
        let k = k_oc / oc;
        let t_out = td.ne[0];
        assert!(k > 0, "col2im_1d: K > 0");

        let elem = |bytes: &[u8], i: usize| -> f32 {
            match t0.ty {
                GgmlType::F32 => {
                    f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
                }
                GgmlType::F16 => {
                    f16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32()
                }
                GgmlType::Bf16 => {
                    bf16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32()
                }
                other => unimplemented!("col2im_1d: unsupported type {other:?}"),
            }
        };
        let store = |bytes: &mut [u8], i: usize, v: f32| {
            match t0.ty {
                GgmlType::F32 => bytes[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes()),
                GgmlType::F16 => {
                    bytes[2 * i..2 * i + 2]
                        .copy_from_slice(&f16::from_f32(v).to_le_bytes());
                }
                GgmlType::Bf16 => {
                    bytes[2 * i..2 * i + 2]
                        .copy_from_slice(&bf16::from_f32(v).to_le_bytes());
                }
                other => unimplemented!("col2im_1d: unsupported type {other:?}"),
            }
        };

        let run = |it0: usize, it1: usize| {
            for oc_i in 0..oc as usize {
                for t_out_i in it0..it1 {
                    // ops.cpp:7045-7053 — gather every (t_in, k) hitting t_abs
                    let t_abs = t_out_i as i64 + p0;
                    let mut t_in_min = (t_abs - k + 1 + s0p - 1).div_euclid(s0p);
                    if t_in_min < 0 {
                        t_in_min = 0;
                    }
                    let mut t_in_max_i = t_abs.div_euclid(s0p);
                    if t_in_max_i > t_in_max {
                        t_in_max_i = t_in_max;
                    }
                    let mut sum = 0f32;
                    for t_in in t_in_min..=t_in_max_i {
                        let kk = t_abs - t_in * s0p;
                        if (0..k).contains(&kk) {
                            // col layout: [K*OC, T_in], element (oc*K + k, t_in)
                            let idx = (oc_i as i64 * k + kk + t_in * k_oc) as usize;
                            let es = t0.ty.type_size();
                            let col = self.rd(s0, es * idx, es);
                            sum += elem(col, 0);
                        }
                    }
                    // dst layout: [T_out, OC], element (t_out, oc)
                    let out_i = t_out_i + oc_i * t_out as usize;
                    let es = td.ty.type_size();
                    let out = self.wr(dst, es * out_i, es);
                    store(out, 0, sum);
                }
            }
        };
        self.par_rows(t_out as usize, nth, team, run);
    }

    /// `ggml_compute_forward_pad_f32<circular = false>` (ops.cpp:8196-8258):
    /// the destination index is linear over dst's ne (contiguous F32), the
    /// source is addressed through its own strides minus the left pads, and
    /// everything inside a pad margin reads 0. The C thread split strides i1
    /// by nth (`i1 = ith; i1 < ne1; i1 += nth`, ops.cpp:8216) with i0/i2/i3
    /// whole; the writes are disjoint, so this port's contiguous row split
    /// over the (i3, i1) pairs computes the same bytes.
    fn forward_pad(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "pad: F32 only");
        assert_eq!(td.ty, GgmlType::F32, "pad: F32 only");
        assert_eq!(td.op_params[8], 0, "pad: circular variant not ported");
        assert_eq!(td.nb[0], 4, "pad: dst->nb[0] == sizeof(float)");
        let (lp0, rp0) = (td.op_params[0] as i64, td.op_params[1] as i64);
        let (lp1, rp1) = (td.op_params[2] as i64, td.op_params[3] as i64);
        let (lp2, rp2) = (td.op_params[4] as i64, td.op_params[5] as i64);
        let (lp3, rp3) = (td.op_params[6] as i64, td.op_params[7] as i64);
        let (ne0, ne1, ne2, ne3) = (td.ne[0], td.ne[1], td.ne[2], td.ne[3]);
        let (nb00, nb01, nb02, nb03) =
            (t0.nb[0] as i64, t0.nb[1] as i64, t0.nb[2] as i64, t0.nb[3] as i64);

        let run = |r0: usize, r1: usize| {
            for r in r0..r1 {
                let i3 = (r / ne1 as usize) as i64;
                let i1 = (r % ne1 as usize) as i64;
                for i2 in 0..ne2 {
                    for i0 in 0..ne0 {
                        // ops.cpp:8245-8257 — dst_idx = i3*(ne0*ne1*ne2) +
                        // i2*(ne0*ne1) + i1*ne0 + i0 (contiguous dst)
                        let dst_idx =
                            (i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0) as usize;
                        let within = (i0 >= lp0 && i0 < ne0 - rp0)
                            && (i1 >= lp1 && i1 < ne1 - rp1)
                            && (i2 >= lp2 && i2 < ne2 - rp2)
                            && (i3 >= lp3 && i3 < ne3 - rp3);
                        if within {
                            let src_idx = ((i3 - lp3) * nb03
                                + (i2 - lp2) * nb02
                                + (i1 - lp1) * nb01
                                + (i0 - lp0) * nb00) as usize;
                            let v = self.rd_f32s(s0, src_idx, 1)[0];
                            self.wr_f32s(dst, dst_idx * 4, 1)[0] = v;
                        } else {
                            self.wr_f32s(dst, dst_idx * 4, 1)[0] = 0.0;
                        }
                    }
                }
            }
        };
        self.par_rows((ne1 * ne3) as usize, nth, team, run);
    }

    /// `ggml_compute_forward_pool_1d_ksp` (ops.cpp:7690-7754) — single thread
    /// (ith == 0), per-row (nr = ne1*ne2*ne3 of src) sliding window with
    /// boundary skips. POOL_AVG divides by the in-range `count`
    /// (`res = count > 0 ? res/count : 0`), NOT by k0 — this is the one
    /// behavioral difference from pool_2d (which divides by ka = k0*k1).
    fn forward_pool_1d(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert!(
            matches!(t0.ty, GgmlType::F32 | GgmlType::F16),
            "pool_1d: F32/F16 src only"
        );
        assert_eq!(td.ty, GgmlType::F32, "pool_1d: F32 dst");
        let p = td.op_params;
        let (op, k0, s0p, p0) = (p[0], p[1] as i64, p[2] as i64, p[3] as i64);
        // op is the C `ggml_op_pool` discriminant (ggml.h:2277): 0 = MAX,
        // 1 = AVG
        let is_avg = op == crate::tensor::Context::GGML_OP_POOL_AVG;
        assert!(op == 0 || op == 1, "pool_1d: AVG/MAX only");
        let iw = t0.ne[0] as i64;
        let ow = td.ne[0] as usize;
        let nr = (t0.ne[1] * t0.ne[2] * t0.ne[3]) as usize;
        let nb01 = t0.nb[1] as usize;
        let nb1d = td.nb[1] as usize;
        for ir in 0..nr {
            let srow_off = ir * nb01;
            let drow_off = ir * nb1d;
            for owi in 0..ow {
                let mut res = if is_avg { 0.0f32 } else { f32::MIN };
                let mut count = 0i64;
                let base = owi as i64 * s0p - p0;
                for ki in 0..k0 {
                    let j = base + ki;
                    if j < 0 || j >= iw {
                        continue;
                    }
                    let v = match t0.ty {
                        GgmlType::F32 => self.rd_f32s(s0, srow_off + (j as usize) * 4, 1)[0],
                        _ => {
                            let h = u16::from_le_bytes(
                                self.rd(s0, srow_off + (j as usize) * 2, 2).try_into().unwrap(),
                            );
                            crate::ops::ggml_compute_fp16_to_fp32(h)
                        }
                    };
                    if is_avg {
                        res += v;
                    } else {
                        res = res.max(v);
                    }
                    count += 1;
                }
                if is_avg {
                    res = if count > 0 { res / count as f32 } else { 0.0 };
                }
                self.wr_f32s(dst, drow_off + owi * 4, 1)[0] = res;
            }
        }
    }

    /// `ggml_compute_forward_relu` → `ggml_vec_relu_f32` (vec.h:922):
    /// `y[i] = (x[i] > 0) ? x[i] : 0` — plain scalar loop, no SIMD variant.
    fn forward_relu(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "relu: F32 only (the audio graphs' dtype)");
        assert_eq!(td.ty, GgmlType::F32, "relu: F32 dst");
        let nc = t0.ne[0] as usize;
        let nr = t0.nrows() as usize;
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne1 = t0.ne[1] as usize;
        let ne2 = t0.ne[2] as usize;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i3 = ir / (ne2 * ne1);
                let i2 = (ir - i3 * ne2 * ne1) / ne1;
                let i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                let src_off = i1 * nb01 + i2 * nb02 + i3 * nb03;
                let dst_off = i1 * nb1 + i2 * nb2 + i3 * nb3;
                let x: &[f32] = self.rd_f32s(s0, src_off, nc);
                let y: &mut [f32] = self.wr_f32s(dst, dst_off, nc);
                for i in 0..nc {
                    y[i] = if x[i] > 0.0 { x[i] } else { 0.0 };
                }
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_roll_f32` (ops.cpp:8328-8373): per (i1,i2,i3) row,
    /// copy the row rotated by s0 (source index wraps); the row indices
    /// themselves are wrapped by s1/s2/s3. Thread-split over ne1*ne2*ne3 like
    /// the C (`per_thread = (total + nth) / nth`).
    fn forward_roll(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert_eq!(t0.ty, GgmlType::F32, "roll: F32 only (ops.cpp:8365)");
        let (sh0, sh1, sh2, sh3) = (
            td.op_params[0] as i64,
            td.op_params[1] as i64,
            td.op_params[2] as i64,
            td.op_params[3] as i64,
        );
        let wrap = |i: i64, ne: i64| -> i64 {
            if i < 0 {
                i + ne
            } else if i >= ne {
                i - ne
            } else {
                i
            }
        };
        let (ne1, ne2, ne3) = (t0.ne[1], t0.ne[2], t0.ne[3]);
        let (nb01, nb02, nb03) = (t0.nb[1] as usize, t0.nb[2] as usize, t0.nb[3] as usize);
        let (nb1, nb2, nb3) = (td.nb[1] as usize, td.nb[2] as usize, td.nb[3] as usize);
        let ne00 = t0.ne[0] as usize;
        let total = (ne1 * ne2 * ne3) as usize;
        let run = |start: usize, end: usize| {
            for i in start..end {
                let i1 = (i as i64) % ne1;
                let i2 = ((i as i64) / ne1) % ne2;
                let i3 = (i as i64) / (ne2 * ne1);
                let dst_off = (i3 as usize) * nb3 + (i2 as usize) * nb2 + (i1 as usize) * nb1;
                let i01 = wrap(i1 - sh1, ne1);
                let i02 = wrap(i2 - sh2, ne2);
                let i03 = wrap(i3 - sh3, ne3);
                let src_off = (i03 as usize) * nb03 + (i02 as usize) * nb02 + (i01 as usize) * nb01;
                let s = wrap(-sh0, ne00 as i64) as usize;
                let n = ne00 - s;
                let dst_row: &mut [f32] = self.wr_f32s(dst, dst_off, ne00);
                let src_a: &[f32] = self.rd_f32s(s0, src_off + s * 4, n);
                dst_row[..n].copy_from_slice(src_a);
                let src_b: &[f32] = self.rd_f32s(s0, src_off, s);
                dst_row[n..].copy_from_slice(src_b);
            }
        };
        self.par_rows(total, nth, team, run);
    }

    /// `ggml_compute_forward_conv_2d_dw` → the whcn branch (ops.cpp:7568-7607):
    /// src [W, H, C, N] contiguous, kernel [KW, KH, 1, C] (F32 or F16 read
    /// element-wise, ops.cpp:7532 ggml_conv_2d_dw_knl_f32). Scalar
    /// accumulation `sum += k * s` — no vec_dot, so no reduction-order
    /// subtleties. The cwhn branch (channels-contiguous src) is rejected
    /// loudly: every audio graph feeds the whcn layout.
    fn forward_conv_2d_dw(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let kern = self.t(dst).src[0].unwrap();
        let src = self.t(dst).src[1].unwrap();
        let tk = self.t(kern);
        let ts = self.t(src);
        let td = self.t(dst);
        assert!(
            matches!(tk.ty, GgmlType::F32 | GgmlType::F16),
            "conv_2d_dw: F32/F16 kernel"
        );
        assert_eq!(ts.ty, GgmlType::F32, "conv_2d_dw: F32 src");
        assert_eq!(td.ty, GgmlType::F32, "conv_2d_dw: F32 dst");
        // ggml_is_contiguous(src) — the whcn fast path (ops.cpp:7671)
        assert!(
            ts.nb[0] == 4 && ts.nb[1] == (ts.ne[0] * 4) as u64,
            "conv_2d_dw: only the contiguous (whcn) src layout is ported"
        );
        let p = &td.op_params;
        let (stride_x, stride_y, pad_x, pad_y, dil_x, dil_y) =
            (p[0] as i64, p[1] as i64, p[2] as i64, p[3] as i64, p[4] as i64, p[5] as i64);
        let channels = ts.ne[2];
        let batch = ts.ne[3];
        let (src_w, src_h) = (ts.ne[0], ts.ne[1]);
        let (dst_w, dst_h) = (td.ne[0], td.ne[1]);
        let (knl_w, knl_h) = (tk.ne[0], tk.ne[1]);

        // thread-split over channels*batch rows (ops.cpp:7585-7589)
        let n = (channels * batch) as usize;
        let run = |start: usize, end: usize| {
            for i in start..end {
                let knl_offset = ((i as i64) % channels) * knl_w * knl_h;
                let src_base = i as usize * (src_w * src_h) as usize * 4;
                let dst_base = i as usize * (dst_w * dst_h) as usize * 4;
                for dy in 0..dst_h {
                    for dx in 0..dst_w {
                        let mut sum = 0.0f32;
                        for ky in 0..knl_h {
                            let sy = dy * stride_y + ky * dil_y - pad_y;
                            if sy < 0 || sy >= src_h {
                                continue;
                            }
                            for kx in 0..knl_w {
                                let sx = dx * stride_x + kx * dil_x - pad_x;
                                if sx < 0 || sx >= src_w {
                                    continue;
                                }
                                let ki = (knl_offset + ky * knl_w + kx) as usize;
                                let k = match tk.ty {
                                    GgmlType::F32 => self.rd_f32s(kern, ki * 4, 1)[0],
                                    _ => {
                                        let h = u16::from_le_bytes(
                                            self.rd(kern, ki * 2, 2).try_into().unwrap(),
                                        );
                                        crate::ops::ggml_compute_fp16_to_fp32(h)
                                    }
                                };
                                let sv = self.rd_f32s(
                                    src,
                                    src_base + ((sy * src_w + sx) as usize) * 4,
                                    1,
                                )[0];
                                // the C loop body `sum += k * s` contracts to
                                // an FMA chain under the reference build's
                                // default -ffp-contract=fast (ops.cpp:7603)
                                sum = k.mul_add(sv, sum);
                            }
                        }
                        let dst_off = dst_base + ((dy * dst_w + dx) as usize) * 4;
                        self.wr_f32s(dst, dst_off, 1)[0] = sum;
                    }
                }
            }
        };
        self.par_rows(n, nth, team, run);
    }

    /// `ggml_compute_forward_pool_2d` (ops.cpp:7767-7852) — single thread
    /// (ith == 0), one output plane per ne2/ne3 slice; the pool window skips
    /// out-of-bounds source elements. POOL_MAX / POOL_AVG only.
    fn forward_pool_2d(&self, dst: TensorId) {
        let s0 = self.t(dst).src[0].unwrap();
        let t0 = self.t(s0);
        let td = self.t(dst);
        assert!(
            matches!(t0.ty, GgmlType::F32 | GgmlType::F16),
            "pool_2d: F32/F16 src only"
        );
        assert_eq!(td.ty, GgmlType::F32, "pool_2d: F32 dst");
        let p = td.op_params;
        let (op, k0, k1, s0p, s1p, p0, p1) =
            (p[0], p[1] as i64, p[2] as i64, p[3] as i64, p[4] as i64, p[5] as i64, p[6] as i64);
        assert!(op == 0 || op == 1, "pool_2d: AVG/MAX only");
        // op is the C `ggml_op_pool` discriminant (ggml.h:2277): 0 = MAX,
        // 1 = AVG (aligned with ggml.h by the POOL_1D round — was inverted
        // before, internally consistently)
        let is_avg = op == crate::tensor::Context::GGML_OP_POOL_AVG;
        let (px, py) = (td.ne[0] as usize, td.ne[1] as usize);
        let ka = (k0 * k1) as usize;
        let ne0 = t0.ne[0] as i64;
        let ne1 = t0.ne[1] as i64;
        let nb01 = t0.nb[1] as usize;
        let nb02 = t0.nb[2] as usize;
        let pa = px * py;
        let mut cplane_off = 0usize;
        let mut dplane_off = 0usize;
        for _i23 in 0..(t0.ne[2] * t0.ne[3]) {
            for oy in 0..py {
                for ox in 0..px {
                    let mut res = if is_avg { 0.0f32 } else { f32::MIN };
                    let ix = -p0 + ox as i64 * s0p;
                    let iy = -p1 + oy as i64 * s1p;
                    for ky in 0..k1 {
                        if iy + ky < 0 || iy + ky >= ne1 {
                            continue;
                        }
                        let srow_off = cplane_off + ((iy + ky) as usize) * nb01;
                        for kx in 0..k0 {
                            let j = ix + kx;
                            if j < 0 || j >= ne0 {
                                continue;
                            }
                            let v = match t0.ty {
                                GgmlType::F32 => self.rd_f32s(s0, srow_off + (j as usize) * 4, 1)[0],
                                _ => {
                                    let h = u16::from_le_bytes(
                                        self.rd(s0, srow_off + (j as usize) * 2, 2)
                                            .try_into()
                                            .unwrap(),
                                    );
                                    crate::ops::ggml_compute_fp16_to_fp32(h)
                                }
                            };
                            if is_avg {
                                res += v;
                            } else {
                                res = res.max(v);
                            }
                        }
                    }
                    if is_avg {
                        res /= ka as f32;
                    }
                    // dplane_off is already in BYTES (advanced by pa*4 per
                    // plane) — only the in-plane index scales by the element
                    self.wr_f32s(dst, dplane_off + (oy * px + ox) * 4, 1)[0] = res;
                }
            }
            cplane_off += nb02;
            dplane_off += pa * 4;
        }
    }

    /// `ggml_compute_forward_arange_f32` (ops.cpp:8386-8405): thread-strided
    /// fill `start + step*i`.
    fn forward_arange(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let td = self.t(dst);
        assert_eq!(td.ty, GgmlType::F32, "arange: F32 only");
        let start = f32::from_bits(td.op_params[0] as u32);
        let step = f32::from_bits(td.op_params[2] as u32);
        let steps = ((f32::from_bits(td.op_params[1] as u32) - start) / step).ceil() as i64;
        assert_eq!(td.ne[0], steps, "arange: ne0 == steps");
        let run = |i0: usize, i1: usize| {
            for i in i0..i1 {
                self.wr_f32s(dst, i * 4, 1)[0] = start + step * i as f32;
            }
        };
        self.par_rows(steps as usize, nth, team, run);
    }

    /// `ggml_dsv4_hc_comb_norm_cols` (ops.cpp:11110-11124): column (= dst
    /// stream) normalization of the 4x4 comb, eps added to the column sum.
    fn dsv4_hc_comb_norm_cols(comb: &mut [f32], eps: f32) {
        const HC: usize = 4;
        for idst in 0..HC {
            let mut sum = eps;
            for isrc in 0..HC {
                sum += comb[idst + HC * isrc];
            }
            let inv_sum = 1.0 / sum;
            for isrc in 0..HC {
                comb[idst + HC * isrc] *= inv_sum;
            }
        }
    }

    /// `ggml_dsv4_hc_comb_norm_rows` (ops.cpp:11126-11140): row (= src stream)
    /// normalization.
    fn dsv4_hc_comb_norm_rows(comb: &mut [f32], eps: f32) {
        const HC: usize = 4;
        for isrc in 0..HC {
            let mut sum = eps;
            for idst in 0..HC {
                sum += comb[idst + HC * isrc];
            }
            let inv_sum = 1.0 / sum;
            for idst in 0..HC {
                comb[idst + HC * isrc] *= inv_sum;
            }
        }
    }

    /// `ggml_compute_forward_dsv4_hc_comb_f32` (ops.cpp:11142-11227): per
    /// token, the 4x4 comb block of `mixes` rows [2*hc .. hc_mix_dim) is
    /// affine-mixed (scale[2], base), softmaxed per src-stream row, +eps,
    /// then column-normalized once and row/col-normalized (n_iter - 1)
    /// times. Threads split the token range (ops.cpp:11182-11186).
    fn forward_dsv4_hc_comb(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let sm = self.t(dst).src[0].unwrap();
        let ss = self.t(dst).src[1].unwrap();
        let sb = self.t(dst).src[2].unwrap();
        let (tm, ts, tb, td) = (self.t(sm), self.t(ss), self.t(sb), self.t(dst));

        assert_eq!(tm.ty, GgmlType::F32);
        assert_eq!(ts.ty, GgmlType::F32);
        assert_eq!(tb.ty, GgmlType::F32);
        assert_eq!(td.ty, GgmlType::F32);

        const HC: i64 = 4;
        const COMB_OFFSET: i64 = 2 * HC;
        const HC_MIX_DIM: i64 = (2 + HC) * HC;

        let n_tokens = tm.ne[1] as usize;

        assert_eq!(tm.ne[0], HC_MIX_DIM);
        assert_eq!(td.ne[0], HC);
        assert_eq!(td.ne[1], HC);
        assert_eq!(td.ne[2] as usize, n_tokens);
        assert!(ts.ne[0] >= 3);
        assert_eq!(tb.ne[0], HC_MIX_DIM);

        let (nbm0, nbm1) = (tm.nb[0] as usize, tm.nb[1] as usize);
        let nbs0 = ts.nb[0] as usize;
        let nbb0 = tb.nb[0] as usize;
        let (nbd0, nbd1, nbd2) =
            (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize);

        let eps = f32::from_bits(td.op_params[0] as u32);
        let n_iter = td.op_params[1];
        assert!(n_iter > 0);

        // ggml.c:6497 — scale_comb = scale[2]
        let scale_comb = self.rd_f32s(ss, 2 * nbs0, 1)[0];

        let run = |it0: usize, it1: usize| {
            let mut comb = [0f32; (HC * HC) as usize];
            for it in it0..it1 {
                // per src-stream softmax over dst (ops.cpp:11191-11205)
                for isrc in 0..HC {
                    let mut max = f32::NEG_INFINITY;
                    for idst in 0..HC {
                        let idx = (idst + HC * isrc) as usize;
                        let xv = self.rd_f32s(sm, (COMB_OFFSET as usize + idx) * nbm0 + it * nbm1, 1)[0];
                        let bv = self.rd_f32s(sb, (COMB_OFFSET as usize + idx) * nbb0, 1)[0];
                        // `xv * scale_comb + bv` — gcc -O2 contracts this into
                        // one vfmadd (single rounding); f32::mul_add is the
                        // same fused op
                        let v = xv.mul_add(scale_comb, bv);
                        comb[idx] = v;
                        max = max.max(v);
                    }
                    let mut sum = 0.0f32;
                    for idst in 0..HC {
                        let idx = (idst + HC * isrc) as usize;
                        let v = (comb[idx] - max).exp();
                        comb[idx] = v;
                        sum += v;
                    }
                    let inv_sum = 1.0 / sum;
                    for idst in 0..HC {
                        let idx = (idst + HC * isrc) as usize;
                        // `comb[idx] * inv_sum + eps` — same fma contraction
                        comb[idx] = comb[idx].mul_add(inv_sum, eps);
                    }
                }

                Self::dsv4_hc_comb_norm_cols(&mut comb, eps);
                for _ in 1..n_iter {
                    Self::dsv4_hc_comb_norm_rows(&mut comb, eps);
                    Self::dsv4_hc_comb_norm_cols(&mut comb, eps);
                }

                for isrc in 0..HC {
                    for idst in 0..HC {
                        let idx = (idst + HC * isrc) as usize;
                        self.wr_f32s(
                            dst,
                            (idst as usize) * nbd0 + (isrc as usize) * nbd1 + it * nbd2,
                            1,
                        )[0] = comb[idx];
                    }
                }
            }
        };
        self.par_rows(n_tokens, nth, team, run);
    }

    /// `ggml_compute_forward_dsv4_hc_pre_f32` (ops.cpp:11248-11308): per
    /// (i0, it) element, sum over the hc streams of x * weights (the fused
    /// form of the per-stream mul/add chain of build_hc_pre's fallback,
    /// deepseek4.cpp:307-314). Threads split the flattened n_embd*n_tokens
    /// range in (i0 fastest) order — the index decomposition of ops.cpp:11281.
    fn forward_dsv4_hc_pre(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let sx = self.t(dst).src[0].unwrap();
        let sw = self.t(dst).src[1].unwrap();
        let (tx, tw, td) = (self.t(sx), self.t(sw), self.t(dst));

        assert_eq!(tx.ty, GgmlType::F32);
        assert_eq!(tw.ty, GgmlType::F32);
        assert_eq!(td.ty, GgmlType::F32);

        let n_embd = tx.ne[0] as usize;
        let hc = tx.ne[1] as usize;
        let n_tokens = tx.ne[2] as usize;

        let scale = f32::from_bits(td.op_params[0] as u32);
        let gated = td.op_params[1] != 0;

        assert_eq!(td.ne[0] as usize, n_embd);
        assert_eq!(td.ne[1] as usize, n_tokens);
        if gated {
            assert_eq!(tw.ne[0] as usize, n_embd);
            assert_eq!(tw.ne[1] as usize, hc);
            assert_eq!(tw.ne[2] as usize, n_tokens);
        } else {
            assert_eq!(tw.ne[0] as usize, hc);
            assert_eq!(tw.ne[1] as usize, n_tokens);
        }

        let (nbx0, nbx1, nbx2) = (tx.nb[0] as usize, tx.nb[1] as usize, tx.nb[2] as usize);
        let (nbw0, nbw1, nbw2) = (tw.nb[0] as usize, tw.nb[1] as usize, tw.nb[2] as usize);
        let (nbd0, nbd1) = (td.nb[0] as usize, td.nb[1] as usize);

        let nr = n_embd * n_tokens;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i0 = ir % n_embd;
                let it = ir / n_embd;

                let mut sum = 0.0f32;
                for ih in 0..hc {
                    let xv = self.rd_f32s(sx, i0 * nbx0 + ih * nbx1 + it * nbx2, 1)[0];
                    let wv = if gated {
                        let gv = self.rd_f32s(sw, i0 * nbw0 + ih * nbw1 + it * nbw2, 1)[0];
                        1.0 / (1.0 + (-gv).exp())
                    } else {
                        self.rd_f32s(sw, ih * nbw0 + it * nbw1, 1)[0]
                    };
                    // `sum += xv * wv` — NOT contracted by gcc here (ops.cpp
                    // :11295 keeps a plain mul + add: the loop body reads
                    // through pointers and -O2 does not fuse across them)
                    sum += xv * wv;
                }

                self.wr_f32s(dst, i0 * nbd0 + it * nbd1, 1)[0] = scale * sum;
            }
        };
        self.par_rows(nr, nth, team, run);
    }

    /// `ggml_compute_forward_dsv4_hc_post_f32` (ops.cpp:11328-11402): per
    /// (i0, dst-stream, token), x*post[dst] + sum_src residual[src]*
    /// comb[dst,src] (or the identity mix when comb is absent). Threads split
    /// the flattened n_embd*hc*n_tokens range (ops.cpp:11381-11386).
    fn forward_dsv4_hc_post(&self, dst: TensorId, nth: usize, team: &Team<'_>) {
        let sx = self.t(dst).src[0].unwrap();
        let sr = self.t(dst).src[1].unwrap();
        let sp = self.t(dst).src[2].unwrap();
        let sc = self.t(dst).src[3];
        let (tx, tr, tp, td) = (self.t(sx), self.t(sr), self.t(sp), self.t(dst));

        assert_eq!(tx.ty, GgmlType::F32);
        assert_eq!(tr.ty, GgmlType::F32);
        assert_eq!(tp.ty, GgmlType::F32);
        assert_eq!(td.ty, GgmlType::F32);

        let n_embd = tx.ne[0] as usize;
        let n_tokens = tx.ne[1] as usize;
        let hc = tr.ne[1] as usize;

        assert_eq!(td.ne[0] as usize, n_embd);
        assert_eq!(td.ne[1] as usize, hc);
        assert_eq!(td.ne[2] as usize, n_tokens);
        assert_eq!(tr.ne[0] as usize, n_embd);
        assert_eq!(tr.ne[2] as usize, n_tokens);
        assert_eq!(tp.ne[0] as usize, hc);
        assert_eq!(tp.ne[1] as usize, n_tokens);

        // comb == NULL: identity mixing, each stream keeps its own residual
        // (ops.cpp:11345-11360)
        let mut nbc0 = 0usize;
        let mut nbc1 = 0usize;
        let mut nbc2 = 0usize;
        if let Some(sc) = sc {
            let tc = self.t(sc);
            assert_eq!(tc.ty, GgmlType::F32);
            assert_eq!(tc.ne[0] as usize, hc);
            assert_eq!(tc.ne[1] as usize, hc);
            assert_eq!(tc.ne[2] as usize, n_tokens);
            nbc0 = tc.nb[0] as usize;
            nbc1 = tc.nb[1] as usize;
            nbc2 = tc.nb[2] as usize;
        }

        let (nbx0, nbx1) = (tx.nb[0] as usize, tx.nb[1] as usize);
        let (nbr0, nbr1, nbr2) = (tr.nb[0] as usize, tr.nb[1] as usize, tr.nb[2] as usize);
        let (nbp0, nbp1) = (tp.nb[0] as usize, tp.nb[1] as usize);
        let (nbd0, nbd1, nbd2) = (td.nb[0] as usize, td.nb[1] as usize, td.nb[2] as usize);

        let nr = n_embd * hc * n_tokens;
        let run = |ir0: usize, ir1: usize| {
            for ir in ir0..ir1 {
                let i0 = ir % n_embd;
                let idst = (ir / n_embd) % hc;
                let it = ir / (n_embd * hc);

                let xv = self.rd_f32s(sx, i0 * nbx0 + it * nbx1, 1)[0];
                let pv = self.rd_f32s(sp, idst * nbp0 + it * nbp1, 1)[0];

                let mut sum = xv * pv;
                match sc {
                    Some(sc) => {
                        for isrc in 0..hc {
                            let rv = self.rd_f32s(sr, i0 * nbr0 + isrc * nbr1 + it * nbr2, 1)[0];
                            let cv = self.rd_f32s(sc, idst * nbc0 + isrc * nbc1 + it * nbc2, 1)[0];
                            // `sum += rv * cv` — also left uncontracted by gcc
                            // (ops.cpp:11390)
                            sum += rv * cv;
                        }
                    }
                    None => {
                        sum += self.rd_f32s(sr, i0 * nbr0 + idst * nbr1 + it * nbr2, 1)[0];
                    }
                }

                self.wr_f32s(dst, i0 * nbd0 + idst * nbd1 + it * nbd2, 1)[0] = sum;
            }
        };
        self.par_rows(nr, nth, team, run);
    }
}

/// ggml_row_size
#[inline]
fn ty_row_bytes(ty: GgmlType, n: usize) -> usize {
    ty.type_size() * n / ty.blck_size()
}

/// vec_dot_type for types whose traits entry lives outside vec_dot.rs — MXFP4
/// (and NVFP4) are registered in ggml-cpu.c:287-296 only (`vec_dot_type =
/// GGML_TYPE_Q8_0`, `vec_dot = ggml_vec_dot_mxfp4_q8_0`), so vec_dot_type()
/// returning None for them is expected, not a gap in the table.
#[inline]
fn mxfp4_vec_dot_type(ty: GgmlType) -> Option<GgmlType> {
    match ty {
        GgmlType::Mxfp4 => Some(GgmlType::Q8_0),
        _ => None,
    }
}

/// 对照 ggml-cpu/quants.c:298 `ggml_vec_dot_mxfp4_q8_0_generic` (QK_MXFP4 ==
/// QK8_0 == 32): per 32-element block, `d = fp16(y.d) * e8m0_half(x.e)` and an
/// exact integer sum of the 32 nibble products, accumulated into one f32.
///
/// Known divergence (same class as PARITY.md item 1): the reference build is
/// compiled with -DGGML_USE_CPU_REPACK and -march=native, and the server
/// measurably repacks the experts (gpt-oss-20b MXFP4 run: ~10 GiB anonymous RSS
/// next to ~9.8 GiB of mapped file pages), so its MoE gemms go through
/// ggml-cpu/repack.cpp's 8x8 outer-product kernels rather than a row-wise dot.
/// The integer sums agree; only the f32 accumulation order differs. The generic
/// scalar form (`ggml_vec_dot_mxfp4_q8_0_generic`) is ported here.
fn vec_dot_mxfp4_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    use crate::blocks::{BlockMxfp4, BlockQ8_0};
    let qk = 32usize;
    assert_eq!(n % qk, 0, "mxfp4 vec_dot: n % 32 != 0");
    assert!(x.len() >= n / qk * 17, "mxfp4 vec_dot: weight row bytes");
    assert!(y.len() >= n / qk * 34, "mxfp4 vec_dot: q8_0 row bytes");
    let xb: &[BlockMxfp4] = bytemuck::cast_slice(&x[..n / qk * 17]);
    let yb: &[BlockQ8_0] = bytemuck::cast_slice(&y[..n / qk * 34]);
    let nb = n / qk;

    let mut sumf = 0f32;
    for ib in 0..nb {
        let d = yb[ib].d.to_f32() * quants::e8m0_to_fp32_half(xb[ib].e);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..qk / 2 {
            sumi1 += yb[ib].qs[j] as i32
                * quants::KVALUES_MXFP4[(xb[ib].qs[j] & 0x0F) as usize] as i32;
            sumi2 += yb[ib].qs[j + qk / 2] as i32
                * quants::KVALUES_MXFP4[(xb[ib].qs[j] >> 4) as usize] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

/// contiguous check without a &Context (ops.rs needs &Context)
    #[inline]
    fn is_contiguous_ctx_of(cpu: &Cpu, id: TensorId) -> bool {
    let t = cpu.t(id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::*;
    use crate::tensor::GgmlOp;

    /// GGML_OP_TOP_K vs the reference's `std::partial_sort` on tie-heavy
    /// inputs — the oracle rows come from a g++ (libstdc++) build of the
    /// reference kernel body (ops.cpp:8550-8604: partial_sort + the
    /// [0]<->[1] swap), seeds/quantization chosen for massive value ties
    /// (deepseek32's ReLU-zeroed indexer scores behave exactly like this).
    /// The tie SET — not just the values — must match, because the DSA
    /// top-k mask unmasks exactly the selected rows.
    #[test]
    fn top_k_matches_libstdcpp_partial_sort_on_ties() {
        let rows: Vec<(&[f32], &[i32])> = vec![
            (
                &[-0.75, 0.25, 0.75, 0.50, -0.50, 0.75, 1.00, -1.00, 1.00, -0.50, -0.75, 0.25,
                    0.50, -0.75, -0.75, -0.75, 0.75, -0.50, 0.75],
                &[8, 6, 18, 16, 2, 5, 12, 3],
            ),
            (
                &[0.00, 1.00, -0.50, 0.50, 1.00, -0.75, -0.25, 0.75, -1.00, 0.50, 0.50, 0.50,
                    0.50, -0.75, 1.00, -0.25, 0.25, -0.25, 0.75],
                &[14, 4, 1, 7, 18, 3, 10, 12],
            ),
            (
                &[-0.50, 0.75, 0.50, 0.25, 0.00, 0.75, -0.75, 1.00, 0.00, 0.50, 0.00, -0.25,
                    -0.25, 1.00, -0.50, 1.00, 0.75, -0.75, 0.00],
                &[7, 15, 13, 1, 5, 16, 9, 2],
            ),
            (
                &[-1.00, -0.50, -0.50, -0.75, -0.25, -0.50, 0.50, -1.00, -0.25, 0.25, -1.00,
                    -0.50, -0.50, -0.50, 0.25, 0.25, 0.50, 0.75, -1.00],
                &[16, 17, 6, 9, 15, 14, 8, 4],
            ),
        ];
        let n = 19usize;
        let k = 8usize;
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, n as i64, rows.len() as i64);
        ctx.arena_resize_tensor(a);
        {
            let all: &mut [f32] = bytemuck::cast_slice_mut(ctx.data_bytes_mut(a).unwrap());
            for (r, (vals, _)) in rows.iter().enumerate() {
                all[r * n..(r + 1) * n].copy_from_slice(vals);
            }
        }
        let dst = ctx.top_k(a, k as i32);
        let mut g = Graph::new(8);
        g.build_forward(&mut ctx, dst);
        graph_compute(&mut ctx, &mut g, 1);
        let out: &[i32] = bytemuck::cast_slice(ctx.data_bytes(dst).unwrap());
        for (r, (_, want)) in rows.iter().enumerate() {
            let got = &out[r * k..(r + 1) * k];
            // the tie SET is what the DSA mask consumes (which rows get
            // unmasked); the within-tie ORDER only reorders set_rows writes
            let mut got_set: Vec<i32> = got.to_vec();
            let mut want_set: Vec<i32> = want.to_vec();
            got_set.sort_unstable();
            want_set.sort_unstable();
            assert_eq!(
                got_set, want_set,
                "top_k row {r}: libstdc++ partial_sort tie set mismatch"
            );
        }
    }

    fn lcg(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as i32 as f32 / (1u32 << 28) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// dequantize a q8_K row (y[i] = qs[i]*d) — quants.rs's dispatcher lacks Q8K
    fn deq_q8_k(bytes: &[u8], n: usize) -> Vec<f32> {
        use crate::blocks::{BlockQ8K, QK_K};
        let nb = n / QK_K;
        let blocks: &[BlockQ8K] = bytemuck::cast_slice(&bytes[..nb * std::mem::size_of::<BlockQ8K>()]);
        let mut out = vec![0f32; n];
        for (i, b) in blocks.iter().enumerate() {
            for j in 0..QK_K {
                out[i * QK_K + j] = b.qs[j] as f32 * b.d;
            }
        }
        out
    }

    // ===================== perf8 repro =====================

    /// Perf8 regression: backend.rs's process-wide tensor↔buffer side table
    /// used to key entries by the Context's *address*; a fresh Context at a
    /// dead Context's address inherited its stale entries, the gallocr then
    /// treated its tensors as externally allocated (`is_allocated`'s
    /// `tensor_buffer(..).is_some()`), never reserved room, and
    /// `init_tensor` indexed buffer_id −1. Serial `--test-threads 1` hit it
    /// as backend_sched's sched_cpu_bit_identity → sched_repeated_compute
    /// (stack-slot reuse). The fix keys by Context::uid (monotonic); this
    /// test re-creates the exact aliasing pressure: build + drop a sched-using
    /// Context, then re-try the two-round sched loop on fresh Contexts until
    /// the allocator would have handed back the recycled address.
    #[cfg(test)]
    mod perf8_repro {
        use crate::backend::cpu_backend_init;
        use crate::backend_sched::{backend_sched_graph_compute, backend_sched_new};
        use crate::backend::{GgmlStatus, GGML_TENSOR_FLAG_INPUT, GGML_TENSOR_FLAG_OUTPUT};
        use crate::tensor::Context;
        use crate::types::GgmlType;
        use crate::Graph;

        fn fill_f32(ctx: &mut Context, t: crate::tensor::TensorId, f: impl Fn(usize) -> f32) {
            let n = ctx.nbytes(t) / 4;
            let vals: Vec<f32> = (0..n).map(f).collect();
            ctx.with_f32_mut(t, |p| p.copy_from_slice(&vals)).unwrap();
        }

        /// same shape as backend_sched::tests::build_toy
        fn build_toy(ctx: &mut Context) -> crate::tensor::TensorId {
            let w = ctx.new_tensor_2d(GgmlType::F32, 8, 4);
            ctx.arena_resize_tensor(w);
            fill_f32(ctx, w, |i| (i as f32 * 0.25 - 1.0).sin());
            let x = ctx.new_tensor_2d(GgmlType::F32, 8, 2);
            ctx.arena_resize_tensor(x);
            fill_f32(ctx, x, |i| (i as f32 * 0.5 - 3.0).cos());
            ctx.tensors[x.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;
            let bias = ctx.new_tensor_2d(GgmlType::F32, 2, 1);
            ctx.arena_resize_tensor(bias);
            fill_f32(ctx, bias, |i| 0.1 * i as f32);
            ctx.tensors[bias.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;
            let mm = ctx.mul_mat(w, x);
            let ad = ctx.add(mm, bias);
            let si = ctx.silu(ad);
            let out = ctx.mul(si, ad);
            ctx.tensors[out.0 as usize].flags |= GGML_TENSOR_FLAG_OUTPUT;
            out
        }

        #[test]
        fn direct_then_sched_repeated() {
            // half 1: sched_cpu_bit_identity's direct compute (4 threads)
            let mut ctx_d = Context::new();
            let out_d = build_toy(&mut ctx_d);
            let mut g = Graph::new(16);
            g.build_forward(&ctx_d, out_d);
            crate::compute::graph_compute(&mut ctx_d, &mut g, 4);
            eprintln!("[perf8] direct half done");

            // half 1b: sched_cpu_bit_identity's scheduler half
            let mut ctx_s = Context::new();
            let out_s = build_toy(&mut ctx_s);
            let mut g1 = Graph::new(16);
            g1.build_forward(&ctx_s, out_s);
            let cpu0 = cpu_backend_init();
            let mut sched0 = backend_sched_new(&[cpu0], None, 64, false, true);
            let st0 = backend_sched_graph_compute(&mut sched0, &mut ctx_s, &mut g1);
            assert_eq!(st0, GgmlStatus::Success);
            drop(sched0);
            drop(ctx_s);
            drop(g1);
            drop(g);
            drop(ctx_d);
            eprintln!("[perf8] bit-identity half done (all dropped)");

            // half 2: sched_repeated_compute's two-round loop
            // The suspected mechanism: backend.rs's process-wide
            // tensor_buffers side table is keyed by (Context ADDRESS,
            // TensorId). A fresh Context allocated at a dead Context's
            // address inherits its stale buffer registrations, so the
            // gallocr treats its tensors as externally allocated, never
            // reserves room for them, and init_tensor then indexes
            // buffer_id -1. Retry with fresh contexts until the allocator
            // hands back a recycled address to prove it.
            let mut panicked_at: Option<usize> = None;
            for attempt in 0..64 {
                let r = std::panic::catch_unwind(|| {
                    let mut ctx = Context::new();
                    let cpu = cpu_backend_init();
                    let mut sched = backend_sched_new(&[cpu], None, 64, false, true);
                    let mut expected: Vec<Vec<u8>> = Vec::new();
                    for round in 0..2 {
                        let mark = ctx.mark();
                        let out = build_toy(&mut ctx);
                        let mut g2 = Graph::new(16);
                        g2.build_forward(&ctx, out);
                        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g2);
                        assert_eq!(st, GgmlStatus::Success);
                        expected.push(ctx.data_bytes(out).unwrap().to_vec());
                        ctx.reset_graph_to(mark);
                    }
                    assert_eq!(expected[0], expected[1]);
                });
                if r.is_err() {
                    panicked_at = Some(attempt);
                    break;
                }
            }
            assert!(
                panicked_at.is_none(),
                "a recycled Context address corrupted the gallocr (attempt {:?})",
                panicked_at
            );
        }
    }

    // ===================== mul_mat =====================

    /// Port-side twin of `parity/ref_mulmat_bench.c` (same shapes, same
    /// quantized-random weights, same warm-up + best-of-N protocol) so the
    /// two columns of a per-op pp64 comparison always come from the same
    /// machine window. `cargo test --release -p ggml --lib mulmat_throughput
    /// -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn mulmat_throughput() {
        use std::time::Instant;
        struct Shape {
            ty: GgmlType,
            quantize: fn(&[f32], &mut [u8]),
            k: usize,
            m: usize,
            ncols: usize,
            name: &'static str,
        }
        let nth: usize = std::env::var("GGML_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let shapes = [
            Shape { ty: GgmlType::Q5_0, quantize: |x, o| crate::quants::quantize_row_q5_0_ref(x, bytemuck::cast_slice_mut(o)), k: 896, m: 4864, ncols: 64, name: "ffn_gate/up Q5_0 4864x896 x64" },
            Shape { ty: GgmlType::Q5_0, quantize: |x, o| crate::quants::quantize_row_q5_0_ref(x, bytemuck::cast_slice_mut(o)), k: 896, m: 896, ncols: 64, name: "attn_q/o    Q5_0  896x896 x64" },
            Shape { ty: GgmlType::Q5_0, quantize: |x, o| crate::quants::quantize_row_q5_0_ref(x, bytemuck::cast_slice_mut(o)), k: 896, m: 128, ncols: 64, name: "attn_k/v    Q5_0  128x896 x64" },
            Shape { ty: GgmlType::Q6K, quantize: |x, o| crate::quants_k::quantize_row_q6_K_ref(x, bytemuck::cast_slice_mut(o)), k: 4864, m: 896, ncols: 64, name: "ffn_down    Q6_K  896x4864 x64 (row vec_dot)" },
            Shape { ty: GgmlType::Q4K, quantize: |x, o| crate::quants_k::quantize_row_q4_K_ref(x, bytemuck::cast_slice_mut(o)), k: 4864, m: 896, ncols: 64, name: "ffn_down    Q4_K  896x4864 x64 (repack gemm)" },
            Shape { ty: GgmlType::Q8_0, quantize: |x, o| crate::quants::quantize_row_q8_0(x, bytemuck::cast_slice_mut(o)), k: 896, m: 151936, ncols: 1, name: "lm_head     Q8_0 151936x896 x1 (gemv)" },
            Shape { ty: GgmlType::Q8_0, quantize: |x, o| crate::quants::quantize_row_q8_0(x, bytemuck::cast_slice_mut(o)), k: 896, m: 151936, ncols: 64, name: "lm_head     Q8_0 151936x896 x64 (llamafile)" },
        ];
        for sh in &shapes {
            let (k, m, ncols) = (sh.k, sh.m, sh.ncols);
            let mut ctx = Context::new();
            let w = ctx.new_tensor_2d(sh.ty, k as i64, m as i64);
            ctx.arena_resize_tensor(w);
            let rs = ty_row_bytes(sh.ty, k);
            {
                let bytes = ctx.data_bytes_mut(w).unwrap();
                for r in 0..m {
                    let row = lcg(k, 7000 + r as u32);
                    (sh.quantize)(&row, &mut bytes[r * rs..(r + 1) * rs]);
                }
            }
            let x = ctx.new_tensor_2d(GgmlType::F32, k as i64, ncols as i64);
            ctx.arena_resize_tensor(x);
            ctx.with_f32_mut(x, |p| {
                for c in 0..ncols {
                    let row = lcg(k, 9000 + c as u32);
                    p[c * k..(c + 1) * k].copy_from_slice(&row);
                }
            })
            .unwrap();
            // one graph with REPS copies of the node: the 7 scoped workers
            // are spawned once per graph_compute, exactly like a real forward
            // (a single-node graph would time thread spawn, not the op —
            // ~100 µs, which swamps the small shapes)
            // cap the arena: the lm_head x64 rows (151936x64 f32 = 39 MB per
            // node) would need 1.5 GB at 40 reps
            let reps: usize = if m as u64 * ncols as u64 > 4_000_000 { 4 } else { 40 };
            let mut ys = Vec::with_capacity(reps);
            for _ in 0..reps {
                ys.push(ctx.mul_mat(w, x));
            }
            let mut g = Graph::new(reps + 8);
            for &y in &ys {
                g.build_forward(&ctx, y);
            }
            graph_compute(&mut ctx, &mut g, nth); // warm-up (repack, faults)
            let iters = if m > 100_000 { 20 } else { 60 };
            let mut best = f64::MAX;
            for _ in 0..iters {
                let t = Instant::now();
                graph_compute(&mut ctx, &mut g, nth);
                best = best.min(t.elapsed().as_secs_f64() / reps as f64 * 1e3);
            }
            let gfl = 2.0 * m as f64 * ncols as f64 * k as f64 / 1e9;
            println!(
                "{:.<44} t={:2} {:8.3} ms/call {:8.1} GF/s",
                sh.name,
                nth,
                best,
                gfl / (best / 1e3)
            );
        }
    }

    /// Quantized mul_mat must equal (bit-for-bit) the integer vec_dot kernels
    /// applied row by row, and stay within 1e-5 of the naive f64 matmul over
    /// dequantized weights.
    #[test]
    fn mul_mat_quantized_all_types() {
        struct Case {
            ty: GgmlType,
            quantize: fn(&[f32], &mut [u8]),
            rows: usize,
        }
        let cases = [
            // Q4_0 with `ne[1] % 8 == 0` is intercepted by the CPU_REPACK 8x8
            // traits before llamafile (ggml-cpu.c:1751) by the reference and by
            // this port, so the row-wise kernel is only the production path for
            // geometries the repack gate rejects — `rows = 31` exercises
            // exactly that. The repack route is pinned bit-exactly by
            // `repack::q4_0_ref_tests` (`q4_0_repack_ref.bin`) and by the
            // `mul_mat_q4_0_repack_wiring…` graph test.
            Case { ty: GgmlType::Q4_0, quantize: |x, o| crate::quants::quantize_row_q4_0_ref(x, bytemuck::cast_slice_mut(o)), rows: 31 },
            Case { ty: GgmlType::Q5_0, quantize: |x, o| crate::quants::quantize_row_q5_0_ref(x, bytemuck::cast_slice_mut(o)), rows: 32 },
            Case { ty: GgmlType::Q8_0, quantize: |x, o| crate::quants::quantize_row_q8_0(x, bytemuck::cast_slice_mut(o)), rows: 32 },
            Case { ty: GgmlType::Q2K, quantize: |x, o| crate::quants_k::quantize_row_q2_K_ref(x, bytemuck::cast_slice_mut(o)), rows: 32 },
            Case { ty: GgmlType::Q3K, quantize: |x, o| crate::quants_k::quantize_row_q3_K_ref(x, bytemuck::cast_slice_mut(o)), rows: 32 },
            // Q4_K with `ne[1] % 8 == 0` is routed to the CPU_REPACK 8x8 kernels
            // by the reference (repack.cpp:5006) and by this port, so the
            // row-wise kernel is only the production path for geometries the
            // repack gate rejects — `rows = 31` exercises exactly that. The
            // repack route is pinned bit-exactly by `repack::q4k_ref_tests`
            // (`q4k_repack_ref.bin`) and by `kquant_real_tensor_tests` on real
            // GGUF tensors.
            Case { ty: GgmlType::Q4K, quantize: |x, o| crate::quants_k::quantize_row_q4_K_ref(x, bytemuck::cast_slice_mut(o)), rows: 31 },
            Case { ty: GgmlType::Q5K, quantize: |x, o| crate::quants_k::quantize_row_q5_K_ref(x, bytemuck::cast_slice_mut(o)), rows: 32 },
            Case { ty: GgmlType::Q6K, quantize: |x, o| crate::quants_k::quantize_row_q6_K_ref(x, bytemuck::cast_slice_mut(o)), rows: 32 },
        ];

        let n = 256usize;
        let cols = 5usize;

        for case in &cases {
            let rows = case.rows;
            let mut ctx = Context::new();
            let w = ctx.new_tensor_2d(case.ty, n as i64, rows as i64);
            ctx.arena_resize_tensor(w);
            // random weights, quantized row by row
            let wbytes = {
                let bytes = ctx.data_bytes_mut(w).unwrap();
                for r in 0..rows {
                    let row = lcg(n, 1000 + r as u32);
                    let rs = ty_row_bytes(case.ty, n);
                    (case.quantize)(&row, &mut bytes[r * rs..(r + 1) * rs]);
                }
                bytes.to_vec()
            };

            let x = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
            ctx.arena_resize_tensor(x);
            let xrows: Vec<Vec<f32>> = (0..cols).map(|c| lcg(n, 2000 + c as u32)).collect();
            ctx.with_f32_mut(x, |p| {
                for c in 0..cols {
                    p[c * n..(c + 1) * n].copy_from_slice(&xrows[c]);
                }
            })
            .unwrap();

            let y = ctx.mul_mat(w, x);
            let mut g = Graph::new(16);
            g.build_forward(&ctx, y);
            graph_compute(&mut ctx, &mut g, 4);

            // dequantized weights for the naive reference
            let mut wdeq = vec![0f32; rows * n];
            for r in 0..rows {
                let rs = ty_row_bytes(case.ty, n);
                quants::dequantize_row(case.ty, &wbytes[r * rs..(r + 1) * rs], &mut wdeq[r * n..(r + 1) * n]);
            }

            let vdt = vec_dot::vec_dot_type(case.ty).unwrap();
            for c in 0..cols {
                // converted activation row (same path mul_mat uses)
                let mut yrow_q = vec![0u8; ty_row_bytes(vdt, n)];
                match vdt {
                    GgmlType::Q8_0 => quants::quantize_row_q8_0(&xrows[c], bytemuck::cast_slice_mut(&mut yrow_q)),
                    GgmlType::Q8K => quants::quantize_row_q8_K(&xrows[c], bytemuck::cast_slice_mut(&mut yrow_q)),
                    _ => unreachable!(),
                }
                // dequantized activation row: the values the kernel actually multiplies
                let yrow_f = match vdt {
                    GgmlType::Q8_0 => {
                        let mut v = vec![0f32; n];
                        quants::dequantize_row(vdt, &yrow_q, &mut v);
                        v
                    }
                    _ => deq_q8_k(&yrow_q, n),
                };
                for r in 0..rows {
                    let got = ctx.f32s(y).unwrap()[c * rows + r];
                    // bit-exact vs the kernel itself
                    let want = vec_dot::vec_dot_row(case.ty, n, &wbytes[r * ty_row_bytes(case.ty, n)..], &yrow_q);
                    assert_eq!(got.to_bits(), want.to_bits(), "{:?} r={r} c={c} kernel mismatch", case.ty);
                    // naive f64 matmul over dequantized weights × dequantized activations
                    // (only float re-association differs); compare against the L1
                    // product scale so heavy cancellation doesn't inflate the error
                    let mut refv = 0f64;
                    let mut scale = 0f64;
                    for j in 0..n {
                        let p = wdeq[r * n + j] as f64 * yrow_f[j] as f64;
                        refv += p;
                        scale += p.abs();
                    }
                    let rel = (got as f64 - refv).abs() / scale.max(1e-9);
                    assert!(rel < 1e-4, "{:?} r={r} c={c}: got={got} naive={refv} rel={rel}", case.ty);
                }
            }
        }
    }

    /// F32×F32 mul_mat: f64 accumulation, tolerance 1e-4 relative.
    #[test]
    fn mul_mat_f32() {
        let n = 896usize;
        let rows = 64usize;
        let cols = 3usize;
        let mut ctx = Context::new();
        let w = ctx.new_tensor_2d(GgmlType::F32, n as i64, rows as i64);
        ctx.arena_resize_tensor(w);
        let wrows: Vec<Vec<f32>> = (0..rows).map(|r| lcg(n, 3000 + r as u32)).collect();
        ctx.with_f32_mut(w, |p| {
            for r in 0..rows {
                p[r * n..(r + 1) * n].copy_from_slice(&wrows[r]);
            }
        })
        .unwrap();

        let x = ctx.new_tensor_2d(GgmlType::F32, n as i64, cols as i64);
        ctx.arena_resize_tensor(x);
        let xrows: Vec<Vec<f32>> = (0..cols).map(|c| lcg(n, 4000 + c as u32)).collect();
        ctx.with_f32_mut(x, |p| {
            for c in 0..cols {
                p[c * n..(c + 1) * n].copy_from_slice(&xrows[c]);
            }
        })
        .unwrap();

        let y = ctx.mul_mat(w, x);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 4);

        for c in 0..cols {
            for r in 0..rows {
                let mut refv = 0f64;
                for j in 0..n {
                    refv += wrows[r][j] as f64 * xrows[c][j] as f64;
                }
                let got = ctx.f32s(y).unwrap()[c * rows + r];
                let rel = ((got as f64 - refv) / refv.abs().max(1.0)).abs();
                assert!(rel < 1e-4, "r={r} c={c} got={got} ref={refv}");
            }
        }
    }

    /// Strided weight view: rows addressed through nb (non-contiguous src0).
    #[test]
    fn mul_mat_q6_k_view_rows() {
        let n = 256usize;
        let big_rows = 8usize;
        let mut ctx = Context::new();
        let wbig = ctx.new_tensor_3d(GgmlType::Q6K, n as i64, big_rows as i64, 2);
        ctx.arena_resize_tensor(wbig);
        {
            let bytes = ctx.data_bytes_mut(wbig).unwrap();
            let rs = ty_row_bytes(GgmlType::Q6K, n);
            for i in 0..big_rows * 2 {
                let row = lcg(n, 5000 + i as u32);
                crate::quants_k::quantize_row_q6_K_ref(&row, bytemuck::cast_slice_mut(&mut bytes[i * rs..(i + 1) * rs]));
            }
        }
        // w = view of the i3=1 slice -> src0 is a view with nb03 stride
        let w = ctx.view_3d(wbig, n as i64, big_rows as i64, 1, ty_row_bytes(GgmlType::Q6K, n), ty_row_bytes(GgmlType::Q6K, n) * big_rows, ty_row_bytes(GgmlType::Q6K, n) * big_rows);
        let x = ctx.new_tensor_2d(GgmlType::F32, n as i64, 2);
        ctx.arena_resize_tensor(x);
        let xrows: Vec<Vec<f32>> = (0..2).map(|c| lcg(n, 6000 + c as u32)).collect();
        ctx.with_f32_mut(x, |p| {
            for c in 0..2 {
                p[c * n..(c + 1) * n].copy_from_slice(&xrows[c]);
            }
        })
        .unwrap();

        let y = ctx.mul_mat(w, x);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 4);

        // reference: dequantize the second slice's rows
        let wbytes = ctx.data_bytes(w).unwrap().to_vec();
        let rs = ty_row_bytes(GgmlType::Q6K, n);
        let mut wdeq = vec![0f32; big_rows * n];
        for r in 0..big_rows {
            quants::dequantize_row(GgmlType::Q6K, &wbytes[r * rs..(r + 1) * rs], &mut wdeq[r * n..(r + 1) * n]);
        }
        for c in 0..2 {
            // the kernel quantizes activations to q8_K; mirror that in the reference
            let mut yq = vec![0u8; ty_row_bytes(GgmlType::Q8K, n)];
            quants::quantize_row_q8_K(&xrows[c], bytemuck::cast_slice_mut(&mut yq));
            let yf = deq_q8_k(&yq, n);
            for r in 0..big_rows {
                let mut refv = 0f64;
                let mut scale = 0f64;
                for j in 0..n {
                    let p = wdeq[r * n + j] as f64 * yf[j] as f64;
                    refv += p;
                    scale += p.abs();
                }
                let got = ctx.f32s(y).unwrap()[c * big_rows + r];
                let rel = (got as f64 - refv).abs() / scale.max(1e-9);
                assert!(rel < 1e-4);
            }
        }
    }

    // ===================== rms_norm / norm / softmax =====================

    #[test]
    fn rms_norm_hand_values() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| {
            p.copy_from_slice(&[3.0, 4.0, 0.0, 0.0, -1.0, 2.0, -2.0, 1.0]);
        })
        .unwrap();
        let y = ctx.rms_norm(a, 1e-6);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 1);

        let out = ctx.f32s(y).unwrap();
        // row 0: sum sq = 25, mean = (25/4 as f64) as f32, scale = 1/sqrt(mean+eps)
        let s0 = 1.0f32 / ((25.0f64 / 4.0) as f32 + 1e-6).sqrt();
        assert_eq!(out[0].to_bits(), (3.0 * s0).to_bits());
        assert_eq!(out[1].to_bits(), (4.0 * s0).to_bits());
        // row 1: sum sq = 1 + 4 + 4 + 1 = 10
        let s1 = 1.0f32 / ((10.0f64 / 4.0) as f32 + 1e-6).sqrt();
        assert_eq!(out[4].to_bits(), (-1.0 * s1).to_bits());
        assert_eq!(out[7].to_bits(), (1.0 * s1).to_bits());
    }

    #[test]
    fn softmax_sums_to_one_with_mask_and_scale() {
        let mut ctx = Context::new();
        let n = 8usize;
        let a = ctx.new_tensor_2d(GgmlType::F32, n as i64, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = ((i as f32) * 0.7).sin();
            }
        })
        .unwrap();
        // F32 mask, ne1 == a ne1 (broadcast dims equal)
        let mask = ctx.new_tensor_2d(GgmlType::F32, n as i64, 2);
        ctx.arena_resize_tensor(mask);
        ctx.with_f32_mut(mask, |p| {
            p.fill(0.0);
            p[3] = f32::NEG_INFINITY;
            p[n + 5] = f32::NEG_INFINITY;
        })
        .unwrap();

        let scale = 0.5f32;
        let y = ctx.soft_max_ext(a, Some(mask), scale, 0.0);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 4);

        let out = ctx.f32s(y).unwrap();
        for row in 0..2 {
            let mut sum = 0f64;
            for i in 0..n {
                sum += out[row * n + i] as f64;
            }
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum {sum}");
        }
        assert_eq!(out[3], 0.0);
        assert_eq!(out[n + 5], 0.0);

        // naive reference for row 0
        let a0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).sin()).collect();
        let mut wp: Vec<f32> = a0.iter().map(|v| v * scale).collect();
        wp[3] = f32::NEG_INFINITY;
        let m = wp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0f64;
        let mut expected = vec![0f32; n];
        for i in 0..n {
            if wp[i].is_finite() {
                expected[i] = ggml_expf(wp[i] - m);
                denom += expected[i] as f64;
            }
        }
        for v in expected.iter_mut() {
            *v /= denom as f32;
        }
        for i in 0..n {
            let rel = ((out[i] - expected[i]) / expected[i].max(1e-30)).abs();
            assert!(rel < 1e-5, "i={i} {} vs {}", out[i], expected[i]);
        }
    }

    #[test]
    fn norm_matches_naive() {
        let mut ctx = Context::new();
        let n = 16usize;
        let a = ctx.new_tensor_2d(GgmlType::F32, n as i64, 3);
        ctx.arena_resize_tensor(a);
        let rows: Vec<Vec<f32>> = (0..3).map(|r| lcg(n, 7000 + r as u32)).collect();
        ctx.with_f32_mut(a, |p| {
            for r in 0..3 {
                p[r * n..(r + 1) * n].copy_from_slice(&rows[r]);
            }
        })
        .unwrap();
        let y = ctx.norm(a, 1e-6);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 2);

        let out = ctx.f32s(y).unwrap();
        for r in 0..3 {
            // f64 mean/variance reference mirroring the C accumulation order
            let mut sum = 0f64;
            for &v in &rows[r] {
                sum += v as f64;
            }
            let mean = (sum as f32) / n as f32;
            let mut vs = 0f64;
            let mut centered = vec![0f32; n];
            for (j, &v) in rows[r].iter().enumerate() {
                let c = v - mean;
                centered[j] = c;
                vs += (c * c) as f64;
            }
            let scale = 1.0f32 / ((vs / n as f64 + 1e-6) as f32).sqrt();
            for j in 0..n {
                let want = centered[j] * scale;
                let rel = ((out[r * n + j] - want) / want.abs().max(1e-6)).abs();
                assert!(rel < 1e-6, "r={r} j={j}");
            }
        }
    }

    // ===================== rope =====================

    /// Task hand case: n_dims=4, pos=1, freq_base=10000, x=[1,0,-1,0], NEOX.
    #[test]
    fn rope_neox_hand_case() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(GgmlType::F32, 4, 1, 1); // [4, 1 head, 1 pos]
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&[1.0, 0.0, -1.0, 0.0])).unwrap();
        let pos = ctx.new_tensor_1d(GgmlType::I32, 1);
        ctx.arena_resize_tensor(pos);
        ctx.with_i32_mut(pos, |p| p[0] = 1).unwrap();

        let y = ctx.rope_ext(a, pos, None, 4, GGML_ROPE_TYPE_NEOX, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 1);

        let out = ctx.f32s(y).unwrap();
        // pair (0,2): theta = 1 * 10000^0 = 1 rad
        let c1 = 1.0f32.cos();
        let s1 = 1.0f32.sin();
        // pair (1,3): theta = 1 * 10000^(-2/4) = 0.01 rad
        let c2 = 0.01f32.cos();
        let s2 = 0.01f32.sin();
        assert!((out[0] - (1.0 * c1 - (-1.0) * s1)).abs() < 1e-6, "out0={}", out[0]);
        assert!((out[1] - (0.0 * c2 - 0.0 * s2)).abs() < 1e-6);
        assert!((out[2] - (1.0 * s1 + (-1.0) * c1)).abs() < 1e-6, "out2={}", out[2]);
        assert!((out[3] - (0.0 * s2 + 0.0 * c2)).abs() < 1e-6);
        // pair modulus preserved
        let m_in = (1.0f64 * 1.0 + (-1.0) * (-1.0)).sqrt();
        let m_out = (out[0] as f64 * out[0] as f64 + out[2] as f64 * out[2] as f64).sqrt();
        assert!((m_in - m_out).abs() < 1e-6);
    }

    /// NEOX, mode=2, freq_base=1e6, n_dims=64: every pair keeps its modulus.
    #[test]
    fn rope_neox_preserves_pair_norms() {
        let n_dims = 64usize;
        let heads = 3usize;
        let seq = 2usize;
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(GgmlType::F32, n_dims as i64, heads as i64, seq as i64);
        ctx.arena_resize_tensor(a);
        let x0 = lcg(n_dims * heads * seq, 777);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&x0)).unwrap();
        let pos = ctx.new_tensor_1d(GgmlType::I32, seq as i64);
        ctx.arena_resize_tensor(pos);
        ctx.with_i32_mut(pos, |p| {
            p[0] = 13;
            p[1] = 41;
        })
        .unwrap();

        let y = ctx.rope_ext(a, pos, None, n_dims as i32, GGML_ROPE_TYPE_NEOX, 4096, 1_000_000.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 4);

        let out = ctx.f32s(y).unwrap();
        for h in 0..heads {
            for s in 0..seq {
                let base = (s * heads + h) * n_dims;
                for i in 0..n_dims / 2 {
                    let mi = ((x0[base + i] as f64).powi(2) + (x0[base + i + n_dims / 2] as f64).powi(2)).sqrt();
                    let mo = ((out[base + i] as f64).powi(2) + (out[base + i + n_dims / 2] as f64).powi(2)).sqrt();
                    assert!((mi - mo).abs() < 1e-4 * mi.max(1.0), "h={h} s={s} i={i} {mi} vs {mo}");
                }
            }
        }
    }

    // ===================== elementwise / misc ops =====================

    #[test]
    fn add_mul_broadcast_and_inplace() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 8, 4);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&lcg(32, 11))).unwrap();
        // row vector broadcast (ne=[8,1]) across all 4 rows
        let b = ctx.new_tensor_2d(GgmlType::F32, 8, 1);
        ctx.arena_resize_tensor(b);
        ctx.with_f32_mut(b, |p| p.copy_from_slice(&lcg(8, 12))).unwrap();

        let y = ctx.add(a, b);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 4);
        let out = ctx.f32s(y).unwrap();
        for r in 0..4 {
            for i in 0..8 {
                let want = ctx.f32s(a).unwrap()[r * 8 + i] + ctx.f32s(b).unwrap()[i];
                assert_eq!(out[r * 8 + i].to_bits(), want.to_bits());
            }
        }

        // inplace mul
        let y2 = ctx.mul_inplace(a, b);
        let mut g2 = Graph::new(8);
        g2.build_forward(&ctx, y2);
        graph_compute(&mut ctx, &mut g2, 4);
        let out2 = ctx.f32s(a).unwrap(); // writes into a
        for r in 0..4 {
            for i in 0..8 {
                let orig = lcg(32, 11)[r * 8 + i] * lcg(8, 12)[i];
                assert_eq!(out2[r * 8 + i].to_bits(), orig.to_bits());
            }
        }
    }

    #[test]
    fn scale_silu_gelu_work() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 16, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&lcg(32, 21))).unwrap();

        let s = ctx.scale(a, -2.5);
        let sil = ctx.silu(s);
        let g = ctx.gelu(sil);
        let mut g0 = Graph::new(8);
        g0.build_forward(&ctx, g);
        graph_compute(&mut ctx, &mut g0, 2);

        let scaled = ctx.f32s(s).unwrap();
        let silud = ctx.f32s(sil).unwrap();
        let gelud = ctx.f32s(g).unwrap();
        let x0 = lcg(32, 21);
        for i in 0..32 {
            assert_eq!(scaled[i].to_bits(), (x0[i] * -2.5).to_bits());
            assert_eq!(silud[i].to_bits(), ggml_silu_f32(scaled[i]).to_bits());
            assert_eq!(gelud[i].to_bits(), ggml_vec_gelu_f32(silud[i]).to_bits());
        }
    }

    // ==================================================================
    // tanh / gelu bit-exact parity vs the reference build
    // (parity/ref_tanh_dump.c → parity/tanh_ref.bin)
    // ==================================================================

    /// One section of parity/tanh_ref.bin: kind 0/1/3/4 carry n f32 inputs +
    /// n f32 outputs; kind 5 is the 65536-entry gelu f16 table (u16 payload).
    enum TanhSection {
        F32 { kind: u32, input: Vec<f32>, output: Vec<f32> },
        F16 { input: Vec<u16>, output: Vec<u16> },
        GeluTable(Vec<u16>),
    }

    fn read_tanh_dump() -> Vec<TanhSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/tanh_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_tanh_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x3148_4E54, "section magic 'TNH1'");
            let kind = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            if kind == 5 {
                let (tab, rest) = c.split_at(n * 2);
                c = rest;
                out.push(TanhSection::GeluTable(
                    tab.chunks_exact(2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect(),
                ));
                continue;
            }
            if kind == 7 {
                let (io, rest) = c.split_at(n * 4);
                c = rest;
                let u = |b: &[u8]| -> Vec<u16> {
                    b.chunks_exact(2).map(|w| u16::from_le_bytes(w.try_into().unwrap())).collect()
                };
                out.push(TanhSection::F16 { input: u(&io[..n * 2]), output: u(&io[n * 2..]) });
                continue;
            }
            let (io, rest) = c.split_at(n * 8);
            c = rest;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect()
            };
            out.push(TanhSection::F32 {
                kind,
                input: f(&io[..n * 4]),
                output: f(&io[n * 4..]),
            });
        }
        out
    }

    /// Build `op(a)` through the real graph path (like the C dumper did) and
    /// return the output bits.
    fn run_unary(tanh: bool, x: &[f32]) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F32, x.len() as i64);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(x)).unwrap();
        let r = if tanh { ctx.tanh(a) } else { ctx.gelu(a) };
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 3);
        ctx.f32s(r).unwrap().to_vec()
    }

    /// tanh: ggml_tanh op output must match the reference bit-for-bit, at every
    /// length (incl. non-16-multiples 63/1000/1023/17/1/15/47) and on edge
    /// cases (inf/nan/denormals/signed zero). The reference has no SIMD tanh —
    /// kind 0 must also equal the raw glibc tanhf dump (kind 3).
    #[test]
    fn tanh_bit_exact_vs_reference() {
        let secs = read_tanh_dump();
        let mut n_tanh = 0usize;
        let mut n_checked_raw = 0usize;
        // section order in the dumper: tanh, gelu, raw tanhf, raw gelu poly
        let mut pending: Option<(Vec<f32>, Vec<f32>)> = None; // (input, raw tanhf)
        for s in &secs {
            let TanhSection::F32 { kind, input, output } = s else { continue };
            match kind {
                0 => {
                    let mine = run_unary(true, input);
                    assert_eq!(mine.len(), output.len());
                    for (i, (m, r)) in mine.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "tanh mismatch at n={} idx={} x={:e} (rust {:e} vs ref {:e})",
                            input.len(), i, input[i], m, r
                        );
                    }
                    n_tanh += input.len();
                    pending = Some((input.clone(), Vec::new()));
                }
                1 => {
                    let mine = run_unary(false, input);
                    for (i, (m, r)) in mine.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "gelu mismatch at n={} idx={} x={:e} (rust {:e} vs ref {:e})",
                            input.len(), i, input[i], m, r
                        );
                    }
                }
                3 => {
                    // raw glibc tanhf — must equal the op output we just checked
                    if let Some((input_op, _)) = pending.take() {
                        // NB: compare bits, not floats (the sweep contains NaNs)
                        assert_eq!(input_op.len(), input.len(), "kind 3 input length");
                        for (i, (a, b)) in input_op.iter().zip(input).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(), "kind3 input idx={i}");
                        }
                        let mine: Vec<f32> = input.iter().map(|v| ggml_tanh_f32(*v)).collect();
                        for (i, (m, r)) in mine.iter().zip(output).enumerate() {
                            assert_eq!(m.to_bits(), r.to_bits(), "raw tanhf mismatch idx={i}");
                        }
                        n_checked_raw += input.len();
                    }
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        assert_eq!(n_tanh, 8143, "full length sweep (7590 LCG + 512 scaled + 41 specials)");
        assert_eq!(n_checked_raw, n_tanh, "raw tanhf cross-check count");
    }

    /// tanh on an F16 tensor (ops.cpp:10162 dispatch → apply_unary_op<op_tanh,
    /// ggml_fp16_t, ggml_fp16_t>): fp16(tanhf(fp16_to_f32(x))). Covers subnormal
    /// f16 outputs (tiny x), ±0, ±inf, NaN payloads and saturation.
    #[test]
    fn tanh_f16_bit_exact_vs_reference() {
        let secs = read_tanh_dump();
        let mut secs = secs.iter().filter_map(|s| match s {
            TanhSection::F16 { input, output } => Some((input, output)),
            _ => None,
        });
        let (input, output) = secs.next().expect("dump has an F16 tanh section");

        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F16, input.len() as i64);
        ctx.arena_resize_tensor(a);
        {
            let bytes = ctx.data_bytes_mut(a).unwrap();
            let dst: &mut [f16] = bytemuck::cast_slice_mut(bytes);
            for (i, h) in input.iter().enumerate() {
                dst[i] = f16::from_bits(*h);
            }
        }
        let r = ctx.tanh(a);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 2);
        let got_bytes = ctx.data_bytes(r).unwrap();
        let got: &[f16] = bytemuck::cast_slice(got_bytes);
        assert_eq!(got.len(), output.len());
        for (i, (g, r)) in got.iter().zip(output).enumerate() {
            assert_eq!(
                g.to_bits(),
                *r,
                "f16 tanh mismatch at {i}: in {:04x} rust {:04x} ref {:04x}",
                input[i],
                g.to_bits(),
                *r
            );
        }
    }

    // ==================================================================
    // exp bit-exact parity vs the reference build
    // (parity/ref_exp_dump.c → parity/exp_ref.bin) — arch batch 10
    // ==================================================================

    /// One section of parity/exp_ref.bin: kind 0/3 carry n f32 inputs + n f32
    /// outputs; kind 7 carries n u16 in + n u16 out.
    enum ExpSection {
        F32 { kind: u32, input: Vec<f32>, output: Vec<f32> },
        F16 { input: Vec<u16>, output: Vec<u16> },
    }

    fn read_exp_dump() -> Vec<ExpSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/exp_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_exp_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x3150_5845, "section magic 'EXP1'");
            let kind = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            if kind == 7 {
                let (io, rest) = c.split_at(n * 4);
                c = rest;
                let u = |b: &[u8]| -> Vec<u16> {
                    b.chunks_exact(2).map(|w| u16::from_le_bytes(w.try_into().unwrap())).collect()
                };
                out.push(ExpSection::F16 { input: u(&io[..n * 2]), output: u(&io[n * 2..]) });
                continue;
            }
            let (io, rest) = c.split_at(n * 8);
            c = rest;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect()
            };
            out.push(ExpSection::F32 {
                kind,
                input: f(&io[..n * 4]),
                output: f(&io[n * 4..]),
            });
        }
        out
    }

    /// Build ggml_exp(a) through the real graph path (like the C dumper) and
    /// return the output bits.
    fn run_exp_graph(x: &[f32]) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F32, x.len() as i64);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(x)).unwrap();
        let r = ctx.exp(a);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 3);
        ctx.f32s(r).unwrap().to_vec()
    }

    /// exp: the op output must match the reference bit-for-bit, at every
    /// length (incl. non-16-multiples 63/1000/1023/17/1/15/47), on the
    /// lightning-attention ranges (±140 crossing both overflow and underflow)
    /// and on the edge cases (±inf/NaNs/denormals/FLT_MAX). The reference has
    /// no SIMD exp — kind 0 must also equal the raw glibc expf dump (kind 3).
    #[test]
    fn exp_bit_exact_vs_reference() {
        let secs = read_exp_dump();
        let mut n_exp = 0usize;
        let mut n_checked_raw = 0usize;
        let mut pending: Option<Vec<f32>> = None;
        for s in &secs {
            let ExpSection::F32 { kind, input, output } = s else { continue };
            match kind {
                0 => {
                    let mine = run_exp_graph(input);
                    assert_eq!(mine.len(), output.len());
                    for (i, (m, r)) in mine.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "exp mismatch at n={} idx={} x={:e} (rust {:e} vs ref {:e})",
                            input.len(), i, input[i], m, r
                        );
                    }
                    n_exp += input.len();
                    pending = Some(input.clone());
                }
                3 => {
                    // raw glibc expf — must equal the op output we just checked
                    if let Some(input_op) = pending.take() {
                        assert_eq!(input_op.len(), input.len(), "kind 3 input length");
                        for (i, (a, b)) in input_op.iter().zip(input).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(), "kind3 input idx={i}");
                        }
                        n_checked_raw += input.len();
                    }
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        assert_eq!(n_exp, 8154, "full length sweep (7590 LCG + 512 scaled + 52 specials)");
        assert_eq!(n_checked_raw, n_exp, "raw expf cross-check count");
    }

    /// exp on an F16 tensor (apply_unary_op<op_exp, f16, f16>):
    /// fp16(expf(fp16_to_f32(x))). Covers ±0/±inf/NaN payloads, saturation to
    /// +inf past 11.09 and to 0 below -11.5.
    #[test]
    fn exp_f16_bit_exact_vs_reference() {
        let secs = read_exp_dump();
        let mut secs = secs.iter().filter_map(|s| match s {
            ExpSection::F16 { input, output } => Some((input, output)),
            _ => None,
        });
        let (input, output) = secs.next().expect("dump has an F16 exp section");

        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F16, input.len() as i64);
        ctx.arena_resize_tensor(a);
        {
            let bytes = ctx.data_bytes_mut(a).unwrap();
            let dst: &mut [f16] = bytemuck::cast_slice_mut(bytes);
            for (i, h) in input.iter().enumerate() {
                dst[i] = f16::from_bits(*h);
            }
        }
        let r = ctx.exp(a);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 2);
        let got_bytes = ctx.data_bytes(r).unwrap();
        let got: &[f16] = bytemuck::cast_slice(got_bytes);
        assert_eq!(got.len(), output.len());
        for (i, (g, r)) in got.iter().zip(output).enumerate() {
            assert_eq!(
                g.to_bits(),
                *r,
                "f16 exp mismatch at {i}: in {:04x} rust {:04x} ref {:04x}",
                input[i],
                g.to_bits(),
                *r
            );
        }
    }

    // ==================================================================
    // gelu_erf bit-exact parity vs the reference build
    // (parity/ref_gelu_erf_dump.c → parity/gelu_erf_ref.bin)
    // ==================================================================

    /// One section of parity/gelu_erf_ref.bin: kind 0/3 carry n f32 inputs +
    /// n f32 outputs; kind 7 carries n u16 in + n u16 out.
    enum GeluErfSection {
        F32 { kind: u32, input: Vec<f32>, output: Vec<f32> },
        F16 { input: Vec<u16>, output: Vec<u16> },
    }

    fn read_gelu_erf_dump() -> Vec<GeluErfSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/gelu_erf_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_gelu_erf_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x4652_4745, "section magic 'EGRF'");
            let kind = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            let n = u32::from_le_bytes(c[..4].try_into().unwrap()) as usize;
            c = &c[4..];
            if kind == 7 {
                let (io, rest) = c.split_at(n * 4);
                c = rest;
                let u = |b: &[u8]| -> Vec<u16> {
                    b.chunks_exact(2).map(|w| u16::from_le_bytes(w.try_into().unwrap())).collect()
                };
                out.push(GeluErfSection::F16 { input: u(&io[..n * 2]), output: u(&io[n * 2..]) });
                continue;
            }
            let (io, rest) = c.split_at(n * 8);
            c = rest;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect()
            };
            out.push(GeluErfSection::F32 {
                kind,
                input: f(&io[..n * 4]),
                output: f(&io[n * 4..]),
            });
        }
        out
    }

    /// Build ggml_gelu_erf(a) through the real graph path (like the C dumper)
    /// and return the output values.
    fn run_gelu_erf_graph(x: &[f32]) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F32, x.len() as i64);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(x)).unwrap();
        let r = ctx.gelu_erf(a);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 3);
        ctx.f32s(r).unwrap().to_vec()
    }

    /// gelu_erf: the op output must match the reference bit-for-bit, at every
    /// length (incl. non-16-multiples 63/1000/1023/17/1/15/47), on the
    /// whisper-enc activation ranges (±8, ±4, ±0.5, ±27) and on the edge
    /// cases (±inf/NaNs/denormals/FLT_MAX, the erf saturation plateaus). The
    /// reference has no SIMD erf — kind 0 must also pair 1:1 with the raw
    /// formula sections (kind 3).
    #[test]
    fn gelu_erf_bit_exact_vs_reference() {
        let secs = read_gelu_erf_dump();
        let mut n_op = 0usize;
        let mut n_raw = 0usize;
        let mut pending: Option<Vec<f32>> = None;
        for s in &secs {
            let GeluErfSection::F32 { kind, input, output } = s else { continue };
            match kind {
                0 => {
                    let mine = run_gelu_erf_graph(input);
                    assert_eq!(mine.len(), output.len());
                    for (i, (m, r)) in mine.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "gelu_erf mismatch at n={} idx={} x={:e} (rust {:e} vs ref {:e})",
                            input.len(), i, input[i], m, r
                        );
                    }
                    n_op += input.len();
                    pending = Some(input.clone());
                }
                3 => {
                    // the raw 0.5f*x*(1.0f + erff(x*SQRT_2_INV)) loop — its
                    // outputs must equal the op outputs we just checked
                    if let Some(input_op) = pending.take() {
                        assert_eq!(input_op.len(), input.len(), "kind 3 input length");
                        for (i, (a, b)) in input_op.iter().zip(input).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(), "kind3 input idx={i}");
                        }
                        n_raw += input.len();
                    }
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        assert_eq!(n_op, 8152, "full length sweep (7590 LCG + 512 scaled + 50 specials)");
        assert_eq!(n_raw, n_op, "raw formula cross-check count");
    }

    /// gelu_erf on an F16 tensor (ops.cpp:2449 →
    /// ggml_compute_forward_gelu_erf_f16 → vec.h:979):
    /// fp16(0.5*xi*(1+erff(xi*sqrt(1/2)))). Covers ±0/subnormals, ±inf/NaNs
    /// and the erf saturation plateau (±64).
    #[test]
    fn gelu_erf_f16_bit_exact_vs_reference() {
        let secs = read_gelu_erf_dump();
        let mut secs = secs.iter().filter_map(|s| match s {
            GeluErfSection::F16 { input, output } => Some((input, output)),
            _ => None,
        });
        let (input, output) = secs.next().expect("dump has an F16 gelu_erf section");

        let mut ctx = Context::new();
        let a = ctx.new_tensor_1d(GgmlType::F16, input.len() as i64);
        ctx.arena_resize_tensor(a);
        {
            let bytes = ctx.data_bytes_mut(a).unwrap();
            let dst: &mut [f16] = bytemuck::cast_slice_mut(bytes);
            for (i, h) in input.iter().enumerate() {
                dst[i] = f16::from_bits(*h);
            }
        }
        let r = ctx.gelu_erf(a);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 2);
        let got_bytes = ctx.data_bytes(r).unwrap();
        let got: &[f16] = bytemuck::cast_slice(got_bytes);
        assert_eq!(got.len(), output.len());
        for (i, (g, r)) in got.iter().zip(output).enumerate() {
            assert_eq!(
                g.to_bits(),
                *r,
                "f16 gelu_erf mismatch at {i}: in {:04x} rust {:04x} ref {:04x}",
                input[i],
                g.to_bits(),
                *r
            );
        }
    }

    // ==================================================================
    // GGML_OP_POOL_1D bit-exact parity vs the reference build
    // (parity/ref_pool1d_dump.c → parity/pool1d_ref.bin)
    // ==================================================================

    struct Pool1dSection {
        kind: u32,
        rows: usize,
        iw: usize,
        ow: usize,
        op: u32,
        k: i32,
        s: i32,
        p: i32,
        input: Vec<f32>,
        output: Vec<f32>,
    }

    fn read_pool1d_dump() -> Vec<Pool1dSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/pool1d_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_pool1d_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        loop {
            let magic = u32::from_le_bytes(c[..4].try_into().unwrap());
            c = &c[4..];
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x5244_3050, "section magic 'P1DR'");
            let mut hdr = [0u32; 8];
            for h in hdr.iter_mut() {
                *h = u32::from_le_bytes(c[..4].try_into().unwrap());
                c = &c[4..];
            }
            let rows = hdr[1] as usize;
            let iw = hdr[2] as usize;
            let ow = hdr[3] as usize;
            let n_in = rows * iw;
            let n_out = rows * ow;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect()
            };
            let input = f(&c[..n_in * 4]);
            c = &c[n_in * 4..];
            let output = f(&c[..n_out * 4]);
            c = &c[n_out * 4..];
            out.push(Pool1dSection {
                kind: hdr[0],
                rows,
                iw,
                ow,
                op: hdr[4],
                k: hdr[5] as i32,
                s: hdr[6] as i32,
                p: hdr[7] as i32,
                input,
                output,
            });
        }
        out
    }

    /// run pool_1d / pool_2d through the real graph path on one dump section
    fn run_pool_graph(sec: &Pool1dSection, f16: bool) -> Vec<f32> {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(
            if f16 { GgmlType::F16 } else { GgmlType::F32 },
            sec.iw as i64,
            sec.rows as i64,
        );
        ctx.arena_resize_tensor(a);
        if f16 {
            let bytes = ctx.data_bytes_mut(a).unwrap();
            let dst: &mut [f16] = bytemuck::cast_slice_mut(bytes);
            for (i, v) in sec.input.iter().enumerate() {
                dst[i] = f16::from_f32(*v);
            }
        } else {
            ctx.with_f32_mut(a, |p| p.copy_from_slice(&sec.input)).unwrap();
        }
        let r = if sec.kind == 2 {
            // the pre-POOL_1D whisper composition (degenerate 2nd axis)
            ctx.pool_2d(a, Context::GGML_OP_POOL_AVG, 2, 1, 2, 1, 0, 0)
        } else {
            ctx.pool_1d(a, sec.op as i32, sec.k, sec.s, sec.p)
        };
        let mut g = Graph::new(8);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 2);
        assert_eq!(ctx.ne(r)[0] as usize, sec.ow, "output width vs the dump header");
        ctx.f32s(r).unwrap().to_vec()
    }

    /// pool_1d: the op output must match the reference bit-for-bit over the
    /// (k,s,p) x width sweep (full/clipped/empty windows, stride gaps, the
    /// whisper (2,2,0) shape), AVG + MAX, F32 and F16 sources; and the (2,2,0)
    /// AVG case must equal the old pool_2d(k0=2,k1=1,s0=2,s1=1) composition
    /// bit-for-bit — the identity the whisper-enc round relied on.
    #[test]
    fn pool_1d_bit_exact_vs_reference() {
        let secs = read_pool1d_dump();
        assert!(!secs.is_empty(), "empty dump");
        let (mut n_op, mut n_f16, mut n_cmp) = (0usize, 0usize, 0usize);
        let mut cmp_inputs: Vec<Vec<f32>> = Vec::new();
        let mut cmp_outputs: Vec<Vec<f32>> = Vec::new();
        for sec in &secs {
            match sec.kind {
                0 => {
                    let mine = run_pool_graph(sec, false);
                    assert_eq!(mine.len(), sec.output.len());
                    for (i, (m, r)) in mine.iter().zip(&sec.output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "pool_1d mismatch op={} k={} s={} p={} iw={} rows={} idx={} (rust {:e} vs ref {:e})",
                            sec.op, sec.k, sec.s, sec.p, sec.iw, sec.rows, i, m, r
                        );
                    }
                    if sec.k == 2 && sec.s == 2 && sec.p == 0 && sec.op as i32 == Context::GGML_OP_POOL_AVG
                    {
                        cmp_inputs.push(sec.input.clone());
                        cmp_outputs.push(sec.output.clone());
                    }
                    n_op += sec.output.len();
                }
                1 => {
                    let mine = run_pool_graph(sec, true);
                    for (i, (m, r)) in mine.iter().zip(&sec.output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "pool_1d F16 mismatch op={} k={} s={} p={} iw={} idx={}",
                            sec.op, sec.k, sec.s, sec.p, sec.iw, i
                        );
                    }
                    n_f16 += sec.output.len();
                }
                2 => {
                    let mine = run_pool_graph(sec, false);
                    for (i, (m, r)) in mine.iter().zip(&sec.output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            r.to_bits(),
                            "pool_2d composition mismatch iw={} idx={}",
                            sec.iw, i
                        );
                    }
                    n_cmp += sec.output.len();
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        assert!(n_op > 5000, "F32 sweep coverage ({n_op})");
        assert!(n_f16 > 200, "F16 sweep coverage ({n_f16})");
        assert!(n_cmp > 0 && n_cmp == cmp_outputs.iter().map(|v| v.len()).sum::<usize>(),
            "composition sections present ({n_cmp})");

        // the equivalence the whisper-enc round documented: for k=2/s=2/p=0
        // AVG, literal pool_1d == the old pool_2d composition — re-proven on
        // this dump, so switching clip.rs to the literal op moves no bits
        for (inp, want) in cmp_inputs.iter().zip(&cmp_outputs) {
            let sec = Pool1dSection {
                kind: 2,
                rows: 1,
                iw: inp.len(),
                ow: want.len(),
                op: Context::GGML_OP_POOL_AVG as u32,
                k: 2,
                s: 2,
                p: 0,
                input: inp.clone(),
                output: Vec::new(),
            };
            let composition = run_pool_graph(&sec, false);
            assert_eq!(composition.len(), want.len());
            for (i, (c, w)) in composition.iter().zip(want).enumerate() {
                assert_eq!(c.to_bits(), w.to_bits(), "pool_1d(2,2,0) vs pool_2d comp idx={i}");
            }
        }
    }

    // ==================================================================
    // conformer-family ops (roll / conv_2d_dw / conv_2d_direct) bit-exact
    // parity vs the reference build
    // (parity/ref_conformops_dump.c → parity/conformops_ref.bin)
    // ==================================================================

    enum ConformSection {
        Roll { shifts: [i32; 4], ne: [i64; 4], input: Vec<f32>, output: Vec<f32> },
        Dw { params: [i32; 6], src_ne: [i64; 4], knl_ne: [i64; 4], src: Vec<f32>, knl_f16: Option<Vec<u16>>, knl_f32: Option<Vec<f32>>, output: Vec<f32> },
        Direct { params: [i32; 6], src_ne: [i64; 4], knl_ne: [i64; 4], src: Vec<f32>, knl: Vec<f32>, output: Vec<f32>, kind: u32 },
    }

    fn read_conformops_dump() -> Vec<ConformSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/conformops_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_conformops_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        let u32v = |c: &mut &[u8]| -> u32 {
            let v = u32::from_le_bytes(c[..4].try_into().unwrap());
            *c = &c[4..];
            v
        };
        let f32v = |c: &mut &[u8], n: usize| -> Vec<f32> {
            let v = c[..n * 4]
                .chunks_exact(4)
                .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>();
            *c = &c[n * 4..];
            v
        };
        loop {
            let magic = u32v(&mut c);
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x504f_4643, "section magic 'CFOP'");
            let kind = u32v(&mut c);
            match kind {
                0 => {
                    let mut shifts = [0i32; 4];
                    for s in shifts.iter_mut() {
                        *s = u32v(&mut c) as i32;
                    }
                    let ne = [
                        u32v(&mut c) as i64,
                        u32v(&mut c) as i64,
                        u32v(&mut c) as i64,
                        u32v(&mut c) as i64,
                    ];
                    let n = ne.iter().product::<i64>() as usize;
                    let input = f32v(&mut c, n);
                    let output = f32v(&mut c, n);
                    out.push(ConformSection::Roll { shifts, ne, input, output });
                }
                1 | 2 => {
                    let mut params = [0i32; 6];
                    for p in params.iter_mut() {
                        *p = u32v(&mut c) as i32;
                    }
                    let rd4 = |c: &mut &[u8]| -> [i64; 4] {
                        let mut v = [0i64; 4];
                        for x in v.iter_mut() {
                            *x = u32v(c) as i64;
                        }
                        v
                    };
                    let src_ne = rd4(&mut c);
                    let knl_ne = rd4(&mut c);
                    let ns = src_ne.iter().product::<i64>() as usize;
                    let nk = knl_ne.iter().product::<i64>() as usize;
                    let src = f32v(&mut c, ns);
                    let (knl_f16, knl_f32) = if kind == 2 {
                        let (hk, rest) = c.split_at(nk * 2);
                        let v: Vec<u16> = hk
                            .chunks_exact(2)
                            .map(|w| u16::from_le_bytes(w.try_into().unwrap()))
                            .collect();
                        c = rest;
                        (Some(v), None)
                    } else {
                        (None, Some(f32v(&mut c, nk)))
                    };
                    let ow = (src_ne[0] + 2 * params[2] as i64 - params[4] as i64 * (knl_ne[0] - 1) - 1)
                        / params[0] as i64
                        + 1;
                    let oh = (src_ne[1] + 2 * params[3] as i64 - params[5] as i64 * (knl_ne[1] - 1) - 1)
                        / params[1] as i64
                        + 1;
                    let nout = ow * oh * src_ne[2] * src_ne[3];
                    let output = f32v(&mut c, nout as usize);
                    out.push(ConformSection::Dw { params, src_ne, knl_ne, src, knl_f16, knl_f32, output });
                }
                3 | 4 => {
                    let mut params = [0i32; 6];
                    for p in params.iter_mut() {
                        *p = u32v(&mut c) as i32;
                    }
                    let rd4 = |c: &mut &[u8]| -> [i64; 4] {
                        let mut v = [0i64; 4];
                        for x in v.iter_mut() {
                            *x = u32v(c) as i64;
                        }
                        v
                    };
                    let src_ne = rd4(&mut c);
                    let knl_ne = rd4(&mut c);
                    let ns = src_ne.iter().product::<i64>() as usize;
                    let nk = knl_ne.iter().product::<i64>() as usize;
                    let src = f32v(&mut c, ns);
                    let knl = f32v(&mut c, nk);
                    let ow = (src_ne[0] + 2 * params[2] as i64 - params[4] as i64 * (knl_ne[0] - 1) - 1)
                        / params[0] as i64
                        + 1;
                    let oh = (src_ne[1] + 2 * params[3] as i64 - params[5] as i64 * (knl_ne[1] - 1) - 1)
                        / params[1] as i64
                        + 1;
                    let nout = ow * oh * knl_ne[3] * src_ne[3];
                    let output = f32v(&mut c, nout as usize);
                    out.push(ConformSection::Direct { params, src_ne, knl_ne, src, knl, output, kind });
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        out
    }

    /// roll + conv_2d_dw_direct (F32 & F16 kernels) + conv_2d_direct must
    /// match the reference bit-for-bit. The dump also carries the im2col
    /// composite outputs (kind 4) — those document that the C composite is a
    /// *different* numeric path (F16 patches), which is exactly why
    /// conv_2d_direct keeps the patches in F32.
    #[test]
    fn conformer_ops_bit_exact_vs_reference() {
        let secs = read_conformops_dump();
        let (mut n_roll, mut n_dw, mut n_dir) = (0usize, 0usize, 0usize);
        let mut composite_differs = 0usize;
        let mut direct_by_input: Vec<([i64; 4], [i64; 4], [i32; 6], Vec<f32>)> = Vec::new();
        for sec in &secs {
            match sec {
                ConformSection::Roll { shifts, ne, input, output } => {
                    let mut ctx = Context::new();
                    let a = ctx.new_tensor(GgmlType::F32, *ne);
                    ctx.arena_resize_tensor(a);
                    ctx.with_f32_mut(a, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.roll(a, shifts[0], shifts[1], shifts[2], shifts[3]);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "roll {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            shifts, ne, i, m, w
                        );
                    }
                    n_roll += output.len();
                }
                ConformSection::Dw { params, src_ne, knl_ne, src, knl_f16, knl_f32, output } => {
                    let mut ctx = Context::new();
                    let s = ctx.new_tensor(GgmlType::F32, *src_ne);
                    ctx.arena_resize_tensor(s);
                    ctx.with_f32_mut(s, |p| p.copy_from_slice(src)).unwrap();
                    let k = if let Some(h) = knl_f16 {
                        let t = ctx.new_tensor(GgmlType::F16, *knl_ne);
                        ctx.arena_resize_tensor(t);
                        let bytes = ctx.data_bytes_mut(t).unwrap();
                        let dst: &mut [f16] = bytemuck::cast_slice_mut(bytes);
                        for (i, b) in h.iter().enumerate() {
                            dst[i] = f16::from_bits(*b);
                        }
                        t
                    } else {
                        let t = ctx.new_tensor(GgmlType::F32, *knl_ne);
                        ctx.arena_resize_tensor(t);
                        ctx.with_f32_mut(t, |p| p.copy_from_slice(knl_f32.as_ref().unwrap()))
                            .unwrap();
                        t
                    };
                    let r = ctx.conv_2d_dw_direct(k, s, params[0], params[1], params[2], params[3], params[4], params[5]);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len(), "dw output size {:?} {:?}", src_ne, knl_ne);
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "conv_2d_dw {:?} {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            params, src_ne, knl_ne, i, m, w
                        );
                    }
                    n_dw += output.len();
                }
                ConformSection::Direct { params, src_ne, knl_ne, src, knl, output, kind } => {
                    let mut ctx = Context::new();
                    let s = ctx.new_tensor(GgmlType::F32, *src_ne);
                    ctx.arena_resize_tensor(s);
                    ctx.with_f32_mut(s, |p| p.copy_from_slice(src)).unwrap();
                    let k = ctx.new_tensor(GgmlType::F32, *knl_ne);
                    ctx.arena_resize_tensor(k);
                    ctx.with_f32_mut(k, |p| p.copy_from_slice(knl)).unwrap();
                    // kind 3 = the direct kernel; kind 4 = the im2col
                    // composite on the same inputs — each port builder against
                    // its own reference output
                    let r = if *kind == 3 {
                        ctx.conv_2d_direct(k, s, params[0], params[1], params[2], params[3], params[4], params[5])
                    } else {
                        ctx.conv_2d(k, s, params[0], params[1], params[2], params[3], params[4], params[5])
                    };
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len(), "direct output size {:?} {:?}", src_ne, knl_ne);
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "conv_2d kind {kind} {:?} {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            params, src_ne, knl_ne, i, m, w
                        );
                    }
                    if *kind == 3 {
                        n_dir += output.len();
                        direct_by_input.push((*src_ne, *knl_ne, *params, output.clone()));
                    } else {
                        // the composite section paired with the last kind 3
                        let (_, _, _, want) = direct_by_input.last().expect("kind 4 after kind 3");
                        let differs = want.iter().zip(output).any(|(a, b)| a.to_bits() != b.to_bits());
                        if differs {
                            composite_differs += 1;
                        }
                    }
                }
            }
        }
        assert!(n_roll > 1000, "roll coverage ({n_roll})");
        assert!(n_dw > 1000, "dw coverage ({n_dw})");
        assert!(n_dir > 1000, "direct coverage ({n_dir})");
        // the dump's composite sections must NOT be bit-identical to the
        // direct ones — the documented reason conv_2d_direct keeps F32 patches
        assert_eq!(composite_differs, direct_by_input.len(), "C composite == C direct (unexpected)");
    }

    // ==================================================================
    // audio-round-4 ops (sub / sin / cos / sqr / mean / pad_reflect_1d /
    // elu / pad_ext) bit-exact parity vs the reference build
    // (parity/ref_audioops_dump.c → parity/audioops_ref.bin)
    // ==================================================================

    enum AudioOpSection {
        Sub { a_ne: [i64; 4], b_ne: [i64; 4], a: Vec<f32>, b: Vec<f32>, output: Vec<f32> },
        Unary { kind: u32, ne: [i64; 4], input: Vec<f32>, output: Vec<f32> },
        Mean { ne: [i64; 4], input: Vec<f32>, output: Vec<f32> },
        PadReflect { pads: [i32; 2], ne: [i64; 4], input: Vec<f32>, output: Vec<f32> },
        PadExt { pads: [i32; 8], ne: [i64; 4], input: Vec<f32>, output: Vec<f32> },
    }

    fn read_audioops_dump() -> Vec<AudioOpSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/audioops_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_audioops_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        let u32v = |c: &mut &[u8]| -> u32 {
            let v = u32::from_le_bytes(c[..4].try_into().unwrap());
            *c = &c[4..];
            v
        };
        let f32v = |c: &mut &[u8], n: usize| -> Vec<f32> {
            let v = c[..n * 4]
                .chunks_exact(4)
                .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>();
            *c = &c[n * 4..];
            v
        };
        let rd4 = |c: &mut &[u8]| -> [i64; 4] {
            let mut v = [0i64; 4];
            for x in v.iter_mut() {
                *x = u32v(c) as i64;
            }
            v
        };
        loop {
            let magic = u32v(&mut c);
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x504f_5541, "section magic 'AUOP'");
            let kind = u32v(&mut c);
            match kind {
                10 => {
                    let a_ne = rd4(&mut c);
                    let b_ne = rd4(&mut c);
                    let na = a_ne.iter().product::<i64>() as usize;
                    let nb = b_ne.iter().product::<i64>() as usize;
                    let a = f32v(&mut c, na);
                    let b = f32v(&mut c, nb);
                    let nout = a_ne.iter().product::<i64>() as usize;
                    let output = f32v(&mut c, nout);
                    out.push(AudioOpSection::Sub { a_ne, b_ne, a, b, output });
                }
                11 | 12 | 13 | 16 => {
                    let ne = rd4(&mut c);
                    let n = ne.iter().product::<i64>() as usize;
                    let input = f32v(&mut c, n);
                    let output = f32v(&mut c, n);
                    out.push(AudioOpSection::Unary { kind, ne, input, output });
                }
                14 => {
                    let ne = rd4(&mut c);
                    let n = ne.iter().product::<i64>() as usize;
                    let input = f32v(&mut c, n);
                    let nout = ne[1] * ne[2] * ne[3] as i64;
                    let output = f32v(&mut c, nout as usize);
                    out.push(AudioOpSection::Mean { ne, input, output });
                }
                15 => {
                    let mut pads = [0i32; 2];
                    for p in pads.iter_mut() {
                        *p = u32v(&mut c) as i32;
                    }
                    let ne = rd4(&mut c);
                    let n = ne.iter().product::<i64>() as usize;
                    let input = f32v(&mut c, n);
                    let nout = (ne[0] + pads[0] as i64 + pads[1] as i64) * ne[1] * ne[2] * ne[3];
                    let output = f32v(&mut c, nout as usize);
                    out.push(AudioOpSection::PadReflect { pads, ne, input, output });
                }
                17 => {
                    let mut pads = [0i32; 8];
                    for p in pads.iter_mut() {
                        *p = u32v(&mut c) as i32;
                    }
                    let ne = rd4(&mut c);
                    let n = ne.iter().product::<i64>() as usize;
                    let input = f32v(&mut c, n);
                    let nout = (ne[0] + pads[0] as i64 + pads[1] as i64)
                        * (ne[1] + pads[2] as i64 + pads[3] as i64)
                        * (ne[2] + pads[4] as i64 + pads[5] as i64)
                        * (ne[3] + pads[6] as i64 + pads[7] as i64);
                    let output = f32v(&mut c, nout as usize);
                    out.push(AudioOpSection::PadExt { pads, ne, input, output });
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        out
    }

    /// sub / sin / cos / sqr / mean / pad_reflect_1d / elu / pad_ext must
    /// match the reference bit-for-bit on the dump's shapes.
    #[test]
    fn audio_ops_bit_exact_vs_reference() {
        let secs = read_audioops_dump();
        let (mut n_sub, mut n_un, mut n_mean, mut n_pad) = (0usize, 0usize, 0usize, 0usize);
        for sec in &secs {
            match sec {
                AudioOpSection::Sub { a_ne, b_ne, a, b, output } => {
                    let mut ctx = Context::new();
                    let ta = ctx.new_tensor(GgmlType::F32, *a_ne);
                    let tb = ctx.new_tensor(GgmlType::F32, *b_ne);
                    ctx.arena_resize_tensor(ta);
                    ctx.arena_resize_tensor(tb);
                    ctx.with_f32_mut(ta, |p| p.copy_from_slice(a)).unwrap();
                    ctx.with_f32_mut(tb, |p| p.copy_from_slice(b)).unwrap();
                    let r = ctx.sub(ta, tb);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "sub {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            a_ne, b_ne, i, m, w
                        );
                    }
                    n_sub += output.len();
                }
                AudioOpSection::Unary { kind, ne, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor(GgmlType::F32, *ne);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = match *kind {
                        11 => ctx.sin(t),
                        12 => ctx.cos(t),
                        13 => ctx.sqr(t),
                        16 => ctx.elu(t),
                        other => panic!("bad unary kind {other}"),
                    };
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "unary kind {kind} {:?} idx {} (rust {:e} vs ref {:e})",
                            ne, i, m, w
                        );
                    }
                    n_un += output.len();
                }
                AudioOpSection::Mean { ne, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor(GgmlType::F32, *ne);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.mean(t);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "mean {:?} idx {} (rust {:e} vs ref {:e})",
                            ne, i, m, w
                        );
                    }
                    n_mean += output.len();
                }
                AudioOpSection::PadReflect { pads, ne, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor(GgmlType::F32, *ne);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.pad_reflect_1d(t, pads[0], pads[1]);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "pad_reflect {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            pads, ne, i, m, w
                        );
                    }
                    n_pad += output.len();
                }
                AudioOpSection::PadExt { pads, ne, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor(GgmlType::F32, *ne);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.pad_ext(
                        t, pads[0], pads[1], pads[2], pads[3], pads[4], pads[5], pads[6],
                        pads[7],
                    );
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 2);
                    let got = ctx.f32s(r).unwrap();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "pad_ext {:?} {:?} idx {} (rust {:e} vs ref {:e})",
                            pads, ne, i, m, w
                        );
                    }
                    n_pad += output.len();
                }
            }
        }
        assert!(n_sub > 500, "sub coverage ({n_sub})");
        assert!(n_un > 3000, "unary coverage ({n_un})");
        assert!(n_mean > 50, "mean coverage ({n_mean})");
        assert!(n_pad > 500, "pad coverage ({n_pad})");
    }

    // ==================================================================
    // audio round 5 (TTS generators) — the genops dump. Owner: agent GEN5.
    // ==================================================================

    enum GenOpSection {
        Col2im {
            f16: bool,
            s0: i32,
            oc: i32,
            p0: i32,
            k_oc: i64,
            t_in: i64,
            input: Vec<u8>,
            output: Vec<u8>,
        },
        Cumsum {
            rows: i64,
            row_len: i64,
            input: Vec<f32>,
            output: Vec<f32>,
        },
        Tri {
            tri_type: i32,
            n: i64,
            input: Vec<f32>,
            output: Vec<f32>,
        },
        Sum {
            rows: i64,
            row_len: i64,
            input: Vec<f32>,
            output: f32,
        },
        LogStep {
            log: bool,
            input: Vec<f32>,
            output: Vec<f32>,
        },
    }

    fn read_genops_dump() -> Vec<GenOpSection> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/genops_ref.bin");
        let bytes = std::fs::read(path)
            .unwrap_or_else(|_| panic!("missing {path}: build via parity/ref_genops_dump.c"));
        let mut c = &bytes[..];
        let mut out = Vec::new();
        let u32v = |c: &mut &[u8]| -> u32 {
            let v = u32::from_le_bytes(c[..4].try_into().unwrap());
            *c = &c[4..];
            v
        };
        let f32v = |c: &mut &[u8], n: usize| -> Vec<f32> {
            let v = c[..n * 4]
                .chunks_exact(4)
                .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
                .collect::<Vec<_>>();
            *c = &c[n * 4..];
            v
        };
        loop {
            let magic = u32v(&mut c);
            if magic == 0xFFFF_FFFF {
                break;
            }
            assert_eq!(magic, 0x314e_4547, "section magic 'GEN1'");
            let kind = u32v(&mut c);
            let n_in = u32v(&mut c) as usize;
            let _n_out = u32v(&mut c) as usize;
            match kind {
                0 | 6 => {
                    let s0 = u32v(&mut c) as i32;
                    let oc = u32v(&mut c) as i32;
                    let p0 = u32v(&mut c) as i32;
                    let k_oc = u32v(&mut c) as i64;
                    let t_in = u32v(&mut c) as i64;
                    let es = if kind == 6 { 2usize } else { 4usize };
                    let input = c[..n_in * es].to_vec();
                    c = &c[n_in * es..];
                    let n_out = (((t_in - 1) * s0 as i64 + k_oc / oc as i64 - 2 * p0 as i64)
                        * oc as i64) as usize;
                    let output = c[..n_out * es].to_vec();
                    c = &c[n_out * es..];
                    out.push(GenOpSection::Col2im {
                        f16: kind == 6,
                        s0,
                        oc,
                        p0,
                        k_oc,
                        t_in,
                        input,
                        output,
                    });
                }
                1 => {
                    let rows = u32v(&mut c) as i64;
                    let row_len = u32v(&mut c) as i64;
                    let input = f32v(&mut c, n_in);
                    let output = f32v(&mut c, n_in);
                    out.push(GenOpSection::Cumsum { rows, row_len, input, output });
                }
                2 => {
                    let tri_type = u32v(&mut c) as i32;
                    let n = u32v(&mut c) as i64;
                    let input = f32v(&mut c, n_in);
                    let output = f32v(&mut c, n_in);
                    out.push(GenOpSection::Tri { tri_type, n, input, output });
                }
                3 => {
                    let rows = u32v(&mut c) as i64;
                    let row_len = u32v(&mut c) as i64;
                    let input = f32v(&mut c, n_in);
                    let output = f32v(&mut c, 1)[0];
                    out.push(GenOpSection::Sum { rows, row_len, input, output });
                }
                4 | 5 => {
                    let input = f32v(&mut c, n_in);
                    let output = f32v(&mut c, n_in);
                    out.push(GenOpSection::LogStep { log: kind == 4, input, output });
                }
                other => panic!("unexpected kind {other}"),
            }
        }
        out
    }

    /// col2im_1d / cumsum / tri / sum / log / step must match the reference
    /// bit-for-bit on the dump's shapes (4-thread graph computes, so the
    /// parallel splits are covered too).
    #[test]
    fn genops_bit_exact_vs_reference() {
        let secs = read_genops_dump();
        let (mut n_c2i, mut n_rest) = (0usize, 0usize);
        assert!(secs.len() >= 25, "dump sections ({})", secs.len());
        for sec in &secs {
            match sec {
                GenOpSection::Col2im { f16, s0, oc, p0, k_oc, t_in, input, output } => {
                    let mut ctx = Context::new();
                    let ty = if *f16 { GgmlType::F16 } else { GgmlType::F32 };
                    let a = ctx.new_tensor_2d(ty, *k_oc, *t_in);
                    ctx.arena_resize_tensor(a);
                    ctx.data_bytes_mut(a).unwrap().copy_from_slice(input);
                    let r = ctx.col2im_1d(a, *s0, *oc, *p0);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.data_bytes(r).unwrap().to_vec();
                    assert_eq!(got.len(), output.len());
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(m, w, "col2im f16={f16} s0={s0} oc={oc} p0={p0} K*OC={k_oc} T_in={t_in} byte {i}");
                    }
                    n_c2i += output.len();
                }
                GenOpSection::Cumsum { rows, row_len, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor_2d(GgmlType::F32, *row_len, *rows);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.cumsum(t);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap();
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "cumsum {rows}x{row_len} idx {i} (rust {:e} vs ref {:e})",
                            m, w
                        );
                    }
                    n_rest += output.len();
                }
                GenOpSection::Tri { tri_type, n, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor_2d(GgmlType::F32, *n, *n);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.tri(t, *tri_type);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap();
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "tri type {tri_type} n={n} idx {i}"
                        );
                    }
                    n_rest += output.len();
                }
                GenOpSection::Sum { rows, row_len, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor_2d(GgmlType::F32, *row_len, *rows);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = ctx.sum(t);
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap()[0];
                    assert_eq!(
                        got.to_bits(),
                        output.to_bits(),
                        "sum {rows}x{row_len} (rust {:e} vs ref {:e})",
                        got, output
                    );
                    n_rest += input.len();
                }
                GenOpSection::LogStep { log, input, output } => {
                    let mut ctx = Context::new();
                    let t = ctx.new_tensor_1d(GgmlType::F32, input.len() as i64);
                    ctx.arena_resize_tensor(t);
                    ctx.with_f32_mut(t, |p| p.copy_from_slice(input)).unwrap();
                    let r = if *log { ctx.log(t) } else { ctx.step(t) };
                    let mut g = Graph::new(8);
                    g.build_forward(&ctx, r);
                    graph_compute(&mut ctx, &mut g, 4);
                    let got = ctx.f32s(r).unwrap();
                    for (i, (m, w)) in got.iter().zip(output).enumerate() {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "{} idx {i} (rust {:e} vs ref {:e})",
                            if *log { "log" } else { "step" },
                            m, w
                        );
                    }
                    n_rest += output.len();
                }
            }
        }
        assert!(n_c2i > 3000, "col2im coverage ({n_c2i})");
        assert!(n_rest > 1000, "rest coverage ({n_rest})");
    }

    /// The lazily-built gelu f16 table (vec.cpp:6 / ggml-cpu.c:3891) must equal
    /// the reference's `ggml_table_gelu_f16` for all 65536 entries.
    #[test]
    fn gelu_f16_table_matches_reference() {
        let secs = read_tanh_dump();
        let tab = secs
            .iter()
            .find_map(|s| match s {
                TanhSection::GeluTable(t) => Some(t),
                _ => None,
            })
            .expect("dump is missing the gelu table section");
        let mine = ggml_table_gelu_f16();
        assert_eq!(mine.len(), tab.len());
        let mut diffs = Vec::new();
        for (i, (m, r)) in mine.iter().zip(tab).enumerate() {
            if m != r {
                diffs.push((i, *m, *r));
            }
        }
        if !diffs.is_empty() {
            panic!(
                "gelu table differs at {} of {} entries, first: {:?}",
                diffs.len(), tab.len(),
                diffs.iter().take(8)
                    .map(|(i, m, r)| (format!("f16 {i:#06x}"),
                                      format!("mine {m:#06x}"), format!("ref {r:#06x}")))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn get_rows_argmax_argsort_sum_rows() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 6, 4);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| {
            p.copy_from_slice(&[1., 9., 3., 7., 5., 2., 8., 1., 4., 6., 0., 3., 2., 2., 2., 9., 1., 0., 5., 4., 7., 8., 3., 6.]);
        })
        .unwrap();

        let ids = ctx.new_tensor_1d(GgmlType::I32, 3);
        ctx.arena_resize_tensor(ids);
        ctx.with_i32_mut(ids, |p| p.copy_from_slice(&[1, 3, 1])).unwrap();

        let gr = ctx.get_rows(a, ids);
        let am = ctx.argmax(gr);
        let asr = ctx.argsort(gr, GGML_SORT_ORDER_ASC);
        let sr = ctx.sum_rows(gr);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, gr);
        graph_compute(&mut ctx, &mut g, 1);
        let mut g2 = Graph::new(8);
        g2.build_forward(&ctx, am);
        graph_compute(&mut ctx, &mut g2, 1);
        let mut g3 = Graph::new(8);
        g3.build_forward(&ctx, asr);
        graph_compute(&mut ctx, &mut g3, 1);
        let mut g4 = Graph::new(8);
        g4.build_forward(&ctx, sr);
        graph_compute(&mut ctx, &mut g4, 1);

        // rows picked: row1=[8,1,4,6,0,3], row3=[5,4,7,8,3,6], row1 again
        let got = ctx.f32s(gr).unwrap();
        assert_eq!(&got[..6], &[8., 1., 4., 6., 0., 3.]);
        assert_eq!(&got[6..12], &[5., 4., 7., 8., 3., 6.]);
        // argmax: row0 max 8@0, row1 max 8@3, row2 max 8@0
        assert_eq!(ctx.i32s(am).unwrap(), &[0, 3, 0]);
        // argsort asc of row0 [8,1,4,6,0,3] -> indices [4,1,5,2,3,0]
        assert_eq!(&ctx.i32s(asr).unwrap()[..6], &[4, 1, 5, 2, 3, 0]);
        // sum rows
        let sums = ctx.f32s(sr).unwrap();
        assert_eq!(sums[0], 8. + 1. + 4. + 6. + 0. + 3.);
        assert_eq!(sums[1], 5. + 4. + 7. + 8. + 3. + 6.);
    }

    #[test]
    fn diag_mask_inf_values() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(GgmlType::F32, 4, 4, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.fill(1.0)).unwrap();
        let y = ctx.diag_mask_inf(a, 2);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 2);
        let out = ctx.f32s(y).unwrap();
        for k in 0..2 {
            for j in 0..4 {
                for i in 0..4 {
                    let v = out[(k * 4 + j) * 4 + i];
                    if i > 2 + j {
                        assert!(v.is_infinite() && v < 0.0, "k={k} j={j} i={i}");
                    } else {
                        assert_eq!(v, 1.0);
                    }
                }
            }
        }
    }

    #[test]
    fn concat_and_cpy_views() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&lcg(8, 31))).unwrap();
        let b = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.arena_resize_tensor(b);
        ctx.with_f32_mut(b, |p| p.copy_from_slice(&lcg(8, 32))).unwrap();

        let c = ctx.concat(a, b, 1);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, c);
        graph_compute(&mut ctx, &mut g, 2);
        let out = ctx.f32s(c).unwrap();
        assert_eq!(ctx.ne(c), &[4, 4, 1, 1]);
        assert_eq!(&out[..8], lcg(8, 31).as_slice());
        assert_eq!(&out[8..], lcg(8, 32).as_slice());

        // cpy through a permuted view into a contiguous dst
        let t = ctx.transpose(a); // [2,4]
        let cont = ctx.cont(t);
        let mut g2 = Graph::new(4);
        g2.build_forward(&ctx, cont);
        graph_compute(&mut ctx, &mut g2, 2);
        let src = ctx.f32s(a).unwrap();
        let got = ctx.f32s(cont).unwrap();
        // cont is [2,4] row-major: cont(v*2 + u) = t(u, v) = a(v, u) = src[u*4 + v]
        for u in 0..2 {
            for v in 0..4 {
                assert_eq!(got[v * 2 + u], src[u * 4 + v]);
            }
        }
    }

    #[test]
    fn repeat_and_get_rows_back() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 1);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&[1., 2., 3., 4.])).unwrap();
        let target = ctx.new_tensor_2d(GgmlType::F32, 4, 3);
        let r = ctx.repeat(a, target);
        let mut g = Graph::new(4);
        g.build_forward(&ctx, r);
        graph_compute(&mut ctx, &mut g, 1);
        let out = ctx.f32s(r).unwrap();
        for rep in 0..3 {
            assert_eq!(&out[rep * 4..rep * 4 + 4], &[1., 2., 3., 4.]);
        }

        // get_rows_back: scatter-add rows
        let gb_src = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.arena_resize_tensor(gb_src);
        ctx.with_f32_mut(gb_src, |p| p.copy_from_slice(&[1., 1., 1., 1., 10., 10., 10., 10.])).unwrap();
        let ids = ctx.new_tensor_1d(GgmlType::I32, 2);
        ctx.arena_resize_tensor(ids);
        ctx.with_i32_mut(ids, |p| p.copy_from_slice(&[2, 2])).unwrap();
        let shape = ctx.new_tensor_2d(GgmlType::F32, 4, 3);
        let gb = ctx.get_rows_back(gb_src, ids, shape);
        let mut g2 = Graph::new(4);
        g2.build_forward(&ctx, gb);
        graph_compute(&mut ctx, &mut g2, 1);
        let got = ctx.f32s(gb).unwrap();
        assert_eq!(&got[..4], &[0., 0., 0., 0.]);
        assert_eq!(&got[4..8], &[0., 0., 0., 0.]);
        assert_eq!(&got[8..], &[11., 11., 11., 11.]);
    }

    // ===================== end-to-end mini graph =====================

    /// 随机 896×896 F32 权重 × 3 token 激活 → rms_norm → mul_mat → softmax,
    /// 与朴素 Rust 实现逐步对照。
    #[test]
    fn end_to_end_rms_mulmat_softmax() {
        let n = 896usize;
        let rows = 896usize;
        let tokens = 3usize;

        let mut ctx = Context::new();
        let w = ctx.new_tensor_2d(GgmlType::F32, n as i64, rows as i64);
        ctx.arena_resize_tensor(w);
        let wrows: Vec<Vec<f32>> = (0..rows).map(|r| lcg(n, 9000 + r as u32)).collect();
        ctx.with_f32_mut(w, |p| {
            for r in 0..rows {
                p[r * n..(r + 1) * n].copy_from_slice(&wrows[r]);
            }
        })
        .unwrap();

        let x = ctx.new_tensor_2d(GgmlType::F32, n as i64, tokens as i64);
        ctx.arena_resize_tensor(x);
        let xrows: Vec<Vec<f32>> = (0..tokens).map(|t| lcg(n, 9500 + t as u32)).collect();
        ctx.with_f32_mut(x, |p| {
            for t in 0..tokens {
                p[t * n..(t + 1) * n].copy_from_slice(&xrows[t]);
            }
        })
        .unwrap();

        let normed = ctx.rms_norm(x, 1e-6);
        let logits = ctx.mul_mat(w, normed);
        let probs = ctx.soft_max(logits);
        let mut g = Graph::new(16);
        g.build_forward(&ctx, probs);
        graph_compute(&mut ctx, &mut g, 4);

        // step-by-step naive reference
        let mut ref_norm = vec![vec![0f32; n]; tokens];
        for t in 0..tokens {
            let mut sum = 0f64;
            for &v in &xrows[t] {
                sum += (v * v) as f64;
            }
            let scale = 1.0f32 / ((sum / n as f64) as f32 + 1e-6).sqrt();
            for j in 0..n {
                ref_norm[t][j] = xrows[t][j] * scale;
            }
        }
        let mut ref_logits = vec![vec![f64::MIN; rows]; tokens]; // placeholder
        for t in 0..tokens {
            let mut mx = f64::NEG_INFINITY;
            for r in 0..rows {
                let mut acc = 0f64;
                for j in 0..n {
                    acc += wrows[r][j] as f64 * ref_norm[t][j] as f64;
                }
                ref_logits[t][r] = acc;
                mx = mx.max(acc);
            }
            // softmax with ggml_expf numerics
            let got_row = &ctx.f32s(probs).unwrap()[t * rows..(t + 1) * rows];
            let mut sum = 0f64;
            let mut exps = vec![0f32; rows];
            for r in 0..rows {
                exps[r] = ggml_expf(((ref_logits[t][r] - mx) as f32));
                sum += exps[r] as f64;
            }
            let inv = (1.0 / sum) as f32;
            let mut psum = 0f64;
            for r in 0..rows {
                let want = exps[r] * inv;
                let rel = ((got_row[r] - want) / want.max(1e-30)).abs();
                assert!(rel < 1e-3, "t={t} r={r} {} vs {want}", got_row[r]);
                psum += got_row[r] as f64;
            }
            assert!((psum - 1.0).abs() < 1e-4, "t={t} sum={psum}");
        }
    }

    // ==================================================================
    // mul_mat_id / add_id / glu / soft_max sinks — gpt-oss operator unit tests
    // ==================================================================

    /// mul_mat_id vs the naive per-(slot, token) expert dot, for a quantized
    /// `as` (MXFP4 + Q4_0, the two shapes gpt-oss ships), a 3D activations
    /// tensor and a **strided** ids view taken from `argsort_top_k` (pins the
    /// nb[1]=n_expert*4 view handling of the C kernels).
    #[test]
    fn mul_mat_id_matches_naive() {
        // b_ne1 covers both shapes the gpt-oss MoE uses: 1 for the gate/up
        // projections (activations reshaped to [n_embd, 1, T]) and
        // n_expert_used for the down projection, where each expert slot reads
        // its *own* activation row (`i11 = slot % b->ne[1]`,
        // ggml-cpu.c:1512-1524 — the slot-indexed row mapping).
        for (ty, b_ne1) in [
            (GgmlType::Mxfp4, 1usize),
            (GgmlType::Q4_0, 1usize),
            (GgmlType::F32, 3usize),
        ] {
            let (n, n_ff, n_expert, n_ids, n_tok) = (64usize, 6usize, 5usize, 3usize, 4usize);
            let mut ctx = Context::new();

            // as: [n, n_ff, n_expert] random-quantized
            let a = ctx.new_tensor_3d(ty, n as i64, n_ff as i64, n_expert as i64);
            ctx.arena_resize_tensor(a);
            let rs_a = ty_row_bytes(ty, n);
            let wrows: Vec<Vec<f32>> = (0..n_ff * n_expert).map(|r| lcg(n, 700 + r as u32)).collect();
            {
                let bytes = ctx.data_bytes_mut(a).unwrap();
                for (r, row) in wrows.iter().enumerate() {
                    let blk = &mut bytes[r * rs_a..(r + 1) * rs_a];
                    match ty {
                        // no MXFP4 quantizer in quants.rs (only dequantize — the
                        // GGUF ships pre-quantized weights): deterministic nibbles
                        // with a sane E8M0 exponent form a valid MXFP4 block stream
                        GgmlType::Mxfp4 => {
                            let _ = row;
                            for b in 0..n / 32 {
                                blk[b * 17] = 120 + (b % 8) as u8; // e: 2^-8 .. 2^-1
                                for j in 0..16 {
                                    blk[b * 17 + 1 + j] = ((b * 7 + j * 13) % 256) as u8;
                                }
                            }
                        }
                        GgmlType::F32 => blk.copy_from_slice(bytemuck::cast_slice(&row[..n])),
                        _ => crate::quants::quantize_row_q4_0_ref(
                            row, bytemuck::cast_slice_mut(blk)),
                    }
                }
            }

            // b: [n, b_ne1, n_tok]
            let b = ctx.new_tensor_3d(GgmlType::F32, n as i64, b_ne1 as i64, n_tok as i64);
            ctx.arena_resize_tensor(b);
            let xrows: Vec<Vec<f32>> = (0..b_ne1 * n_tok).map(|t| lcg(n, 900 + t as u32)).collect();
            ctx.with_f32_mut(b, |p| {
                for t in 0..b_ne1 * n_tok {
                    p[t * n..(t + 1) * n].copy_from_slice(&xrows[t]);
                }
            })
            .unwrap();

            // ids via argsort_top_k of a [n_expert, n_tok] score matrix — the
            // strided view the real graph feeds to mul_mat_id (nb[1] = n_expert*4)
            let score = ctx.new_tensor_2d(GgmlType::F32, n_expert as i64, n_tok as i64);
            ctx.arena_resize_tensor(score);
            ctx.with_f32_mut(score, |p| p.copy_from_slice(&lcg(n_expert * n_tok, 1234))).unwrap();
            let ids = ctx.argsort_top_k(score, n_ids as i32);

            let out = ctx.mul_mat_id(a, b, ids);
            let mut g = Graph::new(16);
            g.build_forward(&ctx, out);
            graph_compute(&mut ctx, &mut g, 3);

            let got = ctx.f32s(out).unwrap().to_vec();
            assert_eq!(ctx.ne(out), &[n_ff as i64, n_ids as i64, n_tok as i64, 1]);

            // naive: dequantize `as` rows, quantize activations like the kernel
            let vdt = vec_dot::vec_dot_type(ty).or_else(|| mxfp4_vec_dot_type(ty)).unwrap();
            assert!(vdt == GgmlType::F32 || vdt == GgmlType::Q8_0);
            let abytes = ctx.data_bytes(a).unwrap().to_vec();
            let mut adeq = vec![0f32; n_ff * n_expert * n];
            for r in 0..n_ff * n_expert {
                match ty {
                    GgmlType::F32 => adeq[r * n..(r + 1) * n]
                        .copy_from_slice(bytemuck::cast_slice(&abytes[r * rs_a..(r + 1) * rs_a])),
                    _ => quants::dequantize_row(ty, &abytes[r * rs_a..(r + 1) * rs_a], &mut adeq[r * n..(r + 1) * n]),
                }
            }
            let idv = ctx.i32s(ids).unwrap().to_vec();
            let idnb1 = ctx.nb(ids)[1] as usize / 4;
            for t in 0..n_tok {
                for s in 0..n_ids {
                    let i11 = s % b_ne1;
                    let mut yq = vec![0u8; ty_row_bytes(vdt, n)];
                    let mut yf = vec![0f32; n];
                    match vdt {
                        GgmlType::Q8_0 => {
                            crate::quants::quantize_row_q8_0(
                                &xrows[i11 + t * b_ne1], bytemuck::cast_slice_mut(&mut yq));
                            quants::dequantize_row(vdt, &yq, &mut yf);
                        }
                        GgmlType::F32 => yf.copy_from_slice(&xrows[i11 + t * b_ne1]),
                        _ => unreachable!(),
                    }
                    let e = idv[s + t * idnb1] as usize;
                    for r in 0..n_ff {
                        let arow = &adeq[(e * n_ff + r) * n..(e * n_ff + r + 1) * n];
                        let want: f64 = arow.iter().zip(&yf).map(|(x, y)| *x as f64 * *y as f64).sum();
                        // dst [n_ff, n_ids, n_tok]: (r, s, t) at r + s*n_ff + t*n_ff*n_ids
                        let got_v = got[r + s * n_ff + t * n_ff * n_ids];
                        assert!(
                            (got_v as f64 - want).abs() <= 1e-4 * want.abs().max(1.0),
                            "{ty:?} s={s} t={t} r={r}: {got_v} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// add_id: dst(i0,i1,i2) = a + b(i0, ids(i1,i2)) with the strided ids view.
    #[test]
    fn add_id_matches_naive() {
        let (n0, n_ids, n_tok, n_expert) = (5usize, 3usize, 4usize, 6usize);
        let mut ctx = Context::new();
        let a = ctx.new_tensor_3d(GgmlType::F32, n0 as i64, n_ids as i64, n_tok as i64);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&lcg(n0 * n_ids * n_tok, 11))).unwrap();
        let b = ctx.new_tensor_2d(GgmlType::F32, n0 as i64, n_expert as i64);
        ctx.arena_resize_tensor(b);
        ctx.with_f32_mut(b, |p| p.copy_from_slice(&lcg(n0 * n_expert, 22))).unwrap();
        let score = ctx.new_tensor_2d(GgmlType::F32, n_expert as i64, n_tok as i64);
        ctx.arena_resize_tensor(score);
        ctx.with_f32_mut(score, |p| p.copy_from_slice(&lcg(n_expert * n_tok, 33))).unwrap();
        let ids = ctx.argsort_top_k(score, n_ids as i32);

        let out = ctx.add_id(a, b, ids);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, 2);

        let av = ctx.f32s(a).unwrap().to_vec();
        let bv = ctx.f32s(b).unwrap().to_vec();
        let idv = ctx.i32s(ids).unwrap().to_vec();
        let idnb1 = ctx.nb(ids)[1] as usize / 4;
        let got = ctx.f32s(out).unwrap();
        for t in 0..n_tok {
            for s in 0..n_ids {
                let e = idv[s + t * idnb1] as usize;
                for i in 0..n0 {
                    assert_eq!(got[i + s * n0 + t * n0 * n_ids], av[i + s * n0 + t * n0 * n_ids] + bv[i + e * n0]);
                }
            }
        }
    }

    /// swiglu_oai: dst = min(gate, limit) / (1 + exp(alpha*-min(gate,limit)))
    /// * (clamp(up) + 1) — ops.cpp:3321-3385 scalar semantics, bit-exact against
    /// the same expression evaluated per element.
    #[test]
    fn swiglu_oai_matches_reference_expression() {
        let (n_ff, n_ids, n_tok) = (7usize, 2usize, 3usize);
        let mut ctx = Context::new();
        let gate = ctx.new_tensor_3d(GgmlType::F32, n_ff as i64, n_ids as i64, n_tok as i64);
        let up = ctx.new_tensor_3d(GgmlType::F32, n_ff as i64, n_ids as i64, n_tok as i64);
        ctx.arena_resize_tensor(gate);
        ctx.arena_resize_tensor(up);
        let gv: Vec<f32> = lcg(n_ff * n_ids * n_tok, 5).iter().map(|v| v * 12.0).collect();
        let uv: Vec<f32> = lcg(n_ff * n_ids * n_tok, 6).iter().map(|v| v * 12.0).collect();
        ctx.with_f32_mut(gate, |p| p.copy_from_slice(&gv)).unwrap();
        ctx.with_f32_mut(up, |p| p.copy_from_slice(&uv)).unwrap();

        let out = ctx.swiglu_oai(gate, up, 1.702, 7.0);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, 2);

        let got = ctx.f32s(out).unwrap();
        for i in 0..gv.len() {
            let x = gv[i].min(7.0f32);
            let yv = uv[i].clamp(-7.0f32, 7.0f32);
            let want = x / (1.0 + (1.702f32 * (-x)).exp()) * (yv + 1.0);
            assert_eq!(got[i].to_bits(), want.to_bits(), "swiglu_oai idx {i}");
        }
        // clamps actually engage
        assert!(gv.iter().any(|&v| v > 7.0) && uv.iter().any(|&v| v > 7.0));
    }

    /// soft_max sinks: `expf(sk - max)` in the denominator + sk in the row max
    /// (ops.cpp:5625-5680) — checked against a direct evaluation.
    #[test]
    fn soft_max_sinks_matches_reference_expression() {
        let (ne00, n_tok, n_head) = (5usize, 2usize, 3usize);
        let mut ctx = Context::new();
        let x = ctx.new_tensor_3d(GgmlType::F32, ne00 as i64, n_tok as i64, n_head as i64);
        ctx.arena_resize_tensor(x);
        let xv = lcg(ne00 * n_tok * n_head, 77);
        ctx.with_f32_mut(x, |p| p.copy_from_slice(&xv)).unwrap();
        let sk = ctx.new_tensor_1d(GgmlType::F32, n_head as i64);
        ctx.arena_resize_tensor(sk);
        let skv = [2.0f32, -3.0, 0.5];
        ctx.with_f32_mut(sk, |p| p.copy_from_slice(&skv)).unwrap();

        let y = ctx.soft_max_ext(x, None, 1.0, 0.0);
        ctx.soft_max_add_sinks(y, Some(sk));
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 2);

        let got = ctx.f32s(y).unwrap().to_vec();
        for h in 0..n_head {
            for t in 0..n_tok {
                // rows are (i01, i02) = (token, head); layout x[i00 + i01*ne00 + i02*ne00*n_tok]
                let mut row = vec![0f32; ne00];
                for i in 0..ne00 {
                    row[i] = xv[i + t * ne00 + h * ne00 * n_tok];
                }
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(skv[h]);
                let mut sum = 0f64;
                for v in &row {
                    sum += (v - max).exp() as f64;
                }
                sum += (skv[h] - max).exp() as f64;
                for i in 0..ne00 {
                    let want = ((row[i] - max).exp() as f64 / sum) as f32;
                    let got_v = got[i + t * ne00 + h * ne00 * n_tok];
                    assert!((got_v - want).abs() < 1e-6, "sinks h={h} t={t} i={i}: {got_v} vs {want}");
                }
            }
        }

        // no-sinks softmax differs (the correction is real)
        let y0 = ctx.soft_max_ext(x, None, 1.0, 0.0);
        let mut g0 = Graph::new(8);
        g0.build_forward(&ctx, y0);
        graph_compute(&mut ctx, &mut g0, 2);
        let plain = ctx.f32s(y0).unwrap().to_vec();
        assert!(plain.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-4));
    }


    /// argsort_top_k keeps the C view stride: T-1 descending ids per column, and
    /// nb[1] == ne0*4 (the full sort row), so downstream nb reads stay correct.
    #[test]
    fn argsort_top_k_view_strides() {
        let mut ctx = Context::new();
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| p.copy_from_slice(&[0.1, 0.9, 0.5, -0.2, 2.0, -1.0, 0.0, 3.0])).unwrap();
        let t = ctx.argsort_top_k(a, 2);
        assert_eq!(ctx.ne(t), &[2, 2, 1, 1]);
        assert_eq!(ctx.nb(t)[1], 4 * 4, "view row stride must stay ne0*4 (ggml_view_4d)");
        let mut g = Graph::new(8);
        g.build_forward(&ctx, t);
        graph_compute(&mut ctx, &mut g, 1);
        // column 0: values [0.1, 0.9, 0.5, -0.2] → desc ids 1, 2
        // column 1: values [2.0, -1.0, 0.0, 3.0]   → desc ids 3, 0
        // element (i, col) at i*4 + col*nb[1] (nb[1] = 4*ne0, the full sort row)
        let b: &[i32] = bytemuck::cast_slice(ctx.data_bytes(t).unwrap());
        let nb1 = ctx.nb(t)[1] as usize / 4;
        assert_eq!(b[0], 1);
        assert_eq!(b[1], 2);
        assert_eq!(b[nb1], 3);
        assert_eq!(b[nb1 + 1], 0);
    }
    // ==================================================================
    // interpolate (GGML_OP_UPSCALE) — the vision position-embedding resize
    // (clip.cpp:312 `resize_position_embeddings`). Ground truth:
    // parity/interp_ref.bin, produced by parity/ref_interp_dump.c against the
    // reference build through the *public* ggml API (same node chain as
    // clip.cpp:325-329).
    // ==================================================================

    #[test]
    fn interpolate_align_corners_vs_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/interp_ref.bin");
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping: {path} not present (build parity/ref_interp_dump.c)");
            return;
        };
        let dim = |i: usize| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let expect: Vec<f32> = bytes[32..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        const N_EMBD: i64 = 1152;
        const N_SIDE: i64 = 48;
        const OUT_SIDE: i64 = 28;

        let mut ctx = Context::new();
        let pos = ctx.new_tensor_2d(GgmlType::F32, N_EMBD, N_SIDE * N_SIDE);
        ctx.arena_resize_tensor(pos);
        ctx.with_f32_mut(pos, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = (((i as u64 * 37) % 2048) as f32 - 1024.0) / 512.0;
            }
        })
        .unwrap();

        let a = ctx.reshape_3d(pos, N_EMBD, N_SIDE, N_SIDE);
        let b = ctx.permute(a, 2, 0, 1, 3);
        let c = ctx.interpolate(
            b,
            OUT_SIDE,
            OUT_SIDE,
            N_EMBD,
            1,
            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS,
        );
        let d = ctx.permute(c, 1, 2, 0, 3);
        let e = ctx.cont_2d(d, N_EMBD, OUT_SIDE * OUT_SIDE);

        let mut g = Graph::new(16);
        g.build_forward(&ctx, e);
        graph_compute(&mut ctx, &mut g, 8);

        assert_eq!(ctx.ne(e), &[dim(0), dim(1), dim(2), dim(3)]);
        let got = ctx.f32s(e).expect("arena tensor");
        assert_eq!(got.len(), expect.len());
        let nbit = got
            .iter()
            .zip(&expect)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        let first_bad = got
            .iter()
            .zip(&expect)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        let worst = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-30))
            .fold(0.0f32, f32::max);
        let worst_abs = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let worst_rel_big = got
            .iter()
            .zip(&expect)
            .filter(|(_, b)| b.abs() > 0.1)
            .map(|(a, b)| (a - b).abs() / b.abs())
            .fold(0.0f32, f32::max);
        eprintln!("interpolate: worst abs {worst_abs:.3e}, worst rel(|v|>0.1) {worst_rel_big:.3e}");
        eprintln!(
            "interpolate: {}/{} bitwise, worst abs {worst_abs:.3e} (worst rel {worst:.3e}), \
             worst rel(|v|>0.1) {worst_rel_big:.3e}, first bad at {first_bad:?}",
            nbit,
            got.len()
        );
        // The reference build compiles this kernel with -O3 -march=native, i.e.
        // GCC's default FP contraction: the four products of the bilinear sum
        // become FMA chains whose intermediate roundings the port's scalar
        // evaluation does not reproduce. The difference is a 1-2 ulp band
        // (absolute, and relative for all but near-cancelling outputs) — see
        // PARITY.md's mtmd section, which measures the effect on the final
        // embeddings.
        assert!(
            worst_abs < 1e-5,
            "interpolate absolute error {worst_abs:.3e} exceeds 1e-5 (first bad {first_bad:?})"
        );
        assert!(
            worst_rel_big < 2e-5,
            "interpolate relative error {worst_rel_big:.3e} on |v|>0.1 exceeds 2e-5"
        );
    }
    /// perf9: the arena reuse planner (`graph_compute_reusing`) must be
    /// bit-identical to the bump path on the protected outputs and must
    /// actually reuse — the arena peak stays strictly below the bump's for a
    /// graph whose intermediates have non-overlapping lifetimes. This pins
    /// the aliasing safety of the gallocr discipline (ggml-alloc.c:623-823):
    /// only tensors whose last consumer already ran get their blocks handed
    /// to later nodes, and GGML_TENSOR_FLAG_OUTPUT tensors are never freed
    /// nor reused.
    #[test]
    fn arena_reuse_bit_identity() {
        let build = |ctx: &mut Context| -> (TensorId, TensorId) {
            let w = ctx.new_tensor_2d(GgmlType::F32, 64, 32);
            ctx.arena_resize_tensor(w);
            let w2 = ctx.new_tensor_2d(GgmlType::F32, 32, 16); // k = n0's rows
            ctx.arena_resize_tensor(w2);
            let x = ctx.new_tensor_2d(GgmlType::F32, 64, 8);
            ctx.arena_resize_tensor(x);
            for t in [w, w2, x] {
                let n = ctx.nbytes(t) / 4;
                let vals: Vec<f32> = (0..n)
                    .map(|i| (((i as i32) % 13) as f32) * 0.25 - 1.0)
                    .collect();
                let bytes: Vec<u8> = bytemuck::cast_slice(&vals).to_vec();
                let span = ctx.data_bytes_mut(t).unwrap();
                span.copy_from_slice(&bytes);
            }
            let n0 = ctx.mul_mat(w, x); // [32, 8]
            ctx.set_name(n0, "n0");
            let n1 = ctx.silu(n0);
            ctx.set_name(n1, "n1");
            let n2 = ctx.mul_mat(w2, n1); // [16, 8]
            ctx.set_name(n2, "n2");
            let n3 = ctx.add(n0, n2); // n0 has two consumers -> no in-place
            ctx.set_name(n3, "n3");
            let n4 = ctx.mul(n3, n3);
            ctx.set_name(n4, "n4");
            (n2, n4)
        };

        // bump
        let mut ctx_a = Context::new();
        let (n2_a, n4_a) = build(&mut ctx_a);
        let mut ga = Graph::new(16);
        ga.build_forward(&ctx_a, n4_a);
        graph_compute(&mut ctx_a, &mut ga, 2);

        // reuse, protecting both tensors the "caller" reads afterwards
        let mut ctx_b = Context::new();
        let (n2_b, n4_b) = build(&mut ctx_b);
        let mut gb = Graph::new(16);
        gb.build_forward(&ctx_b, n4_b);
        graph_compute_reusing(&mut ctx_b, &mut gb, 2, &[n2_b, n4_b]);

        assert_eq!(ctx_a.data_bytes(n4_a).unwrap(), ctx_b.data_bytes(n4_b).unwrap(),
            "reuse path output must be bit-identical to the bump path");
        assert_eq!(ctx_a.data_bytes(n2_a).unwrap(), ctx_b.data_bytes(n2_b).unwrap(),
            "second protected output must be bit-identical too");
        assert!(ctx_b.arena.len() < ctx_a.arena.len(),
            "the planner must reuse freed blocks: reuse arena {} B vs bump {} B",
            ctx_b.arena.len(), ctx_a.arena.len());

        // a second forward on the same context (the steady-state shape): the
        // watermark reset must leave the same plan and identical outputs
        let mark = ctx_b.mark();
        let (n2_c, n4_c) = build(&mut ctx_b);
        let _ = mark;
        let mut gc = Graph::new(16);
        gc.build_forward(&ctx_b, n4_c);
        graph_compute_reusing(&mut ctx_b, &mut gc, 2, &[n2_c, n4_c]);
        assert_eq!(ctx_a.data_bytes(n4_a).unwrap(), ctx_b.data_bytes(n4_c).unwrap(),
            "steady-state (second) reuse forward must stay bit-identical");
    }

}

    #[test]
    fn set_rows_scatters_and_converts() {
        use crate::ops;
        let mut ctx = Context::new();
        // dst: f16 [4 rows x 3]
        let dst = ctx.new_tensor_2d(GgmlType::F16, 3, 4);
        ctx.arena_resize_tensor(dst);
        // src rows: f32 [2 rows x 3]
        let src = ctx.new_tensor_2d(GgmlType::F32, 3, 2);
        ctx.arena_resize_tensor(src);
        ctx.with_f32_mut(src, |p| p.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        // idx: i64 [2]
        let idx = ctx.new_tensor_2d(GgmlType::I64, 2, 1);
        ctx.arena_resize_tensor(idx);
        ctx.data_bytes_mut(idx).unwrap().copy_from_slice(bytemuck::cast_slice(&[3i64, 0i64]));
        let node = ctx.set_rows(dst, src, idx);
        let mut g = crate::graph::Graph::new(8);
        g.build_forward(&ctx, node);
        crate::compute::graph_compute(&mut ctx, &mut g, 2);
        let hs: &[half::f16] = bytemuck::cast_slice(ctx.data_bytes(dst).unwrap());
        assert_eq!(hs[0].to_f32(), 4.0);
        assert_eq!(hs[2].to_f32(), 6.0);
        assert_eq!(hs[9].to_f32(), 1.0);
        assert_eq!(hs[11].to_f32(), 3.0);
        assert_eq!(hs[3].to_f32(), 0.0); // untouched row 1
    }

    /// src1 broadcast along dim0 (`ne10 == 1`, multiple src1 rows in dims 1/2)
    /// — the MoE final `mul(experts, weights)` shape. Regression test for the
    /// F32 fast path (it once scaled the *uninitialized dst* instead of
    /// reading src0 in the broadcast-mul case).
    #[test]
    fn mul_broadcast_dim0() {
        let mut ctx = Context::new();
        // experts [8, 2, 2] * weights [1, 2, 2]
        let a = ctx.new_tensor_3d(GgmlType::F32, 8, 2, 2);
        ctx.arena_resize_tensor(a);
        ctx.with_f32_mut(a, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = i as f32 + 1.0;
            }
        })
        .unwrap();
        let b = ctx.new_tensor_3d(GgmlType::F32, 1, 2, 2);
        ctx.arena_resize_tensor(b);
        ctx.with_f32_mut(b, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = 10.0 * (i as f32 + 1.0);
            }
        })
        .unwrap();
        let y = ctx.mul(a, b);
        let mut g = Graph::new(8);
        g.build_forward(&ctx, y);
        graph_compute(&mut ctx, &mut g, 1);
        let out = ctx.f32s(y).unwrap().to_vec();
        let av = ctx.f32s(a).unwrap().to_vec();
        let bv = ctx.f32s(b).unwrap().to_vec();
            for (i, o) in out.iter().enumerate() {
                let r = i / 8; // row over (ne1, ne2)
                let want = av[i] * bv[r];
                assert_eq!(o.to_bits(), want.to_bits(), "i={i}: {o} vs {want}");
            }
    }

    // ==================================================================
    // arch batch 7 (deepseek4): the fused dsv4 hyper-connection ops,
    // swiglu_clamp, sqrt and rope_ext_back vs the reference CPU backend —
    // the dump is produced by parity/ref_dsv4_dump.c against the pinned
    // tree's libggml (bd4f514db1), bit-exact f32 by construction.
    // ==================================================================

    /// the probe's LCG (uint32 wrap-around) — identical inputs both sides
    fn dsv4_lcg(seed: u32, n: usize) -> Vec<f32> {
        dsv4_lcg_cont(seed, 0, n)
    }

    /// `skip` draws of the same stream first (the probe fills several tensors
    /// from one continuous lcg sequence)
    fn dsv4_lcg_cont(seed: u32, skip: usize, n: usize) -> Vec<f32> {
        let mut state = seed;
        for _ in 0..skip {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
        }
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                ((state >> 8) & 0xffffff) as f32 / 8388608.0 * 0.2 - 0.1
            })
            .collect()
    }

    /// load parity/dsv4_ops_ref.bin: (section element counts, flat f32 data)
    fn dsv4_ref_dump() -> (Vec<usize>, Vec<f32>) {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/dsv4_ops_ref.bin");
        let bytes = std::fs::read(path).expect("parity/dsv4_ops_ref.bin (run parity/ref_dsv4_dump.c)");
        assert_eq!(&bytes[..4], b"DSV4", "dump magic");
        let n_sections = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut sizes = Vec::with_capacity(n_sections);
        for s in 0..n_sections {
            let v = u32::from_le_bytes(bytes[8 + 4 * s..12 + 4 * s].try_into().unwrap());
            sizes.push(v as usize);
        }
        let data_off = 8 + 4 * n_sections;
        let data: Vec<f32> = bytes[data_off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (sizes, data)
    }

    /// every section of the dump, bit-exact (expf/sqrtf are the same libm on
    /// both sides; the kernels share the C's per-element op order)
    #[test]
    fn dsv4_ops_match_reference_dump() {
        let (sizes, ref_data) = dsv4_ref_dump();
        let mut got: Vec<f32> = Vec::new();
        let mut push = |ctx: &Context, t: TensorId| {
            let d = ctx.f32s(t).unwrap();
            got.extend_from_slice(d);
        };

        // -- dsv4_hc_comb: mixes [24, 3], scale [3], base [24], eps/n_iter vars
        {
            let mut ctx = Context::new();
            let mixes = ctx.new_tensor_2d(GgmlType::F32, 24, 3);
            let scale = ctx.new_tensor_1d(GgmlType::F32, 3);
            let base = ctx.new_tensor_1d(GgmlType::F32, 24);
            for t in [mixes, scale, base] {
                ctx.arena_resize_tensor(t);
            }
            let vals = dsv4_lcg(0xd544_0001, 24 * 3);
            ctx.with_f32_mut(mixes, |p| p.copy_from_slice(&vals)).unwrap();
            ctx.with_f32_mut(scale, |p| {
                p[0] = 0.9;
                p[1] = 1.1;
                p[2] = 1.3;
            })
            .unwrap();
            // base continues the same lcg stream after the 24*3 mixes draws
            // (scale is hard-coded in the probe, not drawn)
            let bvals = dsv4_lcg_cont(0xd544_0001, 24 * 3, 24)
                .iter()
                .map(|v| 0.02 * v)
                .collect::<Vec<_>>();
            ctx.with_f32_mut(base, |p| p.copy_from_slice(&bvals)).unwrap();

            let eps_list = [1e-3f32, 1e-2f32];
            let iter_list = [1i32, 3];
            let outs: Vec<TensorId> = (0..4)
                .map(|v| {
                    ctx.dsv4_hc_comb(mixes, scale, base, eps_list[v & 1], iter_list[v >> 1])
                })
                .collect();
            for &o in &outs {
                let mut g = Graph::new(8);
                g.build_forward(&ctx, o);
                graph_compute(&mut ctx, &mut g, 1);
            }
            for &o in &outs {
                push(&ctx, o);
            }
        }

        // -- dsv4_hc_pre: x [12, 4, 3], weights [4, 3]
        {
            let mut ctx = Context::new();
            let x = ctx.new_tensor_3d(GgmlType::F32, 12, 4, 3);
            let w = ctx.new_tensor_2d(GgmlType::F32, 4, 3);
            for t in [x, w] {
                ctx.arena_resize_tensor(t);
            }
            let xv = dsv4_lcg(0x1234_abcd, 12 * 4 * 3);
            ctx.with_f32_mut(x, |p| p.copy_from_slice(&xv)).unwrap();
            // weights continue the same lcg stream after x
            let wv = dsv4_lcg_cont(0x1234_abcd, 12 * 4 * 3, 4 * 3)
                .iter()
                .map(|v| 0.5 + 0.5 * v)
                .collect::<Vec<_>>();
            ctx.with_f32_mut(w, |p| p.copy_from_slice(&wv)).unwrap();

            let pre = ctx.dsv4_hc_pre(x, w);
            let mut g = Graph::new(8);
            g.build_forward(&ctx, pre);
            graph_compute(&mut ctx, &mut g, 1);
            push(&ctx, pre);
        }

        // -- dsv4_hc_post: x [10, 2], residual [10, 4, 2], post [4, 2],
        //    comb [4, 4, 2] — with comb and with comb == NULL
        {
            let mut ctx = Context::new();
            let x = ctx.new_tensor_2d(GgmlType::F32, 10, 2);
            let r = ctx.new_tensor_3d(GgmlType::F32, 10, 4, 2);
            let po = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
            let cb = ctx.new_tensor_3d(GgmlType::F32, 4, 4, 2);
            for t in [x, r, po, cb] {
                ctx.arena_resize_tensor(t);
            }
            let mut state = 0xbeef_0042u32;
            let mut draw = || {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                ((state >> 8) & 0xffffff) as f32 / 8388608.0 * 0.2 - 0.1
            };
            ctx.with_f32_mut(x, |p| {
                for v in p.iter_mut() {
                    *v = draw();
                }
            })
            .unwrap();
            ctx.with_f32_mut(r, |p| {
                for v in p.iter_mut() {
                    *v = draw();
                }
            })
            .unwrap();
            ctx.with_f32_mut(po, |p| {
                for v in p.iter_mut() {
                    *v = 0.5 + 0.5 * draw();
                }
            })
            .unwrap();
            ctx.with_f32_mut(cb, |p| {
                for v in p.iter_mut() {
                    *v = 0.1 * draw() + 0.05;
                }
            })
            .unwrap();

            let with_comb = ctx.dsv4_hc_post(x, r, po, Some(cb));
            let no_comb = ctx.dsv4_hc_post(x, r, po, None);
            for &o in &[with_comb, no_comb] {
                let mut g = Graph::new(8);
                g.build_forward(&ctx, o);
                graph_compute(&mut ctx, &mut g, 1);
            }
            push(&ctx, with_comb);
            push(&ctx, no_comb);
        }

        // -- swiglu_clamp: gate/up [37, 4] at limits 7.0 / 0.05 / 1e-6
        {
            let mut ctx = Context::new();
            let gate = ctx.new_tensor_2d(GgmlType::F32, 37, 4);
            let up = ctx.new_tensor_2d(GgmlType::F32, 37, 4);
            for t in [gate, up] {
                ctx.arena_resize_tensor(t);
            }
            let mut state = 0x5eed_5f01u32;
            let mut draw = || {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                30.0 * (((state >> 8) & 0xffffff) as f32 / 8388608.0 * 0.2 - 0.1)
            };
            ctx.with_f32_mut(gate, |p| {
                for v in p.iter_mut() {
                    *v = draw();
                }
            })
            .unwrap();
            ctx.with_f32_mut(up, |p| {
                for v in p.iter_mut() {
                    *v = draw();
                }
            })
            .unwrap();

            let limits = [7.0f32, 0.05, 1e-6];
            let outs: Vec<TensorId> =
                limits.iter().map(|&l| ctx.swiglu_clamp(gate, up, l)).collect();
            for &o in &outs {
                let mut g = Graph::new(8);
                g.build_forward(&ctx, o);
                graph_compute(&mut ctx, &mut g, 1);
            }
            for &o in &outs {
                push(&ctx, o);
            }
        }

        // -- sqrt: [128] of non-negative values scaled by 100
        {
            let mut ctx = Context::new();
            let a = ctx.new_tensor_1d(GgmlType::F32, 128);
            ctx.arena_resize_tensor(a);
            let vals = dsv4_lcg(0xcafe_f00d, 128);
            ctx.with_f32_mut(a, |p| {
                for (v, s) in p.iter_mut().zip(vals) {
                    let mut v0 = s;
                    if v0 < 0.0 {
                        v0 = -v0;
                    }
                    *v = 100.0 * v0;
                }
            })
            .unwrap();
            let s = ctx.sqrt(a);
            let mut g = Graph::new(8);
            g.build_forward(&ctx, s);
            graph_compute(&mut ctx, &mut g, 1);
            push(&ctx, s);
        }

        // -- rope_ext_back: [16, 3, 5] NEOX, n_dims 8, offset 8, pos 40+7i
        {
            let mut ctx = Context::new();
            let cur = ctx.new_tensor_3d(GgmlType::F32, 16, 3, 5);
            let pos = ctx.new_tensor_1d(GgmlType::I32, 5);
            for t in [cur, pos] {
                ctx.arena_resize_tensor(t);
            }
            let vals = dsv4_lcg(0x1234_5678, 16 * 3 * 5);
            ctx.with_f32_mut(cur, |p| p.copy_from_slice(&vals)).unwrap();
            ctx.with_i32_mut(pos, |p| {
                for (i, v) in p.iter_mut().enumerate() {
                    *v = 40 + 7 * i as i32;
                }
            })
            .unwrap();
            let rb = ctx.rope_ext_back(
                cur, pos, None, 8, 2 /* NEOX */, 0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            );
            let rb = ctx.rope_set_offset(rb, 8);
            let mut g = Graph::new(8);
            g.build_forward(&ctx, rb);
            graph_compute(&mut ctx, &mut g, 1);
            push(&ctx, rb);
        }

        // the section split must agree, then every f32 bit
        let mut off = 0usize;
        let mut sec = 0usize;
        let mut cursor = 0usize;
        while cursor < got.len() {
            assert!(sec < sizes.len(), "more sections computed than the dump has");
            let n = sizes[sec];
            for k in 0..n {
                let g = got[cursor + k];
                let r = ref_data[off + k];
                assert_eq!(
                    g.to_bits(),
                    r.to_bits(),
                    "section {sec} elem {k}: {g} vs {r}"
                );
            }
            off += n;
            cursor += n;
            sec += 1;
        }
        assert_eq!(sec, sizes.len(), "section count mismatch");
    }
