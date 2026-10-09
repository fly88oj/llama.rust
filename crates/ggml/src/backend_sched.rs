//! backend_sched — port of ggml's backend scheduler (the graph splitter) plus
//! the graph allocator (gallocr) it drives.
//!
//! Reference (llama.cpp pinned @ bd4f514db1) — note that at this revision
//! there is no separate `ggml-backend-sched.cpp`: the scheduler lives inside
//! `ggml/src/ggml-backend.cpp` (lines 761-2104) and the allocator in
//! `ggml/src/ggml-alloc.c`.
//!
//! What is ported:
//!   ggml-backend.cpp:761-1589   — `ggml_backend_sched` + `ggml_backend_sched_split_graph`
//!                                 (the 5-pass backend assignment + split/copy insertion)
//!   ggml-backend.cpp:1591-1644  — `ggml_backend_sched_alloc_splits`
//!   ggml-backend.cpp:1646-1846  — `ggml_backend_sched_compute_splits` (the
//!                                 run pipeline: input copies, compute, events)
//!   ggml-backend.cpp:1848-2104  — the public sched API
//!   ggml-alloc.c:108-380        — `ggml_dyn_tallocr` (free-block allocator)
//!   ggml-alloc.c:394-1100       — `ggml_gallocr` (measure/reserve/alloc)
//!
//! Sync increment a7b94df2c -> c35b66744 (6753a033f "refactor selective
//! expert copying to user code"): the scheduler-side selective expert copy
//! was deleted, input copying was extracted into
//! `ggml_backend_sched_copy_input` (ggml-backend.cpp:1811-1842 @c35b66744)
//! behind the `ggml_backend_sched_is_host_weight` predicate (1805-1809),
//! split inputs are now copied in two passes — non-host-weights first, then
//! host weights — so the new `ggml_backend_sched_copy_callback`
//! (ggml-backend.h:317-327, set via `ggml_backend_sched_set_copy_callback`,
//! ggml-backend.cpp:2140-2143) sees the split's non-weight inputs already
//! copied; the public sched API now ends at ggml-backend.cpp:2146.
//!
//! Adaptations forced by the port's tensor model (see backend.rs header):
//!   * the C pointer hash set (`hash_id(tensor)`) is a `HashMap<TensorId, _>`
//!     — TensorId is the port's stable tensor identity;
//!   * C's `sched->ctx` (the private context holding the copy tensors,
//!     `ggml_free`d at every split) becomes a truncation of the shared
//!     Context: the scheduler remembers the tensor range it created and rolls
//!     it back on the next split when the tail is still exactly its own;
//!   * the gallocr's multi-chunk vbuffer machinery (ggml-alloc.c:104-108,
//!     needed for buffer types with a finite max size) is collapsed to the
//!     single-chunk case: with `get_max_size == SIZE_MAX` (every buffer type
//!     this port registers) C never opens a second chunk either — the
//!     "last chunk has virtually endless memory" rule (ggml-alloc.c:167-169)
//!     is what the port implements.

use crate::graph::Graph;
use crate::tensor::{Context, GgmlOp, Storage, TensorId};
use crate::types::MAX_SRC;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::*;

// ggml-backend.cpp:763-773
pub const GGML_SCHED_MAX_BACKENDS: usize = 16;
pub const GGML_SCHED_MAX_SPLIT_INPUTS: usize = 30;
pub const GGML_SCHED_MAX_COPIES: usize = 4;

// ===========================================================================
// dynamic tensor allocator — ggml-alloc.c:62-107 + 120-380 (single chunk)
// ===========================================================================

/// `struct free_block` (ggml-alloc.c:90)
#[derive(Clone, Copy)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// `struct ggml_dyn_tallocr` (ggml-alloc.c:120-131) — one chunk (see module
/// header for the single-chunk adaptation).
struct DynTallocr {
    alignment: usize,
    max_chunk_size: usize,
    /// the single chunk's free blocks, sorted by offset
    free_blocks: Vec<FreeBlock>,
    /// `tallocr_chunk::max_size`
    max_size: usize,
}

/// `aligned_offset` (ggml-alloc.c:56) — with buffer == NULL.
fn aligned_offset(offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (offset + alignment - 1) & !(alignment - 1)
}

impl DynTallocr {
    /// `ggml_dyn_tallocr_new` (ggml-alloc.c:322-337) — `max_chunk_size =
    /// MIN(max_buffer_size, SIZE_MAX/2)` clamped to avoid overflows.
    fn new(alignment: usize, max_buffer_size: usize) -> Self {
        DynTallocr {
            alignment,
            max_chunk_size: max_buffer_size.min(usize::MAX / 2),
            free_blocks: Vec::new(),
            max_size: 0,
        }
    }

    /// `ggml_dyn_tallocr_reset` (ggml-alloc.c:307-319)
    fn reset(&mut self) {
        self.free_blocks.clear();
        self.max_size = 0;
    }

    /// `ggml_dyn_tallocr_insert_block` (ggml-alloc.c:133-148) — keeps the
    /// array sorted by offset to make merging blocks faster
    fn insert_block(&mut self, offset: usize, size: usize) {
        let insert_pos = self.free_blocks.partition_point(|b| b.offset < offset);
        self.free_blocks.insert(insert_pos, FreeBlock { offset, size });
    }

    /// `ggml_dyn_tallocr_alloc` (ggml-alloc.c:212-305). Returns the offset
    /// (the C `buffer_address.offset`; the chunk is always 0 here).
    fn alloc(&mut self, size: usize) -> usize {
        let size = aligned_offset(size, self.alignment);
        let _ = self.max_chunk_size; // single chunk: the cap is virtual (see module header)

        // find the best fitting free block besides the last block
        let mut best_fit: Option<usize> = None; // block index
        let mut best_fit_size = usize::MAX;

        for i in 0..self.free_blocks.len().saturating_sub(1) {
            let block = self.free_blocks[i];
            if block.size >= size && block.size <= best_fit_size {
                best_fit = Some(i);
                best_fit_size = block.size;
            }
        }

        if best_fit.is_none() {
            // no suitable block found, try the last block (this may grow the
            // chunk's size). C ranks the candidate last blocks of each chunk
            // by a reuse factor (alloc.c:265-280); with a single chunk there
            // is exactly one candidate, so the factor comparison degenerates
            // to "does it fit".
            if let Some(li) = self.free_blocks.len().checked_sub(1) {
                let block = self.free_blocks[li];
                let reuse_factor = self.max_size as i64 - block.offset as i64 - size as i64;
                let _ = reuse_factor;
                if block.size >= size {
                    best_fit = Some(li);
                }
            }
        }

        if best_fit.is_none() {
            // none of the existing space is enough — C would open a new chunk
            // (ggml_dyn_tallocr_new_chunk, alloc.c:155) and, at the chunk
            // limit, give the last one virtually endless memory (alloc.c:167).
            // The single-chunk port follows that rule directly.
            let offset = self.max_size;
            self.free_blocks.push(FreeBlock {
                offset,
                size: usize::MAX / 2,
            });
            best_fit = Some(self.free_blocks.len() - 1);
        }

        let idx = best_fit.unwrap();
        let block = &mut self.free_blocks[idx];
        let addr = block.offset;
        block.offset += size;
        block.size = block.size.saturating_sub(size);
        if block.size == 0 {
            // remove block if empty (alloc.c:290-292)
            self.free_blocks.remove(idx);
        }

        self.max_size = self.max_size.max(addr + size);
        addr
    }

    /// `ggml_dyn_tallocr_free_bytes` (ggml-alloc.c:247-280) — naive merging;
    /// the number of free blocks is very small in practice.
    fn free_bytes(&mut self, offset: usize, size: usize) {
        let size = aligned_offset(size, self.alignment);

        // see if we can merge with an existing block
        for i in 0..self.free_blocks.len() {
            let block = self.free_blocks[i];
            // ptr is at the end of the block
            if block.offset + block.size == offset {
                self.free_blocks[i].size += size;
                // merge with the next block
                if i + 1 < self.free_blocks.len()
                    && self.free_blocks[i].offset + self.free_blocks[i].size == self.free_blocks[i + 1].offset
                {
                    self.free_blocks[i].size += self.free_blocks[i + 1].size;
                    self.free_blocks.remove(i + 1);
                }
                return;
            }
            // ptr is at the beginning of the block
            if offset + size == block.offset {
                self.free_blocks[i].offset = offset;
                self.free_blocks[i].size += size;
                // merge with the previous block
                if i > 0 && self.free_blocks[i - 1].offset + self.free_blocks[i - 1].size == self.free_blocks[i].offset
                {
                    self.free_blocks[i - 1].size += self.free_blocks[i].size;
                    self.free_blocks.remove(i);
                }
                return;
            }
        }
        // otherwise, add a new block
        self.insert_block(offset, size);
    }
}

// ===========================================================================
// graph allocator — ggml-alloc.c:394-1100
// ===========================================================================

/// `struct hash_node` (ggml-alloc.c:397)
#[derive(Clone, Copy, Default)]
struct GallocHashNode {
    n_children: i32,
    n_views: i32,
    buffer_id: i32,
    offset: usize, // buffer_address.offset (chunk is always 0)
    allocated: bool,
}

/// `struct tensor_alloc` (ggml-alloc.c:406) — `GGML_BUFFER_ADDRESS_INVALID`
/// is `buffer_id == -1` (offset SIZE_MAX, size 0 = pre-allocated/unused/view).
#[derive(Clone, Copy)]
struct TensorAlloc {
    buffer_id: i32,
    offset: usize,
    size_max: usize,
}

impl TensorAlloc {
    const INVALID: TensorAlloc = TensorAlloc { buffer_id: -1, offset: usize::MAX, size_max: 0 };

    fn is_invalid(&self) -> bool {
        self.buffer_id < 0
    }
}

/// `struct ggml_gallocr` (ggml-alloc.c:421-437). `buf_tallocs` are shared
/// between buffer indices of the same buffer type (ggml_gallocr_new_n,
/// alloc.c:516-521).
pub(crate) struct Gallocr {
    bufts: Vec<BackendBufferTypeRef>,
    /// buffer index → talloc group (the C pointer-shared `buf_tallocs[i]`)
    talloc_of: Vec<usize>,
    tallocs: Vec<DynTallocr>,
    /// `struct vbuffer **buffers` — the reserved backing buffers (shared for
    /// equal buffer types, like the tallocs)
    buffers: Vec<Option<BackendBufferRef>>,
    hash: HashMap<TensorId, GallocHashNode>,
    node_allocs: Vec<TensorAlloc>,
    /// `node_allocs[i].src[j]` (alloc.c:419)
    node_src_allocs: Vec<Vec<Option<TensorAlloc>>>,
    leaf_allocs: Vec<TensorAlloc>,
}

impl Gallocr {
    /// `ggml_gallocr_new_n` (ggml-alloc.c:498-540)
    fn new(bufts: &[BackendBufferTypeRef]) -> Self {
        let mut talloc_of = Vec::with_capacity(bufts.len());
        let mut tallocs = Vec::new();
        for (i, b) in bufts.iter().enumerate() {
            // check if the same buffer type is used multiple times and reuse
            // the same allocator
            let group = bufts[..i].iter().position(|j| buft_eq(j, b));
            let group = match group {
                Some(g) => g,
                None => {
                    tallocs.push(DynTallocr::new(backend_buft_get_alignment(b), backend_buft_get_max_size(b)));
                    tallocs.len() - 1
                }
            };
            talloc_of.push(group);
        }
        Gallocr {
            bufts: bufts.to_vec(),
            talloc_of,
            tallocs,
            buffers: vec![None; bufts.len()],
            hash: HashMap::new(),
            node_allocs: Vec::new(),
            node_src_allocs: Vec::new(),
            leaf_allocs: Vec::new(),
        }
    }

    /// `ggml_gallocr_hash_get` (ggml-alloc.c:514)
    fn hash_get(&mut self, t: TensorId) -> &mut GallocHashNode {
        self.hash.entry(t).or_default()
    }

    /// `ggml_gallocr_is_own` (ggml-alloc.c:519)
    fn is_own(&mut self, t: TensorId) -> bool {
        self.hash_get(t).allocated
    }

    /// `ggml_gallocr_is_allocated` (ggml-alloc.c:524) — `t->data != NULL ||
    /// t->buffer || is_own`. The port's "data != NULL" is "has storage" and
    /// `t->buffer` is the backend side table.
    fn is_allocated(&mut self, ctx: &Context, t: TensorId) -> bool {
        !matches!(ctx.tensors[t.0 as usize].storage, Storage::None)
            || tensor_buffer(ctx, t).is_some()
            || self.is_own(t)
    }

    /// `ggml_gallocr_free_extra_space` (ggml-alloc.c:532-553)
    fn free_extra_space(&mut self, ctx: &Context, node: TensorId, parent: TensorId) {
        let p_hn = *self.hash_get(parent);
        let hn = *self.hash_get(node);

        let parent_size = backend_buft_get_alloc_size(&self.bufts[p_hn.buffer_id as usize], ctx, parent);
        let node_size = backend_buft_get_alloc_size(&self.bufts[hn.buffer_id as usize], ctx, node);
        assert!(parent_size >= node_size);

        // we want the chunks to continue to be aligned after the free
        let alignment = self.tallocs[self.talloc_of[p_hn.buffer_id as usize]].alignment;
        let parent_size = aligned_offset(parent_size, alignment);
        let node_size = aligned_offset(node_size, alignment);

        if parent_size > node_size {
            let p_offset = p_hn.offset + node_size;
            let extra_size = parent_size - node_size;
            self.tallocs[self.talloc_of[p_hn.buffer_id as usize]].free_bytes(p_offset, extra_size);
        }
    }

    /// `ggml_gallocr_allocate_node` (ggml-alloc.c:555-637)
    fn allocate_node(&mut self, ctx: &Context, node: TensorId, buffer_id: i32) {
        assert!(buffer_id >= 0);

        if self.is_allocated(ctx, node) || ctx.tensors[node.0 as usize].view_src.is_some() {
            return;
        }
        self.hash_get(node).allocated = true;

        // try to reuse a parent's buffer (inplace)
        if op_can_inplace(ctx.op(node)) {
            for i in 0..MAX_SRC {
                let Some(parent) = ctx.tensors[node.0 as usize].src[i] else { continue };

                // if the node's data is external, then we cannot re-use it
                if !self.is_own(parent) {
                    continue;
                }

                // outputs cannot be reused
                let out_flag = |t: TensorId| -> bool {
                    let base = match ctx.tensors[t.0 as usize].view_src {
                        Some(v) => v,
                        None => t,
                    };
                    ctx.tensors[base.0 as usize].flags & GGML_TENSOR_FLAG_OUTPUT != 0
                };
                if out_flag(parent) {
                    continue;
                }

                if !are_same_layout(ctx, node, parent) {
                    continue;
                }

                let p_hn = *self.hash_get(parent);
                if p_hn.n_children == 1 && p_hn.n_views == 0 {
                    if ctx.tensors[parent.0 as usize].view_src.is_some() {
                        let view_src = ctx.tensors[parent.0 as usize].view_src.unwrap();
                        let view_src_hn = *self.hash_get(view_src);
                        if view_src_hn.n_views == 1 && view_src_hn.n_children == 0 && view_src_hn.allocated {
                            let hn = self.hash_get(node);
                            hn.buffer_id = p_hn.buffer_id;
                            hn.offset = p_hn.offset;
                            self.hash_get(parent).allocated = false; // avoid freeing the parent
                            self.hash_get(view_src).allocated = false;
                            self.free_extra_space(ctx, node, view_src);
                            return;
                        }
                    } else {
                        let hn = self.hash_get(node);
                        hn.buffer_id = p_hn.buffer_id;
                        hn.offset = p_hn.offset;
                        self.hash_get(parent).allocated = false;
                        self.free_extra_space(ctx, node, parent);
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        let size = backend_buft_get_alloc_size(&self.bufts[buffer_id as usize], ctx, node);
        let group = self.talloc_of[buffer_id as usize];
        let offset = self.tallocs[group].alloc(size);
        let hn = self.hash_get(node);
        hn.buffer_id = buffer_id;
        hn.offset = offset;
    }

    /// `ggml_gallocr_free_node` (ggml-alloc.c:639-662)
    fn free_node(&mut self, ctx: &Context, node: TensorId) {
        // graph outputs are never freed
        if ctx.tensors[node.0 as usize].flags & GGML_TENSOR_FLAG_OUTPUT != 0 {
            return;
        }

        let hn = *self.hash_get(node);
        let group = self.talloc_of[hn.buffer_id as usize];
        let size = backend_buft_get_alloc_size(&self.bufts[hn.buffer_id as usize], ctx, node);
        self.tallocs[group].free_bytes(hn.offset, size);
        self.hash_get(node).allocated = false;
    }

    /// `ggml_gallocr_alloc_graph_impl` (ggml-alloc.c:718-846)
    fn alloc_graph_impl(
        &mut self,
        ctx: &Context,
        nodes: &[TensorId],
        leafs: &[TensorId],
        node_buffer_ids: &[i32],
        leaf_buffer_ids: &[i32],
    ) {
        // clear hash tables
        self.hash.clear();

        let node_bid = |i: usize| -> i32 { node_buffer_ids.get(i).copied().unwrap_or(0) };
        let leaf_bid = |i: usize| -> i32 { leaf_buffer_ids.get(i).copied().unwrap_or(0) };

        // allocate leafs
        for (i, &leaf) in leafs.iter().enumerate() {
            self.allocate_node(ctx, leaf, leaf_bid(i));
        }

        // count number of children and views; allocate other graph inputs
        // and leafs first to avoid overwriting them
        for (i, &node) in nodes.iter().enumerate() {
            // GGML_OP_NONE dep nodes use their srcs only as dependencies
            if ctx.tensors[node.0 as usize].view_src.is_some() && ctx.op(node) != GgmlOp::None {
                let view_src = ctx.tensors[node.0 as usize].view_src.unwrap();
                self.hash_get(view_src).n_views += 1;
            }

            if ctx.tensors[node.0 as usize].flags & GGML_TENSOR_FLAG_INPUT != 0 {
                self.allocate_node(ctx, node, node_bid(i));
            }

            for j in 0..MAX_SRC {
                let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
                self.hash_get(src).n_children += 1;

                // allocate explicit inputs
                if ctx.tensors[src.0 as usize].flags & GGML_TENSOR_FLAG_INPUT != 0 {
                    self.allocate_node(ctx, src, node_bid(i));
                }
            }
        }

        // allocate tensors
        for (i, &node) in nodes.iter().enumerate() {
            let buffer_id = node_bid(i);

            // allocate parents (only leafs need to be allocated at this point)
            for j in 0..MAX_SRC {
                let Some(parent) = ctx.tensors[node.0 as usize].src[j] else { continue };
                self.allocate_node(ctx, parent, buffer_id);
            }

            // allocate node
            self.allocate_node(ctx, node, buffer_id);

            // update parents
            for j in 0..MAX_SRC {
                let Some(parent) = ctx.tensors[node.0 as usize].src[j] else { continue };
                let p_hn = self.hash_get(parent);
                p_hn.n_children -= 1;

                if p_hn.n_children == 0 && p_hn.n_views == 0 {
                    if ctx.tensors[parent.0 as usize].view_src.is_some() {
                        let view_src = ctx.tensors[parent.0 as usize].view_src.unwrap();
                        let view_src_hn = self.hash_get(view_src);
                        view_src_hn.n_views -= 1;
                        if view_src_hn.n_views == 0 && view_src_hn.n_children == 0 && view_src_hn.allocated {
                            self.free_node(ctx, view_src);
                        }
                    } else if p_hn.allocated {
                        self.free_node(ctx, parent);
                    }
                }
            }
        }
    }

    /// the per-tensor `tensor_alloc` (ggml-alloc.c:876-897): INVALID for
    /// views / pre-allocated tensors, else from the hash
    fn tensor_alloc_of(&self, ctx: &Context, t: TensorId) -> TensorAlloc {
        if ctx.tensors[t.0 as usize].view_src.is_some() || !matches!(ctx.tensors[t.0 as usize].storage, Storage::None) {
            return TensorAlloc::INVALID;
        }
        match self.hash.get(&t) {
            Some(hn) if hn.buffer_id >= 0 => TensorAlloc {
                buffer_id: hn.buffer_id,
                offset: hn.offset,
                size_max: backend_buft_get_alloc_size(&self.bufts[hn.buffer_id as usize], ctx, t),
            },
            _ => TensorAlloc::INVALID,
        }
    }

    /// `ggml_gallocr_reserve_n_impl` (ggml-alloc.c:848-949)
    fn reserve_n_impl(
        &mut self,
        ctx: &mut Context,
        nodes: &[TensorId],
        leafs: &[TensorId],
        node_buffer_ids: &[i32],
        leaf_buffer_ids: &[i32],
    ) -> bool {
        // reset allocators
        for t in &mut self.tallocs {
            t.reset();
        }

        // allocate in hash table
        self.alloc_graph_impl(ctx, nodes, leafs, node_buffer_ids, leaf_buffer_ids);

        // set the node_allocs from the hash table
        self.node_allocs = nodes.iter().map(|&n| self.tensor_alloc_of(ctx, n)).collect();
        self.node_src_allocs = nodes
            .iter()
            .map(|&n| {
                let mut srcs = vec![None; MAX_SRC];
                for (j, s) in ctx.tensors[n.0 as usize].src.iter().enumerate() {
                    if let Some(src) = s {
                        srcs[j] = Some(self.tensor_alloc_of(ctx, *src));
                    }
                }
                srcs
            })
            .collect();
        self.leaf_allocs = leafs.iter().map(|&l| self.tensor_alloc_of(ctx, l)).collect();

        // reallocate buffers if needed
        for i in 0..self.bufts.len() {
            // if the buffer type is used multiple times, we reuse the same buffer
            for j in 0..i {
                if self.talloc_of[j] == self.talloc_of[i] {
                    self.buffers[i] = self.buffers[j].clone();
                    break;
                }
            }

            // even if there are no tensors allocated in this buffer, we still
            // need to allocate it to initialize views
            let mut realloc = self.buffers[i].is_none();
            let new_size = self.tallocs[self.talloc_of[i]].max_size;
            if let Some(cur) = &self.buffers[i] {
                if new_size > cur.size {
                    realloc = true;
                }
            }
            if realloc {
                self.buffers[i] = None;
                // ggml_vbuffer_alloc (ggml-alloc.c:376-392) — one chunk per
                // buffer here, USAGE_COMPUTE
                let buft = self.bufts[i].clone();
                let Some(buffer) = backend_buft_alloc_buffer(&buft, ctx, new_size) else {
                    eprintln!("ggml_gallocr_reserve_n: failed to allocate buffer of size {new_size}");
                    return false;
                };
                backend_buffer_set_usage(&buffer, BackendBufferUsage::Compute);
                self.buffers[i] = Some(buffer);
            }
        }
        true
    }

    /// `ggml_gallocr_reserve_n` (ggml-alloc.c:962)
    fn reserve_n(
        &mut self,
        ctx: &mut Context,
        nodes: &[TensorId],
        leafs: &[TensorId],
        node_buffer_ids: &[i32],
        leaf_buffer_ids: &[i32],
    ) -> bool {
        self.reserve_n_impl(ctx, nodes, leafs, node_buffer_ids, leaf_buffer_ids)
    }

    /// `ggml_gallocr_reserve_n_size` (ggml-alloc.c:951-959) — the measure
    /// pass without buffer allocation (the C `no_alloc` mode).
    fn reserve_n_size(
        &mut self,
        ctx: &mut Context,
        nodes: &[TensorId],
        leafs: &[TensorId],
        node_buffer_ids: &[i32],
        leaf_buffer_ids: &[i32],
        sizes: &mut [usize],
    ) {
        for t in &mut self.tallocs {
            t.reset();
        }
        self.alloc_graph_impl(ctx, nodes, leafs, node_buffer_ids, leaf_buffer_ids);
        for (i, size) in sizes.iter_mut().enumerate() {
            *size = if i < self.talloc_of.len() && self.talloc_of[i] < self.tallocs.len() {
                self.tallocs[self.talloc_of[i]].max_size
            } else {
                0
            };
        }
    }

    /// `ggml_gallocr_init_tensor` (ggml-alloc.c:979-1002)
    fn init_tensor(&mut self, ctx: &mut Context, tensor: TensorId, tensor_alloc: &TensorAlloc) {
        if ctx.tensors[tensor.0 as usize].view_src.is_some() {
            if tensor_buffer(ctx, tensor).is_none() {
                debug_assert_eq!(tensor_alloc.offset, usize::MAX);
                let view_src = ctx.tensors[tensor.0 as usize].view_src.unwrap();
                if tensor_buffer(ctx, view_src).is_none() {
                    // this tensor was allocated without ggml-backend
                    return;
                }
                let _ = backend_view_init(ctx, tensor);
            }
        } else if matches!(ctx.tensors[tensor.0 as usize].storage, Storage::None) {
            debug_assert!(!tensor_alloc.is_invalid());
            let buffer = self.buffers[tensor_alloc.buffer_id as usize]
                .as_ref()
                .expect("gallocr buffer not reserved")
                .clone();
            // ggml_vbuffer_tensor_alloc (ggml-alloc.c:399-404)
            let base = backend_buffer_get_base(&buffer).unwrap_or(0);
            let _ = backend_tensor_alloc(&buffer, ctx, tensor, base + tensor_alloc.offset);
        }
    }

    /// `ggml_gallocr_node_needs_realloc` (ggml-alloc.c:1004-1015)
    fn node_needs_realloc(&self, ctx: &Context, node: TensorId, talloc: &TensorAlloc) -> bool {
        let mut node_size = 0;
        if matches!(ctx.tensors[node.0 as usize].storage, Storage::None)
            && ctx.tensors[node.0 as usize].view_src.is_none()
        {
            // if we previously had data but don't now then reallocate
            if talloc.buffer_id < 0 {
                return false;
            }
            node_size = backend_buft_get_alloc_size(&self.bufts[talloc.buffer_id as usize], ctx, node);
        }
        talloc.size_max >= node_size
    }

    /// `ggml_gallocr_needs_realloc` (ggml-alloc.c:1017-1050)
    fn needs_realloc(&self, ctx: &Context, nodes: &[TensorId], leafs: &[TensorId]) -> bool {
        if self.node_allocs.len() != nodes.len() || self.leaf_allocs.len() != leafs.len() {
            return true;
        }

        for (i, &node) in nodes.iter().enumerate() {
            if !self.node_needs_realloc(ctx, node, &self.node_allocs[i]) {
                return true;
            }
            for (j, src) in ctx.tensors[node.0 as usize].src.iter().enumerate() {
                let Some(src) = src else { continue };
                match self.node_src_allocs[i].get(j).copied().flatten() {
                    None => {} // view / pre-allocated at reserve time — fine
                    Some(ta) => {
                        if !self.node_needs_realloc(ctx, *src, &ta) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// `ggml_gallocr_alloc_graph` (ggml-alloc.c:1052-1098)
    fn alloc_graph(
        &mut self,
        ctx: &mut Context,
        nodes: &[TensorId],
        leafs: &[TensorId],
        node_buffer_ids: &[i32],
        leaf_buffer_ids: &[i32],
    ) -> bool {
        if self.needs_realloc(ctx, nodes, leafs) {
            if self.bufts.len() == 1 {
                // reallocating buffers automatically
                if !self.reserve_n(ctx, nodes, leafs, node_buffer_ids, leaf_buffer_ids) {
                    return false;
                }
            } else {
                // cannot reallocate multi buffer graph automatically, call reserve
                return false;
            }
        }

        // reset buffers (ggml_vbuffer_reset, alloc.c:406-410) — none of the
        // ported buffer types carry reset state
        let mut scratch = Context::default();
        for b in self.buffers.iter().flatten() {
            backend_buffer_reset(b, &mut scratch);
        }

        // allocate the graph tensors from the previous assignments — leafs
        for (i, &leaf) in leafs.iter().enumerate() {
            let ta = self.leaf_allocs[i];
            self.init_tensor(ctx, leaf, &ta);
        }
        // nodes
        for (i, &node) in nodes.iter().enumerate() {
            for j in 0..MAX_SRC {
                let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
                if let Some(ta) = self.node_src_allocs[i][j] {
                    self.init_tensor(ctx, src, &ta);
                }
            }
            let ta = self.node_allocs[i];
            self.init_tensor(ctx, node, &ta);
        }
        true
    }

    /// `ggml_gallocr_get_buffer_size` (ggml-alloc.c:1100-1117)
    fn get_buffer_size(&self, buffer_id: usize) -> usize {
        let Some(buffer) = self.buffers[buffer_id].as_ref() else { return 0 };
        for b in self.buffers.iter().take(buffer_id) {
            if let Some(b) = b {
                if Arc::ptr_eq(b, buffer) {
                    // same buffer as a previous one (same buffer type used
                    // multiple times) — only count it the first time
                    return 0;
                }
            }
        }
        buffer.size
    }
}

/// `ggml_op_can_inplace` (ggml-alloc.c:26-53) — ops that return true must not
/// use restrict pointers (their dst may alias src0). The port's op encoding
/// folds GGML_OP_UNARY into `GgmlOp::Silu` and RMS_NORM into `Norm`.
pub(crate) fn op_can_inplace(op: GgmlOp) -> bool {
    matches!(
        op,
        GgmlOp::Fill
            | GgmlOp::Scale
            | GgmlOp::DiagMaskInf
            | GgmlOp::Add
            | GgmlOp::AddId
            | GgmlOp::Sub
            | GgmlOp::Mul
            | GgmlOp::Div
            | GgmlOp::Sqr
            | GgmlOp::Sqrt
            | GgmlOp::Log
            | GgmlOp::Silu // GGML_OP_UNARY
            | GgmlOp::RoPE
            | GgmlOp::RoPEBack
            | GgmlOp::Norm // includes RMS_NORM
            | GgmlOp::Clamp
            | GgmlOp::SoftMax
    )
}

// ===========================================================================
// the scheduler — ggml-backend.cpp:761-2104
// ===========================================================================

/// `struct ggml_backend_sched_split` (ggml-backend.cpp:775-784)
struct SchedSplit {
    backend_id: usize,
    i_start: usize,
    i_end: usize,
    inputs: Vec<TensorId>,
    /// graph view of this split — the port snapshots the node slice (C shares
    /// the nodes array through ggml_graph_view)
    nodes: Vec<TensorId>,
}

/// `ggml_backend_sched_eval_callback` (ggml-backend.h:316) — when `ask` is
/// true the scheduler wants to know if the user wants to observe this node;
/// when false it hands the computed node over and a `false` return cancels
/// the graph compute.
pub type SchedEvalCallback = Box<dyn Fn(&Context, TensorId, bool) -> bool + Send>;

/// `ggml_backend_sched_copy_callback` (ggml-backend.h:317-327 @c35b66744,
/// added by 6753a033f) — callback while copying input weights of a split:
///   * if the user returns false the scheduler simply copies the entire
///     weight;
///   * the callback is called only for input weights in host buffers;
///   * the callback is called after all non-weight inputs of the split have
///     been copied;
///   * `src` is the tensor in the previous split, `dst` is the copy of `src`
///     in the split, `graph` is the compute graph nodes of the split.
///
/// Adaptations (the SchedEvalCallback precedent): the C `user_data` pointer
/// is captured by the closure, and the `ggml_backend_t`/`ggml_cgraph *`
/// arguments become `&BackendRef` / `&[TensorId]` (the port's split graph
/// snapshot). `ctx` is passed mutable so user code can write `dst` through
/// `backend_tensor_set` — the C callback gets raw pointers and can do the
/// same.
pub type SchedCopyCallback = Box<
    dyn Fn(&BackendRef, &mut Context, TensorId, TensorId, &[TensorId]) -> bool + Send,
>;

/// `struct ggml_backend_sched` (ggml-backend.cpp:786-841)
pub struct BackendSched {
    /// true if the scheduler has been reset since the last graph split
    pub(crate) is_reset: bool,
    pub(crate) is_alloc: bool,

    n_backends: usize,
    backends: Vec<BackendRef>,
    bufts: Vec<BackendBufferTypeRef>,
    galloc: Gallocr,

    /// `hv_tensor_backend_ids` — the C `hash_id` / `tensor_backend_id` macros
    /// (the pointer hash set becomes a TensorId map)
    hv_tensor_backend_ids: HashMap<TensorId, i32>,
    /// `hv_tensor_copies` keyed (tensor, backend_id, copy_id)
    hv_tensor_copies: HashMap<(TensorId, usize, usize), TensorId>,

    node_backend_ids: Vec<i32>,
    leaf_backend_ids: Vec<i32>,
    prev_node_backend_ids: Vec<i32>,
    prev_leaf_backend_ids: Vec<i32>,

    /// copy of the graph with modified inputs (`sched->graph`)
    graph_copy_nodes: Vec<TensorId>,
    graph_copy_leafs: Vec<TensorId>,

    splits: Vec<SchedSplit>,

    // pipeline parallelism support
    n_copies: usize,
    cur_copy: usize,
    next_copy: usize,
    events: Vec<Vec<Option<BackendEventRef>>>,
    graph_inputs: Vec<TensorId>,

    callback_eval: Option<SchedEvalCallback>,

    /// `callback_copy` + `callback_copy_user_data` (ggml-backend.cpp:969-970
    /// @c35b66744) — the user_data is captured by the closure
    callback_copy: Option<SchedCopyCallback>,

    op_offload: bool,

    debug: i32,

    /// the port's `sched->ctx` replacement: the tensor range the scheduler
    /// itself created during the current split session (rolled back on the
    /// next split like C's `ggml_free(sched->ctx)`, ggml-backend.cpp:1078)
    sched_tensor_range: Option<(usize, usize)>,
}

/// `ggml_backend_sched_backend_id` (ggml-backend.cpp:879) — priority of the
/// backend, lower id is higher priority.
fn sched_backend_id(sched: &BackendSched, backend: &BackendRef) -> i32 {
    for (i, b) in sched.backends.iter().enumerate() {
        if Arc::ptr_eq(b, backend) {
            return i as i32;
        }
    }
    -1
}

/// `ggml_backend_sched_backend_from_buffer` (ggml-backend.cpp:888) — find
/// highest prio backend that supports the buffer type and the op.
/// (`tensor->view_src ? tensor->view_src->buffer : tensor->buffer` is the
/// port's chain-following `tensor_buffer`.)
fn sched_backend_from_buffer(sched: &BackendSched, ctx: &Context, tensor: TensorId, op: TensorId) -> i32 {
    let Some(buffer) = tensor_buffer(ctx, tensor) else { return -1 };

    for (i, b) in sched.backends.iter().enumerate() {
        if backend_supports_buft(b, &buffer.buft) && backend_supports_op(b, ctx, op) {
            return i as i32;
        }
    }
    -1
}

/// `ggml_backend_sched_backend_id_from_cur` (ggml-backend.cpp:921-985) —
/// returns the backend that should be used for the node based on the current
/// locations. (The C SET_CAUSE debug tags are noted in comments.)
fn sched_backend_id_from_cur(sched: &BackendSched, ctx: &Context, tensor: TensorId) -> i32 {
    // assign pre-allocated nodes to their backend (1.dst)
    let mut cur_backend_id = sched_backend_from_buffer(sched, ctx, tensor, tensor);
    if cur_backend_id != -1 {
        return cur_backend_id;
    }

    // view_src (1.vsrc)
    if let Some(view_src) = ctx.tensors[tensor.0 as usize].view_src {
        cur_backend_id = sched_backend_from_buffer(sched, ctx, view_src, tensor);
        if cur_backend_id != -1 {
            return cur_backend_id;
        }
    }

    if tensor_buffer(ctx, tensor).is_some()
        || ctx.tensors[tensor.0 as usize]
            .view_src
            .is_some_and(|v| tensor_buffer(ctx, v).is_some())
    {
        // since the tensor is pre-allocated, it cannot be moved to another
        // backend (ggml-backend.cpp:938-942 GGML_ABORT)
        let buffer_name = tensor_buffer(ctx, tensor)
            .or_else(|| ctx.tensors[tensor.0 as usize].view_src.and_then(|v| tensor_buffer(ctx, v)))
            .map(|b| backend_buffer_name(&b))
            .unwrap_or_default();
        panic!(
            "pre-allocated tensor ({}) in a buffer ({}) that cannot run the operation ({:?})",
            ctx.tensors[tensor.0 as usize].name,
            buffer_name,
            ctx.tensors[tensor.0 as usize].op
        );
    }

    // graph input (1.inp): last backend (assumed CPU)
    if ctx.tensors[tensor.0 as usize].flags & GGML_TENSOR_FLAG_INPUT != 0 {
        return sched.n_backends as i32 - 1;
    }

    // operations with weights are preferably run on the same backend as the
    // weights (1.wgt) — skip ROPE and FLASH_ATTN_EXT, their freq/sinks
    // tensors are too small to choose a backend based on them
    let allow = !matches!(ctx.tensors[tensor.0 as usize].op, GgmlOp::RoPE | GgmlOp::FlashAttnExt);

    if allow {
        for i in 0..MAX_SRC {
            let Some(src) = ctx.tensors[tensor.0 as usize].src[i] else { continue };
            if let Some(src_buf) = tensor_buffer(ctx, src) {
                if backend_buffer_get_usage(&src_buf) == BackendBufferUsage::Weights {
                    let src_backend_id = sched_backend_from_buffer(sched, ctx, src, tensor);
                    // check if a backend with higher prio wants to offload the
                    // op (1.off)
                    if sched.op_offload
                        && src_backend_id == sched.n_backends as i32 - 1
                        && backend_buffer_is_host(&src_buf)
                    {
                        for b in 0..src_backend_id.max(0) as usize {
                            if backend_supports_op(&sched.backends[b], ctx, tensor)
                                && backend_offload_op(&sched.backends[b], ctx, tensor)
                            {
                                return b as i32;
                            }
                        }
                    }
                    return src_backend_id;
                }
            }
        }
    }

    -1
}

/// `ggml_backend_sched_buffer_supported` (ggml-backend.cpp:1037)
fn sched_buffer_supported(sched: &BackendSched, ctx: &Context, t: TensorId, backend_id: usize) -> bool {
    let buft = if let Some(buf) = tensor_buffer(ctx, t) {
        // the tensor is already allocated
        Some(buf.buft.clone())
    } else {
        // see if the tensor already has a backend assigned, and use the
        // buffer type of that backend
        let mut tensor_backend_id_v = sched.hv_tensor_backend_ids.get(&t).copied().unwrap_or(-1);
        if tensor_backend_id_v == -1 {
            if let Some(vs) = ctx.tensors[t.0 as usize].view_src {
                tensor_backend_id_v = sched.hv_tensor_backend_ids.get(&vs).copied().unwrap_or(-1);
            }
        }
        if tensor_backend_id_v != -1 {
            Some(sched.bufts[tensor_backend_id_v as usize].clone())
        } else {
            None
        }
    };

    buft.is_some_and(|buft| backend_supports_buft(&sched.backends[backend_id], &buft))
}

/// `ggml_backend_sched_set_if_supported` (ggml-backend.cpp:1058)
fn sched_set_if_supported(sched: &BackendSched, ctx: &Context, node: TensorId, cur_backend_id: usize, node_backend_id: &mut i32) {
    if backend_supports_op(&sched.backends[cur_backend_id], ctx, node) {
        *node_backend_id = cur_backend_id as i32;
    }
}

/// `ggml_backend_sched_split_graph` (ggml-backend.cpp:1066-1589) — assigns
/// backends to ops and splits the graph into subgraphs that can be computed
/// on the same backend.
pub fn backend_sched_split_graph(sched: &mut BackendSched, ctx: &mut Context, graph: &mut Graph) {
    // reset splits
    sched.splits.clear();
    sched.graph_inputs.clear();
    sched.is_reset = false;

    // ggml_free(sched->ctx) + re-init: roll back the tensors the scheduler
    // itself created last time when the Context tail is still exactly ours
    if let Some((start, end)) = sched.sched_tensor_range {
        if ctx.tensors.len() == end {
            ctx.reset_graph_to(start);
        }
    }
    sched.sched_tensor_range = Some((ctx.tensors.len(), ctx.tensors.len()));

    // pass 1: assign backends to ops with pre-allocated inputs
    for &leaf in graph.leafs.iter() {
        if sched.hv_tensor_backend_ids.get(&leaf).copied().unwrap_or(-1) == -1 {
            let id = sched_backend_id_from_cur(sched, ctx, leaf);
            sched.hv_tensor_backend_ids.insert(leaf, id);
        }
    }
    for &node in graph.nodes.iter() {
        if sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1) == -1 {
            let id = sched_backend_id_from_cur(sched, ctx, node);
            sched.hv_tensor_backend_ids.insert(node, id);
        }
    }

    // pass 2: expand current backend assignments — assign the same backend
    // to adjacent nodes; expand gpu backends (non last prio) up and down,
    // ignoring cpu (the lowest priority backend); ops unsupported by the
    // backend being expanded are left unassigned
    // expand gpu down
    {
        let mut cur_backend_id: i32 = -1;
        for i in 0..graph.nodes.len() {
            let node = graph.nodes[i];
            if is_view_op(ctx.op(node)) {
                continue;
            }
            let mut node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
            if node_backend_id != -1 {
                if node_backend_id == sched.n_backends as i32 - 1 {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_backend_id;
                }
            } else if cur_backend_id != -1 {
                sched_set_if_supported(sched, ctx, node, cur_backend_id as usize, &mut node_backend_id);
                sched.hv_tensor_backend_ids.insert(node, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        let mut cur_backend_id: i32 = -1;
        for i in (0..graph.nodes.len()).rev() {
            let node = graph.nodes[i];
            if is_view_op(ctx.op(node)) {
                continue;
            }
            let mut node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
            if node_backend_id != -1 {
                if node_backend_id == sched.n_backends as i32 - 1 {
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_backend_id;
                }
            } else if cur_backend_id != -1 {
                sched_set_if_supported(sched, ctx, node, cur_backend_id as usize, &mut node_backend_id);
                sched.hv_tensor_backend_ids.insert(node, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        let mut cur_backend_id: i32 = -1;
        for i in 0..graph.nodes.len() {
            let node = graph.nodes[i];
            if is_view_op(ctx.op(node)) {
                continue;
            }
            let mut node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
            if node_backend_id != -1 {
                cur_backend_id = node_backend_id;
            } else if cur_backend_id != -1 {
                sched_set_if_supported(sched, ctx, node, cur_backend_id as usize, &mut node_backend_id);
                sched.hv_tensor_backend_ids.insert(node, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        let mut cur_backend_id: i32 = -1;
        for i in (0..graph.nodes.len()).rev() {
            let node = graph.nodes[i];
            if is_view_op(ctx.op(node)) {
                continue;
            }
            let mut node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
            if node_backend_id != -1 {
                cur_backend_id = node_backend_id;
            } else if cur_backend_id != -1 {
                sched_set_if_supported(sched, ctx, node, cur_backend_id as usize, &mut node_backend_id);
                sched.hv_tensor_backend_ids.insert(node, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer
    // types; additionally, set remaining unassigned nodes to the backend with
    // the most supported inputs
    for i in 0..graph.nodes.len() {
        let node = graph.nodes[i];
        if is_view_op(ctx.op(node)) {
            continue;
        }
        let node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
        if node_backend_id == -1 {
            // unassigned node: find the backend with the most supported inputs
            let mut n_supported_best: i32 = -1;
            let mut assign: i32 = -1;
            for b in 0..sched.n_backends {
                if backend_supports_op(&sched.backends[b], ctx, node) {
                    let mut n_supported = 0;
                    for j in 0..MAX_SRC {
                        let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
                        let src_assigned = sched.hv_tensor_backend_ids.get(&src).copied().unwrap_or(-1) != -1
                            || ctx.tensors[src.0 as usize]
                                .view_src
                                .is_some_and(|vs| sched.hv_tensor_backend_ids.get(&vs).copied().unwrap_or(-1) != -1);
                        if src_assigned && sched_buffer_supported(sched, ctx, src, b) {
                            n_supported += 1;
                        }
                    }
                    if n_supported > n_supported_best {
                        n_supported_best = n_supported;
                        assign = b as i32;
                    }
                }
            }
            sched.hv_tensor_backend_ids.insert(node, assign);
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for b in 0..node_backend_id {
                if buft_eq(&sched.bufts[b as usize], &sched.bufts[node_backend_id as usize])
                    && backend_supports_op(&sched.backends[b as usize], ctx, node)
                {
                    let mut supported = true;
                    for j in 0..MAX_SRC {
                        let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
                        if !sched_buffer_supported(sched, ctx, src, b as usize) {
                            supported = false;
                            break;
                        }
                    }
                    if supported {
                        sched.hv_tensor_backend_ids.insert(node, b);
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for i in 0..graph.nodes.len() {
        let node = graph.nodes[i];
        let mut cur_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
        if ctx.tensors[node.0 as usize].view_src.is_some() && cur_backend_id == -1 {
            cur_backend_id = sched
                .hv_tensor_backend_ids
                .get(&ctx.tensors[node.0 as usize].view_src.unwrap())
                .copied()
                .unwrap_or(-1);
            sched.hv_tensor_backend_ids.insert(node, cur_backend_id);
        }
        for j in 0..MAX_SRC {
            let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
            if sched.hv_tensor_backend_ids.get(&src).copied().unwrap_or(-1) == -1 {
                let id = if let Some(vs) = ctx.tensors[src.0 as usize].view_src {
                    // views are always on the same backend as the source
                    sched.hv_tensor_backend_ids.get(&vs).copied().unwrap_or(-1)
                } else {
                    cur_backend_id
                };
                sched.hv_tensor_backend_ids.insert(src, id);
            }
        }
        // if the node is still unassigned, assign it to the first backend
        // that supports it
        let mut b = 0;
        while b < sched.n_backends && cur_backend_id == -1 {
            sched_set_if_supported(sched, ctx, node, b, &mut cur_backend_id);
            b += 1;
        }
        sched.hv_tensor_backend_ids.insert(node, cur_backend_id);
        assert!(cur_backend_id != -1, "node '{}' unassigned", ctx.tensors[node.0 as usize].name);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        // find the backend of the first split, skipping view ops
        let mut i = 0;
        let mut first_backend_id = 0usize;
        while i < graph.nodes.len() {
            let node = graph.nodes[i];
            if !is_view_op(ctx.op(node)) {
                first_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(0).max(0) as usize;
                break;
            }
            i += 1;
        }
        sched.splits.push(SchedSplit {
            backend_id: first_backend_id,
            i_start: 0,
            i_end: graph.nodes.len(),
            inputs: Vec::new(),
            nodes: Vec::new(),
        });
        let mut cur_backend_id = first_backend_id as i32;
        while i < graph.nodes.len() {
            let node = graph.nodes[i];
            if is_view_op(ctx.op(node)) {
                i += 1;
                continue;
            }

            let node_backend_id = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
            assert!(node_backend_id != -1, "all nodes should be assigned by now");

            // check if we should start a new split based on the sources of
            // the current node: a weight on a different and incompatible
            // backend — by starting a new split, the memory of the previously
            // offloaded weights can be reused
            let mut need_new_split = false;
            if node_backend_id == cur_backend_id && !sched.splits.last().unwrap().inputs.is_empty() {
                for j in 0..MAX_SRC {
                    let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };
                    if let Some(src_buf) = tensor_buffer(ctx, src) {
                        if backend_buffer_get_usage(&src_buf) == BackendBufferUsage::Weights {
                            let src_backend_id = sched.hv_tensor_backend_ids.get(&src).copied().unwrap_or(-1);
                            if src_backend_id != cur_backend_id
                                && !sched_buffer_supported(sched, ctx, src, cur_backend_id.max(0) as usize)
                            {
                                need_new_split = true;
                                break;
                            }
                        }
                    }
                }
            }

            if node_backend_id != cur_backend_id || need_new_split {
                sched.splits.last_mut().unwrap().i_end = i;
                sched.splits.push(SchedSplit {
                    backend_id: node_backend_id.max(0) as usize,
                    i_start: i,
                    i_end: graph.nodes.len(),
                    inputs: Vec::new(),
                    nodes: Vec::new(),
                });
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for j in 0..MAX_SRC {
                let Some(src) = ctx.tensors[node.0 as usize].src[j] else { continue };

                let src_backend_id = sched.hv_tensor_backend_ids.get(&src).copied().unwrap_or(-1);
                assert!(src_backend_id != -1, "all inputs should be assigned by now");

                if src_backend_id != cur_backend_id
                    && !sched_buffer_supported(sched, ctx, src, cur_backend_id.max(0) as usize)
                {
                    // create a copy of the input in the split's backend
                    if tensor_copy_of(sched, src, cur_backend_id.max(0) as usize, 0).is_none() {
                        let backend = &sched.backends[cur_backend_id.max(0) as usize];
                        for c in 0..sched.n_copies {
                            let copy = dup_tensor_layout(ctx, src);
                            ctx.tensors[copy.0 as usize].name = format!(
                                "{}#{}#{}",
                                backend_name(Some(backend)),
                                ctx.tensors[src.0 as usize].name,
                                c
                            );
                            if sched.n_copies > 1 {
                                ctx.tensors[copy.0 as usize].flags |=
                                    GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT;
                            }
                            sched.hv_tensor_copies.insert((src, cur_backend_id.max(0) as usize, c), copy);
                        }
                        sched.splits.last_mut().unwrap().inputs.push(src);
                    }
                    let cpy = tensor_copy_of(sched, src, cur_backend_id.max(0) as usize, sched.cur_copy).unwrap();
                    ctx.tensors[node.0 as usize].src[j] = Some(cpy);
                }
            }
            i += 1;
        }
        sched.splits.last_mut().unwrap().i_end = graph.nodes.len();
    }

    if sched.debug != 0 {
        backend_sched_print_assignments(sched, ctx, graph);
    }

    // pass 6 (sync batch D, ggml-backend.cpp:1407-1445): collect ALL input
    // tensors into graph_inputs — including inputs not consumed by any node
    // (e.g. the embeddings input of a text-only batch) — so the graph
    // composition does not depend on which inputs are used (which would
    // otherwise cause graph reallocations when switching between batch
    // types). This replaces the per-node-src copy creation the old pass 5
    // did; the copies' cause moves 4.cpy -> 6.cpy upstream.
    if sched.n_copies > 1 {
        for i in 0..graph.leafs.len() {
            let leaf = graph.leafs[i];
            if ctx.tensors[leaf.0 as usize].flags & GGML_TENSOR_FLAG_INPUT == 0 {
                continue;
            }
            let leaf_backend_id = sched.hv_tensor_backend_ids.get(&leaf).copied().unwrap_or(-1);
            assert!(leaf_backend_id != -1, "all leafs should be assigned by now");

            if tensor_copy_of(sched, leaf, leaf_backend_id.max(0) as usize, 0).is_none() {
                let backend = &sched.backends[leaf_backend_id.max(0) as usize];
                for c in 0..sched.n_copies {
                    let tensor_copy = if c == sched.cur_copy {
                        leaf // use the original tensor as the current copy
                    } else {
                        let copy = dup_tensor_layout(ctx, leaf);
                        ctx.tensors[copy.0 as usize].name = format!(
                            "{}#{}#{}",
                            backend_name(Some(backend)),
                            ctx.tensors[leaf.0 as usize].name,
                            c
                        );
                        // ggml_set_input + ggml_set_output (prevent ggml-alloc
                        // from overwriting the tensor)
                        ctx.tensors[copy.0 as usize].flags |=
                            GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT;
                        copy
                    };
                    sched
                        .hv_tensor_copies
                        .insert((leaf, leaf_backend_id.max(0) as usize, c), tensor_copy);
                }
            }

            sched.graph_inputs.push(leaf);
        }
    }

    // swap node_backend_ids and leaf_backend_ids with prevs
    std::mem::swap(&mut sched.node_backend_ids, &mut sched.prev_node_backend_ids);
    std::mem::swap(&mut sched.leaf_backend_ids, &mut sched.prev_leaf_backend_ids);

    // optimize the split graphs and collect the allocation dependencies added
    // by the backends — this needs to happen before we make graph_copy, so
    // they are in sync (ggml-backend.cpp:1442-1463)
    let alloc_deps: Arc<Mutex<HashMap<TensorId, Vec<TensorId>>>> = Arc::new(Mutex::new(HashMap::new()));
    {
        let deps_cell = alloc_deps.clone();
        let mut opt_params = GraphOptimizeParams {
            add_alloc_dep: Some(Box::new(move |_ctx: &mut Context, tensor: TensorId, until: TensorId| {
                let mut deps = deps_cell.lock().unwrap();
                let keep = deps.entry(until).or_default();
                if !keep.contains(&tensor) {
                    keep.push(tensor);
                }
            })),
        };
        for s in 0..sched.splits.len() {
            let mut gv = Graph::new(0);
            gv.nodes = graph.nodes[sched.splits[s].i_start..sched.splits[s].i_end].to_vec();
            backend_graph_optimize(&sched.backends[sched.splits[s].backend_id], ctx, &mut gv, &mut opt_params);
            // the C graph_optimize sorts the view in place — write the
            // (possibly reordered) nodes back through the shared array
            let (a, b) = (sched.splits[s].i_start, sched.splits[s].i_end);
            graph.nodes[a..b].copy_from_slice(&gv.nodes);
        }
    }
    let alloc_deps = Arc::try_unwrap(alloc_deps).unwrap_or_else(|_| unreachable!()).into_inner().unwrap();

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept
    // tensors as srcs (ggml-backend.cpp:1465-1469)
    let n_dep_nodes: usize = alloc_deps.values().map(|k| (k.len() + MAX_SRC - 1) / MAX_SRC).sum();

    // build the graph copy (ggml-backend.cpp:1471-1589)
    sched.graph_copy_nodes.clear();
    sched.graph_copy_leafs.clear();
    sched.node_backend_ids.clear();
    sched.leaf_backend_ids.clear();
    let mut n_dep_nodes_added = 0usize;

    for split in 0..sched.splits.len() {
        // add inputs to the graph copy so that they are allocated by
        // ggml-alloc at the start of the split
        for j in 0..sched.splits[split].inputs.len() {
            let input = sched.splits[split].inputs[j];
            let input_cpy =
                tensor_copy_of(sched, input, sched.splits[split].backend_id, sched.cur_copy)
                    .expect("split input copy missing");

            // add a dependency to the input source so that it is not freed
            // before the copy is done — ggml_view_tensor (op NONE view)
            let input_dep = view_tensor_node(ctx, input);
            ctx.tensors[input_dep.0 as usize].src[0] = Some(input);
            sched
                .node_backend_ids
                .push(sched.hv_tensor_backend_ids.get(&input).copied().unwrap_or(-1));
            sched.graph_copy_nodes.push(input_dep);

            // add a dependency to the input copy so that it is allocated at
            // the start of the split
            sched.node_backend_ids.push(sched.splits[split].backend_id as i32);
            sched.graph_copy_nodes.push(input_cpy);
        }

        for j in sched.splits[split].i_start..sched.splits[split].i_end {
            let node = graph.nodes[j];
            sched
                .node_backend_ids
                .push(sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1));
            sched.graph_copy_nodes.push(node);

            // add a dependency node so that the kept tensors are not freed
            // before this node is computed (ggml-backend.cpp:1526-1540)
            if let Some(keep) = alloc_deps.get(&node) {
                for chunk in keep.chunks(MAX_SRC) {
                    let dep = view_tensor_node(ctx, chunk[0]);
                    for (s, t) in chunk.iter().enumerate() {
                        ctx.tensors[dep.0 as usize].src[s] = Some(*t);
                    }
                    sched.node_backend_ids.push(sched.splits[split].backend_id as i32);
                    sched.graph_copy_nodes.push(dep);
                    n_dep_nodes_added += 1;
                }
            }
        }

        // snapshot the split's node slice (the C ggml_graph_view)
        sched.splits[split].nodes = graph.nodes[sched.splits[split].i_start..sched.splits[split].i_end].to_vec();
    }

    // a mismatch means a backend added a dep with an `until` tensor that is
    // not a node of the optimized graph (ggml-backend.cpp:1544-1545)
    assert_eq!(n_dep_nodes_added, n_dep_nodes);

    if sched.n_copies > 1 {
        // add input copies as leafs so that they are allocated first
        for &input in sched.graph_inputs.clone().iter() {
            let backend_id = sched.hv_tensor_backend_ids.get(&input).copied().unwrap_or(-1);
            for c in 0..sched.n_copies {
                let input_cpy = tensor_copy_of(sched, input, backend_id.max(0) as usize, c).unwrap();
                sched.leaf_backend_ids.push(backend_id);
                sched.graph_copy_leafs.push(input_cpy);
            }
        }
        for split in 0..sched.splits.len() {
            let backend_id = sched.splits[split].backend_id;
            for j in 0..sched.splits[split].inputs.len() {
                let input = sched.splits[split].inputs[j];
                for c in 0..sched.n_copies {
                    let input_cpy = tensor_copy_of(sched, input, backend_id, c).unwrap();
                    sched.leaf_backend_ids.push(backend_id as i32);
                    sched.graph_copy_leafs.push(input_cpy);
                }
            }
        }
    }

    // add leafs from the original graph
    for &leaf in graph.leafs.iter() {
        sched
            .leaf_backend_ids
            .push(sched.hv_tensor_backend_ids.get(&leaf).copied().unwrap_or(-1));
        sched.graph_copy_leafs.push(leaf);
    }

    // close the sched tensor session range
    if let Some((start, _)) = sched.sched_tensor_range {
        sched.sched_tensor_range = Some((start, ctx.tensors.len()));
    }
}

/// `tensor_copy` (ggml-backend.cpp:846)
fn tensor_copy_of(sched: &BackendSched, t: TensorId, backend_id: usize, copy_id: usize) -> Option<TensorId> {
    sched.hv_tensor_copies.get(&(t, backend_id, copy_id)).copied()
}

/// `ggml_dup_tensor_layout` (ggml-backend.cpp:748-755) — a copy with the same
/// memory layout (type + ne + nb), fresh storage.
fn dup_tensor_layout(ctx: &mut Context, tensor: TensorId) -> TensorId {
    let (ty, ne, nb) = {
        let t = &ctx.tensors[tensor.0 as usize];
        (t.ty, t.ne, t.nb)
    };
    let id = ctx.new_tensor(ty, ne);
    ctx.tensors[id.0 as usize].nb = nb;
    id
}

/// `ggml_view_tensor` (ggml.c:1963) — an op-NONE alias with identical ne/nb
/// (`input_dep` / dep nodes in graph_copy).
fn view_tensor_node(ctx: &mut Context, src: TensorId) -> TensorId {
    let (ty, ne, nb, name) = {
        let t = &ctx.tensors[src.0 as usize];
        (t.ty, t.ne, t.nb, t.name.clone())
    };
    let id = ctx.new_tensor(ty, ne);
    let t = &mut ctx.tensors[id.0 as usize];
    t.nb = nb;
    t.view_src = Some(src);
    t.view_offs = 0;
    t.name = format!("{name} (view)");
    id
}

/// `fmt_size` (ggml-backend.cpp:987)
fn fmt_size(size: usize) -> String {
    if size >= 1024 * 1024 {
        format!("{}M", size / 1024 / 1024)
    } else {
        format!("{}K", size / 1024)
    }
}

/// `ggml_backend_sched_print_assignments` (ggml-backend.cpp:997-1035),
/// enabled by `GGML_SCHED_DEBUG` (use_counts / causes are not tracked).
fn backend_sched_print_assignments(sched: &BackendSched, ctx: &Context, graph: &Graph) {
    let mut cur_split = 0usize;
    for i in 0..graph.nodes.len() {
        if cur_split < sched.splits.len() && i == sched.splits[cur_split].i_start {
            let split = &sched.splits[cur_split];
            eprintln!(
                "\n## SPLIT #{}: {} # {} inputs:",
                cur_split,
                backend_name(Some(&sched.backends[split.backend_id])),
                split.inputs.len()
            );
            for j in 0..split.inputs.len() {
                eprint!(
                    "[{} ({})] ",
                    ctx.tensors[split.inputs[j].0 as usize].name,
                    fmt_size(ctx.nbytes(split.inputs[j]))
                );
            }
            eprintln!();
            cur_split += 1;
        }
        let node = graph.nodes[i];
        if is_view_op(ctx.op(node)) {
            continue;
        }
        if sched.debug > 1 {
            eprintln!(
                "node #{:3} ({:?}): {:20} ({}) [backend {}]",
                i,
                ctx.op(node),
                ctx.tensors[node.0 as usize].name,
                fmt_size(ctx.nbytes(node)),
                sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1)
            );
        }
    }
}

/// `ggml_backend_sched_alloc_splits` (ggml-backend.cpp:1591-1644)
fn backend_sched_alloc_splits(sched: &mut BackendSched, ctx: &mut Context) -> bool {
    // buft inequality is what actually forces a re-allocation
    // (ggml-backend.cpp:1594-1595) — the prev arrays start zeroed like the
    // C calloc
    let buft_changed = |cur: i32, prev: i32| -> bool {
        cur != prev && !buft_eq(&sched.bufts[cur as usize], &sched.bufts[prev.max(0) as usize])
    };
    let mut backend_ids_changed = false;
    for i in 0..sched.graph_copy_nodes.len() {
        if buft_changed(sched.node_backend_ids[i], sched.prev_node_backend_ids.get(i).copied().unwrap_or(0)) {
            backend_ids_changed = true;
            break;
        }
    }
    if !backend_ids_changed {
        for i in 0..sched.graph_copy_leafs.len() {
            if buft_changed(sched.leaf_backend_ids[i], sched.prev_leaf_backend_ids.get(i).copied().unwrap_or(0)) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if backend_ids_changed
        || !sched.galloc.alloc_graph(
            ctx,
            &sched.graph_copy_nodes,
            &sched.graph_copy_leafs,
            &sched.node_backend_ids,
            &sched.leaf_backend_ids,
        )
    {
        // the re-allocation may cause the split inputs to be moved to a
        // different address — synchronize without
        // ggml_backend_sched_synchronize to avoid changing cur_copy
        for b in &sched.backends {
            backend_synchronize(b);
        }

        if !sched.galloc.reserve_n(
            ctx,
            &sched.graph_copy_nodes,
            &sched.graph_copy_leafs,
            &sched.node_backend_ids,
            &sched.leaf_backend_ids,
        ) {
            eprintln!("ggml_backend_sched_alloc_splits: failed to reserve graph buffers");
            return false;
        }
        if !sched.galloc.alloc_graph(
            ctx,
            &sched.graph_copy_nodes,
            &sched.graph_copy_leafs,
            &sched.node_backend_ids,
            &sched.leaf_backend_ids,
        ) {
            eprintln!("ggml_backend_sched_alloc_splits: failed to allocate graph");
            return false;
        }
    }

    true
}

/// `ggml_backend_sched_is_host_weight` (ggml-backend.cpp:1805-1809 @c35b66744,
/// added by 6753a033f) — a tensor living in a host buffer with USAGE_WEIGHTS.
fn sched_is_host_weight(ctx: &Context, t: TensorId) -> bool {
    tensor_buffer(ctx, t).is_some_and(|b| {
        backend_buffer_get_usage(&b) == BackendBufferUsage::Weights && backend_buffer_is_host(&b)
    })
}

/// `ggml_backend_sched_copy_input` (ggml-backend.cpp:1811-1842 @c35b66744,
/// extracted from the compute_splits loop by 6753a033f): copy one split input
/// to the split's backend — user inputs immediately (event-sync then copy),
/// everything else after waiting on the split backend's event; a host-weight
/// input is first offered to the copy callback (a `true` return means the
/// user already did the copy); the async-copy attempt falls back to a sync
/// copy when the backend has no `cpy_tensor_async` or it declines.
fn sched_copy_input(sched: &BackendSched, ctx: &mut Context, split_id: usize, input: TensorId) {
    let split_backend_id = sched.splits[split_id].backend_id;
    let split_backend = sched.backends[split_backend_id].clone();
    let input_backend = backend_sched_get_tensor_backend(sched, input)
        .expect("split input must have an assigned backend");
    let input_cpy = tensor_copy_of(sched, input, split_backend_id, sched.cur_copy).unwrap();

    if ctx.tensors[input.0 as usize].flags & GGML_TENSOR_FLAG_INPUT != 0 {
        // inputs from the user must be copied immediately to prevent the user
        // overwriting the data before the copy is done
        if let Some(event) = &sched.events[split_backend_id][sched.cur_copy] {
            backend_event_synchronize(event);
        } else {
            backend_synchronize(&split_backend);
        }
        backend_tensor_copy(ctx, input, input_cpy);
        return;
    }

    // wait for the split backend to finish using the input before
    // overwriting it
    if let Some(event) = &sched.events[split_backend_id][sched.cur_copy] {
        backend_event_wait(&split_backend, event);
    } else {
        backend_synchronize(&split_backend);
    }

    // offer host weights to the copy callback — `true` means the user
    // performed the (possibly partial) copy
    // (ggml-backend.cpp:1835-1838)
    if let Some(callback) = &sched.callback_copy {
        if sched_is_host_weight(ctx, input)
            && callback(
                &split_backend,
                ctx,
                input,
                input_cpy,
                &sched.splits[split_id].nodes,
            )
        {
            return;
        }
    }

    // try async copy, but if not possible, we can still use a sync copy
    // without synchronizing the dst backend, since we handle the
    // synchronization here with multiple copies and events
    // (ggml-backend.cpp:1838-1842)
    let mut done = false;
    if let Some(cpy) = split_backend.iface.cpy_tensor_async {
        done = cpy(&input_backend, &split_backend, ctx, input, input_cpy);
    }
    if !done {
        backend_synchronize(&input_backend);
        if let Some(event) = &sched.events[split_backend_id][sched.cur_copy] {
            backend_event_synchronize(event);
        } else {
            backend_synchronize(&split_backend);
        }
        backend_tensor_copy(ctx, input, input_cpy);
    }
}

/// `ggml_backend_sched_compute_splits` (ggml-backend.cpp:1848-1949) — the
/// run pipeline: per split, copy the inputs, compute, record the event.
fn backend_sched_compute_splits(sched: &BackendSched, ctx: &mut Context) -> GgmlStatus {
    let mut prev_backend_id: i32 = -1;

    for split_id in 0..sched.splits.len() {
        let split_backend_id = sched.splits[split_id].backend_id;
        let split_backend = sched.backends[split_backend_id].clone();

        // ensure the previous split's async work has completed before we
        // start this split — the allocator may have reused buffer regions
        // across splits
        if sched.splits[split_id].inputs.is_empty()
            && prev_backend_id >= 0
            && prev_backend_id != split_backend_id as i32
        {
            if let Some(event) = &sched.events[prev_backend_id as usize][sched.cur_copy] {
                backend_event_synchronize(event);
            } else {
                backend_synchronize(&sched.backends[prev_backend_id as usize]);
            }
        }

        // copy the input tensors to the split backend — the weights in host
        // memory are copied last, so that the copy callback can read the
        // other inputs of the split (ggml-backend.cpp:1871-1885 @c35b66744;
        // the two-pass non-weights-then-host-weights order was introduced by
        // 6753a033f together with the removal of the scheduler-side
        // selective-expert-copy special case below)
        for input_id in 0..sched.splits[split_id].inputs.len() {
            let input = sched.splits[split_id].inputs[input_id];
            if !sched_is_host_weight(ctx, input) {
                sched_copy_input(sched, ctx, split_id, input);
            }
        }
        for input_id in 0..sched.splits[split_id].inputs.len() {
            let input = sched.splits[split_id].inputs[input_id];
            if sched_is_host_weight(ctx, input) {
                sched_copy_input(sched, ctx, split_id, input);
            }
        }

        if sched.callback_eval.is_none() {
            let mut split_graph = Graph::new(0);
            split_graph.nodes = sched.splits[split_id].nodes.clone();
            let ec = backend_graph_compute_async(&split_backend, ctx, &mut split_graph);
            if ec != GgmlStatus::Success {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            // (ggml-backend.cpp:1804-1834): batch nodes the user does not
            // want to observe into single compute calls
            let callback = sched.callback_eval.as_ref().unwrap();
            let nodes = sched.splits[split_id].nodes.clone();
            let mut j0 = 0usize;
            while j0 < nodes.len() {
                let mut t = nodes[j0];
                let mut need = callback(ctx, t, true);

                let mut j1 = j0;
                while !need && j1 < nodes.len() - 1 {
                    j1 += 1;
                    t = nodes[j1];
                    need = callback(ctx, t, true);
                }

                let mut gv = Graph::new(0);
                gv.nodes = nodes[j0..j1 + 1].to_vec();
                let ec = backend_graph_compute_async(&split_backend, ctx, &mut gv);
                if ec != GgmlStatus::Success {
                    return ec;
                }
                backend_synchronize(&split_backend);

                if need && !callback(ctx, t, false) {
                    break;
                }
                j0 = j1 + 1;
            }
        }

        // record the event of this split
        if let Some(event) = &sched.events[split_backend_id][sched.cur_copy] {
            backend_event_record(event, &split_backend);
        }

        prev_backend_id = split_backend_id as i32;
    }

    GgmlStatus::Success
}

// [GGML_SCHED_MOE] removed by upstream 6753a033f ("refactor selective expert
// copying to user code"): the scheduler-side special case that copied only
// the experts a split's MUL_MAT_ID actually used (the old
// ggml-backend.cpp:1693-1781 `moe_copy_used_experts` + its `is_moe_weight`
// gate in compute_splits) was deleted from the C source and is deleted here
// with it. Selective expert copying is now the application's job, done
// through `ggml_backend_sched_set_copy_callback` — the two-pass input copy
// (non-host-weights first, then host weights) guarantees the callback sees
// all non-weight inputs of the split (e.g. the expert ids) already copied.

impl BackendSched {
    /// `ggml_backend_sched_new` (ggml-backend.cpp:1848-1918) — backends with
    /// low index are given priority over backends with high index.
    fn new_inner(
        backends: &[BackendRef],
        bufts: Option<&[BackendBufferTypeRef]>,
        graph_size: usize,
        parallel: bool,
        op_offload: bool,
    ) -> Self {
        assert!(!backends.is_empty());
        assert!(backends.len() <= GGML_SCHED_MAX_BACKENDS);
        // the last backend must be the CPU backend (ggml-backend.cpp:1857)
        assert_eq!(
            backend_dev_type(&backend_get_device(backends.last().unwrap())),
            BackendDevType::Cpu
        );

        let debug = std::env::var("GGML_SCHED_DEBUG").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

        let n_backends = backends.len();
        let n_copies = if parallel { GGML_SCHED_MAX_COPIES } else { 1 };

        let bufts: Vec<BackendBufferTypeRef> = match bufts {
            Some(b) => b.to_vec(),
            None => backends.iter().map(backend_get_default_buffer_type).collect(),
        };
        for b in 0..n_backends {
            assert!(backend_supports_buft(&backends[b], &bufts[b]));
        }

        // pipeline-parallelism events (the CPU device provides none — the
        // slots stay None and the sync paths take over)
        let mut events = vec![vec![None; GGML_SCHED_MAX_COPIES]; GGML_SCHED_MAX_BACKENDS];
        if n_copies > 1 {
            for b in 0..n_backends {
                for c in 0..n_copies {
                    events[b][c] = backend_event_new(Some(&backend_get_device(&backends[b])));
                }
            }
        }

        let mut sched = BackendSched {
            is_reset: false,
            is_alloc: false,
            n_backends,
            backends: backends.to_vec(),
            bufts: bufts.clone(),
            // ggml_gallocr_new_n (ggml-backend.cpp:1912)
            galloc: Gallocr::new(&bufts),
            hv_tensor_backend_ids: HashMap::with_capacity(graph_size),
            hv_tensor_copies: HashMap::new(),
            node_backend_ids: Vec::new(),
            leaf_backend_ids: Vec::new(),
            prev_node_backend_ids: Vec::new(),
            prev_leaf_backend_ids: Vec::new(),
            graph_copy_nodes: Vec::new(),
            graph_copy_leafs: Vec::new(),
            splits: Vec::new(),
            n_copies,
            cur_copy: 0,
            next_copy: 0,
            events,
            graph_inputs: Vec::new(),
            callback_eval: None,
            callback_copy: None,
            op_offload,
            debug,
            sched_tensor_range: None,
        };
        backend_sched_reset(&mut sched);
        sched
    }
}

/// `ggml_backend_sched_new` (ggml-backend.cpp:1848)
pub fn backend_sched_new(
    backends: &[BackendRef],
    bufts: Option<&[BackendBufferTypeRef]>,
    graph_size: usize,
    parallel: bool,
    op_offload: bool,
) -> BackendSched {
    BackendSched::new_inner(backends, bufts, graph_size, parallel, op_offload)
}

/// `ggml_backend_sched_free` (ggml-backend.cpp:1920-1947) — Drop in the port.
impl Drop for BackendSched {
    fn drop(&mut self) {
        for b in self.events.iter().flatten() {
            backend_event_free(b.clone());
        }
    }
}

/// `ggml_backend_sched_reset` (ggml-backend.cpp:1949-1959)
pub fn backend_sched_reset(sched: &mut BackendSched) {
    // reset state for the next run
    if !sched.is_reset {
        sched.hv_tensor_backend_ids.clear();
        sched.hv_tensor_copies.clear();
        sched.is_reset = true;
    }
    sched.is_alloc = false;
}

/// `ggml_backend_sched_reserve_size` (ggml-backend.cpp:1961-1973). (The C
/// `hash_set.size >= n_nodes + n_leafs` assert is moot — the port's hash map
/// grows dynamically.)
pub fn backend_sched_reserve_size(sched: &mut BackendSched, ctx: &mut Context, measure_graph: &mut Graph, sizes: &mut [usize]) {
    backend_sched_reset(sched);
    backend_sched_synchronize(sched);
    backend_sched_split_graph(sched, ctx, measure_graph);

    sched.galloc.reserve_n_size(
        ctx,
        &sched.graph_copy_nodes,
        &sched.graph_copy_leafs,
        &sched.node_backend_ids,
        &sched.leaf_backend_ids,
        sizes,
    );
}

/// `ggml_backend_sched_reserve` (ggml-backend.cpp:1975-1990)
pub fn backend_sched_reserve(sched: &mut BackendSched, ctx: &mut Context, measure_graph: &mut Graph) -> bool {
    backend_sched_synchronize(sched);
    backend_sched_split_graph(sched, ctx, measure_graph);

    if !sched.galloc.reserve_n(
        ctx,
        &sched.graph_copy_nodes,
        &sched.graph_copy_leafs,
        &sched.node_backend_ids,
        &sched.leaf_backend_ids,
    ) {
        return false;
    }

    backend_sched_reset(sched);
    true
}

/// `ggml_backend_sched_alloc_graph` (ggml-backend.cpp:1992-2009)
pub fn backend_sched_alloc_graph(sched: &mut BackendSched, ctx: &mut Context, graph: &mut Graph) -> bool {
    assert!(!sched.is_alloc);

    sched.cur_copy = sched.next_copy;
    sched.next_copy = (sched.next_copy + 1) % sched.n_copies;

    backend_sched_split_graph(sched, ctx, graph);

    if !backend_sched_alloc_splits(sched, ctx) {
        return false;
    }

    sched.is_alloc = true;
    true
}

/// `ggml_backend_sched_graph_compute` (ggml-backend.cpp:2011-2015)
pub fn backend_sched_graph_compute(sched: &mut BackendSched, ctx: &mut Context, graph: &mut Graph) -> GgmlStatus {
    let err = backend_sched_graph_compute_async(sched, ctx, graph);
    backend_sched_synchronize(sched);
    err
}

/// `ggml_backend_sched_graph_compute_async` (ggml-backend.cpp:2017-2030)
pub fn backend_sched_graph_compute_async(sched: &mut BackendSched, ctx: &mut Context, graph: &mut Graph) -> GgmlStatus {
    if !sched.is_reset && !sched.is_alloc {
        backend_sched_reset(sched);
    }

    if !sched.is_alloc && !backend_sched_alloc_graph(sched, ctx, graph) {
        return GgmlStatus::AllocFailed;
    }

    backend_sched_compute_splits(sched, ctx)
}

/// `ggml_backend_sched_synchronize` (ggml-backend.cpp:2032-2043)
pub fn backend_sched_synchronize(sched: &mut BackendSched) {
    for b in &sched.backends {
        backend_synchronize(b);
    }
    if !sched.is_alloc {
        // if the graph is not already allocated, always use copy 0 after a
        // synchronization — this ensures that during generation the same copy
        // is used every time
        sched.next_copy = 0;
    }
}

/// `ggml_backend_sched_set_eval_callback` (ggml-backend.cpp:2045-2049)
pub fn backend_sched_set_eval_callback(sched: &mut BackendSched, callback: SchedEvalCallback) {
    sched.callback_eval = Some(callback);
}

/// `ggml_backend_sched_set_copy_callback` (ggml-backend.cpp:2140-2143
/// @c35b66744, added by 6753a033f) — set a callback to be called when the
/// input weights of a split are being copied. (The C `user_data` is carried
/// by the closure.)
pub fn backend_sched_set_copy_callback(sched: &mut BackendSched, callback: SchedCopyCallback) {
    sched.callback_copy = Some(callback);
}

/// `ggml_backend_sched_get_n_splits` (ggml-backend.cpp:2051)
pub fn backend_sched_get_n_splits(sched: &BackendSched) -> usize {
    sched.splits.len()
}

/// `ggml_backend_sched_get_n_copies` (ggml-backend.cpp:2056)
pub fn backend_sched_get_n_copies(sched: &BackendSched) -> usize {
    sched.n_copies
}

/// `ggml_backend_sched_get_n_backends` (ggml-backend.cpp:2061)
pub fn backend_sched_get_n_backends(sched: &BackendSched) -> usize {
    sched.n_backends
}

/// `ggml_backend_sched_get_backend` (ggml-backend.cpp:2066)
pub fn backend_sched_get_backend(sched: &BackendSched, i: usize) -> BackendRef {
    assert!(i < sched.n_backends);
    sched.backends[i].clone()
}

/// `ggml_backend_sched_get_buffer_type` (ggml-backend.cpp:2072)
pub fn backend_sched_get_buffer_type(sched: &BackendSched, backend: &BackendRef) -> BackendBufferTypeRef {
    let idx = sched_backend_id(sched, backend);
    assert!((idx as usize) < sched.n_backends);
    sched.bufts[idx as usize].clone()
}

/// `ggml_backend_sched_get_buffer_size` (ggml-backend.cpp:2080)
pub fn backend_sched_get_buffer_size(sched: &BackendSched, backend: &BackendRef) -> usize {
    let idx = sched_backend_id(sched, backend);
    assert!((idx as usize) < sched.n_backends);
    sched.galloc.get_buffer_size(idx as usize)
}

/// `ggml_backend_sched_set_tensor_backend` (ggml-backend.cpp:2088)
pub fn backend_sched_set_tensor_backend(sched: &mut BackendSched, node: TensorId, backend: &BackendRef) {
    let backend_index = sched_backend_id(sched, backend);
    assert!(backend_index >= 0 && (backend_index as usize) < sched.n_backends);
    sched.hv_tensor_backend_ids.insert(node, backend_index);
    sched.is_reset = false;
}

/// `ggml_backend_sched_get_tensor_backend` (ggml-backend.cpp:2097)
pub fn backend_sched_get_tensor_backend(sched: &BackendSched, node: TensorId) -> Option<BackendRef> {
    let backend_index = sched.hv_tensor_backend_ids.get(&node).copied().unwrap_or(-1);
    if backend_index == -1 {
        return None;
    }
    Some(sched.backends[backend_index as usize].clone())
}

// ===========================================================================
// mock backend for the two-backend scheduler test
// ===========================================================================

/// A second backend that wraps the same CPU engine under a different name
/// but rejects one op type — built exactly like the reference's own
/// `dummy_backend` (tests/test-alloc.cpp:15-133) plus the op rejection the
/// scheduler split test needs. Its buffer type is a *distinct* arena-backed
/// type that reports `is_host == false` (like a real GPU's device memory),
/// so tensors crossing backends must be copied by the scheduler.
#[cfg(test)]
pub(crate) mod mock {
    use super::*;
    use crate::backend::MockBackendCtx;

    /// the mock's static parameters (fn-pointer ifaces cannot capture) —
    /// carried by the leaked instance for `mock_backend` and by the device
    /// context for the iface fns
    pub(crate) struct MockDeviceParams {
        pub name: &'static str,
        pub reject: GgmlOp,
        pub buft: BackendBufferTypeRef,
    }

    pub struct Mock {
        pub backend: BackendRef,
        pub buft: BackendBufferTypeRef,
    }

    /// one mock per (name, reject) — the buffer type has stable identity
    pub fn mock_backend(name: &'static str, reject_op: Option<GgmlOp>) -> Mock {
        let params: &'static MockDeviceParams = Box::leak(Box::new(MockDeviceParams {
            name,
            reject: reject_op.unwrap_or(GgmlOp::None),
            buft: mock_buft(name),
        }));

        Mock {
            backend: mock_backend_stream(params),
            buft: params.buft.clone(),
        }
    }

    fn mock_backend_stream(params: &MockDeviceParams) -> BackendRef {
        fn mock_get_name(b: &Backend) -> String {
            match &b.context {
                BackendCtx::Mock(c) => c.name.to_string(),
                _ => unreachable!(),
            }
        }
        // wraps the same CPU engine under a different name (ggml-cpu.cpp:175)
        fn mock_graph_compute(b: &BackendRef, ctx: &mut Context, g: &mut Graph) -> GgmlStatus {
            let n_threads = match &b.context {
                BackendCtx::Mock(_) => 2,
                _ => unreachable!(),
            };
            crate::compute::graph_compute(ctx, g, n_threads);
            GgmlStatus::Success
        }
        Arc::new(Backend {
            guid: [
                0x6d, 0x6f, 0x63, 0x6b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01,
            ],
            iface: BackendI {
                get_name: mock_get_name,
                free: None,
                set_tensor_async: None,
                get_tensor_async: None,
                cpy_tensor_async: None,
                synchronize: None,
                graph_compute: mock_graph_compute,
                event_record: None,
                event_wait: None,
                graph_optimize: None,
            },
            device: mock_device_of(params),
            context: BackendCtx::Mock(MockBackendCtx {
                name: params.name,
                reject_op: Some(params.reject),
            }),
        })
    }

    /// one stable device per mock name (the C static ggml_backend_cpu_device);
    /// all iface fns are non-captaining and resolve the payload from context
    fn mock_device_of(params: &MockDeviceParams) -> BackendDeviceRef {
        static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, BackendDeviceRef>>> = std::sync::OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut g = cache.lock().unwrap();
        let key = params.name.as_ptr() as usize;
        if let Some(d) = g.get(&key) {
            return d.clone();
        }
        let device: BackendDeviceRef = Arc::new(BackendDevice {
            iface: BackendDeviceI {
                get_name: mock_dev_name,
                get_description: mock_dev_description,
                get_memory: mock_dev_memory,
                get_type: mock_dev_type,
                get_props: mock_dev_props,
                init_backend: mock_dev_init_backend,
                get_buffer_type: mock_dev_buffer_type,
                get_host_buffer_type: None,
                buffer_from_host_ptr: None,
                supports_op: mock_dev_supports_op,
                supports_buft: mock_dev_supports_buft,
                offload_op: None,
                event_new: None,
                event_free: None,
                event_synchronize: None,
            },
            reg: Mutex::new(None),
            context: BackendDeviceCtx::Mock(Arc::new(MockDeviceParams {
                name: params.name,
                reject: params.reject,
                buft: params.buft.clone(),
            })),
        });
        g.insert(key, device.clone());
        device
    }

    fn mock_dev_name(dev: &BackendDevice) -> String {
        mock_params_of_dev(dev).name.to_string()
    }
    fn mock_dev_description(dev: &BackendDevice) -> String {
        format!("{} mock device", mock_params_of_dev(dev).name)
    }
    fn mock_dev_memory(_dev: &BackendDevice) -> (usize, usize) {
        (usize::MAX / 2, usize::MAX / 2)
    }
    fn mock_dev_type(_dev: &BackendDevice) -> BackendDevType {
        BackendDevType::Gpu
    }
    fn mock_dev_props(dev: &BackendDevice, props: &mut crate::backend::BackendDevProps) {
        props.name = (dev.iface.get_name)(dev);
        props.description = (dev.iface.get_description)(dev);
        props.ty = BackendDevType::Gpu;
        (props.memory_free, props.memory_total) = (dev.iface.get_memory)(dev);
    }
    fn mock_dev_init_backend(dev: &BackendDeviceRef, _params: Option<&str>) -> Option<BackendRef> {
        Some(mock_backend_stream(mock_params_of(dev)))
    }
    fn mock_dev_buffer_type(dev: &BackendDeviceRef) -> BackendBufferTypeRef {
        mock_params_of(dev).buft.clone()
    }
    fn mock_dev_supports_op(dev: &BackendDeviceRef, ctx: &Context, op: TensorId) -> bool {
        ctx.op(op) != mock_params_of(dev).reject
    }
    fn mock_dev_supports_buft(dev: &BackendDeviceRef, buft: &BackendBufferTypeRef) -> bool {
        buft_eq(buft, &mock_params_of(dev).buft)
    }

    fn mock_params_of_dev(dev: &BackendDevice) -> &MockDeviceParams {
        match &dev.context {
            BackendDeviceCtx::Mock(p) => p.downcast_ref::<MockDeviceParams>().unwrap(),
            _ => unreachable!(),
        }
    }

    fn mock_params_of(dev: &BackendDeviceRef) -> &MockDeviceParams {
        mock_params_of_dev(dev)
    }

    fn mock_buft_name(buft: &BackendBufferType) -> String {
        match buft.context.downcast_ref::<String>() {
            Some(n) => format!("{n}_buffer_type"),
            None => String::new(),
        }
    }

    /// stable buffer type per mock name (like the C function-local statics)
    fn mock_buft(name: &'static str) -> BackendBufferTypeRef {
        static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, BackendBufferTypeRef>>> = std::sync::OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut g = cache.lock().unwrap();
        let key = name.as_ptr() as usize;
        if let Some(b) = g.get(&key) {
            return b.clone();
        }
        let buft: BackendBufferTypeRef = Arc::new(BackendBufferType {
            iface: BackendBufferTypeI {
                get_name: mock_buft_name,
                // host-arena backed: the CPU engine can compute on it
                alloc_buffer: crate::backend::cpu_buffer_type_alloc_buffer,
                get_alignment: |_| TENSOR_ALIGNMENT,
                get_max_size: None,
                get_alloc_size: None,
                // NOT host: forces the scheduler to insert copies
                is_host: Some(|_| false),
            },
            device: None,
            context: Arc::new(name.to_string()),
        });
        g.insert(key, buft.clone());
        buft
    }
}

// ===========================================================================
// tests — the scheduler verification required by the GPU enablement plan:
//  1. sched-on-CPU vs direct `graph_compute` bit identity
//  2. the two-backend split test with hand-derived expectations (mirroring
//     what `ggml_backend_sched_split_graph`'s algorithm produces)
//  3. the gallocr inplace reuse (transcription of the reference's own
//     `graph_reuses_allocation`, tests/test-alloc.cpp:628-638)
//  4. repeated computes through one scheduler (graph rebuild flow)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GgmlType;

    fn fill_f32(ctx: &mut Context, t: TensorId, f: impl Fn(usize) -> f32) {
        let n = ctx.nbytes(t) / 4;
        let vals: Vec<f32> = (0..n).map(f).collect();
        ctx.with_f32_mut(t, |p| p.copy_from_slice(&vals)).unwrap();
    }

    /// toy graph: mul_mat + add(bias) + silu + mul chain on F32
    fn build_toy(ctx: &mut Context) -> TensorId {
        let w = ctx.new_tensor_2d(GgmlType::F32, 8, 4); // [8,4]
        ctx.arena_resize_tensor(w);
        fill_f32(ctx, w, |i| (i as f32 * 0.25 - 1.0).sin());
        ctx.set_name(w, "w");

        let x = ctx.new_tensor_2d(GgmlType::F32, 8, 2); // [8,2]
        ctx.arena_resize_tensor(x);
        fill_f32(ctx, x, |i| (i as f32 * 0.5 - 3.0).cos());
        ctx.set_name(x, "x");
        ctx.tensors[x.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;

        let bias = ctx.new_tensor_2d(GgmlType::F32, 2, 1);
        ctx.arena_resize_tensor(bias);
        fill_f32(ctx, bias, |i| 0.1 * i as f32);
        ctx.set_name(bias, "bias");
        ctx.tensors[bias.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;

        let mm = ctx.mul_mat(w, x); // [2,4]
        ctx.set_name(mm, "mm");
        let ad = ctx.add(mm, bias); // broadcast over rows
        ctx.set_name(ad, "ad");
        let si = ctx.silu(ad);
        ctx.set_name(si, "si");
        let out = ctx.mul(si, ad);
        ctx.set_name(out, "out");
        ctx.tensors[out.0 as usize].flags |= GGML_TENSOR_FLAG_OUTPUT;
        out
    }

    /// sched-on-CPU vs direct `compute::graph_compute`: outputs bit-identical
    #[test]
    fn sched_cpu_bit_identity() {
        // direct
        let mut ctx_d = Context::new();
        let out_d = build_toy(&mut ctx_d);
        let mut g = Graph::new(16);
        g.build_forward(&ctx_d, out_d);
        crate::compute::graph_compute(&mut ctx_d, &mut g, 4);

        // through the scheduler with the CPU backend adapter
        let mut ctx_s = Context::new();
        let out_s = build_toy(&mut ctx_s);
        let mut g2 = Graph::new(16);
        g2.build_forward(&ctx_s, out_s);
        let cpu = cpu_backend_init();
        let mut sched = backend_sched_new(&[cpu], None, 64, false, true);
        assert_eq!(backend_sched_get_n_backends(&sched), 1);
        let st = backend_sched_graph_compute(&mut sched, &mut ctx_s, &mut g2);
        assert_eq!(st, GgmlStatus::Success);
        assert_eq!(backend_sched_get_n_splits(&sched), 1, "single backend — one split");

        let d = ctx_d.data_bytes(out_d).unwrap();
        let s = ctx_s.data_bytes(out_s).unwrap();
        assert_eq!(d.len(), s.len());
        assert_eq!(d, s, "sched output must be bit-identical to direct compute");
    }

    /// the two-backend split test: a mock backend (same CPU engine, rejects
    /// SILU) at priority 0 with the weights in its buffer, CPU at priority 1.
    ///
    /// Hand-derived from `ggml_backend_sched_split_graph`'s algorithm
    /// (ggml-backend.cpp:1066-1425):
    ///   pass 1: w/w2 pre-allocated in the mock buffer -> mock (1.dst); the
    ///           INPUT x -> last backend = CPU; n0/n2 follow their weights
    ///           (1.wgt0); n1 (silu) unassigned
    ///   pass 2: expand does not cross the unsupported silu
    ///   pass 3: n1 -> CPU (the only backend supporting it)
    ///   pass 5: splits [mock n0], [cpu n1], [mock n2] with cross-backend
    ///           input copies of x, n0 and n1 respectively
    #[test]
    fn sched_two_backend_split() {
        let mut ctx = Context::new();
        let mock = mock::mock_backend("MOCK", Some(GgmlOp::Silu));
        let cpu = cpu_backend_init();

        // weights pre-allocated in the MOCK buffer with USAGE_WEIGHTS
        // (mul_mat output ne = [a.ne[1], b.ne[1]]: n0/n1/n2 are [4,2])
        let w = ctx.new_tensor_2d(GgmlType::F32, 4, 4); // [4,4]
        let w2 = ctx.new_tensor_2d(GgmlType::F32, 4, 4); // [4,4]
        let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2); // [4,2] input
        let buf = backend_buft_alloc_buffer(&mock.buft, &mut ctx, 512).unwrap();
        let base = backend_buffer_get_base(&buf).unwrap();
        assert_eq!(backend_tensor_alloc(&buf, &mut ctx, w, base), GgmlStatus::Success);
        assert_eq!(backend_tensor_alloc(&buf, &mut ctx, w2, base + 128), GgmlStatus::Success);
        backend_buffer_set_usage(&buf, BackendBufferUsage::Weights);
        fill_f32(&mut ctx, w, |i| (i as f32 * 0.125).sin());
        fill_f32(&mut ctx, w2, |i| (i as f32 * 0.3).cos());
        ctx.set_name(w, "w");
        ctx.set_name(w2, "w2");
        // the input is NOT pre-allocated: the scheduler's gallocr allocates
        // it (INPUT leaf) and the value is set through ggml-backend — the
        // ggml-backend.h:296-300 flow (alloc_graph → tensor_set → compute)
        ctx.set_name(a, "a");
        ctx.tensors[a.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;

        let n0 = ctx.mul_mat(w, a); // [4,2] — mock (weights rule)
        ctx.set_name(n0, "n0");
        let n1 = ctx.silu(n0); // [4,2] — mock rejects SILU -> CPU
        ctx.set_name(n1, "n1");
        let n2 = ctx.mul_mat(w2, n1); // [4,2] — mock (weights rule)
        ctx.set_name(n2, "n2");
        ctx.tensors[n2.0 as usize].flags |= GGML_TENSOR_FLAG_OUTPUT;

        // direct reference computation (same engine, 2 threads like the mock)
        let mut ctx_d = Context::new();
        let (w_d, w2_d, a_d) = {
            let w_d = ctx_d.new_tensor_2d(GgmlType::F32, 4, 4);
            ctx_d.arena_resize_tensor(w_d);
            fill_f32(&mut ctx_d, w_d, |i| (i as f32 * 0.125).sin());
            let w2_d = ctx_d.new_tensor_2d(GgmlType::F32, 4, 4);
            ctx_d.arena_resize_tensor(w2_d);
            fill_f32(&mut ctx_d, w2_d, |i| (i as f32 * 0.3).cos());
            let a_d = ctx_d.new_tensor_2d(GgmlType::F32, 4, 2);
            ctx_d.arena_resize_tensor(a_d);
            fill_f32(&mut ctx_d, a_d, |i| (i as f32 * 0.7).sin());
            (w_d, w2_d, a_d)
        };
        let n0_d = ctx_d.mul_mat(w_d, a_d);
        let n1_d = ctx_d.silu(n0_d);
        let n2_d = ctx_d.mul_mat(w2_d, n1_d);

        // run through the scheduler: explicit alloc → set input → compute
        let mut g = Graph::new(16);
        g.build_forward(&ctx, n2);
        let mut sched = backend_sched_new(&[mock.backend.clone(), cpu], None, 64, false, true);
        assert!(backend_sched_alloc_graph(&mut sched, &mut ctx, &mut g));
        let input_bytes = {
            let n = ctx.nbytes(a) / 4;
            let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
            bytemuck::cast_slice(&vals).to_vec()
        };
        backend_tensor_set(&mut ctx, a, &input_bytes, 0);
        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
        assert_eq!(st, GgmlStatus::Success);

        // direct run
        let mut gd = Graph::new(16);
        gd.build_forward(&ctx_d, n2_d);
        crate::compute::graph_compute(&mut ctx_d, &mut gd, 2);

        // --- the hand-checked split structure ---
        assert_eq!(backend_sched_get_n_splits(&sched), 3, "mock | cpu | mock");
        let ids: Vec<usize> = sched.splits.iter().map(|s| s.backend_id).collect();
        assert_eq!(ids, vec![0, 1, 0]);
        assert_eq!(sched.splits[0].inputs, vec![a], "split 0 copies the input a");
        assert_eq!(sched.splits[1].inputs, vec![n0], "split 1 copies n0 (mock -> cpu)");
        assert_eq!(sched.splits[2].inputs, vec![n1], "split 2 copies n1 (cpu -> mock)");
        assert_eq!(sched.splits[0].nodes, vec![n0]);
        assert_eq!(sched.splits[1].nodes, vec![n1]);
        assert_eq!(sched.splits[2].nodes, vec![n2]);

        // the cross-backend srcs were rewritten to the copies (the C
        // `node->src[j] = tensor_copy(...)` of ggml-backend.cpp:1419)
        let a_cpy = sched.hv_tensor_copies[&(a, 0, 0)];
        let n0_cpy = sched.hv_tensor_copies[&(n0, 1, 0)];
        let n1_cpy = sched.hv_tensor_copies[&(n1, 0, 0)];
        assert_eq!(ctx.tensors[n0.0 as usize].src[1], Some(a_cpy));
        assert_eq!(ctx.tensors[n1.0 as usize].src[0], Some(n0_cpy));
        assert_eq!(ctx.tensors[n2.0 as usize].src[1], Some(n1_cpy));
        // the copies are distinct tensors allocated in the split's backend
        assert_ne!(a, a_cpy);
        assert!(tensor_buffer(&ctx, a_cpy).is_some());
        // backend resolution through the id map
        assert!(backend_sched_get_tensor_backend(&sched, n0).is_some());
        assert!(backend_sched_get_tensor_backend(&sched, n1).is_some());

        // bit-identical output
        assert_eq!(
            ctx.data_bytes(n2).unwrap(),
            ctx_d.data_bytes(n2_d).unwrap(),
            "split graph output must be bit-identical to direct compute"
        );
    }

    /// copy callback (6753a033f, ggml-backend.h:317-327 @c35b66744) — single
    /// backend: the callback is called only for input weights in host buffers
    /// crossing a split, so a CPU-only graph (one backend, no split copies)
    /// must never trigger it.
    #[test]
    fn sched_copy_callback_single_backend_not_called() {
        let mut ctx = Context::new();
        let out = build_toy(&mut ctx);
        let mut g = Graph::new(16);
        g.build_forward(&ctx, out);

        let called = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut sched = backend_sched_new(&[cpu_backend_init()], None, 64, false, true);
        {
            let called = called.clone();
            backend_sched_set_copy_callback(&mut sched, Box::new(move |_b, _ctx, src, _dst, _graph| {
                called.lock().unwrap().push(format!("cb:{}", _ctx.tensors[src.0 as usize].name));
                // returning true would skip the scheduler's own copy — were
                // the callback wrongly invoked, the bit-identity check below
                // would catch the skipped copy too
                true
            }));
        }
        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
        assert_eq!(st, GgmlStatus::Success);
        assert_eq!(backend_sched_get_n_splits(&sched), 1, "single backend — one split");
        assert!(
            called.lock().unwrap().is_empty(),
            "no split copies host weights across backends — callback must not fire"
        );

        // and the compute is unaffected
        let mut ctx_d = Context::new();
        let out_d = build_toy(&mut ctx_d);
        let mut gd = Graph::new(16);
        gd.build_forward(&ctx_d, out_d);
        crate::compute::graph_compute(&mut ctx_d, &mut gd, 4);
        assert_eq!(ctx.data_bytes(out).unwrap(), ctx_d.data_bytes(out_d).unwrap());
    }

    /// copy callback (6753a033f) — two-backend contract: the weight `w` sits
    /// in a CPU (host) buffer with USAGE_WEIGHTS while its MUL_MAT consumer is
    /// pinned to the mock backend via `backend_sched_set_tensor_backend`, so
    /// the scheduler must
    ///   1. copy the split's non-weight input `a` (user INPUT) first,
    ///   2. then offer only `w` to the callback with the split's backend, the
    ///      previous-split `src`, the split-local `dst` copy and the split's
    ///      graph nodes,
    ///   3. a `false` return makes the scheduler copy the entire weight
    ///      (bit-identical output), a `true` return means the user-performed
    ///      copy stands (the scheduler skips its own).
    #[test]
    fn sched_copy_callback_two_backend_contract() {
        // graph: n0 = mul_mat(w, a) — [4,4] x [4,2] -> [4,2]
        let mut setup = |ctx: &mut Context| -> (TensorId, TensorId, TensorId) {
            let w = ctx.new_tensor_2d(GgmlType::F32, 4, 4);
            let a = ctx.new_tensor_2d(GgmlType::F32, 4, 2);
            // w in a CPU (host) buffer with USAGE_WEIGHTS
            // (ggml-backend.cpp:1805-1809 — the is_host_weight predicate)
            let cpu_buft = backend_cpu_buffer_type();
            let buf = backend_buft_alloc_buffer(&cpu_buft, ctx, 1024).unwrap();
            let base = backend_buffer_get_base(&buf).unwrap();
            assert_eq!(backend_tensor_alloc(&buf, ctx, w, base), GgmlStatus::Success);
            backend_buffer_set_usage(&buf, BackendBufferUsage::Weights);
            fill_f32(ctx, w, |i| (i as f32 * 0.125).sin());
            ctx.set_name(w, "w");
            ctx.set_name(a, "a");
            ctx.tensors[a.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;
            let n0 = ctx.mul_mat(w, a);
            ctx.set_name(n0, "n0");
            ctx.tensors[n0.0 as usize].flags |= GGML_TENSOR_FLAG_OUTPUT;
            (w, a, n0)
        };
        let input_vals: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_vals).to_vec();
        // reference mul_mat(scale * w, a) on a plain context
        let direct = |scale: f32| -> Vec<u8> {
            let mut ctx_d = Context::new();
            let w_d = ctx_d.new_tensor_2d(GgmlType::F32, 4, 4);
            ctx_d.arena_resize_tensor(w_d);
            fill_f32(&mut ctx_d, w_d, |i| (i as f32 * 0.125).sin() * scale);
            let a_d = ctx_d.new_tensor_2d(GgmlType::F32, 4, 2);
            ctx_d.arena_resize_tensor(a_d);
            ctx_d.with_f32_mut(a_d, |p| p.copy_from_slice(&input_vals)).unwrap();
            let n0_d = ctx_d.mul_mat(w_d, a_d);
            let mut gd = Graph::new(8);
            gd.build_forward(&ctx_d, n0_d);
            crate::compute::graph_compute(&mut ctx_d, &mut gd, 2);
            ctx_d.data_bytes(n0_d).unwrap().to_vec()
        };

        // ---------- run A: the callback declines (false) -> full copy ----------
        let mut ctx = Context::new();
        let (w, a, n0) = setup(&mut ctx);
        let mock = mock::mock_backend("MOCK", Some(GgmlOp::Silu));
        let cpu = cpu_backend_init();
        let mut g = Graph::new(8);
        g.build_forward(&ctx, n0);
        let mut sched = backend_sched_new(&[mock.backend.clone(), cpu], None, 32, false, true);
        backend_sched_set_tensor_backend(&mut sched, n0, &mock.backend);
        assert!(backend_sched_alloc_graph(&mut sched, &mut ctx, &mut g));
        // after alloc the split copies exist — capture what the callback must see
        let a_cpy = sched.hv_tensor_copies[&(a, 0, 0)];
        let w_cpy = sched.hv_tensor_copies[&(w, 0, 0)];
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        {
            let log = log.clone();
            let expect_input = input_bytes.clone();
            backend_sched_set_copy_callback(&mut sched, Box::new(move |backend, ctx, src, dst, graph| {
                let mut log = log.lock().unwrap();
                // contract: (split backend, src in previous split, dst copy,
                // split graph nodes)
                log.push(format!(
                    "cb:{} src={} dst_is_copy={} graph={}",
                    backend_name(Some(backend)),
                    ctx.tensors[src.0 as usize].name,
                    dst == w_cpy && src == w,
                    graph.len(),
                ));
                // contract: all non-weight inputs of the split have already
                // been copied (two-pass order, ggml-backend.cpp:1872-1885)
                let a_bytes = ctx.data_bytes(a_cpy).unwrap();
                log.push(format!("a_copied_before_callback={}", a_bytes == expect_input.as_slice()));
                false // decline — the scheduler must copy the entire weight
            }));
        }
        backend_tensor_set(&mut ctx, a, &input_bytes, 0);
        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
        assert_eq!(st, GgmlStatus::Success);

        assert_eq!(backend_sched_get_n_splits(&sched), 1, "single pinned node");
        assert_eq!(
            sched.splits[0].inputs.len(),
            2,
            "both w (host weight) and a (user input) cross to the mock"
        );
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 2, "callback fired exactly once");
        assert_eq!(log[0], "cb:MOCK src=w dst_is_copy=true graph=1");
        assert_eq!(log[1], "a_copied_before_callback=true");
        drop(log);
        // declining -> the scheduler copied the whole weight -> bit-identical
        assert_eq!(ctx.data_bytes(n0).unwrap(), &direct(1.0)[..]);

        // ---------- run B: the callback copies 2*w itself (true) -> stands ----------
        let mut ctx = Context::new();
        let (w, a, n0) = setup(&mut ctx);
        let mock = mock::mock_backend("MOCK", Some(GgmlOp::Silu));
        let cpu = cpu_backend_init();
        let mut g = Graph::new(8);
        g.build_forward(&ctx, n0);
        let mut sched = backend_sched_new(&[mock.backend.clone(), cpu], None, 32, false, true);
        backend_sched_set_tensor_backend(&mut sched, n0, &mock.backend);
        assert!(backend_sched_alloc_graph(&mut sched, &mut ctx, &mut g));
        let fired = Arc::new(Mutex::new(0usize));
        {
            let fired = fired.clone();
            backend_sched_set_copy_callback(&mut sched, Box::new(move |_backend, ctx, src, dst, _graph| {
                *fired.lock().unwrap() += 1;
                // the user performs the copy itself: 2*w
                let scaled: Vec<f32> = ctx
                    .f32s(src)
                    .unwrap()
                    .iter()
                    .map(|v| v * 2.0)
                    .collect();
                backend_tensor_set(ctx, dst, bytemuck::cast_slice(&scaled), 0);
                true // handled — the scheduler must NOT copy over it
            }));
        }
        backend_tensor_set(&mut ctx, a, &input_bytes, 0);
        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
        assert_eq!(st, GgmlStatus::Success);
        assert_eq!(*fired.lock().unwrap(), 1);
        // the user copy stands: output == mul_mat(2*w, a), != mul_mat(w, a)
        assert_eq!(ctx.data_bytes(n0).unwrap(), &direct(2.0)[..]);
        assert_ne!(ctx.data_bytes(n0).unwrap(), &direct(1.0)[..]);
    }

    /// gallocr inplace reuse — transcription of the reference's
    /// `graph_reuses_allocation(false)` (tests/test-alloc.cpp:611-638):
    /// consecutive scale ops share storage.
    #[test]
    fn galloc_reuses_inplace_storage() {
        let mut ctx = Context::new();
        let x0 = ctx.new_tensor_1d(GgmlType::F32, 16);
        ctx.set_name(x0, "x0");
        ctx.tensors[x0.0 as usize].flags |= GGML_TENSOR_FLAG_INPUT;
        let x1 = ctx.scale(x0, 2.0);
        ctx.set_name(x1, "x1");
        let x2 = ctx.scale(x1, 2.0);
        ctx.set_name(x2, "x2");
        let x3 = ctx.scale(x2, 2.0);
        ctx.set_name(x3, "x3");
        ctx.tensors[x3.0 as usize].flags |= GGML_TENSOR_FLAG_OUTPUT;

        let mut g = Graph::new(8);
        g.build_forward(&ctx, x3);
        let cpu = cpu_backend_init();
        let mut sched = backend_sched_new(&[cpu], None, 32, false, true);
        assert!(backend_sched_alloc_graph(&mut sched, &mut ctx, &mut g));

        let off = |ctx: &Context, t: TensorId| match ctx.tensors[t.0 as usize].storage {
            Storage::Arena { offset, .. } => offset,
            _ => panic!("not arena storage"),
        };
        assert_eq!(off(&ctx, x1), off(&ctx, x2), "consecutive inplace ops reuse storage");

        // fill the input through the backend API and compute
        let vals: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        backend_tensor_set(&mut ctx, x0, &vals, 0);
        let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
        assert_eq!(st, GgmlStatus::Success);
        let got = ctx.f32s(x3).unwrap();
        for (i, v) in got.iter().enumerate() {
            assert_eq!(v.to_bits(), (i as f32 * 8.0).to_bits(), "x3 = x0 * 8 exactly");
        }
    }

    /// the rebuild flow of the header example (ggml-backend.h:290-300): two
    /// graph computes through one scheduler, with the graph rebuilt between.
    #[test]
    fn sched_repeated_compute() {
        let mut ctx = Context::new();
        let cpu = cpu_backend_init();
        let mut sched = backend_sched_new(&[cpu], None, 64, false, true);

        let mut expected: Vec<Vec<u8>> = Vec::new();
        for round in 0..2 {
            let mark = ctx.mark();
            let out = build_toy(&mut ctx);
            let mut g = Graph::new(16);
            g.build_forward(&ctx, out);
            let st = backend_sched_graph_compute(&mut sched, &mut ctx, &mut g);
            assert_eq!(st, GgmlStatus::Success);
            expected.push(ctx.data_bytes(out).unwrap().to_vec());
            assert_eq!(expected[round].len(), 32);
            // drop the graph tensors (the port's rebuild loop)
            ctx.reset_graph_to(mark);
        }
        // both rounds compute the same toy graph — identical results
        assert_eq!(expected[0], expected[1]);
    }
}
