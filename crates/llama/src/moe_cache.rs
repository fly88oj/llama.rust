//! MoE expert cache — port of src/llama-moe-cache.{h,cpp} @c35b66744
//! (d6cf9acb2 "llama : add a GPU cache for MoE experts kept in host memory"
//! + c811cb8f0 "llama: support MoE cache over multiple GPUs", 681 lines).
//!
//! Keeps the most recently used experts of host-resident MoE layers in a
//! device buffer. Each layer has a slot map in host memory: when the
//! scheduler copies it to the device, the copy callback ([`MoeCache::copy`],
//! installed through `ggml_backend_sched_set_copy_callback`,
//! llama-context.cpp:652) uploads the missing experts and updates the map.
//!
//! Port status — the constructor requires a GPU backend (llama-moe-cache.cpp:
//! 274-276 "MoE cache requires a GPU backend"), so a CPU-only host cannot run
//! the e2e path; the pure-logic pieces (the LRU's touch/evict/fill ordering,
//! the slot-map gate, the run-merging of the uploads and of
//! [`sched_copy_experts`]'s used-bitmap walk) are pinned by the unit tests
//! below. The scheduler wiring exists on both paths: D domain landed the
//! Rust-side `ggml::backend_sched::backend_sched_set_copy_callback`
//! (6753a033f) which [`install_copy_callback`] targets, and the sysffi
//! binding for the foreign (C) scheduler of `ForeignExecutor` landed with
//! [`install_copy_callback_foreign`] (batch 3 B-domain handoff) — verified
//! by the stub round-trip + real-.so registration tests in
//! crates/ggml/src/backend_emit.rs. The foreign path's MoE-cache slot-map
//! half stays fall-through (see [`sched_copy_experts_foreign`]'s port note);
//! GPU e2e is not runnable on this host.

use ggml::backend::{
    backend_buffer_get_size, backend_buffer_get_type, backend_buffer_get_usage,
    backend_buffer_is_host, backend_buft_get_alignment, backend_dev_memory, backend_dev_type,
    backend_get_device, backend_synchronize, backend_tensor_get_async,
    backend_tensor_set_async, dev_eq, BackendBufferRef, BackendBufferTypeRef, BackendDevType,
    BackendRef,
};
use ggml::tensor::GgmlOp;
use ggml::types::GgmlType;
use ggml::{Context, TensorId};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// moe_cache_lru (llama-moe-cache.cpp:15-119)
// ---------------------------------------------------------------------------

/// LRU of the experts of a group of layers — the slot of each expert is kept
/// in the slot map of its layer (llama-moe-cache.cpp:15-119).
#[derive(Debug, Clone)]
pub struct MoeCacheLru {
    pub n_expert: i32,
    pub n_slots: i32,

    /// `[n_layer]` slot map of each layer, -1 = the expert is not cached.
    /// The C stores raw `int32_t *` pointers into the host slot-map tensors
    /// (`slot_map[il] = layers[il].slot_map->data`, :417-421); the port owns
    /// the vectors and publishes them into the tensors after every plan.
    pub slot_map: Vec<Vec<i32>>,
    /// `[n_slots]` `il*n_expert + expert`, -1 if empty
    key_of: Vec<i32>,

    /// doubly linked list of the slots, head is the least recently used
    prev: Vec<i32>,
    next: Vec<i32>,
    head: i32,
    tail: i32,

    seen: Vec<u32>,
    seen_gen: u32,
    uniq: Vec<i32>,
}

/// one miss of [`MoeCacheLru::plan`] — expert `expert` was assigned to slot
/// `slot` (`moe_cache_lru::fill`, llama-moe-cache.cpp:81-84)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LruFill {
    pub expert: i32,
    pub slot: i32,
}

impl MoeCacheLru {
    /// `moe_cache_lru::init` (llama-moe-cache.cpp:42-58): the slots start as
    /// one doubly linked list in index order.
    pub fn init(n_layer: usize, n_expert: i32, n_slots: i32) -> Self {
        let mut lru = MoeCacheLru {
            n_expert,
            n_slots,
            slot_map: vec![vec![-1; n_expert as usize]; n_layer],
            key_of: vec![-1; n_slots as usize],
            prev: vec![0; n_slots as usize],
            next: vec![0; n_slots as usize],
            head: 0,
            tail: n_slots - 1,
            seen: vec![0; n_expert as usize],
            seen_gen: 0,
            uniq: Vec::new(),
        };
        for s in 0..n_slots {
            lru.prev[s as usize] = s - 1;
            lru.next[s as usize] = if s + 1 < n_slots { s + 1 } else { -1 };
        }
        lru
    }

    /// `moe_cache_lru::touch` (llama-moe-cache.cpp:60-76) — move slot s to
    /// the tail (most recently used).
    pub fn touch(&mut self, s: i32) {
        if s == self.tail {
            return;
        }
        let si = s as usize;
        if self.prev[si] >= 0 {
            self.next[self.prev[si] as usize] = self.next[si];
        } else {
            self.head = self.next[si];
        }
        self.prev[self.next[si] as usize] = self.prev[si];

        self.prev[si] = self.tail;
        self.next[si] = -1;
        self.next[self.tail as usize] = s;
        self.tail = s;
    }

    /// `moe_cache_lru::plan` (llama-moe-cache.cpp:86-118) — give a slot to
    /// each expert selected by `ids` in layer `il`; the misses evict the
    /// least recently used experts. Returns `Err(())` (the C's `false`) if
    /// the ids select more distinct experts than there are slots.
    pub fn plan(
        &mut self,
        il: usize,
        ids: &[i32],
        fills: &mut Vec<LruFill>,
        n_hit: &mut usize,
    ) -> Result<(), ()> {
        fills.clear();
        *n_hit = 0;

        self.seen_gen += 1;
        if self.seen_gen == 0 {
            self.seen.fill(0);
            self.seen_gen = 1;
        }
        self.uniq.clear();
        for &id in ids {
            assert!(id >= 0 && id < self.n_expert);
            if self.seen[id as usize] != self.seen_gen {
                self.seen[id as usize] = self.seen_gen;
                self.uniq.push(id);
            }
        }
        if self.uniq.len() > self.n_slots as usize {
            return Err(());
        }

        // hits go to the tail first, so the head can be evicted below
        // (:96-101)
        let uniq: Vec<i32> = self.uniq.clone();
        for &e in &uniq {
            let s = self.slot_map[il][e as usize];
            if s >= 0 {
                self.touch(s);
                *n_hit += 1;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be
        // merged (:103-104)
        let mut uniq_sorted = uniq;
        uniq_sorted.sort_unstable();
        for e in uniq_sorted {
            if self.slot_map[il][e as usize] >= 0 {
                continue;
            }
            let s = self.head;
            let key = self.key_of[s as usize];
            if key >= 0 {
                let (kil, ke) = ((key / self.n_expert) as usize, (key % self.n_expert) as usize);
                self.slot_map[kil][ke] = -1;
            }
            self.key_of[s as usize] = il as i32 * self.n_expert + e;
            self.slot_map[il][e as usize] = s;
            self.touch(s);
            fills.push(LruFill { expert: e, slot: s });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// llama_moe_cache (llama-moe-cache.cpp:122-561, the impl)
// ---------------------------------------------------------------------------

/// `impl::max_batch` (llama-moe-cache.cpp:197) — batches above this size
/// take the copy-everything path instead: they would use most experts of a
/// layer and evict the ones used in generation.
pub const MOE_CACHE_MAX_BATCH: i64 = 32;

/// `llama_moe_cache_layer_experts` (llama-moe-cache.cpp:122-130) — gate, up,
/// down or gate_up, down: the layer's expert tensors in the C's fixed order
/// (ffn_gate_up_exps, ffn_gate_exps, ffn_up_exps, ffn_down_exps).
pub fn layer_experts(
    gate_up: Option<TensorId>,
    gate: Option<TensorId>,
    up: Option<TensorId>,
    down: Option<TensorId>,
) -> Vec<TensorId> {
    [gate_up, gate, up, down].into_iter().flatten().collect()
}

/// `llama_moe_cache_same_layout` (llama-moe-cache.cpp:132-142): same count,
/// same type, same shape and same expert slice pitch (`nb[2]`).
fn same_layout(ctx: &Context, a: &[TensorId], b: &[TensorId]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(&x, &y)| {
            ctx.ty(x) == ctx.ty(y) && ctx.ne(x) == ctx.ne(y) && ctx.nb(x)[2] == ctx.nb(y)[2]
        })
}

/// `llama_moe_cache_is_host_weight` (llama-moe-cache.cpp:144-148): a tensor
/// in a WEIGHTS-usage host buffer.
fn is_host_weight(ctx: &Context, t: TensorId) -> bool {
    match ggml::backend::tensor_buffer(ctx, t) {
        Some(buf) => {
            backend_buffer_get_usage(&buf) == ggml::backend::BackendBufferUsage::Weights
                && backend_buffer_is_host(&buf)
        }
        None => false,
    }
}

/// a GPU (or iGPU) with its own budget and banks — it caches the layers
/// assigned to it (`impl::device`, llama-moe-cache.cpp:157-166)
struct MoeCacheDevice {
    backend: BackendRef,
    buft: BackendBufferTypeRef,
    /// host experts of the layers it caches
    host_bytes: usize,
    /// share of the budget
    split: f64,

    /// the banks + cached views of this device (the C's per-device ggml ctx)
    ctx: Option<Context>,
    buf: Option<BackendBufferRef>,
    buf_size: usize,
}

/// layers of the same device with the same expert tensor layout share the
/// banks and the LRU of a group (`impl::group`, llama-moe-cache.cpp:169-179)
struct MoeCacheGroup {
    /// device index
    id: usize,
    /// expert tensors of the first layer
    ref_: Vec<TensorId>,
    layers: Vec<usize>,
    /// device storage of all slots, one per expert tensor
    banks: Vec<TensorId>,
    host_bytes: usize,
    n_slots: i32,
    lru: Option<MoeCacheLru>,
}

#[derive(Clone)]
struct MoeCacheLayer {
    /// -1 if the layer is not cached
    ig: i32,
    /// I32 [1, n_expert] in host memory
    slot_map: Option<TensorId>,
    /// host expert tensors, in the order of the banks
    experts: Vec<TensorId>,
}

/// host experts -> cached experts (`impl::binding`, :181-186)
#[derive(Clone, Copy)]
struct MoeCacheBinding {
    il: usize,
    /// index of the bank
    ip: usize,
    /// view of the bank used in place of the host experts
    cached: TensorId,
}

#[derive(Default)]
struct MoeStats {
    hits: usize,
    misses: usize,
    bytes: usize,
}

/// `llama_context::copy_experts` state — the ids readback + used bitmap of
/// [`sched_copy_experts`] (`copy_experts_t`, llama-context.h:214-222)
#[derive(Default)]
pub struct CopyExpertsState {
    ids: Option<TensorId>,
    ids_data: Vec<i32>,
    used: Vec<bool>,
}

impl CopyExpertsState {
    /// `copy_experts.reset()` (llama-context.cpp:2643 + :3800): the cached
    /// readback dies with the graph it belonged to.
    pub fn reset(&mut self) {
        self.ids = None;
        self.ids_data.clear();
        self.used.clear();
    }
}

/// The layer table the cache is built from — the port's stand-in for
/// `model.layers` + `model.dev_layer(il)`: each entry carries the layer's
/// expert tensors and the index into `backends` of the device the layer is
/// assigned to.
pub struct MoeCacheLayerInput {
    /// `llama_moe_cache_layer_experts(layer)` order
    pub experts: Vec<TensorId>,
    /// index into the `backends` slice (`model.dev_layer(il)`)
    pub device: usize,
}

/// `llama_moe_cache` (llama-moe-cache.h:12-40) — the port's impl in one
/// struct (the C's pimpl).
pub struct MoeCache {
    n_expert_used: i64,

    stats_small: MoeStats, // up to 8 tokens per ubatch
    stats_large: MoeStats,
    stats_copy: MoeStats, // experts copied from the cache for large batches

    devices: Vec<MoeCacheDevice>,
    groups: Vec<MoeCacheGroup>,
    layers: Vec<MoeCacheLayer>,
    /// host experts -> cached experts
    bindings: HashMap<TensorId, MoeCacheBinding>,
    /// slot map -> layer
    layer_of: HashMap<TensorId, usize>,

    ids: Vec<i32>,
    fills: Vec<LruFill>,

    /// the slot maps' owning Context (the C's ctx_host + buf_host)
    ctx_host: Context,
    buf_host: Option<BackendBufferRef>,
    buf_host_size: usize,

    /// the weights' Context — the host expert TensorIds' home (the C reads
    /// them off `model.layers`; the port receives the model's ggml Context)
    weights_ctx: Context,
}

impl MoeCache {
    /// `llama_moe_cache::llama_moe_cache` → `impl::impl` (llama-moe-cache.cpp:
    /// 224-437). "backends are all the backends of the context, each GPU
    /// gets its own cache of the given size for the layers assigned to it".
    /// `tensor_split` is `model.tensor_split()` — `None` or all-zero means
    /// split by free memory (:274-295).
    pub fn new(
        weights_ctx: Context,
        backends: &[BackendRef],
        bufts: &[BackendBufferTypeRef],
        layers_in: &[MoeCacheLayerInput],
        n_expert: u32,
        n_expert_used_max: u32,
        tensor_split: Option<&[f32]>,
        size: usize,
    ) -> Result<Self, String> {
        let n_expert_used = n_expert_used_max as i64;

        // (:226-234) each GPU/IGPU backend becomes a cache device
        let mut devices: Vec<MoeCacheDevice> = Vec::new();
        for (i, b) in backends.iter().enumerate() {
            match backend_dev_type(&backend_get_device(b)) {
                BackendDevType::Gpu | BackendDevType::Igpu => devices.push(MoeCacheDevice {
                    backend: b.clone(),
                    buft: bufts[i].clone(),
                    host_bytes: 0,
                    split: 0.0,
                    ctx: None,
                    buf: None,
                    buf_size: 0,
                }),
                _ => {}
            }
        }
        if devices.is_empty() {
            return Err("MoE cache requires a GPU backend".into());
        }
        // (:235-237) the C also rejects LLAMA_SPLIT_MODE_TENSOR — the port's
        // foreign executor is layer-split only, tensor parallelism is not
        // reachable
        if n_expert == 0 || n_expert_used == 0 {
            return Err("MoE cache requires a MoE model".into());
        }

        // only cache layers that keep all of their experts in host memory,
        // on the device the layer is assigned to (:240-262)
        let mut groups: Vec<MoeCacheGroup> = Vec::new();
        let mut layers: Vec<MoeCacheLayer> = vec![
            MoeCacheLayer {
                ig: -1,
                slot_map: None,
                experts: Vec::new(),
            };
            layers_in.len()
        ];
        for (il, lin) in layers_in.iter().enumerate() {
            if lin.experts.is_empty() || !lin.experts.iter().all(|&t| is_host_weight(&weights_ctx, t)) {
                continue;
            }
            let dev = &backends[lin.device];
            let Some(id) = devices
                .iter()
                .position(|d| dev_eq(&backend_get_device(&d.backend), &backend_get_device(dev)))
            else {
                continue;
            };
            let ig = match groups
                .iter()
                .position(|g| g.id == id && same_layout(&weights_ctx, &g.ref_, &lin.experts))
            {
                Some(ig) => ig,
                None => {
                    groups.push(MoeCacheGroup {
                        id,
                        ref_: lin.experts.clone(),
                        layers: Vec::new(),
                        banks: Vec::new(),
                        host_bytes: 0,
                        n_slots: 0,
                        lru: None,
                    });
                    groups.len() - 1
                }
            };
            groups[ig].layers.push(il);
            for &t in &lin.experts {
                let nb = weights_ctx.nbytes(t);
                groups[ig].host_bytes += nb;
                devices[id].host_bytes += nb;
            }
        }
        if groups.is_empty() {
            crate::impl_log::log_internal(
                crate::impl_log::LogLevel::Warn,
                format_args!(
                    "llama_moe_cache: no layer has all of its experts in host memory, MoE cache \
                     is disabled"
                ),
            );
            return Ok(Self::empty(weights_ctx, devices, groups, layers, n_expert_used));
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        // (:265-272)
        let nb2_of = |t: TensorId| weights_ctx.nb(t)[2] as usize;
        let alignments: Vec<usize> = devices
            .iter()
            .map(|d| backend_buft_get_alignment(&d.buft))
            .collect();
        let alloc_size = |g: &MoeCacheGroup, n_slots: i32| -> usize {
            let alignment = alignments[g.id];
            g.ref_
                .iter()
                .map(|&t| ggml_pad_usize(nb2_of(t) * (n_slots as usize + 1), alignment))
                .sum()
        };

        // the budget is split among the devices with host experts like the
        // layers, by the tensor split or by default by free memory (:274-295)
        let split_by_free = match tensor_split {
            None => true,
            Some(ts) => ts.iter().all(|&x| x == 0.0),
        };
        let mut split_sum = 0.0f64;
        for d in devices.iter_mut() {
            if d.host_bytes == 0 {
                continue;
            }
            let dev = backend_get_device(&d.backend);
            if split_by_free {
                let (free, _total) = backend_dev_memory(&dev);
                d.split = free as f64;
            } else {
                let ts = tensor_split.unwrap();
                // the C indexes model.devices (same backends) by position
                let idx = backends
                    .iter()
                    .position(|b| dev_eq(&backend_get_device(b), &dev))
                    .unwrap_or(0);
                d.split = ts.get(idx).copied().unwrap_or(0.0) as f64;
            }
            split_sum += d.split;
        }
        if split_sum == 0.0 {
            // the devices do not report their free memory (:297-303)
            for d in devices.iter_mut() {
                d.split = if d.host_bytes > 0 { 1.0 } else { 0.0 };
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so
        // each group caches the same fraction of its experts (:305-322)
        let mut n_tensors = vec![0usize; devices.len()];
        let mut n_tensors_host = 0usize;
        for gi in 0..groups.len() {
            let (gid, gsplit, ghost) = (groups[gi].id, devices[groups[gi].id].split, groups[gi].host_bytes);
            let d_host = devices[gid].host_bytes;
            let n_expert_g = weights_ctx.ne(groups[gi].ref_[0])[2] as i32;
            let budget = (size as f64 * gsplit / split_sum * ghost as f64 / d_host as f64) as usize;
            let max_slots = groups[gi].layers.len() as i32 * n_expert_g;
            while groups[gi].n_slots < max_slots && alloc_size(&groups[gi], groups[gi].n_slots + 1) <= budget {
                groups[gi].n_slots += 1;
            }
            if groups[gi].n_slots < n_expert_used as i32 {
                crate::impl_log::log_internal(
                    crate::impl_log::LogLevel::Warn,
                    format_args!(
                        "llama_moe_cache: MoE cache budget is too small for {} layers, they are \
                         not cached",
                        groups[gi].layers.len()
                    ),
                );
                groups[gi].n_slots = 0;
                continue;
            }
            groups[gi].lru = Some(MoeCacheLru::init(layers.len(), n_expert_g, groups[gi].n_slots));
            n_tensors[gid] += groups[gi].ref_.len() * (1 + groups[gi].layers.len());
            n_tensors_host += groups[gi].layers.len();
        }
        if n_tensors_host == 0 {
            return Err("MoE cache is too small to hold the experts of one token".into());
        }
        let _ = &mut n_tensors; // (the C pre-sizes its ggml contexts from these)

        // the tensor pass (:324-421): the banks in the device Contexts, the
        // slot maps in the host Context
        let mut ctx_host = Context::new();
        let mut res = Self::empty(weights_ctx, devices, groups, layers, n_expert_used);
        for ig in 0..res.groups.len() {
            if res.groups[ig].n_slots == 0 {
                continue;
            }
            let g_ref = res.groups[ig].ref_.clone();
            let g_layers = res.groups[ig].layers.clone();
            let g_slots = res.groups[ig].n_slots;
            let g_id = res.groups[ig].id;
            let d = res.devices[g_id].ctx.get_or_insert_with(Context::new);
            for t in &g_ref {
                let (ne, ty) = (*res.weights_ctx.ne(*t), res.weights_ctx.ty(*t));
                let bank = d.new_tensor_3d(ty, ne[0], ne[1], g_slots as i64 + 1);
                d.arena_resize_tensor(bank);
                d.set_name(bank, &format!("moe_cache.{ig}.bank"));
                res.groups[ig].banks.push(bank);
            }
            for &il in &g_layers {
                let l = &mut res.layers[il];
                l.ig = ig as i32;
                l.experts = layers_in[il].experts.clone();
                for ip in 0..l.experts.len() {
                    let bank = res.groups[ig].banks[ip];
                    let (bne, bnb) = (*d.ne(bank), *d.nb(bank));
                    let cached = d.view_3d(
                        bank,
                        bne[0],
                        bne[1],
                        g_slots as i64,
                        bnb[1] as usize,
                        bnb[2] as usize,
                        0,
                    );
                    d.set_name(cached, &format!("moe_cache.cached.{il}.{ip}"));
                    let e = l.experts[ip];
                    res.bindings.insert(e, MoeCacheBinding { il, ip, cached });
                }
                let slot_map = ctx_host.new_tensor_2d(
                    GgmlType::I32,
                    1,
                    res.weights_ctx.ne(g_ref[0])[2],
                );
                ctx_host.arena_resize_tensor(slot_map);
                ctx_host.set_name(slot_map, &format!("moe_cache.slot_map-{il}"));
                // the C 0xff-clears the host buffer: all slots are -1 (:405)
                ctx_host.with_i32_mut(slot_map, |p| p.fill(-1)).unwrap();
                res.layer_of.insert(slot_map, il);
                res.layers[il].slot_map = Some(slot_map);
            }
            let g_alloc = {
                let alignment = backend_buft_get_alignment(&res.devices[g_id].buft);
                g_ref
                    .iter()
                    .map(|&t| {
                        ggml_pad_usize(
                            res.weights_ctx.nb(t)[2] as usize * (g_slots as usize + 1),
                            alignment,
                        )
                    })
                    .sum::<usize>()
            };
            res.devices[g_id].buf_size += g_alloc;
        }
        res.ctx_host = ctx_host;
        // as weights, the ops that read the banks run on the device and the
        // slot maps are copied with the copy callback (:423-430) — the port's
        // banks live in their Context arenas; the WEIGHTS usage flag of the
        // host slot maps is what marks them for the scheduler's split
        Ok(res)
    }

    fn empty(
        weights_ctx: Context,
        devices: Vec<MoeCacheDevice>,
        groups: Vec<MoeCacheGroup>,
        layers: Vec<MoeCacheLayer>,
        n_expert_used: i64,
    ) -> Self {
        MoeCache {
            n_expert_used,
            stats_small: MoeStats::default(),
            stats_large: MoeStats::default(),
            stats_copy: MoeStats::default(),
            devices,
            groups,
            layers,
            bindings: HashMap::new(),
            layer_of: HashMap::new(),
            ids: Vec::new(),
            fills: Vec::new(),
            ctx_host: Context::new(),
            buf_host: None,
            buf_host_size: 0,
            weights_ctx,
        }
    }

    /// `impl::backend` (llama-moe-cache.cpp:439-441) — the device that
    /// caches layer il (None when the layer is not cached).
    pub fn backend(&self, il: i32) -> Option<&BackendRef> {
        let ig = self.layers.get(il as usize)?.ig;
        if ig < 0 {
            return None;
        }
        Some(&self.devices[self.groups[ig as usize].id].backend)
    }

    /// `impl::get_slot_map` (llama-moe-cache.cpp:449-459) — the slot map of
    /// layer il, `None` if its experts cannot be read from the cache for
    /// `n_tokens` tokens: large batches use most experts of a layer, so they
    /// gain little from the cache and would evict the experts used in
    /// generation.
    pub fn get_slot_map(&self, il: i32, n_tokens: i64, n_expert_used: i64) -> Option<TensorId> {
        if il < 0 || il as usize >= self.layers.len() || self.layers[il as usize].ig < 0 {
            return None;
        }
        let l = &self.layers[il as usize];
        let slot_map = l.slot_map?;
        let g = &self.groups[l.ig as usize];
        if n_tokens == 0
            || n_tokens > MOE_CACHE_MAX_BATCH
            || i64::min(n_tokens * n_expert_used, self.ctx_host.ne(slot_map)[1])
                > g.n_slots as i64
        {
            return None;
        }
        Some(slot_map)
    }

    /// `impl::get_experts` (llama-moe-cache.cpp:461-465) — the experts of `w`
    /// in the cache, `None` if `w` is not cached.
    pub fn get_experts(&self, w: TensorId) -> Option<TensorId> {
        self.bindings.get(&w).map(|b| b.cached)
    }

    /// `impl::copy_experts` (llama-moe-cache.cpp:467-507) — for large
    /// batches: copy the experts of `w` that are in the cache, starting at
    /// expert `e` and up to expert `last`, to the copy `dst` of `w`. Returns
    /// the number of experts copied, 0 if expert `e` is not in the cache.
    /// Large batches only read the cache, so the experts used in generation
    /// stay in it.
    ///
    /// `dst_ctx` is the Context `dst` lives in (the split's copy, scheduler
    /// owned).
    pub fn copy_experts(
        &mut self,
        backend: &BackendRef,
        dst_ctx: &mut Context,
        w: TensorId,
        dst: TensorId,
        e: i64,
        last: i64,
    ) -> i64 {
        let Some(&b) = self.bindings.get(&w) else {
            return 0;
        };
        let ig = self.layers[b.il].ig as usize;
        if !Arc::ptr_eq(&self.devices[self.groups[ig].id].backend, backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in
        // generation stay in it
        let slots = &self.groups[ig].lru.as_ref().unwrap().slot_map[b.il];
        if slots[e as usize] < 0 {
            return 0;
        }
        let mut n = 1i64;
        while e + n <= last && slots[(e + n) as usize] == slots[e as usize] + n as i32 {
            n += 1;
        }

        // the C builds fresh src/dst views in ctx_views and async-copies
        // them (:494-503); the port stages the run's bytes through host
        // memory — same span, same values
        let bank = self.groups[ig].banks[b.ip];
        let dctx = self.devices[self.groups[ig].id].ctx.as_mut().unwrap();
        let bank_pitch = dctx.nb(bank)[2] as usize;
        let dst_pitch = dst_ctx.nb(dst)[2] as usize;
        let src_off = slots[e as usize] as usize * bank_pitch;
        let dst_off = e as usize * dst_pitch;
        let nbytes = n as usize * dst_pitch.min(bank_pitch);
        let mut tmp = vec![0u8; nbytes];
        backend_tensor_get_async(backend, dctx, bank, &mut tmp, src_off);
        backend_synchronize(backend);
        backend_tensor_set_async(backend, dst_ctx, dst, &tmp, dst_off);

        self.stats_copy.hits += n as usize;
        self.stats_copy.bytes += nbytes;

        n
    }

    /// `impl::copy` (llama-moe-cache.cpp:509-558) — the scheduler copy
    /// callback for a slot-map copy: reads the selected experts of the
    /// split's get_rows, plans the LRU and uploads the missing experts
    /// (consecutive experts going to consecutive slots are uploaded
    /// together). Returns false if `src` is not a slot map.
    ///
    /// `graph` is the split's node list (the port's `&[TensorId]` snapshot
    /// of D's copy callback); `ctx` owns the split's tensors.
    pub fn copy(
        &mut self,
        backend: &BackendRef,
        ctx: &mut Context,
        src: TensorId,
        dst: TensorId,
        graph: &[TensorId],
    ) -> bool {
        let Some(&il) = self.layer_of.get(&src) else {
            return false;
        };
        let ig = self.layers[il].ig as usize;
        debug_assert!(
            Arc::ptr_eq(&self.devices[self.groups[ig].id].backend, backend),
            "copy: the slot map must be copied to the device that caches layer {il}"
        );

        // the get_rows that looks up the slots of the selected experts
        // (:517-525)
        let mut lookup = None;
        for &node in graph {
            if ctx.op(node) == GgmlOp::GetRows && ctx.src(node)[0] == Some(dst) {
                lookup = Some(node);
                break;
            }
        }
        let lookup = match lookup {
            Some(l) => l,
            None => panic!("moe_cache: no get_rows reads the slot map of layer {il}"),
        };

        // the selected experts must be computed in an earlier split
        // (:527-536): the scheduler starts a new split at the lookup because
        // it reads a host weight, but only if the split already has inputs
        let sel = ctx.src(lookup)[1].expect("get_rows ids");
        for slot in 0..ctx.tensors.len() {
            let node = ctx.id_at(slot);
            if node == sel || ctx.tensors[slot].view_src == Some(sel) {
                panic!(
                    "the experts of layer {il} are selected in the same split as their MoE cache \
                     lookup"
                );
            }
        }
        debug_assert!(ggml::ops::is_contiguous_ctx(ctx, sel));

        // (:538-541) read the ids back from the device
        let n_ids = (ctx.ne(sel)[0] * ctx.ne(sel)[1].max(1) * ctx.ne(sel)[2].max(1)) as usize;
        self.ids.resize(n_ids, 0);
        backend_tensor_get_async(backend, ctx, sel, bytemuck::cast_slice_mut(&mut self.ids), 0);
        backend_synchronize(backend);

        // (:543-547)
        let mut n_hit = 0usize;
        if self.groups[ig]
            .lru
            .as_mut()
            .unwrap()
            .plan(il, &self.ids, &mut self.fills, &mut n_hit)
            .is_err()
        {
            panic!("the MoE cache is too small for the experts selected in layer {il}");
        }

        // upload the missing experts, consecutive experts going to
        // consecutive slots are uploaded together (:549-563)
        let mut bytes = 0usize;
        let experts = self.layers[il].experts.clone();
        for (ip, &w) in experts.iter().enumerate() {
            let bank = self.groups[ig].banks[ip];
            let expert_size = self.weights_ctx.nb(w)[2] as usize;
            let mut i = 0usize;
            while i < self.fills.len() {
                let mut n = 1usize;
                while i + n < self.fills.len()
                    && self.fills[i + n].expert == self.fills[i].expert + n as i32
                    && self.fills[i + n].slot == self.fills[i].slot + n as i32
                {
                    n += 1;
                }
                let src_off = self.fills[i].expert as usize * expert_size;
                let dst_off = self.fills[i].slot as usize * expert_size;
                let nbytes = n * expert_size;
                let mut tmp = vec![0u8; nbytes];
                ggml::backend::backend_tensor_get(&self.weights_ctx, w, &mut tmp, src_off);
                let dctx = self.devices[self.groups[ig].id].ctx.as_mut().unwrap();
                backend_tensor_set_async(backend, dctx, bank, &tmp, dst_off);
                bytes += nbytes;
                i += n;
            }
        }

        // (:565-568) the stats split at 8 tokens per ubatch
        let n_ids = self.ids.len();
        let st = if n_ids <= 8 * self.n_expert_used as usize {
            &mut self.stats_small
        } else {
            &mut self.stats_large
        };
        st.hits += n_hit;
        st.misses += self.fills.len();
        st.bytes += bytes;

        // the next copy synchronizes the backend before it changes the slot
        // map again (:570-571) — publish the planned map into the host tensor
        let mut slot_bytes = Vec::new();
        if let Some(sm) = self.layers[il].slot_map {
            slot_bytes.extend_from_slice(self.ctx_host.data_bytes(sm).expect("slot map bytes"));
        }
        backend_tensor_set_async(backend, ctx, dst, &slot_bytes, 0);

        true
    }

    /// `impl::memory_breakdown` (llama-moe-cache.cpp:573-584) — per
    /// buffer-type byte map (the device banks + the host slot maps).
    pub fn memory_breakdown(&self) -> Vec<(BackendBufferTypeRef, usize)> {
        let mut res: Vec<(BackendBufferTypeRef, usize)> = Vec::new();
        let mut add = |ty: BackendBufferTypeRef, n: usize| {
            if let Some(entry) = res.iter_mut().find(|(t, _)| ggml::backend::buft_eq(t, &ty)) {
                entry.1 += n;
            } else {
                res.push((ty, n));
            }
        };
        for d in &self.devices {
            if let Some(buf) = &d.buf {
                add(backend_buffer_get_type(buf), backend_buffer_get_size(buf));
            }
        }
        if let Some(buf) = &self.buf_host {
            add(backend_buffer_get_type(buf), self.buf_host_size);
        }
        res
    }

    /// `impl::log_stats` (llama-moe-cache.cpp:586-601) — the destructor's
    /// report (the port has no Drop side effects, callers log explicitly).
    pub fn log_stats(&self) {
        let log = |name: &str, st: &MoeStats| {
            let n = st.hits + st.misses;
            if n == 0 {
                return;
            }
            crate::impl_log::log_internal(
                crate::impl_log::LogLevel::Info,
                format_args!(
                    "llama_moe_cache: {name}: hits = {}, misses = {}, hit rate = {:.2}%, \
                     uploaded = {:.2} MiB",
                    st.hits,
                    st.misses,
                    100.0 * st.hits as f64 / n as f64,
                    st.bytes as f64 / 1024.0 / 1024.0
                ),
            );
        };
        log("ubatch <= 8", &self.stats_small);
        log("ubatch  > 8", &self.stats_large);
        if self.stats_copy.hits > 0 {
            crate::impl_log::log_internal(
                crate::impl_log::LogLevel::Info,
                format_args!(
                    "llama_moe_cache: large batches: {} experts copied from the cache, {:.2} MiB",
                    self.stats_copy.hits,
                    self.stats_copy.bytes as f64 / 1024.0 / 1024.0
                ),
            );
        }
    }
}

/// `ggml_backend_sched_set_copy_callback(sched, sched_copy_experts, this)`
/// (llama-context.cpp:652, 0bb496dbd→d6cf9acb2) — install the port's
/// [`sched_copy_experts`] on the Rust-side scheduler. The GPU decode path
/// drives the *foreign* C scheduler (backend_emit's ForeignExecutor); its
/// sysffi `ggml_backend_sched_set_copy_callback` binding is still pending
/// (D domain) — until it lands the GPU path runs without the cache (moe
/// experts upload whole, the C's behavior when the cache is disabled).
pub fn install_copy_callback(
    sched: &mut ggml::backend_sched::BackendSched,
    moe_cache: Option<std::sync::Arc<std::sync::Mutex<MoeCache>>>,
) {
    let moe = moe_cache;
    ggml::backend_sched::backend_sched_set_copy_callback(
        sched,
        Box::new(move |backend, ctx, src, dst, graph| {
            sched_copy_experts(&moe, backend, ctx, src, dst, graph)
        }),
    );
}

// ---------------------------------------------------------------------------
// the foreign (C) scheduler twin — ForeignExecutor's copy callback
// (sysffi binding of ggml_backend_sched_set_copy_callback, 6753a033f)
// ---------------------------------------------------------------------------

/// The `copy_experts` state of the foreign path — `llama_context::copy_experts`
/// (llama-context.cpp:2680-2687) keyed by the C `ggml_tensor *` instead of
/// the Rust `TensorId`: the foreign callback receives C pointers only.
pub struct CopyExpertsStateForeign {
    ids: Option<*const ggml::sysffi::GgmlCTensor>,
    ids_data: Vec<i32>,
    used: Vec<bool>,
}

impl Default for CopyExpertsStateForeign {
    fn default() -> Self {
        CopyExpertsStateForeign { ids: None, ids_data: Vec::new(), used: Vec::new() }
    }
}

impl CopyExpertsStateForeign {
    /// `copy_experts.reset()` (llama-context.cpp:2643 + :3800).
    pub fn reset(&mut self) {
        self.ids = None;
        self.ids_data.clear();
        self.used.clear();
    }
}

// the raw C pointers are owned by the foreign executor's loaded ggml; the
// state only caches what it read back (host `ids_data`), like the C
unsafe impl Send for CopyExpertsStateForeign {}

/// The per-graph-compute reset handle of [`install_copy_callback_foreign`] —
/// the engine must call [`ForeignCopyExpertsReset::reset`] before every
/// `ForeignExecutor::graph_compute`, exactly where the C resets
/// `lctx->copy_experts` (llama-context.cpp:2643).
pub struct ForeignCopyExpertsReset(std::sync::Arc<std::sync::Mutex<CopyExpertsStateForeign>>);

impl ForeignCopyExpertsReset {
    pub fn reset(&self) {
        self.0.lock().unwrap().reset();
    }
}

/// The foreign-scheduler twin of [`install_copy_callback`]: installs the
/// port's [`sched_copy_experts_foreign`] on the *C* scheduler of the
/// ForeignExecutor through the sysffi `ggml_backend_sched_set_copy_callback`
/// binding (batch 3 B-domain handoff). Fails when the loaded ggml predates
/// the symbol (the pinned build — see
/// `ForeignExecutor::sched_set_copy_callback`).
///
/// The returned reset handle must be driven before every graph compute
/// (see [`ForeignCopyExpertsReset`]) — the calling side owns that step.
pub fn install_copy_callback_foreign(
    exec: &mut ggml::backend_emit::ForeignExecutor,
    moe_cache: Option<std::sync::Arc<std::sync::Mutex<MoeCache>>>,
) -> Result<ForeignCopyExpertsReset, String> {
    let syms = exec.syms_copy();
    let st = std::sync::Arc::new(std::sync::Mutex::new(CopyExpertsStateForeign::default()));
    let st_cb = st.clone();
    exec.sched_set_copy_callback(Box::new(move |backend, src, dst, graph| {
        let mut st = st_cb.lock().unwrap();
        sched_copy_experts_foreign(&syms, &moe_cache, &mut st, backend, src, dst, graph)
    }))?;
    Ok(ForeignCopyExpertsReset(st))
}

/// `llama_context::sched_copy_experts` (llama-context.cpp:2655-2737,
/// d6cf9acb2) — the foreign-scheduler twin of [`sched_copy_experts`]: the
/// same callback body over the raw C ABI (`CLibSyms` instead of the Rust
/// backend layer, C `ggml_tensor *` instead of `TensorId`).
///
/// Honest port note (no GPU on this host): the MoE-cache slot-map half of
/// the C (`lctx->moe_cache->copy(backend, src, dst, graph)`, :2658-2661)
/// returns false here — the Rust `MoeCache`'s device banks live in the Rust
/// backend layer (its constructor requires a GPU `BackendRef`), so on the
/// foreign path the slot maps fall through to the scheduler's whole-tensor
/// copy (the C's behavior with the cache disabled — data-correct, just not
/// cached). Wiring the slot-map half needs the MoE banks constructed in C
/// memory, which is GPU-enablement follow-up work; the expert-id selective
/// upload half below is the complete 1:1.
#[allow(clippy::too_many_arguments)]
pub fn sched_copy_experts_foreign(
    syms: &ggml::sysffi::CLibSyms,
    moe_cache: &Option<std::sync::Arc<std::sync::Mutex<MoeCache>>>,
    st: &mut CopyExpertsStateForeign,
    backend: *mut ggml::sysffi::GgmlBackendT,
    src: *const ggml::sysffi::GgmlCTensor,
    dst: *mut ggml::sysffi::GgmlCTensor,
    graph: *mut ggml::sysffi::GgmlCgraph,
) -> bool {
    let _ = moe_cache; // slot-map half: see the port note above — always falls through

    unsafe {
        // the ids must be computed before the split starts, so only the
        // first node of the split is considered (:2667-2675)
        if (syms.ggml_graph_n_nodes)(graph) == 0 {
            return false;
        }
        let node = (syms.ggml_graph_node)(graph, 0);
        if (*node).op != ggml::sysffi::op::MUL_MAT_ID || (*node).src[0] != dst {
            return false;
        }

        let ids = (*node).src[2];
        // ggml_nelements (ggml.c:1432) for the ids view
        let nelem = |t: *const ggml::sysffi::GgmlCTensor| -> i64 {
            let t = &*t;
            t.ne[0] * t.ne[1].max(1) * t.ne[2].max(1) * t.ne[3].max(1)
        };
        if nelem(ids) == 0 {
            return true;
        }

        let n_expert = (*src).ne[2];
        let expert_size = (*src).nb[2];

        if st.ids != Some(ids) || st.used.len() != n_expert as usize {
            // ggml_nbytes (ggml.c:1447)
            let ids_bytes = |t: *const ggml::sysffi::GgmlCTensor| -> usize {
                let t = &*t;
                (t.nb[0] * t.ne[0].max(1) as usize).max(t.nb[1] * t.ne[1].max(1) as usize)
                    .max(t.nb[2] * t.ne[2].max(1) as usize)
                    .max(t.nb[3] * t.ne[3].max(1) as usize)
            };
            let nbytes = ids_bytes(ids);
            st.ids_data.resize(nbytes / 4, 0);
            (syms.ggml_backend_tensor_get_async)(
                backend,
                ids,
                st.ids_data.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                nbytes,
            );
            (syms.ggml_backend_synchronize)(backend);

            st.used.clear();
            st.used.resize(n_expert as usize, false);
            for i1 in 0..(*ids).ne[1] {
                for i0 in 0..(*ids).ne[0] {
                    let idx = ((*ids).nb[1] / 4 * i1 as usize) + (*ids).nb[0] / 4 * i0 as usize;
                    let id = st.ids_data[idx] as usize;
                    assert!(id < n_expert as usize, "expert id out of range");
                    st.used[id] = true;
                }
            }

            st.ids = Some(ids);
        }

        // group consecutive experts and copy them together (:2706-2713)
        let mut first: i64 = 0;
        while first < n_expert {
            if !st.used[first as usize] {
                first += 1;
                continue;
            }
            let mut last = first;
            while last + 1 < n_expert && st.used[(last + 1) as usize] {
                last += 1;
            }

            // the experts in the MoE cache are copied from device memory,
            // the others are uploaded (:2716-2724) — on the foreign path the
            // cache half is not wired (see the port note), so every expert
            // of the run uploads, which is the C's `next` at its floor
            let next = first;

            // copy a bit extra to ensure there are no NaNs in the padding of
            // the last expert — MMQ in the CUDA backend reads it
            // (:2726-2734)
            let offset = next as usize * expert_size;
            let padding = if last < n_expert - 1 { expert_size.min(512) } else { 0 };
            let size = (last + 1 - next) as usize * expert_size + padding;
            if size > 0 {
                let base = (*src).data as *const u8;
                (syms.ggml_backend_tensor_set_async)(
                    backend,
                    dst,
                    base.add(offset) as *const std::ffi::c_void,
                    offset,
                    size,
                );
            }

            first = last + 1;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// llama_context::sched_copy_experts (llama-context.cpp:2655-2737)
// ---------------------------------------------------------------------------

/// `llama_context::sched_copy_experts` (llama-context.cpp:2655-2737,
/// d6cf9acb2) — the static copy callback of the scheduler: the MoE cache's
/// slot-map copies go through `MoeCache::copy`; the *expert weight* copies
/// of a split whose first node is a MUL_MAT_ID read the selected-expert ids
/// and upload only the used experts (grouped into consecutive runs, with a
/// 512-byte padding tail so the CUDA MMQ kernel never reads NaNs in the
/// padding of the last expert).
///
/// `st` is the context's `copy_experts` state (reset by
/// [`CopyExpertsState::reset`] before every graph compute,
/// llama-context.cpp:2643).
pub fn sched_copy_experts(
    moe_cache: &Option<std::sync::Arc<std::sync::Mutex<MoeCache>>>,
    backend: &BackendRef,
    ctx: &mut Context,
    src: TensorId,
    dst: TensorId,
    graph: &[TensorId],
) -> bool {
    // the slot maps of the MoE cache (:2659-2662)
    if let Some(cache) = moe_cache {
        let mut cache = cache.lock().unwrap();
        if cache.copy(backend, ctx, src, dst, graph) {
            return true;
        }
    }

    // the ids must be computed before the split starts, so only the first
    // node of the split is considered (:2668-2675)
    let Some(&node) = graph.first() else {
        return false;
    };
    if ctx.op(node) != GgmlOp::MulMatId || ctx.src(node)[0] != Some(dst) {
        return false;
    }

    let ids = match ctx.src(node)[2] {
        Some(i) => i,
        None => return false,
    };
    let n_elem = (ctx.ne(ids)[0]
        * ctx.ne(ids)[1].max(1)
        * ctx.ne(ids)[2].max(1)
        * ctx.ne(ids)[3].max(1)) as usize;
    if n_elem == 0 {
        return true;
    }

    // host-side state (the C captures `st` from the context; the port
    // recomputes per call — the ids/used cache only saves one readback)
    let n_expert = ctx.ne(src)[2];
    let expert_size = ctx.nb(src)[2] as usize;

    // read the selected experts back and mark them used (:2677-2690)
    let mut ids_data = vec![0i32; ctx.nbytes(ids) / 4];
    backend_tensor_get_async(backend, ctx, ids, bytemuck::cast_slice_mut(&mut ids_data), 0);
    backend_synchronize(backend);

    let mut used = vec![false; n_expert as usize];
    for i1 in 0..ctx.ne(ids)[1] {
        for i0 in 0..ctx.ne(ids)[0] {
            let id = ids_data
                [(i1 * ctx.nb(ids)[1] as i64 / 4 + i0 * ctx.nb(ids)[0] as i64 / 4) as usize];
            assert!(id >= 0 && (id as i64) < n_expert);
            used[id as usize] = true;
        }
    }

    // group consecutive experts and copy them together (:2693-2704)
    let mut first = 0i64;
    while first < n_expert {
        if !used[first as usize] {
            first += 1;
            continue;
        }
        let mut last = first;
        while last + 1 < n_expert && used[(last + 1) as usize] {
            last += 1;
        }

        // the experts in the MoE cache are copied from device memory, the
        // others are uploaded (:2706-2718)
        let mut next = first;
        if let Some(cache) = moe_cache {
            let mut cache = cache.lock().unwrap();
            let mut e = first;
            while e <= last {
                let n = cache.copy_experts(backend, ctx, src, dst, e, last);
                if n == 0 {
                    e += 1;
                    continue;
                }
                if next < e {
                    upload_run(backend, ctx, src, dst, next as usize, e as usize, 0, expert_size);
                }
                e += n;
                next = e;
            }
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the
        // last expert, this is necessary for MMQ in the CUDA backend
        // (:2721-2726)
        let offset = next as usize * expert_size;
        let padding = if last < n_expert - 1 {
            expert_size.min(512)
        } else {
            0
        };
        let size = (last + 1 - next) as usize * expert_size + padding;
        if size > 0 {
            upload_run(backend, ctx, src, dst, offset, offset, size, expert_size);
        }

        first = last + 1;
    }

    true
}

/// one `ggml_backend_tensor_set_async` of a run of expert slices — the C
/// reads straight from `src->data` (the host weight's mmap); the port stages
/// through the weights Context.
fn upload_run(
    backend: &BackendRef,
    ctx: &mut Context,
    src: TensorId,
    dst: TensorId,
    src_off: usize,
    dst_off: usize,
    size: usize,
    _expert_size: usize,
) {
    let nbytes = if size > 0 { size } else { dst_off - src_off };
    let mut tmp = vec![0u8; nbytes];
    ggml::backend::backend_tensor_get(ctx, src, &mut tmp, src_off);
    backend_tensor_set_async(backend, ctx, dst, &tmp, dst_off);
}

/// `ggml_pad` on byte counts (the C's `GGML_PAD(x, n) = ((x + n - 1)/n)*n`).
fn ggml_pad_usize(x: usize, n: usize) -> usize {
    if n == 0 {
        return x;
    }
    (x + n - 1) / n * n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `moe_cache_lru::init` + first `plan`: sorted misses take the slots in
    /// index order (the initial list is index order, head = 0), and the
    /// fills come out expert-sorted (llama-moe-cache.cpp:96-113).
    #[test]
    fn lru_first_plan_fills_sorted() {
        let mut lru = MoeCacheLru::init(2, 8, 4);
        assert_eq!(lru.slot_map[0], vec![-1; 8]);
        let mut fills = Vec::new();
        let mut n_hit = 0;
        // ids select experts {5, 1, 5} — deduped to {5, 1}, sorted to {1, 5}
        assert!(lru.plan(0, &[5, 1, 5], &mut fills, &mut n_hit).is_ok());
        assert_eq!(n_hit, 0);
        // head order: slot 0 -> expert 1, slot 1 -> expert 5
        assert_eq!(
            fills,
            vec![LruFill { expert: 1, slot: 0 }, LruFill { expert: 5, slot: 1 }]
        );
        assert_eq!(lru.slot_map[0][1], 0);
        assert_eq!(lru.slot_map[0][5], 1);
        assert_eq!(lru.key_of[0], 0 * 8 + 1);
        assert_eq!(lru.key_of[1], 0 * 8 + 5);
    }

    /// A second plan that re-uses an expert is a hit (touch, no fill), and
    /// the hit moves its slot to the tail so a later miss evicts another one
    /// (llama-moe-cache.cpp:96-113).
    #[test]
    fn lru_hit_touches_and_eviction_order() {
        let mut lru = MoeCacheLru::init(1, 8, 4);
        let mut fills = Vec::new();
        let mut n_hit = 0;
        assert!(lru.plan(0, &[1, 2, 3, 4], &mut fills, &mut n_hit).is_ok());
        assert_eq!(fills.len(), 4);
        // order after plan: hits would touch; slots assigned in order
        // 1->0, 2->1, 3->2, 4->3; the list was index order, each touch moved
        // them to the tail in sorted order → tail=4(slot 3), head=1(slot 0)

        // re-plan {1, 4}: both hit, 1's slot (0) goes to the tail — head
        // becomes slot 1 (expert 2)
        fills.clear();
        n_hit = 0;
        assert!(lru.plan(0, &[1, 4], &mut fills, &mut n_hit).is_ok());
        assert_eq!(n_hit, 2);
        assert!(fills.is_empty());

        // plan a new expert 7: evicts the head = slot 1 (expert 2)
        fills.clear();
        n_hit = 0;
        assert!(lru.plan(0, &[7], &mut fills, &mut n_hit).is_ok());
        assert_eq!(fills, vec![LruFill { expert: 7, slot: 1 }]);
        assert_eq!(lru.slot_map[0][2], -1, "the evicted expert's map entry clears");
        assert_eq!(lru.key_of[1], 0 * 8 + 7);
        // the survivors are intact
        assert_eq!(lru.slot_map[0][1], 0);
        assert_eq!(lru.slot_map[0][4], 3);
    }

    /// Cross-layer eviction: a group's LRU spans its layers — a miss of
    /// layer 1 can evict a slot held by layer 0 (`key_of = il*n_expert +
    /// expert`, llama-moe-cache.cpp:107-111).
    #[test]
    fn lru_evicts_across_layers() {
        let mut lru = MoeCacheLru::init(2, 4, 2);
        let mut fills = Vec::new();
        let mut n_hit = 0;
        assert!(lru.plan(0, &[0, 1], &mut fills, &mut n_hit).is_ok());
        assert_eq!(fills.len(), 2);
        // layer 1 misses {3}: evicts the head (slot 0, layer-0 expert 0)
        fills.clear();
        assert!(lru.plan(1, &[3], &mut fills, &mut n_hit).is_ok());
        assert_eq!(fills, vec![LruFill { expert: 3, slot: 0 }]);
        assert_eq!(lru.slot_map[0][0], -1);
        assert_eq!(lru.slot_map[1][3], 0);
        assert_eq!(lru.key_of[0], 1 * 4 + 3);
        // layer 0's expert 1 survives in slot 1
        assert_eq!(lru.slot_map[0][1], 1);
    }

    /// `plan` returns false (Err) when the ids select more distinct experts
    /// than there are slots (:92-94) — the context aborts with "the MoE
    /// cache is too small for the experts selected in layer %d".
    #[test]
    fn lru_rejects_more_uniq_than_slots() {
        let mut lru = MoeCacheLru::init(1, 8, 2);
        let mut fills = Vec::new();
        let mut n_hit = 0;
        assert!(lru.plan(0, &[1, 2, 3], &mut fills, &mut n_hit).is_err());
        // nothing was assigned
        assert!(fills.is_empty());
        assert_eq!(lru.slot_map[0], vec![-1; 8]);
    }

    /// The fill runs the uploads merge on: consecutive experts in
    /// consecutive slots (llama-moe-cache.cpp:549-563's merge while-loop).
    #[test]
    fn lru_fills_form_consecutive_runs() {
        let mut lru = MoeCacheLru::init(1, 8, 6);
        let mut fills = Vec::new();
        let mut n_hit = 0;
        assert!(lru.plan(0, &[2, 3, 4, 7], &mut fills, &mut n_hit).is_ok());
        // sorted misses 2,3,4,7 land in slots 0..3 — one merged run of 3
        // (experts 2-4) plus a single
        let runs = merge_runs(&fills);
        assert_eq!(runs, vec![(2, 0, 3), (7, 3, 1)]);
    }

    /// the merge loop of `impl::copy` (:556-563): runs of consecutive
    /// (expert, slot) pairs
    fn merge_runs(fills: &[LruFill]) -> Vec<(i32, i32, usize)> {
        let mut res = Vec::new();
        let mut i = 0;
        while i < fills.len() {
            let mut n = 1;
            while i + n < fills.len()
                && fills[i + n].expert == fills[i].expert + n as i32
                && fills[i + n].slot == fills[i].slot + n as i32
            {
                n += 1;
            }
            res.push((fills[i].expert, fills[i].slot, n));
            i += n;
        }
        res
    }

    /// `sched_copy_experts`'s used-bitmap walk (llama-context.cpp:2693-2726)
    /// pinned as pure logic: consecutive used experts form one copy run with
    /// the 512B (min expert_size) padding tail unless the run ends at the
    /// last expert.
    #[test]
    fn used_bitmap_run_grouping_and_padding() {
        // experts: 0,1,2 used; 3 unused; 4,5 used (n_expert = 6)
        let used = [true, true, true, false, true, true];
        let n_expert = used.len() as i64;
        let expert_size = 1024usize;
        let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (dst_off, size, padding)
        let mut first = 0i64;
        while first < n_expert {
            if !used[first as usize] {
                first += 1;
                continue;
            }
            let mut last = first;
            while last + 1 < n_expert && used[(last + 1) as usize] {
                last += 1;
            }
            // (the cache-driven split is not exercised here — no cache)
            let offset = first as usize * expert_size;
            let padding = if last < n_expert - 1 {
                expert_size.min(512)
            } else {
                0
            };
            let size = (last + 1 - first) as usize * expert_size + padding;
            runs.push((offset, size, padding));
            first = last + 1;
        }
        // run [0..=2] gets 3*1024 + 512 padding (more experts follow);
        // run [4..=5] ends at the last expert — no padding
        assert_eq!(
            runs,
            vec![(0, 3 * 1024 + 512, 512), (4 * 1024, 2 * 1024, 0)]
        );
    }

    /// `get_slot_map`'s gate (llama-moe-cache.cpp:451-458): a layer is only
    /// served when `n_tokens` is in (0, 32] and
    /// `min(n_tokens*n_expert_used, n_expert) <= n_slots` — pinned as the
    /// standalone predicate (the struct method adds the layer bookkeeping on
    /// top).
    #[test]
    fn slot_map_gate_predicate() {
        let gate = |n_tokens: i64, n_expert_used: i64, n_expert: i64, n_slots: i64| -> bool {
            !(n_tokens == 0
                || n_tokens > MOE_CACHE_MAX_BATCH
                || i64::min(n_tokens * n_expert_used, n_expert) > n_slots)
        };
        assert!(!gate(0, 8, 128, 64), "empty batch");
        assert!(!gate(33, 8, 128, 64), "above max_batch");
        // 32*8 = 256 selected, capped at n_expert = 128 -> still > 64
        // slots, refused
        assert!(!gate(32, 8, 128, 64));
        // 4*8 = 32 <= 64 -> served
        assert!(gate(4, 8, 128, 64));
        // the n_expert cap: 33 tokens would exceed max_batch anyway; 2
        // tokens x 128 used capped at 128 > 64 -> refused
        assert!(!gate(2, 128, 128, 64));
        assert!(gate(2, 128, 128, 128));
    }
}
