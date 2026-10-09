//! KV cache — port of src/llama-kv-cache.cpp / llama-kv-cache.h. Two shapes of
//! the same class live here:
//!
//!   * the plain `llama_kv_cache` (`n_swa = 0`, `swa_type = NONE`) — the port's
//!     original single cache, unchanged for every architecture whose
//!     `hparams.is_swa_any()` is false;
//!   * the `llama_kv_cache_iswa` split (llama-kv-cache-iswa.cpp:52-106): a
//!     *base* cache holding the dense layers plus a second *SWA* cache holding
//!     the `hparams.is_swa(il)` layers (llama-kv-cache-iswa.cpp:53-67 chains
//!     the two filters), each with its own cells, its own size and its own KQ
//!     mask (`n_swa` / `swa_type`, llama-kv-cache.h:270-289).
//!
//! In both shapes the k/v tensors live in the caller's (graph) Context; this
//! struct tracks the TensorIds, the cell bookkeeping and the scatter indices.
//! Layout: k = [n_embd_k_gqa, size], v = [n_embd_v_gqa, size] (v_trans = false,
//! the non-FA CPU layout). Writes go through ggml `set_rows`.

use crate::hparams::{LlamaHparams, LlamaSwaType};
use ggml::types::GgmlType;
use ggml::{Context, TensorId};

/// llama-kv-cells.h llama_kv_cell
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCell {
    /// -1 == empty
    pub pos: i32,
    /// sequences this cell belongs to (bitmask over max 64 seqs)
    pub seq: u64,
    /// `v_cells[s].shift[i]` (llama-kv-cells.h:496-511) — the accumulated
    /// position delta since the last `reset_shift()`, what the K-shift graph
    /// rotates each cached K row by (`set_input_k_shift`,
    /// llama-kv-cache.cpp:1527-1539).
    pub shift: i32,
    /// `v_cells[s].ext[i].tok` — the token the cell caches, what the n-gram
    /// consumers read back (`seq_pos_tok_le`, llama-kv-cells.h:333-345;
    /// written by apply_ubatch when `ubatch.token || ple_n_heads > 0`,
    /// llama-kv-cache.cpp:1142-1151). LLAMA_TOKEN_NULL = -1 when unset.
    pub tok: i32,
}

impl Default for KvCell {
    fn default() -> Self {
        KvCell {
            pos: -1,
            seq: 0,
            shift: 0,
            tok: -1,
        }
    }
}

impl KvCell {
    pub fn is_empty(&self) -> bool {
        self.pos == -1
    }

    /// llama-kv-cells.h `seq_count`
    pub fn seq_count(&self) -> u32 {
        self.seq.count_ones()
    }
}

pub struct KvLayer {
    pub k: TensorId,
    pub v: TensorId,
}

/// contiguous slot range [s0, s1] (inclusive)
#[derive(Clone, Copy, Debug)]
pub struct SlotInfo {
    pub s0: u32,
    pub s1: u32,
}

/// hparams subset the second cache is built from — the `llama_kv_cache_iswa`
/// constructor arguments that come from the model (llama-kv-cache-iswa.cpp:
/// 52-105) plus the size inputs of its `size_swa` formula (:69-81).
#[derive(Clone, Debug)]
pub struct SwaCacheSpec {
    /// hparams.n_swa (llama-hparams.h:169) — becomes the SWA cache's own n_swa
    pub n_swa: u32,
    /// hparams.swa_type (llama-hparams.h:168)
    pub swa_type: LlamaSwaType,
    /// hparams.is_swa(il) per model layer (llama-hparams.cpp:289-295)
    pub is_swa: Vec<bool>,
    /// llama_context_params.swa_full (llama-kv-cache-iswa.cpp:76-81). Default
    /// true in llama_context_default_params (llama-context.cpp:3729) and in
    /// common's params, i.e. the SWA cache is as large as the base one.
    pub swa_full: bool,
    /// llama_kv_cache_iswa's `unified` (:73) — llama_context_params.kv_unified.
    /// Only the size formula reads it; for the port's single sequence both
    /// branches give `n_swa`.
    pub unified: bool,
    /// llama_context_params.n_seq_max (:73)
    pub n_seq_max: u32,
}

impl Default for SwaCacheSpec {
    fn default() -> Self {
        SwaCacheSpec {
            n_swa: 0,
            swa_type: LlamaSwaType::NONE,
            is_swa: Vec::new(),
            swa_full: true,
            unified: true,
            n_seq_max: 1,
        }
    }
}

impl SwaCacheSpec {
    /// The `llama_kv_cache_iswa` model-dependent arguments (llama-hparams.h:168
    /// `swa_type`, :169 `n_swa`, :183 `is_swa_impl` → `is_swa(il)`,
    /// llama-hparams.cpp:289-295). `swa_full`/`unified`/`n_seq_max` keep the
    /// llama_context_params defaults (llama-context.cpp:3729-3730,
    /// llama-model.cpp:2481 passes `cparams.n_ubatch` separately).
    pub fn from_hparams(hp: &LlamaHparams) -> Self {
        SwaCacheSpec {
            n_swa: hp.n_swa,
            swa_type: hp.swa_type,
            is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
            ..SwaCacheSpec::default()
        }
    }
}

/// The second `llama_kv_cache` of the `iswa` pair (llama-kv-cache-iswa.cpp:
/// 102-105) — the SWA cache's cells and geometry. Its tensors stay in the
/// parent's `layers` (llama-kv-cache.h:311 `std::vector<kv_layer> layers`), so
/// a SWA layer's `layers[il]` *is* the SWA cache's tensor: the C keeps one
/// `layers` vector per cache and looks it up through `map_layer_ids`
/// (llama-kv-cache.h:314).
pub struct KvSwaCache {
    pub cells: Vec<KvCell>,
    pub size: u32,
    /// `v_heads` of this cache (llama-kv-cache.h:296), the ring scan start.
    /// `head = last placed cell + 1` and may equal `size` (llama-kv-cache.cpp:
    /// 1182-1186).
    pub head: u32,
    /// the cache's own n_swa / swa_type — llama-kv-cache.h:271/289 "this is
    /// the SWA type of the cache - not to be confused with the model SWA
    /// type"; they are what `is_masked_swa` is called with
    /// (llama-kv-cache.cpp:1057 / :1688 / :2084).
    pub n_swa: u32,
    pub swa_type: LlamaSwaType,
    /// model layer ids this cache holds, ascending (the `filter_swa` chain of
    /// llama-kv-cache-iswa.cpp:61-67)
    pub layer_ids: Vec<usize>,
}

/// Per-step graph inputs of the SWA cache (llama-graph.cpp:612-651
/// `llm_graph_input_attn_kv_iswa::set_input`): the base cache and the SWA
/// cache each carry their own scatter indices and their own KQ mask, because
/// the same ubatch is placed in different cells of the two caches.
#[derive(Clone, Copy, Debug)]
pub struct KvSwaStep {
    /// `self_k_idxs_swa` — cpy_k/cpy_v destination cells (:628-631)
    pub row_idx: TensorId,
    /// `self_kq_mask_swa` — n_swa/swa_type-masked, over the SWA cache's cells
    /// (:635 → llama-kv-cache.cpp:1738 `set_input_kq_mask`)
    pub kq_mask: TensorId,
}

// ===========================================================================
// arch batch 7 (2026-09-24), deepseek4: the compressed DSV4 KV cache
// (`llama_kv_cache_dsv4`, llama-kv-cache-dsv4.cpp) — the fourth cache shape.
//
// The C class composes: the `llama_kv_cache_iswa` raw SWA pair (this port's
// existing `KvCache::new_swa`, :1255-1258), three compressed K caches
// (`kv_csa`/`kv_hca`/`kv_lid`, :1292-1311 — plain `llama_kv_cache`s sized
// GGML_PAD(ceil(kv_size/ratio), 256), each filtered to the layers whose
// `dsv4_compress_ratios[il]` matches) and three `llama_dsv4_comp_state`
// compressor-state planes (csa/hca/lid, :1315-1329).
//
// Stream layout (the multi-sequence port): the C forces
// `unified_raw = false` / `unified_compressed = false` (:1244/:1287), i.e.
// one stream per sequence (`n_stream = n_seq_max`) for the compressed caches
// AND the compressor states — `dsv4_build_comp_plan` throws "DSV4 single
// compressed stream cannot serve multiple sequences" (:442-444) otherwise,
// because the compressor ring state is per-sequence. The port models a
// stream as one row-block of the (widened) 2-D tensors — rows
// `[s*size, (s+1)*size)` of a comp cache, plane group `d*n_stream + s`
// (`d = 0` live, `d = 1..=n_rs_seq` rollback snapshots, :964-966) of a comp
// state — and splits multi-sequence batches one sequence per ubatch
// (`split_seq`, the C's per-seq `split_equal` shape) so each graph step
// serves exactly one stream.
//
// Rollback planes (`n_rs_seq > 0`, :653-716): the C's graph writes per-token
// snapshot planes inside one ubatch (`dsv4_build_state_snapshot`,
// deepseek4.cpp:245-263) and restores the live plane at the first ubatch
// after a pending `rs_idx[seq]` (`dsv4_build_state_restore`,
// deepseek4.cpp:222-243). The port keeps the same plane layout and the same
// plan index math ([`Dsv4Plan::build_seq`]) but performs the restore /
// snapshot row-copies host-side ([`Dsv4CompState::restore_seq`] /
// [`Dsv4CompState::snapshot_shift_seq`], driven by
// `crate::context::DecodeContext`), stepping one token per ubatch while
// `n_rs_seq > 0` — the port's generalization of the C's
// `split_equal(n_keep_tail = n_rs_seq + 1)` trailing-token rule
// ([TAG_RECURRENT_ROLLBACK_SPLITS], :1411-1413): observable rollback
// semantics identical, only the write path differs (host copy vs graph
// set_rows).
//
// Not ported (see PARITY.md): `seq_add`/`seq_div` beyond the raw pair (the C
// forwards them and aborts at the K-shift, :1544-1550 +
// llama-kv-cache.cpp:859 — the port refuses at call time). The dsv4 state
// serialization (:1080-1158/:1594-1673) IS ported — see the
// `llama_state_seq_*` section at the bottom of this file.
// ===========================================================================

/// DSV4_CSA_RATIO / DSV4_HCA_RATIO (llama-kv-cache-dsv4.cpp:18-19 and
/// deepseek4.cpp:266-267)
pub const DSV4_CSA_RATIO: u32 = 4;
pub const DSV4_HCA_RATIO: u32 = 128;

/// `dsv4_comp_size` (llama-kv-cache-dsv4.cpp:28-30): a compressed cache holds
/// one cell per completed block.
pub fn dsv4_comp_size(kv_size: u32, ratio: u32) -> u32 {
    1.max(kv_size.div_ceil(ratio))
}

/// `dsv4_stream_offset` (llama-kv-cache-dsv4.cpp:51-60): the row offset of
/// sequence `seq_id` inside a per-stream tensor of `size` rows per stream —
/// 0 for a single (unified) stream, `seq_id*size` otherwise.
fn dsv4_stream_offset(n_stream: u32, seq_id: i32, size: u32) -> i64 {
    if n_stream <= 1 {
        return 0;
    }
    assert!(
        seq_id >= 0 && (seq_id as u32) < n_stream,
        "DSV4 sequence id out of stream range"
    );
    seq_id as i64 * size as i64
}

/// `llama_dsv4_comp_state` (llama-kv-cache-dsv4.cpp:893-1000) — the per-layer
/// compressor scratch: one F32 `kv` and one F32 `score` plane set of
/// `[n_embd_state, state_size]` rows per `(stream, snapshot)` group. The C's
/// 3-D `[n_embd_state, state_size, n_stream*(1 + n_rs_seq)]` (:964-966) is a
/// row-major 2-D `[n_embd_state, state_size*n_stream*(1 + n_rs_seq)]` here;
/// plane group `d*n_stream + s` occupies rows
/// `[(d*n_stream + s)*state_size, +state_size)`. Layer-filtered to the
/// matching ratio (:1271-1285).
pub struct Dsv4CompState {
    /// model layer ids this state holds, ascending (`map_layer_ids`)
    pub layer_ids: Vec<usize>,
    /// per filtered layer, the F32 plane set (all streams and snapshots)
    pub kv: Vec<TensorId>,
    /// per filtered layer, the F32 plane set (all streams and snapshots)
    pub score: Vec<TensorId>,
    pub ratio: u32,
    /// the token ring length (csa/lid: 2*ratio, hca: ratio, :1316-1329)
    pub state_size: u32,
    /// row width (csa: 2*n_embd_head_k, hca: n_embd_head_k, lid:
    /// 2*indexer_head_size)
    pub n_embd_state: i64,
    /// `n_stream = unified ? 1 : n_seq_max` (:907) — the C forces
    /// `unified_compressed = false` (:1287), so one stream per sequence
    pub n_stream: u32,
    /// number of rollback snapshot plane groups after the live one (:908)
    pub n_rs_seq: u32,
}

impl Dsv4CompState {
    fn layer_slot(&self, il: usize) -> usize {
        self.layer_ids
            .iter()
            .position(|&l| l == il)
            .expect("comp state layer")
    }

    /// `get_n_rows` (:1067-1069) — `state_size*n_stream` (the live plane
    /// group's rows)
    pub fn get_n_rows(&self) -> u32 {
        self.state_size * self.n_stream
    }

    /// the row range of plane group `(d, seq)` — the C's `dsv4_clear_tensor_stream`
    /// offsets (:32-39) and `get_kv`/`get_score` views (:1174-1186) address
    /// the same rows
    fn plane_rows(&self, d: u32, seq: u32) -> std::ops::Range<usize> {
        let start = ((d * self.n_stream + seq) * self.state_size) as usize;
        let len = self.state_size as usize;
        start..start + len
    }

    /// `get_kv_all` (llama-kv-cache-dsv4.cpp:1160-1165): the whole plane set
    /// — `get_n_rows()*(1 + n_rs_seq)` rows (== the tensor). The graph
    /// concatenates this view with the ubatch's scratch rows and gathers with
    /// the plan's indices, which address `[plane 0 | snapshot planes |
    /// scratch | zero-row]` (see [`Dsv4Plan::build_seq`] — the C's gather
    /// base is the plane-0 view of deepseek4.cpp:979-980 and its scratch
    /// follows the live rows directly; the port's graph keeps the full view,
    /// so the plan's scratch offsets start after the snapshot planes —
    /// value-identical, the snapshot rows are never gathered).
    pub fn get_kv_all(&self, il: usize) -> TensorId {
        self.kv[self.layer_slot(il)]
    }

    /// `get_score_all` (:1167-1172)
    pub fn get_score_all(&self, il: usize) -> TensorId {
        self.score[self.layer_slot(il)]
    }

    /// `cpy_kv` (:1188-1190) — set_rows of `cur` at `idxs` into the plane set
    /// (idxs are live-plane row indices, `< get_n_rows()`)
    pub fn cpy_kv(&self, ctx: &mut Context, cur: TensorId, idxs: TensorId, il: usize) -> TensorId {
        ctx.set_rows(self.kv[self.layer_slot(il)], cur, idxs)
    }

    /// `cpy_score` (:1192-1194)
    pub fn cpy_score(
        &self,
        ctx: &mut Context,
        cur: TensorId,
        idxs: TensorId,
        il: usize,
    ) -> TensorId {
        ctx.set_rows(self.score[self.layer_slot(il)], cur, idxs)
    }

    /// `clear` (llama-kv-cache-dsv4.cpp:1002-1023, seq_id < 0 / all — zero
    /// every plane of every stream)
    pub fn clear(&self, ctx: &mut Context) {
        for &t in self.kv.iter().chain(self.score.iter()) {
            let _ = ctx.data_bytes_mut(t).map(|b| b.fill(0));
        }
    }

    /// `clear(seq_id >= 0, data = true)` (:1007-1017) — zero the
    /// `(d, seq_id)` plane groups of `kv` and `score` for every `d` (the
    /// `dsv4_clear_tensor_stream` loop over `d <= n_rs_seq`, :1011-1015)
    pub fn clear_seq(&self, ctx: &mut Context, seq: u32) {
        assert!(seq < self.n_stream);
        for &t in self.kv.iter().chain(self.score.iter()) {
            let row_len = ctx.ne(t)[0] as usize * 4; // F32
            let _ = ctx.data_bytes_mut(t).map(|b| {
                for d in 0..=self.n_rs_seq {
                    let r = self.plane_rows(d, seq);
                    b[r.start * row_len..r.end * row_len].fill(0);
                }
            });
        }
    }

    /// `llama_dsv4_comp_state::seq_cp` (:1025-1037) + `apply_copies`
    /// (:1039-1049): the C clears the destination stream's planes and defers
    /// a copy of the per-stream **live** views (`kv_stream[s]`, :974-977) to
    /// the next update; the port owns host memory, so the copy runs inline.
    /// The snapshot planes of `dst` stay zero (the C copies the live plane
    /// only), and the caller resets `rs_idx[dst]` (:1524-1526).
    pub fn seq_cp(&self, ctx: &mut Context, seq_id_src: u32, seq_id_dst: u32) {
        assert!(seq_id_src < self.n_stream && seq_id_dst < self.n_stream);
        if seq_id_src == seq_id_dst {
            return; // :1029-1031
        }
        self.clear_seq(ctx, seq_id_dst); // clear(seq_id_dst, true) (:1033)
        let row_len = self.n_embd_state as usize * 4; // F32
        let src = self.plane_rows(0, seq_id_src);
        let dst = self.plane_rows(0, seq_id_dst);
        for &t in self.kv.iter().chain(self.score.iter()) {
            let _ = ctx.data_bytes_mut(t).map(|b| {
                b.copy_within(src.start * row_len..src.end * row_len, dst.start * row_len);
            });
        }
    }

    /// `dsv4_build_state_restore` (deepseek4.cpp:222-243) applied host-side:
    /// copy plane group `(rollback, seq)` onto the live group `(0, seq)`
    /// (the plan's `state_restore_src/dst_idxs`, :653-678) and re-shift the
    /// remaining snapshot groups so group `d` continues to hold the state
    /// from `rollback + d` tokens back (clamped at the oldest available —
    /// the C's own single-token-ubatch snapshots degrade the same way, see
    /// the module header).
    pub fn restore_seq(&self, ctx: &mut Context, seq: u32, rollback: u32) {
        assert!(seq < self.n_stream && rollback >= 1 && rollback <= self.n_rs_seq);
        let row_len = self.n_embd_state as usize * 4; // F32
        for &t in self.kv.iter().chain(self.score.iter()) {
            let _ = ctx.data_bytes_mut(t).map(|b| {
                // gather the permutation sources first (groups may alias)
                let mut planes: Vec<Vec<u8>> = Vec::with_capacity(self.n_rs_seq as usize + 1);
                for d in 0..=self.n_rs_seq {
                    let r = self.plane_rows(d, seq);
                    planes.push(b[r.start * row_len..r.end * row_len].to_vec());
                }
                let live = self.plane_rows(0, seq);
                b[live.start * row_len..live.end * row_len]
                    .copy_from_slice(&planes[rollback as usize]);
                for d in 1..=self.n_rs_seq {
                    let src = (rollback + d).min(self.n_rs_seq) as usize;
                    let r = self.plane_rows(d, seq);
                    b[r.start * row_len..r.end * row_len].copy_from_slice(&planes[src]);
                }
            });
        }
    }

    /// `dsv4_build_state_snapshot` (deepseek4.cpp:245-263) applied host-side
    /// for a single-token step (the port's `split_equal(n_keep_tail)`
    /// generalization — one token per ubatch, so every snapshot group `d`
    /// becomes the state from `d` tokens back): save the pre-step live group,
    /// shift `d = n_rs_seq..=2` one group down and move the saved copy into
    /// group 1.
    pub fn snapshot_shift_seq(&self, ctx: &mut Context, seq: u32) {
        if self.n_rs_seq == 0 {
            return;
        }
        assert!(seq < self.n_stream);
        let row_len = self.n_embd_state as usize * 4; // F32
        for &t in self.kv.iter().chain(self.score.iter()) {
            let _ = ctx.data_bytes_mut(t).map(|b| {
                let live = self.plane_rows(0, seq);
                let saved = b[live.start * row_len..live.end * row_len].to_vec();
                for d in (2..=self.n_rs_seq).rev() {
                    let src = self.plane_rows(d - 1, seq);
                    let dst = self.plane_rows(d, seq);
                    b.copy_within(src.start * row_len..src.end * row_len, dst.start * row_len);
                }
                let one = self.plane_rows(1, seq);
                b[one.start * row_len..one.end * row_len].copy_from_slice(&saved);
            });
        }
    }
}

/// The compressed K caches `kv_csa`/`kv_hca`/`kv_lid`
/// (llama-kv-cache-dsv4.cpp:1292-1311): plain K-only `llama_kv_cache`s of
/// `[n_embd_head, GGML_PAD(dsv4_comp_size(kv_size, ratio), 256)]` F16 rows
/// per stream (`unified_compressed = false`, :1287 — the C's 3-D
/// `[n_embd_head, size, n_stream]` storage is the row-block layout here),
/// one per filtered layer. The C never runs `apply_ubatch` on them (their
/// `sinfos` exist for the stream-planned `get_k` view only, :767-802), so
/// unlike the raw caches they carry **no cells** — every write goes through
/// the plan's `state_write_idxs` set_rows and every read is masked by the
/// plan's kq_mask.
pub struct Dsv4CompCache {
    pub layer_ids: Vec<usize>,
    pub k: Vec<TensorId>,
    pub size: u32,
    pub n_embd_head: i64,
    /// streams = sequences (`n_stream = n_seq_max`, :1287/:1294)
    pub n_stream: u32,
    /// the stream of the step being built — the `sinfo.strm[s]` view offset
    /// of `dsv4_build_comp_sinfos` (:779-792); set by the driver per ubatch
    /// (one sequence per ubatch, so a single stream per step)
    pub cur_stream: u32,
}

impl Dsv4CompCache {
    fn layer_slot(&self, il: usize) -> usize {
        self.layer_ids
            .iter()
            .position(|&l| l == il)
            .expect("comp cache layer")
    }

    /// `get_k` over a comp layer — the `ggml_view_4d`
    /// [n_embd_head, 1, n_kv, 1] of `llama_kv_cache::get_k`
    /// (llama-kv-cache.cpp:1266-1287; n_head_kv = 1) at the current
    /// stream's row block (`dsv4_stream_offset(n_stream, cur_stream, size)`)
    pub fn get_k(&self, ctx: &mut Context, il: usize, n_kv: u32) -> TensorId {
        let k = self.k[self.layer_slot(il)];
        let size = self.size as i64;
        let n_embd = self.n_embd_head;
        let rs = GgmlType::F16.row_size(n_embd as usize);
        let off = dsv4_stream_offset(self.n_stream, self.cur_stream as i32, self.size) as usize;
        ctx.view_4d(
            k,
            n_embd,
            1,
            n_kv as i64,
            1,
            rs,
            rs,
            rs * size as usize,
            off * rs,
        )
    }

    /// `cpy_k` (llama-kv-cache.cpp:1318-1346, n_stream = 1): dims 0/1 of
    /// `k_cur` merge into one row dim (the c's `ggml_view_2d`), then one
    /// set_rows at `k_idxs` (already stream-offset by the plan) into the
    /// whole storage
    pub fn cpy_k(
        &self,
        ctx: &mut Context,
        k_cur: TensorId,
        k_idxs: TensorId,
        il: usize,
    ) -> TensorId {
        let (n_embd_head, n_head, n_tokens, nb2) = {
            let (ne, nb) = (ctx.ne(k_cur), ctx.nb(k_cur));
            (ne[0], ne[1], ne[2], nb[2] as usize)
        };
        let rows = ctx.view_2d(k_cur, n_embd_head * n_head, n_tokens, nb2, 0);
        ctx.set_rows(self.k[self.layer_slot(il)], rows, k_idxs)
    }

    /// `clear_compressed(-1, data)` (llama-kv-cache-dsv4.cpp:1728-1732) —
    /// zero the whole K storage
    pub fn clear(&self, ctx: &mut Context) {
        for &t in &self.k {
            let _ = ctx.data_bytes_mut(t).map(|b| b.fill(0));
        }
    }

    /// `clear_compressed(seq_id, data)`'s stream clear (:1736-1744,
    /// `dsv4_clear_tensor_stream` at stream `seq_id`) — zero the row block
    /// `[seq*size, (seq+1)*size)`
    pub fn clear_seq(&self, ctx: &mut Context, seq: u32) {
        assert!(seq < self.n_stream);
        let row_len = GgmlType::F16.row_size(self.n_embd_head as usize);
        let start = dsv4_stream_offset(self.n_stream, seq as i32, self.size) as usize;
        for &t in &self.k {
            let _ = ctx.data_bytes_mut(t).map(|b| {
                b[start * row_len..(start + self.size as usize) * row_len].fill(0);
            });
        }
    }

    /// `llama_kv_cache::seq_cp`'s cross-stream branch (llama-kv-cache.cpp:
    /// 494-536) reduced to the K rows: the C resets the destination stream's
    /// cells and enqueues `ggml_backend_tensor_copy(k_stream[s0],
    /// k_stream[s1])` (:509-510 + update's apply, :817-852); the port owns
    /// host memory, so the row-block copy runs inline
    pub fn seq_cp(&self, ctx: &mut Context, seq_id_src: u32, seq_id_dst: u32) {
        assert!(seq_id_src < self.n_stream && seq_id_dst < self.n_stream);
        if seq_id_src == seq_id_dst {
            return;
        }
        let row_len = GgmlType::F16.row_size(self.n_embd_head as usize);
        let src = dsv4_stream_offset(self.n_stream, seq_id_src as i32, self.size) as usize;
        let dst = dsv4_stream_offset(self.n_stream, seq_id_dst as i32, self.size) as usize;
        for &t in &self.k {
            let _ = ctx.data_bytes_mut(t).map(|b| {
                b.copy_within(
                    src * row_len..(src + self.size as usize) * row_len,
                    dst * row_len,
                );
            });
        }
    }
}

/// `llama_kv_cache_dsv4_context::comp_plan`
/// (llama-kv-cache-dsv4.h, filled by `dsv4_build_comp_plan` :427-737) — the
/// per-ubatch index plan of one compressor. Single-stream, no-rollback slice.
#[derive(Clone, Default, Debug)]
pub struct Dsv4Plan {
    /// completed blocks visible to each token: `(pos[i] + 1)/ratio` (:499)
    pub n_visible: Vec<i32>,
    /// `dsv4_comp_graph_n_stream` (:196-204) — 1 (single sequence)
    pub n_stream: i64,
    /// the padded comp-mask width (:641)
    pub n_kv: u32,
    /// `pos[i] % ratio` per token (:497)
    pub state_pos: Vec<i32>,
    pub state_persist_src_idxs: Vec<i32>,
    pub state_persist_dst_idxs: Vec<i32>,
    /// rollback planes — always empty at n_rs_seq == 0
    pub state_restore_src_idxs: Vec<i32>,
    pub state_restore_dst_idxs: Vec<i32>,
    pub state_snapshot_src_idxs: Vec<i32>,
    pub state_snapshot_dst_idxs: Vec<i32>,
    /// `[all blocks' prev-window | all blocks' cur-window]` for the overlap
    /// compressor, plain block gathers for hca (:461-462/:629-636)
    pub state_read_idxs: Vec<i32>,
    /// comp-cache cell per completed block: `pos/ratio` (:526)
    pub state_write_idxs: Vec<i64>,
    /// the block's source_start position (:527)
    pub state_write_pos: Vec<i32>,
}

impl Dsv4Plan {
    /// the single-sequence, `n_stream == 1`, `n_rs_seq == 0` slice of
    /// [`Dsv4Plan::build_seq`] — the shape the existing callers
    /// (single-sequence decode) use
    pub fn build(
        pos: &[i32],
        ratio: u32,
        overlap: bool,
        state_size: u32,
        kv_size: u32,
    ) -> Dsv4Plan {
        let seq_ids = vec![0i32; pos.len()];
        Self::build_seq(
            pos,
            &seq_ids,
            ratio,
            overlap,
            state_size,
            kv_size,
            1,
            0,
            &[],
        )
    }

    /// `dsv4_build_comp_plan` (llama-kv-cache-dsv4.cpp:427-737) for a
    /// one-sequence ubatch (the port splits multi-sequence batches with
    /// `split_seq`, the per-seq shape of the C's `split_equal` fan-out):
    ///
    /// * every token's fresh state is a *scratch row* of the concatenated
    ///   [plane set | this-ubatch states | zero-row] gather source
    ///   (:473-488); a `pos < 0` read maps past the scratch to the appended
    ///   zero/-inf row. NOTE: the C's gather base is the *live* plane
    ///   (`dsv4_view_2d(…, get_n_rows(), 0)`, deepseek4.cpp:979-980) so its
    ///   scratch starts at `state_rows`; the port's graph concatenates the
    ///   **whole** plane set ([`Dsv4CompState::get_kv_all`]), so the scratch
    ///   starts after the `(1 + n_rs_seq)` plane groups — the two agree at
    ///   `n_rs_seq == 0` and the snapshot rows are never gathered otherwise;
    /// * a block completes at each `(pos + 1) % ratio == 0` token (:519-543),
    ///   its comp-cache cell at `stream_off(kv_size) + pos/ratio` (:524-527);
    /// * the CSA dummy block keeps one write per reserve-block when the
    ///   ubatch's positions do not complete the last block (:547-600);
    /// * the HCA dummy block keeps the write path present when *no* block
    ///   completes (:602-627) — masked out (cell `kv_size - 1`);
    /// * `n_kv` is padded to 256 so the graph shape never changes (:641);
    /// * `n_rs_seq > 0`: the restore/snapshot index vectors of :653-716 are
    ///   computed literally (the port performs the equivalent row copies
    ///   host-side — see the module header).
    #[allow(clippy::too_many_arguments)]
    pub fn build_seq(
        pos: &[i32],
        seq_ids: &[i32],
        ratio: u32,
        overlap: bool,
        state_size: u32,
        kv_size: u32,
        n_stream: u32,
        n_rs_seq: u32,
        rs_idx: &[u32],
    ) -> Dsv4Plan {
        let n_tokens = pos.len();
        assert_eq!(seq_ids.len(), n_tokens);
        // :442-444 — a unified stream cannot serve multiple sequences
        assert!(
            n_stream > 1
                || seq_ids
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    <= 1,
            "DSV4 single compressed stream cannot serve multiple sequences"
        );
        let state_rows = state_size as i64 * n_stream as i64; // :446

        // the port's gather layout: [plane set | scratch | zero-row], so the
        // scratch follows the (1 + n_rs_seq) plane groups (see the doc comment)
        let scratch_off = state_rows * (1 + n_rs_seq as i64);

        let mut plan = Dsv4Plan {
            n_visible: vec![0; n_tokens],
            n_stream: 1, // dsv4_comp_graph_n_stream (:196-204) — one seq per ubatch
            n_kv: 0,
            ..Dsv4Plan::default()
        };

        // :464-471 — (seq, pos) -> token idx (last wins)
        let mut curr_token_idx: std::collections::HashMap<(i32, i32), usize> =
            std::collections::HashMap::with_capacity(n_tokens);
        for (i, (&p, &s)) in pos.iter().zip(seq_ids).enumerate() {
            curr_token_idx.insert((s, p), i);
        }

        // :473-488
        let state_source_idx = |seq_id: i32, p: i32| -> i32 {
            if p < 0 {
                // the graph-appended zero/-inf row after the scratch
                return (scratch_off + n_tokens as i64) as i32;
            }
            if let Some(&i) = curr_token_idx.get(&(seq_id, p)) {
                return (scratch_off + i as i64) as i32;
            }
            let stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            (stream_off + p as i64 % state_size as i64) as i32
        };

        // the unique sequences of this ubatch, first-appearance order
        // (`ubatch.seq_id_unq`)
        let mut seq_id_unq: Vec<i32> = Vec::new();
        for &s in seq_ids {
            if !seq_id_unq.contains(&s) {
                seq_id_unq.push(s);
            }
        }

        // :448-453 persist rows: one per dst, keeping the max-pos source
        let mut persist: Vec<(i32 /*dst*/, i32 /*src*/, i32 /*pos*/)> = Vec::new();
        // :465 — completed blocks per sequence
        let mut state_write_counts: std::collections::HashMap<i32, u32> =
            std::collections::HashMap::new();

        let mut overlap_prev_reads: Vec<i32> = Vec::new();
        let mut overlap_cur_reads: Vec<i32> = Vec::new();

        for (i, &p) in pos.iter().enumerate() {
            if p < 0 {
                continue; // :493-495
            }
            plan.state_pos.push(p % ratio as i32); // :497

            let n_visible = (p as i64 + 1) / ratio as i64; // :499
            plan.n_visible[i] = n_visible as i32;
            plan.n_kv = plan.n_kv.max(n_visible as u32); // :501

            // one sequence per token (the per-seq ubatch slice of :503-504's
            // `for s in 0..ubatch.n_seq_id[i]`)
            let seq_id = seq_ids[i];
            let stream_off = dsv4_stream_offset(n_stream, seq_id, state_size); // :505
            let state_idx = (stream_off + p as i64 % state_size as i64) as i32; // :506

            match persist.iter_mut().find(|r| r.0 == state_idx) {
                None => persist.push((state_idx, i as i32, p)), // :513
                Some(r) => {
                    if p > r.2 {
                        r.1 = i as i32; // :515-516
                        r.2 = p;
                    }
                }
            }

            if (p + 1) % ratio as i32 != 0 {
                continue; // :519-521
            }

            let source_start = p + 1 - ratio as i32; // :523
            let cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size); // :524

            plan.state_write_idxs
                .push(cache_off + p as i64 / ratio as i64); // :526
            plan.state_write_pos.push(source_start); // :527
            *state_write_counts.entry(seq_id).or_insert(0) += 1; // :528

            if overlap {
                let prev_start = source_start - ratio as i32; // :531
                for j in 0..ratio as i32 {
                    overlap_prev_reads.push(state_source_idx(seq_id, prev_start + j));
                    // :534
                }
                for j in 0..ratio as i32 {
                    overlap_cur_reads.push(state_source_idx(seq_id, source_start + j));
                    // :537
                }
            } else {
                for j in 0..ratio as i32 {
                    plan.state_read_idxs
                        .push(state_source_idx(seq_id, source_start + j)); // :541
                }
            }
        }

        // :547-600 — the CSA dummy block pads each sequence's writes to the
        // reserve plan's block count. The port's ubatches hold one sequence
        // (never coupled), so the `dsv4_ubatch_has_coupled` branch (:570-578)
        // is unreachable and the per-seq branch runs for every seq
        if ratio == DSV4_CSA_RATIO && !plan.state_pos.is_empty() {
            assert!(kv_size > 0);

            for &seq_id in &seq_id_unq {
                // :580 — blocks of this seq's tokens
                let n_seq_tokens = seq_ids.iter().filter(|&&s| s == seq_id).count().max(1) as u32;
                let n_blocks = n_seq_tokens.div_ceil(ratio);
                let n_writes = state_write_counts.get(&seq_id).copied().unwrap_or(0);
                if n_writes >= n_blocks {
                    continue; // :585-587
                }
                assert!(
                    n_writes + 1 == n_blocks,
                    "DSV4 CSA sequence positions are not contiguous" // :588-590
                );

                // :592-596 — the seq's first pos >= 0 token
                // (`dsv4_token_has_seq`)
                let i = (0..n_tokens)
                    .find(|&i| pos[i] >= 0 && seq_ids[i] == seq_id)
                    .expect("CSA dummy source token");
                let source_idx = state_source_idx(seq_id, pos[i]); // :553

                let cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size); // :552
                plan.state_write_idxs.push(cache_off + kv_size as i64 - 1); // :555
                plan.state_write_pos.push(0); // :556
                if overlap {
                    for _ in 0..ratio {
                        overlap_prev_reads.push(source_idx); // :560
                        overlap_cur_reads.push(source_idx); // :561
                    }
                } else {
                    for _ in 0..ratio {
                        plan.state_read_idxs.push(source_idx); // :565
                    }
                }
            }
        }

        // :602-627 — the HCA dummy block (the first pos >= 0 token's stream)
        if ratio == DSV4_HCA_RATIO && !plan.state_pos.is_empty() && plan.state_write_idxs.is_empty()
        {
            assert!(kv_size > 0);
            assert!(plan.n_kv < kv_size as i64 as u32);
            let i = pos
                .iter()
                .position(|&p| p >= 0)
                .expect("HCA dummy source token");
            let seq_id = seq_ids[i]; // :617
            let cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size); // :618
            let source_idx = state_source_idx(seq_id, pos[i]); // :619

            plan.state_write_idxs.push(cache_off + kv_size as i64 - 1); // :621
            plan.state_write_pos.push(0); // :622
            for _ in 0..ratio {
                plan.state_read_idxs.push(source_idx); // :625
            }
        }

        if overlap {
            // :629-636 — [all prev | all cur]
            plan.state_read_idxs = overlap_prev_reads;
            plan.state_read_idxs.extend(overlap_cur_reads);
        }

        // :641 — keep the compressed branch's graph topology fixed
        plan.n_kv = plan.n_kv.div_ceil(256).max(1) * 256;

        // :643-651 — dst-ordered persist (dsts are unique)
        persist.sort_by_key(|r| r.0);
        for r in persist {
            plan.state_persist_src_idxs.push(r.1);
            plan.state_persist_dst_idxs.push(r.0);
        }

        // :653-716 — restore/snapshot entries for every layout stream so the
        // (C's) graph tensor sizes do not depend on the ubatch's sequence
        // count. The port applies the same row copies host-side
        // (`Dsv4CompState::restore_seq` / `snapshot_shift_seq`); the vectors
        // are kept literal for documentation and unit tests.
        if n_rs_seq > 0 {
            for stream in 0..n_stream {
                // :658-669 — the seq served by this stream (a unified stream
                // serves any single sequence, :659-661)
                let seq_id = if n_stream == 1 {
                    seq_id_unq.first().copied().unwrap_or(-1)
                } else if seq_id_unq.contains(&(stream as i32)) {
                    stream as i32
                } else {
                    -1
                };

                let stream_off = dsv4_stream_offset(n_stream, seq_id.max(0), state_size); // :671
                let rollback = if seq_id >= 0 && (seq_id as usize) < rs_idx.len() {
                    rs_idx[seq_id as usize]
                } else {
                    0 // :672
                };
                // :674 — the restore source plane (0 when no rollback pends)
                let src_plane = if rollback > 0 && rollback <= n_rs_seq {
                    rollback as i64 * state_rows
                } else {
                    0
                };
                for r in 0..state_size as i64 {
                    plan.state_restore_src_idxs
                        .push((src_plane + stream_off + r) as i32); // :676
                    plan.state_restore_dst_idxs.push((stream_off + r) as i32); // :677
                }

                // :680-688 — this seq's token indices
                let token_idxs: Vec<usize> = if seq_id >= 0 {
                    (0..n_tokens).filter(|&i| seq_ids[i] == seq_id).collect()
                } else {
                    Vec::new()
                };

                let n_seq_tokens = token_idxs.len();
                for d in 1..=n_rs_seq as i64 {
                    let dst_plane = d * state_rows; // :693
                    let prefix = if (d as usize) <= n_seq_tokens {
                        n_seq_tokens - d as usize
                    } else {
                        0 // :694
                    };

                    for r in 0..state_size as i64 {
                        let mut src = (stream_off + r) as i32; // :697

                        for &i_tok in token_idxs.iter().take(prefix) {
                            // :699-704
                            if pos[i_tok] >= 0 && pos[i_tok] % state_size as i32 == r as i32 {
                                src = (scratch_off + i_tok as i64) as i32;
                            }
                        }

                        if n_seq_tokens == 0 {
                            // :706-708 — no-op: copy the snapshot plane onto itself
                            src = (dst_plane + stream_off + r) as i32;
                        }

                        plan.state_snapshot_src_idxs.push(src); // :711
                        plan.state_snapshot_dst_idxs
                            .push((dst_plane + stream_off + r) as i32); // :712
                    }
                }
            }
        }

        plan
    }
}

/// One compressor's per-step graph inputs — `llm_graph_input_dsv4::comp_input`
/// (llama-graph.h:600-620) as built by `dsv4_build_comp_inputs`
/// (llama-graph.cpp:953-981): 1-D I32 index tensors (state_write_idxs I64) and
/// the `[plan.n_kv, n_tokens/n_stream, 1, n_stream]` kq_mask, zeroed for the
/// first `n_visible[i]` rows per query (dsv4_set_kq_mask,
/// llama-graph.cpp:760-803). `k_rot` is the cache's Hadamard input
/// (`build_input_k_rot`, llama-kv-cache.cpp:1437-1455) when `attn_rot_k` is
/// on (always for lid, model-wide when `n_embd_head_k_full ==
/// indexer_head_size`, llama-kv-cache.cpp:321-332).
pub struct Dsv4CompInputs {
    pub state_pos: TensorId,
    pub state_persist_src_idxs: TensorId,
    pub state_persist_dst_idxs: TensorId,
    pub state_read_idxs: TensorId,
    pub state_write_idxs: TensorId,
    pub state_write_pos: TensorId,
    pub kq_mask: TensorId,
    pub k_rot: Option<TensorId>,
}

/// `llama_kv_cache_dsv4`'s compressed half, plus the per-layer compress
/// ratios and the raw cache's `attn_rot_k` fact (the raw iswa pair itself is
/// the port's existing `KvCache::swa`/`swa_step`).
pub struct Dsv4Cache {
    pub csa: Dsv4CompCache,
    pub hca: Dsv4CompCache,
    pub lid: Dsv4CompCache,
    pub csa_state: Dsv4CompState,
    pub hca_state: Dsv4CompState,
    pub lid_state: Dsv4CompState,
    /// `hparams.dsv4_compress_ratios` (per model layer: 0 raw / 4 csa / 128 hca)
    pub ratios: Vec<u32>,
    /// nrot of the raw cache's Hadamard (`build_input_k_rot`); the comp caches
    /// derive theirs from their own n_embd_head
    pub raw_k_rot: bool,
    /// `n_stream = n_seq_max` of the compressed caches/states
    /// (llama-kv-cache-dsv4.cpp:1229/:1287 — forced per-sequence)
    pub n_stream: u32,
    /// `n_rs_seq` (llama-context.cpp:3700-3701 — 0 unless speculative
    /// decoding requested rollback planes)
    pub n_rs_seq: u32,
    /// `rs_idx` (:1231, `n_seq_max` wide) — the pending per-sequence rollback
    /// depth set by `seq_rm` (:1497) and consumed by the next step's plans
    /// (`dsv4_build_comp_plans` :751-762 / `reset_rs_idx_for_ubatches`
    /// :1711-1726)
    pub rs_idx: Vec<u32>,
}

impl Dsv4Cache {
    /// `reset_rs_idx_for_ubatches` (:1711-1726) — the first step touching a
    /// sequence consumed its rollback
    pub fn reset_rs_idx_for(&mut self, seq_ids: &[i32]) {
        if self.n_rs_seq == 0 {
            return;
        }
        for &s in seq_ids {
            if s >= 0 && (s as usize) < self.rs_idx.len() {
                self.rs_idx[s as usize] = 0;
            }
        }
    }
}

/// per-step plans + inputs of the three compressors
/// (`llm_graph_input_dsv4::set_input`, llama-graph.cpp:997-1013)
pub struct Dsv4Step {
    pub plans: [Dsv4Plan; 3],
    pub csa: Dsv4CompInputs,
    pub hca: Dsv4CompInputs,
    pub lid: Dsv4CompInputs,
    /// the raw iswa cache's Hadamard rotation input (`inp_raw->self_k_rot`,
    /// llama-graph.cpp:3468) — None when the raw cache has no attn_rot_k
    pub raw_k_rot: Option<TensorId>,
}

impl Dsv4Step {
    pub fn plan_csa(&self) -> &Dsv4Plan {
        &self.plans[0]
    }
    pub fn plan_hca(&self) -> &Dsv4Plan {
        &self.plans[1]
    }
    pub fn plan_lid(&self) -> &Dsv4Plan {
        &self.plans[2]
    }
}

impl KvCache {
    /// `llama_kv_cache_dsv4` (llama-kv-cache-dsv4.cpp:1210-1336): the raw
    /// iswa pair (every deepseek4 trunk layer is SWA, `set_swa_pattern(0)` +
    /// the loader's per-layer flags, deepseek4.cpp:67-74) plus the three
    /// compressed caches and the three compressor states. `kv_size` is
    /// `cparams.n_ctx_seq`; the compressed sizes follow
    /// `GGML_PAD(dsv4_comp_size(kv_size, ratio), 256)` (:1294/:1302/:1310).
    ///
    /// `n_stream` is `cparams.n_seq_max` (the C's forced-per-sequence
    /// compressed layout, :1287) and `n_rs_seq` the rollback-plane count
    /// (:1222/:1230-1231) — `1, 0` reproduces the original single-sequence
    /// configuration exactly.
    ///
    /// `raw_k_rot` follows llama-kv-cache.cpp:321-332 (the DeepSeek-indexer
    /// special case): on when `n_embd_head_k == indexer_head_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dsv4(
        ctx: &mut Context,
        k_row: &[i64],
        v_row: &[i64],
        size_base: u32,
        spec: &SwaCacheSpec,
        n_ubatch: u32,
        ratios: &[u32],
        n_embd_head_k: i64,
        indexer_head_size: i64,
        n_stream: u32,
        n_rs_seq: u32,
    ) -> Self {
        let mut cache = Self::new_swa(ctx, k_row, v_row, size_base, spec, n_ubatch);

        let mk_comp = |ctx: &mut Context, ratio: u32, n_embd_head: i64| -> Dsv4CompCache {
            let size = dsv4_comp_size(size_base, ratio).div_ceil(256).max(1) * 256;
            let mut layer_ids = Vec::new();
            let mut k = Vec::new();
            for (il, &r) in ratios.iter().enumerate() {
                if r != ratio {
                    continue;
                }
                // [n_embd_head, size] per stream (the C's 3-D [., size, n_stream])
                let t =
                    ctx.new_tensor_2d(GgmlType::F16, n_embd_head, size as i64 * n_stream as i64);
                ctx.arena_resize_tensor(t);
                layer_ids.push(il);
                k.push(t);
            }
            Dsv4CompCache {
                layer_ids,
                k,
                size,
                n_embd_head,
                n_stream,
                cur_stream: 0,
            }
        };

        let mk_state =
            |ctx: &mut Context, ratio: u32, state_size: u32, n_embd_state: i64| -> Dsv4CompState {
                let mut layer_ids = Vec::new();
                let mut kv = Vec::new();
                let mut score = Vec::new();
                for (il, &r) in ratios.iter().enumerate() {
                    if r != ratio {
                        continue;
                    }
                    // one [n_embd_state, state_size] plane per (snapshot, stream)
                    // group — `n_planes = n_stream*(1 + n_rs_seq)` (:964-966)
                    let n_planes = n_stream * (1 + n_rs_seq);
                    let kv_t = ctx.new_tensor_2d(
                        GgmlType::F32,
                        n_embd_state,
                        state_size as i64 * n_planes as i64,
                    );
                    let sc_t = ctx.new_tensor_2d(
                        GgmlType::F32,
                        n_embd_state,
                        state_size as i64 * n_planes as i64,
                    );
                    ctx.arena_resize_tensor(kv_t);
                    ctx.arena_resize_tensor(sc_t);
                    layer_ids.push(il);
                    kv.push(kv_t);
                    score.push(sc_t);
                }
                Dsv4CompState {
                    layer_ids,
                    kv,
                    score,
                    ratio,
                    state_size,
                    n_embd_state,
                    n_stream,
                    n_rs_seq,
                }
            };

        let csa = mk_comp(ctx, DSV4_CSA_RATIO, n_embd_head_k);
        let hca = mk_comp(ctx, DSV4_HCA_RATIO, n_embd_head_k);
        let lid = mk_comp(ctx, DSV4_CSA_RATIO, indexer_head_size);

        // llama-kv-cache-dsv4.cpp:1315-1329 — state_size 2*ratio for the
        // overlap compressors (csa/lid), ratio for hca
        let csa_state = mk_state(ctx, DSV4_CSA_RATIO, 2 * DSV4_CSA_RATIO, 2 * n_embd_head_k);
        let hca_state = mk_state(ctx, DSV4_HCA_RATIO, DSV4_HCA_RATIO, n_embd_head_k);
        let lid_state = mk_state(
            ctx,
            DSV4_CSA_RATIO,
            2 * DSV4_CSA_RATIO,
            2 * indexer_head_size,
        );

        cache.dsv4 = Some(Dsv4Cache {
            csa,
            hca,
            lid,
            csa_state,
            hca_state,
            lid_state,
            ratios: ratios.to_vec(),
            raw_k_rot: n_embd_head_k == indexer_head_size,
            n_stream,
            n_rs_seq,
            rs_idx: vec![0; n_stream as usize],
        });
        // `n_rot_k` (210791069, llama-kv-cache.cpp:334-341): the
        // DeepSeek-indexer clause above is the only way the port's F16
        // planes rotate, and the width is the largest power of two >= 64
        // dividing `n_embd_head_k_all` — `raw_k_rot_nrot` (the same loop the
        // old build-time do-while computed, now fixed at construction so the
        // state blob can persist it)
        if n_embd_head_k == indexer_head_size {
            cache.n_rot_k = raw_k_rot_nrot(n_embd_head_k);
        }
        cache
    }

    /// `clear_compressed(-1, true)` (llama-kv-cache-dsv4.cpp:1728-1761) — the
    /// compressed buffers must never expose stale/uninit rows; also zeroes
    /// the rs rollback bookkeeping (:1756-1760)
    pub fn clear_dsv4(&mut self, ctx: &mut Context) {
        if let Some(d) = self.dsv4.as_mut() {
            d.csa.clear(ctx);
            d.hca.clear(ctx);
            d.lid.clear(ctx);
            d.csa_state.clear(ctx);
            d.hca_state.clear(ctx);
            d.lid_state.clear(ctx);
            d.rs_idx.fill(0);
        }
        self.dsv4_step = None;
    }

    /// `clear_compressed(seq_id, true)` (:1733-1754) — one sequence's
    /// compressed caches (row block) and state plane groups, plus
    /// `rs_idx[seq_id] = 0` (:1756-1757)
    pub fn clear_dsv4_seq(&mut self, ctx: &mut Context, seq: u32) {
        if let Some(d) = self.dsv4.as_mut() {
            d.csa.clear_seq(ctx, seq);
            d.hca.clear_seq(ctx, seq);
            d.lid.clear_seq(ctx, seq);
            d.csa_state.clear_seq(ctx, seq);
            d.hca_state.clear_seq(ctx, seq);
            d.lid_state.clear_seq(ctx, seq);
            if (seq as usize) < d.rs_idx.len() {
                d.rs_idx[seq as usize] = 0;
            }
        }
    }

    /// `llama_kv_cache_dsv4::seq_cp`'s compressed half (llama-kv-cache-dsv4.
    /// cpp:1516-1522): full copies of the three compressed K caches and the
    /// three compressor states (their `seq_cp` + deferred `apply_copies`),
    /// then `rs_idx[seq_id_dst] = 0` (:1524-1526). Requires
    /// `p0 <= 0 && p1 < 0` — "DSV4 only supports full sequence copies"
    /// (:1513).
    pub fn seq_cp_dsv4(&mut self, ctx: &mut Context, seq_id_src: u32, seq_id_dst: u32) {
        let Some(d) = self.dsv4.as_mut() else {
            return;
        };
        assert!(
            (seq_id_src as usize) < d.rs_idx.len() && (seq_id_dst as usize) < d.rs_idx.len(),
            "DSV4 stream out of range"
        );
        d.csa.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.hca.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.lid.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.csa_state.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.hca_state.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.lid_state.seq_cp(ctx, seq_id_src, seq_id_dst);
        d.rs_idx[seq_id_dst as usize] = 0;
    }
}

// ===========================================================================
// arch batch 6 (2026-09-24), deepseek32: the DSA lightning-indexer key cache
// (`llama_kv_cache_dsa`, llama-kv-cache-dsa.cpp:14-54)
// ===========================================================================

/// The `kv_lid` half of `llama_kv_cache_dsa` — a *second* plain
/// `llama_kv_cache` the C builds from a hand-tweaked hparams clone
/// (llama-kv-cache-dsa.cpp:38-53: `n_head_kv = 1`,
/// `n_embd_head_k_full = indexer_head_size`, rope NEOX) so it allocates
/// indexer-key rows `[indexer_head_size, size]`. K-only, and it always
/// receives the same ubatches as the MLA cache, so its cells stay in lockstep
/// with the base cache's (the C runs the identical `prepare` on both,
/// llama-kv-cache-dsa.cpp:131-139).
pub struct KvLidCache {
    pub cells: Vec<KvCell>,
    pub size: u32,
    pub head: u32,
    /// per model layer, the indexer key tensor `[indexer_head_k, size]` F16
    pub layers: Vec<TensorId>,
    /// `hparams_lid.n_embd_head_k_full` (= `hparams.indexer_head_size`)
    pub n_embd_head_k: i64,
    /// the precomputed Walsh-Hadamard matrix of `k_rot` (`attn_rot_hadamard`,
    /// llama-kv-cache.cpp:344-362 + the local `ggml_gen_hadamard` :23-53) —
    /// row-major `[nrot, nrot]`
    pub hadamard: Vec<f32>,
    /// `build_input_k_rot`'s nrot (llama-kv-cache.cpp:1437-1455): the largest
    /// power of two ≥ 64 that divides `indexer_head_size`
    pub nrot: usize,
}

/// Per-step graph inputs of the lid cache (`llm_graph_input_attn_k_dsa::
/// set_input`, llama-graph.cpp:558-574): its own scatter indices, its own
/// (always F16, `cparams_copy.flash_attn = cparams.fused_lid` forces it,
/// llama-graph.cpp:3325-3327) causal mask, and the Hadamard rotation matrix
/// input `self_k_rot_lid`.
#[derive(Clone, Copy, Debug)]
pub struct KvLidStep {
    /// `self_k_idxs_lid`
    pub row_idx: TensorId,
    /// `self_kq_mask_lid` — F16
    pub kq_mask: TensorId,
    /// `self_k_rot_lid` — F32 `[nrot, nrot]`, filled from `KvLidCache::hadamard`
    pub k_rot: TensorId,
}

/// The local `ggml_gen_hadamard` of llama-kv-cache.cpp:23-53 — the
/// orthonormal Walsh-Hadamard rotation matrix (res² == I), row-major.
pub fn gen_hadamard(n: usize) -> Vec<f32> {
    assert!(
        n.is_power_of_two(),
        "gen_hadamard: n must be a power of two"
    );
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut data = vec![0.0f32; n * n];
    data[0] = scale;
    let mut s = 1usize;
    while s < n {
        for i in 0..s {
            for j in 0..s {
                let val = data[i * n + j];
                data[(i + s) * n + j] = val;
                data[i * n + (j + s)] = val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
        s *= 2;
    }
    data
}

/// `build_input_k_rot`'s nrot derivation (llama-kv-cache.cpp:1441-1448):
/// `nrot = 64; do nrot *= 2 while head % nrot == 0; nrot /= 2` — the largest
/// power of two ≥ 64 dividing `head`.
pub fn hadamard_nrot(head: i64) -> usize {
    let mut nrot = 64usize;
    loop {
        let next = nrot * 2;
        if head % next as i64 == 0 {
            nrot = next;
        } else {
            break;
        }
    }
    nrot
}

/// `n_rot_k`'s construction-time derivation (210791069,
/// llama-kv-cache.cpp:334-341): `n_rot = 64; while head % (2*n_rot) == 0
/// { n_rot *= 2 }`. Equivalent to the old graph-time do-while of
/// `build_input_k_rot` (`nrot *= 2 do-while head % nrot == 0; nrot /= 2` —
/// both stop at the largest power of two >= 64 dividing `head`, e.g. 192→64,
/// 128→128, 256→256), which is exactly [`hadamard_nrot`]; kept as its own
/// 1:1 twin because the C now computes it once in the constructor and
/// persists it in the state blob.
pub fn raw_k_rot_nrot(n_embd_head_k_all: i64) -> u32 {
    let mut n_rot = 64u32;
    while n_embd_head_k_all % (2 * n_rot as i64) == 0 {
        n_rot *= 2;
    }
    n_rot
}

/// `llama_kv_cache_iswa`'s SWA cache size (llama-kv-cache-iswa.cpp:69-81):
/// `size_swa = swa_full ? size_base : GGML_PAD(std::min(size_base,
/// n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)`.
///
/// `n_ubatch` is `cparams.n_ubatch` (llama-model.cpp:2481) — the port's
/// `DecodeContext::n_batch`, the largest decode call the driver makes.
pub fn swa_cache_size(
    size_base: u32,
    n_swa: u32,
    n_seq_max: u32,
    n_ubatch: u32,
    unified: bool,
    swa_full: bool,
) -> u32 {
    if swa_full {
        return size_base;
    }
    let want = n_swa * if unified { n_seq_max } else { 1 } + n_ubatch;
    let size = want.min(size_base);
    // GGML_PAD(x, 256)
    size.div_ceil(256) * 256
}

pub struct KvCache {
    /// Per model layer, the k/v tensor of whichever cache holds that layer.
    /// Uniform for the archs whose layers share one attention geometry; per
    /// layer for gemma4 (SWA layers 256*8, dense layers 512*1).
    pub layers: Vec<KvLayer>,
    /// base cache cells (`llama_kv_cache::v_cells`, llama-kv-cache.h:301-303)
    pub cells: Vec<KvCell>,
    /// total cache capacity (tokens)
    pub size: u32,
    pub head: u32,
    /// uniform row width — equals every entry of `k_row` for the archs whose
    /// layers share one attention geometry (all but gemma4)
    pub n_embd_k_gqa: i64,
    pub n_embd_v_gqa: i64,
    /// per-layer row widths. llama-kv-cache.cpp:210-211 allocates
    /// `n_embd_k_gqa(il)` / `n_embd_v_gqa(il)` rows per layer (`!v_trans`), which
    /// is exactly per-layer for gemma4 (SWA layers 256*8, full layers 512*1).
    pub k_row: Vec<i64>,
    pub v_row: Vec<i64>,
    /// The `iswa` split's second cache; None for non-SWA models. Set by
    /// `new_swa`, which is what `llama-model.cpp:2687` selects on
    /// (`hparams.swa_type != LLAMA_SWA_TYPE_NONE`).
    pub swa: Option<KvSwaCache>,
    /// `hparams.is_swa(il)` — the per-layer cache selection of the graph
    /// (llama-graph.cpp:3131-3133 `mctx_cur = is_swa ? mctx_iswa->get_swa() :
    /// mctx_iswa->get_base()`). Empty for non-SWA models.
    pub is_swa: Vec<bool>,
    /// per-step inputs of the SWA side, set by the driver before building a
    /// graph (`llm_graph_input_attn_kv_iswa::set_input` equivalent)
    pub swa_step: Option<KvSwaStep>,
    /// The `llama_kv_cache_dsa` pair's indexer-key cache (deepseek32);
    /// None for every other arch. Selected on `model.arch ==
    /// LLM_ARCH_DEEPSEEK32` (llama-model.cpp:2351-2369).
    pub lid: Option<KvLidCache>,
    /// per-step inputs of the lid cache (`self_k_idxs_lid` /
    /// `self_kq_mask_lid` / `self_k_rot_lid`)
    pub lid_step: Option<KvLidStep>,
    /// The `llama_kv_cache_dsv4` compressed half (deepseek4) — the three
    /// compressed K caches + the three compressor states. Selected on
    /// `model.arch == LLM_ARCH_DEEPSEEK4` (llama-model.cpp:2491-2507); the
    /// raw iswa pair of the same C class is this cache's own `swa`/`is_swa`.
    pub dsv4: Option<Dsv4Cache>,
    /// `has_v = !is_mla()` (llama-kv-cache.cpp:230-234): glm5-next's attn
    /// half of `llama_memory_hybrid_idx` is a K-only MLA cache — its state
    /// blob carries K rows only (the dsa/dsv4 paths derive the same fact
    /// from their own members). The port still materializes a V plane for
    /// shape uniformity; nothing reads or serializes it.
    pub k_only: bool,
    /// `n_rot_k` (210791069, llama-kv-cache.h:281): the construction-time
    /// Hadamard rotation width of the K plane — 0 = no rotation. The C's
    /// `attn_rot_k` bool became this exact width so the state blob can
    /// reject a mismatched restore ("incompatible key rotation"). The port's
    /// planes are F16 (never `ggml_is_quantized`), so only the DeepSeek
    /// lightning-indexer clause of llama-kv-cache.cpp:321-332 ever turns it
    /// on (set by `new_dsv4`).
    pub n_rot_k: u32,
    /// `n_rot_v` (llama-kv-cache.h:282) — the V twin. Always 0 in the port
    /// (F16 planes never satisfy the quantized-type gate).
    pub n_rot_v: u32,
    /// per-step plans + inputs of the three compressors
    /// (`llm_graph_input_dsv4::set_input`)
    pub dsv4_step: Option<Dsv4Step>,
}

impl KvCache {
    /// `llama_kv_cache::size_k_bytes` (llama-kv-cache.cpp:1904-1912) — the
    /// sum of every layer's k tensor. `layers[il]` resolves per model layer,
    /// so an iswa cache's SWA half is included in the same sum.
    pub fn size_k_bytes(&self, ctx: &Context) -> u64 {
        self.layers.iter().map(|l| ctx.nbytes(l.k) as u64).sum()
    }

    /// `llama_kv_cache::size_v_bytes` (llama-kv-cache.cpp:1914-1922).
    pub fn size_v_bytes(&self, ctx: &Context) -> u64 {
        self.layers.iter().map(|l| ctx.nbytes(l.v) as u64).sum()
    }

    /// `llama_kv_cache::total_size` (llama-kv-cache.cpp:1894-1901): every
    /// backing buffer of the cache tree — the k/v planes (both iswa halves),
    /// the dsa lid indexer keys and the dsv4 compressed half's three caches
    /// + compressor states.
    pub fn total_size(&self, ctx: &Context) -> u64 {
        let mut size = self.size_k_bytes(ctx) + self.size_v_bytes(ctx);
        if let Some(lid) = &self.lid {
            size += lid.layers.iter().map(|&k| ctx.nbytes(k) as u64).sum::<u64>();
        }
        if let Some(d) = &self.dsv4 {
            for c in [&d.csa, &d.hca, &d.lid] {
                size += c.k.iter().map(|&k| ctx.nbytes(k) as u64).sum::<u64>();
            }
            for st in [&d.csa_state, &d.hca_state, &d.lid_state] {
                size += st
                    .kv
                    .iter()
                    .chain(st.score.iter())
                    .map(|&t| ctx.nbytes(t) as u64)
                    .sum::<u64>();
            }
        }
        size
    }

    /// `llama_kv_cache::memory_breakdown` (llama-kv-cache.cpp:685) — the
    /// single-buffer-type port folds into one entry: (k, v) byte split.
    pub fn memory_breakdown(&self, ctx: &Context) -> (u64, u64) {
        (self.size_k_bytes(ctx), self.size_v_bytes(ctx))
    }

    pub fn new(
        ctx: &mut Context,
        n_layer: usize,
        n_embd_k_gqa: i64,
        n_embd_v_gqa: i64,
        size: u32,
    ) -> Self {
        Self::new_with_dims(
            ctx,
            &vec![n_embd_k_gqa; n_layer],
            &vec![n_embd_v_gqa; n_layer],
            size,
        )
    }

    /// Per-layer geometry variant (gemma4: `hparams.is_n_embd_k_gqa_variable`).
    pub fn new_with_dims(ctx: &mut Context, k_row: &[i64], v_row: &[i64], size: u32) -> Self {
        assert_eq!(
            k_row.len(),
            v_row.len(),
            "KvCache: per-layer dims must pair up"
        );
        let mut layers = Vec::with_capacity(k_row.len());
        for (&nk, &nv) in k_row.iter().zip(v_row) {
            let k = ctx.new_tensor_2d(GgmlType::F16, nk, size as i64);
            let v = ctx.new_tensor_2d(GgmlType::F16, nv, size as i64);
            ctx.arena_resize_tensor(k);
            ctx.arena_resize_tensor(v);
            layers.push(KvLayer { k, v });
        }
        KvCache {
            layers,
            cells: vec![KvCell::default(); size as usize],
            size,
            head: 0,
            n_embd_k_gqa: k_row[0],
            n_embd_v_gqa: v_row[0],
            k_row: k_row.to_vec(),
            v_row: v_row.to_vec(),
            swa: None,
            is_swa: Vec::new(),
            swa_step: None,
            lid: None,
            lid_step: None,
            dsv4: None,
            dsv4_step: None,
            k_only: false,
            n_rot_k: 0,
            n_rot_v: 0,
        }
    }

    /// `llama_kv_cache_iswa` (llama-kv-cache-iswa.cpp:52-106): two caches whose
    /// layers are the two halves of the `hparams.is_swa(il)` partition. The
    /// base cache keeps `size_base` cells at the given per-layer widths, the
    /// SWA cache gets `swa_cache_size(...)` cells of the same per-layer widths
    /// (:95-105 passes the same `k_row`/`v_row` geometry to both, each cache
    /// filtering its own layers).
    ///
    /// `layers[il]` resolves per model layer, so a builder reads a SWA layer's
    /// tensor without knowing about the split. A layer that `hparams.has_kv(il)`
    /// excludes from *both* caches (the shared-KV layers of gemma4/3n) still
    /// gets a base-sized pair here — the C's `map_layer_ids` has no entry for
    /// it and its graph looks the K/V up through `layer_reuse_cb` instead, which
    /// the port does not implement (see PARITY.md).
    pub fn new_swa(
        ctx: &mut Context,
        k_row: &[i64],
        v_row: &[i64],
        size_base: u32,
        spec: &SwaCacheSpec,
        n_ubatch: u32,
    ) -> Self {
        assert_eq!(
            k_row.len(),
            v_row.len(),
            "KvCache: per-layer dims must pair up"
        );
        assert_eq!(
            spec.is_swa.len(),
            k_row.len(),
            "SwaCacheSpec::is_swa must have one entry per model layer"
        );
        // llama-model.cpp:2687-2690: the iswa cache is selected on
        // `hparams.swa_type != LLAMA_SWA_TYPE_NONE`, which asserts is_swa_any()
        assert!(
            spec.swa_type != LlamaSwaType::NONE && spec.is_swa.iter().any(|&s| s),
            "KvCache::new_swa requires swa_type != NONE and at least one SWA layer \
             (llama-model.cpp:2687-2690)"
        );

        let size_swa = swa_cache_size(
            size_base,
            spec.n_swa,
            spec.n_seq_max,
            n_ubatch,
            spec.unified,
            spec.swa_full,
        );

        let n_layer = k_row.len();
        let mut layers = Vec::with_capacity(n_layer);
        let mut layer_ids = Vec::new();
        for il in 0..n_layer {
            let (nk, nv) = (k_row[il], v_row[il]);
            // the layer's cache: `filter_swa`/`filter_base`
            // (llama-kv-cache-iswa.cpp:53-67)
            let size = if spec.is_swa[il] {
                layer_ids.push(il);
                size_swa
            } else {
                size_base
            };
            let k = ctx.new_tensor_2d(GgmlType::F16, nk, size as i64);
            let v = ctx.new_tensor_2d(GgmlType::F16, nv, size as i64);
            ctx.arena_resize_tensor(k);
            ctx.arena_resize_tensor(v);
            layers.push(KvLayer { k, v });
        }

        KvCache {
            layers,
            cells: vec![KvCell::default(); size_base as usize],
            size: size_base,
            head: 0,
            n_embd_k_gqa: k_row[0],
            n_embd_v_gqa: v_row[0],
            k_row: k_row.to_vec(),
            v_row: v_row.to_vec(),
            swa: Some(KvSwaCache {
                cells: vec![KvCell::default(); size_swa as usize],
                size: size_swa,
                head: 0,
                // the SWA cache's own mask parameters (llama-kv-cache-iswa.cpp:
                // 102-105 passes `hparams.n_swa, hparams.swa_type`)
                n_swa: spec.n_swa,
                swa_type: spec.swa_type,
                layer_ids,
            }),
            is_swa: spec.is_swa.clone(),
            swa_step: None,
            lid: None,
            lid_step: None,
            dsv4: None,
            dsv4_step: None,
            k_only: false,
            n_rot_k: 0,
            n_rot_v: 0,
        }
    }

    /// `hparams.is_swa(il)` (llama-hparams.cpp:289-295)
    pub fn layer_is_swa(&self, il: usize) -> bool {
        self.is_swa.get(il).copied().unwrap_or(false)
    }

    /// `llama_kv_cache_dsa` (llama-kv-cache-dsa.cpp:14-54): the MLA K-only
    /// base cache plus the indexer-key (`lid`) cache — a second plain cache
    /// of `[indexer_head_k, size]` F16 rows with its own cells, plus the
    /// precomputed Hadamard rotation matrix `k_rot` is derived from
    /// (`build_input_k_rot`, llama-kv-cache.cpp:1437-1455).
    pub fn new_dsa(
        ctx: &mut Context,
        k_row: &[i64],
        v_row: &[i64],
        indexer_head_k: i64,
        size: u32,
    ) -> Self {
        let mut cache = Self::new_with_dims(ctx, k_row, v_row, size);
        let nrot = hadamard_nrot(indexer_head_k);
        let mut layers = Vec::with_capacity(k_row.len());
        for _ in 0..k_row.len() {
            let k = ctx.new_tensor_2d(GgmlType::F16, indexer_head_k, size as i64);
            ctx.arena_resize_tensor(k);
            layers.push(k);
        }
        cache.lid = Some(KvLidCache {
            cells: vec![KvCell::default(); size as usize],
            size,
            head: 0,
            layers,
            n_embd_head_k: indexer_head_k,
            hadamard: gen_hadamard(nrot),
            nrot,
        });
        cache
    }

    /// `llama_kv_cache_dsa::seq_rm` (llama-kv-cache-dsa.cpp:61-68) — both
    /// halves; likewise `clear` (:56-59)
    pub fn n_kv_lid(&self) -> u32 {
        match &self.lid {
            Some(l) => Self::n_kv_of(&l.cells, l.size),
            None => 0,
        }
    }

    /// place the ubatch in the lid cache — the same `apply_ubatch` body as the
    /// base cache; the two caches see identical ubatches, so the cells stay
    /// in lockstep (llama-kv-cache-dsa.cpp:131-139 runs the same `prepare`)
    pub fn assign_lid(&mut self, sinfo: SlotInfo, pos: &[i32], seq_id: usize) {
        let Some(l) = self.lid.as_mut() else { return };
        let mask = 1u64 << seq_id;
        for (i, &p) in pos.iter().enumerate() {
            let cell = &mut l.cells[sinfo.s0 as usize + i];
            cell.pos = p;
            cell.seq = mask;
        }
        l.head = (sinfo.s1 + 1) % l.size;
    }

    /// the multi-sequence twin of [`KvCache::assign_lid`] for the batch
    /// driver (`decode_batch` → `step_ubatch`): the lid cache commits the
    /// *same* slot scan the base cache ran (`llama_kv_cache_dsa::prepare`
    /// forwards one find_slot to both halves, llama-kv-cache-dsa.cpp:131-139)
    pub fn apply_ubatch_lid(
        &mut self,
        idxs: &[u32],
        pos: &[i32],
        n_seq_id: &[i32],
        seq_id: &[Vec<i32>],
    ) {
        let Some(l) = self.lid.as_mut() else { return };
        for (i, &idx) in idxs.iter().enumerate() {
            let cell = &mut l.cells[idx as usize];
            *cell = KvCell::default();
            cell.pos = pos[i];
            for s in 0..n_seq_id[i] as usize {
                cell.seq |= 1u64 << seq_id[i][s];
            }
        }
        if let Some(&last) = idxs.last() {
            l.head = last + 1;
        }
    }

    /// llama_kv_cache::get_k over a lid layer — view [head_dim, 1, n_kv, 1]
    pub fn get_k_lid(&self, ctx: &mut Context, il: usize, n_kv: u32) -> TensorId {
        let l = self
            .lid
            .as_ref()
            .expect("get_k_lid without the dsa lid cache");
        let k = l.layers[il];
        let size = l.size as i64;
        let n_embd = l.n_embd_head_k;
        let rs_head = GgmlType::F16.row_size(n_embd as usize);
        let rs_gqa = GgmlType::F16.row_size(n_embd as usize); // n_head_kv == 1
        ctx.view_4d(
            k,
            n_embd,
            1,
            n_kv as i64,
            1,
            rs_head,
            rs_gqa,
            rs_gqa * size as usize,
            0,
        )
    }

    pub fn has_swa(&self) -> bool {
        self.swa.is_some()
    }

    /// the cache the given layer reads/writes (llama-graph.cpp:3133)
    pub fn swa_cache(&self) -> Option<&KvSwaCache> {
        self.swa.as_ref()
    }

    /// `v_cells[s].get_used()` — number of used cells (llama-kv-cells.h:85)
    fn used(cells: &[KvCell]) -> u32 {
        cells.iter().filter(|c| !c.is_empty()).count() as u32
    }

    /// `v_cells[s].used_max_p1()` — last used cell index + 1
    /// (llama-kv-cells.h:97-99)
    fn used_max_p1(cells: &[KvCell]) -> u32 {
        cells
            .iter()
            .rposition(|c| !c.is_empty())
            .map(|i| i as u32 + 1)
            .unwrap_or(0)
    }

    /// `v_cells.seq_pos_min(seq_id)` (llama-kv-cells.h:365-374), -1 if none.
    /// The C keeps a per-sequence position set; the port's cells carry `pos` +
    /// a sequence bitmask, so the set is derived.
    fn seq_pos_min(cells: &[KvCell], seq_mask: u64) -> i32 {
        cells
            .iter()
            .filter(|c| c.seq & seq_mask != 0)
            .map(|c| c.pos)
            .min()
            .unwrap_or(-1)
    }

    /// `v_cells.seq_pos_max(seq_id)` (llama-kv-cells.h:378-387), -1 if none
    fn seq_pos_max(cells: &[KvCell], seq_mask: u64) -> i32 {
        cells
            .iter()
            .filter(|c| c.seq & seq_mask != 0)
            .map(|c| c.pos)
            .max()
            .unwrap_or(-1)
    }

    /// `llama_memory_seq_pos_min(seq_id)` over the base cache
    /// (llama-kv-cache.cpp:2290) — what `batch.rs` validates positions against.
    pub fn seq_pos_min_of(&self, seq_id: usize) -> i32 {
        Self::seq_pos_min(&self.cells, 1u64 << seq_id)
    }

    /// `llama_memory_seq_pos_max(seq_id)` (llama-kv-cache.cpp:2305)
    pub fn seq_pos_max_of(&self, seq_id: usize) -> i32 {
        Self::seq_pos_max(&self.cells, 1u64 << seq_id)
    }

    /// `llama_kv_cells::seq_pos_get` (llama-kv-cells.h:321-328, def4d406a):
    /// the (pos, cell) pairs of sequence `seq_id`, ordered by position —
    /// the k-pool layout reads them (`kpool_layout_update`,
    /// llama-memory-hybrid-idx.cpp).
    pub fn seq_pos_get(&self, seq_id: i32) -> Vec<(i32, u32)> {
        assert!((0..crate::batch::LLAMA_MAX_SEQ as i32).contains(&seq_id));
        let mask = 1u64 << seq_id;
        let mut res: Vec<(i32, u32)> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty() && c.seq & mask != 0)
            .map(|(i, c)| (c.pos, i as u32))
            .collect();
        // the C returns `const std::set<std::pair<llama_pos, uint32_t>> &` —
        // sorted by (pos, cell)
        res.sort_unstable();
        res
    }

    /// `llama_kv_cells::seq_pos_tok_le` (llama-kv-cells.h:333-345): the
    /// token of the sequence's highest-position cell at or before `p` (the
    /// highest cell index wins ties); `LLAMA_TOKEN_NULL` (-1) when none.
    /// The PLE n-gram reads its predecessors through it.
    pub fn seq_pos_tok_le(&self, seq_id: i32, p: i32) -> i32 {
        assert!((0..crate::batch::LLAMA_MAX_SEQ as i32).contains(&seq_id));
        let mask = 1u64 << seq_id;
        // "when several cells share that position, the one with the highest
        // index wins" (:331-332 — the std::set's (pos, cell) ordering with
        // upper_bound({p, UINT32_MAX}) picks the last entry at pos <= p)
        let mut best: Option<(i32, u32)> = None;
        for (i, c) in self.cells.iter().enumerate() {
            if !c.is_empty() && c.seq & mask != 0 && c.pos <= p {
                match best {
                    None => best = Some((c.pos, i as u32)),
                    Some((bp, bi)) => {
                        if c.pos > bp || (c.pos == bp && i as u32 > bi) {
                            best = Some((c.pos, i as u32));
                        }
                    }
                }
            }
        }
        match best {
            Some((_, i)) => self.cells[i as usize].tok,
            None => -1,
        }
    }

    /// llama_kv_cache::find_slot — first-fit contiguous empty range
    ///
    /// note: the C scans from `v_heads[s]` in ring order with the heuristic of
    /// llama-kv-cache.cpp:996-999 and, for a non-SWA cache, only ever picks
    /// empty cells (:1044-1058 — `can_use` needs `is_empty` or a cell that
    /// `is_masked_swa` frees, which never happens with `swa_type == NONE`).
    /// For the port's append-only single sequence (nothing is ever removed
    /// mid-context) the ring scan and this first-fit from 0 select the same
    /// contiguous range, which is what the graph's `SlotInfo` needs.
    pub fn find_slot(&self, n_tokens: u32) -> Option<SlotInfo> {
        if n_tokens == 0 || n_tokens > self.size {
            return None;
        }
        for s0 in 0..=self.size - n_tokens {
            if self.cells[s0 as usize..(s0 + n_tokens) as usize]
                .iter()
                .all(|c| c.is_empty())
            {
                return Some(SlotInfo {
                    s0,
                    s1: s0 + n_tokens - 1,
                });
            }
        }
        None
    }

    /// `llama_kv_cache::find_slot` (llama-kv-cache.cpp:890-1094) for the SWA
    /// cache of the `iswa` pair: the same ring scan, but a cell that is already
    /// used is reusable once its position has fallen out of the window
    /// (:1044-1058):
    ///
    /// ```text
    /// can_use = cells.is_empty(idx)
    ///         | (cells.seq_count(idx) == 1
    ///            && is_masked_swa(n_swa, swa_type, pos_cell, seq_pos_max(seq_cell) + 1))
    /// ```
    ///
    /// `prepare` calls this with `cont = false` (llama-kv-cache.cpp:769), so
    /// `n_test = 1` and the result is the first `n_tokens` usable cells in ring
    /// order — not necessarily contiguous, hence `Vec<u32>` instead of
    /// `SlotInfo`.
    pub fn find_slot_swa(&self, n_tokens: u32) -> Option<Vec<u32>> {
        let swa = self.swa.as_ref()?;
        if n_tokens == 0 || n_tokens > swa.size {
            return None;
        }

        let mut head_cur = swa.head;
        // llama-kv-cache.cpp:996-999
        if head_cur > Self::used(&swa.cells) + 2 * n_tokens {
            head_cur = 0;
        }

        let mut n_tested = 0u32;
        let mut idxs: Vec<u32> = Vec::with_capacity(n_tokens as usize);

        loop {
            // llama-kv-cache.cpp:1016-1020 (n_test == 1 for cont == false)
            if head_cur >= swa.size {
                n_tested += swa.size - head_cur;
                head_cur = 0;
                continue;
            }

            let idx = head_cur;
            head_cur += 1;
            n_tested += 1;

            let cell = &swa.cells[idx as usize];
            let mut can_use = cell.is_empty();
            if !can_use && cell.seq_count() == 1 {
                // SWA mask: reuse a cell whose token can no longer be attended
                // (llama-kv-cache.cpp:1053-1058)
                if LlamaHparams::is_masked_swa(
                    swa.n_swa,
                    swa.swa_type,
                    cell.pos,
                    Self::seq_pos_max(&swa.cells, cell.seq) + 1,
                ) {
                    can_use = true;
                }
            }
            if can_use {
                idxs.push(idx);
            }

            if idxs.len() as u32 == n_tokens {
                break;
            }
            if n_tested >= swa.size {
                return None;
            }
        }

        Some(idxs)
    }

    /// `llama_kv_cache::get_n_kv` (llama-kv-cache.cpp:1250-1263):
    /// `min(cells.size(), max(n_pad, GGML_PAD(cells.used_max_p1(), n_pad)))`
    /// with `n_pad_cur = max(n_pad, 256)` — i.e. the cell-row count is padded
    /// to a multiple of **256** (`n_pad = 1`, llama-model.cpp:2482). The
    /// padding is NOT merely a graph-reuse optimization: the padded rows are
    /// part of the computed shapes — the non-FA softmax runs `n_kv` (padded)
    /// lanes, so its AVX512 `ggml_v_expf` chunks cover the *real* lanes
    /// whenever n_kv is a 256 multiple, and the KQ^T/KQ·V attention GEMMs see
    /// m = k = the padded count, which decides the llamafile tinyBLAS routing
    /// (k % 16). Padded cells read 0.0 (the arena is zeroed, like the C's
    /// `ggml_backend_buffer_clear(buf, 0)`) and are masked -inf
    /// (`cells.is_empty(j)` → skip), so they contribute exact zeros.
    pub fn n_kv(&self) -> u32 {
        Self::n_kv_of(&self.cells, self.size)
    }

    /// the unpadded `cells.used_max_p1()` — the raw used-cell count behind
    /// [`KvCache::n_kv`] (diagnostics/tests; the graph always consumes the
    /// padded value like the C's `mctx->get_n_kv()`)
    pub fn used_cells(&self) -> u32 {
        Self::used_max_p1(&self.cells)
    }

    /// the padded `min(size, max(256, GGML_PAD(used_max_p1, 256)))` of one
    /// cache — `n_pad = 1` ⇒ `n_pad_cur = max(n_pad, 256) = 256`
    fn n_kv_of(cells: &[KvCell], size: u32) -> u32 {
        let used = Self::used_max_p1(cells);
        let padded = (used.div_ceil(256) * 256).max(256);
        padded.min(size)
    }

    /// `llama_kv_cache_context::get_n_kv()` of the SWA cache
    /// (llama-kv-cache.cpp:2743-2745 → :1250-1263, same 256 padding)
    pub fn n_kv_swa(&self) -> u32 {
        match &self.swa {
            Some(s) => Self::n_kv_of(&s.cells, s.size),
            None => 0,
        }
    }

    /// mark cells [s0,s1] used with per-token positions
    pub fn assign(&mut self, sinfo: SlotInfo, pos: &[i32], seq_id: usize) {
        let mask = 1u64 << seq_id;
        for (i, &p) in pos.iter().enumerate() {
            let cell = &mut self.cells[sinfo.s0 as usize + i];
            cell.pos = p;
            cell.seq = mask;
        }
        self.head = (sinfo.s1 + 1) % self.size;
    }

    /// `llama_kv_cache::apply_ubatch` (llama-kv-cache.cpp:1096-1186) for the
    /// SWA cache. Two things happen beyond a plain write:
    ///
    ///   * a cell that is overwritten is removed first (:1117-1131); the
    ///     largest removed position is remembered per sequence;
    ///   * to keep the invariant "all positions between [pos_min, pos_max] of a
    ///     sequence are present in the cache", every position below that
    ///     maximum is purged (:1152-1179) — a sliding window overwrites the
    ///     oldest cells, which would otherwise leave a hole.
    ///
    /// `head` ends up past the last placed cell and may equal `size`
    /// (:1182-1186; `find_slot_swa` wraps it).
    pub fn assign_swa(&mut self, idxs: &[u32], pos: &[i32], seq_id: usize) {
        let swa = self
            .swa
            .as_mut()
            .expect("assign_swa without the iswa cache");
        let mask = 1u64 << seq_id;
        let mut pos_max_rm = -1i32;

        if idxs.is_empty() {
            return;
        }

        for (i, &idx) in idxs.iter().enumerate() {
            let cell = &mut swa.cells[idx as usize];
            if !cell.is_empty() {
                pos_max_rm = pos_max_rm.max(cell.pos);
                *cell = KvCell::default();
            }
            cell.pos = pos[i];
            cell.seq = mask;
        }

        // llama-kv-cache.cpp:1160-1179
        if pos_max_rm >= 0 {
            let pos_min = Self::seq_pos_min(&swa.cells, mask);
            if pos_min >= 0 && pos_min <= pos_max_rm {
                for cell in swa.cells.iter_mut() {
                    if cell.seq & mask == 0 || cell.pos > pos_max_rm {
                        continue;
                    }
                    cell.seq &= !mask;
                    if cell.seq == 0 {
                        *cell = KvCell::default();
                    }
                }
            }
        }

        swa.head = idxs[idxs.len() - 1] + 1;
    }

    // -----------------------------------------------------------------------
    // Multi-sequence slot management — llama-kv-cache.cpp find_slot /
    // apply_ubatch for a batch whose tokens carry several sequence ids. The
    // port's cache is single-stream (`n_stream == 1`, llama-kv-cache.cpp:669
    // picks `split_simple` for it), so `find_slot` runs with `n_seqs = 1` and
    // `n_tokens = ubatch.n_tokens`: the cells are handed out in token order and
    // the only ubatch input the search reads is that token count (:966-974 —
    // `ubatch.pos`/`ubatch.seq_id` are commented out in the C body :1033-1034).
    // -----------------------------------------------------------------------

    /// `llama_kv_cache::find_slot(ubatch, cont = false)` (llama-kv-cache.cpp:
    /// 898-1095) for the base cache: ring scan from `v_heads[0]`, keeping every
    /// usable cell until `n_tokens` are found.
    ///
    /// `cont == false` is what `prepare` passes (:769), i.e. `n_test = 1` and
    /// the picked cells need not be contiguous. `can_use` is "the cell is
    /// empty" for a non-SWA cache (the `is_empty`-only branch of :1042-1061 —
    /// the causal-mask reuse below it stays disabled in the C).
    pub fn find_slot_ubatch(&self, n_tokens: u32) -> Option<Vec<u32>> {
        self.find_slot_ring(n_tokens, 0)
    }

    /// The ring scan of [`KvCache::find_slot_ubatch`] over the SWA cache
    /// (`is_empty(idx) || (seq_count(idx) == 1 && is_masked_swa(...))`,
    /// llama-kv-cache.cpp:1042-1061).
    pub fn find_slot_ubatch_swa(&self, n_tokens: u32) -> Option<Vec<u32>> {
        self.find_slot_ring(n_tokens, 1)
    }

    fn find_slot_ring(&self, n_tokens: u32, which: usize) -> Option<Vec<u32>> {
        let (cells, size, head0) = if which == 1 {
            let s = self.swa.as_ref()?;
            (&s.cells, s.size, s.head)
        } else {
            (&self.cells, self.size, self.head)
        };
        if n_tokens == 0 || n_tokens > size {
            return None;
        }

        let mut head_cur = head0;
        // (:1003-1007) prefer the start of the cache when there is enough room
        if head_cur > Self::used(cells) + 2 * n_tokens {
            head_cur = 0;
        }

        let n_swa = if which == 1 {
            self.swa.as_ref().unwrap().n_swa
        } else {
            0
        };
        let swa_type = if which == 1 {
            self.swa.as_ref().unwrap().swa_type
        } else {
            LlamaSwaType::NONE
        };

        let mut n_tested = 0u32;
        let mut idxs: Vec<u32> = Vec::with_capacity(n_tokens as usize);

        loop {
            // (:1020-1025) n_test == 1 for cont == false
            if head_cur + 1 > size {
                n_tested += size - head_cur;
                head_cur = 0;
                continue;
            }

            let idx = head_cur;
            head_cur += 1;
            n_tested += 1;

            let cell = &cells[idx as usize];
            // (:1042-1061) can_use
            let mut can_use = cell.is_empty();
            if !can_use && cell.seq_count() == 1 {
                let seq_id_cell = cell.seq.trailing_zeros();
                if LlamaHparams::is_masked_swa(
                    n_swa,
                    swa_type,
                    cell.pos,
                    Self::seq_pos_max(cells, 1u64 << seq_id_cell) + 1,
                ) {
                    can_use = true;
                }
            }
            if can_use {
                idxs.push(idx);
            }

            if idxs.len() as u32 == n_tokens {
                break;
            }
            if n_tested >= size {
                return None;
            }
        }

        Some(idxs)
    }

    /// `llama_kv_cache::apply_ubatch` (llama-kv-cache.cpp:1097-1186) for the
    /// base cache: write the cells, remember the largest position each
    /// sequence loses to an overwrite, purge that sequence's positions below
    /// it (:1160-1179) and move the head past the last placed cell (:1180-1185).
    ///
    /// `n_seq_id[i]`/`seq_id[i]` are the ubatch's per-token sequence lists
    /// (:1154-1156 `cells.seq_add(idx, seq_id)` for every entry).
    pub fn apply_ubatch(
        &mut self,
        idxs: &[u32],
        pos: &[i32],
        n_seq_id: &[i32],
        seq_id: &[Vec<i32>],
    ) {
        let mut seq_pos_max_rm = vec![-1i32; 64];
        for (i, &idx) in idxs.iter().enumerate() {
            let cell = &mut self.cells[idx as usize];
            if !cell.is_empty() {
                // (:1120-1129) a reused cell belonged to exactly one sequence
                assert_eq!(
                    cell.seq_count(),
                    1,
                    "apply_ubatch: reused cell holds >1 seq"
                );
                let s = cell.seq.trailing_zeros() as usize;
                seq_pos_max_rm[s] = seq_pos_max_rm[s].max(cell.pos);
                *cell = KvCell::default();
            }
            cell.pos = pos[i];
            for s in 0..n_seq_id[i] as usize {
                cell.seq |= 1u64 << seq_id[i][s];
            }
        }

        // preserve "all positions in [pos_min, pos_max] of a sequence are present"
        for s in 0..64usize {
            if seq_pos_max_rm[s] == -1 {
                continue;
            }
            let mask = 1u64 << s;
            let pos_min = Self::seq_pos_min(&self.cells, mask);
            if pos_min <= seq_pos_max_rm[s] {
                self.seq_rm(s, pos_min, seq_pos_max_rm[s] + 1);
            }
        }

        if let Some(&last) = idxs.last() {
            self.head = last + 1;
        }
    }

    /// `llama_kv_cache::apply_ubatch` for the SWA cache of the `iswa` pair.
    /// Same body, but `seq_rm` there targets the SWA cache only
    /// (llama-kv-cache-iswa.cpp:187-236 runs the per-cache `apply`).
    pub fn apply_ubatch_swa(
        &mut self,
        idxs: &[u32],
        pos: &[i32],
        n_seq_id: &[i32],
        seq_id: &[Vec<i32>],
    ) {
        let mut seq_pos_max_rm = vec![-1i32; 64];
        {
            let s = self
                .swa
                .as_ref()
                .expect("apply_ubatch_swa without the iswa cache");
            for (i, &idx) in idxs.iter().enumerate() {
                let cell = &s.cells[idx as usize];
                if !cell.is_empty() {
                    assert_eq!(
                        cell.seq_count(),
                        1,
                        "apply_ubatch_swa: reused cell holds >1 seq"
                    );
                    let sq = cell.seq.trailing_zeros() as usize;
                    seq_pos_max_rm[sq] = seq_pos_max_rm[sq].max(cell.pos);
                }
            }
        }

        let mut last = 0u32;
        {
            let s = self.swa.as_mut().unwrap();
            for (i, &idx) in idxs.iter().enumerate() {
                let cell = &mut s.cells[idx as usize];
                *cell = KvCell::default();
                cell.pos = pos[i];
                for j in 0..n_seq_id[i] as usize {
                    cell.seq |= 1u64 << seq_id[i][j];
                }
                last = idx;
            }
        }

        for sq in 0..64usize {
            if seq_pos_max_rm[sq] == -1 {
                continue;
            }
            let mask = 1u64 << sq;
            let (ok, pos_min) = {
                let s = self.swa.as_ref().unwrap();
                (true, Self::seq_pos_min(&s.cells, mask))
            };
            if ok && pos_min <= seq_pos_max_rm[sq] {
                self.seq_rm_swa(sq, pos_min, seq_pos_max_rm[sq] + 1);
            }
        }

        if let Some(s) = self.swa.as_mut() {
            s.head = last + 1;
        }
    }

    /// `llama_kv_cache::seq_rm` on the SWA cache only
    /// (llama-kv-cache-iswa.cpp:113-120 splits the two caches' removals).
    fn seq_rm_swa(&mut self, seq_id: usize, p0: i32, p1: i32) {
        let mask = 1u64 << seq_id;
        let Some(s) = self.swa.as_mut() else { return };
        for cell in s.cells.iter_mut() {
            if cell.is_empty() || cell.seq & mask == 0 {
                continue;
            }
            if p0 != -1 && cell.pos < p0 {
                continue;
            }
            if p1 != -1 && cell.pos >= p1 {
                continue;
            }
            cell.seq &= !mask;
            if cell.seq == 0 {
                *cell = KvCell::default();
            }
        }
    }

    /// llama_kv_cache::seq_rm — drop cells of seq in [p0, p1) (-1 = open).
    /// `llama_kv_cache_iswa::seq_rm` applies it to both caches
    /// (llama-kv-cache-iswa.cpp:113-120), so this does too. The head moves
    /// back to the first freed cell when it is before the current one
    /// (:403-419 "If we freed up a slot, set head to it so searching can
    /// start there") — what makes a rolled-back re-decode reuse the same
    /// cells the original decode wrote.
    pub fn seq_rm(&mut self, seq_id: usize, p0: i32, p1: i32) {
        let mask = 1u64 << seq_id;
        let rm = |cells: &mut Vec<KvCell>, head: &mut u32| {
            let size = cells.len() as u32;
            let mut new_head = size;
            for i in 0..cells.len() {
                let cell = &cells[i];
                if cell.is_empty() || cell.seq & mask == 0 {
                    continue;
                }
                if p0 != -1 && cell.pos < p0 {
                    continue;
                }
                if p1 != -1 && cell.pos >= p1 {
                    continue;
                }
                let c = &mut cells[i];
                c.seq &= !mask;
                if c.seq == 0 {
                    *c = KvCell::default();
                    if new_head == size {
                        new_head = i as u32;
                    }
                }
            }
            // :417-419 — restart the next search at the first freed cell
            if new_head != size && new_head < *head {
                *head = new_head;
            }
        };
        rm(&mut self.cells, &mut self.head);
        if let Some(s) = self.swa.as_mut() {
            rm(&mut s.cells, &mut s.head);
        }
        // llama_kv_cache_dsa::seq_rm (llama-kv-cache-dsa.cpp:61-68)
        if let Some(l) = self.lid.as_mut() {
            let mut head_tmp = l.head;
            rm(&mut l.cells, &mut head_tmp);
            l.head = head_tmp;
        }
        // note: `llama_kv_cache_dsv4::seq_rm` (llama-kv-cache-dsv4.cpp:1459-
        // 1480) additionally rolls the three *compressed* caches back to
        // `p0 / ratio` rows on the driver's tail drop (`p0 > pos_max`). The
        // port keeps `n_visible` position-derived instead: capping it by a
        // rolled-back row count moves the port OFF the reference's own
        // draft-mtp stream (measured 16/16 with the position-derived count,
        // PARITY.md's MTP section) — the reference evidently recompletes
        // rolled-back blocks from the live state plane before any query can
        // see the difference. Not ported; documented.
    }

    // -----------------------------------------------------------------------
    // seq_add / seq_div / seq_keep / seq_cp — llama-kv-cache.cpp:451-657, the
    // position-shift family the server's context shift drives
    // (server-context.cpp:2951-2952). The port's single KvCache stands in for
    // whichever composite the C dispatches to, so one call fans out over every
    // cell array it holds: the base cells + the `iswa` SWA cells
    // (llama-kv-cache-iswa.cpp:132-139 forwards to both), the `dsa` lid cells
    // (llama-kv-cache-dsa.cpp:80-88). For `dsv4` the same base+swa pair *is*
    // the raw pair — the C's `llama_kv_cache_dsv4::seq_add` shifts nothing
    // else (llama-kv-cache-dsv4.cpp:1544-1546); its compressed streams cannot
    // follow a position shift at all (`get_can_shift() == false`,
    // llama-kv-cache-dsv4.cpp:1448-1452), so the port rejects the shift at
    // [`crate::context::DecodeContext::seq_add`] instead of the C's abort at
    // the next update().
    // -----------------------------------------------------------------------

    /// `llama_kv_cache::seq_add` (llama-kv-cache.cpp:570-618): every cell of
    /// the sequence with pos in [p0, p1) (-1 = open) gets
    /// `pos += shift` and the shift accumulator `shift += shift`
    /// (`pos_add`, llama-kv-cells.h:440-464). A position pushed below 0 frees
    /// the cell (:451-459) — the head then restarts at the first freed cell,
    /// or at 0 when none was freed (:615-617).
    ///
    /// The C asserts `hparams.n_pos_per_embd() == 1` (:577); the port's cache
    /// cannot see the hparams, so the caller (DecodeContext) rejects M-RoPE.
    pub fn seq_add(&mut self, seq_id: usize, p0: i32, p1: i32, shift: i32) {
        if shift == 0 {
            return; // :582-584
        }
        let p0 = if p0 < 0 { 0 } else { p0 }; // :588-590
        let p1 = if p1 < 0 { i32::MAX } else { p1 }; // :592-594
        if p0 == p1 {
            return; // :596-599
        }
        let mask = 1u64 << seq_id;
        let add = |cells: &mut Vec<KvCell>, head: &mut u32| {
            let size = cells.len() as u32;
            let mut new_head = size; // :586
            for i in 0..cells.len() {
                let cell = &cells[i];
                // `pos_in(i, p0, p1)` + `seq_has(i, seq_id)` (:601-606)
                if cell.pos < p0 || cell.pos >= p1 || cell.seq & mask == 0 {
                    continue;
                }
                // pos_add (llama-kv-cells.h:440-464): pos/shift advance; a
                // negative position frees the cell (and reports it freed)
                let c = &mut cells[i];
                c.pos += shift;
                c.shift += shift;
                if c.pos < 0 {
                    *c = KvCell::default();
                    if new_head == size {
                        new_head = i as u32;
                    }
                }
            }
            // :615-617 — start the next search at the first freed cell
            *head = if new_head != size { new_head } else { 0 };
        };
        add(&mut self.cells, &mut self.head);
        if let Some(s) = self.swa.as_mut() {
            add(&mut s.cells, &mut s.head);
        }
        if let Some(l) = self.lid.as_mut() {
            add(&mut l.cells, &mut l.head);
        }
    }

    /// `llama_kv_cache::seq_div` (llama-kv-cache.cpp:620-657): `pos /= d`
    /// (`pos_div`, llama-kv-cells.h:469-483 — C integer division, truncating
    /// toward zero, same as Rust's), with the *difference* added to the shift
    /// accumulator (`shift += p_old - pos_new`) so the K-shift graph still
    /// rotates by the exact delta. Like the C's, no head update (a division
    /// never frees a cell — `pos_in` only matches pos >= p0 >= 0).
    pub fn seq_div(&mut self, seq_id: usize, p0: i32, p1: i32, d: i32) {
        if d == 1 {
            return; // :631-633
        }
        let p0 = if p0 < 0 { 0 } else { p0 };
        let p1 = if p1 < 0 { i32::MAX } else { p1 };
        if p0 == p1 {
            return;
        }
        let mask = 1u64 << seq_id;
        let div = |cells: &mut [KvCell]| {
            for cell in cells.iter_mut() {
                if cell.pos < p0 || cell.pos >= p1 || cell.seq & mask == 0 {
                    continue;
                }
                let p_old = cell.pos;
                cell.pos /= d;
                cell.shift += p_old - cell.pos; // llama-kv-cells.h:478
            }
        };
        div(&mut self.cells);
        if let Some(s) = self.swa.as_mut() {
            div(&mut s.cells);
        }
        if let Some(l) = self.lid.as_mut() {
            div(&mut l.cells);
        }
    }

    /// `llama_kv_cache::seq_keep` (llama-kv-cache.cpp:543-568): every cell
    /// keeps only `seq_id` in its sequence set (`seq_keep`,
    /// llama-kv-cells.h); a cell left with no sequence is freed, and the head
    /// moves to the first freed cell when it is before the current one
    /// (:554-567).
    pub fn seq_keep(&mut self, seq_id: usize) {
        let mask = 1u64 << seq_id;
        let keep = |cells: &mut Vec<KvCell>, head: &mut u32| {
            let size = cells.len() as u32;
            let mut new_head = size;
            for i in 0..cells.len() {
                let cell = &cells[i];
                if cell.is_empty() {
                    continue;
                }
                if cell.seq & mask != 0 {
                    cells[i].seq = mask; // keep only this sequence
                } else {
                    cells[i] = KvCell::default(); // freed
                    if new_head == size {
                        new_head = i as u32;
                    }
                }
            }
            // :564-567
            if new_head != size && new_head < *head {
                *head = new_head;
            }
        };
        keep(&mut self.cells, &mut self.head);
        if let Some(s) = self.swa.as_mut() {
            keep(&mut s.cells, &mut s.head);
        }
        if let Some(l) = self.lid.as_mut() {
            keep(&mut l.cells, &mut l.head);
        }
    }

    /// `llama_kv_cache::seq_cp` (llama-kv-cache.cpp:451-541), the
    /// single-stream branch — the port's raw half is single-stream by
    /// construction (`n_stream == 1` shared cells), so `s0 == s1` and no
    /// buffer data is copied: cells of `seq_id_src` with pos in [p0, p1)
    /// simply gain `seq_id_dst` (:463-491). The cross-stream branch
    /// (:494-536) enqueues a buffer copy the raw half can never need.
    ///
    /// `dsv4` (llama_kv_cache_dsv4.cpp:1512-1527): the raw pair takes the
    /// same metadata merge (`kv_raw->seq_cp`), then the three compressed K
    /// caches and the three compressor states are copied stream-to-stream
    /// ([`KvCache::seq_cp_dsv4`]) and `rs_idx[dst]` resets. The C asserts
    /// `p0 <= 0 && p1 < 0` ("DSV4 only supports full sequence copies",
    /// :1513); a stream out of range is the C's own assertion territory
    /// (`seq_to_stream.size()` / `n_stream`, :457-458/:1026-1027) — the port
    /// reports it instead of aborting.
    pub fn seq_cp(
        &mut self,
        seq_id_src: usize,
        seq_id_dst: usize,
        p0: i32,
        p1: i32,
    ) -> Result<(), String> {
        if seq_id_src == seq_id_dst {
            return Ok(()); // :469-471
        }
        let p0 = if p0 < 0 { 0 } else { p0 };
        let p1 = if p1 < 0 { i32::MAX } else { p1 };
        let (m0, m1) = (1u64 << seq_id_src, 1u64 << seq_id_dst);
        let cp = |cells: &mut [KvCell]| {
            for cell in cells.iter_mut() {
                if cell.pos < p0 || cell.pos >= p1 {
                    continue;
                }
                if cell.seq & m0 != 0 {
                    cell.seq |= m1; // cells.seq_add(i, seq_id_dst) (:487)
                }
            }
        };
        cp(&mut self.cells);
        if let Some(s) = self.swa.as_mut() {
            cp(&mut s.cells);
        }
        // llama_kv_cache_dsa::seq_cp (llama-kv-cache-dsa.cpp:70-73)
        if let Some(l) = self.lid.as_mut() {
            cp(&mut l.cells);
        }
        // the dsv4 compressed half (llama-kv-cache-dsv4.cpp:1512-1527) — the
        // copy needs the caller's Context, so it runs through the driver
        // (`crate::context::DecodeContext::seq_cp`); here only the guard
        if self.dsv4.is_some() && (p0 > 0 || p1 != i32::MAX) {
            return Err(
                "seq_cp on the deepseek4 dsv4 cache: DSV4 only supports full sequence copies \
                 (llama-kv-cache-dsv4.cpp:1513)"
                    .into(),
            );
        }
        Ok(())
    }

    /// `llama_kv_cache::get_has_shift` (llama-kv-cache.cpp:1209-1214): whether
    /// any cell accumulated a shift since the last reset — what
    /// `init_update` turns into `do_shift` (:746). The C keeps a sticky
    /// per-cell-array flag; the port derives it from the accumulators, which
    /// is value-equivalent: a pending shift of all zeros makes the K-shift
    /// graph rotate by 0 (cos 0 = 1, sin 0 = 0 — the identity), so skipping
    /// it changes nothing.
    pub fn get_has_shift(&self) -> bool {
        let any = |cells: &[KvCell]| cells.iter().any(|c| !c.is_empty() && c.shift != 0);
        any(&self.cells)
            || self.swa.as_ref().map(|s| any(&s.cells)).unwrap_or(false)
            || self.lid.as_ref().map(|l| any(&l.cells)).unwrap_or(false)
    }

    /// `v_cells[s].reset_shift()` (llama-kv-cells.h:57-59) — the K-shift
    /// graph consumed the accumulators (llama-kv-cache.cpp:888-892).
    pub fn reset_shift(&mut self) {
        let rs = |cells: &mut [KvCell]| cells.iter_mut().for_each(|c| c.shift = 0);
        rs(&mut self.cells);
        if let Some(s) = self.swa.as_mut() {
            rs(&mut s.cells);
        }
        if let Some(l) = self.lid.as_mut() {
            rs(&mut l.cells);
        }
    }

    /// llama_kv_cache_iswa::clear (llama-kv-cache-iswa.cpp:108-111); the dsa
    /// pair clears both halves too (llama-kv-cache-dsa.cpp:56-59)
    pub fn clear(&mut self) {
        self.cells.fill(KvCell::default());
        self.head = 0;
        if let Some(s) = self.swa.as_mut() {
            s.cells.fill(KvCell::default());
            s.head = 0;
        }
        if let Some(l) = self.lid.as_mut() {
            l.cells.fill(KvCell::default());
            l.head = 0;
        }
    }

    /// llama_kv_cache::get_k — view [n_embd_head_k, n_head_kv, n_kv, 1]
    pub fn get_k(
        &self,
        ctx: &mut Context,
        il: usize,
        n_embd_head_k: i64,
        n_head_kv: i64,
        n_kv: u32,
    ) -> TensorId {
        let k = self.layers[il].k;
        let size = self.cache_size_of(il) as i64;
        let n_embd = self.k_row[il];
        let rs_head = GgmlType::F16.row_size(n_embd_head_k as usize);
        let rs_gqa = GgmlType::F16.row_size(n_embd as usize);
        ctx.view_4d(
            k,
            n_embd_head_k,
            n_head_kv,
            n_kv as i64,
            1,
            rs_head,
            rs_gqa,
            rs_gqa * size as usize,
            0,
        )
    }

    /// llama_kv_cache::get_v — view [n_embd_head_v, n_head_kv, n_kv, 1]
    pub fn get_v(
        &self,
        ctx: &mut Context,
        il: usize,
        n_embd_head_v: i64,
        n_head_kv: i64,
        n_kv: u32,
    ) -> TensorId {
        let v = self.layers[il].v;
        let size = self.cache_size_of(il) as i64;
        let n_embd = self.v_row[il];
        let rs_head = GgmlType::F16.row_size(n_embd_head_v as usize);
        let rs_gqa = GgmlType::F16.row_size(n_embd as usize);
        ctx.view_4d(
            v,
            n_embd_head_v,
            n_head_kv,
            n_kv as i64,
            1,
            rs_head,
            rs_gqa,
            rs_gqa * size as usize,
            0,
        )
    }

    /// the cell count of the tensor `layers[il]` was allocated with — the base
    /// cache's for a dense layer, the SWA cache's for an SWA layer
    /// (llama-kv-cache.cpp:1266-1284 `kv_size = get_size()`). This is the
    /// `nb[3]`/`nb[2]` stride of the views.
    pub fn cache_size_of(&self, il: usize) -> u32 {
        if self.layer_is_swa(il) {
            self.swa.as_ref().map(|s| s.size).unwrap_or(self.size)
        } else {
            self.size
        }
    }
}

/// The KV cache is the port's `llama_memory_i`: `llama_batch_allocr::init`
/// reads the per-sequence position window off it (llama-batch.cpp:96-101 /
/// :295 / :323-335) — `llama_kv_cache::seq_pos_min/max`, which the C defines
/// through `v_cells.seq_pos_min/max` (llama-kv-cache.cpp:2290-2305).
impl crate::batch::BatchMemory for KvCache {
    fn seq_pos_min(&self, seq_id: i32) -> i32 {
        Self::seq_pos_min_of(self, seq_id as usize)
    }

    fn seq_pos_max(&self, seq_id: i32) -> i32 {
        Self::seq_pos_max_of(self, seq_id as usize)
    }
}

// ===========================================================================
// `llama_state_seq_*` — the sequence-state serialization
// (llama-kv-cache.cpp:2055-2628, llama-kv-cache-dsv4.cpp:1080-1673,
// llama-context.cpp:3142-3213)
//
// The C chains `llama_state_seq_get_data` → `state_seq_write_data` →
// `memory->state_write` → (dsv4) `llama_kv_cache_dsv4::state_write`; the port
// exposes the same surface as [`KvCache::state_seq_write`] /
// [`KvCache::state_seq_read`] over a host [`StateWriter`] / [`StateReader`]
// (the `llama_io_write_host` / `llama_io_read_i` pair), driven by
// `DecodeContext::state_seq_get_data` / `state_seq_set_data` (context.rs).
//
// Byte format (every integer little-endian, exact widths):
//   [u32 io_magic = 0xaf143cd8][i32 seq_id]                    (llama-context.
//     cpp:3142/:3166-3167 — written by the DecodeContext wrapper, not here)
// then, for a dsv4 cache (llama_kv_cache_dsv4::state_write, :1594-1626):
//   [u32 DSV4_STATE_MAGIC][u32 DSV4_STATE_VERSION][u32 mode]
//   <kv_raw: base cache then swa cache, llama_kv_cache::state_write :2055>
//   if !partial: <csa k-cache><hca k-cache><lid k-cache>       (:332-412)
//   <csa comp state><hca comp state><lid comp state>           (:1080-1158)
// for a plain/iswa cache the blob is the <kv_raw> part alone (the port's
// `state_seq_write` without the dsv4 frame).
// ===========================================================================

/// `DSV4_STATE_*` / `DSV4_K_CACHE_STATE_VER` / `DSV4_COMP_STATE_VER`
/// (llama-kv-cache-dsv4.cpp:21-26)
pub const DSV4_STATE_MAGIC: u32 = 0x3456_5344; // DSV4
pub const DSV4_STATE_VERSION: u32 = 1;
pub const DSV4_STATE_MODE_FULL: u32 = 0;
pub const DSV4_STATE_MODE_PARTIAL: u32 = 1;
pub const DSV4_K_CACHE_STATE_VER: u32 = 2;
pub const DSV4_COMP_STATE_VER: u32 = 1;

/// `dsv4_state_n_used_k_rows` (llama-kv-cache-dsv4.cpp:41-49): the compressed
/// rows a pos_max-high sequence has completed (`(pos_max + 1)/ratio`, capped
/// at the cache size).
pub fn dsv4_state_n_used_k_rows(pos_max: i32, ratio: u32, kv_size: u32) -> u32 {
    if pos_max < 0 {
        return 0;
    }
    let n_rows = (pos_max as u64 + 1) / ratio as u64;
    kv_size.min(n_rows as u32)
}

/// `llama_io_write_host` (llama-context.cpp's write half) with the dummy mode
/// of `llama_io_write_dummy` (`state_seq_get_size` runs the same writer and
/// discards the payload, io.h `n_bytes()` still counting). `write_tensor` is
/// immediate: the port owns the cache tensors host-side, so a row-range copy
/// is the whole job (the C defers it to collect the backend buffer layout —
/// same bytes out).
pub struct StateWriter {
    buf: Vec<u8>,
    size_written: usize,
    dummy: bool,
}

impl StateWriter {
    pub fn new(dummy: bool) -> Self {
        StateWriter {
            buf: Vec::new(),
            size_written: 0,
            dummy,
        }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        self.size_written += bytes.len();
        if !self.dummy {
            self.buf.extend_from_slice(bytes);
        }
    }

    pub fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }

    pub fn write_i32(&mut self, v: i32) {
        self.write(&v.to_le_bytes());
    }

    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    /// `io.write_tensor(tensor, offset, size)` (llama-io.h:49-53) — the
    /// `size` bytes of `tensor` at byte `offset`
    pub fn write_tensor(&mut self, gctx: &Context, tensor: TensorId, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        if self.dummy {
            self.size_written += size;
            return;
        }
        let all = gctx.data_bytes(tensor).expect("state write tensor");
        self.buf.extend_from_slice(&all[offset..offset + size]);
        self.size_written += size;
    }

    pub fn n_bytes(&self) -> usize {
        self.size_written
    }

    pub fn into_bytes(self) -> Vec<u8> {
        debug_assert_eq!(self.buf.len(), self.size_written);
        self.buf
    }
}

/// contiguous `[b, e)` runs of an ascending-or-scattered cell index list —
/// the scatter ranges of `state_read_data`.
fn contig_runs(idxs: &[u32]) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let mut i0 = 0usize;
    while i0 < idxs.len() {
        let mut i1 = i0 + 1;
        while i1 < idxs.len() && idxs[i1] == idxs[i1 - 1] + 1 {
            i1 += 1;
        }
        runs.push((idxs[i0], idxs[i1 - 1] + 1));
        i0 = i1;
    }
    runs
}

/// `llama_kv_cache::state_clear` (llama-kv-cache.cpp:2677-2692, 08618ff8e):
/// undo a state_read() of seq_id (-1 for the whole cache) that another memory
/// module failed to complete. The cells another sequence still shares keep
/// their data; the sequence's now-exclusive cells are zeroed.
pub fn state_clear_seq(gctx: &mut Context, kv: &mut KvCache, seq_id: i32) {
    if seq_id == -1 {
        // clear(true) of both halves
        kv.raw_state_clear(false, gctx, -1, &[], false);
        if kv.swa.is_some() {
            kv.raw_state_clear(true, gctx, -1, &[], false);
        }
        return;
    }

    let mask = 1u64 << seq_id;
    for swa_half in [false, true] {
        if swa_half && kv.swa.is_none() {
            continue;
        }
        // the k-only halves mirror the read path's k_only (the dsa/dsv4
        // branches read their base half k-only)
        let k_only = swa_half || kv.lid.is_some() || kv.dsv4.is_some() || kv.k_only;
        let cells = if swa_half {
            &kv.swa.as_ref().unwrap().cells
        } else {
            &kv.cells
        };
        // a cell that another sequence still uses keeps its data
        let idxs: Vec<u32> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty() && c.seq & mask != 0 && c.seq.count_ones() == 1)
            .map(|(i, _)| i as u32)
            .collect();
        let runs = contig_runs(&idxs);
        kv.raw_state_clear(swa_half, gctx, seq_id, &runs, k_only);
    }
}

/// `llama_io_read_host` (llama-context.cpp's read half) — a bounds-checked
/// cursor. The C throws `std::runtime_error` on underrun; the port returns
/// `Err`.
pub struct StateReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> StateReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        StateReader { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("unexpectedly reached end of buffer".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<(), String> {
        buf.copy_from_slice(self.read_bytes(buf.len())?);
        Ok(())
    }

    pub fn read_u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_i32(&mut self) -> Result<i32, String> {
        let mut b = [0u8; 4];
        self.read(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    pub fn read_u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// `io.read_tensor(tensor, offset, size)` (llama-io.h:66-70)
    pub fn read_tensor(
        &mut self,
        gctx: &mut Context,
        tensor: TensorId,
        offset: usize,
        size: usize,
    ) -> Result<(), String> {
        if size == 0 {
            return Ok(());
        }
        let bytes = self.read_bytes(size)?.to_vec();
        let all = gctx.data_bytes_mut(tensor).expect("state read tensor");
        all[offset..offset + size].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn n_bytes(&self) -> usize {
        self.pos
    }
}

/// `llama_clear_tensor_data` (llama-impl.cpp:92-100, 08618ff8e) specialized
/// to whole row ranges: zero `[b, e)` rows of `size_row` bytes each. The C
/// writes zeros through `ggml_backend_tensor_set` in 1 MiB chunks; the port's
/// arena tensors zero in place.
fn zero_tensor_rows(gctx: &mut Context, t: TensorId, size_row: usize, runs: &[(u32, u32)]) {
    if size_row == 0 {
        return;
    }
    let Some(all) = gctx.data_bytes_mut(t) else {
        return;
    };
    for &(b, e) in runs {
        let start = b as usize * size_row;
        let end = e as usize * size_row;
        if end <= all.len() {
            all[start..end].fill(0);
        }
    }
}

/// `dsv4_state_src_stream_range` (llama-kv-cache-dsv4.cpp:206-223): which
/// per-sequence stream rows a save reads.
fn dsv4_state_src_stream_range(n_stream: u32, seq_id: i32) -> Result<(u32, u32), String> {
    if seq_id >= 0 && n_stream > 1 {
        if seq_id as u32 >= n_stream {
            return Err("DSV4 state sequence id out of stream range".into());
        }
        return Ok((seq_id as u32, 1));
    }
    Ok((0, if seq_id >= 0 { 1 } else { n_stream }))
}

/// `dsv4_state_dst_stream_range` (:225-247)
fn dsv4_state_dst_stream_range(n_stream: u32, seq_id: i32, ns: u32) -> Result<u32, String> {
    if seq_id >= 0 {
        if ns != 1 {
            return Err("DSV4 sequence state stream count mismatch".into());
        }
        if n_stream > 1 && seq_id as u32 >= n_stream {
            return Err("DSV4 state sequence id out of stream range".into());
        }
        return Ok(if n_stream > 1 { seq_id as u32 } else { 0 });
    }
    if ns != n_stream {
        return Err("DSV4 full state stream count mismatch".into());
    }
    Ok(0)
}

/// `dsv4_state_write_tensor_streams` (llama-kv-cache-dsv4.cpp:249-289): the
/// metadata quad (type / ne0 / rows / row_size) then `n_rows` rows of every
/// selected stream. The port's per-stream tensors are 2-D row blocks
/// (`[ne0, size * n_stream]`), so stream `s` starts at row `stream * size` —
/// the same bytes the C's 3-D `[ne0, rows, stream]` layout holds.
fn dsv4_state_write_tensor_streams(
    io: &mut StateWriter,
    gctx: &Context,
    tensor: TensorId,
    tensor_rows: u32,
    n_rows: u32,
    s0: u32,
    ns: u32,
    stream_ids: Option<&[u32]>,
) -> Result<(), String> {
    let ty = gctx.ty(tensor);
    let ne0 = gctx.ne(tensor)[0] as u64;
    let rows = n_rows as u64;
    let row_size = ty.row_size(ne0 as usize) as u64;

    if n_rows > tensor_rows {
        return Err("DSV4 state tensor row count exceeds storage".into());
    }

    io.write_i32(ty as i32);
    io.write_u64(ne0);
    io.write_u64(rows);
    io.write_u64(row_size);

    let stream_stride = tensor_rows as usize * row_size as usize;
    let size = n_rows as usize * row_size as usize;
    if size == 0 {
        return Ok(());
    }

    if let Some(ids) = stream_ids {
        if ids.len() != ns as usize {
            return Err("DSV4 state tensor stream map size mismatch".into());
        }
    }

    for s in 0..ns {
        let stream = stream_ids.map(|ids| ids[s as usize]).unwrap_or(s0 + s);
        if stream as i64 >= gctx.ne(tensor)[2].max(gctx.ne(tensor)[1]) {
            return Err("DSV4 state tensor stream out of range".into());
        }
        let offset = stream as usize * stream_stride;
        io.write_tensor(gctx, tensor, offset, size);
    }
    Ok(())
}

/// `dsv4_state_read_tensor_streams` (llama-kv-cache-dsv4.cpp:291-330)
fn dsv4_state_read_tensor_streams(
    io: &mut StateReader,
    gctx: &mut Context,
    tensor: TensorId,
    tensor_rows: u32,
    n_rows: u32,
    s0: u32,
    ns: u32,
) -> Result<(), String> {
    let type_i_ref = io.read_i32()?;
    let ne0_ref = io.read_u64()?;
    let rows_ref = io.read_u64()?;
    let row_size_ref = io.read_u64()?;

    let ty = gctx.ty(tensor);
    let ne0 = gctx.ne(tensor)[0] as u64;
    let rows = n_rows as u64;
    let row_size = ty.row_size(ne0 as usize) as u64;

    if ty as i32 != type_i_ref || ne0 != ne0_ref || rows != rows_ref || row_size != row_size_ref {
        return Err("DSV4 state tensor metadata mismatch".into());
    }
    if n_rows > tensor_rows {
        return Err("DSV4 state tensor row count exceeds storage".into());
    }

    let stream_stride = tensor_rows as usize * row_size as usize;
    let size = n_rows as usize * row_size as usize;
    if size == 0 {
        return Ok(());
    }

    for s in 0..ns {
        let offset = (s0 + s) as usize * stream_stride;
        io.read_tensor(gctx, tensor, offset, size)?;
    }
    Ok(())
}

impl Dsv4CompCache {
    /// `dsv4_state_write_k_cache` (llama-kv-cache-dsv4.cpp:332-362): the
    /// version header, the row/stream/layer counts, then per layer the layer
    /// id + the first `n_rows` rows of the selected streams.
    pub fn state_write_k_cache(
        &self,
        io: &mut StateWriter,
        gctx: &Context,
        seq_id: i32,
        n_rows: u32,
    ) -> Result<(), String> {
        let (s0, ns) = dsv4_state_src_stream_range(self.n_stream, seq_id)?;

        let n_layer = self.layer_ids.len() as u32;
        if n_rows > self.size {
            return Err("DSV4 K-cache state row count exceeds cache size".into());
        }

        io.write_u32(DSV4_K_CACHE_STATE_VER);
        io.write_u32(n_rows);
        io.write_u32(ns);
        io.write_u32(n_layer);

        for &il in &self.layer_ids {
            io.write_u32(il as u32);
            dsv4_state_write_tensor_streams(
                io,
                gctx,
                self.k[self.layer_slot(il)],
                self.size,
                n_rows,
                s0,
                ns,
                None,
            )?;
        }
        Ok(())
    }

    /// `dsv4_state_read_k_cache` (llama-kv-cache-dsv4.cpp:364-412)
    pub fn state_read_k_cache(
        &mut self,
        io: &mut StateReader,
        gctx: &mut Context,
        seq_id: i32,
    ) -> Result<(), String> {
        let version = io.read_u32()?;
        let n_rows_ref = io.read_u32()?;
        let ns = io.read_u32()?;
        let n_layer_ref = io.read_u32()?;

        if version != 1 && version != DSV4_K_CACHE_STATE_VER {
            return Err("DSV4 K-cache state version mismatch".into());
        }
        if version == 1 && n_rows_ref != self.size {
            return Err("DSV4 K-cache state size mismatch".into());
        }
        if n_rows_ref > self.size {
            return Err("DSV4 K-cache state size mismatch".into());
        }

        let s0 = dsv4_state_dst_stream_range(self.n_stream, seq_id, ns)?;

        if n_layer_ref != self.layer_ids.len() as u32 {
            return Err("DSV4 K-cache layer count mismatch".into());
        }

        for &il in &self.layer_ids {
            let il_ref = io.read_u32()?;
            if il_ref != il as u32 {
                return Err("DSV4 K-cache layer id mismatch".into());
            }
            dsv4_state_read_tensor_streams(
                io,
                gctx,
                self.k[self.layer_slot(il)],
                self.size,
                n_rows_ref,
                s0,
                ns,
            )?;
        }
        Ok(())
    }
}

impl Dsv4CompState {
    /// `llama_dsv4_comp_state::state_write` (llama-kv-cache-dsv4.cpp:1080-
    /// 1116): the header (version / ratio / state_size / n_embd_state /
    /// stream count / layer count) then per layer the layer id + the kv and
    /// score plane rows of the streams selected through `stream_ids` — the
    /// rollback-plane group `rs_idx[seq]*n_stream + s0 + s` (the live planes
    /// at rs_idx 0, what a checkpoint of the *current* state needs).
    pub fn state_write(
        &self,
        io: &mut StateWriter,
        gctx: &Context,
        seq_id: i32,
        rs_idx: &[u32],
    ) -> Result<(), String> {
        let (s0, ns) = dsv4_state_src_stream_range(self.n_stream, seq_id)?;

        let mut stream_ids = vec![0u32; ns as usize];
        for s in 0..ns as usize {
            let seq = if seq_id >= 0 {
                seq_id as u32
            } else {
                s0 + s as u32
            };
            if seq as usize >= rs_idx.len() || rs_idx[seq as usize] > self.n_rs_seq {
                return Err("DSV4 recurrent state rollback index out of range".into());
            }
            stream_ids[s] = rs_idx[seq as usize] * self.n_stream + s0 + s as u32;
        }

        io.write_u32(DSV4_COMP_STATE_VER);
        io.write_u32(self.ratio);
        io.write_u32(self.state_size);
        io.write_u32(self.n_embd_state as u32);
        io.write_u32(ns);
        io.write_u32(self.layer_ids.len() as u32);

        for (slot, &il) in self.layer_ids.iter().enumerate() {
            io.write_u32(il as u32);

            dsv4_state_write_tensor_streams(
                io,
                gctx,
                self.kv[slot],
                self.state_size,
                self.state_size,
                s0,
                ns,
                Some(&stream_ids),
            )?;
            dsv4_state_write_tensor_streams(
                io,
                gctx,
                self.score[slot],
                self.state_size,
                self.state_size,
                s0,
                ns,
                Some(&stream_ids),
            )?;
        }
        Ok(())
    }

    /// `llama_dsv4_comp_state::state_read` (:1118-1158) — the restored planes
    /// always land in the live group (stream `s0`'s row blocks, the
    /// `dsv4_state_dst_stream_range` of :1145-1146)
    pub fn state_read(
        &self,
        io: &mut StateReader,
        gctx: &mut Context,
        seq_id: i32,
    ) -> Result<(), String> {
        let version = io.read_u32()?;
        let ratio_ref = io.read_u32()?;
        let state_size_ref = io.read_u32()?;
        let n_embd_state_ref = io.read_u32()?;
        let ns = io.read_u32()?;
        let n_layer_ref = io.read_u32()?;

        if version != DSV4_COMP_STATE_VER {
            return Err("DSV4 compressor state version mismatch".into());
        }
        if ratio_ref != self.ratio
            || state_size_ref != self.state_size
            || n_embd_state_ref != self.n_embd_state as u32
        {
            return Err("DSV4 compressor state metadata mismatch".into());
        }
        if n_layer_ref != self.layer_ids.len() as u32 {
            return Err("DSV4 compressor state layer count mismatch".into());
        }

        let s0 = dsv4_state_dst_stream_range(self.n_stream, seq_id, ns)?;

        for (slot, &il) in self.layer_ids.iter().enumerate() {
            let il_ref = io.read_u32()?;
            if il_ref != il as u32 {
                return Err("DSV4 compressor state layer id mismatch".into());
            }

            dsv4_state_read_tensor_streams(
                io,
                gctx,
                self.kv[slot],
                self.state_size,
                self.state_size,
                s0,
                ns,
            )?;
            dsv4_state_read_tensor_streams(
                io,
                gctx,
                self.score[slot],
                self.state_size,
                self.state_size,
                s0,
                ns,
            )?;
        }
        Ok(())
    }
}

/// one raw cache's serialization view — the base or the SWA half of the
/// port's [`KvCache`] (the C serializes `kv_base` then `kv_swa`,
/// llama-kv-cache-iswa.cpp:259-264). `layer_ids` selects that half's layers;
/// the tensors are the shared `layers[il]` (the port resolves per model
/// layer, each sized by its own cache).
struct RawCacheView<'a> {
    cells: &'a [KvCell],
    layer_ids: Vec<usize>,
    /// this half's own mask parameters (base: 0/NONE — the iswa ctor's
    /// LLAMA_SWA_TYPE_NONE at llama-kv-cache-iswa.cpp:98; swa: its window)
    n_swa: u32,
    swa_type: LlamaSwaType,
    /// `dsv4_make_k_only` (llama-kv-cache-dsv4.cpp:1253): the dsv4 raw pair
    /// carries no V storage (`layer.v_stream == nullptr`), so
    /// `state_write_data`'s value loops skip entirely (:2278-2297
    /// `if (!v) continue`). The port's cache always allocates v tensors; the
    /// serialization just leaves them out, exactly like the C's bytes.
    k_only: bool,
}

impl KvCache {
    /// the base-cache view (`kv_base`): every `!is_swa(il)` layer with a KV
    /// row — the 0-wide-K layers (the recurrent/FFN layers of the hybrids,
    /// where the C's `has_kv`/layer-filter continue never creates a
    /// `layers[]` entry, llama-kv-cache.cpp:165-175) are not serialized
    fn raw_view_base(&self, k_only: bool) -> RawCacheView<'_> {
        RawCacheView {
            cells: &self.cells,
            layer_ids: (0..self.layers.len())
                .filter(|&il| !self.layer_is_swa(il) && self.k_row[il] > 0)
                .collect(),
            n_swa: 0,
            swa_type: LlamaSwaType::NONE,
            k_only,
        }
    }

    /// the SWA-cache view (`kv_swa`)
    fn raw_view_swa(&self, k_only: bool) -> Option<RawCacheView<'_>> {
        self.swa.as_ref().map(|s| RawCacheView {
            cells: &s.cells,
            layer_ids: s.layer_ids.clone(),
            n_swa: s.n_swa,
            swa_type: s.swa_type,
            k_only,
        })
    }

    /// the base half's rotation widths — see raw_state_write's note
    fn raw_half_n_rot_base(&self) -> (u32, u32) {
        let has_layers = (0..self.layers.len())
            .any(|il| !self.layer_is_swa(il) && self.k_row[il] > 0);
        if !has_layers {
            return (0, 0);
        }
        (self.n_rot_k, self.n_rot_v)
    }

    /// the SWA half's rotation widths
    fn raw_half_n_rot_swa(&self) -> (u32, u32) {
        let has_layers = (0..self.layers.len())
            .any(|il| self.layer_is_swa(il) && self.k_row[il] > 0);
        if !has_layers {
            return (0, 0);
        }
        (self.n_rot_k, self.n_rot_v)
    }

    fn raw_half_n_rot(&self, view: &RawCacheView) -> (u32, u32) {
        if view.layer_ids.is_empty() {
            return (0, 0);
        }
        (self.n_rot_k, self.n_rot_v)
    }

    /// `llama_kv_cache::state_write` (llama-kv-cache.cpp:2055-2123) over one
    /// raw half, `n_stream == 1`: the stream count, the kept-cell count, the
    /// per-cell metadata (`state_write_meta`, :2205-2236) and the K/V row
    /// ranges (`state_write_data`, :2238-2335, the `!v_trans` arm — the
    /// port's caches are always v_trans = false).
    fn raw_state_write(
        &self,
        view: &RawCacheView,
        io: &mut StateWriter,
        gctx: &Context,
        seq_id: i32,
    ) {
        io.write_u32(1); // n_stream (:2063)

        let mut cell_count = 0u32;
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        {
            let mut range_begin = view.cells.len() as u32;
            let pos_max = if seq_id >= 0 {
                Self::seq_pos_max(view.cells, 1u64 << seq_id)
            } else {
                -1
            };
            for (i, cell) in view.cells.iter().enumerate() {
                let mut add_cell = !cell.is_empty();
                add_cell = add_cell && (seq_id == -1 || cell.seq & (1u64 << seq_id) != 0);

                // check the cell is not SWA-masked (:2082-2087)
                if add_cell && seq_id != -1 {
                    add_cell =
                        !LlamaHparams::is_masked_swa(view.n_swa, view.swa_type, cell.pos, pos_max);
                }

                if add_cell {
                    cell_count += 1;
                    if range_begin == view.cells.len() as u32 {
                        range_begin = i as u32;
                    }
                } else if range_begin != view.cells.len() as u32 {
                    ranges.push((range_begin, i as u32));
                    range_begin = view.cells.len() as u32;
                }
            }
            if range_begin != view.cells.len() as u32 {
                ranges.push((range_begin, view.cells.len() as u32));
            }
        }

        io.write_u32(cell_count);
        if cell_count == 0 {
            return; // skip empty streams (:2116-2118)
        }

        // state_write_meta (:2205-2236): pos, n_seq_id, the seq ids (in id
        // order, restricted to the saved sequence)
        for &(b, e) in &ranges {
            for i in b..e {
                let cell = &view.cells[i as usize];
                let seq_ids: Vec<i32> = (0..crate::batch::LLAMA_MAX_SEQ as i32)
                    .filter(|&cur| (cur == seq_id || seq_id == -1) && cell.seq & (1u64 << cur) != 0)
                    .collect();

                io.write_i32(cell.pos);
                io.write_u32(seq_ids.len() as u32);
                for s in seq_ids {
                    io.write_i32(s);
                }
            }
        }

        // state_write_data (:2238-2270 keys, :2272-2297 values, v_trans = 0)
        io.write_u32(0); // v_trans
        io.write_u32(view.layer_ids.len() as u32);
        // 210791069 (llama-kv-cache.cpp:2258-2259): persist the exact
        // rotation widths so a mismatched restore is rejected. The width is
        // per cache (each iswa half is its own llama_kv_cache): a half with
        // no participating layers has n_embd_head_k_all == 0 and no
        // rotation. NOTE: upstream @c35b66744 crashes on that half — the
        // DeepSeek-indexer clause of :321-332 does not re-check
        // n_embd_head_k_all > 0, so the `while (0 % (2*n_rot_k) == 0)`
        // doubling overflows the u32 divisor to 0 (SIGFPE, hit by the dsv4
        // state probe on the all-SWA base half). The port writes the
        // guarded value (0), which is what a fixed upstream produces.
        let (n_rot_k, n_rot_v) = self.raw_half_n_rot(view);
        io.write_u32(n_rot_k);
        io.write_u32(n_rot_v);

        for &il in &view.layer_ids {
            let k = self.layers[il].k;
            let k_type_i = gctx.ty(k) as i32;
            let k_size_row = gctx.ty(k).row_size(self.k_row[il] as usize) as u64;
            io.write_i32(k_type_i);
            io.write_u64(k_size_row);
            for &(b, e) in &ranges {
                io.write_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                );
            }
        }
        // the value loops — skipped for a k-only cache (`if (!v) continue`,
        // :2278/:2589; the dsv4 raw pair never allocates them,
        // dsv4_make_k_only :1253)
        if !view.k_only {
            for &il in &view.layer_ids {
                let v = self.layers[il].v;
                let v_type_i = gctx.ty(v) as i32;
                let v_size_row = gctx.ty(v).row_size(self.v_row[il] as usize) as u64;
                io.write_i32(v_type_i);
                io.write_u64(v_size_row);
                for &(b, e) in &ranges {
                    io.write_tensor(
                        gctx,
                        v,
                        b as usize * v_size_row as usize,
                        (e - b) as usize * v_size_row as usize,
                    );
                }
            }
        }
    }

    /// `llama_kv_cache::state_read_sinfo` (llama-kv-cache.cpp:2125-2203) over
    /// one raw half (single-stream, no mirrored layouts — the port has no
    /// `sinfos_in` consumers). Returns the destination cell indices the data
    /// half must scatter into.
    #[allow(clippy::type_complexity)]
    fn raw_state_read(
        &mut self,
        swa_half: bool,
        io: &mut StateReader,
        gctx: &mut Context,
        seq_id: i32,
        k_only: bool,
    ) -> Result<Vec<u32>, String> {
        let n_stream_cur = io.read_u32()?;
        if n_stream_cur != 1 {
            return Err("n_stream mismatch".into());
        }

        let cell_count = io.read_u32()?;
        if cell_count == 0 {
            return Ok(Vec::new());
        }

        let idxs: Vec<u32> = if seq_id >= 0 {
            // state_read_meta's single-sequence arm (:2341-2450): reject an
            // over-large restore BEFORE the sequence is dropped — a failed
            // read must leave the cache as it was (llama-kv-cache.cpp:
            // 2346-2350, 08618ff8e)
            {
                let size = if swa_half {
                    self.swa.as_ref().map(|s| s.size).unwrap_or(self.size)
                } else {
                    self.size
                };
                if cell_count > size {
                    return Err("not enough cells in kv cache".into());
                }
            }
            // ... then drop the sequence from THIS half only
            // (`seq_rm(dest_seq_id, -1, -1)` on that one cache, :2343 — the
            // port's composite seq_rm would wipe the other half's freshly
            // restored cells), then place the restored tokens like a fresh
            // ubatch
            {
                let mask = 1u64 << seq_id;
                if swa_half {
                    let s = self.swa.as_mut().unwrap();
                    for c in s.cells.iter_mut() {
                        if !c.is_empty() && c.seq & mask != 0 {
                            c.seq &= !mask;
                            if c.seq == 0 {
                                *c = KvCell::default();
                            }
                        }
                    }
                } else {
                    for c in self.cells.iter_mut() {
                        if !c.is_empty() && c.seq & mask != 0 {
                            c.seq &= !mask;
                            if c.seq == 0 {
                                *c = KvCell::default();
                            }
                        }
                    }
                }
            }

            let mut positions = Vec::with_capacity(cell_count as usize);
            for _ in 0..cell_count {
                let pos = io.read_i32()?;
                let n_seq_id = io.read_u32()?;
                if n_seq_id != 1 {
                    return Err("invalid seq_id-agnostic kv cell".into());
                }
                // read the sequence id, but directly discard it — dest_seq_id
                // is used instead (:2384-2388)
                let _ = io.read_i32()?;
                positions.push(pos);
            }

            // find_slot(ubatch, false) over this half's cells
            let idxs = if swa_half {
                self.find_slot_swa(cell_count)
            } else {
                self.find_slot(cell_count).map(|s| (s.s0..=s.s1).collect())
            };
            let Some(idxs) = idxs else {
                return Err(format!(
                    "failed to find {cell_count} available cells in kv cache"
                ));
            };

            if swa_half {
                self.assign_swa(&idxs, &positions, seq_id as usize);
            } else {
                let sinfo = SlotInfo {
                    s0: idxs[0],
                    s1: idxs[idxs.len() - 1],
                };
                self.assign(sinfo, &positions, seq_id as usize);
            }
            idxs
        } else {
            // the whole-cache arm (:2451-2505): cells go in from 0. clear()
            // empties both halves + lid (the C's per-cache clear runs once per
            // stream-replacement, :2159-2163); the restored cells land in this
            // half's own storage.
            self.clear();
            let size = if swa_half {
                self.swa.as_ref().map(|s| s.size).unwrap_or(self.size)
            } else {
                self.size
            };
            if cell_count > size {
                return Err("not enough cells in kv cache".into());
            }
            for i in 0..cell_count as usize {
                let pos = io.read_i32()?;
                let n_seq_id = io.read_u32()?;
                if swa_half {
                    let cell = &mut self.swa.as_mut().unwrap().cells[i];
                    cell.pos = pos;
                    cell.seq = 0;
                    for _ in 0..n_seq_id {
                        let s = io.read_i32()?;
                        if s < 0 || s as usize >= crate::batch::LLAMA_MAX_SEQ {
                            return Err(format!("invalid seq_id, {s} is out of range"));
                        }
                        cell.seq |= 1u64 << s;
                    }
                } else {
                    let cell = &mut self.cells[i];
                    cell.pos = pos;
                    cell.seq = 0;
                    for _ in 0..n_seq_id {
                        let s = io.read_i32()?;
                        if s < 0 || s as usize >= crate::batch::LLAMA_MAX_SEQ {
                            return Err(format!("invalid seq_id, {s} is out of range"));
                        }
                        cell.seq |= 1u64 << s;
                    }
                }
            }
            // head restarts at 0 (:2504)
            if swa_half {
                self.swa.as_mut().unwrap().head = 0;
            } else {
                self.head = 0;
            }
            (0..cell_count).collect()
        };

        // state_read_data (:2510-2628): scatter per contiguous run of the
        // destination indices
        let runs: Vec<(u32, u32)> = contig_runs(&idxs);

        // state_read_data (:2510-2628) — a failure past this point must undo
        // the restore (`state_clear`, llama-kv-cache.cpp:2196-2201 +
        // 2694-2783, 08618ff8e): drop the sequence again and zero the K/V
        // rows the failed attempt touched — the attention can still read the
        // data of free cells
        if let Err(e) = self.raw_state_read_data(swa_half, io, gctx, k_only, cell_count, &runs) {
            self.raw_state_clear(swa_half, gctx, seq_id, &runs, k_only);
            return Err(e);
        }

        Ok(idxs)
    }

    /// the data half of `raw_state_read` (`state_read_data`,
    /// llama-kv-cache.cpp:2510-2628): the v_trans/layer-count checks then the
    /// K/V row scatters.
    fn raw_state_read_data(
        &mut self,
        swa_half: bool,
        io: &mut StateReader,
        gctx: &mut Context,
        k_only: bool,
        cell_count: u32,
        runs: &[(u32, u32)],
    ) -> Result<(), String> {
        let v_trans = io.read_u32()?;
        let n_layer = io.read_u32()?;
        // 210791069 (llama-kv-cache.cpp:2552-2556): the rotation widths ride
        // the blob; a mismatch is fatal (:2575-2584)
        let n_rot_k_ref = io.read_u32()?;
        let n_rot_v_ref = io.read_u32()?;

        let layer_ids: Vec<usize> = if swa_half {
            self.swa
                .as_ref()
                .map(|s| s.layer_ids.clone())
                .unwrap_or_default()
        } else {
            // the writer's filter: !is_swa layers with a KV row (the C's
            // layers[] excludes the has_kv=false/filtered layers,
            // llama-kv-cache.cpp:165-175 — state_write_data iterates
            // `layers` only, :2247-2251)
            (0..self.layers.len())
                .filter(|&il| !self.layer_is_swa(il) && self.k_row[il] > 0)
                .collect()
        };
        if n_layer != layer_ids.len() as u32 {
            return Err(format!(
                "mismatched layer count ({n_layer} instead of {})",
                layer_ids.len()
            ));
        }
        if cell_count as usize > self.cells.len() && !swa_half {
            return Err("not enough cells in kv cache to restore state".into());
        }
        if v_trans != 0 {
            // the port's caches are always the C's !v_trans layout; the C
            // errors the same way against a v_trans=0 cache
            return Err("incompatible V transposition".into());
        }
        // (:2575-2584) exact rotation metadata, a mismatched restore is
        // rejected — per half, see raw_state_write's note
        let (n_rot_k, n_rot_v) = if swa_half {
            self.raw_half_n_rot_swa()
        } else {
            self.raw_half_n_rot_base()
        };
        if n_rot_k_ref != n_rot_k {
            return Err(format!(
                "incompatible key rotation ({n_rot_k_ref} instead of {n_rot_k})"
            ));
        }
        if n_rot_v_ref != n_rot_v {
            return Err(format!(
                "incompatible value rotation ({n_rot_v_ref} instead of {n_rot_v})"
            ));
        }

        // keys, one row per cell (:2552-2581)
        for &il in &layer_ids {
            let k = self.layers[il].k;
            let k_type_i_ref = io.read_i32()?;
            if gctx.ty(k) as i32 != k_type_i_ref {
                return Err(format!("mismatched key type (layer {il})"));
            }
            let k_size_row_ref = io.read_u64()?;
            let k_size_row = gctx.ty(k).row_size(self.k_row[il] as usize) as u64;
            if k_size_row != k_size_row_ref {
                return Err(format!("mismatched key row size (layer {il})"));
            }
            for &(b, e) in runs {
                io.read_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                )?;
            }
        }
        // values (:2583-2614 of the !v_trans arm) — absent for a k-only
        // cache (the writer's `if (!v) continue`, :2278/:2589)
        if !k_only {
            for &il in &layer_ids {
                let v = self.layers[il].v;
                let v_type_i_ref = io.read_i32()?;
                if gctx.ty(v) as i32 != v_type_i_ref {
                    return Err(format!("mismatched value type (layer {il})"));
                }
                let v_size_row_ref = io.read_u64()?;
                let v_size_row = gctx.ty(v).row_size(self.v_row[il] as usize) as u64;
                if v_size_row != v_size_row_ref {
                    return Err(format!("mismatched value row size (layer {il})"));
                }
                for &(b, e) in runs {
                    io.read_tensor(
                        gctx,
                        v,
                        b as usize * v_size_row as usize,
                        (e - b) as usize * v_size_row as usize,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// `llama_kv_cache::state_clear` (llama-kv-cache.cpp:2677-2783,
    /// 08618ff8e): undo a state_read() of seq_id (-1 for the whole cache)
    /// that a later read failed to complete — the cleared ranges mirror the
    /// write pattern of state_read_data(), keep both in sync.
    fn raw_state_clear(
        &mut self,
        swa_half: bool,
        gctx: &mut Context,
        seq_id: i32,
        runs: &[(u32, u32)],
        k_only: bool,
    ) {
        let layer_ids: Vec<usize> = if swa_half {
            self.swa
                .as_ref()
                .map(|s| s.layer_ids.clone())
                .unwrap_or_default()
        } else {
            (0..self.layers.len())
                .filter(|&il| !self.layer_is_swa(il) && self.k_row[il] > 0)
                .collect()
        };

        if seq_id == -1 {
            // clear(true): drop every cell of this half...
            if swa_half {
                let s = self.swa.as_mut().unwrap();
                s.cells.fill(KvCell::default());
                s.head = 0;
            } else {
                self.cells.fill(KvCell::default());
                self.head = 0;
            }
            // ...and zero its whole K/V storage
            let full = vec![(0u32, self.size)];
            for &il in &layer_ids {
                let k = self.layers[il].k;
                let k_size_row = gctx.ty(k).row_size(self.k_row[il] as usize);
                zero_tensor_rows(gctx, k, k_size_row, &full);
                if !k_only {
                    let v = self.layers[il].v;
                    let v_size_row = gctx.ty(v).row_size(self.v_row[il] as usize);
                    zero_tensor_rows(gctx, v, v_size_row, &full);
                }
            }
            return;
        }

        // seq_rm(seq_id, -1, -1) of this half (the same strip the read's meta
        // phase does)
        {
            let mask = 1u64 << seq_id;
            let cells = if swa_half {
                let s = self.swa.as_mut().unwrap();
                &mut s.cells
            } else {
                &mut self.cells
            };
            for c in cells.iter_mut() {
                if !c.is_empty() && c.seq & mask != 0 {
                    c.seq &= !mask;
                    if c.seq == 0 {
                        *c = KvCell::default();
                    }
                }
            }
        }

        // zero the K/V data of the failed restore attempt
        if runs.is_empty() {
            return;
        }

        for &il in &layer_ids {
            let k = self.layers[il].k;
            let k_size_row = gctx.ty(k).row_size(self.k_row[il] as usize);
            zero_tensor_rows(gctx, k, k_size_row, runs);
        }
        if !k_only {
            for &il in &layer_ids {
                let v = self.layers[il].v;
                let v_size_row = gctx.ty(v).row_size(self.v_row[il] as usize);
                zero_tensor_rows(gctx, v, v_size_row, runs);
            }
        }
    }

    /// `llama_kv_cache_dsa::state_write`'s lid half (llama-kv-cache-dsa.cpp:
    /// 164-171): `kv_lid->state_write` — a plain `llama_kv_cache::state_write`
    /// over the lid cache's own cells and indexer-key rows. K-only
    /// (`hparams_lid` is a copy of the MLA hparams, so `has_v = !is_mla()`
    /// is false here too, llama-kv-cache.cpp:230-234); `n_swa = 0` /
    /// `LLAMA_SWA_TYPE_NONE` (llama-kv-cache-dsa.cpp:50-51), so no cell is
    /// ever SWA-masked.
    fn lid_state_write(&self, io: &mut StateWriter, gctx: &Context, seq_id: i32) {
        let lid = self
            .lid
            .as_ref()
            .expect("lid_state_write without a lid cache");
        io.write_u32(1); // n_stream

        // the kept-cell ranges over the lid's own (lockstep) cells
        let mut cell_count = 0u32;
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        {
            let mut range_begin = lid.cells.len() as u32;
            for (i, cell) in lid.cells.iter().enumerate() {
                let mut add_cell = !cell.is_empty();
                add_cell = add_cell && (seq_id == -1 || cell.seq & (1u64 << seq_id) != 0);
                if add_cell {
                    cell_count += 1;
                    if range_begin == lid.cells.len() as u32 {
                        range_begin = i as u32;
                    }
                } else if range_begin != lid.cells.len() as u32 {
                    ranges.push((range_begin, i as u32));
                    range_begin = lid.cells.len() as u32;
                }
            }
            if range_begin != lid.cells.len() as u32 {
                ranges.push((range_begin, lid.cells.len() as u32));
            }
        }

        io.write_u32(cell_count);
        if cell_count == 0 {
            return;
        }

        // state_write_meta over the lid cells
        for &(b, e) in &ranges {
            for i in b..e {
                let cell = &lid.cells[i as usize];
                let seq_ids: Vec<i32> = (0..crate::batch::LLAMA_MAX_SEQ as i32)
                    .filter(|&cur| (cur == seq_id || seq_id == -1) && cell.seq & (1u64 << cur) != 0)
                    .collect();
                io.write_i32(cell.pos);
                io.write_u32(seq_ids.len() as u32);
                for s in seq_ids {
                    io.write_i32(s);
                }
            }
        }

        // state_write_data — the indexer-key rows only (k_only)
        io.write_u32(0); // v_trans
        io.write_u32(lid.layers.len() as u32);
        for &k in &lid.layers {
            let k_type_i = gctx.ty(k) as i32;
            let k_size_row = gctx.ty(k).row_size(lid.n_embd_head_k as usize) as u64;
            io.write_i32(k_type_i);
            io.write_u64(k_size_row);
            for &(b, e) in &ranges {
                io.write_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                );
            }
        }
    }

    /// the lid half's `state_read` counterpart — mirrors [`KvCache::
    /// raw_state_read`]'s single-sequence placement over the lid's own cells
    /// (identical to the base's placement: both caches were in lockstep
    /// before the save and the same ubatches replay into both,
    /// llama-kv-cache-dsa.cpp:131-139).
    fn lid_state_read(
        &mut self,
        io: &mut StateReader,
        gctx: &mut Context,
        seq_id: i32,
    ) -> Result<(), String> {
        let n_stream_cur = io.read_u32()?;
        if n_stream_cur != 1 {
            return Err("n_stream mismatch (lid)".into());
        }
        let cell_count = io.read_u32()?;
        if cell_count == 0 {
            return Ok(());
        }

        let idxs: Vec<u32> = if seq_id >= 0 {
            // state_read_meta's single-sequence arm on the lid cells
            {
                let mask = 1u64 << seq_id;
                let lid = self.lid.as_mut().unwrap();
                for c in lid.cells.iter_mut() {
                    if !c.is_empty() && c.seq & mask != 0 {
                        c.seq &= !mask;
                        if c.seq == 0 {
                            *c = KvCell::default();
                        }
                    }
                }
            }
            let mut positions = Vec::with_capacity(cell_count as usize);
            for _ in 0..cell_count {
                let pos = io.read_i32()?;
                let n_seq_id = io.read_u32()?;
                if n_seq_id != 1 {
                    return Err("invalid seq_id-agnostic kv cell (lid)".into());
                }
                let _ = io.read_i32()?; // discarded — dest_seq_id wins
                positions.push(pos);
            }
            // find_slot over the lid cells (first-fit, the lockstep twin of
            // the base cache's placement)
            let lid = self.lid.as_ref().unwrap();
            let mut s0 = None;
            if (cell_count as usize) <= lid.cells.len() {
                for start in 0..=lid.cells.len() - cell_count as usize {
                    if lid.cells[start..start + cell_count as usize]
                        .iter()
                        .all(|c| c.is_empty())
                    {
                        s0 = Some(start as u32);
                        break;
                    }
                }
            }
            let Some(s0) = s0 else {
                return Err(format!(
                    "failed to find {cell_count} available cells in the lid cache"
                ));
            };
            let idxs: Vec<u32> = (s0..s0 + cell_count).collect();
            let lid = self.lid.as_mut().unwrap();
            for (i, &idx) in idxs.iter().enumerate() {
                lid.cells[idx as usize] = KvCell {
                    pos: positions[i],
                    seq: 1u64 << seq_id,
                    shift: 0,
                    tok: -1,
                };
            }
            idxs
        } else {
            // the whole-cache arm: cells go in from 0
            let lid = self.lid.as_mut().unwrap();
            for c in lid.cells.iter_mut() {
                *c = KvCell::default();
            }
            if cell_count as usize > lid.cells.len() {
                return Err("not enough cells in the lid cache".into());
            }
            for i in 0..cell_count as usize {
                let pos = io.read_i32()?;
                let n_seq_id = io.read_u32()?;
                let mut seq = 0u64;
                for _ in 0..n_seq_id {
                    let s = io.read_i32()?;
                    if s < 0 || s as usize >= crate::batch::LLAMA_MAX_SEQ {
                        return Err(format!("invalid seq_id, {s} is out of range"));
                    }
                    seq |= 1u64 << s;
                }
                lid.cells[i] = KvCell { pos, seq, shift: 0, tok: -1 };
            }
            lid.head = 0;
            (0..cell_count).collect()
        };

        // state_read_data — the contiguous-run scatter of the indexer rows
        let runs: Vec<(u32, u32)> = {
            let mut runs = Vec::new();
            let mut i0 = 0usize;
            while i0 < idxs.len() {
                let mut i1 = i0 + 1;
                while i1 < idxs.len() && idxs[i1] == idxs[i1 - 1] + 1 {
                    i1 += 1;
                }
                runs.push((idxs[i0], idxs[i1 - 1] + 1));
                i0 = i1;
            }
            runs
        };

        let v_trans = io.read_u32()?;
        let n_layer = io.read_u32()?;
        let n_layers = self.lid.as_ref().map(|l| l.layers.len()).unwrap_or(0);
        if n_layer != n_layers as u32 {
            return Err(format!(
                "mismatched layer count ({n_layer} instead of {n_layers}, lid)"
            ));
        }
        if v_trans != 0 {
            return Err("incompatible V transposition (lid)".into());
        }

        let lid = self.lid.as_ref().unwrap();
        for &k in &lid.layers {
            let k_type_i_ref = io.read_i32()?;
            if gctx.ty(k) as i32 != k_type_i_ref {
                return Err("mismatched key type (lid)".into());
            }
            let k_size_row_ref = io.read_u64()?;
            let k_size_row = gctx.ty(k).row_size(lid.n_embd_head_k as usize) as u64;
            if k_size_row != k_size_row_ref {
                return Err("mismatched key row size (lid)".into());
            }
            for &(b, e) in &runs {
                io.read_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                )?;
            }
        }
        Ok(())
    }

    /// `llama_state_seq_write` for the cache kinds the port has — the
    /// `memory->state_write` of `llama_context::state_seq_write_data`
    /// (llama-context.cpp:3395-3401):
    ///
    ///  * a dsv4 cache (deepseek4) serializes the full
    ///    `llama_kv_cache_dsv4::state_write` frame of :1594-1626 (magic /
    ///    version / mode, the raw iswa pair, the three compressed K caches,
    ///    the three compressor states);
    ///  * a dsa pair (deepseek32 / dots3note) serializes the MLA base half
    ///    K-only plus the indexer-key lid half
    ///    (`llama_kv_cache_dsa::state_write`, llama-kv-cache-dsa.cpp:164-172
    ///    — `llama_kv_cache_dsa_iswa::state_write`, llama-kv-cache-dsa-iswa.
    ///    cpp:236-242, appends the swa half and gates the pair on !partial);
    ///  * the MSA idx half (minimax-m3) is serialized by the DecodeContext
    ///    wrapper (the port's idx cache lives there, not in this struct);
    ///  * a plain / iswa cache serializes just the raw half(s) of
    ///    `llama_kv_cache::state_write` (:2055) — base then swa
    ///    (llama-kv-cache-iswa.cpp:259-264);
    ///  * the `partial_only` flag (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`)
    ///    skips the compressed halves (the raw cells only), :1595/:1607.
    pub fn state_seq_write(
        &self,
        io: &mut StateWriter,
        gctx: &Context,
        seq_id: i32,
        partial_only: bool,
    ) -> Result<(), String> {
        if let Some(d) = &self.dsv4 {
            // llama_kv_cache_dsv4::state_write (:1594-1603)
            io.write_u32(DSV4_STATE_MAGIC);
            io.write_u32(DSV4_STATE_VERSION);
            io.write_u32(if partial_only {
                DSV4_STATE_MODE_PARTIAL
            } else {
                DSV4_STATE_MODE_FULL
            });

            // kv_raw->state_write (:1605): base then swa — the dsv4 raw pair
            // is k-only (dsv4_make_k_only, :1253)
            let k_only = true;
            self.raw_state_write(&self.raw_view_base(k_only), io, gctx, seq_id);
            if let Some(view) = self.raw_view_swa(k_only) {
                self.raw_state_write(&view, io, gctx, seq_id);
            }

            if !partial_only {
                // (:1607-1620) — the compressed row counts follow the raw
                // pair's pos_max (FIXME note of the C kept: token positions
                // are conflated with rows)
                let pos_max = if seq_id >= 0 {
                    self.seq_pos_max_of(seq_id as usize)
                } else {
                    -1
                };
                let n_rows = |ratio: u32, cache: &Dsv4CompCache| -> u32 {
                    if seq_id >= 0 {
                        dsv4_state_n_used_k_rows(pos_max, ratio, cache.size)
                    } else {
                        cache.size
                    }
                };

                d.csa
                    .state_write_k_cache(io, gctx, seq_id, n_rows(DSV4_CSA_RATIO, &d.csa))?;
                d.hca
                    .state_write_k_cache(io, gctx, seq_id, n_rows(DSV4_HCA_RATIO, &d.hca))?;
                d.lid
                    .state_write_k_cache(io, gctx, seq_id, n_rows(DSV4_CSA_RATIO, &d.lid))?;
            }

            // (:1623-1625)
            d.csa_state.state_write(io, gctx, seq_id, &d.rs_idx)?;
            d.hca_state.state_write(io, gctx, seq_id, &d.rs_idx)?;
            d.lid_state.state_write(io, gctx, seq_id, &d.rs_idx)?;
            return Ok(());
        }

        // the dsa pair (deepseek32 / dots3note): the MLA base half is K-only
        // (is_mla ⇒ has_v = false, llama-kv-cache.cpp:230-234 — model-wide,
        // so the swa half of the dsa_iswa combo is K-only too), then the lid
        // half (llama_kv_cache_dsa::state_write, llama-kv-cache-dsa.cpp:
        // 164-166). dots3note's `llama_kv_cache_dsa_iswa::state_write`
        // (:236-242) appends the swa half after the pair and skips the pair
        // entirely under partial_only.
        if self.lid.is_some() {
            if !partial_only || self.swa.is_none() {
                // (the plain dsa of deepseek32 ignores the flags — GGML_UNUSED)
                self.raw_state_write(&self.raw_view_base(true), io, gctx, seq_id);
                self.lid_state_write(io, gctx, seq_id);
            }
            if let Some(view) = self.raw_view_swa(true) {
                self.raw_state_write(&view, io, gctx, seq_id);
            }
            return Ok(());
        }

        // the plain / iswa shape: the raw half(s) alone (with their V rows
        // unless the cache is a K-only MLA one)
        self.raw_state_write(&self.raw_view_base(self.k_only), io, gctx, seq_id);
        if let Some(view) = self.raw_view_swa(self.k_only) {
            self.raw_state_write(&view, io, gctx, seq_id);
        }
        Ok(())
    }

    /// the `state_seq_read` counterpart (llama-context.cpp:3403-3408 →
    /// `memory->state_read`). Returns the bytes consumed.
    pub fn state_seq_read(
        &mut self,
        io: &mut StateReader,
        gctx: &mut Context,
        seq_id: i32,
        partial_only: bool,
    ) -> Result<(), String> {
        if self.dsv4.is_some() {
            // llama_kv_cache_dsv4::state_read (:1628-1651)
            let magic = io.read_u32()?;
            let version = io.read_u32()?;
            if magic != DSV4_STATE_MAGIC {
                return Err("DSV4 state magic mismatch".into());
            }
            if version != DSV4_STATE_VERSION {
                return Err("DSV4 state version mismatch".into());
            }
            let mode = io.read_u32()?;
            if mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL {
                return Err("DSV4 state mode mismatch".into());
            }
            let blob_partial = mode == DSV4_STATE_MODE_PARTIAL;
            if blob_partial != partial_only {
                return Err("DSV4 state flags mismatch".into());
            }

            // kv_raw->state_read (:1653) — the dsv4 raw pair is k-only
            self.raw_state_read(false, io, gctx, seq_id, true)?;
            if self.swa.is_some() {
                self.raw_state_read(true, io, gctx, seq_id, true)?;
            }

            if !blob_partial {
                // clear_compressed(seq_id, true) (:1656) — the restored rows
                // replace the streams' compressed halves wholesale
                if seq_id >= 0 {
                    self.clear_dsv4_seq(gctx, seq_id as u32);
                } else {
                    self.clear_dsv4(gctx);
                }

                let d = self.dsv4.as_mut().unwrap();
                d.csa.state_read_k_cache(io, gctx, seq_id)?;
                d.hca.state_read_k_cache(io, gctx, seq_id)?;
                d.lid.state_read_k_cache(io, gctx, seq_id)?;
            }

            // the comp states (:1663-1665) — the tensor ids are Copy, so the
            // shared borrow of the cache coexists with the mutable gctx
            {
                let d = self.dsv4.as_ref().unwrap();
                d.csa_state.state_read(io, gctx, seq_id)?;
                d.hca_state.state_read(io, gctx, seq_id)?;
                d.lid_state.state_read(io, gctx, seq_id)?;
            }

            // (:1667-1672) — the restore invalidates any pending rollback
            let d = self.dsv4.as_mut().unwrap();
            if seq_id >= 0 {
                assert!((seq_id as usize) < d.rs_idx.len());
                d.rs_idx[seq_id as usize] = 0;
            } else {
                d.rs_idx.fill(0);
            }
            return Ok(());
        }

        // the dsa pair's read (llama_kv_cache_dsa::state_read, :169-181 →
        // the write order): base then lid, then the dsa_iswa swa half.
        // The lid read is guarded: the MLA part is already restored — undo
        // it, so that a failed restore leaves nothing behind
        // (llama-kv-cache-dsa.cpp:171-181, 08618ff8e)
        if self.lid.is_some() {
            if !partial_only || self.swa.is_none() {
                match self.raw_state_read(false, io, gctx, seq_id, true) {
                    Err(e) => return Err(e), // the base half cleaned itself up
                    Ok(idxs) => {
                        if let Err(e) = self.lid_state_read(io, gctx, seq_id) {
                            let runs = contig_runs(&idxs);
                            self.raw_state_clear(false, gctx, seq_id, &runs, true);
                            return Err(e);
                        }
                    }
                }
            }
            if self.swa.is_some() {
                self.raw_state_read(true, io, gctx, seq_id, true)?;
            }
            return Ok(());
        }

        // the plain / iswa shape
        self.raw_state_read(false, io, gctx, seq_id, self.k_only)?;
        if self.swa.is_some() {
            self.raw_state_read(true, io, gctx, seq_id, self.k_only)?;
        }
        Ok(())
    }

    /// `llama_state_seq_get_size` (llama-context.cpp:3144-3155): run the
    /// writer in dummy mode.
    pub fn state_seq_size(&self, gctx: &Context, seq_id: i32, partial_only: bool) -> usize {
        let mut io = StateWriter::new(true);
        self.state_seq_write(&mut io, gctx, seq_id, partial_only)
            .expect("state_seq_size");
        io.n_bytes()
    }
}

// ===========================================================================
// llama_memory_hybrid_idx — the third cache of the indexed hybrid memory
// (src/llama-memory-hybrid-idx.{cpp,h}, def4d406a). The port's shape follows
// the MSA precedent (`MinimaxMsaCache`): the indexer is a side buffer over
// the *attention* cache's cells — "the indexer uses the attention cache's
// slot layout; a separate one can drift from it" (llama-memory-hybrid-idx.
// cpp:127) — so this struct keeps the indexer key rows + the k-pool layout
// bookkeeping while `KvCache::cells` stays the lockstep twin (one token per
// cell in both).
//
// The reference's `hparams_idx` (llama-memory-hybrid-idx.cpp:52-65):
//   * MQA with a single key head (`n_head_kv = 1`) of
//     `n_embd_head_k_full = indexer_head_size * (kpool > 0 ? kpool_row : 1)`
//     (a7b94df2c:58) — a k-pool indexer caches its per-token rows and the
//     pooled key side by side (glm5-next: key | gate | pooled = 3 slots,
//     qwen4exp: key | pooled = 2), a kpool == 0 indexer (qwen4exp's QSA
//     files) caches the raw indexer key only;
//   * `rope_type = NONE` — cached indexer keys are raw, rotation happens
//     after pooling at read time (glm5-next.cpp:60-61's comment);
//   * MLA-fooled (`n_embd_head_{k,v}_mla_impl = indexer_head_size`) so the
//     cache holds no V tensors — K-only rows.
// ===========================================================================

/// `llama_memory_hybrid_idx::POS_CLEAN` (llama-memory-hybrid-idx.h:111)
pub const HYBRID_IDX_POS_CLEAN: i32 = i32::MAX;

/// `stale_pos_t` (llama-memory-hybrid-idx.h:109) — one stale position per
/// sequence, POS_CLEAN = none.
pub type HybridIdxStale = [i32; crate::batch::LLAMA_MAX_SEQ];

/// `llama_memory_hybrid_idx::stale_pos_clean` (:113-117)
pub fn hybrid_idx_stale_clean() -> HybridIdxStale {
    [HYBRID_IDX_POS_CLEAN; crate::batch::LLAMA_MAX_SEQ]
}

/// `llama_memory_hybrid_idx::kpool_layout::seq`
/// (llama-memory-hybrid-idx.cpp:674-685)
#[derive(Clone, Default)]
pub struct KpoolLayoutSeq {
    pub pos_min: i32,
    /// stream holding this sequence's cells — the port's single-stream cache
    /// (n_stream == 1, unified) pins it to 0
    pub strm: u32,
    /// position and stream-local cell pairs, sorted by position
    pub cells: Vec<(i32, u32)>,
    pub pools: Vec<u32>,
    /// where the pool scan stopped, so an append resumes instead of starting
    /// over (:680-681)
    pub j_next: usize,
    // (43fe9c642 dropped the `shared` member — rep_gen marks each rep once
    // instead of ruling out caching for shared sequences)
}

/// `llama_memory_hybrid_idx::kpool_layout` (:673-691)
#[derive(Clone, Default)]
pub struct KpoolLayout {
    pub seqs: Vec<KpoolLayoutSeq>,
    pub n_pool_real: u32,
}

/// `llama_memory_hybrid_idx_context::kpool_state` (:694-701 + a7b94df2c:379)
/// — which pools of the layout the current ubatch must re-pool, in the
/// layout's pool order.
#[derive(Clone)]
pub struct KpoolState {
    pub is_new: Vec<u32>,
    pub generation: u32,
    pub n_pool_real: u32,
    pub n_new: u32,
    /// graph size of the new pool list, stable across decode steps
    /// (a7b94df2c:379)
    pub n_new_g: u32,
    /// `rep_gen` (43fe9c642, llama-memory-hybrid-idx.cpp:702-703) — per
    /// global cell, the generation that last marked a pool with that rep:
    /// a pool is marked once per rep so sequences sharing cells (a seq_cp,
    /// or tokens decoded for several sequences) never scatter two rows into
    /// one rep (the CPU-backend data race the commit fixes)
    pub rep_gen: Vec<u32>,
}

impl Default for KpoolState {
    /// the C's member initialisers (:375-380): n_new_g = 1, the rest zero
    fn default() -> Self {
        KpoolState {
            is_new: Vec::new(),
            generation: 0,
            n_pool_real: 0,
            n_new: 0,
            n_new_g: 1,
            rep_gen: Vec::new(),
        }
    }
}

/// `kpool_rank` (a7b94df2c, llama-memory-hybrid-idx.cpp:391-395) — rank of
/// (pos, cell) in a sequence's cells sorted by position then cell, or -1
/// when absent. In order mode the rank alone places a token: cells sharing a
/// position (M-RoPE images) have distinct ranks.
fn kpool_rank(cells: &[(i32, u32)], pos: i32, cell: u32) -> i64 {
    // std::lower_bound on the (pos, cell) pair
    let it = cells.partition_point(|&c| c < (pos, cell));
    if it != cells.len() && cells[it].1 == cell && cells[it].0 == pos {
        it as i64
    } else {
        -1
    }
}

/// `llama_memory_hybrid_idx`'s idx cache half + its kpool layout.
pub struct HybridIdxCache {
    /// per model layer, the indexer key tensor `[row_w, size]` F16 (K-only;
    /// only the `is_idx` layers' entries are ever read)
    pub layers: Vec<TensorId>,
    /// the layer filter (`filter_idx`, llama-model.cpp:2456-2458/2702-2706)
    pub is_idx: Vec<bool>,
    pub size: u32,
    /// `hparams_idx.n_embd_head_k_full` =
    /// indexer_head_size * (kpool>0 ? kpool_row : 1)
    /// (llama-memory-hybrid-idx.cpp:57-58, a7b94df2c)
    pub row_w: i64,
    /// `model.hparams.indexer_head_size` — hparams_idx's mla head dims
    pub head_size: i64,
    /// `get_kpool()` = `hparams.indexer_kpool` (:97) — 0 on qwen4exp's QSA
    /// files (no pooled half of the row)
    pub kpool: u32,
    /// `get_kpool_by_order()` = `hparams.indexer_kpool_by_order`
    /// (llama-memory-hybrid-idx.h:85, a7b94df2c) — pools are kpool
    /// consecutive cells in sequence order (qwen4exp) instead of kpool
    /// consecutive positions
    pub kpool_by_order: bool,
    /// `mem_idx_stale` (:145) — the position each sequence must re-pool from
    pub stale: HybridIdxStale,
    /// `kpool_lay` (:134)
    pub kpool_lay: Option<KpoolLayout>,
    /// `kpool_st` (llama-memory-hybrid-idx.h:239) — the per-ubatch state of
    /// the single live ubatch (the port processes one ubatch per decode)
    pub kpool_st: Option<KpoolState>,
}

impl HybridIdxCache {
    /// the `mem_idx` construction of llama-memory-hybrid-idx.cpp:53-73 — one
    /// F16 key plane per filtered layer, `[kpool_row*indexer_head_size,
    /// kv_size]` on glm5-next (key | gate | pooled, row = 3) and qwen4exp
    /// (key | pooled, row = 2), `[indexer_head_size, kv_size]` when the
    /// indexer pools nothing. The cells are the attention cache's
    /// (lockstep), so no cell array lives here.
    pub fn new(
        ctx: &mut Context,
        size: u32,
        head_size: i64,
        kpool: u32,
        kpool_row: u32,
        kpool_by_order: bool,
        is_idx: Vec<bool>,
    ) -> Self {
        // a k-pool indexer caches its per-token rows and the pooled key side
        // by side (llama-memory-hybrid-idx.cpp:55-58, a7b94df2c)
        let row_w = head_size * if kpool > 0 { kpool_row as i64 } else { 1 };
        let mut layers = Vec::with_capacity(is_idx.len());
        for _ in 0..is_idx.len() {
            let t = ctx.new_tensor_2d(GgmlType::F16, row_w, size as i64);
            ctx.arena_resize_tensor(t);
            layers.push(t);
        }
        HybridIdxCache {
            layers,
            is_idx,
            size,
            row_w,
            head_size,
            kpool,
            kpool_by_order,
            stale: hybrid_idx_stale_clean(),
            kpool_lay: None,
            kpool_st: None,
        }
    }

    /// `get_kpool()` (llama-memory-hybrid-idx.h:97)
    pub fn get_kpool(&self) -> u32 {
        self.kpool
    }

    /// `get_kpool_by_order()` (llama-memory-hybrid-idx.h:85, a7b94df2c) —
    /// whether pools are kpool consecutive cells in sequence order (qwen4exp)
    /// instead of kpool consecutive positions.
    pub fn get_kpool_by_order(&self) -> bool {
        self.kpool_by_order
    }

    /// `mem_idx_stale_set` (llama-memory-hybrid-idx.cpp:159-173) — an edit at
    /// p0 leaves every pool that ends before p0 alone (grouping is sequence
    /// relative).
    pub fn stale_set(&mut self, seq_id: i32, p0: i32) {
        let p0 = p0.max(0);
        if seq_id < 0 {
            for p in self.stale.iter_mut() {
                *p = (*p).min(p0);
            }
            return;
        }
        assert!((0..crate::batch::LLAMA_MAX_SEQ as i32).contains(&seq_id));
        self.stale[seq_id as usize] = self.stale[seq_id as usize].min(p0);
    }

    /// `mem_idx_stale_pos` (:176-182) — an edit at or below the first
    /// position moves pos_min, which regroups the whole sequence. Needs the
    /// base cache's cells (seq_pos_min).
    pub fn stale_pos(&self, base_cells: &[KvCell], seq_id: i32, p0: i32) -> i32 {
        if seq_id < 0 {
            return 0;
        }
        let mask = 1u64 << seq_id;
        let pos_min = KvCache::seq_pos_min(base_cells, mask);
        if p0 <= pos_min {
            return 0;
        }
        p0
    }

    /// `mem_idx_stale_clear` (llama-memory-hybrid-idx.h:120) — only after a
    /// successful ubatch (`next()`, :871-875).
    pub fn stale_clear(&mut self) {
        self.stale = hybrid_idx_stale_clean();
    }

    /// `kpool_layout_update` (llama-memory-hybrid-idx.cpp:726-817) — pools
    /// are fixed by the positions relative to the sequence's first one, so
    /// the layout survives a plain append; `mem_idx_stale` tells us when a
    /// sequence edit regrouped them. The port's single-stream cache folds
    /// `n_stream_kv == 1` (unified): every sequence's `strm` is 0 and the
    /// `!unified` early-out of :746-749 is unreachable.
    pub fn kpool_layout_update(&mut self, base_cells: &[KvCell]) -> &KpoolLayout {
        let kpool = self.kpool as i32;
        if self.kpool_lay.is_none() {
            self.kpool_lay = Some(KpoolLayout {
                seqs: vec![KpoolLayoutSeq::default(); crate::batch::LLAMA_MAX_SEQ],
                n_pool_real: 0,
            });
        }
        let lay = self.kpool_lay.as_mut().unwrap();

        lay.n_pool_real = 0;

        for s in 0..crate::batch::LLAMA_MAX_SEQ {
            let sq = &mut lay.seqs[s];
            let sp: Vec<(i32, u32)> = base_cells
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.is_empty() && c.seq & (1u64 << s) != 0)
                .map(|(i, c)| (c.pos, i as u32))
                .collect();
            // (the C reads `cells.seq_pos_get(s)` — a std::set, already
            // sorted by (pos, cell))

            sq.strm = 0;

            let mut n_kept = 0usize;
            if self.stale[s] == HYBRID_IDX_POS_CLEAN
                && !sq.cells.is_empty()
                && !sp.is_empty()
                && sq.pos_min == sp[0].0
            {
                n_kept = sq.cells.len();
                let last = *sq.cells.last().unwrap();
                for &p in sp.iter().filter(|&&p| p > last) {
                    sq.cells.push(p);
                }
            }

            // the appended tail accounts for every cell only if nothing
            // before it was dropped, but an edit can regroup a sequence
            // without changing its cell count, so a stale sequence must
            // rebuild regardless (:766-774)
            if sq.cells.len() != sp.len() || self.stale[s] != HYBRID_IDX_POS_CLEAN {
                sq.cells = sp.clone();
                sq.pools.clear();
                sq.j_next = 0;
                sq.pos_min = if sp.is_empty() { 0 } else { sp[0].0 };
                n_kept = 0;
            }
            // (43fe9c642 dropped the per-sequence `shared` scan and the
            // layout's cache_safe flag: whole-sequence seq_cp shares the
            // pools themselves, and rep_gen marks each rep once instead)
            let _ = n_kept;

            // pools start at the first valid token (:788-809; the order-mode
            // branch is a7b94df2c:477-481)
            let mut j = sq.j_next;
            if self.kpool_by_order {
                // consecutive cells in sequence order, whatever their positions
                while j + self.kpool as usize <= sq.cells.len() {
                    sq.pools.push(j as u32);
                    j += self.kpool as usize;
                }
            } else {
                while j + self.kpool as usize <= sq.cells.len() {
                    let p0 = sq.cells[j].0;
                    if (p0 - sq.pos_min) % kpool != 0 {
                        j += 1;
                        continue;
                    }
                    let mut ok = true;
                    for k in 1..self.kpool as usize {
                        if sq.cells[j + k].0 != p0 + k as i32 {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        sq.pools.push(j as u32);
                        j += self.kpool as usize;
                    } else {
                        j += 1;
                    }
                }
            }
            sq.j_next = j;

            lay.n_pool_real += sq.pools.len() as u32;
        }

        self.kpool_lay.as_ref().unwrap()
    }

    /// `llama_memory_hybrid_idx_context::kpool_build_sizes`
    /// (a7b94df2c:662-670) — sizes only, for the graph-reserve path (the
    /// caller — the C context's i_kpool==0 branch, :527-535 — then lifts
    /// n_pool_real to the cache's pool capacity and sets n_new/n_new_g to
    /// the full reserve; the port rebuilds per step, so it has no caller).
    pub fn kpool_build_sizes(&self) -> KpoolState {
        let lay = self.kpool_lay.as_ref().expect("kpool_build_sizes");
        KpoolState {
            n_pool_real: lay.n_pool_real,
            ..KpoolState::default()
        }
    }

    /// `llama_memory_hybrid_idx_context::kpool_build_state`
    /// (:997-1066 + a7b94df2c:681-771) — which pools this ubatch must
    /// re-pool. `stale_batch` is the constructor capture of `mem_idx_stale`
    /// (:866); the port's single ubatch per decode folds `i_cur == 0`
    /// (:1030). `ubatch_seqs` carries each token's `seq_id[]` list (the C's
    /// `ubatch.seq_id[i][k]`); `ubatch_cells` is each token's indexer cell
    /// (`sinfos_kpool[i_cur]`, a7b94df2c:556 — only read in order mode, the
    /// port's k_idxs order: token i sits in cell `ubatch_cells[i]`).
    ///
    /// Pool cache lifecycle (the C's comment, :988-996):
    /// 1. cpy_k writes each token's key | gate into its idx cache row,
    ///    pooled slots are zeroed.
    /// 2. This marks the pools the ubatch touches or completes as new —
    ///    during decode that's one pool every kpool tokens, zero elsewise.
    /// 3. The graph pools only the new pools and set_rows each result into
    ///    the pooled slot of the pool's last member row.
    /// 4. All pools are gathered in one get_rows via pool_cells, fresh ones
    ///    just written, older ones from whatever batch last wrote them.
    /// A seq_* edit regroups the pools from the edited position on, so it
    /// stales them and the first ubatch of the next batch rebuilds them from
    /// the still-valid key | gate rows. Orphaned pooled slots are never
    /// cleared — a slot is only ever read through pool_cells, which follows
    /// the current grouping.
    pub fn kpool_build_state(
        &mut self,
        ubatch_pos: &[i32],
        ubatch_seqs: &[Vec<i32>],
        ubatch_cells: &[u32],
        stale_batch: &HybridIdxStale,
    ) {
        let lay = self
            .kpool_lay
            .clone()
            .expect("kpool_build_state: layout first (kpool_layout_update)");
        if self.kpool_st.is_none() {
            self.kpool_st = Some(KpoolState::default());
        }
        let st = self.kpool_st.as_mut().unwrap();

        // (43fe9c642, llama-memory-hybrid-idx.cpp:634-671) the rep_gen
        // table spans the cache's global cells — the port's single-stream
        // cache folds `kv_size*idx->get_n_stream()` to `self.size`
        let kv_size = self.size;
        let kpool_u = self.kpool as usize;

        st.n_pool_real = lay.n_pool_real;
        st.n_new = 0;
        st.generation = st.generation.wrapping_add(1);
        if st.generation == 0 {
            st.is_new.fill(0);
            st.rep_gen.fill(0);
            st.generation = 1;
        }
        st.is_new.resize(lay.n_pool_real as usize, 0);
        st.rep_gen.resize(kv_size as usize, 0);
        let gen = st.generation;

        let mut pool_start = [0u32; crate::batch::LLAMA_MAX_SEQ];

        // a pool is marked once per rep: sequences sharing cells (a seq_cp,
        // or tokens decoded for several sequences) share their pools, whose
        // single pooled row they all read through pool_cells, so the scatter
        // rows stay unique (:656-663)
        let mut mark = |st: &mut KpoolState, s: usize, k: usize, pool_start: &[u32; crate::batch::LLAMA_MAX_SEQ]| {
            let sq = &lay.seqs[s];
            let rep = (sq.strm * kv_size + sq.cells[sq.pools[k] as usize + kpool_u - 1].1) as usize;
            if st.rep_gen[rep] != gen {
                st.rep_gen[rep] = gen;
                st.is_new[pool_start[s] as usize + k] = gen;
                st.n_new += 1;
            }
        };

        let mut ip = 0u32;
        for s in 0..crate::batch::LLAMA_MAX_SEQ {
            let sq = &lay.seqs[s];
            pool_start[s] = ip;
            ip += sq.pools.len() as u32;

            // a sequence edit invalidates only pools ending after the edited
            // position (:1029-1041) — `std::lower_bound(pools, stale_from,
            // j => cells[j + kpool - 1].first < p)` (a7b94df2c:718-720: the
            // last member's position is the pool's true end, the old
            // `cells[j].first + kpool` assumed consecutive positions)
            let stale_from = stale_batch[s];
            if stale_from == HYBRID_IDX_POS_CLEAN {
                continue;
            }

            let first = sq
                .pools
                .partition_point(|&j| (sq.cells[j as usize + kpool_u - 1].0 as i64) < stale_from as i64);
            for pi in first..sq.pools.len() {
                mark(st, s, pi, &pool_start);
            }
        }
        assert!(ip as usize == st.is_new.len(), "kpool_build_state: pool count");

        // in order mode a token's cell gives its rank, and the rank its
        // pool: positions cannot, as an image shares one
        // (a7b94df2c:735-741)
        let by_order = self.kpool_by_order;

        // (:1050-1065 + a7b94df2c:743-757) — the pools the ubatch's tokens
        // fall inside
        for (i, &p) in ubatch_pos.iter().enumerate() {
            for &s in &ubatch_seqs[i] {
                let sq = &lay.seqs[s as usize];
                if by_order {
                    let r = kpool_rank(&sq.cells, p, ubatch_cells[i]);
                    assert!(r >= 0, "kpool_build_state: order-mode rank");
                    if (r as u64 / kpool_u as u64) < sq.pools.len() as u64 {
                        mark(st, s as usize, (r as usize) / kpool_u, &pool_start);
                    }
                    continue;
                }
                // `std::upper_bound(pools, p, (pos, j) => pos < cells[j].first)`
                // — the first pool whose start exceeds p; partition_point's
                // predicate is the complement (`start <= p`)
                let it = sq
                    .pools
                    .partition_point(|&j| sq.cells[j as usize].0 as i64 <= p as i64);
                if it == 0 {
                    continue;
                }
                let it = it - 1;
                // a7b94df2c:746 — `p <= cells[*it + kpool - 1].first`: the
                // last member's position is the pool's true end
                if p <= sq.cells[sq.pools[it] as usize + kpool_u - 1].0 {
                    mark(st, s as usize, it, &pool_start);
                }
            }
        }

        // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to
        // that bound so the graph keeps its shape as the count moves; reserve
        // sizes the list for every pool the cache can hold, so never pad past
        // n_pool_max (a7b94df2c:760-770)
        let n_pool_max = self.size / self.kpool; // idx->get_size()/kpool (single stream: n_seq_max folded to 1)
        let mut seqs_unq = 0u32;
        {
            let mut seen = 0u64;
            for seqs in ubatch_seqs {
                for &s in seqs {
                    seen |= 1u64 << s;
                }
            }
            seqs_unq = seen.count_ones();
        }
        let bound = ubatch_pos.len() as u32 / self.kpool + seqs_unq;
        let pad_cap = Self::kpool_pad(st.n_pool_real) - 1;
        st.n_new_g = st.n_new.max(1).max(bound.min(pad_cap).min(n_pool_max));
    }

    /// `kpool_cur()` (llama-memory-hybrid-idx.cpp:1068-1072) — read side of
    /// the step: only valid after `kpool_build_state`.
    pub fn kpool_cur(&self) -> &KpoolState {
        self.kpool_st.as_ref().expect("kpool_cur: state before apply()")
    }

    /// `kpool_pad` (:706-708) — the last padded pool is always unused.
    /// An associated fn (not `&self`) so `kpool_build_state` can call it
    /// while its `st` borrows the state mutably.
    pub fn kpool_pad(n_pool: u32) -> u32 {
        ((n_pool + 1 + 63) / 64 * 64).max(64)
    }

    /// `get_n_kpool()` (:1074-1076)
    pub fn get_n_kpool(&self) -> u32 {
        Self::kpool_pad(self.kpool_cur().n_pool_real)
    }

    /// `get_n_kpool_new()` (:1078-1080) — pools to re-pool this ubatch,
    /// padded to a stable bound, never below 1 (a7b94df2c:783-784 returns
    /// `n_new_g`)
    pub fn get_n_kpool_new(&self) -> u32 {
        self.kpool_cur().n_new_g
    }


    /// `llama_memory_hybrid_idx_context::set_input_kpool`
    /// (llama-memory-hybrid-idx.cpp:1086-1287 + a7b94df2c:791-1058) — fills
    /// the k-pool input tensors of one step. The port's single-stream cache
    /// pins `n_stream_kv == 1`, so `gcell(sq, cell) = cell` (:1123-1125) and
    /// the non-unified `inert` arm of :1177 is unreachable.
    /// `ubatch_pos`/`ubatch_seqs` are the current ubatch (single sequence in
    /// the port: one-entry lists); `ubatch_cells` is each token's indexer
    /// cell (order mode only); `new_pool_pos` (I32 [4*n_new]) carries the
    /// M-RoPE position of each new pool's first member, for pooled keys
    /// rotated at pooling time (a7b94df2c, read by qwen4exp's QSA graph).
    #[allow(clippy::too_many_arguments)]
    pub fn set_input_kpool(
        &self,
        gctx: &mut Context,
        base_cells: &[KvCell],
        n_kv: u32,
        pool_cells: TensorId,
        pool_idxs: TensorId,
        pool_mask: TensorId,
        mask_f16: bool,
        tail_idxs: TensorId,
        // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the
        // live selection slots, -inf for the dead ones — the old
        // gather_mask + gather pair collapsed into one (310991409 removed
        // the gather path of glm5-next's sparse attention, c173a53bd
        // dropped the gather-mode sentinel)
        sel_mask: Option<TensorId>,
        new_pool_idxs: TensorId,
        // new_pool_rep is required since c173a53bd: the graph always
        // scatters the fresh pooled keys back into the cache (build_qsa_sel)
        new_pool_rep: TensorId,
        ubatch_pos: &[i32],
        ubatch_seqs: &[Vec<i32>],
        ubatch_cells: &[u32],
        new_pool_pos: Option<TensorId>,
    ) {
        let kpool = self.kpool as usize;
        let st = self.kpool_cur();
        let lay = self.kpool_lay.as_ref().expect("set_input_kpool layout");

        let n_tokens = ubatch_pos.len();
        let n_pool = gctx.ne(pool_cells)[0] as u32;
        let n_new = st.n_new;
        // the graph always pools at least one entry, padded to a stable
        // bound, see kpool_build_state (:809-810, a7b94df2c)
        let n_new_g = st.n_new_g;

        assert!(n_pool == Self::kpool_pad(st.n_pool_real), "kpool: n_pool");
        assert!(st.is_new.len() == st.n_pool_real as usize, "kpool: is_new");
        assert!(
            gctx.ne(pool_mask)[0] == n_pool as i64
                && gctx.ne(pool_mask)[1] == n_tokens as i64
        );
        assert!(
            gctx.ne(tail_idxs)[0] == kpool as i64 - 1
                && gctx.ne(tail_idxs)[1] == n_tokens as i64
        );
        assert!(
            gctx.ne(pool_idxs)[0] == kpool as i64
                && gctx.ne(pool_idxs)[1] == n_pool as i64
        );
        // the graph always scatters the fresh pooled keys back into the
        // cache (c173a53bd, llama-memory-hybrid-idx.cpp:815-817)
        assert!(
            gctx.ne(new_pool_idxs)[0] == kpool as i64
                && gctx.ne(new_pool_idxs)[1] == n_new_g as i64
        );
        assert!(gctx.ne(new_pool_rep)[0] == n_new_g as i64, "kpool: new_pool_rep");
        if let Some(pos) = new_pool_pos {
            assert!(gctx.ne(pos)[0] == 4 * n_new_g as i64, "kpool: new_pool_pos");
        }

        let by_order = self.kpool_by_order;
        let kv_size = self.size;

        // gcell (:1123-1125) — strm == 0 in the port's single-stream cache
        let gcell = |_strm: u32, cell: u32| cell as i64;

        // sequences present in this ubatch (:1128-1133)
        let mut seq_in_ub = [false; crate::batch::LLAMA_MAX_SEQ];
        for seqs in ubatch_seqs {
            for &s in seqs {
                seq_in_ub[s as usize] = true;
            }
        }

        // use the first ubatch cell for padded gathers (:1136-1143)
        let mut dummy_cell = 0i64;
        {
            let s = ubatch_seqs[0][0] as usize;
            let sq = &lay.seqs[s];
            let p0 = ubatch_pos[0];
            let it = sq.cells.partition_point(|&c| c.0 < p0);
            assert!(
                it < sq.cells.len() && sq.cells[it].0 == p0,
                "kpool: dummy cell (pos {p0} of seq {s})"
            );
            dummy_cell = gcell(sq.strm, sq.cells[it].1);
        }

        // padding and absent cells point at the n_kv sentinel row, one
        // past the live cells (c173a53bd, :813-814 — the gather-mode
        // "dummy cell + separate mask" sentinel is gone with the gather path)
        let sentinel = n_kv as i32;

        // in order mode a token sees the pools and the tail up to its own
        // rank in the sequence, which its cell pins down (:852-864,
        // a7b94df2c) — the port's k_idxs order maps token i to cell
        // `ubatch_cells[i]` (the C's `sinfo.idxs[i / n_tps][i % n_tps]`
        // with ns == 1)
        let mut rank: Vec<i64> = Vec::new();
        if by_order {
            rank.resize(n_tokens, 0);
            for i in 0..n_tokens {
                rank[i] = kpool_rank(
                    &lay.seqs[ubatch_seqs[i][0] as usize].cells,
                    ubatch_pos[i],
                    ubatch_cells[i],
                );
                assert!(rank[i] >= 0, "kpool: order-mode rank");
            }
        }

        let mut gm: Option<TensorId> = None;
        let mut n_sel = 0u32;
        let mut n_top = 0u32; // pools per token in the selection
        if let Some(t) = sel_mask {
            assert!(gctx.ty(t) == GgmlType::F32, "kpool: sel_mask F32");
            let ne = *gctx.ne(t);
            assert!(
                ne[3] == n_tokens as i64 && ne[1] == 1 && ne[2] == 1,
                "kpool: sel_mask shape"
            );
            n_sel = ne[0] as u32;
            n_top = n_sel / self.kpool;
            assert!(n_sel % self.kpool == 0 || n_sel % self.kpool == self.kpool - 1);
            gm = Some(t);
        }

        // pools are laid out per sequence (:1163-1206)
        let mut seq_pool_start = [0u32; crate::batch::LLAMA_MAX_SEQ];
        let mut pool_end: Vec<i32> = Vec::with_capacity(n_pool as usize);

        let mut pcell = vec![0i32; n_pool as usize];
        let mut pidx = vec![0i32; (n_pool as usize) * kpool];
        let n_new_gu = n_new_g as usize;
        let mut nidx = vec![0i32; n_new_gu * kpool];
        let mut nrep = vec![0i64; n_new_gu];
        // new_pool_pos (I32 [4*n_new]) — zeroed first (:897-899, a7b94df2c)
        let mut npos: Vec<i32> = if new_pool_pos.is_some() {
            vec![0i32; 4 * n_new_gu]
        } else {
            Vec::new()
        };

        let mut i_new = 0u32;
        for s in 0..crate::batch::LLAMA_MAX_SEQ {
            let sq = &lay.seqs[s];
            seq_pool_start[s] = pool_end.len() as u32;

            // inert = n_stream_kv > 1 && !seq_in_ub[s] (:849-850) — the
            // port's single-stream cache makes it false for every sequence
            let _inert = !seq_in_ub[s];

            for pi in 0..sq.pools.len() {
                let j = sq.pools[pi] as usize;
                let ip = pool_end.len();
                assert!(ip + 1 < n_pool as usize, "kpool: pool overflow");

                // the pooled key lives in the last member's row (:1184-1186)
                let rep = sq.cells[j + kpool - 1].1;
                pcell[ip] = gcell(sq.strm, rep) as i32;

                for k in 0..kpool {
                    // inert == false ⇒ always the cell (:858-861, c173a53bd
                    // dropped the gather-mode gcell branch)
                    pidx[ip * kpool + k] = sq.cells[j + k].1 as i32;
                }

                if st.is_new[ip] == st.generation {
                    assert!((i_new as usize) < n_new as usize, "kpool: new count");
                    for k in 0..kpool {
                        nidx[i_new as usize * kpool + k] =
                            gcell(sq.strm, sq.cells[j + k].1) as i32;
                    }
                    // nrep is always written (c173a53bd: new_pool_rep is
                    // required, the graph always scatters)
                    nrep[i_new as usize] = gcell(sq.strm, rep);
                    if !npos.is_empty() {
                        // a pooled key is rotated to the M-RoPE position of
                        // its first member (:930-937, a7b94df2c) — sec1/sec2
                        // are the ext y/x of that member's cell. The port's
                        // cells carry no ext, but the reference's text cells
                        // DO: `ubatch.is_pos_2d()` is `n_pos >= 3`
                        // (llama-batch.h:27-30), TRUE for every M-RoPE
                        // batch, so apply_ubatch fills ext.y/ext.x from the
                        // broadcast 1D pos rows (llama-kv-cache.cpp:1136-
                        // 1139) — y == x == the temporal position (verified
                        // bit-exact by tests/qwen4exp_qsa_dump.rs: with
                        // y == x == 0 the IMRoPE sector rule leaves sectors
                        // 1,2,4,5,7 unrotated and the pooled keys diverge)
                        npos[0 * n_new_gu + i_new as usize] = sq.cells[j].0;
                        npos[1 * n_new_gu + i_new as usize] = sq.cells[j].0;
                        npos[2 * n_new_gu + i_new as usize] = sq.cells[j].0;
                        npos[3 * n_new_gu + i_new as usize] = sq.cells[j].0;
                    }
                    i_new += 1;
                }

                pool_end.push(sq.cells[j + kpool - 1].0);
            }
        }
        assert!(i_new == n_new, "kpool: i_new == n_new");

        // Padded entries re-pool cells whose pooled slot is never read: only
        // the reps of complete pools are read. Each entry takes its own cell,
        // entries sharing one would write it from several threads in the
        // scatter (:947-962, a7b94df2c — replaces the old "re-pool the first
        // token's cell when n_new == 0" rule)
        if n_new_g > n_new {
            let mut reps: Vec<i64> = pcell[..pool_end.len()].iter().map(|&c| c as i64).collect();
            reps.sort_unstable();

            let mut pad_cell: i64 = 0;
            for i in n_new as usize..n_new_gu {
                while reps.binary_search(&pad_cell).is_ok() {
                    pad_cell += 1;
                }
                assert!(pad_cell < kv_size as i64, "kpool: pad cell");
                for k in 0..kpool {
                    nidx[i * kpool + k] = pad_cell as i32;
                }
                nrep[i] = pad_cell;
                pad_cell += 1;
            }
        }

        let n_pool_real = pool_end.len() as u32;
        for ip in n_pool_real as usize..n_pool as usize {
            pcell[ip] = dummy_cell as i32; // pool_cells always addresses K
            for k in 0..kpool {
                pidx[ip * kpool + k] = sentinel;
            }
        }

        gctx
            .with_i32_mut(pool_cells, |p| p.copy_from_slice(&pcell))
            .unwrap();
        gctx
            .with_i32_mut(pool_idxs, |p| p.copy_from_slice(&pidx))
            .unwrap();
        gctx
            .with_i32_mut(new_pool_idxs, |p| p.copy_from_slice(&nidx))
            .unwrap();
        {
            let bytes = gctx.data_bytes_mut(new_pool_rep).unwrap();
            bytes.copy_from_slice(bytemuck::cast_slice(&nrep));
        }
        if new_pool_pos.is_some() {
            gctx
                .with_i32_mut(new_pool_pos.unwrap(), |p| p.copy_from_slice(&npos))
                .unwrap();
        }

        // a pool is visible when it belongs to the token's sequence and ends
        // at or before it (:1228-1253 + a7b94df2c:988-993) — the mask fill
        // (F16 or F32 arm of `fill_mask`'s template); order mode counts the
        // pools up to the token's rank instead
        if mask_f16 {
            gctx
                .with_f32_mut(pool_mask, |p| {
                    let h: &mut [half::f16] = bytemuck::cast_slice_mut(p);
                    for i in 0..n_tokens {
                        let s = ubatch_seqs[i][0] as usize;
                        let p = ubatch_pos[i];
                        let row = &mut h[i * n_pool as usize..(i + 1) * n_pool as usize];
                        row.fill(half::f16::NEG_INFINITY);
                        let p0 = seq_pool_start[s] as usize;
                        let p1 = p0 + lay.seqs[s].pools.len();
                        let nv = if by_order {
                            (p1 - p0).min(((rank[i] + 1) / kpool as i64) as usize)
                        } else {
                            pool_end[p0..p1].partition_point(|&e| e <= p)
                        };
                        row[p0..p0 + nv].fill(half::f16::ZERO);
                    }
                })
                .unwrap();
        } else {
            let n_pool_u = n_pool as usize;
            gctx.with_f32_mut(pool_mask, |m| {
                for i in 0..n_tokens {
                    let s = ubatch_seqs[i][0] as usize;
                    let pos = ubatch_pos[i];
                    let row = &mut m[i * n_pool_u..(i + 1) * n_pool_u];
                    row.fill(f32::NEG_INFINITY);
                    let p0 = seq_pool_start[s] as usize;
                    let p1 = p0 + lay.seqs[s].pools.len();
                    let nv = if by_order {
                        (p1 - p0).min(((rank[i] + 1) / kpool as i64) as usize)
                    } else {
                        pool_end[p0..p1].partition_point(|&e| e <= pos)
                    };
                    row[p0..p0 + nv].fill(0.0);
                }
            })
            .unwrap();
        }
        // the gather mask's pool rows (:1247-1252 + a7b94df2c:988-993)
        if let Some(t) = gm {
            let n_sel_u = n_sel as usize;
            gctx.with_f32_mut(t, |g| {
                for i in 0..n_tokens {
                    let s = ubatch_seqs[i][0] as usize;
                    let p = ubatch_pos[i];
                    let p0 = seq_pool_start[s] as usize;
                    let p1 = p0 + lay.seqs[s].pools.len();
                    let nv = if by_order {
                        (p1 - p0).min(((rank[i] + 1) / kpool as i64) as usize)
                    } else {
                        pool_end[p0..p1].partition_point(|&e| e <= p)
                    };
                    let nvc = (nv as u32).min(n_top) as usize;
                    let grow = &mut g[i * n_sel_u..];
                    grow[..nvc * kpool].fill(0.0);
                    grow[nvc * kpool..n_top as usize * kpool].fill(f32::NEG_INFINITY);
                }
            })
            .unwrap();
        }

        // tail_idxs (:1261-1286 + a7b94df2c:1010-1033) — the kpool-1 cells
        // ending at the query
        let mut tidx = vec![0i32; n_tokens * (kpool - 1)];
        for i in 0..n_tokens {
            let s = ubatch_seqs[i][0] as usize;
            let p = ubatch_pos[i];
            let sq = &lay.seqs[s];

            let n_tail = if by_order {
                ((rank[i] + 1) % kpool as i64) as usize
            } else {
                ((p - sq.pos_min + 1) % (self.kpool as i32)) as usize
            };

            for k in 0..kpool - 1 {
                let mut cell = sentinel;
                let mut real = false;
                if k < n_tail && by_order {
                    // order mode: the k cells below the token's rank
                    // (:1018-1021, a7b94df2c)
                    let c = sq.cells[(rank[i] - k as i64) as usize].1;
                    cell = c as i32;
                    real = true;
                } else if k < n_tail {
                    let pt = p - k as i32;
                    let it = sq.cells.partition_point(|&c| c.0 < pt);
                    if it < sq.cells.len() && sq.cells[it].0 == pt {
                        cell = sq.cells[it].1 as i32;
                        real = true;
                    }
                }
                tidx[i * (kpool - 1) + k] = cell;

                if let Some(t) = gm {
                    if n_sel % self.kpool != 0 {
                        let bytes = gctx.data_bytes_mut(t).unwrap();
                        let g: &mut [f32] = bytemuck::cast_slice_mut(bytes);
                        g[i * n_sel as usize + n_top as usize * kpool + k] =
                            if real { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
            }
        }
        gctx
            .with_i32_mut(tail_idxs, |p| p.copy_from_slice(&tidx))
            .unwrap();
    }

    // `set_input_qsa` (llama-memory-hybrid-idx.cpp:334-654 of def4d406a) was
    // DELETED by a7b94df2c (qwen4exp rework, commits 66e0c17ee/159c651f5/
    // c061df198): the block-compressed sparse-attention inputs moved into
    // the model graph — `llama_model_qwen4exp::llm_graph_input_kpool` fills
    // the shared k-pool inputs (`set_input_kpool` above, order mode) and
    // `build_qsa_sel`/`build_attn_qsa` (qwen4exp.cpp:649-1058) do the
    // selection. The port's qwen4exp step lives in DecodeContext's
    // `q4e_kpool_step` (context.rs, the models lane's graph consumes it).

    /// the kept-cell ranges over the lockstep base cells — the MSA
    /// precedent's `msa_idx_cell_ranges` (llama-kv-cache.cpp:2065-2095,
    /// n_swa = 0 for both idx consumers)
    pub fn idx_cell_ranges(&self, base_cells: &[KvCell], seq_id: i32) -> (u32, Vec<(u32, u32)>) {
        let mut cell_count = 0u32;
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut range_begin = base_cells.len() as u32;
        for (i, cell) in base_cells.iter().enumerate() {
            let mut add_cell = !cell.is_empty();
            add_cell = add_cell && (seq_id == -1 || cell.seq & (1u64 << seq_id) != 0);
            if add_cell {
                cell_count += 1;
                if range_begin == base_cells.len() as u32 {
                    range_begin = i as u32;
                }
            } else if range_begin != base_cells.len() as u32 {
                ranges.push((range_begin, i as u32));
                range_begin = base_cells.len() as u32;
            }
        }
        if range_begin != base_cells.len() as u32 {
            ranges.push((range_begin, base_cells.len() as u32));
        }
        (cell_count, ranges)
    }

    /// the idx layers that carry the C's `kv_idx->layers` (the filter's
    /// entries)
    pub fn idx_layers(&self) -> Vec<TensorId> {
        self.layers
            .iter()
            .zip(&self.is_idx)
            .filter(|(_, &f)| f)
            .map(|(&k, _)| k)
            .collect()
    }

    /// `mem_idx->state_write` ([TAG_HYBRID_IDX_STATE], llama-kv-cache.cpp's
    /// state_write over the idx cache's own cells — the lockstep base cells
    /// here). The idx cache is MLA-fooled (:63-65) ⇒ K-only: `v_stream[i] ==
    /// nullptr`, so `state_write_data`'s V loop skips every layer
    /// (llama-kv-cache.cpp:2447 `if (!v) continue`). The section goes last,
    /// so it is a pure suffix (llama-memory-hybrid-idx.cpp:261-262).
    pub fn state_seq_write(
        &self,
        io: &mut StateWriter,
        gctx: &Context,
        base_cells: &[KvCell],
        seq_id: i32,
    ) {
        let idx_layers = self.idx_layers();
        io.write_u32(1); // n_stream

        let (cell_count, ranges) = self.idx_cell_ranges(base_cells, seq_id);
        io.write_u32(cell_count);
        if cell_count == 0 {
            return;
        }

        // state_write_meta — identical pos/seq tuples to the attn half's
        for &(b, e) in &ranges {
            for i in b..e {
                let cell = &base_cells[i as usize];
                let seq_ids: Vec<i32> = (0..crate::batch::LLAMA_MAX_SEQ as i32)
                    .filter(|&cur| (cur == seq_id || seq_id == -1) && cell.seq & (1u64 << cur) != 0)
                    .collect();
                io.write_i32(cell.pos);
                io.write_u32(seq_ids.len() as u32);
                for s in seq_ids {
                    io.write_i32(s);
                }
            }
        }

        // state_write_data — K rows only (no V: MLA-fooled K-only cache)
        io.write_u32(0); // v_trans (fa on)
        io.write_u32(idx_layers.len() as u32);
        // 210791069: the base state_write_data persists the rotation widths
        // (the idx cache is a llama_kv_cache, its blob carries them too).
        // The idx plane is F16 (never quantized) and the hybrid-idx archs
        // are outside the DeepSeek-indexer clause of llama-kv-cache.cpp:
        // 321-332, so both widths are 0
        io.write_u32(0); // n_rot_k
        io.write_u32(0); // n_rot_v
        for &k in &idx_layers {
            let k_size_row = GgmlType::F16.row_size(self.row_w as usize) as u64;
            io.write_i32(GgmlType::F16 as i32);
            io.write_u64(k_size_row);
            for &(b, e) in &ranges {
                io.write_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                );
            }
        }
    }

    /// `mem_idx->state_read_sinfo(nullptr, &sinfos_attn)`
    /// ([TAG_HYBRID_IDX_SINFO], llama-memory-hybrid-idx.cpp:290) — the idx
    /// restore adopts the attention cache's layout instead of searching for
    /// cells of its own: verify the meta against the (already restored) base
    /// cells and scatter the idx rows over the same ranges. On success the
    /// caller runs `stale_set(seq_id, 0)` (+ the shared re-derive).
    pub fn state_seq_read(
        &mut self,
        io: &mut StateReader,
        gctx: &mut Context,
        base_cells: &[KvCell],
        seq_id: i32,
    ) -> Result<(), String> {
        let n_stream_cur = io.read_u32()?;
        if n_stream_cur != 1 {
            return Err("n_stream mismatch (hybrid idx)".into());
        }
        let cell_count = io.read_u32()?;
        if cell_count == 0 {
            return Ok(());
        }

        let (cell_count_cur, ranges) = self.idx_cell_ranges(base_cells, seq_id);
        if cell_count != cell_count_cur {
            return Err(format!(
                "hybrid idx cell count mismatch ({cell_count} instead of {cell_count_cur})"
            ));
        }
        for &(b, e) in &ranges {
            for i in b..e {
                let pos = io.read_i32()?;
                let n_seq_id = io.read_u32()?;
                let mut seq = 0u64;
                for _ in 0..n_seq_id {
                    let s = io.read_i32()?;
                    if s < 0 || s as usize >= crate::batch::LLAMA_MAX_SEQ {
                        return Err(format!("invalid seq_id, {s} is out of range"));
                    }
                    seq |= 1u64 << s;
                }
                let cell = &base_cells[i as usize];
                if cell.pos != pos || cell.seq & seq != seq {
                    return Err(format!(
                        "hybrid idx cell {i} mismatch (pos {pos}, seq {seq:#x}) against the \
                         restored base cells"
                    ));
                }
            }
        }

        let v_trans = io.read_u32()?;
        let n_layer = io.read_u32()?;
        // 210791069: the idx cache's own rotation widths (always 0/0 — F16
        // plane, non-DeepSeek arch) still ride the blob and are checked
        let n_rot_k_ref = io.read_u32()?;
        let n_rot_v_ref = io.read_u32()?;
        let idx_layers = self.idx_layers();
        if n_layer != idx_layers.len() as u32 {
            return Err(format!(
                "mismatched layer count ({n_layer} instead of {}, hybrid idx)",
                idx_layers.len()
            ));
        }
        if v_trans != 0 {
            return Err("incompatible V transposition (hybrid idx)".into());
        }
        if n_rot_k_ref != 0 {
            return Err(format!(
                "incompatible key rotation ({n_rot_k_ref} instead of 0)"
            ));
        }
        if n_rot_v_ref != 0 {
            return Err(format!(
                "incompatible value rotation ({n_rot_v_ref} instead of 0)"
            ));
        }

        let k_size_row = GgmlType::F16.row_size(self.row_w as usize) as u64;
        for &k in &idx_layers {
            let k_type_i_ref = io.read_i32()?;
            if gctx.ty(k) as i32 != k_type_i_ref {
                return Err("mismatched key type (hybrid idx)".into());
            }
            let k_size_row_ref = io.read_u64()?;
            if k_size_row != k_size_row_ref {
                return Err("mismatched key row size (hybrid idx)".into());
            }
            for &(b, e) in &ranges {
                io.read_tensor(
                    gctx,
                    k,
                    b as usize * k_size_row as usize,
                    (e - b) as usize * k_size_row as usize,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `kpool_rank` (a7b94df2c, llama-memory-hybrid-idx.cpp:391-395): the
    /// rank of (pos, cell) in the position-then-cell order, -1 when absent —
    /// cells sharing a position (M-RoPE images) have distinct ranks.
    #[test]
    fn kpool_rank_finds_or_rejects() {
        let cells: Vec<(i32, u32)> = vec![(0, 3), (0, 7), (1, 1), (4, 0)];
        assert_eq!(kpool_rank(&cells, 0, 3), 0);
        assert_eq!(kpool_rank(&cells, 0, 7), 1); // same pos, next cell
        assert_eq!(kpool_rank(&cells, 1, 1), 2);
        assert_eq!(kpool_rank(&cells, 4, 0), 3);
        assert_eq!(kpool_rank(&cells, 0, 1), -1); // cell holds another pos
        assert_eq!(kpool_rank(&cells, 2, 5), -1); // absent
    }

    /// Order-mode pools (a7b94df2c, `indexer_kpool_by_order`): kpool
    /// consecutive **cells** in sequence order, whatever their positions —
    /// repeated positions (M-RoPE images) pool where the position rule would
    /// skip. Plus the stable `n_new_g` graph size (a7b94df2c:760-770): a
    /// prefill pads past its real new-pool count, a decode step never drops
    /// below 1.
    #[test]
    fn kpool_by_order_layout_and_stable_bound() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 1, 16, 16, 64);
        // 6 tokens at positions 0,0,1,2,2,3 — cells 0..5 of seq 0
        cache.assign(SlotInfo { s0: 0, s1: 5 }, &[0, 0, 1, 2, 2, 3], 0);
        let mut idx = HybridIdxCache::new(&mut ctx, 64, 8, 2, 2, true, vec![true]);

        idx.kpool_layout_update(&cache.cells);
        let lay = idx.kpool_lay.as_ref().unwrap();
        // pools of 2 consecutive cells regardless of positions
        assert_eq!(lay.seqs[0].pools, vec![0, 2, 4]);
        assert_eq!(lay.n_pool_real, 3);

        // a decode step at the last token (rank 5 → pool 2): one new pool,
        // the padded graph size stays at 1
        idx.kpool_build_state(
            &[3],
            &[vec![0]],
            &[5],
            &hybrid_idx_stale_clean(),
        );
        assert_eq!(idx.kpool_cur().n_new, 1);
        assert_eq!(idx.get_n_kpool_new(), 1);

        // a prefill over all six tokens: every pool is new (3), the bound
        // n_tokens/kpool + n_seqs_unq = 4 pads the graph size to 4
        let all_pos: Vec<i32> = vec![0, 0, 1, 2, 2, 3];
        let all_cells: Vec<u32> = (0..6).collect();
        let seqs: Vec<Vec<i32>> = vec![vec![0]; 6];
        idx.kpool_build_state(&all_pos, &seqs, &all_cells, &hybrid_idx_stale_clean());
        assert_eq!(idx.kpool_cur().n_new, 3);
        assert_eq!(idx.get_n_kpool_new(), 4);
    }

    /// The position rule still holds without by_order (glm5-next), and the
    /// a7b94df2c comparator change (a pool spans up to its **last member's**
    /// position) marks the same pools as before for consecutive runs.
    #[test]
    fn kpool_position_rule_marks_pool_of_token() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 1, 16, 16, 64);
        cache.assign(SlotInfo { s0: 0, s1: 5 }, &[0, 1, 2, 3, 4, 5], 0);
        let mut idx = HybridIdxCache::new(&mut ctx, 64, 8, 4, 3, false, vec![true]);
        idx.kpool_layout_update(&cache.cells);
        let lay = idx.kpool_lay.as_ref().unwrap();
        // pos_min = 0, kpool = 4: one pool at 0..4, the tail 4,5 incomplete
        assert_eq!(lay.seqs[0].pools, vec![0]);
        // a token inside the pool (pos 2) re-pools it; a tail token (pos 5)
        // completes no pool and only pads
        idx.kpool_build_state(&[2], &[vec![0]], &[2], &hybrid_idx_stale_clean());
        assert_eq!(idx.kpool_cur().n_new, 1);
        idx.kpool_build_state(&[5], &[vec![0]], &[5], &hybrid_idx_stale_clean());
        assert_eq!(idx.kpool_cur().n_new, 0);
        assert_eq!(idx.get_n_kpool_new(), 1); // never below 1
    }

    /// `llama_kv_cells::seq_pos_get` (llama-kv-cells.h:321-328, def4d406a):
    /// the (pos, cell) pairs of one sequence, position-ordered — the k-pool
    /// layout's input.
    #[test]
    fn seq_pos_get_orders_pairs_by_position() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 2, 128, 128, 64);
        // cells 4..8 carry seq 0 at positions 10..14; cell 2 carries seq 1
        cache.assign(SlotInfo { s0: 4, s1: 7 }, &[10, 11, 12, 13], 0);
        cache.assign(SlotInfo { s0: 2, s1: 2 }, &[5], 1);
        assert_eq!(
            cache.seq_pos_get(0),
            vec![(10, 4), (11, 5), (12, 6), (13, 7)]
        );
        assert_eq!(cache.seq_pos_get(1), vec![(5, 2)]);
        // a sequence with no cells is empty
        assert!(cache.seq_pos_get(2).is_empty());
        // shared cells appear for both sequences
        cache.assign(SlotInfo { s0: 4, s1: 4 }, &[10], 1);
        assert_eq!(cache.seq_pos_get(1), vec![(5, 2), (10, 4)]);
    }

    #[test]
    fn slot_lifecycle() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 2, 128, 128, 64);
        let s = cache.find_slot(4).unwrap();
        assert_eq!(s.s0, 0);
        cache.assign(s, &[0, 1, 2, 3], 0);
        // get_n_kv pads to a multiple of 256, capped by cells.size() = 64
        assert_eq!(cache.n_kv(), 64);
        let s2 = cache.find_slot(4).unwrap();
        assert_eq!(s2.s0, 4);
        cache.seq_rm(0, -1, -1);
        assert_eq!(cache.n_kv(), 64);
        // occupied tail must be skipped
        cache.assign(SlotInfo { s0: 60, s1: 63 }, &[10, 11, 12, 13], 0);
        let s3 = cache.find_slot(10).unwrap();
        assert_eq!(s3.s0, 0);
    }

    /// `n_kv` is the 256-padded `used_max_p1` (llama-kv-cache.cpp:1250-1263,
    /// `n_pad_cur = max(n_pad, 256)`), i.e. a *cell index* count: a cell that
    /// was rm'd inside the used range lowers it (and the padding follows).
    #[test]
    fn n_kv_is_used_max_p1() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 1, 16, 16, 32);
        cache.assign(SlotInfo { s0: 0, s1: 3 }, &[0, 1, 2, 3], 0);
        assert_eq!(cache.n_kv(), 32); // PAD(4, 256) capped at size 32
                                      // hole in the middle: still the last used index + 1
        cache.seq_rm(0, 1, 2);
        assert_eq!(cache.n_kv(), 32);
        assert!(cache.cells[1].is_empty());
        // rm the tail: the count follows (cell 2 is still the last used one)
        cache.seq_rm(0, 3, 4);
        assert_eq!(cache.n_kv(), 32);
        // a cache larger than 256 pads up to the multiple: 300 used -> 512
        let mut big = KvCache::new(&mut ctx, 1, 16, 16, 1024);
        big.assign(
            SlotInfo { s0: 0, s1: 299 },
            &(0..300).collect::<Vec<_>>(),
            0,
        );
        assert_eq!(big.n_kv(), 512);
    }

    /// `llama_kv_cache_iswa` geometry (llama-kv-cache-iswa.cpp:69-81).
    #[test]
    fn swa_cache_size_formula() {
        // swa_full (the llama_context_params / common default) -> base size
        assert_eq!(swa_cache_size(2048, 1024, 1, 512, true, true), 2048);
        // gemma-4-12B without swa_full: PAD(min(2048, 1024*1 + 512), 256)
        assert_eq!(swa_cache_size(2048, 1024, 1, 512, true, false), 1536);
        // gpt-oss-20b: PAD(min(2048, 128*1 + 512), 256)
        assert_eq!(swa_cache_size(2048, 128, 1, 512, true, false), 768);
        // size_base below the window request wins via min()
        assert_eq!(swa_cache_size(512, 1024, 1, 512, true, false), 512);
        // unified=false multiplies by 1 instead of n_seq_max
        assert_eq!(swa_cache_size(8192, 1024, 4, 512, false, false), 1536);
        assert_eq!(swa_cache_size(8192, 1024, 4, 512, true, false), 4608);
    }

    /// The `iswa` split: SWA layers land in the second cache with `size_swa`
    /// cells, dense layers in the base one, and each layer's tensor is the one
    /// its cache owns (variant of the `slot_lifecycle` test for the pair).
    #[test]
    fn iswa_split_layers_and_cells() {
        let mut ctx = ggml::Context::new();
        let spec = SwaCacheSpec {
            n_swa: 1000,
            swa_type: LlamaSwaType::STANDARD,
            is_swa: vec![true, false, true, false],
            swa_full: false,
            unified: true,
            n_seq_max: 1,
        };
        let k_row = vec![32i64; 4];
        let v_row = vec![32i64; 4];
        let mut cache = KvCache::new_swa(&mut ctx, &k_row, &v_row, 2048, &spec, 24);
        // PAD(min(2048, 1000*1 + 24), 256) = 1024
        assert_eq!(cache.swa.as_ref().unwrap().size, 1024);
        assert_eq!(cache.size, 2048);
        assert_eq!(cache.swa.as_ref().unwrap().layer_ids, vec![0, 2]);
        assert_eq!(cache.swa.as_ref().unwrap().n_swa, 1000);
        assert_eq!(cache.swa.as_ref().unwrap().swa_type, LlamaSwaType::STANDARD);
        assert!(cache.layer_is_swa(0) && !cache.layer_is_swa(1));
        // per-layer tensor allocation sizes follow the filter
        assert_eq!(
            ctx.ne(cache.layers[0].k)[1],
            cache.swa.as_ref().unwrap().size as i64
        );
        assert_eq!(ctx.ne(cache.layers[1].k)[1], cache.size as i64);
        assert_eq!(cache.cache_size_of(0), 1024);
        assert_eq!(cache.cache_size_of(1), 2048);
        // a SWA layer's view strides through its own cache (get_k nb[3]/nb[2])
        let k0 = cache.get_k(&mut ctx, 0, 32, 1, 4);
        let k1 = cache.get_k(&mut ctx, 1, 32, 1, 4);
        assert_eq!(*ctx.nb(k0), [2, 64, 64, 64 * 1024]);
        assert_eq!(*ctx.nb(k1), [2, 64, 64, 64 * 2048]);
    }

    /// The SWA slot search reuses a cell only once its token has left the
    /// window (llama-kv-cache.cpp:1044-1058), and `assign_swa` then purges the
    /// positions below the overwritten one (:1160-1179).
    ///
    /// `size_swa` is padded to 256 (llama-kv-cache-iswa.cpp:73), so the ring
    /// below is a 256-cell cache — the smallest one that formula can produce.
    #[test]
    fn swa_slot_reuse_out_of_window() {
        let mut ctx = ggml::Context::new();
        let spec = SwaCacheSpec {
            n_swa: 4,
            swa_type: LlamaSwaType::STANDARD,
            is_swa: vec![true, false],
            swa_full: false,
            unified: true,
            n_seq_max: 1,
        };
        // PAD(min(256, 4*1 + 4), 256) = 256
        let mut cache = KvCache::new_swa(&mut ctx, &[16, 16], &[16, 16], 256, &spec, 4);
        assert_eq!(cache.swa.as_ref().unwrap().size, 256);

        // fill it: positions 0..256 in cells 0..256, head past the end
        let idxs = cache.find_slot_swa(256).unwrap();
        assert_eq!(idxs, (0..256).collect::<Vec<u32>>());
        let pos: Vec<i32> = (0..256).collect();
        cache.assign_swa(&idxs, &pos, 0);
        assert_eq!(cache.n_kv_swa(), 256);
        assert_eq!(cache.swa.as_ref().unwrap().head, 256);

        // full cache, sequence max pos 255: for the next position (256) a cell
        // is reusable iff is_masked_swa(4, STANDARD, pos_cell, 256), i.e.
        // pos_cell <= 252 -> cells 0..=252
        let idxs = cache.find_slot_swa(3).unwrap();
        assert_eq!(idxs, vec![0, 1, 2]);
        // an empty cell would be preferred: 253 reusable cells are not enough
        assert_eq!(cache.find_slot_swa(254), None);

        // overwriting cells 4..8 (pos 4..7) with 256..260 purges every position
        // <= 259's own cut, i.e. every position <= 7 (llama-kv-cache.cpp:1160-1179)
        cache.assign_swa(&[4, 5, 6, 7], &[256, 257, 258, 259], 0);
        assert_eq!(cache.n_kv_swa(), 256);
        for i in 0..4 {
            assert!(
                cache.swa.as_ref().unwrap().cells[i].is_empty(),
                "cell {i} purged"
            );
        }
        let live: Vec<i32> = cache
            .swa
            .as_ref()
            .unwrap()
            .cells
            .iter()
            .filter(|c| !c.is_empty())
            .map(|c| c.pos)
            .collect();
        assert_eq!(live.len(), 252); // cells 0..4 (pos 0..3) purged, 4..8 rewritten
        assert_eq!(&live[..4], &[256, 257, 258, 259]);
        assert_eq!(live[4], 8);
        assert_eq!(cache.swa.as_ref().unwrap().head, 8);
    }

    // -----------------------------------------------------------------------
    // the position-shift family (llama-kv-cache.cpp:451-657) — hand-computed
    // cell layouts, the arithmetic the server's context shift depends on
    // -----------------------------------------------------------------------

    /// `llama_kv_cache::seq_add` (llama-kv-cache.cpp:570-618): the position
    /// arithmetic, the free-below-zero rule (`pos_add`,
    /// llama-kv-cells.h:451-459), the stacked shift accumulators
    /// (:447/:496-511) and the head reset (:615-617).
    #[test]
    fn seq_add_position_math() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 1, 16, 16, 16);
        cache.assign(SlotInfo { s0: 0, s1: 7 }, &[0, 1, 2, 3, 4, 5, 6, 7], 0);
        assert_eq!(cache.head, 8);

        // shift everything by -3: cells 0..3 (pos 0..3) fall below 0 and are
        // freed; the survivors at cells 3..8 hold pos 0..5
        cache.seq_add(0, -1, -1, -3);
        let pos: Vec<i32> = cache.cells.iter().map(|c| c.pos).collect();
        assert_eq!(
            pos,
            vec![-1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1]
        );
        // head restarts at the first freed cell (:615-617)
        assert_eq!(cache.head, 0);
        // the accumulators survive for the K-shift graph; a freed cell resets
        // (pos_add's early-out resets shift to 0, llama-kv-cells.h:453)
        assert!(cache.get_has_shift());
        for i in 3..8 {
            assert_eq!(cache.cells[i].shift, -3, "cell {i} kept its accumulator");
        }
        assert_eq!(cache.cells[0].shift, 0);

        // a partial range stacks on the accumulators (:447): pos in [1, 4) are
        // cells 4,5,6 (pos 1,2,3) — they move to 3,4,5 and accumulate to -1
        cache.seq_add(0, 1, 4, 2);
        let pos: Vec<i32> = cache.cells.iter().map(|c| c.pos).collect();
        assert_eq!(pos[3..8].to_vec(), vec![0, 3, 4, 5, 4]);
        assert_eq!(cache.cells[4].shift, -1);
        assert_eq!(cache.cells[5].shift, -1);
        assert_eq!(cache.cells[6].shift, -1);
        assert_eq!(cache.cells[3].shift, -3, "outside [p0, p1): untouched");
        assert_eq!(cache.cells[7].shift, -3);

        // shift == 0 and p0 == p1 are the early-outs (:582-584 / :596-599)
        let before = cache.cells.clone();
        cache.seq_add(0, -1, -1, 0);
        cache.seq_add(0, 5, 5, 7);
        assert_eq!(cache.cells, before);

        // nothing freed -> head restarts at 0 (:617)
        cache.seq_add(0, 0, -1, 1);
        assert_eq!(cache.head, 0);

        // reset_shift clears every accumulator (llama-kv-cells.h:57-59)
        cache.reset_shift();
        assert!(!cache.get_has_shift());
        assert!(cache.cells.iter().all(|c| c.shift == 0));
    }

    /// `seq_cp` (metadata merge, llama-kv-cache.cpp:481-489) + `seq_keep`
    /// (:543-568) + `seq_div` (:620-657, `pos_div` llama-kv-cells.h:469-483)
    /// on one hand-computed layout.
    #[test]
    fn seq_cp_keep_div_math() {
        let mut ctx = ggml::Context::new();
        let mut cache = KvCache::new(&mut ctx, 1, 16, 16, 16);
        cache.assign(SlotInfo { s0: 0, s1: 7 }, &[0, 1, 2, 3, 4, 5, 6, 7], 0);

        // seq_cp(0 -> 1, [2, 5)): cells 2,3,4 gain sequence 1 (:487)
        cache.seq_cp(0, 1, 2, 5).unwrap();
        assert_eq!(cache.cells[2].seq, 0b11);
        assert_eq!(cache.cells[3].seq, 0b11);
        assert_eq!(cache.cells[4].seq, 0b11);
        assert_eq!(cache.cells[5].seq, 0b01, "outside [p0, p1): untouched");
        // same src and dst is the no-op of :469-471
        cache.seq_cp(1, 1, -1, -1).unwrap();
        assert_eq!(cache.cells[2].seq, 0b11);

        // seq_keep(1): every cell keeps only seq 1; the cells without it are
        // freed and the head moves to the first freed cell (:554-567)
        cache.head = 8;
        cache.seq_keep(1);
        for i in [0, 1, 5, 6, 7] {
            assert!(cache.cells[i].is_empty(), "cell {i} freed (not seq 1)");
        }
        for i in 2..5 {
            assert_eq!(cache.cells[i].seq, 0b10, "cell {i} keeps only seq 1");
            assert_eq!(cache.cells[i].pos, i as i32);
        }
        assert_eq!(cache.head, 0, "first freed cell (0) is before head (8)");

        // seq_div(1, d = 2): pos 2 -> 1 (shift +1), 3 -> 1 (+2), 4 -> 2 (+2) —
        // C integer division truncates toward zero, `shift += p_old - pos_new`
        cache.seq_div(1, -1, -1, 2);
        assert_eq!(
            cache.cells[2..5].iter().map(|c| c.pos).collect::<Vec<_>>(),
            vec![1, 1, 2]
        );
        assert_eq!(cache.cells[2].shift, 1);
        assert_eq!(cache.cells[3].shift, 2);
        assert_eq!(cache.cells[4].shift, 2);
        assert!(cache.get_has_shift(), "pos_div marks the cache shifted");
        // d == 1 is the early-out (:631-633)
        let before = cache.cells.clone();
        cache.seq_div(1, -1, -1, 1);
        assert_eq!(cache.cells, before);
    }

    /// `llama_kv_cache_iswa::seq_add` forwards to **both** caches
    /// (llama-kv-cache-iswa.cpp:132-134): the SWA half's cells shift with the
    /// base half's and carry their own accumulators.
    #[test]
    fn seq_add_over_the_iswa_pair() {
        let mut ctx = ggml::Context::new();
        // n_swa 1000 over a 16-cell cache: nothing is window-masked, so the
        // SWA cells stay in lockstep with the base ones
        let spec = SwaCacheSpec {
            n_swa: 1000,
            swa_type: LlamaSwaType::STANDARD,
            is_swa: vec![true, false],
            swa_full: false,
            unified: true,
            n_seq_max: 1,
        };
        // PAD(min(256, 1000*1 + 16), 256) = 256 — size_swa 256 vs size 16
        let mut cache = KvCache::new_swa(&mut ctx, &[16, 16], &[16, 16], 16, &spec, 16);
        assert_eq!(cache.swa.as_ref().unwrap().size, 256);

        // the same token placed in both halves (the driver's step prologue)
        cache.assign(SlotInfo { s0: 0, s1: 3 }, &[10, 11, 12, 13], 0);
        let idxs = cache.find_slot_swa(4).unwrap();
        cache.assign_swa(&idxs, &[10, 11, 12, 13], 0);
        assert_eq!(idxs, vec![0, 1, 2, 3]);

        // shift by -10: no cell falls below 0, both halves move, both halves
        // accumulate (the K-shift graph rotates both, llama-kv-cache.cpp:872
        // runs per cache)
        cache.seq_add(0, -1, -1, -10);
        assert_eq!(
            cache.cells[..4].iter().map(|c| c.pos).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        let swa = cache.swa.as_ref().unwrap();
        assert_eq!(
            swa.cells[..4].iter().map(|c| c.pos).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(cache.cells[..4].iter().all(|c| c.shift == -10));
        assert!(swa.cells[..4].iter().all(|c| c.shift == -10));
        assert!(cache.get_has_shift());

        // a partial range touches both halves identically
        cache.seq_add(0, 2, -1, 1);
        assert_eq!(
            cache.cells[..4].iter().map(|c| c.pos).collect::<Vec<_>>(),
            vec![0, 1, 3, 4]
        );
        assert_eq!(
            cache.swa.as_ref().unwrap().cells[..4]
                .iter()
                .map(|c| c.pos)
                .collect::<Vec<_>>(),
            vec![0, 1, 3, 4]
        );

        // reset_shift clears both halves (:888-892 over every stream)
        cache.reset_shift();
        assert!(!cache.get_has_shift());
        assert!(cache.cells.iter().all(|c| c.shift == 0));
        assert!(cache
            .swa
            .as_ref()
            .unwrap()
            .cells
            .iter()
            .all(|c| c.shift == 0));
    }
}
