//! Architecture forward graphs — port of src/llama-graph.cpp helpers +
//! src/models/qwen2.cpp (both build_attn_mha branches: the non-FA path
//! kq mul_mat → soft_max → kqv, and the flash-attn path ggml_flash_attn_ext,
//! llama-graph.cpp:2626-2669).
//!
//! The builder takes plain TensorId handles so the model layer (model.rs) can
//! wire any storage; all shapes follow the C graph exactly.

use crate::hparams::{LlamaHparams, LlamaSwaType};
use crate::kv_cache::{KvCache, SlotInfo};
use ggml::types::GgmlType;
use ggml::{Context, Graph, TensorId};

/// Per-layer weight handles (qwen2 / llama layout).
pub struct LayerWeights {
    pub attn_norm: TensorId,
    pub wq: TensorId,
    pub wk: TensorId,
    pub wv: TensorId,
    pub wo: TensorId,
    pub wq_b: Option<TensorId>,
    pub wk_b: Option<TensorId>,
    pub wv_b: Option<TensorId>,
    pub ffn_norm: TensorId,
    pub ffn_gate: TensorId,
    pub ffn_down: TensorId,
    pub ffn_up: TensorId,
}

pub struct ModelWeights {
    pub tok_embd: TensorId,
    pub output_norm: TensorId,
    pub output: TensorId,
    pub layers: Vec<LayerWeights>,
}

/// Rope/attention hyperparameters used by the forward pass.
#[derive(Clone, Copy)]
pub struct AttnParams {
    pub n_head: i64,
    pub n_head_kv: i64,
    pub n_embd_head_k: i64,
    pub n_embd_head_v: i64,
    pub n_rot: i64,
    /// GGML rope mode (2 = NEOX for qwen2/llama)
    pub rope_mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub norm_eps: f32,
    /// cparams.flash_attn (llama-context.cpp:230: `params.flash_attn_type !=
    /// LLAMA_FLASH_ATTN_TYPE_DISABLED`) — selects the ggml_flash_attn_ext branch
    /// of build_attn_mha (llama-graph.cpp:2626, `cparams.flash_attn && kq_b ==
    /// nullptr`). Default false: the non-FA path is the verified baseline.
    pub use_flash_attn: bool,
}

/// Per-decode input tensors, filled by the context layer before compute.
#[derive(Clone, Copy, Debug)]
pub struct DecodeInputs {
    /// I32 [n_tokens]
    pub tokens: TensorId,
    /// I32 [n_tokens]
    pub pos: TensorId,
    /// F32 [n_kv_padded, n_tokens]: 0 = attend, -inf = masked
    pub kq_mask: TensorId,
    /// I64 [n_tokens] row destination indices into the cache
    pub row_idx: TensorId,
    /// `inp_out_ids` (build_inp_out_ids, llama-graph.cpp:2480-2496): I32
    /// [n_outputs] token indices whose rows the graph must produce — the
    /// builders gather these rows of the pre-lm_head hidden state
    /// ([`out_rows`]) so the head computes only the requested logits rows.
    /// `None` = every token is an output row (`n_outputs == n_tokens`, the
    /// embeddings / logits_all contexts — llm_graph_input_out_ids::set_input
    /// degenerates to `data[i] = i`, llama-graph.cpp:207-213, and the gather
    /// is elided).
    pub out_ids: Option<TensorId>,
}

/// The `ggml_get_rows(ctx0, cur, inp_out_ids)` every arch graph applies to the
/// post-output-norm hidden state before its lm_head (`build_lora_mm`/
/// `ggml_mul_mat` on the gathered rows): gemma4.cpp:416-417,
/// hrm-text.cpp:200-201, hy-v4.cpp:583-585, ernie4-5.cpp:124-125,
/// openai-moe.cpp:160-162, glm4-moe.cpp:271-272 (the MTP shared head),
/// deepseek2.cpp:395-397. The standard in-loop placement
/// (`il == n_layer - 1 && inp_out_ids`, qwen2.cpp:106-108 / llama.cpp:174-176
/// / qwen3.cpp:114-116 …) additionally prunes the last layer's FFN rows —
/// same selected-row values, so the port folds the gather into this one
/// shared helper at the head instead of per builder loop.
///
/// Value identity: every logits column of the head mul_mat is an independent
/// vec_dot over one hidden-state row's bytes, so gathering rows first computes
/// bit-identical logits for the selected rows as the full [n_vocab × n_tokens]
/// head did (the kernel's per-element accumulation does not depend on the
/// column count).
pub fn out_rows(ctx: &mut Context, cur: TensorId, out_ids: Option<TensorId>) -> TensorId {
    match out_ids {
        Some(ids) => ctx.get_rows(cur, ids),
        None => cur,
    }
}

pub struct ForwardResult {
    /// [n_vocab, n_tokens]
    pub logits: TensorId,
    /// `res->t_embd` (qwen2.cpp:142) — the post-output-norm hidden state
    /// [n_embd, n_tokens], mounted **before** the lm_head (every decoder
    /// builder does `cb(cur, "result_norm", -1); res->t_embd = cur;` — the
    /// lm_head is still built, the C just doesn't consume the logits when
    /// `cparams.embeddings` is on). `build_pooling` mounts on this tensor
    /// (llama-model.cpp:2764-2766); `None` marks an arch whose builder has no
    /// single hidden-state output (the encoder paths, reached through
    /// `EncoderContext`, never go through here).
    pub embd: Option<TensorId>,
    pub graph: Graph,
}

/// build_inp_embd + per-layer loop + lm_head, 对照 src/models/qwen2.cpp graph()
/// 和 llama-graph.cpp build_attn_mha 的非 FA 分支。
pub fn build_qwen2_forward(
    ctx: &mut Context,
    w: &ModelWeights,
    p: &AttnParams,
    kv: &KvCache,
    inp: &DecodeInputs,
    sinfo: SlotInfo,
    n_kv: u32,
    n_tokens: usize,
) -> ForwardResult {
    let t = n_tokens as i64;
    let n_layer = w.layers.len();
    // this builder predates the `iswa` pair: it reads the base cache and its
    // mask unconditionally. SWA models go through the `attn_kv_cached` path
    // (graph_arch.rs), which selects per layer (llama-graph.cpp:3131-3133).
    assert!(
        !kv.has_swa(),
        "build_qwen2_forward: SWA models must use a builder with per-layer cache selection"
    );
    // C expands every intermediate via the cb() macro; the set_rows writes have
    // no downstream consumers in the graph (reads view the cache base), so they
    // must be expanded explicitly or the cache never gets written.
    let mut graph = Graph::new(1024);

    // inpL = embd lookup [n_embd, T]
    // (`build_inp_embd`, llama-graph.cpp:2387 — the lora delta of :2389-2405
    // is added on top of the gathered rows)
    let inp_l = ctx.get_rows(w.tok_embd, inp.tokens);
    let mut inp_l = crate::adapter::lora_embd(ctx, w.tok_embd, inp_l, inp.tokens);

    for il in 0..n_layer {
        let lw = &w.layers[il];
        let inp_sa = inp_l;

        // attn_norm: rms_norm * weight
        let mut cur = ctx.rms_norm(inp_l, p.norm_eps);
        cur = ctx.mul(cur, lw.attn_norm);

        // QKV projections (+ k/v bias for qwen2) — 对照 build_qkv
        // (build_lora_mm at llama-graph.cpp:1689/1703/1717)
        let mut q = crate::adapter::lora_mm(ctx, lw.wq, cur); // [n_embd, T]
        if let Some(b) = lw.wq_b {
            q = ctx.add(q, b);
        }
        let mut k = crate::adapter::lora_mm(ctx, lw.wk, cur); // [n_embd_gqa, T]
        if let Some(b) = lw.wk_b {
            k = ctx.add(k, b);
        }
        let mut v = crate::adapter::lora_mm(ctx, lw.wv, cur);
        if let Some(b) = lw.wv_b {
            v = ctx.add(v, b);
        }

        // reshape to [n_embd_head, heads, T]
        let q = ctx.reshape_3d(q, p.n_embd_head_k, p.n_head, t);
        let k = ctx.reshape_3d(k, p.n_embd_head_k, p.n_head_kv, t);
        let v = ctx.reshape_3d(v, p.n_embd_head_v, p.n_head_kv, t);

        // rope Q/K (NEOX)
        let q = ctx.rope_ext(
            q,
            inp.pos,
            None,
            p.n_rot as i32,
            p.rope_mode,
            p.n_ctx_orig,
            p.freq_base,
            p.freq_scale,
            p.ext_factor,
            p.attn_factor,
            p.beta_fast,
            p.beta_slow,
        );
        let k = ctx.rope_ext(
            k,
            inp.pos,
            None,
            p.n_rot as i32,
            p.rope_mode,
            p.n_ctx_orig,
            p.freq_base,
            p.freq_scale,
            p.ext_factor,
            p.attn_factor,
            p.beta_fast,
            p.beta_slow,
        );

        // store K/V into the cache (set_rows scatter of [n_embd_gqa, T] rows)
        let k_rows = ctx.reshape_2d(k, p.n_embd_head_k * p.n_head_kv, t);
        let v_rows = ctx.reshape_2d(v, p.n_embd_head_v * p.n_head_kv, t);
        let k_dst = ctx.set_rows(kv.layers[il].k, k_rows, inp.row_idx);
        let v_dst = ctx.set_rows(kv.layers[il].v, v_rows, inp.row_idx);
        graph.build_forward(ctx, k_dst);
        graph.build_forward(ctx, v_dst);

        // cache views [head_dim, n_head_kv, n_kv, 1]
        let k_view = kv.get_k(ctx, il, p.n_embd_head_k, p.n_head_kv, n_kv);
        let v_view = kv.get_v(ctx, il, p.n_embd_head_v, p.n_head_kv, n_kv);
        let _ = (k_dst, v_dst); // computed for effect (in-place on cache)

        let kq_scale = 1.0 / (p.n_embd_head_k as f32).sqrt();
        let kqv = if p.use_flash_attn {
            // FA branch — 对照 llama-graph.cpp:2626-2669 (use_flash_attn &&
            // kq_b == nullptr; qwen2 has no KQ bias, so FA is always allowed).
            // max_bias = hparams.f_max_alibi_bias (0.0 — no ALiBi here, same as
            // the non-FA soft_max_ext call below); logit_softcap = 0
            // (hparams.attn_soft_cap is false for qwen2, llama-graph.cpp:2643-2644).
            flash_attn_core(ctx, q, k_view, v_view, inp.kq_mask, kq_scale, 0.0, 0.0)
        } else {
            // [head_dim, heads, T] → permute → [head_dim, T, heads]
            let q = ctx.permute(q, 0, 2, 1, 3);
            let k_view = ctx.permute(k_view, 0, 2, 1, 3); // [head_dim, n_kv, n_head_kv]

            // kq = mul_mat(k, q) → [n_kv, T, heads] (GQA broadcast r2)
            let mut kq = ctx.mul_mat(k_view, q);
            kq = ctx.soft_max_ext(kq, Some(inp.kq_mask), kq_scale, 0.0);

            // non-FA v path: [head_dim, n_head_kv, n_kv] → permute → [head_dim, n_kv, n_head_kv]
            // → transpose → [n_kv, head_dim, n_head_kv] → cont
            let v_view = ctx.permute(v_view, 0, 2, 1, 3);
            let v_t = ctx.transpose(v_view);
            let v_c = ctx.cont(v_t);
            let kqv = ctx.mul_mat(v_c, kq); // [head_dim, T, heads]

            // [head_dim, heads, T] → cont_2d → [n_embd, T] (ggml_cont_2d)
            let kqv = ctx.permute(kqv, 0, 2, 1, 3);
            let kqv = ctx.cont(kqv);
            ctx.reshape_2d(kqv, p.n_embd_head_v * p.n_head, t)
        };
        let attn_out = crate::adapter::lora_mm(ctx, lw.wo, kqv);

        // residual
        let ffn_inp = ctx.add(attn_out, inp_sa);

        // ffn_norm → gate/up → silu(gate)*up → down
        let mut cur = ctx.rms_norm(ffn_inp, p.norm_eps);
        cur = ctx.mul(cur, lw.ffn_norm);
        let gate = crate::adapter::lora_mm(ctx, lw.ffn_gate, cur);
        let up = crate::adapter::lora_mm(ctx, lw.ffn_up, cur);
        let silu_gate = ctx.silu(gate);
        let prod = ctx.mul(silu_gate, up);
        let down = crate::adapter::lora_mm(ctx, lw.ffn_down, prod);
        inp_l = ctx.add(down, ffn_inp);
    }

    // final norm + lm_head
    let mut cur = ctx.rms_norm(inp_l, p.norm_eps);
    cur = ctx.mul(cur, w.output_norm);
    // res->t_embd = cur (qwen2.cpp:142) — before the lm_head
    let embd = cur;
    // the head runs over the inp_out_ids rows only (the port's placement of
    // qwen2.cpp:106-108's `il == n_layer - 1` gather — see `out_rows`)
    let cur = out_rows(ctx, cur, inp.out_ids);
    let logits = crate::adapter::lora_mm(ctx, w.output, cur); // [n_vocab, n_outputs]
    graph.build_forward(ctx, logits);
    ForwardResult {
        logits,
        graph,
        embd: Some(embd),
    }
}

/// FA branch of `build_attn_mha` (llama-graph.cpp:2626-2669) shared by the
/// qwen2 builder and graph_arch's `attn_kv_cached`.
///
/// Inputs (C conventions, dim0 contiguous — see ggml/src/flash_attn.rs docs):
///   * `q`      [DK, H, T] post-rope F32 (C `q` before the permutes)
///   * `k_view` [DK, H_kv, n_kv] F16 kv-cache view (`llama_kv_cache::get_k`,
///              llama-kv-cache.cpp:1266-1284)
///   * `v_view` [DV, H_kv, n_kv] F16 kv-cache view (`get_v` "!v_trans" branch,
///              llama-kv-cache.cpp:1299-1307)
///   * `mask`   F16 [n_kv, T] contiguous — the FA path *requires* F16
///              (llama-graph.cpp:38-39 "flash attention requires an f16 mask")
/// Output: [DV*H, T] F32.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_core(
    ctx: &mut Context,
    q: TensorId,
    k_view: TensorId,
    v_view: TensorId,
    mask: TensorId,
    kq_scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> TensorId {
    flash_attn_core_sinks(
        ctx,
        q,
        k_view,
        v_view,
        mask,
        None,
        kq_scale,
        max_bias,
        logit_softcap,
    )
}

/// FA with per-head attention sinks (C `ggml_flash_attn_ext_add_sinks`,
/// llama-graph.cpp:2646-2647) — src[4] of the FlashAttnExt node.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_core_sinks(
    ctx: &mut Context,
    q: TensorId,
    k_view: TensorId,
    v_view: TensorId,
    mask: TensorId,
    sinks: Option<TensorId>,
    kq_scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> TensorId {
    // C: q/k/v = ggml_permute(ctx0, ., 0, 2, 1, 3) — llama-graph.cpp:2620-2622
    let q_fa = ctx.permute(q, 0, 2, 1, 3); // [DK, T, H]
    let k_fa = ctx.permute(k_view, 0, 2, 1, 3); // [DK, n_kv, H_kv]
    let v_fa = ctx.permute(v_view, 0, 2, 1, 3); // [DV, n_kv, H_kv]
                                                // v_trans: C checks `v->nb[1] > v->nb[2]` on the pre-permute view and only
                                                // then transposes (llama-graph.cpp:2613/2630-2632). Our cache stores
                                                // v = [n_embd_v_gqa, size] (kv_cache.rs:6-7 "v_trans = false layout"), i.e.
                                                // nb[1] = row_size(head_dim) <= nb[2] = row_size(n_embd_v_gqa) → no
                                                // transpose. (The non-FA branch above *does* transpose+cont, exactly C's
                                                // `if (!v_trans)` at llama-graph.cpp:2709-2713.)
                                                // k/v are already F16 (the cache type), so C's F32→F16 casts
                                                // (llama-graph.cpp:2635-2641) are no-ops here.
    let cur = ctx.flash_attn_ext_sinks(
        q_fa,
        k_fa,
        v_fa,
        Some(mask),
        sinks,
        kq_scale,
        max_bias,
        logit_softcap,
    );
    // C: ggml_reshape_2d(cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3])
    // (llama-graph.cpp:2664) — FA output is [DV, H, T, S], flatten to [DV*H, T].
    let ne = *ctx.ne(cur);
    ctx.reshape_2d(cur, ne[0] * ne[1], ne[2] * ne[3])
}

/// The `skip` predicate of `set_input_kq_mask_impl` (llama-kv-cache.cpp:1634-1691):
/// true = the (key position `p0`, query position `p1`) pair is masked out.
/// `p0 < 0` is an empty cell (`cells.is_empty(j)` → `goto skip`, :1638-1641).
pub fn kq_mask_masked(p0: i32, p1: i32, n_swa: u32, swa_type: LlamaSwaType) -> bool {
    if p0 < 0 {
        return true;
    }
    // mask future tokens (:1676-1678)
    if p0 > p1 {
        return true;
    }
    // apply SWA if any (:1682-1691)
    LlamaHparams::is_masked_swa(n_swa, swa_type, p0, p1)
}

/// Fill a KQ mask row set for a decode step: rows = the cache's cell positions
/// `kv_pos` in cell order, cols = the `q_pos` queries. 0 attend / -inf masked.
/// 对照 llama_kv_cache::set_input_kq_mask 的 non-FA F32 填充，含 SWA 分支
/// (`is_masked_swa`, llama-kv-cache.cpp:1688) 与空 cell 的 `goto skip` (:1640)。
///
/// Mask tensor is [n_kv, n_tokens]: element (kv index s, query t) lives at
/// `s + t*n_kv` — the softmax kernel reads row t = query, column s
/// (same layout DecodeContext and graph_arch fill; the earlier s*n_tokens+t
/// variant was an inverted trap, never called by the working paths).
/// `n_swa = 0` / `swa_type = NONE` is the plain causal mask (the same `skip`
/// chain, SWA branch off), which is what `fill_causal_mask` wraps.
pub fn fill_kq_mask(
    mask: &mut [f32],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
) {
    fill_kq_mask_impl(mask, kv_pos, q_pos, n_swa, swa_type, false)
}

/// The ALiBi twin of [`fill_kq_mask`]: `hparams.use_alibi` changes the *kept*
/// value from `mask_keep` (0) to `-|p0 - p1|` (llama-kv-cache.cpp:1692-1697,
/// `data[idst + j] = -std::abs(p0 - p1)`); masked cells still get -INFINITY
/// (`mask_drop`, :1572). The distance rides the mask because the softmax
/// kernel applies it per-head as `wp[i] += slope * mask[i]` (ops.cpp:5636-
/// 5637), so each head's slope scales the offset. The `skip` chain (empty
/// cell / other sequence / future / SWA) is byte-identical to the non-alibi
/// template — only the `alibi` bool of `set_input_kq_mask_impl` flips, which
/// also disables C's per-sequence mask-reuse shortcut (:1608 "not compatible
/// with Alibi position encoding"; the port fills every cell anyway).
pub fn fill_kq_mask_alibi(
    mask: &mut [f32],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
) {
    fill_kq_mask_impl(mask, kv_pos, q_pos, n_swa, swa_type, true)
}

/// shared body of [`fill_kq_mask`] / [`fill_kq_mask_alibi`] — the `<bool alibi>`
/// template parameter of `set_input_kq_mask_impl` (llama-kv-cache.cpp:1566).
fn fill_kq_mask_impl(
    mask: &mut [f32],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
    alibi: bool,
) {
    let n_kv = kv_pos.len();
    for (iq, &qp) in q_pos.iter().enumerate() {
        for (ik, &kp) in kv_pos.iter().enumerate() {
            mask[iq * n_kv + ik] = if kq_mask_masked(kp, qp, n_swa, swa_type) {
                f32::NEG_INFINITY
            } else if alibi {
                // llama-kv-cache.cpp:1693: -std::abs(p0 - p1) as float
                -((qp - kp).abs() as f32)
            } else {
                0.0
            };
        }
    }
}

/// Build the causal KQ mask contents for a decode step (no SWA).
pub fn fill_causal_mask(mask: &mut [f32], kv_pos: &[i32], q_pos: &[i32]) {
    fill_kq_mask(mask, kv_pos, q_pos, 0, LlamaSwaType::NONE)
}

/// F16 twin of `fill_kq_mask` for the FA path (same layout, same
/// 0 / -INFINITY values — C: llama-kv-cache.cpp:1571-1572 `mask_keep =
/// llama_cast<T>(0.0f)`, `mask_drop = llama_cast<T>(-INFINITY)`, element
/// (kv j, token i) at `i*n_kv + j`, llama-kv-cache.cpp:1605). Both values are
/// exactly representable in F16, so the cast adds no rounding.
pub fn fill_kq_mask_f16(
    mask: &mut [half::f16],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
) {
    fill_kq_mask_f16_impl(mask, kv_pos, q_pos, n_swa, swa_type, false)
}

/// F16 ALiBi twin of [`fill_kq_mask_alibi`] — the FA mask carries the same
/// `-|p0 - p1|` kept value through `llama_cast<ggml_fp16_t>` (the template
/// instantiates on T = F16 for flash attention, llama-kv-cache.cpp:1710-1716),
/// so distances beyond 2048 round to F16 exactly like the reference's.
pub fn fill_kq_mask_alibi_f16(
    mask: &mut [half::f16],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
) {
    fill_kq_mask_f16_impl(mask, kv_pos, q_pos, n_swa, swa_type, true)
}

fn fill_kq_mask_f16_impl(
    mask: &mut [half::f16],
    kv_pos: &[i32],
    q_pos: &[i32],
    n_swa: u32,
    swa_type: LlamaSwaType,
    alibi: bool,
) {
    let n_kv = kv_pos.len();
    for (iq, &qp) in q_pos.iter().enumerate() {
        for (ik, &kp) in kv_pos.iter().enumerate() {
            mask[iq * n_kv + ik] = if kq_mask_masked(kp, qp, n_swa, swa_type) {
                half::f16::NEG_INFINITY
            } else if alibi {
                half::f16::from_f32(-((qp - kp).abs() as f32))
            } else {
                half::f16::ZERO
            };
        }
    }
}

/// F16 twin of `fill_causal_mask` (no SWA).
pub fn fill_causal_mask_f16(mask: &mut [half::f16], kv_pos: &[i32], q_pos: &[i32]) {
    fill_kq_mask_f16(mask, kv_pos, q_pos, 0, LlamaSwaType::NONE)
}

/// Convenience: n_embd_gqa helpers matching hparams calculators.
pub fn n_embd_k_gqa(p: &AttnParams) -> i64 {
    p.n_head_kv * p.n_embd_head_k
}
pub fn n_embd_v_gqa(p: &AttnParams) -> i64 {
    p.n_head_kv * p.n_embd_head_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_fill() {
        let mut m = vec![0f32; 3 * 2];
        fill_causal_mask(&mut m, &[0, 1, 2], &[1, 3]);
        // [n_kv=3, n_tokens=2]: element (s, t) at s + t*3
        // query t=0 (pos 1): attend kv s=0,1; mask s=2
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[2], f32::NEG_INFINITY);
        // query t=1 (pos 3): all kv attend
        assert_eq!(m[3], 0.0);
        assert_eq!(m[4], 0.0);
        assert_eq!(m[5], 0.0);
    }

    /// F16 twin of the fill: same 0 / -inf pattern, same layout.
    #[test]
    fn causal_mask_fill_f16() {
        let mut m = vec![half::f16::ZERO; 3 * 2];
        fill_causal_mask_f16(&mut m, &[0, 1, 2], &[1, 3]);
        let v: Vec<f32> = m.iter().map(|x| x.to_f32()).collect();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], f32::NEG_INFINITY); // exact in F16, no rounding
        assert_eq!(&v[3..], &[0.0, 0.0, 0.0]);
    }

    /// The SWA branch of `set_input_kq_mask` (llama-kv-cache.cpp:1682-1691):
    /// a key is dropped once `p1 - p0 >= n_swa` (LLAMA_SWA_TYPE_STANDARD,
    /// llama-hparams.h:479-484), and a cell outside the window is *not* the
    /// same thing as an empty cell — the port must mask both.
    #[test]
    fn swa_mask_window_standard() {
        let n_swa = 4u32;
        // queries at 5 and 9 over cell positions 0..10, STANDARD
        let kv_pos: Vec<i32> = (0..10).collect();
        let q_pos = [5i32, 9];
        let mut m = vec![0f32; 2 * 10];
        fill_kq_mask(&mut m, &kv_pos, &q_pos, n_swa, LlamaSwaType::STANDARD);
        let at = |t: usize, s: usize| m[t * 10 + s];
        // query 5 (row 0): keeps p0 in [2, 5] — 5 - p0 < 4
        assert_eq!(at(0, 1), f32::NEG_INFINITY, "5-1 >= 4 -> out of window");
        assert_eq!(at(0, 2), 0.0, "5-2 = 3 < 4");
        assert_eq!(at(0, 5), 0.0);
        assert_eq!(at(0, 6), f32::NEG_INFINITY, "causal");
        // query 9 (row 1): keeps p0 in [6, 9]
        assert_eq!(at(1, 5), f32::NEG_INFINITY);
        assert_eq!(at(1, 6), 0.0);
        assert_eq!(at(1, 9), 0.0);
        // SWA off the mask is the plain causal one
        let mut c = vec![0f32; 2 * 10];
        fill_causal_mask(&mut c, &kv_pos, &q_pos);
        assert_eq!(c[1], 0.0, "no window: p0=1 is visible for query 5");
        assert_eq!(c[10 + 5], 0.0);
        assert_eq!(c[10 + 9], 0.0);
        // ... which is exactly what the window drops: same cell, query 9
        assert_eq!(c[10], 0.0);
        assert_eq!(at(1, 0), f32::NEG_INFINITY, "9-0 >= 4 -> out of window");
        // empty cells are dropped whatever the SWA type (llama-kv-cache.cpp:1640)
        let mut e = vec![0f32; 1 * 3];
        fill_kq_mask(&mut e, &[-1, 0, 1], &[1], 0, LlamaSwaType::NONE);
        assert_eq!(e, vec![f32::NEG_INFINITY, 0.0, 0.0]);
        // F16 twin gives the same pattern
        let mut h = vec![half::f16::ZERO; 2 * 10];
        fill_kq_mask_f16(&mut h, &kv_pos, &q_pos, n_swa, LlamaSwaType::STANDARD);
        let hf: Vec<f32> = h.iter().map(|x| x.to_f32()).collect();
        assert_eq!(hf, m);
    }

    /// The SWA slot search's `can_use` test and the mask must agree: a cell is
    /// reusable exactly when `is_masked_swa(n_swa, swa_type, pos_cell,
    /// seq_pos_max + 1)` (llama-kv-cache.cpp:1055) — i.e. when the new query
    /// position would have dropped it anyway.
    #[test]
    fn swa_mask_agrees_with_slot_reuse() {
        // CHUNKED (llama4) and SYMMETRIC (modern-bert) are ported through the
        // same table; pin all four branches of llama-hparams.h:472-495
        assert!(!kq_mask_masked(0, 100, 4, LlamaSwaType::NONE));
        assert!(!kq_mask_masked(3, 4, 4, LlamaSwaType::STANDARD));
        assert!(kq_mask_masked(0, 4, 4, LlamaSwaType::STANDARD));
        assert!(kq_mask_masked(3, 4, 4, LlamaSwaType::CHUNKED));
        assert!(!kq_mask_masked(4, 7, 4, LlamaSwaType::CHUNKED));
        assert!(kq_mask_masked(0, 3, 4, LlamaSwaType::SYMMETRIC));
        assert!(!kq_mask_masked(3, 4, 4, LlamaSwaType::SYMMETRIC));
    }

    /// The ALiBi kept value of `set_input_kq_mask_impl<.., alibi=true>`
    /// (llama-kv-cache.cpp:1692-1697): a kept (p0, p1) cell carries
    /// `-|p0 - p1|` instead of 0, masked cells stay -INFINITY — hand-computed
    /// for kv positions [0,1,2,3,5] and queries [2, 4]:
    ///   * query 2: kv 0 → -2, kv 1 → -1, kv 2 → -0, kv 3 (future) → -inf,
    ///     kv 5 (future) → -inf
    ///   * query 4: kv 0 → -4, kv 1 → -3, kv 2 → -2, kv 3 → -1, kv 5 → -inf
    /// Empty cells (p0 < 0) stay -inf in both modes (:1640 `goto skip`).
    #[test]
    fn alibi_mask_kept_value_is_neg_distance() {
        let kv_pos = [0i32, 1, 2, 3, 5];
        let mut m = vec![0f32; 2 * 5];
        fill_kq_mask_alibi(&mut m, &kv_pos, &[2, 4], 0, LlamaSwaType::NONE);
        assert_eq!(
            m[..5].to_vec(),
            vec![-2.0, -1.0, -0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        assert_eq!(
            m[5..].to_vec(),
            vec![-4.0, -3.0, -2.0, -1.0, f32::NEG_INFINITY]
        );
        // non-alibi twin: kept cells are exactly 0
        let mut c = vec![0f32; 2 * 5];
        fill_kq_mask(&mut c, &kv_pos, &[2, 4], 0, LlamaSwaType::NONE);
        assert_eq!(
            c[..5].to_vec(),
            vec![0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        // empty cells drop in the alibi mask too
        let mut e = vec![0f32; 3];
        fill_kq_mask_alibi(&mut e, &[-1, 0, 3], &[3], 0, LlamaSwaType::NONE);
        assert_eq!(e, vec![f32::NEG_INFINITY, -3.0, -0.0]);
    }

    /// F16 twin: the same -|p0-p1| values through `llama_cast<ggml_fp16_t>`
    /// (the FA mask template, llama-kv-cache.cpp:1710-1716). Small distances
    /// are exact in F16; distance 2049 must round like the reference's cast
    /// (f16 has 11-bit significand → integers > 2048 round to even).
    #[test]
    fn alibi_mask_f16_rounding() {
        let kv_pos = [0i32, 1, 2, 3, 5];
        let mut h = vec![half::f16::ZERO; 2 * 5];
        fill_kq_mask_alibi_f16(&mut h, &kv_pos, &[2, 4], 0, LlamaSwaType::NONE);
        let v: Vec<f32> = h.iter().map(|x| x.to_f32()).collect();
        assert_eq!(
            v[..5].to_vec(),
            vec![-2.0, -1.0, -0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        assert_eq!(
            v[5..].to_vec(),
            vec![-4.0, -3.0, -2.0, -1.0, f32::NEG_INFINITY]
        );

        // long-context rounding: integers up to 2048 are exact in F16; above
        // that the spacing is 2, so 2049 ties-to-even → 2048 while 2050 stays
        // (same `llama_cast` round-to-nearest-even as the reference)
        let mut big = vec![half::f16::ZERO; 3];
        fill_kq_mask_alibi_f16(&mut big, &[0, 1, 2], &[2050], 0, LlamaSwaType::NONE);
        let bv: Vec<f32> = big.iter().map(|x| x.to_f32()).collect();
        assert_eq!(bv, vec![-2050.0, -2048.0, -2048.0]);

        // the alibi mask composes with SWA exactly like the plain one: the
        // window drop precedes the kept-value write (llama-kv-cache.cpp:1686
        // `goto skip` wins over the :1693 alibi value)
        let mut s = vec![0f32; 2 * 4];
        fill_kq_mask_alibi(&mut s, &[0, 2, 3, 4], &[3, 4], 2, LlamaSwaType::STANDARD);
        // query 3: kv 0 out of window (3-0 >= 2), kv 2 → -1, kv 3 → -0,
        //          kv 4 future → -inf
        assert_eq!(
            s[..4].to_vec(),
            vec![f32::NEG_INFINITY, -1.0, -0.0, f32::NEG_INFINITY]
        );
        // query 4: kv 0/2 out of window, kv 3 → -1, kv 4 → -0
        assert_eq!(
            s[4..].to_vec(),
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -1.0, -0.0]
        );
    }

    /// `flash_attn_core` layout check: q [DK, H, T] / k,v cache views
    /// [D, H_kv, n_kv] in, [DV*H, T] out, computed against a direct softmax.
    /// This pins the permute/permute-free expectations of llama-graph.cpp:
    /// 2620-2622 (q/k/v → [D, T/n_kv, H/H_kv]) and :2664 (reshape).
    #[test]
    fn flash_attn_core_layout_and_values() {
        use ggml::compute::graph_compute;
        use ggml::graph::Graph;

        let (dk, dv, h, h_kv, t, n_kv) = (4i64, 4i64, 2i64, 1i64, 2i64, 2i64);
        let mut ctx = Context::new();

        // q [DK, H, T] F32 — deterministic values
        let q = ctx.new_tensor_3d(GgmlType::F32, dk, h, t);
        ctx.arena_resize_tensor(q);
        let mut qi = 0f32;
        ctx.with_f32_mut(q, |p| {
            for v in p.iter_mut() {
                qi += 1.0;
                *v = (qi * 0.37).sin();
            }
        })
        .unwrap();

        // k/v caches [DK*H_kv, n_kv] F16 (the kv_cache.rs layout) + views
        let k_cache = ctx.new_tensor_2d(GgmlType::F16, dk * h_kv, n_kv);
        let v_cache = ctx.new_tensor_2d(GgmlType::F16, dv * h_kv, n_kv);
        ctx.arena_resize_tensor(k_cache);
        ctx.arena_resize_tensor(v_cache);
        let kvals: Vec<f32> = (0..dk * h_kv * n_kv)
            .map(|i| (i as f32 * 0.29).cos())
            .collect();
        let vvals: Vec<f32> = (0..dv * h_kv * n_kv)
            .map(|i| (i as f32 * 0.53).sin())
            .collect();
        ctx.data_bytes_mut(k_cache)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(
                &kvals
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect::<Vec<_>>(),
            ));
        ctx.data_bytes_mut(v_cache)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(
                &vvals
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect::<Vec<_>>(),
            ));
        let rs_head = GgmlType::F16.row_size(dk as usize);
        let rs_gqa = GgmlType::F16.row_size((dk * h_kv) as usize);
        let k_view = ctx.view_4d(
            k_cache,
            dk,
            h_kv,
            n_kv,
            1,
            rs_head,
            rs_gqa,
            rs_gqa * n_kv as usize,
            0,
        );
        let rs_head_v = GgmlType::F16.row_size(dv as usize);
        let rs_gqa_v = GgmlType::F16.row_size((dv * h_kv) as usize);
        let v_view = ctx.view_4d(
            v_cache,
            dv,
            h_kv,
            n_kv,
            1,
            rs_head_v,
            rs_gqa_v,
            rs_gqa_v * n_kv as usize,
            0,
        );

        // F16 causal mask [n_kv, T]
        let mask = ctx.new_tensor_2d(GgmlType::F16, n_kv, t);
        ctx.arena_resize_tensor(mask);
        {
            let m: &mut [half::f16] = bytemuck::cast_slice_mut(ctx.data_bytes_mut(mask).unwrap());
            fill_causal_mask_f16(m, &[0, 1], &[0, 1]);
        }

        let scale = 1.0 / (dk as f32).sqrt();
        let out = flash_attn_core(&mut ctx, q, k_view, v_view, mask, scale, 0.0, 0.0);

        // the FA op must be in the graph, and `out` its [DV*H, T] reshape
        let ne = *ctx.ne(out);
        assert_eq!(ne, [dv * h, t, 1, 1]);
        assert_eq!(ctx.op(out), ggml::tensor::GgmlOp::Reshape);
        let fa = ctx.src(out)[0].expect("reshape src");
        assert_eq!(
            ctx.op(fa),
            ggml::tensor::GgmlOp::FlashAttnExt,
            "FA branch must emit GGML_OP_FLASH_ATTN_EXT"
        );
        assert_eq!(
            *ctx.ne(fa),
            [dv, h, t, 1],
            "dst = {{v->ne0, q->ne2, q->ne1, q->ne3}} (ggml.c:5526)"
        );
        assert_eq!(
            *ctx.ne(ctx.src(fa)[0].unwrap()),
            [dk, t, h, 1],
            "FA q is [DK, T, H]"
        );
        assert_eq!(
            *ctx.ne(ctx.src(fa)[1].unwrap()),
            [dk, n_kv, h_kv, 1],
            "FA k is [DK, S_kv, H_kv]"
        );
        assert_eq!(
            *ctx.ne(ctx.src(fa)[2].unwrap()),
            [dv, n_kv, h_kv, 1],
            "FA v is [DV, S_kv, H_kv]"
        );

        let mut g = Graph::new(16);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, 1);

        // naive: for each (t, head) softmax over kv s <= t
        let qf: Vec<f32> = ctx
            .data_bytes(q)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let qat = |d: i64, hh: i64, tt: i64| qf[(d + hh * dk + tt * dk * h) as usize];
        let kat = |d: i64, s: i64| f16(&kvals, d + s * (dk * h_kv));
        let vat = |d: i64, s: i64| f16(&vvals, d + s * (dv * h_kv));
        fn f16(vals: &[f32], i: i64) -> f32 {
            half::f16::from_f32(vals[i as usize]).to_f32()
        }
        let of: Vec<f32> = ctx
            .data_bytes(out)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for tt in 0..t {
            for hh in 0..h {
                let mut sw = Vec::new();
                for s in 0..=tt {
                    let dot: f32 = (0..dk).map(|d| kat(d, s) * qat(d, hh, tt)).sum();
                    sw.push((dot * scale).exp());
                }
                let sum: f32 = sw.iter().sum();
                for d in 0..dv {
                    let want: f32 = sw
                        .iter()
                        .enumerate()
                        .map(|(s, w)| w / sum * vat(d, s as i64))
                        .sum();
                    // out is [DV*H, T]: element (d, hh, tt) at d + hh*dv + tt*dv*h
                    let got = of[(d + hh * dv + tt * dv * h) as usize];
                    // 1e-3: F16 V legitimately uses the fp16 VKQ accumulator
                    // (ops.cpp:8773), the naive reference below does not.
                    assert!(
                        (got - want).abs() < 1e-3,
                        "t={tt} h={hh} d={d}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
