//! gemma4-assistant — Gemma 4's speculative draft head, the port of
//! `src/models/gemma4-assistant.cpp` (pinned bd4f514db1) plus the
//! `is_mem_shared` mode of `common_speculative_impl_draft_mtp`
//! (common/speculative.cpp:1343-1424):
//!
//!   * `llama_model_gemma4_assistant::load_arch_hparams`
//!     (gemma4-assistant.cpp:5-21) — the `gemma4-assistant.*` GGUF keys;
//!   * `llama_model_gemma4_assistant::load_arch_tensors` (:26-79) — the head
//!     tensor set: `nextn.pre_projection` / `nextn.post_projection`, the
//!     tied `token_embd` (== `output`, TENSOR_DUPLICATED) and one
//!     Q-only decoder block per nextn layer (no `wk`/`wv` — the attention
//!     reads the *target's* KV cache);
//!   * `llama_model_gemma4_assistant::graph::graph` (:84-190) — the dual
//!     stream concat (the target's scaled token embedding + the `inp_h`
//!     backbone hidden) → `nextn_proj_pre` → the Q-only iswa attention over
//!     the shared KV → the GELU FFN with the per-layer `out_scale` →
//!     `output_norm` → the tied lm head + `nextn_proj_post` (`t_h_nextn`).
//!
//! The head is *attached to the target context* (the inverse of the C's
//! `cparams.ctx_other`): the C's draft graph reads the target model's
//! `tok_embd` through the ctx_other pointer (:105) and its KV cache through
//! `llama_kv_cache_iswa`'s `share` callback (llama-model.cpp:2687-2707:
//! assistant layer `il` shares the target's layer `n_layer_tgt - 1`, or
//! `n_layer_tgt - 2` when the assistant layer is SWA). The port's tensors are
//! Context-local, so the head's weights are materialized **in the target's
//! ggml Context** (mmap-backed, zero copy — the eagle.rs precedent) and the
//! draft graph is built over the target's `KvCache` directly
//! ([`crate::context::DecodeContext::decode_gemma4_assistant`]).

use std::sync::Arc;

use ggml::{Context, Graph, GgmlType, Gguf, TensorId};

use crate::batch::LlamaBatch;
use crate::graph;
use crate::graph::ForwardResult;
use crate::kv_cache::KvCache;

// ---------------------------------------------------------------------------
// weights + params
// ---------------------------------------------------------------------------

/// one Q-only decoder block (gemma4-assistant.cpp:58-76)
pub struct Gemma4AssistantLayerWeights {
    pub attn_norm: TensorId,
    pub wq: TensorId,
    pub wo: TensorId,
    pub attn_q_norm: TensorId,
    pub attn_post_norm: TensorId,
    /// `layer_output_scale` — {1}, REQUIRED for the head (:69)
    pub out_scale: TensorId,
    /// the proportional-rope factors on the non-SWA layers (:71-74)
    pub rope_freqs: Option<TensorId>,
    pub ffn_norm: TensorId,
    pub ffn_gate: TensorId,
    pub ffn_up: TensorId,
    pub ffn_down: TensorId,
    pub ffn_post_norm: TensorId,
}

pub struct Gemma4AssistantWeights {
    /// `token_embd` — {n_embd, n_vocab}; the head's own copy
    pub tok_embd: TensorId,
    /// `output` — the TENSOR_DUPLICATED re-request of `token_embd` (:35)
    pub output: TensorId,
    pub output_norm: TensorId,
    /// `nextn.pre_projection` — {2*n_embd_backbone, n_embd} (:55)
    pub nextn_proj_pre: TensorId,
    /// `nextn.post_projection` — {n_embd, n_embd_backbone} (:43)
    pub nextn_proj_post: TensorId,
    pub layers: Vec<Gemma4AssistantLayerWeights>,
}

#[derive(Clone)]
pub struct Gemma4AssistantParams {
    /// `embedding_length` — the head's own hidden size (1024)
    pub n_embd: i64,
    /// `embedding_length_out` — the target's hidden size (2816); the loader
    /// throws when it equals n_embd (gemma4-assistant.cpp:39-41)
    pub n_embd_backbone: i64,
    pub n_vocab: i64,
    pub norm_eps: f32,
    pub n_ff: i64,
    pub rope_mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    /// `rope.freq_base_swa` (optional)
    pub freq_base_swa: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub use_flash_attn: bool,
    /// `llama_model_n_layer(model_other)` — the share mapping's anchor
    pub n_layer_tgt: usize,
    /// per assistant layer: `hparams.is_swa(il)` / `n_head(il)` /
    /// `n_embd_head_k(il)` (== v, asserted at load) / `n_rot(il)`
    pub is_swa: Vec<bool>,
    pub n_head: Vec<i64>,
    pub n_embd_head: Vec<i64>,
    pub n_rot: Vec<i64>,
    /// the shared target layers' cache geometry — (head dim, kv heads) of
    /// target layer `n_layer_tgt - 1` (the non-SWA share, llama-model.cpp:
    /// 2700-2703) and `n_layer_tgt - 2` (the SWA share, :2698-2699)
    pub tgt_full: (i64, i64),
    pub tgt_swa: (i64, i64),
}

// ---------------------------------------------------------------------------
// the loader (eagle.rs's HeadLoader recipe, into the TARGET's Context)
// ---------------------------------------------------------------------------

/// one created tensor + its gguf name (the create_tensor mirror: exact shape
/// check, external mmap storage — the tensors land in the *target's* Context)
struct HeadLoader<'a> {
    gguf: &'a Gguf,
    mmap: Arc<memmap2::Mmap>,
    ctx: &'a mut Context,
    by_name: std::collections::HashMap<String, TensorId>,
    n_created: usize,
}

impl<'a> HeadLoader<'a> {
    fn new(gguf: &'a Gguf, mmap: Arc<memmap2::Mmap>, ctx: &'a mut Context) -> Self {
        Self {
            gguf,
            mmap,
            ctx,
            by_name: std::collections::HashMap::new(),
            n_created: 0,
        }
    }

    /// `llama_model_loader::create_tensor` restricted to the head's needs:
    /// exact dims, required-or-optional, external mmap storage. A repeated
    /// name is the TENSOR_DUPLICATED re-request (`output` over `token_embd`,
    /// :34-35).
    fn create_tensor(
        &mut self,
        name: &str,
        ne: &[i64],
        required: bool,
    ) -> Result<Option<TensorId>, String> {
        if let Some(&prev) = self.by_name.get(name) {
            for i in 0..4 {
                let want = if i < ne.len() { ne[i] } else { 1 };
                if want != self.ctx.ne(prev)[i] {
                    return Err(format!(
                        "tensor '{name}' has wrong shape; expected [{:?}], got [{:?}]",
                        &ne[..ne.len().min(4)],
                        self.ctx.ne(prev)
                    ));
                }
            }
            return Ok(Some(prev));
        }
        let Some(ti) = self.gguf.find_tensor(name) else {
            if !required {
                return Ok(None);
            }
            return Err(format!("tensor '{name}' not found"));
        };
        for i in 0..4 {
            let want = if i < ne.len() { ne[i] } else { 1 };
            if want != ti.ne[i] {
                return Err(format!(
                    "tensor '{name}' has wrong shape; expected [{:?}], got [{:?}]",
                    &ne[..ne.len().min(4)],
                    ti.ne
                ));
            }
        }
        let id = self.ctx.new_tensor(ti.ty, ti.ne);
        self.ctx.set_name(id, name);
        // absolute file offset of this tensor's storage inside the mmap
        let data_off = self.gguf.data_offset as usize + ti.offset as usize;
        self.ctx
            .set_external_storage(id, self.mmap.clone(), data_off);
        self.by_name.insert(name.to_string(), id);
        self.n_created += 1;
        Ok(Some(id))
    }
}

/// `load_gemma4_assistant_head` — the hparams + tensors of
/// gemma4-assistant.cpp:5-79, created in `ctx` (the **target's** Context —
/// see the module docs). `tgt_full` / `tgt_swa` carry the shared target
/// layers' (head dim, kv heads) geometry for the KV views.
#[allow(clippy::too_many_arguments)]
pub fn load_gemma4_assistant_head(
    gguf: &Gguf,
    mmap: Arc<memmap2::Mmap>,
    ctx: &mut Context,
    fa: bool,
    n_layer_tgt: usize,
    tgt_full: (i64, i64),
    tgt_swa: (i64, i64),
) -> Result<(Gemma4AssistantWeights, Gemma4AssistantParams), String> {
    let arch = "gemma4-assistant";

    // ---- the generic dims (llama_model_base::load_hparams) ----
    let n_embd = gguf
        .get_u32(&format!("{arch}.embedding_length"))
        .ok_or_else(|| format!("key {arch}.embedding_length not found in model file"))?
        as i64;
    // gemma4-assistant.cpp:39-41 — the out width must carry the target hidden
    let n_embd_backbone = gguf
        .get_u32(&format!("{arch}.embedding_length_out"))
        .ok_or_else(|| {
            format!(
                "Gemma 4 assistant requires embedding_length_out to carry the target hidden \
                 size (key {arch}.embedding_length_out not found)"
            )
        })? as i64;
    if n_embd_backbone == n_embd {
        return Err(
            "Gemma 4 assistant requires embedding_length_out to carry the target hidden size"
                .into(),
        );
    }
    let _n_layer_all = gguf
        .get_u32(&format!("{arch}.block_count"))
        .ok_or_else(|| format!("key {arch}.block_count not found in model file"))?
        as usize;
    let n_layer_nextn = gguf
        .get_u32(&format!("{arch}.nextn_predict_layers"))
        .unwrap_or(0) as usize;
    let n_vocab = gguf
        .find_tensor("token_embd.weight")
        .map(|t| t.ne[1])
        .ok_or("tensor 'token_embd.weight' not found")?;

    // ---- load_arch_hparams (gemma4-assistant.cpp:5-21) ----
    let norm_eps = gguf
        .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .ok_or_else(|| {
            format!("key {arch}.attention.layer_norm_rms_epsilon not found in model file")
        })?;
    // `attention.sliding_window` (:17) — read like the C; the window binds
    // through the shared target cache, not the head's own (it has none)
    let _n_swa = gguf
        .get_u32(&format!("{arch}.attention.sliding_window"))
        .ok_or_else(|| format!("key {arch}.attention.sliding_window not found in model file"))?;
    let is_swa_impl: Vec<bool> = {
        let arr = gguf
            .find_key(&format!("{arch}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                format!(
                    "Gemma4Assistant model requires '{arch}.attention.sliding_window_pattern' \
                     in GGUF metadata"
                )
            })?;
        arr.1
            .iter()
            .map(|v| {
                // the real files write the pattern as a BOOL array; the
                // integer forms are the synthetic convention
                v.as_bool()
                    .or_else(|| v.as_u32().map(|u| u != 0))
                    .or_else(|| v.as_f32().map(|f| f != 0.0))
                    .unwrap_or(false)
            })
            .collect()
    };
    // `attention.shared_kv_layers` (:12-13) — read into a local the C never
    // uses; the head has no K/V of its own regardless
    let _n_kv_shared_layers = gguf.get_u32(&format!("{arch}.attention.shared_kv_layers"));
    let freq_base_swa = gguf
        .get_f32(&format!("{arch}.rope.freq_base_swa"))
        .unwrap_or(0.0);
    let head_k_full = gguf
        .get_u32(&format!("{arch}.attention.key_length"))
        .unwrap_or(0) as i64;
    let head_v_full = gguf
        .get_u32(&format!("{arch}.attention.value_length"))
        .unwrap_or(0) as i64;
    let head_k_swa = gguf
        .get_u32(&format!("{arch}.attention.key_length_swa"))
        .ok_or_else(|| format!("key {arch}.attention.key_length_swa not found in model file"))?
        as i64;
    let head_v_swa = gguf
        .get_u32(&format!("{arch}.attention.value_length_swa"))
        .ok_or_else(|| format!("key {arch}.attention.value_length_swa not found in model file"))?
        as i64;
    // the loader's k == v asserts (:28-33)
    if head_k_full != head_v_full {
        return Err("Gemma 4 assistant requires n_embd_head_k == n_embd_head_v".into());
    }
    if head_k_swa != head_v_swa {
        return Err(
            "Gemma 4 assistant requires n_embd_head_k_swa == n_embd_head_v_swa".into(),
        );
    }
    let n_rot_full = gguf
        .get_u32(&format!("{arch}.rope.dimension_count"))
        .ok_or_else(|| format!("key {arch}.rope.dimension_count not found in model file"))?
        as i64;
    let n_rot_swa = gguf
        .get_u32(&format!("{arch}.rope.dimension_count_swa"))
        .unwrap_or(n_rot_full as u32) as i64;
    let freq_base = gguf.get_f32(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0);
    let n_ctx_train = gguf.get_u32(&format!("{arch}.context_length")).unwrap_or(0) as i32;
    let n_ff = gguf
        .get_u32(&format!("{arch}.feed_forward_length"))
        .unwrap_or(0) as i64;
    let n_head = gguf.get_u32(&format!("{arch}.attention.head_count")).unwrap_or(0) as i64;
    if n_head <= 0 {
        return Err(format!("{arch}: attention.head_count missing or zero"));
    }

    // ---- load_arch_tensors (gemma4-assistant.cpp:26-79) ----
    let mut ld = HeadLoader::new(gguf, mmap, ctx);

    let tok_embd = ld
        .create_tensor("token_embd.weight", &[n_embd, n_vocab], true)?
        .expect("token_embd");
    // `output` — the TENSOR_DUPLICATED re-request of the same tensor (:34-35)
    let output = ld
        .create_tensor("token_embd.weight", &[n_embd, n_vocab], true)?
        .expect("output");
    let output_norm = ld
        .create_tensor("output_norm.weight", &[n_embd], true)?
        .expect("output_norm");
    // masked-embd pair (:37-38) — NOT_REQUIRED, absent from the local files
    ld.create_tensor("masked_embd_centroids.weight", &[], false)?;
    ld.create_tensor("masked_embd_ordering", &[], false)?;

    let nextn_proj_post = ld
        .create_tensor(
            "nextn.post_projection.weight",
            &[n_embd, n_embd_backbone],
            true,
        )?
        .expect("nextn.post_projection");

    // `nextn.pre_projection` — created inside the i == 0 branch (:53-55)
    let mut nextn_proj_pre = None;
    let mut rope_freqs_flag = false; // 0 = required, then TENSOR_DUPLICATED
    let mut layers = Vec::with_capacity(n_layer_nextn);
    for i in 0..n_layer_nextn {
        let is_swa = is_swa_impl
            .get(i)
            .copied()
            .unwrap_or(*is_swa_impl.last().unwrap_or(&false));
        let n_embd_head = if is_swa { head_k_swa } else { head_k_full };

        if i == 0 {
            nextn_proj_pre = Some(
                ld.create_tensor(
                    "nextn.pre_projection.weight",
                    &[2 * n_embd_backbone, n_embd],
                    true,
                )?
                .expect("nextn.pre_projection"),
            );
        }

        let attn_norm = ld
            .create_tensor(&format!("blk.{i}.attn_norm.weight"), &[n_embd], true)?
            .expect("attn_norm");
        let wq = ld
            .create_tensor(
                &format!("blk.{i}.attn_q.weight"),
                &[n_embd, n_embd_head * n_head],
                true,
            )?
            .expect("wq");
        let wo = ld
            .create_tensor(
                &format!("blk.{i}.attn_output.weight"),
                &[n_embd_head * n_head, n_embd],
                true,
            )?
            .expect("wo");
        let attn_q_norm = ld
            .create_tensor(&format!("blk.{i}.attn_q_norm.weight"), &[n_embd_head], true)?
            .expect("attn_q_norm");
        let attn_post_norm = ld
            .create_tensor(&format!("blk.{i}.post_attention_norm.weight"), &[n_embd], true)?
            .expect("attn_post_norm");
        let out_scale = ld
            .create_tensor(&format!("blk.{i}.layer_output_scale.weight"), &[1], true)?
            .expect("out_scale");
        let rope_freqs = if !is_swa {
            // the ROPE_FREQS template carries no blk slot (llama-arch.cpp
            // :611 "rope_freqs") — ONE tensor serves every layer; the first
            // request is required, the rest TENSOR_DUPLICATED (:71-74)
            let t = ld.create_tensor(
                "rope_freqs.weight",
                &[n_embd_head / 2],
                !rope_freqs_flag,
            )?;
            rope_freqs_flag = true;
            t
        } else {
            None
        };
        let ffn_norm = ld
            .create_tensor(&format!("blk.{i}.ffn_norm.weight"), &[n_embd], true)?
            .expect("ffn_norm");
        let ffn_gate = ld
            .create_tensor(&format!("blk.{i}.ffn_gate.weight"), &[n_embd, n_ff], true)?
            .expect("ffn_gate");
        let ffn_up = ld
            .create_tensor(&format!("blk.{i}.ffn_up.weight"), &[n_embd, n_ff], true)?
            .expect("ffn_up");
        let ffn_down = ld
            .create_tensor(&format!("blk.{i}.ffn_down.weight"), &[n_ff, n_embd], true)?
            .expect("ffn_down");
        let ffn_post_norm = ld
            .create_tensor(&format!("blk.{i}.post_ffw_norm.weight"), &[n_embd], true)?
            .expect("ffn_post_norm");

        layers.push(Gemma4AssistantLayerWeights {
            attn_norm,
            wq,
            wo,
            attn_q_norm,
            attn_post_norm,
            out_scale,
            rope_freqs,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_post_norm,
        });
    }

    // done_getting_tensors: every file tensor consumed
    let n_file_tensors = gguf.tensors.len();
    if ld.n_created != n_file_tensors {
        let mut missing: Vec<String> = gguf
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .filter(|n| !ld.by_name.contains_key(n))
            .collect();
        missing.sort_unstable();
        return Err(format!(
            "gemma4-assistant head has {n_file_tensors} tensors but the loader created {} \
             (unknown: {missing:?})",
            ld.n_created
        ));
    }

    let params = Gemma4AssistantParams {
        n_embd,
        n_embd_backbone,
        n_vocab,
        norm_eps,
        n_ff,
        // the GEMMA family ropes NEOX-style — the pairs are offset by
        // n_rot/2 (llama_model_rope_type, llama-model.cpp:3005-3020)
        rope_mode: crate::hparams::LlamaRopeType::NEOX as i32,
        n_ctx_orig: n_ctx_train,
        freq_base,
        freq_scale: 1.0,
        freq_base_swa,
        ext_factor: -1.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        use_flash_attn: fa,
        n_layer_tgt,
        is_swa: (0..n_layer_nextn)
            .map(|i| {
                is_swa_impl
                    .get(i)
                    .copied()
                    .unwrap_or(*is_swa_impl.last().unwrap_or(&false))
            })
            .collect(),
        n_head: vec![n_head; n_layer_nextn],
        n_embd_head: (0..n_layer_nextn)
            .map(|i| {
                let is_swa = is_swa_impl
                    .get(i)
                    .copied()
                    .unwrap_or(*is_swa_impl.last().unwrap_or(&false));
                if is_swa {
                    head_k_swa
                } else {
                    head_k_full
                }
            })
            .collect(),
        n_rot: (0..n_layer_nextn)
            .map(|i| {
                let is_swa = is_swa_impl
                    .get(i)
                    .copied()
                    .unwrap_or(*is_swa_impl.last().unwrap_or(&false));
                if is_swa {
                    n_rot_swa
                } else {
                    n_rot_full
                }
            })
            .collect(),
        tgt_full,
        tgt_swa,
    };

    let weights = Gemma4AssistantWeights {
        tok_embd,
        output,
        output_norm,
        nextn_proj_pre: nextn_proj_pre.expect("nextn.pre_projection (i == 0 branch)"),
        nextn_proj_post,
        layers,
    };
    Ok((weights, params))
}

// ---------------------------------------------------------------------------
// the draft graph — gemma4-assistant.cpp:84-190
// ---------------------------------------------------------------------------

/// `llama_model_gemma4_assistant::graph::graph` (gemma4-assistant.cpp:84-190)
/// over the **target's** KV cache (the `share` mapping of
/// llama-model.cpp:2687-2707: assistant layer il → target layer
/// `n_layer_tgt - 2` when SWA, else `n_layer_tgt - 1`).
#[allow(clippy::too_many_arguments)]
pub fn build_gemma4_assistant_forward(
    ctx: &mut Context,
    w: &Gemma4AssistantWeights,
    p: &Gemma4AssistantParams,
    kv: &KvCache,
    tok_embd_tgt: TensorId,
    tokens: TensorId,
    h: TensorId,
    pos: TensorId,
    kq_mask: TensorId,
    kq_mask_swa: TensorId,
    n_tokens: usize,
) -> ForwardResult {
    let t = n_tokens as i64;
    let mut graph = Graph::new(1024);

    // x = get_rows(model_other->tok_embd, inp_tokens), scaled by
    // sqrt(n_embd_backbone) (:104-107)
    let x = ctx.get_rows(tok_embd_tgt, tokens);
    ctx.set_name(x, "inp_embd_target");
    let x = ctx.scale(x, (p.n_embd_backbone as f32).sqrt());

    // xh = concat(x, inp_h, dim 0) → pre_proj (:109-113)
    let xh = ctx.concat(x, h, 0);
    ctx.set_name(xh, "inp_xh");
    let cur = crate::adapter::lora_mm(ctx, w.nextn_proj_pre, xh);
    ctx.set_name(cur, "pre_proj");

    let mut inp_l = cur;

    for (il, lw) in w.layers.iter().enumerate() {
        let is_swa = p.is_swa[il];
        let n_embd_head = p.n_embd_head[il];
        let n_head = p.n_head[il];
        let (freq_base_l, freq_scale_l) = if is_swa {
            (p.freq_base_swa, 1.0f32)
        } else {
            (p.freq_base, p.freq_scale)
        };
        let n_rot_l = p.n_rot[il] as i32;

        // cur_norm = rms(inpL) * attn_norm (:125-126)
        let cur_norm = build_norm_rms(ctx, inp_l, lw.attn_norm, p.norm_eps);
        ctx.set_name(cur_norm, &format!("attn_norm-{il}"));

        // Qcur = wq(cur_norm) → reshape → q_norm → rope (:128-136)
        let qcur = crate::adapter::lora_mm(ctx, lw.wq, cur_norm);
        let qcur = ctx.reshape_3d(qcur, n_embd_head, n_head, t);
        let qcur = build_norm_rms(ctx, qcur, lw.attn_q_norm, p.norm_eps);
        ctx.set_name(qcur, &format!("Qcur_normed-{il}"));
        let freq_factors = if is_swa { None } else { lw.rope_freqs };
        let qcur = ctx.rope_ext(
            qcur,
            pos,
            freq_factors,
            n_rot_l,
            p.rope_mode,
            p.n_ctx_orig,
            freq_base_l,
            freq_scale_l,
            p.ext_factor,
            p.attn_factor,
            p.beta_fast,
            p.beta_slow,
        );
        ctx.set_name(qcur, &format!("Qcur_pos-{il}"));

        // build_attn(inp_attn, wo, ..., Qcur, k = nullptr, v = nullptr, ...)
        // (:138-139) — the Q-only attention over the shared target KV
        // (llama-graph.cpp:3089-3162's k_cur/v_cur == nullptr arm: no cpy_k /
        // cpy_v scatter, k/v viewed from mctx_cur — the iswa pair behind the
        // share map)
        let il_tgt = if is_swa {
            p.n_layer_tgt - 2
        } else {
            p.n_layer_tgt - 1
        };
        let (head_dim, n_head_kv, n_kv_side) = if is_swa {
            (p.tgt_swa.0, p.tgt_swa.1, kv.n_kv_swa())
        } else {
            (p.tgt_full.0, p.tgt_full.1, kv.n_kv())
        };
        let k_view = kv.get_k(ctx, il_tgt, head_dim, n_head_kv, n_kv_side);
        let v_view = kv.get_v(ctx, il_tgt, head_dim, n_head_kv, n_kv_side);
        let mask = if is_swa { kq_mask_swa } else { kq_mask };

        // hparams.f_attention_scale == 1.0 (gemma4-assistant.cpp:14)
        let kq_scale = 1.0f32;
        let attn = if p.use_flash_attn {
            graph::flash_attn_core_sinks(
                ctx, qcur, k_view, v_view, mask, None, kq_scale, 0.0, 0.0,
            )
        } else {
            // build_attn_mha's non-FA branch (llama-graph.cpp:2670-2733)
            let q = ctx.permute(qcur, 0, 2, 1, 3);
            let k_view = ctx.permute(k_view, 0, 2, 1, 3);
            let kq = ctx.mul_mat(k_view, q);
            ctx.set_name(kq, &format!("kq-{il}"));
            let kq = ctx.soft_max_ext(kq, Some(mask), kq_scale, 0.0);
            ctx.set_name(kq, &format!("kq_soft_max-{il}"));
            let v_view = ctx.permute(v_view, 0, 2, 1, 3);
            let v_t = ctx.transpose(v_view);
            let v_c = ctx.cont(v_t);
            let kqv = ctx.mul_mat(v_c, kq);
            let kqv = ctx.permute(kqv, 0, 2, 1, 3);
            let kqv = ctx.cont(kqv);
            ctx.reshape_2d(kqv, head_dim * n_head, t)
        };
        ctx.set_name(attn, &format!("kqv_out-{il}"));
        let cur = crate::adapter::lora_mm(ctx, lw.wo, attn);
        ctx.set_name(cur, &format!("attn_proj-{il}"));

        // attn_post_norm → residual (:141-145)
        let cur = build_norm_rms(ctx, cur, lw.attn_post_norm, p.norm_eps);
        ctx.set_name(cur, &format!("attn_post_norm-{il}"));
        let attn_out = ctx.add(cur, inp_l);
        ctx.set_name(attn_out, &format!("attn_out-{il}"));

        // the GELU FFN + ffn_post_norm → residual → out_scale (:147-167)
        let cur = build_norm_rms(ctx, attn_out, lw.ffn_norm, p.norm_eps);
        ctx.set_name(cur, &format!("ffn_norm-{il}"));
        // the helper's (up, gate, down) order — build_ffn LLM_FFN_GELU +
        // LLM_FFN_PAR (:147-159)
        let cur =
            crate::graph_arch::build_ffn_gelu_par(ctx, cur, lw.ffn_up, lw.ffn_gate, lw.ffn_down);
        ctx.set_name(cur, &format!("ffn_out-{il}"));
        let cur = build_norm_rms(ctx, cur, lw.ffn_post_norm, p.norm_eps);
        ctx.set_name(cur, &format!("ffn_post_norm--1"));
        let cur = ctx.add(cur, attn_out);
        let cur = ctx.mul(cur, lw.out_scale); // :166-167
        ctx.set_name(cur, &format!("out_scaled-{il}"));

        inp_l = cur;
    }

    // result_norm → the tied lm head + h_next (:170-184)
    let cur = build_norm_rms(ctx, inp_l, w.output_norm, p.norm_eps);
    ctx.set_name(cur, "result_norm");
    let logits = crate::adapter::lora_mm(ctx, w.output, cur);
    ctx.set_name(logits, "result_output");
    let h_next = crate::adapter::lora_mm(ctx, w.nextn_proj_post, cur);
    ctx.set_name(h_next, "h_nextn");

    graph.build_forward(ctx, logits);
    graph.build_forward(ctx, h_next);
    ForwardResult {
        logits,
        graph,
        embd: Some(h_next),
    }
}

/// build_norm LLM_NORM_RMS — the same form graph_arch.rs uses
fn build_norm_rms(ctx: &mut Context, x: TensorId, w: TensorId, eps: f32) -> TensorId {
    let n = ctx.rms_norm(x, eps);
    ctx.mul(n, w)
}

/// the per-step draft result of
/// [`crate::context::DecodeContext::decode_gemma4_assistant`]:
/// `res->t_logits` rows + `res->t_h_nextn` rows (the next step's `inp_h`)
pub struct Gemma4AssistantStep {
    /// F32 `[n_vocab * n_tokens]` — one row per draft token
    pub logits: Vec<f32>,
    /// F32 `[n_embd_backbone * n_tokens]`
    pub h_next: Vec<f32>,
    pub n_tokens: usize,
    n_vocab: usize,
    n_embd_backbone: usize,
}

impl Gemma4AssistantStep {
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    /// `llama_get_embeddings_nextn_ith(ctx_dft, i)` (:1672) — one h row
    pub fn h_row(&self, i: usize) -> Vec<f32> {
        let b = self.n_embd_backbone;
        self.h_next[i * b..(i + 1) * b].to_vec()
    }
}

/// the head attached to a target context — the draft-side state
/// (`cparams.ctx_other == ctx_tgt` inverted: the head's tensors live in the
/// target's Context so the draft graph can view the shared KV cache)
pub struct Gemma4AssistantState {
    pub weights: Gemma4AssistantWeights,
    pub params: Gemma4AssistantParams,
}

impl Gemma4AssistantState {
    /// one draft decode over the shared KV — the graph execution of
    /// `llama_decode(ctx_dft, batch)` for the gemma4-assistant context: the
    /// batch's tokens + h rows, **no KV write** (the target's cache is read
    /// through the share map), the masks filled over the shared cells for
    /// this ubatch (`llm_graph_input_attn_kv_iswa::set_input`,
    /// llama-graph.cpp:612-651 → `set_input_kq_mask`). Runs on the target
    /// Context's scratch arena (`watermark`), one thread like the reference's
    /// single-token draft ubatches.
    pub fn decode(
        &self,
        gctx: &mut Context,
        kv: &KvCache,
        watermark: usize,
        tok_embd_tgt: TensorId,
        batch: &LlamaBatch,
    ) -> Result<Gemma4AssistantStep, String> {
        let n = batch.token.len();
        if n == 0 {
            return Err("gemma4-assistant decode: empty batch".into());
        }
        // `batch.pos` — None = auto (0..n); the driver always sets it
        let pos: Vec<i32> = batch
            .pos
            .clone()
            .unwrap_or_else(|| (0..n as i32).collect());
        let p = &self.params;
        let backbone = p.n_embd_backbone as usize;
        let embd = batch.embd.as_ref().ok_or(
            "gemma4-assistant decode: batch.embd must carry one n_embd_backbone row per token",
        )?;
        if embd.len() != n * backbone {
            return Err(format!(
                "gemma4-assistant decode: batch.embd carries {} values, expected {} (n_tokens \
                 {n} x n_embd_backbone {backbone})",
                embd.len(),
                n * backbone
            ));
        }

        gctx.reset_graph_to(watermark);
        let tokens_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let h_t = gctx.new_tensor_2d(GgmlType::F32, backbone as i64, n as i64);
        let mask_ty = if p.use_flash_attn {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        let n_kv = kv.n_kv();
        let n_kv_swa = kv.n_kv_swa();
        let kq_mask = gctx.new_tensor_2d(mask_ty, n_kv as i64, n as i64);
        let kq_mask_swa = gctx.new_tensor_2d(mask_ty, n_kv_swa as i64, n as i64);
        for t in [tokens_t, pos_t, h_t, kq_mask, kq_mask_swa] {
            gctx.arena_resize_tensor(t);
        }
        gctx.with_i32_mut(tokens_t, |q| q.copy_from_slice(&batch.token))
            .unwrap();
        gctx.with_i32_mut(pos_t, |q| q.copy_from_slice(&pos)).unwrap();
        gctx.with_f32_mut(h_t, |q| q.copy_from_slice(embd)).unwrap();

        // the kq masks over the shared cells (`set_input_kq_mask`,
        // llama-kv-cache.cpp:1557-1705 — causal + the SWA window per side;
        // the assistant context keeps the default causal_attn = true)
        let q_seq: Vec<i32> = batch
            .seq_id
            .as_ref()
            .map(|s| s.iter().map(|ids| ids[0]).collect())
            .unwrap_or_else(|| vec![0; n]);
        {
            let base_cells: Vec<(i32, u64)> = kv.cells[..n_kv as usize]
                .iter()
                .map(|c| (c.pos, c.seq))
                .collect();
            crate::context::fill_mask_seq_pub(
                gctx,
                kq_mask,
                mask_ty,
                &base_cells,
                &q_seq,
                &pos,
                0,
                crate::hparams::LlamaSwaType::NONE,
                false,
                true,
            );
            let swa = kv
                .swa_cache()
                .ok_or("gemma4-assistant decode: the target has no SWA cache pair")?;
            let swa_cells: Vec<(i32, u64)> = swa.cells[..n_kv_swa as usize]
                .iter()
                .map(|c| (c.pos, c.seq))
                .collect();
            crate::context::fill_mask_seq_pub(
                gctx,
                kq_mask_swa,
                mask_ty,
                &swa_cells,
                &q_seq,
                &pos,
                swa.n_swa,
                swa.swa_type,
                false,
                true,
            );
        }

        let result = build_gemma4_assistant_forward(
            gctx,
            &self.weights,
            p,
            kv,
            tok_embd_tgt,
            tokens_t,
            h_t,
            pos_t,
            kq_mask,
            kq_mask_swa,
            n,
        );
        let mut gf = result.graph;
        ggml::compute::graph_compute(gctx, &mut gf, 1);

        let logits: Vec<f32> =
            bytemuck::cast_slice(gctx.data_bytes(result.logits).unwrap()).to_vec();
        let h_next: Vec<f32> =
            bytemuck::cast_slice(gctx.data_bytes(result.embd.unwrap()).unwrap()).to_vec();
        Ok(Gemma4AssistantStep {
            logits,
            h_next,
            n_tokens: n,
            n_vocab: p.n_vocab as usize,
            n_embd_backbone: backbone,
        })
    }
}
