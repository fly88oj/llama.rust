//! eagle — the EAGLE3 head model loader + draft-context assembly, the port of
//! the reference's eagle3 arch support outside its graphs (the graphs live in
//! [`crate::graph_arch`]'s eagle section, the driver in
//! [`crate::speculative`]'s draft-eagle3 impl):
//!
//!   * `llama_model_eagle3::load_arch_hparams` (src/models/eagle3.cpp:3-36) —
//!     the eagle3.* GGUF keys: `target_layers` (exactly 3 — the extract
//!     layers), `target_hidden_size`, `norm_before_residual` / `norm_before_fc`
//!     (optional, default false);
//!   * `llama_model_eagle3::load_arch_tensors` (eagle3.cpp:38-101) — the head
//!     tensor set: fc / enc.output_norm / output_norm / output (optional) /
//!     token_embd (optional) / d2t (optional) + exactly one decoder layer;
//!   * `cparams.ctx_other` (llama-context.cpp:156-163) — when the head carries
//!     no token_embd / output, the C graph reads the *target* model's tensors
//!     through the ctx_other pointer (eagle3.cpp:164-171 / :296-304). The
//!     port's ggml graphs live in one [`ggml::Context`], so the loader
//!     materializes those two tensors in the head's context backed by the
//!     **target file's mmap** (`set_external_storage`) — the same bytes, zero
//!     copy, any ggml type.
//!
//! The generic loader of model.rs does not know the eagle3 arch
//! (`arch_tensors_support` has no arm — model.rs is owned by other port
//! batches), so this module owns the head's hparams + tensors, mirroring the
//! ModelLoader recipe (model.rs:763-930).

use std::sync::Arc;

use ggml::{Context, Gguf, TensorId};

use crate::graph;
use crate::graph_arch::{Eagle3LayerWeights, Eagle3Params, Eagle3Weights};

/// the eagle3 head's loaded state: hparams (`load_arch_hparams`) + every
/// tensor created in the head's own ggml Context (`load_arch_tensors`).
pub struct Eagle3Head {
    pub ctx: Context,
    pub weights: Eagle3Weights,
    pub params: Eagle3Params,
    /// `target_layer_ids` (eagle3.cpp:6-15) — exactly 3 extract layers
    pub target_layer_ids: [i32; 3],
    /// `{arch}.target_hidden_size` (:19) — the target model's n_embd
    pub n_embd_tgt: i64,
    /// n_vocab used to size logits (the target vocab; with d2t the scatter
    /// emits target-vocab rows)
    pub n_vocab: i64,
}

/// one created tensor + its gguf name (the create_tensor mirror: exact shape
/// check, external mmap storage)
struct HeadLoader<'a> {
    gguf: &'a Gguf,
    mmap: Arc<memmap2::Mmap>,
    ctx: Context,
    by_name: std::collections::HashMap<String, TensorId>,
    n_created: usize,
}

impl<'a> HeadLoader<'a> {
    fn new(gguf: &'a Gguf, mmap: Arc<memmap2::Mmap>) -> Self {
        Self {
            gguf,
            mmap,
            ctx: Context::new(),
            by_name: std::collections::HashMap::new(),
            n_created: 0,
        }
    }

    /// `llama_model_loader::create_tensor` (model.rs:836-897) restricted to
    /// what the eagle3 head needs: exact dims, required-or-optional.
    fn create_tensor(
        &mut self,
        name: &str,
        ne: &[i64],
        required: bool,
    ) -> Result<Option<TensorId>, String> {
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

    /// the ctx_other tensor: same bytes, the target file's mmap — the port's
    /// equivalent of the C graph reading `model_other->tok_embd` /
    /// `->output` through the pointer
    fn create_tensor_in(
        &mut self,
        other_gguf: &Gguf,
        other_mmap: &Arc<memmap2::Mmap>,
        name: &str,
        ne: &[i64],
    ) -> Result<TensorId, String> {
        let Some(ti) = other_gguf.find_tensor(name) else {
            return Err(format!(
                "EAGLE3 decoder requires '{name}' (own or from target model) — not in either file"
            ));
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
        self.ctx.set_name(id, &format!("other.{name}"));
        let data_off = other_gguf.data_offset as usize + ti.offset as usize;
        self.ctx
            .set_external_storage(id, other_mmap.clone(), data_off);
        self.by_name.insert(format!("other.{name}"), id);
        Ok(id)
    }
}

/// `load_eagle3_head` — the eagle3 hparams + tensors (eagle3.cpp:3-101).
///
/// `target_gguf` / `target_mmap` are the *target* model's file — needed only
/// when the head has no own `token_embd` / `output` (the ctx_other tensors).
/// `n_vocab` is the shared vocabulary size (the head file's tokenizer;
/// identical to the target's — `common_speculative_are_compatible`'s
/// requirement).
pub fn load_eagle3_head(
    gguf: &Gguf,
    mmap: Arc<memmap2::Mmap>,
    target_gguf: &Gguf,
    target_mmap: Arc<memmap2::Mmap>,
    n_vocab: i64,
    fa: bool,
) -> Result<Eagle3Head, String> {
    // ---- load_arch_hparams (eagle3.cpp:3-36) ----
    let eps = gguf
        .get_f32("eagle3.attention.layer_norm_rms_epsilon")
        .ok_or("key eagle3.attention.layer_norm_rms_epsilon not found in model file")?;

    // ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false) — required
    let target_layer_ids: [i32; 3] = {
        let arr = gguf
            .find_key("eagle3.target_layers")
            .and_then(|v| v.as_array())
            .ok_or("EAGLE3 model requires 'extract_layers' in GGUF metadata")?;
        let ids: Result<Vec<i32>, ()> = arr
            .1
            .iter()
            .map(|v| {
                v.as_u32()
                    .map(|u| u as i32)
                    .or_else(|| v.as_f32().map(|f| f as i32))
                    .ok_or(())
            })
            .collect();
        let ids = ids.map_err(|()| "EAGLE3: malformed eagle3.target_layers array".to_string())?;
        if ids.len() != 3 {
            return Err(format!(
                "EAGLE3 requires exactly 3 entries in 'extract_layers' (got {})",
                ids.len()
            ));
        }
        [ids[0], ids[1], ids[2]]
    };

    let n_embd_tgt =
        gguf.get_u32("eagle3.target_hidden_size")
            .ok_or("key eagle3.target_hidden_size not found in model file")? as i64;

    // optional flags (default false — :26/:33)
    let norm_before_residual = gguf
        .get_bool("eagle3.norm_before_residual")
        .unwrap_or(false);
    let norm_before_fc = gguf.get_bool("eagle3.norm_before_fc").unwrap_or(false);

    // ---- the generic dims the head's graph needs (llama_model_base::
    // load_hparams reads these the same way) ----
    let n_embd = gguf
        .get_u32("eagle3.embedding_length")
        .ok_or("key eagle3.embedding_length not found in model file")? as i64;
    let n_layer = gguf
        .get_u32("eagle3.block_count")
        .ok_or("key eagle3.block_count not found in model file")? as usize;
    let n_head = gguf.get_u32("eagle3.attention.head_count").unwrap_or(0) as i64;
    let n_head_kv = gguf
        .get_u32("eagle3.attention.head_count_kv")
        .unwrap_or(n_head as u32) as i64;
    // key_length/value_length default to n_embd / n_head (llama-hparams.cpp
    // n_embd_head_k_full fallback)
    let n_embd_head_k = gguf
        .get_u32("eagle3.attention.key_length")
        .unwrap_or((n_embd / n_head.max(1)) as u32) as i64;
    let n_embd_head_v = gguf
        .get_u32("eagle3.attention.value_length")
        .unwrap_or((n_embd / n_head.max(1)) as u32) as i64;
    let n_rot = gguf
        .get_u32("eagle3.rope.dimension_count")
        .ok_or("key eagle3.rope.dimension_count not found in model file")? as i64;
    let freq_base = gguf.get_f32("eagle3.rope.freq_base").unwrap_or(10000.0);
    let freq_scale = gguf.get_f32("eagle3.rope.freq_scale").unwrap_or(1.0);
    let n_ctx_train = gguf.get_u32("eagle3.context_length").unwrap_or(0) as i32;

    if n_head <= 0 || n_embd_head_k <= 0 || n_embd_head_v <= 0 {
        return Err("eagle3: attention.head_count missing or zero".into());
    }

    let n_embd_inp_enc = 3 * n_embd_tgt; // hparams.n_embd_inp_enc_impl (:22)
    let n_embd_attn_input = 2 * n_embd; // :42

    // ---- load_arch_tensors (eagle3.cpp:38-101) ----
    let mut ld = HeadLoader::new(gguf, mmap);

    // d2t: draft to target vocabulary mapping (optional, :44-55)
    let n_draft_vocab = gguf
        .find_tensor("d2t")
        .map(|ti| ti.ne[0])
        .unwrap_or(n_vocab);
    let d2t = ld.create_tensor("d2t", &[n_draft_vocab], false)?;
    if d2t.is_some() {
        eprintln!("eagle3: EAGLE3 using d2t mapping (draft_vocab_size = {n_draft_vocab})");
    }

    // Feature fusion layer: projects 3 target layers to draft hidden size (:58)
    let fc = ld
        .create_tensor("fc.weight", &[n_embd_inp_enc, n_embd], true)?
        .expect("fc.weight");

    // RMSNorm on the fused target features, only when norm_before_fc (:61-63)
    let output_norm_enc = if norm_before_fc {
        Some(
            ld.create_tensor("enc.output_norm.weight", &[n_embd_inp_enc], true)?
                .expect("enc.output_norm"),
        )
    } else {
        None
    };

    // Output layer (uses draft vocab size, :66-67)
    let output_norm = ld
        .create_tensor("output_norm.weight", &[n_embd], true)?
        .expect("output_norm");
    let output = ld.create_tensor("output.weight", &[n_embd, n_draft_vocab], false)?;

    // Token embeddings (optional — Llama 3.3 70B EAGLE3 has its own, :69-75)
    let tok_embd = ld.create_tensor("token_embd.weight", &[n_embd, n_vocab], false)?;

    // Single decoder layer (:77-100)
    if n_layer != 1 {
        // GGML_ASSERT(n_layer == 1) of the decoder graph (eagle3.cpp:156)
        return Err(format!(
            "eagle3 has only one decoder layer (block_count = {n_layer})"
        ));
    }
    let n_embd_k_gqa = n_embd_head_k * n_head_kv;
    let n_embd_v_gqa = n_embd_head_v * n_head_kv;
    let layer = Eagle3LayerWeights {
        attn_norm: ld
            .create_tensor("blk.0.attn_norm.weight", &[n_embd], true)?
            .expect("attn_norm"),
        attn_norm_2: ld
            .create_tensor("blk.0.attn_norm_2.weight", &[n_embd], true)?
            .expect("attn_norm_2"),
        wq: ld
            .create_tensor(
                "blk.0.attn_q.weight",
                &[n_embd_attn_input, n_embd_head_k * n_head],
                true,
            )?
            .expect("wq"),
        wk: ld
            .create_tensor(
                "blk.0.attn_k.weight",
                &[n_embd_attn_input, n_embd_k_gqa],
                true,
            )?
            .expect("wk"),
        wv: ld
            .create_tensor(
                "blk.0.attn_v.weight",
                &[n_embd_attn_input, n_embd_v_gqa],
                true,
            )?
            .expect("wv"),
        wo: ld
            .create_tensor(
                "blk.0.attn_output.weight",
                &[n_embd_head_k * n_head, n_embd],
                true,
            )?
            .expect("wo"),
        ffn_norm: ld
            .create_tensor("blk.0.ffn_norm.weight", &[n_embd], true)?
            .expect("ffn_norm"),
        ffn_gate: ld
            .create_tensor("blk.0.ffn_gate.weight", &[n_embd, ffn_dim(gguf)], true)?
            .expect("ffn_gate"),
        ffn_down: ld
            .create_tensor("blk.0.ffn_down.weight", &[ffn_dim(gguf), n_embd], true)?
            .expect("ffn_down"),
        ffn_up: ld
            .create_tensor("blk.0.ffn_up.weight", &[n_embd, ffn_dim(gguf)], true)?
            .expect("ffn_up"),
        rope_freqs: ld.create_tensor("blk.0.rope_freqs.weight", &[n_rot / 2], false)?,
    };

    // done_getting_tensors (model.rs:1089-1096): every file tensor consumed
    let n_file_tensors = gguf.tensors.len();
    if ld.n_created != n_file_tensors {
        let mut missing: Vec<&str> = gguf
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| !ld.by_name.contains_key(*n) && !n.starts_with("other."))
            .collect();
        missing.sort_unstable();
        return Err(format!(
            "eagle3 head has {n_file_tensors} tensors but the loader created {} (unknown: {missing:?})",
            ld.n_created
        ));
    }

    // ---- the ctx_other tensors (llama-context.cpp:156-163 + eagle3.cpp:
    // 164-171/:296-304): own or the target model's ----
    let tok_embd = match tok_embd {
        Some(t) => t,
        None => ld.create_tensor_in(
            target_gguf,
            &target_mmap,
            "token_embd.weight",
            &[n_embd_tgt, n_vocab],
        )?,
    };
    let output = match output {
        Some(t) => t,
        None => ld.create_tensor_in(
            target_gguf,
            &target_mmap,
            "output.weight",
            &[n_embd_tgt, n_vocab],
        )?,
    };

    let attn = graph::AttnParams {
        n_head,
        n_head_kv,
        n_embd_head_k,
        n_embd_head_v,
        n_rot,
        // LLAMA_ROPE_TYPE_NORM (llama-model.cpp:2965)
        rope_mode: crate::hparams::LlamaRopeType::NORM as i32,
        n_ctx_orig: n_ctx_train,
        freq_base,
        freq_scale,
        ext_factor: -1.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        norm_eps: eps,
        use_flash_attn: fa,
    };

    let weights = Eagle3Weights {
        tok_embd,
        output,
        output_norm,
        output_norm_enc,
        fc,
        d2t,
        layers: vec![layer],
    };
    let params = Eagle3Params {
        attn,
        n_embd,
        n_embd_inp_enc,
        norm_before_residual,
        norm_before_fc,
        n_vocab,
        target_layer_ids,
        n_embd_tgt,
    };

    Ok(Eagle3Head {
        ctx: ld.ctx,
        weights,
        params,
        target_layer_ids,
        n_embd_tgt,
        n_vocab,
    })
}

/// `{arch}.feed_forward_length` — n_ff of the decoder layer's FFN
fn ffn_dim(gguf: &Gguf) -> i64 {
    gguf.get_u32("eagle3.feed_forward_length").unwrap_or(0) as i64
}

/// the 1-layer "trunk" bundle a DecodeContext needs for its generic sizing
/// (n_layer / output / n_pos_per_embd) — the head has no trunk graph, so the
/// eagle branch of `DecodeContext::forward` returns before this is ever
/// matched; the tensor ids are the head's own (valid ids of its context).
pub fn eagle_trunk_stub(w: &Eagle3Weights) -> graph::ModelWeights {
    let l = &w.layers[0];
    graph::ModelWeights {
        tok_embd: w.tok_embd,
        output_norm: w.output_norm,
        output: w.output,
        layers: vec![graph::LayerWeights {
            attn_norm: l.attn_norm,
            wq: l.wq,
            wk: l.wk,
            wv: l.wv,
            wo: l.wo,
            wq_b: None,
            wk_b: None,
            wv_b: None,
            ffn_norm: l.ffn_norm,
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
        }],
    }
}
