//! clef — 对照 src/models/clef.cpp (a7b94df2c, upstream 99b95488c "model:
//! add support for clef decision model (text-only)")
//!
//! A Qwen3.5 backbone evaluated **without memory** (create_memory returns
//! nullptr for LLM_ARCH_CLEF, llama-model.cpp:2375-2381) with a joint
//! decision head on top of its hidden states: the head pools the trunk's
//! hidden states over question/option spans marked by the batch's
//! `decision_order` (llama_batch_ext_set_decision_order) and scores each
//! option. The whole model runs as one ubatch — the attention mask is
//! no-cache causal (`build_attn_inp_causal`, clef.cpp:393-407) and the GDN
//! layers start from zero states they never write back
//! (`build_layer_attn_linear`, clef.cpp:478-545).
//!
//! The output is `res->t_embd` only — [1, n_options] padded to n_tokens
//! rows plus a NaN status row on batches whose decision order cannot be
//! used (clef.cpp:371-377); there is no LM head, `t_logits` stays null.

use ggml::{Context, GgmlType, TensorId};

use crate::batch::{
    DECISION_ORDER_NONE, DECISION_ORDER_OPTION, DECISION_ORDER_QUESTION_CHOICE,
    DECISION_ORDER_QUESTION_NOUL, DECISION_ORDER_QUESTION_SCORE,
};
use crate::graph_arch::{attn_no_cache, build_gdn_l2_norm, Qwen35LayerWeights, Qwen35ModelWeights, Qwen35Params};

/// the question-type discriminants the spans reader switches on
const QUESTION_NOUL: i32 = DECISION_ORDER_QUESTION_NOUL;

/// a norm with a bias — the head's blocks are plain pre-norm LayerNorms
/// (clef.cpp:51-55)
#[derive(Debug, Clone, Copy)]
pub struct ClefNormTensors {
    pub w: TensorId,
    pub b: TensorId,
}

/// a head attention's projections with biases (clef.cpp:43-50)
#[derive(Debug, Clone, Copy)]
pub struct ClefAttnTensors {
    pub wq: TensorId,
    pub bq: TensorId,
    pub wk: TensorId,
    pub bk: TensorId,
    pub wv: TensorId,
    pub bv: TensorId,
    pub wo: TensorId,
    pub bo: TensorId,
}

/// one head block: routing blocks (no self attention) then joint blocks
/// (clef.cpp:57-69)
#[derive(Debug, Clone, Copy)]
pub struct ClefHeadLayerTensors {
    /// joint blocks only
    pub self_norm: Option<ClefNormTensors>,
    /// joint blocks only
    pub self_attn: Option<ClefAttnTensors>,
    pub cross_norm: ClefNormTensors,
    /// routing blocks only
    pub cross_norm_kv: Option<ClefNormTensors>,
    pub cross_attn: ClefAttnTensors,
    pub ffn_norm: ClefNormTensors,
    pub ffn_up: TensorId,
    pub ffn_up_b: TensorId,
    pub ffn_down: TensorId,
    pub ffn_down_b: TensorId,
}

/// the model-level head tensors (clef.cpp:96-123)
#[derive(Debug, Clone)]
pub struct ClefHeadTensors {
    /// routing blocks, then joint blocks
    pub layers: Vec<ClefHeadLayerTensors>,
    pub hidden_norm: ClefNormTensors,
    pub option_summary_norm: ClefNormTensors,
    pub field_norm: ClefNormTensors,
    pub option_norm: ClefNormTensors,
    pub proj_memory: TensorId,
    pub proj_question: TensorId,
    pub proj_option_question: TensorId,
    pub proj_global: TensorId,
    pub proj_option_context: TensorId,
    pub proj_option_lexical: TensorId,
    /// prior scale, joint scale, residual gate
    pub scales: TensorId,
    pub type_embd: TensorId,
    pub scorer: TensorId,
    pub scorer_b: TensorId,
    pub scorer_out: TensorId,
    pub scorer_out_b: TensorId,
}

/// `llama_model_clef`'s model-level members (clef.cpp:71-72) — the graph
/// geometry the hparams do not carry
#[derive(Debug, Clone)]
pub struct ClefHeadParams {
    /// `n_layer_routing`
    pub n_layer_routing: usize,
    /// `n_head_decision`
    pub n_head_decision: i64,
}

/// spans read by the head, [start, end) in ubatch token indices
/// (clef.cpp:132-145)
#[derive(Debug, Clone, PartialEq)]
pub struct ClefSpans {
    pub questions: Vec<(i32, i32, i32)>, // (type, start, end)
    pub options: Vec<(i32, i32, i32)>,   // (question index, start, end)
    pub valid: bool,
}

/// `clef_get_spans` (clef.cpp:150-180) — if the batch has no usable order,
/// returns one empty question with one empty option.
pub fn clef_get_spans(decision_order: Option<&[i32]>, n_tokens: usize) -> ClefSpans {
    let mut questions: Vec<(i32, i32, i32)> = Vec::new();
    let mut options: Vec<(i32, i32, i32)> = Vec::new();
    let mut has_option: Vec<bool> = Vec::new();

    // TODO(upstream): support multiple sequences
    let order = decision_order.unwrap_or(&[]);
    let mut ok = !order.is_empty();

    let mut i = 0i32;
    while ok && (i as usize) < n_tokens {
        let o = order[i as usize];
        let mut end = i + 1;
        while (end as usize) < n_tokens && order[end as usize] == o {
            end += 1;
        }
        match o {
            DECISION_ORDER_NONE => {}
            QUESTION_NOUL | DECISION_ORDER_QUESTION_CHOICE | DECISION_ORDER_QUESTION_SCORE => {
                questions.push((o - QUESTION_NOUL, i, end));
                has_option.push(false);
            }
            DECISION_ORDER_OPTION => {
                ok = !questions.is_empty();
                if ok {
                    options.push((questions.len() as i32 - 1, i, end));
                    *has_option.last_mut().unwrap() = true;
                }
            }
            _ => ok = false,
        }
        i = end;
    }

    // each question needs an option
    let valid = ok && !questions.is_empty() && !has_option.iter().any(|&h| !h);
    if !valid {
        questions.clear();
        options.clear();
        questions.push((0, 0, 0));
        options.push((0, 0, 0));
    }
    ClefSpans {
        questions,
        options,
        valid,
    }
}

/// `llama_model_clef::input_decision` (clef.cpp:183-241) — the pooling
/// matrices and indices computed from the spans; `set_input` is
/// [`ClefDecisionInputs::set_input`].
pub struct ClefDecisionInputs {
    /// I32 [n_tokens]
    pub tokens: TensorId,
    /// F32 [n_tokens, n_questions] — mean over the span of the question
    pub pool_q: TensorId,
    /// F32 [n_tokens, n_options] — mean over the span of the option
    pub pool_o: TensorId,
    /// I32 [n_questions]
    pub question_type: TensorId,
    /// I32 [n_options]
    pub option_question: TensorId,
    /// F32 [n_options, n_questions] — 0 if the option belongs to the
    /// question, else -inf
    pub option_mask: TensorId,
    /// F32 [1] — added to the scores: 0, or NaN on invalid input
    pub status: TensorId,
    pub n_questions: usize,
    pub n_options: usize,
}

impl ClefDecisionInputs {
    /// the tensor set of `build_head`'s input creation (clef.cpp:537-548)
    pub fn build(gctx: &mut Context, n_tokens: usize, order: Option<&[i32]>) -> Self {
        let spans = clef_get_spans(order, n_tokens);
        let n_q = spans.questions.len();
        let n_o = spans.options.len();
        let t = n_tokens as i64;
        let tokens = gctx.new_tensor_1d(GgmlType::I32, t);
        let pool_q = gctx.new_tensor_2d(GgmlType::F32, t, n_q as i64);
        let pool_o = gctx.new_tensor_2d(GgmlType::F32, t, n_o as i64);
        let question_type = gctx.new_tensor_1d(GgmlType::I32, n_q as i64);
        let option_question = gctx.new_tensor_1d(GgmlType::I32, n_o as i64);
        let option_mask = gctx.new_tensor_2d(GgmlType::F32, n_o as i64, n_q as i64);
        let status = gctx.new_tensor_1d(GgmlType::F32, 1);
        for ti in [
            tokens,
            pool_q,
            pool_o,
            question_type,
            option_question,
            option_mask,
            status,
        ] {
            gctx.arena_resize_tensor(ti);
        }
        Self {
            tokens,
            pool_q,
            pool_o,
            question_type,
            option_question,
            option_mask,
            status,
            n_questions: n_q,
            n_options: n_o,
        }
    }

    /// `input_decision::set_input` (clef.cpp:193-227)
    pub fn set_input(
        &self,
        gctx: &mut Context,
        tokens: &[i32],
        decision_order: Option<&[i32]>,
    ) {
        let n_tokens = tokens.len();
        let spans = clef_get_spans(decision_order, n_tokens);
        assert!(
            spans.questions.len() == self.n_questions
                && spans.options.len() == self.n_options,
            "clef: span shape changed (can_reuse would have rebuilt the graph)"
        );

        gctx.with_i32_mut(self.tokens, |p| p.copy_from_slice(tokens)).unwrap();

        // the scores are NaN if the batch has a decision order that cannot
        // be used
        let status_data =
            if spans.valid || decision_order.is_none() { 0.0f32 } else { f32::NAN };
        gctx.with_f32_mut(self.status, |p| p[0] = status_data).unwrap();

        let n_q = self.n_questions;
        let n_o = self.n_options;

        // pool_q [n_tokens, n_questions] laid out column-major like the C's
        // row-major [n_tokens * n_questions] buffer filled at
        // `pool_q_data[i * n_tokens + j]` — i.e. F32 [n_tokens, n_questions]
        // with element (j, i) at i*n_tokens + j
        let mut pool_q_data = vec![0.0f32; n_tokens * n_q];
        let mut types = vec![0i32; n_q];
        for (i, &(ty, start, end)) in spans.questions.iter().enumerate() {
            for j in start..end {
                pool_q_data[i * n_tokens + j as usize] = 1.0 / (end - start) as f32;
            }
            types[i] = ty;
        }

        let mut pool_o_data = vec![0.0f32; n_tokens * n_o];
        let mut owner = vec![0i32; n_o];
        let mut mask = vec![f32::NEG_INFINITY; n_o * n_q];
        for (i, &(q, start, end)) in spans.options.iter().enumerate() {
            for j in start..end {
                pool_o_data[i * n_tokens + j as usize] = 1.0 / (end - start) as f32;
            }
            owner[i] = q;
            mask[q as usize * n_o + i] = 0.0;
        }

        gctx.with_f32_mut(self.pool_q, |p| p.copy_from_slice(&pool_q_data)).unwrap();
        gctx.with_f32_mut(self.pool_o, |p| p.copy_from_slice(&pool_o_data)).unwrap();
        gctx.with_i32_mut(self.question_type, |p| p.copy_from_slice(&types)).unwrap();
        gctx.with_i32_mut(self.option_question, |p| p.copy_from_slice(&owner)).unwrap();
        gctx.with_f32_mut(self.option_mask, |p| p.copy_from_slice(&mask)).unwrap();
    }
}

/// `build_head_norm` (clef.cpp:547-549)
fn build_head_norm(
    ctx: &mut Context,
    x: TensorId,
    n: &ClefNormTensors,
    eps: f32,
) -> TensorId {
    let mut cur = ctx.norm(x, eps);
    cur = ctx.mul(cur, n.w);
    ctx.add(cur, n.b)
}

/// `build_head_attn` (clef.cpp:556-576) — q: [n_embd_h, n_q],
/// kv: [n_embd_h, n_kv], no mask, scale 1/sqrt(d_head).
fn build_head_attn(
    ctx: &mut Context,
    q: TensorId,
    kv: TensorId,
    a: &ClefAttnTensors,
    n_head_h: i64,
) -> TensorId {
    let n_embd_h = ctx.ne(q)[0];
    let d_head = n_embd_h / n_head_h;
    let n_q = ctx.ne(q)[1];
    let n_kv = ctx.ne(kv)[1];

    let mut qcur = ctx.mul_mat(a.wq, q);
    qcur = ctx.add(qcur, a.bq);
    let mut kcur = ctx.mul_mat(a.wk, kv);
    kcur = ctx.add(kcur, a.bk);
    let mut vcur = ctx.mul_mat(a.wv, kv);
    vcur = ctx.add(vcur, a.bv);

    // [d_head, n_q, n_head]
    let qcur = ctx.reshape_3d(qcur, d_head, n_head_h, n_q);
    let qcur = ctx.permute(qcur, 0, 2, 1, 3);
    // [d_head, n_kv, n_head]
    let kcur = ctx.reshape_3d(kcur, d_head, n_head_h, n_kv);
    let kcur = ctx.permute(kcur, 0, 2, 1, 3);
    // [n_kv, d_head, n_head]
    let vcur = ctx.reshape_3d(vcur, d_head, n_head_h, n_kv);
    let vcur = ctx.permute(vcur, 1, 2, 0, 3);
    let vcur = ctx.cont(vcur);

    let mut kq = ctx.mul_mat(kcur, qcur); // [n_kv, n_q, n_head]
    kq = ctx.soft_max_ext(kq, None, 1.0 / (d_head as f32).sqrt(), 0.0);

    let mut cur = ctx.mul_mat(vcur, kq); // [d_head, n_q, n_head]
    cur = ctx.permute(cur, 0, 2, 1, 3);
    cur = ctx.cont_2d(cur, n_embd_h, n_q);

    let mut out = ctx.mul_mat(a.wo, cur);
    out = ctx.add(out, a.bo);
    out
}

/// `build_head_ffn` (clef.cpp:578-583) — GELU(erf) FFN with biases
fn build_head_ffn(
    ctx: &mut Context,
    mut cur: TensorId,
    layer: &ClefHeadLayerTensors,
    eps: f32,
) -> TensorId {
    cur = build_head_norm(ctx, cur, &layer.ffn_norm, eps);
    cur = ctx.mul_mat(layer.ffn_up, cur);
    cur = ctx.add(cur, layer.ffn_up_b);
    cur = ctx.gelu_erf(cur);
    cur = ctx.mul_mat(layer.ffn_down, cur);
    ctx.add(cur, layer.ffn_down_b)
}

/// `build_head` (clef.cpp:585-616) — hidden: [n_embd, n_tokens], returns
/// one score per option: [1, n_options]. Ref: JointSchemaHead in
/// joint_schema_model.py of the model repo.
#[allow(clippy::too_many_arguments)]
fn build_head(
    ctx: &mut Context,
    hidden0: TensorId,
    head: &ClefHeadTensors,
    hp: &ClefHeadParams,
    w_output: TensorId,
    w_type_embd: TensorId,
    dec: &ClefDecisionInputs,
    n_tokens: usize,
    eps: f32,
) -> TensorId {
    let n_embd = ctx.ne(hidden0)[0];
    let t = n_tokens as i64;

    let hidden = build_head_norm(ctx, hidden0, &head.hidden_norm, eps);

    let memory = ctx.mul_mat(head.proj_memory, hidden); // [n_embd_h, n_tokens]
    let n_embd_h = ctx.ne(memory)[0];

    // hidden state of the last token
    let nb1 = ctx.nb(hidden)[1] as usize;
    let global = ctx.view_2d(hidden, n_embd, 1, nb1, (t as usize - 1) * nb1);

    // mean of the hidden states over each span
    let hidden_tr = ctx.transpose(hidden);
    let hidden_t = ctx.cont(hidden_tr);
    let question = ctx.mul_mat(hidden_t, dec.pool_q); // [n_embd, n_questions]
    let opt_ctx = ctx.mul_mat(hidden_t, dec.pool_o); // [n_embd, n_options]

    // mean of the output embeddings of the tokens of each option
    let mut lexical = ctx.get_rows(w_output, dec.tokens);
    let lexical_tr = ctx.transpose(lexical);
    let lexical_t = ctx.cont(lexical_tr);
    lexical = ctx.mul_mat(lexical_t, dec.pool_o); // [n_embd, n_options]

    // one query per option
    let mut options = ctx.mul_mat(head.proj_option_context, opt_ctx);
    let lex = ctx.mul_mat(head.proj_option_lexical, lexical);
    options = ctx.add(options, lex);
    let oq = ctx.mul_mat(head.proj_option_question, question);
    let oq = ctx.get_rows(oq, dec.option_question);
    options = ctx.add(options, oq);

    // the options read the prompt (clef.cpp:608-615)
    for (il, layer) in head.layers.iter().enumerate().take(hp.n_layer_routing) {
        let q = build_head_norm(ctx, options, &layer.cross_norm, eps);
        let kv = match layer.cross_norm_kv {
            Some(n_kv) => build_head_norm(ctx, memory, &n_kv, eps),
            None => memory,
        };
        let cur = build_head_attn(ctx, q, kv, &layer.cross_attn, hp.n_head_decision);
        options = ctx.add(options, cur);
        let ffn = build_head_ffn(ctx, options, layer, eps);
        options = ctx.add(options, ffn);
    }

    // one vector per question: its text, a summary of its options, the end
    // of the prompt and its type (clef.cpp:618-630)
    let mut fields = ctx.mul_mat(head.proj_question, question); // [n_embd_h, n_questions]
    {
        // each question weights its own options
        let mut weights = ctx.mul_mat(options, fields); // [n_options, n_questions]
        weights = ctx.scale(weights, 1.0 / (n_embd_h as f32).sqrt());
        weights = ctx.add(weights, dec.option_mask);
        weights = ctx.soft_max(weights);

        let options_tr = ctx.transpose(options);
        let options_t = ctx.cont(options_tr);
        let summary = ctx.mul_mat(options_t, weights);

        let sum = build_head_norm(ctx, summary, &head.option_summary_norm, eps);
        fields = ctx.add(fields, sum);
        let glob = ctx.mul_mat(head.proj_global, global);
        fields = ctx.add(fields, glob);
        let ty = ctx.get_rows(w_type_embd, dec.question_type);
        fields = ctx.add(fields, ty);
    }

    // the questions read each other and the prompt (clef.cpp:633-642)
    for layer in head.layers.iter().skip(hp.n_layer_routing) {
        let cur = build_head_norm(ctx, fields, layer.self_norm.as_ref().unwrap(), eps);
        let self_cur =
            build_head_attn(ctx, cur, cur, layer.self_attn.as_ref().unwrap(), hp.n_head_decision);
        fields = ctx.add(fields, self_cur);

        let cur = build_head_norm(ctx, fields, &layer.cross_norm, eps);
        let cross_cur = build_head_attn(ctx, cur, memory, &layer.cross_attn, hp.n_head_decision);
        fields = ctx.add(fields, cross_cur);

        let ffn = build_head_ffn(ctx, fields, layer, eps);
        fields = ctx.add(fields, ffn);
    }
    let fields = build_head_norm(ctx, fields, &head.field_norm, eps);

    let eps_l2 = 1e-12f32;

    // prior: the output embeddings of the option against the question and
    // the end of the prompt
    let qg = ctx.add(question, global);
    let mut anchor = ctx.l2_norm(qg, eps_l2);
    anchor = ctx.get_rows(anchor, dec.option_question);
    let anchor_n = ctx.l2_norm(anchor, eps_l2);
    let lexical_n = ctx.l2_norm(lexical, eps_l2);
    let prior_prod = ctx.mul(lexical_n, anchor_n);
    let prior = ctx.sum_rows(prior_prod); // [1, n_options]

    // joint: each option against the vector of its question
    let options = build_head_norm(ctx, options, &head.option_norm, eps);
    let field = ctx.get_rows(fields, dec.option_question); // [n_embd_h, n_options]

    let field_n = ctx.l2_norm(field, eps_l2);
    let options_n = ctx.l2_norm(options, eps_l2);
    let cos_prod = ctx.mul(field_n, options_n);
    let cosine = ctx.sum_rows(cos_prod);

    let mut features = ctx.concat(field, options, 0);
    let prod = ctx.mul(field, options);
    features = ctx.concat(features, prod, 0);
    let fd = ctx.sub(field, options);
    let diff = ctx.abs(fd);
    features = ctx.concat(features, diff, 0);

    let mut residual = ctx.mul_mat(head.scorer, features);
    residual = ctx.add(residual, head.scorer_b);
    residual = ctx.gelu_erf(residual);
    residual = ctx.mul_mat(head.scorer_out, residual);
    residual = ctx.add(residual, head.scorer_out_b); // [1, n_options]

    let scale = |ctx: &mut Context, i: usize| -> TensorId {
        ctx.view_1d(head.scales, 1, i * 4)
    };

    let joint_scale1 = scale(ctx, 1);
    let cos_weighted = ctx.mul(cosine, joint_scale1);
    let joint = ctx.add(cos_weighted, residual);

    let prior_scale = scale(ctx, 0);
    let joint_scale2 = scale(ctx, 2);
    let prior_w = ctx.mul(prior, prior_scale);
    let joint_w = ctx.mul(joint, joint_scale2);
    ctx.add(prior_w, joint_w)
}

/// the trunk + head weights — the trunk is a plain [`Qwen35ModelWeights`]
/// (the backbone is qwen35's, clef.cpp:8-9/27)
pub struct ClefModelWeights {
    pub trunk: Qwen35ModelWeights,
    pub head: ClefHeadTensors,
}

/// clef params: the qwen35 trunk geometry + the head facts
#[derive(Clone)]
pub struct ClefParams {
    pub qwen35: Qwen35Params,
    /// `hparams.f_norm_eps` — the LayerNorm eps the head reads
    /// (clef.cpp:21-22)
    pub f_norm_eps: f32,
    pub head: ClefHeadParams,
}

/// 对照 clef.cpp:322-381 `graph::graph` + build_layer_attn (:415-476) +
/// build_layer_attn_linear (:478-545) — the qwen35 trunk without memory,
/// then the head; returns `res->t_embd` (the padded decision scores) and
/// the graph. `kq_mask` is the causal no-cache [n_tokens, n_tokens] mask of
/// `build_attn_inp_causal` (:393-407) — causal even though the batch is
/// processed by the encoder path.
pub fn build_clef_forward(
    ctx: &mut Context,
    w: &ClefModelWeights,
    p: &ClefParams,
    trunk_layers: &[Qwen35LayerWeights],
    tokens: TensorId,
    pos: TensorId,
    kq_mask: TensorId,
    dec: &ClefDecisionInputs,
    n_tokens: usize,
) -> (TensorId, ggml::Graph) {
    let t = n_tokens as i64;
    let n_layer = trunk_layers.len();
    let mut graph = ggml::Graph::new(4096);
    let qp = &p.qwen35;
    let a = &qp.attn;

    let mut inp_l = ctx.get_rows(w.trunk.tok_embd, tokens);
    inp_l = crate::adapter::lora_embd(ctx, w.trunk.tok_embd, inp_l, tokens);

    for il in 0..n_layer {
        let lw = &trunk_layers[il];
        let inp_sa = inp_l;

        let cur = crate::graph_arch::build_norm_rms(ctx, inp_l, lw.attn_norm, a.norm_eps);

        let cur = if qp.is_recr[il] {
            build_clef_layer_attn_linear(ctx, p, lw, cur, t, il)
        } else {
            build_clef_layer_attn(ctx, p, lw, cur, pos, kq_mask, t, il)
        };

        let cur = ctx.add(cur, inp_sa);

        let ffn_residual = cur;
        let normed =
            crate::graph_arch::build_norm_rms(ctx, ffn_residual, lw.attn_post_norm, a.norm_eps);
        let ffn_out = crate::graph_arch::build_ffn_silu_par(
            ctx,
            normed,
            lw.ffn_gate,
            None,
            lw.ffn_up,
            None,
            lw.ffn_down,
            None,
        );
        inp_l = ctx.add(ffn_out, ffn_residual);
    }

    // the final norm feeds the head (clef.cpp:366-368)
    let cur = crate::graph_arch::build_norm_rms(ctx, inp_l, w.trunk.output_norm, a.norm_eps);

    // the head is always evaluated, so that the graph has the same nodes
    // for every batch (:370)
    let cur = build_head(
        ctx,
        cur,
        &w.head,
        &p.head,
        w.trunk.output,
        w.head.type_embd,
        dec,
        n_tokens,
        p.f_norm_eps,
    );

    // row i of the output is the score of option i; pad to n_tokens rows
    // and add the status (:372-375)
    let n_opt = ctx.ne(cur)[1];
    let cur = ctx.pad(cur, 0, (t - n_opt) as i32, 0, 0);
    let cur = ctx.add(cur, dec.status);
    ctx.set_name(cur, "result_decision");

    graph.build_forward(ctx, cur);
    (cur, graph)
}

/// `build_layer_attn` (clef.cpp:415-476) — qwen35's gated full attention
/// over the no-cache causal mask: Q twice as wide (query + gate), per-head
/// q/k RMS norms, IMRoPE, `sigmoid(gate)`, wo.
#[allow(clippy::too_many_arguments)]
fn build_clef_layer_attn(
    ctx: &mut Context,
    p: &ClefParams,
    lw: &Qwen35LayerWeights,
    cur: TensorId,
    inp_pos: TensorId,
    kq_mask: TensorId,
    t: i64,
    il: usize,
) -> TensorId {
    let qp = &p.qwen35;
    let a = &qp.attn;
    let n_embd_head = qp.n_embd_head_v[il] as i64;
    let n_head = qp.n_head[il] as i64;
    let n_head_kv = qp.n_head_kv[il] as i64;
    let n_rot = qp.n_rot[il] as i64;

    // the Q projection outputs query + gate (clef.cpp:428-434)
    let q_full = crate::adapter::lora_mm(ctx, lw.wq.expect("clef wq"), cur);
    let k = crate::adapter::lora_mm(ctx, lw.wk.expect("clef wk"), cur);
    let v = crate::adapter::lora_mm(ctx, lw.wv.expect("clef wv"), cur);

    let stride = (n_embd_head * 2 * 4) as usize;
    let q = ctx.view_3d(q_full, n_embd_head, n_head, t, stride, stride * n_head as usize, 0);
    let gate = ctx.view_3d(
        q_full,
        n_embd_head,
        n_head,
        t,
        stride,
        stride * n_head as usize,
        (n_embd_head * 4) as usize,
    );
    let gate = ctx.cont_2d(gate, n_embd_head * n_head, t);

    let q = crate::graph_arch::build_norm_rms(ctx, q, lw.attn_q_norm.expect("attn_q_norm"), a.norm_eps);
    let k = ctx.reshape_3d(k, n_embd_head, n_head_kv, t);
    let k = crate::graph_arch::build_norm_rms(ctx, k, lw.attn_k_norm.expect("attn_k_norm"), a.norm_eps);
    let v = ctx.reshape_3d(v, n_embd_head, n_head_kv, t);

    let rope = |ctx: &mut Context, x: TensorId| {
        ctx.rope_multi(
            x,
            inp_pos,
            None,
            n_rot as i32,
            qp.rope_sections,
            a.rope_mode,
            a.n_ctx_orig,
            a.freq_base,
            a.freq_scale,
            a.ext_factor,
            a.attn_factor,
            a.beta_fast,
            a.beta_slow,
        )
    };
    let q = rope(ctx, q);
    let k = rope(ctx, k);

    let kq_scale = if qp.f_attention_scale == 0.0 {
        1.0 / (n_embd_head as f32).sqrt()
    } else {
        qp.f_attention_scale
    };

    // build_attn with wo == nullptr, then the gate, then wo (clef.cpp:452-462)
    let cur = attn_no_cache(ctx, q, k, v, None, kq_mask, kq_scale, None, None);
    let gate_sigmoid = ctx.sigmoid(gate);
    let cur = ctx.mul(cur, gate_sigmoid);
    crate::adapter::lora_mm(ctx, lw.wo.expect("clef wo"), cur)
}

/// `build_layer_attn_linear` (clef.cpp:478-545) — the gated delta net over
/// the whole batch: the conv and recurrent states start from zero and are
/// not kept.
fn build_clef_layer_attn_linear(
    ctx: &mut Context,
    p: &ClefParams,
    lw: &Qwen35LayerWeights,
    cur: TensorId,
    t: i64,
    _il: usize,
) -> TensorId {
    let qp = &p.qwen35;
    let d_inner = qp.ssm_d_inner;
    let head_k_dim = qp.ssm_d_state;
    let num_k_heads = qp.ssm_n_group;
    let num_v_heads = qp.ssm_dt_rank;
    let head_v_dim = d_inner / num_v_heads;

    let qkv_mixed = crate::adapter::lora_mm(ctx, lw.wqkv.expect("clef wqkv"), cur);
    let qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    let qkv_mixed = ctx.reshape_3d(qkv_mixed, qkv_dim, t, 1);

    let z = crate::adapter::lora_mm(ctx, lw.wqkv_gate.expect("clef wqkv_gate"), cur);

    let beta = crate::adapter::lora_mm(ctx, lw.ssm_beta.expect("clef ssm_beta"), cur);
    let beta = ctx.reshape_4d(beta, 1, num_v_heads, t, 1);
    let beta = ctx.sigmoid(beta);

    let alpha = crate::adapter::lora_mm(ctx, lw.ssm_alpha.expect("clef ssm_alpha"), cur);
    let alpha = ctx.reshape_3d(alpha, num_v_heads, t, 1);
    let alpha_biased = ctx.add(alpha, lw.ssm_dt_b.expect("clef ssm_dt"));
    let alpha_softplus = ctx.softplus(alpha_biased);
    let gate = ctx.mul(alpha_softplus, lw.ssm_a.expect("clef ssm_a"));
    let gate = ctx.reshape_4d(gate, 1, num_v_heads, t, 1);

    // zero conv state, not kept (clef.cpp:511-513)
    let conv_kernel = lw.ssm_conv1d.expect("clef ssm_conv1d");
    let conv_kernel_size = ctx.ne(conv_kernel)[0];
    let conv_channels = d_inner + 2 * num_k_heads * head_k_dim;

    let conv_state_t = ctx.new_tensor_3d(GgmlType::F32, conv_kernel_size - 1, conv_channels, 1);
    ctx.arena_resize_tensor(conv_state_t);
    let conv_states = ctx.fill(conv_state_t, 0.0);
    let qkv_t = ctx.transpose(qkv_mixed);
    let conv_input = ctx.concat(conv_states, qkv_t, 0);

    let conv_out = ctx.ssm_conv(conv_input, conv_kernel);
    let conv_out = ctx.silu(conv_out);

    let nb1_qkv = GgmlType::F32.row_size(qkv_dim as usize);
    let rs_k = GgmlType::F32.row_size(head_k_dim as usize);
    let rs_v = GgmlType::F32.row_size(head_v_dim as usize);
    let q_conv = ctx.view_4d(
        conv_out,
        head_k_dim,
        num_k_heads,
        t,
        1,
        rs_k,
        nb1_qkv,
        nb1_qkv * t as usize,
        0,
    );
    let k_conv = ctx.view_4d(
        conv_out,
        head_k_dim,
        num_k_heads,
        t,
        1,
        rs_k,
        nb1_qkv,
        nb1_qkv * t as usize,
        (head_k_dim * num_k_heads * 4) as usize,
    );
    let v_conv = ctx.view_4d(
        conv_out,
        head_v_dim,
        num_v_heads,
        t,
        1,
        rs_v,
        nb1_qkv,
        nb1_qkv * t as usize,
        GgmlType::F32.row_size((2 * head_k_dim * num_k_heads) as usize),
    );

    let q_conv = build_gdn_l2_norm(ctx, q_conv, qp.attn.norm_eps / head_k_dim as f32);
    let k_conv = build_gdn_l2_norm(ctx, k_conv, qp.attn.norm_eps / head_k_dim as f32);

    // zero recurrent state, not kept (clef.cpp:533-534)
    let state_t = ctx.new_tensor_4d(GgmlType::F32, head_v_dim, head_v_dim, num_v_heads, 1);
    ctx.arena_resize_tensor(state_t);
    let state = ctx.fill(state_t, 0.0);

    let gdn_out = ctx.gated_delta_net(q_conv, k_conv, v_conv, gate, beta, state, 1);
    // attn scores: [S_v, H_v, T, 1] (delta-net-base.cpp:858-864)
    let output = ctx.view_4d(
        gdn_out,
        head_v_dim,
        num_v_heads,
        t,
        1,
        (head_v_dim * 4) as usize,
        (head_v_dim * num_v_heads * 4) as usize,
        (head_v_dim * num_v_heads * t * 4) as usize,
        0,
    );

    // gated normalization (clef.cpp:538-542)
    let z_4d = ctx.reshape_4d(z, head_v_dim, num_v_heads, t, 1);
    let normed = crate::graph_arch::build_norm_rms(
        ctx,
        output,
        lw.ssm_norm.expect("clef ssm_norm"),
        qp.attn.norm_eps,
    );
    let z_silu = ctx.silu(z_4d);
    let output = ctx.mul(normed, z_silu);

    let output = ctx.reshape_3d(output, head_v_dim * num_v_heads, t, 1);
    let cur = crate::adapter::lora_mm(ctx, lw.ssm_out.expect("clef ssm_out"), output);
    ctx.reshape_2d(cur, qp.n_embd, t)
}

/// `ClefState::decode` — one whole-model pass: the trunk + head over one
/// ubatch, run like `llama_decode` on the no-memory clef context. Returns
/// the padded [1, n_tokens] decision scores (row i = option i's score; the
/// rows past n_options are the pad).
pub struct ClefStep {
    pub scores: Vec<f32>,
    pub n_tokens: usize,
    pub n_options: usize,
}

pub struct ClefState {
    pub weights: ClefModelWeights,
    pub params: ClefParams,
}

impl ClefState {
    pub fn decode(
        &self,
        gctx: &mut Context,
        watermark: usize,
        tokens: &[i32],
        pos: &[i32],
        decision_order: Option<&[i32]>,
    ) -> Result<ClefStep, String> {
        // `watermark` = `gctx.mark()` taken when the weights were mounted —
        // everything below survives the graph reset (the gemma4-assistant
        // precedent)
        assert!(
            watermark <= gctx.mark(),
            "clef decode: watermark past the context tail"
        );
        let n = tokens.len();
        if n == 0 {
            return Err("clef decode: empty batch".into());
        }
        if pos.len() != n {
            return Err("clef decode: tokens/pos length mismatch".into());
        }

        gctx.reset_graph_to(watermark);
        let tokens_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
        // build_inp_pos of an IMRoPE model: 4 sections per token, the text
        // batch broadcasts the same 1D position to every section
        // (llm_graph_input_pos::set_input + set_position_mrope_1d,
        // llama-graph.cpp:130 / mtmd-helper-common.h:126-141)
        let pos_t = gctx.new_tensor_1d(GgmlType::I32, 4 * n as i64);
        // build_attn_inp_causal (clef.cpp:393-407): F16 under FA, F32
        // otherwise; causal even on the encoder path
        let mask_ty = if self.params.qwen35.attn.use_flash_attn {
            GgmlType::F16
        } else {
            GgmlType::F32
        };
        let kq_mask = gctx.new_tensor_2d(mask_ty, n as i64, n as i64);
        for ti in [tokens_t, pos_t, kq_mask] {
            gctx.arena_resize_tensor(ti);
        }
        gctx.with_i32_mut(tokens_t, |q| q.copy_from_slice(tokens)).unwrap();
        gctx.with_i32_mut(pos_t, |q| {
            for s in 0..4 {
                q[s * n..(s + 1) * n].copy_from_slice(pos);
            }
        })
        .unwrap();

        // the causal mask: token j attends to every token whose position
        // is at or before its own (set_input_kq_mask's causal rule for one
        // sequence, llama-kv-cache.cpp:1557-1705; clef.cpp:393-407 pins
        // causal_attn = true even on the encoder path)
        let visible = |i: usize, j: usize| pos[i] <= pos[j];
        if mask_ty == GgmlType::F16 {
            let bytes = gctx.data_bytes_mut(kq_mask).unwrap();
            let m: &mut [half::f16] = bytemuck::cast_slice_mut(bytes);
            for j in 0..n {
                for i in 0..n {
                    m[j * n + i] = half::f16::from_f32(if visible(i, j) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        } else {
            gctx.with_f32_mut(kq_mask, |m| {
                for j in 0..n {
                    for i in 0..n {
                        m[j * n + i] = if visible(i, j) { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
            })
            .unwrap();
        }

        let dec = ClefDecisionInputs::build(gctx, n, decision_order);
        dec.set_input(gctx, tokens, decision_order);

        let (scores_t, mut gf) = build_clef_forward(
            gctx,
            &self.weights,
            &self.params,
            &self.weights.trunk.layers,
            tokens_t,
            pos_t,
            kq_mask,
            &dec,
            n,
        );
        ggml::compute::graph_compute(gctx, &mut gf, 1);

        let scores: Vec<f32> =
            bytemuck::cast_slice(gctx.data_bytes(scores_t).unwrap()).to_vec();
        Ok(ClefStep {
            scores,
            n_tokens: n,
            n_options: dec.n_options,
        })
    }
}
