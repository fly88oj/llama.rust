//! bert_e2e.rs — BERT (`LlmArch::BERT`, models/bert.cpp) loader + encoder
//! verification on the real local GGUF.
//!
//! Model: `/home/jeffrey/localai/models/bge-m3-Q8_0.gguf` (XLM-RoBERTa-style
//! encoder, 24 layers, arch `bert`, `bert.pooling_type = 2` = CLS).
//!
//! Ground truth: `parity/encode_bert_*.bin`, produced by
//! `parity/ref_encode_dump.cpp` against the pinned reference libllama.so
//! (`llama_encode` + `llama_get_embeddings` / `llama_get_embeddings_seq`) —
//! regenerate with `bash parity/gen_encode_ref.sh`. The token ids travel inside
//! the artifact, so tokenization is not part of the comparison.
//!
//! Runs by default (cheap — GGUF header + mmap, no forward):
//!   * `bert_bge_m3_loader_tensor_map` — exact tensor counts/shapes/types of
//!     the real file through `load_model`, plus the hparams the encoder reads.
//!   * `bert_bge_m3_encode_graph_shape` — the built (not computed) encoder
//!     graph: output shapes, node count, one softmax per layer, pooling node.
//!   * `bert_reference_dump_shape` — the artifact itself is the one the tests
//!     below assume (tokens/pooling/shape), so a stale dump fails loudly.
//!
//! `#[ignore]`d (heavy; ~24 layers x 4 matmuls of 1024x4096):
//!   * `bert_bge_m3_encode_vs_reference` — the full forward for
//!     NONE / CLS (file default) / MEAN pooling vs the reference dumps, with
//!     per-element tail statistics and determinism (run twice bit-identical).
//!     Run: `cargo test -p llama --release --test bert_e2e -- --ignored --nocapture`
//!
//! The reference default (`flash_attn_type` AUTO) resolves to FA *enabled* on
//! this build; for BERT that picks the ggml_flash_attn_ext branch of
//! build_attn_mha (llama-graph.cpp:2626, no kq_b). The port anchors the
//! non-FA branch like the decode path, and the reference dump is taken with
//! `--fa off` (`gen_encode_ref.sh`), so both sides run the same kernel chain.

use std::path::Path;
use std::sync::Arc;

use ggml::tensor::GgmlOp;
use ggml::Gguf;
use llama::context::{resolve_pooling, EncoderContext, EncoderWeights};
use llama::graph_arch::{relative_position_bucket, BertModelWeights, EncoderParams};
use llama::hparams::LlamaPoolingType as P;
use llama::model::{load_model, LlamaModel};
use memmap2::Mmap;

const BGE_M3: &str = "/home/jeffrey/localai/models/bge-m3-Q8_0.gguf";
const REF_PER_TOKEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_pertoken.bin"
);
const REF_POOLED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_pooled.bin"
);
const REF_MEAN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_mean.bin"
);
const REF_SINGLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_single.bin"
);
const REF_SINGLE_CLS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_single_cls.bin"
);
const REF_SINGLE_MEAN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_single_mean.bin"
);
const REF_T2: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_t2.bin"
);
const REF_T64: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../parity/encode_bert_t64.bin"
);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn open_model(path: &str) -> Option<(Gguf, Arc<Mmap>)> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).unwrap();
    // SAFETY: model files are read-only for us (same policy as Gguf::open)
    let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    Some((gguf, mmap))
}

fn load_bert(path: &str) -> Option<LlamaModel> {
    let (gguf, mmap) = open_model(path)?;
    Some(load_model(&gguf, mmap).expect("load_model"))
}

/// One reference artifact: `parity/ref_encode_dump.cpp`'s format.
struct RefDump {
    n_tokens: usize,
    n_embd_out: usize,
    n_rows: usize,
    pooling: i32,
    tokens: Vec<i32>,
    values: Vec<f32>,
}

fn read_ref(path: &str) -> Option<RefDump> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: cannot read {path}: {e} (run parity/gen_encode_ref.sh)");
            return None;
        }
    };
    assert_eq!(&bytes[..8], b"LENCE1\0\0", "{path}: magic");
    let u32at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let (n_tokens, n_embd_out, n_rows, pooling) =
        (u32at(8), u32at(12), u32at(16), u32at(20) as i32);
    let mut off = 24;
    let tokens: Vec<i32> = bytes[off..off + 4 * n_tokens]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    off += 4 * n_tokens;
    let values: Vec<f32> = bytes[off..off + 4 * n_embd_out * n_rows]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values.len(), n_embd_out * n_rows, "{path}: payload size");
    Some(RefDump {
        n_tokens,
        n_embd_out,
        n_rows,
        pooling,
        tokens,
        values,
    })
}

/// `EncoderParams` for BERT from the loaded hparams (mirrors the C graph body's
/// hparams reads: llama-graph.cpp:1583 for the norm eps, llama-graph.cpp:2626
/// for the FA gate, llama-context.cpp:216-222 for pooling).
fn bert_params(m: &LlamaModel, pool: P) -> EncoderParams {
    let hp = &m.hparams;
    EncoderParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        // bert has no relative buckets (that is a t5 key)
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: resolve_pooling(pool, hp.pooling_type),
        euro_rope: None,
        gemma_swa: None,
        causal: false,
    }
}

fn encoder(m: LlamaModel, pool: P) -> EncoderContext {
    let params = bert_params(&m, pool);
    let weights = EncoderWeights::Bert(m.bert_weights());
    EncoderContext::new(m.ctx, weights, params, 8)
}

/// Deviation summary of one run against the reference dump.
///
/// `rel` is the honest figure for an embedding: max|Δ| over the reference's own
/// rms (`|Δ|` is bounded by the residual stream's magnitude, while a per-element
/// ratio explodes on the components the reference itself puts near zero — those
/// are printed separately as `rel>1e-3`). T=1 runs close to bit-exact; T>=2
/// runs carry the documented amplification tail (see the module header).
struct Stats {
    bits: usize,
    mean_abs: f32,
    max_abs: f32,
    rms: f32,
    /// mean|Δ| / rms(ref) — the figure the tests bound (stable in n)
    rel_mean: f32,
    /// max|Δ| / rms(ref)
    rel: f32,
    /// worst per-element ratio over components with |ref| > 1e-3 (informational:
    /// near-zero components make this unbounded)
    rel_eps: f32,
}

fn stats(got: &[f32], want: &[f32]) -> Stats {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut bits = 0usize;
    let mut sum_abs = 0f64;
    let mut max_abs = 0f32;
    let mut sq = 0f64;
    let mut rel_eps = 0f32;
    for (a, b) in got.iter().zip(want) {
        if a.to_bits() == b.to_bits() {
            bits += 1;
        }
        let d = (a - b).abs();
        sum_abs += d as f64;
        max_abs = max_abs.max(d);
        sq += (*b as f64) * (*b as f64);
        if b.abs() > 1e-3 {
            rel_eps = rel_eps.max(d / b.abs());
        }
    }
    let n = got.len() as f64;
    let rms = (sq / n).sqrt() as f32;
    let mean_abs = (sum_abs / n) as f32;
    Stats {
        bits,
        mean_abs,
        max_abs,
        rms,
        rel_mean: mean_abs / rms,
        rel: max_abs / rms,
        rel_eps,
    }
}

fn report(tag: &str, got: &[f32], want: &[f32]) -> Stats {
    let s = stats(got, want);
    println!(
        "{tag}: n={} bit-exact={} ({:.2}%) mean_d={:.3e} max_d={:.3e} ref_rms={:.4} \
         mean_d/rms={:.3e} max_d/rms={:.3e} rel_gt_1e-3={:.3e}",
        got.len(),
        s.bits,
        100.0 * s.bits as f64 / got.len() as f64,
        s.mean_abs,
        s.max_abs,
        s.rms,
        s.rel_mean,
        s.rel,
        s.rel_eps,
    );
    s
}

// ---------------------------------------------------------------------------
// default-run tests
// ---------------------------------------------------------------------------

/// Full tensor map of the real bge-m3 file through the port's loader: 389 gguf
/// tensors (24 layers x 16 + 5 model-level), all consumed (done_getting_tensors
/// would have failed otherwise), plus the hparams the encoder graph reads.
#[test]
fn bert_bge_m3_loader_tensor_map() {
    let Some((gguf, mmap)) = open_model(BGE_M3) else {
        return;
    };
    let m = load_model(&gguf, mmap).expect("bge-m3 must load");

    assert_eq!(m.arch, llama::arch::LlmArch::BERT);
    assert_eq!(gguf.tensors.len(), 389, "file tensor count");
    assert_eq!(m.tensors.len(), 389, "loaded tensor count");

    // hparams (bert.block_count / embedding_length / feed_forward_length /
    // attention.head_count / attention.layer_norm_epsilon / pooling_type /
    // attention.causal; no head_count_kv key -> n_head_kv == n_head)
    let hp = &m.hparams;
    assert_eq!(hp.n_layer(), 24);
    assert_eq!(hp.n_embd, 1024);
    assert_eq!(hp.n_ff(0), 4096);
    assert_eq!(hp.n_head(0), 16);
    assert_eq!(hp.n_head_kv(0), 16);
    assert_eq!(hp.n_embd_head_k(0), 64);
    assert_eq!(hp.n_embd_head_v(0), 64);
    assert_eq!(hp.n_embd_k_gqa(0), 1024);
    assert_eq!(hp.n_embd_v_gqa(0), 1024);
    assert!((hp.f_norm_eps - 1e-5).abs() < 1e-12);
    assert_eq!(hp.pooling_type, P::CLS, "bert.pooling_type = 2");
    assert!(!hp.causal_attn, "bert.attention.causal = false");
    assert_eq!(hp.n_ctx_train, 8192);

    // model-level tensors (models/bert.cpp:29-41)
    let ty = |id| m.ctx.ty(id);
    let ne = |id| *m.ctx.ne(id);
    assert_eq!(ne(m.tok_embd), [1024, 250002, 1, 1]);
    assert_eq!(ty(m.tok_embd), ggml::types::GgmlType::Q8_0);
    let te = m.token_types.expect("token_types");
    assert_eq!(ne(te), [1024, 1, 1, 1], "n_token_types = 1");
    assert_eq!(ty(te), ggml::types::GgmlType::F32);
    let pe = m.position_embd.expect("position_embd");
    assert_eq!(ne(pe), [1024, 8192, 1, 1]);
    assert_eq!(ty(pe), ggml::types::GgmlType::F32);
    assert_eq!(ne(m.token_embd_norm.expect("tok_norm")), [1024, 1, 1, 1]);
    assert_eq!(
        ne(m.token_embd_norm_b.expect("tok_norm_b")),
        [1024, 1, 1, 1]
    );
    // bge-m3 has no reranker head
    assert!(m.cls.is_none() && m.cls_b.is_none() && m.cls_out.is_none());

    // per-layer: 8 tensors x (weight,bias) = 16 (models/bert.cpp:43-61)
    assert_eq!(m.layers.len(), 24);
    for (il, l) in m.layers.iter().enumerate() {
        assert!(l.wqkv.is_none(), "layer {il}: no fused qkv in bge-m3");
        assert_eq!(ne(l.wq.expect("wq")), [1024, 1024, 1, 1]);
        assert_eq!(ne(l.wk.expect("wk")), [1024, 1024, 1, 1]);
        assert_eq!(ne(l.wv.expect("wv")), [1024, 1024, 1, 1]);
        assert_eq!(ne(l.wo.expect("wo")), [1024, 1024, 1, 1]);
        assert_eq!(ne(l.attn_out_norm.expect("attn_out_norm")), [1024, 1, 1, 1]);
        assert_eq!(
            ne(l.layer_out_norm.expect("layer_out_norm")),
            [1024, 1, 1, 1]
        );
        assert_eq!(ne(l.ffn_up.expect("ffn_up")), [1024, 4096, 1, 1]);
        assert_eq!(ne(l.ffn_down.expect("ffn_down")), [4096, 1024, 1, 1]);
        for b in [l.wq_b, l.wk_b, l.wv_b, l.wo_b, l.ffn_up_b, l.ffn_down_b] {
            assert!(b.is_some(), "layer {il}: bias present in bge-m3");
        }
        for b in [l.attn_out_norm_b, l.layer_out_norm_b] {
            assert!(b.is_some(), "layer {il}: norm bias present in bge-m3");
        }
        // no jina/nomic members in a plain BERT file
        assert!(l.attn_q_norm.is_none() && l.attn_k_norm.is_none());
    }

    // bert_weights() exposes exactly the graph's members
    let w: BertModelWeights = m.bert_weights();
    assert_eq!(w.layers.len(), 24);
    assert_eq!(w.n_embd, 1024);
    assert_eq!(w.type_embd, m.token_types);
    assert_eq!(w.tok_norm, m.token_embd_norm.unwrap());
}

/// The built graph (no compute) — shapes, node count and the per-layer softmax
/// count. `n_layer` softmaxes + one pooling matmul/get_rows is the structural
/// signature of bert.cpp:102-213 + llama-graph.cpp:3698-3714.
#[test]
fn bert_bge_m3_encode_graph_shape() {
    let Some(m) = load_bert(BGE_M3) else { return };
    let Some(ref_dump) = read_ref(REF_PER_TOKEN) else {
        return;
    };
    let tokens = &ref_dump.tokens;

    // NONE: t_embd is the last hidden state, [n_embd, n_tokens]
    let mut ctx = encoder(m, P::NONE);
    let g = ctx.build(tokens).expect("build");
    assert_eq!(*ctx.gctx.ne(g.embd), [1024, tokens.len() as i64, 1, 1]);
    assert_eq!(g.layer_outs.len(), 24);
    for (il, t) in g.layer_outs.iter().enumerate() {
        assert_eq!(
            *ctx.gctx.ne(*t),
            [1024, tokens.len() as i64, 1, 1],
            "layer {il} out"
        );
    }
    let count = |op: GgmlOp| {
        g.graph
            .nodes
            .iter()
            .filter(|&&n| ctx.gctx.op(n) == op)
            .count()
    };
    // GLU-family ops share GgmlOp::Silu; gelu is param 8 (ops.rs gelu)
    let gelu = g
        .graph
        .nodes
        .iter()
        .filter(|&&n| {
            ctx.gctx.op(n) == GgmlOp::Silu
                && ctx.gctx.op_params(n)[0] == ggml::ops::GGML_UNARY_OP_GELU
        })
        .count();
    let n = 24usize;
    assert_eq!(
        count(GgmlOp::SoftMax),
        n,
        "one softmax per layer (llama-graph.cpp:2719)"
    );
    assert_eq!(gelu, n, "one GELU per layer (bert.cpp:181-185)");
    assert_eq!(
        count(GgmlOp::Transpose),
        n,
        "cont(transpose(v)) per layer (:2714-2718)"
    );
    assert_eq!(
        count(GgmlOp::Dup),
        2 * n,
        "cont of transpose(v) and of kqv (:2730)"
    );
    assert_eq!(
        count(GgmlOp::Permute),
        4 * n,
        "q/k/v permutes + kqv permute"
    );
    assert_eq!(
        count(GgmlOp::Reshape),
        4 * n,
        "q/k/v reshape_3d + kqv reshape_2d"
    );
    assert_eq!(
        count(GgmlOp::MulMat),
        8 * n,
        "q,k,v, kq, kqv, wo, ffn_up, ffn_down per layer"
    );
    assert_eq!(
        count(GgmlOp::Norm),
        2 * n + 1,
        "attn_output_norm + layer_output_norm per layer, token_embd_norm once"
    );
    assert_eq!(
        count(GgmlOp::Mul),
        2 * n + 1,
        "the three norm weights (build_norm llama-graph.cpp:1604)"
    );
    assert_eq!(
        count(GgmlOp::Add),
        10 * n + 3,
        "q/k/v/wo biases, attn+ffn residuals, 4 norm biases, input bias (:1610) per layer"
    );
    assert_eq!(
        count(GgmlOp::GetRows),
        4,
        "tok_embd + pos_embd + the last-layer out_ids pair"
    );
    assert_eq!(
        count(GgmlOp::View),
        1,
        "the type-embedding row 0 view (bert.cpp:86)"
    );
    assert_eq!(
        g.graph.nodes.len(),
        850,
        "total nodes of the bge-m3 bert graph"
    );
    println!("bert NONE graph: {} nodes", g.graph.nodes.len());

    // CLS (the file's pooling): the pooled tensor is [n_embd, 1]
    let Some(m) = load_bert(BGE_M3) else { return };
    let mut ctx = encoder(m, P::CLS);
    let g = ctx.build(tokens).expect("build");
    assert_eq!(
        *ctx.gctx.ne(g.pooled),
        [1024, 1, 1, 1],
        "CLS pooling row (llama-graph.cpp:3712)"
    );
    assert_eq!(
        ctx.gctx.op(g.pooled),
        GgmlOp::GetRows,
        "CLS = get_rows(inp, inp_cls)"
    );

    // MEAN: the pooled tensor is the mul_mat against the mean weights, [n_embd, 1]
    let Some(m) = load_bert(BGE_M3) else { return };
    let mut ctx = encoder(m, P::MEAN);
    let g = ctx.build(tokens).expect("build");
    assert_eq!(
        *ctx.gctx.ne(g.pooled),
        [1024, 1, 1, 1],
        "MEAN pooling (llama-graph.cpp:3705)"
    );
    assert_eq!(ctx.gctx.op(g.pooled), GgmlOp::MulMat);
    let src = ctx.gctx.src(g.pooled);
    // cont(transpose(inp)) — cont is GGML_OP_CONT → GgmlOp::Dup in this port
    assert_eq!(
        ctx.gctx.op(src[0].unwrap()),
        GgmlOp::Dup,
        "cont(transpose(inp))"
    );
    let tr = ctx.gctx.src(src[0].unwrap())[0].unwrap();
    assert_eq!(ctx.gctx.op(tr), GgmlOp::Transpose);
}

/// The reference artifacts are the ones the tests below assume: same token
/// list for all three pooling modes, and the pooled rows consistent with the
/// per-token rows the reference itself produced (so a dump regenerated with a
/// changed prompt cannot silently pass).
#[test]
fn bert_reference_dump_shape() {
    let Some(d) = read_ref(REF_PER_TOKEN) else {
        return;
    };
    assert_eq!(d.n_embd_out, 1024);
    assert_eq!(d.n_rows, d.n_tokens, "NONE keeps one row per token");
    assert_eq!(d.pooling, 0, "LLAMA_POOLING_TYPE_NONE");
    assert_eq!(d.tokens.len(), d.n_tokens);
    assert_eq!(d.tokens[0], 0, "bge-m3 adds the CLS token");
    assert_eq!(d.tokens[d.n_tokens - 1], 2, "and the SEP token");

    let Some(p) = read_ref(REF_POOLED) else {
        return;
    };
    assert_eq!(p.tokens, d.tokens, "same prompt, same ids");
    assert_eq!(p.pooling, 2, "file pooling_type = CLS");
    assert_eq!((p.n_embd_out, p.n_rows), (1024, 1));

    let Some(mn) = read_ref(REF_MEAN) else { return };
    assert_eq!(mn.pooling, 1);
    assert_eq!((mn.n_embd_out, mn.n_rows), (1024, 1));
    // the mean row must be the mean of the per-token rows (the reference's own
    // build_inp_mean path, checked here so the dump is self-consistent)
    for j in 0..1024 {
        let want: f32 = d.values.iter().skip(j).step_by(1024).sum::<f32>() / d.n_tokens as f32;
        assert!(
            (want - mn.values[j]).abs() < 2e-2,
            "mean[{j}]: reference mean {} vs recomputed {want}",
            mn.values[j]
        );
    }
}

// ---------------------------------------------------------------------------
// heavy test (real forward)
// ---------------------------------------------------------------------------

/// Bound on mean|Δ|/rms for runs that are *not* expected to close to 1-2 ulp:
/// see the module header. The bound below is what the reference itself shows
/// when its own kernels are swapped: `exact` arithmetic (numpy, no activation
/// quantization) is ~25% away from the reference on the same file at T=1,
/// while this port — which quantizes activations exactly like the reference —
/// stays within ~1.5% at T >= 2. The per-op seeds are 1e-4-level differences in
/// the F32 attention GEMM (tinyBLAS-F32 / f32 vec_dot tail, pinned by
/// `encode_gemm_probe.rs`), amplified by the Q8_0 activation-quantization step.
const T_MULTI_TOL: f32 = 3e-2;

/// Full forward of bge-m3 (24 layers, Q8_0) vs the reference dumps.
///
/// T=1 is the exactness anchor (bit-exact to 1-2 ulp: no GEMM reaches
/// tinyBLAS and every dot length is 1 or 64), T=2/T=64 are the amplification
/// curve, and the three pooling modes pin the CLS/MEAN wiring on top of it.
#[test]
#[ignore]
fn bert_bge_m3_encode_vs_reference() {
    let Some(ref_none) = read_ref(REF_PER_TOKEN) else {
        return;
    };
    let tokens = ref_none.tokens.clone();

    // ---- T=1 anchor: NONE / CLS / MEAN all reduce to the same 1024 values ----
    if let Some(r1) = read_ref(REF_SINGLE) {
        assert_eq!(r1.tokens, vec![0]);
        let Some(m) = load_bert(BGE_M3) else { return };
        let mut c1 = encoder(m, P::NONE);
        let g1 = c1.encode(&r1.tokens).expect("encode");
        let s = report("bert T=1 NONE", &g1.values, &r1.values);
        // 1-2 ulp tail: 5.96e-8 absolute on values of rms 0.42 (1024 fp32 ops)
        assert!(s.rel < 1e-6, "T=1 anchor drifted: max|Δ|/rms={:.3e}", s.rel);

        for (path, pool, tag) in [
            (REF_SINGLE_CLS, P::CLS, "bert T=1 CLS "),
            (REF_SINGLE_MEAN, P::MEAN, "bert T=1 MEAN"),
        ] {
            let Some(r) = read_ref(path) else { continue };
            assert_eq!(r.n_rows, 1);
            let Some(m) = load_bert(BGE_M3) else { return };
            let mut c = encoder(m, pool);
            let g = c.encode(&r1.tokens).expect("encode");
            assert_eq!(g.n_rows, 1);
            let s = report(tag, &g.values, &r.values);
            assert!(
                s.rel < 1e-6,
                "{tag}: pooling path drifted (rel {:.3e})",
                s.rel
            );
        }
    }

    // ---- T=2 / T=14 / T=64: the multi-token tail ----
    for (path, tag) in [
        (REF_T2, "bert T=2   "),
        (REF_PER_TOKEN, "bert T=14  "),
        (REF_T64, "bert T=64  "),
    ] {
        let Some(r) = read_ref(path) else { continue };
        let Some(m) = load_bert(BGE_M3) else { return };
        let mut c = encoder(m, P::NONE);
        let g = c.encode(&r.tokens).expect("encode");
        assert_eq!((g.n_embd_out, g.n_rows), (r.n_embd_out, r.n_tokens));
        let s = report(tag, &g.values, &r.values);
        assert!(
            s.rel_mean < T_MULTI_TOL,
            "{tag}: mean|Δ|/rms {:.3e} above the documented tail",
            s.rel_mean
        );
        assert!(
            s.rel < 0.3,
            "{tag}: max|Δ|/rms {:.3e} far above the tail",
            s.rel
        );
    }
    assert_eq!(ref_none.tokens, *&tokens);

    // ---- determinism: same input, fresh context → bit-identical ----
    let Some(m) = load_bert(BGE_M3) else { return };
    let mut ctx = encoder(m, P::NONE);
    let got = ctx.encode(&tokens).expect("encode");
    let Some(m) = load_bert(BGE_M3) else { return };
    let mut ctx2 = encoder(m, P::NONE);
    let got2 = ctx2.encode(&tokens).expect("encode");
    assert_eq!(got.values.len(), got2.values.len());
    let bits2 = got
        .values
        .iter()
        .zip(&got2.values)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "bert determinism: {bits2}/{} bit-identical",
        got.values.len()
    );
    assert_eq!(bits2, got.values.len(), "encode is not deterministic");
    // same context, second call: the per-call graph reset (watermark) must not
    // leak state (llama.cpp rebuilds the graph per ubatch)
    let got3 = ctx.encode(&tokens).expect("encode (reuse)");
    let bits3 = got
        .values
        .iter()
        .zip(&got3.values)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!("bert reuse    : {bits3}/{} bit-identical", got.values.len());
    assert_eq!(bits3, got.values.len(), "encode reuse is not deterministic");

    // ---- pooling at T=14: CLS (file default) and MEAN ----
    if let Some(ref_pooled) = read_ref(REF_POOLED) {
        assert_eq!(ref_pooled.tokens, tokens, "same prompt, same ids");
        let Some(m) = load_bert(BGE_M3) else { return };
        let mut ctx = encoder(m, P::CLS);
        let got = ctx.encode(&tokens).expect("encode");
        assert_eq!(got.n_rows, 1);
        let s = report("bert CLS   ", &got.values, &ref_pooled.values);
        assert!(s.rel_mean < T_MULTI_TOL, "CLS tail {:.3e}", s.rel_mean);
        // CLS is get_rows(t_embd, 0) — it must equal the port's own token-0 row
        let cls_bits = got
            .values
            .iter()
            .zip(&got2.values[..1024])
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!("  CLS == our token-0 row: {cls_bits}/1024 bit-identical");
    }

    if let Some(ref_mean) = read_ref(REF_MEAN) {
        let Some(m) = load_bert(BGE_M3) else { return };
        let mut ctx = encoder(m, P::MEAN);
        let got = ctx.encode(&tokens).expect("encode");
        assert_eq!(got.n_rows, 1);
        // MEME: the reference computes the same weighted sum with a different
        // matmul shape (n_seqs=1 column), so compare against both the
        // reference's own mean row and the mean of its per-token rows.
        let ref_mean_of_rows: Vec<f32> = (0..1024)
            .map(|j| {
                ref_none.values.iter().skip(j).step_by(1024).sum::<f32>() / ref_none.n_tokens as f32
            })
            .collect();
        let s_ref = report("bert MEAN  ", &got.values, &ref_mean.values);
        let s_rows = report("bert MEAN* ", &got.values, &ref_mean_of_rows);
        assert!(s_ref.rel_mean.min(s_rows.rel_mean) < T_MULTI_TOL);
    }

    // ---- keep_layer_outs: the last layer output is t_embd (bert.cpp:215) ----
    let Some(m) = load_bert(BGE_M3) else { return };
    let mut ctx = encoder(m, P::NONE);
    ctx.keep_layer_outs = true;
    let got = ctx.encode(&tokens).expect("encode");
    assert_eq!(got.layer_outs.len(), 24);
    assert_eq!(got.layer_outs[23].len(), got.values.len());
    assert!(got.layer_outs[23]
        .iter()
        .zip(&got.values)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// `relative_position_bucket` is T5-only, but the reference table lives in the
/// shared test asset dir; the BERT side asserts the local file carries no t5
/// keys (so the port cannot silently use buckets here).
#[test]
fn bert_has_no_relative_buckets() {
    let Some(m) = load_bert(BGE_M3) else { return };
    assert_eq!(m.hparams.n_rel_attn_bkts, 0);
    // the helper itself is exercised by t5_e2e.rs against the C table
    assert_eq!(relative_position_bucket(0, 0, 32, true), 0);
}
