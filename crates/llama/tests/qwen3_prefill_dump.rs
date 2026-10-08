//! qwen3_prefill_dump.rs — Rust-side mirror of `parity/ref_decode_dump.c`:
//! stream-dump EVERY graph node of one decoder prefill through the port's
//! eval callback (`ggml::compute::set_eval_callback`, the cb_eval hook the
//! imatrix tool uses), writing the exact same `DECDMP1` format so
//! `parity/decode_dump_cmp.py` can bisect the first divergent op against the
//! reference dump.
//!
//! Drives `DecodeContext::decode_embed` (the `-fe` / `--embeddings --pooling
//! none` server path: every token an output row, one ubatch) with the same
//! prompt/token ids the reference probe uses. The element walk mirrors the C
//! probe: 4-dim stride walk over `nb`, F16 -> F32 exact, and NO payload for
//! nodes with `n_elems >= 2^19` (the whole-cache KV views) — the C probe skips
//! exactly the same nodes by the same shape rule, so the streams align.
//!
//! Build/run (release; the model is 0.6 GiB):
//!   cargo test --release -p llama --test qwen3_prefill_dump -- --ignored \
//!       --nocapture
//! env:
//!   QWEN3_DUMP_OUT=<path>   output .bin (default /tmp/port_dec.bin)
//!   QWEN3_FA_OFF=1          run the non-FA attention path (default FA on)
//!
//! then compare:
//!   python3 parity/decode_dump_cmp.py /tmp/ref_dec.bin /tmp/port_dec.bin

use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::types::GgmlType;
use ggml::{Context, Gguf};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{Qwen3LayerWeights, Qwen3ModelWeights};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const QWEN3_EMB: &str = "/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf";

/// Same prompt as parity/ref_decode_dump.c's default.
const PROMPT: &str = "The capital of France is Paris.";

/// nodes at/above this element count carry no payload (2^19, see module doc)
const ELEM_CAP: u64 = 1 << 19;

// ---------------------------------------------------------------------------
// dump state (the eval callback is a plain fn pointer)
// ---------------------------------------------------------------------------

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static ACTIVE: AtomicU32 = AtomicU32::new(0);
/// both dump tests share the process-global DUMP/ACTIVE state — hold this for
/// the whole test so a parallel `--ignored` run of the other one cannot
/// interleave its nodes into the stream (a corrupted mixed dump was exactly
/// the symptom)
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn op_desc(op: ggml::GgmlOp) -> &'static str {
    use ggml::GgmlOp::*;
    // base names only: the port encodes RMS_NORM as Norm+flag and the unary
    // variants (SILU/GELU/...) as Silu+params — the comparator aligns by
    // index/shape/name, the op string is for display
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        Mul => "MUL",
        Div => "DIV",
        Sub => "SUB",
        Norm => "NORM",
        SquaredMulMat => "MUL_MAT_SQ",
        MulMat => "MUL_MAT",
        Scale => "SCALE",
        Cpy => "CPY",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        DiagMaskInf => "DIAG_MASK_INF",
        SoftMax => "SOFT_MAX",
        RoPE => "ROPE",
        RoPEBack => "ROPE_BACK",
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => "SILU",
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => "GLU",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        Clamp => "CLAMP",
        Gdn => "GATED_DELTA_NET",
        Im2col => "IM2COL",
        Upscale => "UPSCALE",
        Fill => "FILL",
        LightningIndexer => "LIGHTNING_INDEXER",
        TopK => "TOP_K",
        // arch batch 7 (deepseek4) ops — the deepseek4 dump lives in
        // arch_batch7_e2e.rs with the full UNARY/GLU-aware op_desc
        Sqrt => "SQRT",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        // arch batch 10 (graniteswitch) — the router lane's right-pad
        Pad => "PAD",
        // display-only arms for enum variants added after this file (neither
        // op appears in a qwen3/gpt-oss graph — exhaustiveness only)
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
        // audio/mean rounds' later variants — display only, exhaustiveness
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn type_desc(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Bf16 => "bf16",
        GgmlType::I64 => "i64",
        GgmlType::I32 => "i32",
        GgmlType::I16 => "i16",
        GgmlType::I8 => "i8",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q5_1 => "q5_1",
        GgmlType::Q8_1 => "q8_1",
        _ => "other",
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len().min(255);
    buf.push(len as u8);
    buf.extend_from_slice(&s.as_bytes()[..len]);
}

fn dump_cb(node: &EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true; // want every node
    }
    let mut guard = DUMP.get().unwrap().lock().unwrap();
    let Some(st) = guard.as_mut() else {
        return true;
    };
    let n: i64 = node.ne.iter().product();
    st.nodes += 1;
    put_str(&mut st.out, op_desc(node.op));
    put_str(&mut st.out, node.name);
    put_str(&mut st.out, type_desc(node.ty));
    st.out
        .extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
    st.out.extend_from_slice(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if n as u64 >= ELEM_CAP {
        return true; // no payload (same shape rule as the C probe)
    }
    let bytes_per = if node.ty == GgmlType::F32 { 4 } else { 2 };
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        // mirror the C probe's zero payload for non-float nodes
        st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
        return true;
    }
    let _ = bytes_per;
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v: f32 = if node.ty == GgmlType::F32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            let h = half::f16::from_le_bytes([data[off], data[off + 1]]);
            h.to_f32()
        };
        st.out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

// ---------------------------------------------------------------------------
// model loading (same pattern as qwen3_e2e.rs)
// ---------------------------------------------------------------------------

fn qwen3_params(m: &LlamaModel, use_flash_attn: bool) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn,
    }
}

fn qwen3_weights(m: &LlamaModel) -> Qwen3ModelWeights {
    Qwen3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, x)| Qwen3LayerWeights {
                attn_norm: x
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: x.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: x.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: x.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: x.wq_b,
                wk_b: x.wk_b,
                wv_b: x.wv_b,
                wo: x.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: x
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: x
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: x.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: x.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: x.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: x.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// Stream every node of one `decode_embed` prefill to `QWEN3_DUMP_OUT`.
#[test]
#[ignore = "manual: real model forward, writes a node dump next to the C probe's"]
fn qwen3_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out_path =
        std::env::var("QWEN3_DUMP_OUT").unwrap_or_else(|_| "/tmp/port_dec.bin".to_string());
    let fa_off = std::env::var("QWEN3_FA_OFF").is_ok();

    if !Path::new(QWEN3_EMB).exists() {
        eprintln!("SKIP: {QWEN3_EMB} not present");
        return;
    }
    let file = std::fs::File::open(QWEN3_EMB).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");

    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize(PROMPT, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!(
        "tokens: {ids:?} ({} tokens, fa={})",
        ids.len(),
        if fa_off { "off" } else { "on" }
    );

    // the model's own build context owns the weight TensorIds (borrow first,
    // then move the ctx out — same order as qwen3_e2e.rs)
    let w = qwen3_weights(&model);
    let attn = qwen3_params(&model, !fa_off);
    let mut gctx: Context = model.ctx;
    let mut dctx = DecodeContext::new_with(
        gctx,
        ForwardWeights::Qwen3(w),
        attn,
        512, // -c 512 (parity/embd_rows_probe.sh)
        8,   // -t 8
        2048,
    )
    .with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);

    DUMP.get_or_init(|| {
        Mutex::new(Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        }))
    });
    // reset the state
    {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        *guard = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    set_eval_callback(Some(dump_cb));
    ACTIVE.store(1, Ordering::SeqCst);
    let embd = dctx.decode_embed(&ids, &pos).expect("decode_embed");
    ACTIVE.store(0, Ordering::SeqCst);
    set_eval_callback(None);

    // assemble the file: magic + tokens + node count + node stream
    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    let mut f = std::fs::File::create(&out_path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
    for &id in &ids {
        f.write_all(&id.to_le_bytes()).unwrap();
    }
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    drop(f);

    println!(
        "qwen3_prefill_node_dump: {nodes} nodes, embd rows {} x {}, -> {out_path}",
        embd.n_rows, embd.n_embd_out
    );
}

/// Same dump for gpt-oss-20b MXFP4 (the FA + sinks + MoE pipeline) — drives
/// `DecodeContext::decode_embed` like the reference probe drives
/// `llama_decode --embeddings`, writing the same DECDMP1 stream.
///
/// Run:
///   cargo test --release -p llama --test qwen3_prefill_dump -- --ignored \
///       --nocapture gpt_oss_prefill_node_dump
/// env:
///   GPTOSS_DUMP_OUT=<path>  (default /tmp/port_oss.bin)
#[test]
#[ignore = "manual: real 12 GiB model forward"]
fn gpt_oss_prefill_node_dump() {
    use llama::graph_arch::{GptOssLayerWeights, GptOssModelWeights, GptOssParams};

    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const GPT_OSS: &str =
        "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";
    // the same prompt parity/ref_decode_dump.c was run with for /tmp/ref_oss.bin
    const OSS_PROMPT: &str = "The capital of France is";
    let out_path =
        std::env::var("GPTOSS_DUMP_OUT").unwrap_or_else(|_| "/tmp/port_oss.bin".to_string());
    let fa_off = std::env::var("QWEN3_FA_OFF").is_ok();

    let file = std::fs::File::open(GPT_OSS).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize(OSS_PROMPT, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();

    let (attn, gp) = {
        let hp = &model.hparams;
        let rope = hp.rope_runtime();
        let n_layer = model.layers.len();
        let attn = AttnParams {
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head_k: hp.n_embd_head_k(0) as i64,
            n_embd_head_v: hp.n_embd_head_v(0) as i64,
            n_rot: hp.n_rot(0) as i64,
            rope_mode: hp.rope_type as i32,
            n_ctx_orig: rope.n_ctx_orig_yarn,
            freq_base: hp.rope_freq_base_train,
            freq_scale: rope.freq_scale,
            ext_factor: rope.ext_factor,
            attn_factor: rope.attn_factor,
            beta_fast: rope.beta_fast,
            beta_slow: rope.beta_slow,
            norm_eps: hp.f_norm_rms_eps,
            use_flash_attn: !fa_off,
        };
        let gp = GptOssParams {
            n_expert: hp.n_expert as i64,
            n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
            swiglu_oai_alpha: 1.702,
            swiglu_oai_limit: 7.0,
            expert_weights_scale: hp.expert_weights_scale,
            is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
            rope_freq_base_swa: hp.rope_freq_base_train_swa,
            rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
        };
        (attn, gp)
    };
    let w = GptOssModelWeights {
        tok_embd: model.tok_embd,
        output_norm: model.output_norm,
        output: model.output,
        output_b: model.output_b,
        layers: model
            .layers
            .iter()
            .enumerate()
            .map(|(il, x)| GptOssLayerWeights {
                attn_norm: x
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: x
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: x.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: x.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: x.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: x.wq_b,
                wk_b: x.wk_b,
                wv_b: x.wv_b,
                wo: x.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: x.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                attn_sinks: x
                    .attn_sinks
                    .unwrap_or_else(|| panic!("layer {il}: attn_sinks")),
                ffn_gate_inp: x
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_inp_b: x
                    .ffn_gate_inp_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_b")),
                ffn_up_exps: x
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_up_exps_b: x
                    .ffn_up_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps_b")),
                ffn_gate_exps: x
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_gate_exps_b: x
                    .ffn_gate_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps_b")),
                ffn_down_exps: x
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_down_exps_b: x
                    .ffn_down_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps_b")),
            })
            .collect(),
    };

    let mut gctx: Context = model.ctx;
    let swa = llama::kv_cache::SwaCacheSpec::from_hparams(&model.hparams);
    let mut dctx =
        DecodeContext::new_with_swa(gctx, ForwardWeights::GptOss(w, gp), attn, 512, 8, 2048, swa)
            .with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);

    DUMP.get_or_init(|| {
        Mutex::new(Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        }))
    });
    {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        *guard = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }
    set_eval_callback(Some(dump_cb));
    ACTIVE.store(1, Ordering::SeqCst);
    let embd = dctx.decode_embed(&ids, &pos).expect("decode_embed");
    ACTIVE.store(0, Ordering::SeqCst);
    set_eval_callback(None);

    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    let mut f = std::fs::File::create(&out_path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
    for &id in &ids {
        f.write_all(&id.to_le_bytes()).unwrap();
    }
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    drop(f);
    println!(
        "gpt_oss_prefill_node_dump: {nodes} nodes, embd rows {} x {}, -> {out_path}",
        embd.n_rows, embd.n_embd_out
    );
}
