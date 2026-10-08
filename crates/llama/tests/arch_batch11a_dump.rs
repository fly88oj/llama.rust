//! arch_batch11a_dump.rs — Rust-side mirror of `parity/ref_decode_dump.c`
//! for the batch-11a archs (same DECDMP1 stream, same shape-skip rule), so
//! `parity/decode_dump_cmp.py` can bisect the first divergent graph node
//! against the reference probe.
//!
//! Drives `DecodeContext::decode_embed` (the `--embeddings --pooling none`
//! path: every token an output row, one ubatch) exactly like the C probe.
//!
//! Run (release):
//!   cargo test --release -p llama --test arch_batch11a_dump -- --ignored \
//!       --nocapture --test-threads 1
//! env:
//!   B11A_DUMP_MODEL=<path>  the synth gguf (default apertus)
//!   B11A_DUMP_OUT=<path>    output .bin (default /tmp/b11a-port.bin)
//!   B11A_FA_OFF=1           the non-FA attention path (default FA on)
//!   B11A_DUMP_PROMPT=<s>    the prompt (default "The capital of France is")

use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::types::GgmlType;
use ggml::{Context, Gguf};
use llama::arch::LlmArch;
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

/// nodes at/above this element count carry no payload (2^19, the C probe rule)
const ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static ACTIVE: AtomicU32 = AtomicU32::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn op_desc(op: ggml::GgmlOp) -> &'static str {
    use ggml::GgmlOp::*;
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
        Sqrt => "SQRT",
        // display-only arms for enum variants added after this file (audio
        // round: sin/cos/sqr/pad_reflect_1d; llama round: mean)
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        Pad => "PAD",
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
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
        return true;
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
        return true;
    }
    if !matches!(node.ty, GgmlType::F32 | GgmlType::F16) {
        st.out.extend(std::iter::repeat(0u8).take(4 * n as usize));
        return true;
    }
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
// the apertus weights bundle (llama-cli's apertus_weights, self-contained)
// ---------------------------------------------------------------------------

fn attn_of(m: &LlamaModel, il: usize, use_flash_attn: bool) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head_k: hp.n_embd_head_k(il) as i64,
        n_embd_head_v: hp.n_embd_head_v(il) as i64,
        n_rot: hp.n_rot(il) as i64,
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

fn forward_of(m: &LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = attn_of(m, 0, fa);
    match m.arch {
        LlmArch::APERTUS => (
            ForwardWeights::Apertus(
                llama::graph_arch::ApertusModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m.layers[..n_trunk]
                        .iter()
                        .map(|l| llama::graph_arch::ApertusLayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            rope_long: l.rope_long,
                            rope_short: l.rope_short,
                            rope_freqs: l.rope_freqs,
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo.unwrap(),
                            wo_b: l.wo_b,
                            attn_q_norm: l.attn_q_norm.unwrap(),
                            attn_q_norm_b: l.attn_q_norm_b,
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            attn_k_norm_b: l.attn_k_norm_b,
                            ffn_norm: l.ffn_norm.unwrap(),
                            ffn_down: l.ffn_down.unwrap(),
                            ffn_up: l.ffn_up.unwrap(),
                        })
                        .collect(),
                },
                llama::graph_arch::ApertusParams {
                    xielu_alpha_n: hp.xielu_alpha_n[..n_trunk].to_vec(),
                    xielu_alpha_p: hp.xielu_alpha_p[..n_trunk].to_vec(),
                    xielu_beta: hp.xielu_beta[..n_trunk].to_vec(),
                    xielu_eps: hp.xielu_eps[..n_trunk].to_vec(),
                    f_attention_scale: hp.f_attention_scale,
                    use_longrope_factors: hp.rope_scaling_type_train
                        == llama::hparams::LlamaRopeScalingType::LONGROPE,
                    attn,
                },
            ),
            attn,
        ),
        LlmArch::GROVEMOE => (
            ForwardWeights::Grovemoe(
                llama::graph_arch::GrovemoeModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m.layers[..n_trunk]
                        .iter()
                        .map(|l| llama::graph_arch::GrovemoeLayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo.unwrap(),
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            attn_q_norm: l.attn_q_norm.unwrap(),
                            ffn_norm: l.ffn_norm.unwrap(),
                            ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                            ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                            ffn_down_exps: l.ffn_down_exps.unwrap(),
                            ffn_up_exps: l.ffn_up_exps.unwrap(),
                            ffn_gate_chexps: l.ffn_gate_chexps.unwrap(),
                            ffn_down_chexps: l.ffn_down_chexps.unwrap(),
                            ffn_up_chexps: l.ffn_up_chexps.unwrap(),
                        })
                        .collect(),
                },
                llama::graph_arch::GrovemoeParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    expert_group_scale: hp.expert_group_scale,
                    n_group_experts: hp.n_group_experts as i64,
                    n_ff_chexp: hp.n_ff_chexp as i64,
                    n_embd_head_k: hp.n_embd_head_k(0) as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            ),
            attn,
        ),
        LlmArch::MINIMAX_M3 => (
            ForwardWeights::MinimaxM3(
                llama::graph_arch::MinimaxM3ModelWeights {
                    tok_embd: m.tok_embd,
                    output_norm: m.output_norm,
                    output: m.output,
                    layers: m.layers[..n_trunk]
                        .iter()
                        .map(|l| llama::graph_arch::MinimaxM3LayerWeights {
                            attn_norm: l.attn_norm.unwrap(),
                            wqkv: l.wqkv,
                            wq: l.wq,
                            wk: l.wk,
                            wv: l.wv,
                            wo: l.wo.unwrap(),
                            attn_q_norm: l.attn_q_norm.unwrap(),
                            attn_k_norm: l.attn_k_norm.unwrap(),
                            ffn_norm: l.ffn_norm.unwrap(),
                            ffn_gate: l.ffn_gate,
                            ffn_down: l.ffn_down,
                            ffn_up: l.ffn_up,
                            ffn_gate_inp: l.ffn_gate_inp,
                            ffn_exp_probs_b: l.ffn_exp_probs_b,
                            ffn_gate_exps: l.ffn_gate_exps,
                            ffn_down_exps: l.ffn_down_exps,
                            ffn_up_exps: l.ffn_up_exps,
                            ffn_gate_shexp: l.ffn_gate_shexp,
                            ffn_down_shexp: l.ffn_down_shexp,
                            ffn_up_shexp: l.ffn_up_shexp,
                            index_q_proj: l.index_q_proj,
                            index_k_proj: l.index_k_proj,
                            index_q_norm: l.index_q_norm,
                            index_k_norm: l.index_k_norm,
                        })
                        .collect(),
                },
                llama::graph_arch::MinimaxM3Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_head: hp.n_head(0) as i64,
                    n_head_kv: hp.n_head_kv(0) as i64,
                    n_embd_head: hp.n_embd_head_k(0) as i64,
                    n_rot: hp.n_rot(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_ff_exp: hp.n_ff_exp(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    msa_blk: hp.indexer_block_size as i64,
                    msa_topk_blocks: hp.indexer_top_k as i64,
                    msa_local: hp.indexer_local_blocks as i64,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                },
            ),
            attn,
        ),
        other => panic!("arch_batch11a_dump: arch {other:?} not wired"),
    }
}

/// Stream every node of one `decode_embed` prefill to `B11A_DUMP_OUT`.
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the batch-11a parity files"]
fn arch_batch11a_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let model_path = std::env::var("B11A_DUMP_MODEL")
        .unwrap_or_else(|_| "/tmp/arch-batch11a/apertus-synth.gguf".to_string());
    let out_path =
        std::env::var("B11A_DUMP_OUT").unwrap_or_else(|_| "/tmp/b11a-port.bin".to_string());
    let prompt = std::env::var("B11A_DUMP_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let fa_off = std::env::var("B11A_FA_OFF").is_ok();

    let file = std::fs::File::open(&model_path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");

    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize(&prompt, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!(
        "tokens: {ids:?} ({} tokens, fa={}, {})",
        ids.len(),
        if fa_off { "off" } else { "on" },
        model_path
    );

    let (w, attn) = forward_of(&model, !fa_off);
    let gctx: Context = model.ctx;
    let mut dctx = DecodeContext::new_with(
        gctx, w, attn, 512, // -c 512
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
    // B11A_DECODE_TAIL=N: append N single-token decode steps to the same
    // node stream — the C probe's --decode-tail mirror. The fed token is
    // FIXED (id 100, B11A_TAIL_TOK) so both sides run the identical graph
    let n_tail: usize = std::env::var("B11A_DECODE_TAIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // B11A_TAIL_IDS="id,id,..." — explicit tail tokens (the C probe's
    // --decode-ids mirror); beyond the list the last id repeats; default 100
    let tail_ids: Vec<i32> = std::env::var("B11A_TAIL_IDS")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    for s in 0..n_tail {
        let tok = if tail_ids.is_empty() {
            100
        } else {
            tail_ids[s.min(tail_ids.len() - 1)]
        };
        let p = (ids.len() + s) as i32;
        dctx.decode(&[tok], &[p]).expect("decode tail");
    }
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
        "arch_batch11a_prefill_node_dump: {nodes} nodes, embd {}x{}, -> {out_path}",
        embd.n_rows, embd.n_embd_out
    );
    let _ = Path::new(&model_path);
}
