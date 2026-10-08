//! glm5_dump.rs — Rust-side mirror of `parity/ref_decode_dump.c` for the
//! NEW glm5-next arch (def4d406a): the same `DECDMP1` node stream, the same
//! shape-skip rule, so `parity/decode_dump_cmp.py` can bisect the first
//! divergent graph node against the NEW-reference probe (batch 19 — the
//! kda/dsa/kpool graph family acceptance).
//!
//! Drives `DecodeContext::decode_embed` (the `--embeddings --pooling none`
//! path: every token an output row, one ubatch) for the prefill, then N
//! single-token `decode` tail steps — the C probe's `--decode-tail` mirror.
//!
//! Run (release):
//!   cargo test --release -p llama --test glm5_dump -- --ignored --nocapture
//! env:
//!   GLM5_DUMP_MODEL=<path>  the synth gguf
//!   GLM5_DUMP_OUT=<path>    output .bin (default /tmp/hq-glm5-port.bin)
//!   GLM5_DUMP_PROMPT=<s>    the prompt (default "hi")
//!   GLM5_DECODE_TAIL=N      tail decode steps (default 12)
//!   GLM5_TAIL_IDS="id,..."  explicit tail tokens (default 100)

use std::io::Write as _;
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
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPreGated => "dsv4_hc_pre_gated(x, gate, scale)",
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
// the glm5-next weights bundle (llama-cli's glm5_weights, self-contained)
// ---------------------------------------------------------------------------

fn glm5_weights(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Glm5NextModelWeights {
    llama::graph_arch::Glm5NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| llama::graph_arch::Glm5NextLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                hc_attn_fn: l.hc_attn_fn,
                hc_attn_base: l.hc_attn_base,
                hc_attn_scale: l.hc_attn_scale,
                hc_ffn_fn: l.hc_ffn_fn,
                hc_ffn_base: l.hc_ffn_base,
                hc_ffn_scale: l.hc_ffn_scale,
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wqkv: l.wqkv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wo: l.wo,
                attn_q_a_norm: l.attn_q_a_norm,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wq_a: l.wq_a,
                wq_b: l.wq_b,
                wkv_a_mqa: l.wkv_a_mqa,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                indexer_kpool_gate: l.indexer_kpool_gate,
                indexer_kpool_ape: l.indexer_kpool_ape,
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
            })
            .collect(),
    }
}

fn forward_of(m: &LlamaModel) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let rope = hp.rope_runtime();
    // the MLA geometry: the attn half's cache rows are the kv_lora_rank-wide
    // latent (key_length = kv_lora, n_rot == 0, one kv head)
    let attn = AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: 1,
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
        // the probe's --fa off default (the verified baseline)
        use_flash_attn: false,
    };
    let params = llama::graph_arch::Glm5NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(0) as i64,
        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_lora_q: hp.n_lora_q as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_rot: hp.n_rot(0) as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_expert_shared: hp.n_expert_shared as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        indexer_kpool_select_tail: hp.indexer_kpool_select_tail,
        is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
        hc: hp.dsv4_hc_mult as i64,
        hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters,
        hc_eps: hp.dsv4_hc_eps,
        f_norm_eps: hp.f_norm_eps,
    };
    (ForwardWeights::Glm5Next(glm5_weights(m, n_trunk), params), attn)
}

/// Stream every node of one `decode_embed` prefill (+ N decode tail steps)
/// to `GLM5_DUMP_OUT` — the C probe's `--decode-tail` mirror.
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the glm5-next parity file"]
fn glm5_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let model_path = std::env::var("GLM5_DUMP_MODEL")
        .unwrap_or_else(|_| "/tmp/syncm-glm5/glm5-next-synth.gguf".to_string());
    let out_path =
        std::env::var("GLM5_DUMP_OUT").unwrap_or_else(|_| "/tmp/hq-glm5-port.bin".to_string());
    let prompt =
        std::env::var("GLM5_DUMP_PROMPT").unwrap_or_else(|_| "hi".to_string());
    let n_tail: usize = std::env::var("GLM5_DECODE_TAIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let tail_ids: Vec<i32> = std::env::var("GLM5_TAIL_IDS")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();

    let file = std::fs::File::open(&model_path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");
    assert_eq!(model.arch, LlmArch::GLM5_NEXT);

    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize(&prompt, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!("tokens: {ids:?} ({} tokens, {})", ids.len(), model_path);

    let (w, attn) = forward_of(&model);
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
    let _embd = dctx.decode_embed(&ids, &pos).expect("decode_embed");
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
    println!("glm5_dump: {nodes} nodes -> {out_path}");
}

// ---------------------------------------------------------------------------
// batch 19 — the graph-side acceptance (a DEFAULT test): regenerate the
// synthetic file (the writer of tests/glm5_e2e.rs, seeded identically),
// drive the same 2-token prefill + 12 decode steps, and bit-compare the
// named graph nodes against the NEW reference probe's stored stream
// (parity/glm5/nodes_ref.bin — ref_decode_dump.c over def4d406a with
// `--fa off --decode-tail 12 --text hi`; parity/glm5_parity.sh regenerates
// it). The comparator pairs every node carrying a graph name and equal
// payload length, and additionally walks the full index-paired sequence:
// all 76 named nodes and all paired values must be byte-identical.
// ---------------------------------------------------------------------------

/// the synthetic geometry — the constants of tests/glm5_e2e.rs (must stay
/// in lockstep with the fixture that produced nodes_ref.bin)
mod synth {
    pub const VOCAB_SPM: &str =
        "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
    pub const N_LAYER: usize = 3;
    pub const N_EMBD: i64 = 64;
    pub const N_HEAD: i64 = 4;
    pub const D_CONV: i64 = 4;
    pub const HEAD_DIM_KDA: i64 = 16;
    pub const KV_LORA: i64 = 16;
    pub const Q_LORA: i64 = 16;
    pub const K_MLA: i64 = 24;
    pub const V_MLA: i64 = 16;
    pub const N_FF_EXP: i64 = 32;
    pub const N_EXPERT: i64 = 4;
    pub const N_EXPERT_SHARED: i64 = 1;
    pub const INDEXER_HEAD: i64 = 4;
    pub const INDEXER_HEAD_SIZE: i64 = 16;
    pub const TOP_K: i64 = 8;
    pub const KPOOL: i64 = 4;
    pub const N_CTX: u32 = 512;

    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32 / 8_388_608.0) - 1.0
        }
    }

    pub fn tensors_for(n_vocab: i64) -> Vec<(String, Vec<i64>, bool)> {
        let mut v: Vec<(String, Vec<i64>, bool)> = Vec::new();
        macro_rules! push {
            ($name:expr, $ne:expr, $norm:expr) => {
                v.push(($name.to_string(), $ne.to_vec(), $norm))
            };
        }
        let hc = 4i64;
        let hc_mix = (2 + hc) * hc;
        let d_inner = N_HEAD * HEAD_DIM_KDA;
        let qk_nope = K_MLA;

        push!("token_embd.weight", vec![N_EMBD, n_vocab], false);
        push!("output_norm.weight", vec![N_EMBD], true);
        push!("output.weight", vec![N_EMBD, n_vocab], false);

        for i in 0..(N_LAYER + 1) as i64 {
            push!(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], true);
            push!(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], true);

            if i < N_LAYER as i64 {
                push!(format!("blk.{i}.hc_attn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
                push!(format!("blk.{i}.hc_attn_base.weight"), vec![hc_mix], false);
                push!(format!("blk.{i}.hc_attn_scale.weight"), vec![3], false);
                push!(format!("blk.{i}.hc_ffn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
                push!(format!("blk.{i}.hc_ffn_base.weight"), vec![hc_mix], false);
                push!(format!("blk.{i}.hc_ffn_scale.weight"), vec![3], false);
            }

            if i == 0 {
                push!(format!("blk.{i}.ssm_conv1d_q.weight"), vec![D_CONV, 1, d_inner, 1], false);
                push!(format!("blk.{i}.ssm_conv1d_k.weight"), vec![D_CONV, 1, d_inner, 1], false);
                push!(format!("blk.{i}.ssm_conv1d_v.weight"), vec![D_CONV, 1, d_inner, 1], false);
                push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, d_inner], false);
                push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, d_inner], false);
                push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, d_inner], false);
                push!(format!("blk.{i}.ssm_f_a.weight"), vec![N_EMBD, HEAD_DIM_KDA], false);
                push!(format!("blk.{i}.ssm_f_b.weight"), vec![HEAD_DIM_KDA, d_inner], false);
                push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, N_HEAD], false);
                push!(format!("blk.{i}.ssm_a"), vec![N_HEAD], false);
                push!(format!("blk.{i}.ssm_dt.bias"), vec![d_inner], false);
                push!(format!("blk.{i}.ssm_g_a.weight"), vec![N_EMBD, HEAD_DIM_KDA], false);
                push!(format!("blk.{i}.ssm_g_b.weight"), vec![HEAD_DIM_KDA, d_inner], false);
                push!(format!("blk.{i}.ssm_norm.weight"), vec![HEAD_DIM_KDA], true);
                push!(format!("blk.{i}.attn_output.weight"), vec![d_inner, N_EMBD], false);
            } else {
                push!(format!("blk.{i}.attn_q_a_norm.weight"), vec![Q_LORA], true);
                push!(format!("blk.{i}.attn_kv_a_norm.weight"), vec![KV_LORA], true);
                push!(format!("blk.{i}.attn_q_a.weight"), vec![N_EMBD, Q_LORA], false);
                push!(format!("blk.{i}.attn_q_b.weight"), vec![Q_LORA, N_HEAD * K_MLA], false);
                push!(format!("blk.{i}.attn_kv_a_mqa.weight"), vec![N_EMBD, KV_LORA], false);
                push!(format!("blk.{i}.attn_k_b.weight"), vec![qk_nope, KV_LORA, N_HEAD], false);
                push!(format!("blk.{i}.attn_v_b.weight"), vec![KV_LORA, V_MLA, N_HEAD], false);
                push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * V_MLA, N_EMBD], false);

                let full = i != 2;
                if full {
                    push!(format!("blk.{i}.indexer.k_norm.weight"), vec![INDEXER_HEAD_SIZE], true);
                    push!(format!("blk.{i}.indexer.k_norm.bias"), vec![INDEXER_HEAD_SIZE], true);
                    push!(format!("blk.{i}.indexer.proj.weight"), vec![N_EMBD, INDEXER_HEAD], false);
                    push!(format!("blk.{i}.indexer.attn_k.weight"), vec![N_EMBD, INDEXER_HEAD_SIZE], false);
                    push!(format!("blk.{i}.indexer.attn_q_b.weight"), vec![Q_LORA, INDEXER_HEAD * INDEXER_HEAD_SIZE], false);
                    push!(format!("blk.{i}.indexer_compressor_gate.weight"), vec![N_EMBD, INDEXER_HEAD_SIZE], false);
                    push!(format!("blk.{i}.indexer_compressor_ape.weight"), vec![INDEXER_HEAD_SIZE, KPOOL], false);
                }
            }

            push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], false);
            push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], true);
            push!(format!("blk.{i}.ffn_gate_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
            push!(format!("blk.{i}.ffn_down_exps.weight"), vec![N_FF_EXP, N_EMBD, N_EXPERT], false);
            push!(format!("blk.{i}.ffn_up_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
            push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);
            push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_EXP * N_EXPERT_SHARED, N_EMBD], false);
            push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);

            if i >= N_LAYER as i64 {
                push!(format!("blk.{i}.nextn.eh_proj.weight"), vec![2 * N_EMBD, N_EMBD], false);
                push!(format!("blk.{i}.nextn.enorm.weight"), vec![N_EMBD], true);
                push!(format!("blk.{i}.nextn.hnorm.weight"), vec![N_EMBD], true);
            }
        }
        v
    }

    /// the writer of tests/glm5_e2e.rs (deterministic, seeded) — byte-equal
    /// files on both sides of the acceptance
    pub fn build_file(path: &str) {
        use ggml::gguf_write::GgufWriter;
        use ggml::types::GgmlType;
        use ggml::{Gguf, GgufType, Value};
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())
            .expect("mkdir synth dir");

        let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
        let n_vocab = src
            .kv
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.tokens")
            .and_then(|(_, v)| v.as_array().map(|a| a.1.len()))
            .expect("tokens array") as i64;

        let mut w = GgufWriter::new(32);
        for (k, val) in &src.kv {
            if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
                w.set_kv(k, val.clone());
            }
        }

        let a = "glm5-next";
        macro_rules! kv {
            ($k:expr, $v:expr) => {
                w.set_kv(&$k, $v)
            };
        }
        kv!("general.architecture", Value::String(a.to_string()));
        kv!("general.name", Value::String("llama-rust-synth-glm5-next".to_string()));
        kv!("general.file_type", Value::U32(0));
        kv!(format!("{a}.context_length"), Value::U32(N_CTX));
        kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
        kv!(format!("{a}.block_count"), Value::U32((N_LAYER + 1) as u32));
        kv!(format!("{a}.nextn_predict_layers"), Value::U32(1));
        kv!(format!("{a}.feed_forward_length"), Value::U32(32));
        kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
        kv!(
            format!("{a}.attention.head_count_kv"),
            Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(1), Value::U32(1), Value::U32(1)])
        );
        kv!(format!("{a}.attention.key_length"), Value::U32((KV_LORA) as u32));
        kv!(format!("{a}.attention.value_length"), Value::U32(V_MLA as u32));
        kv!(format!("{a}.attention.key_length_mla"), Value::U32(K_MLA as u32));
        kv!(format!("{a}.attention.value_length_mla"), Value::U32(V_MLA as u32));
        kv!(format!("{a}.attention.kv_lora_rank"), Value::U32(KV_LORA as u32));
        kv!(format!("{a}.attention.q_lora_rank"), Value::U32(Q_LORA as u32));
        kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
        kv!(format!("{a}.rope.dimension_count"), Value::U32(0));
        kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

        kv!(format!("{a}.ssm.conv_kernel"), Value::U32(D_CONV as u32));
        kv!(format!("{a}.kda.head_dim"), Value::U32(HEAD_DIM_KDA as u32));

        kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
        kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT as u32));
        kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
        kv!(format!("{a}.expert_shared_count"), Value::U32(N_EXPERT_SHARED as u32));

        kv!(format!("{a}.attention.indexer.head_count"), Value::U32(INDEXER_HEAD as u32));
        kv!(format!("{a}.attention.indexer.key_length"), Value::U32(INDEXER_HEAD_SIZE as u32));
        kv!(format!("{a}.attention.indexer.top_k"), Value::U32(TOP_K as u32));
        kv!(format!("{a}.attention.indexer.kpool"), Value::U32(KPOOL as u32));
        kv!(
            format!("{a}.attention.indexer.types"),
            Value::Array(GgufType::Uint32, vec![Value::U32(1), Value::U32(1), Value::U32(0)])
        );

        kv!(format!("{a}.hyper_connection.count"), Value::U32(4));
        kv!(format!("{a}.hyper_connection.sinkhorn_iterations"), Value::U32(1));
        kv!(format!("{a}.hyper_connection.epsilon"), Value::F32(1e-4));

        let table = tensors_for(n_vocab);
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut rng = Rng(0x915_5e17);
        for (name, ne, is_norm) in &table {
            let n: i64 = ne.iter().product();
            let vals: Vec<f32> = if *is_norm {
                (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
            } else {
                let s = 1.0 / (N_EMBD as f32).sqrt();
                (0..n).map(|_| s * rng.next()).collect()
            };
            let ne4 = [ne[0], *ne.get(1).unwrap_or(&1), *ne.get(2).unwrap_or(&1), *ne.get(3).unwrap_or(&1)];
            w.add_tensor(name, GgmlType::F32, ne4);
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for x in &vals {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            data.push(bytes);
        }

        let f = std::fs::File::create(path).expect("create synth gguf");
        let mut bw = std::io::BufWriter::new(f);
        let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
        w.write(&mut bw, &refs).expect("write synth gguf");
        use std::io::Write as _;
        bw.flush().unwrap();
        drop(bw);
    }
}

fn read_nodes(
    d: &[u8],
) -> (Vec<i32>, Vec<(String, String, [i64; 4], Option<Vec<u8>>)>) {
    assert_eq!(&d[..8], b"DECDMP1\0");
    let mut off = 8usize;
    let nt = u32::from_le_bytes(d[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let mut toks = vec![0i32; nt];
    for t in toks.iter_mut() {
        *t = i32::from_le_bytes(d[off..off + 4].try_into().unwrap());
        off += 4;
    }
    let nn = u32::from_le_bytes(d[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let mut out = Vec::with_capacity(nn);
    let mut get_str = |off: &mut usize| -> String {
        let l = d[*off] as usize;
        *off += 1;
        let t = String::from_utf8_lossy(&d[*off..*off + l]).into_owned();
        *off += l;
        t
    };
    for _ in 0..nn {
        let op = get_str(&mut off);
        let name = get_str(&mut off);
        let _ty = get_str(&mut off);
        let mut ne = [0i64; 4];
        for (k, v) in ne.iter_mut().enumerate() {
            let b: [u8; 8] = d[off + k * 8..off + k * 8 + 8].try_into().unwrap();
            *v = i64::from_le_bytes(b);
        }
        off += 32;
        let nel = u64::from_le_bytes(d[off..off + 8].try_into().unwrap());
        off += 8;
        let payload = if nel <= (1 << 19) {
            let p = d[off..off + 4 * nel as usize].to_vec();
            off += 4 * nel as usize;
            Some(p)
        } else {
            None
        };
        out.push((op, name, ne, payload));
    }
    (toks, out)
}

/// run the dump driver in-process over a freshly written synth file and
/// return the raw stream bytes
fn run_port_stream(synth: &str) -> Vec<u8> {
    let file = std::fs::File::open(synth).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("hi", true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();

    let (w, attn) = forward_of(&model);
    let gctx: Context = model.ctx;
    let mut dctx = DecodeContext::new_with(gctx, w, attn, 512, 8, 2048)
        .with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);

    DUMP.get_or_init(|| Mutex::new(Some(DumpState { out: Vec::new(), nodes: 0 })));
    {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        *guard = Some(DumpState { out: Vec::new(), nodes: 0 });
    }
    set_eval_callback(Some(dump_cb));
    ACTIVE.store(1, Ordering::SeqCst);
    let _ = dctx.decode_embed(&ids, &pos).expect("decode_embed");
    for s in 0..12 {
        dctx.decode(&[100i32], &[(ids.len() + s) as i32]).expect("decode tail");
    }
    ACTIVE.store(0, Ordering::SeqCst);
    set_eval_callback(None);
    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    let mut buf = Vec::with_capacity(8 + 4 + 4 * ids.len() + 4 + body.len());
    buf.extend_from_slice(b"DECDMP1\0");
    buf.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &id in &ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    buf.extend_from_slice(&nodes.to_le_bytes());
    buf.extend_from_slice(&body);
    buf
}

/// the batch-19 graph acceptance: the kda/dsa/kpool node stream over the
/// 13 graph builds must be bit-identical to the NEW reference's
#[test]
fn glm5_graph_nodes_bit_exact_vs_reference() {
    let ref_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/glm5/nodes_ref.bin");
    let Some(ref_bytes) = std::fs::read(ref_path).ok() else {
        panic!(
            "missing parity/glm5/nodes_ref.bin — regenerate with parity/glm5_parity.sh \
             (ref_decode_dump.c --fa off --decode-tail 12 --text hi over the NEW reference)"
        );
    };
    let synth = format!("/tmp/hq-glm5-accept-{}.gguf", std::process::id());
    synth::build_file(&synth);
    let port_bytes = run_port_stream(&synth);
    std::fs::remove_file(&synth).ok();

    let (rt, rnodes) = read_nodes(&ref_bytes);
    let (pt, pnodes) = read_nodes(&port_bytes);
    assert_eq!(rt, pt, "the tokenized prefill must match");
    assert_eq!(rnodes.len() > 5000, true, "reference stream present");

    // named-node pairing: every node carrying a graph name on the port side
    // must exist on the reference side with the byte-identical payload
    let mut ridx: std::collections::HashMap<String, Vec<usize>> = Default::default();
    for (i, (_, name, _, _)) in rnodes.iter().enumerate() {
        if !name.is_empty() {
            ridx.entry(name.clone()).or_default().push(i);
        }
    }
    // names repeat across the 13 graph builds — pair the Nth occurrence on
    // the port side with the Nth occurrence on the reference side
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    let mut compared = 0usize;
    for (_, name, ne, pl) in pnodes.iter() {
        // only real graph names — ggml's auto names (" (view)", "(reshaped)",
        // "node_N") are shared across unrelated nodes
        if name.is_empty()
            || name.starts_with(' ')
            || name.starts_with("node_")
            || name.contains(" (")
            || pl.is_none()
        {
            // ggml's derived names ("x (view)" of a named tensor) pair with
            // their parent's value region — the pure-named nodes cover them
            continue;
        }
        // the embeddings driver renames the final norm on the reference
        // side (build_pooling's cb "result_embd_pooled", the C's t_embd tap)
        let ref_name = if name == "result_norm" {
            "result_embd_pooled"
        } else {
            name.as_str()
        };
        let occ = seen.entry(name.clone()).or_insert(0);
        let Some(ixs) = ridx.get(ref_name) else {
            panic!("node {name:?} (ne {ne:?}) has no reference counterpart");
        };
        assert!(
            *occ < ixs.len(),
            "node {name:?}: more port occurrences ({}) than reference ({})",
            *occ + 1,
            ixs.len()
        );
        let rnode = &rnodes[ixs[*occ]];
        *occ += 1;
        assert_eq!(
            rnode.2, *ne,
            "node {name:?} occurrence {} shape",
            *occ - 1
        );
        assert_eq!(
            rnode.3.as_ref().map(|p| p.len()),
            Some(pl.as_ref().unwrap().len()),
            "node {name:?} payload length"
        );
        assert_eq!(
            rnode.3.as_ref().unwrap(),
            pl.as_ref().unwrap(),
            "node {name:?} occurrence {} payload bits",
            *occ - 1
        );
        compared += 1;
    }
    // names present on BOTH sides must occur equally often (the reference
    // names some intermediates the port leaves unnamed — ffn_gate/ffn_up of
    // build_ffn — those are covered by the index-paired sequence compare of
    // parity/decode_dump_cmp.py in glm5_parity.sh instead)
    for (name, ixs) in &ridx {
        if name.starts_with(' ') || name.starts_with("node_") || name.is_empty() || name.contains(" (") {
            continue;
        }
        let port_name = if name == "result_embd_pooled" {
            "result_norm"
        } else {
            name.as_str()
        };
        if let Some(got) = seen.get(port_name) {
            assert_eq!(*got, ixs.len(), "node {name:?} occurrence count");
        }
    }
    assert!(compared >= 70, "expected the kpool family's named nodes, got {compared}");
}
