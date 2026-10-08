//! qwen4exp_qsa_dump.rs — batch 19: the un-ported halves of qwen4exp (the
//! QSA causal_attn path + the PLE n-gram module, qwen4exp.cpp:495-1310),
//! exercised over a synthetic GGUF that turns both on
//! (`attention.compress_ratios = [0,0,4,0]` + a PLE layer-1 module).
//!
//! The writer is the batch-11a qwen4exp family's shape with the two key
//! groups added; the DECDMP1 node-stream acceptance mirrors glm5_dump.rs
//! (ref_decode_dump.c `--fa off --decode-tail 12 --text hi` over the NEW
//! reference def4d406a; parity/glm5_parity.sh documents the recipe).
//!
//! Run (release):
//!   cargo test --release -p llama --test qwen4exp_qsa_dump -- --ignored --nocapture
//!   (the default acceptance test regenerates the file + compares inline)

use std::io::Write as _;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::arch::LlmArch;
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_LAYER: i64 = 4;
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const KLEN: i64 = 16;
const HEAD_KV: i64 = 2;
const HC: i64 = 2;
const HC_LR: i64 = 8;
const N_FF_EXP: i64 = 32;
const N_FF_SH: i64 = 32;
const N_EXPERT: i64 = 4;
const IDX_HEADS: i64 = 2;
const IDX_DIM: i64 = 16;
const IDX_TOPK: i64 = 8;
/// the QSA layer (2) compresses blocks of 4 tokens
const RATIO: i64 = 4;
/// the PLE geometry: layer 1 (recurrent), 3-gram, 2 heads/gram (4 heads),
/// kernel 3, table rows [16, 256]. ple_head_dim * ple_n_heads MUST equal
/// n_embd — build_ple's `ple_key @ emb` feeds the [ple_head_dim*n_heads]
/// gather output into an {n_embd, ...} weight (qwen4exp.cpp:1230)
const PLE_NGRAM: i64 = 3;
const PLE_PER_GRAM: i64 = 2;
const PLE_KERN: i64 = 3;
const PLE_HEAD_DIM: i64 = 16;
const PLE_ROWS: i64 = 256;

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
// the synthetic file — the batch-11a qwen4exp family + QSA + PLE
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn tensors_for(n_vocab: i64) -> Vec<(String, Vec<i64>, f32)> {
    let mut v: Vec<(String, Vec<i64>, f32)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $scale:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $scale))
        };
    }
    let hc = HC;
    let hc_dim = hc * N_EMBD;
    let (head_k, n_k, head_v, n_v) = (8i64, 1i64, 8i64, 2i64);
    let value_dim = head_v * n_v;
    let conv_dim = head_k * n_k * 2 + value_dim;

    push!("token_embd.weight", vec![N_EMBD, n_vocab], 0.11);
    push!("output_hc_norm.weight", vec![N_EMBD, hc], 0.2);
    push!("output_hc_down.weight", vec![hc_dim, HC_LR], 0.11);
    push!("output_hc_up.weight", vec![HC_LR, hc_dim], 0.11);
    push!("output.weight", vec![N_EMBD, n_vocab], 0.11);
    // the flat [ple_head_dim, ple_rows] n-gram gather table
    push!("per_layer_token_embd.weight", vec![PLE_HEAD_DIM, PLE_ROWS], 0.11);

    for i in 0..N_LAYER {
        let recr = i != 2;
        push!(format!("blk.{i}.hc_attn_norm.weight"), vec![N_EMBD, hc], 0.2);
        push!(format!("blk.{i}.hc_attn_down.weight"), vec![hc_dim, HC_LR], 0.11);
        push!(format!("blk.{i}.hc_attn_up.weight"), vec![HC_LR, hc_dim], 0.11);
        push!(format!("blk.{i}.hc_attn_inject.weight"), vec![hc_dim, hc], 0.2);
        push!(format!("blk.{i}.hc_ffn_norm.weight"), vec![N_EMBD, hc], 0.2);
        push!(format!("blk.{i}.hc_ffn_down.weight"), vec![hc_dim, HC_LR], 0.11);
        push!(format!("blk.{i}.hc_ffn_up.weight"), vec![HC_LR, hc_dim], 0.11);
        push!(format!("blk.{i}.hc_ffn_inject.weight"), vec![hc_dim, hc], 0.2);

        if !recr {
            let q = N_HEAD * KLEN * 2; // [q|gate]
            let kv = HEAD_KV * KLEN;
            push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, q], 0.11);
            push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * KLEN, N_EMBD], 0.11);
            push!(format!("blk.{i}.attn_q_norm.weight"), vec![KLEN], 0.2);
            push!(format!("blk.{i}.attn_k_norm.weight"), vec![KLEN], 0.2);
            push!(format!("blk.{i}.indexer.q_proj.weight"), vec![N_EMBD, IDX_HEADS * IDX_DIM], 0.11);
            push!(format!("blk.{i}.indexer.k_proj.weight"), vec![N_EMBD, IDX_DIM], 0.11);
            push!(format!("blk.{i}.indexer.q_norm.weight"), vec![IDX_DIM], 0.2);
            push!(format!("blk.{i}.indexer.k_norm.weight"), vec![IDX_DIM], 0.2);
        } else {
            push!(format!("blk.{i}.attn_qkv.weight"), vec![N_EMBD, conv_dim], 0.11);
            push!(format!("blk.{i}.attn_gate.weight"), vec![N_EMBD, value_dim], 0.11);
            push!(format!("blk.{i}.ssm_conv1d.weight"), vec![4, conv_dim], 0.2);
            push!(format!("blk.{i}.ssm_dt.bias"), vec![n_v], 0.2);
            push!(format!("blk.{i}.ssm_a"), vec![n_v], 0.3);
            push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, n_v], 0.2);
            push!(format!("blk.{i}.ssm_alpha.weight"), vec![N_EMBD, n_v], 0.2);
            push!(format!("blk.{i}.ssm_norm.weight"), vec![head_v], 0.2);
            push!(format!("blk.{i}.ssm_out.weight"), vec![value_dim, N_EMBD], 0.11);
        }

        // the PLE module of layer 1 (qwen4exp.cpp:244-251)
        if i == 1 {
            push!(format!("blk.{i}.ple_key.weight"), vec![N_EMBD, hc_dim], 0.11);
            push!(format!("blk.{i}.ple_value.weight"), vec![N_EMBD, N_EMBD], 0.11);
            push!(format!("blk.{i}.ple_norm_key.weight"), vec![N_EMBD, hc], 0.2);
            push!(format!("blk.{i}.ple_norm_query.weight"), vec![N_EMBD, hc], 0.2);
            push!(format!("blk.{i}.ple_norm_conv.weight"), vec![N_EMBD, hc], 0.2);
            push!(format!("blk.{i}.ple_conv1d.weight"), vec![PLE_KERN, hc_dim], 0.2);
        }

        push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_down_exps.weight"), vec![N_FF_EXP, N_EMBD, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_gate_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_up_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_gate_inp_shexp.weight"), vec![N_EMBD], 0.11);
        push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![N_EMBD, N_FF_SH], 0.11);
        push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![N_EMBD, N_FF_SH], 0.11);
        push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SH, N_EMBD], 0.11);
    }
    v
}

pub fn build_file(path: &str) {
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

    let a = "qwen4exp";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-qwen4exp-qsa-ple".to_string()));
    kv!("general.file_type", Value::U32(0));
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::Array(GgufType::Uint32, vec![Value::U32(2); N_LAYER as usize])
    );
    kv!(format!("{a}.attention.key_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(
        format!("{a}.rope.dimension_sections"),
        Value::Array(GgufType::Int32, vec![Value::I32(8), Value::I32(4), Value::I32(4), Value::I32(0)])
    );
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
    kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(32));
    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
    kv!(format!("{a}.ssm.inner_size"), Value::U32(16));
    kv!(format!("{a}.ssm.state_size"), Value::U32(8));
    kv!(format!("{a}.ssm.time_step_rank"), Value::U32(2));
    kv!(format!("{a}.ssm.group_count"), Value::U32(1));
    kv!(format!("{a}.hyper_connection.count"), Value::U32(HC as u32));
    kv!(format!("{a}.hyper_connection.low_rank"), Value::U32(HC_LR as u32));
    kv!(format!("{a}.attention.indexer.head_count"), Value::U32(IDX_HEADS as u32));
    kv!(format!("{a}.attention.indexer.key_length"), Value::U32(IDX_DIM as u32));
    kv!(format!("{a}.attention.indexer.top_k"), Value::U32(IDX_TOPK as u32));
    // the QSA half: layer 2 compresses blocks of RATIO tokens
    kv!(
        format!("{a}.attention.compress_ratios"),
        Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(0), Value::U32(RATIO as u32), Value::U32(0)])
    );
    kv!(
        format!("{a}.attention.recurrent_layers"),
        Value::Array(GgufType::Uint32, vec![Value::U32(1), Value::U32(1), Value::U32(0), Value::U32(1)])
    );
    // the PLE half (layer 1)
    kv!(format!("{a}.ple.layers"), Value::Array(GgufType::Uint32, vec![Value::U32(1)]));
    kv!(format!("{a}.ple.ngram_size"), Value::U32(PLE_NGRAM as u32));
    kv!(format!("{a}.ple.heads_per_ngram"), Value::U32(PLE_PER_GRAM as u32));
    kv!(format!("{a}.ple.conv_kernel"), Value::U32(PLE_KERN as u32));
    kv!(format!("{a}.ple.eos_token_id"), Value::U32(2));
    kv!(format!("{a}.embedding_length_per_layer_input"), Value::U32(PLE_HEAD_DIM as u32));
    kv!(
        format!("{a}.ple.layer_multipliers"),
        Value::Array(GgufType::Uint64, vec![Value::U64(1000003), Value::U64(1000033), Value::U64(1000037)])
    );
    kv!(
        format!("{a}.ple.head_offsets"),
        Value::Array(
            GgufType::Uint64,
            vec![Value::U64(0), Value::U64(64), Value::U64(128), Value::U64(192)]
        )
    );
    kv!(
        format!("{a}.ple.head_vocab_sizes"),
        Value::Array(GgufType::Uint64, vec![Value::U64(64); 4])
    );

    let table = tensors_for(n_vocab);
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ 7);
    for (name, ne, scale) in &table {
        let n: i64 = ne.iter().product();
        let vals: Vec<f32> = (0..n).map(|_| rng.next() * scale).collect();
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
    bw.flush().unwrap();
    drop(bw);
}

// ---------------------------------------------------------------------------
// the port-side weights bundle + driver
// ---------------------------------------------------------------------------

fn forward_of(m: &LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let rope = hp.rope_runtime();
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
        use_flash_attn: fa,
    };
    let weights = llama::graph_arch::Qwen4ExpModelWeights {
        tok_embd: m.tok_embd,
        hc_head_norm: m.output_norm,
        hc_head_down: m.hc_head_down.expect("hc_head_down"),
        hc_head_up: m.hc_head_up.expect("hc_head_up"),
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(_il, l)| llama::graph_arch::Qwen4ExpLayerWeights {
                hc_attn_norm: l.hc_attn_norm.expect("hc_attn_norm"),
                hc_attn_down: l.hc_attn_down.expect("hc_attn_down"),
                hc_attn_up: l.hc_attn_up.expect("hc_attn_up"),
                hc_attn_inject: l.hc_attn_inject.expect("hc_attn_inject"),
                hc_ffn_norm: l.hc_ffn_norm.expect("hc_ffn_norm"),
                hc_ffn_down: l.hc_ffn_down.expect("hc_ffn_down"),
                hc_ffn_up: l.hc_ffn_up.expect("hc_ffn_up"),
                hc_ffn_inject: l.hc_ffn_inject.expect("hc_ffn_inject"),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                index_q_proj: l.index_q_proj,
                index_k_proj: l.index_k_proj,
                index_q_norm: l.index_q_norm,
                index_k_norm: l.index_k_norm,
                ple_key: l.ple_key,
                ple_value: l.ple_value,
                ple_norm_key: l.ple_norm_key,
                ple_norm_query: l.ple_norm_query,
                ple_norm_conv: l.ple_norm_conv,
                ple_conv1d: l.ple_conv1d,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    };
    (ForwardWeights::Qwen4Exp(weights, qwen4exp_params_of(hp, n_trunk, attn.clone())), attn)
}

fn qwen4exp_params_of(
    hp: &llama::hparams::LlamaHparams,
    n_trunk: usize,
    attn: AttnParams,
) -> llama::graph_arch::Qwen4ExpParams {
    llama::graph_arch::Qwen4ExpParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_trunk).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_trunk).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_trunk).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_trunk).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        hc: hp.dsv4_hc_mult as i64,
        hc_lr: hp.hc_low_rank as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        compress_ratios: hp.dsv4_compress_ratios[..n_trunk].to_vec(),
        is_ple: (0..n_trunk).map(|il| hp.is_ple(il)).collect(),
        ple_ngram_size: hp.ple_ngram_size as i64,
        ple_heads_per_ngram: hp.ple_heads_per_ngram as i64,
        ple_conv_kernel: hp.ple_conv_kernel as i64,
        ple_n_heads: hp.ple_n_heads as i64,
        ple_head_dim: hp.ple_head_dim as i64,
        ple_eos_token_id: hp.ple_eos_token_id as i64,
        ple_image_token_id: hp.ple_image_token_id as i64,
        ple_layer_multipliers: hp.ple_layer_multipliers,
        ple_head_offsets: hp.ple_head_offsets,
        ple_head_vocab_sizes: hp.ple_head_vocab_sizes,
    }
}

/// write the synth to the fixed-name path the parity script drives
#[test]
#[ignore = "writer only — the file lands in /tmp for the reference probe"]
fn qwen4exp_qsa_write_synth_file() {
    let path = "/tmp/hq-q4e/qwen4exp-qsa-ple.gguf";
    build_file(path);
    println!("qwen4exp qsa/ple synth -> {path}");
}

/// Stream every node of one `decode_embed` prefill (+ N decode tail steps)
/// to Q4E_DUMP_OUT — the C probe's `--decode-tail` mirror. Q4E_FA=1 flips
/// `AttnParams::use_flash_attn` (batch 20: the QSA FA arm's F16 kq_mask
/// route, mirrored against the probe's `--fa on` arm).
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the qsa/ple parity file"]
fn qwen4exp_qsa_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let model_path = std::env::var("Q4E_DUMP_MODEL")
        .unwrap_or_else(|_| "/tmp/hq-q4e/qwen4exp-qsa-ple.gguf".to_string());
    let out_path = std::env::var("Q4E_DUMP_OUT")
        .unwrap_or_else(|_| "/tmp/hq-q4e-port.bin".to_string());
    let n_tail: usize = std::env::var("Q4E_DECODE_TAIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let fa = std::env::var("Q4E_FA").map(|v| v == "1").unwrap_or(false);

    let file = std::fs::File::open(&model_path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");
    assert_eq!(model.arch, LlmArch::QWEN4EXP);

    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("hi", true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!("tokens: {ids:?}");

    let (w, attn) = forward_of(&model, fa);
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
    for s in 0..n_tail {
        dctx.decode(&[100i32], &[(ids.len() + s) as i32]).expect("decode tail");
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
    println!("qwen4exp_qsa_dump: {nodes} nodes -> {out_path}");
}

// ---------------------------------------------------------------------------
// the batch-19 acceptance (a DEFAULT test): regenerate the synth, drive the
// same 13 graph builds, bit-compare the named nodes against the stored
// reference stream (parity/qwen4exp_qsa_nodes_ref.bin — the no-embeddings
// probe variant over the NEW reference; the reference's own embeddings
// output extraction aborts on the QSA graph, see PARITY.md 批次 19).
// ---------------------------------------------------------------------------

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

/// run the dump driver in-process over a freshly written synth file
fn run_port_stream(synth: &str, fa: bool) -> Vec<u8> {
    let file = std::fs::File::open(synth).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let model = load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = Vocab::load(&gguf).expect("vocab");
    let ids = vocab.tokenize("hi", true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();

    let (w, attn) = forward_of(&model, fa);
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

/// the QSA/PLE acceptance: named graph nodes over the 13 builds must be
/// bit-identical to the NEW reference's (the no-embeddings probe arm)
#[test]
fn qwen4exp_qsa_ple_nodes_bit_exact_vs_reference() {
    // the static DUMP/ACTIVE globals serialize all dump-driven tests
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ref_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../parity/qwen4exp_qsa_nodes_ref.bin"
    );
    let Some(ref_bytes) = std::fs::read(ref_path).ok() else {
        panic!(
            "missing parity/qwen4exp_qsa_nodes_ref.bin — regenerate with the \
             no-embeddings probe variant (parity/ref_decode_dump_noembd.c) over \
             the NEW reference"
        );
    };
    let synth = format!("/tmp/hq-q4e-accept-{}.gguf", std::process::id());
    build_file(&synth);
    let port_bytes = run_port_stream(&synth, false);
    std::fs::remove_file(&synth).ok();

    let compared = compare_named_nodes(&ref_bytes, &port_bytes);
    assert!(
        compared >= 35,
        "expected the QSA/PLE family's named nodes, got {compared}"
    );
}

/// batch 20 — the QSA **FA arm** acceptance: the same 13 builds with
/// `AttnParams::use_flash_attn = true` (llama-graph.cpp:2633-2669's branch,
/// entered via the F16 kq_mask of llama-graph.cpp:38-39) must stay
/// bit-identical to the NEW reference's `--fa on` probe arm
/// (parity/qwen4exp_qsa_nodes_fa_ref.bin).
#[test]
fn qwen4exp_qsa_fa_nodes_bit_exact_vs_reference() {
    // the static DUMP/ACTIVE globals serialize all dump-driven tests
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ref_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../parity/qwen4exp_qsa_nodes_fa_ref.bin"
    );
    let Some(ref_bytes) = std::fs::read(ref_path).ok() else {
        panic!(
            "missing parity/qwen4exp_qsa_nodes_fa_ref.bin — regenerate with \
             the no-embeddings probe variant (--no-embd --fa on) over the NEW \
             reference"
        );
    };
    let synth = format!("/tmp/hq-q4e-fa-accept-{}.gguf", std::process::id());
    build_file(&synth);
    let port_bytes = run_port_stream(&synth, true);
    std::fs::remove_file(&synth).ok();

    let compared = compare_named_nodes(&ref_bytes, &port_bytes);
    assert!(
        compared >= 35,
        "expected the QSA/PLE FA family's named nodes, got {compared}"
    );
}

/// the shared named-node comparator of the two acceptance tests above
fn compare_named_nodes(ref_bytes: &[u8], port_bytes: &[u8]) -> usize {
    let (rt, rnodes) = read_nodes(ref_bytes);
    let (pt, pnodes) = read_nodes(port_bytes);
    assert_eq!(rt, pt, "the tokenized prefill must match");

    let mut ridx: std::collections::HashMap<String, Vec<usize>> = Default::default();
    for (i, (_, name, _, _)) in rnodes.iter().enumerate() {
        if !name.is_empty() {
            ridx.entry(name.clone()).or_default().push(i);
        }
    }
    let mut seen: std::collections::HashMap<String, usize> = Default::default();
    let mut compared = 0usize;
    for (_, name, ne, pl) in pnodes.iter() {
        if name.is_empty()
            || name.starts_with(' ')
            || name.starts_with("node_")
            || name.contains(" (")
            || pl.is_none()
        {
            continue;
        }
        // the C's cb() renders il = -1 names plain (`ggml_set_name(name)`,
        // llama-graph.cpp's cb macro) — the port's format keeps "--1"
        let ref_name = if let Some(stem) = name.strip_suffix("--1") {
            std::borrow::Cow::Owned(stem.to_string())
        } else {
            std::borrow::Cow::Borrowed(name.as_str())
        };
        let occ = seen.entry(name.clone()).or_insert(0);
        let Some(ixs) = ridx.get(ref_name.as_ref()) else {
            panic!("node {name:?} (ne {ne:?}) has no reference counterpart");
        };
        assert!(*occ < ixs.len(), "node {name:?}: more port occurrences than reference");
        let rnode = &rnodes[ixs[*occ]];
        *occ += 1;
        assert_eq!(rnode.2, *ne, "node {name:?} shape");
        assert_eq!(
            rnode.3.as_ref().unwrap(),
            pl.as_ref().unwrap(),
            "node {name:?} occurrence {} payload bits",
            *occ - 1
        );
        compared += 1;
    }
    compared
}
