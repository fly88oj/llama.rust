//! lfm2_decision_e2e.rs — the LiquidAI d1 / d1-omni decision models
//! (88dcc460d + a657f7e98, c35b66744): the non-causal lfm2 trunk without
//! memory (shortconv + no-cache attention) followed by the 3-question-type
//! decision head (choice / score / noul).
//!
//! The synthetic file: 4 trunk blocks (2 shortconv + 2 attention) + 1 head
//! block, `attention.causal = false`, `decision.block_count = 1`, one token
//! type per question type. The port drives
//! `graph_arch::build_lfm2_decision_forward` directly (the clef.rs /
//! tts-batch convention — no EncoderWeights/CLI routing; the null-memory
//! decode→encode reroute of the reference is a model.rs/B contract item).
//!
//! Acceptance: the sanity cell (load + hparams asserts + finite [3, T]
//! scores), the mask-rule unit cell (the d1-omni media/conv rules pinned
//! against the C's set_input bodies), and the named-node stream bit-compare
//! vs `parity/gen_lfm2_decision_ref.sh` (ref_decode_dump --fa off; decode
//! reroutes the null-memory arch to encode, lfm2.cpp:135-139).

use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::graph_arch::{self, EurobertRope};

const VOCAB_SPM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf"
);
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/synca/lfm2d";
const FILE: &str = "lfm2-decision-synth.gguf";

const N_EMBD: i64 = 128;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HD: i64 = 32;
const N_FF: i64 = 64;
const N_LAYER_TRUNK: usize = 4; // layers 0-1 shortconv, 2-3 attention
const N_LAYER: usize = N_LAYER_TRUNK + 1; // + 1 decision head block
const N_DECISION: u32 = 1;
const PROMPT: &str = "The capital of France is";

// ---------------------------------------------------------------------------
// the writer
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

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn tensors_of() -> Vec<(String, Vec<i64>, String)> {
    let mut t: Vec<(String, Vec<i64>, String)> = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, kind: &str| {
        t.push((name, ne, kind.to_string()));
    };

    push("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], "proj");
    push("token_embd_norm.weight".into(), vec![N_EMBD], "norm");

    // the trunk (lfm2.cpp:88-133): shortconv layers 0-1, attention 2-3
    for i in 0..N_LAYER_TRUNK as i32 {
        push(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], "proj");
        push(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], "proj");
        push(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], "proj");

        if i < 2 {
            push(format!("blk.{i}.shortconv.conv.weight"), vec![3, N_EMBD], "proj");
            push(
                format!("blk.{i}.shortconv.in_proj.weight"),
                vec![N_EMBD, 3 * N_EMBD],
                "proj",
            );
            push(
                format!("blk.{i}.shortconv.out_proj.weight"),
                vec![N_EMBD, N_EMBD],
                "proj",
            );
        } else {
            push(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, HD * N_HEAD], "proj");
            push(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, HD * N_HEAD_KV], "proj");
            push(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, HD * N_HEAD_KV], "proj");
            push(format!("blk.{i}.attn_output.weight"), vec![N_EMBD, N_EMBD], "proj");
            push(format!("blk.{i}.attn_q_norm.weight"), vec![HD], "norm");
            push(format!("blk.{i}.attn_k_norm.weight"), vec![HD], "norm");
        }
    }

    // the decision head block (lfm2.cpp:56-79)
    let i = N_LAYER_TRUNK as i32;
    push(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], "norm");
    push(format!("blk.{i}.attn_norm.bias"), vec![N_EMBD], "bias");
    push(format!("blk.{i}.attn_qkv.weight"), vec![N_EMBD, 3 * N_EMBD], "proj");
    push(format!("blk.{i}.attn_qkv.bias"), vec![3 * N_EMBD], "bias");
    push(format!("blk.{i}.attn_output.weight"), vec![N_EMBD, N_EMBD], "proj");
    push(format!("blk.{i}.attn_output.bias"), vec![N_EMBD], "bias");
    push(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], "norm");
    push(format!("blk.{i}.ffn_norm.bias"), vec![N_EMBD], "bias");
    push(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], "proj");
    push(format!("blk.{i}.ffn_up.bias"), vec![N_FF], "bias");
    push(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], "proj");
    push(format!("blk.{i}.ffn_down.bias"), vec![N_EMBD], "bias");

    // the model-level head tensors (lfm2.cpp:84-93)
    push("token_types.weight".into(), vec![N_EMBD, 3], "proj");
    push("cls.norm.weight".into(), vec![N_EMBD], "norm");
    push("cls.norm.bias".into(), vec![N_EMBD], "bias");
    push("cls.weight".into(), vec![N_EMBD, N_EMBD], "proj");
    push("cls.bias".into(), vec![N_EMBD], "bias");
    push("cls.output.weight".into(), vec![N_EMBD, 1], "proj");
    push("cls.output.bias".into(), vec![1], "bias");

    t
}

fn build_file() -> String {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/synca/lfm2d");
    let path = format!("{OUT_DIR}/{FILE}");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "lfm2";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-lfm2-d1".into()));
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    // per-layer kv heads: 0 on the shortconv layers (is_recr), 2 elsewhere
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::Array(
            ggml::GgufType::Uint32,
            (0..N_LAYER)
                .map(|i| Value::U32(if i < 2 { 0 } else { N_HEAD_KV as u32 }))
                .collect()
        )
    );
    kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    // the decision head's LayerNorm eps (lfm2.cpp:33)
    kv!(format!("{a}.attention.layer_norm_epsilon"), Value::F32(1e-6));
    // non-causal: the decision-model gate (lfm2.cpp:31)
    kv!(format!("{a}.attention.causal"), Value::Bool(false));
    kv!(format!("{a}.shortconv.l_cache"), Value::U32(3));
    kv!(format!("{a}.decision.block_count"), Value::U32(N_DECISION));
    // one token type per question type (lfm2.cpp:81-83)
    kv!("tokenizer.ggml.token_type_count", Value::U32(3));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    let mut rng = Rng(0x5eed_0000_1fd2_0003);
    let tensors = tensors_of();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (name, ne, kind) in &tensors {
        let n: usize = ne.iter().map(|&x| x as usize).product();
        let vals: Vec<f32> = if kind == "norm" {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
        } else if kind == "bias" {
            (0..n).map(|_| 0.02 * rng.next()).collect()
        } else {
            let scale = 1.0 / (N_EMBD as f32).sqrt();
            (0..n).map(|_| rng.next() * scale).collect()
        };
        let ne4 = [ne[0], *ne.get(1).unwrap_or(&1), *ne.get(2).unwrap_or(&1), 1];
        w.add_tensor(name, GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }
    let f = std::fs::File::create(&path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    bw.flush().unwrap();
    println!("wrote {path}");
    path
}

// ---------------------------------------------------------------------------
// the port-side driver
// ---------------------------------------------------------------------------

fn open_synth() -> (llama::model::LlamaModel, llama::vocab::Vocab) {
    let path = format!("{OUT_DIR}/{FILE}");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e} — run lfm2d_write_synth_file first"));
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let m = llama::model::load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
    (m, vocab)
}

fn weights_of(m: &llama::model::LlamaModel) -> graph_arch::Lfm2DecisionModelWeights {
    use graph_arch::{Lfm2DecisionHeadLayerWeights, Lfm2LayerWeights};
    let hp = &m.hparams;
    let layer = |l: &llama::model::LayerTensors| Lfm2LayerWeights {
        attn_norm: l.attn_norm.unwrap(),
        shortconv_conv: l.shortconv_conv,
        shortconv_in_proj: l.shortconv_in_proj,
        shortconv_out_proj: l.shortconv_out_proj,
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wq_b: l.wq_b,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        ffn_norm: l.ffn_norm.unwrap(),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
    };
    let _ = hp;
    graph_arch::Lfm2DecisionModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        type_embd: m.token_types.expect("token_types"),
        cls_norm: m.cls_norm.expect("cls_norm"),
        cls_norm_b: m.cls_norm_b.expect("cls_norm_b"),
        cls: m.cls.expect("cls"),
        cls_b: m.cls_b.expect("cls_b"),
        cls_out: m.cls_out.expect("cls_out"),
        cls_out_b: m.cls_out_b.expect("cls_out_b"),
        trunk_layers: m.layers[..N_LAYER_TRUNK].iter().map(layer).collect(),
        head_layers: m.layers[N_LAYER_TRUNK..N_LAYER]
            .iter()
            .map(|l| Lfm2DecisionHeadLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b.unwrap(),
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
            })
            .collect(),
    }
}

fn params_of(m: &llama::model::LlamaModel) -> graph_arch::Lfm2DecisionParams {
    let hp = &m.hparams;
    graph_arch::Lfm2DecisionParams {
        n_embd: hp.n_embd as i64,
        is_recr: (0..N_LAYER_TRUNK).map(|il| hp.is_recr(il)).collect(),
        n_shortconv_l_cache: hp.n_shortconv_l_cache as i64,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_head: (0..N_LAYER_TRUNK).map(|il| hp.n_head(il) as i64).collect(),
        n_head_kv: (0..N_LAYER_TRUNK)
            .map(|il| hp.n_head_kv(il) as i64)
            .collect(),
        n_embd_head: hp.n_embd_head_k(0) as i64,
        rope: EurobertRope {
            n_rot: hp.n_rot(0) as i32,
            rope_mode: ggml::ops::GGML_ROPE_TYPE_NEOX,
            n_ctx_orig: hp.n_ctx_train as i32,
            freq_base: hp.rope_freq_base_train,
            freq_scale: hp.rope_freq_scale_train,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: hp.yarn_beta_fast,
            beta_slow: hp.yarn_beta_slow,
        },
        f_norm_eps: hp.f_norm_eps,
        head_n_head: (N_LAYER_TRUNK..N_LAYER)
            .map(|il| hp.n_head(il) as i64)
            .collect(),
        head_n_head_kv: (N_LAYER_TRUNK..N_LAYER)
            .map(|il| hp.n_head_kv(il) as i64)
            .collect(),
        n_layer_decision: N_DECISION as usize,
    }
}

struct D1Driver {
    gctx: Context,
    w: graph_arch::Lfm2DecisionModelWeights,
    p: graph_arch::Lfm2DecisionParams,
}

impl D1Driver {
    fn new(m: &mut llama::model::LlamaModel) -> Self {
        let w = weights_of(m);
        let p = params_of(m);
        let gctx = std::mem::replace(&mut m.ctx, Context::new());
        D1Driver { gctx, w, p }
    }

    /// one decision pass — the text-only mask shapes (single sequence, no
    /// media: everything visible, plain neighbor taps)
    fn decide(&mut self, tokens: &[i32]) -> Vec<f32> {
        let n = tokens.len();
        let t = n as i64;
        let watermark = self.gctx.mark();
        self.gctx.reset_graph_to(watermark);

        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let enc_mask = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let head_mask = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let conv_left = self.gctx.new_tensor_2d(GgmlType::F32, 1, t);
        let conv_right = self.gctx.new_tensor_2d(GgmlType::F32, 1, t);
        for x in [tokens_t, pos_t, enc_mask, head_mask, conv_left, conv_right] {
            self.gctx.arena_resize_tensor(x);
        }
        self.gctx
            .with_i32_mut(tokens_t, |q| q.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |q| {
                for (k, v) in q.iter_mut().enumerate() {
                    *v = k as i32;
                }
            })
            .unwrap();
        // text-only single sequence: every position visible in both masks
        self.gctx.with_f32_mut(enc_mask, |q| q.fill(0.0)).unwrap();
        self.gctx.with_f32_mut(head_mask, |q| q.fill(0.0)).unwrap();
        // the neighbor taps: left[i+1] = right[i] = 1 within the sequence
        self.gctx
            .with_f32_mut(conv_left, |q| {
                q[0] = 0.0;
                for v in q.iter_mut().skip(1) {
                    *v = 1.0;
                }
            })
            .unwrap();
        self.gctx
            .with_f32_mut(conv_right, |q| {
                for v in q.iter_mut().take(n - 1) {
                    *v = 1.0;
                }
                q[n - 1] = 0.0;
            })
            .unwrap();

        let inp = graph_arch::Lfm2DecisionInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask_enc: enc_mask,
            kq_mask_head: head_mask,
            conv_left,
            conv_right,
            out_ids: None,
        };
        let result =
            graph_arch::build_lfm2_decision_forward(&mut self.gctx, &self.w, &self.p, &inp, n);
        let scores = result.scores;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);
        bytemuck::cast_slice(self.gctx.data_bytes(scores).unwrap()).to_vec()
    }
}

/// the d1-omni mask rules pinned against the C's set_input bodies
/// (lfm2.cpp:360-384 media / :386-401 conv)
#[test]
fn lfm2d_mask_rules() {
    // 5 entries: text, media, media, text, text — one sequence
    let is_media = [false, true, true, false, false];
    let seq = [0u32; 5];
    let pos = [0i32, 1, 2, 3, 4];

    // enc mask: the media never reads the text
    let enc = graph_arch::lfm2_media_mask_rule(false, &is_media, &seq, 5);
    // row 1 (media): text col 0 masked, media cols visible
    assert_eq!(enc[1 * 5 + 0], f32::NEG_INFINITY);
    assert_eq!(enc[1 * 5 + 1], 0.0);
    assert_eq!(enc[1 * 5 + 2], 0.0);
    // row 0 (text): everything visible (text reads all)
    assert_eq!(enc[0 * 5 + 1], 0.0);

    // head mask: own kind only
    let head = graph_arch::lfm2_media_mask_rule(true, &is_media, &seq, 5);
    assert_eq!(head[0 * 5 + 1], f32::NEG_INFINITY); // text vs media
    assert_eq!(head[1 * 5 + 2], 0.0); // media vs media
    assert_eq!(head[3 * 5 + 4], 0.0); // text vs text

    // conv taps: the last media entry does not read the text on its right
    let (left, right) = graph_arch::lfm2_conv_mask_rule(&is_media, &seq, &pos, 5);
    assert_eq!(left, vec![0.0, 1.0, 1.0, 1.0, 1.0]);
    // right[1]: media reading its right media neighbor — visible
    assert_eq!(right[1], 1.0);
    // right[2]: the LAST media reading text on its right — masked
    assert_eq!(right[2], 0.0);
    assert_eq!(right[3], 1.0);
    assert_eq!(right[4], 0.0); // last entry
}

/// load + decide; sanity-asserts the hparams arm, the decision loader and
/// the [3, T] scores
#[test]
fn lfm2d_synth_load_and_decide() {
    build_file();
    let (mut m, vocab) = open_synth();
    assert_eq!(m.arch, llama::arch::LlmArch::LFM2);

    let hp = m.hparams.clone();
    // the hparams arm (lfm2.cpp:29-35)
    assert_eq!(hp.n_layer_decision, N_DECISION);
    assert!(!hp.causal_attn);
    assert_eq!(hp.f_norm_eps, 1e-6);
    assert_eq!(hp.n_embd_out(), 3); // N_DECISION_TYPES
    // is_recr from the per-layer kv heads
    assert!(hp.is_recr(0) && hp.is_recr(1));
    assert!(!hp.is_recr(2) && !hp.is_recr(3) && !hp.is_recr(4));

    let ids = vocab.tokenize(PROMPT, true, false);
    assert!(!ids.is_empty());

    let mut driver = D1Driver::new(&mut m);
    let scores = driver.decide(&ids);
    assert_eq!(scores.len(), 3 * ids.len(), "[3, T] scores");
    assert!(
        scores.iter().all(|x| x.is_finite()),
        "lfm2 decision: finite scores"
    );
    println!(
        "lfm2 decision ok ({} tokens, scores {:?})",
        ids.len(),
        &scores[..3.min(scores.len())]
    );
}

// ---------------------------------------------------------------------------
// the DECDMP1 dump driver + bitcompare (the ge2 mirror)
// ---------------------------------------------------------------------------

const ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn op_desc(op: ggml::GgmlOp, params: &[i32]) -> &'static str {
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
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => match params.first().copied().unwrap_or(10) {
            4 => "TANH",
            6 => "RELU",
            7 => "SIGMOID",
            8 => "GELU",
            15 => "SOFTPLUS",
            16 => "GELU_ERF",
            _ => "SILU",
        },
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => match params.first().copied().unwrap_or(2) {
            0 => "REGLU",
            1 => "GEGLU",
            3 => "SWIGLU_OAI",
            6 => "SWIGLU_CLAMP",
            _ => "SWIGLU",
        },
        Clamp => "CLAMP",
        TopK => "TOP_K",
        Sqrt => "SQRT",
        Pad => "PAD",
        Sum => "SUM",
        other => {
            let _ = other;
            "OTHER"
        }
    }
}

fn type_desc(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::I32 => "i32",
        GgmlType::I64 => "i64",
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
    put_str(&mut st.out, op_desc(node.op, &node.op_params));
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

#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the lfm2-decision parity file"]
fn lfm2d_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out_path = std::env::var("LFM2D_DUMP_OUT")
        .unwrap_or_else(|_| format!("{OUT_DIR}/port.bin"));

    let (mut m, vocab) = open_synth();
    let ids = vocab.tokenize(PROMPT, true, false);
    println!("lfm2d tokens: {ids:?} ({} tokens)", ids.len());

    DUMP.get_or_init(|| Mutex::new(Some(DumpState { out: Vec::new(), nodes: 0 })));
    {
        let mut g = DUMP.get().unwrap().lock().unwrap();
        *g = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    set_eval_callback(Some(dump_cb));
    let mut driver = D1Driver::new(&mut m);
    driver.decide(&ids);
    set_eval_callback(None);

    let (nodes, body) = {
        let mut guard = DUMP.get().unwrap().lock().unwrap();
        let st = guard.take().unwrap();
        (st.nodes, st.out)
    };
    let mut f = std::fs::File::create(&out_path).expect("create dump");
    f.write_all(b"DECDMP1\0").unwrap();
    f.write_all(&(ids.len() as u32).to_le_bytes()).unwrap();
    for t in &ids {
        f.write_all(&t.to_le_bytes()).unwrap();
    }
    f.write_all(&nodes.to_le_bytes()).unwrap();
    f.write_all(&body).unwrap();
    println!("lfm2d port dump: {nodes} nodes -> {out_path}");
}

#[test]
#[ignore = "manual: writes the synthetic lfm2-decision GGUF"]
fn lfm2d_write_synth_file() {
    build_file();
}

struct PortNode {
    name: String,
    ne: [i64; 4],
    e: Option<Vec<f32>>,
}

fn is_graph_name(name: &str) -> bool {
    !name.is_empty() && !name.starts_with(' ') && !name.starts_with("node_") && !name.contains(" (")
}

fn read_nodes(path: &str) -> (Vec<i32>, Vec<PortNode>) {
    let d = std::fs::read(path).expect("read dump");
    assert_eq!(&d[..8], b"DECDMP1\0", "{path}");
    let mut o = 8usize;
    let n_tok = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
    o += 4;
    let toks: Vec<i32> = (0..n_tok)
        .map(|i| i32::from_le_bytes(d[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
        .collect();
    o += 4 * n_tok;
    let n_nodes = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
    o += 4;
    let mut nodes = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let l = d[o] as usize;
        o += 1;
        let _op = String::from_utf8(d[o..o + l].to_vec()).unwrap();
        o += l;
        let l = d[o] as usize;
        o += 1;
        let name = String::from_utf8(d[o..o + l].to_vec()).unwrap();
        o += l;
        let l = d[o] as usize;
        o += 1;
        let _ty = String::from_utf8(d[o..o + l].to_vec()).unwrap();
        o += l;
        let mut ne = [0i64; 4];
        for e in ne.iter_mut() {
            *e = i64::from_le_bytes(d[o..o + 8].try_into().unwrap());
            o += 8;
        }
        let n = u64::from_le_bytes(d[o..o + 8].try_into().unwrap());
        o += 8;
        let e = if n < ELEM_CAP {
            let v = d[o..o + 4 * n as usize]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            o += 4 * n as usize;
            Some(v)
        } else {
            None
        };
        nodes.push(PortNode { name, ne, e });
    }
    assert_eq!(o, d.len(), "{path}: trailing bytes");
    (toks, nodes)
}

/// every cb-named node on the port side must exist on the reference side
/// with the byte-identical payload (occurrence pairing)
#[test]
#[ignore = "manual: run after parity/gen_lfm2_decision_ref.sh"]
fn lfm2d_reference_bitcompare() {
    let ref_path = format!("{OUT_DIR}/ref.bin");
    let port_path = format!("{OUT_DIR}/port.bin");
    let (ref_toks, ref_nodes) = read_nodes(&ref_path);
    let (port_toks, port_nodes) = read_nodes(&port_path);
    assert_eq!(ref_toks, port_toks, "token streams");
    assert!(
        ref_nodes.len() > 100,
        "reference stream present ({} nodes)",
        ref_nodes.len()
    );

    let mut ridx: std::collections::HashMap<String, Vec<usize>> = Default::default();
    for (i, n) in ref_nodes.iter().enumerate() {
        if is_graph_name(&n.name) {
            ridx.entry(n.name.clone()).or_default().push(i);
        }
    }
    let mut seen: std::collections::HashMap<(String, [i64; 4]), usize> = Default::default();
    let mut compared = 0usize;
    for n in port_nodes.iter() {
        let pname = n.name.clone();
        if !is_graph_name(&pname) || n.e.is_none() {
            continue;
        }
        // the encode reroute renames the scores output
        let lookup = if pname == "decision_scores" {
            "result_embd_pooled".to_string()
        } else {
            pname.clone()
        };
        // shape-aware occurrence pairing: repeated names (the per-head
        // Q/K norms share "norm-{il}") pair with the next same-shape
        // occurrence
        // the cursor stores the LAST matched ref index (shape-interleaved
        // names must not re-pair an already-matched occurrence)
        let occ = seen.entry((pname.clone(), n.ne)).or_insert(usize::MAX);
        let Some(ixs) = ridx.get(&lookup) else {
            panic!("node {pname:?} has no reference counterpart");
        };
        let rix = ixs
            .iter()
            .copied()
            .skip_while(|&ix| *occ != usize::MAX && ix <= *occ)
            .find(|&ix| ref_nodes[ix].ne == n.ne);
        let Some(rix) = rix else {
            panic!("node {pname:?} (ne {:?}): no same-shape reference occurrence left", n.ne);
        };
        *occ = rix;
        let rnode = &ref_nodes[rix];
        assert_eq!(rnode.ne, n.ne, "node {pname:?} (ref ix {rix}) shape");
        assert_eq!(
            rnode.e.as_ref().map(|p| p.len()),
            Some(n.e.as_ref().unwrap().len()),
            "node {pname:?} payload length"
        );
        let re = rnode.e.as_ref().unwrap();
        let pe = n.e.as_ref().unwrap();
        let bad: Vec<usize> = re
            .iter()
            .zip(pe.iter())
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(k, _)| k)
            .collect();
        assert!(
            bad.is_empty(),
            "node {pname:?} occurrence {} payload bits diverge at {}/{} elems (first: {:?} != {:?})",
            *occ - 1,
            bad.len(),
            re.len(),
            re.get(bad[0]),
            pe.get(bad[0])
        );
        compared += 1;
    }
    println!(
        "lfm2 decision: {} port nodes, {compared} cb-named nodes paired, all bit-identical",
        port_nodes.len()
    );
}
