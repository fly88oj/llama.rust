//! gemma_embedding2_e2e.rs — the EmbeddingGemma2 arch (4fbc76dec,
//! c35b66744): the text+vision+audio embedding model — the gemma-embedding
//! body (dual post-norms, QK norms, symmetric-SWA no-cache pair) plus the
//! weightless V RMS norm, the per-layer embedding inputs
//! (per_layer_model_proj + per-layer gate/proj/post-norm), the per-layer
//! out_scale and the final n_embd_out projection.
//!
//! Acceptance follows the k2/glm5 protocol: `ge2_write_synth_file` writes
//! the file, the NEW reference drives it through parity/gen_gemma2_ref.sh
//! (ref_model_saver banner + ref_decode_dump --fa off — decode reroutes
//! the null-memory arch to encode), and this file's port-side dump driver
//! (`ge2_prefill_node_dump`, a direct `build_gemma_embedding2_forward`
//! driver — the tts-batch convention, no EncoderWeights/CLI routing)
//! writes the same DECDMP1 stream for `ge2_reference_bitcompare` (the
//! glm5_dump named-node pairing).

use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::graph_arch::{self, EncodeInputs, EurobertRope};

const VOCAB_SPM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf"
);
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/synca/ge2";
const FILE: &str = "gemma-embedding2-synth.gguf";

const N_EMBD: i64 = 128;
const N_EMBD_PER_LAYER: i64 = 32;
const N_EMBD_OUT: i64 = 64;
const N_LAYER: usize = 3;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 4;
const HD: i64 = 32;
const N_FF: i64 = 96;
const N_SWA: u32 = 8; // binds the 12-token prompt
const PROMPT: &str = "The capital of France is";

// ---------------------------------------------------------------------------
// the writer (the mtp2/k2 recipe)
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
    push(
        "per_layer_model_proj.weight".into(),
        vec![N_EMBD, N_EMBD_PER_LAYER * N_LAYER as i64],
        "proj",
    );
    push(
        "per_layer_proj_norm.weight".into(),
        vec![N_EMBD_PER_LAYER],
        "norm",
    );
    push("output_norm.weight".into(), vec![N_EMBD], "norm");
    push("output.weight".into(), vec![N_EMBD, N_EMBD_OUT], "proj");

    for i in 0..N_LAYER as i32 {
        push(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, HD * N_HEAD], "proj");
        push(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, HD * N_HEAD_KV], "proj");
        push(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, HD * N_HEAD_KV], "proj");
        push(format!("blk.{i}.attn_output.weight"), vec![HD * N_HEAD, N_EMBD], "proj");
        push(format!("blk.{i}.attn_q_norm.weight"), vec![HD], "norm");
        push(format!("blk.{i}.attn_k_norm.weight"), vec![HD], "norm");
        push(format!("blk.{i}.post_attention_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], "proj");
        push(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], "proj");
        push(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], "proj");
        push(format!("blk.{i}.post_ffw_norm.weight"), vec![N_EMBD], "norm");
        push(
            format!("blk.{i}.inp_gate.weight"),
            vec![N_EMBD, N_EMBD_PER_LAYER],
            "proj",
        );
        push(
            format!("blk.{i}.proj.weight"),
            vec![N_EMBD_PER_LAYER, N_EMBD],
            "proj",
        );
        push(format!("blk.{i}.post_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.layer_output_scale.weight"), vec![1], "norm");
    }
    t
}

fn build_file() -> String {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/synca/ge2");
    let path = format!("{OUT_DIR}/{FILE}");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "gemma-embedding2";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-gemma-embedding2".into()));
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.embedding_length_out"), Value::U32(N_EMBD_OUT as u32));
    kv!(format!("{a}.embedding_length_per_layer_input"), Value::U32(N_EMBD_PER_LAYER as u32));
    kv!(format!("{a}.block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(N_HEAD_KV as u32));
    kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.attention.sliding_window"), Value::U32(N_SWA));
    // the every-6th-full default would leave all 3 layers SWA-free; write an
    // explicit pattern so layer 1 is full and 0/2 are windowed
    kv!(
        format!("{a}.attention.sliding_window_pattern"),
        Value::Array(
            ggml::GgufType::Bool,
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)]
        )
    );
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    let mut rng = Rng(0x5eed_0000_9e22_0002);
    let tensors = tensors_of();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (name, ne, kind) in &tensors {
        let n: usize = ne.iter().map(|&x| x as usize).product();
        let vals: Vec<f32> = if kind == "norm" {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
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
// the port-side driver (the tts convention: the builder directly)
// ---------------------------------------------------------------------------

fn open_synth() -> (llama::model::LlamaModel, llama::vocab::Vocab) {
    let path = format!("{OUT_DIR}/{FILE}");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e} — run ge2_write_synth_file first"));
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let m = llama::model::load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
    (m, vocab)
}

struct Ge2Driver {
    gctx: Context,
    w: graph_arch::GemmaEmbedding2ModelWeights,
    p: graph_arch::GemmaEmbedding2Params,
}

impl Ge2Driver {
    fn new(m: &mut llama::model::LlamaModel) -> Self {
        let hp = m.hparams.clone();
        let w = m.gemma_embedding2_weights();
        let p = graph_arch::GemmaEmbedding2Params {
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head: hp.n_embd_head_k(0) as i64,
            norm_rms_eps: hp.f_norm_rms_eps,
            n_embd_per_layer: hp.n_embd_per_layer as i64,
            f_attention_scale: hp.f_attention_scale,
            is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
            freq_base_swa: hp.rope_freq_base_train_swa,
            freq_scale_swa: hp.rope_freq_scale_train_swa,
            rope: EurobertRope {
                n_rot: hp.n_rot(0) as i32,
                // llama_model_rope_type(GEMMA_EMBEDDING2) = NEOX
                rope_mode: ggml::ops::GGML_ROPE_TYPE_NEOX,
                n_ctx_orig: hp.n_ctx_train as i32,
                freq_base: hp.rope_freq_base_train,
                freq_scale: hp.rope_freq_scale_train,
                ext_factor: 0.0,
                attn_factor: 1.0,
                beta_fast: hp.yarn_beta_fast,
                beta_slow: hp.yarn_beta_slow,
            },
        };
        let gctx = std::mem::replace(&mut m.ctx, Context::new());
        Ge2Driver { gctx, w, p }
    }

    /// one encode over `tokens` — the non-causal all-attend mask + the
    /// symmetric-SWA twin; returns the [n_embd_out, T] rows
    fn encode(&mut self, tokens: &[i32]) -> Vec<f32> {
        let n = tokens.len();
        let t = n as i64;
        let watermark = self.gctx.mark();
        self.gctx.reset_graph_to(watermark);

        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let kq_mask = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let kq_mask_swa = self.gctx.new_tensor_2d(GgmlType::F32, t, t);
        let out_ids = self.gctx.new_tensor_1d(GgmlType::I32, t);
        for x in [tokens_t, pos_t, kq_mask, kq_mask_swa, out_ids] {
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
        self.gctx
            .with_i32_mut(out_ids, |q| {
                for (k, v) in q.iter_mut().enumerate() {
                    *v = k as i32;
                }
            })
            .unwrap();
        // non-causal: every position attends every position
        self.gctx
            .with_f32_mut(kq_mask, |q| q.fill(0.0))
            .unwrap();
        // the symmetric-window twin (is_masked_swa, SYMMETRIC)
        let n_swa = self.is_swa_window();
        self.gctx
            .with_f32_mut(kq_mask_swa, |q| {
                for j in 0..n {
                    for i in 0..n {
                        let masked = llama::hparams::LlamaHparams::is_masked_swa(
                            n_swa,
                            llama::hparams::LlamaSwaType::SYMMETRIC,
                            i as i32,
                            j as i32,
                        );
                        q[j * n + i] = if masked { f32::NEG_INFINITY } else { 0.0 };
                    }
                }
            })
            .unwrap();

        let inp = EncodeInputs {
            tokens: tokens_t,
            pos: Some(pos_t),
            pos_bucket: None,
            kq_mask,
            kq_mask_swa: Some(kq_mask_swa),
            out_ids,
            mean: None,
            cls: None,
        };
        let result = graph_arch::build_gemma_embedding2_forward(&mut self.gctx, &self.w, &self.p, &inp, n);
        let embd = result.embd;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);
        bytemuck::cast_slice(self.gctx.data_bytes(embd).unwrap()).to_vec()
    }

    fn is_swa_window(&self) -> u32 {
        // the synth file's attention.sliding_window
        N_SWA
    }
}

/// load + encode; sanity-asserts the hparams arm, the loader and the graph
#[test]
fn ge2_synth_load_and_encode() {
    build_file();
    let (mut m, vocab) = open_synth();
    assert_eq!(m.arch, llama::arch::LlmArch::GEMMA_EMBEDDING2);

    let hp = m.hparams.clone();
    // the hparams arm (gemma-embedding2.cpp:3-19)
    assert_eq!(hp.swa_type, llama::hparams::LlamaSwaType::SYMMETRIC);
    assert!(!hp.causal_attn);
    assert_eq!(hp.f_attention_scale, 1.0);
    assert_eq!(hp.n_swa, N_SWA);
    assert_eq!(hp.f_norm_rms_eps, 1e-5);
    assert_eq!(hp.n_embd_per_layer, N_EMBD_PER_LAYER as u32);
    assert_eq!(hp.n_embd_out(), N_EMBD_OUT as u32);
    // the explicit pattern: layers 0/2 windowed, layer 1 full
    assert!(hp.is_swa(0) && !hp.is_swa(1) && hp.is_swa(2));

    let ids = vocab.tokenize(PROMPT, true, false);
    assert!(!ids.is_empty());

    let mut driver = Ge2Driver::new(&mut m);
    let embd = driver.encode(&ids);
    assert_eq!(embd.len(), (ids.len() * N_EMBD_OUT as usize), "rows*width");
    assert!(
        embd.iter().all(|x| x.is_finite()),
        "ge2: non-finite embeddings"
    );
    println!("gemma-embedding2 encode ok ({} tokens)", ids.len());
}

// ---------------------------------------------------------------------------
// the DECDMP1 dump driver (the k2/glm5 mirror)
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
            7 => "SIGMOID",
            8 => "GELU",
            15 => "SOFTPLUS",
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

/// Stream every node of one encode to GE2_DUMP_OUT — the port half of the
/// node-flow bit-compare.
#[test]
#[ignore = "manual: writes the DECDMP1 node dump for the gemma-embedding2 parity file"]
fn ge2_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out_path =
        std::env::var("GE2_DUMP_OUT").unwrap_or_else(|_| format!("{OUT_DIR}/port.bin"));

    let (mut m, vocab) = open_synth();
    let ids = vocab.tokenize(PROMPT, true, false);
    println!("ge2 tokens: {ids:?} ({} tokens)", ids.len());

    DUMP.get_or_init(|| Mutex::new(Some(DumpState { out: Vec::new(), nodes: 0 })));
    {
        let mut g = DUMP.get().unwrap().lock().unwrap();
        *g = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    set_eval_callback(Some(dump_cb));
    let mut driver = Ge2Driver::new(&mut m);
    driver.encode(&ids);
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
    println!("ge2 port dump: {nodes} nodes -> {out_path}");
}

/// write the synthetic file (used by parity/gen_gemma2_ref.sh)
#[test]
#[ignore = "manual: writes the synthetic gemma-embedding2 GGUF"]
fn ge2_write_synth_file() {
    build_file();
}

// ---------------------------------------------------------------------------
// the reference bit-compare (the glm5 named-node pairing)
// ---------------------------------------------------------------------------

struct PortNode {
    op: String,
    name: String,
    ty: String,
    ne: [i64; 4],
    n: u64,
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
        let op = String::from_utf8(d[o..o + l].to_vec()).unwrap();
        o += l;
        let l = d[o] as usize;
        o += 1;
        let name = String::from_utf8(d[o..o + l].to_vec()).unwrap();
        o += l;
        let l = d[o] as usize;
        o += 1;
        let ty = String::from_utf8(d[o..o + l].to_vec()).unwrap();
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
        nodes.push(PortNode { op, name, ty, ne, n, e });
    }
    assert_eq!(o, d.len(), "{path}: trailing bytes");
    (toks, nodes)
}

/// every cb-named node on the port side must exist on the reference side
/// with the byte-identical payload (occurrence pairing)
#[test]
#[ignore = "manual: run after parity/gen_gemma2_ref.sh"]
fn ge2_reference_bitcompare() {
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
        // the encode reroute's build_pooling renames the final projection
        // even under pooling NONE (the glm5_dump convention)
        let lookup = if pname == "result_embd" {
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
        "gemma-embedding2: {} port nodes, {compared} cb-named nodes paired, all bit-identical",
        port_nodes.len()
    );
}
