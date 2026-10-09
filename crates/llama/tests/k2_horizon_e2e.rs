//! k2_horizon_e2e.rs — the K2 Horizon arch (462524043, c35b66744): dense +
//! MoVA hybrid on synthetic GGUFs. Two files cover the mechanism matrix:
//!   * dense  — all-dense layers, `attn_gate` (the softplus log2(1+2^x)
//!              output gate) on layer 1 only, flat per-head q/k norms, tied
//!              lm_head (the `output == NULL` tok-embd fallback)
//!   * mova   — dense_lead 1; layers 1-3 are MoE+MoVA: `attn_v_gate`
//!              (+ bias on layer 2), `attn_v_exps`, sigmoid router (the
//!              NONE->SIGMOID loader conversion), weights_norm renorm,
//!              weights_scale, shared expert, explicit lm_head
//!
//! Acceptance follows the glm5/mtp2 protocol: `k2_write_synth_files` writes
//! the files, the NEW reference drives them through parity/gen_k2_ref.sh
//! (ref_model_saver banner + ref_decode_dump --fa off --decode-tail 12 node
//! streams), and this file's port-side dump driver (`k2_prefill_node_dump`,
//! a direct `build_k2_horizon_forward` driver — the tts-batch convention,
//! no ForwardWeights/CLI routing) writes the same DECDMP1 stream for
//! `k2_reference_bitcompare`.
//!
//! The decode harness is the mtp2 direct-driver shape (KvCache + causal
//! mask + per-step graph rebuild); flash attention is OFF on both sides
//! (the MTP-batch lesson: the probe needs flash_attn_type=DISABLED for the
//! node streams to align).

use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use ggml::compute::{set_eval_callback, EvalNode};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::kv_cache::{KvCache, SlotInfo};
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf"
);
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/synca/k2";

const N_EMBD: i64 = 128;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HD: i64 = 32;
const N_FF: i64 = 64;
const N_FF_EXP: i64 = 32;
const N_FF_SHEXP: i64 = 24;
const N_EXPERT: u32 = 4;
const N_EXPERT_USED: u32 = 2;
const N_VALUE_EXPERT: u32 = 4;
const N_VALUE_EXPERT_USED: u32 = 2;
const N_CTX: u32 = 512;
const N_GROUPS: u32 = 2; // the group-RMS slice count
const PROMPT: &str = "The capital of France is";
const N_TAIL: usize = 12;

// ---------------------------------------------------------------------------
// the specs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Spec {
    Dense,
    Mova,
}

impl Spec {
    fn name(self) -> &'static str {
        match self {
            Spec::Dense => "dense",
            Spec::Mova => "mova",
        }
    }
    fn n_layer(self) -> usize {
        match self {
            Spec::Dense => 3,
            Spec::Mova => 4,
        }
    }
    fn dense_lead(self) -> u32 {
        match self {
            Spec::Dense => 0,
            Spec::Mova => 1,
        }
    }
    fn is_moe(self) -> bool {
        self == Spec::Mova
    }
}

// ---------------------------------------------------------------------------
// the writer (the mtp2 recipe)
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

/// the tensor roles and their init scales (mtp2's scale_of)
fn scale_of(kind: &str) -> f32 {
    match kind {
        "norm" => 1.0,
        "bias" => 0.02,
        _ => 1.0 / (N_EMBD as f32).sqrt(),
    }
}

fn tensors_of(spec: Spec) -> Vec<(String, Vec<i64>, String)> {
    let n_layer = spec.n_layer();
    let dense_lead = spec.dense_lead();
    let mut t = Vec::new();
    let mut push = |name: String, ne: Vec<i64>, kind: &str| {
        t.push((name, ne, kind.to_string()));
    };

    push("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], "proj");
    push("output_norm.weight".into(), vec![N_EMBD], "norm");
    if spec == Spec::Mova {
        push("output.weight".into(), vec![N_EMBD, N_VOCAB], "proj");
    } // dense file: tied (the dup fallback)

    for i in 0..n_layer as i32 {
        let is_moe = spec.is_moe() && (i as u32 >= dense_lead);
        let is_mova = is_moe; // every MoE layer of this file routes its values

        push(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], "norm");
        push(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, HD * N_HEAD], "proj");
        push(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, HD * N_HEAD_KV], "proj");
        // one norm weight per head, stored flat (k2-horizon.cpp:75-76,
        // TENSOR_ALLOW_RESHAPE views them {head_dim, n_head})
        push(
            format!("blk.{i}.attn_q_norm.weight"),
            vec![HD * N_HEAD],
            "norm",
        );
        push(
            format!("blk.{i}.attn_k_norm.weight"),
            vec![HD * N_HEAD_KV],
            "norm",
        );

        if is_mova {
            push(
                format!("blk.{i}.attn_v_gate.weight"),
                vec![N_EMBD, N_VALUE_EXPERT as i64],
                "proj",
            );
            if i == 2 {
                // the selection-only bias on one layer
                push(
                    format!("blk.{i}.attn_v_gate.bias"),
                    vec![N_VALUE_EXPERT as i64],
                    "bias",
                );
            }
            push(
                format!("blk.{i}.attn_v_exps.weight"),
                vec![N_EMBD, HD * N_HEAD_KV, N_VALUE_EXPERT as i64],
                "proj",
            );
        } else {
            push(
                format!("blk.{i}.attn_v.weight"),
                vec![N_EMBD, HD * N_HEAD_KV],
                "proj",
            );
        }

        push(
            format!("blk.{i}.attn_output.weight"),
            vec![HD * N_HEAD, N_EMBD],
            "proj",
        );
        // the softplus output gate on layers 1 (dense file) / 2 (mova file)
        let gated = match spec {
            Spec::Dense => i == 1,
            Spec::Mova => i == 2,
        };
        if gated {
            push(
                format!("blk.{i}.attn_gate.weight"),
                vec![N_EMBD, HD * N_HEAD],
                "proj",
            );
        }

        push(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], "norm");

        if is_moe {
            push(
                format!("blk.{i}.ffn_gate_inp.weight"),
                vec![N_EMBD, N_EXPERT as i64],
                "proj",
            );
            push(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT as i64], "bias");
            push(
                format!("blk.{i}.ffn_gate_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERT as i64],
                "proj",
            );
            push(
                format!("blk.{i}.ffn_up_exps.weight"),
                vec![N_EMBD, N_FF_EXP, N_EXPERT as i64],
                "proj",
            );
            push(
                format!("blk.{i}.ffn_down_exps.weight"),
                vec![N_FF_EXP, N_EMBD, N_EXPERT as i64],
                "proj",
            );
            // the shared expert (explicit width on this file)
            push(
                format!("blk.{i}.ffn_gate_shexp.weight"),
                vec![N_EMBD, N_FF_SHEXP],
                "proj",
            );
            push(
                format!("blk.{i}.ffn_up_shexp.weight"),
                vec![N_EMBD, N_FF_SHEXP],
                "proj",
            );
            push(
                format!("blk.{i}.ffn_down_shexp.weight"),
                vec![N_FF_SHEXP, N_EMBD],
                "proj",
            );
        } else {
            push(format!("blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], "proj");
            push(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], "proj");
            push(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], "proj");
        }
    }
    t
}

fn build_file(spec: Spec) -> String {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/synca/k2");
    let path = format!("{OUT_DIR}/k2-horizon-{}-synth.gguf", spec.name());

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "k2-horizon";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!(
        "general.name",
        Value::String(format!("llama-rust-synth-k2-{}", spec.name()))
    );
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(N_CTX));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32(spec.n_layer() as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::U32(N_HEAD_KV as u32)
    );
    kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    kv!(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5)
    );
    // the group-RMS slice count (k2-horizon.cpp:7-10)
    kv!(
        format!("{a}.attention.group_norm_groups"),
        Value::U32(N_GROUPS)
    );
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // YaRN betas (k2-horizon.cpp:4-5) — read for the KV contract; no
    // rope.scaling.* keys, so the factors stay identity on both sides
    kv!(format!("{a}.rope.scaling.yarn_beta_fast"), Value::F32(16.0));
    kv!(format!("{a}.rope.scaling.yarn_beta_slow"), Value::F32(4.0));

    if spec.is_moe() {
        kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT));
        kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT_USED));
        kv!(
            format!("{a}.expert_feed_forward_length"),
            Value::U32(N_FF_EXP as u32)
        );
        kv!(
            format!("{a}.leading_dense_block_count"),
            Value::U32(spec.dense_lead())
        );
        kv!(format!("{a}.expert_shared_count"), Value::U32(1));
        kv!(
            format!("{a}.expert_shared_feed_forward_length"),
            Value::U32(N_FF_SHEXP as u32)
        );
        // weights_norm=true exercises the renormalize path; 0.5 the scale
        kv!(format!("{a}.expert_weights_norm"), Value::Bool(true));
        kv!(format!("{a}.expert_weights_scale"), Value::F32(0.5));
        // expert_gating_func omitted -> NONE -> loader forces SIGMOID
        // (k2-horizon.cpp:21-23)
        // MoVA geometry (k2-horizon.cpp:27-28)
        kv!(
            format!("{a}.attention.value_expert_count"),
            Value::U32(N_VALUE_EXPERT)
        );
        kv!(
            format!("{a}.attention.value_expert_used_count"),
            Value::U32(N_VALUE_EXPERT_USED)
        );
    }

    let mut rng = Rng(0x5eed_0000_6b32_0001 + spec.n_layer() as u64);
    let tensors = tensors_of(spec);
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (name, ne, kind) in &tensors {
        let scale = scale_of(kind);
        let n: usize = ne.iter().map(|&x| x as usize).product();
        let vals: Vec<f32> = if kind == "norm" {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
        } else {
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
// the port-side driver — the mtp2 direct-driver shape over
// build_k2_horizon_forward (no ForwardWeights routing; the tts-batch
// convention — context.rs is outside this round's ownership)
// ---------------------------------------------------------------------------

fn k2_params(m: &LlamaModel) -> (graph_arch::K2HorizonModelWeights, graph_arch::K2HorizonParams) {
    let hp = &m.hparams;
    let w = m.k2_horizon_weights();
    let p = graph_arch::K2HorizonParams {
        attn: AttnParams {
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head_k: hp.n_embd_head_k(0) as i64,
            n_embd_head_v: hp.n_embd_head_v(0) as i64,
            n_rot: hp.n_rot(0) as i64,
            // llama_model_rope_type(K2_HORIZON) = NEOX (llama-model.cpp:3220)
            rope_mode: ggml::ops::GGML_ROPE_TYPE_NEOX,
            n_ctx_orig: hp.n_ctx_train as i32,
            freq_base: hp.rope_freq_base_train,
            freq_scale: hp.rope_freq_scale_train,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: hp.yarn_beta_fast,
            beta_slow: hp.yarn_beta_slow,
            norm_eps: hp.f_norm_rms_eps,
            use_flash_attn: false, // the probe runs --fa off
        },
        n_norm_groups: hp.n_norm_groups as i64,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_value_expert: hp.n_value_expert as i64,
        n_value_expert_used: hp.n_value_expert_used as i64,
        expert_gating_func: hp.expert_gating_func as i32,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
    };
    (w, p)
}

struct K2Driver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    w: graph_arch::K2HorizonModelWeights,
    p: graph_arch::K2HorizonParams,
}

impl K2Driver {
    fn new(m: &mut LlamaModel) -> Self {
        let (w, p) = k2_params(m);
        let n_layer = w.layers.len();
        let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
        let kv = {
            let hd_k = p.attn.n_embd_head_k * p.attn.n_head_kv;
            let hd_v = p.attn.n_embd_head_v * p.attn.n_head_kv;
            let ks = vec![hd_k; n_layer];
            let vs = vec![hd_v; n_layer];
            KvCache::new_with_dims(&mut gctx, &ks, &vs, N_CTX)
        };
        let watermark = gctx.mark();
        K2Driver {
            gctx,
            kv,
            watermark,
            w,
            p,
        }
    }

    /// one graph step over `tokens` at `pos` — every row an output row (the
    /// embeddings/pooling-none shape: n_outputs == n_tokens, the gather
    /// elided). Computes and returns the last row of logits.
    fn step(&mut self, tokens: &[i32], pos: &[i32]) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let kq_mask = self
            .gctx
            .new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for t in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(t);
        }
        self.gctx
            .with_i32_mut(tokens_t, |q| q.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |q| q.copy_from_slice(pos))
            .unwrap();
        {
            let bytes = self.gctx.data_bytes_mut(row_idx).unwrap();
            let vals: Vec<i64> = (sinfo.s0..sinfo.s0 + n as u32)
                .map(|i| i as i64)
                .collect();
            bytes.copy_from_slice(bytemuck::cast_slice(&vals));
        }
        {
            // the causal mask over the base cells (one sequence, id 0)
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            let mask: &mut [f32] =
                bytemuck::cast_slice_mut(self.gctx.data_bytes_mut(kq_mask).unwrap());
            llama::graph::fill_causal_mask(mask, &kv_pos, pos);
        }

        let inp = llama::graph::DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let result = graph_arch::build_k2_horizon_forward(
            &mut self.gctx,
            &self.w,
            &self.p,
            &self.kv,
            &inp,
            SlotInfo {
                s0: sinfo.s0,
                s1: sinfo.s1,
            },
            n_kv,
            n,
        );
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);

        let n_vocab = self.gctx.ne(logits)[0] as usize;
        let all: Vec<f32> = bytemuck::cast_slice(self.gctx.data_bytes(logits).unwrap()).to_vec();
        all[n_vocab * (n - 1)..n_vocab * n].to_vec()
    }
}

fn open_synth(spec: Spec) -> (LlamaModel, llama::vocab::Vocab) {
    let path = format!("{OUT_DIR}/k2-horizon-{}-synth.gguf", spec.name());
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e} — run k2_write_synth_files first"));
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let m = load_model(&gguf, mmap.clone()).expect("load model");
    let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
    (m, vocab)
}

/// load + prefill + decode tail; sanity-asserts the hparams arm and the
/// loader, runs the graph, checks the caches and finite logits
#[test]
fn k2_synth_load_and_decode() {
    for spec in [Spec::Dense, Spec::Mova] {
        build_file(spec);
        let (mut m, vocab) = open_synth(spec);
        assert_eq!(m.arch, llama::arch::LlmArch::K2_HORIZON);

        let hp = m.hparams.clone();
        // the hparams arm (k2-horizon.cpp:3-49)
        assert_eq!(hp.n_norm_groups, N_GROUPS);
        assert_eq!(hp.f_norm_rms_eps, 1e-5);
        assert_eq!(hp.yarn_beta_fast, 16.0);
        assert_eq!(hp.yarn_beta_slow, 4.0);
        if spec.is_moe() {
            assert_eq!(hp.n_expert, N_EXPERT);
            assert_eq!(hp.n_expert_used(0) as u32, N_EXPERT_USED);
            assert_eq!(hp.n_layer_dense_lead, spec.dense_lead());
            assert_eq!(hp.n_expert_shared, 1);
            assert_eq!(hp.n_ff_shexp, N_FF_SHEXP as u32);
            assert!(hp.expert_weights_norm);
            assert_eq!(hp.expert_weights_scale, 0.5);
            // NONE -> SIGMOID (k2-horizon.cpp:21-23)
            assert_eq!(hp.expert_gating_func, 2);
            assert_eq!(hp.n_value_expert, N_VALUE_EXPERT);
            assert_eq!(hp.n_value_expert_used, N_VALUE_EXPERT_USED);
        } else {
            assert_eq!(hp.n_value_expert, 0);
            assert_eq!(hp.n_value_expert_used, 0);
        }

        let ids = vocab.tokenize(PROMPT, true, false);
        assert!(!ids.is_empty());
        let pos: Vec<i32> = (0..ids.len() as i32).collect();

        let mut driver = K2Driver::new(&mut m);
        let mut last = driver.step(&ids, &pos);
        for s in 0..N_TAIL {
            let tok = 100; // the probe's fixed tail token
            let p = ids.len() as i32 + s as i32;
            last = driver.step(&[tok], &[p]);
        }
        assert_eq!(driver.kv.used_cells() as usize, ids.len() + N_TAIL);
        assert!(
            last.iter().all(|x| x.is_finite()),
            "{}: finite logits",
            spec.name()
        );
        println!("k2 {} ok ({} prompt tokens)", spec.name(), ids.len());
    }
}

// ---------------------------------------------------------------------------
// the DECDMP1 dump driver (the glm5_dump mirror, over the direct driver)
// ---------------------------------------------------------------------------

/// nodes at/above this element count carry no payload (2^19, the C probe rule)
const ELEM_CAP: u64 = 1 << 19;

struct DumpState {
    out: Vec<u8>,
    nodes: u32,
}

static DUMP: OnceLock<Mutex<Option<DumpState>>> = OnceLock::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// `ggml_op_desc` (ggml.c:1380-1389) — UNARY/GLU variants described by their
/// variant names (the arch_batch7 mapping)
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
        // UNARY ops are described by their variant name (ggml.c:1382)
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
        Fill => "FILL",
        LightningIndexer => "lightning_indexer(q, k, weights, mask)",
        TopK => "TOP_K",
        Sqrt => "SQRT",
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Pad => "PAD",
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
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

/// Stream every node of the prefill + N decode-tail steps to
/// K2_DUMP_OUT — the port half of the node-flow bit-compare.
#[test]
#[ignore = "manual: writes the DECDMP1 node dumps for the k2 parity files"]
fn k2_prefill_node_dump() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = match std::env::var("K2_SPEC").as_deref() {
        Ok("mova") => Spec::Mova,
        _ => Spec::Dense,
    };
    let out_path = std::env::var("K2_DUMP_OUT")
        .unwrap_or_else(|_| format!("/tmp/synca/k2/{}-port.bin", spec.name()));
    let n_tail: usize = std::env::var("K2_DECODE_TAIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_TAIL);

    let (mut m, vocab) = open_synth(spec);
    let ids = vocab.tokenize(PROMPT, true, false);
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    println!(
        "k2 {} tokens: {ids:?} ({} tokens)",
        spec.name(),
        ids.len()
    );

    DUMP.get_or_init(|| Mutex::new(Some(DumpState { out: Vec::new(), nodes: 0 })));
    {
        let mut g = DUMP.get().unwrap().lock().unwrap();
        *g = Some(DumpState {
            out: Vec::new(),
            nodes: 0,
        });
    }

    set_eval_callback(Some(dump_cb));
    let mut driver = K2Driver::new(&mut m);
    driver.step(&ids, &pos);
    for s in 0..n_tail {
        let tok = 100;
        let p = ids.len() as i32 + s as i32;
        driver.step(&[tok], &[p]);
    }
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
    println!("k2 {} port dump: {} nodes -> {out_path}", spec.name(), nodes);
}

/// write the two synthetic files (used by parity/gen_k2_ref.sh)
#[test]
#[ignore = "manual: writes the synthetic k2-horizon GGUFs"]
fn k2_write_synth_files() {
    build_file(Spec::Dense);
    build_file(Spec::Mova);
}

// ---------------------------------------------------------------------------
// the reference bit-compare (runs after parity/gen_k2_ref.sh)
// ---------------------------------------------------------------------------

struct PortNode {
    op: String,
    name: String,
    ty: String,
    ne: [i64; 4],
    n: u64,
    e: Option<Vec<f32>>,
}

impl PortNode {
    fn op_and_name(&self) -> String {
        self.name.clone()
    }
}

/// only real graph names — ggml's auto names ("x (view)", "(reshaped)",
/// "node_N") are shared across unrelated nodes
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

/// compare the port dump against the NEW-reference dump produced by
/// parity/gen_k2_ref.sh (ref_decode_dump --fa off --decode-tail 12) — the
/// glm5_dump acceptance shape: every cb-named node on the port side must
/// exist on the reference side with the byte-identical payload (occurrence
/// pairing, since names repeat across the 13 graph builds)
#[test]
#[ignore = "manual: run after parity/gen_k2_ref.sh"]
fn k2_reference_bitcompare() {
    for spec in [Spec::Dense, Spec::Mova] {
        let name = spec.name();
        let ref_path = format!("{OUT_DIR}/{name}-ref.bin");
        let port_path = format!("{OUT_DIR}/{name}-port.bin");
        let (ref_toks, ref_nodes) = read_nodes(&ref_path);
        let (port_toks, port_nodes) = read_nodes(&port_path);
        assert_eq!(ref_toks, port_toks, "{name}: token streams");
        assert!(
            ref_nodes.len() > 500,
            "{name}: reference stream present ({} nodes)",
            ref_nodes.len()
        );

        // the reference name index
        let mut ridx: std::collections::HashMap<String, Vec<usize>> = Default::default();
        for (i, n) in ref_nodes.iter().enumerate() {
            if is_graph_name(&n.op_and_name()) {
                ridx.entry(n.op_and_name()).or_default().push(i);
            }
        }
        let mut seen: std::collections::HashMap<(String, [i64; 4]), usize> = Default::default();
        let mut compared = 0usize;
        for n in port_nodes.iter() {
            let pname = n.op_and_name();
            if !is_graph_name(&pname) || n.e.is_none() {
                continue;
            }
            // the embeddings driver renames the final norm on the reference
            // side (build_pooling's cb "result_embd_pooled", the t_embd tap)
            let lookup = if pname == "result_norm" {
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
            assert_eq!(
                rnode.ne, n.ne,
                "{name}: node {pname:?} occurrence {} shape",
                *occ - 1
            );
            assert_eq!(
                rnode.e.as_ref().map(|p| p.len()),
                Some(n.e.as_ref().unwrap().len()),
                "{name}: node {pname:?} payload length"
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
                "{name}: node {pname:?} occurrence {} payload bits diverge at {}/{} elems (first: {:?} != {:?})",
                *occ - 1,
                bad.len(),
                re.len(),
                re.get(bad[0]),
                pe.get(bad[0])
            );
            compared += 1;
        }
        println!(
            "k2 {name}: {} port nodes, {compared} cb-named nodes paired, all bit-identical",
            port_nodes.len()
        );
    }
}
