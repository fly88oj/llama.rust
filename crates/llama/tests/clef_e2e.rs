//! clef (a7b94df2c src/models/clef.cpp, upstream 99b95488c) — the decision
//! model: a memory-less qwen35 trunk + the joint decision head over
//! question/option spans.
//!
//! The synthetic file keeps the trunk small but covers both layer kinds
//! (GDN zero-state + gated full attention over the causal no-cache mask) and
//! both head block kinds (routing cross-only + joint self/cross).
//!
//! Acceptance: `clef_scores_match_reference` compares the port's whole-model
//! scores ([1, n_tokens] decision output, row i = option i) against the NEW
//! reference's embeddings dump (`parity/gen_clef_ref.sh` →
//! parity/clef_scores_ref.bin, driven by `llama_process` +
//! `llama_batch_ext_set_decision_order` — the same public path the reference
//! server's /v1/systemone uses).

use std::io::Write as _;
use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::{Context, GgmlType, Gguf, GgufType, Value};
use llama::clef::{ClefModelWeights, ClefState};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};
use memmap2::Mmap;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const N_VOCAB: i64 = 32000;
const OUT: &str = "/tmp/clef-synth/clef-synth.gguf";
const REF_BIN: &str = "parity/clef_scores_ref.bin";

// the trunk: n_embd 64, 4 heads (2 kv), head 16, ffn 96
const N_LAYER: i64 = 2;
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const HEAD_KV: i64 = 2;
const KLEN: i64 = 16;
const N_FF: i64 = 96;
// the GDN geometry: head_v 8 == d_state, 4 v-heads (dt_rank), 1 k-group
const SSM_CONV: i64 = 4;
const SSM_INNER: i64 = 32;
const SSM_STATE: i64 = 8;
const SSM_DT_RANK: i64 = 4;
const SSM_GROUP: i64 = 1;
// the head: 32 wide, 2 attention heads, 1 routing + 1 joint block
const N_EMBD_H: i64 = 32;
const N_HEAD_DEC: i64 = 2;
const N_ROUTING: i64 = 1;
const N_JOINT: i64 = 1;
const N_FF_H: i64 = 48;

// ---------------------------------------------------------------------------
// the synthetic file
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

fn tensors_for() -> Vec<(String, Vec<i64>, f32)> {
    let mut v: Vec<(String, Vec<i64>, f32)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $scale:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $scale))
        };
    }
    let (head_k, n_k, head_v, n_v) = (SSM_STATE, SSM_GROUP, SSM_STATE, SSM_DT_RANK);
    let value_dim = head_v * n_v;
    let key_dim = head_k * n_k;
    let conv_dim = key_dim * 2 + value_dim;

    push!("token_embd.weight", vec![N_EMBD, N_VOCAB], 0.11);
    push!("output.weight", vec![N_EMBD, N_VOCAB], 0.11);
    push!("output_norm.weight", vec![N_EMBD], 0.2);

    for i in 0..N_LAYER {
        let recr = i == 0;
        push!(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], 0.2);
        push!(format!("blk.{i}.post_attention_norm.weight"), vec![N_EMBD], 0.2);
        if recr {
            push!(format!("blk.{i}.attn_qkv.weight"), vec![N_EMBD, conv_dim], 0.11);
            push!(format!("blk.{i}.attn_gate.weight"), vec![N_EMBD, value_dim], 0.11);
            push!(format!("blk.{i}.ssm_conv1d.weight"), vec![SSM_CONV, conv_dim], 0.2);
            push!(format!("blk.{i}.ssm_dt.bias"), vec![SSM_DT_RANK], 0.2);
            push!(format!("blk.{i}.ssm_a"), vec![SSM_DT_RANK], 0.2);
            push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, n_v], 0.11);
            push!(format!("blk.{i}.ssm_alpha.weight"), vec![N_EMBD, n_v], 0.11);
            push!(format!("blk.{i}.ssm_norm.weight"), vec![head_v], 0.2);
            push!(format!("blk.{i}.ssm_out.weight"), vec![value_dim, N_EMBD], 0.11);
        } else {
            let q = N_HEAD * KLEN * 2; // [q|gate]
            let kv = HEAD_KV * KLEN;
            push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, q], 0.11);
            push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * KLEN, N_EMBD], 0.11);
            push!(format!("blk.{i}.attn_q_norm.weight"), vec![KLEN], 0.2);
            push!(format!("blk.{i}.attn_k_norm.weight"), vec![KLEN], 0.2);
        }
        push!(format!("blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], 0.11);
        push!(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], 0.11);
        push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], 0.11);
    }

    // the decision head: routing blocks first, then joint blocks (clef.cpp:
    // 66-87)
    for i in 0..N_ROUTING + N_JOINT {
        let routing = i < N_ROUTING;
        if !routing {
            push!(format!("dec.blk.{i}.attn_norm.weight"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.attn_norm.bias"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.attn_q.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
            push!(format!("dec.blk.{i}.attn_q.bias"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.attn_k.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
            push!(format!("dec.blk.{i}.attn_k.bias"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.attn_v.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
            push!(format!("dec.blk.{i}.attn_v.bias"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.attn_o.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
            push!(format!("dec.blk.{i}.attn_o.bias"), vec![N_EMBD_H], 0.2);
        } else {
            push!(format!("dec.blk.{i}.cross_attn_norm_kv.weight"), vec![N_EMBD_H], 0.2);
            push!(format!("dec.blk.{i}.cross_attn_norm_kv.bias"), vec![N_EMBD_H], 0.2);
        }
        push!(format!("dec.blk.{i}.cross_attn_norm.weight"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.cross_attn_norm.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.cross_attn_q.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
        push!(format!("dec.blk.{i}.cross_attn_q.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.cross_attn_k.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
        push!(format!("dec.blk.{i}.cross_attn_k.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.cross_attn_v.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
        push!(format!("dec.blk.{i}.cross_attn_v.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.cross_attn_o.weight"), vec![N_EMBD_H, N_EMBD_H], 0.11);
        push!(format!("dec.blk.{i}.cross_attn_o.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.ffn_norm.weight"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.ffn_norm.bias"), vec![N_EMBD_H], 0.2);
        push!(format!("dec.blk.{i}.ffn_up.weight"), vec![N_EMBD_H, N_FF_H], 0.11);
        push!(format!("dec.blk.{i}.ffn_up.bias"), vec![N_FF_H], 0.2);
        push!(format!("dec.blk.{i}.ffn_down.weight"), vec![N_FF_H, N_EMBD_H], 0.11);
        push!(format!("dec.blk.{i}.ffn_down.bias"), vec![N_EMBD_H], 0.2);
    }

    push!("decision.hidden_norm.weight", vec![N_EMBD], 0.2);
    push!("decision.hidden_norm.bias", vec![N_EMBD], 0.2);
    push!("decision.option_summary_norm.weight", vec![N_EMBD_H], 0.2);
    push!("decision.option_summary_norm.bias", vec![N_EMBD_H], 0.2);
    push!("decision.field_norm.weight", vec![N_EMBD_H], 0.2);
    push!("decision.field_norm.bias", vec![N_EMBD_H], 0.2);
    push!("decision.option_norm.weight", vec![N_EMBD_H], 0.2);
    push!("decision.option_norm.bias", vec![N_EMBD_H], 0.2);

    push!("decision.proj_memory.weight", vec![N_EMBD, N_EMBD_H], 0.11);
    push!("decision.proj_question.weight", vec![N_EMBD, N_EMBD_H], 0.11);
    push!("decision.proj_option_question.weight", vec![N_EMBD, N_EMBD_H], 0.11);
    push!("decision.proj_global.weight", vec![N_EMBD, N_EMBD_H], 0.11);
    push!("decision.proj_option_context.weight", vec![N_EMBD, N_EMBD_H], 0.11);
    push!("decision.proj_option_lexical.weight", vec![N_EMBD, N_EMBD_H], 0.11);

    // prior scale, joint scale, residual gate — around 1 so the scores are
    // non-trivial
    push!("decision.scales", vec![3], 0.75);
    push!("token_types.weight", vec![N_EMBD_H, 3], 0.11);

    push!("decision.scorer.weight", vec![4 * N_EMBD_H, N_EMBD_H], 0.11);
    push!("decision.scorer.bias", vec![N_EMBD_H], 0.2);
    push!("decision.scorer_out.weight", vec![N_EMBD_H, 1], 0.11);
    push!("decision.scorer_out.bias", vec![1], 0.2);

    v
}

fn build_file(path: &str) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).expect("mkdir");
    }

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "clef";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-clef".to_string()));
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(HEAD_KV as u32));
    kv!(format!("{a}.attention.key_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(
        format!("{a}.rope.dimension_sections"),
        Value::Array(
            GgufType::Int32,
            vec![Value::I32(8), Value::I32(4), Value::I32(4), Value::I32(0)]
        )
    );
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(SSM_CONV as u32));
    kv!(format!("{a}.ssm.inner_size"), Value::U32(SSM_INNER as u32));
    kv!(format!("{a}.ssm.state_size"), Value::U32(SSM_STATE as u32));
    kv!(format!("{a}.ssm.time_step_rank"), Value::U32(SSM_DT_RANK as u32));
    kv!(format!("{a}.ssm.group_count"), Value::U32(SSM_GROUP as u32));
    // layer 0 GDN, layer 1 attention
    kv!(
        format!("{a}.attention.recurrent_layers"),
        Value::Array(
            GgufType::Uint32,
            vec![Value::U32(1), Value::U32(0)]
        )
    );
    // the decision head (clef.cpp:10-25)
    kv!(format!("{a}.decision.routing_block_count"), Value::U32(N_ROUTING as u32));
    kv!(format!("{a}.decision.block_count"), Value::U32(N_JOINT as u32));
    kv!(format!("{a}.decision.head_count"), Value::U32(N_HEAD_DEC as u32));
    kv!(format!("{a}.attention.layer_norm_epsilon"), Value::F32(1e-5));
    // one token type per question type (clef.cpp:110-113)
    w.set_kv("tokenizer.ggml.token_type_count", Value::U32(3));

    let table = tensors_for();
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0x_c1ef_0001);
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
}

#[test]
#[ignore = "writer only — the file lands in /tmp for the reference probe"]
fn clef_write_synth_file() {
    build_file(OUT);
}

// ---------------------------------------------------------------------------
// the port side
// ---------------------------------------------------------------------------

/// the probe batch: [NONE×3, QUESTION_CHOICE×3, OPTION×3, QUESTION_SCORE×2,
/// OPTION×1] — two questions, two options (clef_get_spans' canonical shape)
const TOKENS: [i32; 12] = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 51];
const ORDERS: [i32; 12] = [
    llama::batch::DECISION_ORDER_NONE,
    llama::batch::DECISION_ORDER_NONE,
    llama::batch::DECISION_ORDER_NONE,
    llama::batch::DECISION_ORDER_QUESTION_CHOICE,
    llama::batch::DECISION_ORDER_QUESTION_CHOICE,
    llama::batch::DECISION_ORDER_QUESTION_CHOICE,
    llama::batch::DECISION_ORDER_OPTION,
    llama::batch::DECISION_ORDER_OPTION,
    llama::batch::DECISION_ORDER_OPTION,
    llama::batch::DECISION_ORDER_QUESTION_SCORE,
    llama::batch::DECISION_ORDER_QUESTION_SCORE,
    llama::batch::DECISION_ORDER_OPTION,
];

fn attn_of(hp: &llama::hparams::LlamaHparams) -> AttnParams {
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
        use_flash_attn: false,
    }
}

fn run_port(path: &str) -> Vec<f32> {
    let file = std::fs::File::open(path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let mut model = load_model(&gguf, mmap.clone()).expect("load model");
    assert_eq!(model.arch, llama::arch::LlmArch::CLEF);

    let params = model.clef_params(attn_of(&model.hparams));
    let weights: ClefModelWeights = model.clef_weights();
    let state = ClefState { weights, params };
    let mut gctx: Context = std::mem::replace(&mut model.ctx, Context::new());
    let watermark = gctx.mark();
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let step = state
        .decode(&mut gctx, watermark, &TOKENS, &pos, Some(&ORDERS))
        .expect("clef decode");
    assert_eq!(step.n_options, 2, "two options in the probe batch");
    step.scores
}

/// the acceptance: the port's [1, n_tokens] decision output is bit-identical
/// to the NEW reference's embeddings dump
#[test]
fn clef_scores_match_reference() {
    let Ok(bytes) = std::fs::read(REF_BIN) else {
        eprintln!("{REF_BIN} missing — run parity/gen_clef_ref.sh first");
        return;
    };
    assert_eq!(&bytes[..8], b"CLEFSC1\0", "magic");
    let n_tokens = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n_options = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(n_tokens, TOKENS.len());
    assert_eq!(n_options, 2);

    if !std::path::Path::new(OUT).exists() {
        build_file(OUT);
    }
    let port = run_port(OUT);
    assert_eq!(port.len(), n_tokens, "the padded [1, n_tokens] scores");

    let ref_v: Vec<f32> = bytes[16..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(ref_v.len(), n_tokens);

    let mut diffs = 0usize;
    for (i, (a, b)) in port.iter().zip(ref_v.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            diffs += 1;
            if diffs <= 4 {
                eprintln!("score[{i}]: port {a:.6e} vs ref {b:.6e}");
            }
        }
    }
    assert_eq!(diffs, 0, "{diffs}/{} score rows differ from the reference", n_tokens);
}

/// the spans reader (clef.cpp:150-180) — the canonical shape of the probe
/// batch plus the invalid-input fallbacks
#[test]
fn clef_get_spans_shapes() {
    use llama::batch as b;
    let s = llama::clef::clef_get_spans(Some(&ORDERS), 12);
    assert!(s.valid);
    assert_eq!(s.questions.len(), 2);
    assert_eq!(s.options.len(), 2);
    assert_eq!(s.questions[0], (b::DECISION_ORDER_QUESTION_CHOICE - b::DECISION_ORDER_QUESTION_NOUL, 3, 6));
    assert_eq!(s.options[0], (0, 6, 9));
    assert_eq!(s.questions[1], (b::DECISION_ORDER_QUESTION_SCORE - b::DECISION_ORDER_QUESTION_NOUL, 9, 11));
    assert_eq!(s.options[1], (1, 11, 12));

    // no decision order → one empty question + one empty option, invalid
    let none = llama::clef::clef_get_spans(None, 4);
    assert!(!none.valid);
    assert_eq!(none.questions.len(), 1);
    assert_eq!(none.options.len(), 1);

    // an option with no question before it → invalid
    let orphan = [b::DECISION_ORDER_OPTION, b::DECISION_ORDER_NONE];
    let s = llama::clef::clef_get_spans(Some(&orphan), 2);
    assert!(!s.valid);

    // a question with no option → invalid
    let lonely = [b::DECISION_ORDER_QUESTION_NOUL, b::DECISION_ORDER_NONE];
    let s = llama::clef::clef_get_spans(Some(&lonely), 2);
    assert!(!s.valid);
}

/// the batch-3 upstream guards (clef.cpp:126-165 @c35b66744): the
/// token-presence half of `ok` (:128 — an embd-only batch has no ids for
/// the head to read) and the mixed-batch media walk (:159-162 — "the head
/// reads the token ids of the spans, they cannot be embeddings": an
/// EMBD-typed row inside a span invalidates, an EMBD row OUTSIDE every
/// span (order NONE) does not)
#[test]
fn clef_get_spans_mixed_batch_guards() {
    use llama::batch as b;
    // the canonical order: [NONE×3, QUESTION×3, OPTION×3, QUESTION×2,
    // OPTION×1] (clef_get_spans_shapes' ORDERS)
    let valid = llama::clef::clef_get_spans_ubatch(Some(&ORDERS), None, true, 12);
    assert!(valid.valid, "a pure token batch keeps the canonical spans");

    // an embd-only batch (no token ids) → degenerate + invalid (:128's
    // `ubatch.token != nullptr` half)
    let embd_only = llama::clef::clef_get_spans_ubatch(Some(&ORDERS), None, false, 12);
    assert!(!embd_only.valid);
    assert_eq!(embd_only.questions.len(), 1);
    assert_eq!(embd_only.options.len(), 1);

    // a mixed batch whose EMBD rows sit inside the OPTION span (batch idx
    // 6..8 typed 1) → the media walk invalidates (:159-162)
    let mut types = [0i8; 12];
    types[6] = 1;
    types[7] = 1;
    let mixed_in_span =
        llama::clef::clef_get_spans_ubatch(Some(&ORDERS), Some(&types), true, 12);
    assert!(!mixed_in_span.valid, "an EMBD row inside a span invalidates");
    assert_eq!(mixed_in_span.questions.len(), 1, "the degenerate fallback");

    // the same EMBD rows OUTSIDE every span (order NONE, idx 0..2) stay
    // usable — `!ubatch.type[i] || decision_order[i] == NONE` (:161)
    let mut types_ok = [0i8; 12];
    types_ok[0] = 1;
    types_ok[1] = 1;
    let mixed_outside =
        llama::clef::clef_get_spans_ubatch(Some(&ORDERS), Some(&types_ok), true, 12);
    assert!(mixed_outside.valid, "EMBD rows at order NONE do not invalidate");
    assert_eq!(mixed_outside.questions.len(), 2);
    assert_eq!(mixed_outside.options.len(), 2);
}

/// the no_tokens fallback of `input_decision::set_input` (clef.cpp:186-
/// 187): an embd-only batch 0-fills the head's token input — the inputs
/// build degenerately (one question/option) and the fill lands as 0s
#[test]
fn clef_decision_inputs_no_tokens_fallback() {
    let mut gctx = ggml::Context::new();
    let n = 6usize;
    // an embd-only batch: no ids, no order — the guards degenerate
    let dec = llama::clef::ClefDecisionInputs::build_ubatch(
        &mut gctx,
        n,
        None,
        None,
        false,
    );
    assert_eq!(dec.n_questions, 1);
    assert_eq!(dec.n_options, 1);
    dec.set_input_ubatch(&mut gctx, &[], None, None);
    let ids: Vec<i32> = bytemuck::cast_slice(gctx.data_bytes(dec.tokens).unwrap()).to_vec();
    assert_eq!(ids, vec![0i32; n], "the no_tokens fallback 0-fills");

    // the token-batch form still lands the ids verbatim
    let dec2 = llama::clef::ClefDecisionInputs::build(&mut gctx, n, None);
    dec2.set_input(&mut gctx, &[7, 8, 9, 10, 11, 12], None);
    let ids2: Vec<i32> = bytemuck::cast_slice(gctx.data_bytes(dec2.tokens).unwrap()).to_vec();
    assert_eq!(ids2, vec![7, 8, 9, 10, 11, 12]);
}

/// in-port smoke (no reference needed): the scores are finite and distinct
#[test]
fn clef_synth_smoke() {
    if !std::path::Path::new(OUT).exists() {
        build_file(OUT);
    }
    let scores = run_port(OUT);
    assert_eq!(scores.len(), 12);
    assert!(scores.iter().all(|s| s.is_finite()), "all scores finite");
    assert!(scores[0] != scores[1], "the two options differ");
}
