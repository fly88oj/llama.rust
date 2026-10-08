//! saver_e2e.rs — real-model verification of `llama::saver` (the port of
//! llama-model-saver.cpp) on the qwen2.5-0.5b anchor file.
//!
//! Runs by default when the model exists on this machine:
//!   * `saves_and_round_trips_qwen25` — load → save → reload the saved file →
//!     decode the anchor prompt with both models → bit-identical logits; plus
//!     structural checks of the written KV (scalar collapse of the per-layer
//!     arrays, the tokenizer tables, the tensor order).
//!   * `byte_identical_to_reference_saver` — byte comparison against the
//!     pinned reference's own `llama_model_save_to_file` output; the artifact
//!     is produced by parity/gen_model_saver_ref.sh (the C probe drives
//!     libllama.so with `use_extra_bufts = false` — see the script header)
//!     and passed via `LLAMA_SAVER_REF`.

use std::path::Path;
use std::sync::Arc;

use ggml::{Gguf, Value};
use llama::context::DecodeContext;
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};
use llama::saver;
use llama::vocab::Vocab;
use memmap2::Mmap;

const QWEN25: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const PROMPT: &str = "The capital of France is";

fn open_model(path: &str) -> Option<(Gguf, Arc<Mmap>)> {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).unwrap();
    // SAFETY: read-only usage of a model file
    let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    Some((gguf, mmap))
}

/// `ModelWeights` for qwen2 (the arch_e2e.rs helper).
fn qwen2_weights(m: &LlamaModel) -> llama::graph::ModelWeights {
    use llama::graph::{LayerWeights, ModelWeights};
    ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .map(|x| LayerWeights {
                attn_norm: x.attn_norm.expect("attn_norm"),
                wq: x.wq.expect("wq"),
                wk: x.wk.expect("wk"),
                wv: x.wv.expect("wv"),
                wo: x.wo.expect("wo"),
                wq_b: x.wq_b,
                wk_b: x.wk_b,
                wv_b: x.wv_b,
                ffn_norm: x.ffn_norm.expect("ffn_norm"),
                ffn_gate: x.ffn_gate.expect("ffn_gate"),
                ffn_down: x.ffn_down.expect("ffn_down"),
                ffn_up: x.ffn_up.expect("ffn_up"),
            })
            .collect(),
    }
}

/// `AttnParams` from loaded hparams (ctx_shift_e2e.rs's helper, FA off).
fn attn_params(m: &LlamaModel) -> AttnParams {
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
        use_flash_attn: false,
    }
}

fn decode_logits(model_path: &str) -> Vec<f32> {
    let (gguf, mmap) = open_model(model_path).expect("model");
    let vocab = Vocab::load(&gguf).unwrap();
    let mut model = load_model(&gguf, mmap).unwrap();
    let tokens = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..tokens.len() as i32).collect();
    let weights = qwen2_weights(&model);
    let attn = attn_params(&model);
    let gctx = std::mem::replace(&mut model.ctx, ggml::Context::new());
    let mut ctx = DecodeContext::new(gctx, weights, attn, 512, 8, 512);
    ctx.decode_all(&tokens, &pos).unwrap()
}

fn matches_value(v: &Option<Value>, want: &Value) -> bool {
    use std::mem::discriminant;
    match (v, want) {
        (Some(Value::U32(a)), Value::U32(b)) => a == b,
        (Some(Value::F32(a)), Value::F32(b)) => a.to_bits() == b.to_bits(),
        (Some(Value::String(a)), Value::String(b)) => a == b,
        (Some(Value::Array(t, x)), Value::Array(t2, y)) => {
            discriminant(t) == discriminant(t2) && x.len() == y.len()
        }
        _ => false,
    }
}

#[test]
fn saves_and_round_trips_qwen25() {
    let Some((gguf, mmap)) = open_model(QWEN25) else { return };
    let vocab = Vocab::load(&gguf).unwrap();
    let model = load_model(&gguf, mmap).unwrap();

    // supports_arch (llama-model-saver.cpp:16-27)
    assert!(saver::supports_arch(model.arch));
    assert!(!saver::supports_arch(llama::arch::LlmArch::T5));
    assert!(!saver::supports_arch(llama::arch::LlmArch::GEMMA3N));
    assert!(!saver::supports_arch(llama::arch::LlmArch::STEP35));

    let out = std::env::temp_dir().join("llama_rust_saver_qwen25.gguf");
    saver::save_model_to_file(&model, &vocab, out.to_str().unwrap()).expect("save");

    // structural checks of the written file
    let saved = Gguf::open(&out).unwrap();
    let get = |k: &str| saved.kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
    // every file tensor is present
    assert_eq!(saved.tensors.len(), gguf.tensors.len(), "tensor count");
    for t in &gguf.tensors {
        let st = saved.find_tensor(&t.name).unwrap_or_else(|| panic!("{}", t.name));
        assert_eq!(st.ty, t.ty, "{} type", t.name);
        assert_eq!(st.ne, t.ne, "{} shape", t.name);
    }
    // the byte payload of every tensor is the file's original bytes
    for t in &gguf.tensors {
        assert_eq!(
            saved.tensor_data(&t.name).unwrap(),
            gguf.tensor_data(&t.name).unwrap(),
            "{} payload",
            t.name
        );
    }
    // the per-layer arrays collapsed to scalars (llama-model-saver.cpp:84-96)
    assert!(matches_value(&get("qwen2.attention.head_count"), &Value::U32(14)));
    assert!(matches_value(&get("qwen2.attention.head_count_kv"), &Value::U32(2)));
    assert!(matches_value(&get("qwen2.feed_forward_length"), &Value::U32(4864)));
    // core hparams
    assert!(matches_value(&get("qwen2.context_length"), &Value::U32(32768)));
    assert!(matches_value(&get("qwen2.embedding_length"), &Value::U32(896)));
    assert!(matches_value(&get("qwen2.block_count"), &Value::U32(24)));
    assert!(matches_value(&get("general.architecture"), &Value::String("qwen2".into())));
    assert!(matches_value(&get("general.name"), &Value::String("qwen2.5-0.5b-instruct".into())));
    // vocab tables: 151936 tokens / scores / i32 token types, empty merges
    // array for the BPE... qwen2.5 IS BPE — merges must be carried
    match get("tokenizer.ggml.tokens") {
        Some(Value::Array(t, items)) => {
            assert_eq!(t, ggml::GgufType::String);
            assert_eq!(items.len(), 151936);
        }
        v => panic!("tokenizer.ggml.tokens: {v:?}"),
    }
    assert!(matches!(
        get("tokenizer.ggml.token_type"),
        Some(Value::Array(ggml::GgufType::Int32, _))
    ));
    assert!(matches!(
        get("tokenizer.ggml.scores"),
        Some(Value::Array(ggml::GgufType::Float32, _))
    ));
    // BPE: the merges array is written (non-empty for gpt2-family vocabs)
    assert!(matches!(
        get("tokenizer.ggml.merges"),
        Some(Value::Array(ggml::GgufType::String, _))
    ));
    // rope: no scaling KV in the file -> the loader's "linear" default
    // (llama-model.cpp:1350) and factor 0.0 (freq_scale_train == 1.0,
    // llama-model-saver.cpp:345)
    assert!(matches_value(&get("qwen2.rope.scaling.type"), &Value::String("linear".into())));
    assert!(matches_value(&get("qwen2.rope.scaling.factor"), &Value::F32(0.0)));

    // round trip: decode with the original and the saved file — bit-identical
    let orig = decode_logits(QWEN25);
    let saved_logits = decode_logits(out.to_str().unwrap());
    assert_eq!(orig.len(), saved_logits.len());
    assert_eq!(orig, saved_logits, "round-trip logits differ");
}

/// Byte comparison against the reference's own saver output
/// (parity/gen_model_saver_ref.sh produces the artifact).
#[test]
fn byte_identical_to_reference_saver() {
    let Some(ref_path) = std::env::var_os("LLAMA_SAVER_REF") else {
        eprintln!("skipping: LLAMA_SAVER_REF not set (run parity/gen_model_saver_ref.sh)");
        return;
    };
    let Some((gguf, mmap)) = open_model(QWEN25) else { return };
    let vocab = Vocab::load(&gguf).unwrap();
    let model = load_model(&gguf, mmap).unwrap();
    let out = std::env::temp_dir().join("llama_rust_saver_qwen25_cmp.gguf");
    saver::save_model_to_file(&model, &vocab, out.to_str().unwrap()).expect("save");

    let mine = std::fs::read(&out).unwrap();
    let reference = std::fs::read(&ref_path).unwrap();
    assert_eq!(mine.len(), reference.len(), "file size differs");
    let diffs: Vec<usize> = mine
        .iter()
        .zip(&reference)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    // The one accepted divergence: tokenizer.ggml.eot_token_id. The C's EOT
    // auto-detection scans token_to_id (a libstdc++ unordered_map,
    // llama-vocab.cpp:1381/:2682) and on this build picks "<|im_end|>"
    // (151645) over "<|endoftext|>" (151643); the port's vocab load scans ids
    // ascending and picks the other (vocab.rs:863 documents the divergence).
    // Every other byte of the 492,021,632-byte file is identical; fixing this
    // byte means replicating the libstdc++ unordered_map iteration order in
    // the vocab loader (the same treatment std_sort_by got), outside this
    // change's scope.
    assert_eq!(diffs.len(), 1, "expected exactly the eot byte to differ: {diffs:?}");
    assert_eq!(
        u32::from_le_bytes(mine[diffs[0]..diffs[0] + 4].try_into().unwrap()),
        151643,
        "the differing value must be the port's eot (<|endoftext|>)"
    );
    assert_eq!(
        u32::from_le_bytes(reference[diffs[0]..diffs[0] + 4].try_into().unwrap()),
        151645,
        "the reference value must be its eot (<|im_end|>)"
    );
}
