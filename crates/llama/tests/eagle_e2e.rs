//! eagle_e2e.rs — the EAGLE3 speculative-draft port (speculative.cpp:426-907
//! + src/models/eagle3.cpp @ bd4f514db1) on synthetic GGUFs, following the
//! MTP trilogy's protocol (tests/mtp_e2e.rs):
//!
//!   * the generator writes a **llama-arch target** (the only target arch
//!     whose graph records the per-layer input tap both sides need —
//!     llama.cpp:127 `res->t_layer_inp[il] = inpL`) and two eagle3 heads over
//!     it: `head-own` carries its own token_embd + output (eagle3.cpp:69-75),
//!     `head-other` omits them so both sides must inherit the target's
//!     tensors (the C through `cparams.ctx_other`, llama-context.cpp:156-163;
//!     the port as external-storage tensors backed by the target's mmap,
//!     eagle.rs);
//!   * default tests: head-loader pinning, target-trunk-unchanged (the taps
//!     must not alter the trunk logits — both FA modes), and the full
//!     `--spec-type draft-eagle3` driver — the committed stream must equal
//!     the plain greedy stream at `temperature 0`, with the drafted/accepted
//!     counters reported;
//!   * `#[ignore]d eagle_write_synth_files` writes the three files for
//!     `parity/eagle_parity.sh`, whose reference side is a fresh
//!     `llama-server --spec-type draft-eagle3 -md head.gguf` (first
//!     /completion, temperature 0, cache_prompt=false) plus its server-log
//!     draft stats (`draft acceptance = ...`, server-context.cpp:677-678).
//!
//! Like the MTP trilogy: the synthetic head is not trained — its weights are
//! a deterministic RNG stream, so drafts are a pseudo-random chain and the
//! acceptance rate is near zero. The parity contract that still binds: the
//! drafted token chain (the head's argmax sequence through encoder + decoder
//! + the deferred-boundary bookkeeping) and the drafted/accepted counters
//! must match the reference's, and the committed stream must equal plain
//! greedy on both sides.

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::{Context, Gguf, Value};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{self};
use llama::model::{load_model, LlamaModel};
use llama::sampling::{SamplingContext, SamplingParams};
use llama::speculative::{
    common_speculative_init, speculative_simple_generate, CommonParamsSpeculative,
    CommonSpeculativeType,
};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/arch-eagle";

// the shared geometry: the target (llama arch) and the head use the same
// hidden size / heads so the g rows and the ctx_other tensors line up
const N_LAYER_TGT: usize = 4;
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HEAD_DIM: i64 = 16; // N_EMBD / N_HEAD
const N_ROT: i64 = 16;
const N_FF: i64 = 96;
const N_CTX: u32 = 256;
/// extract_layers = [1, 2, 3] — all < n_layer (the == n_layer case needs the
/// target's trunk nextn tap, which the llama arch does not set)
const TARGET_LAYERS: [i32; 3] = [1, 2, 3];
const N_EMBD_INP_ENC: i64 = 3 * N_EMBD;

#[derive(Clone, Copy, PartialEq, Debug)]
enum HeadKind {
    /// own token_embd + output in the head file
    Own,
    /// no token_embd / output — inherit the target's (the ctx_other path)
    Other,
}

fn target_path() -> String {
    format!("{OUT_DIR}/llama-synth-eagle-tgt.gguf")
}
fn head_path(kind: HeadKind) -> String {
    format!(
        "{OUT_DIR}/eagle3-synth-{}.gguf",
        match kind {
            HeadKind::Own => "own",
            HeadKind::Other => "other",
        }
    )
}

// ---------------------------------------------------------------------------
// the writers (the mtp_e2e recipe: one fixed RNG stream in table order)
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

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Proj,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn start_writer(name: &str) -> GgufWriter {
    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }
    w.set_kv("general.name", Value::String(name.to_string()));
    w.set_kv("general.file_type", Value::U32(0)); // F32
    w
}

fn write_file(mut w: GgufWriter, tensors: &[(String, Vec<i64>, Role)], path: &str, seed: u64) {
    let mut rng = Rng(seed);
    let mut data: Vec<Vec<u8>> = Vec::new();
    for (name, ne, role) in tensors {
        let n: i64 = ne.iter().product();
        let s = match role {
            Role::Norm => 1.0,
            Role::Proj => 1.0 / (N_EMBD as f32).sqrt(),
        };
        let vals: Vec<f32> = (0..n)
            .map(|_| match role {
                Role::Norm => 1.0 + 0.05 * rng.next(),
                Role::Proj => s * rng.next(),
            })
            .collect();
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, ggml::types::GgmlType::F32, ne4);
        data.push(f32_bytes(&vals));
    }
    let f = std::fs::File::create(path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
}

/// the llama-arch target — the arch whose graph records `res->t_layer_inp`
/// (llama.cpp:127), the tap the eagle3 impl reads
/// (`llama_get_embeddings_layer_inp`, speculative.cpp:609-611)
fn build_target() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let mut w = start_writer("llama-rust-synth-eagle3-target");
    let a = "llama";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(N_CTX));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    w.set_kv(&format!("{a}.block_count"), Value::U32(N_LAYER_TGT as u32));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    w.set_kv(
        &format!("{a}.attention.head_count"),
        Value::U32(N_HEAD as u32),
    );
    w.set_kv(
        &format!("{a}.attention.head_count_kv"),
        Value::U32(N_HEAD_KV as u32),
    );
    w.set_kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5),
    );
    w.set_kv(
        &format!("{a}.rope.dimension_count"),
        Value::U32(N_ROT as u32),
    );
    w.set_kv(&format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    let n_gqa_k = HEAD_DIM * N_HEAD_KV;
    let mut t: Vec<(String, Vec<i64>, Role)> = vec![
        (
            "token_embd.weight".into(),
            vec![N_EMBD, N_VOCAB],
            Role::Proj,
        ),
        ("output_norm.weight".into(), vec![N_EMBD], Role::Norm),
        // an explicit lm head: the `head-other` variant inherits it through
        // ctx_other (eagle3.cpp:296-304)
        ("output.weight".into(), vec![N_EMBD, N_VOCAB], Role::Proj),
    ];
    for i in 0..N_LAYER_TGT as i32 {
        t.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![N_EMBD],
            Role::Norm,
        ));
        t.push((
            format!("blk.{i}.attn_q.weight"),
            vec![N_EMBD, HEAD_DIM * N_HEAD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_k.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_v.weight"),
            vec![N_EMBD, n_gqa_k],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.attn_output.weight"),
            vec![HEAD_DIM * N_HEAD, N_EMBD],
            Role::Proj,
        ));
        t.push((format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm));
        t.push((
            format!("blk.{i}.ffn_gate.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_down.weight"),
            vec![N_FF, N_EMBD],
            Role::Proj,
        ));
        t.push((
            format!("blk.{i}.ffn_up.weight"),
            vec![N_EMBD, N_FF],
            Role::Proj,
        ));
    }
    write_file(w, &t, &target_path(), 0xEA61_E000u64.wrapping_add(1));
}

/// the eagle3 head (eagle3.cpp:38-101's tensor table); `kind` selects whether
/// token_embd / output are written into the head file
fn build_head(kind: HeadKind) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let mut w = start_writer("llama-rust-synth-eagle3-head");
    let a = "eagle3";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(N_CTX));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    // exactly one decoder layer (eagle3.cpp:156)
    w.set_kv(&format!("{a}.block_count"), Value::U32(1));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    w.set_kv(
        &format!("{a}.attention.head_count"),
        Value::U32(N_HEAD as u32),
    );
    w.set_kv(
        &format!("{a}.attention.head_count_kv"),
        Value::U32(N_HEAD_KV as u32),
    );
    w.set_kv(
        &format!("{a}.attention.key_length"),
        Value::U32(HEAD_DIM as u32),
    );
    w.set_kv(
        &format!("{a}.attention.value_length"),
        Value::U32(HEAD_DIM as u32),
    );
    w.set_kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5),
    );
    w.set_kv(
        &format!("{a}.rope.dimension_count"),
        Value::U32(N_ROT as u32),
    );
    w.set_kv(&format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // eagle3.cpp:6-11 — exactly 3 extract layers
    w.set_kv(
        &format!("{a}.target_layers"),
        Value::Array(
            ggml::GgufType::Int32,
            vec![
                Value::I32(TARGET_LAYERS[0]),
                Value::I32(TARGET_LAYERS[1]),
                Value::I32(TARGET_LAYERS[2]),
            ],
        ),
    );
    // eagle3.cpp:19
    w.set_kv(
        &format!("{a}.target_hidden_size"),
        Value::U32(N_EMBD as u32),
    );

    let n_gqa_k = HEAD_DIM * N_HEAD_KV;
    let attn_input = 2 * N_EMBD;
    let mut t: Vec<(String, Vec<i64>, Role)> = vec![
        // d2t: draft to target vocabulary mapping — omitted (same vocab)
        // feature fusion (eagle3.cpp:58)
        ("fc.weight".into(), vec![N_EMBD_INP_ENC, N_EMBD], Role::Proj),
        ("output_norm.weight".into(), vec![N_EMBD], Role::Norm),
    ];
    if let HeadKind::Own = kind {
        t.push((
            "token_embd.weight".into(),
            vec![N_EMBD, N_VOCAB],
            Role::Proj,
        ));
        t.push(("output.weight".into(), vec![N_EMBD, N_VOCAB], Role::Proj));
    }
    // the single decoder layer (eagle3.cpp:77-100)
    t.push(("blk.0.attn_norm.weight".into(), vec![N_EMBD], Role::Norm));
    t.push(("blk.0.attn_norm_2.weight".into(), vec![N_EMBD], Role::Norm));
    t.push((
        "blk.0.attn_q.weight".into(),
        vec![attn_input, HEAD_DIM * N_HEAD],
        Role::Proj,
    ));
    t.push((
        "blk.0.attn_k.weight".into(),
        vec![attn_input, n_gqa_k],
        Role::Proj,
    ));
    t.push((
        "blk.0.attn_v.weight".into(),
        vec![attn_input, n_gqa_k],
        Role::Proj,
    ));
    t.push((
        "blk.0.attn_output.weight".into(),
        vec![HEAD_DIM * N_HEAD, N_EMBD],
        Role::Proj,
    ));
    t.push(("blk.0.ffn_norm.weight".into(), vec![N_EMBD], Role::Norm));
    t.push((
        "blk.0.ffn_gate.weight".into(),
        vec![N_EMBD, N_FF],
        Role::Proj,
    ));
    t.push((
        "blk.0.ffn_down.weight".into(),
        vec![N_FF, N_EMBD],
        Role::Proj,
    ));
    t.push(("blk.0.ffn_up.weight".into(), vec![N_EMBD, N_FF], Role::Proj));
    // rope_freqs (optional) omitted

    write_file(
        w,
        &t,
        &head_path(kind),
        0xEA61_E000u64.wrapping_add(2 + kind as u64),
    );
}

/// the tests rewrite the same /tmp files and keep their mmaps alive across
/// loads; serialize whole tests (not just the loads) or a concurrent rewrite
/// of a mapped file is a SIGBUS
fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path).expect("open synth");
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

// ---------------------------------------------------------------------------
// the target trunk driver (the LLAMA arm of llama-cli's forward_weights)
// ---------------------------------------------------------------------------

fn llama_forward(m: &mut LlamaModel, fa: bool) -> (ForwardWeights, AttnParams) {
    let hp = &m.hparams;
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
    let layers = m
        .layers
        .iter()
        .map(|l| graph_arch::LlamaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq.unwrap(),
            wk: l.wk.unwrap(),
            wv: l.wv.unwrap(),
            wo: l.wo.unwrap(),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            wo_b: None,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate.unwrap(),
            ffn_down: l.ffn_down.unwrap(),
            ffn_up: l.ffn_up.unwrap(),
            ffn_gate_b: None,
            ffn_down_b: None,
            ffn_up_b: None,
        })
        .collect();
    (
        ForwardWeights::Llama(graph_arch::LlamaModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            output_b: None,
            layers,
        }),
        attn,
    )
}

fn target_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let (weights, attn) = llama_forward(m, fa);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
}

/// the eagle3 head context (`common_speculative_init_from_params`'s has_draft
/// arm, speculative.cpp:2553-2576)
fn eagle_driver(kind: HeadKind, fa: bool) -> DecodeContext {
    let head_gguf = Gguf::open(&head_path(kind)).expect("open head");
    let head_mmap = {
        let f = std::fs::File::open(&head_path(kind)).unwrap();
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let tgt_gguf = Gguf::open(&target_path()).expect("open target");
    let tgt_mmap = {
        let f = std::fs::File::open(&target_path()).unwrap();
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let n_vocab = Vocab::load(&head_gguf).unwrap().n_tokens();
    let head = llama::eagle::load_eagle3_head(
        &head_gguf,
        head_mmap,
        &tgt_gguf,
        tgt_mmap,
        n_vocab as i64,
        fa,
    )
    .expect("load eagle3 head");
    let stub = llama::eagle::eagle_trunk_stub(&head.weights);
    DecodeContext::new_eagle3(
        head.ctx,
        ForwardWeights::Qwen2(stub),
        (head.weights, head.params),
        n_vocab as usize,
        512,
        8,
        512,
    )
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as i32
}

/// greedy stream of the target (`temperature 0`)
fn plain_greedy(m: &mut LlamaModel, fa: bool, prompt: &[i32], n_predict: usize) -> Vec<i32> {
    let mut d = target_driver(m, fa);
    let mut logits = d
        .decode(prompt, &(0..prompt.len() as i32).collect::<Vec<_>>())
        .expect("prefill")
        .to_vec();
    let mut out = Vec::new();
    for _ in 0..n_predict {
        let id = argmax(&logits);
        out.push(id);
        let p = (prompt.len() + out.len() - 1) as i32;
        logits = d.decode(&[id], &[p]).expect("decode").to_vec();
    }
    out
}

// ---------------------------------------------------------------------------
// default-run tests
// ---------------------------------------------------------------------------

/// the head loads with the pinned geometry: 3 extract layers, the fused fc
/// width, the [2*n_embd] attention input; the ctx_other variant materializes
/// the target's token_embd / output into the head context
#[test]
fn eagle_synth_head_loads() {
    let _files = file_lock();
    build_target();
    for kind in [HeadKind::Own, HeadKind::Other] {
        build_head(kind);
        let head_gguf = Gguf::open(&head_path(kind)).expect("open head");
        let head_mmap = {
            let f = std::fs::File::open(&head_path(kind)).unwrap();
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let tgt_gguf = Gguf::open(&target_path()).unwrap();
        let tgt_mmap = {
            let f = std::fs::File::open(&target_path()).unwrap();
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let n_vocab = Vocab::load(&head_gguf).unwrap().n_tokens();
        let head = llama::eagle::load_eagle3_head(
            &head_gguf,
            head_mmap,
            &tgt_gguf,
            tgt_mmap,
            n_vocab as i64,
            false,
        )
        .expect("head loads");

        assert_eq!(head.target_layer_ids, TARGET_LAYERS);
        assert_eq!(head.params.n_embd, N_EMBD);
        assert_eq!(head.params.n_embd_inp_enc, N_EMBD_INP_ENC);
        assert_eq!(head.params.n_embd_tgt, N_EMBD);
        assert_eq!(
            *head.ctx.ne(head.weights.fc),
            [N_EMBD_INP_ENC, N_EMBD, 1, 1]
        );
        assert_eq!(
            *head.ctx.ne(head.weights.layers[0].wq),
            [2 * N_EMBD, HEAD_DIM * N_HEAD, 1, 1],
            "wq takes the [2*n_embd] concatenated input"
        );
        // the ctx_other tensors: [n_embd_tgt, n_vocab] rows of the target
        assert_eq!(*head.ctx.ne(head.weights.tok_embd), [N_EMBD, N_VOCAB, 1, 1]);
        assert_eq!(*head.ctx.ne(head.weights.output), [N_EMBD, N_VOCAB, 1, 1]);
        assert!(head.weights.d2t.is_none());
        println!(
            "eagle3 head ({:?}) loaded: {} tensors, extract_layers {:?}",
            kind,
            head_gguf.tensors.len(),
            head.target_layer_ids
        );
    }
}

/// the per-layer input taps must not alter the trunk: the target's logits are
/// identical with the taps off and on (all 3 extract layers), both FA modes —
/// the port-side half of parity cell (a). Also pins the tap contents: the
/// buffer rows equal the trunk's residual stream (n_embd wide, one row per
/// token). decode_batch is the driver path that extracts the taps
/// (extract_layer_inputs runs in the decode ubatch loop,
/// llama-context.cpp:2008 — the port's step_ubatch).
#[test]
fn eagle_target_trunk_unchanged() {
    let _files = file_lock();
    build_target();
    let prompt: Vec<i32> = (1..=6).collect();

    for fa in [false, true] {
        // the prefill batch the speculative driver feeds
        // (speculative-simple.cpp:126-139) — last row an output so the logits
        // are comparable
        let batch = {
            let mut b = llama::batch::LlamaBatch::default();
            for (i, &t) in prompt.iter().enumerate() {
                b.add(t, i as i32, &[0], i + 1 == prompt.len());
            }
            b
        };

        // taps off
        let mut m = open_model(&target_path());
        let a = target_driver(&mut m, fa)
            .decode_batch(&batch)
            .expect("plain decode")
            .logits_ith(batch.token.len() as i32 - 1)
            .expect("last row")
            .to_vec();

        // taps on — `llama_set_embeddings_layer_inp(ctx_tgt, lid, true)` for
        // every extract layer (speculative.cpp:514-516)
        let mut m = open_model(&target_path());
        let mut d = target_driver(&mut m, fa);
        for &lid in &TARGET_LAYERS {
            d.set_embeddings_layer_inp(lid as u32, true);
        }
        let b = d
            .decode_batch(&batch)
            .expect("tapped decode")
            .logits_ith(batch.token.len() as i32 - 1)
            .expect("last row")
            .to_vec();

        assert!(
            a == b,
            "llama trunk logits differ with the layer_inp taps on (fa={fa})"
        );
        assert!(a.iter().all(|v| v.is_finite()));
        for &lid in &TARGET_LAYERS {
            let tap = d.get_embeddings_layer_inp(lid as u32);
            assert_eq!(
                tap.len(),
                prompt.len() * N_EMBD as usize,
                "layer {lid} tap rows"
            );
            assert!(tap.iter().all(|v| v.is_finite()), "layer {lid} tap values");
        }
        println!(
            "target trunk-unchanged ok (fa={fa}), greedy first token {}",
            argmax(&a)
        );
    }
}

/// the full `--spec-type draft-eagle3` path on both head variants, both FA
/// modes: the committed stream must equal the target's plain greedy stream
/// (the acceptance criterion), with the drafted/accepted counters reported
#[test]
fn eagle_speculation_matches_plain_greedy() {
    let _files = file_lock();
    build_target();
    for kind in [HeadKind::Own, HeadKind::Other] {
        build_head(kind);
        for fa in [false, true] {
            let prompt: Vec<i32> = (1..=6).collect();
            let n_predict = 12;

            // plain greedy baseline of the target
            let mut m_plain = open_model(&target_path());
            let plain = plain_greedy(&mut m_plain, fa, &prompt, n_predict);

            // the eagle3 driver: target context + head context
            let mut m_tgt = open_model(&target_path());
            let mut tgt = target_driver(&mut m_tgt, fa);
            let ctx_dft = eagle_driver(kind, fa);

            let vocab = Vocab::load(&Gguf::open(&target_path()).unwrap()).unwrap();
            let mut params = CommonParamsSpeculative::default();
            params.types = vec![CommonSpeculativeType::DraftEagle3];
            params.draft.n_max = 3;
            params.draft.p_min = 0.0;

            let mut spec_ctx = common_speculative_init(
                &params,
                1,
                &mut tgt,
                Some(ctx_dft),
                &vocab,
                Some(&vocab),
                0,
                false,
            )
            .expect("init")
            .expect("speculator");

            let n_vocab = tgt.n_vocab() as i32;
            let mut smpl = SamplingContext::new(
                n_vocab,
                SamplingParams {
                    temp: 0.0,
                    ..Default::default()
                },
            );

            let res = speculative_simple_generate(
                &mut tgt,
                &mut spec_ctx,
                &mut smpl,
                &vocab,
                &prompt,
                n_predict as i32,
            )
            .expect("speculative generate");

            // the driver commits whole verify rounds, so it may overshoot
            // n_predict by up to n_max tokens — the requested prefix must
            // match
            assert!(
                res.tokens.len() >= plain.len(),
                "head {:?} fa={fa}: short stream ({})",
                kind,
                res.tokens.len()
            );
            assert_eq!(
                &res.tokens[..plain.len()],
                &plain[..],
                "head {:?} fa={fa}: the eagle3 speculation changed the greedy stream",
                kind
            );
            assert!(
                res.n_drafted > 0,
                "head {:?} fa={fa}: no drafts were generated",
                kind
            );
            println!(
                "eagle3 head={kind:?} fa={fa} — {} tokens, drafted {}, accepted {} ({} target \
                 forwards, {} draft forwards), mean acc len {:.2}",
                res.tokens.len(),
                res.n_drafted,
                res.n_accept,
                res.n_target_forward,
                res.n_draft_forward,
                spec_ctx
                    .impl_stats(0)
                    .map(|s| s.mean_acc_len())
                    .unwrap_or(0.0),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// the #[ignore] generator for the parity runs (parity/eagle_parity.sh drives
// the release llama-cli + the reference llama-server)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "writes /tmp/arch-eagle for parity/eagle_parity.sh"]
fn eagle_write_synth_files() {
    build_target();
    for kind in [HeadKind::Own, HeadKind::Other] {
        build_head(kind);
        let n = Gguf::open(&head_path(kind)).unwrap().tensors.len();
        println!("eagle3 head {:?}: {n} tensors -> {}", kind, head_path(kind));
    }
    println!("target -> {}", target_path());
}

// ---------------------------------------------------------------------------
// batch 18 — the disclosed eagle3 GPU gate flip: `enable_gpu` no longer
// refuses eagle3 setups; the foreign executor now syncs the per-layer input
// taps (`layer_inp-{il}`, the `res->t_layer_inp` store of llama.cpp:127)
// back into their Rust twins the way extract_layer_inputs reads them via
// ggml_backend_tensor_get_async (llama-context.cpp:2265-2290). Without the
// sync a GPU-enabled target panicked in the tap extraction (Storage::None).
// The regression: the eagle3 driver's tap reads on a foreign-executor target
// must match the pure-CPU target's (the foreign *CPU* backend exercises the
// same emission + sync path as the GPU one, no device needed).
// ---------------------------------------------------------------------------

fn eagle_ref_lib_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from("/home/jeffrey/llm/llama.cpp/build-rust-ref/bin");
    dir.join("libggml-base.so").exists().then_some(dir)
}

#[test]
fn eagle3_gpu_tap_sync_matches_cpu() {
    let _files = file_lock();
    build_target();
    const LIDS: [u32; 3] = [1, 2, 3]; // the eagle3 extract layers
    // decode_batch — the ONLY path whose step_ubatch extracts the taps
    // (extract_layer_inputs, llama-context.cpp:2008) AND the path the eagle3
    // driver actually drives; decode_all skips step_ubatch entirely, so a
    // decode_all-based variant compared two EMPTY tap buffers (vacuous —
    // found in the 2026-10-01 resume audit and fixed here)
    let batch = |tokens: &[i32]| {
        let mut b = llama::batch::LlamaBatch::default();
        for (i, &t) in tokens.iter().enumerate() {
            // every row an output so logits_ith(i) covers all 12 positions
            // (llama-batch.cpp:120-131)
            b.add(t, i as i32, &[0], true);
        }
        b
    };
    let prompt: Vec<i32> = (1..=12).collect();

    // ---- the pure-CPU ground truth ----
    let mut m_cpu = open_model(&target_path());
    let mut d_cpu = target_driver(&mut m_cpu, false);
    for &lid in &LIDS {
        d_cpu.set_embeddings_layer_inp(lid, true);
    }
    let logits_cpu = d_cpu
        .decode_batch(&batch(&prompt))
        .expect("cpu decode_batch");
    let taps_cpu: Vec<Vec<f32>> = LIDS
        .iter()
        .map(|&lid| d_cpu.get_embeddings_layer_inp(lid).to_vec())
        .collect();

    // ---- the foreign-executor target (reference CPU backend via DL) ----
    let Some(lib) = eagle_ref_lib_dir() else {
        eprintln!("eagle3_gpu_tap_sync_matches_cpu: reference libs absent, skipping");
        return;
    };
    let mut m_exe = open_model(&target_path());
    let mut d_exe = target_driver(&mut m_exe, false);
    for &lid in &LIDS {
        d_exe.set_embeddings_layer_inp(lid, true);
    }
    let mut cfg = ggml::backend_emit::EmitConfig::new(&lib);
    cfg.device = None; // the foreign CPU backend
    cfg.n_gpu_layers = 0;
    cfg.n_threads = 8;
    d_exe.enable_gpu(cfg).expect("enable_gpu on an eagle3-tapped target");
    let logits_exe = d_exe
        .decode_batch(&batch(&prompt))
        .expect("foreign decode_batch");
    let taps_exe: Vec<Vec<f32>> = LIDS
        .iter()
        .map(|&lid| d_exe.get_embeddings_layer_inp(lid).to_vec())
        .collect();

    // the taps must be REAL rows first (n_tokens * n_embd each — the vacuous
    // variant had 0 rows), then same math through the reference kernels —
    // allow the same last-ulp drift bound the emission test uses
    for (lid, (want, got)) in LIDS.iter().zip(taps_cpu.iter().zip(taps_exe.iter())) {
        assert_eq!(
            want.len(),
            prompt.len() * N_EMBD as usize,
            "tap lid {lid}: expected {} rows",
            prompt.len() * N_EMBD as usize
        );
        assert_eq!(want.len(), got.len(), "tap lid {lid} row count");
        let max_abs = want
            .iter()
            .zip(got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        eprintln!("eagle3 tap lid {lid}: max_abs={max_abs:.3e} scale={scale:.3e}");
        assert!(
            max_abs <= 2e-3 * scale,
            "tap lid {lid} drifted: max_abs={max_abs} scale={scale}"
        );
    }
    // and the greedy argmax chain the eagle3 driver would commit: identical
    // token ids through both engines
    for step in 0..12 {
        let a = logits_cpu.logits_ith(step as i32).expect("cpu row");
        let b = logits_exe.logits_ith(step as i32).expect("exe row");
        assert_eq!(argmax(a), argmax(b), "greedy token at step {step}");
    }
}

// ---------------------------------------------------------------------------
// batch 20 — encode_eagle3 through the emitter: the encoder graph now rides
// `run_graph` (context.rs), so a GPU-enabled eagle3 head encodes on its own
// executor instead of the Rust engine, and the server wires the draft
// context's device placement like the reference (`common_base_params_to_
// speculative`, speculative.cpp:2446-2470: the draft keeps the top-level
// --device and its own -ngld, default auto = every layer). The regression
// here is SELF-CONSISTENCY: the eagle3 speculation driven over an
// executor-enabled target+head pair must commit the same stream and the same
// drafted/accepted counters as the pure port-CPU pair (the batch-18
// four-way record: port-CPU == port-VK == ref-CPU == ref-VK, n_drafted
// 48 / n_accept 0 on all four).
// ---------------------------------------------------------------------------

/// one full `--spec-type draft-eagle3` drive, returning the committed stream
/// and the drafted/accepted counters (the eagle_speculation_matches_plain_
/// greedy shape, factored for the engine-pair comparison)
fn eagle3_spec_drive(
    tgt: &mut DecodeContext,
    ctx_dft: DecodeContext,
    vocab: &Vocab,
    prompt: &[i32],
    n_predict: usize,
) -> (Vec<i32>, usize, usize) {
    let mut params = CommonParamsSpeculative::default();
    params.types = vec![CommonSpeculativeType::DraftEagle3];
    params.draft.n_max = 3;
    params.draft.p_min = 0.0;
    let mut spec_ctx = common_speculative_init(
        &params,
        1,
        tgt,
        Some(ctx_dft),
        vocab,
        Some(vocab),
        0,
        false,
    )
    .expect("init")
    .expect("speculator");
    let n_vocab = tgt.n_vocab() as i32;
    let mut smpl = SamplingContext::new(n_vocab, SamplingParams { temp: 0.0, ..Default::default() });
    let res = speculative_simple_generate(tgt, &mut spec_ctx, &mut smpl, vocab, prompt, n_predict as i32)
        .expect("speculative generate");
    (res.tokens, res.n_drafted.max(0) as usize, res.n_accept.max(0) as usize)
}

fn eagle_vk_lib_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from("/home/jeffrey/llm/build-rust-vk/bin");
    dir.join("libggml-base.so").exists().then_some(dir)
}

/// the foreign-executor eagle3 pair vs the pure-CPU pair on the synthetic
/// files (head-own, fa off): committed stream + drafted/accepted counters.
/// `device`/`ngl` select the executor: None = the foreign CPU backend (the
/// default test), Some("Vulkan0")+99 = the iGPU (the ignored twin below).
fn eagle3_emit_self_consistency(lib: &std::path::Path, device: Option<&str>, ngl: i32, tag: &str) {
    let prompt: Vec<i32> = (1..=6).collect();
    let n_predict = 12;

    // ---- the pure port-CPU pair ----
    let mut m_plain = open_model(&target_path());
    let plain = plain_greedy(&mut m_plain, false, &prompt, n_predict);
    let mut m_tgt = open_model(&target_path());
    let mut tgt = target_driver(&mut m_tgt, false);
    let ctx_dft = eagle_driver(HeadKind::Own, false);
    let vocab = Vocab::load(&Gguf::open(&target_path()).unwrap()).unwrap();
    let (tok_cpu, drafted_cpu, accept_cpu) =
        eagle3_spec_drive(&mut tgt, ctx_dft, &vocab, &prompt, n_predict);

    // ---- the executor-enabled pair (target AND head — the reference's
    // device placement: the draft context is created from the same params,
    // speculative.cpp:2446-2470) ----
    let mut m_exe = open_model(&target_path());
    let mut tgt_exe = target_driver(&mut m_exe, false);
    let mut ctx_dft_exe = eagle_driver(HeadKind::Own, false);
    let mk = |dev: Option<&str>| {
        let mut cfg = ggml::backend_emit::EmitConfig::new(lib);
        cfg.device = dev.map(str::to_string);
        cfg.n_gpu_layers = ngl;
        cfg.n_threads = 8;
        cfg
    };
    tgt_exe.enable_gpu(mk(device)).expect("enable_gpu (target)");
    ctx_dft_exe.enable_gpu(mk(device)).expect("enable_gpu (eagle3 head)");
    let (tok_exe, drafted_exe, accept_exe) =
        eagle3_spec_drive(&mut tgt_exe, ctx_dft_exe, &vocab, &prompt, n_predict);

    // the acceptance criterion first: both committed streams equal the
    // target's plain greedy
    assert!(
        &tok_cpu[..plain.len()] == &plain[..],
        "{tag}: the CPU pair's committed stream left the greedy stream"
    );
    assert!(
        &tok_exe[..plain.len()] == &plain[..],
        "{tag}: the executor pair's committed stream left the greedy stream"
    );
    // self-consistency: engine choice must not change what gets committed
    // or drafted
    assert_eq!(&tok_cpu[..], &tok_exe[..], "{tag}: committed stream differs");
    assert_eq!(drafted_cpu, drafted_exe, "{tag}: n_drafted differs");
    assert_eq!(accept_cpu, accept_exe, "{tag}: n_accept differs");
    assert!(drafted_exe > 0, "{tag}: no drafts were generated");
    println!(
        "{tag}: {} tokens, drafted {drafted_exe}, accepted {accept_exe} — engine-identical",
        tok_exe.len()
    );
}

/// default: the foreign *CPU* backend (the reference build's libggml-cpu via
/// DL) exercises the encoder+decoder emission path without a device
#[test]
fn eagle3_emit_cpu_backend_self_consistency() {
    let _files = file_lock();
    build_target();
    build_head(HeadKind::Own);
    let Some(lib) = eagle_ref_lib_dir() else {
        eprintln!("eagle3_emit_cpu_backend_self_consistency: reference libs absent, skipping");
        return;
    };
    eagle3_emit_self_consistency(&lib, None, 0, "eagle3 emit (foreign-cpu)");
}

/// the iGPU twin — Vulkan0/-ngl 99 on the synthetic pair (manual: the vk
/// build is not a CI fixture; skipped when the build is absent)
#[test]
#[ignore = "needs the Vulkan reference build (/home/jeffrey/llm/build-rust-vk)"]
fn eagle3_emit_vulkan_self_consistency() {
    let _files = file_lock();
    build_target();
    build_head(HeadKind::Own);
    let lib = eagle_vk_lib_dir().expect("the Vulkan reference build is absent");
    eagle3_emit_self_consistency(&lib, Some("Vulkan0"), 99, "eagle3 emit (Vulkan0)");
}
