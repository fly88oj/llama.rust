//! qwen35_e2e.rs — real-model end-to-end verification of
//! `build_qwen35_forward` (src/models/qwen35.cpp: gated delta net + full
//! attention) on Qwen3.6-27B-Q4_K_M, driven through `DecodeContext`
//! (ForwardWeights::Qwen35, including the RecurrentState cells).
//!
//! Reference capture (PARITY.md protocol: **fresh** llama-server, first request
//! on the slot, greedy, default FA):
//!   /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server \
//!       -m Qwen3.6-27B-Q4_K_M.gguf -c 512 -t 8 --port 8861 --host 127.0.0.1
//!   curl -s http://127.0.0.1:8861/completion -H 'Content-Type: application/json' \
//!       -d '{"prompt":"The capital of France is","n_predict":16,
//!            "temperature":0,"logprobs":20,"cache_prompt":false}'
//! (captured 2026-09-24 against reference bd4f514db1)
//!
//! Runs by default (metadata + param wiring only):
//!   * `qwen35_27b_params_and_weights` — gated delta net geometry, the
//!     recurrent/full-attention interleave and the IMROPE sections
//!
//! `#[ignore]`d (manual; release mode, minutes):
//!   * `qwen35_27b_reference_parity` — 15.4 GiB Q4_K_M: prefill + 16 greedy
//!     tokens vs the fresh reference first request (FA on and off)
//!   * `qwen35_state_vs_full_recompute` — step-by-step decoding must reproduce
//!     one prefill batch over the same tokens (the GDN recurrent state has to
//!     carry the identical information)
//!
//! Measured 2026-09-24 (8 threads):
//!   * Qwen3.6-27B-Q4_K_M: **MATCH 16/16** for FA **and** non-FA; text
//!     " Paris.\n\n thinking\nHere's a thinking process:\n\n1.  **"
//!     (prefill 0.21 t/s, gen 0.09-0.10 t/s; the 48 gated-delta-net layers run
//!     the fused GDN op, the 16 full-attention blocks FA/MHA)
//!   * state vs full recompute: **bit-identical** logits (max abs 0.0) —
//!     the recurrent state + KV cache carry exactly the same information as a
//!     batch prefill.
//!
//! Run the heavy tests with:
//!   cargo test --release -p llama --test qwen35_e2e -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{Qwen35LayerWeights, Qwen35ModelWeights, Qwen35Params};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const QWEN35_27B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf";
/// Same architecture with one extra nextn/MTP block (65 blocks in the file,
/// `n_layer()` = 64): the trunk must wire exactly like the plain file.
const QWEN35_27B_MTP: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_M.gguf";

const PROMPT: &str = "The capital of France is";

/// Fresh reference server, first request, greedy 16 (--port 8861, default FA).
/// Prompt tokens = [760, 6511, 314, 9338, 369] (`tokens_evaluated = 5`, no
/// BOS). Text: " Paris.\n\n thinking\nHere's a thinking process:\n\n1.  **".
const REF16_QWEN35_27B: [i32; 16] = [
    11751, 13, 271, 248068, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972,
];
/// Reference per-step top-8 (id, logprob), first three steps.
#[rustfmt::skip]
const REF_TOP8_QWEN35_27B: [[(i32, f32); 8]; 3] = [
    [(11751, -0.4631), (271, -3.6174), (6924, -3.6571), (279, -3.7784), (264, -3.8042), (524, -3.8277), (7172, -4.1166), (198, -4.1818)],
    [(13, -0.3276), (11, -1.7290), (271, -4.0743), (321, -4.4085), (198, -4.4549), (641, -5.1497), (318, -5.3880), (6889, -5.4892)],
    [(271, -1.7047), (561, -2.0746), (198, -2.1436), (11751, -2.2871), (1049, -3.0064), (1061, -3.5688), (3437, -3.5803), (733, -3.7075)],
];

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    size_bytes: u64,
}

fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let size_bytes = std::fs::metadata(path).ok()?.len();
    let file = std::fs::File::open(path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    match load_model(&gguf, mmap.clone()) {
        Ok(model) => Some(Loaded {
            model,
            gguf,
            mmap,
            size_bytes,
        }),
        Err(e) => {
            eprintln!("SKIP: load_model({path}) failed: {e}");
            None
        }
    }
}

fn mem_available_gb() -> f64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0 / 1024.0;
        }
    }
    0.0
}

fn mem_guard(label: &str, need_gb: f64) -> bool {
    let avail = mem_available_gb();
    if avail < need_gb {
        eprintln!("SKIP {label}: {avail:.1} GiB available < {need_gb:.1} GiB required");
        return false;
    }
    true
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

fn logprobs(v: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let mx = v[idx[0]];
    let lse = mx + (v.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>()).ln() as f32;
    idx.into_iter().map(|i| (i as i32, v[i] - lse)).collect()
}

fn text_of(vocab: &Vocab, ids: &[i32]) -> String {
    let mut s = String::new();
    for &id in ids {
        s.push_str(&vocab.token_to_piece(id));
    }
    s
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

pub fn qwen35_params(m: &LlamaModel) -> Qwen35Params {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    // per-layer vectors cover the *trunk* only (n_layer(), not n_layer_all)
    let n_layer = hp.n_layer() as usize;
    Qwen35Params {
        attn: AttnParams {
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
        },
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

pub fn qwen35_weights(m: &LlamaModel) -> Qwen35ModelWeights {
    // trunk only: `llama_model::layers` holds n_layer_all entries and the MTP
    // block(s) live *after* the trunk (qwen35.cpp:157-160), which the C graph
    // never executes (its loop runs il < hparams.n_layer()).
    let n_trunk = m.hparams.n_layer() as usize;
    Qwen35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        cls_out: m.cls_out,
        cls_out_b: m.cls_out_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| Qwen35LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

fn qwen35_dctx(mut l: Loaded, fa: bool, n_ctx: u32, n_batch: usize) -> DecodeContext {
    let mut p = qwen35_params(&l.model);
    p.attn.use_flash_attn = fa;
    let w = qwen35_weights(&l.model);
    let gctx = std::mem::replace(&mut l.model.ctx, Context::new());
    DecodeContext::new_with(
        gctx,
        ForwardWeights::Qwen35(w, p.clone()),
        p.attn,
        n_ctx,
        8,
        n_batch,
    )
}

fn run_greedy(
    dctx: &mut DecodeContext,
    prompt: &[i32],
    n_gen: usize,
) -> (
    Vec<i32>,
    Vec<Vec<(i32, f32)>>,
    Vec<f32>,
    std::time::Duration,
    std::time::Duration,
) {
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let t0 = std::time::Instant::now();
    let mut cur = dctx.decode(prompt, &pos).expect("prefill").to_vec();
    let prefill = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut ids = Vec::new();
    let mut lps = Vec::new();
    let mut pos_i = prompt.len() as i32;
    for _ in 0..n_gen {
        lps.push(logprobs(&cur, 20));
        let tok = argmax(&cur);
        ids.push(tok);
        cur = dctx.decode(&[tok], &[pos_i]).expect("step").to_vec();
        pos_i += 1;
    }
    let gen = t1.elapsed();
    (ids, lps, cur, prefill, gen)
}

// ---------------------------------------------------------------------------
// 1. wiring test (default run)
// ---------------------------------------------------------------------------

#[test]
fn qwen35_27b_params_and_weights() {
    let Some(l) = load_real(QWEN35_27B) else {
        return;
    };
    let p = qwen35_params(&l.model);
    let w = qwen35_weights(&l.model);

    assert_eq!(w.layers.len(), 64);
    assert_eq!(p.n_embd, 5120);
    assert_eq!(p.attn.n_head, 24);
    assert_eq!(p.attn.n_head_kv, 4);
    assert_eq!(p.attn.n_embd_head_k, 256);
    assert_eq!(p.attn.n_rot, 64);
    // IMROPE + sections
    assert_eq!(p.attn.rope_mode, 40, "GGML_ROPE_TYPE_IMROPE");
    assert_eq!(p.rope_sections, [11, 11, 10, 0]);
    // gated delta net geometry
    assert_eq!(p.ssm_d_conv, 4);
    assert_eq!(p.ssm_d_state, 128);
    assert_eq!(p.ssm_n_group, 16);
    assert_eq!(p.ssm_dt_rank, 48);
    assert_eq!(p.ssm_d_inner, 6144);
    assert_eq!(p.n_embd_r, 3 * (6144 + 2 * 16 * 128)); // 30720
    assert_eq!(p.n_embd_s, 128 * 6144); // 786432
                                        // recurrent pattern: 3 of every 4 layers
    assert_eq!(p.is_recr.iter().filter(|&&r| r).count(), 48);
    assert!(p.is_recr[0] && p.is_recr[1] && p.is_recr[2] && !p.is_recr[3]);
    // per-layer tensors
    for (il, lw) in w.layers.iter().enumerate() {
        if p.is_recr[il] {
            assert!(lw.wqkv.is_some() && lw.wqkv_gate.is_some(), "layer {il}");
            assert!(lw.ssm_conv1d.is_some() && lw.ssm_a.is_some() && lw.ssm_norm.is_some());
            assert!(lw.wq.is_none(), "recurrent layer {il} has no wq");
        } else {
            assert!(lw.wq.is_some() && lw.wk.is_some() && lw.wv.is_some() && lw.wo.is_some());
            assert!(
                lw.wqkv.is_none(),
                "attention layer {il} uses separate q/k/v"
            );
        }
    }
    // 4 position ids per token (mrope)
    assert_eq!(ForwardWeights::Qwen35(w, p).n_pos_per_embd(), 4);
}

/// MTP/nextn variant (65 blocks): `n_layer()` excludes the MTP block, so the
/// trunk wiring (and therefore the main forward pass) is identical to the plain
/// file — the MTP block is loaded but never executed (qwen35.cpp:196-198).
#[test]
fn qwen35_27b_mtp_trunk_wiring() {
    let Some(l) = load_real(QWEN35_27B_MTP) else {
        return;
    };
    assert_eq!(l.model.hparams.n_layer_all, 65);
    assert_eq!(l.model.hparams.n_layer_nextn, 1);
    // llama_model::layers keeps an entry per block (the C `layers` vector is
    // sized n_layer_all); the *trunk* is the first n_layer() of them
    assert_eq!(l.model.layers.len(), 65, "trunk + MTP slots");
    let p = qwen35_params(&l.model);
    let w = qwen35_weights(&l.model);
    assert_eq!(p.is_recr.len(), 64);
    assert_eq!(w.layers.len(), 64);
    // the MTP block is a dense attention block, not recurrent
    assert!(!p.is_recr[63]);
    assert_eq!(ForwardWeights::Qwen35(w, p).n_pos_per_embd(), 4);
}

// ---------------------------------------------------------------------------
// 2. real-model reference parity (manual)
// ---------------------------------------------------------------------------

/// Qwen3.6-27B-Q4_K_M (15.4 GiB): prompt "The capital of France is" (5 tokens,
/// no BOS) + 16 greedy tokens vs the fresh reference first request.
///
///   cargo test --release -p llama --test qwen35_e2e -- --ignored --nocapture \
///       qwen35_27b_reference_parity
#[test]
#[ignore = "manual: 15.4 GiB model, prefill + 16 greedy decode steps in release"]
fn qwen35_27b_reference_parity() {
    if !mem_guard("qwen35-27B", 22.0) {
        return;
    }
    let Some(l0) = load_real(QWEN35_27B) else {
        return;
    };
    let vocab = Vocab::load(&l0.gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.tokenize(PROMPT, true, true);
    let file_gib = l0.size_bytes as f64 / 1073741824.0;
    println!("qwen35-27B: prompt ids {prompt:?} ({})", prompt.len());
    assert_eq!(
        prompt,
        vec![760, 6511, 314, 9338, 369],
        "prompt tokenization"
    );
    drop(l0);

    for fa in [false, true] {
        let Some(l) = load_real(QWEN35_27B) else {
            return;
        };
        let mut dctx = qwen35_dctx(l, fa, 512, 64);
        let (ids, lps, last, prefill, gen) = run_greedy(&mut dctx, &prompt, 16);
        assert!(last.iter().all(|v| v.is_finite()), "logits finite");
        println!(
            "qwen35-27B (FA={fa}, {file_gib:.2} GiB): prefill {} tok in {prefill:?} ({:.2} t/s); \
             gen 16 tok in {gen:?} ({:.2} t/s)",
            prompt.len(),
            prompt.len() as f32 / prefill.as_secs_f32(),
            16.0 / gen.as_secs_f32(),
        );
        let label = if fa { "qwen35-27B FA" } else { "qwen35-27B" };
        let matched = ids
            .iter()
            .zip(&REF16_QWEN35_27B)
            .filter(|(a, b)| a == b)
            .count();
        let first = ids.iter().zip(&REF16_QWEN35_27B).position(|(a, b)| a != b);
        println!(
            "{label}: MATCH {matched}/16{}",
            match first {
                Some(k) => format!(
                    " (first divergence step {k}: got {} want {})",
                    ids[k], REF16_QWEN35_27B[k]
                ),
                None => String::new(),
            }
        );
        println!("{label}: text {:?}", text_of(&vocab, &ids));
        if let Some(k) = first {
            if let Some(r8) = REF_TOP8_QWEN35_27B.get(k) {
                let gap = lps[k]
                    .iter()
                    .find(|(id, _)| *id == r8[0].0)
                    .map(|(_, mine)| (mine - r8[0].1).abs())
                    .unwrap_or(f32::NAN);
                let mine_k = lps[k].iter().find(|(id, _)| *id == ids[k]).map(|(_, p)| *p);
                println!(
                    "{label}: step {k} pair-wise logprob gap {gap:.4} (ref top-1 {:.4}); ours {} {:?}",
                    r8[0].1,
                    ids[k],
                    mine_k
                );
                println!(
                    "{label}: step {k} ours top8 {:?}",
                    lps[k]
                        .iter()
                        .take(8)
                        .map(|(id, p)| (*id, *p))
                        .collect::<Vec<_>>()
                );
                println!("{label}: step {k} ref  top8 {:?}", r8.to_vec());
            }
        }
        println!(
            "{label} last-row top-5: {:?}",
            logprobs(&last, 5)
                .iter()
                .map(|(i, p)| (text_of(&vocab, &[*i]), *p))
                .collect::<Vec<_>>()
        );
    }
}

/// Recurrent-state equivalence: decoding the same tokens step by step must land
/// on (nearly) the same logits as one prefill batch — the gated delta net state
/// plus the KV cache have to carry the identical information. No server
/// needed.
///
///   cargo test --release -p llama --test qwen35_e2e -- --ignored --nocapture \
///       qwen35_state_vs_full_recompute
#[test]
#[ignore = "manual: 15.4 GiB model, two forwards (batch vs step-by-step)"]
fn qwen35_state_vs_full_recompute() {
    if !mem_guard("qwen35-27B", 22.0) {
        return;
    }
    let Some(l) = load_real(QWEN35_27B) else {
        return;
    };
    let mut dctx = qwen35_dctx(l, false, 512, 64);
    let ids = [760i32, 6511, 314, 9338, 369, 11751, 13, 271];
    let batch = dctx
        .decode(&ids, &[0, 1, 2, 3, 4, 5, 6, 7])
        .unwrap()
        .to_vec();
    dctx.reset_sequence();
    let mut stepwise = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        stepwise = dctx.decode(&[t], &[i as i32]).unwrap().to_vec();
    }
    let maxabs = batch
        .iter()
        .zip(&stepwise)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = batch.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "qwen35 state vs recompute: max abs {maxabs:.3e} (|logits|max {scale:.3e}); argmax {} vs {}",
        argmax(&batch),
        argmax(&stepwise)
    );
    assert_eq!(argmax(&batch), argmax(&stepwise), "argmax must agree");
    assert!(maxabs < 0.5, "state/recompute diverged: {maxabs}");
}
