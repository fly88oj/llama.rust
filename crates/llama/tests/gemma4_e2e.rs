//! gemma4_e2e.rs — real-model end-to-end verification of
//! `build_gemma4_forward` (src/models/gemma4.cpp) on the GGUFs on this
//! machine, driven through `DecodeContext` (ForwardWeights::Gemma4).
//!
//! Reference capture (PARITY.md protocol: **fresh** llama-server, first request
//! on the slot, greedy, default FA):
//!   /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server \
//!       -m <file> -c 512 -t 8 --port 8860 --host 127.0.0.1
//!   curl -s http://127.0.0.1:8860/completion -H 'Content-Type: application/json' \
//!       -d '{"prompt":"The capital of France is","n_predict":16,
//!            "temperature":0,"logprobs":20,"cache_prompt":false}'
//! (captured 2026-09-24 against reference bd4f514db1, FA auto-enabled)
//!
//! Runs by default (metadata + param wiring only, no tensor data read):
//!   * `gemma4_12b_params_and_weights` — per-layer geometry / SWA pattern /
//!     rope-freqs sharing / softcap wiring for the 12B QAT file
//!   * `gemma4_26b_a4b_params_and_weights` — the same + MoE wiring (128
//!     experts, merged `ffn_gate_up_exps`, router scale, per-expert down scale)
//!
//! `#[ignore]`d (manual; release mode, minutes):
//!   * `gemma4_12b_reference_parity` — 6.5 GiB Q4_0: prefill "The capital of
//!     France is" + 16 greedy tokens vs the fresh reference first request
//!   * `gemma4_26b_a4b_reference_parity` — 13.4 GiB Q4_0 MoE, same protocol
//!
//! Measured 2026-09-24 (both attention branches, 8 threads; re-measured after
//! the SWA port and after the SIMD/tinyBLAS/contracted-FMA kernel work, which
//! moved the MoE file's residuals):
//!   * gemma-4-12B-it-QAT-Q4_0: **MATCH 16/16** for FA **and** non-FA
//!     (text "0111111111111111"; prefill 6 tok ~0.4 s, gen 6.2-6.6 t/s)
//!   * gemma-4-26B-A4B-it-QAT-Q4_0: re-measured after the `inp_out_ids`
//!     port (the prefill head now gathers the single output row like the
//!     reference's, llama-graph.cpp:2480-2496): non-FA **6/16** and FA
//!     **16/16** (was 0/16 / 4/16 after the SIMD/tinyBLAS/contracted-FMA
//!     kernel rounds; the remaining non-FA drift is the kernel-residual band,
//!     not the graph — `GEMMA4_UNIFIED=1` stays bit-identical to the split).
//!     The 12B's 16/16 in both branches *is* re-verified with the SWA split on.
//!   * `gemma4_fa_op_probe` (default run) pins the FA kernel itself for all
//!     four gemma4 geometries (SWA 256x8/256x2, dense 512x1/512x2) against a
//!     direct softmax: worst diff 2-6e-4 (the f16 VKQ accumulator band).
//!
//! Run the heavy tests with:
//!   cargo test --release -p llama --test gemma4_e2e -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{Gemma4LayerWeights, Gemma4ModelWeights, Gemma4Params};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// models / reference data on this machine
// ---------------------------------------------------------------------------

const GEMMA4_12B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
const GEMMA4_26B_A4B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf";

const PROMPT: &str = "The capital of France is";

/// Fresh reference server, first request, greedy 16 (llama-server --port 8860,
/// default FA). Prompt tokens = [2, 818, 5279, 529, 7001, 563] (BOS + 5;
/// `tokens_evaluated = 6`). Text: `0111111111111111`.
const REF16_GEMMA4_12B: [i32; 16] = [
    236771, 236770, 236770, 236770, 236770, 236770, 236770, 236770, 236770, 236770, 236770, 236770,
    236770, 236770, 236770, 236770,
];
/// Reference per-step top-8 (id, logprob) for the first three steps — the
/// pair-wise logprob comparison baseline.
#[rustfmt::skip]
const REF_TOP8_GEMMA4_12B: [[(i32, f32); 8]; 3] = [
    [(236771, -1.1496), (236770, -1.6458), (236800, -2.5596), (236825, -2.6317), (236828, -2.6598), (236832, -2.7162), (236812, -2.8124), (236810, -2.9292)],
    [(236770, -0.7006), (236800, -2.2586), (236825, -2.7427), (236828, -2.7820), (236778, -2.9494), (236812, -2.9996), (236832, -3.1245), (236810, -3.1389)],
    [(236770, -1.0713), (236812, -2.1467), (236825, -2.1681), (236800, -2.4953), (236778, -2.4990), (236771, -2.7354), (236810, -2.8246), (236828, -2.9226)],
];

/// Fresh reference server (:8862), first request, greedy 16. Prompt tokens =
/// [2, 818, 5279, 529, 7001, 563] (`tokens_evaluated = 6`). Text:
/// " Paris.\n<|channel>3/14/2024 12".
const REF16_GEMMA4_26B_A4B: [i32; 16] = [
    9079, 236761, 107, 100, 236800, 236786, 236770, 236812, 236786, 236778, 236771, 236778, 236812,
    236743, 236770, 236778,
];
#[rustfmt::skip]
const REF_TOP8_GEMMA4_26B_A4B: [[(i32, f32); 8]; 3] = [
    [(9079, -1.1344), (506, -1.2514), (236761, -2.7826), (496, -3.5596), (107, -4.0874), (5596, -4.6883), (3198, -4.9830), (236764, -4.9888)],
    [(236761, -0.0660), (236764, -3.3329), (236772, -3.7772), (568, -6.9471), (236793, -7.2652), (236786, -7.7020), (532, -7.7348), (236779, -7.8035)],
    [(107, -0.9699), (1346, -2.1296), (45518, -3.3759), (669, -3.8318), (101, -3.8802), (586, -3.9002), (139, -4.2655), (138, -4.3048)],
];

// ---------------------------------------------------------------------------
// helpers (same shape as hybrid_e2e.rs / gpt_oss_e2e.rs)
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

/// greedy: strict >, first max wins (llama_sampler_init_greedy)
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

/// Top-k (id, logprob) pairs, descending (llama-server convention).
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
// wiring: LlamaModel → Gemma4ModelWeights / Gemma4Params
// ---------------------------------------------------------------------------

pub fn gemma4_params(m: &LlamaModel) -> Gemma4Params {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    let n_layer = m.layers.len();
    Gemma4Params {
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
            // the reference server runs FA by default; both paths are wired
            // (attn_kv_cached), the tests set the flag explicitly
            use_flash_attn: false,
        },
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
        f_attention_scale: hp.f_attention_scale,
        f_final_logit_softcapping: hp.f_final_logit_softcapping,
        n_embd_per_layer: hp.n_embd_per_layer as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: (0..n_layer).map(|il| hp.n_ff_exp(il)).collect(),
    }
}

pub fn gemma4_weights(m: &LlamaModel) -> Gemma4ModelWeights {
    Gemma4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        per_layer_model_proj: m.per_layer_model_proj,
        per_layer_proj_norm: m.per_layer_proj_norm,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| Gemma4LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                rope_freqs: l.rope_freqs,
                out_scale: l.out_scale,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_inp_s: l.ffn_gate_inp_s,
                ffn_pre_norm_2: l.ffn_pre_norm_2,
                ffn_post_norm_1: l.ffn_post_norm_1,
                ffn_post_norm_2: l.ffn_post_norm_2,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_down_exps_s: l.ffn_down_exps_s,
                per_layer_inp_gate: l.per_layer_inp_gate,
                per_layer_proj: l.per_layer_proj,
                per_layer_post_norm: l.per_layer_post_norm,
            })
            .collect(),
    }
}

/// Build the decode context for one run (FA flag selects the attention
/// branch). Takes the model's Context over (the weight tensors' mmap buffers
/// are owned by the Context itself, tensor.rs:200-211).
///
/// The context is built with the `llama_kv_cache_iswa` split
/// (`DecodeContext::new_with_swa` + `SwaCacheSpec::from_hparams`): gemma4 is a
/// SWA architecture (n_swa = 1024), so the 48 layers are split over two caches
/// with their own masks. At `n_ctx = 512 < n_swa` the window never cuts
/// anything, so these short-context anchors are the same numbers as before the
/// split — the long-context SWA evidence is in swa_e2e.rs. `GEMMA4_UNIFIED=1`
/// forces the old single cache (A/B knob for exactly that claim).
fn gemma4_dctx(mut l: Loaded, fa: bool, n_ctx: u32) -> DecodeContext {
    let mut p = gemma4_params(&l.model);
    p.attn.use_flash_attn = fa;
    let spec = llama::kv_cache::SwaCacheSpec::from_hparams(&l.model.hparams);
    let w = gemma4_weights(&l.model);
    let gctx = std::mem::replace(&mut l.model.ctx, Context::new());
    let attn = p.attn;
    if std::env::var("GEMMA4_UNIFIED").is_err() {
        DecodeContext::new_with_swa(gctx, ForwardWeights::Gemma4(w, p), attn, n_ctx, 8, 64, spec)
    } else {
        DecodeContext::new_with(gctx, ForwardWeights::Gemma4(w, p), attn, n_ctx, 8, 64)
    }
}

/// prefill + greedy decode; returns (ids, per-step top-20 logprob lists,
/// last-token logits, prefill time, generate time)
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

/// MATCH x/n against the reference ids, with the first divergence index.
fn report_parity(label: &str, vocab: &Vocab, ids: &[i32], lps: &[Vec<(i32, f32)>], r: &[i32]) {
    let n = r.len();
    assert_eq!(ids.len(), n);
    let matched = ids.iter().zip(r).filter(|(a, b)| a == b).count();
    let first = ids.iter().zip(r).position(|(a, b)| a != b);
    println!(
        "{label}: MATCH {matched}/{n}{}",
        match first {
            Some(k) => format!(" (first divergence step {k}: got {} want {})", ids[k], r[k]),
            None => String::new(),
        }
    );
    println!("{label}: text {:?}", text_of(vocab, ids));
    if let Some(k) = first {
        print!("{label}: step {k} ours  : ");
        for (id, p) in lps[k].iter().take(8) {
            print!("{}{}({:.3}) ", id, text_of(vocab, &[*id]), p);
        }
        println!();
        let want = &lps[k];
        if let Some((_, wp)) = want.iter().find(|(id, _)| *id == r[k]) {
            let mine = want.iter().find(|(id, _)| *id == ids[k]).map(|(_, p)| *p);
            if let Some(mp) = mine {
                println!(
                    "{label}: step {k} pair-wise logprob: want {wp:.4} got {mp:.4} (gap {:+.4})",
                    (mp - wp).abs()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 1. wiring tests (default run)
// ---------------------------------------------------------------------------

#[test]
fn gemma4_12b_params_and_weights() {
    let Some(l) = load_real(GEMMA4_12B) else {
        return;
    };
    let p = gemma4_params(&l.model);
    let w = gemma4_weights(&l.model);

    assert_eq!(w.layers.len(), 48);
    assert_eq!(p.is_swa.len(), 48);
    // SWA pattern: 5 sliding + 1 dense, period 6
    let n_swa = p.is_swa.iter().filter(|&&s| s).count();
    assert_eq!(n_swa, 40);
    assert!(p.is_swa[0] && !p.is_swa[5] && p.is_swa[6]);
    // per-layer dims: SWA 256x8, dense 512x1
    assert_eq!(
        (p.n_embd_head_k[0], p.n_head_kv[0], p.n_rot[0]),
        (256, 8, 256)
    );
    assert_eq!(
        (p.n_embd_head_k[5], p.n_head_kv[5], p.n_rot[5]),
        (512, 1, 512)
    );
    assert_eq!(p.f_attention_scale, 1.0);
    assert_eq!(p.f_final_logit_softcapping, 30.0);
    assert_eq!(p.n_embd_per_layer, 0);
    assert_eq!(p.n_expert, 0, "12B is dense");
    // rope freq factors only on the dense layers, all sharing one tensor
    assert!(w.layers[0].rope_freqs.is_none());
    let rf = w.layers[5].rope_freqs.expect("dense rope_freqs");
    let ne = l.model.ctx.ne(rf);
    assert_eq!(ne[0], 256, "n_embd_head(512)/2");
    for il in 0..48 {
        if p.is_swa[il] {
            assert!(w.layers[il].rope_freqs.is_none(), "layer {il}");
        } else {
            assert_eq!(w.layers[il].rope_freqs, Some(rf), "layer {il}");
        }
    }
    // blk.5 carries no attn_v (V = K), all layers have out_scale
    assert!(w.layers[5].wv.is_none(), "V = K path");
    assert!(w.layers[0].wv.is_some());
    assert!(w.layers.iter().all(|l| l.out_scale.is_some()));
    // kv row widths follow the per-layer geometry
    let (k_row, v_row) = ForwardWeights::Gemma4(w, p).kv_dims(0, 0);
    assert_eq!(k_row[0], 256 * 8);
    assert_eq!(v_row[0], 256 * 8);
    assert_eq!(k_row[5], 512);
    assert_eq!(v_row[5], 512);
}

#[test]
fn gemma4_26b_a4b_params_and_weights() {
    let Some(l) = load_real(GEMMA4_26B_A4B) else {
        return;
    };
    let p = gemma4_params(&l.model);
    let w = gemma4_weights(&l.model);

    assert_eq!(w.layers.len(), 30);
    assert_eq!(p.n_expert, 128);
    assert_eq!(p.n_expert_used[0], 8);
    assert_eq!(p.n_ff_exp[0], 704);
    assert_eq!(p.n_embd, 2816);
    // MoE wiring per layer: merged gate_up experts, router scale, branch norms
    for (il, lw) in w.layers.iter().enumerate() {
        assert!(lw.ffn_gate_inp.is_some(), "layer {il}");
        assert!(lw.ffn_gate_up_exps.is_some(), "merged ffn_gate_up_exps");
        assert!(lw.ffn_gate_inp_s.is_some(), "router scale");
        assert!(lw.ffn_pre_norm_2.is_some() && lw.ffn_post_norm_1.is_some());
        assert!(lw.ffn_post_norm_2.is_some());
        assert!(lw.ffn_down_exps_s.is_some(), "per-expert down scale");
        // SWA 256x8 / dense 512x2 for this model
        assert_eq!((p.n_embd_head_k[0], p.n_head_kv[0]), (256, 8));
        assert_eq!((p.n_embd_head_k[5], p.n_head_kv[5]), (512, 2));
    }
    assert!(!p.is_swa[5] && p.is_swa[4]);
    // token_embd is a Q4_0 expert-mix file; the shared expert FFN is dense
    assert_eq!(l.model.ctx.ne(w.layers[0].ffn_up)[1], 2112);
}

// ---------------------------------------------------------------------------
// 2. real-model reference parity (manual; PARITY.md protocol)
// ---------------------------------------------------------------------------

/// gemma-4-12B-it-QAT-Q4_0 (6.5 GiB): prompt "The capital of France is"
/// (BOS + 5 tokens) + 16 greedy tokens vs the fresh reference first request.
/// Both attention branches (FA on/off) are run and reported.
///
///   cargo test --release -p llama --test gemma4_e2e -- --ignored --nocapture \
///       gemma4_12b_reference_parity
#[test]
#[ignore = "manual: 6.5 GiB model, prefill + 16 greedy decode steps in release"]
fn gemma4_12b_reference_parity() {
    if !mem_guard("gemma4-12B", 10.0) {
        return;
    }
    let Some(l0) = load_real(GEMMA4_12B) else {
        return;
    };
    let vocab = Vocab::load(&l0.gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.tokenize(PROMPT, true, true);
    let file_gib = l0.size_bytes as f64 / 1073741824.0;
    println!("gemma4-12B: prompt ids {prompt:?} ({})", prompt.len());
    // reference reported tokens_evaluated = 6 (BOS + 5)
    assert_eq!(
        prompt,
        vec![2, 818, 5279, 529, 7001, 563],
        "prompt tokenization"
    );
    drop(l0);

    for fa in [false, true] {
        let Some(l) = load_real(GEMMA4_12B) else {
            return;
        };
        let mut dctx = gemma4_dctx(l, fa, 512);
        let (ids, lps, last, prefill, gen) = run_greedy(&mut dctx, &prompt, 16);
        assert!(last.iter().all(|v| v.is_finite()), "logits finite");
        println!(
            "gemma4-12B (FA={fa}, {file_gib:.2} GiB): prefill {} tok in {prefill:?} ({:.1} t/s); \
             gen 16 tok in {gen:?} ({:.2} t/s)",
            prompt.len(),
            prompt.len() as f32 / prefill.as_secs_f32(),
            16.0 / gen.as_secs_f32(),
        );
        let label = if fa { "gemma4-12B FA" } else { "gemma4-12B" };
        report_parity(label, &vocab, &ids, &lps, &REF16_GEMMA4_12B);
        // pair-wise logprob delta at the first divergence (the reference's own
        // top-1 vs ours there), plus the reference's top-8 for context
        if let Some(k) = ids.iter().zip(&REF16_GEMMA4_12B).position(|(a, b)| a != b) {
            if let Some(r8) = REF_TOP8_GEMMA4_12B.get(k) {
                let gap = lps[k]
                    .iter()
                    .find(|(id, _)| *id == r8[0].0)
                    .map(|(_, mine)| (mine - r8[0].1).abs())
                    .unwrap_or(f32::NAN);
                println!(
                    "{label}: FIRST DIVERGENCE step {k}: pair-wise gap {gap:.4} logprob on the \
                     reference top-1 (margin {:.4}); ref top8 {:?}",
                    r8[0].1 - r8[1].1,
                    r8.iter().map(|&(id, _)| id).collect::<Vec<_>>()
                );
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

/// gemma-4-26B-A4B-it-QAT-Q4_0 (13.4 GiB, MoE): same protocol — exercises the
/// merged `ffn_gate_up_exps` path, the router-on-attn_out logits, the branch
/// norms and the per-expert down scale.
///
///   cargo test --release -p llama --test gemma4_e2e -- --ignored --nocapture \
///       gemma4_26b_a4b_reference_parity
#[test]
#[ignore = "manual: 13.4 GiB MoE model, prefill + 16 greedy decode steps in release"]
fn gemma4_26b_a4b_reference_parity() {
    if !mem_guard("gemma4-26B-A4B", 18.0) {
        return;
    }
    let Some(l0) = load_real(GEMMA4_26B_A4B) else {
        return;
    };
    let vocab = Vocab::load(&l0.gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.tokenize(PROMPT, true, true);
    let file_gib = l0.size_bytes as f64 / 1073741824.0;
    println!("gemma4-26B-A4B: prompt ids {prompt:?} ({})", prompt.len());
    drop(l0);

    for fa in [false, true] {
        let Some(l) = load_real(GEMMA4_26B_A4B) else {
            return;
        };
        let mut dctx = gemma4_dctx(l, fa, 512);
        let (ids, lps, last, prefill, gen) = run_greedy(&mut dctx, &prompt, 16);
        assert!(last.iter().all(|v| v.is_finite()), "logits finite");
        println!(
            "gemma4-26B-A4B (FA={fa}, {file_gib:.2} GiB): prefill {} tok in {prefill:?} ({:.1} t/s); \
             gen 16 tok in {gen:?} ({:.2} t/s)",
            prompt.len(),
            prompt.len() as f32 / prefill.as_secs_f32(),
            16.0 / gen.as_secs_f32(),
        );
        let label = if fa {
            "gemma4-26B-A4B FA"
        } else {
            "gemma4-26B-A4B"
        };
        report_parity(label, &vocab, &ids, &lps, &REF16_GEMMA4_26B_A4B);
        if let Some(k) = ids
            .iter()
            .zip(&REF16_GEMMA4_26B_A4B)
            .position(|(a, b)| a != b)
        {
            if let Some(r8) = REF_TOP8_GEMMA4_26B_A4B.get(k) {
                let gap = lps[k]
                    .iter()
                    .find(|(id, _)| *id == r8[0].0)
                    .map(|(_, mine)| (mine - r8[0].1).abs())
                    .unwrap_or(f32::NAN);
                println!(
                    "{label}: FIRST DIVERGENCE step {k}: pair-wise gap {gap:.4} logprob on the \
                     reference top-1 (margin {:.4}); ref top8 {:?}",
                    r8[0].1 - r8[1].1,
                    r8.iter().map(|&(id, _)| id).collect::<Vec<_>>()
                );
                println!(
                    "{label}: step {k} ours top8 {:?}",
                    lps[k]
                        .iter()
                        .take(8)
                        .map(|(id, p)| (*id, *p))
                        .collect::<Vec<_>>()
                );
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
/// FA-vs-non-FA probe on a **discriminative** prompt (the parity prompt
/// "The capital of France is" degenerates into 15 repetitions of "1" for this
/// model, so it cannot discriminate the attention branch). The prompt tokens
/// are the reference's chat-template rendering of the same question
/// (`/tokenize` on the reference server); the two branches must agree on the
/// greedy continuation.
///
///   cargo test --release -p llama --test gemma4_e2e -- --ignored --nocapture \
///       gemma4_12b_fa_vs_nonfa_probe
#[test]
#[ignore = "manual: 6.5 GiB model, two 16-step runs in release"]
fn gemma4_12b_fa_vs_nonfa_probe() {
    if !mem_guard("gemma4-12B", 10.0) {
        return;
    }
    // reference chat-template rendering (gemma4 <start_of_turn> markers)
    let chat_prompt: Vec<i32> = vec![
        236820, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107, 818, 5279, 529, 7001, 563,
        236820, 643, 236779, 1340, 236779, 887, 236813, 107, 236820, 3041, 236779, 1340, 236779,
        887, 236813, 4368, 107,
    ];
    let mut out: Vec<Vec<i32>> = Vec::new();
    for fa in [false, true] {
        let Some(l) = load_real(GEMMA4_12B) else {
            return;
        };
        let vocab = Vocab::load(&l.gguf).expect("vocab");
        let mut dctx = gemma4_dctx(l, fa, 512);
        let (ids, _lps, last, prefill, gen) = run_greedy(&mut dctx, &chat_prompt, 16);
        println!(
            "gemma4-12B chat (FA={fa}): prefill {} tok in {prefill:?}; gen 16 in {gen:?}; text {:?}",
            chat_prompt.len(),
            text_of(&vocab, &ids)
        );
        println!("gemma4-12B chat (FA={fa}): ids {ids:?}");
        println!(
            "gemma4-12B chat (FA={fa}) last top-5: {:?}",
            logprobs(&last, 5)
                .iter()
                .map(|(i, p)| (text_of(&vocab, &[*i]), *p))
                .collect::<Vec<_>>()
        );
        out.push(ids);
    }
    let matched = out[0].iter().zip(&out[1]).filter(|(a, b)| a == b).count();
    println!("gemma4-12B chat FA-vs-non-FA: {matched}/16");
    assert_eq!(out.len(), 2);
}

/// FA-vs-non-FA logit probe on the 26B-A4B (the parity prompt diverges at
/// step 3 under FA only): prints the per-step max-abs logit difference so the
/// FA error magnitude is measurable, independent of tie ordering.
///
///   cargo test --release -p llama --test gemma4_e2e -- --ignored --nocapture \
///       gemma4_26b_fa_vs_nonfa_probe
#[test]
#[ignore = "manual: 13.4 GiB MoE model, two short runs in release"]
fn gemma4_26b_fa_vs_nonfa_probe() {
    if !mem_guard("gemma4-26B-A4B", 18.0) {
        return;
    }
    let prompt: Vec<i32> = vec![2, 818, 5279, 529, 7001, 563];
    let mut per_fa: Vec<Vec<Vec<f32>>> = Vec::new();
    for fa in [false, true] {
        let Some(l) = load_real(GEMMA4_26B_A4B) else {
            return;
        };
        let mut dctx = gemma4_dctx(l, fa, 512);
        let mut steps = Vec::new();
        let pos: Vec<i32> = (0..prompt.len() as i32).collect();
        let mut cur = dctx.decode(&prompt, &pos).unwrap().to_vec();
        steps.push(cur.clone());
        let mut p = prompt.len() as i32;
        for _ in 0..4 {
            let tok = argmax(&cur);
            cur = dctx.decode(&[tok], &[p]).unwrap().to_vec();
            steps.push(cur.clone());
            p += 1;
        }
        for (k, st) in steps.iter().enumerate() {
            println!(
                "  gemma4-26B-A4B (FA={fa}) step {k} top8 {:?}",
                logprobs(st, 8)
                    .iter()
                    .map(|(i, l)| (*i, *l))
                    .collect::<Vec<_>>()
            );
        }
        per_fa.push(steps);
    }
    for k in 0..per_fa[0].len() {
        let a = &per_fa[0][k];
        let b = &per_fa[1][k];
        let maxabs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().map(|v| v.abs()).fold(0f32, f32::max);
        println!(
            "gemma4-26B-A4B FA-vs-non-FA step {k}: max abs logit diff {maxabs:.4e} (|logits| {scale:.2}), \
             argmax {} vs {}",
            argmax(a),
            argmax(b)
        );
    }
}

/// FA op probe with the **26B-A4B dense-layer geometry** (DK = DV = 512,
/// H = 16, H_kv = 2 — the only attention shape in the two gemma4 files that
/// the 12B does not also have; the 12B's dense layers have H_kv = 1). The FA
/// kernel's GQA head mapping and its VKQ accumulation are checked against a
/// direct softmax over the same f16 K/V rows.
///
///   cargo test -p llama --test gemma4_e2e -- --nocapture gemma4_fa_op_probe
#[test]
fn gemma4_fa_op_probe() {
    use ggml::compute::graph_compute;
    use ggml::graph::Graph;
    use ggml::types::GgmlType;

    for (dk, dv, h, h_kv) in [
        (512i64, 512i64, 16i64, 1i64),
        (512, 512, 16, 2),
        (256, 256, 16, 8),
        (256, 256, 16, 2),
    ] {
        let (t, n_kv) = (6i64, 6i64);
        let mut ctx = Context::new();

        // q [DK, T, H] (post-permute layout, like llama-graph.cpp:2621)
        let q = ctx.new_tensor_3d(GgmlType::F32, dk, t, h);
        ctx.arena_resize_tensor(q);
        ctx.with_f32_mut(q, |p| {
            for (i, v) in p.iter_mut().enumerate() {
                *v = ((i as f32) * 0.017).sin() * 0.5;
            }
        })
        .unwrap();

        // f16 k/v caches [n_embd_gqa, n_kv] + the head-strided views
        let k_cache = ctx.new_tensor_2d(GgmlType::F16, dk * h_kv, n_kv);
        let v_cache = ctx.new_tensor_2d(GgmlType::F16, dv * h_kv, n_kv);
        ctx.arena_resize_tensor(k_cache);
        ctx.arena_resize_tensor(v_cache);
        let kvals: Vec<f32> = (0..dk * h_kv * n_kv)
            .map(|i| ((i as f32) * 0.011).cos() * 0.4)
            .collect();
        let vvals: Vec<f32> = (0..dv * h_kv * n_kv)
            .map(|i| ((i as f32) * 0.013).sin() * 0.4)
            .collect();
        ctx.data_bytes_mut(k_cache)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(
                &kvals
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect::<Vec<_>>(),
            ));
        ctx.data_bytes_mut(v_cache)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(
                &vvals
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect::<Vec<_>>(),
            ));
        let rs_head_k = GgmlType::F16.row_size(dk as usize);
        let rs_gqa_k = GgmlType::F16.row_size((dk * h_kv) as usize);
        let rs_head_v = GgmlType::F16.row_size(dv as usize);
        let rs_gqa_v = GgmlType::F16.row_size((dv * h_kv) as usize);
        // permuted views [D, n_kv, H_kv] — what llama-graph's FA branch passes to
        // the op (flash_attn_core permutes the [D, H_kv, n_kv] cache views)
        let k_view = ctx.view_4d(
            k_cache,
            dk,
            n_kv,
            h_kv,
            1,
            rs_gqa_k,
            rs_head_k,
            rs_gqa_k * n_kv as usize,
            0,
        );
        let v_view = ctx.view_4d(
            v_cache,
            dv,
            n_kv,
            h_kv,
            1,
            rs_gqa_v,
            rs_head_v,
            rs_gqa_v * n_kv as usize,
            0,
        );

        // F16 causal mask [n_kv, T]
        let mask = ctx.new_tensor_2d(GgmlType::F16, n_kv, t);
        ctx.arena_resize_tensor(mask);
        {
            let m: &mut [half::f16] = bytemuck::cast_slice_mut(ctx.data_bytes_mut(mask).unwrap());
            m.fill(half::f16::NEG_INFINITY);
            for iq in 0..t as usize {
                for ik in 0..n_kv as usize {
                    if ik <= iq {
                        m[iq * n_kv as usize + ik] = half::f16::ZERO;
                    }
                }
            }
        }

        let scale = 1.0 / (dk as f32).sqrt();
        let out = ctx.flash_attn_ext(q, k_view, v_view, Some(mask), scale, 0.0, 0.0);
        let mut g = Graph::new(16);
        g.build_forward(&ctx, out);
        graph_compute(&mut ctx, &mut g, 1);

        let f32s = |g: &Context, id: ggml::TensorId| -> Vec<f32> {
            g.data_bytes(id)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let qf = f32s(&ctx, q);
        let of = f32s(&ctx, out);
        let f16at = |vals: &[f32], i: usize| half::f16::from_f32(vals[i]).to_f32();

        let mut worst = 0f32;
        for tt in 0..t {
            for hh in 0..h {
                let kvh = (hh / (h / h_kv)) as usize;
                let mut sw = Vec::new();
                for s in 0..=tt {
                    let mut d = 0f32;
                    for i in 0..dk {
                        // k is f16, q is rounded to f16 by the kernel
                        let kv = f16at(
                            &kvals,
                            i as usize + kvh * dk as usize + s as usize * (dk * h_kv) as usize,
                        );
                        // q is [DK, T, H]: element (k, t, h) at k + t*DK + h*DK*T
                        let qv = half::f16::from_f32(
                            qf[i as usize
                                + tt as usize * dk as usize
                                + hh as usize * (dk * t) as usize],
                        )
                        .to_f32();
                        d += kv * qv;
                    }
                    sw.push((d * scale).exp());
                }
                let sum: f32 = sw.iter().sum();
                for d in 0..dv {
                    let want: f32 = sw
                        .iter()
                        .enumerate()
                        .map(|(s, w)| {
                            w / sum
                                * f16at(
                                    &vvals,
                                    d as usize + kvh * dv as usize + s * (dv * h_kv) as usize,
                                )
                        })
                        .sum();
                    let got = of
                        [d as usize + hh as usize * dv as usize + tt as usize * (dv * h) as usize];
                    worst = worst.max((got - want).abs());
                }
            }
        }
        println!("FA op probe (DK={dk}, DV={dv}, H={h}, H_kv={h_kv}): worst abs diff {worst:.3e}");
        assert!(
            worst < 1e-2,
            "FA kernel mismatch for DK={dk} H_kv={h_kv}: {worst:.3e}"
        );
    }
}
