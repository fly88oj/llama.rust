//! swa_e2e.rs — the sliding-window-attention path: hparams derivation, the
//! `llama_kv_cache_iswa` geometry (two caches, per-layer selection, window
//! mask) and the long-context parity run against the reference server.
//!
//! Reference source: src/llama-kv-cache-iswa.cpp (the `iswa` pair),
//! src/llama-kv-cache.cpp `find_slot`/`apply_ubatch`/`set_input_kq_mask`,
//! src/llama-hparams.{h,cpp} (`n_swa`/`swa_type`/`is_masked_swa`),
//! src/llama-graph.cpp:3131-3133 (per-layer cache selection),
//! src/models/gemma4.cpp / openai-moe.cpp (what the local files ask for).
//!
//! Runs by default (metadata + mmap only, no tensor data read):
//!   * `swa_gemma4_12b_hparams_and_geometry` — n_swa = 1024, STANDARD, the
//!     5-SWA/1-dense layer pattern, per-layer head geometry, and the cache
//!     sizes `llama_kv_cache_iswa` derives from them
//!   * `swa_gemma4_26b_a4b_hparams_and_geometry` — the same for the MoE file
//!   * `swa_gpt_oss_20b_hparams_and_geometry` — n_swa = 128, STANDARD, the
//!     every-other-layer pattern (`load_swa_pattern(ml, 2)`)
//!   * `swa_long_prompt_tokenization` — the capture prompt tokenizes to the
//!     reference's `tokens_evaluated` (1164), which is what makes the
//!     long-context comparison a comparison of *graphs*, not tokenizers
//!   * `swa_synthetic_window_matches_naive` — a toy llama-shaped model run
//!     through both caches: below the window the SWA path must be bit-identical
//!     to the unified one, above it must match a naive windowed attention (and
//!     differ from the unified one)
//!
//! `#[ignore]`d (manual; release mode, minutes):
//!   * `gemma4_12b_long_context_swa_parity` — 1164-token prompt (> n_swa =
//!     1024) + 16 greedy tokens, SWA path, vs the fresh reference server
//!   * `gemma4_12b_long_context_unified_baseline` — the same run on the
//!     *unified* cache (what the port did before the SWA port): the "before"
//!     number the SWA run is compared against
//!
//! Reference capture (PARITY.md protocol: **fresh** llama-server, first request
//! on the slot, greedy, default FA, `cache_prompt=false`):
//!   parity/swa_long_capture.sh <model> 2048 8866 swa12b 16
//! i.e.
//!   /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server \
//!       -m <gemma-4-12B-it-QAT-Q4_0.gguf> -c 2048 -t 8 --port 8866 --host 127.0.0.1
//!   curl -s http://127.0.0.1:8866/completion -H 'Content-Type: application/json' \
//!       -d @<(python3 -c 'import json;print(json.dumps({"prompt":
//!            open("parity/swa_long_prompt.txt").read(),"n_predict":16,
//!            "temperature":0,"logprobs":20,"cache_prompt":false}))')
//! (captured 2026-09-24 against reference bd4f514db1; the server reports
//!  `kv_unified = 'true'`, `tokens_evaluated = 1164`, and the captured ids are
//!  `[108, 100, 236770, 236771 x13]` — text "\n\n<|channel>10000000000000").
//!
//! Measured 2026-09-24 (same machine, release, 8 threads, FA on — the
//! reference default, 512-token prefill chunks = the reference's n_ubatch);
//! "gap" = the largest absolute logprob delta on the reference's own top-1
//! token over the 16 steps:
//!
//! | run | greedy vs reference | gap |
//! |---|---|---|
//! | **SWA split** (`gemma4_12b_long_context_swa_parity`) | **16/16** | **0.0314** (step 0) |
//! | unified cache (`..._unified_baseline`, the pre-port behaviour) | 6/16 | 7.5631 (step 12) |
//!
//! The unified baseline diverges at step 2 (ref 236770 at -1.5596; port picked
//! 236779 at -2.8524, a 1.29-logprob gap on a 0.06-margin step) and drifts by
//! up to 7.6 logprobs — a structural break, not noise: every SWA layer was
//! attending 1024+ tokens the reference cannot see.
//!
//! Run the heavy tests with:
//!   cargo test --release -p llama --test swa_e2e -- --ignored --nocapture --test-threads=1

use std::path::Path;
use std::sync::Arc;

use ggml::types::GgmlType;
use ggml::{Context, Gguf, TensorId};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{
    Gemma4LayerWeights, Gemma4ModelWeights, Gemma4Params, LlamaLayerWeights, LlamaModelWeights,
};
use llama::hparams::{LlamaHparams, LlamaSwaType};
use llama::kv_cache::SwaCacheSpec;
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
const GPTOSS20B_MXFP4: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";

/// The capture prompt: 1164 gemma-4 tokens (reference `tokens_evaluated`),
/// i.e. 140 tokens past the n_swa = 1024 window. Shared with the capture
/// script so both sides see the same text.
const LONG_PROMPT: &str = include_str!("../../../parity/swa_long_prompt.txt");
/// The reference's token count for `LONG_PROMPT` (`tokens_evaluated`).
const LONG_PROMPT_TOKENS: usize = 1164;

/// Fresh reference server (:8866) + first request, greedy 16 (default FA).
/// `tokens_evaluated = 1164`; text "\n\n<|channel>10000000000000".
const REF16_GEMMA4_12B_LONG: [i32; 16] = [
    108, 100, 236770, 236771, 236771, 236771, 236771, 236771, 236771, 236771, 236771, 236771,
    236771, 236771, 236771, 236771,
];
/// Reference per-step top-8 (id, logprob), same capture — the pair-wise
/// comparison baseline at the first divergence.
#[rustfmt::skip]
const REF_TOP8_GEMMA4_12B_LONG: [[(i32, f32); 8]; 16] = [
    [(108, -0.4229), (107, -2.1804), (106, -2.1890), (101, -3.4042), (109, -3.4952), (100, -3.6013), (236743, -5.0263), (236771, -5.0700)],
    [(100, -0.0830), (14977, -3.5898), (236771, -4.9019), (236810, -4.9479), (236770, -5.9943), (236812, -6.0703), (236779, -6.1608), (236800, -6.1817)],
    [(236770, -1.5596), (45518, -1.6194), (236771, -2.1045), (236810, -2.4462), (236832, -2.5289), (236779, -2.5758), (236819, -3.0326), (236800, -3.1439)],
    [(236771, -0.9034), (236761, -2.1872), (107, -2.2991), (236770, -3.0828), (236778, -3.1096), (236888, -3.2871), (236819, -3.4628), (236800, -3.7436)],
    [(236771, -0.2190), (236770, -3.1340), (236778, -3.2218), (236908, -3.9732), (236810, -4.0698), (236761, -4.2914), (107, -4.5775), (236812, -5.0389)],
    [(236771, -0.1715), (236832, -2.6606), (236810, -4.0483), (236908, -4.4445), (236825, -4.8704), (236772, -4.9276), (236779, -5.0398), (568, -5.5782)],
    [(236771, -0.0317), (236779, -4.4278), (107, -5.3923), (236772, -6.3201), (236770, -6.3842), (236908, -6.5061), (108, -6.7884), (236810, -6.8067)],
    [(236771, -0.0075), (236770, -6.3861), (107, -6.6392), (236810, -7.0504), (236819, -7.8127), (236778, -8.0664), (236764, -8.1164), (108, -8.1183)],
    [(236771, -0.0097), (236810, -5.4776), (236770, -7.0030), (236812, -7.4383), (236800, -7.4483), (236778, -7.5614), (107, -7.7229), (236819, -7.7296)],
    [(236771, -0.0053), (236810, -6.4134), (236770, -7.0081), (107, -7.7072), (108, -7.7955), (236761, -8.1138), (236764, -8.6135), (236800, -8.8977)],
    [(236771, -0.0047), (236810, -6.8527), (236770, -7.2845), (108, -7.5384), (107, -7.8546), (236832, -8.1532), (236828, -8.3930), (753, -8.5880)],
    [(236771, -0.0014), (236770, -7.6254), (107, -8.9788), (236810, -9.2995), (108, -9.3252), (236829, -9.4409), (236832, -9.5305), (236761, -9.5702)],
    [(236771, -0.0067), (236770, -6.0815), (236810, -7.0600), (236819, -7.1683), (108, -7.4285), (107, -7.5015), (236764, -8.0591), (236832, -8.3708)],
    [(236771, -0.0038), (236770, -6.1974), (236810, -7.2226), (236832, -8.3623), (236819, -8.4589), (236828, -9.3128), (236764, -9.6733), (236825, -9.6994)],
    [(236771, -0.0023), (236810, -7.4816), (236770, -7.6612), (236779, -8.5519), (236819, -8.8579), (236764, -8.8672), (107, -8.8762), (236800, -9.2498)],
    [(236771, -0.0041), (236770, -6.1122), (236810, -7.2538), (107, -8.2109), (236761, -8.7403), (236812, -8.8039), (236832, -9.1450), (236819, -9.2837)],
];

// ---------------------------------------------------------------------------
// helpers (same shape as gemma4_e2e.rs / gpt_oss_e2e.rs)
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

/// the model's hparams as the `iswa` constructor wants them
fn swa_spec(hp: &LlamaHparams) -> SwaCacheSpec {
    SwaCacheSpec::from_hparams(hp)
}

// ---------------------------------------------------------------------------
// gemma4 wiring (same derivations as gemma4_e2e.rs)
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

/// gemma4 decode context; `swa` = Some(..) runs the `llama_kv_cache_iswa`
/// split, None the unified cache (the pre-port behaviour).
fn gemma4_dctx(mut l: Loaded, fa: bool, n_ctx: u32, n_batch: usize, swa: bool) -> DecodeContext {
    let mut p = gemma4_params(&l.model);
    p.attn.use_flash_attn = fa;
    let spec = swa_spec(&l.model.hparams);
    let w = gemma4_weights(&l.model);
    let gctx = std::mem::replace(&mut l.model.ctx, Context::new());
    let attn = p.attn;
    if swa {
        DecodeContext::new_with_swa(
            gctx,
            ForwardWeights::Gemma4(w, p),
            attn,
            n_ctx,
            8,
            n_batch,
            spec,
        )
    } else {
        DecodeContext::new_with(gctx, ForwardWeights::Gemma4(w, p), attn, n_ctx, 8, n_batch)
    }
}

/// Prefill in `n_batch`-sized chunks (the reference splits the same prompt into
/// `n_ubatch` chunks, llama-kv-cache-iswa.cpp:187-236 tries `split_simple`
/// first), then greedy decode; returns (ids, per-step top-20 logprobs,
/// last-token logits, prefill time, gen time).
fn run_greedy_chunked(
    dctx: &mut DecodeContext,
    prompt: &[i32],
    n_gen: usize,
    n_batch: usize,
) -> (
    Vec<i32>,
    Vec<Vec<(i32, f32)>>,
    Vec<f32>,
    std::time::Duration,
    std::time::Duration,
) {
    let t0 = std::time::Instant::now();
    let mut last = Vec::new();
    let mut pos0 = 0i32;
    for chunk in prompt.chunks(n_batch) {
        let pos: Vec<i32> = (pos0..pos0 + chunk.len() as i32).collect();
        last = dctx.decode(chunk, &pos).expect("prefill").to_vec();
        pos0 += chunk.len() as i32;
    }
    let prefill = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut ids = Vec::new();
    let mut lps = Vec::new();
    let mut cur = last;
    for _ in 0..n_gen {
        lps.push(logprobs(&cur, 20));
        let tok = argmax(&cur);
        ids.push(tok);
        cur = dctx.decode(&[tok], &[pos0]).expect("step").to_vec();
        pos0 += 1;
    }
    let gen = t1.elapsed();
    (ids, lps, cur, prefill, gen)
}

/// MATCH x/n against the reference ids, with the first divergence and its
/// pair-wise logprob gap on the reference's own top-1 token.
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
    // pair-wise top-1 logprob gap at every step (teacher-free, but stable)
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (k, lp) in lps.iter().enumerate() {
        let want_id = r[k];
        if let Some((_, wp)) = REF_TOP8_GEMMA4_12B_LONG[k]
            .iter()
            .find(|(id, _)| *id == want_id)
        {
            let gap = lp
                .iter()
                .find(|(id, _)| *id == want_id)
                .map(|(_, mine)| (mine - wp).abs())
                .unwrap_or(f32::NAN);
            if gap > worst {
                worst = gap;
                worst_at = k;
            }
        }
    }
    println!("{label}: worst |delta logprob| on the reference top-1: {worst:.4} (step {worst_at})");
    if let Some(k) = first {
        print!("{label}: step {k} ours  : ");
        for (id, p) in lps[k].iter().take(8) {
            print!("{}{}({:.3}) ", id, text_of(vocab, &[*id]), p);
        }
        println!();
        let r8 = &REF_TOP8_GEMMA4_12B_LONG[k];
        let wp = r8[0].1;
        if let Some((_, mp)) = lps[k].iter().find(|(id, _)| *id == r8[0].0) {
            println!(
                "{label}: step {k} pair-wise logprob: want {wp:.4} got {mp:.4} (gap {:+.4})",
                (mp - wp).abs()
            );
        }
        println!(
            "{label}: step {k} ref top8 {:?} margin {:.4}",
            r8.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            r8[0].1 - r8[1].1
        );
    }
}

// ---------------------------------------------------------------------------
// 1. hparams + cache geometry (default run)
// ---------------------------------------------------------------------------

/// The `llama_kv_cache_iswa` sizes the local gemma-4 file asks for, at the
/// long-context test's `-c 2048` / n_ubatch 512 (llama-model.cpp:2481-2482).
#[test]
fn swa_gemma4_12b_hparams_and_geometry() {
    let Some(l) = load_real(GEMMA4_12B) else {
        return;
    };
    let hp = &l.model.hparams;
    assert_eq!(hp.n_swa, 1024);
    assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
    assert!(hp.is_swa_any());
    // 5 sliding + 1 dense, period 6 (attention.sliding_window_pattern)
    let n_swa_layers = (0..hp.n_layer())
        .filter(|&il| hp.is_swa(il as usize))
        .count();
    assert_eq!(n_swa_layers, 40);
    assert!(hp.is_swa(0) && hp.is_swa(4) && !hp.is_swa(5) && hp.is_swa(6) && !hp.is_swa(11));
    assert_eq!(hp.n_layer(), 48);

    // the two caches' cells: `size_base = cparams.n_ctx_seq` (2048),
    // `size_swa = swa_full ? size_base : PAD(min(size_base, n_swa*n_seq_max +
    // n_ubatch), 256)` (llama-kv-cache-iswa.cpp:69-81). `swa_full` is the
    // llama_context_params / common default (llama-context.cpp:3729).
    let spec = swa_spec(hp);
    assert!(spec.swa_full);
    let size_base = 2048u32;
    let size_swa = llama::kv_cache::swa_cache_size(
        size_base,
        spec.n_swa,
        spec.n_seq_max,
        512,
        spec.unified,
        spec.swa_full,
    );
    assert_eq!(
        size_swa, 2048,
        "swa_full: the SWA cache mirrors the base one"
    );
    assert_eq!(
        llama::kv_cache::swa_cache_size(size_base, spec.n_swa, 1, 512, true, false),
        1536,
        "without swa_full: PAD(min(2048, 1024 + 512), 256)"
    );
    // the partition itself, as the two `layer_filter_cb`s build it
    assert_eq!(spec.is_swa.len(), 48);
    assert_eq!(spec.is_swa.iter().filter(|&&s| s).count(), 40);
}

#[test]
fn swa_gemma4_26b_a4b_hparams_and_geometry() {
    let Some(l) = load_real(GEMMA4_26B_A4B) else {
        return;
    };
    let hp = &l.model.hparams;
    assert_eq!(hp.n_swa, 1024);
    assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
    assert_eq!(hp.n_layer(), 30);
    assert!(!hp.is_swa(5) && hp.is_swa(4) && hp.is_swa(6));
    let spec = swa_spec(hp);
    assert_eq!(spec.is_swa.len(), 30);
    // MoE file: same window, geometry per layer (SWA 256x8, dense 512x2)
    assert_eq!(hp.n_embd_k_gqa(0), 256 * 8);
    assert_eq!(hp.n_embd_k_gqa(5), 512 * 2);
}

/// gpt-oss: `load_swa_pattern(ml, 2)` with `dense_first = false`
/// (openai-moe.cpp:11 → llama-model.cpp:3308-3314
/// `is_swa_impl[il] = n_pattern == 0 || (il % n_pattern < n_pattern - 1)`,
/// i.e. every *even* layer is SWA) and n_swa = 128
/// (`attention.sliding_window`, openai-moe.cpp:8).
#[test]
fn swa_gpt_oss_20b_hparams_and_geometry() {
    let Some(l) = load_real(GPTOSS20B_MXFP4) else {
        return;
    };
    let hp = &l.model.hparams;
    assert_eq!(hp.n_swa, 128);
    assert_eq!(hp.swa_type, LlamaSwaType::STANDARD);
    assert_eq!(hp.n_layer(), 24);
    for il in 0..24u32 {
        assert_eq!(hp.is_swa(il as usize), il % 2 == 0, "layer {il}");
    }
    let spec = swa_spec(hp);
    assert_eq!(spec.is_swa.iter().filter(|&&s| s).count(), 12);
    // without swa_full: PAD(min(2048, 128 + 512), 256) = 768
    assert_eq!(
        llama::kv_cache::swa_cache_size(2048, spec.n_swa, 1, 512, true, false),
        768
    );
    // `rope_freq_base_train_swa` has its own key (openai-moe.cpp:14-17)
    assert_eq!(hp.rope_freq_base_train_swa, hp.rope_freq_base_train);
}

/// The capture prompt must be *the* reference prompt: same tokenizer, same
/// count (`tokens_evaluated = 1164` in the reference response). Everything
/// past token 1024 is outside the SWA window.
#[test]
fn swa_long_prompt_tokenization() {
    let Some(l) = load_real(GEMMA4_12B) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let ids = vocab.tokenize(LONG_PROMPT, true, true);
    assert_eq!(
        ids.len(),
        LONG_PROMPT_TOKENS,
        "prompt must be the captured one"
    );
    assert_eq!(ids[0], 2, "BOS");
    assert!(
        LONG_PROMPT_TOKENS > 1024,
        "the prompt must cross the window"
    );
    println!(
        "long prompt: {} tokens (n_swa = 1024 -> {} tokens outside the window)",
        ids.len(),
        ids.len() - 1024
    );
}

// ---------------------------------------------------------------------------
// 2. the split's mechanics on a synthetic model (default run)
// ---------------------------------------------------------------------------

/// Toy llama-shaped model (2 layers, FFN weights zeroed so the naive reference
/// only needs the attention path) driven through **both** caches:
///
///   * while `n_kv <= n_swa` the SWA path is bit-identical to the unified one
///     (the window has not cut anything yet, so the two mask/cache pairs are
///     the same computation);
///   * past the window the SWA logits must match a **naive windowed attention**
///     (`p1 - p0 >= n_swa` dropped, llama-hparams.h:479-484) and must *differ*
///     from the unified run — i.e. the test can tell the two paths apart.
#[test]
fn swa_synthetic_window_matches_naive() {
    const N_EMBD: usize = 32;
    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const ND: usize = 8;
    const N_FF: usize = 16;
    const N_VOCAB: usize = 32;
    const N_LAYER: usize = 2;
    const N_SWA: u32 = 4;
    const EPS: f32 = 1e-5;

    struct Ws {
        tok_embd: Vec<f32>,
        output_norm: Vec<f32>,
        output: Vec<f32>,
        /// per layer: attn_norm, wq, wk, wv, wo, ffn_norm
        lws: Vec<[Vec<f32>; 6]>,
    }

    struct Toy {
        dctx: DecodeContext,
        ws: Ws,
    }

    fn f32s(g: &Context, id: TensorId) -> Vec<f32> {
        g.data_bytes(id)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    // deterministic toy (LCG); `swa` selects the cache shape, the weights are
    // identical in both runs
    let build_toy = |swa: bool| -> Toy {
        let mut gctx = Context::new();
        let mut rng_state = 4242u32;
        let mut rnd = move || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state >> 8) as f32 / 16777216.0) * 0.4 - 0.2
        };
        fn mk2(
            g: &mut Context,
            rnd: &mut impl FnMut() -> f32,
            n0: i64,
            n1: i64,
            zero: bool,
        ) -> TensorId {
            let id = g.new_tensor_2d(GgmlType::F32, n0, n1);
            g.arena_resize_tensor(id);
            g.with_f32_mut(id, |p| {
                for v in p.iter_mut() {
                    *v = if zero { 0.0 } else { rnd() };
                }
            })
            .unwrap();
            id
        }
        let tok_embd = mk2(&mut gctx, &mut rnd, N_EMBD as i64, N_VOCAB as i64, false);
        let output = mk2(&mut gctx, &mut rnd, N_EMBD as i64, N_VOCAB as i64, false);
        let output_norm = mk2(&mut gctx, &mut rnd, N_EMBD as i64, 1, false);
        let n_gqa = (ND * N_HEAD_KV) as i64;
        let mut ids = Vec::new();
        for _ in 0..N_LAYER {
            let an = mk2(&mut gctx, &mut rnd, N_EMBD as i64, 1, false);
            let wq = mk2(
                &mut gctx,
                &mut rnd,
                N_EMBD as i64,
                (ND * N_HEAD) as i64,
                false,
            );
            let wk = mk2(&mut gctx, &mut rnd, N_EMBD as i64, n_gqa, false);
            let wv = mk2(&mut gctx, &mut rnd, N_EMBD as i64, n_gqa, false);
            let wo = mk2(
                &mut gctx,
                &mut rnd,
                (ND * N_HEAD) as i64,
                N_EMBD as i64,
                false,
            );
            let fnorm = mk2(&mut gctx, &mut rnd, N_EMBD as i64, 1, false);
            // FFN weights are *zero*: gate/up/down contribute nothing
            let gate = mk2(&mut gctx, &mut rnd, N_EMBD as i64, N_FF as i64, true);
            let up = mk2(&mut gctx, &mut rnd, N_EMBD as i64, N_FF as i64, true);
            let down = mk2(&mut gctx, &mut rnd, N_FF as i64, N_EMBD as i64, true);
            ids.push((an, wq, wk, wv, wo, fnorm, gate, up, down));
        }
        let ws = Ws {
            tok_embd: f32s(&gctx, tok_embd),
            output_norm: f32s(&gctx, output_norm),
            output: f32s(&gctx, output),
            lws: ids
                .iter()
                .map(|w| {
                    [
                        f32s(&gctx, w.0),
                        f32s(&gctx, w.1),
                        f32s(&gctx, w.2),
                        f32s(&gctx, w.3),
                        f32s(&gctx, w.4),
                        f32s(&gctx, w.5),
                    ]
                })
                .collect(),
        };
        let layers: Vec<LlamaLayerWeights> = ids
            .iter()
            .map(|w| LlamaLayerWeights {
                attn_norm: w.0,
                wq: w.1,
                wk: w.2,
                wv: w.3,
                wo: w.4,
                wq_b: None,
                wk_b: None,
                wv_b: None,
                wo_b: None,
                ffn_norm: w.5,
                ffn_gate: w.6,
                ffn_down: w.8,
                ffn_up: w.7,
                ffn_gate_b: None,
                ffn_down_b: None,
                ffn_up_b: None,
            })
            .collect();
        let weights = LlamaModelWeights {
            tok_embd,
            output_norm,
            output,
            output_b: None,
            layers,
        };
        let attn = AttnParams {
            n_head: N_HEAD as i64,
            n_head_kv: N_HEAD_KV as i64,
            n_embd_head_k: ND as i64,
            n_embd_head_v: ND as i64,
            n_rot: ND as i64,
            rope_mode: 2, // NEOX
            n_ctx_orig: 512,
            freq_base: 10000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            norm_eps: EPS,
            use_flash_attn: false,
        };
        let dctx = if swa {
            let spec = SwaCacheSpec {
                n_swa: N_SWA,
                swa_type: LlamaSwaType::STANDARD,
                is_swa: vec![true; N_LAYER],
                ..SwaCacheSpec::default()
            };
            DecodeContext::new_with_swa(gctx, ForwardWeights::Llama(weights), attn, 64, 4, 16, spec)
        } else {
            DecodeContext::new_with(gctx, ForwardWeights::Llama(weights), attn, 64, 4, 16)
        };
        Toy { dctx, ws }
    };

    // naive forward over the whole prefix, F16 KV rows like the cache, with
    // `n_swa = 0` (full causal) or the STANDARD window; returns the logits of
    // every query position.
    let naive_all = |ws: &Ws, toks: &[i32], n_swa: u32| -> Vec<Vec<f32>> {
        let scale = 1.0f32 / (ND as f32).sqrt();
        let dot = |w: &[f32], x: &[f32], n0: usize| -> Vec<f32> {
            let n1 = w.len() / n0.max(1);
            (0..n1)
                .map(|j| (0..n0).map(|i| w[i + j * n0] * x[i]).sum::<f32>())
                .collect()
        };
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let sum: f32 = x.iter().map(|v| v * v).sum();
            let sc = 1.0 / (sum / x.len() as f32 + EPS).sqrt();
            x.iter().zip(w).map(|(v, wv)| v * sc * wv).collect()
        };
        let mut x: Vec<Vec<f32>> = toks
            .iter()
            .map(|&t| ws.tok_embd[t as usize * N_EMBD..][..N_EMBD].to_vec())
            .collect();
        for li in 0..N_LAYER {
            let [an, wq, wk, wv, wo, _fnorm] = &ws.lws[li];
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            for (t, xt) in x.iter().enumerate() {
                let xn = rms(xt, an);
                let mut q = dot(wq, &xn, N_EMBD);
                let mut k = dot(wk, &xn, N_EMBD);
                let mut v = dot(wv, &xn, N_EMBD);
                // NEOX rope, pairs (i, i + nd/2)
                let rope = |buf: &mut Vec<f32>, heads: usize| {
                    for hh in 0..heads {
                        let b = hh * ND;
                        for i in 0..ND / 2 {
                            let fi = 1.0f32 / 10000.0f32.powf((2.0 * i as f32) / ND as f32);
                            let th = t as f32 * fi;
                            let (c, s2) = (th.cos(), th.sin());
                            let x0 = buf[b + i];
                            let x1 = buf[b + i + ND / 2];
                            buf[b + i] = x0 * c - x1 * s2;
                            buf[b + i + ND / 2] = x0 * s2 + x1 * c;
                        }
                    }
                };
                rope(&mut q, N_HEAD);
                rope(&mut k, N_HEAD_KV);
                // the KV cache is F16 (kv_cache.rs:6-7 layout)
                for z in k.iter_mut().chain(v.iter_mut()) {
                    *z = half::f16::from_f32(*z).to_f32();
                }
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }
            let mut outs = Vec::new();
            for (t, xt) in x.iter().enumerate() {
                let mut acc = vec![0f32; ND * N_HEAD];
                for hh in 0..N_HEAD {
                    let kvh = hh / (N_HEAD / N_HEAD_KV);
                    let mut sw = Vec::new();
                    for s2 in 0..=t {
                        // the window: p1 - p0 >= n_swa is dropped
                        if n_swa > 0 && (t - s2) as u32 >= n_swa {
                            continue;
                        }
                        let mut d = 0f32;
                        for i in 0..ND {
                            d += ks[s2][kvh * ND + i] * qs[t][hh * ND + i];
                        }
                        sw.push((d * scale).exp());
                    }
                    let sum: f32 = sw.iter().sum();
                    let mut idx = 0;
                    for s2 in 0..=t {
                        if n_swa > 0 && (t - s2) as u32 >= n_swa {
                            continue;
                        }
                        for i in 0..ND {
                            acc[hh * ND + i] += sw[idx] / sum * vs[s2][kvh * ND + i];
                        }
                        idx += 1;
                    }
                }
                let a = dot(wo, &acc, N_EMBD);
                // FFN weights are zero -> the layer output is the residual only
                outs.push(xt.iter().zip(&a).map(|(p, q)| p + q).collect::<Vec<f32>>());
            }
            x = outs;
        }
        // final norm + lm head, per position
        x.iter()
            .map(|h| {
                let xn = rms(h, &ws.output_norm);
                dot(&ws.output, &xn, N_EMBD)
            })
            .collect()
    };

    // 6 tokens: n_kv <= n_swa = 4 for the first 4, the window cuts at 5+
    let toks: Vec<i32> = vec![3, 7, 11, 19, 5, 23, 2, 13];
    let n_short = 4usize;

    let mut uni = build_toy(false);
    let mut swa = build_toy(true);
    assert!(
        !uni.dctx.kv.has_swa(),
        "baseline toy must run the unified cache"
    );
    assert!(swa.dctx.kv.has_swa(), "SWA toy must run the iswa pair");
    assert_eq!(swa.dctx.kv.swa_cache().unwrap().layer_ids, vec![0, 1]);

    // (a) below the window both paths are identical
    let a1 = uni
        .dctx
        .decode(&toks[..n_short], &[0, 1, 2, 3])
        .unwrap()
        .to_vec();
    let b1 = swa
        .dctx
        .decode(&toks[..n_short], &[0, 1, 2, 3])
        .unwrap()
        .to_vec();
    assert_eq!(
        a1, b1,
        "n_kv <= n_swa: SWA path must be bit-identical to unified"
    );

    // (b) step past the window: n_kv = 5 > n_swa = 4
    let a2 = uni
        .dctx
        .decode(&toks[n_short..n_short + 1], &[4])
        .unwrap()
        .to_vec();
    let b2 = swa
        .dctx
        .decode(&toks[n_short..n_short + 1], &[4])
        .unwrap()
        .to_vec();
    let naive_win = naive_all(&swa.ws, &toks[..n_short + 1], N_SWA);
    let naive_full = naive_all(&swa.ws, &toks[..n_short + 1], 0);
    let last_win = &naive_win[n_short];
    let last_full = &naive_full[n_short];
    let d = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let e_swa = d(&b2, last_win);
    let e_uni = d(&a2, last_full);
    let e_gap = d(&b2, &a2);
    println!(
        "synthetic SWA: |swa-naive_window| {e_swa:.4}, |unified-naive_causal| {e_uni:.4}, \
         |swa-unified| {e_gap:.4}"
    );
    assert!(
        e_swa < 5e-2,
        "SWA path must match the naive windowed attention: {e_swa}"
    );
    assert!(
        e_uni < 5e-2,
        "unified path must match the naive causal attention: {e_uni}"
    );
    assert!(
        e_gap > 1e-2,
        "the window must change the result once n_kv > n_swa (else the test proves nothing): {e_gap}"
    );
    // and the two paths must stay different in the following steps
    let a3 = uni.dctx.decode(&toks[5..6], &[5]).unwrap().to_vec();
    let b3 = swa.dctx.decode(&toks[5..6], &[5]).unwrap().to_vec();
    let naive_win = naive_all(&swa.ws, &toks[..6], N_SWA);
    let e3 = d(&b3, &naive_win[5]);
    println!(
        "synthetic SWA step 5: |swa-naive_window| {e3:.4}, |swa-unified| {:.4}",
        d(&b3, &a3)
    );
    assert!(e3 < 5e-2, "step 5 SWA must match the naive window: {e3}");
    assert!(
        d(&b3, &a3) > 1e-2,
        "step 5 must still differ from the unified path"
    );

    // (c) a 2-token batch: `decode_all` rows must match the naive windowed
    // reference for every position, and the gap to the unified path grows as
    // more keys leave the window
    let uni_all = uni.dctx.decode_all(&toks[6..8], &[6, 7]).unwrap();
    let swa_all = swa.dctx.decode_all(&toks[6..8], &[6, 7]).unwrap();
    let naive_win8 = naive_all(&swa.ws, &toks[..8], N_SWA);
    let mut worst = 0f32;
    let mut gap_last = 0f32;
    for k in 0..2 {
        let got = &swa_all[k * N_VOCAB..(k + 1) * N_VOCAB];
        let want = &naive_win8[6 + k];
        worst = worst.max(d(got, want));
        let uni_row = &uni_all[k * N_VOCAB..(k + 1) * N_VOCAB];
        println!(
            "synthetic SWA pos {}: |swa-naive_window| {:.4}, |swa-unified| {:.4}",
            6 + k,
            d(got, want),
            d(got, uni_row)
        );
        gap_last = d(got, uni_row);
    }
    assert!(
        worst < 5e-2,
        "batch SWA rows must match the naive window: {worst}"
    );
    assert!(
        gap_last > 0.03,
        "at position 7 three keys are outside the window -> the paths must diverge clearly: {gap_last}"
    );
    // the SWA cache holds the same tokens as the base cache (separate cells);
    // n_kv is 256-padded (capped at this cache's size 64)
    assert_eq!(swa.dctx.kv.used_cells(), 8);
    assert_eq!(swa.dctx.kv.n_kv(), 64);
    assert_eq!(swa.dctx.kv.n_kv_swa(), 64);
    assert_eq!(
        swa.dctx.kv.swa_cache().unwrap().size,
        64,
        "swa_full default"
    );
}

// ---------------------------------------------------------------------------
// 3. long-context reference parity (manual; PARITY.md protocol)
// ---------------------------------------------------------------------------

/// gemma-4-12B-it-QAT-Q4_0, 1164-token prompt (140 tokens past n_swa = 1024),
/// FA (the reference default), 512-token prefill chunks (= the reference's
/// n_ubatch) + 16 greedy tokens vs the fresh reference first request.
///
///   cargo test --release -p llama --test swa_e2e -- --ignored --nocapture \
///       gemma4_12b_long_context_swa_parity
#[test]
#[ignore = "manual: 6.5 GiB model, 1164-token prefill + 16 greedy steps in release"]
fn gemma4_12b_long_context_swa_parity() {
    long_context_run(true);
}

/// The same run on the **unified** cache — the port's behaviour before the
/// `llama_kv_cache_iswa` port. Kept so the before/after numbers of PARITY.md's
/// SWA section can be reproduced with one command.
///
///   cargo test --release -p llama --test swa_e2e -- --ignored --nocapture \
///       gemma4_12b_long_context_unified_baseline
#[test]
#[ignore = "manual: 6.5 GiB model, 1164-token prefill + 16 greedy steps in release"]
fn gemma4_12b_long_context_unified_baseline() {
    long_context_run(false);
}

fn long_context_run(swa: bool) {
    if swa {
        if !mem_guard("gemma4-12B long", 10.0) {
            return;
        }
    } else if !mem_guard("gemma4-12B long (unified)", 10.0) {
        return;
    }
    let label = if swa {
        "gemma4-12B long SWA"
    } else {
        "gemma4-12B long unified"
    };

    let Some(l0) = load_real(GEMMA4_12B) else {
        return;
    };
    let vocab = Vocab::load(&l0.gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.tokenize(LONG_PROMPT, true, true);
    let file_gib = l0.size_bytes as f64 / 1073741824.0;
    println!(
        "{label}: prompt {} tokens (reference tokens_evaluated = {}), {file_gib:.2} GiB",
        prompt.len(),
        LONG_PROMPT_TOKENS
    );
    assert_eq!(
        prompt.len(),
        LONG_PROMPT_TOKENS,
        "prompt must match the capture"
    );
    drop(l0);

    let n_batch = 512;
    let Some(l) = load_real(GEMMA4_12B) else {
        return;
    };
    let mut dctx = gemma4_dctx(l, true, 2048, n_batch, swa);
    if swa {
        assert!(dctx.kv.has_swa(), "the iswa pair must be in place");
        assert_eq!(dctx.kv.swa_cache().unwrap().size, 2048);
        assert!(dctx.kv.layer_is_swa(0) && dctx.kv.layer_is_swa(4) && !dctx.kv.layer_is_swa(5));
    } else {
        assert!(!dctx.kv.has_swa(), "baseline run uses the unified cache");
    }
    let (ids, lps, last, prefill, gen) = run_greedy_chunked(&mut dctx, &prompt, 16, n_batch);
    assert!(last.iter().all(|v| v.is_finite()), "logits finite");
    println!(
        "{label}: prefill {} tok in {prefill:?} ({:.2} t/s); gen 16 in {gen:?} ({:.3} t/s)",
        prompt.len(),
        prompt.len() as f32 / prefill.as_secs_f32(),
        16.0 / gen.as_secs_f32(),
    );
    report_parity(label, &vocab, &ids, &lps, &REF16_GEMMA4_12B_LONG);
    println!(
        "{label} last-row top-5: {:?}",
        logprobs(&last, 5)
            .iter()
            .map(|(i, p)| (text_of(&vocab, &[*i]), *p))
            .collect::<Vec<_>>()
    );
}
