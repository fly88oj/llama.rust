//! ctx_shift_e2e.rs — the KV position-shift family and the K-shift graph:
//! `llama_kv_cache::seq_add/seq_keep/seq_div/seq_cp`
//! (llama-kv-cache.cpp:451-657), `build_graph_shift` + `llama_context::
//! memory_update` (llama-kv-cache.cpp:2003-2053 / llama-context.cpp:845-905 /
//! :1805) and the server's context shift on top of them
//! (server-context.cpp:2909-2972).
//!
//! Runs by default (toy models, seconds):
//!  * `k_shift_value_preserving_unified` — after `seq_add` drops the first d
//!    positions, the continued generation equals a fresh run of the same
//!    logical (shifted) sequence to ~5e-4: re-rotating a roped K row by -d
//!    equals roping it at the shifted position up to fp rounding, and at the
//!    toy's weight scale the *truncation* perturbation (below) is of the same
//!    order. The K-shift graph must run inside the next `decode` (the port's
//!    `memory_update`) and clear the accumulators.
//!  * `k_shift_value_preserving_iswa` — the same criterion on the `iswa` pair
//!    (one SWA layer + one dense layer): a uniform shift keeps every
//!    query-key distance, so the sliding-window mask is invariant too, and
//!    each cache's own shift vector rotates its own layers.
//!  * `k_shift_value_preserving_large_drop` — the same at the real shift's
//!    scale (prefill 120, drop 63).
//!
//! `#[ignore]`d (manual; release, real model — qwen2.5-0.5b-instruct):
//!  * `qwen2_5_long_context_shift_run` — `-c 128` (padded to 256,
//!    llama-context.cpp:290) with the server's exact shift math (n_keep = 0 +
//!    bos, n_discard = n_left/2, server-context.cpp:2935-2968) and 200 tokens
//!    of greedy generation: pins the hand-computed shift schedule, the splice
//!    arithmetic and finite logits through the shifts. **Semantics:** the
//!    reference's context shift is a *truncation heuristic* — the kept
//!    tokens' cached K/V still encode attention over the discarded context —
//!    so the shifted stream does NOT equal a fresh decode of the spliced
//!    sequence (nor an unshifted run), in the reference exactly as in the
//!    port; the numbers proving this are printed. The exact-stream criterion
//!    (a) — the fresh reference server with the same flags — is
//!    `parity/run_server_parity_shift.sh`.
//!  * `qwen2_5_k_shift_rotation_identity` — what IS exact: the K-shift
//!    rotation itself. Layer 0's cached K is context-independent before RoPE
//!    (`W_k · RMS(embd)`), so after `seq_add(-63)` + the K-shift graph it
//!    equals the fresh row at the shifted position to one extra F16 rounding
//!    (measured 0.125 on rows of magnitude ~10); deeper layers carry the
//!    truncation perturbation and are only reported.
//!
//! Run:
//!   cargo test -p llama --test ctx_shift_e2e
//!   cargo test --release -p llama --test ctx_shift_e2e -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use ggml::types::GgmlType;
use ggml::{Context, Gguf, TensorId};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{LlamaLayerWeights, LlamaModelWeights};
use llama::hparams::LlamaSwaType;
use llama::kv_cache::SwaCacheSpec;
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const QWEN25: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

// ---------------------------------------------------------------------------
// toy harness — a 2-layer llama with deterministic weights (the shape of
// context.rs's toy_qwen2; FFN weights are zero so the attention/KV path
// dominates the signal)
// ---------------------------------------------------------------------------

const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const ND: i64 = 16;
const N_VOCAB: i64 = 100;

/// `swa == None` builds the unified cache; `Some((n_swa, is_swa))` the iswa
/// pair (llama-kv-cache-iswa.cpp:52-106).
fn toy_llama(
    seed: u32,
    n_ctx: u32,
    n_batch: usize,
    swa: Option<(u32, Vec<bool>)>,
) -> DecodeContext {
    let mut gctx = Context::new();
    let mut rng_state = seed;
    let mut rnd = move |lo: f32, hi: f32| {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        lo + (rng_state >> 8) as f32 / 16777216.0 * (hi - lo)
    };
    let mut mk2 = |g: &mut Context, n0: i64, n1: i64, zero: bool| -> TensorId {
        let id = g.new_tensor_2d(GgmlType::F32, n0, n1);
        g.arena_resize_tensor(id);
        g.with_f32_mut(id, |p| {
            for v in p.iter_mut() {
                *v = if zero { 0.0 } else { rnd(-0.1, 0.1) };
            }
        })
        .unwrap();
        id
    };
    let tok_embd = mk2(&mut gctx, N_EMBD, N_VOCAB, false);
    let output = mk2(&mut gctx, N_EMBD, N_VOCAB, false);
    let output_norm = mk2(&mut gctx, N_EMBD, 1, false);
    let layers: Vec<LlamaLayerWeights> = (0..2)
        .map(|_| LlamaLayerWeights {
            attn_norm: mk2(&mut gctx, N_EMBD, 1, false),
            wq: mk2(&mut gctx, N_EMBD, N_EMBD, false),
            wk: mk2(&mut gctx, N_EMBD, ND * N_HEAD_KV, false),
            wv: mk2(&mut gctx, N_EMBD, ND * N_HEAD_KV, false),
            wo: mk2(&mut gctx, N_EMBD, N_EMBD, false),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            wo_b: None,
            ffn_norm: mk2(&mut gctx, N_EMBD, 1, false),
            ffn_gate: mk2(&mut gctx, N_EMBD, 32, true),
            ffn_down: mk2(&mut gctx, 32, N_EMBD, true),
            ffn_up: mk2(&mut gctx, N_EMBD, 32, true),
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
        n_head: N_HEAD,
        n_head_kv: N_HEAD_KV,
        n_embd_head_k: ND,
        n_embd_head_v: ND,
        n_rot: ND,
        rope_mode: 2, // NEOX
        n_ctx_orig: 512,
        freq_base: 1000000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        norm_eps: 1e-6,
        use_flash_attn: false,
    };
    match swa {
        Some((n_swa, is_swa)) => {
            let spec = SwaCacheSpec {
                n_swa,
                swa_type: LlamaSwaType::STANDARD,
                is_swa,
                ..SwaCacheSpec::default()
            };
            DecodeContext::new_with_swa(
                gctx,
                ForwardWeights::Llama(weights),
                attn,
                n_ctx,
                4,
                n_batch,
                spec,
            )
        }
        None => DecodeContext::new_with(
            gctx,
            ForwardWeights::Llama(weights),
            attn,
            n_ctx,
            4,
            n_batch,
        ),
    }
}

fn argmax(v: &[f32]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as i32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// decode `toks` at `pos` in `n_batch`-sized calls (the driver's chunking)
fn decode_chunked(d: &mut DecodeContext, toks: &[i32], pos: &[i32], n_batch: usize) -> Vec<f32> {
    let mut last = Vec::new();
    for (off, c) in toks.chunks(n_batch).enumerate() {
        let p = &pos[off * n_batch..off * n_batch + c.len()];
        last = d.decode(c, p).expect("decode").to_vec();
    }
    last
}

// ---------------------------------------------------------------------------
// 1. the K-shift graph is value-preserving (both cache shapes)
// ---------------------------------------------------------------------------

/// Run A prefill 0..N at 0..N, `seq_add(0, -1, -1, -d)` (the server's shift
/// with n_keep = 0 — server-context.cpp:2952 with p0 = n_keep + n_discard
/// degenerating to the whole sequence), then teacher-force N..N+8. Run B
/// decodes the same logical history fresh: toks[d..N] at 0..N-d, then the
/// same N..N+8. Every step's logits must agree (argmax exactly, values up to
/// the F16 double-rounding of the re-rotated K rows plus — at the toy's
/// weight scale, equally tiny — the truncation perturbation: the kept
/// tokens' cached K/V still encode attention over the dropped prefix; see
/// `qwen2_5_long_context_shift_run` for where that perturbation stops being
/// negligible).
fn value_preservation(swa: Option<(u32, Vec<bool>)>) {
    let n_ctx = 96u32;
    let n_batch = 16usize;
    let d = 8usize;
    let n = 48usize;
    value_preservation_at(swa, n_ctx, n_batch, d, n);
}

/// the same criterion at qwen2.5's scale (prefill 120, drop 63, ctx 160) —
/// the dimensions the real-model test shifts at
#[test]
fn k_shift_value_preserving_large_drop() {
    value_preservation_at(None, 160, 16, 63, 120);
}

fn value_preservation_at(
    swa: Option<(u32, Vec<bool>)>,
    n_ctx: u32,
    n_batch: usize,
    d: usize,
    n: usize,
) {
    let toks: Vec<i32> = (0..n + 8).map(|i| ((i * 37 + 5) % 97) as i32).collect();

    // run A: prefill, shift by -d, continue
    let mut a = toy_llama(4242, n_ctx, n_batch, swa.clone());
    decode_chunked(
        &mut a,
        &toks[..n],
        &(0..n as i32).collect::<Vec<_>>(),
        n_batch,
    );
    a.seq_add(0, -1, -1, -(d as i32)).expect("seq_add");
    // the pending shift is exactly what the next decode's memory_update
    // consumes (llama-context.cpp:1805 → llama-kv-cache.cpp:746)
    assert!(a.kv.get_has_shift(), "seq_add marks the cache shifted");
    // the survivors hold the shifted positions
    let pos: Vec<i32> = a.kv.cells.iter().map(|c| c.pos).collect();
    for i in d..n {
        assert_eq!(pos[i], (i - d) as i32, "cell {i} shifted by -{d}");
    }

    let mut logits_a = Vec::new();
    for k in 0..8 {
        let p = (n - d + k) as i32;
        let l = a
            .decode(&[toks[n + k]], &[p])
            .expect("decode after shift")
            .to_vec();
        if k == 0 {
            assert!(
                !a.kv.get_has_shift(),
                "the decode ran memory_update and reset the accumulators"
            );
        }
        logits_a.push(l);
    }

    // run B: the same logical sequence decoded fresh
    let mut b = toy_llama(4242, n_ctx, n_batch, swa);
    decode_chunked(
        &mut b,
        &toks[d..n],
        &(0..(n - d) as i32).collect::<Vec<_>>(),
        n_batch,
    );
    let mut logits_b = Vec::new();
    for k in 0..8 {
        let p = (n - d + k) as i32;
        let l = b.decode(&[toks[n + k]], &[p]).expect("decode fresh");
        logits_b.push(l.to_vec());
    }

    let mut worst = 0f32;
    for k in 0..8 {
        let (x, y) = (&logits_a[k], &logits_b[k]);
        let delta = max_abs(x, y);
        worst = worst.max(delta);
        assert_eq!(
            argmax(x),
            argmax(y),
            "step {k}: the shifted continuation must pick the same token"
        );
    }
    // the re-rotated F16 K rows differ from fresh ones by one extra F16
    // rounding; the logits band stays well below a token-flipping scale
    assert!(worst < 0.05, "step logits band {worst} exceeds 0.05");
    println!("k-shift value-preserving: worst |Δlogits| over 8 steps = {worst:.5}");
}

#[test]
fn k_shift_value_preserving_unified() {
    value_preservation(None);
}

#[test]
fn k_shift_value_preserving_iswa() {
    // one SWA layer (window 6) + one dense layer: each cache has its own
    // shift vector and its own rotated layers (llama-kv-cache.cpp:866-883 per
    // cache; the uniform shift preserves every query-key distance, so the
    // STANDARD window mask is invariant)
    value_preservation(Some((6, vec![true, false])));
}

// ---------------------------------------------------------------------------
// 2. long-context generation on the real model (manual, release)
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
}

fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let file = std::fs::File::open(path).expect("open model");
    let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    match load_model(&gguf, mmap.clone()) {
        Ok(model) => Some(Loaded { model, gguf, mmap }),
        Err(e) => {
            eprintln!("SKIP: load_model({path}) failed: {e}");
            None
        }
    }
}

/// `AttnParams` from loaded hparams, mirroring llama-cli / DecodeContext wiring.
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

/// A ~48-token prompt: enough head-room below `-c 128` that the first shift
/// fires mid-generation, not during the prompt.
const LONG_PROMPT: &str =
    "The history of Paris begins with the Gallic settlement of the Parisii on the \
island now known as the Ile de la Cite, around 250 BC. The Romans conquered the area in 52 BC and \
founded the city of Lutetia on the left bank of the Seine. The city flourished and";

/// Greedy generation with the server's context-shift math
/// (server-context.cpp:2909-2972 / the port's `Engine::context_shift`)
/// applied whenever the slot is full: n_keep = params.n_keep (<0 = all) +
/// add_bos, capped at n_ctx - 4; n_discard = params.n_discard or n_left/2,
/// clamped to [0, n_left-1]; seq_rm + seq_add(-n_discard); splice the token
/// list. Returns (generated ids, per-step logits rows, the (step, n_keep,
/// n_discard) shift events, the post-splice prompt of every event — the
/// logical sequence the run continues from).
fn run_with_shift(
    n_ctx: u32,
    n_predict: usize,
) -> Option<(
    Vec<i32>,
    Vec<Vec<f32>>,
    Vec<(usize, i32, i32)>,
    Vec<Vec<i32>>,
)> {
    let mut l = load_real(QWEN25)?;
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let prompt: Vec<i32> = vocab.tokenize(LONG_PROMPT, true, true);
    let weights = qwen2_weights(&l.model);
    let attn = attn_params(&l.model);
    // the weights' tensors live in the model's Context (arch_e2e's wiring)
    let mut dctx = DecodeContext::new(l.model.ctx, weights, attn, n_ctx, 8, 512);
    let add_bos = vocab.add_bos;

    let mut ids = Vec::new();
    let mut rows = Vec::new();
    let mut events: Vec<(usize, i32, i32)> = Vec::new();
    let mut snapshots: Vec<Vec<i32>> = Vec::new();
    // prefill (chunked by the driver's n_batch = 512 — one call)
    let mut logits = decode_chunked(
        &mut dctx,
        &prompt,
        &(0..prompt.len() as i32).collect::<Vec<_>>(),
        512,
    )
    .to_vec();
    let mut prompt = prompt;
    for step in 0..n_predict {
        let tok = argmax(&logits);
        // pre_decode's shift check (server-context.cpp:2913) — before the
        // sampled token joins the batch
        if prompt.len() as i64 + 1 >= n_ctx as i64 {
            // Engine::context_shift with n_keep = 0, n_discard = 0 (half)
            let mut n_keep = 0i32;
            if add_bos {
                n_keep += 1;
            }
            n_keep = n_keep.min(n_ctx as i32 - 4);
            let n_left = prompt.len() as i32 - n_keep;
            let n_discard = (n_left / 2).clamp(0, (n_left - 1).max(0));
            dctx.seq_rm(0, n_keep, n_keep + n_discard);
            dctx.seq_add(0, n_keep + n_discard, prompt.len() as i32, -n_discard)
                .expect("seq_add");
            for i in (n_keep + n_discard) as usize..prompt.len() {
                prompt[i - n_discard as usize] = prompt[i];
            }
            prompt.truncate(prompt.len() - n_discard as usize);
            events.push((step, n_keep, n_discard));
            snapshots.push(prompt.clone());
        }
        let pos = prompt.len() as i32;
        rows.push(logits.clone());
        logits = dctx.decode(&[tok], &[pos]).expect("decode").to_vec();
        prompt.push(tok);
        ids.push(tok);
    }
    rows.push(logits.clone());
    Some((ids, rows, events, snapshots))
}

/// The long-context shift run on the real model. **Semantics pinned first:**
/// the reference's context shift is a *truncation heuristic* — `seq_rm` drops
/// `n_discard` cached tokens and `seq_add(-n_discard)` relabels the rest, but
/// the kept tokens' cached K/V rows still encode attention over the discarded
/// context (they were computed when it was present). So the shifted run does
/// NOT equal a fresh decode of the spliced sequence, and does NOT equal the
/// unshifted run — in the reference exactly as in the port (this is why the
/// server reports `truncated: true`). What IS exact is the K-shift rotation
/// itself (`qwen2_5_k_shift_rotation_identity` below) and the whole-stream
/// behaviour against the reference server with the same flags
/// (`parity/run_server_parity_shift.sh`).
///
/// Asserted here (all deterministic hand computations):
///   * the shift schedule: prompt 68 tokens + `-c 128` → the first shift at
///     prompt len 127 (step 59) with n_keep 0 / n_discard 63 (n_left/2), then
///     every 63 further tokens (steps 122, 185);
///   * the splice arithmetic: each event leaves a 64-token prompt;
///   * all 200 generated logits rows finite.
/// Reported (informational): the free-run agreement prefix vs the unshifted
/// `-c 512` stream, and the fresh-replay residual — the magnitude of the
/// truncation semantics, expected to be large for BOTH implementations.
///
///   cargo test --release -p llama --test ctx_shift_e2e -- --ignored --nocapture
#[test]
#[ignore = "manual: qwen2.5-0.5b, greedy long-context runs in release"]
fn qwen2_5_long_context_shift_run() {
    let n_predict = 200usize;
    let Some((shifted, rows_a, events, snapshots)) = run_with_shift(128, n_predict) else {
        return; // model not present
    };
    // the hand-computed schedule: prompt 68, first shift at len 127 (step 59),
    // then every 63 tokens — n_keep 0 (qwen2.5 adds no BOS), n_discard 63
    assert_eq!(
        events,
        vec![(59, 0, 63), (122, 0, 63), (185, 0, 63)],
        "the -c 128 shift schedule"
    );
    for s in &snapshots {
        assert_eq!(s.len(), 64, "each splice leaves a 64-token prompt");
    }
    assert_eq!(shifted.len(), n_predict);
    assert!(
        rows_a.iter().all(|r| r.iter().all(|v| v.is_finite())),
        "all logits rows finite through the shifts"
    );
    println!("qwen2.5-0.5b: -c 128 shift events (step, n_keep, n_discard): {events:?}");

    // informational: the free-run prefix vs the unshifted -c 512 stream
    if let Some((plain, ..)) = run_with_shift(512, n_predict) {
        let mut match_len = 0usize;
        for (i, (a, b)) in shifted.iter().zip(plain.iter()).enumerate() {
            if a != b {
                break;
            }
            match_len = i + 1;
        }
        println!(
            "qwen2.5-0.5b: greedy -c 128 (3 shifts) vs greedy -c 512: \
             {match_len}/{} tokens agree (the truncation semantics + fp band)",
            shifted.len()
        );
    }
}

/// The K-shift rotation identity on the real model: layer 0's cached K row is
/// `W_k · RMS(embd(t))` — **context-independent before RoPE** — so after
/// `seq_add(-63)` + the K-shift graph it must equal the fresh row of the same
/// token at the shifted position, to F16 rounding (one extra rounding of the
/// re-rotated row). Layers >= 1 legitimately differ (their pre-rotation K
/// mixed the discarded context — the truncation semantics); their deltas are
/// reported.
///
///   cargo test --release -p llama --test ctx_shift_e2e -- --ignored --nocapture \
///       qwen2_5_k_shift_rotation_identity
#[test]
#[ignore = "manual: qwen2.5-0.5b, two loaded contexts in release"]
fn qwen2_5_k_shift_rotation_identity() {
    let prompt: Vec<i32>;
    let mut a = {
        let Some(mut l) = load_real(QWEN25) else {
            return;
        };
        let vocab = Vocab::load(&l.gguf).expect("vocab");
        prompt = vocab.tokenize(LONG_PROMPT, true, true);
        let weights = qwen2_weights(&l.model);
        let attn = attn_params(&l.model);
        DecodeContext::new(l.model.ctx, weights, attn, 128, 8, 512)
    };
    // 68 prompt + 59 teacher-forced tokens = 127, positions 0..126
    let gen: Vec<i32> = (0..59).map(|i| 1000 + (i * 13) % 900).collect();
    let mut all = prompt.clone();
    decode_chunked(&mut a, &all, &(0..68).collect::<Vec<_>>(), 512);
    for (i, &t) in gen.iter().enumerate() {
        let _ = a.decode(&[t], &[68 + i as i32]).expect("decode");
        all.push(t);
    }
    assert_eq!(all.len(), 127);

    // the server's shift: n_keep 0, n_discard 63 (the -c 128 schedule)
    a.seq_rm(0, 0, 63);
    a.seq_add(0, 63, 127, -63).expect("seq_add");
    let spliced: Vec<i32> = all[63..].to_vec();
    assert_eq!(spliced.len(), 64);
    // one more decode runs memory_update → the K-shift graph
    let logits = a.decode(&[4242], &[64]).expect("decode");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "post-shift logits finite"
    );

    // fresh: prefill the spliced sequence and decode the same token
    let mut b = {
        let Some(mut l) = load_real(QWEN25) else {
            return;
        };
        let weights = qwen2_weights(&l.model);
        let attn = attn_params(&l.model);
        DecodeContext::new(l.model.ctx, weights, attn, 512, 8, 512)
    };
    decode_chunked(&mut b, &spliced, &(0..64).collect::<Vec<_>>(), 512);
    let _ = b.decode(&[4242], &[64]).expect("decode");

    // a's kept tokens live in cells 63..126 (pos 0..63), b's in cells 0..63
    let row = |ctx: &Context, t: TensorId, cell: usize| -> Vec<f32> {
        let bytes = ctx.data_bytes(t).unwrap();
        let n_embd = 128usize; // n_head_kv 2 * head 64 F16 elements
        bytes[cell * n_embd * 2..(cell + 1) * n_embd * 2]
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    };
    let n_layer = a.kv.layers.len();
    for il in 0..n_layer {
        let (k_a, k_b) = (a.kv.layers[il].k, b.kv.layers[il].k);
        let worst = (63..127usize)
            .zip(0..64usize)
            .map(|(ci, cb)| max_abs(&row(&a.gctx, k_a, ci), &row(&b.gctx, k_b, cb)))
            .fold(0f32, f32::max);
        if il == 0 {
            // the rotation identity: F16 rows of magnitude up to ~10, one
            // extra rounding → ≤ 2 F16 ulp at that scale (measured 0.125)
            assert!(
                worst < 0.2,
                "layer 0's re-rotated K rows must equal the fresh ones to F16 \
                 rounding (got {worst})"
            );
            println!("layer 0 K rows: worst |Δ| {worst:.5} (the F16 rounding)");
        } else if il == 1 || il == n_layer - 1 {
            println!(
                "layer {il} K rows: worst |Δ| {worst:.5} (the truncation semantics — \
                 the pre-rotation K mixed the discarded context)"
            );
        }
    }
}
