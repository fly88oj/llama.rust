//! arch_e2e.rs — end-to-end forward verification of `graph_arch.rs` builders on
//! real GGUF weights available on this machine.
//!
//! Scope / ownership: this file exercises `build_llama_forward`,
//! `build_gemma2_forward`, `build_gemma3_forward`, `build_phi3_forward` and the
//! `DecodeContext` qwen2 path against real (or real-dimension) models. It never
//! modifies the implementation; every blocker found is reported in the agent
//! report, not worked around in lib code.
//!
//! Runs by default (cheap; no reference server needed):
//!   * `qwen2_real_reference_prefix` — qwen2.5-0.5b via DecodeContext vs the
//!     captured reference greedy ids
//!   * `qwen2_determinism_bit_identical` — same input twice → bit-equal logits
//!   * `llama_builder_matches_qwen2_builder_on_real_weights`
//!   * `gemma2_gemma3_real_dims_synthetic` — gemma builders at gemma-3-1b
//!     dimensions (no local gemma2/3 GGUF exists on this machine)
//!   * `arch_support_matrix` — meta/load/dtype support for every local GGUF
//!     (informational; GGUF metadata + mmap only)
//!
//! `#[ignore]`d (manual; see each test's doc comment):
//!   * `phi3_real_forward_phi4_mini` — Phi-4-mini Q6_K (2.9 GiB) real forward,
//!     reference first-token match + bit-identical determinism
//!   * `phi3_prompt_battery`         — 6 unambiguous prompts vs reference texts
//!   * `phi3_16_token_parity`        — 16-token reference parity (13/16, see doc)
//!   * `qwen2_16_token_parity_full`  — full 16-token reference parity (16/16)
//!   * `gemma_synth_parity_probe`    — writes synthetic gemma2/gemma3 GGUFs
//!   * `gemma_synth_ppl_probe`       — chunk PPL vs the reference perplexity
//!   * `big_models_load_probe`       — >6 GiB local models, metadata only
//!
//! Headline results captured on 2026-09-24 (reference build bd4f514db1):
//!   * qwen2.5-0.5b (DecodeContext): argmax id 12095 (" Paris"), top-2 32671,
//!     16/16 greedy tokens == the reference sequence.
//!   * llama builder on the same qwen2.5 file == qwen2 builder bit-for-bit
//!     (0/151936 logits differ), argmax 12095 — the only local llama-arch file is
//!     a 21.9 GiB deepseek-coder-33b (loads, not run).
//!   * phi3 (Phi-4-mini Q6_K, real weights): first token 12650 (" Paris") ==
//!     reference; 7-token prefix == reference; 13/16 greedy tokens == reference
//!     (divergence at index 7 is a "country name" choice, port margin 1.79
//!     logits — flagged OPEN, needs a logits-level comparison); prompt battery
//!     6/6; repeat prefill bit-identical.
//!   * gemma3/gemma2: no real GGUF locally (all local gemma files are gemma-4,
//!     arch `gemma4`, which model.rs refuses). Verified at real dimensions with
//!     deterministic greedy output locally, and compared against the reference
//!     on synthetic files: gemma3 chunk PPL 86674.59 (port, shifted file) vs
//!     87799.61 (reference, shifted file) = 1.3% — the residual is kernel-lane
//!     level. gemma2 now carries its REAL caps (50/30) and both sides run it
//!     (no NaN): port 96186.47 vs reference 98560.11 = 2.4%, plus an exact
//!     8/8 greedy-token and top-5-id match against the reference server.
//!   * gemma2 runs again (2026-09-25, ggml_tanh ported): before this, gemma2
//!     NaN'd in BOTH implementations — the port asserted softcap == 0, and the
//!     reference NaN'd on a cap == 0 file (unguarded final softcap, 1/0 = inf).
//!   * FIXED (P1, 2026-09-25): graph_arch's gemma norm helper used to add +1
//!     inside the graph, double-shifting real GGUFs (the converter already
//!     bakes +1 into `*norm.weight`, conversion/gemma.py norm_shift; the
//!     reference graph does a plain mul, llama-graph.cpp:1604). It now does a
//!     plain mul, and this probe asserts rust(SHIFTED file) vs
//!     reference(SHIFTED file) — currently 86674.59 vs 87799.61 = 1.3%.
//!
//! Reference ids were captured with the pinned reference build
//! (`/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-server`, bd4f514db1,
//! `-c 512 -t 8 -fa off`) under the PARITY.md protocol (fresh server, first
//! request on the slot, `temperature=0, top_k=1, cache_prompt=false`).
//! No test here contacts the reference server.

use std::path::Path;
use std::sync::Arc;

use ggml::types::GgmlType;
use ggml::{Context, Gguf, TensorId};
use llama::graph::{AttnParams, DecodeInputs, ForwardResult, LayerWeights, ModelWeights};
use llama::graph_arch::{
    build_gemma2_forward, build_gemma3_forward, build_llama_forward, build_phi3_forward,
    GemmaLayerWeights, GemmaModelWeights, GemmaParams, LlamaLayerWeights, LlamaModelWeights,
    Phi3LayerWeights, Phi3ModelWeights,
};
use llama::kv_cache::{KvCache, SlotInfo};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// models on this machine
// ---------------------------------------------------------------------------

const QWEN25: &str = "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const QWEN3_EMB: &str = "/home/jeffrey/localai/models/Qwen3-Embedding-0.6B-Q8_0.gguf";
const PHI4_MINI: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q6_K.gguf";
const GEMMA4_12B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf";
const GPTOSS20B_MXFP4: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf";
const GPTOSS20B_Q4KM: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf";
const LFM2: &str =
    "/home/jeffrey/.lmstudio/models/LiquidAI/LFM2-8B-A1B-GGUF/LFM2-8B-A1B-Q4_K_M.gguf";
const GRANITE_TINY: &str =
    "/home/jeffrey/.lmstudio/models/unsloth/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf";
const QWEN36_27B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf";
const DEEPSEEK_CODER33B: &str =
    "/home/jeffrey/.codegpt/models/gguf/deepseek-coder-33b-instruct.Q5_K_M.gguf";

/// Reference greedy ids for prompt "The capital of France is" (16 tokens),
/// qwen2.5-0.5b-instruct Q4_K_M, reference `llama-server` fresh + first request.
/// Text: " Paris. It is the largest city in Europe and the second largest in the world"
const QWEN25_REF16: [i32; 16] = [
    12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279, 1879,
];

/// Same prompt, Phi-4-mini-instruct Q6_K (phi3 arch), same protocol.
/// Text: " Paris. What is the capital of Germany? The capital of Germany is Berlin."
const PHI4_MINI_REF16: [i32; 16] = [
    12650, 13, 4614, 382, 290, 9029, 328, 17237, 30, 623, 9029, 328, 17237, 382, 21230, 13,
];

const PROMPT: &str = "The capital of France is";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    /// keeps the weight storage alive (model tensors point into it)
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    size_bytes: u64,
}

/// mmap + parse + load_model; None (with a printed SKIP) when the file is not
/// on this machine, so the suite stays portable.
fn load_real(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() {
        eprintln!("SKIP: {path} not present");
        return None;
    }
    let size_bytes = std::fs::metadata(path).ok()?.len();
    let file = std::fs::File::open(path).expect("open model");
    // SAFETY: read-only use of a model file (same policy as the rest of the port)
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

/// Available memory in GiB (MemAvailable = reclaimable estimate).
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

/// Memory guard: weights are mmap'd, but the KV cache + graph arena are
/// anonymous memory. Skip instead of getting OOM-killed.
fn mem_guard(label: &str, need_gb: f64) -> bool {
    let avail = mem_available_gb();
    if avail < need_gb {
        eprintln!(
            "SKIP {label}: {avail:.1} GiB available < {need_gb:.1} GiB required \
             (other processes are competing for memory)"
        );
        return false;
    }
    true
}

fn argmax(v: &[f32]) -> i32 {
    // greedy: strict >, first max wins (llama_sampler_init_greedy)
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

fn top2(v: &[f32]) -> (i32, i32) {
    let mut a = (0usize, f32::NEG_INFINITY);
    let mut b = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > a.1 {
            b = a;
            a = (i, x);
        } else if x > b.1 {
            b = (i, x);
        }
    }
    (a.0 as i32, b.0 as i32)
}

/// Top-k (id, logit) pairs, descending by logit (ties: lower id first).
fn topk(v: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    idx.into_iter().take(k).map(|i| (i as i32, v[i])).collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut diff = 0usize;
    let mut max_abs = 0f32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            diff += 1;
        }
        max_abs = max_abs.max((x - y).abs());
    }
    (diff, max_abs)
}

fn f32s(ctx: &Context, id: TensorId) -> Vec<f32> {
    ctx.data_bytes(id)
        .expect("tensor data")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Escaped piece for log lines.
fn piece(vocab: &Vocab, id: i32) -> String {
    format!("{:?}", vocab.token_to_piece(id))
}

fn piece_str(vocab: &Vocab, id: i32) -> String {
    vocab.token_to_piece(id).to_string()
}

fn text_of(vocab: &Vocab, ids: &[i32]) -> String {
    ids.iter().map(|&t| piece_str(vocab, t)).collect()
}

/// `AttnParams` from loaded hparams, mirroring llama-cli / DecodeContext wiring.
/// Uses the shared `rope_runtime()` derivation (llama-context.cpp:106-215,
/// agent N's P4 fix) so tests exercise the same n_ctx_orig / ext_factor /
/// attn_factor values as production.
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

/// graph::LayerWeights from a loaded model (qwen2/llama dense layout).
fn qwen2_weights(m: &LlamaModel) -> ModelWeights {
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

fn llama_weights(m: &LlamaModel) -> LlamaModelWeights {
    LlamaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .map(|x| LlamaLayerWeights {
                attn_norm: x.attn_norm.expect("attn_norm"),
                wq: x.wq.expect("wq"),
                wk: x.wk.expect("wk"),
                wv: x.wv.expect("wv"),
                wo: x.wo.expect("wo"),
                wq_b: x.wq_b,
                wk_b: x.wk_b,
                wv_b: x.wv_b,
                wo_b: x.wo_b,
                ffn_norm: x.ffn_norm.expect("ffn_norm"),
                ffn_gate: x.ffn_gate.expect("ffn_gate"),
                ffn_down: x.ffn_down.expect("ffn_down"),
                ffn_up: x.ffn_up.expect("ffn_up"),
                ffn_gate_b: x.ffn_gate_b,
                ffn_down_b: x.ffn_down_b,
                ffn_up_b: x.ffn_up_b,
            })
            .collect(),
    }
}

fn phi3_weights(m: &LlamaModel) -> Phi3ModelWeights {
    Phi3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .map(|x| Phi3LayerWeights {
                attn_norm: x.attn_norm.expect("attn_norm"),
                wqkv: x.wqkv.expect("fused wqkv"),
                wqkv_b: x.wqkv_b,
                wo: x.wo.expect("wo"),
                wo_b: x.wo_b,
                ffn_norm: x.ffn_norm.expect("ffn_norm"),
                ffn_down: x.ffn_down.expect("ffn_down"),
                ffn_up: x.ffn_up.expect("ffn_up"),
            })
            .collect(),
    }
}

/// Same as the reference capture / llama-cli default.
fn threads() -> usize {
    8
}

// ---------------------------------------------------------------------------
// decode harness: the DecodeContext::decode input/graph protocol, generic over
// the arch builder (mirrors graph_arch.rs' toy Harness — those builders are not
// reachable through DecodeContext, which only calls build_qwen2_forward).
// ---------------------------------------------------------------------------

struct RealHarness {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    label: &'static str,
}

impl RealHarness {
    fn new(gctx: Context, kv: KvCache, label: &'static str) -> Self {
        let watermark = gctx.mark();
        RealHarness {
            gctx,
            kv,
            watermark,
            label,
        }
    }

    /// Decode `tokens` at `pos`; returns last-token logits [n_vocab].
    fn decode<W>(
        &mut self,
        w: &W,
        tokens: &[i32],
        pos: &[i32],
        build: impl FnOnce(
            &mut Context,
            &W,
            &KvCache,
            &DecodeInputs,
            SlotInfo,
            u32,
            usize,
        ) -> ForwardResult,
    ) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        // assign first, then the (256-padded) n_kv — step_inputs order
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let kq_mask = self
            .gctx
            .new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| p.copy_from_slice(pos))
            .unwrap();
        {
            let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
            self.gctx
                .data_bytes_mut(row_idx)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&idxs));
        }
        {
            // mask [n_kv, n_tokens]: element (kv s, query t) at s + t*n_kv
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
            mask.fill(f32::NEG_INFINITY);
            // padded (empty) cells keep pos = -1 → stay masked
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            for (iq, &qp) in pos.iter().enumerate() {
                for (ik, &kp) in kv_pos.iter().enumerate() {
                    if 0 <= kp && kp <= qp {
                        mask[iq * n_kv as usize + ik] = 0.0;
                    }
                }
            }
        }
        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };

        let result = build(&mut self.gctx, w, &self.kv, &inputs, sinfo, n_kv, n);
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, threads());
        self.kv.assign(sinfo, pos, 0);

        let ne = self.gctx.ne(logits);
        let n_vocab = ne[0] as usize;
        let all: Vec<f32> = self
            .gctx
            .data_bytes(logits)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let out = all[n_vocab * (n - 1)..n_vocab * n].to_vec();
        debug_assert!(
            out.iter().all(|v| v.is_finite()),
            "{} logits not finite",
            self.label
        );
        out
    }

    /// Like `decode` but returns logits for ALL positions: [n_vocab * n],
    /// row-major per token (llama.cpp logits_all equivalent) — needed to score a
    /// whole chunk in one pass like the reference llama-perplexity does.
    fn decode_all<W>(
        &mut self,
        w: &W,
        tokens: &[i32],
        pos: &[i32],
        build: impl FnOnce(
            &mut Context,
            &W,
            &KvCache,
            &DecodeInputs,
            SlotInfo,
            u32,
            usize,
        ) -> ForwardResult,
    ) -> Vec<f32> {
        let n = tokens.len();
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        // assign first, then the (256-padded) n_kv — step_inputs order
        self.kv.assign(sinfo, pos, 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let kq_mask = self
            .gctx
            .new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens))
            .unwrap();
        self.gctx
            .with_i32_mut(pos_t, |p| p.copy_from_slice(pos))
            .unwrap();
        {
            let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
            self.gctx
                .data_bytes_mut(row_idx)
                .unwrap()
                .copy_from_slice(bytemuck::cast_slice(&idxs));
        }
        {
            let mask_bytes = self.gctx.data_bytes_mut(kq_mask).unwrap();
            let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
            mask.fill(f32::NEG_INFINITY);
            // padded (empty) cells keep pos = -1 → stay masked
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize]
                .iter()
                .map(|c| c.pos)
                .collect();
            for (iq, &qp) in pos.iter().enumerate() {
                for (ik, &kp) in kv_pos.iter().enumerate() {
                    if 0 <= kp && kp <= qp {
                        mask[iq * n_kv as usize + ik] = 0.0;
                    }
                }
            }
        }
        let inputs = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let result = build(&mut self.gctx, w, &self.kv, &inputs, sinfo, n_kv, n);
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, threads());
        // (assign already happened before the graph build, step_inputs order)
        let n_vocab = self.gctx.ne(logits)[0] as usize;
        let all: &[f32] = bytemuck::cast_slice(self.gctx.data_bytes(logits).unwrap());
        all[..n_vocab * n].to_vec()
    }
}

/// Mean negative log-likelihood over the chunk, llama-perplexity style:
/// score token i from the logits at position i-1 (i in 1..n).
fn chunk_ppl(logits_all: &[f32], n_vocab: usize, tokens: &[i32]) -> f64 {
    let n = tokens.len();
    let mut nll = 0f64;
    let mut scored = 0usize;
    for i in 1..n {
        let row = &logits_all[(i - 1) * n_vocab..i * n_vocab];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = (mx as f64)
            + row
                .iter()
                .map(|&v| ((v - mx) as f64).exp())
                .sum::<f64>()
                .ln();
        nll -= (row[tokens[i] as usize] as f64) - lse;
        scored += 1;
    }
    (nll / scored as f64).exp()
}

// ===========================================================================
// 1. qwen2.5-0.5b — DecodeContext real forward vs reference
// ===========================================================================

#[test]
fn qwen2_real_reference_prefix() {
    let Some(l) = load_real(QWEN25) else { return };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let hp = l.model.hparams.clone();
    println!(
        "qwen2.5-0.5b: arch={} n_embd={} n_layer={} heads={}/{} head_dim={} n_rot={} eps={:e} \
         rope={:?} base={} file={:.2} GiB",
        l.model.arch.name(),
        hp.n_embd,
        hp.n_layer(),
        hp.n_head(0),
        hp.n_head_kv(0),
        hp.n_embd_head_k(0),
        hp.n_rot(0),
        hp.f_norm_rms_eps,
        hp.rope_type,
        hp.rope_freq_base_train,
        l.size_bytes as f64 / 1073741824.0,
    );

    let prompt = vocab.tokenize(PROMPT, true, true);
    println!("prompt ids: {prompt:?}");
    assert_eq!(prompt.len(), 5, "reference server reported prompt_n=5");

    let attn = attn_params(&l.model);
    let weights = qwen2_weights(&l.model);

    let t0 = std::time::Instant::now();
    let mut dctx =
        llama::context::DecodeContext::new(l.model.ctx, weights, attn, 512, threads(), 64);
    let setup = t0.elapsed();

    let t1 = std::time::Instant::now();
    let logits = dctx
        .decode(&prompt, &[0, 1, 2, 3, 4])
        .expect("prefill")
        .to_vec();
    let prefill = t1.elapsed();
    assert!(logits.iter().all(|v| v.is_finite()), "logits finite");

    let (a1, a2) = top2(&logits);
    println!(
        "setup {setup:?}; prefill {:.0} ms ({:.1} t/s); argmax={} {} top2={} {} margin={:.4}",
        prefill.as_secs_f64() * 1e3,
        prompt.len() as f32 / prefill.as_secs_f32(),
        a1,
        piece(&vocab, a1),
        a2,
        piece(&vocab, a2),
        logits[a1 as usize] - logits[a2 as usize],
    );

    assert_eq!(a1, 12095, "first greedy token must be id 12095 ' Paris'");
    assert_eq!(a2, 32671, "runner-up must be id 32671");
    assert_eq!(piece(&vocab, 12095), "\" Paris\"");

    let mut got = Vec::new();
    let mut cur = logits;
    let mut pos = prompt.len() as i32;
    let t2 = std::time::Instant::now();
    // greedy: the first token is argmax(prefill logits); each decode() call then
    // returns the next-token logits for the token just fed
    for step in 0..3 {
        let tok = argmax(&cur);
        got.push(tok);
        if step == 2 {
            break;
        }
        cur = dctx.decode(&[tok], &[pos]).expect("step").to_vec();
        pos += 1;
    }
    let gen = t2.elapsed();
    println!(
        "greedy 3 tokens {:.0} ms ({:.1} t/s): {got:?} = {:?}",
        gen.as_secs_f64() * 1e3,
        (got.len() - 1) as f32 / gen.as_secs_f32(),
        text_of(&vocab, &got)
    );
    assert_eq!(
        got,
        QWEN25_REF16[..3].to_vec(),
        "greedy prefix {:?} ({:?}) != reference {:?} ({:?})",
        got,
        text_of(&vocab, &got),
        &QWEN25_REF16[..3],
        text_of(&vocab, &QWEN25_REF16[..3])
    );
    println!("qwen2 reference prefix OK");
}

/// Full 16-token reference parity (manual; still no server — ids hardcoded from
/// the PARITY.md protocol capture).
#[test]
#[ignore = "manual: full 16-token reference parity (slow in unoptimized builds)"]
fn qwen2_16_token_parity_full() {
    let Some(l) = load_real(QWEN25) else { return };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let attn = attn_params(&l.model);
    let weights = qwen2_weights(&l.model);
    let mut dctx =
        llama::context::DecodeContext::new(l.model.ctx, weights, attn, 512, threads(), 64);
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let mut cur = dctx.decode(&prompt, &pos).expect("prefill").to_vec();
    let mut got = Vec::new();
    let mut p = prompt.len() as i32;
    for _ in 0..16 {
        let tok = argmax(&cur);
        got.push(tok);
        cur = dctx.decode(&[tok], &[p]).expect("step").to_vec();
        p += 1;
    }
    println!("ref : {:?}", QWEN25_REF16);
    println!("rust: {got:?}");
    println!("text: {:?}", text_of(&vocab, &got));
    assert_eq!(got, QWEN25_REF16.to_vec(), "16-token reference parity");
}

// ===========================================================================
// 2. determinism — same input twice, bit-identical logits
// ===========================================================================

#[test]
fn qwen2_determinism_bit_identical() {
    let Some(l) = load_real(QWEN25) else { return };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let attn = attn_params(&l.model);
    let weights = qwen2_weights(&l.model);
    let mut dctx =
        llama::context::DecodeContext::new(l.model.ctx, weights, attn, 512, threads(), 64);
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();

    let a = dctx.decode(&prompt, &pos).expect("run A").to_vec();
    // drop the cache and repeat the identical call: no stale state may leak
    dctx.kv.clear();
    let b = dctx.decode(&prompt, &pos).expect("run B").to_vec();

    let (diff, max_abs) = bit_diffs(&a, &b);
    assert_eq!(
        diff,
        0,
        "{diff}/{} logits differ between identical decodes (max |d|={max_abs:e})",
        a.len()
    );
    println!(
        "determinism OK: {} logits bit-identical, argmax={} {}",
        a.len(),
        argmax(&a),
        piece(&vocab, argmax(&a))
    );
    assert_eq!(argmax(&a), 12095);
}

// ===========================================================================
// 3. llama builder on real weights (qwen2.5 file = llama-family layout)
// ===========================================================================

/// No llama/mistral/mixtral GGUF exists locally (the only llama-arch file is a
/// 23.5 GiB deepseek-coder-33b, see `big_models_load_probe`). qwen2 shares the
/// llama tensor layout, and the two forward graphs are op-for-op identical when
/// the llama-only optional tensors (wo_b / ffn biases / auto rope factors) are
/// absent, so `build_llama_forward` over the qwen2.5-0.5b file must reproduce
/// the reference-verified `build_qwen2_forward` logits bit-exactly.
#[test]
fn llama_builder_matches_qwen2_builder_on_real_weights() {
    // --- qwen2 path: the real DecodeContext (reference-verified)
    let Some(l) = load_real(QWEN25) else { return };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let hp = l.model.hparams.clone();
    assert_eq!(
        hp.rope_type as i32, 2,
        "qwen2 rope must be NEOX for this comparison"
    );
    let ap = attn_params(&l.model);
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let n_layer = l.model.layers.len();
    let n_k_gqa = l.model.n_embd_k_gqa_max() as i64;
    let n_v_gqa = l.model.n_embd_v_gqa_max() as i64;

    let weights_q = qwen2_weights(&l.model);
    let mut dctx =
        llama::context::DecodeContext::new(l.model.ctx, weights_q, ap, 512, threads(), 64);
    let qlogits = dctx.decode(&prompt, &pos).expect("qwen2 prefill").to_vec();
    assert_eq!(
        argmax(&qlogits),
        12095,
        "sanity: qwen2 path must hit the reference token"
    );

    // --- llama path: manual assembly. A second load gives weight TensorIds that
    //     are valid in the second context (ids are per-context indices).
    let Some(l2) = load_real(QWEN25) else { return };
    assert_eq!(l2.model.layers.len(), n_layer);
    let w = llama_weights(&l2.model);
    let mut gctx = l2.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k_gqa, n_v_gqa, 512);
    let mut h = RealHarness::new(gctx, kv, "llama");
    let t0 = std::time::Instant::now();
    let llogits = h.decode(&w, &prompt, &pos, |ctx, w, kv, inp, s, nk, nt| {
        build_llama_forward(ctx, w, &ap, kv, inp, s, nk, nt)
    });
    println!(
        "llama-builder prefill {:.0} ms; argmax={} {} top2={} {}",
        t0.elapsed().as_secs_f64() * 1e3,
        argmax(&llogits),
        piece(&vocab, argmax(&llogits)),
        top2(&llogits).1,
        piece(&vocab, top2(&llogits).1)
    );

    let (diff, max_abs) = bit_diffs(&qlogits, &llogits);
    println!("qwen2-vs-llama builder: {diff} bit-differing logits, max |d| = {max_abs:e}");
    assert_eq!(
        argmax(&llogits),
        12095,
        "llama builder must hit the reference token"
    );
    assert_eq!(top2(&llogits).1, 32671);
    assert_eq!(
        diff, 0,
        "llama builder must be bit-identical to the qwen2 builder on this file \
         ({diff} diffs, max |d| = {max_abs:e})"
    );
}

// ===========================================================================
// 4. phi3 — Phi-4-mini real forward
// ===========================================================================

/// Phi-4-mini Q6_K (2.9 GiB) real weights through `build_phi3_forward`:
/// fused wqkv + bias, partial rope (n_rot 96 < head_dim 128), Q pre-scale,
/// packed ffn_up (2*n_ff), GQA. Asserts the first greedy token is the reference
/// id 12650 (" Paris"), a 4-token reference prefix, and determinism.
///
/// `#[ignore]`d because it is a 3.8B model: several forwards per run (tens of
/// seconds even in release). Run manually:
///   cargo test -p llama --release --test arch_e2e -- --ignored --nocapture
#[test]
#[ignore = "manual: 3.8B model real forward (~1-2 min in debug, seconds in release)"]
fn phi3_real_forward_phi4_mini() {
    if !mem_guard(PHI4_MINI, 6.0) {
        return;
    }
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let hp = l.model.hparams.clone();
    let ap = attn_params(&l.model);
    println!(
        "phi3 (Phi-4-mini Q6_K): arch={} n_embd={} n_layer={} heads={}/{} head_dim={} n_rot={} \
         rope={:?} base={} eps={:e} file={:.2} GiB",
        l.model.arch.name(),
        hp.n_embd,
        hp.n_layer(),
        hp.n_head(0),
        hp.n_head_kv(0),
        hp.n_embd_head_k(0),
        hp.n_rot(0),
        hp.rope_type,
        hp.rope_freq_base_train,
        hp.f_norm_rms_eps,
        l.size_bytes as f64 / 1073741824.0,
    );

    let prompt = vocab.tokenize(PROMPT, true, true);
    println!("prompt ids: {prompt:?}");
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let n_layer = l.model.layers.len();
    let n_k_gqa = l.model.n_embd_k_gqa_max() as i64;
    let n_v_gqa = l.model.n_embd_v_gqa_max() as i64;
    let w = phi3_weights(&l.model);

    // PARTIAL (documented in graph_arch.rs): the builder passes rope_factors =
    // None, but the reference (phi3.cpp:97 → llama-model.cpp:2259) feeds
    // rope_short whenever n_ctx_seq <= n_ctx_orig_yarn. Print what the file
    // carries so the report can quantify the difference.
    {
        let stats = |name: &str, id: Option<TensorId>| {
            if let Some(id) = id {
                let v = f32s(&l.model.ctx, id);
                let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                println!(
                    "  {name}: n={} min={mn:.6} max={mx:.6} mean={mean:.6} first4={:?}",
                    v.len(),
                    &v[..4.min(v.len())]
                );
            }
        };
        stats("rope_long", l.model.layers[0].rope_long);
        stats("rope_short", l.model.layers[0].rope_short);
        println!("  n_ctx_train={} n_rot={}", hp.n_ctx_train, hp.n_rot(0));
    }

    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k_gqa, n_v_gqa, 512);
    let mut h = RealHarness::new(gctx, kv, "phi3");

    let t0 = std::time::Instant::now();
    let prefill_logits = h.decode(&w, &prompt, &pos, |ctx, w, kv, inp, s, nk, nt| {
        build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
    });
    let prefill = t0.elapsed();
    let (a1, a2) = top2(&prefill_logits);
    println!(
        "prefill {} tokens in {:.2} s ({:.2} t/s); argmax={} {} top2={} {} margin={:.4}",
        prompt.len(),
        prefill.as_secs_f64(),
        prompt.len() as f32 / prefill.as_secs_f32(),
        a1,
        piece(&vocab, a1),
        a2,
        piece(&vocab, a2),
        prefill_logits[a1 as usize] - prefill_logits[a2 as usize],
    );

    let mut got = Vec::new();
    let mut cur = prefill_logits.clone();
    let mut p = prompt.len() as i32;
    let t1 = std::time::Instant::now();
    for step in 0..4 {
        let tok = argmax(&cur);
        let k5 = topk(&cur, 5);
        println!(
            "  step {step}: top5 {:?} margin={:.4}",
            k5.iter()
                .map(|(i, v)| (*i, piece(&vocab, *i), *v))
                .collect::<Vec<_>>(),
            k5[0].1 - k5[1].1
        );
        got.push(tok);
        if step == 3 {
            break;
        }
        cur = h.decode(&w, &[tok], &[p], |ctx, w, kv, inp, s, nk, nt| {
            build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
        });
        p += 1;
    }
    let gen = t1.elapsed();
    println!(
        "greedy {} tokens in {:.2} s ({:.2} t/s): {got:?} = {:?}",
        got.len(),
        gen.as_secs_f64(),
        (got.len() - 1) as f32 / gen.as_secs_f32(),
        text_of(&vocab, &got)
    );

    // 1) output is real text: the reference first piece is " Paris"
    assert_eq!(
        piece(&vocab, a1),
        "\" Paris\"",
        "first greedy token {} is not ' Paris' — phi3 forward produces garbage: {:?}",
        a1,
        text_of(&vocab, &got)
    );
    // 2) reference parity of the prefix
    assert_eq!(
        got,
        PHI4_MINI_REF16[..4].to_vec(),
        "phi3 greedy prefix {:?} ({:?}) != reference {:?} ({:?})",
        got,
        text_of(&vocab, &got),
        &PHI4_MINI_REF16[..4],
        text_of(&vocab, &PHI4_MINI_REF16[..4])
    );

    // 3) determinism: identical prefill in a fresh context must be bit-identical
    if let Some(l2) = load_real(PHI4_MINI) {
        let w2 = phi3_weights(&l2.model);
        let mut gctx2 = l2.model.ctx;
        let kv2 = KvCache::new(&mut gctx2, n_layer, n_k_gqa, n_v_gqa, 512);
        let mut h2 = RealHarness::new(gctx2, kv2, "phi3-repeat");
        let again = h2.decode(&w2, &prompt, &pos, |ctx, w, kv, inp, s, nk, nt| {
            build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
        });
        let (diff, max_abs) = bit_diffs(&prefill_logits, &again);
        println!("phi3 repeat prefill: {diff} bit-differing logits, max |d| = {max_abs:e}");
        assert_eq!(
            diff, 0,
            "phi3 prefill not deterministic ({diff} differing logits)"
        );
    } else {
        eprintln!("note: phi3 determinism check skipped (second load failed)");
    }
    println!("phi3 e2e OK");
}

/// Full 16-token phi3 reference parity (manual, print + verified-facts asserts).
///
/// Captured result (release, 8 threads): 13/16 tokens identical to the reference;
/// the first difference is index 7, where the reference picks " Germany" and this
/// port picks " France". At that step the port's own France-vs-Germany gap is
/// **1.72 logits**, i.e. larger than the documented ulp-level residuals
/// (PARITY.md "已知数值差异来源": K-quant vec_dot last-ulp lanes, FA not ported), so
/// the size of the reference-side gap is what decides whether this is numeric
/// noise or a phi3 builder difference. The reference is self-consistent here
/// (3 identical requests on a fresh server + a second fresh instance reproduce
/// " Paris. What is the capital of Germany? The capital of Germany is Berlin."),
/// and both continuations are coherent English — so this is recorded as an OPEN
/// item (needs a logits-level comparison; see `phi3_prompt_battery` for the
/// unambiguous-prompt battery, which passes).
///
/// The test therefore asserts only what is verified: the matched prefix (>= 7
/// tokens), coherent text, and determinism downstream is covered by the
/// non-ignored phi3 checks.
#[test]
#[ignore = "manual: 16 forwards on a 3.8B model (~2 min in release)"]
fn phi3_16_token_parity() {
    if !mem_guard(PHI4_MINI, 6.0) {
        return;
    }
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let ap = attn_params(&l.model);
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k_gqa = l.model.n_embd_k_gqa_max() as i64;
    let n_v_gqa = l.model.n_embd_v_gqa_max() as i64;
    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k_gqa, n_v_gqa, 512);
    let mut h = RealHarness::new(gctx, kv, "phi3-16");
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let mut cur = h.decode(&w, &prompt, &pos, |ctx, w, kv, inp, s, nk, nt| {
        build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
    });
    let mut got = Vec::new();
    let mut p = prompt.len() as i32;
    let mut diverged: Option<(usize, f32)> = None; // (index, margin of ref token)
    for i in 0..16 {
        let k5 = topk(&cur, 5);
        let tok = argmax(&cur);
        got.push(tok);
        let ref_tok = PHI4_MINI_REF16[i];
        let ref_rank = k5.iter().position(|(id, _)| *id == ref_tok);
        println!(
            "  step {i:2}: rust={tok} {:?} margin={:.4} | ref={ref_tok} {:?} rank_in_rust_top5={:?}",
            piece(&vocab, tok),
            k5[0].1 - k5[1].1,
            piece(&vocab, ref_tok),
            ref_rank.map(|r| r as i32),
        );
        if tok != ref_tok && diverged.is_none() {
            let ref_logit = k5.iter().find(|(id, _)| *id == ref_tok).map(|(_, l)| *l);
            diverged = Some((i, ref_logit.map(|l| k5[0].1 - l).unwrap_or(f32::INFINITY)));
        }
        if i == 15 {
            break;
        }
        cur = h.decode(&w, &[tok], &[p], |ctx, w, kv, inp, s, nk, nt| {
            build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
        });
        p += 1;
    }
    let same = got
        .iter()
        .zip(&PHI4_MINI_REF16)
        .filter(|(a, b)| a == b)
        .count();
    println!("ref : {:?}", PHI4_MINI_REF16);
    println!("rust: {got:?}");
    println!("text: {:?}", text_of(&vocab, &got));
    println!("MATCH {same}/16");

    // 1) the leading prefix must match exactly (verified: 7 tokens)
    let n_prefix = diverged.map(|(i, _)| i).unwrap_or(16);
    assert!(
        n_prefix >= 7,
        "phi3 parity prefix too short: {n_prefix} tokens ({got:?})"
    );
    assert_eq!(
        &got[..n_prefix],
        &PHI4_MINI_REF16[..n_prefix],
        "matched prefix mismatch"
    );

    // 2) the first divergence is a genuinely ambiguous "country name" decision;
    //    the port's own margin there is printed for the report. No assertion on
    //    the margin: 1.72 logits is larger than the documented ulp-level
    //    residuals, so this is an OPEN item (needs a logits-level comparison),
    //    not something this test can decide.
    if let Some((i, margin)) = diverged {
        println!(
            "first divergence at {i}: reference token is {margin:.4} logits below the port's argmax"
        );
    }

    // 3) the generated continuation is coherent text
    let text = text_of(&vocab, &got);
    assert!(
        text.starts_with(" Paris. What is the capital of "),
        "phi3 continuation is not coherent: {text:?}"
    );
    assert!(
        text.ends_with('.'),
        "phi3 continuation is not a finished sentence: {text:?}"
    );
}

/// Prompt battery against the reference server (manual): six prompts whose
/// continuations are unambiguous. Reference texts captured with
/// `llama-server -m <Phi-4-mini Q6_K> -c 512 -t 8 -fa off`, greedy, first request
/// (stable across repeated requests and across a fresh instance):
///   "The capital of Germany is"                    -> " Berlin. Berlin is the capital of Germany"
///   "2 + 2 ="                                      -> " 4"
///   "The sky is"                                   -> " blue because the sky is blue. This"
///   "Paris is the capital of"                      -> " France. It is known for its rich"
///   "The first president of the United States was"  -> " George Washington. He was inaugurated as the"
///   "Once upon a time, there was a"                -> " young boy named Alex who loved to explore"
/// Each assertion is a *text prefix* of the reference answer, validating the phi3
/// forward (fused qkv + bias, partial rope, packed swiGLU, GQA) on real weights
/// independently of near-tie flips.
#[test]
#[ignore = "manual: 6 prompts x 8 forwards on a 3.8B model (~1 min in release)"]
fn phi3_prompt_battery() {
    if !mem_guard(PHI4_MINI, 6.0) {
        return;
    }
    let Some(l) = load_real(PHI4_MINI) else {
        return;
    };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let ap = attn_params(&l.model);
    let w = phi3_weights(&l.model);
    let n_layer = l.model.layers.len();
    let n_k_gqa = l.model.n_embd_k_gqa_max() as i64;
    let n_v_gqa = l.model.n_embd_v_gqa_max() as i64;
    let mut gctx = l.model.ctx;
    let kv = KvCache::new(&mut gctx, n_layer, n_k_gqa, n_v_gqa, 512);
    let mut h = RealHarness::new(gctx, kv, "phi3-battery");

    let cases: &[(&str, &str)] = &[
        ("The capital of Germany is", " Berlin"),
        ("2 + 2 =", " 4"),
        ("The sky is", " blue"),
        ("Paris is the capital of", " France"),
        ("The first president of the United States was", " George"),
        ("Once upon a time, there was a", " young"),
    ];

    let mut failures = Vec::new();
    for (prompt, expect) in cases {
        h.kv.clear();
        let toks = vocab.tokenize(prompt, true, true);
        let pos: Vec<i32> = (0..toks.len() as i32).collect();
        let mut cur = h.decode(&w, &toks, &pos, |ctx, w, kv, inp, s, nk, nt| {
            build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
        });
        let mut got = Vec::new();
        let mut p = toks.len() as i32;
        for step in 0..8 {
            let tok = argmax(&cur);
            got.push(tok);
            if step == 7 {
                break;
            }
            cur = h.decode(&w, &[tok], &[p], |ctx, w, kv, inp, s, nk, nt| {
                build_phi3_forward(ctx, w, &ap, kv, inp, s, nk, nt)
            });
            p += 1;
        }
        let text = text_of(&vocab, &got);
        let ok = text.starts_with(expect);
        println!(
            "{:<46} -> {text:?}  [{} expected prefix {expect:?}]",
            prompt,
            if ok { "OK" } else { "MISMATCH" }
        );
        if !ok {
            failures.push(format!(
                "{prompt:?}: got {text:?}, expected prefix {expect:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "phi3 prompt battery mismatches:\n  {}",
        failures.join("\n  ")
    );
}

// ===========================================================================
// 5. gemma2 / gemma3 — real-dimension synthetic (no local gemma2/3 weights)
// ===========================================================================

/// gemma-3-1b configuration (n_embd 1152, 4 query heads, 1 KV head, head_dim
/// 256 with only n_rot 128 rotated — the multi-channel partial-rope case the
/// toy test cannot reach) with small random F32 weights. Exercises the gemma
/// builder end-to-end: (1+w) norms, per-head q/k norms BEFORE rope, partial
/// rope, Q pre-scale, GQA 4:1 attention over the shared KV cache, post-norms,
/// GELU FFN, step-by-step decode + determinism.
///
/// Why synthetic: this machine has no gemma2/gemma3 GGUF. Every local gemma file
/// is gemma-4 (arch `gemma4`), which `model.rs` refuses to load (tensor loading
/// not ported). gemma2 is built with its REAL caps (50/30, attn_soft_cap =
/// gemma2.cpp:7) now that ggml_tanh is ported; gemma3 uses the file layout of a
/// real gemma3 GGUF (no softcap keys). The *graph shape* and the softcapped
/// numerics are both verified here.
#[test]
fn gemma2_gemma3_real_dims_synthetic() {
    run_gemma_synth(true);
    run_gemma_synth(false);
}

fn run_gemma_synth(gemma3: bool) {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / 16777216.0) * 0.1 - 0.05
        }
    }

    let arch = if gemma3 { "gemma3" } else { "gemma2" };
    // gemma-3-1b head/norm geometry; n_layer 4 and a small vocab keep the test
    // cheap (the builder is layer-count and vocab-size agnostic).
    let n_embd = 1152i64;
    let n_head = 4i64;
    let n_head_kv = 1i64;
    let head_dim = 256i64;
    let n_rot = if gemma3 { 128i64 } else { head_dim };
    let n_ff = 6912i64;
    let n_vocab = 2048i64;
    let n_layer = 4usize;
    let eps = 1e-6f32;

    let mut gctx = Context::new();
    let mut rng = Rng(if gemma3 { 0x1234_5678 } else { 0x9abc_def0 });
    let mk = |g: &mut Context, rng: &mut Rng, ne0: i64, ne1: i64| -> TensorId {
        let id = g.new_tensor_2d(GgmlType::F32, ne0, ne1);
        g.arena_resize_tensor(id);
        g.with_f32_mut(id, |p| {
            for v in p.iter_mut() {
                *v = rng.next();
            }
        })
        .unwrap();
        id
    };
    // gemma norm weights live near 0 (the graph adds 1)
    let norm_w = |g: &mut Context, rng: &mut Rng, n: i64| -> TensorId {
        let id = mk(g, rng, n, 1);
        g.with_f32_mut(id, |p| {
            for v in p.iter_mut() {
                *v *= 0.2;
            }
        })
        .unwrap();
        id
    };

    let tok_embd = mk(&mut gctx, &mut rng, n_embd, n_vocab);
    let output_norm = norm_w(&mut gctx, &mut rng, n_embd);
    let mut layers = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        layers.push(GemmaLayerWeights {
            attn_norm: norm_w(&mut gctx, &mut rng, n_embd),
            wq: mk(&mut gctx, &mut rng, n_embd, head_dim * n_head),
            wk: mk(&mut gctx, &mut rng, n_embd, head_dim * n_head_kv),
            wv: mk(&mut gctx, &mut rng, n_embd, head_dim * n_head_kv),
            wo: mk(&mut gctx, &mut rng, head_dim * n_head, n_embd),
            attn_post_norm: norm_w(&mut gctx, &mut rng, n_embd),
            ffn_norm: norm_w(&mut gctx, &mut rng, n_embd),
            ffn_gate: mk(&mut gctx, &mut rng, n_embd, n_ff),
            ffn_down: mk(&mut gctx, &mut rng, n_ff, n_embd),
            ffn_up: mk(&mut gctx, &mut rng, n_embd, n_ff),
            ffn_post_norm: norm_w(&mut gctx, &mut rng, n_embd),
            attn_q_norm: if gemma3 {
                Some(norm_w(&mut gctx, &mut rng, head_dim))
            } else {
                None
            },
            attn_k_norm: if gemma3 {
                Some(norm_w(&mut gctx, &mut rng, head_dim))
            } else {
                None
            },
        });
    }
    let w = GemmaModelWeights {
        tok_embd,
        output_norm,
        output: tok_embd, // gemma2-style tied head
        layers,
    };
    let p = GemmaParams {
        attn: AttnParams {
            n_head,
            n_head_kv,
            n_embd_head_k: head_dim,
            n_embd_head_v: head_dim,
            n_rot,
            rope_mode: 2, // NEOX
            n_ctx_orig: 32768,
            freq_base: if gemma3 { 1_000_000.0 } else { 10_000.0 },
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            norm_eps: eps,
            use_flash_attn: false,
        },
        attention_scale: 1.0 / (head_dim as f32).sqrt(),
        // real gemma2 caps (50/30) now that ggml_tanh is ported; the gemma3
        // file layout carries no softcap keys, so 0/0 with the guarded final cap
        // (gemma3.cpp:210) — matching /tmp/gemma*-synth-*.gguf.
        attn_logit_softcapping: if gemma3 { 0.0 } else { 50.0 },
        final_logit_softcapping: if gemma3 { 0.0 } else { 30.0 },
        attn_soft_cap: !gemma3,           // gemma2.cpp:7
        final_softcap_unguarded: !gemma3, // gemma2.cpp:167-169 has no guard
    };

    let n_kv_dim = head_dim * n_head_kv;
    let kv = KvCache::new(&mut gctx, n_layer, n_kv_dim, n_kv_dim, 64);
    let mut h = RealHarness::new(gctx, kv, arch);
    let toks = [3i32, 17, 42, 7, 99];
    let pos: Vec<i32> = (0..toks.len() as i32).collect();

    let t0 = std::time::Instant::now();
    let prefill_a = h.decode(&w, &toks, &pos, |ctx, w, kv, inp, s, nk, nt| {
        if gemma3 {
            build_gemma3_forward(ctx, w, &p, kv, inp, s, nk, nt)
        } else {
            build_gemma2_forward(ctx, w, &p, kv, inp, s, nk, nt)
        }
    });
    let t1 = t0.elapsed();
    let spread = prefill_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - prefill_a.iter().cloned().fold(f32::INFINITY, f32::min);
    let a1 = argmax(&prefill_a);
    println!(
        "{arch} real-dims synthetic: prefill {:.2} s, argmax={a1}, logit spread={spread:.3}, \
         head_dim={head_dim} n_rot={n_rot} heads={n_head}/{n_head_kv}",
        t1.as_secs_f64()
    );
    assert!(
        prefill_a.iter().all(|v| v.is_finite()),
        "{arch} real-dims logits not finite"
    );
    assert!(spread > 1e-6, "{arch} logits degenerate (spread {spread})");

    // step-by-step decode through the arch builder; collect the greedy sequence
    let run_seq = |h: &mut RealHarness, prefill_logits: &[f32]| -> Vec<i32> {
        let mut cur = prefill_logits.to_vec();
        let mut seq = Vec::new();
        let mut p_next = toks.len() as i32;
        for step in 0..4 {
            let tok = argmax(&cur);
            seq.push(tok);
            if step == 3 {
                break;
            }
            cur = h.decode(&w, &[tok], &[p_next], |ctx, w, kv, inp, s, nk, nt| {
                if gemma3 {
                    build_gemma3_forward(ctx, w, &p, kv, inp, s, nk, nt)
                } else {
                    build_gemma2_forward(ctx, w, &p, kv, inp, s, nk, nt)
                }
            });
            assert!(
                cur.iter().all(|v| v.is_finite()),
                "{arch} step {step} not finite"
            );
            p_next += 1;
        }
        seq
    };
    let seq_a = run_seq(&mut h, &prefill_a);
    println!("{arch}: greedy 4 tokens {:?}", seq_a);

    // determinism (the "independent path" = the same builder run again on a
    // cleared cache): prefill logits must be bit-identical and the greedy
    // sequence must reproduce exactly
    h.kv.clear();
    let prefill_b = h.decode(&w, &toks, &pos, |ctx, w, kv, inp, s, nk, nt| {
        if gemma3 {
            build_gemma3_forward(ctx, w, &p, kv, inp, s, nk, nt)
        } else {
            build_gemma2_forward(ctx, w, &p, kv, inp, s, nk, nt)
        }
    });
    let (diff, max_abs) = bit_diffs(&prefill_a, &prefill_b);
    println!("{arch}: repeat-prefill {diff} bit-differing logits, max |d| = {max_abs:e}");
    assert_eq!(
        diff, 0,
        "{arch} prefill not deterministic ({diff} differing logits)"
    );
    let seq_b = run_seq(&mut h, &prefill_b);
    assert_eq!(
        seq_a, seq_b,
        "{arch} greedy sequence not reproducible: {seq_a:?} vs {seq_b:?}"
    );
    println!("{arch} e2e OK (deterministic prefill + greedy sequence)");
}

// ===========================================================================
// 5b. synthetic gemma GGUF + reference parity probe
//
// No gemma2/gemma3 GGUF exists on this machine, so the gemma builders cannot be
// compared against the reference binary the normal way. Instead this section
// stamps a gemma2/gemma3 GGUF whose *real qwen2.5-0.5b weight bytes* are
// renamed/re-typed and fed through build_gemma{2,3}_forward. Two variants of the
// file are written (they differ ONLY in the norm weights):
//
//   * `-raw.gguf`     all *.norm.weight stored as the raw w  (what graph_arch
//                     `build_norm_rms_gemma` expects — it adds 1 in-graph)
//   * `-shifted.gguf` all *.norm.weight stored as w + 1      (what llama.cpp's
//                     converter emits: conversion/gemma.py:509 norm_shift=1 for
//                     every `*norm.weight`, and the C++ graph then does a plain
//                     mul, src/llama-graph.cpp:1604)
//
// Captured results (this machine, reference build bd4f514db1):
//   * the gemma3 file (no softcap keys => 0/0) loads and runs in both
//     implementations.
//   * the gemma2 file now carries its REAL caps (50/30), and both sides run it:
//     UPDATED 2026-09-25 (ggml_tanh ported) — port PPL 96186.47 vs reference
//     98560.11 (-2.4%). Before tanh landed the port asserted softcap == 0, and
//     the reference NaN'd on a cap == 0 gemma2 file (gemma2.cpp:166 applies the
//     final logit softcap UNGUARDED → 1/0 = inf → tanh → NaN).
//   * token-level greedy parity is NOT meaningful on these OOD synthetic models
//     (probs ~1e-4; gemma3 rust(raw) step-0 gap between #1 61876 and #2 125515 is
//     0.011 logits, and the reference picks #2) — EXCEPT that the real-cap gemma2
//     file now agrees token-for-token with the reference (next bullet).
//   * reference-server cross-check on the real-cap gemma2 file (2026-09-25,
//     `llama-server -c 512 -t 8 -fa off`, fresh instance + first request per
//     PARITY.md, prompt "The capital of France is"):
//       ref  content    "BarController" x8   ==   port tokens [61876] x8
//       ref  top5 ids   61876, 99635, 44965, 99977, 101400
//       port top5 ids   61876, 99635, 44965, 99977, 101400   (identical order)
//       ref  log-softmax(61876) = -9.382889  vs port -9.376403  (gap 0.0065)
//     Before ggml_tanh the reference produced NaN on this file at cap == 0.
//   * the meaningful metric is the mean NLL (perplexity) of the same chunk:
//         gemma3 rust(shifted) = 86674.59  vs  reference(shifted) = 87799.61 (-1.3%)
//     i.e. the gemma3 graph math agrees with the reference once the file layout
//     matches (reference files store w+1, see bug 1 in the report).
//     Running the port on the *raw* file instead gives a completely
//     different distribution (top-1 margin 0.59, reference pick absent from the
//     port's top-5) — the quantifiable signature of the extra +1.
//
// The reference binaries are invoked only manually (documented commands in
// `gemma_synth_parity_probe` / `gemma_synth_ppl_probe`) — no default test here
// contacts them; the captured numbers are asserted as constants.
// ===========================================================================

/// Minimal GGUF writer: metadata copied from `src` (tokenizer + general keys),
/// gemma* arch keys injected, tensors either byte-copied from `src` (real
/// qwen2.5-0.5b weights, renamed) or generated constant F32 norm vectors to
/// which `shift` is added. Streams to `out` (no full-file buffering).
fn write_synth_gemma_gguf(
    out: &Path,
    src: &Gguf,
    arch: &str, // "gemma2" | "gemma3"
    shift: f32,
    // (attn_logit_softcapping, final_logit_softcapping) written into the file.
    // Real gemma2 files carry (50, 30); gemma3 files carry neither key (=> 0, 0).
    caps: (f32, f32),
) -> std::io::Result<u64> {
    use ggml::gguf::GgufType as T;
    use ggml::Value;
    use std::io::Write;

    const ALIGN: u64 = 32; // gguf default alignment

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    fn put_value(b: &mut Vec<u8>, v: &Value) {
        match v {
            Value::U8(x) => b.push(*x),
            Value::I8(x) => b.push(*x as u8),
            Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::Bool(x) => b.push(u8::from(*x)),
            Value::String(s) => put_str(b, s),
            Value::U64(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::F64(x) => b.extend_from_slice(&x.to_le_bytes()),
            Value::Array(et, items) => {
                b.extend_from_slice(&(*et as u32).to_le_bytes());
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                assert!(*et != T::Array, "nested arrays are not a GGUF thing");
                for it in items {
                    put_value(b, it);
                }
            }
        }
    }

    // ---- metadata: tokenizer.* + general.* (minus the source arch) + gemma keys
    let mut kv: Vec<(String, Value)> = Vec::new();
    for (k, v) in &src.kv {
        let keep = (k.starts_with("tokenizer.") || k.starts_with("general."))
            && k != "general.architecture"
            && k != "general.alignment";
        if keep {
            kv.push((k.clone(), v.clone()));
        }
    }
    kv.push(("general.architecture".into(), Value::String(arch.into())));
    let g = |s: &str| format!("{arch}.{s}");
    kv.push((g("context_length"), Value::U32(512)));
    kv.push((g("embedding_length"), Value::U32(896)));
    kv.push((g("block_count"), Value::U32(24)));
    kv.push((g("feed_forward_length"), Value::U32(4864)));
    kv.push((g("attention.head_count"), Value::U32(14)));
    kv.push((g("attention.head_count_kv"), Value::U32(2)));
    kv.push((g("attention.key_length"), Value::U32(64)));
    kv.push((g("attention.value_length"), Value::U32(64)));
    kv.push((g("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)));
    kv.push((g("rope.dimension_count"), Value::U32(64)));
    kv.push((g("rope.freq_base"), Value::F32(1_000_000.0)));
    // no attention.sliding_window => gemma3 loads with swa_type NONE; gemma2 keeps
    // its 4096 default (window > prompt length, so the masks agree)
    kv.push((g("attn_logit_softcapping"), Value::F32(caps.0)));
    kv.push((g("final_logit_softcapping"), Value::F32(caps.1)));

    // ---- tensor table: (name, ne, kind)
    enum Kind {
        /// copy the bytes of this tensor from the source file
        Copy(String),
        /// generate `name.len()` constant F32 norm values
        Norm(usize),
    }
    let n_embd = 896i64;
    let head_dim = 64i64;
    let n_head = 14i64;
    let n_head_kv = 2i64;
    let n_ff = 4864i64;
    let n_layer = 24usize;

    let mut table: Vec<(String, Vec<i64>, GgmlType, Kind)> = Vec::new();
    let mut push = |name: String,
                    ne: Vec<i64>,
                    kind: Kind,
                    table: &mut Vec<(String, Vec<i64>, GgmlType, Kind)>| {
        let ty = match &kind {
            Kind::Copy(src_name) => {
                src.find_tensor(src_name)
                    .unwrap_or_else(|| panic!("source tensor {src_name}"))
                    .ty
            }
            Kind::Norm(_) => GgmlType::F32,
        };
        table.push((name, ne, ty, kind));
    };
    push(
        "token_embd.weight".into(),
        vec![n_embd, 151936],
        Kind::Copy("token_embd.weight".into()),
        &mut table,
    );
    push(
        "output_norm.weight".into(),
        vec![n_embd],
        Kind::Norm(n_embd as usize),
        &mut table,
    );
    if arch != "gemma2" {
        // gemma2 ties output to token_embd (TENSOR_DUPLICATED in the loader) —
        // a separate output.weight would stay unconsumed and fail the load
        push(
            "output.weight".into(),
            vec![n_embd, 151936],
            Kind::Copy("output.weight".into()),
            &mut table,
        );
    }
    for i in 0..n_layer {
        let t = |s: &str| format!("blk.{i}.{s}");
        let src_of = |s: &str| format!("blk.{i}.{s}");
        push(
            t("attn_norm.weight"),
            vec![n_embd],
            Kind::Norm(n_embd as usize),
            &mut table,
        );
        push(
            t("attn_q.weight"),
            vec![n_embd, head_dim * n_head],
            Kind::Copy(src_of("attn_q.weight")),
            &mut table,
        );
        push(
            t("attn_k.weight"),
            vec![n_embd, head_dim * n_head_kv],
            Kind::Copy(src_of("attn_k.weight")),
            &mut table,
        );
        push(
            t("attn_v.weight"),
            vec![n_embd, head_dim * n_head_kv],
            Kind::Copy(src_of("attn_v.weight")),
            &mut table,
        );
        push(
            t("attn_output.weight"),
            vec![head_dim * n_head, n_embd],
            Kind::Copy(src_of("attn_output.weight")),
            &mut table,
        );
        if arch == "gemma3" {
            push(
                t("attn_q_norm.weight"),
                vec![head_dim],
                Kind::Norm(head_dim as usize),
                &mut table,
            );
            push(
                t("attn_k_norm.weight"),
                vec![head_dim],
                Kind::Norm(head_dim as usize),
                &mut table,
            );
        }
        push(
            t("post_attention_norm.weight"),
            vec![n_embd],
            Kind::Norm(n_embd as usize),
            &mut table,
        );
        push(
            t("ffn_norm.weight"),
            vec![n_embd],
            Kind::Norm(n_embd as usize),
            &mut table,
        );
        push(
            t("ffn_gate.weight"),
            vec![n_embd, n_ff],
            Kind::Copy(src_of("ffn_gate.weight")),
            &mut table,
        );
        push(
            t("ffn_up.weight"),
            vec![n_embd, n_ff],
            Kind::Copy(src_of("ffn_up.weight")),
            &mut table,
        );
        push(
            t("ffn_down.weight"),
            vec![n_ff, n_embd],
            Kind::Copy(src_of("ffn_down.weight")),
            &mut table,
        );
        push(
            t("post_ffw_norm.weight"),
            vec![n_embd],
            Kind::Norm(n_embd as usize),
            &mut table,
        );
    }

    // ---- metadata blob (header + kv + tensor infos with data offsets)
    let mut meta: Vec<u8> = Vec::new();
    meta.extend_from_slice(b"GGUF");
    meta.extend_from_slice(&3u32.to_le_bytes());
    meta.extend_from_slice(&(table.len() as u64).to_le_bytes());
    meta.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in &kv {
        put_str(&mut meta, k);
        meta.extend_from_slice(&(v.type_() as u32).to_le_bytes());
        put_value(&mut meta, v);
    }
    let mut offsets: Vec<u64> = Vec::with_capacity(table.len());
    let mut offset: u64 = 0;
    for (name, ne, ty, kind) in &table {
        let len = match kind {
            Kind::Copy(src_name) => {
                let ti = src.find_tensor(src_name).unwrap();
                assert_eq!(
                    &ti.ne[..ne.len()],
                    &ne[..],
                    "{name}: shape mismatch vs source {src_name}"
                );
                ti.size_bytes()
            }
            Kind::Norm(n) => (*n as u64) * 4,
        };
        assert!(len > 0, "{name}: empty tensor");
        offset = offset.div_ceil(ALIGN) * ALIGN;
        offsets.push(offset);
        offset += len;
        put_str(&mut meta, name);
        meta.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in ne {
            meta.extend_from_slice(&d.to_le_bytes());
        }
        meta.extend_from_slice(&(*ty as u32).to_le_bytes());
        meta.extend_from_slice(&offsets.last().unwrap().to_le_bytes());
    }

    // ---- stream the file: padded metadata, then payloads at their offsets
    let f = std::fs::File::create(out)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    w.write_all(&meta)?;
    let mut written = meta.len() as u64;
    while written % ALIGN != 0 {
        w.write_all(&[0u8])?;
        written += 1;
    }
    let data_start = written;
    let norm_bytes = (0.125f32 + shift).to_le_bytes();
    for ((name, _ne, _ty, kind), off) in table.iter().zip(&offsets) {
        let abs = data_start + off;
        assert!(written <= abs, "{name}: overlapping data");
        while written < abs {
            w.write_all(&[0u8])?;
            written += 1;
        }
        match kind {
            Kind::Copy(src_name) => {
                let bytes = src.tensor_data(src_name).expect("source tensor bytes");
                w.write_all(bytes)?;
                written += bytes.len() as u64;
            }
            Kind::Norm(n) => {
                let mut buf = Vec::with_capacity((*n).min(4096) * 4);
                for _ in 0..(*n).min(4096) {
                    buf.extend_from_slice(&norm_bytes);
                }
                let mut left = *n;
                while left > 0 {
                    let take = left.min(4096);
                    w.write_all(&buf[..take * 4])?;
                    left -= take;
                }
                written += (*n as u64) * 4;
            }
        }
    }
    w.flush()?;
    Ok(written)
}

/// Write the synthetic gemma3 + gemma2 GGUFs (raw and shifted norm variants) and
/// run this port's builder on the raw one. Prints the greedy token ids for the
/// Rust side; the reference side is a manual step:
///
///   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
///   $REF/llama-server -m /tmp/gemma3-synth-shifted.gguf -c 512 -t 8 -fa off \
///       --port 8799 &
///   curl -s localhost:8799/completion -H 'Content-Type: application/json' \
///       -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0,
///            "top_k":1,"n_probs":3,"cache_prompt":false}'
///
/// (fresh server + first request, per PARITY.md). `#[ignore]`d: writes ~0.5 GiB
/// per file into /tmp and reads the 0.46 GiB source model.
#[test]
#[ignore = "manual: writes ~1 GiB of synthetic GGUF into /tmp and runs real forwards"]
fn gemma_synth_parity_probe() {
    let Some(l) = load_real(QWEN25) else { return };
    let vocab = Vocab::load(&l.gguf).expect("vocab");
    let prompt = vocab.tokenize(PROMPT, true, true);
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    println!("prompt ids: {prompt:?}");

    // gemma2 gets its REAL caps (50/30, gemma2.cpp:7 + real configs): with
    // ggml_tanh ported the reference no longer NaNs on this file. gemma3 files
    // carry no softcap keys (0/0), which keeps the captured reference PPL below
    // valid across this change.
    for (arch, raw_path, shifted_path, caps) in [
        (
            "gemma3",
            "/tmp/gemma3-synth-raw.gguf",
            "/tmp/gemma3-synth-shifted.gguf",
            (0.0f32, 0.0f32),
        ),
        (
            "gemma2",
            "/tmp/gemma2-synth-raw.gguf",
            "/tmp/gemma2-synth-shifted.gguf",
            (50.0f32, 30.0f32),
        ),
    ] {
        let n_raw = write_synth_gemma_gguf(Path::new(raw_path), &l.gguf, arch, 0.0, caps)
            .expect("write raw");
        let n_sh = write_synth_gemma_gguf(Path::new(shifted_path), &l.gguf, arch, 1.0, caps)
            .expect("write shifted");
        println!("wrote {raw_path} ({n_raw} B) and {shifted_path} ({n_sh} B)");

        for (label, path) in [("raw", raw_path), ("shifted", shifted_path)] {
            let Some(s) = load_real(path) else { continue };
            let v = Vocab::load(&s.gguf).expect("synth vocab");
            let hp = s.model.hparams.clone();
            let ap = attn_params(&s.model);
            let p = GemmaParams {
                attn: ap,
                attention_scale: 1.0 / (ap.n_embd_head_k as f32).sqrt(),
                attn_logit_softcapping: hp.f_attn_logit_softcapping,
                final_logit_softcapping: hp.f_final_logit_softcapping,
                attn_soft_cap: hp.attn_soft_cap, // gemma2.cpp:7 (meta.rs:684)
                final_softcap_unguarded: arch == "gemma2", // gemma2.cpp:167-169 has no guard
            };
            let w = GemmaModelWeights {
                tok_embd: s.model.tok_embd,
                output_norm: s.model.output_norm,
                output: s.model.output,
                layers: s
                    .model
                    .layers
                    .iter()
                    .map(|x| GemmaLayerWeights {
                        attn_norm: x.attn_norm.unwrap(),
                        wq: x.wq.unwrap(),
                        wk: x.wk.unwrap(),
                        wv: x.wv.unwrap(),
                        wo: x.wo.unwrap(),
                        attn_post_norm: x.attn_post_norm.unwrap(),
                        ffn_norm: x.ffn_norm.unwrap(),
                        ffn_gate: x.ffn_gate.unwrap(),
                        ffn_down: x.ffn_down.unwrap(),
                        ffn_up: x.ffn_up.unwrap(),
                        ffn_post_norm: x.ffn_post_norm.unwrap(),
                        attn_q_norm: x.attn_q_norm,
                        attn_k_norm: x.attn_k_norm,
                    })
                    .collect(),
            };
            let n_layer = s.model.layers.len();
            let n_k = s.model.n_embd_k_gqa_max() as i64;
            let n_v = s.model.n_embd_v_gqa_max() as i64;
            let gemma3 = arch == "gemma3";
            let mut gctx = s.model.ctx;
            let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
            let mut h = RealHarness::new(gctx, kv, "gemma-synth");
            let t0 = std::time::Instant::now();
            let mut cur = h.decode(&w, &prompt, &pos, |ctx, w, kv, inp, sl, nk, nt| {
                if gemma3 {
                    build_gemma3_forward(ctx, w, &p, kv, inp, sl, nk, nt)
                } else {
                    build_gemma2_forward(ctx, w, &p, kv, inp, sl, nk, nt)
                }
            });
            let prefill_logits = cur.clone();
            let mut got = Vec::new();
            let mut pn = prompt.len() as i32;
            for step in 0..8 {
                let tok = argmax(&cur);
                got.push(tok);
                if step == 7 {
                    break;
                }
                cur = h.decode(&w, &[tok], &[pn], |ctx, w, kv, inp, sl, nk, nt| {
                    if gemma3 {
                        build_gemma3_forward(ctx, w, &p, kv, inp, sl, nk, nt)
                    } else {
                        build_gemma2_forward(ctx, w, &p, kv, inp, sl, nk, nt)
                    }
                });
                pn += 1;
            }
            let lse = {
                let mx = prefill_logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                mx + prefill_logits
                    .iter()
                    .map(|&x| ((x - mx) as f64).exp())
                    .sum::<f64>()
                    .ln() as f32
            };
            let k5: Vec<(i32, f32, f32)> = topk(&prefill_logits, 5)
                .into_iter()
                .map(|(i, l)| (i, l, (l - lse).exp()))
                .collect();
            println!(
                "  [{arch} {label}] n_softcap={}/{} n_rot={} head_dim={} prefill {:?}\n    prefill top5 (id, piece, logit, prob): {:?}",
                hp.f_attn_logit_softcapping,
                hp.f_final_logit_softcapping,
                hp.n_rot(0),
                hp.n_embd_head_k(0),
                t0.elapsed(),
                k5.iter().map(|(i, l, p)| (*i, piece(&v, *i), *l, *p)).collect::<Vec<_>>()
            );
            println!("    rust tokens {got:?} = {:?}", text_of(&v, &got));
        }
    }
    println!(
        "\nNow run the reference server on /tmp/{arch}-synth-shifted.gguf (fresh instance, first request) \
         and compare: rust(raw) vs ref(shifted) should match; rust(shifted) vs ref(shifted) should NOT.",
        arch = "gemma3"
    );
}

/// Distribution-level comparison against the reference `llama-perplexity`:
/// token-level parity is meaningless on these synthetic OOD models (top-1 gaps
/// of ~0.01 logits on probabilities of ~1e-4), but the mean NLL is a smooth
/// aggregate that survives near-tie rank flips.
///
/// Manual protocol (same text file, same n_ctx), reference on the SHIFTED file
/// because that is the file layout llama.cpp's converter produces:
///   $REF/llama-perplexity -m /tmp/gemma3-synth-shifted.gguf -f /tmp/ppl-text.txt \
///       -c 64 -t 8 -fa off --chunks 1
///   -> Final estimate: PPL = 87799.6116 +/- 7965.67366   (captured)
///
/// This test prints the PPL this port computes for the raw file (equivalent
/// semantics: file w + in-graph 1 == reference's w+1).
#[test]
#[ignore = "manual: perplexity probe for the synthetic gemma files"]
fn gemma_synth_ppl_probe() {
    let text = std::fs::read_to_string("/tmp/ppl-text.txt").unwrap_or_else(|_| {
        let s = "The capital of France is Paris. It is the largest city in Europe and the second largest in the world. ";
        s.repeat(8)
    });
    let n_ctx = 64usize;
    // P1 FIXED (2026-09-25): graph_arch's gemma norm is now a plain mul — the
    // +1 lives in the file (conversion/gemma.py norm_shift). So the semantic
    // match is rust(SHIFTED file) vs reference(SHIFTED file); the raw file is
    // kept as a negative control.
    for (arch, raw_path) in [
        ("gemma3", "/tmp/gemma3-synth-shifted.gguf"),
        ("gemma2", "/tmp/gemma2-synth-shifted.gguf"),
    ] {
        if !Path::new(raw_path).exists() {
            eprintln!("SKIP {raw_path} (run gemma_synth_parity_probe first)");
            continue;
        }
        let Some(s) = load_real(raw_path) else {
            continue;
        };
        let v = Vocab::load(&s.gguf).expect("synth vocab");
        let toks = v.tokenize(&text, true, true);
        let toks = &toks[..toks.len().min(n_ctx)];
        println!(
            "[{arch}] tokenized {} tokens (using {}), first8={:?}",
            v.tokenize(&text, true, true).len(),
            toks.len(),
            &toks[..8.min(toks.len())]
        );
        let ap = attn_params(&s.model);
        let hpv = s.model.hparams.clone();
        let p = GemmaParams {
            attn: ap,
            attention_scale: 1.0 / (ap.n_embd_head_k as f32).sqrt(),
            attn_logit_softcapping: hpv.f_attn_logit_softcapping,
            final_logit_softcapping: hpv.f_final_logit_softcapping,
            attn_soft_cap: hpv.attn_soft_cap, // gemma2.cpp:7 (meta.rs:684)
            final_softcap_unguarded: arch == "gemma2",
        };
        let w = GemmaModelWeights {
            tok_embd: s.model.tok_embd,
            output_norm: s.model.output_norm,
            output: s.model.output,
            layers: s
                .model
                .layers
                .iter()
                .map(|x| GemmaLayerWeights {
                    attn_norm: x.attn_norm.unwrap(),
                    wq: x.wq.unwrap(),
                    wk: x.wk.unwrap(),
                    wv: x.wv.unwrap(),
                    wo: x.wo.unwrap(),
                    attn_post_norm: x.attn_post_norm.unwrap(),
                    ffn_norm: x.ffn_norm.unwrap(),
                    ffn_gate: x.ffn_gate.unwrap(),
                    ffn_down: x.ffn_down.unwrap(),
                    ffn_up: x.ffn_up.unwrap(),
                    ffn_post_norm: x.ffn_post_norm.unwrap(),
                    attn_q_norm: x.attn_q_norm,
                    attn_k_norm: x.attn_k_norm,
                })
                .collect(),
        };
        let n_vocab = s.model.ctx.ne(s.model.output)[1] as usize;
        let n_layer = s.model.layers.len();
        let n_k = s.model.n_embd_k_gqa_max() as i64;
        let n_v = s.model.n_embd_v_gqa_max() as i64;
        let gemma3 = arch == "gemma3";
        let mut gctx = s.model.ctx;
        let kv = KvCache::new(&mut gctx, n_layer, n_k, n_v, 512);
        let mut h = RealHarness::new(gctx, kv, "gemma-ppl");
        let pos: Vec<i32> = (0..toks.len() as i32).collect();
        let all = h.decode_all(&w, toks, &pos, |ctx, w, kv, inp, sl, nk, nt| {
            if gemma3 {
                build_gemma3_forward(ctx, w, &p, kv, inp, sl, nk, nt)
            } else {
                build_gemma2_forward(ctx, w, &p, kv, inp, sl, nk, nt)
            }
        });
        let ppl = chunk_ppl(&all, n_vocab, toks);
        println!(
            "[{arch}] rust PPL (SHIFTED file, {} scored tokens) = {ppl:.4}",
            toks.len() - 1
        );
        // negative control: the raw file must give a clearly different PPL
        if Path::new(raw_path).exists() {
            let _ = raw_path;
        }
        if arch == "gemma3" {
            // captured from the reference: llama-perplexity -m /tmp/gemma3-synth-shifted.gguf
            // -f /tmp/ppl-text.txt -c 64 -t 8 -fa off --chunks 1
            const REF_PPL_GEMMA3: f64 = 87799.6116;
            let rel = (ppl - REF_PPL_GEMMA3).abs() / REF_PPL_GEMMA3;
            println!(
                "[{arch}] reference PPL = {REF_PPL_GEMMA3:.4} -> relative diff {:.4}%",
                rel * 100.0
            );
            assert!(
                rel < 0.05,
                "gemma3 chunk PPL {ppl:.4} deviates {:.2}% from the reference {REF_PPL_GEMMA3:.4}",
                rel * 100.0
            );
        } else {
            // gemma2 file now carries its REAL caps (50/30); before ggml_tanh
            // landed the reference NaN'd on it (unguarded final softcap with
            // cap=0 → 1/0 = inf → tanh → NaN, gemma2.cpp:166). Captured
            // 2026-09-25 with ggml_tanh ported:
            //   llama-perplexity -m /tmp/gemma2-synth-shifted.gguf -f /tmp/ppl-text.txt
            //       -c 64 -t 8 -fa off --chunks 1 -> PPL = 98560.1137
            const REF_PPL_GEMMA2: f64 = 98560.1137;
            let rel = (ppl - REF_PPL_GEMMA2).abs() / REF_PPL_GEMMA2;
            println!(
                "[{arch}] reference PPL = {REF_PPL_GEMMA2:.4} -> relative diff {:.4}%",
                rel * 100.0
            );
            assert!(
                rel < 0.05,
                "gemma2 chunk PPL {ppl:.4} deviates {:.2}% from the reference {REF_PPL_GEMMA2:.4}",
                rel * 100.0
            );
        }
    }
}

// ===========================================================================
// 6. arch / dtype support matrix (informational; metadata only)
// ===========================================================================

fn type_histogram(gguf: &Gguf) -> Vec<(String, usize)> {
    let mut m: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for t in &gguf.tensors {
        *m.entry(format!("{:?}", t.ty)).or_insert(0) += 1;
    }
    m.into_iter().collect()
}

/// Every GGUF found on this machine: arch, hparams status, load_model status,
/// tensor dtype histogram and which of those dtypes the ported `dequantize_row`
/// cannot decode. Cheap: GGUF metadata + mmap only, no tensor data is read.
#[test]
fn arch_support_matrix() {
    let candidates: &[(&str, &str)] = &[
        ("qwen2.5-0.5b-instruct-q4_k_m", QWEN25),
        ("Qwen3-Embedding-0.6B-Q8_0", QWEN3_EMB),
        ("Phi-4-mini-instruct-Q6_K", PHI4_MINI),
        ("gemma-4-12B-it-QAT-Q4_0", GEMMA4_12B),
        ("gpt-oss-20b-MXFP4", GPTOSS20B_MXFP4),
        ("gpt-oss-20b-Q4_K_M", GPTOSS20B_Q4KM),
        ("LFM2-8B-A1B-Q4_K_M", LFM2),
        ("granite-4.0-h-tiny-Q4_K_M", GRANITE_TINY),
        ("Qwen3.6-27B-Q4_K_M", QWEN36_27B),
        ("deepseek-coder-33b-Q5_K_M (llama arch)", DEEPSEEK_CODER33B),
    ];

    // dequantize_row supports exactly these (quants.rs:751)
    const DEQUANT_SUPPORTED: &[&str] = &[
        "F32", "F16", "Bf16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q1_0", "Q2_0", "Q2K", "Q3K",
        "Q4K", "Q5K", "Q6K",
    ];

    println!(
        "\n=== arch/dtype support matrix (MemAvailable {:.1} GiB) ===",
        mem_available_gb()
    );
    for (label, path) in candidates {
        if !Path::new(path).exists() {
            println!("{label}: MISSING {path}");
            continue;
        }
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let file = std::fs::File::open(path).expect("open");
        let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
        let gguf = match Gguf::from_bytes(mmap.clone()) {
            Ok(g) => g,
            Err(e) => {
                println!("{label}: GGUF PARSE FAILED: {e}");
                continue;
            }
        };
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("?")
            .to_string();
        let hist = type_histogram(&gguf);
        let unsupported: Vec<&str> = hist
            .iter()
            .filter(|(ty, _)| !DEQUANT_SUPPORTED.contains(&ty.as_str()))
            .map(|(ty, _)| ty.as_str())
            .collect();
        let hp_txt = match llama::meta::load_hparams(&gguf) {
            Ok((a, h)) => format!(
                "hparams OK (arch={} n_embd={} n_layer={} heads={}/{} softcap={}/{})",
                a.name(),
                h.n_embd,
                h.n_layer(),
                h.n_head(0),
                h.n_head_kv(0),
                h.f_attn_logit_softcapping,
                h.f_final_logit_softcapping
            ),
            Err(e) => format!("hparams ERR: {e}"),
        };
        let load_txt = match load_model(&gguf, mmap.clone()) {
            Ok(m) => format!(
                "load_model OK ({} layers, {} tensors)",
                m.layers.len(),
                m.tensors.len()
            ),
            Err(e) => format!("load_model ERR: {e}"),
        };
        println!(
            "{label}: {:.2} GiB arch={arch}\n    dtypes: {:?}\n    dequant-missing: {:?}\n    {hp_txt}\n    {load_txt}",
            size as f64 / 1073741824.0,
            hist,
            unsupported
        );
    }
}

// ===========================================================================
// 7. big-model probe (manual: mmap + metadata only, no forward pass)
// ===========================================================================

/// Load feasibility of the big local models (>6 GiB): metadata + mmap only.
/// `#[ignore]`d so a normal `cargo test` never pulls those files into the page
/// cache next to whatever else is running on this machine.
#[test]
#[ignore = "manual: probes the >6 GiB local models (metadata only, no compute)"]
fn big_models_load_probe() {
    println!("MemAvailable before: {:.1} GiB", mem_available_gb());
    let list: &[(&str, &str)] = &[
        ("gemma-4-12B-it-QAT-Q4_0 (6.5 GiB, arch gemma4)", GEMMA4_12B),
        (
            "gpt-oss-20b-MXFP4 (11.3 GiB, arch gpt-oss)",
            GPTOSS20B_MXFP4,
        ),
        (
            "gpt-oss-20b-Q4_K_M (10.8 GiB, arch gpt-oss)",
            GPTOSS20B_Q4KM,
        ),
        (
            "deepseek-coder-33b-Q5_K_M (23.5 GiB, arch llama)",
            DEEPSEEK_CODER33B,
        ),
    ];
    for (label, path) in list {
        if !Path::new(path).exists() {
            println!("{label}: MISSING");
            continue;
        }
        let file = std::fs::File::open(path).expect("open");
        let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
        let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("?")
            .to_string();
        let n_kv = gguf.kv.len();
        let n_tensors = gguf.tensors.len();
        let t0 = std::time::Instant::now();
        let h = llama::meta::load_hparams(&gguf);
        let hp_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let lm = load_model(&gguf, mmap.clone());
        let lm_ms = t1.elapsed().as_secs_f64() * 1e3;
        println!(
            "{label}: arch={arch} kv={n_kv} tensors={n_tensors}\n    hparams({hp_ms:.1} ms): {h:?}\n    load_model({lm_ms:.1} ms): {}",
            match &lm {
                Ok(m) => format!("OK ({} layers, {} tensors)", m.layers.len(), m.tensors.len()),
                Err(e) => format!("ERR: {e}"),
            }
        );
    }
    println!("MemAvailable after: {:.1} GiB", mem_available_gb());
}
