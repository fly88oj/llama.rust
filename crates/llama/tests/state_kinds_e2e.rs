//! state_kinds_e2e.rs — the remaining state-serialization kinds
//! (`llama_state_seq_get_data` / `llama_state_get_data` @ bd4f514db1) on the
//! two synthetic models that carry them:
//!
//!   * **deepseek32** (tests/arch_batch6_e2e.rs's `deepseek32-synth.gguf`) —
//!     the dsa pair's **lid half**: `llama_kv_cache_dsa::state_write`
//!     (llama-kv-cache-dsa.cpp:164-172) = the K-only MLA base half followed
//!     by the K-only indexer-key half over its own lockstep cells;
//!   * **minimax-m3** (tests/arch_batch11a_e2e.rs's `minimax-m3-synth.gguf`)
//!     — the **MSA idx half**: `llama_kv_cache_msa::state_write`
//!     (llama-kv-cache-msa.cpp:160-168) = the plain base half followed by
//!     the idx half (K rows + the never-written zero V rows of the
//!     `hparams_idx` clone);
//!   * both — the **whole-context model-info header** of
//!     `llama_state_get_data` (llama-context.cpp:3341-3357:
//!     `llama_io_write_i::write_string(llm_arch_name(arch))` around
//!     `memory->state_write(io)` = seq_id -1).
//!
//! The models are the batch generators' (regenerate with the batch-6/-11a
//! `#[ignore]` writers or `parity/state_kinds_parity.sh`); the default tests
//! are the round-trip continuation bit-identity of the dsv4 protocol
//! (tests/dsv4_state_e2e.rs), the `#[ignore]` `*_dump_blob`s feed
//! `parity/state_kinds_parity.sh` whose reference side is
//! parity/ref_state_kinds.c.

use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};

const DEEPSEEK32: &str = "/tmp/arch-batch6/deepseek32-synth.gguf";
const MINIMAX_M3: &str = "/tmp/arch-batch11a/minimax-m3-synth.gguf";
const OUT_DIR: &str = "/tmp/arch-statekinds";

/// the fixed prefill of parity/ref_state_kinds.c
const TOKS: [i32; 16] = [3, 17, 42, 9, 21, 5, 8, 30, 11, 29, 2, 16, 4, 13, 25, 7];
const N_TAIL: usize = 8;

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e} — run the batch-6/-11a synth writers"));
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process reads
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn synth_attn(m: &LlamaModel, fa: bool) -> AttnParams {
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
        use_flash_attn: fa,
    }
}

/// the deepseek32 driver — tests/arch_batch6_e2e.rs's `forward_of` Deepseek32
/// arm (the shared MLA weight set + the DSA indexer params); `new_with`
/// attaches the dsa lid cache on the Deepseek32 weights (context.rs)
fn deepseek32_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa);
    let n_trunk = hp.n_layer() as usize;
    let ds2 = llama::graph_arch::Deepseek2Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        kv_lora_rank: hp.n_lora_kv as i64,
        rope_yarn_log_mul: hp.rope_yarn_log_mul,
        f_attn_temp_scale: hp.f_attn_temp_scale,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        is_ocr: false,
    };
    let layers = m.layers[..n_trunk]
        .iter()
        .map(|l| llama::graph_arch::Deepseek2LayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wq: l.wq,
            wqkv: l.wqkv,
            wqkv_b: l.wqkv_b,
            wk: l.wk,
            wv: l.wv,
            wq_a: l.wq_a,
            attn_q_a_norm: l.attn_q_a_norm,
            wq_b: l.wq_b,
            wkv_a_mqa: l.wkv_a_mqa,
            attn_kv_a_norm: l.attn_kv_a_norm,
            indexer_k_norm: l.indexer_k_norm,
            indexer_k_norm_b: l.indexer_k_norm_b,
            indexer_proj: l.indexer_proj,
            indexer_attn_k: l.indexer_attn_k,
            indexer_attn_q_b: l.indexer_attn_q_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wkv_b: l.wkv_b,
            wo: l.wo.unwrap(),
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_gate_up_exps: l.ffn_gate_up_exps,
        })
        .collect();
    let weights = ForwardWeights::Deepseek32(
        llama::graph_arch::Deepseek2ModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers,
        },
        llama::graph_arch::Deepseek32Params {
            ds2,
            indexer_n_head: hp.indexer_n_head as i64,
            indexer_head_size: hp.indexer_head_size as i64,
            indexer_top_k: hp.indexer_top_k as i64,
            f_norm_eps: hp.f_norm_eps,
        },
    );
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    // the ref probe's geometry: n_ctx 512, n_ubatch 512, FA ON (v_trans = 0
    // is the only layout the port's caches match)
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
}

/// the minimax-m3 driver — tests/arch_batch11a_e2e.rs's MinimaxM3 arm; the
/// MSA idx cache attaches on the MinimaxM3 weights
fn minimax_m3_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa);
    let n_trunk = hp.n_layer() as usize;
    let layers = m.layers[..n_trunk]
        .iter()
        .map(|l| llama::graph_arch::MinimaxM3LayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            wqkv: l.wqkv,
            wq: l.wq,
            wk: l.wk,
            wv: l.wv,
            wo: l.wo.unwrap(),
            attn_q_norm: l.attn_q_norm.unwrap(),
            attn_k_norm: l.attn_k_norm.unwrap(),
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
            index_q_proj: l.index_q_proj,
            index_k_proj: l.index_k_proj,
            index_q_norm: l.index_q_norm,
            index_k_norm: l.index_k_norm,
        })
        .collect();
    let weights = ForwardWeights::MinimaxM3(
        llama::graph_arch::MinimaxM3ModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers,
        },
        llama::graph_arch::MinimaxM3Params {
            n_embd: hp.n_embd as i64,
            n_head: hp.n_head(0) as i64,
            n_head_kv: hp.n_head_kv(0) as i64,
            n_embd_head: hp.n_embd_head_k(0) as i64,
            n_rot: hp.n_rot(0) as i64,
            n_layer_dense_lead: hp.n_layer_dense_lead,
            n_ff_exp: hp.n_ff_exp(0) as i64,
            n_expert_shared: hp.n_expert_shared as i64,
            n_expert: hp.n_expert as i64,
            n_expert_used: hp.n_expert_used(0) as i64,
            expert_weights_norm: hp.expert_weights_norm,
            expert_weights_scale: hp.expert_weights_scale,
            expert_gating_func: hp.expert_gating_func as i32,
            msa_blk: hp.indexer_block_size as i64,
            msa_topk_blocks: hp.indexer_top_k as i64,
            msa_local: hp.indexer_local_blocks as i64,
            indexer_n_head: hp.indexer_n_head as i64,
            indexer_head_size: hp.indexer_head_size as i64,
            attn,
        },
    );
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
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

/// the driver trait object of one model kind
enum Kind {
    Deepseek32,
    MinimaxM3,
}

impl Kind {
    fn path(&self) -> &'static str {
        match self {
            Kind::Deepseek32 => DEEPSEEK32,
            Kind::MinimaxM3 => MINIMAX_M3,
        }
    }
    fn arch_name(&self) -> &'static str {
        match self {
            Kind::Deepseek32 => "deepseek32",
            Kind::MinimaxM3 => "minimax-m3",
        }
    }
    fn driver(&self, m: &mut LlamaModel, fa: bool) -> DecodeContext {
        match self {
            Kind::Deepseek32 => deepseek32_driver(m, fa),
            Kind::MinimaxM3 => minimax_m3_driver(m, fa),
        }
    }
}

/// prefill + tail: the ref probe's fixed stream — 16 tokens, then 8
/// single-token steps whose fed id is `TOKS[s % 16]` (ref_state_kinds.c).
/// Returns the last step's logits.
fn prefill_and_tail(d: &mut DecodeContext) -> Vec<f32> {
    let pos: Vec<i32> = (0..TOKS.len() as i32).collect();
    let mut logits = d.decode(&TOKS, &pos).expect("prefill").to_vec();
    for s in 0..N_TAIL {
        let p = TOKS.len() as i32 + s as i32;
        logits = d
            .decode(&[TOKS[s % TOKS.len()]], &[p])
            .expect("tail")
            .to_vec();
    }
    logits
}

/// greedy continuation logits after the state restore point
fn continue_greedy(d: &mut DecodeContext, logits0: Vec<f32>, n: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut id = argmax(&logits0);
    let mut p = (TOKS.len() + N_TAIL) as i32;
    for _ in 0..n {
        let logits = d.decode(&[id], &[p]).expect("continue").to_vec();
        id = argmax(&logits);
        out.push(logits);
        p += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// the round-trip tests (the dsv4_state_round_trip protocol)
// ---------------------------------------------------------------------------

/// serialize sequence 0 after prefill+tail, restore into a fresh context of
/// the same model, continue greedy — every step's logits must equal the
/// uninterrupted run's bit-for-bit. The blob carries the lid/idx rows; a
/// dropped or mis-rowed half shifts the logits immediately (the DSA indexer
/// reads its lid rows, the MSA attention its idx rows, every step).
#[test]
fn state_kinds_round_trip() {
    for kind in [Kind::Deepseek32, Kind::MinimaxM3] {
        // fa = true: the v_trans = 0 layout (the ref-comparable one)
        for fa in [true, false] {
            let mut m_a = open_model(kind.path());
            let mut a = kind.driver(&mut m_a, fa);
            let logits_a = prefill_and_tail(&mut a);
            // the checkpoint BEFORE the compared continuation (the dsv4
            // round-trip's structure: the blob is the tail's starting state)
            let blob = a.state_seq_get_data(0, false);
            assert!(
                blob.len() > 64,
                "{}: implausibly small blob",
                kind.arch_name()
            );
            let cont_a = continue_greedy(&mut a, logits_a.clone(), 8);

            let mut m_b = open_model(kind.path());
            let mut b = kind.driver(&mut m_b, fa);
            b.state_seq_set_data(0, &blob, false)
                .unwrap_or_else(|e| panic!("{}: restore: {e}", kind.arch_name()));

            // the restored context's positions: the driver validates against
            // the cache's seq_pos_max — the restore replays the cells
            let cont_b = continue_greedy(&mut b, logits_a, 8);

            assert_eq!(
                cont_a.len(),
                cont_b.len(),
                "{} fa={fa}: continuation length",
                kind.arch_name()
            );
            for (i, (x, y)) in cont_a.iter().zip(&cont_b).enumerate() {
                assert_eq!(
                    x,
                    y,
                    "{} fa={fa}: continuation step {i} diverged",
                    kind.arch_name()
                );
            }
            println!(
                "{} fa={fa}: seq blob {} bytes, 8-step continuation bit-identical",
                kind.arch_name(),
                blob.len()
            );
        }
    }
}

/// the whole-context blob: `state_get_data` = the arch-string header + the
/// seq(-1) serialization (+ the MSA idx half); `state_set_data` validates the
/// arch and restores, and the dummy-mode `state_get_size` matches.
#[test]
fn state_kinds_whole_context_round_trip() {
    for kind in [Kind::Deepseek32, Kind::MinimaxM3] {
        let mut m_a = open_model(kind.path());
        let mut a = kind.driver(&mut m_a, true);
        let logits_a = prefill_and_tail(&mut a);
        let arch = kind.arch_name();
        // the checkpoint BEFORE the compared continuation
        let blob = a.state_get_data(arch);
        let size = a.state_get_size(arch);
        assert_eq!(blob.len(), size, "{arch}: state_get_size mismatch");
        let cont_a = continue_greedy(&mut a, logits_a.clone(), 4);

        // the header: [u32 len][arch bytes] then the memory blob
        let len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        assert_eq!(&blob[4..4 + len], arch.as_bytes(), "{arch}: header arch");

        let mut m_b = open_model(kind.path());
        let mut b = kind.driver(&mut m_b, true);
        b.state_set_data(arch, &blob)
            .unwrap_or_else(|e| panic!("{arch}: whole restore: {e}"));
        // a wrong arch is refused (llama-context.cpp:3370-3373)
        assert!(
            b.state_set_data("llama", &blob).is_err(),
            "{arch}: wrong arch accepted"
        );

        let cont_b = continue_greedy(&mut b, logits_a, 4);
        for (i, (x, y)) in cont_a.iter().zip(&cont_b).enumerate() {
            assert_eq!(x, y, "{arch}: whole-context continuation step {i} diverged");
        }
        println!(
            "{arch}: whole-context blob {} bytes (header {} + memory), 4-step continuation \
             bit-identical",
            blob.len(),
            4 + len
        );
    }
}

// ---------------------------------------------------------------------------
// the #[ignore] dumper for parity/state_kinds_parity.sh
// ---------------------------------------------------------------------------

/// `<tag>:SQST:<len><blob>` + `<tag>:FULL:<len><blob>` — the port halves of
/// ref_state_kinds.c's dumps
#[test]
#[ignore = "dumps /tmp/arch-statekinds for parity/state_kinds_parity.sh"]
fn state_kinds_dump_blobs() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    for kind in [Kind::Deepseek32, Kind::MinimaxM3] {
        for tail in [false, true] {
            let mut m = open_model(kind.path());
            let mut d = kind.driver(&mut m, true);
            // the probe's prefill; the tail variant rolls 8 more steps
            let pos: Vec<i32> = (0..TOKS.len() as i32).collect();
            d.decode(&TOKS, &pos).expect("prefill");
            if tail {
                for s in 0..N_TAIL {
                    let p = TOKS.len() as i32 + s as i32;
                    d.decode(&[TOKS[s % TOKS.len()]], &[p]).expect("tail");
                }
            }

            let tag = format!("{}{}", kind.arch_name(), if tail { "-tail" } else { "" });
            let seq = d.state_seq_get_data(0, false);
            let full = d.state_get_data(kind.arch_name());

            let wrap = |magic: &str, blob: &[u8]| -> Vec<u8> {
                let mut v = Vec::with_capacity(8 + blob.len());
                v.extend_from_slice(magic.as_bytes());
                v.extend_from_slice(&(blob.len() as u32).to_le_bytes());
                v.extend_from_slice(blob);
                v
            };
            std::fs::write(format!("{OUT_DIR}/port-{tag}-seq.bin"), wrap("SQST", &seq)).unwrap();
            std::fs::write(
                format!("{OUT_DIR}/port-{tag}-full.bin"),
                wrap("FULL", &full),
            )
            .unwrap();
            println!(
                "{tag}: seq {} bytes, full {} bytes -> {OUT_DIR}/port-{tag}-*.bin",
                seq.len(),
                full.len()
            );
        }
    }
}
