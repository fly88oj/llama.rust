//! recurrent_state_e2e.rs — the recurrent-memory state serialization
//! (`llama_memory_recurrent::state_write` / `state_read`,
//! llama-memory-recurrent.cpp:766-1224 @ bd4f514db1) on the batch-5
//! synthetic models:
//!
//!   * **mamba2** — the pure-recurrent memory (`llm_arch_is_recurrent`,
//!     llama-model.cpp:2538-2548): the blob is the recurrent half alone
//!     (`[u32 cell_count][pos meta][s_trans|n_layer][per-layer F32 r/s rows]`);
//!   * **jamba** — `llama_memory_hybrid` (llama-memory-hybrid.cpp:190-195):
//!     the plain attn KV half followed by the recurrent half over its own
//!     single cell.
//!
//! The port keeps ONE live recurrent cell (one row per recurrent layer =
//! the sequence's current conv/ssm state — `DecodeContext::recurrent_seq`),
//! so the blob's cell list is exactly that cell, and a restore into a fresh
//! context rewrites the rows verbatim: the continuation after the restore
//! must be bit-identical to the uninterrupted run.
//!
//! The models are tests/arch_batch5_e2e.rs's (regenerate with its `#[ignore]`
//! writer or `parity/state_kinds_parity.sh`); the `#[ignore]`
//! `recurrent_state_dump_blobs` feeds the same script whose reference side is
//! parity/ref_state_kinds.c.

use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel};

const MAMBA2: &str = "/tmp/arch-batch5/mamba2-synth.gguf";
const JAMBA: &str = "/tmp/arch-batch5/jamba-synth.gguf";
const OUT_DIR: &str = "/tmp/arch-statekinds";

/// the fixed prefill of parity/ref_state_kinds.c
const TOKS: [i32; 16] = [3, 17, 42, 9, 21, 5, 8, 30, 11, 29, 2, 16, 4, 13, 25, 7];
const N_TAIL: usize = 8;

fn open_model(path: &str) -> LlamaModel {
    let gguf = Gguf::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e} — run the batch-5 synth writers"));
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: read-only mmap of a file this process reads
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

fn synth_attn(m: &LlamaModel, fa: bool, il: usize) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head_k: hp.n_embd_head_k(il) as i64,
        n_embd_head_v: hp.n_embd_head_v(il) as i64,
        n_rot: hp.n_rot(il) as i64,
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

/// the mamba2 driver — tools/llama-server's MAMBA|MAMBA2 arm (the shared
/// weight set + the mixer enum); `new_with` builds the 0-wide dummy KV rows
/// (`ForwardWeights::kv_dims`) that track the sequence's positions
fn mamba2_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let hp = &m.hparams;
    let attn = synth_attn(m, fa, 0); // pure-recurrent: eps only
    let n_trunk = hp.n_layer() as usize;
    let layers = m.layers[..n_trunk]
        .iter()
        .map(|l| llama::graph_arch::MambaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            mixer: llama::graph_arch::MambaLayerMixer::Mamba2(llama::graph_arch::Mamba2Mixer {
                ssm_in: l.ssm_in.unwrap(),
                ssm_conv1d: l.ssm_conv1d.unwrap(),
                ssm_conv1d_b: l.ssm_conv1d_b,
                ssm_dt_b: l.ssm_dt_b.unwrap(),
                ssm_a: l.ssm_a.unwrap(),
                ssm_d: l.ssm_d.unwrap(),
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out.unwrap(),
            }),
        })
        .collect();
    let weights = ForwardWeights::Mamba(
        llama::graph_arch::MambaModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers,
        },
        llama::graph_arch::MambaParams {
            d_conv: hp.ssm_d_conv as i64,
            d_inner: hp.ssm_d_inner as i64,
            d_state: hp.ssm_d_state as i64,
            dt_rank: hp.ssm_dt_rank as i64,
            n_group: hp.ssm_n_group as i64,
            ssm_dt_b_c_rms: hp.ssm_dt_b_c_rms,
            norm_eps: hp.f_norm_rms_eps,
            n_embd_r: hp.n_embd_r(),
            n_embd_s: hp.n_embd_s(),
        },
    );
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    // the ref probe's geometry: n_ctx 512, n_ubatch 512
    DecodeContext::new_with(gctx, weights, attn, 512, 8, 512)
}

/// the jamba driver — the hybrid: plain attn KV rows on the !is_recr layers
fn jamba_driver(m: &mut LlamaModel, fa: bool) -> DecodeContext {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = synth_attn(m, fa, (0..n_trunk).find(|&il| !hp.is_recr(il)).unwrap());
    let layers = m.layers[..n_trunk]
        .iter()
        .map(|l| llama::graph_arch::JambaLayerWeights {
            attn_norm: l.attn_norm.unwrap(),
            mamba: l.ssm_in.map(|ssm_in| llama::graph_arch::Mamba1Mixer {
                ssm_in,
                ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                ssm_conv1d_b: l.ssm_conv1d_b.expect("ssm_conv1d_b"),
                ssm_x: l.ssm_x.expect("ssm_x"),
                ssm_dt: l.ssm_dt.expect("ssm_dt"),
                ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                ssm_dt_norm: l.ssm_dt_norm,
                ssm_b_norm: l.ssm_b_norm,
                ssm_c_norm: l.ssm_c_norm,
                ssm_a: l.ssm_a.expect("ssm_a"),
                ssm_d: l.ssm_d.expect("ssm_d"),
                ssm_out: l.ssm_out.expect("ssm_out"),
            }),
            wq: l.wq,
            wk: l.wk,
            wv: l.wv,
            wq_b: l.wq_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wo: l.wo,
            ffn_norm: l.ffn_norm.unwrap(),
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_up_exps: l.ffn_up_exps,
        })
        .collect();
    let weights = ForwardWeights::Jamba(
        llama::graph_arch::JambaModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers,
        },
        llama::graph_arch::JambaParams {
            attn,
            n_embd: hp.n_embd as i64,
            is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
            d_conv: hp.ssm_d_conv as i64,
            d_inner: hp.ssm_d_inner as i64,
            d_state: hp.ssm_d_state as i64,
            dt_rank: hp.ssm_dt_rank as i64,
            norm_eps: hp.f_norm_rms_eps,
            n_expert: hp.n_expert as i64,
            n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il)).collect(),
            expert_weights_scale: hp.expert_weights_scale,
            n_embd_r: hp.n_embd_r(),
            n_embd_s: hp.n_embd_s(),
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
    Mamba2,
    Jamba,
}

impl Kind {
    fn path(&self) -> &'static str {
        match self {
            Kind::Mamba2 => MAMBA2,
            Kind::Jamba => JAMBA,
        }
    }
    fn arch_name(&self) -> &'static str {
        match self {
            Kind::Mamba2 => "mamba2",
            Kind::Jamba => "jamba",
        }
    }
    fn driver(&self, m: &mut LlamaModel, fa: bool) -> DecodeContext {
        match self {
            Kind::Mamba2 => mamba2_driver(m, fa),
            Kind::Jamba => jamba_driver(m, fa),
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
// the round-trip tests (the state_kinds_e2e.rs protocol)
// ---------------------------------------------------------------------------

/// serialize sequence 0 after prefill+tail, restore into a fresh context of
/// the same model, continue greedy — every step's logits must equal the
/// uninterrupted run's bit-for-bit. The blob carries the conv/ssm rows; a
/// dropped, mis-rowed or mis-sized half shifts the logits on the very first
/// continued step (the recurrence reads its cells every step).
#[test]
fn recurrent_state_round_trip() {
    for kind in [Kind::Mamba2, Kind::Jamba] {
        for fa in [true, false] {
            let mut m_a = open_model(kind.path());
            let mut a = kind.driver(&mut m_a, fa);
            let logits_a = prefill_and_tail(&mut a);
            // the checkpoint BEFORE the compared continuation
            let blob = a.state_seq_get_data(0, false);
            let size = a.state_seq_get_size(0, false);
            assert_eq!(
                blob.len(),
                size,
                "{}: state_seq_get_size mismatch",
                kind.arch_name()
            );
            assert!(
                blob.len() > 64,
                "{}: implausibly small blob",
                kind.arch_name()
            );
            // the pure-recurrent blob is the recurrent half alone: exactly
            // one cell, at the sequence's last position (the hybrid's blob
            // starts with the attn half — checked by the parity script)
            if let Kind::Mamba2 = kind {
                assert_eq!(
                    u32::from_le_bytes(blob[8..12].try_into().unwrap()),
                    1,
                    "{}: cell_count",
                    kind.arch_name()
                );
                assert_eq!(
                    i32::from_le_bytes(blob[12..16].try_into().unwrap()),
                    (TOKS.len() + N_TAIL - 1) as i32,
                    "{}: the cell pos must be the sequence's last position",
                    kind.arch_name()
                );
            }
            let cont_a = continue_greedy(&mut a, logits_a.clone(), 8);

            let mut m_b = open_model(kind.path());
            let mut b = kind.driver(&mut m_b, fa);
            b.state_seq_set_data(0, &blob, false)
                .unwrap_or_else(|e| panic!("{}: restore: {e}", kind.arch_name()));

            // the restored context's positions: the restore replays the cell
            // (the KV cells mirror it), so the continuation continues at
            // TOKS.len() + N_TAIL
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
/// seq(-1) serialization; `state_set_data` validates the arch and restores,
/// and the dummy-mode `state_get_size` matches (llama-context.cpp:3325-3357).
#[test]
fn recurrent_state_whole_context_round_trip() {
    for kind in [Kind::Mamba2, Kind::Jamba] {
        let mut m_a = open_model(kind.path());
        let mut a = kind.driver(&mut m_a, true);
        let logits_a = prefill_and_tail(&mut a);
        let arch = kind.arch_name();
        // the checkpoint BEFORE the compared continuation
        let blob = a.state_get_data(arch);
        let size = a.state_get_size(arch);
        assert_eq!(blob.len(), size, "{arch}: state_get_size mismatch");
        let cont_a = continue_greedy(&mut a, logits_a.clone(), 4);

        // the header: [u32 len][arch bytes] then the memory blob. The
        // pure-recurrent blob follows directly — its whole-cache meta
        // carries the seq id (state_write_meta :883/:888); the hybrid's
        // starts with the attn half (the parity script checks the layout)
        let len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        assert_eq!(&blob[4..4 + len], arch.as_bytes(), "{arch}: header arch");
        if kind.arch_name() == "mamba2" {
            // [cell_count 1][pos][n_seq_id 1][seq id 0] (state_write_meta
            // :883/:888 — the whole-cache save writes the seq ids)
            let m = 4 + len;
            assert_eq!(u32::from_le_bytes(blob[m..m + 4].try_into().unwrap()), 1);
            assert_eq!(
                i32::from_le_bytes(blob[m + 4..m + 8].try_into().unwrap()),
                (TOKS.len() + N_TAIL - 1) as i32
            );
            assert_eq!(
                u32::from_le_bytes(blob[m + 8..m + 12].try_into().unwrap()),
                1
            );
            assert_eq!(
                i32::from_le_bytes(blob[m + 12..m + 16].try_into().unwrap()),
                0
            );
        }

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

/// the empty sequence's blob — `cell_count` 0 still carries the per-layer
/// header pairs (state_write_data's loops run over the layers, only the row
/// payloads vanish, llama-memory-recurrent.cpp:897-990), and a full `seq_rm`
/// frees the cell (the rs_zero rule): the next decode must reproduce a fresh
/// context's logits exactly.
#[test]
fn recurrent_state_empty_and_rs_zero() {
    let mut m_a = open_model(MAMBA2);
    let mut a = mamba2_driver(&mut m_a, true);
    let hp_n_layer = 4usize; // the batch-5 mamba2 spec
    let n_embd_r = 576usize; // (d_conv-1)*(d_inner + 2*n_group*d_state)
    let n_embd_s = 2048usize; // d_state*d_inner

    // a fresh context: cell_count 0 + the framing + the header pairs (the
    // row payloads vanish with the empty cell list,
    // llama-memory-recurrent.cpp:897-990 — only the per-layer F32-type/
    // row-size headers remain, the loops run over the layers regardless)
    let blob0 = a.state_seq_get_data(0, false);
    let expect0 = 8 // io_magic + seq_id
        + 4 // cell_count = 0
        + 4 // s_trans
        + 4 // n_layer
        + hp_n_layer * (4 + 8) // r headers
        + hp_n_layer * (4 + 8); // s headers
    assert_eq!(
        blob0.len(),
        expect0,
        "the empty blob is the header skeleton"
    );
    assert_eq!(u32::from_le_bytes(blob0[8..12].try_into().unwrap()), 0);
    // the headers still pin the live geometry (n_layer + the row sizes)
    assert_eq!(u32::from_le_bytes(blob0[12..16].try_into().unwrap()), 0); // s_trans
    assert_eq!(
        u32::from_le_bytes(blob0[16..20].try_into().unwrap()),
        hp_n_layer as u32
    );
    assert_eq!(
        u64::from_le_bytes(blob0[24..32].try_into().unwrap()),
        (4 * n_embd_r) as u64,
        "the r header's row size = ggml_row_size(F32, n_embd_r)"
    );
    let s0 = 20 + hp_n_layer * 12; // the first s header
    assert_eq!(
        u64::from_le_bytes(blob0[s0 + 4..s0 + 12].try_into().unwrap()),
        (4 * n_embd_s) as u64,
        "the s header's row size = ggml_row_size(F32, n_embd_s)"
    );

    // decode, then a full seq_rm: the cell is freed (rs_zero) — the blob is
    // the empty shape again and the state restarts from zero
    let pos: Vec<i32> = (0..TOKS.len() as i32).collect();
    let logits = a.decode(&TOKS, &pos).expect("prefill").to_vec();
    let blob1 = a.state_seq_get_data(0, false);
    assert_eq!(u32::from_le_bytes(blob1[8..12].try_into().unwrap()), 1);
    a.seq_rm(0, -1, -1);
    let blob2 = a.state_seq_get_data(0, false);
    assert_eq!(
        blob2.len(),
        expect0,
        "seq_rm(-1,-1) frees the recurrent cell"
    );
    assert_eq!(u32::from_le_bytes(blob2[8..12].try_into().unwrap()), 0);

    // the rs_zero rule: the next decode matches a fresh context's bit for bit
    let logits_after = a.decode(&TOKS, &pos).expect("redecode").to_vec();
    for (i, (x, y)) in logits.iter().zip(&logits_after).enumerate() {
        assert_eq!(x, y, "rs_zero: redecode logit {i} diverged");
    }
    // (the live-state blob's revive round-trip is recurrent_state_round_trip;
    // here the pinned facts are the empty skeleton and the rs_zero redecode)
}

// ---------------------------------------------------------------------------
// the #[ignore] dumper for parity/state_kinds_parity.sh
// ---------------------------------------------------------------------------

/// `<tag>:SQST:<len><blob>` + `<tag>:FULL:<len><blob>` — the port halves of
/// ref_state_kinds.c's dumps (the mamba2 cells of the script's recurrent leg)
#[test]
#[ignore = "dumps /tmp/arch-statekinds for parity/state_kinds_parity.sh"]
fn recurrent_state_dump_blobs() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    for kind in [Kind::Mamba2, Kind::Jamba] {
        for tail in [false, true] {
            let mut m = open_model(kind.path());
            let mut d = kind.driver(&mut m, true);
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
