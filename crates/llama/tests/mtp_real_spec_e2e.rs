//! mtp_real_spec_e2e.rs — batch 20: the deferred **real MTP model** e2e of
//! the `--spec-type draft-mtp` driver (speculative.cpp:2545-2589's target-file
//! arm) over the local qwen35-arch 27B with nextn/MTP tensors
//! (`Qwen3.8-27B`, `qwen35.nextn_predict_layers = 1`).
//!
//! Three cells (all file/memory-gated — they SKIP when the model is absent):
//!   1. plain greedy 16 == the NEW reference's fresh-server first request
//!      (`--fa off`, parity's anchor convention),
//!   2. `--spec-type draft-mtp` (n_max 3) **with the recurrent rollback ring
//!      armed** (`DecodeContext::with_rs_rollback(3)` — the reference's
//!      `cparams.n_rs_seq = speculative.need_n_rs_seq()` = draft.n_max,
//!      common.h:396-404 + common.cpp:1635) must commit exactly the plain
//!      greedy stream — the speculative invariant,
//!   3. (#[ignore], diagnostic) the same spec run **without** the ring: the
//!      port CLI's current construction (n_rs_seq = 0) — printed, not
//!      asserted; this is the documented divergence of PARITY.md 批次 20.
//!
//! Root cause the ring fixes (batch 20, real-model bisect): qwen35 is a
//! hybrid (48 of 64 layers are gated-delta-net). The driver's verify batch
//! advances the recurrent state over the *draft* tokens too, and after a
//! rejected round the reference's `seq_rm` arms the per-token snapshot
//! rollback (llama-memory-recurrent.cpp:193-210, `set_rs_idx`) so the next
//! round restarts from the last accepted token's state. Without the ring
//! (`n_rs_seq = 0`) the port leaves the GDN state after the *rejected*
//! drafts — logits drift by whole units (measured: row-0 top logits moved
//! 18.85→18.04 with a 3-token pollution) and the greedy stream flips inside
//! the model's near-tie bands (token 7 of 16 at n_max 3, token 14 at n_max 1).
//! The synthetic nine-head cell (tests/mtp2_e2e.rs) saw the same mechanism
//! as a single "near-tie numeric-tail" flip at step 14 — this file's real
//! model sizes it as a structural driver omission, not GEMM noise.

use std::path::Path;
use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::arch::LlmArch;
use llama::context::{DecodeContext, ForwardWeights, MtpForward, MtpHeadFacts};
use llama::graph_arch::{
    Qwen35LayerWeights, Qwen35ModelWeights, Qwen35MtpWeights, Qwen35Params, MtpNextn,
};
use llama::graph::AttnParams;
use llama::model::{load_model, LlamaModel, LayerTensors};
use llama::sampling::{SamplingContext, SamplingParams};
use llama::speculative::{
    common_speculative_init, speculative_simple_generate, CommonParamsSpeculative,
    CommonSpeculativeType,
};
use llama::vocab::Vocab;
use memmap2::Mmap;

/// the local qwen35-arch file with nextn/MTP tensors (`qwen35.nextn_predict_
/// layers` — Qwen3.8-27B; the plain Qwen3.6-27B has no nextn block)
const QWEN35_27B_MTP: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf";

const PROMPT: &str = "The capital of France is";
const N_PREDICT: usize = 16;

/// fresh NEW-reference llama-server (`--flash-attn off --spec-type
/// draft-mtp`), first request, greedy 16 — measured 2026-10-02; the spec run
/// and the plain run of the reference agree on it (the invariant holds on the
/// reference side).
#[rustfmt::skip]
const REF16_QWEN38_27B_FA_OFF: [i32; 16] = [
    11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 14898, 369,
];

// ---------------------------------------------------------------------------
// the qwen35 builders (the copies tests/qwen35_e2e.rs and mtp2_e2e.rs carry)
// ---------------------------------------------------------------------------

struct Loaded {
    model: LlamaModel,
    gguf: Gguf,
    _mmap: Arc<Mmap>,
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
        Ok(model) => Some(Loaded {
            model,
            gguf,
            _mmap: mmap,
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

/// tests/qwen35_e2e.rs's `qwen35_params` — per-layer head geometry + IMROPE
/// sections + the GDN cell sizes (qwen35.cpp:165-268)
fn qwen35_params(m: &LlamaModel) -> Qwen35Params {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
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

fn qwen35_layer(l: &LayerTensors) -> Qwen35LayerWeights {
    Qwen35LayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        attn_post_norm: l.attn_post_norm.expect("attn_post_norm"),
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
        ffn_gate: l.ffn_gate.expect("ffn_gate"),
        ffn_up: l.ffn_up.expect("ffn_up"),
        ffn_down: l.ffn_down.expect("ffn_down"),
    }
}

fn qwen35_weights(m: &LlamaModel) -> Qwen35ModelWeights {
    let n_trunk = m.hparams.n_layer() as usize;
    Qwen35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        cls_out: m.cls_out,
        cls_out_b: m.cls_out_b,
        layers: m.layers[..n_trunk].iter().map(qwen35_layer).collect(),
    }
}

/// the MTP layer's `MtpHeadFacts` (the CLI's `mtp_head_facts` — the hparams
/// reads at il = n_layer)
fn mtp_head_facts(hp: &llama::hparams::LlamaHparams) -> MtpHeadFacts {
    let il = hp.n_layer() as usize;
    MtpHeadFacts {
        n_embd: hp.n_embd_out() as i64,
        k_row: hp.n_embd_k_gqa(il) as i64,
        v_row: hp.n_embd_v_gqa(il) as i64,
        n_swa: hp.n_swa,
        swa_type: hp.swa_type,
        is_swa: hp.is_swa(il),
    }
}

fn mtp_nextn9_of(l: &LayerTensors) -> MtpNextn {
    let n = &l.nextn;
    MtpNextn {
        eh_proj: n.eh_proj.expect("nextn.eh_proj"),
        enorm: n.enorm.expect("nextn.enorm"),
        hnorm: n.hnorm.expect("nextn.hnorm"),
        embed_tokens: n.embed_tokens,
        shared_head_head: n.shared_head_head,
        shared_head_norm: n.shared_head_norm,
    }
}

/// qwen35.cpp:288-320/519-644 — the MTP layer is a full-attention
/// trunk-shaped block (the CLI's `qwen35_mtp_weights`)
fn qwen35_mtp_weights(m: &LlamaModel) -> Qwen35MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    Qwen35MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: qwen35_layer(&m.layers[il]),
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head: hp.n_embd_head_k(il) as i64,
        n_rot: hp.n_rot(il) as i32,
    }
}

// ---------------------------------------------------------------------------
// the drivers
// ---------------------------------------------------------------------------

const N_CTX: u32 = 512;
const THREADS: usize = 8;
const N_BATCH: usize = 512;

/// the trunk context — `rs_seq = 0` (the plain CLI construction) unless
/// `rs_rollback` arms the ring
fn trunk_dctx(m: &mut LlamaModel, rs_seq: u32) -> DecodeContext {
    let p = qwen35_params(m);
    let w = qwen35_weights(m);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_with(
        gctx,
        ForwardWeights::Qwen35(w, p.clone()),
        p.attn,
        N_CTX,
        THREADS,
        N_BATCH,
    )
    .with_rs_rollback(rs_seq)
}

/// the MTP draft context (`LLAMA_CONTEXT_TYPE_MTP`, speculative.cpp:2545-2589)
fn mtp_dctx(m: &mut LlamaModel, rs_seq: u32) -> DecodeContext {
    let p = qwen35_params(m);
    let facts = mtp_head_facts(&m.hparams);
    let mtp = MtpForward::Qwen35(qwen35_mtp_weights(m), p.clone(), facts);
    let w = qwen35_weights(m);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_mtp(
        gctx,
        ForwardWeights::Qwen35(w, p.clone()),
        mtp,
        p.attn,
        N_CTX,
        THREADS,
        N_BATCH,
    )
    .with_rs_rollback(rs_seq)
}

fn greedy16(dctx: &mut DecodeContext, prompt: &[i32]) -> Vec<i32> {
    let pos: Vec<i32> = (0..prompt.len() as i32).collect();
    let mut cur = dctx.decode(prompt, &pos).expect("prefill").to_vec();
    let mut out = Vec::new();
    let mut p = prompt.len() as i32;
    for _ in 0..N_PREDICT {
        let id = cur
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap();
        out.push(id);
        cur = dctx.decode(&[id], &[p]).expect("decode").to_vec();
        p += 1;
    }
    out
}

/// one `--spec-type draft-mtp` run (the CLI's construction path,
/// speculative.cpp:2545-2589 + the speculative-simple driver)
fn spec_run(rs_seq: u32, n_max: i32) -> (Vec<i32>, llama::speculative::SpeculativeSimpleResult) {
    let mut m_tgt = load_real(QWEN35_27B_MTP).expect("target model");
    let mut tgt = trunk_dctx(&mut m_tgt.model, rs_seq);
    let mut m_dft = load_real(QWEN35_27B_MTP).expect("draft model");
    let ctx_dft = mtp_dctx(&mut m_dft.model, rs_seq);
    let vocab = Vocab::load(&Gguf::open(QWEN35_27B_MTP).unwrap()).unwrap();
    let prompt = vocab.tokenize(PROMPT, true, false);

    let mut params = CommonParamsSpeculative::default();
    params.types = vec![CommonSpeculativeType::DraftMtp];
    params.draft.n_max = n_max;
    params.draft.p_min = 0.0;
    let mut spec = common_speculative_init(
        &params,
        1,
        &mut tgt,
        Some(ctx_dft),
        &vocab,
        Some(&vocab),
        m_tgt.model.hparams.n_layer_nextn,
        false,
    )
    .expect("init")
    .expect("speculator");
    let mut smpl = SamplingContext::new(
        tgt.n_vocab() as i32,
        SamplingParams {
            temp: 0.0,
            ..Default::default()
        },
    );
    let res = speculative_simple_generate(&mut tgt, &mut spec, &mut smpl, &vocab, &prompt, N_PREDICT as i32)
        .expect("spec generate");
    (res.tokens[..N_PREDICT.min(res.tokens.len())].to_vec(), res)
}

// ---------------------------------------------------------------------------
// the cells
// ---------------------------------------------------------------------------

/// 1. the plain greedy trunk on the real MTP file == the reference's 16
#[test]
fn qwen38_27b_mtp_plain_greedy_16() {
    let Some(mut l) = load_real(QWEN35_27B_MTP) else {
        return;
    };
    if mem_available_gb() < 24.0 {
        eprintln!("SKIP: need >24G, have {:.1}G", mem_available_gb());
        return;
    }
    assert_eq!(l.model.arch, LlmArch::QWEN35);
    assert_eq!(l.model.hparams.n_layer_nextn, 1, "the nextn block");
    let vocab = Vocab::load(&l.gguf).unwrap();
    let prompt = vocab.tokenize(PROMPT, true, false);
    assert_eq!(prompt, vec![760, 6511, 314, 9338, 369]);
    let mut dctx = trunk_dctx(&mut l.model, 0);
    let out = greedy16(&mut dctx, &prompt);
    println!("plain greedy 16: {out:?}");
    assert_eq!(out[..], REF16_QWEN38_27B_FA_OFF[..]);
}

/// 2. `--spec-type draft-mtp` with the rollback ring armed (n_rs_seq =
/// n_max, the reference's `need_n_rs_seq()`) commits the plain greedy stream
#[test]
fn qwen38_27b_mtp_spec_greedy_invariant_with_rollback() {
    if !Path::new(QWEN35_27B_MTP).exists() {
        eprintln!("SKIP: {QWEN35_27B_MTP} not present");
        return;
    }
    if mem_available_gb() < 40.0 {
        // the double load peaks ~32G (2 x 16.5G)
        eprintln!("SKIP: need >40G, have {:.1}G", mem_available_gb());
        return;
    }
    let (spec16, res) = spec_run(3, 3);
    println!("spec (rs=3, n_max=3) 16: {spec16:?}");
    println!(
        "spec counters: n_drafted {} n_accept {} ({}%)",
        res.n_drafted,
        res.n_accept,
        100.0 * res.n_accept as f64 / res.n_drafted.max(1) as f64
    );
    assert_eq!(spec16[..], REF16_QWEN38_27B_FA_OFF[..]);
}

/// 3. the diagnostic: the CLI's current construction (rs_seq = 0) — the
/// documented divergence (run manually; not asserted)
#[test]
#[ignore = "manual: prints the un-armed (n_rs_seq=0) spec stream — the documented batch-20 divergence"]
fn qwen38_27b_mtp_spec_no_rollback_diagnostic() {
    if !Path::new(QWEN35_27B_MTP).exists() {
        eprintln!("SKIP: {QWEN35_27B_MTP} not present");
        return;
    }
    if mem_available_gb() < 40.0 {
        eprintln!("SKIP: need >40G, have {:.1}G", mem_available_gb());
        return;
    }
    let (spec16, res) = spec_run(0, 3);
    println!("spec (rs=0, n_max=3) 16: {spec16:?}");
    println!(
        "spec counters: n_drafted {} n_accept {} ({}%)",
        res.n_drafted,
        res.n_accept,
        100.0 * res.n_accept as f64 / res.n_drafted.max(1) as f64
    );
    println!("reference greedy 16: {:?}", REF16_QWEN38_27B_FA_OFF);
}
