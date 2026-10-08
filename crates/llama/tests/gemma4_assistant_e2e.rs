//! gemma4_assistant_e2e.rs — the mem-shared draft-mtp parity of the
//! gemma4-assistant head (`src/models/gemma4-assistant.cpp`) against the
//! pinned reference LIBRARY, on the real local pair:
//!
//!   * trunk  : gemma-4-26B-A4B-it-QAT-Q4_0.gguf (n_embd 2816 == the head's
//!     `embedding_length_out`)
//!   * head   : the AtomicChat gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf,
//!     re-pinned onto the revision's naming by
//!     `parity/gguf_pin_gemma4_assistant.py` (arch `gemma4-assistant`,
//!     `embedding_length_out`, `nextn_predict_layers`, the
//!     `nextn.{pre,post}_projection.weight` tensor names)
//!
//! The reference's own driver cannot reach the head (the pinned
//! common/speculative.cpp:2562 loads the *target* path as the draft model —
//! see PARITY.md 批次 15 §gemma4-assistant), so the truth comes from
//! `parity/ref_gemma4_assistant_dump.c`: it pairs the contexts by hand
//! (`cparams.ctx_other = ctx_tgt`, llama-context.cpp:147-153) and replays the
//! draft-mtp step shape (speculative.cpp:1602-1751) — seed (id_last,
//! pending_h) at pos0, decode, argmax, feed (id, h_row) at the SAME pos0
//! (the mem-shared rule, :1718-1722).
//!
//! `gemma4_assistant_reference_parity` (ignored; release mode, minutes)
//! compares, bit-exactly:
//!   * the trunk prefill's last `h_nextn` row (the draft's first `inp_h`),
//!   * every draft step's `t_logits` row (n_vocab wide) and `t_h_nextn` row.
//!
//! Run:
//!   python3 parity/gguf_pin_gemma4_assistant.py <assistant.gguf> /tmp/gemma4-assistant-pinned.gguf
//!   ./parity/ref_gemma4_assistant_dump <trunk.gguf> /tmp/gemma4-assistant-pinned.gguf \
//!       /tmp/g4asst-ref-faoff.bin --steps 5 --fa off
//!   cargo test --release -p llama --test gemma4_assistant_e2e -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use ggml::{Context, Gguf};
use llama::batch::LlamaBatch;
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch::{Gemma4LayerWeights, Gemma4ModelWeights, Gemma4Params};
use llama::model::{load_model, LlamaModel};
use llama::vocab::Vocab;
use memmap2::Mmap;

const GEMMA4_26B: &str =
    "/home/jeffrey/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf";
/// the pinned-naming assistant copy (parity/gguf_pin_gemma4_assistant.py)
const ASSISTANT_PINNED: &str = "/tmp/gemma4-assistant-pinned.gguf";
/// the reference dump (parity/ref_gemma4_assistant_dump)
const REF_DUMP: &str = "/tmp/g4asst-ref-faoff.bin";

// ---------------------------------------------------------------------------
// the gemma4 trunk driver (the gemma4_e2e.rs helpers, inlined)
// ---------------------------------------------------------------------------

fn gemma4_params(m: &LlamaModel, fa: bool) -> Gemma4Params {
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
            use_flash_attn: fa,
        },
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: 1.0,
        f_attention_scale: 1.0,
        f_final_logit_softcapping: hp.f_final_logit_softcapping,
        n_embd_per_layer: hp.n_embd_per_layer as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: (0..n_layer).map(|il| hp.n_ff_exp(il) as u32).collect(),
    }
}

fn gemma4_weights(m: &LlamaModel) -> Gemma4ModelWeights {
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
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
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

// ---------------------------------------------------------------------------
// the reference dump
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct RefDump {
    prompt: Vec<i32>,
    n_steps: usize,
    n_vocab: usize,
    n_embd: usize,
    trunk_h_last: Vec<f32>,
    /// per step: (fed_token, logits row, h_next row)
    steps: Vec<(i32, Vec<f32>, Vec<f32>)>,
}

fn parse_ref(path: &str) -> RefDump {
    let b = std::fs::read(path).expect("read the reference dump");
    assert_eq!(&b[..8], b"G4ASST01", "magic");
    let mut off = 8usize;
    let rd_u32 = |off: &mut usize| {
        let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v
    };
    let n_prompt = rd_u32(&mut off) as usize;
    let prompt: Vec<i32> = (0..n_prompt)
        .map(|_| {
            let v = i32::from_le_bytes(b[off..off + 4].try_into().unwrap());
            off += 4;
            v
        })
        .collect();
    let n_steps = rd_u32(&mut off) as usize;
    let n_vocab = rd_u32(&mut off) as usize;
    let n_embd = rd_u32(&mut off) as usize;
    let rd_f32s = |off: &mut usize, n: usize| -> Vec<f32> {
        let v = b[*off..*off + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        *off += 4 * n;
        v
    };
    let trunk_h_last = rd_f32s(&mut off, n_embd);
    let mut steps = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let fed = i32::from_le_bytes(b[off..off + 4].try_into().unwrap());
        off += 4;
        let logits = rd_f32s(&mut off, n_vocab);
        let h = rd_f32s(&mut off, n_embd);
        steps.push((fed, logits, h));
    }
    assert_eq!(off, b.len(), "trailing bytes");
    RefDump {
        prompt,
        n_steps,
        n_vocab,
        n_embd,
        trunk_h_last,
        steps,
    }
}

fn argmax(v: &[f32]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as i32
}

// ---------------------------------------------------------------------------
// the parity test
// ---------------------------------------------------------------------------

/// The mem-shared draft-mtp parity on the real pair — see the module docs.
/// `G4ASST_FA=1` flips the attention branch (the reference dump must match).
#[test]
#[ignore = "real-model parity: needs the 14 GiB trunk + /tmp/gemma4-assistant-pinned.gguf + /tmp/g4asst-ref-faoff.bin (see the module docs)"]
fn gemma4_assistant_reference_parity() {
    for p in [GEMMA4_26B, ASSISTANT_PINNED, REF_DUMP] {
        assert!(Path::new(p).exists(), "{p} missing (see the module docs)");
    }
    let fa = std::env::var("G4ASST_FA").is_ok();
    if std::env::var("G4ASST_NODES").is_ok() {
        g4asst_register_dump_cb();
    }
    let dump_path = std::env::var("G4ASST_REF").unwrap_or_else(|_| REF_DUMP.to_string());
    let r = parse_ref(&dump_path);

    // ---- the trunk context ----
    let file = std::fs::File::open(GEMMA4_26B).unwrap();
    let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
    let gguf = Gguf::from_bytes(mmap.clone()).expect("gguf parse");
    let mut model = load_model(&gguf, mmap.clone()).expect("trunk load");
    let p = gemma4_params(&model, fa);
    // the share map's target geometry (llama-model.cpp:2698-2703)
    let n = p.is_swa.len() - 1;
    let tgt_full = (p.n_embd_head_k[n] as i64, p.n_head_kv[n] as i64);
    let tgt_swa = (p.n_embd_head_k[n - 1] as i64, p.n_head_kv[n - 1] as i64);
    let w = gemma4_weights(&model);
    let gctx = std::mem::replace(&mut model.ctx, Context::new());
    let attn = p.attn;
    let spec = llama::kv_cache::SwaCacheSpec::from_hparams(&model.hparams);
    let mut tgt = DecodeContext::new_with_swa(
        gctx,
        ForwardWeights::Gemma4(w, p),
        attn,
        512,
        8,
        64,
        spec,
    );
    drop(model);

    // ---- the assistant head, attached to the target ----
    let afile = std::fs::File::open(ASSISTANT_PINNED).unwrap();
    let ammap = Arc::new(unsafe { Mmap::map(&afile).unwrap() });
    let agguf = Gguf::from_bytes(ammap.clone()).expect("assistant gguf parse");
    let _vocab_dft = Vocab::load(&agguf).expect("assistant vocab");
    tgt.attach_gemma4_assistant(&agguf, ammap, fa, tgt_full, tgt_swa)
        .expect("attach the head");

    // the driver's taps (speculative.cpp:1420-1421)
    tgt.set_embeddings_nextn(true, false);

    // ---- the trunk prefill (every position an output row, like the probe;
    // decode_batch routes through step_ubatch so the nextn tap extracts) ----
    let n = r.prompt.len();
    let logits_all = {
        let mut batch = LlamaBatch::default();
        for (i, &t) in r.prompt.iter().enumerate() {
            batch.add(t, i as i32, &[0], true);
        }
        let out = tgt.decode_batch(&batch).expect("prefill");
        assert_eq!(out.logits.len(), r.n_vocab * n);
        out.logits
    };
    let last_logits = &logits_all[(n - 1) * r.n_vocab..n * r.n_vocab];
    let h_last = tgt.get_embeddings_nextn_ith((n - 1) as i32).to_vec();
    assert_eq!(h_last.len(), r.n_embd, "the backbone width");
    let mut n_hdiff = 0usize;
    for (a, b) in h_last.iter().zip(&r.trunk_h_last) {
        if a.to_bits() != b.to_bits() {
            n_hdiff += 1;
        }
    }
    println!(
        "g4asst: trunk prefill h_last bit-diff {n_hdiff}/{} (max|d| {:.3e})",
        r.n_embd,
        h_last
            .iter()
            .zip(&r.trunk_h_last)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    );
    assert_eq!(n_hdiff, 0, "the trunk's h_nextn tap diverges from the reference");

    // id_last = the trunk's greedy pick — must match the probe's first feed
    let mut id_last = argmax(last_logits);
    let pos0 = n as i32;

    // ---- the draft loop (the mem-shared rule: every step at pos0) ----
    let mut h_row = h_last;
    let dump_nodes = std::env::var("G4ASST_NODES").ok();
    for (s, (ref_fed, ref_logits, ref_h)) in r.steps.iter().enumerate() {
        let _ = &dump_nodes;
        assert_eq!(*ref_fed, id_last, "step {s}: the fed token diverged");
        let mut batch = LlamaBatch::default();
        batch.add(id_last, pos0, &[0], true);
        batch.embd = Some(h_row.clone());
        // the optional node-level dump of this step (the mirror of the
        // probe's --nodes stream) — arm the callback around exactly this
        // decode
        if let Some(path) = dump_nodes.as_ref() {
            G4_DUMP_ARM.store(true, std::sync::atomic::Ordering::Relaxed);
            G4_DUMP_PATH
                .lock()
                .unwrap()
                .replace(std::fs::File::create(path).unwrap());
        }
        let step = tgt.decode_gemma4_assistant(&batch).expect("draft step");
        G4_DUMP_ARM.store(false, std::sync::atomic::Ordering::Relaxed);
        let _ = G4_DUMP_PATH.lock().unwrap().take();
        assert_eq!(step.logits.len(), r.n_vocab);
        assert_eq!(step.h_next.len(), r.n_embd);

        let mut n_ldiff = 0usize;
        let mut md = 0.0f32;
        for (a, b) in step.logits.iter().zip(ref_logits) {
            if a.to_bits() != b.to_bits() {
                n_ldiff += 1;
                md = md.max((a - b).abs());
            }
        }
        let mut n_hdiff2 = 0usize;
        for (a, b) in step.h_next.iter().zip(ref_h) {
            if a.to_bits() != b.to_bits() {
                n_hdiff2 += 1;
            }
        }
        println!(
            "g4asst: step {s} fed {id_last}: logits bit-diff {n_ldiff}/{} (max|d| {md:.3e}), \
             h_next bit-diff {n_hdiff2}/{}",
            r.n_vocab, r.n_embd
        );
        assert_eq!(n_ldiff, 0, "step {s}: the draft logits diverge from the reference");
        assert_eq!(n_hdiff2, 0, "step {s}: the draft h_next diverges");

        id_last = argmax(&step.logits);
        h_row = step.h_next;
    }
    println!(
        "g4asst: {} draft steps bit-exact against the reference library (fa {})",
        r.steps.len(),
        if fa { "on" } else { "off" }
    );
}


// ---------------------------------------------------------------------------
// the optional node dump (G4ASST_NODES) — the mirror of the C probe's
// --nodes stream: every node of the armed draft step, DECDMP1-style records
// ---------------------------------------------------------------------------

use ggml::compute::{set_eval_callback, EvalNode};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

static G4_DUMP_ARM: AtomicBool = AtomicBool::new(false);
static G4_DUMP_PATH: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

fn g4asst_dump_cb(node: &EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true;
    }
    if !G4_DUMP_ARM.load(Ordering::Relaxed) {
        return true;
    }
    let mut guard = G4_DUMP_PATH.lock().unwrap();
    let Some(f) = guard.as_mut() else {
        return true;
    };
    let mut put = |s: &str| {
        let l = s.len().min(255) as u8;
        let _ = f.write_all(&[l]);
        let _ = f.write_all(&s.as_bytes()[..l as usize]);
    };
    put(op_desc(node.op));
    put(node.name);
    put(type_desc(node.ty));
    let _ = f.write_all(bytemuck::cast_slice(&node.ne));
    let n: i64 = node.ne.iter().product();
    let _ = f.write_all(bytemuck::cast_slice(&[n]));
    if n as u64 >= (1 << 19) || !matches!(node.ty, ggml::types::GgmlType::F32) {
        return true;
    }
    let data = node.data.unwrap_or(&[]);
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v = f32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]);
        let _ = f.write_all(&v.to_le_bytes());
    }
    true
}

/// register the dump callback for the whole run (harmless when never armed)
pub fn g4asst_register_dump_cb() {
    set_eval_callback(Some(g4asst_dump_cb));
}


fn op_desc(op: ggml::GgmlOp) -> &'static str {
    use ggml::GgmlOp::*;
    // base names only: the port encodes RMS_NORM as Norm+flag and the unary
    // variants (SILU/GELU/...) as Silu+params — the comparator aligns by
    // index/shape/name, the op string is for display
    match op {
        None => "NONE",
        Dup => "DUP",
        Add => "ADD",
        Mul => "MUL",
        Div => "DIV",
        Sub => "SUB",
        Norm => "NORM",
        SquaredMulMat => "MUL_MAT_SQ",
        MulMat => "MUL_MAT",
        Scale => "SCALE",
        Cpy => "CPY",
        Reshape => "RESHAPE",
        View => "VIEW",
        Permute => "PERMUTE",
        Transpose => "TRANSPOSE",
        GetRows => "GET_ROWS",
        DiagMaskInf => "DIAG_MASK_INF",
        SoftMax => "SOFT_MAX",
        RoPE => "ROPE",
        RoPEBack => "ROPE_BACK",
        MulMatId => "MUL_MAT_ID",
        Argsort => "ARGSORT",
        ArgMax => "ARGMAX",
        Repeat => "REPEAT",
        Concat => "CONCAT",
        Silu => "SILU",
        SumRows => "SUM_ROWS",
        MulView => "MUL_VIEW",
        SetRows => "SET_ROWS",
        FlashAttnExt => "FLASH_ATTN_EXT",
        AddId => "ADD_ID",
        Glu => "GLU",
        SsmConv => "SSM_CONV",
        SsmScan => "SSM_SCAN",
        Clamp => "CLAMP",
        Gdn => "GATED_DELTA_NET",
        Im2col => "IM2COL",
        Upscale => "UPSCALE",
        Fill => "FILL",
        LightningIndexer => "LIGHTNING_INDEXER",
        TopK => "TOP_K",
        // arch batch 7 (deepseek4) ops — the deepseek4 dump lives in
        // arch_batch7_e2e.rs with the full UNARY/GLU-aware op_desc
        Sqrt => "SQRT",
        Dsv4HcComb => "dsv4_hc_comb(mixes, scale, base)",
        Dsv4HcPre => "dsv4_hc_pre(x, weights)",
        Dsv4HcPost => "dsv4_hc_post(x, residual, post, comb)",
        // arch batch 10 (graniteswitch) — the router lane's right-pad
        Pad => "PAD",
        // display-only arms for enum variants added after this file (neither
        // op appears in a qwen3/gpt-oss graph — exhaustiveness only)
        Pool2d => "POOL_2D",
        Arange => "ARANGE",
        Pool1d => "POOL_1D",
        Roll => "ROLL",
        Conv2dDirect => "CONV_2D_DIRECT",
        Conv2dDw => "CONV_2D_DW",
        // audio/mean rounds' later variants — display only, exhaustiveness
        Sin => "SIN",
        Cos => "COS",
        Sqr => "SQR",
        Mean => "MEAN",
        PadReflect1d => "PAD_REFLECT_1D",
        Sum => "SUM",
        Cumsum => "CUMSUM",
        Tri => "TRI",
        Log => "LOG",
        Col2im1d => "COL2IM_1D",
    }
}

fn type_desc(ty: ggml::types::GgmlType) -> &'static str {
    match ty {
        ggml::types::GgmlType::F32 => "f32",
        ggml::types::GgmlType::F16 => "f16",
        ggml::types::GgmlType::Bf16 => "bf16",
        ggml::types::GgmlType::I64 => "i64",
        ggml::types::GgmlType::I32 => "i32",
        ggml::types::GgmlType::I16 => "i16",
        ggml::types::GgmlType::I8 => "i8",
        ggml::types::GgmlType::Q8_0 => "q8_0",
        ggml::types::GgmlType::Q4_0 => "q4_0",
        ggml::types::GgmlType::Q4_1 => "q4_1",
        ggml::types::GgmlType::Q5_0 => "q5_0",
        ggml::types::GgmlType::Q5_1 => "q5_1",
        ggml::types::GgmlType::Q8_1 => "q8_1",
        _ => "other",
    }
}
