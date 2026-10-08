//! tts_archs_e2e.rs — audio round 5: the LM-side TTS archs
//! (models/{wavtokenizer-dec,pockettts,qwen3tts}.cpp, the last three open
//! ⛔ rows of parity/AUDIT_models.md).
//!
//! Synthetic protocol (the established one): the PORT's GGUF writer builds
//! one file per arch, the port loads it through `load_model` and runs the
//! graph_arch builder over a fixed token sequence (a manual DecodeInputs /
//! KvCache driver for pockettts — the causal decoder — and a direct
//! embd-graph run for wavtokenizer-dec), then the same files go to the
//! pinned reference via `parity/tts_parity.sh` (a `ref_lm_logits_dump`
//! probe over libllama) and the final logits / t_embd are compared at the
//! numeric band.
//!
//! No real TTS GGUF exists on this machine (checked
//! /home/jeffrey/.lmstudio/models and /home/jeffrey/localai/models — only
//! piper onnx + a tts.yaml), so the synthetic protocol is the designed
//! answer, exactly like rounds 1-4 of the mtmd audio family.

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, Value};
use llama::arch::LlmArch;
use llama::graph::{fill_causal_mask, AttnParams, DecodeInputs};
use llama::graph_arch;
use llama::kv_cache::{KvCache, SlotInfo};
use std::sync::Arc as _;
use llama::model::{load_model, LlamaModel};
use memmap2::Mmap;
use std::sync::Arc;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const OUT_DIR: &str = "/tmp/tts-lm-synth";

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 8) as f32 / 16_777_216.0 - 0.5
}

fn tensor_bytes(n: usize, state: &mut u32) -> Vec<u8> {
    (0..n)
        .map(|_| lcg(state).to_le_bytes())
        .flat_map(|b| b.to_vec())
        .collect()
}

struct W {
    w: GgufWriter,
    datas: Vec<Vec<u8>>,
    st: u32,
}

impl W {
    fn new(arch: &str) -> Self {
        let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
        let mut w = GgufWriter::new(32);
        for (k, val) in &src.kv {
            if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
                w.set_kv(k, val.clone());
            }
        }
        w.set_kv("general.architecture", Value::String(arch.to_string()));
        w.set_kv(
            "general.name",
            Value::String(format!("llama-rust-synth-{arch}")),
        );
        w.set_kv("general.file_type", Value::U32(0)); // F32
        Self {
            w,
            datas: Vec::new(),
            st: 0x1234_5678,
        }
    }
    fn kv(&mut self, k: &str, v: Value) {
        self.w.set_kv(k, v);
    }
    fn add(&mut self, name: &str, ne: [i64; 4]) {
        self.w.add_tensor(name, GgmlType::F32, ne);
        let n = (ne[0] * ne[1] * ne[2] * ne[3]).max(0) as usize;
        self.datas.push(tensor_bytes(n, &mut self.st));
    }
    fn finish(self, path: &str) {
        let Self { w, datas, .. } = self;
        let (w, datas) = (w, datas);
        let tmp = format!("{path}.tmp{}", std::process::id());
        let f = std::fs::File::create(&tmp).expect("create synth gguf");
        let mut bw = std::io::BufWriter::new(f);
        let refs: Vec<&[u8]> = datas.iter().map(|d| d.as_slice()).collect();
        w.write(&mut bw, &refs).expect("write synth gguf");
        use std::io::Write as _;
        bw.flush().unwrap();
        std::fs::rename(&tmp, path).expect("publish synth gguf");
    }
}

// ---- pockettts geometry (models/pockettts.cpp) ----------------------------
const PT_N_LAYER: i64 = 2;
const PT_N_EMBD: i64 = 64;
const PT_N_HEAD: i64 = 4;
const PT_N_FF: i64 = 128;

fn write_pockettts() -> String {
    let path = format!("{OUT_DIR}/pockettts-synth.gguf");
    let mut w = W::new("pockettts");
    let a = "pockettts";
    w.kv(
        &format!("{a}.embedding_length"),
        Value::U32(PT_N_EMBD as u32),
    );
    w.kv(&format!("{a}.block_count"), Value::U32(PT_N_LAYER as u32));
    w.kv(&format!("{a}.feed_forward_length"), Value::U32(PT_N_FF as u32));
    w.kv(
        &format!("{a}.attention.head_count"),
        Value::U32(PT_N_HEAD as u32),
    );
    w.kv(&format!("{a}.attention.key_length"), Value::U32(16));
    w.kv(&format!("{a}.attention.value_length"), Value::U32(16));
    // pockettts.cpp:7 — LLM_KV_ATTENTION_LAYERNORM_EPS (the LLM_NORM eps)
    w.kv(
        &format!("{a}.attention.layer_norm_epsilon"),
        Value::F32(1e-5),
    );
    w.kv(&format!("{a}.context_length"), Value::U32(256));

    w.add("token_embd.weight", [PT_N_EMBD, 32000, 1, 1]);
    w.add("output_norm.weight", [PT_N_EMBD, 1, 1, 1]);
    w.add("output_norm.bias", [PT_N_EMBD, 1, 1, 1]);
    // no output head — the loader duplicates token_embd (pockettts.cpp:23-24)
    for il in 0..PT_N_LAYER {
        w.add(&format!("blk.{il}.attn_norm.weight"), [PT_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.attn_norm.bias"), [PT_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.attn_q.weight"), [PT_N_EMBD, PT_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.attn_k.weight"), [PT_N_EMBD, PT_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.attn_v.weight"), [PT_N_EMBD, PT_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.attn_output.weight"), [PT_N_EMBD, PT_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.ffn_norm.weight"), [PT_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.ffn_norm.bias"), [PT_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.ffn_down.weight"), [PT_N_FF, PT_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.ffn_up.weight"), [PT_N_EMBD, PT_N_FF, 1, 1]);
    }
    w.finish(&path);
    path
}

// ---- wavtokenizer-dec geometry (models/wavtokenizer-dec.cpp) ---------------
// features_length (=n_embd, the code embedding) 64, embedding_length
// (=n_embd_out, waveform samples per frame) 96, posnet/convnext width 64,
// 6 posnet blocks + 2 convnext blocks (block_count 6 covers both).
const WV_N_EMBD: i64 = 64;
const WV_N_OUT: i64 = 96;
const WV_N: i64 = 64;
const WV_N_FF: i64 = 96;

fn write_wavtokenizer() -> String {
    let path = format!("{OUT_DIR}/wavtokenizer-dec-synth.gguf");
    let mut w = W::new("wavtokenizer-dec");
    let a = "wavtokenizer-dec";
    w.kv(&format!("{a}.embedding_length"), Value::U32(WV_N_OUT as u32));
    w.kv(&format!("{a}.features_length"), Value::U32(WV_N_EMBD as u32));
    w.kv(&format!("{a}.block_count"), Value::U32(6));
    w.kv(&format!("{a}.feed_forward_length"), Value::U32(WV_N_FF as u32));
    // no attention.head_count: the wavtokenizer hparams read only the norm
    // trio (wavtokenizer-dec.cpp:3-7)
    w.kv(&format!("{a}.context_length"), Value::U32(256));
    w.kv(&format!("{a}.attention.layer_norm_epsilon"), Value::F32(1e-6));
    w.kv(&format!("{a}.attention.group_norm_epsilon"), Value::F32(1e-5));
    w.kv(&format!("{a}.attention.group_norm_groups"), Value::U32(8));
    w.kv(&format!("{a}.posnet.embedding_length"), Value::U32(WV_N as u32));
    w.kv(&format!("{a}.posnet.block_count"), Value::U32(6));
    w.kv(&format!("{a}.convnext.embedding_length"), Value::U32(WV_N as u32));
    w.kv(&format!("{a}.convnext.block_count"), Value::U32(2));

    w.add("token_embd.weight", [WV_N_EMBD, 32000, 1, 1]);
    w.add("conv1d.weight", [7, WV_N_EMBD, WV_N, 1]);
    w.add("conv1d.bias", [1, WV_N, 1, 1]);

    // posnet blocks 0/1/3/4 (resnet), 2 (attention), 5 (trailing norm)
    for il in 0..6i64 {
        match il {
            0 | 1 | 3 | 4 => {
                w.add(&format!("posnet.{il}.norm1.weight"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.norm1.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.conv1.weight"), [3, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.conv1.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.norm2.weight"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.norm2.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.conv2.weight"), [3, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.conv2.bias"), [1, WV_N, 1, 1]);
            }
            2 => {
                w.add(&format!("posnet.{il}.attn_norm.weight"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_norm.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_q.weight"), [1, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.attn_q.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_k.weight"), [1, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.attn_k.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_v.weight"), [1, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.attn_v.bias"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_output.weight"), [1, WV_N, WV_N, 1]);
                w.add(&format!("posnet.{il}.attn_output.bias"), [1, WV_N, 1, 1]);
            }
            _ => {
                // block 5 — the ATTN_NORM slot reuse (wavtokenizer-dec.cpp:70-71)
                w.add(&format!("posnet.{il}.attn_norm.weight"), [1, WV_N, 1, 1]);
                w.add(&format!("posnet.{il}.attn_norm.bias"), [1, WV_N, 1, 1]);
            }
        }
    }

    w.add("token_embd_norm.weight", [WV_N, 1, 1, 1]);
    w.add("token_embd_norm.bias", [WV_N, 1, 1, 1]);

    for il in 0..2i64 {
        w.add(&format!("convnext.{il}.dw.weight"), [7, 1, WV_N, 1]);
        w.add(&format!("convnext.{il}.dw.bias"), [1, WV_N, 1, 1]);
        w.add(&format!("convnext.{il}.norm.weight"), [WV_N, 1, 1, 1]);
        w.add(&format!("convnext.{il}.norm.bias"), [WV_N, 1, 1, 1]);
        w.add(&format!("convnext.{il}.pw1.weight"), [WV_N, WV_N_FF, 1, 1]);
        w.add(&format!("convnext.{il}.pw1.bias"), [WV_N_FF, 1, 1, 1]);
        w.add(&format!("convnext.{il}.pw2.weight"), [WV_N_FF, WV_N, 1, 1]);
        w.add(&format!("convnext.{il}.pw2.bias"), [WV_N, 1, 1, 1]);
        w.add(&format!("convnext.{il}.gamma.weight"), [WV_N, 1, 1, 1]);
    }

    w.add("output_norm.weight", [WV_N, 1, 1, 1]);
    w.add("output_norm.bias", [WV_N, 1, 1, 1]);
    w.add("output.weight", [WV_N, WV_N_OUT, 1, 1]);
    w.add("output.bias", [WV_N_OUT, 1, 1, 1]);
    w.finish(&path);
    path
}

// ---- qwen3tts geometry (a pure qwen3vl typedef, models.h:625-627) ----------
const Q3T_N_LAYER: i64 = 2;
const Q3T_N_EMBD: i64 = 64;
const Q3T_N_HEAD: i64 = 4;
const Q3T_N_HEAD_KV: i64 = 2;
const Q3T_KEY_LENGTH: i64 = 16;
const Q3T_N_FF: i64 = 48;

fn write_qwen3tts() -> String {
    let path = format!("{OUT_DIR}/qwen3tts-synth.gguf");
    let mut w = W::new("qwen3tts");
    let a = "qwen3tts";
    w.kv(&format!("{a}.embedding_length"), Value::U32(Q3T_N_EMBD as u32));
    w.kv(&format!("{a}.block_count"), Value::U32(Q3T_N_LAYER as u32));
    w.kv(&format!("{a}.feed_forward_length"), Value::U32(Q3T_N_FF as u32));
    w.kv(&format!("{a}.attention.head_count"), Value::U32(Q3T_N_HEAD as u32));
    w.kv(&format!("{a}.attention.head_count_kv"), Value::U32(Q3T_N_HEAD_KV as u32));
    w.kv(&format!("{a}.attention.key_length"), Value::U32(Q3T_KEY_LENGTH as u32));
    w.kv(&format!("{a}.attention.value_length"), Value::U32(Q3T_KEY_LENGTH as u32));
    w.kv(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-5),
    );
    w.kv(&format!("{a}.context_length"), Value::U32(256));
    w.kv(
        &format!("{a}.rope.dimension_sections"),
        Value::Array(ggml::GgufType::Uint32, vec![
            Value::U32(4),
            Value::U32(4),
            Value::U32(4),
            Value::U32(0),
        ]),
    );
    // qwen3vl.cpp:5 — NUM_DEEPSTACK_LAYERS, optional (0 here, text-only)

    w.add("token_embd.weight", [Q3T_N_EMBD, 32000, 1, 1]);
    w.add("output_norm.weight", [Q3T_N_EMBD, 1, 1, 1]);
    // [TAG_LLAMA_N_VOCAB_OUT] qwen3vl.cpp:19-23 — the qwen3tts head narrows
    // to the 3072 text tokens
    w.add("output.weight", [Q3T_N_EMBD, 3072, 1, 1]);
    for il in 0..Q3T_N_LAYER {
        w.add(&format!("blk.{il}.attn_norm.weight"), [Q3T_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.attn_q.weight"), [Q3T_N_EMBD, Q3T_KEY_LENGTH * Q3T_N_HEAD, 1, 1]);
        w.add(&format!("blk.{il}.attn_k.weight"), [Q3T_N_EMBD, Q3T_KEY_LENGTH * Q3T_N_HEAD_KV, 1, 1]);
        w.add(&format!("blk.{il}.attn_v.weight"), [Q3T_N_EMBD, Q3T_KEY_LENGTH * Q3T_N_HEAD_KV, 1, 1]);
        w.add(&format!("blk.{il}.attn_output.weight"), [Q3T_KEY_LENGTH * Q3T_N_HEAD, Q3T_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.attn_q_norm.weight"), [Q3T_KEY_LENGTH, 1, 1, 1]);
        w.add(&format!("blk.{il}.attn_k_norm.weight"), [Q3T_KEY_LENGTH, 1, 1, 1]);
        w.add(&format!("blk.{il}.ffn_norm.weight"), [Q3T_N_EMBD, 1, 1, 1]);
        w.add(&format!("blk.{il}.ffn_gate.weight"), [Q3T_N_EMBD, Q3T_N_FF, 1, 1]);
        w.add(&format!("blk.{il}.ffn_down.weight"), [Q3T_N_FF, Q3T_N_EMBD, 1, 1]);
        w.add(&format!("blk.{il}.ffn_up.weight"), [Q3T_N_EMBD, Q3T_N_FF, 1, 1]);
    }
    w.finish(&path);
    path
}

// ---------------------------------------------------------------------------
// loading + port graph runs
// ---------------------------------------------------------------------------

fn load(path: &str) -> Option<LlamaModel> {
    let gguf = Gguf::open(path).expect("open synth gguf");
    let file = std::fs::File::open(path).unwrap();
    let mmap = Arc::new(unsafe { Mmap::map(&file) }.unwrap());
    load_model(&gguf, mmap).map(Some).unwrap_or_else(|e| {
        eprintln!("load {path} failed: {e}");
        None
    })
}

fn pockettts_params(m: &LlamaModel, use_flash_attn: bool) -> AttnParams {
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
        norm_eps: hp.f_norm_eps,
        use_flash_attn,
    }
}

/// the fixed token sequence both sides decode (LLAMA_BATCH tokens of a
/// plain prompt — the reference probe uses llama_tokenize on the same text)
fn probe_tokens() -> Vec<i32> {
    (0..8).map(|i| 100 + i * 7).collect()
}

/// one manual decode over the pockettts builder (the qwen3_e2e Driver
/// protocol): KV assign → graph build → compute → last-row logits.
fn pockettts_logits(m: &mut LlamaModel, tokens: &[i32], use_flash_attn: bool) -> Vec<f32> {
    let w = m.pockettts_weights();
    let attn = pockettts_params(m, use_flash_attn);
    let hp = &m.hparams;
    let k_row = hp.n_embd_head_k(0) as i64 * hp.n_head_kv(0) as i64;
    let v_row = hp.n_embd_head_v(0) as i64 * hp.n_head_kv(0) as i64;
    let gctx = &mut m.ctx;
    let mut kv = KvCache::new(gctx, m.layers.len(), k_row, v_row, 256);
    // the watermark must sit AFTER the cache tensors (reset_graph_to would
    // otherwise drop them — the qwen3_e2e run_sequence order)
    let watermark = gctx.mark();

    let n = tokens.len();
    let sinfo = kv.find_slot(n as u32).expect("kv full");
    let pos: Vec<i32> = (0..n as i32).collect();
    kv.assign(sinfo, &pos, 0);
    let n_kv = kv.n_kv();

    gctx.reset_graph_to(watermark);
    let tokens_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
    let pos_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
    // FA requires an F16 mask, the non-FA path an F32 one
    // (llama-graph.cpp:38-39)
    let mask_ty = if use_flash_attn {
        GgmlType::F16
    } else {
        GgmlType::F32
    };
    let kq_mask = gctx.new_tensor_2d(mask_ty, n_kv as i64, n as i64);
    let row_idx = gctx.new_tensor_1d(GgmlType::I64, n as i64);
    for t in [tokens_t, pos_t, kq_mask, row_idx] {
        gctx.arena_resize_tensor(t);
    }
    gctx.with_i32_mut(tokens_t, |p| p.copy_from_slice(tokens)).unwrap();
    gctx.with_i32_mut(pos_t, |p| p.copy_from_slice(&pos)).unwrap();
    let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
    gctx
        .data_bytes_mut(row_idx)
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&idxs));
    {
        let mask_bytes = gctx.data_bytes_mut(kq_mask).unwrap();
        let kv_pos: Vec<i32> = kv.cells[..n_kv as usize].iter().map(|c| c.pos).collect();
        match mask_ty {
            GgmlType::F16 => {
                let mask: &mut [half::f16] = bytemuck::cast_slice_mut(mask_bytes);
                llama::graph::fill_causal_mask_f16(mask, &kv_pos, &pos);
            }
            _ => {
                let mask: &mut [f32] = bytemuck::cast_slice_mut(mask_bytes);
                fill_causal_mask(mask, &kv_pos, &pos);
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
    let result = graph_arch::build_pockettts_forward(
        gctx,
        &w,
        &attn,
        &kv,
        &inputs,
        SlotInfo {
            s0: sinfo.s0,
            s1: sinfo.s1,
        },
        n_kv,
        n,
    );
    let logits = result.logits;
    let mut gf = result.graph;
    ggml::compute::graph_compute(gctx, &mut gf, 4);

    let n_vocab = gctx.ne(logits)[0] as usize;
    let all: Vec<f32> = bytemuck::cast_slice(gctx.data_bytes(logits).unwrap()).to_vec();
    all[n_vocab * (n - 1)..n_vocab * n].to_vec()
}

/// the wavtokenizer-dec embd graph over the fixed tokens — returns
/// (logits_last_row, t_embd_last_frame) where t_embd is the decoded
/// waveform frame.
fn wavtokenizer_run(m: &mut LlamaModel, tokens: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let w = m.wavtokenizer_dec_weights();
    let p = graph_arch::WavtokenizerDecParams {
        norm_eps: m.hparams.f_norm_eps,
        norm_group_eps: m.hparams.f_norm_group_eps,
        n_norm_groups: m.hparams.n_norm_groups as i64,
        posnet_n_embd: m.hparams.posnet.n_embd as i64,
        n_ff: m.hparams.n_ff(0) as i64,
    };
    let gctx = &mut m.ctx;
    let watermark = gctx.mark();
    gctx.reset_graph_to(watermark);

    let n = tokens.len();
    let tokens_t = gctx.new_tensor_1d(GgmlType::I32, n as i64);
    gctx.arena_resize_tensor(tokens_t);
    gctx.with_i32_mut(tokens_t, |q| q.copy_from_slice(tokens)).unwrap();

    let result = graph_arch::build_wavtokenizer_dec_forward(gctx, &w, &p, tokens_t, n);
    let mut gf = result.graph;
    ggml::compute::graph_compute(gctx, &mut gf, 4);

    let n_vocab = gctx.ne(result.logits)[0] as usize;
    let logits: Vec<f32> = bytemuck::cast_slice(gctx.data_bytes(result.logits).unwrap()).to_vec();
    let n_out = gctx.ne(result.embd)[0] as usize;
    let embd: Vec<f32> = bytemuck::cast_slice(gctx.data_bytes(result.embd).unwrap()).to_vec();
    (
        logits[n_vocab * (n - 1)..n_vocab * n].to_vec(),
        embd[n_out * (n - 1)..n_out * n].to_vec(),
    )
}

fn dump_f32(path: &str, v: &[f32]) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut buf = Vec::with_capacity(4 + v.len() * 4);
    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, buf).expect("dump");
}

#[test]
fn pockettts_synthetic_load_and_graph() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = write_pockettts();
    let Some(mut m) = load(&path) else {
        panic!("port refused to load the synthetic pockettts file");
    };
    assert_eq!(m.arch, LlmArch::POCKETTTS);
    assert_eq!(m.hparams.n_layer(), 2);
    // pockettts.cpp:23-24 — the duplicated token_embd is the head
    let tokens = probe_tokens();
    for fa in [true, false] {
        let logits = pockettts_logits(&mut m, &tokens, fa);
        assert_eq!(logits.len(), 32000);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "pockettts logits not finite (fa={fa})"
        );
        dump_f32(&format!("{OUT_DIR}/port-pockettts-logits-{}.bin", if fa { "fa" } else { "nofa" }), &logits);
    }
}

#[test]
fn wavtokenizer_synthetic_load_and_graph() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = write_wavtokenizer();
    let Some(mut m) = load(&path) else {
        panic!("port refused to load the synthetic wavtokenizer-dec file");
    };
    assert_eq!(m.arch, LlmArch::WAVTOKENIZER_DEC);
    assert_eq!(m.hparams.posnet.n_layer, 6);
    assert_eq!(m.hparams.convnext.n_layer, 2);
    assert_eq!(m.hparams.n_norm_groups, 8);
    let tokens = probe_tokens();
    let (logits, embd) = wavtokenizer_run(&mut m, &tokens);
    // the wavtokenizer lm_head emits the waveform dim (n_embd_out), not a
    // vocab — res->t_embd is the decoded PCM frame
    assert_eq!(logits.len(), WV_N_OUT as usize);
    assert_eq!(embd.len(), WV_N_OUT as usize);
    assert!(
        logits.iter().chain(embd.iter()).all(|v| v.is_finite()),
        "wavtokenizer outputs not finite"
    );
    dump_f32(&format!("{OUT_DIR}/port-wavtokenizer-logits.bin"), &logits);
    dump_f32(&format!("{OUT_DIR}/port-wavtokenizer-embd.bin"), &embd);
}

#[test]
fn qwen3tts_synthetic_load() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = write_qwen3tts();
    let Some(mut m) = load(&path) else {
        panic!("port refused to load the synthetic qwen3tts file");
    };
    // qwen3tts is the qwen3vl typedef: the shared QWEN3VL|QWEN3TTS loader
    // arm fills the qwen3vl tensor set (the graph is the ported
    // build_qwen3vl_forward; the ForwardWeights routing is the integrator
    // item noted in AUDIT_models.md)
    assert_eq!(m.arch, LlmArch::QWEN3TTS);
    assert_eq!(m.hparams.n_layer(), Q3T_N_LAYER as u32);
    assert_eq!(m.hparams.rope_sections, [4, 4, 4, 0]);
    assert!(m.layers.iter().all(|l| l.attn_q_norm.is_some()));
}
