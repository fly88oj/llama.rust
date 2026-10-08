//! tts_pipeline_e2e.rs — the pipeline LAYER e2e the audio batch 5 left open:
//! the `mtmd-helper-gen.cpp` port (mtmd.rs's `Qwen3TtsGenPipeline` /
//! `PocketttsGenPipeline` + the `GenTalker` trunk surface) driven the way the
//! reference's `tools/tts/tts.cpp` drives `mtmd_helper_gen_audio_*`
//! (pinned bd4f514db1):
//!
//!   text → set_input → step_prompt batches → per-frame step_gen
//!   (GEN_CODE + the feedback row through the trunk) → GEN_WAV windows
//!   → get_output (PCM + WAV)
//!
//! No real tts GGUF exists on this machine, so the trunk is synthetic — and
//! because the pipeline feeds the trunk **embedding batches**
//! (`decode_embd_batch`, mtmd-helper-common.h:73 — `batch.token = nullptr`,
//! `batch.embd = h`), the trunk arch must be one whose generic decoder
//! consumes embd-only ubatches on BOTH sides. The dflash dual-mode decoder is
//! exactly that (dflash.cpp:609-677: `ubatch.embd` → the KV-injection graph,
//! `res->t_embd = inp_g`; the port's mirror is context.rs's
//! `dflash_features_input` arm). So each pipeline's fixture is a combined
//! pair:
//!
//!   * trunk: a synthetic **dflash** GGUF (n_embd == the mmproj's
//!     n_mmproj_embd, `target_layers = [0]` so n_embd_inp_enc == n_embd, a
//!     token_embd the pipeline reads through the tok_embd table API, and a
//!     vocab = ggml-vocab-llama-spm + the tts specials as CONTROL tokens —
//!     the `<|codec_*|>` / `<tts_*>` / `<|audio_bos|>` pieces
//!     `find_special_token` scans for, mtmd-helper-gen.cpp:43-51);
//!   * mmproj: the already-proven gen builders of tests/tts_gen_e2e.rs
//!     (byte-identical tensor streams).
//!
//! The trunk is driven through the port's real `DecodeContext` (decode_batch
//! with `batch.embd`), the mmproj through the real `MtmdContext`, and the
//! loop replays a FIXED sampled-token stream (the reference's driver samples
//! from trunk logits, which the inject graph does not produce — the probe
//! feeds its identical fixed list to the real C++ pipeline, so both sides
//! run the same deterministic data flow). parity/tts_pipeline_parity.sh
//! hands the same files + inputs to the pinned reference through
//! parity/ref_tts_pipeline.cpp (llama + libmtmd's exported
//! `mtmd_helper_gen_audio_*` — the exact calls tts.cpp makes) and compares
//! the per-step trunk hidden states, the PCM, the WAV bytes and the trunk's
//! final sequence state (the K/V the injection wrote — the position
//! bookkeeping cross-check).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, Value};
use llama::batch::LlamaBatch;
use llama::clip::ClipFlashAttn;
use llama::context::{DecodeContext, ForwardWeights};
use llama::dflash;
use llama::mtmd::{
    GenAudioInp, GenAudioOutType, GenTalker, MtmdContext, MtmdContextParams, MtmdPosType,
    PocketttsGenPipeline, Qwen3TtsGenPipeline,
};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const SPM_N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/tts2-pipe";
const SEED: u32 = 42;

/// the tts specials appended to the SPM vocab as CONTROL tokens
/// (llama-vocab.cpp's special_tokens_cache takes CONTROL/USER_DEFINED, so
/// `llama_tokenize(..., parse_special=true)` splits on them exactly like
/// mtmd-helper-gen.cpp:164-174's chat wrap expects).
const SPECIALS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|codec_0|>",
    // the codebook-0 CODE space — a real checkpoint keeps ~8k contiguous
    // codes after codec_0; the 16-wide dummy block gives the synthetic the
    // same property (the qwen3tts loop's `sampled = codec_0 + k` tokens must
    // never collide with the specials below)
    "<|codec_code_00|>",
    "<|codec_code_01|>",
    "<|codec_code_02|>",
    "<|codec_code_03|>",
    "<|codec_code_04|>",
    "<|codec_code_05|>",
    "<|codec_code_06|>",
    "<|codec_code_07|>",
    "<|codec_code_08|>",
    "<|codec_code_09|>",
    "<|codec_code_10|>",
    "<|codec_code_11|>",
    "<|codec_code_12|>",
    "<|codec_code_13|>",
    "<|codec_code_14|>",
    "<|codec_code_15|>",
    "<|codec_bos|>",
    "<|codec_eos_token|>",
    "<|codec_pad|>",
    "<|codec_think|>",
    "<|codec_think_bos|>",
    "<|codec_think_eos|>",
    "<|codec_language_english|>",
    "<|codec_language_german|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<|audio_bos|>",
    "<|bos_before_voice|>",
];
const N_VOCAB: i64 = SPM_N_VOCAB + SPECIALS.len() as i64;

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

/// the mmproj-side writer — byte-identical tensor streams to
/// tests/tts_gen_e2e.rs's `W` (the proven builders).
struct W {
    w: GgufWriter,
    datas: Vec<Vec<u8>>,
    st: u32,
}

impl W {
    /// a MIXED-modality mmproj — audio (the speaker encoder) + gen-audio.
    /// The audio half is mandatory wiring, not decoration: the reference's
    /// `mtmd_init_from_file` refuses a file whose clip_init yields only
    /// ctx_gen_a (`if (!ctx_v && !ctx_a) throw`, mtmd.cpp:587-590), so every
    /// real tts mmproj ships the spkenc too; with mixed modalities the
    /// projector types move to the per-modality keys (clip.cpp:1262-1275).
    fn new(name: &str, audio_proj: &str, gen_proj: &str) -> Self {
        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("clip".to_string()));
        w.set_kv(
            "general.name",
            Value::String(format!("llama-rust-synth-{name}")),
        );
        w.set_kv("general.file_type", Value::U32(0)); // F32
        w.set_kv("clip.has_vision_encoder", Value::Bool(false));
        w.set_kv("clip.has_audio_encoder", Value::Bool(true));
        w.set_kv("clip.has_gen_audio_encoder", Value::Bool(true));
        w.set_kv("clip.audio.projector_type", Value::String(audio_proj.to_string()));
        w.set_kv("clip.gen.audio.projector_type", Value::String(gen_proj.to_string()));
        Self {
            w,
            datas: Vec::new(),
            st: 0x5eed_0001,
        }
    }
    /// the AUDIO hparams block — the generic prefix-"audio" keys both loaders
    /// require (clip.cpp:1297-1336) + the mel bins the qwen3tts front-end
    /// builds its filterbank from
    fn audio_hp(&mut self, n_embd: i64, n_head: i64, n_ff: i64, n_layer: i64, n_mel: i64) {
        self.kv("clip.audio.embedding_length", Value::U32(n_embd as u32));
        self.kv("clip.audio.attention.head_count", Value::U32(n_head as u32));
        self.kv("clip.audio.attention.head_count_kv", Value::U32(n_head as u32));
        self.kv("clip.audio.feed_forward_length", Value::U32(n_ff as u32));
        self.kv("clip.audio.block_count", Value::U32(n_layer as u32));
        self.kv("clip.audio.projection_dim", Value::U32(n_embd as u32));
        self.kv("clip.audio.attention.layer_norm_epsilon", Value::F32(1e-5));
        self.kv("clip.audio.num_mel_bins", Value::U32(n_mel as u32));
    }
    fn kv(&mut self, k: &str, v: Value) {
        self.w.set_kv(k, v);
    }
    fn add(&mut self, name: &str, ne: [i64; 4]) {
        self.w.add_tensor(name, GgmlType::F32, ne);
        let n = (ne[0] * ne[1] * ne[2] * ne[3]).max(0) as usize;
        self.datas.push(tensor_bytes(n, &mut self.st));
    }
    /// scaled down so the random-weight DAC decoder stays inside F16 after
    /// the conv stack (see tts_gen_e2e.rs's `add_scaled` — the regime a
    /// trained DAC keeps by construction).
    fn add_scaled(&mut self, name: &str, ne: [i64; 4], scale: f32) {
        self.w.add_tensor(name, GgmlType::F32, ne);
        let n = (ne[0] * ne[1] * ne[2] * ne[3]).max(0) as usize;
        let mut st = self.st;
        let bytes = (0..n)
            .map(|_| (lcg(&mut st) * scale).to_le_bytes())
            .flat_map(|b| b.to_vec())
            .collect();
        self.st = st;
        self.datas.push(bytes);
    }
    fn add_scaled_c2w(&mut self, name: &str, ne: [i64; 4]) {
        self.add_scaled(name, ne, 0.1);
    }
    /// the gen.audio hparams block (clip.cpp:1297-1309 + the per-arch arms)
    fn gen_hp(&mut self, n_embd: i64, n_head: i64, n_ff: i64, n_layer: i64, eps: f32) {
        self.kv("clip.gen.audio.embedding_length", Value::U32(n_embd as u32));
        self.kv("clip.gen.audio.attention.head_count", Value::U32(n_head as u32));
        self.kv("clip.gen.audio.attention.head_count_kv", Value::U32(n_head as u32));
        self.kv("clip.gen.audio.feed_forward_length", Value::U32(n_ff as u32));
        self.kv("clip.gen.audio.block_count", Value::U32(n_layer as u32));
        self.kv("clip.gen.audio.projection_dim", Value::U32(n_embd as u32));
        self.kv("clip.gen.audio.attention.layer_norm_epsilon", Value::F32(eps));
    }
    fn finish(self, path: &str) {
        finish_writer(self.w, self.datas, path);
    }
}

fn finish_writer(w: GgufWriter, datas: Vec<Vec<u8>>, path: &str) {
    let tmp = format!("{path}.tmp{}", std::process::id());
    let f = std::fs::File::create(&tmp).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = datas.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    use std::io::Write as _;
    bw.flush().unwrap();
    std::fs::rename(&tmp, path).expect("publish synth gguf");
}

// ===========================================================================
// the mmproj builders — copied verbatim from tests/tts_gen_e2e.rs (the
// proven streams; only the output directory differs)
// ===========================================================================

const Q3_E: i64 = 32;
const Q3_PRED: i64 = 64;
const Q3_VOCAB0: i64 = 16;
const Q3_VOCAB: i64 = 24;
const Q3_NACO: i64 = 3;
const Q3_CB: i64 = 256;
const Q3_HID: i64 = 512;
const Q3_TFMFF: i64 = 1024;
const Q3_UPFF: i64 = 256;

fn write_qwen3tts_gen(path: &str) {
    let mut w = W::new("qwen3tts-gen", "qwen3tts_spkenc", "qwen3tts_gen");
    // ---- the audio (ECAPA-TDNN speaker encoder) half ------------------
    // clip.cpp:1841-1846 hardcodes the mel front-end (24 kHz, n_fft 1024,
    // window 1024, hop 256); the tensor set is clip.cpp:2923-2962 — shapes
    // are never exercised (no speaker input in the pipeline flow), they only
    // have to load identically on both sides
    w.audio_hp(64, 4, 128, 1, 64);
    w.add("a.conv1d.0.weight", [5, 64, 32, 1]);
    w.add("a.conv1d.0.bias", [32, 1, 1, 1]);
    for bid in 1..=1i64 {
        w.add(&format!("a.blk.{bid}.conv_pw1.weight"), [3, 32, 32, 1]);
        w.add(&format!("a.blk.{bid}.conv_pw1.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{bid}.conv_pw2.weight"), [3, 32, 32, 1]);
        w.add(&format!("a.blk.{bid}.conv_pw2.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{bid}.se_conv1.weight"), [1, 32, 8, 1]);
        w.add(&format!("a.blk.{bid}.se_conv1.bias"), [8, 1, 1, 1]);
        w.add(&format!("a.blk.{bid}.se_conv2.weight"), [1, 8, 32, 1]);
        w.add(&format!("a.blk.{bid}.se_conv2.bias"), [32, 1, 1, 1]);
        for xid in 0..7i64 {
            w.add(&format!("a.blk.{bid}.res2.{xid}.weight"), [3, 32, 32, 1]);
            w.add(&format!("a.blk.{bid}.res2.{xid}.bias"), [32, 1, 1, 1]);
        }
    }
    w.add("a.conv_out.weight", [1, 32, 32, 1]);
    w.add("a.conv_out.bias", [32, 1, 1, 1]);
    w.add("a.asp_attn.weight", [1, 32, 1, 1]);
    w.add("a.asp_attn.bias", [1, 1, 1, 1]);
    w.add("a.asp_tdnn.weight", [5, 32, 32, 1]);
    w.add("a.asp_tdnn.bias", [32, 1, 1, 1]);
    // 3-D [., ., n_mmproj_embd] — clip_n_mmproj_embd's SPKENC arm reads
    // ne[2] (clip.cpp:6017), which must equal the trunk's n_embd_inp
    w.add("mm.a.fc.weight", [8, 8, Q3_E, 1]);
    w.add("mm.a.fc.bias", [8, 1, 1, 1]);

    // ---- the gen half (identical stream to tts_gen_e2e) ----------------
    w.gen_hp(Q3_PRED, 4, 128, 2, 1e-5);
    w.add("a.gen.code.proj_in.weight", [Q3_E, Q3_PRED, 1, 1]);
    w.add("a.gen.code.proj_in.bias", [Q3_PRED, 1, 1, 1]);
    w.add("a.gen.code.embd.weight", [Q3_E, Q3_VOCAB, Q3_NACO, 1]);
    w.add("a.gen.code.head.weight", [Q3_PRED, Q3_VOCAB, Q3_NACO, 1]);
    w.add("a.gen.code.out_embd.weight", [Q3_E, Q3_VOCAB0, 1, 1]);
    w.add("a.gen.code.output_norm.weight", [Q3_PRED, 1, 1, 1]);
    for il in 0..2i64 {
        w.add(&format!("a.gen.code.blk.{il}.attn_q.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_k.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_v.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_out.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_q_norm.weight"), [16, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_k_norm.weight"), [16, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ln1.weight"), [Q3_PRED, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ln2.weight"), [Q3_PRED, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_gate.weight"), [Q3_PRED, 128, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_up.weight"), [Q3_PRED, 128, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_down.weight"), [128, Q3_PRED, 1, 1]);
    }
    w.add_scaled_c2w("a.gen.wav.quant.first.in_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.first.out_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.first.codebook.weight", [Q3_CB, Q3_VOCAB0, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.in_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.out_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.codebook.weight", [Q3_CB, Q3_VOCAB, Q3_NACO, 1]);
    w.add_scaled_c2w("a.gen.wav.pre_conv.weight", [3, Q3_HID, 1024, 1]);
    w.add_scaled_c2w("a.gen.wav.pre_conv.bias", [1024, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.in_proj.weight", [1024, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.in_proj.bias", [Q3_HID, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.out_proj.weight", [Q3_HID, 1024, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.out_proj.bias", [1024, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.output_norm.weight", [Q3_HID, 1, 1, 1]);
    for il in 0..8i64 {
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_gate.weight"), [Q3_HID, Q3_TFMFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"), [Q3_HID, Q3_TFMFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"), [Q3_TFMFF, Q3_HID, 1, 1]);
    }
    for il in 0..2i64 {
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.conv.weight"), [2, 1024, 1024, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.conv.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.dwconv.weight"), [7, 1, 1024, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.dwconv.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.norm.weight"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.norm.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw1.weight"), [1024, Q3_UPFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw1.bias"), [Q3_UPFF, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw2.weight"), [Q3_UPFF, 1024, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw2.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.gamma"), [1024, 1, 1, 1]);
    }
    w.add_scaled_c2w("a.gen.wav.dac.entry.weight", [3, 1024, 512, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.entry.bias", [512, 1, 1, 1]);
    let dac_ocs = [512i64, 512, 512, 512];
    for (il, &oc) in dac_ocs.iter().enumerate() {
        let ic = if il == 0 { 512 } else { dac_ocs[il - 1] };
        let il = il as i64;
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.snake.alpha"), [oc, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.snake.beta"), [oc, 1, 1, 1]);
        let k = if il == 0 { 4 } else { 2 };
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.conv.weight"), [k, oc, ic, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.conv.bias"), [oc, 1, 1, 1]);
        for ir in 0..3i64 {
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.alpha"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.beta"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.weight"), [3, oc, oc, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.bias"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.alpha"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.beta"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.weight"), [1, oc, oc, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.bias"), [oc, 1, 1, 1]);
        }
    }
    w.add_scaled_c2w("a.gen.wav.dac.post_snake.alpha", [1, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_snake.beta", [1, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_conv.weight", [3, 512, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_conv.bias", [1, 1, 1, 1]);
    w.finish(path);
}

const PT_E: i64 = 64;
const PT_LAT: i64 = 32;
const PT_C: i64 = 64;

fn write_pockettts_gen(path: &str) {
    let mut w = W::new("pockettts-gen", "pockettts_spkenc", "pockettts_gen");
    // ---- the audio (mimi SEANet speaker encoder) half ------------------
    // clip.cpp:2965-2968 + load_seanet's encoder arm (clip.cpp:2166-2188);
    // seanet_n_stage is 3 (the hardcoded [4,5,6] ratios)
    w.audio_hp(64, 4, 128, 1, 64);
    w.add("a.seanet.conv_in.weight", [3, 64, 32, 1]);
    w.add("a.seanet.conv_in.bias", [32, 1, 1, 1]);
    w.add("a.seanet.conv_out.weight", [3, 32, 1, 1]);
    w.add("a.seanet.conv_out.bias", [1, 1, 1, 1]);
    for i in 0..3i64 {
        w.add(&format!("a.seanet.blk.{i}.res_conv1.weight"), [3, 32, 32, 1]);
        w.add(&format!("a.seanet.blk.{i}.res_conv1.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.seanet.blk.{i}.res_conv2.weight"), [1, 32, 32, 1]);
        w.add(&format!("a.seanet.blk.{i}.res_conv2.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.seanet.blk.{i}.scale_conv.weight"), [7, 32, 32, 1]);
        w.add(&format!("a.seanet.blk.{i}.scale_conv.bias"), [32, 1, 1, 1]);
    }
    w.add("a.downsample.conv.weight", [16, 1, 32, 1]);
    // ne[1] is n_mmproj_embd (clip.cpp:6019) — the trunk width
    w.add("a.speaker_proj.weight", [32, PT_E, 1, 1]);
    // pockettts_spkenc KEEPS the standard layer table (clip.cpp:2249-2253's
    // exclusion list has qwen3tts_spkenc + pockettts_gen only) — the mimi
    // encoder transformer, attn_out.weight required (clip.cpp:2264)
    for il in 0..1i64 {
        w.add(&format!("a.blk.{il}.attn_norm.weight"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.attn_norm.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.attn_q.weight"), [32, 32, 1, 1]);
        w.add(&format!("a.blk.{il}.attn_k.weight"), [32, 32, 1, 1]);
        w.add(&format!("a.blk.{il}.attn_v.weight"), [32, 32, 1, 1]);
        w.add(&format!("a.blk.{il}.attn_out.weight"), [32, 32, 1, 1]);
        w.add(&format!("a.blk.{il}.ln1.weight"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.ln1.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.ln2.weight"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.ln2.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.ffn_up.weight"), [32, 128, 1, 1]);
        w.add(&format!("a.blk.{il}.ffn_gate.weight"), [32, 128, 1, 1]);
        w.add(&format!("a.blk.{il}.ffn_down.weight"), [128, 32, 1, 1]);
        w.add(&format!("a.blk.{il}.ls1.weight"), [32, 1, 1, 1]);
        w.add(&format!("a.blk.{il}.ls2.weight"), [32, 1, 1, 1]);
    }

    // ---- the gen half (identical stream to tts_gen_e2e) ----------------
    w.gen_hp(PT_C, 4, 128, 2, 1e-5);
    w.kv("clip.gen.audio.model_variant", Value::String("english".to_string()));
    w.add("a.gen.flow.input_proj.weight", [PT_LAT, PT_LAT, 1, 1]);
    w.add("a.gen.flow.input_proj.bias", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.cond_embd.weight", [PT_E, PT_LAT, 1, 1]);
    w.add("a.gen.flow.cond_embd.bias", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.final.ada.weight", [PT_LAT, 2 * PT_LAT, 1, 1]);
    w.add("a.gen.flow.final.ada.bias", [2 * PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.final.proj.weight", [PT_LAT, PT_LAT, 1, 1]);
    w.add("a.gen.flow.final.proj.bias", [PT_LAT, 1, 1, 1]);
    for i in 0..2i64 {
        w.add(&format!("a.gen.flow.time.{i}.freqs"), [PT_LAT / 2, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.up.weight"), [PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.up.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.down.weight"), [PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.down.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.norm"), [PT_LAT, 1, 1, 1]);
    }
    for il in 0..2i64 {
        w.add(&format!("a.gen.flow.blk.{il}.norm.weight"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.norm.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.up.weight"), [PT_LAT, 2 * PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.up.bias"), [2 * PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.down.weight"), [2 * PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.down.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.ada.weight"), [PT_LAT, 3 * PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.ada.bias"), [3 * PT_LAT, 1, 1, 1]);
    }
    w.add("a.gen.out_eos.weight", [PT_E, 1, 1, 1]);
    w.add("a.gen.out_eos.bias", [1, 1, 1, 1]);
    w.add("a.gen.input_linear.weight", [PT_LAT, PT_E, 1, 1]);
    w.add("a.gen.emb_mean", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.emb_std", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.wav.quant_out.weight", [PT_LAT, PT_C, 1, 1]);
    w.add("a.gen.wav.upsample.weight", [32, 1, PT_C, 1]);
    w.add("a.gen.wav.seanet.conv_in.weight", [3, PT_C, 32, 1]);
    w.add("a.gen.wav.seanet.conv_in.bias", [32, 1, 1, 1]);
    w.add("a.gen.wav.seanet.conv_out.weight", [3, 32, 1, 1]);
    w.add("a.gen.wav.seanet.conv_out.bias", [1, 1, 1, 1]);
    for (i, &stride) in [6i64, 5, 4].iter().enumerate() {
        let i = i as i64;
        w.add(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.weight"), [stride + 1, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.weight"), [3, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.weight"), [1, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.bias"), [32, 1, 1, 1]);
    }
    for il in 0..2i64 {
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln1.bias"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln2.bias"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"), [PT_C, 128, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"), [128, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"), [PT_C, 1, 1, 1]);
    }
    w.finish(path);
}

// ===========================================================================
// the synthetic trunk — a standalone dflash model whose embd-batch (KV
// injection) graph is the trunk both sides drive (dflash.cpp:609-677)
// ===========================================================================

/// dflash_e2e.rs's weight roles: norms ~1, projections ~1/sqrt(n_embd)
#[derive(Clone, Copy)]
enum Role {
    Norm,
    Proj,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// the SPM vocab KVs with the tts specials appended (CONTROL type, score 0) —
/// tokens / scores / token_type arrays extended in lockstep so both loaders'
/// `n_tokens == scores.size()` checks hold (llama-vocab.cpp:3020-3044).
fn start_trunk_writer(name: &str) -> GgufWriter {
    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if !k.starts_with("tokenizer.") || k == "tokenizer.chat_template" {
            continue;
        }
        let val = match (k.as_str(), val) {
            ("tokenizer.ggml.tokens", Value::Array(ty, items)) => {
                let mut items = items.clone();
                for s in SPECIALS {
                    let _ = ty; // String array in every SPM fixture
                    items.push(Value::String(s.to_string()));
                }
                Value::Array(*ty, items)
            }
            ("tokenizer.ggml.scores", Value::Array(ggml::GgufType::Float32, items)) => {
                let mut items = items.clone();
                for _ in SPECIALS {
                    items.push(Value::F32(0.0));
                }
                Value::Array(ggml::GgufType::Float32, items)
            }
            ("tokenizer.ggml.scores", Value::Array(ggml::GgufType::Int32, items)) => {
                let mut items = items.clone();
                for _ in SPECIALS {
                    items.push(Value::I32(0));
                }
                Value::Array(ggml::GgufType::Int32, items)
            }
            ("tokenizer.ggml.token_type", Value::Array(ggml::GgufType::Int32, items)) => {
                let mut items = items.clone();
                for _ in SPECIALS {
                    items.push(Value::I32(3)); // LLAMA_TOKEN_TYPE_CONTROL
                }
                Value::Array(ggml::GgufType::Int32, items)
            }
            _ => val.clone(),
        };
        w.set_kv(k, val);
    }
    w.set_kv("general.name", Value::String(name.to_string()));
    w.set_kv("general.file_type", Value::U32(0)); // F32
    w
}

/// one synthetic dflash trunk (dflash.cpp:7-259's hparams + the plain
/// backbone tensor table, exactly tests/dflash_e2e.rs's DraftKind::Dflash
/// set plus the standalone token_embd the tok_embd-table read needs).
/// `n_embd` == the mmproj's n_mmproj_embd; `target_layers = [0]` keeps
/// n_embd_inp_enc == n_embd so `decode_embd_batch`'s n_embd-wide rows hit
/// the injection graph's input width (llama-graph.cpp:76-81's assert).
fn write_trunk(path: &str, n_embd: i64, n_head: i64, n_head_kv: i64, n_ff: i64, seed: u64) {
    let key = 16i64; // n_embd_head_k == n_embd_head_v
    let mut w = start_trunk_writer("llama-rust-synth-tts-trunk");
    let a = "dflash";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(512));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(n_embd as u32));
    w.set_kv(&format!("{a}.block_count"), Value::U32(1));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(n_ff as u32));
    w.set_kv(&format!("{a}.attention.head_count"), Value::U32(n_head as u32));
    w.set_kv(&format!("{a}.attention.head_count_kv"), Value::U32(n_head_kv as u32));
    w.set_kv(&format!("{a}.attention.key_length"), Value::U32(key as u32));
    w.set_kv(&format!("{a}.attention.value_length"), Value::U32(key as u32));
    w.set_kv(&format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    w.set_kv(&format!("{a}.rope.dimension_count"), Value::U32(key as u32));
    w.set_kv(&format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // dflash.cpp:36-38 — required extract-layer array; [0] → 1 fused stream
    w.set_kv(
        &format!("{a}.target_layers"),
        Value::Array(ggml::GgufType::Int32, vec![Value::I32(0)]),
    );
    w.set_kv(&format!("{a}.block_size"), Value::U32(4));

    let mut rng = Rng(seed);
    let mut datas: Vec<Vec<u8>> = Vec::new();
    let mut add = |w: &mut GgufWriter, name: &str, ne: Vec<i64>, role: Role| {
        let n: i64 = ne.iter().product();
        let s = match role {
            Role::Norm => 1.0,
            Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        };
        let mut bytes = Vec::with_capacity((n * 4) as usize);
        for _ in 0..n {
            let v = match role {
                Role::Norm => 1.0 + 0.05 * rng.next(),
                Role::Proj => s * rng.next(),
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(
            name,
            GgmlType::F32,
            [
                ne[0],
                *ne.get(1).unwrap_or(&1),
                *ne.get(2).unwrap_or(&1),
                *ne.get(3).unwrap_or(&1),
            ],
        );
        datas.push(bytes);
    };

    // the standalone embeddings (llama_model_get_tok_embd's source,
    // mtmd-helper-gen.cpp:352-360); the output head falls back to it
    // (dflash.cpp:166-171 TENSOR_DUPLICATED)
    add(&mut w, "token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    // feature fusion (dflash.cpp:161-163) — n_embd_inp_enc == n_embd here
    add(&mut w, "fc.weight", vec![n_embd, n_embd], Role::Proj);
    add(&mut w, "enc.output_norm.weight", vec![n_embd], Role::Norm);
    add(&mut w, "output_norm.weight", vec![n_embd], Role::Norm);
    let n_gqa_k = key * n_head_kv;
    for i in 0..1 {
        add(&mut w, &format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
        add(&mut w, &format!("blk.{i}.attn_q.weight"), vec![n_embd, key * n_head], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_k.weight"), vec![n_embd, n_gqa_k], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_v.weight"), vec![n_embd, n_gqa_k], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_output.weight"), vec![key * n_head, n_embd], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_q_norm.weight"), vec![key], Role::Norm);
        add(&mut w, &format!("blk.{i}.attn_k_norm.weight"), vec![key], Role::Norm);
        add(&mut w, &format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
        add(&mut w, &format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], Role::Proj);
        add(&mut w, &format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
        add(&mut w, &format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
    }
    finish_writer(w, datas, path);
}

fn trunk_path(q3t: bool) -> String {
    format!("{OUT_DIR}/trunk-{}.gguf", if q3t { "q3t" } else { "pt" })
}
fn mmproj_path(q3t: bool) -> String {
    format!(
        "{OUT_DIR}/mmproj-{}.gguf",
        if q3t { "qwen3tts-gen" } else { "pockettts-gen" }
    )
}

fn build_fixtures() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    if !std::path::Path::new(&trunk_path(true)).exists() {
        // heads: q3t 2x2x16 == 32 wide; pt 4q/2kv x16
        write_trunk(&trunk_path(true), Q3_E, 2, 2, 64, 0x775_5111);
    }
    if !std::path::Path::new(&trunk_path(false)).exists() {
        write_trunk(&trunk_path(false), PT_E, 4, 2, 128, 0x775_5222);
    }
    if !std::path::Path::new(&mmproj_path(true)).exists() {
        write_qwen3tts_gen(&mmproj_path(true));
    }
    if !std::path::Path::new(&mmproj_path(false)).exists() {
        write_pockettts_gen(&mmproj_path(false));
    }
}

// ---------------------------------------------------------------------------
// the GenTalker — the port's real embd-batch decode of the dflash trunk
// ---------------------------------------------------------------------------

struct DflashTalker {
    gctx: ggml::Context,
    w: dflash::DflashWeights,
    p: dflash::DflashParams,
    kv: llama::kv_cache::KvCache,
    watermark: usize,
    n_embd: usize,
    /// the last decode's final row (`llama_get_embeddings_ith(lctx, -1)`)
    last_h: Vec<f32>,
}

impl DflashTalker {
    /// `llama_init_from_model` for the trunk — the dflash weights over their
    /// own ggml context, plus the KV cache `DecodeContext::new_dflash`
    /// builds (context.rs:2879-2897's per-layer k/v rows). The decode itself
    /// drives the port's inject builder + KV the same way context.rs's
    /// dflash arm does (`build_dflash_inject_forward` over a step's
    /// `DecodeInputs`); the trunk doubles as its own "target" file (its
    /// token_embd satisfies the ctx_other lookup the loader performs only
    /// when the file omits it).
    fn open(path: &str) -> (Self, Vec<f32>, std::rc::Rc<Vocab>) {
        let gguf = Gguf::open(path).expect("open trunk gguf");
        let file = std::fs::File::open(path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        let vocab = std::rc::Rc::new(Vocab::load(&gguf).expect("vocab"));
        let n_vocab = vocab.n_tokens() as i64;
        let draft =
            dflash::load_dflash_draft(&gguf, mmap.clone(), &gguf, mmap, n_vocab, false)
                .expect("load dflash trunk");
        let n_embd = draft.params.n_embd as usize;
        // the whole token embedding matrix (llama_model_get_tok_embd)
        let tok_embd: Vec<f32> = bytemuck::cast_slice(
            draft.ctx.data_bytes(draft.weights.tok_embd.expect("tok_embd")).unwrap(),
        )
        .to_vec();
        assert_eq!(tok_embd.len(), n_vocab as usize * n_embd);

        let (mut gctx, w, p) = (draft.ctx, draft.weights, draft.params);
        let n_layer = w.layers.len();
        let kv = llama::kv_cache::KvCache::new_with_dims(
            &mut gctx,
            &vec![p.attn.n_embd_head_k * p.attn.n_head_kv; n_layer],
            &vec![p.attn.n_embd_head_v * p.attn.n_head_kv; n_layer],
            512,
        );
        // the watermark must sit AFTER the cache tensors (the dflash_e2e
        // driver order — reset_graph_to would otherwise drop them)
        let watermark = gctx.mark();
        (
            Self {
                gctx,
                w,
                p,
                kv,
                watermark,
                n_embd,
                last_h: Vec::new(),
            },
            tok_embd,
            vocab,
        )
    }
}

impl GenTalker for DflashTalker {
    /// `decode_embd_batch` + `llama_decode` + `llama_get_embeddings_ith(-1)`:
    /// KV slot assign at `pos..pos+n`, then the injection graph — its
    /// `t_embd` (inp_g) rides the result's logits slot, [n_embd] per token.
    fn decode_embd(
        &mut self,
        embd: &[f32],
        n_tokens: usize,
        pos: i32,
        seq_id: i32,
    ) -> Result<Vec<f32>, String> {
        use ggml::types::GgmlType;
        let n = n_tokens;
        assert_eq!(embd.len(), n * self.n_embd);
        let positions: Vec<i32> = (pos..pos + n as i32).collect();
        let sinfo = self.kv.find_slot(n as u32).ok_or("trunk KV full")?;
        self.kv.assign(sinfo, &positions, seq_id as usize);

        self.gctx.reset_graph_to(self.watermark);
        let t = n as i64;
        // the inject graph consumes pos + row_idx only; tokens/mask are the
        // struct's unused companions (dflash.cpp:609-677 never reads them)
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, t);
        let kq_mask = self.gctx.new_tensor_2d(GgmlType::F32, 1, t);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, t);
        for tid in [tokens_t, pos_t, kq_mask, row_idx] {
            self.gctx.arena_resize_tensor(tid);
        }
        self.gctx
            .with_i32_mut(pos_t, |q| q.copy_from_slice(&positions))
            .unwrap();
        let idxs: Vec<i64> = (sinfo.s0 as i64..=sinfo.s1 as i64).collect();
        self.gctx
            .data_bytes_mut(row_idx)
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&idxs));
        // the F32 feature input (`llm_graph_input_embd::embd`, dflash.cpp:612)
        let n_e = self.p.n_embd_inp_enc as i64;
        let features = self.gctx.new_tensor_2d(GgmlType::F32, n_e, t);
        self.gctx.arena_resize_tensor(features);
        self.gctx
            .with_f32_mut(features, |q| q.copy_from_slice(embd))
            .unwrap();

        let inputs = llama::graph::DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let result = dflash::build_dflash_inject_forward(
            &mut self.gctx,
            &self.w,
            &self.p,
            &self.kv,
            &inputs,
            features,
            n,
        );
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 4);
        let all: &[f32] = bytemuck::cast_slice(self.gctx.data_bytes(result.logits).unwrap());
        let w = self.n_embd;
        let row = all[w * (n - 1)..w * n].to_vec();
        self.last_h = row.clone();
        Ok(row)
    }

    /// `llama_memory_seq_rm(seq_id, p0, p1)` — p1 = -1 keeps to the end
    fn seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> Result<(), String> {
        self.kv.seq_rm(seq_id as usize, p0, p1);
        Ok(())
    }
}

impl DflashTalker {
    /// the trunk's final sequence state (`llama_state_seq_get_data`, seq 0) —
    /// the K/V the injection graph wrote, the position-parity cross-check.
    /// The framed blob `DecodeContext::state_seq_get_data` emits
    /// (context.rs:3023-3042): [u32 magic][i32 seq_id]<kv state>.
    fn state(&self) -> Vec<u8> {
        let mut io = llama::kv_cache::StateWriter::new(false);
        io.write_u32(0xaf14_3cd8); // DecodeContext::STATE_SEQ_IO_MAGIC
        io.write_i32(0);
        self.kv
            .state_seq_write(&mut io, &self.gctx, 0, false)
            .expect("trunk state write");
        io.into_bytes()
    }
}

// ---------------------------------------------------------------------------
// dumps (the formats parity/tts_pipeline_parity.sh compares)
// ---------------------------------------------------------------------------

fn dump_blob(path: &str, parts: &[&[u8]]) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut buf = Vec::new();
    for p in parts {
        buf.extend_from_slice(p);
    }
    std::fs::write(path, buf).expect("dump");
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// the numeric dump: prompt rows, the per-step trunk hidden states (the
/// prompt-final h first), the PCM, the trunk's final sequence state.
fn dump_pipeline(tag: &str, n_prompt: usize, hs: &[f32], h_len: usize, pcm: &[f32], state: &[u8]) {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    parts.push((n_prompt as u32).to_le_bytes().to_vec());
    parts.push(((hs.len() / h_len) as u32).to_le_bytes().to_vec());
    parts.push((h_len as u32).to_le_bytes().to_vec());
    parts.push(f32_bytes(hs));
    parts.push((pcm.len() as u32).to_le_bytes().to_vec());
    parts.push(f32_bytes(pcm));
    parts.push((state.len() as u32).to_le_bytes().to_vec());
    parts.push(state.to_vec());
    let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
    dump_blob(&format!("{OUT_DIR}/port-{tag}-pcm.bin"), &refs);
}

fn dump_wav(tag: &str, rate: i32, wav: &[u8]) {
    let parts: Vec<Vec<u8>> = vec![
        (rate as u32).to_le_bytes().to_vec(),
        (wav.len() as u32).to_le_bytes().to_vec(),
        wav.to_vec(),
    ];
    let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
    dump_blob(&format!("{OUT_DIR}/port-{tag}-wav.bin"), &refs);
}

// ---------------------------------------------------------------------------
// the driver (tts.cpp:126-190's loop, fixed sampled stream)
// ---------------------------------------------------------------------------

fn find_special(vocab: &Vocab, piece: &str) -> i32 {
    (0..vocab.n_tokens() as i32)
        .find(|&t| vocab.token_to_piece(t) == piece)
        .unwrap_or_else(|| panic!("missing special {piece}"))
}

/// the fixed qwen3tts backbone token stream: 80 codec_0-frame tokens then
/// the end-of-speech special (exercises the 72-frame window flush mid-run
/// plus the final get_output flush). The reference probe replays the same
/// ids from q3t-sampled.bin.
fn q3t_sampled(vocab: &Vocab) -> Vec<i32> {
    let codec_0 = find_special(vocab, "<|codec_0|>");
    let codec_eos = find_special(vocab, "<|codec_eos_token|>");
    let mut ids = Vec::new();
    for i in 0..80i32 {
        ids.push(codec_0 + (i * 7 + 3) % 16);
    }
    ids.push(codec_eos);
    ids
}

fn run_qwen3tts(out_type: GenAudioOutType) {
    build_fixtures();
    let (talker, tok_embd, vocab) = DflashTalker::open(&trunk_path(true));

    let mctx = MtmdContext::init_from_file(
        &mmproj_path(true),
        Some((&vocab, Q3_E as i32, MtmdPosType::Normal)),
        &MtmdContextParams {
            n_threads: 4,
            flash_attn_type: ClipFlashAttn::Disabled,
            ..Default::default()
        },
    )
    .expect("mtmd init");

    let mut pipe = Qwen3TtsGenPipeline::new(
        talker,
        mctx,
        vocab.clone(),
        tok_embd,
        Q3_E as usize,
        false, // dflash is not an mrope trunk
    )
    .expect("qwen3tts pipeline");

    let sampled = q3t_sampled(&vocab);
    {
        let parts: Vec<Vec<u8>> = vec![
            (sampled.len() as u32).to_le_bytes().to_vec(),
            sampled.iter().flat_map(|t| t.to_le_bytes()).collect(),
        ];
        let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        dump_blob(&format!("{OUT_DIR}/q3t-sampled.bin"), &refs);
    }

    let prompt = "The quick brown fox jumps over the lazy dog.";
    let inp = GenAudioInp {
        prompt,
        lang: "en",
        speaker_ref: None,
        top_k: 20,
        top_p: 0.9,
        seed: SEED,
        out_type,
    };
    pipe.set_input(&inp, None).expect("set_input");
    let n_prompt = loop {
        let left = pipe.step_prompt(8).expect("step_prompt");
        if left == 0 {
            break pipe.n_prompt_rows();
        }
    };
    // tts.cpp:158 — the prompt-final hidden state seeds the loop
    let mut h: Vec<f32> = pipe.talker_ref().last_h.clone();
    assert_eq!(h.len(), Q3_E as usize);
    let mut hs = h.clone();

    let mut stop = false;
    let mut i = 0usize;
    while !stop && i < sampled.len() {
        let (h_next, s) = pipe.step_gen(sampled[i], &h).expect("step_gen");
        if s || h_next.is_none() {
            stop = true;
            break;
        }
        h = h_next.unwrap();
        hs.extend_from_slice(&h);
        i += 1;
    }
    assert!(stop, "the fixed stream must end on codec_eos");
    assert_eq!(i + 1, sampled.len(), "every frame token consumed");

    // get_output flushes the remaining codes before packaging
    // (mtmd-helper-gen.cpp:304-328) — read the PCM through it, never from a
    // pre-flush audio_pcm()
    let (rate, bytes) = pipe.get_output().expect("get_output");
    assert_eq!(rate, 24000);
    if out_type == GenAudioOutType::Pcm {
        let pcm: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!(!pcm.is_empty());
        assert!(pcm.iter().all(|v| v.is_finite()));
        // 81 frames total: the 72-frame window flush + the 9-frame tail
        assert!(pcm.len() % 8 == 0, "8 samples per frame");
        dump_pipeline(
            "q3t",
            n_prompt,
            &hs,
            Q3_E as usize,
            &pcm,
            &pipe.talker_ref().state(),
        );
    } else {
        assert_eq!(&bytes[0..4], b"RIFF");
        dump_wav("q3t", rate, &bytes);
    }
}

fn run_pockettts(out_type: GenAudioOutType) {
    build_fixtures();
    let (talker, tok_embd, vocab) = DflashTalker::open(&trunk_path(false));

    let mctx = MtmdContext::init_from_file(
        &mmproj_path(false),
        Some((&vocab, PT_E as i32, MtmdPosType::Normal)),
        &MtmdContextParams {
            n_threads: 4,
            flash_attn_type: ClipFlashAttn::Disabled,
            ..Default::default()
        },
    )
    .expect("mtmd init");

    let mut pipe = PocketttsGenPipeline::new(talker, mctx, vocab.clone(), tok_embd, PT_E as usize)
        .expect("pockettts pipeline");

    // a two-sentence prompt (> 50 tokens) so split_chunks builds two chunks
    // and finish_chunk exercises seq_rm + the voice-pos re-prompt
    // (mtmd-helper-gen.cpp:757-802 / :815-858): each sentence stays under
    // MAX_TOKEN_PER_CHUNK (50) but their sum exceeds it
    let prompt = "Pocket speech rolls several short words into one steady \
                  clause and then keeps rolling through more words before the \
                  first question mark finally lands right here? A second \
                  clause then follows the same easy rhythm and adds its own \
                  words until its own ending arrives at the very last line.";
    let inp = GenAudioInp {
        prompt,
        lang: "en",
        speaker_ref: None,
        top_k: 0, // pockettts ignores top_k/top_p (pack temp only)
        top_p: 0.0,
        seed: SEED,
        out_type,
    };
    pipe.set_input(&inp, None).expect("set_input");
    assert!(pipe.n_chunks() >= 2, "the prompt must split into chunks");
    let n_prompt = loop {
        let left = pipe.step_prompt(8).expect("step_prompt");
        if left == 0 {
            break pipe.n_prompt_rows();
        }
    };
    let mut h: Vec<f32> = pipe.talker_ref().last_h.clone();
    assert_eq!(h.len(), PT_E as usize);
    let mut hs = h.clone();

    // pockettts has no backbone token — sampled is unused
    // (mtmd-helper-gen.cpp:611 "(void) sampled")
    let mut stop = false;
    let mut steps = 0usize;
    while !stop && steps < 1200 {
        let (h_next, s) = pipe.step_gen(&h).expect("step_gen");
        if s || h_next.is_none() {
            stop = true;
            break;
        }
        h = h_next.unwrap();
        hs.extend_from_slice(&h);
        steps += 1;
    }
    assert!(stop, "the chunk budget must terminate the run");
    eprintln!("pockettts: {steps} steps over {} chunks", pipe.n_chunks());

    // the final window rides get_output's flush
    let (rate, bytes) = pipe.get_output().expect("get_output");
    assert_eq!(rate, 24000);
    if out_type == GenAudioOutType::Pcm {
        let pcm: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!(!pcm.is_empty());
        assert!(pcm.iter().all(|v| v.is_finite()));
        dump_pipeline(
            "pt",
            n_prompt,
            &hs,
            PT_E as usize,
            &pcm,
            &pipe.talker_ref().state(),
        );
    } else {
        assert_eq!(&bytes[0..4], b"RIFF");
        dump_wav("pt", rate, &bytes);
    }
}

#[test]
fn qwen3tts_pipeline_e2e() {
    let _g = file_lock();
    run_qwen3tts(GenAudioOutType::Pcm);
    run_qwen3tts(GenAudioOutType::Wav);
}

#[test]
fn pockettts_pipeline_e2e() {
    let _g = file_lock();
    run_pockettts(GenAudioOutType::Pcm);
    run_pockettts(GenAudioOutType::Wav);
}

// ===========================================================================
// the PLAIN-arch embd batch (batch-18 integrator item): DecodeContext::
// decode_embd + GenTalker for DecodeContext over a qwen3 trunk
// ===========================================================================

/// one synthetic plain qwen3 trunk — the arch family the real qwen3tts
/// backbone belongs to (qwen3tts.cpp:3 is the qwen3vl typedef; its trunk
/// graph is qwen3-style with q/k norms) — same tensor roles as write_trunk's
/// dflash backbone minus the dflash-specific fusion tensors.
fn write_plain_trunk(path: &str) {
    let n_embd = 64i64;
    let key = 16i64;
    let n_head = 4i64;
    let n_head_kv = 2i64;
    let n_ff = 128i64;
    let mut w = start_trunk_writer("llama-rust-synth-tts-trunk-qwen3");
    let a = "qwen3";
    w.set_kv("general.architecture", Value::String(a.to_string()));
    w.set_kv(&format!("{a}.context_length"), Value::U32(512));
    w.set_kv(&format!("{a}.embedding_length"), Value::U32(n_embd as u32));
    w.set_kv(&format!("{a}.block_count"), Value::U32(1));
    w.set_kv(&format!("{a}.feed_forward_length"), Value::U32(n_ff as u32));
    w.set_kv(&format!("{a}.attention.head_count"), Value::U32(n_head as u32));
    w.set_kv(&format!("{a}.attention.head_count_kv"), Value::U32(n_head_kv as u32));
    w.set_kv(&format!("{a}.attention.key_length"), Value::U32(key as u32));
    w.set_kv(&format!("{a}.attention.value_length"), Value::U32(key as u32));
    w.set_kv(&format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    w.set_kv(&format!("{a}.rope.dimension_count"), Value::U32(key as u32));

    let mut rng = Rng(0x5eed_cafe_f00d);
    let mut datas: Vec<Vec<u8>> = Vec::new();
    let mut add = |w: &mut GgufWriter, name: &str, ne: Vec<i64>, role: Role| {
        let n: i64 = ne.iter().product();
        let s = match role {
            Role::Norm => 1.0,
            Role::Proj => 1.0 / (n_embd as f32).sqrt(),
        };
        let mut bytes = Vec::with_capacity((n * 4) as usize);
        for _ in 0..n {
            let v = match role {
                Role::Norm => 1.0 + 0.05 * rng.next(),
                Role::Proj => s * rng.next(),
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.add_tensor(
            name,
            GgmlType::F32,
            [ne[0], *ne.get(1).unwrap_or(&1), *ne.get(2).unwrap_or(&1), *ne.get(3).unwrap_or(&1)],
        );
        datas.push(bytes);
    };

    add(&mut w, "token_embd.weight", vec![n_embd, N_VOCAB], Role::Proj);
    add(&mut w, "output_norm.weight", vec![n_embd], Role::Norm);
    add(&mut w, "output.weight", vec![n_embd, N_VOCAB], Role::Proj);
    let n_gqa_k = key * n_head_kv;
    for i in 0..1 {
        add(&mut w, &format!("blk.{i}.attn_norm.weight"), vec![n_embd], Role::Norm);
        add(&mut w, &format!("blk.{i}.attn_q.weight"), vec![n_embd, key * n_head], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_k.weight"), vec![n_embd, n_gqa_k], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_v.weight"), vec![n_embd, n_gqa_k], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_output.weight"), vec![key * n_head, n_embd], Role::Proj);
        add(&mut w, &format!("blk.{i}.attn_q_norm.weight"), vec![key], Role::Norm);
        add(&mut w, &format!("blk.{i}.attn_k_norm.weight"), vec![key], Role::Norm);
        add(&mut w, &format!("blk.{i}.ffn_norm.weight"), vec![n_embd], Role::Norm);
        add(&mut w, &format!("blk.{i}.ffn_gate.weight"), vec![n_embd, n_ff], Role::Proj);
        add(&mut w, &format!("blk.{i}.ffn_down.weight"), vec![n_ff, n_embd], Role::Proj);
        add(&mut w, &format!("blk.{i}.ffn_up.weight"), vec![n_embd, n_ff], Role::Proj);
    }
    finish_writer(w, datas, path);
}

/// `decode_embd` (the `decode_embd_batch` port, tokens=nullptr,
/// mtmd-helper-common.h:73) over a PLAIN trunk must reproduce the token
/// path's hidden states bit-exactly: feeding token_embd's rows for T as embd
/// rows equals decoding T itself — the C's two `build_inp_embd` arms
/// (llama-graph.cpp:2387-2445 `ggml_build_forward_select(ubatch.token ? 0
/// : 1)`) are the same values for a plain arch, and the port drives the embd
/// arm through the materialised-matrix gather. Drives the trait object
/// (`GenTalker for DecodeContext`) the gen pipelines hold.
#[test]
fn gen_talker_embd_rows_match_token_path() {
    use llama::graph_arch::{Qwen3LayerWeights, Qwen3ModelWeights};
    let _g = file_lock();
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = format!("{OUT_DIR}/trunk-qwen3.gguf");
    if !std::path::Path::new(&path).exists() {
        write_plain_trunk(&path);
    }
    let toks: Vec<i32> = (0..8).map(|i| 100 + i * 7).collect();
    let n = toks.len();
    let pos: Vec<i32> = (0..n as i32).collect();

    let qwen3_trunk = |m: &llama::model::LlamaModel| {
        Qwen3ModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers: m
                .layers
                .iter()
                .map(|x| Qwen3LayerWeights {
                    attn_norm: x.attn_norm.expect("attn_norm"),
                    wq: x.wq.expect("wq"),
                    wk: x.wk.expect("wk"),
                    wv: x.wv.expect("wv"),
                    wq_b: x.wq_b,
                    wk_b: x.wk_b,
                    wv_b: x.wv_b,
                    wo: x.wo.expect("wo"),
                    attn_q_norm: x.attn_q_norm.expect("attn_q_norm"),
                    attn_k_norm: x.attn_k_norm.expect("attn_k_norm"),
                    ffn_norm: x.ffn_norm.expect("ffn_norm"),
                    ffn_gate: x.ffn_gate.expect("ffn_gate"),
                    ffn_down: x.ffn_down.expect("ffn_down"),
                    ffn_up: x.ffn_up.expect("ffn_up"),
                })
                .collect(),
        }
    };
    let qwen3_attn = |m: &llama::model::LlamaModel| {
        let hp = &m.hparams;
        let rope = hp.rope_runtime();
        llama::graph::AttnParams {
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
    };
    let open = || {
        let gguf = Gguf::open(&path).expect("open trunk gguf");
        let file = std::fs::File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }.unwrap());
        llama::model::load_model(&gguf, mmap).expect("load qwen3 trunk")
    };

    // side A — the token path: decode_embed over T (pooling NONE rows), plus
    // token_embd's rows for T (read before the ctx moves out)
    let (h_last_tok, rows) = {
        let mut m = open();
        let ne = m.ctx.ne(m.tok_embd)[0] as usize;
        let all: &[f32] = bytemuck::cast_slice(m.ctx.data_bytes(m.tok_embd).unwrap());
        let mut rows = Vec::with_capacity(n * ne);
        for &t in &toks {
            rows.extend_from_slice(&all[t as usize * ne..(t as usize + 1) * ne]);
        }
        let w = qwen3_trunk(&m);
        let attn = qwen3_attn(&m);
        let gctx = std::mem::replace(&mut m.ctx, ggml::Context::new());
        let mut dctx =
            DecodeContext::new_with(gctx, ForwardWeights::Qwen3(w), attn, 512, 8, 512)
                .with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);
        let e = dctx.decode_embed(&toks, &pos).expect("decode_embed");
        (e.values[e.values.len() - e.n_embd_out..].to_vec(), rows)
    };

    // side B — the embd path: the SAME rows as a decode_embd_batch
    // (tokens=nullptr, mtmd-helper-common.h:73) through the trait object the
    // gen pipelines hold (GenTalker for DecodeContext)
    let h_last_embd = {
        let mut m = open();
        let w = qwen3_trunk(&m);
        let attn = qwen3_attn(&m);
        let gctx = std::mem::replace(&mut m.ctx, ggml::Context::new());
        let mut dctx =
            DecodeContext::new_with(gctx, ForwardWeights::Qwen3(w), attn, 512, 8, 512)
                .with_embeddings(true, llama::hparams::LlamaPoolingType::NONE);
        let talker: &mut dyn GenTalker = &mut dctx;
        let h = talker
            .decode_embd(&rows, n, 0, 0)
            .expect("decode_embd (decode_embd_batch)");
        // seq_rm round-trips through the same trait (llama_memory_seq_rm)
        talker.seq_rm(0, -1, -1).expect("seq_rm");
        h
    };

    assert_eq!(h_last_tok.len(), h_last_embd.len());
    assert_eq!(h_last_tok, h_last_embd, "the embd arm must be bit-identical");
}

/// the tests rewrite the same /tmp fixtures and keep their mmaps alive
/// across loads; serialize whole tests like dflash_e2e.rs does.
fn file_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
