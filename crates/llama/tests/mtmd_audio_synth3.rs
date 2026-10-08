//! mtmd_audio_synth3.rs — audio-encoder round 4: the remaining encoder
//! graphs (granite_speech, gemma4a, parakeet, mimo_audio, qwen3tts_spkenc,
//! pockettts_spkenc) built with the PORT's GGUF writer (the established
//! synthetic protocol), then:
//!
//!   1. the port loads each (`clip_init_from_file` audio branch) and runs the
//!      whole audio path (preprocess → tokens → encoder graph);
//!   2. the same files are handed to the pinned reference `llama-mtmd-cli
//!      --mmproj` by parity/audio_mtmd_parity3.sh — the reference ACCEPTS them
//!      (load + graph build + embeddings) and its MTMD_DEBUG_EMBEDDINGS dump
//!      is compared bit-exactly against the port's (f32::to_bits), for the
//!      default flash-attention path and `-fa off`.
//!
//! The 24 kHz archs (mimo_audio / qwen3tts_spkenc / pockettts_spkenc) run on
//! the rate-matched 24 kHz fixture: a rate-matched WAV is a passthrough in
//! the reference miniaudio path (no resampler) and the only thing the port's
//! no-resample reader accepts.
//!
//! The generated files are deterministic (seeded LCG weights).

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, Value};
use llama::clip::{
    clip_init_from_file, ClipContextParams, ClipFlashAttn, ClipModality, ProjectorType,
};
use llama::mtmd::{MtmdChunk, MtmdContext, MtmdContextParams};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";

const OUT_DIR: &str = "/tmp/mtmd-audio-synth";
/// must match the text model used by parity/audio_mtmd_parity3.sh
const PROJ_DIM: i64 = 896;

fn proj_dim() -> i64 {
    std::env::var("MMPROJ_PROJ_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PROJ_DIM)
}

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

/// the round-4 archs — models/{granite-speech,gemma4a,parakeet,mimo-audio,
/// qwen3tts-spkenc,pockettts-spkenc}.cpp
pub const FAMILY: &[(&str, ProjectorType)] = &[
    ("granite_speech", ProjectorType::GraniteSpeech),
    ("gemma4a", ProjectorType::Gemma4A),
    ("parakeet", ProjectorType::Parakeet),
    ("mimo_audio", ProjectorType::MimoAudio),
    ("qwen3tts_spkenc", ProjectorType::Qwen3TtsSpkEnc),
    ("pockettts_spkenc", ProjectorType::PocketTtsSpkEnc),
];

// granite_speech geometry (models/granite-speech.cpp):
//   mel 64 (stacked 2x32), n_embd 64, 2 layers, chunk 25 (6 blocks, remainder
//   3 → the attn_mask path), conv kernel 9, window 50/downsample 25 → 6 tokens
const GS_N_MEL: i64 = 64; // preprocessor stacks n_mel_bins/2 x 2
const GS_N_EMBD: i64 = 64;
const GS_N_FF: i64 = 32;
const GS_N_HEAD: i64 = 4;
const GS_D_CONV: i64 = 32;

// gemma4a geometry (models/gemma4a.cpp): mel 32, sscp 1→16→16 ch (freq 8),
// n_embd 64, 2 layers, chunked attention C=12/P=12/S=24
const G4A_N_MEL: i64 = 32;
const G4A_SSCP_OC: [i64; 2] = [16, 16];
const G4A_N_EMBD: i64 = 64;
const G4A_N_FF: i64 = 32;
const G4A_N_HEAD: i64 = 4;
const G4A_D_CONV: i64 = 32;

// parakeet geometry (models/parakeet.cpp): mel 32, the lfm2a-style pre-encode
// stack with C=8, n_embd 64, 2 layers, conv kernel 9
const PK_N_MEL: i64 = 32;
const PK_N_EMBD: i64 = 64;
const PK_N_FF: i64 = 32;
const PK_N_HEAD: i64 = 4;
const PK_D_CONV: i64 = 32;

// mimo geometry (models/mimo-audio.cpp): 24 kHz mel 64, conv1d 64→32→32,
// 4 vit layers (skip layer 2), downsample k=2/s=2, RVQ 2x16 bins,
// 2 local layers, group 8
const MI_N_MEL: i64 = 64;
const MI_C: i64 = 32;
const MI_N_EMBD: i64 = 64;
const MI_N_FF: i64 = 32;
const MI_N_HEAD: i64 = 4;
const MI_N_LAYER: i64 = 4;
const MI_BINS: i64 = 16;
const MI_N_Q: i64 = 2;

// qwen3tts spkenc geometry (models/qwen3tts-spkenc.cpp): 24 kHz mel 40,
// ECAPA C=128, 3 blocks at scale 8 (Cs=16), MFA 384, ASP 384
const QS_N_MEL: i64 = 40;
const QS_C: i64 = 128;
const QS_CS: i64 = 16;
const QS_MFA: i64 = 384;

// pockettts geometry (models/pockettts-spkenc.cpp): 24 kHz raw waveform,
// SEANet ratios [4,5,6] C=64 (== the transformer width), 4 transformer
// layers of width 64, downsample 16
const PT_C: i64 = 64;
const PT_N_EMBD: i64 = 64;
const PT_N_FF: i64 = 32;
const PT_N_HEAD: i64 = 4;
const PT_N_LAYER: i64 = 4;

/// write one synthetic round-4 mmproj; returns the path
pub fn write_synthetic_mmproj(arch: &str) -> String {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let path = format!("{OUT_DIR}/mmproj-audio-synth-{arch}.gguf");

    let mut w = GgufWriter::new(32);
    w.set_kv("general.architecture", Value::String("clip".to_string()));
    w.set_kv(
        "general.name",
        Value::String(format!("llama-rust-synth-{arch}")),
    );
    w.set_kv("general.file_type", Value::U32(0)); // F32

    w.set_kv("clip.has_vision_encoder", Value::Bool(false));
    w.set_kv("clip.has_audio_encoder", Value::Bool(true));
    w.set_kv("clip.projector_type", Value::String(arch.to_string()));

    let mut st = 0x3456_789au32;
    let mut datas: Vec<Vec<u8>> = Vec::new();
    let add =
        |w: &mut GgufWriter, name: &str, ne: [i64; 4], st: &mut u32, datas: &mut Vec<Vec<u8>>| {
            w.add_tensor(name, GgmlType::F32, ne);
            datas.push(tensor_bytes((ne[0] * ne[1] * ne[2] * ne[3]) as usize, st));
        };
    // an F32 SCALAR tensor (the gemma4a clamp bounds)
    let add_scalar = |w: &mut GgufWriter, name: &str, v: f32, datas: &mut Vec<Vec<u8>>| {
        w.add_tensor(name, GgmlType::F32, [1, 1, 1, 1]);
        datas.push(v.to_le_bytes().to_vec());
    };
    // common audio hparams
    let hp = |w: &mut GgufWriter,
              n_embd: i64,
              n_head: i64,
              n_ff: i64,
              n_layer: i64,
              eps: f32,
              n_mel: i64| {
        w.set_kv("clip.audio.embedding_length", Value::U32(n_embd as u32));
        w.set_kv("clip.audio.attention.head_count", Value::U32(n_head as u32));
        w.set_kv("clip.audio.feed_forward_length", Value::U32(n_ff as u32));
        w.set_kv("clip.audio.block_count", Value::U32(n_layer as u32));
        w.set_kv("clip.audio.projection_dim", Value::U32(proj_dim() as u32));
        w.set_kv("clip.audio.attention.layer_norm_epsilon", Value::F32(eps));
        w.set_kv("clip.audio.num_mel_bins", Value::U32(n_mel as u32));
    };

    match arch {
        "granite_speech" => {
            hp(&mut w, GS_N_EMBD, GS_N_HEAD, GS_N_FF, 2, 1e-5, GS_N_MEL);
            w.set_kv("clip.audio.chunk_size", Value::U32(25));
            w.set_kv("clip.audio.conv_kernel_size", Value::U32(9));
            w.set_kv("clip.audio.max_pos_emb", Value::U32(64));
            w.set_kv("clip.audio.projector.window_size", Value::U32(50));
            w.set_kv("clip.audio.projector.downsample_rate", Value::U32(25));
            w.set_kv("clip.audio.projector.head_count", Value::U32(4));

            add(
                &mut w,
                "a.input_projection.weight",
                [GS_N_EMBD, GS_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.input_projection.bias",
                [GS_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.enc_ctc_out.weight",
                [GS_N_EMBD, 8, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.enc_ctc_out.bias",
                [8, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.enc_ctc_out_mid.weight",
                [8, GS_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.enc_ctc_out_mid.bias",
                [GS_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );

            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // granite-speech.cpp:119 adds attn_out.bias unconditionally
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [GS_N_EMBD, GS_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.bias"),
                    [GS_N_FF, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [GS_N_FF, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.weight"),
                    [GS_N_EMBD, GS_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.bias"),
                    [GS_N_FF, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.weight"),
                    [GS_N_FF, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.weight"),
                    [GS_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.bias"),
                    [GS_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_dw.weight"),
                    [9, GS_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.weight"),
                    [GS_N_EMBD, 2 * GS_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.bias"),
                    [2 * GS_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.weight"),
                    [GS_D_CONV, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // the Shaw RPE table: [d_head, 2*max_pos_emb+1]
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_rel_pos_emb"),
                    [GS_N_EMBD / GS_N_HEAD, 129, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }

            // the QFormer projector block
            add(
                &mut w,
                "a.proj_query",
                [GS_N_EMBD, 2, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.proj_norm.weight",
                [GS_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.proj_norm.bias",
                [GS_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.proj_linear.weight",
                [GS_N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.proj_linear.bias",
                [proj_dim(), 1, 1, 1],
                &mut st,
                &mut datas,
            );
            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_q.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_q.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_k.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_k.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_v.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_v.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_out.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_out.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_norm.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.self_attn_norm.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_q.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_q.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_k.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_k.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_v.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_v.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_out.weight"),
                    [GS_N_EMBD, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_out.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_norm.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.cross_attn_norm.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_up.weight"),
                    [GS_N_EMBD, GS_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_up.bias"),
                    [GS_N_FF, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_down.weight"),
                    [GS_N_FF, GS_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_down.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_norm.weight"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.proj_blk.{il}.ffn_norm.bias"),
                    [GS_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
        }
        "gemma4a" => {
            hp(&mut w, G4A_N_EMBD, G4A_N_HEAD, G4A_N_FF, 2, 1e-6, G4A_N_MEL);

            // SSCP: conv1 [3,3,1,OC1] then conv2 [3,3,OC1,OC2], each with a
            // channel LayerNorm; the flatten is [ch2*freq2=128, time]
            add(
                &mut w,
                "a.conv1d.0.weight",
                [3, 3, 1, G4A_SSCP_OC[0]],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.0.bias",
                [1, 1, G4A_SSCP_OC[0], 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.0.norm.weight",
                [G4A_SSCP_OC[0], 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.1.weight",
                [3, 3, G4A_SSCP_OC[0], G4A_SSCP_OC[1]],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.1.bias",
                [1, 1, G4A_SSCP_OC[1], 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.1.norm.weight",
                [G4A_SSCP_OC[1], 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.input_projection.weight",
                [G4A_SSCP_OC[1] * 8, G4A_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.input_projection.bias",
                [G4A_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.pre_encode.out.weight",
                [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.pre_encode.out.bias",
                [G4A_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.soft_emb_norm.weight",
                [G4A_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.input_projection.weight",
                [G4A_N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );

            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.weight"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.per_dim_scale.weight"),
                    [G4A_N_EMBD / G4A_N_HEAD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.per_dim_k_scale.weight"),
                    [G4A_N_EMBD / G4A_N_HEAD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k_rel.weight"),
                    [G4A_N_EMBD, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // conv_norm / norm_conv are swapped in GGUF (clip.cpp:3290)
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.weight"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                ); // -> norm_conv
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.bias"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.weight"),
                    [G4A_N_EMBD, 2 * G4A_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.bias"),
                    [2 * G4A_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_dw.weight"),
                    [5, G4A_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.weight"),
                    [G4A_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                ); // -> conv_norm (post-dw RMS scale)
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.weight"),
                    [G4A_D_CONV, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.bias"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.weight"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.weight"),
                    [G4A_N_EMBD, G4A_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.weight"),
                    [G4A_N_FF, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [G4A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [G4A_N_EMBD, G4A_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [G4A_N_FF, G4A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
            // ClippableLinear bounds on one weight (clip.cpp:3315-3333)
            add_scalar(&mut w, "a.blk.0.attn_q.weight.input_max", 10.0, &mut datas);
            add_scalar(&mut w, "a.blk.0.attn_q.weight.input_min", -10.0, &mut datas);
            add_scalar(
                &mut w,
                "a.blk.0.attn_q.weight.output_max",
                1000.0,
                &mut datas,
            );
            add_scalar(
                &mut w,
                "a.blk.0.attn_q.weight.output_min",
                -1000.0,
                &mut datas,
            );
        }
        "parakeet" => {
            hp(&mut w, PK_N_EMBD, PK_N_HEAD, PK_N_FF, 2, 1e-5, PK_N_MEL);
            w.set_kv("clip.audio.subsampling_factor", Value::U32(8));
            w.set_kv("clip.audio.conv_kernel_size", Value::U32(9));

            // the pre-encode conv stack (clip.cpp:3399-3410) + F32 vector
            // tensors for the mel filterbank and window
            for (i, ne) in [
                (0usize, [3i64, 3, 1, 8]),
                (2, [3, 3, 1, 8]),
                (3, [1, 1, 8, 8]),
                (5, [3, 3, 1, 8]),
                (6, [1, 1, 8, 8]),
            ] {
                add(
                    &mut w,
                    &format!("a.conv1d.{i}.weight"),
                    ne,
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.conv1d.{i}.bias"),
                    [1, 1, ne[3], 1],
                    &mut st,
                    &mut datas,
                );
            }
            add(
                &mut w,
                "a.pre_encode.out.weight",
                [4 * 8, PK_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.pre_encode.out.bias",
                [PK_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            // a.mel_filters [n_mel * 257], a.window [400] — raw F32 vectors.
            // A real filterbank/window is non-negative: ln(sum + eps) goes
            // NaN on negative filter taps, so the LCG values are abs()'d.
            w.add_tensor("a.mel_filters", GgmlType::F32, [PK_N_MEL * 257, 1, 1, 1]);
            datas.push(
                (0..(PK_N_MEL * 257) as usize)
                    .map(|_| lcg(&mut st).abs())
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            );
            w.add_tensor("a.window", GgmlType::F32, [400, 1, 1, 1]);
            datas.push(
                (0..400usize)
                    .map(|_| lcg(&mut st).abs())
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            );
            add(
                &mut w,
                "mm.a.norm_pre.weight",
                [PK_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [PK_N_EMBD, PK_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [PK_N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            // mm.model.mlp.{0,1,3} are loaded (TN_MVLM_PROJ_MLP) but the
            // parakeet graph never reads them
            add(
                &mut w,
                "mm.model.mlp.0.weight",
                [1, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.model.mlp.1.weight",
                [1, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.model.mlp.3.weight",
                [1, 1, 1, 1],
                &mut st,
                &mut datas,
            );

            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [PK_N_EMBD, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [PK_N_EMBD, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [PK_N_EMBD, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [PK_N_EMBD, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [PK_N_EMBD, PK_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [PK_N_FF, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.weight"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.bias"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.weight"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.bias"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.weight"),
                    [PK_N_EMBD, PK_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.weight"),
                    [PK_N_FF, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.pos_bias_u"),
                    [PK_N_EMBD / PK_N_HEAD, PK_N_HEAD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.pos_bias_v"),
                    [PK_N_EMBD / PK_N_HEAD, PK_N_HEAD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.linear_pos.weight"),
                    [PK_N_EMBD, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.weight"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.bias"),
                    [PK_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.weight"),
                    [PK_N_EMBD, 2 * PK_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_dw.weight"),
                    [9, PK_D_CONV, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.weight"),
                    [PK_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.bias"),
                    [PK_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm_mean"),
                    [PK_D_CONV, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // variances are non-negative in a real model (sqrt follows)
                w.add_tensor(
                    &format!("a.blk.{il}.conv_norm_var"),
                    GgmlType::F32,
                    [PK_D_CONV, 1, 1, 1],
                );
                datas.push(
                    (0..PK_D_CONV as usize)
                        .map(|_| lcg(&mut st).abs())
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.weight"),
                    [PK_D_CONV, PK_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
        }
        "mimo_audio" => {
            hp(
                &mut w, MI_N_EMBD, MI_N_HEAD, MI_N_FF, MI_N_LAYER, 1e-5, MI_N_MEL,
            );
            w.set_kv("clip.audio.rvq.num_quantizers", Value::U32(MI_N_Q as u32));
            w.set_kv(
                "clip.audio.rvq.codebook_size",
                Value::Array(
                    ggml::GgufType::Int32,
                    vec![Value::I32(MI_BINS as i32), Value::I32(MI_BINS as i32)],
                ),
            );
            w.set_kv("clip.audio.window_size", Value::U32(16));
            w.set_kv(
                "clip.audio.wa_pattern_mode",
                Value::Array(
                    ggml::GgufType::Int32,
                    vec![Value::I32(-1), Value::I32(0), Value::I32(-1), Value::I32(0)],
                ),
            );
            w.set_kv("clip.audio.local_block_count", Value::U32(2));
            w.set_kv("clip.audio.local_group_size", Value::U32(8));

            add(
                &mut w,
                "a.conv1d.1.weight",
                [3, MI_N_MEL, MI_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.1.bias",
                [1, MI_C, 1, 1],
                &mut st,
                &mut datas,
            );
            // conv2 maps the stem channels to n_embd (the vit width)
            add(
                &mut w,
                "a.conv1d.2.weight",
                [3, MI_C, MI_N_EMBD, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.2.bias",
                [1, MI_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.downsample.conv.weight",
                [2, MI_N_EMBD, MI_N_EMBD, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.downsample.norm.weight",
                [MI_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.downsample.norm.bias",
                [MI_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.rvq.codebook.weight",
                [MI_N_EMBD, MI_BINS, MI_N_Q, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.code_embd.weight",
                [MI_N_EMBD, MI_BINS, MI_N_Q, 1],
                &mut st,
                &mut datas,
            );

            for il in 0..MI_N_LAYER {
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [MI_N_EMBD, MI_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [MI_N_FF, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
            add(
                &mut w,
                "a.post_ln.weight",
                [MI_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.post_ln.bias",
                [MI_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );

            // input_local_transformer + projection
            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_q.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_q.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_k.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_k.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_v.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_v.bias"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.attn_out.weight"),
                    [MI_N_EMBD, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.ffn_gate.weight"),
                    [MI_N_EMBD, MI_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.ffn_up.weight"),
                    [MI_N_EMBD, MI_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.ffn_down.weight"),
                    [MI_N_FF, MI_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.ln1.weight"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("mm.a.local_blk.{il}.ln2.weight"),
                    [MI_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
            add(
                &mut w,
                "mm.a.local_norm.weight",
                [MI_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [MI_N_EMBD * 8, MI_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [MI_N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
        }
        "qwen3tts_spkenc" => {
            hp(&mut w, QS_C, 4, 32, 3, 1e-5, QS_N_MEL);

            // stem TDNN k=5: 40 -> 128
            add(
                &mut w,
                "a.conv1d.0.weight",
                [5, QS_N_MEL, QS_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv1d.0.bias",
                [1, 1, QS_C, 1],
                &mut st,
                &mut datas,
            );

            // 3 SE-Res2Net blocks (bid 1..3)
            for bid in 1..=3i64 {
                add(
                    &mut w,
                    &format!("a.blk.{bid}.conv_pw1.weight"),
                    [1, QS_C, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.conv_pw1.bias"),
                    [1, 1, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.conv_pw2.weight"),
                    [1, QS_C, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.conv_pw2.bias"),
                    [1, 1, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.se_conv1.weight"),
                    [1, QS_C, 32, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.se_conv1.bias"),
                    [1, 1, 32, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.se_conv2.weight"),
                    [1, 32, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{bid}.se_conv2.bias"),
                    [1, 1, QS_C, 1],
                    &mut st,
                    &mut datas,
                );
                for xid in 0..7i64 {
                    add(
                        &mut w,
                        &format!("a.blk.{bid}.res2.{xid}.weight"),
                        [3, QS_CS, QS_CS, 1],
                        &mut st,
                        &mut datas,
                    );
                    add(
                        &mut w,
                        &format!("a.blk.{bid}.res2.{xid}.bias"),
                        [1, 1, QS_CS, 1],
                        &mut st,
                        &mut datas,
                    );
                }
            }

            // MFA + ASP + final FC
            add(
                &mut w,
                "a.conv_out.weight",
                [1, 3 * QS_C, 3 * QS_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.conv_out.bias",
                [1, 1, 3 * QS_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.asp_tdnn.weight",
                [1, 3 * QS_MFA, 256, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.asp_tdnn.bias",
                [1, 1, 256, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.asp_attn.weight",
                [1, 256, QS_MFA, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.asp_attn.bias",
                [1, 1, QS_MFA, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.fc.weight",
                [1, 2 * QS_MFA, proj_dim(), 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.fc.bias",
                [1, 1, proj_dim(), 1],
                &mut st,
                &mut datas,
            );
        }
        "pockettts_spkenc" => {
            hp(&mut w, PT_N_EMBD, PT_N_HEAD, PT_N_FF, PT_N_LAYER, 1e-5, 1);

            // the SEANet encoder: conv_in/out [8, 1/32, 32], stages ratios
            // [4,5,6] with res_conv1 [3,32,32], res_conv2 [1,32,32],
            // scale_conv [ratio,32,32]
            add(
                &mut w,
                "a.seanet.conv_in.weight",
                [8, 1, PT_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.seanet.conv_in.bias",
                [PT_C, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.seanet.conv_out.weight",
                [8, PT_C, PT_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.seanet.conv_out.bias",
                [PT_C, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            for (i, ratio) in [4i64, 5, 6].iter().enumerate() {
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.res_conv1.weight"),
                    [3, PT_C, PT_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.res_conv1.bias"),
                    [PT_C, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.res_conv2.weight"),
                    [1, PT_C, PT_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.res_conv2.bias"),
                    [PT_C, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.scale_conv.weight"),
                    [*ratio, PT_C, PT_C, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.seanet.blk.{i}.scale_conv.bias"),
                    [PT_C, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }

            // the mimi transformer (normal ViT layer table + layer scale)
            for il in 0..PT_N_LAYER {
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [PT_N_EMBD, PT_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [PT_N_EMBD, PT_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [PT_N_EMBD, PT_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [PT_N_EMBD, PT_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ls1.weight"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ls2.weight"),
                    [PT_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [PT_N_EMBD, PT_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [PT_N_FF, PT_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }

            // downsample conv (stride 16) + speaker projection
            add(
                &mut w,
                "a.downsample.conv.weight",
                [16, PT_C, PT_C, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.speaker_proj.weight",
                [PT_C, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
        }
        other => panic!("unknown synthetic arch {other}"),
    }

    let refs: Vec<&[u8]> = datas.iter().map(|d| d.as_slice()).collect();
    use std::io::Write;
    {
        let f = std::fs::File::create(&path).expect("create mmproj");
        let mut bw = std::io::BufWriter::new(f);
        w.write(&mut bw, &refs).expect("write mmproj");
        bw.flush().expect("flush");
    }
    path
}

/// read the MTMD_DEBUG_EMBEDDINGS dump format: [i32 n_tokens][i32 n_embd][f32 x n]
fn read_embd_dump(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read dump");
    assert!(bytes.len() >= 8);
    let n_tokens = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let n_embd = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 8 + n_tokens * n_embd * 4);
    bytes[8..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// bit-exact comparison against the reference dump when it exists (unless
/// MTMD_IGNORE_REF is set — the node-dump debugging mode)
fn compare_ref_dump(tag: &str, arch: &str, port: &[f32]) -> Option<()> {
    if std::env::var("MTMD_IGNORE_REF").is_ok() {
        return None;
    }
    let ref_path = format!("{OUT_DIR}/ref-{tag}-{arch}.bin");
    if std::fs::read(&ref_path).is_err() {
        return None;
    }
    let ref_embd = read_embd_dump(&ref_path);
    let bit_eq = ref_embd.len() == port.len()
        && ref_embd
            .iter()
            .zip(port.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !bit_eq {
        let mut max_abs = 0.0f64;
        let mut n_diff = 0usize;
        for (a, b) in ref_embd.iter().zip(port.iter()) {
            if a.to_bits() != b.to_bits() {
                n_diff += 1;
            }
            max_abs = max_abs.max((a - b).abs() as f64);
        }
        // granite_speech and parakeet are NOT bit-exact: both reduce to the
        // reference build's own vectorized ssm_conv (node-bisected to the
        // SSM_CONV node with bit-identical inputs — see PARITY.md's audio
        // round 4 section). Assert the tolerance band loudly instead of
        // pretending.
        if matches!(arch, "granite_speech" | "parakeet") && max_abs < 1e-3 {
            eprintln!(
                "{arch}/{tag}: near-exact but NOT bit-exact: {n_diff}/{} differ, max |delta| = {max_abs:e} (documented: the reference ssm_conv vectorization)",
                port.len()
            );
            return Some(());
        }
        panic!(
            "{arch}: NOT bit-exact vs reference: {n_diff}/{} values differ, max |delta| = {max_abs:e}",
            port.len()
        );
    }
    Some(())
}

fn fixture_wav(sr: i32) -> Vec<u8> {
    let name = if sr == 24000 {
        "mtmd-fixture-audio-24k.wav"
    } else {
        "mtmd-fixture-audio.wav"
    };
    std::fs::read(format!("{}/../../parity/{name}", OUT_DIR))
        .or_else(|_| std::fs::read(format!("parity/{name}")))
        .or_else(|_| std::fs::read(format!("../../parity/{name}")))
        .unwrap_or_else(|_| panic!("fixture {name} (run parity/gen_audio_fixture.py)"))
}

/// run the port's whole audio path on one synthetic arch and return
/// (n_tokens, embeddings, fa-off embeddings)
fn run_synthetic_arch(arch: &str, proj: ProjectorType) -> (u32, Vec<f32>, Vec<f32>) {
    let path = write_synthetic_mmproj(arch);
    eprintln!("synthetic {arch} mmproj: {path}");

    let params = ClipContextParams {
        n_threads: std::env::var("MMPROJ_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
        ..Default::default()
    };
    let mut clip = clip_init_from_file(&path, &params).expect("port loads the audio mmproj");
    assert_eq!(clip.model.modality, ClipModality::Audio);
    assert_eq!(clip.model.proj_type, proj);
    let want_embd = proj_dim();
    assert_eq!(clip.n_mmproj_embd() as i64, want_embd);

    let vgguf = Gguf::open(VOCAB_SPM).expect("vocab fixture");
    let vocab = Vocab::load(&vgguf).expect("vocab");
    let mut mctx = MtmdContext::init_from_file(
        &path,
        None,
        &MtmdContextParams {
            media_marker: "<|media|>".into(),
            ..MtmdContextParams::default()
        },
    )
    .expect("mtmd context on the audio mmproj");

    // sample rate per arch (clip.cpp:1791 / :1848 / :1879 / :1980)
    let (want_sr, want_tokens, want_nx, want_ny): (i32, u32, i64, i64) = match arch {
        // 16 kHz archs — 128 stacked frames from the 2.56 s fixture
        "granite_speech" => (16000, 6, 128, 64),
        "gemma4a" => (16000, 64, 255, 32),
        "parakeet" => (16000, 33, 257, 32),
        // 24 kHz archs — the rate-matched fixture
        "mimo_audio" => (24000, 8, 257, 64),
        "qwen3tts_spkenc" => (24000, 1, 240, 40),
        "pockettts_spkenc" => (24000, 32, 61440, 1),
        other => panic!("{other}"),
    };
    assert_eq!(mctx.audio_sample_rate(), want_sr);

    let wav = fixture_wav(want_sr);
    let chunks = mctx
        .tokenize_audio(&vocab, &wav, false)
        .expect("tokenize_audio");
    let audio_chunk = chunks
        .iter()
        .find(|c| matches!(c, MtmdChunk::Audio(_)))
        .unwrap_or_else(|| panic!("{arch}: no audio chunk"));
    let MtmdChunk::Audio(a) = audio_chunk else {
        unreachable!()
    };

    assert_eq!(a.n_tokens, want_tokens, "{arch} token count");
    assert_eq!(a.batch_f32.entries.len(), 1);
    assert!(a.batch_f32.is_audio);
    assert_eq!(
        a.batch_f32.entries[0].nx as i64, want_nx,
        "{arch} frame count"
    );
    assert_eq!(
        a.batch_f32.entries[0].ny as i64, want_ny,
        "{arch} mel count"
    );

    // the mel chunk, for the reference node-dump probe (ref_clip_graph_dump)
    if let Ok(dir) = std::env::var("MTMD_DUMP_MEL_DIR") {
        let e = &a.batch_f32.entries[0];
        let mut bytes = Vec::with_capacity(8 + e.buf.len() * 4);
        bytes.extend_from_slice(&e.nx.to_le_bytes());
        bytes.extend_from_slice(&e.ny.to_le_bytes());
        for v in &e.buf {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{dir}/mel-{arch}.bin"), bytes).unwrap();
    }
    // the port's raw node dump, keyed by arch (MTMD_DUMP_NODES_ARCH selects
    // which arch gets MTMD_DUMP_NODES_BIN; the fa-off encode wins)
    let dump_this = std::env::var("MTMD_DUMP_NODES_ARCH")
        .map(|a_| a_ == arch)
        .unwrap_or(false);
    if dump_this {
        if let Ok(path) = std::env::var("MTMD_DUMP_NODES_BIN") {
            std::env::set_var("MTMD_DEBUG_NODES_BIN", &path);
        }
    }

    // ---- the encoder graph over the chunk --------------------------------
    let embd = clip.audio_batch_encode(&a.batch_f32).expect("encode");
    assert_eq!(embd.len(), want_tokens as usize * want_embd as usize);
    assert!(
        embd.iter().all(|v| v.is_finite()),
        "{arch}: embeddings must be finite"
    );
    assert!(
        embd.iter().any(|v| *v != 0.0),
        "{arch}: embeddings must not be all zero"
    );

    // ---- the same graph with flash attention off (reference -fa off) ------
    let params_off = ClipContextParams {
        flash_attn_type: ClipFlashAttn::Disabled,
        n_threads: std::env::var("MMPROJ_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
        ..Default::default()
    };
    let mut clip_off =
        clip_init_from_file(&path, &params_off).expect("port loads the audio mmproj (fa off)");
    let embd_off = clip_off
        .audio_batch_encode(&a.batch_f32)
        .expect("encode fa off");
    assert_eq!(embd_off.len(), want_tokens as usize * want_embd as usize);
    assert!(
        embd_off.iter().all(|v| v.is_finite()),
        "{arch}: fa-off embeddings must be finite"
    );
    let dump_off = format!("{OUT_DIR}/port-faoff-{arch}.bin");
    llama::clip::write_embedding_dump(&dump_off, &embd_off, a.n_tokens as i32, want_embd as i32)
        .unwrap();
    if dump_this {
        std::env::remove_var("MTMD_DEBUG_NODES_BIN");
    }

    (a.n_tokens, embd, embd_off)
}

#[test]
fn audio_mmproj_loads_and_tokenizes() {
    for (arch, proj) in FAMILY {
        let (n_tokens, embd, embd_off) = run_synthetic_arch(arch, *proj);
        let dump = format!("{OUT_DIR}/port-default-{arch}.bin");
        let want_embd = proj_dim();
        llama::clip::write_embedding_dump(&dump, &embd, n_tokens as i32, want_embd as i32).unwrap();
        eprintln!(
            "{arch}: wrote {dump} ({n_tokens} tokens x {} embd)",
            proj_dim()
        );
        match compare_ref_dump("default", arch, &embd) {
            Some(()) => eprintln!("{arch}: reference dump matches (flash attn on)"),
            None => eprintln!(
                "{arch}: no reference dump ({OUT_DIR}/ref-default-{arch}.bin), shape check only"
            ),
        }
        match compare_ref_dump("faoff", arch, &embd_off) {
            Some(()) => eprintln!("{arch}: reference dump matches (-fa off)"),
            None => {}
        }
    }
}
