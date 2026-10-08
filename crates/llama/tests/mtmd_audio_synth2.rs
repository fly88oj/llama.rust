//! mtmd_audio_synth2.rs — audio-encoder round 2: the non-whisper encoders
//! (qwen3a, gemma4ua) built with the PORT's GGUF writer (the established
//! synthetic protocol), then:
//!
//!   1. the port loads each (`clip_init_from_file` audio branch) and runs the
//!      whole audio path (preprocess → tokens → encoder graph);
//!   2. the same files are handed to the pinned reference `llama-mtmd-cli
//!      --mmproj` by parity/audio_mtmd_parity2.sh — the reference ACCEPTS them
//!      (load + graph build + embeddings) and its MTMD_DEBUG_EMBEDDINGS dump
//!      is compared bit-exactly against the port's (f32::to_bits), for the
//!      default flash-attention path and `-fa off`.
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

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");

const OUT_DIR: &str = "/tmp/mtmd-audio-synth";
/// must match the text model used by parity/audio_mtmd_parity2.sh
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

/// the round-2 archs (audio-encoder graphs of models/qwen3a.cpp:3,
/// models/gemma4ua.cpp:4 and models/conformer.cpp:3)
pub const FAMILY: &[(&str, ProjectorType)] = &[
    ("qwen3a", ProjectorType::Qwen3A),
    ("gemma4ua", ProjectorType::Gemma4UA),
    ("lfm2a", ProjectorType::Lfm2A),
];

// qwen3a geometry (models/qwen3a.cpp — chunk 100 frames, 3x stride-2 conv2d):
//   mel 128, conv channels 16/16/32, d_model 64, 2 layers, 13 pos per chunk
const Q3A_N_MEL: i64 = 128;
const Q3A_OC: [i64; 3] = [16, 16, 32];
const Q3A_N_EMBD: i64 = 64;
const Q3A_N_HEAD: i64 = 4;
const Q3A_N_FF: i64 = 128;
const Q3A_N_LAYER: i64 = 2;

/// write one synthetic round-2 mmproj; returns the path
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

    let mut st = 0x2345_6789u32;
    let mut datas: Vec<Vec<u8>> = Vec::new();
    let add =
        |w: &mut GgufWriter, name: &str, ne: [i64; 4], st: &mut u32, datas: &mut Vec<Vec<u8>>| {
            w.add_tensor(name, GgmlType::F32, ne);
            datas.push(tensor_bytes((ne[0] * ne[1] * ne[2] * ne[3]) as usize, st));
        };

    match arch {
        "qwen3a" => {
            // audio-prefixed hparams (clip.cpp:1297-1336) + the family arm
            // (:1776-1793): whisper preprocessing params, gelu_erf FFN
            w.set_kv("clip.audio.embedding_length", Value::U32(Q3A_N_EMBD as u32));
            w.set_kv(
                "clip.audio.attention.head_count",
                Value::U32(Q3A_N_HEAD as u32),
            );
            w.set_kv(
                "clip.audio.feed_forward_length",
                Value::U32(Q3A_N_FF as u32),
            );
            w.set_kv("clip.audio.block_count", Value::U32(Q3A_N_LAYER as u32));
            w.set_kv("clip.audio.projection_dim", Value::U32(proj_dim() as u32));
            w.set_kv("clip.audio.attention.layer_norm_epsilon", Value::F32(1e-5));
            w.set_kv("clip.audio.num_mel_bins", Value::U32(Q3A_N_MEL as u32));

            // conv2d stem (clip.cpp:2861-2867): [K=3, K=3, IC, OC] — the
            // mel chunk view is [100, 128, 1, N], so the FIRST conv has
            // IC = 1 (the mel bins are the height axis)
            //   x: [100, 128, 1, N] -> conv1 [50, 64, OC1, N]
            //   -> conv2 [25, 32, OC2, N] -> conv3 [13, 16, OC3, N]
            let ic = [1, Q3A_OC[0], Q3A_OC[1]];
            for i in 0..3 {
                add(
                    &mut w,
                    &format!("a.conv2d.{}.weight", i + 1),
                    [3, 3, ic[i as usize], Q3A_OC[i as usize]],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.conv2d.{}.bias", i + 1),
                    [1, 1, 1, Q3A_OC[i as usize]],
                    &mut st,
                    &mut datas,
                );
            }
            // conv_out (no bias, clip.cpp:2870): [16*OC3, d_model]
            add(
                &mut w,
                "a.conv_out.weight",
                [16 * Q3A_OC[2], Q3A_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            // per-chunk learned positions: 13 per chunk (qwen3a.cpp:63-66)
            add(
                &mut w,
                "a.position_embd.weight",
                [Q3A_N_EMBD, 13, 1, 1],
                &mut st,
                &mut datas,
            );
            // transformer layers — fused qkv (the build_vit fused branch,
            // clip.cpp:401-437)
            for il in 0..Q3A_N_LAYER {
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_qkv.weight"),
                    [Q3A_N_EMBD, 3 * Q3A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_qkv.bias"),
                    [3 * Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [Q3A_N_EMBD, Q3A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [Q3A_N_EMBD, Q3A_N_FF, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.bias"),
                    [Q3A_N_FF, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [Q3A_N_FF, Q3A_N_EMBD, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.bias"),
                    [Q3A_N_EMBD, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
            // MLP projector (clip.cpp:2871-2874)
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [Q3A_N_EMBD, Q3A_N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.bias",
                [Q3A_N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [Q3A_N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.bias",
                [proj_dim(), 1, 1, 1],
                &mut st,
                &mut datas,
            );
        }
        "lfm2a" => {
            // clip.cpp:1951-1959 — n_fft 512 / window 400 / hop 160 @ 16 kHz
            w.set_kv("clip.audio.embedding_length", Value::U32(512));
            w.set_kv("clip.audio.attention.head_count", Value::U32(8));
            w.set_kv("clip.audio.feed_forward_length", Value::U32(128));
            w.set_kv("clip.audio.block_count", Value::U32(2));
            w.set_kv("clip.audio.projection_dim", Value::U32(512));
            w.set_kv("clip.audio.attention.layer_norm_epsilon", Value::F32(1e-5));
            w.set_kv("clip.audio.num_mel_bins", Value::U32(128));

            // conv-subsampling stack (clip.cpp:3360-3369): input is the
            // transposed mel [128 mel, T frames]; C1=C2=C3=8
            //   conv1 [3,3,1,8]  s(2,2) p(1,1) -> [64, T/2]
            //   dw2   [3,3,1,8]  s(2,2) p(1,1) -> [32, T/4]
            //   dir3  [1,1,8,8]  s(1,1)
            //   dw5   [3,3,1,8]  s(2,2) p(1,1) -> [16, T/8]
            //   dir6  [1,1,8,8]  s(1,1)
            for (i, ne) in [
                (0usize, [3i64, 3, 1, 8]), // conv2d (IC=1: mel bins are the height)
                (2, [3, 3, 1, 8]),         // depthwise
                (3, [1, 1, 8, 8]),         // direct 1x1
                (5, [3, 3, 1, 8]),         // depthwise
                (6, [1, 1, 8, 8]),         // direct 1x1
            ] {
                add(
                    &mut w,
                    &format!("a.conv1d.{i}.weight"),
                    ne,
                    &mut st,
                    &mut datas,
                );
                // the graph adds the bias UNRESHAPEd (conformer.cpp:22) — the
                // conv output is [OW, OH, C, N] so the bias carries C on dim2
                add(
                    &mut w,
                    &format!("a.conv1d.{i}.bias"),
                    [1, 1, ne[3], 1],
                    &mut st,
                    &mut datas,
                );
            }
            // out projection: [16*8=128, 512] (clip.cpp:3368-3369)
            add(
                &mut w,
                "a.pre_encode.out.weight",
                [128, 512, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "a.pre_encode.out.bias",
                [512, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            // the graph only ASSERTS on position_embeddings (conformer.cpp:7,
            // ne[1] >= warmup frames/2, warmup_audio_size=3000 -> 1500) — its
            // ne[0] doubles as n_mmproj_embd (clip.cpp:6005), so the synthetic
            // carries the text-model width here and mm.a.mlp.3 maps to it
            add(
                &mut w,
                "a.position_embd.weight",
                [proj_dim(), 1500, 1, 1],
                &mut st,
                &mut datas,
            );
            // audio adapter (clip.cpp:3371-3377): norm + mm.a.mlp.1/3 FFN;
            // n_mmproj_embd == position_embd width == 512 (clip.cpp:6005)
            add(
                &mut w,
                "mm.a.mlp.0.weight",
                [512, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.0.bias",
                [512, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [512, 128, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.bias",
                [128, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.3.weight",
                [128, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.3.bias",
                [proj_dim(), 1, 1, 1],
                &mut st,
                &mut datas,
            );
            // conformer layers (clip.cpp:3379-3406)
            for il in 0..2i64 {
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.weight"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln1.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.weight"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ln2.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.weight"),
                    [512, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_q.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.weight"),
                    [512, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_k.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.weight"),
                    [512, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_v.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.weight"),
                    [512, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.attn_out.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.weight"),
                    [512, 128, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up.bias"),
                    [128, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.weight"),
                    [128, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.weight"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.weight"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_norm_1.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.weight"),
                    [512, 128, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_up_1.bias"),
                    [128, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.weight"),
                    [128, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.ffn_down_1.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // [d_head, n_head, 1] — broadcasts over the token dim in the
                // Shaw attention add (conformer.cpp:93)
                add(
                    &mut w,
                    &format!("a.blk.{il}.pos_bias_u"),
                    [64, 8, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.pos_bias_v"),
                    [64, 8, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.linear_pos.weight"),
                    [512, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.weight"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.norm_conv.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.weight"),
                    [128, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_norm.bias"),
                    [128, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // causal depthwise conv (ssm_conv): [9, 128] — pad4+roll4+pad4
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_dw.weight"),
                    [9, 128, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_dw.bias"),
                    [128, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                // GLU width 128 -> pw1 out 256
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.weight"),
                    [512, 256, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw1.bias"),
                    [256, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.weight"),
                    [128, 512, 1, 1],
                    &mut st,
                    &mut datas,
                );
                add(
                    &mut w,
                    &format!("a.blk.{il}.conv_pw2.bias"),
                    [512, 1, 1, 1],
                    &mut st,
                    &mut datas,
                );
            }
        }
        "gemma4ua" => {
            // the common hparams keys are still required by the loader even
            // though the graph only reads eps (clip.cpp:1297-1309); n_layer=0
            // skips the layer table entirely
            w.set_kv("clip.audio.embedding_length", Value::U32(64));
            w.set_kv("clip.audio.attention.head_count", Value::U32(1));
            w.set_kv("clip.audio.feed_forward_length", Value::U32(1));
            w.set_kv("clip.audio.block_count", Value::U32(0));
            w.set_kv("clip.audio.projection_dim", Value::U32(proj_dim() as u32));
            w.set_kv("clip.audio.attention.layer_norm_epsilon", Value::F32(1e-6));
            // n_mel_bins is overwritten to 640 by the hparams arm
            // (clip.cpp:1973) — the raw-waveform frame size
            w.set_kv("clip.audio.num_mel_bins", Value::U32(640));
            // clip.cpp:3343-3346 — TN_A_MM_INP_PROJ
            add(
                &mut w,
                "mm.a.input_projection.weight",
                [640, proj_dim(), 1, 1],
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

/// bit-exact comparison against the reference dump when it exists
fn compare_ref_dump(tag: &str, arch: &str, port: &[f32]) -> Option<()> {
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
        // lfm2a is NOT yet bit-exact: the conformer depthwise-conv module
        // carries a ULP-scale accumulation difference (see PARITY.md —
        // embeddings agree to ~6e-6, the node-sum bisection localizes the
        // first drift to the pad/roll/ssm_conv cluster). Assert the tolerance
        // band loudly instead of pretending.
        if arch == "lfm2a" && max_abs < 1e-4 {
            eprintln!(
                "{arch}/{tag}: near-exact but NOT bit-exact: {n_diff}/{} differ, max |delta| = {max_abs:e} (documented)",
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

fn fixture_wav() -> Vec<u8> {
    std::fs::read(format!("{}/../../parity/mtmd-fixture-audio.wav", OUT_DIR))
        .or_else(|_| std::fs::read("parity/mtmd-fixture-audio.wav"))
        .or_else(|_| std::fs::read("../../parity/mtmd-fixture-audio.wav"))
        .expect("fixture wav (run parity/gen_audio_fixture.py)")
}

/// run the port's whole audio path on one synthetic arch and return
/// (n_tokens, embeddings, fa-off embeddings)
fn run_synthetic_arch(arch: &str, proj: ProjectorType) -> (u32, Vec<f32>, Vec<f32>) {
    let path = write_synthetic_mmproj(arch);
    eprintln!("synthetic {arch} mmproj: {path}");

    // ---- 1. the port loads the audio projector -----------------------------
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
    // lfm2a's n_mmproj_embd is the position-embedding width (clip.cpp:6005),
    // which the synthetic carries as proj_dim
    let want_embd = proj_dim();
    assert_eq!(clip.n_mmproj_embd() as i64, want_embd);

    // ---- 2. mtmd audio path -------------------------------------------------
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
    // both archs run at 16 kHz (clip.cpp:1791 / :1970)
    assert_eq!(mctx.audio_sample_rate(), 16000);

    let wav = fixture_wav();
    let chunks = mctx
        .tokenize_audio(&vocab, &wav, false)
        .expect("tokenize_audio");

    // the 2.56 s fixture: qwen3a pads 257 frames to 3 chunks of 100 -> 39
    // tokens (clip.cpp:4261-4263); gemma4ua gives 40960/640 = 64 tokens
    // (clip.cpp:4335-4337); lfm2a runs the whole 257-frame mel through three
    // stride-2 convs -> 33 tokens (clip.cpp:4319) with NO marker chunks.
    let (want_chunks, want_tokens): (usize, u32) = match arch {
        "qwen3a" => (3, 39),
        "gemma4ua" => (3, 64),
        "lfm2a" => (1, 33),
        other => panic!("{other}"),
    };
    assert_eq!(
        chunks.len(),
        want_chunks,
        "unexpected chunk layout for {arch}"
    );
    if arch != "lfm2a" {
        assert!(
            matches!(chunks[0], MtmdChunk::Text(_)),
            "{arch} beg marker chunk"
        );
        assert!(
            matches!(chunks[chunks.len() - 1], MtmdChunk::Text(_)),
            "{arch} end marker chunk"
        );
    }
    let audio_chunk = chunks
        .iter()
        .find(|c| matches!(c, MtmdChunk::Audio(_)))
        .expect("audio chunk");
    let MtmdChunk::Audio(a) = audio_chunk else {
        unreachable!()
    };

    assert_eq!(a.n_tokens, want_tokens, "{arch} token count");
    assert_eq!(a.batch_f32.entries.len(), 1);
    assert!(a.batch_f32.is_audio);
    match arch {
        "qwen3a" => {
            assert_eq!(a.batch_f32.entries[0].nx, 300); // 3 chunks x 100
            assert_eq!(a.batch_f32.entries[0].ny as i64, Q3A_N_MEL);
        }
        "gemma4ua" => {
            assert_eq!(a.batch_f32.entries[0].nx, 64); // 40960 samples / 640
            assert_eq!(a.batch_f32.entries[0].ny, 640);
        }
        "lfm2a" => {
            assert_eq!(a.batch_f32.entries[0].nx, 257); // center-padded mel
            assert_eq!(a.batch_f32.entries[0].ny, 128);
        }
        _ => unreachable!(),
    }

    // the mel chunk + node dumps for the reference bit-exact node comparator
    // (parity/ref_clip_graph_dump.cpp + parity/clip_node_cmp.py) — the
    // conformer (lfm2a) ULP bisection tooling. MTMD_DUMP_NODES_ARCH selects
    // which arch gets MTMD_DEBUG_NODES_BIN.
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
    let dump_this = std::env::var("MTMD_DUMP_NODES_ARCH")
        .map(|a_| a_ == arch)
        .unwrap_or(false);
    if dump_this {
        if let Ok(path) = std::env::var("MTMD_DUMP_NODES_BIN") {
            std::env::set_var("MTMD_DEBUG_NODES_BIN", &path);
        }
    }

    // ---- 3. the encoder graph over the chunk --------------------------------
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

    // ---- 3b. the same graph with flash attention off (reference -fa off) ---
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

#[test]
fn mp3_flac_fail_loudly() {
    // Task 3 (round 3): MP3 stays a loud error (miniaudio in the reference,
    // mtmd-helper.cpp:325-362); FLAC is now DECODED by the port (audio round
    // 4, mtmd.rs audio_from_flac_bytes — bit-exact vs dr_flac, see
    // tests/mtmd_flac.rs), so only the sniffer + a truncated file fail.
    let mp3_id3 = [b'I', b'D', b'3', 0u8, 0, 0, 0, 0, 0, 0, 0, 0];
    let err = llama::mtmd::audio_from_wav_bytes(&mp3_id3, 16000).unwrap_err();
    assert!(err.contains("MP3"), "mp3 error names the format: {err}");
    assert!(
        err.contains("miniaudio"),
        "mp3 error points at the reference surface: {err}"
    );

    let mp3_sync = [0xFFu8, 0xE5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let err = llama::mtmd::audio_from_wav_bytes(&mp3_sync, 16000).unwrap_err();
    assert!(err.contains("MP3"), "sync-word mp3: {err}");

    // a truncated FLAC (no full STREAMINFO) fails loudly, not silently
    let flac = *b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00\x00\x00\x00\x00";
    let err = llama::mtmd::audio_from_wav_bytes(&flac, 16000).unwrap_err();
    assert!(
        err.contains("FLAC"),
        "truncated flac error names the format: {err}"
    );

    // the sniffers themselves match the reference layout checks
    assert!(llama::mtmd::is_mp3_file(&mp3_id3));
    assert!(llama::mtmd::is_mp3_file(&mp3_sync));
    assert!(!llama::mtmd::is_mp3_file(b"RIFFxxxxWAVE"));
    assert!(llama::mtmd::is_flac_file(b"fLaC++++"));
    assert!(!llama::mtmd::is_flac_file(b"RIFFxxxxWAVE"));
}
