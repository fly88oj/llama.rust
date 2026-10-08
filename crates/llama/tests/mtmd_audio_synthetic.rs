//! mtmd_audio_synthetic.rs — synthetic audio projectors (the whisper-enc
//! family: qwen2a/ultravox/voxtral/meralion/glma/musicflamingo) built with the
//! PORT's GGUF writer (the established synthetic protocol, see
//! tests/arch_batch*_e2e.rs), then:
//!
//!   1. the port loads each (`clip_init_from_file` audio branch) and the
//!      loaded hparams match what the metadata says (clip.cpp:1297-1336,
//!      :1774-1793);
//!   2. `mtmd_tokenize_audio` on parity/mtmd-fixture-audio.wav produces the
//!      mel chunks wrapped per-arch (mtmd.cpp:1556-1651, clip.cpp:4237-4276);
//!   3. `audio_batch_encode` runs the whisper-enc graph
//!      (models/whisper-enc.cpp:3-137) over each chunk and the output token
//!      count matches `clip_n_output_tokens`;
//!   4. the same files are handed to the pinned reference `llama-mtmd-cli
//!      --mmproj` by parity/audio_mtmd_parity.sh — the reference ACCEPTS them
//!      (load + graph build + embeddings) and its MTMD_DEBUG_EMBEDDINGS dump
//!      is compared bit-exactly against the port's (f32::to_bits).
//!
//! The generated files are deterministic (seeded LCG weights).

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, Value};
use llama::clip::{clip_init_from_file, ClipContextParams, ClipModality, ProjectorType};
use llama::mtmd::{MtmdChunk, MtmdContext, MtmdContextParams};
use llama::vocab::Vocab;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";

const OUT_DIR: &str = "/tmp/mtmd-audio-synth";
const N_MEL: i64 = 80;
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const N_FF: i64 = 128;
const N_LAYER: i64 = 2;
/// must match the text model used by parity/audio_mtmd_parity.sh
/// (qwen2.5-0.5b: n_embd 896) — the mmproj output feeds its embedding row.
/// `MMPROJ_PROJ_DIM` overrides it for manual end-to-end runs with another
/// text model (e.g. 5120 for the local qwen35); the committed parity record
/// always uses the 896 default.
const PROJ_DIM: i64 = 896;

fn proj_dim() -> i64 {
    std::env::var("MMPROJ_PROJ_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PROJ_DIM)
}
/// whisper position budget: 3000 mel frames -> 1500 tokens (whisper-enc.cpp:6)
const N_POS: i64 = 1500;
/// StackAudioFrames factor of ultravox/voxtral/meralion/glma
const STACK: i64 = 2;
/// meralion's hidden width (any legal value; fixed for the record)
const MERALION_HID: i64 = 96;

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

/// the whisper-enc family archs the synthetic protocol covers
pub const FAMILY: &[(&str, ProjectorType)] = &[
    ("qwen2a", ProjectorType::Qwen2A),
    ("ultravox", ProjectorType::Ultravox),
    ("voxtral", ProjectorType::Voxtral),
    ("meralion", ProjectorType::Meralion),
    ("glma", ProjectorType::Glma),
    ("musicflamingo", ProjectorType::MusicFlamingo),
];

/// write one synthetic whisper-family mmproj; returns the path. `arch` is the
/// projector_type string (FAMILY's first column).
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

    // modality + projector (clip-impl.h:33-34, KEY_PROJ_TYPE)
    w.set_kv("clip.has_vision_encoder", Value::Bool(false));
    w.set_kv("clip.has_audio_encoder", Value::Bool(true));
    w.set_kv("clip.projector_type", Value::String(arch.to_string()));

    // audio-prefixed hparams (clip.cpp:1297-1336)
    w.set_kv("clip.audio.embedding_length", Value::U32(N_EMBD as u32));
    w.set_kv("clip.audio.attention.head_count", Value::U32(N_HEAD as u32));
    w.set_kv("clip.audio.feed_forward_length", Value::U32(N_FF as u32));
    w.set_kv("clip.audio.block_count", Value::U32(N_LAYER as u32));
    w.set_kv("clip.audio.projection_dim", Value::U32(proj_dim() as u32));
    w.set_kv("clip.audio.attention.layer_norm_epsilon", Value::F32(1e-5));
    w.set_kv("clip.audio.num_mel_bins", Value::U32(N_MEL as u32));
    // clip.audio.projector.stack_factor (clip.cpp:1782-1786) — required for
    // ultravox/voxtral/meralion/glma, optional (and absent) for qwen2a
    if matches!(arch, "ultravox" | "voxtral" | "meralion" | "glma") {
        w.set_kv(
            "clip.audio.projector.stack_factor",
            Value::U32(STACK as u32),
        );
    }

    let mut st = 0x1234_5678u32;

    let mut datas: Vec<Vec<u8>> = Vec::new();
    let add =
        |w: &mut GgufWriter, name: &str, ne: [i64; 4], st: &mut u32, datas: &mut Vec<Vec<u8>>| {
            let idx = w.add_tensor(name, GgmlType::F32, ne);
            let _ = idx;
            datas.push(tensor_bytes(
                (ne[0] as usize) * (ne[1] as usize) * (ne[2] as usize) * (ne[3] as usize),
                st,
            ));
        };

    // whisper-enc conv1d pair (clip.cpp:2852-2857): weight [K, IC, OC]; the
    // bias is [1, OC] — ggml_add broadcasts dim-wise (ggml.c:1589) and the
    // conv output is [OL, OC, N], so a plain [OC] vector would not divide OL
    add(
        &mut w,
        "a.conv1d.1.weight",
        [3, N_MEL, N_EMBD, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.conv1d.1.bias",
        [1, N_EMBD, 1, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.conv1d.2.weight",
        [3, N_EMBD, N_EMBD, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.conv1d.2.bias",
        [1, N_EMBD, 1, 1],
        &mut st,
        &mut datas,
    );
    // learned positions (whisper-enc.cpp:34-38)
    add(
        &mut w,
        "a.position_embd.weight",
        [N_EMBD, N_POS, 1, 1],
        &mut st,
        &mut datas,
    );
    // pre/post layernorm (build_vit, clip.cpp:337-360/:566-570)
    add(
        &mut w,
        "a.pre_ln.weight",
        [N_EMBD, 1, 1, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.pre_ln.bias",
        [N_EMBD, 1, 1, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.post_ln.weight",
        [N_EMBD, 1, 1, 1],
        &mut st,
        &mut datas,
    );
    add(
        &mut w,
        "a.post_ln.bias",
        [N_EMBD, 1, 1, 1],
        &mut st,
        &mut datas,
    );

    for il in 0..N_LAYER {
        add(
            &mut w,
            &format!("a.blk.{il}.ln1.weight"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ln1.bias"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ln2.weight"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ln2.bias"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        // k has no bias (whisper-enc.cpp:32)
        add(
            &mut w,
            &format!("a.blk.{il}.attn_k.weight"),
            [N_EMBD, N_EMBD, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.attn_q.weight"),
            [N_EMBD, N_EMBD, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.attn_q.bias"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.attn_v.weight"),
            [N_EMBD, N_EMBD, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.attn_v.bias"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.attn_out.weight"),
            [N_EMBD, N_EMBD, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ffn_up.weight"),
            [N_EMBD, N_FF, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ffn_up.bias"),
            [N_FF, 1, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ffn_down.weight"),
            [N_FF, N_EMBD, 1, 1],
            &mut st,
            &mut datas,
        );
        add(
            &mut w,
            &format!("a.blk.{il}.ffn_down.bias"),
            [N_EMBD, 1, 1, 1],
            &mut st,
            &mut datas,
        );
    }

    // per-arch projector tensors (the loader arms, clip.cpp:2817-2862/:3136-3184)
    match arch {
        "qwen2a" => {
            // clip.cpp:2858-2859
            add(
                &mut w,
                "mm.a.fc.weight",
                [N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.fc.bias",
                [proj_dim(), 1, 1, 1],
                &mut st,
                &mut datas,
            );
        }
        "ultravox" => {
            // ffn in [n_embd*stack -> 2*n_embd] (swiglu halves), ffn out
            // [n_embd -> proj], rms norm weights (no biases)
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [N_EMBD * STACK, 2 * N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.norm_pre.weight",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.norm_mid.weight",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
        }
        "voxtral" => {
            // plain gelu_erf FFN on top of the stacked frames, no biases
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [N_EMBD * STACK, N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [N_EMBD, proj_dim(), 1, 1],
                &mut st,
                &mut datas,
            );
        }
        "musicflamingo" => {
            // no stack: [n_embd -> n_embd] -> proj, with biases
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [N_EMBD, N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.bias",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [N_EMBD, proj_dim(), 1, 1],
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
        "meralion" => {
            // ln + linear0 (frame compression) + silu, gate/pool GLU, out
            add(
                &mut w,
                "mm.a.norm_pre.weight",
                [N_EMBD * STACK, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.norm_pre.bias",
                [N_EMBD * STACK, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.0.weight",
                [N_EMBD * STACK, MERALION_HID, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.0.bias",
                [MERALION_HID, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [MERALION_HID, MERALION_HID, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.bias",
                [MERALION_HID, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [MERALION_HID, MERALION_HID, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.bias",
                [MERALION_HID, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.3.weight",
                [MERALION_HID, proj_dim(), 1, 1],
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
        }
        "glma" => {
            // the norm_pre of glma runs BEFORE build_stack (whisper-enc.cpp:
            // 121-124), so it is n_embd-wide, not n_embd*stack
            add(
                &mut w,
                "mm.a.norm_pre.weight",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.norm_pre.bias",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.weight",
                [N_EMBD * STACK, N_EMBD, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.1.bias",
                [N_EMBD, 1, 1, 1],
                &mut st,
                &mut datas,
            );
            add(
                &mut w,
                "mm.a.mlp.2.weight",
                [N_EMBD, proj_dim(), 1, 1],
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
            // BOI/EOI embeddings in the projector output space (the concat of
            // whisper-enc.cpp:126-127 is over dim 1)
            add(&mut w, "v.boi", [proj_dim(), 1, 1, 1], &mut st, &mut datas);
            add(&mut w, "v.eoi", [proj_dim(), 1, 1, 1], &mut st, &mut datas);
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
/// (parity/audio_mtmd_parity.sh writes ref-<tag>-<arch>.bin); None = no
/// reference present (shape check only)
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
        // report the band for the record
        let mut max_abs = 0.0f64;
        let mut n_diff = 0usize;
        for (a, b) in ref_embd.iter().zip(port.iter()) {
            if a.to_bits() != b.to_bits() {
                n_diff += 1;
            }
            max_abs = max_abs.max((a - b).abs() as f64);
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
    let params = ClipContextParams::default();
    let mut clip = clip_init_from_file(&path, &params).expect("port loads the audio mmproj");
    assert_eq!(clip.model.modality, ClipModality::Audio);
    assert_eq!(clip.model.proj_type, proj);
    let hp = clip.hparams().clone();
    assert_eq!(hp.n_mel_bins as i64, N_MEL);
    assert_eq!(hp.n_embd as i64, N_EMBD);
    assert_eq!(hp.n_layer as i64, N_LAYER);
    // whisper preprocessing defaults (clip.cpp:1791-1795)
    assert_eq!(hp.audio_sample_rate, 16000);
    assert_eq!(hp.audio_n_fft, 400);
    assert_eq!(hp.audio_window_len, 400);
    assert_eq!(hp.audio_hop_len, 160);
    assert_eq!(hp.audio_chunk_len, 30);
    assert_eq!(hp.ffn_op, llama::clip::FfnOp::GeluErf);
    let want_stack: i64 = if matches!(arch, "ultravox" | "voxtral" | "meralion" | "glma") {
        STACK
    } else {
        0
    };
    assert_eq!(hp.proj_stack_factor as i64, want_stack);
    assert_eq!(clip.n_mmproj_embd() as i64, proj_dim());

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
    assert_eq!(mctx.audio_sample_rate(), 16000);

    let wav = fixture_wav();
    let chunks = mctx
        .tokenize_audio(&vocab, &wav, false)
        .expect("tokenize_audio");

    // the 2.56 s fixture pads to >3000 frames -> exactly one 3000-frame chunk
    // (mtmd.cpp:1556-1651). Markers per arch (mtmd.cpp:933-963): qwen2a wraps
    // the chunk in beg+end, voxtral/musicflamingo have only a beg marker,
    // ultravox/glma/meralion emit bare embeddings
    let want_chunks = match arch {
        "qwen2a" => 3,
        "voxtral" | "musicflamingo" => 2,
        _ => 1,
    };
    assert_eq!(
        chunks.len(),
        want_chunks,
        "unexpected chunk layout for {arch}"
    );
    if want_chunks > 1 {
        assert!(
            matches!(chunks[0], MtmdChunk::Text(_)),
            "{arch} beg marker chunk"
        );
    }
    if arch == "qwen2a" {
        assert!(
            matches!(chunks[2], MtmdChunk::Text(_)),
            "qwen2a end marker chunk"
        );
    }
    let audio_chunk = chunks
        .iter()
        .find(|c| matches!(c, MtmdChunk::Audio(_)))
        .expect("audio chunk");
    let MtmdChunk::Audio(a) = audio_chunk else {
        unreachable!()
    };

    // clip.cpp:4237-4276 per arch (3000 frames):
    //   stack archs: align(3000,2)/2 = 1500 -> /2 conv = 750
    //   (+ avgpool /2 for voxtral, +2 BOI/EOI for glma)
    let want_tokens: u32 = match arch {
        "qwen2a" => 750,        // 3000/2 conv /2 avgpool
        "ultravox" => 750,      // stack 1500 /2 conv
        "voxtral" => 375,       // stack 1500 /2 conv /2 avgpool
        "meralion" => 750,      // stack 1500 /2 conv
        "glma" => 752,          // 3000/2 conv /2 stack +2 boi/eoi
        "musicflamingo" => 750, // 3000/2 conv /2 avgpool
        other => panic!("{other}"),
    };
    assert_eq!(a.n_tokens, want_tokens, "{arch} token count");
    assert_eq!(a.batch_f32.entries.len(), 1);
    assert_eq!(a.batch_f32.entries[0].nx, 3000);
    assert_eq!(a.batch_f32.entries[0].ny as i64, N_MEL);
    assert!(a.batch_f32.is_audio);

    // ---- 3. the whisper-enc graph over the chunk ----------------------------
    let embd = clip.audio_batch_encode(&a.batch_f32).expect("encode");
    assert_eq!(embd.len(), want_tokens as usize * proj_dim() as usize);
    assert!(
        embd.iter().all(|v| v.is_finite()),
        "{arch}: embeddings must be finite"
    );
    assert!(
        embd.iter().any(|v| *v != 0.0),
        "{arch}: embeddings must not be all zero"
    );

    // ---- 3b. the same graph with flash attention off (reference -fa off) ---
    // both sides default to AUTO -> ENABLED on CPU; DISABLED exercises the
    // soft_max_ext path of build_attn (clip.cpp:800-812)
    let params_off = ClipContextParams {
        flash_attn_type: llama::clip::ClipFlashAttn::Disabled,
        ..Default::default()
    };
    let mut clip_off =
        clip_init_from_file(&path, &params_off).expect("port loads the audio mmproj (fa off)");
    let embd_off = clip_off
        .audio_batch_encode(&a.batch_f32)
        .expect("encode fa off");
    assert_eq!(embd_off.len(), want_tokens as usize * proj_dim() as usize);
    assert!(
        embd_off.iter().all(|v| v.is_finite()),
        "{arch}: fa-off embeddings must be finite"
    );
    let dump_off = format!("{OUT_DIR}/port-faoff-{arch}.bin");
    llama::clip::write_embedding_dump(&dump_off, &embd_off, a.n_tokens as i32, proj_dim() as i32)
        .unwrap();

    (a.n_tokens, embd, embd_off)
}

#[test]
fn audio_mmproj_loads_and_tokenizes() {
    // the whole family: load, tokenize, encode, then bit-compare against the
    // reference dumps when parity/audio_mtmd_parity.sh has produced them
    // (both the default flash-attention path and the -fa off path)
    for (arch, proj) in FAMILY {
        let (n_tokens, embd, embd_off) = run_synthetic_arch(arch, *proj);
        // the port's own dump for the parity script / manual comparison
        let dump = format!("{OUT_DIR}/port-default-{arch}.bin");
        llama::clip::write_embedding_dump(&dump, &embd, n_tokens as i32, proj_dim() as i32)
            .unwrap();
        eprintln!(
            "{arch}: wrote {dump} ({n_tokens} tokens x {} embd)",
            proj_dim()
        );
        match compare_ref_dump("default", arch, &embd) {
            Some(()) => eprintln!("{arch}: BIT-EXACT vs reference dump (flash attn on)"),
            None => eprintln!(
                "{arch}: no reference dump ({OUT_DIR}/ref-default-{arch}.bin), shape check only"
            ),
        }
        match compare_ref_dump("faoff", arch, &embd_off) {
            Some(()) => eprintln!("{arch}: BIT-EXACT vs reference dump (-fa off)"),
            None => {
                let _ = &embd_off;
            }
        }
    }
}
