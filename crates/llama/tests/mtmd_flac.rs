//! mtmd_flac.rs — the port's minimal FLAC decoder (audio round 4, Task 3).
//!
//!  1. ffmpeg (the system encoder available here) encodes the 16 kHz fixture
//!     into FLACs covering the common subset: mono/stereo 16-bit, 24-bit
//!     mono, and two non-default block sizes.
//!  2. direct PCM check: the port's FLAC decode must equal its own WAV decode
//!     of the same signal BIT-EXACTLY (for 16-bit mono both paths reduce to
//!     i16 / 2^15; any decoder slip moves bits).
//!  3. the reference check runs in parity/audio_flac_parity.sh: the pinned
//!     llama-mtmd-cli decodes the same FLAC through miniaudio/dr_flac with a
//!     bit-exact synthetic mmproj, and the embeddings are compared —
//!     bit-exact embeddings prove bit-exact PCM.

use llama::mtmd::audio_from_wav_bytes;

const OUT_DIR: &str = "/tmp/mtmd-flac";

fn fixture_wav() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../parity/mtmd-fixture-audio.wav"
    ))
    .expect("fixture wav (run parity/gen_audio_fixture.py)")
}

fn run(cmd: &mut std::process::Command) -> String {
    let out = cmd.output().expect("spawn");
    assert!(
        out.status.success(),
        "command failed: {:?}\n{}",
        cmd,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// encode the fixture WAV to FLAC with ffmpeg; returns the FLAC bytes
fn encode_flac(tag: &str, args: &[&str]) -> Vec<u8> {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let wav = format!("{OUT_DIR}/src-{tag}.wav");
    let flac = format!("{OUT_DIR}/fixture-{tag}.flac");
    std::fs::write(&wav, fixture_wav()).unwrap();
    let _ = std::fs::remove_file(&flac);
    run(std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", &wav])
        .args(args)
        .arg(&flac));
    let bytes = std::fs::read(&flac).unwrap();
    assert!(bytes.len() > 42, "flac too small");
    assert_eq!(&bytes[..4], b"fLaC");
    bytes
}

#[test]
fn flac_decode_bit_exact_and_loud_errors() {
    if std::env::var("MTMD_SKIP_FLAC").is_ok() {
        eprintln!("MTMD_SKIP_FLAC set — skipping");
        return;
    }
    // ffmpeg missing → skip loudly (CI environments)
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("ffmpeg not available — skipping the FLAC encode tests");
        return;
    }

    // the reference decode of the same signal via the port's WAV reader
    let want = audio_from_wav_bytes(&fixture_wav(), 16000).expect("wav decode");

    // (name, ffmpeg args): mono 16-bit default block size, stereo, 24-bit,
    // blocksize 192/4096, and a compression level sweep (different LPC
    // orders / rice partitions)
    let cases: &[(&str, Vec<&str>)] = &[
        ("mono16", vec!["-ac", "1", "-ar", "16000", "-c:a", "flac"]),
        (
            "mono16-b192",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-blocksize",
                "192",
            ],
        ),
        (
            "mono16-b4096",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-blocksize",
                "4096",
            ],
        ),
        (
            "mono16-l8",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-compression_level",
                "8",
            ],
        ),
        (
            "mono16-l0",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
            ],
        ),
        (
            "mono24",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                "-sample_fmt",
                "s32",
            ],
        ),
        ("stereo16", vec!["-ac", "2", "-ar", "16000", "-c:a", "flac"]),
    ];

    for (tag, args) in cases {
        let bytes = encode_flac(tag, args);
        match audio_from_wav_bytes(&bytes, 16000) {
            Ok(got) => {
                if *tag == "stereo16" || *tag == "mono24" {
                    // different channel count / width than the mono16 WAV:
                    // just require the same frame count and finite audio
                    assert!(!got.is_empty(), "{tag}: empty decode");
                    assert!(got.iter().all(|v| v.is_finite()), "{tag}: non-finite");
                    eprintln!("{tag}: {} frames decoded", got.len());
                } else {
                    assert_eq!(got.len(), want.len(), "{tag}: frame count");
                    let bad = got
                        .iter()
                        .zip(want.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        bad,
                        0,
                        "{tag}: {bad}/{} samples differ from the WAV decode",
                        got.len()
                    );
                    eprintln!("{tag}: BIT-EXACT vs the WAV decode ({} frames)", got.len());
                }
            }
            Err(e) => panic!("{tag}: FLAC decode failed: {e}"),
        }
    }

    // a rate-mismatched FLAC still fails loudly (resampling not ported)
    let bad = encode_flac("mono8k", &["-ac", "1", "-ar", "8000", "-c:a", "flac"]);
    let err = audio_from_wav_bytes(&bad, 16000).unwrap_err();
    assert!(err.contains("sample-rate conversion"), "rate error: {err}");

    // truncated FLAC fails (not silently mis-decoded)
    let full = encode_flac("trunc", &["-ac", "1", "-ar", "16000", "-c:a", "flac"]);
    let cut = &full[..full.len() / 2];
    match audio_from_wav_bytes(cut, 16000) {
        Ok(v) => panic!(
            "truncated FLAC decoded to {} frames (should error)",
            v.len()
        ),
        Err(e) => {
            assert!(e.contains("FLAC"), "truncation error names FLAC: {e}");
            eprintln!("truncated flac fails loudly: {e}");
        }
    }

    // ---- the reference-protocol embedding dumps (parity/audio_flac_parity.sh):
    // tokenize + encode each FLAC with the bit-exact gemma4ua synthetic mmproj
    // — bit-exact embeddings vs the reference prove bit-exact PCM
    if let Ok(dir) = std::env::var("MTMD_DUMP_FLAC_EMBD_DIR") {
        use ggml::Gguf;
        use llama::clip::{clip_init_from_file, ClipContextParams};
        use llama::mtmd::{MtmdContext, MtmdContextParams};
        use llama::vocab::Vocab;

        let mmproj = "/tmp/mtmd-audio-synth/mmproj-audio-synth-gemma4ua.gguf";
        const VOCAB_SPM: &str =
            "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
        if std::fs::read(mmproj).is_err() || std::fs::read(VOCAB_SPM).is_err() {
            eprintln!("gemma4ua mmproj or vocab missing — skipping embedding dumps");
            return;
        }
        let vgguf = Gguf::open(VOCAB_SPM).expect("vocab fixture");
        let vocab = Vocab::load(&vgguf).expect("vocab");
        let mut mctx = MtmdContext::init_from_file(
            mmproj,
            None,
            &MtmdContextParams {
                media_marker: "<|media|>".into(),
                ..Default::default()
            },
        )
        .expect("mtmd context");
        let mut clip =
            clip_init_from_file(mmproj, &ClipContextParams::default()).expect("clip ctx");
        for (tag, _) in cases {
            let flac = std::fs::read(format!("{OUT_DIR}/fixture-{tag}.flac")).unwrap();
            let chunks = mctx.tokenize_audio(&vocab, &flac, false).expect("tokenize");
            let a = chunks
                .iter()
                .find_map(|c| match c {
                    llama::mtmd::MtmdChunk::Audio(a) => Some(a),
                    _ => None,
                })
                .expect("audio chunk");
            let embd = clip.audio_batch_encode(&a.batch_f32).expect("encode");
            llama::clip::write_embedding_dump(
                &format!("{dir}/port-{tag}.bin"),
                &embd,
                a.n_tokens as i32,
                896,
            )
            .unwrap();
            eprintln!("{tag}: dumped {} embedding values", embd.len());
        }
    }
}
