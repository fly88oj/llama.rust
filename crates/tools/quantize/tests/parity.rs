//! parity.rs — integration tests for the `llama-quantize` port.
//!
//! * `mini_model_roundtrip` runs by default: it builds a tiny valid GGUF,
//!   quantizes it with the tool binary, and checks the result structurally
//!   (KV list, tensor types, offsets, payload sizes).
//! * `parity_vs_reference_binary` (`#[ignore]`) is the strong test: it runs the
//!   pinned reference `llama-quantize` and this port on the same real model and
//!   compares the outputs byte-for-byte, then validates the Rust output with
//!   the Rust GGUF reader.
//!
//! Run the ignored one with:
//!   cargo test -p llama-quantize --test parity -- --ignored --nocapture
//!
//! Environment overrides:
//!   QZ_REF_BIN   reference binary   (default /home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-quantize)
//!   QZ_MODEL     test model         (default /home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf)

use std::path::{Path, PathBuf};
use std::process::Command;

use ggml::gguf::{Gguf, Value};
use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;

const DEFAULT_REF_BIN: &str =
    "/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/llama-quantize";
const DEFAULT_MODEL: &str =
    "/home/jeffrey/localai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn our_bin() -> &'static str {
    env!("CARGO_BIN_EXE_llama-quantize")
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("llama_rust_quantize_test_{name}"))
}

fn run(bin: &str, args: &[&str]) -> std::process::Output {
    Command::new(bin).args(args).output().expect("spawn")
}

// ---------------------------------------------------------------------------
// mini model fixture
// ---------------------------------------------------------------------------

/// A minimal but valid qwen2 GGUF: 1 layer, n_embd 64, n_ff 64, vocab 32.
/// Tensor shapes are chosen so all three relevant paths are exercised:
/// `ne[0] = 64` (divisible by 32, not by 256 → fallback), `ne[0] = 640`
/// (divisible by neither → F16 demotion), 1-D norms (never quantized).
fn write_mini_model(path: &Path) -> Vec<(String, GgmlType, [i64; 4])> {
    let mut w = GgufWriter::new(32);
    w.set_kv("general.architecture", Value::String("qwen2".into()));
    w.set_kv("general.name", Value::String("mini".into()));
    w.set_kv("general.file_type", Value::U32(0)); // F32 input
    w.set_kv("qwen2.block_count", Value::U32(1));
    w.set_kv("qwen2.context_length", Value::U32(128));
    w.set_kv("qwen2.embedding_length", Value::U32(64));
    w.set_kv("qwen2.feed_forward_length", Value::U32(64));
    w.set_kv("qwen2.attention.head_count", Value::U32(4));
    w.set_kv("qwen2.attention.head_count_kv", Value::U32(1));
    w.set_kv("qwen2.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    w.set_kv("qwen2.rope.freq_base", Value::F32(10000.0));
    w.set_kv("qwen2.vocab_size", Value::U32(32));
    w.set_kv("tokenizer.ggml.model", Value::String("gpt2".into()));
    w.set_kv(
        "tokenizer.ggml.tokens",
        Value::Array(
            ggml::gguf::GgufType::String,
            (0..32).map(|i| Value::String(format!("t{i}"))).collect(),
        ),
    );

    let layout: Vec<(String, GgmlType, [i64; 4])> = vec![
        ("token_embd.weight".into(), GgmlType::F32, [64, 32, 1, 1]),
        ("output.weight".into(), GgmlType::F32, [64, 32, 1, 1]),
        ("output_norm.weight".into(), GgmlType::F32, [64, 1, 1, 1]),
        ("blk.0.attn_norm.weight".into(), GgmlType::F32, [64, 1, 1, 1]),
        ("blk.0.attn_q.weight".into(), GgmlType::F32, [64, 64, 1, 1]),
        // 100 is divisible by neither 256 nor 32: the demoted type is still
        // incompatible → F16 (llama-quant.cpp:411-421)
        ("blk.0.attn_k.weight".into(), GgmlType::F32, [100, 64, 1, 1]),
        ("blk.0.attn_v.weight".into(), GgmlType::F32, [64, 64, 1, 1]),
        ("blk.0.attn_output.weight".into(), GgmlType::F32, [64, 64, 1, 1]),
        ("blk.0.ffn_gate.weight".into(), GgmlType::F32, [64, 64, 1, 1]),
        ("blk.0.ffn_up.weight".into(), GgmlType::F32, [64, 64, 1, 1]),
        // 640 = 256*2 + 128: divisible by 32, so the 256-block targets demote
        ("blk.0.ffn_down.weight".into(), GgmlType::F32, [640, 64, 1, 1]),
    ];

    // deterministic pseudorandom payloads
    let mut state = 0x1234_5678u32;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let payloads: Vec<Vec<u8>> = layout
        .iter()
        .map(|(_, ty, ne)| {
            let n = ne.iter().product::<i64>() as usize;
            let mut bytes = Vec::with_capacity(n * ty.type_size());
            for _ in 0..n {
                bytes.extend_from_slice(&next().to_le_bytes());
            }
            bytes
        })
        .collect();

    for (name, ty, ne) in &layout {
        w.add_tensor(name, *ty, *ne);
    }
    let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
    let mut out: Vec<u8> = Vec::new();
    w.write(&mut out, &refs).unwrap();
    std::fs::write(path, &out).unwrap();
    layout
}

/// Quantize the mini model with the tool binary and read the result back.
#[test]
fn mini_model_roundtrip() {
    let src = tmp("mini_in.gguf");
    write_mini_model(&src);

    let cases: &[(&str, &[(&str, GgmlType)])] = &[
        (
            "Q4_0",
            &[
                ("token_embd.weight", GgmlType::Q4_0),
                // OUTPUT branch: Q4_0 → Q6_K → Q8_0 (64 % 256 != 0)
                ("output.weight", GgmlType::Q8_0),
                ("output_norm.weight", GgmlType::F32), // 1-D → untouched
                ("blk.0.attn_q.weight", GgmlType::Q4_0),
                ("blk.0.attn_k.weight", GgmlType::F16), // 100 % 32 != 0
                ("blk.0.ffn_down.weight", GgmlType::Q4_0), // 640 % 32 == 0
            ],
        ),
        (
            "Q4_K_M",
            &[
                ("token_embd.weight", GgmlType::Q5_0), // Q4_K → 64 % 256 → Q5_0
                ("output.weight", GgmlType::Q8_0),     // OUTPUT branch → Q8_0
                ("blk.0.attn_q.weight", GgmlType::Q5_0),
                // n_attention_wv = 1 → use_more_bits(0, 1) → Q6_K → Q8_0
                ("blk.0.attn_v.weight", GgmlType::Q8_0),
                ("blk.0.attn_k.weight", GgmlType::F16), // 100 % 32 != 0
                ("blk.0.ffn_down.weight", GgmlType::Q8_0), // Q6_K → Q8_0 (640 % 32 == 0)
            ],
        ),
        (
            "F16",
            &[
                ("token_embd.weight", GgmlType::F16),
                ("blk.0.attn_q.weight", GgmlType::F16),
                ("blk.0.attn_k.weight", GgmlType::F16),
                ("output_norm.weight", GgmlType::F32),
            ],
        ),
    ];

    for (ftype, checks) in cases {
        let out = tmp(&format!("mini_{ftype}.gguf"));
        let r = run(our_bin(), &[src.to_str().unwrap(), out.to_str().unwrap(), ftype]);
        assert!(
            r.status.success(),
            "{ftype} failed: {}",
            String::from_utf8_lossy(&r.stderr)
        );

        let g = Gguf::open(&out).expect("read back");
        // file_type / quantization_version were (re)written
        assert_eq!(g.find_key("general.file_type").unwrap().as_u32(), Some(
            match *ftype {
                "Q4_0" => 2,
                "Q4_K_M" => 15,
                _ => 1,
            }
        ));
        assert_eq!(g.find_key("general.quantization_version").unwrap().as_u32(), Some(2));
        // the KV list keeps the input order except for the two rewritten keys,
        // which move to the end (gguf.cpp gguf_set_val_* = remove + append)
        let keys: Vec<&str> = g.kv.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys[keys.len() - 2], "general.quantization_version");
        assert_eq!(keys[keys.len() - 1], "general.file_type");

        for (name, want_ty) in *checks {
            let t = g.find_tensor(name).unwrap_or_else(|| panic!("{ftype}: {name} missing"));
            assert_eq!(t.ty, *want_ty, "{ftype}: {name}");
            assert_eq!(
                t.size_bytes(),
                t.ty.type_size() as u64 * (t.n_elements() as u64 / t.ty.blck_size() as u64)
            );
            // payload is inside the file and (for paged tensors) aligned
            assert_eq!(t.offset % g.alignment, 0, "{ftype}: {name} unaligned");
            assert!(
                g.tensor_data(name).is_some(),
                "{ftype}: {name} payload out of file bounds"
            );
        }
        // the reader must accept every tensor
        for t in &g.tensors {
            assert!(g.tensor_data(&t.name).is_some(), "{ftype}: {} unreadable", t.name);
        }
        std::fs::remove_file(&out).ok();
    }
    std::fs::remove_file(&src).ok();
}

// ---------------------------------------------------------------------------
// reference parity (manual; needs the local model + reference binary)
// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    ftype: &'static str,
    opts: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case { name: "q1_0", ftype: "Q1_0", opts: &[] },
    Case { name: "q2_0", ftype: "Q2_0", opts: &[] },
    Case { name: "q4_0", ftype: "Q4_0", opts: &[] },
    Case { name: "q4_1", ftype: "Q4_1", opts: &[] },
    Case { name: "q5_0", ftype: "Q5_0", opts: &[] },
    Case { name: "q5_1", ftype: "Q5_1", opts: &[] },
    Case { name: "q8_0", ftype: "Q8_0", opts: &[] },
    Case { name: "q2_k", ftype: "Q2_K", opts: &[] },
    Case { name: "q3_k_s", ftype: "Q3_K_S", opts: &[] },
    Case { name: "q3_k_m", ftype: "Q3_K_M", opts: &[] },
    Case { name: "q3_k_l", ftype: "Q3_K_L", opts: &[] },
    Case { name: "q4_k_s", ftype: "Q4_K_S", opts: &[] },
    Case { name: "q4_k_m", ftype: "Q4_K_M", opts: &[] },
    Case { name: "q5_k_s", ftype: "Q5_K_S", opts: &[] },
    Case { name: "q5_k_m", ftype: "Q5_K_M", opts: &[] },
    Case { name: "q6_k", ftype: "Q6_K", opts: &[] },
    Case { name: "f16", ftype: "F16", opts: &[] },
    Case { name: "bf16", ftype: "BF16", opts: &[] },
    Case { name: "copy", ftype: "COPY", opts: &[] },
    Case { name: "pure_q4km", ftype: "Q4_K_M", opts: &["--pure"] },
    Case { name: "leave_output", ftype: "Q4_K_M", opts: &["--leave-output-tensor"] },
    Case { name: "tok_embd_q4_0", ftype: "Q4_K_M", opts: &["--token-embedding-type", "Q4_0"] },
    Case { name: "out_tensor_q5_0", ftype: "Q4_K_M", opts: &["--output-tensor-type", "Q5_0"] },
    Case { name: "tensor_type", ftype: "Q4_K_M", opts: &["--tensor-type", "ffn_down=q8_0"] },
    Case {
        name: "tensor_type_multi",
        ftype: "Q4_K_M",
        opts: &["--tensor-type", "attn_v=q5_0", "--tensor-type", "ffn_gate=q4_0"],
    },
];

#[test]
#[ignore = "runs the reference binary on a 491 MB model (see module docs)"]
fn parity_vs_reference_binary() {
    let ref_bin = std::env::var("QZ_REF_BIN").unwrap_or_else(|_| DEFAULT_REF_BIN.into());
    let model = std::env::var("QZ_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    if !Path::new(&ref_bin).exists() || !Path::new(&model).exists() {
        eprintln!("skipping: need {ref_bin} and {model}");
        return;
    }
    let outdir = std::env::temp_dir().join("llama_rust_quantize_parity");
    std::fs::create_dir_all(&outdir).unwrap();

    let mut failures = Vec::new();
    for c in CASES {
        let mine = outdir.join(format!("rust-{}.gguf", c.name));
        let reference = outdir.join(format!("ref-{}.gguf", c.name));
        let _ = std::fs::remove_file(&mine);
        let _ = std::fs::remove_file(&reference);

        let mut my_args = vec!["--allow-requantize"];
        my_args.extend_from_slice(c.opts);
        my_args.extend_from_slice(&[&model, mine.to_str().unwrap(), c.ftype]);
        let r1 = run(our_bin(), &my_args);

        let mut ref_args = vec!["--allow-requantize"];
        ref_args.extend_from_slice(c.opts);
        ref_args.extend_from_slice(&[&model, reference.to_str().unwrap(), c.ftype]);
        let r2 = run(&ref_bin, &ref_args);

        assert_eq!(
            r1.status.code(),
            r2.status.code(),
            "{}: exit status differs\nrust stderr:\n{}\nref stderr:\n{}",
            c.name,
            String::from_utf8_lossy(&r1.stderr),
            String::from_utf8_lossy(&r2.stderr)
        );
        let (a, b) = (std::fs::read(&mine).unwrap(), std::fs::read(&reference).unwrap());
        if a == b {
            println!("{}: IDENTICAL ({} bytes)", c.name, a.len());
        } else {
            let first = a.iter().zip(&b).position(|(x, y)| x != y);
            failures.push(format!(
                "{}: DIFFER rust={} ref={} first_diff={:?}",
                c.name,
                a.len(),
                b.len(),
                first
            ));
        }
    }
    assert!(failures.is_empty(), "byte differences:\n{}", failures.join("\n"));
    println!("all {} cases byte-identical", CASES.len());
}

/// The Rust output must be loadable by the Rust reader/model loader.
#[test]
#[ignore = "runs the reference binary on a 491 MB model (see module docs)"]
fn rust_output_is_loadable() {
    use std::sync::Arc;
    let model = std::env::var("QZ_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    if !Path::new(&model).exists() {
        eprintln!("skipping: need {model}");
        return;
    }
    let outdir = std::env::temp_dir().join("llama_rust_quantize_parity");
    std::fs::create_dir_all(&outdir).unwrap();
    let out = outdir.join("rust-q4_0.gguf");
    if !out.exists() {
        let r = run(
            our_bin(),
            &["--allow-requantize", &model, out.to_str().unwrap(), "Q4_0"],
        );
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    }

    let gguf = Gguf::open(&out).unwrap();
    // structural checks on every tensor
    let mut total = 0u64;
    for t in &gguf.tensors {
        let data = gguf.tensor_data(&t.name).expect("payload in bounds");
        assert_eq!(data.len() as u64, t.size_bytes(), "{}", t.name);
        assert_eq!(t.offset % gguf.alignment, 0, "{}", t.name);
        total += t.size_bytes();
    }
    println!(
        "{} tensors, {} MiB payload, {} MiB file",
        gguf.tensors.len(),
        total / 1024 / 1024,
        out.metadata().unwrap().len() / 1024 / 1024
    );

    // the full Rust model loader must accept it
    let f = std::fs::File::open(&out).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    let m = llama::model::load_model(&gguf, mmap).expect("Rust model loader accepts the file");
    println!("loaded: {} layers, n_embd {}", m.n_layer(), m.n_embd());
}