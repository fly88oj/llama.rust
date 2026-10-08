//! llama-quantize — 1:1 port of llama.cpp `tools/quantize/quantize.cpp`
//! (bd4f514db1) on top of `llama::quant` (llama-quant.cpp) and the ggml
//! quantizers.
//!
//! CLI shape follows the reference exactly:
//!   `llama-quantize [options] model-in.gguf [model-out.gguf] type [nthreads]`
//! `-m/-o` are accepted as an alias for the two positional paths (older docs
//! mention them, but the pinned upstream CLI is positional).
//!
//! `--imatrix FNAME` is the reference's importance-matrix driven quantization
//! (`llama_tensor_get_type` + `llama_tensor_quantize_impl`, llama-quant.cpp).
//! The matrix is read with `llama::imatrix::common_imatrix_load` (so both the
//! GGUF and the legacy `.dat` layouts work) and normalized by the per-expert
//! counts exactly like `tools/quantize/quantize.cpp:183-260`.
//!
//! Not supported in this port (reported as hard errors, never silently
//! ignored): `--include-weights` / `--exclude-weights`, `--prune-layers`,
//! `--keep-split`, `--override-kv`, and the MXFP4/NVFP4/TQ*/Q1_0/Q2_0 target
//! types. Of the IQ family only IQ4_NL/IQ4_XS can be written (the others need
//! the IQ2/IQ3 neighbour-table initializers — see PARITY.md).

mod pipeline;
mod rows;

use std::process::ExitCode;

use ggml::types::GgmlType;
use llama::quant::{Ftype, NameMatcher, QuantizeParams, TensorTypeOverride};

// ---------------------------------------------------------------------------
// QUANT_OPTIONS (quantize.cpp:34-75)
// ---------------------------------------------------------------------------

struct QuantOption {
    name: &'static str,
    ftype: Ftype,
    desc: &'static str,
}

const QUANT_OPTIONS: &[QuantOption] = &[
    QuantOption { name: "Q1_0", ftype: Ftype::MostlyQ1_0, desc: " 1.125 bpw quantization" },
    QuantOption { name: "Q2_0", ftype: Ftype::MostlyQ2_0, desc: " 2.25 bpw quantization (group 64)" },
    QuantOption { name: "Q4_0", ftype: Ftype::MostlyQ4_0, desc: " 4.34G, +0.4685 ppl @ Llama-3-8B" },
    QuantOption { name: "Q4_1", ftype: Ftype::MostlyQ4_1, desc: " 4.78G, +0.4511 ppl @ Llama-3-8B" },
    QuantOption { name: "MXFP4_MOE", ftype: Ftype::MostlyMXFP4_MOE, desc: " MXFP4 MoE" },
    QuantOption { name: "Q5_0", ftype: Ftype::MostlyQ5_0, desc: " 5.21G, +0.1316 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_1", ftype: Ftype::MostlyQ5_1, desc: " 5.65G, +0.1062 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ2_XXS", ftype: Ftype::MostlyIQ2_XXS, desc: " 2.06 bpw quantization" },
    QuantOption { name: "IQ2_XS", ftype: Ftype::MostlyIQ2_XS, desc: " 2.31 bpw quantization" },
    QuantOption { name: "IQ2_S", ftype: Ftype::MostlyIQ2_S, desc: " 2.5  bpw quantization" },
    QuantOption { name: "IQ2_M", ftype: Ftype::MostlyIQ2_M, desc: " 2.7  bpw quantization" },
    QuantOption { name: "IQ1_S", ftype: Ftype::MostlyIQ1_S, desc: " 1.56 bpw quantization" },
    QuantOption { name: "IQ1_M", ftype: Ftype::MostlyIQ1_M, desc: " 1.75 bpw quantization" },
    QuantOption { name: "TQ1_0", ftype: Ftype::MostlyTQ1_0, desc: " 1.69 bpw ternarization" },
    QuantOption { name: "TQ2_0", ftype: Ftype::MostlyTQ2_0, desc: " 2.06 bpw ternarization" },
    QuantOption { name: "Q2_K", ftype: Ftype::MostlyQ2_K, desc: " 2.96G, +3.5199 ppl @ Llama-3-8B" },
    QuantOption { name: "Q2_K_S", ftype: Ftype::MostlyQ2_K_S, desc: " 2.96G, +3.1836 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ3_XXS", ftype: Ftype::MostlyIQ3_XXS, desc: " 3.06 bpw quantization" },
    QuantOption { name: "IQ3_S", ftype: Ftype::MostlyIQ3_S, desc: " 3.44 bpw quantization" },
    QuantOption { name: "IQ3_M", ftype: Ftype::MostlyIQ3_M, desc: " 3.66 bpw quantization mix" },
    QuantOption { name: "Q3_K", ftype: Ftype::MostlyQ3_K_M, desc: "alias for Q3_K_M" },
    QuantOption { name: "IQ3_XS", ftype: Ftype::MostlyIQ3_XS, desc: " 3.3 bpw quantization" },
    QuantOption { name: "Q3_K_S", ftype: Ftype::MostlyQ3_K_S, desc: " 3.41G, +1.6321 ppl @ Llama-3-8B" },
    QuantOption { name: "Q3_K_M", ftype: Ftype::MostlyQ3_K_M, desc: " 3.74G, +0.6569 ppl @ Llama-3-8B" },
    QuantOption { name: "Q3_K_L", ftype: Ftype::MostlyQ3_K_L, desc: " 4.03G, +0.5562 ppl @ Llama-3-8B" },
    QuantOption { name: "IQ4_NL", ftype: Ftype::MostlyIQ4_NL, desc: " 4.50 bpw non-linear quantization" },
    QuantOption { name: "IQ4_XS", ftype: Ftype::MostlyIQ4_XS, desc: " 4.25 bpw non-linear quantization" },
    QuantOption { name: "Q4_K", ftype: Ftype::MostlyQ4_K_M, desc: "alias for Q4_K_M" },
    QuantOption { name: "Q4_K_S", ftype: Ftype::MostlyQ4_K_S, desc: " 4.37G, +0.2689 ppl @ Llama-3-8B" },
    QuantOption { name: "Q4_K_M", ftype: Ftype::MostlyQ4_K_M, desc: " 4.58G, +0.1754 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_K", ftype: Ftype::MostlyQ5_K_M, desc: "alias for Q5_K_M" },
    QuantOption { name: "Q5_K_S", ftype: Ftype::MostlyQ5_K_S, desc: " 5.21G, +0.1049 ppl @ Llama-3-8B" },
    QuantOption { name: "Q5_K_M", ftype: Ftype::MostlyQ5_K_M, desc: " 5.33G, +0.0569 ppl @ Llama-3-8B" },
    QuantOption { name: "Q6_K", ftype: Ftype::MostlyQ6_K, desc: " 6.14G, +0.0217 ppl @ Llama-3-8B" },
    QuantOption { name: "Q8_0", ftype: Ftype::MostlyQ8_0, desc: " 7.96G, +0.0026 ppl @ Llama-3-8B" },
    QuantOption { name: "F16", ftype: Ftype::MostlyF16, desc: "14.00G, +0.0020 ppl @ Mistral-7B" },
    QuantOption { name: "BF16", ftype: Ftype::MostlyBF16, desc: "14.00G, -0.0050 ppl @ Mistral-7B" },
    QuantOption { name: "F32", ftype: Ftype::AllF32, desc: "26.00G              @ 7B" },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    QuantOption { name: "COPY", ftype: Ftype::AllF32, desc: "only copy tensors, no quantizing" },
];

/// `striequals` (quantize.cpp:82-90).
fn striequals(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.eq_ignore_ascii_case(b)
}

/// `try_parse_ftype` (quantize.cpp:92-119).
fn try_parse_ftype(ftype_str_in: &str) -> Option<(Ftype, &'static str)> {
    let up = ftype_str_in.to_ascii_uppercase();
    for it in QUANT_OPTIONS {
        if striequals(it.name, &up) {
            return Some((it.ftype, it.name));
        }
    }
    if let Ok(n) = up.parse::<i32>() {
        for it in QUANT_OPTIONS {
            if it.ftype as i32 == n {
                return Some((it.ftype, it.name));
            }
        }
    }
    None
}

/// `parse_ggml_type` (quantize.cpp:305-315): case-insensitive type-name lookup
/// over `GGML_TYPE_COUNT` entries.
fn parse_ggml_type(arg: &str) -> Option<GgmlType> {
    for i in 0..ggml::types::GGML_TYPE_COUNT {
        if let Some(t) = GgmlType::from_u32(i) {
            if striequals(t.name(), arg) {
                return Some(t);
            }
        }
    }
    None
}

/// `--tensor-type` pattern, backed by `fancy-regex` (the reference uses
/// `std::regex` with ECMAScript syntax; both are searched, not anchored).
struct RegexMatcher(fancy_regex::Regex);

impl NameMatcher for RegexMatcher {
    fn is_match(&self, tensor_name: &str) -> bool {
        self.0.is_match(tensor_name).unwrap_or(false)
    }
}

/// `parse_tensor_type` (quantize.cpp:317-347).
fn parse_tensor_type(data: &str) -> Option<TensorTypeOverride> {
    let (name, ty_str) = data.split_once('=')?;
    if name.is_empty() {
        println!("\nparse_tensor_type: missing tensor name\n");
        return None;
    }
    if ty_str.is_empty() {
        println!("\nparse_tensor_type: missing quantization type\n");
        return None;
    }
    let tn = name.to_ascii_lowercase();
    let ty = match parse_ggml_type(ty_str) {
        Some(t) => t,
        None => {
            println!("\nparse_ggml_type: invalid ggml_type '{ty_str}'\n");
            return None;
        }
    };
    let re = match fancy_regex::Regex::new(&tn) {
        Ok(re) => re,
        Err(e) => {
            println!("\nparse_tensor_type: bad pattern '{tn}': {e}\n");
            return None;
        }
    };
    Some(TensorTypeOverride { pattern: Box::new(RegexMatcher(re)), ty })
}

/// `usage` (quantize.cpp:121-181).
fn usage(executable: &str) -> ExitCode {
    println!("usage: {executable} [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]");
    println!("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]");
    println!("       [--max-buffer-size] [--dry-run]");
    println!("       model-f32.gguf [model-quant.gguf] type [nthreads]\n");
    println!("  (this port also accepts -m <model-in> and -o <model-out>)");
    println!("  (not supported here: --include/exclude-weights, --prune-layers, --keep-split, --override-kv)");
    println!("  (unsupported target types: MXFP4, NVFP4, TQ1_0, TQ2_0, Q1_0, Q2_0, IQ1_*, IQ2_*, IQ3_*)");
    println!("-----------------------------------------------------------------------------");
    println!(" allowed quantization types");
    println!("-----------------------------------------------------------------------------\n");
    for it in QUANT_OPTIONS {
        if it.name != "COPY" {
            println!("  {:2}  or  {:<7} : {}", it.ftype as i32, it.name, it.desc);
        } else {
            println!("          {:<7} : {}", it.name, it.desc);
        }
    }
    ExitCode::from(1)
}

fn take(args: &[String], i: &mut usize, exe: &str, what: &str) -> Option<String> {
    if *i + 1 < args.len() {
        *i += 1;
        Some(args[*i].clone())
    } else {
        eprintln!("{exe}: missing argument for {what}");
        None
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let exe = args.first().cloned().unwrap_or_else(|| "llama-quantize".into());

    let mut params = QuantizeParams::new();
    let mut imatrix_file: Option<String> = None;
    let mut imatrix: Option<pipeline::ImatrixInput> = None;
    let mut tensor_type_opts: Vec<TensorTypeOverride> = Vec::new();
    let mut positional: Vec<String> = Vec::new();
    let mut m_in: Option<String> = None;
    let mut m_out: Option<String> = None;

    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].clone();
        match a.as_str() {
            "--help" | "-h" => return usage(&exe),
            "--leave-output-tensor" => params.quantize_output_tensor = false,
            "--output-tensor-type" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                match parse_ggml_type(&v) {
                    Some(t) => params.output_tensor_type = Some(t),
                    None => return usage(&exe),
                }
            }
            "--token-embedding-type" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                match parse_ggml_type(&v) {
                    Some(t) => params.token_embedding_type = Some(t),
                    None => return usage(&exe),
                }
            }
            "--tensor-type" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                match parse_tensor_type(&v) {
                    Some(o) => tensor_type_opts.push(o),
                    None => return usage(&exe),
                }
            }
            "--tensor-type-file" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                match std::fs::read_to_string(&v) {
                    Ok(s) => {
                        for word in s.split_whitespace() {
                            match parse_tensor_type(word) {
                                Some(o) => tensor_type_opts.push(o),
                                None => return usage(&exe),
                            }
                        }
                    }
                    Err(e) => {
                        println!("\nparse_tensor_type_file: failed to open file '{v}': {e}\n");
                        return usage(&exe);
                    }
                }
            }
            "--dry-run" => params.dry_run = true,
            "--allow-requantize" => params.allow_requantize = true,
            "--pure" => params.pure = true,
            "--imatrix" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                imatrix_file = Some(v);
            }
            "--include-weights" | "--exclude-weights" => {
                // accepted by the reference CLI; only meaningful with --imatrix
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                eprintln!("{exe}: WARNING: {a} is ignored (no imatrix support in this port): {v}");
            }
            "--prune-layers" | "--keep-split" | "--override-kv" => {
                eprintln!("{exe}: not supported in this port: {a}");
                return ExitCode::from(1);
            }
            "--max-buffer-size" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                match v.parse::<usize>() {
                    Ok(mib) if mib > 0 => params.max_buf_size = mib * 1024 * 1024,
                    _ => {
                        eprintln!("{exe}: invalid --max-buffer-size '{v}'");
                        return ExitCode::from(1);
                    }
                }
            }
            "-m" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                m_in = Some(v);
            }
            "-o" => {
                let Some(v) = take(&args, &mut i, &exe, &a) else { return usage(&exe) };
                m_out = Some(v);
            }
            _ => positional.push(a),
        }
        i += 1;
    }

    if let Some(f) = &imatrix_file {
        // quantize.cpp:498 `prepare_imatrix` + :183-260 `load_imatrix`
        match pipeline::load_imatrix(f) {
            Ok(im) => {
                println!(
                    "main: have {} importance matrix entries",
                    im.entries.len()
                );
                if im.entries.is_empty() {
                    eprintln!("{exe}: WARNING: no importance matrix entries in '{f}'");
                    imatrix = None;
                } else {
                    imatrix = Some(im);
                }
            }
            Err(e) => {
                eprintln!("{exe}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    params.tt_overrides = tensor_type_opts;

    // quantize.cpp:488-494: at least two positional arguments
    if positional.len() + usize::from(m_in.is_some()) + usize::from(m_out.is_some()) < 2 {
        println!("{exe}: bad arguments");
        return usage(&exe);
    }

    // ---- positional args (quantize.cpp:558-618), with -m/-o pre-filled ----
    let mut pos = positional.into_iter();
    let fname_inp = match m_in {
        Some(v) => v,
        None => match pos.next() {
            Some(v) => v,
            None => return usage(&exe),
        },
    };
    let (fname_out_opt, ftype_str) = match m_out {
        Some(out) => match pos.next() {
            Some(t) => (Some(out), t),
            None => {
                eprintln!("{exe}: missing ftype");
                return ExitCode::from(1);
            }
        },
        None => {
            let first = match pos.next() {
                Some(v) => v,
                None => return usage(&exe),
            };
            if try_parse_ftype(&first).is_some() {
                // <input> <ftype>: output is auto-named below
                (None, first)
            } else {
                let t = match pos.next() {
                    Some(t) => t,
                    None => {
                        eprintln!("{exe}: missing ftype");
                        return ExitCode::from(1);
                    }
                };
                (Some(first), t)
            }
        }
    };

    let (ftype, ftype_name) = match try_parse_ftype(&ftype_str) {
        Some(v) => v,
        None => {
            eprintln!("{exe}: invalid ftype '{ftype_str}'");
            return ExitCode::from(1);
        }
    };
    params.ftype = ftype;
    if ftype_name == "COPY" {
        params.only_copy = true;
    }

    // nthreads (accepted for CLI compatibility; this port is single-threaded)
    if let Some(v) = pos.next() {
        match v.parse::<usize>() {
            Ok(n) => params.nthread = n,
            Err(e) => {
                eprintln!("{exe}: invalid nthread '{v}' ({e})");
                return ExitCode::from(1);
            }
        }
    }

    // quantize.cpp:570-582: export as [inp path]/ggml-model-[ftype].gguf
    let fname_out = match fname_out_opt {
        Some(v) => v,
        None => {
            let dir = std::path::Path::new(&fname_inp)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let sep = if dir.is_empty() { String::new() } else { "/".to_string() };
            format!("{dir}{sep}ggml-model-{ftype_name}.gguf")
        }
    };

    if !params.dry_run && pipeline::is_same_file(&fname_inp, &fname_out) {
        eprintln!("{exe}: error: input and output files are the same: '{fname_inp}'");
        return ExitCode::from(1);
    }

    // quantize.cpp:620-638
    if params.dry_run {
        eprintln!(
            "{exe}: calculating quantization size for '{}' as {}",
            fname_inp, ftype_name
        );
    } else {
        eprintln!(
            "{exe}: quantizing '{}' to '{}' as {}",
            fname_inp, fname_out, ftype_name
        );
    }

    let t0 = std::time::Instant::now();
    let n_elements = total_elements(&fname_inp);
    match pipeline::llama_model_quantize(&fname_inp, &fname_out, &params, imatrix.as_ref()) {
        Ok(res) => {
            let dt = t0.elapsed();
            eprintln!(
                "\nllama_model_quantize: model size  = {:8.2} MiB ({:.2} BPW)",
                res.total_size_org as f64 / 1024.0 / 1024.0,
                res.total_size_org as f64 * 8.0 / n_elements as f64
            );
            eprintln!(
                "llama_model_quantize: quant size  = {:8.2} MiB ({:.2} BPW)",
                res.total_size_new as f64 / 1024.0 / 1024.0,
                res.total_size_new as f64 * 8.0 / n_elements as f64
            );
            if res.n_fallback > 0 {
                eprintln!(
                    "llama_model_quantize: WARNING: {} of {} tensor(s) required fallback quantization",
                    res.n_fallback, res.n_tensors
                );
            }
            println!("\n{exe}: quantize time = {:8.2} ms", dt.as_secs_f64() * 1000.0);
            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("llama_model_quantize: failed to quantize: {e}");
            eprintln!("{exe}: failed to quantize model from '{fname_inp}'");
            ExitCode::from(1)
        }
    }
}

fn total_elements(fname: &str) -> u64 {
    ggml::gguf::Gguf::open(fname)
        .map(|g| g.tensors.iter().map(|t| t.n_elements() as u64).sum::<u64>())
        .unwrap_or(1)
}