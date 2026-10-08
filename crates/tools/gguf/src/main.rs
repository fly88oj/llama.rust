//! llama-gguf (rust) — GGUF metadata dump.
//!
//! The C tool of this name (`examples/gguf/gguf.cpp`, pinned bd4f514db1) is a
//! writer/reader self-test: `llama-gguf <file> r|w [n]`. It has no metadata
//! dump mode and no `--kv`/`--tensors` flags. This port therefore provides
//!
//!   1. the reference-compatible self-test read mode (`<file> r`), printing
//!      exactly the stable lines the C tool prints (same prefixes, same field
//!      order); the C per-tensor data dump ("reading tensor N data" /
//!      `data[:10]` / pointer addresses / `ctx_data size`) is intentionally
//!      not reproduced — it is the C self-test's data check, not metadata, and
//!      its output contains heap addresses.
//!   2. a metadata dump (`-m model.gguf`, default) with key/type/value for
//!      every kv entry and name/size/offset/type/n_elts/ne for every tensor.
//!      Line shapes mirror the reference so the common fields diff cleanly;
//!      the type/value columns are extensions (the C tool prints keys only).
//!      Arrays and long strings are truncated for display.
//!
//! `w` (write synthetic test file) is not ported: the write path is already
//! covered byte-for-byte by the ggml::gguf_write tests, and the C tool's file
//! content is seeded by libc `rand()` (glibc-specific), so it is not a
//! meaningful cross-port artifact.

use std::fmt::Write as _;
use std::io::Write as _;

use ggml::{Gguf, GgmlType, Value};

/// Output is buffered into one string and written with a single `write_all`,
/// so a closed pipe (`| head`) is a silent exit instead of a Rust panic.
fn emit(out: &str) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
    let _ = lock.flush();
}

/// `ggml_type_name` — the lowercase spellings of ggml.c's `type_traits` table
/// (our `GgmlType::name()` uses uppercase enum spellings; the reference prints
/// this table).
fn ggml_type_name(ty: GgmlType) -> &'static str {
    use GgmlType::*;
    match ty {
        F32 => "f32",
        F16 => "f16",
        Q4_0 => "q4_0",
        Q4_1 => "q4_1",
        Q5_0 => "q5_0",
        Q5_1 => "q5_1",
        Q8_0 => "q8_0",
        Q8_1 => "q8_1",
        Q2K => "q2_K",
        Q3K => "q3_K",
        Q4K => "q4_K",
        Q5K => "q5_K",
        Q6K => "q6_K",
        Q8K => "q8_K",
        Iq2Xxs => "iq2_xxs",
        Iq2Xs => "iq2_xs",
        Iq3Xxs => "iq3_xxs",
        Iq1S => "iq1_s",
        Iq4Nl => "iq4_nl",
        Iq3S => "iq3_s",
        Iq2S => "iq2_s",
        Iq4Xs => "iq4_xs",
        I8 => "i8",
        I16 => "i16",
        I32 => "i32",
        I64 => "i64",
        F64 => "f64",
        Iq1M => "iq1_m",
        Bf16 => "bf16",
        Tq1_0 => "tq1_0",
        Tq2_0 => "tq2_0",
        Mxfp4 => "mxfp4",
        Nvfp4 => "nvfp4",
        Q1_0 => "q1_0",
        Q2_0 => "q2_0",
    }
}

const STR_MAX: usize = 128;
const ARR_MAX: usize = 8;

fn truncate_str(s: &str) -> String {
    if s.chars().count() <= STR_MAX {
        s.to_string()
    } else {
        let head: String = s.chars().take(STR_MAX).collect();
        format!("{head}...")
    }
}

/// Newlines/tabs inside values (chat templates!) would break the one-line-per-kv
/// shape; escape them like a C string literal would.
fn escape_ctrl(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{x}"),
        Value::F64(x) => format!("{x}"),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => format!("\"{}\"", escape_ctrl(&truncate_str(s))),
        Value::Array(ty, items) => {
            let shown: Vec<String> = items.iter().take(ARR_MAX).map(fmt_value).collect();
            if items.len() > ARR_MAX {
                format!(
                    "[{} ...] ({} total, element type {})",
                    shown.join(", "),
                    items.len(),
                    ty.name()
                )
            } else {
                format!("[{}]", shown.join(", "))
            }
        }
    }
}

fn print_header(out: &mut String, prefix: &str, g: &Gguf) {
    // exact reference lines (`gguf_ex_read_0: version:      %d`)
    let _ = writeln!(out, "{prefix}: version:      {}", g.version);
    let _ = writeln!(out, "{prefix}: alignment:   {}", g.alignment);
    let _ = writeln!(out, "{prefix}: data offset: {}", g.data_offset);
}

/// Reference-compatible `r` mode: the stable lines of `gguf_ex_read_0` and
/// `gguf_ex_read_1` (everything except the per-tensor data dump).
fn run_ref_read(g: &Gguf) -> String {
    let mut out = String::new();
    for prefix in ["gguf_ex_read_0", "gguf_ex_read_1"] {
        let with_type = prefix == "gguf_ex_read_1";

        print_header(&mut out, prefix, g);

        let _ = writeln!(out, "{prefix}: n_kv: {}", g.kv.len());
        for (i, (key, _)) in g.kv.iter().enumerate() {
            let _ = writeln!(out, "{prefix}: kv[{i}]: key = {key}");
        }

        if !with_type {
            // the reference probes a fixed key from its own write test
            const FINDKEY: &str = "some.parameter.string";
            match g.kv.iter().position(|(k, _)| k == FINDKEY) {
                None => {
                    let _ = writeln!(out, "{prefix}: find key: {FINDKEY} not found.");
                }
                Some(idx) => {
                    let val = g.get_str(FINDKEY).unwrap_or("");
                    let _ = writeln!(
                        out,
                        "{prefix}: find key: {FINDKEY} found, kv[{idx}] value = {val}"
                    );
                }
            }
        }

        let _ = writeln!(out, "{prefix}: n_tensors: {}", g.tensors.len());
        for (i, t) in g.tensors.iter().enumerate() {
            let size = t.size_bytes();
            if with_type {
                // reference: n_elements = size / ggml_type_size(type) — for
                // quantized types this is the number of *blocks* (a quirk of
                // examples/gguf; reproduced for line-compatibility)
                let n_elts = size / t.ty.type_size() as u64;
                let _ = writeln!(
                    out,
                    "{prefix}: tensor[{i}]: name = {}, size = {size}, offset = {}, type = {}, n_elts = {n_elts}",
                    t.name,
                    t.offset,
                    ggml_type_name(t.ty)
                );
            } else {
                let _ = writeln!(
                    out,
                    "{prefix}: tensor[{i}]: name = {}, size = {size}, offset = {}",
                    t.name, t.offset
                );
            }
        }
    }
    out
}

/// Default metadata dump: kv (key/type/value) + tensors (name/size/offset/
/// type/n_elts/ne), same line shapes as the reference plus the information it
/// omits.
fn run_dump(g: &Gguf, kv: bool, tensors: bool) -> String {
    let mut out = String::new();
    print_header(&mut out, "llama-gguf", g);

    if kv {
        let _ = writeln!(out, "llama-gguf: n_kv: {}", g.kv.len());
        for (i, (key, value)) in g.kv.iter().enumerate() {
            match value {
                // arrays advertise their element type, like gguf_type_name()
                Value::Array(ty, _) => {
                    let _ = writeln!(
                        out,
                        "llama-gguf: kv[{i}]: key = {key}, type = array({}), value = {}",
                        ty.name(),
                        fmt_value(value)
                    );
                }
                _ => {
                    let _ = writeln!(
                        out,
                        "llama-gguf: kv[{i}]: key = {key}, type = {}, value = {}",
                        value.type_().name(),
                        fmt_value(value)
                    );
                }
            }
        }
    }

    if tensors {
        let _ = writeln!(out, "llama-gguf: n_tensors: {}", g.tensors.len());
        for (i, t) in g.tensors.iter().enumerate() {
            let size = t.size_bytes();
            let n_elts = size / t.ty.type_size() as u64;
            let _ = writeln!(
                out,
                "llama-gguf: tensor[{i}]: name = {}, size = {size}, offset = {}, type = {}, n_elts = {n_elts}, ne = [{}, {}, {}, {}]",
                t.name,
                t.offset,
                ggml_type_name(t.ty),
                t.ne[0],
                t.ne[1],
                t.ne[2],
                t.ne[3]
            );
        }
    }
    out
}

/// One-line-per-field summary (no reference equivalent; convenience only).
fn run_brief(g: &Gguf) -> String {
    let mut out = String::new();
    let arch = g.get_str("general.architecture").unwrap_or("?");
    let name = g.find_key("general.name").map(fmt_value).unwrap_or_default();
    let size_label = g.find_key("general.size_label").map(fmt_value).unwrap_or_default();
    let file_type = g.find_key("general.file_type").map(fmt_value).unwrap_or_default();
    let _ = writeln!(
        out,
        "llama-gguf: {} kv, {} tensors, version {}",
        g.kv.len(),
        g.tensors.len(),
        g.version
    );
    let _ = writeln!(out, "llama-gguf: architecture = {arch}");
    if !name.is_empty() {
        let _ = writeln!(out, "llama-gguf: name = {name}");
    }
    if !size_label.is_empty() {
        let _ = writeln!(out, "llama-gguf: size_label = {size_label}");
    }
    if !file_type.is_empty() {
        let _ = writeln!(out, "llama-gguf: file_type = {file_type}");
    }
    out
}

fn usage(prog: &str) {
    println!("usage: {prog} -m model.gguf [--kv] [--tensors] [--brief]");
    println!("       {prog} <file.gguf> r [n]     (reference-compatible read dump)");
    println!();
    println!("  -m, --model FILE   gguf file to read");
    println!("  --kv               print only the kv section of the metadata dump");
    println!("  --tensors          print only the tensor section of the metadata dump");
    println!("  --brief            one-line-per-field summary");
    println!("  -h, --help         this help");
    println!();
    println!("  default dump: version/alignment/data offset + kv[<i>] key/type/value");
    println!("  + tensor[<i>] name/size/offset/type/n_elts/ne.  Long strings are cut at");
    println!("  {STR_MAX} chars and arrays after {ARR_MAX} elements (both marked with \"...\");");
    println!("  \\n \\r \\t inside string values are escaped.");
    println!();
    println!("  note: n_elts = size / ggml_type_size(type), i.e. the *block* count for");
    println!("  quantized types (the reference's own formula); `ne` gives the tensor shape.");
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let prog = argv.first().cloned().unwrap_or_else(|| "llama-gguf".into());

    let mut model: Option<String> = None;
    let mut kv_only = false;
    let mut tensors_only = false;
    let mut brief = false;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        match a {
            "-h" | "--help" | "--usage" => {
                usage(&prog);
                return;
            }
            "-m" | "--model" => {
                i += 1;
                match argv.get(i) {
                    Some(v) => model = Some(v.clone()),
                    None => {
                        eprintln!("error: {a} requires an argument");
                        std::process::exit(1);
                    }
                }
            }
            "--kv" => kv_only = true,
            "--tensors" => tensors_only = true,
            "--brief" => brief = true,
            s if s.starts_with("--model=") => model = Some(s["--model=".len()..].to_string()),
            s if s.starts_with('-') && s.len() > 1 => {
                eprintln!("error: unknown option '{s}'");
                usage(&prog);
                std::process::exit(1);
            }
            s => positional.push(s.to_string()),
        }
        i += 1;
    }

    // mode 1: reference-compatible `<file> r|w [n]`
    if !positional.is_empty() {
        if positional.len() < 2 {
            usage(&prog);
            std::process::exit(1);
        }
        let fname = &positional[0];
        let mode = &positional[1];
        let g = match Gguf::open(fname) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("gguf_ex_read_0: failed to load '{fname}': {e}");
                std::process::exit(1);
            }
        };
        match mode.as_str() {
            "r" => emit(&run_ref_read(&g)),
            "w" => {
                eprintln!(
                    "error: 'w' (write synthetic test file) is not ported; the GGUF write path \
                     is covered by the byte-exact ggml::gguf_write tests"
                );
                std::process::exit(1);
            }
            other => {
                eprintln!("error: mode must be r or w (got '{other}')");
                std::process::exit(1);
            }
        }
        return;
    }

    // mode 2: metadata dump
    let Some(path) = model else {
        usage(&prog);
        std::process::exit(1);
    };
    let g = match Gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: could not read '{path}': {e}");
            std::process::exit(1);
        }
    };
    let _ = std::io::stdout().flush();

    if brief {
        emit(&run_brief(&g));
        return;
    }

    let (kv, tensors) = match (kv_only, tensors_only) {
        (true, false) => (true, false),
        (false, true) => (false, true),
        _ => (true, true),
    };
    emit(&run_dump(&g, kv, tensors));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_match_ggml_spelling() {
        assert_eq!(ggml_type_name(GgmlType::Q4K), "q4_K");
        assert_eq!(ggml_type_name(GgmlType::Q8_0), "q8_0");
        assert_eq!(ggml_type_name(GgmlType::F32), "f32");
        assert_eq!(ggml_type_name(GgmlType::Bf16), "bf16");
        assert_eq!(ggml_type_name(GgmlType::Q1_0), "q1_0");
        assert_eq!(ggml_type_name(GgmlType::Mxfp4), "mxfp4");
    }

    #[test]
    fn value_formatting_and_truncation() {
        assert_eq!(fmt_value(&Value::U32(7)), "7");
        assert_eq!(fmt_value(&Value::F32(0.5)), "0.5");
        assert_eq!(fmt_value(&Value::Bool(true)), "true");
        assert_eq!(fmt_value(&Value::String("hi".into())), "\"hi\"");
        let long = "x".repeat(200);
        let s = fmt_value(&Value::String(long));
        assert!(s.ends_with("...\""), "{s}");
        assert_eq!(s.chars().count(), STR_MAX + 5);

        let arr = Value::Array(
            ggml::GgufType::String,
            (0..10).map(|i| Value::String(format!("t{i}"))).collect(),
        );
        let out = fmt_value(&arr);
        assert!(out.contains("..."), "{out}");
        assert!(out.contains("10 total"), "{out}");

        let small = Value::Array(ggml::GgufType::Uint32, vec![Value::U32(1), Value::U32(2)]);
        assert_eq!(fmt_value(&small), "[1, 2]");
    }
}