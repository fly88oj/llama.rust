//! llama-gguf-split (rust) — port of `tools/gguf-split/gguf-split.cpp`
//! (pinned bd4f514db1, 609 lines).
//!
//! Split/merge a GGUF along the `split.*` KV convention:
//!   * `split.no` (u16) — this file's 0-based index
//!   * `split.count` (u16) — total number of files (0 in a merged file)
//!   * `split.tensors.count` (i32) — tensors across *all* files
//! File naming: `{prefix}-{no+1:05}-of-{count:05}.gguf` (llama.cpp:544).
//!
//! Byte-exactness contract (verified cross-tool against the reference):
//!   * every kv write goes through the reference's `gguf_set_val_*`
//!     remove-then-append semantics (gguf.cpp:1246-1334) — NOT
//!     `GgufWriter::set_kv`'s in-place override — so the kv order of the
//!     output matches the C tool exactly (model kv in file order, then
//!     split.no / split.count / split.tensors.count appended).
//!   * output contexts are `gguf_init_empty()` (gguf.cpp:429-431): alignment
//!     32 regardless of the input's `general.alignment` kv (the kv value is
//!     still copied verbatim; the quirk is the reference's).
//!   * tensor bytes are copied raw from the input and padded with zeros to
//!     GGUF_DEFAULT_ALIGNMENT after each tensor (gguf-split.cpp:347-348).
//!   * merge keeps `split.no`/`split.tensors.count` and rewrites
//!     `split.count` to 0 — "Do not trigger merge if we try to merge again
//!     the output" (gguf-split.cpp:491) — which moves that key to the kv
//!     list's end via the remove+append semantics.

use std::io::{Read, Seek, SeekFrom, Write};

use ggml::gguf::{
    split_path, split_prefix, Gguf, Value, GGUF_DEFAULT_ALIGNMENT,
    GGUF_KEY_GENERAL_ALIGNMENT, LLM_KV_SPLIT_COUNT, LLM_KV_SPLIT_NO, LLM_KV_SPLIT_TENSORS_COUNT,
};
use ggml::gguf_write::GgufWriter;
use ggml::GgmlType;

/// `enum split_operation` (gguf-split.cpp:29-33)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SplitOperation {
    None,
    Split,
    Merge,
}

/// `enum split_mode` (gguf-split.cpp:35-39)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SplitMode {
    None,
    Tensor,
    Size,
}

/// `struct split_params` (gguf-split.cpp:41-51)
struct SplitParams {
    operation: SplitOperation,
    mode: SplitMode,
    n_bytes_split: u64,
    n_split_tensors: i32, // default 128
    input: String,
    output: String,
    no_tensor_first_split: bool,
    dry_run: bool,
    delete_splits: bool,
}

impl Default for SplitParams {
    fn default() -> Self {
        SplitParams {
            operation: SplitOperation::None,
            mode: SplitMode::None,
            n_bytes_split: 0,
            n_split_tensors: 128,
            input: String::new(),
            output: String::new(),
            no_tensor_first_split: false,
            dry_run: false,
            delete_splits: false,
        }
    }
}

/// `split_print_usage` (gguf-split.cpp:53-71)
fn split_print_usage(executable: &str) {
    let default_params = SplitParams::default();
    println!();
    println!("usage: {executable} [options] GGUF_IN GGUF_OUT");
    println!();
    println!("Apply a GGUF operation on IN to OUT.");
    println!();
    println!("options:");
    println!("  -h, --help              show this help message and exit");
    println!("  --version               show version and build info");
    println!("  --split                 split GGUF to multiple GGUF (enabled by default)");
    println!("  --merge                 merge multiple GGUF to a single GGUF");
    println!(
        "  --split-max-tensors     max tensors in each split (default: {})",
        default_params.n_split_tensors
    );
    println!("  --split-max-size N(M|G) max size per split");
    println!("  --no-tensor-first-split do not add tensors to the first split (disabled by default)");
    println!("  --dry-run               only print out a split plan and exit, without writing any new files");
    println!("  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge");
    println!();
}

/// `atoi`/`sscanf("%d")` — leading whitespace, optional sign, digits; the rest
/// of the string is ignored; no digits at all yields 0.
fn atoi(s: &str) -> i32 {
    let t = s.trim_start();
    let bytes = t.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut val: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        val = (val * 10 + (bytes[i] - b'0') as i64).min(i64::from(i32::MAX));
        i += 1;
    }
    if !bytes.is_empty() && bytes[0] == b'-' {
        val = -val;
    }
    val as i32
}

/// `split_str_to_n_bytes` (gguf-split.cpp:74-90) — "128M" / "4G" (SI units).
fn split_str_to_n_bytes(s: &str) -> Result<u64, String> {
    if s.is_empty() {
        // std::string::back() on an empty string is UB in the reference;
        // treated here as the no-valid-unit error
        return Err("error: supported units are M (megabytes) or G (gigabytes), but got: ".into());
    }
    let last = s.as_bytes()[s.len() - 1] as char;
    let n = atoi(s);
    let n_bytes = if last == 'M' {
        n as u64 * 1000 * 1000 // megabytes
    } else if last == 'G' {
        n as u64 * 1000 * 1000 * 1000 // gigabytes
    } else {
        return Err(format!(
            "error: supported units are M (megabytes) or G (gigabytes), but got: {last}"
        ));
    };
    if n <= 0 {
        return Err("error: size must be a positive value".into());
    }
    Ok(n_bytes)
}

/// `split_params_parse_ex` (gguf-split.cpp:92-181). Note the loop condition
/// `strncmp(argv[arg_idx], "--", 2) == 0`: only *leading* `--`-prefixed
/// arguments are parsed as flags — a bare `-h` never enters the loop and falls
/// through to the positional-arity error, exactly like the reference.
fn split_params_parse_ex(argv: &[String], params: &mut SplitParams) -> Result<(), String> {
    let mut invalid_param = false;
    let mut arg = String::new();

    let mut arg_idx = 1usize;
    while arg_idx < argv.len() && argv[arg_idx].starts_with("--") {
        arg = argv[arg_idx].clone();
        // std::replace(arg.begin(), arg.end(), '_', '-')
        let arg = arg.replace('_', "-");

        let mut arg_found = false;
        if arg == "-h" || arg == "--help" {
            // dead arm for "-h" in the reference (the loop requires "--");
            // "--help" is reachable
            split_print_usage(&argv[0]);
            std::process::exit(0);
        } else if arg == "--version" {
            eprintln!(
                "version: {} (build {}, commit {})",
                env!("CARGO_PKG_VERSION"),
                0,
                "bd4f514db1"
            );
            std::process::exit(0);
        } else if arg == "--dry-run" {
            arg_found = true;
            params.dry_run = true;
        } else if arg == "--no-tensor-first-split" {
            arg_found = true;
            params.no_tensor_first_split = true;
        } else if arg == "--merge" {
            arg_found = true;
            if params.operation != SplitOperation::None && params.operation != SplitOperation::Merge
            {
                return Err("error: either --split or --merge can be specified, but not both".into());
            }
            params.operation = SplitOperation::Merge;
        } else if arg == "--split" {
            arg_found = true;
            if params.operation != SplitOperation::None && params.operation != SplitOperation::Split
            {
                return Err("error: either --split or --merge can be specified, but not both".into());
            }
            params.operation = SplitOperation::Split;
        } else if arg == "--split-max-tensors" {
            arg_idx += 1;
            if arg_idx >= argv.len() {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if params.mode != SplitMode::None && params.mode != SplitMode::Tensor {
                return Err(
                    "error: either --split-max-tensors or --split-max-size can be specified, but not both"
                        .into(),
                );
            }
            params.mode = SplitMode::Tensor;
            params.n_split_tensors = atoi(&argv[arg_idx]);
        } else if arg == "--split-max-size" {
            arg_idx += 1;
            if arg_idx >= argv.len() {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if params.mode != SplitMode::None && params.mode != SplitMode::Size {
                return Err(
                    "error: either --split-max-tensors or --split-max-size can be specified, but not both"
                        .into(),
                );
            }
            params.mode = SplitMode::Size;
            params.n_bytes_split = split_str_to_n_bytes(&argv[arg_idx])?;
        } else if arg == "--delete-splits" {
            arg_found = true;
            params.delete_splits = true;
        }

        if !arg_found {
            return Err(format!("error: unknown argument: {arg}"));
        }
        arg_idx += 1;
    }

    // the operation is split if not specified
    if params.operation == SplitOperation::None {
        params.operation = SplitOperation::Split;
    }
    // the split mode is by tensor if not specified
    if params.mode == SplitMode::None {
        params.mode = SplitMode::Tensor;
    }

    if invalid_param {
        return Err(format!("error: invalid parameter for argument: {arg}"));
    }

    if argv.len() - arg_idx != 2 {
        return Err("error: bad arguments".into());
    }

    params.input = argv[arg_idx].clone();
    params.output = argv[arg_idx + 1].clone();
    Ok(())
}

/// `split_params_parse` (gguf-split.cpp:183-194)
fn split_params_parse(argv: &[String], params: &mut SplitParams) {
    if let Err(e) = split_params_parse_ex(argv, params) {
        eprintln!("{e}");
        split_print_usage(&argv[0]);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// gguf_set_val_* / gguf_set_kv on a fresh writer — remove-then-append
// (gguf.cpp:1246-1334), unlike GgufWriter::set_kv's in-place override.
// ---------------------------------------------------------------------------

/// `gguf_check_reserved_keys` (gguf.cpp:1229-1238)
fn gguf_check_reserved_keys(key: &str, val: &Value) {
    if key == GGUF_KEY_GENERAL_ALIGNMENT {
        match val {
            Value::U32(v) => assert!(
                *v > 0 && (v & (v - 1)) == 0,
                "{GGUF_KEY_GENERAL_ALIGNMENT} must be power of 2"
            ),
            _ => panic!("{GGUF_KEY_GENERAL_ALIGNMENT} must be type u32"),
        }
    }
}

/// `gguf_set_val_*` (gguf.cpp:1240-1334): `gguf_remove_key` + `emplace_back`.
fn set_val(w: &mut GgufWriter, key: &str, val: Value) {
    gguf_check_reserved_keys(key, &val);
    w.kv.retain(|(k, _)| k != key);
    w.kv.push((key.to_string(), val));
}

/// `gguf_set_kv` (gguf.cpp:1338-1388) into a fresh writer: source kv order is
/// preserved (every set_val on a disjoint key appends).
fn set_kv_from(w: &mut GgufWriter, src: &Gguf) {
    for (k, v) in &src.kv {
        set_val(w, k, v.clone());
    }
}

/// `GGML_PAD(nbytes, GGUF_DEFAULT_ALIGNMENT)`
fn pad32(n: u64) -> u64 {
    n.div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT
}

// ---------------------------------------------------------------------------
// split_strategy (gguf-split.cpp:203-367)
// ---------------------------------------------------------------------------

struct SplitStrategy<'a> {
    params: &'a SplitParams,
    gguf: &'a Gguf,
    n_tensors: usize,
    /// one ctx_out per one output file
    ctx_outs: Vec<GgufWriter>,
}

impl<'a> SplitStrategy<'a> {
    /// The lambda `new_ctx_out` (gguf-split.cpp:229-246): push the current
    /// ctx_out (0-tensor files are fatal unless allowed) and start a fresh one
    /// carrying the split.* bookkeeping — model metadata goes into split 0
    /// only.
    fn new_ctx_out(
        ctx_outs: &mut Vec<GgufWriter>,
        ctx_out: &mut Option<GgufWriter>,
        gguf: &Gguf,
        n_tensors: usize,
        i_split: &mut i64,
        allow_no_tensors: bool,
    ) {
        *i_split += 1;
        if let Some(c) = ctx_out.take() {
            if c.tensors.is_empty() && !allow_no_tensors {
                eprintln!("error: one of splits have 0 tensors. Maybe size or tensors limit is too small");
                std::process::exit(1);
            }
            ctx_outs.push(c);
        }
        // gguf_init_empty() — alignment GGUF_DEFAULT_ALIGNMENT (gguf.cpp:429)
        let mut w = GgufWriter::new(GGUF_DEFAULT_ALIGNMENT);
        // Save all metadata in first split only
        if *i_split == 0 {
            set_kv_from(&mut w, gguf);
        }
        set_val(&mut w, LLM_KV_SPLIT_NO, Value::U16(*i_split as u16));
        set_val(&mut w, LLM_KV_SPLIT_COUNT, Value::U16(0)); // placeholder
        set_val(&mut w, LLM_KV_SPLIT_TENSORS_COUNT, Value::I32(n_tensors as i32));
        *ctx_out = Some(w);
    }

    fn new(params: &'a SplitParams, gguf: &'a Gguf) -> Self {
        let n_tensors = gguf.tensors.len();

        // because we need to know list of tensors for each file in advance, we
        // will build all the ctx_out for all output splits
        let mut ctx_outs: Vec<GgufWriter> = Vec::new();
        let mut ctx_out: Option<GgufWriter> = None;
        let mut i_split: i64 = -1;

        // initialize ctx_out for the first split
        Self::new_ctx_out(&mut ctx_outs, &mut ctx_out, gguf, n_tensors, &mut i_split, false);

        // skip first split if no_tensor_first_split is set
        if params.no_tensor_first_split {
            Self::new_ctx_out(&mut ctx_outs, &mut ctx_out, gguf, n_tensors, &mut i_split, true);
        }

        // process tensors one by one
        let mut curr_tensors_size: u64 = 0; // current size by counting only tensors size (without metadata)
        for i in 0..n_tensors {
            let t = &gguf.tensors[i];
            // calculate the "imaginary" size = the current size + next tensor size
            let n_bytes = pad32(t.size_bytes());
            let next_tensors_size = curr_tensors_size + n_bytes;
            if Self::should_split(params, i as i64, next_tensors_size, n_tensors) {
                Self::new_ctx_out(
                    &mut ctx_outs,
                    &mut ctx_out,
                    gguf,
                    n_tensors,
                    &mut i_split,
                    false,
                );
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensors_size;
            }
            ctx_out
                .as_mut()
                .unwrap()
                .add_tensor(&t.name, t.ty, t.ne);
        }

        // push the last ctx_out
        ctx_outs.push(ctx_out.unwrap());

        // set the correct n_split for all ctx_out
        let n_split = ctx_outs.len() as u16;
        for ctx in &mut ctx_outs {
            set_val(ctx, LLM_KV_SPLIT_COUNT, Value::U16(n_split));
        }

        SplitStrategy { params, gguf, n_tensors, ctx_outs }
    }

    /// `should_split` (gguf-split.cpp:287-297)
    fn should_split(params: &SplitParams, i_tensor: i64, next_size: u64, n_tensors: usize) -> bool {
        if params.mode == SplitMode::Size {
            // split by max size per file
            next_size > params.n_bytes_split
        } else if params.mode == SplitMode::Tensor {
            // split by number of tensors per file (n_split_tensors == 0 makes
            // the C `i % 0` UB/SIGFPE; the Rust remainder panics likewise)
            i_tensor > 0
                && (i_tensor as usize) < n_tensors
                && i_tensor % params.n_split_tensors as i64 == 0
        } else {
            // should never happen
            panic!("invalid mode");
        }
    }

    /// `print_info` (gguf-split.cpp:299-313)
    fn print_info(&self) {
        println!("n_split: {}", self.ctx_outs.len());
        for (i_split, ctx_out) in self.ctx_outs.iter().enumerate() {
            // re-calculate the real gguf size for each split (= metadata size
            // + total size of all tensors)
            let mut total_size = ctx_out.meta_size();
            for t in &ctx_out.tensors {
                total_size += t.nbytes();
            }
            let total_size = total_size / 1000 / 1000; // convert to megabytes
            println!(
                "split {:05}: n_tensors = {}, total_size = {total_size}M",
                i_split + 1,
                ctx_out.tensors.len()
            );
        }
    }

    /// `write` (gguf-split.cpp:315-356) — meta + raw tensor bytes + zero pad.
    fn write(&self) -> std::io::Result<()> {
        let n_split = self.ctx_outs.len();
        for (i_split, ctx_out) in self.ctx_outs.iter().enumerate() {
            // construct file path
            let path = split_path(&self.params.output, i_split as i32, n_split as i32);

            // open the output file
            print!("Writing file {path} ... ");
            let _ = std::io::stdout().flush();
            let fout = std::fs::File::create(&path)?;

            // write metadata, then tensors: raw copy from the input file plus
            // zero padding to GGUF_DEFAULT_ALIGNMENT (copy_file_to_file/zeros)
            let mut bw = std::io::BufWriter::with_capacity(1 << 22, fout);
            let data: Vec<&[u8]> = ctx_out
                .tensors
                .iter()
                .map(|t| self.gguf.tensor_data(&t.name).unwrap_or_else(|| panic!("tensor {} not in input", t.name)))
                .collect();
            ctx_out.write(&mut bw, &data)?;
            bw.flush()?;

            println!("done");
        }
        Ok(())
    }
}

/// `gguf_split` (gguf-split.cpp:369-405)
fn gguf_split(split_params: &SplitParams) {
    let f_input = std::fs::File::open(&split_params.input);
    if f_input.is_err() {
        eprintln!(
            "gguf_split:  failed to open input GGUF from {}",
            split_params.input
        );
        std::process::exit(1);
    }

    let ctx_gguf = Gguf::open_single(&split_params.input);
    let ctx_gguf = match ctx_gguf {
        Ok(g) => g,
        Err(_) => {
            eprintln!(
                "gguf_split:  failed to load input GGUF from {}",
                split_params.input
            );
            std::process::exit(1);
        }
    };

    // prepare the strategy
    let strategy = SplitStrategy::new(split_params, &ctx_gguf);
    let n_split = strategy.ctx_outs.len();
    strategy.print_info();

    if !split_params.dry_run {
        // write all output splits
        if let Err(e) = strategy.write() {
            eprintln!("gguf_split: failed to write output: {e}");
            std::process::exit(1);
        }
    }

    eprintln!(
        "gguf_split: {} gguf split written with a total of {} tensors.",
        n_split, strategy.n_tensors
    );
}

/// `zeros(std::ofstream&, size_t)` (gguf-split.cpp:196-201) over a writer.
fn zeros<W: Write>(w: &mut W, n: u64) -> std::io::Result<()> {
    const CHUNK: usize = 1 << 16;
    let zero = [0u8; CHUNK];
    let mut left = n as usize;
    while left > 0 {
        let n = left.min(CHUNK);
        w.write_all(&zero[..n])?;
        left -= n;
    }
    Ok(())
}

/// `copy_file_to_file` (gguf-split.cpp:358-366): raw byte copy of `len` bytes
/// from `in_offset` (the buffer is a chunked equivalent of read_buf).
fn copy_file_to_file(
    f_in: &mut std::fs::File,
    w: &mut impl Write,
    in_offset: u64,
    len: u64,
) -> std::io::Result<()> {
    use std::io::Seek as _;
    f_in.seek(SeekFrom::Start(in_offset))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        let n = left.min(buf.len() as u64) as usize;
        f_in.read_exact(&mut buf[..n])?;
        w.write_all(&buf[..n])?;
        left -= n as u64;
    }
    Ok(())
}

/// `gguf_merge` (gguf-split.cpp:407-591)
fn gguf_merge(split_params: &SplitParams) {
    eprintln!("gguf_merge: {} -> {}", split_params.input, split_params.output);
    let mut n_split: u32 = 1;
    let mut total_tensors: i32 = 0;

    // avoid overwriting existing output file
    if std::path::Path::new(&split_params.output).exists() {
        eprintln!("gguf_merge: output file {} already exists", split_params.output);
        std::process::exit(1);
    }

    let mut ctx_out = GgufWriter::new(GGUF_DEFAULT_ALIGNMENT); // gguf_init_empty

    // First pass to find KV and tensors metadata
    let mut ctx_ggufs: Vec<Gguf> = Vec::new();
    let mut split_path_cur = split_params.input.clone();
    let mut split_prefix_buf = String::new();

    let mut i_split: u32 = 0;
    while i_split < n_split {
        if i_split > 0 {
            split_path_cur =
                split_path(&split_prefix_buf, i_split as i32, n_split as i32);
        }
        eprint!("gguf_merge: reading metadata {split_path_cur} ...");

        let ctx_gguf = match Gguf::open_single(&split_path_cur) {
            Ok(g) => g,
            Err(_) => {
                eprintln!();
                eprintln!(
                    "gguf_merge:  failed to load input GGUF from {}",
                    split_params.input
                );
                std::process::exit(1);
            }
        };

        if i_split == 0 {
            let Some(v) = ctx_gguf.find_key(LLM_KV_SPLIT_COUNT) else {
                eprintln!();
                eprintln!("gguf_merge: input file does not contain {LLM_KV_SPLIT_COUNT} metadata");
                std::process::exit(1);
            };
            let Some(n) = v.as_u16() else {
                eprintln!();
                eprintln!("gguf_merge: input file does not contain {LLM_KV_SPLIT_COUNT} metadata");
                std::process::exit(1);
            };
            n_split = n as u32;
            if n_split < 1 {
                eprintln!();
                eprintln!(
                    "gguf_merge: input file does not contain a valid split count {n_split}"
                );
                std::process::exit(1);
            }

            // Verify the file naming and extract split_prefix
            match split_prefix(&split_path_cur, 0, n_split as i32) {
                Some(p) => split_prefix_buf = p,
                None => {
                    eprintln!();
                    eprintln!(
                        "gguf_merge: unexpected input file name: {split_path_cur} i_split=0 n_split={n_split}"
                    );
                    std::process::exit(1);
                }
            }

            // Do not trigger merge if we try to merge again the output:
            // gguf_set_val_u16(ctx_gguf, LLM_KV_SPLIT_COUNT, 0) — remove+append
            let mut kv0 = ctx_gguf.kv.clone();
            gguf_check_reserved_keys(LLM_KV_SPLIT_COUNT, &Value::U16(0));
            kv0.retain(|(k, _)| k != LLM_KV_SPLIT_COUNT);
            kv0.push((LLM_KV_SPLIT_COUNT.to_string(), Value::U16(0)));

            // Set metadata from the first split
            for (k, v) in &kv0 {
                set_val(&mut ctx_out, k, v.clone());
            }
        }

        let n_tensors = ctx_gguf.tensors.len();
        for t in &ctx_gguf.tensors {
            ctx_out.add_tensor(&t.name, t.ty, t.ne);
        }
        total_tensors += n_tensors as i32;

        ctx_ggufs.push(ctx_gguf);
        eprint!("\u{1b}[3Ddone\n");
        i_split += 1;
    }

    let mut fout: Option<std::io::BufWriter<std::fs::File>> = None;
    if !split_params.dry_run {
        let file = std::fs::File::create(&split_params.output).unwrap_or_else(|e| {
            eprintln!("gguf_merge: failed to open {}: {e}", split_params.output);
            std::process::exit(1);
        });
        fout = Some(std::io::BufWriter::with_capacity(1 << 22, file));
        // placeholder for the meta data
        let meta_size = ctx_out.meta_size();
        zeros(fout.as_mut().unwrap(), meta_size).unwrap();
    }

    // Write tensors data
    let mut merge_error = false;
    for i_split in 0..n_split {
        let path = split_path(&split_prefix_buf, i_split as i32, n_split as i32);
        let mut f_input = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("gguf_merge:  failed to open input GGUF from {path}");
                std::process::exit(1);
            }
        };
        eprint!("gguf_merge: writing tensors {path} ...");

        let ctx_gguf = &ctx_ggufs[i_split as usize];
        for (i_tensor, t) in ctx_gguf.tensors.iter().enumerate() {
            let n_bytes = t.size_bytes();
            let offset = ctx_gguf.data_offset + t.offset;
            if let Some(w) = fout.as_mut() {
                // write tensor data + padding
                copy_file_to_file(&mut f_input, w, offset, n_bytes).unwrap_or_else(|e| {
                    eprintln!("gguf_merge: read/write failure on {path}: {e}");
                    std::process::exit(1);
                });
                zeros(w, pad32(n_bytes) - n_bytes).unwrap_or_else(|e| {
                    eprintln!("gguf_merge: write failure on {path}: {e}");
                    std::process::exit(1);
                });
            }
            let _ = i_tensor;
        }

        drop(f_input);
        eprint!("\u{1b}[3Ddone\n");

        if !split_params.dry_run && split_params.delete_splits {
            match std::fs::remove_file(&path) {
                Ok(()) => eprintln!("gguf_merge: deleted file {path}"),
                Err(_) => {
                    merge_error = true;
                    eprintln!("error: failed to delete {path}");
                }
            }
        }
    }

    if !split_params.dry_run {
        // go back to beginning of file and write the updated metadata
        let mut w = fout.take().unwrap();
        w.flush().unwrap();
        let file = w.into_inner().unwrap();
        let mut file = std::io::BufWriter::with_capacity(
            1 << 22,
            { let mut f = file; f.seek(SeekFrom::Start(0)).unwrap(); f },
        );
        ctx_out.write_meta(&mut file).unwrap();
        file.flush().unwrap();
    }
    drop(ctx_out);

    eprintln!(
        "gguf_merge: {} merged from {n_split} split with {total_tensors} tensors.",
        split_params.output
    );

    if merge_error {
        std::process::exit(1);
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut params = SplitParams::default();
    split_params_parse(&argv, &mut params);

    match params.operation {
        SplitOperation::Split => gguf_split(&params),
        SplitOperation::Merge => gguf_merge(&params),
        SplitOperation::None => {
            split_print_usage(&argv[0]);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// tests — split/merge round-trip on a synthetic file, split.* convention,
// naming helpers, byte-level kv juggling
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ggml::gguf::GgufType;

    /// deterministic filler (LCG — content only matters by being stable)
    fn filler(seed: u32, n: usize) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// a small synthetic gguf: 5 F32 tensors of 256 B each + 2 Q4_0 of 18 B
    fn synth_model(path: &std::path::Path) -> Vec<Vec<u8>> {
        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("test".into()));
        w.set_kv("test.block_count", Value::U32(2));
        w.set_kv("general.alignment", Value::U32(32));
        let mut payloads = Vec::new();
        for i in 0..5 {
            w.add_tensor(&format!("t{i}.weight"), GgmlType::F32, [64, 1, 1, 1]);
            payloads.push(filler(0x1000 + i as u32, 256));
        }
        for i in 0..2 {
            w.add_tensor(&format!("q{i}.weight"), GgmlType::Q4_0, [32, 1, 1, 1]);
            payloads.push(filler(0x2000 + i as u32, 18));
        }
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        let mut buf = Vec::new();
        w.write(&mut buf, &refs).unwrap();
        std::fs::write(path, &buf).unwrap();
        payloads
    }

    #[test]
    fn split_merge_roundtrip_and_split_kv() {
        let dir = std::env::temp_dir().join("gguf-split-test");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let payloads = synth_model(&model);
        let g0 = Gguf::open_single(&model).unwrap();

        // --- split by tensors: max 2 per file → 4 files
        let out = dir.join("parts-t").join("model.gguf");
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        let params = SplitParams {
            operation: SplitOperation::Split,
            mode: SplitMode::Tensor,
            n_split_tensors: 2,
            input: model.to_string_lossy().into_owned(),
            output: out.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let strategy = SplitStrategy::new(&params, &g0);
        assert_eq!(strategy.ctx_outs.len(), 4);
        strategy.write().unwrap();

        // split.* convention on every part
        let mut merged_tensors = Vec::new();
        for i in 0..4 {
            let p = split_path(&out.to_string_lossy(), i, 4);
            let g = Gguf::open_single(&p).unwrap();
            assert_eq!(g.find_key(LLM_KV_SPLIT_NO).and_then(|v| v.as_u16()), Some(i as u16));
            assert_eq!(g.find_key(LLM_KV_SPLIT_COUNT).and_then(|v| v.as_u16()), Some(4));
            assert_eq!(g.get_u32(LLM_KV_SPLIT_TENSORS_COUNT), Some(7));
            if i == 0 {
                // metadata lives in the first split only
                assert_eq!(g.get_str("general.architecture"), Some("test"));
                // kv order: model kv, then split.no / split.tensors.count with
                // split.count last — the final set_val(SPLIT_COUNT, n) moves
                // it to the end (gguf.cpp remove+append)
                let keys: Vec<&str> = g.kv.iter().map(|(k, _)| k.as_str()).collect();
                assert_eq!(keys, [
                    "general.architecture", "test.block_count", "general.alignment",
                    "split.no", "split.tensors.count", "split.count"
                ]);
            } else {
                assert_eq!(g.get_str("general.architecture"), None);
                assert_eq!(g.kv.len(), 3);
            }
            assert_eq!(g.tensors.len(), if i < 3 { 2 } else { 1 });
            merged_tensors.extend(g.tensors.iter().map(|t| (t.name.clone(), t.ty, t.ne, t.offset)));
        }
        // tensor order and per-file offsets restart at 0
        for i in 0..4 {
            let p = split_path(&out.to_string_lossy(), i, 4);
            let g = Gguf::open_single(&p).unwrap();
            assert_eq!(g.tensors[0].offset, 0);
        }
        let names: Vec<String> = merged_tensors.iter().map(|(n, ..)| n.clone()).collect();
        assert_eq!(names, g0.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>());

        // --- the reader (Gguf::open) merges the parts natively: every tensor
        // resolves to the same bytes as the single-file model
        let gsplit = Gguf::open(split_path(&out.to_string_lossy(), 0, 4)).unwrap();
        assert_eq!(gsplit.tensors.len(), 7);
        assert_eq!(gsplit.get_str("general.architecture"), Some("test"));
        for (i, t) in g0.tensors.iter().enumerate() {
            assert_eq!(gsplit.tensors[i].name, t.name);
            assert_eq!(gsplit.tensor_data(&t.name).unwrap(), g0.tensor_data(&t.name).unwrap());
        }
        // the data of parts 2+ really comes from the other files (part index set)
        assert_eq!(gsplit.tensors[6].part, 3);

        // --- merge back: split.count = 0, split.no/tensors.count kept
        let merged = dir.join("merged.gguf");
        let mparams = SplitParams {
            operation: SplitOperation::Merge,
            input: split_path(&out.to_string_lossy(), 0, 4),
            output: merged.to_string_lossy().into_owned(),
            ..Default::default()
        };
        gguf_merge(&mparams);
        let gm = Gguf::open_single(&merged).unwrap();
        assert_eq!(gm.tensors.len(), 7);
        assert_eq!(gm.find_key(LLM_KV_SPLIT_COUNT).and_then(|v| v.as_u16()), Some(0));
        assert_eq!(gm.find_key(LLM_KV_SPLIT_NO).and_then(|v| v.as_u16()), Some(0));
        // split.count was removed from its old slot and re-appended last
        let keys: Vec<&str> = gm.kv.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.last(), Some(&LLM_KV_SPLIT_COUNT));
        for (i, p) in payloads.iter().enumerate() {
            let name = &g0.tensors[i].name;
            assert_eq!(gm.tensor_data(name).unwrap(), p.as_slice());
        }
        // merged file is a valid single-file model for the split-aware reader
        let gm2 = Gguf::open(&merged).unwrap();
        assert_eq!(gm2.tensors.len(), 7);

        // --- size mode: max 800 bytes per file
        let out2 = dir.join("parts-s").join("model.gguf");
        std::fs::create_dir_all(out2.parent().unwrap()).unwrap();
        let sparams = SplitParams {
            operation: SplitOperation::Split,
            mode: SplitMode::Size,
            n_bytes_split: 800,
            input: model.to_string_lossy().into_owned(),
            output: out2.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let strategy2 = SplitStrategy::new(&sparams, &g0);
        // 5 F32 tensors padded to 256 → two per file until the Q4_0s
        assert!(strategy2.ctx_outs.len() >= 2);
        strategy2.write().unwrap();
        let gsplit2 = Gguf::open(split_path(&out2.to_string_lossy(), 0, strategy2.ctx_outs.len() as i32)).unwrap();
        assert_eq!(gsplit2.tensors.len(), 7);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_tensor_first_split_makes_metadata_only_file() {
        let dir = std::env::temp_dir().join("gguf-split-test-ntfs");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        synth_model(&model);
        let g0 = Gguf::open_single(&model).unwrap();

        let out = dir.join("out.gguf");
        let params = SplitParams {
            operation: SplitOperation::Split,
            mode: SplitMode::Tensor,
            n_split_tensors: 10,
            no_tensor_first_split: true,
            input: model.to_string_lossy().into_owned(),
            output: out.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let strategy = SplitStrategy::new(&params, &g0);
        assert_eq!(strategy.ctx_outs.len(), 2);
        assert_eq!(strategy.ctx_outs[0].tensors.len(), 0); // metadata only
        assert_eq!(strategy.ctx_outs[1].tensors.len(), 7);
        strategy.write().unwrap();

        let p1 = Gguf::open_single(split_path(&out.to_string_lossy(), 0, 2)).unwrap();
        assert_eq!(p1.tensors.len(), 0);
        assert_eq!(p1.get_str("general.architecture"), Some("test"));
        let p2 = Gguf::open_single(split_path(&out.to_string_lossy(), 1, 2)).unwrap();
        assert_eq!(p2.tensors.len(), 7);
        assert_eq!(p2.get_str("general.architecture"), None); // no model kv

        // the split-aware reader still reconstructs the full model
        let gm = Gguf::open(split_path(&out.to_string_lossy(), 0, 2)).unwrap();
        assert_eq!(gm.tensors.len(), 7);
        assert_eq!(gm.get_str("general.architecture"), Some("test"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_and_size_parsing() {
        let mut p = SplitParams::default();
        split_params_parse_ex(
            &[
                "gguf-split".into(),
                "--split".into(),
                "--split-max-size".into(),
                "128M".into(),
                "in.gguf".into(),
                "out.gguf".into(),
            ],
            &mut p,
        )
        .unwrap();
        assert_eq!(p.n_bytes_split, 128 * 1000 * 1000);
        assert_eq!(p.operation, SplitOperation::Split);
        assert_eq!(p.mode, SplitMode::Size);
        assert_eq!(p.input, "in.gguf");

        // underscore normalization (std::replace '_' → '-')
        let mut p2 = SplitParams::default();
        split_params_parse_ex(
            &["gguf-split".into(), "--dry_run".into(), "a".into(), "b".into()],
            &mut p2,
        )
        .unwrap();
        assert!(p2.dry_run);

        assert_eq!(split_str_to_n_bytes("4G").unwrap(), 4_000_000_000);
        assert!(split_str_to_n_bytes("4X").is_err());
        assert!(split_str_to_n_bytes("0M").is_err());

        // mutual exclusions
        let mut p3 = SplitParams::default();
        assert!(split_params_parse_ex(
            &["x".into(), "--merge".into(), "--split".into(), "a".into(), "b".into()],
            &mut p3
        )
        .is_err());
        let mut p4 = SplitParams::default();
        assert!(split_params_parse_ex(
            &[
                "x".into(), "--split-max-tensors".into(), "4".into(),
                "--split-max-size".into(), "1G".into(), "a".into(), "b".into(),
            ],
            &mut p4
        )
        .is_err());
        // arity error
        let mut p5 = SplitParams::default();
        assert!(split_params_parse_ex(&["x".into(), "a".into()], &mut p5).is_err());
        // leading-flags-only loop: a bare "-h" is a positional (reference quirk)
        let mut p6 = SplitParams::default();
        assert_eq!(
            split_params_parse_ex(&["x".into(), "-h".into()], &mut p6).unwrap_err(),
            "error: bad arguments"
        );
        assert_eq!(atoi("128abc"), 128);
        assert_eq!(atoi("  -7"), -7);
        assert_eq!(atoi("junk"), 0);
    }

    #[test]
    fn split_naming_helpers() {
        assert_eq!(split_path("m", 0, 3), "m-00001-of-00003.gguf");
        assert_eq!(split_path("m", 2, 3), "m-00003-of-00003.gguf");
        assert_eq!(split_prefix("m-00002-of-00003.gguf", 1, 3).unwrap(), "m");
        assert_eq!(split_prefix("m.gguf", 0, 1), None);
        assert_eq!(split_prefix("m-00002-of-00003.gguf", 0, 3), None);
        assert_eq!(Value::Array(GgufType::Uint16, vec![]).as_u16(), None);
        assert_eq!(Value::U16(3).as_u16(), Some(3));
    }
}
