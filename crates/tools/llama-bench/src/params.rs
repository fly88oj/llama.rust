//! `cmd_params` + `parse_cmd_params` + `get_cmd_params_instances`
//! (llama-bench.cpp:217-258 enums, :260-355 defaults, :357-460 usage,
//! :462-489 `ggml_type_from_name`, :491-1245 `parse_cmd_params`,
//! :1247-1452 `get_cmd_params_instances`).
//!
//! Every flag of the pinned revision is still *parsed* (so the same command
//! line is accepted and the same instance matrix is built), but the CPU-only
//! engine cannot honour some of them; each one is reported once on stderr —
//! see `warn_ignored`.
//!
//! Not replicated: libstdc++'s `std::stoi`/`stof` accept a numeric *prefix*
//! ("8abc" -> 8), the Rust parses are strict; the failure lines still print the
//! same `error: stoi` / `error: stof` text.

use ggml::types::GgmlType;

use crate::util;

// ---------------------------------------------------------------------------
// enums (llama-bench.cpp:217-258)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    None,
    Csv,
    Json,
    Jsonl,
    Markdown,
    Sql,
}

impl OutputFormat {
    /// `output_format_str` (llama-bench.cpp:219-236)
    pub fn as_str(self) -> &'static str {
        match self {
            OutputFormat::None => "none",
            OutputFormat::Csv => "csv",
            OutputFormat::Json => "json",
            OutputFormat::Jsonl => "jsonl",
            OutputFormat::Markdown => "md",
            OutputFormat::Sql => "sql",
        }
    }

    /// `output_format_from_str` (llama-bench.cpp:238-256)
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "none" => OutputFormat::None,
            "csv" => OutputFormat::Csv,
            "json" => OutputFormat::Json,
            "jsonl" => OutputFormat::Jsonl,
            "md" => OutputFormat::Markdown,
            "sql" => OutputFormat::Sql,
            _ => return None,
        })
    }
}

/// `llama_split_mode` — `split_mode_str` at llama-bench.cpp:1798-1811
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SplitMode::None => "none",
            SplitMode::Layer => "layer",
            SplitMode::Row => "row",
            SplitMode::Tensor => "tensor",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "none" => SplitMode::None,
            "layer" => SplitMode::Layer,
            "row" => SplitMode::Row,
            "tensor" => SplitMode::Tensor,
            _ => return None,
        })
    }
}

/// `llama_load_mode` — `llama_load_mode_name` (src/llama.cpp:36-50)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl LoadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            LoadMode::Auto => "auto",
            LoadMode::None => "none",
            LoadMode::Mmap => "mmap",
            LoadMode::Mlock => "mlock",
            LoadMode::MmapMlock => "mmap+mlock",
            LoadMode::DirectIo => "dio",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => LoadMode::Auto,
            "none" => LoadMode::None,
            "mmap" => LoadMode::Mmap,
            "mlock" => LoadMode::Mlock,
            "mmap+mlock" => LoadMode::MmapMlock,
            "dio" => LoadMode::DirectIo,
            _ => return None,
        })
    }
}

/// `llama_lazy_mode` — `lazy_mode_str` (llama-bench.cpp:1813-1826)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LazyMode {
    Off,
    Auto,
    On,
}

impl LazyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            LazyMode::Off => "off",
            LazyMode::Auto => "auto",
            LazyMode::On => "on",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "on" => LazyMode::On,
            "auto" => LazyMode::Auto,
            "off" => LazyMode::Off,
            _ => return None,
        })
    }
}

/// `llama_flash_attn_type` — the *numeric* value is printed in the `fa` column
/// (`std::to_string((int) flash_attn)`, llama-bench.cpp:1723).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlashAttnType {
    Disabled = 0,
    Enabled = 1,
    Auto = -1,
}

impl FlashAttnType {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
    /// `common_arg_utils::is_truthy/is_falsey/is_autoy` (common/arg.cpp:1331-1341)
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "on" | "enabled" | "true" | "1" => FlashAttnType::Enabled,
            "off" | "disabled" | "false" | "0" => FlashAttnType::Disabled,
            "auto" | "-1" => FlashAttnType::Auto,
            _ => return None,
        })
    }
    /// The value the graph finally uses. `LLAMA_FLASH_ATTN_TYPE_AUTO` is
    /// resolved by `llama_context::resolve_fused_ops` (llama-context.cpp:
    /// 503-560): the probe runs when the fused FLASH_ATTN node of a 1-token
    /// graph lands on the layer's own device — true for every model of this
    /// CPU-only engine, so AUTO == ENABLED here (verified against the
    /// reference binary: default and `-fa on` produce the same graph).
    pub fn enabled(self) -> bool {
        !matches!(self, FlashAttnType::Disabled)
    }
}

/// `std::vector<ggml_backend_dev_t>` of llama-bench.cpp (only the three states
/// reachable without GPU backends: `{}` = auto, `{nullptr}` = none, and the
/// explicit list, which the C rejects for CPU devices).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Devices {
    Auto,
    None,
    /// the C's explicit device list; unreachable here because
    /// `parse_devices_arg` rejects CPU devices and this engine has no others,
    /// kept so `as_str` / the markdown column match the C
    #[allow(dead_code)]
    List(Vec<String>),
}

impl Devices {
    /// `devices_to_string` (llama-bench.cpp:187-215)
    pub fn as_str(&self) -> String {
        match self {
            Devices::Auto => "auto".to_string(),
            Devices::None => "none".to_string(),
            Devices::List(v) => v.join("/"),
        }
    }
}

/// One `llama_model_tensor_buft_override`: (pattern, buffer type name). The
/// trailing `(nullptr, nullptr)` sentinel of every `-ot` group is kept in the
/// vector, exactly like the C (llama-bench.cpp:963-1006).
pub type TensorBuftOverride = (Option<String>, Option<String>);

/// `llama_max_devices()` (llama.h) — the port's CPU-only engine never has more.
pub const LLAMA_MAX_DEVICES: usize = 16;

// ---------------------------------------------------------------------------
// cmd_params (llama-bench.cpp:282-355)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CmdParams {
    pub model: Vec<String>,
    pub n_prompt: Vec<i32>,
    pub n_gen: Vec<i32>,
    pub n_pg: Vec<(i32, i32)>,
    pub n_depth: Vec<i32>,
    pub n_batch: Vec<i32>,
    pub n_ubatch: Vec<i32>,
    pub type_k: Vec<GgmlType>,
    pub type_v: Vec<GgmlType>,
    pub n_threads: Vec<i32>,
    pub cpu_mask: Vec<String>,
    pub cpu_strict: Vec<bool>,
    pub poll: Vec<i32>,
    pub n_gpu_layers: Vec<i32>,
    pub n_cpu_moe: Vec<i32>,
    pub split_mode: Vec<SplitMode>,
    pub load_mode: Vec<LoadMode>,
    pub lazy_mode: Vec<LazyMode>,
    pub main_gpu: Vec<i32>,
    pub no_kv_offload: Vec<bool>,
    pub flash_attn: Vec<FlashAttnType>,
    pub devices: Vec<Devices>,
    pub tensor_split: Vec<Vec<f32>>,
    pub tensor_buft_overrides: Vec<Vec<TensorBuftOverride>>,
    pub embeddings: Vec<bool>,
    pub no_op_offload: Vec<bool>,
    pub no_host: Vec<bool>,
    pub repack: Vec<bool>,
    pub fit_params_target: Vec<u64>,
    pub fit_params_min_ctx: Vec<u32>,
    /// `--numa` (ggml_numa_strategy) — accepted, not honoured (no NUMA engine)
    pub numa: Option<String>,
    pub reps: i32,
    pub prio: i32,
    pub delay: i32,
    pub verbose: bool,
    pub progress: bool,
    pub no_warmup: bool,
    pub output_format: OutputFormat,
    pub output_format_stderr: OutputFormat,
    /// rust-only diagnostic: print the tokens actually evaluated per test on
    /// stderr (`--rust-token-counts`); the table output is unaffected
    pub rust_token_counts: bool,
}

impl Default for CmdParams {
    /// `cmd_params_defaults` (llama-bench.cpp:310-355)
    fn default() -> Self {
        CmdParams {
            model: vec!["models/7B/ggml-model-q4_0.gguf".to_string()],
            n_prompt: vec![512],
            n_gen: vec![128],
            n_pg: vec![],
            n_depth: vec![0],
            n_batch: vec![2048],
            n_ubatch: vec![512],
            type_k: vec![GgmlType::F16],
            type_v: vec![GgmlType::F16],
            n_threads: vec![util::cpu_get_num_math()],
            cpu_mask: vec!["0x0".to_string()],
            cpu_strict: vec![false],
            poll: vec![50],
            n_gpu_layers: vec![-1],
            n_cpu_moe: vec![0],
            split_mode: vec![SplitMode::Layer],
            load_mode: vec![LoadMode::Auto],
            lazy_mode: vec![LazyMode::Auto],
            main_gpu: vec![0],
            no_kv_offload: vec![false],
            flash_attn: vec![FlashAttnType::Auto],
            devices: vec![Devices::Auto],
            tensor_split: vec![vec![0.0; LLAMA_MAX_DEVICES]],
            tensor_buft_overrides: vec![vec![(None, None)]],
            embeddings: vec![false],
            no_op_offload: vec![false],
            no_host: vec![false],
            // llama_model_default_params().use_extra_bufts = true
            repack: vec![true],
            fit_params_target: vec![0],
            fit_params_min_ctx: vec![0],
            numa: None,
            reps: 5,
            prio: 0,
            delay: 0,
            verbose: false,
            progress: false,
            no_warmup: false,
            output_format: OutputFormat::Markdown,
            output_format_stderr: OutputFormat::None,
            rust_token_counts: false,
        }
    }
}

// ---------------------------------------------------------------------------
// helpers: string_split / parse_int_range (llama-bench.cpp:1828-1878)
// ---------------------------------------------------------------------------

/// `string_split<std::string>(s, delim)` (common/common.h:817-831): empty
/// fields are kept, nothing is trimmed. `Some("")` for a failed string.
pub fn string_split(s: &str, delim: char) -> Vec<String> {
    s.split(delim).map(|p| p.to_string()).collect()
}

/// `string_split<bool>` (common/common.h:802-815): `istringstream >> bool`,
/// i.e. an integer-style parse where anything that fails becomes false.
pub fn string_split_bool(s: &str, delim: char) -> Vec<bool> {
    s.split(delim).map(|t| matches!(t.trim().parse::<i64>(), Ok(v) if v != 0)).collect()
}

/// `std::regex("[;/]+")` token split (llama-bench.cpp:890)
fn split_on_runs(s: &str, seps: &[char]) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if seps.contains(&c) {
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    parts.push(cur);
    parts
}

/// `parse_int_range` (llama-bench.cpp:1844-1878): `first[-last[(+|*)step]]`,
/// repeated, comma separated. Faithful to the anchored regex: a token that is
/// not of that shape aborts the parse instead of being skipped, `*` multiplies,
/// `+` adds, and a non-advancing range is rejected.
pub fn parse_int_range(s: &str, allow_negative: bool) -> Result<Vec<i32>, String> {
    let mut out = Vec::new();
    let mut rest = s;
    loop {
        if rest.is_empty() {
            return Ok(out);
        }
        let bytes = rest.as_bytes();
        let mut i = 0;
        let first = parse_int_at(rest, &mut i, allow_negative).ok_or_else(|| "invalid range format".to_string())?;
        let mut last = first;
        let mut op = '+';
        let mut step = 1;
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            last = parse_int_at(rest, &mut i, false).ok_or_else(|| "invalid range format".to_string())?;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'*') {
                op = bytes[i] as char;
                i += 1;
                step = parse_int_at(rest, &mut i, false).ok_or_else(|| "invalid range format".to_string())?;
            }
        }
        // `(?:,|$)` after every element (the comma is part of the match)
        if i < bytes.len() {
            if bytes[i] != b',' {
                return Err("invalid range format".to_string());
            }
            i += 1;
        }
        let mut v = first;
        while v <= last {
            out.push(v);
            let prev = v;
            match op {
                // the C's int arithmetic; wrapping keeps release/debug identical
                '+' => v = v.wrapping_add(step),
                '*' => v = v.wrapping_mul(step),
                _ => return Err("invalid range format".to_string()),
            }
            if v <= prev {
                return Err("invalid range".to_string());
            }
        }
        rest = &rest[i..];
    }
}

fn parse_int_at(s: &str, i: &mut usize, allow_negative: bool) -> Option<i32> {
    let b = s.as_bytes();
    let start = *i;
    if allow_negative && b.get(start) == Some(&b'-') {
        *i += 1;
    }
    let digits_start = *i;
    while *i < b.len() && b[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i == digits_start {
        *i = start;
        return None;
    }
    s[start..*i].parse::<i32>().ok()
}

/// `ggml_type_name` (ggml.c's `type_traits[].type_name`) — the port's
/// `GgmlType::name` uses the C *macros'* spelling ("F16"), llama-bench prints
/// the trait table's lowercase one ("f16").
pub fn ggml_type_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F64 => "f64",
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q1_0 => "q1_0",
        GgmlType::Q2_0 => "q2_0",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q5_1 => "q5_1",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q8_1 => "q8_1",
        GgmlType::Mxfp4 => "mxfp4",
        GgmlType::Nvfp4 => "nvfp4",
        GgmlType::Q2K => "q2_K",
        GgmlType::Q3K => "q3_K",
        GgmlType::Q4K => "q4_K",
        GgmlType::Q5K => "q5_K",
        GgmlType::Q6K => "q6_K",
        GgmlType::Q8K => "q8_K",
        GgmlType::Iq2Xxs => "iq2_xxs",
        GgmlType::Iq2Xs => "iq2_xs",
        GgmlType::Iq3Xxs => "iq3_xxs",
        GgmlType::Iq3S => "iq3_s",
        GgmlType::Iq2S => "iq2_s",
        GgmlType::Iq1S => "iq1_s",
        GgmlType::Iq1M => "iq1_m",
        GgmlType::Iq4Nl => "iq4_nl",
        GgmlType::Iq4Xs => "iq4_xs",
        GgmlType::Bf16 => "bf16",
        GgmlType::Tq1_0 => "tq1_0",
        GgmlType::Tq2_0 => "tq2_0",
        GgmlType::I8 => "i8",
        GgmlType::I16 => "i16",
        GgmlType::I32 => "i32",
        GgmlType::I64 => "i64",
    }
}

/// `ggml_type_from_name` (llama-bench.cpp:462-489) — only these names are
/// accepted for `-ctk`/`-ctv` at this revision.
pub fn ggml_type_from_name(s: &str) -> Option<GgmlType> {
    Some(match s {
        "f16" => GgmlType::F16,
        "bf16" => GgmlType::Bf16,
        "q8_0" => GgmlType::Q8_0,
        "q4_0" => GgmlType::Q4_0,
        "q4_1" => GgmlType::Q4_1,
        "q5_0" => GgmlType::Q5_0,
        "q5_1" => GgmlType::Q5_1,
        "iq4_nl" => GgmlType::Iq4Nl,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// parse_cmd_params (llama-bench.cpp:491-1245)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ParseError {
    /// the argument being handled, for the C's final message
    pub arg: String,
    /// optional `error: ...` line printed first (the C's catch block)
    pub detail: Option<String>,
}

pub fn print_usage(prog: &str) {
    // llama-bench.cpp:357-460 (help text; the Rust-only flag is marked as such)
    let d = CmdParams::default();
    let joined = |v: &[String]| v.join(",");
    let joined_i = |v: &[i32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let joined_b = |v: &[bool]| v.iter().map(|x| if *x { "1" } else { "0" }).collect::<Vec<_>>().join(",");
    println!("usage: {prog} [options]");
    println!();
    println!("options:");
    println!("  -h, --help");
    println!("  --version                                   show version and build info");
    println!("  --numa <distribute|isolate|numactl>         numa mode (default: disabled)");
    println!("  -r, --repetitions <n>                       number of times to repeat each test (default: {})", d.reps);
    println!("  --prio <-1|0|1|2|3>                         process/thread priority (default: {})", d.prio);
    println!("  --delay <0...N> (seconds)                   delay between each test (default: {})", d.delay);
    println!("  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: {})", d.output_format.as_str());
    println!("  -oe, --output-err <csv|json|jsonl|md|sql>   output format printed to stderr (default: {})", d.output_format_stderr.as_str());
    println!("  --list-devices                              list available devices and exit");
    println!("  -v, --verbose                               verbose output");
    println!("  --progress                                  print test progress indicators");
    println!("  --no-warmup                                 skip warmup runs before benchmarking");
    println!("  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)");
    println!("  -fitc, --fit-ctx <n>                        minimum ctx size for --fit-target (default: 4096)");
    println!("  --rust-token-counts                         rust-only: report tokens evaluated per test on stderr");
    println!();
    println!("test parameters:");
    println!("  -m, --model <filename>                            (default: {})", joined(&d.model));
    println!("  -hf, -hfr, --hf-repo <user>/<model>[:quant]       Hugging Face model repository (not supported in this port)");
    println!("  -hff, --hf-file <file>                            Hugging Face model file (not supported in this port)");
    println!("  -hft, --hf-token <token>                          Hugging Face access token (not supported in this port)");
    println!("  --offline                                         Offline mode (not supported in this port)");
    println!("  -p, --n-prompt <n>                                (default: {})", joined_i(&d.n_prompt));
    println!("  -n, --n-gen <n>                                   (default: {})", joined_i(&d.n_gen));
    println!("  -pg <pp,tg>                                       (default: {})", d.n_pg.iter().map(|(a, b)| format!("{a},{b}")).collect::<Vec<_>>().join(","));
    println!("  -d, --n-depth <n>                                 (default: {})", joined_i(&d.n_depth));
    println!("  -b, --batch-size <n>                              (default: {})", joined_i(&d.n_batch));
    println!("  -ub, --ubatch-size <n>                            (default: {})", joined_i(&d.n_ubatch));
    println!("  -ctk, --cache-type-k <t>                          (default: {})", d.type_k.iter().map(|t| crate::params::ggml_type_name(*t)).collect::<Vec<_>>().join(","));
    println!("  -ctv, --cache-type-v <t>                          (default: {})", d.type_v.iter().map(|t| crate::params::ggml_type_name(*t)).collect::<Vec<_>>().join(","));
    println!("  -t, --threads <n>                                 (default: {})", joined_i(&d.n_threads));
    println!("  -C, --cpu-mask <hex,hex>                          (default: {})", joined(&d.cpu_mask));
    println!("  --cpu-strict <0|1>                                (default: {})", joined_b(&d.cpu_strict));
    println!("  --poll <0...100>                                  (default: {})", joined_i(&d.poll));
    println!("  -ngl, --n-gpu-layers <n>                          (default: {})", joined_i(&d.n_gpu_layers));
    println!("  -ncmoe, --n-cpu-moe <n>                           (default: {})", joined_i(&d.n_cpu_moe));
    println!("  -sm, --split-mode <none|layer|row|tensor>         (default: {})", d.split_mode.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(","));
    println!("  -mg, --main-gpu <i>                               (default: {})", joined_i(&d.main_gpu));
    println!("  -nkvo, --no-kv-offload <0|1>                      (default: {})", joined_b(&d.no_kv_offload));
    println!("  -fa, --flash-attn <on|off|auto>                   (default: {})", d.flash_attn.iter().map(|f| match f {
        FlashAttnType::Enabled => "on",
        FlashAttnType::Disabled => "off",
        FlashAttnType::Auto => "auto",
    }).collect::<Vec<_>>().join(","));
    println!("  -dev, --device <dev0/dev1/...>                    (default: auto)");
    println!("  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: {})", d.load_mode.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(","));
    println!("  -lzm, --lazy-mode <on|auto|off>                   (default: {})", d.lazy_mode.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(","));
    println!("  -embd, --embeddings <0|1>                         (default: {})", joined_b(&d.embeddings));
    println!("  -ts, --tensor-split <ts0/ts1/..>                  (default: 0)");
    println!("  -ot --override-tensor <tensor name pattern>=<buffer type>;...");
    println!("                                                    (default: disabled)");
    println!("  -nopo, --no-op-offload <0|1>                      (default: 0)");
    println!("  --no-host <0|1>                                   (default: {})", joined_b(&d.no_host));
    println!("  --repack <0|1>                                    (default: {})", joined_b(&d.repack));
    println!();
    println!("Multiple values can be given for each parameter by separating them with ','");
    println!("or by specifying the parameter multiple times. Ranges can be given as");
    println!("'first-last' or 'first-last+step' or 'first-last*mult'.");
}

pub fn parse_cmd_params(argv: &[String]) -> Result<CmdParams, ParseError> {
    let mut params = CmdParams {
        // scalar fields start at their defaults, vector fields start empty so
        // the "set defaults" pass at the end can fill them (llama-bench.cpp:1075-1160)
        model: vec![],
        n_prompt: vec![],
        n_gen: vec![],
        n_pg: vec![],
        n_depth: vec![],
        n_batch: vec![],
        n_ubatch: vec![],
        type_k: vec![],
        type_v: vec![],
        n_threads: vec![],
        cpu_mask: vec![],
        cpu_strict: vec![],
        poll: vec![],
        n_gpu_layers: vec![],
        n_cpu_moe: vec![],
        split_mode: vec![],
        load_mode: vec![],
        lazy_mode: vec![],
        main_gpu: vec![],
        no_kv_offload: vec![],
        flash_attn: vec![],
        devices: vec![],
        tensor_split: vec![],
        tensor_buft_overrides: vec![],
        embeddings: vec![],
        no_op_offload: vec![],
        no_host: vec![],
        repack: vec![],
        fit_params_target: vec![],
        fit_params_min_ctx: vec![],
        ..Default::default()
    };

    let invalid = |arg: &str| ParseError { arg: arg.to_string(), detail: None };
    let invalid_detail = |arg: &str, detail: String| ParseError { arg: arg.to_string(), detail: Some(detail) };

    let mut i = 1;
    while i < argv.len() {
        // the C replaces '_' with '-' in any "--" argument before matching
        let mut arg = argv[i].clone();
        if arg.starts_with("--") {
            arg = arg.replace('_', "-");
        }
        let next = |i: &mut usize| -> Option<String> {
            *i += 1;
            argv.get(*i).cloned()
        };
        let s = arg.as_str();
        // value of the *current* option, for the arms that take one
        macro_rules! val {
            () => {
                match next(&mut i) {
                    Some(v) => v,
                    None => return Err(invalid(s)),
                }
            };
        }
        match s {
            "-h" | "--help" => {
                print_usage(&argv[0]);
                std::process::exit(0);
            }
            "--version" => {
                // llama_print_build_info(llama_version()) (common/common.cpp:1197);
                // the second line names the port's compiler instead of the C's
                println!("version: {} (build {}, commit {})", crate::BUILD_VERSION, crate::BUILD_NUMBER, crate::BUILD_COMMIT);
                println!("built with rustc for {} {}", std::env::consts::OS, std::env::consts::ARCH);
                std::process::exit(0);
            }
            "-m" | "--model" => {
                let v = val!();
                params.model.extend(string_split(&v, ','));
            }
            "-hf" | "-hfr" | "--hf-repo" => {
                let _ = val!();
                eprintln!("llama-bench: error: Hugging Face download is not supported in this port");
                std::process::exit(1);
            }
            "-hff" | "--hf-file" | "-hft" | "--hf-token" => {
                let _ = val!();
                eprintln!("llama-bench: error: Hugging Face download is not supported in this port");
                std::process::exit(1);
            }
            "--offline" => {
                eprintln!("llama-bench: error: Hugging Face download is not supported in this port");
                std::process::exit(1);
            }
            "-p" | "--n-prompt" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_prompt.extend(p);
            }
            "-n" | "--n-gen" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_gen.extend(p);
            }
            "-pg" => {
                let v = val!();
                let p = string_split(&v, ',');
                if p.len() != 2 {
                    return Err(invalid(s));
                }
                let a = p[0].trim().parse::<i32>().map_err(|_| invalid_detail(s, "error: stoi".to_string()))?;
                let b = p[1].trim().parse::<i32>().map_err(|_| invalid_detail(s, "error: stoi".to_string()))?;
                params.n_pg.push((a, b));
            }
            "-d" | "--n-depth" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_depth.extend(p);
            }
            "-b" | "--batch-size" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_batch.extend(p);
            }
            "-ub" | "--ubatch-size" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_ubatch.extend(p);
            }
            "-ctk" | "--cache-type-k" => {
                let v = val!();
                let mut types = Vec::new();
                for t in string_split(&v, ',') {
                    match ggml_type_from_name(&t) {
                        Some(gt) => types.push(gt),
                        None => return Err(invalid(s)),
                    }
                }
                params.type_k.extend(types);
            }
            "-ctv" | "--cache-type-v" => {
                let v = val!();
                let mut types = Vec::new();
                for t in string_split(&v, ',') {
                    match ggml_type_from_name(&t) {
                        Some(gt) => types.push(gt),
                        None => return Err(invalid(s)),
                    }
                }
                params.type_v.extend(types);
            }
            "-dev" | "--device" => {
                let v = val!();
                for combo in string_split(&v, ',') {
                    let trimmed = combo.trim();
                    if trimmed.is_empty() {
                        eprintln!("error: no devices specified");
                        return Err(invalid(s));
                    }
                    if trimmed == "auto" {
                        params.devices.push(Devices::Auto);
                    } else if trimmed == "none" {
                        params.devices.push(Devices::None);
                    } else {
                        // `parse_devices_arg` (llama-bench.cpp:133-169) rejects
                        // CPU devices ("invalid device: %s"); this engine has
                        // no non-CPU devices at all
                        let name = trimmed.split('/').next().unwrap_or(trimmed).trim();
                        eprintln!("error: invalid device: {name}");
                        return Err(invalid(s));
                    }
                }
            }
            "--list-devices" => {
                // `common_print_available_devices` (common/arg.cpp:1138-1163)
                // — only non-CPU devices are listed, and this port has none
                println!("Available devices:");
                println!("  (none)");
                std::process::exit(0);
            }
            "-t" | "--threads" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_threads.extend(p);
            }
            "-C" | "--cpu-mask" => {
                let v = val!();
                params.cpu_mask.extend(string_split(&v, ','));
            }
            "--cpu-strict" => {
                let v = val!();
                params.cpu_strict.extend(string_split_bool(&v, ','));
            }
            "--poll" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.poll.extend(p);
            }
            "-ngl" | "--n-gpu-layers" => {
                let v = val!();
                let p = parse_int_range(&v, true).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_gpu_layers.extend(p);
            }
            "-ncmoe" | "--n-cpu-moe" => {
                let v = val!();
                let p = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
                params.n_cpu_moe.extend(p);
            }
            "-sm" | "--split-mode" => {
                let v = val!();
                let mut modes = Vec::new();
                for m in string_split(&v, ',') {
                    match SplitMode::from_str(&m) {
                        Some(mode) => modes.push(mode),
                        None => return Err(invalid(s)),
                    }
                }
                params.split_mode.extend(modes);
            }
            "-lm" | "--load-mode" => {
                let v = val!();
                let mut modes = Vec::new();
                for m in string_split(&v, ',') {
                    match LoadMode::from_str(&m) {
                        Some(mode) => modes.push(mode),
                        None => return Err(invalid(s)),
                    }
                }
                params.load_mode.extend(modes);
            }
            "-lzm" | "--lazy-mode" => {
                let v = val!();
                let mut modes = Vec::new();
                for m in string_split(&v, ',') {
                    match LazyMode::from_str(&m) {
                        Some(mode) => modes.push(mode),
                        None => return Err(invalid(s)),
                    }
                }
                params.lazy_mode.extend(modes);
            }
            "-mg" | "--main-gpu" => {
                let v = val!();
                params.main_gpu = parse_int_range(&v, false).map_err(|e| invalid_detail(s, format!("error: {e}")))?;
            }
            "-nkvo" | "--no-kv-offload" => {
                let v = val!();
                params.no_kv_offload.extend(string_split_bool(&v, ','));
            }
            "--numa" => {
                let v = val!();
                if !matches!(v.as_str(), "distribute" | "" | "isolate" | "numactl") {
                    return Err(invalid(s));
                }
                // `llama_numa_init(params.numa)` — the CPU-only engine has no
                // NUMA strategy (reported in warn_ignored)
                params.numa = Some(v);
            }
            "-fa" | "--flash-attn" => {
                let v = val!();
                let mut types = Vec::new();
                for t in string_split(&v, ',') {
                    match FlashAttnType::from_str(&t) {
                        Some(ft) => types.push(ft),
                        None => return Err(invalid(s)),
                    }
                }
                params.flash_attn.extend(types);
            }
            "-embd" | "--embeddings" => {
                let v = val!();
                params.embeddings.extend(string_split_bool(&v, ','));
            }
            "-nopo" | "--no-op-offload" => {
                let v = val!();
                params.no_op_offload.extend(string_split_bool(&v, ','));
            }
            "--no-host" => {
                let v = val!();
                params.no_host.extend(string_split_bool(&v, ','));
            }
            "--repack" => {
                let v = val!();
                params.repack.extend(string_split_bool(&v, ','));
            }
            "-ts" | "--tensor-split" => {
                let v = val!();
                for ts in string_split(&v, ',') {
                    // llama-bench.cpp:886-905: split on runs of ';' or '/'
                    // (the C's `[;/]+` regex), pad to llama_max_devices()
                    let parts = split_on_runs(&ts, &[';', '/']);
                    if parts.len() > LLAMA_MAX_DEVICES {
                        return Err(invalid(s));
                    }
                    let mut split = vec![0.0f32; LLAMA_MAX_DEVICES];
                    for (k, p) in parts.iter().enumerate() {
                        split[k] = p.trim().parse::<f32>().map_err(|_| invalid_detail(s, "error: stof".to_string()))?;
                    }
                    params.tensor_split.push(split);
                }
            }
            "-ot" | "--override-tensor" => {
                let v = val!();
                // llama-bench.cpp:908-1013: groups separated by ',', entries by
                // ';', `pattern=buft`; the only buffer type of a CPU-only build
                // is "CPU" (ggml-cpu.cpp's buf_type_cpu name)
                for group in string_split(&v, ',') {
                    let mut overrides: Vec<TensorBuftOverride> = Vec::new();
                    if group.is_empty() {
                        overrides.push((None, None));
                        params.tensor_buft_overrides.push(overrides);
                        continue;
                    }
                    for entry in group.split(';').filter(|e| !e.is_empty()) {
                        let Some((name, buft)) = entry.split_once('=') else {
                            return Err(invalid(s));
                        };
                        if buft != "CPU" {
                            println!("error: unrecognized buffer type '{buft}'");
                            println!("Available buffer types:");
                            println!("  CPU");
                            return Err(invalid(s));
                        }
                        overrides.push((Some(name.to_string()), Some(buft.to_string())));
                    }
                    overrides.push((None, None)); // sentinel
                    params.tensor_buft_overrides.push(overrides);
                }
            }
            "-r" | "--repetitions" => {
                let v = val!();
                params.reps = v.trim().parse::<i32>().map_err(|_| invalid_detail(s, "error: stoi".to_string()))?;
            }
            "--prio" => {
                let v = val!();
                params.prio = v.trim().parse::<i32>().map_err(|_| invalid_detail(s, "error: stoi".to_string()))?;
            }
            "--delay" => {
                let v = val!();
                params.delay = v.trim().parse::<i32>().map_err(|_| invalid_detail(s, "error: stoi".to_string()))?;
            }
            "-o" | "--output" => {
                let v = val!();
                match OutputFormat::from_str(&v) {
                    Some(f) => params.output_format = f,
                    None => return Err(invalid(s)),
                }
            }
            "-oe" | "--output-err" => {
                let v = val!();
                match OutputFormat::from_str(&v) {
                    Some(f) => params.output_format_stderr = f,
                    None => return Err(invalid(s)),
                }
            }
            "-v" | "--verbose" => params.verbose = true,
            "--progress" => params.progress = true,
            "--no-warmup" => params.no_warmup = true,
            "-fitt" | "--fit-target" => {
                let v = val!();
                for x in string_split(&v, ',') {
                    params.fit_params_target.push(x.trim().parse::<u64>().map_err(|_| invalid_detail(s, "error: stoull".to_string()))?);
                }
            }
            "-fitc" | "--fit-ctx" => {
                let v = val!();
                for x in string_split(&v, ',') {
                    params.fit_params_min_ctx.push(x.trim().parse::<u32>().map_err(|_| invalid_detail(s, "error: stoul".to_string()))?);
                }
            }
            "--rust-token-counts" => params.rust_token_counts = true,
            _ => return Err(invalid(s)),
        }
        i += 1;
    }

    // set defaults (llama-bench.cpp:1075-1160)
    let d = CmdParams::default();
    macro_rules! fill {
        ($f:ident) => {
            if params.$f.is_empty() {
                params.$f = d.$f.clone();
            }
        };
    }
    fill!(model);
    fill!(n_prompt);
    fill!(n_gen);
    fill!(n_pg);
    fill!(n_depth);
    fill!(n_batch);
    fill!(n_ubatch);
    fill!(type_k);
    fill!(type_v);
    fill!(n_gpu_layers);
    fill!(n_cpu_moe);
    fill!(split_mode);
    fill!(load_mode);
    fill!(lazy_mode);
    fill!(main_gpu);
    fill!(no_kv_offload);
    fill!(flash_attn);
    fill!(devices);
    fill!(tensor_split);
    fill!(tensor_buft_overrides);
    fill!(embeddings);
    fill!(no_op_offload);
    fill!(no_host);
    fill!(repack);
    fill!(n_threads);
    fill!(cpu_mask);
    fill!(cpu_strict);
    fill!(poll);
    fill!(fit_params_target);
    fill!(fit_params_min_ctx);

    Ok(params)
}

// ---------------------------------------------------------------------------
// cmd_params_instance (llama-bench.cpp:1164-1245) + expansion
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CmdParamsInstance {
    pub model: String,
    pub n_prompt: i32,
    pub n_gen: i32,
    pub n_depth: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub type_k: GgmlType,
    pub type_v: GgmlType,
    pub n_threads: i32,
    pub cpu_mask: String,
    pub cpu_strict: bool,
    pub poll: i32,
    pub n_gpu_layers: i32,
    pub n_cpu_moe: i32,
    pub split_mode: SplitMode,
    pub load_mode: LoadMode,
    pub lazy_mode: LazyMode,
    pub main_gpu: i32,
    pub no_kv_offload: bool,
    pub flash_attn: FlashAttnType,
    pub devices: Devices,
    pub tensor_split: Vec<f32>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub embeddings: bool,
    pub no_op_offload: bool,
    pub no_host: bool,
    pub repack: bool,
    pub fit_target: u64,
    pub fit_min_ctx: u32,
}

impl CmdParamsInstance {
    /// `to_llama_cparams` (llama-bench.cpp:1225-1240): only the fields the
    /// port's `DecodeContext` takes are kept; `n_ctx = n_prompt + n_gen + n_depth`.
    pub fn n_ctx(&self) -> i32 {
        self.n_prompt + self.n_gen + self.n_depth
    }
}

/// `get_cmd_params_instances` (llama-bench.cpp:1247-1452). The loop order is
/// the C's ("this ordering minimizes the number of times that each model needs
/// to be reloaded", :1350); note the per-outer body emits the `n_prompt`
/// instances first, then `n_gen`, then `n_pg`.
#[allow(clippy::too_many_arguments)]
pub fn get_cmd_params_instances(params: &CmdParams) -> Vec<CmdParamsInstance> {
    let mut instances = Vec::new();

    // clang-format off (the C has the same 26-deep loop nest here)
    for m in &params.model {
    for fpt in &params.fit_params_target {
    for fpc in &params.fit_params_min_ctx {
    for nl in &params.n_gpu_layers {
    for ncmoe in &params.n_cpu_moe {
    for sm in &params.split_mode {
    for lm in &params.load_mode {
    for lzm in &params.lazy_mode {
    for mg in &params.main_gpu {
    for devs in &params.devices {
    for ts in &params.tensor_split {
    for ot in &params.tensor_buft_overrides {
    for noh in &params.no_host {
    for rpk in &params.repack {
    for embd in &params.embeddings {
    for nopo in &params.no_op_offload {
    for nb in &params.n_batch {
    for nub in &params.n_ubatch {
    for tk in &params.type_k {
    for tv in &params.type_v {
    for nkvo in &params.no_kv_offload {
    for fa in &params.flash_attn {
    for nt in &params.n_threads {
    for cm in &params.cpu_mask {
    for cs in &params.cpu_strict {
    for nd in &params.n_depth {
    for pl in &params.poll {
        let base = CmdParamsInstance {
            model: m.clone(),
            n_prompt: 0,
            n_gen: 0,
            n_depth: *nd,
            n_batch: *nb,
            n_ubatch: *nub,
            type_k: *tk,
            type_v: *tv,
            n_threads: *nt,
            cpu_mask: cm.clone(),
            cpu_strict: *cs,
            poll: *pl,
            n_gpu_layers: *nl,
            n_cpu_moe: *ncmoe,
            split_mode: *sm,
            load_mode: *lm,
            lazy_mode: *lzm,
            main_gpu: *mg,
            no_kv_offload: *nkvo,
            flash_attn: *fa,
            devices: devs.clone(),
            tensor_split: ts.clone(),
            tensor_buft_overrides: ot.clone(),
            embeddings: *embd,
            no_op_offload: *nopo,
            no_host: *noh,
            repack: *rpk,
            fit_target: *fpt,
            fit_min_ctx: *fpc,
        };
        for n_prompt in &params.n_prompt {
            if *n_prompt == 0 {
                continue;
            }
            instances.push(CmdParamsInstance { n_prompt: *n_prompt, ..base.clone() });
        }
        for n_gen in &params.n_gen {
            if *n_gen == 0 {
                continue;
            }
            instances.push(CmdParamsInstance { n_prompt: 0, n_gen: *n_gen, ..base.clone() });
        }
        for (a, b) in &params.n_pg {
            if *a == 0 && *b == 0 {
                continue;
            }
            instances.push(CmdParamsInstance { n_prompt: *a, n_gen: *b, ..base.clone() });
        }
    }}}}}}}}}}}}}}}}}}}}}}}}}}}
    // clang-format on

    instances
}

// ---------------------------------------------------------------------------
// rust-only: report the flags the CPU-only engine cannot honour
// ---------------------------------------------------------------------------

/// Prints one warning per flag whose *value* changes the reference's execution
/// but not this engine's (the C's CPU backend ignores several of them silently
/// too, e.g. `n_gpu_layers`). Called once, before the benchmark loop.
pub fn warn_ignored(params: &CmdParams) {
    let d = CmdParams::default();
    let warn = |flag: &str, detail: &str| {
        eprintln!("llama-bench: WARNING: {flag} is ignored in this port (CPU-only engine): {detail}");
    };
    let bools = |v: &[bool]| v.iter().map(|b| if *b { '1' } else { '0' }).collect::<Vec<_>>().iter().collect::<String>();
    let ints = |v: &[i32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    // the KV cache of the port's DecodeContext is F16-only
    // (crates/llama/src/kv_cache.rs:230-231)
    if params.type_k != d.type_k || params.type_v != d.type_v {
        let name = |v: &[GgmlType]| v.iter().map(|t| ggml_type_name(*t)).collect::<Vec<_>>().join(",");
        warn("-ctk/-ctv", &format!("cache types are fixed to f16 (requested {} / {})", name(&params.type_k), name(&params.type_v)));
    }
    if params.n_gpu_layers != d.n_gpu_layers {
        warn("-ngl", &format!("no GPU offload ({})", ints(&params.n_gpu_layers)));
    }
    if params.embeddings != d.embeddings {
        warn("-embd", &format!("no embeddings mode for decoder archs ({})", bools(&params.embeddings)));
    }
    if params.cpu_mask != d.cpu_mask {
        warn("-C/--cpu-mask", &format!("thread affinity is not settable ({})", params.cpu_mask.join(",")));
    }
    if params.cpu_strict != d.cpu_strict {
        warn("--cpu-strict", &format!("thread affinity is not settable ({})", bools(&params.cpu_strict)));
    }
    if params.poll != d.poll {
        warn("--poll", &format!("threadpool poll is not settable ({})", ints(&params.poll)));
    }
    if let Some(numa) = &params.numa {
        if !numa.is_empty() {
            warn("--numa", numa);
        }
    }
    if params.prio != d.prio {
        warn("--prio", &format!("process priority is not settable ({})", params.prio));
    }
    if params.lazy_mode != d.lazy_mode {
        warn("-lzm/--lazy-mode", &format!("weights are materialized eagerly ({})", params.lazy_mode.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(",")));
    }
    if params.load_mode.iter().any(|m| !matches!(m, LoadMode::Auto | LoadMode::Mmap)) {
        warn("-lm/--load-mode", &format!("the loader always mmaps ({})", params.load_mode.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(",")));
    }
    if params.fit_params_target != d.fit_params_target || params.fit_params_min_ctx != d.fit_params_min_ctx {
        warn("-fitt/-fitc", "no device-memory fitting (no `common_fit_params`)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn int_range_matches_the_c_regex() {
        assert_eq!(parse_int_range("512", false).unwrap(), vec![512]);
        assert_eq!(parse_int_range("64-128+32", false).unwrap(), vec![64, 96, 128]);
        assert_eq!(parse_int_range("1-4*2", false).unwrap(), vec![1, 2, 4]);
        // ranges accumulate in one argument, comma separated
        assert_eq!(parse_int_range("2,4-6", false).unwrap(), vec![2, 4, 5, 6]);
        // a trailing comma is accepted (the regex eats it), as is "" (empty vector)
        assert_eq!(parse_int_range("8,", false).unwrap(), vec![8]);
        assert_eq!(parse_int_range("", false).unwrap(), Vec::<i32>::new());
        // negatives only with allow_negative (-ngl)
        assert_eq!(parse_int_range("-1", true).unwrap(), vec![-1]);
        assert!(parse_int_range("-1", false).is_err());
        // a non-advancing range is rejected ("invalid range")
        assert_eq!(parse_int_range("5-5+0", false).unwrap_err(), "invalid range");
        // malformed tokens abort instead of being skipped
        assert_eq!(parse_int_range("abc", false).unwrap_err(), "invalid range format");
        assert_eq!(parse_int_range("1-", false).unwrap_err(), "invalid range format");
        assert_eq!(parse_int_range("1--2", false).unwrap_err(), "invalid range format");
        assert_eq!(parse_int_range("1,,2", false).unwrap_err(), "invalid range format");
    }

    #[test]
    fn split_keeps_empty_fields() {
        // string_split<std::string>: "" fields survive, so `-ctk f16,` is an error
        assert_eq!(string_split("f16,", ','), vec!["f16".to_string(), String::new()]);
        assert_eq!(string_split("a,b", ','), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(string_split("", ','), vec![String::new()]);
        assert_eq!(ggml_type_from_name("f16"), Some(GgmlType::F16));
        assert_eq!(ggml_type_from_name("q4_k"), None); // not in the C's list
        assert_eq!(ggml_type_from_name("iq4_nl"), Some(GgmlType::Iq4Nl));
        assert_eq!(string_split_bool("0,1", ','), vec![false, true]);
    }

    #[test]
    fn arg_parsing_and_defaults() {
        let p = parse_cmd_params(&argv(&["llama-bench"]))
            .unwrap_or_else(|_| panic!("defaults"));
        // cmd_params_defaults (llama-bench.cpp:310-355)
        assert_eq!(p.model, vec!["models/7B/ggml-model-q4_0.gguf".to_string()]);
        assert_eq!(p.n_prompt, vec![512]);
        assert_eq!(p.n_gen, vec![128]);
        assert_eq!(p.n_batch, vec![2048]);
        assert_eq!(p.n_ubatch, vec![512]);
        assert_eq!(p.n_depth, vec![0]);
        assert_eq!(p.type_k, vec![GgmlType::F16]);
        assert_eq!(p.n_gpu_layers, vec![-1]);
        assert_eq!(p.flash_attn, vec![FlashAttnType::Auto]);
        assert_eq!(p.reps, 5);
        assert_eq!(p.output_format, OutputFormat::Markdown);
        assert_eq!(p.output_format_stderr, OutputFormat::None);
        assert_eq!(p.n_threads, vec![util::cpu_get_num_math()]);

        let p = parse_cmd_params(&argv(&[
            "llama-bench",
            "-m",
            "/tmp/a.gguf,/tmp/b.gguf",
            "-p",
            "64",
            "-n",
            "16",
            "-t",
            "8",
            "-r",
            "2",
            "-fa",
            "on",
            "-o",
            "csv",
            "-oe",
            "json",
            "-ub",
            "256",
            "--no-warmup",
            "--progress",
            "-ctk",
            "f16,bf16",
        ]))
        .unwrap();
        assert_eq!(p.model, vec!["/tmp/a.gguf".to_string(), "/tmp/b.gguf".to_string()]);
        assert_eq!(p.n_prompt, vec![64]);
        assert_eq!(p.n_gen, vec![16]);
        assert_eq!(p.n_threads, vec![8]);
        assert_eq!(p.reps, 2);
        assert_eq!(p.flash_attn, vec![FlashAttnType::Enabled]);
        assert_eq!(p.output_format, OutputFormat::Csv);
        assert_eq!(p.output_format_stderr, OutputFormat::Json);
        assert_eq!(p.n_ubatch, vec![256]);
        assert!(p.no_warmup && p.progress);
        assert_eq!(p.type_k, vec![GgmlType::F16, GgmlType::Bf16]);

        // '_' -> '-' mangling only applies to "--" arguments
        let p = parse_cmd_params(&argv(&["llama-bench", "--n_prompt", "32", "--no_warmup"])).unwrap();
        assert_eq!(p.n_prompt, vec![32]);
        assert!(p.no_warmup);

        // "-m" repeated accumulates; '-p ""' is an empty range -> default kept
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "a", "-m", "b", "-p", ""])).unwrap();
        assert_eq!(p.model, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(p.n_prompt, vec![512]);

        // errors (llama-bench.cpp:1060-1073)
        assert!(parse_cmd_params(&argv(&["llama-bench", "--nope"])).is_err());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-p", "abc"])).is_err());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-o", "xml"])).is_err());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-r"])).is_err());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-ngl"])).is_err());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-ngl", "-2"])).is_ok());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-dev", "none"])).is_ok());
        assert!(parse_cmd_params(&argv(&["llama-bench", "-dev", "CUDA0"])).is_err());
    }

    #[test]
    fn instance_matrix_order_and_expansion() {
        let p = parse_cmd_params(&argv(&[
            "llama-bench",
            "-m",
            "q.gguf",
            "-p",
            "64",
            "-n",
            "16",
            "-t",
            "8",
            "-fa",
            "on",
        ]))
        .unwrap();
        let inst = get_cmd_params_instances(&p);
        // one n_prompt and one n_gen instance per outer combination
        assert_eq!(inst.len(), 2);
        assert_eq!((inst[0].n_prompt, inst[0].n_gen), (64, 0));
        assert_eq!((inst[1].n_prompt, inst[1].n_gen), (0, 16));
        // n_ctx = n_prompt + n_gen + n_depth (llama-bench.cpp:1225-1231)
        assert_eq!(inst[0].n_ctx(), 64);
        assert_eq!(inst[1].n_ctx(), 16);
        assert_eq!(inst[0].flash_attn, FlashAttnType::Enabled);
        assert_eq!(inst[0].n_threads, 8);
        assert_eq!(inst[0].model, "q.gguf");

        // -pg adds a third instance; a matrix on -t multiplies the outer product
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "q.gguf", "-p", "64", "-n", "16", "-pg", "32,8", "-t", "4,8"])).unwrap();
        let inst = get_cmd_params_instances(&p);
        assert_eq!(inst.len(), 6);
        let shape: Vec<(&str, i32, i32, i32)> = inst
            .iter()
            .map(|i| if i.n_prompt > 0 && i.n_gen > 0 { ("pg", i.n_prompt, i.n_gen, i.n_threads) } else if i.n_prompt > 0 { ("pp", i.n_prompt, i.n_gen, i.n_threads) } else { ("tg", i.n_prompt, i.n_gen, i.n_threads) })
            .collect();
        assert_eq!(
            shape,
            vec![
                ("pp", 64, 0, 4),
                ("tg", 0, 16, 4),
                ("pg", 32, 8, 4),
                ("pp", 64, 0, 8),
                ("tg", 0, 16, 8),
                ("pg", 32, 8, 8),
            ]
        );

        // n_prompt == 0 / n_gen == 0 / (0,0) -pg are skipped (llama-bench.cpp:1363/1416/1447)
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "q.gguf", "-p", "0", "-n", "0", "-pg", "0,0"])).unwrap();
        assert!(get_cmd_params_instances(&p).is_empty());
    }
}