//! `test` + the output printers (llama-bench.cpp:1453-1985):
//! `get_fields`/`get_values`/`get_map`/`get_field_type` and the
//! csv / json / jsonl / markdown / sql printers, field order and escaping
//! included.

use crate::params::{CmdParams, CmdParamsInstance, Devices, FlashAttnType, LazyMode, LoadMode, SplitMode};
use crate::util;

/// `test::field_type` (llama-bench.cpp:1611-1635) — which of the four C
/// categories a field falls into (drives JSON quoting and the markdown widths).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldType {
    Str,
    Bool,
    Int,
    Float,
}

/// `test` (llama-bench.cpp:1453-1773)
pub struct Test {
    pub cpu_info: String,
    pub gpu_info: String,
    pub model_filename: String,
    pub model_type: String,
    pub model_size: u64,
    pub model_n_params: u64,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_threads: i32,
    pub cpu_mask: String,
    pub cpu_strict: bool,
    pub poll: i32,
    pub type_k: String,
    pub type_v: String,
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
    pub tensor_buft_overrides: Vec<(Option<String>, Option<String>)>,
    pub embeddings: bool,
    pub no_op_offload: bool,
    pub no_host: bool,
    pub repack: bool,
    pub fit_target: u64,
    pub fit_min_ctx: u32,
    pub n_prompt: i32,
    pub n_gen: i32,
    pub n_depth: i32,
    pub test_time: String,
    pub samples_ns: Vec<u64>,
    /// rust-only: tokens the engine really evaluated in each *timed* repeat
    /// (the warmup runs are not counted), for the token-count parity check
    pub evaluated_prompt: Vec<u64>,
    pub evaluated_gen: Vec<u64>,
    pub evaluated_depth: Vec<u64>,
}

impl Test {
    /// `test::test` (llama-bench.cpp:1530-1553) for the fields that come from
    /// the model (`llama_model_desc` / `llama_model_size` /
    /// `llama_model_n_params`) plus the RFC 3339 timestamp.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inst: &CmdParamsInstance,
        model_type: String,
        model_size: u64,
        model_n_params: u64,
    ) -> Self {
        Test {
            cpu_info: get_cpu_info(),
            gpu_info: get_gpu_info(),
            model_filename: inst.model.clone(),
            model_type,
            model_size,
            model_n_params,
            n_batch: inst.n_batch,
            n_ubatch: inst.n_ubatch,
            n_threads: inst.n_threads,
            cpu_mask: inst.cpu_mask.clone(),
            cpu_strict: inst.cpu_strict,
            poll: inst.poll,
            type_k: crate::params::ggml_type_name(inst.type_k).to_string(),
            type_v: crate::params::ggml_type_name(inst.type_v).to_string(),
            n_gpu_layers: inst.n_gpu_layers,
            n_cpu_moe: inst.n_cpu_moe,
            split_mode: inst.split_mode,
            load_mode: inst.load_mode,
            lazy_mode: inst.lazy_mode,
            main_gpu: inst.main_gpu,
            no_kv_offload: inst.no_kv_offload,
            flash_attn: inst.flash_attn,
            devices: inst.devices.clone(),
            tensor_split: inst.tensor_split.clone(),
            tensor_buft_overrides: inst.tensor_buft_overrides.clone(),
            embeddings: inst.embeddings,
            no_op_offload: inst.no_op_offload,
            no_host: inst.no_host,
            repack: inst.repack,
            fit_target: inst.fit_target,
            fit_min_ctx: inst.fit_min_ctx,
            n_prompt: inst.n_prompt,
            n_gen: inst.n_gen,
            n_depth: inst.n_depth,
            test_time: util::utc_time_rfc3339(),
            samples_ns: Vec::new(),
            evaluated_prompt: Vec::new(),
            evaluated_gen: Vec::new(),
            evaluated_depth: Vec::new(),
        }
    }

    /// `avg_ns` (llama-bench.cpp:1556)
    pub fn avg_ns(&self) -> u64 {
        util::avg_u64(&self.samples_ns)
    }

    /// `stdev_ns` (llama-bench.cpp:1558)
    pub fn stdev_ns(&self) -> u64 {
        util::stdev_ns_u64(&self.samples_ns)
    }

    /// `get_ts` (llama-bench.cpp:1560-1566)
    pub fn get_ts(&self) -> Vec<f64> {
        util::get_ts(&self.samples_ns, self.n_prompt + self.n_gen)
    }

    /// `avg_ts` (llama-bench.cpp:1568)
    pub fn avg_ts(&self) -> f64 {
        util::avg(&self.get_ts())
    }

    /// `stdev_ts` (llama-bench.cpp:1570)
    pub fn stdev_ts(&self) -> f64 {
        util::stdev_f64(&self.get_ts())
    }

    /// `test::get_backend` (llama-bench.cpp:1572-1590): all backends except
    /// CPU, with RPC last; this engine registers only the CPU backend.
    pub fn get_backend() -> &'static str {
        "CPU"
    }

    /// `test::get_fields` (llama-bench.cpp:1592-1609) — the exact order of the
    /// CSV header and of the JSON object keys.
    pub fn get_fields() -> Vec<&'static str> {
        vec![
            "build_commit", "build_number", "cpu_info", "gpu_info", "backends",
            "model_filename", "model_type", "model_size", "model_n_params", "n_batch",
            "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll",
            "type_k", "type_v", "n_gpu_layers", "n_cpu_moe", "split_mode",
            "main_gpu", "no_kv_offload", "flash_attn", "devices", "tensor_split",
            "tensor_buft_overrides", "load_mode", "lazy_mode",
            "embeddings",
            "no_op_offload", "no_host", "repack", "fit_target", "fit_min_ctx",
            "n_prompt", "n_gen", "n_depth",
            "test_time", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
        ]
    }

    /// `test::get_field_type` (llama-bench.cpp:1611-1635)
    pub fn get_field_type(field: &str) -> FieldType {
        if matches!(
            field,
            "build_number" | "n_batch" | "n_ubatch" | "n_threads" | "poll" | "model_size"
                | "model_n_params" | "n_gpu_layers" | "main_gpu" | "n_prompt" | "n_gen" | "n_depth"
                | "avg_ns" | "stddev_ns" | "no_op_offload" | "n_cpu_moe" | "fit_target" | "fit_min_ctx"
                | "flash_attn"
        ) {
            return FieldType::Int;
        }
        if matches!(field, "f16_kv" | "no_kv_offload" | "cpu_strict" | "embeddings" | "no_host" | "repack") {
            return FieldType::Bool;
        }
        if matches!(field, "avg_ts" | "stddev_ts") {
            return FieldType::Float;
        }
        // "load_mode" | "lazy_mode" | everything else (llama-bench.cpp:1630-1634)
        FieldType::Str
    }

    /// `test::get_values` (llama-bench.cpp:1637-1734) — same order as
    /// `get_fields`.
    pub fn get_values(&self) -> Vec<String> {
        let tensor_split_str = self.tensor_split_str();
        let tensor_buft_overrides_str = self.tensor_buft_overrides_str();
        vec![
            crate::BUILD_COMMIT.to_string(),
            crate::BUILD_NUMBER.to_string(),
            self.cpu_info.clone(),
            self.gpu_info.clone(),
            Self::get_backend().to_string(),
            self.model_filename.clone(),
            self.model_type.clone(),
            self.model_size.to_string(),
            self.model_n_params.to_string(),
            self.n_batch.to_string(),
            self.n_ubatch.to_string(),
            self.n_threads.to_string(),
            self.cpu_mask.clone(),
            (self.cpu_strict as i32).to_string(),
            self.poll.to_string(),
            self.type_k.clone(),
            self.type_v.clone(),
            self.n_gpu_layers.to_string(),
            self.n_cpu_moe.to_string(),
            self.split_mode.as_str().to_string(),
            self.main_gpu.to_string(),
            (self.no_kv_offload as i32).to_string(),
            self.flash_attn.as_i32().to_string(),
            self.devices.as_str(),
            tensor_split_str,
            tensor_buft_overrides_str,
            self.load_mode.as_str().to_string(),
            self.lazy_mode.as_str().to_string(),
            (self.embeddings as i32).to_string(),
            (self.no_op_offload as i32).to_string(),
            (self.no_host as i32).to_string(),
            (self.repack as i32).to_string(),
            self.fit_target.to_string(),
            self.fit_min_ctx.to_string(),
            self.n_prompt.to_string(),
            self.n_gen.to_string(),
            self.n_depth.to_string(),
            self.test_time.clone(),
            self.avg_ns().to_string(),
            self.stdev_ns().to_string(),
            util::fmt_to_string_f64(self.avg_ts()),
            util::fmt_to_string_f64(self.stdev_ts()),
        ]
    }

    /// llama-bench.cpp:1638-1659: `%.2f` per non-zero device, joined by '/'
    fn tensor_split_str(&self) -> String {
        let mut max_nonzero = 0;
        for (i, v) in self.tensor_split.iter().enumerate() {
            if *v > 0.0 {
                max_nonzero = i;
            }
        }
        let mut s = String::new();
        for i in 0..=max_nonzero {
            s.push_str(&format!("{:.2}", self.tensor_split.get(i).copied().unwrap_or(0.0)));
            if i < max_nonzero {
                s.push('/');
            }
        }
        s
    }

    /// llama-bench.cpp:1660-1678
    fn tensor_buft_overrides_str(&self) -> String {
        let v = &self.tensor_buft_overrides;
        let mut s = String::new();
        if v.len() == 1 {
            // the only element must be the null pattern sentinel
            s.push_str("none");
        } else {
            for i in 0..v.len().saturating_sub(1) {
                match &v[i].0 {
                    None => s.push_str("none"),
                    Some(pattern) => {
                        s.push_str(pattern);
                        s.push('=');
                        s.push_str(v[i].1.as_deref().unwrap_or(""));
                    }
                }
                if i + 2 < v.len() {
                    s.push(';');
                }
            }
        }
        s
    }

    /// `test::get_map` (llama-bench.cpp:1736-1743)
    pub fn get_map(&self) -> Vec<(&'static str, String)> {
        Self::get_fields().into_iter().zip(self.get_values()).collect()
    }

    pub fn map_get<'a>(map: &'a [(&'static str, String)], field: &str) -> &'a str {
        map.iter().find(|(k, _)| *k == field).map(|(_, v)| v.as_str()).unwrap_or("")
    }
}

/// `get_cpu_info` (llama-bench.cpp:120-131): the CPU + ACCEL device
/// descriptions joined with ", " — one device in this engine.
pub fn get_cpu_info() -> String {
    util::cpu_info()
}

/// `get_gpu_info` (llama-bench.cpp:133-144): GPU/IGPU devices; none here.
pub fn get_gpu_info() -> String {
    String::new()
}

// ---------------------------------------------------------------------------
// printers (llama-bench.cpp:1745-1985)
// ---------------------------------------------------------------------------

/// `csv_printer::escape_csv` (llama-bench.cpp:1754-1763)
pub fn escape_csv(field: &str) -> String {
    let mut escaped = String::from("\"");
    for c in field.chars() {
        if c == '"' {
            escaped.push('"');
        }
        escaped.push(c);
    }
    escaped.push('"');
    escaped
}

/// `escape_json` (llama-bench.cpp:1783-1798)
pub fn escape_json(value: &str) -> String {
    let mut escaped = String::new();
    for c in value.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            c if (c as u32) <= 0x1f => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

/// `format_json_value` (llama-bench.cpp:1800-1810)
pub fn format_json_value(field: &str, value: &str) -> String {
    match Test::get_field_type(field) {
        FieldType::Str => format!("\"{}\"", escape_json(value)),
        FieldType::Bool => if value == "0" { "false" } else { "true" }.to_string(),
        _ => value.to_string(),
    }
}

/// `join` (llama-bench.cpp:1960-1971)
pub fn join<T: std::fmt::Display>(values: &[T], delim: &str) -> String {
    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(delim)
}

pub enum Printer {
    Csv,
    Json { first: bool },
    Jsonl,
    Markdown { fields: Vec<String> },
    Sql,
}

impl Printer {
    /// `create_printer` (llama-bench.cpp:2205-2221)
    pub fn create(format: crate::params::OutputFormat) -> Option<Printer> {
        use crate::params::OutputFormat as F;
        Some(match format {
            F::None => return None,
            F::Csv => Printer::Csv,
            F::Json => Printer::Json { first: true },
            F::Jsonl => Printer::Jsonl,
            F::Markdown => Printer::Markdown { fields: Vec::new() },
            F::Sql => Printer::Sql,
        })
    }

    /// `printer::print_header` (llama-bench.cpp:1770 / 1812-1816 / 1853 /
    /// 1875-1980 / 2043-2056)
    pub fn print_header(&mut self, params: &CmdParams, out: &mut String) {
        match self {
            Printer::Csv => {
                out.push_str(&join(&Test::get_fields(), ","));
                out.push('\n');
            }
            Printer::Json { .. } => out.push_str("[\n"),
            Printer::Jsonl => {}
            Printer::Markdown { fields } => {
                // select fields to print (llama-bench.cpp:1925-1980)
                fields.clear();
                fields.push("model".to_string());
                fields.push("size".to_string());
                fields.push("params".to_string());
                fields.push("backend".to_string());
                let backend = Test::get_backend();
                let is_cpu_backend =
                    backend.contains("CPU") || backend.contains("BLAS") || backend.contains("ZenDNN");
                let d = CmdParams::default();
                if !is_cpu_backend {
                    fields.push("n_gpu_layers".to_string());
                }
                if params.n_cpu_moe.len() > 1 || params.n_cpu_moe != d.n_cpu_moe {
                    fields.push("n_cpu_moe".to_string());
                }
                if params.n_threads.len() > 1 || params.n_threads != d.n_threads || is_cpu_backend {
                    fields.push("n_threads".to_string());
                }
                if params.cpu_mask.len() > 1 || params.cpu_mask != d.cpu_mask {
                    fields.push("cpu_mask".to_string());
                }
                if params.cpu_strict.len() > 1 || params.cpu_strict != d.cpu_strict {
                    fields.push("cpu_strict".to_string());
                }
                if params.poll.len() > 1 || params.poll != d.poll {
                    fields.push("poll".to_string());
                }
                if params.n_batch.len() > 1 || params.n_batch != d.n_batch {
                    fields.push("n_batch".to_string());
                }
                if params.n_ubatch.len() > 1 || params.n_ubatch != d.n_ubatch {
                    fields.push("n_ubatch".to_string());
                }
                if params.type_k.len() > 1 || params.type_k != d.type_k {
                    fields.push("type_k".to_string());
                }
                if params.type_v.len() > 1 || params.type_v != d.type_v {
                    fields.push("type_v".to_string());
                }
                if params.main_gpu.len() > 1 || params.main_gpu != d.main_gpu {
                    fields.push("main_gpu".to_string());
                }
                if params.split_mode.len() > 1 || params.split_mode != d.split_mode {
                    fields.push("split_mode".to_string());
                }
                if params.no_kv_offload.len() > 1 || params.no_kv_offload != d.no_kv_offload {
                    fields.push("no_kv_offload".to_string());
                }
                if params.flash_attn.len() > 1 || params.flash_attn != d.flash_attn {
                    fields.push("flash_attn".to_string());
                }
                if params.devices.len() > 1 || params.devices != d.devices {
                    fields.push("devices".to_string());
                }
                if params.tensor_split.len() > 1 || params.tensor_split != d.tensor_split {
                    fields.push("tensor_split".to_string());
                }
                if params.tensor_buft_overrides.len() > 1 || params.tensor_buft_overrides != d.tensor_buft_overrides {
                    fields.push("tensor_buft_overrides".to_string());
                }
                if params.load_mode.len() > 1 || params.load_mode != d.load_mode {
                    fields.push("load_mode".to_string());
                }
                if params.lazy_mode.len() > 1 || params.lazy_mode != d.lazy_mode {
                    fields.push("lazy_mode".to_string());
                }
                if params.embeddings.len() > 1 || params.embeddings != d.embeddings {
                    fields.push("embeddings".to_string());
                }
                if params.no_op_offload.len() > 1 || params.no_op_offload != d.no_op_offload {
                    fields.push("no_op_offload".to_string());
                }
                if params.no_host.len() > 1 || params.no_host != d.no_host {
                    fields.push("no_host".to_string());
                }
                if params.repack.len() > 1 || params.repack != d.repack {
                    fields.push("repack".to_string());
                }
                if params.fit_params_target.len() > 1 || params.fit_params_target != d.fit_params_target {
                    fields.push("fit_target".to_string());
                }
                if params.fit_params_min_ctx.len() > 1 || params.fit_params_min_ctx != d.fit_params_min_ctx {
                    fields.push("fit_min_ctx".to_string());
                }
                fields.push("test".to_string());
                fields.push("t/s".to_string());

                out.push('|');
                for field in fields.iter() {
                    out.push_str(&format!(" {} |", pad(get_field_display_name(field), get_field_width(field))));
                }
                out.push('\n');
                out.push('|');
                for field in fields.iter() {
                    let width = get_field_width(field);
                    let dashes = "-".repeat((width.abs() - 1) as usize);
                    out.push_str(&format!(" {dashes}{} |", if width > 0 { ":" } else { "-" }));
                }
                out.push('\n');
            }
            Printer::Sql => {
                out.push_str("CREATE TABLE IF NOT EXISTS llama_bench (\n");
                let fields = Test::get_fields();
                for (i, field) in fields.iter().enumerate() {
                    out.push_str(&format!(
                        "  {} {}{}\n",
                        field,
                        get_sql_field_type(field),
                        if i < fields.len() - 1 { "," } else { "" }
                    ));
                }
                out.push_str(");\n");
                out.push('\n');
            }
        }
    }

    /// `printer::print_test` (llama-bench.cpp:1766-1770 / 1818-1840 /
    /// 1848-1868 / 1982-2039 / 2058-2066)
    pub fn print_test(&mut self, t: &Test, out: &mut String) {
        match self {
            Printer::Csv => {
                let values: Vec<String> = t.get_values().iter().map(|v| escape_csv(v)).collect();
                out.push_str(&join(&values, ","));
                out.push('\n');
            }
            Printer::Json { first } => {
                if *first {
                    *first = false;
                } else {
                    out.push_str(",\n");
                }
                out.push_str("  {\n");
                let fields = Test::get_fields();
                let values = t.get_values();
                for (f, v) in fields.iter().zip(values.iter()) {
                    out.push_str(&format!("    \"{}\": {},\n", f, format_json_value(f, v)));
                }
                out.push_str(&format!("    \"samples_ns\": [ {} ],\n", join(&t.samples_ns, ", ")));
                out.push_str(&format!("    \"samples_ts\": [ {} ]\n", join(&ts_g6(&t.get_ts()), ", ")));
                out.push_str("  }");
            }
            Printer::Jsonl => {
                out.push('{');
                let fields = Test::get_fields();
                let values = t.get_values();
                for (f, v) in fields.iter().zip(values.iter()) {
                    out.push_str(&format!("\"{}\": {}, ", f, format_json_value(f, v)));
                }
                out.push_str(&format!("\"samples_ns\": [ {} ],", join(&t.samples_ns, ", ")));
                out.push_str(&format!("\"samples_ts\": [ {} ]", join(&ts_g6(&t.get_ts()), ", ")));
                out.push_str("}\n");
            }
            Printer::Markdown { fields } => {
                let vmap = t.get_map();
                out.push('|');
                for field in fields.iter() {
                    let value: String = match field.as_str() {
                        "model" => t.model_type.clone(),
                        "size" => {
                            if t.model_size < 1024 * 1024 * 1024 {
                                format!("{:.2} MiB", t.model_size as f64 / 1024.0 / 1024.0)
                            } else {
                                format!("{:.2} GiB", t.model_size as f64 / 1024.0 / 1024.0 / 1024.0)
                            }
                        }
                        "params" => {
                            if t.model_n_params < 1000 * 1000 * 1000 {
                                format!("{:.2} M", t.model_n_params as f64 / 1e6)
                            } else {
                                format!("{:.2} B", t.model_n_params as f64 / 1e9)
                            }
                        }
                        "backend" => Test::get_backend().to_string(),
                        "test" => test_name(t),
                        "t/s" => format!("{:.2} ± {:.2}", t.avg_ts(), t.stdev_ts()),
                        other => Test::map_get(&vmap, other).to_string(),
                    };
                    let mut width = get_field_width(field);
                    if field == "t/s" {
                        // HACK: the utf-8 character is 2 bytes (llama-bench.cpp:2033-2036)
                        width += 1;
                    }
                    out.push_str(&format!(" {} |", pad(&value, width)));
                }
                out.push('\n');
            }
            Printer::Sql => {
                out.push_str(&format!("INSERT INTO llama_bench ({}) ", join(&Test::get_fields(), ", ")));
                out.push_str("VALUES (");
                let values = t.get_values();
                for (i, v) in values.iter().enumerate() {
                    out.push_str(&format!("'{}'{}", v, if i < values.len() - 1 { ", " } else { "" }));
                }
                out.push_str(");\n");
            }
        }
    }

    /// `printer::print_footer` (llama-bench.cpp:1843 / 2036-2038)
    pub fn print_footer(&mut self, out: &mut String) {
        match self {
            Printer::Json { .. } => out.push_str("\n]\n"),
            Printer::Markdown { .. } => {
                out.push_str(&format!("\nbuild: {} ({})\n", crate::BUILD_COMMIT, crate::BUILD_NUMBER));
            }
            _ => {}
        }
    }
}

/// the C prints `samples_ts` through `ostringstream <<` (6 significant digits)
fn ts_g6(ts: &[f64]) -> Vec<String> {
    ts.iter().map(|v| util::fmt_g6(*v)).collect()
}

/// `printf("%*s", width, s)` — negative width left-justifies and pads to
/// `abs(width)`, and the count is in *bytes* (the markdown printer relies on
/// that for the two-byte `±`, llama-bench.cpp:2033-2036).
fn pad(s: &str, width: i32) -> String {
    let n = s.len();
    if width < 0 {
        let w = (-width) as usize;
        if n >= w { s.to_string() } else { format!("{s}{}", " ".repeat(w - n)) }
    } else {
        let w = width as usize;
        if n >= w { s.to_string() } else { format!("{}{s}", " ".repeat(w - n)) }
    }
}

/// the markdown `test` column (llama-bench.cpp:2001-2016): `pp<n>`, `tg<n>` or
/// `pp<n>+tg<m>`, with ` @ d<depth>` appended when the depth is non-zero
pub fn test_name(t: &Test) -> String {
    let mut v = if t.n_prompt > 0 && t.n_gen == 0 {
        format!("pp{}", t.n_prompt)
    } else if t.n_gen > 0 && t.n_prompt == 0 {
        format!("tg{}", t.n_gen)
    } else {
        format!("pp{}+tg{}", t.n_prompt, t.n_gen)
    };
    if t.n_depth > 0 {
        v.push_str(&format!(" @ d{}", t.n_depth));
    }
    v
}

/// `markdown_printer::get_field_width` (llama-bench.cpp:1878-1922)
pub fn get_field_width(field: &str) -> i32 {
    match field {
        "model" => return -30,
        "t/s" => return 20,
        "size" | "params" => return 10,
        "n_gpu_layers" => return 3,
        "n_threads" => return 7,
        "n_batch" => return 7,
        "n_ubatch" => return 8,
        "type_k" | "type_v" => return 6,
        "split_mode" => return 6,
        "load_mode" => return 10,
        "flash_attn" => return 3,
        "devices" => return -12,
        "test" => return 15,
        "no_op_offload" => return 4,
        "no_host" => return 4,
        "repack" => return 3,
        _ => {}
    }
    let width = std::cmp::max(field.len() as i32, 10);
    if Test::get_field_type(field) == FieldType::Str {
        -width
    } else {
        width
    }
}

/// `markdown_printer::get_field_display_name` (llama-bench.cpp:1924-1957)
pub fn get_field_display_name(field: &str) -> &str {
    match field {
        "n_gpu_layers" => "ngl",
        "split_mode" => "sm",
        "n_threads" => "threads",
        "no_kv_offload" => "nkvo",
        "flash_attn" => "fa",
        "load_mode" => "lm",
        "embeddings" => "embd",
        "no_op_offload" => "nopo",
        "no_host" => "noh",
        "repack" => "rpk",
        "devices" => "dev",
        "tensor_split" => "ts",
        "tensor_buft_overrides" => "ot",
        "fit_target" => "fitt",
        "fit_min_ctx" => "fitc",
        other => other,
    }
}

/// `sql_printer::get_sql_field_type` (llama-bench.cpp:2043-2056)
pub fn get_sql_field_type(field: &str) -> &'static str {
    match Test::get_field_type(field) {
        FieldType::Str => "TEXT",
        FieldType::Bool | FieldType::Int => "INTEGER",
        FieldType::Float => "REAL",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{parse_cmd_params, OutputFormat};

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// fixed test record: the samples from the reference JSON capture of
    /// `-p 64 -n 16 -t 8 -r 2 -fa off` (avg_ns 98978370 etc.) so the expected
    /// strings below are the reference's own numbers.
    fn fixed_test() -> Test {
        let p = parse_cmd_params(&argv(&[
            "llama-bench", "-m", "/tmp/q.gguf", "-p", "64", "-n", "16", "-t", "8", "-r", "2", "-fa", "off",
        ]))
        .unwrap();
        let inst = crate::params::get_cmd_params_instances(&p).remove(0);
        let mut t = Test::new(&inst, "qwen2 1B Q4_K - Medium".to_string(), 485452288, 630167424);
        t.test_time = "2026-09-24T22:56:39Z".to_string();
        t.samples_ns = vec![99503886, 98452854];
        t.cpu_info = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S".to_string();
        t
    }

    #[test]
    fn fields_and_types_match_the_c_tables() {
        // the 42 fields of llama-bench.cpp:1594-1610 (repack added by
        // #28968), in order
        let fields = Test::get_fields();
        assert_eq!(fields.len(), 42);
        assert_eq!(fields[0], "build_commit");
        assert_eq!(fields[22], "flash_attn");
        assert_eq!(fields[31], "repack");
        assert_eq!(fields[37], "test_time");
        assert_eq!(fields[41], "stddev_ts");
        // the C's (buggy but load-bearing) type table: flash_attn is INT and
        // printed unquoted, no_kv_offload/embeddings are BOOL
        assert_eq!(Test::get_field_type("flash_attn"), FieldType::Int);
        assert_eq!(Test::get_field_type("no_kv_offload"), FieldType::Bool);
        assert_eq!(Test::get_field_type("embeddings"), FieldType::Bool);
        assert_eq!(Test::get_field_type("avg_ts"), FieldType::Float);
        assert_eq!(Test::get_field_type("model_type"), FieldType::Str);
        assert_eq!(format_json_value("flash_attn", "-1"), "-1");
        assert_eq!(format_json_value("no_kv_offload", "0"), "false");
        assert_eq!(format_json_value("no_kv_offload", "1"), "true");
        assert_eq!(format_json_value("model_type", "qwen2 1B"), "\"qwen2 1B\"");
    }

    #[test]
    fn values_follow_the_c_aggregation() {
        let t = fixed_test();
        let v = t.get_values();
        assert_eq!(v.len(), 42);
        assert_eq!(v[0], crate::BUILD_COMMIT);
        assert_eq!(v[1], crate::BUILD_NUMBER.to_string());
        assert_eq!(v[5], "/tmp/q.gguf");
        assert_eq!(v[6], "qwen2 1B Q4_K - Medium");
        assert_eq!(v[7], "485452288");
        assert_eq!(v[8], "630167424");
        assert_eq!(v[9], "2048"); // n_batch
        assert_eq!(v[10], "512"); // n_ubatch
        assert_eq!(v[11], "8"); // n_threads
        assert_eq!(v[12], "0x0"); // cpu_mask
        assert_eq!(v[13], "0"); // cpu_strict
        assert_eq!(v[14], "50"); // poll
        assert_eq!(v[15], "f16");
        assert_eq!(v[16], "f16");
        assert_eq!(v[17], "-1"); // n_gpu_layers
        assert_eq!(v[19], "layer"); // split_mode
        assert_eq!(v[20], "0"); // main_gpu
        assert_eq!(v[22], "0"); // flash_attn = disabled
        assert_eq!(v[23], "auto"); // devices
        assert_eq!(v[24], "0.00"); // tensor_split
        assert_eq!(v[25], "none"); // tensor_buft_overrides
        assert_eq!(v[26], "auto"); // load_mode
        assert_eq!(v[27], "auto"); // lazy_mode
        assert_eq!(v[31], "1"); // repack = use_extra_bufts default
        assert_eq!(v[34], "64"); // n_prompt
        assert_eq!(v[35], "0"); // n_gen
        assert_eq!(v[37], "2026-09-24T22:56:39Z");
        // avg_ns = (99503886 + 98452854) / 2 ; avg_ts = mean(1e9*64/t)
        assert_eq!(t.avg_ns(), 98978370);
        assert_eq!(v[38], "98978370");
        let ts = t.get_ts();
        assert!((ts[0] - 1e9 * 64.0 / 99503886.0).abs() < 1e-9);
        assert!((t.avg_ts() - (ts[0] + ts[1]) / 2.0).abs() < 1e-9);
        // std::to_string(double) formatting of the two t/s fields
        assert_eq!(v[40], util::fmt_to_string_f64(t.avg_ts()));
        assert_eq!(v[41], util::fmt_to_string_f64(t.stdev_ts()));
        // the tg instance differs only in n_prompt/n_gen/test name
        assert_eq!(Test::map_get(&t.get_map(), "n_depth"), "0");
    }

    #[test]
    fn markdown_header_column_selection() {
        // plain CPU run: model/size/params/backend/threads/test/t-s
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "q.gguf", "-p", "64", "-n", "16", "-t", "8"])).unwrap();
        let mut pr = Printer::create(OutputFormat::Markdown).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        let expected = "\
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
";
        assert_eq!(out, expected);

        // -fa on adds the `fa` column after `threads`
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "q.gguf", "-p", "64", "-n", "16", "-t", "8", "-fa", "on"])).unwrap();
        let mut pr = Printer::create(OutputFormat::Markdown).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        assert!(out.contains("| threads |  fa |"), "{out}");
        assert!(out.contains("| ------: | --: |"), "{out}");

        // -ctk/-ctv and -b/-ub add type_k/type_v/n_batch/n_ubatch in the C order
        let p = parse_cmd_params(&argv(&[
            "llama-bench", "-m", "q.gguf", "-p", "16", "-b", "128", "-ub", "64", "-ctk", "q8_0", "-ctv", "q8_0",
        ]))
        .unwrap();
        let mut pr = Printer::create(OutputFormat::Markdown).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        assert!(out.contains("| n_batch | n_ubatch | type_k | type_v |"), "{out}");
    }

    #[test]
    fn markdown_row_and_footer_text() {
        let t = fixed_test();
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "/tmp/q.gguf", "-p", "64", "-n", "16", "-t", "8", "-r", "2", "-fa", "off"])).unwrap();
        let mut pr = Printer::create(OutputFormat::Markdown).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        out.clear();
        pr.print_test(&t, &mut out);
        // 462.96 MiB / 630.17 M / CPU / 8 threads / fa=0 / pp64
        assert_eq!(
            out,
            "| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CPU        |       8 |   0 |            pp64 |        646.62 ± 4.86 |\n"
        );
        let mut out = String::new();
        pr.print_footer(&mut out);
        assert_eq!(out, "\nbuild: def4d406a (11325)\n");

        // tg row: `tg16` and n_prompt=0 (test name from n_gen only)
        let mut t2 = fixed_test();
        t2.n_prompt = 0;
        t2.n_gen = 16;
        t2.samples_ns = vec![109117216, 112192781];
        let mut out = String::new();
        pr.print_test(&t2, &mut out);
        assert!(out.contains("|            tg16 |"), "{out}");
        assert!(out.contains("|   0 |            tg16"), "{out}");

        // -d adds " @ dN" to the test column
        let mut t3 = fixed_test();
        t3.n_depth = 4096;
        let mut out = String::new();
        pr.print_test(&t3, &mut out);
        assert!(out.contains("|    pp64 @ d4096 |"), "{out}");
    }

    #[test]
    fn csv_header_and_row_text() {
        let t = fixed_test();
        let mut pr = Printer::create(OutputFormat::Csv).unwrap();
        let mut out = String::new();
        pr.print_header(&parse_cmd_params(&argv(&["llama-bench"])).unwrap(), &mut out);
        assert_eq!(
            out,
            "build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,load_mode,lazy_mode,embeddings,no_op_offload,no_host,repack,fit_target,fit_min_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n"
        );
        out.clear();
        pr.print_test(&t, &mut out);
        let fields: Vec<&str> = out.trim_end().split(',').collect();
        assert_eq!(fields.len(), 42);
        assert_eq!(fields[0], "\"def4d406a\"");
        assert_eq!(fields[1], "\"11325\"");
        assert_eq!(fields[2], "\"AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\"");
        assert_eq!(fields[6], "\"qwen2 1B Q4_K - Medium\"");
        assert_eq!(fields[22], "\"0\"");
        assert_eq!(fields[37], "\"2026-09-24T22:56:39Z\"");
        // every field is quoted and inner quotes are doubled
        assert_eq!(escape_csv("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn json_and_jsonl_text() {
        let t = fixed_test();
        let p = parse_cmd_params(&argv(&["llama-bench", "-m", "/tmp/q.gguf", "-p", "64", "-t", "8", "-r", "2", "-fa", "off"])).unwrap();
        let mut pr = Printer::create(OutputFormat::Json).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        out.clear();
        pr.print_test(&t, &mut out);
        // note: a Rust `\`-continued string literal strips the next line's leading
        // whitespace, so the JSON indentation is spelled out here
        let expected_prefix = [
            "  {",
            "    \"build_commit\": \"def4d406a\",",
            "    \"build_number\": 11325,",
            "    \"cpu_info\": \"AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\",",
            "    \"gpu_info\": \"\",",
            "    \"backends\": \"CPU\",",
            "",
        ]
        .join("\n");
        assert!(out.starts_with(&expected_prefix), "{out}");
        // booleans unquoted, strings quoted, floats bare, flash_attn an int
        assert!(out.contains("\"cpu_strict\": false,\n"), "{out}");
        assert!(out.contains("\"no_kv_offload\": false,\n"), "{out}");
        assert!(out.contains("\"flash_attn\": 0,\n"), "{out}");
        assert!(out.contains("\"avg_ts\": 646.624148,\n"), "{out}");
        assert!(out.contains("\"samples_ns\": [ 99503886, 98452854 ],\n"), "{out}");
        // samples_ts via ostringstream (6 significant digits)
        assert!(out.contains("\"samples_ts\": [ 643.191, 650.057 ]\n"), "{out}");
        assert!(out.ends_with("  }"), "{out}");
        let mut foot = String::new();
        pr.print_footer(&mut foot);
        assert_eq!(foot, "\n]\n");

        // jsonl: one line, every field followed by ", ", then the samples
        let mut pr = Printer::create(OutputFormat::Jsonl).unwrap();
        let mut out = String::new();
        pr.print_header(&p, &mut out);
        assert!(out.is_empty());
        pr.print_test(&t, &mut out);
        assert!(out.starts_with("{\"build_commit\": \"def4d406a\", \"build_number\": 11325, "), "{out}");
        assert!(out.ends_with("\"samples_ts\": [ 643.191, 650.057 ]}\n"), "{out}");
        assert_eq!(out.matches('\n').count(), 1);
        // escaping rules (llama-bench.cpp:1783-1798)
        assert_eq!(escape_json("a\"b\\c"), "a\\\"b\\\\c");
        assert_eq!(escape_json("\u{1}"), "\\u0001");
    }

    #[test]
    fn sql_text() {
        let t = fixed_test();
        let mut pr = Printer::create(OutputFormat::Sql).unwrap();
        let mut out = String::new();
        pr.print_header(&parse_cmd_params(&argv(&["llama-bench"])).unwrap(), &mut out);
        assert!(out.starts_with("CREATE TABLE IF NOT EXISTS llama_bench (\n  build_commit TEXT,\n  build_number INTEGER,\n"), "{out}");
        assert!(out.ends_with("  stddev_ts REAL\n);\n\n"), "{out}");
        out.clear();
        pr.print_test(&t, &mut out);
        assert!(out.starts_with("INSERT INTO llama_bench (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, n_cpu_moe, split_mode, main_gpu, no_kv_offload, flash_attn, devices, tensor_split, tensor_buft_overrides, load_mode, lazy_mode, embeddings, no_op_offload, no_host, repack, fit_target, fit_min_ctx, n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts) VALUES ('def4d406a', '11325', "), "{out}");
        assert!(out.ends_with("'646.624148', '4.855261');\n"), "{out}");
    }

    #[test]
    fn aggregate_rounding_matches_hand_computation() {
        // 3 samples -> integer avg_ns truncates, stddev_ns uses the C's u64 path
        let mut t = fixed_test();
        t.samples_ns = vec![100, 200, 300];
        assert_eq!(t.avg_ns(), 200);
        assert_eq!(t.stdev_ns(), 100);
        let ts = t.get_ts();
        assert!((ts[0] - 1e9 * 64.0 / 100.0).abs() < 1e-9);
        assert!((t.avg_ts() - util::avg(&ts)).abs() < 1e-9);
        assert!((t.stdev_ts() - util::stdev_f64(&ts)).abs() < 1e-9);
    }
}