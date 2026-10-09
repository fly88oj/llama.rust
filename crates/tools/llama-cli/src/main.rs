//! llama-cli (rust) — port of tools/cli/main.cpp for the CPU path.
//!
//! Arch dispatch covers every builder in `graph_arch.rs` (qwen2/llama/qwen3/
//! gemma2/gemma3/gemma4/phi3/granite-hybrid/lfm2moe/qwen35/gpt-oss); an arch
//! with a loader but no builder exits with an error naming it (the reference
//! aborts inside `llama_model::create_memory`/graph build).
//!
//! Non-flash-attn graph (verify against anchor: reference `llama-cli -fa off`); sampling via
//! `llama::sampling::SamplingContext` (common_sampler replacement) with the
//! reference default chain, EOG stopping (`llama_vocab_is_eog`) and a minimal
//! single-turn chat mode (GGUF `tokenizer.chat_template` -> llama::chat).

use std::io::Write as _;
use std::sync::Arc;

mod diffusion;
mod interactive;

use llama::chat::{self, ChatMessage, ChatTemplate, Role};
use llama::context::DecodeContext;
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::graph_arch;
use llama::hparams::LlamaHparams;
use llama::model::{load_model, LlamaModel};
use llama::sampling::{CommonSamplerType, GrammarSampler, SamplingContext, SamplingParams};
use llama::speculative::{
    common_speculative_init, common_speculative_n_max_params, common_speculative_types_from_names,
    speculative_simple_generate, CommonParamsSpeculative,
};

/// `-fa / --flash-attn [on|off|auto]` (reference common/arg.cpp:1751-1765).
/// `auto` mirrors `LLAMA_FLASH_ATTN_TYPE_AUTO`, whose CPU resolution enables
/// FA (llama-context.cpp:230-231 + probe :34-38) — but per the wiring task the
/// `auto` mode is treated as `off` here until that probe exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlashAttn {
    Off,
    On,
    /// parsed but currently == Off (task: "auto behaves as off for now")
    Auto,
}

impl FlashAttn {
    fn on(self) -> bool {
        matches!(self, FlashAttn::On)
    }
}

/// CLI options (defaults mirror the previous llama-cli plus the reference
/// `common_params_sampling` defaults for the temperature chain).
/// `enum llama_load_mode` (llama.h) — the `-lm/--load-mode` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    Dio,
}

/// The weights mapping the model's tensors live in (the port's constant-mmap
/// load: every external storage slot holds the same `Arc<Mmap>`).
fn model_mapping(model: &LlamaModel) -> Option<&memmap2::Mmap> {
    model
        .ctx
        .external
        .iter()
        .find_map(|e| e.downcast_ref::<memmap2::Mmap>())
}

fn model_external_ptr(model: &LlamaModel) -> *const u8 {
    model_mapping(model).expect("mmap-backed model").as_ptr()
}

fn model_external_len(model: &LlamaModel) -> usize {
    model_mapping(model).expect("mmap-backed model").len()
}

struct Args {
    model_path: String,
    prompt: String,
    n_predict: usize,
    n_threads: usize,
    n_ctx: u32,
    /// <= 0 => greedy (llama_sampler_temp_impl)
    temp: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: i32,
    /// `--samplers "a;b;c"` / `--sampler-seq|--sampling-seq "abc"`
    /// (`params.sampling.samplers`, common/arg.cpp:1982-2003)
    samplers: Vec<CommonSamplerType>,
    /// `--dry-multiplier` (arg.cpp:2123, 0.0 = disabled)
    dry_multiplier: f32,
    /// `--dry-base` (arg.cpp:2130, values < 1.0 ignored)
    dry_base: f32,
    /// `--dry-allowed-length` (arg.cpp:2141)
    dry_allowed_length: i32,
    /// `--dry-penalty-last-n` (arg.cpp:2148, 0 = disabled, < 0 errors)
    dry_penalty_last_n: i32,
    /// `--dry-sequence-breaker` (arg.cpp:2158 — the first use clears the
    /// `["\n", ":", "\"", "*"]` default, "none" clears outright)
    dry_sequence_breakers: Vec<String>,
    /// `--adaptive-target` (arg.cpp:2184, negative = disabled)
    adaptive_target: f32,
    /// `--adaptive-decay` (arg.cpp:2194, 0.0 - 0.99)
    adaptive_decay: f32,
    /// u32; `-1` (or 0xFFFFFFFF) => LLAMA_DEFAULT_SEED (random)
    seed: u32,
    ignore_eos: bool,
    /// single-turn chat mode (reference `-cnv`); formats `-p` as a user message
    chat: bool,
    /// `--chat-template NAME|<template source>` (None => GGUF chat_template)
    chat_template: Option<String>,
    perplexity: Option<String>,
    /// `--grammar GBNF` / `--grammar-file FILE` / `-j, --json-schema SCHEMA` /
    /// `-jf, --json-schema-file FILE` (reference common/arg.cpp:2265-2300): the
    /// GBNF text constraining generation (`llama_sampler_init_grammar`). All
    /// four flags assign the *same* `common_grammar` field, so the last one on
    /// the command line wins.
    grammar: Option<String>,
    /// which flag supplied `grammar` — only for the startup log; the reference
    /// keeps `common_grammar::type` for the generation-prompt prefill, which is
    /// unreachable from llama-cli (`params.generation_prompt` stays empty:
    /// only the server/chat-auto-parser sets it, common/sampling.cpp:290-297).
    grammar_source: Option<&'static str>,
    /// `-fa / --flash-attn`; default off keeps the verified non-FA anchor
    flash_attn: FlashAttn,
    /// `-lm/--load-mode MODE` (common/arg.cpp's load-mode flag): `mlock` /
    /// `mmap+mlock` pin the weights mapping in RAM (llama_mlock); `auto` /
    /// `none` / `mmap` are the port's constant-mmap load; `dio` errors like
    /// the reference on a non-supporting platform.
    load_mode: LoadMode,
    /// `--lora FNAME` / `--lora-scaled FNAME:SCALE` entries, in command-line
    /// order (reference `common_params::lora_adapters`, common/common.h:531;
    /// both flags append, `--lora` with scale 1.0 — common/arg.cpp:2950-2974)
    lora_adapters: Vec<(String, f32)>,
    /// `params.speculative` (common.h:370-401) — `-md`/`--spec-*` surface of
    /// common/arg.cpp:4135-4253. Defaults to `types = {none}` (common.h:371),
    /// so a bare `-md` is a no-op exactly like the reference.
    speculative: CommonParamsSpeculative,
    /// `-fe/--embedding/--embeddings` (arg.cpp:3473-3476) — the port's
    /// encoder-embedding mode: BERT/EUROBERT/T5ENCODER run through
    /// `EncoderContext` (the `llama_encode` path, like llama-server's -fe).
    /// The pinned reference's own cli rejects the flag outright (arg.cpp:824
    /// — the old tools/main `-fe` print is gone from this revision), so this
    /// arm is the port-side wiring the audit's item 1 asked for; verified
    /// against the reference `llama_encode` dumps (parity/encode_*.bin).
    embedding: bool,
    /// `--pooling {none,mean,cls,last}` (arg.cpp:2310)
    pooling: Option<String>,
    /// `--embd-ids "0,581,...` — override the encoded token ids (the dumper
    /// flag of parity/gen_encode_ref.sh, port-side sugar so the CLI cells can
    /// hit the .bin anchors exactly)
    embd_ids: Option<String>,
    // ---- the interactive surface (tools/completion, common/arg.cpp
    // :1767-1963 + :3816-3824; defaults per common.h:461-590) ----
    /// `-i/--interactive` (arg.cpp:1919; common.h:560 default false)
    interactive: bool,
    /// `-if/--interactive-first` (arg.cpp:1926; common.h:561)
    interactive_first: bool,
    /// `--in-prefix STRING` (arg.cpp:1948) — also clears enable_chat_template
    input_prefix: String,
    /// `--in-prefix-bos` (arg.cpp:1940) — same template clearing side effect
    input_prefix_bos: bool,
    /// `--in-suffix STRING` (arg.cpp:1956) — same template clearing side effect
    input_suffix: String,
    /// `-r/--reverse-prompt PROMPT` (arg.cpp:1885, repeatable)
    antiprompt: Vec<String>,
    /// `-mli/--multiline-input` (arg.cpp:1933; common.h:566 default false)
    multiline_input: bool,
    /// `--simple-io` (arg.cpp:3816; common.h:567 default false)
    simple_io: bool,
    /// `--display-prompt/--no-display-prompt` (arg.cpp:1489; common.h:577
    /// default true)
    display_prompt: bool,
    /// `-e/--escape/--no-escape` (arg.cpp:1849; common.h:565 default true)
    escape: bool,
    /// `-sys/--system-prompt PROMPT` (arg.cpp:1774)
    system_prompt: String,
    /// `--show-timings/--no-show-timings` (arg.cpp:1790; common.h:570
    /// default true)
    show_timings: bool,
    /// `-sp/--special` (arg.cpp:1893; common.h:557 default false)
    special: bool,
    /// `--verbose-prompt` (arg.cpp:1481; common.h:576)
    verbose_prompt: bool,
    /// `-ptc/--print-token-count N` (arg.cpp:1862; common.h:461 default -1)
    n_print: i32,
    /// `--prompt-cache FNAME` (arg.cpp:1864; common.h:513)
    prompt_cache: String,
    /// `--prompt-cache-all` (arg.cpp:1871; common.h:562)
    prompt_cache_all: bool,
    /// `--prompt-cache-ro` (arg.cpp:1878; common.h:563)
    prompt_cache_ro: bool,
    /// `-st/--single-turn` (arg.cpp:1910; common.h:585)
    single_turn: bool,
    /// `-cnv/-no-cnv` (arg.cpp:1899) — `common_conversation_mode`
    /// (common.h:137-141): 0 disabled / 1 enabled / 2 auto (the default)
    conversation_mode: u8,
    /// `-co/--color [on|off|auto]` (arg.cpp:1497) — `params.use_color`
    /// defaults to `tty_can_use_colors()` (arg.cpp:1409), i.e. auto
    use_color: bool,
    /// `params.enable_chat_template` (common.h:647 default true; cleared by
    /// the --in-prefix/-bos/-suffix flags)
    enable_chat_template: bool,
    /// `--keep N` (arg.cpp:1679-1685; common.h:453 default 0, -1 = all)
    n_keep: i32,
    /// `--context-shift/--no-context-shift` (arg.cpp:1737-1741; common.h:571
    /// default false)
    ctx_shift: bool,
    /// `--diffusion-*` family (arg.cpp:4439-4484, the LLAMA_EXAMPLE_DIFFUSION
    /// flags; `common_params_diffusion` defaults per common.h:607-616)
    diffusion_steps: i32,
    diffusion_visual: bool,
    diffusion_eps: f32,
    diffusion_algorithm: i32,
    diffusion_alg_temp: f32,
    diffusion_block_length: i32,
    diffusion_cfg_scale: f32,
    diffusion_add_gumbel_noise: bool,
    // ---- GPU tasks ②/③: the foreign-backend surface ----
    /// `-ngl/--gpu-layers/--n-gpu-layers N` (common/arg.cpp's
    /// LLAMA_ARG_N_GPU_LAYERS) — number of layers to store in device memory;
    /// the *tail* layers incl. the output slot go first (llama-model.cpp:1521
    /// `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`). 0 (default)
    /// keeps the port's own CPU engine.
    n_gpu_layers: i32,
    /// `--device NAME` (`-dev`, arg.cpp) — the foreign backend device to use
    /// (e.g. `Vulkan0`); `cpu` selects the reference's libggml-cpu via DL
    /// (proves the emission plumbing without a GPU)
    device: Option<String>,
    /// `--ggml-libs DIR` — port-side: where the foreign ggml build lives
    /// (libggml-base.so + libggml-<backend>.so); the reference resolves its
    /// own build at startup, the port takes it explicitly
    ggml_libs: Option<String>,
    /// run on the foreign *CPU* backend even without -ngl (the reference's
    /// kernels through the DL path — the emission-layer verification mode)
    foreign_cpu: bool,
    /// `--list-devices` — print the foreign build's registered devices, exit
    list_devices: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model_path: String::new(),
            prompt: String::from("The capital of France is"),
            n_predict: 32,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            n_ctx: 512,
            // 0.0 keeps the historic llama-cli-rust default (greedy); pass
            // `--temp 0.8` for the reference default chain
            temp: 0.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            // common_params_sampling defaults (common.h:239-244, 259, 265-275)
            samplers: CommonSamplerType::default_chain(),
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_sequence_breakers: vec![
                "\n".to_string(),
                ":".to_string(),
                "\"".to_string(),
                "*".to_string(),
            ],
            adaptive_target: -1.0,
            adaptive_decay: 0.90,
            seed: 42,
            ignore_eos: false,
            chat: false,
            chat_template: None,
            perplexity: None,
            grammar: None,
            grammar_source: None,
            flash_attn: FlashAttn::Off,
            load_mode: LoadMode::Auto,
            lora_adapters: Vec::new(),
            speculative: CommonParamsSpeculative::default(),
            embedding: false,
            pooling: None,
            embd_ids: None,
            // ---- interactive defaults (common.h:461-590, :647) ----
            interactive: false,
            interactive_first: false,
            input_prefix: String::new(),
            input_prefix_bos: false,
            input_suffix: String::new(),
            antiprompt: Vec::new(),
            multiline_input: false,
            simple_io: false,
            display_prompt: true,
            escape: true,
            system_prompt: String::new(),
            show_timings: true,
            special: false,
            verbose_prompt: false,
            n_print: -1,
            prompt_cache: String::new(),
            prompt_cache_all: false,
            prompt_cache_ro: false,
            single_turn: false,
            conversation_mode: interactive::CONVERSATION_AUTO,
            use_color: tty_can_use_colors(),
            enable_chat_template: true,
            n_keep: 0,
            ctx_shift: false,
            // common_params_diffusion defaults (common.h:403-415)
            diffusion_steps: 128,
            diffusion_visual: false,
            diffusion_eps: 0.0,
            diffusion_algorithm: 4, // DIFFUSION_ALGORITHM_CONFIDENCE_BASED
            diffusion_alg_temp: 0.0,
            diffusion_block_length: 0,
            diffusion_cfg_scale: 0.0,
            diffusion_add_gumbel_noise: false,
            // GPU surface defaults: CPU-only, no device, no foreign libs
            n_gpu_layers: 0,
            device: None,
            ggml_libs: None,
            foreign_cpu: false,
            list_devices: false,
        }
    }
}

/// `tty_can_use_colors` (common.cpp:1185-1202): NO_COLOR beats everything,
/// a "dumb" TERM or no terminal disables.
fn tty_can_use_colors() -> bool {
    if let Ok(no_color) = std::env::var("NO_COLOR") {
        if !no_color.is_empty() {
            return false;
        }
    }
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return false;
        }
        std::io::IsTerminal::is_terminal(&std::io::stdout())
    } else {
        false
    }
}

/// `common_arg_utils::is_truthy` (arg.cpp:1331-1333)
fn is_truthy(v: &str) -> bool {
    matches!(v, "on" | "enabled" | "true" | "1")
}

/// `common_arg_utils::is_falsey` (arg.cpp:1335-1337)
fn is_falsey(v: &str) -> bool {
    matches!(v, "off" | "disabled" | "false" | "0")
}

/// `common_arg_utils::is_autoy` (arg.cpp:1339-1341)
fn is_autoy(v: &str) -> bool {
    matches!(v, "auto" | "-1")
}

const USAGE: &str = "usage: llama-cli -m model.gguf [-p prompt] [-n n] [-t threads] [-c ctx]
       [--temp t] [--top-k k] [--top-p p] [--min-p p] [--repeat-penalty r]
       [--repeat-last-n n] [--seed s] [--ignore-eos] [-fa on|off|auto] [--flash-attn]
       [--samplers \"a;b;c\"] [--sampler-seq|--sampling-seq abc]
       [-lm MODE|--load-mode MODE] (auto|none|mmap|mlock|mmap+mlock|dio)
       [--dry-multiplier m] [--dry-base b] [--dry-allowed-length n]
       [--dry-penalty-last-n n] [--dry-sequence-breaker s|--dry-sequence-breaker none]
       [--adaptive-target p] [--adaptive-decay d]
       [--lora FNAME] [--lora-scaled FNAME:SCALE,...]
       [--chat] [--chat-template NAME|TEMPLATE] [--single-turn] [-cnv]
       [--grammar GBNF | --grammar-file FILE | -j/--json-schema SCHEMA
        | -jf/--json-schema-file FILE] [--perplexity FILE]
       [-md FNAME|--model-draft|--spec-draft-model FNAME] [--spec-type TYPES]
       [--spec-draft-n-max N] [--spec-draft-n-min N]
       [--spec-draft-p-min P|--draft-p-min P] [--spec-draft-p-split P|--draft-p-split P]
       [--spec-draft-backend-sampling|--no-spec-draft-backend-sampling]
       [--spec-synth-len L] [--spec-synth-rates P0,P1,...]
       [-ngl N|--gpu-layers N] [--device NAME] [--ggml-libs DIR]
       [--foreign-cpu] [--list-devices]";

/// `--lora-scaled`'s own usage block (arg.cpp:2960-2963) — `common_arg::to_string()`
/// in the "error while handling argument" wrapper (arg.cpp:868-873).
const LORA_SCALED_USAGE: &str = "\
--lora-scaled FNAME:SCALE,...           path to LoRA adapter with user defined scaling
                                        (format: FNAME:SCALE,...)
                                        note: use comma-separated values
";

/// `parse_csv_row` (common/arg.cpp:1347-1389): comma-separated with `"…"`
/// quoting and `""` escapes; quotes inside an unquoted field are literal. The
/// last field is always pushed, so an empty string yields one empty field.
fn parse_csv_row(input: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let bytes: Vec<char> = input.chars().collect();

    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == '"' {
            if !in_quotes {
                if !field.is_empty() {
                    // quote in the middle of an unquoted field: literal
                    field.push('"');
                } else {
                    in_quotes = true;
                }
            } else if i + 1 < bytes.len() && bytes[i + 1] == '"' {
                field.push('"');
                i += 1;
            } else {
                in_quotes = false;
            }
        } else if ch == ',' {
            if in_quotes {
                field.push(',');
            } else {
                fields.push(std::mem::take(&mut field));
            }
        } else {
            field.push(ch);
        }
        i += 1;
    }
    fields.push(field);
    fields
}

/// `string_split<std::string>(item, ':')` (common/common.h:817-831): keeps
/// empty fields, so "a:b:" yields 3 parts and is rejected by the caller.
fn string_split(s: &str, delim: char) -> Vec<String> {
    s.split(delim).map(|p| p.to_string()).collect()
}

/// `std::stof` (arg.cpp:2970): parses the longest valid float prefix and
/// throws on garbage ("invalid_argument"). The reference's wrapper prints the
/// exception text, here "stof".
fn stof(s: &str) -> Result<f32, String> {
    let t = s.trim_start();
    for end in (1..=t.len()).rev() {
        if !t.is_char_boundary(end) {
            continue;
        }
        if let Ok(v) = t[..end].parse::<f32>() {
            return Ok(v);
        }
    }
    Err("stof".to_string())
}

/// `-j` / `-jf` body: `json_schema_to_grammar(json::parse(schema))`
/// (common/arg.cpp:2282/2299, `force_gbnf` left at its `false` default —
/// json-schema-to-grammar.h:9). The error text is returned verbatim: the
/// reference's converter throws
/// `std::invalid_argument("JSON schema conversion failed:\n" + …)`
/// (json-schema-to-grammar.cpp:976-998) which the CLI prints to stderr and
/// exits 1 on; the port's `json_schema_to_grammar` produces that same text
/// (crates/llama/src/json_schema.rs:2563-2567).
fn schema_to_grammar(schema: &str) -> Result<String, String> {
    let json = llama::json_schema::Json::parse(schema)?;
    llama::json_schema::json_schema_to_grammar(&json, false)
}

/// `common_arg::to_string()` block of the two schema flags (arg.cpp:2279-2300),
/// as the reference prints it from `parse_cli_args`' exception wrapper. The
/// block carries its own trailing newline (that is why the reference shows two
/// blank lines before "to show complete usage").
const JSON_SCHEMA_USAGE: &str = "\
-j,    --json-schema SCHEMA             JSON schema to constrain generations (https://json-schema.org/), e.g.
                                        `{\"type\": \"object\"}` for any JSON object
";

/// `-jf`'s block (the description is longer, so `common_arg::to_string` wraps it
/// at 120 columns).
const JSON_SCHEMA_FILE_USAGE: &str = "\
-jf,   --json-schema-file FILE          File containing a JSON schema to constrain generations
                                        (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON
                                        object
";

/// `error while handling argument "<arg>": <msg>\n\nusage:\n<usage>\n\nto show
/// complete usage, run with -h` — the reference's wrapper for any
/// option-handler exception (arg.cpp:866-873). `msg` is inserted verbatim
/// (some carry their own trailing newline, e.g. the -jf file-open error).
fn arg_error(arg: &str, msg: &str, usage: &str) -> String {
    format!(
        "error while handling argument \"{arg}\": {msg}\n\nusage:\n{usage}\n\n\
         to show complete usage, run with -h"
    )
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 1;
    // arg.cpp:2165 — `static bool defaults_cleared` of --dry-sequence-breaker:
    // only the first use clears the default breaker list
    let mut dry_defaults_cleared = false;
    // value fetch helper
    macro_rules! val {
        () => {{
            i += 1;
            if i >= argv.len() {
                return Err(format!("missing value for {}", argv[i - 1]));
            }
            argv[i].clone()
        }};
    }
    while i < argv.len() {
        match argv[i].as_str() {
            "-m" | "--model" => a.model_path = val!(),
            "-p" | "--prompt" => a.prompt = val!(),
            "-n" | "--n-predict" => a.n_predict = val!().parse().map_err(|e| format!("-n: {e}"))?,
            "-t" | "--threads" => a.n_threads = val!().parse().map_err(|e| format!("-t: {e}"))?,
            // GPU tasks ②/③ — the reference's -ngl family (arg.cpp
            // LLAMA_ARG_N_GPU_LAYERS) plus the port's --ggml-libs resolver
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => {
                a.n_gpu_layers = val!().parse().map_err(|e| format!("-ngl: {e}"))?
            }
            "--device" | "-dev" => a.device = Some(val!()),
            "--ggml-libs" => a.ggml_libs = Some(val!()),
            "--foreign-cpu" => a.foreign_cpu = true,
            "--list-devices" => a.list_devices = true,
            "-c" | "--ctx-size" => a.n_ctx = val!().parse().map_err(|e| format!("-c: {e}"))?,
            "--temp" | "--temperature" => {
                a.temp = val!().parse().map_err(|e| format!("--temp: {e}"))?
            }
            "--top-k" => a.top_k = val!().parse().map_err(|e| format!("--top-k: {e}"))?,
            "--top-p" => a.top_p = val!().parse().map_err(|e| format!("--top-p: {e}"))?,
            "--min-p" => a.min_p = val!().parse().map_err(|e| format!("--min-p: {e}"))?,
            "--repeat-penalty" => {
                a.repeat_penalty = val!()
                    .parse()
                    .map_err(|e| format!("--repeat-penalty: {e}"))?
            }
            "--repeat-last-n" => {
                a.repeat_last_n = val!()
                    .parse()
                    .map_err(|e| format!("--repeat-last-n: {e}"))?
            }
            // --samplers "names;separated;by;semicolons" (arg.cpp:1982-1988)
            "--samplers" => {
                let v = val!();
                let names: Vec<String> = v.split(';').map(|s| s.to_string()).collect();
                a.samplers = llama::sampling::common_sampler_types_from_names(&names);
            }
            // --sampler-seq/--sampling-seq "chars" (arg.cpp:1998-2003)
            "--sampler-seq" | "--sampling-seq" => {
                a.samplers = llama::sampling::common_sampler_types_from_chars(&val!());
            }
            // DRY flags (arg.cpp:2123-2182)
            "--dry-multiplier" => {
                a.dry_multiplier = val!().parse().map_err(|e| format!("--dry-multiplier: {e}"))?
            }
            "--dry-base" => {
                // arg.cpp:2133-2139: values < 1.0 are silently ignored
                let v: f32 = val!().parse().map_err(|e| format!("--dry-base: {e}"))?;
                if v >= 1.0 {
                    a.dry_base = v;
                }
            }
            "--dry-allowed-length" => {
                a.dry_allowed_length =
                    val!().parse().map_err(|e| format!("--dry-allowed-length: {e}"))?
            }
            "--dry-penalty-last-n" => {
                let v: i32 = val!().parse().map_err(|e| format!("--dry-penalty-last-n: {e}"))?;
                if v < 0 {
                    return Err(format!("error: invalid dry-penalty-last-n = {v}"));
                }
                a.dry_penalty_last_n = v;
            }
            "--dry-sequence-breaker" => {
                // arg.cpp:2163-2181: the first use clears the defaults
                // (static defaults_cleared); "none" clears the list outright
                let v = val!();
                if !dry_defaults_cleared {
                    a.dry_sequence_breakers.clear();
                    dry_defaults_cleared = true;
                }
                if v == "none" {
                    a.dry_sequence_breakers.clear();
                } else {
                    a.dry_sequence_breakers.push(v);
                }
            }
            // adaptive-p flags (arg.cpp:2184-2202)
            "--adaptive-target" => {
                a.adaptive_target = val!().parse().map_err(|e| format!("--adaptive-target: {e}"))?
            }
            "--adaptive-decay" => {
                a.adaptive_decay = val!().parse().map_err(|e| format!("--adaptive-decay: {e}"))?
            }
            "--seed" | "-s" => {
                let v: i64 = val!().parse().map_err(|e| format!("--seed: {e}"))?;
                a.seed = v as u32; // -1 => LLAMA_DEFAULT_SEED
            }
            "--ignore-eos" => a.ignore_eos = true,
            "--chat" => a.chat = true,
            "--chat-template" => {
                a.chat_template = Some(val!());
                a.chat = true;
            }
            // (-st/--single-turn parsed with the interactive family above)
            "-lm" | "--load-mode" => {
                // llama_load_mode_from_str (llama.cpp:565-596): the accepted
                // mode words; anything else is "invalid argument"
                a.load_mode = match val!().as_str() {
                    "auto" => LoadMode::Auto,
                    "none" => LoadMode::None,
                    "mmap" => LoadMode::Mmap,
                    "mlock" => LoadMode::Mlock,
                    "mmap+mlock" => LoadMode::MmapMlock,
                    "dio" => LoadMode::Dio,
                    other => {
                        return Err(format!("error: invalid load mode: {other}"))
                    }
                };
            }
            "-fa" | "--flash-attn" => {
                // value is optional (reference: "[on|off|auto]" + is_truthy/
                // is_falsey/is_autoy, common/arg.cpp:1754-1764); a following
                // token is consumed only when it is a recognised mode word so
                // `-fa -p ...` keeps working.
                let mut mode = "on".to_string();
                if i + 1 < argv.len() {
                    let next = argv[i + 1].to_ascii_lowercase();
                    if matches!(
                        next.as_str(),
                        "on" | "off" | "auto" | "true" | "false" | "1" | "0" | "yes" | "no"
                    ) {
                        i += 1;
                        mode = argv[i].clone();
                    }
                }
                a.flash_attn = match mode.to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" | "yes" | "enabled" => FlashAttn::On,
                    "off" | "false" | "0" | "no" | "disabled" => FlashAttn::Off,
                    "auto" => FlashAttn::Auto,
                    other => return Err(format!(
                        "error: unknown value for --flash-attn: '{other}' (expected on|off|auto)"
                    )),
                };
            }
            "--grammar" => {
                // common/arg.cpp:2265-2271 — COMMON_GRAMMAR_TYPE_USER
                a.grammar = Some(val!());
                a.grammar_source = Some("--grammar");
            }
            "--grammar-file" => {
                // common/arg.cpp:2273-2277 — read_file(value)
                let path = val!();
                match std::fs::read_to_string(&path) {
                    Ok(s) => a.grammar = Some(s),
                    Err(e) => return Err(format!("--grammar-file {path}: {e}")),
                }
                a.grammar_source = Some("--grammar-file");
            }
            "-j" | "--json-schema" => {
                // common/arg.cpp:2279-2283 —
                // `params.sampling.grammar = {OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))}`
                let flag = argv[i].clone();
                let value = val!();
                let gbnf = schema_to_grammar(&value)
                    .map_err(|e| arg_error(&flag, &e, JSON_SCHEMA_USAGE))?;
                a.grammar = Some(gbnf);
                a.grammar_source = Some("-j/--json-schema");
            }
            "-jf" | "--json-schema-file" => {
                // common/arg.cpp:2286-2300: read the file, then the same
                // parse+convert as `-j`
                let flag = argv[i].clone();
                let path = val!();
                let schema = std::fs::read_to_string(&path).map_err(|_| {
                    // string_format("error: failed to open file '%s'\n", …) — the
                    // trailing newline is part of the message (arg.cpp:2291)
                    arg_error(
                        &flag,
                        &format!("error: failed to open file '{path}'\n"),
                        JSON_SCHEMA_FILE_USAGE,
                    )
                })?;
                let gbnf = schema_to_grammar(&schema)
                    .map_err(|e| arg_error(&flag, &e, JSON_SCHEMA_FILE_USAGE))?;
                a.grammar = Some(gbnf);
                a.grammar_source = Some("-jf/--json-schema-file");
            }
            "--perplexity" => a.perplexity = Some(val!()),
            "--lora" => {
                // common/arg.cpp:2950-2959 — `parse_csv_row(value)`, scale 1.0
                let v = val!();
                for item in parse_csv_row(&v) {
                    a.lora_adapters.push((item, 1.0));
                }
            }
            "--lora-scaled" => {
                // common/arg.cpp:2960-2974 — `string_split(item, ':')` must
                // yield exactly two parts; the scale is `std::stof`
                let flag = argv[i].clone();
                let v = val!();
                for item in parse_csv_row(&v) {
                    let parts = string_split(&item, ':');
                    if parts.len() != 2 {
                        return Err(arg_error(
                            &flag,
                            "lora-scaled format: FNAME:SCALE",
                            LORA_SCALED_USAGE,
                        ));
                    }
                    let scale =
                        stof(&parts[1]).map_err(|e| arg_error(&flag, &e, LORA_SCALED_USAGE))?;
                    a.lora_adapters.push((parts[0].clone(), scale));
                }
            }
            // ---- speculative decoding (common/arg.cpp:4135-4253, the flags
            // this revision exposes for {SPECULATIVE, SERVER, CLI}) ----
            "--spec-draft-model" | "-md" | "--model-draft" => {
                // arg.cpp:4236-4243 — `params.speculative.draft.mparams.path = value`
                a.speculative.draft.model_path = val!();
            }
            "--spec-type" => {
                // arg.cpp:4244-4253 — split on ',', `common_speculative_types_from_names`,
                // insert (append) into params.speculative.types. The default
                // `{none}` stays in front: `common_speculative_init` only looks
                // at the per-type enable bits, so `--spec-type draft-simple`
                // yields {none, draft-simple} — exactly the reference's state.
                let v = val!();
                let names: Vec<String> = v.split(',').map(|s| s.to_string()).collect();
                match common_speculative_types_from_names(&names) {
                    Ok(types) => a.speculative.types.extend(types),
                    Err(e) => return Err(format!("unknown option '--spec-type': {e}")),
                }
            }
            "--spec-draft-n-max" => {
                // arg.cpp:4135-4144 — negative is "invalid value"
                let v: i32 = val!()
                    .parse()
                    .map_err(|_| "--spec-draft-n-max: invalid value")?;
                if v < 0 {
                    return Err("--spec-draft-n-max: invalid value".into());
                }
                a.speculative.draft.n_max = v;
            }
            "--spec-draft-n-min" => {
                // arg.cpp:4145-4151
                let v: i32 = val!()
                    .parse()
                    .map_err(|_| "--spec-draft-n-min: invalid value")?;
                a.speculative.draft.n_min = v;
            }
            "--spec-draft-p-min" | "--draft-p-min" => {
                // arg.cpp:4192-4198 — std::stof
                let v = val!();
                a.speculative.draft.p_min = stof(&v).map_err(|e| format!("--draft-p-min: {e}"))?;
            }
            "--spec-draft-p-split" | "--draft-p-split" => {
                // arg.cpp:4185-4191 — std::stof
                let v = val!();
                a.speculative.draft.p_split =
                    stof(&v).map_err(|e| format!("--draft-p-split: {e}"))?;
            }
            "--spec-draft-backend-sampling" => {
                // arg.cpp:4199-4207 — boolean pair (value flags: no argument)
                a.speculative.draft.backend_sampling = true;
            }
            "--no-spec-draft-backend-sampling" => {
                a.speculative.draft.backend_sampling = false;
            }
            "--spec-draft-sampling" => {
                // arg.cpp:4219-4233 (a7b94df2c): how the draft is sampled
                let v = val!();
                match v.as_str() {
                    "greedy" => a.speculative.draft.probabilistic = false,
                    "probabilistic" => a.speculative.draft.probabilistic = true,
                    _ => {
                        return Err(
                            "invalid value, must be one of: greedy, probabilistic".into()
                        )
                    }
                }
            }
            "--spec-synth-len" => {
                // arg.cpp:4152-4164 — stod, strip; -1.0 is "invalid value"
                let v = val!();
                let text = v.trim();
                let length: f64 = text
                    .parse()
                    .map_err(|_| "--spec-synth-len: invalid value".to_string())?;
                if length == -1.0 {
                    return Err("--spec-synth-len: invalid value".into());
                }
                a.speculative.synth_len = length;
            }
            "--spec-synth-rates" => {
                // arg.cpp:4165-4183 — comma list of stod values
                let v = val!();
                let mut rates = Vec::new();
                for raw in v.split(',') {
                    let rate: f64 = raw
                        .trim()
                        .parse()
                        .map_err(|_| "--spec-synth-rates: invalid value".to_string())?;
                    rates.push(rate);
                }
                a.speculative.synth_rates = rates;
            }
            // ---- the ngram family's value flags (arg.cpp:4254-4378) — the
            // same add_opt U16/I32 pattern as --spec-draft-n-max
            // ----
            "--spec-ngram-mod-n-min" => {
                // arg.cpp:4255-4264
                a.speculative.ngram_mod.n_min = val!().parse().map_err(|_| {
                    "--spec-ngram-mod-n-min: invalid value".to_string()
                })?;
            }
            "--spec-ngram-mod-n-max" => {
                // arg.cpp:4265-4274
                a.speculative.ngram_mod.n_max = val!().parse().map_err(|_| {
                    "--spec-ngram-mod-n-max: invalid value".to_string()
                })?;
            }
            "--spec-ngram-mod-n-match" => {
                // arg.cpp:4275-4285
                a.speculative.ngram_mod.n_match = val!().parse().map_err(|_| {
                    "--spec-ngram-mod-n-match: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-size-n" => {
                // arg.cpp:4286-4295
                a.speculative.ngram_simple.size_n = val!().parse().map_err(|_| {
                    "--spec-ngram-simple-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-size-m" => {
                // arg.cpp:4296-4305
                a.speculative.ngram_simple.size_m = val!().parse().map_err(|_| {
                    "--spec-ngram-simple-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-min-hits" => {
                // arg.cpp:4306-4316
                a.speculative.ngram_simple.min_hits = val!().parse().map_err(|_| {
                    "--spec-ngram-simple-min-hits: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-size-n" => {
                // arg.cpp:4317-4326
                a.speculative.ngram_map_k.size_n = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-size-m" => {
                // arg.cpp:4327-4336
                a.speculative.ngram_map_k.size_m = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-min-hits" => {
                // arg.cpp:4337-4347
                a.speculative.ngram_map_k.min_hits = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k-min-hits: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-size-n" => {
                // arg.cpp:4348-4357
                a.speculative.ngram_map_k4v.size_n = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k4v-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-size-m" => {
                // arg.cpp:4358-4367
                a.speculative.ngram_map_k4v.size_m = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k4v-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-min-hits" => {
                // arg.cpp:4368-4377
                a.speculative.ngram_map_k4v.min_hits = val!().parse().map_err(|_| {
                    "--spec-ngram-map-k4v-min-hits: invalid value".to_string()
                })?;
            }
            // ---- embeddings (`--embedding/--embeddings`, arg.cpp:3473-3476)
            // ----
            // The flags live in common's option table, but tools/cli does not
            // register them in its `arg_to_options` set — the pinned reference
            // rejects them with `error: invalid argument: %s` (arg.cpp:824 /
            // :1217; the old tools/main `-fe` embedding print is gone from
            // this revision, so there is no behaviour to mirror beyond the
            // rejection). Same for `--pooling` (arg.cpp:2310) and
            // `--embd-normalize` (arg.cpp:3283) — decoder-model embeddings are
            // served by llama-server's `-fe` path instead.
            "-fe" | "--embedding" | "--embeddings" => {
                a.embedding = true;
            }
            "--no-embeddings" => {
                a.embedding = false;
            }
            "--pooling" => {
                a.pooling = Some(val!().to_string());
            }
            "--embd-ids" => {
                a.embd_ids = Some(val!().to_string());
            }
            // `--embd-normalize` (arg.cpp:3283) — not implemented (the
            // reference's own cli rejects it at this revision too)
            "--embd-normalize" => {
                return Err(format!("error: invalid argument: {}", argv[i]));
            }
            // removed params (arg.cpp:4378-4395) — `arg_removed` aborts with
            // the message below
            "--draft" | "--draft-n" | "--draft-max" => {
                return Err("the argument has been removed. use --spec-draft-n-max or \
                            --spec-ngram-mod-n-max"
                    .into());
            }
            "--draft-min" | "--draft-n-min" => {
                return Err("the argument has been removed. use --spec-draft-n-min or \
                            --spec-ngram-mod-n-min"
                    .into());
            }
            // ---- the interactive surface (common/arg.cpp:1481-1963,
            // :3816-3824) — tools/completion's flag family ----
            "--verbose-prompt" => a.verbose_prompt = true,
            "--display-prompt" => a.display_prompt = true,
            "--no-display-prompt" => a.display_prompt = false,
            "-co" | "--color" => {
                // arg.cpp:1497-1511 — on/off/auto, auto = tty_can_use_colors
                let v = val!();
                a.use_color = match v.as_str() {
                    v if is_truthy(v) => true,
                    v if is_falsey(v) => false,
                    v if is_autoy(v) => tty_can_use_colors(),
                    _ => {
                        return Err(format!("error: unknown value for --color: '{v}'\n"));
                    }
                };
            }
            "-sys" | "--system-prompt" => a.system_prompt = val!(),
            "-sysf" | "--system-prompt-file" => {
                // arg.cpp:1810-1819 — read the file, strip one trailing \n
                let f = val!();
                match std::fs::read_to_string(&f) {
                    Ok(mut s) => {
                        if s.ends_with('\n') {
                            s.pop();
                        }
                        a.system_prompt = s;
                    }
                    Err(_) => {
                        return Err(format!("error: failed to open file '{f}'\n"));
                    }
                }
            }
            "--show-timings" => a.show_timings = true,
            "--no-show-timings" => a.show_timings = false,
            "-e" | "--escape" => a.escape = true,
            "--no-escape" => a.escape = false,
            "-ptc" | "--print-token-count" => {
                a.n_print = val!().parse().map_err(|_| "-ptc: invalid value".to_string())?;
            }
            "--prompt-cache" => a.prompt_cache = val!(),
            "--prompt-cache-all" => a.prompt_cache_all = true,
            "--prompt-cache-ro" => a.prompt_cache_ro = true,
            "-r" | "--reverse-prompt" => a.antiprompt.push(val!()),
            "-sp" | "--special" => a.special = true,
            "-cnv" | "--conversation" => a.conversation_mode = interactive::CONVERSATION_ENABLED,
            "-no-cnv" | "--no-conversation" => {
                a.conversation_mode = interactive::CONVERSATION_DISABLED;
            }
            "-st" | "--single-turn" => a.single_turn = true,
            "-i" | "--interactive" => a.interactive = true,
            "-if" | "--interactive-first" => a.interactive_first = true,
            "-mli" | "--multiline-input" => a.multiline_input = true,
            "--in-prefix-bos" => {
                // arg.cpp:1940-1945 — also clears enable_chat_template
                a.input_prefix_bos = true;
                a.enable_chat_template = false;
            }
            "--in-prefix" => {
                // arg.cpp:1948-1954
                a.input_prefix = val!();
                a.enable_chat_template = false;
            }
            "--in-suffix" => {
                // arg.cpp:1956-1962
                a.input_suffix = val!();
                a.enable_chat_template = false;
            }
            "--simple-io" => a.simple_io = true,
            "--keep" => {
                // arg.cpp:1679-1685
                let v: i32 = val!().parse().map_err(|_| "--keep: invalid value".to_string())?;
                if v < -1 {
                    return Err("error: invalid value for --keep".into());
                }
                a.n_keep = v;
            }
            "--context-shift" => a.ctx_shift = true,
            "--no-context-shift" => a.ctx_shift = false,
            // ---- the diffusion family (arg.cpp:4439-4484) ----
            "--diffusion-steps" => {
                a.diffusion_steps =
                    val!().parse().map_err(|_| "--diffusion-steps: invalid value".to_string())?;
            }
            "--diffusion-visual" => a.diffusion_visual = true,
            "--diffusion-eps" => {
                a.diffusion_eps = stof(&val!()).map_err(|e| format!("--diffusion-eps: {e}"))?;
            }
            "--diffusion-algorithm" => {
                a.diffusion_algorithm = val!()
                    .parse()
                    .map_err(|_| "--diffusion-algorithm: invalid value".to_string())?;
            }
            "--diffusion-alg-temp" => {
                a.diffusion_alg_temp =
                    stof(&val!()).map_err(|e| format!("--diffusion-alg-temp: {e}"))?;
            }
            "--diffusion-block-length" => {
                a.diffusion_block_length = val!()
                    .parse()
                    .map_err(|_| "--diffusion-block-length: invalid value".to_string())?;
            }
            "--diffusion-cfg-scale" => {
                a.diffusion_cfg_scale =
                    stof(&val!()).map_err(|e| format!("--diffusion-cfg-scale: {e}"))?;
            }
            "--diffusion-add-gumbel-noise" => {
                a.diffusion_add_gumbel_noise = stof(&val!())
                    .map_err(|e| format!("--diffusion-add-gumbel-noise: {e}"))?
                    != 0.0;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}\n{USAGE}")),
        }
        i += 1;
    }
    // --list-devices runs before any model work (the reference prints devices
    // with no -m; arg.cpp LLAMA_ARG_LIST_DEVICES has no model requirement)
    if a.model_path.is_empty() && !a.list_devices {
        return Err(USAGE.to_string());
    }
    Ok(a)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // `--list-devices` (the reference's llama-cli --list-devices, common/
    // arg.cpp LLAMA_ARG_LIST_DEVICES): print the foreign build's devices and
    // exit before any model work
    if args.list_devices {
        let dir = args.ggml_libs.clone().unwrap_or_else(|| {
            eprintln!("--list-devices needs --ggml-libs DIR (the foreign ggml build's bin/)");
            std::process::exit(1);
        });
        match ggml::backend_emit::list_devices(std::path::Path::new(&dir)) {
            Ok(devs) => {
                println!("Available devices:");
                for d in devs {
                    println!("  {d}");
                }
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // "infer the speculative type from the draft GGUF metadata when none is
    // requested" (common/arg.cpp:564-570): `spec_types_is_default()` — the
    // types are exactly the `{none}` default (arg.cpp:357-359) — plus a
    // non-empty `-md` path reads `common_speculative_types_from_gguf` on the
    // DRAFT file (speculative.cpp:2290-2325: the nextn trio's eh_proj at
    // block_count-1 selects draft-mtp; the markov head distinguishes
    // dspark). The HF-sidecar half of the reference's inference
    // (arg.cpp:544-562) is download machinery the local-file port omits.
    if args.speculative.types.len() == 1
        && args.speculative.types[0] == llama::speculative::CommonSpeculativeType::None
        && !args.speculative.draft.model_path.is_empty()
    {
        let types = llama::speculative::common_speculative_types_from_gguf(
            &args.speculative.draft.model_path,
        );
        if !types.is_empty() {
            args.speculative.types = types;
        }
    }

    let _t0 = std::time::Instant::now();
    let gguf = match ggml::Gguf::open(&args.model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to open gguf: {e}");
            std::process::exit(1);
        }
    };

    // the reference's model-load banner: `llama_model_loader::print_info`
    // runs first (llama.cpp:324, right after the loader is constructed),
    // then `model->print_info()` after hparams/vocab/stats (llama.cpp:362 —
    // which ends with `vocab.print_info()`, llama-model.cpp:2174). Both go
    // through LLAMA_LOG_INFO, i.e. stderr under the default callback.
    {
        let (n_elements, n_bytes) = llama::display::loader_stats(&gguf);
        let ftype = gguf.get_u32("general.file_type").unwrap_or(15) as i32;
        llama::display::loader_print_info(gguf.version, ftype, n_bytes, n_elements);
    }

    let mmap: Arc<memmap2::Mmap> = {
        // reopen and mmap the same file (gguf reader owns its own map; we need
        // a second one to hand to the model loader)
        let f = std::fs::File::open(&args.model_path).unwrap();
        // SAFETY: read-only usage of a model file
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };

    let vocab = llama::vocab::Vocab::load(&gguf).expect("vocab");
    let model = load_model(&gguf, mmap).expect("model");
    llama::display::model_print_info(&model, &vocab);

    // `-lm mlock` / `-lm mmap+mlock`: pin the weights mapping in RAM (the
    // reference's LLAMA_LOAD_MODE_MLOCK/MMAP_MLOCK paths — llama-model.cpp:1434
    // + the loader's per-tensor grow_to, llama-model-loader.cpp:1661-1663; the
    // port locks the whole mapping once, right after the load). Declared after
    // `model` so the drop order releases the lock before the mapping unmaps.
    // `-lm dio` fails like the reference on a platform without DirectIO
    // support (the port's build has none).
    if args.load_mode == LoadMode::Dio {
        eprintln!("error: failed to load model: DirectIO not supported");
        std::process::exit(1);
    }
    let mut mlock = llama::mlock::Mlock::new();
    if matches!(args.load_mode, LoadMode::Mlock | LoadMode::MmapMlock) {
        mlock.init(model_external_ptr(&model) as *mut u8);
        mlock.grow_to(model_external_len(&model));
    }
    let hp = &model.hparams;

    // ---- chat prompt formatting (llama::chat, non-jinja builtin path) ----
    let mut prompt = args.prompt.clone();
    let mut chat_eot: Option<&'static str> = None;
    if args.chat {
        let tmpl_src: &str = match &args.chat_template {
            Some(s) => s.as_str(),
            None => gguf.get_str("tokenizer.chat_template").unwrap_or(""),
        };
        if tmpl_src.is_empty() {
            eprintln!("[chat] no --chat-template and the model has no tokenizer.chat_template");
            std::process::exit(1);
        }
        // exact builtin name wins; otherwise sniff the template source
        let tmpl = ChatTemplate::from_name(tmpl_src).unwrap_or_else(|| chat::detect(tmpl_src));
        let msgs = [ChatMessage::new(Role::User, args.prompt.clone())];
        match chat::apply(tmpl, &msgs, true) {
            Ok(formatted) => {
                eprintln!(
                    "[chat] template = {} ({}), user message {} chars -> prompt {} chars",
                    tmpl.name().unwrap_or("detected"),
                    format!("{tmpl:?}"),
                    args.prompt.len(),
                    formatted.len()
                );
                eprintln!("[chat] formatted prompt: {formatted:?}");
                prompt = formatted;
                chat_eot = Some(chat::eot_prefix(tmpl));
            }
            Err(e) => {
                eprintln!("[chat] failed to apply template: {e}");
                std::process::exit(1);
            }
        }
    }

    // tokenise
    let tokens = vocab.tokenize(&prompt, true, true);
    // the interactive REPL echoes the prompt itself (completion.cpp:704-723)
    // — the port's single-shot diagnostic print is suppressed there
    let model_chat_template: String = gguf
        .get_str("tokenizer.chat_template")
        .unwrap_or("")
        .to_string();
    let has_chat_template =
        !model_chat_template.is_empty() || args.chat_template.as_deref().is_some_and(|s| !s.is_empty());
    let use_interactive = args.interactive
        || args.interactive_first
        || args.conversation_mode == interactive::CONVERSATION_ENABLED
        || (args.conversation_mode == interactive::CONVERSATION_AUTO
            && has_chat_template
            && args.single_turn);
    if !use_interactive {
        print!("prompt tokens: {:?}\n{prompt}", tokens);
        {
            std::io::stdout().flush().ok();
        }
    }

    // ---- the encoder-embedding mode (BERT / eurobert / t5encoder / variants) ----
    // The port's `-fe` wiring for the encoder archs: `llama_encode` through
    // `EncoderContext` (exactly what llama-server's -fe branch builds), the
    // pooled/none rows printed in full. The pinned reference's own cli
    // rejects the flag (arg.cpp:824), so this arm is verified against the
    // reference `llama_encode` dumps (parity/encode_*.bin — the same
    // artifacts bert_e2e.rs/t5_e2e.rs pin the library with) plus the
    // reference server's /embedding endpoint.
    //
    // Encoder-only archs have no generation graph in this port (or the
    // reference: llama_model_has_decoder is false for them), so the CLI always
    // drives the embedding path for them — the reference `llama-embedding`
    // example's `-fe` semantics, with the pool default taken from the GGUF
    // (resolve_pooling(UNSPECIFIED, hparams.pooling_type)).
    let encoder_only = matches!(
        model.arch,
        llama::arch::LlmArch::BERT
            | llama::arch::LlmArch::EUROBERT
            | llama::arch::LlmArch::T5ENCODER
            | llama::arch::LlmArch::GEMMA_EMBEDDING
            | llama::arch::LlmArch::GEMMA_EMBEDDING2
            | llama::arch::LlmArch::LLAMA_EMBED
            | llama::arch::LlmArch::JINA_BERT_V2
            | llama::arch::LlmArch::JINA_BERT_V3
            | llama::arch::LlmArch::NOMIC_BERT
            | llama::arch::LlmArch::NOMIC_BERT_MOE
            | llama::arch::LlmArch::NEO_BERT
            | llama::arch::LlmArch::MODERN_BERT
    );
    if encoder_only || args.embedding {
        if encoder_only && !args.embedding {
            eprintln!(
                "note: arch '{}' is encoder-only — embedding via llama_encode \
                 (llama-embedding -fe semantics; pool from the GGUF)",
                model.arch.name()
            );
        }
        let enc_ids: Vec<i32> = match &args.embd_ids {
            Some(list) => list
                .split(',')
                .map(|v| v.trim().parse::<i32>())
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| {
                    eprintln!("--embd-ids: invalid id list");
                    std::process::exit(1);
                }),
            None => tokens.clone(),
        };
        match model.arch {
            llama::arch::LlmArch::BERT
            | llama::arch::LlmArch::EUROBERT
            | llama::arch::LlmArch::T5ENCODER
            | llama::arch::LlmArch::GEMMA_EMBEDDING
            | llama::arch::LlmArch::GEMMA_EMBEDDING2
            | llama::arch::LlmArch::LLAMA_EMBED
            | llama::arch::LlmArch::JINA_BERT_V2
            | llama::arch::LlmArch::JINA_BERT_V3
            | llama::arch::LlmArch::NOMIC_BERT
            | llama::arch::LlmArch::NOMIC_BERT_MOE
            | llama::arch::LlmArch::NEO_BERT
            | llama::arch::LlmArch::MODERN_BERT => {
                let pool_arg = args.pooling.as_deref().and_then(|v| match v {
                    "none" => Some(llama::hparams::LlamaPoolingType::NONE),
                    "mean" => Some(llama::hparams::LlamaPoolingType::MEAN),
                    "cls" => Some(llama::hparams::LlamaPoolingType::CLS),
                    "last" => Some(llama::hparams::LlamaPoolingType::LAST),
                    _ => {
                        eprintln!("invalid value for --pooling: {v}");
                        std::process::exit(1);
                    }
                });
                let pool = llama::context::resolve_pooling(
                    pool_arg.unwrap_or(llama::hparams::LlamaPoolingType::UNSPECIFIED),
                    model.hparams.pooling_type,
                );
                // eurobert ropes its Q/K (eurobert.cpp:65-75 — the same facts
                // the server's encoder branch assembles); the bert-variant
                // family ropes the same way (bert_variants_e2e.rs' encoder()
                // helper passes euro_rope for every variant, incl. the
                // neo/modern bert graphs' ggml_rope_ext calls)
                let euro = matches!(
                    model.arch,
                    llama::arch::LlmArch::EUROBERT
                        | llama::arch::LlmArch::LLAMA_EMBED
                        | llama::arch::LlmArch::JINA_BERT_V2
                        | llama::arch::LlmArch::JINA_BERT_V3
                        | llama::arch::LlmArch::NOMIC_BERT
                        | llama::arch::LlmArch::NOMIC_BERT_MOE
                        | llama::arch::LlmArch::NEO_BERT
                        | llama::arch::LlmArch::MODERN_BERT
                )
                .then(|| {
                    let rope = model.hparams.rope_runtime();
                    graph_arch::EurobertRope {
                        n_rot: model.hparams.n_rot(0) as i32,
                        rope_mode: model.hparams.rope_type as i32,
                        n_ctx_orig: rope.n_ctx_orig_yarn,
                        freq_base: model.hparams.rope_freq_base_train,
                        freq_scale: rope.freq_scale,
                        ext_factor: rope.ext_factor,
                        attn_factor: rope.attn_factor,
                        beta_fast: rope.beta_fast,
                        beta_slow: rope.beta_slow,
                    }
                });
                // llama-embed rides llama_encode, which FORCES cparams.causal_attn =
                // false on the encoder path (llama-context.cpp:1526-1529) —
                // the no-cache mask fills non-causally
                let causal = false;
                // gemma-embedding's symmetric-SWA facts (gemma-embedding.cpp:3-28);
                // gemma-embedding2 reads the same window pair through the same
                // plumbing (gemma-embedding2.cpp:4-6 sets swa_type = SYMMETRIC)
                let gemma_swa = matches!(
                    model.arch,
                    llama::arch::LlmArch::GEMMA_EMBEDDING
                        | llama::arch::LlmArch::GEMMA_EMBEDDING2
                )
                .then(|| {
                    let hp = &model.hparams;
                    let rope = hp.rope_runtime();
                    let gr = graph_arch::EurobertRope {
                        n_rot: hp.n_rot(0) as i32,
                        rope_mode: hp.rope_type as i32,
                        n_ctx_orig: rope.n_ctx_orig_yarn,
                        freq_base: hp.rope_freq_base_train,
                        freq_scale: rope.freq_scale,
                        ext_factor: rope.ext_factor,
                        attn_factor: rope.attn_factor,
                        beta_fast: rope.beta_fast,
                        beta_slow: rope.beta_slow,
                    };
                    (
                        graph_arch::GemmaEmbeddingSwa {
                            is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                            n_swa: hp.n_swa,
                            freq_base_swa: hp.rope_freq_base_train_swa,
                            freq_scale_swa: hp.rope_freq_scale_train_swa,
                        },
                        gr,
                        hp.f_attention_scale,
                    )
                });
                let params = graph_arch::EncoderParams {
                    n_head: model.hparams.n_head(0) as i64,
                    n_head_kv: model.hparams.n_head_kv(0) as i64,
                    n_embd_head: model.hparams.n_embd_head_k(0) as i64,
                    n_rel_attn_bkts: model.hparams.n_rel_attn_bkts,
                    f_norm_eps: model.hparams.f_norm_eps,
                    f_norm_rms_eps: model.hparams.f_norm_rms_eps,
                    pool,
                    euro_rope: euro,
                    gemma_swa,
                    causal,
                };
                let enc_w = match model.arch {
                    llama::arch::LlmArch::EUROBERT => {
                        llama::context::EncoderWeights::Eurobert(model.eurobert_weights())
                    }
                    llama::arch::LlmArch::T5ENCODER => {
                        llama::context::EncoderWeights::T5Encoder(model.t5_encoder_weights())
                    }
                    llama::arch::LlmArch::GEMMA_EMBEDDING => {
                        llama::context::EncoderWeights::GemmaEmbedding(model.gemma_embedding_weights())
                    }
                    // gemma-embedding2 (gemma-embedding2.cpp:3-19): the
                    // bundled params carry the per-layer-input facts + the
                    // SWA rope pair; the mask window rides gemma_swa above
                    // (the ge2 test assembly, tests/gemma_embedding2_e2e.rs)
                    llama::arch::LlmArch::GEMMA_EMBEDDING2 => {
                        let hp = &model.hparams;
                        let rope = hp.rope_runtime();
                        let p = graph_arch::GemmaEmbedding2Params {
                            n_head: hp.n_head(0) as i64,
                            n_head_kv: hp.n_head_kv(0) as i64,
                            n_embd_head: hp.n_embd_head_k(0) as i64,
                            norm_rms_eps: hp.f_norm_rms_eps,
                            n_embd_per_layer: hp.n_embd_per_layer as i64,
                            f_attention_scale: hp.f_attention_scale,
                            is_swa: (0..hp.n_layer() as usize)
                                .map(|il| hp.is_swa(il))
                                .collect(),
                            freq_base_swa: hp.rope_freq_base_train_swa,
                            freq_scale_swa: hp.rope_freq_scale_train_swa,
                            rope: graph_arch::EurobertRope {
                                n_rot: hp.n_rot(0) as i32,
                                // llama_model_rope_type(GEMMA_EMBEDDING2) =
                                // NEOX (llama-model.cpp:3184)
                                rope_mode: hp.rope_type as i32,
                                n_ctx_orig: rope.n_ctx_orig_yarn,
                                freq_base: hp.rope_freq_base_train,
                                freq_scale: rope.freq_scale,
                                ext_factor: rope.ext_factor,
                                attn_factor: rope.attn_factor,
                                beta_fast: rope.beta_fast,
                                beta_slow: rope.beta_slow,
                            },
                        };
                        llama::context::EncoderWeights::GemmaEmbedding2(
                            model.gemma_embedding2_weights(),
                            p,
                        )
                    }
                    llama::arch::LlmArch::LLAMA_EMBED => {
                        llama::context::EncoderWeights::LlamaEmbed(llama_embed_weights(&model))
                    }
                    // the bert-variant family (bert_variants_e2e.rs' encoder():
                    // the same BertVariantParams the tests assemble from the
                    // hparams — jina/nomic use the alibi/plain facts, the MoE
                    // variant adds its expert facts)
                    llama::arch::LlmArch::JINA_BERT_V2
                    | llama::arch::LlmArch::JINA_BERT_V3
                    | llama::arch::LlmArch::NOMIC_BERT => {
                        let vp = graph_arch::BertVariantParams {
                            max_alibi_bias: model.hparams.f_max_alibi_bias,
                            moe_every_n_layers: 0,
                            n_expert: 0,
                            n_expert_used: 0,
                            expert_weights_scale: 0.0,
                            n_ff: model.hparams.n_ff(0) as i64,
                        };
                        let variant = match model.arch {
                            llama::arch::LlmArch::JINA_BERT_V2 => graph_arch::BertVariant::JinaV2,
                            llama::arch::LlmArch::JINA_BERT_V3 => graph_arch::BertVariant::JinaV3,
                            _ => graph_arch::BertVariant::Nomic,
                        };
                        llama::context::EncoderWeights::BertVariant(
                            model.bert_variant_weights(variant),
                            vp,
                        )
                    }
                    llama::arch::LlmArch::NOMIC_BERT_MOE => {
                        let vp = graph_arch::BertVariantParams {
                            max_alibi_bias: model.hparams.f_max_alibi_bias,
                            moe_every_n_layers: model.hparams.moe_every_n_layers,
                            n_expert: model.hparams.n_expert as i64,
                            n_expert_used: model.hparams.n_expert_used(0) as i64,
                            expert_weights_scale: model.hparams.expert_weights_scale,
                            n_ff: model.hparams.n_ff(0) as i64,
                        };
                        llama::context::EncoderWeights::BertVariant(
                            model.bert_variant_weights(graph_arch::BertVariant::NomicMoe),
                            vp,
                        )
                    }
                    llama::arch::LlmArch::NEO_BERT => {
                        llama::context::EncoderWeights::NeoBert(model.neo_bert_weights())
                    }
                    // modern-bert's symmetric-SWA facts + the resolved FFN op
                    // (bert_variants_e2e.rs:778-789)
                    llama::arch::LlmArch::MODERN_BERT => {
                        let hp = &model.hparams;
                        let mp = graph_arch::ModernBertParams {
                            swa: graph_arch::ModernBertSwa {
                                is_swa: (0..hp.n_layer() as usize)
                                    .map(|il| hp.is_swa(il))
                                    .collect(),
                                n_swa: hp.n_swa,
                                freq_base: hp.rope_freq_base_train,
                                freq_scale: hp.rope_freq_scale_train,
                                freq_base_swa: hp.rope_freq_base_train_swa,
                                freq_scale_swa: hp.rope_freq_scale_train_swa,
                            },
                            ffn_op: hp.llm_ffn_op,
                        };
                        llama::context::EncoderWeights::ModernBert(model.modern_bert_weights(), mp)
                    }
                    _ => llama::context::EncoderWeights::Bert(model.bert_weights()),
                };
                let mut ectx = llama::context::EncoderContext::new(
                    model.ctx,
                    enc_w,
                    params,
                    args.n_threads,
                );
                let emb = match ectx.encode(&enc_ids) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("encode failed: {e}");
                        std::process::exit(1);
                    }
                };
                println!(
                    "embedding: {} x {} (pooling {:?}, {} tokens)",
                    emb.n_rows, emb.n_embd_out, pool, enc_ids.len()
                );
                for r in 0..emb.n_rows {
                    let row = &emb.values[r * emb.n_embd_out..(r + 1) * emb.n_embd_out];
                    let strs: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
                    println!("embed[{r}]: {}", strs.join(" "));
                }
                // the bit-exactness handle for the parity scripts — the LE
                // f32 bytes of the first row, hex
                let head: Vec<u8> = emb.values[..emb.n_embd_out]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                println!(
                    "embed-hex-le: {}",
                    head.iter().map(|b| format!("{b:02x}")).collect::<String>()
                );
                return;
            }
            other => {
                eprintln!(
                    "error: --embedding is wired for encoder models only in this port \
                     (arch '{}'); the pinned reference's cli rejects the flag for every \
                     arch (arg.cpp:824)",
                    other.name()
                );
                std::process::exit(1);
            }
        }
    }

    // wire graph weights from the model layers (the full arch dispatch lives
    // in [`forward_weights`] below — the draft model of the speculative path
    // reuses it)
    let (weights, attn) = match forward_weights(&model, args.flash_attn.on(), args.n_ctx) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "attn: heads={}/{} head_k={} n_rot={} rope_mode={} base={} scale={} eps={} flash_attn={:?}{}",
        attn.n_head, attn.n_head_kv, attn.n_embd_head_k, attn.n_rot, attn.rope_mode, attn.freq_base, attn.freq_scale, attn.norm_eps, args.flash_attn,
        if args.flash_attn == FlashAttn::Auto { " (auto -> off)" } else { "" }
    );
    // n_batch: max tokens per decode call; the reference default ubatch is 512
    // (n_ubatch). The old hardcoded 64 panicked on prompts > 64 tokens.
    let n_batch = 512.min(args.n_ctx as usize);

    // ---- LoRA adapters (common/common.cpp:1342-1358 load, :1507-1508 apply) ----
    // Loaded into the model's own Context *before* the DecodeContext, so their
    // tensors sit below the graph watermark and survive `reset_graph_to`
    // (the port's replacement for the C's heap-owned adapter tensors).
    let mut gctx = model.ctx;
    let mut loras: Vec<(std::rc::Rc<llama::adapter::AdapterLora>, f32)> = Vec::new();
    if !args.lora_adapters.is_empty() {
        let model_dims: std::collections::HashMap<String, [i64; 4]> = model
            .tensors
            .iter()
            .map(|(name, &id)| (name.clone(), *gctx.ne(id)))
            .collect();
        for (path, scale) in &args.lora_adapters {
            match llama::adapter::load_adapter_lora(
                &mut gctx,
                model.arch,
                &|name| model_dims.get(name).copied(),
                path,
            ) {
                Ok(adapter) => loras.push((adapter, *scale)),
                Err(_) => {
                    // common/common.cpp:1346-1349 + :1440-1442
                    eprintln!("failed to load lora adapter '{path}'");
                    std::process::exit(1);
                }
            }
        }
    }

    // arch batch 8: the iswa split for SWA models — llama-model.cpp:2687-2690
    // (`llama_kv_cache_iswa` when `swa_type != NONE && is_swa_any()`), the
    // same selection llama-server performs. Without it the cohere2moe /
    // exaone-moe / gemma4 / olmo2 SWA layers would run on one unified cache
    // and the window would never bind. The dsa (deepseek32) and dsv4
    // (deepseek4) cache forms take precedence in the C's switch — those archs
    // construct their own pair inside `DecodeContext::new_impl` and must stay
    // on the `new_with` path (their hparams DO carry swa keys).
    let swa_spec = {
        let hp = &model.hparams;
        let own_cache = matches!(
            weights,
            llama::context::ForwardWeights::Deepseek32(..)
                | llama::context::ForwardWeights::Deepseek4(..)
                // glm-dsa carries its own dsa-lid cache pair (batch 13)
                | llama::context::ForwardWeights::GlmDsa(..)
        );
        (!own_cache && hp.swa_type != llama::hparams::LlamaSwaType::NONE && hp.is_swa_any())
            .then(|| llama::kv_cache::SwaCacheSpec::from_hparams(hp))
    };
    let mut dctx = match swa_spec {
        Some(spec) => DecodeContext::new_with_swa(
            gctx,
            weights,
            attn,
            args.n_ctx,
            args.n_threads,
            n_batch,
            spec,
        ),
        None => DecodeContext::new_with(gctx, weights, attn, args.n_ctx, args.n_threads, n_batch),
    };
    // `cparams.n_rs_seq = params.speculative.need_n_rs_seq()` — arm the
    // recurrent-state rollback ring when a draft that replays rejected tokens
    // (MTP/eagle3/dflash/dspark) runs on a hybrid model, or the verify batch
    // pollutes the GDN state past rejected drafts (common.h:396-404 +
    // common.cpp:1635). The MTP draft context below takes it too: the port's
    // draft is a full-trunk second context over the same file, whose GDN
    // state the rejected-draft replay also dirties (proven by the 27B real
    // e2e: target-only arming still flips at token 8, both-armed matches the
    // reference 16/16 — see tests/mtp_real_spec_e2e.rs)
    let n_rs_seq: u32 = if args
        .speculative
        .types
        .iter()
        .any(|t| {
            matches!(
                t,
                llama::speculative::CommonSpeculativeType::DraftMtp
                    | llama::speculative::CommonSpeculativeType::DraftEagle3
                    | llama::speculative::CommonSpeculativeType::DraftDflash
                    | llama::speculative::CommonSpeculativeType::DraftDspark
            )
        }) {
        args.speculative.draft.n_max.max(0) as u32
    } else {
        0
    };
    if n_rs_seq > 0 {
        dctx = dctx.with_rs_rollback(n_rs_seq);
    }

    // GPU tasks ②/③: `-ngl N > 0` (or `--device`) drives the decode graphs on
    // a dlopen'ed foreign ggml through the backend_emit translator; weights
    // and KV land per the reference's layer-split rule (llama-model.cpp:1521)
    let mut gpu_mode_ran = false;
    if args.n_gpu_layers > 0 || args.device.is_some() || args.foreign_cpu {
        let lib_dir = args.ggml_libs.clone().unwrap_or_else(|| {
            eprintln!(
                "gpu mode needs the foreign ggml build dir: pass --ggml-libs <dir> \
                 (the reference build's bin/, holding libggml-base.so and libggml-*.so)"
            );
            std::process::exit(1);
        });
        let mut cfg = ggml::backend_emit::EmitConfig::new(&lib_dir);
        cfg.n_gpu_layers = args.n_gpu_layers;
        cfg.device = args.device.clone();
        cfg.n_threads = args.n_threads;
        // `--device cpu` (or -ngl 0 with a --ggml-libs dir) = the foreign CPU
        // backend: the reference's libggml-cpu via DL, no device buffer
        let device = match cfg.device.as_deref() {
            Some(d) if d.eq_ignore_ascii_case("cpu") => None,
            d => d.map(|s| s.to_string()),
        };
        cfg.device = device;
        if let Err(e) = dctx.enable_gpu(cfg) {
            eprintln!("failed to enable gpu backends: {e}");
            std::process::exit(1);
        }
        gpu_mode_ran = true;
    }

    // `common_set_adapter_lora` (common/common.cpp:1667-1676) →
    // `llama_set_adapters_lora` (llama-context.cpp:4069); zero scales are
    // dropped, so `--lora-scaled f:0` is bit-for-bit the base model.
    if let Err(e) = llama::adapter::set_adapters_lora(&dctx.gctx, &loras) {
        eprintln!("failed to apply lora adapter: {e}");
        std::process::exit(1);
    }

    // --perplexity FILE mode: ppl over first min(len, n_ctx) tokens
    // (llama-perplexity chunk-1 semantics: mean nll over tokens 1..n)
    if let Some(f) = &args.perplexity {
        let text = std::fs::read_to_string(f).unwrap();
        let toks = vocab.tokenize(&text, true, true);
        let n = toks.len().min(args.n_ctx as usize);
        println!("tokens: {} (scoring first {})", toks.len(), n);
        let nv = dctx.n_vocab();
        let mut nll = 0f64;
        let mut scored = 0usize;
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + 64).min(n);
            let ids: Vec<i32> = toks[pos..end].to_vec();
            let positions: Vec<i32> = (pos as i32..end as i32).collect();
            let all = dctx.decode_all(&ids, &positions).expect("decode");
            // logits at position j predict toks[j+1]
            for j in 0..(end - pos) {
                let global = pos + j;
                if global + 1 < n {
                    let row = &all[j * nv..(j + 1) * nv];
                    let p = softmax_logprob(row, toks[global + 1] as usize);
                    nll += p;
                    scored += 1;
                }
            }
            pos = end;
        }
        println!(
            "ppl: {:.4} ({} scored)",
            (-nll / scored as f64).exp(),
            scored
        );
        return;
    }

    // ---- sampler (common_sampler_init: logit-bias, then the params.samplers
    // chain — penalties, dry, top-n-sigma, top-k, typical, top-p, min-p, xtc,
    // temp-ext by default — then dist / adaptive-p) ------------------------
    // temp <= 0 keeps the greedy semantics of `llama_sampler_temp_impl`
    // (strict `>`, first max wins — NOT the divergence documented in
    // sampling::GreedySampler, which takes the last max).
    let params = SamplingParams {
        seed: args.seed,
        temp: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        n_prev: args.repeat_last_n,
        penalty_last_n: args.repeat_last_n,
        penalty_repeat: args.repeat_penalty,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers: args.dry_sequence_breakers.clone(),
        adaptive_target: args.adaptive_target,
        adaptive_decay: args.adaptive_decay,
        samplers: args.samplers.clone(),
        ..Default::default()
    };
    let n_vocab = dctx.n_vocab() as i32;
    // the vocab lowers the DRY string breakers into token sequences
    // (llama_sampler_init_dry, llama-sampler.cpp:3640)
    let mut sampler = SamplingContext::new_with_vocab(n_vocab, Some(&vocab), params);
    // `common_sampler_init` (common/sampling.cpp:212-275): the grammar sampler is
    // built outside the chain and applied via common_sampler_sample. The text
    // comes from whichever of --grammar/--grammar-file/-j/-jf came last
    // (arg.cpp:2265-2300 all assign the same `params.sampling.grammar`).
    let mut grammar: Option<GrammarSampler> = match &args.grammar {
        Some(gbnf) => match llama::sampling::init_grammar(&vocab, gbnf) {
            Ok(g) => {
                eprintln!(
                    "grammar: source={} root={} ({} rules, {} initial stacks)",
                    args.grammar_source.unwrap_or("?"),
                    g.grammar_root,
                    g.grammar.rules.len(),
                    g.grammar.stacks.len()
                );
                Some(g)
            }
            Err(e) => {
                // common/sampling.cpp:274-276 `throw std::runtime_error("failed to parse grammar")`
                eprintln!("failed to parse grammar: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };
    eprintln!(
        "sampling: temp={} top_k={} top_p={} min_p={} repeat_penalty={} last_n={} seed={}{}",
        args.temp,
        args.top_k,
        args.top_p,
        args.min_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.seed,
        if args.temp <= 0.0 { " (greedy)" } else { "" }
    );

    // ---- the diffusion driver (examples/diffusion/diffusion-cli.cpp) ----
    // llada / dream / rnd1 graphs have no causal decode loop — they run the
    // masked-iterative diffusion_generate instead
    if llama::arch::llm_arch_is_diffusion(model.arch) {
        diffusion::run(&args, &vocab, &mut dctx, &gguf);
        return;
    }

    // ---- the interactive REPL (tools/completion/completion.cpp) ----
    // The dispatch mirrors the reference's conversation/interactive
    // resolution: explicit -i/-if/-cnv always engage; AUTO+chat-template
    // engages only for -st (a plain `-p` run keeps the port's documented
    // direct single-shot path — the reference equivalent of `-no-cnv`)
    {
        let chat_template_override: String = match &args.chat_template {
            Some(s) => s.clone(),
            None => model_chat_template.clone(),
        };
        if use_interactive {
            if common_speculative_n_max_params(&args.speculative) > 0 {
                eprintln!("speculative decoding is not wired for the interactive mode");
                std::process::exit(1);
            }
            let templates_init = llama::chat_tools::ChatTemplatesInit {
                chat_template_override,
                chat_template_tool_use: gguf
                    .get_str("tokenizer.chat_template.tool_use")
                    .unwrap_or("")
                    .to_string(),
                bos_token: String::from_utf8_lossy(
                    &vocab.token_to_piece_special(vocab.token_bos(), true),
                )
                .into_owned(),
                eos_token: String::from_utf8_lossy(
                    &vocab.token_to_piece_special(vocab.token_eos(), true),
                )
                .into_owned(),
                add_bos: vocab.add_bos,
                add_eos: vocab.add_eos,
            };
            let templates = if has_chat_template {
                match llama::chat_tools::ChatTemplates::init(&templates_init) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        eprintln!("warning: chat template parsing error: {e}");
                        None
                    }
                }
            } else {
                None
            };
            let code = interactive::run(
                &args,
                &vocab,
                &mut dctx,
                &mut sampler,
                &mut grammar,
                templates,
                has_chat_template,
            );
            std::process::exit(code);
        }
    }

    // ---- speculative decoding (examples/speculative-simple/speculative-
    // simple.cpp:17-377): active when a draft type was requested. The pinned
    // revision's own tools/cli/main.cpp parses --spec-* but never consumes
    // params.speculative — the port wires the example's driver here so `-md
    // ... --spec-type draft-simple` actually speculates. ----
    if common_speculative_n_max_params(&args.speculative) > 0 {
        if args.grammar.is_some() {
            eprintln!("speculative decoding does not support --grammar/-j constraints");
            std::process::exit(1);
        }
        if args.perplexity.is_some() {
            eprintln!("--perplexity cannot be combined with speculative decoding");
            std::process::exit(1);
        }
        run_speculative(&args, &vocab, &mut dctx, &mut sampler, &tokens, chat_eot);
        return;
    }

    // prefill
    let ids: Vec<i32> = tokens.clone();
    let pos: Vec<i32> = (0..ids.len() as i32).collect();
    let t1 = std::time::Instant::now();
    let mut logits = dctx.decode(&ids, &pos).expect("prefill").to_vec();
    println!(
        " [prompt: {} tokens in {:?} = {:.1} t/s]",
        ids.len(),
        t1.elapsed(),
        ids.len() as f32 / t1.elapsed().as_secs_f32()
    );

    // common_sampler_accept(..., is_generated = false) over the prompt tokens
    // (drives the penalty/prev ring buffer)
    for &t in &ids {
        sampler.accept(t);
    }

    // generate
    let mut out_tokens: Vec<i32> = Vec::with_capacity(args.n_predict);
    let mut out_text = String::new();
    let mut hit_eog = false;
    let mut hit_eot = false;
    let dump_top3 = |step: usize, lg: &[f32]| {
        if std::env::var("LLAMA_RUST_DEBUG").is_ok() {
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]));
            // strict `>` first-max argmax (the temp<=0 semantics)
            let mut gm = 0usize;
            for (i, &v) in lg.iter().enumerate() {
                if v > lg[gm] {
                    gm = i;
                }
            }
            // full-row log_softmax (same max-stable f64 LSE as softmax_logprob)
            // so the printed top-5 is directly comparable to the reference
            // server's `logprobs` (llama-server top_logprobs are normalized)
            let max = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse: f64 = lg
                .iter()
                .map(|&v| (v - max) as f64)
                .map(f64::exp)
                .sum::<f64>()
                .ln()
                + max as f64;
            println!(
                "step {step}: top5 {:?} greedy={} eos={} eot={}",
                idx[..5]
                    .iter()
                    .map(|&i| (i as i32, lg[i], (lg[i] as f64 - lse)))
                    .collect::<Vec<_>>(),
                gm,
                lg.get(151643).copied().unwrap_or(f32::NAN),
                lg.get(151645).copied().unwrap_or(f32::NAN),
            );
        }
    };
    let mut cur_pos = ids.len() as i32;
    let t2 = std::time::Instant::now();
    for step in 0..args.n_predict {
        dump_top3(step, &logits);
        // common_sampler_sample(..., grammar_first = false): sample, check the
        // grammar, resample with the grammar applied if the token was rejected
        let tok = match grammar.as_mut() {
            Some(g) => match sampler.sample_with_grammar(&logits, g) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("\n[grammar] {e}");
                    break;
                }
            },
            None => sampler.sample(&logits),
        };
        out_tokens.push(tok);

        // end-of-generation token (llama_vocab_is_eog) — reference stops and
        // prints " [end of text]" (the EOG piece itself renders empty with
        // token_to_piece(special=false))
        if !args.ignore_eos && vocab.is_eog(tok) {
            hit_eog = true;
            break;
        }

        let piece = vocab.token_to_piece(tok);
        out_text.push_str(piece);
        if !args.chat {
            print!("{piece}");
            std::io::stdout().flush().ok();
        }

        // chat mode: stop at the template's end-of-turn marker
        if let Some(eot) = chat_eot {
            if !eot.is_empty() && out_text.contains(eot) {
                hit_eot = true;
                break;
            }
        }

        logits = dctx.decode(&[tok], &[cur_pos]).expect("decode").to_vec();
        cur_pos += 1;
        if cur_pos as u32 >= args.n_ctx {
            eprintln!("\n[context full]");
            break;
        }
    }
    let dt = t2.elapsed().as_secs_f32();

    if args.chat {
        // truncate the reply at the end-of-turn marker
        let reply = match chat_eot {
            Some(eot) if !eot.is_empty() => out_text.split(eot).next().unwrap_or("").to_string(),
            _ => out_text.clone(),
        };
        println!("{reply}");
        if hit_eog || hit_eot {
            eprintln!(" [end of text]");
        }
    } else if hit_eog {
        eprintln!(" [end of text]");
    }

    println!(
        "\n [gen: {} tokens in {:.1}s = {:.1} t/s]",
        out_tokens.len(),
        dt,
        out_tokens.len() as f32 / dt
    );
    println!("gen tokens: {:?}", out_tokens);

    // [TAG_EMIT_EXIT] the foreign-backend epilogue: a dlopen'ed libgomp (the
    // foreign CPU side of the scheduler) leaves worker threads whose dynamic
    // TLS glibc's exit handlers tear down before the threads stop — a
    // SIGSEGV at process exit (glibc "dlopen'd module with TLS" class, hit
    // with -t >= 2). _exit skips those handlers; everything is printed.
    if gpu_mode_ran {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        ggml::sysffi::exit_now(0);
    }
}

// ---------------------------------------------------------------------------
// speculative decoding — the driver of examples/speculative-simple
// (speculative-simple.cpp:17-377) reached from the CLI's --spec-* surface
// ---------------------------------------------------------------------------

/// Load the draft model (`common_speculative_init_from_params`,
/// speculative.cpp:2523-2604), init the speculator (`common_speculative_init`,
/// :2619) and run [`speculative_simple_generate`] (the example's decode loop).
fn run_speculative(
    args: &Args,
    vocab_tgt: &llama::vocab::Vocab,
    tgt: &mut DecodeContext,
    smpl: &mut SamplingContext,
    prompt: &[i32],
    chat_eot: Option<&'static str>,
) {
    // the SPC_DBG traces (`-v` on the reference server) — LLAMA_SPEC_VERBOSE=1
    // enables the port's mirror (speculative.rs's spec_dbg), whose candidate
    // lines carry the same seq/pos/token ids as the C's :1676-1680
    if std::env::var("LLAMA_SPEC_VERBOSE").is_ok() {
        llama::speculative::spec_set_verbose(true);
    }

    // `common_speculative_init_from_params`: `LOG_INF("%s: loading draft model
    // '%s'\n")` + `LOG_ERR("%s: failed to load draft model, '%s'\n")`
    // (speculative.cpp:2536-2544). The model loads only when `has_dft()`
    // (:2533-2535); without a draft path the speculator itself reports
    // "draft-simple requires a draft context" below.
    //
    // `--spec-type mtp` takes the :2577-2589 branch instead: a second
    // context over the *target* model with `ctx_type = MTP`
    // (the has_draft arm at :2557-2576 also loads `params.model.path` — not
    // the `-md` path — so MTP always drafts from the target file; the port
    // reloads it, the C shares the loaded llama_model between contexts).
    let path = args.speculative.draft.model_path.clone();
    let has_dft = args.speculative.has_dft();
    let spec_mtp = args
        .speculative
        .types
        .contains(&llama::speculative::CommonSpeculativeType::DraftMtp);
    // `--spec-type draft-eagle3` — the has_draft arm over the eagle head file
    // (speculative.cpp:2553-2576): the draft context is the eagle3 head
    // context (encoder + one-layer decoder) with `cparams.ctx_other = ctx_tgt`
    let spec_eagle = args
        .speculative
        .types
        .contains(&llama::speculative::CommonSpeculativeType::DraftEagle3);
    // `--spec-type draft-dflash / draft-dspark` — the has_draft arm over the
    // dflash draft file (speculative.cpp:2553-2576): the draft context is the
    // dflash dual-mode decoder (KV injection + noise block) with
    // `cparams.ctx_other = ctx_tgt`
    let spec_dflash = args
        .speculative
        .types
        .contains(&llama::speculative::CommonSpeculativeType::DraftDflash)
        || args
            .speculative
            .types
            .contains(&llama::speculative::CommonSpeculativeType::DraftDspark);
    // the gemma4-assistant head — the `-md` + `--spec-type draft-mtp` pair
    // whose draft file is a gemma4-assistant GGUF: the head ATTACHES to the
    // target context (the port's `cparams.ctx_other == ctx_tgt`,
    // llama-context.cpp:147-153) and drafts over the shared target KV
    // (`is_mem_shared`, speculative.cpp:1423) — no draft context exists
    let mut gemma4_shared = false;
    let (ctx_dft, vocab_dft, n_layer_nextn) = if spec_mtp
        && has_dft
        && ggml::Gguf::open(&path)
            .ok()
            .and_then(|g| g.find_key("general.architecture").and_then(|v| v.as_str().map(str::to_string)))
            .as_deref()
            == Some("gemma4-assistant")
    {
        eprintln!("attaching gemma4-assistant draft head '{path}' to the target context");
        let gguf_dft = match ggml::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let vocab_dft = match llama::vocab::Vocab::load(&gguf_dft) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let mmap_dft = {
            let f = std::fs::File::open(&path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        // the share map's target geometry (llama-model.cpp:2698-2703): the
        // assistant layer `il` reads target layer `n_layer - 1` (non-SWA) /
        // `n_layer - 2` (SWA) — (head dim, kv heads) of those two layers
        let (tgt_full, tgt_swa) = match &tgt.weights {
            llama::context::ForwardWeights::Gemma4(_, p) => {
                let n = p.is_swa.len() - 1;
                (
                    (p.n_embd_head_k[n] as i64, p.n_head_kv[n] as i64),
                    (p.n_embd_head_k[n - 1] as i64, p.n_head_kv[n - 1] as i64),
                )
            }
            _ => {
                eprintln!("gemma4-assistant requires a gemma4 target model");
                std::process::exit(1);
            }
        };
        if let Err(e) = tgt.attach_gemma4_assistant(
            &gguf_dft,
            mmap_dft,
            args.flash_attn.on(),
            tgt_full,
            tgt_swa,
        ) {
            eprintln!("failed to attach the gemma4-assistant head: {e}");
            std::process::exit(1);
        }
        // the shared width must match the target's h_nextn tap
        // (`n_embd == llama_model_n_embd_out(model_tgt)`, speculative.cpp:1374-1376)
        let head = tgt.gemma4_assistant.as_ref().unwrap();
        if head.params.n_embd_backbone as usize != tgt.n_embd_out() {
            eprintln!(
                "draft-mtp: MTP input row width must match the target h_nextn width ({} != {})",
                head.params.n_embd_backbone,
                tgt.n_embd_out()
            );
            std::process::exit(1);
        }
        eprintln!(
            "gemma4-assistant: n_embd = {}, backbone = {}, layers = {} (shared target KV)",
            head.params.n_embd,
            head.params.n_embd_backbone,
            head.params.is_swa.len()
        );
        gemma4_shared = true;
        (None, Some(vocab_dft), 0)
    } else if spec_dflash {
        eprintln!("loading draft model '{path}'");
        let gguf_dft = match ggml::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let vocab_dft = match llama::vocab::Vocab::load(&gguf_dft) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        // the ctx_other tensors (the draft's optional tok_embd / output may
        // live in the target file) + the target's hidden size — dflash.rs
        let gguf_tgt = match ggml::Gguf::open(&args.model_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to open the target model for the dflash draft: {e}");
                std::process::exit(1);
            }
        };
        let mmap_dft = {
            let f = std::fs::File::open(&path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let mmap_tgt = {
            let f = std::fs::File::open(&args.model_path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let draft = match llama::dflash::load_dflash_draft(
            &gguf_dft,
            mmap_dft,
            &gguf_tgt,
            mmap_tgt,
            vocab_dft.n_tokens() as i64,
            args.flash_attn.on(),
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        eprintln!(
            "draft: arch = dflash, extract_layers = {:?}, n_embd = {}, block_size = {}, \
             dspark = {}",
            draft.params.target_layer_ids,
            draft.params.n_embd,
            draft.params.block_size,
            draft.weights.dspark_markov_w1.is_some()
        );
        // `cparams.n_ctx = llama_n_ctx(ctx_tgt)` (speculative.cpp:2550)
        let n_ctx_tgt = tgt.n_ctx();
        let n_batch_dft = 512.min(n_ctx_tgt as usize);
        let stub = llama::dflash::dflash_trunk_stub(&draft.weights);
        let ctx_dft = DecodeContext::new_dflash(
            draft.ctx,
            llama::context::ForwardWeights::Qwen2(stub),
            (draft.weights, draft.params),
            vocab_dft.n_tokens() as usize,
            n_ctx_tgt,
            args.n_threads,
            n_batch_dft,
        );
        (Some(ctx_dft), Some(vocab_dft), 0)
    } else if spec_eagle {
        eprintln!("loading draft model '{path}'");
        let gguf_dft = match ggml::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let vocab_dft = match llama::vocab::Vocab::load(&gguf_dft) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        // the ctx_other tensors (the head's optional tok_embd/output may live
        // in the target file) — the port materializes them from the target's
        // mmap, see eagle.rs
        let gguf_tgt = match ggml::Gguf::open(&args.model_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to open the target model for the eagle head: {e}");
                std::process::exit(1);
            }
        };
        let mmap_dft = {
            let f = std::fs::File::open(&path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let mmap_tgt = {
            let f = std::fs::File::open(&args.model_path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let head = match llama::eagle::load_eagle3_head(
            &gguf_dft,
            mmap_dft,
            &gguf_tgt,
            mmap_tgt,
            vocab_dft.n_tokens() as i64,
            args.flash_attn.on(),
        ) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        eprintln!(
            "draft: arch = eagle3, extract_layers = {:?}, n_embd = {}, target_hidden_size = {}",
            head.params.target_layer_ids, head.params.n_embd, head.params.n_embd_tgt
        );
        // `cparams.n_ctx = llama_n_ctx(ctx_tgt)` (speculative.cpp:2550)
        let n_ctx_tgt = tgt.n_ctx();
        let n_batch_dft = 512.min(n_ctx_tgt as usize);
        let stub = llama::eagle::eagle_trunk_stub(&head.weights);
        let ctx_dft = DecodeContext::new_eagle3(
            head.ctx,
            llama::context::ForwardWeights::Qwen2(stub),
            (head.weights, head.params),
            vocab_dft.n_tokens() as usize,
            n_ctx_tgt,
            args.n_threads,
            n_batch_dft,
        );
        (Some(ctx_dft), Some(vocab_dft), 0)
    } else if spec_mtp {
        let model_path = args.model_path.clone();
        eprintln!("creating MTP draft context against the target model '{model_path}'");
        let gguf_dft = match ggml::Gguf::open(&model_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to create MTP context: {e}");
                std::process::exit(1);
            }
        };
        let mmap_dft = {
            let f = std::fs::File::open(&model_path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let model_dft = match load_model(&gguf_dft, mmap_dft) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("failed to create MTP context: {e}");
                std::process::exit(1);
            }
        };
        let n_layer_nextn = model_dft.hparams.n_layer_nextn;
        let n_ctx_tgt = tgt.n_ctx();
        let (weights_dft, attn_dft) =
            match forward_weights(&model_dft, args.flash_attn.on(), n_ctx_tgt) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };
        eprintln!(
            "mtp: arch = {}, n_layer = {}, n_layer_nextn = {}",
            model_dft.arch.name(),
            model_dft.hparams.n_layer(),
            n_layer_nextn
        );
        // the graph_mtp dispatch — the deepseek family + glm-dsa (batch 13)
        // and the nine GLM4-style heads (MTP batch 18): the arch params ride
        // the trunk bundle (cloned), the MTP layer's own facts come from
        // hparams at il = n_layer
        let facts = mtp_head_facts(&model_dft.hparams);
        let mtp = match (&weights_dft, model_dft.arch) {
            (llama::context::ForwardWeights::Deepseek2(_, p), _) => {
                llama::context::MtpForward::Deepseek2(deepseek2_mtp_weights(&model_dft), *p)
            }
            (llama::context::ForwardWeights::Deepseek32(_, p), _) => {
                llama::context::MtpForward::Deepseek32(deepseek2_mtp_weights(&model_dft), *p)
            }
            (llama::context::ForwardWeights::Deepseek4(_, p), _) => {
                llama::context::MtpForward::Deepseek4(deepseek4_mtp_weights(&model_dft), p.clone())
            }
            // arch batch 13 — glm-dsa's graph_mtp (glm-dsa.cpp:539-769)
            (llama::context::ForwardWeights::GlmDsa(_, p), _) => {
                llama::context::MtpForward::GlmDsa(glm_dsa_mtp_weights(&model_dft), p.clone())
            }
            // MTP batch 18 — the nine GLM4-style heads
            (llama::context::ForwardWeights::Qwen35(_, p), _) => {
                llama::context::MtpForward::Qwen35(qwen35_mtp_weights(&model_dft), p.clone(), facts)
            }
            (llama::context::ForwardWeights::Qwen35Moe(_, p), _) => {
                llama::context::MtpForward::Qwen35Moe(
                    qwen35moe_mtp_weights(&model_dft),
                    p.clone(),
                    facts,
                )
            }
            (llama::context::ForwardWeights::Qwen3Next(_, p), _) => {
                llama::context::MtpForward::Qwen3Next(
                    qwen3next_mtp_weights(&model_dft),
                    p.clone(),
                    facts,
                )
            }
            (llama::context::ForwardWeights::Glm4Moe(_, p), _) => {
                llama::context::MtpForward::Glm4Moe(
                    glm4_moe_mtp_weights(&model_dft),
                    p.clone(),
                    facts,
                )
            }
            (llama::context::ForwardWeights::Cohere2Moe(_, p), _) => {
                llama::context::MtpForward::Cohere2Moe(
                    cohere2moe_mtp_weights(&model_dft),
                    p.clone(),
                    facts,
                )
            }
            (llama::context::ForwardWeights::BailingMoe3(_, p), _) => {
                // the MTP builder reads the clamp vectors' `.last()` — the
                // MTP layer's own entry (index n_layer), so the clone is
                // extended past the trunk slice the trunk params carry
                let il = model_dft.hparams.n_layer() as usize;
                let mut p = p.clone();
                p.swiglu_clamp_exp = model_dft.hparams.swiglu_clamp_exp[..=il].to_vec();
                p.swiglu_clamp_shexp = model_dft.hparams.swiglu_clamp_shexp[..=il].to_vec();
                llama::context::MtpForward::BailingMoe3(
                    bailingmoe3_mtp_weights(&model_dft),
                    p,
                    facts,
                )
            }
            (llama::context::ForwardWeights::HyV3(_, p), _) => {
                llama::context::MtpForward::HyV3(hy_v3_mtp_weights(&model_dft), p.clone(), facts)
            }
            (llama::context::ForwardWeights::Mimo2(_, p), _) => {
                llama::context::MtpForward::Mimo2(mimo2_mtp_weights(&model_dft), p.clone(), facts)
            }
            (llama::context::ForwardWeights::Step35(_, p), _) => {
                llama::context::MtpForward::Step35(step35_mtp_weights(&model_dft), p.clone(), facts)
            }
            // b9acf138a — glm5-next's graph_mtp (the DSA+kpool NextN block);
            // the MTP layer's is_recr slot is absent from the trunk-length
            // vector, so the layer carries no hc mixers and no KDA set (the
            // nextn block is always a DSA layer, glm5-next.cpp:1059)
            (llama::context::ForwardWeights::Glm5Next(_, p), _) => {
                llama::context::MtpForward::Glm5Next(glm5_next_mtp_weights(&model_dft), p.clone())
            }
            // batch 42f — qwen4exp's graph_mtp (the hc-wide eh_proj + one
            // QSA/dense attention round + the nextn hc head mixer); the
            // params clone extends the per-layer vectors past n_layer with
            // the MTP layer's own facts (the bailingmoe3 precedent)
            (llama::context::ForwardWeights::Qwen4Exp(_, p), _) => {
                let il = model_dft.hparams.n_layer() as usize;
                let hp = &model_dft.hparams;
                let mut p = p.clone();
                p.n_head.push(hp.n_head(il));
                p.n_head_kv.push(hp.n_head_kv(il));
                p.n_embd_head_k.push(hp.n_embd_head_k(il));
                p.n_embd_head_v.push(hp.n_embd_head_v(il));
                p.n_rot.push(hp.n_rot(il));
                p.is_recr.push(hp.is_recr(il));
                p.is_ple.push(hp.is_ple(il));
                // the one shared compress ratio (qwen4exp.cpp:64-72) — the
                // MTP layer's own entry when the array covers n_layer_all
                p.compress_ratios.push(
                    hp.dsv4_compress_ratios
                        .get(il)
                        .copied()
                        .unwrap_or(if hp.indexer_kpool > 0 {
                            hp.indexer_kpool
                        } else {
                            0
                        }),
                );
                llama::context::MtpForward::Qwen4Exp(qwen4exp_mtp_weights(&model_dft), p)
            }
            (_, arch) => {
                eprintln!(
                    "draft-mtp: arch {} has no ported MTP graph (see PARITY.md)",
                    arch.name()
                );
                std::process::exit(1);
            }
        };
        let n_batch_dft = 512.min(n_ctx_tgt as usize);
        // the target's `need_n_rs_seq()` value (common.h:396-404) — the
        // port's MTP draft is a full-trunk context whose GDN state the
        // rejected-draft replay also dirties; armed alongside the target
        // (proven by the 27B real e2e — tests/mtp_real_spec_e2e.rs)
        let n_rs_seq_dft: u32 = args
            .speculative
            .types
            .iter()
            .any(|t| {
                matches!(
                    t,
                    llama::speculative::CommonSpeculativeType::DraftMtp
                        | llama::speculative::CommonSpeculativeType::DraftEagle3
                        | llama::speculative::CommonSpeculativeType::DraftDflash
                        | llama::speculative::CommonSpeculativeType::DraftDspark
                )
            })
            .then(|| args.speculative.draft.n_max.max(0) as u32)
            .unwrap_or(0);
        let ctx_dft = DecodeContext::new_mtp(
            model_dft.ctx,
            weights_dft,
            mtp,
            attn_dft,
            n_ctx_tgt,
            args.n_threads,
            n_batch_dft,
        )
        .with_rs_rollback(n_rs_seq_dft);
        // the draft vocab is the target's own (same file)
        (Some(ctx_dft), None, n_layer_nextn)
    } else if has_dft {
        eprintln!("loading draft model '{path}'");
        let gguf_dft = match ggml::Gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let vocab_dft = match llama::vocab::Vocab::load(&gguf_dft) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        let mmap_dft = {
            let f = std::fs::File::open(&path).unwrap();
            // SAFETY: read-only use of a model file
            Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
        };
        let model_dft = match load_model(&gguf_dft, mmap_dft) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("failed to load draft model, '{path}': {e}");
                std::process::exit(1);
            }
        };
        // the draft context holds as many tokens per sequence as the target
        // context (`cparams.n_ctx = llama_n_ctx(ctx_tgt)`, speculative.cpp:2550)
        let n_ctx_tgt = tgt.n_ctx();
        let (weights_dft, attn_dft) =
            match forward_weights(&model_dft, args.flash_attn.on(), n_ctx_tgt) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };
        eprintln!(
            "draft: arch = {}, n_layer = {}, vocab = {}",
            model_dft.arch.name(),
            model_dft.hparams.n_layer(),
            vocab_dft.id_to_token.len()
        );

        let n_batch_dft = 512.min(n_ctx_tgt as usize);
        let ctx_dft = DecodeContext::new_with(
            model_dft.ctx,
            weights_dft,
            attn_dft,
            n_ctx_tgt,
            args.n_threads,
            n_batch_dft,
        );
        (Some(ctx_dft), Some(vocab_dft), 0)
    } else {
        (None, None, 0)
    };

    // `common_speculative_init(params.speculative, 1)` — the vocab
    // compatibility error of `common_speculative_impl_draft_simple::new`
    // (speculative.cpp:238-245: "draft model vocab type must match target
    // model to use speculation") surfaces here as `Err`
    let mut spec = match common_speculative_init(
        &args.speculative,
        1,
        tgt,
        ctx_dft,
        vocab_tgt,
        // draft-mtp drafts from the target file: its vocab is the target's
        vocab_dft.as_ref().or(Some(vocab_tgt)),
        n_layer_nextn,
        gemma4_shared,
    ) {
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
        // `if (spec == nullptr) { LOG_ERR("%s", "failed to initialize
        // speculative decoding\n"); return 1; }` (speculative-simple.cpp:121-124)
        Ok(None) => {
            eprintln!("failed to initialize speculative decoding");
            std::process::exit(1);
        }
        Ok(Some(s)) => s,
    };

    let res = match speculative_simple_generate(
        tgt,
        &mut spec,
        smpl,
        vocab_tgt,
        prompt,
        args.n_predict as i32,
    ) {
        Ok(r) => r,
        Err(e) => {
            // "the prompt exceeds the context size / batch size"
            // (speculative-simple.cpp:80-90)
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // process the accepted tokens (speculative-simple.cpp:305-322 — the C
    // prints each piece inside the loop; the port streams them here)
    let mut out_text = String::new();
    for &t in &res.tokens {
        out_text.push_str(vocab_tgt.token_to_piece(t));
    }
    if args.chat {
        // truncate the reply at the template's end-of-turn marker
        let reply = match chat_eot {
            Some(eot) if !eot.is_empty() => out_text.split(eot).next().unwrap_or("").to_string(),
            _ => out_text.clone(),
        };
        println!("{reply}");
    } else {
        print!("{out_text}");
        std::io::stdout().flush().ok();
    }

    // the example's epilogue (speculative-simple.cpp:350-362)
    let dt = res.t_us as f64 / 1e6;
    let tps = if dt > 0.0 {
        res.n_predict as f64 / dt
    } else {
        0.0
    };
    println!(
        "\n [gen: {} tokens in {:.1}s = {:.1} t/s]",
        res.tokens.len(),
        dt,
        tps
    );
    println!("gen tokens: {:?}", res.tokens);
    println!(
        "spec: decoded {:4} tokens in {dt:8.3} seconds, speed: {tps:8.3} t/s",
        res.n_predict
    );
    println!("spec: n_draft   = {}", args.speculative.draft.n_max);
    println!("spec: n_predict = {}", res.n_predict);
    println!("spec: n_drafted = {}", res.n_drafted);
    println!("spec: n_accept  = {}", res.n_accept);
    let accept_pct = if res.n_drafted > 0 {
        100.0 * res.n_accept as f64 / res.n_drafted as f64
    } else {
        0.0
    };
    println!("spec: accept    = {accept_pct:.3}%");
    // `common_speculative_print_stats(spec)` (:362)
    print!("{}", spec.print_stats());
}

// ---------------------------------------------------------------------------
// arch dispatch — one arm per builder in graph_arch.rs; the `other` arm names
// the arch and errors out (the reference aborts inside the graph build). Used
// for the target model *and* the speculative draft model.
// ---------------------------------------------------------------------------

fn forward_weights(
    model: &LlamaModel,
    use_flash_attn: bool,
    n_ctx: u32,
) -> Result<(llama::context::ForwardWeights, AttnParams), String> {
    let hp = &model.hparams;
    // layer-0 `AttnParams` (the hybrid archs re-derive it from their first
    // attention layer in the dispatch below — their layer 0 is recurrent)
    let mut attn = attn_params(hp, 0, use_flash_attn);
    let n_trunk = hp.n_layer() as usize;
    let weights = match model.arch {
        llama::arch::LlmArch::QWEN2 => {
            let layers: Vec<LayerWeights> = model
                .layers
                .iter()
                .map(|l| LayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    wq_b: l.wq_b,
                    wk_b: l.wk_b,
                    wv_b: l.wv_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                })
                .collect();
            llama::context::ForwardWeights::Qwen2(ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers,
            })
        }
        llama::arch::LlmArch::LLAMA => {
            use llama::graph_arch::LlamaModelWeights;
            let layers = model
                .layers
                .iter()
                .map(|l| llama::graph_arch::LlamaLayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    wq_b: l.wq_b,
                    wk_b: l.wk_b,
                    wv_b: l.wv_b,
                    wo_b: l.wo_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                    ffn_gate_b: l.ffn_gate_b,
                    ffn_down_b: l.ffn_down_b,
                    ffn_up_b: l.ffn_up_b,
                })
                .collect();
            llama::context::ForwardWeights::Llama(LlamaModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        llama::arch::LlmArch::PHI3 => {
            let layers = model
                .layers
                .iter()
                .map(|l| llama::graph_arch::Phi3LayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wqkv: l.wqkv.expect("wqkv"),
                    wqkv_b: l.wqkv_b,
                    wo: l.wo.expect("wo"),
                    wo_b: l.wo_b,
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                })
                .collect();
            llama::context::ForwardWeights::Phi3(llama::graph_arch::Phi3ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        llama::arch::LlmArch::GEMMA2 | llama::arch::LlmArch::GEMMA3 => {
            let gemma2 = model.arch == llama::arch::LlmArch::GEMMA2;
            let layers = model
                .layers
                .iter()
                .map(|l| llama::graph_arch::GemmaLayerWeights {
                    attn_norm: l.attn_norm.expect("attn_norm"),
                    wq: l.wq.expect("wq"),
                    wk: l.wk.expect("wk"),
                    wv: l.wv.expect("wv"),
                    wo: l.wo.expect("wo"),
                    attn_post_norm: l.attn_post_norm.expect("attn_post_norm"),
                    ffn_norm: l.ffn_norm.expect("ffn_norm"),
                    ffn_gate: l.ffn_gate.expect("ffn_gate"),
                    ffn_down: l.ffn_down.expect("ffn_down"),
                    ffn_up: l.ffn_up.expect("ffn_up"),
                    ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
                    attn_q_norm: l.attn_q_norm,
                    attn_k_norm: l.attn_k_norm,
                })
                .collect();
            let gp = llama::graph_arch::GemmaParams {
                attn,
                attention_scale: 1.0 / (attn.n_embd_head_k as f32).sqrt(),
                attn_logit_softcapping: hp.f_attn_logit_softcapping,
                final_logit_softcapping: hp.f_final_logit_softcapping,
                attn_soft_cap: hp.attn_soft_cap,
                final_softcap_unguarded: gemma2, // gemma2.cpp:167 unguarded
            };
            let w = llama::graph_arch::GemmaModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers,
            };
            if gemma2 {
                llama::context::ForwardWeights::Gemma2(w, gp)
            } else {
                llama::context::ForwardWeights::Gemma3(w, gp)
            }
        }
        llama::arch::LlmArch::QWEN3 => {
            // weights mirror qwen3_e2e.rs::qwen3_weights; the builder takes the
            // global layer-0 `AttnParams` (qwen3_e2e.rs::qwen3_params — qwen3
            // reads only f_norm_rms_eps, the head geometry is uniform)
            llama::context::ForwardWeights::Qwen3(qwen3_weights(model, n_trunk))
        }
        llama::arch::LlmArch::OPENAI_MOE => {
            // gpt-oss: the builder needs `p` (head/rope geometry) *and* `gp`
            // (MoE width, per-layer SWA pattern + SWA rope copies) —
            // openai-moe.cpp:3-18 + llama-model.cpp:2251
            let gp = gpt_oss_params(hp, n_trunk);
            llama::context::ForwardWeights::GptOss(gpt_oss_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::GEMMA4 => {
            let gp = gemma4_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Gemma4(gemma4_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::GRANITE_HYBRID => {
            // attention geometry lives on the attention layers (layer 0 is the
            // mamba2 mixer) — granite-hybrid.cpp:143-198
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            let gp = granite_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Granite(granite_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::LFM2 | llama::arch::LlmArch::LFM2MOE => {
            // the lfm2 build fork (lfm2.cpp:137-139 — build_arch_graph's
            // graph_decision vs graph dispatch): n_layer_decision > 0 loads
            // the Decision form (no memory — create_memory's nullptr arm,
            // llama-model.cpp:2385-2387); lfm2moe ships no decision files
            // (its loader arm keeps the plain path)
            if model.arch == llama::arch::LlmArch::LFM2 && hp.n_layer_decision > 0 {
                llama::context::ForwardWeights::Lfm2Decision(
                    model.lfm2_decision_weights(),
                    model.lfm2_decision_params(),
                )
            } else {
                attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
                let gp = lfm2_params(hp, n_trunk, attn);
                llama::context::ForwardWeights::Lfm2(lfm2_weights(model, n_trunk), gp)
            }
        }
        llama::arch::LlmArch::QWEN35 => {
            // per-layer head geometry (the GDN layers differ from the attention
            // ones) + IMROPE/rope_sections — qwen35.cpp:165-268
            let gp = qwen35_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Qwen35(qwen35_weights(model, n_trunk), gp)
        }
        // ---- arch batch (2026-09-24) ----
        llama::arch::LlmArch::GPT2 => {
            // no rope: the graph gathers learned positions and never calls
            // ggml_rope_ext (gpt2.cpp:58-148); `attn` still carries the LayerNorm
            // eps, which is the norm `f_norm_eps` of gpt2.cpp:4
            attn.norm_eps = hp.f_norm_eps;
            llama::context::ForwardWeights::Gpt2(
                gpt2_weights(model, n_trunk),
                llama::graph_arch::Gpt2Params { attn },
            )
        }
        llama::arch::LlmArch::PHI2 => {
            attn.norm_eps = hp.f_norm_eps; // phi2.cpp:4
            llama::context::ForwardWeights::Phi2(
                phi2_weights(model, n_trunk),
                llama::graph_arch::Phi2Params { attn },
            )
        }
        llama::arch::LlmArch::STARCODER2 => {
            attn.norm_eps = hp.f_norm_eps; // starcoder2.cpp:4
            llama::context::ForwardWeights::StarCoder2(
                starcoder2_weights(model, n_trunk),
                llama::graph_arch::StarCoder2Params { attn },
            )
        }
        llama::arch::LlmArch::COMMAND_R => {
            attn.norm_eps = hp.f_norm_eps; // command-r.cpp:5
            llama::context::ForwardWeights::CommandR(
                command_r_weights(model, n_trunk),
                llama::graph_arch::CommandRParams {
                    attn,
                    logit_scale: hp.f_logit_scale,
                },
            )
        }
        llama::arch::LlmArch::GPTNEOX => {
            attn.norm_eps = hp.f_norm_eps; // gptneox.cpp:4
            llama::context::ForwardWeights::GptNeox(
                gptneox_weights(model, n_trunk),
                llama::graph_arch::GptNeoxParams {
                    attn,
                    use_par_res: hp.use_par_res,
                },
            )
        }
        llama::arch::LlmArch::OLMO2 => {
            // olmo2.cpp:4 — RMS norm; `attn_params` already carries
            // f_norm_rms_eps, so no override here
            llama::context::ForwardWeights::Olmo2(
                olmo2_weights(model, n_trunk),
                llama::graph_arch::Olmo2Params { attn },
            )
        }
        // ---- arch batch 2 (2026-09-25) ----
        llama::arch::LlmArch::CODESHELL => {
            attn.norm_eps = hp.f_norm_eps; // codeshell.cpp:4
            llama::context::ForwardWeights::Codeshell(
                codeshell_weights(model, n_trunk),
                llama::graph_arch::CodeshellParams { attn },
            )
        }
        llama::arch::LlmArch::ORION => {
            attn.norm_eps = hp.f_norm_eps; // orion.cpp:4
            llama::context::ForwardWeights::Orion(
                orion_weights(model, n_trunk),
                llama::graph_arch::OrionParams { attn },
            )
        }
        llama::arch::LlmArch::OLMO => {
            attn.norm_eps = hp.f_norm_eps; // olmo.cpp:4
            llama::context::ForwardWeights::Olmo(
                olmo_weights(model, n_trunk),
                llama::graph_arch::OlmoParams {
                    attn,
                    f_clamp_kqv: hp.f_clamp_kqv,
                },
            )
        }
        llama::arch::LlmArch::XVERSE => {
            // xverse.cpp:4 — RMS; `attn_params` already carries the eps
            llama::context::ForwardWeights::Xverse(
                xverse_weights(model, n_trunk),
                llama::graph_arch::XverseParams { attn },
            )
        }
        llama::arch::LlmArch::INTERNLM2 => llama::context::ForwardWeights::Internlm2(
            internlm2_weights(model, n_trunk),
            llama::graph_arch::Internlm2Params { attn },
        ),
        llama::arch::LlmArch::EXAONE => llama::context::ForwardWeights::Exaone(
            exaone_weights(model, n_trunk),
            llama::graph_arch::ExaoneParams { attn },
        ),
        llama::arch::LlmArch::GEMMA => {
            // gemma.cpp:41-139 — v1: Q is pre-scaled by 1/sqrt(n_embd_head_v)
            // (:86) and build_attn runs with kq_scale 1.0 (:91); the RMS eps
            // comes from meta.rs's GEMMA arm, so no override
            let attention_scale = 1.0 / (attn.n_embd_head_v as f32).sqrt();
            llama::context::ForwardWeights::Gemma1(
                gemma1_weights(model, n_trunk),
                llama::graph_arch::Gemma1Params {
                    attn,
                    attention_scale,
                },
            )
        }
        llama::arch::LlmArch::FALCON => {
            attn.norm_eps = hp.f_norm_eps; // falcon.cpp:4
            llama::context::ForwardWeights::Falcon(
                falcon_weights(model, n_trunk),
                llama::graph_arch::FalconParams { attn },
            )
        }
        // ---- arch batch 3 (2026-09-27): the ALiBi family + the cheap
        // no-rope archs (same derivations as arch_batch3_e2e.rs::assemble) ----
        llama::arch::LlmArch::BAICHUAN => {
            // baichuan.cpp:4-15: 13B (n_layer == 40) = alibi 8.0, no rope;
            // 7B (32) = rope, no alibi (RMS norm — attn carries f_norm_rms_eps)
            let use_rope = hp.n_layer() == 32;
            let f_max_alibi_bias = if hp.n_layer() == 40 { 8.0 } else { 0.0 };
            llama::context::ForwardWeights::Baichuan(
                baichuan_weights(model, n_trunk),
                llama::graph_arch::BaichuanParams {
                    attn,
                    f_max_alibi_bias,
                    use_rope,
                },
            )
        }
        llama::arch::LlmArch::BLOOM => {
            attn.norm_eps = hp.f_norm_eps; // LayerNorm family (bloom.cpp)
            llama::context::ForwardWeights::Bloom(
                bloom_weights(model, n_trunk),
                llama::graph_arch::BloomParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            )
        }
        llama::arch::LlmArch::MPT => {
            attn.norm_eps = hp.f_norm_eps; // mpt.cpp LayerNorm
            llama::context::ForwardWeights::Mpt(
                mpt_weights(model, n_trunk),
                llama::graph_arch::MptParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                    f_clamp_kqv: hp.f_clamp_kqv,
                },
            )
        }
        llama::arch::LlmArch::STARCODER => {
            attn.norm_eps = hp.f_norm_eps; // starcoder.cpp LayerNorm
            llama::context::ForwardWeights::Starcoder(
                starcoder_weights(model, n_trunk),
                llama::graph_arch::StarcoderParams { attn },
            )
        }
        llama::arch::LlmArch::REFACT => {
            // RMS norm; refact.cpp:12 — alibi 8.0 unconditional
            llama::context::ForwardWeights::Refact(
                refact_weights(model, n_trunk),
                llama::graph_arch::RefactParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            )
        }
        llama::arch::LlmArch::PLAMO => {
            // RMS norm — attn carries f_norm_rms_eps (plamo.cpp)
            llama::context::ForwardWeights::Plamo(
                plamo_weights(model, n_trunk),
                llama::graph_arch::PlamoParams { attn },
            )
        }
        llama::arch::LlmArch::STABLELM => {
            attn.norm_eps = hp.f_norm_eps; // stablelm.cpp LayerNorm
            llama::context::ForwardWeights::Stablelm(
                stablelm_weights(model, n_trunk),
                llama::graph_arch::StablelmParams { attn },
            )
        }
        llama::arch::LlmArch::GRANITE
        | llama::arch::LlmArch::MINICPM
        | llama::arch::LlmArch::GRANITE_MOE => {
            // dense granite / minicpm (`minicpm::graph = granite::graph`,
            // models.h:1739) and granite-moe (`granite_moe::graph =
            // granite::graph`, models.h:1680 — granite.cpp's MoE + shared
            // expert branch, granite.cpp:293-306) — `GraniteParams::dense`
            // (all layers attend, no recurrent cells; the MoE fields come off
            // hparams: n_expert / n_expert_used / n_ff_shexp); RMS norm.
            // granite-moe does not read rope_finetuned (unlike dense granite),
            // so its rope_pattern stays all-1 — `has_rope` follows hparams.
            llama::context::ForwardWeights::Granite(
                granite_dense_weights(model),
                llama::graph_arch::GraniteParams::dense(
                    attn,
                    hp,
                    hp.f_logit_scale,
                    hp.f_residual_scale,
                    hp.f_embedding_scale,
                    hp.f_attention_scale,
                ),
            )
        }
        // ---- arch batch 4 (2026-09-28): the MoE family + its dense
        // hangers-on (same derivations as arch_batch4_e2e.rs::assemble) ----
        llama::arch::LlmArch::QWEN2MOE => {
            // qwen2moe.cpp:126-165 — MoE norm_w=false + the sigmoid-gated
            // shared expert: the 1-D `ffn_gate_inp_shexp` router (:54) with
            // silu(x)/x applied to the shexp FFN output (:147-165)
            llama::context::ForwardWeights::Qwen2Moe(
                qwen2moe_weights(model, n_trunk),
                llama::graph_arch::Qwen2MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::QWEN3MOE => {
            // qwen3moe.cpp:136-151 — MoE norm_w=true, no shared expert; the
            // qwen3 attention stack (per-head q/k norms) with tied head allowed
            llama::context::ForwardWeights::Qwen3Moe(
                qwen3moe_weights(model, n_trunk),
                llama::graph_arch::Qwen3MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::PHIMOE => {
            // phimoe's graph is phi3's (models.h:661) — biased RMS norms + the
            // phi3 Q pre-scale live inside the builder; MoE norm_w=true
            // (phi3.cpp:151-165). The rope factors are resolved per layer
            // against this run's context (see phimoe_weights)
            llama::context::ForwardWeights::Phimoe(
                phimoe_weights(model, n_trunk, n_ctx, attn.n_ctx_orig),
                llama::graph_arch::PhimoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::ARCTIC => {
            // arctic.cpp:136-154 — the double FFN: dense square SwiGLU on
            // ffn_norm(ffn_inp) plus MoE (norm_w=true) on ffn_norm_exps(inpSA)
            llama::context::ForwardWeights::Arctic(
                arctic_weights(model, n_trunk),
                llama::graph_arch::ArcticParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::OLMOE => {
            // olmoe.cpp:28-29 — the full-width [n_embd] q/k norms only line up
            // with the 2D projections when n_embd_k_gqa == n_embd, i.e.
            // n_head_kv == n_head (the real OlmoE-1B is MHA); a GQA file trips
            // ggml_mul's can_repeat check in the reference too — error out
            // clearly instead of computing nonsense
            if hp.n_head_kv(0) != hp.n_head(0) {
                return Err(format!(
                    "olmoe requires n_head_kv == n_head for its full-width attn_k_norm \
                     (got {} != {}; GQA files break in the reference as well, olmoe.cpp:29)",
                    hp.n_head_kv(0),
                    hp.n_head(0)
                ));
            }
            llama::context::ForwardWeights::Olmoe(
                olmoe_weights(model, n_trunk),
                llama::graph_arch::OlmoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::ERNIE4_5_MOE => {
            // ernie4-5-moe.cpp:26 (n_moe_layer_step required > 0) — the
            // dense-lead / MoE-step layer split (:63-109, incl. the optional
            // exp_probs_b router bias and the ungated shared expert) is the
            // builder's, driven by these params
            llama::context::ForwardWeights::Ernie45Moe(
                ernie45moe_weights(model, n_trunk),
                llama::graph_arch::Ernie45MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_moe_layer_step: hp.n_moe_layer_step,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_ff_shexp: hp.n_ff_shexp as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::SMOLLM3 => {
            // smollm3.cpp:5 — n_no_rope_layer_step fixed 4; :62 the 0 →
            // 1/sqrt(head) attention-scale fallback is the builder's
            llama::context::ForwardWeights::Smollm3(
                smollm3_weights(model, n_trunk),
                llama::graph_arch::Smollm3Params {
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    attn,
                    f_attention_scale: hp.f_attention_scale,
                },
            )
        }
        llama::arch::LlmArch::SEED_OSS => {
            // seed-oss.cpp:62-70 — same 0 → 1/sqrt(head) fallback; no ffn_norm
            // (attn_post_norm doubles as the FFN norm, :113-116)
            llama::context::ForwardWeights::SeedOss(
                seed_oss_weights(model, n_trunk),
                llama::graph_arch::SeedOssParams {
                    attn,
                    f_attention_scale: hp.f_attention_scale,
                },
            )
        }
        llama::arch::LlmArch::OPENELM => {
            // openelm.cpp:26-28/67-69 — per-layer head counts (and FFN
            // widths) off the GGUF arrays; `attn` carries layer 0's geometry,
            // the builder overrides per layer, and the KV cache rows are
            // per-layer (`ForwardWeights::kv_dims`)
            llama::context::ForwardWeights::Openelm(
                openelm_weights(model, n_trunk),
                llama::graph_arch::OpenelmParams { attn },
            )
        }
        // ---- arch batch 5 (2026-09-24): the mamba family ----
        llama::arch::LlmArch::MAMBA | llama::arch::LlmArch::MAMBA2 => {
            // pure-recurrent: no attention layers, no rope (rope_type NONE);
            // `attn` only carries the eps. mamba2 reaches the same graph as
            // mamba (models.h:942), distinguished by the mixer enum.
            let mamba2 = model.arch == llama::arch::LlmArch::MAMBA2;
            llama::context::ForwardWeights::Mamba(
                mamba_weights(model, n_trunk, mamba2),
                llama::graph_arch::MambaParams {
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    dt_rank: hp.ssm_dt_rank as i64,
                    n_group: hp.ssm_n_group as i64,
                    ssm_dt_b_c_rms: hp.ssm_dt_b_c_rms,
                    norm_eps: hp.f_norm_rms_eps,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        llama::arch::LlmArch::JAMBA => {
            // jamba.cpp:121-134 — the attention layers are rope-less with
            // scale 1/sqrt(n_embd_head); geometry from the first attention
            // layer (layer 0 is a mamba1 mixer)
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Jamba(
                jamba_weights(model, n_trunk),
                llama::graph_arch::JambaParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    dt_rank: hp.ssm_dt_rank as i64,
                    norm_eps: hp.f_norm_rms_eps,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il)).collect(),
                    expert_weights_scale: hp.expert_weights_scale,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        // NEMOTRON_H_MOE reuses the nemotron-h graph + params wholesale
        // (models.h:1539-1543; only graph_mtp differs — documented-skip)
        llama::arch::LlmArch::NEMOTRON_H | llama::arch::LlmArch::NEMOTRON_H_MOE => {
            // nemotron-h.cpp:271-277 — rope-less attention with the
            // f_attention_scale override; `first_attn_layer` is not enough
            // here (an FFN-only layer is non-recurrent too), so pick the
            // first true attention layer (n_ff == 0)
            let il0 = (0..n_trunk)
                .find(|&il| !hp.is_recr(il) && hp.n_ff(il) == 0)
                .unwrap_or(0);
            attn = attn_params(hp, il0, use_flash_attn);
            llama::context::ForwardWeights::NemotronH(
                nemotron_h_weights(model, n_trunk),
                llama::graph_arch::NemotronHParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    n_ff: (0..n_trunk).map(|il| hp.n_ff(il) as i64).collect(),
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    n_ssm_head: hp.ssm_dt_rank as i64,
                    n_group: hp.ssm_n_group as i64,
                    norm_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        // ---- arch batch 6 (2026-09-24): the DeepSeek MLA family ----
        llama::arch::LlmArch::DEEPSEEK2 | llama::arch::LlmArch::DEEPSEEK2OCR => {
            // deepseek2.cpp:417-713 (the OCR variant rides the same graph,
            // models.h:1335). `attn_params(hp, 0)` already carries the cache
            // geometry the C builds the K-only MLA cache with:
            // n_embd_head_k = attention.key_length = kv_lora_rank +
            // qk_rope_head_dim, n_head_kv = 1 (MLA→MQA absorption,
            // deepseek2.cpp:589). deepseek2-ocr files are plain MHA
            // (head_count_kv == head_count, key/value_length = n_embd/n_head).
            let is_ocr = model.arch == llama::arch::LlmArch::DEEPSEEK2OCR;
            llama::context::ForwardWeights::Deepseek2(
                deepseek2_weights(model, n_trunk),
                llama::graph_arch::Deepseek2Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    rope_yarn_log_mul: hp.rope_yarn_log_mul,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    is_ocr,
                },
            )
        }
        llama::arch::LlmArch::DEEPSEEK => {
            // deepseek.cpp:74-194 — the non-MLA v2 base (plain MHA + MoE)
            llama::context::ForwardWeights::Deepseek(
                deepseek_weights(model, n_trunk),
                llama::graph_arch::DeepseekParams {
                    attn,
                    f_attention_scale: hp.f_attention_scale,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::DEEPSEEK32 => {
            // deepseek32.cpp:162-484 — the MLA graph + the DSA lightning
            // indexer over the `llama_kv_cache_dsa` pair (KvCache::new_dsa in
            // context.rs). The indexer F16 mask / k_rot / lid row indices are
            // DecodeContext step inputs (KvLidStep).
            llama::context::ForwardWeights::Deepseek32(
                deepseek2_weights(model, n_trunk),
                llama::graph_arch::Deepseek32Params {
                    ds2: llama::graph_arch::Deepseek2Params {
                        attn,
                        n_embd: hp.n_embd as i64,
                        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                        kv_lora_rank: hp.n_lora_kv as i64,
                        rope_yarn_log_mul: hp.rope_yarn_log_mul,
                        f_attn_temp_scale: hp.f_attn_temp_scale,
                        n_layer_dense_lead: hp.n_layer_dense_lead,
                        n_expert: hp.n_expert as i64,
                        n_expert_used: hp.n_expert_used(0) as i64,
                        expert_weights_norm: hp.expert_weights_norm,
                        expert_weights_scale: hp.expert_weights_scale,
                        expert_gating_func: hp.expert_gating_func as i32,
                        is_ocr: false,
                    },
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k as i64,
                    f_norm_eps: hp.f_norm_eps,
                },
            )
        }
        // ---- arch batch 7 (deepseek4): hyper-connections + the compressed
        // DSV4 cache (KvCache::new_dsv4 in context.rs; the compressor plan
        // inputs are DecodeContext step inputs, Dsv4Step) ----
        llama::arch::LlmArch::DEEPSEEK4 => {
            let ratios = hp.dsv4_compress_ratios[..n_trunk].to_vec();
            llama::context::ForwardWeights::Deepseek4(
                deepseek4_weights(model, n_trunk),
                llama::graph_arch::Deepseek4Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    hc_mult: hp.dsv4_hc_mult as i64,
                    hc_eps: hp.dsv4_hc_eps,
                    hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters as i32,
                    o_group_count: hp.dsv4_o_group_count as i64,
                    o_lora_rank: hp.dsv4_o_lora_rank as i64,
                    compress_rope_base: hp.dsv4_compress_rope_base,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    swiglu_clamp_exp: hp.swiglu_clamp_exp[..n_trunk].to_vec(),
                    swiglu_clamp_shexp: hp.swiglu_clamp_shexp[..n_trunk].to_vec(),
                    ratios,
                    hash_layer_count: hp.dsv4_hash_layer_count,
                    n_swa: hp.n_swa,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                },
            )
        }
        // ---- arch batch 6b (2026-09-24): nemotron / grok / chameleon / deci
        // / jais / falcon-h1 / plamo2 ----
        llama::arch::LlmArch::NEMOTRON => {
            // nemotron.cpp:4 — LayerNorm eps (norm_eps carries f_norm_eps like
            // the gpt2 family)
            attn.norm_eps = hp.f_norm_eps;
            llama::context::ForwardWeights::Nemotron(
                nemotron_weights(model, n_trunk),
                llama::graph_arch::NemotronParams { attn },
            )
        }
        llama::arch::LlmArch::GROK => {
            // grok.cpp:3-33 — the scale/softcap keys with their old-GGUF
            // defaults land in the hparams; the graph applies the GROK kq
            // softcap only in the non-FA branch (llama-graph.cpp:2682-2689)
            llama::context::ForwardWeights::Grok(
                grok_weights(model, n_trunk),
                llama::graph_arch::GrokParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    f_logit_scale: hp.f_logit_scale,
                    f_embedding_scale: hp.f_embedding_scale,
                    f_final_logit_softcapping: hp.f_final_logit_softcapping,
                    f_attn_out_scale: hp.f_attn_out_scale,
                    f_attn_logit_softcapping: hp.f_attn_logit_softcapping,
                },
            )
        }
        llama::arch::LlmArch::CHAMELEON => {
            llama::context::ForwardWeights::Chameleon(
                chameleon_weights(model, n_trunk),
                llama::graph_arch::ChameleonParams {
                    attn,
                    // chameleon.cpp:6-7 — the qk-norm eps is hard-coded 1e-5
                    // into f_norm_eps by the loader; swin_norm rides the GGUF
                    f_norm_eps_qk: hp.f_norm_eps,
                    swin_norm: hp.swin_norm,
                },
            )
        }
        llama::arch::LlmArch::DECI => {
            // the geometry may vary per layer (deci.cpp:30-34); `attn` keeps
            // the rope + global head dims, the builder overrides the head
            // counts per layer
            let il0 = (0..n_trunk).find(|&il| hp.n_head_kv(il) > 0).unwrap_or(0);
            attn = attn_params(hp, il0, use_flash_attn);
            llama::context::ForwardWeights::Deci(
                deci_weights(model, n_trunk, n_ctx, attn.n_ctx_orig),
                llama::graph_arch::DeciParams {
                    attn,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il)).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il)).collect(),
                    n_ff: (0..n_trunk).map(|il| hp.n_ff(il) as i64).collect(),
                    f_attention_scale: hp.f_attention_scale,
                },
            )
        }
        llama::arch::LlmArch::JAIS => {
            // jais.cpp:4-5 — LayerNorm eps + the GGUF-KV ALiBi bias; rope
            // never runs (rope_type NONE)
            attn.norm_eps = hp.f_norm_eps;
            llama::context::ForwardWeights::Jais(
                jais_weights(model, n_trunk),
                llama::graph_arch::JaisParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            )
        }
        llama::arch::LlmArch::FALCON_H1 => {
            // falcon-h1.cpp:112-176 — every layer runs attention AND the
            // mamba2 mixer (is_recr all true); attention geometry from layer 0
            llama::context::ForwardWeights::FalconH1(
                falcon_h1_weights(model, n_trunk),
                llama::graph_arch::FalconH1Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    n_ssm_head: hp.ssm_dt_rank as i64,
                    n_group: hp.ssm_n_group as i64,
                    norm_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        llama::arch::LlmArch::PLAMO2 => {
            // plamo2.cpp:110-201 — the hybrid with its own mamba mixer;
            // attention geometry from the first attention layer (layer 0 is
            // recurrent in every plamo2 file so far)
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Plamo2(
                plamo2_weights(model, n_trunk),
                llama::graph_arch::Plamo2Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
                    d_conv: hp.ssm_d_conv as i64,
                    d_inner: hp.ssm_d_inner as i64,
                    d_state: hp.ssm_d_state as i64,
                    n_heads: hp.ssm_dt_rank as i64,
                    n_group: hp.ssm_n_group as i64,
                    norm_eps: hp.f_norm_rms_eps,
                    dt_dim: 64.max((hp.n_embd / 16) as i64),
                    n_embd_r: hp.n_embd_r(),
                    n_embd_s: hp.n_embd_s(),
                },
            )
        }
        // ---- arch batch 8 (2026-09-30): the MoE long-tail family ----
        llama::arch::LlmArch::HUNYUAN_MOE => {
            // hunyuan-moe.cpp:137-159 — the MoE branch (norm_topk_prob = true,
            // softmax) plus the shared expert as a plain SwiGLU MLP
            llama::context::ForwardWeights::HunyuanMoe(
                hunyuan_moe_weights(model, n_trunk),
                llama::graph_arch::HunyuanMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::DOTS1 => {
            llama::context::ForwardWeights::Dots1(
                dots1_weights(model, n_trunk),
                llama::graph_arch::Dots1Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::BAILINGMOE => {
            // bailingmoe.cpp:111 — kq_scale = 1/sqrt(n_rot) (carried by
            // attn.n_rot); softmax gating with norm_w from hparams
            llama::context::ForwardWeights::Bailingmoe(
                bailingmoe_weights(model, n_trunk),
                llama::graph_arch::BailingmoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::BAILINGMOE2 => {
            llama::context::ForwardWeights::Bailingmoe2(
                bailingmoe2_weights(model, n_trunk),
                llama::graph_arch::Bailingmoe2Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::GLM4_MOE => {
            // glm4-moe.cpp:304-308 — the mrope branch needs the vision-model
            // rope sections; text GLM-4.5 files carry none
            if hp.use_mrope() {
                return Err(
                    "glm4-moe: rope.dimension_sections (mrope) files are not supported by this port"
                        .to_string(),
                );
            }
            llama::context::ForwardWeights::Glm4Moe(
                glm4_moe_weights(model, n_trunk),
                llama::graph_arch::Glm4MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::MINIMAX_M2 => {
            llama::context::ForwardWeights::MinimaxM2(
                minimax_m2_weights(model, n_trunk),
                llama::graph_arch::MinimaxM2Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::COHERE2MOE => {
            llama::context::ForwardWeights::Cohere2Moe(
                cohere2moe_weights(model, n_trunk),
                llama::graph_arch::Cohere2MoeParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    norm_ln_eps: hp.f_norm_eps,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    logit_scale: hp.f_logit_scale,
                },
            )
        }
        llama::arch::LlmArch::EXAONE_MOE => {
            llama::context::ForwardWeights::ExaoneMoe(
                exaone_moe_weights(model, n_trunk),
                llama::graph_arch::ExaoneMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        // ---- arch batch 9 (2026-10): the linear-attention family ----
        llama::arch::LlmArch::PLAMO3 => {
            llama::context::ForwardWeights::Plamo3(
                plamo3_weights(model, n_trunk),
                llama::graph_arch::Plamo3Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                },
            )
        }
        llama::arch::LlmArch::QWEN3NEXT => {
            // layer 0 is recurrent (interval-4 pattern) — the attention
            // geometry lives on the first attention layer
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Qwen3Next(
                qwen3next_weights(model, n_trunk),
                qwen3next_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::KIMI_LINEAR => {
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::KimiLinear(
                kimi_linear_weights(model, n_trunk),
                kimi_linear_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::BAILINGMOE3 => {
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::BailingMoe3(
                bailingmoe3_weights(model, n_trunk),
                bailingmoe3_params(hp, n_trunk, attn),
            )
        }
        // ---- arch batch 10 (2026-10): the small-arch + EXP-op batch ----
        llama::arch::LlmArch::SMALLTHINKER => {
            llama::context::ForwardWeights::Smallthinker(
                smallthinker_weights(model, n_trunk),
                llama::graph_arch::SmallthinkerParams {
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    attn,
                    n_layer: n_trunk,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::LLADA_MOE => {
            llama::context::ForwardWeights::LladaMoe(
                llada_moe_weights(model, n_trunk),
                llama::graph_arch::LladaMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                },
            )
        }
        llama::arch::LlmArch::MINIMAX_01 => {
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Minimax01(
                minimax01_weights(model, n_trunk),
                minimax01_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::GRANITE_SWITCH => {
            llama::context::ForwardWeights::GraniteSwitch(
                graniteswitch_weights(model, n_trunk),
                graniteswitch_params(hp, n_trunk, attn),
            )
        }
        // ---- arch batch 11a (2026-10): the long-tail queue, first half ----
        llama::arch::LlmArch::APERTUS => {
            llama::context::ForwardWeights::Apertus(
                apertus_weights(model, n_trunk),
                apertus_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::GROVEMOE => {
            llama::context::ForwardWeights::Grovemoe(
                grovemoe_weights(model, n_trunk),
                grovemoe_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::QWEN35MOE => {
            llama::context::ForwardWeights::Qwen35Moe(
                qwen35moe_weights(model, n_trunk),
                qwen35moe_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::KIMI_K3 => {
            // the MLA geometry (attention.key_length = [kv_lora|rope] x
            // head_count_kv = 1) of the first non-KDA layer, kimi-linear style
            attn = attn_params(hp, first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::KimiK3(
                kimi_k3_weights(model, n_trunk),
                kimi_k3_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::DOTS3NOTE => {
            llama::context::ForwardWeights::Dots3Note(
                dots3note_weights(model, n_trunk),
                dots3note_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::MINIMAX_M3 => {
            llama::context::ForwardWeights::MinimaxM3(
                minimax_m3_weights(model, n_trunk),
                minimax_m3_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::QWEN4EXP => {
            llama::context::ForwardWeights::Qwen4Exp(
                qwen4exp_weights(model, n_trunk),
                qwen4exp_params(hp, n_trunk, attn),
            )
        }
        // ---- batch 19: glm5-next (the hybrid_idx family) ----
        llama::arch::LlmArch::GLM5_NEXT => {
            // the MLA geometry: the attn half's cache rows are the
            // kv_lora_rank-wide latent (key_length = kv_lora, n_rot == 0,
            // one kv head)
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64;
            a.n_embd_head_v = hp.n_embd_head_v(0) as i64;
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            attn = a;
            llama::context::ForwardWeights::Glm5Next(
                glm5_weights(model, n_trunk),
                glm5_params(hp, n_trunk, a),
            )
        }
        // ---- batch 20 (the c35b66744 sync): k2-horizon (dense + MoVA) ----
        llama::arch::LlmArch::K2_HORIZON => {
            llama::context::ForwardWeights::K2Horizon(
                model.k2_horizon_weights(),
                k2_horizon_params(hp, attn),
            )
        }
        // ---- arch batch 11b (2026-10): the long-tail queue, second half ----
        llama::arch::LlmArch::ARCEE => {
            llama::context::ForwardWeights::Arcee(
                arcee_weights(model, n_trunk),
                llama::graph_arch::ArceeParams {
                    attn,
                    // arcee.cpp never reads attention.scale — stays 0.0, the
                    // 1/sqrt(head) fallback the builder applies
                    f_attention_scale: hp.f_attention_scale,
                },
            )
        }
        llama::arch::LlmArch::JAIS2 => {
            // jais2.cpp:4 — LayerNorm eps (norm_eps carries f_norm_eps like
            // the gpt2/nemotron family)
            attn.norm_eps = hp.f_norm_eps;
            llama::context::ForwardWeights::Jais2(
                jais2_weights(model, n_trunk),
                llama::graph_arch::Jais2Params { attn },
            )
        }
        llama::arch::LlmArch::TALKIE => {
            llama::context::ForwardWeights::Talkie(
                talkie_weights(model, n_trunk),
                llama::graph_arch::TalkieParams {
                    attn,
                    logit_scale: hp.f_logit_scale,
                },
            )
        }
        llama::arch::LlmArch::NANBEIGE => {
            // the loop expansion: n_layer_all = n_layer_phys * n_loops (the
            // loader aliased the layer structs already)
            let n_all = hp.n_layer_all as usize;
            llama::context::ForwardWeights::Nanbeige(
                nanbeige_weights(model, n_all),
                llama::graph_arch::NanbeigeParams {
                    attn,
                    n_layer_phys: hp.nanbeige_n_layer_phys as usize,
                    n_loops: hp.nanbeige_n_loops as usize,
                    skip_loop_final_norm: hp.nanbeige_skip_loop_final_norm,
                    f_attention_scale: hp.f_attention_scale,
                },
            )
        }
        llama::arch::LlmArch::DREAM => {
            llama::context::ForwardWeights::Dream(
                dream_weights(model, n_trunk),
                llama::graph_arch::DreamParams { attn },
            )
        }
        llama::arch::LlmArch::RND1 => {
            llama::context::ForwardWeights::Rnd1(
                rnd1_weights(model, n_trunk),
                llama::graph_arch::Rnd1Params {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        // ---- arch batch 12 (2026-10): the final long-tail queue ----
        llama::arch::LlmArch::HRM_TEXT => {
            // the cache slots ARE the layer list (the loader aliased the two
            // physical stacks onto them)
            let n_slot = hp.n_layer() as usize;
            llama::context::ForwardWeights::HrmText(
                hrm_text_weights(model, n_slot),
                llama::graph_arch::HrmTextParams {
                    attn,
                    n_layers_per_stack: hp.n_hrm_layers_per_stack as usize,
                    n_h_cycles: hp.n_hrm_h_cycles as usize,
                    n_l_cycles: hp.n_hrm_l_cycles as usize,
                    f_embedding_scale: hp.f_embedding_scale,
                },
            )
        }
        llama::arch::LlmArch::LAGUNA => {
            llama::context::ForwardWeights::Laguna(
                laguna_weights(model, n_trunk),
                llama::graph_arch::LagunaParams {
                    attn,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    has_swa: hp.swa_type != llama::hparams::LlamaSwaType::NONE && hp.is_swa_any(),
                    n_rot_swa: hp.n_rot_swa as i32,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_ctx_train: hp.n_ctx_train as i32,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_gating_func: hp.expert_gating_func as i32,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::MAPLE => {
            llama::context::ForwardWeights::Maple(
                maple_weights(model, n_trunk),
                llama::graph_arch::MapleParams {
                    attn,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_rot_swa: hp.n_rot_swa as i32,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                },
            )
        }
        // ---- arch batch 13 (2026-09): the P0 standard-attention queue ----
        llama::arch::LlmArch::COHERE2 => {
            llama::context::ForwardWeights::Cohere2(
                cohere2_weights(model, n_trunk),
                llama::graph_arch::Cohere2Params {
                    attn,
                    norm_ln_eps: hp.f_norm_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    logit_scale: hp.f_logit_scale,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
            )
        }
        llama::arch::LlmArch::CHATGLM => {
            llama::context::ForwardWeights::Chatglm(
                chatglm_weights(model, n_trunk),
                llama::graph_arch::ChatglmParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                },
            )
        }
        llama::arch::LlmArch::BITNET => {
            llama::context::ForwardWeights::Bitnet(
                bitnet_weights(model, n_trunk),
                llama::graph_arch::BitnetParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                },
            )
        }
        llama::arch::LlmArch::DBRX => {
            // LayerNorm eps (dbrx.cpp:4) — norm_ln_eps carries it, the
            // builder never reads the rms one
            llama::context::ForwardWeights::Dbrx(
                dbrx_weights(model, n_trunk),
                llama::graph_arch::DbrxParams {
                    attn,
                    norm_ln_eps: hp.f_norm_eps,
                    clamp_kqv: hp.f_clamp_kqv,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::ERNIE4_5 => {
            // the dense files run ernie4-5-moe's dense branch everywhere —
            // n_layer_dense_lead = n_layer makes every layer dense (the C's
            // arch guard produces exactly that); n_moe_layer_step keeps the
            // assert's >0 requirement (the predicate never fires)
            llama::context::ForwardWeights::Ernie45Moe(
                ernie45moe_weights(model, n_trunk),
                llama::graph_arch::Ernie45MoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_moe_layer_step: 1,
                    n_layer_dense_lead: hp.n_layer(),
                    n_ff_shexp: hp.n_ff_shexp as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::MISTRAL3 => {
            llama::context::ForwardWeights::Mistral3(
                mistral3_weights(model, n_trunk, n_ctx),
                llama::graph_arch::Mistral3Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    f_attn_temp_offset: hp.f_attn_temp_offset,
                    n_attn_temp_floor_scale: hp.n_attn_temp_floor_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::MINICPM3 => {
            // the cache stores the repeated-k_pe MHA rows — n_head_kv == n_head
            // (minicpm3.cpp:184-186)
            let mut a = attn;
            a.n_head_kv = a.n_head;
            llama::context::ForwardWeights::Minicpm3(
                minicpm3_weights(model, n_trunk, n_ctx),
                llama::graph_arch::Minicpm3Params {
                    attn: a,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    q_lora_rank: hp.n_lora_q as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    scale_embd: 12.0,
                    scale_depth: 1.4,
                    n_embd_base: 256,
                },
            )
        }
        llama::arch::LlmArch::GLM4 => {
            llama::context::ForwardWeights::Glm4(
                glm4_weights(model, n_trunk),
                llama::graph_arch::Glm4Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::EXAONE4 => {
            llama::context::ForwardWeights::Exaone4(
                exaone4_weights(model, n_trunk),
                llama::graph_arch::Exaone4Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    swa_none: hp.swa_type == llama::hparams::LlamaSwaType::NONE,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
            )
        }
        llama::arch::LlmArch::LLAMA4 => {
            // per-layer rope frequencies — get_rope_freq_base/scale
            // (llama4.cpp:137-138): the *_swa pair on the SWA layers
            let freq_base = (0..n_trunk)
                .map(|il| {
                    if hp.is_swa(il) {
                        hp.rope_freq_base_train_swa
                    } else {
                        hp.rope_freq_base_train
                    }
                })
                .collect();
            let freq_scale = (0..n_trunk)
                .map(|il| {
                    if hp.is_swa(il) {
                        hp.rope_freq_scale_train_swa
                    } else {
                        hp.rope_freq_scale_train
                    }
                })
                .collect();
            llama::context::ForwardWeights::Llama4(
                llama4_weights(model, n_trunk),
                llama::graph_arch::Llama4Params {
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    f_attention_scale: hp.f_attention_scale,
                    use_kq_norm: hp.use_kq_norm,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    f_attn_temp_offset: hp.f_attn_temp_offset,
                    n_attn_temp_floor_scale: hp.n_attn_temp_floor_scale,
                    freq_base,
                    freq_scale,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::QWEN2VL => {
            llama::context::ForwardWeights::Qwen2Vl(
                qwen2vl_weights(model, n_trunk),
                llama::graph_arch::Qwen2VlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::QWEN3VL | llama::arch::LlmArch::QWEN3VLMOE => {
            llama::context::ForwardWeights::Qwen3Vl(
                qwen3vl_weights(model, n_trunk),
                llama::graph_arch::Qwen3VlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                    n_deepstack_layers: hp.n_deepstack_layers as usize,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::GLM_DSA => {
            // the MLA geometry (attention.key_length = [kv_lora|rope] x
            // head_count_kv = 1) — the deepseek2 convention
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64; // kv_lora_rank + qk_rope
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            llama::context::ForwardWeights::GlmDsa(
                glm_dsa_weights(model, n_trunk),
                llama::graph_arch::GlmDsaParams {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    rope_yarn_log_mul: hp.rope_yarn_log_mul,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    indexer_top_k: hp.indexer_top_k as i64,
                    f_norm_eps: hp.f_norm_eps,
                    is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
                },
            )
        }
        // ---- arch batch 14 (2026-10): the RWKV family + gemma3n ----
        llama::arch::LlmArch::RWKV6 => {
            // pure-recurrent: no attention layers, no rope (rope_type NONE);
            // `attn` only carries the eps
            llama::context::ForwardWeights::Rwkv6(rwkv6_weights(model, n_trunk, false), rwkv6_params(hp))
        }
        llama::arch::LlmArch::RWKV6QWEN2 => {
            llama::context::ForwardWeights::Rwkv6Qwen2(rwkv6_weights(model, n_trunk, true), rwkv6_params(hp))
        }
        llama::arch::LlmArch::RWKV7 => {
            llama::context::ForwardWeights::Rwkv7(rwkv7_weights(model, n_trunk, false), rwkv7_params(hp))
        }
        llama::arch::LlmArch::ARWKV7 => {
            llama::context::ForwardWeights::Arwkv7(rwkv7_weights(model, n_trunk, true), rwkv7_params(hp))
        }
        llama::arch::LlmArch::GEMMA3N => {
            llama::context::ForwardWeights::Gemma3n(
                gemma3n_weights(model, n_trunk),
                llama::graph_arch::Gemma3nParams {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_altup: hp.n_altup as i64,
                    i_altup_act: hp.i_altup_act as i64,
                    n_embd_altup: hp.n_embd_altup as i64,
                    laurel_rank: hp.laurel_rank as i64,
                    // models.h:843-844 — the graph's in-class constants
                    n_layer_sparsity: 10,
                    f_sparsity_std_mul: 1.644_853_4,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    n_layer_kv_from_start: hp.n_layer_kv_from_start as i64,
                    f_attention_scale: hp.f_attention_scale,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    norm_eps: hp.f_norm_rms_eps,
                    f_final_logit_softcapping: hp.f_final_logit_softcapping,
                },
            )
        }
        // ---- arch batch 15 (2026-10): the P1+P2 queue ----
        llama::arch::LlmArch::QWEN => {
            llama::context::ForwardWeights::Qwen1(
                batch15_qwen1(model, n_trunk),
                llama::graph_arch::Qwen1Params { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::MAINCODER => {
            llama::context::ForwardWeights::Maincoder(
                batch15_maincoder(model, n_trunk),
                llama::graph_arch::MaincoderParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::PANGU_EMBED => {
            llama::context::ForwardWeights::PanguEmbed(
                batch15_pangu_embed(model, n_trunk),
                llama::graph_arch::PanguEmbedParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::COGVLM => {
            llama::context::ForwardWeights::Cogvlm(
                batch15_cogvlm(model, n_trunk),
                llama::graph_arch::CogvlmParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::SPARK2_5 => {
            llama::context::ForwardWeights::Spark25(
                batch15_spark25(model, n_trunk),
                llama::graph_arch::Spark25Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
                },
            )
        }
        llama::arch::LlmArch::MUSE_GLIMMER => {
            llama::context::ForwardWeights::MuseGlimmer(
                batch15_muse_glimmer(model, n_trunk),
                llama::graph_arch::MuseGlimmerParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
                    logit_scale: hp.f_logit_scale,
                    final_logit_softcapping: hp.f_final_logit_softcapping,
                },
            )
        }
        llama::arch::LlmArch::LLADA => {
            llama::context::ForwardWeights::Llada(
                batch15_llada(model, n_trunk),
                llama::graph_arch::LladaParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::PLM => {
            // the repeated-k_pe K rows make the cache MHA-wide (the
            // minicpm3 convention — k_states is n_head wide, plm.cpp:155-157)
            let mut a = attn;
            a.n_head_kv = a.n_head;
            attn = a;
            llama::context::ForwardWeights::Plm(
                batch15_plm(model, n_trunk),
                llama::graph_arch::PlmParams {
                    attn: a,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    kv_lora_rank: hp.n_lora_kv as i64,
                },
            )
        }
        llama::arch::LlmArch::HUNYUAN_VL | llama::arch::LlmArch::HUNYUAN_DENSE => {
            llama::context::ForwardWeights::HunyuanVl(
                batch15_hunyuan_vl(model, n_trunk),
                llama::graph_arch::HunyuanVlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    use_mrope: hp.use_mrope(),
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::GRANITE_SWA => {
            llama::context::ForwardWeights::GraniteSwa(
                batch15_granite_swa(model, n_trunk),
                llama::graph_arch::GraniteSwaParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    has_rope: (0..n_trunk).map(|il| hp.has_rope(il)).collect(),
                    deepstack_mapping: hp.deepstack_mapping_arr.clone(),
                    inp_embd_rows: None,
                    f_logit_scale: hp.f_logit_scale,
                    f_residual_scale: hp.f_residual_scale,
                    f_embedding_scale: hp.f_embedding_scale,
                    f_attention_scale: hp.f_attention_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: (0..n_trunk).map(|il| hp.n_expert_used(il) as i64).collect(),
                    n_ff_shexp: hp.n_ff_shexp as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::AFMOE => {
            llama::context::ForwardWeights::Afmoe(
                batch15_afmoe(model, n_trunk),
                llama::graph_arch::AfmoeParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_embd: hp.n_embd as i64,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::MELLUM => {
            llama::context::ForwardWeights::Mellum(
                batch15_mellum(model, n_trunk),
                llama::graph_arch::MellumParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: if hp.swa_type == llama::hparams::LlamaSwaType::NONE {
                        Vec::new()
                    } else {
                        (0..n_trunk).map(|il| hp.is_swa(il)).collect()
                    },
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::PADDLEOCR => {
            llama::context::ForwardWeights::PaddleOcr(
                batch15_paddleocr(model, n_trunk),
                llama::graph_arch::PaddleOcrParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::HY_V3 => {
            llama::context::ForwardWeights::HyV3(
                batch15_hy_v3(model, n_trunk),
                llama::graph_arch::HyV3Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::MIMO2 => {
            llama::context::ForwardWeights::Mimo2(
                batch15_mimo2(model, n_trunk),
                llama::graph_arch::Mimo2Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
                    v_scale: hp.f_attn_value_scale,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_scale: hp.expert_weights_scale,
                },
            )
        }
        llama::arch::LlmArch::STEP35 => {
            llama::context::ForwardWeights::Step35(
                batch15_step35(model, n_trunk),
                llama::graph_arch::Step35Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    n_rot: (0..n_trunk).map(|il| hp.n_rot(il) as i64).collect(),
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                },
            )
        }
        llama::arch::LlmArch::HY_V4 => {
            // the MLA geometry (key_length = [kv_lora|rope], head_count_kv 1)
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64; // kv_lora_rank + qk_rope
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            attn = a;
            llama::context::ForwardWeights::HyV4(
                batch15_hy_v4(model, n_trunk),
                llama::graph_arch::HyV4Params {
                    attn: a,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    hc_mult: hp.dsv4_hc_mult as i64,
                    hc_eps: hp.dsv4_hc_eps,
                    hc_magnitude: hp.hc_magnitude,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    n_expert_shared: hp.n_expert_shared as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    indexer_top_k: hp.indexer_top_k as i64,
                    indexer_n_head: hp.indexer_n_head as i64,
                    indexer_head_size: hp.indexer_head_size as i64,
                    f_norm_eps: hp.f_norm_eps,
                    is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
                },
            )
        }
        llama::arch::LlmArch::MISTRAL4 => {
            // the deepseek2 graph verbatim (models.h:1393-1395) — the
            // NEMOTRON_H_MOE precedent: build deepseek2's weights under the
            // mistral4 arch string
            let is_ocr = false;
            llama::context::ForwardWeights::Deepseek2(
                deepseek2_weights(model, n_trunk),
                llama::graph_arch::Deepseek2Params {
                    attn,
                    n_embd: hp.n_embd as i64,
                    n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
                    n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
                    kv_lora_rank: hp.n_lora_kv as i64,
                    rope_yarn_log_mul: hp.rope_yarn_log_mul,
                    f_attn_temp_scale: hp.f_attn_temp_scale,
                    n_layer_dense_lead: hp.n_layer_dense_lead,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                    expert_weights_norm: hp.expert_weights_norm,
                    expert_weights_scale: hp.expert_weights_scale,
                    expert_gating_func: hp.expert_gating_func as i32,
                    is_ocr,
                },
            )
        }
        other => {
            return Err(format!(
                "arch '{}' ({other:?}) has a loader but no forward builder in this port \
                 (see FILE_MAP.md's architecture matrix)",
                other.name()
            ));
        }
    };
    Ok((weights, attn))
}

// ---------------------------------------------------------------------------
// arch wiring: LlamaModel (+LlamaHparams) -> graph_arch weight bundles
// (the same derivations the per-arch e2e tests use, so the CLI reaches the
// builder those tests verified)
// ---------------------------------------------------------------------------

/// `AttnParams` of layer `il` (llama-model.cpp's generic head geometry +
/// `rope_runtime()` = the full llama-context.cpp:106-215 rope derivation:
/// n_ctx_orig fallback chain, negative ext_factor mapping, yarn mscale/cancel
/// terms, `yarn_attn_factor *= rope_attn_factor`).
fn attn_params(hp: &LlamaHparams, il: usize, use_flash_attn: bool) -> AttnParams {
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head_k: hp.n_embd_head_k(il) as i64,
        n_embd_head_v: hp.n_embd_head_v(il) as i64,
        n_rot: hp.n_rot(il) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn,
    }
}

/// First non-recurrent layer — the hybrid archs keep their attention geometry
/// there (granite-hybrid / lfm2moe params derive it exactly like this).
fn first_attn_layer(hp: &LlamaHparams, n_layer: usize) -> usize {
    (0..n_layer).find(|&il| !hp.is_recr(il)).unwrap_or(0)
}

/// qwen3.cpp:31-46 — separate Q/K/V + per-head attn_q_norm/attn_k_norm
/// (both required there), optional q/k/v biases.
fn qwen3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3ModelWeights {
    graph_arch::Qwen3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        // qwen3.cpp:22-26: no output.weight -> tok_embd (TENSOR_DUPLICATED)
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// openai-moe.cpp:30-66 — 22 tensors per layer (q/k/v + wo biases, attn_sinks,
/// router bias, per-expert biases; all required there except the q/k/v biases).
fn gpt_oss_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GptOssModelWeights {
    graph_arch::GptOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GptOssLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                attn_sinks: l
                    .attn_sinks
                    .unwrap_or_else(|| panic!("layer {il}: attn_sinks")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_inp_b: l
                    .ffn_gate_inp_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_b")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_up_exps_b: l
                    .ffn_up_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps_b")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_gate_exps_b: l
                    .ffn_gate_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps_b")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_down_exps_b: l
                    .ffn_down_exps_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps_b")),
            })
            .collect(),
    }
}

/// openai-moe.cpp:3-18 + llama-graph.cpp:2288-2290: the MoE/SWA parameters.
fn gpt_oss_params(hp: &LlamaHparams, n_layer: usize) -> graph_arch::GptOssParams {
    graph_arch::GptOssParams {
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        swiglu_oai_alpha: 1.702,
        swiglu_oai_limit: 7.0,
        expert_weights_scale: hp.expert_weights_scale,
        // openai-moe.cpp:12 `load_swa_pattern(ml, 2)` -> the is_swa_impl vector
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        // llama-model.cpp:2251 get_rope_freq_base/scale
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
    }
}

/// gemma4.cpp:12-147 — fused or separate qkv, per-head q/k norms, optional
/// rope freq factors / out_scale, dense|MoE FFN with the second branch.
fn gemma4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma4ModelWeights {
    graph_arch::Gemma4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        per_layer_model_proj: m.per_layer_model_proj,
        per_layer_proj_norm: m.per_layer_proj_norm,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gemma4LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                rope_freqs: l.rope_freqs,
                out_scale: l.out_scale,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_inp_s: l.ffn_gate_inp_s,
                ffn_pre_norm_2: l.ffn_pre_norm_2,
                ffn_post_norm_1: l.ffn_post_norm_1,
                ffn_post_norm_2: l.ffn_post_norm_2,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_down_exps_s: l.ffn_down_exps_s,
                per_layer_inp_gate: l.per_layer_inp_gate,
                per_layer_proj: l.per_layer_proj,
                per_layer_post_norm: l.per_layer_post_norm,
            })
            .collect(),
    }
}

/// gemma4.cpp:3-25 + the per-layer vectors (gemma4 is the arch whose head
/// dims/rot/n_ff vary per layer: SWA layers 256x8, full layers 512x1).
fn gemma4_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Gemma4Params {
    graph_arch::Gemma4Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
        f_attention_scale: hp.f_attention_scale,
        f_final_logit_softcapping: hp.f_final_logit_softcapping,
        n_embd_per_layer: hp.n_embd_per_layer as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: (0..n_layer).map(|il| hp.n_ff_exp(il)).collect(),
    }
}

/// granite-hybrid.cpp:12-142 — mamba2 mixer + attention tensors, optional
/// biases, MoE experts + shared expert.
fn granite_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GraniteLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_conv1d_b: l.ssm_conv1d_b,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_d: l.ssm_d,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                wo_b: l.wo_b,
                // granite-hybrid.cpp:222 passes `layer.rope_freqs` (NULL on
                // layer 0 — the C loader skips blk.0.rope_freqs)
                rope_freqs: if il == 0 { None } else { l.rope_freqs },
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// granite-hybrid.cpp:4-9/124-139 + hparams — `attn` must be an *attention*
/// layer's geometry (see `first_attn_layer`).
fn granite_params(
    hp: &LlamaHparams,
    n_layer: usize,
    attn: AttnParams,
) -> graph_arch::GraniteParams {
    graph_arch::GraniteParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        has_rope: (0..n_layer).map(|il| hp.has_rope(il)).collect(),
        d_conv: hp.ssm_d_conv as i64,
        d_inner: hp.ssm_d_inner as i64,
        d_state: hp.ssm_d_state as i64,
        n_ssm_head: hp.ssm_dt_rank as i64,
        n_group: hp.ssm_n_group as i64,
        f_logit_scale: hp.f_logit_scale,
        f_residual_scale: hp.f_residual_scale,
        f_embedding_scale: hp.f_embedding_scale,
        f_attention_scale: hp.f_attention_scale,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_shexp: hp.n_ff_shexp as i64,
        expert_weights_scale: hp.expert_weights_scale,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

/// mamba.cpp:75-110 / mamba2.cpp:60-88 — the pure-recurrent stack. `mamba2`
/// picks the mixer enum variant the shared mamba graph branches on
/// (mamba.cpp:87 `model.arch == LLM_ARCH_MAMBA2`).
fn mamba_weights(m: &LlamaModel, n_trunk: usize, mamba2: bool) -> graph_arch::MambaModelWeights {
    graph_arch::MambaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::MambaLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mixer: if mamba2 {
                    graph_arch::MambaLayerMixer::Mamba2(graph_arch::Mamba2Mixer {
                        ssm_in: l.ssm_in.expect("ssm_in"),
                        ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                        ssm_conv1d_b: l.ssm_conv1d_b,
                        ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                        ssm_a: l.ssm_a.expect("ssm_a"),
                        ssm_d: l.ssm_d.expect("ssm_d"),
                        ssm_norm: l.ssm_norm,
                        ssm_out: l.ssm_out.expect("ssm_out"),
                    })
                } else {
                    graph_arch::MambaLayerMixer::Mamba1(graph_arch::Mamba1Mixer {
                        ssm_in: l.ssm_in.expect("ssm_in"),
                        ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                        ssm_conv1d_b: l.ssm_conv1d_b.expect("ssm_conv1d_b"),
                        ssm_x: l.ssm_x.expect("ssm_x"),
                        ssm_dt: l.ssm_dt.expect("ssm_dt"),
                        ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                        ssm_dt_norm: l.ssm_dt_norm,
                        ssm_b_norm: l.ssm_b_norm,
                        ssm_c_norm: l.ssm_c_norm,
                        ssm_a: l.ssm_a.expect("ssm_a"),
                        ssm_d: l.ssm_d.expect("ssm_d"),
                        ssm_out: l.ssm_out.expect("ssm_out"),
                    })
                },
            })
            .collect(),
    }
}

/// jamba.cpp:67-128 — mamba1 layers (with the dt/B/C RMS trio) or rope-less
/// attention, then dense|MoE FFN tensors.
fn jamba_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::JambaModelWeights {
    graph_arch::JambaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::JambaLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mamba: l.ssm_in.map(|ssm_in| graph_arch::Mamba1Mixer {
                    ssm_in,
                    ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                    ssm_conv1d_b: l.ssm_conv1d_b.expect("ssm_conv1d_b"),
                    ssm_x: l.ssm_x.expect("ssm_x"),
                    ssm_dt: l.ssm_dt.expect("ssm_dt"),
                    ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                    ssm_dt_norm: l.ssm_dt_norm,
                    ssm_b_norm: l.ssm_b_norm,
                    ssm_c_norm: l.ssm_c_norm,
                    ssm_a: l.ssm_a.expect("ssm_a"),
                    ssm_d: l.ssm_d.expect("ssm_d"),
                    ssm_out: l.ssm_out.expect("ssm_out"),
                }),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo,
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
            })
            .collect(),
    }
}

/// nemotron-h.cpp:79-181 — per layer exactly one of {mamba2 mixer, attention,
/// relu² FFN(dense|MoE)} keyed on the loader's is_recr / n_ff split.
fn nemotron_h_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::NemotronHModelWeights {
    graph_arch::NemotronHModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::NemotronHLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                mamba: l.ssm_in.map(|ssm_in| graph_arch::Mamba2Mixer {
                    ssm_in,
                    ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                    ssm_conv1d_b: l.ssm_conv1d_b,
                    ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                    ssm_a: l.ssm_a.expect("ssm_a"),
                    ssm_d: l.ssm_d.expect("ssm_d"),
                    ssm_norm: l.ssm_norm,
                    ssm_out: l.ssm_out.expect("ssm_out"),
                }),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo,
                wo_b: l.wo_b,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_latent_down: l.ffn_latent_down,
                ffn_latent_up: l.ffn_latent_up,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// deepseek2.cpp:98-158 (+deepseek2ocr.cpp:40-73 — the OCR layers populate
/// only the wqkv/wq/wk/wv/wo + FFN fields, the MLA ones stay None).
fn deepseek2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek2ModelWeights {
    graph_arch::Deepseek2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk].iter().map(deepseek2_layer_of).collect(),
    }
}

/// the `model.layers[i]` → `Deepseek2LayerWeights` mapping (deepseek2.cpp /
/// deepseek32.cpp load_arch_tensors field-for-field)
fn deepseek2_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Deepseek2LayerWeights {
    graph_arch::Deepseek2LayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        wq: l.wq,
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wk: l.wk,
        wv: l.wv,
        wq_a: l.wq_a,
        attn_q_a_norm: l.attn_q_a_norm,
        wq_b: l.wq_b,
        wkv_a_mqa: l.wkv_a_mqa,
        attn_kv_a_norm: l.attn_kv_a_norm,
        indexer_k_norm: l.indexer_k_norm,
        indexer_k_norm_b: l.indexer_k_norm_b,
        indexer_proj: l.indexer_proj,
        indexer_attn_k: l.indexer_attn_k,
        indexer_attn_q_b: l.indexer_attn_q_b,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        wkv_b: l.wkv_b,
        wo: l.wo.expect("wo"),
        ffn_norm: l.ffn_norm.expect("ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

/// the MTP block of a deepseek2/deepseek32 nextn file: the layer at
/// `n_layer` (full trunk-shaped tensor set + the `nextn.*` head,
/// deepseek2.cpp:94-160)
fn deepseek2_mtp_weights(m: &LlamaModel) -> graph_arch::Deepseek2MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let l = &m.layers[il];
    graph_arch::Deepseek2MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn_of(l),
        layer: deepseek2_layer_of(l),
    }
}

/// the MTP block of a deepseek4 nextn file (deepseek4.cpp:109-181)
fn deepseek4_mtp_weights(m: &LlamaModel) -> graph_arch::Deepseek4MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let l = &m.layers[il];
    graph_arch::Deepseek4MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.expect("output_hc_fn"),
        hc_head_base: m.hc_head_base.expect("output_hc_base"),
        hc_head_scale: m.hc_head_scale.expect("output_hc_scale"),
        nextn: mtp_nextn_of(l),
        layer: deepseek4_layer_of(l),
    }
}

/// `llama_layer::nextn` of a deepseek MTP layer — the eh_proj/enorm/hnorm
/// trio the graph_mtp classes assert plus the optional head tensors
fn mtp_nextn_of(l: &llama::model::LayerTensors) -> graph_arch::DeepseekMtpNextn {
    let n = &l.nextn;
    graph_arch::DeepseekMtpNextn {
        eh_proj: n.eh_proj.expect("nextn.eh_proj"),
        enorm: n.enorm.expect("nextn.enorm"),
        hnorm: n.hnorm.expect("nextn.hnorm"),
        embed_tokens: n.embed_tokens,
        shared_head_head: n.shared_head_head,
        shared_head_norm: n.shared_head_norm,
    }
}

/// glm5-next's MTP block (b9acf138a, glm5-next.cpp:543-666) — the nextn trio
/// plus the MTP layer `model.layers[n_layer]`'s trunk-shaped DSA block
fn glm5_next_mtp_weights(m: &LlamaModel) -> graph_arch::Glm5NextMtpWeights {
    let il = m.hparams.n_layer() as usize;
    let l = &m.layers[il];
    graph_arch::Glm5NextMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn_of(&m.layers[il]),
        layer: graph_arch::Glm5NextLayerWeights {
            attn_norm: l.attn_norm.unwrap_or_else(|| panic!("mtp layer {il}: attn_norm")),
            ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("mtp layer {il}: ffn_norm")),
            hc_attn_fn: None,
            hc_attn_base: None,
            hc_attn_scale: None,
            hc_ffn_fn: None,
            hc_ffn_base: None,
            hc_ffn_scale: None,
            ssm_q_conv: None,
            ssm_k_conv: None,
            ssm_v_conv: None,
            wq: None,
            wk: None,
            wv: None,
            wqkv: None,
            ssm_f_a: None,
            ssm_f_b: None,
            ssm_beta: None,
            ssm_a: None,
            ssm_dt_b: None,
            ssm_g_a: None,
            ssm_g_b: None,
            ssm_o_norm: None,
            wo: l.wo,
            attn_q_a_norm: l.attn_q_a_norm,
            attn_kv_a_norm: l.attn_kv_a_norm,
            wq_a: l.wq_a,
            wq_b: l.wq_b,
            wkv_a_mqa: l.wkv_a_mqa,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            indexer_k_norm: l.indexer_k_norm,
            indexer_k_norm_b: l.indexer_k_norm_b,
            indexer_proj: l.indexer_proj,
            indexer_attn_k: l.indexer_attn_k,
            indexer_attn_q_b: l.indexer_attn_q_b,
            indexer_kpool_gate: l.indexer_kpool_gate,
            indexer_kpool_ape: l.indexer_kpool_ape,
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
        },
    }
}

// ---------------------------------------------------------------------------
// MTP batch 18 — the nine GLM4-style MTP heads' extractors (the MTP layer at
// index n_layer, loaded by model.rs like a trunk block plus the nextn trio;
// the layer mappings mirror tests/mtp2_e2e.rs's batch-17 converters)
// ---------------------------------------------------------------------------

/// `llama_layer::nextn` of the nine-arch family — `MtpNextn`
/// (mtp_shared_head's trio + the optional embed/head tensors)
fn mtp_nextn9_of(l: &llama::model::LayerTensors) -> graph_arch::MtpNextn {
    let n = &l.nextn;
    graph_arch::MtpNextn {
        eh_proj: n.eh_proj.expect("nextn.eh_proj"),
        enorm: n.enorm.expect("nextn.enorm"),
        hnorm: n.hnorm.expect("nextn.hnorm"),
        embed_tokens: n.embed_tokens,
        shared_head_head: n.shared_head_head,
        shared_head_norm: n.shared_head_norm,
    }
}

/// the MTP layer's facts for [`llama::context::MtpHeadFacts`] — the hparams
/// reads the reference's MTP context makes at il = n_layer()
/// (`n_embd_out()`, the cache rows `n_embd_{k,v}_gqa(il)`, the iswa inputs)
fn mtp_head_facts(hp: &LlamaHparams) -> llama::context::MtpHeadFacts {
    let il = hp.n_layer() as usize;
    llama::context::MtpHeadFacts {
        n_embd: hp.n_embd_out() as i64,
        k_row: hp.n_embd_k_gqa(il) as i64,
        v_row: hp.n_embd_v_gqa(il) as i64,
        n_swa: hp.n_swa,
        swa_type: hp.swa_type,
        is_swa: hp.is_swa(il),
    }
}

/// qwen35.cpp:288-320/519-644 — the MTP layer is a full-attention
/// trunk-shaped block (gated attention + dense SwiGLU)
fn qwen35_mtp_weights(m: &LlamaModel) -> graph_arch::Qwen35MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    graph_arch::Qwen35MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: qwen35_layer_of(&m.layers[il]),
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head: hp.n_embd_head_k(il) as i64,
        n_rot: hp.n_rot(il) as i32,
    }
}

fn qwen35_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Qwen35LayerWeights {
    graph_arch::Qwen35LayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        attn_post_norm: l.attn_post_norm.expect("mtp attn_post_norm"),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: l.wqkv,
        wqkv_gate: l.wqkv_gate,
        ssm_conv1d: l.ssm_conv1d,
        ssm_dt_b: l.ssm_dt_b,
        ssm_a: l.ssm_a,
        ssm_beta: l.ssm_beta,
        ssm_alpha: l.ssm_alpha,
        ssm_norm: l.ssm_norm,
        ssm_out: l.ssm_out,
        ffn_gate: l.ffn_gate.expect("mtp ffn_gate"),
        ffn_up: l.ffn_up.expect("mtp ffn_up"),
        ffn_down: l.ffn_down.expect("mtp ffn_down"),
    }
}

/// qwen35moe.cpp:551-741 — the MoE twin (sigmoid-gated shared expert)
fn qwen35moe_mtp_weights(m: &LlamaModel) -> graph_arch::Qwen35MoeMtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    graph_arch::Qwen35MoeMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: qwen35moe_layer_of(&m.layers[il]),
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head: hp.n_embd_head_k(il) as i64,
        n_rot: hp.n_rot(il) as i32,
    }
}

fn qwen35moe_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Qwen35MoeLayerWeights {
    graph_arch::Qwen35MoeLayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        attn_post_norm: l.attn_post_norm.expect("mtp attn_post_norm"),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: l.wqkv,
        wqkv_gate: l.wqkv_gate,
        ssm_conv1d: l.ssm_conv1d,
        ssm_dt_b: l.ssm_dt_b,
        ssm_a: l.ssm_a,
        ssm_beta: l.ssm_beta,
        ssm_alpha: l.ssm_alpha,
        ssm_norm: l.ssm_norm,
        ssm_out: l.ssm_out,
        ffn_gate_inp: l.ffn_gate_inp.expect("mtp ffn_gate_inp"),
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps.expect("mtp ffn_down_exps"),
        ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

/// qwen3next.cpp:543-650 — the gated-attention MTP block over the MoE FFN
fn qwen3next_mtp_weights(m: &LlamaModel) -> graph_arch::Qwen3NextMtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    graph_arch::Qwen3NextMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: qwen3next_layer_of(&m.layers[il]),
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head: hp.n_embd_head_k(il) as i64,
    }
}

fn qwen3next_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Qwen3NextLayerWeights {
    graph_arch::Qwen3NextLayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        attn_post_norm: l.attn_post_norm.expect("mtp attn_post_norm"),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        wqkv: l.wqkv,
        wqkv_gate: l.wqkv_gate,
        ssm_in: l.ssm_in,
        ssm_conv1d: l.ssm_conv1d,
        ssm_dt_b: l.ssm_dt_b,
        ssm_a: l.ssm_a,
        ssm_beta_alpha: l.ssm_beta_alpha,
        ssm_norm: l.ssm_norm,
        ssm_out: l.ssm_out,
        ffn_gate_inp: l.ffn_gate_inp.expect("mtp ffn_gate_inp"),
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps.expect("mtp ffn_down_exps"),
        ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

/// glm4-moe.cpp:539-644 — the GLM4-style skeleton head
fn glm4_moe_mtp_weights(m: &LlamaModel) -> graph_arch::Glm4MoeMtpWeights {
    let il = m.hparams.n_layer() as usize;
    graph_arch::Glm4MoeMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: glm4_moe_layer_of(&m.layers[il]),
    }
}

fn glm4_moe_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Glm4MoeLayerWeights {
    graph_arch::Glm4MoeLayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wq_b: l.wq_b,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        wo: l.wo.expect("mtp wo"),
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        attn_post_norm: l.attn_post_norm.expect("mtp attn_post_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

/// cohere2moe.cpp:449-571 — the LayerNorm/attention-sinks family (the MTP
/// layer is never SWA-gated in the graph — the window rides the cache)
fn cohere2moe_mtp_weights(m: &LlamaModel) -> graph_arch::Cohere2MoeMtpWeights {
    let il = m.hparams.n_layer() as usize;
    graph_arch::Cohere2MoeMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: cohere2moe_layer_of(&m.layers[il]),
    }
}

fn cohere2moe_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Cohere2MoeLayerWeights {
    graph_arch::Cohere2MoeLayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wq_b: l.wq_b,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        wo: l.wo.expect("mtp wo"),
        wo_b: l.wo_b,
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

/// bailingmoe3.cpp:412-617 — the gated-MLA MTP block (shared_head_norm
/// REQUIRED, the LAYER_OUT_NORM tensor)
fn bailingmoe3_mtp_weights(m: &LlamaModel) -> graph_arch::BailingMoe3MtpWeights {
    let il = m.hparams.n_layer() as usize;
    graph_arch::BailingMoe3MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: bailingmoe3_layer_of(&m.layers[il]),
    }
}

fn bailingmoe3_layer_of(l: &llama::model::LayerTensors) -> graph_arch::BailingMoe3LayerWeights {
    graph_arch::BailingMoe3LayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        ssm_q_conv: l.ssm_q_conv,
        ssm_k_conv: l.ssm_k_conv,
        ssm_v_conv: l.ssm_v_conv,
        wqkv: l.wqkv,
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        ssm_f_a: l.ssm_f_a,
        ssm_beta: l.ssm_beta,
        ssm_a: l.ssm_a,
        ssm_dt_b: l.ssm_dt_b,
        ssm_g_a: l.ssm_g_a,
        ssm_o_norm: None,
        wq_a: l.wq_a,
        attn_q_a_norm: l.attn_q_a_norm,
        wq_b: l.wq_b,
        wq_mla: None,
        wkv_a_mqa: l.wkv_a_mqa,
        attn_kv_a_norm: l.attn_kv_a_norm,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        wqkv_gate: l.wqkv_gate,
        wo: l.wo.expect("mtp wo"),
        ffn_norm: l.ffn_norm.expect("mtp ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

/// hy-v3.cpp:449-586 — the per-head q/k-norm MoE head
fn hy_v3_mtp_weights(m: &LlamaModel) -> graph_arch::HyV3MtpWeights {
    let il = m.hparams.n_layer() as usize;
    graph_arch::HyV3MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: hy_v3_layer_of(&m.layers[il]),
    }
}

fn hy_v3_layer_of(l: &llama::model::LayerTensors) -> graph_arch::HyV3LayerWeights {
    graph_arch::HyV3LayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wq_b: l.wq_b,
        wk: l.wk,
        wk_b: l.wk_b,
        wv: l.wv,
        wv_b: l.wv_b,
        wo: l.wo.expect("mtp wo"),
        attn_q_norm: l.attn_q_norm.expect("mtp attn_q_norm"),
        attn_k_norm: l.attn_k_norm.expect("mtp attn_k_norm"),
        ffn_norm: l.ffn_norm.expect("mtp ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

/// mimo2.cpp:467-585 — attention sinks + the v_scale head
fn mimo2_mtp_weights(m: &LlamaModel) -> graph_arch::Mimo2MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    let swa = hp.is_swa(il);
    graph_arch::Mimo2MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: mimo2_layer_of(&m.layers[il]),
        layer_out_norm: m.layers[il].layer_out_norm,
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        freq_base: if swa {
            hp.rope_freq_base_train_swa
        } else {
            hp.rope_freq_base_train
        },
        freq_scale: if swa {
            hp.rope_freq_scale_train_swa
        } else {
            hp.rope_freq_scale_train
        },
    }
}

fn mimo2_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Mimo2LayerWeights {
    graph_arch::Mimo2LayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wq_b: l.wq_b,
        wk: l.wk,
        wk_b: l.wk_b,
        wv: l.wv,
        wv_b: l.wv_b,
        wo: l.wo.expect("mtp wo"),
        attn_sinks: l.attn_sinks,
        ffn_norm: l.ffn_norm.expect("mtp ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
    }
}

/// step35.cpp:503-616 — the MTP head whose layer may itself be an SWA layer
/// (`w.is_swa` selects the rope_freqs path)
fn step35_mtp_weights(m: &LlamaModel) -> graph_arch::Step35MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    let swa = hp.is_swa(il);
    graph_arch::Step35MtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: mtp_nextn9_of(&m.layers[il]),
        layer: step35_layer_of(&m.layers[il]),
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        is_swa: swa,
        n_rot: hp.n_rot(il) as i32,
        freq_base: if swa {
            hp.rope_freq_base_train_swa
        } else {
            hp.rope_freq_base_train
        },
        freq_scale: if swa {
            hp.rope_freq_scale_train_swa
        } else {
            hp.rope_freq_scale_train
        },
    }
}

fn step35_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Step35LayerWeights {
    graph_arch::Step35LayerWeights {
        attn_norm: l.attn_norm.expect("mtp attn_norm"),
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        rope_freqs: l.rope_freqs,
        wqkv: l.wqkv,
        wqkv_b: l.wqkv_b,
        wq: l.wq,
        wq_b: l.wq_b,
        wk: l.wk,
        wk_b: l.wk_b,
        wv: l.wv,
        wv_b: l.wv_b,
        wo: l.wo.expect("mtp wo"),
        wqkv_gate: l.wqkv_gate,
        ffn_norm: l.ffn_norm.expect("mtp ffn_norm"),
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

/// deepseek4.cpp:82-183 — the trunk weights (hyper-connection mixers, the
/// o_group/o_lora output lora, the per-ratio compressors, hash layers).
fn deepseek4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Deepseek4ModelWeights {
    graph_arch::Deepseek4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.expect("output_hc_fn"),
        hc_head_base: m.hc_head_base.expect("output_hc_base"),
        hc_head_scale: m.hc_head_scale.expect("output_hc_scale"),
        layers: m.layers[..n_trunk].iter().map(deepseek4_layer_of).collect(),
    }
}

/// the `model.layers[i]` → `Deepseek4LayerWeights` mapping
fn deepseek4_layer_of(l: &llama::model::LayerTensors) -> graph_arch::Deepseek4LayerWeights {
    graph_arch::Deepseek4LayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        attn_sinks: l.attn_sinks.expect("attn_sinks"),
        wq_a: l.wq_a.expect("attn_q_a"),
        attn_q_a_norm: l.attn_q_a_norm.expect("attn_q_a_norm"),
        wq_b: l.wq_b.expect("attn_q_b"),
        wkv: l.wkv_a_mqa.expect("attn_kv"),
        attn_kv_norm: l.attn_kv_a_norm.expect("attn_kv_norm"),
        wo_a: l.wo_a.expect("attn_output_a"),
        wo_b: l.wo_b_dsv4.expect("attn_output_b"),
        hc_attn_fn: l.hc_attn_fn.expect("hc_attn_fn"),
        hc_attn_base: l.hc_attn_base.expect("hc_attn_base"),
        hc_attn_scale: l.hc_attn_scale.expect("hc_attn_scale"),
        hc_ffn_fn: l.hc_ffn_fn.expect("hc_ffn_fn"),
        hc_ffn_base: l.hc_ffn_base.expect("hc_ffn_base"),
        hc_ffn_scale: l.hc_ffn_scale.expect("hc_ffn_scale"),
        attn_comp_wkv: l.attn_comp_wkv,
        attn_comp_wgate: l.attn_comp_wgate,
        attn_comp_ape: l.attn_comp_ape,
        attn_comp_norm: l.attn_comp_norm,
        indexer_proj: l.indexer_proj,
        indexer_attn_q_b: l.indexer_attn_q_b,
        indexer_comp_wkv: l.indexer_comp_wkv,
        indexer_comp_wgate: l.indexer_comp_wgate,
        indexer_comp_ape: l.indexer_comp_ape,
        indexer_comp_norm: l.indexer_comp_norm,
        ffn_norm: l.ffn_norm.expect("ffn_norm"),
        ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
        ffn_gate_tid2eid: l.ffn_gate_tid2eid,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_exp_probs_b_vl: l.ffn_exp_probs_b_vl,
        ffn_gate_exps: l.ffn_gate_exps.expect("ffn_gate_exps"),
        ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
        ffn_up_exps: l.ffn_up_exps.expect("ffn_up_exps"),
        ffn_gate_shexp: l.ffn_gate_shexp.expect("ffn_gate_shexp"),
        ffn_down_shexp: l.ffn_down_shexp.expect("ffn_down_shexp"),
        ffn_up_shexp: l.ffn_up_shexp.expect("ffn_up_shexp"),
    }
}

/// deepseek.cpp:34-67 — plain MHA (build_qkv weights) + dense-lead/MoE FFN.
fn deepseek_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DeepseekModelWeights {
    graph_arch::DeepseekModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::DeepseekLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// lfm2moe.cpp — shortconv block + gated attention tensors, dense|MoE FFN.
fn lfm2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Lfm2ModelWeights {
    graph_arch::Lfm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Lfm2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                shortconv_conv: l.shortconv_conv,
                shortconv_in_proj: l.shortconv_in_proj,
                shortconv_out_proj: l.shortconv_out_proj,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
            })
            .collect(),
    }
}

/// k2-horizon.cpp:3-49 — the loader-side facts of the dense + MoVA hybrid
/// (the same derivation tests/k2_horizon_e2e.rs::k2_params pins; `attn` is
/// the uniform layer-0 geometry, NEOX via llama_model_rope_type).
fn k2_horizon_params(hp: &LlamaHparams, attn: AttnParams) -> graph_arch::K2HorizonParams {
    graph_arch::K2HorizonParams {
        attn,
        n_norm_groups: hp.n_norm_groups as i64,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_value_expert: hp.n_value_expert as i64,
        n_value_expert_used: hp.n_value_expert_used as i64,
        expert_gating_func: hp.expert_gating_func as i32,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// lfm2moe.cpp:3-20 + hparams — `attn` must be an attention layer's geometry.
fn lfm2_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Lfm2Params {
    graph_arch::Lfm2Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_shortconv_l_cache: hp.n_shortconv_l_cache as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: (0..n_layer).map(|il| hp.n_expert_used(il)).collect(),
        n_ff_exp: hp.n_ff_exp(0),
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        causal_attn: hp.causal_attn,
        n_embd_r: hp.n_embd_r(),
    }
}

/// qwen35.cpp:31-153 — attention layers carry separate q/k/v + q/k norms,
/// recurrent (gated delta net) layers the fused wqkv + ssm tensors.
fn qwen35_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35ModelWeights {
    graph_arch::Qwen35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        cls_out: m.cls_out,
        cls_out_b: m.cls_out_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen35LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

/// qwen35.cpp:3-30 + hparams — IMROPE sections, per-layer head geometry, the
/// gated delta net geometry and the recurrent state cell sizes.
fn qwen35_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35Params {
    graph_arch::Qwen35Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

// ---------------------------------------------------------------------------
// arch batch (2026-09-24): gpt2 / phi2 / starcoder2 / command-r / gptneox /
// olmo2 — weight bundles mirroring load_arch_tensors (model.rs) 1:1
// ---------------------------------------------------------------------------

/// gpt2.cpp:15-51 — fused wqkv + both LayerNorm biases, learned pos embedding.
fn gpt2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gpt2ModelWeights {
    graph_arch::Gpt2ModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap_or_else(|| panic!("pos_embd")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gpt2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// phi2.cpp:13-41 — separate qkv (fused allowed by `create_tensor_qkv`), both
/// LayerNorm biases and a required lm-head bias.
fn phi2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Phi2ModelWeights {
    graph_arch::Phi2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        output_b: m.output_b.unwrap_or_else(|| panic!("output_b")),
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Phi2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// starcoder2.cpp:16-53 — LayerNorm norms with biases, split qkv, GELU FFN.
fn starcoder2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StarCoder2ModelWeights {
    graph_arch::StarCoder2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StarCoder2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// command-r.cpp:13-40 — biasless LayerNorm, split qkv, optional per-head Q/K
/// norms (n_layer >= 64), SwiGLU FFN.
fn command_r_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::CommandRModelWeights {
    graph_arch::CommandRModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::CommandRLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wq_b: l.wq_b,
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wk_b: l.wk_b,
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// gptneox.cpp:54-87 — fused wqkv + fused bias, LayerNorm biases, GELU FFN.
fn gptneox_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GptNeoxModelWeights {
    graph_arch::GptNeoxModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GptNeoxLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// olmo2.cpp:26-49 — no attn_norm; per-head q/k norms on the flat projections
/// plus post-attention / post-FFN norms.
fn olmo2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Olmo2ModelWeights {
    graph_arch::Olmo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Olmo2LayerWeights {
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
            })
            .collect(),
    }
}

// ---- arch batch 2 (2026-09-25) weight assembly ----

/// codeshell.cpp:12-46 — LN biases everywhere, biased qkv/ffn, required
/// `output.weight` (the token embedding falls back to it, not the other way).
fn codeshell_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::CodeshellModelWeights {
    graph_arch::CodeshellModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::CodeshellLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// orion.cpp:12-37 — LN biases, no `attn_output.bias`, unbiased SwiGLU FFN.
fn orion_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OrionModelWeights {
    graph_arch::OrionModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OrionLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// olmo.cpp:15-37 — no norm tensors at all (weightless graph norms).
fn olmo_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OlmoModelWeights {
    graph_arch::OlmoModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OlmoLayerWeights {
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// xverse.cpp:14-35 — RMS norms, required head.
fn xverse_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::XverseModelWeights {
    graph_arch::XverseModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::XverseLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// internlm2.cpp:13-38 — as xverse, but NORM-mode rope (llama-model.cpp:2935).
fn internlm2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Internlm2ModelWeights {
    graph_arch::Internlm2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Internlm2LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// exaone.cpp:12-40 — optional per-layer rope freq factors.
fn exaone_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ExaoneModelWeights {
    graph_arch::ExaoneModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ExaoneLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// gemma.cpp:13-35 — v1 (no post-norms, head is always the tied embedding).
fn gemma1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma1ModelWeights {
    graph_arch::Gemma1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Gemma1LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// falcon.cpp:13-44 — fused qkv, optional 40B second attention norm.
fn falcon_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::FalconModelWeights {
    graph_arch::FalconModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::FalconLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                attn_norm_2: l.attn_norm_2,
                attn_norm_2_b: l.attn_norm_2_b,
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

// ---- arch batch 3 (2026-09-27) weight assembly (mirror of
// arch_batch3_e2e.rs's per-arch helpers) ----

/// baichuan.cpp:28-121 — RMS norms, separate Q/K/V, SwiGLU FFN.
fn baichuan_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BaichuanModelWeights {
    graph_arch::BaichuanModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BaichuanLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// bloom.cpp:46-151 — LN-with-bias everywhere, fused biased qkv, token-embedding
/// norm, ALiBi.
fn bloom_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BloomModelWeights {
    graph_arch::BloomModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: m
            .token_embd_norm
            .unwrap_or_else(|| panic!("token_embd_norm")),
        tok_norm_b: m
            .token_embd_norm_b
            .unwrap_or_else(|| panic!("token_embd_norm_b")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BloomLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// mpt.cpp:55-171 — fused qkv, optional learned positions, optional q/k norms,
/// GELU-seq FFN with act scales, ALiBi from GGUF KV.
fn mpt_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MptModelWeights {
    graph_arch::MptModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MptLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l.attn_norm_b,
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l.ffn_norm_b,
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b,
                attn_q_norm: l.attn_q_norm,
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm,
                attn_k_norm_b: l.attn_k_norm_b,
                ffn_act: l.ffn_act,
            })
            .collect(),
    }
}

/// starcoder.cpp:46-154 — LN-with-bias, fused biased qkv, required learned
/// positions, GELU FFN, no ALiBi in this revision.
fn starcoder_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StarcoderModelWeights {
    graph_arch::StarcoderModelWeights {
        tok_embd: m.tok_embd,
        pos_embd: m.position_embd.unwrap_or_else(|| panic!("pos_embd")),
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StarcoderLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                wqkv_b: l.wqkv_b.unwrap_or_else(|| panic!("layer {il}: wqkv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_down_b: l
                    .ffn_down_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_b")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_up_b: l.ffn_up_b.unwrap_or_else(|| panic!("layer {il}: ffn_up_b")),
            })
            .collect(),
    }
}

/// refact.cpp:41-160 — RMS norms, separate Q/K/V, ALiBi 8.0, SwiGLU with
/// optional biases (rope_freqs loaded but never read).
fn refact_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::RefactModelWeights {
    graph_arch::RefactModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::RefactLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plamo.cpp:39-136 — RMS norm, separate Q/K/V, NEOX rope, double residual.
fn plamo_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::PlamoModelWeights {
    graph_arch::PlamoModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::PlamoLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// stablelm.cpp:43-172 — LN(-bias), separate Q/K/V, optional per-head q/k LNs,
/// NEOX (partial) rope, sequential-or-parallel residual.
fn stablelm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::StablelmModelWeights {
    graph_arch::StablelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::StablelmLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wq: l.wq.unwrap_or_else(|| panic!("layer {il}: wq")),
                wk: l.wk.unwrap_or_else(|| panic!("layer {il}: wk")),
                wv: l.wv.unwrap_or_else(|| panic!("layer {il}: wv")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                ffn_norm: l.ffn_norm,
                ffn_norm_b: l.ffn_norm_b,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// granite.cpp:123-320 (dense) — the granite tensor set minus the mamba2 mixer
/// (all `ssm_*` None); unlike the hybrid arm every layer may carry rope_freqs
/// (layer 0 is an attention layer here).
fn granite_dense_weights(m: &LlamaModel) -> graph_arch::GraniteModelWeights {
    graph_arch::GraniteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GraniteLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_in: None,
                ssm_conv1d: None,
                ssm_conv1d_b: None,
                ssm_dt_b: None,
                ssm_a: None,
                ssm_d: None,
                ssm_norm: None,
                ssm_out: None,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                wo_b: l.wo_b,
                rope_freqs: l.rope_freqs,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

// ---- arch batch 4 (2026-09-28) weight assembly (mirror of
// arch_batch4_e2e.rs's per-arch helpers; granite-moe goes through
// granite_dense_weights above) ----

/// qwen2moe.cpp:19-57 — every layer MoE: the shared-expert router
/// `ffn_gate_inp_shexp` (:54) + the shexp gate/down/up trio.
fn qwen2moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen2MoeModelWeights {
    graph_arch::Qwen2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen2MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_inp_shexp: l
                    .ffn_gate_inp_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp_shexp")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

// ---- arch batch 8 (2026-09-30): the MoE long-tail family's weight bundles
// (same derivations as arch_batch8_e2e.rs) ----

/// hunyuan-moe.cpp:19-49 — experts at the dense n_ff, the shared expert MLP,
/// per-head q/k norms applied after rope.
fn hunyuan_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::HunyuanMoeModelWeights {
    graph_arch::HunyuanMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::HunyuanMoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

/// dots1.cpp:19-67 — dense lead layers + MoE layers with the fat shared
/// expert (n_ff_exp * n_expert_shared wide) and the optional router bias.
fn dots1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Dots1ModelWeights {
    graph_arch::Dots1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Dots1LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// bailingmoe.cpp:19-55 — MoE in every layer, shared expert, no q/k norms.
fn bailingmoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BailingmoeModelWeights {
    graph_arch::BailingmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BailingmoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_shexp: l
                    .ffn_gate_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_shexp")),
                ffn_down_shexp: l
                    .ffn_down_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_shexp")),
                ffn_up_shexp: l
                    .ffn_up_shexp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_shexp")),
            })
            .collect(),
    }
}

/// bailingmoe2.cpp:21-83 — fused qkv + per-head q/k norms + dense lead.
fn bailingmoe2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Bailingmoe2ModelWeights {
    graph_arch::Bailingmoe2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Bailingmoe2LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// glm4-moe.cpp:29-122 — optional q/k norms, attn_post_norm as the FFN norm,
/// optional shared expert, required router bias.
fn glm4_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4MoeModelWeights {
    graph_arch::Glm4MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Glm4MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// minimax-m2.cpp:15-40 — full-width q/k norms, experts at n_ff, required
/// router bias.
fn minimax_m2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MinimaxM2ModelWeights {
    graph_arch::MinimaxM2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MinimaxM2LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_exp_probs_b: l
                    .ffn_exp_probs_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_exp_probs_b")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// cohere2moe.cpp:40-143 — the dense/MoE split with the optional fused
/// `ffn_gate_up_exps` tensor and the optional shared expert.
fn cohere2moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2MoeModelWeights {
    graph_arch::Cohere2MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Cohere2MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// exaone-moe.cpp:28-101 — per-head q/k norms, dense lead, unconditional
/// shared expert on the MoE layers.
fn exaone_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ExaoneMoeModelWeights {
    graph_arch::ExaoneMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ExaoneMoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// qwen3moe.cpp:17-55 — separate q/k/v + per-head q/k norms, MoE experts only
/// (no shared expert anywhere).
fn qwen3moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3MoeModelWeights {
    graph_arch::Qwen3MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// nemotron.cpp:22-43 — LayerNorm+bias norms, create_tensor_qkv attention,
/// relu² up/down MLP with the optional biases.
fn nemotron_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::NemotronModelWeights {
    graph_arch::NemotronModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.expect("nemotron output_norm_b"),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::NemotronLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_norm_b: l.attn_norm_b.expect("attn_norm_b"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_norm_b: l.ffn_norm_b.expect("ffn_norm_b"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b,
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b,
            })
            .collect(),
    }
}

/// grok.cpp:54-79 — the post-norm attention + GELU MoE (+ optional dense
/// branch) tensor set; `ffn_post_norm` is whichever name the loader filled.
fn grok_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GrokModelWeights {
    graph_arch::GrokModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::GrokLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                attn_out_norm: l.attn_out_norm.expect("attn_out_norm"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
                ffn_up_exps: l.ffn_up_exps.expect("ffn_up_exps"),
                ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
            })
            .collect(),
    }
}

/// chameleon.cpp:29-46 — the full-width q/k LayerNorms + SwiGLU tensors.
fn chameleon_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ChameleonModelWeights {
    graph_arch::ChameleonModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::ChameleonLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_q_norm: l.attn_q_norm.expect("attn_q_norm"),
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm.expect("attn_k_norm"),
                attn_k_norm_b: l.attn_k_norm_b,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_up: l.ffn_up.expect("ffn_up"),
            })
            .collect(),
    }
}

/// deci.cpp:28-73 — the per-layer attention kinds; the rope factors resolved
/// like the reference's `get_rope_factors` (llama-model.cpp:2259-2272), see
/// `phimoe_weights`.
fn deci_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
    n_ctx_orig_yarn: i32,
) -> graph_arch::DeciModelWeights {
    graph_arch::DeciModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::DeciLayerWeights {
                attn_norm: l.attn_norm,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo,
                wo_b: l.wo_b,
                // get_rope_factors (llama-model.cpp:2259-2272)
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                },
                ffn_norm: l.ffn_norm,
                ffn_gate: l.ffn_gate,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down: l.ffn_down,
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up,
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// jais.cpp:25-48 — everything required, including the fused qkv bias.
fn jais_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::JaisModelWeights {
    graph_arch::JaisModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.expect("jais output_norm_b"),
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::JaisLayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                attn_norm_b: l.attn_norm_b.expect("attn_norm_b"),
                wqkv: l.wqkv.expect("wqkv"),
                wqkv_b: l.wqkv_b.expect("wqkv_b"),
                wo: l.wo.expect("wo"),
                wo_b: l.wo_b.expect("wo_b"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_norm_b: l.ffn_norm_b.expect("ffn_norm_b"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_gate_b: l.ffn_gate_b.expect("ffn_gate_b"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b.expect("ffn_down_b"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b.expect("ffn_up_b"),
            })
            .collect(),
    }
}

/// falcon-h1.cpp:68-105 — every layer carries the mamba2 mixer AND the
/// attention tensors.
fn falcon_h1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::FalconH1ModelWeights {
    graph_arch::FalconH1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::FalconH1LayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                ssm_in: l.ssm_in.expect("ssm_in"),
                ssm_conv1d: l.ssm_conv1d.expect("ssm_conv1d"),
                ssm_conv1d_b: l.ssm_conv1d_b,
                ssm_dt_b: l.ssm_dt_b.expect("ssm_dt_b"),
                ssm_a: l.ssm_a.expect("ssm_a"),
                ssm_d: l.ssm_d.expect("ssm_d"),
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out.expect("ssm_out"),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.expect("wo"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_gate: l.ffn_gate.expect("ffn_gate"),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_down_b: l.ffn_down_b,
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plamo2.cpp:59-103 — the mamba/attention per-layer split; every layer
/// carries the post-mixer/post-FFN norms and the SWIGLU FFN.
fn plamo2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Plamo2ModelWeights {
    graph_arch::Plamo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .map(|l| graph_arch::Plamo2LayerWeights {
                attn_norm: l.attn_norm.expect("attn_norm"),
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_x: l.ssm_x,
                ssm_dt: l.ssm_dt,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_d: l.ssm_d,
                ssm_out: l.ssm_out,
                ssm_dt_norm: l.ssm_dt_norm,
                ssm_b_norm: l.ssm_b_norm,
                ssm_c_norm: l.ssm_c_norm,
                wqkv: l.wqkv,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wo: l.wo,
                attn_post_norm: l.attn_post_norm.expect("attn_post_norm"),
                ffn_norm: l.ffn_norm.expect("ffn_norm"),
                ffn_down: l.ffn_down.expect("ffn_down"),
                ffn_up: l.ffn_up.expect("ffn_up"),
                ffn_post_norm: l.ffn_post_norm.expect("ffn_post_norm"),
            })
            .collect(),
    }
}

/// phimoe.cpp:17-45 — the phi3 tensor set (biased RMS norms, fused-or-
/// separate qkv, required wo bias) + the MoE experts. `rope_factors` resolved
/// like the reference's `get_rope_factors` (llama-model.cpp:2259-2272): the
/// layer's own rope_freqs, else long vs short by n_ctx_seq (== n_ctx, the CLI
/// decodes a single sequence) against n_ctx_orig_yarn.
fn phimoe_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
    n_ctx_orig_yarn: i32,
) -> graph_arch::PhimoeModelWeights {
    graph_arch::PhimoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap_or_else(|| panic!("output_norm_b")),
        output: m.output,
        output_b: m.output_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::PhimoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_norm_b: l
                    .attn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm_b")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b.unwrap_or_else(|| panic!("layer {il}: wo_b")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_norm_b: l
                    .ffn_norm_b
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_b")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                // get_rope_factors (llama-model.cpp:2259-2272)
                rope_factors: if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                },
            })
            .collect(),
    }
}

/// arctic.cpp:19-49 — the double FFN per layer: dense square gate/down/up +
/// the MoE quartet behind `ffn_norm_exps`.
fn arctic_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ArcticModelWeights {
    graph_arch::ArcticModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ArcticLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_norm_exps: l
                    .ffn_norm_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_norm_exps")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// olmoe.cpp:15-46 — the full-width [n_embd] q/k norms are required (the MHA
/// precondition is checked by the caller).
fn olmoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OlmoeModelWeights {
    graph_arch::OlmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OlmoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l
                    .ffn_up_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
            })
            .collect(),
    }
}

/// ernie4-5.cpp:26-69 (the ERNIE4_5_MOE branch) — dense lead layers carry the
/// gate/down/up trio, MoE layers the router (+ optional exp_probs_b) and the
/// experts (+ optional shexp trio); both halves stay Option and the builder
/// picks per layer from the dense-lead/MoE-step split.
fn ernie45moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Ernie45MoeModelWeights {
    graph_arch::Ernie45MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Ernie45MoeLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// smollm3.cpp:16-39 — the plain qwen2 tensor set (the nope rope pattern is
/// params-only).
fn smollm3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Smollm3ModelWeights {
    graph_arch::Smollm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Smollm3LayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// seed-oss.cpp:19-42 — q width n_head*head_dim (may differ from n_embd),
/// attn_post_norm doubling as the FFN norm.
fn seed_oss_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::SeedOssModelWeights {
    graph_arch::SeedOssModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::SeedOssLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

/// openelm.cpp:18-43 — the fused per-layer-width wqkv + per-head q/k norms;
/// the per-layer head counts / FFN widths ride on the layer weights
/// (openelm.cpp:67-69).
fn openelm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::OpenelmModelWeights {
    graph_arch::OpenelmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::OpenelmLayerWeights {
                attn_norm: l
                    .attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                attn_q_norm: l
                    .attn_q_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l
                    .attn_k_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                n_head: m.hparams.n_head(il) as i64,
                n_head_kv: m.hparams.n_head_kv(il) as i64,
                n_ff: m.hparams.n_ff(il) as i64,
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// arch batch 9 (2026-10): the linear-attention family — plamo3 / qwen3next /
// kimi-linear / bailingmoe3 weight bundles mirroring load_arch_tensors 1:1
// ---------------------------------------------------------------------------

/// plamo3.cpp:34-57 — the fused-qkv + post-norm + swiglu dense stack.
fn plamo3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Plamo3ModelWeights {
    graph_arch::Plamo3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Plamo3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv.unwrap_or_else(|| panic!("layer {il}: wqkv")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_post_norm: l
                    .ffn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: ffn_post_norm")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

/// qwen3next.cpp:68-106 — attention layers carry the QG-wide q projection +
/// q/k norms; the GDN layers the wqkv/wqkv_gate (or legacy ssm_in) +
/// conv1d/dt/a/beta_alpha/norm/out set; every layer the MoE tail + the gated
/// shared expert.
fn qwen3next_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3NextModelWeights {
    graph_arch::Qwen3NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen3NextLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_in: l.ssm_in,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta_alpha: l.ssm_beta_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l
                    .ffn_gate_inp
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// qwen3next.cpp:4-29 + hparams — per-layer head geometry, the GDN geometry
/// and the recurrent state cell sizes (the qwen35_params shape).
fn qwen3next_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen3NextParams {
    graph_arch::Qwen3NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// kimi-linear.cpp:43-165 — the KDA layers carry the per-stream conv kernels
/// + f_a/f_b/beta/a/dt/g_a/g_b/o_norm set; the MLA layers the (optional q
/// compression +) wkv_a_mqa + the split wk_b/wv_b or legacy wkv_b.
fn kimi_linear_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::KimiLinearModelWeights {
    graph_arch::KimiLinearModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::KimiLinearLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wkv_b: l.wkv_b,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// kimi-linear.cpp:4-32 + hparams — the MLA dims, the KDA geometry and the
/// recurrent state cell sizes. `attn` carries whatever cache geometry the
/// file's attention layers describe ([kv_lora|rope] x 1 for the split files,
/// [qk_head_dim x n_head] for the legacy wkv_b ones).
fn kimi_linear_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::KimiLinearParams {
    graph_arch::KimiLinearParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_qk_rope: hp.n_rot(first_attn_layer(hp, n_layer)) as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
    }
}

/// bailingmoe3.cpp:76-158 — the KDA set (single-stage f_a, the safe gate) +
/// the MLA layers (optional q compression + the output gate) + the dense/MoE
/// FFNs with the swiglu_clamp limits.
fn bailingmoe3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BailingMoe3ModelWeights {
    graph_arch::BailingMoe3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::BailingMoe3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_o_norm: l.ssm_norm,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wqkv_gate: l.wqkv_gate,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// bailingmoe3.cpp:6-44 + hparams — the safe-gate bound, the MLA dims, the
/// KDA geometry, the swiglu_clamp arrays and the MoE knobs.
fn bailingmoe3_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::BailingMoe3Params {
    graph_arch::BailingMoe3Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_lora_q: hp.n_lora_q as i64,
        n_embd_head_qk_rope: hp.n_rot(first_attn_layer(hp, n_layer)) as i64,
        rope_sections: hp.rope_sections,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        swiglu_clamp_exp: hp.swiglu_clamp_exp[..n_layer].to_vec(),
        swiglu_clamp_shexp: hp.swiglu_clamp_shexp[..n_layer].to_vec(),
    }
}

// ---- arch batch 10 (2026-10): the small-arch + EXP-op family ----

fn smallthinker_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::SmallthinkerModelWeights {
    graph_arch::SmallthinkerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::SmallthinkerLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn llada_moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::LladaMoeModelWeights {
    graph_arch::LladaMoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LladaMoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

fn minimax01_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Minimax01ModelWeights {
    graph_arch::Minimax01ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Minimax01LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_norm_2: l.attn_norm_2,
                wqkv_la: l.wqkv,
                wg: l.wqkv_gate,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_exp_probs_b: l.ffn_exp_probs_b,
            })
            .collect(),
    }
}

fn minimax01_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Minimax01Params {
    graph_arch::Minimax01Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(first_attn_layer(hp, n_layer)) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        f_residual_scale: hp.f_residual_scale,
        n_embd_head_la: hp.n_embd_head_la as i64,
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

fn graniteswitch_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteSwitchModelWeights {
    graph_arch::GraniteSwitchModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let sl = l.switch_lora.expect("graniteswitch switch_lora");
                graph_arch::GraniteSwitchLayerWeights {
                    attn_norm: l.attn_norm.unwrap(),
                    wqkv: l.wqkv.unwrap(),
                    wo: l.wo.unwrap(),
                    ffn_norm: l.ffn_norm.unwrap(),
                    ffn_gate: l.ffn_gate.unwrap(),
                    ffn_down: l.ffn_down.unwrap(),
                    ffn_up: l.ffn_up.unwrap(),
                    sl,
                }
            })
            .collect(),
        token_to_slot: m.graniteswitch_token_to_slot.clone(),
        token_to_substitute: m.graniteswitch_token_to_substitute.clone(),
    }
}

fn graniteswitch_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::GraniteSwitchParams {
    graph_arch::GraniteSwitchParams {
        attn,
        n_embd: hp.n_embd as i64,
        router_layer: hp.router_layer as usize,
        n_head: (0..n_layer).map(|il| hp.n_head(il) as i64).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il) as i64).collect(),
        has_rope: (0..=n_layer).map(|il| hp.has_rope(il)).collect(),
        n_ff: hp.n_ff(0) as i64,
        f_logit_scale: hp.f_logit_scale,
        f_residual_scale: hp.f_residual_scale,
        f_embedding_scale: hp.f_embedding_scale,
        f_attention_scale: hp.f_attention_scale,
        n_adapters: hp.graniteswitch_n_adapters as i64,
        router_gain: hp.graniteswitch_router_gain,
    }
}

// ---------------------------------------------------------------------------
// arch batch 11a (2026-10): apertus / grovemoe / qwen35moe / kimi-k3 /
// dots3note / minimax-m3 / qwen4exp — the same derivations the per-arch e2e
// tests use (crates/llama/tests/arch_batch11a_e2e.rs)
// ---------------------------------------------------------------------------

/// apertus.cpp:17-54 — per-head q/k RMS norms + the xIELU FFN.
fn apertus_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ApertusModelWeights {
    graph_arch::ApertusModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::ApertusLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                rope_long: l.rope_long,
                rope_short: l.rope_short,
                rope_freqs: l.rope_freqs,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_q_norm_b: l.attn_q_norm_b,
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_k_norm_b: l.attn_k_norm_b,
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
            })
            .collect(),
    }
}

fn apertus_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::ApertusParams {
    graph_arch::ApertusParams {
        attn,
        xielu_alpha_n: hp.xielu_alpha_n[..n_layer].to_vec(),
        xielu_alpha_p: hp.xielu_alpha_p[..n_layer].to_vec(),
        xielu_beta: hp.xielu_beta[..n_layer].to_vec(),
        xielu_eps: hp.xielu_eps[..n_layer].to_vec(),
        f_attention_scale: hp.f_attention_scale,
        use_longrope_factors: hp.rope_scaling_type_train
            == llama::hparams::LlamaRopeScalingType::LONGROPE,
    }
}

/// grovemoe.cpp:16-61 — GQA + the dual (expert, chunk-expert) MoE.
fn grovemoe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GrovemoeModelWeights {
    graph_arch::GrovemoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::GrovemoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_exps: l
                    .ffn_gate_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_exps")),
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_up_exps: l.ffn_up_exps.unwrap_or_else(|| panic!("layer {il}: ffn_up_exps")),
                ffn_gate_chexps: l
                    .ffn_gate_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_gate_chexps")),
                ffn_down_chexps: l
                    .ffn_down_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_chexps")),
                ffn_up_chexps: l
                    .ffn_up_chexps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_up_chexps")),
            })
            .collect(),
    }
}

fn grovemoe_params(hp: &LlamaHparams, _n_layer: usize, attn: AttnParams) -> graph_arch::GrovemoeParams {
    graph_arch::GrovemoeParams {
        attn,
        n_embd: hp.n_embd as i64,
        expert_group_scale: hp.expert_group_scale,
        n_group_experts: hp.n_group_experts as i64,
        n_ff_chexp: hp.n_ff_chexp as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// qwen35moe.cpp:36-147 — the trunk GDN/attention layers + the MoE everywhere.
fn qwen35moe_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35MoeModelWeights {
    graph_arch::Qwen35MoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen35MoeLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

fn qwen35moe_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35MoeParams {
    graph_arch::Qwen35MoeParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
    }
}

/// kimi-k3.cpp:53-167 — KDA layers (per-stream convs + the full-rank gate) +
/// nope-MLA layers + the latent MoE + the residual-bank scores.
fn kimi_k3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::KimiK3ModelWeights {
    graph_arch::KimiK3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_res_score: m.output_res_score,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::KimiK3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                attn_res_score: l.attn_res_score,
                ffn_res_score: l.ffn_res_score,
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g: l.ssm_g,
                ssm_o_norm: l.ssm_norm,
                wo: l.wo,
                wq_a: l.wq_a,
                attn_q_a_norm: l.attn_q_a_norm,
                wq_b: l.wq_b,
                wq_mla: l.wq,
                wkv_a_mqa: l.wkv_a_mqa,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wkv_b: l.wkv_b,
                wqkv_gate: l.wqkv_gate,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_routed_down: l.ffn_routed_down,
                ffn_routed_up: l.ffn_routed_up,
                ffn_routed_norm: l.ffn_routed_norm,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn kimi_k3_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::KimiK3Params {
    let fa = first_attn_layer(hp, n_layer);
    graph_arch::KimiK3Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(fa) as i64,
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_lora_q: hp.n_lora_q as i64,
        n_embd_head_qk_rope: hp.n_rot(fa) as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        attn_res_block_size: hp.attn_res_block_size,
        situ_beta: hp.situ_beta,
        situ_linear_beta: hp.situ_linear_beta,
        n_expert_latent: hp.n_expert_latent as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
    }
}

/// dots3note.cpp:48-146 — the trunk MLA/indexer/MoE tensors.
fn dots3note_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Dots3NoteModelWeights {
    graph_arch::Dots3NoteModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Dots3NoteLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_q_a_norm: l
                    .attn_q_a_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_q_a_norm")),
                attn_kv_a_norm: l
                    .attn_kv_a_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_kv_a_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                wq_a: l.wq_a.unwrap_or_else(|| panic!("layer {il}: wq_a")),
                wq_b: l.wq_b.unwrap_or_else(|| panic!("layer {il}: wq_b")),
                wkv_a_mqa: l.wkv_a_mqa.unwrap_or_else(|| panic!("layer {il}: wkv_a_mqa")),
                wk_b: l.wk_b.unwrap_or_else(|| panic!("layer {il}: wk_b")),
                wv_b: l.wv_b.unwrap_or_else(|| panic!("layer {il}: wv_b")),
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                wqkv_gate: l.wqkv_gate.unwrap_or_else(|| panic!("layer {il}: wqkv_gate")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

fn dots3note_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Dots3NoteParams {
    graph_arch::Dots3NoteParams {
        n_embd: hp.n_embd as i64,
        attn,
        n_head: (0..n_layer).map(|il| hp.n_head(il) as i32).collect(),
        n_rot: hp.n_rot(0) as i64,
        is_swa: (0..n_layer).map(|il| hp.is_swa(il)).collect(),
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_lora_kv_swa: hp.n_lora_kv_swa as i64,
        n_embd_head_k_mla_swa: hp.n_embd_head_k_mla_swa as i64,
        n_embd_head_v_mla_swa: hp.n_embd_head_v_mla_swa as i64,
        f_norm_eps: hp.f_norm_eps,
        rope_freq_base_swa: hp.rope_freq_base_train_swa,
        rope_freq_scale_swa: hp.rope_freq_scale_train_swa,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k,
        is_indexer_full: (0..n_layer).map(|il| hp.is_indexer_full(il)).collect(),
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
    }
}

/// minimax-m3.cpp:41-90 — GQA + per-head q/k norms + the dense/MoE split.
fn minimax_m3_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MinimaxM3ModelWeights {
    graph_arch::MinimaxM3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::MinimaxM3LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap_or_else(|| panic!("layer {il}: wo")),
                attn_q_norm: l.attn_q_norm.unwrap_or_else(|| panic!("layer {il}: attn_q_norm")),
                attn_k_norm: l.attn_k_norm.unwrap_or_else(|| panic!("layer {il}: attn_k_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                index_q_proj: l.index_q_proj,
                index_k_proj: l.index_k_proj,
                index_q_norm: l.index_q_norm,
                index_k_norm: l.index_k_norm,
            })
            .collect(),
    }
}

fn minimax_m3_params(hp: &LlamaHparams, _n_layer: usize, attn: AttnParams) -> graph_arch::MinimaxM3Params {
    graph_arch::MinimaxM3Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_ff_exp: hp.n_ff_exp(0) as i64,
        n_expert_shared: hp.n_expert_shared as i64,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        msa_blk: hp.indexer_block_size as i64,
        msa_topk_blocks: hp.indexer_top_k as i64,
        msa_local: hp.indexer_local_blocks as i64,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
    }
}

/// qwen4exp.cpp:150-259 — the HC mixers + the GDN/gated-attention layers.
/// qwen4exp's MTP block (qwen4exp.cpp:526-612) — the nextn trio + the
/// nextn hc head mixer + the MTP layer's trunk-shaped set
fn qwen4exp_mtp_weights(m: &LlamaModel) -> graph_arch::Qwen4ExpMtpWeights {
    let il = m.hparams.n_layer() as usize;
    let l = &m.layers[il];
    graph_arch::Qwen4ExpMtpWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        nextn_eh_proj: l.nextn.eh_proj.expect("mtp nextn.eh_proj"),
        nextn_enorm: l.nextn.enorm.expect("mtp nextn.enorm"),
        nextn_hnorm: l.nextn.hnorm.expect("mtp nextn.hnorm"),
        nextn_hc_head_norm: l.nextn.hc_head_norm.expect("mtp nextn hc_head_norm"),
        nextn_hc_head_down: l.nextn.hc_head_down.expect("mtp nextn hc_head_down"),
        nextn_hc_head_up: l.nextn.hc_head_up.expect("mtp nextn hc_head_up"),
        layer: graph_arch::Qwen4ExpLayerWeights {
            hc_attn_norm: l.hc_attn_norm.unwrap_or_else(|| panic!("mtp layer {il}: hc_attn_norm")),
            hc_attn_down: l.hc_attn_down.unwrap_or_else(|| panic!("mtp layer {il}: hc_attn_down")),
            hc_attn_up: l.hc_attn_up.unwrap_or_else(|| panic!("mtp layer {il}: hc_attn_up")),
            hc_attn_inject: l
                .hc_attn_inject
                .unwrap_or_else(|| panic!("mtp layer {il}: hc_attn_inject")),
            hc_ffn_norm: l.hc_ffn_norm.unwrap_or_else(|| panic!("mtp layer {il}: hc_ffn_norm")),
            hc_ffn_down: l.hc_ffn_down.unwrap_or_else(|| panic!("mtp layer {il}: hc_ffn_down")),
            hc_ffn_up: l.hc_ffn_up.unwrap_or_else(|| panic!("mtp layer {il}: hc_ffn_up")),
            hc_ffn_inject: l
                .hc_ffn_inject
                .unwrap_or_else(|| panic!("mtp layer {il}: hc_ffn_inject")),
            wq: l.wq,
            wk: l.wk,
            wv: l.wv,
            wo: l.wo,
            attn_q_norm: l.attn_q_norm,
            attn_k_norm: l.attn_k_norm,
            index_q_proj: l.index_q_proj,
            index_k_proj: l.index_k_proj,
            index_q_norm: l.index_q_norm,
            index_k_norm: l.index_k_norm,
            ple_key: l.ple_key,
            ple_value: l.ple_value,
            ple_norm_key: l.ple_norm_key,
            ple_norm_query: l.ple_norm_query,
            ple_norm_conv: l.ple_norm_conv,
            ple_conv1d: l.ple_conv1d,
            wqkv: l.wqkv,
            wqkv_gate: l.wqkv_gate,
            ssm_conv1d: l.ssm_conv1d,
            ssm_dt_b: l.ssm_dt_b,
            ssm_a: l.ssm_a,
            ssm_beta: l.ssm_beta,
            ssm_alpha: l.ssm_alpha,
            ssm_norm: l.ssm_norm,
            ssm_out: l.ssm_out,
            ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("mtp layer {il}: ffn_gate_inp")),
            ffn_gate_up_exps: l.ffn_gate_up_exps,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_down_exps: l
                .ffn_down_exps
                .unwrap_or_else(|| panic!("mtp layer {il}: ffn_down_exps")),
            ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
        },
    }
}

fn qwen4exp_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen4ExpModelWeights {    graph_arch::Qwen4ExpModelWeights {
        tok_embd: m.tok_embd,
        hc_head_norm: m.hc_head_norm.unwrap_or_else(|| panic!("hc_head_norm")),
        hc_head_down: m.hc_head_down.unwrap_or_else(|| panic!("hc_head_down")),
        hc_head_up: m.hc_head_up.unwrap_or_else(|| panic!("hc_head_up")),
        output: m.output,
        per_layer_tok_embd: m.per_layer_tok_embd,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen4ExpLayerWeights {
                hc_attn_norm: l
                    .hc_attn_norm
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_norm")),
                hc_attn_down: l
                    .hc_attn_down
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_down")),
                hc_attn_up: l.hc_attn_up.unwrap_or_else(|| panic!("layer {il}: hc_attn_up")),
                hc_attn_inject: l
                    .hc_attn_inject
                    .unwrap_or_else(|| panic!("layer {il}: hc_attn_inject")),
                hc_ffn_norm: l
                    .hc_ffn_norm
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_norm")),
                hc_ffn_down: l
                    .hc_ffn_down
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_down")),
                hc_ffn_up: l.hc_ffn_up.unwrap_or_else(|| panic!("layer {il}: hc_ffn_up")),
                hc_ffn_inject: l
                    .hc_ffn_inject
                    .unwrap_or_else(|| panic!("layer {il}: hc_ffn_inject")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                index_q_proj: l.index_q_proj,
                index_k_proj: l.index_k_proj,
                index_q_norm: l.index_q_norm,
                index_k_norm: l.index_k_norm,
                ple_key: l.ple_key,
                ple_value: l.ple_value,
                ple_norm_key: l.ple_norm_key,
                ple_norm_query: l.ple_norm_query,
                ple_norm_conv: l.ple_norm_conv,
                ple_conv1d: l.ple_conv1d,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate_inp: l.ffn_gate_inp.unwrap_or_else(|| panic!("layer {il}: ffn_gate_inp")),
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l
                    .ffn_down_exps
                    .unwrap_or_else(|| panic!("layer {il}: ffn_down_exps")),
                ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

fn qwen4exp_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen4ExpParams {
    graph_arch::Qwen4ExpParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        hc: hp.dsv4_hc_mult as i64,
        hc_lr: hp.hc_low_rank as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
        // batch 19 — the QSA indexer + PLE halves
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        compress_ratios: hp.dsv4_compress_ratios[..n_layer].to_vec(),
        is_ple: (0..n_layer).map(|il| hp.is_ple(il)).collect(),
        ple_ngram_size: hp.ple_ngram_size as i64,
        ple_heads_per_ngram: hp.ple_heads_per_ngram as i64,
        ple_conv_kernel: hp.ple_conv_kernel as i64,
        ple_n_heads: hp.ple_n_heads as i64,
        ple_head_dim: hp.ple_head_dim as i64,
        ple_eos_token_id: hp.ple_eos_token_id as i64,
        ple_image_token_id: hp.ple_image_token_id as i64,
        ple_layer_multipliers: hp.ple_layer_multipliers,
        ple_head_offsets: hp.ple_head_offsets,
        ple_head_vocab_sizes: hp.ple_head_vocab_sizes,
    }
}

fn softmax_logprob(logits: &[f32], token: usize) -> f64 {
    // log p = logit - logsumexp (max-stable, f64 accumulation)
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse: f64 = logits
        .iter()
        .map(|&v| (v - max) as f64)
        .map(f64::exp)
        .sum::<f64>()
        .ln();
    (logits[token] as f64 - max as f64) - lse
}

// ---------------------------------------------------------------------------
// arch batch 11b (2026-10) — the long-tail queue, second half: arcee / jais2
// / talkie / nanbeige / dream / rnd1 (eurobert is encoder-only: the CLI's
// generation path does not reach it, the EncoderContext tests do)
// ---------------------------------------------------------------------------

/// arcee.cpp:27-41 — the llama tensor set + the per-layer rope_freqs factors
fn arcee_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ArceeModelWeights {
    graph_arch::ArceeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::ArceeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                rope_freqs: l.rope_freqs,
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// jais2.cpp:26-45 — LN pairs + biases on every norm and both MLP matrices
fn jais2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Jais2ModelWeights {
    graph_arch::Jais2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b.unwrap(),
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Jais2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: l.attn_norm_b.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_norm_b: l.ffn_norm_b.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_b: l.ffn_up_b.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_b: l.ffn_down_b.unwrap(),
            })
            .collect(),
    }
}

/// talkie.cpp:19-33 — no norms except the [1, n_head] q gain, the layer_out_scale
fn talkie_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::TalkieModelWeights {
    graph_arch::TalkieModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::TalkieLayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                out_scale: l.out_scale.unwrap(),
            })
            .collect(),
    }
}

/// nanbeige.cpp:49-73 — the physical stack plus the aliased loop slots (the
/// loader already mirrored layers[i + j*n_phys] = layers[i])
fn nanbeige_weights(m: &LlamaModel, n_all: usize) -> graph_arch::NanbeigeModelWeights {
    graph_arch::NanbeigeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_all)
            .map(|l| graph_arch::NanbeigeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                rope_freqs: l.rope_freqs,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// dream.cpp:32-45 — qwen2's tensor set (fused-or-separate qkv, SwiGLU gate)
fn dream_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DreamModelWeights {
    graph_arch::DreamModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::DreamLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// rnd1.cpp:29-57 — qwen3moe's tensor set (per-head q/k norms + the softmax
/// MoE experts at the dense n_ff fallback width)
fn rnd1_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Rnd1ModelWeights {
    graph_arch::Rnd1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Rnd1LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

// ---- arch batch 12 (2026-10): the final long-tail queue --------------------
// hrm-text / laguna / maple (crates/llama/tests/arch_batch12_e2e.rs)

/// hrm-text.cpp:34-86 — the two physical stacks' tensor set (qkv + gate +
/// wo + SwiGLU FFN), aliased onto the n_slot cache slots by the loader
fn hrm_text_weights(m: &LlamaModel, n_slot: usize) -> graph_arch::HrmTextModelWeights {
    graph_arch::HrmTextModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        hrm_z_l_init: m.hrm_z_l_init.expect("hrm.z_l_init"),
        layers: m
            .layers
            .iter()
            .take(n_slot)
            .map(|l| graph_arch::HrmTextLayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// laguna.cpp:65-146 — per-layer head counts, the two gate widths, qk norms,
/// the dense-lead / MoE split and the always-on shared expert
fn laguna_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::LagunaModelWeights {
    graph_arch::LagunaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LagunaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_up: l.ffn_up,
                ffn_down: l.ffn_down,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// maple.cpp:24-59 — qkv + per-head qk norms + the softmax-MoE expert stack
fn maple_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::MapleModelWeights {
    graph_arch::MapleModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MapleLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// arch batch 13 (2026-09): the P0 standard-attention queue's weight bundles
// ---------------------------------------------------------------------------

/// cohere2.cpp:21-44
fn cohere2_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Cohere2ModelWeights {
    graph_arch::Cohere2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Cohere2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// chatglm.cpp:25-52
fn chatglm_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::ChatglmModelWeights {
    graph_arch::ChatglmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::ChatglmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
            })
            .collect(),
    }
}

/// bitnet.cpp:12-45
fn bitnet_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::BitnetModelWeights {
    graph_arch::BitnetModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::BitnetLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_sub_norm: l.attn_sub_norm.unwrap(),
                wq: l.wq.unwrap(),
                wq_s: l.wq_s,
                wk: l.wk.unwrap(),
                wk_s: l.wk_s,
                wv: l.wv.unwrap(),
                wv_s: l.wv_s,
                wo: l.wo.unwrap(),
                wo_s: l.wo_s,
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_sub_norm: l.ffn_sub_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_gate_s: l.ffn_gate_s,
                ffn_down: l.ffn_down.unwrap(),
                ffn_down_s: l.ffn_down_s,
                ffn_up: l.ffn_up.unwrap(),
                ffn_up_s: l.ffn_up_s,
            })
            .collect(),
    }
}

/// dbrx.cpp:13-41
fn dbrx_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::DbrxModelWeights {
    graph_arch::DbrxModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::DbrxLayerWeights {
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                attn_norm: l.attn_norm.unwrap(),
                attn_out_norm: l.attn_out_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
            })
            .collect(),
    }
}

/// mistral3.cpp:29-87 — the rope factors resolved like the reference's
/// `get_rope_factors` (llama-model.cpp:2259-2272, the deci/phimoe pattern)
fn mistral3_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
) -> graph_arch::Mistral3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Mistral3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let rope_factors = if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                };
                graph_arch::Mistral3LayerWeights {
                    attn_norm: l.attn_norm.unwrap(),
                    wqkv: l.wqkv,
                    wqkv_b: l.wqkv_b,
                    wq: l.wq,
                    wk: l.wk,
                    wv: l.wv,
                    wq_b: l.wq_b,
                    wk_b: l.wk_b,
                    wv_b: l.wv_b,
                    wo: l.wo.unwrap(),
                    wo_b: l.wo_b,
                    ffn_norm: l.ffn_norm.unwrap(),
                    rope_factors,
                    ffn_gate: l.ffn_gate,
                    ffn_gate_b: l.ffn_gate_b,
                    ffn_down: l.ffn_down,
                    ffn_down_b: l.ffn_down_b,
                    ffn_up: l.ffn_up,
                    ffn_up_b: l.ffn_up_b,
                    ffn_gate_inp: l.ffn_gate_inp,
                    ffn_gate_exps: l.ffn_gate_exps,
                    ffn_down_exps: l.ffn_down_exps,
                    ffn_up_exps: l.ffn_up_exps,
                    ffn_gate_shexp: l.ffn_gate_shexp,
                    ffn_down_shexp: l.ffn_down_shexp,
                    ffn_up_shexp: l.ffn_up_shexp,
                }
            })
            .collect(),
    }
}

/// minicpm3.cpp:14-57 — same get_rope_factors resolution
fn minicpm3_weights(
    m: &LlamaModel,
    n_trunk: usize,
    n_ctx_seq: u32,
) -> graph_arch::Minicpm3ModelWeights {
    let rope = m.hparams.rope_runtime();
    graph_arch::Minicpm3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| {
                let rope_factors = if l.rope_freqs.is_some() {
                    l.rope_freqs
                } else if n_ctx_seq as i64 > rope.n_ctx_orig_yarn as i64 {
                    l.rope_long
                } else {
                    l.rope_short
                };
                graph_arch::Minicpm3LayerWeights {
                    attn_norm: l.attn_norm.unwrap(),
                    attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                    attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                    wq_a: l.wq_a.unwrap(),
                    wq_b: l.wq_b.unwrap(),
                    wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                    wkv_b: l.wkv_b.unwrap(),
                    wo: l.wo.unwrap(),
                    ffn_norm: l.ffn_norm.unwrap(),
                    ffn_gate: l.ffn_gate.unwrap(),
                    ffn_down: l.ffn_down.unwrap(),
                    ffn_up: l.ffn_up.unwrap(),
                    rope_factors,
                }
            })
            .collect(),
    }
}

/// glm4.cpp:15-62 (trunk layers only — the port loads trunk-only files)
fn glm4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm4ModelWeights {
    graph_arch::Glm4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Glm4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

/// exaone4.cpp:24-71 (trunk layers only)
fn exaone4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Exaone4ModelWeights {
    graph_arch::Exaone4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Exaone4LayerWeights {
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                rope_freqs: l.rope_freqs,
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
            })
            .collect(),
    }
}

/// llama4.cpp:44-94
fn llama4_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Llama4ModelWeights {
    graph_arch::Llama4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Llama4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                rope_freqs: l.rope_freqs,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// qwen2vl.cpp:8-36
fn qwen2vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen2VlModelWeights {
    graph_arch::Qwen2VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: m.output_b,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen2VlLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// qwen3vl.cpp:16-54 / qwen3vlmoe.cpp:16-58 (one bundle for both)
fn qwen3vl_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen3VlModelWeights {
    graph_arch::Qwen3VlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen3VlLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wq_b: l.wq_b,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
            })
            .collect(),
    }
}

/// glm-dsa.cpp:74-187 (trunk layers only)
fn glm_dsa_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::GlmDsaModelWeights {
    graph_arch::GlmDsaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::GlmDsaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wq_a: l.wq_a.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                wk_b: l.wk_b.unwrap(),
                wv_b: l.wv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// glm-dsa.cpp:74-187 — the MTP layer (`model.layers[n_layer]`) plus the
/// nextn head (glm-dsa.cpp:176-185)
fn glm_dsa_mtp_weights(m: &LlamaModel) -> graph_arch::GlmDsaMtpWeights {
    let hp = &m.hparams;
    let il = hp.n_layer() as usize;
    let l = &m.layers[il];
    graph_arch::GlmDsaMtpWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        nextn: graph_arch::DeepseekMtpNextn {
            eh_proj: l.nextn.eh_proj.expect("nextn.eh_proj"),
            enorm: l.nextn.enorm.expect("nextn.enorm"),
            hnorm: l.nextn.hnorm.expect("nextn.hnorm"),
            embed_tokens: l.nextn.embed_tokens,
            shared_head_head: l.nextn.shared_head_head,
            shared_head_norm: l.nextn.shared_head_norm,
        },
        layer: graph_arch::GlmDsaLayerWeights {
            attn_norm: l.attn_norm.expect("mtp attn_norm"),
            attn_q_a_norm: l.attn_q_a_norm.expect("mtp attn_q_a_norm"),
            attn_kv_a_norm: l.attn_kv_a_norm.expect("mtp attn_kv_a_norm"),
            wq_a: l.wq_a.expect("mtp wq_a"),
            wq_b: l.wq_b.expect("mtp wq_b"),
            wkv_a_mqa: l.wkv_a_mqa.expect("mtp wkv_a_mqa"),
            wk_b: l.wk_b.expect("mtp wk_b"),
            wv_b: l.wv_b.expect("mtp wv_b"),
            wo: l.wo.expect("mtp wo"),
            ffn_norm: l.ffn_norm.expect("mtp ffn_norm"),
            indexer_k_norm: l.indexer_k_norm,
            indexer_k_norm_b: l.indexer_k_norm_b,
            indexer_proj: l.indexer_proj,
            indexer_attn_k: l.indexer_attn_k,
            indexer_attn_q_b: l.indexer_attn_q_b,
            ffn_gate: l.ffn_gate,
            ffn_down: l.ffn_down,
            ffn_up: l.ffn_up,
            ffn_gate_inp: l.ffn_gate_inp,
            ffn_exp_probs_b: l.ffn_exp_probs_b,
            ffn_gate_exps: l.ffn_gate_exps,
            ffn_down_exps: l.ffn_down_exps,
            ffn_up_exps: l.ffn_up_exps,
            ffn_gate_shexp: l.ffn_gate_shexp,
            ffn_down_shexp: l.ffn_down_shexp,
            ffn_up_shexp: l.ffn_up_shexp,
        },
    }
}

// ---------------------------------------------------------------------------
// arch batch 14 (2026-10): the RWKV family + gemma3n
// ---------------------------------------------------------------------------

/// rwkv6 / rwkv6qwen2 tensor bundle (rwkv6.cpp:29-86 / rwkv6qwen2.cpp:29-77).
/// `qwen2` selects the qwen2 variant's Option set (no LN0 / attn_norm pair /
/// channel mix; SwiGLU FFN instead).
fn rwkv6_weights(m: &LlamaModel, n_trunk: usize, qwen2: bool) -> graph_arch::Rwkv6ModelWeights {
    graph_arch::Rwkv6ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if qwen2 { None } else { m.token_embd_norm },
        tok_norm_b: if qwen2 { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: m.output_norm_b,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Rwkv6LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: if qwen2 { None } else { l.attn_norm_b },
                attn_norm_2: if qwen2 { None } else { l.attn_norm_2 },
                attn_norm_2_b: if qwen2 { None } else { l.attn_norm_2_b },
                time_mix_w1: l.time_mix_w1.unwrap(),
                time_mix_w2: l.time_mix_w2.unwrap(),
                time_mix_lerp_x: l.time_mix_lerp_x.unwrap(),
                time_mix_lerp_w: l.time_mix_lerp_w,
                time_mix_lerp_k: l.time_mix_lerp_k,
                time_mix_lerp_v: l.time_mix_lerp_v,
                time_mix_lerp_r: l.time_mix_lerp_r,
                time_mix_lerp_g: l.time_mix_lerp_g,
                time_mix_lerp_fused: l.time_mix_lerp_fused,
                time_mix_first: l.time_mix_first,
                time_mix_decay: l.time_mix_decay.unwrap(),
                time_mix_decay_w1: l.time_mix_decay_w1.unwrap(),
                time_mix_decay_w2: l.time_mix_decay_w2.unwrap(),
                time_mix_key: l.time_mix_key.unwrap(),
                time_mix_value: l.time_mix_value.unwrap(),
                time_mix_receptance: l.time_mix_receptance.unwrap(),
                time_mix_gate: l.time_mix_gate.unwrap(),
                time_mix_key_b: l.time_mix_key_b,
                time_mix_value_b: l.time_mix_value_b,
                time_mix_receptance_b: l.time_mix_receptance_b,
                time_mix_ln: if qwen2 { None } else { l.time_mix_ln },
                time_mix_ln_b: if qwen2 { None } else { l.time_mix_ln_b },
                time_mix_output: l.time_mix_output.unwrap(),
                channel_mix_lerp_k: if qwen2 { None } else { l.channel_mix_lerp_k },
                channel_mix_lerp_r: if qwen2 { None } else { l.channel_mix_lerp_r },
                channel_mix_key: if qwen2 { None } else { l.channel_mix_key },
                channel_mix_value: if qwen2 { None } else { l.channel_mix_value },
                channel_mix_receptance: if qwen2 { None } else { l.channel_mix_receptance },
                ffn_norm: if qwen2 { l.ffn_norm } else { None },
                ffn_gate: if qwen2 { l.ffn_gate } else { None },
                ffn_down: if qwen2 { l.ffn_down } else { None },
                ffn_up: if qwen2 { l.ffn_up } else { None },
            })
            .collect(),
    }
}

fn rwkv6_params(hp: &LlamaHparams) -> graph_arch::Rwkv6Params {
    graph_arch::Rwkv6Params {
        n_embd: hp.n_embd as i64,
        wkv_head_size: hp.wkv_head_size as i64,
        time_mix_extra_dim: hp.time_mix_extra_dim as i64,
        token_shift_count: hp.token_shift_count as i64,
        rescale_every_n_layers: hp.rescale_every_n_layers as i64,
        norm_eps: hp.f_norm_eps,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

/// rwkv7 / arwkv7 tensor bundle (rwkv7.cpp:49-116 / arwkv7.cpp:49-112).
/// `arwkv` selects the arwkv7 Option set (no LN0 / attn_norm pair / channel
/// mix; SwiGLU FFN instead; optional gating + ln).
fn rwkv7_weights(m: &LlamaModel, n_trunk: usize, arwkv: bool) -> graph_arch::Rwkv7ModelWeights {
    graph_arch::Rwkv7ModelWeights {
        tok_embd: m.tok_embd,
        tok_norm: if arwkv { None } else { m.token_embd_norm },
        tok_norm_b: if arwkv { None } else { m.token_embd_norm_b },
        output_norm: m.output_norm,
        output_norm_b: if arwkv { None } else { m.output_norm_b },
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Rwkv7LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_norm_b: if arwkv { None } else { l.attn_norm_b },
                attn_norm_2: if arwkv { None } else { l.attn_norm_2 },
                attn_norm_2_b: if arwkv { None } else { l.attn_norm_2_b },
                time_mix_w0: l.time_mix_w0.unwrap(),
                time_mix_w1: l.time_mix_w1.unwrap(),
                time_mix_w2: l.time_mix_w2.unwrap(),
                time_mix_a0: l.time_mix_a0.unwrap(),
                time_mix_a1: l.time_mix_a1.unwrap(),
                time_mix_a2: l.time_mix_a2.unwrap(),
                time_mix_v0: l.time_mix_v0.unwrap(),
                time_mix_v1: l.time_mix_v1.unwrap(),
                time_mix_v2: l.time_mix_v2.unwrap(),
                time_mix_g1: l.time_mix_g1,
                time_mix_g2: l.time_mix_g2,
                time_mix_lerp_fused: l.time_mix_lerp_fused.unwrap(),
                time_mix_k_k: l.time_mix_k_k.unwrap(),
                time_mix_k_a: l.time_mix_k_a.unwrap(),
                time_mix_r_k: l.time_mix_r_k.unwrap(),
                time_mix_key: l.time_mix_key.unwrap(),
                time_mix_value: l.time_mix_value.unwrap(),
                time_mix_receptance: l.time_mix_receptance.unwrap(),
                time_mix_ln: l.time_mix_ln,
                time_mix_ln_b: l.time_mix_ln_b,
                time_mix_output: l.time_mix_output.unwrap(),
                channel_mix_lerp_k: if arwkv { None } else { l.channel_mix_lerp_k },
                channel_mix_key: if arwkv { None } else { l.channel_mix_key },
                channel_mix_value: if arwkv { None } else { l.channel_mix_value },
                ffn_norm: if arwkv { l.ffn_norm } else { None },
                ffn_gate: if arwkv { l.ffn_gate } else { None },
                ffn_down: if arwkv { l.ffn_down } else { None },
                ffn_up: if arwkv { l.ffn_up } else { None },
            })
            .collect(),
    }
}

fn rwkv7_params(hp: &LlamaHparams) -> graph_arch::Rwkv7Params {
    graph_arch::Rwkv7Params {
        n_embd: hp.n_embd as i64,
        wkv_head_size: hp.wkv_head_size as i64,
        token_shift_count: hp.token_shift_count as i64,
        norm_eps: hp.f_norm_eps,
        norm_rms_eps: hp.f_norm_rms_eps,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

/// gemma3n tensor bundle (gemma3n.cpp:21-76).
fn gemma3n_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Gemma3nModelWeights {
    graph_arch::Gemma3nModelWeights {
        tok_embd: m.tok_embd,
        output: m.output,
        output_norm: m.output_norm,
        altup_proj: m.altup_proj.unwrap(),
        altup_unembd_proj: m.altup_unembd_proj.unwrap(),
        per_layer_tok_embd: m.per_layer_tok_embd.unwrap(),
        per_layer_model_proj: m.per_layer_model_proj.unwrap(),
        per_layer_proj_norm: m.per_layer_proj_norm.unwrap(),
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Gemma3nLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                per_layer_inp_gate: l.per_layer_inp_gate.unwrap(),
                per_layer_proj: l.per_layer_proj.unwrap(),
                per_layer_post_norm: l.per_layer_post_norm.unwrap(),
                altup_correct_coef: l.altup_correct_coef.unwrap(),
                altup_correct_scale: l.altup_correct_scale.unwrap(),
                altup_predict_coef: l.altup_predict_coef.unwrap(),
                altup_router: l.altup_router.unwrap(),
                altup_router_norm: l.altup_router_norm.unwrap(),
                laurel_l: l.laurel_l.unwrap(),
                laurel_r: l.laurel_r.unwrap(),
                laurel_post_norm: l.laurel_post_norm.unwrap(),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// arch batch 15 (2026-10) — the P1+P2 weight bundles (verbatim copies of
// llama-cli's wiring; the server and the CLI drive the identical graph)
// ---------------------------------------------------------------------------

/// qwen.cpp:13-37
fn batch15_qwen1(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen1ModelWeights {
    graph_arch::Qwen1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Qwen1LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wqkv_b: l.wqkv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// maincoder.cpp:12-41
fn batch15_maincoder(m: &LlamaModel, n_trunk: usize) -> graph_arch::MaincoderModelWeights {
    graph_arch::MaincoderModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MaincoderLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// pangu-embed.cpp:13-52
fn batch15_pangu_embed(m: &LlamaModel, n_trunk: usize) -> graph_arch::PanguEmbedModelWeights {
    graph_arch::PanguEmbedModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PanguEmbedLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// cogvlm.cpp:12-47
fn batch15_cogvlm(m: &LlamaModel, n_trunk: usize) -> graph_arch::CogvlmModelWeights {
    graph_arch::CogvlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::CogvlmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv.unwrap(),
                wo: l.wo.unwrap(),
                visexp_attn_wqkv: l.visexp_attn_wqkv.unwrap(),
                visexp_attn_wo: l.visexp_attn_wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                visexp_ffn_gate: l.visexp_ffn_gate.unwrap(),
                visexp_ffn_down: l.visexp_ffn_down.unwrap(),
                visexp_ffn_up: l.visexp_ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// spark2-5.cpp:20-50
fn batch15_spark25(m: &LlamaModel, n_trunk: usize) -> graph_arch::Spark25ModelWeights {
    graph_arch::Spark25ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Spark25LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// muse-glimmer.cpp:21-55
fn batch15_muse_glimmer(m: &LlamaModel, n_trunk: usize) -> graph_arch::MuseGlimmerModelWeights {
    graph_arch::MuseGlimmerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MuseGlimmerLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// llada.cpp:19-60
fn batch15_llada(m: &LlamaModel, n_trunk: usize) -> graph_arch::LladaModelWeights {
    graph_arch::LladaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: None,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::LladaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
            })
            .collect(),
    }
}

/// plm.cpp:13-42
fn batch15_plm(m: &LlamaModel, n_trunk: usize) -> graph_arch::PlmModelWeights {
    graph_arch::PlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PlmLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wq: l.wq.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wkv_b: l.wkv_b.unwrap(),
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// hunyuan-vl.cpp:23-54 (+ the hunyuan-dense typedef)
fn batch15_hunyuan_vl(m: &LlamaModel, n_trunk: usize) -> graph_arch::HunyuanVlModelWeights {
    graph_arch::HunyuanVlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HunyuanVlLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// granite-swa.cpp:56-115
fn batch15_granite_swa(m: &LlamaModel, n_trunk: usize) -> graph_arch::GraniteSwaModelWeights {
    graph_arch::GraniteSwaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::GraniteSwaLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wo_b: l.wo_b,
                attn_sinks: l.attn_sinks.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_b: l.ffn_gate_b,
                ffn_down_b: l.ffn_down_b,
                ffn_up_b: l.ffn_up_b,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// afmoe.cpp:38-101
fn batch15_afmoe(m: &LlamaModel, n_trunk: usize) -> graph_arch::AfmoeModelWeights {
    graph_arch::AfmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::AfmoeLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_post_norm: l.attn_post_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_post_norm: l.ffn_post_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
            })
            .collect(),
    }
}

/// mellum.cpp:27-64
fn batch15_mellum(m: &LlamaModel, n_trunk: usize) -> graph_arch::MellumModelWeights {
    graph_arch::MellumModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::MellumLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate_inp: l.ffn_gate_inp.unwrap(),
                ffn_up_exps: l.ffn_up_exps.unwrap(),
                ffn_gate_exps: l.ffn_gate_exps.unwrap(),
                ffn_down_exps: l.ffn_down_exps.unwrap(),
            })
            .collect(),
    }
}

/// paddleocr.cpp (the ernie4_5 tensor set)
fn batch15_paddleocr(m: &LlamaModel, n_trunk: usize) -> graph_arch::PaddleOcrModelWeights {
    graph_arch::PaddleOcrModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::PaddleOcrLayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate.unwrap(),
                ffn_down: l.ffn_down.unwrap(),
                ffn_up: l.ffn_up.unwrap(),
            })
            .collect(),
    }
}

/// hy-v3.cpp:22-97 (trunk tensors; the MTP block's nextn set is loaded but
/// its graph_mtp is the documented batch-15 skip)
fn batch15_hy_v3(m: &LlamaModel, n_trunk: usize) -> graph_arch::HyV3ModelWeights {
    graph_arch::HyV3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HyV3LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_q_norm: l.attn_q_norm.unwrap(),
                attn_k_norm: l.attn_k_norm.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_up_exps: l.ffn_gate_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// mimo2.cpp:25-82 (trunk tensors; graph_mtp is the documented batch-15 skip)
fn batch15_mimo2(m: &LlamaModel, n_trunk: usize) -> graph_arch::Mimo2ModelWeights {
    graph_arch::Mimo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Mimo2LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                attn_sinks: l.attn_sinks,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
            })
            .collect(),
    }
}

/// step35.cpp:37-182 (trunk tensors; graph_mtp is the documented batch-15 skip)
fn batch15_step35(m: &LlamaModel, n_trunk: usize) -> graph_arch::Step35ModelWeights {
    graph_arch::Step35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::Step35LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                rope_freqs: l.rope_freqs,
                wqkv: l.wqkv,
                wqkv_b: l.wqkv_b,
                wq: l.wq,
                wq_b: l.wq_b,
                wk: l.wk,
                wk_b: l.wk_b,
                wv: l.wv,
                wv_b: l.wv_b,
                wo: l.wo.unwrap(),
                wqkv_gate: l.wqkv_gate,
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
            })
            .collect(),
    }
}

/// hy-v4.cpp:70-155
fn batch15_hy_v4(m: &LlamaModel, n_trunk: usize) -> graph_arch::HyV4ModelWeights {
    graph_arch::HyV4ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        hc_head_fn: m.hc_head_fn.unwrap(),
        hc_head_base: m.hc_head_base.unwrap(),
        hc_head_scale: m.hc_head_scale.unwrap(),
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| graph_arch::HyV4LayerWeights {
                attn_norm: l.attn_norm.unwrap(),
                attn_sinks: l.attn_sinks.unwrap(),
                wq_a: l.wq_a.unwrap(),
                attn_q_a_norm: l.attn_q_a_norm.unwrap(),
                wq_b: l.wq_b.unwrap(),
                wkv_a_mqa: l.wkv_a_mqa.unwrap(),
                attn_kv_a_norm: l.attn_kv_a_norm.unwrap(),
                wk_b: l.wk_b.unwrap(),
                wv_b: l.wv_b.unwrap(),
                wo: l.wo.unwrap(),
                wqkv_gate: l.wqkv_gate.unwrap(),
                indexer_attn_q_b: l.indexer_attn_q_b,
                indexer_attn_k: l.indexer_attn_k,
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                hc_attn_fn: l.hc_attn_fn.unwrap(),
                hc_attn_base: l.hc_attn_base.unwrap(),
                hc_attn_scale: l.hc_attn_scale.unwrap(),
                hc_ffn_fn: l.hc_ffn_fn.unwrap(),
                hc_ffn_base: l.hc_ffn_base.unwrap(),
                hc_ffn_scale: l.hc_ffn_scale.unwrap(),
                ffn_norm: l.ffn_norm.unwrap(),
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// llama-embed's weight bundle (llama.cpp:36-61 — the LLAMA tensor set)
fn llama_embed_weights(model: &LlamaModel) -> llama::graph_arch::LlamaModelWeights {
    let layers = model
        .layers
        .iter()
        .map(|l| llama::graph_arch::LlamaLayerWeights {
            attn_norm: l.attn_norm.expect("attn_norm"),
            wq: l.wq.expect("wq"),
            wk: l.wk.expect("wk"),
            wv: l.wv.expect("wv"),
            wo: l.wo.expect("wo"),
            wq_b: l.wq_b,
            wk_b: l.wk_b,
            wv_b: l.wv_b,
            wo_b: l.wo_b,
            ffn_norm: l.ffn_norm.expect("ffn_norm"),
            ffn_gate: l.ffn_gate.expect("ffn_gate"),
            ffn_down: l.ffn_down.expect("ffn_down"),
            ffn_up: l.ffn_up.expect("ffn_up"),
            ffn_gate_b: l.ffn_gate_b,
            ffn_down_b: l.ffn_down_b,
            ffn_up_b: l.ffn_up_b,
        })
        .collect();
    llama::graph_arch::LlamaModelWeights {
        tok_embd: model.tok_embd,
        output_norm: model.output_norm,
        output: model.output,
        output_b: model.output_b,
        layers,
    }
}

/// batch 19 — glm5-next's per-layer tensor set (glm5-next.cpp:61-186)
fn glm5_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Glm5NextModelWeights {
    graph_arch::Glm5NextModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Glm5NextLayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                ffn_norm: l.ffn_norm.unwrap_or_else(|| panic!("layer {il}: ffn_norm")),
                hc_attn_fn: l.hc_attn_fn,
                hc_attn_base: l.hc_attn_base,
                hc_attn_scale: l.hc_attn_scale,
                hc_ffn_fn: l.hc_ffn_fn,
                hc_ffn_base: l.hc_ffn_base,
                hc_ffn_scale: l.hc_ffn_scale,
                ssm_q_conv: l.ssm_q_conv,
                ssm_k_conv: l.ssm_k_conv,
                ssm_v_conv: l.ssm_v_conv,
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wqkv: l.wqkv,
                ssm_f_a: l.ssm_f_a,
                ssm_f_b: l.ssm_f_b,
                ssm_beta: l.ssm_beta,
                ssm_a: l.ssm_a,
                ssm_dt_b: l.ssm_dt_b,
                ssm_g_a: l.ssm_g_a,
                ssm_g_b: l.ssm_g_b,
                ssm_o_norm: l.ssm_norm,
                wo: l.wo,
                attn_q_a_norm: l.attn_q_a_norm,
                attn_kv_a_norm: l.attn_kv_a_norm,
                wq_a: l.wq_a,
                wq_b: l.wq_b,
                wkv_a_mqa: l.wkv_a_mqa,
                wk_b: l.wk_b,
                wv_b: l.wv_b,
                indexer_k_norm: l.indexer_k_norm,
                indexer_k_norm_b: l.indexer_k_norm_b,
                indexer_proj: l.indexer_proj,
                indexer_attn_k: l.indexer_attn_k,
                indexer_attn_q_b: l.indexer_attn_q_b,
                indexer_kpool_gate: l.indexer_kpool_gate,
                indexer_kpool_ape: l.indexer_kpool_ape,
                ffn_gate: l.ffn_gate,
                ffn_down: l.ffn_down,
                ffn_up: l.ffn_up,
                ffn_gate_inp: l.ffn_gate_inp,
                ffn_exp_probs_b: l.ffn_exp_probs_b,
                ffn_gate_exps: l.ffn_gate_exps,
                ffn_down_exps: l.ffn_down_exps,
                ffn_up_exps: l.ffn_up_exps,
                ffn_gate_shexp: l.ffn_gate_shexp,
                ffn_down_shexp: l.ffn_down_shexp,
                ffn_up_shexp: l.ffn_up_shexp,
            })
            .collect(),
    }
}

/// glm5-next's params (glm5-next.cpp:7-59) — the KDA geometry takes the
/// n_embd_r 3*(d_conv-1)*d_inner branch, n_embd_s = head_dim²*n_head
/// (llama-hparams.cpp:216-223/:243-247)
fn glm5_params(hp: &LlamaHparams, n_trunk: usize, attn: AttnParams) -> graph_arch::Glm5NextParams {
    graph_arch::Glm5NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(0) as i64,
        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
        // the KDA branch of llama_hparams::n_embd_r/n_embd_s
        // (llama-hparams.cpp:216-223/:243-247)
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_lora_q: hp.n_lora_q as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_rot: hp.n_rot(0) as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_expert_shared: hp.n_expert_shared as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        indexer_kpool_select_tail: hp.indexer_kpool_select_tail,
        is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
        hc: hp.dsv4_hc_mult as i64,
        hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters,
        hc_eps: hp.dsv4_hc_eps,
        f_norm_eps: hp.f_norm_eps,
    }
}
