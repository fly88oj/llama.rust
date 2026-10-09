//! llama-server — port of tools/server (main.cpp, server.cpp,
//! server-context.cpp, server-http.cpp, server-schema.cpp, server-task.cpp).
//!
//! Deployment path of llama.cpp: an HTTP server with slot-based multi-sequence
//! decoding. The port implements the llama.cpp-native API surface —
//! `GET /health`, `GET /props`, `POST /completion`, `POST /completions`,
//! `POST /tokenize`, `POST /detokenize` — plus the OpenAI-compatible family:
//! `POST /v1/completions`, `POST /v1/chat/completions` (+ `/chat/completions`,
//! with `tools`/`tool_calls` through the ported jinja + autoparser path),
//! `POST /embedding`/`/embeddings`/`/v1/embeddings`, `GET /models`/`/v1/models`,
//! the `/rerank` family (the reference's "not supported" answer — no local
//! model carries rank pooling) and `GET /slots`.
//!
//! Not ported (documented in PARITY.md): the Responses/Anthropic/transcription
//! APIs, `/infill`, `/apply-template`, `/metrics`, the built-in web UI,
//! multimodal input, KV shifting and the idle-slot sleep/purge logic. The
//! `/slots` state endpoint (`POST /slots/{id}?action=save|restore|erase`,
//! server-context.cpp:5285-5390) is wired over the port's sequence-state API
//! (the plain/iswa KV rows + the dsv4 frame; the recurrent cells / dsa lid /
//! MSA idx are not serialized — those archs answer a clean error, see
//! PARITY.md). Speculative decoding
//! (server-context.cpp :1109-1137, :1259-1300, :2995-3075, :3742-3757,
//! :3897-4017) is wired for the `draft-simple` family with the reference's
//! per-slot state machine; the checkpoint (FULL-seq-rm) and synthetic-replay
//! branches are not (the port's KV cache always supports partial removal).

mod api;
mod chat;
mod engine;
mod http;
mod server_decision;
mod server_mcp;
mod server_tools;
mod subproc;
mod tls;
mod ui;
mod ui_assets;
mod weights;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use llama::context::DecodeContext;
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::json_schema::Json;
use llama::model::{load_model, LlamaModel};
use llama::speculative::{
    common_speculative_init, common_speculative_n_max_params,
    common_speculative_types_from_names, CommonParamsSpeculative,
};
use llama::vocab::Vocab;

use api::{TaskParams, ERROR_TYPE_INVALID_REQUEST, ERROR_TYPE_SERVER};
use engine::{Engine, Server, Slot, SlotState, Task, TaskKind};
use http::{Body, HttpServer, Request, Response, StreamEvent};
use server_tools::register_gcp_compat;

/// `llama_build_info()` — the reference stamps `system_fingerprint` and
/// `/props.build_info` with its own; the port keeps its build string there.
pub const BUILD_INFO: &str = "b0-bd4f514db";

/// `common_params` subset the port's server honours.
struct Args {
    model: String,
    host: String,
    port: u16,
    n_ctx: u32,
    /// `--parallel` (`-np`), -1 = auto (4, server.cpp:157-160)
    n_parallel: i32,
    n_predict: i32,
    n_batch: usize,
    n_ubatch: usize,
    n_threads: usize,
    flash_attn: bool,
    lora: Vec<(String, f32)>,
    /// `--embeddings` (`-fe`) — enable the embedding endpoints
    embeddings: bool,
    /// `--context-shift` / `--no-context-shift` (common/arg.cpp:1737-1741) —
    /// context shift on infinite text generation. Default **off** at the
    /// pinned revision (common.h:571 `bool ctx_shift = false`).
    context_shift: bool,
    /// `--keep N` (common/arg.cpp:1679-1685) — tokens to keep from the initial
    /// prompt when shifting; default 0, -1 = all (common.h:453). The request
    /// schema's `n_keep` default (server-schema.cpp:529).
    n_keep: i32,
    /// `--pooling {none,mean,cls,last}` — `LLAMA_POOLING_TYPE_UNSPECIFIED`
    /// unless given (llama-context.cpp:216-222 resolves it)
    pooling: llama::hparams::LlamaPoolingType,
    /// `--slot-save-path PATH` (common/arg.cpp:3610-3620) — the directory the
    /// /slots save/restore files live under; empty (the default) disables the
    /// endpoint ("This server does not support slots action. Start it with
    /// `--slot-save-path`", server-context.cpp:4773-4777). Must be an existing
    /// directory, and gets the trailing separator appended like the C.
    slot_save_path: String,
    /// `params.speculative` (common.h:370-401) — the `-md`/`--spec-*` surface
    /// of common/arg.cpp:4135-4253 (also honoured by the server example set)
    speculative: CommonParamsSpeculative,
    /// `--tools TOOL1,TOOL2,...` (arg.cpp:3412-3419) — the built-in server
    /// tools ("all" selects every one)
    tools: Vec<String>,
    /// `--tools-runtime OPTION` (arg.cpp:3421-3431) — the isolate every tool
    /// call runs through (`docker-container:<id>` / `podman-container:<id>` /
    /// `ssh:<target>`)
    tools_runtime: String,
    /// `--mcp-servers-config PATH` (arg.cpp:3433-3439)
    mcp_servers_config: String,
    /// `--mcp-servers-json JSON` (arg.cpp:3441-3447)
    mcp_servers_json: String,
    /// `--ui-mcp-proxy/--no-ui-mcp-proxy` (arg.cpp:3403-3409) — the CORS
    /// proxy route (`params.ui_mcp_proxy`)
    ui_mcp_proxy: bool,
    /// `--ui/--no-ui` (+deprecated `--webui/--no-webui`, arg.cpp:3464-3470)
    /// — the embedded web client (`params.ui`, default true, common.h:666)
    ui: bool,
    /// `--path PATH` (arg.cpp:3339-3345) — serve the web client from this
    /// directory instead of the embedded table (`params.public_path`)
    public_path: String,
    /// `--api-prefix PREFIX` (arg.cpp:3382-3388) — prefix path the server
    /// serves from (`params.api_prefix`, default "")
    api_prefix: String,
    // ---- the GPU surface (common/arg.cpp + the port's --ggml-libs resolver;
    // same wiring as llama-cli's, batch 20) ----
    /// `-ngl, --gpu-layers, --n-gpu-layers N` (arg.cpp:2786-2806, env
    /// LLAMA_ARG_N_GPU_LAYERS): an exact number, 'auto' (-1, the common.h:475
    /// default) or 'all' (-2). Negative values resolve to *every* layer at
    /// load time (`llama_model::n_gpu_layers`, llama-model.cpp:2002-2004)
    n_gpu_layers: i32,
    /// `-dev, --device <dev1,dev2,..>` (arg.cpp:2737-2745, env
    /// LLAMA_ARG_DEVICE) — the foreign backend device to offload to. The
    /// port's executor holds ONE GPU device (a comma list keeps its first
    /// name, matching the single-GPU layer split the emitter ports);
    /// `--device cpu` (the reference's CPU backend name) selects the foreign
    /// CPU backend without a device buffer
    device: Option<String>,
    /// `--list-devices` (arg.cpp:2746-2752) — print the devices the foreign
    /// build registers and exit (before any model work)
    list_devices: bool,
    /// `--ggml-libs DIR` — port-side: where the foreign ggml build lives (the
    /// reference resolves its backends from its own build; the port dlopens
    /// the pinned tree's)
    ggml_libs: Option<String>,
    /// `--foreign-cpu` — port-side: run everything on the foreign *CPU*
    /// backend even without -ngl (proves the emission layer without a GPU)
    foreign_cpu: bool,
    /// `--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N`
    /// (arg.cpp:4221-4239, `.set_examples({SPECULATIVE, SERVER, CLI})`, env
    /// LLAMA_ARG_N_GPU_LAYERS_DRAFT): same auto/-1 (the common.h:341 default)
    /// /all/-2 grammar, for the draft model. The draft context is created
    /// from the *speculative* params (`common_base_params_to_speculative`,
    /// speculative.cpp:2446-2470: `result.n_gpu_layers =
    /// params_spec.n_gpu_layers` — the draft does NOT inherit the main -ngl,
    /// it defaults to its own auto)
    n_gpu_layers_draft: i32,
    /// `--spec-draft-device, -devd, --device-draft` (arg.cpp:4212-4219,
    /// `.set_spec().set_examples({SPECULATIVE, SERVER, CLI})`): the draft's
    /// device; "default: follows --device" (the help text) =
    /// `common_base_params_to_speculative`'s `result = params` inheritance
    device_draft: Option<String>,
    /// `-ctxcp, --ctx-checkpoints, --swa-checkpoints N` (arg.cpp:1700-1707,
    /// env LLAMA_ARG_CTX_CHECKPOINTS) — max number of context checkpoints to
    /// create per slot (common.h:637 default 32). 0 disables the checkpoint
    /// machinery
    n_ctx_checkpoints: i32,
    /// `-cms, --checkpoint-min-step N` (arg.cpp:1708-1718, env
    /// LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT) — minimum spacing between
    /// context checkpoints in tokens (common.h:639 default 8192, 0 = no
    /// minimum; negative values are rejected like the C)
    checkpoint_min_step: i32,
}

/// `parse_csv_row` (common/arg.cpp:1347-1389) — the port's copy (same as
/// llama-cli's): comma-separated with `"…"` quoting and `""` escapes
fn parse_csv_row(input: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let chars: Vec<char> = input.chars().collect();

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' {
            if !in_quotes {
                if !field.is_empty() {
                    // quote in the middle of an unquoted field: literal
                    field.push('"');
                } else {
                    in_quotes = true;
                }
            } else if i + 1 < chars.len() && chars[i + 1] == '"' {
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

/// `res_403` (server.cpp:315-323) — the disabled-feature answer of
/// /tools and /cors-proxy
fn feature_disabled_handler() -> Arc<dyn Fn(&Request) -> Response + Send + Sync> {
    Arc::new(|_req: &Request| Response {
        status: 403,
        content_type: "application/json; charset=utf-8".into(),
        body: Body::Full(
            Json::Object(vec![(
                "error".into(),
                Json::Object(vec![
                    ("message".into(), Json::String("this feature is disabled".into())),
                    ("type".into(), Json::String("feature_disabled".into())),
                ]),
            )])
            .dump(),
        ),
        headers: Vec::new(),
        terminal_done: false,
    })
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        n_ctx: 4096,
        n_parallel: -1,
        n_predict: -1,
        n_batch: 2048,
        n_ubatch: 512,
        n_threads: 8,
            flash_attn: false,
            lora: Vec::new(),
            embeddings: false,
            context_shift: false, // common.h:571
            n_keep: 0,            // common.h:453
            pooling: llama::hparams::LlamaPoolingType::UNSPECIFIED,
            slot_save_path: String::new(), // arg.cpp:3612 default (unset)
            speculative: CommonParamsSpeculative::default(),
            tools: Vec::new(),           // arg.cpp:3415 default (no tools)
            tools_runtime: String::new(), // arg.cpp:3425 default (host)
            mcp_servers_config: String::new(),
            mcp_servers_json: String::new(),
            ui_mcp_proxy: false, // common.h default (disabled)
            ui: true,           // common.h:666 default (enabled)
            public_path: String::new(), // common.h:634 default
            api_prefix: String::new(),  // common.h:635 default
            // GPU surface: common.h:475 (`n_gpu_layers = -1` auto) and
            // common.h:341 (the draft's own -1 auto default)
            n_gpu_layers: -1,
            device: None,
            list_devices: false,
            ggml_libs: None,
            foreign_cpu: false,
            n_gpu_layers_draft: -1,
            device_draft: None,
            // common.h:637 / :639 defaults
            n_ctx_checkpoints: 32,
            checkpoint_min_step: 8192,
        };
    let mut host_given = false;
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].clone();
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            argv.get(*i).cloned().ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg.as_str() {
            "-m" | "--model" => a.model = value(&mut i)?,
            "--host" => {
                a.host = value(&mut i)?;
                host_given = true;
            }
            "--port" => a.port = value(&mut i)?.parse().map_err(|_| "bad --port")?,
            "-c" | "--ctx-size" => a.n_ctx = value(&mut i)?.parse().map_err(|_| "bad -c")?,
            "-np" | "--parallel" => a.n_parallel = value(&mut i)?.parse().map_err(|_| "bad -np")?,
            "-n" | "--n-predict" => a.n_predict = value(&mut i)?.parse().map_err(|_| "bad -n")?,
            "-b" | "--batch-size" => a.n_batch = value(&mut i)?.parse().map_err(|_| "bad -b")?,
            "-ub" | "--ubatch-size" => a.n_ubatch = value(&mut i)?.parse().map_err(|_| "bad -ub")?,
            "-t" | "--threads" => a.n_threads = value(&mut i)?.parse().map_err(|_| "bad -t")?,
            "-fa" | "--flash-attn" => {
                let v = value(&mut i)?;
                a.flash_attn = matches!(v.as_str(), "on" | "true" | "1" | "auto");
            }
            "--lora" => a.lora.push((value(&mut i)?, 1.0)),
            "--lora-scaled" => {
                let v = value(&mut i)?;
                let (path, scale) = v.rsplit_once(':').ok_or("--lora-scaled wants PATH:SCALE")?;
                a.lora.push((path.to_string(), scale.parse().map_err(|_| "bad lora scale")?));
            }
            // `--slot-save-path PATH` (common/arg.cpp:3610-3620): the
            // directory must exist ("Invalid value for --slot-save-path:
            // not a directory"), and the trailing separator is appended
            // ---- the tools / MCP / agent surface (arg.cpp:3403-3463) ----
            "--tools" => {
                // arg.cpp:3412-3419 — parse_csv_row of tool names
                let v = value(&mut i)?;
                a.tools = parse_csv_row(&v);
            }            "--tools-runtime" => a.tools_runtime = value(&mut i)?,
            "--mcp-servers-config" => a.mcp_servers_config = value(&mut i)?,
            "--mcp-servers-json" => a.mcp_servers_json = value(&mut i)?,
            "-ag" | "--agent" => {
                // arg.cpp:3450-3461 — enable CORS proxy + all built-in tools
                a.tools = vec!["all".to_string()];
                a.ui_mcp_proxy = true;
            }
            "-no-ag" | "--no-agent" => {
                a.tools.clear();
                a.ui_mcp_proxy = false;
            }
            "--ui-mcp-proxy" | "--webui-mcp-proxy" => a.ui_mcp_proxy = true,
            "--no-ui-mcp-proxy" | "--no-webui-mcp-proxy" => a.ui_mcp_proxy = false,
            // ---- the web client surface (arg.cpp:3339-3345 / :3382-3388 /
            //      :3464-3470) ----
            "--ui" | "--webui" => a.ui = true,
            "--no-ui" | "--no-webui" => a.ui = false,
            "--path" => a.public_path = value(&mut i)?,
            "--api-prefix" => a.api_prefix = value(&mut i)?,
            // ---- the GPU surface: the -ngl family every example honours
            //      (arg.cpp:2786-2806) + -dev/--list-devices
            //      (arg.cpp:2737-2752) + the spec-draft twins
            //      (arg.cpp:4212-4239) + the port's --ggml-libs resolver ----
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => {
                // arg.cpp:2790-2796 — 'auto' -> -1, 'all' -> -2, else stoi
                let v = value(&mut i)?;
                a.n_gpu_layers = match v.as_str() {
                    "auto" => -1,
                    "all" => -2,
                    _ => v.parse().map_err(|_| format!("invalid value for -ngl: '{v}'"))?,
                };
            }
            "-ngld" | "--spec-draft-ngl" | "--gpu-layers-draft" | "--n-gpu-layers-draft" => {
                // arg.cpp:4226-4231 — the same grammar, the draft's field
                let v = value(&mut i)?;
                a.n_gpu_layers_draft = match v.as_str() {
                    "auto" => -1,
                    "all" => -2,
                    _ => v
                        .parse()
                        .map_err(|_| format!("invalid value for -ngld: '{v}'"))?,
                };
            }
            "-dev" | "--device" => {
                // arg.cpp:2737-2745 — comma-separated list; the port's
                // executor holds one GPU device, so keep the first name (a
                // "none" list is the reference's "don't offload" spelling —
                // the port spells that by omitting --device)
                let v = value(&mut i)?;
                a.device = Some(v.split(',').next().unwrap_or("").trim().to_string());
            }
            "-devd" | "--spec-draft-device" | "--device-draft" => {
                // arg.cpp:4212-4219 — the draft's device list (same
                // single-device keep-first rule as --device above)
                let v = value(&mut i)?;
                a.device_draft = Some(v.split(',').next().unwrap_or("").trim().to_string());
            }
            "--list-devices" => a.list_devices = true,
            "--ggml-libs" => a.ggml_libs = Some(value(&mut i)?),
            "--foreign-cpu" => a.foreign_cpu = true,
            // ---- the context-checkpoint surface (arg.cpp:1700-1718; env
            //      vars are not read — the port's arg parser takes flags
            //      only, like every other flag here) ----
            "-ctxcp" | "--ctx-checkpoints" | "--swa-checkpoints" => {
                a.n_ctx_checkpoints = value(&mut i)?.parse().map_err(|_| "bad --ctx-checkpoints")?;
            }
            "-cms" | "--checkpoint-min-step" => {
                // arg.cpp:1713-1716 — negative values are rejected
                let v: i32 =
                    value(&mut i)?.parse().map_err(|_| "bad --checkpoint-min-step")?;
                if v < 0 {
                    return Err("invalid value for --checkpoint-min-step: must be non-negative"
                        .to_string());
                }
                a.checkpoint_min_step = v;
            }
            "--slot-save-path" => {                let v = value(&mut i)?;
                let dir = std::path::Path::new(&v);
                if !dir.is_dir() {
                    return Err(format!(
                        "invalid value for --slot-save-path: '{v}' is not a directory"
                    ));
                }
                a.slot_save_path = if v.ends_with(std::path::MAIN_SEPARATOR) {
                    v
                } else {
                    format!("{v}{}", std::path::MAIN_SEPARATOR)
                };
            }
            // `--embeddings` (`-fe`, common/arg.cpp) — the embedding endpoints
            "-fe" | "--embeddings" => a.embeddings = true,
            "--no-embeddings" => a.embeddings = false,
            // `--context-shift` / `--no-context-shift` (common/arg.cpp:
            // 1737-1741; default off at this revision, common.h:571)
            "--context-shift" => a.context_shift = true,
            "--no-context-shift" => a.context_shift = false,
            // `--keep N` (common/arg.cpp:1679-1685): "number of tokens to keep
            // from the initial prompt (default: 0, -1 = all)"
            "--keep" => {
                let v: i32 = value(&mut i)?.parse().map_err(|_| "bad --keep")?;
                a.n_keep = v;
            }
            // `--pooling {none,mean,cls,last}` (common/arg.cpp)
            "--pooling" => {
                let v = value(&mut i)?;
                a.pooling = match v.as_str() {
                    "none" | "0" => llama::hparams::LlamaPoolingType::NONE,
                    "mean" | "1" => llama::hparams::LlamaPoolingType::MEAN,
                    "cls" | "2" => llama::hparams::LlamaPoolingType::CLS,
                    "last" | "3" => llama::hparams::LlamaPoolingType::LAST,
                    _ => return Err(format!("invalid value for --pooling: {v}")),
                };
            }
            // ---- speculative decoding (common/arg.cpp:4135-4253; the server
            // is one of the example sets these flags are enabled for) ----
            "--spec-draft-model" | "-md" | "--model-draft" => {
                // arg.cpp:4236-4243
                a.speculative.draft.model_path = value(&mut i)?;
            }
            "--spec-type" => {
                // arg.cpp:4244-4253 — append the parsed list (the default
                // `{none}` stays; only the per-type bits matter)
                let v = value(&mut i)?;
                let names: Vec<String> = v.split(',').map(|s| s.to_string()).collect();
                match common_speculative_types_from_names(&names) {
                    Ok(types) => a.speculative.types.extend(types),
                    Err(e) => return Err(format!("--spec-type: {e}")),
                }
            }
            "--spec-draft-n-max" => {
                // arg.cpp:4135-4144
                let v: i32 =
                    value(&mut i)?.parse().map_err(|_| "--spec-draft-n-max: invalid value")?;
                if v < 0 {
                    return Err("--spec-draft-n-max: invalid value".into());
                }
                a.speculative.draft.n_max = v;
            }
            "--spec-draft-n-min" => {
                // arg.cpp:4145-4151
                let v: i32 =
                    value(&mut i)?.parse().map_err(|_| "--spec-draft-n-min: invalid value")?;
                a.speculative.draft.n_min = v;
            }
            "--spec-draft-p-min" | "--draft-p-min" => {
                // arg.cpp:4192-4198
                let v = value(&mut i)?;
                a.speculative.draft.p_min = stof(&v).map_err(|e| format!("--draft-p-min: {e}"))?;
            }
            "--spec-draft-p-split" | "--draft-p-split" => {
                // arg.cpp:4185-4191
                let v = value(&mut i)?;
                a.speculative.draft.p_split =
                    stof(&v).map_err(|e| format!("--draft-p-split: {e}"))?;
            }
            "--spec-draft-backend-sampling" => {
                // arg.cpp:4199-4207
                a.speculative.draft.backend_sampling = true;
            }
            "--no-spec-draft-backend-sampling" => {
                a.speculative.draft.backend_sampling = false;
            }
            "--spec-draft-sampling" => {
                // arg.cpp:4219-4233 (a7b94df2c): how the draft is sampled
                let v = value(&mut i)?;
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
                // arg.cpp:4152-4164
                let v = value(&mut i)?;
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
                // arg.cpp:4165-4183
                let v = value(&mut i)?;
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
            // removed params (arg.cpp:4382-4395)
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
            // ---- the ngram family's value flags (arg.cpp:4254-4378) ----
            "--spec-ngram-mod-n-min" => {
                a.speculative.ngram_mod.n_min = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-mod-n-min: invalid value".to_string()
                })?;
            }
            "--spec-ngram-mod-n-max" => {
                a.speculative.ngram_mod.n_max = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-mod-n-max: invalid value".to_string()
                })?;
            }
            "--spec-ngram-mod-n-match" => {
                a.speculative.ngram_mod.n_match = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-mod-n-match: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-size-n" => {
                a.speculative.ngram_simple.size_n = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-simple-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-size-m" => {
                a.speculative.ngram_simple.size_m = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-simple-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-simple-min-hits" => {
                a.speculative.ngram_simple.min_hits = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-simple-min-hits: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-size-n" => {
                a.speculative.ngram_map_k.size_n = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-size-m" => {
                a.speculative.ngram_map_k.size_m = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k-min-hits" => {
                a.speculative.ngram_map_k.min_hits = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k-min-hits: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-size-n" => {
                a.speculative.ngram_map_k4v.size_n = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k4v-size-n: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-size-m" => {
                a.speculative.ngram_map_k4v.size_m = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k4v-size-m: invalid value".to_string()
                })?;
            }
            "--spec-ngram-map-k4v-min-hits" => {
                a.speculative.ngram_map_k4v.min_hits = value(&mut i)?.parse().map_err(|_| {
                    "--spec-ngram-map-k4v-min-hits: invalid value".to_string()
                })?;
            }
            "-h" | "--help" => {
                println!(
                    "usage: llama-server -m MODEL [--host H] [--port P] [-c N] [-np N] [-n N]\n\
                     \x20                  [-b N] [-ub N] [-t N] [-fa on|off] [--lora PATH] [--lora-scaled PATH:SCALE]\n\
                     \x20                  [-md FNAME|--spec-draft-model FNAME] [--spec-type TYPES]\n\
                     \x20                  [--spec-draft-n-max N] [--spec-draft-n-min N] [--spec-draft-p-min P]\n\
                     \x20                  [-ngl N|--gpu-layers N] [-dev NAME|--device NAME] [--list-devices]\n\
                     \x20                  [-ngld N|--spec-draft-ngl N] [-devd NAME|--spec-draft-device NAME]\n\
                     \x20                  [--ggml-libs DIR] [--foreign-cpu]\n\
                     \n\
                     endpoints: GET /health, GET /props, POST /completion, POST /completions,\n\
                     \x20          POST /tokenize, POST /detokenize"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}' (try --help)")),
        }
        i += 1;
    }
    if a.model.is_empty() && !a.list_devices {
        // --list-devices may run bare: the reference's flag callback exits
        // inside the parser, before any param validation (arg.cpp:2746-2752)
        return Err("-m/--model is required".into());
    }
    if a.n_parallel == 0 {
        return Err("invalid value for n_parallel".into());
    }
    // security: the listener binds exactly what the user asked for; without
    // `--host` that is loopback only (server-http.cpp:120-160)
    if !host_given && !is_loopback(&a.host) {
        return Err("refusing to bind a non-loopback address without an explicit --host".into());
    }
    if host_given && !is_loopback(&a.host) {
        eprintln!(
            "llama-server: warning: binding {0} exposes the server beyond loopback",
            a.host
        );
    }
    Ok(a)
}

/// `GGML_PAD(x, 256)`
fn pad256(x: u32) -> u32 {
    x.div_ceil(256) * 256
}

fn is_loopback(host: &str) -> bool {
    host == "127.0.0.1"
        || host == "::1"
        || host == "localhost"
        || host.starts_with("127.")
}

/// `std::stof` (arg.cpp:4189/4196): parses the longest valid float prefix.
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

/// `AttnParams` of layer `il` — the same derivation llama-cli uses
/// (`llama-hparams::rope_runtime`), so the server and the CLI reach identical
/// numerics.
fn attn_params(hp: &llama::hparams::LlamaHparams, il: usize, use_flash_attn: bool) -> AttnParams {
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

/// The arch dispatch of the server — every arch llama-cli reaches (the dense
/// qwen2/llama/phi3/gemma family, the arch batches 1-8: gpt-oss, gemma4,
/// granite(+hybrid), lfm2moe, qwen35, the mamba family, the deepseek family
/// with its dsa/dsv4 driver forms, the batch-8 MoE long tail, ..., through the
/// batch 9-12 families plamo3 ... maple); the weight bundles live in
/// `weights.rs` (verbatim copies of llama-cli's wiring, so the two tools
/// drive the identical graph). `n_ctx_seq` is the slot-context budget the
/// reference's graph build sees through `cparams.n_ctx_seq`
/// (llama-context.cpp:289-297) — phimoe/deci resolve their long/short rope
/// factors against it (`get_rope_factors`, llama-model.cpp:2259-2272).
/// eurobert (batch 11b) is encoder-only and rides the `EncoderContext` branch
/// below like BERT.
fn forward_weights(
    model: &LlamaModel,
    use_flash_attn: bool,
    n_ctx_seq: u32,
) -> Result<(llama::context::ForwardWeights, AttnParams), String> {
    let hp = &model.hparams;
    let n_trunk = hp.n_layer() as usize;
    let mut attn = attn_params(hp, 0, use_flash_attn);
    let w = match model.arch {
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
            llama::context::ForwardWeights::Llama(llama::graph_arch::LlamaModelWeights {
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
        llama::arch::LlmArch::QWEN3 => {
            use llama::graph_arch::{Qwen3LayerWeights, Qwen3ModelWeights};
            llama::context::ForwardWeights::Qwen3(Qwen3ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers: model.layers[..n_trunk]
                    .iter()
                    .map(|l| Qwen3LayerWeights {
                        attn_norm: l.attn_norm.expect("attn_norm"),
                        wq: l.wq.expect("wq"),
                        wk: l.wk.expect("wk"),
                        wv: l.wv.expect("wv"),
                        wq_b: l.wq_b,
                        wk_b: l.wk_b,
                        wv_b: l.wv_b,
                        wo: l.wo.expect("wo"),
                        attn_q_norm: l.attn_q_norm.expect("attn_q_norm"),
                        attn_k_norm: l.attn_k_norm.expect("attn_k_norm"),
                        ffn_norm: l.ffn_norm.expect("ffn_norm"),
                        ffn_gate: l.ffn_gate.expect("ffn_gate"),
                        ffn_down: l.ffn_down.expect("ffn_down"),
                        ffn_up: l.ffn_up.expect("ffn_up"),
                    })
                    .collect(),
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
                final_softcap_unguarded: gemma2,
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
        llama::arch::LlmArch::OPENAI_MOE => {
            // gpt-oss: the builder needs `p` (head/rope geometry) *and* `gp`
            // (MoE width, per-layer SWA pattern + SWA rope copies) —
            // openai-moe.cpp:3-18 + llama-model.cpp:2251
            let gp = weights::gpt_oss_params(hp, n_trunk);
            llama::context::ForwardWeights::GptOss(weights::gpt_oss_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::GEMMA4 => {
            let gp = weights::gemma4_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Gemma4(weights::gemma4_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::GRANITE_HYBRID => {
            // attention geometry lives on the attention layers (layer 0 is the
            // mamba2 mixer) — granite-hybrid.cpp:143-198
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            let gp = weights::granite_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Granite(weights::granite_weights(model, n_trunk), gp)
        }
        llama::arch::LlmArch::LFM2 | llama::arch::LlmArch::LFM2MOE => {
            // the lfm2 build fork (lfm2.cpp:137-139): n_layer_decision > 0
            // loads the Decision form (no memory — create_memory's nullptr
            // arm, llama-model.cpp:2385-2387); lfm2moe ships no decision
            // files — the same fork llama-cli drives
            if model.arch == llama::arch::LlmArch::LFM2 && hp.n_layer_decision > 0 {
                llama::context::ForwardWeights::Lfm2Decision(
                    model.lfm2_decision_weights(),
                    model.lfm2_decision_params(),
                )
            } else {
                attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
                let gp = weights::lfm2_params(hp, n_trunk, attn);
                llama::context::ForwardWeights::Lfm2(weights::lfm2_weights(model, n_trunk), gp)
            }
        }
        llama::arch::LlmArch::QWEN35 => {
            // per-layer head geometry (the GDN layers differ from the attention
            // ones) + IMROPE/rope_sections — qwen35.cpp:165-268
            let gp = weights::qwen35_params(hp, n_trunk, attn);
            llama::context::ForwardWeights::Qwen35(weights::qwen35_weights(model, n_trunk), gp)
        }
        // ---- arch batch 1 cont. (2026-09-24) ----
        llama::arch::LlmArch::GPT2 => {
            // no rope: the graph gathers learned positions and never calls
            // ggml_rope_ext (gpt2.cpp:58-148); `attn` still carries the LayerNorm
            // eps, which is the norm `f_norm_eps` of gpt2.cpp:4
            attn.norm_eps = hp.f_norm_eps;
            llama::context::ForwardWeights::Gpt2(
                weights::gpt2_weights(model, n_trunk),
                llama::graph_arch::Gpt2Params { attn },
            )
        }
        llama::arch::LlmArch::PHI2 => {
            attn.norm_eps = hp.f_norm_eps; // phi2.cpp:4
            llama::context::ForwardWeights::Phi2(
                weights::phi2_weights(model, n_trunk),
                llama::graph_arch::Phi2Params { attn },
            )
        }
        llama::arch::LlmArch::STARCODER2 => {
            attn.norm_eps = hp.f_norm_eps; // starcoder2.cpp:4
            llama::context::ForwardWeights::StarCoder2(
                weights::starcoder2_weights(model, n_trunk),
                llama::graph_arch::StarCoder2Params { attn },
            )
        }
        llama::arch::LlmArch::COMMAND_R => {
            attn.norm_eps = hp.f_norm_eps; // command-r.cpp:5
            llama::context::ForwardWeights::CommandR(
                weights::command_r_weights(model, n_trunk),
                llama::graph_arch::CommandRParams {
                    attn,
                    logit_scale: hp.f_logit_scale,
                },
            )
        }
        llama::arch::LlmArch::GPTNEOX => {
            attn.norm_eps = hp.f_norm_eps; // gptneox.cpp:4
            llama::context::ForwardWeights::GptNeox(
                weights::gptneox_weights(model, n_trunk),
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
                weights::olmo2_weights(model, n_trunk),
                llama::graph_arch::Olmo2Params { attn },
            )
        }
        // ---- arch batch 2 (2026-09-25) ----
        llama::arch::LlmArch::CODESHELL => {
            attn.norm_eps = hp.f_norm_eps; // codeshell.cpp:4
            llama::context::ForwardWeights::Codeshell(
                weights::codeshell_weights(model, n_trunk),
                llama::graph_arch::CodeshellParams { attn },
            )
        }
        llama::arch::LlmArch::ORION => {
            attn.norm_eps = hp.f_norm_eps; // orion.cpp:4
            llama::context::ForwardWeights::Orion(
                weights::orion_weights(model, n_trunk),
                llama::graph_arch::OrionParams { attn },
            )
        }
        llama::arch::LlmArch::OLMO => {
            attn.norm_eps = hp.f_norm_eps; // olmo.cpp:4
            llama::context::ForwardWeights::Olmo(
                weights::olmo_weights(model, n_trunk),
                llama::graph_arch::OlmoParams {
                    attn,
                    f_clamp_kqv: hp.f_clamp_kqv,
                },
            )
        }
        llama::arch::LlmArch::XVERSE => {
            // xverse.cpp:4 — RMS; `attn_params` already carries the eps
            llama::context::ForwardWeights::Xverse(
                weights::xverse_weights(model, n_trunk),
                llama::graph_arch::XverseParams { attn },
            )
        }
        llama::arch::LlmArch::INTERNLM2 => llama::context::ForwardWeights::Internlm2(
            weights::internlm2_weights(model, n_trunk),
            llama::graph_arch::Internlm2Params { attn },
        ),
        llama::arch::LlmArch::EXAONE => llama::context::ForwardWeights::Exaone(
            weights::exaone_weights(model, n_trunk),
            llama::graph_arch::ExaoneParams { attn },
        ),
        llama::arch::LlmArch::GEMMA => {
            // gemma.cpp:41-139 — v1: Q is pre-scaled by 1/sqrt(n_embd_head_v)
            // (:86) and build_attn runs with kq_scale 1.0 (:91); the RMS eps
            // comes from meta.rs's GEMMA arm, so no override
            let attention_scale = 1.0 / (attn.n_embd_head_v as f32).sqrt();
            llama::context::ForwardWeights::Gemma1(
                weights::gemma1_weights(model, n_trunk),
                llama::graph_arch::Gemma1Params {
                    attn,
                    attention_scale,
                },
            )
        }
        llama::arch::LlmArch::FALCON => {
            attn.norm_eps = hp.f_norm_eps; // falcon.cpp:4
            llama::context::ForwardWeights::Falcon(
                weights::falcon_weights(model, n_trunk),
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
                weights::baichuan_weights(model, n_trunk),
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
                weights::bloom_weights(model, n_trunk),
                llama::graph_arch::BloomParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            )
        }
        llama::arch::LlmArch::MPT => {
            attn.norm_eps = hp.f_norm_eps; // mpt.cpp LayerNorm
            llama::context::ForwardWeights::Mpt(
                weights::mpt_weights(model, n_trunk),
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
                weights::starcoder_weights(model, n_trunk),
                llama::graph_arch::StarcoderParams { attn },
            )
        }
        llama::arch::LlmArch::REFACT => {
            // RMS norm; refact.cpp:12 — alibi 8.0 unconditional
            llama::context::ForwardWeights::Refact(
                weights::refact_weights(model, n_trunk),
                llama::graph_arch::RefactParams {
                    attn,
                    f_max_alibi_bias: hp.f_max_alibi_bias,
                },
            )
        }
        llama::arch::LlmArch::PLAMO => {
            // RMS norm — attn carries f_norm_rms_eps (plamo.cpp)
            llama::context::ForwardWeights::Plamo(
                weights::plamo_weights(model, n_trunk),
                llama::graph_arch::PlamoParams { attn },
            )
        }
        llama::arch::LlmArch::STABLELM => {
            attn.norm_eps = hp.f_norm_eps; // stablelm.cpp LayerNorm
            llama::context::ForwardWeights::Stablelm(
                weights::stablelm_weights(model, n_trunk),
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
                weights::granite_dense_weights(model),
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
                weights::qwen2moe_weights(model, n_trunk),
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
                weights::qwen3moe_weights(model, n_trunk),
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
            // against this run's slot-context budget (see phimoe_weights)
            llama::context::ForwardWeights::Phimoe(
                weights::phimoe_weights(model, n_trunk, n_ctx_seq, attn.n_ctx_orig),
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
                weights::arctic_weights(model, n_trunk),
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
                weights::olmoe_weights(model, n_trunk),
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
                weights::ernie45moe_weights(model, n_trunk),
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
                weights::smollm3_weights(model, n_trunk),
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
                weights::seed_oss_weights(model, n_trunk),
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
                weights::openelm_weights(model, n_trunk),
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
                weights::mamba_weights(model, n_trunk, mamba2),
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
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Jamba(
                weights::jamba_weights(model, n_trunk),
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
                weights::nemotron_h_weights(model, n_trunk),
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
                weights::deepseek2_weights(model, n_trunk),
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
                weights::deepseek_weights(model, n_trunk),
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
                weights::deepseek2_weights(model, n_trunk),
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
                weights::deepseek4_weights(model, n_trunk),
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
                weights::nemotron_weights(model, n_trunk),
                llama::graph_arch::NemotronParams { attn },
            )
        }
        llama::arch::LlmArch::GROK => {
            // grok.cpp:3-33 — the scale/softcap keys with their old-GGUF
            // defaults land in the hparams; the graph applies the GROK kq
            // softcap only in the non-FA branch (llama-graph.cpp:2682-2689)
            llama::context::ForwardWeights::Grok(
                weights::grok_weights(model, n_trunk),
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
                weights::chameleon_weights(model, n_trunk),
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
                weights::deci_weights(model, n_trunk, n_ctx_seq, attn.n_ctx_orig),
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
                weights::jais_weights(model, n_trunk),
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
                weights::falcon_h1_weights(model, n_trunk),
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
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Plamo2(
                weights::plamo2_weights(model, n_trunk),
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
                weights::hunyuan_moe_weights(model, n_trunk),
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
                weights::dots1_weights(model, n_trunk),
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
                weights::bailingmoe_weights(model, n_trunk),
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
                weights::bailingmoe2_weights(model, n_trunk),
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
                weights::glm4_moe_weights(model, n_trunk),
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
                weights::minimax_m2_weights(model, n_trunk),
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
                weights::cohere2moe_weights(model, n_trunk),
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
                weights::exaone_moe_weights(model, n_trunk),
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
                weights::plamo3_weights(model, n_trunk),
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
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Qwen3Next(
                weights::qwen3next_weights(model, n_trunk),
                weights::qwen3next_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::KIMI_LINEAR => {
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::KimiLinear(
                weights::kimi_linear_weights(model, n_trunk),
                weights::kimi_linear_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::BAILINGMOE3 => {
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::BailingMoe3(
                weights::bailingmoe3_weights(model, n_trunk),
                weights::bailingmoe3_params(hp, n_trunk, attn),
            )
        }
        // ---- arch batch 10 (2026-10): the small-arch + EXP-op batch ----
        llama::arch::LlmArch::SMALLTHINKER => {
            llama::context::ForwardWeights::Smallthinker(
                weights::smallthinker_weights(model, n_trunk),
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
                weights::llada_moe_weights(model, n_trunk),
                llama::graph_arch::LladaMoeParams {
                    attn,
                    n_expert: hp.n_expert as i64,
                    n_expert_used: hp.n_expert_used(0) as i64,
                },
            )
        }
        llama::arch::LlmArch::MINIMAX_01 => {
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::Minimax01(
                weights::minimax01_weights(model, n_trunk),
                weights::minimax01_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::GRANITE_SWITCH => {
            llama::context::ForwardWeights::GraniteSwitch(
                weights::graniteswitch_weights(model, n_trunk),
                weights::graniteswitch_params(hp, n_trunk, attn),
            )
        }
        // ---- arch batch 11a (2026-10): the long-tail queue, first half ----
        llama::arch::LlmArch::APERTUS => {
            llama::context::ForwardWeights::Apertus(
                weights::apertus_weights(model, n_trunk),
                weights::apertus_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::GROVEMOE => {
            llama::context::ForwardWeights::Grovemoe(
                weights::grovemoe_weights(model, n_trunk),
                weights::grovemoe_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::QWEN35MOE => {
            llama::context::ForwardWeights::Qwen35Moe(
                weights::qwen35moe_weights(model, n_trunk),
                weights::qwen35moe_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::KIMI_K3 => {
            // the MLA geometry (attention.key_length = [kv_lora|rope] x
            // head_count_kv = 1) of the first non-KDA layer, kimi-linear style
            attn = attn_params(hp, weights::first_attn_layer(hp, n_trunk), use_flash_attn);
            llama::context::ForwardWeights::KimiK3(
                weights::kimi_k3_weights(model, n_trunk),
                weights::kimi_k3_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::DOTS3NOTE => {
            llama::context::ForwardWeights::Dots3Note(
                weights::dots3note_weights(model, n_trunk),
                weights::dots3note_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::MINIMAX_M3 => {
            llama::context::ForwardWeights::MinimaxM3(
                weights::minimax_m3_weights(model, n_trunk),
                weights::minimax_m3_params(hp, n_trunk, attn),
            )
        }
        llama::arch::LlmArch::QWEN4EXP => {
            llama::context::ForwardWeights::Qwen4Exp(
                weights::qwen4exp_weights(model, n_trunk),
                weights::qwen4exp_params(hp, n_trunk, attn),
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
                weights::glm5_weights(model, n_trunk),
                weights::glm5_params(hp, n_trunk, a),
            )
        }
        // ---- batch 20 (the c35b66744 sync): k2-horizon (dense + MoVA) —
        // the same construction llama-cli drives ----
        llama::arch::LlmArch::K2_HORIZON => {
            llama::context::ForwardWeights::K2Horizon(
                model.k2_horizon_weights(),
                llama::graph_arch::K2HorizonParams {
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
                },
            )
        }
        // ---- arch batch 11b (2026-10): the long-tail queue, second half ----
        llama::arch::LlmArch::ARCEE => {
            llama::context::ForwardWeights::Arcee(
                weights::arcee_weights(model, n_trunk),
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
                weights::jais2_weights(model, n_trunk),
                llama::graph_arch::Jais2Params { attn },
            )
        }
        llama::arch::LlmArch::TALKIE => {
            llama::context::ForwardWeights::Talkie(
                weights::talkie_weights(model, n_trunk),
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
                weights::nanbeige_weights(model, n_all),
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
                weights::dream_weights(model, n_trunk),
                llama::graph_arch::DreamParams { attn },
            )
        }
        llama::arch::LlmArch::RND1 => {
            llama::context::ForwardWeights::Rnd1(
                weights::rnd1_weights(model, n_trunk),
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
                weights::hrm_text_weights(model, n_slot),
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
                weights::laguna_weights(model, n_trunk),
                llama::graph_arch::LagunaParams {
                    attn,
                    n_head: (0..n_trunk).map(|il| hp.n_head(il) as i64).collect(),
                    n_head_kv: (0..n_trunk).map(|il| hp.n_head_kv(il) as i64).collect(),
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    has_swa: hp.swa_type != llama::hparams::LlamaSwaType::NONE
                        && hp.is_swa_any(),
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
                weights::maple_weights(model, n_trunk),
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
                weights::cohere2_weights(model, n_trunk),
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
                weights::chatglm_weights(model, n_trunk),
                llama::graph_arch::ChatglmParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                },
            )
        }
        llama::arch::LlmArch::BITNET => {
            llama::context::ForwardWeights::Bitnet(
                weights::bitnet_weights(model, n_trunk),
                llama::graph_arch::BitnetParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                },
            )
        }
        llama::arch::LlmArch::DBRX => {
            llama::context::ForwardWeights::Dbrx(
                weights::dbrx_weights(model, n_trunk),
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
            // arch guard produces exactly that)
            llama::context::ForwardWeights::Ernie45Moe(
                weights::ernie45moe_weights(model, n_trunk),
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
                weights::mistral3_weights(model, n_trunk, n_ctx_seq),
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
            let mut a = attn;
            a.n_head_kv = a.n_head;
            llama::context::ForwardWeights::Minicpm3(
                weights::minicpm3_weights(model, n_trunk, n_ctx_seq),
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
                weights::glm4_weights(model, n_trunk),
                llama::graph_arch::Glm4Params {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::EXAONE4 => {
            llama::context::ForwardWeights::Exaone4(
                weights::exaone4_weights(model, n_trunk),
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
                weights::llama4_weights(model, n_trunk),
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
                weights::qwen2vl_weights(model, n_trunk),
                llama::graph_arch::Qwen2VlParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::QWEN3VL | llama::arch::LlmArch::QWEN3VLMOE => {
            llama::context::ForwardWeights::Qwen3Vl(
                weights::qwen3vl_weights(model, n_trunk),
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
            let mut a = attn;
            a.n_embd_head_k = hp.n_embd_head_k(0) as i64; // kv_lora_rank + qk_rope
            a.n_head_kv = 1;
            a.n_rot = hp.n_rot(0) as i64;
            llama::context::ForwardWeights::GlmDsa(
                weights::glm_dsa_weights(model, n_trunk),
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
            llama::context::ForwardWeights::Rwkv6(
                weights::rwkv6_weights(model, n_trunk, false),
                weights::rwkv6_params(hp),
            )
        }
        llama::arch::LlmArch::RWKV6QWEN2 => {
            llama::context::ForwardWeights::Rwkv6Qwen2(
                weights::rwkv6_weights(model, n_trunk, true),
                weights::rwkv6_params(hp),
            )
        }
        llama::arch::LlmArch::RWKV7 => {
            llama::context::ForwardWeights::Rwkv7(
                weights::rwkv7_weights(model, n_trunk, false),
                weights::rwkv7_params(hp),
            )
        }
        llama::arch::LlmArch::ARWKV7 => {
            llama::context::ForwardWeights::Arwkv7(
                weights::rwkv7_weights(model, n_trunk, true),
                weights::rwkv7_params(hp),
            )
        }
        llama::arch::LlmArch::GEMMA3N => {
            llama::context::ForwardWeights::Gemma3n(
                weights::gemma3n_weights(model, n_trunk),
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
                batch15_s_qwen1(model, n_trunk),
                llama::graph_arch::Qwen1Params { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::MAINCODER => {
            llama::context::ForwardWeights::Maincoder(
                batch15_s_maincoder(model, n_trunk),
                llama::graph_arch::MaincoderParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::PANGU_EMBED => {
            llama::context::ForwardWeights::PanguEmbed(
                batch15_s_pangu_embed(model, n_trunk),
                llama::graph_arch::PanguEmbedParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::COGVLM => {
            llama::context::ForwardWeights::Cogvlm(
                batch15_s_cogvlm(model, n_trunk),
                llama::graph_arch::CogvlmParams { attn, norm_rms_eps: hp.f_norm_rms_eps },
            )
        }
        llama::arch::LlmArch::SPARK2_5 => {
            llama::context::ForwardWeights::Spark25(
                batch15_s_spark25(model, n_trunk),
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
                batch15_s_muse_glimmer(model, n_trunk),
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
                batch15_s_llada(model, n_trunk),
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
                batch15_s_plm(model, n_trunk),
                llama::graph_arch::PlmParams {
                    attn: a,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    kv_lora_rank: hp.n_lora_kv as i64,
                },
            )
        }
        llama::arch::LlmArch::HUNYUAN_VL | llama::arch::LlmArch::HUNYUAN_DENSE => {
            llama::context::ForwardWeights::HunyuanVl(
                batch15_s_hunyuan_vl(model, n_trunk),
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
                batch15_s_granite_swa(model, n_trunk),
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
                batch15_s_afmoe(model, n_trunk),
                llama::graph_arch::AfmoeParams {
                    n_no_rope_layer_step: hp.n_no_rope_layer_step,
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    is_swa: (0..n_trunk).map(|il| hp.is_swa(il)).collect(),
                    freq_base: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_base_train_swa } else { hp.rope_freq_base_train })
                        .collect(),
                    freq_scale: (0..n_trunk)
                        .map(|il| if hp.is_swa(il) { hp.rope_freq_scale_train_swa } else { hp.rope_freq_scale_train })
                        .collect(),
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
                batch15_s_mellum(model, n_trunk),
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
                batch15_s_paddleocr(model, n_trunk),
                llama::graph_arch::PaddleOcrParams {
                    attn,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    rope_sections: hp.rope_sections,
                },
            )
        }
        llama::arch::LlmArch::HY_V3 => {
            llama::context::ForwardWeights::HyV3(
                batch15_s_hy_v3(model, n_trunk),
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
                batch15_s_mimo2(model, n_trunk),
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
                batch15_s_step35(model, n_trunk),
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
                batch15_s_hy_v4(model, n_trunk),
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
                weights::deepseek2_weights(model, n_trunk),
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
                "arch '{}' ({other:?}) is not wired into llama-server (see FILE_MAP.md's \
                 architecture matrix)",
                other.name()
            ))
        }
    };
    Ok((w, attn))
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("llama-server: {e}");
            std::process::exit(1);
        }
    };

    // `--list-devices` (arg.cpp:2746-2752): print the devices and exit before
    // any model work. The port resolves the device registry from the foreign
    // ggml build (llama-cli's --list-devices twin)
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
                eprintln!("llama-server: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let shutdown = Arc::new(AtomicBool::new(false));

    // The model loads before the routes are published: /health answers 503
    // "Loading model" meanwhile (`middleware_server_state`, server-http.cpp:
    // 307-325), so the listener starts first for readiness probes.
    let mut routes_table = HttpServer::new();
    // `path_prefix = params.api_prefix` (server-http.cpp:114) — every route
    // registered below mounts under the prefix (server-http.cpp:673/724/745)
    routes_table.set_path_prefix(&args.api_prefix);
    routes_table.ready.store(false, Ordering::Relaxed);
    let ready = routes_table.ready.clone();

    // The engine owns the (non-`Send`) model context and runs on this thread;
    // the connection threads only see the task queue, the vocabulary and the
    // precomputed props (engine::Server).
    let (mut engine, vocab, model_meta, chat_templates_init, gpu_mode_ran) =
        match load_engine(&args, shutdown.clone()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("llama-server: failed to load the model: {e}");
                std::process::exit(1);
            }
        };
    let props = props_json(&engine);
    let server = Arc::new(engine::Server {
        queue: engine::Queue::new(),
        vocab: vocab.clone(),
        props: props.dump(),
        n_slots: engine.slots.len(),
        n_processing: std::sync::atomic::AtomicUsize::new(0),
        n_predict_default: args.n_predict,
        n_keep_default: args.n_keep,
        speculative_types: llama::speculative::common_speculative_type_name_str(
            &args.speculative.types,
        ),
        next_task_id: std::sync::atomic::AtomicI64::new(0),
        embeddings: args.embeddings,
        pooling: engine.pooling,
        // `params_base.chat_params` (server-common.h:83-110) — the inputs of
        // `common_chat_templates_init`; the handler thread rebuilds the
        // templates from them (the library's parsed template holds `Rc`
        // values and is not shareable across connection threads)
        chat_templates_init: chat_templates_init,
        // `params_base.enable_reasoning != 0 && template_supports_thinking`
        // (server-context.cpp:1448-1453; the port has no --reasoning flag, so
        // the default -1 = auto applies)
        enable_thinking: engine.enable_thinking,
        model_meta,
        slot_save_path: args.slot_save_path.clone(),
        slots_json: std::sync::RwLock::new("[]".to_string()),
    });

    register_routes(&mut routes_table, &server);
    let routes_cell: Arc<std::sync::Mutex<Option<http::Routes>>> =
        Arc::new(std::sync::Mutex::new(None));

    // ---- the tools / MCP / GCP / CORS-proxy surface (server.cpp:310-375,
    // server-http.cpp:797-923, server-cors-proxy.h) ----
    // `mcp_mgr.start(params)` runs the warmup before any route publishes
    let mut mcp_local = server_mcp::ServerMcp::new();
    if let Err(e) = mcp_local.start(&args.mcp_servers_config, &args.mcp_servers_json) {
        eprintln!("llama-server: MCP starting failed: {e}");
        std::process::exit(1);
    }
    let mcp_mgr = Arc::new(mcp_local);

    let tools_enabled = !args.tools.is_empty() || !mcp_mgr.is_empty();
    let tools: Arc<Vec<String>> = if tools_enabled {
        match server_tools::setup(&args.tools, &mcp_mgr, &args.tools_runtime) {
            Ok(t) => {
                if !args.tools.is_empty() {
                    eprintln!("warning: server tools (experimental) enabled");
                }
                if !args.tools_runtime.is_empty() {
                    eprintln!("warning: tools runtime (experimental) enabled");
                }
                if !mcp_mgr.is_empty() {
                    eprintln!("warning: MCP servers (experimental) enabled");
                }
                Arc::new(t)
            }
            Err(e) => {
                eprintln!("llama-server: tools setup failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        Arc::new(Vec::new())
    };

    // /tools GET+POST — the live handlers, or the disabled-feature 403
    // (server.cpp:315-323, :360-374)
    {
        let tools = tools.clone();
        let mcp_mgr = mcp_mgr.clone();
        let handler = if tools_enabled {
            Arc::new(move |req: &Request| server_tools::handle_tools_get(&tools, &mcp_mgr, req))
                as Arc<dyn Fn(&Request) -> Response + Send + Sync>
        } else {
            feature_disabled_handler()
        };
        routes_table.add("GET", "/tools", handler);
    }
    {
        let tools = tools.clone();
        let mcp_mgr = mcp_mgr.clone();
        let tools_runtime = Arc::new(args.tools_runtime.clone());
        let handler = if tools_enabled {
            Arc::new(
                move |req: &Request| {
                    server_tools::handle_tools_post(&tools, &tools_runtime, &mcp_mgr, req)
                },
            ) as Arc<dyn Fn(&Request) -> Response + Send + Sync>
        } else {
            feature_disabled_handler()
        };
        routes_table.add("POST", "/tools", handler);
    }

    // /cors-proxy GET+POST (server.cpp:337-344, server-cors-proxy.h)
    for method in ["GET", "POST"] {
        let handler = if args.ui_mcp_proxy {
            eprintln!("warning: MCP proxy (experimental) enabled");
            Arc::new(move |req: &Request| crate::server_tools::cors_proxy(req))
                as Arc<dyn Fn(&Request) -> Response + Send + Sync>
        } else {
            feature_disabled_handler()
        };
        routes_table.add(method, "/cors-proxy", handler);
    }

    // the embedded web client + the --path mount (server-http.cpp:360-478).
    // The routes ride `add`'s `path_prefix + path` (the reference's
    // `srv->Get(params.api_prefix + ...)`, server-http.cpp:447-466); the
    // mount is registered straight on the listener, prefix and all
    // (server-http.cpp:379-385).
    ui::register_ui(&mut routes_table, &args.api_prefix, &args.public_path, args.ui);

    // Google Cloud Platform (Vertex AI) compat — server-http.cpp:797-923:
    // AIP_MODE/AIP_HEALTH_ROUTE/AIP_PREDICT_ROUTE/AIP_HTTP_PORT. The handler
    // dispatches per-instance through the routing table, handed over via
    // `routes_cell` right after `into_routes`
    register_gcp_compat(&mut routes_table, &routes_cell);
    let routes = routes_table.into_routes();
    *routes_cell.lock().unwrap() = Some(routes.clone());
    {
        let host = args.host.clone();
        // `gcp_params`' port override (server-http.cpp:117-124): AIP_MODE
        // pins the listen port to AIP_HTTP_PORT (default 8080)
        let gcp = server_tools::GcpParams::from_env();
        let port = if gcp.enabled {
            if args.port != gcp.port {
                eprintln!(
                    "Google Cloud Platform compat: overriding server port {} with AIP_HTTP_PORT {}",
                    args.port, gcp.port
                );
            }
            gcp.port
        } else {
            args.port
        };
        let ready = ready.clone();
        std::thread::spawn(move || {
            if let Err(e) = http::serve_routes(&host, port, routes, ready) {
                eprintln!("llama-server: failed to bind {host}:{port}: {e}");
                std::process::exit(1);
            }
        });
    }
    ready.store(true, Ordering::Relaxed);

    // `queue_tasks.on_update_slots` in the main thread; SIGINT's default action
    // terminates the process like the reference's.
    engine.run(&server);

    // [TAG_EMIT_EXIT] (llama-cli's epilogue): a dlopen'ed libgomp (the foreign
    // CPU side of the scheduler) leaves worker threads whose dynamic TLS
    // glibc's exit handlers tear down before the threads stop — a SIGSEGV at
    // process exit. _exit skips those handlers; the listener is gone by now.
    if gpu_mode_ran {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        ggml::sysffi::exit_now(0);
    }
}

/// The model + weights + DecodeContext wiring (server-context.cpp:1145-1270
/// `server_context_impl::init`). Returns the engine (which must stay on this
/// thread — the sampler chains are not `Send`), the vocabulary and the chat
/// templates' init inputs (`common_chat_templates_init`,
/// server-context.cpp:1455).
/// whether the GPU surface asks for the foreign executor at all — the same
/// trigger llama-cli uses (`-ngl N > 0 || --device || --foreign-cpu`). A bare
/// `-ngl auto/all` (the -1/-2 defaults) does NOT trigger without a device:
/// the reference's `act_gpu_layers = devices.empty() ? 0 : …`
/// (llama-model.cpp:1590) resolves auto to *zero* offloaded layers when no
/// GPU device exists, which is the port's plain CPU engine
fn gpu_requested(args: &Args) -> bool {
    args.n_gpu_layers > 0 || args.device.is_some() || args.foreign_cpu
}

/// assemble the executor's config from the parsed flags (llama-cli's
/// wiring): resolve the lib dir, map `--device cpu` to the foreign CPU
/// backend, and apply the reference's ngl resolution — a negative value
/// (auto/-1, all/-2) is `n_layer_all + 1` (every layer,
/// llama-model.cpp:2002-2004), capped to nothing when no GPU device is
/// selected (the `act_gpu_layers` guard above)
fn emit_config(args: &Args, device: &Option<String>, ngl_raw: i32) -> ggml::backend_emit::EmitConfig {
    let lib_dir = args.ggml_libs.clone().unwrap_or_else(|| {
        eprintln!(
            "gpu mode needs the foreign ggml build dir: pass --ggml-libs <dir> \
             (the reference build's bin/, holding libggml-base.so and libggml-*.so)"
        );
        std::process::exit(1);
    });
    let mut cfg = ggml::backend_emit::EmitConfig::new(&lib_dir);
    // `--device cpu` = the reference's CPU backend by name: foreign CPU
    // kernels, no device buffer (the reference's libggml-cpu via DL)
    cfg.device = match device.as_deref() {
        Some(d) if d.eq_ignore_ascii_case("cpu") => None,
        d => d.map(|s| s.to_string()),
    };
    cfg.n_gpu_layers = match cfg.device {
        // no GPU device: act_gpu_layers == 0 (llama-model.cpp:1590) — the
        // foreign CPU executor with everything host-bound
        None => 0,
        // auto/all -> every layer (llama-model.cpp:2002-2004); any value
        // >= n_layer_all + 1 is equivalent under layer_on_gpu's truncation
        Some(_) if ngl_raw < 0 => i32::MAX / 2,
        Some(_) => ngl_raw,
    };
    cfg.n_threads = args.n_threads;
    cfg
}

#[allow(clippy::type_complexity)]
fn load_engine(
    args: &Args,
    shutdown: Arc<AtomicBool>,
) -> Result<(Engine, Arc<Vocab>, ModelMeta, llama::chat_tools::ChatTemplatesInit, bool), String> {
    let t0 = std::time::Instant::now();
    let gguf = ggml::Gguf::open(&args.model).map_err(|e| format!("failed to open gguf: {e}"))?;
    let f = std::fs::File::open(&args.model).map_err(|e| e.to_string())?;
    // SAFETY: read-only usage of a model file
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? });
    let vocab = Arc::new(Vocab::load(&gguf).map_err(|e| format!("vocab: {e}"))?);
    let model = load_model(&gguf, mmap).map_err(|e| format!("model: {e}"))?;

    // `n_parallel` auto (server.cpp:156-160)
    let n_parallel = if args.n_parallel < 0 { 4 } else { args.n_parallel };

    // ---- the decision model setup (common.cpp:1232-1252 +
    // server-context.cpp:1165-1171, upstream a7b94df2c) ----
    // these decision models return a score for each token via the embeddings
    // output, so embedding mode is forced on (pooling NONE)
    let decision_type = server_decision::common_get_decision_type(&gguf);
    // (common.cpp:1285-1290, a657f7e98: LFM2_D1_OMNI joins the embeddings family)
    let decision_reads_embd = matches!(
        decision_type,
        server_decision::CommonDecisionType::Laya
            | server_decision::CommonDecisionType::Kev
            | server_decision::CommonDecisionType::Clef
            | server_decision::CommonDecisionType::Lfm2D1Omni
    );
    let embeddings = args.embeddings || decision_reads_embd;
    let pooling_arg = if decision_reads_embd {
        llama::hparams::LlamaPoolingType::NONE
    } else {
        args.pooling
    };
    // embeddings need the whole batch in one ubatch, so n_batch must not be
    // larger than n_ubatch (common.cpp:1244-1252)
    let n_batch = if embeddings && args.n_batch > args.n_ubatch {
        eprintln!(
            "llama-server: embeddings enabled: setting n_batch = n_ubatch = {}",
            args.n_ubatch
        );
        args.n_ubatch
    } else if embeddings {
        args.n_batch.min(args.n_ubatch)
    } else {
        args.n_batch
    };
    // `decision.init(model_tgt)` (server-context.cpp:1165-1171): the loader
    // rejects the model when its decision metadata is invalid
    let decision = if decision_type != server_decision::CommonDecisionType::None {
        match server_decision::ServerDecisionContext::init(&gguf, vocab.clone()) {
            Ok(dc) => Some(dc),
            Err(e) => return Err(format!("failed to init decision model: {e}")),
        }
    } else {
        None
    };
    // the `/models` meta (`server_context_meta`, server-context.cpp:4185-4232)
    let model_size: u64 = {
        let ctx = &model.ctx;
        // `llama_model_size` = the loader's `n_bytes` — the sum of every
        // tensor's storage bytes (llama-model-loader.cpp:592)
        model.tensors.values().map(|&id| ctx.nbytes(id) as u64).sum()
    };
    let model_n_params: u64 = {
        let ctx = &model.ctx;
        // `llama_model_n_params` — the sum of every tensor's element count
        model
            .tensors
            .values()
            .map(|&id| ctx.ne(id).iter().map(|&d| d.max(0) as u64).product::<u64>())
            .sum()
    };
    let model_hparams_n_ctx_train = model.hparams.n_ctx_train;
    let model_hparams_n_embd = model.hparams.n_embd_inp();
    let resolved_pooling = llama::context::resolve_pooling(pooling_arg, model.hparams.pooling_type);

    // an encoder-only model (BERT / eurobert / t5encoder) runs through
    // `llama_encode` (`EncoderContext`); everything else through the decode
    // graph
    let mut spec = None;
    // `cparams.n_rs_seq` of the decode arm, hoisted for the Engine's
    // seq_rm capability facts below (the decode arm re-derives it locally)
    let n_rs_seq_eng: u32 = if args
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
    // set by the decode arm below when the foreign executor takes over (the
    // encoder arm refuses gpu_requested above, so it stays false there)
    let mut gpu_mode_ran = false;
    let is_encoder = matches!(
        model.arch,
        llama::arch::LlmArch::BERT
            | llama::arch::LlmArch::EUROBERT
            | llama::arch::LlmArch::T5ENCODER
            // arch batch 15 — gemma-embedding (the symmetric-SWA encoder)
            // + llama-embed (the causal no-cache graph<true>)
            | llama::arch::LlmArch::GEMMA_EMBEDDING
            | llama::arch::LlmArch::LLAMA_EMBED
            // batch 20 — gemma-embedding2 (the null-memory embedding family:
            // create_memory returns nullptr, llama-model.cpp:2402 — decode
            // reroutes to encode, llama-context.cpp:1729-1732)
            | llama::arch::LlmArch::GEMMA_EMBEDDING2
    );
    // the lfm2 d1-omni decision model (a657f7e98): `decision.block_count > 0`
    // on the non-causal trunk makes lfm2.cpp:139 pick `graph_decision` and
    // `create_memory` return nullptr (llama-model.cpp:2383-2386) — a
    // null-memory graph whose output is the [3, T] score tensor
    // (`res->t_embd`). The reference drives it on ctx_tgt through
    // llama_decode's "no memory -> encode()" reroute
    // (llama-context.cpp:1729-1732); the port's DecodeContext builds the
    // hybrid-memory decode graph instead, so the model rides its own
    // stateless decision core (engine.rs `D1OmniCore`, the
    // tests/lfm2_decision_e2e.rs driver). (A causal lfm2 decision file —
    // d1-3B — keeps the plain decode core: it has a vocab head and no
    // decision blocks.)
    let is_lfm2_decision = model.arch == llama::arch::LlmArch::LFM2
        && !model.hparams.causal_attn
        && model.hparams.n_layer_decision > 0;
    let core = if is_lfm2_decision {
        if !args.lora.is_empty() {
            return Err("--lora is not wired for decision models".into());
        }
        if gpu_requested(&args) {
            return Err(
                "gpu mode is not wired for decision models in the port (the decision core has \
                 no backend_emit emission path)"
                    .into(),
            );
        }
        let w = weights::lfm2_decision_weights(&model);
        let p = weights::lfm2_decision_params(&model);
        let n_embd_out = model.hparams.n_embd_out() as usize;
        let gctx = model.ctx;
        engine::Core::Decision(engine::D1OmniCore {
            gctx,
            w,
            p,
            n_embd_out,
            n_threads: args.n_threads,
        })
    } else if is_encoder {
        if !args.lora.is_empty() {
            return Err("--lora is not wired for encoder models".into());
        }
        // the EncoderContext has no foreign-executor path — GPU flags on an
        // embedding model would silently degrade to CPU, so refuse them
        if gpu_requested(&args) {
            return Err(
                "gpu mode is not wired for encoder/embedding models in the port (the \
                 EncoderContext has no backend_emit emission path)"
                    .into(),
            );
        }
        // eurobert ropes its Q/K (eurobert.cpp:65-75 — the EurobertRope facts
        // the batch-11b tests assembled the same way)
        let euro = matches!(
            model.arch,
            llama::arch::LlmArch::EUROBERT | llama::arch::LlmArch::LLAMA_EMBED
        )
        .then(|| {
            let rope = model.hparams.rope_runtime();
            llama::graph_arch::EurobertRope {
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
        // llama-embed's graph<true> runs the CAUSAL no-cache mask
        let causal = model.arch == llama::arch::LlmArch::LLAMA_EMBED;
        // gemma-embedding's symmetric-SWA facts (gemma-embedding.cpp:3-28);
        // gemma-embedding2 reads the same window pair (its loader also sets
        // swa_type = SYMMETRIC, gemma-embedding2.cpp:4-6)
        let gemma_swa = matches!(
            model.arch,
            llama::arch::LlmArch::GEMMA_EMBEDDING | llama::arch::LlmArch::GEMMA_EMBEDDING2
        )
        .then(|| {
            let hp = &model.hparams;
            let rope = hp.rope_runtime();
            let gr = llama::graph_arch::EurobertRope {
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
                llama::graph_arch::GemmaEmbeddingSwa {
                    is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                    n_swa: hp.n_swa,
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                },
                gr,
                hp.f_attention_scale,
            )
        });
        let params = llama::graph_arch::EncoderParams {
            n_head: model.hparams.n_head(0) as i64,
            n_head_kv: model.hparams.n_head_kv(0) as i64,
            n_embd_head: model.hparams.n_embd_head_k(0) as i64,
            n_rel_attn_bkts: model.hparams.n_rel_attn_bkts,
            f_norm_eps: model.hparams.f_norm_eps,
            f_norm_rms_eps: model.hparams.f_norm_rms_eps,
            pool: resolved_pooling,
            euro_rope: euro,
            gemma_swa,
            causal,
        };
        let enc_w = match model.arch {
            llama::arch::LlmArch::EUROBERT => {
                llama::context::EncoderWeights::Eurobert(model.eurobert_weights())
            }
            // t5encoder.cpp:45 — the loader-only twin of t5's encoder half
            // (models.h: `using graph = llama_model_t5::graph<true>`)
            llama::arch::LlmArch::T5ENCODER => {
                llama::context::EncoderWeights::T5Encoder(model.t5_encoder_weights())
            }
            llama::arch::LlmArch::GEMMA_EMBEDDING => {
                llama::context::EncoderWeights::GemmaEmbedding(model.gemma_embedding_weights())
            }
            // gemma-embedding2 — the bundled params carry the per-layer-input
            // facts + the SWA rope pair (the ge2 test assembly,
            // tests/gemma_embedding2_e2e.rs); text embeddings only — the
            // vision/audio halves ride mtmd, not ported
            llama::arch::LlmArch::GEMMA_EMBEDDING2 => {
                let hp = &model.hparams;
                let rope = hp.rope_runtime();
                let p = llama::graph_arch::GemmaEmbedding2Params {
                    n_head: hp.n_head(0) as i64,
                    n_head_kv: hp.n_head_kv(0) as i64,
                    n_embd_head: hp.n_embd_head_k(0) as i64,
                    norm_rms_eps: hp.f_norm_rms_eps,
                    n_embd_per_layer: hp.n_embd_per_layer as i64,
                    f_attention_scale: hp.f_attention_scale,
                    is_swa: (0..hp.n_layer() as usize).map(|il| hp.is_swa(il)).collect(),
                    freq_base_swa: hp.rope_freq_base_train_swa,
                    freq_scale_swa: hp.rope_freq_scale_train_swa,
                    rope: llama::graph_arch::EurobertRope {
                        n_rot: hp.n_rot(0) as i32,
                        // llama_model_rope_type(GEMMA_EMBEDDING2) = NEOX
                        // (llama-model.cpp:3184)
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
                llama::context::EncoderWeights::LlamaEmbed(weights::llama_embed_weights(&model))
            }
            _ => llama::context::EncoderWeights::Bert(model.bert_weights()),
        };
        engine::Core::Encode(llama::context::EncoderContext::new(
            model.ctx,
            enc_w,
            params,
            args.n_threads,
        ))
    } else {
        // `cparams.n_ctx_seq` (llama-context.cpp:289-297): the padded whole
        // pool with the unified cache (`-np` auto), else pad(n_ctx/n_parallel)
        // — the per-slot budget the graph build resolves rope factors against
        let n_ctx_seq = if args.n_parallel < 0 {
            pad256(args.n_ctx)
        } else {
            pad256(args.n_ctx / n_parallel.max(1) as u32).max(1)
        };
        let (weights, attn) = forward_weights(&model, args.flash_attn, n_ctx_seq)?;

        // LoRA adapters load into the model context before the DecodeContext,
        // so their tensors stay below the graph watermark (llama-cli's
        // pattern; common/common.cpp:1342-1358 / :1507-1508)
        let mut gctx = model.ctx;
        let mut loras = Vec::new();
        if !args.lora.is_empty() {
            let dims: HashMap<String, [i64; 4]> =
                model.tensors.iter().map(|(n, &id)| (n.clone(), *gctx.ne(id))).collect();
            for (path, scale) in &args.lora {
                let adapter = llama::adapter::load_adapter_lora(
                    &mut gctx,
                    model.arch,
                    &|name| dims.get(name).copied(),
                    path,
                )
                .map_err(|_| format!("failed to load lora adapter '{path}'"))?;
                loras.push((adapter, *scale));
            }
        }

        // the iswa split for SWA models (llama-model.cpp:2687-2690).
        // Deepseek32/Deepseek4 carry their own cache forms (dsa-lid / dsv4,
        // built inside `DecodeContext::new_impl`) whose precedence in the C's
        // switch beats the iswa split — their hparams DO carry swa keys, so
        // they must stay on the `new_with` path (same exclusion as llama-cli).
        let hp = &model.hparams;
        let own_cache = matches!(
            weights,
            llama::context::ForwardWeights::Deepseek32(..)
                | llama::context::ForwardWeights::Deepseek4(..)
                // glm-dsa carries its own dsa-lid cache pair (batch 13)
                | llama::context::ForwardWeights::GlmDsa(..)
        );
        let swa = (!own_cache
            && hp.swa_type != llama::hparams::LlamaSwaType::NONE
            && hp.is_swa_any())
        .then(|| llama::kv_cache::SwaCacheSpec::from_hparams(hp));
        // `cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)` (llama-context.cpp:290)
        // — the KV cache (and `llama_n_ctx`) see the padded value, so the slot
        // budget below and the cache size stay consistent for a `-c` that is
        // not a multiple of 256
        let n_ctx_padded = pad256(args.n_ctx);
        // `cparams.n_rs_seq = params.speculative.need_n_rs_seq()`
        // (common.cpp:1723): the rollback types widen the recurrent/dsv4
        // state; everything else keeps 0
        let n_rs_seq = if args
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
        // mut: `common_speculative_init` may flip the nextn tap on it
        // (draft-mtp, speculative.cpp:1420)
        let mut dctx = if let llama::context::ForwardWeights::Deepseek4(..) = weights {
            // the compressed dsv4 half needs one stream per sequence
            // (`unified_compressed = false` forced, llama-kv-cache-dsv4.cpp:
            // 1287) — the server decodes `n_parallel` sequences, so the
            // context is built with n_seq_max = n_parallel and the reference's
            // n_rs_seq (llama-cli's single-sequence `new_with` path is what
            // the CLI's arm does)
            DecodeContext::new_with_dsv4(
                gctx,
                weights,
                attn,
                n_ctx_padded,
                args.n_threads,
                args.n_ubatch,
                n_parallel as u32,
                n_rs_seq,
            )
        } else {
            match swa {
                Some(spec) => DecodeContext::new_with_swa(
                    gctx,
                    weights,
                    attn,
                    n_ctx_padded,
                    args.n_threads,
                    args.n_ubatch,
                    spec,
                ),
                None => DecodeContext::new_with(
                    gctx,
                    weights,
                    attn,
                    n_ctx_padded,
                    args.n_threads,
                    args.n_ubatch,
                ),
            }
        }
        // every branch arms the recurrent rollback ring — a draft that
        // replays rejected tokens (MTP/eagle3/dflash/dspark) otherwise lets
        // the verify batch pollute the GDN state past rejected drafts
        // (cparams.n_rs_seq, common.cpp:1635; 0 = the unarmed default; dsv4
        // re-asserts its constructional value)
        .with_rs_rollback(n_rs_seq)
        .with_embeddings(args.embeddings, resolved_pooling);

        // ---- GPU mode (llama-cli's wiring, batch 20): -ngl N > 0 / --device /
        // --foreign-cpu drive the decode graphs on a dlopen'ed foreign ggml
        // through the backend_emit translator; weights and KV land per the
        // reference's layer-split rule (llama-model.cpp:1521) ----
        if gpu_requested(&args) {
            let mut cfg = emit_config(&args, &args.device, args.n_gpu_layers);
            eprintln!(
                "llama-server: gpu mode: device {}, n_gpu_layers {} (auto/all resolve to every \
                 layer, llama-model.cpp:2002-2004)",
                cfg.device.as_deref().unwrap_or("<foreign-cpu>"),
                args.n_gpu_layers
            );
            if let Err(e) = dctx.enable_gpu(cfg) {
                return Err(format!("failed to enable gpu backends: {e}"));
            }
            gpu_mode_ran = true;
        }
        llama::adapter::set_adapters_lora(&dctx.gctx, &loras).map_err(|e| e.to_string())?;

        // ---- speculative decoding (server-context.cpp:1109-1300): load the
        // draft model (`common_speculative_init_from_params`,
        // speculative.cpp:2523-2604) and init the speculator
        // (`common_speculative_init`, :1261) ----
        let has_spec = common_speculative_n_max_params(&args.speculative) > 0;
        // `--spec-type mtp` needs the target model's own MTP block (the
        // has_draft arm of speculative.cpp:2557-2576 loads `params.model.path`
        // — the *target* file — and the no-draft arm :2577-2589 shares
        // model_tgt). Only the deepseek family carries ported graph_mtp
        // builders, and those trunks are not wired into llama-server's
        // `forward_weights` (batch 6/7 parity drove llama-cli) — the MTP
        // draft context is therefore reported as unwired here instead of
        // silently degrading. (integrator item, PARITY.md)
        // the gemma4-assistant head — the `-md <assistant.gguf>` +
        // `--spec-type draft-mtp` pair whose draft file is a gemma4-assistant
        // GGUF: the head ATTACHES to the target context (the port's
        // `ctx_other == ctx_tgt`, llama-context.cpp:147-153) and drafts over
        // the shared target KV (`is_mem_shared`, speculative.cpp:1423) — no
        // draft context exists, `common_speculative_init` gets the
        // gemma4_shared flag below
        let spec_mtp = args
            .speculative
            .types
            .contains(&llama::speculative::CommonSpeculativeType::DraftMtp);
        let mut gemma4_dft_vocab: Option<llama::vocab::Vocab> = None;
        let mut gemma4_shared = false;
        if has_spec
            && spec_mtp
            && args.speculative.has_dft()
            && ggml::Gguf::open(args.speculative.draft.model_path.clone())
                .ok()
                .and_then(|g| {
                    g.find_key("general.architecture")
                        .and_then(|v| v.as_str().map(str::to_string))
                })
                .as_deref()
                == Some("gemma4-assistant")
        {
            let path = args.speculative.draft.model_path.clone();
            eprintln!("llama-server: attaching gemma4-assistant draft head '{path}'");
            let gguf_dft = ggml::Gguf::open(&path)
                .map_err(|e| format!("failed to load draft model, '{path}': {e}"))?;
            let vocab_dft = llama::vocab::Vocab::load(&gguf_dft)
                .map_err(|e| format!("draft vocab: {e}"))?;
            let mmap_dft = {
                let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                // SAFETY: read-only usage of a model file
                Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
            };
            // the share map's target geometry (llama-model.cpp:2698-2703)
            let (tgt_full, tgt_swa) = match &dctx.weights {
                llama::context::ForwardWeights::Gemma4(_, p) => {
                    let n = p.is_swa.len() - 1;
                    (
                        (p.n_embd_head_k[n] as i64, p.n_head_kv[n] as i64),
                        (p.n_embd_head_k[n - 1] as i64, p.n_head_kv[n - 1] as i64),
                    )
                }
                _ => {
                    return Err(
                        "gemma4-assistant requires a gemma4 target model (-m)".into(),
                    );
                }
            };
            dctx
                .attach_gemma4_assistant(&gguf_dft, mmap_dft, args.flash_attn, tgt_full, tgt_swa)
                .map_err(|e| format!("failed to attach the gemma4-assistant head: {e}"))?;
            let head = dctx.gemma4_assistant.as_ref().unwrap();
            if head.params.n_embd_backbone as usize != dctx.n_embd_out() {
                return Err(format!(
                    "draft-mtp: MTP input row width must match the target h_nextn width ({} != \
                     {})",
                    head.params.n_embd_backbone,
                    dctx.n_embd_out()
                ));
            }
            eprintln!(
                "llama-server: gemma4-assistant: n_embd = {}, backbone = {}, layers = {} \
                 (shared target KV)",
                head.params.n_embd,
                head.params.n_embd_backbone,
                head.params.is_swa.len()
            );
            gemma4_shared = true;
            gemma4_dft_vocab = Some(vocab_dft);
        }
        if has_spec && spec_mtp && !gemma4_shared {
            return Err(
                "--spec-type mtp: the deepseek-family MTP draft context is wired in llama-cli, \
                 not in llama-server (the server's deepseek trunk graphs are not wired; the \
                 gemma4-assistant head IS wired — pass -md <assistant.gguf>; see PARITY.md)"
                    .into(),
            );
        }
        if has_spec {
            // `common_speculative_init_from_params` loads the draft model only
            // when `has_draft` (speculative.cpp:2533-2544); without a draft
            // path the speculator reports "draft-simple requires a draft
            // context" below, exactly like the C
            let path = args.speculative.draft.model_path.clone();
            // `--spec-type draft-dflash / draft-dspark` — the dflash draft
            // context (speculative.cpp:2553-2576): the dual-mode decoder over
            // the dflash draft file, `cparams.ctx_other = ctx_tgt`
            let spec_dflash = args
                .speculative
                .types
                .contains(&llama::speculative::CommonSpeculativeType::DraftDflash)
                || args
                    .speculative
                    .types
                    .contains(&llama::speculative::CommonSpeculativeType::DraftDspark);
            // `--spec-type draft-eagle3` — the head context
            // (speculative.cpp:2553-2576): encoder + one-layer decoder over
            // the eagle head file, `cparams.ctx_other = ctx_tgt`
            let spec_eagle = args
                .speculative
                .types
                .contains(&llama::speculative::CommonSpeculativeType::DraftEagle3);
            // the gemma4-assistant head rides the target context — no
            // draft context to build, only its vocabulary for the samplers
            let (ctx_dft, vocab_dft) = if gemma4_shared {
                (None, gemma4_dft_vocab.take())
            } else if spec_dflash {
                eprintln!("llama-server: loading draft model '{path}'");
                let gguf_dft = ggml::Gguf::open(&path)
                    .map_err(|e| format!("failed to load draft model, '{path}': {e}"))?;
                let vocab_dft = llama::vocab::Vocab::load(&gguf_dft)
                    .map_err(|e| format!("draft vocab: {e}"))?;
                let gguf_tgt = ggml::Gguf::open(&args.model)
                    .map_err(|e| format!("failed to open the target model for the dflash draft: {e}"))?;
                let mmap_dft = {
                    let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                    // SAFETY: read-only usage of a model file
                    Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
                };
                let mmap_tgt = {
                    let f = std::fs::File::open(&args.model).map_err(|e| e.to_string())?;
                    // SAFETY: read-only usage of a model file
                    Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
                };
                let draft = llama::dflash::load_dflash_draft(
                    &gguf_dft,
                    mmap_dft,
                    &gguf_tgt,
                    mmap_tgt,
                    vocab_dft.n_tokens() as i64,
                    args.flash_attn,
                )
                .map_err(|e| format!("draft model: {e}"))?;
                eprintln!(
                    "llama-server: draft: arch = dflash, extract_layers = {:?}, n_embd = {}, \
                     block_size = {}, dspark = {}",
                    draft.params.target_layer_ids,
                    draft.params.n_embd,
                    draft.params.block_size,
                    draft.weights.dspark_markov_w1.is_some()
                );
                // `cparams.n_ctx = llama_n_ctx(ctx_tgt)` (speculative.cpp:2550)
                let stub = llama::dflash::dflash_trunk_stub(&draft.weights);
                let ctx_dft = DecodeContext::new_dflash(
                    draft.ctx,
                    llama::context::ForwardWeights::Qwen2(stub),
                    (draft.weights, draft.params),
                    vocab_dft.n_tokens() as usize,
                    dctx.n_ctx(),
                    args.n_threads,
                    args.n_ubatch,
                )
                    .with_rs_rollback(n_rs_seq);
                (Some(ctx_dft), Some(vocab_dft))
            } else if spec_eagle {
                eprintln!("llama-server: loading draft model '{path}'");
                let gguf_dft = ggml::Gguf::open(&path)
                    .map_err(|e| format!("failed to load draft model, '{path}': {e}"))?;
                let vocab_dft = llama::vocab::Vocab::load(&gguf_dft)
                    .map_err(|e| format!("draft vocab: {e}"))?;
                let gguf_tgt = ggml::Gguf::open(&args.model)
                    .map_err(|e| format!("failed to open the target model for the eagle head: {e}"))?;
                let mmap_dft = {
                    let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                    // SAFETY: read-only usage of a model file
                    Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
                };
                let mmap_tgt = {
                    let f = std::fs::File::open(&args.model).map_err(|e| e.to_string())?;
                    // SAFETY: read-only usage of a model file
                    Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
                };
                let head = llama::eagle::load_eagle3_head(
                    &gguf_dft,
                    mmap_dft,
                    &gguf_tgt,
                    mmap_tgt,
                    vocab_dft.n_tokens() as i64,
                    args.flash_attn,
                )
                .map_err(|e| format!("draft model: {e}"))?;
                eprintln!(
                    "llama-server: draft: arch = eagle3, extract_layers = {:?}, n_embd = {}, \
                     target_hidden_size = {}",
                    head.params.target_layer_ids, head.params.n_embd, head.params.n_embd_tgt
                );
                // `cparams.n_ctx = llama_n_ctx(ctx_tgt)` (speculative.cpp:2550)
                let stub = llama::eagle::eagle_trunk_stub(&head.weights);
                let ctx_dft = DecodeContext::new_eagle3(
                    head.ctx,
                    llama::context::ForwardWeights::Qwen2(stub),
                    (head.weights, head.params),
                    vocab_dft.n_tokens() as usize,
                    dctx.n_ctx(),
                    args.n_threads,
                    args.n_ubatch,
                )
                // the draft replays rejected tokens through the same ring
                // (`cparams.n_rs_seq`, common.h:396-404 — batch 20b's
                // llama-server handover item)
                .with_rs_rollback(n_rs_seq);
                (Some(ctx_dft), Some(vocab_dft))
            } else if args.speculative.has_dft() {
                eprintln!("llama-server: loading draft model '{path}'");
                let gguf_dft = ggml::Gguf::open(&path)
                    .map_err(|e| format!("failed to load draft model, '{path}': {e}"))?;
                let vocab_dft = llama::vocab::Vocab::load(&gguf_dft)
                    .map_err(|e| format!("draft vocab: {e}"))?;
                let mmap_dft = {
                    let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                    // SAFETY: read-only usage of a model file
                    Arc::new(unsafe { memmap2::Mmap::map(&f).map_err(|e| e.to_string())? })
                };
                let model_dft =
                    load_model(&gguf_dft, mmap_dft).map_err(|e| format!("draft model: {e}"))?;
                // the draft's rope factors resolve against the target context
                // (`cparams.n_ctx = llama_n_ctx(ctx_tgt)`, speculative.cpp:2550)
                let (weights_dft, attn_dft) =
                    forward_weights(&model_dft, args.flash_attn, dctx.n_ctx())?;
                eprintln!(
                    "llama-server: draft: arch = {}, n_layer = {}, vocab = {}",
                    model_dft.arch.name(),
                    model_dft.hparams.n_layer(),
                    vocab_dft.id_to_token.len()
                );
                // the draft context holds as many tokens per sequence as the
                // target context (`cparams.n_ctx = llama_n_ctx(ctx_tgt)`,
                // speculative.cpp:2550)
                let ctx_dft = DecodeContext::new_with(
                    model_dft.ctx,
                    weights_dft,
                    attn_dft,
                    dctx.n_ctx(),
                    args.n_threads,
                    args.n_ubatch,
                )
                // same ring as the target (common.h:396-404, batch 20b
                // handover)
                .with_rs_rollback(n_rs_seq);
                (Some(ctx_dft), Some(vocab_dft))
            } else {
                (None, None)
            };
            // ---- GPU mode on the draft context — the reference's device
            // placement: the draft model/context is created from the
            // *speculative* params (`common_base_params_to_speculative`,
            // speculative.cpp:2446-2470), whose n_gpu_layers is the draft's
            // own -ngld (default auto = every layer, never the main -ngl) and
            // whose devices default to the top-level --device (the `result =
            // params` inheritance; -devd overrides). Applies to every draft
            // head the port builds: eagle3, dflash/dspark and draft-simple ----
            let mut ctx_dft = ctx_dft;
            let dft_device = args.device_draft.clone().or_else(|| args.device.clone());
            if (gpu_mode_ran || args.n_gpu_layers_draft > 0 || args.device_draft.is_some())
                && dft_device.is_some()
            {
                if let Some(dft) = ctx_dft.as_mut() {
                    let cfg = emit_config(&args, &dft_device, args.n_gpu_layers_draft);
                    eprintln!(
                        "llama-server: gpu mode (draft): device {}, n_gpu_layers {} (the \
                         -ngld surface, arg.cpp:4221-4239)",
                        cfg.device.as_deref().unwrap_or("<foreign-cpu>"),
                        args.n_gpu_layers_draft
                    );
                    dft.enable_gpu(cfg)
                        .map_err(|e| format!("failed to enable gpu backends on the draft: {e}"))?;
                }
            }
            // `common_speculative_init(params_base.speculative,
            // params_base.n_parallel)` (server-context.cpp:1261) — an
            // exception fails the load unless it was only a synth request;
            // `nullptr` drops the draft context again (:1277-1281). The
            // target context is handed in for draft-mtp's embeddings_nextn
            // tap (speculative.cpp:1420).
            match common_speculative_init(
                &args.speculative,
                n_parallel as u32,
                &mut dctx,
                ctx_dft,
                &vocab,
                vocab_dft.as_ref().or(gemma4_dft_vocab.as_ref()),
                0,
                gemma4_shared,
            ) {
                Err(e) => {
                    return Err(format!(
                        "failed to initialize speculative decoding context: {e}"
                    ));
                }
                Ok(None) => {
                    eprintln!("llama-server: no speculative decoding context (no implementation)");
                }
                Ok(Some(s)) => spec = Some(s),
            }
            if let Some(s) = spec.as_ref() {
                if !s.synth_probs.is_empty() {
                    return Err(
                        "synthetic acceptance replay is not ported in llama-server (see PARITY.md)"
                            .into(),
                    );
                }
            }
        }

        let nn = dctx.n_vocab() as i32;
        let nv = vocab.id_to_token.len() as i32;
        if nn != nv {
            eprintln!("llama-server: warning: vocab size {nv} != graph output rows {nn}");
        }

        engine::Core::Decode(dctx)
    };
    let n_vocab = vocab.id_to_token.len() as i32;

    // `n_ctx_slot()` (server-context.cpp:4027-4035) = `llama_n_ctx_seq` capped
    // by the training context. The reference reaches it through
    // `cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx / cparams.n_seq_max, 256)`
    // (llama-context.cpp:289-297, the non-unified branch — an explicit `-np`
    // leaves `kv_unified` false); with `-np` at its auto default the reference
    // sets `kv_unified` and the slot budget is the whole pool.
    // The port's KV cache is always one unified pool of `n_ctx` cells.
    let n_ctx_seq = if args.n_parallel < 0 {
        pad256(args.n_ctx)
    } else {
        pad256(args.n_ctx / n_parallel.max(1) as u32).max(1)
    };
    let slot_n_ctx = n_ctx_seq.min(pad256(args.n_ctx));
    let slots: Vec<Slot> = (0..n_parallel)
        .map(|i| Slot {
            id: i,
            state: SlotState::Idle,
            n_ctx: slot_n_ctx,
            prompt_tokens: Vec::new(),
            checkpoints: std::collections::VecDeque::new(),
            task: None,
            sampler: None,
            grammar: None,
            stats: api::GenStats::default(),
            sampled: 0,
            i_batch: -1,
            generated: llama::chat_tools::ChatInput::default(),
            generated_tokens: Vec::new(),
            n_sent_text: 0,
            has_next_token: false,
            has_new_line: false,
            stop: api::StopType::None,
            stopping_word: String::new(),
            truncated: false,
            probs_output: Vec::new(),
            n_predict_max: args.n_predict,
            t_last_used: 0,
            sent_begin: false,
            chat: None,
            lazy: None,
            spec_draft_q: Vec::new(),
            spec_draft: Vec::new(),
            spec_i_batch: Vec::new(),
            prev: None,
        })
        .collect();

    let ftype = gguf.get_u32("general.file_type").map(api::ftype_name).unwrap_or_else(|| "unknown".into());
    let bos = vocab.token_bos();
    let eos = vocab.token_eos();
    let bos_token = api::token_piece(&vocab, bos, true);
    let eos_token = api::token_piece(&vocab, eos, true);

    // `common_chat_templates_init(model_tgt, params_base.chat_template)`
    // (server-context.cpp:1455; chat.cpp:757-855): the GGUF's
    // `tokenizer.chat_template` (+ `.tool_use`), the BOS/EOS pieces and the
    // vocab's add_bos/add_eos flags. The template sources the /props endpoints
    // report are the *verbatim* GGUF values (`common_chat_templates_source`).
    let chat_templates_init = llama::chat_tools::ChatTemplatesInit {
        chat_template_override: gguf.get_str("tokenizer.chat_template").unwrap_or("").to_string(),
        chat_template_tool_use: gguf
            .get_str("tokenizer.chat_template.tool_use")
            .unwrap_or("")
            .to_string(),
        bos_token: bos_token.clone(),
        eos_token: eos_token.clone(),
        add_bos: vocab.add_bos,
        add_eos: vocab.add_eos,
    };
    // The reference never refuses to *load* a model over its chat template
    // (the full minja engine parses everything the converter emits; a broken
    // template only fails chat requests later). The port's mini-jinja supports
    // a subset, so a parse failure degrades the same way: no templates, a
    // warning, and `/v1/chat/completions` reports the parse error per request.
    let chat_templates = match llama::chat_tools::ChatTemplates::init(&chat_templates_init) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "llama-server: warning: chat template parsing error: {e} \
                 (chat endpoints will fail; other endpoints are unaffected)"
            );
            llama::chat_tools::ChatTemplates::init(&llama::chat_tools::ChatTemplatesInit {
                // an empty source parses to the passthrough template
                chat_template_override: String::new(),
                chat_template_tool_use: String::new(),
                bos_token: bos_token.clone(),
                eos_token: eos_token.clone(),
                add_bos: vocab.add_bos,
                add_eos: vocab.add_eos,
            })
            .expect("empty template parses")
        }
    };
    // "thinking is enabled if: 1. It's not explicitly disabled via
    // --reasoning off 2. The chat template supports it"
    // (server-context.cpp:1448-1453) — the port has no --reasoning flag, so
    // only the template probe decides
    let enable_thinking = chat::templates_support_enable_thinking(&chat_templates);
    // the reference's jinja lexer normalises the template source it stores
    // (`this->src = lexer_res.source`, chat.h:63 — minja strips the trailing
    // newline); the port's mini-jinja keeps the raw GGUF value, so the /props
    // string is trimmed to match (rendering uses the raw source, which the
    // library's parity tests pin byte-for-byte)
    let chat_template = chat_templates.source("").trim_end_matches('\n').to_string();
    let chat_template_tool_use = chat_templates.source("tool_use").trim_end_matches('\n').to_string();
    // `common_chat_templates_get_caps` (chat.cpp:1525-1533) — the caps of the
    // more expressive template
    let chat_template_caps = Json::Object(
        chat_templates
            .get_caps()
            .into_iter()
            .map(|(k, v)| (k, Json::Bool(v)))
            .collect(),
    );

    // the checkpoint machinery facts: `n_swa = params_base.swa_full ? 0 :
    // llama_model_n_swa(model_tgt)` (server-context.cpp:1337; the port has
    // no --swa-full flag) and `common_context_can_seq_rm` resolved to
    // FULL/RS (common.cpp:1553-1596 — a recurrent half refuses partial
    // seq_rm without the rollback ring, n_rs_seq > 0 bounds either memory)
    let model_n_swa = model.hparams.n_swa as i32;
    let seq_rm_bounded = match &core {
        engine::Core::Decode(d) => {
            d.weights.recurrent_dims().is_some() || n_rs_seq_eng > 0
        }
        _ => false,
    };

    let engine = Engine {
        core,
        vocab: vocab.clone(),
        n_vocab,
        slots,
        pending: Default::default(),
        deferred: Default::default(),
        n_batch,
        n_predict_default: args.n_predict,
        model_name: args.model.clone(),
        model_path: args.model.clone(),
        ctx_shift: args.context_shift, // common/arg.cpp:1737-1741
        n_keep_default: args.n_keep,   // common.h:453 / server-schema.cpp:529
        n_ctx_checkpoints: args.n_ctx_checkpoints, // common.h:637
        checkpoint_min_step: args.checkpoint_min_step, // common.h:639
        n_swa: model_n_swa,
        seq_rm_bounded,
        shutdown: shutdown.clone(),
        spec,
        decision,
        pooling: resolved_pooling,
        ftype,
        media_marker: format!("<__media_{}__>", random_string(32)),
        chat_template,
        chat_template_caps,
        chat_template_tool_use,
        bos_token,
        eos_token,
        enable_thinking,
    };
    // the `/models` meta snapshot shared with the connection threads
    // (`server_context_meta`, server-context.cpp:4185-4232)
    let model_meta = ModelMeta {
        model_name: args.model.clone(),
        vocab_type: match vocab.ty {
            llama::vocab::VocabType::None => 0,
            llama::vocab::VocabType::Spm => 1,
            llama::vocab::VocabType::Bpe => 2,
            llama::vocab::VocabType::Wpm => 3,
            llama::vocab::VocabType::Ugm => 4,
            llama::vocab::VocabType::Rwkv => 5,
            llama::vocab::VocabType::Plamo2 => 6,
            llama::vocab::VocabType::Test => 7,
            // LLAMA_VOCAB_TYPE_PLAMO3 = 8 (abeada335, B domain)
            llama::vocab::VocabType::Plamo3 => 8,
        },
        n_vocab: vocab.id_to_token.len() as u64,
        n_ctx: slot_n_ctx as u64,
        n_ctx_train: model_hparams_n_ctx_train as u64,
        n_embd: model_hparams_n_embd as u64,
        n_params: model_n_params,
        size: model_size,
        ftype: engine.ftype.clone(),
        has_mtmd: false,
        // `server_model_output_modalities(common_get_decision_type(model))`
        // (server-context.cpp:4548, 4d60b4d08)
        model_output_modalities: server_decision::server_model_output_modalities(
            decision_type,
        ),
    };
    eprintln!(
        "llama-server: model loaded in {:?} (n_ctx = {}, n_parallel = {} -> slot n_ctx = {}, \
         n_batch = {}, n_ubatch = {}, threads = {}, fa = {}, embeddings = {}, pooling = {})",
        t0.elapsed(),
        args.n_ctx,
        n_parallel,
        slot_n_ctx,
        n_batch,
        args.n_ubatch,
        args.n_threads,
        args.flash_attn,
        args.embeddings,
        pooling_name(resolved_pooling),
    );
    Ok((engine, vocab, model_meta, chat_templates_init, gpu_mode_ran))
}

/// `llama_pooling_type(ctx_tgt)` — the resolved pooling of the server's context
/// (`resolve_pooling`, llama-context.cpp:216-222)
fn pooling_name(p: llama::hparams::LlamaPoolingType) -> &'static str {
    match p {
        llama::hparams::LlamaPoolingType::NONE => "none",
        llama::hparams::LlamaPoolingType::MEAN => "mean",
        llama::hparams::LlamaPoolingType::CLS => "cls",
        llama::hparams::LlamaPoolingType::LAST => "last",
        _ => "unspecified",
    }
}

/// the `server_context_meta` subset `/models` reports (server-context.h)
#[derive(Clone)]
pub struct ModelMeta {
    pub model_name: String,
    pub vocab_type: u64,
    pub n_vocab: u64,
    pub n_ctx: u64,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_params: u64,
    pub size: u64,
    pub ftype: String,
    pub has_mtmd: bool,
    /// `model_output_modalities` (server-context.h:23, 4d60b4d08) — output
    /// modalities for GET /models
    pub model_output_modalities: Vec<&'static str>,
}

/// `random_string()` (server-common.cpp:108-124) — 32 alphanumeric characters.
fn random_string(n: usize) -> String {
    api::random_string(n)
}

/// `server_routes::init_routes` (server-context.cpp:4646-5140) — the endpoints
/// the port implements.
fn register_routes(routes_table: &mut HttpServer, server: &Arc<engine::Server>) {
    // GET /health, GET /v1/health (server-context.cpp:4654-4666)
    for path in ["/health", "/v1/health"] {
        let server = server.clone();
        routes_table.add(
            "GET",
            path,
            Arc::new(move |_req: &Request| {
                let processing = server.n_processing.load(Ordering::Relaxed);
                if processing > 0 {
                    Response::ok(format!(
                        "{{\"status\":\"ok\",\"slots_idle\":{},\"slots_processing\":{processing}}}",
                        server.n_slots - processing
                    ))
                } else {
                    Response::ok("{\"status\":\"ok\"}")
                }
            }),
        );
    }
    // GET /props (server-context.cpp:4804-4816)
    {
        let server = server.clone();
        routes_table.add(
            "GET",
            "/props",
            Arc::new(move |_req: &Request| Response::ok(server.props.clone())),
        );
    }

    // POST /completion and POST /completions (server-context.cpp:4906-4916)
    for path in ["/completion", "/completions"] {
        let server = server.clone();
        routes_table.add("POST", path, Arc::new(move |req: &Request| handle_completion(&server, req, api::ResponseType::None)));
    }
    // POST /v1/completions — the OAI alias with the text_completion shape
    // (server-context.cpp:4918-4928 `post_completions_oai`)
    {
        let server = server.clone();
        routes_table.add(
            "POST",
            "/v1/completions",
            Arc::new(move |req: &Request| handle_completion(&server, req, api::ResponseType::OaiCmpl)),
        );
    }
    // POST /chat/completions and POST /v1/chat/completions
    // (server.cpp:261-262, server-context.cpp:4930-4944)
    for path in ["/chat/completions", "/v1/chat/completions"] {
        let server = server.clone();
        routes_table.add("POST", path, Arc::new(move |req: &Request| handle_chat_completions(&server, req)));
    }

    // POST /embedding, /embeddings (legacy) and /v1/embeddings (OAI)
    // (server.cpp:270-272, server-context.cpp:5139-5146)
    for path in ["/embedding", "/embeddings"] {
        let server = server.clone();
        routes_table.add("POST", path, Arc::new(move |req: &Request| handle_embeddings(&server, req, api::ResponseType::None)));
    }
    {
        let server = server.clone();
        routes_table.add(
            "POST",
            "/v1/embeddings",
            Arc::new(move |req: &Request| handle_embeddings(&server, req, api::ResponseType::OaiEmbd)),
        );
    }

    // the /rerank family (server.cpp:273-276, server-context.cpp:5147-5220) —
    // no local model carries rank pooling, so the port keeps the reference's
    // "not supported" answer (the port has no RANK pooling at all)
    for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        let server = server.clone();
        routes_table.add("POST", path, Arc::new(move |_req: &Request| handle_rerank(&server)));
    }

    // POST /v1/systemone (server.cpp:282 + server-context.cpp:5443-5446,
    // upstream a7b94df2c) — the TypeSafe decision API
    {
        let server = server.clone();
        routes_table.add(
            "POST",
            "/v1/systemone",
            Arc::new(move |req: &Request| handle_systemone(&server, req)),
        );
    }

    // GET /models, GET /v1/models (server.cpp:256-257, server-context.cpp:5072-5082)
    for path in ["/models", "/v1/models"] {
        let server = server.clone();
        routes_table.add("GET", path, Arc::new(move |_req: &Request| handle_models(&server)));
    }

    // GET /slots (server.cpp:290, server-context.cpp:4729-4800) — the status
    // snapshot the engine publishes after each pass — and POST /slots/:id_slot
    // (save / restore / erase, server.cpp:291 + server-context.cpp:4771-4800)
    {
        let server = server.clone();
        routes_table.add(
            "GET",
            "/slots",
            Arc::new(move |_req: &Request| {
                Response::ok(server.slots_json.read().unwrap().clone())
            }),
        );
    }
    {
        let server = server.clone();
        routes_table.add(
            "POST",
            "/slots/:id_slot",
            Arc::new(move |req: &Request| handle_post_slots(&server, req)),
        );
    }

    // POST /tokenize (server-context.cpp:5084-5123)
    {
        let server = server.clone();
        routes_table.add("POST", "/tokenize", Arc::new(move |req: &Request| handle_tokenize(&server, req)));
    }
    // POST /detokenize (server-context.cpp:5125-5137)
    {
        let server = server.clone();
        routes_table.add(
            "POST",
            "/detokenize",
            Arc::new(move |req: &Request| handle_detokenize(&server, req)),
        );
    }
}

/// `gen_chatcmplid()` (server-common.cpp:126-128) — every OAI completion
/// (chat and text) shares the same id space
fn gen_chatcmplid() -> String {
    format!("chatcmpl-{}", random_string(32))
}

/// `POST /completion` / `POST /v1/completions` (server-context.cpp:4257-4560
/// `handle_completions_impl`); `res_type` selects the response family.
/// `common_chat_msg_delimiters_parse` (common/chat.cpp:126-139) +
/// `delimiters.tokenize` + the USER-role half of
/// `common_chat_msg_delimiters::split` (common/chat.cpp:143-165, reached
/// through `server_tokens::find_message_spans`, server-common.cpp:791-797):
/// scan the token stream for the delimiter token sequences (the first
/// delimiter in list order wins at a position) and report where user
/// messages start. No media chunks in the port, so the `skips` map is empty.
fn message_user_starts(vocab: &Vocab, tokens: &[i32], data: &Json) -> Vec<usize> {
    let Some(Json::Array(delims)) = data.at("message_delimiters") else {
        return Vec::new();
    };
    let mut parsed: Vec<(bool, Vec<i32>)> = Vec::new(); // (is_user, tokens)
    for d in delims {
        let role = d.at("role").and_then(|v| v.get_str().ok()).unwrap_or("");
        let delimiter = d.at("delimiter").and_then(|v| v.get_str().ok()).unwrap_or("");
        // `common_tokenize(vocab, d.delimiter, false, true)` (chat.cpp:107-112)
        parsed.push((role == "user", vocab.tokenize(delimiter, false, true)));
    }
    let mut out = Vec::new();
    'pos: for i in 0..tokens.len() {
        for (is_user, dt) in &parsed {
            if i + dt.len() > tokens.len() || dt.is_empty() {
                continue;
            }
            if tokens[i..i + dt.len()] == dt[..] {
                if *is_user {
                    out.push(i);
                }
                continue 'pos;
            }
        }
    }
    out
}

fn handle_completion(server: &Arc<engine::Server>, req: &Request, res_type: api::ResponseType) -> Response {
    let data = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            // the reference reports the nlohmann exception as a server error
            // (server.cpp:60-79)
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ));
        }
    };
    if !data.is_object() {
        return Response::error(api::json_error(
            "request body must be a JSON object",
            ERROR_TYPE_INVALID_REQUEST.0,
            ERROR_TYPE_INVALID_REQUEST.1,
        ));
    }

    // "prompt" may be a string, an array of strings (one task per string), a
    // token-id array, or a mixed string/token array
    // (`tokenize_input_prompts`, server-common.cpp:1015-1043: an array that
    // contains ANY number is ONE subprompt through `tokenize_mixed`; a pure
    // string array is one prompt per string)
    let array_has_number = match data.at("prompt") {
        Some(Json::Array(a)) => a.iter().any(|v| matches!(v, Json::Int(_) | Json::Uint(_))),
        _ => false,
    };
    enum PromptInput {
        Tasks(Vec<(String, Vec<i32>)>),
        Tokens(Vec<i32>, String),
    }
    let prompts: PromptInput = match data.at("prompt") {
        Some(Json::String(s)) => PromptInput::Tasks(vec![(s.clone(), server.vocab.tokenize(s, true, true))]),
        Some(Json::Array(a)) if !array_has_number => {
            let mut v = Vec::new();
            for p in a {
                match p {
                    Json::String(s) => v.push((s.clone(), server.vocab.tokenize(s, true, true))),
                    _ => {
                        return Response::error(api::json_error(
                            "prompt array must contain strings",
                            ERROR_TYPE_INVALID_REQUEST.0,
                            ERROR_TYPE_INVALID_REQUEST.1,
                        ))
                    }
                }
            }
            PromptInput::Tasks(v)
        }
        // one mixed subprompt: strings tokenize with add_special, numbers are
        // token ids verbatim (`tokenize_mixed`, server-common.cpp:941-968)
        Some(Json::Array(a)) => {
            let mut toks: Vec<i32> = Vec::new();
            for p in a {
                match p {
                    Json::String(s) => toks.extend(server.vocab.tokenize(s, true, true)),
                    Json::Int(i) => toks.push(*i as i32),
                    Json::Uint(u) => toks.push(*u as i32),
                    _ => {
                        return Response::error(api::json_error(
                            "\"prompt\" elements must be a string, a list of tokens, or a list of \
                             mixed strings & tokens",
                            ERROR_TYPE_INVALID_REQUEST.0,
                            ERROR_TYPE_INVALID_REQUEST.1,
                        ))
                    }
                }
            }
            PromptInput::Tokens(toks, data.at("prompt").unwrap().dump())
        }
        _ => {
            return Response::error(api::json_error(
                "[json.exception.type_error.302] type must be string, but is null",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ))
        }
    };
    let (prompt_list, prompt_text_of): (Vec<Vec<i32>>, Box<dyn Fn(usize) -> String>) = match prompts {
        PromptInput::Tasks(v) => (
            v.iter().map(|(_, t)| t.clone()).collect(),
            Box::new(move |i: usize| v[i].0.clone()),
        ),
        // a token-id prompt has no text form — report the raw JSON like the C
        PromptInput::Tokens(t, raw) => (vec![t], Box::new(move |_|
            format!("<token ids: {raw}>")
        )),
    };

    let base = {
        let mut p = TaskParams::default();
        // the schema's defaults come from `params_base` (server-schema.cpp:17)
        p.n_predict = server.n_predict_default;
        // `params.n_keep = params_base.n_keep` (server-schema.cpp:529)
        p.n_keep = server.n_keep_default;
        // the request base inherits common_params' reasoning-format default
        p.reasoning_format = "deepseek".into();
        // `generation_settings.speculative.types` inherits the server's
        // selection (server-task.cpp:81)
        p.speculative_types = server.speculative_types.clone();
        p
    };
    let params = match api::eval_llama_cmpl_schema(&data, &base, &server.vocab) {
        Ok(p) => p,
        Err(msg) => {
            return Response::error(api::json_error(
                &msg,
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ))
        }
    };

    // tokenize (`tokenize_input_prompts` with add_special = true)
    let mut receivers: Vec<mpsc::Receiver<StreamEvent>> = Vec::new();
    let mut pending: Vec<Task> = Vec::new();
    let completion_id = if res_type == api::ResponseType::None {
        String::new()
    } else {
        gen_chatcmplid()
    };
    for (index, prompt_tokens) in prompt_list.iter().enumerate() {
        let tokens = prompt_tokens.clone();
        if tokens.is_empty() {
            return Response::error(api::json_error(
                "prompt is empty",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ));
        }
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        receivers.push(rx);
        let mut p = params.clone();
        // "OAI-compat" (server-context.cpp:4327-4329)
        p.res_type = res_type;
        p.oaicompat_cmpl_id = completion_id.clone();
        p.oaicompat_model = server.model_meta.model_name.clone();
        // `task.params.message_spans = task.tokens.find_message_spans(delims)`
        // (server-context.cpp:4829) — the request's `message_delimiters`
        p.message_user_starts = message_user_starts(&server.vocab, &tokens, &data);
        pending.push(Task {
            id: 0,
            index: index as i32,
            kind: TaskKind::Completion,
            tokens,
            params: p,
            prompt_text: prompt_text_of(index),
            slot_action: None,
            decision_body: Json::Null,
            tx,
        });
    }
    for t in pending.iter_mut() {
        t.id = server.new_task_id();
    }
    for t in pending {
        server.queue.submit(t);
    }

    if params.stream {
        // "in streaming mode, the first error must be treated as non-stream
        // response" (server-context.cpp:4368-4377)
        let first = match receivers.first().map(|r| r.recv()) {
            Some(Ok(ev)) => ev,
            _ => return Response::error(api::json_error("no response", api::ERROR_TYPE_SERVER.0, api::ERROR_TYPE_SERVER.1)),
        };
        if let StreamEvent::Frame(f) = &first {
            if let Ok(v) = Json::parse(f) {
                if v.at("error").is_some() {
                    return Response::error(v.dump());
                }
            }
        }
        // relay every task's frames into one SSE stream
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        std::thread::spawn(move || {
            if tx.send(first).is_err() {
                return;
            }
            for r in receivers.into_iter() {
                for ev in r.iter() {
                    let done = matches!(ev, StreamEvent::Done);
                    if tx.send(ev).is_err() {
                        return;
                    }
                    if done {
                        break;
                    }
                }
            }
            let _ = tx.send(StreamEvent::Done);
        });
        // OAI streams terminate with `data: [DONE]` (server-context.cpp:4425-4433)
        let mut resp = Response::stream(rx);
        resp.terminal_done = res_type != api::ResponseType::None;
        return resp;
    }

    // non-streaming: wait for every task's final frame
    let mut results: Vec<Json> = Vec::new();
    for rx in receivers {
        let mut final_json: Option<Json> = None;
        for ev in rx.iter() {
            match ev {
                StreamEvent::Frame(f) => {
                    let Ok(v) = Json::parse(&f) else { continue };
                    if v.at("error").is_some() {
                        return Response::error(v.dump());
                    }
                    final_json = Some(v);
                }
                StreamEvent::Done => break,
                StreamEvent::Ping => {}
            }
        }
        match final_json {
            Some(v) => results.push(v),
            None => {
                return Response::error(api::json_error(
                    "no response from the engine",
                    api::ERROR_TYPE_SERVER.0,
                    api::ERROR_TYPE_SERVER.1,
                ))
            }
        }
    }
    if results.len() == 1 {
        Response::ok(results.pop().unwrap().dump())
    } else if res_type == api::ResponseType::OaiChat || res_type == api::ResponseType::OaiCmpl {
        // "if multiple results in OAI format, we need to re-format them"
        // (server-context.cpp:4350-4357): the choices of every result join
        // the first object's array
        let mut arr = results;
        let mut first = arr.remove(0);
        let mut choices = match first.at("choices") {
            Some(Json::Array(c)) => c.clone(),
            _ => Vec::new(),
        };
        for other in arr {
            if let Some(choice) = other.at("choices").and_then(|c| c.at_idx(0)).cloned() {
                choices.push(choice);
            }
        }
        first.set("choices", Json::Array(choices));
        Response::ok(first.dump())
    } else {
        Response::ok(Json::Array(results).dump())
    }
}

/// `POST /v1/chat/completions` (server-context.cpp:4930-4944): parse the chat
/// body through `oaicompat_chat_params_parse`, then run the completion flow
/// with the chat response family.
fn handle_chat_completions(server: &Arc<engine::Server>, req: &Request) -> Response {
    let body = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ));
        }
    };
    // `common_chat_templates_init` (chat.cpp:757-855) — the handler thread
    // owns its templates (the parsed program holds `Rc` values, so it cannot
    // be shared; the init inputs were validated at load, this cannot fail)
    let templates = match llama::chat_tools::ChatTemplates::init(&server.chat_templates_init) {
        Ok(t) => t,
        Err(e) => {
            return Response::error(api::json_error(
                &format!("chat template parsing error: {e}"),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ))
        }
    };
    let opts = chat::ChatOptions {
        templates: &templates,
        enable_thinking: server.enable_thinking,
        reasoning_format: llama::chat_tools::ReasoningFormat::Deepseek,
    };
    let parsed = match chat::oaicompat_chat_params_parse(&body, &opts) {
        Ok(p) => p,
        Err(e) => return Response::error(e.to_response_json()),
    };
    // the chat metadata rides on the task params (server-schema.cpp:295-320:
    // chat_format / grammar / grammar_lazy / grammar_triggers /
    // preserved_tokens / generation_prompt / chat_parser)
    let mut params = {
        let mut base = TaskParams::default();
        base.n_predict = server.n_predict_default;
        // `params.n_keep = params_base.n_keep` (server-schema.cpp:529)
        base.n_keep = server.n_keep_default;
        base.speculative_types = server.speculative_types.clone();
        // the request base inherits common_params' reasoning-format default
        base.reasoning_format = "deepseek".into();
        match api::eval_llama_cmpl_schema(&parsed.llama_params, &base, &server.vocab) {
            Ok(p) => p,
            Err(msg) => {
                return Response::error(api::json_error(
                    &msg,
                    ERROR_TYPE_INVALID_REQUEST.0,
                    ERROR_TYPE_INVALID_REQUEST.1,
                ))
            }
        }
    };
    params.chat_format = parsed.chat_format.to_string();
    params.generation_prompt = parsed.generation_prompt;
    params.grammar_prefill = parsed.grammar_needs_prefill;
    // the serialized autoparser arena (`chat_parser_params.parser`)
    params.chat_parser = parsed.parser;

    // one task on the rendered prompt — tokenized with special-token parsing
    // (the chat prompt carries the template's `<|im_…|>` markers)
    let prompt = parsed
        .llama_params
        .at("prompt")
        .and_then(|v| v.get_str().ok())
        .unwrap_or("")
        .to_string();
    let tokens = chat::tokenize_prompt(&server.vocab, &prompt, true);
    if tokens.is_empty() {
        return Response::error(api::json_error(
            "prompt is empty",
            ERROR_TYPE_INVALID_REQUEST.0,
            ERROR_TYPE_INVALID_REQUEST.1,
        ));
    }
    let (tx, rx) = mpsc::channel::<StreamEvent>();
    params.res_type = api::ResponseType::OaiChat;
    params.oaicompat_cmpl_id = gen_chatcmplid();
    params.oaicompat_model = server.model_meta.model_name.clone();
    // `task.params.message_spans` (server-context.cpp:4829) — the chat
    // handler's `llama_params` carries the autoparser-derived
    // `message_delimiters` (chat.rs, server-common.cpp:1454)
    params.message_user_starts =
        message_user_starts(&server.vocab, &tokens, &parsed.llama_params);
    let stream = params.stream;
    let task = Task {
        id: server.new_task_id(),
        index: 0,
        kind: TaskKind::Completion,
        tokens,
        params,
        prompt_text: prompt,
        slot_action: None,
            decision_body: Json::Null,
        tx,
    };
    server.queue.submit(task);

    if stream {
        // the first error answers as a non-stream response
        // (server-context.cpp:4368-4377)
        let first = match rx.recv() {
            Ok(ev) => ev,
            _ => {
                return Response::error(api::json_error(
                    "no response",
                    api::ERROR_TYPE_SERVER.0,
                    api::ERROR_TYPE_SERVER.1,
                ))
            }
        };
        if let StreamEvent::Frame(f) = &first {
            if let Ok(v) = Json::parse(f) {
                if v.at("error").is_some() {
                    return Response::error(v.dump());
                }
            }
        }
        let (tx2, rx2) = mpsc::channel::<StreamEvent>();
        std::thread::spawn(move || {
            if tx2.send(first).is_err() {
                return;
            }
            for ev in rx.iter() {
                let done = matches!(ev, StreamEvent::Done);
                if tx2.send(ev).is_err() {
                    return;
                }
                if done {
                    break;
                }
            }
            let _ = tx2.send(StreamEvent::Done);
        });
        let mut resp = Response::stream(rx2);
        resp.terminal_done = true;
        return resp;
    }

    // non-stream: the final frame is the whole chat completion
    let mut final_json: Option<Json> = None;
    for ev in rx.iter() {
        match ev {
            StreamEvent::Frame(f) => {
                let Ok(v) = Json::parse(&f) else { continue };
                if v.at("error").is_some() {
                    return Response::error(v.dump());
                }
                final_json = Some(v);
            }
            StreamEvent::Done => break,
            StreamEvent::Ping => {}
        }
    }
    match final_json {
        Some(v) => Response::ok(v.dump()),
        None => Response::error(api::json_error(
            "no response from the engine",
            api::ERROR_TYPE_SERVER.0,
            api::ERROR_TYPE_SERVER.1,
        )),
    }
}

/// nlohmann's `json.type_name()` (`json.exception.type_error.302` messages)
fn json_type_name(v: &Json) -> &'static str {
    match v {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Int(_) | Json::Uint(_) => "number",
        Json::Double(_) => "number",
        Json::String(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

/// `POST /v1/systemone` (server-context.cpp:5446-5534): parse + task assembly
/// happen engine-side (the decision context owns the model's fitted
/// temperatures); this handler submits the body and relays the single result
/// frame. The "not a decision model" 501 comes back through the same frame
/// path (`format_error_response(..., ERROR_TYPE_NOT_SUPPORTED)`).
fn handle_systemone(server: &Arc<engine::Server>, req: &Request) -> Response {
    let body = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ))
        }
    };

    let (tx, rx) = mpsc::channel::<StreamEvent>();
    let task = Task {
        id: server.new_task_id(),
        index: 0,
        kind: TaskKind::Decision,
        tokens: Vec::new(),
        params: TaskParams::default(),
        prompt_text: String::new(),
        slot_action: None,
        decision_body: body,
        tx,
    };
    server.queue.submit(task);

    for ev in rx.iter() {
        match ev {
            StreamEvent::Frame(f) => {
                let Ok(v) = Json::parse(&f) else { continue };
                if v.at("error").is_some() {
                    return Response::error(v.dump());
                }
                return Response::ok(v.dump());
            }
            StreamEvent::Done => break,
            StreamEvent::Ping => {}
        }
    }
    Response::error(api::json_error(
        "no response",
        api::ERROR_TYPE_SERVER.0,
        api::ERROR_TYPE_SERVER.1,
    ))
}

/// `POST /embedding(s)` / `POST /v1/embeddings`
/// (server-context.cpp:5390-5491 `handle_embeddings_impl`).
fn handle_embeddings(server: &Arc<engine::Server>, req: &Request, mut res_type: api::ResponseType) -> Response {
    // "This server does not support embeddings. Start it with `--embeddings`"
    if !server.embeddings {
        return Response::error(api::json_error(
            "This server does not support embeddings. Start it with `--embeddings`",
            api::ERROR_TYPE_NOT_SUPPORTED.0,
            api::ERROR_TYPE_NOT_SUPPORTED.1,
        ));
    }
    // "Pooling type 'none' is not OAI compatible"
    if res_type == api::ResponseType::OaiEmbd && server.pooling == llama::hparams::LlamaPoolingType::NONE
    {
        return Response::error(api::json_error(
            "Pooling type 'none' is not OAI compatible. Please use a different pooling type",
            ERROR_TYPE_INVALID_REQUEST.0,
            ERROR_TYPE_INVALID_REQUEST.1,
        ));
    }
    let body = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ))
        }
    };

    // "input" (OAI) or "content" (legacy — not OAI compatible)
    let prompt = if let Some(v) = body.at("input") {
        v.clone()
    } else if let Some(v) = body.at("content") {
        res_type = api::ResponseType::None;
        v.clone()
    } else {
        return Response::error(api::json_error(
            "\"input\" or \"content\" must be provided",
            ERROR_TYPE_INVALID_REQUEST.0,
            ERROR_TYPE_INVALID_REQUEST.1,
        ));
    };

    let mut use_base64 = false;
    if let Some(fmt) = body.at("encoding_format") {
        // `json_value` throws `[json.exception.type_error.302]` on a non-string
        // (mapped to 400 by the common_json_error catch added in a7b94df2c,
        // server.cpp:64-68 — test_embedding_invalid_request)
        let Json::String(format) = fmt else {
            return Response::error(api::json_error(
                &format!(
                    "[json.exception.type_error.302] type must be string, but is {}",
                    json_type_name(fmt)
                ),
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ));
        };
        if format == "base64" {
            use_base64 = true;
        } else if format != "float" {
            return Response::error(api::json_error(
                "The format to return the embeddings in. Can be either float or base64",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ));
        }
    }

    // `tokenize_input_prompts` + `tokenize_entry` (server-context.cpp:5408-5418,
    // add_special = true): a string, an array of per-task entries, or a single
    // entry. An entry is a string, an object with `prompt_string`, or an
    // object with an OAI typed-content array
    // (`tokenize_oai_content_array`, server-common.cpp:1197-1222). The port
    // has no mtmd context in the server, so media parts answer with the
    // reference's "not supported" runtime error.
    fn tokenize_embedding_entry(server: &engine::Server, entry: &Json) -> Result<Vec<i32>, String> {
        if let Json::Object(_) = entry {
            if let Some(content) = entry.at("content") {
                // tokenize_oai_content_array: "content" must be an array
                let Json::Array(parts) = content else {
                    return Err("\"content\" must be an array".to_string());
                };
                let mut ptext = String::new();
                for p in parts {
                    let ty = p
                        .at("type")
                        .and_then(|v| v.get_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    match ty.as_str() {
                        "text" => {
                            ptext.push_str(
                                &p.at("text")
                                    .and_then(|v| v.get_str().ok())
                                    .unwrap_or_default(),
                            );
                        }
                        "image_url" => {
                            return Err("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj".to_string())
                        }
                        "input_audio" => {
                            return Err("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj".to_string())
                        }
                        "input_video" | "video_url" => {
                            return Err("video input is not supported - hint: if this is unexpected, you may need to provide the mmproj".to_string())
                        }
                        _ => return Err("unsupported content[].type".to_string()),
                    }
                }
                return Ok(server.vocab.tokenize(&ptext, true, true));
            }
            if let Some(ps) = entry.at("prompt_string") {
                // tokenize_input_subprompt's object arm
                // (server-common.cpp:996-1011)
                if entry.at("multimodal_data").is_some() {
                    return Err("Multimodal data provided, but model does not support multimodal requests.".to_string());
                }
                let st = ps.get_str().map_err(|_| {
                    "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens."
                        .to_string()
                })?;
                return Ok(server.vocab.tokenize(&st, true, true));
            }
        }
        match entry {
            Json::String(st) => Ok(server.vocab.tokenize(st, true, true)),
            _ => Err("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.".to_string()),
        }
    }

    let inputs: Vec<(String, Vec<i32>)> = match &prompt {
        Json::String(s) => {
            match tokenize_embedding_entry(server, &prompt) {
                Ok(toks) => vec![(s.clone(), toks)],
                Err(msg) => {
                    return Response::error(api::json_error(
                        &msg,
                        ERROR_TYPE_SERVER.0,
                        ERROR_TYPE_SERVER.1,
                    ))
                }
            }
        }
        Json::Array(a) => {
            // an array that contains ANY number is ONE mixed subprompt
            // (`tokenize_mixed`); otherwise one task per entry
            let array_has_number =
                a.iter().any(|v| matches!(v, Json::Int(_) | Json::Uint(_)));
            let mut v = Vec::new();
            if array_has_number {
                let mut toks: Vec<i32> = Vec::new();
                for p in a.iter() {
                    match p {
                        Json::String(st) => toks.extend(server.vocab.tokenize(st, true, true)),
                        Json::Int(i) => toks.push(*i as i32),
                        Json::Uint(u) => toks.push(*u as i32),
                        _ => {
                            return Response::error(api::json_error(
                                "\"prompt\" elements must be a string, a list of tokens, or a list of mixed strings & tokens",
                                ERROR_TYPE_SERVER.0,
                                ERROR_TYPE_SERVER.1,
                            ))
                        }
                    }
                }
                v.push((prompt.dump(), toks));
            } else {
                for p in a.iter() {
                    match tokenize_embedding_entry(server, p) {
                        Ok(toks) => {
                            let text = match p {
                                Json::String(st) => st.clone(),
                                other => other.dump(),
                            };
                            v.push((text, toks));
                        }
                        Err(msg) => {
                            return Response::error(api::json_error(
                                &msg,
                                ERROR_TYPE_SERVER.0,
                                ERROR_TYPE_SERVER.1,
                            ))
                        }
                    }
                }
            }
            v
        }
        _ => {
            return Response::error(api::json_error(
                "[json.exception.type_error.302] type must be string, but is null",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ))
        }
    };

    // `tokenize_input_prompts` throws on an empty list
    // ("prompt must not be empty" — `std::invalid_argument` since
    // a7b94df2c, server-common.cpp:1026, so a 400)
    if inputs.is_empty() {
        return Response::error(api::json_error(
            "\"prompt\" must not be empty",
            api::ERROR_TYPE_INVALID_REQUEST.0,
            api::ERROR_TYPE_INVALID_REQUEST.1,
        ));
    }

    // `embd_normalize` — the request may override the server default (2)
    let embd_normalize = body
        .at("embd_normalize")
        .and_then(|v| v.get_i64().ok())
        .map(|v| v as i32)
        .unwrap_or(2);

    // one embedding task per input; the engine answers sequentially
    let mut receivers: Vec<mpsc::Receiver<StreamEvent>> = Vec::new();
    let mut tasks: Vec<Task> = Vec::new();
    for (index, (input, tokens)) in inputs.iter().enumerate() {
        let tokens = tokens.clone();
        if tokens.is_empty() {
            return Response::error(api::json_error(
                "Input content cannot be empty",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ));
        }
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        receivers.push(rx);
        let mut p = TaskParams::default();
        p.res_type = res_type;
        p.embd_normalize = embd_normalize;
        tasks.push(Task {
            id: 0,
            index: index as i32,
            kind: TaskKind::Embedding,
            tokens,
            params: p,
            prompt_text: input.clone(),
            slot_action: None,
            decision_body: Json::Null,
            tx,
        });
    }
    for t in tasks.iter_mut() {
        t.id = server.new_task_id();
    }
    for t in tasks {
        server.queue.submit(t);
    }

    let mut responses: Vec<Json> = Vec::new();
    for rx in receivers {
        for ev in rx.iter() {
            match ev {
                StreamEvent::Frame(f) => {
                    let Ok(v) = Json::parse(&f) else { continue };
                    if v.at("error").is_some() {
                        return Response::error(v.dump());
                    }
                    responses.push(v);
                }
                StreamEvent::Done => break,
                StreamEvent::Ping => {}
            }
        }
    }
    if res_type == api::ResponseType::OaiEmbd {
        if use_base64 {
            Response::ok(embeddings_oai_base64(&body, &server.model_meta.model_name, &responses).dump())
        } else {
            Response::ok(api::format_embeddings_response_oaicompat(&body, &server.model_meta.model_name, &responses).dump())
        }
    } else {
        Response::ok(Json::Array(responses).dump())
    }
}

/// the base64 variant of `format_embeddings_response_oaicompat`
/// (server-common.cpp:1439-1452): the f32 array's little-endian bytes in
/// standard base64
fn embeddings_oai_base64(request: &Json, model_name: &str, embeddings: &[Json]) -> Json {
    let mut n_tokens = 0i64;
    let mut data = Vec::new();
    for (i, elem) in embeddings.iter().enumerate() {
        let vals: Vec<f32> = elem
            .at("embedding")
            .map(|e| e.iter().filter_map(|v| v.get_f64().ok().map(|x| x as f32)).collect())
            .unwrap_or_default();
        n_tokens += elem.at("tokens_evaluated").and_then(|v| v.get_i64().ok()).unwrap_or(0);
        let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        data.push(Json::Object(vec![
            ("embedding".into(), Json::String(base64_encode(&bytes))),
            ("index".into(), Json::Int(i as i64)),
            ("object".into(), Json::String("embedding".into())),
            ("encoding_format".into(), Json::String("base64".into())),
        ]));
    }
    Json::Object(vec![
        (
            "model".into(),
            request
                .at("model")
                .and_then(|v| v.get_str().ok())
                .map(|s| Json::String(s.to_string()))
                .unwrap_or(Json::String(model_name.to_string())),
        ),
        ("object".into(), Json::String("list".into())),
        (
            "usage".into(),
            Json::Object(vec![
                ("prompt_tokens".into(), Json::Int(n_tokens)),
                ("total_tokens".into(), Json::Int(n_tokens)),
            ]),
        ),
        ("data".into(), Json::Array(data)),
    ])
}

/// standard base64 (RFC 4648, with padding) — `base64::encode` of the
/// reference's httplib payload path
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// the `/rerank` family (server-context.cpp:5147-5161): without RANK pooling
/// the reference answers 501 "This server does not support reranking. Start
/// it with `--reranking`" — the port has no RANK pooling at all, so this is
/// its answer on every model.
fn handle_rerank(server: &Arc<engine::Server>) -> Response {
    let _ = server;
    Response::error(api::json_error(
        "This server does not support reranking. Start it with `--reranking`",
        api::ERROR_TYPE_NOT_SUPPORTED.0,
        api::ERROR_TYPE_NOT_SUPPORTED.1,
    ))
}

/// `GET /models`, `GET /v1/models` — `get_res_models`
/// (server-context.cpp:4566-4595)
fn handle_models(server: &Arc<engine::Server>) -> Response {
    let m = &server.model_meta;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let model_name = Json::String(m.model_name.clone());
    let capabilities = if m.has_mtmd {
        vec![Json::String("completion".into()), Json::String("multimodal".into())]
    } else {
        vec![Json::String("completion".into())]
    };
    let models = Json::Array(vec![Json::Object(vec![
        ("name".into(), model_name.clone()),
        ("model".into(), model_name.clone()),
        ("modified_at".into(), Json::String(String::new())),
        ("size".into(), Json::String(String::new())),
        // "dummy value, llama.cpp does not support managing model file's hash"
        ("digest".into(), Json::String(String::new())),
        ("type".into(), Json::String("model".into())),
        ("description".into(), Json::String(String::new())),
        ("tags".into(), Json::Array(vec![Json::String(String::new())])),
        ("capabilities".into(), Json::Array(capabilities)),
        ("parameters".into(), Json::String(String::new())),
        (
            "details".into(),
            Json::Object(vec![
                ("parent_model".into(), Json::String(String::new())),
                ("format".into(), Json::String("gguf".into())),
                ("family".into(), Json::String(String::new())),
                ("families".into(), Json::Array(vec![Json::String(String::new())])),
                ("parameter_size".into(), Json::String(String::new())),
                ("quantization_level".into(), Json::String(String::new())),
            ]),
        ),
    ])]);
    // `get_res_model_info` (server-context.cpp:4543-4563)
    let info = Json::Object(vec![
        ("id".into(), model_name.clone()),
        ("aliases".into(), Json::Array(vec![model_name])),
        ("tags".into(), Json::Array(Vec::new())),
        ("object".into(), Json::String("model".into())),
        // `server_model_architecture_json(has_inp_image, has_inp_audio,
        // has_inp_video, model_output_modalities)` (server-context.cpp:4902-4906,
        // 4d60b4d08) — the port has no mmproj wiring, so the input flags
        // (chat_params.allow_image/audio/video) are always false
        (
            "architecture".into(),
            server_decision::server_model_architecture_json(
                false,
                false,
                false,
                &m.model_output_modalities,
            ),
        ),
        ("created".into(), Json::Int(t)),
        ("owned_by".into(), Json::String("llamacpp".into())),
        (
            "meta".into(),
            Json::Object(vec![
                ("vocab_type".into(), Json::Int(m.vocab_type as i64)),
                ("n_vocab".into(), Json::Int(m.n_vocab as i64)),
                ("n_ctx".into(), Json::Int(m.n_ctx as i64)),
                ("n_ctx_train".into(), Json::Int(m.n_ctx_train as i64)),
                ("n_embd".into(), Json::Int(m.n_embd as i64)),
                ("n_params".into(), Json::Int(m.n_params as i64)),
                ("size".into(), Json::Int(m.size as i64)),
                ("ftype".into(), Json::String(m.ftype.clone())),
            ]),
        ),
    ]);
    Response::ok(
        Json::Object(vec![
            ("models".into(), models),
            ("object".into(), Json::String("list".into())),
            ("data".into(), Json::Array(vec![info])),
        ])
        .dump(),
    )
}

/// `json::parse` failures: nlohmann's message for the unexpected-end-of-input
/// case, the port's own text otherwise (the accepted/rejected inputs match).
fn json_parse_error_message(e: &str, body: &str) -> String {
    if e.contains("unexpected end") || e.contains("end of input") {
        // nlohmann reports the failure point as line/column; the port's parser
        // gives the byte offset (json_schema.rs's `parse_error.101` text)
        let byte = e
            .split("at byte ")
            .nth(1)
            .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(body.len());
        let head = &body.as_bytes()[..byte.min(body.len())];
        let line = 1 + head.iter().filter(|&&b| b == b'\n').count();
        let column = match head.iter().rposition(|&b| b == b'\n') {
            Some(p) => head.len() - p,
            None => head.len() + 1,
        };
        format!(
            "[json.exception.parse_error.101] parse error at line {line}, column {column}: syntax \
             error while parsing value - unexpected end of input; expected '[', '{{', or a literal"
        )
    } else {
        e.to_string()
    }
}

/// `POST /tokenize` (server-context.cpp:5084-5123)
fn handle_tokenize(server: &Arc<Server>, req: &Request) -> Response {
    let body = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ))
        }
    };
    let vox = &server.vocab;
    let mut tokens_response: Vec<Json> = Vec::new();
    if let Some(content) = body.at("content") {
        let add_special = body.at("add_special").map(|v| matches!(v, Json::Bool(true))).unwrap_or(false);
        let parse_special = body.at("parse_special").map(|v| matches!(v, Json::Bool(true))).unwrap_or(true);
        let with_pieces = body.at("with_pieces").map(|v| matches!(v, Json::Bool(true))).unwrap_or(false);
        // `tokenize_mixed`: a string, or an array of strings / token ids
        let tokens: Vec<i32> = match content {
            Json::String(s) => vox.tokenize(s, add_special, parse_special),
            Json::Array(a) => {
                let mut out = Vec::new();
                for e in a {
                    match e {
                        Json::String(s) => out.extend(vox.tokenize(s, add_special, parse_special)),
                        _ => {
                            if let Ok(t) = e.get_i64() {
                                out.push(t as i32);
                            }
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        };
        if with_pieces {
            for &token in tokens.iter() {
                let piece = vox.token_to_piece(token);
                // "Check if the piece is valid UTF-8; if not, store as an array
                // of byte values" (server-context.cpp:5098-5110)
                let piece_json = if std::str::from_utf8(piece.as_bytes()).is_ok() {
                    Json::String(piece.to_string())
                } else {
                    Json::Array(piece.as_bytes().iter().map(|&b| Json::Int(b as i64)).collect())
                };
                tokens_response.push(Json::Object(vec![
                    ("id".into(), Json::Int(token as i64)),
                    ("piece".into(), piece_json),
                ]));
            }
        } else {
            tokens_response = tokens.iter().map(|&t| Json::Int(t as i64)).collect();
        }
    }
    Response::ok(Json::Object(vec![("tokens".into(), Json::Array(tokens_response))]).dump())
}

/// `POST /detokenize` (server-context.cpp:5125-5137)
fn handle_detokenize(server: &Arc<Server>, req: &Request) -> Response {
    let body = match Json::parse(&req.body_str()) {
        Ok(v) => v,
        Err(e) => {
            return Response::error(api::json_error(
                &json_parse_error_message(&e, &req.body_str()),
                api::ERROR_TYPE_SERVER.0,
                api::ERROR_TYPE_SERVER.1,
            ))
        }
    };
    let vox = &server.vocab;
    let mut content = String::new();
    if let Some(Json::Array(a)) = body.at("tokens") {
        let ids: Vec<i32> = a.iter().filter_map(|v| v.get_i64().ok().map(|t| t as i32)).collect();
        content = vox.detokenize(&ids, true);
    }
    Response::ok(Json::Object(vec![("content".into(), Json::String(content))]).dump())
}

/// `server_routes::post_slots` + `handle_slots_save/restore/erase`
/// (server-context.cpp:4771-4800, :5285-5390): `POST /slots/{id_slot}
/// ?action=save|restore|erase`, with the filename in the JSON body of the
/// save/restore actions. The task is queued like any other; the engine thread
/// (the DecodeContext's owner) runs it and answers through the result channel.
fn handle_post_slots(server: &Arc<Server>, req: &Request) -> Response {
    // "This server does not support slots action. Start it with
    // `--slot-save-path`" (server-context.cpp:4773-4777)
    if server.slot_save_path.is_empty() {
        return Response::error(api::json_error(
            "This server does not support slots action. Start it with `--slot-save-path`",
            api::ERROR_TYPE_NOT_SUPPORTED.0,
            api::ERROR_TYPE_NOT_SUPPORTED.1,
        ));
    }
    // `std::stoi(req.get_param("id_slot"))` — the path capture, "Invalid slot
    // ID" on a non-integer (server-context.cpp:4783-4790)
    let id_slot: i32 = match req.get_param("id_slot", "").parse() {
        Ok(v) => v,
        Err(_) => {
            return Response::error(api::json_error(
                "Invalid slot ID",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ))
        }
    };
    let action = req.get_param("action", "");
    if !matches!(action.as_str(), "save" | "restore" | "erase") {
        return Response::error(api::json_error(
            "Invalid action",
            ERROR_TYPE_INVALID_REQUEST.0,
            ERROR_TYPE_INVALID_REQUEST.1,
        ));
    }
    // erase takes no body; save/restore read "filename"
    // (`request_data.at("filename")` — a missing key or a malformed body is
    // the nlohmann exception path → the 500 wrapper of server.cpp:58-69)
    let mut filename = String::new();
    let mut filepath = String::new();
    if action != "erase" {
        let data = match Json::parse(&req.body_str()) {
            Ok(v) => v,
            Err(e) => {
                return Response::error(api::json_error(
                    &json_parse_error_message(&e, &req.body_str()),
                    api::ERROR_TYPE_SERVER.0,
                    api::ERROR_TYPE_SERVER.1,
                ))
            }
        };
        match data.at("filename").map(|v| v.get_str().ok()) {
            Some(Some(f)) => filename = f.to_string(),
            _ => {
                return Response::error(api::json_error(
                    "[json.exception.out_of_range.403] key 'filename' not found",
                    api::ERROR_TYPE_SERVER.0,
                    api::ERROR_TYPE_SERVER.1,
                ))
            }
        }
        // `fs_validate_filename` → "Invalid filename"
        // (server-context.cpp:5294-5297)
        if !engine::fs_validate_filename(&filename) {
            return Response::error(api::json_error(
                "Invalid filename",
                ERROR_TYPE_INVALID_REQUEST.0,
                ERROR_TYPE_INVALID_REQUEST.1,
            ));
        }
        filepath = format!("{}{filename}", server.slot_save_path);
    }
    let kind = match action.as_str() {
        "save" => TaskKind::SlotSave,
        "restore" => TaskKind::SlotRestore,
        _ => TaskKind::SlotErase,
    };
    let (tx, rx) = mpsc::channel::<StreamEvent>();
    let task = Task {
        id: server.new_task_id(),
        index: 0,
        kind,
        tokens: Vec::new(),
        params: TaskParams::default(),
        prompt_text: String::new(),
        slot_action: Some(engine::SlotAction { id_slot, filename, filepath }),
        decision_body: Json::Null,
        tx,
    };
    server.queue.submit(task);
    // the single result frame (rd.next of server-context.cpp:5310-5325)
    for ev in rx.iter() {
        match ev {
            StreamEvent::Frame(f) => return Response::ok(f),
            StreamEvent::Done => break,
            StreamEvent::Ping => {}
        }
    }
    Response::error(api::json_error(
        "no response from the engine",
        api::ERROR_TYPE_SERVER.0,
        api::ERROR_TYPE_SERVER.1,
    ))
}

/// `get_res_props` (server-context.cpp:4599-4638)
fn props_json(e: &Engine) -> Json {
    let mut params = TaskParams::default();
    params.n_predict = e.n_predict_default;
    let default_generation_settings = Json::Object(vec![
        ("params".into(), api::task_params_to_json(&params, true)),
        ("n_ctx".into(), Json::Int(e.slots.first().map(|s| s.n_ctx).unwrap_or(0) as i64)),
    ]);
    let mut props = Json::Object(vec![
        ("default_generation_settings".into(), default_generation_settings),
        ("total_slots".into(), Json::Int(e.slots.len() as i64)),
        ("model_alias".into(), Json::String(e.model_name.clone())),
        ("model_ftype".into(), Json::String(e.ftype.clone())),
        ("model_path".into(), Json::String(e.model_path.clone())),
        (
            "modalities".into(),
            Json::Object(vec![
                ("vision".into(), Json::Bool(false)),
                ("video".into(), Json::Bool(false)),
                ("audio".into(), Json::Bool(false)),
            ]),
        ),
        ("media_marker".into(), Json::String(e.media_marker.clone())),
        ("endpoint_slots".into(), Json::Bool(true)),
        ("endpoint_props".into(), Json::Bool(false)),
        ("endpoint_metrics".into(), Json::Bool(false)),
        ("ui".into(), Json::Bool(true)),
        ("ui_settings".into(), Json::Object(Vec::new())),
        ("chat_template".into(), Json::String(e.chat_template.clone())),
        ("chat_template_caps".into(), e.chat_template_caps.clone()),
        ("bos_token".into(), Json::String(e.bos_token.clone())),
        ("eos_token".into(), Json::String(e.eos_token.clone())),
        ("build_info".into(), Json::String(BUILD_INFO.to_string())),
        ("is_sleeping".into(), Json::Bool(false)),
        ("cors_proxy_enabled".into(), Json::Bool(false)),
    ]);
    if !e.chat_template_tool_use.is_empty() {
        props.set("chat_template_tool_use", Json::String(e.chat_template_tool_use.clone()));
    }
    props
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Result<Args, String> {
        let mut argv = vec!["llama-server".to_string()];
        argv.extend(v.iter().map(|s| s.to_string()));
        parse_args(&argv)
    }

    /// `--host`/`--port` handling: loopback by default (server-http.cpp:120-150),
    /// an explicit host is honoured, `--port` defaults to 8080, `--parallel`
    /// defaults to auto (-1) and a missing model is an error.
    #[test]
    fn arg_host_port_and_defaults() {
        let a = args(&["-m", "m.gguf"]).expect("minimal args");
        assert_eq!(a.host, "127.0.0.1");
        assert_eq!(a.port, 8080);
        assert_eq!(a.n_parallel, -1, "-np auto (server.cpp:157-160)");
        assert_eq!(a.n_ctx, 4096);
        assert_eq!(a.n_predict, -1);
        assert!(!a.flash_attn);

        let a = args(&["-m", "m.gguf", "--host", "0.0.0.0", "--port", "9999"]).expect("explicit host");
        assert_eq!(a.host, "0.0.0.0");
        assert_eq!(a.port, 9999);
        assert!(!is_loopback(&a.host));

        assert!(args(&[]).is_err(), "a model is required");
        assert!(args(&["-m", "m.gguf", "-np", "0"]).is_err(), "n_parallel == 0 is rejected");
        assert!(args(&["-m", "m.gguf", "--nope"]).is_err(), "unknown args are rejected");
    }

    /// `GGML_PAD(x, 256)` of `cparams.n_ctx` / `n_ctx_seq`
    /// (llama-context.cpp:289-297) — the slot budget the reference reports.
    #[test]
    fn n_ctx_padding() {
        assert_eq!(pad256(512), 512);
        assert_eq!(pad256(128), 256);
        assert_eq!(pad256(0), 0);
        assert_eq!(pad256(1024), 1024);
        assert_eq!(pad256(1025), 1280);
    }

    /// `--lora` / `--lora-scaled PATH:SCALE`
    /// (`common_set_adapter_lora`'s arguments, common/common.cpp:1667-1676).
    #[test]
    fn lora_args() {
        let a = args(&["-m", "m.gguf", "--lora", "a.gguf", "--lora-scaled", "b.gguf:0.5"]).unwrap();
        assert_eq!(a.lora, vec![("a.gguf".to_string(), 1.0), ("b.gguf".to_string(), 0.5)]);
        assert!(args(&["-m", "m.gguf", "--lora-scaled", "b.gguf"]).is_err());
    }

    /// nlohmann's parse-error text for a truncated body: the port's parser
    /// reports the byte offset, the C reports line/column
    /// (server.cpp:62-79 + `common_json::parse`).
    #[test]
    fn malformed_json_message() {
        let msg = json_parse_error_message(
            "[json.exception.parse_error.101] parse error at byte 10: unexpected end of input",
            "{\"prompt\":",
        );
        assert!(
            msg.starts_with("[json.exception.parse_error.101] parse error at line 1, column 11:"),
            "{msg}"
        );
        let msg2 = json_parse_error_message(
            "[json.exception.parse_error.101] parse error at byte 7: unexpected end of input",
            "{\n\"a\": ",
        );
        assert!(msg2.contains("line 2, column 6"), "{msg2}");
    }
}
// ---------------------------------------------------------------------------
// arch batch 15 (2026-10) — the P1+P2 weight bundles (verbatim copies of
// llama-cli's wiring; the server and the CLI drive the identical graph)
// ---------------------------------------------------------------------------

/// qwen.cpp:13-37
fn batch15_s_qwen1(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Qwen1ModelWeights {
    llama::graph_arch::Qwen1ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::Qwen1LayerWeights {
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
fn batch15_s_maincoder(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::MaincoderModelWeights {
    llama::graph_arch::MaincoderModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::MaincoderLayerWeights {
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
fn batch15_s_pangu_embed(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::PanguEmbedModelWeights {
    llama::graph_arch::PanguEmbedModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::PanguEmbedLayerWeights {
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
fn batch15_s_cogvlm(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::CogvlmModelWeights {
    llama::graph_arch::CogvlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::CogvlmLayerWeights {
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
fn batch15_s_spark25(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Spark25ModelWeights {
    llama::graph_arch::Spark25ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::Spark25LayerWeights {
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
fn batch15_s_muse_glimmer(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::MuseGlimmerModelWeights {
    llama::graph_arch::MuseGlimmerModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::MuseGlimmerLayerWeights {
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
fn batch15_s_llada(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::LladaModelWeights {
    llama::graph_arch::LladaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        output_b: None,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::LladaLayerWeights {
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
fn batch15_s_plm(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::PlmModelWeights {
    llama::graph_arch::PlmModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::PlmLayerWeights {
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
fn batch15_s_hunyuan_vl(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::HunyuanVlModelWeights {
    llama::graph_arch::HunyuanVlModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::HunyuanVlLayerWeights {
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
fn batch15_s_granite_swa(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::GraniteSwaModelWeights {
    llama::graph_arch::GraniteSwaModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::GraniteSwaLayerWeights {
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
fn batch15_s_afmoe(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::AfmoeModelWeights {
    llama::graph_arch::AfmoeModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::AfmoeLayerWeights {
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
fn batch15_s_mellum(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::MellumModelWeights {
    llama::graph_arch::MellumModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::MellumLayerWeights {
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
fn batch15_s_paddleocr(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::PaddleOcrModelWeights {
    llama::graph_arch::PaddleOcrModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::PaddleOcrLayerWeights {
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
fn batch15_s_hy_v3(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::HyV3ModelWeights {
    llama::graph_arch::HyV3ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::HyV3LayerWeights {
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
fn batch15_s_mimo2(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Mimo2ModelWeights {
    llama::graph_arch::Mimo2ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::Mimo2LayerWeights {
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
fn batch15_s_step35(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::Step35ModelWeights {
    llama::graph_arch::Step35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        layers: m
            .layers
            .iter()
            .take(n_trunk)
            .map(|l| llama::graph_arch::Step35LayerWeights {
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
fn batch15_s_hy_v4(m: &LlamaModel, n_trunk: usize) -> llama::graph_arch::HyV4ModelWeights {
    llama::graph_arch::HyV4ModelWeights {
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
            .map(|l| llama::graph_arch::HyV4LayerWeights {
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
