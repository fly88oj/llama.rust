//! llama-bench (rust) — port of `tools/llama-bench/llama-bench.cpp` (pinned
//! bd4f514db1).
//!
//! Structure, flags, defaults, the test matrix, the measurement protocol and
//! the five output formats (csv/json/jsonl/md/sql) follow the C literally; the
//! line references in this file are llama-bench.cpp unless stated otherwise.
//!
//! Ported:
//!   * `-m/-p/-n/-pg/-d/-b/-ub/-t/-r/-fa/-o/-oe/-v/--progress/--no-warmup/
//!     --delay/--list-devices/--version/--numa/-C/--cpu-strict/--poll/-ngl/
//!     -ncmoe/-sm/-lm/-lzm/-mg/-nkvo/-dev/-ts/-ot/-embd/-nopo/--no-host/
//!     -fitt/-fitc/-ctk/-ctv` — all parsed, all matrix dimensions kept, so the
//!     instance count, ordering and table columns match the C;
//!   * the timings: warmup (unless `--no-warmup`), `llama_memory_clear` before
//!     every repeat, one `test` per (model, …, n_prompt|n_gen|pp+tg) with the
//!     wall clock around the decode calls only, `avg`/`stdev` ported literally
//!     (u64 `avg_ns`/`stddev_ns`, double `avg_ts`/`stddev_ts`);
//!   * `-d/--n-depth` processes the depth tokens before the timed region.
//!
//! Not ported (see PARITY.md for the full list): the GPU/backend-selectable
//! machinery (`-ngl`, `-sm`, `-dev`, `-ts`, `-ot`, `-nkvo`, `-nopo`,
//! `--no-host`, `-ncmoe` — the CPU backend ignores them just like the C's CPU
//! build does), quantized KV cache types (`-ctk/-ctv`, this engine's cache is
//! F16), the threadpool knobs (`-C/--cpu-strict/--poll/--prio/--numa`),
//! `-embd/--embeddings`, `-lzm`, `-lm` beyond mmap, `-fitt/-fitc`, the
//! HuggingFace download flags (`-hf/-hff/-hft/--offline` are hard errors) and
//! the `-d` context-state cache (`llama_state_seq_get/set_data`); the depth is
//! simply re-processed, which is what the C does on a state mismatch too.
//! `llama_synchronize` is a no-op here (the port's `graph_compute` is
//! synchronous) and the model is reloaded per instance instead of being kept
//! while `equal_mparams` holds (:2311-2322) — both are outside the timed region.
//!
//! One behaviour deliberately resolved differently: `-fa auto` (the default)
//! resolves to **enabled**, matching the reference on this CPU
//! (`llama_context::resolve_fused_ops`'s probe, llama-context.cpp:503-560:
//! default and `-fa on` produce the same graph); the other tools of this port
//! map `auto` to off.
//!
//! rust-only flags: `--rust-token-counts` prints the tokens each test really
//! evaluated (stderr), for the token-count parity check.

mod archs;
mod engine;
mod model_info;
mod params;
mod report;
mod util;

use std::sync::Arc;

use llama::model::load_model;
use llama::vocab::Vocab;

use params::{CmdParams, CmdParamsInstance};
use report::{test_name, Printer, Test};
use util::GlibcRand;

/// `llama_commit()` / `llama_build_number()` / `llama_version()` of the pinned
/// revision (printed by `--version` and as `build_commit`/`build_number` in
/// every output format, :1765-1766 / :2245).
pub const BUILD_COMMIT: &str = "def4d406a";
pub const BUILD_NUMBER: i32 = 11325;
pub const BUILD_VERSION: &str = "0.5.0-dev";

/// a printer's target stream, buffered per header/test like the C's `fprintf`
/// plus `fflush` (llama-bench.cpp:2263-2276 / :2477-2484)
struct Sink {
    buf: String,
    stderr: bool,
}

impl Sink {
    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        use std::io::Write;
        if self.stderr {
            let mut e = std::io::stderr();
            let _ = e.write_all(self.buf.as_bytes());
            let _ = e.flush();
        } else {
            let mut o = std::io::stdout();
            let _ = o.write_all(self.buf.as_bytes());
            let _ = o.flush();
        }
        self.buf.clear();
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let params = match params::parse_cmd_params(&argv) {
        Ok(p) => p,
        Err(e) => {
            if let Some(detail) = &e.detail {
                eprintln!("{detail}");
            }
            eprintln!("error: invalid parameter for argument: {}", e.arg);
            params::print_usage(&argv[0]);
            std::process::exit(1);
        }
    };
    params::warn_ignored(&params);

    // `if (!params.verbose) llama_log_set(llama_null_log_callback, NULL)`
    // (llama-bench.cpp:2285-2287): silence the library logs, but — since the
    // upstream verbosity fix (#28229) — GGML_LOG_ERROR still passes through
    fn null_log_callback(level: llama::impl_log::LogLevel, text: &str, _ud: usize) {
        if level == llama::impl_log::LogLevel::Error {
            eprint!("{text}");
        }
    }
    if !params.verbose {
        llama::impl_log::log_set(Some(null_log_callback), 0);
    }

    // initialize printers (:2263-2276)
    let mut p = Printer::create(params.output_format);
    let mut p_err = Printer::create(params.output_format_stderr);
    let mut sink = Sink { buf: String::new(), stderr: false };
    let mut sink_err = Sink { buf: String::new(), stderr: true };
    if let Some(pr) = p.as_mut() {
        pr.print_header(&params, &mut sink.buf);
        sink.flush();
    }
    if let Some(pr) = p_err.as_mut() {
        pr.print_header(&params, &mut sink_err.buf);
        sink_err.flush();
    }

    let instances = params::get_cmd_params_instances(&params);
    let params_count = instances.len();

    for (idx, inst) in instances.iter().enumerate() {
        let params_idx = idx + 1;
        if params.progress {
            eprintln!("llama-bench: benchmark {params_idx}/{params_count}: starting");
        }
        let t = match run_instance(&params, inst, params_idx, params_count) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        if let Some(pr) = p.as_mut() {
            pr.print_test(&t, &mut sink.buf);
            sink.flush();
        }
        if let Some(pr) = p_err.as_mut() {
            pr.print_test(&t, &mut sink_err.buf);
            sink_err.flush();
        }
        if params.rust_token_counts {
            let fmt = |v: &[u64]| format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
            eprintln!(
                "#rust-token-counts: test={} n_prompt={} n_gen={} n_depth={} reps={} evaluated_prompt={} evaluated_gen={} evaluated_depth={}",
                test_name(&t),
                t.n_prompt,
                t.n_gen,
                t.n_depth,
                params.reps,
                fmt(&t.evaluated_prompt),
                fmt(&t.evaluated_gen),
                fmt(&t.evaluated_depth)
            );
        }
    }

    if let Some(pr) = p.as_mut() {
        pr.print_footer(&mut sink.buf);
        sink.flush();
    }
    if let Some(pr) = p_err.as_mut() {
        pr.print_footer(&mut sink_err.buf);
        sink_err.flush();
    }
}

/// one `cmd_params_instance`: load, build the context, warm up, repeat
/// (:2296-2490)
fn run_instance(
    params: &CmdParams,
    inst: &CmdParamsInstance,
    params_idx: usize,
    params_count: usize,
) -> Result<Test, String> {
    fn bench_err(msg: String) -> String {
        format!("llama_bench: error: {msg}")
    }

    // `mparams.use_extra_bufts = repack` (llama-bench.cpp:1255): the port's
    // repack cache gates on the same switch (the model load repacks through
    // the lazy cache on first forward)
    ggml::repack::set_repack_override(Some(inst.repack));

    let gguf = ggml::Gguf::open(&inst.model).map_err(|e| bench_err(format!("failed to load model '{}': {e}", inst.model)))?;
    let mmap: Arc<memmap2::Mmap> = {
        let f = std::fs::File::open(&inst.model)
            .map_err(|e| bench_err(format!("failed to open '{}': {e}", inst.model)))?;
        // SAFETY: read-only usage of a model file
        Arc::new(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| bench_err(format!("failed to mmap '{}': {e}", inst.model)))?)
    };
    let vocab = Vocab::load(&gguf).map_err(|e| bench_err(format!("failed to load vocabulary: {e}")))?;
    let model = load_model(&gguf, mmap).map_err(|e| bench_err(format!("failed to load model '{}': {e}", inst.model)))?;

    let n_vocab_vocab = vocab.id_to_token.len();
    let model_type = model_info::model_type(model.arch, &model.hparams, n_vocab_vocab as u32, &gguf);
    let (model_size, model_n_params) = model_info::size_and_params(&gguf);

    let flash_attn = inst.flash_attn.enabled();
    let attn = archs::build_attn(&model.hparams, model.arch, flash_attn);
    if params.verbose {
        eprintln!(
            "llama_bench: verbose: model '{}' type '{}' size {} params {}",
            inst.model, model_type, model_size, model_n_params
        );
        eprintln!(
            "llama_bench: verbose: n_ctx = {}, n_batch = {}, n_ubatch = {}, n_threads = {}",
            inst.n_ctx(),
            inst.n_batch,
            inst.n_ubatch,
            inst.n_threads
        );
        if inst.flash_attn == params::FlashAttnType::Auto {
            eprintln!("resolve_fused_ops: Flash Attention enabled (auto, CPU probe)");
        }
    }
    let weights = archs::build_weights(&model, attn)?;

    // llama_init_from_model + llama_set_n_threads: the port's driver takes the
    // thread count and the ubatch at construction (:2326-2331 / :2346-2366)
    let n_ctx = inst.n_ctx().max(1) as u32;
    let n_threads = inst.n_threads.max(0) as usize;
    let n_ubatch = inst.n_ubatch.max(1);
    let mut dctx = llama::context::DecodeContext::new_with(
        model.ctx,
        weights,
        attn,
        n_ctx,
        n_threads,
        n_ubatch as usize,
    );

    // the C uses `llama_vocab_n_tokens`; the port's `decode` indexes the
    // embedding rows, so a vocab longer than the model's output tensor would
    // read out of bounds — clamp (never happens for a well-formed file)
    let n_vocab_model = dctx.n_vocab() as i32;
    let mut n_vocab = n_vocab_vocab as i32;
    if n_vocab > n_vocab_model {
        eprintln!(
            "llama_bench: WARNING: vocabulary has {n_vocab} tokens but the model has {n_vocab_model} embeddings; clamping the random tokens"
        );
        n_vocab = n_vocab_model;
    }
    let add_bos = vocab.get_add_bos();
    let bos = vocab.token_bos();

    // the KV cache of this engine is F16 whatever `-ctk/-ctv` asked for
    // (`warn_ignored` reports the mismatch): the row prints the type that was
    // really used, where the C would print the requested one
    let mut reported = inst.clone();
    reported.type_k = ggml::types::GgmlType::F16;
    reported.type_v = ggml::types::GgmlType::F16;
    let mut t = Test::new(&reported, model_type, model_size, model_n_params);

    // llama_memory_clear(llama_get_memory(ctx), false) (:2338)
    dctx.reset_sequence();

    // cool off before the test (:2340-2343)
    if params.delay > 0 {
        std::thread::sleep(std::time::Duration::from_secs(params.delay as u64));
    }

    let mut rng = GlibcRand::new();
    let mut pos = 0i32;

    // warmup run (:2368-2394); its token counts are not part of the
    // rust-only diagnostic (only the timed repeats are)
    if !params.no_warmup {
        let mut warmup_evaluated = 0u64;
        if t.n_prompt > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {params_idx}/{params_count}: warmup prompt run");
            }
            engine::test_prompt(
                &mut dctx, n_vocab, add_bos, bos, &mut rng, t.n_prompt, t.n_batch, n_ubatch, &mut pos, &mut warmup_evaluated,
            )
            .map_err(|e| format!("llama_bench: error: failed to run prompt warmup: {e}"))?;
        }
        if t.n_gen > 0 {
            if params.progress {
                eprintln!("llama-bench: benchmark {params_idx}/{params_count}: warmup generation run");
            }
            engine::test_gen(&mut dctx, n_vocab, add_bos, bos, &mut rng, 1, &mut pos, &mut warmup_evaluated)
                .map_err(|e| format!("llama_bench: error: failed to run gen warmup: {e}"))?;
        }
    }

    for i in 0..params.reps {
        // llama_memory_clear + the depth run (:2396-2437)
        dctx.reset_sequence();
        pos = 0;
        let mut depth_evaluated = 0u64;
        if t.n_depth > 0 {
            if params.progress {
                eprintln!(
                    "llama-bench: benchmark {params_idx}/{params_count}: depth run {}/{reps}",
                    i + 1,
                    reps = params.reps
                );
            }
            engine::test_prompt(
                &mut dctx, n_vocab, add_bos, bos, &mut rng, t.n_depth, t.n_batch, n_ubatch, &mut pos, &mut depth_evaluated,
            )
            .map_err(|e| format!("llama_bench: error: failed to run depth: {e}"))?;
        }

        let t_start = util::get_time_ns();

        let mut prompt_evaluated = 0u64;
        let mut gen_evaluated = 0u64;
        if t.n_prompt > 0 {
            if params.progress {
                eprintln!(
                    "llama-bench: benchmark {params_idx}/{params_count}: prompt run {}/{reps}",
                    i + 1,
                    reps = params.reps
                );
            }
            engine::test_prompt(
                &mut dctx, n_vocab, add_bos, bos, &mut rng, t.n_prompt, t.n_batch, n_ubatch, &mut pos, &mut prompt_evaluated,
            )
            .map_err(|e| format!("llama_bench: error: failed to run prompt: {e}"))?;
        }
        if t.n_gen > 0 {
            if params.progress {
                eprintln!(
                    "llama-bench: benchmark {params_idx}/{params_count}: generation run {}/{reps}",
                    i + 1,
                    reps = params.reps
                );
            }
            engine::test_gen(&mut dctx, n_vocab, add_bos, bos, &mut rng, t.n_gen, &mut pos, &mut gen_evaluated)
                .map_err(|e| format!("llama_bench: error: failed to run gen: {e}"))?;
        }
        t.evaluated_prompt.push(prompt_evaluated);
        t.evaluated_gen.push(gen_evaluated);
        t.evaluated_depth.push(depth_evaluated);

        let t_ns = util::get_time_ns() - t_start;
        t.samples_ns.push(t_ns);
    }

    Ok(t)
}