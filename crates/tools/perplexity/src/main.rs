//! llama-perplexity (rust) — port of `tools/perplexity/perplexity.cpp`
//! (pinned bd4f514db1): the non-strided `perplexity()` v1 path plus the three
//! extra scorers (`hellaswag_score` :744, `winogrande_score` :1101,
//! `multiple_choice_score` :1405 — see src/scorers.rs).
//!
//! Reference semantics reproduced line-by-line from the C source
//! (perplexity.cpp:444-662 + `llama_perplexity()` main):
//!
//!   * `params.n_ctx` defaults to 512, `params.escape = false` (so -p/-f are
//!     NOT escape-processed; `-f` strips one trailing newline, common/arg.cpp
//!     read_file + -f handler).
//!   * tokens = `common_tokenize(ctx, prompt, add_special = true,
//!     parse_special = false)`; the tool needs `tokens >= 2*n_ctx` tokens or
//!     it errors out (and still exits 0, like the reference).
//!   * `n_chunk_max = tokens / n_ctx`; `n_chunk = --chunks < 0 ? n_chunk_max :
//!     min(--chunks, n_chunk_max)`; chunks are consecutive non-overlapping
//!     `[i*n_ctx, (i+1)*n_ctx)` windows.
//!   * `first = n_ctx/2`: only the last half of each window is scored, so every
//!     scored token has at least n_ctx/2 tokens of left context. The window's
//!     first token is replaced by BOS when `vocab.add_bos` (restored after).
//!   * for i in 0..n_ctx-1-first: logits row at position `first+i` vs target
//!     token `first+i+1`; `nll += log_softmax(row, target)` where
//!     log_softmax = (logit[tok] - max) - log(sum exp(logit[i] - max)) — the
//!     exact `log_softmax(n_vocab, logits, tok)` of perplexity.cpp:60 evaluated
//!     in f32 expf / f64 accumulation.
//!   * stdout: `[<chunk+1>]<ppl so far>,` after each chunk (cumulative nll over
//!     cumulative count), then a newline; stderr: the "Final estimate: PPL =
//!     %.4lf +/- %.5lf" line with the standard deviation of log-prob.
//!   * scorer mode wiring (llama_perplexity, :2031-2038): with
//!     `--hellaswag/--winogrande/--multiple-choice` the context is
//!     `n_parallel = max(4, -np)` sequences wide (`kv_unified`), so
//!     `llama_n_ctx()` inside the scorer = n_parallel * n_ctx and
//!     `llama_n_seq_max()` = n_parallel; plain PPL uses n_parallel =
//!     max(1, n_batch/n_ctx) (the port runs its chunks at n_seq=1).
//!
//! Differences vs the C tool (documented, not observable in the PPL value):
//!   * chunks run sequentially in one sequence (`n_seq = 1`) instead of
//!     `n_batch/n_ctx` parallel sequences; the KV is cleared between chunks
//!     (`llama_memory_clear`) exactly like the reference, and the PPL math is
//!     per-chunk independent, so the values are the same.
//!   * no `--ppl-stride` v2 path, no KL-divergence-base mode, no
//!     `--save-all-logits` dump (the reference's binary log-prob table);
//!     `--dump-nll FILE` instead writes the per-scored-token NLL as f64 LE,
//!     which is what the reference's table encodes (row i = target token
//!     `first+i+1`), so per-position alignment can be diffed directly.
//!   * `-b/--batch-size` is accepted for CLI compatibility; decoding is driven
//!     by `-c` (one ubatch per half-window).
//!   * the context/graph is our Rust port (`DecodeContext::decode_all`), so the
//!     printed values track our logits (see the alignment report).

mod scorers;

use std::io::Write as _;
use std::sync::Arc;

use llama::arch::LlmArch;
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::model::load_model;
use llama::vocab::Vocab;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlashAttn {
    Off,
    On,
    /// parsed like the reference, treated as off (no CPU FA probe)
    Auto,
}

impl FlashAttn {
    fn on(self) -> bool {
        matches!(self, FlashAttn::On)
    }
}

struct Args {
    model: Option<String>,
    file: Option<String>,
    prompt: Option<String>,
    n_ctx: i32,
    n_chunks: i32,
    n_batch: i32,
    n_threads: usize,
    flash_attn: FlashAttn,
    /// rust-only diagnostic: dump the per-scored-token NLL (f64 LE) in order,
    /// to diff against the reference's `--logits-file` quantized log-probs
    dump_nll: Option<String>,
    // --- the three scorer modes (perplexity.cpp:2031-2089) ---
    hellaswag: bool,
    hellaswag_tasks: usize, // common.h default 400
    winogrande: bool,
    winogrande_tasks: usize, // 0 = all
    multiple_choice: bool,
    multiple_choice_tasks: usize, // 0 = all
    /// `-np/--parallel` (common_params.n_parallel default 1)
    n_parallel: i32,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: None,
            file: None,
            prompt: None,
            // llama_perplexity(): params.n_ctx = 512
            n_ctx: 512,
            // common_params.n_chunks = -1 (unlimited)
            n_chunks: -1,
            // common_params.n_batch = 2048
            n_batch: 2048,
            n_threads: 0,
            flash_attn: FlashAttn::Auto,
            dump_nll: None,
            hellaswag: false,
            hellaswag_tasks: 400,
            winogrande: false,
            winogrande_tasks: 0,
            multiple_choice: false,
            multiple_choice_tasks: 0,
            n_parallel: 1,
        }
    }
}

fn usage(prog: &str) {
    println!("usage: {prog} -m model.gguf -f wiki.test.raw [-c 512] [--chunks N]");
    println!();
    println!("  -m, --model FILE     model to load");
    println!("  -f, --file FILE      text file to evaluate (one trailing newline stripped)");
    println!("  -p, --prompt TEXT    prompt text instead of a file");
    println!("  -c, --ctx-size N     context window per chunk (default: 512)");
    println!("  --chunks N           max number of chunks to process (default: -1 = all)");
    println!("  -b, --batch-size N   accepted for compatibility (default: 2048)");
    println!("  -t, --threads N      CPU threads (default: all)");
    println!("  -fa, --flash-attn [on|off|auto]  (default: auto -> off)");
    println!("  --dump-nll FILE      rust-only diagnostic: per-scored-token NLL (f64 LE)");
    println!("  --hellaswag          compute HellaSwag score over random tasks from datafile supplied with -f");
    println!("  --hellaswag-tasks N  number of tasks to use for the HellaSwag score (default: 400)");
    println!("  --winogrande         compute Winogrande score over random tasks from datafile supplied with -f");
    println!("  --winogrande-tasks N number of tasks to use for the Winogrande score (default: 0=all)");
    println!("  --multiple-choice    compute multiple choice score over random tasks from datafile supplied with -f");
    println!("  --multiple-choice-tasks N  number of tasks to use (default: 0=all)");
    println!("  -np, --parallel N    number of parallel sequences (default: 1; scorers force >= 4)");
    println!("  -h, --help           this help");
}

fn parse_i32(name: &str, v: &str) -> i32 {
    match v.parse::<i32>() {
        Ok(x) => x,
        Err(_) => {
            eprintln!("error: invalid value for {name}: '{v}'");
            std::process::exit(1);
        }
    }
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].clone();
        let mut next = |name: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("error: {name} requires an argument"))
        };
        let s = arg.as_str();
        match s {
            "-h" | "--help" | "--usage" => {
                usage(&argv[0]);
                std::process::exit(0);
            }
            "-m" | "--model" => a.model = Some(next(s)?),
            "-f" | "--file" => a.file = Some(next(s)?),
            "-p" | "--prompt" => a.prompt = Some(next(s)?),
            "-c" | "--ctx-size" => a.n_ctx = parse_i32(s, &next(s)?),
            "--chunks" => a.n_chunks = parse_i32(s, &next(s)?),
            "-b" | "--batch-size" => a.n_batch = parse_i32(s, &next(s)?),
            "-t" | "--threads" => a.n_threads = parse_i32(s, &next(s)?).max(1) as usize,
            "-fa" | "--flash-attn" => {
                let v = next(s)?;
                a.flash_attn = match v.as_str() {
                    "on" | "1" => FlashAttn::On,
                    "off" | "0" => FlashAttn::Off,
                    "auto" => FlashAttn::Auto,
                    other => return Err(format!("error: invalid value for {s}: '{other}'")),
                };
            }
            "--dump-nll" => a.dump_nll = Some(next(s)?),
            "--hellaswag" => a.hellaswag = true,
            "--winogrande" => a.winogrande = true,
            "--multiple-choice" => a.multiple_choice = true,
            "--hellaswag-tasks" => a.hellaswag_tasks = next(s)?.parse().unwrap_or(0),
            "--winogrande-tasks" => a.winogrande_tasks = next(s)?.parse().unwrap_or(0),
            "--multiple-choice-tasks" => a.multiple_choice_tasks = next(s)?.parse().unwrap_or(0),
            "-np" | "--parallel" | "-parallel" => a.n_parallel = parse_i32(s, &next(s)?).max(1),
            _ => {
                if let Some(v) = s.strip_prefix("--model=") {
                    a.model = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--file=") {
                    a.file = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--prompt=") {
                    a.prompt = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--ctx-size=") {
                    a.n_ctx = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--chunks=") {
                    a.n_chunks = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--batch-size=") {
                    a.n_batch = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--threads=") {
                    a.n_threads = parse_i32(s, v).max(1) as usize;
                } else if let Some(v) = s.strip_prefix("--dump-nll=") {
                    a.dump_nll = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--hellaswag-tasks=") {
                    a.hellaswag_tasks = v.parse().unwrap_or(0);
                } else if let Some(v) = s.strip_prefix("--winogrande-tasks=") {
                    a.winogrande_tasks = v.parse().unwrap_or(0);
                } else if let Some(v) = s.strip_prefix("--multiple-choice-tasks=") {
                    a.multiple_choice_tasks = v.parse().unwrap_or(0);
                } else {
                    return Err(format!("error: unknown option '{s}'"));
                }
            }
        }
        i += 1;
    }
    Ok(a)
}

/// unit test helper: builds the args for a command line
#[cfg(test)]
fn parse_for_test(argv: &[String]) -> Result<Args, String> {
    parse_args(argv)
}

/// perplexity.cpp:60 — returns the *negative* log-likelihood of `tok`
/// (the C code returns `log_softmax = logit - log_sum_exp` and the caller
/// negates it: `const double v = -results.log_softmax`).
fn nll_of_row(row: &[f32], tok: usize) -> f64 {
    let mut max_logit = row[0];
    for &v in &row[1..] {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum_exp = 0.0f64;
    for &v in row {
        sum_exp += (v - max_logit).exp() as f64;
    }
    let log_softmax = (row[tok] - max_logit) as f64 - sum_exp.ln();
    -log_softmax
}

fn build_attn(hp: &llama::hparams::LlamaHparams, flash_attn: bool) -> AttnParams {
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn: flash_attn,
    }
}

/// Arch dispatch copied from crates/tools/llama-cli (same supported set:
/// qwen2/llama/phi3/gemma2/gemma3).
fn build_weights(model: &llama::model::LlamaModel, attn: AttnParams) -> ForwardWeights {
    match model.arch {
        LlmArch::QWEN2 => {
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
            ForwardWeights::Qwen2(ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                layers,
            })
        }
        LlmArch::LLAMA => {
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
            ForwardWeights::Llama(llama::graph_arch::LlamaModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        LlmArch::PHI3 => {
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
            ForwardWeights::Phi3(llama::graph_arch::Phi3ModelWeights {
                tok_embd: model.tok_embd,
                output_norm: model.output_norm,
                output: model.output,
                output_b: model.output_b,
                layers,
            })
        }
        LlmArch::GEMMA2 | LlmArch::GEMMA3 => {
            let gemma2 = model.arch == LlmArch::GEMMA2;
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
            let hp = &model.hparams;
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
                ForwardWeights::Gemma2(w, gp)
            } else {
                ForwardWeights::Gemma3(w, gp)
            }
        }
        other => {
            eprintln!("perplexity: arch {other:?} has a loader but no forward graph yet (see FILE_MAP.md)");
            std::process::exit(1);
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // llama_perplexity(): "perplexity tool requires '--ctx-size' > 0"
    let n_ctx = args.n_ctx;
    if n_ctx <= 0 {
        eprintln!("perplexity: perplexity tool requires '--ctx-size' > 0");
        std::process::exit(1);
    }
    let n_ctx = n_ctx as usize;

    // prompt bytes: -f (read raw — the multiple-choice dataset is binary! —
    // and strip one trailing newline, the reference's read_file + -f handler)
    // else -p
    let prompt_bytes: Vec<u8> = if let Some(f) = &args.file {
        match std::fs::read(f) {
            Ok(mut s) => {
                if s.last() == Some(&b'\n') {
                    s.pop();
                }
                s
            }
            Err(e) => {
                eprintln!("error: failed to open file '{f}': {e}");
                std::process::exit(1);
            }
        }
    } else {
        args.prompt.clone().unwrap_or_default().into_bytes()
    };

    let Some(model_path) = args.model.clone() else {
        eprintln!("error: no model specified (-m/--model)");
        std::process::exit(1);
    };

    let gguf = match ggml::Gguf::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("perplexity: unable to load model '{model_path}': {e}");
            std::process::exit(1);
        }
    };
    let mmap: Arc<memmap2::Mmap> = {
        let f = std::fs::File::open(&model_path).unwrap();
        // SAFETY: read-only usage of a model file
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let vocab = match Vocab::load(&gguf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("perplexity: failed to load vocabulary: {e}");
            std::process::exit(1);
        }
    };
    let model = match load_model(&gguf, mmap) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("perplexity: failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let n_threads = if args.n_threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    } else {
        args.n_threads
    };

    // ---- the three scorer modes (llama_perplexity, perplexity.cpp:2031-2038
    // + :2079-2089): n_parallel = max(4, -np), unified KV, context widened to
    // n_parallel * n_ctx ----
    if args.hellaswag || args.winogrande || args.multiple_choice {
        let n_parallel = std::cmp::max(4, args.n_parallel);
        let n_ctx_total = (n_parallel as usize) * n_ctx;
        let n_batch = (args.n_batch.max(1) as usize).min(n_ctx_total);
        // the port's DecodeContext batch is the reference's `n_ubatch`
        // (default 512 — cparams; -b only drives the reference's logical
        // batch split in decode_helper, which the port does internally)
        let n_ubatch = 512usize;

        let attn = build_attn(&model.hparams, args.flash_attn.on());
        let weights = build_weights(&model, attn);
        let mut dctx = DecodeContext::new_with(
            model.ctx,
            weights,
            attn,
            n_ctx_total as u32,
            n_threads,
            n_ubatch,
        );

        let sparams = scorers::ScorerParams {
            n_ctx: n_ctx_total,
            n_seq_max: n_parallel as usize,
            n_batch,
            hellaswag_tasks: args.hellaswag_tasks,
        };
        if args.hellaswag {
            let prompt = String::from_utf8_lossy(&prompt_bytes);
            scorers::hellaswag_score(&mut dctx, &vocab, &sparams, &prompt);
        } else if args.winogrande {
            let prompt = String::from_utf8_lossy(&prompt_bytes);
            scorers::winogrande_score(&mut dctx, &vocab, &sparams, &prompt, args.winogrande_tasks);
        } else {
            scorers::multiple_choice_score(
                &mut dctx,
                &vocab,
                &sparams,
                &prompt_bytes,
                args.multiple_choice_tasks,
            );
        }
        // LOG("\n") after the scorer returns (perplexity.cpp:2091); the
        // reference's llama_perf_context_print / memory breakdown lines are
        // not ported (timings, not scores)
        println!();
        return;
    }

    // ---- chunked perplexity path ----
    let prompt: String = String::from_utf8_lossy(&prompt_bytes).into_owned();

    // perplexity.cpp:457-458
    let add_bos = vocab.get_add_bos();
    if vocab.get_add_eos() {
        eprintln!("perplexity: warning: model sets add_eos; the reference asserts !add_eos here");
    }

    eprintln!("perplexity: tokenizing the input ..");
    // common_tokenize(ctx, params.prompt, true) — parse_special defaults false
    let tokens: Vec<i32> = vocab.tokenize(&prompt, true, false);

    if (tokens.len() as usize) < 2 * n_ctx {
        eprintln!(
            "perplexity: you need at least {} tokens to evaluate perplexity with a context of {}",
            2 * n_ctx,
            n_ctx
        );
        eprintln!(
            "perplexity: the data file you provided tokenizes to only {} tokens",
            tokens.len()
        );
        std::process::exit(0); // reference returns 0 here too
    }

    let n_chunk_max = tokens.len() / n_ctx;
    let n_chunk = if args.n_chunks < 0 {
        n_chunk_max
    } else {
        (args.n_chunks as usize).min(n_chunk_max)
    };
    let n_batch = args.n_batch.max(1) as usize;
    let n_seq = std::cmp::max(1, n_batch / n_ctx);

    eprintln!(
        "perplexity: calculating perplexity over {n_chunk} chunks, n_ctx={n_ctx}, batch_size={n_batch}, n_seq={n_seq}"
    );
    eprintln!("perplexity: (rust: chunks are evaluated sequentially with n_seq=1)");

    let attn = build_attn(&model.hparams, args.flash_attn.on());
    let weights = build_weights(&model, attn);
    // one ubatch per half-window; the KV holds exactly one chunk
    let mut dctx = DecodeContext::new_with(model.ctx, weights, attn, n_ctx as u32, n_threads, n_ctx);

    let n_vocab = dctx.n_vocab();
    let first = n_ctx / 2;

    let mut count = 0usize;
    let mut nll = 0.0f64;
    let mut nll2 = 0.0f64;
    let mut out = String::new();
    let mut nll_dump: Vec<u8> = Vec::new();

    for i in 0..n_chunk {
        let start = i * n_ctx;
        let mut chunk: Vec<i32> = tokens[start..start + n_ctx].to_vec();

        // "add BOS token for the first batch of each chunk" (perplexity.cpp:569)
        let token_org = chunk[0];
        if add_bos {
            chunk[0] = vocab.token_bos();
        }

        // clear the KV cache between chunks (llama_memory_clear, :553)
        dctx.reset_sequence();

        // decode in two ubatches: the last half is the scored one, and only
        // those rows are materialized (llama.cpp asks for logits from `first`
        // on via batch.logits)
        let pos0: Vec<i32> = (0..first as i32).collect();
        if let Err(e) = dctx.decode(&chunk[..first], &pos0) {
            eprintln!("perplexity : failed to decode: {e}");
            std::process::exit(1);
        }
        let pos1: Vec<i32> = (first as i32..n_ctx as i32).collect();
        let all = match dctx.decode_all(&chunk[first..], &pos1) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("perplexity : failed to decode: {e}");
                std::process::exit(1);
            }
        };

        // rows are positions `first..n_ctx`; row j predicts chunk[first + j + 1]
        for j in 0..(n_ctx - 1 - first) {
            let row = &all[j * n_vocab..(j + 1) * n_vocab];
            let target = chunk[first + j + 1] as usize;
            let v = nll_of_row(row, target);
            nll += v;
            nll2 += v * v;
            count += 1;
            if args.dump_nll.is_some() {
                nll_dump.extend_from_slice(&v.to_le_bytes());
            }
        }

        chunk[0] = token_org; // restore (faithful; the buffer is not reused)

        // "[%d]%.4lf," — cumulative PPL so far
        out.push_str(&format!("[{}]{:.4},", i + 1, (nll / count as f64).exp()));
    }
    out.push('\n');

    if let Some(f) = &args.dump_nll {
        if let Err(e) = std::fs::write(f, &nll_dump) {
            eprintln!("perplexity: failed to write --dump-nll file '{f}': {e}");
        } else {
            eprintln!("perplexity: wrote {} per-token NLL values to {f}", nll_dump.len() / 8);
        }
    }

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
    let _ = lock.flush();

    nll2 /= count as f64;
    nll /= count as f64;
    let ppl = nll.exp();
    nll2 -= nll * nll;
    if nll2 > 0.0 {
        nll2 = (nll2 / (count as f64 - 1.0)).sqrt();
        eprintln!("perplexity: Final estimate: PPL = {ppl:.4} +/- {:.5}", nll2 * ppl);
    } else {
        eprintln!("perplexity: Unexpected negative standard deviation of log(prob)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(logits: &[f32], tok: usize) -> f64 {
        nll_of_row(logits, tok)
    }

    #[test]
    fn nll_matches_log_softmax_definition() {
        // hand-computed log-softmax for a 3-way row
        let logits = [1.0f32, 2.0, 3.0];
        let max = 3.0f32;
        let sum: f64 = [1.0f64, 2.0, 3.0]
            .iter()
            .map(|v| ((*v as f32) - max).exp() as f64)
            .sum();
        for tok in 0..3 {
            let want = -((logits[tok] - max) as f64 - sum.ln());
            assert!((row(&logits, tok) - want).abs() < 1e-12);
        }
        // the most likely token has the smallest nll
        assert!(row(&logits, 2) < row(&logits, 1));
        assert!(row(&logits, 1) < row(&logits, 0));
    }

    #[test]
    fn chunk_count_semantics() {
        let tokens = 3000usize;
        let n_ctx = 512usize;
        let n_chunk_max = tokens / n_ctx; // 5
        assert_eq!(n_chunk_max, 5);
        let pick = |c: i32| if c < 0 { n_chunk_max } else { (c as usize).min(n_chunk_max) };
        assert_eq!(pick(-1), 5);
        assert_eq!(pick(1), 1);
        assert_eq!(pick(9), 5);
    }

    #[test]
    fn arg_parsing() {
        let a = parse_for_test(&[
            "llama-perplexity".into(),
            "-m".into(),
            "m.gguf".into(),
            "-f".into(),
            "t.txt".into(),
            "-c".into(),
            "256".into(),
            "--chunks".into(),
            "2".into(),
            "-t".into(),
            "8".into(),
            "-fa".into(),
            "on".into(),
        ])
        .unwrap();
        assert_eq!(a.model.as_deref(), Some("m.gguf"));
        assert_eq!(a.file.as_deref(), Some("t.txt"));
        assert_eq!(a.n_ctx, 256);
        assert_eq!(a.n_chunks, 2);
        assert_eq!(a.n_threads, 8);
        assert_eq!(a.flash_attn, FlashAttn::On);

        let b = parse_for_test(&["p".into(), "--ctx-size=128".into(), "--chunks=1".into()]).unwrap();
        assert_eq!(b.n_ctx, 128);
        assert_eq!(b.n_chunks, 1);
        assert!(parse_for_test(&["p".into(), "--nope".into()]).is_err());
    }
}