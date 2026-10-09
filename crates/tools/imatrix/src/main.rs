//! llama-imatrix (rust) — port of llama.cpp `tools/imatrix/imatrix.cpp`
//! (pinned bd4f514db1).
//!
//! CLI shape follows the reference `print_usage` (imatrix.cpp:28-36) and the
//! `LLAMA_EXAMPLE_IMATRIX` options of common/arg.cpp:
//!
//!   `llama-imatrix -m model.gguf -f some-text.txt [-o imatrix.gguf]
//!    [--output-format {gguf,dat}] [--no-ppl] [--process-output] [--chunk 123]
//!    [--save-frequency 0] [--output-frequency 10]
//!    [--in-file imatrix-prev-0.gguf ...] [--parse-special] [--show-statistics]`
//!
//! The collector itself lives in `llama::imatrix` (it is also what
//! `llama-quantize --imatrix` loads); this file is the driver: the tokenize /
//! chunk loop of `compute_imatrix` (imatrix.cpp:791-966) and the `main`
//! prelude (imatrix.cpp:1077-1193).
//!
//! Differences vs the C tool (documented, none observable in the GGUF):
//!   * arch coverage: the same forward-graph set as `llama-perplexity`
//!     (qwen2/llama/phi3/gemma2/gemma3) — other archs stop with a message
//!     instead of building a graph;
//!   * the logits/perplexity path is driven by `DecodeContext::decode_batch`
//!     with a real `LlamaBatch` (the C's `common_batch_add` shape), so `n_seq`
//!     parallel sequences and the `j*n_batch + k` position rule are identical;
//!   * `--no-ppl` (the C default is `--ppl`): with ppl on, the logits are read
//!     back exactly like the reference (`llama_get_logits_ith`), otherwise the
//!     extra logits copy is skipped.

use std::process::ExitCode;
use std::sync::Arc;

use ggml::Gguf;
use llama::arch::LlmArch;
use llama::batch::LlamaBatch;
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::{AttnParams, LayerWeights, ModelWeights};
use llama::imatrix::{
    load_imatrix_into, save_imatrix, show_statistics, with_collector, ImatrixParams,
};
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
    out_file: String,
    /// `-c/--ctx-size` (default 512, imatrix.cpp:1084)
    n_ctx: i32,
    /// `-b/--batch-size` (common_params: 2048)
    n_batch: i32,
    /// `-ub/--ubatch-size` (common_params: 512) — the port's max ubatch
    n_ubatch: i32,
    /// `--chunks` (common_params.n_chunks: -1 = all)
    n_chunks: i32,
    /// `--chunk/--from-chunk` (params.i_chunk, imatrix.cpp:809)
    i_chunk: i32,
    n_out_freq: i32,
    n_save_freq: i32,
    imat_dat: i8,
    process_output: bool,
    compute_ppl: bool,
    show_statistics: bool,
    parse_special: bool,
    in_files: Vec<String>,
    /// `--nextn` (arg.cpp:3183-3190, upstream a7b94df2c): `params.load_mtp` —
    /// collect data for the MTP/NextN layers
    nextn: bool,
    /// `-md/--model-draft/--spec-draft-model` — an MTP-only draft file whose
    /// NextN layer collects over the trunk's hidden states
    model_draft: Option<String>,
    n_threads: usize,
    flash_attn: FlashAttn,
    /// `--verbose` count (`LOG_DBGV` threshold, common/arg.cpp)
    verbose: u8,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: None,
            file: None,
            prompt: None,
            // params.out_file = "imatrix.gguf" (imatrix.cpp:1082)
            out_file: "imatrix.gguf".to_string(),
            // params.n_ctx = 512 (imatrix.cpp:1084)
            n_ctx: 512,
            n_batch: 2048,
            n_ubatch: 512,
            n_chunks: -1,
            i_chunk: 0,
            n_out_freq: 10,
            n_save_freq: 0,
            imat_dat: 0,
            process_output: false,
            compute_ppl: true,
            show_statistics: false,
            parse_special: false,
            in_files: Vec::new(),
            nextn: false,
            model_draft: None,
            n_threads: 0,
            flash_attn: FlashAttn::Auto,
            verbose: 0,
        }
    }
}

/// `print_usage` (imatrix.cpp:28-36, upstream a7b94df2c: +--nextn/-md).
fn print_usage(prog: &str) {
    println!();
    println!("example usage:");
    println!();
    println!(
        "    {prog} \\\n       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {{gguf,dat}}] [--no-ppl] \\\n       [--process-output] [--nextn] [--model-draft draft.gguf] [--chunk 123] [--save-frequency 0] \\\n       [--output-frequency 10] [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] \\\n       [--parse-special] [--show-statistics] [...]"
    );
    println!();
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
                print_usage(&argv[0]);
                std::process::exit(0);
            }
            "-m" | "--model" => a.model = Some(next(s)?),
            "-f" | "--file" => a.file = Some(next(s)?),
            "-p" | "--prompt" => a.prompt = Some(next(s)?),
            "-o" | "--output" | "--output-file" => a.out_file = next(s)?,
            "-c" | "--ctx-size" => a.n_ctx = parse_i32(s, &next(s)?),
            "-b" | "--batch-size" => a.n_batch = parse_i32(s, &next(s)?),
            "-ub" | "--ubatch-size" => a.n_ubatch = parse_i32(s, &next(s)?),
            "--chunks" => a.n_chunks = parse_i32(s, &next(s)?),
            "--chunk" | "--from-chunk" => a.i_chunk = parse_i32(s, &next(s)?),
            "-ofreq" | "--output-frequency" => a.n_out_freq = parse_i32(s, &next(s)?),
            "--save-frequency" => a.n_save_freq = parse_i32(s, &next(s)?),
            "--output-format" => {
                let v = next(s)?;
                if v == "gguf" {
                    a.imat_dat = -1;
                } else if v == "dat" {
                    a.imat_dat = 1;
                } else {
                    return Err("error: invalid output format".to_string());
                }
            }
            "--process-output" => a.process_output = true,
            "--nextn" => a.nextn = true,
            "-md" | "--model-draft" | "--spec-draft-model" => a.model_draft = Some(next(s)?),
            "--ppl" => a.compute_ppl = true,
            "--no-ppl" => a.compute_ppl = false,
            "--show-statistics" => a.show_statistics = true,
            "--parse-special" => a.parse_special = true,
            "--verbose" | "-v" => a.verbose += 1,
            "--in-file" => {
                let v = next(s)?;
                // arg.cpp:1820-1831 accepts comma-separated values
                for item in v.split(',') {
                    if item.is_empty() {
                        continue;
                    }
                    if !std::path::Path::new(item).exists() {
                        return Err(format!("error: failed to open file '{item}'\n"));
                    }
                    a.in_files.push(item.to_string());
                }
            }
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
            _ => {
                if let Some(v) = s.strip_prefix("--model=") {
                    a.model = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--file=") {
                    a.file = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--prompt=") {
                    a.prompt = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--output=") {
                    a.out_file = v.to_string();
                } else if let Some(v) = s.strip_prefix("--ctx-size=") {
                    a.n_ctx = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--batch-size=") {
                    a.n_batch = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--ubatch-size=") {
                    a.n_ubatch = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--chunks=") {
                    a.n_chunks = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--chunk=") {
                    a.i_chunk = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--output-frequency=") {
                    a.n_out_freq = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--save-frequency=") {
                    a.n_save_freq = parse_i32(s, v);
                } else if let Some(v) = s.strip_prefix("--output-format=") {
                    match v {
                        "gguf" => a.imat_dat = -1,
                        "dat" => a.imat_dat = 1,
                        _ => return Err("error: invalid output format".to_string()),
                    }
                } else if let Some(v) = s.strip_prefix("--in-file=") {
                    a.in_files.push(v.to_string());
                } else if let Some(v) = s.strip_prefix("--model-draft=") {
                    a.model_draft = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--spec-draft-model=") {
                    a.model_draft = Some(v.to_string());
                } else if let Some(v) = s.strip_prefix("--threads=") {
                    a.n_threads = parse_i32(s, v).max(1) as usize;
                } else {
                    return Err(format!("error: unknown option '{s}'"));
                }
            }
        }
        i += 1;
    }
    Ok(a)
}

/// `build_attn` — AttnParams from hparams (same construction as llama-perplexity,
/// which the arch graphs share).
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

/// `qwen35_weights` (the llama-server/llama-bench dispatch's copy, weights.rs
/// :604-644 — qwen35.cpp:31-153): the trunk slice of the layer table;
/// attention layers carry separate q/k/v + q/k norms, recurrent (gated
/// delta net) layers the fused wqkv + ssm tensors.
fn qwen35_trunk_weights(m: &llama::model::LlamaModel, attn: AttnParams) -> ForwardWeights {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    ForwardWeights::Qwen35(
        llama::graph_arch::Qwen35ModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            cls_out: m.cls_out,
            cls_out_b: m.cls_out_b,
            layers: m.layers[..n_trunk]
                .iter()
                .enumerate()
                .map(|(il, l)| qwen35_layer_of_trunk(il, l))
                .collect(),
        },
        qwen35_params_of(m, attn),
    )
}

/// the trunk twin of [`qwen35_layer_of`] — the required-tensor panics of the
/// server's copy (a trunk file must carry its layers).
fn qwen35_layer_of_trunk(il: usize, l: &llama::model::LayerTensors) -> llama::graph_arch::Qwen35LayerWeights {
    llama::graph_arch::Qwen35LayerWeights {
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
        ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
        ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
        ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
    }
}

/// Arch dispatch (same supported set as llama-perplexity / llama-cli's qwen2,
/// llama, phi3, gemma2/3 arms + the qwen35 trunk the NextN collection needs).
fn build_weights(model: &llama::model::LlamaModel, attn: AttnParams) -> ForwardWeights {
    match model.arch {
        LlmArch::QWEN35 => qwen35_trunk_weights(model, attn),
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
            eprintln!("llama-imatrix: arch {other:?} has a loader but no forward graph yet");
            std::process::exit(1);
        }
    }
}

/// `log_softmax` + the per-token NLL accumulation of `process_logits`
/// (imatrix.cpp:745-789): `nll += -(logit[tok] - max - log(sum exp(logit-max)))`.
fn nll_of_row(row: &[f32], tok: usize) -> (f64, f32, f32) {
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
    let logit = row[tok];
    let prob = ((row[tok] - max_logit).exp() / sum_exp as f32) as f32;
    (-log_softmax, logit, prob)
}

/// `static bool compute_imatrix(llama_context * ctx, const common_params &,
/// const int32_t n_ctx)` (imatrix.cpp:791-966).
#[allow(clippy::too_many_arguments)]

// ---------------------------------------------------------------------------
// MTP/NextN collection (imatrix.cpp:1237-1425, upstream a7b94df2c)
// ---------------------------------------------------------------------------

fn mtp_head_facts(hp: &llama::hparams::LlamaHparams) -> llama::context::MtpHeadFacts {
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

fn mtp_nextn9_of(l: &llama::model::LayerTensors) -> llama::graph_arch::MtpNextn {
    let n = &l.nextn;
    llama::graph_arch::MtpNextn {
        eh_proj: n.eh_proj.expect("nextn.eh_proj"),
        enorm: n.enorm.expect("nextn.enorm"),
        hnorm: n.hnorm.expect("nextn.hnorm"),
        embed_tokens: n.embed_tokens,
        shared_head_head: n.shared_head_head,
        shared_head_norm: n.shared_head_norm,
    }
}

fn qwen35_layer_of(l: &llama::model::LayerTensors) -> llama::graph_arch::Qwen35LayerWeights {
    llama::graph_arch::Qwen35LayerWeights {
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

fn qwen35_mtp_weights(m: &llama::model::LlamaModel) -> llama::graph_arch::Qwen35MtpWeights {
    let il = m.hparams.n_layer() as usize;
    let hp = &m.hparams;
    llama::graph_arch::Qwen35MtpWeights {
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

/// `struct model_file_shape` + `model_read_file_shape`
/// (imatrix.cpp:1318-1408) — the GGUF metadata the `-md`/`--nextn`
/// validation reads.
struct ModelFileShape {
    valid: bool,
    is_split: bool,
    arch: String,
    n_layer_all: u32,
    n_embd_out: u32,
    nextn_idx: std::collections::BTreeSet<u32>,
    has_trunk: bool,
    has_nextn: bool,
    n_nextn_layers: u32,
    first_nextn: String,
}

impl ModelFileShape {
    fn n_trunk(&self) -> u32 {
        if self.n_layer_all >= self.n_nextn_layers {
            self.n_layer_all - self.n_nextn_layers
        } else {
            0
        }
    }
}

fn model_read_file_shape(model_path: &str) -> ModelFileShape {
    let mut shape = ModelFileShape {
        valid: false,
        is_split: false,
        arch: String::new(),
        n_layer_all: 0,
        n_embd_out: 0,
        nextn_idx: std::collections::BTreeSet::new(),
        has_trunk: false,
        has_nextn: false,
        n_nextn_layers: 0,
        first_nextn: String::new(),
    };

    let Ok(gguf) = Gguf::open(model_path) else {
        return shape;
    };
    shape.valid = true;

    if let Some(v) = gguf.find_key("general.architecture").and_then(|v| v.as_str()) {
        shape.arch = v.to_string();
        let prefix = format!("{}.", shape.arch);

        if let Some(ggml::gguf::Value::U32(n)) = gguf.find_key(&format!("{prefix}block_count")) {
            shape.n_layer_all = *n;
        }
        if let Some(ggml::gguf::Value::U32(n)) =
            gguf.find_key(&format!("{prefix}nextn_predict_layers"))
        {
            shape.n_nextn_layers = *n;
        }
        if let Some(ggml::gguf::Value::U32(n)) =
            gguf.find_key(&format!("{prefix}embedding_length_out"))
        {
            shape.n_embd_out = *n;
        } else if let Some(ggml::gguf::Value::U32(n)) =
            gguf.find_key(&format!("{prefix}embedding_length"))
        {
            shape.n_embd_out = *n;
        }
    }

    if let Some(ggml::gguf::Value::U16(n)) = gguf.find_key("split.count") {
        shape.is_split = *n > 1;
    }

    for t in &gguf.tensors {
        let name = &t.name;
        if name.contains(".nextn.") {
            shape.has_nextn = true;
            if shape.first_nextn.is_empty() {
                shape.first_nextn = name.clone();
            }
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    let num = &rest[..dot];
                    let ok = !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit());
                    let after = &rest[dot + 1..];
                    if ok && after.starts_with("nextn.") {
                        if let Ok(idx) = num.parse::<u32>() {
                            shape.nextn_idx.insert(idx);
                        }
                    }
                }
            }
        } else if name.starts_with("blk.0.") {
            shape.has_trunk = true;
        }
    }

    shape
}

/// `struct nextn_collector` (imatrix.cpp:1237-1252) — drives the model's
/// MTP/NextN layer over the trunk's hidden states so its matrices are
/// collected too.
struct NextnCollector {
    ctx: DecodeContext,
    batch: LlamaBatch,
    n_embd: usize,
    #[allow(dead_code)]
    own_lm_head: bool,
    pending_h: Vec<f32>,
}

impl NextnCollector {
    /// `nextn_collector_init` (imatrix.cpp:1411-1425): an
    /// `LLAMA_CONTEXT_TYPE_MTP` context (`n_rs_seq = 0`, common.cpp:1216)
    /// over the model — the draft file for `-md`, the trunk's own nextn
    /// block for `--nextn`. `draft_side` switches the `weights` bundle to a
    /// sizing-only stub: an MTP-only draft has no trunk tensors, and
    /// `DecodeContext::new_mtp` never dispatches `weights` (the MTP branch
    /// of forward returns first — the `new_eagle3` weights_stub precedent,
    /// context.rs:3158-3166); it only reads `output` (the logits buffer's
    /// n_vocab) and `n_layer()` (the layer-input tap count, unused on an MTP
    /// context).
    fn init(
        model: &mut llama::model::LlamaModel,
        n_threads: usize,
        n_ubatch: usize,
        n_ctx: u32,
        flash_attn: bool,
        draft_side: bool,
    ) -> Result<Self, String> {
        let facts = mtp_head_facts(&model.hparams);
        let attn = build_attn(&model.hparams, flash_attn);
        let mtp = llama::context::MtpForward::Qwen35(
            qwen35_mtp_weights(model),
            qwen35_params_of(model, attn),
            facts,
        );
        let own_lm_head = mtp_head_own_lm_head(model);
        let weights = if draft_side {
            ForwardWeights::Qwen35(
                llama::graph_arch::Qwen35ModelWeights {
                    tok_embd: model.tok_embd,
                    output_norm: model.output_norm,
                    output: model.output,
                    cls_out: None,
                    cls_out_b: None,
                    layers: Vec::new(),
                },
                qwen35_params_of(model, attn),
            )
        } else {
            build_weights(model, attn)
        };
        let n_embd = model.hparams.n_embd_out() as usize;
        let gctx = std::mem::replace(&mut model.ctx, ggml::Context::new());
        let ctx = DecodeContext::new_mtp(
            gctx,
            weights,
            mtp,
            attn,
            n_ctx,
            n_threads,
            n_ubatch,
        );
        Ok(NextnCollector {
            ctx,
            batch: LlamaBatch::default(),
            n_embd,
            own_lm_head,
            pending_h: Vec::new(),
        })
    }

    /// `clear_memory` (imatrix.cpp:1246-1249)
    fn clear_memory(&mut self) {
        self.ctx.reset_sequence();
        // a chunk's last h-row has no next token, the trunk restarts at
        // position 0
        self.pending_h.clear();
    }

    /// `bool nextn_collector::decode(...)` (imatrix.cpp:1427-1461): pair the
    /// trunk's nextn h-rows with the next batch token and decode the layer.
    fn decode(&mut self, ctx_trunk: &DecodeContext, batch_trunk: &LlamaBatch) -> bool {
        let n_last = batch_trunk.token.len() as i32 - 1;
        let n_embd = self.n_embd;
        self.batch.clear();

        let pos = batch_trunk.pos.as_ref().expect("trunk batch positions");
        if !self.pending_h.is_empty() {
            let idx = self.batch.token.len();
            self.batch.add(batch_trunk.token[0], pos[0], &[0], self.own_lm_head);
            self.batch
                .embd
                .get_or_insert_with(Vec::new)
                .resize((idx + 1) * n_embd, 0.0);
            self.batch.embd.as_mut().unwrap()[idx * n_embd..(idx + 1) * n_embd]
                .copy_from_slice(&self.pending_h);
        }

        let mut h_last: Option<Vec<f32>> = None;
        let mut i = 0i32;
        while i <= n_last {
            let h = ctx_trunk.get_embeddings_nextn_ith(i).to_vec();
            if h.is_empty() {
                eprintln!("nextn_collector::decode: no NextN hidden state for row {i}");
                return false;
            }
            if i == n_last {
                h_last = Some(h);
                break;
            }
            let idx = self.batch.token.len();
            self.batch.add(
                batch_trunk.token[(i + 1) as usize],
                pos[(i + 1) as usize],
                &[0],
                self.own_lm_head,
            );
            self.batch
                .embd
                .get_or_insert_with(Vec::new)
                .resize((idx + 1) * n_embd, 0.0);
            self.batch.embd.as_mut().unwrap()[idx * n_embd..(idx + 1) * n_embd]
                .copy_from_slice(&h);
            i += 1;
        }

        if !self.batch.token.is_empty() {
            if let Err(e) = self.ctx.decode_batch(&self.batch) {
                eprintln!("nextn_collector::decode: failed to decode the NextN layer: {e}");
                return false;
            }
        }

        if let Some(h) = h_last {
            self.pending_h = h;
        }

        true
    }
}

/// the qwen35 trunk params bundle the MTP graph reads (the CLI's
/// `qwen35_params`, llama-cli/src/main.rs:5460-5484)
fn qwen35_params_of(
    m: &llama::model::LlamaModel,
    attn: llama::graph::AttnParams,
) -> llama::graph_arch::Qwen35Params {
    let hp = &m.hparams;
    let n_layer = hp.n_layer() as usize;
    llama::graph_arch::Qwen35Params {
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

/// `own_lm_head`: the file has `blk.<n>.nextn.shared_head_head.weight`
/// (imatrix.cpp:1283-1306 `nextn_read_model_info`)
fn mtp_head_own_lm_head(m: &llama::model::LlamaModel) -> bool {
    let il = m.hparams.n_layer() as usize;
    m.layers.get(il)
        .map(|l| l.nextn.shared_head_head.is_some())
        .unwrap_or(false)
}

fn compute_imatrix(
    dctx: &mut DecodeContext,
    vocab: &Vocab,
    params: &Args,
    n_ctx: usize,
    nextn: &mut Option<NextnCollector>,
) -> bool {
    let add_bos = vocab.get_add_bos();

    if vocab.get_add_eos() {
        eprintln!("compute_imatrix: warning: model sets add_eos; the reference asserts !add_eos here");
    }

    eprintln!("compute_imatrix: tokenizing the input ..");
    let t0 = std::time::Instant::now();
    let prompt = params
        .file
        .as_ref()
        .and_then(|f| std::fs::read_to_string(f).ok())
        .or_else(|| params.prompt.clone())
        .unwrap_or_default();
    // `common_tokenize(ctx, params.prompt, true, params.parse_special)`
    // (imatrix.cpp:804)
    let mut tokens: Vec<i32> = vocab.tokenize(&prompt, true, params.parse_special);
    eprintln!(
        "compute_imatrix: tokenization took {} ms",
        t0.elapsed().as_micros() as f64 * 1e-3
    );

    // imatrix.cpp:809-816 — `--chunk N` drops the first N chunks
    if params.i_chunk > 0 {
        if (params.i_chunk as usize + 2) * n_ctx >= tokens.len() {
            eprintln!(
                "compute_imatrix: there will be not enough tokens left after removing {} chunks",
                params.i_chunk
            );
            return false;
        }
        eprintln!(
            "compute_imatrix: removing initial {} chunks ({} tokens)",
            params.i_chunk,
            params.i_chunk as usize * n_ctx
        );
        tokens.drain(..params.i_chunk as usize * n_ctx);
    }

    if tokens.len() < 2 * n_ctx {
        eprintln!(
            "compute_imatrix: you need at least {} tokens for a context of {} tokens",
            2 * n_ctx,
            n_ctx
        );
        eprintln!(
            "compute_imatrix: the data file you provided tokenizes to only {} tokens",
            tokens.len()
        );
        return false;
    }

    let n_chunk_max = tokens.len() / n_ctx;
    let n_chunk = if params.n_chunks < 0 {
        n_chunk_max
    } else {
        (params.n_chunks as usize).min(n_chunk_max)
    };
    let n_vocab = dctx.n_vocab();
    let n_batch = params.n_batch.max(1) as usize;

    let mut count = 0usize;
    let mut nll = 0.0f64;
    let mut nll2 = 0.0f64;

    let num_batches = (n_ctx + n_batch - 1) / n_batch;
    let n_seq = std::cmp::max(1, n_batch / n_ctx);

    // GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0)
    assert!(n_batch < n_ctx || n_batch % n_ctx == 0);
    // GGML_ASSERT(params.n_ctx == n_seq * n_ctx)
    assert_eq!(params.n_ctx as usize, n_seq * n_ctx);

    eprintln!(
        "compute_imatrix: computing over {n_chunk} chunks, n_ctx={n_ctx}, batch_size={n_batch}, n_seq={n_seq}"
    );

    let mut batch = LlamaBatch::default();
    let mut logits_hist: Vec<f32> = Vec::new();
    let mut prob_hist: Vec<f32> = Vec::new();
    if params.compute_ppl {
        logits_hist.resize(tokens.len(), 0.0);
        prob_hist.resize(tokens.len(), 0.0);
    }

    let mut i = 0usize;
    while i < n_chunk {
        let start = i * n_ctx;
        let end = start + n_ctx;

        let n_seq_batch = std::cmp::min(n_seq, n_chunk - i);

        let t_start = std::time::Instant::now();

        // clear the KV cache (imatrix.cpp:868)
        dctx.reset_sequence();
        if let Some(nc) = nextn.as_mut() {
            nc.clear_memory();
        }

        // `std::vector<float> logits` (imatrix.cpp:850-853): with num_batches
        // > 1 the per-batch logits of the whole chunk are concatenated; that
        // configuration implies n_seq == 1 (n_batch < n_ctx), so the rows are
        // the chunk's tokens in order.
        let mut chunk_logits: Vec<f32> = Vec::new();
        let mut last_out: Option<llama::context::BatchOutput> = None;

        for j in 0..num_batches {
            let batch_start = start + j * n_batch;
            let batch_size = std::cmp::min(end - batch_start, n_batch);

            // common_batch_clear + common_batch_add (imatrix.cpp:877-897)
            batch.clear();
            for seq in 0..n_seq_batch {
                let seq_start = batch_start + seq * n_ctx;

                // save original token and restore it after eval
                let token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if add_bos && j == 0 {
                    tokens[seq_start] = vocab.token_bos();
                }
                for k in 0..batch_size {
                    // NOTE: specifying all logits to get activations for the
                    // output.weight tensor and also for the perplexity
                    // calculation (imatrix.cpp:888-892)
                    batch.add(tokens[seq_start + k], (j * n_batch + k) as i32, &[seq as i32], true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            match dctx.decode_batch(&batch) {
                Ok(out) => {
                    if params.compute_ppl && num_batches > 1 {
                        chunk_logits.extend_from_slice(&out.logits);
                    }
                    last_out = Some(out);
                }
                Err(e) => {
                    eprintln!("compute_imatrix : failed to eval: {e}");
                    return false;
                }
            }

            // `if (nextn && !nextn->decode(ctx, batch))` (imatrix.cpp:1533-1537)
            if let Some(nc) = nextn.as_mut() {
                if !nc.decode(dctx, &batch) {
                    return false;
                }
            }
        }

        if i == 0 {
            let t_total = t_start.elapsed().as_secs_f64();
            // `LOG_INF("%s: %.2f seconds per pass - ETA ", ...)` (imatrix.cpp:916)
            // has no trailing newline; the minutes figure is a `LOG()` on stdout
            eprint!("compute_imatrix: {t_total:.2} seconds per pass - ETA ");
            let mut total_seconds = (t_total * n_chunk as f64 / n_seq as f64) as i64;
            if total_seconds >= 60 * 60 {
                print!("{} hours ", total_seconds / (60 * 60));
                total_seconds %= 60 * 60;
            }
            print!("{:.2} minutes\n", total_seconds as f64 / 60.0);
        }

        if params.compute_ppl {
            let out = last_out.as_ref().expect("decode output");
            let first = n_ctx / 2;
            for seq in 0..n_seq_batch {
                // `llama_get_logits_ith(ctx, seq*n_ctx)` with num_batches > 1,
                // else the concatenated per-batch buffer (imatrix.cpp:928)
                let all_logits: &[f32] = if num_batches > 1 {
                    &chunk_logits
                } else {
                    &out.logits
                };
                let row_of = |token_in_batch: usize| -> &[f32] {
                    if num_batches > 1 {
                        &all_logits[token_in_batch * n_vocab..(token_in_batch + 1) * n_vocab]
                    } else {
                        let row = out.output_ids[token_in_batch];
                        assert!(row >= 0, "all tokens request logits");
                        let row = row as usize;
                        &all_logits[row * n_vocab..(row + 1) * n_vocab]
                    }
                };

                let first_tok = start + seq * n_ctx + first;
                let n_scored = n_ctx - 1 - first;
                for t in 0..n_scored {
                    // rows `first..n_ctx` of this sequence, predicting
                    // `tokens[first + t + 1]` (imatrix.cpp:930-936)
                    let row = row_of(seq * n_ctx + first + t);
                    let tok = tokens[first_tok + t + 1] as usize;
                    let (v, logit, prob) = nll_of_row(row, tok);
                    nll += v;
                    nll2 += v * v;
                    logits_hist[first_tok + t] = logit;
                    prob_hist[first_tok + t] = prob;
                }
                count += n_scored;

                print!("[{}]{:.4},", i + seq + 1, (nll / count as f64).exp());
            }
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
        }

        i += n_seq_batch;
    }

    println!();

    if params.compute_ppl {
        nll2 /= count as f64;
        nll /= count as f64;
        let ppl = nll.exp();
        nll2 -= nll * nll;
        if nll2 > 0.0 {
            nll2 = (nll2 / (count as f64 - 1.0)).sqrt();
            println!("Final estimate: PPL = {ppl:.4} +/- {:.5}", nll2 * ppl);
        } else {
            println!("Unexpected negative standard deviation of log(prob)");
        }
    }

    true
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    // `const bool use_draft = !params.speculative.draft.mparams.path.empty()`
    // (imatrix.cpp:1835-1840, upstream a7b94df2c)
    let use_draft = args.model_draft.is_some();
    // the load_mtp flag: `--nextn` or `-md`
    let load_mtp = args.nextn || use_draft;

    // `common_params_parse` + `g_collector.set_params(params)` (imatrix.cpp:1089-1094)
    llama::imatrix::set_verbosity(args.verbose);
    let imatrix_params = ImatrixParams {
        n_ctx: args.n_ctx,
        n_parallel: 1,
        n_out_freq: args.n_out_freq,
        n_save_freq: args.n_save_freq,
        process_output: args.process_output,
        out_file: args.out_file.clone(),
        prompt_file: args.file.clone().unwrap_or_default(),
        imat_dat: args.imat_dat,
        // `params.speculative.draft.n_max = 0` (imatrix.cpp:1838) — keep the
        // trunk's context identical to a plain run; the flag itself carries
        // the `-md` effect (the C sets `params.load_mtp = true`)
        load_mtp,
        chunk_size_loaded: 0,
        n_layer_nextn: 0,
    };
    with_collector(|c| c.set_params(imatrix_params.clone()));

    // ---- --show-statistics (imatrix.cpp:1096-1101) ----
    if args.show_statistics {
        // `if (params.in_files.empty()) { return false; }` — a single file is
        // required (imatrix.cpp:1610-1613, upstream a7b94df2c drops the
        // upfront >1 check; the load still takes only the first)
        if args.in_files.is_empty() {
            eprintln!();
            eprintln!("Error: a single imatrix file is required to compute tensor statistics");
            eprintln!();
            return ExitCode::from(1);
        }
        let loaded = with_collector(|c| {
            load_imatrix_into(
                &mut c.m_stats,
                &mut c.m_datasets,
                &mut c.m_last_chunk,
                &mut c.m_params,
                &args.in_files[0],
            )
        });
        if !loaded {
            eprintln!();
            eprintln!("Error: {} is not a valid imatrix file", args.in_files[0]);
            eprintln!();
            return ExitCode::from(1);
        }
        let ok = with_collector(|c| show_statistics(&args.in_files[0], c));
        return if ok { ExitCode::from(0) } else { ExitCode::from(1) };
    }

    // `const int32_t n_ctx = params.n_ctx` (imatrix.cpp:1103-1108)
    if args.n_ctx <= 0 {
        eprintln!("main: imatrix tool requires '--ctx-size' > 0");
        return ExitCode::from(1);
    }
    let n_ctx = args.n_ctx as usize;

    // ---- n_seq rewrite (imatrix.cpp:1110-1120) ----
    let n_seq = std::cmp::max(1, args.n_batch / args.n_ctx);
    let n_kv = n_seq * args.n_ctx;

    // `--nextn`/`--model-draft` need a single sequence per batch
    // (imatrix.cpp:1862-1866, upstream a7b94df2c)
    if load_mtp && n_seq > 1 {
        eprintln!(
            "main: '--nextn' and '--model-draft' need a single sequence per batch, set '--batch-size' to at most '--ctx-size' ({})",
            args.n_ctx
        );
        return ExitCode::from(1);
    }

    let rebuilt = ImatrixParams {
        n_ctx: n_kv,
        n_parallel: n_seq,
        ..imatrix_params.clone()
    };
    let n_batch = std::cmp::min(args.n_batch, n_kv);
    with_collector(|c| c.set_params(rebuilt));

    // ---- --in-file (imatrix.cpp:1122-1128) ----
    for in_file in &args.in_files {
        eprintln!("main : loading imatrix from '{in_file}'");
        let ok = with_collector(|c| {
            load_imatrix_into(
                &mut c.m_stats,
                &mut c.m_datasets,
                &mut c.m_last_chunk,
                &mut c.m_params,
                in_file,
            )
        });
        if !ok {
            eprintln!("main : failed to load {in_file}");
            return ExitCode::from(1);
        }
    }

    // ---- no prompt: only combine the loaded matrices (imatrix.cpp:1130-1147) ----
    if args.file.is_none() && args.prompt.is_none() {
        eprintln!("No prompt provided; combining precomputed matrices only.");
        if args.in_files.is_empty() {
            eprintln!("Error: No prompt provided and no precomputed matrices (--in-file) to combine.");
            return ExitCode::from(1);
        }
        if args.in_files.len() == 1 {
            eprintln!("main : saving imatrix to '{}'", args.out_file);
        } else {
            eprintln!("main : saving combined imatrix to '{}'", args.out_file);
        }
        let res = with_collector(|c| {
            let params = c.m_params.clone();
            let datasets = c.m_datasets.clone();
            let last = c.m_last_chunk;
            save_imatrix(&c.m_stats, &params, &datasets, last, -1)
        });
        if let Err(e) = res {
            eprintln!("save_imatrix: {e}");
            return ExitCode::from(1);
        }
        return ExitCode::from(0);
    }

    // ---- `--nextn`/`-md` shape validation (imatrix.cpp:1903-1958, upstream
    // a7b94df2c) ----
    let Some(model_path) = args.model.clone() else {
        eprintln!("error: no model specified (-m/--model)");
        return ExitCode::from(1);
    };
    {
        let shape = model_read_file_shape(&model_path);
        if use_draft && shape.n_nextn_layers > 0 {
            eprintln!("main: model already includes NextN layers");
            return ExitCode::from(1);
        }
        if use_draft && shape.has_nextn {
            eprintln!(
                "main: the model already includes NextN layers (NextN tensor '{}')",
                shape.first_nextn
            );
            return ExitCode::from(1);
        }
        if !use_draft && !shape.has_trunk && (shape.n_nextn_layers > 0 || shape.has_nextn) {
            eprintln!("main: '{model_path}' is a NextN draft, use with --model-draft/-md");
            return ExitCode::from(1);
        }
        if use_draft {
            let path_md = args.model_draft.clone().unwrap();
            let shape_md = model_read_file_shape(&path_md);
            if !shape_md.valid || shape_md.is_split {
                eprintln!("main: '{path_md}' is not a readable single-shard GGUF");
                return ExitCode::from(1);
            }
            if shape.valid && shape_md.arch != shape.arch {
                eprintln!(
                    "main: the draft architecture '{}' does not match the target architecture '{}'",
                    shape_md.arch, shape.arch
                );
                return ExitCode::from(1);
            }
            if shape_md.n_nextn_layers == 0 || shape_md.has_trunk {
                eprintln!(
                    "main: '{path_md}' is not a draft (no NextN layers or trunk tensors present)"
                );
                return ExitCode::from(1);
            }
            if shape_md.n_nextn_layers > 1 {
                eprintln!(
                    "main: multi-layer NextN drafts are not supported (found {})",
                    shape_md.n_nextn_layers
                );
                return ExitCode::from(1);
            }
            if shape_md.n_trunk() == 0 {
                eprintln!("main: NextN layers sharing the trunk's KV cache are not supported");
                return ExitCode::from(1);
            }
            if shape.valid && !shape_md.nextn_idx.contains(&shape.n_trunk()) {
                eprintln!(
                    "main: no NextN tensors at layer index {} in '{path_md}' found",
                    shape.n_trunk()
                );
                return ExitCode::from(1);
            }
            if shape.valid && shape_md.n_embd_out != shape.n_embd_out {
                eprintln!(
                    "main: the draft output width {} does not match the target's {}",
                    shape_md.n_embd_out, shape.n_embd_out
                );
                return ExitCode::from(1);
            }
            // the imatrix driver only feeds one spec type through `-md`
            // (`draft-mtp`); the port's args have no `--spec-type` surface so
            // the check collapses to the constant
        }
    }

    // ---- model + context (imatrix.cpp:1149-1183) ----
    // `params.cb_eval = ik_collect_imatrix` (imatrix.cpp:1154): the hook lives
    // at the ggml layer, so it is registered before the first graph runs.
    ggml::compute::set_eval_callback(Some(llama::imatrix::eval_callback));

    let Some(model_path) = args.model.clone() else {
        eprintln!("error: no model specified (-m/--model)");
        return ExitCode::from(1);
    };
    let gguf = match Gguf::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("llama-imatrix: unable to load model '{model_path}': {e}");
            return ExitCode::from(1);
        }
    };
    let mmap: Arc<memmap2::Mmap> = {
        let Ok(f) = std::fs::File::open(&model_path) else {
            eprintln!("llama-imatrix: unable to open '{model_path}'");
            return ExitCode::from(1);
        };
        // SAFETY: read-only usage of a model file
        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
    };
    let vocab = match Vocab::load(&gguf) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("llama-imatrix: failed to load vocabulary: {e}");
            return ExitCode::from(1);
        }
    };
    let model = match load_model(&gguf, mmap) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("llama-imatrix: failed to load model: {e}");
            return ExitCode::from(1);
        }
    };

    let n_ctx_train = model.hparams.n_ctx_train as i32;
    if args.n_ctx > n_ctx_train {
        eprintln!(
            "main: model was trained on only {n_ctx_train} context tokens ({} specified)",
            args.n_ctx
        );
    }

    let n_threads = if args.n_threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    } else {
        args.n_threads
    };

    let attn = build_attn(&model.hparams, args.flash_attn.on());
    // cached before `model.ctx` moves into the trunk DecodeContext (the
    // `--nextn` arm's own_lm_head probe, imatrix.cpp:2020)
    let trunk_own_lm_head = mtp_head_own_lm_head(&model);
    let weights = build_weights(&model, attn);
    let mut dctx = DecodeContext::new_with(
        model.ctx,
        weights,
        attn,
        n_kv as u32,
        n_threads,
        n_batch as usize,
    );

    // ---- the NextN collector (imatrix.cpp:1983-2038, upstream a7b94df2c) ----
    let mut nextn: Option<NextnCollector> = None;
    if load_mtp {
        // `model_src = use_draft ? model_draft : model` (imatrix.cpp:1998):
        // the `-md` arm loads the draft file with `load_mtp = true`
        // (imatrix.cpp:1969-1972, `mparams_sidecar.load_mtp = true`) and
        // constructs the collector's MTP context over *it*; the `--nextn`
        // arm reuses the trunk's own nextn block.
        let mut model_src: Option<llama::model::LlamaModel> = None;
        let (n_heads, n_trunk_src, own_lm_head) = if use_draft {
            let path_md = args.model_draft.clone().unwrap();
            let gguf_md = match Gguf::open(&path_md) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("main: unable to load draft '{path_md}': {e}");
                    return ExitCode::from(1);
                }
            };
            let mmap_md: Arc<memmap2::Mmap> = {
                let Ok(f) = std::fs::File::open(&path_md) else {
                    eprintln!("main: failed to open '{path_md}'");
                    return ExitCode::from(1);
                };
                // SAFETY: read-only usage of a model file
                Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
            };
            let mut md = match load_model(&gguf_md, mmap_md) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("main: failed to load the draft model '{path_md}': {e}");
                    return ExitCode::from(1);
                }
            };
            // `common_speculative_are_compatible(model, model_draft)`
            // (imatrix.cpp:1982-1987): the trunk/draft vocabularies must
            // line up — the port checks the token counts and the type
            {
                let v_md = match Vocab::load(&gguf_md) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("main: failed to load the draft vocabulary: {e}");
                        return ExitCode::from(1);
                    }
                };
                if v_md.n_tokens() != vocab.n_tokens() {
                    eprintln!("main: the target and draft vocab are not compatible");
                    return ExitCode::from(1);
                }
            }
            // `nextn_read_model_info(path_md, n_trunk, n_heads)`
            // (imatrix.cpp:1256-1280): own_lm_head = the file carries
            // `blk.<n_trunk>.nextn.shared_head_head.weight`
            let n_heads = md.hparams.n_layer_nextn;
            let n_trunk = md.hparams.n_layer();
            let own = md
                .layers
                .get(n_trunk as usize)
                .map(|l| l.nextn.shared_head_head.is_some())
                .unwrap_or(false);
            let r = (n_heads, n_trunk, own);
            model_src = Some(md);
            r
        } else {
            let n_heads = model.hparams.n_layer_nextn;
            let n_trunk = model.hparams.n_layer();
            (n_heads, n_trunk, trunk_own_lm_head)
        };
        let n_trunk = model.hparams.n_layer();

        if n_heads == 0 {
            eprintln!(
                "main: the model has no NextN layers, '{}' has no effect",
                if use_draft { "-md" } else { "--nextn" }
            );
        } else if n_trunk_src == 0 {
            eprintln!("main: NextN layers sharing the trunk's KV cache are not supported");
            return ExitCode::from(1);
        } else if n_heads > 1 {
            eprintln!("main: multi-layer NextN drafts are not supported (found {n_heads})");
            return ExitCode::from(1);
        } else if !(if use_draft {
            // `!info.has_layers` (imatrix.cpp:2030-2031) — probed on the
            // *source* file, i.e. the draft when `-md`
            model_src
                .as_ref()
                .unwrap()
                .layers
                .get(n_trunk_src as usize)
                .map(|l| l.nextn.eh_proj.is_some())
                .unwrap_or(false)
        } else {
            model
                .layers
                .get(n_trunk as usize)
                .map(|l| l.nextn.eh_proj.is_some())
                .unwrap_or(false)
        }) {
            let path_src = if use_draft {
                args.model_draft.clone().unwrap()
            } else {
                model_path.clone()
            };
            eprintln!(
                "main: no NextN tensor in '{path_src}', '{}' has no effect",
                if use_draft { "-md" } else { "--nextn" }
            );
        } else if !matches!(model.arch, LlmArch::QWEN35) {
            eprintln!(
                "main: '--nextn' collection for arch {} is not ported in the imatrix tool yet \
                 (the qwen35 family is); see PARITY.md",
                model.arch.name()
            );
            return ExitCode::from(1);
        } else {
            // `nextn_collector_init(model_src, ...)` (imatrix.cpp:1411-1425):
            // the MTP context over the source model — the draft for `-md`,
            // a fresh reload of the trunk file for `--nextn` (the port
            // cannot host two DecodeContexts over one ggml Context, and the
            // trunk ctx took `model.ctx`)
            let mut model_mtp = match model_src.take() {
                Some(md) => md,
                None => {
                    let gguf2 = match Gguf::open(&model_path) {
                        Ok(g) => g,
                        Err(e) => {
                            eprintln!("main: unable to reload model '{model_path}': {e}");
                            return ExitCode::from(1);
                        }
                    };
                    let mmap2: Arc<memmap2::Mmap> = {
                        let Ok(f) = std::fs::File::open(&model_path) else {
                            eprintln!("main: failed to open '{model_path}'");
                            return ExitCode::from(1);
                        };
                        Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() })
                    };
                    match load_model(&gguf2, mmap2) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("main: failed to reload model: {e}");
                            return ExitCode::from(1);
                        }
                    }
                }
            };
            match NextnCollector::init(
                &mut model_mtp,
                n_threads,
                n_batch as usize,
                n_kv as u32,
                args.flash_attn.on(),
                use_draft,
            ) {
                Ok(nc) => nextn = Some(nc),
                Err(e) => {
                    eprintln!("main: {e}");
                    return ExitCode::from(1);
                }
            }
            // `llama_set_embeddings_nextn(ctx, true, /*masked*/ false)`
            // (imatrix.cpp:2034) — the trunk's tap feeds the collector
            // (panics on an arch whose nextn tap is not ported, like the C
            // asserts)
            dctx.set_embeddings_nextn(true, false);
            // `g_collector.set_n_layer_nextn(n_heads)` (imatrix.cpp:2035)
            with_collector(|c| c.m_params.n_layer_nextn = n_heads as i32);

            eprintln!("main: processing {n_heads} NextN layer(s) from block {n_trunk}");
        }
    }

    if !compute_imatrix(&mut dctx, &vocab, &args, n_ctx, &mut nextn) {
        return ExitCode::from(1);
    }

    // `g_collector.save_imatrix()` (imatrix.cpp:1185)
    let res = with_collector(|c| {
        let params = c.m_params.clone();
        let datasets = c.m_datasets.clone();
        let last = c.m_last_chunk;
        save_imatrix(&c.m_stats, &params, &datasets, last, -1)
    });
    if let Err(e) = res {
        eprintln!("save_imatrix: {e}");
        return ExitCode::from(1);
    }

    println!();
    ExitCode::from(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_parsing() {
        let a = parse_args(&[
            "llama-imatrix".into(),
            "-m".into(),
            "m.gguf".into(),
            "-f".into(),
            "t.txt".into(),
            "-o".into(),
            "out.gguf".into(),
            "--chunk".into(),
            "3".into(),
            "--output-frequency".into(),
            "7".into(),
            "--save-frequency".into(),
            "1".into(),
            "--process-output".into(),
            "--no-ppl".into(),
            "--output-format".into(),
            "dat".into(),
            "-c".into(),
            "64".into(),
            "-b".into(),
            "64".into(),
            "-t".into(),
            "1".into(),
        ])
        .unwrap();
        assert_eq!(a.model.as_deref(), Some("m.gguf"));
        assert_eq!(a.file.as_deref(), Some("t.txt"));
        assert_eq!(a.out_file, "out.gguf");
        assert_eq!(a.i_chunk, 3);
        assert_eq!(a.n_out_freq, 7);
        assert_eq!(a.n_save_freq, 1);
        assert!(a.process_output);
        assert!(!a.compute_ppl);
        assert_eq!(a.imat_dat, 1);
        assert_eq!(a.n_ctx, 64);
        assert!(parse_args(&["p".into(), "--nope".into()]).is_err());
    }

    /// `(layer, tensor)` split and the `Σ(Act²)` ordering key of the
    /// statistics table (imatrix.cpp:989-997).
    #[test]
    fn tensor_name_split() {
        // (imatrix.cpp:96-145, upstream a7b94df2c): the `j += size > 4 ? 1 : 2`
        // quirk skips the role part for a 4-part `blk.N.role.weight`, so the
        // tensor falls back to the full input; longer names keep the
        // dotted role path (e.g. `nextn.blk.28.post_projection.weight`)
        assert_eq!(
            llama::imatrix::process_tensor_name("blk.5.ffn_up.weight"),
            ("5".to_string(), "blk.5.ffn_up.weight".to_string())
        );
        assert_eq!(
            llama::imatrix::process_tensor_name("output.weight"),
            ("-".to_string(), "output".to_string())
        );
    }

    /// `log_softmax` (imatrix.cpp:745-755): the returned NLL matches the
    /// f64/f32 mix of the C (`expf` in f32, `log` in f64).
    #[test]
    fn nll_matches_reference_formula() {
        let logits = [1.0f32, 2.0, 3.0];
        let max = 3.0f32;
        let sum: f64 = logits.iter().map(|v| ((*v) - max).exp() as f64).sum();
        for tok in 0..3 {
            let want = -((logits[tok] - max) as f64 - sum.ln());
            let (got, _, _) = nll_of_row(&logits, tok);
            assert!((got - want).abs() < 1e-12);
        }
    }
}