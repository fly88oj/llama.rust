//! diffusion.rs — port of `examples/diffusion/{diffusion.cpp,diffusion-cli.cpp}`
//! (pinned bd4f514db1, 408 + 266 lines): the masked-iterative decode loop for
//! the diffusion archs (llada / llada-moe / dream / rnd1 — graphs already in
//! `graph_arch.rs`, `llm_arch_is_diffusion` gates the dispatch in main.rs).
//!
//! The driver lives in llama-cli behind the `--diffusion-*` flags exactly
//! like the reference's `llama-diffusion-cli` example binary.
//!
//! Mapping (diffusion.cpp → Rust):
//!   `calculate_confidence`  (:13-46)  -> [`calculate_confidence`]
//!   `calculate_transfer_count` (:49-73) -> [`calculate_transfer_count`]
//!   `add_gumbel_noise`      (:75-88)  -> [`add_gumbel_noise`]
//!   `get_num_transfer_tokens` (:90-101) -> [`get_num_transfer_tokens`]
//!   `diffusion_generate`    (:103-408) -> [`generate`]
//! and diffusion-cli.cpp → [`run`] (arg plumbing, `diffusion_step_callback`,
//! the final detokenize).

use llama::context::DecodeContext;
use llama::sampling::{
    init_dist, init_temp, init_top_k, init_top_p, Mt19937, SamplerChain, TokenData,
    TokenDataArray,
};
use llama::vocab::{Token, Vocab};

use crate::Args;

/// `diffusion_algorithm` (diffusion.h:7-13)
pub const DIFFUSION_ALGORITHM_ORIGIN: i32 = 0;
pub const DIFFUSION_ALGORITHM_ENTROPY_BASED: i32 = 1;
pub const DIFFUSION_ALGORITHM_MARGIN_BASED: i32 = 2;
pub const DIFFUSION_ALGORITHM_RANDOM: i32 = 3;
pub const DIFFUSION_ALGORITHM_CONFIDENCE_BASED: i32 = 4;

/// `diffusion_transfer_schedule` (diffusion.h:16-19)
pub const DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED: i32 = 0;
pub const DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED: i32 = 1;

/// `diffusion_params` (diffusion.h:27-50)
pub struct DiffusionParams {
    pub steps: i32,
    pub temperature: f32,
    pub mask_token_id: Token,
    pub seed: u32,
    pub shift_logits: bool,
    pub top_p: f32,
    pub top_k: i32,
    pub algorithm: i32,
    pub schedule: i32,
    pub cfg_scale: f32,
    pub eps: f32,
    pub block_length: i32,
    pub alg_temp: f32,
    pub add_gumbel_noise: bool,
    pub max_length: i32,
}

/// `calculate_confidence` (diffusion.cpp:13-46)
fn calculate_confidence(cur_p: &TokenDataArray, algorithm: i32, rng: &mut Mt19937) -> f32 {
    match algorithm {
        DIFFUSION_ALGORITHM_CONFIDENCE_BASED => cur_p.data[cur_p.selected as usize].p,

        DIFFUSION_ALGORITHM_ENTROPY_BASED => {
            let mut entropy = 0.0f32;
            let epsilon = 1e-10f32;
            for i in 0..cur_p.size {
                let prob = cur_p.data[i].p;
                entropy += prob * (prob + epsilon).ln();
            }
            -entropy // Higher entropy = lower confidence
        }

        DIFFUSION_ALGORITHM_MARGIN_BASED => {
            if cur_p.size > 1 {
                cur_p.data[0].p - cur_p.data[1].p
            } else {
                cur_p.data[0].p
            }
        }

        DIFFUSION_ALGORITHM_RANDOM => rng.canonical_float(), // Random confidence

        DIFFUSION_ALGORITHM_ORIGIN => cur_p.data[cur_p.selected as usize].p,

        _ => 0.0,
    }
}

/// `calculate_transfer_count` (diffusion.cpp:49-73)
fn calculate_transfer_count(
    step: i32,
    total_steps: i32,
    remaining_masked: usize,
    schedule: i32,
    eps: f32,
    num_transfer_tokens: &[i32],
) -> i32 {
    match schedule {
        DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED => {
            let t = 1.0f32 - step as f32 / total_steps as f32 * (1.0 - eps);
            let s = 1.0f32 - (step + 1) as f32 / total_steps as f32 * (1.0 - eps);
            let p_transfer = if step < total_steps - 1 { 1.0 - s / t } else { 1.0 };
            (remaining_masked as f32 * p_transfer) as i32
        }

        DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED => {
            if !num_transfer_tokens.is_empty() && step < num_transfer_tokens.len() as i32 {
                num_transfer_tokens[step as usize]
            } else {
                remaining_masked as i32 / (total_steps - step)
            }
        }

        _ => remaining_masked as i32 / (total_steps - step),
    }
}

/// `add_gumbel_noise` (diffusion.cpp:75-88) — note the reference's unusual
/// form: `logits[i] = exp(logits[i]) / pow(-log(u), temperature)`
fn add_gumbel_noise(logits: &mut [f32], temperature: f32, rng: &mut Mt19937) {
    if temperature == 0.0 {
        return;
    }

    for v in logits.iter_mut() {
        let mut noise = rng.canonical_double();
        // Prevent log(0)
        noise = noise.max(1e-20);
        let gumbel_noise = (-noise.ln()).powf(temperature as f64);
        *v = v.exp() / gumbel_noise as f32;
    }
}

/// `get_num_transfer_tokens` (diffusion.cpp:90-101)
fn get_num_transfer_tokens(mask_count: i32, steps: i32) -> Vec<i32> {
    let mut num_transfer_tokens = vec![0i32; steps.max(0) as usize];

    let base = mask_count / steps;
    let remainder = mask_count % steps;

    for (i, slot) in num_transfer_tokens.iter_mut().enumerate() {
        *slot = base + if (i as i32) < remainder { 1 } else { 0 };
    }

    num_transfer_tokens
}

/// the step callback's progress bar (diffusion-cli.cpp:30-38) — `LOG_INF`
/// (stderr) with `\r`
fn print_progress_bar(step: i32, total_steps: i32) {
    let progress_percent = (step * 100) / total_steps;
    let progress_bars = (step * 50) / total_steps;
    eprint!(
        "\rdiffusion step: {}/{} [{}{}] {}%",
        step,
        total_steps,
        "=".repeat(progress_bars as usize),
        " ".repeat((50 - progress_bars) as usize),
        progress_percent
    );
}

/// `diffusion_generate` (diffusion.cpp:103-408). Returns `n_generated`
/// (0 = failure / nothing to do).
pub fn generate(
    dctx: &mut DecodeContext,
    vocab: &Vocab,
    input_tokens: &[Token],
    output_tokens: &mut [Token],
    params: &DiffusionParams,
    visual_mode: bool,
    n_input: usize,
) -> i32 {
    if input_tokens.is_empty() || params.max_length <= n_input as i32 {
        return 0;
    }

    let max_length = params.max_length as usize;

    // Initialize with input and pad with mask tokens (:116-118)
    output_tokens[..n_input].copy_from_slice(&input_tokens[..n_input]);
    for t in output_tokens[n_input..max_length].iter_mut() {
        *t = params.mask_token_id;
    }

    let mut rng = Mt19937::new(params.seed);

    // `llama_set_causal_attn(ctx, false)` (:122) — the diffusion graphs are
    // the no-cache non-causal builders; the port's flag rides along
    dctx.set_causal_attn(false);

    let n_vocab = dctx.n_vocab();

    // Setup sampler chain (:133-143)
    let mut sampler = SamplerChain::new();
    if params.top_k > 0 {
        sampler.add(init_top_k(params.top_k));
    }
    if params.top_p < 1.0 {
        sampler.add(init_top_p(params.top_p, 1));
    }
    if params.temperature > 0.0 {
        sampler.add(init_temp(params.temperature));
    }
    sampler.add(init_dist(params.seed));

    let mut dist_sampler = init_dist(params.seed);

    let mut cond_logits_buffer: Vec<f32> = Vec::new();
    let mut un_x_buffer: Vec<Token> = Vec::new();
    if params.cfg_scale > 0.0 {
        cond_logits_buffer = vec![0.0; n_vocab * max_length];
        un_x_buffer = vec![0; max_length];
    }

    // For block-based processing (:159-169)
    let mut num_transfer_tokens: Vec<i32> = Vec::new();
    let num_blocks;
    let mut steps_per_block = params.steps;

    if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED {
        assert!(
            params.max_length % params.block_length == 0,
            "max_length % block_length == 0 (diffusion.cpp:165)"
        );
        num_blocks = params.max_length / params.block_length;
        assert!(
            params.steps % num_blocks == 0,
            "steps % num_blocks == 0 (diffusion.cpp:167)"
        );
        steps_per_block = params.steps / num_blocks;
    } else {
        num_blocks = 1;
    }

    let mut candidates: Vec<TokenData> = Vec::with_capacity(n_vocab);

    let time_start = std::time::Instant::now();

    let mut total_sampling_us: u64 = 0;

    for block_num in 0..num_blocks {
        let block_start = if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED {
            n_input as i32 + block_num * params.block_length
        } else {
            0
        };
        let block_end = if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED {
            (n_input as i32 + (block_num + 1) * params.block_length).min(params.max_length)
        } else {
            params.max_length
        };

        // Count masked tokens in current block (:183-192)
        if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED {
            let mut block_mask_count = 0;
            for i in block_start..block_end {
                if output_tokens[i as usize] == params.mask_token_id {
                    block_mask_count += 1;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for step in 0..steps_per_block {
            let global_step = block_num * steps_per_block + step;

            // the step callback (:197-202) — always continue
            if visual_mode {
                eprint!("\x1b[2J\x1b[H"); // Clear screen and move cursor to top-left
                print_progress_bar(global_step, params.steps);
                eprintln!();
                let mut current_text = String::from(" ");
                for &tok in &output_tokens[n_input..max_length] {
                    let token_str = if tok != vocab.token_mask() {
                        String::from_utf8_lossy(&vocab.token_to_piece_special(tok, false))
                            .into_owned()
                    } else {
                        " ".to_string()
                    };
                    current_text.push_str(&token_str);
                }
                eprintln!("{current_text}");
            } else {
                print_progress_bar(global_step, params.steps);
            }

            // Setup batch (:204-211): the whole sequence, every position an
            // output row — the port's decode_all. The diffusion graphs are
            // the no-cache builders (they never read KV cells), so the cache
            // is cleared each step to keep find_slot from exhausting it —
            // the C reuses cells via its same-(seq,pos) slot matching
            dctx.kv.clear();
            let batch_tokens: Vec<Token> = output_tokens[..max_length].to_vec();
            let positions: Vec<i32> = (0..max_length as i32).collect();

            let mut logits: Vec<f32>;

            if params.cfg_scale > 0.0 {
                // conditional pass (:215-222)
                let cond = match dctx.decode_all(&batch_tokens, &positions) {
                    Ok(l) => l,
                    Err(_) => {
                        eprintln!("Failed to generate conditional");
                        break;
                    }
                };
                cond_logits_buffer.copy_from_slice(&cond);

                // Unconditional generation (mask input) (:224-237)
                un_x_buffer.copy_from_slice(&output_tokens[..max_length]);
                for t in un_x_buffer[..n_input].iter_mut() {
                    *t = params.mask_token_id;
                }
                let un_tokens: Vec<Token> = un_x_buffer.clone();
                let uncond = match dctx.decode_all(&un_tokens, &positions) {
                    Ok(l) => l,
                    Err(_) => {
                        eprintln!("Failed to generate unconditional");
                        break;
                    }
                };

                // Apply CFG (:240-245)
                for i in 0..cond_logits_buffer.len() {
                    cond_logits_buffer[i] = uncond[i]
                        + (params.cfg_scale + 1.0) * (cond_logits_buffer[i] - uncond[i]);
                }
                logits = std::mem::take(&mut cond_logits_buffer);
            } else {
                match dctx.decode_all(&batch_tokens, &positions) {
                    Ok(l) => logits = l,
                    Err(e) => {
                        eprintln!("generate: failed to decode at step {global_step}, ret = {e}");
                        break;
                    }
                }
            }

            // the shift_logits position mapping (:260-265)
            fn get_logits_for_pos(logits: &[f32], pos: usize, n_vocab: usize, shift: bool) -> &[f32] {
                if shift {
                    if pos == 0 {
                        &[]
                    } else {
                        let s = (pos - 1) * n_vocab;
                        &logits[s..s + n_vocab]
                    }
                } else {
                    let s = pos * n_vocab;
                    &logits[s..s + n_vocab]
                }
            }

            let time_start_sampling = std::time::Instant::now();

            let mut mask_positions: Vec<usize> = Vec::with_capacity(max_length);
            for i in 0..max_length {
                if output_tokens[i] == params.mask_token_id {
                    // For block-based, only consider current block (:272-275)
                    if params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED
                        || ((i as i32) >= block_start && (i as i32) < block_end)
                    {
                        mask_positions.push(i);
                    }
                }
            }

            if mask_positions.is_empty() {
                break;
            }

            if params.add_gumbel_noise && params.temperature > 0.0 {
                add_gumbel_noise(&mut logits, params.temperature, &mut rng);
            }

            if params.algorithm == DIFFUSION_ALGORITHM_ORIGIN {
                // (:287-311)
                let transfer_count = calculate_transfer_count(
                    step,
                    steps_per_block,
                    mask_positions.len(),
                    params.schedule,
                    params.eps,
                    &num_transfer_tokens,
                );
                let p_transfer = transfer_count as f32 / mask_positions.len() as f32;

                for &pos in &mask_positions {
                    if rng.canonical_float() < p_transfer {
                        let pos_logits: Vec<f32> =
                            get_logits_for_pos(&logits, pos, n_vocab, params.shift_logits).to_vec();
                        candidates.clear();
                        for (token_id, &l) in pos_logits.iter().enumerate() {
                            candidates.push(TokenData {
                                id: token_id as Token,
                                logit: l,
                                p: 0.0,
                            });
                        }

                        let mut cur_p = TokenDataArray {
                            data: std::mem::take(&mut candidates),
                            size: n_vocab,
                            selected: -1,
                            sorted: false,
                        };

                        sampler.apply(&mut cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected as usize].id;
                    }
                }
            } else {
                // (:312-388)
                let mut confidences: Vec<(f32, usize)> = Vec::with_capacity(mask_positions.len());
                let mut sampled_tokens: Vec<Token> = vec![0; mask_positions.len()];

                for (i, &pos) in mask_positions.iter().enumerate() {
                    let pos_logits: Vec<f32> = get_logits_for_pos(&logits, pos, n_vocab, params.shift_logits).to_vec();

                    candidates.clear();
                    for (token_id, &l) in pos_logits.iter().enumerate() {
                        candidates.push(TokenData {
                            id: token_id as Token,
                            logit: l,
                            p: 0.0,
                        });
                    }

                    let mut cur_p = TokenDataArray {
                        data: std::mem::take(&mut candidates),
                        size: n_vocab,
                        selected: -1,
                        sorted: false,
                    };

                    sampler.apply(&mut cur_p);
                    let sampled_token = cur_p.data[cur_p.selected as usize].id;

                    let conf = calculate_confidence(&cur_p, params.algorithm, &mut rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.push((conf, i));
                }

                let transfer_count = calculate_transfer_count(
                    step,
                    steps_per_block,
                    mask_positions.len(),
                    params.schedule,
                    params.eps,
                    &num_transfer_tokens,
                );

                if transfer_count > 0 {
                    if params.alg_temp == 0.0 {
                        // deterministic partial_sort by confidence (:346-361)
                        let take = (transfer_count as usize).min(confidences.len());
                        confidences
                            .sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

                        for &(_, mask_idx) in &confidences[..take] {
                            let pos = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        // sample transfer positions by confidence temperature
                        // (:363-387)
                        let take = (transfer_count as usize).min(confidences.len());
                        let mut conf_candidates: Vec<TokenData> = confidences
                            .iter()
                            .map(|&(conf, i)| TokenData {
                                id: i as Token,
                                logit: conf / params.alg_temp,
                                p: 0.0,
                            })
                            .collect();

                        for _ in 0..take {
                            let mut conf_array = TokenDataArray {
                                size: conf_candidates.len(),
                                selected: -1,
                                sorted: false,
                                // rebuild the Vec each draw the way the C's
                                // cur_p keeps mutating in place
                                data: std::mem::take(&mut conf_candidates),
                            };
                            dist_sampler.apply(&mut conf_array);
                            let selected_idx = conf_array.selected as usize;
                            let mask_idx = selected_idx;
                            let pos = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];

                            conf_candidates = conf_array.data;
                            conf_candidates[selected_idx].p = 0.0;
                        }
                    }
                }
            }

            total_sampling_us += time_start_sampling.elapsed().as_micros() as u64;
        }
    }

    let total_time_us = time_start.elapsed().as_micros() as u64;
    eprintln!(
        "\ntotal time: {:.2}ms, time per step: {:.2}ms, sampling time per step: {:.2}ms",
        total_time_us as f64 / 1000.0,
        total_time_us as f64 / 1000.0 / params.steps as f64,
        total_sampling_us as f64 / 1000.0 / params.steps as f64
    );

    params.max_length
}

/// `main` of diffusion-cli.cpp (diffusion-cli.cpp:102-266): arg plumbing,
/// mask-token lookup, the shift_logits GGUF key, generate, detokenize.
pub fn run(args: &Args, vocab: &Vocab, dctx: &mut DecodeContext, gguf: &ggml::Gguf) {
    // `format_input_text` (:75-100) — the port's diffusion path takes the raw
    // prompt (params.enable_chat_template is not wired for diffusion archs;
    // the chat-template apply of the reference needs a model-side template
    // reader the port resolves through its own chat_tools, unused here)
    let formatted_prompt = args.prompt.clone();

    let input_tokens = vocab.tokenize(&formatted_prompt, true, true);
    let n_input = input_tokens.len();

    if n_input as u32 >= args.n_ctx {
        eprintln!(
            "error: input too long ({n_input} tokens), max context is {}",
            args.n_ctx
        );
        std::process::exit(1);
    }

    let mask_token_id = vocab.token_mask();
    if mask_token_id == llama::vocab::TOKEN_NULL {
        eprintln!("error: model has no mask token");
        std::process::exit(1);
    }

    // `diffusion.shift_logits` GGUF key (:182-187) — default true
    let shift_logits = gguf
        .get_str("diffusion.shift_logits")
        .map(|v| v == "true")
        .unwrap_or(true);

    // Use either eps or block length, but not both (:189-190)
    assert!(
        (args.diffusion_eps == 0.0) ^ (args.diffusion_block_length == 0),
        "diffusion: use either --diffusion-eps or --diffusion-block-length, not both"
    );

    let mut params = DiffusionParams {
        steps: args.diffusion_steps,
        temperature: args.temp,
        mask_token_id,
        seed: args.seed,
        shift_logits,
        top_p: args.top_p,
        top_k: args.top_k,
        algorithm: args.diffusion_algorithm,
        schedule: DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED,
        cfg_scale: args.diffusion_cfg_scale,
        eps: 0.0,
        block_length: 0,
        alg_temp: args.diffusion_alg_temp,
        add_gumbel_noise: args.diffusion_add_gumbel_noise,
        max_length: args.n_ubatch() as i32,
    };
    if args.diffusion_eps != 0.0 {
        params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        params.eps = args.diffusion_eps;
    } else if args.diffusion_block_length != 0 {
        params.schedule = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
        params.block_length = args.diffusion_block_length;
    }

    // the diffusion_params banner (:215-244)
    let alg_names = [
        "DIFFUSION_ALGORITHM_ORIGIN",
        "DIFFUSION_ALGORITHM_ENTROPY_BASED",
        "DIFFUSION_ALGORITHM_MARGIN_BASED",
        "DIFFUSION_ALGORITHM_RANDOM",
        "DIFFUSION_ALGORITHM_CONFIDENCE_BASED",
    ];
    let sched_names = [
        "DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED",
        "DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED",
    ];
    let alg_name = if (0..=4).contains(&params.algorithm) {
        alg_names[params.algorithm as usize]
    } else {
        "UNKNOWN"
    };
    let sched_name = if (0..=1).contains(&params.schedule) {
        sched_names[params.schedule as usize]
    } else {
        "UNKNOWN"
    };
    eprintln!("diffusion_params: - mask_token_id            llama_token      = {mask_token_id}");
    eprintln!("diffusion_params: - steps                   u32              = {}", params.steps);
    eprintln!("diffusion_params: - max_length              u32              = {}", params.max_length);
    eprintln!("diffusion_params: - algorithm               enum             = {} ({alg_name})", params.algorithm);
    eprintln!("diffusion_params: - schedule                enum             = {} ({sched_name})", params.schedule);
    eprintln!("diffusion_params: - temperature             f32              = {:.3}", params.temperature);
    if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED {
        eprintln!("diffusion_params: - eps                     f32              = {:.6}", params.eps);
        eprintln!("diffusion_params: - alg_temp                f32              = {:.3}", params.alg_temp);
    }
    if params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED {
        eprintln!("diffusion_params: - block_length            u32              = {}", params.block_length);
        eprintln!("diffusion_params: - cfg_scale               f32              = {:.3}", params.cfg_scale);
    }

    let max_length = params.max_length as usize;
    let mut output_tokens = vec![0 as Token; max_length];
    let n_generated = generate(
        dctx,
        vocab,
        &input_tokens,
        &mut output_tokens,
        &params,
        args.diffusion_visual,
        n_input,
    );

    if n_generated > 0 {
        if args.diffusion_visual {
            // clear screen and move cursor to top-left
            eprint!("\x1b[2J\x1b[H");
        }

        let out = &output_tokens[n_input..];
        let output_data = vocab.detokenize(out, false);
        println!("\n{output_data}");
    } else {
        println!("Error: diffusion generation failed");
    }
}

impl Args {
    /// `params.n_ubatch` — the port's ubatch is fixed at 512 (the reference
    /// default); `max_length` rides on it
    fn n_ubatch(&self) -> usize {
        512.min(self.n_ctx as usize)
    }
}

// ---------------------------------------------------------------------------
// tests — the pure scheduling/sampling helpers against the C's arithmetic
// (diffusion.cpp:13-101). The decode loop itself is verified by the
// synthetic-arch e2e cells (see PARITY.md): no reference diffusion binary
// exists in the pinned build to compare transcripts against.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tda(size: usize, selected: i64, probs: &[f32]) -> TokenDataArray {
        let mut data = Vec::with_capacity(size);
        for (i, &p) in probs.iter().enumerate() {
            data.push(TokenData {
                id: i as Token,
                logit: 0.0,
                p,
            });
        }
        while data.len() < size {
            data.push(TokenData {
                id: data.len() as Token,
                logit: 0.0,
                p: 0.0,
            });
        }
        TokenDataArray {
            data,
            size,
            selected,
            sorted: false,
        }
    }

    #[test]
    fn confidence_per_algorithm() {
        // diffusion.cpp:13-46
        let mut rng = Mt19937::new(42);
        // selected probability 0.25
        let cur = tda(4, 1, &[0.5, 0.25, 0.25, 0.0]);
        assert_eq!(
            calculate_confidence(&cur, DIFFUSION_ALGORITHM_CONFIDENCE_BASED, &mut rng),
            0.25
        );
        assert_eq!(calculate_confidence(&cur, DIFFUSION_ALGORITHM_ORIGIN, &mut rng), 0.25);
        // margin: data[0].p - data[1].p
        assert_eq!(
            calculate_confidence(&cur, DIFFUSION_ALGORITHM_MARGIN_BASED, &mut rng),
            0.25
        );
        // entropy: -sum p*ln(p+eps)
        let cur2 = tda(2, 0, &[1.0, 0.0]);
        let e = calculate_confidence(&cur2, DIFFUSION_ALGORITHM_ENTROPY_BASED, &mut rng);
        assert!((e - 0.0).abs() < 1e-5, "deterministic dist has ~0 entropy, got {e}");
        // random in [0,1)
        let r = calculate_confidence(&cur, DIFFUSION_ALGORITHM_RANDOM, &mut rng);
        assert!((0.0..1.0).contains(&r));
        // unknown -> 0
        assert_eq!(calculate_confidence(&cur, 9, &mut rng), 0.0);
    }

    #[test]
    fn transfer_count_timestep_schedule() {
        // diffusion.cpp:56-62 — TIMESTEP_BASED
        // remaining=10, steps=10, eps=0: step 0: t=1, s=0.9 -> p=0.1 -> 1
        let n = calculate_transfer_count(0, 10, 10, DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED, 0.0, &[]);
        assert_eq!(n, 1);
        // last step transfers everything
        let n = calculate_transfer_count(9, 10, 5, DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED, 0.0, &[]);
        assert_eq!(n, 5);
        // eps=0.1: step 0: t=1, s=1-0.1*(0.9)=0.91 -> p=0.09 -> floor(10*0.09)=0
        let n = calculate_transfer_count(0, 10, 10, DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED, 0.1, &[]);
        assert_eq!(n, 0);
    }

    #[test]
    fn transfer_count_block_and_default() {
        // diffusion.cpp:64-72
        let n = calculate_transfer_count(0, 10, 9, DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED, 0.0, &[3, 3, 3]);
        assert_eq!(n, 3);
        // out of range -> fallback division
        let n = calculate_transfer_count(5, 10, 9, DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED, 0.0, &[3]);
        assert_eq!(n, 1); // 9 / (10-5)
        // unknown schedule -> fallback
        assert_eq!(calculate_transfer_count(1, 4, 6, 7, 0.0, &[]), 2);
    }

    #[test]
    fn num_transfer_tokens_even_split() {
        // diffusion.cpp:90-101
        assert_eq!(get_num_transfer_tokens(7, 3), vec![3, 2, 2]);
        assert_eq!(get_num_transfer_tokens(6, 3), vec![2, 2, 2]);
        assert_eq!(get_num_transfer_tokens(0, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn gumbel_noise_shape() {
        // diffusion.cpp:75-88 — exp(logit)/pow(-log u, T); deterministic for
        // a fixed seed through the port's bit-exact generate_canonical
        let mut rng = Mt19937::new(7);
        let mut logits = vec![1.0f32, 2.0];
        add_gumbel_noise(&mut logits, 1.0, &mut rng);
        // both got rescaled (finite, positive)
        for l in logits {
            assert!(l.is_finite() && l > 0.0);
        }
        // temperature 0 is a no-op
        let mut rng2 = Mt19937::new(7);
        let mut frozen = vec![1.0f32, 2.0];
        add_gumbel_noise(&mut frozen, 0.0, &mut rng2);
        assert_eq!(frozen, vec![1.0, 2.0]);
    }
}
