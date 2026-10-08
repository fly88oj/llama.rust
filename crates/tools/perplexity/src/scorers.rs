//! HellaSwag / Winogrande / multiple-choice scorers — port of
//! `tools/perplexity/perplexity.cpp` (pinned bd4f514db1):
//!   * `hellaswag_score`        (perplexity.cpp:744-1015)
//!   * `load_winogrande_from_csv` (:1031-1090) + `winogrande_score` (:1101-1301)
//!   * `deserialize_string`/`multiple_choice_answers`/`multiple_choice_task`
//!     (:1303-1344) + `multiple_choice_prepare_one_task` (:1346-1387) +
//!     `multiple_choice_score` (:1405-1693)
//!   * `decode_helper` (:664-696), `compute_logprobs` (:700-742),
//!     `softmax` (:40-58)
//!
//! Stream mapping (common/log.cpp:113-116): `LOG()` (level NONE) → stdout,
//! `LOG_INF`/`LOG_ERR` → stderr. The accumulated-score tables are stdout;
//! progress/status lines are stderr.
//!
//! Randomness must be bit-exact with the reference binary (built with the
//! local g++ 13 / libstdc++ 13):
//!   * `std::mt19937` — the standard MT19937 twister (32-bit draws).
//!   * `std::uniform_int_distribution<size_t>` under libstdc++ 13 takes the
//!     `_S_nd<uint64_t>` Lemire "nearly divisionless" branch for a generator
//!     whose range is exactly 2^32-1 (bits/uniform_int_dist.h:301-311) —
//!     ported verbatim in [`lemire_u64`].
//!   * the winogrande/multiple-choice selections use the raw
//!     `int(scale*rng()*aux.size())` float idiom — ported in f32 arithmetic.

use std::io::Write as _;

use llama::batch::LlamaBatch;
use llama::context::DecodeContext;
use llama::vocab::{Vocab, VocabType};

/// `K_TOKEN_CHUNK` (perplexity.cpp:698)
const K_TOKEN_CHUNK: usize = 4;

// ---------------------------------------------------------------------------
// std::mt19937 + the two selection idioms
// ---------------------------------------------------------------------------

/// `std::mt19937` (MT19937, 32-bit output; parameters from the standard).
pub struct Mt19937 {
    state: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, idx: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000)
                    | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624] ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.state[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

/// Lemire's nearly-divisionless downscale `_S_nd<uint64_t>` of libstdc++ 13
/// (bits/uniform_int_dist.h:262-287): one draw r ∈ [0, 2^32), product = r *
/// range in u64; reject while the low 32 bits are below `-range % range`;
/// result = product >> 32.
fn lemire_u64(rng: &mut Mt19937, range: u32) -> u64 {
    let mut product = (rng.next_u32() as u64) * (range as u64);
    let mut low = product as u32;
    if low < range {
        let threshold = 0u32.wrapping_sub(range) % range;
        while low < threshold {
            product = (rng.next_u32() as u64) * (range as u64);
            low = product as u32;
        }
    }
    product >> 32
}

/// `std::uniform_int_distribution<size_t> dist(a, b); dist(rng)` as compiled
/// by g++ 13 (the reference build's compiler): mt19937's urngrange is
/// exactly UINT32_MAX, so the `__urngrange == __UINT32_MAX__` branch picks
/// `_S_nd<uint64_t>` with the range widened to u32 (uniform_int_dist.h:
/// 296-311).
fn uniform_int_size_t(rng: &mut Mt19937, a: usize, b: usize) -> usize {
    let uerange = (b - a + 1) as u32; // __urange + 1
    let ret = lemire_u64(rng, uerange);
    ret as usize + a
}

/// `int j = int(scale*rng()*aux.size())` with
/// `float scale = 1/(1.f + (float)rng.max())` (= 2^-32 exactly) — the
/// winogrande (:1122-1126) and multiple-choice (:1446-1449) selections.
fn scale_draw_index(rng: &mut Mt19937, aux_len: usize) -> usize {
    let scale = 1.0f32 / (1.0f32 + u32::MAX as f32);
    let r = rng.next_u32() as f32;
    let j = ((scale * r) * (aux_len as f32)) as i32 as usize;
    // guard the C's out-of-bounds corner (top ~2^-24 draws round scale*r up
    // to exactly 1.0f) — undefined behavior there, clamped here
    j.min(aux_len - 1)
}

// ---------------------------------------------------------------------------
// softmax / logprobs (perplexity.cpp:40-58, :700-742)
// ---------------------------------------------------------------------------

/// `softmax` (perplexity.cpp:40-58): f32 exps, f64 sum, f32 /= sum.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut probs = vec![0f32; logits.len()];
    let mut max_logit = logits[0];
    for &v in logits {
        max_logit = max_logit.max(v);
    }
    let mut sum_exp = 0.0f64;
    for i in 0..logits.len() {
        // Subtract the maximum logit value for numerical stability
        let exp_logit = (logits[i] - max_logit).exp();
        sum_exp += exp_logit as f64;
        probs[i] = exp_logit;
    }
    for p in &mut probs {
        *p = (*p as f64 / sum_exp) as f32;
    }
    probs
}

/// `compute_logprobs` (perplexity.cpp:700-742), single-threaded (the K_TOKEN_CHUNK
/// partitioning only affects which thread writes which result).
fn compute_logprobs(
    batch_logits: &[f32],
    n_vocab: usize,
    eval_pairs: &[(usize, i32)],
    eval_results: &mut Vec<f32>,
) {
    eval_results.clear();
    eval_results.resize(eval_pairs.len(), 0.0);
    for (i, (row, tok)) in eval_pairs.iter().enumerate() {
        let logits = &batch_logits[row * n_vocab..(row + 1) * n_vocab];
        let mut max_logit = logits[0];
        for &v in &logits[1..] {
            max_logit = max_logit.max(v);
        }
        let mut sum_p = 0.0f32;
        for &v in logits {
            sum_p += (v - max_logit).exp();
        }
        eval_results[i] = logits[*tok as usize] - max_logit - sum_p.ln();
    }
}

// ---------------------------------------------------------------------------
// shared batch driver
// ---------------------------------------------------------------------------

/// `decode_helper` (perplexity.cpp:664-696): decode the batch and gather the
/// logits of every output-flagged token into `batch_logits`, in batch order.
/// The port's `decode_batch` already splits into `n_batch` ubatches internally
/// (the C's explicit batch views) and returns the output rows in batch order —
/// causality makes the values independent of the split points.
fn decode_helper(
    dctx: &mut DecodeContext,
    batch: &LlamaBatch,
    batch_logits: &mut Vec<f32>,
    n_vocab: usize,
) -> bool {
    match dctx.decode_batch(batch) {
        Ok(out) => {
            batch_logits.clear();
            batch_logits.extend_from_slice(&out.logits);
            let _ = n_vocab;
            true
        }
        Err(e) => {
            eprintln!("failed to decode the batch, ret = {e}");
            false
        }
    }
}

/// `common_tokenize(ctx, text, true)` — add_special=true, parse_special=false.
fn tokenize(vocab: &Vocab, text: &str) -> Vec<i32> {
    vocab.tokenize(text, true, false)
}

/// `std::getline(strstream, line, '\n')` loop (hellaswag idiom, :769-771):
/// a final line without trailing '\n' IS processed.
fn getline_all(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.is_empty() {
        lines.clear();
    } else if text.ends_with('\n') {
        lines.pop();
    }
    lines
}

// ---------------------------------------------------------------------------
// hellaswag_score (perplexity.cpp:744-1015)
// ---------------------------------------------------------------------------

pub struct ScorerParams {
    /// `llama_n_ctx(ctx)` — n_parallel * the -c value
    pub n_ctx: usize,
    /// `llama_n_seq_max(ctx)` — n_parallel
    pub n_seq_max: usize,
    pub n_batch: usize,
    pub hellaswag_tasks: usize,
}

struct HsData {
    context: String,
    gold_ending_idx: usize,
    ending: [String; 4],
    ending_logprob_count: [usize; 4],
    ending_logprob: [f64; 4],

    i_logits: usize,      // starting index of logits in the batch
    common_prefix: usize, // max number of initial tokens that are the same in all sentences
    required_tokens: usize,
    seq_tokens: [Vec<i32>; 4],
}

/// `std::stoi` — leading whitespace/sign/digits; the reference lets a parse
/// failure terminate the process, so a failure here aborts.
fn stoi(s: &str) -> i32 {
    let t = s.trim_start();
    let bytes = t.as_bytes();
    let mut i = 0;
    let neg = !bytes.is_empty() && bytes[0] == b'-';
    if !bytes.is_empty() && (bytes[0] == b'+' || bytes[0] == b'-') {
        i += 1;
    }
    let mut val: i64 = 0;
    let mut any = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        val = (val * 10 + (bytes[i] - b'0') as i64).min(i64::from(i32::MAX));
        i += 1;
        any = true;
    }
    if !any {
        panic!("stoi: no conversion: '{s}'");
    }
    if neg {
        val = -val;
    }
    val as i32
}

pub fn hellaswag_score(
    dctx: &mut DecodeContext,
    vocab: &Vocab,
    params: &ScorerParams,
    prompt: &str,
) {
    // Calculates hellaswag score (acc_norm) from prompt
    //
    // Data extracted from the HellaSwag validation dataset (MIT license)
    // https://github.com/rowanz/hellaswag/blob/master/data/hellaswag_val.jsonl
    // All used data fields are preprocessed as in
    // lm-evaluation-harness/lm_eval/tasks/hellaswag.py#L62-L68
    //
    // Datafile layout: 6 lines per task:
    //   ['activity_label'] + ": " + ['ctx']  - the context
    //   ['label'] - index of the gold ending
    //   ['endings'][0..3]

    let mut prompt_lines: Vec<String> = getline_all(prompt).iter().map(|s| s.to_string()).collect();

    if prompt_lines.len() % 6 != 0 {
        eprintln!("hellaswag_score : number of lines in prompt not a multiple of 6.");
        return;
    }

    let mut hs_task_count = prompt_lines.len() / 6;
    eprintln!("hellaswag_score : loaded {hs_task_count} tasks from prompt.");

    let is_spm = vocab.ty == VocabType::Spm;
    eprintln!("================================= is_spm = {}", is_spm as i32);

    // The tasks should be randomized so the score stabilizes quickly.
    let randomize_tasks = true;

    // Number of tasks to use when computing the score
    if params.hellaswag_tasks < hs_task_count {
        hs_task_count = params.hellaswag_tasks;
    }

    // The random seed should not impact the final result, kept hardcoded
    let mut rng = Mt19937::new(1);

    eprintln!(
        "hellaswag_score : selecting {hs_task_count} {} tasks.",
        if randomize_tasks { "randomized" } else { "the first" }
    );

    // Select and read data from prompt lines
    let mut hs_data: Vec<HsData> = Vec::with_capacity(hs_task_count);
    for i in 0..hs_task_count {
        let mut idx = i;

        // Select a random example of those left in the prompt
        if randomize_tasks {
            idx = uniform_int_size_t(&mut rng, 0, prompt_lines.len() / 6 - 1);
        }

        let context = prompt_lines[idx * 6].clone();
        let gold_ending_idx = stoi(&prompt_lines[idx * 6 + 1]) as usize;
        let mut ending: [String; 4] = Default::default();
        let mut seq_tokens: [Vec<i32>; 4] = Default::default();
        for (j, e) in ending.iter_mut().enumerate() {
            *e = prompt_lines[idx * 6 + 2 + j].clone();
            seq_tokens[j] = tokenize(vocab, &format!("{} {}", context, e));
        }

        // determine the common prefix of the endings (the C indexes
        // seq_tokens[1..3][k] for k < seq_tokens[0].len() — out of bounds when
        // a shorter ending ends first; guarded to "mismatch" here)
        let mut common_prefix = 0usize;
        'outer: for k in 0..seq_tokens[0].len() {
            for s in 1..4 {
                if k >= seq_tokens[s].len() || seq_tokens[s][k] != seq_tokens[0][k] {
                    break 'outer;
                }
            }
            common_prefix += 1;
        }
        let required_tokens = common_prefix
            + seq_tokens.iter().map(|t| t.len() - common_prefix).sum::<usize>();

        hs_data.push(HsData {
            context,
            gold_ending_idx,
            ending,
            ending_logprob_count: [0; 4],
            ending_logprob: [0.0; 4],
            i_logits: 0,
            common_prefix,
            required_tokens,
            seq_tokens,
        });

        // Delete the selected random example from the prompt
        if randomize_tasks {
            prompt_lines.drain(idx * 6..idx * 6 + 6);
        }
    }

    eprintln!("hellaswag_score : calculating hellaswag score over selected tasks.");

    println!("\ntask\tacc_norm\t95% confidence interval");

    let mut acc = 0.0f64;

    let n_ctx = params.n_ctx;
    let n_vocab = dctx.n_vocab();

    let max_tasks_per_batch = 32usize;
    // min(4*32, llama_n_seq_max) — the port caps sequences at LLAMA_MAX_SEQ
    // (64) like the reference's unified-KV bitset
    let max_seq = (4 * max_tasks_per_batch).min(params.n_seq_max).min(64);

    let mut batch = LlamaBatch::default();
    let mut tok_logits: Vec<f32>;
    let mut batch_logits: Vec<f32> = Vec::with_capacity(n_ctx * n_vocab);
    let mut eval_pairs: Vec<(usize, i32)> = Vec::new();
    let mut eval_results: Vec<f32> = Vec::new();

    let mut i0 = 0usize;
    while i0 < hs_task_count {
        let mut n_cur = 0usize;

        let mut i1 = i0;
        let mut i_logits = 0usize; // logits needed before this point in the batch

        batch.clear();

        // batch as much tasks as possible into the available context
        // each task has 4 unique sequence ids - one for each ending
        // the common prefix is shared among the 4 sequences to save tokens
        // we extract logits only from the last common token and from all
        // ending tokens of each sequence
        while n_cur + hs_data[i1].required_tokens <= n_ctx {
            let mut n_logits_of = |hs_cur: &HsData, s0: i32| -> usize {
                let mut n_logits = 0;
                for i in 0..hs_cur.common_prefix {
                    batch.add(hs_cur.seq_tokens[0][i], i as i32, &[s0, s0 + 1, s0 + 2, s0 + 3], false);
                }
                if let Some(l) = batch.logits.as_mut() {
                    let n = l.len();
                    l[n - 1] = true; // logits for the last token of the common prefix
                }
                n_logits += 1;
                for s in 0..4 {
                    let seq_tokens_size = hs_cur.seq_tokens[s].len();
                    // (TODO upstream: don't evaluate the last token of each sequence)
                    for i in hs_cur.common_prefix..seq_tokens_size {
                        let needs_logits = i < seq_tokens_size - 1;
                        batch.add(hs_cur.seq_tokens[s][i], i as i32, &[s0 + s as i32], needs_logits);
                        n_logits += needs_logits as usize;
                    }
                }
                n_logits
            };

            let s0 = 4 * (i1 - i0) as i32;
            if s0 + 4 > max_seq as i32 {
                break;
            }
            let n = n_logits_of(&hs_data[i1], s0);

            hs_data[i1].i_logits = i_logits;
            i_logits += n;

            n_cur += hs_data[i1].required_tokens;
            i1 += 1;
            if i1 == hs_task_count {
                break;
            }
        }

        if i0 == i1 {
            eprintln!(
                "hellaswag_score : task {i0} does not fit in the context window (requires {} tokens)",
                hs_data[i0].required_tokens
            );
            return;
        }

        dctx.reset_sequence(); // llama_memory_clear

        // decode all tasks [i0, i1)
        if !decode_helper(dctx, &batch, &mut batch_logits, n_vocab) {
            eprintln!("hellaswag_score: llama_decode() failed");
            return;
        }

        // Compute log-probs: first collect all tasks
        eval_pairs.clear();
        for hs_cur in &hs_data[i0..i1] {
            let mut li = 1; // skip the last logit of the common prefix
            for s in 0..4 {
                for j in hs_cur.common_prefix..hs_cur.seq_tokens[s].len() - 1 {
                    eval_pairs.push((hs_cur.i_logits + li, hs_cur.seq_tokens[s][j + 1]));
                    li += 1;
                }
            }
        }
        // Then we do the actual calculation
        compute_logprobs(&batch_logits, n_vocab, &eval_pairs, &mut eval_results);

        let mut ir = 0usize;

        // compute the logprobs for each ending of the decoded tasks
        for i in i0..i1 {
            // get the logits of the last token of the common prefix
            tok_logits = batch_logits
                [hs_data[i].i_logits * n_vocab..(hs_data[i].i_logits + 1) * n_vocab]
                .to_vec();
            let first_probs = softmax(&tok_logits);

            for s in 0..4 {
                hs_data[i].ending_logprob_count[s] = 1;
                hs_data[i].ending_logprob[s] =
                    first_probs[hs_data[i].seq_tokens[s][hs_data[i].common_prefix] as usize].ln()
                        as f64;
                for j in hs_data[i].common_prefix..hs_data[i].seq_tokens[s].len() - 1 {
                    hs_data[i].ending_logprob[s] += eval_results[ir] as f64;
                    ir += 1;
                    hs_data[i].ending_logprob_count[s] += 1;
                }
                hs_data[i].ending_logprob[s] /= hs_data[i].ending_logprob_count[s] as f64;
            }

            // Find the ending with maximum logprob
            let mut ending_logprob_max_idx = 0usize;
            let mut ending_logprob_max_val = hs_data[i].ending_logprob[0];
            for s in 1..4 {
                if hs_data[i].ending_logprob[s] > ending_logprob_max_val {
                    ending_logprob_max_idx = s;
                    ending_logprob_max_val = hs_data[i].ending_logprob[s];
                }
            }

            // If the gold ending got the maximum logprob add one accuracy point
            if ending_logprob_max_idx == hs_data[i].gold_ending_idx {
                acc += 1.0;
            }

            let freq = acc / (i + 1) as f64;

            let za = 1.95996398454f64;

            // Wilson score interval, more accurate
            let z = za * za / (i + 1) as f64;
            let cnf = z * ((i + 1) as f64 * (4.0 * freq * (1.0 - freq) + z)).sqrt() / (za + za);
            let a = (freq + z * 0.5 - cnf) / (1.0 + z);
            let b = (freq + z * 0.5 + cnf) / (1.0 + z);

            // Print the accumulated accuracy mean x 100 and confidence interval
            println!("{}\t{:.8}%\t[{:.4}%, {:.4}%]", i + 1, freq * 100.0, a * 100.0, b * 100.0);
        }

        i0 = i1 - 1;
        i0 += 1;
    }

    println!();
    let _ = std::io::stdout().flush();
}

// ---------------------------------------------------------------------------
// winogrande (perplexity.cpp:1017-1301)
// ---------------------------------------------------------------------------

struct WinograndeEntry {
    first: String,
    second: String,
    choices: [String; 2],
    answer: i32,

    i_logits: usize,
    common_prefix: usize,
    required_tokens: usize,
    n_base1: usize, // tokens for context + choice 1
    n_base2: usize, // tokens for context + choice 2
    seq_tokens: [Vec<i32>; 2],
}

/// `load_winogrande_from_csv` (perplexity.cpp:1031-1090). CSV line shape:
/// `index,sentence,choice1,choice2,answer`. Note the getline idiom
/// `if (in.fail() || in.eof()) break;` (:1038) — a *final line without a
/// trailing newline is dropped* (the -f handler strips exactly one trailing
/// '\n'), so only '\n'-terminated lines are processed.
fn load_winogrande_from_csv(prompt: &str) -> Vec<WinograndeEntry> {
    let mut result = Vec::new();
    let all: Vec<&str> = prompt.split('\n').collect();
    // getline with the fail||eof check processes all but the last piece
    // (that piece is either the trailing "" after a final '\n' or an
    // unterminated tail)
    let lines = &all[..all.len() - 1];
    for line in lines {
        let bytes = line.as_bytes();
        let mut comma_pos = [0usize; 4];
        let mut ipos = 0;
        let mut quote_open = false;
        for (i, &b) in bytes.iter().enumerate() {
            if !quote_open {
                if b == b',' {
                    comma_pos[ipos] = i;
                    ipos += 1;
                    if ipos == 4 {
                        break;
                    }
                } else if b == b'"' {
                    quote_open = true;
                }
            } else if b == b'"' {
                quote_open = false;
            }
        }
        if ipos != 4 {
            eprintln!("load_winogrande_from_csv: failed to find comma separators in <{line}>");
            continue;
        }
        let sentence = if bytes[comma_pos[0] + 1] == b'"' {
            &line[comma_pos[0] + 2..comma_pos[1] - 1]
        } else {
            &line[comma_pos[0] + 1..comma_pos[1]]
        };
        let choice1 = &line[comma_pos[1] + 1..comma_pos[2]];
        let choice2 = &line[comma_pos[2] + 1..comma_pos[3]];
        let answer = &line[comma_pos[3] + 1..];
        let _index = &line[..comma_pos[0]];
        let mut where_ = 0usize;
        while where_ < sentence.len() {
            if sentence.as_bytes()[where_] == b'_' {
                break;
            }
            where_ += 1;
        }
        if where_ == sentence.len() {
            eprintln!("load_winogrande_from_csv: no _ in <{sentence}>");
            continue;
        }
        // std::istringstream stream(answer); stream >> i_answer
        let i_answer: i32 = match answer.trim_start().split(|c: char| !c.is_ascii_digit() && c != '-' && c != '+')
            .next()
            .and_then(|t| t.parse::<i64>().ok().map(|v| v as i32))
        {
            Some(v) => v,
            None => {
                eprintln!("load_winogrande_from_csv: failed to parse answer <{answer}>");
                continue;
            }
        };
        if !(1..=2).contains(&i_answer) {
            eprintln!("load_winogrande_from_csv: failed to parse answer <{answer}>");
            continue;
        }
        result.push(WinograndeEntry {
            first: sentence[..where_].to_string(),
            second: sentence[where_ + 1..].to_string(),
            choices: [choice1.to_string(), choice2.to_string()],
            answer: i_answer,
            i_logits: 0,
            common_prefix: 0,
            required_tokens: 0,
            n_base1: 0,
            n_base2: 0,
            seq_tokens: [Vec::new(), Vec::new()],
        });
    }
    result
}

pub fn winogrande_score(
    dctx: &mut DecodeContext,
    vocab: &Vocab,
    params: &ScorerParams,
    prompt: &str,
    winogrande_tasks: usize,
) {
    // Evaluates the Winogrande score.
    // Uses a CSV containing task index, sentence, choice 1, choice 2, answer (1 or 2)

    const K_MIN_TRAILING_CTX: usize = 3;

    let mut data = load_winogrande_from_csv(prompt);
    if data.is_empty() {
        eprintln!("winogrande_score: no tasks");
        return;
    }

    eprintln!("winogrande_score : loaded {} tasks from prompt.", data.len());

    if winogrande_tasks > 0 && winogrande_tasks < data.len() {
        eprintln!("winogrande_score : selecting {winogrande_tasks} random tasks");
        let mut rng = Mt19937::new(1);
        let mut aux: Vec<usize> = (0..data.len()).collect();
        let mut selected: Vec<WinograndeEntry> = Vec::with_capacity(winogrande_tasks);
        for _ in 0..winogrande_tasks {
            let j = scale_draw_index(&mut rng, aux.len());
            let idx = aux[j];
            // std::move leaves the source moved-from; the port removes it
            selected.push(std::mem::replace(
                &mut data[idx],
                WinograndeEntry {
                    first: String::new(),
                    second: String::new(),
                    choices: [String::new(), String::new()],
                    answer: 0,
                    i_logits: 0,
                    common_prefix: 0,
                    required_tokens: 0,
                    n_base1: 0,
                    n_base2: 0,
                    seq_tokens: [Vec::new(), Vec::new()],
                },
            ));
            aux[j] = *aux.last().unwrap();
            aux.pop();
        }
        data = selected;
    }

    eprintln!("winogrande_score : tokenizing selected tasks");

    for task in &mut data {
        task.seq_tokens[0] = tokenize(vocab, &format!("{}{}{}", task.first, task.choices[0], task.second));
        task.seq_tokens[1] = tokenize(vocab, &format!("{}{}{}", task.first, task.choices[1], task.second));

        task.common_prefix = 0;
        // (same out-of-bounds guard as hellaswag's common-prefix loop)
        'outer: for k in 0..task.seq_tokens[0].len() {
            if k >= task.seq_tokens[1].len() || task.seq_tokens[0][k] != task.seq_tokens[1][k] {
                break 'outer;
            }
            task.common_prefix += 1;
        }

        // TODO(upstream): the last token of each sequence need not be evaluated
        task.required_tokens = task.common_prefix
            + (task.seq_tokens[0].len() - task.common_prefix)
            + (task.seq_tokens[1].len() - task.common_prefix);

        task.n_base1 = tokenize(vocab, &format!("{}{}", task.first, task.choices[0])).len();
        task.n_base2 = tokenize(vocab, &format!("{}{}", task.first, task.choices[1])).len();
    }

    eprintln!("winogrande_score : calculating winogrande score over selected tasks.");

    let n_ctx = params.n_ctx;
    let n_vocab = dctx.n_vocab();

    let max_tasks_per_batch = 128usize;
    let max_seq = (2 * max_tasks_per_batch).min(params.n_seq_max).min(64);

    let mut batch = LlamaBatch::default();
    let mut batch_logits: Vec<f32> = Vec::with_capacity(n_ctx * n_vocab);
    let mut eval_pairs: Vec<(usize, i32)> = Vec::new();
    let mut eval_results: Vec<f32> = Vec::new();

    let mut n_correct = 0i32;
    let mut n_done = 0i32;

    let mut i0 = 0usize;
    while i0 < data.len() {
        let mut n_cur = 0usize;

        let mut i1 = i0;
        let mut i_logits = 0usize;

        batch.clear();

        while n_cur + data[i1].required_tokens <= n_ctx {
            let s0 = 2 * (i1 - i0) as i32;
            if s0 + 2 > max_seq as i32 {
                break;
            }
            let mut n_logits = 0usize;

            for i in 0..data[i1].common_prefix {
                batch.add(data[i1].seq_tokens[0][i], i as i32, &[s0, s0 + 1], false);
            }
            if let Some(l) = batch.logits.as_mut() {
                let n = l.len();
                l[n - 1] = true;
            }
            n_logits += 1;

            for s in 0..2 {
                // TODO(upstream): end before the last token
                for i in data[i1].common_prefix..data[i1].seq_tokens[s].len() {
                    batch.add(data[i1].seq_tokens[s][i], i as i32, &[s0 + s as i32], true);
                    n_logits += 1;
                }
            }

            data[i1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += data[i1].required_tokens;
            i1 += 1;
            if i1 == data.len() {
                break;
            }
        }

        if i0 == i1 {
            eprintln!(
                "winogrande_score : task {i0} does not fit in the context window (requires {} tokens)",
                data[i0].required_tokens
            );
            return;
        }

        dctx.reset_sequence(); // llama_memory_clear

        // decode all tasks [i0, i1)
        if !decode_helper(dctx, &batch, &mut batch_logits, n_vocab) {
            eprintln!("winogrande_score: llama_decode() failed");
            return;
        }

        eval_pairs.clear();
        for task in &data[i0..i1] {
            let skip_choice = task.seq_tokens[0].len() - task.common_prefix > K_MIN_TRAILING_CTX
                && task.seq_tokens[1].len() - task.common_prefix > K_MIN_TRAILING_CTX;

            let n_base1 = if skip_choice { task.n_base1 } else { task.common_prefix };
            let last_1st = if task.seq_tokens[0].len() - n_base1 > 1 { 1 } else { 0 };
            let mut li = n_base1 - task.common_prefix;
            for j in n_base1 - 1..task.seq_tokens[0].len() - 1 - last_1st {
                eval_pairs.push((task.i_logits + li, task.seq_tokens[0][j + 1]));
                li += 1;
            }
            let n_base2 = if skip_choice { task.n_base2 } else { task.common_prefix };
            let last_2nd = if task.seq_tokens[1].len() - n_base2 > 1 { 1 } else { 0 };
            // FIXME(upstream): this uses the wrong first logits when not
            // skipping the choice word
            li = task.seq_tokens[0].len() - task.common_prefix + n_base2 - task.common_prefix;
            for j in n_base2 - 1..task.seq_tokens[1].len() - 1 - last_2nd {
                eval_pairs.push((task.i_logits + li, task.seq_tokens[1][j + 1]));
                li += 1;
            }
        }
        compute_logprobs(&batch_logits, n_vocab, &eval_pairs, &mut eval_results);

        let mut ir = 0usize;
        for i in i0..i1 {
            let task = &data[i];

            let skip_choice = task.seq_tokens[0].len() - task.common_prefix > K_MIN_TRAILING_CTX
                && task.seq_tokens[1].len() - task.common_prefix > K_MIN_TRAILING_CTX;

            let n_base1 = if skip_choice { task.n_base1 } else { task.common_prefix };
            let last_1st = if task.seq_tokens[0].len() - n_base1 > 1 { 1 } else { 0 };
            let mut score_1st = 0.0f32;
            for _ in n_base1 - 1..task.seq_tokens[0].len() - 1 - last_1st {
                score_1st += eval_results[ir];
                ir += 1;
            }
            // (size_t subtraction wraps in the C on n_base > len; wrap too)
            score_1st /= task.seq_tokens[0].len().wrapping_sub(n_base1).wrapping_sub(last_1st) as f32;

            let n_base2 = if skip_choice { task.n_base2 } else { task.common_prefix };
            let last_2nd = if task.seq_tokens[1].len() - n_base2 > 1 { 1 } else { 0 };
            let mut score_2nd = 0.0f32;
            for _ in n_base2 - 1..task.seq_tokens[1].len() - 1 - last_2nd {
                score_2nd += eval_results[ir];
                ir += 1;
            }
            score_2nd /= task.seq_tokens[1].len().wrapping_sub(n_base2).wrapping_sub(last_2nd) as f32;

            let result = if score_1st > score_2nd { 1 } else { 2 };

            if result == task.answer {
                n_correct += 1;
            }
            n_done += 1;

            // print the accumulated accuracy mean x 100
            println!(
                "{}\t{:.4}\t{:10.6}  {:10.6}  {}  {}",
                i + 1,
                100.0 * n_correct as f64 / n_done as f64,
                score_1st,
                score_2nd,
                result,
                task.answer
            );
        }

        i0 = i1 - 1;
        i0 += 1;
    }

    println!();
    let _ = std::io::stdout().flush();

    if n_done < 100 {
        return;
    }

    let p = (n_correct as f32) / (n_done as f32);
    let sigma = 100.0f32 * (p * (1.0 - p) / (n_done - 1) as f32).sqrt();

    // "%.4lf" of the promoted `100*p` / sigma (float) values
    eprintln!(
        "Final Winogrande score({n_done} tasks): {:.4} +/- {:.4}",
        (100.0f32 * p) as f64,
        sigma as f64
    );
}

// ---------------------------------------------------------------------------
// multiple choice (perplexity.cpp:1303-1693)
// ---------------------------------------------------------------------------

/// `deserialize_string` (perplexity.cpp:1303-1310)
fn deserialize_string(data: &[u8], pos: &mut usize) -> Option<String> {
    if *pos + 4 > data.len() {
        return None;
    }
    let size = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
    *pos += 4;
    if *pos + size > data.len() {
        return None;
    }
    let s = String::from_utf8_lossy(&data[*pos..*pos + size]).into_owned();
    *pos += size;
    Some(s)
}

/// `multiple_choice_answers` (perplexity.cpp:1312-1327)
struct MultipleChoiceAnswers {
    answers: Vec<String>,
    labels: Vec<i32>,
}

impl MultipleChoiceAnswers {
    fn deserialize(data: &[u8], pos: &mut usize) -> Option<Self> {
        if *pos + 4 > data.len() {
            return None;
        }
        let n = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
        *pos += 4;
        if n > 100 {
            return None; // 100 as max. number of answers
        }
        let mut answers = Vec::with_capacity(n);
        for _ in 0..n {
            answers.push(deserialize_string(data, pos)?);
        }
        let mut labels = vec![0i32; n];
        if *pos + n * 4 > data.len() {
            return None;
        }
        for (i, l) in labels.iter_mut().enumerate() {
            *l = i32::from_le_bytes(data[*pos + i * 4..*pos + i * 4 + 4].try_into().unwrap());
        }
        *pos += n * 4;
        Some(MultipleChoiceAnswers { answers, labels })
    }
}

/// `multiple_choice_task` (perplexity.cpp:1329-1344)
struct MultipleChoiceTask {
    question: String,
    mc1: MultipleChoiceAnswers,
    _mc2: MultipleChoiceAnswers,

    // For evaluation
    i_logits: usize,
    common_prefix: usize,
    required_tokens: usize,
    seq_tokens: Vec<Vec<i32>>,
    log_probs: Vec<f32>,
}

impl MultipleChoiceTask {
    fn deserialize(data: &[u8], pos: &mut usize) -> Option<Self> {
        let question = deserialize_string(data, pos)?;
        let mc1 = MultipleChoiceAnswers::deserialize(data, pos)?;
        let mc2 = MultipleChoiceAnswers::deserialize(data, pos)?;
        Some(MultipleChoiceTask {
            question,
            mc1,
            _mc2: mc2,
            i_logits: 0,
            common_prefix: 0,
            required_tokens: 0,
            seq_tokens: Vec::new(),
            log_probs: Vec::new(),
        })
    }
}

/// `multiple_choice_prepare_one_task` (perplexity.cpp:1346-1387)
fn multiple_choice_prepare_one_task(vocab: &Vocab, task: &mut MultipleChoiceTask, log_error: bool) -> bool {
    if task.question.is_empty() || task.mc1.answers.is_empty() {
        if log_error {
            eprintln!("multiple_choice_prepare_one_task: found bad task with empty question and/or answers");
        }
        return false;
    }
    task.seq_tokens.reserve(task.mc1.answers.len());
    for answer in &task.mc1.answers {
        if answer.is_empty() {
            if log_error {
                eprintln!("multiple_choice_prepare_one_task: found empty answer");
            }
            return false;
        }
        task.seq_tokens.push(tokenize(vocab, &format!("{} {}", task.question, answer)));
    }
    let mut min_len = task.seq_tokens[0].len();
    for seq in &task.seq_tokens {
        min_len = min_len.min(seq.len());
    }
    task.common_prefix = 0;
    for k in 0..min_len {
        let token = task.seq_tokens[0][k];
        let mut all_same = true;
        for seq in task.seq_tokens.iter().skip(1) {
            if seq[k] != token {
                all_same = false;
                break;
            }
        }
        if !all_same {
            break;
        }
        task.common_prefix += 1;
    }
    task.required_tokens = task.common_prefix;
    for seq in &task.seq_tokens {
        task.required_tokens += seq.len() - task.common_prefix;
    }
    true
}

/// Writes the reference's binary multiple-choice dataset layout (the inverse
/// of `multiple_choice_task::deserialize`) — used by the tests and the parity
/// fixtures to build inputs.
pub fn serialize_mc_task(out: &mut Vec<u8>, question: &str, answers: &[&str], label: usize) {
    out.extend_from_slice(&(question.len() as u32).to_le_bytes());
    out.extend_from_slice(question.as_bytes());
    // mc1
    out.extend_from_slice(&(answers.len() as u32).to_le_bytes());
    for a in answers {
        out.extend_from_slice(&(a.len() as u32).to_le_bytes());
        out.extend_from_slice(a.as_bytes());
    }
    for (i, _) in answers.iter().enumerate() {
        out.extend_from_slice(&((i == label) as i32).to_le_bytes());
    }
    // mc2: one u32 answer count of 0 (an empty set deserializes to just that)
    out.extend_from_slice(&0u32.to_le_bytes());
}

/// `multiple_choice_score` (perplexity.cpp:1405-1693). `prompt` is the raw
/// binary dataset read with -f.
pub fn multiple_choice_score(
    dctx: &mut DecodeContext,
    vocab: &Vocab,
    params: &ScorerParams,
    prompt: &[u8],
    multiple_choice_tasks: usize,
) {
    // Calculates score for multiple choice tasks with single correct answer
    // (TruthfulQA / ARC / MMLU style) from the binary dataset in the prompt.

    if prompt.len() < 4 {
        eprintln!("multiple_choice_score: no tasks");
        return;
    }
    let n_task = u32::from_le_bytes(prompt[0..4].try_into().unwrap());
    let mut pos = 4usize;
    if n_task == 0 {
        eprintln!("multiple_choice_score: no tasks");
        return;
    }
    eprintln!("multiple_choice_score: there are {n_task} tasks in prompt");
    if pos + n_task as usize * 4 > prompt.len() {
        eprintln!("multiple_choice_score: failed to read task positions from prompt");
        return;
    }
    let task_pos: Vec<u32> = (0..n_task as usize)
        .map(|i| u32::from_le_bytes(prompt[4 + i * 4..8 + i * 4].try_into().unwrap()))
        .collect();
    // sequential reads continue right after the positions array
    pos = 4 + n_task as usize * 4;

    let mut n_task = n_task as usize;
    let mut tasks: Vec<MultipleChoiceTask> = Vec::new();
    if multiple_choice_tasks == 0 || multiple_choice_tasks >= n_task {
        // Use all tasks
        eprint!("multiple_choice_score: reading tasks");
        let _ = std::io::stderr().flush();
        let n_dot = std::cmp::max(n_task / 100, 1);
        for i in 0..n_task {
            match MultipleChoiceTask::deserialize(prompt, &mut pos) {
                Some(t) => tasks.push(t),
                None => {
                    eprintln!();
                    eprintln!("multiple_choice_score: failed to read task {} of {n_task}", i + 1);
                    return;
                }
            }
            if (i + 1) % n_dot == 0 {
                print!("."); // LOG(".") → stdout
                let _ = std::io::stdout().flush();
            }
        }
        println!("done");
    } else {
        eprintln!(
            "multiple_choice_score: selecting {multiple_choice_tasks} random tasks from {n_task} tasks available"
        );
        let mut rng = Mt19937::new(1);
        let mut aux: Vec<usize> = (0..n_task).collect();
        for _ in 0..multiple_choice_tasks {
            let j = scale_draw_index(&mut rng, aux.len());
            let idx = aux[j];
            aux[j] = *aux.last().unwrap();
            aux.pop();
            let mut p = task_pos[idx] as usize;
            match MultipleChoiceTask::deserialize(prompt, &mut p) {
                Some(t) => tasks.push(t),
                None => {
                    eprintln!(
                        "multiple_choice_score: failed to read task {idx} at position {}",
                        task_pos[idx]
                    );
                    return;
                }
            }
        }
        n_task = multiple_choice_tasks;
    }

    eprint!("multiple_choice_score: preparing task data");
    let _ = std::io::stderr().flush();
    let n_dot = std::cmp::max(tasks.len() / 100, 1);
    for (i, task) in tasks.iter_mut().enumerate() {
        if !multiple_choice_prepare_one_task(vocab, task, true) {
            return;
        }
        if (i + 1) % n_dot == 0 {
            print!("."); // LOG(".") → stdout
            let _ = std::io::stdout().flush();
        }
    }
    println!("done");

    eprintln!(
        "multiple_choice_score : calculating TruthfulQA score over {} tasks.",
        tasks.len()
    );

    println!("\ntask\tacc_norm");

    let n_ctx = params.n_ctx;
    let n_vocab = dctx.n_vocab();

    let max_tasks_per_batch = 32usize;
    let max_seq = (4 * max_tasks_per_batch).min(params.n_seq_max).min(64);

    let mut batch = LlamaBatch::default();
    let mut tok_logits: Vec<f32>;
    let mut batch_logits: Vec<f32> = Vec::with_capacity(n_ctx * n_vocab);
    let mut eval_pairs: Vec<(usize, i32)> = Vec::new();
    let mut eval_results: Vec<f32> = Vec::new();
    let mut batch_indeces: Vec<i32> = Vec::new();

    let mut n_done = 0i32;
    let mut n_correct = 0i32;
    let mut n_tot_answers = 0usize;

    let mut i0 = 0usize;
    while i0 < tasks.len() {
        let mut n_cur = 0usize;

        let mut i1 = i0;
        let mut i_logits = 0usize;

        batch.clear();

        // batch as much tasks as possible into the available context
        let mut s0 = 0i32;
        while n_cur + tasks[i1].required_tokens <= n_ctx {
            let num_answers = tasks[i1].seq_tokens.len();
            if s0 as usize + num_answers > max_seq {
                if s0 == 0 {
                    eprintln!(
                        "multiple_choice_score : task {i0} requires a higher -np|--parallel value (at least {num_answers})"
                    );
                    return;
                }
                break;
            }

            if batch_indeces.len() != num_answers {
                batch_indeces.resize(num_answers, 0);
            }
            for (s, b) in batch_indeces.iter_mut().enumerate() {
                *b = s0 + s as i32;
            }

            let mut n_logits = 0usize;
            for i in 0..tasks[i1].common_prefix {
                batch.add(tasks[i1].seq_tokens[0][i], i as i32, &batch_indeces, false);
            }
            if let Some(l) = batch.logits.as_mut() {
                let n = l.len();
                l[n - 1] = true; // logits for the last common-prefix token
            }
            n_logits += 1;

            for s in 0..tasks[i1].seq_tokens.len() {
                let seq_tokens_size = tasks[i1].seq_tokens[s].len();
                // TODO(upstream): don't evaluate the last token of each sequence
                for i in tasks[i1].common_prefix..seq_tokens_size {
                    let needs_logits = i < seq_tokens_size - 1;
                    batch.add(tasks[i1].seq_tokens[s][i], i as i32, &[s0 + s as i32], needs_logits);
                    n_logits += needs_logits as usize;
                }
            }

            s0 += num_answers as i32;

            tasks[i1].i_logits = i_logits;
            i_logits += n_logits;

            n_cur += tasks[i1].required_tokens;
            i1 += 1;
            if i1 == tasks.len() {
                break;
            }
        }

        if i0 == i1 {
            eprintln!(
                "multiple_choice_score : task {i0} does not fit in the context window (requires {} tokens)",
                tasks[i0].required_tokens
            );
            return;
        }

        dctx.reset_sequence(); // llama_memory_clear

        // decode all tasks [i0, i1)
        if !decode_helper(dctx, &batch, &mut batch_logits, n_vocab) {
            eprintln!("multiple_choice_score: llama_decode() failed");
            return;
        }

        // Compute log-probs in parallel: first collect all tasks
        eval_pairs.clear();
        for cur_task in &tasks[i0..i1] {
            let mut li = 1; // skip the last logit of the common prefix
            for s in 0..cur_task.seq_tokens.len() {
                for j in cur_task.common_prefix..cur_task.seq_tokens[s].len() - 1 {
                    eval_pairs.push((cur_task.i_logits + li, cur_task.seq_tokens[s][j + 1]));
                    li += 1;
                }
            }
        }
        compute_logprobs(&batch_logits, n_vocab, &eval_pairs, &mut eval_results);

        let mut ir = 0usize;

        // compute the logprobs for each ending of the decoded tasks
        for i in i0..i1 {
            // get the logits of the last token of the common prefix
            tok_logits = batch_logits
                [tasks[i].i_logits * n_vocab..(tasks[i].i_logits + 1) * n_vocab]
                .to_vec();
            let first_probs = softmax(&tok_logits);

            let n_ans = tasks[i].seq_tokens.len();
            tasks[i].log_probs.resize(n_ans, 0.0);
            for s in 0..n_ans {
                let mut count = 1usize;
                let mut log_prob =
                    first_probs[tasks[i].seq_tokens[s][tasks[i].common_prefix] as usize].ln();
                for _j in tasks[i].common_prefix..tasks[i].seq_tokens[s].len() - 1 {
                    count += 1;
                    log_prob += eval_results[ir];
                    ir += 1;
                }
                tasks[i].log_probs[s] = log_prob / count as f32;
            }

            // Find the ending with maximum logprob
            let mut logprob_max_idx = 0usize;
            let mut logprob_max_val = tasks[i].log_probs[0];
            for (s, &v) in tasks[i].log_probs.iter().enumerate().skip(1) {
                if v > logprob_max_val {
                    logprob_max_val = v;
                    logprob_max_idx = s;
                }
            }

            n_tot_answers += tasks[i].log_probs.len();
            if tasks[i].mc1.labels[logprob_max_idx] == 1 {
                n_correct += 1;
            }
            n_done += 1;

            // Print the accumulated accuracy mean x 100
            println!("{n_done}\t{:.8}", 100.0 * n_correct as f64 / n_done as f64);
        }

        i0 = i1 - 1;
        i0 += 1;
    }

    if n_done < 100 && (multiple_choice_tasks != 0 && multiple_choice_tasks < n_task) {
        return;
    }

    let p = (n_correct as f32) / (n_done as f32);
    let sigma = (p * (1.0 - p) / (n_done - 1) as f32).sqrt();
    println!();
    // "%.4f" of the promoted `100.f*p` / `100.f*sigma` (float) values
    eprintln!(
        "Final result: {:.4} +/- {:.4}",
        (100.0f32 * p) as f64,
        (100.0f32 * sigma) as f64
    );
    let p = (n_done as f32) / (n_tot_answers as f32);
    let sigma = (p * (1.0 - p) / (n_done - 1) as f32).sqrt();
    eprintln!(
        "Random chance: {:.4} +/- {:.4}",
        (100.0f32 * p) as f64,
        (100.0f32 * sigma) as f64
    );

    eprintln!();
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_vectors() {
        // canonical MT19937 test vectors (seed 5489: first outputs
        // 3499211612, 581869302, 3890346734, 3586334585, 545404204)
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next_u32(), 3499211612);
        assert_eq!(rng.next_u32(), 581869302);
        assert_eq!(rng.next_u32(), 3890346734);
        assert_eq!(rng.next_u32(), 3586334585);
        assert_eq!(rng.next_u32(), 545404204);
        // seed 1 first outputs — g++ 13 printout (parity/ref_rng_vec)
        let mut rng1 = Mt19937::new(1);
        assert_eq!(rng1.next_u32(), 1791095845);
        assert_eq!(rng1.next_u32(), 4282876139);
        assert_eq!(rng1.next_u32(), 3093770124);
        assert_eq!(rng1.next_u32(), 4005303368);
        assert_eq!(rng1.next_u32(), 491263);
        // crossing the 624-draw block boundary keeps the sequence going
        let mut rngb = Mt19937::new(1);
        for _ in 0..624 {
            rngb.next_u32();
        }
        assert_eq!(rngb.next_u32(), 1104314680);
    }

    #[test]
    fn lemire_matches_libstdcpp_vectors() {
        // libstdc++ 13 Lemire branch: uniform_int_distribution<size_t>(0, 9)
        // over mt19937(1) — g++ 13 printout (parity/ref_rng_vec):
        // 4 9 7 9 0 1 3 9 1 2
        let mut rng = Mt19937::new(1);
        let draws: Vec<usize> = (0..10).map(|_| uniform_int_size_t(&mut rng, 0, 9)).collect();
        assert_eq!(draws, vec![4, 9, 7, 9, 0, 1, 3, 9, 1, 2]);

        // dist(0, 59) — the selection sequence for a 60-task HellaSwag file
        let mut rng3 = Mt19937::new(1);
        let draws60: Vec<usize> = (0..8).map(|_| uniform_int_size_t(&mut rng3, 0, 59)).collect();
        assert_eq!(draws60, vec![25, 59, 43, 55, 0, 7, 18, 59]);

        // ... and dist(0, 0) always yields 0 regardless of the draw
        let mut rng2 = Mt19937::new(7);
        for _ in 0..16 {
            assert_eq!(uniform_int_size_t(&mut rng2, 0, 0), 0);
        }
    }

    #[test]
    fn scale_draw_matches_libstdcpp_vectors() {
        // int(2^-32 * rng() * aux.size()) over mt19937(1), aux.size()=5 —
        // g++ 13 printout (parity/ref_rng_vec): 2 4 3 4 0
        let mut rng = Mt19937::new(1);
        let draws: Vec<usize> = (0..5).map(|_| scale_draw_index(&mut rng, 5)).collect();
        assert_eq!(draws, vec![2, 4, 3, 4, 0]);
    }

    #[test]
    fn softmax_f32_math() {
        let p = softmax(&[1.0f32, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn compute_logprobs_definition() {
        let n_vocab = 4;
        let logits = vec![1.0f32, 2.0, 3.0, 0.5];
        let mut res = Vec::new();
        compute_logprobs(&logits, n_vocab, &[(0, 2)], &mut res);
        // logprob = logit[tok] - max - ln(sum exp(l - max))
        let max = 3.0f32;
        let sum: f32 = [1.0f32, 2.0, 3.0, 0.5].iter().map(|v| (v - max).exp()).sum();
        assert!((res[0] - (3.0 - max - sum.ln())).abs() < 1e-6);
    }

    #[test]
    fn winogrande_csv_parsing() {
        let csv = "0,Sarah was better than Maria so _ got the cases.,Sarah,Maria,2\n\
                   1,\"Quote with , inside and _ here\",a,b,1\n";
        let e = load_winogrande_from_csv(csv);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].first, "Sarah was better than Maria so ");
        assert_eq!(e[0].second, " got the cases.");
        assert_eq!(e[0].choices, ["Sarah", "Maria"]);
        assert_eq!(e[0].answer, 2);
        // quoted sentence field: quotes stripped, embedded comma kept
        assert_eq!(e[1].first, "Quote with , inside and ");
        assert_eq!(e[1].second, " here");
        assert_eq!(e[1].choices, ["a", "b"]);
        assert_eq!(e[1].answer, 1);

        // the getline fail||eof idiom drops a final unterminated line
        let e2 = load_winogrande_from_csv("0,a _ b,c,d,1");
        assert!(e2.is_empty());
        // malformed lines are skipped with an error
        let e3 = load_winogrande_from_csv("no commas here\n1,a _ b,c,d,1\n");
        assert_eq!(e3.len(), 1);
        // no underscore / bad answer
        let e4 = load_winogrande_from_csv("2,no blank,c,d,1\n3,x _ y,c,d,3\n");
        assert_eq!(e4.len(), 0);
    }

    #[test]
    fn mc_dataset_serialization_roundtrip() {
        let mut t0 = Vec::new();
        serialize_mc_task(&mut t0, "What is 2+2?", &["3", "4", "5"], 1);
        let mut t1 = Vec::new();
        serialize_mc_task(&mut t1, "Sky color?", &["green", "blue"], 1);
        let mut data = Vec::new();
        data.extend_from_slice(&2u32.to_le_bytes());
        // task positions point at each task record (4 + n*4 header)
        data.extend_from_slice(&((4 + 2 * 4) as u32).to_le_bytes());
        data.extend_from_slice(&((4 + 2 * 4 + t0.len()) as u32).to_le_bytes());
        data.extend_from_slice(&t0);
        data.extend_from_slice(&t1);

        let n = u32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(n, 2);
        let mut p = 4 + 8;
        let a = MultipleChoiceTask::deserialize(&data, &mut p).unwrap();
        assert_eq!(a.question, "What is 2+2?");
        assert_eq!(a.mc1.answers, vec!["3", "4", "5"]);
        assert_eq!(a.mc1.labels, vec![0, 1, 0]);
        let b = MultipleChoiceTask::deserialize(&data, &mut p).unwrap();
        assert_eq!(b.question, "Sky color?");
        assert_eq!(b.mc1.answers.len(), 2);
        assert_eq!(b.mc1.labels, vec![0, 1]);
    }

    #[test]
    fn getline_variants() {
        assert_eq!(getline_all("a\nb"), vec!["a", "b"]);
        assert_eq!(getline_all("a\nb\n"), vec!["a", "b"]);
        assert_eq!(getline_all("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(getline_all(""), Vec::<&str>::new());
        assert_eq!(getline_all("\n"), vec![""]);
    }
}
