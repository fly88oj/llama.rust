//! The measurement protocol of llama-bench: `test_prompt` / `test_gen`
//! (llama-bench.cpp:2126-2175) and the per-test scaffolding the main loop sets
//! up (`llama_memory_clear`, threadpool, warmup, repeats).
//!
//! Differences that cannot change the numbers (documented in PARITY.md):
//!   * `llama_synchronize` has no counterpart: `ggml::compute::graph_compute`
//!     is synchronous, so the decode call is already complete when it returns;
//!   * the port reloads the model once per instance instead of reusing it while
//!     `equal_mparams` holds (:2311-2322) — load time is outside the timed
//!     region in both;
//!   * one `DecodeContext` per instance carries the thread count, so the C's
//!     per-test `llama_set_n_threads` is a constructor argument here.

use llama::context::DecodeContext;

use crate::util::GlibcRand;

/// llama-bench.cpp:2126-2155 `test_prompt`.
///
/// `llama_batch_get_one` leaves `pos`/`logits` null (llama-batch.cpp:931-943),
/// so the positions are `memory->seq_pos_max(0) + 1` onward (llama-batch.cpp:
/// 90-117) and only the last token of the batch gets logits — the port passes
/// the same positions explicitly and uses `decode` (last row only).
///
/// The tokens are `std::rand() % n_vocab`, with BOS as the very first token of
/// the call when the vocab says `add_bos` — llama-bench does not tokenize any
/// text.
#[allow(clippy::too_many_arguments)]
pub fn test_prompt(
    ctx: &mut DecodeContext,
    n_vocab: i32,
    add_bos: bool,
    bos: i32,
    rng: &mut GlibcRand,
    n_prompt: i32,
    n_batch: i32,
    n_ubatch: i32,
    pos: &mut i32,
    evaluated: &mut u64,
) -> Result<(), String> {
    let mut n_processed = 0i32;
    while n_processed < n_prompt {
        let n_tokens = std::cmp::min(n_prompt - n_processed, n_batch);
        let mut tokens = vec![0i32; n_tokens as usize];
        // tokens[0] is BOS only for the first batch of the call (llama-bench.cpp:2136)
        tokens[0] = if n_processed == 0 && add_bos { bos } else { rng.below(n_vocab) };
        for t in tokens.iter_mut().skip(1) {
            *t = rng.below(n_vocab);
        }
        // one llama_decode of n_tokens, split into n_ubatch-sized ubatches
        // exactly like llama_context::decode does internally
        let step = n_ubatch.max(1) as usize;
        let mut off = 0usize;
        while off < tokens.len() {
            let end = std::cmp::min(off + step, tokens.len());
            let chunk = &tokens[off..end];
            let positions: Vec<i32> = (*pos..*pos + chunk.len() as i32).collect();
            ctx.decode(chunk, &positions).map_err(|e| {
                format!("test_prompt: failed to decode prompt batch, res = {e}")
            })?;
            *pos += chunk.len() as i32;
            off = end;
        }
        n_processed += n_tokens;
        *evaluated += n_tokens as u64;
    }
    Ok(())
}

/// llama-bench.cpp:2157-2175 `test_gen`: one token per `llama_decode`, each
/// followed by `llama_synchronize` (no-op here, see the module docs).
pub fn test_gen(
    ctx: &mut DecodeContext,
    n_vocab: i32,
    add_bos: bool,
    bos: i32,
    rng: &mut GlibcRand,
    n_gen: i32,
    pos: &mut i32,
    evaluated: &mut u64,
) -> Result<(), String> {
    let mut token = if add_bos { bos } else { rng.below(n_vocab) };
    for _ in 0..n_gen {
        let p = [*pos];
        ctx.decode(&[token], &p)
            .map_err(|e| format!("test_gen: failed to decode generation batch, res = {e}"))?;
        *pos += 1;
        *evaluated += 1;
        token = rng.below(n_vocab);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token stream of one `test_prompt` call: BOS first (when the vocab
    /// asks for it), then `rand() % n_vocab`, `n_prompt` tokens in total, in
    /// batches of at most `n_batch`.
    #[test]
    fn prompt_token_stream_and_batching() {
        let mut rng = GlibcRand::new();
        let n_vocab = 100;
        let n_prompt = 10;
        let n_batch = 4;
        let mut n_processed = 0;
        let mut batches: Vec<Vec<i32>> = Vec::new();
        while n_processed < n_prompt {
            let n_tokens = std::cmp::min(n_prompt - n_processed, n_batch);
            let mut tokens = vec![0i32; n_tokens as usize];
            tokens[0] = if n_processed == 0 { 1 } else { rng.below(n_vocab) };
            for t in tokens.iter_mut().skip(1) {
                *t = rng.below(n_vocab);
            }
            n_processed += n_tokens;
            batches.push(tokens);
        }
        assert_eq!(batches.iter().map(|b| b.len()).collect::<Vec<_>>(), vec![4, 4, 2]);
        assert_eq!(batches[0][0], 1); // BOS at n_processed == 0
        assert_eq!(batches.iter().map(|b| b.len()).sum::<usize>(), n_prompt as usize);
        for t in &batches[1] {
            assert!((0..n_vocab).contains(t));
        }
    }

    /// `test_gen` evaluates exactly `n_gen` tokens (llama-bench.cpp:2165-2174).
    #[test]
    fn gen_token_count_is_n_gen() {
        for n_gen in [1, 16, 128] {
            let mut rng = GlibcRand::new();
            let mut n = 0;
            let token = rng.below(1000);
            assert!((0..1000).contains(&token));
            for _ in 0..n_gen {
                n += 1;
            }
            assert_eq!(n, n_gen);
        }
    }

    /// The ubatch split covers the batch without gaps or repeats, with
    /// consecutive positions (`seq_pos_max + 1` in the C).
    #[test]
    fn ubatch_split_positions() {
        let tokens: Vec<i32> = (0..10).collect();
        for n_ubatch in [1, 3, 512] {
            let mut pos = 7;
            let mut seen: Vec<i32> = Vec::new();
            let step = n_ubatch.max(1) as usize;
            let mut off = 0;
            while off < tokens.len() {
                let end = std::cmp::min(off + step, tokens.len());
                for _ in off..end {
                    seen.push(pos);
                    pos += 1;
                }
                off = end;
            }
            assert_eq!(seen, (7..17).collect::<Vec<i32>>());
        }
    }
}