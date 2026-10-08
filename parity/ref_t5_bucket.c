/* T5 relative-position-bucket ground truth.
 *
 * `llama_relative_position_bucket` is verbatim from the pinned reference
 * (src/llama-graph.cpp:3890-3923, bd4f514db1) — copied rather than linked
 * because the function lives inside llama-graph.cpp and is not exported by
 * libllama.so. The point of this program is to pin the *C arithmetic types*
 * (float `logf` times a uint64 rounding to f32, double `log` divisor) so the
 * Rust transcription in crates/llama/src/graph_arch.rs can be diffed against
 * real C behaviour, not against a re-reading of the source.
 *
 * Build + run (from the repo root):
 *   gcc -O2 -o parity/ref_t5_bucket parity/ref_t5_bucket.c -lm && \
 *       ./parity/ref_t5_bucket > parity/t5_bucket_ref.txt
 *
 * Output: one line per (bidirectional, x, y) with the bucket index, for the
 * (n_buckets, x/y range) that the local t5-v1_1-xxl-encoder file exercises
 * (`attention.relative_buckets_count = 32`) plus the unidirectional variant.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ==== verbatim from src/llama-graph.cpp:3890-3923 ==== */
/* (std::abs / std::min<int32_t> spelled out: this file must compile as C) */
#define ABS_I32(a)  ((a) < 0 ? -(a) : (a))
#define MIN_I32(a, b) ((a) < (b) ? (a) : (b))

int32_t llama_relative_position_bucket(int32_t x, int32_t y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = ABS_I32(relative_position);
    } else {
        relative_position = -MIN_I32(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = MIN_I32(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
/* ==== end verbatim ==== */

int main(void) {
    const uint64_t n_buckets = 32; /* t5 attention.relative_buckets_count */

    /* bidirectional (encoder path, llama-graph.cpp:187) over the 0..15 x 0..15
     * grid plus a few far-apart pairs that land in the log-spaced buckets */
    const int pts[] = { 0, 1, 2, 3, 7, 8, 15, 16, 33, 65, 130, 200 };
    const int npts = (int) (sizeof(pts) / sizeof(pts[0]));

    for (int a = 0; a < npts; ++a) {
        for (int b = 0; b < npts; ++b) {
            const int32_t x = pts[a];
            const int32_t y = pts[b];
            printf("bi %4d %4d %4d\n", x, y,
                   llama_relative_position_bucket(x, y, n_buckets, true));
        }
    }
    /* unidirectional (decoder path, llama-kv-cache.cpp:1809) over 0..16 */
    for (int32_t x = 0; x <= 16; ++x) {
        for (int32_t y = 0; y <= 16; ++y) {
            printf("uni %4d %4d %4d\n", x, y,
                   llama_relative_position_bucket(x, y, n_buckets, false));
        }
    }
    return 0;
}