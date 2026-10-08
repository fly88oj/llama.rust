/* Pinned kernel-to-kernel microbench: ggml_vec_dot_q6_K_q8_K (the AVX2 body
 * arch/x86/quants.c:2426 runs on this host) called directly, same protocol as
 * the port's simd_x86::tests::kernel_throughput (L1/L2-resident rows, warm-up,
 * best-of-N). Build:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -x c++ parity/ref_vecdot_q6k_bench.c -I$PINNED/ggml/include \
 *       -o /tmp/perf-ref_vecdot_q6k -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: /tmp/perf-ref_vecdot_q6k [n] [rows] [reps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"

// internal to ggml-cpu (ggml/src/ggml-cpu/quants.h:56); declared here to
// avoid pulling the private header chain
extern "C" void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
// quants.h:32 — Q8_K is not a public quantize type (ggml_quantize_chunk
// asserts); the internal row quantizer is what mul_mat's wdata stage runs
extern "C" void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint32_t st0 = 0x1234567u;
static uint32_t lcg(void) { st0 = st0 * 1664525u + 1013904223u; return st0; }
static float lcg_f(void) { return ((int32_t)lcg()) / (float)(1u << 28) * 1.8f - 0.9f; }

int main(int argc, char ** argv) {
    const int n    = argc > 1 ? atoi(argv[1]) : 4864; // ffn_down row length
    const int rows = argc > 2 ? atoi(argv[2]) : 8192;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    const size_t bx_size = ggml_row_size(GGML_TYPE_Q6_K, n);
    const size_t by_size = ggml_row_size(GGML_TYPE_Q8_K, n);

    // 64 distinct x rows so the bench streams more than a single cache line
    // set, but stays cache-resident (64 * (210+292)*n/256 bytes ~= 2-6 MB for
    // n=4864 -> L2/L3, matching the port's kernel_throughput footprint).
    const int nrows_x = 64;
    void ** xs = (void **) malloc(sizeof(void *) * nrows_x);
    void ** ys = (void **) malloc(sizeof(void *) * 64);
    float * tmp = (float *) malloc(sizeof(float) * n);
    for (int i = 0; i < nrows_x; i++) {
        xs[i] = (void *) malloc(bx_size);
        for (int j = 0; j < n; j++) tmp[j] = lcg_f();
        ggml_quantize_chunk(GGML_TYPE_Q6_K, tmp, xs[i], 0, 1, n, NULL);
    }
    for (int i = 0; i < 64; i++) {
        ys[i] = (void *) malloc(by_size);
        for (int j = 0; j < n; j++) tmp[j] = lcg_f();
        quantize_row_q8_K(tmp, ys[i], n);
    }

    float s = 0.f;
    for (int i = 0; i < 512; i++) { // warm-up
        ggml_vec_dot_q6_K_q8_K(n, &s, 0, xs[i % nrows_x], 0, ys[i % 64], 0, 1);
    }
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        float acc = 0.f;
        for (int i = 0; i < rows; i++) {
            ggml_vec_dot_q6_K_q8_K(n, &s, 0, xs[i % nrows_x], 0, ys[i % 64], 0, 1);
            acc += s;
        }
        if (acc == 12345.678f) printf("x");
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    printf("q6_K x q8_K  n=%d rows=%d  %7.1f ns/row  %6.2f G el/s  (best of %d)\n",
           n, rows, best / rows * 1e9, rows * (double)n / best / 1e9, reps);
    return 0;
}
