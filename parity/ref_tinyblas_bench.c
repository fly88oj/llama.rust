/* Timing probe: llamafile_sgemm on the qwen2.5-0.5b hot prefill shapes, to
 * compare the reference's tinyBLAS_Q0_AVX throughput with the port's.
 * Single-threaded (ith=0, nth=1): a direct call on an undriven pool executes
 * only the caller's share, so nth=1 is the only complete configuration — the
 * kernel throughput comparison is per-core anyway.
 *
 * Build:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -x c++ parity/ref_tinyblas_bench.c -I$PINNED/ggml/include \
 *       -o parity/ref_tinyblas_bench -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_tinyblas_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
bool llamafile_sgemm(const void * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint32_t st = 0x1234567u;
static uint32_t lcg(void) {
    st = st * 1664525u + 1013904223u;
    return st;
}

int main(void) {
    // shapes from the qwen2.5-0.5b Q4_K_M pp64/tg16 profile:
    //   (type, m=weight rows, n=activation rows, k=blocks of 32)
    struct shape { enum ggml_type ty; int m, n, k; const char *name; } shapes[] = {
        { GGML_TYPE_Q5_0, 4864, 64, 28, "ffn_gate/up 4864x896 x64" },
        { GGML_TYPE_Q8_0, 151936, 64, 28, "lm_head 151936x896 x64" },
        { GGML_TYPE_Q5_0, 896, 64, 28, "attn_q/o 896x896 x64" },
        { GGML_TYPE_Q5_0, 4864, 1, 28, "ffn gemv 4864x896 x1" },
        { GGML_TYPE_Q8_0, 151936, 1, 28, "lm_head gemv x1" },
    };

    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        struct shape * sh = &shapes[s];
        const int bs = ggml_type_size(sh->ty);
        const int blck = 32;
        const size_t ab = (size_t) sh->m * sh->k * bs;
        const size_t bb = (size_t) sh->n * sh->k * ggml_type_size(GGML_TYPE_Q8_0);
        uint8_t * ap = (uint8_t *) malloc(ab);
        uint8_t * bp = (uint8_t *) malloc(bb);
        for (size_t i = 0; i < ab; i++) ap[i] = (uint8_t) lcg();
        for (size_t i = 0; i < bb; i++) bp[i] = (uint8_t) lcg();
        float * C = (float *) malloc(sizeof(float) * (size_t) sh->m * sh->n);

        // ggml_compute_params {int ith; int nth; size_t wsize; void* wdata;
        // threadpool*} (ggml-cpu-impl.h:18): ith=0/nth=1, null wdata
        long long params[3] = { 0, 0, 0 };
        ((int *) params)[1] = 1; // nth

        const int reps = sh->n >= 64 ? 5 : 50;
        llamafile_sgemm(params, sh->m, sh->n, sh->k, ap, sh->k,
                        bp, sh->k, C, sh->m,
                        sh->ty, GGML_TYPE_Q8_0, GGML_TYPE_F32);
        double sum = 0;
        for (int r = 0; r < reps; r++) {
            double s0 = now_ms();
            llamafile_sgemm(params, sh->m, sh->n, sh->k, ap, sh->k,
                            bp, sh->k, C, sh->m,
                            sh->ty, GGML_TYPE_Q8_0, GGML_TYPE_F32);
            sum += now_ms() - s0;
        }
        const double gf = 2.0 * sh->m * sh->n * sh->k * blck / 1e9;
        printf("%-28s m=%7d n=%2d k=%2d : %8.3f ms/call  %7.1f GFLOP/s  C[0]=%.4f\n",
               sh->name, sh->m, sh->n, sh->k, sum / reps, gf / (sum / reps) * 1e3, C[0]);

        free(C); free(ap); free(bp);
    }
    return 0;
}
