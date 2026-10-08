/* ref_tinyblas_bench8.c — 8-thread twin of parity/ref_tinyblas_bench.c.
 *
 * The 1-thread harness can only drive ith=0/nth=1 (an undriven pool executes
 * nothing on the other shards), so it cannot answer the question this round
 * needs: how does the *reference's* tinyBLAS_Q0_AVX scale across the 8-thread
 * duty split (gemm4xN's `duty = (tiles + nth - 1) / nth` job walk,
 * sgemm.cpp:1530-1536)? This harness spawns 8 pthreads, each calling the
 * exported `llamafile_sgemm` with its own ith/nth over the same A/B/C —
 * exactly what a ggml threadpool does inside ggml_compute_forward_mul_mat
 * (ggml-cpu.c:1306/:1389, one call per thread per broadcast plane).
 *
 * Protocol matches the Rust side's bench_tests::bench_tinyblas_shapes 8t
 * column: warm-up + best-of-5, one barrier join per rep (pthread_join of the
 * 8 workers = the region's team barrier).
 *
 * Build:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -x c++ parity/ref_tinyblas_bench8.c -I$PINNED/ggml/include \
 *       -o parity/ref_tinyblas_bench8 -L$REF -lggml -lggml-cpu -lggml-base \
 *       -lm -lpthread -Wl,-rpath,$REF
 * Run: ./parity/ref_tinyblas_bench8 [threads]
 */
#include <pthread.h>
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

static int g_m, g_n, g_k, g_nth;
static const uint8_t *g_a, *g_b;
static float *g_c;

struct worker_args { int ith; };

static void *worker(void *varg) {
    struct worker_args *wa = (struct worker_args *) varg;
    /* ggml_compute_params {int ith; int nth; size_t wsize; void* wdata;
     * threadpool*} (ggml-cpu-impl.h:18): only ith/nth are read by sgemm */
    long long params[3] = { 0, 0, 0 };
    ((int *) params)[0] = wa->ith;
    ((int *) params)[1] = g_nth;
    llamafile_sgemm(params, g_m, g_n, g_k, g_a, g_k,
                    g_b, g_k, g_c, g_m,
                    GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_F32);
    return NULL;
}

/* one rep: spawn nth workers, join them all (the region barrier) */
static double rep_once(void) {
    pthread_t th[64];
    struct worker_args wa[64];
    double t0 = now_ms();
    for (int i = 0; i < g_nth; i++) {
        wa[i].ith = i;
        pthread_create(&th[i], NULL, worker, &wa[i]);
    }
    for (int i = 0; i < g_nth; i++)
        pthread_join(th[i], NULL);
    return now_ms() - t0;
}

int main(int argc, char **argv) {
    g_nth = argc > 1 ? atoi(argv[1]) : 8;
    /* shapes from the qwen2.5-0.5b Q4_K_M pp64 profile */
    struct shape { int m, n, k; const char *name; } shapes[] = {
        { 4864, 64, 28, "ffn_gate/up 4864x896 x64" },
        { 896, 64, 28, "attn_q/o 896x896 x64" },
    };
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        struct shape * sh = &shapes[s];
        g_m = sh->m; g_n = sh->n; g_k = sh->k;
        const size_t ab = (size_t) sh->m * sh->k * 22; /* q5_0 block */
        const size_t bb = (size_t) sh->n * sh->k * 34; /* q8_0 block */
        uint8_t *ap = (uint8_t *) malloc(ab), *bp = (uint8_t *) malloc(bb);
        for (size_t i = 0; i < ab; i++) ap[i] = (uint8_t) lcg();
        for (size_t i = 0; i < bb; i++) bp[i] = (uint8_t) lcg();
        float *C = (float *) malloc(sizeof(float) * (size_t) sh->m * sh->n);
        g_a = ap; g_b = bp; g_c = C;

        rep_once(); /* warm-up */
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double ms = rep_once();
            if (ms < best) best = ms;
        }
        const double gf = 2.0 * sh->m * sh->n * sh->k * 32 / 1e9;
        printf("Q5_0 %-24s m=%7d n=%2d k=%2d nth=%d : %8.3f ms/call  %7.1f GFLOP/s  C[0]=%.4f\n",
               sh->name, sh->m, sh->n, sh->k, g_nth, best, gf / best * 1e3, C[0]);
        free(C); free(ap); free(bp);
    }
    return 0;
}
