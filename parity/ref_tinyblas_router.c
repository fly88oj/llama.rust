/* ref_tinyblas_router.c — ground truth for the one shape the tinyBLAS dump
 * (parity/ref_tinyblas_dump.c, k <= 1024) does not cover and that the gpt-oss
 * graph actually runs: the MoE router GEMM
 *
 *   m = 32 experts, n = 5 activation columns, k = 2880
 *   (mul_mat(ffn_gate_inp [2880 x 32], cur [2880 x 5]) -> ggml-cpu.c:1304)
 *
 * Output, in order:
 *   [u32 nth=1][f32 C[m*n]]  — llamafile_sgemm with a 1-thread pool
 *   [u32 nth=8][f32 C[m*n]]  — the same call with an 8-thread pool
 * plus the inputs the Rust side must feed its own kernel:
 *   parity/tinyblas_router_ab.bin = [f32 A[m*k]][f32 B[n*k]]
 *
 * Build (from the repo root):
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -std=c11 -march=native parity/ref_tinyblas_router.c \
 *       -I$PINNED/ggml/include -I$PINNED/ggml/src -I$PINNED/ggml/src/ggml-cpu \
 *       -o parity/ref_tinyblas_router -L$REF -lggml-base -lggml-cpu -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_tinyblas_router
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-threading.h"
#include "llamafile/sgemm.h"

static uint32_t lcg_state = 0x12345678u;
static float next_val(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float) (int32_t) lcg_state / (float) (1 << 30)) * 0.75f - 0.125f;
}

#define M 32
#define N 5
#define K 2880

/* one llamafile_sgemm call with `nth` threads, exactly as ggml-cpu.c:1304 does */
static int call(int nth, const float * a, const float * b, float * c) {
    struct ggml_compute_params params;
    memset(&params, 0, sizeof(params));
    params.ith        = 0;
    params.nth        = nth;
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
    params.threadpool = ggml_threadpool_new(&tpp);
    if (!params.threadpool) {
        fprintf(stderr, "threadpool\n");
        return 0;
    }
    const int ret = llamafile_sgemm(&params, M, N, K, a, K, b, K, c, M,
                                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32) ? 1 : 0;
    ggml_threadpool_free(params.threadpool);
    return ret;
}

/* the same GEMM through the *graph* path (ggml_compute_forward_mul_mat ->
 * llamafile_sgemm), i.e. what the reference actually runs inside a model. This
 * is the only way to see the threaded branch: a direct call with an undriven
 * threadpool runs job 0 only (see the header of this file). */
static int graph_call(int nth, const float * a, const float * b, float * c) {
    struct ggml_init_params ip = { .mem_size = 64u * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * td = ggml_mul_mat(ctx, ta, tb);
    memcpy(ta->data, a, (size_t) M * K * sizeof(float));
    memcpy(tb->data, b, (size_t) N * K * sizeof(float));
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, td);
    struct ggml_init_params ip2 = { .mem_size = 64u * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctxc = ggml_init(ip2);
    ggml_graph_compute_with_ctx(ctxc, gf, nth);
    memcpy(c, td->data, (size_t) M * N * sizeof(float));
    ggml_free(ctxc);
    ggml_free(ctx);
    return 1;
}

int main(int argc, char ** argv) {
    const char * out = "parity/tinyblas_router_ref.bin";
    const char * ab  = "parity/tinyblas_router_ab.bin";
    if (argc > 1) out = argv[1];
    if (argc > 2) ab  = argv[2];

    float * a = malloc((size_t) M * K * sizeof(float));
    float * b = malloc((size_t) N * K * sizeof(float));
    float * c1 = calloc((size_t) M * N, sizeof(float));
    float * c8 = calloc((size_t) M * N, sizeof(float));
    if (a == NULL || b == NULL || c1 == NULL || c8 == NULL) {
        fprintf(stderr, "out of memory\n");
        free(a); free(b); free(c1); free(c8);
        return 1;
    }
    for (size_t i = 0; i < (size_t) M * K; i++) a[i] = next_val();
    for (size_t i = 0; i < (size_t) N * K; i++) b[i] = next_val();

    const int r1 = call(1, a, b, c1);
    const int r8 = call(8, a, b, c8);
    float * cg = calloc((size_t) M * N, sizeof(float));
    if (cg == NULL) { fprintf(stderr, "out of memory\n"); free(a); free(b); free(c1); free(c8); free(cg); return 1; }
    graph_call(8, a, b, cg);
    const int same = memcmp(c1, cg, (size_t) M * N * sizeof(float)) == 0;

    FILE * f = fopen(out, "wb");
    if (f == NULL) {
        perror("fopen out"); free(a); free(b); free(c1); free(c8); free(cg); return 1;
    }
    uint32_t nth = 1;
    const size_t nc = (size_t) M * N;
    if (fwrite(&nth, 4, 1, f) != 1 || fwrite(c1, sizeof(float), nc, f) != nc) {
        fprintf(stderr, "short write\n"); fclose(f); return 1;
    }
    nth = 8;
    if (fwrite(&nth, 4, 1, f) != 1 || fwrite(cg, sizeof(float), nc, f) != nc) {
        fprintf(stderr, "short write\n"); fclose(f); return 1;
    }
    fclose(f);

    FILE * g = fopen(ab, "wb");
    if (g == NULL) {
        perror("fopen ab"); free(a); free(b); free(c1); free(c8); free(cg); return 1;
    }
    const size_t na = (size_t) M * K, nb = (size_t) N * K;
    if (fwrite(a, sizeof(float), na, g) != na || fwrite(b, sizeof(float), nb, g) != nb) {
        fprintf(stderr, "short write\n"); fclose(g); return 1;
    }
    fclose(g);

    printf("llamafile_sgemm(F32,F32,m=%d,n=%d,k=%d): direct nth=1 -> %d, graph nth=8 -> %d, ",
           M, N, K, r1, r8);
    if (same) {
        printf("direct nth=1 == graph nth=8\n");
    } else {
        printf("direct nth=1 != graph nth=8\n");
        for (int i = 0; i < M * N; i++) {
            if (c1[i] != c8[i]) {
                printf("  first differing element C[%d]: direct=%.9g graph=%.9g\n",
                       i, (double) c1[i], (double) cg[i]);
                break;
            }
        }
    }
    printf("wrote %s and %s\n", out, ab);
    free(a); free(b); free(c1); free(c8); free(cg);
    return 0;
}