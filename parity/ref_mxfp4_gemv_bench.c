/* ref_mxfp4_gemv_bench.c — direct timing probe for the reference's MXFP4
 * repack gemv (ggml_gemv_mxfp4_8x8_q8_0, arch/x86/repack.cpp:1700), the
 * kernel that dominates gpt-oss pp64 (MulMatId = ~85% of the hot forward).
 *
 * The gpt-oss shapes: n = 2880, one activation row (nr == 1 — every call site
 * passes 1), nc = 2880 weight rows = 360 8-row tiles per call. The forward
 * issues one call per (expert, token) pair.
 *
 * Build:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -x c++ parity/ref_mxfp4_gemv_bench.c -I$PINNED/ggml/include \
 *       -o parity/ref_mxfp4_gemv_bench -L$REF -lggml -lggml-cpu -lggml-base \
 *       -lm -Wl,-rpath,$REF
 * Run: ./parity/ref_mxfp4_gemv_bench
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx,
                             const void * vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx,
                             const void * vy, int nr, int nc);
/* same contract as the port's quantize_mat_q8_0_4x8: k elements -> 4 rows of
 * interleaved block_q8_0x4 (repack.h:146) */
void ggml_quantize_mat_q8_0_4x8(const float * x, void * vy, int64_t k);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static uint32_t st = 0xabcdefu;
static uint32_t lcg(void) {
    st = st * 1664525u + 1013904223u;
    return st;
}

int main(int argc, char **argv) {
    /* argv[1] == "clean": sanitize the tile E8M0 scales to 120..135 so no
     * col*row product can underflow to a denormal (the FP-assist regime the
     * gpt-oss weights never hit: their block scales sit around 2^-4). The
     * unsanitized mode keeps the original fully-random bytes. */
    int clean = argc > 1 && !strcmp(argv[1], "clean");
    const int n = 2880, nc = 2880, nb = n / 32;
    /* block_mxfp4x8 = 8 e8m0 scales + 128 qs bytes = 136 */
    const size_t vxsz = (size_t)(nc / 8) * nb * 136;
    const size_t vysz = (size_t)nb * 34; /* block_q8_0 */
    uint8_t *vx = (uint8_t *) malloc(vxsz);
    uint8_t *vy = (uint8_t *) malloc(vysz);
    float *s = (float *) malloc(sizeof(float) * nc);
    for (size_t i = 0; i < vxsz; i++) vx[i] = (uint8_t) lcg();
    for (size_t i = 0; i < vysz; i++) vy[i] = (uint8_t) lcg();
    if (clean) {
        for (size_t b = 0; b < vxsz; b += 136)
            for (int j = 0; j < 8; j++) vx[b + j] = 120 + (vx[b + j] % 16);
    }

    /* ---- gemv nr=1 on random q8 blocks (as before) ---- */
    ggml_gemv_mxfp4_8x8_q8_0(n, s, 0, vx, vy, 1, nc);
    double best = 1e18;
    for (int r = 0; r < 200; r++) {
        double t0 = now_us();
        ggml_gemv_mxfp4_8x8_q8_0(n, s, 0, vx, vy, 1, nc);
        double dt = now_us() - t0;
        if (dt < best) best = dt;
    }
    const double gf = 2.0 * n * nc / 1e9;
    printf("ref gemv  n=%d nc=%d nr=1 %s: %8.2f us/call  %7.1f GF/s  s[0]=%f\n",
           n, nc, clean ? "CLEAN" : "rand", best, gf / (best / 1e6), s[0]);

    /* ---- gemm nr=128: activations quantized through the reference's own
     * ggml_quantize_mat_q8_0_4x8 (well-formed f16 d), same weight buffer ---- */
    const int nr = 128;
    float *act = (float *) malloc(sizeof(float) * (size_t)nr * n);
    for (int i = 0; i < nr * n; i++)
        act[i] = (int)(lcg() >> 8) / (float)(1 << 24) - 1.0f;
    uint8_t *vy4 = (uint8_t *) malloc((size_t)(nr / 4) * nb * 136);
    for (int g = 0; g < nr / 4; g++)
        ggml_quantize_mat_q8_0_4x8(act + (size_t)g * 4 * n, vy4 + (size_t)g * nb * 136, n);
    float *sg = (float *) malloc(sizeof(float) * (size_t)nr * nc);
    ggml_gemm_mxfp4_8x8_q8_0(n, sg, nc, vx, vy4, nr, nc);
    best = 1e18;
    for (int r = 0; r < 50; r++) {
        double t0 = now_us();
        ggml_gemm_mxfp4_8x8_q8_0(n, sg, nc, vx, vy4, nr, nc);
        double dt = now_us() - t0;
        if (dt < best) best = dt;
    }
    printf("ref gemm  n=%d nc=%d nr=%d %s: %8.2f us/call  %7.1f GF/s  sg[0]=%f\n",
           n, nc, nr, clean ? "CLEAN" : "rand", best, gf * nr / (best / 1e6), sg[0]);
    return 0;
}
