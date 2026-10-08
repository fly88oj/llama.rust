/* Isolate the reference's mul_mat numerics for the deepseek4 Hadamard shape
 * (rot [64,64] F32 x x [64,24] F32, single thread) — the arch-batch-7 node
 * dump showed this gemm 1-2 ulp off the port's. Prints the raw f32 outputs.
 *
 *   gcc -O2 -I pinned/ggml/include parity/ref_hadamard_mm_probe.c -o /tmp/hmm \
 *     -L build-rust-ref/bin -lggml -lggml-base -lggml-cpu \
 *     -Wl,-rpath,build-rust-ref/bin && /tmp/hmm
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static void gen_hadamard(float * d, int n) {
    const float s = 1.0f / sqrtf((float) n);
    memset(d, 0, n * n * sizeof(float));
    d[0] = s;
    for (int i = 1; i < n; i <<= 1) {
        for (int x = 0; x < i; x++) {
            for (int y = 0; y < i; y++) {
                const float v = d[x * n + y];
                d[(x + i) * n + y] = v;
                d[x * n + y + i] = v;
                d[(x + i) * n + y + i] = -v;
            }
        }
    }
}

int main(void) {
    struct ggml_init_params ip = { 32u << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 24);

    gen_hadamard((float *) rot->data, 64);

    /* deterministic input */
    unsigned st = 12345;
    float * px = (float *) x->data;
    for (int i = 0; i < 64 * 24; i++) {
        st = st * 1103515245u + 12345u;
        px[i] = ((st >> 8) & 0xffffff) / 8388608.0f * 2.0f - 1.0f;
    }

    struct ggml_tensor * y = ggml_mul_mat(ctx, rot, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 8 /* nth, like the parity runs */);

    const float * py = (const float *) y->data;
    for (int i = 0; i < 6; i++) {
        printf("%d %.9g %08x\n", i, py[i], *(unsigned *)&py[i]);
    }
    printf("... %d %.9g %08x\n", 64 * 24 - 1, py[64 * 24 - 1], *(unsigned *)&py[64 * 24 - 1]);
    return 0;
}
