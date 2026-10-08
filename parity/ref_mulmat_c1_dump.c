/* Companion to ref_mulmat_dump.c: same production-path mul_mat graph, but with
 * a single activation column (ne11 = 1). That shape disables
 * llamafile_sgemm (`if (n < 2) return false;`, llamafile/sgemm.cpp:3820), so
 * the reference runs the ggml_vec_dot_* kernels for every type — including
 * F16/BF16, which for ne11 >= 2 are handed to llamafile tinyBLAS instead.
 * Used by vec_dot::mulmat_tests to check all 12 types bit-exactly through our
 * mul_mat dispatch.
 *
 * Section: u32 'VMM1' | u32 type_id | u32 n | u32 R | u32 C |
 *          xq bytes | y f32 bytes | dst f32 (R*C) . EOF 0xFFFFFFFF.
 *
 * Build/run:
 *   gcc ref_mulmat_c1_dump.c -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -o ref_mulmat_c1_dump -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml -lggml-cpu -lggml-base -lm -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   ./ref_mulmat_c1_dump mulmat_ref_c1.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"

extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0x7a17c0deu;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 1.8f - 0.9f;
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "mulmat_ref_c1.bin";
    const int n = 1024, R = 4, C = 1;
    float *w = malloc(n * R * 4), *y = malloc(n * C * 4);
    for (int i = 0; i < n * R; i++) w[i] = next_val();
    for (int i = 0; i < n * C; i++) y[i] = next_val();

    int types[] = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 1, 30}; /* + F16, BF16 */

    FILE *f = fopen(out, "wb");
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        enum ggml_type ty = (enum ggml_type)types[t];
        struct ggml_init_params ip = { .mem_size = 256 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *a = ggml_new_tensor_2d(ctx, ty, n, R);
        struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, C);
        struct ggml_tensor *d = ggml_mul_mat(ctx, a, b);
        memcpy(b->data, y, (size_t)n * C * 4);
        ggml_quantize_chunk(ty, w, a->data, 0, R, n, NULL);

        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, d);
        struct ggml_context *ctxc = ggml_init(ip);
        ggml_graph_compute_with_ctx(ctxc, gf, 4);

        uint32_t magic = 0x314D4D56, tid = types[t], nn = n, rr = R, cc = C;
        fwrite(&magic, 4, 1, f); fwrite(&tid, 4, 1, f); fwrite(&nn, 4, 1, f);
        fwrite(&rr, 4, 1, f); fwrite(&cc, 4, 1, f);
        size_t xs = ggml_row_size(ty, (int64_t)n * R);
        fwrite(a->data, 1, xs, f);
        fwrite(y, 4, (size_t)n * C, f);
        fwrite(d->data, 4, (size_t)R * C, f);
        printf("type %2d -> %.6f\n", types[t], ((float *)d->data)[0]);
        ggml_free(ctxc);
        ggml_free(ctx);
    }
    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}