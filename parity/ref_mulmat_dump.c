/* Production-path mul_mat ground truth: builds a real ggml graph
 * (quantized [1024 x R] weights × f32 [1024 x C] activations) and computes it
 * through the same CPU path llama-cli uses. Section per type.
 *
 * Section: u32 'VMM1' | u32 type_id | u32 n | u32 R | u32 C |
 *          xq bytes | y f32 bytes | dst f32 (R*C) . EOF 0xFFFFFFFF.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-alloc.h"

typedef void (*qfn)(const float *, void *, int64_t);
extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0x5eed1234u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 1.8f - 0.9f;
}

static void quantize_rows(enum ggml_type ty, const float *x, void *dst, int n, int rows);

static void quantize_rows(enum ggml_type ty, const float *x, void *dst, int n, int rows) {
    ggml_quantize_chunk(ty, x, dst, 0, rows, n, NULL);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "mulmat_ref.bin";
    const int n = 1024, R = 4, C = 3;
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
        quantize_rows(ty, w, a->data, n, R);

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
        float *dd = (float *)d->data;
        printf("type %2d -> %.6f %.6f\n", types[t], dd[0], dd[1]);
        ggml_free(ctxc);
        ggml_free(ctx);
    }
    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
