/* Tiled K-quant matmul ground truth (sync batch D).
 *
 * Dumps MUL_MAT / MUL_MAT_ID outputs computed by the NEW reference build
 * (def4d406a) through the production graph path, which since the iqp removal
 * dispatches `ggml_compute_forward_mul_mat_tiled` / `..._mul_mat_id_tiled`
 * (ggml-cpu.c:1269 / :1678) for every supported K-quant with batch >= 8.
 * Sections cover the standard (>16 src1 rows), narrow (<=16, k_extent) and
 * ragged/batch-clamped window paths plus a mixed-batch MUL_MAT_ID (some
 * experts >= 8 rows -> tiled, others -> stock vec_dot).
 *
 * Build (from the repo root):
 *   NEXT=/home/jeffrey/llm/llama.cpp-next
 *   REF=$NEXT/build-rust-ref/bin
 *   gcc -O2 -std=c11 parity/ref_tiled_dump.c -I$NEXT/ggml/include \
 *       -o /tmp/syncg-ref_tiled_dump -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: /tmp/syncg-ref_tiled_dump parity/tiled_ref.bin
 *
 * File format (all little-endian):
 *   u32 magic 'VMT1' | u32 kind (0 = MUL_MAT, 1 = MUL_MAT_ID) |
 *   u32 type_id | u32 n | u32 rows(R) | u32 cols(C) |
 *   kind 0: xq bytes (row_size(n)*R) | y f32 (n*C) | dst f32 (R*C)
 *   kind 1: u32 n_as | u32 n_ids | u32 ne12 |
 *           xq bytes (row_size(n)*R*n_as) | y f32 (n*C*ne12) |
 *           ids i32 (n_ids*ne12) | dst f32 (R*n_ids*ne12)
 *   EOF: u32 0xFFFFFFFF
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-alloc.h"

static uint32_t lcg = 0x7eedbeefu;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 1.8f - 0.9f;
}

static void wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_f32(FILE *f, float v) { fwrite(&v, 4, 1, f); }

/* IQ quantizers require an importance matrix (ggml_quantize_requires_imatrix);
 * a flat 1.0 vector keeps the artifact reproducible (only the xq bytes are
 * replayed, so its exact contents do not affect the port). */
static float *ones(int n) {
    float *v = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) v[i] = 1.0f;
    return v;
}

static void dump_mm(FILE *f, enum ggml_type ty, int n, int R, int C, int nth) {
    float *w = malloc((size_t)n * R * 4);
    float *y = malloc((size_t)n * C * 4);
    float *im = ones(n);
    for (int i = 0; i < n * R; i++) w[i] = next_val();
    for (int i = 0; i < n * C; i++) y[i] = next_val();

    struct ggml_init_params ip = { .mem_size = 512 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, ty, n, R);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, C);
    struct ggml_tensor *d = ggml_mul_mat(ctx, a, b);
    memcpy(b->data, y, (size_t)n * C * 4);
    ggml_quantize_chunk(ty, w, a->data, 0, R, n, im);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    struct ggml_context *ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, nth);

    wr_u32(f, 0x31544D56); /* 'VMT1' LE */
    wr_u32(f, 0);          /* kind = MUL_MAT */
    wr_u32(f, ty); wr_u32(f, n); wr_u32(f, R); wr_u32(f, C);
    fwrite(a->data, 1, ggml_row_size(ty, (int64_t)n * R), f);
    fwrite(y, 4, (size_t)n * C, f);
    fwrite(d->data, 4, (size_t)R * C, f);
    const float *dd = (const float *)d->data;
    printf("mm   type %2d n=%4d R=%4d C=%4d -> %.6f %.6f\n", ty, n, R, C, dd[0], dd[R * C - 1]);
    ggml_free(ctxc);
    ggml_free(ctx);
    free(w);
    free(y);
    free(im);
}

static void dump_mmid(FILE *f, enum ggml_type ty, int n, int R, int C, int n_as, int n_ids, int ne12, int nth) {
    /* C == ne11 (rows per token plane) */
    float *w = malloc((size_t)n * R * n_as * 4);
    float *y = malloc((size_t)n * C * ne12 * 4);
    float *im = ones(n);
    int32_t *ids = malloc((size_t)n_ids * ne12 * 4);
    for (int i = 0; i < n * R * n_as; i++) w[i] = next_val();
    for (int i = 0; i < n * C * ne12; i++) y[i] = next_val();
    /* mixed routing: token 0 spreads over all experts, later tokens skew to
     * experts 0/1 so some experts stay < 8 routed rows (stock path) */
    for (int t = 0; t < ne12; t++) {
        for (int e = 0; e < n_ids; e++) {
            ids[t * n_ids + e] = (t == 0) ? (e % n_as) : ((e + t) % 2 ? (e % n_as) : (t % n_as));
        }
    }

    struct ggml_init_params ip = { .mem_size = 512 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *a   = ggml_new_tensor_3d(ctx, ty, n, R, n_as);
    struct ggml_tensor *b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, C, ne12);
    struct ggml_tensor *iid = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, ne12);
    struct ggml_tensor *d   = ggml_mul_mat_id(ctx, a, b, iid);
    memcpy(b->data, y, (size_t)n * C * ne12 * 4);
    memcpy(iid->data, ids, (size_t)n_ids * ne12 * 4);
    ggml_quantize_chunk(ty, w, a->data, 0, (int64_t)R * n_as, n, im);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    struct ggml_context *ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, nth);

    size_t nout = (size_t)R * n_ids * ne12;
    wr_u32(f, 0x31544D56); /* 'VMT1' LE */
    wr_u32(f, 1);          /* kind = MUL_MAT_ID */
    wr_u32(f, ty); wr_u32(f, n); wr_u32(f, R); wr_u32(f, C);
    wr_u32(f, n_as); wr_u32(f, n_ids); wr_u32(f, ne12);
    fwrite(a->data, 1, ggml_row_size(ty, (int64_t)n * R * n_as), f);
    fwrite(y, 4, (size_t)n * C * ne12, f);
    fwrite(ids, 4, (size_t)n_ids * ne12, f);
    fwrite(d->data, 4, nout, f);
    const float *dd = (const float *)d->data;
    printf("mmid type %2d n=%4d R=%4d C=%4d as=%d ids=%d t=%d -> %.6f %.6f\n",
           ty, n, R, C, n_as, n_ids, ne12, dd[0], dd[nout - 1]);
    ggml_free(ctxc);
    ggml_free(ctx);
    free(w);
    free(y);
    free(im);
    free(ids);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "tiled_ref.bin";
    const int nth = argc > 2 ? atoi(argv[2]) : 4;
    int types[] = {12, 13, 14, 11, 10, 23, 16, 17, 22, 18, 21, 19, 29};
    FILE *f = fopen(out, "wb");

    /* NB: R % 8 != 0 on purpose — the reference only repacks tensors the
     * model loader put in its CPU_REPACK buffer (src0->extra); ad-hoc probe
     * tensors always reach the tiled path, while the port's lazy repack
     * approximates "would the loader repack it" by geometry alone and would
     * take R % 8 == 0 shapes down the 8x8 repack lane instead (a documented
     * routing approximation, see PARITY.md batch D). */
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        /* standard path: C > 16, ragged windows on both axes */
        dump_mm(f, (enum ggml_type)types[t], 512, 300, 40, nth);
        /* narrow path: C <= 16 (k_extent chunks, num_k > 1 when profitable) */
        dump_mm(f, (enum ggml_type)types[t], 512, 300, 12, nth);
        /* minimum profitable batch */
        dump_mm(f, (enum ggml_type)types[t], 512, 12, 8, nth);
        /* C > 256: multiple batch-clamped iir1 windows */
        dump_mm(f, (enum ggml_type)types[t], 512, 188, 300, nth);
    }
    /* MUL_MAT_ID: mixed expert batches (tiled + stock per node) */
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        /* ggml_mul_mat_id requires ids->ne[0] % ne11 == 0: n_ids 4, ne11 2 */
        dump_mmid(f, (enum ggml_type)types[t], 512, 140, 2, 5, 4, 16, nth);
    }

    wr_u32(f, 0xFFFFFFFFu);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
