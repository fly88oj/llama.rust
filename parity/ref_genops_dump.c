/* TTS-generator op ground-truth dumper — drives the reference build's ggml
 * ops through the real graph-compute path (ggml_graph_compute_with_ctx, the
 * same dispatch a model graph takes) and writes bit-exact outputs for:
 *
 *   ggml_col2im_1d  (ggml.c:4679 → ops.cpp:7070 ggml_compute_forward_col2im_1d)
 *   ggml_cumsum     (ggml.c:2508 → ops.cpp:1440)
 *   ggml_tri        (ggml.c:5368 → ops.cpp:2324)
 *   ggml_sum        (ggml.c:2477 → ops.cpp:1382)
 *   ggml_log        (ggml.c:2387 → unary-ops.cpp:297)
 *   ggml_step       (ggml.c:2695 → unary-ops.cpp:10158, GGML_UNARY_OP_STEP)
 *
 * Format (per section): u32 magic 'GEN1' | u32 kind | u32 n_in | u32 n_out |
 *   shape/param words | payload (f32 unless the kind says otherwise)
 *
 *   kind 0 = col2im_1d(f32): u32 s0, oc, p0; u32 ne0, ne1 | ne0*ne1 in |
 *            ne0*ne1 out (out = [(T_in-1)*s0+K-2*p0, oc])
 *   kind 1 = cumsum(f32): u32 rows, row_len | n in | n out
 *   kind 2 = tri(f32): u32 tri_type, n | n in | n out (square n x n)
 *   kind 3 = sum(f32): u32 rows, row_len | n in | 1 out
 *   kind 4 = log(f32): u32 n | n in | n out
 *   kind 5 = step(f32): u32 n | n in | n out
 *   kind 6 = col2im_1d(f16): as kind 0, payload u16
 * EOF: magic 0xFFFFFFFF
 *
 * Build (against the reference build's shared libs):
 *   gcc -O2 -o parity/ref_genops_dump parity/ref_genops_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_genops_dump parity/genops_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x1234567u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    /* keep the magnitudes tame so the f16 round trip is exercised but the
     * scatter-add sums stay representable */
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 2.0f - 1.0f;
}

static void hdr(FILE *f, uint32_t kind, uint32_t n_in, uint32_t n_out,
                const uint32_t *extra, uint32_t n_extra) {
    uint32_t magic = 0x314e4547u; /* "GEN1" little-endian */
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&n_in, 4, 1, f);
    fwrite(&n_out, 4, 1, f);
    fwrite(extra, 4, n_extra, f);
}

/* kind 0/6: scatter-add [K*OC, T_in] -> [T_out, OC] */
static void run_col2im(FILE *f, int s0, int oc, int p0, int k, int t_in, int as_f16) {
    int k_oc = k * oc;
    int n = k_oc * t_in;
    int t_out = (t_in - 1) * s0 + k - 2 * p0;

    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 8 + (size_t) t_out * oc * 8 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    float * vals = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) vals[i] = next_val();

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, as_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                                k_oc, t_in);
    if (as_f16) {
        uint16_t * h = (uint16_t *) a->data;
        for (int i = 0; i < n; i++) {
            ggml_fp16_t e = ggml_fp32_to_fp16(vals[i]);
            memcpy(&h[i], &e, 2);
        }
    } else {
        memcpy(a->data, vals, (size_t) n * sizeof(float));
    }

    struct ggml_tensor * r = ggml_col2im_1d(ctx, a, s0, oc, p0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    /* 4 threads: the T_out band split must not change any byte */
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(1);
    }

    uint32_t extra[5] = { (uint32_t) s0, (uint32_t) oc, (uint32_t) p0,
                          (uint32_t) k_oc, (uint32_t) t_in };
    hdr(f, as_f16 ? 6u : 0u, (uint32_t) n, (uint32_t) (t_out * oc), extra, 5);
    if (as_f16) {
        fwrite(a->data, 2, n, f);
        fwrite(r->data, 2, (size_t) t_out * oc, f);
    } else {
        fwrite(a->data, 4, n, f);
        fwrite(r->data, 4, (size_t) t_out * oc, f);
    }
    free(vals);
    ggml_free(ctx);
}

/* kind 1: cumsum over rows */
static void run_cumsum(FILE *f, int rows, int row_len) {
    int n = rows * row_len;
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 8 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_len, rows);
    for (int i = 0; i < n; i++) ((float *) a->data)[i] = next_val();

    struct ggml_tensor * r = ggml_cumsum(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) exit(1);

    uint32_t extra[2] = { (uint32_t) rows, (uint32_t) row_len };
    hdr(f, 1, (uint32_t) n, (uint32_t) n, extra, 2);
    fwrite(a->data, 4, n, f);
    fwrite(r->data, 4, n, f);
    ggml_free(ctx);
}

/* kind 2: tri over an n x n square */
static void run_tri(FILE *f, int tri_type, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * n * 8 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    for (int i = 0; i < n * n; i++) ((float *) a->data)[i] = next_val();

    struct ggml_tensor * r = ggml_tri(ctx, a, (enum ggml_tri_type) tri_type);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) exit(1);

    uint32_t extra[2] = { (uint32_t) tri_type, (uint32_t) n };
    hdr(f, 2, (uint32_t) (n * n), (uint32_t) (n * n), extra, 2);
    fwrite(a->data, 4, n * n, f);
    fwrite(r->data, 4, n * n, f);
    ggml_free(ctx);
}

/* kind 3: whole-tensor sum -> scalar */
static void run_sum(FILE *f, int rows, int row_len) {
    int n = rows * row_len;
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 4 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_len, rows);
    for (int i = 0; i < n; i++) ((float *) a->data)[i] = next_val();

    struct ggml_tensor * r = ggml_sum(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) exit(1);

    uint32_t extra[2] = { (uint32_t) rows, (uint32_t) row_len };
    hdr(f, 3, (uint32_t) n, 1, extra, 2);
    fwrite(a->data, 4, n, f);
    fwrite(r->data, 4, 1, f);
    ggml_free(ctx);
}

/* kind 4/5: log / step over a fixed pattern covering the log masks' domain
 * (0 / 1 inputs of the keep-masks plus generic values) */
static void run_unary(FILE *f, uint32_t kind, const float *in, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 8 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(a->data, in, (size_t) n * sizeof(float));

    struct ggml_tensor * r = kind == 4 ? ggml_log(ctx, a) : ggml_step(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) exit(1);

    hdr(f, kind, (uint32_t) n, (uint32_t) n, NULL, 0);
    fwrite(in, 4, n, f);
    fwrite(r->data, 4, n, f);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "parity/genops_ref.bin";
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    /* col2im_1d shape grid: every (s0, oc, p0, k, t_in) the generators use.
     * qwen3tts: stride 2..512, k == 2*s0 (dac) or k == s0 (upsample);
     * pockettts seanet: stride 4/5/6 with k > s0 (overlap tails), depthwise
     * (k_oc = K*OC with OC up to 1024). */
    run_col2im(f, 2,  8,  0, 4,  3, 0);
    run_col2im(f, 2,  8,  0, 2,  5, 0);
    run_col2im(f, 4,  16, 0, 8,  2, 0);
    run_col2im(f, 4,  16, 0, 6,  4, 0);
    run_col2im(f, 5,  7,  0, 9,  3, 0);
    run_col2im(f, 6,  13, 0, 11, 2, 0);
    run_col2im(f, 2,  64, 0, 4,  9, 0);
    run_col2im(f, 8,  32, 0, 16, 4, 0);
    run_col2im(f, 16, 96, 0, 32, 3, 0);
    run_col2im(f, 512, 1, 0, 1024, 2, 0);
    run_col2im(f, 2,  8,  0, 4,  3, 1); /* f16 variant */

    /* cumsum grids: flat/ragged rows */
    run_cumsum(f, 1, 1);
    run_cumsum(f, 1, 17);
    run_cumsum(f, 16, 1);
    run_cumsum(f, 3, 257);
    run_cumsum(f, 7, 64);

    /* tri: all four types, incl. the 16x16 causal-cache size */
    run_tri(f, 0, 4);
    run_tri(f, 1, 8);
    run_tri(f, 2, 16);
    run_tri(f, 3, 16);
    run_tri(f, 2, 5);

    /* sum: the keep-mask counting shapes */
    run_sum(f, 1, 1);
    run_sum(f, 1, 50);
    run_sum(f, 1, 4096);
    run_sum(f, 16, 896);

    /* log/step: mask-domain pattern (0/1 and -inf-free), then generic */
    {
        enum { NU = 40 };
        float in[NU];
        int i = 0;
        in[i++] = 0.0f;
        in[i++] = 1.0f;
        in[i++] = 0.0f;
        in[i++] = 1.0f;
        in[i++] = 1.0f;
        in[i++] = 0.0f;
        in[i++] = 0.0f;
        in[i++] = 0.0f;
        in[i++] = 1.0f;
        in[i++] = 1.0f;
        for (; i < NU; i++) in[i] = next_val();
        /* step over sign boundary values too */
        in[20] = -0.0f;
        in[21] = 0.25f;
        in[22] = -1e-30f;
        run_unary(f, 4, in, NU);
        run_unary(f, 5, in, NU);
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", path);
    return 0;
}
