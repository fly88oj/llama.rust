/* audio-round-4 op ground-truth dumper — drives the reference build's
 * ggml_sub / ggml_sin / ggml_cos / ggml_sqr / ggml_mean /
 * ggml_pad_reflect_1d / ggml_elu / ggml_pad_ext through the real
 * graph-compute path (single thread, like the audio encoder graphs' n_tasks
 * splits) and writes bit-exact outputs.
 *
 * Sections (u32 magic 'AUOP' = 0x504F5541 then):
 *   kind 10 sub             : u32[4] ne_a u32[4] ne_b | a f32 | b f32 | out
 *   kind 11 sin             : u32[4] ne | in f32 | out
 *   kind 12 cos             : u32[4] ne | in f32 | out
 *   kind 13 sqr             : u32[4] ne | in f32 | out
 *   kind 14 mean            : u32[4] ne | in f32 | out
 *   kind 15 pad_reflect_1d  : u32[2]{p0,p1} u32[4] ne | in | out
 *   kind 16 elu             : u32[4] ne | in f32 | out
 *   kind 17 pad_ext         : u32[8]{lp0,rp0,..lp3,rp3} u32[4] ne | in | out
 * EOF: magic 0xFFFFFFFF.
 *
 * Build:
 *   gcc -O2 -o parity/ref_audioops_dump parity/ref_audioops_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_audioops_dump parity/audioops_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x660dc517u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 4.0f - 2.0f;
}

static void hdr(FILE * f, uint32_t kind) {
    uint32_t magic = 0x504F5541u; /* "AUOP" little-endian */
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
}

static void dim4(FILE * f, const int64_t * ne) {
    uint32_t v[4] = { (uint32_t) ne[0], (uint32_t) ne[1], (uint32_t) ne[2], (uint32_t) ne[3] };
    fwrite(v, 4, 4, f);
}

static struct ggml_tensor * mk4(struct ggml_context * ctx, const int64_t ne[4], int fill) {
    struct ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    if (fill) {
        float * d = (float *) t->data;
        int64_t n = ne[0] * ne[1] * ne[2] * ne[3];
        for (int64_t i = 0; i < n; i++) {
            d[i] = next_val();
        }
    }
    return t;
}

static void dump_out(FILE * f, struct ggml_tensor * r) {
    int64_t n = r->ne[0] * r->ne[1] * r->ne[2] * r->ne[3];
    fwrite(r->data, 4, (size_t) n, f);
}

static void run1(struct ggml_context * ctx, struct ggml_tensor * r, FILE * f) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed %d\n", (int) st);
        exit(1);
    }
    dump_out(f, r);
}

static struct ggml_context * mkctx(size_t extra) {
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + extra + (1 << 24);
    struct ggml_init_params ip = { ctx_size, NULL, false };
    return ggml_init(ip);
}

int main(int argc, char ** argv) {
    const char * out = "audioops_ref.bin";
    if (argc > 1) {
        out = argv[1];
    }
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* ---- sub (broadcast b over a, the RVQ/statistics shapes) ---- */
    {
        /* {a_ne, b_ne} pairs: full-shape, row-broadcast, scalar */
        static const int64_t cfg[][8] = {
            {13, 5, 2, 1,  13, 5, 2, 1},
            {32, 7, 1, 1,  32, 1, 1, 1},
            {24, 6, 1, 1,  1,  6, 1, 1},
            {9,  4, 3, 1,  9,  4, 1, 1},
        };
        for (int ci = 0; ci < (int)(sizeof(cfg) / sizeof(cfg[0])); ci++) {
            const int64_t ane[4] = { cfg[ci][0], cfg[ci][1], cfg[ci][2], cfg[ci][3] };
            const int64_t bne[4] = { cfg[ci][4], cfg[ci][5], cfg[ci][6], cfg[ci][7] };
            struct ggml_context * ctx = mkctx(0);
            struct ggml_tensor * a = mk4(ctx, ane, 1);
            struct ggml_tensor * b = mk4(ctx, bne, 1);
            hdr(f, 10);
            dim4(f, ane);
            dim4(f, bne);
            fwrite(a->data, 4, (size_t)(ane[0] * ane[1] * ane[2] * ane[3]), f);
            fwrite(b->data, 4, (size_t)(bne[0] * bne[1] * bne[2] * bne[3]), f);
            struct ggml_tensor * r = ggml_sub(ctx, a, b);
            run1(ctx, r, f);
            ggml_free(ctx);
        }
    }

    /* ---- sin / cos / sqr / elu over shared shapes ---- */
    {
        static const int64_t shapes[][4] = {
            {64, 8, 1, 1}, {128, 13, 1, 1}, {257, 1, 1, 1}, {17, 3, 2, 1},
        };
        for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
            const int64_t * ne = shapes[si];
            int64_t n = ne[0] * ne[1] * ne[2] * ne[3];
            /* the same input feeds all four unary ops */
            struct ggml_context * ctx = mkctx((size_t)n * 8);
            struct ggml_tensor * a = mk4(ctx, ne, 1);

            hdr(f, 11);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)n, f);
            run1(ctx, ggml_sin(ctx, a), f);

            hdr(f, 12);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)n, f);
            run1(ctx, ggml_cos(ctx, a), f);

            hdr(f, 13);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)n, f);
            run1(ctx, ggml_sqr(ctx, a), f);

            hdr(f, 16);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)n, f);
            run1(ctx, ggml_elu(ctx, a), f);
            ggml_free(ctx);
        }
    }

    /* ---- mean (row-wise over ne0) ---- */
    {
        static const int64_t shapes[][4] = {
            {25, 64, 1, 1},  /* speaker C-major layout, T on ne0 */
            {128, 3, 1, 1},
            {7, 5, 2, 1},
        };
        for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
            const int64_t * ne = shapes[si];
            struct ggml_context * ctx = mkctx(0);
            struct ggml_tensor * a = mk4(ctx, ne, 1);
            hdr(f, 14);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)(ne[0] * ne[1] * ne[2] * ne[3]), f);
            run1(ctx, ggml_mean(ctx, a), f);
            ggml_free(ctx);
        }
    }

    /* ---- pad_reflect_1d ---- */
    {
        static const int cfg[][6] = {
            /* {ne0, ne1, p0, p1} */
            {31, 8, 2, 2}, {40, 16, 4, 4}, {25, 3, 1, 2}, {64, 1, 8, 8},
        };
        for (int ci = 0; ci < (int)(sizeof(cfg) / sizeof(cfg[0])); ci++) {
            const int64_t ne[4] = { cfg[ci][0], cfg[ci][1], 1, 1 };
            struct ggml_context * ctx = mkctx(0);
            struct ggml_tensor * a = mk4(ctx, ne, 1);
            hdr(f, 15);
            uint32_t pp[2] = { (uint32_t) cfg[ci][2], (uint32_t) cfg[ci][3] };
            fwrite(pp, 4, 2, f);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)(ne[0] * ne[1]), f);
            run1(ctx, ggml_pad_reflect_1d(ctx, a, cfg[ci][2], cfg[ci][3]), f);
            ggml_free(ctx);
        }
    }

    /* ---- pad_ext (both-side pads, the parakeet/pocket-tts shapes) ---- */
    {
        static const int cfg[][12] = {
            /* {ne0..ne3, lp0, rp0, lp1, rp1, lp2, rp2} */
            {40, 3, 1, 1,   2, 2, 0, 0, 0, 0, 0, 0},
            {17, 2, 2, 1,   0, 3, 0, 0, 0, 0, 0, 0},
            {20, 4, 1, 1,   0, 0, 0, 0, 5, 3, 0, 0},
            {33, 2, 1, 1,   4, 1, 1, 0, 0, 0, 0, 0},
        };
        for (int ci = 0; ci < (int)(sizeof(cfg) / sizeof(cfg[0])); ci++) {
            const int64_t ne[4] = { cfg[ci][0], cfg[ci][1], cfg[ci][2], cfg[ci][3] };
            struct ggml_context * ctx = mkctx(0);
            struct ggml_tensor * a = mk4(ctx, ne, 1);
            hdr(f, 17);
            uint32_t pp[8];
            for (int k = 0; k < 8; k++) {
                pp[k] = (uint32_t) cfg[ci][4 + k];
            }
            fwrite(pp, 4, 8, f);
            dim4(f, ne);
            fwrite(a->data, 4, (size_t)(ne[0] * ne[1] * ne[2] * ne[3]), f);
            run1(ctx, ggml_pad_ext(ctx, a,
                        cfg[ci][4], cfg[ci][5], cfg[ci][6], cfg[ci][7],
                        cfg[ci][8], cfg[ci][9], cfg[ci][10], cfg[ci][11]), f);
            ggml_free(ctx);
        }
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}
