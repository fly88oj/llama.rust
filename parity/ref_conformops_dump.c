/* conformer-family op ground-truth dumper — drives the reference build's
 * ggml_roll, ggml_conv_2d_dw_direct and ggml_conv_2d_direct through the real
 * graph-compute path and writes bit-exact outputs.
 *
 * Sections (u32 magic 'CFOP' then):
 *   kind 0 roll F32        : u32[4]{s0,s1,s2,s3} u32[4] ne | n f32 in | n out
 *   kind 1 conv_2d_dw F32k : u32[6]{s0,s1,p0,p1,d0,d1} u32[4] src_ne u32[4]
 *                            knl_ne | src | knl | out
 *   kind 2 conv_2d_dw F16k : same layout, knl dumped as u16
 *   kind 3 conv_2d_direct  : u32[6] params u32[4] src_ne u32[4] knl_ne |
 *                            src f32 | knl f32 | out f32
 *   kind 4 ggml_conv_2d (the im2col composite, F16 patches) on the same
 *                            inputs as kind 3 — documents that C's composite
 *                            is NOT bit-equal to the direct kernel (F32
 *                            patches), which is why the port's
 *                            conv_2d_direct keeps the patches in F32.
 * EOF: magic 0xFFFFFFFF.
 *
 * Build:
 *   gcc -O2 -o /tmp/s2g-tinyblas/ref_conformops_dump parity/ref_conformops_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-next/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./ref_conformops_dump parity/conformops_ref.bin
 * (regenerated for sync batch D2: a7b94df2c's tinyBLAS K tails flip the
 *  conv_2d_direct F32 patch GEMM (k = KW*KH*IC, e.g. 24) onto tinyBLAS —
 *  2224 bytes of ±1ulp output shifts vs the def4d406a artifact.)
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x5851f42du;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 4.0f - 2.0f;
}

static void hdr(FILE * f, uint32_t kind) {
    uint32_t magic = 0x504f4643u; /* "CFOP" little-endian */
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
}

static void dim4(FILE * f, const int64_t * ne) {
    uint32_t v[4] = { (uint32_t) ne[0], (uint32_t) ne[1], (uint32_t) ne[2], (uint32_t) ne[3] };
    fwrite(v, 4, 4, f);
}

static struct ggml_tensor * mk4(struct ggml_context * ctx, enum ggml_type ty, const int64_t ne[4], int fill) {
    struct ggml_tensor * t = ggml_new_tensor(ctx, ty, 4, ne);
    if (fill) {
        float * d = (float *) t->data;
        int64_t n = ne[0] * ne[1] * ne[2] * ne[3];
        if (ty == GGML_TYPE_F16) {
            uint16_t * h = (uint16_t *) t->data;
            for (int64_t i = 0; i < n; i++) {
                h[i] = ggml_fp32_to_fp16(next_val());
            }
        } else {
            for (int64_t i = 0; i < n; i++) {
                d[i] = next_val();
            }
        }
    }
    return t;
}

static void dump_out(FILE * f, struct ggml_tensor * r) {
    int64_t n = r->ne[0] * r->ne[1] * r->ne[2] * r->ne[3];
    fwrite(r->data, 4, (size_t) n, f);
}

/* run a single-output graph and dump dst */
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

static void dump_in_f32(FILE * f, struct ggml_tensor * t) {
    int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    fwrite(t->data, 4, (size_t) n, f);
}

int main(int argc, char ** argv) {
    const char * out = "conformops_ref.bin";
    if (argc > 1) {
        out = argv[1];
    }
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* ---- roll ---- */
    {
        static const int64_t shapes[][4] = {
            {7, 3, 1, 1}, {8, 2, 2, 1}, {13, 5, 1, 1}, {16, 1, 1, 1},
            {5, 4, 3, 2}, {64, 3, 1, 1}, {9, 8, 1, 1}, {31, 1, 1, 1},
        };
        static const int shifts[][4] = {
            {1, 0, 0, 0}, {2, 0, 0, 0}, {4, 1, 0, 0}, {1, 1, 1, 1},
            {3, 2, 0, 0}, {6, 0, 0, 0}, {1, 2, 0, 1}, {2, 1, 1, 0},
        };
        for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
            for (int gi = 0; gi < (int)(sizeof(shifts) / sizeof(shifts[0])); gi++) {
                const int64_t * ne = shapes[si];
                const int * sh = shifts[gi];
                if (sh[0] >= ne[0] || sh[1] >= ne[1] || sh[2] >= ne[2] || sh[3] >= ne[3]) {
                    continue;
                }
                /* ggml_graph_compute_with_ctx takes the plan + wdata from
                 * the same pool — the convs want a big scratch */
                size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() +
                                  (size_t) (ne[0] * ne[1] * ne[2] * ne[3]) * 8 + (1 << 26);
                struct ggml_init_params ip = { ctx_size, NULL, false };
                struct ggml_context * ctx = ggml_init(ip);
                struct ggml_tensor * a = mk4(ctx, GGML_TYPE_F32, ne, 1);
                hdr(f, 0);
                uint32_t s[4] = { (uint32_t) sh[0], (uint32_t) sh[1], (uint32_t) sh[2], (uint32_t) sh[3] };
                fwrite(s, 4, 4, f);
                dim4(f, ne);
                dump_in_f32(f, a);
                struct ggml_tensor * r = ggml_roll(ctx, a, sh[0], sh[1], sh[2], sh[3]);
                run1(ctx, r, f);
                ggml_free(ctx);
            }
        }
    }

    /* ---- conv_2d_dw_direct (F32 and F16 kernels) ---- */
    {
        /* (src_w, src_h, C, N, kw, kh, s0, s1, p0, p1, d0, d1) */
        static const int cfg[][12] = {
            {16, 8, 3, 1, 3, 3, 2, 2, 1, 1, 1, 1},
            {25, 16, 4, 1, 3, 3, 2, 2, 1, 1, 1, 1},
            {13, 7, 2, 1, 3, 3, 1, 1, 0, 0, 1, 1},
            {10, 10, 5, 2, 1, 1, 1, 1, 0, 0, 1, 1},
            {8, 6, 2, 1, 3, 1, 2, 1, 1, 0, 1, 1},
            {31, 9, 3, 1, 5, 5, 3, 2, 2, 2, 1, 1},
            {12, 12, 1, 1, 3, 3, 2, 2, 1, 1, 1, 1},
            {6, 6, 8, 1, 3, 3, 1, 1, 1, 1, 2, 2},
        };
        for (int ci = 0; ci < (int)(sizeof(cfg) / sizeof(cfg[0])); ci++) {
            for (int kh16 = 0; kh16 < 2; kh16++) {
                const int * c = cfg[ci];
                const int64_t src_ne[4] = { c[0], c[1], c[2], c[3] };
                const int64_t knl_ne[4] = { c[4], c[5], 1, c[2] };
                size_t n_src = (size_t) (c[0] * c[1] * c[2] * c[3]);
                size_t n_knl = (size_t) (c[4] * c[5] * c[2]);
                size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() +
                                  (n_src + n_knl) * 8 + (1 << 26);
                struct ggml_init_params ip = { ctx_size, NULL, false };
                struct ggml_context * ctx = ggml_init(ip);
                struct ggml_tensor * src = mk4(ctx, GGML_TYPE_F32, src_ne, 1);
                struct ggml_tensor * knl = mk4(ctx, kh16 ? GGML_TYPE_F16 : GGML_TYPE_F32, knl_ne, 1);
                hdr(f, kh16 ? 2 : 1);
                uint32_t pp[6] = { (uint32_t) c[6], (uint32_t) c[7], (uint32_t) c[8],
                                   (uint32_t) c[9], (uint32_t) c[10], (uint32_t) c[11] };
                fwrite(pp, 4, 6, f);
                dim4(f, src_ne);
                dim4(f, knl_ne);
                dump_in_f32(f, src);
                fwrite(knl->data, kh16 ? 2 : 4, n_knl, f);
                struct ggml_tensor * r =
                    ggml_conv_2d_dw_direct(ctx, knl, src, c[6], c[7], c[8], c[9], c[10], c[11]);
                run1(ctx, r, f);
                ggml_free(ctx);
            }
        }
    }

    /* ---- conv_2d_direct vs the im2col composite ---- */
    {
        /* (src_w, src_h, IC, N, kw, kh, OC, s0, s1, p0, p1, d0, d1) */
        static const int cfg[][13] = {
            {16, 8, 3, 1, 1, 1, 4, 1, 1, 0, 0, 1, 1},
            {25, 16, 4, 1, 1, 1, 8, 1, 1, 0, 0, 1, 1},
            {13, 7, 2, 2, 1, 1, 3, 1, 1, 0, 0, 1, 1},
            {10, 10, 5, 1, 3, 3, 6, 2, 1, 1, 1, 1, 1},
            {8, 32, 16, 1, 1, 1, 12, 1, 1, 0, 0, 1, 1},
            {31, 9, 3, 1, 3, 1, 5, 1, 1, 1, 0, 1, 1},
        };
        for (int ci = 0; ci < (int)(sizeof(cfg) / sizeof(cfg[0])); ci++) {
            const int * c = cfg[ci];
            const int64_t src_ne[4] = { c[0], c[1], c[2], c[3] };
            const int64_t knl_ne[4] = { c[4], c[5], c[2], c[6] };
            size_t n_src = (size_t) (c[0] * c[1] * c[2] * c[3]);
            size_t n_knl = (size_t) (c[4] * c[5] * c[2] * c[6]);
            size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 2 +
                              (n_src + n_knl) * 8 + (1 << 26);
            struct ggml_init_params ip = { ctx_size, NULL, false };
            struct ggml_context * ctx = ggml_init(ip);
            struct ggml_tensor * src = mk4(ctx, GGML_TYPE_F32, src_ne, 1);
            struct ggml_tensor * knl = mk4(ctx, GGML_TYPE_F32, knl_ne, 1);
            uint32_t pp[6] = { (uint32_t) c[7], (uint32_t) c[8], (uint32_t) c[9],
                               (uint32_t) c[10], (uint32_t) c[11], (uint32_t) c[12] };
            /* kind 3: direct */
            hdr(f, 3);
            fwrite(pp, 4, 6, f);
            dim4(f, src_ne);
            dim4(f, knl_ne);
            dump_in_f32(f, src);
            dump_in_f32(f, knl);
            struct ggml_tensor * r3 =
                ggml_conv_2d_direct(ctx, knl, src, c[7], c[8], c[9], c[10], c[11], c[12]);
            run1(ctx, r3, f);
            /* kind 4: composite on the same inputs (same ctx, same data) */
            hdr(f, 4);
            fwrite(pp, 4, 6, f);
            dim4(f, src_ne);
            dim4(f, knl_ne);
            dump_in_f32(f, src);
            dump_in_f32(f, knl);
            struct ggml_tensor * r4 =
                ggml_conv_2d(ctx, knl, src, c[7], c[8], c[9], c[10], c[11], c[12]);
            run1(ctx, r4, f);
            ggml_free(ctx);
        }
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}
