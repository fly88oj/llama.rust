/* ggml_pool_1d ground-truth dumper — drives the reference build's
 * ggml_pool_1d (ggml.c:5071 -> GGML_OP_POOL_1D, kernel ops.cpp:7690-7754
 * ggml_compute_forward_pool_1d_ksp) through the real graph-compute path and
 * writes bit-exact outputs.
 *
 * Sections (each: u32 magic 'P1DR' | u32 kind | u32 rows | u32 iw | u32 ow |
 *           u32 op | u32 k | u32 s | u32 p | rows*iw f32 in | rows*ow f32 out):
 *   kind 0 = ggml_pool_1d(F32, op, k, s, p)   — the op under test
 *   kind 1 = ggml_pool_1d(F16 src, AVG/MAX)   — the FP16 read path
 *   kind 2 = ggml_pool_2d(AVG, k=2, k1=1, s=2, s1=1, p=0, p1=0) on the same
 *            input — the composition the whisper-enc round used before the
 *            literal op existed; kind 0 (k2/s2/p0 AVG) must equal it bitwise.
 * EOF: magic 0xFFFFFFFF.
 *
 * Build (against the reference build's shared libs):
 *   gcc -O2 -o parity/ref_pool1d_dump parity/ref_pool1d_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_pool1d_dump parity/pool1d_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x24635342u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 6.0f - 3.0f;
}

static uint16_t f16_lcg = 0;
static uint16_t next_f16(void) {
    lcg = lcg * 1103515245u + 12345u;
    float v = ((float)(int32_t)lcg / (float)(1 << 30)) * 4.0f - 2.0f;
    return ggml_fp32_to_fp16(v);
}

static void dump_case(FILE * f, uint32_t kind, int op, int k, int s, int p,
                      int rows, int iw, int use_f16) {
    /* output width the C builder produces (ggml.c:5065) — float division,
     * truncated toward zero after the +1 (see main) */
    int ow = (int)(((float)((int64_t) iw + 2 * p - k) / (float) s) + 1.0f);

    size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                      (size_t) rows * (iw + ow) * 4 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    /* a [iw, rows] 2-D tensor (row = one pool row, exactly what the graphs
     * feed: cont(transpose(x)) of [n_embd, n_pos]) */
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx,
        use_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, iw, rows);

    float * fin = malloc((size_t) rows * iw * sizeof(float));
    for (int i = 0; i < rows * iw; i++) {
        fin[i] = next_val();
    }
    if (use_f16) {
        uint16_t * h = (uint16_t *) a->data;
        for (int i = 0; i < rows * iw; i++) {
            h[i] = ggml_fp32_to_fp16(fin[i]);
        }
    } else {
        memcpy(a->data, fin, (size_t) rows * iw * sizeof(float));
    }

    struct ggml_tensor * r;
    if (kind == 2) {
        /* the pre-POOL_1D whisper composition (pool_2d with a degenerate 2nd
         * axis) — always AVG 2/2/0 */
        r = ggml_pool_2d(ctx, a, GGML_OP_POOL_AVG, 2, 1, 2, 1, 0.0f, 0.0f);
    } else {
        r = ggml_pool_1d(ctx, a, (enum ggml_op_pool) op, k, s, p);
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed %d\n", (int) st);
        exit(1);
    }

    uint32_t magic = 0x52443050u; /* "P1DR" little-endian */
    uint32_t hdr[8] = { kind, (uint32_t) rows, (uint32_t) iw, (uint32_t) ow,
                        (uint32_t) op, (uint32_t) k, (uint32_t) s, (uint32_t) p };
    fwrite(&magic, 4, 1, f);
    fwrite(hdr, 4, 8, f);
    fwrite(fin, 4, (size_t) rows * iw, f);
    fwrite(r->data, 4, (size_t) rows * ow, f);
    free(fin);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * out = "pool1d_ref.bin";
    if (argc > 1) {
        out = argv[1];
    }
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* (k, s, p) grids that cover: full windows (count == k), clipped left
     * edge, clipped right edge, fully-skipped windows (count == 0 -> AVG 0),
     * stride > kernel (gaps), and the exact whisper shape (2, 2, 0) */
    static const int ksp[][3] = {
        {2, 2, 0}, /* the whisper-enc nn.AvgPool1d(2, stride=2) */
        {1, 1, 0}, /* identity */
        {2, 1, 0}, /* overlapping windows */
        {3, 1, 0},
        {3, 1, 1}, /* symmetric pad, even width (clips both edges) */
        {3, 2, 0},
        {3, 2, 1},
        {4, 4, 2},
        {5, 3, 2},
        {2, 3, 0}, /* stride > kernel: count == 1 windows */
        {8, 8, 0}, /* k == iw: single window */
        {9, 4, 0}, /* k > iw: count == 0 windows on short rows */
        {7, 5, 3},
        {6, 2, 1},
    };
    static const int widths[] = { 7, 8, 9, 16, 31, 100, 1, 2, 3, 13, 64 };
    static const int rowsv[]  = { 1, 3 };

    for (int gi = 0; gi < (int)(sizeof(ksp) / sizeof(ksp[0])); gi++) {
        for (int wi = 0; wi < (int)(sizeof(widths) / sizeof(widths[0])); wi++) {
            int iw = widths[wi];
            int k = ksp[gi][0], s = ksp[gi][1], p = ksp[gi][2];
            /* replicate ggml_calc_pool_output_size exactly: `2 * p` is float,
             * so the numerator (and the division) happen in FLOAT and the
             * result is truncated toward zero AFTER the +1 — differs from
             * integer division when |iw + 2p - k| < s and iw + 2p - k < 0 */
            int ow64 = (int)(((float)((int64_t) iw + 2 * p - k) / (float) s) + 1.0f);
            if (ow64 <= 0) {
                continue; /* builder asserts ne[0] > 0 */
            }
            for (int op = 0; op < 2; op++) {
                dump_case(f, 0, op, k, s, p, rowsv[gi & 1], iw, 0);
            }
            /* F16 read path on a subset (AVG + MAX, short widths) */
            if (wi < 5) {
                dump_case(f, 1, 0, k, s, p, 1, iw, 1);
                dump_case(f, 1, 1, k, s, p, 2, iw, 1);
            }
            /* the old whisper composition, only for the (2,2,0) case */
            if (k == 2 && s == 2 && p == 0) {
                dump_case(f, 2, 0, 2, 2, 0, rowsv[gi & 1], iw, 0);
            }
        }
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}
