/* MXFP4 / NVFP4 reference dump harness — ground-truth bytes/floats from the
 * pinned llama.cpp (bd4f514db1) ggml-base for bit-exact Rust parity tests.
 *
 * Writes parity/quants_ref_fp4.bin (SEPARATE file: ref_quants_dump.c's
 * quants_ref.bin / quants_ref_iq.bin and their section counts stay untouched).
 *
 * Section format (little-endian), identical layout to ref_quants_dump.c:
 *   u32 magic | u32 ggml_type_id | u64 n_elements | u64 quantized_byte_len |
 *   quantized bytes | n * f32 dequantized values
 * EOF: u32 0xFFFF.
 *
 * Two magics:
 *   0x54434553 ("SECT") — bytes produced by the reference quantizer
 *                         (quantize_row_mxfp4_ref / quantize_row_nvfp4_ref),
 *                         so the Rust side can also check quantize parity.
 *   0x52415753 ("SRAW") — hand-built raw block bytes exercising every E8M0
 *                         exponent / UE4M3 sub-scale / E2M1 nibble code
 *                         (dequantize parity only).
 *
 * Build:
 *   gcc parity/ref_fp4_dump.c -I$PINNED/ggml/src -I$PINNED/ggml/include \
 *       -o parity/ref_fp4_dump -L$REF/bin -lggml-base -lm \
 *       -Wl,-rpath,$REF/bin
 * Run: ./parity/ref_fp4_dump parity/quants_ref_fp4.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_IMPL
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"

#define MAGIC_SECT 0x54434553u /* quantizer-produced section */
#define MAGIC_RAW  0x52415753u /* hand-built block bytes        */

/* Same LCG as ref_quants_dump.c (and quants.rs `ref_input`) so the Rust test
 * can regenerate the inputs for any length. */
static uint32_t lcg_state;
static void lcg_reset(void) { lcg_state = 0x12345678u; }
static float next_val(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg_state / (float)(1 << 30)) * 0.75f - 0.125f;
}

static void write_section(FILE *f, uint32_t magic, uint32_t type_id, uint64_t n,
                          const void *qbytes, uint64_t qlen, const float *deq) {
    fwrite(&magic, 4, 1, f);
    fwrite(&type_id, 4, 1, f);
    fwrite(&n, 8, 1, f);
    fwrite(&qlen, 8, 1, f);
    fwrite(qbytes, 1, qlen, f);
    fwrite(deq, 4, n, f);
}

/* ---- quantizer-produced sections ------------------------------------- */

static void dump_quant(FILE *f, uint32_t tid, uint64_t n) {
    lcg_reset();
    float *x = malloc(n * sizeof(float));
    float *d = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; ++i) x[i] = next_val();

    size_t qlen = ggml_row_size((enum ggml_type)tid, (int64_t)n);
    void *qb = calloc(1, qlen);
    switch (tid) {
        case 39: quantize_row_mxfp4_ref(x, (block_mxfp4 *)qb, (int64_t)n); break;
        case 40: quantize_row_nvfp4_ref(x, (block_nvfp4 *)qb, (int64_t)n); break;
        default: fprintf(stderr, "bad tid %u\n", tid); exit(1);
    }
    switch (tid) {
        case 39: dequantize_row_mxfp4((const block_mxfp4 *)qb, d, (int64_t)n); break;
        case 40: dequantize_row_nvfp4((const block_nvfp4 *)qb, d, (int64_t)n); break;
    }
    write_section(f, MAGIC_SECT, tid, n, qb, qlen, d);
    free(x); free(d); free(qb);
}

/* ---- hand-built raw block sections ----------------------------------- */

/* Every E8M0 byte is reachable from a file; pick the interesting exponents:
 * 0/1 = 2^-128 / 2^-127 (denormal boundary), 127 = 0.5, 128 = 1.0,
 * 254/255 = 2^126 / 2^127 (largest finite / overflow-into-inf with kvalue 12). */
static void dump_mxfp4_raw(FILE *f) {
    const int64_t n = 1024;
    const int nb = (int)(n / QK_MXFP4);
    static const uint8_t es[] = { 0, 1, 2, 3, 63, 126, 127, 128, 129, 253, 254, 255 };
    block_mxfp4 *y = calloc(nb, sizeof(block_mxfp4));
    float *d = malloc(n * sizeof(float));

    for (int i = 0; i < nb; ++i) {
        y[i].e = es[i % (int)(sizeof(es) / sizeof(es[0]))];
        /* low nibble = j, high nibble = j+5 (mod 16): all 16 E2M1 codes in
         * both halves of every block. */
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            y[i].qs[j] = (uint8_t)((j & 0x0F) | (((j + 5) & 0x0F) << 4));
        }
    }
    /* explicit extremes */
    y[0].e = 0;   memset(y[0].qs, 0x00, sizeof(y[0].qs));    /* d = 2^-128, all zeros   */
    y[1].e = 255; memset(y[1].qs, 0xFF, sizeof(y[1].qs));    /* d = 2^127,  all -12 -> -inf */
    y[2].e = 1;   memset(y[2].qs, 0x88, sizeof(y[2].qs));    /* d = 2^-127, all zeros   */
    y[3].e = 128; memset(y[3].qs, 0xFF, sizeof(y[3].qs));    /* d = 1.0,    all -12     */
    y[4].e = 254; memset(y[4].qs, 0xFF, sizeof(y[4].qs));    /* d = 2^126,  all -12 -> -inf */

    dequantize_row_mxfp4(y, d, n);
    write_section(f, MAGIC_RAW, 39, (uint64_t)n, y, (uint64_t)nb * sizeof(block_mxfp4), d);
    free(y); free(d);
}

/* UE4M3 sub-block scales: 0 and 0x7F are the two special values (both -> 0),
 * exp=0 covers the subnormal range, exp=15 the top range. */
static void dump_nvfp4_raw(FILE *f) {
    const int64_t n = 1024;
    const int nb = (int)(n / QK_NVFP4);
    const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;
    static const uint8_t ds[] = { 0x00, 0x01, 0x07, 0x08, 0x09, 0x38, 0x39, 0x40, 0x76, 0x77, 0x78, 0x7E, 0x7F };
    block_nvfp4 *y = calloc(nb, sizeof(block_nvfp4));
    float *d = malloc(n * sizeof(float));

    for (int i = 0; i < nb; ++i) {
        for (int s = 0; s < n_sub; ++s) {
            y[i].d[s] = ds[(i * n_sub + s) % (int)(sizeof(ds) / sizeof(ds[0]))];
        }
        for (int j = 0; j < QK_NVFP4 / 2; ++j) {
            y[i].qs[j] = (uint8_t)((j & 0x0F) | (((j + 5) & 0x0F) << 4));
        }
    }
    /* explicit extremes */
    for (int s = 0; s < n_sub; ++s) y[0].d[s] = 0x00;
    memset(y[0].qs, 0x00, sizeof(y[0].qs));                   /* all zero scales, zero codes */
    for (int s = 0; s < n_sub; ++s) y[1].d[s] = 0x7F;
    memset(y[1].qs, 0xFF, sizeof(y[1].qs));                   /* 0x7F special-case scale -> 0 */
    for (int s = 0; s < n_sub; ++s) y[2].d[s] = 0x7E;
    memset(y[2].qs, 0xFF, sizeof(y[2].qs));                   /* max finite scale, all -12   */
    for (int s = 0; s < n_sub; ++s) y[3].d[s] = 0x08;
    memset(y[3].qs, 0xFF, sizeof(y[3].qs));                   /* scale 1.0, all -12          */
    for (int s = 0; s < n_sub; ++s) y[4].d[s] = 0x01;
    memset(y[4].qs, 0xFF, sizeof(y[4].qs));                   /* subnormal scale 2^-10       */
    for (int s = 0; s < n_sub; ++s) y[5].d[s] = 0x78;
    memset(y[5].qs, 0xFF, sizeof(y[5].qs));                   /* scale 128, all -12          */

    dequantize_row_nvfp4(y, d, n);
    write_section(f, MAGIC_RAW, 40, (uint64_t)n, y, (uint64_t)nb * sizeof(block_nvfp4), d);
    free(y); free(d);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "quants_ref_fp4.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* MXFP4 (39): 1024 = 32 blocks, plus the short round lengths 96 / 320. */
    dump_quant(f, 39, 1024);
    dump_quant(f, 39, 96);
    dump_quant(f, 39, 320);
    /* NVFP4 (40): 1024 = 16 blocks, plus 192 / 320 (64-element granularity). */
    dump_quant(f, 40, 1024);
    dump_quant(f, 40, 192);
    dump_quant(f, 40, 320);

    dump_mxfp4_raw(f);
    dump_nvfp4_raw(f);

    uint32_t end = 0xFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s (8 sections)\n", out);
    return 0;
}