/* Reference quant dump harness — produces ground-truth bytes/floats from the
 * pinned llama.cpp (bd4f514db1) ggml-base for bit-exact Rust parity tests.
 *
 * Output format (little-endian), per section:
 *   u32 magic 0x54434553 ("SECT") | u32 ggml_type_id | u64 n_elements |
 *   u64 quantized_byte_len | quantized bytes | n * f32 dequantized values
 * EOF: u32 type_id 0xFFFF.
 *
 * Build:
 *   gcc ref_quants_dump.c -I$PINNED/ggml/src -o ref_quants_dump \
 *       -L$REF/bin -lggml-base -lm -Wl,-rpath,$REF/bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_IMPL
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"

/* IQ2_XXS / IQ2_XS / IQ1_S refuse a NULL imatrix (ggml_quantize_requires_imatrix);
 * feed them a synthetic all-ones imatrix instead. */
static const float *iq_imatrix_for(uint32_t tid, const float *ones) {
    return (tid == 16 || tid == 17 || tid == 19) ? ones : NULL;
}


static uint32_t lcg_state = 0x12345678u;
static float next_val(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg_state / (float)(1 << 30)) * 0.75f - 0.125f;
}

#define N 1024 /* multiple of QK_K=256 */

static void write_section(FILE *f, uint32_t type_id, const void *qbytes, uint64_t qlen, const float *deq) {
    uint32_t magic = 0x54434553u;
    uint64_t n = N;
    fwrite(&magic, 4, 1, f);
    fwrite(&type_id, 4, 1, f);
    fwrite(&n, 8, 1, f);
    fwrite(&qlen, 8, 1, f);
    fwrite(qbytes, 1, qlen, f);
    fwrite(deq, 4, N, f);
}

#define DUMP(TID, BT, QFN, DFN, BLCK)                                          \
    do {                                                                       \
        BT *y = calloc(N / (BLCK), sizeof(BT));                                \
        float *d = malloc(N * sizeof(float));                                  \
        QFN(x, y, N);                                                          \
        DFN(y, d, N);                                                          \
        write_section(f, (TID), y, (uint64_t)(N / (BLCK)) * sizeof(BT), d);    \
        free(y); free(d);                                                      \
    } while (0)

/* q8_1 has no dequantize fn; reuse quantize + manual dequant is skipped — the
 * Rust side tests q8_1 quantize bytes only. */
#define DUMP_QONLY(TID, BT, QFN, BLCK)                                         \
    do {                                                                       \
        BT *y = calloc(N / (BLCK), sizeof(BT));                                \
        float *d = malloc(N * sizeof(float));                                  \
        memset(d, 0, N * sizeof(float));                                       \
        QFN(x, y, N);                                                          \
        write_section(f, (TID), y, (uint64_t)(N / (BLCK)) * sizeof(BT), d);    \
        free(y); free(d);                                                      \
    } while (0)

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "quants_ref.bin";
    float *x = malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) x[i] = next_val();

    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    DUMP(2,  block_q4_0, quantize_row_q4_0_ref, dequantize_row_q4_0, QK4_0);
    DUMP(3,  block_q4_1, quantize_row_q4_1_ref, dequantize_row_q4_1, QK4_1);
    DUMP(6,  block_q5_0, quantize_row_q5_0_ref, dequantize_row_q5_0, QK5_0);
    DUMP(7,  block_q5_1, quantize_row_q5_1_ref, dequantize_row_q5_1, QK5_1);
    DUMP(8,  block_q8_0, quantize_row_q8_0_ref, dequantize_row_q8_0, QK8_0);
    DUMP_QONLY(9, block_q8_1, quantize_row_q8_1_ref, QK8_1);
    DUMP(10, block_q2_K, quantize_row_q2_K_ref, dequantize_row_q2_K, QK_K);
    DUMP(11, block_q3_K, quantize_row_q3_K_ref, dequantize_row_q3_K, QK_K);
    DUMP(12, block_q4_K, quantize_row_q4_K_ref, dequantize_row_q4_K, QK_K);
    DUMP(13, block_q5_K, quantize_row_q5_K_ref, dequantize_row_q5_K, QK_K);
    DUMP(14, block_q6_K, quantize_row_q6_K_ref, dequantize_row_q6_K, QK_K);
    DUMP_QONLY(15, block_q8_K, quantize_row_q8_K_ref, QK_K);

    uint32_t end = 0xFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);

    /* ---- i-quant sections (agent D) ------------------------------------
     * Written to a SEPARATE file quants_ref_iq.bin so that quants_ref.bin —
     * whose section count (12) is asserted by the pre-existing quants.rs
     * parity test — stays byte-identical. Same on-disk section format.
     *
     * Quantized bytes come from ggml_quantize_chunk with a NULL imatrix,
     * except IQ2_XXS / IQ2_XS / IQ1_S whose quantizers assert on a missing
     * imatrix (see ggml_quantize_requires_imatrix); those get a synthetic
     * all-ones imatrix. Dequantization uses the generic scalar *_ref fns. */
    const char *out2 = argc > 2 ? argv[2] : "quants_ref_iq.bin";
    FILE *g = fopen(out2, "wb");
    if (!g) { perror("fopen iq"); return 1; }

    float *ones = malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) ones[i] = 1.f;

    {   /* each type: quantize via chunk, then dequantize with the generic fn */
        #define IQ_CASE(TID, BT, DFN)                                              \
            do {                                                                   \
                uint64_t row_size = ggml_row_size((enum ggml_type)(TID), N);       \
                BT *y = calloc(1, row_size);                                       \
                float *d = malloc(N * sizeof(float));                              \
                ggml_quantize_chunk((enum ggml_type)(TID), x, y, 0, 1, N,          \
                                    iq_imatrix_for((TID), ones));                 \
                DFN(y, d, N);                                                      \
                write_section(g, (TID), y, row_size, d);                           \
                free(y); free(d);                                                  \
            } while (0)
        IQ_CASE(16, block_iq2_xxs, dequantize_row_iq2_xxs); /* ones imatrix */
        IQ_CASE(17, block_iq2_xs,  dequantize_row_iq2_xs);  /* ones imatrix */
        IQ_CASE(18, block_iq3_xxs, dequantize_row_iq3_xxs);
        IQ_CASE(19, block_iq1_s,   dequantize_row_iq1_s);   /* ones imatrix */
        IQ_CASE(20, block_iq4_nl,  dequantize_row_iq4_nl);
        IQ_CASE(21, block_iq3_s,   dequantize_row_iq3_s);
        IQ_CASE(22, block_iq2_s,   dequantize_row_iq2_s);
        IQ_CASE(23, block_iq4_xs,  dequantize_row_iq4_xs);
        IQ_CASE(29, block_iq1_m,   dequantize_row_iq1_m);
        #undef IQ_CASE
    }

    fwrite(&end, 4, 1, g);
    fclose(g);
    free(ones);
    printf("wrote %s\n", out2);

    /* ---- Q1_0 / Q2_0 sections (their own file, same reason as the iq file:
     * quants_ref.bin's 12-section count is asserted by the pre-existing test).
     * Both types are new in this revision and are the only ggml quantizers the
     * Rust port had gated off, so they get their own ground truth. */
    const char *out3 = argc > 3 ? argv[3] : "quants_ref_q1q2.bin";
    FILE *h = fopen(out3, "wb");
    if (!h) { perror("fopen q1q2"); return 1; }

    DUMP(41, block_q1_0, quantize_row_q1_0_ref, dequantize_row_q1_0, QK1_0);
    DUMP(42, block_q2_0, quantize_row_q2_0_ref, dequantize_row_q2_0, QK2_0);

    fwrite(&end, 4, 1, h);
    fclose(h);
    printf("wrote %s\n", out3);
    return 0;
}
