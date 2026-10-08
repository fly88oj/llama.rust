/* Quantized vec_dot ground-truth dumper — calls the ACTUAL exported x86
 * kernels + runtime activation quantizers directly (no traits struct).
 *
 * Section: u32 magic 'VDD1' | u32 ggml_type_id | u32 n(=1024) |
 *   quantized x bytes | quantized y bytes | result f32.  EOF 0xFFFFFFFF.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

typedef void (*vdfn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*qfn)(const float *, void *, int64_t);
extern void quantize_row_q8_0(const float *, void *, int64_t);
extern void quantize_row_q8_1(const float *, void *, int64_t);
extern void quantize_row_q8_K(const float *, void *, int64_t);
extern void quantize_row_q4_0_ref(const float *, void *, int64_t);
extern void quantize_row_q4_1_ref(const float *, void *, int64_t);
extern void quantize_row_q5_0_ref(const float *, void *, int64_t);
extern void quantize_row_q5_1_ref(const float *, void *, int64_t);
extern void quantize_row_q8_0_ref(const float *, void *, int64_t);
extern void quantize_row_q2_K_ref(const float *, void *, int64_t);
extern void quantize_row_q3_K_ref(const float *, void *, int64_t);
extern void quantize_row_q4_K_ref(const float *, void *, int64_t);
extern void quantize_row_q5_K_ref(const float *, void *, int64_t);
extern void quantize_row_q6_K_ref(const float *, void *, int64_t);
extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0x5eed1234u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 1.8f - 0.9f;
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "vecdot_ref.bin";
    const int n = 1024;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    for (int i = 0; i < n; i++) { x[i] = next_val(); y[i] = next_val(); }

    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

#define CASE(TID, QREF, VD, RUNTIME_Q, VT)                                   \
    do {                                                                     \
        size_t xs = ggml_row_size((enum ggml_type)(TID), n);                 \
        size_t ys = ggml_row_size((enum ggml_type)(VT), n);                  \
        void *xq = malloc(xs), *yq = malloc(ys);                             \
        QREF(x, xq, n);                                                      \
        RUNTIME_Q(y, yq, n);                                                 \
        float s = 0;                                                         \
        ((vdfn)(VD))(n, &s, sizeof(float), xq, 0, yq, 0, 1);                 \
        uint32_t magic = 0x31444456, tyy = (TID), nn = n;                    \
        fwrite(&magic, 4, 1, f); fwrite(&tyy, 4, 1, f); fwrite(&nn, 4, 1, f);\
        fwrite(xq, 1, xs, f); fwrite(yq, 1, ys, f); fwrite(&s, 4, 1, f);     \
        printf("type %2d -> %.6f\n", (TID), s);                              \
        free(xq); free(yq);                                                  \
    } while (0)

    extern void ggml_vec_dot_q4_0_q8_0(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q4_1_q8_1(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q5_0_q8_0(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q5_1_q8_1(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q8_0_q8_0(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q2_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q3_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q4_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q5_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    extern void ggml_vec_dot_q6_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);

    CASE(2,  quantize_row_q4_0_ref, ggml_vec_dot_q4_0_q8_0, quantize_row_q8_0, 8);
    CASE(3,  quantize_row_q4_1_ref, ggml_vec_dot_q4_1_q8_1, quantize_row_q8_1, 9);
    CASE(6,  quantize_row_q5_0_ref, ggml_vec_dot_q5_0_q8_0, quantize_row_q8_0, 8);
    CASE(7,  quantize_row_q5_1_ref, ggml_vec_dot_q5_1_q8_1, quantize_row_q8_1, 9);
    CASE(8,  quantize_row_q8_0_ref, ggml_vec_dot_q8_0_q8_0, quantize_row_q8_0, 8);
    CASE(10, quantize_row_q2_K_ref, ggml_vec_dot_q2_K_q8_K, quantize_row_q8_K, 15);
    CASE(11, quantize_row_q3_K_ref, ggml_vec_dot_q3_K_q8_K, quantize_row_q8_K, 15);
    CASE(12, quantize_row_q4_K_ref, ggml_vec_dot_q4_K_q8_K, quantize_row_q8_K, 15);
    CASE(13, quantize_row_q5_K_ref, ggml_vec_dot_q5_K_q8_K, quantize_row_q8_K, 15);
    CASE(14, quantize_row_q6_K_ref, ggml_vec_dot_q6_K_q8_K, quantize_row_q8_K, 15);

    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
