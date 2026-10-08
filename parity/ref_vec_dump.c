/* vec_dot / soft_max ground-truth dumper — calls the reference AVX512 kernels
 * directly and writes bit-exact results for Rust lane-order parity tests.
 *
 * Format (per section):
 *   u32 magic 'VEC1' | u32 kind (0=f32dot,1=f16dot,2=softmaxsum) | u32 n |
 *   [n f32 x] [n f32 y] | result f32 (dot / partial-sum with max=0)
 * EOF: magic 0xFFFFFFFF
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"

/* exported from libggml-cpu (internal vec.h prototypes, re-declared here) */
extern void ggml_vec_dot_f32(int n, float *s, size_t bs, const float *x, size_t bx, const float *y, size_t by, int nrc);
extern void ggml_vec_dot_f16(int n, float *s, size_t bs, uint16_t *x, size_t bx, uint16_t *y, size_t by, int nrc);
extern double ggml_vec_soft_max_f32(const int n, float *y, const float *x, float max);
extern void ggml_cpu_fp32_to_fp16(const float *, uint16_t *, int64_t);

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
    if (exp == 0) {
        if (man == 0) { f = sign << 31; }
        else {
            uint32_t e = 0, m = man;
            while (!(m & 0x400)) { m <<= 1; e++; }
            m &= 0x3ff;
            f = (sign << 31) | ((127u - 24u + e) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) { f = (sign << 31) | 0x7f800000u | (man << 13); }
    else { f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13); }
    float out; memcpy(&out, &f, 4); return out;
}
#define GGML_CPU_FP16_TO_FP32_INLINE(h) f16_bits_to_f32((uint16_t)(h))

static uint32_t lcg = 0x9e3779b9u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 2.0f - 1.0f;
}

static void dump_section(FILE *f, uint32_t kind, const float *x, const float *y, int n, float res) {
    uint32_t magic = 0x31434556; // "VEC1"
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    uint32_t nn = n;
    fwrite(&nn, 4, 1, f);
    fwrite(x, 4, n, f);
    fwrite(y, 4, n, f);
    fwrite(&res, 4, 1, f);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "vec_ref.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* representative lengths: full 64-steps, mixtures, and leftovers */
    int lens[] = {896, 64, 128, 192, 65, 71, 151936 > 4096 ? 4096 : 4096, 63, 1, 8};
    for (int li = 0; li < (int)(sizeof(lens)/sizeof(lens[0])); li++) {
        int n = lens[li];
        float *x = malloc(n * sizeof(float));
        float *y = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) { x[i] = next_val(); y[i] = next_val(); }

        float s = 0;
        // f32 dot
        ggml_vec_dot_f32(n, &s, sizeof(float), x, 0, y, 0, 1);
        dump_section(f, 0, x, y, n, s);

        // f16 dot (convert with the reference converter)
        uint16_t *xh = malloc(n * 2), *yh = malloc(n * 2);
        ggml_cpu_fp32_to_fp16(x, xh, n);
        ggml_cpu_fp32_to_fp16(y, yh, n);
        s = 0;
        ggml_vec_dot_f16(n, &s, sizeof(float), xh, 0, yh, 0, 1);
        if (n == 65) {
            double manual = 0;
            for (int i = 0; i < n; i++) manual += GGML_CPU_FP16_TO_FP32_INLINE(xh[i]) * GGML_CPU_FP16_TO_FP32_INLINE(yh[i]);
            fprintf(stderr, "n=%d s=%f manual=%f\n", n, s, manual);
        }
        // dump f16 inputs widened back to f32 (Rust re-narrows; f16→f32→f16 is identity)
        {
            float *xw = malloc(n * 4), *yw = malloc(n * 4);
            for (int i = 0; i < n; i++) {
                uint32_t h32 = (uint32_t) xh[i];
                uint32_t sign = (h32 >> 15) & 1, exp = (h32 >> 10) & 0x1f, man = h32 & 0x3ff;
                uint32_t f;
                if (exp == 0) {
                    if (man == 0) { f = sign << 31; }
                    else { // subnormal f16 → normalize
                        uint32_t e = 0; uint32_t m = man;
                        while (!(m & 0x400)) { m <<= 1; e++; }
                        m &= 0x3ff;
                        f = (sign << 31) | ((127 - 15 - 10 - e + 1) << 23) | (m << 13);
                    }
                } else if (exp == 0x1f) {
                    f = (sign << 31) | 0x7f800000 | (man << 13);
                } else {
                    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                }
                memcpy(&xw[i], &f, 4);
                h32 = (uint32_t) yh[i]; sign = (h32 >> 15) & 1; exp = (h32 >> 10) & 0x1f; man = h32 & 0x3ff;
                if (exp == 0) {
                    if (man == 0) { f = sign << 31; }
                    else { uint32_t e = 0; uint32_t m = man; while (!(m & 0x400)) { m <<= 1; e++; } m &= 0x3ff;
                           f = (sign << 31) | ((127 - 15 - 10 - e + 1) << 23) | (m << 13); }
                } else if (exp == 0x1f) { f = (sign << 31) | 0x7f800000 | (man << 13); }
                else { f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13); }
                memcpy(&yw[i], &f, 4);
            }
            dump_section(f, 1, xw, yw, n, s);
            // also dump the raw f16 bits (u16 widened to f32-bit slots, kind 3)
            {
                float *xb = malloc(n * 4), *yb = malloc(n * 4);
                for (int i = 0; i < n; i++) {
                    uint32_t bxv = xh[i], byv = yh[i];
                    memcpy(&xb[i], &bxv, 4);
                    memcpy(&yb[i], &byv, 4);
                }
                dump_section(f, 3, xb, yb, n, s);
                free(xb); free(yb);
            }
            free(xw); free(yw);
        }

        // soft_max sum (y output written too; max param 0 → exp(x))
        float *dst = malloc(n * sizeof(float));
        double sum = ggml_vec_soft_max_f32(n, dst, x, 0.0f);
        float sumf = (float) sum;
        dump_section(f, 2, x, x, n, sumf);

        free(x); free(y); free(xh); free(yh); free(dst);
    }
    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
