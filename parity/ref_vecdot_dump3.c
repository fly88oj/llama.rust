/* vec_dot ground truth v3 — the AUDIT_ggml.md §5-A.1 round: the 12 types that
 * had no vec_dot in the Rust port (Q1_0/Q2_0/NVFP4/IQ2_XXS/IQ2_XS/IQ2_S/
 * IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_NL/IQ4_XS).
 *
 * Same harness discipline as ref_vecdot_dump2.c: dlsym the ACTUAL x86 kernels
 * from libggml-cpu.so.0 (the reference build, -march=native + GGML_AVX512 off
 * => the `#if defined(__AVX2__)` bodies). Weight bytes are synthetic (LCG)
 * with the float fields forced finite — the IQ quantizers need imatrix-side
 * machinery the harness does not replicate, and the kernel contract is over
 * block bytes, which need not come from a quantizer. y comes from the
 * reference runtime activation quantizers like every other dump.
 *
 * Section: u32 'VDD3' | u32 type_id | u32 n | xq bytes | yq bytes | f32 result
 * EOF 0xFFFFFFFF.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"

typedef void (*vdfn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*qfn)(const float *, void *, int64_t);
extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0x5eed1234u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 1.8f - 0.9f;
}
static uint8_t next_byte(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (uint8_t)(lcg >> 13);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "vecdot3_ref.bin";
    const char *lib = "/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/libggml-cpu.so.0";
    void *h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    /* the kernels convert fp16 scales through ggml_table_f32_f16, which
     * ggml_cpu_init() fills — without it every d reads as 0 and the dump is
     * garbage (this is what silently zeroed the earlier vecdot_ref.bin) */
    void (*cpu_init)(void) = (void (*)(void))dlsym(h, "ggml_cpu_init");
    if (!cpu_init) { fprintf(stderr, "dlsym missing ggml_cpu_init\n"); return 1; }
    cpu_init();

    /* (ggml type id, vec_dot kernel, activation quantizer, vec_dot_type id) */
    struct spec { int tid; const char *vd; const char *yq; int vt; } specs[] = {
        {41, "ggml_vec_dot_q1_0_q8_0",   "quantize_row_q8_0", 8},
        {42, "ggml_vec_dot_q2_0_q8_0",   "quantize_row_q8_0", 8},
        {40, "ggml_vec_dot_nvfp4_q8_0",  "quantize_row_q8_0", 8},
        {16, "ggml_vec_dot_iq2_xxs_q8_K", "quantize_row_q8_K", 15},
        {17, "ggml_vec_dot_iq2_xs_q8_K",  "quantize_row_q8_K", 15},
        {22, "ggml_vec_dot_iq2_s_q8_K",   "quantize_row_q8_K", 15},
        {18, "ggml_vec_dot_iq3_xxs_q8_K", "quantize_row_q8_K", 15},
        {21, "ggml_vec_dot_iq3_s_q8_K",   "quantize_row_q8_K", 15},
        {19, "ggml_vec_dot_iq1_s_q8_K",   "quantize_row_q8_K", 15},
        {29, "ggml_vec_dot_iq1_m_q8_K",   "quantize_row_q8_K", 15},
        {20, "ggml_vec_dot_iq4_nl_q8_0",  "quantize_row_q8_0", 8},
        {23, "ggml_vec_dot_iq4_xs_q8_K",  "quantize_row_q8_K", 15},
    };

    const int n = 1024;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    for (int i = 0; i < n; i++) { x[i] = next_val(); y[i] = next_val(); }

    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }
    for (unsigned s = 0; s < sizeof(specs) / sizeof(specs[0]); s++) {
        struct spec *sp = &specs[s];
        qfn qy = (qfn)dlsym(h, sp->yq);
        vdfn vd = (vdfn)dlsym(h, sp->vd);
        if (!qy || !vd) { fprintf(stderr, "dlsym missing %s\n", sp->vd); return 1; }
        size_t xs = ggml_row_size((enum ggml_type)sp->tid, n);
        size_t ys = ggml_row_size((enum ggml_type)sp->vt, n);
        void *xq = malloc(xs), *yq = malloc(ys);
        uint8_t *xb = (uint8_t *)xq;
        for (size_t i = 0; i < xs; ++i) xb[i] = next_byte();
        size_t blck = ggml_blck_size((enum ggml_type)sp->tid);
        size_t tsz = ggml_type_size((enum ggml_type)sp->tid);
        size_t nb = n / blck;
        for (size_t ib = 0; ib < nb; ++ib) {
            uint8_t *b = xb + ib * tsz;
            if (sp->tid == 29) {
                /* iq1_m: no fp16 d; the scale u16 is nibble-assembled from the
                 * 4 scales u16s — force it to fp16 0x3C00 (1.0) so the result
                 * stays finite: sc[0] bits 12-15 = 3, sc[1] bits 8-11 = C. */
                uint8_t *sc = b;
                uint16_t *sc16 = (uint16_t *)sc;
                sc16[0] = (sc16[0] & 0x0fff) | 0x3000;
                sc16[1] = (sc16[1] & 0xf0ff) | 0x0c00;
                sc16[2] = sc16[2] & 0xff0f;
                sc16[3] = sc16[3] & 0xfff0;
            } else if (sp->tid != 40) {
                /* every other type keeps an fp16 d in the first two bytes:
                 * force a finite, small magnitude (exponent 0x1C..0x1E). */
                b[1] = (b[1] & 0x03) | 0x38;
            }
        }
        qy(y, yq, n);
        float res = 0;
        ((vdfn)vd)(n, &res, sizeof(float), xq, 0, yq, 0, 1);
        uint32_t magic = 0x33444456, tyy = (uint32_t)sp->tid, nn = (uint32_t)n;
        fwrite(&magic, 4, 1, f); fwrite(&tyy, 4, 1, f); fwrite(&nn, 4, 1, f);
        fwrite(xq, 1, xs, f); fwrite(yq, 1, ys, f); fwrite(&res, 4, 1, f);
        printf("type %2d -> %.6f\n", sp->tid, res);
        free(xq); free(yq);
    }
    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
