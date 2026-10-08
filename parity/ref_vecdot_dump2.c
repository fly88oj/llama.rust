/* vec_dot ground truth v2 — dlsym the kernels from libggml-cpu explicitly
 * (libggml-base exports same-named quantize symbols; interposition ambiguated
 * the earlier harness). Quantized inputs are dumped so Rust reuses them
 * byte-for-byte — no cross-lib quantizer ambiguity at all.
 *
 * Section: u32 'VDD2' | u32 type_id | u32 n | xq bytes | yq bytes | f32 result
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

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "vecdot_ref.bin";
    const char *lib = "/home/jeffrey/llm/llama.cpp/build-rust-ref/bin/libggml-cpu.so.0";
    void *h = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    struct spec { int tid; const char *qname; const char *vdname; const char *yname; int vt; } specs[] = {
        {2,  "quantize_row_q4_0_ref", "ggml_vec_dot_q4_0_q8_0", "quantize_row_q8_0", 8},
        {3,  "quantize_row_q4_1_ref", "ggml_vec_dot_q4_1_q8_1", "quantize_row_q8_1", 9},
        {6,  "quantize_row_q5_0_ref", "ggml_vec_dot_q5_0_q8_0", "quantize_row_q8_0", 8},
        {7,  "quantize_row_q5_1_ref", "ggml_vec_dot_q5_1_q8_1", "quantize_row_q8_1", 9},
        {8,  "quantize_row_q8_0_ref", "ggml_vec_dot_q8_0_q8_0", "quantize_row_q8_0", 8},
        {10, "quantize_row_q2_K_ref", "ggml_vec_dot_q2_K_q8_K", "quantize_row_q8_K", 15},
        {11, "quantize_row_q3_K_ref", "ggml_vec_dot_q3_K_q8_K", "quantize_row_q8_K", 15},
        {12, "quantize_row_q4_K_ref", "ggml_vec_dot_q4_K_q8_K", "quantize_row_q8_K", 15},
        {13, "quantize_row_q5_K_ref", "ggml_vec_dot_q5_K_q8_K", "quantize_row_q8_K", 15},
        {14, "quantize_row_q6_K_ref", "ggml_vec_dot_q6_K_q8_K", "quantize_row_q8_K", 15},
    };

    const int n = 1024;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    for (int i = 0; i < n; i++) { x[i] = next_val(); y[i] = next_val(); }

    FILE *f = fopen(out, "wb");
    for (unsigned s = 0; s < sizeof(specs) / sizeof(specs[0]); s++) {
        struct spec *sp = &specs[s];
        qfn qx = (qfn)dlsym(h, sp->qname);
        qfn qy = (qfn)dlsym(h, sp->yname);
        vdfn vd = (vdfn)dlsym(h, sp->vdname);
        if (!qx || !qy || !vd) { fprintf(stderr, "dlsym missing %s\n", sp->qname); return 1; }
        size_t xs = ggml_row_size((enum ggml_type)sp->tid, n);
        size_t ys = ggml_row_size((enum ggml_type)sp->vt, n);
        void *xq = malloc(xs), *yq = malloc(ys);
        qx(x, xq, n);
        qy(y, yq, n);
        float res = 0;
        vd(n, &res, sizeof(float), xq, 0, yq, 0, 1);
        if (s == 0) {
            printf("debug: xs=%zu ys=%zu xq[0..8]=", xs, ys);
            for (int i = 0; i < 8; i++) printf("%02x ", ((unsigned char *)xq)[i]);
            printf(" yq[0..8]=");
            for (int i = 0; i < 8; i++) printf("%02x ", ((unsigned char *)yq)[i]);
            printf("\n  x[0..4]=%f %f %f %f  d-f16-le=%04x\n", x[0], x[1], x[2], x[3],
                   (unsigned)(((uint16_t *)xq)[0]));
        }
        uint32_t magic = 0x32444456, tid = sp->tid, nn = n;
        fwrite(&magic, 4, 1, f); fwrite(&tid, 4, 1, f); fwrite(&nn, 4, 1, f);
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
