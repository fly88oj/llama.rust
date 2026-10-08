/* ggml_gelu_erf ground-truth dumper — drives the reference build's
 * ggml_gelu_erf through the real graph-compute path and writes bit-exact
 * outputs.
 *
 * Why the graph path: ggml_vec_gelu_erf_f32 is `inline static` in vec.h (no
 * exported symbol), and the unary op is dispatched inside ggml-cpu.so.
 * Building the tiny graph and calling ggml_graph_compute_with_ctx
 * reproduces exactly what a model graph does (ggml-cpu.c:3476).
 *
 * Ground truth for: ggml_gelu_erf  (ggml.c:2796 → ops.cpp:10180 →
 *                              ops.cpp:2436 ggml_compute_forward_gelu_erf →
 *                              vec.h:1010 ggml_vec_gelu_erf_f32 —
 *                              a plain scalar libm erff loop; no
 *                              ggml_v_gelu_erf SIMD variant exists at this
 *                              commit, and erf is not in libmvec's default
 *                              auto-vectorization set)
 *
 * Format (per section):
 *   u32 magic 'EGRF' | u32 kind | u32 n | payload
 *   kind 0 = ggml_gelu_erf(f32) : n f32 in | n f32 out
 *   kind 3 = raw formula (f32)  : n f32 in | n f32 out
 *   kind 7 = ggml_gelu_erf(f16) : n u16 in | n u16 out
 * EOF: magic 0xFFFFFFFF
 *
 * Build (against the reference build's shared libs; the dumper itself is a
 * plain -O2 build — the *library* was built -O3 -march=native, which is
 * what matters):
 *   gcc -O2 -o parity/ref_gelu_erf_dump parity/ref_gelu_erf_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L$REF/bin -Wl,-rpath,$REF/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ($REF = /home/jeffrey/llm/llama.cpp/build-rust-ref)
 *   ./parity/ref_gelu_erf_dump parity/gelu_erf_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

static const float SQRT_2_INV = 0.70710678118654752440084436210484f;

static uint32_t lcg = 0x6c078965u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 2.0f - 1.0f;
}

/* --- edge cases: the erff polynomial/double-path switches, gelu_erf's
 *     saturation plateaus (erf -> +-1, so y -> 0 / y -> x), denormals,
 *     inf/nan propagation through 0.5*x*(1 + erf) --- */
static const uint32_t specials_bits[] = {
    0x00000000u, /* +0 */
    0x80000000u, /* -0 */
    0x00000001u, /* +denormal min */
    0x007fffffu, /* +max denormal */
    0x00800000u, /* +min normal */
    0x80000001u, /* -denormal min */
    0x3f800000u, /* 1.0 */
    0xbf800000u, /* -1.0 */
    0x3f000000u, /* 0.5 */
    0xbf000000u, /* -0.5 */
    0x40000000u, /* 2.0 */
    0xc0000000u, /* -2.0 */
    0x40490fdbu, /* pi */
    0x40a00000u, /* 5.0 */
    0xc0a00000u, /* -5.0 */
    0x41200000u, /* 10.0 */
    0xc1200000u, /* -10.0 */
    0x41700000u, /* 15.0 */
    0xc1700000u, /* -15.0 */
    0x41a00000u, /* 20.0 */
    0xc1a00000u, /* -20.0 */
    0x42340000u, /* 45.0 (erff saturates to 1 well before) */
    0x42480000u, /* 50.0 */
    0x42c80000u, /* 100.0 */
    0xc2c80000u, /* -100.0 */
    0x477fff00u, /* 65535.0 */
    0x7f7fffffu, /* FLT_MAX */
    0xff7fffffu, /* -FLT_MAX */
    0x7f800000u, /* +inf -> x */
    0xff800000u, /* -inf -> -inf*0 = NaN? erf(-inf)=-1 so 0.5*-inf*0 = NaN */
    0x7fc00000u, /* qnan -> qnan */
    0xffc00000u, /* -qnan */
    0x7f800001u, /* snan */
    0x39000000u, /* 4.88e-4 */
    0x33000000u, /* 2.98e-8 */
    0x0c000000u, /* tiny */
    0x3dccccccu, /* 0.1 */
    0x3f000001u, /* 0.50000006 */
    0x3effffffu, /* 0.49999997 */
    0x3f7ffffeu, /* 0.9999999 */
    0x3f800001u, /* 1.0000001 */
    0x40400000u, /* 3.0 */
    0xbfc00000u, /* -1.5 */
    0x3f490fdbu, /* 0.7853981 (pi/4) */
    0xbf490fdbu, /* -0.7853981 */
    0x3eaaaaabu, /* 0.33333334 (erff series region) */
    0xbeaaaaabu, /* -0.33333334 */
    0x3ec00000u, /* 0.375 */
    0x3f400000u, /* 0.75 */
    0xbf400000u, /* -0.75 */
};

/* Run nn elements of in[] through the reference ggml_gelu_erf and write out[]. */
static void run_gelu_erf_f32(FILE *f, uint32_t kind, const float *in, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * sizeof(float) * 2 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(a->data, in, (size_t) n * sizeof(float));

    struct ggml_tensor * r = ggml_gelu_erf(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    uint32_t magic = 0x46524745u; /* "EGRF" little-endian */
    uint32_t nn = (uint32_t) n;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 4, n, f);
    fwrite(r->data, 4, n, f);

    ggml_free(ctx);
}

/* Scalar reference computed by the same libm the reference .so links:
 *   kind 3 = the raw ggml_vec_gelu_erf_f32 body (vec.h:1010-1015). The op
 *   output (kind 0) must equal it bit-for-bit — that is the proof that this
 *   build has no SIMD/libmvec erf path. The expression is plain multiplies
 *   plus one add over a call result, so -O2/-O3 -ffp-contract cannot fuse
 *   anything. */
static void dump_scalar_formula(FILE *f, uint32_t kind, const float *in, int n) {
    uint32_t magic = 0x46524745u;
    uint32_t nn = (uint32_t) n;
    float * out = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float xi = in[i];
        out[i] = 0.5f * xi * (1.0f + erff(xi * SQRT_2_INV));
    }
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 4, n, f);
    fwrite(out, 4, n, f);
    free(out);
}

/* kind 7 = ggml_gelu_erf on an F16 tensor (ops.cpp:2449 dispatch →
 * ggml_compute_forward_gelu_erf_f16 → ggml_vec_gelu_erf_f16, i.e.
 * fp16(0.5*xi*(1+erff(xi*sqrt(1/2))) with xi = fp16_to_f32(x))).
 * Payload: n u16 input bits then n u16 output bits. */
static void run_gelu_erf_f16(FILE *f, const uint16_t *in, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 2 * 2 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    memcpy(a->data, in, (size_t) n * 2);
    struct ggml_tensor * r = ggml_gelu_erf(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    uint32_t magic = 0x46524745u, kind = 7, nn = (uint32_t) n;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 2, n, f);
    fwrite(r->data, 2, n, f);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    const char * out = "gelu_erf_ref.bin";
    if (argc > 1) {
        out = argv[1];
    }
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* LCG lengths: 16-multiples, non-multiples, tiny, large — inputs in
     * [-1, 1) land in erff's polynomial region */
    int lens[] = {1024, 1000, 63, 16, 17, 1, 33, 255, 1023, 4096, 47, 15};
    for (int li = 0; li < (int)(sizeof(lens)/sizeof(lens[0])); li++) {
        int n = lens[li];
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = next_val();
        run_gelu_erf_f32(f, 0, x, n);      /* ggml_gelu_erf */
        dump_scalar_formula(f, 3, x, n);   /* raw formula (same libm) */
        free(x);
    }

    /* the whisper-enc activation range and the erf saturation approach:
     * conv/mlp outputs span a few units, the erf argument x/sqrt(2) reaches
     * ±2.8 at |x| = 4 (erf(2.8) ~ 0.99999) and the plateaus by |x| ~ 6 */
    {
        int n = 512;
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            switch (i & 3) {
                case 0: x[i] = next_val() * 8.0f; break;
                case 1: x[i] = next_val() * 4.0f; break;
                case 2: x[i] = next_val() * 0.5f; break;
                default: x[i] = next_val() * 27.0f; break;
            }
        }
        run_gelu_erf_f32(f, 0, x, n);
        dump_scalar_formula(f, 3, x, n);
        free(x);
    }

    /* edge-case sweep */
    {
        int n = (int)(sizeof(specials_bits)/sizeof(specials_bits[0]));
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            memcpy(&x[i], &specials_bits[i], 4);
        }
        run_gelu_erf_f32(f, 0, x, n);
        dump_scalar_formula(f, 3, x, n);
        free(x);
    }

    /* F16 coverage: LCG in [-1,1), scaled to the saturation range, the f16
     * specials (subnormals, ±inf, NaNs, max) and the exact f16 grid points
     * 0.5/1/2/4 uLP apart near the erf switch regions. */
    {
        int n = 600;
        uint16_t * h = malloc(n * 2);
        int k = 0;
        for (int i = 0; i < 256; i++) {
            float v = next_val();
            h[k++] = ggml_fp32_to_fp16(v);           /* [-1,1) */
            h[k++] = ggml_fp32_to_fp16(v * 8.0f);
        }
        static const uint16_t hsp[] = {
            0x0000, 0x8000, 0x0001, 0x0003, 0x00ff,  /* ±0, subnormals */
            0x0100, 0x8100,                          /* min normal ± */
            0x3800, 0xb800,                          /* ±0.5 */
            0x3c00, 0xbc00,                          /* ±1 */
            0x4000, 0xc000,                          /* ±2 */
            0x4200, 0xc200,                          /* ±8 */
            0x4800, 0xc800,                          /* ±64 (plateau) */
            0x7bff, 0xfbff,                          /* ±65504 */
            0x7c00, 0xfc00,                          /* ±inf */
            0x7e00, 0xfe00,                          /* qNaNs */
            0x3555, 0xb555,                          /* ±0.3333 */
            0x3b07, 0xbb07,                          /* ±0.8789-ish */
        };
        for (int i = 0; i < (int)(sizeof(hsp)/sizeof(hsp[0])); i++) {
            h[k++] = hsp[i];
        }
        while (k < n) { h[k++] = ggml_fp32_to_fp16(next_val() * 16.0f); }
        run_gelu_erf_f16(f, h, n);
        free(h);
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}
