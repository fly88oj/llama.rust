/* tanh / gelu ground-truth dumper — drives the reference build's ggml_tanh and
 * ggml_gelu through the real graph-compute path and writes bit-exact outputs.
 *
 * Why the graph path: ggml_vec_tanh_f32 is `inline static` in vec.h (no
 * exported symbol), and the unary op is dispatched inside ggml-cpu.so.  Building
 * the tiny graph and calling ggml_graph_compute_with_ctx reproduces exactly what
 * a model graph does (ggml-cpu.c:3476).
 *
* Ground truth for: ggml_tanh     (ggml.c:2709 → ops.cpp:10162 →
 *                                  unary-ops.cpp:253 → op_tanh == tanhf)
 *                   ggml_gelu     (ggml.c:2782 → ops.cpp:10178 →
 *                                  ops.cpp:2206 ggml_compute_forward_gelu_f32 →
 *                                  vec.h:987: the f16 LOOKUP TABLE, not the poly)
 *
 * Format (per section):
 *   u32 magic 'TNH1' | u32 kind | u32 n | payload
 *   kind 0 = ggml_tanh(f32)   : n f32 in | n f32 out
 *   kind 1 = ggml_gelu(f32)   : n f32 in | n f32 out
 *   kind 3 = raw tanhf(f32)   : n f32 in | n f32 out
 *   kind 5 = ggml_table_gelu_f16                                    : 65536 u16
 *   kind 7 = ggml_tanh(f16)   : n u16 in | n u16 out
 * EOF: magic 0xFFFFFFFF
 *
 * Build (against the reference build's shared libs; note the libs live in
 * $REF/bin, not $REF/ggml/src, and the dumper itself is a plain -O2 build —
 * the *library* was built -O3 -march=native, which is what matters):
 *   gcc -O2 -o parity/ref_tanh_dump parity/ref_tanh_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L$REF/bin -Wl,-rpath,$REF/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ($REF = /home/jeffrey/llm/llama.cpp/build-rust-ref)
 *   ./parity/ref_tanh_dump parity/tanh_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x9e3779b9u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 2.0f - 1.0f;
}

/* --- edge cases: exercise every branch of glibc tanhf (poly, 2^x path,
 *     |x|>~9 saturation, inf/nan, denormals, signed zero) --- */
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
    0x41200000u, /* 10.0 */
    0xc1200000u, /* -10.0 */
    0x41a00000u, /* 20.0 */
    0xc1a00000u, /* -20.0 */
    0x42b00000u, /* 88.0 */
    0xc2b00000u, /* -88.0 */
    0x42c80000u, /* 100.0 */
    0xc2c80000u, /* -100.0 */
    0x7f7fffffu, /* FLT_MAX */
    0xff7fffffu, /* -FLT_MAX */
    0x7f800000u, /* +inf */
    0xff800000u, /* -inf */
    0x7fc00000u, /* qnan */
    0xffc00000u, /* -qnan */
    0x7f800001u, /* snan */
    0x39000000u, /* 4.88e-4 */
    0x33000000u, /* 2.98e-8 (1 ulp-ish, tanh ~ x) */
    0x0c000000u, /* tiny */
    0x3f000001u, /* 0.50000006 */
    0x3effffffu, /* 0.49999997 */
    0x3dccccccu, /* 0.1 */
    0x40ffffffu, /* 7.999999 */
    0x41000000u, /* 8.0 */
    0x41100000u, /* 9.0 */
    0x40e00000u, /* 7.0 */
    0x40400000u, /* 3.0 */
    0x39d6c0c1u, /* ~4.1e-4 */
};

/* Run nn elements of in[] through the reference ggml op `op` and write out[]. */
static void run_op(FILE *f, uint32_t kind, int op_tanh, const float *in, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * sizeof(float) * 2 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(a->data, in, (size_t) n * sizeof(float));

    struct ggml_tensor * r = op_tanh ? ggml_tanh(ctx, a) : ggml_gelu(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    uint32_t magic = 0x31484e54u; /* "TNH1" little-endian */
    uint32_t nn = (uint32_t) n;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 4, n, f);
    fwrite(r->data, 4, n, f);

    ggml_free(ctx);
}

/* Scalar reference computed by the same libm the reference .so links:
 *   kind 3 = raw tanhf(x) (op_tanh body, unary-ops.cpp:19). The op output
 *   (kind 0) must equal it bit-for-bit — that is the proof that this build has
 *   no SIMD/libmvec tanh. */
static void dump_scalar(FILE *f, uint32_t kind, const float *in, int n) {
    uint32_t magic = 0x31484e54u;
    uint32_t nn = (uint32_t) n;
    float * out = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) {
        out[i] = tanhf(in[i]);
    }
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 4, n, f);
    fwrite(out, 4, n, f);
    free(out);
}

/* kind 7 = ggml_tanh on an F16 tensor (ops.cpp:10162 → unary_op<op_tanh> →
 * apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>, i.e. fp16(tanhf(fp16_to_f32))).
 * Payload: n u16 input bits then n u16 output bits. */
static void run_op_f16(FILE *f, const uint16_t *in, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * 2 * 2 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    memcpy(a->data, in, (size_t) n * 2);
    struct ggml_tensor * r = ggml_tanh(ctx, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    uint32_t magic = 0x31484e54u, kind = 7, nn = (uint32_t) n;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(in, 2, n, f);
    fwrite(r->data, 2, n, f);
    ggml_free(ctx);
}

/* kind 5 = the reference's GELU f16 lookup table (vec.cpp:6, filled at
 * ggml_cpu_init from ggml_gelu_f32; ggml-cpu.c:3891). 128 KiB payload. */
static void dump_gelu_table(FILE *f) {
    extern uint16_t ggml_table_gelu_f16[1 << 16];
    /* no fused guard op is needed: ggml_cpu_init() already ran inside the
     * graph computes above (ggml_graph_compute_with_ctx → ggml_cpu_init). */
    uint32_t magic = 0x31484e54u, kind = 5, n = 1 << 16;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&n, 4, 1, f);
    fwrite(ggml_table_gelu_f16, 2, n, f);
}

int main(int argc, char **argv) {
    const char * out = argc > 1 ? argv[1] : "tanh_ref.bin";
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* LCG lengths: 16-multiples, non-multiples, tiny, large */
    int lens[] = {1024, 1000, 63, 16, 17, 1, 33, 255, 1023, 4096, 47, 15};
    for (int li = 0; li < (int)(sizeof(lens)/sizeof(lens[0])); li++) {
        int n = lens[li];
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = next_val();
        run_op(f, 0, 1, x, n);   /* ggml_tanh */
        run_op(f, 1, 0, x, n);   /* ggml_gelu */
        dump_scalar(f, 3, x, n); /* raw tanhf (same libm as the .so) */
        free(x);
    }

    /* tanh over the LCG scaled into the wide range seen by softcap inputs
     * (softcap divides logits by 30/50 → mostly |x| < 1, but a few large) */
    {
        int n = 512;
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = next_val() * 64.0f;
        run_op(f, 0, 1, x, n);
        run_op(f, 1, 0, x, n);
        dump_scalar(f, 3, x, n);
        free(x);
    }

    /* edge-case sweep */
    {
        int n = (int)(sizeof(specials_bits)/sizeof(specials_bits[0]));
        float * x = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) memcpy(&x[i], &specials_bits[i], 4);
        run_op(f, 0, 1, x, n);
        run_op(f, 1, 0, x, n);
        dump_scalar(f, 3, x, n);
        free(x);
    }

    /* the GELU f16 table (needs ggml_cpu_init, run by the computes above) */
    dump_gelu_table(f);

    /* F16 tanh: LCG f16 values + every interesting f16 pattern class */
    {
        int n = 600;
        uint16_t * h = malloc(n * 2);
        for (int i = 0; i < n; i++) {
            /* f16 = round-to-nearest of an LCG value in [-8, 8) */
            float v = next_val() * 8.0f;
            h[i] = ggml_fp32_to_fp16(v); /* same scalar conversion this build uses */
        }
        /* pattern sweep: denormals, smallest normals, ±0, ±inf, NaNs, saturation */
        static const uint16_t pats[] = {
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x8400, 0x7bff, 0xfbff,
            0x7c00, 0xfc00, 0x7c01, 0x7e00, 0xfc01, 0xfe00, 0x7fff, 0xffff,
            0x3555, 0xb555, 0x2e66, 0x1c00, 0x9c00, 0x3800, 0xb800, 0x3c00, 0xbc00,
        };
        for (int i = 0; i < (int)(sizeof(pats)/sizeof(pats[0])); i++) h[n - 1 - i] = pats[i];
        run_op_f16(f, h, n);
        free(h);
    }

    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}