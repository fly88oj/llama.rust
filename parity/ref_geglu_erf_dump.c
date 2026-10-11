/* ggml_geglu_erf_split ground-truth dumper — drives the reference build's
 * GLU op with GGML_GLU_OP_GEGLU_ERF through the real graph-compute path and
 * writes bit-exact outputs.
 *
 * Why the graph path: ggml_vec_geglu_erf_f32 is `inline static` in vec.h (no
 * exported symbol), and the GLU dispatch lives inside ggml-cpu.so. Building
 * the tiny graph and calling ggml_graph_compute_with_ctx reproduces exactly
 * what a model graph does.
 *
 * Ground truth for: ggml_geglu_erf_split (ggml.c:3092 → ggml_glu_impl with
 * b != NULL, GGML_GLU_OP_GEGLU_ERF → ops.cpp ggml_compute_forward_glu →
 * ggml_compute_forward_geglu_erf_f32 → vec.h:1501 ggml_vec_geglu_erf_f32 =
 * 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV)) * g — a plain scalar libm loop).
 * ModernBERT encoders reach it via LLM_FFN_GEGLU_ERF (6184e92c5,
 * modern-bert.cpp:148-154).
 *
 * Format (per section):
 *   u32 magic 'GGRF' | u32 kind | u32 n | payload
 *   kind 0 = ggml_geglu_erf_split(f32, f32): n f32 x | n f32 g | n f32 out
 *   kind 3 = raw formula (f32):            n f32 x | n f32 g | n f32 out
 * EOF: magic 0xFFFFFFFF
 *
 * Build (against the reference build's shared libs; the dumper itself is a
 * plain -O2 build — the *library* was built -O3, which is what matters):
 *   gcc -O2 -o parity/ref_geglu_erf_dump parity/ref_geglu_erf_dump.c \
 *     -I$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/ggml/include \
 *     -L$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
 *     -Wl,-rpath,$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_geglu_erf_dump parity/geglu_erf_ref.bin
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

/* --- edge cases (same battery as ref_gelu_erf_dump.c: the erff poly/double
 *     path switches, saturation plateaus, denormals, inf/nan through
 *     0.5*x*(1+erf)*g) --- */
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
    0x7f800000u, /* +inf */
    0xff800000u, /* -inf */
    0x7fc00000u, /* qnan */
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

/* Run n elements of x[]/g[] through the reference ggml_geglu_erf_split. */
static void run_geglu_erf_f32(FILE *f, uint32_t kind, const float *x, const float *g, int n) {
    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() +
                            (size_t) n * sizeof(float) * 3 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(a->data, x, (size_t) n * sizeof(float));
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    memcpy(b->data, g, (size_t) n * sizeof(float));

    struct ggml_tensor * r = ggml_geglu_erf_split(ctx, a, b);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    uint32_t magic = 0x46524747u; /* "GGRF" little-endian */
    uint32_t nn = (uint32_t) n;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(x, 4, n, f);
    fwrite(g, 4, n, f);
    fwrite(r->data, 4, n, f);

    ggml_free(ctx);
}

/* Scalar reference computed by the same libm the reference .so links:
 *   kind 3 = the raw ggml_vec_geglu_erf_f32 body (vec.h:1501-1507). The op
 *   output (kind 0) must equal it bit-for-bit. Left-associated multiplies
 *   over a call result — -ffp-contract cannot fuse anything. */
static void dump_scalar_formula(FILE *f, uint32_t kind, const float *x, const float *g, int n) {
    uint32_t magic = 0x46524747u;
    uint32_t nn = (uint32_t) n;
    float * out = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        out[i] = 0.5f * xi * (1.0f + erff(xi * SQRT_2_INV)) * g[i];
    }
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(x, 4, n, f);
    fwrite(g, 4, n, f);
    fwrite(out, 4, n, f);
    free(out);
}

int main(int argc, char **argv) {
    const char * out = "geglu_erf_ref.bin";
    if (argc > 1) {
        out = argv[1];
    }
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* LCG lengths: 16-multiples, non-multiples, tiny, large — inputs in
     * [-1, 1) land in erff's polynomial region. The first section gets a
     * raw-formula twin over the same arrays (kind 3) for the 1:1 check. */
    int lens[] = {1024, 1000, 63, 16, 17, 1, 33, 255, 1023, 4096, 47, 15};
    for (int li = 0; li < (int)(sizeof(lens)/sizeof(lens[0])); li++) {
        int n = lens[li];
        float * x = malloc(n * sizeof(float));
        float * g = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            x[i] = next_val();
            g[i] = next_val();
        }
        run_geglu_erf_f32(f, 0, x, g, n);
        if (li == 0) {
            dump_scalar_formula(f, 3, x, g, n);
        }
        free(x); free(g);
    }

    /* the specials battery, once as x (g LCG) and once as g (x LCG) */
    {
        int n = (int)(sizeof(specials_bits)/sizeof(specials_bits[0]));
        float * x = malloc(n * sizeof(float));
        float * g = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            memcpy(&x[i], &specials_bits[i], 4);
            g[i] = next_val();
        }
        run_geglu_erf_f32(f, 0, x, g, n);
        for (int i = 0; i < n; i++) {
            x[i] = next_val();
            memcpy(&g[i], &specials_bits[i], 4);
        }
        run_geglu_erf_f32(f, 0, x, g, n);
        /* specials on both sides (inf*0 / nan paths through the product) */
        for (int i = 0; i < n; i++) {
            memcpy(&x[i], &specials_bits[i], 4);
            memcpy(&g[i], &specials_bits[n - 1 - i], 4);
        }
        run_geglu_erf_f32(f, 0, x, g, n);
        free(x); free(g);
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}
