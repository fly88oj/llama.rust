/* fp32→fp16 conversion probe — bit-exact comparison of the OLD reference
 * .so (libggml-cpu.so.0.25.3 = c35b66744) against the NEW reference .so
 * (libggml-cpu.so.0.26.0 = aa94f2086), over a battery built for the
 * round-to-nearest-even boundaries.
 *
 * Purpose: 2bbca8f20 ("ggml-cpu: vectorize fp32 to fp16 conversion") adds a
 * __VXE__/__VXE2__ (IBM Z s390x) vector path to ggml_cpu_fp32_to_fp16 and
 * vectorizes __lzs_f16cx4_store in simd-mappings.h — all inside the s390x
 * guards. The x86-64 build keeps the __F16C__ path untouched, so the two
 * .so's must produce identical bits on x86. This probe proves it; the
 * accompanying Rust test (fp32_to_fp16_bit_exact_vs_reference) proves the
 * port's conversion matches the same battery.
 *
 * Battery (deterministic, written to the output for the Rust side):
 *   - all 65536 f16 bit patterns → f32 (exact) — RNE must be the identity
 *   - the midpoints between adjacent f16 values (ties → even) and ±1 f32-ulp
 *     around each midpoint and around each f16 value (the RNE decision
 *     boundaries), in normal and denormal f16 ranges
 *   - specials: ±0, ±Inf, qNaN/sNaN payloads, FLT_MAX, values just above
 *     f16 max (→ ±Inf), the largest finite, denormal cutoffs
 *   - an LCG sweep
 *
 * Output format (LE):
 *   u32 magic 'F16C' | u32 n | n × f32 battery | n × u16 old_out |
 *   n × u16 new_out | n × u16 scalar_out (the ggml-impl.h bit-trick twin)
 * EOF: none (single section)
 *
 * Build: gcc -O2 -o /tmp/s4a/ref_fp16cvt_probe parity/ref_fp16cvt_probe.c -ldl -lm
 * Run:   /tmp/s4a/ref_fp16cvt_probe parity/fp16cvt_probe.bin
 * The probe dlopens the versioned .so files by full path, so both versions
 * coexist in build-rust-ref/bin.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>

typedef void (*bulk_fn)(const float *, uint16_t *, int64_t);

/* The scalar GGML_FP32_TO_FP16 bit-trick (ggml-impl.h:417-439) — the twin
 * the portable Rust conversion claims to match. NaN/overflow collapse to
 * 0x7E00/0xFE00 (unlike the F16C bulk path, which preserves NaN payloads —
 * a real C-side difference between the scalar macro and ggml_cpu_fp32_to_
 * fp16's SIMD chunks; the columns keep them apart). */

/* llama-rust-data-home: resolve corpus paths against $LLAMA_RUST_DATA_HOME (fallback: $HOME). */
static const char *llama_rust_data_path(const char *rel) {
    static char path[4096];
    const char *root = getenv("LLAMA_RUST_DATA_HOME");
    if (!root) root = getenv("HOME");
    if (!root) root = "/var/lib/llama-rust";
    snprintf(path, sizeof path, "%s%s", root, rel);
    return path;
}

static uint16_t scalar_fp32_to_fp16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    uint32_t w;
    memcpy(&w, &f, 4);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    {
        uint32_t bias_bits = (bias >> 1) + 0x07800000u;
        float bias_f;
        memcpy(&bias_f, &bias_bits, 4);
        base = bias_f + base; /* fp32_from_bits((bias>>1) + 0x07800000) + base */
    }
    uint32_t bits;
    memcpy(&bits, &base, 4);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

/* f16 → f32 (exact) */
static float h2f(uint16_t h) {
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000u;
    uint32_t two_w = w + w;
    uint32_t exp_offset = 0xE0u << 23;
    float exp_scale;
    uint32_t exp_scale_bits = 0x07800000u;
    memcpy(&exp_scale, &exp_scale_bits, 4);
    float normalized = (float)(((double)(two_w >> 4) + (double)exp_offset) * 1.0) * exp_scale;
    /* bit-trick (ggml_compute_fp16_to_fp32) */
    {
        float norm;
        uint32_t nb = ((two_w >> 4) + exp_offset);
        memcpy(&norm, &nb, 4);
        norm = norm * exp_scale;
        float den;
        uint32_t db = (two_w >> 17) | (126u << 23);
        memcpy(&den, &db, 4);
        den = den - 0.5f;
        float r;
        uint32_t rb;
        if (two_w < (1u << 27)) {
            r = den;
        } else {
            r = norm;
        }
        uint32_t rb2 = sign | (*(uint32_t *)&r == 0 ? 0 : 0);
        /* reassemble properly */
        uint32_t bits;
        uint32_t rbits;
        memcpy(&rbits, &r, 4);
        bits = sign | rbits;
        float out;
        memcpy(&out, &bits, 4);
        (void)normalized;
        (void)rb;
        (void)rb2;
        return out;
    }
}

static uint32_t lcg = 0x1234567u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((int32_t)lcg) / (float)(1 << 30); /* [-2, 2) */
}

int main(int argc, char **argv) {
    const char *out_path = argc > 1 ? argv[1] : "fp16cvt_probe.bin";

    const char *old_so = llama_rust_data_path("/llm/llama.cpp-next/build-rust-ref/bin/libggml-cpu.so.0.25.3");
    const char *new_so = llama_rust_data_path("/llm/llama.cpp-next/build-rust-ref/bin/libggml-cpu.so.0.26.0");

    void *h_old = dlopen(old_so, RTLD_NOW | RTLD_LOCAL);
    void *h_new = dlopen(new_so, RTLD_NOW | RTLD_LOCAL);
    if (!h_old || !h_new) {
        fprintf(stderr, "dlopen failed: %s / %s\n", dlerror(), dlerror());
        return 1;
    }
    bulk_fn f_old = (bulk_fn)dlsym(h_old, "ggml_cpu_fp32_to_fp16");
    bulk_fn f_new = (bulk_fn)dlsym(h_new, "ggml_cpu_fp32_to_fp16");
    if (!f_old || !f_new) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }

    /* ---- build the battery ---- */
    size_t cap = 1 << 21, n = 0;
    float *bat = malloc(cap * sizeof(float));
    if (!bat) { return 1; }

    /* 1. all f16 patterns → f32 (exact): identity under RNE */
    for (uint32_t h = 0; h < 65536; h++) {
        float f = h2f((uint16_t)h);
        memcpy(&bat[n++], &f, 4);
    }

    /* 2. RNE decision boundaries: for adjacent f16 pairs (a<b, same sign),
     *    the midpoint m=(a+b)/2 ties → even; one f32 ulp on either side of
     *    m decides. Walk the positive normals and denormals. */
    for (int den = 0; den < 2; den++) {
        uint32_t h_start = den ? 1 : 0x0400; /* denormals : smallest normal */
        uint32_t h_end = den ? 0x0400 : 0x7C00;
        float prev = h2f((uint16_t)h_start);
        for (uint32_t h = h_start + 1; h < h_end; h++) {
            float cur = h2f((uint16_t)h);
            float mid = (prev + cur) * 0.5f;
            /* mid, ±1 f32 ulp, and the values themselves */
            bat[n++] = mid;
            uint32_t mb; memcpy(&mb, &mid, 4);
            uint32_t up = mb + 1, dn = mb - 1;
            float f; memcpy(&f, &up, 4); bat[n++] = f;
            memcpy(&f, &dn, 4); bat[n++] = f;
            bat[n++] = prev;
            bat[n++] = cur;
            prev = cur;
        }
    }
    /* mirrored negatives */
    {
        size_t m = n;
        for (size_t i = 0; i < m && n < cap; i++) {
            uint32_t b; memcpy(&b, &bat[i], 4);
            b ^= 0x80000000u;
            memcpy(&bat[n++], &b, 4);
        }
    }

    /* 3. specials: overflow → ±Inf, NaNs with payloads, FLT_MAX, ±0 */
    {
        const uint32_t sp[] = {
            0x00000000u, 0x80000000u,             /* ±0 */
            0x7F800000u, 0xFF800000u,             /* ±Inf */
            0x7FC00000u, 0xFFC00000u,             /* qNaN */
            0x7F800001u, 0xFF800001u,             /* sNaN min payload */
            0x7FFFFFFFu, 0xFFFFFFFFu,             /* NaN max payload */
            0x7F7FFFFFu, 0xFF7FFFFFu,             /* ±FLT_MAX */
            0x477FF000u, 0x477FFF00u,             /* just below f16 max */
            0x47800000u, 0x47800001u,             /* just above 65504 → Inf */
            0x477FFF01u, 0x477FF801u,
            0xC7800000u, 0xC7800001u,             /* -just above → -Inf */
            0x38800000u, 0x387FFFFFu,             /* f16 min normal ±1ulp */
            0x387FFFFEu, 0x38800001u,
            0x33000000u, 0x32000000u, 0x30000000u, /* denormal f32s */
            0x0C000000u, 0x04000000u, 0x00000001u, /* f32 denormals */
            0x007FFFFFu, 0x807FFFFFu,
            0x3F800000u, 0xBF800000u, 0x40490FDBu,
        };
        for (size_t i = 0; i < sizeof(sp)/sizeof(sp[0]); i++) {
            float f; memcpy(&f, &sp[i], 4); bat[n++] = f;
        }
    }

    /* 4. LCG sweep */
    for (int i = 0; i < 4096 && n < cap; i++) {
        bat[n++] = next_val() * 1024.0f; /* widen into overflow territory */
    }
    for (int i = 0; i < 4096 && n < cap; i++) {
        bat[n++] = next_val() * 1e-5f; /* denormal territory */
    }

    /* ---- run both .so's + the scalar twin ---- */
    uint16_t *o_old = malloc(n * 2);
    uint16_t *o_new = malloc(n * 2);
    uint16_t *o_scl = malloc(n * 2);
    /* chunked through the SIMD paths too (16/8/4-wide + tails) */
    size_t off = 0;
    const size_t chunks[] = {16, 7, 3, 1, 64, 15, 31, 256, 5, 11};
    for (size_t ci = 0; ci < sizeof(chunks)/sizeof(chunks[0]) && off < n; ci++) {
        size_t c = chunks[ci];
        while (off + c <= n) {
            f_old(bat + off, o_old + off, (int64_t)c);
            f_new(bat + off, o_new + off, (int64_t)c);
            off += c;
        }
    }
    if (off < n) { /* whole remainder */
        f_old(bat + off, o_old + off, (int64_t)(n - off));
        f_new(bat + off, o_new + off, (int64_t)(n - off));
    }
    for (size_t i = 0; i < n; i++) {
        o_scl[i] = scalar_fp32_to_fp16(bat[i]);
    }

    /* ---- compare + report ---- */
    size_t diffs = 0;
    size_t scl_vs_simd = 0; /* scalar macro vs bulk SIMD — expected on NaN payloads */
    for (size_t i = 0; i < n; i++) {
        if (o_old[i] != o_new[i]) {
            if (diffs < 10) {
                fprintf(stderr, "DIFF at %zu: f32 %08x -> old %04x new %04x\n",
                        i, *(uint32_t *)&bat[i], o_old[i], o_new[i]);
            }
            diffs++;
        }
        if (o_scl[i] != o_new[i]) {
            scl_vs_simd++;
        }
    }
    fprintf(stderr, "battery n=%zu  old(c35b66744)-vs-new(aa94f2086) diffs: %zu"
                    "  scalar-vs-bulk-SIMD diffs: %zu (NaN payload collapse)\n",
            n, diffs, scl_vs_simd);

    FILE *f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 1; }
    uint32_t magic = 0x43363146u; /* "F16C" */
    uint32_t nn = (uint32_t)n;
    fwrite(&magic, 4, 1, f);
    fwrite(&nn, 4, 1, f);
    fwrite(bat, 4, n, f);
    fwrite(o_old, 2, n, f);
    fwrite(o_new, 2, n, f);
    fwrite(o_scl, 2, n, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out_path);

    return diffs == 0 ? 0 : 2;
}
