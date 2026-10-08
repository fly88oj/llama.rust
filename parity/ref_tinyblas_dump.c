/* ref_tinyblas_dump.c — direct ground truth for the llamafile tinyBLAS GEMM
 * path (ggml/src/ggml-cpu/llamafile/sgemm.cpp), the reference's production
 * route for multi-column mul_mat.
 *
 * Unlike parity/ref_mulmat_dump.c (a full graph run) this calls the *exported*
 * `llamafile_sgemm` symbol directly, which answers both questions the Rust port
 * needs:
 *
 *   1. routing: the returned `bool` is exactly the reference's decision for a
 *      given (Atype, Btype, m, n, k) — including every bail condition (`n < 2`,
 *      `Ctype != F32`, the `Btype`-pairing guard of each `case`, and
 *      `tinyBLAS::matmul`'s `k % KN` / `m % 4` gate). The Rust side's
 *      `ggml::tinyblas::accepts` must return the same value for every case.
 *   2. values: when it returns true, `C` is the reference tinyBLAS output, byte
 *      for byte the target the Rust kernel has to reproduce (per output
 *      element; the tile/thread split cannot change a value).
 *
 * Artifact layout (little endian, sections in call order):
 *   u32 'VTB1' | u32 Atype | u32 Btype | u32 m | u32 n | u32 k | u32 ret |
 *   A bytes (m * lda * ggml_type_size(Atype)) |
 *   B bytes (n * ldb * ggml_type_size(Btype)) |
 *   C f32 (m * n)
 * terminated by u32 0xFFFFFFFF.
 *
 * `A`/`B` are generated with the same LCG as the other dumps and converted with
 * the reference's own quantizers (`ggml_quantize_chunk`), i.e. `B` is the
 * `vec_dot_type` form ggml-cpu.c:1389 hands to the second attempt. `lda`/`ldb`
 * are `k` (`k` = elements for the float classes, = 32-value blocks for the
 * quantized ones, exactly what the C call sites pass).
 *
 * Build (needs the internal cpu header for `struct ggml_compute_params` and a
 * real single-thread threadpool for `ggml_barrier`):
 *   REF=/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin   (a7b94df2c)
 *   NEXT=/home/jeffrey/llm/llama.cpp-next
 *   cc -O2 -std=gnu11 -march=native parity/ref_tinyblas_dump.c \
 *      -I$NEXT/ggml/include -I$NEXT/ggml/src -I$NEXT/ggml/src/ggml-cpu \
 *      -o /tmp/s2g-tinyblas/ref_tinyblas_dump -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *      -Wl,-rpath,$REF
 * Run: ./ref_tinyblas_dump parity/tinyblas_ref.bin
 * (regenerated for sync batch D2: a7b94df2c's K tails flip every k%16 / k%32
 * float case from rejected to accepted-with-values — 950 tail cases.)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h" /* struct ggml_compute_params, ggml_threadpool */
#include "ggml-cpu/llamafile/sgemm.h"

extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0x5eed1234u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float) (int32_t) lcg / (float) (1 << 28)) * 1.8f - 0.9f;
}

/* The reference build is AVX2+FMA+F16C+AVX512F+AVX512BF16 (GGML_NATIVE=ON,
 * -march=native): print the macros this translation unit sees with the same
 * -march=native, i.e. the branches sgemm.cpp instantiates. */
static uint32_t build_flags(void) {
    uint32_t v = 0;
#ifdef __AVX2__
    v |= 1u << 0;
#endif
#ifdef __AVX512F__
    v |= 1u << 1;
#endif
#ifdef __AVX512DQ__
    v |= 1u << 2;
#endif
#ifdef __AVX512BW__
    v |= 1u << 3;
#endif
#ifdef __AVX512VNNI__
    v |= 1u << 4;
#endif
#ifdef __AVX512VL__
    v |= 1u << 5;
#endif
#ifdef __AVX512BF16__
    v |= 1u << 6;
#endif
#ifdef __F16C__
    v |= 1u << 7;
#endif
#ifdef __FMA__
    v |= 1u << 8;
#endif
    return v;
}

static void print_build_flags(FILE * f) {
    static const char * const names[] = { "AVX2",  "AVX512F",  "AVX512DQ",   "AVX512BW",
                                          "AVX512VNNI", "AVX512VL", "AVX512BF16", "F16C",
                                          "FMA" };
    uint32_t v = build_flags();
    fprintf(f, "build:");
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        fprintf(f, " %s=%d", names[i], (int) ((v >> i) & 1u));
    }
    fprintf(f, "\n");
}

/* One direct `llamafile_sgemm` call, exactly as ggml-cpu.c:1306/:1389 invoke it
 * (per-slice m/n/k, lda/ldb in elements/blocks). */
static int sgemm_call(int m, int n, int k, const void * a, int lda, const void * b, int ldb,
                      float * c, enum ggml_type aty, enum ggml_type bty) {
    struct ggml_compute_params params;
    memset(&params, 0, sizeof(params));
    params.ith = 0;
    params.nth = 1;
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
    params.threadpool = ggml_threadpool_new(&tpp); /* real pool: ggml_barrier needs the struct */
    if (!params.threadpool) {
        fprintf(stderr, "threadpool\n");
        exit(1);
    }
    int ret = llamafile_sgemm(&params, m, n, k, a, lda, b, ldb, c, m, aty, bty, GGML_TYPE_F32) ? 1 : 0;
    ggml_threadpool_free(params.threadpool);
    return ret;
}

static void write_case(FILE * f, enum ggml_type aty, enum ggml_type bty, int m, int n, int k) {
    const int a_bs = ggml_blck_size(aty), b_bs = ggml_blck_size(bty);
    const int a_ts = ggml_type_size(aty);
    /* `k` is `ne00/blck_size(Atype)` (what the C call sites pass); both rows
     * carry the same number of *values*, so the B row is `row_size(bty, k_elems)`. */
    const int k_elems = k * a_bs;
    const size_t a_bytes = (size_t) m * (size_t) ggml_row_size(aty, k_elems);
    const size_t b_bytes = (size_t) n * (size_t) ggml_row_size(bty, k_elems);

    float * wa = malloc((size_t) k_elems * m * sizeof(float));
    float * wb = malloc((size_t) k_elems * n * sizeof(float));
    for (size_t i = 0; i < (size_t) k_elems * m; i++) wa[i] = next_val();
    for (size_t i = 0; i < (size_t) k_elems * n; i++) wb[i] = next_val();

    void * abuf = calloc(1, a_bytes ? a_bytes : 1);
    void * bbuf = calloc(1, b_bytes ? b_bytes : 1);
    float * cbuf = malloc((size_t) m * n * sizeof(float));

    /* A/B in `Atype`/`Btype` form: the reference quantizers for the block types
     * (`from_float` of the type traits table), plain f32/f16/bf16 otherwise. */
    if (a_bs > 1) {
        ggml_quantize_chunk(aty, wa, abuf, 0, m, k_elems, NULL);
    } else if (aty == GGML_TYPE_F16) {
        for (int i = 0; i < k_elems * m; i++) ((ggml_fp16_t *) abuf)[i] = ggml_fp32_to_fp16(wa[i]);
    } else if (aty == GGML_TYPE_BF16) {
        for (int i = 0; i < k_elems * m; i++) ((ggml_bf16_t *) abuf)[i] = ggml_fp32_to_bf16(wa[i]);
    } else {
        memcpy(abuf, wa, (size_t) k_elems * m * sizeof(float));
    }
    if (b_bs > 1) {
        ggml_quantize_chunk(bty, wb, bbuf, 0, n, k_elems, NULL);
    } else if (bty == GGML_TYPE_F16) {
        for (int i = 0; i < k_elems * n; i++) ((ggml_fp16_t *) bbuf)[i] = ggml_fp32_to_fp16(wb[i]);
    } else if (bty == GGML_TYPE_BF16) {
        for (int i = 0; i < k_elems * n; i++) ((ggml_bf16_t *) bbuf)[i] = ggml_fp32_to_bf16(wb[i]);
    } else {
        memcpy(bbuf, wb, (size_t) k_elems * n * sizeof(float));
    }

    for (size_t i = 0; i < (size_t) m * n; i++) cbuf[i] = -12345.0f; /* sentinel: unwritten on bail */
    int ret = sgemm_call(m, n, k, abuf, k, bbuf, k, cbuf, aty, bty);

    uint32_t hdr[7] = { 0x31425456u, (uint32_t) aty, (uint32_t) bty, (uint32_t) m,
                        (uint32_t) n,   (uint32_t) k,   (uint32_t) ret };
    fwrite(hdr, sizeof(hdr), 1, f);
    if (ret) {
        /* Only accepted cases carry a payload: a bailed case is a routing fact
         * (the kernel is never entered, so nothing reads A/B), and this keeps
         * the artifact small. */
        fwrite(abuf, 1, a_bytes, f);
        fwrite(bbuf, 1, b_bytes, f);
        fwrite(cbuf, sizeof(float), (size_t) m * n, f);
    }

    free(wa);
    free(wb);
    free(abuf);
    free(bbuf);
    free(cbuf);
}

int main(int argc, char ** argv) {
    const char * out = argc > 1 ? argv[1] : "tinyblas_ref.bin";
    FILE * f = fopen(out, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out);
        return 1;
    }
    /* `ggml_cpu_init()` (ggml-cpu.c:3877) fills `ggml_table_f32_f16`, which
     * `GGML_CPU_FP16_TO_FP32` (simd-mappings.h:153) looks up — without it every
     * block delta converts to 0.0 and every Q0 case would "verify" as all-zero. */
    ggml_backend_cpu_init();
    print_build_flags(stderr);
    {
        uint32_t hdr[2] = { 0x30525456u /* 'VTR0' */, build_flags() };
        fwrite(hdr, sizeof(hdr), 1, f);
    }

    /* ---- the type switch itself (sgemm.cpp:3827-4149) ---- */
    /* A types with a `ggml_quantize_chunk` branch (Q8_1/Q8_K are missing from
     * it in this revision and Q4_1/Q5_1 have no sgemm case at all — the Rust
     * side asserts those rejections by Atype). */
    const enum ggml_type wt[] = { GGML_TYPE_F32,  GGML_TYPE_F16, GGML_TYPE_BF16,
                                  GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0,
                                  GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    /* Q8_1/Q8_K are not Btypes of any accepted case and have no
     * `ggml_quantize_chunk` branch in this revision, so the artifact covers the
     * B types the switch can actually accept (F32/F16/BF16/Q8_0); the Rust
     * side asserts the remaining rejections directly (`accepts` unit tests). */
    const enum ggml_type bt[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
                                  GGML_TYPE_Q8_0 };
    for (unsigned i = 0; i < sizeof(wt) / sizeof(wt[0]); i++) {
        for (unsigned j = 0; j < sizeof(bt) / sizeof(bt[0]); j++) {
            (void) 0;
            /* 8 blocks (or 256 elements for the float types) so that every
             * quantizer sees whole rows */
            write_case(f, wt[i], bt[j], 4, 3, ggml_blck_size(wt[i]) > 1 ? 8 : 256);
        }
    }

    /* ---- the float classes over every gate combination ----
     * k = 16/32/48/64/80/112/1024 covers k % 16, k % 32 and the encoder
     * attention lengths; m = 1..64 covers the m % 4 gate (and the m % 8 / % 16
     * BM selection); n = 1..17 the n >= 2 gate and the RN tails. */
    const int fm[] = { 1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 17, 64 };
    const int fn[] = { 1, 2, 3, 4, 5, 6, 7, 8, 14, 17 };
    const int fk[] = { 1, 2, 3, 4, 8, 14, 16, 32, 48, 64, 80, 112, 1024 };
    const int fbk[] = { 1, 2, 3, 28, 90 };
    for (unsigned t = 0; t < 3; t++) {
        const enum ggml_type ty = (enum ggml_type[]) { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }[t];
        for (unsigned a = 0; a < sizeof(fm) / sizeof(fm[0]); a++) {
            for (unsigned b = 0; b < sizeof(fn) / sizeof(fn[0]); b++) {
                for (unsigned c = 0; c < sizeof(fk) / sizeof(fk[0]); c++) {
                    if (fm[a] * fn[b] * fk[c] > 12000) continue; /* keep the artifact small */
                    write_case(f, ty, ty, fm[a], fn[b], fk[c]);
                }
            }
        }
    }

    /* ---- Q0 classes: k in 32-value blocks, B = q8_0 (the vec_dot_type) ---- */
    for (unsigned t = 0; t < 3; t++) {
        const enum ggml_type ty =
            (enum ggml_type[]) { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0 }[t];
        for (unsigned a = 0; a < sizeof(fm) / sizeof(fm[0]); a++) {
            for (unsigned b = 0; b < sizeof(fn) / sizeof(fn[0]); b++) {
                for (unsigned c = 0; c < sizeof(fbk) / sizeof(fbk[0]); c++) {
                    const int k = fbk[c]; /* 32-value blocks: 1,2,3,28 (896),90 (2880) */
                    if (fm[a] * k > 400 || fn[b] * k > 200) continue;
                    write_case(f, ty, GGML_TYPE_Q8_0, fm[a], fn[b], k);
                }
            }
        }
    }

    /* ---- the shapes the BERT/T5 encoders actually run (see PARITY.md) ----
     * kq  = mul_mat(k_perm [64, T, H], q_perm [64, T, H])  -> m = T, n = T, k = 64
     * kqv = mul_mat(v_cont [T, 64, H], kq [T, T, H])       -> m = 64, n = T, k = T
     * with T = 1..64. (The kq src1 is a permuted view, i.e. the reference skips
     * the contiguous-src1 attempt; the *values* are what the graph dumps pin.) */
    const int toks[] = { 1, 2, 3, 4, 8, 14, 16, 32, 64 };
    for (unsigned i = 0; i < sizeof(toks) / sizeof(toks[0]); i++) {
        const int T = toks[i];
        write_case(f, GGML_TYPE_F32, GGML_TYPE_F32, T, T, 64); /* kq  */
        write_case(f, GGML_TYPE_F32, GGML_TYPE_F32, 64, T, T); /* kqv */
    }
    /* gpt-oss / bge-m3 projection shapes: k = 2880/4096, m = 2880/4096, n = 8 */
    write_case(f, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, 896, 64, 28);
    write_case(f, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1024, 14, 32);

    /* ---- weight types without a quantizer in the port (MXFP4/IQ4_NL, i.e.
     * what should never reach tinyBLAS on the Rust side) or without a
     * `ggml_quantize_chunk` branch (Q8_1), plus pairs whose switch case does
     * not exist (Q4_1/Q5_1/K-quants/TQ1_0). The payload is zero-filled (no
     * quantizer needed) — except that IQ4_NL *is* an accepted smallB when B is
     * Q8_0 (sgemm.cpp:4131), so such a case carries a real C section computed
     * from the zero weights, which is what the Rust side reports as the one
     * unimplemented case. */
    {
        const enum ggml_type rt[] = { GGML_TYPE_MXFP4, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1,
                                      GGML_TYPE_Q4_1,  GGML_TYPE_Q5_1,  GGML_TYPE_Q2_K,
                                      GGML_TYPE_Q3_K,  GGML_TYPE_Q5_K,  GGML_TYPE_TQ1_0 };
        const enum ggml_type rb[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
                                      GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K };
        const int m = 8, n = 4, k = 2, k_elems = k * 32;
        for (unsigned i = 0; i < sizeof(rt) / sizeof(rt[0]); i++) {
            for (unsigned j = 0; j < sizeof(rb) / sizeof(rb[0]); j++) {
                const size_t ab = (size_t) m * ggml_row_size(rt[i], k_elems);
                const size_t bb = (size_t) n * ggml_row_size(rb[j], k_elems);
                void * abuf = calloc(1, ab ? ab : 1);
                void * bbuf = calloc(1, bb ? bb : 1);
                float * cbuf = calloc(1, sizeof(float) * (size_t) m * n);
                struct ggml_compute_params params;
                memset(&params, 0, sizeof(params));
                params.nth = 1;
                struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
                params.threadpool = ggml_threadpool_new(&tpp);
                int ret = llamafile_sgemm(&params, m, n, k, abuf, k, bbuf, k, cbuf, m, rt[i],
                                          rb[j], GGML_TYPE_F32)
                              ? 1
                              : 0;
                ggml_threadpool_free(params.threadpool);
                uint32_t hdr[7] = { 0x31425456u, (uint32_t) rt[i], (uint32_t) rb[j],
                                    (uint32_t) m,   (uint32_t) n,     (uint32_t) k,
                                    (uint32_t) ret };
                fwrite(hdr, sizeof(hdr), 1, f);
                if (ret) {
                    fwrite(abuf, 1, ab, f);
                    fwrite(bbuf, 1, bb, f);
                    fwrite(cbuf, sizeof(float), (size_t) m * n, f);
                }
                free(abuf);
                free(bbuf);
                free(cbuf);
            }
        }
    }

    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}