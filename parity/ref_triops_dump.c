// ref_triops_dump.c — the reference-side op-level bit-compare for arch
// batch 18's GGML_OP_DIAG + GGML_OP_SOLVE_TRI (and the GGML_OP_SET inplace
// scatter the chunked delta-net's chunk loop leans on). The reference CPU
// backend implements all three (ops.cpp:5434 / :10824 / :4769) even though
// no llama.cpp model graph emits DIAG/SOLVE_TRI — the chunked delta-net
// builder is unreachable (fused_gdn_ch hardwired true, llama-context.cpp:234)
// — so this probe IS the parity oracle.
//
// Both ops are F32-only in the reference (ggml_compute_forward_diag /
// _solve_tri GGML_ABORT on anything else), so F32 is the whole type matrix.
//
// NOTE on solve_tri arithmetic: the reference .so is built with GCC -O3 and
// the default -ffp-contract=fast, and GCC's vectorizer shapes the inner
// substitution loop per trip count (zmm 16-wide / ymm 8-wide vector bodies
// with in-order lane adds = the two-rounding form, scalar FMA remainder
// tails). The port's forward_solve_tri reproduces the observable rounding
// of that binary; tests/mtp2_e2e.rs::mtp2_tri_ops_bitcompare is the check.
//
// build:
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       parity/ref_triops_dump.c -o parity/ref_triops_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run: ./parity/ref_triops_dump <out.bin>
//
// File format: char magic[8] = "TRIOPS\0\0", then a sequence of records
//   [i32 n_elements][n_elements * f32] for, in order:
//   solve_tri #1 (n=8,  k=3, B1=2, B2=1, threads=1)
//   solve_tri #2 (n=64, k=64, B1=1, B2=3, threads=4)
//   solve_tri #3 (n=1,  k=1, B1=1, B2=1, threads=1)  degenerate
//   diag      #1 (ne0=5,  ne2=2, ne3=1)
//   diag      #2 (ne0=64, ne2=3, ne3=2)
//   set_inplace #1 (3x4 region into a 5x7 dst)

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// the port's RNG (tests/mtp2_e2e.rs's Rng) reimplemented 1:1
static uint64_t rs = 0;
static float rng_next(void) {
    rs += 0x9e3779b97f4a7c15ull;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return ((z >> 40) * 1.0f / 8388608.0f) - 1.0f;
}

static FILE * out;

static void dump_tensor(const struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    int32_t n32 = (int32_t) n;
    fwrite(&n32, 4, 1, out);
    fwrite(t->data, 4, n, out);
}

static void fill_rand(struct ggml_tensor * t, float scale) {
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; i++) ((float *) t->data)[i] = scale * rng_next();
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "/tmp/gops-triops-ref.bin";

    struct ggml_init_params ip = { .mem_size = 64*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    out = fopen(out_path, "wb");
    fwrite("TRIOPS\0\0", 1, 8, out);

    // ---- solve_tri cases ------------------------------------------------
    {
        // #1: n=8, k=3, B1=2, B2=1 — A lower-triangular with a real diagonal
        const int n = 8, k = 3, b1 = 2;
        struct ggml_tensor * A = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n, n, b1, 1);
        struct ggml_tensor * B = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, n, b1, 1);
        rs = 0x1234abc;
        fill_rand(A, 0.3f);
        // keep only the lower triangle (incl. diagonal), diagonal in [0.5, 1.5]
        for (int bi = 0; bi < b1; bi++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    float * p = (float *) ((char *) A->data + bi*A->nb[2]) + i*n + j;
                    if (j > i) *p = 0.0f;
                    if (j == i) *p = 0.5f + 0.5f*(0.5f*rng_next() + 0.5f);
                }
        fill_rand(B, 0.7f);
        struct ggml_tensor * X = ggml_solve_tri(ctx, A, B, true, true, false);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, X);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        dump_tensor(X);

        // #2: n=64, k=64, B2=3 — chunk-size shapes, 4 threads (the split is
        // over ne02*ne03*k solve columns; a 4-thread run must stay bit-exact
        // vs the port's 4-thread run — and vs 1-thread, per the scalar sum)
        const int n2 = 64, k2 = 64, b2 = 3;
        struct ggml_tensor * A2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n2, n2, 1, b2);
        struct ggml_tensor * B2t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k2, n2, 1, b2);
        rs = 0xfeed987;
        fill_rand(A2, 0.2f);
        for (int bi = 0; bi < b2; bi++)
            for (int i = 0; i < n2; i++)
                for (int j = 0; j < n2; j++) {
                    float * p = (float *) ((char *) A2->data + bi*A2->nb[3]) + i*n2 + j;
                    if (j > i) *p = 0.0f;
                    if (j == i) *p = 0.75f + 0.5f*(0.5f*rng_next() + 0.5f);
                }
        fill_rand(B2t, 0.5f);
        struct ggml_tensor * X2 = ggml_solve_tri(ctx, A2, B2t, true, true, false);
        struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, X2);
        ggml_graph_compute_with_ctx(ctx, gf2, 4);
        dump_tensor(X2);

        // #3: degenerate 1x1
        struct ggml_tensor * A3 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
        struct ggml_tensor * B3 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
        ((float *) A3->data)[0] = 2.0f;
        ((float *) B3->data)[0] = -7.0f;
        struct ggml_tensor * X3 = ggml_solve_tri(ctx, A3, B3, true, true, false);
        struct ggml_cgraph * gf3 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf3, X3);
        ggml_graph_compute_with_ctx(ctx, gf3, 1);
        dump_tensor(X3);
    }

    // ---- diag cases ------------------------------------------------------
    {
        // #1: ne0=5, ne2=2, ne3=1
        struct ggml_tensor * d1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 1, 2, 1);
        rs = 0xd1a9;
        fill_rand(d1, 1.5f);
        struct ggml_tensor * D1 = ggml_diag(ctx, d1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, D1);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        dump_tensor(D1);

        // #2: ne0=64, ne2=3, ne3=2 (the chunked dnet's CS-scale identity)
        struct ggml_tensor * d2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 3, 2);
        fill_rand(d2, 1.0f);
        struct ggml_tensor * D2 = ggml_diag(ctx, d2);
        struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf2, D2);
        ggml_graph_compute_with_ctx(ctx, gf2, 1);
        dump_tensor(D2);
    }

    // ---- set_inplace case ------------------------------------------------
    {
        // dst [5, 7] contiguous; write b [3, 4] at element offset 2 — the
        // region's row stride is dst's own nb[1] (5*4), the higher strides
        // are dst-sized (b's ne2/ne3 are 1, so they never advance), exactly
        // how the chunk loop passes v's own nb[1..3]
        struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 7);
        struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
        rs = 0x5e7;
        fill_rand(dst, 0.4f);
        fill_rand(b, 1.2f);
        struct ggml_tensor * r = ggml_set_inplace(ctx, dst, b,
                ggml_row_size(GGML_TYPE_F32, 5), ggml_row_size(GGML_TYPE_F32, 5*7), ggml_row_size(GGML_TYPE_F32, 5*7), 2*4);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        dump_tensor(r);
    }

    fclose(out);
    printf("ok %s\n", out_path);
    return 0;
}
