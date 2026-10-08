/* WKV ground-truth dumper — drives the reference build's three fused RWKV ops
 * through the real graph-compute path and writes bit-exact I/O.
 *
 * Ground truth for: ggml_rwkv_wkv6          (ggml.c:5873 → ops.cpp:10413-10603)
 *                   ggml_gated_linear_attn  (ggml.c:5916 → ops.cpp:10623-11418)
 *                   ggml_rwkv_wkv7          (ggml.c:5959 → ops.cpp:11422-11617)
 *
 * Format (per section):
 *   u32 magic 'WKV1' | u32 kind | u32 nd | nd u32 dims | payload
 *   kind 20 = rwkv_wkv6:
 *     dims = {S, H, T, n_seqs}, then
 *     k, v, r {S,H,T} f32 ; tf {S,H} f32 ; td {S,H,T} f32 ; state {S*S*H, n_seqs} f32
 *     out   {S*H, T + S*n_seqs} f32
 *   kind 21 = gated_linear_attn:
 *     dims = {S, H, T, n_seqs, scale_bits}, then
 *     k, v, q, g {S,H,T} f32 ; state {S*S*H, n_seqs} f32
 *     out   {S*H, T + S*n_seqs} f32
 *   kind 22 = rwkv_wkv7:
 *     dims = {S, H, T, n_seqs}, then
 *     r, w, k, v, a, b {S,H,T} f32 ; state {S*S*H, n_seqs} f32
 *     out   {S*H, T + S*n_seqs} f32
 * EOF: magic 0xFFFFFFFF
 *
 * Computed at n_threads == 1 (the deterministic reference semantics; the
 * wkv7 x86 SIMD path's 64-wide window has a cross-head write race at
 * nth > 1 for head_size < GGML_F32_STEP — see the port's wkv.rs header).
 *
 * Every src tensor is followed in the context pool by a zero-filled guard
 * (and the dst node is followed by one too, allocated after the graph
 * build). For the wkv6/gla kernels (whose 16-wide chunks never leave their
 * tensors) the guards are inert safety margin. For wkv7 they only matter
 * at head_size < 64 — and there the reference is nondeterministic anyway
 * (it reads/writes allocator metadata and self-poisons the guard with
 * NaNs mid-recurrence; parity/wkv7_oob_proof.c), which is why this dump's
 * wkv7 ladder is head_size ≥ 64 only: shapes where the window provably
 * never leaves the tensors and the reference is run-to-run stable.
 *
 * Build (against the reference build's shared libs):
 *   gcc -O2 -o parity/ref_wkv_dump parity/ref_wkv_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_wkv_dump parity/wkv_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

#define MAGIC 0x31564B57u /* "WKV1" */

static uint32_t lcg = 0x2545F491u;
/* [-1,1) like ref_ssm_dump.c — well-conditioned for the state recurrence */
static float rnd(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(int32_t)lcg / (float)(1 << 30) * 0.5f;
}

static void hdr(FILE *f, uint32_t kind, const uint32_t *dims, int nd) {
    fwrite(&(uint32_t){MAGIC}, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&(uint32_t){nd}, 4, 1, f);
    fwrite(dims, 4, nd, f);
}

static void wr_f32s(FILE *f, const float *p, int n) { fwrite(p, 4, n, f); }

/* a src tensor {S,H,T} filled with rnd(), followed by a zero guard so the
 * wkv7 64-wide window's reads past the end see zeros */
static struct ggml_tensor *mk_src(struct ggml_context *ctx, int S, int H, int T) {
    struct ggml_tensor *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
    float *p = (float *)t->data;
    for (int i = 0; i < S * H * T; i++) p[i] = rnd();
    struct ggml_tensor *g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
    memset(g->data, 0, 512 * sizeof(float));
    return t;
}

static struct ggml_tensor *mk_state(struct ggml_context *ctx, int S, int H, int n_seqs) {
    struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)S * S * H, n_seqs);
    float *p = (float *)t->data;
    for (int i = 0; i < S * S * H * n_seqs; i++) p[i] = rnd();
    struct ggml_tensor *g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    memset(g->data, 0, 1024 * sizeof(float));
    return t;
}

/* allocate the zero guard that follows the dst node in the pool (the wkv7
 * state writes run past dst's end for head_size < 64) */
static void guard_after_dst(struct ggml_context *ctx) {
    struct ggml_tensor *g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    memset(g->data, 0, 1024 * sizeof(float));
}

static void run_and_dump(FILE *f, struct ggml_context *ctx, struct ggml_tensor *r, uint32_t kind,
                         const uint32_t *dims, int nd, const struct ggml_tensor **ins, int n_ins) {
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    guard_after_dst(ctx);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(1);
    }
    hdr(f, kind, dims, nd);
    for (int i = 0; i < n_ins; i++) wr_f32s(f, (const float *)ins[i]->data, (int)ggml_nelements(ins[i]));
    wr_f32s(f, (const float *)r->data, (int)ggml_nelements(r));
}

/* kind 20 — ggml_rwkv_wkv6 */
static void dump_wkv6(FILE *f, int S, int H, int T, int n_seqs) {
    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() +
                            (size_t)(5 * (S * H * T + 512) + S * H + S * S * H * n_seqs + 1024 +
                                     (size_t)S * H * (T + S * n_seqs) + 1024) * sizeof(float) + (1 << 20);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor *k = mk_src(ctx, S, H, T);
    struct ggml_tensor *v = mk_src(ctx, S, H, T);
    struct ggml_tensor *r = mk_src(ctx, S, H, T);
    struct ggml_tensor *tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, H);
    { float *p = (float *)tf->data; for (int i = 0; i < S * H; i++) p[i] = rnd();
      struct ggml_tensor *g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512); memset(g->data, 0, 512 * 4); }
    struct ggml_tensor *td = mk_src(ctx, S, H, T);
    struct ggml_tensor *st = mk_state(ctx, S, H, n_seqs);

    struct ggml_tensor *out = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, st);
    uint32_t dims[4] = {(uint32_t)S, (uint32_t)H, (uint32_t)T, (uint32_t)n_seqs};
    const struct ggml_tensor *ins[6] = {k, v, r, tf, td, st};
    run_and_dump(f, ctx, out, 20, dims, 4, ins, 6);
    ggml_free(ctx);
}

/* kind 21 — ggml_gated_linear_attn */
static void dump_gla(FILE *f, int S, int H, int T, int n_seqs) {
    const float scale = powf((float)S, -0.5f);
    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() +
                            (size_t)(4 * (S * H * T + 512) + S * S * H * n_seqs + 1024 +
                                     (size_t)S * H * (T + S * n_seqs) + 1024) * sizeof(float) + (1 << 20);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor *k = mk_src(ctx, S, H, T);
    struct ggml_tensor *v = mk_src(ctx, S, H, T);
    struct ggml_tensor *q = mk_src(ctx, S, H, T);
    struct ggml_tensor *g = mk_src(ctx, S, H, T);
    struct ggml_tensor *st = mk_state(ctx, S, H, n_seqs);

    struct ggml_tensor *out = ggml_gated_linear_attn(ctx, k, v, q, g, st, scale);
    uint32_t sb; memcpy(&sb, &scale, 4);
    uint32_t dims[5] = {(uint32_t)S, (uint32_t)H, (uint32_t)T, (uint32_t)n_seqs, sb};
    const struct ggml_tensor *ins[5] = {k, v, q, g, st};
    run_and_dump(f, ctx, out, 21, dims, 5, ins, 5);
    ggml_free(ctx);
}

/* kind 22 — ggml_rwkv_wkv7 */
static void dump_wkv7(FILE *f, int S, int H, int T, int n_seqs) {
    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() +
                            (size_t)(6 * (S * H * T + 512) + S * S * H * n_seqs + 1024 +
                                     (size_t)S * H * (T + S * n_seqs) + 1024) * sizeof(float) + (1 << 20);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor *r = mk_src(ctx, S, H, T);
    struct ggml_tensor *w = mk_src(ctx, S, H, T);
    struct ggml_tensor *k = mk_src(ctx, S, H, T);
    struct ggml_tensor *v = mk_src(ctx, S, H, T);
    struct ggml_tensor *a = mk_src(ctx, S, H, T);
    struct ggml_tensor *b = mk_src(ctx, S, H, T);
    struct ggml_tensor *st = mk_state(ctx, S, H, n_seqs);

    struct ggml_tensor *out = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, st);
    uint32_t dims[4] = {(uint32_t)S, (uint32_t)H, (uint32_t)T, (uint32_t)n_seqs};
    const struct ggml_tensor *ins[7] = {r, w, k, v, a, b, st};
    run_and_dump(f, ctx, out, 22, dims, 4, ins, 7);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "parity/wkv_ref.bin";
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }

    /* wkv6 — canonical 64, the fixture's 16, a scalar-tail 20, n_seqs 2 */
    dump_wkv6(f, 64, 2, 5, 1);
    dump_wkv6(f, 16, 4, 3, 1);
    dump_wkv6(f, 20, 2, 3, 1);
    dump_wkv6(f, 32, 3, 4, 2);
    /* gla — same ladder */
    dump_gla(f, 64, 2, 5, 1);
    dump_gla(f, 16, 4, 3, 1);
    dump_gla(f, 20, 2, 3, 1);
    dump_gla(f, 32, 3, 4, 2);
    /* wkv7 — ONLY head_size ≥ GGML_F32_STEP (64): the x86 body's fixed
     * 64-float window stays inside the head row and the head's own state
     * block exactly when head_size ≥ 64 (i·S + 63 ≤ S²−1 ⟺ S ≥ 64), so
     * those shapes are fully deterministic. For head_size < 64 the window
     * reads AND writes past the tensors into ggml pool memory (object
     * headers with heap pointers; the region past dst is then overwritten
     * with NaNs mid-recurrence) — the reference's own output is
     * ASLR-dependent and differs from process run to process run, so no
     * bit-exact oracle can exist. Proof + per-run checksum table:
     * parity/wkv7_oob_proof.c (S=64 constant checksum across runs; S=16/48/
     * 32 all vary, mostly NaN). Ladder: 64 (every public RWKV7 GGUF), a
     * multi-seq 64, and 128 (two window steps per row). */
    dump_wkv7(f, 64, 2, 5, 1);
    dump_wkv7(f, 64, 2, 6, 2);
    dump_wkv7(f, 128, 2, 4, 1);

    fwrite(&(uint32_t){0xFFFFFFFFu}, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", path);
    return 0;
}
