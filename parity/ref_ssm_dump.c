/* SSM ground-truth dumper — drives the reference build's ggml_ssm_conv and
 * ggml_ssm_scan through the real graph-compute path and writes bit-exact I/O.
 *
 * Ground truth for: ggml_ssm_conv (ggml.c:5659 → ops.cpp:9701)
 *                   ggml_ssm_scan (ggml.c:5729 → ops.cpp:9771)
 *
 * Format (per section):
 *   u32 magic 'SSM1' | u32 kind | u32 n | payload
 *   kind 10 = ssm_conv:
 *     n = d_conv | d_inner | n_t | n_s  (4 u32), then
 *     sx  {d_conv-1+n_t, d_inner, n_s} f32
 *     c   {d_conv, d_inner} f32
 *     out {d_inner, n_t, n_s} f32
 *   kind 11 = ssm_scan:
 *     n = d_state | head_dim | n_head | n_group | n_seq_tokens | n_seqs | K |
 *         a_ne0  (8 u32), then
 *     s {d_state, head_dim, n_head, <n_seqs + 1 slots>} f32
 *     x {head_dim, n_head, n_seq_tokens, n_seqs} f32
 *     dt {n_head, n_seq_tokens, n_seqs} f32
 *     A {a_ne0, n_head} f32
 *     B {d_state, n_group, n_seq_tokens, n_seqs} f32
 *     C (same shape) f32
 *     ids {n_seqs} i32
 *     out {nelements(x) + K*d_state*head_dim*n_head*n_seqs} f32
 * EOF: magic 0xFFFFFFFF
 *
 * The extra state slot (n_seqs + 1) exists so the `ids` indirection is
 * exercised (seq 1 points at slot 1).
 *
 * Build (against the reference build's shared libs):
 *   gcc -O2 -o parity/ref_ssm_dump parity/ref_ssm_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   ./parity/ref_ssm_dump parity/ssm_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

#define MAGIC 0x314D5353u /* "SSM1" */

static uint32_t lcg = 0x2545F491u;
/* NOTE: (int32_t)lcg / 2^28 spans [-8,8]; ref_tanh_dump.c relies on that wide
 * range, but the state-space dump needs well-conditioned values (an exploding
 * mamba2 state would turn 1-ulp differences into large relative ones): rnd() is
 * a proper [-1,1) generator. */
static float rnd(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(int32_t)lcg / (float)(1 << 30) * 0.5f; /* [-1,1) */
}

static void hdr(FILE *f, uint32_t kind, const uint32_t *dims, int nd) {
    fwrite(&(uint32_t){MAGIC}, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&(uint32_t){nd}, 4, 1, f);
    fwrite(dims, 4, nd, f);
}

/* kind 10 — ggml_ssm_conv (causal 1D convolution, bias applied outside) */
static void dump_ssm_conv(FILE *f, int d_conv, int d_inner, int n_t, int n_s) {
    const int ncs = d_conv - 1 + n_t;
    const int nel_sx = ncs * d_inner * n_s;
    const int nel_c = d_conv * d_inner;
    const int nel_out = d_inner * n_t * n_s;

    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() +
                            (size_t)(nel_sx + nel_c + nel_out) * sizeof(float) + 65536;
    struct ggml_context * ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, d_inner, n_s);
    struct ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
    float * p = (float *) sx->data;
    for (int i = 0; i < nel_sx; i++) p[i] = rnd();
    p = (float *) c->data;
    for (int i = 0; i < nel_c; i++) p[i] = rnd();

    struct ggml_tensor * r = ggml_ssm_conv(ctx, sx, c);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); exit(1);
    }

    uint32_t dims[4] = {(uint32_t) d_conv, (uint32_t) d_inner, (uint32_t) n_t, (uint32_t) n_s};
    hdr(f, 10, dims, 4);
    fwrite(sx->data, 4, nel_sx, f);
    fwrite(c->data, 4, nel_c, f);
    fwrite(r->data, 4, nel_out, f);
    ggml_free(ctx);
}

/* kind 11 — ggml_ssm_scan (mamba2 selective scan) */
static void dump_ssm_scan(FILE *f, int d_state, int head_dim, int n_head, int n_group,
                          int n_tok, int n_seqs, int K, int a_ne0) {
    const int slots = n_seqs + 1;
    const int nel_s = d_state * head_dim * n_head * slots;
    const int nel_x = head_dim * n_head * n_tok * n_seqs;
    const int nel_dt = n_head * n_tok * n_seqs;
    const int nel_a = a_ne0 * n_head;
    const int nel_b = d_state * n_group * n_tok * n_seqs;
    const int nel_out = nel_x + K * d_state * head_dim * n_head * n_seqs;

    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() +
                            (size_t)(nel_s + nel_x + nel_dt + nel_a + 2*nel_b + nel_out) * sizeof(float)
                            + n_seqs * 4 + 65536;
    struct ggml_context * ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * s  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, slots);
    struct ggml_tensor * x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs);
    struct ggml_tensor * dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok, n_seqs);
    struct ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a_ne0, n_head);
    struct ggml_tensor * B  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    struct ggml_tensor * C  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok, n_seqs);
    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);

    float * p = (float *) s->data;
    for (int i = 0; i < nel_s; i++) p[i] = rnd();
    p = (float *) x->data;
    for (int i = 0; i < nel_x; i++) p[i] = rnd();
    p = (float *) dt->data;
    for (int i = 0; i < nel_dt; i++) p[i] = 0.2f + 0.4f * (rnd() * 0.5f + 0.5f); /* softplus range, well-conditioned */
    p = (float *) A->data;
    for (int i = 0; i < nel_a; i++) p[i] = -(0.2f + 1.3f * (rnd() * 0.5f + 0.5f)); /* strictly negative → decaying state */
    p = (float *) B->data;
    for (int i = 0; i < nel_b; i++) p[i] = rnd();
    p = (float *) C->data;
    for (int i = 0; i < nel_b; i++) p[i] = rnd();
    int32_t * pi = (int32_t *) ids->data;
    for (int i = 0; i < n_seqs; i++) pi[i] = (i == 0) ? 0 : (slots - 1); /* seq 1 -> last slot */

    struct ggml_tensor * r = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); exit(1);
    }

    uint32_t dims[8] = {(uint32_t) d_state, (uint32_t) head_dim, (uint32_t) n_head,
                        (uint32_t) n_group, (uint32_t) n_tok, (uint32_t) n_seqs,
                        (uint32_t) K, (uint32_t) a_ne0};
    hdr(f, 11, dims, 8);
    fwrite(s->data, 4, nel_s, f);
    fwrite(x->data, 4, nel_x, f);
    fwrite(dt->data, 4, nel_dt, f);
    fwrite(A->data, 4, nel_a, f);
    fwrite(B->data, 4, nel_b, f);
    fwrite(C->data, 4, nel_b, f);
    fwrite(ids->data, 4, n_seqs, f);
    fwrite(r->data, 4, nel_out, f);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    const char * out = argc > 1 ? argv[1] : "ssm_ref.bin";
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* ssm_conv: (d_conv-1) > 0, single and multi seq, single token */
    dump_ssm_conv(f, 3, 5, 4, 1);
    dump_ssm_conv(f, 4, 8, 6, 3);
    dump_ssm_conv(f, 4, 8, 1, 2);   /* n_t == 1: the state-only roll case */
    dump_ssm_conv(f, 2, 3, 5, 1);   /* d_conv == 2 */

    /* ssm_scan: granite-like (n_group == 1, scalar A), grouped (n_group 2/4),
     * multi-token + multi-seq, K > 1 rollback snapshots, single token */
    dump_ssm_scan(f, 4, 3, 2, 1, 3, 1, 1, 1);
    dump_ssm_scan(f, 8, 4, 4, 2, 5, 2, 1, 1);
    dump_ssm_scan(f, 8, 2, 8, 4, 3, 2, 3, 1); /* K = 3 snapshots */
    dump_ssm_scan(f, 4, 3, 2, 1, 1, 1, 1, 1); /* n_t == 1 */
    dump_ssm_scan(f, 5, 2, 3, 1, 4, 1, 1, 5); /* mamba-1 element-wise A (a_ne0 = d_state) */

    /* granite-4.0-h-* shape: d_state = 128 >= GGML_F32_STEP (64), so the
     * reference takes the AVX512 4x16-lane accumulation + scalar tail — the
     * lane structure the port must reproduce for bit-exactness */
    dump_ssm_scan(f, 128, 64, 48, 1, 5, 1, 1, 1);
    dump_ssm_scan(f, 128, 64, 48, 1, 1, 1, 1, 1);   /* single token: isolates the
                                                     * per-element state update
                                                     * (FMA vs mul/add) + reduce */

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", out);
    return 0;
}