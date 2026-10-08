/* wkv7_oob_proof.c — the empirical proof that the reference build's
 * ggml_rwkv_wkv7 output is NONDETERMINISTIC for head_size < GGML_F32_STEP
 * (64 on this AVX512F host), while head_size == 64 is stable.
 *
 * The x86 body of ggml_compute_forward_rwkv_wkv7_f32 (ops.cpp:11506-11560)
 * sweeps a fixed 64-float window (j += GGML_F32_STEP, kk < GGML_F32_ARR)
 * over r/w/k/a/b AND the state rows regardless of head_size. For head_size
 * < 64 the window leaves the tensors: the last heads of the last token read
 * past the {S,H,T} tensors' ends, and the last head's rows i > S-4 read AND
 * WRITE past the state area's end. What is there is ggml context-pool memory:
 * 32-byte ggml_object headers (whose `next` field is a heap POINTER — its
 * low 32 bits land in one float lane as an ASLR-dependent value) and, past
 * the dst tensor, the graph/hash-set objects — which the kernel itself then
 * overwrites mid-recurrence (the probe shows the region becoming -NaN).
 * The reference's sub-64 output therefore depends on heap addresses and
 * self-poisoned scratch: it differs from process run to process run.
 *
 * head_size == 64 (every public RWKV7 GGUF) never leaves the head row / the
 * head's own state block: i*64 + 63 <= 4095 == S*S-1, so no OOB access at
 * all — deterministic and exactly reproduced by the Rust port
 * (parity/wkv_ref.bin kind-22 S=64 section, bit-exact).
 *
 * Usage (each invocation = one fresh process = one ASLR draw):
 *   parity/wkv7_oob_proof S H T n_seqs
 * e.g.
 *   for i in $(seq 8); do parity/wkv7_oob_proof 16 4 3 1; done   # varies
 *   for i in $(seq 8); do parity/wkv7_oob_proof 64 2 5 1; done   # constant
 *
 * Build:
 *   gcc -O2 -o parity/wkv7_oob_proof parity/wkv7_oob_proof.c \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x2545F491u;
static float rnd(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(int32_t)lcg / (float)(1 << 30) * 0.5f;
}

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

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s S H T n_seqs\n", argv[0]); return 2; }
    const int S = atoi(argv[1]), H = atoi(argv[2]), T = atoi(argv[3]), n_seqs = atoi(argv[4]);

    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() +
                            (size_t)(6 * (S * H * T + 512) + S * S * H * n_seqs + 1024 +
                                     (size_t)S * H * (T + S * n_seqs) + 1024) * sizeof(float) + (1 << 20);
    struct ggml_context *ctx = ggml_init((struct ggml_init_params){ctx_size, NULL, false});

    struct ggml_tensor *r = mk_src(ctx, S, H, T);
    struct ggml_tensor *w = mk_src(ctx, S, H, T);
    struct ggml_tensor *k = mk_src(ctx, S, H, T);
    struct ggml_tensor *v = mk_src(ctx, S, H, T);
    struct ggml_tensor *a = mk_src(ctx, S, H, T);
    struct ggml_tensor *b = mk_src(ctx, S, H, T);
    struct ggml_tensor *st = mk_state(ctx, S, H, n_seqs);

    struct ggml_tensor *out = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, st);
    const int out_len = S * H * T + S * S * H * n_seqs;

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    struct ggml_tensor *gdst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    memset(gdst->data, 0, 1024 * sizeof(float));

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) return 1;

    uint64_t sum = 0;
    int oob = 0;
    for (int i = 0; i < out_len; i++) {
        float e = ((float *)out->data)[i];
        uint32_t bb; memcpy(&bb, &e, 4);
        sum = sum * 31 + bb;
        if (isnan(e)) oob++;
    }
    /* what the first OOB state read (S<64: last head, row S-3) sees — past
     * the src state tensor: its guard's ggml_object header, incl. the heap
     * pointer lane */
    float *st_end = (float *)st->data + S * S * H * n_seqs;
    uint32_t ptr_lane; memcpy(&ptr_lane, &st_end[4], 4);
    printf("S=%d H=%d T=%d seqs=%d: checksum=%016llx nan=%d st_guard_lane4=%08x (%g)\n",
           S, H, T, n_seqs, (unsigned long long)sum, oob, ptr_lane, st_end[4]);
    return 0;
}
