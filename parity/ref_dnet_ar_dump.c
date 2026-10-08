// ref_dnet_ar_dump.c — the reference-side twin of the port's
// build_delta_net_autoregressive bit-compare: the op chain of
// src/models/delta-net-base.cpp:289-374 built verbatim with the reference
// ggml over deterministic inputs (the llama-level path is unreachable — the
// CPU build always runs the fused op, llama-context.cpp:233).
//
// build:
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       parity/ref_dnet_ar_dump.c -o parity/ref_dnet_ar_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run: ./parity/ref_dnet_ar_dump <out.bin>
//
// File format: char magic[8] = "DNETAR\0\0", then o then s_new as raw F32
// (S_v*H_v + S_v*S_v*H_v floats, n_seqs = 1).

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

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "/tmp/mtp2/dnet-ar-ref.bin";
    const int S = 8, H = 3;

    struct ggml_init_params ip = { .mem_size = 64*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    // the inputs (the same stream the port test builds, same order)
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, 1);
    struct ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, 1);
    struct ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    rs = 0xbeed171;
    for (int i = 0; i < S*H; i++) ((float *) q->data)[i] = 0.09f * rng_next();
    for (int i = 0; i < S*H; i++) ((float *) k->data)[i] = 0.09f * rng_next();
    for (int i = 0; i < S*H; i++) ((float *) v->data)[i] = 0.11f * rng_next();
    for (int i = 0; i < H;   i++) ((float *) g->data)[i] = -0.05f - 0.4f * (0.5f * rng_next() + 0.5f);
    for (int i = 0; i < H;   i++) ((float *) b->data)[i] = 0.5f + 0.45f * (0.5f * rng_next() + 0.5f);
    for (int i = 0; i < S*S*H; i++) ((float *) s->data)[i] = 0.03f * rng_next();

    // delta-net-base.cpp:325-370, verbatim
    const float scale = 1.0f / sqrtf((float) S);
    struct ggml_tensor * qq = ggml_scale(ctx, q, scale);
    qq = ggml_permute(ctx, qq, 0, 2, 1, 3);
    struct ggml_tensor * kk = ggml_permute(ctx, k, 0, 2, 1, 3);
    struct ggml_tensor * vv = ggml_permute(ctx, v, 0, 2, 1, 3);

    struct ggml_tensor * gg = ggml_reshape_4d(ctx, g, 1, g->ne[0], H, 1);
    struct ggml_tensor * bb = ggml_reshape_4d(ctx, b, 1, 1, H, 1);

    gg = ggml_exp(ctx, gg);
    struct ggml_tensor * ss = ggml_mul(ctx, s, gg);

    struct ggml_tensor * sk = ggml_mul(ctx, ss, kk);
    sk = ggml_sum_rows(ctx, sk);

    struct ggml_tensor * d = ggml_sub(ctx, vv, ggml_transpose(ctx, sk));
    d = ggml_mul(ctx, d, bb);

    struct ggml_tensor * d_t = ggml_transpose(ctx, d);

    struct ggml_tensor * kr = ggml_repeat(ctx, kk, ss);
    struct ggml_tensor * kd = ggml_mul(ctx, kr, d_t);
    ss = ggml_add(ctx, ss, kd);

    struct ggml_tensor * s_q = ggml_mul(ctx, ss, qq);
    struct ggml_tensor * o = ggml_sum_rows(ctx, s_q);
    o = ggml_permute(ctx, o, 2, 0, 1, 3);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, ss);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    FILE * f = fopen(out_path, "wb");
    fwrite("DNETAR\0\0", 1, 8, f);
    fwrite(o->data, 4, S*H, f);
    fwrite(ss->data, 4, S*S*H, f);
    fclose(f);
    printf("ok %s\n", out_path);
    return 0;
}
