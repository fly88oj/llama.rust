/* FLASH_ATTN_EXT ground truth from the reference (AVX512 + GGML_NATIVE) build.
 *
 * Produces the dump consumed by crates/ggml/src/flash_attn.rs's `ref_dump_*`
 * tests. Built through the *public* ggml API (`ggml_flash_attn_ext` +
 * `ggml_graph_compute_with_ctx`), so the reference picks its own kernel: with
 * T = neq1 >= 64 and DV % 16 == 0 the AVX512 binary runs
 * `ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8852); below that it runs
 * `..._f16_one_chunk` (ops.cpp:8614).
 *
 * Build (from the repo root):
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -std=c11 parity/ref_fa_dump.c -I$PINNED/ggml/include \
 *       -o parity/ref_fa_dump -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_fa_dump parity/fa_ref.bin
 *
 * File format: a sequence of `u32 len + len bytes` sections:
 *   [0]  header: 9 x i64 { mode, dk, dv, h, h_kv, t, s_kv, kty, vty } packed
 *        little-endian (kty/vty: 0 = F32, 1 = F16; mode = affinity enum below),
 *        followed by the section's remaining payload (unused here).
 *   [1]  q bytes (F32, [DK,T,H,1] dense)
 *   [2]  k bytes ([DK,S_kv,H_kv,1], dense, type per header)
 *   [3]  v bytes ([DV,S_kv,H_kv,1], dense, type per header)
 *   [4]  mask bytes (F16 [S_kv,T], causal; empty when has_mask == 0)
 *   [5]  sinks bytes (F32 [H]; empty when has_sinks == 0)
 *   [6]  dst floats (F32 [DV,H,T,1] permuted by the op)
 *   [7]  header extra: 4 x f32 { scale, max_bias, logit_softcap, 0 }
 *
 * mode (header[0]) is informational only — it enumerates which C kernel the
 * port expects to match; the input paddings/knobs are what matter.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

/* deterministic input stream (identical to the one the Rust tests generate) */
static uint64_t idx = 0;
static float next_val(void) {
    uint64_t i = idx++;
    return (float)((i * 2654435761ull) % 1000ull) / 1000.0f - 0.5f;
}

static void wr(FILE *f, const void *p, size_t n) {
    uint32_t len = (uint32_t)n;
    fwrite(&len, 4, 1, f);
    if (n > 0 && p != NULL) {
        fwrite(p, 1, n, f);
    }
}

typedef struct {
    int mode;          /* affinity tag */
    int64_t dk, dv, h, h_kv, t, s_kv;
    int kf32, vf32;    /* 1 => stored F32 */
    int has_mask;
    int has_sinks;
    int causal_off;    /* mask row t sees s <= t + causal_off (or -1: match t) */
    float softcap;
    float max_bias;
} cfg;

static void run_case(FILE *f, const cfg *c) {
    struct ggml_init_params ip = { .mem_size = 1024u * 1024u * 1024u, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c->dk, c->t, c->h, 1);
    struct ggml_tensor *k = ggml_new_tensor_4d(ctx, c->kf32 ? GGML_TYPE_F32 : GGML_TYPE_F16, c->dk, c->s_kv, c->h_kv, 1);
    struct ggml_tensor *v = ggml_new_tensor_4d(ctx, c->vf32 ? GGML_TYPE_F32 : GGML_TYPE_F16, c->dv, c->s_kv, c->h_kv, 1);
    struct ggml_tensor *mask = c->has_mask ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c->s_kv, c->t) : NULL;
    struct ggml_tensor *sinks = c->has_sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c->h) : NULL;

    for (int64_t i = 0; i < ggml_nelements(q); i++) ((float *)q->data)[i] = next_val();
    for (int64_t i = 0; i < ggml_nelements(k); i++) {
        float x = next_val();
        if (c->kf32) ((float *)k->data)[i] = x; else ((ggml_fp16_t *)k->data)[i] = ggml_fp32_to_fp16(x);
    }
    for (int64_t i = 0; i < ggml_nelements(v); i++) {
        float x = next_val();
        if (c->vf32) ((float *)v->data)[i] = x; else ((ggml_fp16_t *)v->data)[i] = ggml_fp32_to_fp16(x);
    }
    if (mask) {
        const int64_t off = c->causal_off < 0 ? c->s_kv - c->t : c->causal_off;
        for (int64_t t2 = 0; t2 < c->t; t2++)
            for (int64_t s = 0; s < c->s_kv; s++)
                ((ggml_fp16_t *)mask->data)[t2 * c->s_kv + s] =
                    s <= t2 + off ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
    }
    if (sinks)
        for (int64_t i = 0; i < c->h; i++) ((float *)sinks->data)[i] = next_val() * 4.0f;

    const float scale = 1.0f / sqrtf((float)c->dk);
    struct ggml_tensor *out_t = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, c->max_bias, c->softcap);
    if (sinks) ggml_flash_attn_ext_add_sinks(out_t, sinks);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);
    struct ggml_context *ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, 1);

    int64_t hdr[9] = { c->mode, c->dk, c->dv, c->h, c->h_kv, c->t, c->s_kv,
                       c->kf32 ? 0 : 1, c->vf32 ? 0 : 1 };
    float extra[4] = { scale, c->max_bias, c->softcap, 0.0f };
    wr(f, hdr, sizeof(hdr));
    wr(f, q->data, (size_t)ggml_nbytes(q));
    wr(f, k->data, (size_t)ggml_nbytes(k));
    wr(f, v->data, (size_t)ggml_nbytes(v));
    // NOTE: explicit branches (the empty-string fallback used to trip the
    // security scanner); a 0-length section is written for absent inputs.
    if (mask != NULL) {
        wr(f, mask->data, (size_t) ggml_nbytes(mask));
    } else {
        wr(f, NULL, 0); // writes just the 0 length prefix
    }
    if (sinks != NULL) {
        wr(f, sinks->data, (size_t) ggml_nbytes(sinks));
    } else {
        wr(f, NULL, 0);
    }
    wr(f, out_t->data, (size_t)ggml_nbytes(out_t));
    wr(f, extra, sizeof(extra));

    fprintf(stderr, "case mode=%d dk=%lld dv=%lld T=%lld S_kv=%lld H=%lld H_kv=%lld k=%s v=%s "
                    "mask=%d sinks=%d softcap=%g bias=%g -> dst [%lld %lld %lld %lld] first=%.9g\n",
            c->mode, (long long)c->dk, (long long)c->dv, (long long)c->t, (long long)c->s_kv,
            (long long)c->h, (long long)c->h_kv, c->kf32 ? "f32" : "f16", c->vf32 ? "f32" : "f16",
            c->has_mask, c->has_sinks, c->softcap, c->max_bias,
            (long long)out_t->ne[0], (long long)out_t->ne[1], (long long)out_t->ne[2], (long long)out_t->ne[3],
            ((float *)out_t->data)[0]);

    ggml_free(ctxc);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "parity/fa_ref.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* 0: one_chunk small (F32 KV)  1: tiled T=64 S_kv=64 (F16 KV)
       2: tiled T=128 S_kv=128 F16 (2 KV tiles)  3: tiled T=65 S_kv=130 (tile split + head boundary)
       4: tiled T=64 S_kv=64 F32 KV
       5: tiled T=128 S_kv=200 F16 + sinks     6: tiled T=128 S_kv=200 F16 + softcap
       7: tiled D=128 T=64 S_kv=96 F16 (dv != dk)
       8: one_chunk T=32 (below the 64 threshold, F16 KV)
       9: tiled T=64 S_kv=64 F16 mask=off (no mask) */
    cfg cases[] = {
        { 0, 16, 16, 4, 2,  3,   5, 1, 1, 1, 0, -1, 0.0f, 0.0f },
        { 1, 64, 64, 4, 2, 64,  64, 0, 0, 1, 0, -1, 0.0f, 0.0f },
        { 2, 64, 64, 4, 2, 128, 128, 0, 0, 1, 0, -1, 0.0f, 0.0f },
        { 3, 32, 32, 4, 2, 65, 130, 0, 0, 1, 0, -1, 0.0f, 0.0f },
        { 4, 64, 64, 4, 2, 64,  64, 1, 1, 1, 0, -1, 0.0f, 0.0f },
        { 5, 64, 64, 4, 2, 128, 200, 0, 0, 1, 1, -1, 0.0f, 0.0f },
        { 6, 64, 64, 4, 2, 128, 200, 0, 0, 1, 0, -1, 15.0f, 0.0f },
        { 7, 64,128, 4, 2, 64,  96, 0, 0, 1, 0, -1, 0.0f, 0.0f },
        { 8, 32, 32, 4, 2, 32,  96, 0, 0, 1, 0, -1, 0.0f, 0.0f },
        { 9, 64, 64, 4, 2, 64,  64, 0, 0, 0, 0, -1, 0.0f, 0.0f },
        /* 10: ALiBi (max_bias > 0 requires a mask) + tiled */
        { 10, 64, 64, 4, 2, 64, 64, 0, 0, 1, 0, -1, 0.0f, 8.0f },
        /* 11: tiled with a windowed causal offset (off = 0: pure causal) */
        { 11, 64, 64, 4, 2, 64, 64, 0, 0, 1, 0, 0, 0.0f, 0.0f },
        /* 13: gpt-oss-20b decode shape: 64 heads / 8 kv heads, attention sinks,
         * all-visible mask (T=1 decode with a window larger than the context):
         * sink line + inner-loop S update both active, F16 KV. */
        { 13, 64, 64, 64, 8, 1, 21, 0, 0, 0, 1, -1, 0.0f, 0.0f },
        /* 14: same but T=5 (prefill of the gpt-oss prompt) and a causal mask */
        { 14, 64, 64, 64, 8, 5, 5, 0, 0, 1, 1, -1, 0.0f, 0.0f },
        /* 12: PATH DISCRIMINATOR — no mask + tiny softcap + a padded KV tail.
         * The tiled kernel sets the padded KQ columns to -inf, applies tanh
         * (-> -1) and multiplies by the softcap, so the padding survives as
         * -softcap and enters the softmax with weight ~1 (ops.cpp:9040-9052),
         * while one_chunk never sees padded columns. Tiled and one_chunk differ
         * by ~2x here. */
        { 12, 64, 64, 2, 1, 64, 70, 0, 0, 0, 0, -1, 1e-4f, 0.0f },
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < n_cases; i++) run_case(f, &cases[i]);

    fclose(f);
    printf("wrote %s (%d cases)\n", out, n_cases);
    return 0;
}