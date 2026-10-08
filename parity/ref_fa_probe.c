/* Row-count invariance probe for FLASH_ATTN_EXT (reference, AVX512 build).
 *
 * Question this probe answers (PARITY.md's FA row-shape tail): with byte-
 * identical Q/K/V/mask/sinks inputs, does the *reference's* attention change
 * when the same logical rows are computed as a T-row batch (a speculative
 * verify batch) instead of T separate 1-row batches (plain decode)?
 *
 * For a T-row case it runs, with nth threads:
 *   [A] one T-row graph                (the verify-batch shape)
 *   [B] T 1-row graphs, sliced         (the decode shape; q row t + mask row t
 *                                        + the shared k/v/sinks)
 *   [C] one (T-1)-row graph of rows 1..T-1 (partial-tile check)
 * and prints, per (t, head): the first element whose bits differ between A and
 * B, plus totals. Exit status is 0 even on differences (this is a probe).
 *
 * The input stream matches parity/ref_fa_dump.c's next_val so a case here can
 * be reproduced bit-for-bit in the Rust tests (flash_attn.rs's Rng differs —
 * the probe writes its inputs to the dump file instead).
 *
 * Output file format (probe mode): repeated records
 *   u32 tag, u32 t, u32 nth, u32 len, len bytes of f32 dst
 * where tag 0 = the T-row run [A], 1 = the 1-row slice run [B] for token t,
 * 2 = the (T-1)-row run [C]. Inputs (q/k/v/mask/sinks) are written first as
 *   u32 0xF17E0001 magic, then the u32-len sections q,k,v,mask,sinks and an
 *   f32 header { scale, max_bias, softcap } + i64 { dk,dv,h,h_kv,t,s_kv }.
 *
 * Build (repo root):
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -std=c11 parity/ref_fa_probe.c -I$PINNED/ggml/include \
 *       -o parity/ref_fa_probe -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_fa_probe parity/fa_probe.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

static uint64_t idx = 0;
static float next_val(void) {
    uint64_t i = idx++;
    return (float)((i * 2654435761ull) % 1000ull) / 1000.0f - 0.5f;
}

static void wr(FILE *f, const void *p, size_t n) {
    uint32_t len = (uint32_t)n;
    fwrite(&len, 4, 1, f);
    if (n > 0 && p != NULL) fwrite(p, 1, n, f);
}

typedef struct {
    int64_t dk, dv, h, h_kv, t, s_kv;
    int has_mask, has_sinks;
    int64_t causal_off; /* row t sees s <= t + off */
    float softcap, max_bias;
} pcfg;

/* fills q/k/v/mask/sinks exactly like ref_fa_dump.c's run_case */
static void fill_inputs(const pcfg *c, float *q, ggml_fp16_t *k, ggml_fp16_t *v,
                        ggml_fp16_t *mask, float *sinks) {
    const int64_t nq = c->dk * c->t * c->h;
    const int64_t nk = c->dk * c->s_kv * c->h_kv;
    const int64_t nv = c->dv * c->s_kv * c->h_kv;
    for (int64_t i = 0; i < nq; i++) q[i] = next_val();
    for (int64_t i = 0; i < nk; i++) k[i] = ggml_fp32_to_fp16(next_val());
    for (int64_t i = 0; i < nv; i++) v[i] = ggml_fp32_to_fp16(next_val());
    if (mask) {
        const int64_t off = c->causal_off < 0 ? c->s_kv - c->t : c->causal_off;
        for (int64_t t2 = 0; t2 < c->t; t2++)
            for (int64_t s = 0; s < c->s_kv; s++)
                mask[t2 * c->s_kv + s] =
                    s <= t2 + off ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
    }
    if (sinks)
        for (int64_t i = 0; i < c->h; i++) sinks[i] = next_val() * 4.0f;
}

/* runs attention over rows [t0, t1) of the T-row inputs and returns dst
 * (caller frees). q bytes/strides are copied per row so the sliced runs get
 * *identical* values; k/v/mask/sinks are shared. */
static float *run_slice(const pcfg *c, const float *q_full, const ggml_fp16_t *k,
                        const ggml_fp16_t *v, const ggml_fp16_t *mask_full,
                        const float *sinks, int t0, int t1, int nth) {
    const int64_t rows = t1 - t0;
    struct ggml_init_params ip = { .mem_size = 1024u * 1024u * 1024u, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);

    float *q_rows = malloc((size_t)(c->dk * rows * c->h) * sizeof(float));
    for (int64_t h = 0; h < c->h; h++)
        for (int64_t t = 0; t < rows; t++)
            memcpy(q_rows + (h * rows + t) * c->dk, q_full + (h * c->t + (t0 + t)) * c->dk,
                   (size_t)c->dk * sizeof(float));

    ggml_fp16_t *mask_rows = NULL;
    if (mask_full) {
        mask_rows = malloc((size_t)(c->s_kv * rows) * sizeof(ggml_fp16_t));
        for (int64_t t = 0; t < rows; t++)
            memcpy(mask_rows + t * c->s_kv, mask_full + (t0 + t) * c->s_kv,
                   (size_t)c->s_kv * sizeof(ggml_fp16_t));
    }

    struct ggml_tensor *qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c->dk, rows, c->h, 1);
    struct ggml_tensor *kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c->dk, c->s_kv, c->h_kv, 1);
    struct ggml_tensor *vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c->dv, c->s_kv, c->h_kv, 1);
    struct ggml_tensor *mt = mask_rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c->s_kv, rows) : NULL;
    struct ggml_tensor *st = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c->h) : NULL;

    memcpy(qt->data, q_rows, (size_t)ggml_nbytes(qt));
    memcpy(kt->data, k, (size_t)ggml_nbytes(kt));
    memcpy(vt->data, v, (size_t)ggml_nbytes(vt));
    if (mt) memcpy(mt->data, mask_rows, (size_t)ggml_nbytes(mt));
    if (st) memcpy(st->data, sinks, (size_t)ggml_nbytes(st));

    const float scale = 1.0f / sqrtf((float)c->dk);
    struct ggml_tensor *out_t = ggml_flash_attn_ext(ctx, qt, kt, vt, mt, scale, c->max_bias, c->softcap);
    if (st) ggml_flash_attn_ext_add_sinks(out_t, st);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);
    struct ggml_context *ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, nth);

    /* dst layout [DV, H, rows] — repack to (t, h, d) row-major for the caller */
    const int64_t ne0 = out_t->ne[0], ne1 = out_t->ne[1], ne2 = out_t->ne[2];
    const float *src = (const float *)out_t->data;
    float *out = malloc((size_t)(ne0 * ne1 * ne2) * sizeof(float));
    for (int64_t t = 0; t < ne2; t++)
        for (int64_t h = 0; h < ne1; h++)
            for (int64_t d = 0; d < ne0; d++)
                out[(t * ne1 + h) * ne0 + d] = src[(t * ne1 + h) * ne0 + d];

    free(q_rows);
    free(mask_rows);
    ggml_free(ctxc);
    ggml_free(ctx);
    return out;
}

static int bits_diff(const float *a, const float *b, int64_t n) {
    int d = 0;
    for (int64_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], 4) != 0) d++;
    return d;
}

static void probe_case(FILE *f, const char *name, const pcfg *c, int nth) {
    const int64_t nq = c->dk * c->t * c->h;
    const int64_t nk = c->dk * c->s_kv * c->h_kv;
    const int64_t nv = c->dv * c->s_kv * c->h_kv;
    float *q = malloc(nq * sizeof(float));
    ggml_fp16_t *k = malloc(nk * 2);
    ggml_fp16_t *v = malloc(nv * 2);
    ggml_fp16_t *m = c->has_mask ? malloc(c->s_kv * c->t * 2u) : NULL;
    float *sinks = c->has_sinks ? malloc(c->h * sizeof(float)) : NULL;
    fill_inputs(c, q, k, v, m, sinks);

    float scale = 1.0f / sqrtf((float)c->dk);
    if (f) {
        uint32_t magic = 0xF17E0001u;
        fwrite(&magic, 4, 1, f);
        int64_t geom[6] = { c->dk, c->dv, c->h, c->h_kv, c->t, c->s_kv };
        float knobs[3] = { scale, c->max_bias, c->softcap };
        wr(f, geom, sizeof(geom));
        wr(f, knobs, sizeof(knobs));
        wr(f, q, nq * 4);
        wr(f, k, nk * 2);
        wr(f, v, nv * 2);
        if (m != NULL) { wr(f, m, c->s_kv * c->t * 2u); } else { wr(f, NULL, 0); }
        if (sinks != NULL) { wr(f, sinks, c->h * 4); } else { wr(f, NULL, 0); }
    }

    const int64_t row_elems = c->dv * c->h;
    float *batch = run_slice(c, q, k, v, m, sinks, 0, c->t, nth);
    if (f) { uint32_t hdr[3] = { 0, (uint32_t)c->t, (uint32_t)nth }; fwrite(hdr, 4, 3, f); wr(f, batch, row_elems * c->t * 4); }

    int total = 0;
    for (int t = 0; t < c->t; t++) {
        float *one = run_slice(c, q, k, v, m, sinks, t, t + 1, nth);
        if (f) { uint32_t hdr[3] = { 1, (uint32_t)t, (uint32_t)nth }; fwrite(hdr, 4, 3, f); wr(f, one, row_elems * 4); }
        int d = bits_diff(batch + t * row_elems, one, row_elems);
        if (d > 0) {
            printf("  %s nth=%d t=%d: %d/%lld elements differ (batch vs 1-row)\n",
                   name, nth, t, d, (long long)row_elems);
        }
        total += d;
        free(one);
    }
    if (c->t >= 2) {
        float *rest = run_slice(c, q, k, v, m, sinks, 1, c->t, nth);
        if (f) { uint32_t hdr[3] = { 2, (uint32_t)(c->t - 1), (uint32_t)nth }; fwrite(hdr, 4, 3, f); wr(f, rest, row_elems * (c->t - 1) * 4); }
        int d = bits_diff(batch + row_elems, rest, row_elems * (c->t - 1));
        if (d > 0) printf("  %s nth=%d rows[1..%lld): %d elements differ (batch vs (T-1)-row)\n",
                          name, nth, (long long)c->t, d);
        free(rest);
    }
    printf("  %s nth=%d T=%lld S_kv=%lld: batch-vs-slices total %d / %lld differ\n",
           name, nth, (long long)c->t, (long long)c->s_kv, total,
           (long long)(row_elems * c->t));
    free(batch);
    free(q); free(k); free(v); free(m); free(sinks);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : NULL;
    FILE *f = out ? fopen(out, "wb") : NULL;
    if (out && !f) { perror("fopen"); return 1; }

    /* gpt-oss decode/verify geometry: 64 q heads / 8 kv heads, sinks, F16 KV.
     * ctx = 21 positions, the last 5 are the verify batch (window 16). */
    pcfg goss = { 64, 64, 64, 8, 5, 21, 1, 1, -1, 0.0f, 0.0f };
    /* gpt-oss prefill of the 5-token prompt (mode 14's shape) */
    pcfg pref = { 64, 64, 64, 8, 5, 5, 1, 1, -1, 0.0f, 0.0f };
    /* a 4-row verify batch (deepseek-MTP-like row count, small S_kv) */
    pcfg v4 = { 64, 64, 16, 2, 4, 9, 1, 0, -1, 0.0f, 0.0f };
    /* decode past the split-KV threshold (neq1==1 && nek1>=512): a T=4 batch
     * at S_kv=600 vs its T=1 slices — at nth>1 the reference's 1-row runs take
     * use_split_kv_path (ops.cpp:9261) while the 4-row run takes one_chunk,
     * so this measures the reference's cross-path row invariance too. */
    pcfg v4l = { 64, 64, 16, 2, 4, 600, 1, 0, 596, 0.0f, 0.0f };
    /* decode-T1 past the split-KV threshold (trivially self-consistent) */
    pcfg splitk = { 64, 64, 8, 1, 1, 600, 1, 0, 599, 0.0f, 0.0f };

    for (int nth = 1; nth <= 8; nth += 7) {
        probe_case(f, "goss-verify-T5", &goss, nth);
        probe_case(f, "goss-prefill-T5", &pref, nth);
        probe_case(f, "verify-T4", &v4, nth);
        probe_case(f, "verify-T4-S600", &v4l, nth);
        probe_case(f, "decode-T1-S600", &splitk, nth);
    }
    if (f) { fclose(f); printf("wrote %s\n", out); }
    return 0;
}
