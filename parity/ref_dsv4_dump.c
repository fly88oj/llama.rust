/* DSV4 op-kernel dumper: builds the deepseek4 hyper-connection / swiglu-clamp
 * / sqrt / rope-back op graphs on deterministic inputs, computes them with the
 * reference CPU backend and writes the raw f32 outputs to
 * parity/dsv4_ops_ref.bin — the oracle for the Rust kernel tests
 * (crates/ggml/src/compute.rs dsv4 oracle, arch batch 7).
 *
 * Build + run (see PARITY.md; the reference tree is pinned at bd4f514db1):
 *
 *   gcc parity/ref_dsv4_dump.c -O2 -o /tmp/ref_dsv4_dump \
 *     -I /home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L /home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lggml -lggml-base -lggml-cpu \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   /tmp/ref_dsv4_dump parity/dsv4_ops_ref.bin
 *
 * File layout (all f32, section offsets recorded in the header):
 *   magic 'D','S','V','4' | u32 n_sections | u32 sizes[n_sections] (elements)
 *   then the raw outputs, in program order.
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the same LCG the Rust tests use — bit-identical inputs on both sides */
static uint32_t lcg_state;
static float lcg_next(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return ((lcg_state >> 8) & 0xffffff) / 8388608.0f * 0.2f - 0.1f;
}

static float out_buf[1 << 20];
static int out_elems = 0;
static int n_sections = 0;
static int sec_sizes[64];

static void dump_tensor(struct ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t n = ggml_nelements(t);
    memcpy(out_buf + out_elems, t->data, n * sizeof(float));
    out_elems += (int) n;
    sec_sizes[n_sections++] = (int) n;
}

int main(int argc, char **argv) {
    const char * out_path;
    if (argc > 1) {
        out_path = argv[1];
    } else {
        out_path = "parity/dsv4_ops_ref.bin";
    }

    /* ---------------- dsv4_hc_comb ---------------- */
    {
        // hc = 4 -> hc_mix_dim = 24; two token rows, eps/n_iter variants
        const int64_t ne_m[2] = {24, 3};
        const int64_t ne_s[1] = {3};
        const int64_t ne_b[1] = {24};

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_m[0], ne_m[1]);
        struct ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne_s[0]);
        struct ggml_tensor * base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne_b[0]);

        lcg_state = 0xd5440001u;
        float * pm = (float *) mixes->data;
        for (int i = 0; i < ggml_nelements(mixes); i++) { pm[i] = lcg_next(); }
        float * ps = (float *) scale->data;
        ps[0] = 0.9f; ps[1] = 1.1f; ps[2] = 1.3f;
        float * pb = (float *) base->data;
        for (int i = 0; i < ggml_nelements(base); i++) { pb[i] = 0.02f * lcg_next(); }

        const float eps_list[2] = {1e-3f, 1e-2f};
        const int32_t iter_list[2] = {1, 3};

        for (int v = 0; v < 4; v++) {
            struct ggml_tensor * comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, eps_list[v & 1], iter_list[v >> 1]);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, comb);
            ggml_graph_compute_with_ctx(ctx, gf, 4);
            dump_tensor(comb);
        }
        ggml_free(ctx);
    }

    /* ---------------- dsv4_hc_pre (non-gated) ---------------- */
    {
        const int64_t n_embd = 12;
        const int64_t hc = 4;
        const int64_t nt = 3;

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);

        lcg_state = 0x1234abcdu;
        float * px = (float *) x->data;
        for (int i = 0; i < ggml_nelements(x); i++) { px[i] = lcg_next(); }
        float * pw = (float *) w->data;
        for (int i = 0; i < ggml_nelements(w); i++) { pw[i] = 0.5f + 0.5f * lcg_next(); }

        struct ggml_tensor * pre = ggml_dsv4_hc_pre(ctx, x, w);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, pre);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        dump_tensor(pre);
        ggml_free(ctx);
    }

    /* ---------------- dsv4_hc_post (comb present / absent) ---------------- */
    {
        const int64_t n_embd = 10;
        const int64_t hc = 4;
        const int64_t nt = 2;

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt);
        struct ggml_tensor * r  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
        struct ggml_tensor * po = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);
        struct ggml_tensor * cb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, nt);

        lcg_state = 0xbeef0042u;
        float * px = (float *) x->data;
        for (int i = 0; i < ggml_nelements(x); i++) { px[i] = lcg_next(); }
        float * pr = (float *) r->data;
        for (int i = 0; i < ggml_nelements(r); i++) { pr[i] = lcg_next(); }
        float * pp = (float *) po->data;
        for (int i = 0; i < ggml_nelements(po); i++) { pp[i] = 0.5f + 0.5f * lcg_next(); }
        float * pc = (float *) cb->data;
        for (int i = 0; i < ggml_nelements(cb); i++) { pc[i] = 0.1f * lcg_next() + 0.05f; }

        struct ggml_cgraph * gf = ggml_new_graph(ctx);

        struct ggml_tensor * with_comb = ggml_dsv4_hc_post(ctx, x, r, po, cb);
        ggml_build_forward_expand(gf, with_comb);

        struct ggml_tensor * no_comb = ggml_dsv4_hc_post(ctx, x, r, po, NULL);
        ggml_build_forward_expand(gf, no_comb);

        ggml_graph_compute_with_ctx(ctx, gf, 4);
        dump_tensor(with_comb);
        dump_tensor(no_comb);
        ggml_free(ctx);
    }

    /* ---------------- swiglu_clamp (f32) ---------------- */
    {
        const int64_t n = 37;

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 4);
        struct ggml_tensor * up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 4);

        lcg_state = 0x5eed5f01u;
        float * pg = (float *) gate->data;
        for (int i = 0; i < ggml_nelements(gate); i++) { pg[i] = 30.0f * lcg_next(); }
        float * pu = (float *) up->data;
        for (int i = 0; i < ggml_nelements(up); i++) { pu[i] = 30.0f * lcg_next(); }

        struct ggml_cgraph * gf = ggml_new_graph(ctx);

        const float limit_list[3] = {7.0f, 0.05f, 1e-6f};
        struct ggml_tensor * outs[3];
        for (int v = 0; v < 3; v++) {
            outs[v] = ggml_swiglu_clamp(ctx, gate, up, limit_list[v]);
            ggml_build_forward_expand(gf, outs[v]);
        }
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        for (int v = 0; v < 3; v++) { dump_tensor(outs[v]); }
        ggml_free(ctx);
    }

    /* ---------------- sqrt ---------------- */
    {
        const int64_t n = 128;

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        lcg_state = 0xcafef00du;
        float * pa = (float *) a->data;
        for (int i = 0; i < n; i++) {
            float v = lcg_next();
            if (v < 0.0f) { v = -v; }
            pa[i] = 100.0f * v;
        }

        struct ggml_tensor * s = ggml_sqrt(ctx, a);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, s);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        dump_tensor(s);
        ggml_free(ctx);
    }

    /* ---------------- rope_ext_back (NEOX, n_offs = 8) ---------------- */
    {
        // deepseek4 de-ropes with rope_set_offset(n_embd_head_nope = 8) —
        // rotate only channels [8, 16) of a 16-wide head (NEOX half-split)
        const int64_t n_embd_head = 16;
        const int64_t n_head = 3;
        const int64_t nt = 5;

        struct ggml_init_params ip = {64u << 20, NULL, false};
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, nt);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);

        lcg_state = 0x12345678u;
        float * pc = (float *) cur->data;
        for (int i = 0; i < ggml_nelements(cur); i++) { pc[i] = lcg_next(); }
        int32_t * pp = (int32_t *) pos->data;
        for (int i = 0; i < nt; i++) { pp[i] = 40 + 7 * i; }

        struct ggml_tensor * rb = ggml_rope_ext_back(ctx, cur, pos, NULL, 8,
                GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        rb = ggml_rope_set_offset(rb, 8);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, rb);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        dump_tensor(rb);
        ggml_free(ctx);
    }

    /* write the dump */
    FILE * f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    unsigned char magic[4] = {'D','S','V','4'};
    fwrite(magic, 1, 4, f);
    fwrite(&n_sections, sizeof(int), 1, f);
    fwrite(sec_sizes, sizeof(int), n_sections, f);
    fwrite(out_buf, sizeof(float), out_elems, f);
    fclose(f);

    printf("dsv4 ops dump: %d sections, %d floats -> %s\n", n_sections, out_elems, out_path);
    return 0;
}
