// ref_dnet_ch_dump.c — the reference-side twin of the port's
// build_delta_net_chunking bit-compare: the op chain of
// src/models/delta-net-base.cpp:17-287 built verbatim with the reference
// ggml over deterministic inputs. The llama-level path is unreachable (the
// CPU build always runs the fused op — fused_gdn_ch is hardwired true,
// llama-context.cpp:234, and no CLI flag clears it), so this ggml-level
// chain IS the acceptance, exactly like batch 17's ref_dnet_ar_dump.c.
//
// Both the GDA branch (g [1, H_v, T, B], CS=64) and the KDA branch
// (g [S_k, H_k, T, B], CS=16) are exercised; the KDA case needs H_k == H_v.
//
// build:
//   gcc -O2 -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       parity/ref_dnet_ch_dump.c -o parity/ref_dnet_ch_dump \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lggml -lggml-base -lggml-cpu -lm \
//       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
// run: ./parity/ref_dnet_ch_dump <out.bin>
//
// File format: char magic[8] = "DNETCH\0\0", then per case (GDA first, KDA
// second): [i32 n_elems][o as raw F32][i32 n_elems][s_new as raw F32].

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

// utility to get one slice from the third dimension (delta-net-base.cpp:6-11)
static struct ggml_tensor * get_slice_2d(struct ggml_context * ctx0, struct ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

// delta-net-base.cpp:17-287 `build_delta_net_chunking`, verbatim
static void build_delta_net_chunking(struct ggml_context * ctx0, struct ggml_cgraph * gf,
        struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
        struct ggml_tensor * g, struct ggml_tensor * b, struct ggml_tensor * s,
        struct ggml_tensor ** o_out, struct ggml_tensor ** s_out) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    const float scale = 1.0f / sqrtf((float) S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    struct ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    struct ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));

    struct ggml_tensor * kb = NULL;
    struct ggml_tensor * kq = NULL;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        struct ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        struct ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        struct ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        struct ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        struct ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        struct ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        struct ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        struct ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        struct ggml_tensor * g_cs_i = g_cs;
        struct ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        struct ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);

    // [CS, CS, n_chunks, H_k * n_seqs]
    struct ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);

    struct ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    struct ggml_tensor * lhs = ggml_add(ctx0, attn, identity);

    attn = ggml_neg(ctx0, attn);

    struct ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    struct ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    struct ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);

    // [1, CS, n_chunks, H_v * n_seqs] KDA: [S_k, CS, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    struct ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // get last element in g_cumsum along CS dimension (ne0)
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    struct ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));

    struct ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    struct ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    struct ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    struct ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        struct ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        struct ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        struct ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        struct ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        struct ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        struct ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);

        // [CS, S_v, 1, H_v * n_seqs]
        struct ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);

        // [S_v, CS, 1, H_v * n_seqs]
        struct ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);

        // [S_v, CS, 1, H_v * n_seqs]
        struct ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);

        // [S_v, CS, 1, H_v * n_seqs]
        struct ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        struct ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        struct ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
    }

    // truncate padded tokens
    struct ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);

    ggml_build_forward_expand(gf, o);
    ggml_build_forward_expand(gf, s);
    *o_out = o;
    *s_out = s;
}

static void dump_tensor(FILE * f, const struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    int32_t n32 = (int32_t) n;
    fwrite(&n32, 4, 1, f);
    fwrite(t->data, 4, n, f);
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "/tmp/mtp2/dnet-ch-ref.bin";

    FILE * f = fopen(out_path, "wb");
    fwrite("DNETCH\0\0", 1, 8, f);

    // ---- case 1: GDA (qwen3next geometry) — CS=64, 2 chunks + padding ----
    {
        struct ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context * ctx = ggml_init(ip);

        // S=8, H=3, T=70 (pad->128, 2 chunks), B=2
        const int S = 8, H = 3, T = 70, B = 2;
        struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        struct ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
        rs = 0xbeed171;
        for (int i = 0; i < S*H*T*B; i++) ((float *) q->data)[i] = 0.09f * rng_next();
        for (int i = 0; i < S*H*T*B; i++) ((float *) k->data)[i] = 0.09f * rng_next();
        for (int i = 0; i < S*H*T*B; i++) ((float *) v->data)[i] = 0.11f * rng_next();
        for (int i = 0; i <   H*T*B; i++) ((float *) g->data)[i] = -0.05f - 0.4f * (0.5f * rng_next() + 0.5f);
        for (int i = 0; i <   H*T*B; i++) ((float *) b->data)[i] = 0.5f + 0.45f * (0.5f * rng_next() + 0.5f);
        for (int i = 0; i < S*S*H*B; i++) ((float *) s->data)[i] = 0.03f * rng_next();

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * o = NULL, * sn = NULL;
        build_delta_net_chunking(ctx, gf, q, k, v, g, b, s, &o, &sn);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        // o is permuted [S_v, H_v, T, B]; read it through its own layout
        {
            float * buf = calloc(ggml_nelements(o), 4);
            for (int64_t i3 = 0; i3 < o->ne[3]; i3++)
                for (int64_t i2 = 0; i2 < o->ne[2]; i2++)
                    for (int64_t i1 = 0; i1 < o->ne[1]; i1++)
                        for (int64_t i0 = 0; i0 < o->ne[0]; i0++) {
                            buf[i0 + o->ne[0]*(i1 + o->ne[1]*(i2 + o->ne[2]*i3))] =
                                *(float *) ((char *) o->data + i0*o->nb[0] + i1*o->nb[1] + i2*o->nb[2] + i3*o->nb[3]);
                        }
            int32_t n32 = (int32_t) ggml_nelements(o);
            fwrite(&n32, 4, 1, f);
            fwrite(buf, 4, ggml_nelements(o), f);
            free(buf);
            dump_tensor(f, sn);
        }
        ggml_free(ctx);
    }

    // ---- case 2: KDA — CS=16, 3 chunks (T=40), H_k == H_v ---------------
    {
        struct ggml_init_params ip = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context * ctx = ggml_init(ip);

        const int S = 6, H = 2, T = 40, B = 1;
        struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
        struct ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B); // KDA
        struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
        struct ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
        rs = 0x5dca11;
        for (int i = 0; i < S*H*T*B; i++) ((float *) q->data)[i] = 0.09f * rng_next();
        for (int i = 0; i < S*H*T*B; i++) ((float *) k->data)[i] = 0.09f * rng_next();
        for (int i = 0; i < S*H*T*B; i++) ((float *) v->data)[i] = 0.11f * rng_next();
        for (int i = 0; i < S*H*T*B; i++) ((float *) g->data)[i] = -0.05f - 0.4f * (0.5f * rng_next() + 0.5f);
        for (int i = 0; i <   H*T*B; i++) ((float *) b->data)[i] = 0.5f + 0.45f * (0.5f * rng_next() + 0.5f);
        for (int i = 0; i < S*S*H*B; i++) ((float *) s->data)[i] = 0.03f * rng_next();

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * o = NULL, * sn = NULL;
        build_delta_net_chunking(ctx, gf, q, k, v, g, b, s, &o, &sn);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        {
            float * buf = calloc(ggml_nelements(o), 4);
            for (int64_t i3 = 0; i3 < o->ne[3]; i3++)
                for (int64_t i2 = 0; i2 < o->ne[2]; i2++)
                    for (int64_t i1 = 0; i1 < o->ne[1]; i1++)
                        for (int64_t i0 = 0; i0 < o->ne[0]; i0++) {
                            buf[i0 + o->ne[0]*(i1 + o->ne[1]*(i2 + o->ne[2]*i3))] =
                                *(float *) ((char *) o->data + i0*o->nb[0] + i1*o->nb[1] + i2*o->nb[2] + i3*o->nb[3]);
                        }
            int32_t n32 = (int32_t) ggml_nelements(o);
            fwrite(&n32, 4, 1, f);
            fwrite(buf, 4, ggml_nelements(o), f);
            free(buf);
            dump_tensor(f, sn);
        }
        ggml_free(ctx);
    }

    fclose(f);
    printf("ok %s\n", out_path);
    return 0;
}
