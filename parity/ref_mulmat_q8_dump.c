/* Q8_0 mul_mat ground truth in the shapes the BERT/T5 **encoder** graphs use.
 *
 * The existing parity/ref_mulmat_dump.c only covers m=4 rows × n=3 columns
 * (where `llamafile_sgemm` picks the `gemm4xN<3>` tile). The encoders run
 * m=1024/4096 rows with n = n_tokens columns, i.e. the `gemm4xN<4>` +
 * `gemm4xN<2>` tiles — and Q8_0 is the one weight type whose activation type
 * (also Q8_0) makes `llamafile_sgemm` *accept* the call, so those tiles decide
 * whether the reference's GEMM is the vec_dot path or tinyBLAS.
 *
 * Section: u32 'VMQ8' | u32 n | u32 R | u32 C | xq bytes | y f32 | dst f32 | EOF u32 -1
 * Build: gcc -O2 -std=c11 parity/ref_mulmat_q8_dump.c -I$PIN/ggml/include \
 *          -o parity/ref_mulmat_q8_dump -L$REF -lggml -lggml-cpu -lggml-base -lm -Wl,-rpath,$REF
 * Run:   ./parity/ref_mulmat_q8_dump parity/mulmat_q8_bert_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-alloc.h"

extern size_t ggml_row_size(enum ggml_type, int64_t);

static uint32_t lcg = 0xb0a7c0deu;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float) (int32_t) lcg / (float) (1 << 28)) * 1.8f - 0.9f;
}

static void run_case(FILE * f, int n, int R, int C) {
    struct ggml_init_params ip = { .mem_size = 256u * 1024u * 1024u, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    float * w = malloc((size_t) n * R * 4);
    float * y = malloc((size_t) n * C * 4);
    for (int i = 0; i < n * R; i++) w[i] = next_val();
    for (int i = 0; i < n * C; i++) y[i] = next_val();

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n, R);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, C);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
    memcpy(b->data, y, (size_t) n * C * 4);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w, a->data, 0, R, n, NULL);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    struct ggml_context * ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, 8);

    uint32_t magic = 0x38514D56u, nn = n, rr = R, cc = C;
    fwrite(&magic, 4, 1, f); fwrite(&nn, 4, 1, f); fwrite(&rr, 4, 1, f);
    fwrite(&cc, 4, 1, f);
    size_t xs = ggml_row_size(GGML_TYPE_Q8_0, (int64_t) n * R);
    fwrite(a->data, 1, xs, f);
    fwrite(y, 4, (size_t) n * C, f);
    fwrite(d->data, 4, (size_t) R * C, f);

    free(w); free(y);
    ggml_free(ctxc);
    ggml_free(ctx);
}

static void run_case_f32(FILE * f, int n, int R, int C) {
    struct ggml_init_params ip = { .mem_size = 256u * 1024u * 1024u, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, R);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, C);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
    for (int64_t i = 0; i < ggml_nelements(a); i++) ((float *) a->data)[i] = next_val();
    for (int64_t i = 0; i < ggml_nelements(b); i++) ((float *) b->data)[i] = next_val();

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    struct ggml_context * ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, 8);

    uint32_t magic = 0x32334D56u, nn = n, rr = R, cc = C;
    fwrite(&magic, 4, 1, f); fwrite(&nn, 4, 1, f); fwrite(&rr, 4, 1, f);
    fwrite(&cc, 4, 1, f);
    fwrite(a->data, 4, (size_t) n * R, f);
    fwrite(b->data, 4, (size_t) n * C, f);
    fwrite(d->data, 4, (size_t) R * C, f);

    ggml_free(ctxc);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * out = argc > 1 ? argv[1] : "mulmat_q8_bert_ref.bin";
    FILE * f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out); return 1; }

    /* bge-m3 encoder shapes: k = 1024 (attn 1024x1024, ffn 4096x1024), n =
     * n_tokens. The two big-row cases are the real shapes; the small-row ones
     * cover the tile selectors (n % 4 == 0 -> gemm4xN<4>, else the <2|3> tails)
     * without carrying megabytes of weights. */
    run_case(f, 1024, 1024, 14);  /* attn_q, T=14 (real shape) */
    run_case(f, 1024, 4096, 14);  /* ffn_up, T=14 (real shape) */
    run_case(f, 1024, 1024, 1);   /* gemv */
    run_case(f, 1024, 64, 2);
    run_case(f, 1024, 64, 3);
    run_case(f, 1024, 64, 4);
    run_case(f, 1024, 64, 64);
    /* t5-xxl encoder shapes (k = 4096, real n_ff is 10240; Q5_K is a separate
     * case — Q8_0 here only pins the t5-attn shape hypothesis) */
    run_case(f, 4096, 64, 4);
    run_case(f, 4096, 64, 17);

    /* F32 attention GEMMs (build_attn_mha kq / kqv): kq is
     * mul_mat(k_perm [64, T, H], q_perm [64, T, H]) → m = T, n = T, k = 64;
     * kqv is mul_mat(v_cont [T, 64, H], kq [T, T, H]) → m = 64, n = T, k = T. */
    const int toks[] = { 1, 2, 3, 4, 8, 14, 16, 64 };
    for (unsigned i = 0; i < sizeof(toks) / sizeof(toks[0]); i++) {
        const int T = toks[i];
        run_case_f32(f, 64, T, T);   /* kq   */
        run_case_f32(f, T, 64, T);   /* kqv  */
    }

    uint32_t end = 0xFFFFFFFFu;
    fwrite(&end, 4, 1, f);
    fclose(f);
    return 0;
}