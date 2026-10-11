/* prec_set_src probe (O3, X domain @ aa94f2086) — pin the [TAG_GGML_PREC]
 * src-slot op_params encoding of the reference .so against the port's
 * ops::tests::prec_set_src_encoding (same shapes, same calls). Successor of
 * parity/prec_acc_probe.c (batch 42b, then pinned @ c35b66744); this one
 * covers ggml_prec_set_src (ggml.c:3317-3341):
 *
 *   gcc -O2 -I $LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/ggml/include \
 *     parity/prec_src_probe.c -o /tmp/r4o3_prec_src_probe \
 *     -L $LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml -lggml-base -lggml-cpu \
 *     -Wl,-rpath,$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
 *   && /tmp/r4o3_prec_src_probe | tee parity/prec_src_ref.txt
 *
 * Dumps the first 5 i32 op_params slots (as hex) of MUL_MAT (+hadamard hint
 * +acc) / MUL_MAT_ID after ggml_prec_set_src, plus the boolean returns for
 * the rejected cases: idx == 0 and idx == 2 on MUL_MAT (only idx == 1 is
 * legal, ggml.c:3328-3330), FLASH_ATTN_EXT (default branch) and ADD.
 */
#include "ggml.h"
#include <stdio.h>

static void dump(const char * tag, struct ggml_tensor * t) {
    const int32_t * p = (const int32_t *) t->op_params;
    printf("%s op=%d params:", tag, (int) t->op);
    for (int i = 0; i < 5; i++) {
        printf(" [%d]=%08x(%d)", i, (unsigned) p[i], p[i]);
    }
    printf("\n");
}

int main(void) {
    struct ggml_init_params ip = { 16u << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    /* MUL_MAT: hint slot 1, acc slot 0, then src1 prec on slot 2+1=3 */
    struct ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
    struct ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_hint(mm, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_prec_set_acc(mm, GGML_PREC_BF16);
    const bool r1 = ggml_prec_set_src(mm, GGML_PREC_Q8, 1);
    printf("mul_mat prec_set_src(Q8,1) -> %d\n", r1);
    dump("mul_mat", mm);

    /* MUL_MAT: F32 rewrite of the same slot (mistral4 ffn_moe_down hint,
     * llama-graph.cpp:2368) */
    const bool r2 = ggml_prec_set_src(mm, GGML_PREC_F32, 1);
    printf("mul_mat prec_set_src(F32,1) -> %d\n", r2);
    dump("mul_mat", mm);

    /* MUL_MAT_ID: same slot 3 encoding */
    struct ggml_tensor * we   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 4);
    struct ggml_tensor * x3   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 2, 2);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, we, x3, ids);
    const bool r3 = ggml_prec_set_src(mmid, GGML_PREC_Q4, 1);
    printf("mul_mat_id prec_set_src(Q4,1) -> %d\n", r3);
    dump("mul_mat_id", mmid);

    /* idx != 1 on MUL_MAT: rejected without a write (ggml.c:3328-3330) */
    const bool r4 = ggml_prec_set_src(mm, GGML_PREC_Q4, 0);
    const bool r5 = ggml_prec_set_src(mm, GGML_PREC_Q4, 2);
    printf("mul_mat prec_set_src(Q4,0) -> %d (expect 0)\n", r4);
    printf("mul_mat prec_set_src(Q4,2) -> %d (expect 0)\n", r5);
    dump("mul_mat", mm);

    /* FLASH_ATTN_EXT: default branch rejects (no src prec slot) */
    struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 3, 2, 1);
    struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 2, 1);
    struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 2, 1);
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, NULL, 0.125f, 0.0f, 0.0f);
    const bool r6 = ggml_prec_set_src(fa, GGML_PREC_Q8, 1);
    printf("flash_attn_ext prec_set_src(Q8,1) -> %d (expect 0)\n", r6);
    dump("flash_attn_ext", fa);

    /* non-encodable op: ADD rejects */
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
    struct ggml_tensor * s = ggml_add(ctx, x, y);
    const bool r7 = ggml_prec_set_src(s, GGML_PREC_Q8, 1);
    printf("add prec_set_src(Q8,1) -> %d (expect 0)\n", r7);
    dump("add", s);
    (void) s;
    return 0;
}
