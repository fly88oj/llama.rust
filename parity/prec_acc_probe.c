/* prec_set_acc probe (batch 42b, X domain) — pin the [TAG_GGML_PREC]
 * op_params encoding of the reference .so against the port's
 * ops::tests::prec_set_acc_encoding (same shapes, same calls). The reference
 * is the NEW pinned tree @ c35b66744:
 *
 *   gcc -O2 -I /home/jeffrey/llm/llama.cpp-next/ggml/include \
 *     parity/prec_acc_probe.c -o /tmp/closx_prec_probe \
 *     -L /home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml -lggml-base -lggml-cpu \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *   && /tmp/closx_prec_probe | tee parity/prec_acc_ref.txt
 *
 * Dumps the first 5 i32 op_params slots (as hex) of a MUL_MAT (+hadamard
 * hint), a MUL_MAT_ID and a FLASH_ATTN_EXT after ggml_prec_set_acc, plus the
 * boolean return for a non-encodable op (ADD) — ggml.c:3291-3315.
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

    /* MUL_MAT: hint on slot 1 first, then the acc hint on slot 0 */
    struct ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    struct ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
    struct ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_hint(mm, GGML_HINT_SRC0_IS_HADAMARD);
    const bool r1 = ggml_prec_set_acc(mm, GGML_PREC_BF16);
    printf("mul_mat prec_set_acc(BF16) -> %d\n", r1);
    dump("mul_mat", mm);

    /* MUL_MAT_ID: we [16,8,4] x x3 [16,2,2], ids [2,2] */
    struct ggml_tensor * we   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 4);
    struct ggml_tensor * x3   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 2, 2);
    struct ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, we, x3, ids);
    const bool r2 = ggml_prec_set_acc(mmid, GGML_PREC_F32);
    printf("mul_mat_id prec_set_acc(F32) -> %d\n", r2);
    dump("mul_mat_id", mmid);

    /* FLASH_ATTN_EXT: q [16,3,2,1], k/v [16,5,2,1], scale 0.125 */
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 3, 2, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 2, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 5, 2, 1);
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, NULL, 0.125f, 0.0f, 0.0f);
    const bool r3 = ggml_prec_set_acc(fa, GGML_PREC_BF16);
    printf("flash_attn_ext prec_set_acc(BF16) -> %d\n", r3);
    dump("flash_attn_ext", fa);

    /* non-encodable op: ADD rejects */
    struct ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
    struct ggml_tensor * s = ggml_add(ctx, x, y);
    const bool r4 = ggml_prec_set_acc(s, GGML_PREC_BF16);
    printf("add prec_set_acc(BF16) -> %d (expect 0)\n", r4);
    dump("add", s);
    (void) s;
    return 0;
}
