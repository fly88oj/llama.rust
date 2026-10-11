/* acc negative-offset guard probe — evidence for 8ae386707 ("ggml: fix OOB
 * write in ggml_acc with negative offset", #30135).
 *
 * ggml_acc_impl narrowed a size_t offset to int32_t without a check, so an
 * offset >= 2^31 truncated negative and the forward sign-extended it to a
 * huge size_t — the bounds assert wrapped and the kernel wrote below the dst
 * buffer. The fix adds, before the narrowing:
 *
 *     GGML_ASSERT(offset < (size_t)(1 << 30));   // ggml.c:2202 @aa94f2086
 *
 * The PORT HAS NO GGML_OP_ACC (the op is unused by every ported arch; the
 * port's ggml crate never implemented it), so there is no port twin to fix
 * — this probe pins the *reference* behaviour instead: the NEW reference
 * build (libggml-base.so.0.26.0 = aa94f2086) must refuse offset >= 1<<30 at
 * graph-build time with the guard assert, and must still accept a small
 * offset. (Running the OLD .so with a huge offset is deliberately NOT done
 * here: the unguarded path only corrupts memory at compute time, and this
 * probe never computes.)
 *
 * Build: gcc -O2 -o /tmp/r4a/ref_acc_negoff parity/ref_acc_negoff.c \
 *          -I<ref>/ggml/include -L<ref>/build-rust-ref/bin -lggml-base \
 *          -Wl,-rpath,<ref>/build-rust-ref/bin
 * Run:   /tmp/r4a/ref_acc_negoff ok      -> "accepted"  (small offset fine)
 *        /tmp/r4a/ref_acc_negoff guard   -> aborts with the ggml.c:2202 assert
 */
#include "ggml.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s ok|guard\n", argv[0]); return 2; }
    const int guard = strcmp(argv[1], "guard") == 0;

    struct ggml_init_params ip = { .mem_size = 1 << 20, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);

    // ggml_acc(ctx, a, b, nb1, nb2, nb3, offset) — the offset the guard checks
    const size_t offset = guard ? ((size_t) 1 << 30) : 0;
    struct ggml_tensor * r = ggml_acc(ctx, a, b, 4 * 8, 0, 0, offset);

    printf("accepted: offset=%zu result=%s\n", offset, ggml_type_name(r->type));
    ggml_free(ctx);
    return 0;
}
