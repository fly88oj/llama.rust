/* Reference GGUF writer dump — writes the same logical content as
 * crates/ggml/src/gguf_write.rs's `gguf_write_bit_exact_vs_reference` test
 * through the reference gguf.cpp writer (bd4f514db1) so the Rust writer can be
 * asserted byte-identical.
 *
 * Build (from repo root):
 *   gcc parity/ref_gguf_write.c -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *       -o parity/ref_gguf_write \
 *       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin -lggml-base -lm \
 *       -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   ./parity/ref_gguf_write parity/gguf_write_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "gguf.h"

static uint32_t state = 0x12345678u;
static uint8_t next_byte(void) {
    state = state * 1664525u + 1013904223u;
    return (uint8_t)(state >> 24);
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "gguf_write_ref.bin";

    struct ggml_init_params ip = { .mem_size = 4 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *gctx = ggml_init(ip);

    struct gguf_context *ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "test");
    gguf_set_val_u32(ctx, "test.block_count", 2);
    gguf_set_val_f32(ctx, "test.f", 1.5f);

    int32_t arr[3] = { -1, 7, 42 };
    gguf_set_arr_data(ctx, "test.arr", GGUF_TYPE_INT32, arr, 3);

    const char *strs[2] = { "alpha", "beta" };
    gguf_set_arr_str(ctx, "test.strs", strs, 2);

    struct ggml_tensor *a = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 16);
    struct ggml_tensor *b = ggml_new_tensor_1d(gctx, GGML_TYPE_Q4_0, 32);
    ggml_set_name(a, "a.weight");
    ggml_set_name(b, "b.weight");
    for (size_t i = 0; i < 64; ++i) ((uint8_t *) a->data)[i] = next_byte();
    for (size_t i = 0; i < 18; ++i) ((uint8_t *) b->data)[i] = next_byte();

    gguf_add_tensor(ctx, a);
    gguf_add_tensor(ctx, b);

    if (!gguf_write_to_file(ctx, out, false)) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    printf("wrote %s\n", out);

    gguf_free(ctx);
    ggml_free(gctx);
    return 0;
}