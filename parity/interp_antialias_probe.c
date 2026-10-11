/* interp_antialias_probe.c — .so ground truth dump for GGML_OP_UPSCALE
 * with GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS (the siglip
 * default of clip_graph::resize_position_embeddings, clip-graph.h:12), at
 * the exact shape of parity/d1omni_v_parity.sh's g9 fixture: a 9x9 learned
 * grid interpolated to the 16x16 patch grid of the 128px cb bitmap, n_embd
 * 16, deterministic contents (the synthetic-mmproj recipe).
 *
 * It runs the graph chain of clip.cpp:312-330 against the pinned reference
 * .so and writes the cont_2d result for the port's
 * clip::tests::interp_antialias_port_vs_ref to compare against:
 *
 *   gcc -O2 -std=c11 parity/interp_antialias_probe.c \
 *     -I $LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/ggml/include \
 *     -o /tmp/r4o3/interp_antialias_probe \
 *     -L $LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml -lggml-base -lggml-cpu -lm \
 *     -Wl,-rpath,$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin
 *   /tmp/r4o3/interp_antialias_probe parity/interp_antialias_ref.bin
 *
 * File format: [i64 ne0][i64 ne1][i64 ne2][i64 ne3][f32 data...] (LE).
 */
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_EMBD 16
#define IN_SIDE 9
#define OUT_SIDE 16

/* the same deterministic contents recipe as the synthetic mmproj tensors */
static void gen(float * dst, size_t n, int tensor_idx) {
    for (size_t i = 0; i < n; ++i) {
        uint64_t k = (uint64_t)tensor_idx * 7919 + (uint64_t)i * 104729;
        float v = (float)(k % 2001) / 10000.0f - 0.1f;
        dst[i] = v;
    }
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "parity/interp_antialias_ref.bin";

    struct ggml_init_params ip = { 16u << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * t_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, IN_SIDE * IN_SIDE);
    gen((float *) t_pos->data, (size_t) N_EMBD * IN_SIDE * IN_SIDE, 3);

    /* clip.cpp:312-330's chain */
    struct ggml_tensor * a = ggml_reshape_3d(ctx, t_pos, N_EMBD, IN_SIDE, IN_SIDE);
    struct ggml_tensor * b = ggml_permute(ctx, a, 2, 0, 1, 3);
    struct ggml_tensor * c = ggml_interpolate(ctx, b, OUT_SIDE, OUT_SIDE, N_EMBD, 1,
            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);
    struct ggml_tensor * d = ggml_permute(ctx, c, 1, 2, 0, 3);
    struct ggml_tensor * e = ggml_cont_2d(ctx, d, N_EMBD, OUT_SIDE * OUT_SIDE);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    int64_t hdr[4] = { e->ne[0], e->ne[1], e->ne[2], e->ne[3] };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(e->data, sizeof(float), (size_t) ggml_nelements(e), f);
    fclose(f);
    printf("interp_antialias_probe: wrote %s (ne = [%lld %lld %lld %lld])\n",
           out_path, (long long) e->ne[0], (long long) e->ne[1],
           (long long) e->ne[2], (long long) e->ne[3]);
    ggml_free(ctx);
    return 0;
}
