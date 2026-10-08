/* Timing probe: whole-graph ggml_graph_compute on the qwen2.5-0.5b pp64
 * mul_mat shapes, on the reference's own threadpool — the per-op wall times
 * the in-model profile of the port should be compared against. Weights are
 * plain in-context tensors (NOT the CPU_REPACK buffer), so Q4_K here times
 * the row-wise vec_dot fallback, not the repack gemm — the shapes table marks
 * which route the in-model tensor takes.
 *
 * Build:
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   gcc -O2 -x c++ parity/ref_mulmat_bench.c -I$PINNED/ggml/include \
 *       -o parity/ref_mulmat_bench -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_mulmat_bench [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint32_t st = 0x1234567u;
static uint32_t lcg(void) {
    st = st * 1664525u + 1013904223u;
    return st;
}

static float lcg_f(void) {
    return ((int32_t) lcg()) / (float) (1u << 28) * 1.8f - 0.9f;
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    struct shape { enum ggml_type ty; int64_t k, m, n; const char * name; } shapes[] = {
        { GGML_TYPE_Q5_0,   896, 4864, 64, "ffn_gate/up Q5_0 4864x896 x64 (llamafile)" },
        { GGML_TYPE_Q5_0,   896,  896, 64, "attn_q/o    Q5_0  896x896 x64 (llamafile)" },
        { GGML_TYPE_Q5_0,   896,  128, 64, "attn_k/v    Q5_0  128x896 x64 (llamafile)" },
        { GGML_TYPE_Q6_K,  4864,  896, 64, "ffn_down    Q6_K  896x4864 x64 (row vec_dot)" },
        { GGML_TYPE_Q4_K,  4864,  896, 64, "ffn_down    Q4_K  896x4864 x64 (PLAIN row vec_dot; in-model = repack)" },
        { GGML_TYPE_Q8_0,   896, 151936, 1, "lm_head     Q8_0 151936x896 x1 (gemv)" },
        { GGML_TYPE_Q8_0,   896, 151936, 64, "lm_head     Q8_0 151936x896 x64 (llamafile)" },
    };

    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        struct shape * sh = &shapes[s];
        const int64_t k = sh->k, m = sh->m, n = sh->n;

        // context: weights + activations + result + graph metadata (+ slack)
        size_t ctx_size = 16u << 20;
        ctx_size += ggml_tensor_overhead() * 8 + 2 * ggml_graph_overhead();
        ctx_size += (size_t) ggml_row_size(sh->ty, k * m);
        ctx_size += (size_t) ggml_row_size(GGML_TYPE_F32, k * n);
        ctx_size += (size_t) ggml_row_size(GGML_TYPE_F32, m * n);
        struct ggml_init_params ip = { /*.mem_size=*/ ctx_size, /*.mem_buffer=*/ NULL,
                                       /*.no_alloc=*/ false };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, sh->ty, k, m);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);

        // deterministic weights (quantized with the reference quantizer) + f32 activations
        {
            float * tmp = (float *) malloc(sizeof(float) * k * m);
            for (int64_t i = 0; i < k * m; i++) tmp[i] = lcg_f();
            ggml_quantize_chunk(sh->ty, tmp, w->data, /*start=*/ 0, /*nrows=*/ m, /*n_per_row=*/ k, NULL);
            free(tmp);
            float * xd = (float *) x->data;
            for (int64_t i = 0; i < k * n; i++) xd[i] = lcg_f();
        }

        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);

        struct ggml_cplan plan = ggml_graph_plan(g, nth, NULL);
        if (plan.work_size > 0) {
            plan.work_data = (uint8_t *) malloc(plan.work_size);
        }
        ggml_graph_compute(g, &plan); // warm-up (page faults, wdata settle)

        const int iters = 100;
        double best = 1e30;
        for (int it = 0; it < iters; it++) {
            double t0 = now_ms();
            ggml_graph_compute(g, &plan);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        double gfl = 2.0 * m * n * k / 1e9;
        printf("%-58s t=%2d %8.3f ms/call %8.1f GF/s\n", sh->name, nth, best, gfl / (best / 1e3));

        ggml_free(ctx);
    }
    return 0;
}
