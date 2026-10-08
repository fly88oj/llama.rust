/* Replay the reference's MUL_MAT_ID (hash-layer MoE, F32 experts) in
 * isolation: gate_exps [128,24,4] x cur3 [128,1,6] with the hash ids
 * (token-parity table) — arch batch 7 node-dump bisection.
 *
 *   gcc -O2 -I pinned/ggml/include parity/ref_mmid_probe.c -o /tmp/mmid \
 *     -L build-rust-ref/bin -lggml -lggml-base -lggml-cpu \
 *     -Wl,-rpath,build-rust-ref/bin
 *   /tmp/mmid /tmp/gate_exps.bin /tmp/ref_cur3.bin /tmp/ref_gate_mm.bin
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <exps.bin> <cur3.bin> <ref.bin>\n", argv[0]); return 1; }

    static float exps[128 * 24 * 4], cur[128 * 6], yref[24 * 2 * 6];
    FILE * f = fopen(argv[1], "rb"); fread(exps, 4, 128 * 24 * 4, f); fclose(f);
    f = fopen(argv[2], "rb"); fread(cur, 4, 128 * 6, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(yref, 4, 24 * 2 * 6, f); fclose(f);

    struct ggml_init_params ip = { 64u << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 24, 4);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, 6);
    memcpy(a->data, exps, sizeof(exps));
    memcpy(b->data, cur, sizeof(cur));

    /* ids [n_ids=2, n_tokens=6]: the hash table values = token_id % 2 */
    const int toks_arr[6] = {1, 450, 7483, 310, 3444, 338};
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 6);
    /* the file's actual table columns are the constant (0, 1) */
    for (int t = 0; t < 6; t++) {
        ((int32_t *) ids->data)[0 + t * 2] = 0;
        ((int32_t *) ids->data)[1 + t * 2] = 1;
    }

    /* also replay the graph-side get_rows: table [2, 32000] from the file's
     * parity pattern, gathered by the prompt tokens */
    struct ggml_tensor * tbl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 32000);
    {
        int32_t * tv = (int32_t *) tbl->data;
        for (int64_t i = 0; i < 2 * 32000; i++) tv[i] = (int32_t) (i % 2);
    }
    struct ggml_tensor * toks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
    for (int t = 0; t < 6; t++) ((int32_t *) toks->data)[t] = toks_arr[t];
    struct ggml_tensor * ids2 = ggml_get_rows(ctx, tbl, toks);

    struct ggml_tensor * y = ggml_mul_mat_id(ctx, a, b, ids);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_build_forward_expand(gf, ids2);
    ggml_graph_compute_with_ctx(ctx, gf, 8);
    {
        const int32_t * v = (const int32_t *) ids2->data;
        printf("get_rows ids:");
        for (int t = 0; t < 6; t++) printf(" [%d,%d]", v[0 + t * 2], v[1 + t * 2]);
        printf("\n");
    }

    const float * py = (const float *) y->data;
    int nd = 0;
    for (int i = 0; i < 24 * 2 * 6; i++) {
        if (py[i] != yref[i]) {
            if (nd < 8) printf("%d ref %.9g (%08x) isolated %.9g (%08x)\n", i,
                               yref[i], *(unsigned *)&yref[i], py[i], *(unsigned *)&py[i]);
            nd++;
        }
    }
    printf("isolated-vs-graph: %d/%d differ\n", nd, 24 * 2 * 6);
    return 0;
}
