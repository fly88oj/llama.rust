/* ggml_clamp non-contiguous-view ground-truth dumper — drives the NEW
 * reference build's CLAMP kernel (65840ed53 "ggml: fix CLAMP on
 * non-contiguous views (CPU, CUDA)", c35b66744) through the real
 * graph-compute path and writes bit-exact inputs/outputs.
 *
 * What is probed: before 65840ed53 the f32/f16 clamp kernels addressed row j
 * at data + j*nb[1] — correct only when nb[2] == ne[1]*nb[1] and
 * nb[3] == ne[2]*nb[2] (contiguous dims). The fix decomposes j into
 * (i1, i2, i3) over ne[1]/ne[2] and addresses
 * data + i1*nb1 + i2*nb2 + i3*nb3 (GGML_TENSOR_UNARY_OP_LOCALS,
 * ops.cpp:6049-6119 @c35b66744). Sections cover
 *   * 3D + 4D non-contiguous VIEW sources (plain clamp, dst = contiguous dup)
 *   * clamp_inplace ON a non-contiguous view (dst itself non-contiguous)
 *   * f32 + f16 (the f16 kernel round-trips through f32)
 *   * contiguous controls (the fix must not change those bytes)
 *
 * Section format (all little-endian):
 *   u32 magic 'CLM1' | u32 kind | u32 min_bits | u32 max_bits
 *   i64 pne[4] (parent ne) | i64 ne[4] (view ne) | u64 nb[4] (view strides)
 *   u64 view_offset | u32 is_f16 | u32 parent_nbytes | parent bytes (BEFORE)
 *   u32 out_nbytes | out bytes (AFTER: dst data for plain clamp, the whole
 *   parent for inplace)
 *   kinds: 0 f32 3D view   1 f32 4D view   2 f32 inplace 3D view
 *          3 f16 3D view   4 f16 4D view   5 f16 inplace 3D view
 *          6 f32 contiguous control      7 f16 contiguous control
 * EOF: magic 0xFFFFFFFF
 *
 * Build (against the NEW reference tree c35b66744):
 *   gcc -O2 -o /tmp/syncd/ref_clamp_dump parity/ref_clamp_dump.c \
 *     -I/home/jeffrey/llm/llama.cpp-next/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin \
 *     -lggml-cpu -lggml-base -lggml -lm
 *   /tmp/syncd/ref_clamp_dump parity/clamp_ref.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-cpu.h"

static uint32_t lcg = 0x9e3779b9u;
static float next_val(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(int32_t)lcg / (float)(1 << 28)) * 2.0f - 1.0f;
}

static const float CLAMP_MIN = -0.3f;
static const float CLAMP_MAX =  0.3f;

/* One probe section: a parent tensor (contiguous, type-sized by is_f16) is
 * allocated, filled with LCG values (converted through fp32->fp16 for f16),
 * a strided VIEW is taken over it, and clamp (plain or inplace) runs over
 * the view through the real graph path. Everything the port needs to
 * reproduce the tensors bit-exactly is written to the dump. */
static void run_case(FILE *f, uint32_t kind, int is_f16,
                     const int64_t * pne,      /* parent ne[4] */
                     const int64_t * vne,      /* view ne[4]   */
                     const size_t  * vnb,      /* view nb[0..3] (nb0 = el) */
                     size_t voff, int inplace) {
    const size_t el = is_f16 ? 2 : 4;
    const enum ggml_type ty = is_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    const size_t parent_elems = (size_t) pne[0] * pne[1] * pne[2] * pne[3];
    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead()
                            + parent_elems * el * 2 + 65536;
    struct ggml_init_params ip = { ctx_size, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor * parent = ggml_new_tensor_4d(ctx, ty, pne[0], pne[1], pne[2], pne[3]);
    /* deterministic contents: f32 LCG values, rounded through fp16 for the
     * f16 parents (the dump carries the exact parent bytes, so the port
     * replays them without re-deriving the conversion) */
    {
        float * tmp = malloc(parent_elems * sizeof(float));
        for (size_t i = 0; i < parent_elems; i++) tmp[i] = next_val();
        if (is_f16) {
            /* the exported bulk converter (ggml-cpu.h:145) — same conversion
             * the library itself uses */
            ggml_cpu_fp32_to_fp16(tmp, (ggml_fp16_t *) parent->data, (int64_t) parent_elems);
        } else {
            memcpy(parent->data, tmp, parent_elems * 4);
        }
        free(tmp);
    }

    struct ggml_tensor * view;
    if (vne[2] == 1 && vne[3] == 1) {
        view = ggml_view_2d(ctx, parent, vne[0], vne[1], vnb[1], voff);
    } else if (vne[3] == 1) {
        view = ggml_view_3d(ctx, parent, vne[0], vne[1], vne[2], vnb[1], vnb[2], voff);
    } else {
        view = ggml_view_4d(ctx, parent, vne[0], vne[1], vne[2], vne[3],
                            vnb[1], vnb[2], vnb[3], voff);
    }

    struct ggml_tensor * r = inplace
        ? ggml_clamp_inplace(ctx, view, CLAMP_MIN, CLAMP_MAX)
        : ggml_clamp(ctx, view, CLAMP_MIN, CLAMP_MAX);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed %d\n", (int) st); exit(1); }

    /* snapshot the parent BEFORE is not needed: for plain clamp the parent is
     * read-only, for inplace the interesting post-state is the whole parent
     * (the kernel must touch only the view's rows) */
    const uint8_t * out_data = inplace ? (const uint8_t *) parent->data
                                       : (const uint8_t *) r->data;
    const size_t out_nbytes = inplace ? ggml_nbytes(parent) : ggml_nbytes(r);

    uint32_t magic = 0x314D4C43u; /* "CLM1" little-endian */
    uint32_t mn, mx;
    memcpy(&mn, &CLAMP_MIN, 4);
    memcpy(&mx, &CLAMP_MAX, 4);
    uint32_t f16flag = (uint32_t) is_f16;
    int64_t pne4[4] = { pne[0], pne[1], pne[2], pne[3] };
    int64_t ne4[4] = { vne[0], vne[1], vne[2], vne[3] };
    uint64_t nb4[4] = { el, vnb[1], vnb[2], vnb[3] };
    uint64_t off64 = (uint64_t) voff;
    uint32_t pn = (uint32_t) ggml_nbytes(parent);
    uint32_t on = (uint32_t) out_nbytes;
    fwrite(&magic, 4, 1, f);
    fwrite(&kind, 4, 1, f);
    fwrite(&mn, 4, 1, f);
    fwrite(&mx, 4, 1, f);
    fwrite(pne4, 8, 4, f);
    fwrite(ne4, 8, 4, f);
    fwrite(nb4, 8, 4, f);
    fwrite(&off64, 8, 1, f);
    fwrite(&f16flag, 4, 1, f);
    fwrite(&pn, 4, 1, f);
    fwrite(parent->data, 1, pn, f); /* BEFORE-state parent bytes */
    fwrite(&on, 4, 1, f);
    fwrite(out_data, 1, on, f);

    ggml_free(ctx);
}

int main(int argc, char **argv) {
    const char * out = "clamp_ref.bin";
    if (argc > 1) out = argv[1];
    FILE * f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* parent ne[4] = [5, 6, 4, 3] — view dims/strides chosen so that
     * nb[2] != ne[1]*nb[1] and (4D) nb[3] != ne[2]*nb[2]: the old kernel's
     * j*nb1 addressing provably lands on different rows (e.g. 3D view row
     * j=3: old -> 3*40=120, new -> i2=1 -> 240) */

    /* kind 0: f32 3D non-contiguous view, plain clamp */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 1};
        size_t  vnb[] = {4, 40, 240, 480};
        run_case(f, 0, 0, pne, vne, vnb, 8, 0);
    }
    /* kind 1: f32 4D non-contiguous view, plain clamp */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 2};
        size_t  vnb[] = {4, 40, 240, 960};
        run_case(f, 1, 0, pne, vne, vnb, 16, 0);
    }
    /* kind 2: f32 clamp_inplace on a 3D non-contiguous view */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 1};
        size_t  vnb[] = {4, 40, 240, 480};
        run_case(f, 2, 0, pne, vne, vnb, 8, 1);
    }
    /* kind 3: f16 3D non-contiguous view, plain clamp */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 1};
        size_t  vnb[] = {2, 20, 120, 240};
        run_case(f, 3, 1, pne, vne, vnb, 4, 0);
    }
    /* kind 4: f16 4D non-contiguous view, plain clamp */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 2};
        size_t  vnb[] = {2, 20, 120, 480};
        run_case(f, 4, 1, pne, vne, vnb, 8, 0);
    }
    /* kind 5: f16 clamp_inplace on a 3D non-contiguous view */
    {
        int64_t pne[] = {5, 6, 4, 3};
        int64_t vne[] = {5, 3, 2, 1};
        size_t  vnb[] = {2, 20, 120, 240};
        run_case(f, 5, 1, pne, vne, vnb, 4, 1);
    }
    /* kind 6: f32 contiguous control — identical to the pre-fix bytes */
    {
        int64_t pne[] = {5, 4, 3, 2};
        int64_t vne[] = {5, 4, 3, 2};
        size_t  vnb[] = {4, 20, 80, 240};
        run_case(f, 6, 0, pne, vne, vnb, 0, 0);
    }
    /* kind 7: f16 contiguous control */
    {
        int64_t pne[] = {5, 4, 3, 2};
        int64_t vne[] = {5, 4, 3, 2};
        size_t  vnb[] = {2, 10, 40, 120};
        run_case(f, 7, 1, pne, vne, vnb, 0, 0);
    }

    uint32_t eof = 0xFFFFFFFFu;
    fwrite(&eof, 4, 1, f);
    fclose(f);
    return 0;
}
