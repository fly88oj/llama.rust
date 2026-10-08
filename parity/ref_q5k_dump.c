/* ref_q5k_dump.c — production-path mul_mat ground truth for the K-quants that
 * the granite-hybrid / gpt-oss Q4_K_M ports compute with row-wise vec_dot
 * kernels (crates/ggml/src/vec_dot.rs), on the REAL tensor bytes of the GGUFs
 * on this machine.
 *
 * Q5_K is 代理 T's suspected granite residual source (blk.N.ffn_{gate,up}_shexp
 * = [1536, 1024] Q5_K, 2 per layer x 40 layers). This tool answers the two
 * questions the Rust side cannot answer alone:
 *
 *   (1) does the *reference* production path for a Q5_K (or Q4_K/Q6_K) tensor
 *       use the row-wise vec_dot kernels at all — i.e. does the CPU_REPACK
 *       buffer type give the tensor a repack trait on this host?
 *       `ggml_repack_get_optimal_repack_type` is the gate
 *       (repack.cpp:4925): AVX2/Q4_K -> q4_K_8x8_q8_K, MXFP4 -> mxfp4_8x8_q8_0,
 *       but the Q5_K/Q6_K branches require ggml_cpu_has_neon() (ARM only), so on
 *       this x86 host Q5_K/Q6_K must come back NULL (= plain path). We detect it
 *       by allocating the tensor in ggml_backend_cpu_repack_buffer_type() and
 *       reading tensor->extra after init_tensor.
 *   (2) bit-exact outputs of the real production path (ggml graph mul_mat, the
 *       same route llama-cli takes: ggml_graph_compute -> ggml_compute_forward
 *       -> ggml_cpu_extra_compute_forward for repack tensors) for our Rust
 *       mul_mat to be diffed against with to_bits().
 *
 * Input descriptor file (little endian):
 *   u32 type_id | u32 n_per_row | u32 n_rows | u32 n_act_rows |
 *   f32 act[n_act_rows * n_per_row] | weight bytes (row_size(n) * n_rows)
 * (build one with parity/mk_q5k_desc.py, which reads the real tensor bytes out
 *  of the GGUF at data_offset + tensor offset; the exact commands for the
 *  committed artifacts are in that script's header)
 *
 * Output artifact, sequence of sections (u32 len + len bytes):
 *   [0] hdr: u32 ty | u32 n | u32 nrows | u32 nact | u32 flags | u32 pad
 *           flags bit0 = repack buffer assigned a trait (extra != NULL)
 *                bit1 = repack-graph output section present
 *   [1] weight bytes (verbatim copy of the descriptor payload)
 *   [2] activation floats (nact * n)
 *   [3] plain-graph mul_mat output (f32, nrows * nact)
 *   [4] repacked tensor bytes (nbytes) or empty when bit0 == 0
 *   [5] repack-graph mul_mat output (f32, nrows*nact) or empty
 *   [6] q8_K activation bytes produced by the reference's production
 *       quantizer (type_traits_cpu[Q8_K].from_float) for the nact rows
 *   [7] llamafile_sgemm output (f32 nrows*nact) or empty; flags bit2 = it
 *       returned true, i.e. the reference routed this GEMM to tinyBLAS
 *
 * Build (from the repo root). NOTE: must be compiled as C (or the repack entry
 * point has to be declared extern "C") because the repack.h gate is a C++-only
 * header; the tool binds the mangled symbol directly (see the __asm__ below).
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   cc -O2 -std=gnu11 parity/ref_q5k_dump.c -I$PINNED/ggml/include -I$PINNED/ggml/src \
 *      -I$PINNED/ggml/src/ggml-cpu -o parity/ref_q5k_dump \
 *      -L$REF -lggml -lggml-cpu -lggml-base -lm -Wl,-rpath,$REF
 * Run: ./parity/ref_q5k_dump <descriptor.bin> <out.bin>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h"   /* struct ggml_compute_params */
#include "ggml-cpu/llamafile/sgemm.h"

/* libggml-cpu.so exports this with C++ linkage (repack.h itself is C++-only:
 * it pulls in ggml-common.h which includes <cstdint>), so bind the mangled
 * symbol directly and keep this tool a plain C file like the other dumps. */
extern ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void)
    __asm__("_Z35ggml_backend_cpu_repack_buffer_typev");
extern size_t ggml_row_size(enum ggml_type, int64_t);

#define QK_K_ 256
#define SIZE_BLOCK_Q8_K 292   /* sizeof(block_q8_K): f32 d + 256 i8 + 16 i16 */

static void write_section(FILE * f, const void * data, uint64_t len) {
    uint32_t l = (uint32_t) len;
    fwrite(&l, sizeof(l), 1, f);
    if (len) fwrite(data, 1, len, f);
}

/* real graph mul_mat: weights [n x nrows] (raw bytes), activations f32 [n x nact]
 * -> dst f32 [nrows x nact]. Computed through the CPU graph path llama-cli uses,
 * so ggml_cpu_extra_compute_forward() sees the repack trait when present. */
static int graph_mul_mat(enum ggml_type ty, int n, int nrows, int nact,
                         const void * wbytes, const float * act, float * out) {
    struct ggml_init_params ip = {
        .mem_size = 512ull * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, ty, n, nrows);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, nact);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
    memcpy(a->data, wbytes, ggml_nbytes(a));
    memcpy(b->data, act, (size_t) n * nact * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    struct ggml_context * ctxc = ggml_init(ip);
    ggml_graph_compute_with_ctx(ctxc, gf, 8);
    memcpy(out, d->data, (size_t) nrows * nact * sizeof(float));
    ggml_free(ctxc);
    ggml_free(ctx);
    return 0;
}

/* llamafile_sgemm probe: ggml-cpu.c:1383 makes a *second* attempt with the
 * converted Q8_0/Q8_K activation rows (Btype = vec_dot_type) whenever
 * src1->type != vec_dot_type; for Q4_0/Q5_0/Q8_0/IQ4_NL weights that call
 * returns true on AVX2/AVX512 (the `if (Btype != GGML_TYPE_<A>) return false`
 * guard is satisfied by Q8_0!) and the whole GEMM is done by
 * tinyBLAS_Q0_AVX instead of vec_dot. K-quants have no sgemm case at all, so
 * the probe must report false for Q4_K/Q5_K/Q6_K — that is what makes our
 * row-wise mul_mat the reference's production path for granite's shexp. */
static int sgemm_probe(enum ggml_type ty, enum ggml_type bty, int n, int nrows, int nact,
                       const void * xb, const void * yq, float * out) {
    struct ggml_compute_params params;
    memset(&params, 0, sizeof(params));
    params.ith = 0;
    params.nth = 1;
    return llamafile_sgemm(&params, nrows, nact, n / ggml_blck_size(ty), xb,
                           ggml_row_size(ty, n) / ggml_type_size(ty), yq,
                           ggml_row_size(bty, n) / ggml_type_size(bty), out,
                           (int64_t) nrows, ty, bty, GGML_TYPE_F32) ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <descriptor.bin> <out.bin>\n", argv[0]);
        return 1;
    }
    FILE * in = fopen(argv[1], "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    uint32_t hdr[4];
    if (fread(hdr, 4, 4, in) != 4) { fprintf(stderr, "short descriptor\n"); return 1; }
    const enum ggml_type ty = (enum ggml_type) hdr[0];
    const int n     = (int) hdr[1];
    const int nrows = (int) hdr[2];
    const int nact  = (int) hdr[3];

    float * act = malloc((size_t) n * nact * sizeof(float));
    if (fread(act, sizeof(float), (size_t) n * nact, in) != (size_t) n * nact) {
        fprintf(stderr, "short activation payload\n"); return 1;
    }
    const size_t wsize = ggml_row_size(ty, n) * nrows;
    uint8_t * wbytes = malloc(wsize);
    if (fread(wbytes, 1, wsize, in) != wsize) { fprintf(stderr, "short weight payload\n"); return 1; }
    fclose(in);

    printf("descriptor: type %d (%s) n=%d nrows=%d nact=%d wsize=%zu\n",
           (int) ty, ggml_type_name(ty), n, nrows, nact, wsize);

    /* CPU backend registration: gives us the repack buffer type + cpu features */
    ggml_backend_register(ggml_backend_cpu_reg());

    float * out_plain = malloc((size_t) nrows * nact * sizeof(float));
    graph_mul_mat(ty, n, nrows, nact, wbytes, act, out_plain);

    /* ---- CPU_REPACK probe: does this type get a repack trait on this host? ---- */
    uint32_t flags = 0;
    float * out_repack = NULL;
    uint8_t * repacked = NULL;
    size_t repacked_len = 0;
    ggml_backend_buffer_type_t rbuft = ggml_backend_cpu_repack_buffer_type();
    if (rbuft) {
        struct ggml_init_params ip = { .mem_size = 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true };
        struct ggml_context * wctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(wctx, ty, n, nrows);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(rbuft, ggml_backend_buft_get_alloc_size(rbuft, w));
        if (buf) {
            ggml_backend_buffer_init_tensor(buf, w);
            w->buffer = buf;
            w->data   = ggml_backend_buffer_get_base(buf);
            if (w->extra != NULL) {
                flags |= 1u;
                ggml_backend_tensor_set(w, wbytes, 0, wsize);
                repacked_len = wsize;
                repacked = malloc(repacked_len);
                memcpy(repacked, w->data, repacked_len);

                /* second graph, this weight tensor lives in the repack buffer */
                struct ggml_init_params ip2 = { .mem_size = 512ull * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
                struct ggml_context * ctx = ggml_init(ip2);
                struct ggml_tensor * a = ggml_new_tensor_2d(ctx, ty, n, nrows);
                struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, nact);
                struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
                memcpy(a->data, wbytes, ggml_nbytes(a));
                memcpy(b->data, act, (size_t) n * nact * sizeof(float));
                /* swap a's storage for the repack buffer so the trait is visible */
                a->buffer = buf;
                a->data   = w->data;
                a->extra  = w->extra;
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, d);
                struct ggml_context * ctxc = ggml_init(ip2);
                ggml_graph_compute_with_ctx(ctxc, gf, 8);
                out_repack = malloc((size_t) nrows * nact * sizeof(float));
                memcpy(out_repack, d->data, (size_t) nrows * nact * sizeof(float));
                flags |= 2u;
                ggml_free(ctxc);
                ggml_free(ctx);
                printf("repack trait: YES (extra=%p) — the reference production path for this tensor is the 8x8 repack kernel\n", w->extra);
            } else {
                printf("repack trait: NULL — the reference runs the plain vec_dot path for %s on this host\n",
                       ggml_type_name(ty));
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(wctx);
    }

    /* q8_K activations as the reference's production quantizer emits them */
    const struct ggml_type_traits_cpu * tt = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    uint8_t * q8k = NULL;
    size_t q8k_len = 0;
    if (tt && tt->from_float) {
        q8k_len = (size_t) nact * (n / QK_K_)*SIZE_BLOCK_Q8_K;   /* sizeof(block_q8_K) */
        q8k = calloc(1, q8k_len);
        for (int r = 0; r < nact; r++) {
            tt->from_float(act + (size_t) r * n, q8k + (size_t) r * (n / QK_K_)*SIZE_BLOCK_Q8_K, n);
        }
    }

    /* ---- tinyBLAS probe on the converted-activation rows ----
     * uses the same quantizer the vec_dot path uses (vec_dot_type of the weight) */
    float * out_sgemm = NULL;
    uint32_t sgemm_ret = 0;
    {
        const struct ggml_type_traits_cpu * tb = ggml_get_type_traits_cpu(ty);
        if (tb && tb->vec_dot_type != GGML_TYPE_COUNT) {
            const enum ggml_type bty = tb->vec_dot_type;
            const struct ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(bty);
            uint8_t * yq = NULL;
            size_t yqlen = 0;
            if (bty == GGML_TYPE_Q8_0) {
                yqlen = (size_t) nact * (n / 32) * 34;
                yq = calloc(1, yqlen);
                for (int r = 0; r < nact; r++) tq->from_float(act + (size_t) r * n, yq + (size_t) r * (n / 32) * 34, n);
            } else if (bty == GGML_TYPE_Q8_K) {
                yq = q8k;
                yqlen = q8k_len;
            }
            if (yq) {
                out_sgemm = calloc((size_t) nrows * nact, sizeof(float));
                sgemm_ret = (uint32_t) sgemm_probe(ty, bty, n, nrows, nact, wbytes, yq, out_sgemm);
                printf("llamafile_sgemm(Atype=%s, Btype=%s, m=%d, n=%d, k=%d): %s\n",
                       ggml_type_name(ty), ggml_type_name(bty), nrows, nact,
                       n / (int) ggml_blck_size(ty), sgemm_ret ? "TRUE -> tinyBLAS owns this GEMM" : "false -> vec_dot");
                flags |= sgemm_ret ? 4u : 0u;
            }
        }
    }

    FILE * f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    const uint32_t ohdr[6] = { (uint32_t) ty, (uint32_t) n, (uint32_t) nrows, (uint32_t) nact, flags, 0 };
    write_section(f, ohdr, sizeof(ohdr));
    write_section(f, wbytes, wsize);
    write_section(f, act, (size_t) n * nact * sizeof(float));
    write_section(f, out_plain, (size_t) nrows * nact * sizeof(float));
    write_section(f, repacked, repacked_len);
    write_section(f, out_repack, out_repack ? (size_t) nrows * nact * sizeof(float) : 0);
    write_section(f, q8k, q8k_len);
    write_section(f, out_sgemm, out_sgemm ? (size_t) nrows * nact * sizeof(float) : 0);
    fclose(f);
    printf("wrote %s (flags=%u)\n", argv[2], flags);
    return 0;
}