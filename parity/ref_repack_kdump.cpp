/* CPU_REPACK reference dump for the K-quant 8x8 path — ground-truth repacked
 * bytes + gemv/gemm outputs from the pinned llama.cpp (bd4f514db1) CPU backend
 * for bit-exact Rust parity tests (crates/ggml/src/repack.rs, "Q4_K 8x8").
 *
 * Sibling of parity/ref_repack_dump.cpp (which covers MXFP4); same build line,
 * same "only public API" rule: the tensor is allocated in
 * ggml_backend_cpu_repack_buffer_type() so the buffer's init_tensor assigns the
 * real trait (ggml_repack_get_optimal_repack_type, repack.cpp:5006-5011:
 * `ggml_cpu_has_avx2() && ne[1] % 8 == 0` -> q4_K_8x8_q8_K) and set_tensor runs
 * the real repack (repack_q4_K_to_q4_K_8_bl, repack.cpp:3573). The kernels are
 * the exported entry points the reference's own dispatch calls
 * (ggml_gemv_q4_K_8x8_q8_K / ggml_gemm_q4_K_8x8_q8_K, repack.cpp:4379/4484 ->
 * arch/x86/repack.cpp:1464/2042; the AVX2 gemv body and the AVX512BW/DQ gemm
 * body are what this build links).
 *
 * Sections dumped (single file, each section = u32 len + bytes):
 *   [0] u32 type_id | u32 n_per_row | u32 nrows | u32 nr_gemv | u32 nr_gemm |
 *       u32 nc  (nc == nrows)
 *   [1] src block_q4_K bytes                 (nrows * nb * 144)
 *   [2] repacked block_q4_Kx8 bytes          (same length)
 *   [3] plain block_q8_K rows for the gemv   (nr_gemv * nb * 292)
 *   [4] block_q8_Kx4 tiles for the gemm      (nr_gemm/4 * nb * 1168)
 *   [5] gemv AVX output (f32, nc + 64 slots, sentinel elsewhere)
 *   [6] gemm AVX output (f32, nr_gemm*bs + 64 slots)
 *   [7] gemv *_generic output
 *   [8] gemm *_generic output
 *   [9] activation floats for the gemv rows
 *  [10] activation floats for the gemm rows
 *  [11] u32 trait_present | u32 pad  (1 = the repack buffer gave it a trait)
 *
 * Build (from the repo root):
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   g++ -O2 -std=c++17 parity/ref_repack_kdump.cpp \
 *       -I$PINNED/ggml/include -I$PINNED/ggml/src -I$PINNED/ggml/src/ggml-cpu \
 *       -o parity/ref_repack_kdump -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_repack_kdump parity/q4k_repack_ref.bin
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "repack.h"

extern "C" ggml_backend_reg_t ggml_backend_cpu_reg(void);

// the trait gate the buffer applies is a C++ inline; re-declare the type check
#define N_PER_ROW 256
#define NROWS     16
#define NR_GEMV   1
#define NR_GEMM   8
#define NC        NROWS
#define BS_GEMM   NC
#define SENTINEL  1234.5f

static uint32_t lcg_state = 0x13579bdfu;
static uint32_t next_u32(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}
static float next_f32(void) {
    return ((float)(int32_t)next_u32() / (float)(1 << 30)) * 1.5f - 0.75f;
}

static void write_section(FILE * f, const void * data, uint64_t len) {
    uint32_t l = (uint32_t) len;
    fwrite(&l, sizeof(l), 1, f);
    if (len) fwrite(data, 1, len, f);
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "parity/q4k_repack_ref.bin";

    ggml_backend_register(ggml_backend_cpu_reg());
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        fprintf(stderr, "no CPU backend device\n");
        return 1;
    }

    const ggml_type type = GGML_TYPE_Q4_K;
    const int64_t    nb   = N_PER_ROW / QK_K;
    const size_t     nb_row = (size_t) nb * sizeof(block_q4_K);

    // ---- 1. source Q4_K bytes (random nibbles + 6-bit scales/mins) ----
    std::vector<uint8_t> src((size_t) NROWS * nb_row);
    for (size_t i = 0; i < src.size(); i += sizeof(block_q4_K)) {
        block_q4_K * b = (block_q4_K *) &src[i];
        // a spread of d/dmin, including the negative and zero edge values
        const uint16_t dh = (uint16_t) (0x2000 + next_u32() % 0x2000);
        const uint16_t dm = (uint16_t) (0x2000 + next_u32() % 0x2000);
        // d/dmin are the first two ggml_half fields; write them through the byte
        // view so the dumper does not depend on the aggregate's spelling
        uint8_t * fb = (uint8_t *) b;
        memcpy(fb + 0, &dh, 2);
        memcpy(fb + 2, &dm, 2);
        for (int j = 0; j < K_SCALE_SIZE; j++) b->scales[j] = (uint8_t) next_u32();
        for (int j = 0; j < QK_K / 2; j++) b->qs[j] = (uint8_t) next_u32();
    }

    // ---- 2. repack through the real CPU_REPACK buffer type ----
    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    printf("buft name: %s\n", ggml_backend_buft_name(repack_buft));

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor *  w   = ggml_new_tensor_2d(ctx, type, N_PER_ROW, NROWS);
    const size_t   wbytes = ggml_nbytes(w);
    GGML_ASSERT(wbytes == src.size());

    ggml_backend_buffer_t wbuf = ggml_backend_buft_alloc_buffer(repack_buft, wbytes);
    GGML_ASSERT(wbuf != nullptr);
    GGML_ASSERT(ggml_backend_buffer_init_tensor(wbuf, w) == GGML_STATUS_SUCCESS);
    const uint32_t trait_present = w->extra != nullptr ? 1u : 0u;
    w->buffer = wbuf;
    w->data   = ggml_backend_buffer_get_base(wbuf);
    ggml_backend_tensor_set(w, src.data(), 0, wbytes); // -> repack_q4_K_to_q4_K_8_bl
    const uint8_t * repacked = (const uint8_t *) w->data;
    printf("repacked tensor: %lld x %lld, %zu bytes, trait %s\n",
           (long long) w->ne[0], (long long) w->ne[1], wbytes,
           trait_present ? "PRESENT" : "NULL");

    // ---- 3. activations ----
    const ggml_type_traits_cpu * tt_q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    GGML_ASSERT(tt_q8->from_float != nullptr);

    std::vector<float> act_g(NR_GEMV * N_PER_ROW);
    for (float & v : act_g) v = next_f32();
    std::vector<uint8_t> q8((size_t) NR_GEMV * nb * sizeof(block_q8_K));
    for (int r = 0; r < NR_GEMV; r++) {
        tt_q8->from_float(&act_g[r * N_PER_ROW], &q8[(size_t) r * nb * sizeof(block_q8_K)], N_PER_ROW);
    }

    GGML_ASSERT(sizeof(block_q8_Kx4) == 4 * sizeof(float) + QK_K * 4 + (QK_K / 4) * sizeof(int16_t));
    std::vector<float> act_m(NR_GEMM * N_PER_ROW);
    for (float & v : act_m) v = next_f32();
    std::vector<uint8_t> q8x4((size_t) (NR_GEMM / 4) * nb * sizeof(block_q8_Kx4));
    for (int r = 0; r < NR_GEMM; r += 4) {
        // INTER_SIZE == NB_COLS == 8 for q4_K_8x8_q8_K -> ggml_quantize_mat_t<8, Q8_K>
        // -> ggml_quantize_mat_q8_K_4x8 (repack.cpp:336)
        ggml_quantize_mat_q8_K_4x8(&act_m[r * N_PER_ROW], &q8x4[(size_t) (r / 4) * nb * sizeof(block_q8_Kx4)], N_PER_ROW);
    }

    // ---- 4. kernels ----
    const size_t sg_len = (size_t) NR_GEMV * NR_GEMV + NC + 64;
    const size_t sm_len = (size_t) NR_GEMM * BS_GEMM + 64;
    std::vector<float> sg_avx(sg_len, SENTINEL), sg_gen(sg_len, SENTINEL);
    std::vector<float> sm_avx(sm_len, SENTINEL), sm_gen(sm_len, SENTINEL);

    ggml_gemv_q4_K_8x8_q8_K(N_PER_ROW, sg_avx.data(), NC, repacked, q8.data(), NR_GEMV, NC);
    ggml_gemv_q4_K_8x8_q8_K_generic(N_PER_ROW, sg_gen.data(), NC, repacked, q8.data(), NR_GEMV, NC);
    ggml_gemm_q4_K_8x8_q8_K(N_PER_ROW, sm_avx.data(), BS_GEMM, repacked, q8x4.data(), NR_GEMM, NC);
    ggml_gemm_q4_K_8x8_q8_K_generic(N_PER_ROW, sm_gen.data(), BS_GEMM, repacked, q8x4.data(), NR_GEMM, NC);

    // ---- 5. dump ----
    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    const uint32_t hdr[6] = { (uint32_t) type, (uint32_t) N_PER_ROW, (uint32_t) NROWS,
                              (uint32_t) NR_GEMV, (uint32_t) NR_GEMM, (uint32_t) NC };
    write_section(f, hdr, sizeof(hdr));
    write_section(f, src.data(), src.size());
    write_section(f, repacked, wbytes);
    write_section(f, q8.data(), q8.size());
    write_section(f, q8x4.data(), q8x4.size());
    write_section(f, sg_avx.data(), sg_avx.size() * sizeof(float));
    write_section(f, sm_avx.data(), sm_avx.size() * sizeof(float));
    write_section(f, sg_gen.data(), sg_gen.size() * sizeof(float));
    write_section(f, sm_gen.data(), sm_gen.size() * sizeof(float));
    write_section(f, act_g.data(), act_g.size() * sizeof(float));
    write_section(f, act_m.data(), act_m.size() * sizeof(float));
    const uint32_t tr[2] = { trait_present, 0 };
    write_section(f, tr, sizeof(tr));
    fclose(f);

    bool avx_matches_generic_g = memcmp(sg_avx.data(), sg_gen.data(), sg_avx.size() * sizeof(float)) == 0;
    bool avx_matches_generic_m = memcmp(sm_avx.data(), sm_gen.data(), sm_avx.size() * sizeof(float)) == 0;
    printf("wrote %s\n", out_path);
    printf("gemv AVX == generic: %s\n", avx_matches_generic_g ? "yes" : "no");
    printf("gemm AVX == generic: %s\n", avx_matches_generic_m ? "yes" : "no");

    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return 0;
}