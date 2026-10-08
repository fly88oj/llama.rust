/* CPU_REPACK reference dump harness — ground-truth repacked bytes + gemv/gemm
 * outputs from the pinned llama.cpp (bd4f514db1) CPU backend for bit-exact Rust
 * parity tests (crates/ggml/src/repack.rs).
 *
 * The reference binary is built with GGML_NATIVE=ON + GGML_USE_CPU_REPACK=ON on
 * this machine (AVX512F/BW/DQ/VNNI + AVX2), so it takes exactly the code path
 * this dump captures:
 *   * MXFP4 -> mxfp4_8x8_q8_0 traits (repack.cpp:5093-5099) -> the buffer's
 *     `set_tensor` runs make_block_mxfp4x8 (repack.cpp:4178-4200) and gemv/gemm
 *     dispatch to ggml_gemv_mxfp4_8x8_q8_0 / ggml_gemm_mxfp4_8x8_q8_0
 *     (arch/x86/repack.cpp:1700 / 3511 -> gemv|gemm_q4_b32_8x8_q8_0_lut_avx).
 *   * Everything is produced through the *public* backend API
 *     (ggml_backend_cpu_repack_buffer_type + set_tensor) plus the exported
 *     extern "C" kernels, so no reference implementation is re-typed here.
 *
 * Sections dumped (single file, each section = u32 len + bytes):
 *   [0] u32 n_per_row | u32 nrows | u32 nr_gemv | u32 nr_gemm | u32 nc | u32 pad
 *   [1] src block_mxfp4 bytes              (nrows*nb*17)
 *   [2] repacked 8x8 bytes                 (same length)
 *   [3] q8_0 activations, plain            (nr_gemv*nb*34)
 *   [4] q8_0x4 activations, 4x8 interleave (nr_gemm/4*nb*136)
 *   [5] gemv AVX output   (f32, NR*NR+nc+64 slots, sentinel 1234.5f elsewhere)
 *   [6] gemm AVX output   (f32, nr_gemm*bs+64 slots)
 *   [7] gemv *_generic output (same size as [5])
 *   [8] gemm *_generic output (same size as [6])
 *   [9] activation floats for the gemv rows (f32, nr_gemv*n_per_row)
 *  [10] activation floats for the gemm rows (f32, nr_gemm*n_per_row)
 *
 * Build (from the repo root):
 *   PINNED=/home/jeffrey/llm/llama.cpp-pinned
 *   REF=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin
 *   g++ -O2 -std=c++17 parity/ref_repack_dump.cpp \
 *       -I$PINNED/ggml/include -I$PINNED/ggml/src -I$PINNED/ggml/src/ggml-cpu \
 *       -o parity/ref_repack_dump -L$REF -lggml -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,$REF
 * Run: ./parity/ref_repack_dump parity/repack_ref.bin
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "repack.h"

// libggml-cpu.so entry point (the backend module self-registers through this)
extern "C" ggml_backend_reg_t ggml_backend_cpu_reg(void);

#define N_PER_ROW 96
#define NROWS     16
#define NR_GEMV   1
#define NR_GEMM   20
#define NC        NROWS
#define BS_GEMM   NC
#define SENTINEL  1234.5f

static uint32_t lcg_state = 0x2468ace0u;
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
    fwrite(data, 1, len, f);
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "parity/repack_ref.bin";

    // ---- register the CPU backend (its reg entry creates the devices) ----
    ggml_backend_register(ggml_backend_cpu_reg());
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        fprintf(stderr, "no CPU backend device\n");
        return 1;
    }

    const ggml_type type   = GGML_TYPE_MXFP4;
    const int64_t    nb     = N_PER_ROW / QK_MXFP4;             // k-blocks per row
    const size_t     nb_row = (size_t) nb * sizeof(block_mxfp4);

    // ---- 1. source MXFP4 bytes (random nibbles + E8M0 exponents) ----
    std::vector<uint8_t> src((size_t) NROWS * nb_row);
    for (size_t i = 0; i < src.size(); i += sizeof(block_mxfp4)) {
        block_mxfp4 * b = (block_mxfp4 *) &src[i];
        const size_t ib = i / sizeof(block_mxfp4);
        // a spread of exponents covering normals plus the denormal edge (0,1,2)
        b->e = (ib % 16 == 15) ? (uint8_t) (next_u32() % 3) : (uint8_t) (118 + next_u32() % 16);
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            b->qs[j] = (uint8_t) next_u32(); // covers all 16 E2M1 codes both nibbles
        }
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
    w->buffer = wbuf;                                  // what ggml_backend_alloc_ctx_tensors would do
    w->data   = ggml_backend_buffer_get_base(wbuf);     // idem (single-tensor buffer)
    ggml_backend_tensor_set(w, src.data(), 0, wbytes); // -> repack_mxfp4_to_mxfp4_8_bl
    const uint8_t * repacked = (const uint8_t *) w->data;
    printf("repacked tensor: %lld x %lld, %zu bytes\n", (long long) w->ne[0], (long long) w->ne[1], wbytes);

    // ---- 3. activations ----
    const ggml_type_traits_cpu * tt_q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    GGML_ASSERT(tt_q8->from_float != nullptr);

    std::vector<float> act_g(NR_GEMV * N_PER_ROW);
    for (float & v : act_g) v = next_f32();
    std::vector<uint8_t> q8((size_t) NR_GEMV * nb * sizeof(block_q8_0));
    for (int r = 0; r < NR_GEMV; r++) {
        tt_q8->from_float(&act_g[r * N_PER_ROW], &q8[(size_t) r * nb * sizeof(block_q8_0)], N_PER_ROW);
    }

    GGML_ASSERT(sizeof(block_q8_0x4) == 4 * sizeof(ggml_half) + QK8_0 * 4);
    std::vector<float> act_m(NR_GEMM * N_PER_ROW);
    for (float & v : act_m) v = next_f32();
    std::vector<uint8_t> q8x4((size_t) (NR_GEMM / 4) * nb * sizeof(block_q8_0x4));
    for (int r = 0; r < NR_GEMM; r += 4) {
        // INTER_SIZE == NB_COLS == 8 for mxfp4_8x8_q8_0 -> ggml_quantize_mat_t<8,_>
        // -> ggml_quantize_mat_q8_0_4x8 (repack.cpp:325), NOT the 4x4 variant
        ggml_quantize_mat_q8_0_4x8(&act_m[r * N_PER_ROW], &q8x4[(size_t) (r / 4) * nb * sizeof(block_q8_0x4)], N_PER_ROW);
    }

    // ---- 4. kernels ----
    const size_t sg_len = (size_t) NR_GEMV * NR_GEMV + NC + 64;
    const size_t sm_len = (size_t) NR_GEMM * BS_GEMM + 64;
    std::vector<float> sg_avx(sg_len, SENTINEL), sg_gen(sg_len, SENTINEL);
    std::vector<float> sm_avx(sm_len, SENTINEL), sm_gen(sm_len, SENTINEL);

    ggml_gemv_mxfp4_8x8_q8_0(N_PER_ROW, sg_avx.data(), NC, repacked, q8.data(), NR_GEMV, NC);
    ggml_gemv_mxfp4_8x8_q8_0_generic(N_PER_ROW, sg_gen.data(), NC, repacked, q8.data(), NR_GEMV, NC);
    ggml_gemm_mxfp4_8x8_q8_0(N_PER_ROW, sm_avx.data(), BS_GEMM, repacked, q8x4.data(), NR_GEMM, NC);
    ggml_gemm_mxfp4_8x8_q8_0_generic(N_PER_ROW, sm_gen.data(), BS_GEMM, repacked, q8x4.data(), NR_GEMM, NC);

    // ---- 5. dump ----
    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    const uint32_t hdr[6] = { (uint32_t) N_PER_ROW, (uint32_t) NROWS, (uint32_t) NR_GEMV,
                              (uint32_t) NR_GEMM, (uint32_t) NC, 0 };
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