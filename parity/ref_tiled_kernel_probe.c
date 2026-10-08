/* Direct-kernel ground truth for the tiled family (sync batch D): dlsym the
 * exported tiled_run_microtile / tiled_repack_src1 instantiations from the
 * NEW reference libggml-cpu and dump (tile inputs, acc outputs) pairs so the
 * Rust port can pin the kernel in isolation (no driver, no quantization).
 *
 * Build: gcc -O2 parity/ref_tiled_kernel_probe.c -o /tmp/syncg-tkprobe -ldl
 * Run:   /tmp/syncg-tkprobe parity/tiled_kernel_ref.bin
 *
 * Section (per case): 'VTK1' | u32 variant | u32 num_k | u32 slab |
 *   src0 tile bytes (q 65536, d 1024, dmin 1024, scales 16384, mins 16384) |
 *   src1 tile bytes (q 65536, bsums 16384, d 1024) |
 *   acc pre bytes (65536*4) | repacked-src1 q bytes (65536) |
 *   acc post bytes (65536*4). EOF 0xFFFFFFFF.
 */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_K 256
#define TILE_ROWS 256
#define MICRO 16
#define NB_MAX (TILE_K / 16)

struct tiled_tile_src0 {
    _Alignas(64) uint8_t q[TILE_ROWS * TILE_K];
    float d[TILE_ROWS];
    float dmin[TILE_ROWS];
    int32_t scales[TILE_ROWS * NB_MAX];
    int32_t mins[TILE_ROWS * NB_MAX];
};
struct tiled_tile_src1 {
    _Alignas(64) int8_t q[TILE_ROWS * TILE_K];
    _Alignas(64) int32_t bsums[(TILE_K / MICRO) * TILE_ROWS];
    float d[TILE_ROWS];
};
struct tiled_ws {
    struct tiled_tile_src0 src0;
    struct tiled_tile_src1 src1;
    _Alignas(64) float acc[TILE_ROWS * TILE_ROWS];
};

typedef void (*microtile_fn)(const struct tiled_tile_src0 *, const struct tiled_tile_src1 *,
                             int, int, int, int, float *, int);
typedef void (*repack1_fn)(struct tiled_tile_src1 *, int, int, bool);

static uint32_t lcg = 0x6b2c9f1d;
static uint32_t nxt(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }
static float nxt_f(void) {
    int32_t v = (int32_t)(nxt() & 0xFFFF);
    return (float)v / (float)(1 << 14) - 2.0f;
}

int main(int argc, char **argv) {
    const char *so = "/home/jeffrey/llm/llama.cpp-next/build-rust-ref/bin/libggml-cpu.so";
    const char *out = argc > 1 ? argv[1] : "tiled_kernel_ref.bin";
    void *h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    /* q4_K/q5_K <32,true,0,false> */
    microtile_fn mt_q4k = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi32ELb1ELi0ELb0EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_q6k = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi16ELb0ELi32ELb1EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_iq2xxs = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi32ELb0ELi128ELb1EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_iq4xs = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi32ELb0ELi128ELb0EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_q3k = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi16ELb0ELi4ELb1EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_q2k = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi16ELb1ELi0ELb0EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    microtile_fn mt_iq2xs = (microtile_fn)dlsym(h, "_Z19tiled_run_microtileILi16ELb0ELi128ELb1EEvRK15tiled_tile_src0RK15tiled_tile_src1iiiiPfi");
    repack1_fn repack1 = (repack1_fn)dlsym(h, "_Z17tiled_repack_src1P15tiled_tile_src1iib");
    if (!mt_q4k || !mt_q6k || !mt_iq2xxs || !mt_iq4xs || !mt_q3k || !mt_q2k || !mt_iq2xs || !repack1) {
        fprintf(stderr, "dlsym failed\n");
        return 1;
    }

    struct tiled_ws *ws = aligned_alloc(64, sizeof(*ws));
    memset(ws, 0, sizeof(*ws));
    FILE *f = fopen(out, "wb");

    struct { const char *name; microtile_fn fn; int subblk; int bias; } cases[] = {
        {"q4k", mt_q4k, 32, 0}, {"q6k", mt_q6k, 16, 32}, {"iq2xxs", mt_iq2xxs, 32, 128},
        {"iq4xs", mt_iq4xs, 32, 128}, {"q3k", mt_q3k, 16, 4}, {"q2k", mt_q2k, 16, 0},
        {"iq2xs", mt_iq2xs, 16, 128},
    };
    /* tile coords + num_k/slab combos: standard (nk=1,s=0) and narrow
     * (nk=2/4, slab = each) */
    /* num_k > 1 (narrow) always runs at j0 == 0 in the driver — the tile
     * only holds num_k*256 bytes per row band, so other bands would read out
     * of the workspace */
    int coord[][4] = {
        {0, 0, 1, 0}, {0, 0, 2, 0}, {0, 0, 2, 1}, {0, 0, 4, 0}, {0, 0, 4, 3},
        {8, 16, 1, 0}, {16, 240, 1, 0}, {240, 0, 1, 0}, {240, 240, 1, 0},
        {128, 128, 1, 0}, {240, 16, 1, 0}, {8, 0, 4, 2},
    };
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        int subblk = cases[c].subblk;
        int nb = TILE_K / subblk;
        int ns = subblk / MICRO;
        for (unsigned v = 0; v < sizeof(coord) / sizeof(coord[0]); v++) {
            /* deterministic tiles: plausible code ranges so the integer math
             * stays in the q4_K/q8_K operating envelope */
            for (int i = 0; i < TILE_ROWS * TILE_K; i++)
                ws->src0.q[i] = cases[c].bias ? (uint8_t)(nxt() & 0xFF) : (uint8_t)(nxt() % 16);
            for (int i = 0; i < TILE_ROWS; i++) {
                ws->src0.d[i] = nxt_f();
                ws->src0.dmin[i] = nxt_f();
            }
            for (int i = 0; i < TILE_ROWS * NB_MAX; i++) {
                ws->src0.scales[i] = (int32_t)(nxt() % 64) - 32;
                ws->src0.mins[i] = (int32_t)(nxt() % 64) - 32;
            }
            for (int i = 0; i < TILE_ROWS * TILE_K; i++)
                ws->src1.q[i] = (int8_t)(nxt() & 0xFF);
            for (int i = 0; i < (TILE_K / MICRO) * TILE_ROWS; i++)
                ws->src1.bsums[i] = (int32_t)(nxt() & 0xFFFF) - 32768;
            for (int i = 0; i < TILE_ROWS; i++) ws->src1.d[i] = nxt_f();

            int num_k = coord[v][2], slab = coord[v][3];

            /* random-ish initial acc (the += semantics must show through) */
            for (int i = 0; i < TILE_ROWS * TILE_ROWS; i++) ws->acc[i] = nxt_f();

            /* dump the PRE-repack tiles so the replay applies its own
             * repack_src1 (the transpose is part of what is pinned) */
            uint32_t hdr[4] = {0x314B5456 /* 'VTK1' */, c, num_k, slab};
            fwrite(hdr, 4, 4, f);
            fwrite(&ws->src0, sizeof(ws->src0), 1, f);
            fwrite(&ws->src1, sizeof(ws->src1), 1, f);
            fwrite(ws->acc, 4, TILE_ROWS * TILE_ROWS, f);

            /* repack ONLY the bands this call reads, exactly once (the driver
             * repacks just-in-time; num_k > 1 is always band 0 only) */
            if (num_k > 1) {
                repack1(&ws->src1, 0, num_k, true);
            } else {
                repack1(&ws->src1, (coord[v][1] / MICRO) * MICRO, 1, true);
            }

            cases[c].fn(&ws->src0, &ws->src1, coord[v][0], coord[v][1], num_k, slab,
                        ws->acc, TILE_ROWS);

            fwrite(ws->acc, 4, TILE_ROWS * TILE_ROWS, f);
        }
        printf("%s: %zu variants\n", cases[c].name, sizeof(coord) / sizeof(coord[0]));
    }
    uint32_t end = 0xFFFFFFFF;
    fwrite(&end, 4, 1, f);
    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
