// ref_gemm_ab.c — isolated A/B bench of the reference's repack gemm kernels,
// driven by the same on-disk inputs the port-side test writes (see
// crates/ggml/src/repack.rs, bench_repack_gemm_vs_ref / _q4_0_vs_ref). This
// is the C side of the kernel-level comparison instrument: same repacked
// weights, same quantized activations, same shapes; only the kernel differs.
//
//   ./ref_gemm_ab <case> <n> <nc> <nr>
//
//   case ∈ mxfp4 | q4_0 | q4_k
//
// reads  /tmp/gemmab/<case>_{rep,act}.bin   (must match <n>/<nc>/<nr> — the
//                                           exact tile sizes are validated)
// writes /tmp/gemmab/<case>_ref_out.bin     (for the port's bit comparison)
//         /tmp/gemmab/<case>_ref_meta.txt   ("n nc nr fnv1a(rep||act)" — the
//                                           port refuses stale outputs on
//                                           this handshake)
// prints best-of-5 wall time + GB/s of weight traffic
//
// build (from the repo root, against the pinned reference build):
//   gcc -O2 -o /tmp/ref_gemm_ab parity/ref_gemm_ab.c \
//     -L$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin \
//     -l:libggml-cpu.so.0.26.0 -l:libggml-base.so.0.26.0 \
//     -Wl,-rpath,$LLAMA_RUST_DATA_HOME/llm/llama.cpp-next/build-rust-ref/bin
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ggml-cpu exports (repack.h signatures); ggml_cpu_init fills the
// ggml_table_f32_e8m0_half lookup the kernels read (ggml-cpu.c:3895)
extern void ggml_cpu_init(void);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(*len);
    if (fread(p, 1, *len, f) != *len) { perror("fread"); exit(2); }
    fclose(f);
    return p;
}

static uint64_t fnv1a2(const void *a, size_t na, const void *b, size_t nb) {
    const uint8_t *pa = a, *pb = b;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < na; i++) { h ^= pa[i]; h *= 0x100000001b3ULL; }
    for (size_t i = 0; i < nb; i++) { h ^= pb[i]; h *= 0x100000001b3ULL; }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <mxfp4|q4_0|q4_k> <n> <nc> <nr>\n", argv[0]);
        return 1;
    }
    ggml_cpu_init(); // without it every col_scale reads 0.0 (all-zero table)
    const char *cs = argv[1];
    int n = atoi(argv[2]), nc = atoi(argv[3]), nr = atoi(argv[4]);

    // per-case geometry: q4_K works on 256-wide tiles, the lut formats on
    // 32-wide blocks (tile sizes mirror crates/ggml/src/repack.rs)
    int is_q4k = !strcmp(cs, "q4_k");
    int is_mxfp4 = !strcmp(cs, "mxfp4");
    int is_q4_0 = !strcmp(cs, "q4_0");
    if (!is_q4k && !is_mxfp4 && !is_q4_0) {
        fprintf(stderr, "unknown case '%s' (mxfp4 | q4_0 | q4_k)\n", cs);
        return 1;
    }
    size_t kblk = is_q4k ? 256 : 32;
    size_t rep_tile = is_q4k ? 1152 : (is_mxfp4 ? 136 : 144);
    size_t act_tile = is_q4k ? 1168 : 136;
    // one rep_tile covers EIGHT weight rows (the 8x8 interleave)
    size_t want_rep = (size_t)(nc / 8) * (n / (int)kblk) * rep_tile;
    size_t want_act = (size_t)(nr / 4) * (n / (int)kblk) * act_tile;

    char path[256];
    size_t rep_len, act_len;
    snprintf(path, sizeof path, "/tmp/gemmab/%s_rep.bin", cs);
    void *rep = read_all(path, &rep_len);
    snprintf(path, sizeof path, "/tmp/gemmab/%s_act.bin", cs);
    void *act = read_all(path, &act_len);
    if (rep_len != want_rep || act_len != want_act) {
        fprintf(stderr, "input/shape mismatch for %s n=%d nc=%d nr=%d: rep %zu (want %zu), act %zu (want %zu)\n",
                cs, n, nc, nr, rep_len, want_rep, act_len, want_act);
        return 2;
    }

    float *out = malloc((size_t)nr * nc * sizeof(float));
    memset(out, 0, (size_t)nr * nc * sizeof(float));

    double best = 1e9;
    for (int r = 0; r < 5; r++) {
        memset(out, 0, (size_t)nr * nc * sizeof(float));
        double t0 = now_s();
        if (is_mxfp4)
            ggml_gemm_mxfp4_8x8_q8_0(n, out, nc, rep, act, nr, nc);
        else if (is_q4_0)
            ggml_gemm_q4_0_8x8_q8_0(n, out, nc, rep, act, nr, nc);
        else
            ggml_gemm_q4_K_8x8_q8_K(n, out, nc, rep, act, nr, nc);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    snprintf(path, sizeof path, "/tmp/gemmab/%s_ref_out.bin", cs);
    FILE *f = fopen(path, "wb");
    fwrite(out, sizeof(float), (size_t)nr * nc, f);
    fclose(f);

    snprintf(path, sizeof path, "/tmp/gemmab/%s_ref_meta.txt", cs);
    f = fopen(path, "w");
    fprintf(f, "%d %d %d %016llx\n", n, nc, nr,
            (unsigned long long)fnv1a2(rep, rep_len, act, act_len));
    fclose(f);

    double bytes = (double)want_rep; // one full pass over the repacked weights
    printf("ref %-6s n=%d nc=%d nr=%d: %9.3f ms  %7.2f GB/s (weights)  out[0]=%.4f out[1]=%.4f out[nc]=%.4f\n",
           cs, n, nc, nr, best * 1e3, bytes / best / 1e9, out[0], out[1], out[nc]);
    return 0;
}
