// ref_gemm_ab.c — isolated A/B bench of the reference's repack gemm kernels,
// driven by the same on-disk inputs the port-side test writes (see
// crates/ggml/src/repack.rs, bench_repack_gemm_vs_ref). This is the C side of
// the kernel-level comparison instrument: same repacked weights, same
// quantized activations, same shapes; only the kernel differs.
//
//   ./ref_gemm_ab <case>      # mxfp4 | q4_k
//
// reads  /tmp/gemmab/<case>_{rep,act}.bin
// writes /tmp/gemmab/<case>_ref_out.bin   (for the port's bit comparison)
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
void ggml_gemm_q4_K_8x8_q8_K(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);

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

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <mxfp4|q4_k> <n> <nc> <nr> <weight_bytes_per_tile>\n", argv[0]);
        return 1;
    }
    ggml_cpu_init(); // without it every col_scale reads 0.0 (all-zero table)
    const char *cs = argv[1];
    int n = atoi(argv[2]), nc = atoi(argv[3]), nr = atoi(argv[4]);
    char path[256];
    size_t rep_len, act_len;
    snprintf(path, sizeof path, "/tmp/gemmab/%s_rep.bin", cs);
    void *rep = read_all(path, &rep_len);
    snprintf(path, sizeof path, "/tmp/gemmab/%s_act.bin", cs);
    void *act = read_all(path, &act_len);
    (void)argv[5];

    float *out = malloc((size_t)nr * nc * sizeof(float));
    memset(out, 0, (size_t)nr * nc * sizeof(float));

    double best = 1e9;
    for (int r = 0; r < 5; r++) {
        memset(out, 0, (size_t)nr * nc * sizeof(float));
        double t0 = now_s();
        if (!strcmp(cs, "mxfp4"))
            ggml_gemm_mxfp4_8x8_q8_0(n, out, nc, rep, act, nr, nc);
        else if (!strcmp(cs, "q4_0"))
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

    // weight traffic: repacked tile bytes per 8 rows per 32-col block
    double bytes = (double)nc * n / 32.0 * (cs[0] == 'm' ? 136.0 : 144.0);
    double sum = 0; int nz = 0;
    for (int i = 0; i < nr * nc; i++) { sum += out[i]; if (out[i] != 0.0f) nz++; }
    printf("ref %-6s n=%d nc=%d nr=%d: %9.3f ms  %7.2f GB/s (weights)  out[0]=%.4f out[1]=%.4f out[nc]=%.4f nz=%d/%d sum=%.1f\n",
           cs, n, nc, nr, best * 1e3, bytes / best / 1e9, out[0], out[1], out[nc], nz, nr * nc, sum);
    return 0;
}
