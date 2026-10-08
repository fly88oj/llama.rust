/* ref_clip_graph_dump.cpp — bit-exact node dump of the reference clip audio
 * encoder graphs. Loads a synthetic mmproj through the real libmtmd
 * (clip_init + clip_image_batch_encode) with the ggml eval callback, and
 * writes every computed node's raw bytes:
 *
 *   [u32 name_len][name bytes][u32 ne[4]][u64 nbytes][data bytes]
 *   ... one record per computed (non-view, non-noop) node, in graph order
 *
 * The mel input is read from a binary file ([i32 nx][i32 ny][f32 x nx*ny])
 * dumped by the port, so both sides run the identical input.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o parity/ref_clip_graph_dump parity/ref_clip_graph_dump.cpp \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/tools/mtmd \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lmtmd -lggml -lggml-base -lggml-cpu
 * Usage:
 *   ./parity/ref_clip_graph_dump <mmproj.gguf> <mel.bin> <out.bin> <fa 0|1> [n_threads]
 */
#include "clip-impl.h"
#include "clip.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static FILE * g_out = nullptr;

static void dump_node(struct ggml_tensor * t) {
    if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
        t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE) {
        return; // layout-only nodes: the port dumps none of these either
    }
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || t->data == nullptr) {
        return;
    }
    const size_t bs = ggml_nbytes(t);

    const char * name = ggml_op_desc(t);
    uint32_t nl = (uint32_t) strlen(name);
    fwrite(&nl, 4, 1, g_out);
    fwrite(name, 1, nl, g_out);
    uint32_t ne[4] = { (uint32_t) t->ne[0], (uint32_t) t->ne[1], (uint32_t) t->ne[2], (uint32_t) t->ne[3] };
    fwrite(ne, 4, 4, g_out);
    uint64_t nb = (uint64_t) bs;
    fwrite(&nb, 8, 1, g_out);
    fwrite(t->data, 1, bs, g_out);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        return true; // always retrieve data
    }
    dump_node(t);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <mmproj> <mel.bin> <out.bin> <fa 0|1> [n_threads]\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const char * mel_path = argv[2];
    const char * out_path = argv[3];
    const int fa_on = atoi(argv[4]);
    const int n_threads = argc > 5 ? atoi(argv[5]) : 4;

    /* the mel chunk dumped by the port: [i32 nx][i32 ny][f32 x nx*ny] */
    FILE * mf = fopen(mel_path, "rb");
    if (!mf) { perror("mel fopen"); return 1; }
    int32_t nx = 0, ny = 0;
    if (fread(&nx, 4, 1, mf) != 1 || fread(&ny, 4, 1, mf) != 1) { return 1; }
    std::vector<float> mel((size_t) nx * ny);
    if (fread(mel.data(), 4, mel.size(), mf) != mel.size()) { return 1; }
    fclose(mf);

    g_out = fopen(out_path, "wb");
    if (!g_out) { perror("out fopen"); return 1; }

    clip_context_params params;
    memset(&params, 0, sizeof(params));
    params.flash_attn_type = fa_on ? CLIP_FLASH_ATTN_TYPE_ENABLED : CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    params.cb_eval = cb_eval;
    params.use_gpu = false;

    clip_init_result res = clip_init(mmproj, params);
    if (res.ctx_a == nullptr) {
        fprintf(stderr, "clip_init failed (no audio ctx)\n");
        return 1;
    }
    clip_ctx * ctx = res.ctx_a;

    clip_image_f32 img;
    img.set_size({ nx, ny }, false, true);
    img.cpy_buf(mel);

    clip_image_f32_batch batch;
    batch.entries.push_back(img);
    batch.is_audio = true;

    std::vector<float> embd;
    if (!clip_image_batch_encode(ctx, n_threads, &batch, embd)) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }

    clip_free(ctx);
    fclose(g_out);
    fprintf(stderr, "wrote %s (%d tokens x %d embd)\n", out_path, (int) (embd.size() / 896), 896);
    return 0;
}
