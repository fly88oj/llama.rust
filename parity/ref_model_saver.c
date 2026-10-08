// ref_model_saver.c — drives the pinned reference's llama_model_save_to_file
// (llama.cpp:498-503 -> llama_model_saver, llama-model-saver.cpp) so the
// port's saver output can be byte-compared against it.
//
// The pinned build ships no tool that calls the saver (finetune is not built),
// so this probe links libllama.so directly:
//
//   gcc -O2 parity/ref_model_saver.c -o /tmp/ref_model_saver \
//       -I/home/jeffrey/llm/llama.cpp-pinned/include \
//       -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
//       -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//       -lllama -lggml -lggml-base
//   LD_LIBRARY_PATH=/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
//     /tmp/ref_model_saver <model.gguf> <out.gguf>

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    struct llama_model_params p = llama_model_default_params();
    // no CPU_REPACK extra buffer types: the saver walks the loaded tensors'
    // storage, and repacked buffers both change the bytes and crash the
    // reference's write path — the probe pins the plain mmap'd weights
    p.use_extra_bufts = false;
    struct llama_model * m = llama_model_load_from_file(argv[1], p);
    if (!m) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    llama_model_save_to_file(m, argv[2]);
    fprintf(stderr, "saved %s -> %s\n", argv[1], argv[2]);

    llama_model_free(m);
    llama_backend_free();
    return 0;
}
