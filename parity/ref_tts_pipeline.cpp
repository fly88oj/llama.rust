/* ref_tts_pipeline.cpp — reference-side driver of the TTS *pipeline* layer
 * (pinned bd4f514db1): the exact calls tools/tts/tts.cpp makes
 * (mtmd_helper_gen_audio_{init,set_input,step_prompt,step_gen,get_output},
 * mtmd-helper.h:189-292) over a synthetic trunk + gen mmproj pair.
 *
 * The trunk is the port's synthetic dflash GGUF (an arch whose generic
 * decoder consumes embd-only ubatches — dflash.cpp:609-677's KV-injection
 * graph — exactly what the pipeline's decode_embd_batch feeds,
 * mtmd-helper-common.h:73-100), and the backbone token stream is a FIXED
 * list (qwen3tts; tts.cpp:150 samples from trunk logits, which the injection
 * graph does not produce — the fixed stream keeps both sides on the same
 * deterministic data flow).
 *
 * Dumps (the formats parity/tts_pipeline_parity.sh compares against the
 * port's tests/tts_pipeline_e2e.rs):
 *   {out}-pcm.bin: [u32 n_prompt][u32 n_h][u32 h_len][f32 h × n_h]
 *                  [u32 n_pcm][f32 pcm][u32 state_len][u8 trunk state]
 *                  (n_h counts the prompt-final h plus one row per step)
 *   {out}-wav.bin: [u32 rate][u32 wav_len][u8 wav bytes]
 *
 * Build:
 *   g++ -O2 -std=c++17 -o parity/ref_tts_pipeline parity/ref_tts_pipeline.cpp \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/include \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/tools/mtmd \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lmtmd -llama -lggml -lggml-base -lggml-cpu
 * Usage:
 *   ./parity/ref_tts_pipeline <trunk.gguf> <mmproj.gguf> <qwen3tts|pockettts> \
 *       <pcm|wav> <prompt> <lang> <seed> <top_k> <top_p> <out-prefix> \
 *       [sampled.bin]
 */
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void put_u32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }

static void die(const char * what) {
    fprintf(stderr, "ref_tts_pipeline: %s failed\n", what);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 11) {
        fprintf(stderr, "usage: %s <trunk> <mmproj> <qwen3tts|pockettts> <pcm|wav> "
                        "<prompt> <lang> <seed> <top_k> <top_p> <out-prefix> [sampled.bin]\n",
                argv[0]);
        return 1;
    }
    const char * trunk   = argv[1];
    const char * mmproj  = argv[2];
    const std::string mode   = argv[3];
    const std::string outty  = argv[4];
    const char * prompt   = argv[5];
    const char * lang     = argv[6];
    const uint32_t seed   = (uint32_t) strtoul(argv[7], nullptr, 10);
    const int32_t  top_k  = atoi(argv[8]);
    const float    top_p  = strtof(argv[9], nullptr);
    const char * outpre   = argv[10];

    llama_backend_init();

    // the trunk: plain load + a context with embeddings on (tts.cpp:53
    // `params.embedding = true` — the hidden states must flow)
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(trunk, mp);
    if (!model) die("llama_model_load_from_file");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = 512;
    cp.n_batch     = 512;
    cp.n_seq_max   = 1;
    cp.embeddings  = true;
    cp.n_threads   = 4;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) die("llama_init_from_model");

    // the generator
    mtmd_context_params mtp = mtmd_context_params_default();
    mtp.use_gpu  = false;
    mtp.warmup   = false;
    mtp.n_threads = 4;
    mtp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_context * mctx = mtmd_init_from_file(mmproj, model, mtp);
    if (!mctx) die("mtmd_init_from_file");

    mtmd_helper_gen_audio * gen = mtmd_helper_gen_audio_init(lctx, mctx);
    if (!gen) die("mtmd_helper_gen_audio_init");

    // the fixed backbone stream (the port's q3t-sampled.bin)
    std::vector<llama_token> sampled;
    if (argc > 11) {
        FILE * f = fopen(argv[11], "rb");
        if (!f) die("open sampled.bin");
        uint32_t n = 0;
        if (fread(&n, 4, 1, f) != 1) die("read sampled n");
        sampled.resize(n);
        if (fread(sampled.data(), 4, n, f) != n) die("read sampled ids");
        fclose(f);
    }

    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id      = 0;
    inp.prompt      = prompt;
    inp.prompt_len  = strlen(prompt);
    inp.speaker_ref = nullptr;
    inp.lang        = lang;
    inp.top_k       = top_k;
    inp.top_p       = top_p;
    inp.seed        = seed;
    inp.out_type    = outty == "wav" ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV
                                     : MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

    if (mtmd_helper_gen_audio_set_input(gen, &inp) != 0) die("set_input");

    // stage 1: the prompt batches (tts.cpp:130-140 — n_batch 8 like the port)
    for (;;) {
        int32_t ret = mtmd_helper_gen_audio_step_prompt(gen, 8);
        if (ret < 0) die("step_prompt");
        if (ret == 0) break;
    }

    // the per-step trunk hidden states (the prompt-final row first,
    // tts.cpp:158 llama_get_embeddings_ith(lctx, -1))
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> hs;
    {
        const float * h0 = llama_get_embeddings_ith(lctx, -1);
        if (!h0) die("prompt embeddings");
        hs.assign(h0, h0 + n_embd);
    }

    // stage 2+3: the generation loop (tts.cpp:163-180's flow, the fixed
    // token stream instead of common_sampler_sample)
    size_t step = 0;
    bool stop = false;
    while (!stop && step < 1200) {
        llama_token t = LLAMA_TOKEN_NULL;
        if (mode == "qwen3tts") {
            if (step >= sampled.size()) break;
            t = sampled[step];
        }
        const float * h_in = hs.data() + (hs.size() - n_embd);
        const float * h_next = nullptr;
        if (mtmd_helper_gen_audio_step_gen(gen, t, h_in, &h_next, &stop) != 0) {
            die("step_gen");
        }
        if (!h_next) break; // stopped without generating a frame
        hs.insert(hs.end(), h_next, h_next + n_embd);
        step++;
    }
    fprintf(stderr, "ref_tts_pipeline: %s %s — %zu steps\n", mode.c_str(), outty.c_str(), step);

    // the output + the trunk's final sequence state
    int32_t      rate  = 0;
    const char * data  = nullptr;
    size_t       dlen  = 0;
    int64_t      n_samples = 0;
    if (mtmd_helper_gen_audio_get_output(gen, &rate, &data, &dlen, &n_samples) != 0) {
        die("get_output");
    }

    if (outty == "pcm") {
        const size_t state_size = llama_state_seq_get_size(lctx, 0);
        std::vector<uint8_t> state(state_size);
        const size_t got = llama_state_seq_get_data(lctx, state.data(), state_size, 0);
        state.resize(got);

        std::string path = std::string(outpre) + "-pcm.bin";
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) die("open dump");
        put_u32(f, (uint32_t) 0); // n_prompt placeholder — the script patches it from the port dump
        put_u32(f, (uint32_t) (hs.size() / n_embd));
        put_u32(f, (uint32_t) n_embd);
        fwrite(hs.data(), 4, hs.size(), f);
        put_u32(f, (uint32_t) (dlen / 4));
        fwrite(data, 1, dlen, f);
        put_u32(f, (uint32_t) state.size());
        fwrite(state.data(), 1, state.size(), f);
        fclose(f);
        fprintf(stderr, "ref_tts_pipeline: wrote %s (%zu h rows, %zu pcm, %zu state)\n",
                path.c_str(), hs.size() / n_embd, dlen / 4, state.size());
    } else {
        std::string path = std::string(outpre) + "-wav.bin";
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) die("open dump");
        put_u32(f, (uint32_t) rate);
        put_u32(f, (uint32_t) dlen);
        fwrite(data, 1, dlen, f);
        fclose(f);
        fprintf(stderr, "ref_tts_pipeline: wrote %s (%zu bytes, %d Hz)\n",
                path.c_str(), dlen, rate);
    }

    mtmd_helper_gen_audio_free(gen);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
