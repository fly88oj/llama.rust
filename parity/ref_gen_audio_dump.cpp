/* ref_gen_audio_dump.cpp — reference-side dump of the TTS generator
 * pipelines (qwen3tts_gen / pockettts_gen), pinned bd4f514db1.
 *
 * Drives libmtmd exactly the way mtmd_gen_audio_process does
 * (mtmd.cpp:1922-2031): clip_init picks the gen-audio context, then
 * clip_encode runs with gen_process = GEN_CODE (fixed hidden state + code0,
 * fixed seed — the mt19937 draws are then deterministic) or GEN_WAV
 * (codes/feats window + optional state_in blob), and the outputs are
 * written raw:
 *
 *   [u32 n_codes][i32 codes][u32 n_embd][f32 embd][u32 n_feats][f32 feats]
 *   [u32 is_eos][u32 n_samples][f32 audio][u32 state_size][u8 state]
 *
 * (the same format the port's tests/tts_gen_e2e.rs writes — compared by
 * parity/tts_parity.sh).
 *
 * Build:
 *   g++ -O2 -std=c++17 -o parity/ref_gen_audio_dump parity/ref_gen_audio_dump.cpp \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/tools/mtmd \
 *     -I/home/jeffrey/llm/llama.cpp-pinned/ggml/include \
 *     -L/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -Wl,-rpath,/home/jeffrey/llm/llama.cpp/build-rust-ref/bin \
 *     -lmtmd -lggml -lggml-base -lggml-cpu
 * Usage:
 *   ./parity/ref_gen_audio_dump <mmproj.gguf> gencode <hstate.bin> <out.bin> \
 *       <seed> <code0> <top_k> <top_p> <temp> [n_threads]
 *   ./parity/ref_gen_audio_dump <mmproj.gguf> genwav-codes <codes.bin> <out.bin> \
 *       <seed> [n_threads]                (codes.bin: n_frames*n_codes i32, cold)
 *   ./parity/ref_gen_audio_dump <mmproj.gguf> genwav-codes <codes.bin> <out.bin> \
 *       <seed> [n_threads]                (codes.bin: n_frames*n_codes i32 + state blob)
 *   ./parity/ref_gen_audio_dump <mmproj.gguf> genwav-feats <feats.bin> <out.bin> \
 *       <seed> <n_latent> [n_threads]     (feats.bin: n_latent*n_frames f32 [+ state])
 */
#include "clip-impl.h"
#include "clip.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static void put_u32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }

template <typename T>
static std::vector<T> read_blob(const char * path) {
    FILE * fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "open %s failed\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<T> v(sz / sizeof(T));
    if (fread(v.data(), sizeof(T), v.size(), fp) != v.size()) { exit(1); }
    fclose(fp);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <mmproj> <gencode|genwav-codes|genwav-feats> <in.bin> <out.bin> <seed> ...\n", argv[0]);
        return 1;
    }
    const char * mmproj = argv[1];
    const std::string mode = argv[2];
    const char * in_path  = argv[3];
    const char * out_path = argv[4];
    const uint32_t seed = (uint32_t) strtoul(argv[5], nullptr, 10);
    int n_threads = argc > 10 ? atoi(argv[10]) : 4;

    clip_context_params params;
    memset(&params, 0, sizeof(params));
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    params.use_gpu = false;

    clip_init_result res = clip_init(mmproj, params);
    if (res.ctx_gen_a == nullptr) {
        fprintf(stderr, "clip_init failed (no gen-audio ctx)\n");
        return 1;
    }
    clip_ctx * ctx = res.ctx_gen_a;

    FILE * out = fopen(out_path, "wb");
    if (!out) { perror("out fopen"); return 1; }

    if (mode == "gencode") {
        if (argc < 10) { fprintf(stderr, "gencode needs code0 top_k top_p temp (argc=%d)\n", argc); return 1; }
        const int32_t code0 = atoi(argv[6]);
        const int32_t top_k = atoi(argv[7]);
        const float   top_p = strtof(argv[8], nullptr);
        const float   temp  = strtof(argv[9], nullptr);

        auto h = read_blob<float>(in_path);
        const int n_embd = clip_n_mmproj_embd(ctx);

        clip_image_f32 hidden_state;
        hidden_state.set_size({n_embd, 1}, false, true);
        hidden_state.cpy_buf(h);

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(hidden_state));

        std::vector<float>   out_embd(n_embd);
        std::vector<int32_t> out_codes;
        std::vector<float>   out_feats;
        bool is_eos = false;

        clip_encode_params p;
        p.imgs        = &batch;
        p.n_threads   = n_threads;
        p.gen_process = CLIP_GEN_PROCESS_GEN_CODE;
        p.out_embd    = &out_embd;
        p.out_codes   = &out_codes;
        p.out_feats   = &out_feats;
        p.code0       = code0;
        p.top_k       = top_k;
        p.top_p       = top_p;
        p.seed        = seed;
        p.temp        = temp;
        p.out_is_eos  = &is_eos;

        if (!clip_encode(ctx, &p)) {
            fprintf(stderr, "clip_encode failed (gen_code)\n");
            return 1;
        }

        put_u32(out, (uint32_t) out_codes.size());
        fwrite(out_codes.data(), 4, out_codes.size(), out);
        put_u32(out, (uint32_t) out_embd.size());
        fwrite(out_embd.data(), 4, out_embd.size(), out);
        put_u32(out, (uint32_t) out_feats.size());
        fwrite(out_feats.data(), 4, out_feats.size(), out);
        put_u32(out, is_eos ? 1u : 0u);
        put_u32(out, 0); // no audio in gen_code
        put_u32(out, 0); // no state in gen_code
        fprintf(stderr, "gen_code: %d codes, %d embd\n", (int) out_codes.size(), (int) out_embd.size());
    } else {
        // clip_encode's lazy warmup builds the graph from the *batch* with
        // default gen params (GEN_CODE, clip.cpp:4438-4439), and the GEN_WAV
        // placeholder batch ([1,1], mtmd.cpp:2000-2002) crashes that build.
        // The reference's own usage always runs GEN_CODE first
        // (mtmd-helper-gen.cpp:262 → :405), so warm up the same way: one
        // GEN_CODE call with a properly sized zero hidden state.
        {
            const int n_e = clip_n_mmproj_embd(ctx);
            clip_image_f32 warm;
            warm.set_size({n_e, 1}, false, true);
            warm.cpy_buf(std::vector<float>(n_e, 0.0f));
            clip_image_f32_batch wbatch;
            wbatch.is_audio = true;
            wbatch.entries.push_back(std::move(warm));
            std::vector<float>   w_embd(n_e);
            std::vector<int32_t> w_codes;
            std::vector<float>   w_feats;
            clip_encode_params wp;
            wp.imgs        = &wbatch;
            wp.n_threads   = n_threads;
            wp.gen_process = CLIP_GEN_PROCESS_GEN_CODE;
            wp.out_embd    = &w_embd;
            wp.out_codes   = &w_codes;
            wp.out_feats   = &w_feats;
            wp.seed        = seed;
            if (!clip_encode(ctx, &wp)) {
                fprintf(stderr, "warmup gen_code failed\n");
                return 1;
            }
        }
        // genwav: the input blob is codes (i32) or feats (f32); a trailing
        // state blob (read from the port's dump) rides after the payload —
        // its size is whatever the file holds beyond the payload
        std::vector<int32_t> codes;
        std::vector<float>   feats;
        std::vector<uint8_t> state;
        int64_t n_payload = 0;
        if (mode == "genwav-codes") {
            codes = read_blob<int32_t>(in_path);
        } else if (mode == "genwav-feats") {
            feats = read_blob<float>(in_path);
        } else {
            fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 1;
        }

        // gen_wav has no hidden-state input, the batch entry is a placeholder
        // (mtmd.cpp:1998-2006)
        clip_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));

        clip_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        std::vector<float>   out_audio;
        std::vector<uint8_t> out_state;

        clip_encode_params p;
        p.imgs        = &batch;
        p.n_threads   = n_threads;
        p.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
        p.seed        = seed;
        p.out_audio   = &out_audio;
        p.state_out   = &out_state;
        if (mode == "genwav-codes") {
            // the last 0-or-more bytes beyond a whole frame count are the
            // state — the caller packs [codes][state] and the state size is
            // discovered from the port's own dump via $TTS_STATE_SIZE
            const char * ss = getenv("TTS_STATE_SIZE");
            if (ss) {
                long sb = atol(ss);
                long nb_codes = (long) codes.size() * 4;
                long total = nb_codes + sb;
                // re-read raw to split
                FILE * fp = fopen(in_path, "rb");
                fseek(fp, 0, SEEK_END);
                long fsz = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                std::vector<uint8_t> raw(fsz);
                fread(raw.data(), 1, fsz, fp);
                fclose(fp);
                codes.resize((fsz - sb) / 4);
                memcpy(codes.data(), raw.data(), fsz - sb);
                state.resize(sb);
                memcpy(state.data(), raw.data() + fsz - sb, sb);
                (void) total; (void) nb_codes;
            }
            p.codes = &codes;
        } else {
            const char * ss = getenv("TTS_STATE_SIZE");
            if (ss) {
                long sb = atol(ss);
                FILE * fp = fopen(in_path, "rb");
                fseek(fp, 0, SEEK_END);
                long fsz = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                std::vector<uint8_t> raw(fsz);
                fread(raw.data(), 1, fsz, fp);
                fclose(fp);
                feats.resize((fsz - sb) / 4);
                memcpy(feats.data(), raw.data(), fsz - sb);
                state.resize(sb);
                memcpy(state.data(), raw.data() + fsz - sb, sb);
            }
            p.feats = &feats;
        }
        if (!state.empty()) {
            p.state_in = &state;
        }

        if (!clip_encode(ctx, &p)) {
            fprintf(stderr, "clip_encode failed (gen_wav)\n");
            return 1;
        }

        put_u32(out, 0); // no codes in gen_wav
        put_u32(out, 0); // no embd
        put_u32(out, 0); // no feats
        put_u32(out, 0); // no eos
        put_u32(out, (uint32_t) out_audio.size());
        fwrite(out_audio.data(), 4, out_audio.size(), out);
        put_u32(out, (uint32_t) out_state.size());
        if (!out_state.empty()) {
            fwrite(out_state.data(), 1, out_state.size(), out);
        }
        fprintf(stderr, "gen_wav: %d samples, %d state bytes\n", (int) out_audio.size(), (int) out_state.size());
    }

    clip_free(ctx);
    fclose(out);
    return 0;
}
