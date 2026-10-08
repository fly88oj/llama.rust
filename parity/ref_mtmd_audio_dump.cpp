// ref_mtmd_audio_dump.cpp — reference-side ground truth for the
// mtmd-audio.cpp port (agent: specialized chat parsers + audio mtmd).
//
// Compiles tools/mtmd/mtmd-audio.cpp from the PINNED tree directly into the
// probe (its statics become reachable), stubs clip_get_hparams so the
// preprocessors can be constructed without a clip_ctx, and runs every
// preprocessor on a raw f32 PCM fixture. Each output mel is dumped as:
//     i64 n_len, i64 n_len_org, i64 n_mel, n_mel*n_len f32 data
// followed by the streaming-ISTFT output:
//     i64 n_out, n_out f32 samples
//
// Build: see parity/gen_mtmd_audio_ref.sh. Consumed by
// crates/llama/tests/mtmd_audio_parity.rs.

#include "mtmd-audio.cpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static clip_hparams g_hp;
const clip_hparams * clip_get_hparams(const struct clip_ctx *) { return &g_hp; }

// clip.cpp owns this (clip-impl.h:814); stub it so the single-TU probe links
struct clip_logger_state g_logger_state;

static std::vector<float> read_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    std::vector<float> out(s.size() / sizeof(float));
    memcpy(out.data(), s.data(), out.size() * sizeof(float));
    return out;
}

static void write_i64(std::ofstream & o, int64_t v) { o.write((const char *) &v, sizeof(v)); }
static void write_f32(std::ofstream & o, float v)   { o.write((const char *) &v, sizeof(v)); }

static void dump_mels(std::ofstream & o, const std::vector<mtmd_audio_mel> & mels) {
    write_i64(o, (int64_t) mels.size());
    for (const auto & m : mels) {
        write_i64(o, m.n_len);
        write_i64(o, m.n_len_org);
        write_i64(o, m.n_mel);
        for (float v : m.data) write_f32(o, v);
    }
}

template <typename T>
static void run_preproc(std::ofstream & o, const char * name, T & proc, const std::vector<float> & samples) {
    proc.initialize();
    std::vector<mtmd_audio_mel> out;
    const bool ok = proc.preprocess(samples.data(), samples.size(), out);
    printf("%s: ok=%d chunks=%zu\n", name, (int) ok, out.size());
    write_i64(o, ok ? 1 : 0);
    if (ok) {
        dump_mels(o, out);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s SAMPLES_F32 OUT_BIN\n", argv[0]);
        return 1;
    }
    const std::vector<float> samples = read_f32(argv[1]);
    std::ofstream out(argv[2], std::ios::binary);

    // deterministic parakeet filterbank + window (mirrored in the Rust test)
    {
        const int n_mel = 64, n_fft = 512;
        const size_t n_fft_bins = n_fft / 2 + 1;
        g_hp.mel_filters.resize(n_mel * n_fft_bins);
        for (size_t i = 0; i < g_hp.mel_filters.size(); i++) {
            g_hp.mel_filters[i] = (float) (((i * 37) % 97) / 97.0);
        }
        g_hp.window.resize(400);
        for (size_t i = 0; i < g_hp.window.size(); i++) {
            g_hp.window[i] = 0.5f * (1.0f - cosf((float) (2.0 * M_PI * i / 400.0)));
        }
    }

    { // whisper
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 128; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 400; g_hp.audio_window_len = 400; g_hp.audio_hop_len = 160;
        mtmd_audio_preprocessor_whisper p(nullptr);
        run_preproc(out, "whisper", p, samples);
    }
    { // qwen3a
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 128; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 512; g_hp.audio_window_len = 400; g_hp.audio_hop_len = 160;
        mtmd_audio_preprocessor_qwen3a p(nullptr);
        run_preproc(out, "qwen3a", p, samples);
    }
    { // dots3note
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 128; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 400; g_hp.audio_window_len = 400; g_hp.audio_hop_len = 160;
        g_hp.audio_chunk_len = 60;
        mtmd_audio_preprocessor_dots3note p(nullptr);
        run_preproc(out, "dots3note", p, samples);
    }
    { // mimo_audio
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 40; g_hp.audio_sample_rate = 24000;
        g_hp.audio_n_fft = 960; g_hp.audio_window_len = 960; g_hp.audio_hop_len = 240;
        mtmd_audio_preprocessor_mimo_audio p(nullptr);
        run_preproc(out, "mimo_audio", p, samples);
    }
    { // qwen3tts_spk
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 80; g_hp.audio_sample_rate = 24000;
        g_hp.audio_n_fft = 1024; g_hp.audio_window_len = 1024; g_hp.audio_hop_len = 256;
        mtmd_audio_preprocessor_qwen3tts_spk p(nullptr);
        run_preproc(out, "qwen3tts_spk", p, samples);
    }
    { // conformer
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 80; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 512; g_hp.audio_window_len = 400; g_hp.audio_hop_len = 160;
        mtmd_audio_preprocessor_conformer p(nullptr);
        run_preproc(out, "conformer", p, samples);
    }
    { // granite_speech
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 80; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 512; g_hp.audio_window_len = 400; g_hp.audio_hop_len = 160;
        mtmd_audio_preprocessor_granite_speech p(nullptr);
        run_preproc(out, "granite_speech", p, samples);
    }
    { // gemma4a
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 128; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = 1024; g_hp.audio_window_len = 640; g_hp.audio_hop_len = 320;
        mtmd_audio_preprocessor_gemma4a p(nullptr);
        run_preproc(out, "gemma4a", p, samples);
    }
    { // gemma4ua
        g_hp = clip_hparams{};
        g_hp.n_mel_bins = 640; g_hp.audio_sample_rate = 16000;
        g_hp.audio_n_fft = -1; g_hp.audio_window_len = -1; g_hp.audio_hop_len = -1;
        mtmd_audio_preprocessor_gemma4ua p(nullptr);
        run_preproc(out, "gemma4ua", p, samples);
    }
    { // parakeet (uses the deterministic mel_filters/window set above)
        clip_hparams hp = clip_hparams{};
        hp.n_mel_bins = 64; hp.audio_sample_rate = 16000;
        hp.audio_n_fft = 512; hp.audio_window_len = 400; hp.audio_hop_len = 160;
        const size_t n_fft_bins = 512 / 2 + 1;
        hp.mel_filters.resize(64 * n_fft_bins);
        for (size_t i = 0; i < hp.mel_filters.size(); i++) {
            hp.mel_filters[i] = (float) (((i * 37) % 97) / 97.0);
        }
        hp.window.resize(400);
        for (size_t i = 0; i < hp.window.size(); i++) {
            hp.window[i] = 0.5f * (1.0f - cosf((float) (2.0 * M_PI * i / 400.0)));
        }
        g_hp = hp;
        mtmd_audio_preprocessor_parakeet p(nullptr);
        run_preproc(out, "parakeet", p, samples);
    }
    { // pockettts
        g_hp = clip_hparams{};
        g_hp.audio_sample_rate = 24000; g_hp.mimi_downsample = 8;
        mtmd_audio_preprocessor_pockettts p(nullptr);
        p.initialize();
        std::vector<mtmd_audio_mel> o;
        const bool ok = p.preprocess(samples.data(), samples.size(), o);
        printf("pockettts: ok=%d chunks=%zu\n", (int) ok, o.size());
        write_i64(out, ok ? 1 : 0);
        if (ok) dump_mels(out, o);
    }

    // streaming ISTFT: deterministic magnitude/phase spectra
    {
        mtmd_audio_streaming_istft istft(256, 64);
        std::vector<float> total;
        for (int frame = 0; frame < 16; frame++) {
            std::vector<float> spec((256 / 2 + 1) * 2, 0.0f);
            for (int j = 0; j < 256 / 2 + 1; j++) {
                const double mag  = 0.5 * (1.0 + sin(0.37 * j + 0.11 * frame));
                const double ph   = 0.9 * j - 1.3 * frame + 0.5 * sin(frame * 0.7);
                spec[2 * j + 0]   = (float) (mag * cos(ph));
                spec[2 * j + 1]   = (float) (mag * sin(ph));
            }
            auto chunk = istft.process_frame(spec.data());
            total.insert(total.end(), chunk.begin(), chunk.end());
        }
        auto tail = istft.flush();
        total.insert(total.end(), tail.begin(), tail.end());
        printf("istft: %zu samples\n", total.size());
        write_i64(out, (int64_t) total.size());
        for (float v : total) write_f32(out, v);
    }

    out.close();
    return 0;
}
