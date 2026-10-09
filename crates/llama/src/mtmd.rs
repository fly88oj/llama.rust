//! mtmd — port of `tools/mtmd/mtmd.cpp` (vision half) and
//! `tools/mtmd/mtmd-helper.cpp` (the decode glue), plus the bitmap loaders the
//! reference gets from `stb_image` (mtmd-helper.cpp:36).
//!
//! Line numbers refer to the pinned worktree @ bd4f514db1. The piece that makes
//! this usable end to end is [`MtmdEval`]: it owns the text decoder and feeds
//! the vision embeddings into it. The reference does that through
//! `llama_batch.embd` (mtmd-helper-common.h:82 `decode_embd_batch`); see the
//! [`MtmdEval`] docs for how the port reproduces it without that field.
//!
//! Scope: the text+image prompt path of `qwen3vl_merger` mmproj files — media
//! markers, chunking, M-RoPE decoder positions, image decode (PNG/BMP/PPM) and
//! the per-chunk eval. NOT ported: audio, video, llava-uhd slicing, lazy
//! bitmaps, the `libmtmd` C ABI (`mtmd.h`), and the model-specific chat glue of
//! `mtmd-helper.cpp` (`mtmd_helper_gen_*`).

use crate::clip::{
    self, ClipContext, ClipContextParams, ClipFlashAttn, ClipImageF32, ClipImageF32Batch,
    ClipImageU8,
};
use crate::context::{DecodeContext, ForwardWeights};
use crate::mtmd_audio::{AudioMel, WhisperPreproc};
use crate::vocab::Vocab;
use ggml::{Context, GgmlType, Graph, TensorId};

/// mtmd.h:97 `mtmd_context_params` (the fields this port honours)
#[derive(Clone)]
pub struct MtmdContextParams {
    pub n_threads: usize,
    pub flash_attn_type: ClipFlashAttn,
    /// mtmd.h — the string that separates text from media in the prompt
    pub media_marker: String,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        Self {
            n_threads: 4,
            flash_attn_type: ClipFlashAttn::Auto,
            media_marker: mtmd_default_marker(),
            image_min_tokens: -1,
            image_max_tokens: -1,
        }
    }
}

/// mtmd.cpp:447 `mtmd_default_marker`
pub fn mtmd_default_marker() -> String {
    "<__media__>".to_string()
}

/// mtmd.cpp:200 `mtmd_pos_type` — how the decoder positions of an image chunk
/// are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdPosType {
    /// one position per token
    Normal,
    /// qwen-vl M-RoPE: (t, y, x, z) per token
    Mrope,
}

/// mtmd.cpp:210 `mtmd_image_tokens`
#[derive(Clone)]
pub struct MtmdImageTokens {
    pub nx: u32,
    pub ny: u32,
    pub pos: MtmdPosType,
    pub batch_f32: ClipImageF32Batch,
}

impl MtmdImageTokens {
    /// mtmd.cpp:221 `n_tokens()`
    pub fn n_tokens(&self) -> usize {
        (self.nx * self.ny) as usize
    }

    /// mtmd.cpp:2525 `mtmd_image_tokens_get_n_pos`
    pub fn n_pos(&self) -> i32 {
        match self.pos {
            MtmdPosType::Mrope => self.nx.max(self.ny) as i32,
            MtmdPosType::Normal => self.n_tokens() as i32,
        }
    }

    /// mtmd.cpp:2466 `mtmd_image_tokens_get_decoder_pos`
    pub fn decoder_pos(&self, pos_0: i32, i: usize) -> [i32; 4] {
        match self.pos {
            MtmdPosType::Mrope => [
                pos_0,
                pos_0 + (i as u32 % self.nx) as i32,
                pos_0 + (i as u32 / self.nx) as i32,
                0,
            ],
            MtmdPosType::Normal => [pos_0 + i as i32, 0, 0, 0],
        }
    }
}

/// mtmd.cpp:323 `mtmd_input_chunk`
#[derive(Clone)]
pub enum MtmdChunk {
    Text(Vec<i32>),
    Image(MtmdImageTokens),
    /// MTMD_INPUT_CHUNK_TYPE_AUDIO (mtmd.cpp:334)
    Audio(MtmdAudioTokens),
}

impl MtmdChunk {
    /// mtmd.cpp:2341 `mtmd_input_chunk_get_n_tokens`
    pub fn n_tokens(&self) -> usize {
        match self {
            MtmdChunk::Text(t) => t.len(),
            MtmdChunk::Image(img) => img.n_tokens(),
            MtmdChunk::Audio(a) => a.n_tokens as usize,
        }
    }

    /// mtmd.cpp:2351 `mtmd_input_chunk_get_n_pos`
    pub fn n_pos(&self) -> i32 {
        match self {
            MtmdChunk::Text(t) => t.len() as i32,
            MtmdChunk::Image(img) => img.n_pos(),
            // audio positions count like a 1-D image grid (mtmd.cpp:341-345)
            MtmdChunk::Audio(a) => a.n_tokens as i32,
        }
    }
}

/// `mtmd_context` (mtmd.cpp:479), vision half.
/// `mtmd_audio_tokens` (mtmd.cpp:288-303): one mel chunk of an audio input
#[derive(Clone)]
pub struct MtmdAudioTokens {
    pub n_tokens: u32,
    pub batch_f32: ClipImageF32Batch,
}

/// one media part of a tokenized prompt (`mtmd_input_part`, mtmd.h:169) —
/// what `mtmd_tokenize_from_parts` interleaves between the text segments
pub enum MtmdMedia<'a> {
    Image(&'a ClipImageU8),
    /// raw audio bytes (WAV buffer)
    Audio(&'a [u8]),
}

/// `mtmd_input_chunk` of type AUDIO (mtmd.cpp:409-411)
pub struct MtmdAudioChunk {
    pub tokens: MtmdAudioTokens,
}

/// `mtmd_audio_preprocessor` selection (mtmd.cpp:920-1010). Every arch this
/// port loads selects its literal preprocessor here (mtmd.cpp:933-1007).
enum AudioPreproc {
    None,
    /// the whisper-enc family (mtmd.cpp:937-963)
    Whisper(WhisperPreproc),
    /// qwen3a (mtmd.cpp:943-947)
    Qwen3a(crate::mtmd_audio::Qwen3aPreproc),
    /// lfm2a conformer (mtmd.cpp:967-971)
    Conformer(crate::mtmd_audio::ConformerPreproc),
    /// d1omni_a conformer + 30 s cut / 0.5 s pad (mtmd.cpp:990-993)
    D1omni(crate::mtmd_audio::D1omniPreproc),
    /// gemma4ua — raw waveform frames (mtmd.cpp:985-990)
    Gemma4ua(crate::mtmd_audio::Gemma4uaPreproc),
    /// granite_speech (mtmd.cpp:977-980)
    GraniteSpeech(crate::mtmd_audio::GraniteSpeechPreproc),
    /// gemma4a (mtmd.cpp:982-985)
    Gemma4a(crate::mtmd_audio::Gemma4aPreproc),
    /// parakeet (mtmd.cpp:1008-1011)
    Parakeet(crate::mtmd_audio::ParakeetPreproc),
    /// mimo_audio (mtmd.cpp:996-1000)
    MimoAudio(crate::mtmd_audio::MimoAudioPreproc),
    /// qwen3tts speaker encoder (mtmd.cpp:1002-1005)
    Qwen3TtsSpk(crate::mtmd_audio::Qwen3TtsSpkPreproc),
    /// pocket-tts speaker encoder (mtmd.cpp:1006-1009)
    PocketTts(crate::mtmd_audio::PocketTtsPreproc),
}

impl AudioPreproc {
    fn sample_rate(&self) -> i32 {
        match self {
            AudioPreproc::None => -1,
            AudioPreproc::Whisper(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Qwen3a(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Conformer(p) => p.hparams().audio_sample_rate,
            AudioPreproc::D1omni(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Gemma4ua(p) => p.hparams().audio_sample_rate,
            AudioPreproc::GraniteSpeech(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Gemma4a(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Parakeet(p) => p.hparams().audio_sample_rate,
            AudioPreproc::MimoAudio(p) => p.hparams().audio_sample_rate,
            AudioPreproc::Qwen3TtsSpk(p) => p.hparams().audio_sample_rate,
            AudioPreproc::PocketTts(p) => p.hparams().audio_sample_rate,
        }
    }

    fn initialize(&mut self) {
        match self {
            AudioPreproc::None => {}
            AudioPreproc::Whisper(p) => p.initialize(),
            AudioPreproc::Qwen3a(p) => p.initialize(),
            AudioPreproc::Conformer(p) => p.initialize(),
            AudioPreproc::D1omni(p) => p.initialize(),
            AudioPreproc::Gemma4ua(p) => p.initialize(),
            AudioPreproc::GraniteSpeech(p) => p.initialize(),
            AudioPreproc::Gemma4a(p) => p.initialize(),
            AudioPreproc::Parakeet(p) => p.initialize(),
            AudioPreproc::MimoAudio(p) => p.initialize(),
            AudioPreproc::Qwen3TtsSpk(p) => p.initialize(),
            AudioPreproc::PocketTts(p) => p.initialize(),
        }
    }

    fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        match self {
            AudioPreproc::None => unreachable!("no audio preprocessor"),
            AudioPreproc::Whisper(p) => p.preprocess(samples, output),
            AudioPreproc::Qwen3a(p) => p.preprocess(samples, output),
            AudioPreproc::Conformer(p) => p.preprocess(samples, output),
            AudioPreproc::D1omni(p) => p.preprocess(samples, output),
            AudioPreproc::Gemma4ua(p) => p.preprocess(samples, output),
            AudioPreproc::GraniteSpeech(p) => p.preprocess(samples, output),
            AudioPreproc::Gemma4a(p) => p.preprocess(samples, output),
            AudioPreproc::Parakeet(p) => p.preprocess(samples, output),
            AudioPreproc::MimoAudio(p) => p.preprocess(samples, output),
            AudioPreproc::Qwen3TtsSpk(p) => p.preprocess(samples, output),
            AudioPreproc::PocketTts(p) => p.preprocess(samples, output),
        }
    }
}

pub struct MtmdContext {
    pub clip: ClipContext,
    pub n_threads: usize,
    pub media_marker: String,
    /// the token strings wrapping an image (mtmd.cpp:696: qwen-vl uses
    /// `<|vision_start|>` / `<|vision_end|>`)
    pub img_beg: String,
    pub img_end: String,
    pub pos_type: MtmdPosType,
    /// `n_embd_text` — -1 when no text model was provided
    pub n_embd_text: i32,
    /// `aud_beg`/`aud_end` (mtmd.cpp:499) and `audio_preproc`
    /// (mtmd.cpp:522 / init_audio, mtmd.cpp:920) for audio projectors
    pub aud_beg: String,
    pub aud_end: String,
    audio_preproc: AudioPreproc,
    /// mtmd.cpp:513/515 — the token after each slice / after the overview
    /// (llava-uhd tiling; cohere2v uses <|IMG_LINE_BREAK|> for both,
    /// mtmd.cpp:805-806). Empty == no marker; a missing vocab piece yields
    /// the C's LLAMA_TOKEN_NULL (-1) sentinel, like `lookup_token`.
    pub tok_sli_img_end: Vec<i32>,
    pub tok_ov_img_end: Vec<i32>,
    out_embd: Vec<f32>,
    /// `ctx_gen_a` (mtmd.cpp:485) — the gen-audio context of a MIXED
    /// audio+gen mmproj. The primary `clip` stays the audio (speaker
    /// encoder) context like the reference's ctx_a; the GEN calls route
    /// here (mtmd.cpp:1874/:1922+ read ctx_gen_a). `None` = the primary is
    /// already the gen context (single-modality file).
    clip_gen: Option<clip::ClipContext>,
}

impl MtmdContext {
    /// mtmd.cpp:1085 `mtmd_init_from_file`. `text_rope_type` is the decoder's
    /// rope type (`llama_model_rope_type`), which decides the position layout.
    pub fn init_from_file(
        mmproj_fname: &str,
        text_model: Option<(&Vocab, i32, MtmdPosType)>,
        params: &MtmdContextParams,
    ) -> Result<Self, String> {
        if params.media_marker.is_empty() {
            return Err("media_marker must not be empty".into());
        }
        let clip_params = ClipContextParams {
            flash_attn_type: params.flash_attn_type,
            n_threads: params.n_threads,
            image_min_tokens: params.image_min_tokens,
            image_max_tokens: params.image_max_tokens,
        };
        let clip = clip::clip_init_from_file(mmproj_fname, &clip_params)?;

        // mtmd.cpp:584-585 + clip.cpp:4004-4009: a mixed audio+gen mmproj
        // carries BOTH contexts — the primary stays the audio ctx, the GEN
        // half gets its own (the reference's ctx_gen_a)
        let clip_gen = if clip.model.modality != clip::ClipModality::GenAudio {
            match ggml::Gguf::open(mmproj_fname) {
                Ok(g) if g.get_bool("clip.has_gen_audio_encoder").unwrap_or(false) => {
                    Some(clip::clip_init_modality(
                        mmproj_fname,
                        &clip_params,
                        clip::ClipModality::GenAudio,
                    )?)
                }
                _ => None,
            }
        } else {
            None
        };

        // mtmd.cpp:696 — the vision markers of this projector family
        let (img_beg, img_end, aud_beg, aud_end, audio_preproc);
        // mtmd.cpp:508-517 — the llava-uhd tiling markers. Only the two
        // cohere2v sets are populated (mtmd.cpp:801-810): every tile and the
        // thumbnail end with <|IMG_LINE_BREAK|>. The remaining marker sets
        // (ov/slices/slice start, mid-row, row end) stay empty like the C's
        // defaults for this projector.
        let mut tok_sli_img_end: Vec<i32> = Vec::new();
        let mut tok_ov_img_end: Vec<i32> = Vec::new();
        match clip.model.proj_type {
            clip::ProjectorType::Qwen2Vl
            | clip::ProjectorType::Qwen25Vl
            | clip::ProjectorType::Qwen3Vl
            | clip::ProjectorType::Ling3Vl => {
                // mtmd.cpp:707-709: no audio markers for the vision-only path
                img_beg = "<|vision_start|>".to_string();
                img_end = "<|vision_end|>".to_string();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::None;
                tok_sli_img_end = Vec::new();
                tok_ov_img_end = Vec::new();
            }
            // mtmd.cpp:801-810 — <|START_OF_IMG|> (tile embeddings)
            // <|IMG_LINE_BREAK|> ... <|END_OF_IMG|>; slices first, then the
            // thumbnail (ov_img_first = false)
            clip::ProjectorType::Cohere2V => {
                img_beg = "<|START_OF_IMG|>".to_string();
                img_end = "<|END_OF_IMG|>".to_string();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::None;
                let lookup = |text: &str| -> Vec<i32> {
                    lookup_token(text_model.map(|(v, _, _)| v), text)
                };
                tok_sli_img_end = lookup("<|IMG_LINE_BREAK|>");
                tok_ov_img_end = tok_sli_img_end.clone();
            }
            // mtmd.cpp:623 `init_audio` + :933-963 — every whisper-enc family
            // projector uses the whisper mel preprocessor; only qwen2a,
            // voxtral and musicflamingo wrap the chunks in marker tokens
            // tts generators: GEN_WAV decoders, never audio-input encoders —
            // no markers, no preprocessor (mtmd.cpp's gen path is separate)
            clip::ProjectorType::Qwen3TtsGen | clip::ProjectorType::PocketTtsGen => {
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::None;
            }
            clip::ProjectorType::Qwen2A => {
                // mtmd.cpp:937-941 (qwen2a/qwen2.5o):
                // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = WhisperPreproc::new(clip.audio_hparams());
                preproc.initialize(); // NOT thread-safe (mtmd-audio.h:59)
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|audio_bos|>".to_string();
                aud_end = "<|audio_eos|>".to_string();
                audio_preproc = AudioPreproc::Whisper(preproc);
            }
            clip::ProjectorType::Voxtral => {
                // mtmd.cpp:948-952: [BEGIN_AUDIO] ... (embeddings) ...
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = WhisperPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "[BEGIN_AUDIO]".to_string();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Whisper(preproc);
            }
            clip::ProjectorType::MusicFlamingo => {
                // mtmd.cpp:955-958: <sound> ... (embeddings) ...
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = WhisperPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<sound>".to_string();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Whisper(preproc);
            }
            // mtmd.cpp:960-963 — ultravox/glma/meralion emit bare embeddings
            clip::ProjectorType::Ultravox
            | clip::ProjectorType::Glma
            | clip::ProjectorType::Meralion => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = WhisperPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Whisper(preproc);
            }
            // mtmd.cpp:943-947 — <|audio_start|> ... <|audio_end|>
            clip::ProjectorType::Qwen3A => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::Qwen3aPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|audio_start|>".to_string();
                aud_end = "<|audio_end|>".to_string();
                audio_preproc = AudioPreproc::Qwen3a(preproc);
            }
            // mtmd.cpp:967-971 — lfm2a runs the conformer preprocessor with
            // bare embeddings (no marker tokens)
            clip::ProjectorType::Lfm2A => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::ConformerPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Conformer(preproc);
            }
            // mtmd.cpp:990-993 — d1omni_a: the same conformer preprocessing
            // (30 s cut / 0.5 s pad, mtmd-audio.cpp:998) with bare embeddings
            clip::ProjectorType::D1OmniA => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::D1omniPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::D1omni(preproc);
            }
            // mtmd.cpp:985-990 — <|audio> ... <audio|>, raw-waveform embedder
            clip::ProjectorType::Gemma4UA => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::Gemma4uaPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|audio>".to_string();
                aud_end = "<audio|>".to_string();
                audio_preproc = AudioPreproc::Gemma4ua(preproc);
            }
            // mtmd.cpp:985-990 — <|audio> ... <audio|>, raw-waveform embedder
            clip::ProjectorType::Gemma4UA => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::Gemma4uaPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|audio>".to_string();
                aud_end = "<audio|>".to_string();
                audio_preproc = AudioPreproc::Gemma4ua(preproc);
            }
            // mtmd.cpp:977-980 — granite_speech: bare embeddings
            clip::ProjectorType::GraniteSpeech => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc =
                    crate::mtmd_audio::GraniteSpeechPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::GraniteSpeech(preproc);
            }
            // mtmd.cpp:982-985 — gemma4a: <|audio> ... <audio|>
            clip::ProjectorType::Gemma4A => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::Gemma4aPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|audio>".to_string();
                aud_end = "<audio|>".to_string();
                audio_preproc = AudioPreproc::Gemma4a(preproc);
            }
            // mtmd.cpp:1008-1011 — parakeet: bare embeddings
            clip::ProjectorType::Parakeet => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::ParakeetPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Parakeet(preproc);
            }
            // mtmd.cpp:996-1000 — mimo_audio: <|mimo_audio_start|> ...
            clip::ProjectorType::MimoAudio => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::MimoAudioPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = "<|mimo_audio_start|>".to_string();
                aud_end = "<|mimo_audio_end|>".to_string();
                audio_preproc = AudioPreproc::MimoAudio(preproc);
            }
            // mtmd.cpp:1002-1005 — qwen3tts_spkenc: bare embeddings
            clip::ProjectorType::Qwen3TtsSpkEnc => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::Qwen3TtsSpkPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::Qwen3TtsSpk(preproc);
            }
            // mtmd.cpp:1006-1009 — pockettts_spkenc: bare embeddings
            clip::ProjectorType::PocketTtsSpkEnc => {
                eprintln!(
                    "mtmd_context_init: audio input is in experimental stage and may have reduced quality:\n    https://github.com/ggml-org/llama.cpp/discussions/13759"
                );
                let mut preproc = crate::mtmd_audio::PocketTtsPreproc::new(clip.audio_hparams());
                preproc.initialize();
                img_beg = String::new();
                img_end = String::new();
                aud_beg = String::new();
                aud_end = String::new();
                audio_preproc = AudioPreproc::PocketTts(preproc);
            }
            clip::ProjectorType::Unknown => {
                return Err("projector unknown is not ported".to_string())
            }
        }

        let (n_embd_text, pos_type) = match text_model {
            Some((_vocab, n_embd, pos)) => (n_embd, pos),
            None => (-1, MtmdPosType::Normal),
        };
        if n_embd_text > 0 && n_embd_text != clip.n_mmproj_embd() {
            return Err(format!(
                "mismatch between text model n_embd ({n_embd_text}) and mmproj ({})",
                clip.n_mmproj_embd()
            ));
        }

        Ok(Self {
            clip,
            n_threads: params.n_threads,
            media_marker: params.media_marker.clone(),
            img_beg,
            img_end,
            pos_type,
            n_embd_text,
            aud_beg,
            aud_end,
            audio_preproc,
            tok_sli_img_end,
            tok_ov_img_end,
            out_embd: Vec::new(),
            clip_gen,
        })
    }

    /// `mtmd_get_audio_sample_rate` (mtmd.h:362): -1 when the projector has
    /// no audio encoder
    pub fn audio_sample_rate(&self) -> i32 {
        self.audio_preproc.sample_rate()
    }

    /// `mtmd_tokenize` for audio (mtmd.cpp:1556-1651): decode the PCM,
    /// preprocess to mel chunks, wrap each chunk in the audio begin/end
    /// markers. `audio_bytes` is a WAV buffer.
    pub fn tokenize_audio(
        &mut self,
        vocab: &Vocab,
        audio_bytes: &[u8],
        add_special: bool,
    ) -> Result<Vec<MtmdChunk>, String> {
        let mut chunks: Vec<MtmdChunk> = Vec::new();
        self.add_audio(vocab, audio_bytes, &mut chunks)?;

        // mtmd.cpp:1290 — BOS/EOS handling of the whole chunk list
        let add_text = |chunks: &mut Vec<MtmdChunk>, toks: Vec<i32>| {
            if toks.is_empty() {
                return;
            }
            match chunks.last_mut() {
                Some(MtmdChunk::Text(t)) => t.extend(toks),
                _ => chunks.push(MtmdChunk::Text(toks)),
            }
        };
        if add_special && vocab.get_add_bos() {
            let bos = vocab.token_bos();
            match chunks.first_mut() {
                Some(MtmdChunk::Text(t)) => t.insert(0, bos),
                _ => chunks.insert(0, MtmdChunk::Text(vec![bos])),
            }
        }
        if add_special && vocab.get_add_eos() {
            add_text(&mut chunks, vec![vocab.token_eos()]);
        }
        Ok(chunks)
    }

    /// mtmd.cpp:1560 `mtmd_tokenizer::add_audio` — decode the PCM, preprocess
    /// to mel chunks and append them (with the arch's markers) to `chunks`.
    fn add_audio(
        &mut self,
        vocab: &Vocab,
        audio_bytes: &[u8],
        chunks: &mut Vec<MtmdChunk>,
    ) -> Result<(), String> {
        let sample_rate = self.audio_sample_rate();
        if sample_rate < 0 {
            return Err("This model does not support audio input".into());
        }
        // audio_helpers::decode_audio_from_buf (mtmd-helper.cpp:397)
        let samples = audio_from_wav_bytes(audio_bytes, sample_rate)
            .map_err(|e| format!("Unable to read WAV audio file from buffer: {e}"))?;
        if samples.is_empty() {
            return Err("error: empty audio data".into());
        }

        if !self.aud_beg.is_empty() {
            push_text(chunks, vocab.tokenize(&self.aud_beg.clone(), false, true));
        }

        // preprocess audio (mtmd.cpp:1597-1607)
        let mut mel_spec_chunks: Vec<AudioMel> = Vec::new();
        let ok = self
            .audio_preproc
            .preprocess(&samples, &mut mel_spec_chunks);
        if !ok {
            return Err("Unable to preprocess audio".into());
        }

        // consider each mel_spec as a separate audio chunk (mtmd.cpp:1610)
        for mel_spec in mel_spec_chunks {
            let mel_f32 = ClipImageF32::from_audio_mel(
                mel_spec.n_len as i32,
                mel_spec.n_mel as i32,
                &mel_spec.data,
            );
            let n_tokens = self.clip.n_output_tokens(&mel_f32);

            chunks.push(MtmdChunk::Audio(MtmdAudioTokens {
                n_tokens: n_tokens.max(0) as u32,
                batch_f32: ClipImageF32Batch {
                    entries: vec![mel_f32],
                    is_audio: true,
                },
            }));
        }

        if !self.aud_end.is_empty() {
            push_text(chunks, vocab.tokenize(&self.aud_end.clone(), false, true));
        }
        Ok(())
    }

    pub fn n_mmproj_embd(&self) -> i32 {
        self.clip.n_mmproj_embd()
    }

    pub fn support_vision(&self) -> bool {
        true
    }

    /// mtmd.cpp:1736 `mtmd_tokenize` / :1275 `mtmd_tokenizer::tokenize`.
    ///
    /// `text` is split on the media marker; every image is preprocessed and
    /// becomes an image chunk wrapped in the projector's begin/end tokens
    /// (mtmd.cpp:1345-1552). `add_special` mirrors `mtmd_input_text.add_special`.
    pub fn tokenize(
        &mut self,
        vocab: &Vocab,
        text: &str,
        bitmaps: &[ClipImageU8],
        add_special: bool,
    ) -> Result<Vec<MtmdChunk>, String> {
        // split on the marker, then interleave (mtmd.cpp:1313)
        let mut segments: Vec<&str> = Vec::new();
        let mut rest = text;
        loop {
            let Some(pos) = rest.find(&self.media_marker) else {
                segments.push(rest);
                break;
            };
            let (head, tail) = rest.split_at(pos);
            segments.push(head);
            rest = &tail[self.media_marker.len()..];
        }
        if segments.len() != bitmaps.len() + 1 {
            return Err("number of media markers does not match the number of loaded media".into());
        }
        let media: Vec<MtmdMedia> = bitmaps.iter().map(MtmdMedia::Image).collect();
        self.tokenize_parts(vocab, &segments, &media, add_special)
    }

    /// mtmd.cpp:305 `mtmd_tokenize_from_parts` — `segments.len()` text pieces
    /// with one media part between each consecutive pair (the API the
    /// reference CLI uses, mtmd-cli.cpp:269-330).
    pub fn tokenize_parts(
        &mut self,
        vocab: &Vocab,
        segments: &[&str],
        media: &[MtmdMedia<'_>],
        add_special: bool,
    ) -> Result<Vec<MtmdChunk>, String> {
        if segments.len() != media.len() + 1 {
            return Err(format!(
                "number of media markers ({}) does not match number of loaded media ({})",
                media.len(),
                segments.len() - 1
            ));
        }
        let mut chunks: Vec<MtmdChunk> = Vec::new();
        let add_text = |chunks: &mut Vec<MtmdChunk>, toks: Vec<i32>| {
            if toks.is_empty() {
                return;
            }
            // mtmd.cpp:1330 `add_text(tokens)`: extend the trailing text chunk
            match chunks.last_mut() {
                Some(MtmdChunk::Text(t)) => t.extend(toks),
                _ => chunks.push(MtmdChunk::Text(toks)),
            }
        };

        // interleave text and media parts (mtmd-cli.cpp:289-300 builds the
        // same part list before handing it to mtmd_tokenize_from_parts)
        for (i, seg) in segments.iter().enumerate() {
            add_text(&mut chunks, vocab.tokenize(seg, false, true));
            match media.get(i) {
                Some(MtmdMedia::Image(bmp)) => self.add_media(vocab, bmp, &mut chunks)?,
                Some(MtmdMedia::Audio(bytes)) => self.add_audio(vocab, bytes, &mut chunks)?,
                None => {}
            }
        }

        // mtmd.cpp:1290 — BOS/EOS handling of the whole chunk list
        if add_special && vocab.get_add_bos() {
            let bos = vocab.token_bos();
            match chunks.first_mut() {
                Some(MtmdChunk::Text(t)) => t.insert(0, bos),
                _ => chunks.insert(0, MtmdChunk::Text(vec![bos])),
            }
        }
        if add_special && vocab.get_add_eos() {
            add_text(&mut chunks, vec![vocab.token_eos()]);
        }
        Ok(chunks)
    }

    /// mtmd.cpp:1338 `mtmd_tokenizer::add_media`
    fn add_media(
        &mut self,
        vocab: &Vocab,
        bmp: &ClipImageU8,
        chunks: &mut Vec<MtmdChunk>,
    ) -> Result<(), String> {
        // the begin marker first (mtmd.cpp:1349)
        let beg = vocab.tokenize(&self.img_beg.clone(), false, true);
        push_text(chunks, beg);

        if bmp.nx <= 0 || bmp.ny <= 0 {
            return Err(format!(
                "invalid bitmap dimensions: nx = {}, ny = {}",
                bmp.nx, bmp.ny
            ));
        }
        let preproc = self.clip.image_preprocess_full(bmp)?;

        // mtmd.cpp:1478 — handle llava-uhd style preprocessing (output either
        // a grid, or overview-only)
        let has_tiling_grid = (preproc.grid_x > 0 && preproc.grid_y > 0) || preproc.has_overview();
        if has_tiling_grid {
            // mtmd.cpp:1482: no "frame merging" for llava-uhd style — one
            // bitmap per call, and each tile is its own image chunk
            // (split_batch_to_chunk, mtmd.cpp:1676-1716)
            let n_col = preproc.grid_x;
            let n_row = preproc.grid_y;
            let overview = preproc.overview.clone().ok_or(
                "split_batch_to_chunk: invalid overview image for llava-uhd style preprocessing",
            )?;

            // the tile chunks, row-major (chunks[y * n_col + x], mtmd.cpp:1507)
            if !preproc.entries.is_empty() {
                assert_eq!(preproc.entries.len() as i32, n_row * n_col);
                for y in 0..n_row {
                    for x in 0..n_col {
                        let e = &preproc.entries[(y * n_col + x) as usize];
                        let n_tokens = self.clip.n_output_tokens(e);
                        assert!(n_tokens > 0);
                        chunks.push(MtmdChunk::Image(MtmdImageTokens {
                            nx: n_tokens as u32,
                            ny: 1,
                            pos: self.pos_type,
                            batch_f32: ClipImageF32Batch {
                                entries: vec![e.clone()],
                                is_audio: false,
                            },
                        }));
                        // mtmd.cpp:1513 — the marker after every slice
                        // (tok_sli_img_mid / tok_row_end are empty for cohere2v)
                        push_text(chunks, self.tok_sli_img_end.clone());
                    }
                }
            }

            // mtmd.cpp:1530-1534 — the overview last (ov_img_first == false)
            let n_tokens = self.clip.n_output_tokens(&overview);
            assert!(n_tokens > 0);
            chunks.push(MtmdChunk::Image(MtmdImageTokens {
                nx: n_tokens as u32,
                ny: 1,
                pos: self.pos_type,
                batch_f32: ClipImageF32Batch {
                    entries: vec![overview],
                    is_audio: false,
                },
            }));
            push_text(chunks, self.tok_ov_img_end.clone());

            // the end marker (mtmd.cpp:1567)
            let end = vocab.tokenize(&self.img_end.clone(), false, true);
            push_text(chunks, end);
            return Ok(());
        }

        if preproc.entries.is_empty() {
            // mtmd.cpp:1543: no image tokens produced by preprocessor
            return Err("no image tokens produced by preprocessor".into());
        }
        let entries = preproc.entries;
        let n_tokens = self.clip.n_output_tokens(&entries[0]);
        let (nx, ny) = if self.pos_type == MtmdPosType::Mrope {
            // mtmd.cpp:1507 — M-RoPE needs the grid, others only the count
            (
                self.clip.n_output_tokens_x(&entries[0]) as u32,
                self.clip.n_output_tokens_y(&entries[0]) as u32,
            )
        } else {
            (n_tokens as u32, 1)
        };
        chunks.push(MtmdChunk::Image(MtmdImageTokens {
            nx,
            ny,
            pos: self.pos_type,
            batch_f32: ClipImageF32Batch {
                entries,
                is_audio: false,
            },
        }));

        // the end marker (mtmd.cpp:1547)
        let end = vocab.tokenize(&self.img_end.clone(), false, true);
        push_text(chunks, end);
        Ok(())
    }

    /// mtmd.cpp:1799 `mtmd_encode_chunk` → :1774 `mtmd_encode_impl`: the
    /// embeddings of an image chunk, `[n_mmproj_embd * n_tokens]`. Audio
    /// chunks take the same `clip_image_batch_encode` path through the audio
    /// encoder (mtmd.cpp:1822-1832).
    pub fn encode_chunk(&mut self, chunk: &MtmdChunk) -> Result<&[f32], String> {
        match chunk {
            MtmdChunk::Image(img) => {
                let embd = self.clip.image_batch_encode(&img.batch_f32)?;
                self.out_embd = embd;
                Ok(&self.out_embd)
            }
            MtmdChunk::Audio(audio) => {
                let embd = self.clip.audio_batch_encode(&audio.batch_f32)?;
                self.out_embd = embd;
                Ok(&self.out_embd)
            }
            MtmdChunk::Text(_) => Err("mtmd_encode_chunk has no effect for text chunks".into()),
        }
    }

    /// mtmd.cpp:1863 `mtmd_get_output_embd`
    pub fn get_output_embd(&self) -> &[f32] {
        &self.out_embd
    }

    /// mtmd.cpp:2188 `mtmd_decode_use_mrope`
    pub fn decode_use_mrope(&self) -> bool {
        self.pos_type == MtmdPosType::Mrope
    }
}

fn push_text(chunks: &mut Vec<MtmdChunk>, toks: Vec<i32>) {
    if toks.is_empty() {
        return;
    }
    match chunks.last_mut() {
        Some(MtmdChunk::Text(t)) => t.extend(toks),
        _ => chunks.push(MtmdChunk::Text(toks)),
    }
}

/// mtmd.cpp:652 `lookup_token` — the exact-piece scan over the whole vocab
/// (special form). No vocab / no match returns the C's LLAMA_TOKEN_NULL (-1).
fn lookup_token(vocab: Option<&Vocab>, token_text: &str) -> Vec<i32> {
    let Some(vocab) = vocab else {
        return vec![-1];
    };
    let n_vocab = vocab.n_tokens();
    for i in 0..n_vocab {
        if vocab.token_to_piece(i as i32) == token_text {
            return vec![i as i32];
        }
    }
    vec![-1]
}

// ======================================================================
// the text-side driver — mtmd-helper.cpp
// ======================================================================

/// The text model driver of `mtmd_helper_eval_chunk_single` (:158) and
/// `mtmd_helper_decode_image_chunk` (:117).
///
/// **How the vision embeddings reach the text graph.** The C hands
/// `llama_batch.embd` to `llama_decode`, and `build_inp_embd`
/// (llama-graph.cpp:2387) reads those rows instead of looking up token
/// embeddings. This port's `LlamaBatch` has no `embd` field, so the same effect
/// is obtained one level lower: the whole prompt's input matrix is materialised
/// as a *weight* tensor below the decoder's graph watermark — text rows are
/// `tok_embd`'s own rows read through the port's `get_rows` kernels, image rows
/// are the clip embeddings — and the decoder is constructed with that matrix as
/// its `tok_embd`, the "tokens" being `[0, n)` row indices into it. Identical
/// values in, identical graph out; the lm head keeps pointing at the file's
/// real output tensor, and generated tokens get their rows appended to the same
/// matrix.
pub struct MtmdEval {
    pub dctx: DecodeContext,
    /// the substituted input matrix ([n_embd, n_capacity], F32)
    pub input_embd: TensorId,
    /// the model's own token-embedding tensor (source of the generated rows)
    tok_embd_src: TensorId,
    n_embd: usize,
    n_pos_per_embd: usize,
    /// flat decoder positions, `[section * n_capacity + row]`
    positions: Vec<i32>,
    /// next free row (== prompt rows once the prompt is consumed)
    row_cursor: usize,
    /// how many prompt rows have been decoded
    decoded: usize,
    /// `n_past` for the text model (`mtmd_input_chunk_get_n_pos` accounting)
    pub n_past: i32,
}

impl MtmdEval {
    /// Materialise `chunks` into the input matrix and build the decoder.
    ///
    /// `gctx` must already hold the text model's tensors; `weights` is the
    /// forward-weight set built from them (the helpers `llama-cli` uses),
    /// whose `tok_embd` is replaced here. `images` holds one embedding buffer
    /// per image chunk, in chunk order (see [`MtmdContext::encode_chunk`]).
    /// `n_gen_capacity` rows are reserved for tokens generated later.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut gctx: Context,
        mut weights: ForwardWeights,
        attn: crate::graph::AttnParams,
        n_ctx: u32,
        n_threads: usize,
        n_batch: usize,
        mtmd: &MtmdContext,
        chunks: &[MtmdChunk],
        images: &[Vec<f32>],
        n_gen_capacity: usize,
    ) -> Result<Self, String> {
        let n_pos_per_embd = if mtmd.decode_use_mrope() { 4 } else { 1 };
        let tok_embd = weights_tok_embd(&weights);
        let n_embd = gctx.ne(tok_embd)[0] as usize;
        if n_embd != mtmd.n_mmproj_embd() as usize {
            return Err(format!(
                "text model n_embd {n_embd} != mmproj n_embd {}",
                mtmd.n_mmproj_embd()
            ));
        }

        let n_rows: usize = chunks.iter().map(|c| c.n_tokens()).sum();
        if n_rows > n_ctx as usize {
            return Err(format!("prompt of {n_rows} tokens exceeds n_ctx {n_ctx}"));
        }
        let n_capacity = n_rows + n_gen_capacity;
        if n_capacity > n_ctx as usize {
            return Err(format!(
                "prompt + generation capacity ({n_capacity}) exceeds n_ctx {n_ctx}"
            ));
        }

        // ---- positions + the text rows to look up ---------------------------
        let mut positions = vec![0i32; n_pos_per_embd * n_capacity];
        let mut text_rows: Vec<i32> = Vec::new();
        let mut row = 0usize;
        let mut pos = 0i32;
        let mut img_idx = 0usize;
        // row -> (is image, image index, token id) is not needed: the fill pass
        // below walks the chunks in the same order with the same cursors.
        for chunk in chunks {
            match chunk {
                MtmdChunk::Text(toks) => {
                    for k in 0..toks.len() {
                        for s in 0..n_pos_per_embd {
                            positions[s * n_capacity + row + k] = pos + k as i32;
                        }
                        text_rows.push(toks[k]);
                    }
                    row += toks.len();
                    pos += toks.len() as i32;
                }
                MtmdChunk::Image(img) => {
                    let n_tok = img.n_tokens();
                    if img_idx >= images.len() {
                        return Err("missing encoded image embeddings".into());
                    }
                    let embd = &images[img_idx];
                    img_idx += 1;
                    if embd.len() != n_tok * n_embd {
                        return Err(format!(
                            "image embeddings have {} elements, expected {}",
                            embd.len(),
                            img.n_tokens() * n_embd
                        ));
                    }
                    let _ = &n_tok;
                    for i in 0..img.n_tokens() {
                        let p = img.decoder_pos(pos, i);
                        for (s, v) in p.iter().enumerate().take(n_pos_per_embd) {
                            positions[s * n_capacity + row + i] = *v;
                        }
                    }
                    row += img.n_tokens();
                    pos += img.n_pos();
                }
                MtmdChunk::Audio(a) => {
                    // audio embeddings occupy plain consecutive positions
                    // (mtmd.cpp:341-345: one pos per token, mrope not used)
                    if img_idx >= images.len() {
                        return Err("missing encoded audio embeddings".into());
                    }
                    let embd = &images[img_idx];
                    img_idx += 1;
                    if embd.len() != a.n_tokens as usize * n_embd {
                        return Err(format!(
                            "audio embeddings have {} elements, expected {}",
                            embd.len(),
                            a.n_tokens as usize * n_embd
                        ));
                    }
                    for i in 0..a.n_tokens as usize {
                        positions[row + i] = pos + i as i32;
                    }
                    row += a.n_tokens as usize;
                    pos += a.n_tokens as i32;
                }
            }
        }
        if img_idx != images.len() {
            return Err("more encoded images than image chunks".into());
        }
        assert_eq!(row, n_rows);

        // ---- the input matrix ------------------------------------------------
        // text rows come out of the port's own get_rows kernels, so they are
        // bit-identical to what the unmodified graph would read
        let dequant = read_tok_embd_rows(&mut gctx, tok_embd, &text_rows, n_threads);

        let input_embd = gctx.new_tensor_2d(GgmlType::F32, n_embd as i64, n_capacity as i64);
        gctx.set_name(input_embd, "mtmd_inp_embd");
        gctx.arena_resize_tensor(input_embd);
        {
            let dst = gctx.data_bytes_mut(input_embd).expect("arena tensor");
            let mut dq = 0usize;
            let mut off = 0usize;
            let mut img_idx = 0usize;
            for chunk in chunks {
                match chunk {
                    MtmdChunk::Text(toks) => {
                        for _ in 0..toks.len() {
                            let src = &dequant[dq * n_embd..(dq + 1) * n_embd];
                            dst[off..off + n_embd * 4].copy_from_slice(bytemuck::cast_slice(src));
                            dq += 1;
                            off += n_embd * 4;
                        }
                    }
                    MtmdChunk::Image(img) => {
                        let embd = &images[img_idx];
                        img_idx += 1;
                        let nb = img.n_tokens() * n_embd * 4;
                        dst[off..off + nb].copy_from_slice(bytemuck::cast_slice(embd));
                        off += nb;
                    }
                    MtmdChunk::Audio(a) => {
                        // audio rows take the same slot shape (mtmd.cpp:1822)
                        let embd = &images[img_idx];
                        img_idx += 1;
                        let nb = a.n_tokens as usize * n_embd * 4;
                        dst[off..off + nb].copy_from_slice(bytemuck::cast_slice(embd));
                        off += nb;
                    }
                }
            }
            assert_eq!(off, n_rows * n_embd * 4);
        }

        set_weights_tok_embd(&mut weights, input_embd);
        let dctx = DecodeContext::new_with(gctx, weights, attn, n_ctx, n_threads, n_batch);
        Ok(Self {
            dctx,
            input_embd,
            tok_embd_src: tok_embd,
            n_embd,
            n_pos_per_embd,
            positions,
            row_cursor: n_rows,
            decoded: 0,
            n_past: 0,
        })
    }

    fn row_pos(&self, row: usize, section: usize) -> i32 {
        let n_capacity = self.positions.len() / self.n_pos_per_embd;
        self.positions[section * n_capacity + row]
    }

    /// `mtmd_helper_eval_chunk_single` (:158) for text chunks and
    /// `mtmd_helper_decode_image_chunk` (:117) for image chunks (whose
    /// embeddings were already computed). Batches are split at `n_batch`.
    pub fn eval_chunk(
        &mut self,
        chunk: &MtmdChunk,
        logits_last: bool,
    ) -> Result<Option<Vec<f32>>, String> {
        let n_tokens = chunk.n_tokens();
        if n_tokens == 0 {
            return Ok(None);
        }
        let n_batch = self.dctx.n_batch.max(1);
        let start = self.decoded;
        let mut last_logits = None;
        let mut first = 0usize;
        while first < n_tokens {
            let n = (n_tokens - first).min(n_batch);
            let mut batch = crate::batch::LlamaBatch::default();
            for k in 0..n {
                batch.token.push((start + first + k) as i32);
                batch.seq_id.get_or_insert_with(Vec::new).push(vec![0]);
                batch
                    .logits
                    .get_or_insert_with(Vec::new)
                    .push(logits_last && first + k + 1 == n_tokens);
            }
            // `decode_embd_batch` (mtmd-helper-common.h:82) lays positions out
            // section-major: `pos[i*n_tokens + k]` (llama-batch.cpp:784 reads
            // them back that way for an embedding batch)
            let mut pos = vec![0i32; self.n_pos_per_embd * n];
            for s in 0..self.n_pos_per_embd {
                for k in 0..n {
                    pos[s * n + k] = self.row_pos(start + first + k, s);
                }
            }
            batch.pos = Some(pos);
            let out = self.dctx.decode_batch(&batch)?;
            if let Some(l) = out.logits_ith(-1) {
                last_logits = Some(l.to_vec());
            }
            first += n;
        }
        self.decoded += n_tokens;
        self.n_past += chunk.n_pos();
        Ok(last_logits)
    }

    /// The generation step of `mtmd-cli.cpp:206` (`common_batch_add` + decode):
    /// append `token`'s embedding row to the input matrix and decode it.
    pub fn decode_token(&mut self, token: i32, logits: bool) -> Result<Option<Vec<f32>>, String> {
        if self.row_cursor >= self.positions.len() / self.n_pos_per_embd {
            return Err("generation exceeds the reserved embedding rows".into());
        }
        let row = self.row_cursor;
        let n_capacity = self.positions.len() / self.n_pos_per_embd;
        // this row's position: the text convention broadcasts one value
        for s in 0..self.n_pos_per_embd {
            self.positions[s * n_capacity + row] = self.n_past;
        }
        let mut vals = vec![0f32; self.n_embd];
        read_tok_embd_row(&self.dctx.gctx, self.tok_embd_src, token, &mut vals);
        let off = row * self.n_embd * 4;
        self.dctx
            .gctx
            .data_bytes_mut(self.input_embd)
            .expect("arena tensor")[off..off + self.n_embd * 4]
            .copy_from_slice(bytemuck::cast_slice(&vals));
        self.row_cursor += 1;
        self.n_past += 1;

        let mut batch = crate::batch::LlamaBatch::default();
        batch.token.push(row as i32);
        batch.pos = Some(
            (0..self.n_pos_per_embd)
                .map(|s| self.row_pos(row, s))
                .collect(),
        );
        batch.seq_id = Some(vec![vec![0]]);
        batch.logits = Some(vec![logits]);
        let out = self.dctx.decode_batch(&batch)?;
        Ok(out.logits_ith(0).map(|l| l.to_vec()))
    }
}

/// `build_inp_embd` (llama-graph.cpp:2387) reads `inp->embd` when the batch
/// carries embeddings, otherwise `get_rows(tok_embd, tokens)`. The port feeds
/// the mixed matrix through the second path, so the `tok_embd` slot of the
/// forward weights has to be swapped for the materialised matrix.
fn weights_tok_embd(w: &ForwardWeights) -> TensorId {
    use ForwardWeights::*;
    match w {
        Qwen2(x) => x.tok_embd,
        Llama(x) => x.tok_embd,
        Qwen3(x) => x.tok_embd,
        Gemma2(x, _) | Gemma3(x, _) => x.tok_embd,
        Gemma4(x, _) => x.tok_embd,
        Phi3(x) => x.tok_embd,
        Granite(x, _) => x.tok_embd,
        Lfm2(x, _) => x.tok_embd,
        Qwen35(x, _) => x.tok_embd,
        GptOss(x, _) => x.tok_embd,
        Gpt2(x, _) => x.tok_embd,
        Phi2(x, _) => x.tok_embd,
        StarCoder2(x, _) => x.tok_embd,
        CommandR(x, _) => x.tok_embd,
        GptNeox(x, _) => x.tok_embd,
        Olmo2(x, _) => x.tok_embd,
        // compatibility wildcard for archs landed after this match was written
        // (arch batch 2, 2026-09-25): every ForwardWeights variant carries a
        // `tok_embd`, and `ForwardWeights::tok_embd` is the exhaustive
        // accessor over the same field
        _ => w.tok_embd(),
    }
}

fn set_weights_tok_embd(w: &mut ForwardWeights, id: TensorId) {
    use ForwardWeights::*;
    match w {
        Qwen2(x) => x.tok_embd = id,
        Llama(x) => x.tok_embd = id,
        Qwen3(x) => x.tok_embd = id,
        Gemma2(x, _) | Gemma3(x, _) => x.tok_embd = id,
        Gemma4(x, _) => x.tok_embd = id,
        Phi3(x) => x.tok_embd = id,
        Granite(x, _) => x.tok_embd = id,
        Lfm2(x, _) => x.tok_embd = id,
        Qwen35(x, _) => x.tok_embd = id,
        GptOss(x, _) => x.tok_embd = id,
        Gpt2(x, _) => x.tok_embd = id,
        Phi2(x, _) => x.tok_embd = id,
        StarCoder2(x, _) => x.tok_embd = id,
        CommandR(x, _) => x.tok_embd = id,
        GptNeox(x, _) => x.tok_embd = id,
        Olmo2(x, _) => x.tok_embd = id,
        // see `weights_tok_embd`
        _ => w.set_tok_embd(id),
    }
}

/// Read rows `ids` of `tok_embd` through the port's own `get_rows` kernels, so
/// the text rows of the substituted matrix are bit-identical to what the graph
/// would have produced (compute.rs `forward_get_rows`).
fn read_tok_embd_rows(
    gctx: &mut Context,
    tok_embd: TensorId,
    ids: &[i32],
    n_threads: usize,
) -> Vec<f32> {
    let n_embd = gctx.ne(tok_embd)[0] as usize;
    if ids.is_empty() {
        return Vec::new();
    }
    let mark = gctx.mark();
    let idx = gctx.new_tensor_1d(GgmlType::I32, ids.len() as i64);
    gctx.arena_resize_tensor(idx);
    gctx.with_i32_mut(idx, |p| p.copy_from_slice(ids)).unwrap();
    let out = gctx.get_rows(tok_embd, idx);
    let mut g = Graph::new(4);
    g.build_forward(gctx, out);
    ggml::compute::graph_compute(gctx, &mut g, n_threads);
    let vals = gctx.f32s(out).expect("arena tensor").to_vec();
    gctx.reset_graph_to(mark);
    assert_eq!(vals.len(), ids.len() * n_embd);
    vals
}

/// Single row, same kernels as `forward_get_rows`. Used for generated tokens,
/// where the graph watermark must not be touched any more.
fn read_tok_embd_row(gctx: &Context, tok_embd: TensorId, id: i32, out: &mut [f32]) {
    let ty = gctx.ty(tok_embd);
    let n_embd = gctx.ne(tok_embd)[0] as usize;
    let nb1 = gctx.nb(tok_embd)[1] as usize;
    assert!(
        id >= 0 && (id as usize) < gctx.ne(tok_embd)[1] as usize,
        "token id out of range"
    );
    let off = id as usize * nb1;
    let row_bytes = ty.row_size(n_embd);
    let bytes = gctx.data_bytes(tok_embd).expect("tok_embd storage");
    match ty {
        GgmlType::F32 => {
            let src: &[f32] = bytemuck::cast_slice(&bytes[off..off + row_bytes]);
            out.copy_from_slice(src);
        }
        GgmlType::F16 => {
            let src: &[half::f16] = bytemuck::cast_slice(&bytes[off..off + row_bytes]);
            for (o, v) in out.iter_mut().zip(src) {
                *o = v.to_f32();
            }
        }
        GgmlType::Bf16 => {
            let src: &[half::bf16] = bytemuck::cast_slice(&bytes[off..off + row_bytes]);
            for (o, v) in out.iter_mut().zip(src) {
                *o = v.to_f32();
            }
        }
        q => ggml::quants::dequantize_row(q, &bytes[off..off + row_bytes], out),
    }
}

// ======================================================================
// bitmap loading — the reference uses stb_image (mtmd-helper.cpp:36,
// `stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3)` at :407). The port
// implements the formats its own test harness and the CLI need:
// PNG (8/16-bit, non-interlaced, colour types 0/2/3/4/6), BMP (24/32-bit
// uncompressed) and binary PPM/PGM. JPEG and the stb-only formats are not
// ported.
// ======================================================================

/// `mtmd_helper_bitmap_init_from_buf` (mtmd-helper.cpp:360) for images:
/// sniff the format, decode to RGB8, 3 channels like `stbi_load(..., 3)`.
pub fn bitmap_from_bytes(bytes: &[u8]) -> Result<ClipImageU8, String> {
    if bytes.len() >= 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return png_decode(bytes);
    }
    if bytes.len() >= 2 && &bytes[..2] == b"BM" {
        return bmp_decode(bytes);
    }
    if bytes.len() >= 2 && bytes[0] == b'P' && (bytes[1] == b'6' || bytes[1] == b'5') {
        return ppm_decode(bytes);
    }
    Err("unsupported image format (PNG/BMP/PPM only)".into())
}

pub fn bitmap_from_file(path: &str) -> Result<ClipImageU8, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    bitmap_from_bytes(&bytes).map_err(|e| format!("{path}: {e}"))
}

// ======================================================================
// audio loading — the reference uses miniaudio (`ma_decoder`,
// mtmd-helper.cpp:325 `decode_audio_from_buf`: f32 mono output at the
// projector's sample rate; WAV/MP3/FLAC in, resampling included). The port
// implements the uncompressed-RIFF subset the parity fixtures use: PCM
// 8/16/24/32-bit and IEEE 32/64-bit float, any channel count (mixed to mono
// by averaging, like miniaudio's channel converter). MP3/FLAC and sample-rate
// conversion are NOT ported — a rate mismatch is an error instead of a
// silent resample.
// ======================================================================

/// `audio_helpers::is_audio_file` (mtmd-helper.cpp:306-322), WAV branch only.
pub fn is_wav_file(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// `audio_helpers::is_audio_file`'s mp3 sniffer (mtmd-helper.cpp:310-315):
/// an ID3v2 tag or an MPEG frame sync word.
pub fn is_mp3_file(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && (&bytes[..3] == b"ID3" || (bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0))
}

/// `audio_helpers::is_audio_file`'s flac sniffer (mtmd-helper.cpp:317).
pub fn is_flac_file(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == b"fLaC"
}

// ---------------------------------------------------------------------
// FLAC (audio round 4) — a minimal, spec-bounded decoder for the common
// subset the reference's dr_flac handles: STREAMINFO + VERBATIM/FIXED/LPC
// frames, all standard block sizes/orders, all channel assignments. The
// numeric pipeline is copied from the reference (vendor/miniaudio/miniaudio.h
// ma_dr_flac__*), so the decoded PCM is bit-identical for files the
// reference itself accepts:
//   * subframe samples are FLAC-native i32 (bps-wide, sign-extended)
//   * stereo decorrelation uses the exact drflac kernels (left/right/mid-side)
//   * each channel is left-aligned to 32 bits: `(u32)s << (32-bps + wasted)`
//   * s32 -> f32 is `(float)((double)x / 2147483648.0)` (miniaudio.h:46270
//     ma_pcm_s32_to_f32__reference)
//   * N -> mono mixes in f32: sequential sum, then /N (miniaudio.h:55649
//     ma_channel_converter_process_pcm_frames__mono_out, f32 case) — the
//     same recipe the port's WAV reader already uses
// Resampling is still NOT ported (the reference resamples via
// ma_linear_resampler): a rate mismatch fails loudly, same as WAV.
// ---------------------------------------------------------------------

/// MSB-first bit reader over a byte slice with CRC-8/CRC-16 tracking
/// (dr_flac validates both; so does this port).
struct FlacBits<'a> {
    data: &'a [u8],
    /// bit position from the start of `data`
    pos: usize,
}

impl<'a> FlacBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        FlacBits { data, pos: 0 }
    }

    fn read_bit(&mut self) -> Option<u32> {
        let byte = self.pos / 8;
        if byte >= self.data.len() {
            return None;
        }
        let bit = 7 - (self.pos % 8);
        self.pos += 1;
        Some(((self.data[byte] >> bit) & 1) as u32)
    }

    /// `ma_dr_flac__read_uint32` — n <= 32 bits, MSB first
    fn read_u32(&mut self, n: u32) -> Option<u32> {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Some(v)
    }

    /// `ma_dr_flac__read_int32` — n bits, sign extended
    fn read_i32(&mut self, n: u32) -> Option<i32> {
        if n == 0 {
            return Some(0);
        }
        let v = self.read_u32(n)?;
        // sign extend from bit n-1
        let shift = 32 - n;
        Some(((v << shift) as i32) >> shift)
    }

    /// `ma_dr_flac__seek_past_next_set_bit` — count of zero bits before the
    /// next 1 bit (the unary code of the wasted-bits flag)
    fn seek_past_next_set_bit(&mut self) -> Option<u32> {
        let mut count = 0u32;
        loop {
            match self.read_bit() {
                Some(0) => count += 1,
                Some(_) => return Some(count),
                None => return None,
            }
        }
    }

    fn byte_aligned_pos(&self) -> usize {
        self.pos.div_ceil(8) * 8
    }
}

/// CRC-8 (poly 0x07) over the last `bits` bits — dr_flac seeds with
/// 0xCE (the sync code's contribution) and folds the frame-header bits
/// after the sync word, which start mid-byte, so this is a BIT-level walk
/// (the canonical MSB-first CRC; dr_flac's table path is the same CRC).
fn flac_crc8_bits(seed: u8, data: &[u8], start_bit: usize, end_bit: usize) -> u8 {
    let mut crc = seed;
    for i in start_bit..end_bit {
        let bit = (data[i / 8] >> (7 - (i % 8))) & 1;
        crc ^= bit << 7;
        crc = if crc & 0x80 != 0 {
            (crc << 1) ^ 0x07
        } else {
            crc << 1
        };
    }
    crc
}

/// CRC-16 (poly 0x8005, init 0)
fn flac_crc16(seed: u16, data: &[u8]) -> u16 {
    let mut crc = seed;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

struct FlacStreamInfo {
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    /// 0 = unknown (streamed encoders)
    total_samples: u64,
}

fn flac_parse_streaminfo(data: &[u8]) -> Result<FlacStreamInfo, String> {
    if data.len() < 34 {
        return Err("FLAC STREAMINFO too short".into());
    }
    let rd64 = |o: usize| -> u64 {
        let mut v = 0u64;
        for k in 0..8 {
            v = (v << 8) | data[o + k] as u64;
        }
        v
    };
    // bytes 10..18: (20 bits sr)(3 bits ch-1)(5 bits bps-1)(36 bits total)
    let hi = rd64(10);
    let sample_rate = ((hi >> 44) & 0xFFFFF) as u32;
    let channels = (((hi >> 41) & 0x7) + 1) as u8;
    let bits_per_sample = (((hi >> 36) & 0x1F) + 1) as u8;
    let total_samples = hi & 0xF_FFFF_FFFF;
    if sample_rate == 0 {
        return Err("FLAC STREAMINFO has no sample rate".into());
    }
    if !(4..=32).contains(&bits_per_sample) {
        return Err(format!(
            "FLAC bits-per-sample {bits_per_sample} out of range"
        ));
    }
    Ok(FlacStreamInfo {
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum SubframeType {
    Constant,
    Verbatim,
    Fixed,
    Lpc,
}

struct FlacFrameHeader {
    block_size: u32,
    sample_rate: u32,
    channel_assignment: u8,
    bits_per_sample: u8,
}

/// `ma_dr_flac__read_next_flac_frame_header` — returns None at end of stream,
/// or the header. Malformed frames are skipped exactly like dr_flac (the
/// caller keeps scanning for the next sync code).
fn flac_read_frame_header(
    bs: &mut FlacBits,
    si: &FlacStreamInfo,
) -> Result<Option<FlacFrameHeader>, String> {
    const SAMPLE_RATE_TABLE: [u32; 12] = [
        0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
    ];
    const BITS_PER_SAMPLE_TABLE: [u8; 8] = [0, 8, 12, 0xFF, 16, 20, 24, 0xFF];

    loop {
        // find the 14-bit sync code 0b11111111111110
        let mut sync = 0u16;
        loop {
            let Some(b) = bs.read_bit() else {
                return Ok(None);
            };
            sync = ((sync << 1) | b as u16) & 0x3FFF;
            if sync == 0b11_1111_1111_1110 {
                break;
            }
        }
        let header_start_bit = bs.pos - 14;

        let Some(reserved) = bs.read_bit() else {
            return Ok(None);
        };
        if reserved == 1 {
            continue;
        }
        let Some(blocking_strategy) = bs.read_bit() else {
            return Ok(None);
        };
        let Some(block_size_code) = bs.read_u32(4) else {
            return Ok(None);
        };
        if block_size_code == 0 {
            continue;
        }
        let Some(sample_rate_code) = bs.read_u32(4) else {
            return Ok(None);
        };
        let Some(channel_assignment) = bs.read_u32(4) else {
            return Ok(None);
        };
        if channel_assignment > 10 {
            continue;
        }
        let Some(bits_per_sample_code) = bs.read_u32(3) else {
            return Ok(None);
        };
        if bits_per_sample_code == 3 || bits_per_sample_code == 7 {
            continue;
        }
        let Some(reserved2) = bs.read_bit() else {
            return Ok(None);
        };
        if reserved2 == 1 {
            continue;
        }
        let _ = blocking_strategy;

        // UTF-8 coded frame/sample number (dr_flac: ma_dr_flac__read_utf8_coded_number)
        let Some(first) = bs.read_u32(8) else {
            return Ok(None);
        };
        let n_extra = if first & 0x80 == 0 {
            0
        } else if first & 0xE0 == 0xC0 {
            1
        } else if first & 0xF0 == 0xE0 {
            2
        } else if first & 0xF8 == 0xF0 {
            3
        } else if first & 0xFC == 0xF8 {
            4
        } else if first & 0xFE == 0xFC {
            5
        } else if first == 0xFE {
            6
        } else {
            continue; // invalid UTF-8 lead byte
        };
        let mut number = if n_extra == 0 {
            first
        } else {
            first & (0x3F >> n_extra)
        };
        let mut ok = true;
        for _ in 0..n_extra {
            let Some(cont) = bs.read_u32(8) else {
                return Ok(None);
            };
            if cont & 0xC0 != 0x80 {
                ok = false;
                break;
            }
            number = (number << 6) | (cont & 0x3F);
        }
        if !ok {
            continue;
        }
        let _ = number;

        let block_size = match block_size_code {
            1 => 192,
            c @ 2..=5 => 576u32 << (c - 2),
            6 => {
                let Some(v) = bs.read_u32(8) else {
                    return Ok(None);
                };
                v + 1
            }
            7 => {
                let Some(v) = bs.read_u32(16) else {
                    return Ok(None);
                };
                if v == 0xFFFF {
                    return Ok(None);
                }
                v + 1
            }
            c => 256u32 << (c - 8),
        };

        let sample_rate = match sample_rate_code {
            c @ 0..=11 => {
                if c == 0 {
                    si.sample_rate
                } else {
                    SAMPLE_RATE_TABLE[c as usize]
                }
            }
            12 => {
                let Some(v) = bs.read_u32(8) else {
                    return Ok(None);
                };
                v * 1000
            }
            13 => {
                let Some(v) = bs.read_u32(16) else {
                    return Ok(None);
                };
                v
            }
            14 => {
                let Some(v) = bs.read_u32(16) else {
                    return Ok(None);
                };
                v * 10
            }
            _ => continue, // 15: invalid / from STREAMINFO (no side data)
        };

        let mut bits_per_sample = BITS_PER_SAMPLE_TABLE[bits_per_sample_code as usize];
        if bits_per_sample == 0 {
            bits_per_sample = si.bits_per_sample;
        }
        if bits_per_sample != si.bits_per_sample {
            // dr_flac rejects frames that disagree with STREAMINFO
            return Err("FLAC frame bits-per-sample disagrees with STREAMINFO".into());
        }

        // CRC-8 over everything after the sync code, seeded 0xCE (the sync
        // bits' contribution — the header bits start mid-byte, so the CRC is
        // a bit-level walk)
        let Some(stored_crc8) = bs.read_u32(8) else {
            return Ok(None);
        };
        let crc_end_bit = bs.pos - 8;
        let crc = flac_crc8_bits(0xCE, bs.data, header_start_bit + 14, crc_end_bit);
        if crc as u32 != stored_crc8 {
            continue; // CRC mismatch: resync like dr_flac
        }

        return Ok(Some(FlacFrameHeader {
            block_size,
            sample_rate,
            channel_assignment: channel_assignment as u8,
            bits_per_sample,
        }));
    }
}

/// `ma_dr_flac__calculate_prediction_32` — wrapping i32 accumulation with an
/// arithmetic shift.
#[allow(clippy::needless_range_loop)]
fn flac_prediction_32(order: usize, shift: i32, coef: &[i32; 32], s: &[i32]) -> i32 {
    let mut prediction = 0i32;
    for j in 0..order {
        prediction = prediction.wrapping_add(coef[j].wrapping_mul(s[s.len() - 1 - j]));
    }
    prediction >> shift
}

/// `ma_dr_flac__calculate_prediction_64` — i64 accumulation, truncated.
#[allow(clippy::needless_range_loop)]
fn flac_prediction_64(order: usize, shift: i32, coef: &[i32; 32], s: &[i32]) -> i32 {
    let mut prediction = 0i64;
    for j in 0..order {
        prediction += coef[j] as i64 * s[s.len() - 1 - j] as i64;
    }
    ((prediction >> shift) & 0xFFFF_FFFF) as u32 as i32
}

/// `ma_dr_flac__ilog2_u32`
fn flac_ilog2(mut x: u32) -> u32 {
    let mut r = 0;
    while x > 0 {
        r += 1;
        x >>= 1;
    }
    r
}

/// decode one residual + prediction step (both rice and unencoded paths feed
/// this): dr_flac's `pSamplesOut[i] = decodedRice + prediction`
fn flac_push_sample(
    decoded: u32,
    bits_per_sample: u32,
    order: usize,
    shift: i32,
    precision: u32,
    coef: &[i32; 32],
    out: &mut Vec<i32>,
) {
    let prediction = if bits_per_sample + precision + flac_ilog2(order as u32) > 32 {
        flac_prediction_64(order, shift, coef, out)
    } else {
        flac_prediction_32(order, shift, coef, out)
    };
    let s = (decoded as i32).wrapping_add(prediction);
    out.push(s);
}

/// `ma_dr_flac__decode_samples_with_residual` — the partitioned Rice / Rice2
/// residual, reconstructing into `out` (already holding the warmup samples).
fn flac_read_residual(
    bs: &mut FlacBits,
    bits_per_sample: u32,
    block_size: u32,
    order: usize,
    shift: i32,
    precision: u32,
    coef: &[i32; 32],
    out: &mut Vec<i32>,
) -> Result<(), String> {
    let Some(method) = bs.read_u32(2) else {
        return Err("FLAC stream truncated in residual coding method".into());
    };
    if method > 1 {
        return Err("FLAC residual coding method > 1".into());
    }
    let rice_bits = if method == 0 { 4 } else { 5 };
    let escape_code = if method == 0 { 15 } else { 31 };

    let Some(partition_order) = bs.read_u32(4) else {
        return Err("FLAC stream truncated in partition order".into());
    };
    if partition_order > 8 {
        return Err(format!("FLAC partition order {partition_order} > 8"));
    }
    if block_size / (1 << partition_order) < order as u32 {
        return Err("FLAC partition too small for the LPC order".into());
    }

    let mut samples_in_partition = (block_size / (1 << partition_order)) - order as u32;
    let mut partitions_remaining = 1u32 << partition_order;
    loop {
        let Some(mut rice_param) = bs.read_u32(rice_bits) else {
            return Err("FLAC stream truncated in rice parameter".into());
        };
        let escaped = rice_param == escape_code;
        let unencoded_bits = if escaped {
            let Some(n) = bs.read_u32(5) else {
                return Err("FLAC stream truncated in escape bits".into());
            };
            rice_param = 0xFF;
            n
        } else {
            0
        };

        for _ in 0..samples_in_partition {
            let decoded = if !escaped {
                // rice: unary quotient + k-bit remainder, zigzag-decoded
                let mut zero_counter = 0u32;
                loop {
                    let Some(bit) = bs.read_bit() else {
                        return Err("FLAC stream truncated in rice unary".into());
                    };
                    if bit == 0 {
                        zero_counter += 1;
                    } else {
                        break;
                    }
                }
                let mut v = if rice_param > 0 {
                    let Some(r) = bs.read_u32(rice_param) else {
                        return Err("FLAC stream truncated in rice remainder".into());
                    };
                    r
                } else {
                    0
                };
                v |= zero_counter << rice_param;
                if v & 0x01 != 0 {
                    !(v >> 1)
                } else {
                    v >> 1
                }
            } else {
                // unencoded: sign-extended n-bit value
                let Some(v) = bs.read_i32(unencoded_bits) else {
                    return Err("FLAC stream truncated in unencoded residual".into());
                };
                v as u32
            };
            flac_push_sample(decoded, bits_per_sample, order, shift, precision, coef, out);
        }

        if partitions_remaining == 1 {
            break;
        }
        partitions_remaining -= 1;
        if partition_order != 0 {
            samples_in_partition = block_size / (1 << partition_order);
        }
    }
    Ok(())
}

/// decode one subframe into `out` (cleared on entry); returns the subframe's
/// wasted-bits count (the caller needs it for the left-alignment shift)
#[allow(clippy::too_many_arguments)]
fn flac_decode_subframe(
    bs: &mut FlacBits,
    header: &FlacFrameHeader,
    channel_index: usize,
    out: &mut Vec<i32>,
) -> Result<u32, String> {
    // subframe header (dr_flac read_subframe_header)
    let Some(h) = bs.read_u32(8) else {
        return Err("FLAC stream truncated in subframe header".into());
    };
    if h & 0x80 != 0 {
        return Err("FLAC subframe header has the zero bit set".into());
    }
    let ty = (h & 0x7E) >> 1;
    let (stype, order): (SubframeType, usize) = if ty == 0 {
        (SubframeType::Constant, 0)
    } else if ty == 1 {
        (SubframeType::Verbatim, 0)
    } else if ty & 0x20 != 0 {
        (SubframeType::Lpc, (ty & 0x1F) as usize + 1)
    } else if ty & 0x08 != 0 {
        let o = (ty & 0x07) as usize;
        if o > 4 {
            return Err("FLAC fixed subframe order > 4".into());
        }
        (SubframeType::Fixed, o)
    } else {
        return Err("FLAC reserved subframe type".into());
    };

    let mut wasted_bits = 0u32;
    if h & 0x01 == 1 {
        let Some(n) = bs.seek_past_next_set_bit() else {
            return Err("FLAC stream truncated in wasted bits".into());
        };
        wasted_bits = n + 1;
    }

    // side channels carry one extra bit (dr_flac decode_subframe)
    let mut subframe_bps = header.bits_per_sample as u32;
    let ca = header.channel_assignment;
    if (ca == 8 || ca == 10) && channel_index == 1 {
        subframe_bps += 1;
    } else if ca == 9 && channel_index == 0 {
        subframe_bps += 1;
    }
    if subframe_bps > 32 {
        return Err("FLAC side-channel bits-per-sample > 32".into());
    }
    if wasted_bits >= subframe_bps {
        return Err("FLAC wasted bits >= bits-per-sample".into());
    }
    subframe_bps -= wasted_bits;

    if header.block_size < order as u32 {
        return Err("FLAC block size < LPC order".into());
    }

    out.clear();
    match stype {
        SubframeType::Constant => {
            let Some(s) = bs.read_i32(subframe_bps) else {
                return Err("FLAC stream truncated in constant subframe".into());
            };
            for _ in 0..header.block_size {
                out.push(s);
            }
        }
        SubframeType::Verbatim => {
            for _ in 0..header.block_size {
                let Some(s) = bs.read_i32(subframe_bps) else {
                    return Err("FLAC stream truncated in verbatim subframe".into());
                };
                out.push(s);
            }
        }
        SubframeType::Fixed => {
            const FIXED_COEFS: [[i32; 4]; 5] = [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, -1, 0, 0],
                [3, -3, 1, 0],
                [4, -6, 4, -1],
            ];
            for _ in 0..order {
                let Some(s) = bs.read_i32(subframe_bps) else {
                    return Err("FLAC stream truncated in fixed warmup".into());
                };
                out.push(s);
            }
            let mut coef = [0i32; 32];
            coef[..4].copy_from_slice(&FIXED_COEFS[order]);
            flac_read_residual(bs, subframe_bps, header.block_size, order, 0, 4, &coef, out)?;
        }
        SubframeType::Lpc => {
            for _ in 0..order {
                let Some(s) = bs.read_i32(subframe_bps) else {
                    return Err("FLAC stream truncated in LPC warmup".into());
                };
                out.push(s);
            }
            let Some(precision_raw) = bs.read_u32(4) else {
                return Err("FLAC stream truncated in LPC precision".into());
            };
            if precision_raw == 15 {
                return Err("FLAC LPC precision 15 is invalid".into());
            }
            let precision = precision_raw + 1;
            let Some(shift_raw) = bs.read_i32(5) else {
                return Err("FLAC stream truncated in LPC shift".into());
            };
            if shift_raw < 0 {
                return Err("FLAC negative LPC shift".into());
            }
            let mut coef = [0i32; 32];
            for c in coef.iter_mut().take(order) {
                let Some(v) = bs.read_i32(precision) else {
                    return Err("FLAC stream truncated in LPC coefficients".into());
                };
                *c = v;
            }
            flac_read_residual(
                bs,
                subframe_bps,
                header.block_size,
                order,
                shift_raw,
                precision,
                &coef,
                out,
            )?;
        }
    }
    Ok(wasted_bits)
}

/// decode one FLAC frame: subframes, byte-align, CRC-16, stereo
/// decorrelation + 32-bit left alignment. Returns None at end of stream.
fn flac_decode_frame(
    bs: &mut FlacBits,
    si: &FlacStreamInfo,
    out: &mut Vec<i32>,
) -> Result<Option<()>, String> {
    // the CRC-16 covers the whole frame INCLUDING the sync code (dr_flac
    // resets its CRC at find_and_seek_to_next_sync_code, i.e. the frame's
    // first byte) — capture the position before the header read consumes it
    let frame_start_byte = bs.pos / 8;
    let Some(header) = flac_read_frame_header(bs, si)? else {
        return Ok(None);
    };

    let channel_count: usize = match header.channel_assignment {
        0 => 1,
        c @ 1..=7 => c as usize,
        8..=10 => 2,
        _ => return Err("FLAC channel assignment > 10".into()),
    };
    if channel_count != si.channels as usize {
        return Err("FLAC frame channel count disagrees with STREAMINFO".into());
    }

    // subframe wasted bits, for the left-alignment shifts
    let mut wasted = [0u32; 8];
    let mut subframes: Vec<Vec<i32>> = Vec::with_capacity(channel_count);
    for ch in 0..channel_count {
        let mut sub = Vec::new();
        wasted[ch] = flac_decode_subframe(bs, &header, ch, &mut sub)?;
        subframes.push(sub);
    }

    // byte alignment padding + CRC-16
    let pad_bits = ((8 - (bs.pos % 8)) % 8) as u32;
    if pad_bits > 0 {
        if bs.read_u32(pad_bits).is_none() {
            return Err("FLAC stream truncated in frame padding".into());
        }
    }
    let Some(stored_crc16) = bs.read_u32(16) else {
        return Err("FLAC stream truncated in frame CRC".into());
    };
    if bs.pos % 8 != 0 {
        return Err("FLAC frame CRC not byte aligned".into());
    }
    let frame_end = bs.pos / 8 - 2;
    let crc = flac_crc16(0, &bs.data[frame_start_byte..frame_end]);
    if crc as u32 != stored_crc16 {
        // dr_flac drops frames with a bad CRC-16
        return Ok(Some(()));
    }

    let unused = 32 - header.bits_per_sample as u32;
    let n = header.block_size as usize;
    match header.channel_assignment {
        0..=7 => {
            // independent channels: interleaved, each left-aligned by its own
            // unused + wasted shift (dr_flac read_pcm_frames_s32 general path)
            for i in 0..n {
                for ch in 0..channel_count {
                    let shift = unused + wasted[ch];
                    out.push(((subframes[ch][i] as u32) << shift) as i32);
                }
            }
        }
        8 => {
            // left/side (dr_flac __decode_left_side)
            for i in 0..n {
                let left = (subframes[0][i] as u32) << (unused + wasted[0]);
                let side = (subframes[1][i] as u32) << (unused + wasted[1]);
                let right = left.wrapping_sub(side);
                out.push(left as i32);
                out.push(right as i32);
            }
        }
        9 => {
            // right/side
            for i in 0..n {
                let side = (subframes[0][i] as u32) << (unused + wasted[0]);
                let right = (subframes[1][i] as u32) << (unused + wasted[1]);
                let left = right.wrapping_add(side);
                out.push(left as i32);
                out.push(right as i32);
            }
        }
        10 => {
            // mid/side (dr_flac __decode_mid_side)
            for i in 0..n {
                let mid = (subframes[0][i] as u32) << wasted[0];
                let side = (subframes[1][i] as u32) << wasted[1];
                let mid2 = (mid << 1) | (side & 0x01);
                let l = ((((mid2.wrapping_add(side)) as i32) >> 1) as u32) << unused;
                let r = ((((mid2.wrapping_sub(side)) as i32) >> 1) as u32) << unused;
                out.push(l as i32);
                out.push(r as i32);
            }
        }
        _ => unreachable!(),
    }
    Ok(Some(()))
}

/// decode a FLAC buffer to mono f32 at `target_sample_rate` — the port side
/// of the reference's miniaudio drflac path (mtmd-helper.cpp:325).
pub fn audio_from_flac_bytes(bytes: &[u8], target_sample_rate: i32) -> Result<Vec<f32>, String> {
    if bytes.len() < 8 {
        return Err("FLAC file too short".into());
    }
    // metadata blocks: "fLaC" + STREAMINFO (type 0) required first
    let mut pos = 4usize;
    let mut si: Option<FlacStreamInfo> = None;
    loop {
        if pos + 4 > bytes.len() {
            return Err("FLAC metadata truncated".into());
        }
        let last = bytes[pos] & 0x80 != 0;
        let btype = bytes[pos] & 0x7F;
        let len = ((bytes[pos + 1] as usize) << 16)
            | ((bytes[pos + 2] as usize) << 8)
            | bytes[pos + 3] as usize;
        pos += 4;
        if pos + len > bytes.len() {
            return Err("FLAC metadata block overruns the file".into());
        }
        if btype == 0 && si.is_none() {
            si = Some(flac_parse_streaminfo(&bytes[pos..pos + len])?);
        }
        pos += len;
        if last {
            break;
        }
    }
    let si = si.ok_or("FLAC file has no STREAMINFO")?;
    if si.sample_rate != target_sample_rate as u32 {
        return Err(format!(
            "sample-rate conversion is not supported (file {} Hz, projector {} Hz); \
             the reference resamples via miniaudio",
            si.sample_rate, target_sample_rate
        ));
    }
    let channels = si.channels as usize;

    // decode all frames
    let mut bs = FlacBits::new(&bytes[pos.min(bytes.len())..]);
    let mut interleaved: Vec<i32> = Vec::new();
    loop {
        match flac_decode_frame(&mut bs, &si, &mut interleaved)? {
            Some(()) => {}
            None => break,
        }
        if si.total_samples > 0 && interleaved.len() as u64 >= si.total_samples * channels as u64 {
            break;
        }
    }
    if si.total_samples > 0 {
        let want = si.total_samples as usize * channels;
        if interleaved.len() < want {
            return Err(format!(
                "FLAC decoded {} frames, STREAMINFO promises {}",
                interleaved.len() / channels,
                si.total_samples
            ));
        }
        interleaved.truncate(want);
    }

    // s32 -> f32 (miniaudio ma_pcm_s32_to_f32__reference) then the mono mix
    // (ma_channel_converter mono_out, f32 case) — the WAV reader's recipe
    let n_frames = interleaved.len() / channels;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut acc = 0.0f32;
        for c in 0..channels {
            let v = interleaved[f * channels + c];
            acc += (v as f64 / 2147483648.0) as f32;
        }
        out.push(acc / channels as f32);
    }
    Ok(out)
}

/// `audio_helpers::decode_audio_from_buf` (mtmd-helper.cpp:325-362) for
/// uncompressed WAV input: decode to mono f32 at `target_sample_rate`.
///
/// **Not ported, deliberately** (see PARITY.md): the reference decodes MP3
/// (miniaudio's built-in mp1/mp2/mp3 decoder, miniaudio.h `ma_mp3_*` /
/// `ma_decoder_init_memory` with `ma_encoding_format_mp3`) and FLAC
/// (`ma_flac_*` / `ma_encoding_format_flac`) and RESAMPLES to the target
/// rate (`ma_decoder_config_init(.., target_sample_rate)` +
/// `ma_linear_resampler`). Reproducing bit-exact PCM from compressed input
/// without the vendored miniaudio is out of scope for the port; the port
/// fails LOUDLY on those formats instead of silently mis-decoding.
pub fn audio_from_wav_bytes(bytes: &[u8], target_sample_rate: i32) -> Result<Vec<f32>, String> {
    if !is_wav_file(bytes) {
        // name the format explicitly when the port can sniff it (the same
        // sniffers the reference's is_audio_file uses)
        if is_mp3_file(bytes) {
            return Err(
                "MP3 audio is not supported by this port (the reference decodes it via \
                 miniaudio's built-in mp1/mp2/mp3 decoder, mtmd-helper.cpp:325 \
                 ma_decoder_init_memory + miniaudio.h drmp3); convert to 16 kHz mono WAV first"
                    .into(),
            );
        }
        if is_flac_file(bytes) {
            return audio_from_flac_bytes(bytes, target_sample_rate);
        }
        return Err("not a RIFF/WAVE file".into());
    }
    let riff_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let end = (8 + riff_len).min(bytes.len());

    let mut pos = 12usize;
    let mut format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0i32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= end {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + len).min(end);
        match id {
            b"fmt " => {
                if body_end - body_start < 16 {
                    return Err("malformed fmt chunk".into());
                }
                let rd16 = |o: usize| {
                    u16::from_le_bytes([bytes[body_start + o], bytes[body_start + o + 1]])
                };
                let rd32 = |o: usize| {
                    u32::from_le_bytes([
                        bytes[body_start + o],
                        bytes[body_start + o + 1],
                        bytes[body_start + o + 2],
                        bytes[body_start + o + 3],
                    ])
                };
                format = rd16(0);
                channels = rd16(2);
                sample_rate = rd32(4) as i32;
                bits = rd16(14);
                // WAVE_FORMAT_EXTENSIBLE: the real format is in the subformat GUID
                if format == 0xFFFE && body_end - body_start >= 40 {
                    format = rd16(24);
                }
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        pos = body_start + len + (len & 1); // chunks are word-aligned
    }

    let data = data.ok_or("WAV has no data chunk")?;
    if channels == 0 {
        return Err("WAV has no channels".into());
    }
    if sample_rate != target_sample_rate {
        return Err(format!(
            "sample-rate conversion is not supported (file {} Hz, projector {} Hz); \
             the reference resamples via miniaudio",
            sample_rate, target_sample_rate
        ));
    }

    // decode one interleaved frame to f32, then mix to mono (average)
    let bytes_per_sample = (bits as usize + 7) / 8;
    let n_frames = data.len() / (bytes_per_sample * channels as usize);
    let mut out = Vec::with_capacity(n_frames);
    let decode = |b: &[u8]| -> Result<f32, String> {
        Ok(match (format, bits) {
            (0x0001, 8) => (b[0] as f32 - 128.0) / 128.0, // unsigned 8-bit PCM
            (0x0001, 16) => {
                let v = i16::from_le_bytes([b[0], b[1]]);
                v as f32 / 32768.0
            }
            (0x0001, 24) => {
                let v = ((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16)) << 8 >> 8;
                v as f32 / 8388608.0
            }
            (0x0001, 32) => {
                let v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                v as f32 / 2147483648.0
            }
            (0x0003, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (0x0003, 64) => {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            }
            _ => return Err(format!("unsupported WAV format {format:#x}/{bits} bits")),
        })
    };

    for f in 0..n_frames {
        let mut acc = 0.0f32;
        for c in 0..channels as usize {
            let off = (f * channels as usize + c) * bytes_per_sample;
            acc += decode(&data[off..off + bytes_per_sample])?;
        }
        out.push(acc / channels as f32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------

fn png_decode(bytes: &[u8]) -> Result<ClipImageU8, String> {
    let mut pos = 8usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut interlace = 0u8;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let data = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or("truncated PNG chunk")?;
        match kind {
            b"IHDR" => {
                width = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
                height = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
                bit_depth = data[8];
                color_type = data[9];
                interlace = data[12];
            }
            b"PLTE" => {
                palette = data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            }
            b"tRNS" => trns = data.to_vec(),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + len; // length + type + data + crc
    }
    if width == 0 || height == 0 {
        return Err("PNG has no IHDR".into());
    }
    if interlace != 0 {
        return Err("interlaced PNG is not supported".into());
    }
    if bit_depth != 8 && bit_depth != 16 {
        return Err(format!("unsupported PNG bit depth {bit_depth}"));
    }
    let channels = match color_type {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        other => return Err(format!("unsupported PNG colour type {other}")),
    };
    let sample_bytes = if bit_depth == 16 { 2 } else { 1 };
    let bpp = channels * sample_bytes;
    let raw = inflate_zlib(&idat)?;
    let stride = width * bpp;
    let expected = height * (stride + 1);
    if raw.len() < expected {
        return Err(format!("PNG IDAT too short: {} < {expected}", raw.len()));
    }

    // unfilter (PNG spec §9.2)
    let mut img = vec![0u8; height * stride];
    for y in 0..height {
        let ft = raw[y * (stride + 1)];
        let src = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        let (prev, cur) = img.split_at_mut(y * stride);
        let prev = if y == 0 {
            &[][..]
        } else {
            &prev[(y - 1) * stride..]
        };
        for x in 0..stride {
            let a = if x >= bpp { cur[x - bpp] } else { 0 };
            let b = prev.get(x).copied().unwrap_or(0);
            let c = if x >= bpp {
                prev.get(x - bpp).copied().unwrap_or(0)
            } else {
                0
            };
            cur[x] = match ft {
                0 => src[x],
                1 => src[x].wrapping_add(a),
                2 => src[x].wrapping_add(b),
                3 => src[x].wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => src[x].wrapping_add(paeth(a, b, c)),
                other => return Err(format!("unsupported PNG filter {other}")),
            };
        }
    }

    // to RGB8
    let mut out = ClipImageU8::new(width as i32, height as i32);
    for y in 0..height {
        for x in 0..width {
            let p = &img[y * stride + x * bpp..];
            let s = |i: usize| -> u8 {
                // 16-bit samples are reduced to their high byte (stb's default
                // 8-bit output does the same)
                p[i * sample_bytes]
            };
            let rgb = match color_type {
                0 => [s(0), s(0), s(0)],
                2 => [s(0), s(1), s(2)],
                3 => {
                    let idx = s(0) as usize;
                    *palette.get(idx).ok_or("PNG palette index out of range")?
                }
                4 => [s(0), s(0), s(0)],
                6 => [s(0), s(1), s(2)],
                _ => unreachable!(),
            };
            out.set_pixel(x as i32, y as i32, rgb);
        }
    }
    let _ = trns;
    Ok(out)
}

/// PNG spec §9.4 Paeth predictor
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p - a as i16).abs();
    let pb = (p - b as i16).abs();
    let pc = (p - c as i16).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

// ---------------------------------------------------------------------
// zlib / DEFLATE (RFC 1950/1951) — only what PNG needs
// ---------------------------------------------------------------------

fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 6 {
        return Err("zlib stream too short".into());
    }
    let cmf = data[0];
    if cmf & 0x0F != 8 {
        return Err("unsupported zlib compression method".into());
    }
    if (data[0] as u16 * 256 + data[1] as u16) % 31 != 0 {
        return Err("bad zlib header checksum".into());
    }
    inflate(&data[2..data.len() - 4])
}

/// RFC 1951 §3.1.1: fields are consumed LSB-first (`val |= byte << bitcnt`,
/// the `bits()` of puff.c), while Huffman codes are walked one bit at a time —
/// so a single LSB-first reader serves both.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// number of valid bits held in `acc` (the low `bit` bits)
    bit: u32,
    acc: u64,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit: 0,
            acc: 0,
        }
    }
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        while self.bit < n {
            let b = *self.data.get(self.pos).ok_or("deflate: out of input")?;
            self.acc |= (b as u64) << self.bit;
            self.pos += 1;
            self.bit += 8;
        }
        let v = self.acc & ((1u64 << n) - 1);
        self.acc >>= n;
        self.bit -= n;
        Ok(v as u32)
    }
    fn align(&mut self) {
        let drop = self.bit % 8;
        self.acc >>= drop;
        self.bit -= drop;
    }
    fn take_bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        self.align();
        let mut out = Vec::with_capacity(n);
        // whole bytes left in the accumulator are the next bytes of the stream
        while self.bit >= 8 {
            out.push(self.acc as u8);
            self.acc >>= 8;
            self.bit -= 8;
        }
        if out.len() < n {
            let need = n - out.len();
            let end = (self.pos + need).min(self.data.len());
            out.extend_from_slice(&self.data[self.pos..end]);
            self.pos = end;
        }
        if out.len() != n {
            return Err(format!(
                "deflate: out of input (stored block): want {n}, got {}, pos {}, len {}",
                out.len(),
                self.pos,
                self.data.len()
            ));
        }
        Ok(out)
    }
}

struct Huffman {
    /// code length per symbol
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + counts[i - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Self { counts, symbols }
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, String> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("deflate: bad Huffman code".into())
    }
}

fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut br = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let final_block = br.bits(1)? == 1;
        match br.bits(2)? {
            0 => {
                // RFC 1951 §3.2.4: skip to the byte boundary first (puff.c
                // `bits(s, s->bitcnt & 7)`), then LEN/NLEN
                br.align();
                let n = br.bits(16)? as usize;
                let ninv = br.bits(16)? as usize;
                if n != (!ninv & 0xffff) {
                    return Err("deflate: stored block LEN/NLEN mismatch".into());
                }
                let bytes = br.take_bytes(n)?;
                out.extend_from_slice(&bytes);
            }
            1 => {
                // fixed Huffman tables (RFC 1951 §3.2.6)
                let mut lit_len = [0u8; 288];
                for (i, l) in lit_len.iter_mut().enumerate() {
                    *l = if i < 144 {
                        8
                    } else if i < 256 {
                        9
                    } else if i < 280 {
                        7
                    } else {
                        8
                    };
                }
                let lit = Huffman::new(&lit_len);
                let dist = Huffman::new(&[5u8; 30]);
                inflate_block(&mut br, &lit, &dist, &mut out)?;
            }
            2 => {
                let hlit = br.bits(5)? as usize + 257;
                let hdist = br.bits(5)? as usize + 1;
                let hclen = br.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut clen = [0u8; 19];
                for &o in ORDER.iter().take(hclen) {
                    clen[o] = br.bits(3)? as u8;
                }
                let clh = Huffman::new(&clen);
                let mut lengths = vec![0u8; hlit + hdist];
                let mut i = 0usize;
                while i < lengths.len() {
                    let sym = clh.decode(&mut br)?;
                    match sym {
                        0..=15 => {
                            lengths[i] = sym as u8;
                            i += 1;
                        }
                        16 => {
                            let prev = *lengths.get(i - 1).ok_or("deflate: repeat with no prev")?;
                            let n = 3 + br.bits(2)? as usize;
                            for _ in 0..n {
                                lengths[i] = prev;
                                i += 1;
                            }
                        }
                        17 => {
                            let n = 3 + br.bits(3)? as usize;
                            i += n;
                        }
                        18 => {
                            let n = 11 + br.bits(7)? as usize;
                            i += n;
                        }
                        _ => return Err("deflate: bad code-length symbol".into()),
                    }
                }
                let lit = Huffman::new(&lengths[..hlit]);
                let dist = Huffman::new(&lengths[hlit..]);
                inflate_block(&mut br, &lit, &dist, &mut out)?;
            }
            _ => return Err("deflate: bad block type".into()),
        }
        if final_block {
            break;
        }
    }
    Ok(out)
}

const LEN_BASE: [usize; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [usize; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn inflate_block(
    br: &mut BitReader<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    loop {
        let sym = lit.decode(br)? as usize;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let i = sym - 257;
                let len = LEN_BASE[i] + br.bits(LEN_EXTRA[i])? as usize;
                let dsym = dist.decode(br)? as usize;
                let d = DIST_BASE[dsym] + br.bits(DIST_EXTRA[dsym])? as usize;
                if d > out.len() {
                    return Err("deflate: distance beyond output".into());
                }
                let start = out.len() - d;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
            _ => return Err("deflate: bad literal/length symbol".into()),
        }
    }
}

// ---------------------------------------------------------------------
// BMP
// ---------------------------------------------------------------------

fn bmp_decode(bytes: &[u8]) -> Result<ClipImageU8, String> {
    if bytes.len() < 54 {
        return Err("BMP too short".into());
    }
    let data_off = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
    let header_size = u32::from_le_bytes(bytes[14..18].try_into().unwrap()) as usize;
    let width = i32::from_le_bytes(bytes[18..22].try_into().unwrap());
    let height = i32::from_le_bytes(bytes[22..26].try_into().unwrap());
    let planes = u16::from_le_bytes(bytes[26..28].try_into().unwrap());
    let bpp = u16::from_le_bytes(bytes[28..30].try_into().unwrap());
    let compression = u32::from_le_bytes(bytes[30..34].try_into().unwrap());
    if header_size < 40 || planes != 1 {
        return Err("unsupported BMP header".into());
    }
    if compression != 0 {
        return Err("compressed BMP is not supported".into());
    }
    if bpp != 24 && bpp != 32 {
        return Err(format!("unsupported BMP bit depth {bpp}"));
    }
    let top_down = height < 0;
    let (w, h) = (width as usize, height.unsigned_abs() as usize);
    let stride = ((w * (bpp as usize / 8) + 3) / 4) * 4;
    if data_off + stride * h > bytes.len() {
        return Err("BMP pixel data truncated".into());
    }
    let mut img = ClipImageU8::new(w as i32, h as i32);
    for y in 0..h {
        let src_y = if top_down { y } else { h - 1 - y };
        let row = &bytes[data_off + src_y * stride..];
        for x in 0..w {
            let p = &row[x * (bpp as usize / 8)..];
            img.set_pixel(x as i32, y as i32, [p[2], p[1], p[0]]); // BGR(A) -> RGB
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------------
// PPM / PGM (P6 / P5)
// ---------------------------------------------------------------------

fn ppm_decode(bytes: &[u8]) -> Result<ClipImageU8, String> {
    let mut fields: Vec<usize> = Vec::new();
    let mut pos = 2usize;
    while fields.len() < 3 {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if start == pos {
            return Err("bad PPM header".into());
        }
        fields.push(
            std::str::from_utf8(&bytes[start..pos])
                .map_err(|_| "bad PPM header")?
                .parse()
                .map_err(|_| "bad PPM header")?,
        );
    }
    pos += 1; // single whitespace after maxval
    let (w, h, maxval) = (fields[0], fields[1], fields[2]);
    if maxval > 255 {
        return Err("16-bit PPM is not supported".into());
    }
    let channels = if bytes[1] == b'6' { 3 } else { 1 };
    if pos + w * h * channels > bytes.len() {
        return Err("PPM pixel data truncated".into());
    }
    let mut img = ClipImageU8::new(w as i32, h as i32);
    for y in 0..h {
        for x in 0..w {
            let p = &bytes[pos + (y * w + x) * channels..];
            let rgb = if channels == 3 {
                [p[0], p[1], p[2]]
            } else {
                [p[0], p[0], p[0]]
            };
            img.set_pixel(x as i32, y as i32, rgb);
        }
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_PNG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/mtmd-fixture.png");
    const FIXTURE_BMP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/mtmd-fixture.bmp");
    const FIXTURE_PPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/mtmd-fixture.ppm");

    /// All three formats decode to the same RGB8 bitmap (parity/gen_mtmd_fixture.py).
    #[test]
    fn image_decoders_agree() {
        let Ok(png) = bitmap_from_file(FIXTURE_PNG) else {
            eprintln!("skipping: {FIXTURE_PNG} not present (run parity/gen_mtmd_fixture.py)");
            return;
        };
        assert_eq!((png.nx, png.ny), (320, 224));
        let bmp = bitmap_from_file(FIXTURE_BMP).expect("bmp");
        let ppm = bitmap_from_file(FIXTURE_PPM).expect("ppm");
        assert_eq!(bmp.buf, png.buf, "BMP pixels must match the PNG");
        assert_eq!(ppm.buf, png.buf, "PPM pixels must match the PNG");
        // spot checks against the generator's rectangles
        assert_eq!(png.get_pixel(50, 40), [255, 0, 0]);
        assert_eq!(png.get_pixel(150, 40), [0, 255, 0]);
        assert_eq!(png.get_pixel(260, 40), [0, 0, 255]);
        assert_eq!(png.get_pixel(80, 150), [255, 255, 255]);
        assert_eq!(png.get_pixel(200, 120), [0, 0, 0]);
        assert_eq!(png.get_pixel(280, 180), [255, 255, 0]);
    }

    /// DEFLATE round-trip against python's zlib (the PNG IDAT decoder).
    #[test]
    fn inflate_matches_zlib() {
        let Ok(z) = std::fs::read("/tmp/z_hello.bin") else {
            eprintln!("skipping: /tmp/z_hello.bin not present");
            return;
        };
        let raw = std::fs::read("/tmp/z_hello_raw.bin").unwrap();
        let got = inflate_zlib(&z).expect("inflate");
        assert_eq!(got, raw);
        for lvl in [0, 1, 6, 9] {
            let z = std::fs::read(format!("/tmp/z_lvl{lvl}.bin")).unwrap();
            let got = inflate_zlib(&z).unwrap_or_else(|e| panic!("level {lvl}: {e}"));
            assert_eq!(got, raw, "level {lvl}");
        }
    }
}

// ======================================================================
// audio round 5 — the GEN driver surface (mtmd.cpp:1867-2040) and the
// mtmd-helper-gen.cpp pipelines
// ======================================================================

/// `mtmd_gen_audio_type` (mtmd.h:370).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenAudioType {
    None,
    Qwen3Tts,
    PocketTts,
}

/// `mtmd_gen_audio_info` (mtmd.h:376) — `mtmd_gen_audio_get_info`
/// (mtmd.cpp:1871-1893).
pub struct GenAudioInfo {
    pub ty: GenAudioType,
    pub sample_rate: i32,
    /// `hparams.gen_model_variant` (clip.cpp:1342)
    pub model_variant: String,
}

/// `mtmd_helper_gen_audio_outtype` (mtmd-helper.h).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenAudioOutType {
    Pcm,
    Wav,
}

impl MtmdContext {
    /// the GEN calls' context — `ctx_gen_a` when a mixed file split it out,
    /// else the (single-modality gen) primary (mtmd.cpp:1874/:1922+)
    fn gen_clip(&self) -> &clip::ClipContext {
        self.clip_gen.as_ref().unwrap_or(&self.clip)
    }

    fn gen_clip_mut(&mut self) -> &mut clip::ClipContext {
        self.clip_gen.as_mut().unwrap_or(&mut self.clip)
    }

    /// `mtmd_gen_audio_get_info` (mtmd.cpp:1871-1893).
    pub fn gen_audio_info(&self) -> GenAudioInfo {
        // mtmd.cpp:1878 reads clip_get_hparams(ctx_gen_a)->gen_model_variant —
        // the HPARAMS field the loader fills (clip.cpp:1342), not a model
        // tensor-slot field. The pipeline's per-pack settings key on it
        // (pockettts_pack), so reading the empty model-level field silently
        // fell back to the default temperature — the pipeline-layer e2e
        // (tests/tts_pipeline_e2e.rs) caught it as a temp-0.7-vs-0.3 h-state
        // divergence from step 1.
        match self.gen_clip().model.proj_type {
            clip::ProjectorType::Qwen3TtsGen => GenAudioInfo {
                ty: GenAudioType::Qwen3Tts,
                sample_rate: 24000,
                model_variant: self.gen_clip().hparams().gen_model_variant.clone(),
            },
            clip::ProjectorType::PocketTtsGen => GenAudioInfo {
                ty: GenAudioType::PocketTts,
                sample_rate: 24000,
                model_variant: self.gen_clip().hparams().gen_model_variant.clone(),
            },
            _ => GenAudioInfo {
                ty: GenAudioType::None,
                sample_rate: 0,
                model_variant: String::new(),
            },
        }
    }

    /// `mtmd_gen_inp_default` (mtmd.cpp:1895-1920) — the per-pipeline
    /// sampling defaults; `seed = UINT32_MAX` (entropy) like the C.
    pub fn gen_inp_default(&self) -> clip::GenInp<'static> {
        let (top_k, top_p, temp) = self.gen_clip().gen_inp_defaults();
        clip::GenInp {
            process: clip::ClipGenProcess::GenCode,
            code0: 0,
            embd: None,
            top_k,
            top_p,
            seed: u32::MAX,
            temp,
            codes: None,
            feats: None,
            state: None,
        }
    }

    /// `mtmd_gen_audio_process` (mtmd.cpp:2033) — the GEN_CODE / GEN_WAV
    /// step over the loaded generator.
    pub fn gen_audio_process(&mut self, inp: &clip::GenInp) -> Result<clip::GenOut, String> {
        self.gen_clip_mut().gen_audio_process(inp)
    }
}

/// The `llama_context` surface the gen pipelines drive
/// (mtmd-helper-gen.cpp's `decode_embd_batch` + `llama_decode` +
/// `llama_get_embeddings_ith` + `llama_memory_seq_rm`).
///
/// [`DecodeContext`] implements this directly
/// ([`DecodeContext::decode_embd`] — `decode_embd_batch`'s tokens=nullptr
/// batch, mtmd-helper-common.h:73, on the vector-embeddings arm of
/// `build_inp_embd`, llama-graph.cpp:2387-2445); the MTP/eagle3/dflash hook
/// contexts decode their own embd batches through `decode_batch`
/// (speculative.cpp:1521-1547 / :687-696 / :1145-1156, the dflash.cpp:
/// 609-677 dual-mode decoder), and the pipeline tests' `DflashTalker`
/// mirrors that hook as a plain trait object.
pub trait GenTalker {
    /// `decode_embd_batch` + `llama_decode`: feed `n_tokens` F32 rows
    /// `[n_tokens][n_embd]` at position `pos`, return the last row's
    /// embeddings (`llama_get_embeddings_ith(-1)`).
    fn decode_embd(
        &mut self,
        embd: &[f32],
        n_tokens: usize,
        pos: i32,
        seq_id: i32,
    ) -> Result<Vec<f32>, String>;
    /// `llama_memory_seq_rm(seq_id, p0, p1)` — p1 = -1 keeps to the end.
    fn seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> Result<(), String>;
}

impl GenTalker for crate::context::DecodeContext {
    /// `decode_embd_batch` + `llama_decode` + `llama_get_embeddings_ith(-1)`
    /// over a plain trunk (mtmd-helper-common.h:73-105; the materialised-matrix
    /// equivalence of the C's embd input arm is documented on the method).
    fn decode_embd(
        &mut self,
        embd: &[f32],
        n_tokens: usize,
        pos: i32,
        seq_id: i32,
    ) -> Result<Vec<f32>, String> {
        crate::context::DecodeContext::decode_embd(self, embd, n_tokens, pos, seq_id)
    }
    /// `llama_memory_seq_rm` — the port's single-cache [`DecodeContext::seq_rm`]
    /// (llama-kv-cache.cpp:1201-1250) reports no failure mode.
    fn seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> Result<(), String> {
        crate::context::DecodeContext::seq_rm(self, seq_id, p0, p1);
        Ok(())
    }
}

/// mtmd-helper-gen.cpp:25-41 — `--tts-lang` codes → the codec_language
/// special-token names.
fn tts_resolve_lang(lang: &str) -> &str {
    match lang {
        "zh" => "chinese",
        "en" => "english",
        "de" => "german",
        "it" => "italian",
        "pt" => "portuguese",
        "es" => "spanish",
        "ja" => "japanese",
        "ko" => "korean",
        "fr" => "french",
        "ru" => "russian",
        other => other,
    }
}

/// mtmd-helper-gen.cpp:43-51 `find_special_token`.
fn find_special_token(vocab: &Vocab, piece: &str) -> Option<i32> {
    (0..vocab.n_tokens() as i32).find(|&t| vocab.token_to_piece(t) == piece)
}

/// mtmd-helper-gen.cpp:53-77 `write_wav16` — RIFF PCM16 mono.
fn write_wav16(pcm: &[f32], rate: i32) -> Vec<u8> {
    let data_sz = (pcm.len() * 2) as u32;
    let riff_sz = 36 + data_sz;
    let mut buf = Vec::with_capacity(44 + pcm.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_sz.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&(rate as u32).to_le_bytes());
    buf.extend_from_slice(&((rate as u32) * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_sz.to_le_bytes());
    for &v in pcm {
        let s = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

/// `mtmd_helper_gen_audio_set_input`'s input (mtmd-helper.h).
pub struct GenAudioInp<'a> {
    pub prompt: &'a str,
    /// `--tts-lang` code (qwen3tts)
    pub lang: &'a str,
    /// the voice-conditioning reference (raw audio bytes, the spkenc path)
    pub speaker_ref: Option<&'a [u8]>,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub out_type: GenAudioOutType,
}

/// mtmd-helper-gen.cpp:79-459 — the Qwen3-TTS pipeline: the backbone samples
/// codec_0, the code_predictor gives the other 15 codebooks, then code2wav
/// decodes them to PCM.
pub struct Qwen3TtsGenPipeline<T: GenTalker> {
    talker: T,
    mctx: MtmdContext,
    vocab: std::rc::Rc<Vocab>,
    n_embd: usize,
    info: GenAudioInfo,
    // vocab specials (ensure_cache, mtmd-helper-gen.cpp:331-365)
    codec_0: i32,
    codec_bos: i32,
    codec_eos: i32,
    codec_pad: i32,
    c_think: i32,
    c_think_b: i32,
    c_think_e: i32,
    tts_pad: i32,
    tts_bos: i32,
    tts_eos: i32,
    tok_embd: Vec<f32>,
    // must match hparams.wav_tfm_swa (mtmd-helper-gen.cpp:438)
    window_frames: usize,
    // per-generation state
    seq_id: i32,
    mrope: bool,
    pos: i32,
    prompt_rows: Vec<f32>,
    prompt_pos: usize,
    n_prompt: usize,
    top_k: i32,
    top_p: f32,
    seed: u32,
    out_type: GenAudioOutType,
    codes_buf: Vec<i32>,
    c2w_state: Vec<u8>,
    audio_pcm: Vec<f32>,
    overlay: Vec<f32>,
}

impl<T: GenTalker> Qwen3TtsGenPipeline<T> {
    /// `mtmd_helper_gen_audio_init` + `ensure_cache` (mtmd-helper-gen.cpp:
    /// 1011-1015 / :331-365): the special-token table and the whole
    /// token-embedding matrix are read once.
    pub fn new(
        talker: T,
        mctx: MtmdContext,
        vocab: std::rc::Rc<Vocab>,
        tok_embd: Vec<f32>,
        n_embd: usize,
        use_mrope: bool,
    ) -> Result<Self, String> {
        let info = mctx.gen_audio_info();
        if info.ty != GenAudioType::Qwen3Tts {
            return Err("not a qwen3-tts generator".to_string());
        }
        let f = |name: &str| {
            find_special_token(&vocab, name)
                .ok_or_else(|| format!("missing special token {name} in vocab"))
        };
        let codec_0 = f("<|codec_0|>")?;
        let codec_bos = f("<|codec_bos|>")?;
        let codec_eos = f("<|codec_eos_token|>")?;
        let codec_pad = f("<|codec_pad|>")?;
        let c_think = f("<|codec_think|>")?;
        let c_think_b = f("<|codec_think_bos|>")?;
        let c_think_e = f("<|codec_think_eos|>")?;
        let tts_pad = f("<tts_pad>")?;
        let tts_bos = f("<tts_text_bos>")?;
        let tts_eos = f("<tts_text_eod>")?;
        if tok_embd.is_empty() || tok_embd.len() % n_embd != 0 {
            return Err("token embedding table does not match n_embd".to_string());
        }
        Ok(Self {
            talker,
            mctx,
            vocab,
            n_embd,
            info,
            codec_0,
            codec_bos,
            codec_eos,
            codec_pad,
            c_think,
            c_think_b,
            c_think_e,
            tts_pad,
            tts_bos,
            tts_eos,
            tok_embd,
            // hparams.wav_tfm_swa (clip.cpp:1861 hardcodes 72)
            window_frames: 72,
            seq_id: 0,
            mrope: use_mrope,
            pos: 0,
            prompt_rows: Vec::new(),
            prompt_pos: 0,
            n_prompt: 0,
            top_k: 50,
            top_p: 1.0,
            seed: u32::MAX,
            out_type: GenAudioOutType::Wav,
            codes_buf: Vec::new(),
            c2w_state: Vec::new(),
            audio_pcm: Vec::new(),
            overlay: Vec::new(),
        })
    }

    fn row(&self, t: i32) -> Vec<f32> {
        self.tok_embd[t as usize * self.n_embd..(t as usize + 1) * self.n_embd].to_vec()
    }

    fn sum_row(&self, a: i32, b: i32) -> Vec<f32> {
        let mut va = self.row(a);
        let vb = self.row(b);
        for (x, y) in va.iter_mut().zip(vb) {
            *x += y;
        }
        va
    }

    /// `set_input` (mtmd-helper-gen.cpp:126-217): the upstream chat wrap +
    /// the tts_pad/codec_pad embedding sums.
    pub fn set_input(&mut self, inp: &GenAudioInp, speaker_embd: Option<&[f32]>) -> Result<(), String> {
        self.reset();
        self.seq_id = 0;

        let lang = tts_resolve_lang(if inp.lang.is_empty() { "english" } else { inp.lang });
        let c_lang = find_special_token(&self.vocab, &format!("<|codec_language_{lang}|>"))
            .ok_or_else(|| format!("unknown language '{lang}'"))?;

        // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
        // (mtmd-helper-gen.cpp:164-174)
        let full = format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", inp.prompt);
        let ids = self.vocab.tokenize(&full, false, true);
        if ids.len() < 8 {
            return Err("tokenization failed".to_string());
        }

        let mut prompt: Vec<Vec<f32>> = Vec::new();
        for &t in &ids[..3] {
            prompt.push(self.row(t));
        }
        prompt.push(self.sum_row(self.tts_pad, self.c_think));
        prompt.push(self.sum_row(self.tts_pad, self.c_think_b));
        prompt.push(self.sum_row(self.tts_pad, c_lang));
        prompt.push(self.sum_row(self.tts_pad, self.c_think_e));
        if let Some(sp) = speaker_embd {
            if sp.len() != self.n_embd {
                return Err("speaker embedding width mismatch".to_string());
            }
            let mut v = self.row(self.tts_pad);
            for (x, y) in v.iter_mut().zip(sp) {
                *x += y;
            }
            prompt.push(v);
        }
        prompt.push(self.sum_row(self.tts_bos, self.codec_pad));
        for &t in &ids[3..ids.len() - 5] {
            prompt.push(self.sum_row(t, self.codec_pad));
        }
        prompt.push(self.sum_row(self.tts_eos, self.codec_pad));
        prompt.push(self.sum_row(self.tts_pad, self.codec_bos));

        self.n_prompt = prompt.len();
        self.prompt_rows = prompt.into_iter().flatten().collect();
        self.prompt_pos = 0;
        self.pos = 0;

        let def = self.mctx.gen_inp_default();
        self.top_k = if inp.top_k > 0 { inp.top_k } else { def.top_k };
        self.top_p = if inp.top_p > 0.0 { inp.top_p } else { def.top_p };
        self.seed = inp.seed;
        self.out_type = inp.out_type;

        // every generated frame adds tts_pad on top of the codes embedding
        // (mtmd-helper-gen.cpp:212-214)
        self.overlay = self.row(self.tts_pad);
        Ok(())
    }

    /// `reset` (mtmd-helper-gen.cpp:111-124).
    pub fn reset(&mut self) {
        self.seq_id = 0;
        self.pos = 0;
        self.codes_buf.clear();
        self.c2w_state.clear();
        self.audio_pcm.clear();
        self.overlay.clear();
        self.prompt_rows.clear();
        self.prompt_pos = 0;
        self.n_prompt = 0;
    }

    /// `step_prompt` (mtmd-helper-gen.cpp:219-247) — decodes at most
    /// `n_batch` prompt rows; returns the remaining count.
    pub fn step_prompt(&mut self, n_batch: usize) -> Result<usize, String> {
        assert!(n_batch > 0);
        if self.prompt_pos >= self.n_prompt {
            return Ok(0);
        }
        let n = std::cmp::min(n_batch, self.n_prompt - self.prompt_pos);
        let start = self.prompt_pos * self.n_embd;
        let end = start + n * self.n_embd;
        // a plain text/codec stream: all mrope sections equal, so the
        // port's 1d position stream covers it (the C passes
        // set_position_mrope_1d for mrope models — same positions)
        let embd: Vec<f32> = self.prompt_rows[start..end].to_vec();
        self.talker
            .decode_embd(&embd, n, self.pos, self.seq_id)?;
        self.pos += n as i32;
        self.prompt_pos += n;
        Ok(self.n_prompt - self.prompt_pos)
    }

    /// `step_gen` (mtmd-helper-gen.cpp:249-302) — one frame from the
    /// backbone's `sampled` token + `h_state`; returns (h_state_out, stop).
    #[allow(clippy::type_complexity)]
    pub fn step_gen(
        &mut self,
        sampled: i32,
        h_state_in: &[f32],
    ) -> Result<(Option<Vec<f32>>, bool), String> {
        // backbone signals end-of-speech with a token, no frame for this step
        if sampled == self.codec_eos || self.vocab.is_eog(sampled) {
            return Ok((None, true));
        }

        let mut inp = self.mctx.gen_inp_default();
        inp.process = clip::ClipGenProcess::GenCode;
        inp.code0 = sampled - self.codec_0;
        inp.embd = Some(h_state_in);
        inp.top_k = self.top_k;
        inp.top_p = self.top_p;
        inp.seed = self.seed;
        let out = self.mctx.gen_audio_process(&inp)?;

        self.codes_buf.extend_from_slice(&out.codes);
        if !out.codes.is_empty() && self.codes_buf.len() / out.codes.len() >= self.window_frames {
            self.flush_gen_wav()?;
        }

        // the feedback embedding + the tts_pad overlay
        // (mtmd-helper-gen.cpp:282-283)
        let mut fb = out.embd.clone();
        if fb.len() != self.n_embd {
            return Err("gen_code embd width mismatch".to_string());
        }
        for (x, y) in fb.iter_mut().zip(&self.overlay) {
            *x += y;
        }

        let pos = self.pos;
        self.pos += 1;
        let h = self.talker.decode_embd(&fb, 1, pos, self.seq_id)?;
        Ok((Some(h), false))
    }

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:400-421) — one GEN_WAV call over
    /// the buffered codes, state carried across batches.
    fn flush_gen_wav(&mut self) -> Result<(), String> {
        if self.codes_buf.is_empty() {
            return Ok(());
        }
        let mut inp = self.mctx.gen_inp_default();
        inp.process = clip::ClipGenProcess::GenWav;
        inp.codes = Some(&self.codes_buf);
        // same seed as gen_code, else clip reseeds mid-generation
        inp.seed = self.seed;
        let state = std::mem::take(&mut self.c2w_state);
        inp.state = if state.is_empty() { None } else { Some(&state) };
        let out = self.mctx.gen_audio_process(&inp)?;
        self.audio_pcm.extend_from_slice(&out.audio);
        self.c2w_state = out.state;
        self.codes_buf.clear();
        Ok(())
    }

    /// `get_output` (mtmd-helper-gen.cpp:304-328).
    pub fn get_output(&mut self) -> Result<(i32, Vec<u8>), String> {
        self.flush_gen_wav()?;
        let rate = self.info.sample_rate;
        match self.out_type {
            GenAudioOutType::Pcm => Ok((
                rate,
                self.audio_pcm
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )),
            GenAudioOutType::Wav => Ok((rate, write_wav16(&self.audio_pcm, rate))),
        }
    }

    pub fn audio_pcm(&self) -> &[f32] {
        &self.audio_pcm
    }

    /// driver introspection (tts.cpp's loop needs the prompt row count and
    /// the trunk's last hidden state) — the pipeline-layer e2e's accessors
    pub fn n_prompt_rows(&self) -> usize {
        self.n_prompt
    }

    pub fn talker_ref(&self) -> &T {
        &self.talker
    }
}

/// mtmd-helper-gen.cpp:461-484 — the per-pack yaml settings that live
/// outside the checkpoint.
#[derive(Clone, Copy)]
pub struct PocketttsPackSettings {
    pub temp: f32,
    pub frames_after_eos: i32,
    pub pad_short_text: bool,
}

fn pockettts_pack(variant: &str) -> PocketttsPackSettings {
    match variant {
        "english" => PocketttsPackSettings { temp: 0.3, frames_after_eos: 0, pad_short_text: false },
        "english_2026-01" => PocketttsPackSettings { temp: 0.7, frames_after_eos: 0, pad_short_text: true },
        "english_2026-04" => PocketttsPackSettings { temp: 0.3, frames_after_eos: 0, pad_short_text: false },
        "french_24l" => PocketttsPackSettings { temp: 0.7, frames_after_eos: 8, pad_short_text: false },
        _ => PocketttsPackSettings { temp: 0.0, frames_after_eos: 0, pad_short_text: false },
    }
}

/// mtmd-helper-gen.cpp:488-994 — the pocket-tts pipeline: the backbone emits
/// no token, the flow net turns each hidden state into a latent, the
/// end-of-speech head lives in the mmproj.
pub struct PocketttsGenPipeline<T: GenTalker> {
    talker: T,
    mctx: MtmdContext,
    vocab: std::rc::Rc<Vocab>,
    n_embd: usize,
    info: GenAudioInfo,
    pack: PocketttsPackSettings,
    bos_before_voice: Option<i32>,
    audio_bos: i32,
    tok_embd: Vec<f32>,
    seq_id: i32,
    pos: i32,
    prompt_rows: Vec<f32>,
    prompt_pos: usize,
    n_prompt: usize,
    seed: u32,
    out_type: GenAudioOutType,
    step_idx: i32,
    eos_step: i32,
    frames_after_eos: i32,
    // MAX_TOKEN_PER_CHUNK (mtmd-helper-gen.cpp:979)
    max_chunk_tokens: usize,
    frame_rate: f64,
    chunks: Vec<Vec<i32>>,
    chunk_idx: usize,
    n_voice_pos: i32,
    chunk_budget: i32,
    // latents are decoded a window at a time (mtmd-helper-gen.cpp:987)
    window_frames: usize,
    feats_buf: Vec<f32>,
    dec_state: Vec<u8>,
    audio_pcm: Vec<f32>,
}

impl<T: GenTalker> PocketttsGenPipeline<T> {
    pub fn new(
        talker: T,
        mctx: MtmdContext,
        vocab: std::rc::Rc<Vocab>,
        tok_embd: Vec<f32>,
        n_embd: usize,
    ) -> Result<Self, String> {
        let info = mctx.gen_audio_info();
        if info.ty != GenAudioType::PocketTts {
            return Err("not a pocket-tts generator".to_string());
        }
        let audio_bos = find_special_token(&vocab, "<|audio_bos|>")
            .ok_or("missing <|audio_bos|> in vocab")?;
        let pack = pockettts_pack(&info.model_variant);
        let bos_before_voice = find_special_token(&vocab, "<|bos_before_voice|>");
        if tok_embd.is_empty() || tok_embd.len() % n_embd != 0 {
            return Err("token embedding table does not match n_embd".to_string());
        }
        Ok(Self {
            talker,
            mctx,
            vocab,
            n_embd,
            info,
            pack,
            bos_before_voice,
            audio_bos,
            tok_embd,
            seq_id: 0,
            pos: 0,
            prompt_rows: Vec::new(),
            prompt_pos: 0,
            n_prompt: 0,
            seed: u32::MAX,
            out_type: GenAudioOutType::Wav,
            step_idx: 0,
            eos_step: -1,
            frames_after_eos: 3,
            max_chunk_tokens: 50,
            frame_rate: 12.5,
            chunks: Vec::new(),
            chunk_idx: 0,
            n_voice_pos: 0,
            chunk_budget: 0,
            window_frames: 8,
            feats_buf: Vec::new(),
            dec_state: Vec::new(),
            audio_pcm: Vec::new(),
        })
    }

    fn push_embd_row(&mut self, t: i32) {
        let row = &self.tok_embd[t as usize * self.n_embd..(t as usize + 1) * self.n_embd];
        self.prompt_rows.extend_from_slice(row);
    }

    /// `prepare_text` (mtmd-helper-gen.cpp:861-890) — the reference's text
    /// normalization, it affects quality.
    fn prepare_text(input: &str, pad_short: bool) -> String {
        let mut s = String::with_capacity(input.len() + 1);
        for c in input.chars() {
            match c {
                '\n' | '\r' => s.push(' '),
                ';' => s.push(','),
                _ => s.push(c),
            }
        }
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        let mut s = trimmed.to_string();
        let first = s.chars().next().unwrap();
        if first.is_ascii_lowercase() {
            let up = first.to_ascii_uppercase();
            s.replace_range(..first.len_utf8(), &up.to_string());
        }
        let last = s.chars().last().unwrap();
        if last.is_ascii_alphanumeric() {
            s.push('.');
        }
        if pad_short && Self::count_words(&s) < 5 {
            s = format!("{}{}", " ".repeat(8), s);
        }
        s
    }

    fn count_words(s: &str) -> i32 {
        let mut n = 0;
        let mut in_word = false;
        for c in s.chars() {
            if c == ' ' {
                in_word = false;
            } else if !in_word {
                in_word = true;
                n += 1;
            }
        }
        n
    }

    /// the token ids of the pieces the reference splits on
    /// (mtmd-helper-gen.cpp:730-737) — the leading dummy-prefix token drops
    fn punct_ids(&self, s: &str) -> Vec<i32> {
        let ids = self.vocab.tokenize(s, false, false);
        if ids.len() <= 1 {
            return Vec::new();
        }
        ids[1..].to_vec()
    }

    /// `split_on` (mtmd-helper-gen.cpp:740-755) — cut after runs of boundary
    /// tokens so punctuation stays with the sentence it ends.
    fn split_on(ids: &[i32], boundary: &[i32]) -> Vec<Vec<i32>> {
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut prev_was_boundary = false;
        for (i, &id) in ids.iter().enumerate() {
            let is_boundary = boundary.contains(&id);
            if !is_boundary && prev_was_boundary {
                out.push(ids[start..i].to_vec());
                start = i;
            }
            prev_was_boundary = is_boundary;
        }
        out.push(ids[start..].to_vec());
        out
    }

    /// `split_chunks` (mtmd-helper-gen.cpp:757-802).
    fn split_chunks(&self, ids: &[i32]) -> Vec<Vec<i32>> {
        if ids.len() <= self.max_chunk_tokens {
            return vec![ids.to_vec()];
        }
        let eos_punct = self.punct_ids(".!...?");
        let mid_punct = self.punct_ids(",;:");

        // oversized sentences split again on weaker punctuation
        let mut segments: Vec<Vec<i32>> = Vec::new();
        for seg in Self::split_on(ids, &eos_punct) {
            if seg.len() <= self.max_chunk_tokens {
                segments.push(seg);
                continue;
            }
            let sub = Self::split_on(&seg, &mid_punct);
            if sub.len() > 1 {
                segments.extend(sub);
            } else {
                segments.push(seg);
            }
        }

        let mut out: Vec<Vec<i32>> = Vec::new();
        for seg in segments {
            if seg.is_empty() {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.len() + seg.len() <= self.max_chunk_tokens => {
                    last.extend_from_slice(&seg);
                }
                _ => out.push(seg),
            }
        }
        if out.is_empty() {
            out.push(ids.to_vec());
        }
        out
    }

    /// `arm_chunk_budget` (mtmd-helper-gen.cpp:805-812).
    fn arm_chunk_budget(&mut self, idx: usize) {
        let n_tok = self.chunks[idx].len();
        self.chunk_budget = ((n_tok as f64 / 3.0 + 2.0) * self.frame_rate).ceil() as i32;
        self.frames_after_eos = if self.pack.frames_after_eos > 0 {
            self.pack.frames_after_eos
        } else if n_tok <= 6 {
            5
        } else {
            3
        };
        self.step_idx = 0;
        self.eos_step = -1;
    }

    /// `set_input` (mtmd-helper-gen.cpp:512-581): voice conditioning rows,
    /// the first chunk's text, the audio BOS.
    pub fn set_input(&mut self, inp: &GenAudioInp, voice_rows: Option<&[f32]>) -> Result<(), String> {
        self.reset();
        self.seq_id = 0;

        self.pack = pockettts_pack(&self.info.model_variant);

        let text = Self::prepare_text(inp.prompt, self.pack.pad_short_text);
        if text.is_empty() {
            return Err("empty prompt".to_string());
        }
        let ids = self.vocab.tokenize(&text, false, false);
        if ids.is_empty() {
            return Err("tokenization failed".to_string());
        }

        // long inputs degrade badly, each chunk restarts from the voice
        // conditioning (mtmd-helper-gen.cpp:545-551)
        self.chunks = self.split_chunks(&ids);
        self.chunk_idx = 0;

        self.prompt_rows.clear();
        if let Some(voice) = voice_rows {
            if voice.len() % self.n_embd != 0 {
                return Err("voice conditioning width mismatch".to_string());
            }
            if let Some(b) = self.bos_before_voice {
                self.push_embd_row(b);
            }
            self.prompt_rows.extend_from_slice(voice);
        }
        // every later chunk rewinds to here and re-prompts
        self.n_voice_pos = (self.prompt_rows.len() / self.n_embd) as i32;

        let chunk0 = self.chunks[0].clone();
        for &t in &chunk0 {
            self.push_embd_row(t);
        }
        self.push_embd_row(self.audio_bos);
        self.arm_chunk_budget(0);

        self.n_prompt = self.prompt_rows.len() / self.n_embd;
        self.prompt_pos = 0;
        self.pos = 0;
        self.seed = inp.seed;
        self.out_type = inp.out_type;
        Ok(())
    }

    /// `reset` (mtmd-helper-gen.cpp:492-510).
    pub fn reset(&mut self) {
        self.seq_id = 0;
        self.pos = 0;
        self.feats_buf.clear();
        self.dec_state.clear();
        self.audio_pcm.clear();
        self.prompt_rows.clear();
        self.prompt_pos = 0;
        self.n_prompt = 0;
        self.step_idx = 0;
        self.eos_step = -1;
        self.chunks.clear();
        self.chunk_idx = 0;
        self.n_voice_pos = 0;
        self.chunk_budget = 0;
    }

    /// `step_prompt` (mtmd-helper-gen.cpp:583-609).
    pub fn step_prompt(&mut self, n_batch: usize) -> Result<usize, String> {
        assert!(n_batch > 0);
        if self.prompt_pos >= self.n_prompt {
            return Ok(0);
        }
        let n = std::cmp::min(n_batch, self.n_prompt - self.prompt_pos);
        let start = self.prompt_pos * self.n_embd;
        let end = start + n * self.n_embd;
        let embd: Vec<f32> = self.prompt_rows[start..end].to_vec();
        self.talker.decode_embd(&embd, n, self.pos, self.seq_id)?;
        self.pos += n as i32;
        self.prompt_pos += n;
        Ok(self.n_prompt - self.prompt_pos)
    }

    /// `step_gen` (mtmd-helper-gen.cpp:611-664) — the backbone output is
    /// continuous, there is no token to consume.
    #[allow(clippy::type_complexity)]
    pub fn step_gen(
        &mut self,
        h_state_in: &[f32],
    ) -> Result<(Option<Vec<f32>>, bool), String> {
        let mut inp = self.mctx.gen_inp_default();
        inp.process = clip::ClipGenProcess::GenCode;
        inp.embd = Some(h_state_in);
        // clip only reseeds when the seed changes, pass the same one
        inp.seed = self.seed;
        if self.pack.temp > 0.0 {
            inp.temp = self.pack.temp;
        }
        let out = self.mctx.gen_audio_process(&inp)?;
        if out.is_eos && self.eos_step < 0 {
            self.eos_step = self.step_idx;
        }
        // the stopping step's frame is discarded; the budget is the
        // reference's fallback (mtmd-helper-gen.cpp:630-639)
        let chunk_done = (self.eos_step >= 0
            && self.step_idx >= self.eos_step + self.frames_after_eos)
            || self.step_idx >= self.chunk_budget;
        if chunk_done {
            return self.finish_chunk();
        }

        self.feats_buf.extend_from_slice(&out.feats);
        self.step_idx += 1;
        if !out.feats.is_empty()
            && self.feats_buf.len() / out.feats.len() >= self.window_frames
        {
            self.flush_gen_wav()?;
        }

        let pos = self.pos;
        self.pos += 1;
        let h = self.talker.decode_embd(&out.embd, 1, pos, self.seq_id)?;
        Ok((Some(h), false))
    }

    /// `finish_chunk` (mtmd-helper-gen.cpp:815-858) — end the current chunk
    /// and re-prompt the next on top of the voice.
    #[allow(clippy::type_complexity)]
    fn finish_chunk(&mut self) -> Result<(Option<Vec<f32>>, bool), String> {
        self.flush_gen_wav()?;
        // the decoder restarts too, the next chunk's audio is not
        // continuous with this one
        self.dec_state.clear();

        if self.chunk_idx + 1 >= self.chunks.len() {
            return Ok((None, true));
        }
        self.chunk_idx += 1;

        // drop this chunk's text and audio, keep the voice conditioning
        self.talker.seq_rm(self.seq_id, self.n_voice_pos, -1)?;
        self.pos = self.n_voice_pos;

        self.prompt_rows.clear();
        let chunk = self.chunks[self.chunk_idx].clone();
        for &t in &chunk {
            self.push_embd_row(t);
        }
        self.push_embd_row(self.audio_bos);
        self.arm_chunk_budget(self.chunk_idx);

        let n_rows = self.prompt_rows.len() / self.n_embd;
        assert!(n_rows > 0);
        let embd = std::mem::take(&mut self.prompt_rows);
        let pos = self.pos;
        self.pos += n_rows as i32;
        let h = self.talker.decode_embd(&embd, n_rows, pos, self.seq_id)?;
        Ok((Some(h), false))
    }

    /// `flush_gen_wav` (mtmd-helper-gen.cpp:940-960).
    fn flush_gen_wav(&mut self) -> Result<(), String> {
        if self.feats_buf.is_empty() {
            return Ok(());
        }
        let mut inp = self.mctx.gen_inp_default();
        inp.process = clip::ClipGenProcess::GenWav;
        inp.feats = Some(&self.feats_buf);
        inp.seed = self.seed;
        let state = std::mem::take(&mut self.dec_state);
        inp.state = if state.is_empty() { None } else { Some(&state) };
        let out = self.mctx.gen_audio_process(&inp)?;
        self.audio_pcm.extend_from_slice(&out.audio);
        self.dec_state = out.state;
        self.feats_buf.clear();
        Ok(())
    }

    /// `get_output` (mtmd-helper-gen.cpp:666-690).
    pub fn get_output(&mut self) -> Result<(i32, Vec<u8>), String> {
        self.flush_gen_wav()?;
        let rate = self.info.sample_rate;
        match self.out_type {
            GenAudioOutType::Pcm => Ok((
                rate,
                self.audio_pcm
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )),
            GenAudioOutType::Wav => Ok((rate, write_wav16(&self.audio_pcm, rate))),
        }
    }

    pub fn audio_pcm(&self) -> &[f32] {
        &self.audio_pcm
    }

    /// driver introspection — the pipeline-layer e2e's accessors
    /// (the chunk count proves `split_chunks` carved a long prompt)
    pub fn n_prompt_rows(&self) -> usize {
        self.n_prompt
    }

    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn talker_ref(&self) -> &T {
        &self.talker
    }
}
