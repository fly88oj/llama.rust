//! clip — port of `tools/mtmd/clip.cpp` (the vision path) plus the image side of
//! `tools/mtmd/mtmd-image.cpp`.
//!
//! Line numbers refer to the pinned worktree @ bd4f514db1. Only the
//! `qwen3vl_merger` projector is wired end to end (`clip_graph_qwen3vl::build`,
//! models/qwen3vl.cpp:3) — the local mmproj files use it. The loader, the
//! hparams and the preprocessor machinery are structured after the C so more
//! projector types can be filled in without reshaping anything.
//!
//! Scope (see the final report for the full list of what is NOT ported):
//!   * hparams parsing for the vision modality, `clip.*` metadata only;
//!   * the tensor table of the qwen2/qwen2.5/qwen3-VL family;
//!   * the ViT graph: conv patch embedding, spatial merge, bilinear position
//!     interpolation, M-RoPE attention, GELU FFN, the 4x merger MLP;
//!   * `img_tool` preprocessing (Pillow-compatible bicubic/bilinear/Lanczos
//!     resampling, smart-resize geometry) and `clip_encode`.
//!
//! Not here: the audio encoder (`mtmd-audio.cpp`), video/temporal merge,
//! the remaining ~50 projector types, LoRA/`cb_eval`, `--image-min/max-tokens`
//! overrides beyond what the hparams carry, and the `libmtmd` C ABI.

use ggml::ops::GGML_ROPE_TYPE_NEOX;
use ggml::ops::{GGML_ROPE_TYPE_VISION, GGML_SCALE_FLAG_ALIGN_CORNERS, GGML_SCALE_MODE_BILINEAR};
use ggml::{Context, GgmlType, Gguf, Graph, TensorId};
use std::sync::Arc;

// ======================================================================
// projector types — clip-impl.h:400-460 (`PROJECTOR_TYPE_NAMES`)
// ======================================================================

/// The subset of `projector_type` this port can execute. Everything else is
/// recognised by name (so a wrong mmproj fails loudly) but has no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorType {
    Qwen2Vl,
    Qwen25Vl,
    Qwen3Vl,
    /// Ling 3.0 VL — the qwen3vl tower with a norm-only merger and the
    /// projector MLP at the top level (models/ling3vl.cpp:5-86)
    Ling3Vl,
    /// Cohere2 vision — the SigLIP tower over square llava-uhd tiles with a
    /// swapped-swiglu two-layer projector (models/siglip.cpp:48-56,
    /// mtmd-image.cpp:1145, clip-impl.h:513 "cohere2v", upstream 50a6c5cf7)
    Cohere2V,

    /// audio-only whisper-encoder family (`clip.audio.projector_type` /
    /// `clip.projector_type` == "qwen2a", clip-impl.h:533)
    Qwen2A,
    /// the whisper-enc siblings (clip-impl.h:533-537): all run the same
    /// `clip_graph_whisper_enc` graph (clip.cpp:1025-1030), they only differ
    /// in the projector tail (models/whisper-enc.cpp:55-130)
    Ultravox,
    Voxtral,
    Meralion,
    Glma,
    MusicFlamingo,
    /// Qwen3-ASR encoder (models/qwen3a.cpp:3, clip.cpp:1088)
    Qwen3A,
    /// Gemma4 unified-audio embedder — encoder-free, a single rms_norm+mm
    /// over the raw waveform frames (models/gemma4ua.cpp:4, clip.cpp:1076)
    Gemma4UA,
    /// LFM2-audio conformer encoder (models/conformer.cpp:3, clip.cpp:1072)
    Lfm2A,
    /// d1-omni audio — the lfm2a conformer plus a norm+down+up residual
    /// block after the projector and the 30 s cut / 0.5 s pad audio
    /// preprocessor (models/conformer.cpp:210-216, mtmd-audio.cpp:998,
    /// clip-impl.h:492 "d1omni_a", upstream a657f7e98)
    D1OmniA,
    /// Gemma4-audio conformer encoder — chunked local attention +
    /// ClippableLinear (models/gemma4a.cpp:11, clip.cpp:1074)
    Gemma4A,
    /// Granite-Speech conformer + QFormer projector
    /// (models/granite-speech.cpp:5, clip.cpp:1082)
    GraniteSpeech,
    /// Parakeet RNN-T conformer encoder (models/parakeet.cpp:7, clip.cpp:1136)
    Parakeet,
    /// MiMo audio tokenizer encoder (models/mimo-audio.cpp:3, clip.cpp:1094)
    MimoAudio,
    /// qwen3-tts ECAPA-TDNN speaker encoder
    /// (models/qwen3tts-spkenc.cpp:154, clip.cpp:1098)
    Qwen3TtsSpkEnc,
    /// pocket-tts mimi-SEANet speaker encoder
    /// (models/pockettts-spkenc.cpp:39, clip.cpp:1102)
    PocketTtsSpkEnc,
    /// qwen3-tts code_predictor + code2wav output generator
    /// (models/qwen3tts-gen.cpp:657, clip.cpp:1113-1119)
    Qwen3TtsGen,
    /// pocket-tts flow-matching + mimi decoder output generator
    /// (models/pockettts-gen.cpp:139, clip.cpp:1107-1112)
    PocketTtsGen,
    Unknown,
}

impl ProjectorType {
    /// clip-impl.h:518 `clip_projector_type_from_string`
    pub fn from_str(s: &str) -> Self {
        match s {
            "qwen2vl_merger" => ProjectorType::Qwen2Vl,
            "qwen2.5vl_merger" => ProjectorType::Qwen25Vl,
            "qwen3vl_merger" => ProjectorType::Qwen3Vl,
            "ling3vl" => ProjectorType::Ling3Vl,
            "cohere2v" => ProjectorType::Cohere2V,
            "qwen2a" => ProjectorType::Qwen2A,
            "ultravox" => ProjectorType::Ultravox,
            "voxtral" => ProjectorType::Voxtral,
            "meralion" => ProjectorType::Meralion,
            "glma" => ProjectorType::Glma,
            "musicflamingo" => ProjectorType::MusicFlamingo,
            "qwen3a" => ProjectorType::Qwen3A,
            "gemma4ua" => ProjectorType::Gemma4UA,
            "lfm2a" => ProjectorType::Lfm2A,
            "d1omni_a" => ProjectorType::D1OmniA,
            "gemma4a" => ProjectorType::Gemma4A,
            "granite_speech" => ProjectorType::GraniteSpeech,
            "parakeet" => ProjectorType::Parakeet,
            "mimo_audio" => ProjectorType::MimoAudio,
            "qwen3tts_spkenc" => ProjectorType::Qwen3TtsSpkEnc,
            "pockettts_spkenc" => ProjectorType::PocketTtsSpkEnc,
            "qwen3tts_gen" => ProjectorType::Qwen3TtsGen,
            "pockettts_gen" => ProjectorType::PocketTtsGen,
            _ => ProjectorType::Unknown,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ProjectorType::Qwen2Vl => "qwen2vl_merger",
            ProjectorType::Qwen25Vl => "qwen2.5vl_merger",
            ProjectorType::Qwen3Vl => "qwen3vl_merger",
            ProjectorType::Ling3Vl => "ling3vl",
            ProjectorType::Cohere2V => "cohere2v",
            ProjectorType::Qwen2A => "qwen2a",
            ProjectorType::Ultravox => "ultravox",
            ProjectorType::Voxtral => "voxtral",
            ProjectorType::Meralion => "meralion",
            ProjectorType::Glma => "glma",
            ProjectorType::MusicFlamingo => "musicflamingo",
            ProjectorType::Qwen3A => "qwen3a",
            ProjectorType::Gemma4UA => "gemma4ua",
            ProjectorType::Lfm2A => "lfm2a",
            ProjectorType::D1OmniA => "d1omni_a",
            ProjectorType::Gemma4A => "gemma4a",
            ProjectorType::GraniteSpeech => "granite_speech",
            ProjectorType::Parakeet => "parakeet",
            ProjectorType::MimoAudio => "mimo_audio",
            ProjectorType::Qwen3TtsSpkEnc => "qwen3tts_spkenc",
            ProjectorType::PocketTtsSpkEnc => "pockettts_spkenc",
            ProjectorType::Qwen3TtsGen => "qwen3tts_gen",
            ProjectorType::PocketTtsGen => "pockettts_gen",
            ProjectorType::Unknown => "unknown",
        }
    }

    /// every arch dispatched to `clip_graph_whisper_enc` (clip.cpp:1025-1030)
    /// and thus to the audio (whisper) loader/hparams arms
    fn is_whisper_family(self) -> bool {
        matches!(
            self,
            ProjectorType::Qwen2A
                | ProjectorType::Ultravox
                | ProjectorType::Voxtral
                | ProjectorType::Meralion
                | ProjectorType::Glma
                | ProjectorType::MusicFlamingo
        )
    }

    /// clip-model.h:881 `audio_has_avgpool` — nn.AvgPool1d(2, stride=2) at the
    /// end of build_vit (clip.cpp:546-556)
    fn audio_has_avgpool(self) -> bool {
        matches!(
            self,
            ProjectorType::Qwen2A | ProjectorType::Voxtral | ProjectorType::MusicFlamingo
        )
    }

    /// clip-model.h:887 `audio_has_stack_frames` — StackAudioFrames right
    /// after build_vit (whisper-enc.cpp:48-53)
    fn audio_has_stack_frames(self) -> bool {
        matches!(
            self,
            ProjectorType::Ultravox | ProjectorType::Voxtral | ProjectorType::Meralion
        )
    }

    /// C `PROJECTOR_TYPE_QWEN2VL == PROJECTOR_TYPE_QWEN3VL` in every switch that
    /// shares the patch-merge geometry (clip.cpp:1661, :4155, :4796, :6052).
    fn is_qwen_vl_merger(self) -> bool {
        matches!(
            self,
            ProjectorType::Qwen2Vl | ProjectorType::Qwen25Vl | ProjectorType::Qwen3Vl
                | ProjectorType::Ling3Vl
        )
    }

    /// clip.cpp:4051 / :4077 / :4157 — `clip_n_output_tokens_{x,y}` divide by 2
    fn merger_2x2(self) -> bool {
        self.is_qwen_vl_merger()
    }
}

// ======================================================================
// images — clip-impl.h:584 (`clip_image_u8`) / :657 (`clip_image_f32`)
// ======================================================================

/// RGB uint8 image, `buf.len() == nx*ny*3` (empty buffer == placeholder).
#[derive(Clone, Default)]
pub struct ClipImageU8 {
    pub nx: i32,
    pub ny: i32,
    pub buf: Vec<u8>,
}

impl ClipImageU8 {
    pub fn new(nx: i32, ny: i32) -> Self {
        let mut img = Self {
            nx: 0,
            ny: 0,
            buf: Vec::new(),
        };
        img.set_size(nx, ny, false);
        img
    }

    pub fn set_size(&mut self, nx: i32, ny: i32, is_placeholder: bool) {
        self.nx = nx;
        self.ny = ny;
        if is_placeholder {
            self.buf.clear();
        } else {
            self.buf.resize(nx as usize * ny as usize * 3, 0);
        }
    }

    pub fn is_placeholder(&self) -> bool {
        self.buf.is_empty()
    }

    /// clip-impl.h:616
    pub fn get_pixel(&self, x: i32, y: i32) -> [u8; 3] {
        if self.is_placeholder() {
            return [0, 0, 0];
        }
        let idx = (y as usize * self.nx as usize + x as usize) * 3;
        [self.buf[idx], self.buf[idx + 1], self.buf[idx + 2]]
    }

    /// clip-impl.h:625
    pub fn set_pixel(&mut self, x: i32, y: i32, rgb: [u8; 3]) {
        if self.is_placeholder() {
            return;
        }
        let idx = (y as usize * self.nx as usize + x as usize) * 3;
        self.buf[idx..idx + 3].copy_from_slice(&rgb);
    }

    pub fn n_elements(&self) -> usize {
        self.nx as usize * self.ny as usize * 3
    }
}

/// `clip_image_f32` — same RGBRGB layout in f32, already normalised when the
/// preprocessor produced it (`clip_image_f32::normalize`, clip-impl.h:726).
#[derive(Clone, Default)]
pub struct ClipImageF32 {
    pub nx: i32,
    pub ny: i32,
    pub buf: Vec<f32>,
}

impl ClipImageF32 {
    pub fn new(nx: i32, ny: i32) -> Self {
        Self {
            nx,
            ny,
            buf: vec![0.0; nx as usize * ny as usize * 3],
        }
    }

    pub fn is_placeholder(&self) -> bool {
        self.buf.is_empty()
    }

    /// clip-impl.h:707
    pub fn from_u8(img: &ClipImageU8) -> Self {
        if img.is_placeholder() {
            return Self {
                nx: img.nx,
                ny: img.ny,
                buf: Vec::new(),
            };
        }
        Self {
            nx: img.nx,
            ny: img.ny,
            buf: img.buf.iter().map(|&v| v as f32 / 255.0).collect(),
        }
    }

    /// clip-impl.h:726 — per-channel (x - mean) / std.
    pub fn normalize(&mut self, mean: &[f32; 3], std: &[f32; 3]) {
        if self.is_placeholder() {
            return;
        }
        for px in self.buf.chunks_exact_mut(3) {
            px[0] = (px[0] - mean[0]) / std[0];
            px[1] = (px[1] - mean[1]) / std[1];
            px[2] = (px[2] - mean[2]) / std[2];
        }
    }
}

impl ClipImageF32 {
    /// mel-chunk form (mtmd.cpp:1617-1623): nx = frames, ny = mel bins, the
    /// buffer is n_len*n_mel f32 (mel-major rows of n_len)
    pub fn from_audio_mel(n_len: i32, n_mel: i32, data: &[f32]) -> Self {
        ClipImageF32 {
            nx: n_len,
            ny: n_mel,
            buf: data.to_vec(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ClipImageF32Batch {
    pub entries: Vec<ClipImageF32>,
    pub is_audio: bool,
}

// ======================================================================
// hparams — clip-model.h:49
// ======================================================================

/// clip-model.h:33 Pillow-compatible resampling kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeAlgo {
    Bilinear,
    Bicubic,
    Lanczos,
}

/// clip-model.h:43
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadStyle {
    None,
    Ceil,
    Nearest,
}

/// clip-model.h:14 `ffn_op_type`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnOp {
    Gelu,
    GeluErf,
    Silu,
    GeluQuick,
    ReluSqr,
}

/// clip-model.h:22 `norm_type`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    Normal,
    Rms,
}

/// clip-model.h:49 `clip_hparams` (vision fields; the audio ones are dropped —
/// this port has no audio encoder).
#[derive(Debug, Clone)]
pub struct ClipHparams {
    pub image_size: i32,
    pub patch_size: i32,
    pub n_embd: i32,
    pub n_ff: i32,
    pub projection_dim: i32,
    pub n_head: i32,
    pub n_head_kv: i32,
    pub n_embd_head: i32,
    pub n_layer: i32,
    /// clip-model.h:60 — patch merges **per side**
    pub n_merge: i32,

    pub image_min_pixels: i32,
    pub image_max_pixels: i32,
    pub image_resize_algo: ResizeAlgo,
    pub image_resize_pad: PadStyle,
    pub image_pad_color: [u8; 3],

    // ---- llava-uhd slicing (clip-model.h:70-80) -----------------------------
    /// `clip.vision.preproc_max_tiles` (KEY_PREPROC_MAX_TILES) — cohere2v's
    /// square-tile grid cap
    pub preproc_max_tiles: i32,
    /// resize algo / padding / pad color for the refined (tiled) image
    /// (clip-model.h:76-80; PAD_NONE for cohere2v — tiles stretch to the grid)
    pub image_resize_algo_rf: ResizeAlgo,
    pub image_pad_rf: PadStyle,
    pub image_pad_color_rf: [u8; 3],
    /// resize algo / padding / pad color for the overview (thumbnail) image
    pub image_resize_algo_ov: ResizeAlgo,
    pub image_pad_ov: PadStyle,
    pub image_pad_color_ov: [u8; 3],

    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],

    /// clip-model.h:87 — size the warmup uses (unused here: no warmup pass)
    pub warmup_image_size: i32,

    pub ffn_op: FfnOp,

    pub eps: f32,

    /// clip-model.h:188 custom values from the context params
    pub custom_image_min_tokens: i32,
    pub custom_image_max_tokens: i32,

    // ---- audio fields (clip-model.h:121-179) — the whisper-enc family ------
    /// whisper preprocessor mel bins (clip.audio.num_mel_bins)
    pub n_mel_bins: i32,
    /// ultravox-family StackAudioFrames factor
    pub proj_stack_factor: i32,
    /// audio-to-mel preprocessor params (clip-model.h:135-139)
    pub audio_chunk_len: i32, // in seconds
    pub audio_sample_rate: i32,
    pub audio_n_fft: i32,
    pub audio_window_len: i32,
    pub audio_hop_len: i32,

    // ---- granite_speech (clip-model.h:142-153, clip.cpp:1976-1994) --------
    /// attention context size — squared for attn_dists/attn_mask
    pub audio_chunk_size: i32,
    /// depthwise-conv kernel (odd)
    pub audio_conv_kernel_size: i32,
    /// max relative distance of the Shaw RPE table
    pub audio_max_pos_emb: i32,
    pub audio_proj_window_size: i32,
    pub audio_proj_downsample_rate: i32,
    pub audio_proj_head_count: i32,
    /// `clip.audio.feature_layer` (KEY_FEATURE_LAYERS, common optional path)
    pub feature_layers: Vec<i32>,

    // ---- parakeet (clip-model.h:167-170, clip.cpp:1488-1500) --------------
    /// fixed 8 — must match the conv strides (asserted in C)
    pub subsampling_factor: i32,
    /// `a.mel_filters` / `a.window` read as raw F32 vectors at load
    /// (clip.cpp get_vector, TN_MEL_FILTERS/TN_WINDOW)
    pub mel_filters: Vec<f32>,
    pub window: Vec<f32>,

    // ---- mimo_audio (clip-model.h:121-134, clip.cpp:1795-1834) ------------
    pub rvq_num_quantizers: i32,
    pub rvq_codebook_size: Vec<i32>,
    pub attn_window_size: i32,
    /// -1 (full attention) / 0 (windowed) per layer
    pub wa_pattern_mode: Vec<i32>,
    /// `clip.gen.audio.model_variant` (clip-impl.h:98) — the c2w/pocket decoder selector
    pub gen_model_variant: String,
    pub audio_local_n_layer: i32,
    pub audio_local_group_size: i32,

    // ---- pocket-tts (clip-model.h:157-166, clip.cpp:1875-1891) ------------
    pub seanet_ratios: Vec<i32>,
    pub seanet_n_stage: i32,
    pub mimi_downsample: i32,
    /// the mimi transformer's attention window ("context", clip.cpp:1885)
    pub mimi_tfm_context: i32,

    // ---- gen audio (audio round 5) ----------------------------------------
    /// hparams.rope_theta — qwen3tts's code_predictor uses 1e6
    /// (clip.cpp:1846), pocket-tts 1e4 (:1887)
    pub rope_theta: f32,
    // ---- qwen3tts code2wav (clip.cpp:1849-1860, hardcoded from the config) -
    pub wav_tfm_n_layer: i32,
    pub wav_tfm_n_embd: i32,
    pub wav_tfm_n_ff: i32,
    pub wav_tfm_n_head: i32,
    pub wav_tfm_n_head_kv: i32,
    pub wav_tfm_eps: f32,
    pub wav_tfm_rope_theta: f32,
    /// the pre_transformer's sliding window == frames per batch (72)
    pub wav_tfm_swa: i32,
    pub wav_upsample_n_block: i32,
    pub wav_dac_n_block: i32,
    pub wav_dac_n_res: i32,
    // ---- pocket-tts flow lm (clip.cpp:1888-1890) ---------------------------
    pub flow_n_step: i32,
    pub gen_eos_threshold: f32,
}

impl Default for ClipHparams {
    fn default() -> Self {
        Self {
            image_size: 0,
            patch_size: 0,
            n_embd: 0,
            n_ff: 0,
            projection_dim: 0,
            n_head: 0,
            n_head_kv: 0,
            n_embd_head: 0,
            n_layer: 0,
            n_merge: 1,
            image_min_pixels: -1,
            image_max_pixels: -1,
            image_resize_algo: ResizeAlgo::Bicubic,
            image_resize_pad: PadStyle::Ceil,
            image_pad_color: [0, 0, 0],
            // clip-model.h:73-80 defaults (PAD_CEIL refined / PAD_NONE overview)
            preproc_max_tiles: 0,
            image_resize_algo_rf: ResizeAlgo::Bicubic,
            image_pad_rf: PadStyle::Ceil,
            image_pad_color_rf: [0, 0, 0],
            image_resize_algo_ov: ResizeAlgo::Bicubic,
            image_pad_ov: PadStyle::None,
            image_pad_color_ov: [0, 0, 0],
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            warmup_image_size: 0,
            ffn_op: FfnOp::Gelu,
            eps: 1e-6,
            custom_image_min_tokens: -1,
            custom_image_max_tokens: -1,
            n_mel_bins: 0,
            proj_stack_factor: 0,
            audio_chunk_len: -1,
            audio_sample_rate: -1,
            audio_n_fft: -1,
            audio_window_len: -1,
            audio_hop_len: -1,
            audio_chunk_size: 0,
            audio_conv_kernel_size: 0,
            audio_max_pos_emb: 0,
            audio_proj_window_size: 0,
            audio_proj_downsample_rate: 0,
            audio_proj_head_count: 0,
            feature_layers: Vec::new(),
            subsampling_factor: 0,
            mel_filters: Vec::new(),
            window: Vec::new(),
            rvq_num_quantizers: 0,
            rvq_codebook_size: Vec::new(),
            attn_window_size: 0,
            wa_pattern_mode: Vec::new(),
            gen_model_variant: String::new(),
            audio_local_n_layer: 0,
            audio_local_group_size: 0,
            seanet_ratios: Vec::new(),
            seanet_n_stage: 0,
            mimi_downsample: 0,
            mimi_tfm_context: 0,
            rope_theta: 10000.0,
            wav_tfm_n_layer: 0,
            wav_tfm_n_embd: 0,
            wav_tfm_n_ff: 0,
            wav_tfm_n_head: 0,
            wav_tfm_n_head_kv: 0,
            wav_tfm_eps: 1e-5,
            wav_tfm_rope_theta: 10000.0,
            wav_tfm_swa: 0,
            wav_upsample_n_block: 0,
            wav_dac_n_block: 0,
            wav_dac_n_res: 0,
            flow_n_step: 0,
            gen_eos_threshold: 0.0,
        }
    }
}

impl ClipHparams {
    /// clip-model.h:190 `set_limit_image_tokens`
    fn set_limit_image_tokens(&mut self, n_tokens_min: i32, n_tokens_max: i32) {
        let patch_area = self.patch_size * self.patch_size * self.n_merge * self.n_merge;
        self.image_min_pixels = (if self.custom_image_min_tokens > 0 {
            self.custom_image_min_tokens
        } else {
            n_tokens_min
        }) * patch_area;
        self.image_max_pixels = (if self.custom_image_max_tokens > 0 {
            self.custom_image_max_tokens
        } else {
            n_tokens_max
        }) * patch_area;
        self.warmup_image_size = (self.image_max_pixels as f32).sqrt() as i32;
    }

    /// the audio fields `mtmd_audio_preprocessor`s read (mtmd-audio.h:53-60)
    pub fn audio_hparams(&self) -> crate::mtmd_audio::AudioHparams {
        crate::mtmd_audio::AudioHparams {
            n_mel_bins: self.n_mel_bins,
            audio_chunk_len: self.audio_chunk_len,
            audio_sample_rate: self.audio_sample_rate,
            audio_n_fft: self.audio_n_fft,
            audio_window_len: self.audio_window_len,
            audio_hop_len: self.audio_hop_len,
            mel_filters: self.mel_filters.clone(),
            window: self.window.clone(),
            mimi_downsample: self.mimi_downsample,
            ..Default::default()
        }
    }

    /// clip-model.h:208-219 `set_warmup_n_tokens`
    fn set_warmup_n_tokens(&mut self, n_tokens: i32) {
        let mut n_tok_per_side = (n_tokens as f32).sqrt() as i32;
        assert_eq!(
            n_tok_per_side * n_tok_per_side,
            n_tokens,
            "n_tokens must be n*n"
        );
        // do not warmup with more tokens than the max allowed
        // (clip-model.h:213-215, upstream a7b94df2c)
        if self.custom_image_max_tokens > 0 && n_tokens > self.custom_image_max_tokens {
            n_tok_per_side = 1.max((self.custom_image_max_tokens as f32).sqrt() as i32);
        }
        self.warmup_image_size = n_tok_per_side * self.patch_size * self.n_merge;
    }
}

// ======================================================================
// model — clip-model.h:235 (`clip_layer`) / the vision half of `clip_model`
// ======================================================================

pub struct ClipLayer {
    /// qwen3tts-gen's per-head q/k RMS-norm gains (`attn_{q,k}_norm`, qwen3tts-gen.cpp:155-158)
    pub q_norm: Option<TensorId>,
    pub k_norm: Option<TensorId>,
    pub ln_1_w: Option<TensorId>,
    pub ln_1_b: Option<TensorId>,
    /// separate q/k/v of the whisper-enc family (clip.cpp:2260-2281)
    pub q_w: Option<TensorId>,
    pub q_b: Option<TensorId>,
    pub k_w: Option<TensorId>,
    pub k_b: Option<TensorId>,
    pub v_w: Option<TensorId>,
    pub v_b: Option<TensorId>,
    pub qkv_w: Option<TensorId>,
    pub qkv_b: Option<TensorId>,
    pub o_w: Option<TensorId>,
    pub o_b: Option<TensorId>,
    pub ff_up_w: Option<TensorId>,
    pub ff_up_b: Option<TensorId>,
    pub ff_gate_w: Option<TensorId>,
    pub ff_gate_b: Option<TensorId>,
    pub ff_down_w: Option<TensorId>,
    pub ff_down_b: Option<TensorId>,
    pub ln_2_w: Option<TensorId>,
    pub ln_2_b: Option<TensorId>,

    // ---- conformer-family fields (lfm2a, models/conformer.cpp) -----------
    /// norm before FFN1 (TN_FFN_NORM "a.blk.N.ffn_norm.*")
    pub ff_norm_w: Option<TensorId>,
    pub ff_norm_b: Option<TensorId>,
    /// norm before FFN2 (TN_FFN_NORM_1 "a.blk.N.ffn_norm_1.*")
    pub ff_norm_1_w: Option<TensorId>,
    pub ff_norm_1_b: Option<TensorId>,
    /// FFN2 weights (TN_FFN_UP_1 / TN_FFN_DOWN_1)
    pub ff_up_1_w: Option<TensorId>,
    pub ff_up_1_b: Option<TensorId>,
    pub ff_down_1_w: Option<TensorId>,
    pub ff_down_1_b: Option<TensorId>,
    /// rel-pos attention biases (TN_POS_BIAS_U/V, no weight suffix)
    pub pos_bias_u: Option<TensorId>,
    pub pos_bias_v: Option<TensorId>,
    /// rel-pos projection (TN_LINEAR_POS)
    pub linear_pos_w: Option<TensorId>,
    /// norm before the conv module (TN_NORM_CONV)
    pub norm_conv_w: Option<TensorId>,
    pub norm_conv_b: Option<TensorId>,
    /// folded batch-norm after the depthwise conv (TN_CONV_NORM)
    pub conv_norm_w: Option<TensorId>,
    pub conv_norm_b: Option<TensorId>,
    /// depthwise conv (TN_CONV_DW)
    pub conv_dw_w: Option<TensorId>,
    pub conv_dw_b: Option<TensorId>,
    /// pointwise convs of the conv module (TN_CONV_PW1/PW2)
    pub conv_pw1_w: Option<TensorId>,
    pub conv_pw1_b: Option<TensorId>,
    pub conv_pw2_w: Option<TensorId>,
    pub conv_pw2_b: Option<TensorId>,

    // ---- gemma4a (models/gemma4a.cpp, clip.cpp:3273-3341) ------------------
    /// norm before the attention (TN_A_ATTN_PRE_NORM; ln_1_w when absent)
    pub attn_pre_norm_w: Option<TensorId>,
    pub attn_post_norm_w: Option<TensorId>,
    /// per-dim Q/K scaling (TN_A_PER_DIM_SCALE / _K_SCALE)
    pub per_dim_scale_w: Option<TensorId>,
    pub per_dim_k_scale_w: Option<TensorId>,
    /// the RPE projection of chunked local attention (TN_A_ATTN_K_REL)
    pub attn_k_rel_w: Option<TensorId>,
    pub ff_post_norm_w: Option<TensorId>,
    pub ff_post_norm_1_w: Option<TensorId>,

    // ---- granite_speech (clip.cpp:3455-3530) -------------------------------
    /// Shaw RPE position table (TN_ATTN_REL_POS_EMB, no weight suffix)
    pub attn_rel_pos_emb: Option<TensorId>,

    // ---- parakeet (clip.cpp:3393-3434) --------------------------------------
    /// folded batch-norm mean/var (TN_CONV_NORM_MEAN / _VAR)
    pub conv_norm_mean: Option<TensorId>,
    pub conv_norm_var: Option<TensorId>,

    // ---- pockettts (pockettts-spkenc.cpp:27/33) -----------------------------
    /// layer scale (TN_LS_1 / TN_LS_2, weight-only)
    pub ls_1_w: Option<TensorId>,
    pub ls_2_w: Option<TensorId>,

    // ---- qwen3tts spkenc (clip.cpp:2923-2962) -------------------------------
    /// SE block convs (TN_A_SE_CONV1/2)
    pub se_conv1_w: Option<TensorId>,
    pub se_conv1_b: Option<TensorId>,
    pub se_conv2_w: Option<TensorId>,
    pub se_conv2_b: Option<TensorId>,
    /// Res2Net dilated branch convs, 7 per block (TN_A_CONV_RES2)
    pub res2_conv_w: Vec<Option<TensorId>>,
    pub res2_conv_b: Vec<Option<TensorId>>,
}

/// `CLIP_MODALITY_*` (clip.h:35-39)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipModality {
    Vision,
    Audio,
    /// CLIP_MODALITY_GEN_AUDIO — the output generators (audio round 5)
    GenAudio,
}

/// clip-model.h `clip_qformer_layer` — one self+cross attention + FFN block
/// of the granite_speech QFormer (clip.cpp:3508-3530)
#[derive(Default)]
pub struct QFormerLayer {
    pub q_w: Option<TensorId>,
    pub q_b: Option<TensorId>,
    pub k_w: Option<TensorId>,
    pub k_b: Option<TensorId>,
    pub v_w: Option<TensorId>,
    pub v_b: Option<TensorId>,
    pub o_w: Option<TensorId>,
    pub o_b: Option<TensorId>,
    pub ln_1_w: Option<TensorId>,
    pub ln_1_b: Option<TensorId>,
    pub cross_attn_q_w: Option<TensorId>,
    pub cross_attn_q_b: Option<TensorId>,
    pub cross_attn_k_w: Option<TensorId>,
    pub cross_attn_k_b: Option<TensorId>,
    pub cross_attn_v_w: Option<TensorId>,
    pub cross_attn_v_b: Option<TensorId>,
    pub cross_attn_o_w: Option<TensorId>,
    pub cross_attn_o_b: Option<TensorId>,
    pub cross_attn_norm_w: Option<TensorId>,
    pub cross_attn_norm_b: Option<TensorId>,
    pub ff_up_w: Option<TensorId>,
    pub ff_up_b: Option<TensorId>,
    pub ff_down_w: Option<TensorId>,
    pub ff_down_b: Option<TensorId>,
    pub ln_2_w: Option<TensorId>,
    pub ln_2_b: Option<TensorId>,
}

/// clip-model.h `clip_qformer_block` — granite_speech's single QFormer
/// projector (qf_proj_query / norm / linear + 2 layers)
#[derive(Default)]
pub struct QFormerBlock {
    pub qf_proj_query: Option<TensorId>,
    pub qf_proj_norm_w: Option<TensorId>,
    pub qf_proj_norm_b: Option<TensorId>,
    pub qf_proj_linear_w: Option<TensorId>,
    pub qf_proj_linear_b: Option<TensorId>,
    pub layers: Vec<QFormerLayer>,
}

/// clip-model.h `clip_seanet` — the pocket-tts SEANet conv stack
/// (clip.cpp:2166 `load_seanet`, encoder half)
#[derive(Default)]
pub struct Seanet {
    pub conv_in_w: Option<TensorId>,
    pub conv_in_b: Option<TensorId>,
    pub conv_out_w: Option<TensorId>,
    pub conv_out_b: Option<TensorId>,
    pub stages: Vec<SeanetStage>,
}

#[derive(Default)]
pub struct SeanetStage {
    pub res_conv1_w: Option<TensorId>,
    pub res_conv1_b: Option<TensorId>,
    pub res_conv2_w: Option<TensorId>,
    pub res_conv2_b: Option<TensorId>,
    pub scale_conv_w: Option<TensorId>,
    pub scale_conv_b: Option<TensorId>,
}

// ======================================================================
// gen audio (audio round 5) — clip-model.h:453 `clip_flow_net`,
// :489 `clip_code2wav`, models.h:385 `c2w_state_slot`. Owner: agent GEN5.
// ======================================================================

/// clip-model.h:453 `clip_flow_net` — pocket-tts's flow-matching velocity net
#[derive(Default)]
pub struct FlowNet {
    pub input_proj_w: Option<TensorId>,
    pub input_proj_b: Option<TensorId>,
    pub cond_embd_w: Option<TensorId>,
    pub cond_embd_b: Option<TensorId>,
    /// -> shift, scale, gate
    pub final_ada_w: Option<TensorId>,
    pub final_ada_b: Option<TensorId>,
    pub final_proj_w: Option<TensorId>,
    pub final_proj_b: Option<TensorId>,
    pub time: Vec<FlowTimeEmbd>,
    pub blocks: Vec<FlowBlock>,
}

/// clip-model.h:467 `clip_flow_net::time_embd` — cos/sin(t*freqs) MLP
#[derive(Default)]
pub struct FlowTimeEmbd {
    pub freqs: Option<TensorId>,
    pub up_w: Option<TensorId>,
    pub up_b: Option<TensorId>,
    pub down_w: Option<TensorId>,
    pub down_b: Option<TensorId>,
    /// RMSNorm alpha
    pub norm: Option<TensorId>,
}

/// clip-model.h:455 `clip_flow_net::block` — one AdaLN res block
#[derive(Default)]
pub struct FlowBlock {
    pub norm_w: Option<TensorId>,
    pub norm_b: Option<TensorId>,
    pub up_w: Option<TensorId>,
    pub up_b: Option<TensorId>,
    pub down_w: Option<TensorId>,
    pub down_b: Option<TensorId>,
    pub ada_w: Option<TensorId>,
    pub ada_b: Option<TensorId>,
}

/// clip-model.h:491 `clip_code2wav::upsample_block`
#[derive(Default)]
pub struct C2wUpsample {
    pub conv_w: Option<TensorId>,
    pub conv_b: Option<TensorId>,
    pub dwconv_w: Option<TensorId>,
    pub dwconv_b: Option<TensorId>,
    pub norm_w: Option<TensorId>,
    pub norm_b: Option<TensorId>,
    pub pw1_w: Option<TensorId>,
    pub pw1_b: Option<TensorId>,
    pub pw2_w: Option<TensorId>,
    pub pw2_b: Option<TensorId>,
    pub gamma: Option<TensorId>,
}

/// clip-model.h:506 `clip_code2wav::dac_res`
#[derive(Default)]
pub struct C2wDacRes {
    pub act1_alpha: Option<TensorId>,
    pub act1_beta: Option<TensorId>,
    pub conv1_w: Option<TensorId>,
    pub conv1_b: Option<TensorId>,
    pub act2_alpha: Option<TensorId>,
    pub act2_beta: Option<TensorId>,
    pub conv2_w: Option<TensorId>,
    pub conv2_b: Option<TensorId>,
}

/// clip-model.h:518 `clip_code2wav::dac_block`
#[derive(Default)]
pub struct C2wDac {
    pub snake_alpha: Option<TensorId>,
    pub snake_beta: Option<TensorId>,
    pub conv_w: Option<TensorId>,
    pub conv_b: Option<TensorId>,
    pub res: Vec<C2wDacRes>,
}

/// clip-model.h:489 `clip_code2wav` — qwen3-tts RVQ codes -> raw PCM
#[derive(Default)]
pub struct Code2Wav {
    pub quant_first_in_w: Option<TensorId>,
    pub quant_first_out_w: Option<TensorId>,
    pub quant_first_cb_w: Option<TensorId>,
    pub quant_rest_in_w: Option<TensorId>,
    pub quant_rest_out_w: Option<TensorId>,
    /// codebooks merged 3D [15, vocab, dim]
    pub quant_rest_cb_w: Option<TensorId>,
    pub pre_conv_w: Option<TensorId>,
    pub pre_conv_b: Option<TensorId>,
    pub tfm_in_proj_w: Option<TensorId>,
    pub tfm_in_proj_b: Option<TensorId>,
    pub tfm_out_proj_w: Option<TensorId>,
    pub tfm_out_proj_b: Option<TensorId>,
    pub tfm_output_norm_w: Option<TensorId>,
    /// reuses the generic ClipLayer fields (ln_1/attn/ln_2/ffn/ls_1/ls_2)
    pub tfm_layers: Vec<ClipLayer>,
    pub upsample: Vec<C2wUpsample>,
    pub dac_entry_w: Option<TensorId>,
    pub dac_entry_b: Option<TensorId>,
    pub dac: Vec<C2wDac>,
    pub dac_post_snake_alpha: Option<TensorId>,
    pub dac_post_snake_beta: Option<TensorId>,
    pub dac_post_conv_w: Option<TensorId>,
    pub dac_post_conv_b: Option<TensorId>,
}

/// models.h:385 `c2w_state_slot` — one persisted state buffer of the
/// generators' GEN_WAV half
pub struct C2wStateSlot {
    pub name: String,
    pub ne0: i64,
    pub ne1: i64,
}

/// clip.h:61 `clip_gen_process_type`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipGenProcess {
    GenCode,
    GenWav,
}

/// the four clamp scalars of gemma4a's ClippableLinear
/// (`<w>.input_max/.input_min/.output_max/.output_min`, clip.cpp:3315-3333)
#[derive(Clone, Copy, Debug)]
pub struct ClampInfo {
    pub inp_min: f32,
    pub inp_max: f32,
    pub out_min: f32,
    pub out_max: f32,
}

pub struct ClipModel {
    pub hparams: ClipHparams,
    pub proj_type: ProjectorType,
    pub has_vision: bool,
    /// CLIP_MODALITY_AUDIO when only `clip.has_audio_encoder` is set
    pub modality: ClipModality,
    /// whisper-enc conv1d + qwen2a projector tensors (clip.cpp:2852-2862)
    pub conv1d_1_w: Option<TensorId>,
    pub conv1d_1_b: Option<TensorId>,
    pub conv1d_2_w: Option<TensorId>,
    pub conv1d_2_b: Option<TensorId>,
    pub mm_fc_w: Option<TensorId>,
    pub mm_fc_b: Option<TensorId>,

    /// ultravox/meralion/glma `mm.a.norm_pre` (clip.cpp:2825/:2850/:3180) and
    /// ultravox `mm.a.norm_mid` (clip.cpp:2826)
    pub mm_norm_pre_w: Option<TensorId>,
    pub mm_norm_pre_b: Option<TensorId>,
    pub mm_norm_mid_w: Option<TensorId>,
    /// `mm.a.mlp.{1,2}` of every whisper-family projector tail
    /// (ultravox/voxtral/musicflamingo/glma; meralion uses 0..3)
    pub mm_2_w: Option<TensorId>,
    pub mm_2_b: Option<TensorId>,
    /// meralion's out_proj (clip.cpp:2845-2846)
    pub mm_3_w: Option<TensorId>,
    pub mm_3_b: Option<TensorId>,
    /// d1omni_a's residual block after the projector — `mm.a.mlp.{4,5,6}`
    /// norm/down/up (clip-model.h:626-629, clip.cpp:3440-3447)
    pub mm_4_w: Option<TensorId>,
    pub mm_4_b: Option<TensorId>,
    pub mm_5_w: Option<TensorId>,
    pub mm_5_b: Option<TensorId>,
    pub mm_6_w: Option<TensorId>,
    pub mm_6_b: Option<TensorId>,
    /// glma's BOI/EOI embeddings (clip.cpp:3183-3184; TN_TOK_BOI is literally
    /// "v.boi" — the loader does not swap the "v." prefix for audio)
    pub mm_boi: Option<TensorId>,
    pub mm_eoi: Option<TensorId>,

    /// qwen3a's 3x stride-2 conv2d stem + projection (clip.cpp:2861-2869;
    /// weights [K, K, IC, OC], `a.conv2d.{1..3}.*`)
    pub conv2d_w: [Option<TensorId>; 3],
    pub conv2d_b: [Option<TensorId>; 3],
    /// qwen3a `a.conv_out.weight` (no bias) / lfm2a+parakeet
    /// `a.pre_encode.out.weight` (clip.cpp:2870/:3379)
    pub conv_out_w: Option<TensorId>,
    /// gemma4a/ua `mm.a.input_projection.weight` / gemma4ua's loader arm
    /// loads `mm.input_projection.weight` (TN_MM_INP_PROJ, clip.cpp:3344)
    pub mm_input_proj_w: Option<TensorId>,
    /// lfm2a/parakeet pre-encode conv stack — `a.conv1d.{0,2,3,5,6}.*`
    /// (clip.cpp:3360-3367/:3377-3384); the weights are conv2D kernels
    /// [KW, KH, IC, OC] despite the tensor names
    pub pre_conv_w: [Option<TensorId>; 7],
    pub pre_conv_b: [Option<TensorId>; 7],
    /// lfm2a/parakeet `a.pre_encode.out.*` (clip.cpp:3368-3369)
    pub pre_encode_out_w: Option<TensorId>,
    pub pre_encode_out_b: Option<TensorId>,

    // ---- granite_speech (clip.cpp:3455-3471) -------------------------------
    /// `a.input_projection.*` (TN_INP_PROJ) + the CTC mid-branch
    pub inp_proj_w: Option<TensorId>,
    pub inp_proj_b: Option<TensorId>,
    pub ctc_out_w: Option<TensorId>,
    pub ctc_out_b: Option<TensorId>,
    pub ctc_out_mid_w: Option<TensorId>,
    pub ctc_out_mid_b: Option<TensorId>,
    /// the single QFormer projector block (model.qf_proj_blocks[0])
    pub qf: Option<Box<QFormerBlock>>,

    // ---- gemma4a (clip.cpp:3273-3313) ---------------------------------------
    pub sscp_conv_w: [Option<TensorId>; 2],
    pub sscp_conv_b: [Option<TensorId>; 2],
    pub sscp_norm_w: [Option<TensorId>; 2],
    pub sscp_inp_proj_w: Option<TensorId>,
    pub sscp_inp_proj_b: Option<TensorId>,
    pub audio_out_proj_w: Option<TensorId>,
    pub audio_out_proj_b: Option<TensorId>,
    pub mm_soft_emb_norm_w: Option<TensorId>,
    /// `<name>.weight` → {input_max, input_min, output_max, output_min}
    /// scalars (clip.cpp:3315-3333) — ClippableLinear
    pub clamp_info_map: std::collections::HashMap<String, ClampInfo>,

    // ---- mimo_audio (clip.cpp:2875-2933) ------------------------------------
    pub downsample_conv_w: Option<TensorId>,
    pub downsample_norm_w: Option<TensorId>,
    pub downsample_norm_b: Option<TensorId>,
    /// `a.rvq.codebook.weight` [dim, max_bins, n_q]
    pub rvq_codebook: Option<TensorId>,
    /// `mm.a.code_embd.weight` [dim, vocab, n_q]
    pub mm_a_code_embd: Option<TensorId>,
    /// the input_local_transformer stack (mm.a.local_blk.*)
    pub mm_a_local_layers: Vec<ClipLayer>,
    pub mm_a_local_norm_w: Option<TensorId>,

    // ---- qwen3tts spkenc (clip.cpp:2960-2961) -------------------------------
    /// `a.conv_out.*` (TN_CONV_OUT) — the MFA aggregation conv
    pub spk_conv_out_w: Option<TensorId>,
    pub spk_conv_out_b: Option<TensorId>,
    /// attentive statistics pooling (TN_A_ASP_ATTN / TN_A_ASP_TDNN)
    pub spk_asp_attn_w: Option<TensorId>,
    pub spk_asp_attn_b: Option<TensorId>,
    pub spk_asp_tdnn_w: Option<TensorId>,
    pub spk_asp_tdnn_b: Option<TensorId>,

    // ---- pockettts spkenc (clip.cpp:2965-2968 + load_seanet :2166) ---------
    pub seanet: Option<Box<Seanet>>,
    /// `a.downsample.conv.weight` (TN_A_DOWNSAMPLE_CONV, no bias)
    pub pocket_downsample_w: Option<TensorId>,
    /// `a.speaker_proj.weight` (TN_A_SPEAKER_PROJ)
    pub spk_proj_w: Option<TensorId>,

    // ---- gen audio (audio round 5, clip.cpp:2969-3134) ---------------------
    /// pocket-tts flow-matching head (clip-model.h:453 `clip_flow_net`)
    pub flow: Option<Box<FlowNet>>,
    /// qwen3-tts code2wav decoder (clip-model.h:489 `clip_code2wav`)
    pub c2w: Option<Box<Code2Wav>>,
    /// pocket-tts mimi decoder transformer ("a.gen.wav.tfm.*", clip.cpp:3021)
    pub gen_tfm_layers: Vec<ClipLayer>,
    /// `a.gen.code.proj_in.*` — optional, absent when the predictor shares
    /// the talker hidden size (clip.cpp:3042-3043)
    pub gen_code_proj_in_w: Option<TensorId>,
    pub gen_code_proj_in_b: Option<TensorId>,
    /// `a.gen.code.embd.weight` [n_embd_talker, vocab, n_acoustic]
    pub gen_code_embd_w: Option<TensorId>,
    /// `a.gen.code.head.weight` [n_embd_pred, vocab, n_acoustic]
    pub gen_code_head_w: Option<TensorId>,
    /// `a.gen.code.out_embd.weight` [n_embd_talker, vocab0]
    pub gen_code_out_embd_w: Option<TensorId>,
    /// `a.gen.code.output_norm.weight`
    pub gen_code_norm_w: Option<TensorId>,
    /// `a.gen.out_eos.*` — the pocket-tts end-of-speech probe
    pub gen_out_eos_w: Option<TensorId>,
    pub gen_out_eos_b: Option<TensorId>,
    /// `a.gen.input_linear.weight` [n_out, n_latent]
    pub gen_input_lin_w: Option<TensorId>,
    /// `a.gen.emb_mean` / `a.gen.emb_std` (raw F32 scalar tensors)
    pub gen_emb_mean: Option<TensorId>,
    pub gen_emb_std: Option<TensorId>,
    /// `a.gen.wav.quant_out.weight` — the mimi DummyQuantizer up-projection
    pub gen_quant_out_w: Option<TensorId>,
    /// `a.gen.wav.upsample.weight` — depthwise transposed conv [K, 1, C]
    pub gen_upsample_w: Option<TensorId>,
    /// clip.gen.audio.model_variant (clip-impl.h:98)
    pub gen_model_variant: String,

    pub class_embedding: Option<TensorId>,
    pub pre_ln_w: Option<TensorId>,
    pub pre_ln_b: Option<TensorId>,
    pub post_ln_w: Option<TensorId>,
    pub post_ln_b: Option<TensorId>,
    pub patch_bias: Option<TensorId>,
    pub patch_embeddings_0: Option<TensorId>,
    pub patch_embeddings_1: Option<TensorId>,
    pub position_embeddings: Option<TensorId>,

    pub mm_0_w: Option<TensorId>,
    pub mm_0_b: Option<TensorId>,
    pub mm_1_w: Option<TensorId>,
    pub mm_1_b: Option<TensorId>,
    /// `mm.input_norm.{weight,bias}` — the ling3vl merger norm
    /// (TN_MM_INP_NORM, clip-impl.h:148-149)
    pub mm_input_norm_w: Option<TensorId>,
    pub mm_input_norm_b: Option<TensorId>,

    pub layers: Vec<ClipLayer>,
}

// ======================================================================
// context — clip.cpp:147 `clip_ctx`
// ======================================================================

/// clip.h:43
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipFlashAttn {
    Auto,
    Disabled,
    Enabled,
}

/// clip.h:49 `clip_context_params` (the fields that mean something here)
#[derive(Debug, Clone)]
pub struct ClipContextParams {
    pub flash_attn_type: ClipFlashAttn,
    pub n_threads: usize,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
}

impl Default for ClipContextParams {
    fn default() -> Self {
        Self {
            flash_attn_type: ClipFlashAttn::Auto,
            n_threads: 4,
            image_min_tokens: -1,
            image_max_tokens: -1,
        }
    }
}

pub struct ClipContext {
    pub model: ClipModel,
    /// weight + graph tensors (weights are `Storage::External` into the mmap)
    pub ctx: Context,
    /// tensor count after loading — `reset_graph_to` drops the per-image graph
    pub watermark: usize,
    pub flash_attn_type: ClipFlashAttn,
    pub n_threads: usize,
    /// `ctx->out_embd` (mtmd.cpp:483): reused across encodes
    pub out_embd: Vec<f32>,
    /// `clip_set_debug_output_embeddings` (clip.cpp:6085)
    pub debug_output_embeddings: bool,
    /// `ctx->rng` (clip.cpp:178) — the gen-audio draw source. Ported
    /// bit-exactly (mt19937 + libstdc++'s `generate_canonical` /
    /// polar-method normal) so a fixed `seed` reproduces the reference's
    /// `inp_rand_*` / `inp_noise` draws.
    pub(crate) rng: Mt19937,
    /// `ctx->rng_seed` (clip.cpp:179) — reseed only when it changes
    /// (clip.cpp:4442-4445)
    pub(crate) rng_seed: u32,
    /// keeps the mmap alive for the whole context lifetime
    _mmap: Arc<memmap2::Mmap>,
}

// ======================================================================
// loader — clip.cpp:1158 `clip_model_loader`
// ======================================================================

/// Key macros, clip-impl.h:41-100.
mod keys {
    pub const PROJ_TYPE: &str = "clip.projector_type";
    pub const VISION_PROJ_TYPE: &str = "clip.vision.projector_type";
    #[allow(dead_code)]
    pub const HAS_VISION: &str = "clip.has_vision_encoder";
    pub const USE_GELU: &str = "clip.use_gelu";
    pub const USE_SILU: &str = "clip.use_silu";
    pub const IMAGE_SIZE: &str = "clip.vision.image_size";
    pub const PATCH_SIZE: &str = "clip.vision.patch_size";
    pub const IMAGE_MEAN: &str = "clip.vision.image_mean";
    pub const IMAGE_STD: &str = "clip.vision.image_std";
    pub const SPATIAL_MERGE_SIZE: &str = "clip.vision.spatial_merge_size";
    // ---- cohere2 / llava-uhd (clip-impl.h:53-62) -----------------------------
    /// clip.vision.image_min_pixels — qwen2.5-vl optional limit (clip.cpp:1704)
    pub const IMAGE_MIN_PIXELS: &str = "clip.vision.image_min_pixels";
    /// clip.vision.image_max_pixels — qwen2.5-vl optional limit (clip.cpp:1707)
    pub const IMAGE_MAX_PIXELS: &str = "clip.vision.image_max_pixels";
    /// clip.vision.preproc_max_tiles — cohere2v grid cap (clip.cpp:1523)
    pub const PREPROC_MAX_TILES: &str = "clip.vision.preproc_max_tiles";
    /// clip.vision.image_resize_algo — the lfm2/d1omni_v optional override
    /// (clip-impl.h:60, read at clip.cpp:1533-1543)
    #[allow(dead_code)] // its reading arm is the lfm2 vision family — gap open
    pub const IMAGE_RESIZE_ALGO: &str = "clip.vision.image_resize_algo";
    /// clip.vision.projector.scale_factor — cohere2v n_merge (clip.cpp:1522)
    pub const PROJ_SCALE_FACTOR: &str = "clip.vision.projector.scale_factor";
    pub const LAYER_NORM_EPS: &str = "clip.vision.attention.layer_norm_epsilon";
    pub const WIN_ATTN_PATTERN: &str = "clip.vision.attention.window_pattern";
    /// clip-impl.h:60 — `clip.%s.*` templates
    pub const N_EMBD: &str = "clip.vision.embedding_length";
    pub const N_FF: &str = "clip.vision.feed_forward_length";
    pub const N_BLOCK: &str = "clip.vision.block_count";
    pub const PROJ_DIM: &str = "clip.vision.projection_dim";
    pub const N_HEAD: &str = "clip.vision.attention.head_count";
    pub const N_HEAD_KV: &str = "clip.vision.attention.head_count_kv";
    pub const N_EMBD_HEAD: &str = "clip.vision.attention.head_dim";
    /// clip.audio.projector_type (clip-impl.h:80)
    pub const AUDIO_PROJ_TYPE: &str = "clip.audio.projector_type";
    /// clip.audio.num_mel_bins (clip-impl.h:81)
    pub const A_NUM_MEL_BINS: &str = "clip.audio.num_mel_bins";
    /// clip.audio.projector.stack_factor (clip-impl.h:82) — required for the
    /// StackAudioFrames projectors (clip.cpp:1782-1786)
    pub const A_PROJ_STACK_FACTOR: &str = "clip.audio.projector.stack_factor";
    // ---- audio round 4 (clip-impl.h:83-94) ----------------------------------
    pub const A_CHUNK_SIZE: &str = "clip.audio.chunk_size";
    pub const A_CONV_KERNEL_SIZE: &str = "clip.audio.conv_kernel_size";
    pub const A_MAX_POS_EMB: &str = "clip.audio.max_pos_emb";
    pub const A_PROJ_WINDOW_SIZE: &str = "clip.audio.projector.window_size";
    pub const A_PROJ_DOWNSAMPLE_RATE: &str = "clip.audio.projector.downsample_rate";
    pub const A_PROJ_HEAD_COUNT: &str = "clip.audio.projector.head_count";
    pub const AUDIO_SUBSMPL_FACTOR: &str = "clip.audio.subsampling_factor";
    pub const A_RVQ_NUM_QUANTIZERS: &str = "clip.audio.rvq.num_quantizers";
    pub const A_RVQ_CODEBOOK_SIZE: &str = "clip.audio.rvq.codebook_size";
    pub const A_WA_PATTERN_MODE: &str = "clip.audio.wa_pattern_mode";
    pub const A_ATTN_WINDOW_SIZE: &str = "clip.audio.window_size";
    pub const A_LOCAL_BLOCK_COUNT: &str = "clip.audio.local_block_count";
    pub const A_LOCAL_GROUP_SIZE: &str = "clip.audio.local_group_size";
    // ---- audio round 5 (clip-impl.h:33-36, :95-98) ---------------------------
    pub const HAS_GEN_AUDIO_ENC: &str = "clip.has_gen_audio_encoder";
    pub const GEN_AUDIO_PROJ_TYPE: &str = "clip.gen.audio.projector_type";
    pub const GEN_AUDIO_VARIANT: &str = "clip.gen.audio.model_variant";
}

/// Tensor name macros, clip-impl.h:105-163 (`prefix` == "v" for vision).
mod tnames {
    pub const POS_EMBD: &str = "v.position_embd.weight";
    pub const CLASS_EMBD: &str = "v.class_embd";
    /// clip-impl.h:107 — deliberately without a `.0` suffix for backward compat
    pub const PATCH_EMBD: &str = "v.patch_embd.weight";
    pub const PATCH_EMBD_1: &str = "v.patch_embd.weight.1";
    pub const PATCH_BIAS: &str = "v.patch_embd.bias";
    pub const LN_PRE: &str = "v.pre_ln";
    pub const LN_POST: &str = "v.post_ln";
    pub fn attn_qkv(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.attn_qkv.{suffix}")
    }
    pub fn attn_out(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.attn_out.{suffix}")
    }
    pub fn ln1(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.ln1.{suffix}")
    }
    pub fn ln2(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.ln2.{suffix}")
    }
    pub fn ffn_up(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.ffn_up.{suffix}")
    }
    pub fn ffn_down(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.ffn_down.{suffix}")
    }
    pub fn ffn_gate(il: i32, suffix: &str) -> String {
        format!("v.blk.{il}.ffn_gate.{suffix}")
    }
    pub fn mm(idx: i32, suffix: &str) -> String {
        format!("mm.{idx}.{suffix}")
    }
    /// clip-impl.h:195 TN_MM_AUDIO_MLP "mm.a.mlp.%d.%s"
    pub fn mm_audio_mlp(idx: i32, suffix: &str) -> String {
        format!("mm.a.mlp.{idx}.{suffix}")
    }
    /// clip-impl.h:196 TN_MM_AUDIO_FC "mm.a.fc.%s"
    pub fn mm_audio_fc(suffix: &str) -> String {
        format!("mm.a.fc.{suffix}")
    }
    /// clip-impl.h:197 TN_MM_NORM_PRE "mm.a.norm_pre.%s"
    pub fn mm_norm_pre(suffix: &str) -> String {
        format!("mm.a.norm_pre.{suffix}")
    }
    /// clip-impl.h:198 TN_MM_NORM_MID "mm.a.norm_mid.%s"
    pub fn mm_norm_mid(suffix: &str) -> String {
        format!("mm.a.norm_mid.{suffix}")
    }
    /// clip-impl.h:297-298 — the BOI/EOI embeddings keep their literal "v."
    /// prefix in audio models too (clip.cpp:3183-3184)
    pub const TOK_BOI: &str = "v.boi";
    pub const TOK_EOI: &str = "v.eoi";
}

struct Loader<'a> {
    gguf: &'a Gguf,
    ctx: Context,
    mmap: Arc<memmap2::Mmap>,
    /// clip.cpp:2158 `loaded_tensor_names` — a second request for the same name
    /// is a bug in the tensor table
    loaded: std::collections::HashSet<String>,
    n_loaded: usize,
}

impl<'a> Loader<'a> {
    fn new(gguf: &'a Gguf, mmap: Arc<memmap2::Mmap>) -> Self {
        Self {
            gguf,
            ctx: Context::new(),
            mmap,
            loaded: Default::default(),
            n_loaded: 0,
        }
    }

    /// clip.cpp:2161 `get_tensor`: create the context tensor with the file's own
    /// type/shape and point it at the mmap.
    fn get(&mut self, name: &str, required: bool) -> Result<Option<TensorId>, String> {
        if self.loaded.contains(name) {
            return Err(format!("tensor already loaded: {name}"));
        }
        let Some(ti) = self.gguf.find_tensor(name) else {
            if required {
                return Err(format!("unable to find tensor {name}"));
            }
            return Ok(None);
        };
        let id = self.ctx.new_tensor(ti.ty, ti.ne);
        self.ctx.set_name(id, name);
        let off = self.gguf.data_offset as usize + ti.offset as usize;
        self.ctx.set_external_storage(id, self.mmap.clone(), off);
        self.loaded.insert(name.to_string());
        self.n_loaded += 1;
        Ok(Some(id))
    }

    fn get_req(&mut self, name: &str) -> Result<TensorId, String> {
        self.get(name, true).map(|v| v.unwrap())
    }

    fn get_opt(&mut self, name: &str) -> Result<Option<TensorId>, String> {
        self.get(name, false)
    }

    /// clip.cpp:2191 `get_vector` — read an F32 tensor's raw bytes straight
    /// from the mmap (NOT registered in the graph tensor table). Returns an
    /// empty vec when absent.
    fn get_vector_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let Some(ti) = self.gguf.find_tensor(name) else {
            return Ok(Vec::new());
        };
        if ti.ty != GgmlType::F32 {
            return Err(format!("{name} must be F32, was {:?}", ti.ty));
        }
        let n = ti.ne.iter().product::<i64>() as usize;
        let off = self.gguf.data_offset as usize + ti.offset as usize;
        let bytes = &self.mmap[off..off + n * 4];
        Ok(bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect())
    }
}

/// clip.cpp:1244 `load_hparams` — vision modality. Returns the model with every
/// hparam the vision graph reads.
fn load_hparams(
    gguf: &Gguf,
    params: &ClipContextParams,
    modality: ClipModality,
) -> Result<(ClipHparams, ProjectorType), String> {
    let is_audio = modality == ClipModality::Audio;
    let is_gen_audio = modality == ClipModality::GenAudio;
    let mut hparams = ClipHparams::default();
    hparams.custom_image_min_tokens = params.image_min_tokens;
    hparams.custom_image_max_tokens = params.image_max_tokens;

    // projector type (clip.cpp:1260): `clip.projector_type`, falling back to the
    // modality-specific key for models with mixed modalities
    let mut proj = gguf.get_str(keys::PROJ_TYPE).unwrap_or("").to_string();
    if proj.is_empty() {
        // audio-only files carry the audio key (clip.cpp:1266); the gen-audio
        // context reads clip.gen.audio.projector_type (clip.cpp:1272)
        let fallback = if is_gen_audio {
            keys::GEN_AUDIO_PROJ_TYPE
        } else {
            keys::AUDIO_PROJ_TYPE
        };
        proj = gguf.get_str(fallback).unwrap_or("").to_string();
    }
    if proj.is_empty() && !is_gen_audio {
        proj = gguf
            .get_str(keys::VISION_PROJ_TYPE)
            .unwrap_or("")
            .to_string();
    }
    let mut proj_type = ProjectorType::from_str(&proj);
    if proj_type == ProjectorType::Unknown {
        return Err(format!("unknown projector type: {proj}"));
    }
    // legacy correction (clip.cpp:1284-1289): "qwen2.5o" resolves per modality
    if proj == "qwen2.5o" {
        proj_type = if is_audio {
            ProjectorType::Qwen2A
        } else {
            ProjectorType::Qwen25Vl
        };
    }

    // modality-prefixed keys (clip.cpp:1298): "vision" / "audio" / "gen.audio"
    let prefix = if is_gen_audio {
        "gen.audio"
    } else if is_audio {
        "audio"
    } else {
        "vision"
    };
    let pkey = |k: &str| k.replace("vision", prefix);

    // common hparams (clip.cpp:1297-1309)
    let gu32 = |k: &str| gguf.get_u32(k).map(|v| v as i32);
    hparams.n_embd = gu32(&pkey(keys::N_EMBD)).ok_or("missing clip.vision.embedding_length")?;
    hparams.n_head = gu32(&pkey(keys::N_HEAD)).ok_or("missing clip.vision.attention.head_count")?;
    hparams.n_embd_head = gu32(&pkey(keys::N_EMBD_HEAD)).unwrap_or(0);
    hparams.n_ff = gu32(&pkey(keys::N_FF)).ok_or("missing clip.vision.feed_forward_length")?;
    hparams.n_layer = gu32(&pkey(keys::N_BLOCK)).ok_or("missing clip.vision.block_count")?;
    hparams.projection_dim =
        gu32(&pkey(keys::PROJ_DIM)).ok_or("missing clip.vision.projection_dim")?;
    hparams.eps = gguf
        .get_f32(&pkey(keys::LAYER_NORM_EPS))
        .ok_or("missing layer_norm_epsilon")?;
    hparams.n_head_kv = gu32(&pkey(keys::N_HEAD_KV)).unwrap_or(hparams.n_head);

    if is_audio {
        // clip.cpp:1331-1336: mel bins; image_size/patch_size are unused but set
        hparams.n_mel_bins = gu32(keys::A_NUM_MEL_BINS).unwrap_or(0);
        hparams.image_size = 0;
        hparams.patch_size = 1;
    } else if is_gen_audio {
        // clip.cpp:1338-1343 — gen-audio skips the image keys entirely, only
        // the model variant is read
        hparams.image_size = 0;
        hparams.patch_size = 1;
        hparams.gen_model_variant =
            gguf.get_str(keys::GEN_AUDIO_VARIANT).unwrap_or("").to_string();
    } else {
        hparams.image_size = gu32(keys::IMAGE_SIZE).ok_or("missing clip.vision.image_size")?;
        hparams.patch_size = gu32(keys::PATCH_SIZE).ok_or("missing clip.vision.patch_size")?;
    }

    hparams.warmup_image_size = hparams.image_size;

    // ffn op selection (clip.cpp:1368)
    let use_gelu = gguf.get_bool(keys::USE_GELU).unwrap_or(false);
    let use_silu = gguf.get_bool(keys::USE_SILU).unwrap_or(false);
    if use_gelu && use_silu {
        return Err("both use_gelu and use_silu are set to true".into());
    }
    hparams.ffn_op = if use_gelu {
        FfnOp::Gelu
    } else if use_silu {
        FfnOp::Silu
    } else {
        FfnOp::GeluQuick
    };

    // image mean/std (clip.cpp:1396) — vision modality only
    if modality == ClipModality::Vision {
        let mean = gguf.get_f32_slice(keys::IMAGE_MEAN).unwrap_or_default();
        let std = gguf.get_f32_slice(keys::IMAGE_STD).unwrap_or_default();
        if mean.len() < 3 || std.len() < 3 {
            return Err(format!(
                "image_mean/image_std arrays must have at least 3 elements, got {} and {}",
                mean.len(),
                std.len()
            ));
        }
        hparams.image_mean = [mean[0], mean[1], mean[2]];
        hparams.image_std = [std[0], std[1], std[2]];
    }

    // per-projector params (clip.cpp:1661)
    match proj_type {
        ProjectorType::Qwen2A
        | ProjectorType::Ultravox
        | ProjectorType::Voxtral
        | ProjectorType::Meralion
        | ProjectorType::Glma
        | ProjectorType::MusicFlamingo
        | ProjectorType::Qwen3A => {
            // clip.cpp:1774-1793 (the ultravox/whisper family incl. qwen3a):
            // gelu_erf FFN, whisper preprocessing params; stack_factor only
            // required for ultravox/voxtral/meralion/glma
            let require_stack = matches!(
                proj_type,
                ProjectorType::Ultravox
                    | ProjectorType::Voxtral
                    | ProjectorType::Meralion
                    | ProjectorType::Glma
            );
            if require_stack && gguf.get_u32(keys::A_PROJ_STACK_FACTOR).is_none() {
                // clip.cpp:1786 get_u32(..., require_stack) throws "Key not found"
                return Err(format!("Key not found: {}", keys::A_PROJ_STACK_FACTOR));
            }
            hparams.proj_stack_factor = gu32(keys::A_PROJ_STACK_FACTOR).unwrap_or(0);
            hparams.ffn_op = FfnOp::GeluErf;
            hparams.audio_chunk_len = 30; // in seconds
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 400;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
            // clip.cpp:2095: FFT-based preprocessors need [1, 256] mel bins
            if hparams.n_mel_bins <= 0 || hparams.n_mel_bins > 256 {
                return Err(format!(
                    "n_mel_bins ({}) must be in range [1, 256]",
                    hparams.n_mel_bins
                ));
            }
        }
        ProjectorType::Qwen2Vl | ProjectorType::Qwen25Vl | ProjectorType::Qwen3Vl
        | ProjectorType::Ling3Vl => {
            hparams.n_merge = gu32(keys::SPATIAL_MERGE_SIZE).unwrap_or(2);
            hparams.image_resize_algo = ResizeAlgo::Bicubic;
            // KEY_WIN_ATTN_PATTERN is required for qwen2.5-vl; not read here —
            // the window-attention branch of the graph is not ported
            // (clip.cpp:1666, models/qwen2vl.cpp:37-99).
            if gguf.get_u32(keys::WIN_ATTN_PATTERN).is_some() {
                log_line(&format!(
                    "warning: {} has window attention, which this port does not implement",
                    proj_type.name()
                ));
            }
            hparams.set_limit_image_tokens(8, 4096);
            // clip.cpp:1704-1713 — optional model-provided pixel limits; the
            // custom (context-param) values take precedence
            if hparams.custom_image_min_tokens <= 0 {
                hparams.image_min_pixels = gu32(keys::IMAGE_MIN_PIXELS)
                    .unwrap_or(hparams.image_min_pixels);
            }
            if hparams.custom_image_max_tokens <= 0 {
                hparams.image_max_pixels = gu32(keys::IMAGE_MAX_PIXELS)
                    .unwrap_or(hparams.image_max_pixels);
            }
            hparams.set_warmup_n_tokens(46 * 46);
        }
        ProjectorType::Cohere2V => {
            // clip.cpp:1520-1528 — tiles stretch the image to the grid, the
            // refined image is never padded
            hparams.image_pad_rf = PadStyle::None;
            hparams.n_merge = gu32(keys::PROJ_SCALE_FACTOR)
                .ok_or("Key not found: clip.vision.projector.scale_factor")?;
            hparams.preproc_max_tiles = gu32(keys::PREPROC_MAX_TILES)
                .ok_or("Key not found: clip.vision.preproc_max_tiles")?;
            if hparams.preproc_max_tiles <= 0 || hparams.preproc_max_tiles > 256 {
                return Err(format!(
                    "load_hparams: preproc_max_tiles ({}) must be in range [1, 256]",
                    hparams.preproc_max_tiles
                ));
            }
        }
        ProjectorType::Gemma4UA => {
            // clip.cpp:1968-1974: encoder-free — raw 16 kHz waveform chunked
            // into 640-sample frames, no FFT/filterbank
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.eps = 1e-6;
            hparams.n_mel_bins = 640;
        }
        ProjectorType::Lfm2A | ProjectorType::D1OmniA => {
            // clip.cpp:1951-1959 (clip.cpp:1990: d1omni_a shares the arm)
            hparams.audio_chunk_len = 1; // in seconds
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        }
        ProjectorType::Gemma4A => {
            // clip.cpp:1955-1966: frame 320 (20 ms), n_fft 512, hop 160
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 320;
            hparams.audio_hop_len = 160;
            hparams.eps = 1e-6; // hardcoded (conversion bug workaround)
        }
        ProjectorType::GraniteSpeech => {
            // clip.cpp:1976-1994
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
            let g = |k: &str| gguf.get_u32(k).map(|v| v as i32);
            hparams.audio_chunk_size =
                g(keys::A_CHUNK_SIZE).ok_or("Key not found: clip.audio.chunk_size")?;
            if !(1..=8192).contains(&hparams.audio_chunk_size) {
                return Err(format!(
                    "audio_chunk_size ({}) must be in [1, 8192]",
                    hparams.audio_chunk_size
                ));
            }
            hparams.audio_conv_kernel_size =
                g(keys::A_CONV_KERNEL_SIZE).ok_or("Key not found: clip.audio.conv_kernel_size")?;
            hparams.audio_max_pos_emb =
                g(keys::A_MAX_POS_EMB).ok_or("Key not found: clip.audio.max_pos_emb")?;
            hparams.audio_proj_window_size = g(keys::A_PROJ_WINDOW_SIZE)
                .ok_or("Key not found: clip.audio.projector.window_size")?;
            hparams.audio_proj_downsample_rate = g(keys::A_PROJ_DOWNSAMPLE_RATE)
                .ok_or("Key not found: clip.audio.projector.downsample_rate")?;
            hparams.audio_proj_head_count = g(keys::A_PROJ_HEAD_COUNT)
                .ok_or("Key not found: clip.audio.projector.head_count")?;
            // feature layers loaded in the common optional path
            // (clip.cpp:1412, KEY_FEATURE_LAYERS with the audio prefix)
            hparams.feature_layers = get_arr_i32(gguf, &format!("clip.{}.feature_layer", prefix));
            for v in &hparams.feature_layers {
                if *v < -1 || *v > hparams.n_layer {
                    return Err(format!(
                        "feature layer {v} is out of range (n_layer: {})",
                        hparams.n_layer
                    ));
                }
            }
        }
        ProjectorType::Parakeet => {
            // clip.cpp:1488-1500
            hparams.subsampling_factor =
                gguf.get_u32(keys::AUDIO_SUBSMPL_FACTOR)
                    .ok_or("Key not found: clip.audio.subsampling_factor")? as i32;
            if hparams.subsampling_factor != 8 {
                return Err(
                    "subsampling_factor must match the conv strides in clip_graph_parakeet::build()"
                        .into(),
                );
            }
            hparams.audio_conv_kernel_size =
                gguf.get_u32(keys::A_CONV_KERNEL_SIZE)
                    .ok_or("Key not found: clip.audio.conv_kernel_size")? as i32;
            if hparams.audio_conv_kernel_size <= 0 || hparams.audio_conv_kernel_size % 2 == 0 {
                return Err("audio_conv_kernel_size must be a positive odd integer".into());
            }
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        }
        ProjectorType::MimoAudio => {
            // clip.cpp:1795-1834
            hparams.rvq_num_quantizers = gguf
                .get_u32(keys::A_RVQ_NUM_QUANTIZERS)
                .map(|v| v as i32)
                .unwrap_or(0);
            hparams.rvq_codebook_size = get_arr_i32(gguf, keys::A_RVQ_CODEBOOK_SIZE);
            if hparams.rvq_num_quantizers <= 0 {
                return Err("mimo_audio: missing clip.audio.rvq.num_quantizers".into());
            }
            if hparams.rvq_codebook_size.len() != hparams.rvq_num_quantizers as usize {
                return Err(format!(
                    "mimo_audio: clip.audio.rvq.codebook_size length ({}) must equal clip.audio.rvq.num_quantizers ({})",
                    hparams.rvq_codebook_size.len(),
                    hparams.rvq_num_quantizers
                ));
            }
            hparams.ffn_op = FfnOp::GeluErf; // PyTorch F.gelu default
            hparams.audio_sample_rate = 24000;
            hparams.audio_n_fft = 960;
            hparams.audio_window_len = 960;
            hparams.audio_hop_len = 240;
            hparams.attn_window_size =
                gguf.get_u32(keys::A_ATTN_WINDOW_SIZE)
                    .ok_or("Key not found: clip.audio.window_size")? as i32;
            hparams.wa_pattern_mode = get_arr_i32(gguf, keys::A_WA_PATTERN_MODE);
            if hparams.wa_pattern_mode.len() != hparams.n_layer as usize {
                return Err(format!(
                    "mimo_audio: clip.audio.wa_pattern_mode length ({}) must equal n_layer ({})",
                    hparams.wa_pattern_mode.len(),
                    hparams.n_layer
                ));
            }
            hparams.audio_local_n_layer =
                gguf.get_u32(keys::A_LOCAL_BLOCK_COUNT)
                    .ok_or("Key not found: clip.audio.local_block_count")? as i32;
            hparams.audio_local_group_size =
                gguf.get_u32(keys::A_LOCAL_GROUP_SIZE)
                    .ok_or("Key not found: clip.audio.local_group_size")? as i32;
            if hparams.audio_local_group_size <= 0 {
                return Err("mimo_audio: clip.audio.local_group_size must be > 0".into());
            }
        }
        ProjectorType::Qwen3TtsSpkEnc => {
            // clip.cpp:1843-1849 — ECAPA-TDNN mel front-end, Slaney defaults
            hparams.audio_sample_rate = 24000;
            hparams.audio_n_fft = 1024;
            hparams.audio_window_len = 1024;
            hparams.audio_hop_len = 256;
        }
        ProjectorType::PocketTtsSpkEnc => {
            // clip.cpp:1875-1891 (shared with pockettts_gen)
            hparams.audio_sample_rate = 24000;
            hparams.seanet_ratios = vec![4, 5, 6]; // encoder reverses [6,5,4]
            hparams.seanet_n_stage = 3;
            hparams.mimi_downsample = 16;
            hparams.mimi_tfm_context = 250;
        }
        ProjectorType::Qwen3TtsGen => {
            // clip.cpp:1843-1860 — hardcoded from the reference configs
            hparams.rope_theta = 1000000.0;
            hparams.wav_tfm_n_layer = 8;
            hparams.wav_tfm_n_embd = 512;
            hparams.wav_tfm_n_ff = 1024;
            hparams.wav_tfm_n_head = 16;
            hparams.wav_tfm_n_head_kv = 16;
            hparams.wav_tfm_eps = 1e-5;
            hparams.wav_tfm_rope_theta = 10000.0;
            hparams.wav_upsample_n_block = 2;
            hparams.wav_dac_n_block = 4;
            hparams.wav_dac_n_res = 3;
            // matches the reference decoder's sliding_window
            hparams.wav_tfm_swa = 72;
        }
        ProjectorType::PocketTtsGen => {
            // clip.cpp:1875-1891 (shared with pockettts_spkenc)
            hparams.audio_sample_rate = 24000;
            hparams.seanet_ratios = vec![4, 5, 6];
            hparams.seanet_n_stage = 3;
            hparams.mimi_downsample = 16;
            hparams.mimi_tfm_context = 250;
            hparams.rope_theta = 10000.0;
            // flow_lm defaults, pocket_tts/default_parameters.py
            hparams.flow_n_step = 1;
            hparams.gen_eos_threshold = -4.0;
        }
        ProjectorType::Unknown => unreachable!(),
    }

    // sanity check (clip.cpp:2019)
    if hparams.image_size < 0 {
        return Err(format!(
            "image_size ({}) cannot be negative",
            hparams.image_size
        ));
    }
    if hparams.image_size > 8192 {
        return Err(format!(
            "image_size ({}) is too large (max 8192)",
            hparams.image_size
        ));
    }
    if hparams.patch_size <= 0 || hparams.patch_size >= 65536 {
        return Err(format!(
            "patch_size ({}) must be positive and less than 65536",
            hparams.patch_size
        ));
    }
    if hparams.n_embd <= 0 {
        return Err(format!(
            "n_embd ({}) must be greater than 0",
            hparams.n_embd
        ));
    }
    if hparams.image_max_pixels < hparams.image_min_pixels {
        return Err(format!(
            "image_max_pixels ({}) is less than image_min_pixels ({})",
            hparams.image_max_pixels, hparams.image_min_pixels
        ));
    }
    if hparams.n_merge <= 0 || hparams.n_merge >= 65536 {
        return Err(format!(
            "n_merge ({}) must be greater than 0 and less than 65536",
            hparams.n_merge
        ));
    }

    Ok((hparams, proj_type))
}

/// clip.cpp:1533-1543 — the `KEY_IMAGE_RESIZE_ALGO` string parse (the
/// lfm2/d1omni_v hparams arm). The reading arm itself lands with the lfm2
/// vision tower family (documented gap); the parse is ported and unit-tested
/// so the key's contract is pinned.
fn parse_resize_algo(resize_algo: &str) -> Result<ResizeAlgo, String> {
    match resize_algo {
        "bilinear" => Ok(ResizeAlgo::Bilinear),
        "bicubic" => Ok(ResizeAlgo::Bicubic),
        "lanczos" => Ok(ResizeAlgo::Lanczos),
        _ => Err(format!("unsupported image resize algo: {resize_algo}")),
    }
}

fn log_line(s: &str) {
    eprintln!("clip: {s}");
}

/// `get_arr_int` (clip.cpp:1290) — an int-array KV value
fn get_arr_i32(gguf: &Gguf, key: &str) -> Vec<i32> {
    match gguf.find_key(key) {
        Some(ggml::Value::Array(_, v)) => v
            .iter()
            .filter_map(|x| match x {
                ggml::Value::I32(i) => Some(*i),
                ggml::Value::U32(u) => Some(*u as i32),
                ggml::Value::I64(i) => Some(*i as i32),
                ggml::Value::U64(u) => Some(*u as i32),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// an all-None `ClipLayer` for the archs that own their layer arrays
/// (qwen3tts spkenc blocks, mimo's input_local_transformer)
fn empty_layer() -> ClipLayer {
    ClipLayer {
        q_norm: None,
        k_norm: None,
        ln_1_w: None,
        ln_1_b: None,
        q_w: None,
        q_b: None,
        k_w: None,
        k_b: None,
        v_w: None,
        v_b: None,
        qkv_w: None,
        qkv_b: None,
        o_w: None,
        o_b: None,
        ff_up_w: None,
        ff_up_b: None,
        ff_gate_w: None,
        ff_gate_b: None,
        ff_down_w: None,
        ff_down_b: None,
        ln_2_w: None,
        ln_2_b: None,
        ff_norm_w: None,
        ff_norm_b: None,
        ff_norm_1_w: None,
        ff_norm_1_b: None,
        ff_up_1_w: None,
        ff_up_1_b: None,
        ff_down_1_w: None,
        ff_down_1_b: None,
        pos_bias_u: None,
        pos_bias_v: None,
        linear_pos_w: None,
        norm_conv_w: None,
        norm_conv_b: None,
        conv_norm_w: None,
        conv_norm_b: None,
        conv_dw_w: None,
        conv_dw_b: None,
        conv_pw1_w: None,
        conv_pw1_b: None,
        conv_pw2_w: None,
        conv_pw2_b: None,
        attn_pre_norm_w: None,
        attn_post_norm_w: None,
        per_dim_scale_w: None,
        per_dim_k_scale_w: None,
        attn_k_rel_w: None,
        ff_post_norm_w: None,
        ff_post_norm_1_w: None,
        attn_rel_pos_emb: None,
        conv_norm_mean: None,
        conv_norm_var: None,
        ls_1_w: None,
        ls_2_w: None,
        se_conv1_w: None,
        se_conv1_b: None,
        se_conv2_w: None,
        se_conv2_b: None,
        res2_conv_w: Vec::new(),
        res2_conv_b: Vec::new(),
    }
}

/// clip.cpp:2380-2500 — the common vision tensor block plus the
/// qwen2vl/qwen2.5vl/qwen3vl projector table (clip.cpp:2475).
fn load_tensors(loader: &mut Loader, model: &mut ClipModel) -> Result<(), String> {
    let proj = model.proj_type;
    // the tensor-name prefix follows the modality (clip.cpp:2224 uses
    // `prefix` = "v" for vision / "a" for audio / "a.gen.code" for the
    // gen-audio code_predictor (clip.cpp:2122-2124)
    let pfx = match model.modality {
        ClipModality::Audio => "a",
        ClipModality::GenAudio => "a.gen.code",
        ClipModality::Vision => "v",
    };
    let pname = |k: &str| k.replacen("v.", &format!("{pfx}."), 1);

    // clip.cpp:2224 — common block
    model.class_embedding = loader.get_opt(&pname(tnames::CLASS_EMBD))?;
    model.pre_ln_w = loader.get_opt(&pname(&format!("{}.weight", tnames::LN_PRE)))?;
    model.pre_ln_b = loader.get_opt(&pname(&format!("{}.bias", tnames::LN_PRE)))?;
    model.post_ln_w = loader.get_opt(&pname(&format!("{}.weight", tnames::LN_POST)))?;
    model.post_ln_b = loader.get_opt(&pname(&format!("{}.bias", tnames::LN_POST)))?;

    model.patch_bias = loader.get_opt(tnames::PATCH_BIAS)?;
    model.patch_embeddings_0 = loader.get_opt(tnames::PATCH_EMBD)?;
    model.patch_embeddings_1 = loader.get_opt(tnames::PATCH_EMBD_1)?;
    model.position_embeddings = loader.get_opt(&pname(tnames::POS_EMBD))?;

    let n_layer = model.hparams.n_layer;
    // clip.cpp:2249 — the standard-layer table is skipped for the archs whose
    // layer arrays mean something else (qwen3tts_spkenc owns its layers;
    // pockettts_gen keeps its transformer in gen_tfm_layers)
    let has_standard_layers =
        !matches!(proj, ProjectorType::Qwen3TtsSpkEnc | ProjectorType::PocketTtsGen);
    let n_layers_to_load = if has_standard_layers { n_layer } else { 0 };
    model.layers.reserve(n_layers_to_load as usize);
    for il in 0..n_layers_to_load {
        let layer = ClipLayer {
            // qwen3tts_gen's code_predictor qk-norms (clip.cpp:2265-2266,
            // TN_ATTN_Q_NORM / TN_ATTN_K_NORM in the generic layer loop)
            q_norm: loader.get_opt(&pname(&format!("v.blk.{il}.attn_q_norm.weight")))?,
            k_norm: loader.get_opt(&pname(&format!("v.blk.{il}.attn_k_norm.weight")))?,
            q_w: loader.get_opt(&pname(&format!("v.blk.{il}.attn_q.weight")))?,
            q_b: loader.get_opt(&pname(&format!("v.blk.{il}.attn_q.bias")))?,
            k_w: loader.get_opt(&pname(&format!("v.blk.{il}.attn_k.weight")))?,
            k_b: loader.get_opt(&pname(&format!("v.blk.{il}.attn_k.bias")))?,
            v_w: loader.get_opt(&pname(&format!("v.blk.{il}.attn_v.weight")))?,
            v_b: loader.get_opt(&pname(&format!("v.blk.{il}.attn_v.bias")))?,
            ln_1_w: loader.get_opt(&pname(&tnames::ln1(il, "weight")))?,
            ln_1_b: loader.get_opt(&pname(&tnames::ln1(il, "bias")))?,
            ln_2_w: loader.get_opt(&pname(&tnames::ln2(il, "weight")))?,
            ln_2_b: loader.get_opt(&pname(&tnames::ln2(il, "bias")))?,
            qkv_w: loader.get_opt(&pname(&tnames::attn_qkv(il, "weight")))?,
            qkv_b: loader.get_opt(&pname(&tnames::attn_qkv(il, "bias")))?,
            // clip.cpp:2264 — attn_out.weight is required
            o_w: Some(loader.get_req(&pname(&tnames::attn_out(il, "weight")))?),
            o_b: loader.get_opt(&pname(&tnames::attn_out(il, "bias")))?,
            ff_up_w: loader.get_opt(&pname(&tnames::ffn_up(il, "weight")))?,
            ff_up_b: loader.get_opt(&pname(&tnames::ffn_up(il, "bias")))?,
            ff_gate_w: loader.get_opt(&pname(&tnames::ffn_gate(il, "weight")))?,
            ff_gate_b: loader.get_opt(&pname(&tnames::ffn_gate(il, "bias")))?,
            ff_down_w: loader.get_opt(&pname(&tnames::ffn_down(il, "weight")))?,
            ff_down_b: loader.get_opt(&pname(&tnames::ffn_down(il, "bias")))?,
            // ls_1/ls_2 (pockettts layer scale, TN_LS_1/TN_LS_2)
            // ls1/ls2 layer scale (TN_LS_1/TN_LS_2 — "ls1", no underscore)
            ls_1_w: loader.get_opt(&pname(&format!("v.blk.{il}.ls1.weight")))?,
            ls_2_w: loader.get_opt(&pname(&format!("v.blk.{il}.ls2.weight")))?,
            // conformer-family tensors load in the per-arch arm below
            ff_norm_w: None,
            ff_norm_b: None,
            ff_norm_1_w: None,
            ff_norm_1_b: None,
            ff_up_1_w: None,
            ff_up_1_b: None,
            ff_down_1_w: None,
            ff_down_1_b: None,
            pos_bias_u: None,
            pos_bias_v: None,
            linear_pos_w: None,
            norm_conv_w: None,
            norm_conv_b: None,
            conv_norm_w: None,
            conv_norm_b: None,
            conv_dw_w: None,
            conv_dw_b: None,
            conv_pw1_w: None,
            conv_pw1_b: None,
            conv_pw2_w: None,
            conv_pw2_b: None,
            attn_pre_norm_w: None,
            attn_post_norm_w: None,
            per_dim_scale_w: None,
            per_dim_k_scale_w: None,
            attn_k_rel_w: None,
            ff_post_norm_w: None,
            ff_post_norm_1_w: None,
            attn_rel_pos_emb: None,
            conv_norm_mean: None,
            conv_norm_var: None,
            se_conv1_w: None,
            se_conv1_b: None,
            se_conv2_w: None,
            se_conv2_b: None,
            res2_conv_w: Vec::new(),
            res2_conv_b: Vec::new(),
        };
        if layer.ff_up_w.is_none() || layer.ff_down_w.is_none() {
            return Err(format!("blk.{il}: ffn_up/ffn_down weight missing"));
        }
        // the merger graph uses fused qkv; the whisper-enc graph uses separate
        // q/k/v (build_vit's else branch, clip.cpp:412-437) and the gen-audio
        // code_predictor likewise (qwen3tts-gen.cpp:148-150), so those layers
        // legitimately have no attn_qkv.weight — the siglip tower (cohere2v)
        // too: HF SigLIP keeps separate q/k/v projections and the converter
        // does not fuse them (conversion fuse_qkv defaults off)
        if layer.qkv_w.is_none()
            && model.modality == ClipModality::Vision
            && !matches!(model.proj_type, ProjectorType::Cohere2V)
        {
            return Err(format!(
                "blk.{il}: attn_qkv.weight missing (separate q/k/v not ported)"
            ));
        }
        if model.proj_type == ProjectorType::Cohere2V
            && (layer.q_w.is_none()
                || layer.k_w.is_none()
                || layer.v_w.is_none()
                || layer.ln_1_w.is_none()
                || layer.ln_2_w.is_none())
        {
            // the siglip tower runs build_vit's separate-qkv branch
            // (clip.cpp:412-437): attn_{q,k,v} + ln1/ln2 are required
            return Err(format!(
                "blk.{il}: siglip tower needs separate attn_q/attn_k/attn_v + ln1/ln2"
            ));
        }
        model.layers.push(layer);
    }

    // projector-specific tensors (clip.cpp:2530). The whisper-family arms use
    // required loads (C `get_tensor`) — a missing tensor fails the load.
    match proj {
        ProjectorType::Qwen2A => {
            // clip.cpp:2852-2862: conv1d pair + the qwen2a fc projector
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_fc_w = Some(loader.get_req(&tnames::mm_audio_fc("weight"))?);
            model.mm_fc_b = Some(loader.get_req(&tnames::mm_audio_fc("bias"))?);
        }
        ProjectorType::Ultravox => {
            // clip.cpp:2817-2827
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
            model.mm_norm_pre_w = Some(loader.get_req(&tnames::mm_norm_pre("weight"))?);
            model.mm_norm_mid_w = Some(loader.get_req(&tnames::mm_norm_mid("weight"))?);
        }
        ProjectorType::Meralion => {
            // clip.cpp:2828-2850: whisper conv layers + the 4-linear adaptor
            // (linear_0 frame compression + SiLU, linear_1 gate, linear_2 pool,
            // linear_3 out) and ln_speech
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_0_w = Some(loader.get_req(&tnames::mm_audio_mlp(0, "weight"))?);
            model.mm_0_b = Some(loader.get_req(&tnames::mm_audio_mlp(0, "bias"))?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm_audio_mlp(1, "bias"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
            model.mm_2_b = Some(loader.get_req(&tnames::mm_audio_mlp(2, "bias"))?);
            model.mm_3_w = Some(loader.get_req(&tnames::mm_audio_mlp(3, "weight"))?);
            model.mm_3_b = Some(loader.get_req(&tnames::mm_audio_mlp(3, "bias"))?);
            model.mm_norm_pre_w = Some(loader.get_req(&tnames::mm_norm_pre("weight"))?);
            model.mm_norm_pre_b = Some(loader.get_req(&tnames::mm_norm_pre("bias"))?);
        }
        ProjectorType::Voxtral => {
            // clip.cpp:3136-3142 (no projector biases)
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
        }
        ProjectorType::MusicFlamingo => {
            // clip.cpp:3145-3154
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm_audio_mlp(1, "bias"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
            model.mm_2_b = Some(loader.get_req(&tnames::mm_audio_mlp(2, "bias"))?);
        }
        ProjectorType::Glma => {
            // clip.cpp:3171-3184
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm_audio_mlp(1, "bias"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
            model.mm_2_b = Some(loader.get_req(&tnames::mm_audio_mlp(2, "bias"))?);
            model.mm_norm_pre_w = Some(loader.get_req(&tnames::mm_norm_pre("weight"))?);
            model.mm_norm_pre_b = Some(loader.get_req(&tnames::mm_norm_pre("bias"))?);
            model.mm_boi = Some(loader.get_req(tnames::TOK_BOI)?);
            model.mm_eoi = Some(loader.get_req(tnames::TOK_EOI)?);
        }
        ProjectorType::Qwen2Vl | ProjectorType::Qwen25Vl | ProjectorType::Qwen3Vl => {
            model.mm_0_w = loader.get_opt(&tnames::mm(0, "weight"))?;
            model.mm_0_b = loader.get_opt(&tnames::mm(0, "bias"))?;
            model.mm_1_w = loader.get_opt(&tnames::mm(2, "weight"))?;
            model.mm_1_b = loader.get_opt(&tnames::mm(2, "bias"))?;
            if model.mm_0_w.is_none() || model.mm_1_w.is_none() {
                return Err("mm.0.weight / mm.2.weight missing".into());
            }
        }
        // cohere2v (clip.cpp:2802-2809): the two-layer llava-style projector,
        // TN_LLAVA_PROJ mm.{1,2}.{weight,bias} — all four required
        ProjectorType::Cohere2V => {
            model.mm_1_w = Some(loader.get_req(&tnames::mm(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm(1, "bias"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm(2, "weight"))?);
            model.mm_2_b = Some(loader.get_req(&tnames::mm(2, "bias"))?);
        }
        // ling3vl (clip.cpp:2509-2518): the merger is norm-only — the tower's
        // merger.norm plus the top-level mm.0/mm.2 projector MLP
        ProjectorType::Ling3Vl => {
            model.mm_input_norm_w = Some(loader.get_req("mm.input_norm.weight")?);
            model.mm_input_norm_b = Some(loader.get_req("mm.input_norm.bias")?);
            model.mm_0_w = Some(loader.get_req(&tnames::mm(0, "weight"))?);
            model.mm_0_b = Some(loader.get_req(&tnames::mm(0, "bias"))?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm(2, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm(2, "bias"))?);
        }
        ProjectorType::Qwen3A => {
            // clip.cpp:2861-2874: 3x conv2d + conv_out + the MLP projector
            for i in 0..3 {
                model.conv2d_w[i] = Some(loader.get_req(&format!("a.conv2d.{}.weight", i + 1))?);
                model.conv2d_b[i] = Some(loader.get_req(&format!("a.conv2d.{}.bias", i + 1))?);
            }
            model.conv_out_w = Some(loader.get_req("a.conv_out.weight")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm_audio_mlp(1, "bias"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
            model.mm_2_b = Some(loader.get_req(&tnames::mm_audio_mlp(2, "bias"))?);
        }
        ProjectorType::Gemma4UA => {
            // clip.cpp:3343-3346 — TN_A_MM_INP_PROJ
            model.mm_input_proj_w = Some(loader.get_req("mm.a.input_projection.weight")?);
        }
        ProjectorType::Lfm2A | ProjectorType::D1OmniA => {
            // clip.cpp:3360-3410: pre-encode conv stack + out projection
            // (clip.cpp:3423: d1omni_a shares the arm)
            for i in [0usize, 2, 3, 5, 6] {
                model.pre_conv_w[i] = Some(loader.get_req(&format!("a.conv1d.{i}.weight"))?);
                model.pre_conv_b[i] = Some(loader.get_req(&format!("a.conv1d.{i}.bias"))?);
            }
            model.pre_encode_out_w = Some(loader.get_req("a.pre_encode.out.weight")?);
            model.pre_encode_out_b = Some(loader.get_req("a.pre_encode.out.bias")?);
            // audio adapter (clip.cpp:3371-3377): mm.a.mlp.0/1/3
            model.mm_0_w = Some(loader.get_req(&tnames::mm_audio_mlp(0, "weight"))?);
            model.mm_0_b = Some(loader.get_req(&tnames::mm_audio_mlp(0, "bias"))?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_1_b = Some(loader.get_req(&tnames::mm_audio_mlp(1, "bias"))?);
            model.mm_3_w = Some(loader.get_req(&tnames::mm_audio_mlp(3, "weight"))?);
            model.mm_3_b = Some(loader.get_req(&tnames::mm_audio_mlp(3, "bias"))?);
            // clip.cpp:3439-3447 — d1omni_a's residual block after the
            // projector: norm, down, up
            if proj == ProjectorType::D1OmniA {
                model.mm_4_w = Some(loader.get_req(&tnames::mm_audio_mlp(4, "weight"))?);
                model.mm_4_b = Some(loader.get_req(&tnames::mm_audio_mlp(4, "bias"))?);
                model.mm_5_w = Some(loader.get_req(&tnames::mm_audio_mlp(5, "weight"))?);
                model.mm_5_b = Some(loader.get_req(&tnames::mm_audio_mlp(5, "bias"))?);
                model.mm_6_w = Some(loader.get_req(&tnames::mm_audio_mlp(6, "weight"))?);
                model.mm_6_b = Some(loader.get_req(&tnames::mm_audio_mlp(6, "bias"))?);
            }
            // per-layer conformer tensors (clip.cpp:3379-3406)
            for il in 0..n_layer {
                let layer = &mut model.layers[il as usize];
                layer.ff_norm_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.weight"))?);
                layer.ff_norm_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.bias"))?);
                layer.ff_norm_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.weight"))?);
                layer.ff_norm_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.bias"))?);
                layer.ff_up_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.weight"))?);
                layer.ff_up_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.bias"))?);
                layer.ff_down_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.weight"))?);
                layer.ff_down_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.bias"))?);
                layer.pos_bias_u = Some(loader.get_req(&format!("a.blk.{il}.pos_bias_u"))?);
                layer.pos_bias_v = Some(loader.get_req(&format!("a.blk.{il}.pos_bias_v"))?);
                layer.norm_conv_w = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.weight"))?);
                layer.norm_conv_b = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.bias"))?);
                layer.linear_pos_w =
                    Some(loader.get_req(&format!("a.blk.{il}.linear_pos.weight"))?);
                layer.conv_norm_w = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.weight"))?);
                layer.conv_norm_b = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.bias"))?);
                layer.conv_dw_w = Some(loader.get_req(&format!("a.blk.{il}.conv_dw.weight"))?);
                layer.conv_dw_b = Some(loader.get_req(&format!("a.blk.{il}.conv_dw.bias"))?);
                layer.conv_pw1_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.weight"))?);
                layer.conv_pw1_b = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.bias"))?);
                layer.conv_pw2_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.weight"))?);
                layer.conv_pw2_b = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.bias"))?);
            }
        }
        ProjectorType::GraniteSpeech => {
            // clip.cpp:3455-3530
            model.inp_proj_w = Some(loader.get_req("a.input_projection.weight")?);
            model.inp_proj_b = Some(loader.get_req("a.input_projection.bias")?);
            model.ctc_out_w = Some(loader.get_req("a.enc_ctc_out.weight")?);
            model.ctc_out_b = Some(loader.get_req("a.enc_ctc_out.bias")?);
            model.ctc_out_mid_w = Some(loader.get_req("a.enc_ctc_out_mid.weight")?);
            model.ctc_out_mid_b = Some(loader.get_req("a.enc_ctc_out_mid.bias")?);
            for il in 0..n_layer {
                let layer = &mut model.layers[il as usize];
                layer.attn_rel_pos_emb =
                    Some(loader.get_req(&format!("a.blk.{il}.attn_rel_pos_emb"))?);
                layer.ff_norm_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.weight"))?);
                layer.ff_norm_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.bias"))?);
                layer.ff_norm_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.weight"))?);
                layer.ff_norm_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.bias"))?);
                layer.ff_up_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.weight"))?);
                layer.ff_up_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.bias"))?);
                layer.ff_down_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.weight"))?);
                layer.ff_down_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.bias"))?);
                layer.norm_conv_w = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.weight"))?);
                layer.norm_conv_b = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.bias"))?);
                layer.conv_norm_w = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.weight"))?);
                layer.conv_norm_b = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.bias"))?);
                layer.conv_dw_w = Some(loader.get_req(&format!("a.blk.{il}.conv_dw.weight"))?);
                layer.conv_pw1_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.weight"))?);
                layer.conv_pw1_b = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.bias"))?);
                layer.conv_pw2_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.weight"))?);
                layer.conv_pw2_b = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.bias"))?);
            }
            // the single QFormer block (clip.cpp:3494-3530)
            let mut qf = Box::new(QFormerBlock::default());
            qf.qf_proj_query = Some(loader.get_req("a.proj_query")?);
            qf.qf_proj_norm_w = Some(loader.get_req("a.proj_norm.weight")?);
            qf.qf_proj_norm_b = Some(loader.get_req("a.proj_norm.bias")?);
            qf.qf_proj_linear_w = Some(loader.get_req("a.proj_linear.weight")?);
            qf.qf_proj_linear_b = Some(loader.get_req("a.proj_linear.bias")?);
            for il in 0..2 {
                let mut pl = QFormerLayer::default();
                pl.q_w = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_q.weight"))?);
                pl.q_b = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_q.bias"))?);
                pl.k_w = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_k.weight"))?);
                pl.k_b = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_k.bias"))?);
                pl.v_w = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_v.weight"))?);
                pl.v_b = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_v.bias"))?);
                pl.o_w = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_out.weight"))?);
                pl.o_b = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_out.bias"))?);
                pl.ln_1_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_norm.weight"))?);
                pl.ln_1_b = Some(loader.get_req(&format!("a.proj_blk.{il}.self_attn_norm.bias"))?);
                pl.cross_attn_q_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_q.weight"))?);
                pl.cross_attn_q_b =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_q.bias"))?);
                pl.cross_attn_k_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_k.weight"))?);
                pl.cross_attn_k_b =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_k.bias"))?);
                pl.cross_attn_v_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_v.weight"))?);
                pl.cross_attn_v_b =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_v.bias"))?);
                pl.cross_attn_o_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_out.weight"))?);
                pl.cross_attn_o_b =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_out.bias"))?);
                pl.cross_attn_norm_w =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_norm.weight"))?);
                pl.cross_attn_norm_b =
                    Some(loader.get_req(&format!("a.proj_blk.{il}.cross_attn_norm.bias"))?);
                pl.ff_up_w = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_up.weight"))?);
                pl.ff_up_b = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_up.bias"))?);
                pl.ff_down_w = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_down.weight"))?);
                pl.ff_down_b = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_down.bias"))?);
                pl.ln_2_w = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_norm.weight"))?);
                pl.ln_2_b = Some(loader.get_req(&format!("a.proj_blk.{il}.ffn_norm.bias"))?);
                qf.layers.push(pl);
            }
            model.qf = Some(qf);
        }
        ProjectorType::Gemma4A => {
            // clip.cpp:3273-3313
            for i in 0..2usize {
                model.sscp_conv_w[i] = Some(loader.get_req(&format!("a.conv1d.{i}.weight"))?);
                model.sscp_conv_b[i] = loader.get_opt(&format!("a.conv1d.{i}.bias"))?;
                model.sscp_norm_w[i] = loader.get_opt(&format!("a.conv1d.{i}.norm.weight"))?;
            }
            model.sscp_inp_proj_w = Some(loader.get_req("a.input_projection.weight")?);
            model.sscp_inp_proj_b = loader.get_opt("a.input_projection.bias")?;
            model.audio_out_proj_w = loader.get_opt("a.pre_encode.out.weight")?;
            model.audio_out_proj_b = loader.get_opt("a.pre_encode.out.bias")?;
            model.mm_soft_emb_norm_w = loader.get_opt("mm.a.soft_emb_norm.weight")?;
            model.mm_input_proj_w = loader.get_opt("mm.a.input_projection.weight")?;
            for il in 0..n_layer {
                let layer = &mut model.layers[il as usize];
                layer.ff_norm_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.weight"))?);
                layer.attn_pre_norm_w =
                    loader.get_opt(&format!("a.blk.{il}.attn_pre_norm.weight"))?;
                layer.per_dim_scale_w =
                    loader.get_opt(&format!("a.blk.{il}.per_dim_scale.weight"))?;
                layer.per_dim_k_scale_w =
                    loader.get_opt(&format!("a.blk.{il}.per_dim_k_scale.weight"))?;
                layer.attn_k_rel_w = loader.get_opt(&format!("a.blk.{il}.attn_k_rel.weight"))?;
                // conv_norm / norm_conv are swapped in GGUF (upstream
                // tensor_mapping.py) — loaded in reverse order (clip.cpp:3290)
                layer.norm_conv_w = loader.get_opt(&format!("a.blk.{il}.conv_norm.weight"))?;
                layer.norm_conv_b = loader.get_opt(&format!("a.blk.{il}.conv_norm.bias"))?;
                layer.conv_pw1_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.weight"))?);
                layer.conv_pw1_b = loader.get_opt(&format!("a.blk.{il}.conv_pw1.bias"))?;
                layer.conv_dw_w = Some(loader.get_req(&format!("a.blk.{il}.conv_dw.weight"))?);
                layer.conv_dw_b = loader.get_opt(&format!("a.blk.{il}.conv_dw.bias"))?;
                layer.conv_norm_w = loader.get_opt(&format!("a.blk.{il}.norm_conv.weight"))?;
                layer.conv_norm_b = loader.get_opt(&format!("a.blk.{il}.norm_conv.bias"))?;
                layer.conv_pw2_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.weight"))?);
                layer.conv_pw2_b = loader.get_opt(&format!("a.blk.{il}.conv_pw2.bias"))?;
                layer.ff_norm_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.weight"))?);
                layer.ff_up_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.weight"))?);
                layer.ff_up_1_b = loader.get_opt(&format!("a.blk.{il}.ffn_up_1.bias"))?;
                layer.ff_down_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.weight"))?);
                layer.ff_down_1_b = loader.get_opt(&format!("a.blk.{il}.ffn_down_1.bias"))?;
                layer.ff_post_norm_1_w =
                    loader.get_opt(&format!("a.blk.{il}.ffn_post_norm_1.weight"))?;
            }
            // clamp info AFTER all tensors are loaded (clip.cpp:3315-3333):
            // every loaded <name>.weight gets the four optional scalars
            let names: Vec<String> = loader
                .loaded
                .iter()
                .filter(|n| n.ends_with(".weight"))
                .cloned()
                .collect();
            for name in names {
                let scal = |n: &str, d: f32| -> Result<f32, String> {
                    let v = loader.get_vector_f32(n)?;
                    if v.is_empty() {
                        return Ok(d);
                    }
                    if v.len() != 1 {
                        return Err(format!(
                            "expected scalar tensor '{n}' but got {} elements",
                            v.len()
                        ));
                    }
                    Ok(v[0])
                };
                let base = name.trim_end_matches(".weight");
                let inp_max = scal(&format!("{base}.input_max"), f32::MAX)?;
                let inp_min = scal(&format!("{base}.input_min"), -f32::MAX)?;
                let out_max = scal(&format!("{base}.output_max"), f32::MAX)?;
                let out_min = scal(&format!("{base}.output_min"), -f32::MAX)?;
                model.clamp_info_map.insert(
                    name.clone(),
                    ClampInfo {
                        inp_min,
                        inp_max,
                        out_min,
                        out_max,
                    },
                );
            }
        }
        ProjectorType::Parakeet => {
            // clip.cpp:3393-3434 — mel filters/window read as raw vectors
            model.hparams.mel_filters = loader.get_vector_f32("a.mel_filters")?;
            model.hparams.window = loader.get_vector_f32("a.window")?;
            for i in [0usize, 2, 3, 5, 6] {
                model.pre_conv_w[i] = Some(loader.get_req(&format!("a.conv1d.{i}.weight"))?);
                model.pre_conv_b[i] = Some(loader.get_req(&format!("a.conv1d.{i}.bias"))?);
            }
            model.pre_encode_out_w = Some(loader.get_req("a.pre_encode.out.weight")?);
            model.pre_encode_out_b = Some(loader.get_req("a.pre_encode.out.bias")?);
            model.mm_norm_pre_w = loader.get_opt(&tnames::mm_norm_pre("weight"))?;
            model.mm_0_w = loader.get_opt(&tnames::mm_audio_mlp(1, "weight"))?;
            model.mm_1_w = loader.get_opt(&tnames::mm_audio_mlp(2, "weight"))?;
            for il in 0..n_layer {
                let layer = &mut model.layers[il as usize];
                layer.linear_pos_w =
                    Some(loader.get_req(&format!("a.blk.{il}.linear_pos.weight"))?);
                layer.pos_bias_u = Some(loader.get_req(&format!("a.blk.{il}.pos_bias_u"))?);
                layer.pos_bias_v = Some(loader.get_req(&format!("a.blk.{il}.pos_bias_v"))?);
                layer.conv_pw1_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw1.weight"))?);
                layer.conv_pw1_b = loader.get_opt(&format!("a.blk.{il}.conv_pw1.bias"))?;
                layer.conv_dw_w = Some(loader.get_req(&format!("a.blk.{il}.conv_dw.weight"))?);
                layer.conv_dw_b = loader.get_opt(&format!("a.blk.{il}.conv_dw.bias"))?;
                layer.conv_norm_w = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.weight"))?);
                layer.conv_norm_b = Some(loader.get_req(&format!("a.blk.{il}.conv_norm.bias"))?);
                layer.conv_norm_mean = Some(loader.get_req(&format!("a.blk.{il}.conv_norm_mean"))?);
                layer.conv_norm_var = Some(loader.get_req(&format!("a.blk.{il}.conv_norm_var"))?);
                layer.conv_pw2_w = Some(loader.get_req(&format!("a.blk.{il}.conv_pw2.weight"))?);
                layer.conv_pw2_b = loader.get_opt(&format!("a.blk.{il}.conv_pw2.bias"))?;
                layer.ff_norm_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.weight"))?);
                layer.ff_norm_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm.bias"))?);
                layer.ff_norm_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.weight"))?);
                layer.ff_norm_1_b = Some(loader.get_req(&format!("a.blk.{il}.ffn_norm_1.bias"))?);
                layer.ff_up_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_up_1.weight"))?);
                layer.ff_up_1_b = loader.get_opt(&format!("a.blk.{il}.ffn_up_1.bias"))?;
                layer.ff_down_1_w = Some(loader.get_req(&format!("a.blk.{il}.ffn_down_1.weight"))?);
                layer.ff_down_1_b = loader.get_opt(&format!("a.blk.{il}.ffn_down_1.bias"))?;
                layer.norm_conv_w = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.weight"))?);
                layer.norm_conv_b = Some(loader.get_req(&format!("a.blk.{il}.norm_conv.bias"))?);
            }
            // sound projection head (TN_MVLM_PROJ_MLP): mm.model.mlp.{0,1,3}
            model.mm_2_w = Some(loader.get_req("mm.model.mlp.0.weight")?);
            model.mm_3_w = Some(loader.get_req("mm.model.mlp.1.weight")?);
            model.conv_out_w = Some(loader.get_req("mm.model.mlp.3.weight")?);
        }
        ProjectorType::MimoAudio => {
            // clip.cpp:2875-2933
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.1.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.1.bias")?);
            model.conv1d_2_w = Some(loader.get_req("a.conv1d.2.weight")?);
            model.conv1d_2_b = Some(loader.get_req("a.conv1d.2.bias")?);
            model.downsample_conv_w = Some(loader.get_req("a.downsample.conv.weight")?);
            model.downsample_norm_w = Some(loader.get_req("a.downsample.norm.weight")?);
            model.downsample_norm_b = Some(loader.get_req("a.downsample.norm.bias")?);
            model.rvq_codebook = Some(loader.get_req("a.rvq.codebook.weight")?);
            model.mm_a_code_embd = Some(loader.get_req("mm.a.code_embd.weight")?);
            {
                // bound the GGUF-side bin counts by the tensors' shapes
                // (clip.cpp:2889-2897)
                let max_bins = loader.ctx.ne(model.rvq_codebook.unwrap())[1];
                let max_embd = loader.ctx.ne(model.mm_a_code_embd.unwrap())[1];
                for bins in &model.hparams.rvq_codebook_size {
                    if *bins <= 0 || *bins as i64 > max_bins || *bins as i64 > max_embd {
                        return Err(format!(
                            "mimo_audio: clip.audio.rvq.codebook_size entry ({bins}) out of range for codebook/code_embd tensors"
                        ));
                    }
                }
            }
            let n_local = model.hparams.audio_local_n_layer;
            for il in 0..n_local {
                let mut layer = empty_layer();
                layer.q_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_q.weight"))?);
                layer.q_b = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_q.bias"))?);
                layer.k_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_k.weight"))?);
                layer.k_b = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_k.bias"))?);
                layer.v_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_v.weight"))?);
                layer.v_b = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_v.bias"))?);
                layer.o_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.attn_out.weight"))?);
                layer.ff_gate_w =
                    Some(loader.get_req(&format!("mm.a.local_blk.{il}.ffn_gate.weight"))?);
                layer.ff_up_w =
                    Some(loader.get_req(&format!("mm.a.local_blk.{il}.ffn_up.weight"))?);
                layer.ff_down_w =
                    Some(loader.get_req(&format!("mm.a.local_blk.{il}.ffn_down.weight"))?);
                layer.ln_1_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.ln1.weight"))?);
                layer.ln_2_w = Some(loader.get_req(&format!("mm.a.local_blk.{il}.ln2.weight"))?);
                model.mm_a_local_layers.push(layer);
            }
            model.mm_a_local_norm_w = Some(loader.get_req("mm.a.local_norm.weight")?);
            model.mm_1_w = Some(loader.get_req(&tnames::mm_audio_mlp(1, "weight"))?);
            model.mm_2_w = Some(loader.get_req(&tnames::mm_audio_mlp(2, "weight"))?);
        }
        ProjectorType::Qwen3TtsSpkEnc => {
            // clip.cpp:2923-2962 — the layers are the SE-Res2Net blocks
            // (no standard-layer table, has_standard_layers == false)
            model.conv1d_1_w = Some(loader.get_req("a.conv1d.0.weight")?);
            model.conv1d_1_b = Some(loader.get_req("a.conv1d.0.bias")?);
            for il in 0..n_layer {
                let bid = il + 1;
                let mut layer = empty_layer();
                layer.conv_pw1_w = Some(loader.get_req(&format!("a.blk.{bid}.conv_pw1.weight"))?);
                layer.conv_pw1_b = Some(loader.get_req(&format!("a.blk.{bid}.conv_pw1.bias"))?);
                layer.conv_pw2_w = Some(loader.get_req(&format!("a.blk.{bid}.conv_pw2.weight"))?);
                layer.conv_pw2_b = Some(loader.get_req(&format!("a.blk.{bid}.conv_pw2.bias"))?);
                layer.se_conv1_w = Some(loader.get_req(&format!("a.blk.{bid}.se_conv1.weight"))?);
                layer.se_conv1_b = Some(loader.get_req(&format!("a.blk.{bid}.se_conv1.bias"))?);
                layer.se_conv2_w = Some(loader.get_req(&format!("a.blk.{bid}.se_conv2.weight"))?);
                layer.se_conv2_b = Some(loader.get_req(&format!("a.blk.{bid}.se_conv2.bias"))?);
                layer.res2_conv_w = vec![None; 7];
                layer.res2_conv_b = vec![None; 7];
                for xid in 0..7 {
                    layer.res2_conv_w[xid] =
                        Some(loader.get_req(&format!("a.blk.{bid}.res2.{xid}.weight"))?);
                    layer.res2_conv_b[xid] =
                        Some(loader.get_req(&format!("a.blk.{bid}.res2.{xid}.bias"))?);
                }
                model.layers.push(layer);
            }
            model.spk_conv_out_w = Some(loader.get_req("a.conv_out.weight")?);
            model.spk_conv_out_b = Some(loader.get_req("a.conv_out.bias")?);
            model.spk_asp_attn_w = Some(loader.get_req("a.asp_attn.weight")?);
            model.spk_asp_attn_b = Some(loader.get_req("a.asp_attn.bias")?);
            model.spk_asp_tdnn_w = Some(loader.get_req("a.asp_tdnn.weight")?);
            model.spk_asp_tdnn_b = Some(loader.get_req("a.asp_tdnn.bias")?);
            model.mm_fc_w = Some(loader.get_req(&tnames::mm_audio_fc("weight"))?);
            model.mm_fc_b = Some(loader.get_req(&tnames::mm_audio_fc("bias"))?);
        }
        ProjectorType::PocketTtsSpkEnc => {
            // clip.cpp:2965-2968 + load_seanet (clip.cpp:2166-2188, encoder)
            let mut seanet = Box::new(Seanet::default());
            seanet.conv_in_w = Some(loader.get_req("a.seanet.conv_in.weight")?);
            seanet.conv_in_b = Some(loader.get_req("a.seanet.conv_in.bias")?);
            seanet.conv_out_w = Some(loader.get_req("a.seanet.conv_out.weight")?);
            seanet.conv_out_b = Some(loader.get_req("a.seanet.conv_out.bias")?);
            for i in 0..model.hparams.seanet_n_stage {
                let mut st = SeanetStage::default();
                st.res_conv1_w =
                    Some(loader.get_req(&format!("a.seanet.blk.{i}.res_conv1.weight"))?);
                st.res_conv1_b = Some(loader.get_req(&format!("a.seanet.blk.{i}.res_conv1.bias"))?);
                st.res_conv2_w =
                    Some(loader.get_req(&format!("a.seanet.blk.{i}.res_conv2.weight"))?);
                st.res_conv2_b = Some(loader.get_req(&format!("a.seanet.blk.{i}.res_conv2.bias"))?);
                st.scale_conv_w =
                    Some(loader.get_req(&format!("a.seanet.blk.{i}.scale_conv.weight"))?);
                st.scale_conv_b =
                    Some(loader.get_req(&format!("a.seanet.blk.{i}.scale_conv.bias"))?);
                seanet.stages.push(st);
            }
            model.seanet = Some(seanet);
            model.pocket_downsample_w = Some(loader.get_req("a.downsample.conv.weight")?);
            model.spk_proj_w = Some(loader.get_req("a.speaker_proj.weight")?);
        }
        ProjectorType::PocketTtsGen => {
            // clip.cpp:2969-3046 + load_seanet(is_decoder = true)
            let mut flow = Box::new(FlowNet::default());
            flow.input_proj_w = Some(loader.get_req("a.gen.flow.input_proj.weight")?);
            flow.input_proj_b = Some(loader.get_req("a.gen.flow.input_proj.bias")?);
            flow.cond_embd_w = Some(loader.get_req("a.gen.flow.cond_embd.weight")?);
            flow.cond_embd_b = Some(loader.get_req("a.gen.flow.cond_embd.bias")?);
            flow.final_ada_w = Some(loader.get_req("a.gen.flow.final.ada.weight")?);
            flow.final_ada_b = Some(loader.get_req("a.gen.flow.final.ada.bias")?);
            flow.final_proj_w = Some(loader.get_req("a.gen.flow.final.proj.weight")?);
            flow.final_proj_b = Some(loader.get_req("a.gen.flow.final.proj.bias")?);
            // clip.cpp:2982-2985 — exactly two timestep embedders
            flow.time.reserve(2);
            for i in 0..2 {
                flow.time.push(FlowTimeEmbd {
                    freqs: Some(loader.get_req(&format!("a.gen.flow.time.{i}.freqs"))?),
                    up_w: Some(loader.get_req(&format!("a.gen.flow.time.{i}.up.weight"))?),
                    up_b: Some(loader.get_req(&format!("a.gen.flow.time.{i}.up.bias"))?),
                    down_w: Some(loader.get_req(&format!("a.gen.flow.time.{i}.down.weight"))?),
                    down_b: Some(loader.get_req(&format!("a.gen.flow.time.{i}.down.bias"))?),
                    norm: Some(loader.get_req(&format!("a.gen.flow.time.{i}.norm"))?),
                });
            }
            // clip.cpp:2992-3009 — one AdaLN block per flow depth, the count
            // is only known from the tensors (probe until absent)
            loop {
                let probe = loader.get_opt(&format!("a.gen.flow.blk.{}.norm.weight", flow.blocks.len()))?;
                let Some(norm_w) = probe else { break };
                let il = flow.blocks.len();
                flow.blocks.push(FlowBlock {
                    norm_w: Some(norm_w),
                    norm_b: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.norm.bias"))?),
                    up_w: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.up.weight"))?),
                    up_b: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.up.bias"))?),
                    down_w: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.down.weight"))?),
                    down_b: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.down.bias"))?),
                    ada_w: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.ada.weight"))?),
                    ada_b: Some(loader.get_req(&format!("a.gen.flow.blk.{il}.ada.bias"))?),
                });
            }
            model.flow = Some(flow);

            model.gen_out_eos_w = Some(loader.get_req("a.gen.out_eos.weight")?);
            model.gen_out_eos_b = Some(loader.get_req("a.gen.out_eos.bias")?);
            model.gen_input_lin_w = Some(loader.get_req("a.gen.input_linear.weight")?);
            model.gen_emb_mean = Some(loader.get_req("a.gen.emb_mean")?);
            model.gen_emb_std = Some(loader.get_req("a.gen.emb_std")?);

            // mimi decoder
            model.gen_quant_out_w = Some(loader.get_req("a.gen.wav.quant_out.weight")?);
            model.gen_upsample_w = Some(loader.get_req("a.gen.wav.upsample.weight")?);
            // load_seanet(is_decoder = true) — clip.cpp:2166-2188
            let mut seanet = Box::new(Seanet::default());
            seanet.conv_in_w = Some(loader.get_req("a.gen.wav.seanet.conv_in.weight")?);
            seanet.conv_in_b = Some(loader.get_req("a.gen.wav.seanet.conv_in.bias")?);
            seanet.conv_out_w = Some(loader.get_req("a.gen.wav.seanet.conv_out.weight")?);
            seanet.conv_out_b = Some(loader.get_req("a.gen.wav.seanet.conv_out.bias")?);
            for i in 0..model.hparams.seanet_n_stage {
                let mut st = SeanetStage::default();
                st.res_conv1_w =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.weight"))?);
                st.res_conv1_b =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.bias"))?);
                st.res_conv2_w =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.weight"))?);
                st.res_conv2_b =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.bias"))?);
                st.scale_conv_w =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.weight"))?);
                st.scale_conv_b =
                    Some(loader.get_req(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.bias"))?);
                seanet.stages.push(st);
            }
            model.seanet = Some(seanet);
            // clip.cpp:3021-3041 — the mimi decoder transformer
            for il in 0..n_layer {
                let mut layer = empty_layer();
                layer.ln_1_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"))?);
                layer.ln_1_b = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln1.bias"))?);
                layer.q_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"))?);
                layer.k_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"))?);
                layer.v_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"))?);
                layer.o_w =
                    Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"))?);
                layer.ls_1_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"))?);
                layer.ln_2_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"))?);
                layer.ln_2_b = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln2.bias"))?);
                layer.ff_up_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"))?);
                layer.ff_down_w =
                    Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"))?);
                layer.ls_2_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"))?);
                model.gen_tfm_layers.push(layer);
            }
        }
        ProjectorType::Qwen3TtsGen => {
            // clip.cpp:3041-3134 — code_predictor (proj_in optional) + code2wav
            model.gen_code_proj_in_w = loader.get_opt("a.gen.code.proj_in.weight")?;
            model.gen_code_proj_in_b = loader.get_opt("a.gen.code.proj_in.bias")?;
            model.gen_code_embd_w = Some(loader.get_req("a.gen.code.embd.weight")?);
            model.gen_code_head_w = Some(loader.get_req("a.gen.code.head.weight")?);
            model.gen_code_out_embd_w = Some(loader.get_req("a.gen.code.out_embd.weight")?);
            model.gen_code_norm_w = Some(loader.get_req("a.gen.code.output_norm.weight")?);

            let mut c2w = Box::new(Code2Wav::default());
            c2w.quant_first_in_w = Some(loader.get_req("a.gen.wav.quant.first.in_proj.weight")?);
            c2w.quant_first_out_w = Some(loader.get_req("a.gen.wav.quant.first.out_proj.weight")?);
            c2w.quant_first_cb_w = Some(loader.get_req("a.gen.wav.quant.first.codebook.weight")?);
            c2w.quant_rest_in_w = Some(loader.get_req("a.gen.wav.quant.rest.in_proj.weight")?);
            c2w.quant_rest_out_w = Some(loader.get_req("a.gen.wav.quant.rest.out_proj.weight")?);
            c2w.quant_rest_cb_w = Some(loader.get_req("a.gen.wav.quant.rest.codebook.weight")?);
            c2w.pre_conv_w = Some(loader.get_req("a.gen.wav.pre_conv.weight")?);
            c2w.pre_conv_b = Some(loader.get_req("a.gen.wav.pre_conv.bias")?);
            c2w.tfm_in_proj_w = Some(loader.get_req("a.gen.wav.tfm.in_proj.weight")?);
            c2w.tfm_in_proj_b = Some(loader.get_req("a.gen.wav.tfm.in_proj.bias")?);
            c2w.tfm_out_proj_w = Some(loader.get_req("a.gen.wav.tfm.out_proj.weight")?);
            c2w.tfm_out_proj_b = Some(loader.get_req("a.gen.wav.tfm.out_proj.bias")?);
            c2w.tfm_output_norm_w = Some(loader.get_req("a.gen.wav.tfm.output_norm.weight")?);
            // loaded manually, the generic model.layers loop is taken by
            // code_predictor (clip.cpp:3069)
            for il in 0..model.hparams.wav_tfm_n_layer {
                let mut layer = empty_layer();
                layer.q_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"))?);
                layer.k_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"))?);
                layer.v_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"))?);
                layer.o_w =
                    Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"))?);
                layer.ln_1_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"))?);
                layer.ln_2_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"))?);
                layer.ls_1_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"))?);
                layer.ls_2_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"))?);
                layer.ff_gate_w =
                    Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ffn_gate.weight"))?);
                layer.ff_up_w = Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"))?);
                layer.ff_down_w =
                    Some(loader.get_req(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"))?);
                c2w.tfm_layers.push(layer);
            }
            // upsample: 2x (causal ConvTranspose1d + ConvNeXt block)
            for il in 0..model.hparams.wav_upsample_n_block {
                c2w.upsample.push(C2wUpsample {
                    conv_w: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.conv.weight"))?),
                    conv_b: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.conv.bias"))?),
                    dwconv_w: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.dwconv.weight"))?),
                    dwconv_b: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.dwconv.bias"))?),
                    norm_w: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.norm.weight"))?),
                    norm_b: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.norm.bias"))?),
                    pw1_w: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.pw1.weight"))?),
                    pw1_b: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.pw1.bias"))?),
                    pw2_w: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.pw2.weight"))?),
                    pw2_b: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.pw2.bias"))?),
                    gamma: Some(loader.get_req(&format!("a.gen.wav.up.blk.{il}.gamma"))?),
                });
            }
            // DAC decoder
            c2w.dac_entry_w = Some(loader.get_req("a.gen.wav.dac.entry.weight")?);
            c2w.dac_entry_b = Some(loader.get_req("a.gen.wav.dac.entry.bias")?);
            for il in 0..model.hparams.wav_dac_n_block {
                let mut blk = C2wDac {
                    snake_alpha: Some(loader.get_req(&format!("a.gen.wav.dac.blk.{il}.snake.alpha"))?),
                    snake_beta: Some(loader.get_req(&format!("a.gen.wav.dac.blk.{il}.snake.beta"))?),
                    conv_w: Some(loader.get_req(&format!("a.gen.wav.dac.blk.{il}.conv.weight"))?),
                    conv_b: Some(loader.get_req(&format!("a.gen.wav.dac.blk.{il}.conv.bias"))?),
                    res: Vec::new(),
                };
                for ir in 0..model.hparams.wav_dac_n_res {
                    blk.res.push(C2wDacRes {
                        act1_alpha: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.alpha"))?,
                        ),
                        act1_beta: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.beta"))?,
                        ),
                        conv1_w: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.weight"))?,
                        ),
                        conv1_b: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.bias"))?,
                        ),
                        act2_alpha: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.alpha"))?,
                        ),
                        act2_beta: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.beta"))?,
                        ),
                        conv2_w: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.weight"))?,
                        ),
                        conv2_b: Some(
                            loader.get_req(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.bias"))?,
                        ),
                    });
                }
                c2w.dac.push(blk);
            }
            c2w.dac_post_snake_alpha = Some(loader.get_req("a.gen.wav.dac.post_snake.alpha")?);
            c2w.dac_post_snake_beta = Some(loader.get_req("a.gen.wav.dac.post_snake.beta")?);
            c2w.dac_post_conv_w = Some(loader.get_req("a.gen.wav.dac.post_conv.weight")?);
            c2w.dac_post_conv_b = Some(loader.get_req("a.gen.wav.dac.post_conv.bias")?);
            model.c2w = Some(c2w);
        }
        ProjectorType::Unknown => unreachable!(),
    }

    Ok(())
}

/// clip.cpp:3968 `clip_init`
pub fn clip_init_from_file(fname: &str, params: &ClipContextParams) -> Result<ClipContext, String> {
    let gguf = Gguf::open(fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    // the loader mmaps the file body (ggml_backend_cpu_buffer_from_ptr in C)
    let file = std::fs::File::open(fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    let mmap =
        Arc::new(unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap failed: {e}"))?);

    let has_vision = gguf.get_bool("clip.has_vision_encoder").unwrap_or(false);
    let has_audio = gguf.get_bool("clip.has_audio_encoder").unwrap_or(false);
    let has_gen_audio = gguf.get_bool(keys::HAS_GEN_AUDIO_ENC).unwrap_or(false);
    // clip.cpp:1240-1255: a single-modality file selects its modality; this
    // port runs one modality per context (vision or audio or gen-audio)
    let modality = if has_gen_audio && !has_vision && !has_audio {
        ClipModality::GenAudio
    } else if has_audio && !has_vision {
        ClipModality::Audio
    } else if has_vision {
        ClipModality::Vision
    } else {
        return Err("mmproj has no vision or audio encoder".to_string());
    };
    clip_init_modality(fname, params, modality)
}

/// the forced-modality entry a MIXED mmproj needs: the reference's clip_init
/// builds one clip_ctx per modality in the file (clip.cpp:3980-4009 — ctx_v /
/// ctx_a / ctx_gen_a), and `mtmd_init_from_file` then holds the audio
/// (speaker-encoder) ctx as its primary while the GEN calls route to
/// ctx_gen_a (mtmd.cpp:584-585 / :1922+). The port's MtmdContext calls this
/// for the gen half of an audio+gen file.
pub fn clip_init_modality(
    fname: &str,
    params: &ClipContextParams,
    modality: ClipModality,
) -> Result<ClipContext, String> {
    let gguf = Gguf::open(fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    let file = std::fs::File::open(fname).map_err(|e| format!("failed to open {fname}: {e}"))?;
    let mmap =
        Arc::new(unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap failed: {e}"))?);
    let is_audio = modality == ClipModality::Audio;
    let has_vision = modality == ClipModality::Vision;

    let (hparams, proj_type) = load_hparams(&gguf, params, modality)?;

    let mut loader = Loader::new(&gguf, mmap.clone());
    let mut model = ClipModel {
        hparams,
        proj_type,
        has_vision,
        modality,
        conv1d_1_w: None,
        conv1d_1_b: None,
        conv1d_2_w: None,
        conv1d_2_b: None,
        mm_fc_w: None,
        mm_fc_b: None,
        mm_norm_pre_w: None,
        mm_norm_pre_b: None,
        mm_norm_mid_w: None,
        mm_2_w: None,
        mm_2_b: None,
        mm_3_w: None,
        mm_3_b: None,
        mm_4_w: None,
        mm_4_b: None,
        mm_5_w: None,
        mm_5_b: None,
        mm_6_w: None,
        mm_6_b: None,
        mm_boi: None,
        mm_eoi: None,
        conv2d_w: [None, None, None],
        conv2d_b: [None, None, None],
        conv_out_w: None,
        mm_input_proj_w: None,
        pre_conv_w: [None; 7],
        pre_conv_b: [None; 7],
        pre_encode_out_w: None,
        pre_encode_out_b: None,
        inp_proj_w: None,
        inp_proj_b: None,
        ctc_out_w: None,
        ctc_out_b: None,
        ctc_out_mid_w: None,
        ctc_out_mid_b: None,
        qf: None,
        sscp_conv_w: [None, None],
        sscp_conv_b: [None, None],
        sscp_norm_w: [None, None],
        sscp_inp_proj_w: None,
        sscp_inp_proj_b: None,
        audio_out_proj_w: None,
        audio_out_proj_b: None,
        mm_soft_emb_norm_w: None,
        clamp_info_map: Default::default(),
        downsample_conv_w: None,
        downsample_norm_w: None,
        downsample_norm_b: None,
        rvq_codebook: None,
        mm_a_code_embd: None,
        mm_a_local_layers: Vec::new(),
        mm_a_local_norm_w: None,
        spk_conv_out_w: None,
        spk_conv_out_b: None,
        spk_asp_attn_w: None,
        spk_asp_attn_b: None,
        spk_asp_tdnn_w: None,
        spk_asp_tdnn_b: None,
        seanet: None,
        pocket_downsample_w: None,
        spk_proj_w: None,
        flow: None,
        c2w: None,
        gen_tfm_layers: Vec::new(),
        gen_code_proj_in_w: None,
        gen_code_proj_in_b: None,
        gen_code_embd_w: None,
        gen_code_head_w: None,
        gen_code_out_embd_w: None,
        gen_code_norm_w: None,
        gen_out_eos_w: None,
        gen_out_eos_b: None,
        gen_input_lin_w: None,
        gen_emb_mean: None,
        gen_emb_std: None,
        gen_quant_out_w: None,
        gen_upsample_w: None,
        gen_model_variant: String::new(),
        class_embedding: None,
        pre_ln_w: None,
        pre_ln_b: None,
        post_ln_w: None,
        post_ln_b: None,
        patch_bias: None,
        patch_embeddings_0: None,
        patch_embeddings_1: None,
        position_embeddings: None,
        mm_0_w: None,
            mm_input_norm_w: None,
            mm_input_norm_b: None,
        mm_0_b: None,
        mm_1_w: None,
        mm_1_b: None,
        layers: Vec::new(),
    };
    load_tensors(&mut loader, &mut model)?;

    if is_audio {
        // requirements of the whisper-enc graph (models/whisper-enc.cpp:6)
        // and qwen3a (models/qwen3a.cpp:64): learned position embeddings
        if matches!(
            proj_type,
            ProjectorType::Qwen2A
                | ProjectorType::Ultravox
                | ProjectorType::Voxtral
                | ProjectorType::Meralion
                | ProjectorType::Glma
                | ProjectorType::MusicFlamingo
                | ProjectorType::Qwen3A
                | ProjectorType::Lfm2A
        ) && model.position_embeddings.is_none()
        {
            return Err(format!(
                "{} requires a.position_embd.weight",
                proj_type.name()
            ));
        }
        log_line(&format!("modality:          audio (experimental)"));
        log_line(&format!("n_mel_bins:        {}", model.hparams.n_mel_bins));
        log_line(&format!(
            "proj_stack_factor: {}",
            model.hparams.proj_stack_factor
        ));
        log_line(&format!(
            "audio_sample_rate: {}",
            model.hparams.audio_sample_rate
        ));
        log_line(&format!("audio_n_fft:       {}", model.hparams.audio_n_fft));
    } else if modality == ClipModality::GenAudio {
        log_line("modality:          gen-audio (experimental)");
    } else if proj_type == ProjectorType::Cohere2V {
        // requirements of the siglip graph (models/siglip.cpp:5-11): one
        // patch conv and the square learned position grid
        if model.patch_embeddings_0.is_none() || model.position_embeddings.is_none() {
            return Err(format!(
                "{} requires v.patch_embd.weight and v.position_embd.weight",
                proj_type.name()
            ));
        }
    } else {
        // requirements of the one graph this port builds (models/qwen3vl.cpp:4-6)
        if model.patch_embeddings_0.is_none()
            || model.patch_embeddings_1.is_none()
            || model.position_embeddings.is_none()
        {
            return Err(format!(
                "{} requires v.patch_embd.weight, v.patch_embd.weight.1 and v.position_embd.weight",
                proj_type.name()
            ));
        }
    }

    let hp = &model.hparams;
    log_line(&format!("projector:        {}", proj_type.name()));
    log_line(&format!("n_embd:           {}", hp.n_embd));
    log_line(&format!(
        "n_head:           {} (kv {})",
        hp.n_head, hp.n_head_kv
    ));
    log_line(&format!("n_ff:             {}", hp.n_ff));
    log_line(&format!("n_layer:          {}", hp.n_layer));
    log_line(&format!("projection_dim:   {}", hp.projection_dim));
    log_line(&format!("image_size:       {}", hp.image_size));
    log_line(&format!("patch_size:       {}", hp.patch_size));
    log_line(&format!("n_merge:          {}", hp.n_merge));

    let ctx = loader.ctx;
    let watermark = ctx.mark();
    Ok(ClipContext {
        model,
        ctx,
        watermark,
        // clip.cpp:3711 — AUTO means "try FA": the CPU backend supports
        // flash_attn_ext, so it stays enabled (see clip_model_loader::warmup)
        flash_attn_type: match params.flash_attn_type {
            ClipFlashAttn::Auto => ClipFlashAttn::Enabled,
            v => v,
        },
        n_threads: params.n_threads,
        out_embd: Vec::new(),
        rng: Mt19937::seeded_from_entropy(),
        rng_seed: u32::MAX,
        debug_output_embeddings: std::env::var("MTMD_DEBUG_EMBEDDINGS").is_ok(),
        _mmap: mmap,
    })
}

impl ClipContext {
    /// clip.cpp:6069 `clip_get_hparams`
    pub fn hparams(&self) -> &ClipHparams {
        &self.model.hparams
    }

    /// `clip_get_image_max_tokens` (clip.cpp:4064-4074, upstream a7b94df2c):
    /// max number of output tokens per image, -1 if not dynamic size
    pub fn get_image_max_tokens(&self) -> i32 {
        let hparams = &self.model.hparams;
        // deepseek4v's `dsv4_max_n_token` arm (clip.cpp:4066-4068) has no
        // ported projector instance yet — the family is in the documented
        // ~50-projector gap
        if hparams.image_max_pixels <= 0 {
            return -1;
        }
        let patch_area = hparams.patch_size
            * hparams.patch_size
            * hparams.n_merge
            * hparams.n_merge;
        hparams.image_max_pixels / patch_area
    }

    /// clip.cpp:643 `clip_n_mmproj_embd`
    pub fn n_mmproj_embd(&self) -> i32 {
        // audio projectors report the projector output width
        // (clip.cpp:5976-5991): mm_fc_w/mm_2_w/mm_3_w -> ne[1]
        match self.model.proj_type {
            ProjectorType::Qwen2A => {
                return self.ctx.ne(self.model.mm_fc_w.unwrap())[1] as i32;
            }
            ProjectorType::Ultravox | ProjectorType::Voxtral | ProjectorType::MusicFlamingo => {
                return self.ctx.ne(self.model.mm_2_w.unwrap())[1] as i32;
            }
            ProjectorType::Meralion => {
                return self.ctx.ne(self.model.mm_3_w.unwrap())[1] as i32;
            }
            ProjectorType::Glma => {
                return self.ctx.ne(self.model.mm_2_w.unwrap())[1] as i32;
            }
            // clip.cpp:6000-6001 — the merger output width
            ProjectorType::Ling3Vl => {
                return self.ctx.ne(self.model.mm_1_b.unwrap())[0] as i32;
            }
            // clip.cpp:5989
            ProjectorType::Qwen3A => {
                return self.ctx.ne(self.model.mm_2_w.unwrap())[1] as i32;
            }
            // clip.cpp:6113-6114 — cohere2v's mm.2 output width
            ProjectorType::Cohere2V => {
                return self.ctx.ne(self.model.mm_2_w.unwrap())[1] as i32;
            }
            // clip.cpp:5972 (gemma4v/uv/a/ua share the arm)
            ProjectorType::Gemma4UA | ProjectorType::Gemma4A => {
                return self.ctx.ne(self.model.mm_input_proj_w.unwrap())[1] as i32;
            }
            // clip.cpp:6005 — the conformer's output width IS the learned
            // position-embedding width (d_model)
            ProjectorType::Lfm2A => {
                return self.ctx.ne(self.model.position_embeddings.unwrap())[0] as i32;
            }
            // clip.cpp:6124-6125 — d1omni_a reports the out_proj width
            // (position embeddings are optional for it, a657f7e98)
            ProjectorType::D1OmniA => {
                return self.ctx.ne(self.model.mm_3_w.unwrap())[1] as i32;
            }
            // clip.cpp:6008
            ProjectorType::GraniteSpeech => {
                return self
                    .ctx
                    .ne(self.model.qf.as_ref().unwrap().qf_proj_linear_w.unwrap())[1]
                    as i32;
            }
            // clip.cpp:6024
            ProjectorType::Parakeet => {
                return self.ctx.ne(self.model.mm_1_w.unwrap())[1] as i32;
            }
            // clip.cpp:6015
            ProjectorType::MimoAudio => {
                return self.ctx.ne(self.model.mm_2_w.unwrap())[1] as i32;
            }
            // clip.cpp:6020 — a conv weight [K, IC, OC]: the OUTPUT channel
            // count is ne[2]
            ProjectorType::Qwen3TtsSpkEnc => {
                return self.ctx.ne(self.model.mm_fc_w.unwrap())[2] as i32;
            }
            // clip.cpp:6027
            ProjectorType::PocketTtsSpkEnc => {
                return self.ctx.ne(self.model.spk_proj_w.unwrap())[1] as i32;
            }
            // clip.cpp:6019 — the gen side reports gen_code_out_embd_w->ne[0]
            ProjectorType::Qwen3TtsGen => {
                return self.ctx.ne(self.model.gen_code_out_embd_w.unwrap())[0] as i32;
            }
            // clip.cpp:6023
            ProjectorType::PocketTtsGen => {
                return self.ctx.ne(self.model.gen_input_lin_w.unwrap())[1] as i32;
            }
            _ => {}
        }
        self.model.hparams.projection_dim
    }

    /// clip.h:76 `clip_n_output_tokens` — the number of text tokens an image
    /// expands into (clip.cpp:4094).
    pub fn n_output_tokens(&self, img: &ClipImageF32) -> i32 {
        let hp = &self.model.hparams;
        if self.model.proj_type == ProjectorType::Glma {
            // clip.cpp:4268-4276: glma halves once, divides by the merge
            // factor, then adds the BOI/EOI embedding rows
            let mut n_patches = img.nx;
            // whisper downscales input token by half after conv1d
            n_patches /= 2;
            // reshape by merge_factor
            n_patches /= hp.proj_stack_factor;
            // for BOI and EOI token embeddings
            n_patches += 2;
            return n_patches;
        }
        if self.model.proj_type.is_whisper_family() {
            // clip.cpp:4238-4258 (the whisper family): frames in, then the
            // conv1d pair halves the token count, then nn.AvgPool1d(2, stride=2)
            // halves it again for qwen2a/voxtral/musicflamingo
            // (clip-model.h:881-885 audio_has_avgpool)
            let mut n_patches = img.nx;
            if self.model.proj_type.audio_has_stack_frames() {
                assert!(hp.proj_stack_factor > 0, "stack_factor must be set");
                let n_len = clip_align(n_patches, hp.proj_stack_factor);
                n_patches = n_len / hp.proj_stack_factor;
            }
            // whisper downscales input token by half after conv1d
            n_patches /= 2;
            if self.model.proj_type.audio_has_avgpool() {
                // divide by 2 because of nn.AvgPool1d(2, stride=2)
                n_patches /= 2;
            }
            return n_patches;
        }
        let patch = hp.patch_size;
        let mut n_patches = (img.nx / patch) * (img.ny / patch);
        if self.model.proj_type.merger_2x2() {
            // dynamic size (2 conv, so double patch size)
            let x_patch = img.nx / (patch * 2);
            let y_patch = img.ny / (patch * 2);
            n_patches = x_patch * y_patch;
        }
        // clip.cpp:4281-4287 (gemma3/idefics3/cohere2v/...): both X and Y are
        // downscaled by the pixel-shuffle scale factor
        if self.model.proj_type == ProjectorType::Cohere2V {
            let scale_factor = hp.n_merge;
            n_patches /= scale_factor * scale_factor;
        }
        // clip.cpp:4261-4263 (qwen3a): chunk_size=100 frames -> 3x stride-2
        // conv2d -> 13 tokens per chunk
        if self.model.proj_type == ProjectorType::Qwen3A {
            const CHUNK_SIZE: i32 = 100;
            const TOKENS_PER_CHUNK: i32 = 13;
            n_patches = (img.nx / CHUNK_SIZE) * TOKENS_PER_CHUNK;
        }
        // clip.cpp:4335-4337 (gemma4ua): no downsampling, one token per raw
        // waveform frame
        if self.model.proj_type == ProjectorType::Gemma4UA {
            n_patches = img.nx;
        }
        // clip.cpp:4322-4324 (lfm2a): three stride-2 convolutions
        // (clip.cpp:4424: d1omni_a shares the arm)
        if matches!(
            self.model.proj_type,
            ProjectorType::Lfm2A | ProjectorType::D1OmniA
        ) {
            n_patches = ((((img.nx + 1) / 2) + 1) / 2 + 1) / 2;
        }
        // clip.cpp:4321-4329 (gemma4a): two stride-2 conv2d, p=1, k=3
        // O = floor((I - 1) / 2) + 1
        if self.model.proj_type == ProjectorType::Gemma4A {
            let mut n = img.nx;
            for _ in 0..2 {
                n = (n - 1) / 2 + 1;
            }
            n_patches = n;
        }
        // clip.cpp:4331-4333 (parakeet): the fixed conv-stride subsampling
        if self.model.proj_type == ProjectorType::Parakeet {
            let sf = self.model.hparams.subsampling_factor;
            n_patches = (img.nx + sf - 1) / sf;
        }
        // clip.cpp:4339-4346 (mimo_audio): conv1(s=1)+conv2(s=2) then the
        // RVQ downsample conv (k=2, s=2), grouped by local_group_size
        if self.model.proj_type == ProjectorType::MimoAudio {
            let group_size = self.model.hparams.audio_local_group_size;
            let mut n = img.nx;
            n = (n - 1) / 2 + 1;
            n = (n - 2) / 2 + 1;
            n_patches = (n + group_size - 1) / group_size;
        }
        // clip.cpp:4348-4352 (granite_speech): window count x queries/window
        if self.model.proj_type == ProjectorType::GraniteSpeech {
            let ws = self.model.hparams.audio_proj_window_size;
            let ds = self.model.hparams.audio_proj_downsample_rate;
            n_patches = ((img.nx + ws - 1) / ws) * (ws / ds);
        }
        // clip.cpp:4354-4357 (qwen3tts_spkenc): pooling gives one embedding
        if self.model.proj_type == ProjectorType::Qwen3TtsSpkEnc {
            n_patches = 1;
        }
        // clip.cpp:4364-4368 (pockettts_spkenc): one row per 12.5 Hz frame
        if self.model.proj_type == ProjectorType::PocketTtsSpkEnc {
            let hop = self.model.hparams.mimi_downsample * 120;
            n_patches = img.nx / hop;
        }
        n_patches
    }

    /// clip.cpp:4044 `clip_n_output_tokens_x` (the M-RoPE x count)
    pub fn n_output_tokens_x(&self, img: &ClipImageF32) -> i32 {
        let hp = &self.model.hparams;
        if self.model.proj_type.merger_2x2() {
            return (img.nx / hp.patch_size) / 2;
        }
        self.n_output_tokens(img)
    }

    /// clip.cpp:4071 `clip_n_output_tokens_y`
    pub fn n_output_tokens_y(&self, img: &ClipImageF32) -> i32 {
        let hp = &self.model.hparams;
        if self.model.proj_type.merger_2x2() {
            return (img.ny / hp.patch_size) / 2;
        }
        1
    }

    /// clip.cpp:4409 `clip_image_batch_encode` for the audio modality: builds
    /// the whisper-enc graph (models/whisper-enc.cpp:3-137) over one mel chunk
    /// (`imgs.entries[0]`: nx = frames, ny = mel bins) and returns the projected
    /// embeddings as a flat `[n_mmproj_embd * n_tokens]` buffer.
    ///
    /// Every op here is a bit-exact one (mul_mat, add, gelu_erf, norm,
    /// soft_max_ext/flash_attn_ext, pool/stack) — see the parity record in
    /// PARITY.md. The two constructions that stand in for ops the port's ggml
    /// does not expose are `pool_1d` and `swiglu_swapped` below.
    pub fn audio_batch_encode(&mut self, imgs: &ClipImageF32Batch) -> Result<Vec<f32>, String> {
        if imgs.entries.len() != 1 {
            // clip_encode: "audio input ... GGML_ASSERT(imgs.entries.size() == 1)"
            return Err(format!(
                "audio batch size {} is not supported (expected 1)",
                imgs.entries.len()
            ));
        }
        let img = &imgs.entries[0];
        if img.is_placeholder() {
            return Err("audio tokens batch is placeholder".into());
        }

        self.out_embd.clear();
        let expected_tokens = self.n_output_tokens(img);
        let expected = expected_tokens * self.n_audio_embd();
        let ClipContext {
            ctx,
            model,
            watermark,
            flash_attn_type,
            n_threads,
            out_embd,
            ..
        } = self;

        ctx.reset_graph_to(*watermark);
        let (mut graph, nodes) = build_audio_graph(ctx, model, img, *flash_attn_type)?;

        // clip_encode (audio arm, clip.cpp:4551-4559): the mel buffer goes in
        // verbatim — [n_frames, n_mel] row-major over the frame dim
        let n = img.buf.len();
        debug_assert_eq!(
            n,
            (img.nx * img.ny) as usize,
            "mel buffer must be n_len*n_mel"
        );
        ctx.arena_resize_tensor(nodes.inp_raw);
        ctx.with_f32_mut(nodes.inp_raw, |p| p.copy_from_slice(&img.buf))
            .expect("inp_raw is an arena tensor");

        // named inputs (clip.cpp set_inputs) — conformer's pos_emb etc.
        for (name, t, data) in &nodes.extra {
            ctx.arena_resize_tensor(*t);
            match data {
                ExtraInput::F32(v) => {
                    ctx.with_f32_mut(*t, |p| p.copy_from_slice(v))
                        .expect(&format!("{name} is an arena tensor"));
                }
                ExtraInput::I32(v) => {
                    ctx.with_i32_mut(*t, |p| p.copy_from_slice(v))
                        .expect(&format!("{name} is an arena tensor"));
                }
            }
        }

        let compute = |ctx: &mut Context, graph: &mut Graph| {
            ggml::compute::graph_compute(ctx, graph, *n_threads)
        };
        if let Ok(path) = std::env::var("MTMD_DEBUG_NODES_BIN") {
            debug_dump_nodes_bin_run(&path, || compute(ctx, &mut graph));
        } else {
            compute(ctx, &mut graph);
        }

        if let Ok(path) = std::env::var("MTMD_DEBUG_NODES") {
            let _ = debug_dump_nodes(ctx, &graph, &path);
        }

        // clip.cpp:5790 — the last graph node is the embedding tensor
        let out = nodes.out;
        let ne = *ctx.ne(out);
        let n_elems = ne.iter().product::<i64>() as usize;
        let out_ty = ctx.ty(out);
        if out_ty != GgmlType::F32 {
            return Err(format!("expected F32 embeddings, got {out_ty:?}"));
        }
        // clip.cpp:5794-5799 — token count sanity against clip_n_output_tokens
        if ne[1] as i32 != expected_tokens {
            return Err(format!(
                "expected output {expected_tokens} tokens, got {}",
                ne[1]
            ));
        }
        if n_elems as i32 != expected {
            return Err(format!(
                "expected output {expected} elements, got {n_elems}"
            ));
        }
        out_embd.resize(n_elems, 0.0);
        out_embd.copy_from_slice(ctx.f32s(out).expect("arena tensor"));
        let debug = self.debug_output_embeddings;
        let result = std::mem::take(out_embd);

        if debug {
            dump_debug_embeddings(ctx, out);
        }
        Ok(result)
    }

    /// `clip_n_audio_embd` — the audio projector output width
    /// (clip_n_mmproj_embd for the audio modality: mm_fc_w/mm_2_w/mm_3_w
    /// -> ne[1], clip.cpp:5976-5991)
    pub fn n_audio_embd(&self) -> i32 {
        self.n_mmproj_embd()
    }

    /// the audio preprocessing hparams (see [`ClipHparams::audio_hparams`])
    pub fn audio_hparams(&self) -> crate::mtmd_audio::AudioHparams {
        self.model.hparams.audio_hparams()
    }

    /// clip.cpp:4409 `clip_image_batch_encode` → clip.cpp:4427 `clip_encode`.
    ///
    /// Builds the vision graph, runs it and returns the projected embeddings as
    /// a flat `[n_mmproj_embd * n_tokens]` buffer (the C `out_batch_embd`).
    pub fn image_batch_encode(&mut self, imgs: &ClipImageF32Batch) -> Result<Vec<f32>, String> {
        let n_batch = imgs.entries.len();
        if n_batch != 1 {
            return Err(format!(
                "batch size {n_batch} is not supported (no temporal merge)"
            ));
        }
        if imgs.entries[0].is_placeholder() {
            return Err("cannot encode a placeholder image".into());
        }

        // the siglip tower (cohere2v) takes its own graph: separate-qkv ViT +
        // pixel shuffle + swapped-swiglu projector (clip.cpp:940)
        if self.model.proj_type == ProjectorType::Cohere2V {
            return self.image_batch_encode_siglip(imgs);
        }

        self.out_embd.clear();
        let expected = self.n_output_tokens(&imgs.entries[0]) * self.n_mmproj_embd();
        let ClipContext {
            ctx,
            model,
            watermark,
            flash_attn_type,
            n_threads,
            out_embd,
            ..
        } = self;

        ctx.reset_graph_to(*watermark);
        let (mut graph, nodes) = build_graph(ctx, model, &imgs.entries[0], *flash_attn_type)?;

        // set inputs (clip.cpp:4538-4579): the raw image, channel-major
        let img = &imgs.entries[0];
        let (nx, ny) = (img.nx as usize, img.ny as usize);
        let n = nx * ny;
        let mut inp_raw = vec![0.0f32; 3 * n];
        for y in 0..ny {
            for x in 0..nx {
                let src = 3 * (y * nx + x);
                let dst = y * nx + x;
                inp_raw[dst] = img.buf[src];
                inp_raw[n + dst] = img.buf[src + 1];
                inp_raw[2 * n + dst] = img.buf[src + 2];
            }
        }
        ctx.arena_resize_tensor(nodes.inp_raw);
        ctx.with_f32_mut(nodes.inp_raw, |p| p.copy_from_slice(&inp_raw))
            .expect("inp_raw is an arena tensor");

        // M-RoPE positions (clip.cpp:4796, shared by qwen2-vl / qwen3-vl)
        ctx.arena_resize_tensor(nodes.positions);
        ctx.with_i32_mut(nodes.positions, |p| {
            p.copy_from_slice(&nodes.positions_data)
        })
        .expect("positions is an arena tensor");

        ggml::compute::graph_compute(ctx, &mut graph, *n_threads);

        if let Ok(path) = std::env::var("MTMD_DEBUG_NODES") {
            let _ = debug_dump_nodes(ctx, &graph, &path);
        }

        // clip.cpp:5790 — the last graph node is the embedding tensor
        let out = *graph.nodes.last().expect("non-empty graph");
        let n_elems = ctx.ne(out).iter().product::<i64>() as usize;
        let out_ty = ctx.ty(out);
        if out_ty != GgmlType::F32 {
            return Err(format!("expected F32 embeddings, got {out_ty:?}"));
        }
        if n_elems as i32 != expected {
            return Err(format!(
                "expected output {expected} elements, got {n_elems}"
            ));
        }
        out_embd.resize(n_elems, 0.0);
        out_embd.copy_from_slice(ctx.f32s(out).expect("arena tensor"));
        let debug = self.debug_output_embeddings;
        let result = std::mem::take(out_embd);

        if debug {
            dump_debug_embeddings(ctx, out);
        }
        Ok(result)
    }

    /// the cohere2v encode path: `clip_encode` over the siglip graph
    /// (clip.cpp:4538-4579 pixel upload, :5790 output copy). The tiles arrive
    /// one chunk at a time (mtmd.cpp:1478+), so this is always a single entry.
    fn image_batch_encode_siglip(&mut self, imgs: &ClipImageF32Batch) -> Result<Vec<f32>, String> {
        self.out_embd.clear();
        let expected =
            self.n_output_tokens(&imgs.entries[0]) * self.n_mmproj_embd();
        let ClipContext {
            ctx,
            model,
            watermark,
            flash_attn_type,
            n_threads,
            out_embd,
            ..
        } = self;

        ctx.reset_graph_to(*watermark);
        let (mut graph, nodes) = build_siglip_graph(ctx, model, &imgs.entries[0], *flash_attn_type)?;

        // set inputs (clip.cpp:4538-4579): the raw image, channel-major
        let img = &imgs.entries[0];
        let (nx, ny) = (img.nx as usize, img.ny as usize);
        let n = nx * ny;
        let mut inp_raw = vec![0.0f32; 3 * n];
        for y in 0..ny {
            for x in 0..nx {
                let src = 3 * (y * nx + x);
                let dst = y * nx + x;
                inp_raw[dst] = img.buf[src];
                inp_raw[n + dst] = img.buf[src + 1];
                inp_raw[2 * n + dst] = img.buf[src + 2];
            }
        }
        ctx.arena_resize_tensor(nodes.inp_raw);
        ctx.with_f32_mut(nodes.inp_raw, |p| p.copy_from_slice(&inp_raw))
            .expect("inp_raw is an arena tensor");

        ggml::compute::graph_compute(ctx, &mut graph, *n_threads);

        if let Ok(path) = std::env::var("MTMD_DEBUG_NODES") {
            let _ = debug_dump_nodes(ctx, &graph, &path);
        }

        // clip.cpp:5790 — the last graph node is the embedding tensor
        let out = *graph.nodes.last().expect("non-empty graph");
        let n_elems = ctx.ne(out).iter().product::<i64>() as usize;
        let out_ty = ctx.ty(out);
        if out_ty != GgmlType::F32 {
            return Err(format!("expected F32 embeddings, got {out_ty:?}"));
        }
        if n_elems as i32 != expected {
            return Err(format!(
                "expected output {expected} elements, got {n_elems}"
            ));
        }
        out_embd.resize(n_elems, 0.0);
        out_embd.copy_from_slice(ctx.f32s(out).expect("arena tensor"));
        let debug = self.debug_output_embeddings;
        let result = std::mem::take(out_embd);

        if debug {
            dump_debug_embeddings(ctx, out);
        }
        Ok(result)
    }
}

/// clip.cpp:5868 — `MTMD_DEBUG_EMBEDDINGS=<path>` dumps
/// `[int32 n_tokens][int32 n_embd][f32 data]`, the format the reference
/// `llama-mtmd-debug` / parity scripts read.
fn dump_debug_embeddings(cx: &Context, out: TensorId) {
    let ne = cx.ne(out);
    let (n_embd, n_tokens) = (ne[0] as i32, ne[1] as i32);
    let data = cx.f32s(out).expect("arena tensor");
    log_line("=== MTMD_DEBUG_EMBEDDINGS ===");
    log_line(&format!("Shape: [{n_embd}, {n_tokens}]"));
    if let Ok(path) = std::env::var("MTMD_DEBUG_EMBEDDINGS") {
        if path != "1" {
            let mut bytes = Vec::with_capacity(8 + data.len() * 4);
            bytes.extend_from_slice(&n_tokens.to_le_bytes());
            bytes.extend_from_slice(&n_embd.to_le_bytes());
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            match std::fs::write(&path, &bytes) {
                Ok(()) => log_line(&format!("wrote {path}")),
                Err(e) => log_line(&format!("failed to write {path}: {e}")),
            }
        }
    }
    log_line("=== END MTMD_DEBUG_EMBEDDINGS ===");
}

// ======================================================================
// vision graph — models/qwen3vl.cpp:3 `clip_graph_qwen3vl::build`
// ======================================================================

pub struct GraphNodes {
    pub inp_raw: TensorId,
    pub positions: TensorId,
    /// the M-RoPE position ids to upload into `positions`
    pub positions_data: Vec<i32>,
    pub out: TensorId,
}

/// The ViT graph of the qwen2/qwen3-VL family. Split out of `ClipContext` so
/// the borrow of `ctx` and `model` is disjoint (both live in the same struct).
fn build_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, GraphNodes), String> {
    let hp = &model.hparams;
    assert_eq!(img.nx % (hp.patch_size * 2), 0, "image width not aligned");
    assert_eq!(img.ny % (hp.patch_size * 2), 0, "image height not aligned");

    let (pw, ph) = (
        (img.nx / hp.patch_size) as i64,
        (img.ny / hp.patch_size) as i64,
    );
    let n_pos = pw * ph;
    let n_embd = hp.n_embd as i64;
    let n_layer = hp.n_layer as usize;
    // clip-graph.h:59 — for the merger family the value is n_merge^2
    let merge_factor: i64 = if hp.n_merge > 0 {
        (hp.n_merge as i64).pow(2)
    } else {
        4
    };
    assert_eq!(merge_factor, 4, "only the 2x2 merge is ported");

    // clip-graph.h:48 `d_head`: clip.vision.attention.head_dim wins, else n_embd/n_head
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    // models/qwen3vl.cpp:12 — qwen3-vl normalises with plain layer norm
    let norm_t = if model.proj_type == ProjectorType::Qwen25Vl {
        NormType::Rms
    } else {
        NormType::Normal
    };

    let mut g = Graph::new(4096);

    // build_inp_with_temporal_merge (models/qwen2vl.cpp:3): still-image input is
    // the sum of the two patch-embedding convs
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 3, 1);
    cx.set_name(inp_raw, "inp_raw");
    let pe0 = model.patch_embeddings_0.unwrap();
    let pe1 = model.patch_embeddings_1.unwrap();
    let conv0 = cx.conv_2d(pe0, inp_raw, hp.patch_size, hp.patch_size, 0, 0, 1, 1);
    let conv1 = cx.conv_2d(pe1, inp_raw, hp.patch_size, hp.patch_size, 0, 0, 1, 1);
    let mut inp = cx.add(conv0, conv1);

    // spatial merge (models/qwen3vl.cpp:18-31)
    {
        inp = cx.permute(inp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
        inp = cx.cont_4d(inp, n_embd * 2, pw / 2, ph, 1);
        inp = cx.reshape_4d(inp, n_embd * 2, pw / 2, 2, (ph / 2).max(1));
        inp = cx.permute(inp, 0, 2, 1, 3);
        inp = cx.cont_3d(inp, n_embd, n_pos, 1);
    }

    // patch bias (models/qwen3vl.cpp:33)
    if let Some(b) = model.patch_bias {
        inp = cx.add(inp, b);
    }

    // learned position embedding, resized to the patch grid (models/qwen3vl.cpp:40)
    match model.position_embeddings {
        Some(pe) => {
            let pos = resize_position_embeddings(cx, pe, n_embd, pw, ph, hp.patch_size as i64);
            let pos = cx.cont_4d(pos, n_embd * 2, pw / 2, ph, 1);
            let pos = cx.reshape_4d(pos, n_embd * 2, pw / 2, 2, (ph / 2).max(1));
            let pos = cx.permute(pos, 0, 2, 1, 3);
            let pos = cx.cont_3d(pos, n_embd, n_pos, 1);
            inp = cx.add(inp, pos);
        }
        None => return Err("no position embedding".into()),
    }

    let mut inp_l = inp;

    // M-RoPE positions, `n_pos * 4` i32 (models/qwen3vl.cpp:56). The ids are
    // the merge-tile-major grid the spatial merge produced (clip.cpp:4796).
    let positions = cx.new_tensor_1d(GgmlType::I32, n_pos * 4);
    cx.set_name(positions, "positions");
    let positions_data = qwen_vl_positions(pw, ph, hp.n_merge as i64);

    // pre-layernorm (models/qwen3vl.cpp:61)
    if let (Some(w), Some(b)) = (model.pre_ln_w, model.pre_ln_b) {
        inp_l = build_norm(cx, inp_l, Some(w), Some(b), norm_t, hp.eps);
    }

    for il in 0..n_layer {
        let layer = &model.layers[il];
        let mut cur = inp_l;

        cur = build_norm(cx, cur, layer.ln_1_w, layer.ln_1_b, norm_t, hp.eps);

        // self-attention (models/qwen3vl.cpp:80): fused qkv, then M-RoPE
        {
            let qkv_w = layer.qkv_w.unwrap();
            cur = cx.mul_mat(qkv_w, cur);
            if let Some(qkv_b) = layer.qkv_b {
                cur = cx.add(cur, qkv_b);
            }
            let qkv_ty = cx.ty(cur);
            // models/qwen3vl.cpp:84 — nb1 = ggml_row_size(type, d_head)
            let row = qkv_ty.row_size(d_head as usize);
            let qkv_nb1 = cx.nb(cur)[1] as usize;
            let q = cx.view_3d(cur, d_head, n_head, n_pos, row, qkv_nb1, 0);
            let k = cx.view_3d(
                cur,
                d_head,
                n_head,
                n_pos,
                row,
                qkv_nb1,
                qkv_ty.row_size(n_embd as usize),
            );
            let v = cx.view_3d(
                cur,
                d_head,
                n_head,
                n_pos,
                row,
                qkv_nb1,
                qkv_ty.row_size(2 * n_embd as usize),
            );

            // ggml_rope_multi (models/qwen3vl.cpp:104): VISION mode, four
            // sections of d_head/4, freq_base 10000, n_ctx_orig 32768
            let sections = [
                (d_head / 4) as i32,
                (d_head / 4) as i32,
                (d_head / 4) as i32,
                (d_head / 4) as i32,
            ];
            let q = cx.rope_multi(
                q,
                positions,
                None,
                (d_head / 2) as i32,
                sections,
                GGML_ROPE_TYPE_VISION,
                32768,
                10000.0,
                1.0,
                0.0,
                1.0,
                32.0,
                1.0,
            );
            let k = cx.rope_multi(
                k,
                positions,
                None,
                (d_head / 2) as i32,
                sections,
                GGML_ROPE_TYPE_VISION,
                32768,
                10000.0,
                1.0,
                0.0,
                1.0,
                32.0,
                1.0,
            );

            cur = build_attn(cx, layer, q, k, v, kq_scale, flash_attn);
        }

        // residual (models/qwen3vl.cpp:120)
        cur = cx.add(cur, inp_l);
        inp_l = cur;

        // ffn (models/qwen3vl.cpp:127-135)
        cur = build_norm(cx, cur, layer.ln_2_w, layer.ln_2_b, norm_t, hp.eps);
        cur = build_ffn(cx, cur, layer, hp.ffn_op);
        cur = cx.add(inp_l, cur);
        inp_l = cur;
    }

    // post-layernorm (models/qwen3vl.cpp:164)
    if let (Some(w), Some(b)) = (model.post_ln_w, model.post_ln_b) {
        inp_l = build_norm(cx, inp_l, Some(w), Some(b), norm_t, hp.eps);
    }

    // multimodal projection (models/qwen3vl.cpp:169-176): 2x2 merge is folded
    // into the reshape, then the two-layer GELU MLP
    let mut embeddings = if model.proj_type == ProjectorType::Ling3Vl {
        // ling3vl (models/ling3vl.cpp:66-77): norm-only merger — a LayerNorm
        // over each patch BEFORE the 2x2 fold, then the top-level MLP
        let e = build_norm(
            cx,
            inp_l,
            model.mm_input_norm_w,
            model.mm_input_norm_b,
            norm_t,
            hp.eps,
        );
        cx.reshape_3d(e, n_embd * 4, n_pos / 4, 1)
    } else {
        cx.reshape_3d(inp_l, n_embd * 4, n_pos / 4, 1)
    };
    embeddings = build_ffn_plain(
        cx,
        embeddings,
        model.mm_0_w.unwrap(),
        model.mm_0_b,
        model.mm_1_w.unwrap(),
        model.mm_1_b,
        FfnOp::Gelu,
    );

    // the C asserts the merged token count here (clip.cpp:5793)
    assert_eq!(
        cx.ne(embeddings)[1],
        n_pos / 4,
        "unexpected output token count"
    );

    g.build_forward(cx, embeddings);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        GraphNodes {
            inp_raw,
            positions,
            positions_data,
            out,
        },
    ))
}

// ======================================================================
// siglip vision tower — models/siglip.cpp:3 `clip_graph_siglip::build`
// (the cohere2v slice; gemma3/idefics3/lfm2/janus/phi4 tails stay in the
// documented projector gap — only the COHERE2V projector is loadable here)
// ======================================================================

/// input handle of the siglip graph — no M-RoPE positions, just the pixels
pub struct SiglipGraphNodes {
    pub inp_raw: TensorId,
    pub out: TensorId,
}

/// clip.cpp:572 `clip_graph::build_inp` — the conv2d patch embedding, brought
/// to `[n_embd, n_patches, n_batch]`. `n_batch` is 1 here: the port encodes
/// each llava-uhd tile as its own chunk (mtmd.cpp:1478 does the same), so the
/// batch dim never carries multiple tiles.
fn build_siglip_inp(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    n_patches: i64,
    n_embd: i64,
) -> (TensorId, TensorId) {
    let hp = &model.hparams;
    // clip.cpp:584 `build_inp_raw`: [nx, ny, 3, n_batch], a graph input
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 3, 1);
    cx.set_name(inp_raw, "inp_raw");
    // clip.cpp:574 — one conv, stride == patch, no pad
    let mut inp = cx.conv_2d(
        model.patch_embeddings_0.unwrap(),
        inp_raw,
        hp.patch_size,
        hp.patch_size,
        0,
        0,
        1,
        1,
    );
    // [n_patches, n_embd, n_batch] -> transpose -> [n_embd, n_patches, B]
    inp = cx.reshape_3d(inp, n_patches, n_embd, 1);
    let t = cx.transpose(inp);
    inp = cx.cont(t);
    if let Some(b) = model.patch_bias {
        inp = cx.add(inp, b);
    }
    (inp_raw, inp)
}

/// clip.cpp:901 `clip_graph::build_patch_merge_permute` — the pixel shuffle
/// that folds a `scale_factor`-by-`scale_factor` neighborhood of patches into
/// one merged token (aka pixel_unshuffle / patch merger).
fn build_patch_merge_permute(
    cx: &mut Context,
    cur: TensorId,
    scale_factor: i64,
    patch_size: i32,
    img_nx: i32,
    img_ny: i32,
) -> TensorId {
    assert!(scale_factor > 1);
    let n_embd = cx.ne(cur)[0];
    // clip.cpp:904-905 — the patch grid of the image being encoded
    let mut width = (img_nx / patch_size) as i64;
    let mut height = (img_ny / patch_size) as i64;

    // clip.cpp:908-917 — pad the grid up to a multiple of the scale factor
    let pad_width = clip_align(width as i32, scale_factor as i32) as i64 - width;
    let pad_height = clip_align(height as i32, scale_factor as i32) as i64 - height;
    let mut cur = cx.reshape_3d(cur, n_embd, width, height);
    if pad_width != 0 || pad_height != 0 {
        cur = cx.pad(cur, 0, pad_width as i32, pad_height as i32, 0);
        width += pad_width;
        height += pad_height;
    }

    // clip.cpp:919-923 — unshuffle h
    cur = cx.reshape_3d(cur, n_embd * scale_factor, width / scale_factor, height);
    cur = cx.permute(cur, 0, 2, 1, 3);

    // clip.cpp:925-927 — unshuffle w
    cur = cx.cont_3d(
        cur,
        n_embd * scale_factor * scale_factor,
        height / scale_factor,
        width / scale_factor,
    );
    cur = cx.permute(cur, 0, 2, 1, 3);

    // clip.cpp:929 — flatten the grid into the token dim
    cx.cont_2d(cur, cx.ne(cur)[0], cx.ne(cur)[1] * cx.ne(cur)[2])
}

/// models/siglip.cpp:3 `clip_graph_siglip::build` for `cohere2v`: the ViT
/// (`build_vit`, shared with the audio family as `build_vit_audio_opts`)
/// followed by the idefics3-style square pixel shuffle and the two-layer
/// swapped-swiglu projector.
fn build_siglip_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, SiglipGraphNodes), String> {
    let hp = &model.hparams;
    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let n_head_kv = hp.n_head_kv as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();
    let n_patches =
        ((img.nx / hp.patch_size) as i64) * ((img.ny / hp.patch_size) as i64);

    let mut g = Graph::new(4096);

    let (inp_raw, inp) = build_siglip_inp(cx, model, img, n_patches, n_embd);

    // siglip.cpp:6-10 — cohere2v does NOT resize the position grid: the tiles
    // are exactly image_size x image_size, so the learned square grid fits
    let learned_pos_embd = model.position_embeddings;

    // siglip.cpp:11-16 `build_vit`: plain LayerNorm, the model's FFN op
    // (use_gelu -> GELU for cohere2), learned positions, no add_pos hook
    let cur = build_vit_audio_opts(
        cx,
        model,
        inp,
        n_patches,
        NormType::Normal,
        hp.ffn_op,
        learned_pos_embd,
        n_embd,
        n_head,
        n_head_kv,
        d_head,
        kq_scale,
        flash_attn,
        &BuildVitAudioOpts::default(),
    )
    .0;

    // siglip.cpp:48-56 (COHERE2V) — tiles are square, so the pixel shuffle is
    // the same as Idefics3, then linear_1 / swiglu(gate)*x / linear_2
    let cur = build_patch_merge_permute(
        cx,
        cur,
        hp.n_merge as i64,
        hp.patch_size,
        img.nx,
        img.ny,
    );
    let cur = cx.mul_mat(model.mm_1_w.unwrap(), cur);
    let cur = cx.add(cur, model.mm_1_b.unwrap());
    // linear_1 output is [x, gate], HF computes silu(gate) * x
    let cur = swiglu_swapped(cx, cur);
    let cur = cx.mul_mat(model.mm_2_w.unwrap(), cur);
    let cur = cx.add(cur, model.mm_2_b.unwrap());

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((g, SiglipGraphNodes { inp_raw, out }))
}

/// clip.cpp:312 `clip_graph::resize_position_embeddings` — bilinear resize of
/// the learned 2D grid to the image's patch grid.
fn resize_position_embeddings(    cx: &mut Context,
    pos_embd: TensorId,
    n_embd: i64,
    width: i64,
    height: i64,
    _patch_size: i64,
) -> TensorId {
    const MODE: u32 = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS; // clip-graph.h:11 DEFAULT_INTERPOLATION_MODE
    let ne = *cx.ne(pos_embd);
    let n_per_side = (ne[1] as f64).sqrt() as i64;
    assert!(
        n_per_side * n_per_side == ne[1],
        "position embedding is not a square grid"
    );
    if width == n_per_side && height == n_per_side {
        return pos_embd;
    }
    let p = cx.reshape_3d(pos_embd, n_embd, n_per_side, n_per_side);
    let p = cx.permute(p, 2, 0, 1, 3); // -> (n_per_side, n_per_side, n_embd)
    let p = cx.interpolate(p, width, height, n_embd, 1, MODE);
    let p = cx.permute(p, 1, 2, 0, 3); // -> (n_embd, width, height)
    cx.cont_2d(p, n_embd, width * height)
}

/// clip.cpp:591 `clip_graph::build_norm`
fn build_norm(
    cx: &mut Context,
    cur: TensorId,
    mw: Option<TensorId>,
    mb: Option<TensorId>,
    ty: NormType,
    eps: f32,
) -> TensorId {
    let mut cur = match ty {
        NormType::Rms => cx.rms_norm(cur, eps),
        NormType::Normal => cx.norm(cur, eps),
    };
    if let Some(w) = mw {
        cur = cx.mul(cur, w);
    }
    if let Some(b) = mb {
        cur = cx.add(cur, b);
    }
    cur
}

/// clip.cpp:616 `clip_graph::build_ffn` — the `gate == NULL` (parallel-FFN) and
/// gated variants, restricted to the ops the merger family uses.
fn build_ffn(cx: &mut Context, cur: TensorId, layer: &ClipLayer, ffn_t: FfnOp) -> TensorId {
    let up = layer.ff_up_w.unwrap();
    let gate = layer.ff_gate_w;
    build_ffn_impl(
        cx,
        cur,
        up,
        layer.ff_up_b,
        gate,
        layer.ff_gate_b,
        layer.ff_down_w,
        layer.ff_down_b,
        ffn_t,
    )
}

/// clip.cpp:616 with a (possibly absent) gate; shared by the block FFN and the
/// merger MLP (`mm.0` → gelu → `mm.2`).
fn build_ffn_impl(
    cx: &mut Context,
    cur: TensorId,
    up: TensorId,
    up_b: Option<TensorId>,
    gate: Option<TensorId>,
    gate_b: Option<TensorId>,
    down: Option<TensorId>,
    down_b: Option<TensorId>,
    ffn_t: FfnOp,
) -> TensorId {
    let mut tmp = cx.mul_mat(up, cur);
    if let Some(b) = up_b {
        tmp = cx.add(tmp, b);
    }

    let mut cur = if let Some(g) = gate {
        let mut c = cx.mul_mat(g, cur);
        if let Some(b) = gate_b {
            c = cx.add(c, b);
        }
        c
    } else {
        tmp
    };

    if gate.is_some() {
        // the gated FFN form (clip.cpp:641-670): the activation applies to
        // the GATE half and multiplies the UP half through the fused split
        // GLU ops — `ggml_swiglu_split(cur, tmp)` for SILU,
        // `ggml_geglu_split` for GELU, `ggml_geglu_erf_split` for GELU_ERF
        cur = match ffn_t {
            FfnOp::Silu => cx.swiglu_split(cur, tmp),
            FfnOp::Gelu => cx.geglu_split(cur, tmp),
            other => unimplemented!("gated ffn op {other:?} not ported"),
        };
    } else {
        match ffn_t {
            FfnOp::Gelu => {
                cur = cx.gelu(cur);
            }
            // the whisper-enc family (clip.cpp:1787 forces FFN_GELU_ERF): the
            // parallel FFN form `ggml_gelu_erf` (clip.cpp:655-658)
            FfnOp::GeluErf => {
                cur = cx.gelu_erf(cur);
            }
            // the conformer family's half-step FFNs (conformer.cpp:78/:195)
            FfnOp::Silu => {
                cur = cx.silu(cur);
            }
            // FFN_RELU_SQR (clip.cpp:672-676): relu then square — the parakeet
            // sound projection head (parakeet.cpp:415)
            FfnOp::ReluSqr => {
                cur = cx.relu(cur);
                cur = cx.sqr(cur);
            }
            // the remaining activations exist in ggml but are not reachable from
            // the ported projector types
            other => unimplemented!("ffn op {other:?} not ported"),
        }
    }

    if let Some(d) = down {
        cur = cx.mul_mat(d, cur);
    }
    if let Some(b) = down_b {
        cur = cx.add(cur, b);
    }
    cur
}

/// `build_ffn(cur, up, up_b, NULL, NULL, down, down_b, ...)` — the merger MLP.
fn build_ffn_plain(
    cx: &mut Context,
    cur: TensorId,
    up: TensorId,
    up_b: Option<TensorId>,
    down: TensorId,
    down_b: Option<TensorId>,
    ffn_t: FfnOp,
) -> TensorId {
    build_ffn_impl(cx, cur, up, up_b, None, None, Some(down), down_b, ffn_t)
}

/// clip.cpp:748 `clip_graph::build_attn`. Both branches are ported: the FA path
/// (k/v cast to F16, `flash_attn_ext` — what the reference CPU build runs for
/// AUTO/enabled) and the reference non-FA path (soft_max_ext over a mul_mat).
fn build_attn(
    cx: &mut Context,
    layer: &ClipLayer,
    q_cur: TensorId,
    k_cur: TensorId,
    v_cur: TensorId,
    kq_scale: f32,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    build_attn_ex(
        cx, layer.o_w, layer.o_b, q_cur, k_cur, v_cur, None, kq_scale, flash_attn,
    )
}

/// clip.cpp:748 with an explicit output projection and kq_mask — the callers
/// that own their attention (mimo's input_local_transformer with a block-
/// diagonal mask, pocket-tts's mimi transformer, granite's QFormer with
/// `wo_b` possibly NULL).
fn build_attn_ex(
    cx: &mut Context,
    o_w: Option<TensorId>,
    o_b: Option<TensorId>,
    q_cur: TensorId,
    k_cur: TensorId,
    v_cur: TensorId,
    kq_mask: Option<TensorId>,
    kq_scale: f32,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    let q = cx.permute(q_cur, 0, 2, 1, 3);
    let k = cx.permute(k_cur, 0, 2, 1, 3);

    let mut cur = if flash_attn == ClipFlashAttn::Enabled {
        let v = cx.permute(v_cur, 0, 2, 1, 3);
        // clip.cpp:776 — the CPU FA kernel only consumes F16 KV
        let k = cx.cast(k, GgmlType::F16);
        let v = cx.cast(v, GgmlType::F16);
        // clip.cpp:780 — the mask goes through the same F16 cast
        let kq_mask = kq_mask.map(|m| cx.cast(m, GgmlType::F16));
        // ggml_prec_set_acc(.., F32): the CPU dispatch treats prec F32
        // identically to the default (ops.cpp:9348)
        let cur = cx.flash_attn_ext(q, k, v, kq_mask, kq_scale, 0.0, 0.0);
        cx.reshape_2d(
            cur,
            cx.ne(cur)[0] * cx.ne(cur)[1],
            cx.ne(cur)[2] * cx.ne(cur)[3],
        )
    } else {
        let v = cx.permute(v_cur, 1, 2, 0, 3);
        let v = cx.cont(v);
        let kq = cx.mul_mat(k, q);
        let kq = cx.soft_max_ext(kq, kq_mask, kq_scale, 0.0);
        let kqv = cx.mul_mat(v, kq);
        let cur = cx.permute(kqv, 0, 2, 1, 3);
        cx.cont_2d(
            cur,
            cx.ne(cur)[0] * cx.ne(cur)[1],
            cx.ne(cur)[2] * cx.ne(cur)[3],
        )
    };
    let _ = k;

    // clip.cpp:810 — output projection
    if let Some(w) = o_w {
        cur = cx.mul_mat(w, cur);
    }
    if let Some(b) = o_b {
        cur = cx.add(cur, b);
    }
    cur
}

/// clip.cpp:4796 — the M-RoPE position ids for the qwen-VL merger family.
/// The token order produced by the spatial merge is merge-tile raster order,
/// with (dy, dx) inside each 2x2 tile (see `build_graph`).
fn qwen_vl_positions(pw: i64, ph: i64, merge_ratio: i64) -> Vec<i32> {
    let n_pos = (pw * ph) as usize;
    let mut positions = vec![0i32; n_pos * 4];
    let mut ptr = 0usize;
    let mut y = 0;
    while y < ph {
        let mut x = 0;
        while x < pw {
            for dy in 0..2 {
                for dx in 0..2 {
                    positions[ptr] = (y + dy) as i32;
                    positions[n_pos + ptr] = (x + dx) as i32;
                    positions[2 * n_pos + ptr] = (y + dy) as i32;
                    positions[3 * n_pos + ptr] = (x + dx) as i32;
                    ptr += 1;
                }
            }
            x += merge_ratio;
        }
        y += merge_ratio;
    }
    assert_eq!(ptr, n_pos);
    positions
}

// ======================================================================
// audio graph — models/whisper-enc.cpp:3 `clip_graph_whisper_enc::build`
// (+ the build_vit path it shares with the vision models, clip.cpp:337-570)
// ======================================================================

/// clip-impl.h:427 `CLIP_ALIGN(x, n)`
fn clip_align(x: i32, n: i32) -> i32 {
    (x + n - 1) / n * n
}

/// ggml.c:4616 `ggml_conv_1d` + :4637 `ggml_conv_1d_ph` (padding = kernel/2).
/// `a` = kernel [K, IC, OC], `b` = signal [W, IC, N] → result [OL, OC, N].
fn conv_1d_ph(cx: &mut Context, a: TensorId, b: TensorId, s0: i32, d0: i32) -> TensorId {
    let p0 = (cx.ne(a)[0] / 2) as i32; // ggml.c:4643
    let a_ty = cx.ty(a);
    // ggml.c:4619 — an F32 kernel reads the patches as F16
    let dst_type = if a_ty == GgmlType::Bf16 {
        GgmlType::F32
    } else {
        GgmlType::F16
    };
    let im2col = cx.im2col(a, b, s0, 0, p0, 0, d0, 0, false, dst_type);
    let (in0, in1, in2) = {
        let t = cx.ne(im2col);
        (t[0], t[1], t[2])
    };
    let (ane0, ane1, ane2) = {
        let t = cx.ne(a);
        (t[0], t[1], t[2])
    };
    // ggml.c:4625-4627 — [ICK, N*OL] x [ICK, OC]
    let lhs = cx.reshape_2d(im2col, in0, in2 * in1);
    let rhs = cx.reshape_2d(a, ane0 * ane1, ane2);
    let result = cx.mul_mat(lhs, rhs);
    // ggml.c:4629 — [OL, OC, N]
    cx.reshape_3d(result, in1, ane2, in2)
}

/// `ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0)` (whisper-enc.cpp:549,
/// the nn.AvgPool1d(2, stride=2) of clip.cpp:546-556) — now the literal op
/// (ggml.c:5071, kernel ops.cpp:7690-7754).
///
/// History: before the port's ggml grew GGML_OP_POOL_1D this was expressed as
/// `pool_2d(AVG, k0=2, k1=1, s0=2, s1=1, p0=0, p1=0)`, proven bit-exact for
/// this (k,s,p): pool_1d divides by the in-range `count`, pool_2d by
/// `ka = k0*k1` (ops.cpp:7729 vs :7802); for k=2/s=2/p=0 every window is fully
/// in range so count == ka == 2 and both kernels accumulate the same two
/// source values in the same order. The six-projector parity run was repeated
/// after the switch: identical bits (see PARITY.md, the POOL_1D section).
fn pool_1d_avg2(cx: &mut Context, cur: TensorId) -> TensorId {
    cx.pool_1d(cur, Context::GGML_OP_POOL_AVG, 2, 2, 0)
}

/// `ggml_swiglu_swapped` (ggml.c:3055 → ggml_glu_impl with src1 == NULL and
/// swapped = 1, whisper-enc.cpp:66): silu(**second** half) * first half.
///
/// The port's ggml only exposes the split GLU constructor (`swiglu_split`,
/// ops.rs), whose kernel (compute.rs forward_glu, C ops.cpp:3178) asserts
/// src1 != NULL. Splitting the halves with views and swapping the argument
/// order produces the identical element-wise kernel input: C's swapped GLU
/// runs `ggml_vec_swiglu_f32(nc, dst, src0_p + nc, src0_p)` (ops.cpp:3224
/// with the :3221-3223 offsets), which is `silu(x) * g` over the same value
/// pairs this builds — bit-exact, just a different addressing pattern.
fn swiglu_swapped(cx: &mut Context, cur: TensorId) -> TensorId {
    let ne = *cx.ne(cur);
    let nc = ne[0] / 2;
    let row = cx.nb(cur)[1] as usize;
    // two half-width column views of every row
    let first = cx.view_2d(cur, nc, ne[1], row, 0);
    let second = cx.view_2d(cur, nc, ne[1], row, nc as usize * cx.ty(cur).type_size());
    // swiglu_split(a, b) = silu(a) * b; swapped GLU silus the second half
    cx.swiglu_split(second, first)
}

/// clip.cpp:872 `clip_graph::build_stack` — StackAudioFrames (ultravox).
fn build_stack(cx: &mut Context, cur: TensorId, stack_factor: i32, n_embd: i32) -> TensorId {
    if stack_factor <= 1 {
        return cur;
    }
    let total_elements: i64 = cx.ne(cur).iter().product();
    let stride = (n_embd * stack_factor) as i64;

    // GGML_PAD(x, n) (ggml.h:267)
    let padded_len = (total_elements + stride - 1) & !(stride - 1);
    let pad = padded_len - total_elements;

    let mut cur = cur;
    if pad > 0 {
        // pad the tensor to make it divisible by stride
        let v = cx.view_1d(cur, total_elements, 0);
        cur = cx.pad(v, pad as i32, 0, 0, 0);
    }
    // reshape to [stride, padded_len / stride]
    let ty = cx.ty(cur);
    cx.view_2d(
        cur,
        stride,
        padded_len / stride,
        ty.row_size(stride as usize),
        0,
    )
}

/// clip.cpp:337 `clip_graph::build_vit` — the exact path the whisper family
/// exercises: learned position embedding (:341), pre-LN (:349), the separate
/// q/k/v branch (:412-437 — whisper layers have no fused qkv and no q/k norm),
/// `build_attn` (:748), residual (:490), ln_2 + FFN (:494-527), residual 2
/// (:556), AvgPool1d (:546, qwen2a/voxtral/musicflamingo), post-LN (:559) and
/// the batch-dim restore (:565). The branches this family never takes
/// (fused qkv, q/k norm, add_pos, layer scale, MoE FFN, attn masks) are not
/// built.
/// clip-graph.h:14 `build_vit_opts` — the knobs mimo_audio needs (per-layer
/// attention masks, skipping the automatic post-LN, layer-output capture).
/// Everything defaults to the plain path the whisper family takes.
#[derive(Default)]
pub struct BuildVitAudioOpts {
    /// one mask per layer (`attn_mask_layers`); a None entry falls back to
    /// `attn_mask`
    pub attn_mask_layers: Vec<Option<TensorId>>,
    pub attn_mask: Option<TensorId>,
    /// capture the residual output of layer `il` (mimo's skip layer 2)
    pub capture_layer: Option<i32>,
    /// build_vit must not apply the post-LN itself (mimo adds the skip first)
    pub skip_post_ln: bool,
    /// `add_pos` hook: mimo applies RoPE to Q/K here
    pub add_pos: bool,
    /// the positions for the add_pos rope (mimo_audio_positions)
    pub inp_pos: Option<TensorId>,
    /// the encoder's rope theta (mimo: 10000)
    pub rope_theta: f32,
}

#[allow(clippy::too_many_arguments)]
fn build_vit_audio(
    cx: &mut Context,
    model: &ClipModel,
    mut inp: TensorId,
    n_pos: i64,
    norm_t: NormType,
    ffn_t: FfnOp,
    learned_pos_embd: Option<TensorId>,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    d_head: i64,
    kq_scale: f32,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    build_vit_audio_opts(
        cx,
        model,
        inp,
        n_pos,
        norm_t,
        ffn_t,
        learned_pos_embd,
        n_embd,
        n_head,
        n_head_kv,
        d_head,
        kq_scale,
        flash_attn,
        &BuildVitAudioOpts::default(),
    )
    .0
}

/// the full `clip_graph::build_vit` (clip.cpp:337-566) — the opts-typed core
/// behind [`build_vit_audio`]; the plain callers pass default opts.
#[allow(clippy::too_many_arguments)]
fn build_vit_audio_opts(
    cx: &mut Context,
    model: &ClipModel,
    mut inp: TensorId,
    n_pos: i64,
    norm_t: NormType,
    ffn_t: FfnOp,
    learned_pos_embd: Option<TensorId>,
    n_embd: i64,
    n_head: i64,
    n_head_kv: i64,
    d_head: i64,
    kq_scale: f32,
    flash_attn: ClipFlashAttn,
    opts: &BuildVitAudioOpts,
) -> (TensorId, Option<TensorId>) {
    let eps = model.hparams.eps;

    // batch dim: inp is [n_embd, n_pos, B] (clip.cpp:339)
    let b = cx.ne(inp)[2];
    if let Some(pe) = learned_pos_embd {
        inp = cx.add(inp, pe); // clip.cpp:341-343
    }
    // flatten batch; unflatten again in attention (clip.cpp:346)
    let mut inp = cx.reshape_2d(inp, n_embd, n_pos * b);

    // pre-layernorm (clip.cpp:349-352)
    if let Some(w) = model.pre_ln_w {
        inp = build_norm(cx, inp, Some(w), model.pre_ln_b, norm_t, eps);
    }
    let mut inp_l = inp;
    let mut captured: Option<TensorId> = None;

    // loop over layers (clip.cpp:355)
    for (il, layer) in model.layers.iter().enumerate() {
        let mut cur = inp_l;

        // clip.cpp:358-361 — the per-layer mask, falling back to opts.attn_mask
        let mut attn_mask = opts.attn_mask;
        if opts.attn_mask_layers.len() > il {
            attn_mask = opts.attn_mask_layers[il];
        }

        // layernorm1 (clip.cpp:367)
        cur = build_norm(cx, cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps);

        // self-attention — fused qkv (clip.cpp:401-437, the qwen3a shape) or
        // separate q, k, v (clip.cpp:412-437, the whisper shape)
        let (mut q_cur, mut k_cur, v_cur);
        {
            if let Some(qkv_w) = layer.qkv_w {
                // fused qkv — clip.cpp:403-437: Q/K/V as [d_head, n_head,
                // n_pos, B] views of the fused output (no GQA)
                let mut c = cx.mul_mat(qkv_w, cur);
                if let Some(qkv_b) = layer.qkv_b {
                    c = cx.add(c, qkv_b);
                }
                let c_ty = cx.ty(c);
                let row = c_ty.row_size(d_head as usize);
                let cnb1 = cx.nb(c)[1] as usize;
                let batch_stride = cnb1 * n_pos as usize;
                q_cur = cx.view_4d(c, d_head, n_head, n_pos, b, row, cnb1, batch_stride, 0);
                k_cur = cx.view_4d(
                    c,
                    d_head,
                    n_head,
                    n_pos,
                    b,
                    row,
                    cnb1,
                    batch_stride,
                    c_ty.row_size((n_head * d_head) as usize),
                );
                v_cur = cx.view_4d(
                    c,
                    d_head,
                    n_head,
                    n_pos,
                    b,
                    row,
                    cnb1,
                    batch_stride,
                    c_ty.row_size((2 * n_head * d_head) as usize),
                );
            } else {
                // separate q, k, v — clip.cpp:412-437
                let mut q = cx.mul_mat(layer.q_w.unwrap(), cur);
                if let Some(qb) = layer.q_b {
                    q = cx.add(q, qb);
                }
                let mut k = cx.mul_mat(layer.k_w.unwrap(), cur);
                if let Some(kb) = layer.k_b {
                    k = cx.add(k, kb);
                }
                let mut v = cx.mul_mat(layer.v_w.unwrap(), cur);
                if let Some(vb) = layer.v_b {
                    v = cx.add(v, vb);
                }
                // no q_norm/k_norm in the audio family → norm_per_head is false
                // (clip.cpp:439-456 not taken)
                q_cur = cx.reshape_4d(q, d_head, n_head, n_pos, b);
                k_cur = cx.reshape_4d(k, d_head, n_head_kv, n_pos, b);
                v_cur = cx.reshape_4d(v, d_head, n_head_kv, n_pos, b);
            }
        }

        // add_pos hook (clip.cpp:459-464): mimo applies neox RoPE to Q/K
        if opts.add_pos {
            let pos = opts.inp_pos.expect("add_pos needs positions");
            q_cur = cx.rope_ext(
                q_cur,
                pos,
                None,
                d_head as i32,
                GGML_ROPE_TYPE_NEOX,
                0,
                opts.rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            k_cur = cx.rope_ext(
                k_cur,
                pos,
                None,
                d_head as i32,
                GGML_ROPE_TYPE_NEOX,
                0,
                opts.rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
        }

        // build_attn returns a flat 2D [n_embd, n_pos*B] (clip.cpp:465)
        cur = build_attn_ex(
            cx, layer.o_w, layer.o_b, q_cur, k_cur, v_cur, attn_mask, kq_scale, flash_attn,
        );

        // re-add the layer input, e.g., residual (clip.cpp:490)
        cur = cx.add(cur, inp_l);
        inp_l = cur;

        // callback_layer_out (clip.cpp:494-496): mimo's encoder skip layer
        if opts.capture_layer == Some(il as i32) {
            captured = Some(cur);
        }

        // layernorm2 (clip.cpp:503)
        cur = build_norm(cx, cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps);

        // ffn (clip.cpp:507-513)
        cur = build_ffn(cx, cur, layer, ffn_t);

        // residual 2 (clip.cpp:556)
        cur = cx.add(inp_l, cur);
        inp_l = cur;
    }

    // nn.AvgPool1d(2, stride=2) over the token dim (clip.cpp:545-556)
    if model.proj_type.audio_has_avgpool() {
        let mut c = cx.transpose(inp_l);
        c = cx.cont(c);
        c = pool_1d_avg2(cx, c);
        c = cx.transpose(c);
        inp_l = cx.cont(c);
    }

    // post-layernorm (clip.cpp:559-562, `&& !opts.skip_post_ln`)
    if let Some(w) = model.post_ln_w {
        if !opts.skip_post_ln {
            inp_l = build_norm(cx, inp_l, Some(w), model.post_ln_b, norm_t, eps);
        }
    }

    // restore the batch dim (clip.cpp:565-566)
    assert_eq!(cx.ne(inp_l)[1] % b, 0);
    let out = cx.reshape_3d(inp_l, n_embd, cx.ne(inp_l)[1] / b, b);
    (out, captured)
}

/// input handle of the whisper-enc graph
pub struct AudioGraphNodes {
    pub inp_raw: TensorId,
    pub out: TensorId,
    /// named graph inputs beyond inp_raw (clip.cpp `clip_set_inputs`): filled
    /// by `audio_batch_encode` before compute
    pub extra: Vec<(&'static str, TensorId, ExtraInput)>,
}

/// one `set_input_f32`/`set_input_i32` payload (clip.cpp:4535)
pub enum ExtraInput {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

/// the audio-encoder graph dispatcher (clip.cpp:1072-1140 `clip_graph::create`):
/// the whisper family → [`build_whisper_graph`], the newer encoders each get
/// their literal builder.
fn build_audio_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    match model.proj_type {
        ProjectorType::Gemma4UA => build_gemma4ua_graph(cx, model, img),
        ProjectorType::Qwen3A => build_qwen3a_graph(cx, model, img, flash_attn),
        ProjectorType::Lfm2A => build_conformer_graph(cx, model, img),
        ProjectorType::Gemma4A => build_gemma4a_graph(cx, model, img),
        ProjectorType::GraniteSpeech => build_granite_speech_graph(cx, model, img, flash_attn),
        ProjectorType::Parakeet => build_parakeet_graph(cx, model, img),
        ProjectorType::MimoAudio => build_mimo_graph(cx, model, img, flash_attn),
        ProjectorType::Qwen3TtsSpkEnc => build_qwen3tts_spkenc_graph(cx, model, img),
        ProjectorType::PocketTtsSpkEnc => build_pockettts_spkenc_graph(cx, model, img, flash_attn),
        _ => build_whisper_graph(cx, model, img, flash_attn),
    }
}

/// models/gemma4ua.cpp:4 `clip_graph_gemma4ua::build` — transpose the raw
/// waveform frames to [n_mel(=640), n_frames], rms_norm and one linear
/// (the Gemma4UnifiedMultimodalEmbedder, gemma4ua.cpp:9-15).
fn build_gemma4ua_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    let mut g = Graph::new(512);

    // build_inp_raw(1) (gemma4ua.cpp:5): [n_frames, 640, 1, 1] raw samples
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    // ggml_cont(ggml_permute(inp, 1, 0, 2, 3)) — [640, n_frames]
    let cur = cx.permute(inp_raw, 1, 0, 2, 3);
    let mut cur = cx.cont(cur);

    // embedding_pre_projection_norm (gemma4ua.cpp:12-13)
    cur = cx.rms_norm(cur, hp.eps);
    // build_mm(model.mm_input_proj_w, cur) (gemma4ua.cpp:13)
    let cur = cx.mul_mat(model.mm_input_proj_w.unwrap(), cur);

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra: Vec::new(),
        },
    ))
}

/// models/qwen3a.cpp:3 `clip_graph_qwen3a::build` — 100-frame chunked conv2d
/// stem (3x stride-2 + gelu_erf), conv_out projection, per-chunk learned
/// positions via repeat, build_vit (fused qkv) and the gelu_erf MLP tail.
fn build_qwen3a_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    // qwen3a.cpp:9-14
    let n_frames = img.nx as i64;
    let n_mel = img.ny as i64;
    let chunk_size: i64 = 100; // n_window * 2 (n_window=50 from model config)
    let n_chunks = n_frames / chunk_size;
    assert_eq!(
        n_frames % chunk_size,
        0,
        "preprocessor pads to chunk multiples"
    );
    assert_eq!(img.buf.len() as i64, n_frames * n_mel);

    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let n_head_kv = hp.n_head_kv as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    let mut g = Graph::new(4096);

    // build_inp_raw(1) (qwen3a.cpp:7)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, n_frames, n_mel, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    // view mel as batched chunks: [chunk_size, n_mel, 1, n_chunks]
    // (qwen3a.cpp:18-24); nb[1] = n_frames*4 (stride over mel bins of the
    // row-major frame layout), nb[3] = chunk_size*4
    let inp_v = cx.view_4d(
        inp_raw,
        chunk_size,
        n_mel,
        1,
        n_chunks,
        (n_frames * 4) as usize,   // nb[1]
        (chunk_size * 4) as usize, // nb[2]
        (chunk_size * 4) as usize, // nb[3]
        0,
    );
    let mut inp = cx.cont(inp_v);

    // 3 x conv2d + gelu (qwen3a.cpp:28-43)
    for i in 0..3 {
        let w = model.conv2d_w[i].unwrap();
        inp = cx.conv_2d(w, inp, 2, 2, 1, 1, 1, 1);
        if let Some(b) = model.conv2d_b[i] {
            let b4 = cx.reshape_4d(b, 1, 1, cx.ne(inp)[2], 1);
            inp = cx.add(inp, b4);
        }
        inp = cx.gelu_erf(inp);
    }

    // permute [OW, OH, OC, n_chunks] -> [OH, OC, OW, n_chunks], flatten to
    // [OH*OC, OW*n_chunks] (qwen3a.cpp:48-49)
    let inp = cx.permute(inp, 2, 0, 1, 3);
    let inp = cx.cont(inp);
    let (ne0, ne1, ne2, ne3) = {
        let t = cx.ne(inp);
        (t[0], t[1], t[2], t[3])
    };
    let mut inp = cx.reshape_2d(inp, ne0 * ne1, ne2 * ne3);

    // project to d_model (qwen3a.cpp:52-56)
    inp = cx.mul_mat(model.conv_out_w.unwrap(), inp);
    // conv_out has no bias in the loader arm (clip.cpp:2870); the C still
    // checks model.conv_out_b — always NULL for this arch
    let n_pos = cx.ne(inp)[1];

    // per-chunk positional embeddings: repeat pos[0:13] for each chunk
    // (qwen3a.cpp:60-70)
    let pos_embd = model.position_embeddings.expect("checked at load");
    {
        let tokens_per_chunk = n_pos / n_chunks; // 13
        let pne = *cx.ne(pos_embd);
        let pos_tmp = cx.view_2d(
            pos_embd,
            pne[0],
            tokens_per_chunk,
            cx.nb(pos_embd)[1] as usize,
            0,
        );
        let tgt = cx.new_tensor_2d(GgmlType::F32, pne[0], n_pos);
        let pos_rep = cx.repeat(pos_tmp, tgt);
        inp = cx.add(inp, pos_rep);
    }

    // build_vit (qwen3a.cpp:72-75): NORM_TYPE_NORMAL, gelu_erf FFN, no
    // learned pos (added above), no add_pos hook
    let mut cur = build_vit_audio(
        cx,
        model,
        inp,
        n_pos,
        NormType::Normal,
        hp.ffn_op,
        None,
        n_embd,
        n_head,
        n_head_kv,
        d_head,
        kq_scale,
        flash_attn,
    );

    // MLP projector (qwen3a.cpp:78-83)
    cur = build_ffn_plain(
        cx,
        cur,
        model.mm_1_w.unwrap(),
        model.mm_1_b,
        model.mm_2_w.unwrap(),
        model.mm_2_b,
        FfnOp::GeluErf,
    );

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra: Vec::new(),
        },
    ))
}

/// models/conformer.cpp:3 `clip_graph_conformer::build` (the LFM2-audio
/// encoder): conv-subsampling pre-encode, then conformer blocks — half-step
/// FFN / Shaw relative-position self-attention / depthwise-conv module /
/// half-step FFN / out norm — and the audio adapter MLP. The sinusoidal
/// relative-position table is a graph input filled from
/// `set_inputs` (clip.cpp:5635-5651).
#[allow(clippy::too_many_arguments)]
fn build_conformer_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    // conformer.cpp:4-7 (relaxed by a657f7e98 for d1omni_a files that ship no
    // learned position embeddings — the tensor is only asserted, never used)
    let n_frames = img.nx as i64;
    let n_pos = n_frames / 2;
    let n_pos_embd = (((((n_frames + 1) / 2) + 1) / 2 + 1) / 2) * 2 - 1;
    if let Some(pe) = model.position_embeddings {
        assert!(
            cx.ne(pe)[1] >= n_pos,
            "position_embeddings too small"
        );
    }

    // d_model is hardcoded 512 in the reference graph (conformer.cpp:9)
    let d_model: i64 = 512;
    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    const NORM_EPS: f32 = 1e-5; // conformer.cpp:75 etc — literal per call
    let fc_factor = 0.5f32; // conformer.cpp:82

    let mut g = Graph::new(4096);

    // the sinusoidal RPE input (conformer.cpp:9-12); filled in set_inputs
    let pos_emb = cx.new_tensor_2d(GgmlType::F32, d_model, n_pos_embd);
    cx.set_name(pos_emb, "pos_emb");

    // build_inp_raw(1) (conformer.cpp:14)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    // [n_mel, n_frames] (conformer.cpp:16)
    let mut cur = cx.transpose(inp_raw);
    let mut cur = cx.cont(cur);

    // ---- pre-encode: conv subsampling (conformer.cpp:18-62) ---------------
    {
        // layer.0 - conv2d + bias + relu (conformer.cpp:21-26)
        cur = cx.conv_2d(model.pre_conv_w[0].unwrap(), cur, 2, 2, 1, 1, 1, 1);
        cur = cx.add(cur, model.pre_conv_b[0].unwrap());
        cur = cx.relu_inplace(cur);

        // layer.2 - depthwise conv2d (conformer.cpp:29-30)
        cur = cx.conv_2d_dw_direct(model.pre_conv_w[2].unwrap(), cur, 2, 2, 1, 1, 1, 1);
        cur = cx.add(cur, model.pre_conv_b[2].unwrap());

        // layer.3 - direct conv2d (conformer.cpp:34-35)
        cur = cx.conv_2d_direct(model.pre_conv_w[3].unwrap(), cur, 1, 1, 0, 0, 1, 1);
        cur = cx.add(cur, model.pre_conv_b[3].unwrap());

        // layer.4 - relu (conformer.cpp:39)
        cur = cx.relu_inplace(cur);

        // layer.5 - depthwise conv2d (conformer.cpp:42-43)
        cur = cx.conv_2d_dw_direct(model.pre_conv_w[5].unwrap(), cur, 2, 2, 1, 1, 1, 1);
        cur = cx.add(cur, model.pre_conv_b[5].unwrap());

        // layer.6 - direct conv2d (conformer.cpp:47-48)
        cur = cx.conv_2d_direct(model.pre_conv_w[6].unwrap(), cur, 1, 1, 0, 0, 1, 1);
        cur = cx.add(cur, model.pre_conv_b[6].unwrap());

        // layer.7 - relu (conformer.cpp:52)
        cur = cx.relu_inplace(cur);

        // flatten channel and frequency axis (conformer.cpp:55-56)
        let t = cx.permute(cur, 0, 2, 1, 3);
        let t = cx.cont(t);
        let (t0, t1, t2) = {
            let n = cx.ne(t);
            (n[0], n[1], n[2])
        };
        cur = cx.reshape_2d(t, t0 * t1, t2);

        // out projection (conformer.cpp:59-60)
        cur = cx.mul_mat(model.pre_encode_out_w.unwrap(), cur);
        cur = cx.add(cur, model.pre_encode_out_b.unwrap());
    }

    // ---- conformer blocks (conformer.cpp:67-204) ---------------------------
    for layer in &model.layers {
        let mut residual = cur;

        // feed_forward1 (half-step) — conformer.cpp:75-83
        cur = build_norm(
            cx,
            cur,
            layer.ff_norm_w,
            layer.ff_norm_b,
            NormType::Normal,
            NORM_EPS,
        );
        cur = build_ffn(cx, cur, layer, FfnOp::Silu);
        residual = {
            let scaled = cx.scale(cur, fc_factor);
            cx.add(residual, scaled)
        };

        // self-attention with Shaw RPE — conformer.cpp:86-151
        {
            cur = build_norm(
                cx,
                residual,
                layer.ln_1_w,
                layer.ln_1_b,
                NormType::Normal,
                NORM_EPS,
            );

            let mut qcur = cx.mul_mat(layer.q_w.unwrap(), cur);
            qcur = cx.add(qcur, layer.q_b.unwrap());
            let qcol = cx.ne(qcur)[1];
            qcur = cx.reshape_3d(qcur, d_head, n_head, qcol);
            let mut q_bias_u = cx.add(qcur, layer.pos_bias_u.unwrap());
            q_bias_u = cx.permute(q_bias_u, 0, 2, 1, 3);
            let mut q_bias_v = cx.add(qcur, layer.pos_bias_v.unwrap());
            q_bias_v = cx.permute(q_bias_v, 0, 2, 1, 3);

            let mut kcur = cx.mul_mat(layer.k_w.unwrap(), cur);
            kcur = cx.add(kcur, layer.k_b.unwrap());
            kcur = cx.reshape_3d(kcur, d_head, n_head, cx.ne(kcur)[1]);
            kcur = {
                let t = cx.permute(kcur, 0, 2, 1, 3);
                cx.cont(t)
            };

            let mut vcur = cx.mul_mat(layer.v_w.unwrap(), cur);
            vcur = cx.add(vcur, layer.v_b.unwrap());
            vcur = cx.reshape_3d(vcur, d_head, n_head, cx.ne(vcur)[1]);
            vcur = {
                let t = cx.permute(vcur, 1, 2, 0, 3);
                cx.cont(t)
            };

            // content scores (conformer.cpp:110-112)
            let mut matrix_ac = cx.mul_mat(q_bias_u, kcur);
            matrix_ac = {
                let t = cx.permute(matrix_ac, 1, 0, 2, 3);
                cx.cont(t)
            };

            // position scores (conformer.cpp:114-120)
            let mut p = cx.mul_mat(layer.linear_pos_w.unwrap(), pos_emb);
            p = cx.reshape_3d(p, d_head, n_head, cx.ne(p)[1]);
            p = cx.permute(p, 0, 2, 1, 3);
            let mut matrix_bd = cx.mul_mat(q_bias_v, p);
            matrix_bd = {
                let t = cx.permute(matrix_bd, 1, 0, 2, 3);
                cx.cont(t)
            };

            // rel shift (conformer.cpp:123-133)
            {
                let pos_len = cx.ne(matrix_bd)[0];
                let q_len = cx.ne(matrix_bd)[1];
                let h = cx.ne(matrix_bd)[2];
                matrix_bd = cx.pad_ext(matrix_bd, 1, 0, 0, 0, 0, 0, 0, 0);
                matrix_bd = cx.reshape_3d(matrix_bd, q_len, pos_len + 1, h);
                let (nb1, nb2) = (cx.nb(matrix_bd)[1] as usize, cx.nb(matrix_bd)[2] as usize);
                let nb0 = cx.nb(matrix_bd)[0] as usize;
                matrix_bd =
                    cx.view_3d(matrix_bd, q_len, pos_len, h, nb1, nb2, nb0 * q_len as usize);
                matrix_bd = cx.cont_3d(matrix_bd, pos_len, q_len, h);
            }

            // conformer.cpp:135-139
            let (ac0, bd1, bd2, bdn1, bdn2) = {
                let ac = cx.ne(matrix_ac);
                let bdn = cx.nb(matrix_bd);
                (
                    ac[0],
                    cx.ne(matrix_bd)[1],
                    cx.ne(matrix_bd)[2],
                    bdn[1] as usize,
                    bdn[2] as usize,
                )
            };
            matrix_bd = cx.view_3d(matrix_bd, ac0, bd1, bd2, bdn1, bdn2, 0);
            let mut scores = cx.add(matrix_ac, matrix_bd);
            scores = cx.scale(scores, 1.0f32 / (d_head as f32).sqrt());

            // conformer.cpp:141-144
            let attn = cx.soft_max(scores);
            let mut x = cx.mul_mat(attn, vcur);
            x = cx.permute(x, 2, 0, 1, 3);
            let (x0, x1, x2) = {
                let n = cx.ne(x);
                (n[0], n[1], n[2])
            };
            x = cx.cont_2d(x, x0 * x1, x2);

            // conformer.cpp:146-148
            let mut out = cx.mul_mat(layer.o_w.expect("attn_out.weight"), x);
            out = cx.add(out, layer.o_b.unwrap());
            cur = out;
        }

        // conformer.cpp:153-155
        residual = cx.add(residual, cur);
        cur = build_norm(
            cx,
            residual,
            layer.norm_conv_w,
            layer.norm_conv_b,
            NormType::Normal,
            NORM_EPS,
        );

        // conv module — conformer.cpp:158-188
        {
            let mut x = cx.mul_mat(layer.conv_pw1_w.unwrap(), cur);
            x = cx.add(x, layer.conv_pw1_b.unwrap());

            // GLU with a sigmoid gate on the second half (conformer.cpp:166-171)
            {
                let (xn0, xn1, xnb1, xnb0) = {
                    let n = cx.ne(x);
                    let b = cx.nb(x);
                    (n[0], n[1], b[1] as usize, b[0] as usize)
                };
                let d = xn0 / 2;
                let gate_v = cx.view_2d(x, d, xn1, xnb1, d as usize * xnb0);
                // a657f7e98 (conformer.cpp:167): cont the strided view before
                // the sigmoid so the op sees a contiguous tensor on CUDA
                let gate_c = cx.cont(gate_v);
                let gate = cx.sigmoid(gate_c);
                let signal = cx.view_2d(x, d, xn1, xnb1, 0);
                x = cx.mul(signal, gate);
                let t = cx.transpose(x);
                x = cx.cont(t);
            }

            // causal depthwise conv via pad+roll+pad (conformer.cpp:174-178)
            x = cx.pad(x, 4, 0, 0, 0);
            x = cx.roll(x, 4, 0, 0, 0);
            x = cx.pad(x, 4, 0, 0, 0);
            x = cx.ssm_conv(x, layer.conv_dw_w.unwrap());
            x = cx.add(x, layer.conv_dw_b.unwrap());

            // folded batch norm + silu (conformer.cpp:180-181)
            let t = cx.mul(x, layer.conv_norm_w.unwrap());
            x = cx.add(t, layer.conv_norm_b.unwrap());
            x = cx.silu(x);

            // pointwise_conv2 (conformer.cpp:184-185)
            x = cx.mul_mat(layer.conv_pw2_w.unwrap(), x);
            x = cx.add(x, layer.conv_pw2_b.unwrap());

            cur = x;
        }

        // conformer.cpp:190-204
        residual = cx.add(residual, cur);
        cur = build_norm(
            cx,
            residual,
            layer.ff_norm_1_w,
            layer.ff_norm_1_b,
            NormType::Normal,
            NORM_EPS,
        );
        // ffn2 — the up/down pair of the second half-step
        cur = build_ffn_impl(
            cx,
            cur,
            layer.ff_up_1_w.unwrap(),
            layer.ff_up_1_b,
            None,
            None,
            layer.ff_down_1_w,
            layer.ff_down_1_b,
            FfnOp::Silu,
        );
        let scaled = cx.scale(cur, fc_factor);
        residual = cx.add(residual, scaled);
        cur = build_norm(
            cx,
            residual,
            layer.ln_2_w,
            layer.ln_2_b,
            NormType::Normal,
            NORM_EPS,
        );
    }

    // audio adapter (conformer.cpp:207-209)
    cur = build_norm(
        cx,
        cur,
        model.mm_0_w,
        model.mm_0_b,
        NormType::Normal,
        NORM_EPS,
    );
    cur = build_ffn_plain(
        cx,
        cur,
        model.mm_1_w.unwrap(),
        model.mm_1_b,
        model.mm_3_w.unwrap(),
        model.mm_3_b,
        FfnOp::GeluErf,
    );

    // d1omni_a: residual block after the projector (conformer.cpp:210-216) —
    // LayerNorm(1e-5) then a GELU-ERF down/up pair, added back to `cur`
    if let Some(mm_4_w) = model.mm_4_w {
        let x = build_norm(cx, cur, Some(mm_4_w), model.mm_4_b, NormType::Normal, 1e-5);
        let x = build_ffn_plain(
            cx,
            x,
            model.mm_5_w.unwrap(),
            model.mm_5_b,
            model.mm_6_w.unwrap(),
            model.mm_6_b,
            FfnOp::GeluErf,
        );
        cur = cx.add(cur, x);
    }

    // the pos_emb input payload (clip.cpp:5635-5651 set_inputs)
    let n_tok = ((((img.nx + 1) / 2) + 1) / 2 + 1) / 2;
    let seq_len = n_tok as usize * 2 - 1;
    let mut pos_data = vec![0.0f32; (d_model as usize) * seq_len];
    {
        let n_inv = (d_model / 2) as usize;
        let mut inv_freq = vec![0.0f64; n_inv];
        for (i, f) in inv_freq.iter_mut().enumerate() {
            *f = (-(f64::ln(10000.0) / d_model as f64) * (2.0 * i as f64)).exp();
        }
        for pos in 0..seq_len {
            for i in 0..n_inv {
                // the C multiplies int * double and narrows once (the `const
                // float ang` of clip.cpp:5646)
                let ang = (((n_tok as i64 - pos as i64 - 1) as f64) * inv_freq[i]) as f32;
                pos_data[pos * d_model as usize + 2 * i] = ang.sin();
                pos_data[pos * d_model as usize + 2 * i + 1] = ang.cos();
            }
        }
    }
    let extra = vec![("pos_emb", pos_emb, ExtraInput::F32(pos_data))];

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

/// ggml.c:4616 `ggml_conv_1d` with an explicit pad — the mimo downsample conv
/// (mimo-audio.cpp:68, s=2 p=0) and the pocket-tts SEANet convs
/// (pockettts-seanet.cpp:41) need p0 values other than K/2.
fn conv_1d_gen(cx: &mut Context, a: TensorId, b: TensorId, s0: i32, p0: i32, d0: i32) -> TensorId {
    let a_ty = cx.ty(a);
    // ggml.c:4619 — an F32 kernel reads the patches as F16
    let dst_type = if a_ty == GgmlType::Bf16 {
        GgmlType::F32
    } else {
        GgmlType::F16
    };
    let im2col = cx.im2col(a, b, s0, 0, p0, 0, d0, 0, false, dst_type);
    let (in0, in1, in2) = {
        let t = cx.ne(im2col);
        (t[0], t[1], t[2])
    };
    let (ane0, ane1, ane2) = {
        let t = cx.ne(a);
        (t[0], t[1], t[2])
    };
    // ggml.c:4625-4627 — [ICK, N*OL] x [ICK, OC]
    let lhs = cx.reshape_2d(im2col, in0, in2 * in1);
    let rhs = cx.reshape_2d(a, ane0 * ane1, ane2);
    let result = cx.mul_mat(lhs, rhs);
    // ggml.c:4629 — [OL, OC, N]
    cx.reshape_3d(result, in1, ane2, in2)
}

// ======================================================================
// audio round 4 — granite_speech / gemma4a / parakeet / mimo /
// qwen3tts_spkenc / pockettts_spkenc
// ======================================================================

/// models/granite-speech.cpp:5 `clip_graph_granite_speech::build` — chunked
/// Shaw-RPE conformer with a mid-stack CTC branch and a QFormer projector.
/// The `attn_dists` i32 input and the optional remainder `attn_mask` are the
/// set_inputs payloads (clip.cpp:5634-5680).
fn build_granite_speech_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    // granite-speech.cpp:6-14
    let n_frames = img.nx as i64;
    let context_size = hp.audio_chunk_size as i64;
    let ctc_layer = (hp.n_layer / 2) as i64;
    let conv_kernel = hp.audio_conv_kernel_size as i64;
    let conv_pad = conv_kernel / 2;

    let num_blocks = (n_frames + context_size - 1) / context_size;
    let padded_len = num_blocks * context_size;
    let remainder = n_frames % context_size;

    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();
    // granite-speech.cpp:17-18
    let proj_input_dim = n_embd * (hp.feature_layers.len() as i64 + 1);
    let use_feature_concat = !hp.feature_layers.is_empty();

    let mut g = Graph::new(4096);

    // granite-speech.cpp:20-30
    let attn_dists = cx.new_tensor_1d(GgmlType::I32, context_size * context_size);
    cx.set_name(attn_dists, "attn_dists");
    let attn_mask = if remainder > 0 {
        let m = cx.new_tensor_4d(GgmlType::F32, context_size, context_size, 1, num_blocks);
        cx.set_name(m, "attn_mask");
        Some(m)
    } else {
        None
    };

    // build_inp_raw(1) (granite-speech.cpp:32)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    let mut cur = {
        let t = cx.transpose(inp_raw);
        cx.cont(t)
    };
    cur = cx.mul_mat(model.inp_proj_w.unwrap(), cur);
    cur = cx.add(cur, model.inp_proj_b.unwrap());

    // capture layer 0 if requested (granite-speech.cpp:41-47)
    let mut concat_result: Option<TensorId> = None;
    if use_feature_concat && hp.feature_layers.contains(&0) {
        concat_result = Some(cur);
    }

    for (il, layer) in model.layers.iter().enumerate() {
        let il64 = il as i64;
        let mut residual = cur;

        // ffn1 (half-step, granite-speech.cpp:53-68)
        {
            let mut ffn1 = build_norm(
                cx,
                residual,
                layer.ff_norm_w,
                layer.ff_norm_b,
                NormType::Normal,
                hp.eps,
            );
            ffn1 = build_ffn(cx, ffn1, layer, FfnOp::Silu);
            residual = {
                let s = cx.scale(ffn1, 0.5);
                cx.add(residual, s)
            };
        }

        // chunked Shaw-RPE attention (granite-speech.cpp:70-121)
        {
            let mut normed = build_norm(
                cx,
                residual,
                layer.ln_1_w,
                layer.ln_1_b,
                NormType::Normal,
                hp.eps,
            );
            if n_frames < padded_len {
                normed = cx.pad(normed, 0, (padded_len - n_frames) as i32, 0, 0);
            }

            let q = cx.mul_mat(layer.q_w.unwrap(), normed);
            let k = cx.mul_mat(layer.k_w.unwrap(), normed);
            let v = cx.mul_mat(layer.v_w.unwrap(), normed);

            let q = cx.reshape_4d(q, d_head, n_head, context_size, num_blocks);
            let k = cx.reshape_4d(k, d_head, n_head, context_size, num_blocks);
            let v = cx.reshape_4d(v, d_head, n_head, context_size, num_blocks);

            let q_perm = cx.permute(q, 0, 2, 1, 3);
            let k_perm = {
                let t = cx.permute(k, 0, 2, 1, 3);
                cx.cont(t)
            };

            let kq = cx.mul_mat(k_perm, q_perm);

            // Shaw RPE (granite-speech.cpp:94-101): pos_emb ne[2]=1 broadcasts
            // against Q ne[2]=num_blocks in mul_mat
            let mut pos_emb = cx.get_rows(layer.attn_rel_pos_emb.unwrap(), attn_dists);
            pos_emb = cx.reshape_3d(pos_emb, d_head, context_size, context_size);
            pos_emb = cx.reshape_4d(pos_emb, d_head, context_size, 1, context_size);

            let q_shaw = cx.permute(q, 0, 1, 3, 2);
            let mut pos_attn = cx.mul_mat(pos_emb, q_shaw);
            pos_attn = {
                let t = cx.permute(pos_attn, 0, 2, 3, 1);
                cx.cont(t)
            };

            let scores = cx.add(kq, pos_attn);
            let attn_weights = cx.soft_max_ext(scores, attn_mask, kq_scale, 0.0);

            let v_perm = {
                let t = cx.permute(v, 1, 2, 0, 3);
                cx.cont(t)
            };
            let mut attn_out = cx.mul_mat(v_perm, attn_weights);

            attn_out = cx.permute(attn_out, 0, 2, 1, 3);
            attn_out = cx.cont_2d(attn_out, n_embd, padded_len);

            if n_frames < padded_len {
                attn_out = cx.view_2d(attn_out, n_embd, n_frames, cx.nb(attn_out)[1] as usize, 0);
            }

            let mut out = cx.mul_mat(layer.o_w.unwrap(), attn_out);
            if let Some(b) = layer.o_b {
                out = cx.add(out, b);
            }
            cur = out;
        }

        residual = cx.add(residual, cur);

        // conv module (granite-speech.cpp:125-162) — note: conv_dw has NO bias
        // in this arch (the loader never loads one) and the folded BN follows
        // the depthwise conv directly
        {
            cur = build_norm(
                cx,
                residual,
                layer.norm_conv_w,
                layer.norm_conv_b,
                NormType::Normal,
                hp.eps,
            );

            let mut x = cx.mul_mat(layer.conv_pw1_w.unwrap(), cur);
            x = cx.add(x, layer.conv_pw1_b.unwrap());

            // GLU (granite-speech.cpp:136-143)
            {
                let (xn0, xn1, xnb1, xnb0) = {
                    let n = cx.ne(x);
                    let b = cx.nb(x);
                    (n[0], n[1], b[1] as usize, b[0] as usize)
                };
                let d = xn0 / 2;
                let gate_v = cx.view_2d(x, d, xn1, xnb1, d as usize * xnb0);
                let gate = cx.sigmoid(gate_v);
                let signal = cx.view_2d(x, d, xn1, xnb1, 0);
                x = cx.mul(signal, gate);
                x = {
                    let t = cx.transpose(x);
                    cx.cont(t)
                };
            }

            x = cx.pad_ext(x, conv_pad as i32, conv_pad as i32, 0, 0, 0, 0, 0, 0);
            x = cx.ssm_conv(x, layer.conv_dw_w.unwrap());

            // folded batch norm (granite-speech.cpp:153)
            let m = cx.mul(x, layer.conv_norm_w.unwrap());
            x = cx.add(m, layer.conv_norm_b.unwrap());
            x = cx.silu(x);

            x = cx.mul_mat(layer.conv_pw2_w.unwrap(), x);
            x = cx.add(x, layer.conv_pw2_b.unwrap());

            cur = x;
        }

        residual = cx.add(residual, cur);

        // ffn2 (half-step, granite-speech.cpp:166-180)
        {
            let mut ffn2 = build_norm(
                cx,
                residual,
                layer.ff_norm_1_w,
                layer.ff_norm_1_b,
                NormType::Normal,
                hp.eps,
            );
            ffn2 = build_ffn_impl(
                cx,
                ffn2,
                layer.ff_up_1_w.unwrap(),
                layer.ff_up_1_b,
                None,
                None,
                layer.ff_down_1_w,
                layer.ff_down_1_b,
                FfnOp::Silu,
            );
            residual = {
                let s = cx.scale(ffn2, 0.5);
                cx.add(residual, s)
            };
        }

        cur = build_norm(
            cx,
            residual,
            layer.ln_2_w,
            layer.ln_2_b,
            NormType::Normal,
            hp.eps,
        );

        // capture intermediate layer (granite-speech.cpp:186-196)
        if use_feature_concat && hp.feature_layers.contains(&((il + 1) as i32)) {
            concat_result = Some(match concat_result {
                None => cur,
                Some(c) => cx.concat(c, cur, 0),
            });
        }

        // CTC branch (granite-speech.cpp:198-207)
        if il64 + 1 == ctc_layer {
            let mut mid = cx.mul_mat(model.ctc_out_w.unwrap(), cur);
            mid = cx.add(mid, model.ctc_out_b.unwrap());
            mid = cx.soft_max(mid);
            mid = cx.mul_mat(model.ctc_out_mid_w.unwrap(), mid);
            mid = cx.add(mid, model.ctc_out_mid_b.unwrap());
            cur = cx.add(cur, mid);
        }
    }

    // append the final output (granite-speech.cpp:210-215)
    if use_feature_concat {
        if let Some(c) = concat_result {
            cur = cx.concat(c, cur, 0);
        }
    }

    // QFormer projector (granite-speech.cpp:219-304)
    {
        let qf = model.qf.as_ref().unwrap();
        let window_size = hp.audio_proj_window_size as i64;
        let num_queries = window_size / hp.audio_proj_downsample_rate as i64;
        let proj_n_head = hp.audio_proj_head_count as i64;
        let proj_d_head = n_embd / proj_n_head;
        let proj_kq_scale = 1.0f32 / (proj_d_head as f32).sqrt();
        let proj_eps = 1e-12f32;
        let nblocks_proj = (n_frames + window_size - 1) / window_size;
        let padded_proj = nblocks_proj * window_size;

        if n_frames < padded_proj {
            cur = cx.pad(cur, 0, (padded_proj - n_frames) as i32, 0, 0);
        }

        let enc_windows = cx.reshape_3d(cur, proj_input_dim, window_size, nblocks_proj);

        let mut queries = build_norm(
            cx,
            qf.qf_proj_query.unwrap(),
            qf.qf_proj_norm_w,
            qf.qf_proj_norm_b,
            NormType::Normal,
            proj_eps,
        );
        {
            let q_3d = cx.reshape_3d(queries, n_embd, num_queries, 1);
            let q_shape = cx.new_tensor_3d(GgmlType::F32, n_embd, num_queries, nblocks_proj);
            queries = cx.repeat(q_3d, q_shape);
        }

        for pl in &qf.layers {
            // self-attention (granite-speech.cpp:249-266)
            {
                let q = {
                    let t = cx.mul_mat(pl.q_w.unwrap(), queries);
                    cx.add(t, pl.q_b.unwrap())
                };
                let k = {
                    let t = cx.mul_mat(pl.k_w.unwrap(), queries);
                    cx.add(t, pl.k_b.unwrap())
                };
                let v = {
                    let t = cx.mul_mat(pl.v_w.unwrap(), queries);
                    cx.add(t, pl.v_b.unwrap())
                };

                let q4 = cx.reshape_4d(q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                let k4 = cx.reshape_4d(k, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                let v4 = cx.reshape_4d(v, proj_d_head, proj_n_head, num_queries, nblocks_proj);

                let sa_out = build_attn_ex(
                    cx,
                    pl.o_w,
                    pl.o_b,
                    q4,
                    k4,
                    v4,
                    None,
                    proj_kq_scale,
                    flash_attn,
                );
                let sa_out = cx.reshape_3d(sa_out, n_embd, num_queries, nblocks_proj);

                let sum = cx.add(sa_out, queries);
                queries = build_norm(cx, sum, pl.ln_1_w, pl.ln_1_b, NormType::Normal, proj_eps);
            }

            // cross-attention (granite-speech.cpp:268-285)
            {
                let q = {
                    let t = cx.mul_mat(pl.cross_attn_q_w.unwrap(), queries);
                    cx.add(t, pl.cross_attn_q_b.unwrap())
                };
                let k = {
                    let t = cx.mul_mat(pl.cross_attn_k_w.unwrap(), enc_windows);
                    cx.add(t, pl.cross_attn_k_b.unwrap())
                };
                let v = {
                    let t = cx.mul_mat(pl.cross_attn_v_w.unwrap(), enc_windows);
                    cx.add(t, pl.cross_attn_v_b.unwrap())
                };

                let q4 = cx.reshape_4d(q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                let k4 = cx.reshape_4d(k, proj_d_head, proj_n_head, window_size, nblocks_proj);
                let v4 = cx.reshape_4d(v, proj_d_head, proj_n_head, window_size, nblocks_proj);

                let ca_out = build_attn_ex(
                    cx,
                    pl.cross_attn_o_w,
                    pl.cross_attn_o_b,
                    q4,
                    k4,
                    v4,
                    None,
                    proj_kq_scale,
                    flash_attn,
                );
                let ca_out = cx.reshape_3d(ca_out, n_embd, num_queries, nblocks_proj);

                let sum = cx.add(ca_out, queries);
                queries = build_norm(
                    cx,
                    sum,
                    pl.cross_attn_norm_w,
                    pl.cross_attn_norm_b,
                    NormType::Normal,
                    proj_eps,
                );
            }

            // ffn (granite-speech.cpp:287-298)
            {
                let ffn_out = build_ffn_impl(
                    cx,
                    queries,
                    pl.ff_up_w.unwrap(),
                    pl.ff_up_b,
                    None,
                    None,
                    pl.ff_down_w,
                    pl.ff_down_b,
                    FfnOp::Gelu,
                );
                let sum = cx.add(ffn_out, queries);
                queries = build_norm(cx, sum, pl.ln_2_w, pl.ln_2_b, NormType::Normal, proj_eps);
            }
        }

        cur = cx.reshape_2d(queries, n_embd, num_queries * nblocks_proj);
        let proj = cx.mul_mat(qf.qf_proj_linear_w.unwrap(), cur);
        cur = cx.add(proj, qf.qf_proj_linear_b.unwrap());
    }

    // set_inputs payloads (clip.cpp:5634-5680)
    let mut extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        let cs = context_size as usize;
        let mut dists = vec![0i32; cs * cs];
        for i in 0..context_size as usize {
            for j in 0..context_size as usize {
                let mut d = i as i64 - j as i64;
                if d < -context_size {
                    d = -context_size;
                }
                if d > context_size {
                    d = context_size;
                }
                dists[i * cs + j] = (d + hp.audio_max_pos_emb as i64) as i32;
            }
        }
        extra.push(("attn_dists", attn_dists, ExtraInput::I32(dists)));
        if let Some(m) = attn_mask {
            let mut mask = vec![0.0f32; cs * cs * num_blocks as usize];
            let last_block_offset = (num_blocks as usize - 1) * cs * cs;
            for q in 0..context_size as usize {
                for k in 0..context_size as usize {
                    if q as i64 >= remainder || k as i64 >= remainder {
                        mask[last_block_offset + q * cs + k] = f32::NEG_INFINITY;
                    }
                }
            }
            extra.push(("attn_mask", m, ExtraInput::F32(mask)));
        }
    }

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

/// gemma4a's ClippableLinear `build_mm` (gemma4a.cpp:279-288): clamp the
/// input, matmul, clamp the output — the four scalars live in the
/// clamp_info_map keyed by the weight's GGUF name (clip.cpp:3315-3333).
fn gemma4a_mm(cx: &mut Context, model: &ClipModel, w: TensorId, x: TensorId) -> TensorId {
    let ci = model
        .clamp_info_map
        .get(cx.name(w))
        .copied()
        .unwrap_or(ClampInfo {
            inp_min: -f32::MAX,
            inp_max: f32::MAX,
            out_min: -f32::MAX,
            out_max: f32::MAX,
        });
    let clamped = cx.clamp(x, ci.inp_min, ci.inp_max);
    let out = cx.mul_mat(w, clamped);
    cx.clamp(out, ci.out_min, ci.out_max)
}

/// gemma4a's FFN — `build_ffn(cur, up, NULL, NULL, down, NULL, FFN_SILU, il)`
/// with every mm through `build_mm` (the ClippableLinear clamps; gemma4a
/// loads no FFN biases): clamp-in, mm, clamp-out, silu, clamp-in, mm,
/// clamp-out (gemma4a.cpp:74-79)
fn gemma4a_ffn(
    cx: &mut Context,
    model: &ClipModel,
    cur: TensorId,
    up: TensorId,
    down: TensorId,
) -> TensorId {
    let mut x = gemma4a_mm(cx, model, up, cur);
    x = cx.silu(x);
    gemma4a_mm(cx, model, down, x)
}

/// models/gemma4a.cpp:11 `clip_graph_gemma4a::build` — SSCP subsampling, then
/// conformer blocks with chunked local attention (C=12, P=12, S=24), softcap
/// tanh, per-dim Q/K scales and ClippableLinear everywhere.
fn build_gemma4a_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    let res_weight = 0.5f32; // gemma4a.cpp:12
    let norm_eps = 1e-6f32; // gemma4a.cpp:13

    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };

    let mut g = Graph::new(4096);

    // 1. input (gemma4a.cpp:16-17)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");
    let mut cur = {
        let t = cx.transpose(inp_raw);
        cx.cont(t)
    };

    // 2. subsampling conv2d (gemma4a.cpp:19-44)
    {
        for i in 0..2usize {
            cur = cx.conv_2d(model.sscp_conv_w[i].unwrap(), cur, 2, 2, 1, 1, 1, 1);
            if let Some(b) = model.sscp_conv_b[i] {
                cur = cx.add(cur, b);
            }
            // nn.LayerNorm(channels): permute ch to ne[0], normalize, back
            if let Some(w) = model.sscp_norm_w[i] {
                cur = {
                    let t = cx.permute(cur, 1, 2, 0, 3);
                    cx.cont(t)
                };
                cur = cx.norm(cur, norm_eps);
                cur = cx.mul(cur, w);
                cur = {
                    let t = cx.permute(cur, 2, 0, 1, 3);
                    cx.cont(t)
                };
            }
            cur = cx.relu(cur);
        }
        // flatten [freq, time, ch, 1] -> [ch*freq, time]
        cur = {
            let t = cx.permute(cur, 1, 2, 0, 3);
            cx.cont(t)
        };
        let (c0, c1, c2) = {
            let n = cx.ne(cur);
            (n[0], n[1], n[2])
        };
        cur = cx.reshape_2d(cur, c0 * c1, c2);
        if let Some(w) = model.sscp_inp_proj_w {
            cur = gemma4a_mm(cx, model, w, cur);
            if let Some(b) = model.sscp_inp_proj_b {
                cur = cx.add(cur, b);
            }
        }
    }

    let n_pos = cx.ne(cur)[1];

    // chunked local attention constants (gemma4a.cpp:49-55)
    let c_chunks: i64 = 12;
    let p_past: i64 = 12;
    let s_ctx: i64 = c_chunks + p_past;
    let r_rpe: i64 = p_past + 1;
    let b_blocks: i64 = (n_pos + c_chunks - 1) / c_chunks;
    let n_padded = b_blocks * c_chunks;
    let pad_seq = n_padded - n_pos;

    // inputs: blocked RPE + blocked attention mask (gemma4a.cpp:58-64)
    let pos_emb = cx.new_tensor_2d(GgmlType::F32, n_head * d_head, r_rpe);
    cx.set_name(pos_emb, "pos_emb");
    let kq_mask = cx.new_tensor_3d(GgmlType::F32, s_ctx, c_chunks, b_blocks);
    cx.set_name(kq_mask, "kq_mask");

    // 3. conformer blocks (gemma4a.cpp:66-256)
    for layer in &model.layers {
        let mut residual = cur;

        // FFN 1 (half-step, gemma4a.cpp:71-81) — build_ffn routes every mm
        // through build_mm, i.e. the ClippableLinear clamps apply here too
        if layer.ff_norm_w.is_some() && layer.ff_up_w.is_some() && layer.ff_down_w.is_some() {
            cur = build_norm(cx, residual, layer.ff_norm_w, None, NormType::Rms, norm_eps);
            cur = gemma4a_ffn(
                cx,
                model,
                cur,
                layer.ff_up_w.unwrap(),
                layer.ff_down_w.unwrap(),
            );
            if let Some(w) = layer.ff_post_norm_w {
                cur = build_norm(cx, cur, Some(w), None, NormType::Rms, norm_eps);
            }
            residual = {
                let s = cx.scale(cur, res_weight);
                cx.add(residual, s)
            };
        }

        // chunked local self-attention with RPE (gemma4a.cpp:83-205)
        if layer.q_w.is_some() && layer.k_w.is_some() && layer.v_w.is_some() && layer.o_w.is_some()
        {
            let q_scale = (1.0f32 / (d_head as f32).sqrt()) / 2.0f32.ln(); // gemma4a.cpp:85
                                                                           // gemma4a.cpp:86: k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f)
            let k_scale = (1.0f32 + 1.0f32.exp()).ln() / 2.0f32.ln();
            let softcap = 50.0f32; // gemma4a.cpp:87

            let attn_norm_w = layer.attn_pre_norm_w.or(layer.ln_1_w);
            cur = match attn_norm_w {
                Some(w) => build_norm(cx, residual, Some(w), None, NormType::Rms, norm_eps),
                None => residual,
            };

            let mut qcur = gemma4a_mm(cx, model, layer.q_w.unwrap(), cur);
            let mut kcur = gemma4a_mm(cx, model, layer.k_w.unwrap(), cur);
            let vcur = gemma4a_mm(cx, model, layer.v_w.unwrap(), cur);

            // [n_embd, n_pos] -> [D, H, N]
            qcur = cx.reshape_3d(qcur, d_head, n_head, n_pos);
            kcur = cx.reshape_3d(kcur, d_head, n_head, n_pos);
            let mut vcur = cx.reshape_3d(vcur, d_head, n_head, n_pos);

            // Q/K scaling (gemma4a.cpp:103-111)
            qcur = cx.scale(qcur, q_scale);
            if let Some(w) = layer.per_dim_scale_w {
                let w3 = cx.reshape_3d(w, d_head, 1, 1);
                qcur = cx.mul(qcur, w3);
            }
            kcur = cx.scale(kcur, k_scale);
            if let Some(w) = layer.per_dim_k_scale_w {
                let w3 = cx.reshape_3d(w, d_head, 1, 1);
                kcur = cx.mul(kcur, w3);
            }

            // Q blocking (gemma4a.cpp:113-117) — ggml_pad(Qcur, 0, 0,
            // pad_seq, 0) right-pads dim2 (the token dim)
            qcur = cx.pad_ext(qcur, 0, 0, 0, 0, 0, pad_seq as i32, 0, 0);
            qcur = cx.reshape_4d(qcur, d_head, n_head, c_chunks, b_blocks);
            qcur = {
                let t = cx.permute(qcur, 0, 3, 1, 2);
                cx.cont(t)
            };

            // K/V block context extraction via overlapping view
            // (gemma4a.cpp:122-135)
            let extract_blocks = |cx: &mut Context, mut t: TensorId| -> TensorId {
                let pad_kv = s_ctx * b_blocks - n_pos;
                // left pad by P, right pad to S*B (gemma4a.cpp:129)
                t = cx.pad_ext(
                    t,
                    0,
                    0,
                    0,
                    0,
                    p_past as i32,
                    (pad_kv - p_past) as i32,
                    0,
                    0,
                );
                let (nb1, nb2) = (cx.nb(t)[1] as usize, cx.nb(t)[2] as usize);
                t = cx.view_4d(
                    t,
                    d_head,
                    n_head,
                    s_ctx,
                    b_blocks,
                    nb1,
                    nb2,
                    c_chunks as usize * nb2,
                    0,
                );
                cx.cont(t)
            };

            let mut kblk = extract_blocks(cx, kcur);
            kblk = {
                let t = cx.permute(kblk, 0, 3, 1, 2);
                cx.cont(t)
            };

            let mut vblk = extract_blocks(cx, vcur);
            vblk = {
                let t = cx.permute(vblk, 1, 3, 0, 2);
                cx.cont(t)
            };

            // content attention: Q @ K^T (gemma4a.cpp:145-147)
            let mut matrix_ac = cx.mul_mat(kblk, qcur);

            // relative position attention (gemma4a.cpp:149-173)
            if let Some(rel_w) = layer.attn_k_rel_w {
                let mut p = cx.mul_mat(rel_w, pos_emb);
                p = cx.reshape_3d(p, d_head, n_head, r_rpe);
                p = {
                    let t = cx.permute(p, 0, 2, 1, 3);
                    cx.cont(t)
                };

                let q_flat = cx.reshape_3d(qcur, d_head, c_chunks * b_blocks, n_head);
                let mut matrix_bd = cx.mul_mat(p, q_flat);
                matrix_bd = cx.reshape_4d(matrix_bd, r_rpe, c_chunks, b_blocks, n_head);

                // blocked relative shift (gemma4a.cpp:161-170)
                {
                    matrix_bd =
                        cx.pad_ext(matrix_bd, 0, (s_ctx + 1 - r_rpe) as i32, 0, 0, 0, 0, 0, 0);
                    matrix_bd = cx.reshape_3d(matrix_bd, (s_ctx + 1) * c_chunks, b_blocks, n_head);
                    let (nb1, nb2) = (cx.nb(matrix_bd)[1] as usize, cx.nb(matrix_bd)[2] as usize);
                    matrix_bd =
                        cx.view_3d(matrix_bd, c_chunks * s_ctx, b_blocks, n_head, nb1, nb2, 0);
                    matrix_bd = cx.cont(matrix_bd);
                    matrix_bd = cx.reshape_4d(matrix_bd, s_ctx, c_chunks, b_blocks, n_head);
                }

                matrix_ac = cx.add(matrix_ac, matrix_bd);
            }

            let mut scores = matrix_ac;

            // softcap (gemma4a.cpp:177-180)
            scores = cx.scale(scores, 1.0 / softcap);
            scores = cx.tanh(scores);
            scores = cx.scale(scores, softcap);

            // blocked attention mask (gemma4a.cpp:182-183)
            scores = cx.add(scores, kq_mask);

            let attn = cx.soft_max(scores);

            // attn @ V (gemma4a.cpp:185-196)
            let mut x = cx.mul_mat(vblk, attn);
            x = {
                let t = cx.permute(x, 0, 2, 3, 1);
                cx.cont(t)
            };
            x = cx.cont_2d(x, d_head * n_head, c_chunks * b_blocks);
            if pad_seq > 0 {
                x = cx.view_2d(x, d_head * n_head, n_pos, cx.nb(x)[1] as usize, 0);
                x = cx.cont(x);
            }

            x = gemma4a_mm(cx, model, layer.o_w.unwrap(), x);
            if let Some(b) = layer.o_b {
                x = cx.add(x, b);
            }

            if let Some(w) = layer.attn_post_norm_w {
                x = build_norm(cx, x, Some(w), None, NormType::Rms, norm_eps);
            }
            residual = cx.add(residual, x);
        }

        // conv module (gemma4a.cpp:207-237)
        if layer.norm_conv_w.is_some()
            && layer.conv_pw1_w.is_some()
            && layer.conv_dw_w.is_some()
            && layer.conv_pw2_w.is_some()
        {
            cur = build_norm(
                cx,
                residual,
                layer.norm_conv_w,
                None,
                NormType::Rms,
                norm_eps,
            );
            let mut x = gemma4a_mm(cx, model, layer.conv_pw1_w.unwrap(), cur);

            // GLU (gemma4a.cpp:212-220)
            {
                let (xn0, xn1, xnb1, xnb0) = {
                    let n = cx.ne(x);
                    let b = cx.nb(x);
                    (n[0], n[1], b[1] as usize, b[0] as usize)
                };
                let d = xn0 / 2;
                let gate_v = cx.view_2d(x, d, xn1, xnb1, d as usize * xnb0);
                let gate = {
                    let t = cx.cont(gate_v);
                    cx.sigmoid(t)
                };
                let signal = cx.view_2d(x, d, xn1, xnb1, 0);
                x = cx.mul(signal, gate);
                x = {
                    let t = cx.transpose(x);
                    cx.cont(t)
                };
            }

            // causal depthwise conv (gemma4a.cpp:222-228), left padded only
            x = cx.pad_ext(x, 4, 0, 0, 0, 0, 0, 0, 0);
            x = cx.ssm_conv(x, layer.conv_dw_w.unwrap());
            if let Some(b) = layer.conv_dw_b {
                x = cx.add(x, b);
            }

            if let Some(w) = layer.conv_norm_w {
                x = cx.rms_norm(x, norm_eps);
                x = cx.mul(x, w);
            }
            x = cx.silu(x);
            x = gemma4a_mm(cx, model, layer.conv_pw2_w.unwrap(), x);
            residual = cx.add(residual, x);
        }

        // FFN 2 (half-step, gemma4a.cpp:239-249) — same build_mm clamps
        if layer.ff_norm_1_w.is_some() && layer.ff_up_1_w.is_some() && layer.ff_down_1_w.is_some() {
            cur = build_norm(
                cx,
                residual,
                layer.ff_norm_1_w,
                None,
                NormType::Rms,
                norm_eps,
            );
            cur = gemma4a_ffn(
                cx,
                model,
                cur,
                layer.ff_up_1_w.unwrap(),
                layer.ff_down_1_w.unwrap(),
            );
            if let Some(w) = layer.ff_post_norm_1_w {
                cur = build_norm(cx, cur, Some(w), None, NormType::Rms, norm_eps);
            }
            residual = {
                let s = cx.scale(cur, res_weight);
                cx.add(residual, s)
            };
        }

        // layer output norm (gemma4a.cpp:251-254)
        cur = match layer.ln_2_w {
            Some(w) => build_norm(cx, residual, Some(w), None, NormType::Rms, norm_eps),
            None => residual,
        };
    }

    // 4. output projection (gemma4a.cpp:258-264)
    if let Some(w) = model.audio_out_proj_w {
        cur = gemma4a_mm(cx, model, w, cur);
        if let Some(b) = model.audio_out_proj_b {
            cur = cx.add(cur, b);
        }
    }

    // 5. audio multimodal embedder (gemma4a.cpp:266-273)
    cur = cx.rms_norm(cur, norm_eps);
    if let Some(w) = model.mm_soft_emb_norm_w {
        cur = cx.mul(cur, w);
    }
    if let Some(w) = model.mm_input_proj_w {
        cur = gemma4a_mm(cx, model, w, cur);
    }

    // set_inputs payloads (clip.cpp:5429-5482)
    let mut extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        // blocked causal attention mask
        let mut mask = vec![-1e9f32; (s_ctx * c_chunks * b_blocks) as usize];
        for b in 0..b_blocks {
            for q in 0..c_chunks {
                let gq = b * c_chunks + q;
                for k in 0..s_ctx {
                    let gk = b * c_chunks - p_past + k;
                    if gq < n_pos && gk >= 0 && gk < n_pos && gk <= gq && (gq - gk) < p_past {
                        mask[(k + q * s_ctx + b * s_ctx * c_chunks) as usize] = 0.0;
                    }
                }
            }
        }
        extra.push(("kq_mask", kq_mask, ExtraInput::F32(mask)));

        // sinusoidal RPE: 13 positions [12, 11, ..., 0]
        let num_timescales = n_embd / 2;
        let log_timescale_increment = 10000.0f32.ln() / (num_timescales - 1).max(1) as f32;
        let rpe_len = (p_past + 1) as usize;
        let mut pos_data = vec![0.0f32; (n_embd * r_rpe) as usize];
        for p in 0..rpe_len {
            let position = p_past as f32 - p as f32;
            for i in 0..num_timescales as usize {
                let inv_ts = (-(i as f32) * log_timescale_increment).exp();
                let scaled = position * inv_ts;
                pos_emb_data_set(
                    &mut pos_data,
                    p,
                    i,
                    num_timescales as usize,
                    n_embd as usize,
                    scaled,
                );
            }
        }
        extra.push(("pos_emb", pos_emb, ExtraInput::F32(pos_data)));
    }

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

/// the gemma4a sinusoidal RPE writes sin at [p][i] and cos at [p][i+nt]
/// (clip.cpp:5472-5481)
fn pos_emb_data_set(data: &mut [f32], p: usize, i: usize, nt: usize, n_embd: usize, scaled: f32) {
    data[p * n_embd + i] = scaled.sin();
    data[p * n_embd + i + nt] = scaled.cos();
}

/// models/parakeet.cpp:7 `clip_graph_parakeet::build` — conv subsampling, a
/// conformer encoder with relative-position attention (full for short input,
/// chunked local windows above the 8192 threshold) and a relu-sqr projection.
fn build_parakeet_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    let mut g = Graph::new(8192);

    // conv subsampling (parakeet.cpp:9-63) — the lfm2a pre-encode stack
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");
    let cur0 = cx.transpose(inp_raw);
    let mut cur = cx.cont(cur0);

    cur = cx.conv_2d(model.pre_conv_w[0].unwrap(), cur, 2, 2, 1, 1, 1, 1);
    cur = cx.add(cur, model.pre_conv_b[0].unwrap());
    cur = cx.relu(cur);

    cur = cx.conv_2d_dw_direct(model.pre_conv_w[2].unwrap(), cur, 2, 2, 1, 1, 1, 1);
    cur = cx.add(cur, model.pre_conv_b[2].unwrap());

    cur = cx.conv_2d(model.pre_conv_w[3].unwrap(), cur, 1, 1, 0, 0, 1, 1);
    cur = cx.add(cur, model.pre_conv_b[3].unwrap());
    cur = cx.relu(cur);

    cur = cx.conv_2d_dw_direct(model.pre_conv_w[5].unwrap(), cur, 2, 2, 1, 1, 1, 1);
    cur = cx.add(cur, model.pre_conv_b[5].unwrap());

    cur = cx.conv_2d(model.pre_conv_w[6].unwrap(), cur, 1, 1, 0, 0, 1, 1);
    cur = cx.add(cur, model.pre_conv_b[6].unwrap());
    cur = cx.relu(cur);

    // [freq, time, chan, batch] -> [freq, chan, time] -> [(freq*chan), time]
    cur = {
        let t = cx.permute(cur, 0, 2, 1, 3);
        cx.cont(t)
    };
    let (n_freq, n_chan, n_frames) = {
        let n = cx.ne(cur);
        (n[0], n[1], n[2])
    };
    cur = cx.reshape_2d(cur, n_freq * n_chan, n_frames);

    cur = cx.mul_mat(model.pre_encode_out_w.unwrap(), cur);
    cur = cx.add(cur, model.pre_encode_out_b.unwrap());

    // encoder geometry (parakeet.cpp:67-78)
    let n_state = hp.n_embd as i64;
    let fc_factor = 0.5f32;
    let n_time = cx.ne(cur)[1];
    const PARAKEET_LOCAL_ATTN_THRESHOLD: i64 = 8192;
    const PARAKEET_LOCAL_ATTN_WINDOW: i64 = 128;
    let local_attn = n_time > PARAKEET_LOCAL_ATTN_THRESHOLD;
    let att_left = if local_attn {
        PARAKEET_LOCAL_ATTN_WINDOW
    } else {
        n_time - 1
    };
    let att_right = if local_attn {
        PARAKEET_LOCAL_ATTN_WINDOW
    } else {
        n_time - 1
    };
    let window_size = if local_attn {
        att_left + att_right + 1
    } else {
        2 * n_time - 1
    };
    let d_half = n_state / 2;
    let mask_dim = if local_attn { window_size } else { n_time };

    let n_head = hp.n_head as i64;
    let d_head = n_state / n_head;
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    // inputs (parakeet.cpp:81-99)
    let attn_mask = cx.new_tensor_2d(GgmlType::F32, mask_dim, n_time);
    cx.set_name(attn_mask, "attn_mask");
    let local_mask = if local_attn {
        let chunk = att_left + att_right;
        let m = cx.new_tensor_2d(GgmlType::F32, chunk + window_size - 1, chunk);
        cx.set_name(m, "local_mask");
        Some(m)
    } else {
        None
    };
    let pos_freqs = cx.new_tensor_1d(GgmlType::F32, d_half);
    cx.set_name(pos_freqs, "pos_freqs");
    let rel_positions = cx.new_tensor_2d(GgmlType::F32, 1, window_size);
    cx.set_name(rel_positions, "rel_positions");

    // in-graph sinusoidal RPE (parakeet.cpp:101-107)
    let freqs = cx.repeat_4d(pos_freqs, d_half, window_size, 1, 1);
    let theta = cx.mul(freqs, rel_positions);
    let sin_t = {
        let t = cx.sin(theta);
        cx.reshape_3d(t, 1, d_half, window_size)
    };
    let cos_t = {
        let t = cx.cos(theta);
        cx.reshape_3d(t, 1, d_half, window_size)
    };
    let pos_emb = {
        let cat = cx.concat(sin_t, cos_t, 0);
        let cont = cx.cont(cat);
        cx.reshape_2d(cont, n_state, window_size)
    };

    // encoder layers (parakeet.cpp:109-407)
    for layer in &model.layers {
        // FFN1 (parakeet.cpp:111-126)
        let residual0 = cur;
        cur = cx.norm(cur, hp.eps);
        let m = cx.mul(cur, layer.ff_norm_w.unwrap());
        cur = cx.add(m, layer.ff_norm_b.unwrap());
        cur = build_ffn(cx, cur, layer, FfnOp::Silu);
        let scaled = cx.scale(cur, fc_factor);
        cur = cx.add(residual0, scaled);

        // self-attention with relative position encoding
        let residual = cur;
        cur = cx.norm(cur, hp.eps);
        let m = cx.mul(cur, layer.ln_1_w.unwrap());
        cur = cx.add(m, layer.ln_1_b.unwrap());

        let mut q_cur = cx.mul_mat(layer.q_w.unwrap(), cur);
        let mut k_cur = cx.mul_mat(layer.k_w.unwrap(), cur);
        let mut v_cur = cx.mul_mat(layer.v_w.unwrap(), cur);

        q_cur = cx.reshape_3d(q_cur, d_head, n_head, n_time);
        k_cur = cx.reshape_3d(k_cur, d_head, n_head, n_time);
        v_cur = cx.reshape_3d(v_cur, d_head, n_head, n_time);

        // [n_state, window_size] -> [feat, head, window] -> [feat, window, head]
        let mut pos = cx.mul_mat(layer.linear_pos_w.unwrap(), pos_emb);
        pos = cx.reshape_3d(pos, d_head, n_head, window_size);
        pos = {
            let t = cx.permute(pos, 0, 2, 1, 3);
            cx.cont(t)
        };

        if local_attn {
            // chunked local attention (parakeet.cpp:158-263)
            let chunk = att_left + att_right;
            let n_group = (n_time + chunk - 1) / chunk;
            let n_time_padded = n_group * chunk;
            let n_kv_chunk = chunk + window_size - 1;
            let n_kv_dense = n_kv_chunk * n_group;
            let need_padding = n_time_padded > n_time;

            q_cur = {
                let t = cx.permute(q_cur, 0, 2, 1, 3);
                cx.cont(t)
            };
            k_cur = {
                let t = cx.permute(k_cur, 0, 2, 1, 3);
                cx.cont(t)
            };
            v_cur = {
                let t = cx.permute(v_cur, 0, 2, 1, 3);
                cx.cont(t)
            };

            // content bias / position bias (parakeet.cpp:170-176)
            let bias_u = cx.reshape_3d(layer.pos_bias_u.unwrap(), d_head, 1, n_head);
            let q_u = cx.add(q_cur, bias_u);
            let bias_v = cx.reshape_3d(layer.pos_bias_v.unwrap(), d_head, 1, n_head);
            let q_v = cx.add(q_cur, bias_v);

            // right pad the time dimension (parakeet.cpp:178-181)
            let mut q_u_padded = q_u;
            if need_padding {
                q_u_padded = cx.pad_ext(q_u, 0, 0, 0, (n_time_padded - n_time) as i32, 0, 0, 0, 0);
            }
            q_u_padded = cx.reshape_4d(q_u_padded, d_head, chunk, n_group, n_head);

            // pad front and back for the first/last frames (parakeet.cpp:183-187)
            let mut k_padded =
                cx.pad_ext(k_cur, 0, 0, att_left as i32, att_right as i32, 0, 0, 0, 0);
            if n_kv_dense > cx.ne(k_padded)[1] {
                k_padded = cx.pad_ext(
                    k_padded,
                    0,
                    0,
                    0,
                    (n_kv_dense - cx.ne(k_padded)[1]) as i32,
                    0,
                    0,
                    0,
                    0,
                );
            }

            // sliding-window view (parakeet.cpp:189-196)
            let (knb1, knb2) = (cx.nb(k_padded)[1] as usize, cx.nb(k_padded)[2] as usize);
            let mut k_chunk = cx.view_4d(
                k_padded,
                d_head,
                n_kv_chunk,
                n_group,
                n_head,
                knb1,
                chunk as usize * knb1,
                knb2,
                0,
            );
            k_chunk = cx.cont(k_chunk);

            let mut content_scores = cx.mul_mat(k_chunk, q_u_padded);

            // trim to window_size scores per query (parakeet.cpp:200-210)
            let (cwnb0, cwnb2, cwnb3) = {
                let b = cx.nb(content_scores);
                (b[0] as usize, b[2] as usize, b[3] as usize)
            };
            content_scores = cx.view_4d(
                content_scores,
                window_size,
                chunk,
                n_group,
                n_head,
                (chunk + window_size) as usize * cwnb0,
                cwnb2,
                cwnb3,
                0,
            );
            content_scores = cx.cont(content_scores);

            content_scores = cx.reshape_3d(content_scores, window_size, n_time_padded, n_head);
            if need_padding {
                let (cnb1, cnb2) = (
                    cx.nb(content_scores)[1] as usize,
                    cx.nb(content_scores)[2] as usize,
                );
                content_scores =
                    cx.view_3d(content_scores, window_size, n_time, n_head, cnb1, cnb2, 0);
            }

            // Q_v (parakeet.cpp:219-223)
            let q_v = {
                let t = cx.permute(q_v, 0, 2, 1, 3);
                cx.cont(t)
            };
            let rel_pos_scores = cx.mul_mat(pos, q_v);

            let mut attn_scores = cx.add(content_scores, rel_pos_scores);
            attn_scores = cx.soft_max_ext(attn_scores, Some(attn_mask), kq_scale, 0.0);

            // expand probs back to n_kv_chunk width (parakeet.cpp:227-240)
            let mut probs_padded = attn_scores;
            if need_padding {
                probs_padded = cx.pad_ext(
                    attn_scores,
                    0,
                    0,
                    0,
                    (n_time_padded - n_time) as i32,
                    0,
                    0,
                    0,
                    0,
                );
            }
            probs_padded = cx.reshape_4d(probs_padded, window_size, chunk, n_group, n_head);
            probs_padded = cx.pad_ext(probs_padded, 0, chunk as i32, 0, 0, 0, 0, 0, 0);
            let (pnb0, pnb2, pnb3) = {
                let b = cx.nb(probs_padded);
                (b[0] as usize, b[2] as usize, b[3] as usize)
            };
            probs_padded = cx.view_4d(
                probs_padded,
                n_kv_chunk,
                chunk,
                n_group,
                n_head,
                n_kv_chunk as usize * pnb0,
                pnb2,
                pnb3,
                0,
            );
            probs_padded = cx.cont(probs_padded);
            probs_padded = cx.mul(probs_padded, local_mask.unwrap());

            // V (parakeet.cpp:242-256)
            let mut v_padded =
                cx.pad_ext(v_cur, 0, 0, att_left as i32, att_right as i32, 0, 0, 0, 0);
            if n_kv_dense > cx.ne(v_padded)[1] {
                v_padded = cx.pad_ext(
                    v_padded,
                    0,
                    0,
                    0,
                    (n_kv_dense - cx.ne(v_padded)[1]) as i32,
                    0,
                    0,
                    0,
                    0,
                );
            }
            v_padded = {
                let t = cx.transpose(v_padded);
                cx.cont(t)
            };

            let (vnb0, vnb2) = (cx.nb(v_padded)[0] as usize, cx.nb(v_padded)[2] as usize);
            let mut v_chunk = cx.view_4d(
                v_padded,
                n_kv_chunk,
                d_head,
                n_group,
                n_head,
                cx.nb(v_padded)[1] as usize,
                chunk as usize * vnb0,
                vnb2,
                0,
            );
            v_chunk = cx.cont(v_chunk);

            cur = cx.mul_mat(v_chunk, probs_padded);
            cur = cx.reshape_3d(cur, d_head, n_time_padded, n_head);
            if need_padding {
                let (cnb1, cnb2) = (cx.nb(cur)[1] as usize, cx.nb(cur)[2] as usize);
                cur = cx.view_3d(cur, d_head, n_time, n_head, cnb1, cnb2, 0);
            }
            cur = {
                let t = cx.permute(cur, 0, 2, 1, 3);
                cx.cont(t)
            };
            cur = cx.reshape_2d(cur, n_state, n_time);
            cur = cx.mul_mat(layer.o_w.unwrap(), cur);
        } else {
            // full attention (parakeet.cpp:264-336)
            let q_u = cx.add(q_cur, layer.pos_bias_u.unwrap());

            let k_prep = cx.permute(k_cur, 0, 2, 1, 3);
            let q_prep = cx.permute(q_u, 0, 2, 1, 3);
            let content_scores = cx.mul_mat(k_prep, q_prep);

            let q_v0 = cx.add(q_cur, layer.pos_bias_v.unwrap());
            let q_v = {
                let t = cx.permute(q_v0, 0, 2, 1, 3);
                cx.cont(t)
            };
            let mut rel_pos_scores = cx.mul_mat(pos, q_v);

            // relative positional shift, center-offset variant
            // (parakeet.cpp:284-317)
            {
                let pos_window = cx.ne(rel_pos_scores)[0];
                let n_frame = cx.ne(rel_pos_scores)[1];
                let n_head_l = cx.ne(rel_pos_scores)[2];

                rel_pos_scores = cx.pad_ext(rel_pos_scores, 1, 0, 0, 0, 0, 0, 0, 0);

                rel_pos_scores = cx.reshape_3d(rel_pos_scores, n_frame, pos_window + 1, n_head_l);
                rel_pos_scores = cx.cont(rel_pos_scores);

                let center = pos_window / 2;
                let offset = cx.nb(rel_pos_scores)[0] as usize * (center + 1) as usize;
                let (rnb2, rnb0) = (
                    cx.nb(rel_pos_scores)[2] as usize,
                    cx.nb(rel_pos_scores)[0] as usize,
                );
                rel_pos_scores = cx.view_3d(
                    rel_pos_scores,
                    n_frame,
                    pos_window,
                    n_head_l,
                    pos_window as usize * 4,
                    rnb2,
                    offset,
                );
                let _ = rnb0;
                rel_pos_scores = cx.cont(rel_pos_scores);

                let (cne0, cne1) = (cx.ne(content_scores)[0], cx.ne(content_scores)[1]);
                let (vnb1, vnb2) = (
                    cx.nb(rel_pos_scores)[1] as usize,
                    cx.nb(rel_pos_scores)[2] as usize,
                );
                rel_pos_scores = cx.view_3d(
                    rel_pos_scores,
                    cne0,
                    cne1,
                    cx.ne(rel_pos_scores)[2],
                    vnb1,
                    vnb2,
                    0,
                );
                rel_pos_scores = cx.cont(rel_pos_scores);
            }

            let mut attn_scores = cx.add(content_scores, rel_pos_scores);
            attn_scores = cx.scale(attn_scores, kq_scale);
            attn_scores = cx.add(attn_scores, attn_mask);

            let probs = cx.soft_max(attn_scores);

            let v_prep = {
                let t = cx.permute(v_cur, 1, 2, 0, 3);
                cx.cont(t)
            };
            cur = cx.mul_mat(probs, v_prep);

            cur = cx.permute(cur, 2, 0, 1, 3);
            cur = cx.cont_2d(cur, n_state, n_time);
            cur = cx.mul_mat(layer.o_w.unwrap(), cur);
        }

        cur = cx.add(residual, cur);

        // convolution module (parakeet.cpp:343-391)
        let residual_conv = cur;
        cur = cx.norm(cur, hp.eps);
        let m = cx.mul(cur, layer.norm_conv_w.unwrap());
        cur = cx.add(m, layer.norm_conv_b.unwrap());

        cur = cx.mul_mat(layer.conv_pw1_w.unwrap(), cur);

        {
            let d = cx.ne(cur)[0] / 2;
            let (cn1, cnb1, cnb0) = {
                let n = cx.ne(cur);
                let b = cx.nb(cur);
                (n[1], b[1] as usize, b[0] as usize)
            };
            let signal = cx.view_2d(cur, d, cn1, cnb1, 0);
            let gate_v = cx.view_2d(cur, d, cn1, cnb1, d as usize * cnb0);
            let gate = cx.sigmoid(gate_v);
            cur = cx.mul(signal, gate);
        }

        cur = {
            let t = cx.transpose(cur);
            cx.cont(t)
        };

        // causal depthwise conv (parakeet.cpp:365-371)
        let dw_pad = ((hp.audio_conv_kernel_size - 1) / 2) as i64;
        cur = cx.pad_ext(cur, dw_pad as i32, dw_pad as i32, 0, 0, 0, 0, 0, 0);

        cur = cx.ssm_conv(cur, layer.conv_dw_w.unwrap());

        // folded batch-norm from mean/var (parakeet.cpp:377-380)
        cur = cx.sub(cur, layer.conv_norm_mean.unwrap());
        let std = cx.sqrt(layer.conv_norm_var.unwrap());
        cur = cx.div(cur, std);
        let m = cx.mul(cur, layer.conv_norm_w.unwrap());
        cur = cx.add(m, layer.conv_norm_b.unwrap());

        cur = cx.silu(cur);

        cur = cx.mul_mat(layer.conv_pw2_w.unwrap(), cur);

        cur = cx.add(residual_conv, cur);

        // FFN2 (parakeet.cpp:393-403)
        let residual_f2 = cur;
        cur = cx.norm(cur, hp.eps);
        let m = cx.mul(cur, layer.ff_norm_1_w.unwrap());
        cur = cx.add(m, layer.ff_norm_1_b.unwrap());
        cur = build_ffn_impl(
            cx,
            cur,
            layer.ff_up_1_w.unwrap(),
            None,
            None,
            None,
            layer.ff_down_1_w,
            None,
            FfnOp::Silu,
        );
        let scaled = cx.scale(cur, 0.5);
        cur = cx.add(residual_f2, scaled);

        cur = cx.norm(cur, hp.eps);
        let m = cx.mul(cur, layer.ln_2_w.unwrap());
        cur = cx.add(m, layer.ln_2_b.unwrap());
    }

    // tail (parakeet.cpp:409-416)
    cur = cx.rms_norm(cur, 1e-6);
    cur = cx.mul(cur, model.mm_norm_pre_w.unwrap());
    cur = build_ffn_plain(
        cx,
        cur,
        model.mm_0_w.unwrap(),
        None,
        model.mm_1_w.unwrap(),
        None,
        FfnOp::ReluSqr,
    );

    // set_inputs payloads (clip.cpp:5552-5628)
    let mut extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        let n_q = n_time as usize;
        let n_k = mask_dim as usize;
        let n_tokens_real = ((img.nx as i64 + hp.subsampling_factor as i64 - 1)
            / hp.subsampling_factor as i64) as usize;
        let mask_value = -1e30f32;
        let mut mask_data = vec![0.0f32; n_q * n_k];
        if n_k == n_q {
            for q in 0..n_q {
                for k in 0..n_k {
                    mask_data[q * n_k + k] = if k >= n_tokens_real { mask_value } else { 0.0 };
                }
            }
        } else {
            let att_left = (n_k / 2) as i64;
            for q in 0..n_q {
                for k in 0..n_k {
                    let key = q as i64 - att_left + k as i64;
                    mask_data[q * n_k + k] = if key >= 0 && (key as usize) < n_tokens_real {
                        0.0
                    } else {
                        mask_value
                    };
                }
            }
        }
        extra.push(("attn_mask", attn_mask, ExtraInput::F32(mask_data)));

        if let Some(lm) = local_mask {
            let lm_k = cx.ne(lm)[0] as usize;
            let lm_q = cx.ne(lm)[1] as usize;
            let window_size_l = lm_k - lm_q + 1;
            let mut lm_data = vec![0.0f32; lm_q * lm_k];
            for q in 0..lm_q {
                for k in 0..lm_k {
                    let rel = k as i64 - q as i64;
                    lm_data[q * lm_k + k] = if rel >= 0 && (rel as usize) < window_size_l {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
            extra.push(("local_mask", lm, ExtraInput::F32(lm_data)));
        }

        // rotation frequencies (clip.cpp:5600-5608):
        // freqs[k] = expf(-(float(k * 2) * log_10000 / float(n_state)))
        let log_10000 = 10000.0f32.ln();
        let mut freqs = vec![0.0f32; d_half as usize];
        for (k, f) in freqs.iter_mut().enumerate() {
            *f = (-(k as f32 * 2.0) * log_10000 / n_state as f32).exp();
        }
        extra.push(("pos_freqs", pos_freqs, ExtraInput::F32(freqs)));

        // relative positional distances (clip.cpp:5610-5628)
        let ws = window_size as usize;
        let mut pos = vec![0.0f32; ws];
        if local_attn {
            let att_left = (window_size / 2) as i64;
            for (t, v) in pos.iter_mut().enumerate() {
                *v = (att_left - t as i64) as f32;
            }
        } else {
            let n_time_c = ((window_size + 1) / 2) as i64;
            for (t, v) in pos.iter_mut().enumerate() {
                *v = (n_time_c - 1 - t as i64) as f32;
            }
        }
        extra.push(("rel_positions", rel_positions, ExtraInput::F32(pos)));
    }

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

/// models/mimo-audio.cpp:3 `clip_graph_mimo_audio::build` — whisper-style
/// conv stem, a masked build_vit with per-layer RoPE, a stride-2 downsample
/// conv, the RVQ quantization loop and the input_local_transformer.
fn build_mimo_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let n_head_kv = hp.n_head_kv as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();
    const ENCODER_ROPE_THETA: f32 = 10000.0; // clip.cpp:1808

    let mut g = Graph::new(8192);

    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    // conv1d stem (mimo-audio.cpp:6-12)
    let mut cur = conv_1d_ph(cx, model.conv1d_1_w.unwrap(), inp_raw, 1, 1);
    cur = cx.add(cur, model.conv1d_1_b.unwrap());
    cur = cx.gelu_erf(cur);

    cur = conv_1d_ph(cx, model.conv1d_2_w.unwrap(), cur, 2, 1);
    cur = cx.add(cur, model.conv1d_2_b.unwrap());
    cur = cx.gelu_erf(cur);

    let mut inp_l = {
        let t = cx.transpose(cur);
        cx.cont(t)
    };
    let n_pos = cx.ne(inp_l)[1];

    assert_eq!(hp.wa_pattern_mode.len(), hp.n_layer as usize);

    // graph inputs (mimo-audio.cpp:20-30)
    let inp_pos = cx.new_tensor_1d(GgmlType::I32, n_pos);
    cx.set_name(inp_pos, "mimo_audio_positions");
    let full_mask = cx.new_tensor_2d(GgmlType::F32, n_pos, n_pos);
    cx.set_name(full_mask, "mimo_audio_full_mask");
    let window_mask = cx.new_tensor_2d(GgmlType::F32, n_pos, n_pos);
    cx.set_name(window_mask, "mimo_audio_window_mask");

    // build_vit opts (mimo-audio.cpp:32-60)
    let mut opts = BuildVitAudioOpts {
        attn_mask_layers: Vec::new(),
        attn_mask: None,
        capture_layer: Some(2), // encoder_skip_layer_id=3 (1-indexed)
        skip_post_ln: true,
        add_pos: true,
        inp_pos: Some(inp_pos),
        rope_theta: ENCODER_ROPE_THETA,
    };
    for mode in &hp.wa_pattern_mode {
        opts.attn_mask_layers.push(if *mode == -1 {
            Some(full_mask)
        } else {
            Some(window_mask)
        });
    }

    let (vit_out, skip_hidden) = build_vit_audio_opts(
        cx,
        model,
        inp_l,
        n_pos,
        NormType::Normal,
        hp.ffn_op,
        None,
        n_embd,
        n_head,
        n_head_kv,
        d_head,
        kq_scale,
        flash_attn,
        &opts,
    );
    // build_vit restores a (size-1) batch dim — flatten it back
    inp_l = cx.reshape_2d(vit_out, n_embd, n_pos);

    let skip_hidden = skip_hidden.expect("skip layer captured");
    inp_l = cx.add(inp_l, skip_hidden);

    inp_l = build_norm(
        cx,
        inp_l,
        model.post_ln_w,
        model.post_ln_b,
        NormType::Normal,
        hp.eps,
    );

    // downsample: strided conv + gelu + layernorm (mimo-audio.cpp:65-74)
    {
        let mut ds = {
            let t = cx.transpose(inp_l);
            cx.cont(t)
        };
        ds = conv_1d_gen(cx, model.downsample_conv_w.unwrap(), ds, 2, 0, 1);
        ds = cx.gelu_erf(ds);
        ds = {
            let t = cx.transpose(ds);
            cx.cont(t)
        };
        ds = build_norm(
            cx,
            ds,
            model.downsample_norm_w,
            model.downsample_norm_b,
            NormType::Normal,
            hp.eps,
        );
        inp_l = ds;
    }

    // RVQ quantize (mimo-audio.cpp:76-106)
    let mut codes: Vec<TensorId> = Vec::new();
    {
        let codebook = model.rvq_codebook.unwrap();
        let dim = cx.ne(codebook)[0];
        assert_eq!(dim, cx.ne(inp_l)[0]);
        assert_eq!(hp.rvq_codebook_size.len() as i64, cx.ne(codebook)[2]);

        let mut residual = inp_l;
        for q in 0..hp.rvq_codebook_size.len() {
            let bins = hp.rvq_codebook_size[q] as i64;
            let (cb_nb1, cb_nb2) = (cx.nb(codebook)[1] as usize, cx.nb(codebook)[2] as usize);
            let mut codebook_q = cx.view_2d(codebook, dim, bins, cb_nb1, q * cb_nb2);
            codebook_q = cx.cont(codebook_q);

            let cb_sqr = cx.sqr(codebook_q);
            let codebook_norm = cx.sum_rows(cb_sqr);
            let codebook_norm = {
                let t = cx.transpose(codebook_norm);
                cx.cont(t)
            };

            let dot = cx.mul_mat(codebook_q, residual);
            let dot2 = cx.scale(dot, 2.0);
            let scores = cx.sub(dot2, codebook_norm);

            let idx = cx.argmax(scores);
            codes.push(idx);

            let quant = cx.get_rows(codebook_q, idx);
            residual = cx.sub(residual, quant);
        }
    }

    // codes -> LLM embeddings (mimo-audio.cpp:108-126)
    let mut code_embd_sum: Option<TensorId> = None;
    {
        let embd = model.mm_a_code_embd.unwrap();
        let dim = cx.ne(embd)[0];
        let vocab = cx.ne(embd)[1];
        assert_eq!(codes.len() as i64, cx.ne(embd)[2]);

        for (i, code) in codes.iter().enumerate() {
            let (enb1, enb2) = (cx.nb(embd)[1] as usize, cx.nb(embd)[2] as usize);
            let mut table_i = cx.view_2d(embd, dim, vocab, enb1, i * enb2);
            table_i = cx.cont(table_i);

            let embd_i = cx.get_rows(table_i, *code);
            code_embd_sum = Some(match code_embd_sum {
                None => embd_i,
                Some(s) => cx.add(s, embd_i),
            });
        }
    }
    let code_embd_sum = code_embd_sum.unwrap();

    // input_local_transformer (mimo-audio.cpp:128-214)
    let projected;
    let mut local_extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        let group_size = hp.audio_local_group_size as i64;
        assert!(group_size > 0);
        let n_pos_ds = cx.ne(code_embd_sum)[1];
        let n_groups = (n_pos_ds + group_size - 1) / group_size;
        let n_padded = n_groups * group_size;

        let mut cur_local = code_embd_sum;
        if n_padded != n_pos_ds {
            cur_local = cx.pad(cur_local, 0, (n_padded - n_pos_ds) as i32, 0, 0);
        }

        let local_pos = cx.new_tensor_1d(GgmlType::I32, n_padded);
        cx.set_name(local_pos, "mimo_audio_local_positions");
        let local_mask = cx.new_tensor_2d(GgmlType::F32, n_padded, n_padded);
        cx.set_name(local_mask, "mimo_audio_local_mask");

        // the local transformer's input payloads (clip.cpp:5512-5530) — the
        // data only depends on the padded length, filled here while the
        // shapes are in scope
        {
            let gs = group_size as usize;
            let np = n_padded as usize;
            let local_positions: Vec<i32> = (0..np).map(|i| (i % gs) as i32).collect();
            local_extra.push((
                "mimo_audio_local_positions",
                local_pos,
                ExtraInput::I32(local_positions),
            ));
            let neg_inf = f32::MIN;
            let mut lm = vec![0.0f32; np * np];
            for q in 0..np {
                for k in 0..np {
                    lm[q * np + k] = if (q / gs) == (k / gs) { 0.0 } else { neg_inf };
                }
            }
            local_extra.push(("mimo_audio_local_mask", local_mask, ExtraInput::F32(lm)));
        }

        const LOCAL_ROPE_THETA: f32 = 640000.0; // mimo-audio.cpp:157
        let apply_local_rope = |cx: &mut Context, x: TensorId| -> TensorId {
            cx.rope_ext(
                x,
                local_pos,
                None,
                d_head as i32,
                GGML_ROPE_TYPE_NEOX,
                0,
                LOCAL_ROPE_THETA,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        };

        for layer in &model.mm_a_local_layers {
            let attn_in = build_norm(cx, cur_local, layer.ln_1_w, None, NormType::Rms, hp.eps);

            let mut qcur = cx.mul_mat(layer.q_w.unwrap(), attn_in);
            if let Some(b) = layer.q_b {
                qcur = cx.add(qcur, b);
            }
            let mut kcur = cx.mul_mat(layer.k_w.unwrap(), attn_in);
            if let Some(b) = layer.k_b {
                kcur = cx.add(kcur, b);
            }
            let mut vcur = cx.mul_mat(layer.v_w.unwrap(), attn_in);
            if let Some(b) = layer.v_b {
                vcur = cx.add(vcur, b);
            }

            let q3 = cx.reshape_3d(qcur, d_head, n_head, n_padded);
            let k3 = cx.reshape_3d(kcur, d_head, n_head, n_padded);
            let v3 = cx.reshape_3d(vcur, d_head, n_head, n_padded);

            let qr = apply_local_rope(cx, q3);
            let kr = apply_local_rope(cx, k3);

            let attn_out = build_attn_ex(
                cx,
                layer.o_w,
                None,
                qr,
                kr,
                v3,
                Some(local_mask),
                kq_scale,
                flash_attn,
            );
            cur_local = cx.add(cur_local, attn_out);

            let ffn_in = build_norm(cx, cur_local, layer.ln_2_w, None, NormType::Rms, hp.eps);
            let ffn_out = build_ffn(cx, ffn_in, layer, FfnOp::Silu);
            cur_local = cx.add(cur_local, ffn_out);
        }

        cur_local = build_norm(
            cx,
            cur_local,
            model.mm_a_local_norm_w,
            None,
            NormType::Rms,
            hp.eps,
        );

        // flatten each group into one (group_size*n_embd)-dim vector
        let grouped = cx.reshape_2d(cur_local, n_embd * group_size, n_groups);

        // AudioProjection: Linear -> GELU -> Linear, no biases
        projected = build_ffn_plain(
            cx,
            grouped,
            model.mm_1_w.unwrap(),
            None,
            model.mm_2_w.unwrap(),
            None,
            FfnOp::GeluErf,
        );
    }

    // set_inputs payloads (clip.cpp:5479-5530)
    let mut extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        let n_pos_u = n_pos as usize;
        let positions: Vec<i32> = (0..n_pos_u as i32).collect();
        extra.push(("mimo_audio_positions", inp_pos, ExtraInput::I32(positions)));

        let window = hp.attn_window_size as i64;
        assert!(window > 0);
        let neg_inf = f32::MIN; // std::numeric_limits<float>::lowest()
        let mut full = vec![0.0f32; n_pos_u * n_pos_u];
        let mut win = vec![0.0f32; n_pos_u * n_pos_u];
        for q in 0..n_pos_u {
            for k in 0..n_pos_u {
                let causal_ok = k <= q;
                full[q * n_pos_u + k] = if causal_ok { 0.0 } else { neg_inf };
                win[q * n_pos_u + k] = if causal_ok && (q - k) as i64 <= window {
                    0.0
                } else {
                    neg_inf
                };
            }
        }
        extra.push(("mimo_audio_full_mask", full_mask, ExtraInput::F32(full)));
        extra.push(("mimo_audio_window_mask", window_mask, ExtraInput::F32(win)));
    }
    extra.extend(local_extra);

    g.build_forward(cx, projected);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

/// models/qwen3tts-spkenc.cpp:8 `clip_graph_qwen3tts_spkenc::conv1d_same` —
/// a K-tap "same"-padding conv1d built from pad_reflect + a hand-rolled F32
/// im2col + mul_mat (the reflect pad brings T onto ne[0] first).
fn spk_conv1d_same(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: TensorId,
    dilation: i64,
) -> TensorId {
    let k = cx.ne(w)[0];
    let ic = cx.ne(w)[1];
    let oc = cx.ne(w)[2];
    let pad = ((k - 1) * dilation / 2) as i32;

    let mut x_t = {
        let t = cx.transpose(x);
        cx.cont(t)
    };
    if pad > 0 {
        x_t = cx.pad_reflect_1d(x_t, pad, pad);
    }
    let x4d = cx.reshape_4d(x_t, cx.ne(x_t)[0], ic, 1, 1);

    // dummy F32 kernel — im2col only reads its shape
    let dummy = cx.new_tensor_4d(GgmlType::F32, k, ic, 1, 1);

    let col = cx.im2col(
        dummy,
        x4d,
        1,
        1,
        0,
        0,
        dilation as i32,
        1,
        false,
        GgmlType::F32,
    );
    let t_out = cx.ne(col)[1];
    let col = cx.reshape_2d(col, k * ic, t_out);

    let w2d = cx.reshape_2d(w, k * ic, oc);
    let mut y = cx.mul_mat(w2d, col);

    let b2d = cx.reshape_2d(b, oc, 1);
    y = cx.add(y, b2d);
    y
}

/// models/qwen3tts-spkenc.cpp:39 `res2net` — split the channel axis into
/// `scale` chunks and chain dilated conv branches.
fn spk_res2net(
    cx: &mut Context,
    x: TensorId,
    layer: &ClipLayer,
    dilation: i64,
    scale: i64,
) -> TensorId {
    let c_dim = cx.ne(x)[0];
    let t = cx.ne(x)[1];
    let cs = c_dim / scale;

    let mut outs: Vec<TensorId> = Vec::with_capacity(scale as usize);
    let mut prev: Option<TensorId> = None;
    for i in 0..scale {
        let chunk = {
            let v = cx.view_2d(
                x,
                cs,
                t,
                cx.nb(x)[1] as usize,
                (i * cs) as usize * cx.nb(x)[0] as usize,
            );
            cx.cont(v)
        };
        if i == 0 {
            outs.push(chunk);
            continue;
        }
        let inp = if i >= 2 {
            cx.add(chunk, prev.unwrap())
        } else {
            chunk
        };
        let mut y = spk_conv1d_same(
            cx,
            inp,
            layer.res2_conv_w[(i - 1) as usize].unwrap(),
            layer.res2_conv_b[(i - 1) as usize].unwrap(),
            dilation,
        );
        y = cx.relu(y);
        outs.push(y);
        prev = Some(y);
    }

    let mut acc = outs[0];
    for o in outs.into_iter().skip(1) {
        acc = cx.concat(acc, o, 0);
    }
    acc
}

/// models/qwen3tts-spkenc.cpp:73 `se_block` — squeeze-and-excitation gate.
fn spk_se_block(cx: &mut Context, x: TensorId, layer: &ClipLayer) -> TensorId {
    // temporal mean over T (keepdim)
    let x_t = {
        let t = cx.transpose(x);
        cx.cont(t)
    };
    let mean = cx.mean(x_t);
    let mean = {
        let t = cx.transpose(mean);
        cx.cont(t)
    };

    let mut h = spk_conv1d_same(
        cx,
        mean,
        layer.se_conv1_w.unwrap(),
        layer.se_conv1_b.unwrap(),
        1,
    );
    h = cx.relu(h);
    h = spk_conv1d_same(
        cx,
        h,
        layer.se_conv2_w.unwrap(),
        layer.se_conv2_b.unwrap(),
        1,
    );
    h = cx.sigmoid(h);

    cx.mul(x, h)
}

/// models/qwen3tts-spkenc.cpp:154 `clip_graph_qwen3tts_spkenc::build` — the
/// ECAPA-TDNN speaker encoder: TDNN stem, 3 SE-Res2Net blocks, multi-layer
/// feature aggregation, attentive statistics pooling and a final FC.
fn build_qwen3tts_spkenc_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
) -> Result<(Graph, AudioGraphNodes), String> {
    const SPK_RES2NET_SCALE: i64 = 8;
    const SPK_DILATIONS: [i64; 3] = [2, 3, 4];

    let mut g = Graph::new(8192);

    // inp_raw: [T, n_mel, 1, 1] (qwen3tts-spkenc.cpp:156-161)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");
    let inp = cx.reshape_2d(inp_raw, cx.ne(inp_raw)[0], cx.ne(inp_raw)[1]);
    let mel = {
        let t = cx.transpose(inp);
        cx.cont(t)
    };

    // frontend conv0 TDNN k=5 (qwen3tts-spkenc.cpp:163-166)
    let mut cur = spk_conv1d_same(
        cx,
        mel,
        model.conv1d_1_w.unwrap(),
        model.conv1d_1_b.unwrap(),
        1,
    );
    cur = cx.relu(cur);

    // 3 SE-Res2Net blocks (qwen3tts-spkenc.cpp:168-175)
    assert_eq!(model.layers.len(), 3);
    let mut blk_out = [cur; 3];
    for (il, layer) in model.layers.iter().enumerate() {
        // se_res2net_block (qwen3tts-spkenc.cpp:88-97)
        let residual = cur;
        let mut h = spk_conv1d_same(
            cx,
            cur,
            layer.conv_pw1_w.unwrap(),
            layer.conv_pw1_b.unwrap(),
            1,
        );
        h = cx.relu(h);
        h = spk_res2net(cx, h, layer, SPK_DILATIONS[il], SPK_RES2NET_SCALE);
        h = spk_conv1d_same(
            cx,
            h,
            layer.conv_pw2_w.unwrap(),
            layer.conv_pw2_b.unwrap(),
            1,
        );
        h = cx.relu(h);
        h = spk_se_block(cx, h, layer);
        cur = cx.add(h, residual);
        blk_out[il] = cur;
    }

    // multi-layer feature aggregation (qwen3tts-spkenc.cpp:177-182)
    let mut cat = cx.concat(blk_out[0], blk_out[1], 0);
    cat = cx.concat(cat, blk_out[2], 0);
    let mut mfa = spk_conv1d_same(
        cx,
        cat,
        model.spk_conv_out_w.unwrap(),
        model.spk_conv_out_b.unwrap(),
        1,
    );
    mfa = cx.relu(mfa);

    // attentive statistics pooling: [1536, T] -> [3072, 1]
    let stats = spk_asp(cx, mfa, model);

    // final FC k=1 (qwen3tts-spkenc.cpp:188-193)
    let mut emb = spk_conv1d_same(cx, stats, model.mm_fc_w.unwrap(), model.mm_fc_b.unwrap(), 1);
    emb = cx.reshape_1d(emb, cx.ne(emb)[0]);
    emb = cx.cont(emb);

    g.build_forward(cx, emb);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra: Vec::new(),
        },
    ))
}

/// attentive statistics pooling with the model weights passed explicitly
/// (the C helper reaches for model.spk_asp_* directly)
fn spk_asp(cx: &mut Context, x: TensorId, model: &ClipModel) -> TensorId {
    let t_len = cx.ne(x)[1];

    let x_t = {
        let t = cx.transpose(x);
        cx.cont(t)
    };
    let mean = cx.mean(x_t);
    let mean = {
        let t = cx.transpose(mean);
        cx.cont(t)
    };

    let mean_rep = cx.repeat(mean, x);
    let centered = cx.sub(x, mean_rep);
    let var_t = {
        let s = cx.sqr(centered);
        let t = cx.transpose(s);
        cx.cont(t)
    };
    let mut var = cx.mean(var_t);
    var = {
        let t = cx.transpose(var);
        cx.cont(t)
    };
    var = cx.scale_bias(var, 1.0, 1e-12);
    let std = cx.sqrt(var);

    let std_rep = cx.repeat(std, x);
    let mut cat = cx.concat(x, mean_rep, 0);
    cat = cx.concat(cat, std_rep, 0);

    // attention TDNN (3C -> attn_c) + ReLU, tanh, then 1x1 conv
    let mut a = spk_conv1d_same(
        cx,
        cat,
        model.spk_asp_tdnn_w.unwrap(),
        model.spk_asp_tdnn_b.unwrap(),
        1,
    );
    a = cx.relu(a);
    a = cx.tanh(a);
    a = spk_conv1d_same(
        cx,
        a,
        model.spk_asp_attn_w.unwrap(),
        model.spk_asp_attn_b.unwrap(),
        1,
    );

    // softmax over T
    let a_t = {
        let t = cx.transpose(a);
        cx.cont(t)
    };
    let w_t = cx.soft_max(a_t);
    let w = {
        let t = cx.transpose(w_t);
        cx.cont(t)
    };

    // weighted mean: sum(w * x) over T, times T
    let wx = cx.mul(w, x);
    let wx_t = {
        let t = cx.transpose(wx);
        cx.cont(t)
    };
    let mut w_mean = cx.mean(wx_t);
    w_mean = cx.scale(w_mean, t_len as f32);
    let w_mean = {
        let t = cx.transpose(w_mean);
        cx.cont(t)
    };

    // weighted std: sum(w * (x - w_mean)^2) over T
    let w_mean_rep = cx.repeat(w_mean, x);
    let dev = cx.sub(x, w_mean_rep);
    let dev2 = cx.sqr(dev);
    let w_var_in = cx.mul(w, dev2);
    let w_var_t = {
        let t = cx.transpose(w_var_in);
        cx.cont(t)
    };
    let mut w_var = cx.mean(w_var_t);
    w_var = cx.scale(w_var, t_len as f32);
    let w_var = {
        let t = cx.transpose(w_var);
        cx.cont(t)
    };
    let w_var = cx.scale_bias(w_var, 1.0, 1e-12);
    let w_std = cx.sqrt(w_var);

    cx.concat(w_mean, w_std, 0)
}

/// pockettts-seanet.cpp:14 `clip_graph_pockettts_seanet::conv1d` — a causal
/// T-first conv1d (left context from replicate padding on a cold start).
fn seanet_conv1d(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    stride: i64,
    dilation: i64,
    pad_replicate: bool,
) -> TensorId {
    let k_size = (cx.ne(w)[0] - 1) * dilation + 1;
    let p_total = k_size - stride;

    // trailing padding so the last frame is not dropped (pockettts-seanet.cpp:20-22)
    let n_frames = (cx.ne(x)[0] - k_size + p_total).div_euclid(stride)
        + i64::from((cx.ne(x)[0] - k_size + p_total).rem_euclid(stride) != 0);
    let ideal_len = n_frames * stride + k_size - p_total;
    let p_extra = ideal_len - cx.ne(x)[0];

    let mut x = x;
    if pad_replicate && p_total > 0 {
        // repeat the first frame instead of zero-padding (pockettts-seanet.cpp:31-36)
        let first = cx.view_2d(x, 1, cx.ne(x)[1], cx.nb(x)[1] as usize, 0);
        let left = cx.repeat_4d(first, p_total, cx.ne(x)[1], 1, 1);
        x = cx.concat(left, x, 0);
        x = cx.pad_ext(x, 0, p_extra as i32, 0, 0, 0, 0, 0, 0);
    } else {
        x = cx.pad_ext(x, p_total as i32, p_extra as i32, 0, 0, 0, 0, 0, 0);
    }

    let mut y = conv_1d_gen(cx, w, x, stride as i32, 0, dilation as i32);
    y = cx.reshape_2d(y, cx.ne(y)[0], cx.ne(y)[1]);
    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        y = cx.add(y, b2);
    }
    y
}

/// pockettts-seanet.cpp:102 `res_unit` — ELU, dilated conv, ELU, pointwise.
fn seanet_res_unit(cx: &mut Context, x: TensorId, st: &SeanetStage, dilation: i64) -> TensorId {
    let mut h = cx.elu(x);
    h = seanet_conv1d(
        cx,
        h,
        st.res_conv1_w.unwrap(),
        st.res_conv1_b,
        1,
        dilation,
        false,
    );
    h = cx.elu(h);
    h = seanet_conv1d(cx, h, st.res_conv2_w.unwrap(), st.res_conv2_b, 1, 1, false);
    cx.add(x, h)
}

/// pockettts-seanet.cpp:112 `encode` — the mimi encoder stack.
fn seanet_encode(cx: &mut Context, model: &ClipModel, x: TensorId) -> TensorId {
    let seanet = model.seanet.as_ref().unwrap();
    let mut cur = seanet_conv1d(
        cx,
        x,
        seanet.conv_in_w.unwrap(),
        seanet.conv_in_b,
        1,
        1,
        false,
    );

    for (i, st) in seanet.stages.iter().enumerate() {
        let stride = model.hparams.seanet_ratios[i] as i64;
        cur = seanet_res_unit(cx, cur, st, 1);
        cur = cx.elu(cur);
        cur = seanet_conv1d(
            cx,
            cur,
            st.scale_conv_w.unwrap(),
            st.scale_conv_b,
            stride,
            1,
            false,
        );
    }

    cur = cx.elu(cur);
    seanet_conv1d(
        cx,
        cur,
        seanet.conv_out_w.unwrap(),
        seanet.conv_out_b,
        1,
        1,
        false,
    )
}

/// models/pockettts-spkenc.cpp:39 `clip_graph_pockettts_spkenc::build` — the
/// mimi SEANet encoder + transformer + downsample + speaker projection.
fn build_pockettts_spkenc_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    let mut g = Graph::new(8192);

    // the waveform as a single-row "mel" (pockettts-spkenc.cpp:41-42)
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");
    let mut cur = cx.reshape_2d(inp_raw, cx.ne(inp_raw)[0], cx.ne(inp_raw)[1]);

    cur = seanet_encode(cx, model, cur);

    // [T, 512] -> [512, T] for the transformer
    cur = {
        let t = cx.transpose(cur);
        cx.cont(t)
    };

    let inp_pos = cx.new_tensor_1d(GgmlType::I32, cx.ne(cur)[1]);
    cx.set_name(inp_pos, "inp_pos");
    let kq_mask = cx.new_tensor_2d(GgmlType::F32, cx.ne(cur)[1], cx.ne(cur)[1]);
    cx.set_name(kq_mask, "kq_mask");

    // the mimi transformer layers (pockettts-spkenc.cpp:60-63)
    for layer in &model.layers {
        // tfm_layer_forward (pockettts-spkenc.cpp:7-37)
        let inp = cur;
        cur = build_norm(
            cx,
            cur,
            layer.ln_1_w,
            layer.ln_1_b,
            NormType::Normal,
            hp.eps,
        );

        let qcur = cx.mul_mat(layer.q_w.unwrap(), cur);
        let kcur = cx.mul_mat(layer.k_w.unwrap(), cur);
        let vcur = cx.mul_mat(layer.v_w.unwrap(), cur);

        let n_pos = cx.ne(cur)[1];
        let q3 = cx.reshape_3d(qcur, d_head, n_head, n_pos);
        let k3 = cx.reshape_3d(kcur, d_head, n_head, n_pos);
        let v3 = cx.reshape_3d(vcur, d_head, n_head, n_pos);

        let qr = cx.rope_ext(
            q3,
            inp_pos,
            None,
            d_head as i32,
            ggml::ops::GGML_ROPE_TYPE_NORMAL,
            0,
            10000.0, // hparams.rope_theta default
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        let kr = cx.rope_ext(
            k3,
            inp_pos,
            None,
            d_head as i32,
            ggml::ops::GGML_ROPE_TYPE_NORMAL,
            0,
            10000.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );

        cur = build_attn_ex(
            cx,
            layer.o_w,
            None,
            qr,
            kr,
            v3,
            Some(kq_mask),
            kq_scale,
            flash_attn,
        );
        cur = cx.mul(cur, layer.ls_1_w.unwrap());
        cur = cx.add(cur, inp);

        let inp2 = cur;
        cur = build_norm(
            cx,
            cur,
            layer.ln_2_w,
            layer.ln_2_b,
            NormType::Normal,
            hp.eps,
        );
        cur = build_ffn_impl(
            cx,
            cur,
            layer.ff_up_w.unwrap(),
            None,
            None,
            None,
            layer.ff_down_w,
            None,
            FfnOp::Gelu,
        );
        cur = cx.mul(cur, layer.ls_2_w.unwrap());
        cur = cx.add(cur, inp2);
    }

    // downsample to the model frame rate (pockettts-spkenc.cpp:65-68)
    cur = {
        let t = cx.transpose(cur);
        cx.cont(t)
    };
    cur = seanet_conv1d(
        cx,
        cur,
        model.pocket_downsample_w.unwrap(),
        None,
        hp.mimi_downsample as i64,
        1,
        true,
    );

    // voice latent -> backbone embd (pockettts-spkenc.cpp:70-73)
    cur = {
        let t = cx.transpose(cur);
        cx.cont(t)
    };
    cur = cx.mul_mat(model.spk_proj_w.unwrap(), cur);

    // set_inputs payloads (clip.cpp:4512-4535 set_pockettts_tfm_inputs)
    let mut extra: Vec<(&'static str, TensorId, ExtraInput)> = Vec::new();
    {
        let n_pos = cx.ne(inp_pos)[0] as usize;
        let positions: Vec<i32> = (0..n_pos as i32).collect();
        extra.push(("inp_pos", inp_pos, ExtraInput::I32(positions)));

        let max_pos =
            crate::mtmd_audio::POCKETTTS_MAX_SPK_SECONDS as i64 * hp.audio_sample_rate as i64 / 120;
        assert!(
            (n_pos as i64) <= max_pos,
            "pocket-tts speaker reference too long for a dense mask"
        );

        let context = hp.mimi_tfm_context as i64;
        let mut mask = vec![f32::NEG_INFINITY; n_pos * n_pos];
        for q in 0..n_pos as i64 {
            for k in 0..n_pos as i64 {
                let delta = q - k;
                if delta >= 0 && delta < context {
                    mask[(q * n_pos as i64 + k) as usize] = 0.0;
                }
            }
        }
        extra.push(("kq_mask", kq_mask, ExtraInput::F32(mask)));
    }

    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra,
        },
    ))
}

// ======================================================================
// audio round 5 — the TTS output generators (models/qwen3tts-gen.cpp,
// models/pockettts-gen.cpp, models/pockettts-seanet.cpp decode half).
// Owner: agent GEN5.
// ======================================================================

/// named graph inputs/outputs of one generator build (the C graphs address
/// them by name — set_inputs / the output collection in clip.cpp:5800-5870)
#[derive(Default)]
pub struct GenGraphIo {
    /// GEN_CODE's backbone hidden state (named "inp_raw" like the C)
    pub h_state: Option<TensorId>,
    pub inp_code0: Option<TensorId>,
    /// one uniform draw per acoustic codebook ("inp_rand_0..")
    pub inp_rands: Vec<TensorId>,
    /// pocket-tts GEN_CODE's gaussian noise ("inp_noise")
    pub inp_noise: Option<TensorId>,
    /// pocket-tts GEN_WAV's latent window ("inp_feats")
    pub inp_feats: Option<TensorId>,
    /// qwen3-tts GEN_WAV's code window ("inp_codes", [n_frames, n_codes])
    pub inp_codes: Option<TensorId>,
    /// GEN_WAV state inputs by slot name ("state_in_<name>")
    pub state_in: Vec<(String, TensorId)>,
    /// GEN_WAV state outputs by slot name ("state_out_<name>")
    pub state_out: Vec<(String, TensorId)>,
    /// this frame's sampled codes (I32 [1, n_codes])
    pub out_codes: Option<TensorId>,
    /// the feedback embedding for the backbone (F32 [n_embd, 1])
    pub out_embd: Option<TensorId>,
    /// pocket-tts end-of-speech score (scalar)
    pub out_eos_score: Option<TensorId>,
    /// pocket-tts continuous latents (F32 [n_latent, 1])
    pub out_feats: Option<TensorId>,
    /// the PCM samples (F32 [n_samples])
    pub out_audio: Option<TensorId>,
}

/// qwen3tts-gen.cpp:6 `code_gen::do_sampling` — on-graph top-k, top-p and a
/// uniform draw against the inverse CDF.
fn q3t_do_sampling(
    cx: &mut Context,
    logits: TensorId,
    inp_rand: TensorId,
    top_k: i32,
    top_p: f32,
) -> TensorId {
    let n = cx.ne(logits).iter().product::<i64>();
    let logits = cx.reshape_1d(logits, n);
    let n_vocab = cx.ne(logits)[0];

    // sort a's rows by idx (qwen3tts-gen.cpp:11-14)
    let sort_by = |cx: &mut Context, a: TensorId, idx: TensorId| -> TensorId {
        let a2d = cx.reshape_2d(a, 1, cx.ne(a)[0]);
        let r = cx.get_rows(a2d, idx);
        cx.reshape_1d(r, cx.ne(idx)[0])
    };

    let mut cur = logits;
    let mut candidates: Option<TensorId> = None; // maps row index back to vocab id

    if top_k > 0 && (top_k as i64) < n_vocab {
        let idx = cx.top_k(cur, top_k);
        candidates = Some(idx);
        cur = sort_by(cx, cur, idx);
    }

    if top_p < 1.0 {
        let sorted_idx = cx.argsort(cur, ggml::ops::GGML_SORT_ORDER_DESC);
        let sorted_logits = sort_by(cx, cur, sorted_idx);
        candidates = Some(match candidates {
            Some(c) => sort_by(cx, c, sorted_idx),
            None => sorted_idx,
        });

        let probs = cx.soft_max(sorted_logits);
        let cdf = cx.cumsum(probs);

        // keep_mask[i] = 1 once cdf[i] crosses top_p (qwen3tts-gen.cpp:34-39)
        let cdf_scaled = cx.scale_bias(cdf, -1.0, top_p);
        let keep_mask = cx.step(cdf_scaled);
        let idxf = cx.sum(keep_mask);
        let idxf = cx.clamp(idxf, 0.0, (cx.ne(keep_mask)[0] - 1) as f32);
        let ones = cx.scale_bias(idxf, 0.0, 1.0);

        // top-p must include the crossing element, so force it to 1
        let keep_mask_2d = cx.reshape_2d(keep_mask, 1, cx.ne(keep_mask)[0]);
        let idxf_i32 = cx.cast(idxf, GgmlType::I32);
        let keep_mask_2d = cx.set_rows(keep_mask_2d, ones, idxf_i32);
        // back to the flat width — the ORIGINAL keep_mask's ne[0] (the 2-D
        // view's ne[0] is 1); qwen3tts-gen.cpp:57 reads the pre-reshape
        // tensor, which stays in scope in the C. The pipeline-layer e2e
        // (tests/tts_pipeline_e2e.rs, top_p 0.9 + top_k < n_vocab) is the
        // first driver through this arm — the old test ran top_p 1.0.
        let n_flat = cx.ne(keep_mask)[0];
        let keep_mask = cx.reshape_1d(keep_mask_2d, n_flat);

        // log(1) = 0 (keep), log(0) = -inf (drop)
        let bias = cx.log(keep_mask);
        cur = cx.add(sorted_logits, bias);
    }

    // draw one token: find where the cdf crosses inp_rand
    let probs = cx.soft_max(cur);
    let cumsum = cx.cumsum(probs);

    let diff = cx.sub(cumsum, inp_rand);
    let cross_mask = cx.step(diff);
    let idxf = cx.sum(cross_mask);
    let neg = cx.scale_bias(idxf, -1.0, cx.ne(cross_mask)[0] as f32);
    let mut idx = cx.cast(neg, GgmlType::I32);

    if let Some(c) = candidates {
        let cand_2d = cx.reshape_2d(c, 1, cx.ne(c)[0]);
        idx = cx.get_rows(cand_2d, idx);
    }
    idx
}

/// qwen3tts-gen.cpp:71 `code_gen::cache_set` — append `value` as the last row
/// then gather the rows back into place.
fn q3t_cache_set(
    cx: &mut Context,
    cache: TensorId,
    row_idx: i64,
    value: TensorId,
) -> TensorId {
    let n_embd = cx.ne(cache)[0];
    let n_cache = cx.ne(cache)[1];
    assert!((0..n_cache).contains(&row_idx));

    let value_2d = cx.reshape_2d(value, n_embd, 1);
    let cache_ext = cx.concat(cache, value_2d, 1); // [n_embd, n_cache + 1]

    // gather indices [0..row_idx-1, n_cache, row_idx+1..n_cache-1] built via
    // concat — ggml_set_rows needs F32/F16 values, not an I32 index array
    let mut idx = q3t_const_i32(cx, cache, n_cache as f32);
    if row_idx > 0 {
        let arange = cx.arange(0.0, row_idx as f32, 1.0);
        let prefix = cx.cast(arange, GgmlType::I32);
        idx = cx.concat(prefix, idx, 0);
    }
    if row_idx < n_cache - 1 {
        let arange = cx.arange((row_idx + 1) as f32, n_cache as f32, 1.0);
        let suffix = cx.cast(arange, GgmlType::I32);
        idx = cx.concat(idx, suffix, 0);
    }
    cx.get_rows(cache_ext, idx)
}

/// qwen3tts-gen.cpp:98 `code_gen::const_i32` — a const i32 with no host
/// upload: view a tensor, zero it via scale, add value, cast to i32.
fn q3t_const_i32(cx: &mut Context, anchor: TensorId, value: f32) -> TensorId {
    let mut v = cx.view_1d(anchor, 1, 0);
    if cx.ty(v) != GgmlType::F32 {
        v = cx.cast(v, GgmlType::F32);
    }
    let scaled = cx.scale_bias(v, 0.0, value);
    cx.cast(scaled, GgmlType::I32)
}

/// qwen3tts-gen.cpp:107 `code_gen::causal_mask_row` — the keep-mask row for a
/// query at position pos over an n_kv_pad window.
fn q3t_causal_mask_row(cx: &mut Context, n_kv_pad: i64, pos: i64) -> TensorId {
    let t = cx.new_tensor_2d(GgmlType::F32, n_kv_pad, n_kv_pad);
    let ones = cx.fill(t, 1.0);
    let keep = cx.tri(ones, ggml::ops::GGML_TRI_TYPE_LOWER_DIAG);
    let row = cx.view_1d(keep, n_kv_pad, (pos as usize) * cx.nb(keep)[1] as usize);
    let mask = cx.log(row); // 0 = keep, -inf = masked
    cx.reshape_4d(mask, n_kv_pad, 1, 1, 1)
}

/// qwen3tts-gen.cpp:116 `code_gen::project_in` — talker -> predictor hidden
fn q3t_project_in(
    cx: &mut Context,
    model: &ClipModel,
    cur: TensorId,
) -> TensorId {
    let Some(w) = model.gen_code_proj_in_w else {
        return cur;
    };
    let cur = cx.mul_mat(w, cur);
    if let Some(b) = model.gen_code_proj_in_b {
        return cx.add(cur, b);
    }
    cur
}

/// qwen3tts-gen.cpp:128 `code_gen::layer_forward` — one predictor layer at
/// position pos; writes k/v into the cache row pos. Returns (new cur, new
/// k_cache, new v_cache).
#[allow(clippy::too_many_arguments)]
fn q3t_layer_forward(
    cx: &mut Context,
    model: &ClipModel,
    layer: &ClipLayer,
    cur: TensorId,
    inp_pos: TensorId,
    kq_mask: TensorId,
    k_cache_layer: TensorId,
    v_cache_layer: TensorId,
    n_kv_pad: i64,
    pos: i64,
    flash_attn: ClipFlashAttn,
) -> (TensorId, TensorId, TensorId) {
    let hp = &model.hparams;
    let n_head = hp.n_head as i64;
    let n_head_kv = hp.n_head_kv as i64;
    // real head_dim, not n_embd / n_head
    let d_head = cx.ne(layer.q_w.unwrap())[1] / n_head;
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    let residual = cur;

    let h = cx.rms_norm(cur, hp.eps);
    let h = cx.mul(h, layer.ln_1_w.unwrap());

    let q = cx.mul_mat(layer.q_w.unwrap(), h);
    let k = cx.mul_mat(layer.k_w.unwrap(), h);
    let v = cx.mul_mat(layer.v_w.unwrap(), h);

    let q = cx.reshape_3d(q, d_head, n_head, 1);
    let k = cx.reshape_3d(k, d_head, n_head_kv, 1);

    // qk-norm (qwen3tts-gen.cpp:155-158)
    let q = cx.rms_norm(q, hp.eps);
    let q = cx.mul(q, layer.q_norm.unwrap());
    let k = cx.rms_norm(k, hp.eps);
    let k = cx.mul(k, layer.k_norm.unwrap());

    let q = cx.rope_ext(
        q,
        inp_pos,
        None,
        d_head as i32,
        GGML_ROPE_TYPE_NEOX,
        0,
        hp.rope_theta,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );
    let k = cx.rope_ext(
        k,
        inp_pos,
        None,
        d_head as i32,
        GGML_ROPE_TYPE_NEOX,
        0,
        hp.rope_theta,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );

    // write k/v into the cache at row pos, flat layout (qwen3tts-gen.cpp:165-168)
    let k_flat = cx.reshape_1d(k, d_head * n_head_kv);
    let k_cache_layer = q3t_cache_set(cx, k_cache_layer, pos, k_flat);
    let v_cache_layer = q3t_cache_set(cx, v_cache_layer, pos, v);

    let q_cur = cx.reshape_4d(q, d_head, n_head, 1, 1);
    let k_cur = cx.reshape_4d(k_cache_layer, d_head, n_head_kv, n_kv_pad, 1);
    let v_cur = cx.reshape_4d(v_cache_layer, d_head, n_head_kv, n_kv_pad, 1);

    let attn_out = build_attn_ex(
        cx,
        layer.o_w,
        layer.o_b,
        q_cur,
        k_cur,
        v_cur,
        Some(kq_mask),
        kq_scale,
        flash_attn,
    );

    let cur = cx.add(residual, attn_out);

    let h2 = cx.rms_norm(cur, hp.eps);
    let h2 = cx.mul(h2, layer.ln_2_w.unwrap());

    let gate = cx.mul_mat(layer.ff_gate_w.unwrap(), h2);
    let up = cx.mul_mat(layer.ff_up_w.unwrap(), h2);
    let gu = cx.swiglu_split(gate, up);
    let down = cx.mul_mat(layer.ff_down_w.unwrap(), gu);

    (cx.add(cur, down), k_cache_layer, v_cache_layer)
}

/// qwen3tts-gen.cpp:191 `code_gen::prefill` — position 0 (hidden bridge) and
/// position 1 (embed(code0), sample with lm_head[0]).
#[allow(clippy::too_many_arguments)]
fn q3t_prefill(
    cx: &mut Context,
    model: &ClipModel,
    k_cache: &mut Vec<TensorId>,
    v_cache: &mut Vec<TensorId>,
    out_code_cache: TensorId,
    h_state: TensorId,
    code0_embd: TensorId,
    inp_rand0: TensorId,
    top_k: i32,
    top_p: f32,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    let n_kv_pad = cx.ne(k_cache[0])[1];
    {
        let mut cur = q3t_project_in(cx, model, h_state);
        let kq_mask = q3t_causal_mask_row(cx, n_kv_pad, 0);
        let inp_pos = q3t_const_i32(cx, k_cache[0], 0.0);
        for (il, layer) in model.layers.iter().enumerate() {
            let (c, k, v) = q3t_layer_forward(
                cx,
                model,
                layer,
                cur,
                inp_pos,
                kq_mask,
                k_cache[il],
                v_cache[il],
                n_kv_pad,
                0,
                flash_attn,
            );
            cur = c;
            k_cache[il] = k;
            v_cache[il] = v;
        }
        // position 0's output is unused, it only seeded the cache
    }
    {
        let mut cur = q3t_project_in(cx, model, code0_embd);
        let kq_mask = q3t_causal_mask_row(cx, n_kv_pad, 1);
        let inp_pos = q3t_const_i32(cx, k_cache[0], 1.0);
        for (il, layer) in model.layers.iter().enumerate() {
            let (c, k, v) = q3t_layer_forward(
                cx,
                model,
                layer,
                cur,
                inp_pos,
                kq_mask,
                k_cache[il],
                v_cache[il],
                n_kv_pad,
                1,
                flash_attn,
            );
            cur = c;
            k_cache[il] = k;
            v_cache[il] = v;
        }

        let cur = cx.rms_norm(cur, model.hparams.eps);
        let cur = cx.mul(cur, model.gen_code_norm_w.unwrap());

        let head_w = model.gen_code_head_w.unwrap();
        // lm_head[0]
        let head_g = cx.view_2d(head_w, cx.ne(head_w)[0], cx.ne(head_w)[1], cx.nb(head_w)[1] as usize, 0);
        let logits = cx.mul_mat(head_g, cur);

        let sampled = q3t_do_sampling(cx, logits, inp_rand0, top_k, top_p);
        // out_code_cache = cache_set(out_code_cache, 1, sampled)
        // (qwen3tts-gen.cpp:226 — the prefill's second position writes row 1)
        q3t_cache_set(cx, out_code_cache, 1, sampled)
    }
}

/// qwen3tts-gen.cpp:236 `code_gen::step` — one decode step of the predictor.
#[allow(clippy::too_many_arguments)]
fn q3t_step(
    cx: &mut Context,
    model: &ClipModel,
    k_cache: &mut Vec<TensorId>,
    v_cache: &mut Vec<TensorId>,
    out_code_cache: TensorId,
    inp_rand: TensorId,
    step_idx: i64,
    top_k: i32,
    top_p: f32,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    let n_acoustic = cx.ne(model.gen_code_head_w.unwrap())[2];
    assert!((1..n_acoustic).contains(&step_idx));
    assert_eq!(k_cache.len(), model.layers.len());
    assert_eq!(v_cache.len(), model.layers.len());

    let n_kv_pad = cx.ne(k_cache[0])[1];
    let pos = step_idx + 1; // new cache row and RoPE position

    // embed the previous code via this step's codebook table
    let code_in = cx.view_1d(out_code_cache, 1, (step_idx as usize) * cx.nb(out_code_cache)[1] as usize);

    let embd_w = model.gen_code_embd_w.unwrap(); // [n_embd_talker, vocab, n_acoustic]
    let embd_g = cx.view_2d(
        embd_w,
        cx.ne(embd_w)[0],
        cx.ne(embd_w)[1],
        cx.nb(embd_w)[1] as usize,
        ((step_idx - 1) as usize) * cx.nb(embd_w)[2] as usize,
    );
    let cur = cx.get_rows(embd_g, code_in);
    let cur = cx.reshape_1d(cur, cx.ne(cur)[0]);

    let cur = q3t_project_in(cx, model, cur);

    let kq_mask = q3t_causal_mask_row(cx, n_kv_pad, pos);
    let inp_pos = q3t_const_i32(cx, k_cache[0], pos as f32);

    let mut cur = cur;
    for (il, layer) in model.layers.iter().enumerate() {
        let (c, k, v) = q3t_layer_forward(
            cx,
            model,
            layer,
            cur,
            inp_pos,
            kq_mask,
            k_cache[il],
            v_cache[il],
            n_kv_pad,
            pos,
            flash_attn,
        );
        cur = c;
        k_cache[il] = k;
        v_cache[il] = v;
    }

    // final norm, this step's lm_head, sample, write the result
    let cur = cx.rms_norm(cur, model.hparams.eps);
    let cur = cx.mul(cur, model.gen_code_norm_w.unwrap());

    let head_w = model.gen_code_head_w.unwrap(); // [n_embd_pred, vocab, n_acoustic]
    let head_g = cx.view_2d(
        head_w,
        cx.ne(head_w)[0],
        cx.ne(head_w)[1],
        cx.nb(head_w)[1] as usize,
        (step_idx as usize) * cx.nb(head_w)[2] as usize,
    );
    let logits = cx.mul_mat(head_g, cur);

    let sampled = q3t_do_sampling(cx, logits, inp_rand, top_k, top_p);

    q3t_cache_set(cx, out_code_cache, pos, sampled)
}

// ---- qwen3tts code2wav (qwen3tts-gen.cpp:289-605) -------------------------

/// qwen3tts-gen.cpp:289 `code2wav::causal_conv1d` — stride 1, prepend the
/// persisted left-context instead of zero padding.
fn q3t_causal_conv1d(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    dilation: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let k = cx.ne(w)[0] as i64;
    let pad = (k - 1) * dilation;

    let mut x_full = x;
    if pad > 0 {
        let left = state_in(state_name); // [pad, IC]
        x_full = cx.concat(left, x, 0);
    }
    let mut y = conv_1d_gen(cx, w, x_full, 1, 0, dilation as i32); // [T, OC, 1]
    y = cx.reshape_2d(y, cx.ne(y)[0], cx.ne(y)[1]);
    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        y = cx.add(y, b2);
    }
    if pad > 0 {
        let new_left = cx.view_2d(
            x_full,
            pad,
            cx.ne(x_full)[1],
            cx.nb(x_full)[1] as usize,
            ((cx.ne(x_full)[0] - pad) as usize) * cx.nb(x_full)[0] as usize,
        );
        let new_left = cx.cont(new_left);
        state_out.push((state_name.to_string(), new_left));
    }
    y
}

/// qwen3tts-gen.cpp:313 `code2wav::causal_conv1d_dw` — depthwise variant.
fn q3t_causal_conv1d_dw(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let k = cx.ne(w)[0];
    let pad = k - 1;

    let mut x_full = x;
    if pad > 0 {
        let left = state_in(state_name); // [pad, C]
        x_full = cx.concat(left, x, 0);
    }
    // ggml_conv_1d_dw (ggml.c:4648): reshape + im2col(depthwise) + mul_mat
    let b4 = cx.reshape_4d(x_full, cx.ne(x_full)[0], 1, cx.ne(x_full)[1], cx.ne(x_full)[2]);
    let dst_type = if cx.ty(w) == GgmlType::Bf16 {
        GgmlType::F32
    } else {
        GgmlType::F16
    };
    let im2col = cx.im2col(w, b4, 1, 0, 0, 0, 1, 0, false, dst_type);
    let mut y = cx.mul_mat(im2col, w);
    y = cx.reshape_3d(y, cx.ne(y)[0], cx.ne(y)[2], 1);
    y = cx.reshape_2d(y, cx.ne(y)[0], cx.ne(y)[1]);
    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        y = cx.add(y, b2);
    }
    if pad > 0 {
        let new_left = cx.view_2d(
            x_full,
            pad,
            cx.ne(x_full)[1],
            cx.nb(x_full)[1] as usize,
            ((cx.ne(x_full)[0] - pad) as usize) * cx.nb(x_full)[0] as usize,
        );
        let new_left = cx.cont(new_left);
        state_out.push((state_name.to_string(), new_left));
    }
    y
}

/// qwen3tts-gen.cpp:337 `code2wav::causal_conv_transpose1d` — the
/// (kernel - stride) overlap tail is kept as state for the next call.
fn q3t_causal_conv_transpose1d(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    stride: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let k = cx.ne(w)[0];
    let oc = cx.ne(w)[1];
    let trim = k - stride;
    let emit_len = cx.ne(x)[0] * stride;

    // transposed conv as GEMM + col2im scatter-add (qwen3tts-gen.cpp:343-348)
    let w2 = cx.reshape_2d(w, k * oc, cx.ne(w)[2]);
    let w2t = cx.transpose(w2);
    let w2 = cx.cont(w2t);
    let xt = cx.transpose(x);
    let xt = cx.cont(xt);
    let col = cx.mul_mat(w2, xt);
    let y = cx.col2im_1d(col, stride as i32, oc as i32, 0);

    let mut out = y;
    if trim > 0 {
        let tail = state_in(state_name); // [trim, OC]
        let head = cx.view_2d(y, trim, cx.ne(y)[1], cx.nb(y)[1] as usize, 0);
        let head = cx.add(head, tail);
        if emit_len > trim {
            let middle = cx.view_2d(
                y,
                emit_len - trim,
                cx.ne(y)[1],
                cx.nb(y)[1] as usize,
                (trim as usize) * cx.nb(y)[0] as usize,
            );
            out = cx.concat(head, middle, 0);
        } else {
            out = head;
        }
        let new_tail = cx.view_2d(
            y,
            trim,
            cx.ne(y)[1],
            cx.nb(y)[1] as usize,
            (emit_len as usize) * cx.nb(y)[0] as usize,
        );
        let new_tail = cx.cont(new_tail);
        state_out.push((state_name.to_string(), new_tail));
    }
    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        out = cx.add(out, b2);
    }
    out
}

/// qwen3tts-gen.cpp:371 `code2wav::snake` — SnakeBeta activation with
/// alpha/inv_beta folded at conversion time.
fn q3t_snake(cx: &mut Context, x: TensorId, alpha: TensorId, beta: TensorId) -> TensorId {
    let a = cx.reshape_2d(alpha, 1, cx.ne(alpha)[0]);
    let b = cx.reshape_2d(beta, 1, cx.ne(beta)[0]);

    let xa = cx.mul(x, a);
    let s = cx.sin(xa);
    let s = cx.sqr(s);
    let s = cx.mul(s, b);
    cx.add(x, s)
}

/// qwen3tts-gen.cpp:387 `code2wav::quant_decode` — RVQ codebook decode.
fn q3t_quant_decode(cx: &mut Context, model: &ClipModel, inp_codes: TensorId) -> TensorId {
    let c2w = model.c2w.as_ref().unwrap();
    let t_len = cx.ne(inp_codes)[0];

    // ids for codebook group g over all T frames, [T] I32
    let group_ids = |cx: &mut Context, g: i64| -> TensorId {
        cx.view_1d(inp_codes, t_len, (g as usize) * cx.nb(inp_codes)[1] as usize)
    };

    let ids0 = group_ids(cx, 0);
    let sem = cx.get_rows(c2w.quant_first_cb_w.unwrap(), ids0); // [256, T]
    let sem_out = cx.mul_mat(c2w.quant_first_out_w.unwrap(), sem); // [512, T]

    let mut acc: Option<TensorId> = None;
    let n_acoustic = cx.ne(c2w.quant_rest_cb_w.unwrap())[2];
    for g in 1..=n_acoustic {
        let cb = c2w.quant_rest_cb_w.unwrap();
        let cb_g = cx.view_2d(
            cb,
            cx.ne(cb)[0],
            cx.ne(cb)[1],
            cx.nb(cb)[1] as usize,
            ((g - 1) as usize) * cx.nb(cb)[2] as usize,
        );
        let ids_g = group_ids(cx, g);
        let embd = cx.get_rows(cb_g, ids_g); // [256, T]
        acc = Some(match acc {
            Some(a) => cx.add(a, embd),
            None => embd,
        });
    }
    let ac_out = cx.mul_mat(c2w.quant_rest_out_w.unwrap(), acc.unwrap()); // [512, T]

    cx.add(sem_out, ac_out)
}

/// qwen3tts-gen.cpp:417 `code2wav::tfm_layer_forward` — one pre_transformer
/// layer over a batch of N = sliding_window new frames.
fn q3t_tfm_layer_forward(
    cx: &mut Context,
    model: &ClipModel,
    layer: &ClipLayer,
    cur: TensorId,
    il: usize,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    let hp = &model.hparams;
    let n_head = hp.wav_tfm_n_head as i64;
    let n_head_kv = hp.wav_tfm_n_head_kv as i64;
    let d_head = cx.ne(layer.q_w.unwrap())[1] / n_head;
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();
    let w = hp.wav_tfm_swa as i64; // == N, frames per batch
    let n = cx.ne(cur)[1];
    let prefix = w - 1;
    let total_kv = prefix + n;

    let residual = cur;
    let h = cx.rms_norm(cur, hp.wav_tfm_eps);
    let h = cx.mul(h, layer.ln_1_w.unwrap());

    let q = cx.mul_mat(layer.q_w.unwrap(), h); // [n_head*d_head, N]
    let k = cx.mul_mat(layer.k_w.unwrap(), h);
    let v = cx.mul_mat(layer.v_w.unwrap(), h);

    let q = cx.reshape_3d(q, d_head, n_head, n);
    let k = cx.reshape_3d(k, d_head, n_head_kv, n);

    // real, ever-increasing positions: base (persisted) .. base+N-1
    let base = cx.reshape_1d(state_in("tfm_pos"), 1);
    let offset = cx.arange(0.0, n as f32, 1.0);
    let pos = cx.add(offset, base);
    let pos = cx.cast(pos, GgmlType::I32);

    let q = cx.rope_ext(
        q,
        pos,
        None,
        d_head as i32,
        GGML_ROPE_TYPE_NEOX,
        0,
        hp.wav_tfm_rope_theta,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );
    let k = cx.rope_ext(
        k,
        pos,
        None,
        d_head as i32,
        GGML_ROPE_TYPE_NEOX,
        0,
        hp.wav_tfm_rope_theta,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    );

    // the position counter is the same for all layers, push it once from
    // layer 0 (qwen3tts-gen.cpp:449-451)
    if il == 0 {
        let t = state_in("tfm_pos");
        let o = cx.scale_bias(t, 1.0, n as f32);
        state_out.push(("tfm_pos".to_string(), o));
    }

    let k_new = cx.reshape_2d(k, d_head * n_head_kv, n);
    let v_new = cx.reshape_2d(v, d_head * n_head_kv, n);

    let old_k = state_in(&format!("tfm_k_{il}")); // [d_head*n_head_kv, W-1]
    let old_v = state_in(&format!("tfm_v_{il}"));

    let k_full = cx.concat(old_k, k_new, 1); // [.., prefix+N]
    let v_full = cx.concat(old_v, v_new, 1);

    // next batch's prefix: the last (W-1) frames of this batch
    let nk = cx.view_2d(
        k_full,
        cx.ne(k_full)[0],
        prefix,
        cx.nb(k_full)[1] as usize,
        (n as usize) * cx.nb(k_full)[1] as usize,
    );
    let nk = cx.cont(nk);
    state_out.push((format!("tfm_k_{il}"), nk));
    let nv = cx.view_2d(
        v_full,
        cx.ne(v_full)[0],
        prefix,
        cx.nb(v_full)[1] as usize,
        (n as usize) * cx.nb(v_full)[1] as usize,
    );
    let nv = cx.cont(nv);
    state_out.push((format!("tfm_v_{il}"), nv));

    // banded causal mask: key j is visible to query i iff 0 <= (prefix+i)-j < W
    let pos_k = cx.arange(0.0, total_kv as f32, 1.0);
    let pos_k = cx.reshape_2d(pos_k, total_kv, 1);
    let pos_q = cx.arange(prefix as f32, (prefix + n) as f32, 1.0);
    let pos_q = cx.reshape_2d(pos_q, 1, n);
    let pos_q_grid = cx.repeat_4d(pos_q, total_kv, n, 1, 1);
    let diff = cx.sub(pos_q_grid, pos_k); // [total_kv, N]

    let sb = cx.scale_bias(diff, 1.0, 0.5);
    let causal_keep = cx.step(sb); // diff >= 0
    let sb = cx.scale_bias(diff, -1.0, w as f32 - 0.5);
    let in_window = cx.step(sb); // diff < W
    let keep = cx.mul(causal_keep, in_window);

    // on a cold start, key j is real state only when j >= prefix - tfm_pos
    let pk = cx.add(pos_k, base);
    let sb = cx.scale_bias(pk, 1.0, 0.5 - prefix as f32);
    let warm = cx.step(sb);
    let keep = cx.mul(keep, warm);

    let mask = cx.log(keep);
    let mask = cx.reshape_4d(mask, total_kv, n, 1, 1); // 0 = keep, -inf = masked

    let q_cur = cx.reshape_4d(q, d_head, n_head, n, 1);
    let k_cur = cx.reshape_4d(k_full, d_head, n_head_kv, total_kv, 1);
    let v_cur = cx.reshape_4d(v_full, d_head, n_head_kv, total_kv, 1);

    let mut attn_out = build_attn_ex(
        cx,
        layer.o_w,
        layer.o_b,
        q_cur,
        k_cur,
        v_cur,
        Some(mask),
        kq_scale,
        flash_attn,
    );
    if let Some(ls) = layer.ls_1_w {
        attn_out = cx.mul(attn_out, ls);
    }
    let cur = cx.add(residual, attn_out);

    let residual2 = cur;
    let h2 = cx.rms_norm(cur, hp.wav_tfm_eps);
    let h2 = cx.mul(h2, layer.ln_2_w.unwrap());

    let gate = cx.mul_mat(layer.ff_gate_w.unwrap(), h2);
    let up = cx.mul_mat(layer.ff_up_w.unwrap(), h2);
    let gu = cx.swiglu_split(gate, up);
    let mut down = cx.mul_mat(layer.ff_down_w.unwrap(), gu);
    if let Some(ls) = layer.ls_2_w {
        down = cx.mul(down, ls);
    }
    cx.add(residual2, down)
}

/// qwen3tts-gen.cpp:511 `code2wav::convnext_block`.
fn q3t_convnext_block(
    cx: &mut Context,
    blk: &C2wUpsample,
    x: TensorId,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_prefix: &str,
) -> TensorId {
    let residual = x;

    // dwconv -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> layer scale -> res
    let h = q3t_causal_conv1d_dw(
        cx,
        x,
        blk.dwconv_w.unwrap(),
        blk.dwconv_b,
        state_in,
        state_out,
        &format!("{state_prefix}_dwconv"),
    );
    let hc = cx.transpose(h);
    let mut hc = cx.cont(hc); // [C, T]

    hc = cx.norm(hc, 1e-6);
    hc = cx.mul(hc, blk.norm_w.unwrap());
    hc = cx.add(hc, blk.norm_b.unwrap());

    // pw1 -> gelu -> pw2 in one helper call (build_ffn_impl's plain path)
    let g = build_ffn_plain(
        cx,
        hc,
        blk.pw1_w.unwrap(),
        blk.pw1_b,
        blk.pw2_w.unwrap(),
        blk.pw2_b,
        FfnOp::Gelu,
    );
    let g = cx.mul(g, blk.gamma.unwrap());

    let g_t = cx.transpose(g);
    let g_t = cx.cont(g_t); // back to [T, C]
    cx.add(residual, g_t)
}

/// qwen3tts-gen.cpp:534 `code2wav::dac_res_unit`.
fn q3t_dac_res_unit(
    cx: &mut Context,
    res: &C2wDacRes,
    x: TensorId,
    dilation: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let residual = x;
    let h = q3t_snake(cx, x, res.act1_alpha.unwrap(), res.act1_beta.unwrap());
    let h = q3t_causal_conv1d(cx, h, res.conv1_w.unwrap(), res.conv1_b, dilation, state_in, state_out, state_name);
    let h = q3t_snake(cx, h, res.act2_alpha.unwrap(), res.act2_beta.unwrap());
    // k=1, no left-context needed
    let h = q3t_causal_conv1d(cx, h, res.conv2_w.unwrap(), res.conv2_b, 1, state_in, state_out, "");
    cx.add(residual, h)
}

/// qwen3tts-gen.cpp:544 `code2wav::decode` — RVQ codes -> raw PCM for a batch
/// of N = sliding_window frames.
fn q3t_code2wav_decode(
    cx: &mut Context,
    model: &ClipModel,
    inp_codes: TensorId,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    flash_attn: ClipFlashAttn,
) -> TensorId {
    let c2w = model.c2w.as_ref().unwrap();
    let hp = &model.hparams;

    // 1. quantizer decode: N frames of 16 codes -> [512, N] (C-first)
    let hidden = q3t_quant_decode(cx, model, inp_codes);

    // 2. pre_conv: [512, N] -> T-first [N, 512] -> causal conv k=3 -> [N, 1024]
    let ht = cx.transpose(hidden);
    let mut x = cx.cont(ht);
    x = q3t_causal_conv1d(cx, x, c2w.pre_conv_w.unwrap(), c2w.pre_conv_b, 1, state_in, state_out, "pre_conv");

    // 3. pre_transformer: back to C-first [1024, N], project down, layers, up
    let ct = cx.transpose(x);
    let mut cur = cx.cont(ct);
    cur = cx.mul_mat(c2w.tfm_in_proj_w.unwrap(), cur);
    cur = cx.add(cur, c2w.tfm_in_proj_b.unwrap()); // [512 (tfm hidden), N]

    for il in 0..hp.wav_tfm_n_layer as usize {
        cur = q3t_tfm_layer_forward(
            cx,
            model,
            &c2w.tfm_layers[il],
            cur,
            il,
            state_in,
            state_out,
            flash_attn,
        );
    }

    cur = cx.rms_norm(cur, hp.wav_tfm_eps);
    cur = cx.mul(cur, c2w.tfm_output_norm_w.unwrap());
    cur = cx.mul_mat(c2w.tfm_out_proj_w.unwrap(), cur);
    cur = cx.add(cur, c2w.tfm_out_proj_b.unwrap()); // [1024, N]

    // 4. upsample: 2x (causal ConvTranspose1d stride 2 + ConvNeXt block);
    // kernel == stride here, so no overlap tail to persist
    let ct = cx.transpose(cur);
    let mut x = cx.cont(ct); // [N, 1024]
    for (il, up) in c2w.upsample.iter().enumerate() {
        x = q3t_causal_conv_transpose1d(cx, x, up.conv_w.unwrap(), up.conv_b, 2, state_in, state_out, "");
        x = q3t_convnext_block(cx, up, x, state_in, state_out, &format!("up{il}"));
    }

    // 5. DAC decoder: conv_pre -> 4 blocks (SnakeBeta -> ConvTranspose1d ->
    // 3 res units) -> conv_post
    const DAC_DILATIONS: [i64; 3] = [1, 3, 9];

    x = q3t_causal_conv1d(cx, x, c2w.dac_entry_w.unwrap(), c2w.dac_entry_b, 1, state_in, state_out, "dac_entry");

    for (il, blk) in c2w.dac.iter().enumerate() {
        // kernel == 2*stride for all 4 blocks
        let stride = cx.ne(blk.conv_w.unwrap())[0] / 2;
        let blk_name = format!("dac{il}");
        x = q3t_snake(cx, x, blk.snake_alpha.unwrap(), blk.snake_beta.unwrap());
        x = q3t_causal_conv_transpose1d(
            cx,
            x,
            blk.conv_w.unwrap(),
            blk.conv_b,
            stride,
            state_in,
            state_out,
            &format!("{blk_name}_tail"),
        );
        for (ir, res) in blk.res.iter().enumerate() {
            x = q3t_dac_res_unit(
                cx,
                res,
                x,
                DAC_DILATIONS[ir],
                state_in,
                state_out,
                &format!("{blk_name}_res{ir}"),
            );
        }
    }

    x = q3t_snake(
        cx,
        x,
        c2w.dac_post_snake_alpha.unwrap(),
        c2w.dac_post_snake_beta.unwrap(),
    );
    x = q3t_causal_conv1d(
        cx,
        x,
        c2w.dac_post_conv_w.unwrap(),
        c2w.dac_post_conv_b,
        1,
        state_in,
        state_out,
        "dac_post_conv",
    ); // [n_samples, 1]

    x = cx.clamp(x, -1.0, 1.0);
    cx.reshape_1d(x, cx.ne(x)[0])
}

/// qwen3tts-gen.cpp:609 `list_c2w_state_slots` — the GEN_WAV persisted state
/// buffers (shape lookup only).
pub fn list_c2w_state_slots(model: &ClipModel, cx: &Context) -> Vec<C2wStateSlot> {
    let mut slots = Vec::new();
    let Some(c2w) = model.c2w.as_ref() else {
        return slots; // not a code2wav model
    };
    let hp = &model.hparams;

    slots.push(C2wStateSlot { name: "tfm_pos".into(), ne0: 1, ne1: 1 });

    // prefix is (W-1) frames, the batch itself gives the other N=W frames
    let d_head = cx.ne(c2w.tfm_layers[0].q_w.unwrap())[0] / hp.wav_tfm_n_head as i64;
    let kv_ch = d_head * hp.wav_tfm_n_head_kv as i64;
    let prefix = hp.wav_tfm_swa as i64 - 1;
    for il in 0..hp.wav_tfm_n_layer {
        slots.push(C2wStateSlot { name: format!("tfm_k_{il}"), ne0: kv_ch, ne1: prefix });
        slots.push(C2wStateSlot { name: format!("tfm_v_{il}"), ne0: kv_ch, ne1: prefix });
    }

    let ne = |t: Option<TensorId>| -> [i64; 4] { t.map(|t| *cx.ne(t)).unwrap_or([0, 0, 0, 0]) };

    slots.push(C2wStateSlot {
        name: "pre_conv".into(),
        ne0: ne(c2w.pre_conv_w)[0] - 1,
        ne1: ne(c2w.pre_conv_w)[1],
    });

    for (il, up) in c2w.upsample.iter().enumerate() {
        slots.push(C2wStateSlot {
            name: format!("up{il}_dwconv"),
            ne0: ne(up.dwconv_w)[0] - 1,
            ne1: ne(up.dwconv_w)[2],
        });
    }

    slots.push(C2wStateSlot {
        name: "dac_entry".into(),
        ne0: ne(c2w.dac_entry_w)[0] - 1,
        ne1: ne(c2w.dac_entry_w)[1],
    });

    const DAC_DILATIONS: [i64; 3] = [1, 3, 9];
    for (il, blk) in c2w.dac.iter().enumerate() {
        let stride = ne(blk.conv_w)[0] / 2; // kernel == 2*stride for all 4 blocks
        let blk_name = format!("dac{il}");
        slots.push(C2wStateSlot {
            name: format!("{blk_name}_tail"),
            ne0: stride,
            ne1: ne(blk.conv_w)[1],
        });
        for (ir, res) in blk.res.iter().enumerate() {
            slots.push(C2wStateSlot {
                name: format!("{blk_name}_res{ir}"),
                ne0: (ne(res.conv1_w)[0] - 1) * DAC_DILATIONS[ir],
                ne1: ne(res.conv1_w)[1],
            });
        }
    }

    slots.push(C2wStateSlot {
        name: "dac_post_conv".into(),
        ne0: ne(c2w.dac_post_conv_w)[0] - 1,
        ne1: ne(c2w.dac_post_conv_w)[1],
    });

    slots
}

/// models/qwen3tts-gen.cpp:657 `clip_graph_qwen3tts_gen::build` — both
/// sub-graphs are built, `gen_process` selects which runs (the port selects
/// via the outputs it expands into the Graph, same ancestor set as the C's
/// `ggml_build_forward_select`).
pub fn build_qwen3tts_gen_graph(
    cx: &mut Context,
    model: &ClipModel,
    gen_process: ClipGenProcess,
    top_k: i32,
    top_p: f32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, GenGraphIo), String> {
    let mut g = Graph::new(65536);
    let mut io = GenGraphIo::default();

    // ---- GEN_CODE: backbone hidden state -> 16 RVQ codes + next-step embd --
    // not build_inp_raw(): a GEN_WAV call's `img` has no hidden-state data
    let n_mmproj_embd = cx.ne(model.gen_code_out_embd_w.unwrap())[0];
    let h_state = cx.new_tensor_1d(GgmlType::F32, n_mmproj_embd);
    cx.set_name(h_state, "inp_raw");
    io.h_state = Some(h_state);

    let code0 = cx.new_tensor_1d(GgmlType::I32, 1);
    cx.set_name(code0, "inp_code0");
    io.inp_code0 = Some(code0);

    let code0_embd = cx.get_rows(model.gen_code_out_embd_w.unwrap(), code0);
    let code0_embd = cx.reshape_1d(code0_embd, cx.ne(code0_embd)[0]);

    let n_acoustic = cx.ne(model.gen_code_head_w.unwrap())[2]; // 15
    let n_codes = n_acoustic + 1; // 16
    let n_kv_pad = n_codes;
    let n_head = model.hparams.n_head as i64;
    let n_head_kv = model.hparams.n_head_kv as i64;
    let d_head = cx.ne(model.layers[0].q_w.unwrap())[1] / n_head;

    // zero-filled per layer k/v caches, so masked-out rows can't hold garbage
    let mut k_cache = Vec::with_capacity(model.layers.len());
    let mut v_cache = Vec::with_capacity(model.layers.len());
    for _ in 0..model.layers.len() {
        let kt = cx.new_tensor_2d(GgmlType::F32, d_head * n_head_kv, n_kv_pad);
        k_cache.push(cx.fill(kt, 0.0));
        let vt = cx.new_tensor_2d(GgmlType::F32, d_head * n_head_kv, n_kv_pad);
        v_cache.push(cx.fill(vt, 0.0));
    }

    let out_code_cache0 = cx.new_tensor_2d(GgmlType::I32, 1, n_codes);
    let mut out_code_cache = q3t_cache_set(cx, out_code_cache0, 0, code0);

    let mut inp_rands = Vec::new();
    let inp_rand0 = cx.new_tensor_1d(GgmlType::F32, 1);
    cx.set_name(inp_rand0, "inp_rand_0");
    inp_rands.push(inp_rand0);

    out_code_cache = q3t_prefill(
        cx,
        model,
        &mut k_cache,
        &mut v_cache,
        out_code_cache,
        h_state,
        code0_embd,
        inp_rand0,
        top_k,
        top_p,
        flash_attn,
    );

    for g_i in 1..n_acoustic {
        let inp_rand = cx.new_tensor_1d(GgmlType::F32, 1);
        cx.set_name(inp_rand, &format!("inp_rand_{g_i}"));
        inp_rands.push(inp_rand);
        out_code_cache = q3t_step(
            cx,
            model,
            &mut k_cache,
            &mut v_cache,
            out_code_cache,
            inp_rand,
            g_i,
            top_k,
            top_p,
            flash_attn,
        );
    }
    io.inp_rands = inp_rands;

    // output 1: this frame's 16 sampled codes, for the caller's code2wav
    let out_codes = cx.cont(out_code_cache);
    cx.set_name(out_codes, "out_codes");
    io.out_codes = Some(out_codes);

    // output 2: sum of all 16 codebook embeddings, fed back to the talker
    let mut out_embd = code0_embd;
    for g_i in 1..=n_acoustic {
        let code_g = cx.view_1d(
            out_code_cache,
            1,
            (g_i as usize) * cx.nb(out_code_cache)[1] as usize,
        );
        let embd_w = model.gen_code_embd_w.unwrap();
        let embd_g = cx.view_2d(
            embd_w,
            cx.ne(embd_w)[0],
            cx.ne(embd_w)[1],
            cx.nb(embd_w)[1] as usize,
            ((g_i - 1) as usize) * cx.nb(embd_w)[2] as usize,
        );
        let e = cx.get_rows(embd_g, code_g);
        let e = cx.reshape_1d(e, cx.ne(e)[0]);
        out_embd = cx.add(out_embd, e);
    }
    let out_embd = cx.reshape_2d(out_embd, cx.ne(out_embd)[0], 1);
    io.out_embd = Some(out_embd);

    // ---- GEN_WAV: 16 RVQ codes -> raw PCM ----------------------------------
    let n_frames = model.hparams.wav_tfm_swa as i64; // frames per batch

    let inp_codes = cx.new_tensor_2d(GgmlType::I32, n_frames, n_codes);
    cx.set_name(inp_codes, "inp_codes");
    io.inp_codes = Some(inp_codes);

    let mut state_out: Vec<(String, TensorId)> = Vec::new();
    // the state_in list keeps `list_c2w_state_slots` order — the driver packs
    // and unpacks the caller-facing blob in exactly that order (the C walks
    // the slot list on both sides, clip.cpp:4497 / :5856)
    let slots = list_c2w_state_slots(model, cx);
    let state_in_ids: std::collections::HashMap<String, TensorId> = slots
        .iter()
        .map(|slot| {
            let t = cx.new_tensor_2d(GgmlType::F32, slot.ne0, slot.ne1);
            cx.set_name(t, &format!("state_in_{}", slot.name));
            (slot.name.clone(), t)
        })
        .collect();
    io.state_in = slots
        .iter()
        .map(|slot| (slot.name.clone(), state_in_ids[&slot.name]))
        .collect();
    let lookup = move |name: &str| -> TensorId { state_in_ids[name] };

    let out_audio = q3t_code2wav_decode(cx, model, inp_codes, &lookup, &mut state_out, flash_attn);
    cx.set_name(out_audio, "out_audio");
    io.out_audio = Some(out_audio);

    for (name, t) in &state_out {
        cx.set_name(*t, &format!("state_out_{name}"));
    }
    io.state_out = state_out;

    // both sub-graphs are always built; the selection mirrors
    // ggml_build_forward_select's picks (qwen3tts-gen.cpp:759-767)
    match gen_process {
        ClipGenProcess::GenCode => {
            g.build_forward(cx, out_codes);
            g.build_forward(cx, out_embd);
        }
        ClipGenProcess::GenWav => {
            g.build_forward(cx, out_audio);
            for (_, t) in &io.state_out {
                g.build_forward(cx, *t);
            }
        }
    }
    Ok((g, io))
}

// ---- pocket-tts generator (pockettts-gen.cpp) -----------------------------

/// pockettts-gen.cpp:12 `modulate` — x * (1 + scale) + shift.
fn pt_modulate(cx: &mut Context, x: TensorId, shift: TensorId, scale: TensorId) -> TensorId {
    let sc = cx.scale_bias(scale, 1.0, 1.0);
    let cur = cx.mul(x, sc);
    cx.add(cur, shift)
}

/// pockettts-gen.cpp:18 `time_embed` — the TimestepEmbedder.
fn pt_time_embed(cx: &mut Context, te: &FlowTimeEmbd, t: f32) -> TensorId {
    // t is a graph-build constant, so the cos/sin table is folded
    let args = cx.scale(te.freqs.unwrap(), t);
    let cos = cx.cos(args);
    let sin = cx.sin(args);
    let emb = cx.concat(cos, sin, 0);

    let cur = cx.mul_mat(te.up_w.unwrap(), emb);
    let cur = cx.add(cur, te.up_b.unwrap());
    let cur = cx.silu(cur);
    let cur = cx.mul_mat(te.down_w.unwrap(), cur);
    let mut cur = cx.add(cur, te.down_b.unwrap());

    // this "RMSNorm" divides by the unbiased variance, not the mean square,
    // and rescales the input, not the centered value (pockettts-gen.cpp:30-39)
    {
        let n = cx.ne(cur)[0];
        let mean = cx.mean(cur);
        let dev = cx.sub(cur, mean);
        let var = cx.sqr(dev);
        let var = cx.mean(var);
        let var = cx.scale_bias(var, n as f32 / (n - 1) as f32, 1e-5);
        let var = cx.sqrt(var);
        cur = cx.div(cur, var);
        cur = cx.mul(cur, te.norm.unwrap());
    }
    cur
}

/// pockettts-gen.cpp:45 `flow_forward` — one velocity evaluation v(cond,s,t,x).
fn pt_flow_forward(
    cx: &mut Context,
    model: &ClipModel,
    cond: TensorId,
    x: TensorId,
    s: f32,
    t: f32,
) -> TensorId {
    let flow = model.flow.as_ref().unwrap();

    let cur = cx.mul_mat(flow.input_proj_w.unwrap(), x);
    let cur = cx.add(cur, flow.input_proj_b.unwrap());

    // the two time conditions are averaged, then added to the projected state
    let ts0 = pt_time_embed(cx, &flow.time[0], s);
    let ts1 = pt_time_embed(cx, &flow.time[1], t);
    let ts = cx.add(ts0, ts1);
    let ts = cx.scale(ts, 1.0 / flow.time.len() as f32);

    let c = cx.mul_mat(flow.cond_embd_w.unwrap(), cond);
    let c = cx.add(c, flow.cond_embd_b.unwrap());

    let y = cx.add(ts, c);

    let n_ch = cx.ne(flow.blocks[0].norm_w.unwrap())[0];

    let mut cur = cur;
    for blk in &flow.blocks {
        // AdaLN modulation from silu(y)
        let sy = cx.silu(y);
        let modu = cx.mul_mat(blk.ada_w.unwrap(), sy);
        let modu = cx.add(modu, blk.ada_b.unwrap());

        let shift = cx.view_1d(modu, n_ch, 0);
        let scale = cx.view_1d(modu, n_ch, (n_ch as usize) * cx.nb(modu)[0] as usize);
        let gate = cx.view_1d(modu, n_ch, (2 * n_ch as usize) * cx.nb(modu)[0] as usize);

        let h = build_norm(cx, cur, blk.norm_w, blk.norm_b, NormType::Normal, 1e-6);
        let h = pt_modulate(cx, h, shift, scale);
        let h = cx.mul_mat(blk.up_w.unwrap(), h);
        let h = cx.add(h, blk.up_b.unwrap());
        let h = cx.silu(h);
        let h = cx.mul_mat(blk.down_w.unwrap(), h);
        let h = cx.add(h, blk.down_b.unwrap());

        let gh = cx.mul(gate, h);
        cur = cx.add(cur, gh);
    }

    // final layer: the norm has no weights, only the AdaLN modulation
    let sy = cx.silu(y);
    let modu = cx.mul_mat(flow.final_ada_w.unwrap(), sy);
    let modu = cx.add(modu, flow.final_ada_b.unwrap());

    let shift = cx.view_1d(modu, n_ch, 0);
    let scale = cx.view_1d(modu, n_ch, (n_ch as usize) * cx.nb(modu)[0] as usize);

    let cur = build_norm(cx, cur, None, None, NormType::Normal, 1e-6);
    let cur = pt_modulate(cx, cur, shift, scale);
    let cur = cx.mul_mat(flow.final_proj_w.unwrap(), cur);
    cx.add(cur, flow.final_proj_b.unwrap())
}

/// pockettts-gen.cpp:102 `list_pockettts_state_slots`.
pub fn list_pockettts_state_slots(model: &ClipModel, cx: &Context) -> Vec<C2wStateSlot> {
    let mut slots = Vec::new();
    if model.gen_upsample_w.is_none() {
        return slots; // not a pocket-tts decoder
    }
    let seanet = model.seanet.as_ref().unwrap();
    let hp = &model.hparams;

    slots.push(C2wStateSlot { name: "tfm_pos".into(), ne0: 1, ne1: 1 });

    let ne = |t: Option<TensorId>| -> [i64; 4] { t.map(|t| *cx.ne(t)).unwrap_or([0, 0, 0, 0]) };

    let n_embd_a = ne(model.gen_tfm_layers[0].q_w)[1];
    let prefix = hp.mimi_tfm_context as i64 - 1;
    for il in 0..model.gen_tfm_layers.len() {
        slots.push(C2wStateSlot { name: format!("tfm_k_{il}"), ne0: n_embd_a, ne1: prefix });
        slots.push(C2wStateSlot { name: format!("tfm_v_{il}"), ne0: n_embd_a, ne1: prefix });
    }

    // upsample is depthwise, its output channel count is the input one
    slots.push(C2wStateSlot {
        name: "up".into(),
        ne0: ne(model.gen_upsample_w)[0] - hp.mimi_downsample as i64,
        ne1: ne(model.gen_upsample_w)[2],
    });

    slots.push(C2wStateSlot {
        name: "dec_in".into(),
        ne0: ne(seanet.conv_in_w)[0] - 1,
        ne1: ne(seanet.conv_in_w)[1],
    });
    for i in 0..hp.seanet_n_stage as usize {
        let stride = hp.seanet_ratios[hp.seanet_n_stage as usize - 1 - i] as i64;
        slots.push(C2wStateSlot {
            name: format!("dec_up_{i}"),
            ne0: ne(seanet.stages[i].scale_conv_w)[0] - stride,
            ne1: ne(seanet.stages[i].scale_conv_w)[1],
        });
        slots.push(C2wStateSlot {
            name: format!("dec_res_{i}"),
            ne0: ne(seanet.stages[i].res_conv1_w)[0] - 1,
            ne1: ne(seanet.stages[i].res_conv1_w)[1],
        });
    }
    slots.push(C2wStateSlot {
        name: "dec_out".into(),
        ne0: ne(seanet.conv_out_w)[0] - 1,
        ne1: ne(seanet.conv_out_w)[1],
    });

    slots
}

/// pockettts-seanet.cpp:14 `conv1d` — the stateful decode variant: left
/// context from the persisted state instead of replicate padding.
fn pt_seanet_conv1d_st(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    stride: i64,
    dilation: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let k_size = (cx.ne(w)[0] - 1) * dilation + 1;
    let p_total = k_size - stride;

    // trailing padding so the last frame is not dropped
    let rem = (cx.ne(x)[0] - k_size + p_total).rem_euclid(stride);
    let n_frames = (cx.ne(x)[0] - k_size + p_total).div_euclid(stride) + i64::from(rem != 0);
    let ideal_len = n_frames * stride + k_size - p_total;
    let p_extra = ideal_len - cx.ne(x)[0];

    let mut x = x;
    if !state_name.is_empty() && p_total > 0 {
        // streaming: the left context is the tail of the previous call
        let left = state_in(state_name); // [p_total, IC]
        x = cx.concat(left, x, 0);
        let tail = cx.view_2d(
            x,
            p_total,
            cx.ne(x)[1],
            cx.nb(x)[1] as usize,
            ((cx.ne(x)[0] - p_total) as usize) * cx.nb(x)[0] as usize,
        );
        let tail = cx.cont(tail);
        state_out.push((state_name.to_string(), tail));
    } else {
        x = cx.pad_ext(x, p_total as i32, p_extra as i32, 0, 0, 0, 0, 0, 0);
    }

    let mut y = conv_1d_gen(cx, w, x, stride as i32, 0, dilation as i32);
    y = cx.reshape_2d(y, cx.ne(y)[0], cx.ne(y)[1]);
    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        y = cx.add(y, b2);
    }
    y
}

/// pockettts-seanet.cpp:51 `conv_transpose1d` — the stateful decode variant
/// (overlap-add of the tail the previous call held back).
fn pt_seanet_conv_transpose1d_st(
    cx: &mut Context,
    x: TensorId,
    w: TensorId,
    b: Option<TensorId>,
    stride: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_name: &str,
) -> TensorId {
    let k = cx.ne(w)[0];
    let t_in = cx.ne(x)[0];
    let p_total = k - stride;
    let depthwise = cx.ne(w)[1] == 1 && cx.ne(w)[2] > 1;
    let oc = if depthwise { cx.ne(w)[2] } else { cx.ne(w)[1] };
    let emit_len = t_in * stride;

    // one column per input step, holding the [K, OC] window col2im
    // scatter-adds at t * stride (pockettts-seanet.cpp:62-74)
    let mut col;
    if depthwise {
        // one group per channel: a batched matmul over the channels scales
        // the kernel by each step
        let krn = cx.reshape_3d(w, 1, k, oc); // [1, K, OC]
        let xs = cx.reshape_3d(x, 1, t_in, oc); // [1, T, OC]
        col = cx.mul_mat(krn, xs); // [K, T, OC]
        col = cx.permute(col, 0, 2, 1, 3); // [K, OC, T]
        col = cx.cont(col);
        col = cx.reshape_2d(col, k * oc, t_in);
    } else {
        let w2 = cx.reshape_2d(w, k * oc, cx.ne(w)[2]);
        let w2t = cx.transpose(w2);
        let w2 = cx.cont(w2t); // [IC, K * OC]
        let xt = cx.transpose(x);
        let xt = cx.cont(xt); // [IC, T]
        col = cx.mul_mat(w2, xt);
    }
    let full = cx.col2im_1d(col, stride as i32, oc as i32, 0); // [emit_len + p_total, OC]

    let out;
    if state_name.is_empty() || p_total == 0 {
        let v = cx.view_2d(full, emit_len, cx.ne(full)[1], cx.nb(full)[1] as usize, 0);
        out = cx.cont(v);
    } else {
        // overlap-add the tail the previous call held back
        let prev = state_in(state_name); // [p_total, OC]
        let head_v = cx.view_2d(full, p_total, cx.ne(full)[1], cx.nb(full)[1] as usize, 0);
        let head = cx.add(head_v, prev);
        if emit_len > p_total {
            let rest = cx.view_2d(
                full,
                emit_len - p_total,
                cx.ne(full)[1],
                cx.nb(full)[1] as usize,
                (p_total as usize) * cx.nb(full)[0] as usize,
            );
            out = cx.concat(head, rest, 0);
        } else {
            out = head;
        }
        let tail = cx.view_2d(
            full,
            p_total,
            cx.ne(full)[1],
            cx.nb(full)[1] as usize,
            (emit_len as usize) * cx.nb(full)[0] as usize,
        );
        let tail = cx.cont(tail);
        state_out.push((state_name.to_string(), tail));
    }

    if let Some(b) = b {
        let b2 = cx.reshape_2d(b, 1, cx.ne(b)[0]);
        return cx.add(out, b2);
    }
    out
}

/// pockettts-seanet.cpp:102 `res_unit` — the stateful decode variant.
fn pt_seanet_res_unit_st(
    cx: &mut Context,
    model: &ClipModel,
    x: TensorId,
    stage_idx: usize,
    dilation: i64,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    state_prefix: &str,
) -> TensorId {
    let st = &model.seanet.as_ref().unwrap().stages[stage_idx];
    let mut h = cx.elu(x);
    h = pt_seanet_conv1d_st(
        cx,
        h,
        st.res_conv1_w.unwrap(),
        st.res_conv1_b,
        1,
        dilation,
        state_in,
        state_out,
        state_prefix,
    );
    h = cx.elu(h);
    // the second conv is pointwise, it needs no left context
    h = pt_seanet_conv1d_st(cx, h, st.res_conv2_w.unwrap(), st.res_conv2_b, 1, 1, state_in, state_out, "");
    cx.add(x, h)
}

/// pockettts-seanet.cpp:135 `decode` — the streaming mimi SEANet decoder.
/// `stream` is the C's `!state_in.empty()` — true exactly on the GEN_WAV
/// graphs (the driver builds the state map for those alone).
fn pt_seanet_decode(
    cx: &mut Context,
    model: &ClipModel,
    x: TensorId,
    state_in: &dyn Fn(&str) -> TensorId,
    state_out: &mut Vec<(String, TensorId)>,
    stream: bool,
) -> TensorId {
    let seanet = model.seanet.as_ref().unwrap();
    let hp = &model.hparams;

    let mut cur = pt_seanet_conv1d_st(
        cx,
        x,
        seanet.conv_in_w.unwrap(),
        seanet.conv_in_b,
        1,
        1,
        state_in,
        state_out,
        if stream { "dec_in" } else { "" },
    );

    for i in 0..hp.seanet_n_stage as usize {
        // the decoder mirrors the encoder, the ratios are walked backwards
        let stride = hp.seanet_ratios[hp.seanet_n_stage as usize - 1 - i] as i64;
        cur = cx.elu(cur);
        cur = pt_seanet_conv_transpose1d_st(
            cx,
            cur,
            seanet.stages[i].scale_conv_w.unwrap(),
            seanet.stages[i].scale_conv_b,
            stride,
            state_in,
            state_out,
            &if stream { format!("dec_up_{i}") } else { String::new() },
        );
        cur = pt_seanet_res_unit_st(
            cx,
            model,
            cur,
            i,
            1,
            state_in,
            state_out,
            &if stream { format!("dec_res_{i}") } else { String::new() },
        );
    }

    cur = cx.elu(cur);
    pt_seanet_conv1d_st(
        cx,
        cur,
        seanet.conv_out_w.unwrap(),
        seanet.conv_out_b,
        1,
        1,
        state_in,
        state_out,
        if stream { "dec_out" } else { "" },
    )
}

/// models/pockettts-gen.cpp:139 `clip_graph_pockettts_gen::build`.
pub fn build_pockettts_gen_graph(
    cx: &mut Context,
    model: &ClipModel,
    img_nx: i64,
    img_ny: i64,
    gen_process: ClipGenProcess,
    n_step: i32,
    n_frames: i64,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, GenGraphIo), String> {
    let mut g = Graph::new(32768);
    let mut io = GenGraphIo::default();
    let hp = &model.hparams;
    assert!(n_step > 0);

    if gen_process == ClipGenProcess::GenCode {
        // the backbone hidden state arrives as the single batch entry
        let h_state = cx.new_tensor_4d(GgmlType::F32, img_nx, img_ny, 1, 1);
        cx.set_name(h_state, "inp_raw");
        io.h_state = Some(h_state);
        let h_state = cx.reshape_2d(h_state, cx.ne(model.gen_input_lin_w.unwrap())[1], 1);

        // end-of-speech probe, thresholded on the host side
        let eos = cx.mul_mat(model.gen_out_eos_w.unwrap(), h_state);
        let eos = cx.add(eos, model.gen_out_eos_b.unwrap());
        cx.set_name(eos, "out_eos_score");
        io.out_eos_score = Some(eos);

        let n_latent = cx.ne(model.gen_input_lin_w.unwrap())[0];

        let noise = cx.new_tensor_2d(GgmlType::F32, n_latent, 1);
        cx.set_name(noise, "inp_noise");
        io.inp_noise = Some(noise);

        // lsd_decode: integrate the velocity field from the noise sample
        let mut cur = noise;
        for i in 0..n_step {
            let s = i as f32 / n_step as f32;
            let t = (i + 1) as f32 / n_step as f32;
            let v = pt_flow_forward(cx, model, h_state, cur, s, t);
            let dv = cx.scale(v, 1.0 / n_step as f32);
            cur = cx.add(cur, dv);
        }
        cx.set_name(cur, "out_feats");
        io.out_feats = Some(cur);

        // the same latent, projected into the backbone's input space
        let embd = cx.mul_mat(model.gen_input_lin_w.unwrap(), cur);
        io.out_embd = Some(embd);

        g.build_forward(cx, eos);
        g.build_forward(cx, cur);
        g.build_forward(cx, embd);
        return Ok((g, io));
    }

    // GEN_WAV: [n_latent, n_frames] latents -> PCM
    let n_latent = cx.ne(model.gen_input_lin_w.unwrap())[0];
    let feats = cx.new_tensor_2d(GgmlType::F32, n_latent, n_frames);
    cx.set_name(feats, "inp_feats");
    io.inp_feats = Some(feats);

    // denormalize, then the DummyQuantizer up-projection
    let cur = cx.mul(feats, model.gen_emb_std.unwrap());
    let cur = cx.add(cur, model.gen_emb_mean.unwrap());
    let mut cur = cx.mul_mat(model.gen_quant_out_w.unwrap(), cur);

    let mut state_out: Vec<(String, TensorId)> = Vec::new();
    // slot-list order, same as the c2w side (the driver's blob layout,
    // clip.cpp:4497 / :5856)
    let slots = list_pockettts_state_slots(model, cx);
    let state_in_ids: std::collections::HashMap<String, TensorId> = slots
        .iter()
        .map(|slot| {
            let t = cx.new_tensor_2d(GgmlType::F32, slot.ne0, slot.ne1);
            cx.set_name(t, &format!("state_in_{}", slot.name));
            (slot.name.clone(), t)
        })
        .collect();
    io.state_in = slots
        .iter()
        .map(|slot| (slot.name.clone(), state_in_ids[&slot.name]))
        .collect();
    let lookup = move |name: &str| -> TensorId { state_in_ids[name] };

    // model frame rate -> encoder frame rate, depthwise transposed conv
    let ct = cx.transpose(cur);
    let mut cur = cx.cont(ct);
    cur = pt_seanet_conv_transpose1d_st(
        cx,
        cur,
        model.gen_upsample_w.unwrap(),
        None,
        hp.mimi_downsample as i64,
        &lookup,
        &mut state_out,
        "up",
    );

    let ct = cx.transpose(cur);
    let mut cur = cx.cont(ct);

    // positions continue across calls, the counter lives in the state
    let n_pos = cx.ne(cur)[1];
    let prefix = hp.mimi_tfm_context as i64 - 1;
    let n_kv = prefix + n_pos;

    let base = cx.reshape_1d(lookup("tfm_pos"), 1);
    let ar = cx.arange(0.0, n_pos as f32, 1.0);
    let sum = cx.add(ar, base);
    let inp_pos = cx.cast(sum, GgmlType::I32);
    {
        let t = lookup("tfm_pos");
        let o = cx.scale_bias(t, 1.0, n_pos as f32);
        state_out.push(("tfm_pos".to_string(), o));
    }

    // banded causal mask over [cached prefix | this chunk]
    let pos_k = cx.arange(0.0, n_kv as f32, 1.0);
    let pos_k = cx.reshape_2d(pos_k, n_kv, 1);
    let pos_q = cx.arange(prefix as f32, (prefix + n_pos) as f32, 1.0);
    let pos_q = cx.reshape_2d(pos_q, 1, n_pos);
    let grid = cx.repeat_4d(pos_q, n_kv, n_pos, 1, 1);
    let diff = cx.sub(grid, pos_k);

    let sb = cx.scale_bias(diff, 1.0, 0.5);
    let m1 = cx.step(sb); // delta >= 0
    let sb = cx.scale_bias(diff, -1.0, hp.mimi_tfm_context as f32 - 0.5);
    let m2 = cx.step(sb); // delta < context
    let keep = cx.mul(m1, m2);
    // mask out cache rows that hold no real frame yet
    let pk = cx.add(pos_k, base);
    let sb = cx.scale_bias(pk, 1.0, 0.5 - prefix as f32);
    let warm = cx.step(sb);
    let keep = cx.mul(keep, warm);
    let kq_mask = cx.log(keep);
    let kq_mask = cx.reshape_4d(kq_mask, n_kv, n_pos, 1, 1);

    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    for (il, layer) in model.gen_tfm_layers.iter().enumerate() {
        let inp = cur;

        cur = build_norm(cx, cur, layer.ln_1_w, layer.ln_1_b, NormType::Normal, hp.eps);

        let qcur = cx.mul_mat(layer.q_w.unwrap(), cur);
        let kcur = cx.mul_mat(layer.k_w.unwrap(), cur);
        let vcur = cx.mul_mat(layer.v_w.unwrap(), cur);

        let q3 = cx.reshape_3d(qcur, d_head, n_head, n_pos);
        let k3 = cx.reshape_3d(kcur, d_head, n_head, n_pos);

        let qr = cx.rope_ext(
            q3,
            inp_pos,
            None,
            d_head as i32,
            ggml::ops::GGML_ROPE_TYPE_NORMAL,
            0,
            hp.rope_theta,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        let kr = cx.rope_ext(
            k3,
            inp_pos,
            None,
            d_head as i32,
            ggml::ops::GGML_ROPE_TYPE_NORMAL,
            0,
            hp.rope_theta,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );

        // prepend the cached window, then keep this chunk's tail
        let k_name = format!("tfm_k_{il}");
        let v_name = format!("tfm_v_{il}");
        let k2 = cx.reshape_2d(kr, d_head * n_head, n_pos);
        let k_full = cx.concat(lookup(&k_name), k2, 1);
        let v_full = cx.concat(lookup(&v_name), vcur, 1);
        {
            let nk = cx.view_2d(
                k_full,
                cx.ne(k_full)[0],
                prefix,
                cx.nb(k_full)[1] as usize,
                (n_pos as usize) * cx.nb(k_full)[1] as usize,
            );
            let nk = cx.cont(nk);
            state_out.push((k_name, nk));
            let nv = cx.view_2d(
                v_full,
                cx.ne(v_full)[0],
                prefix,
                cx.nb(v_full)[1] as usize,
                (n_pos as usize) * cx.nb(v_full)[1] as usize,
            );
            let nv = cx.cont(nv);
            state_out.push((v_name, nv));
        }

        let q_cur = cx.reshape_4d(qr, d_head, n_head, n_pos, 1);
        let k_cur = cx.reshape_4d(k_full, d_head, n_head, n_kv, 1);
        let v_cur = cx.reshape_4d(v_full, d_head, n_head, n_kv, 1);

        cur = build_attn_ex(
            cx,
            layer.o_w,
            None,
            q_cur,
            k_cur,
            v_cur,
            Some(kq_mask),
            kq_scale,
            flash_attn,
        );
        cur = cx.mul(cur, layer.ls_1_w.unwrap());
        cur = cx.add(cur, inp);

        let inp = cur;
        cur = build_norm(cx, cur, layer.ln_2_w, layer.ln_2_b, NormType::Normal, hp.eps);
        cur = build_ffn_plain(
            cx,
            cur,
            layer.ff_up_w.unwrap(),
            layer.ff_up_b,
            layer.ff_down_w.unwrap(),
            layer.ff_down_b,
            FfnOp::Gelu,
        );
        cur = cx.mul(cur, layer.ls_2_w.unwrap());
        cur = cx.add(cur, inp);
    }

    let ct = cx.transpose(cur);
    let mut cur = cx.cont(ct);
    cur = pt_seanet_decode(cx, model, cur, &lookup, &mut state_out, true);

    for (name, t) in &state_out {
        cx.set_name(*t, &format!("state_out_{name}"));
    }
    io.state_out = state_out;

    // [n_samples, 1] -> [n_samples], clamped like the reference output
    let cur = cx.reshape_1d(cur, cx.ne(cur)[0]);
    let cur = cx.clamp(cur, -1.0, 1.0);
    cx.set_name(cur, "out_audio");
    io.out_audio = Some(cur);

    g.build_forward(cx, cur);
    for (_, t) in &io.state_out {
        g.build_forward(cx, *t);
    }
    Ok((g, io))
}

/// models/whisper-enc.cpp:3 `clip_graph_whisper_enc::build` — conv1d stem,
/// build_vit transformer stack and the per-projector tail. `img` is the mel
/// chunk (nx = frames, ny = mel bins).
fn build_whisper_graph(
    cx: &mut Context,
    model: &ClipModel,
    img: &ClipImageF32,
    flash_attn: ClipFlashAttn,
) -> Result<(Graph, AudioGraphNodes), String> {
    let hp = &model.hparams;
    // whisper-enc.cpp:4-6
    let n_frames = img.nx as i64;
    let n_pos = n_frames / 2;
    let pos_embd = model.position_embeddings.expect("checked at load");
    assert!(
        cx.ne(pos_embd)[1] >= n_pos,
        "position_embeddings too small for {n_frames} frames"
    );

    let n_embd = hp.n_embd as i64;
    let n_head = hp.n_head as i64;
    let n_head_kv = hp.n_head_kv as i64;
    // clip-graph.h:59 d_head
    let d_head = if hp.n_embd_head > 0 {
        hp.n_embd_head as i64
    } else {
        n_embd / n_head
    };
    // clip-graph.h:63 kq_scale
    let kq_scale = 1.0f32 / (d_head as f32).sqrt();

    let mut g = Graph::new(4096);

    // build_inp_raw(1) (whisper-enc.cpp:8, clip.cpp:588-594): the mel chunk is
    // a single-"channel" "image" of [frames, mel bins]
    let inp_raw = cx.new_tensor_4d(GgmlType::F32, img.nx as i64, img.ny as i64, 1, 1);
    cx.set_name(inp_raw, "inp_raw");

    // conv1d block (whisper-enc.cpp:11-25)
    let inp;
    {
        // convolution + gelu
        let mut cur = conv_1d_ph(cx, model.conv1d_1_w.unwrap(), inp_raw, 1, 1);
        cur = cx.add(cur, model.conv1d_1_b.unwrap());

        cur = cx.gelu_erf(cur);

        cur = conv_1d_ph(cx, model.conv1d_2_w.unwrap(), cur, 2, 1);
        cur = cx.add(cur, model.conv1d_2_b.unwrap());

        cur = cx.gelu_erf(cur);
        // transpose
        let t = cx.transpose(cur);
        inp = cx.cont(t);
    }

    // sanity check, only one layer (whisper-enc.cpp:27-32)
    let l0 = &model.layers[0];
    assert!(l0.ln_1_w.is_some() && l0.ln_1_b.is_some());
    assert!(l0.ln_2_w.is_some() && l0.ln_2_b.is_some());
    assert!(l0.q_b.is_some());
    assert!(l0.v_b.is_some());
    assert!(l0.k_b.is_none(), "no bias for k");

    // whisper-enc.cpp:34-38 — the first n_pos learned positions
    let pos_ne = *cx.ne(pos_embd);
    let pos_embd_selected = cx.view_2d(pos_embd, pos_ne[0], n_pos, cx.nb(pos_embd)[1] as usize, 0);

    // whisper-enc.cpp:39-44 — NORM_TYPE_NORMAL, hparams.ffn_op (gelu_erf),
    // no add_pos hook
    let mut cur = build_vit_audio(
        cx,
        model,
        inp,
        n_pos,
        NormType::Normal,
        hp.ffn_op,
        Some(pos_embd_selected),
        n_embd,
        n_head,
        n_head_kv,
        d_head,
        kq_scale,
        flash_attn,
    );

    // StackAudioFrames (whisper-enc.cpp:48-53; ultravox/voxtral/meralion)
    if model.proj_type.audio_has_stack_frames() {
        cur = build_stack(cx, cur, hp.proj_stack_factor, hp.n_embd);
    }

    // projector (whisper-enc.cpp:55-130)
    cur = match model.proj_type {
        ProjectorType::Ultravox => {
            // UltravoxProjector (whisper-enc.cpp:56-73)
            // pre-norm (rms 1e-6)
            cur = cx.rms_norm(cur, 1e-6);
            cur = cx.mul(cur, model.mm_norm_pre_w.unwrap());

            // ffn in
            cur = cx.mul_mat(model.mm_1_w.unwrap(), cur);

            // swiglu — the second half passed through is silu, not the first
            cur = swiglu_swapped(cx, cur);

            // mid-norm (rms 1e-6)
            cur = cx.rms_norm(cur, 1e-6);
            cur = cx.mul(cur, model.mm_norm_mid_w.unwrap());

            // ffn out
            cx.mul_mat(model.mm_2_w.unwrap(), cur)
        }
        ProjectorType::Qwen2A => {
            // projector (whisper-enc.cpp:76-78)
            let cur = cx.mul_mat(model.mm_fc_w.unwrap(), cur);
            cx.add(cur, model.mm_fc_b.unwrap())
        }
        ProjectorType::Voxtral | ProjectorType::MusicFlamingo => {
            // projector (whisper-enc.cpp:81-96): plain FFN with gelu_erf;
            // voxtral has no biases (clip.cpp:3136-3142), musicflamingo does
            build_ffn_plain(
                cx,
                cur,
                model.mm_1_w.unwrap(),
                model.mm_1_b,
                model.mm_2_w.unwrap(),
                model.mm_2_b,
                FfnOp::GeluErf,
            )
        }
        ProjectorType::Meralion => {
            // stack (above) -> ln -> linear0+silu -> GLU -> out
            // (whisper-enc.cpp:99-118)
            cur = cx.norm(cur, hp.eps);
            cur = cx.mul(cur, model.mm_norm_pre_w.unwrap());
            cur = cx.add(cur, model.mm_norm_pre_b.unwrap());

            cur = cx.mul_mat(model.mm_0_w.unwrap(), cur);
            cur = cx.add(cur, model.mm_0_b.unwrap());
            cur = cx.silu(cur);

            let mut gate = cx.mul_mat(model.mm_1_w.unwrap(), cur);
            gate = cx.add(gate, model.mm_1_b.unwrap());
            gate = cx.silu(gate);

            let mut pool = cx.mul_mat(model.mm_2_w.unwrap(), cur);
            pool = cx.add(pool, model.mm_2_b.unwrap());

            cur = cx.mul(gate, pool);

            cur = cx.mul_mat(model.mm_3_w.unwrap(), cur);
            cx.add(cur, model.mm_3_b.unwrap())
        }
        ProjectorType::Glma => {
            // whisper-enc.cpp:121-127
            cur = cx.norm(cur, hp.eps);
            cur = cx.mul(cur, model.mm_norm_pre_w.unwrap());
            cur = cx.add(cur, model.mm_norm_pre_b.unwrap());
            cur = build_stack(cx, cur, hp.proj_stack_factor, hp.n_embd);
            cur = build_ffn_plain(
                cx,
                cur,
                model.mm_1_w.unwrap(),
                model.mm_1_b,
                model.mm_2_w.unwrap(),
                model.mm_2_b,
                hp.ffn_op,
            );
            cur = cx.concat(model.mm_boi.unwrap(), cur, 1);
            cx.concat(cur, model.mm_eoi.unwrap(), 1)
        }
        other => {
            return Err(format!(
                "unknown projector type for the audio graph: {}",
                other.name()
            ))
        }
    };

    // whisper-enc.cpp:132-135
    g.build_forward(cx, cur);
    let out = *g.nodes.last().expect("non-empty graph");
    Ok((
        g,
        AudioGraphNodes {
            inp_raw,
            out,
            extra: Vec::new(),
        },
    ))
}

// ======================================================================
// preprocessing — mtmd-image.cpp (img_tool + the preprocessors)
// ======================================================================

/// mtmd-image.h:30 `mtmd_image_preprocessor` — the base class is flattened into
/// an enum: the port implements the families its loaders select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePreprocessorKind {
    /// mtmd-image.h:121 — resize to a multiple of patch*n_merge keeping the
    /// aspect ratio (qwen-vl, ours)
    DynSize,
    /// mtmd-image.h:113 — stretch to image_size x image_size (fixed-size models)
    FixedSize,
    /// mtmd-image.h:194 `mtmd_image_preprocessor_cohere2v` — stretch the image
    /// to a square-tile grid, add a thumbnail when there is more than 1 tile
    Cohere2Tiles,
}

/// mtmd-image.h:11 `mtmd_image_preproc_out` — the preprocessor result the
/// tokenizer assembles chunks from (entries + overview + the llava-uhd grid).
#[derive(Default)]
pub struct ImagePreprocOut {
    pub entries: Vec<ClipImageF32>,
    /// the overview (downscaled) image; `None` == `!has_overview()`
    pub overview: Option<ClipImageF32>,
    pub grid_x: i32,
    pub grid_y: i32,
}

impl ImagePreprocOut {
    pub fn has_overview(&self) -> bool {
        self.overview.as_ref().is_some_and(|o| o.nx > 0 || o.ny > 0)
    }
}

/// mtmd-image.h:64 `mtmd_image_preprocessor_llava_uhd::slice_coordinates`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SliceCoordinates {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

/// mtmd-image.h:70 `mtmd_image_preprocessor_llava_uhd::slice_instructions`
/// (the pinpoints/minicpm grid search of the base class stays unported —
/// cohere2v owns its `get_slice_instructions`, mtmd-image.cpp:1166)
#[derive(Default)]
struct SliceInstructions {
    overview_size: (i32, i32),
    refined_size: (i32, i32),
    grid_size: (i32, i32),
    slices: Vec<SliceCoordinates>,
}

/// mtmd-image.cpp:612 `slice_output`
#[derive(Default)]
struct SliceOutput {
    overview: ClipImageU8,
    slices: Vec<ClipImageU8>,
}

impl ClipContext {
    /// mtmd.cpp:696 — the preprocessor each projector type selects.
    pub fn image_preprocessor(&self) -> ImagePreprocessorKind {
        match self.model.proj_type {
            ProjectorType::Qwen2Vl
            | ProjectorType::Qwen25Vl
            | ProjectorType::Qwen3Vl
            | ProjectorType::Ling3Vl => {
                ImagePreprocessorKind::DynSize
            }
            // mtmd.cpp:801-810 — cohere2v picks the square-tile slicer
            ProjectorType::Cohere2V => ImagePreprocessorKind::Cohere2Tiles,
            // audio-only projectors (incl. the tts generators): no image
            // preprocessor is ever selected
            ProjectorType::Qwen3TtsGen | ProjectorType::PocketTtsGen => {
                ImagePreprocessorKind::FixedSize
            }
            // audio-only projectors: no image preprocessor is ever selected
            ProjectorType::Qwen2A
            | ProjectorType::Ultravox
            | ProjectorType::Voxtral
            | ProjectorType::Meralion
            | ProjectorType::Glma
            | ProjectorType::MusicFlamingo
            | ProjectorType::Qwen3A
            | ProjectorType::Gemma4UA
            | ProjectorType::Lfm2A
            | ProjectorType::D1OmniA
            | ProjectorType::Gemma4A
            | ProjectorType::GraniteSpeech
            | ProjectorType::Parakeet
            | ProjectorType::MimoAudio
            | ProjectorType::Qwen3TtsSpkEnc
            | ProjectorType::PocketTtsSpkEnc
            | ProjectorType::Unknown => ImagePreprocessorKind::FixedSize,
        }
    }

    /// mtmd-image.cpp:772 `mtmd_image_preprocessor_dyn_size::preprocess` /
    /// :756 `..._fixed_size::preprocess`. Returns the normalised image entries.
    pub fn image_preprocess(&self, img: &ClipImageU8) -> Result<Vec<ClipImageF32>, String> {
        let out = self.image_preprocess_full(img)?;
        Ok(out.entries)
    }

    /// the preprocessor entry of mtmd.cpp:1373-1390 — one `preprocess` call
    /// per bitmap, producing the entries / overview / grid the tokenizer
    /// assembles image chunks from.
    pub fn image_preprocess_full(&self, img: &ClipImageU8) -> Result<ImagePreprocOut, String> {
        let hp = &self.model.hparams;
        match self.image_preprocessor() {
            ImagePreprocessorKind::Cohere2Tiles => {
                // mtmd-image.cpp:1145-1163
                // `mtmd_image_preprocessor_cohere2v::preprocess`
                let inst = self.get_slice_instructions_cohere2v((img.nx, img.ny));
                let sliced = self.slice_image(img, &inst);

                let mut output = ImagePreprocOut::default();
                // mtmd_image_preproc_out::append_overview (mtmd-image.cpp:22)
                let mut ov = ClipImageF32::from_u8(&sliced.overview);
                ov.normalize(&hp.image_mean, &hp.image_std);
                if sliced.slices.is_empty() {
                    // no tiles: the overview alone is the entry list
                    output.overview = Some(ov);
                    return Ok(output);
                }
                // slices first, then thumbnail (ov_img_first == false,
                // mtmd.cpp:808)
                for s in &sliced.slices {
                    let mut e = ClipImageF32::from_u8(s);
                    e.normalize(&hp.image_mean, &hp.image_std);
                    output.entries.push(e);
                }
                output.overview = Some(ov);
                output.grid_x = inst.grid_size.0;
                output.grid_y = inst.grid_size.1;
                Ok(output)
            }
            kind => {
                let mut resized = ClipImageU8::default();
                match kind {
                    ImagePreprocessorKind::DynSize => {
                        assert!(hp.image_min_pixels > 0 && hp.image_max_pixels > 0);
                        let target = img_tool::calc_size_preserved_ratio(
                            (img.nx, img.ny),
                            hp.patch_size as i64 * hp.n_merge as i64,
                            hp.image_min_pixels as i64,
                            hp.image_max_pixels as i64,
                            0,
                        );
                        img_tool::resize(
                            img,
                            &mut resized,
                            target,
                            hp.image_resize_algo,
                            hp.image_resize_pad,
                            hp.image_pad_color,
                        );
                    }
                    ImagePreprocessorKind::FixedSize => {
                        let sz = hp.image_size;
                        img_tool::resize(
                            img,
                            &mut resized,
                            (sz, sz),
                            hp.image_resize_algo,
                            hp.image_resize_pad,
                            hp.image_pad_color,
                        );
                    }
                    ImagePreprocessorKind::Cohere2Tiles => unreachable!(),
                }
                let mut out = ClipImageF32::from_u8(&resized);
                out.normalize(&hp.image_mean, &hp.image_std);
                Ok(ImagePreprocOut {
                    entries: vec![out],
                    overview: None,
                    grid_x: 0,
                    grid_y: 0,
                })
            }
        }
    }

    /// mtmd-image.cpp:1166
    /// `mtmd_image_preprocessor_cohere2v::get_slice_instructions` — pick the
    /// grid with the least upscale; if all grids need downscale, pick the one
    /// with the least downscale. Grids are visited by tile count, then by
    /// width, same order as HF for ties.
    fn get_slice_instructions_cohere2v(&self, original_size: (i32, i32)) -> SliceInstructions {
        let hp = &self.model.hparams;
        let tile = hp.image_size;

        let mut best_down = -1.0f64;
        let mut best_up = f64::MAX;
        let mut grid_down = (1, 1);
        let mut grid_up = (0, 0);
        for n in 1..=hp.preproc_max_tiles {
            for w in 1..=n {
                if n % w != 0 {
                    continue;
                }
                let g = (w, n / w);
                let scale = f64::min(
                    (g.0 as f64 * tile as f64) / original_size.0 as f64,
                    (g.1 as f64 * tile as f64) / original_size.1 as f64,
                );
                if scale < 1.0 {
                    if scale > best_down {
                        best_down = scale;
                        grid_down = g;
                    }
                } else if scale < best_up {
                    best_up = scale;
                    grid_up = g;
                }
            }
        }
        let grid = if grid_up.0 > 0 { grid_up } else { grid_down };

        let mut inst = SliceInstructions {
            overview_size: (tile, tile),
            refined_size: (tile * grid.0, tile * grid.1),
            grid_size: grid,
            slices: Vec::new(),
        };
        if grid.0 * grid.1 > 1 {
            for y in 0..grid.1 {
                for x in 0..grid.0 {
                    inst.slices.push(SliceCoordinates {
                        x: x * tile,
                        y: y * tile,
                        w: tile,
                        h: tile,
                    });
                }
            }
        }
        inst
    }

    /// mtmd-image.cpp:613 `mtmd_image_preprocessor_llava_uhd::slice_image` —
    /// the overview resize, the refined (grid-sized) resize and the crops.
    fn slice_image(&self, img: &ClipImageU8, inst: &SliceInstructions) -> SliceOutput {
        let hp = &self.model.hparams;
        let mut output = SliceOutput::default();

        // resize to overview size (algo/pad/color *_ov, mtmd-image.cpp:619)
        img_tool::resize(
            img,
            &mut output.overview,
            inst.overview_size,
            hp.image_resize_algo_ov,
            hp.image_pad_ov,
            hp.image_pad_color_ov,
        );

        if inst.slices.is_empty() {
            // no slices, just return the overview image
            return output;
        }

        // resize to refined size (algo/pad/color *_rf, mtmd-image.cpp:630)
        let mut refined_img = ClipImageU8::default();
        img_tool::resize(
            img,
            &mut refined_img,
            inst.refined_size,
            hp.image_resize_algo_rf,
            hp.image_pad_rf,
            hp.image_pad_color_rf,
        );

        // create slices
        for slice in &inst.slices {
            let mut img_slice = ClipImageU8::default();
            img_tool::crop(
                &refined_img,
                &mut img_slice,
                slice.x,
                slice.y,
                slice.w,
                slice.h,
            );
            output.slices.push(img_slice);
        }
        output
    }
}

/// mtmd-image.cpp:38 `img_tool` — image manipulation helpers. `resize_pillow`
/// is a faithful port (fixed-point weights, 22 fractional bits) because the
/// resampled pixels decide the vision embeddings.
pub mod img_tool {
    use super::{ClipImageU8, PadStyle, ResizeAlgo};

    /// mtmd-image.cpp:112 `calc_size_opt`
    #[derive(Clone, Copy)]
    pub struct CalcSizeOpt {
        pub align_size: i64,
        pub min_pixels: i64,
        pub max_pixels: i64,
        pub longest_edge: i64,
    }

    /// mtmd-image.cpp:122 `calc_size_preserved_ratio` ("smart_resize")
    pub fn calc_size_preserved_ratio(
        inp: (i32, i32),
        align_size: i64,
        min_pixels: i64,
        max_pixels: i64,
        longest_edge: i64,
    ) -> (i32, i32) {
        let opt = CalcSizeOpt {
            align_size,
            min_pixels,
            max_pixels,
            longest_edge,
        };
        assert!(opt.align_size > 0);
        let (width, height) = (inp.0 as i64, inp.1 as i64);
        if width <= 0 || height <= 0 {
            return (0, 0);
        }
        let f = opt.align_size as f64;
        let round_by = |x: f64| ((x / f).round() * f) as i64;
        let ceil_by = |x: f64| ((x / f).ceil() * f) as i64;
        let floor_by = |x: f64| ((x / f).floor() * f) as i64;

        let (mut w_bar, mut h_bar);
        if opt.longest_edge > 0 {
            let scale = (opt.longest_edge as f64 / width as f64)
                .min(opt.longest_edge as f64 / height as f64);
            w_bar = ceil_by(width as f64 * scale);
            h_bar = ceil_by(height as f64 * scale);
        } else {
            w_bar = round_by(width as f64).max(opt.align_size);
            h_bar = round_by(height as f64).max(opt.align_size);
        }

        if opt.max_pixels > 0 && h_bar * w_bar > opt.max_pixels {
            let beta = ((height * width) as f64 / opt.max_pixels as f64).sqrt();
            h_bar = floor_by(height as f64 / beta).max(opt.align_size);
            w_bar = floor_by(width as f64 / beta).max(opt.align_size);
        } else if opt.min_pixels > 0 && h_bar * w_bar < opt.min_pixels {
            let beta = (opt.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ceil_by(height as f64 * beta);
            w_bar = ceil_by(width as f64 * beta);
        }
        (w_bar as i32, h_bar as i32)
    }

    /// mtmd-image.cpp:39 `img_tool::resize`
    pub fn resize(
        src: &ClipImageU8,
        dst: &mut ClipImageU8,
        target: (i32, i32),
        algo: ResizeAlgo,
        pad: PadStyle,
        pad_color: [u8; 3],
    ) {
        dst.set_size(target.0, target.1, src.is_placeholder());
        if src.is_placeholder() {
            return;
        }
        if (dst.nx, dst.ny) == (src.nx, src.ny) {
            dst.buf.copy_from_slice(&src.buf);
            return;
        }
        if pad == PadStyle::None {
            resize_pillow(src, dst, target.0, target.1, algo);
            return;
        }
        let mut resized = ClipImageU8::default();
        let scale_w = target.0 as f32 / src.nx as f32;
        let scale_h = target.1 as f32 / src.ny as f32;
        let scale = scale_w.min(scale_h);
        let (new_width, new_height) = if pad == PadStyle::Nearest {
            (
                (((src.nx as f32 * scale).round()) as i32).min(target.0),
                (((src.ny as f32 * scale).round()) as i32).min(target.1),
            )
        } else {
            (
                (((src.nx as f32 * scale).ceil()) as i32).min(target.0),
                (((src.ny as f32 * scale).ceil()) as i32).min(target.1),
            )
        };
        resize_pillow(src, &mut resized, new_width, new_height, algo);
        fill(dst, pad_color);
        let (offset_x, offset_y) = if pad == PadStyle::Nearest {
            (
                ((target.0 - new_width) as f32 / 2.0).round() as i32,
                ((target.1 - new_height) as f32 / 2.0).round() as i32,
            )
        } else {
            ((target.0 - new_width) / 2, (target.1 - new_height) / 2)
        };
        composite(dst, &resized, offset_x, offset_y);
    }

    /// mtmd-image.cpp:160 `img_tool::composite`
    pub fn composite(dst: &mut ClipImageU8, src: &ClipImageU8, offset_x: i32, offset_y: i32) {
        if src.is_placeholder() {
            return;
        }
        for y in 0..src.ny {
            for x in 0..src.nx {
                let (dx, dy) = (x + offset_x, y + offset_y);
                if dx < 0 || dy < 0 || dx >= dst.nx || dy >= dst.ny {
                    continue;
                }
                dst.set_pixel(dx, dy, src.get_pixel(x, y));
            }
        }
    }

    /// mtmd-image.cpp:182 `img_tool::fill`
    pub fn fill(img: &mut ClipImageU8, color: [u8; 3]) {
        if img.is_placeholder() {
            return;
        }
        for y in 0..img.ny {
            for x in 0..img.nx {
                img.set_pixel(x, y, color);
            }
        }
    }

    /// mtmd-image.cpp:127 `img_tool::crop` — the pixel-copy window the
    /// llava-uhd slicer cuts tiles with
    pub fn crop(
        image: &ClipImageU8,
        dst: &mut ClipImageU8,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
    ) {
        assert!(x >= 0 && y >= 0 && w > 0 && h > 0);
        assert!(x + w <= image.nx && y + h <= image.ny);
        dst.set_size(w, h, image.is_placeholder());
        if image.is_placeholder() {
            // no-op for placeholder image, just set the size and return
            return;
        }
        for i in 0..h {
            for j in 0..w {
                dst.set_pixel(j, i, image.get_pixel(x + j, y + i));
            }
        }
    }

    /// mtmd-image.cpp:204 `resize_pillow` — separable Pillow-compatible
    /// resampling with 22-bit fixed point weights.
    fn resize_pillow(
        src: &ClipImageU8,
        dst: &mut ClipImageU8,
        target_width: i32,
        target_height: i32,
        algo: ResizeAlgo,
    ) {
        // mtmd-image.cpp:212 — 32 - 8 (uint8 pixel) - 2 (accumulator headroom)
        const PRECISION_BITS: u32 = 22;

        let filter_support = match algo {
            ResizeAlgo::Bilinear => 1.0f64,
            ResizeAlgo::Bicubic => 2.0f64,
            ResizeAlgo::Lanczos => 3.0f64,
        };

        // mtmd-image.cpp:226 `resample_filter`
        let resample_filter = |x: f64| -> f64 {
            if algo == ResizeAlgo::Lanczos {
                if (-3.0..3.0).contains(&x) {
                    let sinc = |v: f64| {
                        if v == 0.0 {
                            return 1.0;
                        }
                        let pi_v = v * std::f64::consts::PI;
                        pi_v.sin() / pi_v
                    };
                    return sinc(x) * sinc(x / 3.0);
                }
                return 0.0;
            }
            let x = if x < 0.0 { -x } else { x };
            if algo == ResizeAlgo::Bilinear {
                return if x < 1.0 { 1.0 - x } else { 0.0 };
            }
            const A: f64 = -0.5; // Pillow's bicubic a
            if x < 1.0 {
                return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            }
            if x < 2.0 {
                return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
            }
            0.0
        };

        let clip8 = |v: i32| -> u8 {
            if v < 0 {
                0
            } else if v > 255 {
                255
            } else {
                v as u8
            }
        };

        // mtmd-image.cpp:278 `precompute_weights`
        let precompute_weights = |in_size: usize, out_size: usize| -> (usize, Vec<i32>, Vec<i32>) {
            assert!(in_size > 0 && out_size > 0);
            let mut filterscale = in_size as f64 / out_size as f64;
            let scale = filterscale;
            if filterscale < 1.0 {
                filterscale = 1.0;
            }
            let support = filter_support * filterscale;
            let ksize = (support.ceil() as usize) * 2 + 1;
            let mut pre = vec![0.0f64; out_size * ksize];
            let mut bounds = vec![0i32; out_size * 2];
            for xx in 0..out_size {
                let center = (xx as f64 + 0.5) * scale;
                let mut ww = 0.0f64;
                let ss = 1.0 / filterscale;
                let mut xmin = (center - support + 0.5) as i64;
                if xmin < 0 {
                    xmin = 0;
                }
                let mut xmax = (center + support + 0.5) as i64;
                if xmax > in_size as i64 {
                    xmax = in_size as i64;
                }
                xmax -= xmin;
                for x in 0..xmax as usize {
                    let w = resample_filter((x as f64 + xmin as f64 - center + 0.5) * ss);
                    pre[xx * ksize + x] = w;
                    ww += w;
                }
                for x in 0..xmax as usize {
                    if ww != 0.0 {
                        pre[xx * ksize + x] /= ww;
                    }
                }
                for x in xmax as usize..ksize {
                    pre[xx * ksize + x] = 0.0;
                }
                bounds[xx * 2] = xmin as i32;
                bounds[xx * 2 + 1] = xmax as i32;
            }
            // mtmd-image.cpp:350 — Pillow adds +/-0.5 then truncates toward zero
            let fxp_scale = std::f64::consts::PI; // placeholder, replaced below
            let _ = fxp_scale;
            let fxp_scale = (1u64 << PRECISION_BITS) as f64;
            let mut weights = vec![0i32; out_size * ksize];
            for i in 0..out_size * ksize {
                let rounded = pre[i] * fxp_scale + if pre[i] < 0.0 { -0.5 } else { 0.5 };
                weights[i] = rounded as i32;
            }
            (ksize, bounds, weights)
        };

        let (src_width, src_height) = (src.nx as usize, src.ny as usize);
        if target_width <= 0 || target_width > 65536 || target_height <= 0 || target_height > 65536
        {
            panic!("resize target {target_width}x{target_height} is out of range (max 65536)");
        }
        let need_horizontal = target_width as usize != src_width;
        let need_vertical = target_height as usize != src_height;

        let (ks_h, bounds_h, weights_h) = if need_horizontal {
            precompute_weights(src_width, target_width as usize)
        } else {
            (0, Vec::new(), Vec::new())
        };
        let (ks_v, bounds_v, weights_v) = if need_vertical {
            precompute_weights(src_height, target_height as usize)
        } else {
            (0, Vec::new(), Vec::new())
        };

        // mtmd-image.cpp:363 `resample_horizontal`
        let resample_horizontal =
            |data: &[u8], in_nx: usize, in_ny: usize, out_nx: usize| -> Vec<u8> {
                let mut out = vec![0u8; out_nx * in_ny * 3];
                for yy in 0..in_ny {
                    let src_row = &data[yy * in_nx * 3..];
                    let dst_row = &mut out[yy * out_nx * 3..];
                    for xx in 0..out_nx {
                        let xmin = bounds_h[xx * 2] as usize;
                        let xcnt = bounds_h[xx * 2 + 1] as usize;
                        let k = &weights_h[xx * ks_h..];
                        let mut p = xmin * 3;
                        let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                        for x in 0..xcnt {
                            ss[0] += src_row[p] as i32 * k[x];
                            ss[1] += src_row[p + 1] as i32 * k[x];
                            ss[2] += src_row[p + 2] as i32 * k[x];
                            p += 3;
                        }
                        dst_row[xx * 3] = clip8(ss[0] >> PRECISION_BITS);
                        dst_row[xx * 3 + 1] = clip8(ss[1] >> PRECISION_BITS);
                        dst_row[xx * 3 + 2] = clip8(ss[2] >> PRECISION_BITS);
                    }
                }
                out
            };

        // mtmd-image.cpp:406 `resample_vertical`
        let resample_vertical = |data: &[u8], in_nx: usize, out_ny: usize| -> Vec<u8> {
            let row_elems = in_nx * 3;
            let mut out = vec![0u8; row_elems * out_ny];
            let mut acc = vec![0i32; row_elems];
            for yy in 0..out_ny {
                let ymin = bounds_v[yy * 2] as usize;
                let ycnt = bounds_v[yy * 2 + 1] as usize;
                let k = &weights_v[yy * ks_v..];
                acc.iter_mut().for_each(|v| *v = 1 << (PRECISION_BITS - 1));
                for y in 0..ycnt {
                    let row = &data[(ymin + y) * row_elems..];
                    let w = k[y];
                    for i in 0..row_elems {
                        acc[i] += row[i] as i32 * w;
                    }
                }
                let dst_row = &mut out[yy * row_elems..];
                for i in 0..row_elems {
                    dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
                }
            }
            out
        };

        // mtmd-image.cpp:441 main resampling logic
        if need_horizontal && need_vertical {
            let temp = resample_horizontal(&src.buf, src_width, src_height, target_width as usize);
            dst.set_size(target_width, target_height, false);
            dst.buf = resample_vertical(&temp, target_width as usize, target_height as usize);
        } else if need_horizontal {
            dst.set_size(target_width, src_height as i32, false);
            dst.buf = resample_horizontal(&src.buf, src_width, src_height, target_width as usize);
        } else if need_vertical {
            dst.set_size(src_width as i32, target_height, false);
            dst.buf = resample_vertical(&src.buf, src_width, target_height as usize);
        } else {
            dst.set_size(src.nx, src.ny, false);
            dst.buf.copy_from_slice(&src.buf);
        }
    }
}

/// Build the whole preprocess+encode path for one bitmap, the way
/// `mtmd_image_tokens` holds it. Returns (embeddings, n_tokens_x, n_tokens_y).
pub fn clip_encode_image(
    cx: &mut ClipContext,
    img_u8: &ClipImageU8,
) -> Result<(Vec<f32>, i32, i32, i32), String> {
    let entries = cx.image_preprocess(img_u8)?;
    let n_tokens = cx.n_output_tokens(&entries[0]);
    let nx = cx.n_output_tokens_x(&entries[0]);
    let ny = cx.n_output_tokens_y(&entries[0]);
    let batch = ClipImageF32Batch {
        entries,
        is_audio: false,
    };
    let embd = cx.image_batch_encode(&batch)?;
    Ok((embd, n_tokens, nx, ny))
}

/// Per-node dump used to localise a divergence against the reference's
/// `common_debug_cb_eval` line-by-line log (`llama-mtmd-debug` with
/// `-fa off` prints every node). One line per graph node:
/// `idx op (dst_ty) dst_ne | src0 name{ne} [| src1 name{ne}] | sum`.
/// The sum is the reference's own accumulation (`common_debug_print_tensor`,
/// common/debug.cpp:335: f32, row-major i3/i2/i1/i0 over the node's strides).

/// `MTMD_DEBUG_NODES_BIN=<path>`: the raw byte dump the bit-exact node
/// comparator (parity/ref_clip_graph_dump.cpp) reads —
/// `[u32 name_len][name][u32 ne x4][u64 nbytes][data]` per computed node.
/// Collected DURING compute through the eval callback (a post-hoc walk would
/// read tensors whose arena memory a later node already reused).
static NODE_DUMP: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

fn node_dump_cb(node: &ggml::compute::EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true; // always retrieve data
    }
    use ggml::tensor::GgmlOp as O;
    if matches!(
        node.op,
        O::None | O::View | O::Reshape | O::Permute | O::Transpose
    ) {
        return true;
    }
    let n: i64 = node.ne.iter().product();
    if n <= 0 {
        return true;
    }
    let Some(data) = node.data else { return true };
    let nbytes = (n as usize) * node.ty.type_size();
    if data.len() < nbytes {
        return true;
    }
    // the reference's ggml_op_desc: UNARY carries the kind in op_params[0];
    // the port's NORM carries the RMS flag in op_params[1]
    let name: &str = if node.op == O::Silu {
        match node.op_params[0] {
            v if v == ggml::ops::GGML_UNARY_OP_TANH => "TANH",
            v if v == ggml::ops::GGML_UNARY_OP_ELU => "ELU",
            v if v == ggml::ops::GGML_UNARY_OP_RELU => "RELU",
            v if v == ggml::ops::GGML_UNARY_OP_SIGMOID => "SIGMOID",
            v if v == ggml::ops::GGML_UNARY_OP_GELU => "GELU",
            v if v == ggml::ops::GGML_UNARY_OP_GELU_ERF => "GELU_ERF",
            v if v == ggml::ops::GGML_UNARY_OP_SILU => "SILU",
            v if v == ggml::ops::GGML_UNARY_OP_EXP => "EXP",
            _ => "UNARY",
        }
    } else if node.op == O::Norm && node.op_params[1] == ggml::ops::OP_FLAG_NORM_IS_RMS {
        "RMS_NORM"
    } else {
        op_name_c(node.op)
    };
    let mut guard = NODE_DUMP.lock().unwrap();
    if let Some(out) = guard.as_mut() {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        for d in node.ne {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        out.extend_from_slice(&(nbytes as u64).to_le_bytes());
        out.extend_from_slice(&data[..nbytes]);
    }
    true
}

/// the non-UNARY half of [`c_op_name`]
fn op_name_c(op: ggml::tensor::GgmlOp) -> &'static str {
    use ggml::tensor::GgmlOp as O;
    match op {
        O::None => "NONE",
        O::Dup => "DUP", // the port maps GGML_OP_CONT onto Dup as well
        O::Add => "ADD",
        O::Mul => "MUL",
        O::Div => "DIV",
        O::Sub => "SUB",
        O::Sqr => "SQR",
        O::Sqrt => "SQRT",
        O::Sin => "SIN",
        O::Cos => "COS",
        O::SumRows => "SUM_ROWS",
        O::Mean => "MEAN",
        O::ArgMax => "ARGMAX",
        O::Repeat => "REPEAT",
        O::Concat => "CONCAT",
        O::Norm => "NORM",
        O::MulMat => "MUL_MAT",
        O::MulMatId => "MUL_MAT_ID",
        O::Scale => "SCALE",
        O::SetRows => "SET_ROWS",
        O::GetRows => "GET_ROWS",
        O::DiagMaskInf => "DIAG_MASK_INF",
        O::SoftMax => "SOFT_MAX",
        O::RoPE => "ROPE",
        O::RoPEBack => "ROPE_BACK",
        O::Clamp => "CLAMP",
        O::Im2col => "IM2COL",
        O::Pool1d => "POOL_1D",
        O::Pool2d => "POOL_2D",
        O::Upscale => "UPSCALE",
        O::Pad => "PAD",
        O::PadReflect1d => "PAD_REFLECT_1D",
        O::Roll => "ROLL",
        O::Argsort => "ARGSORT",
        O::TopK => "TOP_K",
        O::Fill => "FILL",
        O::FlashAttnExt => "FLASH_ATTN_EXT",
        O::SsmConv => "SSM_CONV",
        O::SsmScan => "SSM_SCAN",
        O::Gdn => "GATED_DELTA_NET",
        O::LightningIndexer => "LIGHTNING_INDEXER",
        O::Dsv4HcComb => "DSV4_HC_COMB",
        O::Dsv4HcPre => "DSV4_HC_PRE",
        O::Dsv4HcPost => "DSV4_HC_POST",
        O::Glu => "GLU",
        O::Conv2dDirect => "CONV_2D",
        O::Conv2dDw => "CONV_2D_DW",
        O::AddId => "ADD_ID",
        // layout-only ops are filtered before the name is consulted
        O::View | O::Reshape | O::Permute | O::Transpose => "NONE",
        // ops the audio graphs never build (other agents' additions)
        _ => "UNARY",
    }
}

/// dump every computed node of the NEXT audio graph compute into
/// `path` via the eval callback
fn debug_dump_nodes_bin_run(path: &str, compute: impl FnOnce()) {
    *NODE_DUMP.lock().unwrap() = Some(Vec::new());
    ggml::compute::set_eval_callback(Some(node_dump_cb));
    compute();
    ggml::compute::set_eval_callback(None);
    let taken = std::mem::take(&mut *NODE_DUMP.lock().unwrap());
    if let Some(bytes) = taken {
        let _ = std::fs::write(path, bytes);
    }
}

fn debug_dump_nodes(cx: &Context, graph: &Graph, path: &str) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (i, &id) in graph.nodes.iter().enumerate() {
        let ty = cx.ty(id);
        let ne = cx.ne(id);
        let nb = cx.nb(id);
        let mut sum = 0.0f32;
        if let Some(bytes) = cx.data_bytes(id) {
            for i3 in 0..ne[3] as usize {
                for i2 in 0..ne[2] as usize {
                    for i1 in 0..ne[1] as usize {
                        for i0 in 0..ne[0] as usize {
                            let off = i0 * nb[0] as usize
                                + i1 * nb[1] as usize
                                + i2 * nb[2] as usize
                                + i3 * nb[3] as usize;
                            if off + 4 > bytes.len() {
                                continue;
                            }
                            let v = match ty {
                                GgmlType::F32 => {
                                    f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
                                }
                                GgmlType::F16 => half::f16::from_le_bytes(
                                    bytes[off..off + 2].try_into().unwrap(),
                                )
                                .to_f32(),
                                _ => 0.0,
                            };
                            sum += v;
                        }
                    }
                }
            }
        }
        let shape = |ne: &[i64; 4]| format!("{{{}, {}, {}, {}}}", ne[0], ne[1], ne[2], ne[3]);
        let mut line = format!("{i} id{id:?} {:?} ({ty:?}) {}", cx.op(id), shape(ne));
        for s in cx.src(id).iter().take(3).flatten() {
            let _ = write!(line, " | id{:?} {} {}", s, cx.name(*s), shape(cx.ne(*s)));
        }
        let _ = write!(line, " | sum={sum}");
        // the same sample positions the reference's common_debug_print_tensor
        // (common/debug.cpp:347) prints with n = 3, in print order
        let sample_idx = |n: i64| -> Vec<i64> {
            let mut idx: Vec<i64> = (0..3.min(n)).collect();
            if n > 6 {
                idx.extend([n - 3, n - 2, n - 1]);
            }
            idx
        };
        let read = |i0: i64, i1: i64, i2: i64, i3: i64| -> f32 {
            let bytes = cx.data_bytes(id).unwrap_or(&[]);
            let off = i0 as usize * nb[0] as usize
                + i1 as usize * nb[1] as usize
                + i2 as usize * nb[2] as usize
                + i3 as usize * nb[3] as usize;
            match ty {
                GgmlType::F32 if off + 4 <= bytes.len() => {
                    f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
                }
                GgmlType::F16 if off + 2 <= bytes.len() => {
                    half::f16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()).to_f32()
                }
                _ => 0.0,
            }
        };
        let mut vals = Vec::new();
        for i3 in sample_idx(ne[3]) {
            for i2 in sample_idx(ne[2]) {
                for i1 in sample_idx(ne[1]) {
                    for i0 in sample_idx(ne[0]) {
                        vals.push(read(i0, i1, i2, i3));
                    }
                }
            }
        }
        let _ = write!(line, " | vals={vals:?}");
        out.push_str(&line);
        out.push('\n');
    }
    std::fs::write(path, out)
}

/// `MTMD_DEBUG_EMBEDDINGS=<path>` format (clip.cpp:5884):
/// `[int32 n_tokens][int32 n_embd][f32 data]`. Used by the parity scripts to
/// diff against the reference `llama-mtmd-debug` / `llama-mtmd-cli` dump.
pub fn write_embedding_dump(
    path: &str,
    embd: &[f32],
    n_tokens: i32,
    n_embd: i32,
) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(8 + embd.len() * 4);
    bytes.extend_from_slice(&n_tokens.to_le_bytes());
    bytes.extend_from_slice(&n_embd.to_le_bytes());
    for v in embd {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes)
}

/// The `cb` pattern of `llama-mtmd-debug -p encode --image cb`
/// (debug/mtmd-debug.cpp:136): a checkerboard in **normalised** value space
/// (0.0 = black, 1.0 = white), fed straight to `clip_image_encode` without
/// preprocessing. Returns (nx, ny, raw f32 bitmap).
pub fn debug_pattern_cb(size: i32) -> ClipImageF32 {
    let mut img = ClipImageF32::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let v = if (x + y) % 2 != 0 { 0.0 } else { 1.0 };
            let i = ((y * size + x) * 3) as usize;
            img.buf[i] = v;
            img.buf[i + 1] = v;
            img.buf[i + 2] = v;
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sc(x: i32, y: i32, w: i32, h: i32) -> SliceCoordinates {
        SliceCoordinates { x, y, w, h }
    }
    use ggml::gguf_write::GgufWriter;
    use ggml::Value;

    /// Paths of the local multimodal models (skipped when absent).
    const MMPROJ_JACKRONG: &str =
        "/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/mmproj-F32.gguf";
    const TEXT_JACKRONG: &str =
        "/home/jeffrey/.lmstudio/models/Jackrong/Qwen3.8-27B-MTP-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf";

    // ------------------------------------------------------------------
    // synthetic clip GGUF — the geometry of qwen3vl_merger at toy scale
    // ------------------------------------------------------------------

    /// A tiny but structurally complete `qwen3vl_merger` mmproj: image_size 64,
    /// patch 8, n_embd 16, 1 block, projection_dim 24, 2x2 merge. Weights are
    /// deterministic pseudo-random f32, which is enough to exercise every node
    /// of the graph (shape bugs surface immediately, numerics are checked
    /// against the reference separately).
    fn write_synth_mmproj(path: &str) {
        const IMAGE_SIZE: i64 = 64;
        const PATCH: i64 = 8;
        const N_EMBD: i64 = 16;
        const N_FF: i64 = 32;
        const N_HEAD: i64 = 2;
        const PROJ: i64 = 24; // projection_dim = 4*... the merger output
        const N_POS: i64 = (IMAGE_SIZE / PATCH) * (IMAGE_SIZE / PATCH);

        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("clip".into()));
        w.set_kv(
            "general.name",
            Value::String("llama-rust-synth-mtmd".into()),
        );
        w.set_kv("general.file_type", Value::U32(0));
        w.set_kv("clip.has_vision_encoder", Value::Bool(true));
        w.set_kv(
            "clip.projector_type",
            Value::String("qwen3vl_merger".into()),
        );
        w.set_kv("clip.use_gelu", Value::Bool(true));
        w.set_kv("clip.vision.image_size", Value::U32(IMAGE_SIZE as u32));
        w.set_kv("clip.vision.patch_size", Value::U32(PATCH as u32));
        w.set_kv("clip.vision.embedding_length", Value::U32(N_EMBD as u32));
        w.set_kv("clip.vision.feed_forward_length", Value::U32(N_FF as u32));
        w.set_kv("clip.vision.block_count", Value::U32(1));
        w.set_kv(
            "clip.vision.attention.head_count",
            Value::U32(N_HEAD as u32),
        );
        w.set_kv("clip.vision.projection_dim", Value::U32(PROJ as u32));
        w.set_kv("clip.vision.spatial_merge_size", Value::U32(2));
        w.set_kv("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6));
        w.set_kv(
            "clip.vision.image_mean",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );
        w.set_kv(
            "clip.vision.image_std",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );

        let mut names: Vec<(String, [i64; 4])> = Vec::new();
        let mut push =
            |w: &mut GgufWriter, names: &mut Vec<(String, [i64; 4])>, n: String, ne: [i64; 4]| {
                w.add_tensor(&n, GgmlType::F32, ne);
                names.push((n, ne));
            };
        push(
            &mut w,
            &mut names,
            "v.patch_embd.weight".into(),
            [PATCH, PATCH, 3, N_EMBD],
        );
        push(
            &mut w,
            &mut names,
            "v.patch_embd.weight.1".into(),
            [PATCH, PATCH, 3, N_EMBD],
        );
        push(
            &mut w,
            &mut names,
            "v.position_embd.weight".into(),
            [N_EMBD, N_POS, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.post_ln.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.post_ln.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln1.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln1.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln2.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln2.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_qkv.weight".into(),
            [N_EMBD, 3 * N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_qkv.bias".into(),
            [3 * N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_up.weight".into(),
            [N_EMBD, N_FF, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_up.bias".into(),
            [N_FF, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.weight".into(),
            [N_FF, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.0.weight".into(),
            [4 * N_EMBD, 4 * N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.0.bias".into(),
            [4 * N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.2.weight".into(),
            [4 * N_EMBD, PROJ, 1, 1],
        );
        push(&mut w, &mut names, "mm.2.bias".into(), [PROJ, 1, 1, 1]);

        // deterministic weights in [-0.1, 0.1): no NaN/Inf risk in the chain
        let payloads: Vec<Vec<u8>> = names
            .iter()
            .enumerate()
            .map(|(ti, (_, ne))| {
                let n: i64 = ne.iter().product();
                let mut bytes = Vec::with_capacity(n as usize * 4);
                for i in 0..n {
                    let v = (((ti * 7919 + i as usize * 104729) % 2001) as f32 / 10000.0) - 0.1;
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
        let f = std::fs::File::create(path).expect("create synth mmproj");
        let mut bw = std::io::BufWriter::new(f);
        w.write(&mut bw, &refs).expect("write synth mmproj");
    }

    fn synth_image(w: i32, h: i32) -> ClipImageU8 {
        let mut img = ClipImageU8::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 7 + y * 13) % 256) as u8;
                img.set_pixel(x, y, [v, v.wrapping_mul(3), v.wrapping_add(97)]);
            }
        }
        img
    }

    /// The whole pipeline at toy scale: metadata parsing, the tensor table, the
    /// preprocessor geometry and the ViT graph — no 1.8 GB file needed.
    #[test]
    fn synth_mmproj_load_preprocess_encode() {
        let dir = std::env::temp_dir().join("llama-rust-mtmd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-mmproj.gguf");
        let path_s = path.to_str().unwrap().to_string();
        write_synth_mmproj(&path_s);

        let mut cx =
            clip_init_from_file(&path_s, &ClipContextParams::default()).expect("load synth");
        assert_eq!(cx.model.proj_type, ProjectorType::Qwen3Vl);
        assert_eq!(cx.n_mmproj_embd(), 24);
        let hp = cx.hparams().clone();
        assert_eq!(
            (hp.image_size, hp.patch_size, hp.n_embd, hp.n_layer),
            (64, 8, 16, 1)
        );
        assert_eq!(hp.n_merge, 2);
        // set_limit_image_tokens(8, 4096), clip.cpp:1673
        assert_eq!(hp.image_min_pixels, 8 * 8 * 8 * 2 * 2);
        assert_eq!(hp.image_max_pixels, 4096 * 8 * 8 * 2 * 2);

        // preprocess geometry: 100x70 -> align 16 -> 96x64 (smart_resize)
        let img = synth_image(100, 70);
        let entries = cx.image_preprocess(&img).expect("preprocess");
        assert_eq!(entries.len(), 1);
        assert_eq!((entries[0].nx, entries[0].ny), (96, 64));
        // merge grid: (96/8/2) x (64/8/2) = 6 x 4 tokens
        assert_eq!(cx.n_output_tokens(&entries[0]), 24);
        assert_eq!(cx.n_output_tokens_x(&entries[0]), 6);
        assert_eq!(cx.n_output_tokens_y(&entries[0]), 4);

        // encode: [projection_dim, n_tokens]
        let (embd, n_tokens, nx, ny) = clip_encode_image(&mut cx, &img).expect("encode");
        assert_eq!((n_tokens, nx, ny), (24, 6, 4));
        assert_eq!(embd.len(), 24 * 24);
        assert!(
            embd.iter().all(|v| v.is_finite()),
            "embeddings must be finite"
        );
        assert!(
            embd.iter().any(|v| *v != 0.0),
            "embeddings must not be all zero"
        );

        // the square image_size case takes the `n_per_side` fast path of
        // resize_position_embeddings (clip.cpp:321)
        let img64 = synth_image(64, 64);
        let (embd64, n64, _, _) = clip_encode_image(&mut cx, &img64).expect("encode 64");
        assert_eq!(n64, 16);
        assert_eq!(embd64.len(), 24 * 16);

        // repeated encodes reuse the same graph/arena: the second call must
        // produce bit-identical output
        let (embd64b, _, _, _) = clip_encode_image(&mut cx, &img64).expect("encode 64 again");
        assert_eq!(embd64, embd64b, "repeated encode must be bit-identical");

        let _ = TEXT_JACKRONG;
    }

    // ------------------------------------------------------------------
    // synthetic ling3vl mmproj — the same tower, the norm-only merger
    // (models/ling3vl.cpp, upstream f830688e9)
    // ------------------------------------------------------------------

    /// The qwen3vl synthetic with `clip.projector_type = "ling3vl"` plus the
    /// `mm.input_norm.{weight,bias}` merger norm (TN_MM_INP_NORM).
    fn write_synth_mmproj_ling3vl(path: &str) {
        const IMAGE_SIZE: i64 = 64;
        const PATCH: i64 = 8;
        const N_EMBD: i64 = 16;
        const N_FF: i64 = 32;
        const N_HEAD: i64 = 2;
        const PROJ: i64 = 24;
        const N_POS: i64 = (IMAGE_SIZE / PATCH) * (IMAGE_SIZE / PATCH);

        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("clip".into()));
        w.set_kv(
            "general.name",
            Value::String("llama-rust-synth-ling3vl".into()),
        );
        w.set_kv("general.file_type", Value::U32(0));
        w.set_kv("clip.has_vision_encoder", Value::Bool(true));
        w.set_kv("clip.projector_type", Value::String("ling3vl".into()));
        w.set_kv("clip.use_gelu", Value::Bool(true));
        w.set_kv("clip.vision.image_size", Value::U32(IMAGE_SIZE as u32));
        w.set_kv("clip.vision.patch_size", Value::U32(PATCH as u32));
        w.set_kv("clip.vision.embedding_length", Value::U32(N_EMBD as u32));
        w.set_kv("clip.vision.feed_forward_length", Value::U32(N_FF as u32));
        w.set_kv("clip.vision.block_count", Value::U32(1));
        w.set_kv(
            "clip.vision.attention.head_count",
            Value::U32(N_HEAD as u32),
        );
        w.set_kv("clip.vision.projection_dim", Value::U32(PROJ as u32));
        w.set_kv("clip.vision.spatial_merge_size", Value::U32(2));
        w.set_kv("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6));
        w.set_kv(
            "clip.vision.image_mean",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );
        w.set_kv(
            "clip.vision.image_std",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );

        let mut names: Vec<(String, [i64; 4])> = Vec::new();
        let mut push =
            |w: &mut GgufWriter, names: &mut Vec<(String, [i64; 4])>, n: String, ne: [i64; 4]| {
                w.add_tensor(&n, GgmlType::F32, ne);
                names.push((n, ne));
            };
        push(
            &mut w,
            &mut names,
            "v.patch_embd.weight".into(),
            [PATCH, PATCH, 3, N_EMBD],
        );
        push(
            &mut w,
            &mut names,
            "v.patch_embd.weight.1".into(),
            [PATCH, PATCH, 3, N_EMBD],
        );
        push(
            &mut w,
            &mut names,
            "v.position_embd.weight".into(),
            [N_EMBD, N_POS, 1, 1],
        );
        push(&mut w, &mut names, "v.post_ln.weight".into(), [N_EMBD, 1, 1, 1]);
        push(&mut w, &mut names, "v.post_ln.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln1.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(&mut w, &mut names, "v.blk.0.ln1.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln2.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(&mut w, &mut names,
            "v.blk.0.ln2.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_qkv.weight".into(),
            [N_EMBD, 3 * N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_qkv.bias".into(),
            [3 * N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_up.weight".into(),
            [N_EMBD, N_FF, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_up.bias".into(),
            [N_FF, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.weight".into(),
            [N_FF, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        // the ling3vl merger norm (mm.input_norm.*, clip-impl.h:148-149)
        push(
            &mut w,
            &mut names,
            "mm.input_norm.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.input_norm.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        // the top-level projector MLP (mm.0 / mm.2, TN_LLAVA_PROJ 0/2)
        push(
            &mut w,
            &mut names,
            "mm.0.weight".into(),
            [4 * N_EMBD, 4 * N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.0.bias".into(),
            [4 * N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "mm.2.weight".into(),
            [4 * N_EMBD, PROJ, 1, 1],
        );
        push(&mut w, &mut names, "mm.2.bias".into(), [PROJ, 1, 1, 1]);

        let payloads: Vec<Vec<u8>> = names
            .iter()
            .enumerate()
            .map(|(ti, (_, ne))| {
                let n: i64 = ne.iter().product();
                let mut bytes = Vec::with_capacity(n as usize * 4);
                for i in 0..n {
                    let v = (((ti * 7919 + i as usize * 104729) % 2001) as f32 / 10000.0) - 0.1;
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
        let f = std::fs::File::create(path).expect("create synth ling3vl mmproj");
        let mut bw = std::io::BufWriter::new(f);
        w.write(&mut bw, &refs).expect("write synth ling3vl mmproj");
    }

    /// ling3vl at toy scale: metadata, merger-norm loading, preprocessor
    /// geometry and the norm-only merger graph; with `LLAMA_E2E_DUMP` set the
    /// embeddings of parity/mtmd-fixture.png are dumped in
    /// MTMD_DEBUG_EMBEDDINGS format for the reference-side byte comparison
    /// (`parity/ling3vl_parity.sh`).
    #[test]
    fn synth_mmproj_ling3vl_load_preprocess_encode() {
        let dir = std::env::temp_dir().join("llama-rust-mtmd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-ling3vl-mmproj.gguf");
        let path_s = path.to_str().unwrap().to_string();
        write_synth_mmproj_ling3vl(&path_s);

        let mut cx =
            clip_init_from_file(&path_s, &ClipContextParams::default()).expect("load synth");
        assert_eq!(cx.model.proj_type, ProjectorType::Ling3Vl);
        assert_eq!(cx.n_mmproj_embd(), 24);
        let hp = cx.hparams().clone();
        assert_eq!(hp.n_merge, 2);

        // preprocess geometry matches the qwen3vl family (smart_resize)
        let img = synth_image(100, 70);
        let entries = cx.image_preprocess(&img).expect("preprocess");
        assert_eq!((entries[0].nx, entries[0].ny), (96, 64));
        assert_eq!(cx.n_output_tokens(&entries[0]), 24);
        assert_eq!(cx.n_output_tokens_x(&entries[0]), 6);
        assert_eq!(cx.n_output_tokens_y(&entries[0]), 4);

        // encode: [projection_dim, n_tokens] through the norm-only merger
        let (embd, n_tokens, nx, ny) = clip_encode_image(&mut cx, &img).expect("encode");
        assert_eq!((n_tokens, nx, ny), (24, 6, 4));
        assert_eq!(embd.len(), 24 * 24);
        assert!(embd.iter().all(|v| v.is_finite()));
        assert!(embd.iter().any(|v| *v != 0.0));

        // determinism across repeated encodes
        let (embd2, _, _, _) = clip_encode_image(&mut cx, &img).expect("encode again");
        assert_eq!(embd, embd2, "repeated encode must be bit-identical");

        // the reference-side dump for parity/ling3vl_parity.sh: the SAME
        // fixture PNG both sides see, MTMD_DEBUG_EMBEDDINGS format
        if std::env::var_os("LLAMA_E2E_DUMP").is_some() {
            let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../parity/mtmd-fixture.png");
            let img_u8 = std::fs::read(&png)
                .ok()
                .and_then(|bytes| crate::mtmd::bitmap_from_bytes(&bytes).ok());
            if let Some(img_u8) = img_u8 {
                let (e, n, _, _) = clip_encode_image(&mut cx, &img_u8).expect("encode fixture");
                write_embedding_dump("/tmp/ling3vl-port.bin", &e, n, 24).unwrap();
                eprintln!("ling3vl: dumped {} embeddings to /tmp/ling3vl-port.bin", n);
            }
        }
    }

    /// Metadata + geometry of the real local mmproj (skipped when absent).
    #[test]
    fn real_mmproj_metadata_and_geometry() {
        let Ok(mut cx) = clip_init_from_file(MMPROJ_JACKRONG, &ClipContextParams::default()) else {
            eprintln!("skipping: {MMPROJ_JACKRONG} not present");
            return;
        };
        let hp = cx.hparams().clone();
        assert_eq!(cx.model.proj_type, ProjectorType::Qwen3Vl);
        assert_eq!(hp.image_size, 768);
        assert_eq!(hp.patch_size, 16);
        assert_eq!(hp.n_embd, 1152);
        assert_eq!(hp.n_ff, 4304);
        assert_eq!(hp.n_layer, 27);
        assert_eq!(hp.n_head, 16);
        assert_eq!(hp.n_head_kv, 16);
        assert_eq!(hp.projection_dim, 5120);
        assert_eq!(hp.n_merge, 2);
        assert_eq!(hp.ffn_op, FfnOp::Gelu); // clip.use_gelu = true
        assert_eq!(hp.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(hp.image_std, [0.5, 0.5, 0.5]);
        assert_eq!(cx.n_mmproj_embd(), 5120);

        // dyn_size output tokens for a 448x448 image: (448/32)^2 = 196
        let img = synth_image(448, 448);
        let entries = cx.image_preprocess(&img).expect("preprocess");
        assert_eq!((entries[0].nx, entries[0].ny), (448, 448));
        assert_eq!(cx.n_output_tokens(&entries[0]), 196);
        assert_eq!(cx.n_output_tokens_x(&entries[0]), 14);
        assert_eq!(cx.n_output_tokens_y(&entries[0]), 14);
    }

    // ------------------------------------------------------------------
    // synthetic cohere2v mmproj — the siglip tower (separate q/k/v ViT with
    // learned positions) + the swapped-swiglu two-layer projector
    // (upstream 50a6c5cf7, conversion/command_r.py Cohere2VisionModel)
    // ------------------------------------------------------------------

    /// The tensor/metadata layout a converted `Cohere2VisionModel` carries:
    /// separate attn_{q,k,v} (HF SigLIP, no conversion qkv fusion), pre/post
    /// layernorms, patch bias, TN_LLAVA_PROJ mm.{1,2} with biases, and the
    /// cohere2v kv (scale_factor + preproc_max_tiles + use_gelu).
    fn write_synth_mmproj_cohere2v(path: &str) {
        const IMAGE_SIZE: i64 = 64; // the tile edge
        const PATCH: i64 = 8;
        const N_EMBD: i64 = 16;
        const N_FF: i64 = 32;
        const N_HEAD: i64 = 2;
        const N_MERGE: i64 = 2;
        const MAX_TILES: i64 = 4;
        const D1: i64 = 24; // the hidden width between linear_1 / linear_2
        // == n_mmproj_embd (mm.2 out); must equal the parity text model's
        // n_embd (qwen2.5-0.5b: 896) — mtmd_init_from_file checks the match
        const PROJ: i64 = 896;
        const N_POS: i64 = (IMAGE_SIZE / PATCH) * (IMAGE_SIZE / PATCH);

        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("clip".into()));
        w.set_kv(
            "general.name",
            Value::String("llama-rust-synth-cohere2v".into()),
        );
        w.set_kv("general.file_type", Value::U32(0));
        w.set_kv("clip.has_vision_encoder", Value::Bool(true));
        w.set_kv("clip.projector_type", Value::String("cohere2v".into()));
        // command_r.py set_gguf_parameters: add_vision_use_gelu(True)
        w.set_kv("clip.use_gelu", Value::Bool(true));
        w.set_kv("clip.vision.image_size", Value::U32(IMAGE_SIZE as u32));
        w.set_kv("clip.vision.patch_size", Value::U32(PATCH as u32));
        w.set_kv("clip.vision.embedding_length", Value::U32(N_EMBD as u32));
        w.set_kv("clip.vision.feed_forward_length", Value::U32(N_FF as u32));
        w.set_kv("clip.vision.block_count", Value::U32(1));
        w.set_kv(
            "clip.vision.attention.head_count",
            Value::U32(N_HEAD as u32),
        );
        w.set_kv("clip.vision.projection_dim", Value::U32(PROJ as u32));
        w.set_kv(
            "clip.vision.attention.layer_norm_epsilon",
            Value::F32(1e-6),
        );
        // command_r.py: add_vision_projector_scale_factor(downsample_factor)
        w.set_kv(
            "clip.vision.projector.scale_factor",
            Value::U32(N_MERGE as u32),
        );
        // command_r.py: add_vision_preproc_max_tiles(max_patches)
        w.set_kv(
            "clip.vision.preproc_max_tiles",
            Value::U32(MAX_TILES as u32),
        );
        w.set_kv(
            "clip.vision.image_mean",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );
        w.set_kv(
            "clip.vision.image_std",
            Value::Array(ggml::GgufType::Float32, vec![Value::F32(0.5); 3]),
        );

        let mut names: Vec<(String, [i64; 4])> = Vec::new();
        let mut push =
            |w: &mut GgufWriter, names: &mut Vec<(String, [i64; 4])>, n: String, ne: [i64; 4]| {
                w.add_tensor(&n, GgmlType::F32, ne);
                names.push((n, ne));
            };
        push(
            &mut w,
            &mut names,
            "v.patch_embd.weight".into(),
            [PATCH, PATCH, 3, N_EMBD],
        );
        push(
            &mut w,
            &mut names,
            "v.patch_embd.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.position_embd.weight".into(),
            [N_EMBD, N_POS, 1, 1],
        );
        push(&mut w, &mut names, "v.pre_ln.weight".into(), [N_EMBD, 1, 1, 1]);
        push(&mut w, &mut names, "v.pre_ln.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.post_ln.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(&mut w, &mut names, "v.post_ln.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln1.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(&mut w, &mut names, "v.blk.0.ln1.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_q.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_q.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_k.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_k.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_v.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_v.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.weight".into(),
            [N_EMBD, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.attn_out.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ln2.weight".into(),
            [N_EMBD, 1, 1, 1],
        );
        push(&mut w, &mut names, "v.blk.0.ln2.bias".into(), [N_EMBD, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_up.weight".into(),
            [N_EMBD, N_FF, 1, 1],
        );
        push(&mut w, &mut names, "v.blk.0.ffn_up.bias".into(), [N_FF, 1, 1, 1]);
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.weight".into(),
            [N_FF, N_EMBD, 1, 1],
        );
        push(
            &mut w,
            &mut names,
            "v.blk.0.ffn_down.bias".into(),
            [N_EMBD, 1, 1, 1],
        );
        // TN_LLAVA_PROJ mm.{1,2}.{weight,bias}; linear_1 outputs [x, gate]
        push(
            &mut w,
            &mut names,
            "mm.1.weight".into(),
            [N_EMBD * N_MERGE * N_MERGE, 2 * D1, 1, 1],
        );
        push(&mut w, &mut names, "mm.1.bias".into(), [2 * D1, 1, 1, 1]);
        push(&mut w, &mut names, "mm.2.weight".into(), [D1, PROJ, 1, 1]);
        push(&mut w, &mut names, "mm.2.bias".into(), [PROJ, 1, 1, 1]);

        // deterministic weights in [-0.1, 0.1): no NaN/Inf risk in the chain
        let payloads: Vec<Vec<u8>> = names
            .iter()
            .enumerate()
            .map(|(ti, (_, ne))| {
                let n: i64 = ne.iter().product();
                let mut bytes = Vec::with_capacity(n as usize * 4);
                for i in 0..n {
                    let v = (((ti * 7919 + i as usize * 104729) % 2001) as f32 / 10000.0) - 0.1;
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
        let f = std::fs::File::create(path).expect("create synth cohere2v mmproj");
        let mut bw = std::io::BufWriter::new(f);
        w.write(&mut bw, &refs).expect("write synth cohere2v mmproj");
    }

    /// The cohere2v pipeline at toy scale: kv parsing, the tensor table, the
    /// llava-uhd square-tile slicing, the siglip ViT and the projector.
    #[test]
    fn synth_cohere2v_load_preprocess_encode() {
        let dir = std::env::temp_dir().join("llama-rust-mtmd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-mmproj-cohere2v.gguf");
        let path_s = path.to_str().unwrap().to_string();
        write_synth_mmproj_cohere2v(&path_s);

        let mut cx =
            clip_init_from_file(&path_s, &ClipContextParams::default()).expect("load synth");
        assert_eq!(cx.model.proj_type, ProjectorType::Cohere2V);
        assert_eq!(cx.n_mmproj_embd(), 896);
        let hp = cx.hparams().clone();
        assert_eq!(
            (hp.image_size, hp.patch_size, hp.n_embd, hp.n_layer, hp.n_merge),
            (64, 8, 16, 1, 2)
        );
        assert_eq!(hp.preproc_max_tiles, 4);
        // clip.cpp:1521 — the refined image is never padded
        assert_eq!(hp.image_pad_rf, PadStyle::None);

        // slicing geometry (mtmd-image.cpp:1166): a 150x100 image with tile 64
        // and max 4 tiles picks the (2,2) grid (least downscale), stretches
        // the image to 128x128 and adds a 64x64 thumbnail
        let img = synth_image(150, 100);
        let pre = cx.image_preprocess_full(&img).expect("preprocess");
        assert_eq!(pre.grid_x, 2);
        assert_eq!(pre.grid_y, 2);
        assert_eq!(pre.entries.len(), 4);
        for e in &pre.entries {
            assert_eq!((e.nx, e.ny), (64, 64));
            assert_eq!(cx.n_output_tokens(e), 16); // (64/8)^2 / 2^2
        }
        let ov = pre.overview.as_ref().expect("overview");
        assert_eq!((ov.nx, ov.ny), (64, 64));
        assert_eq!(cx.n_output_tokens(ov), 16);

        // a small image takes the overview-only path (grid 1x1 -> no slices)
        let small = synth_image(50, 40);
        let pre2 = cx.image_preprocess_full(&small).expect("preprocess small");
        assert_eq!(pre2.entries.len(), 0);
        assert!(pre2.has_overview());
        assert_eq!(pre2.grid_x, 0);
        assert_eq!(pre2.grid_y, 0);
        let ov2 = pre2.overview.as_ref().unwrap();
        assert_eq!((ov2.nx, ov2.ny), (64, 64));

        // encode one tile: [n_mmproj_embd=20, 16 tokens]
        let batch = ClipImageF32Batch {
            entries: vec![pre.entries[0].clone()],
            is_audio: false,
        };
        let embd = cx.image_batch_encode(&batch).expect("encode");
        assert_eq!(embd.len(), 896 * 16);
        assert!(embd.iter().all(|v| v.is_finite()));
        assert!(embd.iter().any(|v| *v != 0.0));

        // repeated encodes reuse the same graph/arena: bit-identical output
        let embd2 = cx.image_batch_encode(&batch).expect("encode again");
        assert_eq!(embd, embd2, "repeated encode must be bit-identical");

        // the FA-off path builds the soft_max_ext attention instead
        let mut cx_off = clip_init_from_file(
            &path_s,
            &ClipContextParams {
                flash_attn_type: ClipFlashAttn::Disabled,
                ..Default::default()
            },
        )
        .expect("reload fa-off");
        let embd_off = cx_off.image_batch_encode(&batch).expect("encode fa off");
        assert_eq!(embd_off.len(), 896 * 16);

        let _ = TEXT_JACKRONG;
    }

    /// clip.cpp:1533-1543 — KEY_IMAGE_RESIZE_ALGO parse contract
    #[test]
    fn resize_algo_parse() {
        assert_eq!(parse_resize_algo("bilinear").unwrap(), ResizeAlgo::Bilinear);
        assert_eq!(parse_resize_algo("bicubic").unwrap(), ResizeAlgo::Bicubic);
        assert_eq!(parse_resize_algo("lanczos").unwrap(), ResizeAlgo::Lanczos);
        assert!(parse_resize_algo("nearest").is_err());
    }

    /// cohere2v grid selection (mtmd-image.cpp:1166) against hand-computed
    /// HF behavior: least upscale wins, then least downscale.
    #[test]
    fn cohere2v_slice_instructions() {
        let dir = std::env::temp_dir().join("llama-rust-mtmd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-mmproj-cohere2v.gguf");
        let path_s = path.to_str().unwrap().to_string();
        write_synth_mmproj_cohere2v(&path_s);
        let cx = clip_init_from_file(&path_s, &ClipContextParams::default()).unwrap();

        // 150x100, tile 64, max 4: all grids downscale; best is (2,2) with
        // scale min(128/150, 128/100) = 0.8533
        let inst = cx.get_slice_instructions_cohere2v((150, 100));
        assert_eq!(inst.grid_size, (2, 2));
        assert_eq!(inst.refined_size, (128, 128));
        assert_eq!(inst.overview_size, (64, 64));
        assert_eq!(inst.slices.len(), 4);
        assert_eq!(inst.slices[1], sc(64, 0, 64, 64));

        // 700x500, tile 64, max 4: (4,1) upscales least? min(256/700, 64/500)
        // = 0.128 <1; (2,2): min(128/700,128/500)=0.1829; (3,1) is not a
        // divisor of any n<=4 except n=3: (1,3) min(64/700,192/500)=0.0914,
        // (3,1) min(192/700,64/500)=0.128. Best downscale: (2,2) 0.1829.
        let inst = cx.get_slice_instructions_cohere2v((700, 500));
        assert_eq!(inst.grid_size, (2, 2));
        assert_eq!(inst.slices.len(), 4);

        // 40x30 (smaller than the tile): grid (1,1), no slices
        let inst = cx.get_slice_instructions_cohere2v((40, 30));
        assert_eq!(inst.grid_size, (1, 1));
        assert!(inst.slices.is_empty());
        assert_eq!(inst.refined_size, (64, 64));

        // a wide image: 100x150 with max 4 — (1,3) would need n=3: scale
        // min(64/100, 192/150) = 0.64; (2,2): min(0.8533...) wait 128/100=1.28,
        // 128/150=0.8533 -> 0.8533 (best); so (2,2)
        let inst = cx.get_slice_instructions_cohere2v((100, 150));
        assert_eq!(inst.grid_size, (2, 2));
    }

    /// Encoder parity against the reference: `llama-mtmd-debug -p encode
    /// --image cb -n 64` with `MTMD_DEBUG_EMBEDDINGS=<path>` on the same
    /// synthetic mmproj dumps the reference's embeddings of the raw `cb`
    /// bitmap. Run via parity/cohere2v_parity.sh (see PARITY.md).
    #[test]
    #[ignore = "needs the synthetic mmproj + a reference dump"]
    fn clip_cb_parity_dump_cohere2v() {
        let size = std::env::var("MTMD_CB_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let fa = match std::env::var("MTMD_FA").as_deref() {
            Ok("off") => ClipFlashAttn::Disabled,
            Ok("on") => ClipFlashAttn::Enabled,
            _ => ClipFlashAttn::Auto,
        };
        let mmproj = std::env::var("MTMD_COHERE2V_MMPROJ").unwrap_or_else(|_| {
            let dir = std::env::temp_dir().join("llama-rust-mtmd");
            dir.join("synth-mmproj-cohere2v.gguf")
                .to_str()
                .unwrap()
                .to_string()
        });
        write_synth_mmproj_cohere2v(&mmproj);
        let params = ClipContextParams {
            flash_attn_type: fa,
            ..Default::default()
        };
        let mut cx = clip_init_from_file(&mmproj, &params).expect("load synth cohere2v");
        let img = debug_pattern_cb(size);
        let batch = ClipImageF32Batch {
            entries: vec![img],
            is_audio: false,
        };
        let embd = cx.image_batch_encode(&batch).expect("encode");
        let n_tokens = cx.n_output_tokens(&batch.entries[0]);
        let out = std::env::var("MTMD_CB_OUT")
            .unwrap_or_else(|_| format!("/tmp/rust_cb_cohere2v_{size}.bin"));
        write_embedding_dump(&out, &embd, n_tokens, cx.n_mmproj_embd()).unwrap();
        eprintln!(
            "wrote {out} ({n_tokens} tokens x {} embd)",
            cx.n_mmproj_embd()
        );
    }

    /// Encoder parity against the reference: `llama-mtmd-debug -p encode
    /// --image cb -n <size>` with `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the
    /// embeddings of the same raw `cb` bitmap.
    ///
    /// Run: `cargo test -p llama --release -- --ignored clip_cb_parity` after
    /// producing the reference dump (see parity/mtmd_parity.sh).
    #[test]
    #[ignore = "needs the 1.8 GB mmproj + a reference dump"]
    fn clip_cb_parity_dump() {
        let size = std::env::var("MTMD_CB_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(448);
        let fa = match std::env::var("MTMD_FA").as_deref() {
            Ok("off") => ClipFlashAttn::Disabled,
            Ok("on") => ClipFlashAttn::Enabled,
            _ => ClipFlashAttn::Auto,
        };
        let params = ClipContextParams {
            flash_attn_type: fa,
            ..Default::default()
        };
        let mut cx = clip_init_from_file(MMPROJ_JACKRONG, &params).expect("load mmproj");
        let img = debug_pattern_cb(size);
        let batch = ClipImageF32Batch {
            entries: vec![img],
            is_audio: false,
        };
        let embd = cx.image_batch_encode(&batch).expect("encode");
        let n_tokens = cx.n_output_tokens(&batch.entries[0]);
        let out = format!("/tmp/rust_cb_{size}.bin");
        write_embedding_dump(&out, &embd, n_tokens, cx.n_mmproj_embd()).unwrap();
        eprintln!(
            "wrote {out} ({n_tokens} tokens x {} embd)",
            cx.n_mmproj_embd()
        );
    }
}

// ======================================================================
// audio round 5 — the gen-audio driver surface
//
// clip.cpp:4427-4620 (clip_encode's gen branches) + clip.cpp:5174-5337
// (set_inputs' QWEN3TTS_GEN / POCKETTTS_GEN arms) + clip.cpp:5800-5865 (the
// out_codes / out_feats / out_eos_score / out_audio / state_out read-back),
// driven through the same graph builders the reference dispatches to
// (clip.cpp:1104-1124): build_qwen3tts_gen_graph / build_pockettts_gen_graph.
// ======================================================================

/// `std::mt19937` (clip.cpp:178) — the gen-audio draw source. The reference
/// seeds it from `params->seed` (clip.cpp:4442-4445, UINT32_MAX = entropy),
/// and every draw the graphs consume comes through
/// `std::uniform_real_distribution<float>` (clip.cpp:5330) or
/// `std::normal_distribution<float>` (clip.cpp:5186). Both libstdc++
/// distributions are ported bit-exactly, so `seed == reference seed` ⇒
/// identical `inp_rand_*` / `inp_noise` values on both sides.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    /// `std::random_device{}()` seeding — entropy via a thread-local counter
    /// + address hash (the values only need to differ, never to match).
    pub fn seeded_from_entropy() -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut h = RandomState::new().build_hasher();
        h.write_u64(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0));
        h.write_usize(&mut 0usize as *const _ as usize);
        Self::new(h.finish() as u32)
    }

    /// `mt19937::seed(s)` — init_genrand (mt19937 is word-standard; the
    /// 1812433253 multiplier chain and the tempering below are the
    /// 32-bit Mersenne Twister constants).
    pub fn new(seed: u32) -> Self {
        let mut s = [0u32; 624];
        s[0] = seed;
        for i in 1..624 {
            let prev = s[i - 1];
            s[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state: s, idx: 624 }
    }

    pub fn seed(&mut self, s: u32) {
        *self = Self::new(s);
    }

    /// `mt19937::operator()` — twist + temper.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000)
                    | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= 0x9908_b0df;
                }
                self.state[i] = next;
            }
            self.idx = 0;
        }
        let mut y = self.state[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `std::uniform_real_distribution<float>(0,1)` over mt19937 —
    /// `__generate_canonical<float, 24>` with a 32-bit generator is one
    /// draw: `x / 2^32` rounded once to float (the long-double intermediate
    /// is exact for a power-of-two divisor).
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / 4294967296.0) as f32
    }

    /// `std::normal_distribution<float>(mean, stddev)` — libstdc++'s
    /// Marsaglia polar method, float arithmetic throughout, one saved
    /// variate per rejected-pair accept (the distribution object state).
    pub fn normal_f32(&mut self, mean: f32, stddev: f32, saved: &mut Option<f32>) -> f32 {
        if let Some(v) = saved.take() {
            return v * stddev + mean;
        }
        loop {
            let x = 2.0f32 * self.uniform_f32() - 1.0;
            let y = 2.0f32 * self.uniform_f32() - 1.0;
            let r2 = x * x + y * y;
            if r2 > 1.0 || r2 == 0.0 {
                continue;
            }
            let mult = (-2.0f32 * r2.ln() / r2).sqrt();
            *saved = Some(x * mult);
            return y * mult * stddev + mean;
        }
    }
}

/// the gen outputs are small tensors scattered mid-graph whose arena room a
/// later op's scratch can reuse — the C protects exactly these with
/// `ggml_set_output` (qwen3tts-gen.cpp:715/750-756 marks out_codes /
/// out_audio / every state_out_*), which the port's Graph has no equivalent
/// of. The port instead captures every completed node's bytes inside the
/// eval callback (the established MTMD_DEBUG_NODES_BIN lesson: post-hoc
/// walks read reused arena memory) and the collect step resolves each
/// output by its data pointer.
type GenCaptureMap = std::collections::HashMap<usize, Vec<u8>>;
static GEN_CAPTURE: std::sync::OnceLock<std::sync::Mutex<GenCaptureMap>> =
    std::sync::OnceLock::new();

/// the capture-all shortcut's memory guard: the C marks only designated
/// outputs with `ggml_set_output` and never copies intermediates, so a huge
/// or malformed intermediate tensor (2026-09-28: one 386 MiB node ballooned
/// the map to 68 GiB and tripped the global OOM killer, taking unrelated
/// processes with it) must simply not be captured — `gen_captured_bytes`
/// then takes its live-storage-read fallback for that tensor.
///
/// Root cause of that 386 MiB figure (same-day audit): the eval callback's
/// `data` slice spans from the node's view start to the END of the whole
/// arena (`Cpu::rd_rest`, rest = arena_len - offset — ggml-cpu hands the
/// callback the same raw `t->data` tail), and the no-reuse arena of the
/// qwen3tts GEN_WAV graph legitimately reaches ~412 MiB (1121 nodes; the
/// largest single node is an 8 MiB upsample-weight transpose). Copying
/// `data.to_vec()` per node therefore copied arena tails, not tensors —
/// O(n²) bytes in total. The fix below copies exactly `ggml_nbytes(node)`
/// bytes (the C formula, ne/nb/type), so this budget now only ever skips
/// genuinely huge tensors or true budget exhaustion.
static GEN_CAPTURE_BUDGET: std::sync::atomic::AtomicIsize =
    std::sync::atomic::AtomicIsize::new(GEN_CAPTURE_TOTAL_MAX);
/// single-node cap: real gen outputs are KBs–MBs (audio windows, states)
const GEN_CAPTURE_NODE_MAX: usize = 32 << 20;
/// per-compute total cap across all captured nodes
const GEN_CAPTURE_TOTAL_MAX: isize = 256 << 20;

/// `ggml_nbytes` (ggml.c:2269-2283) from the callback's view of a node.
fn eval_node_nbytes(node: &ggml::compute::EvalNode<'_>) -> usize {
    let blck = node.ty.blck_size();
    let tsz = node.ty.type_size();
    let mut n = if blck == 1 {
        tsz
    } else {
        node.ne[0] as usize * node.nb[0] as usize / blck
    };
    for i in if blck == 1 { 0 } else { 1 }..ggml::types::MAX_DIMS {
        n += (node.ne[i] as usize).saturating_sub(1) * node.nb[i] as usize;
    }
    n
}

fn gen_capture_clear() {
    GEN_CAPTURE
        .get_or_init(|| std::sync::Mutex::new(GenCaptureMap::default()))
        .lock()
        .unwrap()
        .clear();
    GEN_CAPTURE_BUDGET.store(GEN_CAPTURE_TOTAL_MAX, std::sync::atomic::Ordering::Relaxed);
}

fn gen_compute_captured(ctx: &mut Context, graph: &mut Graph, n_threads: usize) {
    gen_capture_clear();
    let cb = |node: &ggml::compute::EvalNode<'_>, ask: bool| -> bool {
        if ask {
            return true; // want every node
        }
        if let Some(data) = node.data {
            // only the node's OWN bytes — `data` itself runs to the end of
            // the whole arena (see GEN_CAPTURE_BUDGET's root-cause note)
            let own = eval_node_nbytes(node).min(data.len());
            // oversized or over-budget nodes are skipped, not captured —
            // see GEN_CAPTURE_BUDGET above for why this must be bounded
            if own > GEN_CAPTURE_NODE_MAX
                || GEN_CAPTURE_BUDGET.fetch_sub(own as isize, std::sync::atomic::Ordering::Relaxed)
                    < own as isize
            {
                if std::env::var_os("GEN_CAPTURE_DEBUG").is_some() {
                    eprintln!(
                        "gen_capture skip: op={:?} ne={:?} own={}",
                        node.op, node.ne, own
                    );
                }
                return true;
            }
            // keyed by the node's data pointer (view start); a later node on
            // the same storage overwrites the entry, which is exactly the
            // freshest-completed value semantics we need
            GEN_CAPTURE
                .get_or_init(|| std::sync::Mutex::new(GenCaptureMap::default()))
                .lock()
                .unwrap()
                .insert(data.as_ptr() as usize, data[..own].to_vec());
        }
        true
    };
    ggml::compute::set_eval_callback(Some(cb));
    ggml::compute::graph_compute(ctx, graph, n_threads);
    ggml::compute::set_eval_callback(None);
}

/// the bytes of `id` as captured during the compute pass (falls back to the
/// live storage read — e.g. when no callback ran)
fn gen_captured_bytes(ctx: &Context, id: TensorId) -> Option<Vec<u8>> {
    let nbytes = ctx.nbytes(id);
    let from_capture = (|| {
        let bytes = ctx.data_bytes(id)?;
        let key = bytes.as_ptr() as usize;
        GEN_CAPTURE
            .get_or_init(|| std::sync::Mutex::new(GenCaptureMap::default()))
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
    })();
    // the captured entry spans the node's whole backing storage (EvalNode
    // hands out the parent slice for views) — trim to the tensor's own bytes
    match from_capture {
        Some(v) => Some(v[..nbytes.min(v.len())].to_vec()),
        None => ctx.data_bytes(id).map(|b| b[..nbytes.min(b.len())].to_vec()),
    }
}

/// `mtmd_gen_inp` (mtmd.h:392) — the caller-facing process input.
#[derive(Clone, Copy)]
pub struct GenInp<'a> {
    pub process: ClipGenProcess,
    // GEN_CODE (mtmd.h:395-401)
    pub code0: i32,
    pub embd: Option<&'a [f32]>,
    pub top_k: i32,
    pub top_p: f32,
    /// UINT32_MAX = entropy (reseed); a fixed seed reproduces the reference
    /// draw-for-draw (clip.cpp:4442-4445 reseeds only on change)
    pub seed: u32,
    /// noise scale for flow-matching decoders (mtmd.h:401)
    pub temp: f32,
    // GEN_WAV (mtmd.h:403-410) — exactly one of codes / feats
    pub codes: Option<&'a [i32]>,
    pub feats: Option<&'a [f32]>,
    pub state: Option<&'a [u8]>,
}

/// `mtmd_gen_out` (mtmd.h:413) — owned (the C points into ctx-owned buffers
/// valid until the next call; the port returns the vectors).
#[derive(Default)]
pub struct GenOut {
    pub codes: Vec<i32>,
    pub feats: Vec<f32>,
    pub embd: Vec<f32>,
    pub is_eos: bool,
    pub audio: Vec<f32>,
    pub state: Vec<u8>,
}

impl ClipContext {
    /// clip.cpp:4442-4445 — reseed only when the seed value changes, so the
    /// same seed threaded through a whole generation keeps one stream.
    fn gen_reseed(&mut self, seed: u32) {
        if seed != self.rng_seed {
            self.rng_seed = seed;
            self.rng.seed(seed);
        }
    }

    /// `mtmd_gen_audio_process` (mtmd.cpp:1922-2031) over the gen mmproj —
    /// the GEN_CODE step (backbone hidden state → codes/feats/eos + the
    /// feedback embd) and the GEN_WAV step (codes/feats window + state →
    /// PCM + next state) of both generators.
    pub fn gen_audio_process(&mut self, inp: &GenInp) -> Result<GenOut, String> {
        if self.model.modality != ClipModality::GenAudio {
            return Err("model does not support audio generation".to_string());
        }
        let mut out = GenOut::default();

        // clip.cpp:4442 — the seed reaches the rng before any input is set
        self.gen_reseed(inp.seed);

        let ClipContext {
            ctx,
            model,
            watermark,
            flash_attn_type,
            n_threads,
            rng,
            ..
        } = self;

        ctx.reset_graph_to(*watermark);

        let proj = model.proj_type;
        if !matches!(proj, ProjectorType::Qwen3TtsGen | ProjectorType::PocketTtsGen) {
            return Err(format!("projector {proj:?} has no gen-audio pipeline"));
        }

        // ---- set_inputs (clip.cpp:5174-5337) ------------------------------
        // every input tensor is filled before the graph is built on the port
        // (same values, same shapes as the C's ggml_backend_tensor_set)
        match proj {
            ProjectorType::Qwen3TtsGen => {
                if inp.process == ClipGenProcess::GenWav {
                    let codes = inp.codes.ok_or("gen_wav (qwen3tts) requires codes")?;
                    let n_codes = ctx.ne(model.gen_code_head_w.unwrap())[2] + 1;
                    let n_frames_w = model.hparams.wav_tfm_swa as i64;
                    let n_frames = codes.len() as i64 / n_codes;
                    if n_frames == 0 || n_frames > n_frames_w {
                        return Err(format!(
                            "gen_wav: {n_frames} frames outside 1..={n_frames_w}"
                        ));
                    }
                    // codes are ggml_get_rows indices — bound them against
                    // the codebook vocab (clip.cpp:5296-5309)
                    let vocab_first = ctx.ne(model.c2w.as_ref().unwrap().quant_first_cb_w.unwrap())[1];
                    let vocab_rest = ctx.ne(model.c2w.as_ref().unwrap().quant_rest_cb_w.unwrap())[1];
                    for (idx, &c) in codes.iter().enumerate() {
                        let g = (idx as i64) % n_codes;
                        let vocab = if g == 0 { vocab_first } else { vocab_rest };
                        if c < 0 || c as i64 >= vocab {
                            return Err(format!(
                                "code out of range (frame {}, group {g}, code {c}, vocab {vocab})",
                                (idx as i64) / n_codes
                            ));
                        }
                    }
                    let (graph, io) = build_qwen3tts_gen_graph(
                        ctx,
                        model,
                        inp.process,
                        inp.top_k,
                        inp.top_p,
                        *flash_attn_type,
                    )?;
                    let mut graph = graph;

                    // frame-major input to group-major, rear-padded with
                    // code 0 up to one window (clip.cpp:5311-5317)
                    let mut codes_w = vec![0i32; (n_frames_w * n_codes) as usize];
                    for f in 0..n_frames {
                        for g in 0..n_codes {
                            codes_w[(g * n_frames_w + f) as usize] = codes[(f * n_codes + g) as usize];
                        }
                    }
                    let t_codes = io.inp_codes.expect("inp_codes");
                    ctx.arena_resize_tensor(t_codes);
                    ctx.with_i32_mut(t_codes, |p| p.copy_from_slice(&codes_w))
                        .expect("inp_codes is an arena tensor");

                    // state_in slots (clip.cpp:4495-4503)
                    let mut offset = 0usize;
                    for (name, t) in &io.state_in {
                        let nb = (ctx.ne(*t).iter().product::<i64>() * 4) as usize;
                        if let Some(state) = inp.state {
                            if state.len() < offset + nb {
                                return Err(format!(
                                    "gen_wav: state blob too short (have {}, need {}+{})",
                                    state.len(), offset, nb
                                ));
                            }
                            ctx.arena_resize_tensor(*t);
                            let src: &[f32] =
                                bytemuck::cast_slice(&state[offset..offset + nb]);
                            ctx.with_f32_mut(*t, |p| p.copy_from_slice(src))
                                .expect("state_in is an arena tensor");
                        } else {
                            // cold start: zero-filled (ggml_set_input zeroes)
                            ctx.arena_resize_tensor(*t);
                            ctx.with_f32_mut(*t, |p| p.fill(0.0))
                                .expect("state_in is an arena tensor");
                        }
                        let _ = name;
                        offset += nb;
                    }

                    gen_compute_captured(ctx, &mut graph, *n_threads);
                    return Self::gen_collect_outputs(ctx, model, &io, Some(n_frames), ClipGenProcess::GenWav);
                }
                // GEN_CODE (clip.cpp:5319-5335)
                let embd = inp.embd.ok_or("gen_code requires the hidden state")?;
                let code0 = inp.code0;
                let vocab0 = ctx.ne(model.gen_code_out_embd_w.unwrap())[1];
                if code0 < 0 || code0 as i64 >= vocab0 {
                    return Err(format!("code0 out of range ({code0}, vocab {vocab0})"));
                }
                let (graph, io) = build_qwen3tts_gen_graph(
                    ctx,
                    model,
                    inp.process,
                    inp.top_k,
                    inp.top_p,
                    *flash_attn_type,
                )?;
                let mut graph = graph;

                let t_raw = io.h_state.expect("inp_raw");
                ctx.arena_resize_tensor(t_raw);
                if embd.len() != ctx.ne(t_raw).iter().product::<i64>() as usize {
                    return Err("gen_code: embd length != n_mmproj_embd".to_string());
                }
                ctx.with_f32_mut(t_raw, |p| p.copy_from_slice(embd))
                    .expect("inp_raw is an arena tensor");

                let t_code0 = io.inp_code0.expect("inp_code0");
                ctx.arena_resize_tensor(t_code0);
                ctx.with_i32_mut(t_code0, |p| p[0] = code0)
                    .expect("inp_code0 is an arena tensor");

                // one uniform(0,1) draw per acoustic codebook
                // (clip.cpp:5329-5335)
                for t in &io.inp_rands {
                    let r = rng.uniform_f32();
                    ctx.arena_resize_tensor(*t);
                    ctx.with_f32_mut(*t, |p| p[0] = r)
                        .expect("inp_rand is an arena tensor");
                }

                gen_compute_captured(ctx, &mut graph, *n_threads);
                return Self::gen_collect_outputs(ctx, model, &io, None, inp.process);
            }
            ProjectorType::PocketTtsGen => {
                if inp.process == ClipGenProcess::GenWav {
                    let feats = inp.feats.ok_or("gen_wav (pockettts) requires feats")?;
                    let n_latent = ctx.ne(model.gen_input_lin_w.unwrap())[0];
                    let n_frames = feats.len() as i64 / n_latent;
                    let (graph, io) = build_pockettts_gen_graph(
                        ctx,
                        model,
                        n_latent,
                        n_frames,
                        inp.process,
                        model.hparams.flow_n_step.max(1),
                        n_frames,
                        *flash_attn_type,
                    )?;
                    let mut graph = graph;

                    let t_feats = io.inp_feats.expect("inp_feats");
                    ctx.arena_resize_tensor(t_feats);
                    ctx.with_f32_mut(t_feats, |p| p.copy_from_slice(feats))
                        .expect("inp_feats is an arena tensor");

                    let mut offset = 0usize;
                    for (_, t) in &io.state_in {
                        let nb = (ctx.ne(*t).iter().product::<i64>() * 4) as usize;
                        if let Some(state) = inp.state {
                            if state.len() < offset + nb {
                                return Err(format!(
                                    "gen_wav: state blob too short (have {}, need {}+{})",
                                    state.len(), offset, nb
                                ));
                            }
                            ctx.arena_resize_tensor(*t);
                            let src: &[f32] =
                                bytemuck::cast_slice(&state[offset..offset + nb]);
                            ctx.with_f32_mut(*t, |p| p.copy_from_slice(src))
                                .expect("state_in is an arena tensor");
                        } else {
                            ctx.arena_resize_tensor(*t);
                            ctx.with_f32_mut(*t, |p| p.fill(0.0))
                                .expect("state_in is an arena tensor");
                        }
                        offset += nb;
                    }

                    ggml::compute::graph_compute(ctx, &mut graph, *n_threads);
                    return Self::gen_collect_outputs(ctx, model, &io, None, inp.process);
                }
                // GEN_CODE (clip.cpp:5181-5192)
                let embd = inp.embd.ok_or("gen_code requires the hidden state")?;
                let n_latent = ctx.ne(model.gen_input_lin_w.unwrap())[0];
                let n_state = ctx.ne(model.gen_input_lin_w.unwrap())[1];
                let n_frames = 1i64;
                let (graph, io) = build_pockettts_gen_graph(
                    ctx,
                    model,
                    embd.len() as i64,
                    1,
                    inp.process,
                    model.hparams.flow_n_step.max(1),
                    n_frames,
                    *flash_attn_type,
                )?;
                let mut graph = graph;

                let t_raw = io.h_state.expect("inp_raw");
                ctx.arena_resize_tensor(t_raw);
                if embd.len() != (n_state as usize) {
                    return Err("gen_code: embd length != n_mmproj_embd".to_string());
                }
                ctx.with_f32_mut(t_raw, |p| p.copy_from_slice(embd))
                    .expect("inp_raw is an arena tensor");

                // flow matching starts from gaussian noise, std = sqrt(temp)
                // (clip.cpp:5182-5191); Config.default_temperature = 0.7
                let t_noise = io.inp_noise.expect("inp_noise");
                ctx.arena_resize_tensor(t_noise);
                let temp = if inp.temp > 0.0 { inp.temp } else { 0.7 };
                let mut saved: Option<f32> = None;
                ctx.with_f32_mut(t_noise, |p| {
                    for v in p.iter_mut() {
                        *v = rng.normal_f32(0.0, temp.sqrt(), &mut saved);
                    }
                })
                .expect("inp_noise is an arena tensor");
                let _ = n_latent;

                gen_compute_captured(ctx, &mut graph, *n_threads);
                return Self::gen_collect_outputs(ctx, model, &io, None, inp.process);
            }
            _ => unreachable!("checked above"),
        }
    }

    /// clip.cpp:5800-5865 — the gen outputs read back from the graph.
    /// `qwen3tts_frames` carries the caller's real frame count for the
    /// code-0 rear-padding tail drop (clip.cpp:5839-5846).
    fn gen_collect_outputs(
        ctx: &mut Context,
        model: &ClipModel,
        io: &GenGraphIo,
        qwen3tts_frames: Option<i64>,
        process: ClipGenProcess,
    ) -> Result<GenOut, String> {
        let mut out = GenOut::default();
        // only the outputs the process expands are materialized
        // (clip_encode reads them back by name, absent ⇒ skip,
        // clip.cpp:5805-5820's nullptr guards)
        let (want_code, want_wav) = match process {
            ClipGenProcess::GenCode => (true, false),
            ClipGenProcess::GenWav => (false, true),
        };

        // out_codes (clip.cpp:5805-5812)
        if want_code {
            if let Some(t) = io.out_codes {
                if let Some(v) = gen_captured_bytes(ctx, t) {
                    out.codes = bytemuck::cast_slice(&v).to_vec();
                }
            }
        }
        // out_feats (clip.cpp:5813-5820)
        if want_code {
            if let Some(t) = io.out_feats {
                if let Some(v) = gen_captured_bytes(ctx, t) {
                    out.feats = bytemuck::cast_slice(&v).to_vec();
                }
            }
        }
        // out_eos_score (clip.cpp:5821-5829) — score > gen_eos_threshold
        if want_code {
            if let Some(t) = io.out_eos_score {
                if let Some(v) = gen_captured_bytes(ctx, t) {
                    let s: &[f32] = bytemuck::cast_slice(&v);
                    out.is_eos = s[0] > model.hparams.gen_eos_threshold;
                }
            }
        }
        // out_audio (clip.cpp:5830-5847) — plus the qwen3tts tail drop
        if want_wav {
            let Some(t) = io.out_audio else {
                return Err("out_audio requested but graph has no \"out_audio\" tensor".to_string())
            };
            let audio: Vec<f32> = gen_captured_bytes(ctx, t)
                .ok_or("out_audio not materialized")?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            out.audio = audio;
            if let Some(n_frames) = qwen3tts_frames {
                let n_frames_w = model.hparams.wav_tfm_swa as i64;
                if n_frames < n_frames_w {
                    let hop = out.audio.len() as i64 / n_frames_w;
                    out.audio.truncate((n_frames * hop) as usize);
                }
            }
        }
        // state_out (clip.cpp:5848-5865) — written in list_gen_state_slots
        // order (the slot-list order, NOT the graph's push order — the C
        // walks `list_gen_state_slots` and looks each name up in the graph,
        // clip.cpp:5856-5862)
        if want_wav && !io.state_out.is_empty() {
            let slots: Vec<(String, TensorId)> = {
                let by_name: std::collections::HashMap<String, TensorId> =
                    io.state_out.iter().cloned().collect();
                let list = match model.proj_type {
                    ProjectorType::Qwen3TtsGen => list_c2w_state_slots(model, ctx),
                    ProjectorType::PocketTtsGen => list_pockettts_state_slots(model, ctx),
                    _ => Vec::new(),
                };
                list.into_iter()
                    .map(|slot| {
                        let t = by_name
                            .get(&slot.name)
                            .copied()
                            .unwrap_or_else(|| panic!("state_out_{} missing from the graph", slot.name));
                        (slot.name, t)
                    })
                    .collect()
            };
            let mut total = 0usize;
            for (_, t) in &slots {
                total += (ctx.ne(*t).iter().product::<i64>() * 4) as usize;
            }
            out.state.resize(total, 0);
            let mut offset = 0usize;
            for (_, t) in &slots {
                let nb = (ctx.ne(*t).iter().product::<i64>() * 4) as usize;
                let src = gen_captured_bytes(ctx, *t)
                    .ok_or("state_out not materialized")?;
                out.state[offset..offset + nb].copy_from_slice(&src[..nb]);
                offset += nb;
            }
        }
        // out_embd — the feedback embedding for the backbone
        // (clip_encode reads it as the graph's last node in C,
        // qwen3tts-gen.cpp:758; the port reads the named tensor)
        if want_code {
            if let Some(t) = io.out_embd {
                if let Some(v) = gen_captured_bytes(ctx, t) {
                    out.embd = v
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                }
            }
        }
        Ok(out)
    }

    /// `mtmd_gen_audio_get_info` (mtmd.cpp:1871-1893).
    pub fn gen_audio_sample_rate(&self) -> i32 {
        match self.model.proj_type {
            ProjectorType::Qwen3TtsGen | ProjectorType::PocketTtsGen => 24000,
            _ => 0,
        }
    }

    /// `mtmd_gen_inp_default`'s per-pipeline sampling defaults
    /// (mtmd.cpp:1903-1918).
    pub fn gen_inp_defaults(&self) -> (i32, f32, f32) {
        match self.model.proj_type {
            // Qwen3-TTS-12Hz generation_config.json
            ProjectorType::Qwen3TtsGen => (50, 1.0, 0.9),
            // pocket-tts default_parameters.py
            ProjectorType::PocketTtsGen => (50, 1.0, 0.7),
            _ => (0, 1.0, 0.0),
        }
    }
}
