//! mtmd_audio.rs — literal port of `tools/mtmd/mtmd-audio.cpp` (whisper-style
//! FFT/log-mel preprocessing for the audio multimodal projectors) plus the
//! `mtmd_audio_cache` / preprocessor declarations of `tools/mtmd/mtmd-audio.h`.
//!
//! Mapping (C++ → Rust):
//!   * `mtmd_audio_mel`            → [`AudioMel`]
//!   * `mtmd_audio_mel_filters`    → [`AudioMelFilters`]
//!   * `mtmd_audio_cache`          → [`AudioCache`] (`fill_sin_cos_table`,
//!     `fill_hann_window`, `fill_mel_filterbank_matrix`)
//!   * `dft_impl`/`fft_impl` (templates) → [`dft_impl`]/[`fft_impl`] with
//!     const bools; `fft`/`ifft` keep their names
//!   * `filter_params`             → [`FilterParams`]
//!   * `log_mel_spectrogram_worker_thread` / `log_mel_spectrogram` →
//!     [`log_mel_spectrogram_worker`] / [`log_mel_spectrogram`]
//!   * `mtmd_audio_preprocessor_whisper`      → [`WhisperPreproc`]
//!   * `mtmd_audio_preprocessor_qwen3a`       → [`Qwen3aPreproc`]
//!   * `mtmd_audio_preprocessor_dots3note`    → [`Dots3NotePreproc`]
//!   * `mtmd_audio_preprocessor_mimo_audio`   → [`MimoAudioPreproc`]
//!   * `mtmd_audio_preprocessor_qwen3tts_spk` → [`Qwen3TtsSpkPreproc`]
//!   * `mtmd_audio_preprocessor_conformer`    → [`ConformerPreproc`]
//!   * `mtmd_audio_preprocessor_granite_speech` → [`GraniteSpeechPreproc`]
//!   * `mtmd_audio_preprocessor_gemma4a`      → [`Gemma4aPreproc`]
//!   * `mtmd_audio_preprocessor_gemma4ua`     → [`Gemma4uaPreproc`]
//!   * `mtmd_audio_preprocessor_parakeet`     → [`ParakeetPreproc`]
//!   * `mtmd_audio_preprocessor_pockettts`    → [`PocketTtsPreproc`]
//!   * `mtmd_audio_streaming_istft`           → [`StreamingIstdft`]
//!
//! Deviation (documented): the C++ worker threads split the mel frames across
//! 4 threads (`n_threads = 4`); every frame is computed independently and
//! written by exactly one thread, so the output does not depend on the thread
//! count — the port runs the same frame loop sequentially.

/// `mtmd_audio_mel` (mtmd-audio.h:12-18)
#[derive(Clone, Default, Debug)]
pub struct AudioMel {
    pub n_len: i64,
    pub n_len_org: i64,
    pub n_mel: i64,
    pub data: Vec<f32>,
}

/// `mtmd_audio_mel_filters` (mtmd-audio.h:20-25)
#[derive(Clone, Default, Debug)]
pub struct AudioMelFilters {
    pub n_mel: i64,
    pub n_fft: i64,
    pub data: Vec<f32>,
}

/// `mtmd_audio_cache` (mtmd-audio.h:28-51)
#[derive(Clone, Default, Debug)]
pub struct AudioCache {
    pub sin_vals: Vec<f32>,
    pub cos_vals: Vec<f32>,
    pub hann_window: Vec<f32>,
    pub filters: AudioMelFilters,
}

impl AudioCache {
    /// `fill_sin_cos_table` (mtmd-audio.cpp:17-25)
    pub fn fill_sin_cos_table(&mut self, n: u32) {
        self.sin_vals = vec![0.0; n as usize];
        self.cos_vals = vec![0.0; n as usize];
        for i in 0..n as usize {
            // double theta = (2 * M_PI * i) / n;  sin_vals[i] = sinf(theta)
            // (sinf/cosf take float — glibc's f32 routines, not the f64 ones)
            let theta = ((2.0 * std::f64::consts::PI * i as f64) / n as f64) as f32;
            self.sin_vals[i] = theta.sin();
            self.cos_vals[i] = theta.cos();
        }
    }

    /// `fill_hann_window` (mtmd-audio.cpp:27-33)
    pub fn fill_hann_window(&mut self, length: u32, periodic: bool) {
        self.hann_window = vec![0.0; length as usize];
        let offset: i32 = if periodic { 0 } else { -1 };
        for i in 0..length as usize {
            // cosf takes the double expression narrowed to float, the outer
            // 0.5*(1.0 - .) runs in double, one narrowing store (mtmd-audio.cpp:31)
            let arg =
                ((2.0 * std::f64::consts::PI * i as f64) / (length as f64 + offset as f64)) as f32;
            self.hann_window[i] = (0.5 * (1.0 - arg.cos() as f64)) as f32;
        }
    }

    /// `fill_mel_filterbank_matrix` (mtmd-audio.cpp:35-130). Default arguments
    /// mirror the header: fmin 0.0, fmax -1 (auto), slaney_area_norm true,
    /// scale 1.0, use_htk false.
    pub fn fill_mel_filterbank_matrix(
        &mut self,
        n_mel: i64,
        n_fft: i64,
        sample_rate: i32,
        fmin: f32,
        fmax: f32,
        slaney_area_norm: bool,
        scale: f32,
        use_htk: bool,
    ) {
        assert!(n_mel > 0 && n_fft > 1);
        let mut fmax = fmax;
        if fmax <= 0.0 {
            fmax = 0.5 * sample_rate as f32;
        }

        // hz_to_mel / mel_to_hz (mtmd-audio.cpp:48-70)
        if use_htk {
            let hz: Box<dyn Fn(f64) -> f64> =
                Box::new(|f_hz| 2595.0 * (1.0 + f_hz / 700.0).log10());
            let mel: Box<dyn Fn(f64) -> f64> = Box::new(|m| 700.0 * (10f64.powf(m / 2595.0) - 1.0));
            self.fill_mel_filterbank_inner(
                n_mel,
                n_fft,
                sample_rate,
                fmin as f64,
                fmax as f64,
                slaney_area_norm,
                scale,
                hz.as_ref(),
                mel.as_ref(),
            );
        } else {
            // Slaney scale (matches librosa default)
            let min_log_hz = 1000.0;
            let lin_slope = 3.0 / 200.0;
            let min_log_mel = min_log_hz * lin_slope;
            let log_step = 6.4f64.ln() / 27.0;
            let hz: Box<dyn Fn(f64) -> f64> = Box::new(move |f_hz: f64| {
                if f_hz < min_log_hz {
                    f_hz * lin_slope
                } else {
                    min_log_mel + (f_hz / min_log_hz).ln() / log_step
                }
            });
            let mel: Box<dyn Fn(f64) -> f64> = Box::new(move |m: f64| {
                if m < min_log_mel {
                    m / lin_slope
                } else {
                    min_log_hz * ((m - min_log_mel) * log_step).exp()
                }
            });
            self.fill_mel_filterbank_inner(
                n_mel,
                n_fft,
                sample_rate,
                fmin as f64,
                fmax as f64,
                slaney_area_norm,
                scale,
                hz.as_ref(),
                mel.as_ref(),
            );
        }
    }

    fn fill_mel_filterbank_inner(
        &mut self,
        n_mel: i64,
        n_fft: i64,
        sample_rate: i32,
        fmin: f64,
        fmax: f64,
        slaney_area_norm: bool,
        scale: f32,
        hz_to_mel: &dyn Fn(f64) -> f64,
        mel_to_hz: &dyn Fn(f64) -> f64,
    ) {
        // infer N_fft from n_fft_bins
        let bin_hz_step = sample_rate as f64 / n_fft as f64;

        // mel grid: n_mel + 2 edges
        let m_lo = hz_to_mel(fmin);
        let m_hi = hz_to_mel(fmax);
        let mut mel_pts = vec![0.0f64; (n_mel + 2) as usize];
        for (i, pt) in mel_pts.iter_mut().enumerate() {
            *pt = m_lo + (m_hi - m_lo) * (i as f64 / (n_mel + 1) as f64);
        }

        // convert to Hz
        let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

        let n_fft_bins = n_fft / 2 + 1;

        // Validate allocation size
        assert!(
            (n_mel as usize) * (n_fft_bins as usize) <= usize::MAX,
            "mel filterbank allocation too large"
        );

        // filterbank
        let mut out = vec![0.0f32; (n_mel * n_fft_bins) as usize];
        for m in 0..n_mel {
            let f_left = hz_pts[m as usize];
            let f_center = hz_pts[m as usize + 1];
            let f_right = hz_pts[m as usize + 2];

            let denom_l = (f_center - f_left).max(1e-30);
            let denom_r = (f_right - f_center).max(1e-30);
            let enorm = if slaney_area_norm {
                2.0 / (f_right - f_left).max(1e-30)
            } else {
                1.0
            };

            for k in 0..n_fft_bins {
                let f = k as f64 * bin_hz_step;
                let mut w = 0.0;
                if f >= f_left && f <= f_center {
                    w = (f - f_left) / denom_l;
                } else if f > f_center && f <= f_right {
                    w = (f_right - f) / denom_r;
                }
                out[(m * n_fft_bins + k) as usize] = (w * enorm * scale as f64) as f32;
            }
        }

        self.filters.n_mel = n_mel;
        self.filters.n_fft = n_fft;
        self.filters.data = out;
    }
}

// ---------------------------------------------------------------------------
// DFT / FFT (mtmd-audio.cpp:132-271)
// ---------------------------------------------------------------------------

/// Unified DFT implementation for both forward and inverse transforms
/// (mtmd-audio.cpp:138-174). `inverse` selects exp(±2πi·k·n/N) and the 1/N
/// scale; `real_input` takes a stride-1 real input instead of interleaved.
fn dft_impl(
    cache: &AudioCache,
    input: &[f32],
    in_off: usize,
    n: i32,
    out: &mut [f32],
    out_off: usize,
    inverse: bool,
    real_input: bool,
) {
    let n_sin_cos_vals = cache.sin_vals.len() as i32;
    let sin_cos_step = n_sin_cos_vals / n;

    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    let scale: f32 = if inverse { 1.0 / n as f32 } else { 1.0 };

    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;

        for n_ in 0..n {
            let idx =
                ((k as i64 * n_ as i64 * sin_cos_step as i64) % n_sin_cos_vals as i64) as usize;
            let cos_val = cache.cos_vals[idx];
            let sin_val = cache.sin_vals[idx];

            if real_input {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                let in_re = input[in_off + n_ as usize];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                let in_re = input[in_off + n_ as usize * 2];
                let in_im = input[in_off + n_ as usize * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }
        }

        out[out_off + k as usize * 2] = re * scale;
        out[out_off + k as usize * 2 + 1] = im * scale;
    }
}

/// Cooley-Tukey FFT/IFFT unified implementation (mtmd-audio.cpp:182-261).
/// The C++ recurses with pointers into the same scratch buffers
/// (`even = in + N`, `even_fft = out + 2*N`, `odd_fft = even_fft + N`); the
/// offsets below reproduce that layout exactly.
fn fft_impl(
    cache: &AudioCache,
    input: &mut [f32],
    in_off: usize,
    n: i32,
    out: &mut [f32],
    out_off: usize,
    inverse: bool,
    real_input: bool,
) {
    assert!(n > 0);
    let n_sin_cos_vals = cache.sin_vals.len() as i32;

    if n == 1 {
        out[out_off] = input[in_off];
        if real_input {
            out[out_off + 1] = 0.0;
        } else {
            out[out_off + 1] = input[in_off + 1];
        }
        return;
    }

    let half_n = n / 2;
    if n - half_n * 2 == 1 {
        // Odd N: fall back to DFT
        dft_impl(cache, input, in_off, n, out, out_off, inverse, real_input);
        return;
    }

    // Split into even and odd
    if real_input {
        // Real input: stride is 1, copy only real values
        let even = in_off + n as usize;
        for i in 0..half_n as usize {
            input[even + i] = input[in_off + 2 * i];
        }
        let even_fft = out_off + 2 * n as usize;
        fft_impl(cache, input, even, half_n, out, even_fft, inverse, true);

        let odd = even;
        for i in 0..half_n as usize {
            input[odd + i] = input[in_off + 2 * i + 1];
        }
        let odd_fft = even_fft + n as usize;
        fft_impl(cache, input, odd, half_n, out, odd_fft, inverse, true);
    } else {
        // Complex input: stride is 2, copy complex pairs
        let even = in_off + n as usize * 2;
        for i in 0..half_n as usize {
            input[even + i * 2] = input[in_off + 2 * i * 2];
            input[even + i * 2 + 1] = input[in_off + 2 * i * 2 + 1];
        }
        let even_fft = out_off + 2 * n as usize;
        fft_impl(cache, input, even, half_n, out, even_fft, inverse, false);

        let odd = even;
        for i in 0..half_n as usize {
            input[odd + i * 2] = input[in_off + (2 * i + 1) * 2];
            input[odd + i * 2 + 1] = input[in_off + (2 * i + 1) * 2 + 1];
        }
        let odd_fft = even_fft + n as usize;
        fft_impl(cache, input, odd, half_n, out, odd_fft, inverse, false);
    }

    let even_fft = out_off + 2 * n as usize;
    let odd_fft = even_fft + n as usize;

    let sin_cos_step = n_sin_cos_vals / n;

    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    let scale: f32 = if inverse { 0.5 } else { 1.0 };

    for k in 0..half_n as usize {
        let idx = k * sin_cos_step as usize; // t = 2*M_PI*k/N
        let re = cache.cos_vals[idx];
        let im = sign * cache.sin_vals[idx];

        let re_odd = out[odd_fft + 2 * k];
        let im_odd = out[odd_fft + 2 * k + 1];

        out[out_off + 2 * k] = scale * (out[even_fft + 2 * k] + re * re_odd - im * im_odd);
        out[out_off + 2 * k + 1] = scale * (out[even_fft + 2 * k + 1] + re * im_odd + im * re_odd);

        out[out_off + 2 * (k + half_n as usize)] =
            scale * (out[even_fft + 2 * k] - re * re_odd + im * im_odd);
        out[out_off + 2 * (k + half_n as usize) + 1] =
            scale * (out[even_fft + 2 * k + 1] - re * im_odd - im * re_odd);
    }
}

/// Forward FFT for real input (used by mel spectrogram) — mtmd-audio.cpp:264-266
fn fft(
    cache: &AudioCache,
    input: &mut [f32],
    in_off: usize,
    n: i32,
    out: &mut [f32],
    out_off: usize,
) {
    fft_impl(cache, input, in_off, n, out, out_off, false, true);
}

/// Inverse FFT for complex input — mtmd-audio.cpp:269-271
fn ifft(
    cache: &AudioCache,
    input: &mut [f32],
    in_off: usize,
    n: i32,
    out: &mut [f32],
    out_off: usize,
) {
    fft_impl(cache, input, in_off, n, out, out_off, true, false);
}

/// `filter_params` (mtmd-audio.cpp:273-286)
#[derive(Clone, Copy)]
struct FilterParams {
    n_mel: i64,
    n_fft_bins: i64,
    hann_window_size: i32,
    hop_length: i32,
    sample_rate: i32,
    no_padding: bool,
    center_padding: bool,
    preemph: f32,
    use_natural_log: bool,
    norm_per_feature: bool,
    use_magnitude: bool, // |X| instead of |X|^2
    mel_floor: f32,
    mel_floor_add: bool,       // log(x + floor) instead of log(max(x, floor))
    std_eps_after_sqrt: bool,  // std + eps instead of sqrt(var + eps)
}

impl Default for FilterParams {
    fn default() -> Self {
        FilterParams {
            n_mel: 0,
            n_fft_bins: 0,
            hann_window_size: 0,
            hop_length: 0,
            sample_rate: 0,
            no_padding: false,
            center_padding: false,
            preemph: 0.0,
            use_natural_log: false,
            norm_per_feature: false,
            use_magnitude: false,
            mel_floor: 5.960464477539063e-08,
            mel_floor_add: false,
            std_eps_after_sqrt: false,
        }
    }
}

/// `log_mel_spectrogram_worker_thread` (mtmd-audio.cpp:288-365) — one slice of
/// the frame range (the C++ threads each take `i % n_threads`; every frame is
/// independent, so the port computes them in order).
fn log_mel_spectrogram_worker(
    hann: &[f32],
    samples: &[f32],
    n_samples: i32,
    frame_size: i32,
    frame_step: i32,
    params: &FilterParams,
    cache: &AudioCache,
    out: &mut AudioMel,
) {
    let mut fft_in = vec![0.0f32; frame_size as usize * 2];
    let mut fft_out = vec![0.0f32; frame_size as usize * 2 * 2 * 2];

    let n_fft_bins = params.n_fft_bins;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert!(n_fft_bins == 1 + (frame_size / 2) as i64);
    assert_eq!(cache.sin_vals.len(), cache.cos_vals.len());
    // calculate FFT only when fft_in are not all zero
    let mut i: i64 = 0;
    while i < (n_samples / frame_step + 1) as i64 && i < out.n_len {
        let offset = (i * frame_step as i64) as usize;

        // apply Hann window (~10% faster)
        let valid_len = frame_size.min(0.max(n_samples - offset as i32)) as usize;
        for j in 0..valid_len {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if valid_len < frame_size as usize {
            for x in fft_in[valid_len..].iter_mut() {
                *x = 0.0;
            }
        }

        // FFT
        fft(cache, &mut fft_in, 0, frame_size, &mut fft_out, 0);

        // Calculate modulus^2 (power) or modulus (magnitude)
        for j in 0..n_fft_bins as usize {
            let power = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
            fft_out[j] = if params.use_magnitude {
                power.sqrt()
            } else {
                power
            };
        }

        // mel spectrogram
        for j in 0..out.n_mel {
            let mut sum = 0.0f64;
            // unroll loop (suggested by GH user @lunixbochs); the C++ products
            // and adds inside each group run in f32, the += in f64
            let mut k = 0usize;
            while k + 3 < n_fft_bins as usize {
                let idx = j as usize * n_fft_bins as usize + k;
                let group = (fft_out[k] * cache.filters.data[idx])
                    + (fft_out[k + 1] * cache.filters.data[idx + 1])
                    + (fft_out[k + 2] * cache.filters.data[idx + 2])
                    + (fft_out[k + 3] * cache.filters.data[idx + 3]);
                sum += group as f64;
                k += 4;
            }
            // handle n_fft remainder
            while k < n_fft_bins as usize {
                let product = fft_out[k] * cache.filters.data[j as usize * n_fft_bins as usize + k];
                sum += product as f64;
                k += 1;
            }
            sum = if params.mel_floor_add {
                sum + params.mel_floor as f64
            } else {
                sum.max(params.mel_floor as f64)
            };
            sum = if params.use_natural_log {
                sum.ln()
            } else {
                sum.log10()
            };
            out.data[j as usize * out.n_len as usize + i as usize] = sum as f32;
        }
        i += 1;
    }

    // Otherwise fft_out are all zero
    let tail = if params.use_natural_log {
        (1e-10f64).ln()
    } else {
        (1e-10f64).log10()
    } as f32;
    while i < out.n_len {
        for j in 0..out.n_mel {
            out.data[j as usize * out.n_len as usize + i as usize] = tail;
        }
        i += 1;
    }
}

/// `log_mel_spectrogram` (mtmd-audio.cpp:368-538) — ref:
/// https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
fn log_mel_spectrogram(
    samples_in: &[f32],
    n_samples_in: i32,
    params: &FilterParams,
    cache: &AudioCache,
    out: &mut AudioMel,
) -> bool {
    out.n_len_org = n_samples_in as i64;
    let mut n_samples = n_samples_in;

    // Hann window
    let frame_size = (params.n_fft_bins - 1) as i32 * 2;
    let frame_step = params.hop_length;

    // Padding
    let mut samples_padded: Vec<f32>;
    if params.no_padding {
        // no padding, use samples as-is
        samples_padded = samples_in.to_vec();
    } else if params.center_padding {
        let pad_amount = (frame_size / 2) as usize;
        samples_padded = vec![0.0; n_samples as usize + 2 * pad_amount];
        samples_padded[pad_amount..pad_amount + n_samples as usize].copy_from_slice(samples_in);
    } else {
        // existing padding logic
        let stage_1_pad = params.sample_rate as i64 * 30;
        let stage_2_pad = (frame_size / 2) as i64;
        let total = n_samples as i64 + stage_1_pad + stage_2_pad * 2;
        samples_padded = vec![0.0; total as usize];
        let n = n_samples as usize;
        let s2 = stage_2_pad as usize;
        samples_padded[s2..s2 + n].copy_from_slice(&samples_in[..n]);
        // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
        for x in samples_padded[n + s2..(n_samples as i64 + stage_1_pad + 2 * stage_2_pad) as usize]
            .iter_mut()
        {
            *x = 0.0;
        }
        // reflective pad 200 samples at the beginning of audio
        if n_samples < stage_2_pad as i32 + 1 {
            // TODO: Handle short audio differently or return error
            return false;
        }
        // std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin()):
        // samples_padded[i] = samples[1 + (stage_2_pad - 1 - i)] for i in 0..stage_2_pad
        for i in 0..s2 {
            samples_padded[i] = samples_in[1 + (s2 - 1 - i)];
        }
    }
    n_samples = samples_padded.len() as i32;

    // preemphasis
    if params.preemph != 0.0 {
        let pad_amount = (frame_size / 2) as usize;
        let preemph = 0.97f32;
        let mut prev = samples_padded[pad_amount];
        let mut i = pad_amount + 1;
        while (i as i64) + (pad_amount as i64) < n_samples as i64 {
            let cur = samples_padded[i];
            samples_padded[i] = cur - preemph * prev;
            prev = cur;
            i += 1;
        }
    }

    let samples: &[f32] = samples_padded.as_slice();

    // pad hann window if it's smaller than frame_size
    // TODO: probably unnecessary here? (or better doing it in g_cache?)
    let mut hann_window_padded: Vec<f32>;
    let hann: &[f32] = if params.hann_window_size < frame_size {
        hann_window_padded = vec![0.0; frame_size as usize];
        let padding = ((frame_size - params.hann_window_size) / 2) as usize;
        hann_window_padded[padding..padding + params.hann_window_size as usize]
            .copy_from_slice(&cache.hann_window[..params.hann_window_size as usize]);
        hann_window_padded.as_slice()
    } else {
        &cache.hann_window
    };

    assert!(params.n_fft_bins > 0);
    assert!(params.hop_length > 0);
    out.n_mel = params.n_mel;
    out.n_len = ((n_samples - frame_size) / frame_step + 1) as i64;
    // Validate dimensions before allocation to prevent integer overflow
    if out.n_mel <= 0 || out.n_len <= 0 {
        eprintln!(
            "log_mel_spectrogram: invalid mel dimensions n_mel={} n_len={}",
            out.n_mel, out.n_len
        );
        return false;
    }
    let total_size = out.n_mel as usize * out.n_len as usize;
    if total_size > usize::MAX / std::mem::size_of::<f32>() {
        eprintln!(
            "log_mel_spectrogram: size overflow: n_mel={} n_len={}",
            out.n_mel, out.n_len
        );
        return false;
    }
    if n_samples < frame_size {
        eprintln!("log_mel_spectrogram: not enough samples after padding");
        return false;
    }
    out.data.resize(total_size, 0.0);

    log_mel_spectrogram_worker(
        hann, samples, n_samples, frame_size, frame_step, params, cache, out,
    );

    let effective_n_len = n_samples_in / frame_step;
    if params.norm_per_feature {
        assert!(effective_n_len > 1);
        for i in 0..out.n_mel {
            let mut mean = 0.0f64;
            for j in 0..effective_n_len as i64 {
                mean += out.data[i as usize * out.n_len as usize + j as usize] as f64;
            }
            mean /= effective_n_len as f64;

            let mut var = 0.0f64;
            for j in 0..effective_n_len as i64 {
                let value = out.data[i as usize * out.n_len as usize + j as usize] as f64 - mean;
                var += value * value;
            }
            var /= (effective_n_len - 1) as f64; // unbiased
            let mstd = if params.std_eps_after_sqrt {
                var.sqrt() + 1e-5
            } else {
                (var + 1e-5).sqrt()
            };

            for j in 0..effective_n_len as i64 {
                let value = &mut out.data[i as usize * out.n_len as usize + j as usize];
                *value = ((*value as f64 - mean) / mstd) as f32;
            }

            // pad the rest with zeros
            for j in effective_n_len as i64..out.n_len {
                out.data[i as usize * out.n_len as usize + j as usize] = 0.0;
            }
        }
    } else if !params.no_padding {
        // Whisper-style clamping and normalization (NOT used by Gemma4)
        let mut mmax = -1e20f64;
        let mel_size = out.data.len();
        for &v in out.data.iter() {
            if v as f64 > mmax {
                mmax = v as f64;
            }
        }

        mmax -= 8.0;

        for v in out.data.iter_mut() {
            if (*v as f64) < mmax {
                *v = mmax as f32;
            }
            *v = ((*v as f64 + 4.0) / 4.0) as f32;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// audio hparams view — the audio fields of clip_hparams (clip-model.h:121-179)
// ---------------------------------------------------------------------------

/// The audio-related fields of `clip_hparams` the preprocessors read
/// (clip-model.h:121-179). `clip::ClipHparams` carries these; this view keeps
/// mtmd_audio decoupled from the model loader.
#[derive(Clone, Default, Debug)]
pub struct AudioHparams {
    pub n_mel_bins: i32,      // whisper preprocessor
    pub audio_chunk_len: i32, // in seconds
    pub audio_sample_rate: i32,
    pub audio_n_fft: i32,
    pub audio_window_len: i32,
    pub audio_hop_len: i32,
    // parakeet
    pub mel_filters: Vec<f32>,
    pub window: Vec<f32>,
    // pocket-tts
    pub mimi_downsample: i32, // encoder frame rate / model frame rate
}

/// `clip_hparams::pockettts_max_spk_seconds` (clip-model.h:156)
pub const POCKETTTS_MAX_SPK_SECONDS: i32 = 30;

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_whisper (mtmd-audio.cpp:540-620)
// ---------------------------------------------------------------------------

pub struct WhisperPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl WhisperPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        WhisperPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    /// `initialize` (mtmd-audio.cpp:544-548) — NOT thread-safe
    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            true,
            1.0,
            false,
        );
    }

    /// `preprocess` (mtmd-audio.cpp:550-620)
    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let mut n_samples = samples.len();
        if n_samples == 0 {
            // empty audio
            return false;
        }

        let mut smpl: Vec<f32>;
        let samples: &[f32] = {
            // reflection padding needs one sample plus half an FFT window
            let min_samples = self.hparams.audio_n_fft as usize / 2 + 1;
            if n_samples < min_samples {
                smpl = vec![0.0; min_samples];
                smpl[..n_samples].copy_from_slice(samples);
                n_samples = smpl.len();
                smpl.as_slice()
            } else {
                samples
            }
        };

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.center_padding = false;
        params.preemph = 0.0; // disabled
        params.use_natural_log = false;
        params.norm_per_feature = false;

        // make sure the cache is initialized
        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let mut out_full = AudioMel::default();
        let ok = log_mel_spectrogram(
            samples,
            n_samples as i32,
            &params,
            &self.cache,
            &mut out_full,
        );
        if !ok {
            return false;
        }

        // because the cgraph in clip.cpp only accepts 3000 frames each, we need to split the mel
        // we always expect the mel to have 3000 silent frames at the end
        let frames_per_chunk: i64 = 3000;
        assert!(out_full.n_len > frames_per_chunk);
        let mut off: i64 = 0;
        while off < out_full.n_len {
            let n_len = frames_per_chunk.min(out_full.n_len - off);
            if n_len < frames_per_chunk {
                break; // last incomplete chunk will always be a padded chunk, safe to ignore
            }

            let mut out_chunk = AudioMel {
                n_len,
                n_mel: out_full.n_mel,
                n_len_org: out_full.n_mel, // unused
                data: Vec::with_capacity((out_full.n_mel * n_len) as usize),
            };

            for i in 0..out_full.n_mel {
                let src = i as usize * out_full.n_len as usize + off as usize;
                out_chunk
                    .data
                    .extend_from_slice(&out_full.data[src..src + frames_per_chunk as usize]);
            }

            output.push(out_chunk);
            off += frames_per_chunk;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_qwen3a (mtmd-audio.cpp:638-724)
// ---------------------------------------------------------------------------

pub struct Qwen3aPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl Qwen3aPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        Qwen3aPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            true,
            1.0,
            false,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let n_samples = samples.len();
        if n_samples == 0 {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        // Reflection-pad n_fft/2 samples at each end, matching WhisperFeatureExtractor center=True
        let pad = self.hparams.audio_n_fft as usize / 2; // = 200

        let mut padded = vec![0.0f32; n_samples + 2 * pad];
        // Reflect start: padded[0..pad-1] = samples[pad..1] (reversed)
        for (i, item) in padded[..pad].iter_mut().enumerate() {
            let src = pad as i64 - i as i64; // samples[pad], samples[pad-1], ..., samples[1]
            *item = if (src as usize) < n_samples {
                samples[src as usize]
            } else {
                0.0
            };
        }
        padded[pad..pad + n_samples].copy_from_slice(samples);
        // Reflect end: padded[n+pad..n+2*pad-1] = samples[n-2..n-pad-1] (reversed)
        for i in 0..pad {
            let src = n_samples as i64 - 2 - i as i64; // samples[n-2], samples[n-3], ...
            padded[n_samples + pad + i] = if src >= 0 { samples[src as usize] } else { 0.0 };
        }

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.no_padding = true; // reflection padding already applied above
        params.use_natural_log = false; // log10

        let mut mel_full = AudioMel::default();
        let ok = log_mel_spectrogram(
            &padded,
            padded.len() as i32,
            &params,
            &self.cache,
            &mut mel_full,
        );
        if !ok {
            return false;
        }

        // Whisper-style normalization: clamp to (max - 8), scale to [-1, 1]
        {
            let mut mmax = -1e20f64;
            for &v in mel_full.data.iter() {
                if v as f64 > mmax {
                    mmax = v as f64;
                }
            }
            mmax -= 8.0;
            for v in mel_full.data.iter_mut() {
                *v = (((*v as f64).max(mmax) + 4.0) / 4.0) as f32;
            }
        }

        // The effective frame count: center-padded STFT gives ~n_samples/hop_length frames.
        // We take min(mel_full.n_len, n_samples/hop + 1) to avoid including excess frames.
        let n_eff = mel_full
            .n_len
            .min((n_samples / self.hparams.audio_hop_len as usize) as i64 + 1);

        // Split into inference windows matching n_window_infer=800 from model config.
        // Each window is padded to the next multiple of chunk_size for the cgraph.
        // The mtmd caller loops over output entries, so long audio is handled automatically.
        let chunk_size: i64 = 100; // conv sub-chunk size (n_window * 2, n_window=50)
        let window_size: i64 = 800; // mel frames per forward pass (n_window_infer=800)

        let mut off: i64 = 0;
        while off < n_eff {
            let win_eff = window_size.min(n_eff - off);
            let n_chunks = (win_eff + chunk_size - 1) / chunk_size;
            let n_padded = n_chunks * chunk_size;

            let mut out = AudioMel::default();
            out.n_mel = mel_full.n_mel;
            out.n_len = n_padded;
            out.n_len_org = win_eff;
            out.data = vec![0.0; (out.n_mel * out.n_len) as usize];
            for m in 0..out.n_mel {
                let copy_len = win_eff.min(mel_full.n_len - off);
                if copy_len > 0 {
                    let src = m as usize * mel_full.n_len as usize + off as usize;
                    let dst = m as usize * out.n_len as usize;
                    out.data[dst..dst + copy_len as usize]
                        .copy_from_slice(&mel_full.data[src..src + copy_len as usize]);
                }
            }
            output.push(out);
            off += window_size;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_dots3note (mtmd-audio.cpp:740-818)
// ---------------------------------------------------------------------------

pub struct Dots3NotePreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl Dots3NotePreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        Dots3NotePreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            true,
            1.0,
            false,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let n_samples = samples.len();
        if n_samples == 0 {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let pad = self.hparams.audio_n_fft as usize / 2; // center=True padding
        let hop = self.hparams.audio_hop_len as usize;
        let chunk_samples =
            self.hparams.audio_chunk_len as usize * self.hparams.audio_sample_rate as usize;

        let mut start: usize = 0;
        while start < n_samples {
            let n_chunk = chunk_samples.min(n_samples - start);
            let chunk = &samples[start..start + n_chunk];

            let n_valid = (n_chunk / hop) as i64;
            if n_valid == 0 {
                start += chunk_samples;
                continue; // sub-hop tail, contributes no frames
            }

            // reflect-pad the start; the reference zero-pads partial chunks to 60s before the STFT,
            // so a partial chunk sees zeros past its end while a full chunk reflects its own tail
            let mut padded = vec![0.0f32; n_chunk + 2 * pad];
            for (i, item) in padded[..pad].iter_mut().enumerate() {
                let src = pad as i64 - i as i64;
                *item = if (src as usize) < n_chunk {
                    chunk[src as usize]
                } else {
                    0.0
                };
            }
            padded[pad..pad + n_chunk].copy_from_slice(chunk);
            if n_chunk == chunk_samples {
                for i in 0..pad {
                    let src = n_chunk as i64 - 2 - i as i64;
                    padded[n_chunk + pad + i] = if src >= 0 { chunk[src as usize] } else { 0.0 };
                }
            }

            let mut params = FilterParams::default();
            params.n_mel = self.hparams.n_mel_bins as i64;
            params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
            params.hann_window_size = self.hparams.audio_window_len;
            params.hop_length = self.hparams.audio_hop_len;
            params.sample_rate = self.hparams.audio_sample_rate;
            params.no_padding = true; // padding already applied above
            params.use_natural_log = false;

            let mut mel_full = AudioMel::default();
            if !log_mel_spectrogram(
                &padded,
                padded.len() as i32,
                &params,
                &self.cache,
                &mut mel_full,
            ) {
                return false;
            }
            assert!(mel_full.n_len >= n_valid);

            // per-chunk whisper-style normalization, then keep only the valid frames
            let mut out = AudioMel::default();
            out.n_mel = mel_full.n_mel;
            out.n_len = n_valid;
            out.n_len_org = n_valid;
            out.data = vec![0.0; (out.n_mel * out.n_len) as usize];

            let mut mmax = -1e20f64;
            for m in 0..out.n_mel {
                for t in 0..n_valid {
                    let v = mel_full.data[m as usize * mel_full.n_len as usize + t as usize] as f64;
                    if v > mmax {
                        mmax = v;
                    }
                }
            }
            mmax -= 8.0;
            for m in 0..out.n_mel {
                for t in 0..n_valid {
                    let v = (mel_full.data[m as usize * mel_full.n_len as usize + t as usize]
                        as f64)
                        .max(mmax);
                    out.data[m as usize * n_valid as usize + t as usize] = ((v + 4.0) / 4.0) as f32;
                }
            }

            output.push(out);
            start += chunk_samples;
        }
        !output.is_empty()
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_mimo_audio (mtmd-audio.cpp:828-884)
// ---------------------------------------------------------------------------

pub struct MimoAudioPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl MimoAudioPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        MimoAudioPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    /// torchaudio MelSpectrogram(power=1.0, center=True) + log(clip(min=1e-7)):
    /// HTK mel scale, no Slaney area norm, magnitude spectrogram, natural log,
    /// reflect-padded by n_fft/2 on each side.
    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            self.hparams.audio_sample_rate as f32 / 2.0,
            /*slaney_area_norm=*/ false,
            /*scale=*/ 1.0,
            /*use_htk=*/ true,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let n_samples = samples.len();
        if n_samples == 0 {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let pad = self.hparams.audio_n_fft as usize / 2;

        let mut padded = vec![0.0f32; n_samples + 2 * pad];
        for (i, item) in padded[..pad].iter_mut().enumerate() {
            let src = pad as i64 - i as i64;
            *item = if (src as usize) < n_samples {
                samples[src as usize]
            } else {
                0.0
            };
        }
        padded[pad..pad + n_samples].copy_from_slice(samples);
        for i in 0..pad {
            let src = n_samples as i64 - 2 - i as i64;
            padded[n_samples + pad + i] = if src >= 0 { samples[src as usize] } else { 0.0 };
        }

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.no_padding = true; // reflect padding already applied above
        params.use_natural_log = true;
        params.use_magnitude = true;
        params.mel_floor = 1e-7;
        params.norm_per_feature = false;

        let mut out = AudioMel::default();
        let ok = log_mel_spectrogram(&padded, padded.len() as i32, &params, &self.cache, &mut out);
        if !ok {
            return false;
        }

        output.push(out);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_qwen3tts_spk (mtmd-audio.cpp:893-944)
// ---------------------------------------------------------------------------

pub struct Qwen3TtsSpkPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl Qwen3TtsSpkPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        Qwen3TtsSpkPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            true,
            1.0,
            false,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let n_samples = samples.len();
        if n_samples == 0 {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        // reflect pad by (n_fft - hop) / 2 = 384, matching center=False STFT framing
        let pad = (self.hparams.audio_n_fft - self.hparams.audio_hop_len) as usize / 2;
        if n_samples < pad + 1 {
            return false;
        }

        let mut padded = vec![0.0f32; n_samples + 2 * pad];
        for (i, item) in padded[..pad].iter_mut().enumerate() {
            *item = samples[pad - i];
        }
        padded[pad..pad + n_samples].copy_from_slice(samples);
        for i in 0..pad {
            padded[n_samples + pad + i] = samples[n_samples - 2 - i];
        }

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.no_padding = true; // reflect padding already applied above
        params.use_natural_log = true;
        params.use_magnitude = true;
        params.mel_floor = 1e-5;

        let mut out = AudioMel::default();
        let ok = log_mel_spectrogram(&padded, padded.len() as i32, &params, &self.cache, &mut out);
        if !ok {
            return false;
        }

        output.push(out);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_conformer (mtmd-audio.cpp:950-990)
// ---------------------------------------------------------------------------

pub struct ConformerPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl ConformerPreproc {
    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn new(hparams: AudioHparams) -> Self {
        ConformerPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        // NeMo uses a symmetric window: torch.hann_window(periodic=False)
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, false);
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            true,
            1.0,
            false,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        // empty audio
        if samples.is_empty() {
            return false;
        }

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.center_padding = true;
        params.preemph = 0.97;
        params.use_natural_log = true;
        params.norm_per_feature = true;
        params.mel_floor_add = true;
        params.std_eps_after_sqrt = true;

        // make sure the cache is initialized
        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let mut out_full = AudioMel::default();
        let ok = log_mel_spectrogram(
            samples,
            samples.len() as i32,
            &params,
            &self.cache,
            &mut out_full,
        );
        if !ok {
            return false;
        }

        output.push(out_full);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_granite_speech (mtmd-audio.cpp:996-1093)
// ---------------------------------------------------------------------------

pub struct GraniteSpeechPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl GraniteSpeechPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        GraniteSpeechPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);
        self.cache
            .fill_hann_window(self.hparams.audio_window_len as u32, true);
        self.cache.fill_mel_filterbank_matrix(
            (self.hparams.n_mel_bins / 2) as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            -1.0,
            false,
            1.0,
            true,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        if samples.is_empty() {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let n_fft = self.hparams.audio_n_fft;
        let pad = n_fft as usize / 2;
        let n_samples = samples.len();

        // reflect padding
        let n_padded = n_samples + 2 * pad;
        let mut padded = vec![0.0f32; n_padded];
        padded[pad..pad + n_samples].copy_from_slice(samples);
        for i in 0..pad {
            let mut src = i + 1;
            if src >= n_samples {
                src = n_samples - 1;
            }
            padded[pad - 1 - i] = samples[src];
        }
        for i in 0..pad {
            let mut src: i64 = n_samples as i64 - 2 - i as i64;
            if src < 0 {
                src = 0;
            }
            padded[pad + n_samples + i] = samples[src as usize];
        }

        let mut params = FilterParams::default();
        params.n_mel = (self.hparams.n_mel_bins / 2) as i64;
        params.n_fft_bins = 1 + (n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.no_padding = true;
        params.center_padding = false;
        params.preemph = 0.0;
        params.use_natural_log = false;
        params.norm_per_feature = false;
        params.mel_floor = 1e-10;

        let mut mel = AudioMel::default();
        if !log_mel_spectrogram(&padded, n_padded as i32, &params, &self.cache, &mut mel) {
            return false;
        }

        let mut mmax = -1e20f64;
        for &v in mel.data.iter() {
            if v as f64 > mmax {
                mmax = v as f64;
            }
        }
        mmax -= 8.0;

        for v in mel.data.iter_mut() {
            if (*v as f64) < mmax {
                *v = mmax as f32;
            }
            *v = ((*v as f64 + 4.0) / 4.0) as f32;
        }

        let mut n_frames = mel.n_len;
        if n_frames % 2 == 1 {
            n_frames -= 1;
        }
        let n_mel = mel.n_mel;
        let n_stacked = n_frames / 2;

        let mut stacked = AudioMel::default();
        stacked.n_mel = 2 * n_mel;
        stacked.n_len = n_stacked;
        stacked.n_len_org = n_samples as i64;
        stacked.data = vec![0.0; (2 * n_mel * n_stacked) as usize];

        for t in 0..n_stacked {
            for m in 0..n_mel {
                stacked.data[m as usize * n_stacked as usize + t as usize] =
                    mel.data[m as usize * mel.n_len as usize + (2 * t) as usize];
                stacked.data[(m + n_mel) as usize * n_stacked as usize + t as usize] =
                    mel.data[m as usize * mel.n_len as usize + (2 * t + 1) as usize];
            }
        }

        output.push(stacked);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_gemma4a (mtmd-audio.cpp:1099-1174)
// ---------------------------------------------------------------------------

pub struct Gemma4aPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl Gemma4aPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        Gemma4aPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);

        // Standard periodic Hann window, zero-padded to FFT size
        self.cache.hann_window = vec![0.0; self.hparams.audio_n_fft as usize];
        for i in 0..self.hparams.audio_window_len as usize {
            // 0.5f - 0.5f * cosf((2.0f * (float)M_PI * i) / window_len) — f32 math
            let arg =
                (2.0f32 * std::f32::consts::PI * i as f32) / self.hparams.audio_window_len as f32;
            self.cache.hann_window[i] = 0.5f32 - 0.5f32 * arg.cos();
        }

        // HTK mel scale, no Slaney area normalization
        self.cache.fill_mel_filterbank_matrix(
            self.hparams.n_mel_bins as i64,
            self.hparams.audio_n_fft as i64,
            self.hparams.audio_sample_rate,
            0.0,
            self.hparams.audio_sample_rate as f32 / 2.0,
            /*slaney_area_norm=*/ false,
            /*scale=*/ 1.0,
            /*use_htk=*/ true,
        );
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        if samples.is_empty() {
            return false;
        }

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_n_fft; // window is zero-padded to FFT size
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;
        params.no_padding = true;
        params.center_padding = false;
        params.preemph = 0.0;
        params.use_natural_log = true;
        params.use_magnitude = true;
        params.mel_floor = 0.001;
        params.norm_per_feature = false;

        // Split into 30-second chunks (model context limit, ~750 tokens each)
        let chunk_samples = 30 * self.hparams.audio_sample_rate as usize;
        let mut off: usize = 0;
        while off < samples.len() {
            let chunk_len = chunk_samples.min(samples.len() - off);

            // Semicausal left-padding + right-padding to match PyTorch frame count
            let pad_left = (self.hparams.audio_window_len / 2) as usize;
            let fft_size = self.hparams.audio_n_fft as usize;
            let hop = self.hparams.audio_hop_len as usize;
            let n_with_left = chunk_len + pad_left;
            // PyTorch: unfold(size=frame_length+1, step=hop) on semicausal-padded waveform
            let pt_frames = ((n_with_left as i64 - (self.hparams.audio_window_len as i64 + 1))
                / hop as i64
                + 1) as i64;
            let n_padded_needed = (pt_frames - 1) * hop as i64 + fft_size as i64;
            let total_pad = ((n_padded_needed - chunk_len as i64) as usize).max(pad_left);
            let mut padded_samples = vec![0.0f32; total_pad + chunk_len];
            padded_samples[pad_left..pad_left + chunk_len]
                .copy_from_slice(&samples[off..off + chunk_len]);

            let mut out_chunk = AudioMel::default();
            let ok = log_mel_spectrogram(
                &padded_samples,
                padded_samples.len() as i32,
                &params,
                &self.cache,
                &mut out_chunk,
            );
            if !ok {
                return false;
            }

            // Trim to PyTorch frame count
            out_chunk.n_len = out_chunk.n_len.min(pt_frames);

            output.push(out_chunk);
            off += chunk_samples;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_gemma4ua (mtmd-audio.cpp:1383-1415)
// ---------------------------------------------------------------------------

pub struct Gemma4uaPreproc {
    hparams: AudioHparams,
}

impl Gemma4uaPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        Gemma4uaPreproc { hparams }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    /// no-op: no FFT or filterbank needed
    pub fn initialize(&mut self) {}

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        if samples.is_empty() {
            return false;
        }

        let frame_size = self.hparams.n_mel_bins as usize; // 640 samples per token @ 16 kHz = 40 ms
        let n_tokens = (samples.len() + frame_size - 1) / frame_size;

        let mut mel = AudioMel::default();
        mel.n_len = n_tokens as i64;
        mel.n_len_org = n_tokens as i64;
        mel.n_mel = frame_size as i64;
        mel.data = vec![0.0; frame_size * n_tokens];

        // Store mel-major (data[f * n_tokens + t]) so the ggml tensor loads as
        // [n_tokens, frame_size] with ne[0]=n_tokens, ne[1]=frame_size.
        // The graph builder transposes before RMSNorm so normalization is over frame_size.
        for t in 0..n_tokens {
            for f in 0..frame_size {
                let src = t * frame_size + f;
                mel.data[f * n_tokens + t] = if src < samples.len() {
                    samples[src]
                } else {
                    0.0
                };
            }
        }

        output.push(mel);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_parakeet (mtmd-audio.cpp:1180-1377)
// ---------------------------------------------------------------------------

pub struct ParakeetPreproc {
    hparams: AudioHparams,
    cache: AudioCache,
}

impl ParakeetPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        ParakeetPreproc {
            hparams,
            cache: AudioCache::default(),
        }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    /// `worker_thread` (mtmd-audio.cpp:1180-1251) — the port computes the same
    /// frame loop sequentially (see the module doc)
    fn worker_thread(
        &self,
        window_func: &[f32],
        window_size: i32,
        samples: &[f32],
        n_samples: i32,
        frame_size: i32,
        frame_step: i32,
        n_fft_bins: i32,
        mel: &mut AudioMel,
    ) {
        let mut fft_in = vec![0.0f32; frame_size as usize * 2];
        let mut fft_out = vec![0.0f32; frame_size as usize * 2 * 2 * 2];

        let n_fb = n_fft_bins;
        assert!(n_fb == 1 + frame_size / 2);

        let eps = 5.960464477539063e-08f64;

        let mut i: i32 = 0;
        while i < (n_samples / frame_step + 1).min(mel.n_len as i32) {
            let offset = (i * frame_step) as usize;
            let window_pad_left = ((frame_size - window_size) / 2) as usize;

            // Zero-pad left.
            for x in fft_in[..window_pad_left].iter_mut() {
                *x = 0.0;
            }

            // Apply windowed samples in the center.
            let n_to_process = window_size.min(n_samples - offset as i32) as usize;
            for j in 0..n_to_process {
                fft_in[window_pad_left + j] =
                    window_func[j] * samples[offset + window_pad_left + j];
            }

            // Zero-pad right.
            for x in fft_in[window_pad_left + n_to_process..frame_size as usize].iter_mut() {
                *x = 0.0;
            }

            // FFT.
            fft(&self.cache, &mut fft_in, 0, frame_size, &mut fft_out, 0);

            // Calculate modulus^2 of complex numbers.
            for j in 0..n_fb as usize {
                fft_out[j] =
                    fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
            }

            // mel spectrogram.
            for j in 0..mel.n_mel as usize {
                let mut sum = 0.0f64;
                let mut k = 0usize;
                while k + 3 < n_fb as usize {
                    let group = (fft_out[k] * self.cache.filters.data[j * n_fb as usize + k])
                        + (fft_out[k + 1] * self.cache.filters.data[j * n_fb as usize + k + 1])
                        + (fft_out[k + 2] * self.cache.filters.data[j * n_fb as usize + k + 2])
                        + (fft_out[k + 3] * self.cache.filters.data[j * n_fb as usize + k + 3]);
                    sum += group as f64;
                    k += 4;
                }
                while k < n_fb as usize {
                    let product = fft_out[k] * self.cache.filters.data[j * n_fb as usize + k];
                    sum += product as f64;
                    k += 1;
                }
                mel.data[j * mel.n_len as usize + i as usize] = (sum + eps).ln() as f32;
            }
            i += 1;
        }

        // Otherwise fft_out are all zero.
        let empty_sum = eps.ln() as f32;
        while i < mel.n_len as i32 {
            for j in 0..mel.n_mel as usize {
                mel.data[j * mel.n_len as usize + i as usize] = empty_sum;
            }
            i += 1;
        }
    }

    pub fn initialize(&mut self) {
        self.cache
            .fill_sin_cos_table(self.hparams.audio_n_fft as u32);

        let n_fft = (self.hparams.audio_n_fft / 2 + 1) as usize;
        assert!(self.hparams.mel_filters.len() == self.hparams.n_mel_bins as usize * n_fft);
        self.cache.filters.n_mel = self.hparams.n_mel_bins as i64;
        self.cache.filters.n_fft = n_fft as i64;
        self.cache.filters.data = self.hparams.mel_filters.clone();

        assert!(self.hparams.window.len() == self.hparams.audio_window_len as usize);
        assert!(self.hparams.window.len() <= self.hparams.audio_n_fft as usize);
        self.cache.hann_window = self.hparams.window.clone();
    }

    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        let n_samples_in = samples.len();
        if n_samples_in == 0 {
            return false;
        }

        let mut params = FilterParams::default();
        params.n_mel = self.hparams.n_mel_bins as i64;
        params.n_fft_bins = 1 + (self.hparams.audio_n_fft / 2) as i64;
        params.hann_window_size = self.hparams.audio_window_len;
        params.hop_length = self.hparams.audio_hop_len;
        params.sample_rate = self.hparams.audio_sample_rate;

        assert!(!self.cache.sin_vals.is_empty());
        assert!(!self.cache.cos_vals.is_empty());
        assert!(!self.cache.filters.data.is_empty());

        let window_func: &[f32] = &self.cache.hann_window;
        let window_size = params.hann_window_size;
        let frame_size = (params.n_fft_bins - 1) as i32 * 2;
        let frame_step = params.hop_length;

        // Apply preemphasis filter (high-pass): x[i] = x[i] - 0.97 * x[i-1]
        let mut samples_preprocessed = samples.to_vec();
        {
            let preemph = 0.97f32;
            let mut i = n_samples_in as i64 - 1;
            while i > 0 {
                samples_preprocessed[i as usize] = samples_preprocessed[i as usize]
                    - preemph * samples_preprocessed[i as usize - 1];
                i -= 1;
            }
        }

        // Parakeet uses centered constant padding
        let pad = (frame_size / 2) as usize;
        let mut samples_padded = vec![0.0f32; n_samples_in + 2 * pad];
        samples_padded[pad..pad + n_samples_in].copy_from_slice(&samples_preprocessed);

        let mut out_full = AudioMel::default();
        out_full.n_mel = params.n_mel;
        out_full.n_len =
            ((samples_padded.len() as i64 - frame_size as i64) / frame_step as i64 + 1);
        out_full.n_len_org = out_full.n_len;
        out_full.data = vec![0.0; (out_full.n_mel * out_full.n_len) as usize];

        self.worker_thread(
            window_func,
            window_size,
            &samples_padded,
            samples_padded.len() as i32,
            frame_size,
            frame_step,
            params.n_fft_bins as i32,
            &mut out_full,
        );

        // Per-feature normalization (only on valid frames)
        {
            let eps = 1e-5f64;
            let valid_frames = (n_samples_in as i32 / frame_step) as usize;

            for j in 0..out_full.n_mel as usize {
                let mut sum = 0.0f64;
                let mut sq_diff_sum = 0.0f64;

                // Calculate Mean ONLY on valid audio frames
                for i in 0..valid_frames {
                    sum += out_full.data[j * out_full.n_len as usize + i] as f64;
                }
                let mean = sum / valid_frames as f64;

                // Calculate Variance ONLY on valid audio frames
                for i in 0..valid_frames {
                    let diff = out_full.data[j * out_full.n_len as usize + i] as f64 - mean;
                    sq_diff_sum += diff * diff;
                }

                let std_dev = (sq_diff_sum / (valid_frames as f64 - 1.0)).sqrt();
                let denominator = std_dev + eps;

                // Apply to ALL frames (including the padded ones)
                for i in 0..out_full.n_len as usize {
                    out_full.data[j * out_full.n_len as usize + i] =
                        ((out_full.data[j * out_full.n_len as usize + i] as f64 - mean)
                            / denominator) as f32;
                }
            }
        }

        output.push(out_full);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_preprocessor_pockettts (mtmd-audio.cpp:1521-1557)
// ---------------------------------------------------------------------------

pub struct PocketTtsPreproc {
    hparams: AudioHparams,
}

impl PocketTtsPreproc {
    pub fn new(hparams: AudioHparams) -> Self {
        PocketTtsPreproc { hparams }
    }

    /// the hparams this preprocessor was constructed with
    pub fn hparams(&self) -> &AudioHparams {
        &self.hparams
    }

    pub fn initialize(&mut self) {}

    /// mimi takes the raw 24kHz waveform, there is no mel front-end;
    /// the samples are handed over as a single-row "mel", to reuse the normal
    /// chunk path
    pub fn preprocess(&self, samples: &[f32], output: &mut Vec<AudioMel>) -> bool {
        // the encoder needs whole frames, see pad_for_conv1d() in the reference
        let frame_size = self.hparams.mimi_downsample as i64 * 120;
        let mut n_samples = samples.len();
        if n_samples == 0 || frame_size <= 0 {
            return false;
        }

        // the mimi transformer mask is dense, so cost is quadratic in the reference length
        let max_samples = POCKETTTS_MAX_SPK_SECONDS as i64 * self.hparams.audio_sample_rate as i64;
        if n_samples as i64 > max_samples {
            eprintln!(
                "mtmd_audio_preprocessor_pockettts::preprocess: speaker reference is {:.1} s, truncating to the first {} s",
                n_samples as f64 / self.hparams.audio_sample_rate as f64,
                POCKETTTS_MAX_SPK_SECONDS
            );
            n_samples = max_samples as usize;
        }

        let n_frames = (n_samples as i64 + frame_size - 1) / frame_size;
        let n_padded = n_frames * frame_size;

        let mut out = AudioMel::default();
        out.n_mel = 1;
        out.n_len = n_padded;
        out.n_len_org = n_samples as i64;
        out.data = vec![0.0; n_padded as usize];
        out.data[..n_samples].copy_from_slice(&samples[..n_samples]);

        output.push(out);
        true
    }
}

// ---------------------------------------------------------------------------
// mtmd_audio_streaming_istft (mtmd-audio.cpp:1421-1519)
// ---------------------------------------------------------------------------

/// Streaming ISTFT — converts spectrogram frames back to audio one frame at a
/// time (mtmd-audio.h:166-196).
pub struct StreamingIstdft {
    n_fft: i32,
    hop_length: i32,
    n_fft_bins: i32,
    cache: AudioCache,
    overlap_buffer: Vec<f32>,
    window_sum_buffer: Vec<f32>,
    padding_to_remove: i32,
    ifft_in: Vec<f32>,
    ifft_out: Vec<f32>,
}

impl StreamingIstdft {
    /// `mtmd_audio_streaming_istft` ctor (mtmd-audio.cpp:1421-1433)
    pub fn new(n_fft: i32, hop_length: i32) -> Self {
        assert!(n_fft > 0 && hop_length > 0 && hop_length <= n_fft);
        let mut cache = AudioCache::default();
        cache.fill_sin_cos_table(n_fft as u32);
        cache.fill_hann_window(n_fft as u32, true);
        StreamingIstdft {
            n_fft,
            hop_length,
            n_fft_bins: n_fft / 2 + 1,
            cache,
            overlap_buffer: vec![0.0; n_fft as usize],
            window_sum_buffer: vec![0.0; n_fft as usize],
            padding_to_remove: (n_fft - hop_length) / 2,
            ifft_in: vec![0.0; n_fft as usize * 2 * 4], // extra space for recursive IFFT
            ifft_out: vec![0.0; n_fft as usize * 2 * 4],
        }
    }

    /// `reset` (mtmd-audio.cpp:1435-1439)
    pub fn reset(&mut self) {
        self.overlap_buffer.fill(0.0);
        self.window_sum_buffer.fill(0.0);
        self.padding_to_remove = (self.n_fft - self.hop_length) / 2;
    }

    /// `process_frame` (mtmd-audio.cpp:1441-1487) — `frame_spectrum` is
    /// `[n_fft_bins x 2]` interleaved real/imag; returns up to hop_length
    /// samples.
    pub fn process_frame(&mut self, frame_spectrum: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.hop_length as usize];

        // copy frequencies
        for j in 0..self.n_fft_bins as usize {
            self.ifft_in[j * 2] = frame_spectrum[j * 2];
            self.ifft_in[j * 2 + 1] = frame_spectrum[j * 2 + 1];
        }

        // mirror negative frequencies
        for j in 1..(self.n_fft_bins as usize - 1) {
            let mirror_idx = self.n_fft as usize - j;
            self.ifft_in[mirror_idx * 2] = self.ifft_in[j * 2];
            self.ifft_in[mirror_idx * 2 + 1] = -self.ifft_in[j * 2 + 1]; // conjugate
        }

        {
            let cache = &self.cache;
            let n_fft = self.n_fft as usize;
            let mut input = std::mem::take(&mut self.ifft_in);
            let mut output_buf = std::mem::take(&mut self.ifft_out);
            ifft(cache, &mut input, 0, n_fft as i32, &mut output_buf, 0);
            self.ifft_in = input;
            let _ = output_buf;
            self.ifft_out = output_buf;
        }

        // update window sum and overlap buffer
        for j in 0..self.n_fft as usize {
            self.window_sum_buffer[j] += self.cache.hann_window[j] * self.cache.hann_window[j];
            self.overlap_buffer[j] += self.ifft_out[j * 2] * self.cache.hann_window[j];
        }

        // extract hop_length samples with normalization
        for i in 0..self.hop_length as usize {
            if self.window_sum_buffer[i] > 1e-8 {
                output[i] = self.overlap_buffer[i] / self.window_sum_buffer[i];
            } else {
                output[i] = self.overlap_buffer[i];
            }
        }

        // shift buffers left by hop_length
        let hop = self.hop_length as usize;
        let n = self.n_fft as usize;
        self.overlap_buffer.copy_within(hop.., 0);
        for x in self.overlap_buffer[n - hop..].iter_mut() {
            *x = 0.0;
        }
        self.window_sum_buffer.copy_within(hop.., 0);
        for x in self.window_sum_buffer[n - hop..].iter_mut() {
            *x = 0.0;
        }

        // Remove padding if needed
        let to_remove = self.padding_to_remove.min(output.len() as i32);
        self.padding_to_remove -= to_remove;
        output.drain(..to_remove as usize);

        output
    }

    /// `flush` (mtmd-audio.cpp:1489-1519)
    pub fn flush(&mut self) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();

        // Extract remaining samples from overlap buffer
        // Continue until we've extracted all meaningful samples
        let mut remaining = self.n_fft - self.hop_length;
        while remaining > 0 {
            let chunk_size = remaining.min(self.hop_length) as usize;

            for i in 0..chunk_size {
                let sample = if self.window_sum_buffer[i] > 1e-8 {
                    self.overlap_buffer[i] / self.window_sum_buffer[i]
                } else {
                    self.overlap_buffer[i]
                };
                output.push(sample);
            }

            // Shift buffers
            let n = self.n_fft as usize;
            self.overlap_buffer.copy_within(chunk_size.., 0);
            for x in self.overlap_buffer[n - chunk_size..].iter_mut() {
                *x = 0.0;
            }
            self.window_sum_buffer.copy_within(chunk_size.., 0);
            for x in self.window_sum_buffer[n - chunk_size..].iter_mut() {
                *x = 0.0;
            }

            remaining -= chunk_size as i32;
        }

        output
    }
}
