//! llama-mtmd-cli (rust) — port of `tools/mtmd/mtmd-cli.cpp` for the CPU path.
//!
//! Flow (mtmd-cli.cpp:297-420): load the text model, load the mmproj, apply the
//! model's chat template to the user message with the media marker in front of
//! it (:485), tokenize into text/image chunks, encode the image chunks
//! (`mtmd_batch_encode` → `mtmd_encode_chunk`), decode every chunk
//! (`mtmd_helper_eval_chunk_single` / `mtmd_helper_decode_image_chunk`, with the
//! image chunks carrying their embeddings) and generate greedily.
//!
//! Flags mirror the reference: `-m`, `--mmproj`, `--image`, `-p`, `-n`, `-t`,
//! `-fa`, `--temp`, `-c`, plus `--dump-embd` (this port's parity hook, the
//! reference uses the `MTMD_DEBUG_EMBEDDINGS` env var) and `--prompt-tokens`
//! (print the tokenized chunks).
//!
//! NOT ported (see the module report): interactive chat mode, audio, video,
//! `--image-min/max-tokens` (they exist as flags and reach the clip hparams, but
//! only through what `mtmd_context_params` carries), Jinja `--jinja` mode
//! (`llama::chat::apply_str` exists but the CLI uses the builtin path) and the
//! `--grammar` family.

use std::sync::Arc;

use llama::chat::{self, ChatMessage, Role};
use llama::clip::{self, ClipFlashAttn};
use llama::context::{DecodeContext, ForwardWeights};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::hparams::{LlamaHparams, LlamaRopeType};
use llama::model::{load_model, LlamaModel};
use llama::mtmd::{self, MtmdChunk, MtmdContext, MtmdContextParams, MtmdEval, MtmdPosType};
use llama::vocab::Vocab;
use memmap2::Mmap;

const USAGE: &str = "\
usage: llama-mtmd-cli -m <text model> --mmproj <mmproj> --image <image> -p <prompt> [options]

  -m,    --model FNAME     text model (GGUF)
  --mmproj FNAME           multimodal projector (GGUF)
  --image FNAME            image to describe (PNG / BMP / PPM)
  --audio FNAME            audio to describe (WAV)
  -p,    --prompt PROMPT   user prompt; <__media__> marks where the media goes
  -n,    --predict N       number of tokens to generate (default 32)
  -t,    --threads N       CPU threads (default 8)
  -fa,   --flash-attn      on|off|auto (default auto)
  --temp T                 sampling temperature; only 0 (greedy) is implemented
  -c,    --ctx-size N      context size (default 4096)
  --dump-embd FNAME        write the vision embeddings (reference
                           MTMD_DEBUG_EMBEDDINGS format)
  --prompt-tokens          print the tokenized chunk layout
  --logits-top K           print the top-k logits of every generation step
  --embd-file FNAME        use the vision embeddings from a dump file
  --image-min-tokens N     minimum image tokens per image (clip.vision.image_min_pixels)
  --image-max-tokens N     maximum image tokens per image (clip.vision.image_max_pixels)
  -h,    --help";

#[derive(Clone, Copy, PartialEq, Eq)]
enum FlashAttn {
    Off,
    On,
    Auto,
}

impl FlashAttn {
    fn on(self) -> bool {
        matches!(self, FlashAttn::On)
    }
    /// `mtmd_get_clip_flash_attn_type` (mtmd.cpp:449)
    fn clip(self) -> ClipFlashAttn {
        match self {
            FlashAttn::Auto => ClipFlashAttn::Auto,
            FlashAttn::On => ClipFlashAttn::Enabled,
            FlashAttn::Off => ClipFlashAttn::Disabled,
        }
    }
}

struct Args {
    model: String,
    mmproj: String,
    images: Vec<String>,
    /// `--audio` (mtmd-cli.cpp:46 --audio <path>): WAV files to describe
    audios: Vec<String>,
    prompt: String,
    n_predict: usize,
    n_threads: usize,
    n_ctx: u32,
    flash_attn: FlashAttn,
    temp: f32,
    dump_embd: Option<String>,
    prompt_tokens: bool,
    /// print the top-k logits of every generation step (parity diagnostics)
    logits_top: usize,
    /// load the vision embeddings from a dump instead of encoding the image
    /// (reference `MTMD_DEBUG_EMBEDDINGS` format) — parity diagnostics
    embd_file: Option<String>,
    /// `--image-min/max-tokens` (mtmd.h:97 → clip_hparams.custom_image_*)
    image_min_tokens: i32,
    image_max_tokens: i32,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: String::new(),
            mmproj: String::new(),
            images: Vec::new(),
            audios: Vec::new(),
            prompt: String::new(),
            n_predict: 32,
            n_threads: 8,
            n_ctx: 4096,
            flash_attn: FlashAttn::Auto,
            temp: 0.0,
            dump_embd: None,
            prompt_tokens: false,
            logits_top: 0,
            embd_file: None,
            image_min_tokens: -1,
            image_max_tokens: -1,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    let need = |it: &mut dyn Iterator<Item = String>, flag: &str| -> Result<String, String> {
        it.next().ok_or_else(|| format!("{flag} needs a value"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--model" => a.model = need(&mut it, &arg)?,
            "--mmproj" => a.mmproj = need(&mut it, &arg)?,
            "--image" => a.images.push(need(&mut it, &arg)?),
            "--audio" => a.audios.push(need(&mut it, &arg)?),
            "-p" | "--prompt" => a.prompt = need(&mut it, &arg)?,
            "-n" | "--predict" | "--n-predict" => {
                a.n_predict = need(&mut it, &arg)?.parse().map_err(|e| format!("-n: {e}"))?
            }
            "-t" | "--threads" => {
                a.n_threads = need(&mut it, &arg)?.parse().map_err(|e| format!("-t: {e}"))?
            }
            "-c" | "--ctx-size" => {
                a.n_ctx = need(&mut it, &arg)?.parse().map_err(|e| format!("-c: {e}"))?
            }
            "-fa" | "--flash-attn" => {
                a.flash_attn = match need(&mut it, &arg)?.as_str() {
                    "on" => FlashAttn::On,
                    "off" => FlashAttn::Off,
                    "auto" => FlashAttn::Auto,
                    other => return Err(format!("-fa: unknown value '{other}'")),
                }
            }
            "--temp" => {
                a.temp = need(&mut it, &arg)?.parse().map_err(|e| format!("--temp: {e}"))?
            }
            "--dump-embd" => a.dump_embd = Some(need(&mut it, &arg)?),
            "--prompt-tokens" => a.prompt_tokens = true,
            "--embd-file" => a.embd_file = Some(need(&mut it, &arg)?),
            "--image-min-tokens" => {
                a.image_min_tokens =
                    need(&mut it, &arg)?.parse().map_err(|e| format!("--image-min-tokens: {e}"))?
            }
            "--image-max-tokens" => {
                a.image_max_tokens =
                    need(&mut it, &arg)?.parse().map_err(|e| format!("--image-max-tokens: {e}"))?
            }
            "--logits-top" => {
                a.logits_top = need(&mut it, &arg)?.parse().map_err(|e| format!("--logits-top: {e}"))?
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}' (see --help)")),
        }
    }
    if a.model.is_empty() || a.mmproj.is_empty() || a.prompt.is_empty() {
        return Err(format!("-m, --mmproj and -p are required\n\n{USAGE}"));
    }
    Ok(a)
}

// ---------------------------------------------------------------------------
// text-model wiring — the same helpers llama-cli uses (private there, so they
// live here): per-layer weights for the arch of the text model.
// ---------------------------------------------------------------------------

/// `llama-cli/src/main.rs:967` `attn_params` (llama-context.cpp:106-215 rope
/// runtime values via `hparams::rope_runtime`)
fn attn_params(hp: &LlamaHparams, il: usize, use_flash_attn: bool) -> AttnParams {
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(il) as i64,
        n_head_kv: hp.n_head_kv(il) as i64,
        n_embd_head_k: hp.n_embd_head_k(il) as i64,
        n_embd_head_v: hp.n_embd_head_v(il) as i64,
        n_rot: hp.n_rot(il) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn,
    }
}

/// qwen35.cpp:196-198 — the trunk layers; the MTP/nextn blocks are extra dense
/// blocks the main pass never executes.
fn qwen35_weights(m: &LlamaModel, n_trunk: usize) -> graph_arch::Qwen35ModelWeights {
    graph_arch::Qwen35ModelWeights {
        tok_embd: m.tok_embd,
        output_norm: m.output_norm,
        output: m.output,
        cls_out: m.cls_out,
        cls_out_b: m.cls_out_b,
        layers: m.layers[..n_trunk]
            .iter()
            .enumerate()
            .map(|(il, l)| graph_arch::Qwen35LayerWeights {
                attn_norm: l.attn_norm.unwrap_or_else(|| panic!("layer {il}: attn_norm")),
                attn_post_norm: l
                    .attn_post_norm
                    .unwrap_or_else(|| panic!("layer {il}: attn_post_norm")),
                wq: l.wq,
                wk: l.wk,
                wv: l.wv,
                wo: l.wo,
                attn_q_norm: l.attn_q_norm,
                attn_k_norm: l.attn_k_norm,
                wqkv: l.wqkv,
                wqkv_gate: l.wqkv_gate,
                ssm_conv1d: l.ssm_conv1d,
                ssm_dt_b: l.ssm_dt_b,
                ssm_a: l.ssm_a,
                ssm_beta: l.ssm_beta,
                ssm_alpha: l.ssm_alpha,
                ssm_norm: l.ssm_norm,
                ssm_out: l.ssm_out,
                ffn_gate: l.ffn_gate.unwrap_or_else(|| panic!("layer {il}: ffn_gate")),
                ffn_up: l.ffn_up.unwrap_or_else(|| panic!("layer {il}: ffn_up")),
                ffn_down: l.ffn_down.unwrap_or_else(|| panic!("layer {il}: ffn_down")),
            })
            .collect(),
    }
}

/// llama-cli/src/main.rs:1350 `qwen35_params`
fn qwen35_params(hp: &LlamaHparams, n_layer: usize, attn: AttnParams) -> graph_arch::Qwen35Params {
    graph_arch::Qwen35Params {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_layer).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_layer).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_layer).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_layer).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_layer).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_layer).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
    }
}

fn first_attn_layer(hp: &LlamaHparams, n_layer: usize) -> usize {
    (0..n_layer).find(|&il| !hp.is_recr(il)).unwrap_or(0)
}

/// The decoder's position layout (mtmd.cpp:567 `llama_model_rope_type`).
fn pos_type_of(hp: &LlamaHparams) -> MtmdPosType {
    match hp.rope_type {
        LlamaRopeType::MROPE | LlamaRopeType::IMROPE => MtmdPosType::Mrope,
        _ => MtmdPosType::Normal,
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("llama-mtmd-cli: {e}");
            std::process::exit(1);
        }
    };
    if args.temp != 0.0 {
        eprintln!("llama-mtmd-cli: only --temp 0 (greedy) is implemented");
        std::process::exit(1);
    }

    // ---- text model -------------------------------------------------------
    let file = std::fs::File::open(&args.model).expect("open text model");
    let mmap = Arc::new(unsafe { Mmap::map(&file) }.expect("mmap text model"));
    let gguf = ggml::Gguf::open(&args.model).expect("parse text model");
    let model = load_model(&gguf, mmap).expect("load text model");
    let vocab = Vocab::load(&gguf).expect("vocab");
    let hp = model.hparams.clone();
    let n_layer = hp.n_layer() as usize;
    let n_embd_text = hp.n_embd as i32;
    eprintln!(
        "llama-mtmd-cli: text model: arch {:?}, n_layer {n_layer}, n_embd {n_embd_text}",
        model.arch
    );

    // ---- mmproj -----------------------------------------------------------
    let mtmd_params = MtmdContextParams {
        n_threads: args.n_threads,
        flash_attn_type: args.flash_attn.clip(),
        image_min_tokens: args.image_min_tokens,
        image_max_tokens: args.image_max_tokens,
        ..Default::default()
    };
    let mut mctx = MtmdContext::init_from_file(
        &args.mmproj,
        Some((&vocab, n_embd_text, pos_type_of(&hp))),
        &mtmd_params,
    )
    .expect("load mmproj");

    // ---- images -----------------------------------------------------------
    let mut bitmaps = Vec::new();
    for path in &args.images {
        match mtmd::bitmap_from_file(path) {
            Ok(b) => {
                eprintln!("llama-mtmd-cli: loaded {path} ({}x{})", b.nx, b.ny);
                bitmaps.push(b);
            }
            Err(e) => {
                eprintln!("llama-mtmd-cli: {e}");
                std::process::exit(1);
            }
        }
    }
    // ---- audio (mtmd-cli.cpp:496 load_media on an audio file) --------------
    let mut audio_buffers: Vec<Vec<u8>> = Vec::new();
    for path in &args.audios {
        match std::fs::read(path) {
            Ok(bytes) => {
                eprintln!("llama-mtmd-cli: loaded {path} ({} bytes)", bytes.len());
                audio_buffers.push(bytes);
            }
            Err(e) => {
                eprintln!("llama-mtmd-cli: failed to read {path}: {e}");
                std::process::exit(1);
            }
        }
    }
    if bitmaps.is_empty() && audio_buffers.is_empty() {
        eprintln!("llama-mtmd-cli: no --image or --audio given");
        std::process::exit(1);
    }

    // mtmd-cli.cpp:485 — the marker goes in front of the prompt when it is
    // missing ("most models require the marker before each image")
    let marker = mtmd::mtmd_default_marker();
    let mut prompt = args.prompt.clone();
    if !prompt.contains(&marker) {
        for _ in 0..(bitmaps.len() + audio_buffers.len()) {
            prompt = format!("{marker}{prompt}");
        }
    }

    // chat template (mtmd-cli.cpp:242 chat_add_and_format → common_chat_format_single)
    let formatted = match gguf.get_str("tokenizer.chat_template") {
        Some(src) => {
            let tmpl = chat::detect(src);
            let msgs = [ChatMessage::new(Role::User, prompt.clone())];
            match chat::apply(tmpl, &msgs, true) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("llama-mtmd-cli: chat template failed ({e}), using the raw prompt");
                    prompt.clone()
                }
            }
        }
        None => prompt.clone(),
    };
    eprintln!("llama-mtmd-cli: prompt: {}", formatted.replace('\n', "\\n"));

    // ---- tokenize ---------------------------------------------------------
    // mtmd-cli.cpp:277-306 — split the formatted chat on the media marker and
    // interleave text and media parts (mtmd_tokenize_from_parts)
    let mut segments: Vec<&str> = Vec::new();
    {
        let mut rest = formatted.as_str();
        loop {
            let Some(pos) = rest.find(&marker) else {
                segments.push(rest);
                break;
            };
            let (head, tail) = rest.split_at(pos);
            segments.push(head);
            rest = &tail[marker.len()..];
        }
    }
    let media: Vec<mtmd::MtmdMedia> = bitmaps
        .iter()
        .map(mtmd::MtmdMedia::Image)
        .chain(audio_buffers.iter().map(|b| mtmd::MtmdMedia::Audio(b.as_slice())))
        .collect();
    let chunks = mctx
        .tokenize_parts(&vocab, &segments, &media, true)
        .expect("tokenize");
    if args.prompt_tokens {
        for (i, c) in chunks.iter().enumerate() {
            match c {
                MtmdChunk::Text(t) => eprintln!("  chunk {i}: text, {} tokens", t.len()),
                MtmdChunk::Image(img) => eprintln!(
                    "  chunk {i}: image, {}x{} tokens ({} total)",
                    img.nx,
                    img.ny,
                    img.n_tokens()
                ),
                MtmdChunk::Audio(a) => {
                    eprintln!("  chunk {i}: audio, {} tokens", a.n_tokens)
                }
            }
        }
    }

    // ---- encode the image chunks -----------------------------------------
    let mut images: Vec<Vec<f32>> = Vec::new();
    match &args.embd_file {
        Some(path) => {
            let bytes = std::fs::read(path).expect("read --embd-file");
            let n_tokens = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
            let n_embd = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
            let data: Vec<f32> = bytes[8..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            eprintln!("llama-mtmd-cli: using {path} ({n_tokens} tokens x {n_embd} embd)");
            assert_eq!(data.len(), n_tokens as usize * n_embd as usize);
            for chunk in &chunks {
                let chunk_tokens = match chunk {
                    MtmdChunk::Image(img) => Some(img.n_tokens() as i32),
                    MtmdChunk::Audio(a) => Some(a.n_tokens as i32),
                    MtmdChunk::Text(_) => None,
                };
                if let Some(ct) = chunk_tokens {
                    assert_eq!(ct, n_tokens, "embedding dump token count");
                    images.push(data.clone());
                }
            }
        }
        None => {
            for chunk in &chunks {
                // mtmd-cli.cpp: image and audio chunks alike go through
                // mtmd_encode_chunk (the audio encoder runs the whisper-enc
                // graph); text chunks have no embeddings
                if matches!(chunk, MtmdChunk::Image(_) | MtmdChunk::Audio(_)) {
                    let embd = mctx.encode_chunk(chunk).expect("encode chunk").to_vec();
                    images.push(embd);
                }
            }
        }
    }
    if let Some(path) = &args.dump_embd {
        let media_tokens = |c: &MtmdChunk| -> Option<i32> {
            match c {
                MtmdChunk::Image(img) => Some(img.n_tokens() as i32),
                MtmdChunk::Audio(a) => Some(a.n_tokens as i32),
                MtmdChunk::Text(_) => None,
            }
        };
        if let Some(mt) = chunks.iter().find_map(media_tokens) {
            let embd = &images[0];
            clip::write_embedding_dump(path, embd, mt, mctx.n_mmproj_embd())
                .expect("write embedding dump");
            eprintln!(
                "llama-mtmd-cli: wrote {path} ({mt} tokens x {} embd)",
                mctx.n_mmproj_embd()
            );
        }
    }

    // ---- text decoder -----------------------------------------------------
    let n_trunk = n_layer;
    let attn = attn_params(&hp, first_attn_layer(&hp, n_trunk), args.flash_attn.on());
    let weights = match model.arch {
        llama::arch::LlmArch::QWEN35 => {
            let p = qwen35_params(&hp, n_trunk, attn);
            ForwardWeights::Qwen35(qwen35_weights(&model, n_trunk), p)
        }
        other => {
            eprintln!(
                "llama-mtmd-cli: arch '{}' ({other:?}) has no forward builder wired in this port",
                other.name()
            );
            std::process::exit(1);
        }
    };

    let n_batch = 512.min(args.n_ctx as usize);
    let mut eval = MtmdEval::new(
        model.ctx,
        weights,
        attn,
        args.n_ctx,
        args.n_threads,
        n_batch,
        &mctx,
        &chunks,
        &images,
        args.n_predict,
    )
    .expect("build the multimodal decoder");

    // ---- decode the prompt ------------------------------------------------
    let mut logits: Option<Vec<f32>> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        let l = eval.eval_chunk(chunk, last).expect("decode chunk");
        if l.is_some() {
            logits = l;
        }
    }

    // ---- generate ---------------------------------------------------------
    let mut out_tokens: Vec<i32> = Vec::new();
    let mut text = String::new();
    for step in 0..args.n_predict {
        let Some(lg) = logits.clone() else { break };
        if args.logits_top > 0 {
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_unstable_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
            let top: Vec<String> = idx[..args.logits_top.min(lg.len())]
                .iter()
                .map(|&i| format!("{}={:.6}", i, lg[i]))
                .collect();
            eprintln!("step {step}: top: {}", top.join(" "));
        }
        let mut best = 0usize;
        for (i, v) in lg.iter().enumerate() {
            if *v > lg[best] {
                best = i;
            }
        }
        let tok = best as i32;
        if vocab.is_eog(tok) {
            break;
        }
        out_tokens.push(tok);
        let piece = vocab.token_to_piece(tok);
        print!("{piece}");
        use std::io::Write as _;
        std::io::stdout().flush().ok();
        text.push_str(piece);
        logits = eval.decode_token(tok, true).expect("decode generated token");
    }
    println!();
    eprintln!(
        "llama-mtmd-cli: generated {} tokens, n_past = {}",
        out_tokens.len(),
        eval.n_past
    );
    eprintln!("llama-mtmd-cli: text = {text:?}");
    let _ = DecodeContext::n_vocab; // (docs) DecodeContext is used through MtmdEval
}