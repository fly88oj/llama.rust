//! tts_gen_e2e.rs — audio round 5: the GEN side of the TTS mmprojs
//! (tools/mtmd/models/{qwen3tts-gen,pockettts-gen,pockettts-seanet}.cpp).
//!
//! Synthetic protocol: the PORT's GGUF writer builds one gen mmproj per
//! pipeline (clip.has_gen_audio_encoder + the a.gen.* tensor set the
//! reference's clip.cpp:3039-3134 / :2970-3037 loaders read), the port
//! loads it through `clip_init_from_file` (the GenAudio modality) and runs
//! the GEN_CODE step (fixed hidden state + fixed seed → the mt19937 draws
//! the reference would make) and the GEN_WAV step (a codes/feats window,
//! with the streaming state threaded across two calls), then
//! parity/tts_parity.sh hands the same files + inputs to the pinned
//! reference through parity/ref_gen_audio_dump.cpp (clip_init +
//! clip_encode over libmtmd, the exact call mtmd_gen_audio_process makes)
//! and compares codes / feats / embd / eos / PCM / state.

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::Value;
use llama::clip::{
    clip_init_from_file, ClipContextParams, ClipFlashAttn, ClipGenProcess, ClipModality,
    GenInp, ProjectorType,
};

const OUT_DIR: &str = "/tmp/tts-gen-synth";
const SEED: u32 = 42;

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 8) as f32 / 16_777_216.0 - 0.5
}

fn tensor_bytes(n: usize, state: &mut u32) -> Vec<u8> {
    (0..n)
        .map(|_| lcg(state).to_le_bytes())
        .flat_map(|b| b.to_vec())
        .collect()
}

struct W {
    w: GgufWriter,
    datas: Vec<Vec<u8>>,
    st: u32,
}

impl W {
    fn new(proj: &str) -> Self {
        let mut w = GgufWriter::new(32);
        w.set_kv("general.architecture", Value::String("clip".to_string()));
        w.set_kv(
            "general.name",
            Value::String(format!("llama-rust-synth-{proj}")),
        );
        w.set_kv("general.file_type", Value::U32(0)); // F32
        w.set_kv("clip.has_vision_encoder", Value::Bool(false));
        w.set_kv("clip.has_audio_encoder", Value::Bool(false));
        w.set_kv("clip.has_gen_audio_encoder", Value::Bool(true));
        w.set_kv("clip.projector_type", Value::String(proj.to_string()));
        Self {
            w,
            datas: Vec::new(),
            st: 0x5eed_0001,
        }
    }
    fn kv(&mut self, k: &str, v: Value) {
        self.w.set_kv(k, v);
    }
    fn add(&mut self, name: &str, ne: [i64; 4]) {
        self.w.add_tensor(name, GgmlType::F32, ne);
        let n = (ne[0] * ne[1] * ne[2] * ne[3]).max(0) as usize;
        self.datas.push(tensor_bytes(n, &mut self.st));
    }
    /// like `add`, but the values are scaled down — the code2wav decoder's
    /// random weights grow the DAC activations past the F32->F16 im2col
    /// conversion range (ggml.c:4619 converts patches to F16 even for F32
    /// kernels), and both the port AND the pinned reference then emit inf/
    /// NaN in the streamed state (bit-identically — the overflow is in the
    /// shared conv path). A real trained DAC keeps activations bounded; the
    /// scaled fixture reproduces that regime so the state threading is
    /// compared on finite values.
    fn add_scaled(&mut self, name: &str, ne: [i64; 4], scale: f32) {
        self.w.add_tensor(name, GgmlType::F32, ne);
        let n = (ne[0] * ne[1] * ne[2] * ne[3]).max(0) as usize;
        let mut st = self.st;
        let bytes = (0..n)
            .map(|_| (lcg(&mut st) * scale).to_le_bytes())
            .flat_map(|b| b.to_vec())
            .collect();
        self.st = st;
        self.datas.push(bytes);
    }
    /// the code2wav tensor scale — see add_scaled's doc. 0.1 keeps every
    /// DAC activation comfortably inside F16 after the ~15 stacked convs of
    /// the random-weight decoder (empirically verified: no inf/NaN state
    /// slots on either side).
    fn add_scaled_c2w(&mut self, name: &str, ne: [i64; 4]) {
        self.add_scaled(name, ne, 0.1);
    }
    /// the gen.audio hparams block (clip.cpp:1297-1309 + the per-arch arms)
    fn gen_hp(&mut self, n_embd: i64, n_head: i64, n_ff: i64, n_layer: i64, eps: f32) {
        self.kv("clip.gen.audio.embedding_length", Value::U32(n_embd as u32));
        self.kv("clip.gen.audio.attention.head_count", Value::U32(n_head as u32));
        self.kv("clip.gen.audio.attention.head_count_kv", Value::U32(n_head as u32));
        self.kv("clip.gen.audio.feed_forward_length", Value::U32(n_ff as u32));
        self.kv("clip.gen.audio.block_count", Value::U32(n_layer as u32));
        self.kv("clip.gen.audio.projection_dim", Value::U32(n_embd as u32));
        self.kv("clip.gen.audio.attention.layer_norm_epsilon", Value::F32(eps));
    }
    fn finish(self, path: &str) {
        let Self { w, datas, .. } = self;
        let (w, datas) = (w, datas);
        let tmp = format!("{path}.tmp{}", std::process::id());
        let f = std::fs::File::create(&tmp).expect("create synth gguf");
        let mut bw = std::io::BufWriter::new(f);
        let refs: Vec<&[u8]> = datas.iter().map(|d| d.as_slice()).collect();
        w.write(&mut bw, &refs).expect("write synth gguf");
        use std::io::Write as _;
        bw.flush().unwrap();
        std::fs::rename(&tmp, path).expect("publish synth gguf");
    }
}

// ===========================================================================
// qwen3tts_gen — the c2w hparams are hardcoded (clip.cpp:1844-1862):
// wav_tfm_n_layer 8 / n_embd 512 / n_ff 1024 / n_head 16(kv 16) / eps 1e-5 /
// rope 1e4 / swa 72 / upsample blocks 2 / dac blocks 4 / res 3.
// ===========================================================================

const Q3_E: i64 = 32; // talker (backbone) embedding width = n_mmproj_embd
const Q3_PRED: i64 = 64; // code_predictor hidden
const Q3_VOCAB0: i64 = 16; // codebook-0 vocab
const Q3_VOCAB: i64 = 24; // acoustic codebook vocab
const Q3_NACO: i64 = 3; // acoustic codebooks (head ne[2])
const Q3_CB: i64 = 256; // RVQ codebook width
const Q3_HID: i64 = 512; // wav transformer width (hardcoded)
const Q3_TFMFF: i64 = 1024;
const Q3_UPFF: i64 = 256; // ConvNeXt block FF

fn write_qwen3tts_gen() -> String {
    let path = format!("{OUT_DIR}/mmproj-qwen3tts-gen.gguf");
    let mut w = W::new("qwen3tts_gen");
    // the code_predictor's transformer = the generic gen.audio hparams
    w.gen_hp(Q3_PRED, 4, 128, 2, 1e-5);

    // ---- code_predictor (clip.cpp:3041-3047) ---------------------------
    w.add("a.gen.code.proj_in.weight", [Q3_E, Q3_PRED, 1, 1]);
    w.add("a.gen.code.proj_in.bias", [Q3_PRED, 1, 1, 1]);
    w.add("a.gen.code.embd.weight", [Q3_E, Q3_VOCAB, Q3_NACO, 1]);
    w.add("a.gen.code.head.weight", [Q3_PRED, Q3_VOCAB, Q3_NACO, 1]);
    w.add("a.gen.code.out_embd.weight", [Q3_E, Q3_VOCAB0, 1, 1]);
    w.add("a.gen.code.output_norm.weight", [Q3_PRED, 1, 1, 1]);
    // the generic layer loop (prefix a.gen.code, clip.cpp:2256-2299)
    for il in 0..2i64 {
        w.add(&format!("a.gen.code.blk.{il}.attn_q.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_k.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_v.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_out.weight"), [Q3_PRED, Q3_PRED, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_q_norm.weight"), [16, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.attn_k_norm.weight"), [16, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ln1.weight"), [Q3_PRED, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ln2.weight"), [Q3_PRED, 1, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_gate.weight"), [Q3_PRED, 128, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_up.weight"), [Q3_PRED, 128, 1, 1]);
        w.add(&format!("a.gen.code.blk.{il}.ffn_down.weight"), [128, Q3_PRED, 1, 1]);
    }

    // ---- code2wav: RVQ decode (clip.cpp:3053-3058) ----------------------
    w.add_scaled_c2w("a.gen.wav.quant.first.in_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.first.out_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.first.codebook.weight", [Q3_CB, Q3_VOCAB0, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.in_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.out_proj.weight", [Q3_CB, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.quant.rest.codebook.weight", [Q3_CB, Q3_VOCAB, Q3_NACO, 1]);

    w.add_scaled_c2w("a.gen.wav.pre_conv.weight", [3, Q3_HID, 1024, 1]);
    w.add_scaled_c2w("a.gen.wav.pre_conv.bias", [1024, 1, 1, 1]);

    // ---- pre_transformer (clip.cpp:3063-3085) ---------------------------
    w.add_scaled_c2w("a.gen.wav.tfm.in_proj.weight", [1024, Q3_HID, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.in_proj.bias", [Q3_HID, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.out_proj.weight", [Q3_HID, 1024, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.out_proj.bias", [1024, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.tfm.output_norm.weight", [Q3_HID, 1, 1, 1]);
    for il in 0..8i64 {
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"), [Q3_HID, Q3_HID, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"), [Q3_HID, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_gate.weight"), [Q3_HID, Q3_TFMFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"), [Q3_HID, Q3_TFMFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"), [Q3_TFMFF, Q3_HID, 1, 1]);
    }

    // ---- upsample: 2 blocks (clip.cpp:3088-3102) -------------------------
    for il in 0..2i64 {
        // kernel == stride (2): no overlap tail
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.conv.weight"), [2, 1024, 1024, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.conv.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.dwconv.weight"), [7, 1, 1024, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.dwconv.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.norm.weight"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.norm.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw1.weight"), [1024, Q3_UPFF, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw1.bias"), [Q3_UPFF, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw2.weight"), [Q3_UPFF, 1024, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.pw2.bias"), [1024, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.up.blk.{il}.gamma"), [1024, 1, 1, 1]);
    }

    // ---- DAC decoder: entry + 4 blocks × 3 res + post (clip.cpp:3105-3133)
    w.add_scaled_c2w("a.gen.wav.dac.entry.weight", [3, 1024, 512, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.entry.bias", [512, 1, 1, 1]);
    // block 0 upsamples (K = 2*stride); blocks 1-3 keep the rate
    let dac_ocs = [512i64, 512, 512, 512];
    for (il, &oc) in dac_ocs.iter().enumerate() {
        let ic = if il == 0 { 512 } else { dac_ocs[il - 1] };
        let il = il as i64;
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.snake.alpha"), [oc, 1, 1, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.snake.beta"), [oc, 1, 1, 1]);
        let k = if il == 0 { 4 } else { 2 };
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.conv.weight"), [k, oc, ic, 1]);
        w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.conv.bias"), [oc, 1, 1, 1]);
        for (ir, &dil) in [1i64, 3, 9].iter().enumerate() {
            let ir = ir as i64;
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.alpha"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act1.beta"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.weight"), [3, oc, oc, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv1.bias"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.alpha"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.act2.beta"), [oc, 1, 1, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.weight"), [1, oc, oc, 1]);
            w.add_scaled_c2w(&format!("a.gen.wav.dac.blk.{il}.res.{ir}.conv2.bias"), [oc, 1, 1, 1]);
            let _ = dil;
        }
    }
    w.add_scaled_c2w("a.gen.wav.dac.post_snake.alpha", [1, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_snake.beta", [1, 1, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_conv.weight", [3, 512, 1, 1]);
    w.add_scaled_c2w("a.gen.wav.dac.post_conv.bias", [1, 1, 1, 1]);

    w.finish(&path);
    path
}

// ===========================================================================
// pockettts_gen — the mimi/flow hparams hardcoded at clip.cpp:1863-1878:
// seanet ratios [4,5,6], n_stage 3, downsample 16, tfm context 250,
// rope 1e4, flow_n_step 1, eos threshold -4.
// ===========================================================================

const PT_E: i64 = 64; // backbone width (n_mmproj_embd)
const PT_LAT: i64 = 32; // latent width
const PT_C: i64 = 64; // mimi transformer / seanet channel width

fn write_pockettts_gen() -> String {
    let path = format!("{OUT_DIR}/mmproj-pockettts-gen.gguf");
    let mut w = W::new("pockettts_gen");
    // the mimi decoder transformer = the generic gen.audio hparams
    w.gen_hp(PT_C, 4, 128, 2, 1e-5);
    w.kv("clip.gen.audio.model_variant", Value::String("english".to_string()));

    // ---- flow net (clip.cpp:2972-3009) -----------------------------------
    w.add("a.gen.flow.input_proj.weight", [PT_LAT, PT_LAT, 1, 1]);
    w.add("a.gen.flow.input_proj.bias", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.cond_embd.weight", [PT_E, PT_LAT, 1, 1]);
    w.add("a.gen.flow.cond_embd.bias", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.final.ada.weight", [PT_LAT, 2 * PT_LAT, 1, 1]);
    w.add("a.gen.flow.final.ada.bias", [2 * PT_LAT, 1, 1, 1]);
    w.add("a.gen.flow.final.proj.weight", [PT_LAT, PT_LAT, 1, 1]);
    w.add("a.gen.flow.final.proj.bias", [PT_LAT, 1, 1, 1]);
    for i in 0..2i64 {
        w.add(&format!("a.gen.flow.time.{i}.freqs"), [PT_LAT / 2, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.up.weight"), [PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.up.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.down.weight"), [PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.down.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.time.{i}.norm"), [PT_LAT, 1, 1, 1]);
    }
    for il in 0..2i64 {
        w.add(&format!("a.gen.flow.blk.{il}.norm.weight"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.norm.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.up.weight"), [PT_LAT, 2 * PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.up.bias"), [2 * PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.down.weight"), [2 * PT_LAT, PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.down.bias"), [PT_LAT, 1, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.ada.weight"), [PT_LAT, 3 * PT_LAT, 1, 1]);
        w.add(&format!("a.gen.flow.blk.{il}.ada.bias"), [3 * PT_LAT, 1, 1, 1]);
    }

    w.add("a.gen.out_eos.weight", [PT_E, 1, 1, 1]);
    w.add("a.gen.out_eos.bias", [1, 1, 1, 1]);
    w.add("a.gen.input_linear.weight", [PT_LAT, PT_E, 1, 1]);
    w.add("a.gen.emb_mean", [PT_LAT, 1, 1, 1]);
    w.add("a.gen.emb_std", [PT_LAT, 1, 1, 1]);

    // ---- mimi decoder (clip.cpp:3017-3037) -------------------------------
    w.add("a.gen.wav.quant_out.weight", [PT_LAT, PT_C, 1, 1]);
    // depthwise, K = 2*stride (the SEANet/qwen3tts learned-upsample
    // convention) — K == stride would leave no overlap tail, so the graph
    // builds no state_out_up and BOTH the port and the pinned reference
    // abort (clip.cpp:5859 "state_out requested but graph has no ...")
    w.add("a.gen.wav.upsample.weight", [32, 1, PT_C, 1]);
    w.add("a.gen.wav.seanet.conv_in.weight", [3, PT_C, 32, 1]);
    w.add("a.gen.wav.seanet.conv_in.bias", [32, 1, 1, 1]);
    w.add("a.gen.wav.seanet.conv_out.weight", [3, 32, 1, 1]);
    w.add("a.gen.wav.seanet.conv_out.bias", [1, 1, 1, 1]);
    // stages: the decoder walks the ratios backwards — [6, 5, 4]
    for (i, &stride) in [6i64, 5, 4].iter().enumerate() {
        let i = i as i64;
        w.add(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.weight"), [stride + 1, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.scale_conv.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.weight"), [3, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv1.bias"), [32, 1, 1, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.weight"), [1, 32, 32, 1]);
        w.add(&format!("a.gen.wav.seanet.blk.{i}.res_conv2.bias"), [32, 1, 1, 1]);
    }
    for il in 0..2i64 {
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln1.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln1.bias"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_q.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_k.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_v.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.attn_out.weight"), [PT_C, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ls1.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln2.weight"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ln2.bias"), [PT_C, 1, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ffn_up.weight"), [PT_C, 128, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ffn_down.weight"), [128, PT_C, 1, 1]);
        w.add(&format!("a.gen.wav.tfm.blk.{il}.ls2.weight"), [PT_C, 1, 1, 1]);
    }

    w.finish(&path);
    path
}

// ---------------------------------------------------------------------------
// shared fixture inputs + dumps
// ---------------------------------------------------------------------------

fn hstate_blob(n: usize) -> Vec<f32> {
    let mut st = 0xfeed_beefu32;
    (0..n).map(|_| lcg(&mut st) * 0.2).collect()
}

fn feats_blob(n_latent: i64, n_frames: i64) -> Vec<f32> {
    let mut st = 0xd00d_1234u32;
    (0..n_latent * n_frames).map(|_| lcg(&mut st) * 0.2).collect()
}

fn codes_blob(n_frames: i64, n_codes: i64) -> Vec<i32> {
    // frame-major, inside both codebook vocabs
    let mut st = 0xc0de_9999u32;
    (0..n_frames * n_codes)
        .map(|i| {
            let g = i % n_codes;
            let vocab = if g == 0 { Q3_VOCAB0 } else { Q3_VOCAB };
            (lcg(&mut st).abs() * (vocab as f32 - 1.0)) as i32
        })
        .collect()
}

fn dump_blob(path: &str, parts: &[&[u8]]) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut buf = Vec::new();
    for p in parts {
        buf.extend_from_slice(p);
    }
    std::fs::write(path, buf).expect("dump");
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn gen_dump(path: &str, out: &llama::clip::GenOut) {
    // [u32 n_codes][i32][u32 n_embd][f32][u32 n_feats][f32][u32 is_eos]
    // [u32 n_samples][f32][u32 state_size][u8]
    let mut parts: Vec<Vec<u8>> = Vec::new();
    parts.push((out.codes.len() as u32).to_le_bytes().to_vec());
    parts.push(i32_bytes(&out.codes));
    parts.push((out.embd.len() as u32).to_le_bytes().to_vec());
    parts.push(f32_bytes(&out.embd));
    parts.push((out.feats.len() as u32).to_le_bytes().to_vec());
    parts.push(f32_bytes(&out.feats));
    parts.push((out.is_eos as u32).to_le_bytes().to_vec());
    parts.push((out.audio.len() as u32).to_le_bytes().to_vec());
    parts.push(f32_bytes(&out.audio));
    parts.push((out.state.len() as u32).to_le_bytes().to_vec());
    parts.push(out.state.clone());
    let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
    dump_blob(path, &refs);
}

fn load_gen(path: &str) -> llama::clip::ClipContext {
    let params = ClipContextParams {
        flash_attn_type: ClipFlashAttn::Disabled,
        ..Default::default()
    };
    let cx = clip_init_from_file(path, &params).expect("load gen mmproj");
    assert_eq!(cx.model.modality, ClipModality::GenAudio);
    cx
}

#[test]
fn qwen3tts_gen_code_and_wav() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = write_qwen3tts_gen();
    let mut cx = load_gen(&path);
    assert_eq!(cx.model.proj_type, ProjectorType::Qwen3TtsGen);
    assert_eq!(cx.n_mmproj_embd() as i64, Q3_E);

    // GEN_CODE: fixed hidden state + code0, fixed seed (the reference probe
    // reseeds the same mt19937 → identical inp_rand_* draws)
    let h = hstate_blob(Q3_E as usize);
    let inp = GenInp {
        process: ClipGenProcess::GenCode,
        code0: 5,
        embd: Some(&h),
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.9,
        codes: None,
        feats: None,
        state: None,
    };
    let out = cx.gen_audio_process(&inp).expect("gen_code");
    let n_codes = (Q3_NACO + 1) as usize;
    assert_eq!(out.codes.len(), n_codes, "16-code frame");
    assert!(out.codes.iter().all(|&c| c >= 0 && c < Q3_VOCAB as i32));
    assert_eq!(out.embd.len(), Q3_E as usize);
    assert!(out.embd.iter().all(|v| v.is_finite()));
    assert!(out.feats.is_empty(), "qwen3tts is a discrete pipeline");
    gen_dump(&format!("{OUT_DIR}/port-q3t-gencode.bin"), &out);

    // GEN_WAV: two streamed calls — cold start, then with the carried state
    let frames1 = 8i64;
    let codes1 = codes_blob(frames1, n_codes as i64);
    let inp = GenInp {
        process: ClipGenProcess::GenWav,
        code0: 0,
        embd: None,
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.0,
        codes: Some(&codes1),
        feats: None,
        state: None,
    };
    let out1 = cx.gen_audio_process(&inp).expect("gen_wav cold");
    // 8 frames × (2×2 upsample × 2 dac) = 8 × 8 = 64 samples
    assert!(!out1.audio.is_empty());
    assert!(!out1.state.is_empty(), "the streaming state must flow");
    assert!(out1.audio.iter().all(|v| (-1.0..=1.0).contains(v)));
    gen_dump(&format!("{OUT_DIR}/port-q3t-genwav1.bin"), &out1);

    let frames2 = 4i64;
    let codes2 = codes_blob(frames2, n_codes as i64);
    let state = out1.state.clone();
    let inp = GenInp {
        process: ClipGenProcess::GenWav,
        code0: 0,
        embd: None,
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.0,
        codes: Some(&codes2),
        feats: None,
        state: Some(&state),
    };
    let out2 = cx.gen_audio_process(&inp).expect("gen_wav streamed");
    assert!(!out2.audio.is_empty());
    gen_dump(&format!("{OUT_DIR}/port-q3t-genwav2.bin"), &out2);

    // the fixture inputs for the reference probe
    dump_blob(
        &format!("{OUT_DIR}/q3t-hstate.bin"),
        &[&f32_bytes(&h)],
    );
    dump_blob(
        &format!("{OUT_DIR}/q3t-codes1.bin"),
        &[&i32_bytes(&codes1)],
    );
    dump_blob(
        &format!("{OUT_DIR}/q3t-codes2.bin"),
        &[&i32_bytes(&codes2), &state],
    );
}

#[test]
fn pockettts_gen_code_and_wav() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let path = write_pockettts_gen();
    let mut cx = load_gen(&path);
    assert_eq!(cx.model.proj_type, ProjectorType::PocketTtsGen);
    assert_eq!(cx.n_mmproj_embd() as i64, PT_E);

    // GEN_CODE: the flow net integrates one velocity step from the gaussian
    // noise the fixed seed draws
    let h = hstate_blob(PT_E as usize);
    let inp = GenInp {
        process: ClipGenProcess::GenCode,
        code0: 0,
        embd: Some(&h),
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.7,
        codes: None,
        feats: None,
        state: None,
    };
    let out = cx.gen_audio_process(&inp).expect("gen_code");
    assert_eq!(out.feats.len(), PT_LAT as usize, "one latent vector");
    assert_eq!(out.embd.len(), PT_E as usize);
    assert!(out.feats.iter().chain(out.embd.iter()).all(|v| v.is_finite()));
    // the eos bit is a threshold on a random-weight head — it is compared
    // against the reference verbatim in the dump instead of asserted here
    gen_dump(&format!("{OUT_DIR}/port-pt-gencode.bin"), &out);

    // GEN_WAV: cold, then streamed (the mimi decoder state bridges windows)
    let frames = 8i64;
    let feats = feats_blob(PT_LAT, frames);
    let inp = GenInp {
        process: ClipGenProcess::GenWav,
        code0: 0,
        embd: None,
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.7,
        codes: None,
        feats: Some(&feats),
        state: None,
    };
    let out1 = cx.gen_audio_process(&inp).expect("gen_wav cold");
    // 8 latents × 16 upsample × 6*5*4 seanet = 8 × 16 × 120 = 15360 samples
    assert!(!out1.audio.is_empty());
    assert!(!out1.state.is_empty());
    assert!(out1.audio.iter().all(|v| (-1.0..=1.0).contains(v)));
    gen_dump(&format!("{OUT_DIR}/port-pt-genwav1.bin"), &out1);

    let state = out1.state.clone();
    let feats2 = feats_blob(PT_LAT, 4);
    let inp = GenInp {
        process: ClipGenProcess::GenWav,
        code0: 0,
        embd: None,
        top_k: 50,
        top_p: 1.0,
        seed: SEED,
        temp: 0.7,
        codes: None,
        feats: Some(&feats2),
        state: Some(&state),
    };
    let out2 = cx.gen_audio_process(&inp).expect("gen_wav streamed");
    assert!(!out2.audio.is_empty());
    gen_dump(&format!("{OUT_DIR}/port-pt-genwav2.bin"), &out2);

    dump_blob(&format!("{OUT_DIR}/pt-hstate.bin"), &[&f32_bytes(&h)]);
    dump_blob(&format!("{OUT_DIR}/pt-feats1.bin"), &[&f32_bytes(&feats)]);
    dump_blob(
        &format!("{OUT_DIR}/pt-feats2.bin"),
        &[&f32_bytes(&feats2), &state],
    );
}
