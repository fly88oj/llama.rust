//! t5_dec_e2e.rs — the decoder half of the full T5 arch
//! (`llama_model_t5::graph<false>`, src/models/t5.cpp:110-262 + the dec.blk.*
//! tensor set of :60-107 — MTP batch 17, 2026-09-29). The encoder half
//! (`graph<true>`, the t5encoder-arch twin) is tests/t5_e2e.rs.
//!
//! Acceptance: the graph-level bit-compare the MTP batch uses — the port's
//! driver (the arch_batch5 `Driver` shape: a dec_n_layer KV cache + the
//! cross-attention inputs carried by hand, `llama_context::cross`'s capture,
//! llama-context.cpp:1625-1649) replays encode(prompt) → decode(dec_start,
//! argmax, …) and dumps every step's logits row;
//! `parity/ref_t5_dec_dump.c` drives the pinned reference's own
//! llama_encode/llama_decode pair over the identical chain; the
//! `t5_decoder_reference_bitcompare` cell (ignored; runs after the probe)
//! compares byte-for-byte. The in-port default tests pin the loader (both
//! halves' tensor sets) and the chain's finiteness/determinism.

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::context::{EncoderContext, EncoderWeights};
use llama::graph::DecodeInputs;
use llama::graph::AttnParams;
use llama::graph_arch::{self, T5CrossInputs, T5DecoderParams};
use llama::kv_cache::KvCache;
use llama::model::{load_model, LlamaModel};

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const N_VOCAB: i64 = 32000;
const OUT_DIR: &str = "/tmp/mtp2";

const N_LAYER: usize = 2; // encoder AND decoder layer count
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 2;
const HD: i64 = 32;
const N_FF: i64 = 48;
const N_PROMPT: usize = 6;
const N_STEPS: usize = 12;

// ---------------------------------------------------------------------------
// the synth file (arch t5: the enc.blk.* + dec.blk.* sets, no ffn_gate —
// the t5 1.0 RELU-SEQ FFN; attn_rel_b on layer 0 only, both halves)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Role {
    Norm,
    Proj,
    Bias,
}

fn tensors_for() -> Vec<(String, Vec<i64>, Role)> {
    let mut v: Vec<(String, Vec<i64>, Role)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $role:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $role))
        };
    }
    push!("token_embd.weight", vec![N_EMBD, N_VOCAB], Role::Proj);
    push!("output.weight", vec![N_EMBD, N_VOCAB], Role::Proj);
    push!("enc.output_norm.weight", vec![N_EMBD], Role::Norm);
    push!("dec.output_norm.weight", vec![N_EMBD], Role::Norm);
    for i in 0..N_LAYER as i32 {
        // encoder half (t5.cpp:60-78)
        push!(format!("enc.blk.{i}.attn_norm.weight"), vec![N_EMBD], Role::Norm);
        push!(format!("enc.blk.{i}.attn_q.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("enc.blk.{i}.attn_k.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("enc.blk.{i}.attn_v.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("enc.blk.{i}.attn_o.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("enc.blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm);
        push!(format!("enc.blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], Role::Proj);
        push!(format!("enc.blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], Role::Proj);
        #[cfg(ignore_relu_variant)]
        push!(format!("enc.blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], Role::Proj);
        // decoder half (t5.cpp:80-107)
        push!(format!("dec.blk.{i}.attn_norm.weight"), vec![N_EMBD], Role::Norm);
        push!(format!("dec.blk.{i}.attn_q.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.attn_k.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.attn_v.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.attn_o.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.cross_attn_norm.weight"), vec![N_EMBD], Role::Norm);
        push!(format!("dec.blk.{i}.cross_attn_q.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.cross_attn_k.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.cross_attn_v.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.cross_attn_o.weight"), vec![N_EMBD, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.ffn_norm.weight"), vec![N_EMBD], Role::Norm);
        push!(format!("dec.blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], Role::Proj);
        push!(format!("dec.blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], Role::Proj);
    }
    // the relative-bias tables on layer 0 only (t5.cpp:63/:87 — NOT_REQUIRED,
    // the graphs fall back to layer 0's)
    push!("enc.blk.0.attn_rel_b.weight", vec![N_HEAD, 32], Role::Bias);
    push!("dec.blk.0.attn_rel_b.weight", vec![N_HEAD, 32], Role::Bias);
    v
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn build_file() -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    let a = "t5";
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-t5-dec".to_string()));
    kv!("general.file_type", Value::U32(0));
    kv!(format!("{a}.context_length"), Value::U32(512));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.decoder_block_count"), Value::U32(N_LAYER as u32));
    kv!(format!("{a}.decoder_start_token_id"), Value::U32(1));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-6));
    kv!(format!("{a}.attention.relative_buckets_count"), Value::U32(32));

    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f5d1);
    let table = tensors_for();
    for (name, ne, role) in &table {
        let n: i64 = ne.iter().product();
        let s = match role {
            Role::Norm => 1.0,
            Role::Bias => 0.02,
            Role::Proj => 1.0 / (N_EMBD as f32).sqrt(),
        };
        let vals: Vec<f32> = match role {
            Role::Norm => (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect(),
            _ => (0..n).map(|_| s * rng.next()).collect(),
        };
        let ne4 = [
            ne[0],
            *ne.get(1).unwrap_or(&1),
            *ne.get(2).unwrap_or(&1),
            *ne.get(3).unwrap_or(&1),
        ];
        w.add_tensor(name, GgmlType::F32, ne4);
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for x in &vals {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        data.push(bytes);
    }
    let path = format!("{OUT_DIR}/t5-synth-dec.gguf");
    let f = std::fs::File::create(&path).expect("create");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write");
    use std::io::Write as _;
    bw.flush().unwrap();
    (table.len(), std::fs::metadata(&path).unwrap().len())
}

fn open_model() -> LlamaModel {
    let path = format!("{OUT_DIR}/t5-synth-dec.gguf");
    let gguf = Gguf::open(&path).expect("open synth");
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("load synth model")
}

// ---------------------------------------------------------------------------
// the chain: encode(prompt) → decode(dec_start, argmax, …)
// ---------------------------------------------------------------------------

fn synth_attn(m: &LlamaModel) -> AttnParams {
    let hp = &m.hparams;
    AttnParams {
        n_head: N_HEAD,
        n_head_kv: N_HEAD,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: 0, // t5 has no rope
        rope_mode: 0,
        n_ctx_orig: 512,
        freq_base: 10_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        norm_eps: hp.f_norm_rms_eps,
        use_flash_attn: false,
    }
}

struct DecDriver {
    gctx: Context,
    kv: KvCache,
    watermark: usize,
    w: graph_arch::T5DecoderModelWeights,
    p: T5DecoderParams,
    cross_embd: ggml::TensorId,
    cross_mask: ggml::TensorId,
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as i32
}

impl DecDriver {
    fn step(&mut self, tok: i32, pos: i32) -> Vec<f32> {
        let n = 1usize;
        let sinfo = self.kv.find_slot(n as u32).expect("kv full");
        self.kv.assign(sinfo, &[pos], 0);
        let n_kv = self.kv.n_kv();

        self.gctx.reset_graph_to(self.watermark);
        let tokens_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let pos_t = self.gctx.new_tensor_1d(GgmlType::I32, n as i64);
        let kq_mask = self.gctx.new_tensor_2d(GgmlType::F32, n_kv as i64, n as i64);
        let row_idx = self.gctx.new_tensor_1d(GgmlType::I64, n as i64);
        let pos_bucket = self.gctx.new_tensor_2d(GgmlType::I32, n_kv as i64, n as i64);
        for t in [tokens_t, pos_t, kq_mask, row_idx, pos_bucket] {
            self.gctx.arena_resize_tensor(t);
        }
        self.gctx.with_i32_mut(tokens_t, |p| p[0] = tok).unwrap();
        self.gctx.with_i32_mut(pos_t, |p| p[0] = pos).unwrap();
        {
            let bytes = self.gctx.data_bytes_mut(row_idx).unwrap();
            bytes.copy_from_slice(bytemuck::cast_slice(&[sinfo.s0 as i64]));
        }
        {
            // the causal mask over the cells
            let kv_pos: Vec<i32> = self.kv.cells[..n_kv as usize].iter().map(|c| c.pos).collect();
            let mask: &mut [f32] =
                bytemuck::cast_slice_mut(self.gctx.data_bytes_mut(kq_mask).unwrap());
            llama::graph::fill_causal_mask(mask, &kv_pos, &[pos]);
        }
        {
            // set_input_pos_bucket (llama-kv-cache.cpp:1790-1812): the CAUSAL
            // buckets between each cell's pos and the query pos
            let bkts = self.p.n_rel_attn_bkts as u32;
            let cells: Vec<i32> = self.kv.cells[..n_kv as usize].iter().map(|c| c.pos).collect();
            let data: &mut [i32] =
                bytemuck::cast_slice_mut(self.gctx.data_bytes_mut(pos_bucket).unwrap());
            for (j, &p0) in cells.iter().enumerate() {
                let p0 = if p0 < 0 { -1 } else { p0 };
                data[j] = graph_arch::relative_position_bucket(p0, pos, bkts as u64, false);
            }
        }

        let inp = DecodeInputs {
            tokens: tokens_t,
            pos: pos_t,
            kq_mask,
            row_idx,
            out_ids: None,
        };
        let cross = T5CrossInputs {
            cross_embd: self.cross_embd,
            cross_kq_mask: self.cross_mask,
            pos_bucket_dec: pos_bucket,
        };
        let result = graph_arch::build_t5_decoder_forward(
            &mut self.gctx, &self.w, &self.p, &self.kv, &inp, &cross, n_kv, n,
        );
        let logits = result.logits;
        let mut gf = result.graph;
        ggml::compute::graph_compute(&mut self.gctx, &mut gf, 8);
        self.gctx
            .data_bytes(logits)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// the full chain + the dump: per step the fed token + the logits row. The
/// format mirrors parity/ref_t5_dec_dump.c byte-for-byte ("T5DEC\0\0\0").
fn run_chain() -> Vec<(i32, Vec<f32>)> {
    // 1) encode (a separate model load — the EncoderContext owns its ggml
    //    Context; the C shares the model, speculative.cpp:2582's port-side
    //    convention)
    {
        let _ = build_file();
    }
    let prompt: Vec<i32> = (1..=N_PROMPT as i32).collect();
    let mut cross_values: Vec<f32>;
    {
        let mut m = open_model();
        let hp = &m.hparams;
        let enc_w = EncoderWeights::T5Encoder(m.t5_encoder_weights());
        let params = graph_arch::EncoderParams {
            n_head: N_HEAD,
            n_head_kv: N_HEAD,
            n_embd_head: hp.n_embd_head_k(0) as i64,
            n_rel_attn_bkts: hp.n_rel_attn_bkts,
            f_norm_eps: hp.f_norm_eps,
            f_norm_rms_eps: hp.f_norm_rms_eps,
            pool: llama::hparams::LlamaPoolingType::NONE,
            euro_rope: None,
            gemma_swa: None,
            causal: false,
        };
        let gctx = std::mem::replace(&mut m.ctx, Context::new());
        let mut enc = EncoderContext::new(gctx, enc_w, params, 8);
        let emb = enc.encode(&prompt).expect("encode");
        assert_eq!(emb.n_rows, N_PROMPT);
        assert_eq!(emb.n_embd_out, N_EMBD as usize);
        assert!(emb.values.iter().all(|v| v.is_finite()));
        cross_values = emb.values;
    }

    // 2) the decoder chain over a second load
    let mut m = open_model();
    let hp = &m.hparams;
    assert_eq!(hp.dec_n_layer as usize, N_LAYER);
    let w = m.t5_decoder_weights();
    let attn = synth_attn(&m);
    let p = T5DecoderParams {
        n_head: N_HEAD,
        n_head_kv: N_HEAD,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts as i64,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        attn,
    };
    let mut gctx = std::mem::replace(&mut m.ctx, Context::new());
    // the cross inputs (persistent across steps — they live above the
    // watermark like the recurrent cells)
    let cross_embd = gctx.new_tensor_2d(GgmlType::F32, N_EMBD, N_PROMPT as i64);
    let cross_mask = gctx.new_tensor_2d(GgmlType::F32, N_PROMPT as i64, 1);
    for t in [cross_embd, cross_mask] {
        gctx.arena_resize_tensor(t);
    }
    {
        let bytes = gctx.data_bytes_mut(cross_embd).unwrap();
        let f: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        f.copy_from_slice(&cross_values);
    }
    {
        // single sequence: every encoder position attends
        let bytes = gctx.data_bytes_mut(cross_mask).unwrap();
        let f: &mut [f32] = bytemuck::cast_slice_mut(bytes);
        f.iter_mut().for_each(|v| *v = 0.0);
    }
    let kv = KvCache::new_with_dims(
        &mut gctx,
        &[HD * N_HEAD; N_LAYER],
        &[HD * N_HEAD; N_LAYER],
        512,
    );
    let watermark = gctx.mark();
    let mut d = DecDriver {
        gctx,
        kv,
        watermark,
        w,
        p,
        cross_embd,
        cross_mask,
    };

    let mut out = Vec::new();
    let mut tok = hp.dec_start_token_id;
    for step in 0..N_STEPS {
        let lg = d.step(tok, step as i32);
        assert!(lg.iter().all(|v| v.is_finite()), "step {step}: non-finite");
        out.push((tok, lg.clone()));
        tok = argmax(&lg);
    }
    out
}

fn write_dump(path: &str, chain: &[(i32, Vec<f32>)]) {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"T5DEC\0\0\0");
    bytes.extend_from_slice(&(chain.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(N_VOCAB as u32).to_le_bytes());
    for (tok, lg) in chain {
        bytes.extend_from_slice(&tok.to_le_bytes());
        for v in lg {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).expect("write dump");
}

/// loader pins + the chain: both halves' tensor sets consumed, finite logits,
/// deterministic repeat
#[test]
fn t5_decoder_load_and_chain() {
    let _ = build_file();
    let m = open_model();
    assert_eq!(m.arch, llama::arch::LlmArch::T5);
    assert_eq!(m.hparams.dec_n_layer as usize, N_LAYER);
    assert_eq!(m.hparams.dec_start_token_id, 1);
    let mut want: Vec<String> = tensors_for().into_iter().map(|(n, _, _)| n).collect();
    want.sort();
    want.dedup();
    let mut got: Vec<String> = m.tensors.keys().cloned().collect();
    got.sort();
    assert_eq!(got, want, "created tensor set");
    let l = &m.layers[0];
    assert!(l.dec_wo_cross.is_some(), "dec_wo_cross");
    assert!(l.enc_wq.is_some(), "enc_wq (both halves share the vec)");
    assert!(m.enc_output_norm.is_some(), "enc.output_norm slot");
    drop(m);

    let chain = run_chain();
    write_dump(&format!("{OUT_DIR}/t5-dec-port.bin"), &chain);
    // deterministic: a second run is byte-identical
    let chain2 = run_chain();
    assert_eq!(chain.len(), chain2.len());
    for ((t1, a), (t2, b)) in chain.iter().zip(&chain2) {
        assert_eq!(t1, t2);
        assert_eq!(a, b, "the chain must be deterministic");
    }
    println!(
        "t5 decoder chain ok — {} steps, first tok {}, argmax {}",
        chain.len(),
        chain[0].0,
        argmax(&chain[0].1)
    );
}

/// the reference bit-compare (parity/ref_t5_dec_dump writes
/// /tmp/mtp2/t5-dec-ref.bin)
#[test]
#[ignore = "needs /tmp/mtp2/t5-dec-ref.bin (parity/gen_t5_dec_ref.sh)"]
fn t5_decoder_reference_bitcompare() {
    let ref_path = format!("{OUT_DIR}/t5-dec-ref.bin");
    let port_path = format!("{OUT_DIR}/t5-dec-port.bin");
    assert!(
        std::path::Path::new(&ref_path).exists(),
        "run parity/gen_t5_dec_ref.sh first"
    );
    let a = std::fs::read(&ref_path).expect("read ref dump");
    let b = std::fs::read(&port_path).expect("read port dump");
    assert_eq!(&a[..8], b"T5DEC\0\0\0", "ref magic");
    assert_eq!(&b[..8], b"T5DEC\0\0\0", "port magic");
    assert_eq!(a, b, "the t5 decoder chain differs from the reference");
    println!("t5 decoder: reference bit-compare PASS ({} bytes)", a.len());
}

/// debug probe: the encoder rows of the synth file (the cross state) vs
/// /tmp/mtp2/t5-enc-ref.bin — isolates the encoder half of any residual
#[test]
#[ignore = "debug probe"]
fn t5_encoder_rows_isolate() {
    {
        let _ = build_file();
    }
    let mut m = open_model();
    let hp = &m.hparams;
    let enc_w = EncoderWeights::T5Encoder(m.t5_encoder_weights());
    let params = graph_arch::EncoderParams {
        n_head: N_HEAD,
        n_head_kv: N_HEAD,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: llama::hparams::LlamaPoolingType::NONE,
        euro_rope: None,
        gemma_swa: None,
        causal: false,
    };
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let mut enc = EncoderContext::new(gctx, enc_w, params, 8);
    let emb = enc.encode(&(1..=N_PROMPT as i32).collect::<Vec<_>>()).expect("encode");
    let ref_bytes = std::fs::read("/tmp/mtp2/t5-enc-ref.bin").expect("run the C probe first");
    let theirs: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let nd = theirs
        .iter()
        .zip(&emb.values)
        .filter(|(a, b)| a != b)
        .count();
    let mx = theirs
        .iter()
        .zip(&emb.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("t5 enc rows: {nd}/{} differ, max {mx:.3e}", theirs.len());
}

// the node-dump bisect (the mtp2 cb, F32-payload-only rule)
static T5_DUMP: std::sync::OnceLock<std::sync::Mutex<Option<Vec<u8>>>> = std::sync::OnceLock::new();
static T5_NODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn t5_dump_cb(node: &ggml::compute::EvalNode<'_>, ask: bool) -> bool {
    if ask {
        return true;
    }
    let mut guard = T5_DUMP.get_or_init(Default::default).lock().unwrap();
    let Some(out) = guard.as_mut() else { return true };
    let n: i64 = node.ne.iter().product();
    T5_NODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut put = |s: &str| {
        let l = s.len().min(255);
        out.push(l as u8);
        out.extend_from_slice(&s.as_bytes()[..l]);
    };
    put(&format!("{:?}", node.op));
    put(node.name);
    put(match node.ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::I32 => "i32",
        GgmlType::I64 => "i64",
        _ => "other",
    });
    out.extend_from_slice(&node.ne.map(|v| v.to_le_bytes()).concat());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let data = node.data.unwrap_or(&[]);
    if node.ty != GgmlType::F32 || n as u64 >= (1 << 19) {
        return true;
    }
    for flat in 0..n as usize {
        let mut rem = flat as i64;
        let mut off = 0usize;
        for d in 0..4 {
            let idx = rem % node.ne[d];
            rem /= node.ne[d];
            off += (idx as u64 * node.nb[d]) as usize;
        }
        let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        out.extend_from_slice(&v.to_le_bytes());
    }
    true
}

#[test]
#[ignore = "debug probe: T5_NODES=1 dumps the encoder node stream"]
fn t5_encoder_node_dump() {
    {
        let _ = build_file();
    }
    ggml::compute::set_eval_callback(Some(t5_dump_cb));
    T5_DUMP.get_or_init(|| std::sync::Mutex::new(Some(Vec::new())));
    let mut m = open_model();
    let hp = &m.hparams;
    let enc_w = EncoderWeights::T5Encoder(m.t5_encoder_weights());
    let params = graph_arch::EncoderParams {
        n_head: N_HEAD,
        n_head_kv: N_HEAD,
        n_embd_head: hp.n_embd_head_k(0) as i64,
        n_rel_attn_bkts: hp.n_rel_attn_bkts,
        f_norm_eps: hp.f_norm_eps,
        f_norm_rms_eps: hp.f_norm_rms_eps,
        pool: llama::hparams::LlamaPoolingType::NONE,
        euro_rope: None,
        gemma_swa: None,
        causal: false,
    };
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    let mut enc = EncoderContext::new(gctx, enc_w, params, 8);
    let _ = enc.encode(&(1..=N_PROMPT as i32).collect::<Vec<_>>()).expect("encode");
    ggml::compute::set_eval_callback(None);
    let st = T5_DUMP.get().unwrap().lock().unwrap().take().unwrap();
    std::fs::write("/tmp/mtp2/t5-enc-nodes-port.bin", &st).unwrap();
    println!(
        "t5 enc nodes: {} -> /tmp/mtp2/t5-enc-nodes-port.bin",
        T5_NODES.load(std::sync::atomic::Ordering::Relaxed)
    );
}
