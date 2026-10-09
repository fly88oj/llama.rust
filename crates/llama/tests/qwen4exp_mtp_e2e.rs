//! qwen4exp_mtp_e2e.rs — the qwen4exp `graph_mtp` draft head (qwen4exp.cpp:
//! 526-612) driven end-to-end through the port's MTP context
//! (`DecodeContext::new_mtp` + `MtpForward::Qwen4Exp`), on a synthetic
//! nextn GGUF (the qwen4exp_qsa_dump.rs writer's shape + a 5th block_count
//! MTP layer: full-attention **QSA** — `compress_ratios = [0,0,4,0,4]` —
//! plus the `blk.4.nextn.*` trio and the `nextn hc_head_*` mixer).
//!
//! The acceptance is the mtp2 protocol: the 12-step draft chain (token 1 +
//! the zero h row at pos 0, then (argmax, h_nextn) per step — the h rows
//! are **hc-wide** here, `n_embd_out() == n_embd * hc`) dumps every step's
//! t_logits row and t_h_nextn row; `parity/ref_mtp2_dump` replays the
//! identical chain through the NEW reference's own MTP context and
//! `qwen4exp_mtp_reference_bitcompare` byte-compares the two dumps.
//!
//! The MTP context's hard part (batch 42e/42f): the draft memory keeps only
//! the MTP block's attention AND indexer (llama-model.cpp:2750-2756) — the
//! recurrent half is empty, so the port's `decode_batch` prologue now
//! builds the q4e kpool step (+ the PLE cells) for the draft context too.

use std::io::Write as _;
use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Context, Gguf, GgufType, Value};
use llama::arch::LlmArch;
use llama::batch::LlamaBatch;
use llama::context::{DecodeContext, ForwardWeights, MtpForward};
use llama::graph::AttnParams;
use llama::graph_arch;
use llama::model::{load_model, LlamaModel};
use memmap2::Mmap;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const OUT_DIR: &str = "/tmp/mtp2";
const N_STEPS: usize = 12;

// the qwen4exp_qsa_dump.rs geometry + one MTP block
const N_LAYER: i64 = 4; // trunk
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4;
const KLEN: i64 = 16;
const HEAD_KV: i64 = 2;
const HC: i64 = 2;
const HC_LR: i64 = 8;
const N_FF_EXP: i64 = 32;
const N_FF_SH: i64 = 32;
const N_EXPERT: i64 = 4;
const IDX_HEADS: i64 = 2;
const IDX_DIM: i64 = 16;
const IDX_TOPK: i64 = 8;
const RATIO: i64 = 4;
const N_CTX: u32 = 512;

fn path() -> String {
    format!("{OUT_DIR}/qwen4exp-synth-mtp.gguf")
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

/// the qwen4exp_qsa_dump.rs table + the MTP block (a full-attention QSA
/// layer + the nextn trio + the nextn hc head mixer)
fn tensors_for(n_vocab: i64) -> Vec<(String, Vec<i64>, f32)> {
    let mut v: Vec<(String, Vec<i64>, f32)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $scale:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $scale))
        };
    }
    let hc = HC;
    let hc_dim = hc * N_EMBD;
    let (head_k, n_k, head_v, n_v) = (8i64, 1i64, 8i64, 2i64);
    let value_dim = head_v * n_v;
    let conv_dim = head_k * n_k * 2 + value_dim;

    push!("token_embd.weight", vec![N_EMBD, n_vocab], 0.11);
    push!("output_hc_norm.weight", vec![N_EMBD, hc], 0.2);
    push!("output_hc_down.weight", vec![hc_dim, HC_LR], 0.11);
    push!("output_hc_up.weight", vec![HC_LR, hc_dim], 0.11);
    push!("output.weight", vec![N_EMBD, n_vocab], 0.11);

    for i in 0..(N_LAYER + 1) {
        // the MTP layer (4) is a full-attention QSA block like layer 2
        let recr = i != 2 && i != N_LAYER;
        push!(format!("blk.{i}.hc_attn_norm.weight"), vec![N_EMBD, hc], 0.2);
        push!(format!("blk.{i}.hc_attn_down.weight"), vec![hc_dim, HC_LR], 0.11);
        push!(format!("blk.{i}.hc_attn_up.weight"), vec![HC_LR, hc_dim], 0.11);
        push!(format!("blk.{i}.hc_attn_inject.weight"), vec![hc_dim, hc], 0.2);
        push!(format!("blk.{i}.hc_ffn_norm.weight"), vec![N_EMBD, hc], 0.2);
        push!(format!("blk.{i}.hc_ffn_down.weight"), vec![hc_dim, HC_LR], 0.11);
        push!(format!("blk.{i}.hc_ffn_up.weight"), vec![HC_LR, hc_dim], 0.11);
        push!(format!("blk.{i}.hc_ffn_inject.weight"), vec![hc_dim, hc], 0.2);

        if !recr {
            let q = N_HEAD * KLEN * 2; // [q|gate]
            let kv = HEAD_KV * KLEN;
            push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, q], 0.11);
            push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, kv], 0.11);
            push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * KLEN, N_EMBD], 0.11);
            push!(format!("blk.{i}.attn_q_norm.weight"), vec![KLEN], 0.2);
            push!(format!("blk.{i}.attn_k_norm.weight"), vec![KLEN], 0.2);
            push!(format!("blk.{i}.indexer.q_proj.weight"), vec![N_EMBD, IDX_HEADS * IDX_DIM], 0.11);
            push!(format!("blk.{i}.indexer.k_proj.weight"), vec![N_EMBD, IDX_DIM], 0.11);
            push!(format!("blk.{i}.indexer.q_norm.weight"), vec![IDX_DIM], 0.2);
            push!(format!("blk.{i}.indexer.k_norm.weight"), vec![IDX_DIM], 0.2);
        } else {
            push!(format!("blk.{i}.attn_qkv.weight"), vec![N_EMBD, conv_dim], 0.11);
            push!(format!("blk.{i}.attn_gate.weight"), vec![N_EMBD, value_dim], 0.11);
            push!(format!("blk.{i}.ssm_conv1d.weight"), vec![4, conv_dim], 0.2);
            push!(format!("blk.{i}.ssm_dt.bias"), vec![n_v], 0.2);
            push!(format!("blk.{i}.ssm_a"), vec![n_v], 0.3);
            push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, n_v], 0.2);
            push!(format!("blk.{i}.ssm_alpha.weight"), vec![N_EMBD, n_v], 0.2);
            push!(format!("blk.{i}.ssm_norm.weight"), vec![head_v], 0.2);
            push!(format!("blk.{i}.ssm_out.weight"), vec![value_dim, N_EMBD], 0.11);
        }

        push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_down_exps.weight"), vec![N_FF_EXP, N_EMBD, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_gate_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_up_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], 0.11);
        push!(format!("blk.{i}.ffn_gate_inp_shexp.weight"), vec![N_EMBD], 0.11);
        push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![N_EMBD, N_FF_SH], 0.11);
        push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![N_EMBD, N_FF_SH], 0.11);
        push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_SH, N_EMBD], 0.11);

        if i >= N_LAYER {
            // the nextn trio + the nextn hc head mixer (qwen4exp.cpp:274-285)
            push!(format!("blk.{i}.nextn.eh_proj.weight"), vec![2 * N_EMBD, N_EMBD], 0.11);
            push!(format!("blk.{i}.nextn.enorm.weight"), vec![N_EMBD], 0.2);
            push!(format!("blk.{i}.nextn.hnorm.weight"), vec![N_EMBD, hc], 0.2);
            push!(format!("blk.{i}.nextn.hc_head_norm.weight"), vec![N_EMBD, hc], 0.2);
            push!(format!("blk.{i}.nextn.hc_head_down.weight"), vec![hc_dim, HC_LR], 0.11);
            push!(format!("blk.{i}.nextn.hc_head_up.weight"), vec![HC_LR, hc_dim], 0.11);
        }
    }
    v
}

fn build_file(path: &str) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/mtp2");

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let n_vocab = src.kv.iter()
        .find(|(k, _)| k == "tokenizer.ggml.tokens")
        .and_then(|(_, v)| v.as_array().map(|a| a.1.len()))
        .expect("tokens array") as i64;

    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "qwen4exp";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-qwen4exp-mtp".to_string()));
    kv!("general.file_type", Value::U32(0));
    kv!(format!("{a}.context_length"), Value::U32(N_CTX));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32((N_LAYER + 1) as u32));
    kv!(format!("{a}.nextn_predict_layers"), Value::U32(1));
    kv!(format!("{a}.feed_forward_length"), Value::U32(32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::Array(GgufType::Uint32, vec![Value::U32(2); (N_LAYER + 1) as usize])
    );
    kv!(format!("{a}.attention.key_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(KLEN as u32));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(
        format!("{a}.rope.dimension_sections"),
        Value::Array(GgufType::Int32, vec![Value::I32(8), Value::I32(4), Value::I32(4), Value::I32(0)])
    );
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
    kv!(format!("{a}.expert_shared_feed_forward_length"), Value::U32(32));
    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
    kv!(format!("{a}.ssm.inner_size"), Value::U32(16));
    kv!(format!("{a}.ssm.state_size"), Value::U32(8));
    kv!(format!("{a}.ssm.time_step_rank"), Value::U32(2));
    kv!(format!("{a}.ssm.group_count"), Value::U32(1));
    kv!(format!("{a}.hyper_connection.count"), Value::U32(HC as u32));
    kv!(format!("{a}.hyper_connection.low_rank"), Value::U32(HC_LR as u32));
    kv!(format!("{a}.attention.indexer.head_count"), Value::U32(IDX_HEADS as u32));
    kv!(format!("{a}.attention.indexer.key_length"), Value::U32(IDX_DIM as u32));
    kv!(format!("{a}.attention.indexer.top_k"), Value::U32(IDX_TOPK as u32));
    // the QSA half: layer 2 and the MTP layer 4 compress blocks of RATIO
    // tokens
    kv!(
        format!("{a}.attention.compress_ratios"),
        Value::Array(
            GgufType::Uint32,
            vec![
                Value::U32(0),
                Value::U32(0),
                Value::U32(RATIO as u32),
                Value::U32(0),
                Value::U32(RATIO as u32)
            ]
        )
    );
    kv!(
        format!("{a}.attention.recurrent_layers"),
        Value::Array(
            GgufType::Uint32,
            vec![
                Value::U32(1),
                Value::U32(1),
                Value::U32(0),
                Value::U32(1),
                Value::U32(0)
            ]
        )
    );

    let table = tensors_for(n_vocab);
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0xbee_f000 ^ 7);
    for (name, ne, scale) in &table {
        let n: i64 = ne.iter().product();
        let vals: Vec<f32> = (0..n).map(|_| rng.next() * scale).collect();
        let ne4 = [ne[0], *ne.get(1).unwrap_or(&1), *ne.get(2).unwrap_or(&1), *ne.get(3).unwrap_or(&1)];
        w.add_tensor(name, GgmlType::F32, ne4);
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for x in &vals {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        data.push(bytes);
    }

    let f = std::fs::File::create(path).expect("create synth gguf");
    let mut bw = std::io::BufWriter::new(f);
    let refs: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    w.write(&mut bw, &refs).expect("write synth gguf");
    bw.flush().unwrap();
    drop(bw);
    let bytes = std::fs::metadata(path).unwrap().len();
    (table.len(), bytes)
}

static FILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ensure_synth() {
    let _g = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !std::path::Path::new(&path()).exists() {
        let (n, bytes) = build_file(&path());
        println!("qwen4exp mtp synth: {n} tensors, {bytes} bytes -> {}", path());
    }
}

fn load_synth() -> LlamaModel {
    ensure_synth();
    let p = path();
    let gguf = Gguf::open(&p).expect("open synth");
    let f = std::fs::File::open(&p).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("qwen4exp synth must load")
}

// ---------------------------------------------------------------------------
// the weights bundle (the qwen4exp_qsa_dump.rs converters + the MTP arm)
// ---------------------------------------------------------------------------

fn layer_weights(l: &llama::model::LayerTensors) -> graph_arch::Qwen4ExpLayerWeights {
    graph_arch::Qwen4ExpLayerWeights {
        hc_attn_norm: l.hc_attn_norm.expect("hc_attn_norm"),
        hc_attn_down: l.hc_attn_down.expect("hc_attn_down"),
        hc_attn_up: l.hc_attn_up.expect("hc_attn_up"),
        hc_attn_inject: l.hc_attn_inject.expect("hc_attn_inject"),
        hc_ffn_norm: l.hc_ffn_norm.expect("hc_ffn_norm"),
        hc_ffn_down: l.hc_ffn_down.expect("hc_ffn_down"),
        hc_ffn_up: l.hc_ffn_up.expect("hc_ffn_up"),
        hc_ffn_inject: l.hc_ffn_inject.expect("hc_ffn_inject"),
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wo: l.wo,
        attn_q_norm: l.attn_q_norm,
        attn_k_norm: l.attn_k_norm,
        index_q_proj: l.index_q_proj,
        index_k_proj: l.index_k_proj,
        index_q_norm: l.index_q_norm,
        index_k_norm: l.index_k_norm,
        ple_key: l.ple_key,
        ple_value: l.ple_value,
        ple_norm_key: l.ple_norm_key,
        ple_norm_query: l.ple_norm_query,
        ple_norm_conv: l.ple_norm_conv,
        ple_conv1d: l.ple_conv1d,
        wqkv: l.wqkv,
        wqkv_gate: l.wqkv_gate,
        ssm_conv1d: l.ssm_conv1d,
        ssm_dt_b: l.ssm_dt_b,
        ssm_a: l.ssm_a,
        ssm_beta: l.ssm_beta,
        ssm_alpha: l.ssm_alpha,
        ssm_norm: l.ssm_norm,
        ssm_out: l.ssm_out,
        ffn_gate_inp: l.ffn_gate_inp.expect("ffn_gate_inp"),
        ffn_gate_up_exps: l.ffn_gate_up_exps,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_down_exps: l.ffn_down_exps.expect("ffn_down_exps"),
        ffn_gate_inp_shexp: l.ffn_gate_inp_shexp,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
    }
}

fn params_of(
    hp: &llama::hparams::LlamaHparams,
    n_all: usize,
    attn: AttnParams,
) -> graph_arch::Qwen4ExpParams {
    graph_arch::Qwen4ExpParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: (0..n_all).map(|il| hp.n_head(il)).collect(),
        n_head_kv: (0..n_all).map(|il| hp.n_head_kv(il)).collect(),
        n_embd_head_k: (0..n_all).map(|il| hp.n_embd_head_k(il)).collect(),
        n_embd_head_v: (0..n_all).map(|il| hp.n_embd_head_v(il)).collect(),
        n_rot: (0..n_all).map(|il| hp.n_rot(il)).collect(),
        is_recr: (0..n_all).map(|il| hp.is_recr(il)).collect(),
        rope_sections: hp.rope_sections,
        f_attention_scale: hp.f_attention_scale,
        hc: hp.dsv4_hc_mult as i64,
        hc_lr: hp.hc_low_rank as i64,
        ssm_d_conv: hp.ssm_d_conv as i64,
        ssm_d_inner: hp.ssm_d_inner as i64,
        ssm_d_state: hp.ssm_d_state as i64,
        ssm_dt_rank: hp.ssm_dt_rank as i64,
        ssm_n_group: hp.ssm_n_group as i64,
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        expert_weights_scale: hp.expert_weights_scale,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        compress_ratios: hp.dsv4_compress_ratios[..n_all].to_vec(),
        is_ple: (0..n_all).map(|il| hp.is_ple(il)).collect(),
        ple_ngram_size: hp.ple_ngram_size as i64,
        ple_heads_per_ngram: hp.ple_heads_per_ngram as i64,
        ple_conv_kernel: hp.ple_conv_kernel as i64,
        ple_n_heads: hp.ple_n_heads as i64,
        ple_head_dim: hp.ple_head_dim as i64,
        ple_eos_token_id: hp.ple_eos_token_id as i64,
        ple_image_token_id: hp.ple_image_token_id as i64,
        ple_layer_multipliers: hp.ple_layer_multipliers,
        ple_head_offsets: hp.ple_head_offsets,
        ple_head_vocab_sizes: hp.ple_head_vocab_sizes,
    }
}

fn forward_of(m: &LlamaModel) -> (ForwardWeights, MtpForward, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let n_all = n_trunk + 1;
    let rope = hp.rope_runtime();
    let attn = AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: hp.n_head_kv(0) as i64,
        n_embd_head_k: hp.n_embd_head_k(0) as i64,
        n_embd_head_v: hp.n_embd_head_v(0) as i64,
        n_rot: hp.n_rot(0) as i64,
        rope_mode: hp.rope_type as i32,
        n_ctx_orig: rope.n_ctx_orig_yarn,
        freq_base: hp.rope_freq_base_train,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        norm_eps: hp.f_norm_rms_eps,
        // the ref probe's flash_attn_type = DISABLED (the verified baseline)
        use_flash_attn: false,
    };
    // the params' per-layer vectors cover n_layer_all (the MTP layer's own
    // facts at il = n_layer — the extension the CLI's assembly does)
    let params = params_of(hp, n_all, attn);
    let weights = ForwardWeights::Qwen4Exp(
        graph_arch::Qwen4ExpModelWeights {
            tok_embd: m.tok_embd,
            hc_head_norm: m.output_norm,
            hc_head_down: m.hc_head_down.expect("hc_head_down"),
            hc_head_up: m.hc_head_up.expect("hc_head_up"),
            output: m.output,
            per_layer_tok_embd: m.per_layer_tok_embd,
            layers: m.layers[..n_trunk].iter().map(layer_weights).collect(),
        },
        params.clone(),
    );
    let l = &m.layers[n_trunk];
    let mtp = MtpForward::Qwen4Exp(
        graph_arch::Qwen4ExpMtpWeights {
            tok_embd: m.tok_embd,
            output: m.output,
            nextn_eh_proj: l.nextn.eh_proj.expect("nextn.eh_proj"),
            nextn_enorm: l.nextn.enorm.expect("nextn.enorm"),
            nextn_hnorm: l.nextn.hnorm.expect("nextn.hnorm"),
            nextn_hc_head_norm: l.nextn.hc_head_norm.expect("nextn.hc_head_norm"),
            nextn_hc_head_down: l.nextn.hc_head_down.expect("nextn.hc_head_down"),
            nextn_hc_head_up: l.nextn.hc_head_up.expect("nextn.hc_head_up"),
            layer: layer_weights(l),
        },
        params,
    );
    (weights, mtp, attn)
}

fn mtp_driver_of(m: &mut LlamaModel) -> DecodeContext {
    let (weights, mtp, attn) = forward_of(m);
    let gctx = std::mem::replace(&mut m.ctx, Context::new());
    DecodeContext::new_mtp(gctx, weights, mtp, attn, N_CTX, 8, 512)
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

/// the 12-step draft chain (ref_mtp2_dump's loop) — the h rows are
/// n_embd*hc wide (n_embd_out of this arch)
fn run_chain(m: &mut LlamaModel) -> Vec<(i32, Vec<f32>, Vec<f32>)> {
    let hc = m.hparams.dsv4_hc_mult as usize;
    let n_embd = m.hparams.n_embd as usize * hc;
    let n_vocab = m.ctx.ne(m.output)[1] as usize;
    let mut d = mtp_driver_of(m);
    d.set_embeddings_nextn(true, false);

    let mut out = Vec::new();
    let mut tok = 1i32;
    let mut h = vec![0.0f32; n_embd];
    for step in 0..N_STEPS {
        let mut b = LlamaBatch::default();
        b.add(tok, step as i32, &[0], true);
        b.embd = Some(h.clone());
        let o = d.decode_batch(&b).expect("qwen4exp mtp draft step");
        assert_eq!(o.n_outputs, 1);
        let lg = o.logits[..n_vocab].to_vec();
        let hnext = d.get_embeddings_nextn_ith(0).to_vec();
        assert!(
            lg.iter().all(|v| v.is_finite()),
            "qwen4exp mtp: non-finite logits at step {step}"
        );
        assert!(
            hnext.iter().all(|v| v.is_finite()),
            "qwen4exp mtp: non-finite h_nextn at step {step}"
        );
        out.push((tok, lg.clone(), hnext.clone()));
        tok = argmax(&lg);
        h = hnext;
    }
    out
}

fn write_dump(path: &str, chain: &[(i32, Vec<f32>, Vec<f32>)]) {
    assert!(!chain.is_empty());
    let n_vocab = chain[0].1.len();
    let n_embd = chain[0].2.len();
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"MTP2P\0\0\0");
    bytes.extend_from_slice(&(chain.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_vocab as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_embd as u32).to_le_bytes());
    for (tok, lg, h) in chain {
        bytes.extend_from_slice(&tok.to_le_bytes());
        for v in lg {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in h {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).expect("write dump");
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// the default cell: the synth loads, the MTP draft context drives the
/// 12-step chain with finite values, and the dump lands for the bit-compare
#[test]
fn qwen4exp_mtp_synth_load_and_chain() {
    ensure_synth();
    let mut m = load_synth();
    assert_eq!(m.arch, LlmArch::QWEN4EXP);
    assert_eq!(m.hparams.n_layer(), N_LAYER as u32);
    assert_eq!(m.hparams.n_layer_nextn, 1);
    let l = &m.layers[N_LAYER as usize];
    assert!(l.nextn.eh_proj.is_some(), "the nextn trio");
    assert!(l.nextn.hc_head_norm.is_some(), "the nextn hc head mixer");
    assert!(l.index_q_proj.is_some(), "the MTP layer's QSA indexer");
    assert!(l.wq.is_some(), "the MTP layer is a full-attention block");

    let chain = run_chain(&mut m);
    write_dump(&format!("{OUT_DIR}/qwen4exp-port.bin"), &chain);
    let (tok0, lg, _) = &chain[0];
    println!(
        "qwen4exp mtp: chain ok — {} steps, first tok {tok0}, argmax {}",
        chain.len(),
        argmax(lg)
    );
}

/// the reference bit-compare: parity/gen_qwen4exp_mtp_ref.sh drives
/// parity/ref_mtp2_dump over the same synth; this cell compares the two
/// dumps byte-for-byte
#[test]
#[ignore = "needs /tmp/mtp2/qwen4exp-ref.bin (parity/gen_qwen4exp_mtp_ref.sh)"]
fn qwen4exp_mtp_reference_bitcompare() {
    let ref_path = format!("{OUT_DIR}/qwen4exp-ref.bin");
    let port_path = format!("{OUT_DIR}/qwen4exp-port.bin");
    assert!(
        std::path::Path::new(&ref_path).exists(),
        "run parity/gen_qwen4exp_mtp_ref.sh first"
    );
    let a = std::fs::read(&ref_path).expect("read ref dump");
    let b = std::fs::read(&port_path).expect("read port dump");
    assert_eq!(&a[..8], b"MTP2P\0\0\0", "ref magic");
    assert_eq!(&b[..8], b"MTP2P\0\0\0", "port magic");
    if a != b {
        let rd_u32 = |o: usize| u32::from_le_bytes(a[o..o + 4].try_into().unwrap());
        let n_steps = rd_u32(8) as usize;
        let n_vocab = rd_u32(12) as usize;
        let n_embd = rd_u32(16) as usize;
        let row = 4 + 4 * (n_vocab + n_embd);
        let mut step = usize::MAX;
        for s in 0..n_steps {
            let o = 20 + s * row;
            if a[o..o + row] != b[o..o + row] {
                step = s;
                break;
            }
        }
        panic!(
            "qwen4exp mtp dump differs (first at step {step}; {n_steps} steps, n_vocab \
             {n_vocab}, n_embd {n_embd})"
        );
    }
    println!(
        "qwen4exp mtp: reference bit-compare PASS ({} bytes)",
        a.len()
    );
}

/// the #[ignore] generator for the parity runs
#[test]
#[ignore = "writes /tmp/mtp2/qwen4exp-synth-mtp.gguf + the port chain dump"]
fn qwen4exp_mtp_write_synth_and_chain() {
    {
        let _g = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (n, bytes) = build_file(&path());
        println!("qwen4exp mtp synth: {n} tensors, {bytes} bytes -> {}", path());
    }
    qwen4exp_mtp_synth_load_and_chain();
}
