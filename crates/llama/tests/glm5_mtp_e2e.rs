//! glm5_mtp_e2e.rs — the glm5-next `graph_mtp` draft head (b9acf138a,
//! src/models/glm5-next.cpp:543-666) driven end-to-end through the port's
//! `LLAMA_CONTEXT_TYPE_MTP` context (`DecodeContext::new_mtp` +
//! `MtpForward::Glm5Next`), on the batch-19 synthetic GGUF (the writer of
//! tests/glm5_e2e.rs, seeded identically — the file already ships the
//! NextN block: `block_count = n_layer + 1`, the `blk.{n_layer}.nextn.*`
//! trio, a full-indexer DSA NextN layer).
//!
//! The acceptance follows the mtp2 protocol (tests/mtp2_e2e.rs +
//! parity/gen_mtp2_ref.sh): the 12-step draft chain (token 1 + the zero h
//! row at pos 0, then (argmax, h_nextn) per step — speculative.cpp:1616-
//! 1767's non-chained, non-mem-shared shape) dumps every step's t_logits
//! row and t_h_nextn row; `parity/ref_mtp2_dump` (the NEW reference, arch
//! agnostic — `ctx_type = LLAMA_CONTEXT_TYPE_MTP` + `load_mtp = true`)
//! replays the identical chain and `glm5_mtp_reference_bitcompare`
//! byte-compares the two dumps. The node-stream bisect (DECDMP1 rules,
//! ref probe's `--nodes`) rides the same generator.
//!
//! The MTP context's hard part is the HybridIdxCache step: the nextn layer
//! is a DSA+kpool layer, so the draft context's kpool inputs must step with
//! it (llama-model.cpp:2501-2508's `filter_attn = filter_idx = il >=
//! n_layer()` — the port sizes the caches n_layer_all with zero-width
//! trunk rows and builds the kpool step per decode, `build_idx_step` now
//! hooked into `decode_batch`'s prologue like the C's apply()).

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
use llama::vocab::Vocab;
use memmap2::Mmap;

const VOCAB_SPM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf");
const OUT_DIR: &str = "/tmp/mtp2";
const N_STEPS: usize = 12;

// the synthetic geometry — tests/glm5_e2e.rs's constants (must stay in
// lockstep: the same seed writes the same file both sides of the bit-compare)
const N_LAYER: usize = 3; // trunk + 1 NextN block
const N_EMBD: i64 = 64;
const N_HEAD: i64 = 4; // KDA heads == MLA heads on this file
const D_CONV: i64 = 4;
const HEAD_DIM_KDA: i64 = 16; // d_inner = 4*16 = 64
const KV_LORA: i64 = 16;
const Q_LORA: i64 = 16;
const K_MLA: i64 = 24; // nope-only (n_rot == 0)
const V_MLA: i64 = 16;
const N_FF: i64 = 32;
const N_FF_EXP: i64 = 32;
const N_EXPERT: i64 = 4;
const N_EXPERT_SHARED: i64 = 1;
const INDEXER_HEAD: i64 = 4;
const INDEXER_HEAD_SIZE: i64 = 16;
const TOP_K: i64 = 8;
const KPOOL: i64 = 4; // TOP_K % KPOOL == 0, > 1
const N_CTX: u32 = 512;

fn path() -> String {
    format!("{OUT_DIR}/glm5-next-synth-mtp.gguf")
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

/// the tensor table of tests/glm5_e2e.rs (glm5-next.cpp:61-186): layer 0
/// KDA, 1 full-indexer DSA, 2 shared-indexer DSA, 3 the NextN DSA block.
fn tensors_for(n_vocab: i64) -> Vec<(String, Vec<i64>, bool)> {
    let mut v: Vec<(String, Vec<i64>, bool)> = Vec::new();
    macro_rules! push {
        ($name:expr, $ne:expr, $norm:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $norm))
        };
    }
    let hc = 4i64;
    let hc_mix = (2 + hc) * hc;
    let d_inner = N_HEAD * HEAD_DIM_KDA;
    let qk_nope = K_MLA; // n_rot == 0

    push!("token_embd.weight", vec![N_EMBD, n_vocab], false);
    push!("output_norm.weight", vec![N_EMBD], true);
    push!("output.weight", vec![N_EMBD, n_vocab], false);

    for i in 0..(N_LAYER + 1) as i64 {
        push!(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], true);
        push!(format!("blk.{i}.ffn_norm.weight"), vec![N_EMBD], true);

        if i < N_LAYER as i64 {
            push!(format!("blk.{i}.hc_attn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
            push!(format!("blk.{i}.hc_attn_base.weight"), vec![hc_mix], false);
            push!(format!("blk.{i}.hc_attn_scale.weight"), vec![3], false);
            push!(format!("blk.{i}.hc_ffn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
            push!(format!("blk.{i}.hc_ffn_base.weight"), vec![hc_mix], false);
            push!(format!("blk.{i}.hc_ffn_scale.weight"), vec![3], false);
        }

        if i == 0 {
            push!(format!("blk.{i}.ssm_conv1d_q.weight"), vec![D_CONV, 1, d_inner, 1], false);
            push!(format!("blk.{i}.ssm_conv1d_k.weight"), vec![D_CONV, 1, d_inner, 1], false);
            push!(format!("blk.{i}.ssm_conv1d_v.weight"), vec![D_CONV, 1, d_inner, 1], false);
            push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, d_inner], false);
            push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, d_inner], false);
            push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, d_inner], false);
            push!(format!("blk.{i}.ssm_f_a.weight"), vec![N_EMBD, HEAD_DIM_KDA], false);
            push!(format!("blk.{i}.ssm_f_b.weight"), vec![HEAD_DIM_KDA, d_inner], false);
            push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, N_HEAD], false);
            push!(format!("blk.{i}.ssm_a"), vec![N_HEAD], false);
            push!(format!("blk.{i}.ssm_dt.bias"), vec![d_inner], false);
            push!(format!("blk.{i}.ssm_g_a.weight"), vec![N_EMBD, HEAD_DIM_KDA], false);
            push!(format!("blk.{i}.ssm_g_b.weight"), vec![HEAD_DIM_KDA, d_inner], false);
            push!(format!("blk.{i}.ssm_norm.weight"), vec![HEAD_DIM_KDA], true);
            push!(format!("blk.{i}.attn_output.weight"), vec![d_inner, N_EMBD], false);
        } else {
            push!(format!("blk.{i}.attn_q_a_norm.weight"), vec![Q_LORA], true);
            push!(format!("blk.{i}.attn_kv_a_norm.weight"), vec![KV_LORA], true);
            push!(format!("blk.{i}.attn_q_a.weight"), vec![N_EMBD, Q_LORA], false);
            push!(format!("blk.{i}.attn_q_b.weight"), vec![Q_LORA, N_HEAD * K_MLA], false);
            push!(format!("blk.{i}.attn_kv_a_mqa.weight"), vec![N_EMBD, KV_LORA], false);
            push!(format!("blk.{i}.attn_k_b.weight"), vec![qk_nope, KV_LORA, N_HEAD], false);
            push!(format!("blk.{i}.attn_v_b.weight"), vec![KV_LORA, V_MLA, N_HEAD], false);
            push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * V_MLA, N_EMBD], false);

            let full = i != 2;
            if full {
                push!(format!("blk.{i}.indexer.k_norm.weight"), vec![INDEXER_HEAD_SIZE], true);
                push!(format!("blk.{i}.indexer.k_norm.bias"), vec![INDEXER_HEAD_SIZE], true);
                push!(format!("blk.{i}.indexer.proj.weight"), vec![N_EMBD, INDEXER_HEAD], false);
                push!(format!("blk.{i}.indexer.attn_k.weight"), vec![N_EMBD, INDEXER_HEAD_SIZE], false);
                push!(format!("blk.{i}.indexer.attn_q_b.weight"), vec![Q_LORA, INDEXER_HEAD * INDEXER_HEAD_SIZE], false);
                push!(format!("blk.{i}.indexer_compressor_gate.weight"), vec![N_EMBD, INDEXER_HEAD_SIZE], false);
                push!(format!("blk.{i}.indexer_compressor_ape.weight"), vec![INDEXER_HEAD_SIZE, KPOOL], false);
            }
        }

        push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], false);
        push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], true);
        push!(format!("blk.{i}.ffn_gate_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_down_exps.weight"), vec![N_FF_EXP, N_EMBD, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_up_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);
        push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_EXP * N_EXPERT_SHARED, N_EMBD], false);
        push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);

        if i >= N_LAYER as i64 {
            push!(format!("blk.{i}.nextn.eh_proj.weight"), vec![2 * N_EMBD, N_EMBD], false);
            push!(format!("blk.{i}.nextn.enorm.weight"), vec![N_EMBD], true);
            push!(format!("blk.{i}.nextn.hnorm.weight"), vec![N_EMBD], true);
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

    let a = "glm5-next";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!("general.name", Value::String("llama-rust-synth-glm5-next".to_string()));
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(N_CTX));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    kv!(format!("{a}.block_count"), Value::U32((N_LAYER + 1) as u32));
    kv!(format!("{a}.nextn_predict_layers"), Value::U32(1));
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(
        format!("{a}.attention.head_count_kv"),
        Value::Array(GgufType::Uint32, vec![Value::U32(0), Value::U32(1), Value::U32(1), Value::U32(1)])
    );
    kv!(format!("{a}.attention.key_length"), Value::U32((KV_LORA) as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(V_MLA as u32));
    kv!(format!("{a}.attention.key_length_mla"), Value::U32(K_MLA as u32));
    kv!(format!("{a}.attention.value_length_mla"), Value::U32(V_MLA as u32));
    kv!(format!("{a}.attention.kv_lora_rank"), Value::U32(KV_LORA as u32));
    kv!(format!("{a}.attention.q_lora_rank"), Value::U32(Q_LORA as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(0));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(D_CONV as u32));
    kv!(format!("{a}.kda.head_dim"), Value::U32(HEAD_DIM_KDA as u32));

    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
    kv!(format!("{a}.expert_shared_count"), Value::U32(N_EXPERT_SHARED as u32));

    kv!(format!("{a}.attention.indexer.head_count"), Value::U32(INDEXER_HEAD as u32));
    kv!(format!("{a}.attention.indexer.key_length"), Value::U32(INDEXER_HEAD_SIZE as u32));
    kv!(format!("{a}.attention.indexer.top_k"), Value::U32(TOP_K as u32));
    kv!(format!("{a}.attention.indexer.kpool"), Value::U32(KPOOL as u32));
    kv!(
        format!("{a}.attention.indexer.types"),
        Value::Array(GgufType::Uint32, vec![Value::U32(1), Value::U32(1), Value::U32(0)])
    );

    kv!(format!("{a}.hyper_connection.count"), Value::U32(4));
    kv!(format!("{a}.hyper_connection.sinkhorn_iterations"), Value::U32(1));
    kv!(format!("{a}.hyper_connection.epsilon"), Value::F32(1e-4));

    let table = tensors_for(n_vocab);
    let mut data: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0x915_5e17);
    for (name, ne, is_norm) in &table {
        let n: i64 = ne.iter().product();
        let vals: Vec<f32> = if *is_norm {
            (0..n).map(|_| 1.0 + 0.05 * rng.next()).collect()
        } else {
            let s = 1.0 / (N_EMBD as f32).sqrt();
            (0..n).map(|_| s * rng.next()).collect()
        };
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
        println!("glm5 mtp synth: {n} tensors, {bytes} bytes -> {}", path());
    }
}

fn load_synth() -> LlamaModel {
    ensure_synth();
    let p = path();
    let gguf = Gguf::open(&p).expect("open synth");
    let f = std::fs::File::open(&p).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { Mmap::map(&f).unwrap() });
    load_model(&gguf, mmap).expect("glm5-next synth must load")
}

// ---------------------------------------------------------------------------
// the weights bundles (llama-cli's glm5_weights/glm5_params twins)
// ---------------------------------------------------------------------------

fn glm5_layer_weights(
    l: &llama::model::LayerTensors,
    il: usize,
) -> graph_arch::Glm5NextLayerWeights {
    let _ = il;
    graph_arch::Glm5NextLayerWeights {
        attn_norm: l.attn_norm.expect("attn_norm"),
        ffn_norm: l.ffn_norm.expect("ffn_norm"),
        hc_attn_fn: l.hc_attn_fn,
        hc_attn_base: l.hc_attn_base,
        hc_attn_scale: l.hc_attn_scale,
        hc_ffn_fn: l.hc_ffn_fn,
        hc_ffn_base: l.hc_ffn_base,
        hc_ffn_scale: l.hc_ffn_scale,
        ssm_q_conv: l.ssm_q_conv,
        ssm_k_conv: l.ssm_k_conv,
        ssm_v_conv: l.ssm_v_conv,
        wq: l.wq,
        wk: l.wk,
        wv: l.wv,
        wqkv: l.wqkv,
        ssm_f_a: l.ssm_f_a,
        ssm_f_b: l.ssm_f_b,
        ssm_beta: l.ssm_beta,
        ssm_a: l.ssm_a,
        ssm_dt_b: l.ssm_dt_b,
        ssm_g_a: l.ssm_g_a,
        ssm_g_b: l.ssm_g_b,
        ssm_o_norm: l.ssm_norm,
        wo: l.wo,
        attn_q_a_norm: l.attn_q_a_norm,
        attn_kv_a_norm: l.attn_kv_a_norm,
        wq_a: l.wq_a,
        wq_b: l.wq_b,
        wkv_a_mqa: l.wkv_a_mqa,
        wk_b: l.wk_b,
        wv_b: l.wv_b,
        indexer_k_norm: l.indexer_k_norm,
        indexer_k_norm_b: l.indexer_k_norm_b,
        indexer_proj: l.indexer_proj,
        indexer_attn_k: l.indexer_attn_k,
        indexer_attn_q_b: l.indexer_attn_q_b,
        indexer_kpool_gate: l.indexer_kpool_gate,
        indexer_kpool_ape: l.indexer_kpool_ape,
        ffn_gate: l.ffn_gate,
        ffn_down: l.ffn_down,
        ffn_up: l.ffn_up,
        ffn_gate_inp: l.ffn_gate_inp,
        ffn_exp_probs_b: l.ffn_exp_probs_b,
        ffn_gate_exps: l.ffn_gate_exps,
        ffn_down_exps: l.ffn_down_exps,
        ffn_up_exps: l.ffn_up_exps,
        ffn_gate_shexp: l.ffn_gate_shexp,
        ffn_down_shexp: l.ffn_down_shexp,
        ffn_up_shexp: l.ffn_up_shexp,
    }
}

fn glm5_params(hp: &llama::hparams::LlamaHparams, n_trunk: usize, attn: AttnParams) -> graph_arch::Glm5NextParams {
    graph_arch::Glm5NextParams {
        attn,
        n_embd: hp.n_embd as i64,
        n_head: hp.n_head(0) as i64,
        is_recr: (0..n_trunk).map(|il| hp.is_recr(il)).collect(),
        n_embd_r: hp.n_embd_r(),
        n_embd_s: hp.n_embd_s(),
        ssm_d_conv: hp.ssm_d_conv as i64,
        n_embd_head_kda: hp.n_embd_head_kda as i64,
        kda_gate_lower_bound: hp.kda_gate_lower_bound,
        n_lora_q: hp.n_lora_q as i64,
        n_lora_kv: hp.n_lora_kv as i64,
        n_embd_head_k_mla: hp.n_embd_head_k_mla() as i64,
        n_embd_head_v_mla: hp.n_embd_head_v_mla() as i64,
        n_rot: hp.n_rot(0) as i64,
        n_layer_dense_lead: hp.n_layer_dense_lead,
        n_expert: hp.n_expert as i64,
        n_expert_used: hp.n_expert_used(0) as i64,
        n_expert_shared: hp.n_expert_shared as i64,
        expert_weights_norm: hp.expert_weights_norm,
        expert_weights_scale: hp.expert_weights_scale,
        expert_gating_func: hp.expert_gating_func as i32,
        indexer_n_head: hp.indexer_n_head as i64,
        indexer_head_size: hp.indexer_head_size as i64,
        indexer_top_k: hp.indexer_top_k as i64,
        indexer_kpool: hp.indexer_kpool as i64,
        indexer_kpool_select_tail: hp.indexer_kpool_select_tail,
        is_indexer_full: (0..n_trunk).map(|il| hp.is_indexer_full(il)).collect(),
        hc: hp.dsv4_hc_mult as i64,
        hc_sinkhorn_iters: hp.dsv4_hc_sinkhorn_iters,
        hc_eps: hp.dsv4_hc_eps,
        f_norm_eps: hp.f_norm_eps,
    }
}

fn attn_of(m: &LlamaModel) -> AttnParams {
    let hp = &m.hparams;
    let rope = hp.rope_runtime();
    AttnParams {
        n_head: hp.n_head(0) as i64,
        n_head_kv: 1,
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
    }
}

/// the (trunk weights, MtpForward, attn) triple of a loaded synth — the
/// llama-cli assembly's twin
fn forward_of(m: &LlamaModel) -> (ForwardWeights, MtpForward, AttnParams) {
    let hp = &m.hparams;
    let n_trunk = hp.n_layer() as usize;
    let attn = attn_of(m);
    let params = glm5_params(hp, n_trunk, attn);
    let weights = ForwardWeights::Glm5Next(
        graph_arch::Glm5NextModelWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            layers: m.layers[..n_trunk]
                .iter()
                .enumerate()
                .map(|(il, l)| glm5_layer_weights(l, il))
                .collect(),
        },
        params.clone(),
    );
    let il = n_trunk;
    let l = &m.layers[il];
    let mtp = MtpForward::Glm5Next(
        graph_arch::Glm5NextMtpWeights {
            tok_embd: m.tok_embd,
            output_norm: m.output_norm,
            output: m.output,
            nextn: graph_arch::MtpNextn {
                eh_proj: l.nextn.eh_proj.expect("nextn.eh_proj"),
                enorm: l.nextn.enorm.expect("nextn.enorm"),
                hnorm: l.nextn.hnorm.expect("nextn.hnorm"),
                embed_tokens: l.nextn.embed_tokens,
                shared_head_head: l.nextn.shared_head_head,
                shared_head_norm: l.nextn.shared_head_norm,
            },
            layer: glm5_layer_weights(l, il),
        },
        params,
    );
    (weights, mtp, attn)
}

/// the MTP draft context of a loaded synth (`LLAMA_CONTEXT_TYPE_MTP`,
/// speculative.cpp:2545-2547) — new_mtp with the Glm5Next arm (the
/// hybrid-idx caches: the K-only MLA kv + the idx HybridIdxCache)
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

/// the draft chain both sides replay (ref_mtp2_dump's main loop): one token
/// + one F32 h row per step, the previous step's argmax + t_h_nextn row
/// feeding the next — 12 steps
fn run_chain(m: &mut LlamaModel) -> Vec<(i32, Vec<f32>, Vec<f32>)> {
    let n_embd = m.hparams.n_embd as usize;
    let n_vocab = m.ctx.ne(m.output)[1] as usize;
    let mut d = mtp_driver_of(m);
    // the ref probe's tap (llama_set_embeddings_nextn(ctx, true, false) —
    // unmasked, the standalone context reads row i directly)
    d.set_embeddings_nextn(true, false);

    let mut out = Vec::new();
    let mut tok = 1i32;
    let mut h = vec![0.0f32; n_embd];
    for step in 0..N_STEPS {
        let mut b = LlamaBatch::default();
        b.add(tok, step as i32, &[0], true);
        b.embd = Some(h.clone());
        let o = d.decode_batch(&b).expect("glm5 mtp draft step");
        assert_eq!(o.n_outputs, 1);
        let lg = o.logits[..n_vocab].to_vec();
        let hnext = d.get_embeddings_nextn_ith(0).to_vec();
        assert!(lg.iter().all(|v| v.is_finite()), "glm5 mtp: non-finite logits at step {step}");
        assert!(
            hnext.iter().all(|v| v.is_finite()),
            "glm5 mtp: non-finite h_nextn at step {step}"
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
fn glm5_mtp_synth_load_and_chain() {
    ensure_synth();
    let mut m = load_synth();
    assert_eq!(m.arch, LlmArch::GLM5_NEXT);
    assert_eq!(m.hparams.n_layer(), N_LAYER as u32);
    assert_eq!(m.hparams.n_layer_nextn, 1);
    let l = &m.layers[N_LAYER];
    assert!(l.nextn.eh_proj.is_some(), "the nextn trio");
    assert!(l.nextn.enorm.is_some());
    assert!(l.nextn.hnorm.is_some());
    // the NextN layer is a full-indexer DSA block (glm5-next.cpp:1059)
    assert!(l.indexer_attn_q_b.is_some(), "the full k-pool indexer");

    let chain = run_chain(&mut m);
    write_dump(&format!("{OUT_DIR}/glm5-next-port.bin"), &chain);
    let (tok0, lg, _) = &chain[0];
    println!(
        "glm5-next mtp: chain ok — {} steps, first tok {tok0}, argmax {}",
        chain.len(),
        argmax(lg)
    );
}

/// the reference bit-compare: parity/gen_glm5_mtp_ref.sh drives
/// parity/ref_mtp2_dump (the NEW reference's own MTP context, --nodes for
/// the bisect) over the same synth; this cell compares the two dumps
/// byte-for-byte (logits + h_nextn rows of the whole chain — an error
/// anywhere in the DSA+kpool graph compounds through the fed-back h rows)
#[test]
#[ignore = "needs /tmp/mtp2/glm5-next-ref.bin (parity/gen_glm5_mtp_ref.sh)"]
fn glm5_mtp_reference_bitcompare() {
    let ref_path = format!("{OUT_DIR}/glm5-next-ref.bin");
    let port_path = format!("{OUT_DIR}/glm5-next-port.bin");
    assert!(
        std::path::Path::new(&ref_path).exists(),
        "run parity/gen_glm5_mtp_ref.sh first"
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
            "glm5-next mtp dump differs (first at step {step}; {n_steps} steps, n_vocab \
             {n_vocab}, n_embd {n_embd})"
        );
    }
    println!(
        "glm5-next mtp: reference bit-compare PASS ({} bytes)",
        a.len()
    );
}

/// the #[ignore] generator for the parity runs (the port side of
/// gen_glm5_mtp_ref.sh)
#[test]
#[ignore = "writes /tmp/mtp2/glm5-next-synth-mtp.gguf + the port chain dump"]
fn glm5_mtp_write_synth_and_chain() {
    {
        let _g = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (n, bytes) = build_file(&path());
        println!("glm5-next mtp synth: {n} tensors, {bytes} bytes -> {}", path());
    }
    glm5_mtp_synth_load_and_chain();
}
