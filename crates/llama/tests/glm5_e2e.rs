//! glm5_e2e.rs — the NEW `glm5-next` arch (def4d406a, src/models/glm5-next.cpp)
//! loader-side acceptance: a synthetic GGUF written with the port's writer
//! must load through the port's `load_model` AND through the NEW reference
//! (`parity/glm5_parity.sh` drives the reference's default-params load and
//! byte-compares the print_info banner — loader 1:1: KV set, tensor
//! names/shapes/types, hparams derivations).
//!
//! The graph side (kda/dsa/kpool builders) is the documented batch-A gap:
//! it waits on the kv-cache lane's `llama_memory_hybrid_idx` kpool port
//! (see PARITY.md 同步批次 A).

use std::io::Write as _;
use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, GgufType, Value};
use llama::arch::LlmArch;
use llama::model::load_model;

const VOCAB_SPM: &str = "/home/jeffrey/llm/llama.cpp-pinned/models/ggml-vocab-llama-spm.gguf";
const OUT_DIR: &str = "/tmp/syncm-glm5";

// the synthetic geometry (loader-level; the graph never runs)
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

fn synth_path() -> String {
    format!("{OUT_DIR}/glm5-next-synth-{}.gguf", std::process::id())
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

/// the tensor table — exactly what `llama_model_glm5_next::load_arch_tensors`
/// (glm5-next.cpp:61-186) requests: layer 0 KDA, 1 full-indexer DSA,
/// 2 shared-indexer DSA, 3 the NextN DSA block.
fn tensors_for(n_vocab: i64) -> Vec<(String, Vec<i64>, bool)> {
    let mut v: Vec<(String, Vec<i64>, bool)> = Vec::new(); // (name, ne, is_norm)
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

        // the mHC mixers live on the trunk layers only (glm5-next.cpp:87-94)
        if i < N_LAYER as i64 {
            push!(format!("blk.{i}.hc_attn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
            push!(format!("blk.{i}.hc_attn_base.weight"), vec![hc_mix], false);
            push!(format!("blk.{i}.hc_attn_scale.weight"), vec![3], false);
            push!(format!("blk.{i}.hc_ffn_fn.weight"), vec![hc * N_EMBD, hc_mix], false);
            push!(format!("blk.{i}.hc_ffn_base.weight"), vec![hc_mix], false);
            push!(format!("blk.{i}.hc_ffn_scale.weight"), vec![3], false);
        }

        if i == 0 {
            // the KDA layer (glm5-next.cpp:100-121), 4D conv layout
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
            // the nope-MLA DSA layer (glm5-next.cpp:122-139)
            push!(format!("blk.{i}.attn_q_a_norm.weight"), vec![Q_LORA], true);
            push!(format!("blk.{i}.attn_kv_a_norm.weight"), vec![KV_LORA], true);
            push!(format!("blk.{i}.attn_q_a.weight"), vec![N_EMBD, Q_LORA], false);
            push!(format!("blk.{i}.attn_q_b.weight"), vec![Q_LORA, N_HEAD * K_MLA], false);
            push!(format!("blk.{i}.attn_kv_a_mqa.weight"), vec![N_EMBD, KV_LORA], false);
            push!(format!("blk.{i}.attn_k_b.weight"), vec![qk_nope, KV_LORA, N_HEAD], false);
            push!(format!("blk.{i}.attn_v_b.weight"), vec![KV_LORA, V_MLA, N_HEAD], false);
            push!(format!("blk.{i}.attn_output.weight"), vec![N_HEAD * V_MLA, N_EMBD], false);

            // the k-pool indexer: full on layers 1 and 3 (the NextN block
            // always full), absent on the shared layer 2
            // (glm5-next.cpp:141-154)
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

        // all-MoE (n_layer_dense_lead = 0) — glm5-next.cpp:161-174
        push!(format!("blk.{i}.ffn_gate_inp.weight"), vec![N_EMBD, N_EXPERT], false);
        push!(format!("blk.{i}.exp_probs_b.bias"), vec![N_EXPERT], true);
        push!(format!("blk.{i}.ffn_gate_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_down_exps.weight"), vec![N_FF_EXP, N_EMBD, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_up_exps.weight"), vec![N_EMBD, N_FF_EXP, N_EXPERT], false);
        push!(format!("blk.{i}.ffn_gate_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);
        push!(format!("blk.{i}.ffn_down_shexp.weight"), vec![N_FF_EXP * N_EXPERT_SHARED, N_EMBD], false);
        push!(format!("blk.{i}.ffn_up_shexp.weight"), vec![N_EMBD, N_FF_EXP * N_EXPERT_SHARED], false);

        if i >= N_LAYER as i64 {
            // the NextN block (glm5-next.cpp:177-183)
            push!(format!("blk.{i}.nextn.eh_proj.weight"), vec![2 * N_EMBD, N_EMBD], false);
            push!(format!("blk.{i}.nextn.enorm.weight"), vec![N_EMBD], true);
            push!(format!("blk.{i}.nextn.hnorm.weight"), vec![N_EMBD], true);
        }
    }
    v
}

fn build_file(path: &str) -> (usize, u64) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir /tmp/syncm-glm5");

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
    // layer 0 is KDA (head_count_kv == 0 → is_recr), the rest are DSA
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
    // nope-only MLA: n_rot == 0 (glm5-next.cpp:935 asserts it)
    kv!(format!("{a}.rope.dimension_count"), Value::U32(0));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));

    // the KDA keys (glm5-next.cpp:14-16)
    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(D_CONV as u32));
    kv!(format!("{a}.kda.head_dim"), Value::U32(HEAD_DIM_KDA as u32));

    // the MoE keys (glm5-next.cpp:25-37)
    kv!(format!("{a}.expert_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_used_count"), Value::U32(N_EXPERT as u32));
    kv!(format!("{a}.expert_feed_forward_length"), Value::U32(N_FF_EXP as u32));
    kv!(format!("{a}.expert_shared_count"), Value::U32(N_EXPERT_SHARED as u32));

    // the k-pool indexer (glm5-next.cpp:40-47): layer 2 shares layer 1's
    kv!(format!("{a}.attention.indexer.head_count"), Value::U32(INDEXER_HEAD as u32));
    kv!(format!("{a}.attention.indexer.key_length"), Value::U32(INDEXER_HEAD_SIZE as u32));
    kv!(format!("{a}.attention.indexer.top_k"), Value::U32(TOP_K as u32));
    kv!(format!("{a}.attention.indexer.kpool"), Value::U32(KPOOL as u32));
    kv!(
        format!("{a}.attention.indexer.types"),
        Value::Array(GgufType::Uint32, vec![Value::U32(1), Value::U32(1), Value::U32(0)])
    );

    // mHC (glm5-next.cpp:50-53) — hc_mult must be 4
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

fn load_synth() -> (llama::model::LlamaModel, String) {
    let path = synth_path();
    let (n_tensors, _bytes) = build_file(&path);
    assert!(n_tensors > 0);
    let gguf = Gguf::open(&path).expect("open synth");
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: read-only mmap of a file this process just wrote
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
    let m = load_model(&gguf, mmap).expect("glm5-next synth must load");
    (m, path)
}

/// the fixed-name copy `parity/glm5_parity.sh` drives the NEW reference with
/// (the reference load must succeed and print the same print_info payload).
#[test]
#[ignore = "writer only — the file lands in /tmp for the parity script"]
fn glm5_write_synth_file() {
    let path = format!("{OUT_DIR}/glm5-next-synth.gguf");
    let (n, bytes) = build_file(&path);
    println!("glm5-next: {n} tensors, {bytes} bytes -> {path}");
}

/// glm5-next loads 1:1: the arch tables resolve, every hparams derivation of
/// load_arch_hparams (glm5-next.cpp:7-59) lands, and the loader consumes
/// exactly the file's tensors (done_getting_tensors). The reference side of
/// the acceptance is parity/glm5_parity.sh (the print_info banner, NEW ref).
#[test]
fn glm5_synth_loader_parity() {
    let (m, path) = load_synth();
    let hp = &m.hparams;

    assert_eq!(m.arch, LlmArch::GLM5_NEXT);
    assert_eq!(m.arch.name(), "glm5-next");
    assert_eq!(hp.n_layer(), N_LAYER as u32);
    assert_eq!(hp.n_layer_all, (N_LAYER + 1) as u32);
    assert_eq!(hp.n_layer_nextn, 1);

    // glm5-next.cpp:8-16
    assert_eq!(hp.f_norm_rms_eps, 1e-5);
    assert_eq!(hp.n_embd_head_k_mla() as i64, K_MLA);
    assert_eq!(hp.n_embd_head_v_mla() as i64, V_MLA);
    assert_eq!(hp.n_lora_q as i64, Q_LORA);
    assert_eq!(hp.n_lora_kv as i64, KV_LORA);
    assert_eq!(hp.ssm_d_conv as i64, D_CONV);
    assert_eq!(hp.n_embd_head_kda as i64, HEAD_DIM_KDA);
    // the MLA cache holds the compressed latent (glm5-next.cpp:19)
    assert_eq!(hp.n_embd_head_v_full, KV_LORA as u32);

    // glm5-next.cpp:21-23 — the loop covers n_layer_all
    assert!(hp.is_recr(0));
    assert!(!hp.is_recr(1));
    assert!(!hp.is_recr(2));
    assert!(!hp.is_recr(3));

    // glm5-next.cpp:40-47 — the k-pool indexer
    assert_eq!(hp.indexer_n_head as i64, INDEXER_HEAD);
    assert_eq!(hp.indexer_head_size as i64, INDEXER_HEAD_SIZE);
    assert_eq!(hp.indexer_top_k as i64, TOP_K);
    assert_eq!(hp.indexer_kpool as i64, KPOOL);
    assert!(hp.indexer_kpool_select_tail); // the loader default
    assert!(hp.is_indexer_full(0));
    assert!(hp.is_indexer_full(1));
    assert!(!hp.is_indexer_full(2));
    // past n_layer the C's accessor GGML_ABORTs — glm5-next.cpp:958 guards
    // those layers with `il >= n_layer() ||` first, so the fill-1 default is
    // only read through the impl array here:
    assert_eq!(hp.is_indexer_full_impl[3], 1);

    // glm5-next.cpp:50-53 — mHC
    assert_eq!(hp.dsv4_hc_mult, 4);
    assert_eq!(hp.dsv4_hc_sinkhorn_iters, 1);
    assert_eq!(hp.dsv4_hc_eps, 1e-4);

    // the sigmoid gating default (glm5-next.cpp:31-33)
    assert_eq!(
        hp.expert_gating_func,
        llama::hparams::LlamaExpertGatingFuncType::SIGMOID as u32
    );

    // llama_model_rope_type: GLM5_NEXT → NONE (llama-model.cpp:3048)
    assert_eq!(hp.rope_type, llama::hparams::LlamaRopeType::NONE);

    // llm_type: 45 → 320B.A18B on real files (glm5-next.cpp:55-58); the
    // synthetic's 3 layers are LLM_TYPE_UNKNOWN. n_vocab is unused in the
    // glm5-next arm of the switch.
    assert_eq!(
        llama::display::llm_type_of(m.arch, hp, 32000),
        llama::display::LlmType::UNKNOWN
    );
    {
        let mut h45 = hp.clone();
        h45.n_layer_all = 45;
        h45.n_layer_nextn = 0; // n_layer() == 45
        assert_eq!(
            llama::display::llm_type_of(LlmArch::GLM5_NEXT, &h45, 32000),
            llama::display::LlmType::B320B_A18
        );
    }

    // the graph side is the documented batch-A gap — the builder is not
    // wired into a DecodeContext yet
    drop(m);
    std::fs::remove_file(&path).ok();
}

