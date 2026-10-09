//! imatrix_md_parity.rs — synthetic qwen35 trunk + MTP-only draft fixtures
//! for the `-md` NextN collection lane (task ③, X domain).
//!
//! `gen` (ignored; driven by parity/imatrix_md_parity.sh) writes
//! /tmp/closx-md/{trunk,draft}-synth.gguf — the same recipe as the mtp2_e2e
//! synthetic files (crates/llama/tests/mtp2_e2e.rs: the llama-spm vocab
//! fixture, all-attention recurrent_layers, IMROPE sections, the REQUIRED
//! GDN ssm keys) split the way upstream's `--export-lora`-style MTP split
//! produces: the trunk file has blocks 0..n_layer and no nextn tensors; the
//! draft file has block n_layer (+ its nextn trio) and no trunk blocks
//! (`nextn_flags`'s MTP-only probe, model.rs:23071-23096). Both sides of the
//! parity run load the SAME pair, so any stdout divergence is a port bug.
//!
//! `probe` (ignored) loads both files through the port's `load_model` and
//! pins the shape fields the `-md` gate reads (the draft is a valid
//! MTP-only file, the trunk a valid trunk-only file).

use std::sync::Arc;

use ggml::gguf_write::GgufWriter;
use ggml::types::GgmlType;
use ggml::{Gguf, GgufType, Value};

const VOCAB_SPM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../llama/tests/tokenizer_fixtures/vocab/ggml-vocab-llama-spm.gguf"
);
const OUT_DIR: &str = "/tmp/closx-md";

const N_VOCAB: i64 = 32000;
const N_LAYER: usize = 2; // trunk; the draft's MTP layer is block N_LAYER
const N_EMBD: i64 = 128;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HD: i64 = 32;
const N_FF: i64 = 64;
const N_CTX: u32 = 64;

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

/// the qwen35 tensor names of one block. `recurrent` marks a GDN layer
/// (gated delta net: fused wqkv + ssm tensors, qwen35.cpp:80-89); the
/// geometry follows the mtp2 ssm keys: d_state=8, n_group=2, dt_rank=4 →
/// key_dim = 8*2 = 16, value_dim = 8*4 = 32, conv_dim = 16*2+32 = 64.
/// Layout: trunk = [GDN, attn] (a legal hybrid — the reference's
/// all-attention layout aborts inside `llm_graph_input_mem_hybrid::
/// set_input`, ggml-backend.cpp:345 GGML_ASSERT(buffer) on the zero-size
/// s_copy), draft = the MTP layer (always an attention block, qwen35.cpp:
/// 101-111 has no GDN arm) + the nextn trio. `both` = trunk + the MTP layer
/// in one file (the `--nextn` counterpart fixture: same nextn block, so
/// `--nextn` on it must produce exactly what `-md` produces on the split
/// pair — the self-consistency leg of the parity run).
fn tensors_for_block(flavor: Flavor) -> Vec<(String, Vec<i64>, f32)> {
    let mut v: Vec<(String, Vec<i64>, f32)> = Vec::new();
    let proj = 1.0 / (N_EMBD as f32).sqrt();
    macro_rules! push {
        ($name:expr, $ne:expr, $s:expr) => {
            v.push(($name.to_string(), $ne.to_vec(), $s))
        };
    }
    push!("token_embd.weight", vec![N_EMBD, N_VOCAB], proj);
    push!("output_norm.weight", vec![N_EMBD], 1.0);
    push!("output.weight", vec![N_EMBD, N_VOCAB], proj);

    let key_dim: i64 = 8 * 2;
    let value_dim: i64 = 8 * 4;
    let conv_dim: i64 = key_dim * 2 + value_dim;
    // (block index, is the GDN layer, is the MTP layer)
    let blocks: &[(usize, bool, bool)] = match flavor {
        Flavor::Draft => &[(N_LAYER, false, true)], // MTP-only (+ nextn trio)
        // trunk: [GDN, attn]; `both` appends the MTP block
        Flavor::Trunk => &[(0, true, false), (1, false, false)],
        Flavor::Both => &[(0, true, false), (1, false, false), (N_LAYER, false, true)],
    };
    for &(i, recurrent, mtp) in blocks {
        let i = i as i32;
        push!(format!("blk.{i}.attn_norm.weight"), vec![N_EMBD], 1.0);
        push!(format!("blk.{i}.post_attention_norm.weight"), vec![N_EMBD], 1.0);
        if recurrent {
            push!(format!("blk.{i}.attn_qkv.weight"), vec![N_EMBD, conv_dim], proj);
            push!(format!("blk.{i}.attn_gate.weight"), vec![N_EMBD, value_dim], proj);
            push!(format!("blk.{i}.ssm_conv1d.weight"), vec![4, conv_dim], proj);
            push!(format!("blk.{i}.ssm_dt.bias"), vec![4], 0.02);
            push!(format!("blk.{i}.ssm_a"), vec![4], proj);
            push!(format!("blk.{i}.ssm_beta.weight"), vec![N_EMBD, value_dim / 8], proj);
            push!(format!("blk.{i}.ssm_alpha.weight"), vec![N_EMBD, value_dim / 8], proj);
            push!(format!("blk.{i}.ssm_norm.weight"), vec![8], 1.0);
            push!(format!("blk.{i}.ssm_out.weight"), vec![value_dim, N_EMBD], proj);
        } else {
            push!(format!("blk.{i}.attn_q.weight"), vec![N_EMBD, 2 * HD * N_HEAD], proj);
            push!(format!("blk.{i}.attn_k.weight"), vec![N_EMBD, HD * N_HEAD_KV], proj);
            push!(format!("blk.{i}.attn_v.weight"), vec![N_EMBD, HD * N_HEAD_KV], proj);
            push!(format!("blk.{i}.attn_output.weight"), vec![HD * N_HEAD, N_EMBD], proj);
            push!(format!("blk.{i}.attn_q_norm.weight"), vec![HD], 1.0);
            push!(format!("blk.{i}.attn_k_norm.weight"), vec![HD], 1.0);
        }
        push!(format!("blk.{i}.ffn_gate.weight"), vec![N_EMBD, N_FF], proj);
        push!(format!("blk.{i}.ffn_down.weight"), vec![N_FF, N_EMBD], proj);
        push!(format!("blk.{i}.ffn_up.weight"), vec![N_EMBD, N_FF], proj);
        if mtp {
            // the NextN trio (mtp2_e2e.rs:414-420) — present in the draft and
            // the `both` file, never in the plain trunk
            push!(format!("blk.{i}.nextn.eh_proj.weight"), vec![2 * N_EMBD, N_EMBD], proj);
            push!(format!("blk.{i}.nextn.enorm.weight"), vec![N_EMBD], 1.0);
            push!(format!("blk.{i}.nextn.hnorm.weight"), vec![N_EMBD], 1.0);
        }
    }
    v
}

/// the file flavor: `trunk` (blocks 0..n_layer, no nextn), `draft`
/// (MTP-only, no trunk blocks) or `both` (the `--nextn` counterpart).
#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Trunk,
    Draft,
    Both,
}

impl Flavor {
    fn n_layer_all(self) -> usize {
        match self {
            Flavor::Trunk => N_LAYER,
            Flavor::Draft | Flavor::Both => N_LAYER + 1,
        }
    }
    fn tensors(self) -> Vec<(String, Vec<i64>, f32)> {
        tensors_for_block(self)
    }
    fn is_draft(self) -> bool {
        self == Flavor::Draft
    }
}

fn write_file(path: &str, flavor: Flavor) {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let draft = flavor.is_draft();

    let src = Gguf::open(VOCAB_SPM).expect("open vocab fixture");
    let mut w = GgufWriter::new(32);
    for (k, val) in &src.kv {
        if k.starts_with("tokenizer.") && k != "tokenizer.chat_template" {
            w.set_kv(k, val.clone());
        }
    }

    let a = "qwen35";
    macro_rules! kv {
        ($k:expr, $v:expr) => {
            w.set_kv(&$k, $v)
        };
    }
    kv!("general.architecture", Value::String(a.to_string()));
    kv!(
        "general.name",
        Value::String(format!(
            "llama-rust-synth-{a}-{}",
            match flavor {
                Flavor::Trunk => "trunk",
                Flavor::Draft => "md",
                Flavor::Both => "both",
            }
        ))
    );
    kv!("general.file_type", Value::U32(0)); // F32
    kv!(format!("{a}.context_length"), Value::U32(N_CTX));
    kv!(format!("{a}.embedding_length"), Value::U32(N_EMBD as u32));
    // the trunk's block_count is the trunk layer count; the draft/both files
    // carry n_trunk + n_nextn (the MTP layer lives at block n_trunk —
    // model_file_shape::n_trunk, imatrix.cpp:1318-1327)
    kv!(
        format!("{a}.block_count"),
        Value::U32(flavor.n_layer_all() as u32)
    );
    if flavor != Flavor::Trunk {
        kv!(format!("{a}.nextn_predict_layers"), Value::U32(1));
    }
    kv!(format!("{a}.feed_forward_length"), Value::U32(N_FF as u32));
    kv!(format!("{a}.attention.head_count"), Value::U32(N_HEAD as u32));
    kv!(format!("{a}.attention.head_count_kv"), Value::U32(N_HEAD_KV as u32));
    kv!(format!("{a}.attention.key_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.value_length"), Value::U32(HD as u32));
    kv!(format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
    kv!(format!("{a}.rope.dimension_count"), Value::U32(16));
    kv!(format!("{a}.rope.freq_base"), Value::F32(10_000.0));
    // IMRoPE sections (sum n_rot/2) + the GDN keys (REQUIRED even though
    // this file has no recurrent layer — mtp2_e2e.rs:526-541)
    kv!(
        format!("{a}.rope.dimension_sections"),
        Value::Array(GgufType::Uint32, vec![
            Value::U32(5), Value::U32(2), Value::U32(1), Value::U32(0)
        ])
    );
    kv!(format!("{a}.ssm.conv_kernel"), Value::U32(4));
    kv!(format!("{a}.ssm.inner_size"), Value::U32(32));
    kv!(format!("{a}.ssm.state_size"), Value::U32(8));
    kv!(format!("{a}.ssm.time_step_rank"), Value::U32(4));
    kv!(format!("{a}.ssm.group_count"), Value::U32(2));
    // the hybrid layout [GDN, attn(, attn for the MTP layer)] —
    // qwen35.cpp:17-21's default interleave needs a GDN layer present or
    // the reference's hybrid memory aborts (see tensors_for_block)
    kv!(
        format!("{a}.attention.recurrent_layers"),
        Value::Array(
            GgufType::Uint32,
            (0..flavor.n_layer_all())
                .map(|i| Value::U32(if i == 0 { 1 } else { 0 }))
                .collect()
        )
    );

    // deterministic contents — both sides of the parity run load the same
    // bytes; the seed derives from the tensor NAME so a block shared by two
    // flavors (the MTP layer of `draft` and `both`) carries identical bytes
    // and the `--nextn` run can reproduce the `-md` run exactly
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for (name, ne, s) in flavor.tensors() {
        let ne1 = ne.get(1).copied().unwrap_or(1);
        w.add_tensor(&name, GgmlType::F32, [ne[0], ne1, 1, 1]);
        let mut seed = 0xcbf2_9ce4_8422_2325u64; // FNV-1a of the name
        for b in name.bytes() {
            seed = (seed ^ b as u64) * 0x1000_0000_01b3;
        }
        let mut rng = Rng(seed);
        let n: usize = ne.iter().product::<i64>() as usize;
        payloads.push(
            (0..n)
                .map(|_| (rng.next() * s).to_le_bytes())
                .flatten()
                .collect(),
        );
    }
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let mut buf = Vec::new();
    w.write(&mut buf, &refs).unwrap();
    std::fs::write(path, &buf).unwrap();
    eprintln!("wrote {path} ({} tensors)", refs.len());
}

/// the parity prompt: enough tokens for 2 chunks of n_ctx=64 under the spm
/// tokenizer (~1 token per word)
fn write_prompt(path: &str) {
    let sentence = "the quick brown fox jumps over the lazy dog while the curious cat watches from the warm windowsill and the rain falls gently on the quiet garden ";
    let text = sentence.repeat(12);
    std::fs::write(path, text).unwrap();
    eprintln!("wrote {path}");
}

#[test]
#[ignore]
fn gen() {
    write_file(&format!("{OUT_DIR}/trunk-synth.gguf"), Flavor::Trunk);
    write_file(&format!("{OUT_DIR}/draft-synth.gguf"), Flavor::Draft);
    write_file(&format!("{OUT_DIR}/both-synth.gguf"), Flavor::Both);
    write_prompt(&format!("{OUT_DIR}/prompt.txt"));
}

/// pin the shape fields the `-md` gate reads through the port's own loader
#[test]
#[ignore]
fn probe() {
    for (path, flavor) in [
        (format!("{OUT_DIR}/trunk-synth.gguf"), Flavor::Trunk),
        (format!("{OUT_DIR}/draft-synth.gguf"), Flavor::Draft),
        (format!("{OUT_DIR}/both-synth.gguf"), Flavor::Both),
    ] {
        let gguf = Gguf::open(&path).expect("open");
        let f = std::fs::File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });
        let m = llama::model::load_model(&gguf, mmap).expect("load");
        assert_eq!(m.arch, llama::arch::LlmArch::QWEN35);
        if flavor == Flavor::Draft {
            assert_eq!(m.hparams.n_layer_nextn, 1);
            assert_eq!(m.hparams.n_layer(), N_LAYER as u32);
            // MTP-only: the trunk tensors degrade to NOT_REQUIRED (nextn_flags)
            assert!(m.layers[0].attn_norm.is_none());
            let mtp = &m.layers[N_LAYER].nextn;
            assert!(mtp.eh_proj.is_some(), "eh_proj");
            assert!(mtp.enorm.is_some());
            assert!(mtp.hnorm.is_some());
            assert!(mtp.shared_head_head.is_none(), "own_lm_head = false");
            // the draft's global tensors are plain TensorIds (non-optional
            // slots) — their presence is implied by load_model succeeding
            let _ = (m.output, m.tok_embd);
        } else if flavor == Flavor::Both {
            assert_eq!(m.hparams.n_layer_nextn, 1);
            assert_eq!(m.hparams.n_layer(), N_LAYER as u32);
            assert!(m.layers[0].attn_norm.is_some());
            assert!(m.layers[N_LAYER].nextn.eh_proj.is_some());
        } else {
            assert_eq!(m.hparams.n_layer_nextn, 0);
            assert_eq!(m.hparams.n_layer(), N_LAYER as u32);
            assert!(m.layers[0].attn_norm.is_some());
            // trunk-only: no nextn block at all (layers.len() == n_layer)
            assert!(m.layers.get(N_LAYER).is_none());
        }
    }
}
